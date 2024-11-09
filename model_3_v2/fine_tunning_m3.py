import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from sklearn.metrics import precision_recall_fscore_support
import time
import os
from utils_m3 import WarmupCosineScheduler

def fine_tune(
    net,
    train_loader,
    val_loader,
    config,
    writer,
    save_dir,
    unfreeze_strategy='all',
    start_epoch=0
):
    device = next(net.parameters()).device
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()
    
    # Recreate data loaders with optimized batch size
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.ft_batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.ft_batch_size * 2,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    # Calculate gradient accumulation steps
    effective_batch_size = 128  # Target batch size
    accumulation_steps = max(1, effective_batch_size // config.ft_batch_size)
    
    # Prepare the model for fine-tuning
    if unfreeze_strategy == 'all':
        for param in net.parameters():
            param.requires_grad = True
    elif unfreeze_strategy == 'gradual':
        net.freeze_encoder()
        net.classifier.requires_grad_(True)
    elif unfreeze_strategy == 'last_n':
        net.freeze_encoder()
        for param in net.f.conv3.parameters():
            param.requires_grad = True
        for param in net.f.bn3.parameters():
            param.requires_grad = True
        net.classifier.requires_grad_(True)
    
    # Optimizer setup
    encoder_params = [p for name, p in net.named_parameters() if 'classifier' not in name and p.requires_grad]
    classifier_params = [p for name, p in net.named_parameters() if 'classifier' in name and p.requires_grad]
    
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': config.ft_encoder_lr},
        {'params': classifier_params, 'lr': config.ft_classifier_lr}
    ], weight_decay=config.weight_decay)
    
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=config.ft_warmup_epochs,
        total_epochs=config.ft_epochs
    )
    
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(start_epoch, config.ft_epochs):
        # Training phase
        net.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad(set_to_none=True)
        
        train_bar = tqdm(train_loader, desc=f'Fine-tuning Epoch {epoch + 1}/{config.ft_epochs}')
        
        for batch_idx, (inputs, _, targets) in enumerate(train_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            with autocast():
                outputs = net.classify(inputs)
                loss = criterion(outputs, targets) / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            train_loss += loss.item() * accumulation_steps
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            train_bar.set_postfix({
                'Loss': f'{loss.item() * accumulation_steps:.4f}',
                'Acc': f'{100. * train_correct / train_total:.2f}%'
            })
            
            if batch_idx % config.log_interval == 0:
                writer.add_scalar('Fine-tune/train_loss_step', 
                                loss.item() * accumulation_steps,
                                epoch * len(train_loader) + batch_idx)
                
            # Memory management
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        # Calculate training metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        net.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for batch_idx, (inputs, _, targets) in enumerate(tqdm(val_loader, desc='Validation')):
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = net.classify(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                
                # Memory management
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Calculate additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted'
        )
        
        # Log metrics
        writer.add_scalar('Fine-tune/train_loss_epoch', train_loss, epoch)
        writer.add_scalar('Fine-tune/train_acc', train_acc, epoch)
        writer.add_scalar('Fine-tune/val_loss', val_loss, epoch)
        writer.add_scalar('Fine-tune/val_acc', val_acc, epoch)
        writer.add_scalar('Fine-tune/precision', precision, epoch)
        writer.add_scalar('Fine-tune/recall', recall, epoch)
        writer.add_scalar('Fine-tune/f1', f1, epoch)
        
        # Learning rate scheduling
        scheduler.step()
        current_lr_encoder = optimizer.param_groups[0]['lr']
        current_lr_classifier = optimizer.param_groups[1]['lr']
        writer.add_scalar('Fine-tune/learning_rate_encoder', current_lr_encoder, epoch)
        writer.add_scalar('Fine-tune/learning_rate_classifier', current_lr_classifier, epoch)
        
        # Gradual unfreezing if using gradual strategy
        if unfreeze_strategy == 'gradual' and epoch == config.ft_unfreeze_epoch:
            logging.info("Unfreezing encoder layers...")
            for param in net.parameters():
                param.requires_grad = True
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            
            # Save checkpoint
            checkpoint_path = os.path.join(save_dir, f'best_fine_tuned_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'train_loss': train_loss,
                'best_epoch': best_epoch
            }, checkpoint_path)
            
            logging.info(f'Saved best model with validation accuracy: {val_acc:.2f}%')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.early_stopping_patience:
            logging.info(f'Early stopping triggered after {epoch + 1} epochs')
            break
        
        # Log epoch summary
        logging.info(f'Epoch: {epoch + 1}/{config.ft_epochs}')
        logging.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logging.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        logging.info(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
        
    return best_val_acc, best_epoch

def load_fine_tuned_model(net, path):
    """Load a fine-tuned model checkpoint."""
    if os.path.exists(path):
        checkpoint = torch.load(path)
        net.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded fine-tuned model from {path}")
        return net, checkpoint['val_acc']
    else:
        logging.warning(f"No fine-tuned model found at {path}")
        return net, None