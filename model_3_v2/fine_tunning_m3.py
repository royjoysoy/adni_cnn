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
    """
    Fine-tune a pre-trained model.
    
    Args:
        net: Pre-trained model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        writer: TensorBoard writer
        save_dir: Directory to save checkpoints
        unfreeze_strategy: Strategy for unfreezing layers ('all', 'gradual', or 'last_n')
        start_epoch: Starting epoch number
    """
    device = next(net.parameters()).device
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()
    
    # Prepare the model for fine-tuning
    if unfreeze_strategy == 'all':
        # Unfreeze all layers
        for param in net.parameters():
            param.requires_grad = True
    elif unfreeze_strategy == 'gradual':
        # Start with only classifier unfrozen
        net.freeze_encoder()
        net.classifier.requires_grad_(True)
    elif unfreeze_strategy == 'last_n':
        # Unfreeze last n layers (example: last conv block and classifier)
        net.freeze_encoder()
        for param in net.f.conv3.parameters():
            param.requires_grad = True
        for param in net.f.bn3.parameters():
            param.requires_grad = True
        net.classifier.requires_grad_(True)
    
    # Use different learning rates for pre-trained layers and new layers
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
        
        train_bar = tqdm(train_loader, desc=f'Fine-tuning Epoch {epoch + 1}/{config.ft_epochs}')
        
        for batch_idx, (inputs, _, targets) in enumerate(train_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = net.classify(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * train_correct / train_total:.2f}%'
            })
            
            if batch_idx % config.log_interval == 0:
                writer.add_scalar('Fine-tune/train_loss_step', 
                                loss.item(),
                                epoch * len(train_loader) + batch_idx)
        
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
            for inputs, _, targets in tqdm(val_loader, desc='Validation'):
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = net.classify(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
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
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Fine-tune/learning_rate', current_lr, epoch)
        
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
                'train_loss': train_loss
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