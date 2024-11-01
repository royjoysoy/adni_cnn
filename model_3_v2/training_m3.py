import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging
import time
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import CosineAnnealingLR 
from utils_m3 import calculate_f_score 
from utils_m3 import WarmupCosineScheduler, TrainingConfig 
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.data import DataLoader

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data_3d(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    
    # Make sure x and y are on the same device
    device = x.device
    y = y.to(device)
    
    index = torch.randperm(batch_size).to(device)

    y_a, y_b = y, y[index]

    bbx1, bby1, bbz1, bbx2, bby2, bbz2 = rand_bbox_3d(x.size(), lam)
    x[:, :, bbz1:bbz2, bbx1:bbx2, bby1:bby2] = x[index, :, bbz1:bbz2, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) * (bbz2 - bbz1) / (x.size()[-1] * x.size()[-2] * x.size()[-3]))

    return x, y_a, y_b, lam

def rand_bbox_3d(size, lam):
    W, H, D = size[2], size[3], size[4]
    cut_rat = np.cbrt(1. - lam)
    cut_w, cut_h, cut_d = (int(W * cut_rat), int(H * cut_rat), int(D * cut_rat))

    cx, cy, cz = (np.random.randint(W), np.random.randint(H), np.random.randint(D))

    bbx1, bby1, bbz1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H), np.clip(cz - cut_d // 2, 0, D)
    bbx2, bby2, bbz2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H), np.clip(cz + cut_d // 2, 0, D)

    return bbx1, bby1, bbz1, bbx2, bby2, bbz2

def train_epoch(net, data_loader, optimizer, temperature, simclr_batch_size, aug_type, simclr_epoch, total_simclr_epochs, writer, scaler):
    net.train()
    total_loss, total_num = 0.0, 0
    train_bar = tqdm(data_loader, desc=f'Train Epoch: [{simclr_epoch}/{total_simclr_epochs}] - {aug_type}')
    
    for step, data in enumerate(train_bar):
        if aug_type in ['mixup', 'cutmix']:
            pos_1, pos_2, target, idx = data
            pos_1, pos_2, target = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True), target.cuda(non_blocking=True)
            if aug_type == 'mixup':
                # Generate mixup data for pos_1 and pos_2 separately
                pos_1, targets_a, targets_b, lam = mixup_data(pos_1, target)
                pos_2, _, _, _ = mixup_data(pos_2, target)  # Generate new mixup for pos_2
            else:  # cutmix
                pos_1, targets_a, targets_b, lam = cutmix_data_3d(pos_1, target)
                pos_2, _, _, _ = cutmix_data_3d(pos_2, target)  # Generate new cutmix for pos_2
        else:
            pos_1, pos_2, target = data
            pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            out_1 = net(pos_1)
            out_2 = net(pos_2)
            out = torch.cat([out_1, out_2], dim=0)
            sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
            mask = (torch.ones_like(sim_matrix) - torch.eye(2 * simclr_batch_size, device=sim_matrix.device)).bool()
            sim_matrix = sim_matrix.masked_select(mask).view(2 * simclr_batch_size, -1)
            pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
            loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_num += simclr_batch_size
        total_loss += loss.item() * simclr_batch_size
        train_bar.set_description(f'Train Epoch: [{simclr_epoch}/{total_simclr_epochs}] - {aug_type} Loss: {total_loss / total_num:.4f}')

        if step % 10 == 0:  # Log every 10 steps
            writer.add_scalar(f'Loss/train_{aug_type}_step', loss.item(), simclr_epoch * len(data_loader) + step)
            logging.info(f'Epoch: [{simclr_epoch}/{total_simclr_epochs}], Step: [{step}/{len(data_loader)}], Aug: {aug_type}, Loss: {loss.item():.4f}')

    avg_loss = total_loss / total_num
    writer.add_scalar(f'Loss/train_{aug_type}_epoch', avg_loss, simclr_epoch)
    logging.info(f'Epoch: [{simclr_epoch}/{total_simclr_epochs}], Aug: {aug_type}, Avg Train Loss: {avg_loss:.4f}')
    return avg_loss

def test(net, memory_data_loader, test_data_loader, simclr_epoch, total_simclr_epochs, k, temperature, writer):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, 3, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, 3) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                             .format(simclr_epoch, total_simclr_epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    test_accuracy = total_top1 / total_num * 100

    writer.add_scalar('test/accuracy', test_accuracy, simclr_epoch)
    logging.info(f'Test Epoch: [{simclr_epoch}/{total_simclr_epochs}] Acc@1:{total_top1 / total_num * 100:.2f}% Acc@5:{total_top5 / total_num * 100:.2f}%')
    return test_accuracy

# model 2 update 4
def run_ablation_study(net, train_loaders, val_loader, optimizer, simclr_epochs, temperature, simclr_batch_size, k, save_top_n, writer, scaler, warmup_epochs):
    results = {aug_type: {'train_loss': [], 'val_accuracy': [], 'aug_time': 0} for aug_type in train_loaders.keys()}
    best_models = []
    best_acc = 0
    total_simclr_time = 0
    no_improve = 0
    patience = 10  # Early stopping patience
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=warmup_epochs, total_epochs=simclr_epochs)
    
    for simclr_epoch in range(1, simclr_epochs + 1):
        logging.info(f"Starting SimCLR epoch {simclr_epoch}/{simclr_epochs}")
        epoch_val_accuracy = 0
        for aug_type, train_loader in train_loaders.items():
            logging.info(f"Training with augmentation: {aug_type}")
            aug_start_time = time.time()
            train_loss = train_epoch(net, train_loader, optimizer, temperature, simclr_batch_size, aug_type, simclr_epoch, simclr_epochs, writer, scaler)
            aug_end_time = time.time()
            results[aug_type]['aug_time'] += aug_end_time - aug_start_time
            val_accuracy = test(net, train_loaders['base'], val_loader, simclr_epoch, simclr_epochs, k, temperature, writer)
            epoch_val_accuracy += val_accuracy
            results[aug_type]['train_loss'].append(train_loss)
            results[aug_type]['val_accuracy'].append(val_accuracy)
            logging.info(f'SimCLR Epoch {simclr_epoch}/{simclr_epochs}, Aug: {aug_type}, Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
            writer.add_scalar(f'Loss/train_{aug_type}', train_loss, simclr_epoch)
            writer.add_scalar(f'Accuracy/val_{aug_type}', val_accuracy, simclr_epoch)
        
        avg_val_accuracy = epoch_val_accuracy / len(train_loaders)
        
        if avg_val_accuracy > best_acc:
            best_acc = avg_val_accuracy
            model_info = {
                'simclr_epoch': simclr_epoch,
                'accuracy': avg_val_accuracy,
                'state_dict': net.state_dict()
            }
            best_models.append(model_info)
            best_models.sort(key=lambda x: x['accuracy'], reverse=True)
            best_models = best_models[:save_top_n]
            logging.info(f"New best accuracy: {best_acc:.2f}%")
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= patience:
            logging.info(f"Early stopping triggered at epoch {simclr_epoch}")
            break
        
        scheduler.step()
        
        # Log the learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate/train', current_lr, simclr_epoch)
        logging.info(f"Current learning rate: {current_lr}")
        
        total_simclr_time += sum(results[aug_type]['aug_time'] for aug_type in results)
        logging.info(f"Finished SimCLR epoch {simclr_epoch}/{simclr_epochs}")
        logging.info(f"Average validation accuracy: {avg_val_accuracy:.2f}%")
    
    logging.info(f"Ablation study completed. Total best models saved: {len(best_models)}")
    return results, best_models, total_simclr_time

def linear_eval(net, train_loader, val_loader, optimizer, linear_eval_epochs, linear_eval_batch_size, writer):
    start_time = time.time()
    net.freeze_encoder()
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    best_epoch = 0
    patience = 10
    no_improve = 0
    device = next(net.parameters()).device

    # Create a scheduler for linear evaluation
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Create new data loaders with the specified batch size
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    train_loader = DataLoader(train_dataset, batch_size=linear_eval_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=linear_eval_batch_size, shuffle=False)

    for linear_eval_epoch in range(linear_eval_epochs):
        net.classifier.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Linear Eval Train Epoch {linear_eval_epoch+1}/{linear_eval_epochs}')
        for pos_1, _, target in train_bar:
            pos_1, target = pos_1.to(device), target.to(device)
            with torch.no_grad():
                features = net.get_features(pos_1)
            outputs = net.classifier(features)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_bar.set_postfix(loss=f'{loss.item():.4f}')
        train_loss /= len(train_loader)
        writer.add_scalar('Loss/train', train_loss, linear_eval_epoch)

        # Validation
        net.classifier.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_bar = tqdm(val_loader, desc=f'Linear Eval Val Epoch {linear_eval_epoch+1}/{linear_eval_epochs}')
        with torch.no_grad():
            for pos_1, _, target in val_bar:
                pos_1, target = pos_1.to(device), target.to(device)
                features = net.get_features(pos_1)
                outputs = net.classifier(features)
                loss = criterion(outputs, target)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                val_bar.set_postfix(acc=f'{100.*correct/total:.2f}%')

        val_loss /= len(val_loader)
        accuracy = 100.0 * correct / total
        writer.add_scalar('Loss/val', val_loss, linear_eval_epoch)
        writer.add_scalar('Accuracy/val', accuracy, linear_eval_epoch)

        # Step the scheduler
        scheduler.step(val_loss)

        # Log the learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate/linear_eval', current_lr, linear_eval_epoch)

        logging.info(f'Linear Eval Epoch {linear_eval_epoch+1}/{linear_eval_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.2f}%, LR: {current_lr:.6f}')

        if accuracy > best_acc:
            best_acc = accuracy
            best_epoch = linear_eval_epoch
            torch.save(net.state_dict(), 'best_linear_eval_model.pth')
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            logging.info(f"Early stopping at epoch {linear_eval_epoch+1}")
            break

    logging.info(f'Best Linear Evaluation Accuracy: {best_acc:.2f}% at epoch {best_epoch+1}')
    end_time = time.time()
    linear_eval_time = end_time - start_time
    return best_acc, best_epoch, linear_eval_time

def linear_eval_all_augs(net: torch.nn.Module, 
                        train_loaders: dict, 
                        val_loader, 
                        config: TrainingConfig, 
                        writer) -> dict:
    """
    Perform linear evaluation for each augmentation type using WarmupCosineScheduler.
    """
    results = {}
    device = next(net.parameters()).device
    
    for aug_type, train_loader in train_loaders.items():
        logging.info(f"\nStarting linear evaluation for augmentation type: {aug_type}")
        
        # Reset the classifier for each augmentation type
        net.classifier = None
        net.add_classifier()
        net.classifier = net.classifier.to(device)
        
        optimizer = torch.optim.AdamW(
            net.classifier.parameters(),
            lr=config.linear_eval_lr,
            weight_decay=config.linear_eval_weight_decay
        )
        
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=config.linear_eval_warmup_epochs,
            total_epochs=config.linear_eval_epochs
        )
        
        start_time = time.time()
        net.freeze_encoder()
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.0
        best_epoch = 0
        no_improve = 0
        
        # Create new data loaders with the specified batch size
        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset
        current_train_loader = DataLoader(
            train_dataset, 
            batch_size=config.linear_eval_batch_size,
            shuffle=True,
            num_workers=config.num_workers
        )
        current_val_loader = DataLoader(
            val_dataset,
            batch_size=config.linear_eval_batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )
        
        for linear_eval_epoch in range(config.linear_eval_epochs):
            # Training phase
            net.classifier.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_bar = tqdm(current_train_loader,
                           desc=f'[{aug_type}] Linear Eval Train Epoch {linear_eval_epoch+1}/{config.linear_eval_epochs}')
            
            for data in train_bar:
                # Handle different data formats based on augmentation type
                if aug_type in ['mixup', 'cutmix']:
                    pos_1, pos_2, target, _ = data  # Ignore the idx
                elif len(data) == 3:  # If data contains (pos_1, pos_2, target)
                    pos_1, _, target = data  # Only use pos_1 for linear evaluation
                elif len(data) == 2:  # If data contains (input, target)
                    pos_1, target = data
                else:
                    raise ValueError(f"Unexpected data format with {len(data)} elements")
                
                pos_1, target = pos_1.to(device), target.to(device)
                
                with torch.no_grad():
                    features = net.get_features(pos_1)
                outputs = net.classifier(features)
                
                # For mixup/cutmix, we don't need to handle the interpolated labels 
                # during linear evaluation, just use the original labels
                loss = criterion(outputs, target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()
                train_bar.set_postfix(
                    loss=f'{loss.item():.4f}',
                    acc=f'{100.*train_correct/train_total:.2f}%'
                )
            
            train_loss /= len(current_train_loader)
            train_acc = 100. * train_correct / train_total
            
            # Validation phase
            net.classifier.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            val_bar = tqdm(current_val_loader, 
                         desc=f'[{aug_type}] Linear Eval Val Epoch {linear_eval_epoch+1}/{config.linear_eval_epochs}')
            
            with torch.no_grad():
                for data in val_bar:
                    # Handle different data formats for validation
                    if len(data) == 3:  # If data contains (pos_1, pos_2, target)
                        pos_1, _, target = data
                    elif len(data) == 2:  # If data contains (input, target)
                        pos_1, target = data
                    else:
                        raise ValueError(f"Unexpected validation data format with {len(data)} elements")
                    
                    pos_1, target = pos_1.to(device), target.to(device)
                    features = net.get_features(pos_1)
                    outputs = net.classifier(features)
                    loss = criterion(outputs, target)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
                    val_bar.set_postfix(acc=f'{100.*correct/total:.2f}%')
            
            val_loss /= len(current_val_loader)
            accuracy = 100.0 * correct / total
            
            # Step the scheduler
            scheduler.step()
            
            # Log metrics
            writer.add_scalar(f'Loss/train_{aug_type}', train_loss, linear_eval_epoch)
            writer.add_scalar(f'Loss/val_{aug_type}', val_loss, linear_eval_epoch)
            writer.add_scalar(f'Accuracy/train_{aug_type}', train_acc, linear_eval_epoch)
            writer.add_scalar(f'Accuracy/val_{aug_type}', accuracy, linear_eval_epoch)
            
            # Log the learning rate
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar(f'Learning_Rate/linear_eval_{aug_type}', current_lr, linear_eval_epoch)
            
            logging.info(f'[{aug_type}] Linear Eval Epoch {linear_eval_epoch+1}/{config.linear_eval_epochs}: '
                        f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                        f'Val Loss: {val_loss:.4f}, Val Acc: {accuracy:.2f}%, '
                        f'LR: {current_lr:.6f}')
            
            if accuracy > best_acc:
                best_acc = accuracy
                best_epoch = linear_eval_epoch
                # Save the best model for this augmentation type
                torch.save({
                    'epoch': linear_eval_epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'accuracy': accuracy
                }, os.path.join(config.save_dir, f'best_linear_eval_model_{aug_type}.pth'))
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= config.early_stopping_patience:
                logging.info(f"[{aug_type}] Early stopping at epoch {linear_eval_epoch+1}")
                break
        
        end_time = time.time()
        eval_time = end_time - start_time
        
        results[aug_type] = {
            'best_accuracy': best_acc,
            'best_epoch': best_epoch,
            'eval_time': eval_time
        }
        
        logging.info(f'[{aug_type}] Best Linear Evaluation Accuracy: {best_acc:.2f}% at epoch {best_epoch+1}')
        logging.info(f'[{aug_type}] Evaluation time: {eval_time/60:.2f} minutes')
    
    # Find the best performing augmentation
    best_aug = max(results.items(), key=lambda x: x[1]['best_accuracy'])
    logging.info(f"\nBest performing augmentation: {best_aug[0]} with accuracy: {best_aug[1]['best_accuracy']:.2f}%")
    
    return results

def final_evaluation(net, test_loader, device, save_dir):
    CLASS_MAPPING = {
        0: 'CN',
        1: 'MCI',
        2: 'Dementia'
    }
    class_names = [CLASS_MAPPING[i] for i in range(3)]
    net.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, _, target in test_loader:
            data, target = data.to(device), target.to(device)
            features = net(data, return_features=True)
            output = net.classifier(features)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    cm = confusion_matrix(all_targets, all_preds)
    cm = cm[::-1]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.yticks([0.5, 1.5, 2.5], class_names[::-1])
    plt.xticks([0.5, 1.5, 2.5], class_names)

    try:
        plt.savefig(os.path.join(save_dir, f'final_confusion_matrix_{time.strftime("%Y%m%d-%H%M%S")}.png'))
    except Exception as e:
        logging.error(f"Error saving confusion matrix: {str(e)}")
    finally:
        plt.close()

    precision, recall, f1, support = precision_recall_fscore_support(all_targets, all_preds, average=None)

    f3 = calculate_f_score(precision, recall, beta=3)
    f5 = calculate_f_score(precision, recall, beta=5)

    accuracy = np.sum(np.diag(cm)) / np.sum(cm)

    # Calculate macro and weighted averages
    macro_avg_f1 = np.mean(f1)
    weighted_avg_f1 = np.average(f1, weights=support)

    report = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f3': f3,
        'f5': f5,
        'support': support,
        'macro avg': {'f1-score': macro_avg_f1},
        'weighted avg': {'f1-score': weighted_avg_f1}
    }

    # Generate classification report
    current_date = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_filename = f'classification_report_{current_date}.txt'
    report_path = os.path.join(save_dir, report_filename)

    with open(report_path, 'w') as f:
        f.write("Classification Report:\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name}:\n")
            f.write(f"  Precision: {precision[i]:.4f}\n")
            f.write(f"  Recall: {recall[i]:.4f}\n")
            f.write(f"  F1-score: {f1[i]:.4f}\n")
            f.write(f"  F3-score: {f3[i]:.4f}\n")
            f.write(f"  F5-score: {f5[i]:.4f}\n")
            f.write(f"  Support: {support[i]:.1f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro Avg F1-score: {macro_avg_f1:.4f}\n")
        f.write(f"Weighted Avg F1-score: {weighted_avg_f1:.4f}\n")

    logging.info(f"Classification report saved to {report_path}")

    return report, accuracy