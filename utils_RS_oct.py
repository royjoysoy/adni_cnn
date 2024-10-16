import argparse
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import logging
import psutil
import GPUtil
import sys
import numpy as np
from datetime import datetime

def get_args():
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dimension for SimCLR')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature parameter for contrastive loss')
    parser.add_argument('--k', default=8, type=int, help='Number of neighbors for kNN classifier')
    parser.add_argument('--simclr_batch_size', default=32, type=int, help='Batch size for SimCLR training') # SimCLR batch size
    parser.add_argument('--linear_eval_batch_size', default=32, type=int, help='Batch size for linear evaluation')
    parser.add_argument('--simclr_epochs', default=64, type=int, help='Number of epochs to train for SimCLR') # SimCLR epoch set
    parser.add_argument('--linear_eval_epochs', default=128, type=int, help='Number of epochs for linear evaluation')
    parser.add_argument('--lr', default=1e-7, type=float, help='Initial learning rate')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data CSV file')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')

    # Learning rate scheduler arguments
    parser.add_argument('--scheduler_factor', default=0.1, type=float, help='Factor by which the learning rate will be reduced')
    parser.add_argument('--scheduler_patience', default=5, type=int, help='Number of epochs with no improvement after which learning rate will be reduced')
    parser.add_argument('--scheduler_min_lr', default=1e-6, type=float, help='A lower bound on the learning rate')

    # Other arguments
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay for optimizer')
    parser.add_argument('--early_stopping_patience', default=15, type=int, help='Patience for early stopping')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers for data loading')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility')
    parser.add_argument('--log_interval', default=10, type=int, help='How many batches to wait before logging training status')
    parser.add_argument('--save_interval', default=5, type=int, help='How many epochs to wait before saving a model checkpoint')

    # Check if the script is being run in a notebook
    if 'ipykernel' in sys.modules:
        # If in a notebook, use default values
        args = parser.parse_args([
            '--data_path', '/home/simclr_project/simclr/SimCLR_RS/df_modified.csv',
            '--save_dir', '/home/simclr_project/simclr/SimCLR_RS/linear_eval_results'
        ])
    else:
        args = parser.parse_args()

    return args

def calculate_f_score(precision, recall, beta=1):
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-10)

def save_model(model, optimizer, simclr_epoch, path):
    torch.save({
        'simclr_epoch': simclr_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    logging.info(f"Model saved to {path}")

def save_epoch_number(simclr_epoch, linear_eval_epoch, path):
    with open(path, 'w') as f:
        f.write(f"SimCLR epoch: {simclr_epoch}\n")
        f.write(f"Linear eval epoch: {linear_eval_epoch}\n")
    logging.info(f"Epoch numbers saved to {path}")

def load_epoch_number(path):
    # Find the most recent epoch file
    epoch_files = [f for f in os.listdir(os.path.dirname(path)) if f.startswith('epoch_') and f.endswith('.txt')]
    if epoch_files:
        most_recent_epoch = max(epoch_files)
        path = os.path.join(os.path.dirname(path), most_recent_epoch)

    if os.path.exists(path):
        with open(path, 'r') as f:
            lines = f.readlines()
            simclr_epoch = int(lines[0].split(': ')[1])
            linear_eval_epoch = int(lines[1].split(': ')[1])
        logging.info(f"Loaded SimCLR epoch {simclr_epoch} and Linear eval epoch {linear_eval_epoch} from {path}")
        return simclr_epoch, linear_eval_epoch
    else:
        logging.info(f"No epoch file found at {path}, starting from epoch 0 for both")
        return 0, 0  # Return 0 for both if no file is found
    
def load_model(net, optimizer, path):
    # Find the most recent model file
    model_files = [f for f in os.listdir(os.path.dirname(path)) if f.startswith('latest_model_') and f.endswith('.pth')]
    if model_files:
        most_recent_model = max(model_files)
        path = os.path.join(os.path.dirname(path), most_recent_model)

    if os.path.exists(path):
        checkpoint = torch.load(path)
        
        # Remove unexpected keys
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        
        net.load_state_dict(model_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        simclr_epoch = checkpoint['simclr_epoch']
        
        # If classifier exists in the loaded model, add it to the current model
        if 'classifier.weight' in checkpoint['model_state_dict']:
            net.add_classifier()
            net.classifier.load_state_dict({
                'weight': checkpoint['model_state_dict']['classifier.weight'],
                'bias': checkpoint['model_state_dict']['classifier.bias']
            })
        
        # If adaptive_layer exists in the loaded model, add it to the current model
        if 'adaptive_layer.weight' in checkpoint['model_state_dict']:
            if net.adaptive_layer is None:
                net.adaptive_layer = nn.Linear(864, 1728).to(net.f.fc1.weight.device)
            net.adaptive_layer.load_state_dict({
                'weight': checkpoint['model_state_dict']['adaptive_layer.weight'],
                'bias': checkpoint['model_state_dict']['adaptive_layer.bias']
            })
        
        logging.info(f"Model loaded from {path}, resuming from SimCLR epoch {simclr_epoch}")
        return net, optimizer, simclr_epoch
    
    logging.warning(f"No model found at {path}, starting from scratch")
    return net, optimizer, 0

def plot_learning_curves(results, save_dir, current_datetime):
    plt.figure(figsize=(12, 8))
    for aug_type, data in results.items():
        simclr_epochs = range(1, len(data['train_loss']) + 1)
        plt.plot(simclr_epochs, data['train_loss'], label=f'{aug_type} - Train Loss')
        plt.plot(simclr_epochs, data['val_accuracy'], label=f'{aug_type} - Val Accuracy')

    plt.title('Learning Curves for Different Augmentation Types')
    plt.xlabel('SimCLR Epochs')
    plt.ylabel('Loss / Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'learning_curves_{current_datetime}.png'))
    plt.close()

def get_hardware_info():
    gpus = GPUtil.getGPUs()
    gpu_info = f"{gpus[0].name}, {gpus[0].memoryTotal}MB" if gpus else "No GPU"
    cpu_info = f"{psutil.cpu_count(logical=False)} cores, {psutil.cpu_count()} threads"
    ram_info = f"{psutil.virtual_memory().total / (1024**3):.2f} GB"
    return f"GPU: {gpu_info}, CPU: {cpu_info}, RAM: {ram_info}"