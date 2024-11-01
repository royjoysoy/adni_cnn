import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import skimage.transform as skTrans
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import logging
from datetime import datetime
import sys

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Set up logging
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'logs/training_log_{current_time}.log'

# Configure logging to write to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)

# Custom logging function
def log_info(message):
    logging.info(message)

# Log system information
log_info(f"Python version: {sys.version}")
log_info(f"PyTorch version: {torch.__version__}")
log_info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log_info(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_info(f"Using device: {device}")

# Load and prepare data
log_info("\nLoading data...")
df = pd.read_csv('/home/simclr_project/simclr/SimCLR_RS/df_modified.csv')
log_info(f"Loaded dataset with {len(df)} samples")

# Using correct column names from the CSV file
PATH_COLUMN = 'New_Path'
LABEL_COLUMN = 'DX2'

# Split data into train and validation sets
paths = df[PATH_COLUMN].tolist()
labels = df[LABEL_COLUMN]
X_train, X_val, y_train, y_val = train_test_split(paths, labels, test_size=0.2, random_state=42, stratify=labels)

train_df = pd.DataFrame({PATH_COLUMN: X_train, 'labels': y_train})
val_df = pd.DataFrame({PATH_COLUMN: X_val, 'labels': y_val})

# Log dataset information
log_info(f"\nTotal samples: {len(df)}")
log_info(f"Training samples: {len(train_df)}")
log_info(f"Validation samples: {len(val_df)}")
log_info("\nLabel distribution in training set:")
log_info(str(train_df['labels'].value_counts()))
log_info("\nLabel distribution in validation set:")
log_info(str(val_df['labels'].value_counts()))

class CustomImageDataset(Dataset):
    def __init__(self, data_df):
        self.data_df = data_df

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        img_path = self.data_df.iloc[idx][PATH_COLUMN]
        label = self.data_df.iloc[idx]['labels']
        label = torch.tensor(label, dtype=torch.int64)
        label = torch.nn.functional.one_hot(label, num_classes=3).float()
        
        img = nib.load(img_path).get_fdata()
        img = skTrans.resize(img, (64, 64, 64), order=1, preserve_range=True)
        img = (img - img.min()) / (img.max() - img.min())
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        
        return img, label

class Custom3DCNN(nn.Module):
    def __init__(self):
        super(Custom3DCNN, self).__init__()
        
        # Convolution Block 1
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Convolution Block 2
        self.conv2 = nn.Conv3d(64, 64, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=3)
        
        # Convolution Block 3
        self.conv3 = nn.Conv3d(64, 64, kernel_size=7, stride=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.relu3 = nn.ReLU()
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(1728, 864)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(864, 100)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(100, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Convolution Blocks
        out = self.maxpool1(self.relu1(self.bn1(self.conv1(x))))
        out = self.maxpool2(self.relu2(self.bn2(self.conv2(out))))
        out = self.relu3(self.bn3(self.conv3(out)))
        
        # Fully Connected Layers
        out = out.view(-1, 64*3*3*3)
        out = self.dropout1(self.fc1(out))
        out = self.dropout2(self.fc2(out))
        out = self.softmax(self.fc3(out))
        
        return out

def train_loop(dataloader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch, (img, label) in enumerate(dataloader):
        img, label = img.to(device), label.to(device)
        
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if batch % 10 == 0:
            current = (batch + 1) * len(img)
            log_info(f"loss: {loss.item():>7f}  [{current:>5d}/{len(dataloader.dataset):>5d}]")
    
    return total_loss / len(dataloader)

def validate(dataloader, model, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for img, label in dataloader:
            img, label = img.to(device), label.to(device)
            output = model(img)
            
            val_loss += criterion(output, label).item()
            correct += (output.argmax(1) == label.argmax(1)).type(torch.float).sum().item()
            
            all_labels.extend(label.argmax(1).cpu().numpy())
            all_preds.extend(output.argmax(1).cpu().numpy())
    
    val_loss /= len(dataloader)
    accuracy = 100 * correct / len(dataloader.dataset)
    
    log_info(f"Validation Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {val_loss:>8f}")
    
    return val_loss, accuracy, all_labels, all_preds

# Create data loaders
train_dataset = CustomImageDataset(train_df)
val_dataset = CustomImageDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, drop_last=True)

# Initialize model, criterion, and optimizer
model = Custom3DCNN().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
train_losses = []
val_losses = []
val_accuracies = []

log_info("\n=== Starting Training ===")
log_info(f"Total epochs: {num_epochs}")
log_info(f"Training samples: {len(train_loader.dataset)}")
log_info(f"Validation samples: {len(val_loader.dataset)}")

try:
    start_time = datetime.now()
    log_info(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    for epoch in range(num_epochs):
        epoch_start_time = datetime.now()
        log_info(f"\nEpoch {epoch+1}/{num_epochs}")
        log_info("------------------")
        
        # Training phase
        log_info("Training phase:")
        train_loss = train_loop(train_loader, model, criterion, optimizer, device)
        
        # Validation phase
        log_info("\nValidation phase:")
        val_loss, accuracy, labels, preds = validate(val_loader, model, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(accuracy)
        
        epoch_end_time = datetime.now()
        epoch_duration = epoch_end_time - epoch_start_time
        
        log_info(f"\nEpoch {epoch+1} Summary:")
        log_info(f"Train Loss: {train_loss:.4f}")
        log_info(f"Validation Loss: {val_loss:.4f}")
        log_info(f"Validation Accuracy: {accuracy:.2f}%")
        log_info(f"Epoch Duration: {epoch_duration}")
        
        # Save confusion matrix plot
        if (epoch + 1) % 5 == 0:
            cm = confusion_matrix(labels, preds)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - Epoch {epoch+1}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(f'logs/confusion_matrix_epoch_{epoch+1}.png')
            plt.close()
            log_info(f"Confusion matrix saved for epoch {epoch+1}")
    
    end_time = datetime.now()
    total_duration = end_time - start_time
    
    log_info("\n=== Training Completed Successfully ===")
    log_info(f"Training duration: {total_duration}")
    log_info(f"Final Results:")
    log_info(f"Final Training Loss: {train_losses[-1]:.4f}")
    log_info(f"Final Validation Loss: {val_losses[-1]:.4f}")
    log_info(f"Final Validation Accuracy: {val_accuracies[-1]:.2f}%")
    
    # Save the model
    save_path = 'logs/adni_model.pt'
    torch.save(model.state_dict(), save_path)
    log_info(f"\nModel saved successfully to {save_path}")
    
    # Save training history plots
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('logs/training_history.png')
    plt.close()
    
    log_info("\n=== All processes completed successfully ===")
    log_info("1. Training completed")
    log_info("2. Model saved")
    log_info("3. Plots generated")
    log_info(f"4. Log file saved to: {log_filename}")

except Exception as e:
    log_info("\n=== Error occurred during training ===")
    log_info(f"Error message: {str(e)}")
    raise
    
finally:
    log_info("\n=== Script Execution Finished ===")