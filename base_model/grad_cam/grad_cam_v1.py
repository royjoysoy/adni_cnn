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
import time
import datetime as dt

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Set up logging
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'logs/training_log_{current_time}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)

def log_info(message):
    logging.info(message)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_info(f"Using device: {device}")

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
        
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv3d(64, 64, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=3)
        
        self.conv3 = nn.Conv3d(64, 64, kernel_size=7, stride=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.relu3 = nn.ReLU()
        
        self.fc1 = nn.Linear(1728, 864)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(864, 100)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(100, 3)
        self.softmax = nn.Softmax(dim=1)
        
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        # Store activations for GradCAM
        self.activations = x
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        
        x = x.view(-1, 64*3*3*3)
        x = self.dropout1(self.fc1(x))
        x = self.dropout2(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self):
        return self.activations

class GradCAM:
    def __init__(self, model):
        self.model = model
        
    def generate_cam(self, input_image, target_class=None):
    # 기존 텐서의 requires_grad 상태를 저장
        original_requires_grad = input_image.requires_grad
    
        try:
            # gradient 활성화
            input_image.requires_grad = True
            
            # Set model to eval mode but enable gradients
            self.model.eval()
            
            # Forward pass
            output = self.model(input_image)
            
            if target_class is None:
                target_class = output.argmax(dim=1)
            
            # Zero all existing gradients
            self.model.zero_grad()
            
            # Get model output for target class
            target_output = output[0, target_class]
            
            # Backward pass
            target_output.backward()
        
            # Get gradients and activations
            gradients = self.model.get_activations_gradient()
            activations = self.model.get_activations()
        
            if gradients is None or activations is None:
                return None
        
            # Calculate weights
            weights = torch.mean(gradients, dim=(2, 3, 4))[0]
        
            # Generate CAM
            cam = torch.zeros(activations.shape[2:], dtype=torch.float32).to(device)
            for i, w in enumerate(weights):
                cam += w * activations[0, i, :, :, :]
        
            # Apply ReLU and normalize
            cam = F.relu(cam)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-7)  # Added small epsilon to prevent division by zero
        
            # CPU로 이동하고 numpy로 변환
            cam = cam.cpu().detach().numpy()
        
            # 메모리 정리
            del gradients
            del activations
            torch.cuda.empty_cache()  # GPU 메모리 정리
        
            return cam
        
        except Exception as e:
            logging.error(f"Error in GradCAM generation: {str(e)}")
            return None
        
        finally:
            # 원래 상태로 복구
            input_image.requires_grad = original_requires_grad

def save_gradcam_visualization(model, image, label, save_path):
    """GradCAM visualization with error handling"""
    try:
        gradcam = GradCAM(model)
        cam = gradcam.generate_cam(image.unsqueeze(0).to(device))
        
        if cam is None:
            logging.warning(f"Failed to generate GradCAM for {save_path}")
            return
        
        # Get central slice
        original_slice = image[0, :, :, cam.shape[2]//2].cpu().numpy()
        cam_slice = cam[:, :, cam.shape[2]//2]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        ax1.imshow(original_slice, cmap='gray')
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # GradCAM
        ax2.imshow(cam_slice, cmap='jet')
        ax2.set_title('GradCAM')
        ax2.axis('off')
        
        # Overlay
        ax3.imshow(original_slice, cmap='gray')
        ax3.imshow(cam_slice, cmap='jet', alpha=0.5)
        ax3.set_title('Overlay')
        ax3.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    except Exception as e:
        logging.error(f"Error in saving GradCAM visualization: {str(e)}")

def validate(dataloader, model, criterion, device, epoch):
    model.eval()
    val_loss = 0
    correct = 0
    all_labels = []
    all_preds = []
    total_processed = 0  # 추가: 실제로 처리된 배치 수 tracking
    
    os.makedirs(f'logs/gradcam_epoch_{epoch}', exist_ok=True)
    
    with torch.set_grad_enabled(True):
        for batch_idx, (img, label) in enumerate(dataloader):
            try:
                img, label = img.to(device), label.to(device)
                output = model(img)
                
                val_loss += criterion(output, label).item()
                correct += (output.argmax(1) == label.argmax(1)).type(torch.float).sum().item()
                
                all_labels.extend(label.argmax(1).cpu().numpy())
                all_preds.extend(output.argmax(1).cpu().numpy())
                
                total_processed += 1  # 성공적으로 처리된 배치 카운트
                
                if batch_idx % 5 == 0:
                    save_gradcam_visualization(
                        model, 
                        img[0],
                        label[0],
                        f'logs/gradcam_epoch_{epoch}/gradcam_batch_{batch_idx}.png'
                    )
            except Exception as e:
                logging.error(f"Error during validation batch {batch_idx}: {str(e)}")
                continue
    
    # 실제 처리된 배치 수로 나누기
    if total_processed > 0:  # 0으로 나누기 방지
        val_loss /= total_processed
        accuracy = 100 * correct / (total_processed * dataloader.batch_size)
    else:
        val_loss = float('inf')
        accuracy = 0.0
        logging.error("No batches were successfully processed during validation")
    
    return val_loss, accuracy, all_labels, all_preds

def kms_train_loop(model, train_loader, optimizer, loss_fn, device):
    """KMS training loop function"""
    model.train()
    train_loss = []
    train_acc = []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        target_idx = target.argmax(dim=1, keepdim=True)
        correct = pred.eq(target_idx).sum().item()
        acc = correct / len(data)
        
        train_loss.append(loss.item())
        train_acc.append(acc)
        
        if batch_idx % 10 == 0:
            log_info(f'Train Batch [{batch_idx}/{len(train_loader)}] '
                    f'Loss: {loss.item():.6f} Acc: {acc:.4f}')
    
    return np.mean(train_loss), np.mean(train_acc)

def main():
    # Load and prepare data
    log_info("\nLoading data...")
    df = pd.read_csv('/home/simclr_project/simclr/SimCLR_RS/df_modified.csv')
    log_info(f"Loaded dataset with {len(df)} samples")

    global PATH_COLUMN, LABEL_COLUMN
    PATH_COLUMN = 'New_Path'
    LABEL_COLUMN = 'DX2'

    # Split data
    paths = df[PATH_COLUMN].tolist()
    labels = df[LABEL_COLUMN]
    X_train, X_val, y_train, y_val = train_test_split(
        paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_df = pd.DataFrame({PATH_COLUMN: X_train, 'labels': y_train})
    val_df = pd.DataFrame({PATH_COLUMN: X_val, 'labels': y_val})

    # Create data loaders
    train_dataset = CustomImageDataset(train_df)
    val_dataset = CustomImageDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, drop_last=True)

    # Initialize model, criterion, and optimizer
    model = Custom3DCNN().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training parameters
    num_epochs = 30
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    log_info("\n=== Starting Training ===")
    start_time = datetime.now()

    try:
        for epoch in range(num_epochs):
            epoch_start_time = datetime.now()
            log_info(f"\nEpoch {epoch+1}/{num_epochs}")
            log_info("-" * 20)
            
            try:
                # Training phase using kms_train_loop
                train_loss, train_acc = kms_train_loop(model, train_loader, optimizer, criterion, device)
                    
                # Validation phase with GradCAM
                val_loss, val_acc, labels, preds = validate(val_loader, model, criterion, device, epoch)
                
                train_losses.append(train_loss)
                train_accuracies.append(train_acc)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
            
                epoch_duration = datetime.now() - epoch_start_time
            
                log_info(f"\nEpoch {epoch+1} Summary:")
                log_info(f"Train Loss: {train_loss:.4f}")
                log_info(f"Train Accuracy: {train_acc:.2f}")
                log_info(f"Validation Loss: {val_loss:.4f}")
                log_info(f"Validation Accuracy: {val_acc:.2f}%")
                log_info(f"Epoch Duration: {epoch_duration}")

                # Save confusion matrix plot
                if (epoch + 1) % 5 == 0:
                    try:
                        cm = confusion_matrix(labels, preds)
                        plt.figure(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                        plt.title(f'Confusion Matrix - Epoch {epoch+1}')
                        plt.ylabel('True Label')
                        plt.xlabel('Predicted Label')
                        plt.savefig(f'logs/confusion_matrix_epoch_{epoch+1}.png')
                        plt.close()
                    except Exception as e:
                        log_info(f"Error creating confusion matrix for epoch {epoch+1}: {str(e)}")

            except Exception as e:
                log_info(f"\nError during epoch {epoch+1}: {str(e)}")
                continue  # Skip to next epoch if current one fails
            
        # Save model
        torch.save(model.state_dict(), 'logs/adni_model.pt')
        
        # Create and save training history plots
        try:   
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.subplot(1, 3, 2)
            plt.plot(train_accuracies, label='Train Accuracy')
            plt.plot(val_accuracies, label='Validation Accuracy')
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.tight_layout()
            plt.savefig('logs/training_history.png')
            plt.close()

        except Exception as e:
            log_info(f"Error creating training history plots: {str(e)}")
            
        log_info("\n=== Training Completed Successfully ===")
        log_info(f"Total duration: {datetime.now() - start_time}")
        
    except Exception as e:
        log_info("\n=== Error occurred during training ===")
        log_info(f"Error message: {str(e)}")
        raise
        
    finally:
        log_info("\n=== Script Execution Finished ===")

if __name__ == "__main__":
    main()