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
from scipy.ndimage import zoom
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import logging
from datetime import datetime
import sys
import time
import datetime as dt

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)
os.makedirs('logs/gradcam', exist_ok=True)  # 추가: GradCAM 저장 디렉토리

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

# Load the AAL atlas
atlas_path = 'aal116MNI_temp.nii'
try:
    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata()
    atlas_affine = atlas_img.affine
    
    # Normalize the atlas to the CNN input size (64x64x64)
    atlas_resized = zoom(atlas_data, (64 / atlas_data.shape[0],
                                    64 / atlas_data.shape[1],
                                    64 / atlas_data.shape[2]), order=1)
    log_info("Successfully loaded and resized AAL atlas")
except Exception as e:
    log_info(f"Error loading AAL atlas: {str(e)}")
    raise

class CustomImageDataset(Dataset):
    def __init__(self, data_df):
        self.data_df = data_df

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        try:
            img_path = self.data_df.iloc[idx][PATH_COLUMN]
            label = self.data_df.iloc[idx]['labels']
            label = torch.tensor(label, dtype=torch.int64)
            label = torch.nn.functional.one_hot(label, num_classes=3).float()
            
            img = nib.load(img_path).get_fdata()
            
            # Resize to 64x64x64
            img_resized = zoom(img, (64 / img.shape[0],
                                   64 / img.shape[1],
                                   64 / img.shape[2]), order=1)
            
            img = (img_resized - np.mean(img_resized)) / np.std(img_resized)
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
            
            return img, label
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {str(e)}")
            raise

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

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.bn1(self.conv1(x))))
        out = self.maxpool2(self.relu2(self.bn2(self.conv2(out))))
        out = self.relu3(self.bn3(self.conv3(out)))
        
        out = out.view(-1, 64*3*3*3)
        out = self.dropout1(self.fc1(out))
        out = self.dropout2(self.fc2(out))
        out = self.softmax(self.fc3(out))
        
        return out

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activation = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activation = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, class_idx, inputs):
        try:
            self.model.eval()
            self.outputs = self.model(inputs)
            
            # 디버깅을 위한 출력 추가
            predicted_class = self.outputs.argmax(dim=1)[0].item()
            print(f"Predicted class: {predicted_class}")
            print(f"Target class: {class_idx}")
            print(f"Output probabilities: {self.outputs[0].detach().cpu().numpy()}")
            
            if class_idx >= self.outputs.shape[1]:
                logging.error(f"Invalid class index: {class_idx}, must be less than {self.outputs.shape[1]}")
                return None

            # 해당 클래스에 대한 그래디언트 계산
            target = self.outputs[0, class_idx]
            target.backward()

            if self.gradients is None:
                logging.error("No gradients computed!")
                return None

            # CAM 생성
            weights = self.gradients.mean(dim=[2, 3, 4], keepdim=True)
            cam = (weights * self.activation).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            
            # Numpy로 변환
            cam = cam.squeeze().detach().cpu().numpy()
            
            if cam.size == 0:
                logging.error("Empty CAM tensor")
                return None
                
            return cam
            
        except Exception as e:
            logging.error(f"Error in GradCAM generation: {str(e)}")
            return None

def save_gradcam_visualization(inputs, cam, class_idx, save_path, epoch, batch_idx):
    try:
        if cam is None:
            return
            
        # Ensure the save directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Create a larger figure to accommodate all views
        plt.figure(figsize=(20, 15))
        
        # Get middle slices for each view
        mid_z = inputs.shape[2] // 2  # Axial
        mid_y = inputs.shape[1] // 2  # Coronal
        mid_x = inputs.shape[0] // 2  # Sagittal
        
        # Row 1: Original MRI
        plt.subplot(3, 3, 1)
        plt.title('Original MRI (Axial)')
        original_slice = inputs.squeeze().cpu().numpy()
        plt.imshow(original_slice[:, :, mid_z], cmap='gray')
        plt.axis('off')
        
        plt.subplot(3, 3, 2)
        plt.title('Original MRI (Coronal)')
        plt.imshow(original_slice[:, mid_y, :], cmap='gray')
        plt.axis('off')
        
        plt.subplot(3, 3, 3)
        plt.title('Original MRI (Sagittal)')
        plt.imshow(original_slice[mid_x, :, :].T, cmap='gray')
        plt.axis('off')
        
        # Row 2: AAL Atlas
        plt.subplot(3, 3, 4)
        plt.title('AAL Atlas (Axial)')
        plt.imshow(atlas_resized[:, :, mid_z], cmap='gray')
        plt.axis('off')
        
        plt.subplot(3, 3, 5)
        plt.title('AAL Atlas (Coronal)')
        plt.imshow(atlas_resized[:, mid_y, :], cmap='gray')
        plt.axis('off')
        
        plt.subplot(3, 3, 6)
        plt.title('AAL Atlas (Sagittal)')
        plt.imshow(atlas_resized[mid_x, :, :].T, cmap='gray')
        plt.axis('off')
        
        # Row 3: GradCAM Overlay
        # Resize CAM to match atlas size
        cam_resized = zoom(cam, (atlas_resized.shape[0] / cam.shape[0],
                                atlas_resized.shape[1] / cam.shape[1],
                                atlas_resized.shape[2] / cam.shape[2]), order=1)
        
        plt.subplot(3, 3, 7)
        plt.title(f'GradCAM Class {class_idx} (Axial)')
        plt.imshow(atlas_resized[:, :, mid_z], cmap='gray')
        plt.imshow(cam_resized[:, :, mid_z], cmap='jet', alpha=0.5)
        plt.axis('off')
        
        plt.subplot(3, 3, 8)
        plt.title(f'GradCAM Class {class_idx} (Coronal)')
        plt.imshow(atlas_resized[:, mid_y, :], cmap='gray')
        plt.imshow(cam_resized[:, mid_y, :], cmap='jet', alpha=0.5)
        plt.axis('off')
        
        plt.subplot(3, 3, 9)
        plt.title(f'GradCAM Class {class_idx} (Sagittal)')
        plt.imshow(atlas_resized[mid_x, :, :].T, cmap='gray')
        plt.imshow(cam_resized[mid_x, :, :].T, cmap='jet', alpha=0.5)
        plt.axis('off')
        
        plt.suptitle(f'Epoch {epoch}, Batch {batch_idx}', y=0.95)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close('all')
        
    except Exception as e:
        logging.error(f"Error saving GradCAM visualization: {str(e)}")
    finally:
        plt.close('all')
        torch.cuda.empty_cache()

def train_one_epoch(train_loader, model, criterion, optimizer, device, patience=None):
    model.train()
    total_loss = 0
    patience_list = []
    
    for batch, (img, label) in enumerate(train_loader):
        try:
            img, label = img.to(device), label.to(device)
            
            optimizer.zero_grad()
            pred = model(img)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch % 10 == 0:
                current = (batch + 1) * len(img)
                log_info(f"loss: {loss.item():>7f}  [{current:>5d}/{len(train_loader.dataset):>5d}]")
                
                if patience:
                    if len(patience_list) < patience:
                        patience_list.append(loss.item())
                    else:
                        max_loss = max(patience_list)
                        patience_list.append(loss.item())
                        patience_list = patience_list[-patience:]
                        if loss.item() > max_loss:
                            current_lr = optimizer.param_groups[0]['lr']
                            optimizer.param_groups[0]['lr'] = round(current_lr * 0.5, 10)
                            log_info(f'Reducing learning rate to: {optimizer.param_groups[0]["lr"]}')
                            
        except Exception as e:
            logging.error(f"Error in training batch {batch}: {str(e)}")
            continue
            
    return total_loss / len(train_loader)

def validate(dataloader, model, criterion, device, epoch):
    model.eval()
    val_loss = 0
    correct = 0
    all_labels = []
    all_preds = []
    num_samples_to_visualize = 3
    
    try:
        with torch.no_grad():
            for batch_idx, (img, label) in enumerate(dataloader):
                img, label = img.to(device), label.to(device)
                output = model(img)
                
                val_loss += criterion(output, label).item()
                pred = output.argmax(1)
                target = label.argmax(1)
                correct += (pred == target).type(torch.float).sum().item()
                
                all_labels.extend(target.cpu().numpy())
                all_preds.extend(pred.cpu().numpy())
                
                # GradCAM visualization for selected samples
                # validate 함수 내부의 GradCAM 생성 부분
                if batch_idx < num_samples_to_visualize:
                    try:
                        with torch.enable_grad():
                            grad_cam = GradCAM(model, model.conv3)
                            
                            for j in range(min(3, len(img))):
                                true_class = target[j].item()
                                pred_class = pred[j].item()
                                
                                # 실제 클래스와 예측 클래스 모두에 대해 GradCAM 생성
                                cam_true = grad_cam.generate(true_class, img[j:j+1])
                                cam_pred = grad_cam.generate(pred_class, img[j:j+1])
                                
                                if cam_true is not None:
                                    save_path = f'logs/gradcam/gradcam_true_epoch{epoch}_batch{batch_idx}_sample{j}_class{true_class}.png'
                                    save_gradcam_visualization(img[j], cam_true, true_class, save_path, epoch, batch_idx)
                                
                                if cam_pred is not None and pred_class != true_class:
                                    save_path = f'logs/gradcam/gradcam_pred_epoch{epoch}_batch{batch_idx}_sample{j}_class{pred_class}.png'
                                    save_gradcam_visualization(img[j], cam_pred, pred_class, save_path, epoch, batch_idx)
                                
                    except Exception as e:
                        logging.error(f"Error generating GradCAM in validation: {str(e)}")
                        continue

        # Calculate metrics
        val_loss /= len(dataloader)
        accuracy = 100 * correct / len(dataloader.dataset)
        
        # Log class distribution
        class_counts = pd.Series(all_labels).value_counts()
        log_info(f"\nClass distribution in validation set:\n{class_counts}")
        
        return val_loss, accuracy, all_labels, all_preds
        
    except Exception as e:
        logging.error(f"Error in validation: {str(e)}")
        return float('inf'), 0.0, [], []

def main():
    try:
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

        # Log class distribution
        train_class_counts = train_df['labels'].value_counts()
        log_info(f"\nClass distribution in training set:\n{train_class_counts}")

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, drop_last=True)

        # Initialize model, criterion, and optimizer
        model = Custom3DCNN().to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        # Training parameters
        num_epochs = 30
        patience = 5

        # Training metrics
        train_losses = []
        val_losses = []
        val_accuracies = []

        log_info("\n=== Starting Training ===")
        start_time = datetime.now()

        for epoch in range(num_epochs):
            try:
                epoch_start_time = datetime.now()
                log_info(f"\nEpoch {epoch+1}/{num_epochs}")
                log_info("-" * 20)
                
                # Training phase
                train_loss = train_one_epoch(train_loader, model, criterion, optimizer, device, patience)
                
                # Validation phase
                val_loss, accuracy, labels, preds = validate(val_loader, model, criterion, device, epoch)
                
                # Record metrics
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                val_accuracies.append(accuracy)
                
                epoch_duration = datetime.now() - epoch_start_time
                
                # Log epoch summary
                log_info(f"\nEpoch {epoch+1} Summary:")
                log_info(f"Train Loss: {train_loss:.4f}")
                log_info(f"Validation Loss: {val_loss:.4f}")
                log_info(f"Validation Accuracy: {accuracy:.2f}%")
                log_info(f"Epoch Duration: {epoch_duration}")

                # Save confusion matrix every 5 epochs
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
                        logging.error(f"Error saving confusion matrix: {str(e)}")

                # Memory cleanup
                torch.cuda.empty_cache()
            
            except Exception as e:
                logging.error(f"Error in epoch {epoch + 1}: {str(e)}")
                continue
        
        # Save final model
        torch.save(model.state_dict(), 'logs/adni_model.pt')
        
        # Create and save final training plots
        try:
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
            
        except Exception as e:
            logging.error(f"Error creating training history plots: {str(e)}")
        
        log_info("\n=== Training Completed Successfully ===")
        log_info(f"Total duration: {datetime.now() - start_time}")
        
    except Exception as e:
        log_info("\n=== Error occurred during training ===")
        log_info(f"Error message: {str(e)}")
        raise
        
    finally:
        # Cleanup
        plt.close('all')
        torch.cuda.empty_cache()
        log_info("\n=== Script Execution Finished ===")

if __name__ == "__main__":
    main()