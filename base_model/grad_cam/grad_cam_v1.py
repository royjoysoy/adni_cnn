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
import SimpleITK as sitk

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Set up logging first
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
    """Helper function for logging info messages"""
    logging.info(message)

# Define template path
TEMPLATE_PATH = "/home/simclr_project/simclr/SimCLR_RS/templates/mni305.cor.mgz"

# Verify template exists
if not os.path.exists(TEMPLATE_PATH):
    raise FileNotFoundError(f"Template not found at {TEMPLATE_PATH}")
    
log_info(f"Using template from: {TEMPLATE_PATH}")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_info(f"Using device: {device}")

def register_to_template(moving_image, fixed_template):
    """Register moving image to template using SimpleITK"""
    try:
        registration_method = sitk.ImageRegistrationMethod()
        
        # Set up registration parameters
        registration_method.SetMetricAsMeanSquares()
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, 
                                                         numberOfIterations=100)
        registration_method.SetInitialTransform(sitk.TranslationTransform(3))
        
        # Perform registration
        final_transform = registration_method.Execute(sitk.GetImageFromArray(moving_image),
                                                    sitk.GetImageFromArray(fixed_template))
        
        # Apply transform
        registered_image = sitk.Resample(sitk.GetImageFromArray(moving_image),
                                       sitk.GetImageFromArray(fixed_template),
                                       final_transform,
                                       sitk.sitkLinear)
        
        return sitk.GetArrayFromImage(registered_image)
    except Exception as e:
        logging.error(f"Registration failed: {str(e)}")
        return moving_image

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
        img = (img - img.mean()) / img.std()
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
        original_requires_grad = input_image.requires_grad
    
        try:
            input_image.requires_grad = True
            self.model.eval()
            
            output = self.model(input_image)
            
            if target_class is None:
                target_class = output.argmax(dim=1)
            
            self.model.zero_grad()
            target_output = output[0, target_class]
            target_output.backward()
        
            gradients = self.model.get_activations_gradient()
            activations = self.model.get_activations()
        
            if gradients is None or activations is None:
                return None
        
            weights = torch.mean(gradients, dim=(2, 3, 4))[0]
            cam = torch.zeros(activations.shape[2:], dtype=torch.float32).to(device)
            
            for i, w in enumerate(weights):
                cam += w * activations[0, i, :, :, :]
        
            cam = F.relu(cam)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-7)
            
            cam = cam.cpu().detach().numpy()
            
            del gradients
            del activations
            torch.cuda.empty_cache()
        
            return cam
        
        except Exception as e:
            logging.error(f"Error in GradCAM generation: {str(e)}")
            return None
        
        finally:
            input_image.requires_grad = original_requires_grad

def save_gradcam_visualization(model, image, label, save_path, template_path=None):
    try:
        # Generate GradCAM
        gradcam = GradCAM(model)
        cam = gradcam.generate_cam(image.unsqueeze(0).to(device))
        
        if cam is None:
            logging.warning(f"Failed to generate GradCAM for {save_path}")
            return
        
        # Resize CAM
        cam = torch.tensor(cam).to(device)
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(64, 64, 64),
            mode='trilinear',
            align_corners=False
        ).squeeze().cpu().numpy()
        
        # Process input image
        image_np = image[0].cpu().numpy()
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        
        # Load and preprocess template
        if template_path and os.path.exists(template_path):
            try:
                template_img = nib.load(template_path)
                template = template_img.get_fdata()
                
                if template.ndim != 3:
                    raise ValueError(f"Template should be 3D, got {template.ndim}D")
                
                # Get orientation and ensure consistency
                template_orientation = nib.aff2axcodes(template_img.affine)
                template = nib.as_closest_canonical(template_img).get_fdata()
                
                # Single resize operation
                template = skTrans.resize(template, (64, 64, 64), order=1, preserve_range=True)
                
                # Register template to input image
                template = register_to_template(template, image_np)
                
                # Robust normalization
                p1, p99 = np.percentile(template, (1, 99))
                template = np.clip(template, p1, p99)
                template = (template - template.min()) / (template.max() - template.min())
                
                if np.isnan(template).any() or np.isinf(template).any():
                    raise ValueError("Template contains NaN or Inf values")
                    
            except Exception as e:
                logging.error(f"Error processing template: {str(e)}")
                template = image_np
        else:
            template = image_np
            
        # Create visualization
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        views = ['Sagittal', 'Coronal', 'Axial']
        
        for i, view in enumerate(views):
            if view == 'Sagittal':
                img_slice = image_np[:, :, image_np.shape[2]//2]
                cam_slice = cam[:, :, cam.shape[2]//2]
                temp_slice = template[:, :, template.shape[2]//2]
            elif view == 'Coronal':
                img_slice = image_np[:, image_np.shape[1]//2, :]
                cam_slice = cam[:, cam.shape[1]//2, :]
                temp_slice = template[:, template.shape[1]//2, :]
            else:  # Axial
                img_slice = image_np[image_np.shape[0]//2, :, :]
                cam_slice = cam[cam.shape[0]//2, :, :]
                temp_slice = template[template.shape[0]//2, :, :]
            
            # Plot slices
            axes[i,0].imshow(img_slice, cmap='gray')
            axes[i,0].set_title(f'{view} - Original')
            axes[i,0].axis('off')
            
            axes[i,1].imshow(temp_slice, cmap='gray')
            axes[i,1].set_title(f'{view} - Template')
            axes[i,1].axis('off')
            
            axes[i,2].imshow(cam_slice, cmap='jet')
            axes[i,2].set_title(f'{view} - GradCAM')
            axes[i,2].axis('off')
            
            axes[i,3].imshow(temp_slice, cmap='gray')
            axes[i,3].imshow(cam_slice, cmap='jet', alpha=0.6)
            axes[i,3].set_title(f'{view} - Template + GradCAM')
            axes[i,3].axis('off')
        
        plt.suptitle(f'GradCAM Visualization (Class {label.argmax().item()})')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    except Exception as e:
        logging.error(f"Error in saving GradCAM visualization: {str(e)}")

def kms_train_loop(model, train_loader, optimizer, loss_fn, device):
    """Training loop function"""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
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
        
        total_loss += loss.item()
        total_correct += correct
        total_samples += len(data)
        
        if batch_idx % 10 == 0:
            batch_acc = (correct / len(data)) * 100
            log_info(f'Train Batch [{batch_idx}/{len(train_loader)}] '
                    f'Loss: {loss.item():.6f} Acc: {batch_acc:.2f}%')
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = (total_correct / total_samples) * 100
    
    return avg_loss, avg_acc

def validate(dataloader, model, criterion, device, epoch):
    """Validation function with GradCAM visualization"""
    model.eval()
    val_loss = 0
    correct = 0
    all_labels = []
    all_preds = []
    total_processed = 0
    
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
                
                total_processed += 1
                
                if batch_idx % 5 == 0:
                    save_gradcam_visualization(
                        model, 
                        img[0],
                        label[0],
                        f'logs/gradcam_epoch_{epoch}/gradcam_batch_{batch_idx}.png',
                        template_path=TEMPLATE_PATH
                    )
            except Exception as e:
                logging.error(f"Error during validation batch {batch_idx}: {str(e)}")
                continue
    
    if total_processed > 0:
        val_loss /= total_processed
        accuracy = 100 * correct / (total_processed * dataloader.batch_size)
    else:
        val_loss = float('inf')
        accuracy = 0.0
        logging.error("No batches were successfully processed during validation")
    
    return val_loss, accuracy, all_labels, all_preds

def main():
    """Main training function"""
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
                # Training phase
                train_loss, train_acc = kms_train_loop(model, train_loader, optimizer, criterion, device)
                    
                # Validation phase with GradCAM
                val_loss, val_acc, labels, preds = validate(val_loader, model, criterion, device, epoch)
                
                # Record metrics
                train_losses.append(train_loss)
                train_accuracies.append(train_acc)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
            
                epoch_duration = datetime.now() - epoch_start_time
            
                # Log epoch summary
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
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                  yticklabels=sorted(range(3), reverse=True))
                        plt.title(f'Confusion Matrix - Epoch {epoch+1}')
                        plt.ylabel('True Label')
                        plt.xlabel('Predicted Label')
                        plt.savefig(f'logs/confusion_matrix_epoch_{epoch+1}.png')
                        plt.close()
                    except Exception as e:
                        log_info(f"Error creating confusion matrix for epoch {epoch+1}: {str(e)}")

            except Exception as e:
                log_info(f"\nError during epoch {epoch+1}: {str(e)}")
                continue
            
        # Save final model
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