
'''기본 설정/유틸리티
    ↓
이미지 처리 함수들
    ↓
데이터셋 클래스
    ↓
모델 및 시각화 클래스들
    ↓
학습/검증 함수들
    ↓
메인 실행 함수
'''
# 1. Libraries

# 기본 Python 라이브러리
import os
import sys
import time
from datetime import datetime
import logging

# 수치 계산 및 데이터 처리
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

# 딥러닝 관련
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 이미지 처리
import nibabel as nib
import SimpleITK as sitk
import skimage.transform as skTrans
from skimage import exposure

# 시각화
import matplotlib.pyplot as plt
import seaborn as sns

# 머신러닝 도구
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from contextlib import contextmanager

# 2. 유틸리티 클래스/함수들을 한 곳에 모음
class ResourceManager:
    @staticmethod
    @contextmanager
    def manage_resources():
        try:
            yield
        finally:
            plt.close('all')
            torch.cuda.empty_cache()

# 3. 이미지 처리 관련 함수들을 한 곳에 모음
def standardize_intensity(image):
    """
    Standardize image intensity to [-1,1] range with robust normalization
    """
    # Handle empty or invalid images
    if image is None or image.size == 0:
        return np.zeros_like(image)
        
    # Make a copy to avoid modifying the original
    image = image.copy()
    
    # Remove outliers
    p2, p98 = np.percentile(image, (2, 98))
    image = np.clip(image, p2, p98)
    
    # Normalize to [0,1] first
    image_min = image.min()
    image_max = image.max()
    if image_max - image_min != 0:
        image = (image - image_min) / (image_max - image_min)
    
    # Then scale to [-1,1]
    image = (image * 2) - 1
    
    # Ensure the output is strictly within [-1,1]
    return np.clip(image, -1, 1)

def enhance_contrast(image):
    """Enhance image contrast"""
    return exposure.equalize_adapthist(image, clip_limit=0.03)

def register_to_template(moving_image, fixed_template):
    try:
        # SimpleITK 이미지로 변환 시 원본 방향 정보 유지
        moving_sitk = sitk.GetImageFromArray(moving_image.astype(np.float32))
        fixed_sitk = sitk.GetImageFromArray(fixed_template.astype(np.float32))
        
        # 중심 정렬을 위한 초기 변환
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_sitk, 
            moving_sitk,
            sitk.Euler3DTransform(),  # Affine 대신 Euler3D 사용
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        
        # 등록 방법 설정
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetInitialTransform(initial_transform)
        
        # 최적화 파라미터 조정
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=1.0,  # 학습률 증가
            numberOfIterations=1000,
            convergenceMinimumValue=1e-6
        )
        
        # 메트릭 설정 변경
        registration_method.SetMetricAsMeanSquares()
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.8)  # 샘플링 비율 증가
        
        final_transform = registration_method.Execute(fixed_sitk, moving_sitk)
        
        # 변환 적용
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_sitk)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(final_transform)
        
        registered_image = resampler.Execute(moving_sitk)
        return sitk.GetArrayFromImage(registered_image)
        
    except Exception as e:
        logging.error(f"Registration failed: {str(e)}")
        return moving_image
    
# 4. 데이터 관련 클래스
class CustomImageDataset(Dataset):
    def __init__(self, data_df, path_column='New_Path', label_column='labels'):
        self.data_df = data_df
        self.path_column = path_column
        self.label_column = label_column

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        img_path = self.data_df.iloc[idx][self.path_column]
        label = self.data_df.iloc[idx][self.label_column]
        label = torch.tensor(label, dtype=torch.int64)
        label = torch.nn.functional.one_hot(label, num_classes=3).float()
        
        # NIFTI 파일 로드
        nifti_img = nib.load(img_path)
        affine = nifti_img.affine # affine 행렬 확인
        img_data = nifti_img.get_fdata() # 이미지 데이터 가져오기
    
        # RAS+ 방향으로 이미지 재정렬 (Right, Anterior, Superior)
        if not np.allclose(affine, np.eye(4)):  # affine이 단위행렬이 아닌 경우
            # 방향 정보 확인
            orientation = nib.aff2axcodes(affine)
        
            # RAS+ 방향으로 변환이 필요한 경우
            if orientation != ('R', 'A', 'S'):
                # 축 순서 변경
                img_data = np.transpose(img_data, [0, 2, 1])
            
                # 필요한 축에 대해 뒤집기
                if orientation[0] != 'R':
                    img_data = np.flip(img_data, axis=0)
                if orientation[1] != 'A':
                    img_data = np.flip(img_data, axis=1)
                if orientation[2] != 'S':
                    img_data = np.flip(img_data, axis=2)

        # 크기 조정
        img_data = skTrans.resize(img_data, (64, 64, 64), order=1, preserve_range=True)
    
        # 정규화
        img_data = (img_data - img_data.mean()) / img_data.std()
    
        return torch.tensor(img_data, dtype=torch.float32).unsqueeze(0), label

# 5. 모델 관련 클래스들
class Custom3DCNN(nn.Module):
    def __init__(self):
        super(Custom3DCNN, self).__init__()
        self.gradients = {}
        self.activations = {}
        
        # 레이어 정의
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
        
        # Hook 등록
        self.conv3.register_forward_hook(self._save_activation('conv3'))
        
    def _save_activation(self, name):
        def hook(module, input, output):
            self.activations[name] = output
        return hook
        
    def _save_gradient(self, name):
        def hook(grad):
            self.gradients[name] = grad
        return hook

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
        
        if x.requires_grad:
            x.register_hook(self._save_gradient('conv3'))
        
        x = x.view(-1, 64*3*3*3)
        x = self.dropout1(self.fc1(x))
        x = self.dropout2(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)
    
    def get_activations_gradient(self):
        return self.gradients.get('conv3', None)
    
    def get_activations(self):
        return self.activations.get('conv3', None)

class GradCAM:
    def __init__(self, model):
        self.model = model
        
    def generate_cam(self, input_image, target_class=None):
        batch_size = input_image.size(0)
        cams = []
        
        for i in range(batch_size):
            single_input = input_image[i:i+1]
            output = self.model(single_input)
            
            if target_class is None:
                current_target = output.argmax(dim=1)
            else:
                current_target = target_class[i] if isinstance(target_class, torch.Tensor) else torch.tensor([target_class], device=device)
            
            self.model.zero_grad()
            target_output = output[0, current_target]
            target_output.backward(retain_graph=(i < batch_size-1))
            
            gradients = self.model.get_activations_gradient()
            activations = self.model.get_activations()
            
            if gradients is None or activations is None:
                continue
                
            weights = torch.mean(gradients, dim=(2, 3, 4))[0]
            cam = torch.zeros(activations.shape[2:], dtype=torch.float32).to(device)
            
            for j, w in enumerate(weights):
                cam += w * activations[0, j, :, :, :]
                
            cam = F.relu(cam)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-7)
            cams.append(cam.cpu().detach())
            
        return torch.stack(cams) if cams else None

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
    """Helper function for logging info messages"""
    logging.info(message)

# Define template path
TEMPLATE_PATH = "/home/simclr_project/simclr/SimCLR_RS/templates/aal116MNI_temp.nii"

# Verify template exists
if not os.path.exists(TEMPLATE_PATH):
    raise FileNotFoundError(f"Template not found at {TEMPLATE_PATH}")
    
log_info(f"Using template from: {TEMPLATE_PATH}")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_info(f"Using device: {device}")


def save_gradcam_visualization(model, image, label, save_path, template_path=None):
    try:
        with ResourceManager.manage_resources():
            with torch.amp.autocast('cuda'):
                gradcam = GradCAM(model)
                cam = gradcam.generate_cam(image.unsqueeze(0).to(device))
            
            if cam is None:
                logging.warning(f"Failed to generate GradCAM for {save_path}")
                return
            
        # GradCAM 생성 후 즉시 CPU로 이동
        cam = cam.cpu().numpy()
        
        # Resize CAM to match input size
        cam = torch.tensor(cam).to(device)
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(64, 64, 64),
            mode='trilinear',
            align_corners=False
        ).squeeze().cpu().numpy()
        
        # Process input image
        image_np = image.cpu().numpy()

        # AAL Template processing
        template = None
        if template_path and os.path.exists(template_path):
            try:
                aal_img = nib.load(template_path)
                template_data = aal_img.get_fdata()
            
                # RAS+ 방향으로 변환
                orientation = nib.aff2axcodes(aal_img.affine)
                if orientation != ('R', 'A', 'S'):
                    template_data = np.transpose(template_data, [0, 2, 1])
                    if orientation[0] != 'R':
                        template_data = np.flip(template_data, axis=0)
                    if orientation[1] != 'A':
                        template_data = np.flip(template_data, axis=1)
                    if orientation[2] != 'S':
                        template_data = np.flip(template_data, axis=2)
            
                # 크기 조정
                template_data = skTrans.resize(
                    template_data,
                    image_np.shape,
                    order=1,
                    preserve_range=True,
                    anti_aliasing=True,
                    mode='constant'
                )
                
                # Min-max normalization for visualization
                template = (template_data - template_data.min()) / (template_data.max() - template_data.min())
                
            except Exception as e:
                logging.error(f"AAL template processing failed: {str(e)}")
                template = None
        
        if template is None:
            template = standardize_intensity(image_np)
            
            
        # Create visualization
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))

        # 컬러맵 설정
        for ax in axes.flat:
            ax.set_aspect('equal')
        
        # Get middle slices
        mid_x = template.shape[0] // 2
        mid_y = template.shape[1] // 2
        mid_z = template.shape[2] // 2
        
        # views = [
        #     ('Sagittal', lambda x: np.rot90(x[mid_x, :, :], k=-1)),
        #     ('Coronal', lambda x: np.rot90(x[:, mid_y, :], k=-1)),
        #     ('Axial', lambda x: np.flipud(x[:, :, mid_z]))
        # ]

        # 수정된 코드
        views = [
            ('Coronal', lambda x: x[:, mid_y, :]),  # rot90 제거
            ('Axial', lambda x: x[:, :, mid_z]),    # flipud 제거
            ('Sagittal', lambda x: x[mid_x, :, :])  # rot90 제거
        ]

#         views = [
#             ('Coronal', lambda x: np.rot90(x[:, mid_y, :], k=1)),     # 90도 회전
#             ('Axial', lambda x: np.rot90(x[:, :, mid_z], k=1)),       # 90도 회전
#             ('Sagittal', lambda x: np.rot90(x[mid_x, :, :], k=1))     # 90도 회전
# ]

        for row, (view_name, slice_func) in enumerate(views):
            img_slice = slice_func(image_np)
            cam_slice = slice_func(cam)
            temp_slice = slice_func(template)
            
            # Adjust contrast for better visualization
            img_slice = exposure.equalize_adapthist(img_slice)
            
            # extent 설정 수정
            extent = [-img_slice.shape[1]/2, img_slice.shape[1]/2, 
                      -img_slice.shape[0]/2, img_slice.shape[0]/2]
            
            # Original image
            axes[row,0].imshow(img_slice, cmap='gray', extent=extent)
            axes[row,0].set_title(f'{view_name} - Original')
            axes[row,0].axis('off')
            
            # AAL template
            axes[row,1].imshow(temp_slice, cmap='gray', extent=extent)
            axes[row,1].set_title(f'{view_name} - AAL Template')
            axes[row,1].axis('off')
            
            # GradCAM
            axes[row,2].imshow(cam_slice, cmap='jet', extent=extent)
            axes[row,2].set_title(f'{view_name} - GradCAM')
            axes[row,2].axis('off')
            
            # Overlay
            axes[row,3].imshow(temp_slice, cmap='gray', extent=extent)
            axes[row,3].imshow(cam_slice, cmap='jet', alpha=0.7, extent=extent) # alpha값을 올렸음 overlay 3rd 칼럼 잘보이게 하려고
            axes[row,3].set_title(f'{view_name} - AAL + GradCAM')
            axes[row,3].axis('off')
        
        plt.suptitle(f'GradCAM Visualization with AAL Atlas (Class {label.argmax().item()})')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
        
    except Exception as e:
        logging.error(f"Error in saving GradCAM visualization: {str(e)}")
        plt.close('all')
        
    finally:
        plt.close('all')
        torch.cuda.empty_cache()

# 6. 학습/검증 관련 함수들
def train_loop(model, train_loader, optimizer, loss_fn, device):
    """Training loop function"""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.amp.autocast('cuda'):
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
   model.eval()
   val_loss = 0
   correct = 0
   total_samples = 0
   all_labels = []
   all_preds = []
   total_processed = 0

   os.makedirs(f'logs/gradcam_epoch_{epoch}', exist_ok=True)

   with ResourceManager.manage_resources():
       with torch.set_grad_enabled(True):
           with torch.amp.autocast('cuda'):
               for batch_idx, (img, label) in enumerate(dataloader):
                   try:
                       img, label = img.to(device), label.to(device)
                       output = model(img)
               
                       loss = criterion(output, label)
                       val_loss += loss.item()
               
                       pred = output.argmax(1)
                       target = label.argmax(1)
                       correct += (pred == target).type(torch.float).sum().item()
                       total_samples += img.size(0)
               
                       all_labels.extend(target.cpu().numpy())
                       all_preds.extend(pred.cpu().numpy())
               
                       total_processed += 1
                       
                       # 매 배치마다 GradCAM 생성
                       for i in range(min(3, len(img))):
                        save_gradcam_visualization(
                            model,
                            img[i], 
                            label[i],
                            f'logs/gradcam_epoch_{epoch}/gradcam_batch_{batch_idx}_img_{i}.png',
                            template_path=TEMPLATE_PATH
                        )

               
                    #    # GradCAM 생성 (일부 배치에 대해서만)
                    #    if batch_idx % 5 == 0: # 5번째 마다 GradCam 생성
                    #        for i in range(min(3, len(img))):  # 배치당 최대 3개 이미지만 처리
                    #            save_gradcam_visualization(
                    #                model,
                    #                img[i],
                    #                label[i],
                    #                f'logs/gradcam_epoch_{epoch}/gradcam_batch_{batch_idx}_img_{i}.png',
                    #                template_path=TEMPLATE_PATH
                    #           )
               
                       # 메모리 정리
                       if batch_idx % 10 == 0:
                           torch.cuda.empty_cache()
                           
                   except Exception as e:
                       logging.error(f"Error during validation batch {batch_idx}: {str(e)}")
                       continue

   # 결과 계산
   if total_processed > 0:
       val_loss /= total_processed
       accuracy = 100 * correct / total_samples
       
       # 로깅
       logging.info(f"Validation Epoch: {epoch}")
       logging.info(f"Average loss: {val_loss:.4f}")
       logging.info(f"Accuracy: {accuracy:.2f}%")
       
       # 혼동 행렬 생성 및 저장
       try:
           cm = confusion_matrix(all_labels, all_preds)
           plt.figure(figsize=(10, 8))
           sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
           plt.title(f'Confusion Matrix - Epoch {epoch}')
           plt.ylabel('True Label')
           plt.xlabel('Predicted Label')
           plt.savefig(f'logs/confusion_matrix_epoch_{epoch}.png')
           plt.close()
       except Exception as e:
           logging.error(f"Error creating confusion matrix: {str(e)}")
           
   else:
       val_loss = float('inf')
       accuracy = 0.0
       logging.error("No batches were successfully processed during validation")
   
   return val_loss, accuracy, all_labels, all_preds
#.7 메인함수
def main():
    """Main training function"""
    # Load and prepare data
    log_info("\nLoading data...")
    df = pd.read_csv('/home/simclr_project/simclr/SimCLR_RS/df_modified.csv')
    log_info(f"Loaded dataset with {len(df)} samples")

    
    path_column = 'New_Path'
    label_column = 'DX2'

    # Split data
    paths = df[path_column].tolist()
    labels = df[label_column]
    X_train, X_val, y_train, y_val = train_test_split(
        paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_df = pd.DataFrame({path_column: X_train, 'labels': y_train})
    val_df = pd.DataFrame({path_column: X_val, 'labels': y_val})

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
            # Clear memory at start of each epoch
            torch.cuda.empty_cache()
            epoch_start_time = datetime.now()
            log_info(f"\nEpoch {epoch+1}/{num_epochs}")
            log_info("-" * 20)
            
            try:
                # Training phase
                train_loss, train_acc = train_loop(model, train_loader, optimizer, criterion, device)
                torch.cuda.empty_cache()  # Clear after training

                # Validation phase with GradCAM
                val_loss, val_acc, labels, preds = validate(val_loader, model, criterion, device, epoch)
                torch.cuda.empty_cache()  # Clear after validation

                # Record metrics
                train_losses.append(train_loss)
                train_accuracies.append(train_acc)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
            
                epoch_duration = datetime.now() - epoch_start_time

                # Memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
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