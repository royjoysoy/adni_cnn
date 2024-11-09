import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import nibabel as nib
import skimage.transform as skTrans
from sklearn.model_selection import train_test_split
import torchio as tio
import numpy as np
import logging

print("Start of file")
print("Importing torch")
print("Importing Dataset")
print("Dataset imported successfully")

print("Defining device")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("About to define ADNI_3_Class")

class ADNI_3_Class(Dataset):
    def __init__(self, data_df, transform=None, target_transform=None, aug_type='base'):
        self.data_df = data_df
        self.transform = transform
        self.target_transform = target_transform
        self.targets = data_df['DX2'].tolist()
        self.aug_type = aug_type

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        img_path = self.data_df.iloc[idx]['New_Path']
        target = self.data_df.iloc[idx]['DX2']
        target = torch.tensor(target, dtype=torch.long)

        try:
            img = nib.load(img_path).get_fdata()
            img = skTrans.resize(img, (64, 64, 64), order=1, preserve_range=True)
            img = (img - img.min()) / (img.max() - img.min())
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {str(e)}")
            # Return a zero tensor of correct shape if image loading fails
            img = torch.zeros(1, 64, 64, 64, dtype=torch.float32)

        if self.transform:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)
        else:
            pos_1 = pos_2 = img

        if self.target_transform:
            target = self.target_transform(target)

        if self.aug_type in ['mixup', 'cutmix']:
            return pos_1, pos_2, target, idx
        else:
            return pos_1, pos_2, target

# Custom mean and std for normalization - determined not to use them
# custom_mean = 17.806
# custom_std = 34.073

def get_mr_transforms():
    # Define strong augmentations with optimized parameters
    mr_transforms = {
        'flip': tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.8),  # Increased probability
        'anisotropy': tio.RandomAnisotropy(downsampling=(1, 3.0)),  # Increased range
        'swap': tio.RandomSwap(patch_size=20, num_iterations=150),  # Increased patch size
        'elastic': tio.RandomElasticDeformation(
            num_control_points=8,
            max_displacement=8.0,
            locked_borders=2
        ),
        'bias_field': tio.RandomBiasField(coefficients=0.6),  # Increased coefficient
        'blur': tio.RandomBlur(std=(0, 5)),  # Increased blur range
        'gamma': tio.RandomGamma(log_gamma=(-0.4, 0.4)),  # Increased range
        'spike': tio.RandomSpike(num_spikes=2, intensity=(0.1, 1.2)),  # Increased intensity
        'ghost': tio.RandomGhosting(num_ghosts=3, intensity=(0.5, 1.2)),  # Increased ghosts
        'noise': tio.RandomNoise(mean=0, std=(0, 0.3)),  # Increased noise
        'motion': tio.RandomMotion(degrees=15, translation=12),  # Increased motion
        'mixup': tio.Compose([]),
        'cutmix': tio.Compose([])
    }

    # Define combined transformations with strong augmentation chains
    combined_transforms = {
        # Combination 1: Geometric + Intensity (Strong)
        'geometric_intensity': tio.Compose([
            tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.8),
            tio.RandomAffine(
                scales=(0.9, 1.1),
                degrees=15,
                translation=10,
                isotropic=True,
                p=0.9
            ),
            tio.RandomElasticDeformation(
                num_control_points=8,
                max_displacement=8.0,
                locked_borders=2
            ),
            tio.RandomGamma(log_gamma=(-0.4, 0.4)),
            tio.RandomBlur(std=(0, 4)),
            tio.RandomBiasField(coefficients=0.5)
        ]),
        
        # Combination 2: Noise + Artifact (Strong)
        'noise_artifact': tio.Compose([
            tio.RandomNoise(mean=0, std=(0, 0.3)),
            tio.RandomSpike(num_spikes=2, intensity=(0.1, 1.2)),
            tio.RandomGhosting(num_ghosts=3, intensity=(0.5, 1.2)),
            tio.RandomMotion(degrees=15, translation=12),
            tio.RandomBlur(std=(0, 3)),
            tio.RandomAnisotropy(downsampling=(1, 2.5))
        ]),

        # Combination 3: Clinical + Acquisition (New)
        'clinical_acquisition': tio.Compose([
            tio.RandomAffine(
                scales=(0.95, 1.05),
                degrees=10,
                translation=8,
                isotropic=False
            ),
            tio.RandomBiasField(coefficients=0.4),
            tio.RandomNoise(mean=0, std=(0, 0.2)),
            tio.RandomGhosting(num_ghosts=2, intensity=(0.3, 0.8)),
            tio.RandomBlur(std=(0, 2)),
            tio.RandomGamma(log_gamma=(-0.3, 0.3))
        ])
    }

    # Merge transforms dictionaries
    mr_transforms.update(combined_transforms)

    # Base transform with optimized parameters
    base_transform = tio.Compose([
        tio.ToCanonical(),
        tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
        tio.RandomAffine(
            scales=(0.95, 1.05),
            degrees=10,
            translation=5,
            isotropic=True,
            p=0.8
        ),
        tio.RescaleIntensity(
            out_min_max=(0, 1),
            percentiles=(0.5, 99.5),  # Adjusted percentiles
            masking_method="threshold"
        ),
        tio.RandomNoise(
            mean=0,
            std=0.015,
            p=0.3
        ),
    ])

    transforms = {'base': base_transform}
    transforms.update(mr_transforms)
    
    return transforms

def get_data_loaders(train_df, val_df, test_df, batch_size, transforms):
    train_loaders = {}
    for aug_type, transform in transforms.items():
        train_dataset = ADNI_3_Class(data_df=train_df, transform=transform, aug_type=aug_type)
        train_loaders[aug_type] = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=True,
            pin_memory=True,  
            num_workers=8     
        )

    # Use base transform for validation and test (no augmentation needed)
    val_dataset = ADNI_3_Class(
        data_df=val_df, 
        transform=tio.Compose([
            tio.ToCanonical(),
            tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(1, 99))
        ]), 
        aug_type='base'
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )

    test_dataset = ADNI_3_Class(
        data_df=test_df, 
        transform=tio.Compose([
            tio.ToCanonical(),
            tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(1, 99))
        ]), 
        aug_type='base'
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )

    return train_loaders, val_loader, test_loader


print("End of file")