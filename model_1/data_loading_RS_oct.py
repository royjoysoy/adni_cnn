import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import nibabel as nib
import skimage.transform as skTrans
from sklearn.model_selection import train_test_split
import torchio as tio
import numpy as np

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

        img = nib.load(img_path).get_fdata()
        img = skTrans.resize(img, (64, 64, 64), order=1, preserve_range=True)
        img = (img - img.min()) / (img.max() - img.min())
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

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
    return {
        'base': tio.Compose([
            tio.ToCanonical()
            #tio.RescaleIntensity(out_min_max=(0, 1)),
        ]),
        'flip': tio.RandomFlip(axes=(0, 1, 2)),
        'anisotropy': tio.RandomAnisotropy(downsampling=(1, 2.5)),
        'swap': tio.RandomSwap(patch_size=15, num_iterations=100),
        'elastic': tio.RandomElasticDeformation(num_control_points=7, max_displacement=7.5),
        'bias_field': tio.RandomBiasField(coefficients=0.5),
        'blur': tio.RandomBlur(std=(0, 4)),
        'gamma': tio.RandomGamma(log_gamma=(-0.3, 0.3)),
        'spike': tio.RandomSpike(num_spikes=1, intensity=(0.1, 1)),
        'ghost': tio.RandomGhosting(num_ghosts=2, intensity=(0.5, 1)),
        'noise': tio.RandomNoise(mean=0, std=(0, 0.25)),
        'motion': tio.RandomMotion(degrees=10, translation=10),
        'mixup': tio.Compose([]),  # Placeholder for mixup
        'cutmix': tio.Compose([])  # Placeholder for cutmix
    }
def get_data_loaders(train_df, val_df, test_df, batch_size, transforms):
    train_loaders = {}
    for aug_type, transform in transforms.items():
        train_dataset = ADNI_3_Class(data_df=train_df, transform=transform, aug_type=aug_type)
        train_loaders[aug_type] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Use 'base' transform for validation and test datasets
    val_dataset = ADNI_3_Class(data_df=val_df, transform=transforms['base'], aug_type='base')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = ADNI_3_Class(data_df=test_df, transform=transforms['base'], aug_type='base')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loaders, val_loader, test_loader


print("End of file")