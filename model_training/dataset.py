"""
PyTorch Dataset for APTOS 2019 Diabetic Retinopathy Detection.

Expected folder layout
----------------------
data/
├── train.csv            # columns: id_code (str), diagnosis (int 0-4)
└── train_images/
    ├── 000c1434d8d7.png
    └── ...
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class RetinalDataset(Dataset):
    def __init__(self, df, img_dir: str, transform=None):
        """
        Args:
            df        : pandas DataFrame with columns 'id_code' and 'diagnosis'
            img_dir   : path to folder containing .png retinal images
            transform : torchvision transform pipeline
        """
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        img_name  = self.df.loc[idx, "id_code"] + ".png"
        img_path  = os.path.join(self.img_dir, img_name)
        image     = Image.open(img_path).convert("RGB")
        label     = int(self.df.loc[idx, "diagnosis"])

        if self.transform:
            image = self.transform(image)

        return image, label


# ─── Transform Factories ──────────────────────────────────────────────────────

_MEAN = [0.485, 0.456, 0.406]   # ImageNet stats (good default for medical images)
_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Augmentation pipeline for training.
    Geometric + colour jitter helps generalise on small medical datasets.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05,
        ),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Deterministic pipeline for validation / inference.
    No augmentation — just resize, tensor, normalise.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])
