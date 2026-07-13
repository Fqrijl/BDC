"""Dataset + transforms untuk BDC2026.

File ini di-generate otomatis oleh cell %%writefile di notebook.
Dipisah ke .py karena Windows + num_workers>0 pakai spawn: worker process
perlu meng-import class Dataset dari modul yang bisa di-import (class yang
didefinisikan langsung di notebook tidak bisa di-pickle ke worker).
"""
from __future__ import annotations

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class TrashDataset(Dataset):
    def __init__(self, df, transform=None, fallback_size: int = 224):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.fallback_size = fallback_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row["path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.fallback_size, self.fallback_size))
        if self.transform:
            img = self.transform(img)
        return img, int(row["label"]), row["path"]


def build_transforms(mean, std, img_size: int, train: bool):
    resize_to = int(img_size * 1.14)
    if train:
        return T.Compose([
            T.Resize((resize_to, resize_to)),
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
