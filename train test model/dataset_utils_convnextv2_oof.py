"""Dataset untuk notebook OOF Collection ConvNeXt V2-Base -- ditulis ke file eksternal
supaya NUM_WORKERS>0 aman di Windows (worker spawn butuh class yang importable,
bukan didefinisikan di cell notebook)."""
from __future__ import annotations

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F_tf


def safe_open(path, img_size):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (img_size, img_size))


class TrainValDataset(Dataset):
    def __init__(self, dataframe, label_mapping, img_size, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.label_mapping = label_mapping
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        image = safe_open(row["filepath"], self.img_size)
        image_np = np.array(image)
        image_tensor = self.transform(image=image_np)["image"] if self.transform else F_tf.to_tensor(image)
        return image_tensor, self.label_mapping[row["label"]]


class IndexedDataset(Dataset):
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        return image, label, self.indices[idx]


class TestDataset(Dataset):
    def __init__(self, dataframe, img_size, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        image = safe_open(row["filepath"], self.img_size)
        image_np = np.array(image)
        image_tensor = self.transform(image=image_np)["image"] if self.transform else F_tf.to_tensor(image)
        return image_tensor, row["id"]
