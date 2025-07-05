"""
loaders.py

Functions for loading datasets (KHATT, IFN/ENIT, AHW, custom) from CSV/JSON files and reading configuration from config.yaml.
Also provides unified preprocessing, master CSV creation, and PyTorch DataLoader utilities.
"""

import os
import pandas as pd
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from typing import List
from .preprocess import get_transforms

# Sample function to load dataset from CSV
def load_dataset_from_csv(csv_path):
    """Load images and labels from a CSV file."""
    df = pd.read_csv(csv_path)
    # TODO: Adapt this to your CSV format (image_path, label, etc.)
    images = df['image_path'].tolist()
    labels = df['label'].tolist()
    return images, labels

# Sample function to read config.yaml
def read_config(config_path='config.yaml'):
    """Read configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# TODO: Implement load_dataset_from_json and support for multiple dataset formats.

def build_master_csv(config='../project/data/config.yaml', output_csv='../data/combined_labels.csv'):
    """
    Walks through each dataset folder, loads image paths and texts, and saves a master CSV with columns [dataset, image_path, text].
    Assumes each dataset has a labels.csv with columns [image_path, text].
    """
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)
    rows = []
    for dataset, path in cfg['datasets'].items():
        label_path = os.path.join(path, 'labels.csv')
        if not os.path.exists(label_path):
            print(f"Warning: {label_path} not found, skipping {dataset}.")
            continue
        df = pd.read_csv(label_path)
        for _, row in df.iterrows():
            rows.append({'dataset': dataset, 'image_path': os.path.abspath(os.path.join(path, str(row['image_path']))), 'text': row['text']})
    master_df = pd.DataFrame(rows)
    master_df.to_csv(output_csv, index=False)
    print(f"Master CSV saved to {output_csv} with {len(master_df)} samples.")


class OCRDataset(Dataset):
    """
    PyTorch Dataset for OCR images and texts, with preprocessing transforms.
    """
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row['image_path'])  # type: ignore
        if img is None:
            raise FileNotFoundError(f"Image not found: {row['image_path']}")
        if self.transform:
            img = self.transform(image=img)['image']
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        text = row['text']
        return img, text


def get_dataloader(dataset_names: List[str], batch_size: int, shuffle: bool, train: bool = True, master_csv='../project/data/combined_labels.csv'):
    """
    Filters the master CSV by dataset names, wraps in OCRDataset, applies transforms, and returns a DataLoader.
    """
    num_workers = 0
    df = pd.read_csv(master_csv)
    df = df[df['dataset'].isin(dataset_names)].reset_index(drop=True)
    transform = get_transforms(train)
    dataset = OCRDataset(df, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2) 