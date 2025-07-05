"""
preprocess.py

Image preprocessing utilities: resize, grayscale, normalization, and augmentations for OCR datasets.
"""

import cv2
import numpy as np
from torchvision import transforms
import albumentations as A

def to_grayscale(x, **_):
    if len(x.shape) == 2:
        return x
    return (0.299 * x[...,0] + 0.587 * x[...,1] + 0.114 * x[...,2]).astype('uint8')
def get_transforms(train: bool, height: int = 32, width: int = 128):
    """
    Returns a composed transform for OCR images.
    - Converts to grayscale
    - Resizes to fixed height, preserves aspect ratio, pads to fixed width
    - Normalizes pixel values to [0,1]
    - Applies augmentations if train=True (random rotation, skew, brightness/contrast)
    """
    # cv2.COLOR_BGR2GRAY = 6, cv2.INTER_LINEAR = 1
    aug_list = [
       A.Lambda(image=to_grayscale),
        A.Resize(height, width, interpolation=1),
        A.PadIfNeeded(min_height=height, min_width=width, border_mode=0),
        A.Normalize(mean=0.5, std=0.5, max_pixel_value=255.0),
    ]
    if train:
        aug_list += [
            A.Rotate(limit=5, border_mode=0, p=0.5),
            A.Perspective(scale=(0.01, 0.05), p=0.3),
            A.RandomBrightnessContrast(p=0.3),
        ]
    return A.Compose(aug_list)

# TODO: Implement image transformation and augmentation functions. 