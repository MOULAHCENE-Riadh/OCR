"""
cnn_ctc.py

PyTorch implementation of the CNN‑BLSTM‑CTC architecture for Arabic handwritten OCR.
"""

import torch
import torch.nn as nn

class CNNBLSTMCTC(nn.Module):
    """
    CNN-BLSTM-CTC model for sequence-to-sequence OCR.
    Args:
        num_classes (int): Number of output classes (including blank for CTC).
        img_height (int): Input image height.
    """
    def __init__(self, num_classes, img_height=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2,2),  # (B, 64, H/2, W/2)
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2,2), # (B, 128, H/4, W/4)
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2,1)), # (B, 256, H/8, W/4)
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d((2,1)), # (B, 512, H/16, W/4)
            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU() # (B, 512, H/16-1, W/4-1)
        )
        # Calculate feature size after CNN
        cnn_out_h = img_height // 16 - 1
        self.rnn = nn.LSTM(
            input_size=512 * cnn_out_h,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (B, 1, H, W)
        features = self.cnn(x)  # (B, C, H', W')
        b, c, h, w = features.size()
        features = features.permute(0, 3, 1, 2)  # (B, W', C, H')
        features = features.contiguous().view(b, w, c * h)  # (B, W', C*H')
        rnn_out, _ = self.rnn(features)  # (B, W', 2*hidden)
        out = self.fc(rnn_out)  # (B, W', num_classes)
        out = out.permute(1, 0, 2)  # (W', B, num_classes) for CTC loss
        return out 