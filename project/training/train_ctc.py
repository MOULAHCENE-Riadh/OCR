"""
train_ctc.py

Training loop for CNN‑BLSTM‑CTC model, including CTC collate function and logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from project.models.cnn_ctc import CNNBLSTMCTC
from project.data.loaders import get_dataloader
import numpy as np
from collections import Counter

# --- Character Vocabulary Utilities ---
class ArabicVocab:
    """
    Character vocabulary for Arabic OCR, with CTC blank.
    """
    def __init__(self, texts):
        chars = set(''.join(texts))
        self.chars = sorted(list(chars))
        self.blank = '<BLANK>'
        self.idx2char = [self.blank] + self.chars
        self.char2idx = {c: i+1 for i, c in enumerate(self.chars)}
        self.char2idx[self.blank] = 0

    def encode(self, text):
        return [self.char2idx.get(c, 0) for c in text]

    def decode(self, indices):
        return ''.join([self.idx2char[i] for i in indices if i != 0])

    @property
    def num_classes(self):
        return len(self.idx2char)

# --- Collate Function for CTC ---
def ctc_collate(batch, vocab):
    images, texts = zip(*batch)
    images = torch.stack(images)
    targets = [torch.tensor(vocab.encode(t), dtype=torch.long) for t in texts]
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets = torch.cat(targets)
    return images, targets, target_lengths, texts

# --- Training Loop ---
def train_ctc(
    epochs=10,
    batch_size=16,
    lr=1e-3,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    dataset_names=['khatt'],
    master_csv='data/combined_labels.csv',
    img_height=32,
    img_width=128
):
    # Build vocab from all texts
    import pandas as pd
    df = pd.read_csv(master_csv)
    texts = df[df['dataset'].isin(dataset_names)]['text'].tolist()
    vocab = ArabicVocab(texts)

    # DataLoader
    dataloader = get_dataloader(dataset_names, batch_size, shuffle=True, train=True, master_csv=master_csv)
    model = CNNBLSTMCTC(num_classes=vocab.num_classes, img_height=img_height).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            images, texts = batch
            images = images.to(device)
            # Encode targets
            targets = [torch.tensor(vocab.encode(t), dtype=torch.long) for t in texts]
            target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
            targets_concat = torch.cat(targets).to(device)
            # Model forward
            logits = model(images)  # (T, B, C)
            input_lengths = torch.full(size=(logits.size(1),), fill_value=logits.size(0), dtype=torch.long).to(device)
            # CTC Loss
            loss = criterion(logits, targets_concat, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    print("Training complete.")
    return model, vocab 