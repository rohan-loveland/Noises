#!/usr/bin/env python3
"""
MLBird.py - Low-RAM Version (16GB)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys
from tqdm import tqdm
from collections import Counter

# ====================== CONFIG ======================
DATA_CSV = Path("5sSpectrograms_tensors/train_5s_spectrograms.csv")
TENSOR_DIR = Path("5sSpectrograms_tensors")
BATCH_SIZE = 96                    # Good compromise for 4070 Laptop
NUM_WORKERS = 6                    # Reduced to avoid too much RAM pressure
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
TEST_SPLIT = 0.2
SEED = 42

# ====================== DATASET ======================
class SpectrogramDataset(Dataset):
    def __init__(self, df, tensor_dir):
        self.df = df.reset_index(drop=True)
        self.tensor_dir = Path(tensor_dir)
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.df['class_name'])
        self.num_classes = len(self.label_encoder.classes_)

        print(f"Loaded {len(self.df):,} samples, {self.num_classes} classes")
        print("Class distribution:", dict(Counter(self.df['class_name'].values)))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npy_path = self.tensor_dir / row['spectrogram_npy_path']
        
        # Memory-mapped loading = much faster + very low RAM usage
        spec = np.load(npy_path, mmap_mode='r').astype(np.float32)
        spec = torch.from_numpy(spec).unsqueeze(0)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return spec, label


# ====================== MODEL (unchanged) ======================
class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ====================== MAIN ======================
def main():
    if not DATA_CSV.exists():
        print(f"Error: {DATA_CSV} not found.")
        sys.exit(1)

    print("Loading metadata...")
    df = pd.read_csv(DATA_CSV)[['class_name', 'spectrogram_npy_path']].dropna()
    print(f"Total samples: {len(df):,}")

    train_df, test_df = train_test_split(
        df, test_size=TEST_SPLIT, stratify=df['class_name'], random_state=SEED
    )
    print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

    train_dataset = SpectrogramDataset(train_df, TENSOR_DIR)
    test_dataset = SpectrogramDataset(test_df, TENSOR_DIR)
    num_classes = train_dataset.num_classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    pin_memory = True
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=3,          # Lower for low RAM
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        prefetch_factor=3
    )

    model = AudioCNN(num_classes).to(device)
    class_counts = np.bincount(train_dataset.labels)
    class_weights = torch.tensor(1.0 / (class_counts + 1e-6), dtype=torch.float32).to(device)
    class_weights = class_weights / class_weights.sum()

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = GradScaler(device=device.type, enabled=device.type == 'cuda')

    best_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = correct = total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for specs, labels in pbar:
            specs, labels = specs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(device_type=device.type):
                outputs = model(specs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(loss=running_loss/(total/BATCH_SIZE), acc=100.*correct/total)

        # Validation...
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for specs, labels in test_loader:
                specs, labels = specs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(specs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        print(f"Epoch {epoch+1} - Val Accuracy: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'label_encoder': train_dataset.label_encoder,
                'num_classes': num_classes,
                'val_acc': val_acc
            }, 'mlbird_best_model.pth')
            print("  → Saved best model")

    print("\n✅ Training complete! Best validation accuracy:", best_acc)


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    main()