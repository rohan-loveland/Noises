"""
Optimized DINOv3 Pretraining on Mel-Spectrograms for BirdCLEF
Uses timm DINOv3 ViT backbones + improved SSL recipe.
"""

import sys
import time
from pathlib import Path
import argparse
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import timm
import numpy as np
import pandas as pd
import random

from Dinov3PretrainDataset import Dinov3PretrainDataset
import torchvision.transforms as transforms   # Make sure this is imported

# ====================== CONFIG ======================
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 5e-5
SUBSET = 0.25
SEED = 42
MODEL_NAME = "vit_small_patch16_dinov3.lvd1689m"  # Best: change to this after timm update
DIM = 384
MOMENTUM = 0.996
CENTER_MOMENTUM = 0.9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

def get_spectrogram_transforms():
    """Improved spectrogram augmentations - RandomErasing AFTER ToTensor()"""
    global_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3)], p=0.8),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # SpecAugment-style masking - must be after ToTensor()
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.1, 2.0)),
    ])

    local_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.3, 0.9)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3)], p=0.8),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.4, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ])
    return global_transform, local_transform


def dino_loss(student_output, teacher_output, center, temperature=0.1, teacher_temp=0.04):
    """Improved DINO loss with centering."""
    student_out = student_output / temperature
    teacher_out = (teacher_output.detach() - center) / teacher_temp
    teacher_prob = torch.softmax(teacher_out, dim=1)
    loss = - (teacher_prob * torch.log_softmax(student_out, dim=1)).sum(dim=1).mean()
    return loss


def main():
    parser = argparse.ArgumentParser(description="Optimized DINOv3 Pretraining on Spectrograms")
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--subset', type=float, default=SUBSET)
    parser.add_argument('--model', type=str, default=MODEL_NAME)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    print("=== Optimized DINOv3 Pretraining on Spectrograms ===")
    print(f"Model: {args.model}")

    csv_path = "5sSpectrograms_tensors/train_5s_spectrograms.csv"
    tensor_dir = "5sSpectrograms_tensors"
    
    df = pd.read_csv(csv_path)
    if args.subset < 1.0:
        group_col = 'class_name' if 'class_name' in df.columns else 'primary_label'
        def stratified_sample(g):
            n = max(5, int(len(g) * args.subset))
            return g.sample(n=n, random_state=SEED) if len(g) >= n else g
        df = df.groupby(group_col, group_keys=False).apply(stratified_sample).reset_index(drop=True)

    print(f"Pretraining on {len(df):,} samples")

    global_tf, local_tf = get_spectrogram_transforms()
    dataset = Dinov3PretrainDataset(df, tensor_dir, global_transform=global_tf, local_transform=local_tf)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True)

    # Models
    student = timm.create_model(args.model, pretrained=True, num_classes=0).to(device)
    teacher = timm.create_model(args.model, pretrained=True, num_classes=0).to(device)
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.04)
    scaler = GradScaler()

    center = torch.zeros(1, DIM, device=device)
    start_epoch = 0

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location='cpu')
        student.load_state_dict(ckpt['student_state_dict'])
        teacher.load_state_dict(ckpt.get('teacher_state_dict', student.state_dict()))
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        center = ckpt.get('center', center)
        start_epoch = ckpt.get('epoch', 0)
        print(f"Resumed from epoch {start_epoch}")

    for param in teacher.parameters():
        param.requires_grad = False

    print("Starting training...")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        student.train()
        teacher.eval()
        running_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for views in pbar:
            global_view, local_view = views[0].to(device), views[1].to(device)

            optimizer.zero_grad()
            with autocast(device_type=device.type):
                s_global = student(global_view)
                s_local = student(local_view)
                with torch.no_grad():
                    t_global = teacher(global_view)

                loss = (dino_loss(s_global, t_global, center) + 
                        dino_loss(s_local, t_global, center)) / 2

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Momentum + center update
            with torch.no_grad():
                for ps, pt in zip(student.parameters(), teacher.parameters()):
                    pt.data = MOMENTUM * pt.data + (1 - MOMENTUM) * ps.data
                center = CENTER_MOMENTUM * center + (1 - CENTER_MOMENTUM) * t_global.mean(dim=0, keepdim=True)

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (pbar.n + 1))

        print(f"Epoch {epoch+1} completed. Avg Loss: {running_loss / len(loader):.4f}")

        checkpoint = {
            'student_state_dict': student.state_dict(),
            'teacher_state_dict': teacher.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'center': center,
            'epoch': epoch + 1,
        }
        if (epoch + 1) % 2 == 0 or epoch == args.epochs - 1:
            torch.save(checkpoint, f'dinov3_pretrained2_epoch{epoch+1}.pth')

    torch.save(checkpoint, 'dinov3_pretrained_final.pth')
    print(f"Pretraining finished in {(time.time() - start_time)/60:.1f} minutes")


if __name__ == "__main__":
    main()