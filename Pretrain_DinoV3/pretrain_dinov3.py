"""
DINOv3 Pretraining on Mel-Spectrograms
Self-supervised ViT (DINO-style) pretraining on our bird/insect/noise dataset.
Uses existing .npy files (no regeneration). Outputs custom checkpoint for Dinov3DataStream.
Isolated in Pretrain_DinoV3/ per plan.
"""
import sys
import time
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import timm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "MLBird"))  # for SpectrogramDataset patterns

# from MLBird import SpectrogramDataset  # Base patterns reused in Dinov3PretrainDataset
from Dinov3PretrainDataset import Dinov3PretrainDataset

# ====================== CONFIG ======================
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 5e-5  # Lower LR for stability on diverse samples
SUBSET = 0.25  # Larger subset with stratification + oversampling rares to counter 98% Aves skew (ensures samples of every class in pretrain)
SEED = 42
MODEL_NAME = "vit_small_patch16_224"  # timm ViT for DINO
DIM = 384  # embedding dim
MOMENTUM = 0.996  # teacher momentum

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_dino_transforms():
    """Spectrogram-specific augmentations for DINO (2 views: global + local)."""
    # Global view (full spectrogram)
    global_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4)], p=0.8),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats for 3-channel
    ])
    # Local view (smaller crop for multi-crop DINO)
    local_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 0.8)),  # Match ViT input size (96 caused assertion error)
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4)], p=0.8),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return global_transform, local_transform

def dino_loss(student_output, teacher_output, temperature=0.1):
    """Simplified DINO loss (cross-entropy on softened teacher predictions)."""
    student_out = student_output / temperature
    teacher_out = teacher_output.detach() / 0.04  # sharper teacher
    teacher_prob = torch.softmax(teacher_out, dim=1)
    loss = - (teacher_prob * torch.log_softmax(student_out, dim=1)).sum(dim=1).mean()
    return loss

def main():
    parser = argparse.ArgumentParser(description="DINOv3 Pretraining on Spectrograms")
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--subset', type=float, default=SUBSET)
    parser.add_argument('--model', type=str, default=MODEL_NAME)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from (e.g. dinov3_pretrained_epoch3.pth)')
    args = parser.parse_args()

    print("=== DINOv3 Pretraining ===")
    print(f"Dataset: 5sSpectrograms_tensors/ (Mel-spectrograms treated as images)")
    print(f"Model: {args.model} (ViT for self-supervised learning)")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, Subset: {args.subset}")
    if args.resume:
        print(f"Resume from: {args.resume}")
    print()

    # Reuse CSV and base dataset logic from MLBird/SpectrogramDataStream
    csv_path = "5sSpectrograms_tensors/train_5s_spectrograms.csv"
    tensor_dir = "5sSpectrograms_tensors"
    
    # Load full df for unlabeled pretraining; use stratified sampling to ensure all classes (Aves dominant + rares like Reptilia/Insecta) are represented
    import pandas as pd
    df = pd.read_csv(csv_path)
    if args.subset < 1.0:
        # Stratified sample with min samples per class to counter extreme Aves skew (98%+ of data). Ensures at least some samples of rares (Insecta, Amphibia, Reptilia, Mammalia).
        group_col = 'class_name' if 'class_name' in df.columns else 'primary_label'
        def stratified_sample(group):
            if len(group) == 0:
                return group
            n = max(5, int(len(group) * args.subset))  # min 5 samples per class
            return group.sample(n=n, random_state=SEED) if len(group) >= n else group
        df = df.groupby(group_col, group_keys=False).apply(stratified_sample).reset_index(drop=True)
    print(f"Pretraining on {len(df):,} spectrograms (stratified across classes)")
    group_col = 'class_name' if 'class_name' in df.columns else 'primary_label'
    print("Class distribution:", df[group_col].value_counts().to_dict())

    # Create SSL dataset with 2 views (global + local crops)
    global_tf, local_tf = get_dino_transforms()
    dataset = Dinov3PretrainDataset(df, tensor_dir, global_transform=global_tf, local_transform=local_tf)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    start_epoch = 0
    # Student and Teacher ViT (timm)
    student = timm.create_model(args.model, pretrained=False, num_classes=DIM).to(device)
    teacher = timm.create_model(args.model, pretrained=False, num_classes=DIM).to(device)
    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.04)
    scaler = GradScaler()

    if args.resume and Path(args.resume).exists():
        print(f"Loading checkpoint for resume: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        student.load_state_dict(checkpoint['student_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resuming from epoch {start_epoch} (loss ~{checkpoint.get('loss', 'N/A'):.4f})")
        # Sync teacher from restored student
        teacher.load_state_dict(student.state_dict())
    else:
        teacher.load_state_dict(student.state_dict())  # init same

    for param in teacher.parameters():
        param.requires_grad = False

    criterion = dino_loss

    print("Starting DINO pretraining...")
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        student.train()
        teacher.eval()
        running_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for views in pbar:  # views = (global_view, local_view)
            global_view, local_view = views[0].to(device), views[1].to(device)
            
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                # Student on both views
                student_global = student(global_view)
                student_local = student(local_view)
                # Teacher on global only (standard DINO)
                with torch.no_grad():
                    teacher_global = teacher(global_view)
                
                loss = criterion(student_global, teacher_global) + criterion(student_local, teacher_global)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Momentum update for teacher
            with torch.no_grad():
                for param_s, param_t in zip(student.parameters(), teacher.parameters()):
                    param_t.data = MOMENTUM * param_t.data + (1 - MOMENTUM) * param_s.data
            
            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (pbar.n + 1))
        
        print(f"Epoch {epoch+1} completed. Avg Loss: {running_loss / len(loader):.4f}")
        
        # Save checkpoint (enhanced for resume: includes optimizer/scaler)
        checkpoint_dict = {
            'student_state_dict': student.state_dict(),
            'teacher_state_dict': teacher.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'epoch': epoch + 1,
            'loss': running_loss / len(loader),
            'model_name': args.model,
            'dim': DIM
        }
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:  # More frequent for resume convenience
            ckpt_path = f'dinov3_pretrained_epoch{epoch+1}.pth'
            torch.save(checkpoint_dict, ckpt_path)
            print(f"  → Saved checkpoint {ckpt_path}")
    
    total_time = time.time() - start_time
    print(f"\nPretraining complete! Total time: {total_time/60:.1f} min")
    # Save final model (full state for resume)
    final_path = 'dinov3_pretrained.pth'
    torch.save(checkpoint_dict, final_path)
    print(f"Final model saved as {final_path} (student weights + full state, ready for Dinov3DataStream and resume).")

if __name__ == "__main__":
    main()
