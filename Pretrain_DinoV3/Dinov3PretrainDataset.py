import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Dinov3PretrainDataset(Dataset):
    """
    Unlabeled dataset for DINO pretraining.
    Returns two augmented views of each spectrogram (global + local crop).
    Reuses MLBird.py SpectrogramDataset loading logic (mmap .npy).
    No labels needed for self-supervised learning.
    """
    def __init__(self, df, tensor_dir, global_transform=None, local_transform=None):
        self.df = df.reset_index(drop=True)
        self.tensor_dir = Path(tensor_dir)
        self.global_transform = global_transform
        self.local_transform = local_transform
        
        # Optional label encoder (for potential downstream eval, not used in SSL)
        self.label_encoder = LabelEncoder()
        if 'class_name' in self.df.columns:
            self.labels = self.label_encoder.fit_transform(self.df['class_name'])
        else:
            self.labels = None
        self.num_classes = len(self.label_encoder.classes_) if self.labels is not None else 0
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npy_path = self.tensor_dir / row['spectrogram_npy_path']
        
        # Reuse low-RAM mmap loading from MLBird.py:50 and Dinov3DataStream
        spec = np.load(npy_path, mmap_mode='r').astype(np.float32)
        
        # Convert to PIL Image (grayscale spectrogram as image, per Dinov3DataStream)
        if spec.ndim == 2:
            spec_norm = ((spec - spec.min()) / (spec.max() - spec.min() + 1e-8) * 255).clip(0, 255).astype(np.uint8)
            image = Image.fromarray(spec_norm).convert('RGB')  # Repeat to 3 channels for ViT
        else:
            image = Image.fromarray(spec.squeeze().astype(np.uint8)).convert('RGB')
        
        # Return two views for DINO (global and local crop)
        if self.global_transform and self.local_transform:
            view1 = self.global_transform(image)
            view2 = self.local_transform(image)
            return view1, view2
        else:
            # Fallback
            return torch.from_numpy(spec).unsqueeze(0), torch.from_numpy(spec).unsqueeze(0)
    
    @property
    def num_classes(self):
        return getattr(self, '_num_classes', 0)
    
    @num_classes.setter
    def num_classes(self, value):
        self._num_classes = value
