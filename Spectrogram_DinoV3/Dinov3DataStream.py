import numpy as np
import torch
from pathlib import Path
import pandas as pd
from collections import defaultdict
import json
from datetime import datetime
from PIL import Image
import torchvision.transforms as transforms

# Try to import HF transformers for DinoV2; graceful fallback
try:
    from transformers import AutoImageProcessor, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. Install with: pip install transformers torchvision")


class Dinov3DataStream:
    """
    DinoV2 (DinoV3 alias) embedding stream for A_RED.
    Loads existing .npy Mel-spectrograms, converts to image, extracts 384-dim semantic embedding.
    Completely compatible with existing 5sSpectrograms_tensors/*.npy + CSV.
    No audio regeneration needed. Reuses discovery tracker.
    """
    def __init__(self, csv_path: str = "5sSpectrograms_tensors/train_5s_spectrograms.csv", 
                 tensor_dir: str = "5sSpectrograms_tensors", max_samples=None, shuffle=True, seed=42,
                 dino_model_name: str = "facebook/dinov2-small"):
        self.csv_path = Path(csv_path)
        self.tensor_dir = Path(tensor_dir)
        self.max_samples = max_samples
        self.stream_counter = 0
        self.dino_model_name = dino_model_name
        self.embed_dim = 384  # small model
        
        # Hidden discovery tracker (for testing/reference only - not visible to ARED algorithm)
        self.discovery_tracker = DiscoveryTracker()
        
        # Load metadata (reuse exact logic from SpectrogramDataStream)
        self.df = pd.read_csv(self.csv_path)
        if 'spectrogram_npy_path' not in self.df.columns:
            raise ValueError("CSV must contain 'spectrogram_npy_path' column")
        
        if shuffle:
            self.df = self.df.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        if max_samples is not None:
            self.df = self.df.head(max_samples).reset_index(drop=True)
        
        self.n_samples = len(self.df)
        print(f"Loaded {self.n_samples:,} spectrogram samples for DinoV2 streaming")
        print(f"Example path: {self.df.iloc[0]['spectrogram_npy_path']}")
        print(f"DinoV2 model: {dino_model_name} (embed dim={self.embed_dim})")
        
        # Load DinoV2 model + processor (one-time, cached)
        if TRANSFORMERS_AVAILABLE:
            self.processor = AutoImageProcessor.from_pretrained(dino_model_name)
            self.model = AutoModel.from_pretrained(dino_model_name)
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.to('cuda')
            print("DinoV2 model loaded successfully (semantic embeddings enabled)")
        else:
            self.processor = None
            self.model = None
            print("WARNING: Falling back to mean-pooled spectrogram (no DinoV2)")
        
        # Image transform for spectrogram-as-image
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),  # grayscale normalized
        ])
    
    def stream_new_data_point(self):
        """Load .npy → DinoV2 embedding (384-dim vector). Reuses tracker."""
        if self.stream_counter >= self.n_samples:
            raise StopIteration("No more data points")
        
        row = self.df.iloc[self.stream_counter]
        npy_rel_path = row['spectrogram_npy_path']
        npy_path = self.tensor_dir / npy_rel_path
        
        if not npy_path.exists():
            print(f"  WARNING: File not found - {npy_path}")
            self.stream_counter += 1
            if self.stream_counter >= self.n_samples:
                raise StopIteration("No more data points")
            return self.stream_new_data_point()
        
        # Record for hidden discovery tracking BEFORE algorithm sees the point
        true_label = self.get_true_label_for_idx(self.stream_counter)
        self.discovery_tracker.record_examined_point(true_label, self.stream_counter)
        
        # Load spectrogram (reuse from original)
        spec = np.load(npy_path).astype(np.float32)
        
        # Convert to image for DinoV2 (grayscale spectrogram)
        if spec.ndim == 2:
            # Normalize to [0, 255] for PIL
            spec_norm = ((spec - spec.min()) / (spec.max() - spec.min() + 1e-8) * 255).astype(np.uint8)
            image = Image.fromarray(spec_norm)
        else:
            image = Image.fromarray(spec.squeeze().astype(np.uint8))
        
        if self.model is not None and TRANSFORMERS_AVAILABLE:
            # DinoV2 embedding
            if torch.cuda.is_available():
                inputs = self.processor(images=image, return_tensors="pt").to('cuda')
            else:
                inputs = self.processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token or mean pool (DINOv2 best practice: last_hidden_state mean)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        else:
            # Fallback: mean-pool as in original (for testing without deps)
            if spec.ndim == 2:
                embedding = np.mean(spec, axis=1)  # simple reduction
            else:
                embedding = spec.flatten()[:384]  # truncate for compatibility
        
        self.stream_counter += 1
        return embedding.astype(np.float32)
    
    def get_remaining_num_points(self):
        return self.n_samples - self.stream_counter
    
    def get_true_label_for_idx(self, stream_idx):
        """For oracle or evaluation only - program should not use during normal streaming"""
        if stream_idx >= len(self.df):
            return None
        row = self.df.iloc[stream_idx]
        return row.get('class_name', row.get('primary_label', 'unknown'))


# Reuse DiscoveryTracker from parent module (import will be handled in main)
from Spectrogram_A_RED.SpectrogramDataStream import DiscoveryTracker
