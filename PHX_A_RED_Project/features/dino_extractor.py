"""
DinoV3 / DINOv2 feature extractor for spectrograms.

Converts a Mel-spectrogram .npy into an RGB image (exactly as done in the
original pretraining + Dinov3DataStream), runs a timm ViT, and returns
L2-normalized embedding.

Supports:
- Hub pretrained backbones (for generic DinoV2)
- Custom student checkpoint from our pretraining (student_state_dict)
"""
from pathlib import Path
from typing import Optional, Union
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import timm


class DinoV3Extractor:
    def __init__(
        self,
        model_name: str = "vit_small_patch16_dinov3.lvd1689m",
        embed_dim: int = 384,
        device: Optional[torch.device] = None,
        pretrained_path: Optional[Union[str, Path]] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = embed_dim
        self.model_name = model_name

        print(f"[DinoV3Extractor] Creating model {model_name} on {self.device}")
        self.model = timm.create_model(model_name, pretrained=pretrained_path is None, num_classes=0).to(self.device)

        if pretrained_path and Path(pretrained_path).exists():
            print(f"[DinoV3Extractor] Loading custom weights from {pretrained_path}")
            ckpt = torch.load(pretrained_path, map_location=self.device)
            # Support both our pretrainer format and plain state_dict
            state = ckpt.get("student_state_dict", ckpt)
            self.model.load_state_dict(state, strict=False)
            print("[DinoV3Extractor] Custom weights loaded.")
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def _spec_to_image(self, spec: np.ndarray) -> Image.Image:
        """Convert (128, T) or (T, 128) spectrogram to RGB PIL image."""
        if spec.ndim == 2:
            spec_norm = ((spec - spec.min()) / (spec.max() - spec.min() + 1e-8) * 255).clip(0, 255).astype(np.uint8)
            return Image.fromarray(spec_norm).convert("RGB")
        return Image.fromarray(spec.squeeze().astype(np.uint8)).convert("RGB")

    @torch.no_grad()
    def extract_from_array(self, spec: np.ndarray) -> np.ndarray:
        img = self._spec_to_image(spec)
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        feats = self.model(tensor)
        emb = feats.squeeze(0).cpu().numpy().astype(np.float32)
        # Same normalization used in the original Dino streaming code
        emb = (emb - emb.mean()) / (emb.std() + 1e-8)
        return emb

    @torch.no_grad()
    def extract_from_npy(self, npy_path: Union[str, Path]) -> np.ndarray:
        spec = np.load(npy_path, mmap_mode="r").astype(np.float32)
        return self.extract_from_array(spec)

    @torch.no_grad()
    def extract_batch_from_npy(self, npy_paths: list) -> np.ndarray:
        embs = [self.extract_from_npy(p) for p in npy_paths]
        return np.stack(embs)
