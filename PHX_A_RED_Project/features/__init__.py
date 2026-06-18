"""
Feature extractors for ARED.

Perch (TensorFlow) and DinoV3 (PyTorch/timm) are loaded on demand only.
This avoids pulling TensorFlow into pure DinoV3 or spectrogram runs,
which is especially important in mixed TF+PyTorch GPU environments
(tf-gpu conda envs etc.) where loading both frameworks can cause
CUDA context conflicts or segfaults.
"""

__all__ = ["PerchEmbedder", "DinoV3Extractor"]


def __getattr__(name: str):
    if name == "PerchEmbedder":
        from .perch_embedder import PerchEmbedder as _cls
        return _cls
    if name == "DinoV3Extractor":
        from .dino_extractor import DinoV3Extractor as _cls
        return _cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

