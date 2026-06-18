"""Feature extractors for ARED (Perch audio, DinoV3 spectrogram images)."""
from .perch_embedder import PerchEmbedder
from .dino_extractor import DinoV3Extractor

__all__ = ["PerchEmbedder", "DinoV3Extractor"]
