"""
PHX_A_RED_Project

Consolidated, object-oriented refactoring of the ARED experiment frontends.

Provides:
- Unified data streams for spectrograms, Perch embeddings, and DinoV3 embeddings.
- A single NoRelevanceOracle.
- Reusable discovery tracking, shifting-kappa controller.
- A high-level AREDExperiment runner.
- Feature extractors and pretraining utilities.

All new code only. Original sources in sibling folders are left untouched.
"""

from .data.oracle import NoRelevanceOracle
from .data.base_stream import BaseDataStream, SpectrogramDataStream, PerchDataStream, Dinov3DataStream
from .experiment import AREDExperiment, run_ared

__all__ = [
    "NoRelevanceOracle",
    "BaseDataStream",
    "SpectrogramDataStream",
    "PerchDataStream",
    "Dinov3DataStream",
    "AREDExperiment",
    "run_ared",
]
