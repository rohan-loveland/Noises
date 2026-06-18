"""Thin executable runners for different ARED frontends.

Run with:
    python -m PHX_A_RED_Project.runners.spectrogram
    python -m PHX_A_RED_Project.runners.perch
    python -m PHX_A_RED_Project.runners.dinov3
    python -m PHX_A_RED_Project.runners.with_shifting_kappa --target 5
"""

from .spectrogram import main as run_spectrogram
from .perch import main as run_perch
from .dinov3 import main as run_dinov3
from .with_shifting_kappa import main as run_with_shifting_kappa

__all__ = ["run_spectrogram", "run_perch", "run_dinov3", "run_with_shifting_kappa"]
