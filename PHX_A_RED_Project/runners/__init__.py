"""Thin executable runners for different ARED frontends.

Run with:
    python -m PHX_A_RED_Project.runners.spectrogram
    python -m PHX_A_RED_Project.runners.perch
    python -m PHX_A_RED_Project.runners.dinov3
    python -m PHX_A_RED_Project.runners.with_shifting_kappa --target 5
    python -m PHX_A_RED_Project.runners.random_baseline --num-points 300 --label-column class_name

The imports below are lazy (PEP 562 __getattr__) so that
`python -m PHX_A_RED_Project.runners.xxx` does not trigger
the "found in sys.modules after import of package" RuntimeWarning.
"""

__all__ = ["run_spectrogram", "run_perch", "run_dinov3", "run_with_shifting_kappa", "run_random_baseline"]


def __getattr__(name: str):
    if name == "run_spectrogram":
        from .spectrogram import main as _m
        return _m
    if name == "run_perch":
        from .perch import main as _m
        return _m
    if name == "run_dinov3":
        from .dinov3 import main as _m
        return _m
    if name == "run_with_shifting_kappa":
        from .with_shifting_kappa import main as _m
        return _m
    if name == "run_random_baseline":
        from .random_baseline import main as _m
        return _m
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
