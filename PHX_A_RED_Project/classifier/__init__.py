"""
PHX_A_RED_Project.classifier

SoundClassifier (supervised) and AREDClassifier (ARED-selected prototype NN).
"""

__all__ = ["AREDClassifier", "SoundClassifier"]


def __getattr__(name: str):
    if name == "AREDClassifier":
        from .ared_classifier import AREDClassifier as _cls
        return _cls
    if name == "SoundClassifier":
        from .sound_classifier import SoundClassifier as _cls
        return _cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

