from .difmatte import DifMatte
from .difmodel import DifModel

try:
    from .vitmatte import ViTMatte
except (ImportError, ModuleNotFoundError):  # legacy-only architecture
    ViTMatte = None

__all__ = ["DifMatte", "DifModel", "ViTMatte"]
