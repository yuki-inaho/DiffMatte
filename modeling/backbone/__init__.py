from .backbone import Backbone
from .res34 import BasicBlock, ResNet_D
from .swin import Swin
from .utils import *
from .vit import ViT

def create_backbone(name, kwargs):
    if name == "ViT":
        cls = ViT
    elif name == "Res34":
        cls = ResNet_D
    elif name == "Swin":
        cls = Swin
    elif name == "ViT_d646":
        # This historical variant still uses the legacy Detectron2 helpers.
        from .vit_grid import ViT as ViTGrid

        cls = ViTGrid
    else:
        raise KeyError(f"Unknown backbone: {name}")
    return cls(**kwargs)


__all__ = [
    "Backbone",
    "BasicBlock",
    "ResNet_D",
    "Swin",
    "ViT",
    "create_backbone",
]
