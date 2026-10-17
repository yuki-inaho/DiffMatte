"""Small compatibility utilities used by the inference/export path.

The original project depends on timm/fvcore/fairscale/detectron2 for a few helper
classes.  ONNX conversion does not need those frameworks, so this module keeps
only the tiny pieces required by the model definitions.
"""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


class DropPath(nn.Module):
    """Per-sample stochastic depth, equivalent to timm's DropPath."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    """Minimal timm-compatible MLP block."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return self.drop2(x)


def to_2tuple(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


def trunc_normal_(tensor: torch.Tensor, std: float = 1.0) -> torch.Tensor:
    return nn.init.trunc_normal_(tensor, std=std)


def c2_msra_fill(module: nn.Module) -> None:
    """Detectron2/fvcore Caffe2 MSRA initialization for Conv2d."""
    if not isinstance(module, nn.Conv2d):
        raise TypeError(f"Expected nn.Conv2d, got {type(module)!r}")
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class CNNBlockBase(nn.Module):
    """Subset of detectron2.layers.CNNBlockBase used by DiffMatte."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride


Conv2d = nn.Conv2d


def get_norm(norm: str | Callable[[int], nn.Module] | None, channels: int) -> nn.Module | None:
    if norm is None or norm == "":
        return None
    if callable(norm):
        return norm(channels)
    if norm == "BN":
        return nn.BatchNorm2d(channels)
    if norm == "GN":
        return nn.GroupNorm(32, channels)
    if norm == "LN":
        # Detectron2's LN in this code path is channel-wise LayerNorm for NCHW.
        return LayerNorm2d(channels)
    raise KeyError(f"Unsupported norm: {norm}")


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


def checkpoint_wrapper(module: nn.Module) -> nn.Module:
    """Inference-safe replacement for fairscale checkpoint_wrapper.

    Activation checkpointing is disabled in all released inference configs, so
    returning the module unchanged preserves inference behavior.
    """
    return module
