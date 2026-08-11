"""Detectron2-free construction and checkpoint loading for released DiffMatte variants."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.listconfig import ListConfig
from omegaconf.nodes import AnyNode
from torch import nn

from modeling.backbone.res34 import BasicBlock, ResNet_D
from modeling.backbone.swin import Swin
from modeling.backbone.vit import ViT
from modeling.decoder.DUNet import DUNet
from modeling.meta_arch.difmodel import DifModel

PIXEL_MEAN = (123.675 / 255.0, 116.280 / 255.0, 103.530 / 255.0)
PIXEL_STD = (58.395 / 255.0, 57.120 / 255.0, 57.375 / 255.0)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    image_size: int
    backbone_in_channel: int
    decoder_in_channel: int
    diffusion_scale: float = 0.2
    diffusion_steps: int = 10


_SPECS = {
    "ViTS": ModelSpec("ViTS", 512, 4, 7),
    "ViTS_1024": ModelSpec("ViTS_1024", 1024, 4, 7),
    "ViTB": ModelSpec("ViTB", 512, 4, 7),
    "res34": ModelSpec("res34", 512, 6, 7),
    "swin_t": ModelSpec("swin_t", 512, 6, 7),
}


def get_model_spec(name: str) -> ModelSpec:
    normalized = name.removesuffix(".py").split("/")[-1]
    try:
        return _SPECS[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted(_SPECS))
        raise ValueError(f"Unsupported config {name!r}. Choose one of: {choices}") from exc


def _build_vit(*, image_size: int, embed_dim: int, num_heads: int) -> ViT:
    return ViT(
        in_chans=4,
        img_size=image_size,
        patch_size=16,
        embed_dim=embed_dim,
        depth=12,
        num_heads=num_heads,
        drop_path_rate=0.0,
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=[0, 1, 3, 4, 6, 7, 9, 10],
        residual_block_indexes=[2, 5, 8, 11],
        use_rel_pos=True,
        out_feature="last_feat",
    )


def build_core_model(name: str) -> DifModel:
    """Build the trainable PyTorch core without Detectron2/LazyConfig."""
    spec = get_model_spec(name)

    if spec.name in {"ViTS", "ViTS_1024"}:
        backbone = _build_vit(image_size=spec.image_size, embed_dim=384, num_heads=6)
        decoder = DUNet(
            model_channels=32,
            emb_channels=32,
            downsample_in=[7, 32, 64, 128],
            upsample_in=[384, 256, 128, 64, 32],
        )
    elif spec.name == "ViTB":
        backbone = _build_vit(image_size=spec.image_size, embed_dim=768, num_heads=12)
        decoder = DUNet(
            model_channels=32,
            emb_channels=32,
            downsample_in=[7, 32, 64, 128],
            upsample_in=[768, 256, 128, 64, 32],
        )
    elif spec.name == "res34":
        backbone = ResNet_D(BasicBlock, in_channels=6, layers=[3, 4, 4, 2])
        decoder = DUNet(
            model_channels=32,
            emb_channels=32,
            downsample_in=[7, 32, 64, 128, 128],
            upsample_in=[512, 256, 256, 128, 64, 32],
        )
    elif spec.name == "swin_t":
        backbone = Swin(
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False,
        )
        decoder = DUNet(
            model_channels=32,
            emb_channels=32,
            downsample_in=[7, 32, 64, 128, 128],
            upsample_in=[768, 256, 256, 128, 64, 32],
        )
    else:  # pragma: no cover - guarded by get_model_spec
        raise AssertionError(spec.name)

    return DifModel(
        backbone=backbone,
        pixel_mean=list(PIXEL_MEAN),
        pixel_std=list(PIXEL_STD),
        decoder=decoder,
        backbone_in_channel=spec.backbone_in_channel,
        decoder_in_channel=spec.decoder_in_channel,
    )


def _state_dict_from_checkpoint(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict) and candidate:
                return candidate
        if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    raise ValueError(
        "Unsupported checkpoint structure. Expected a state_dict or a dict containing "
        "'model'/'state_dict'."
    )


def _strip_common_prefix(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if state and sum(k.startswith(prefix) for k in state) >= max(1, int(len(state) * 0.8)):
        return {k.removeprefix(prefix): v for k, v in state.items()}
    return state


def normalize_checkpoint_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert common DDP/Detectron2 wrapper prefixes to DifModel state keys."""
    result = dict(state)
    for prefix in ("module.", "_orig_mod."):
        result = _strip_common_prefix(result, prefix)

    # DetectionCheckpointer was used on DifMatte, whose trainable network lives
    # under ``difmatte.model``.  The pure export model is that inner DifModel.
    result = _strip_common_prefix(result, "model.")
    return result


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    # Detectron2 checkpointers embed OmegaConf metadata alongside tensor weights.
    # Explicitly allow only the known metadata types instead of disabling the
    # weights-only unpickler for arbitrary checkpoint classes.
    safe_types = [
        dict,
        list,
        int,
        defaultdict,
        Any,
        AnyNode,
        ContainerMetadata,
        ListConfig,
        Metadata,
    ]
    with torch.serialization.safe_globals(safe_types):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = normalize_checkpoint_state(_state_dict_from_checkpoint(checkpoint))
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)

    if strict and (missing or unexpected):
        details = []
        if missing:
            details.append(f"missing={missing[:20]}")
        if unexpected:
            details.append(f"unexpected={unexpected[:20]}")
        raise RuntimeError("Checkpoint/model mismatch: " + "; ".join(details))
    return missing, unexpected
