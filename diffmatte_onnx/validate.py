"""Validate a DiffMatte ONNX graph against the PyTorch checkpoint on one image."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .model_factory import build_core_model, get_model_spec, load_checkpoint
from .sampler import DiffMatteDDIM


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch and ONNX Runtime DiffMatte outputs"
    )
    parser.add_argument("--config", default="ViTS_1024")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--trimap", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    return parser.parse_args()


def load_inputs(image_path: Path, trimap_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(image_path) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    with Image.open(trimap_path) as source:
        gray = np.asarray(source.convert("L"), dtype=np.float32) / 255.0
    if rgb.shape[:2] != gray.shape:
        raise ValueError(f"Image and trimap sizes differ: {rgb.shape[:2]} vs {gray.shape}")

    quantized = np.empty_like(gray, dtype=np.float32)
    quantized[gray > 0.9] = 1.0
    quantized[(gray >= 0.1) & (gray <= 0.9)] = 0.5
    quantized[gray < 0.1] = 0.0
    image = np.transpose(rgb, (2, 0, 1))[None, ...]
    trimap = quantized[None, None, ...]
    return np.ascontiguousarray(image), np.ascontiguousarray(trimap)


def _save_alpha(path: Path, alpha: np.ndarray) -> None:
    image = Image.fromarray(np.rint(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8))
    image.save(path)


def main() -> None:
    args = _parse_args()
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Install export dependencies with `uv sync --extra export`.") from exc

    image_np, trimap_np = load_inputs(args.image, args.trimap)
    rng = np.random.default_rng(args.seed)
    noise_np = rng.standard_normal(
        (image_np.shape[0], 1, image_np.shape[2], image_np.shape[3]),
        dtype=np.float32,
    )

    spec = get_model_spec(args.config)
    core = build_core_model(spec.name)
    load_checkpoint(core, args.checkpoint, strict=True)
    sampler = DiffMatteDDIM(core.eval(), steps=args.steps, scale=spec.diffusion_scale).eval()

    inputs = tuple(torch.from_numpy(value) for value in (image_np, trimap_np, noise_np))
    with torch.inference_mode():
        torch_alpha = sampler(*inputs).cpu().numpy()

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    ort_alpha = session.run(
        ["alpha"],
        {"image": image_np, "trimap": trimap_np, "initial_noise": noise_np},
    )[0]
    absolute_diff = np.abs(torch_alpha - ort_alpha)
    np.testing.assert_allclose(ort_alpha, torch_alpha, atol=args.atol, rtol=args.rtol)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _save_alpha(args.output_dir / "pytorch_alpha.png", torch_alpha[0, 0])
    _save_alpha(args.output_dir / "onnx_alpha.png", ort_alpha[0, 0])
    diff_max = float(absolute_diff.max(initial=0.0))
    diff_visual = absolute_diff[0, 0] / diff_max if diff_max > 0.0 else absolute_diff[0, 0]
    _save_alpha(args.output_dir / "absolute_diff_normalized.png", diff_visual)
    np.save(args.output_dir / "initial_noise.npy", noise_np, allow_pickle=False)

    report = {
        "config": spec.name,
        "checkpoint": str(args.checkpoint.resolve()),
        "onnx": str(args.model.resolve()),
        "image": str(args.image.resolve()),
        "trimap": str(args.trimap.resolve()),
        "shape": list(image_np.shape),
        "steps": args.steps,
        "seed": args.seed,
        "provider": session.get_providers(),
        "max_abs_error": diff_max,
        "mean_abs_error": float(absolute_diff.mean()),
        "atol": args.atol,
        "rtol": args.rtol,
        "allclose": True,
    }
    report_path = args.output_dir / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
