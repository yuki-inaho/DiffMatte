"""CLI for exporting a released DiffMatte checkpoint to a fixed-step ONNX graph."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model_factory import build_core_model, get_model_spec, load_checkpoint
from .sampler import DiffMatteDDIM


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export DiffMatte DDIM inference to ONNX")
    parser.add_argument("--config", default="ViTS_1024", help="ViTS, ViTS_1024, ViTB, res34, or swin_t")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--device", default="cpu", help="cpu or cuda[:index]")
    parser.add_argument("--verify", action="store_true", help="Compare PyTorch and ONNX Runtime outputs")
    parser.add_argument("--atol", type=float, default=3e-4)
    parser.add_argument("--rtol", type=float, default=3e-4)
    parser.add_argument("--legacy-exporter", action="store_true", help="Use TorchScript-based exporter fallback")
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Export a symbolic batch dimension (default: fixed N=1, the contract of the released models)",
    )
    return parser.parse_args()


def _validate_shape(height: int, width: int) -> None:
    if height <= 0 or width <= 0:
        raise ValueError("height/width must be positive")
    if height < 32 or width < 32:
        raise ValueError("height/width must be at least 32")


def _export(
    sampler: DiffMatteDDIM,
    example_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    output: Path,
    *,
    opset: int,
    legacy: bool,
    dynamic_batch: bool,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    input_names = ["image", "trimap", "initial_noise"]
    output_names = ["alpha"]

    if not legacy:
        # PyTorch 2.7 specialized the batch-1 example to a static 1; newer releases keep the symbol.
        dynamic_shapes = None
        if dynamic_batch:
            batch = torch.export.Dim("batch", min=1)
            dynamic_shapes = ({0: batch}, {0: batch}, {0: batch})
        try:
            torch.onnx.export(
                sampler,
                example_inputs,
                str(output),
                input_names=input_names,
                output_names=output_names,
                opset_version=opset,
                dynamo=True,
                dynamic_shapes=dynamic_shapes,
                external_data=False,
                optimize=True,
            )
            return
        # Exporter failures span several exception types across PyTorch releases.
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] torch.export-based ONNX export failed: {exc}")
            print("[warn] retrying with the legacy exporter")

    torch.onnx.export(
        sampler,
        example_inputs,
        str(output),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        dynamo=False,
        dynamic_axes={
            "image": {0: "batch"},
            "trimap": {0: "batch"},
            "initial_noise": {0: "batch"},
            "alpha": {0: "batch"},
        }
        if dynamic_batch
        else None,
        do_constant_folding=True,
    )


def _check_onnx(path: Path) -> bool:
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Install the export extra: uv sync --extra export") from exc
    model = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(model)
    image_input = next(value for value in model.graph.input if value.name == "image")
    batch_dim = image_input.type.tensor_type.shape.dim[0]
    return bool(batch_dim.dim_param)


def _verify_ort(
    sampler: DiffMatteDDIM,
    example_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    output: Path,
    *,
    atol: float,
    rtol: float,
) -> dict[str, float]:
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Install ONNX Runtime: uv sync --extra export --extra runtime-cpu "
            "(or --extra runtime-gpu)"
        ) from exc

    with torch.inference_mode():
        torch_out = sampler(*example_inputs).detach().cpu().numpy()

    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(output), providers=providers)
    feed = {
        "image": example_inputs[0].detach().cpu().numpy(),
        "trimap": example_inputs[1].detach().cpu().numpy(),
        "initial_noise": example_inputs[2].detach().cpu().numpy(),
    }
    ort_out = session.run(["alpha"], feed)[0]
    abs_diff = np.abs(torch_out - ort_out)
    np.testing.assert_allclose(ort_out, torch_out, atol=atol, rtol=rtol)
    return {
        "max_abs_error": float(abs_diff.max(initial=0.0)),
        "mean_abs_error": float(abs_diff.mean()),
    }


def main() -> None:
    args = _parse_args()
    spec = get_model_spec(args.config)
    height = args.height or spec.image_size
    width = args.width or spec.image_size
    _validate_shape(height, width)

    device = torch.device(args.device)
    core = build_core_model(spec.name)
    missing, unexpected = load_checkpoint(core, args.checkpoint, strict=True)
    assert not missing and not unexpected
    core.eval().to(device)

    sampler = DiffMatteDDIM(
        core,
        steps=args.steps,
        scale=spec.diffusion_scale,
    ).eval().to(device)

    generator = torch.Generator(device="cpu").manual_seed(0)
    image = torch.rand((1, 3, height, width), generator=generator, dtype=torch.float32).to(device)
    trimap = torch.full((1, 1, height, width), 0.5, dtype=torch.float32, device=device)
    noise = torch.randn((1, 1, height, width), generator=generator, dtype=torch.float32).to(device)
    example_inputs = (image, trimap, noise)

    with torch.inference_mode():
        _export(
            sampler,
            example_inputs,
            args.output,
            opset=args.opset,
            legacy=args.legacy_exporter,
            dynamic_batch=args.dynamic_batch,
        )

    dynamic_batch = _check_onnx(args.output)
    report: dict[str, object] = {
        "config": spec.name,
        "checkpoint": str(args.checkpoint),
        "onnx": str(args.output),
        "height": height,
        "width": width,
        "steps": args.steps,
        "opset": args.opset,
        "dynamic_batch": dynamic_batch,
        "dynamic_spatial": False,
    }
    if args.verify:
        report["verification"] = _verify_ort(
            sampler,
            example_inputs,
            args.output,
            atol=args.atol,
            rtol=args.rtol,
        )

    report_path = args.output.with_suffix(args.output.suffix + ".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
