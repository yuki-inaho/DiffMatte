"""元の FP32 ONNX から ConvRot + UINT8 保存モデルを作り、実画像の CPU 推論まで確認します。

比較対象は同じ入力・同じ初期乱数 (seed 42) で元モデルを ONNX Runtime CPU に通したアルファです。
正解アルファとの精度ではなく、元モデルとの一致度を確認します。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

# `uv sync --extra cpu` で入る依存がなければスキップします。
for _module in ("onnx", "onnxruntime", "onnxsim", "openvino", "psutil"):
    pytest.importorskip(_module)

import onnx  # noqa: E402
from onnx import TensorProto  # noqa: E402

from conftest import DEMO_DIR, run_module, sha256  # noqa: E402
from diffmatte_cpu.benchmark_models import difference_metrics  # noqa: E402
from diffmatte_cpu.run_diffmatte import read_source_inputs  # noqa: E402

pytestmark = pytest.mark.e2e

SEED = 42
# bulb は 960x640 のため、右・下を端の画素で 643 行まで埋めて推論し、元寸法へ戻します。
IMAGES = {"retriever": False, "bulb": True}
# 元モデルとの未知領域の一致度の上限。配布時の実測 (EPYC 9V74) に約 2 倍の余裕を持たせています。
# 推奨: MAE 0.00145/0.00165, P99 0.0112/0.0192, 最大 0.0625/0.115
# FP32: MAE 0.00093/0.00110, P99 0.0070/0.0131, 最大 0.0366/0.0891
LIMITS = {
    "openvino_recommended": {"mae": 0.003, "rmse": 0.008, "absolute_error_p99": 0.035, "absolute_error_max": 0.25},
    "openvino_f32": {"mae": 0.002, "rmse": 0.006, "absolute_error_p99": 0.025, "absolute_error_max": 0.2},
    "ort_uint8": {"mae": 0.002, "rmse": 0.006, "absolute_error_p99": 0.025, "absolute_error_max": 0.2},
}


def demo_pair(name: str) -> tuple[Path, Path]:
    return DEMO_DIR / f"{name}_rgb.png", DEMO_DIR / f"{name}_trimap.png"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def converted(e2e_dir: Path, original_onnx: Path, e2e_threads: int) -> Path:
    prepared = e2e_dir / "prepared.onnx"
    output = e2e_dir / "diffmatte_convrot_uint8.onnx"
    run_module("convert_diffmatte", "--input", original_onnx, "--output", prepared,
               "--variant", "prepare", "--threads", e2e_threads)
    run_module("convert_diffmatte", "--input", prepared, "--output", output, "--variant", "weights",
               "--prepared", "--group-size", 64, "--weight-group-size", 16, "--linear-activation", "fp32",
               "--threads", e2e_threads)
    return output


@pytest.fixture(scope="session")
def reference_alpha(e2e_dir: Path, original_onnx: Path, e2e_threads: int):
    cache: dict[str, Path] = {}

    def get(name: str) -> Path:
        if name not in cache:
            image, trimap = demo_pair(name)
            output_dir = e2e_dir / name / "original_ort"
            arguments = ["--model", original_onnx, "--image", image, "--trimap", trimap,
                         "--output-dir", output_dir, "--threads", e2e_threads, "--save-npy",
                         "--seed", SEED, "--warmup", 0, "--runs", 1]
            if IMAGES[name]:
                arguments.append("--pad-to-model")
            run_module("run_diffmatte", *arguments)
            cache[name] = output_dir / "alpha.npy"
        return cache[name]

    return get


def test_conversion_shrinks_model_and_keeps_contract(converted: Path, original_onnx: Path) -> None:
    size = converted.stat().st_size
    assert size <= 0.32 * original_onnx.stat().st_size, f"圧縮後 {size} bytes は元の 32% を超えています"

    model = onnx.load(str(converted))
    onnx.checker.check_model(model)
    shapes = {
        value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        for value in list(model.graph.input) + list(model.graph.output)
    }
    assert shapes == {
        "image": [1, 3, 643, 960], "trimap": [1, 1, 643, 960],
        "initial_noise": [1, 1, 643, 960], "alpha": [1, 1, 643, 960],
    }
    for value in list(model.graph.input) + list(model.graph.output):
        assert value.type.tensor_type.elem_type == TensorProto.FLOAT

    metadata = {prop.key: prop.value for prop in model.metadata_props}
    assert metadata["convrot.enabled"] == "true"
    assert metadata["convrot.group_size"] == "64"
    assert metadata["quantization.linear_activation"] == "fp32"

    report = read_json(converted.with_suffix(".conversion.json"))
    assert report["rotation"]["counts"]["rotated_matmul"] == 48
    assert len(report["weight_storage"]) == report["inventory"]["operators"]["DequantizeLinear"] == 106
    assert report["inventory"]["initializer_bytes_by_type"]["UINT8"] > 30_000_000

    # 出力 matting head の 2 Conv の重みは量子化せず FP32 定数のまま保ちます。
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    head_weights = [node.input[1] for node in model.graph.node
                    if node.op_type == "Conv" and "matting_head" in node.input[1]]
    assert head_weights
    for name in head_weights:
        assert initializers[name].data_type == TensorProto.FLOAT


def test_conversion_matches_reference_model(converted: Path) -> None:
    configured = os.environ.get("DIFFMATTE_CONVROT_REFERENCE")
    if not configured:
        pytest.skip("DIFFMATTE_CONVROT_REFERENCE に配布済みの変換モデルを指定すると照合します。")
    reference = onnx.load(str(Path(configured).expanduser()))
    candidate = onnx.load(str(converted))
    assert list(reference.graph.node) == list(candidate.graph.node)
    expected = {tensor.name: onnx.numpy_helper.to_array(tensor) for tensor in reference.graph.initializer}
    actual = {tensor.name: onnx.numpy_helper.to_array(tensor) for tensor in candidate.graph.initializer}
    assert expected.keys() == actual.keys()
    total = differing = 0
    for name, array in expected.items():
        assert array.shape == actual[name].shape and array.dtype == actual[name].dtype, name
        if array.dtype == np.uint8:
            delta = np.abs(array.astype(np.int16) - actual[name].astype(np.int16))
            # 回転行列積の浮動小数点の丸め差 (CPU の SIMD 差) による 1 LSB だけを許容します。
            assert int(delta.max(initial=0)) <= 1, name
            total += array.size
            differing += int(np.count_nonzero(delta))
        else:
            np.testing.assert_allclose(actual[name], array, rtol=1e-5, atol=1e-7, err_msg=name)
    assert differing / total < 1e-4


def run_candidate(mode: str, model: Path, name: str, output_dir: Path, threads: int) -> Path:
    image, trimap = demo_pair(name)
    common = ["--model", model, "--image", image, "--trimap", trimap, "--output-dir", output_dir,
              "--threads", threads, "--seed", SEED, "--warmup", 0, "--runs", 1]
    if IMAGES[name]:
        common.append("--pad-to-model")
    if mode == "openvino_recommended":
        run_module("run_openvino", *common, "--precision", "bf16", "--keep-head-fp32")
    elif mode == "openvino_f32":
        run_module("run_openvino", *common, "--precision", "f32")
    else:
        run_module("run_diffmatte", *common, "--prepack-float-weights", "--save-npy")
    return output_dir


@pytest.mark.parametrize("name", list(IMAGES))
@pytest.mark.parametrize("mode", list(LIMITS))
def test_cpu_inference_matches_original(mode: str, name: str, converted: Path, reference_alpha,
                                        e2e_dir: Path, e2e_threads: int) -> None:
    reference_path = reference_alpha(name)
    output_dir = run_candidate(mode, converted, name, e2e_dir / name / mode, e2e_threads)
    timing = read_json(output_dir / "timing.json")

    assert timing["input"]["seed"] == SEED
    assert timing["input"]["padding"]["performed"] is IMAGES[name]
    reference_timing = read_json(reference_path.with_name("timing.json"))
    assert timing["input"]["feeds_sha256"] == reference_timing["input"]["feeds_sha256"]

    reference = np.load(reference_path, allow_pickle=False)
    alpha = np.load(output_dir / "alpha.npy", allow_pickle=False)
    image, trimap = demo_pair(name)
    rgb, tri = read_source_inputs(image, trimap)
    assert alpha.shape == reference.shape == rgb.shape[-2:]
    assert np.isfinite(alpha).all()
    assert (alpha >= -1e-3).all() and (alpha <= 1 + 1e-3).all()

    unknown = difference_metrics(reference, alpha, tri[0, 0] == 0.5)
    full = difference_metrics(reference, alpha)
    for key, limit in LIMITS[mode].items():
        assert unknown[key] <= limit, f"{mode}/{name}: 未知領域 {key}={unknown[key]:.6g} > {limit}"

    runtime = timing["runtime"]
    if mode.startswith("openvino"):
        assert runtime["effective_properties"]["EXECUTION_DEVICES"] == "['CPU']"
    if mode == "openvino_recommended":
        head = runtime["head_precision"]
        assert head["runtime_verification_passed"] is True
        assert head["head_call_count"] == 10 and head["target_node_count"] == 60
        assert not head["missing_runtime_targets"] and not head["non_fp32_runtime_nodes"]
        # BF16 命令を持たない CPU では OpenVINO が FP32 に戻すため、実際の実行精度を記録と照合します。
        if "BF16" in runtime["optimization_capabilities"]:
            assert runtime["bf16_runtime_node_count"] > 0
        else:
            assert runtime["bf16_runtime_node_count"] == 0

    record = {
        "mode": mode, "image": name, "model": str(converted), "model_sha256": sha256(converted),
        "device": runtime.get("device_name"), "threads": e2e_threads,
        "bf16_runtime_node_count": runtime.get("bf16_runtime_node_count"),
        "median_seconds": timing["timing_seconds"]["median"],
        "unknown_region": unknown, "full_image": full,
    }
    (output_dir / "e2e_result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                                                encoding="utf-8")
