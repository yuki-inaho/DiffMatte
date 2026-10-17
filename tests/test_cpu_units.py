from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from diffmatte_cpu.convert_diffmatte import quantize_array_uint8  # noqa: E402
from diffmatte_cpu.convrot import self_test  # noqa: E402


def test_convrot_rotation_preserves_linear_outputs() -> None:
    report = self_test()
    assert report["status"] == "passed"
    assert report["onnx_max_abs_error"] <= 3e-6


def test_uint8_weight_quantization_round_trip() -> None:
    weights = np.random.default_rng(0).standard_normal((8, 4, 3, 3)).astype(np.float32)
    quantized, scale, zero = quantize_array_uint8(weights, 0)
    assert quantized.dtype == zero.dtype == np.uint8 and scale.dtype == np.float32
    restored = (quantized.astype(np.float32) - zero[:, None, None, None]) * scale[:, None, None, None]
    # 丸め誤差は半ステップ。ゼロ点の丸めで端の値だけ 1 ステップまで clip されることがあります。
    assert np.all(np.abs(restored - weights) <= scale[:, None, None, None] + 1e-6)
    assert np.mean(np.abs(restored - weights)) <= float(scale.max()) * 0.5
