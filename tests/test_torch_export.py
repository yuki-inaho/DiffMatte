import gc

import pytest
import torch

from diffmatte_onnx.model_factory import build_core_model
from diffmatte_onnx.sampler import DiffMatteDDIM


@pytest.mark.parametrize("config", ["ViTS", "res34", "swin_t"])
def test_sampler_is_torch_export_captureable(config: str) -> None:
    model = build_core_model(config).eval()
    sampler = DiffMatteDDIM(model, steps=1, scale=0.2).eval()
    args = (
        torch.rand(1, 3, 32, 32),
        torch.full((1, 1, 32, 32), 0.5),
        torch.randn(1, 1, 32, 32),
    )
    exported = torch.export.export(sampler, args)
    assert exported is not None
    del exported, sampler, model
    gc.collect()
