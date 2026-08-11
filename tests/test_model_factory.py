from collections import OrderedDict

import torch

from diffmatte_onnx.model_factory import get_model_spec, normalize_checkpoint_state


def test_config_path_normalization() -> None:
    assert get_model_spec("configs/ViTS_1024.py").name == "ViTS_1024"


def test_detectron2_wrapper_prefix_is_removed() -> None:
    state = OrderedDict(
        {
            "module.model.backbone.weight": torch.tensor([1.0]),
            "module.model.decoder.weight": torch.tensor([2.0]),
        }
    )
    normalized = normalize_checkpoint_state(state)
    assert set(normalized) == {"backbone.weight", "decoder.weight"}
