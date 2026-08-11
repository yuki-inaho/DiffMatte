import torch

from diffmatte_onnx.model_factory import build_core_model
from diffmatte_onnx.sampler import DiffMatteDDIM
from modeling.diffusion.script_utils import create_uniform_gaussian_diffusion


def test_fixed_ddim_matches_upstream_uniform_sampler() -> None:
    torch.manual_seed(7)
    model = build_core_model("ViTS").eval()
    image = torch.rand(1, 3, 64, 64)
    trimap = torch.full((1, 1, 64, 64), 0.5)
    trimap[:, :, :8, :] = 0.0
    trimap[:, :, -8:, :] = 1.0
    noise = torch.randn(1, 1, 64, 64)

    with torch.inference_mode():
        actual = DiffMatteDDIM(model, steps=2, scale=0.2).eval()(image, trimap, noise)
        diffusion = create_uniform_gaussian_diffusion(
            steps=2,
            noise_schedule="linear",
            scale=0.2,
            predict_xstart=True,
        )
        expected = diffusion.ddim_sample(
            model,
            trimap.shape,
            {"image": image.clone(), "trimap": trimap.clone()},
            noise=noise.clone(),
            device=torch.device("cpu"),
        )
        expected = torch.clamp(expected, 0.0, 1.0)
        expected = torch.where(trimap == 0.0, torch.zeros_like(expected), expected)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
