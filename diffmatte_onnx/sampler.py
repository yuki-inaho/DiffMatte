"""ONNX-friendly fixed-step DDIM sampler for DiffMatte."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from modeling.meta_arch.difmodel import DifModel


class DiffMatteDDIM(nn.Module):
    """End-to-end DiffMatte inference graph with one cached backbone pass.

    Inputs use the same conventions as ``run_one_image.py``:
      * image: float32 RGB in [0, 1], NCHW
      * trimap: float32 in {0, 0.5, 1}, N1HW
      * initial_noise: float32 standard-normal-like tensor, N1HW

    Spatial dimensions are intended to be fixed at ONNX export time; batch may
    remain dynamic.  Initial noise is an explicit input so ONNX inference is
    deterministic and can be compared bit-for-bit for a fixed runtime/provider.
    """

    def __init__(
        self,
        model: DifModel,
        *,
        steps: int = 10,
        scale: float = 0.2,
        time_difference: float = 0.0,
    ) -> None:
        super().__init__()
        if steps <= 0:
            raise ValueError("steps must be > 0")
        if not (0.0 < scale <= 1.0):
            raise ValueError("scale must be in (0, 1]")
        self.model = model
        self.steps = int(steps)
        self.scale = float(scale)
        self.time_difference = float(time_difference)

    @staticmethod
    def _one_hot_trimap(trimap: torch.Tensor) -> torch.Tensor:
        indexes = (trimap * 2.0).to(torch.int64)
        return F.one_hot(indexes, num_classes=3).squeeze(1).permute(0, 3, 1, 2).to(trimap.dtype)

    @staticmethod
    def _pad32(x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2], x.shape[-1]
        pad_h = (32 - height % 32) % 32
        pad_w = (32 - width % 32) % 32
        return F.pad(x, (0, pad_w, 0, pad_h), value=0.0)

    def _prepare_condition(
        self, image: torch.Tensor, trimap: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_norm = (image - self.model.pixel_mean) / self.model.pixel_std
        tri = self._one_hot_trimap(trimap)

        # Decoder conditioning in released configs: RGB + 3-way trimap = 6 ch.
        if self.model.decoder_in_channel == 4:
            decoder_cond = image_norm
        elif self.model.decoder_in_channel == 7:
            decoder_cond = torch.cat((image_norm, tri), dim=1)
        else:
            raise RuntimeError(
                "The ONNX path supports decoder_in_channel=4 or 7. "
                f"Got {self.model.decoder_in_channel}."
            )
        decoder_cond = decoder_cond * 2.0 - 1.0

        if self.model.backbone_in_channel == 4:
            backbone_input = torch.cat((image_norm, trimap), dim=1)
        elif self.model.backbone_in_channel == 6:
            backbone_input = torch.cat((image_norm, tri), dim=1)
        else:
            raise RuntimeError(f"Unsupported backbone_in_channel={self.model.backbone_in_channel}")

        return self._pad32(backbone_input), decoder_cond

    @staticmethod
    def _normalize_variance(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        std = torch.std(x, dim=(1, 2, 3), keepdim=True, unbiased=False)
        return x / torch.clamp(std, min=eps)

    @staticmethod
    def _linear_gamma(t: torch.Tensor) -> torch.Tensor:
        return torch.clamp(1.0 - t, min=1e-9)

    @staticmethod
    def _safe_div(numer: torch.Tensor, denom: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        return numer / torch.clamp(denom, min=eps)

    def _decoder_step(
        self,
        features: torch.Tensor,
        decoder_cond: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        if self.scale < 1.0:
            x_for_model = self._normalize_variance(x_t)
        else:
            x_for_model = x_t
        decoder_input = torch.cat((decoder_cond, x_for_model), dim=1)
        decoder_input = self._pad32(decoder_input)
        pred = self.model.decoder(features, decoder_input, timestep)["phas"]
        return pred[:, :, : x_t.shape[-2], : x_t.shape[-1]]

    def forward(
        self,
        image: torch.Tensor,
        trimap: torch.Tensor,
        initial_noise: torch.Tensor,
    ) -> torch.Tensor:
        backbone_input, decoder_cond = self._prepare_condition(image, trimap)
        features = self.model.backbone(backbone_input)

        batch = image.shape[0]
        x_t = initial_noise
        times = torch.linspace(1.0, 0.0, self.steps + 1, device=image.device, dtype=image.dtype)

        for i in range(self.steps):
            t = times[i].expand(batch)
            t_next = times[i + 1].expand(batch)

            gamma = self._linear_gamma(t).view(batch, 1, 1, 1)
            gamma_next = self._linear_gamma(t_next).view(batch, 1, 1, 1)
            alpha = torch.sqrt(gamma) * self.scale
            sigma = torch.sqrt(1.0 - gamma)
            alpha_next = torch.sqrt(gamma_next) * self.scale
            sigma_next = torch.sqrt(1.0 - gamma_next)

            # Keep ordering consistent with the original sampler.  The released
            # configuration has time_difference=0, but the option is retained.
            if self.time_difference != 0.0:
                t_next = torch.clamp(t_next - self.time_difference, min=0.0)

            pred_xstart = torch.clamp(
                self._decoder_step(features, decoder_cond, x_t, t),
                -1.0,
                1.0,
            )
            pred_noise = self._safe_div(x_t - alpha * pred_xstart, sigma)
            x_t = pred_xstart * alpha_next + pred_noise * sigma_next

        # At t_next=0, x_t == scale * last predicted x0.  This exactly matches
        # UniformGauss.ddim_sample()'s final unnormalize_img(noised_pha / scale).
        alpha_matte = (x_t / self.scale + 1.0) * 0.5
        alpha_matte = torch.clamp(alpha_matte, 0.0, 1.0)

        # Preserve original DiffMatte behavior for strict conversion parity.
        # run_one_image.py uses trimap values 0/0.5/1 while DifMatte checks 0/2,
        # therefore only known background is clamped in the upstream code.
        alpha_matte = torch.where(trimap == 0.0, torch.zeros_like(alpha_matte), alpha_matte)
        return alpha_matte
