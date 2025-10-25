from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn

from .utils import warn_on_unused_kwargs

ScaleSpec = Sequence[Tuple[int, int, int]]


def _default_scales() -> List[Tuple[int, int, int]]:
    return [
        (1024, 120, 600),
        (512, 60, 240),
        (256, 30, 120),
    ]


class MultiScaleSTFTLoss(nn.Module):
    """Multi-resolution spectral loss for audio signals."""

    def __init__(
        self,
        scales: ScaleSpec | None = None,
        magnitude_weight: float = 1.0,
        convergence_weight: float = 1.0,
        eps: float = 1e-7,
        **kwargs,
    ) -> None:
        super().__init__()
        warn_on_unused_kwargs(kwargs)

        scales = list(scales) if scales is not None else _default_scales()
        if not scales:
            raise ValueError("At least one STFT scale must be provided")
        self.scales = [tuple(map(int, scale)) for scale in scales]
        self.magnitude_weight = magnitude_weight
        self.convergence_weight = convergence_weight
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        if prediction.dim() != 2 or prediction.shape != target.shape:
            raise ValueError("prediction and target must be (batch, time) with matching shapes")

        mag_loss_total = prediction.new_tensor(0.0)
        sc_loss_total = prediction.new_tensor(0.0)
        for fft_size, hop_length, win_length in self.scales:
            pred_mag = self._stft_mag(prediction, fft_size, hop_length, win_length)
            target_mag = self._stft_mag(target, fft_size, hop_length, win_length)

            sc = self._spectral_convergence(pred_mag, target_mag)
            mag = torch.mean(torch.abs(pred_mag - target_mag))
            sc_loss_total = sc_loss_total + sc
            mag_loss_total = mag_loss_total + mag

        loss = (
            self.convergence_weight * sc_loss_total
            + self.magnitude_weight * mag_loss_total
        )
        return {
            "loss": loss,
            "spectral_convergence": sc_loss_total,
            "log_magnitude": mag_loss_total,
        }

    def _stft_mag(
        self,
        audio: torch.Tensor,
        fft_size: int,
        hop_length: int,
        win_length: int,
    ) -> torch.Tensor:
        window = torch.hann_window(win_length, device=audio.device, dtype=audio.dtype)
        stft = torch.stft(
            audio,
            n_fft=fft_size,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            return_complex=True,
        )
        return torch.abs(stft)

    def _spectral_convergence(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = torch.linalg.norm(target - pred, ord="fro", dim=(-2, -1))
        denom = torch.linalg.norm(target, ord="fro", dim=(-2, -1))
        denom = denom.clamp(min=self.eps)
        return (diff / denom).mean()
