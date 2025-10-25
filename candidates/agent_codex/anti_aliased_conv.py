from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from .utils import warn_on_unused_kwargs


def _sinc_filter(kernel_size: int, cutoff: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd for symmetric filter")
    half = kernel_size // 2
    indices = torch.arange(-half, half + 1, device=device, dtype=dtype)
    filter_kernel = 2 * cutoff * torch.sinc(2 * cutoff * indices)
    window = torch.hann_window(kernel_size, dtype=dtype, device=device)
    filter_kernel = filter_kernel * window
    filter_kernel = filter_kernel / filter_kernel.sum()
    return filter_kernel


class AntiAliasedConv(nn.Module):
    """Strided convolution preceded by configurable low-pass filtering."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 2,
        cutoff: Optional[float] = None,
        filter_size: int = 5,
        bias: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        warn_on_unused_kwargs(kwargs)

        if stride < 1:
            raise ValueError("stride must be >= 1")
        if filter_size % 2 == 0:
            raise ValueError("filter_size must be odd")

        self.stride = stride
        self.cutoff = cutoff if cutoff is not None else 0.5 / stride
        self.filter_size = filter_size

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=bias,
        )
        self.register_buffer("low_pass", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("Input must be (batch, channels, time)")

        if self.stride > 1:
            lp = self._ensure_filter(x)
            x = nn.functional.conv1d(
                x,
                lp,
                padding=self.filter_size // 2,
                groups=x.size(1),
            )
        return self.conv(x)

    def _ensure_filter(self, x: torch.Tensor) -> torch.Tensor:
        if self.low_pass is None or self.low_pass.size(0) != x.size(1):
            kernel = _sinc_filter(
                self.filter_size,
                cutoff=self.cutoff,
                device=x.device,
                dtype=x.dtype,
            )
            kernel = kernel.view(1, 1, -1)
            kernel = kernel.repeat(x.size(1), 1, 1)
            self.low_pass = kernel
        return self.low_pass
