from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .utils import warn_on_unused_kwargs


class CausalConv1d(nn.Module):
    """Causal convolution layer supporting streaming inference."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
        bias: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        warn_on_unused_kwargs(kwargs)

        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        if stride < 1:
            raise ValueError("stride must be positive")

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.receptive_field = (kernel_size - 1) * dilation + 1

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            bias=bias,
        )

        self.register_buffer("_cache", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("Input must be (batch, channels, time)")

        padding = (self.kernel_size - 1) * self.dilation
        if padding > 0:
            cache = self._ensure_cache(x, padding)
            x_cat = torch.cat([cache, x], dim=-1)
            out = self.conv(x_cat)
            self._cache = x[:, :, -padding:].detach()
        else:
            out = self.conv(x)
            self._cache = None

        expected = (x.size(-1) + self.stride - 1) // self.stride
        if out.size(-1) > expected:
            out = out[..., -expected:]
        return out

    def _ensure_cache(self, x: torch.Tensor, padding: int) -> torch.Tensor:
        cache = self._cache
        if cache is None or cache.size(0) != x.size(0) or cache.size(1) != x.size(1):
            cache = x.new_zeros(x.size(0), x.size(1), padding)
        elif cache.size(-1) != padding:
            if cache.size(-1) > padding:
                cache = cache[..., -padding:]
            else:
                extra = x.new_zeros(x.size(0), x.size(1), padding - cache.size(-1))
                cache = torch.cat([cache, extra], dim=-1)
        return cache

    def reset(self) -> None:
        self._cache = None
