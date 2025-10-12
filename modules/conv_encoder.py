from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn

from .utils import warn_on_unused_kwargs


class ConvEncoder(nn.Module):
    """Hierarchical CNN encoder that reports intermediate feature maps."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_layers: int = 4,
        **kwargs,
    ) -> None:
        super().__init__()
        warn_on_unused_kwargs(kwargs)

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_layers = num_layers

        blocks: List[nn.Sequential] = []
        current_in = in_channels
        for i in range(num_layers):
            out_channels = base_channels * (2 ** i)
            block = nn.Sequential(
                nn.Conv2d(current_in, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )
            blocks.append(block)
            current_in = out_channels

        self.blocks = nn.ModuleList(blocks)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        if x.dim() != 4:
            raise ValueError(
                f"Expected input with 4 dims (batch, channels, height, width), got {x.shape}"
            )
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.size(1)}"
            )

        features: List[torch.Tensor] = []
        current = x
        for block in self.blocks:
            current = block(current)
            features.append(current)

        pooled = self.global_pool(current).flatten(1)
        shape_info = {
            "input": tuple(x.shape),
            "output": tuple(current.shape),
        }

        return {
            "features": features,
            "pooled": pooled,
            "shape": shape_info,
        }
