from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from .utils import warn_on_unused_kwargs


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(out_channels)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = out[..., : x.size(-1)]
        out = self.relu(self.dropout(out))
        out = self.conv2(out)
        out = out[..., : x.size(-1)]
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        out = self.norm(out + res)
        return self.relu(out)


class TimeSeriesEncoder(nn.Module):
    """Encodes temporal signals using dilated convolutions (TCN style)."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()
        warn_on_unused_kwargs(kwargs)

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        layers: List[TemporalBlock] = []
        channels = in_channels
        for i in range(num_layers):
            out_channels = base_channels * (2 ** i)
            layers.append(
                TemporalBlock(
                    channels,
                    out_channels,
                    kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                )
            )
            channels = out_channels
        self.network = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError("Input must be (batch, channels, length)")

        encoded = self.network(x)
        pooled = self.global_pool(encoded).squeeze(-1)
        return {
            "encoded": encoded,
            "pooled": pooled,
        }
