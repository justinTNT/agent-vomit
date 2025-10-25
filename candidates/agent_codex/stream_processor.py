from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, Optional, Tuple

import torch
from torch import nn

from .utils import warn_on_unused_kwargs


class StreamProcessor(nn.Module):
    """Processes temporal data streams with sliding windows and reducers."""

    def __init__(
        self,
        window_size: float = 1000.0,
        stride: float = 250.0,
        reducer: str = "mean",
        max_buffer: int = 512,
        **kwargs,
    ) -> None:
        super().__init__()
        warn_on_unused_kwargs(kwargs)

        if window_size <= 0 or stride <= 0:
            raise ValueError("window_size and stride must be positive")
        if reducer not in {"mean", "sum", "last"}:
            raise ValueError("reducer must be one of {'mean', 'sum', 'last'}")

        self.window_size = float(window_size)
        self.stride = float(stride)
        self.reducer = reducer
        self.max_buffer = int(max_buffer)

        self.buffer: Deque[Tuple[float, torch.Tensor]] = deque()
        self.last_emit: Optional[float] = None

    def forward(
        self,
        data: torch.Tensor,
        timestamp: Optional[float] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not isinstance(data, torch.Tensor):
            raise TypeError("data must be a torch.Tensor")
        ts = float(timestamp) if timestamp is not None else time.time() * 1000.0
        self.buffer.append((ts, data.detach().clone()))

        while len(self.buffer) > self.max_buffer:
            self.buffer.popleft()

        window_start = ts - self.window_size
        while self.buffer and self.buffer[0][0] < window_start:
            self.buffer.popleft()

        if self.last_emit is not None and ts - self.last_emit < self.stride:
            return None

        if not self.buffer:
            return None

        stacked = torch.stack([item for _, item in self.buffer], dim=0)
        if self.reducer == "mean":
            aggregated = stacked.mean(dim=0)
        elif self.reducer == "sum":
            aggregated = stacked.sum(dim=0)
        else:
            aggregated = self.buffer[-1][1]

        self.last_emit = ts
        return {
            "timestamp": ts,
            "aggregated": aggregated,
            "count": torch.tensor(len(self.buffer), dtype=torch.long),
        }

    def clear(self) -> None:
        self.buffer.clear()
        self.last_emit = None
