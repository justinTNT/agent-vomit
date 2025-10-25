from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Tuple

import torch
from torch import nn

from .utils import warn_on_unused_kwargs


class StreamJoiner(nn.Module):
    """Temporal stream joiner with window buffering."""

    def __init__(
        self,
        join_type: str = "inner",
        window_size: float = 1000.0,
        timeout: float = 5.0,
        **kwargs,
    ) -> None:
        super().__init__()
        warn_on_unused_kwargs(kwargs)

        if join_type.lower() != "inner":
            raise ValueError("Currently only 'inner' join_type is supported")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        self.join_type = join_type.lower()
        self.window_size = float(window_size)
        self.timeout = float(timeout)
        self.buffers: Dict[str, Deque[Tuple[float, torch.Tensor]]] = defaultdict(deque)
        self.last_join_timestamp: Optional[float] = None

    def forward(
        self,
        stream_name: str,
        data: torch.Tensor,
        timestamp: Optional[float] = None,
    ) -> Optional[Dict[str, object]]:
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor data, got {type(data)!r}")

        ts = float(timestamp) if timestamp is not None else time.time() * 1000.0
        buffered = data.detach().clone()
        self.buffers[stream_name].append((ts, buffered))

        self._expire_old_entries(reference_time=ts)
        return self._attempt_join(reference_time=ts)

    def clear(self) -> None:
        for buffer in self.buffers.values():
            buffer.clear()
        self.last_join_timestamp = None

    # Internal helpers -------------------------------------------------

    def _expire_old_entries(self, reference_time: float) -> None:
        lower_bound = reference_time - self.window_size
        timeout_bound = reference_time - self.timeout
        for buffer in self.buffers.values():
            while buffer and (buffer[0][0] < lower_bound or buffer[0][0] < timeout_bound):
                buffer.popleft()

    def _attempt_join(self, reference_time: float) -> Optional[Dict[str, object]]:
        if not self.buffers:
            return None
        if any(len(buffer) == 0 for buffer in self.buffers.values()):
            return None

        candidate_ts = max(buffer[0][0] for buffer in self.buffers.values())
        matched: Dict[str, torch.Tensor] = {}
        matched_timestamps = []

        for name, buffer in self.buffers.items():
            match_idx: Optional[int] = None
            for idx, (ts, _) in enumerate(buffer):
                if abs(ts - candidate_ts) <= self.window_size:
                    match_idx = idx
                    break
            if match_idx is None:
                return None
            for _ in range(match_idx):
                buffer.popleft()
            ts, tensor = buffer.popleft()
            matched[name] = tensor
            matched_timestamps.append(ts)

        join_timestamp = sum(matched_timestamps) / len(matched_timestamps)
        self.last_join_timestamp = join_timestamp

        return {
            "timestamp": join_timestamp,
            "data": matched,
            "streams": tuple(matched.keys()),
        }
