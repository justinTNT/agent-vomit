from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
from torch import nn

from .utils import warn_on_unused_kwargs


class DataSampler(nn.Module):
    """Advanced sampler supporting class-balanced and temperature scaling."""

    def __init__(
        self,
        class_counts: Optional[Dict[int, int]] = None,
        strategy: str = "balanced",
        temperature: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        warn_on_unused_kwargs(kwargs)

        self.strategy = strategy
        self.temperature = temperature
        self.update(class_counts or {})

    def update(self, class_counts: Dict[int, int]) -> None:
        if not class_counts:
            self.class_counts = {0: 1}
        else:
            self.class_counts = {
                int(cls): int(count)
                for cls, count in class_counts.items()
                if count > 0
            }
        total = sum(self.class_counts.values())
        self.class_probs = {
            cls: count / total
            for cls, count in self.class_counts.items()
        }

    def forward(
        self,
        labels: torch.Tensor,
        num_samples: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if labels.dim() != 1:
            raise ValueError("labels must be 1D tensor of class indices")
        if num_samples is None:
            num_samples = labels.size(0)

        class_weights = self._compute_weights()
        weights = torch.tensor(
            [class_weights.get(int(label.item()), 0.0) for label in labels],
            dtype=torch.float32,
            device=labels.device,
        )
        if weights.sum() == 0:
            weights = torch.ones_like(weights)
        weights = weights / weights.sum()
        indices = torch.multinomial(weights, num_samples, replacement=True, generator=generator)
        return indices

    def _compute_weights(self) -> Dict[int, float]:
        if self.strategy == "balanced":
            weights = {cls: 1.0 / count for cls, count in self.class_counts.items()}
        elif self.strategy == "temperature":
            weights = {
                cls: (count ** (-1.0 / max(self.temperature, 1e-6)))
                for cls, count in self.class_counts.items()
            }
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

        total = sum(weights.values())
        return {cls: weight / total for cls, weight in weights.items()}
