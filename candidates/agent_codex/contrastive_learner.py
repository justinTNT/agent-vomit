from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .utils import warn_on_unused_kwargs


class ContrastiveLearner(nn.Module):
    """InfoNCE-based contrastive learner with memory queue."""

    def __init__(
        self,
        feature_dim: int = 128,
        queue_size: int = 4096,
        momentum: float = 0.99,
        temperature: float = 0.07,
        **kwargs,
    ) -> None:
        super().__init__()
        warn_on_unused_kwargs(kwargs)

        if not (0 < momentum < 1):
            raise ValueError("momentum must be in (0, 1)")
        if queue_size < feature_dim:
            raise ValueError("queue_size must be >= feature_dim")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.feature_dim = feature_dim
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature

        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        queue = torch.randn(queue_size, feature_dim)
        queue = torch.nn.functional.normalize(queue, dim=1)
        self.register_buffer("queue", queue)
        self.register_buffer("queue_ptr", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        keys = keys.detach().to(self.queue.device)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        end = ptr + batch_size
        if end <= self.queue_size:
            self.queue[ptr:end] = keys
        else:
            first = self.queue_size - ptr
            self.queue[ptr:] = keys[:first]
            self.queue[: end % self.queue_size] = keys[first:]
        self.queue_ptr.fill_(end % self.queue_size)

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> Dict[str, torch.Tensor]:
        if query.shape != key.shape:
            raise ValueError("query and key must share shape")
        if query.dim() != 2 or query.size(1) != self.feature_dim:
            raise ValueError(
                f"query/key must be (batch, {self.feature_dim})"
            )

        query = self.projector(query)
        key = self.projector(key)

        query = torch.nn.functional.normalize(query, dim=1)
        key = torch.nn.functional.normalize(key, dim=1)

        logits_pos = torch.sum(query * key, dim=1, keepdim=True)
        logits_neg = torch.mm(query, self.queue.t())
        logits = torch.cat([logits_pos, logits_neg], dim=1)
        logits = logits / self.temperature

        labels = torch.zeros(query.size(0), dtype=torch.long, device=query.device)

        self._dequeue_and_enqueue(key)

        return {
            "logits": logits,
            "labels": labels,
        }
