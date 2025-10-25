from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .utils import warn_on_unused_kwargs


class MemoryBank(nn.Module):
    """Differentiable memory with FIFO replacement and similarity queries."""

    def __init__(
        self,
        feature_dim: int = 128,
        bank_size: int = 1024,
        temperature: float = 0.07,
        normalize: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        warn_on_unused_kwargs(kwargs)

        if bank_size < 1:
            raise ValueError("bank_size must be >= 1")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.feature_dim = feature_dim
        self.bank_size = bank_size
        self.temperature = temperature
        self.normalize = normalize

        memory = torch.zeros(bank_size, feature_dim, dtype=torch.float32)
        self.register_buffer("memory", memory)
        self.register_buffer("ptr", torch.tensor(0, dtype=torch.long))

    def add(self, vectors: torch.Tensor) -> None:
        if vectors.dim() != 2 or vectors.size(1) != self.feature_dim:
            raise ValueError(
                f"vectors must have shape (batch, {self.feature_dim})"
            )
        vectors = vectors.detach()
        if self.normalize:
            vectors = torch.nn.functional.normalize(vectors, dim=-1)

        batch = vectors.size(0)
        ptr = int(self.ptr.item())
        end = ptr + batch
        if end <= self.bank_size:
            self.memory[ptr:end] = vectors
        else:
            first = self.bank_size - ptr
            self.memory[ptr:] = vectors[:first]
            self.memory[: end % self.bank_size] = vectors[first:]
        self.ptr.fill_(end % self.bank_size)

    def forward(self, query: torch.Tensor, top_k: int = 5) -> Dict[str, torch.Tensor]:
        if query.dim() != 2 or query.size(1) != self.feature_dim:
            raise ValueError(
                f"query must have shape (batch, {self.feature_dim})"
            )
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if self.normalize:
            query = torch.nn.functional.normalize(query, dim=-1)

        similarity = torch.matmul(query, self.memory.t()) / self.temperature
        top_k = min(top_k, self.bank_size)
        scores, indices = torch.topk(similarity, k=top_k, dim=-1)
        retrieved = self.memory[indices]
        return {
            "scores": scores,
            "indices": indices,
            "retrieved": retrieved,
        }

    def clear(self) -> None:
        self.memory.zero_()
        self.ptr.zero_()
