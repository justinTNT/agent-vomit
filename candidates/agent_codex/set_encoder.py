from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from .utils import warn_on_unused_kwargs


class SetEncoder(nn.Module):
    """Permutation-invariant encoder based on DeepSets."""

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 3,
        pooling: str = "mean",
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()
        warn_on_unused_kwargs(kwargs)

        if pooling not in {"mean", "sum", "max"}:
            raise ValueError("pooling must be one of {'mean', 'sum', 'max'}")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.pooling = pooling

        phi_layers: List[nn.Module] = []
        prev_dim = input_dim
        for _ in range(num_layers):
            phi_layers.extend(
                [nn.Linear(prev_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            )
            prev_dim = hidden_dim
        self.phi = nn.Sequential(*phi_layers)

        rho_layers: List[nn.Module] = []
        prev_dim = hidden_dim
        for _ in range(num_layers - 1):
            rho_layers.extend(
                [nn.Linear(prev_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            )
            prev_dim = hidden_dim
        rho_layers.append(nn.Linear(prev_dim, output_dim))
        self.rho = nn.Sequential(*rho_layers)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError("Input must be (batch, set_size, feature_dim)")

        batch, set_size, _ = x.shape
        if mask is None:
            mask = torch.ones(batch, set_size, dtype=torch.bool, device=x.device)
        if mask.shape != (batch, set_size):
            raise ValueError("mask must match (batch, set_size)")

        masked_x = x * mask.unsqueeze(-1)
        element_embeddings = self.phi(masked_x)

        if self.pooling == "mean":
            pooled = element_embeddings.sum(dim=1)
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = pooled / denom
        elif self.pooling == "sum":
            pooled = element_embeddings.sum(dim=1)
        else:
            masked = element_embeddings.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            pooled, _ = masked.max(dim=1)

        set_embedding = self.norm(self.rho(pooled))
        return {
            "set_embedding": set_embedding,
            "element_embeddings": element_embeddings,
        }
