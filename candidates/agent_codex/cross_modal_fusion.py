from __future__ import annotations

from typing import Dict, Iterable, Mapping

import torch
from torch import nn

from .utils import warn_on_unused_kwargs


class CrossModalFusion(nn.Module):
    """Fuse multiple modality embeddings with configurable strategy."""

    def __init__(
        self,
        modality_dims: Mapping[str, int],
        hidden_dim: int = 256,
        fusion_type: str = "gated",
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()
        warn_on_unused_kwargs(kwargs)

        if not modality_dims:
            raise ValueError("modality_dims must contain at least one modality")
        fusion_type = fusion_type.lower()
        if fusion_type not in {"gated", "mean", "concat"}:
            raise ValueError(
                "fusion_type must be one of {'gated', 'mean', 'concat'}"
            )
        self.fusion_type = fusion_type
        self.modalities = tuple(modality_dims.keys())

        self.projections = nn.ModuleDict()
        for name, dim in modality_dims.items():
            self.projections[name] = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        if fusion_type == "gated":
            self.gate_mlp = nn.Sequential(
                nn.Linear(hidden_dim * len(self.modalities), hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, len(self.modalities)),
            )
        elif fusion_type == "concat":
            self.concat_proj = nn.Sequential(
                nn.Linear(hidden_dim * len(self.modalities), hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if set(inputs.keys()) != set(self.modalities):
            missing = set(self.modalities) - set(inputs.keys())
            extra = set(inputs.keys()) - set(self.modalities)
            raise KeyError(
                f"Expected modalities {self.modalities}; missing={missing}, extra={extra}"
            )

        projections = []
        for name in self.modalities:
            tensor = inputs[name]
            if tensor.dim() != 2:
                raise ValueError(
                    f"Input for modality '{name}' must be 2D (batch, features)"
                )
            projections.append(self.projections[name](tensor))

        hidden = torch.stack(projections, dim=1)
        if self.fusion_type == "mean":
            fused = hidden.mean(dim=1)
            weights = torch.full(
                (hidden.size(0), len(self.modalities)), 1.0 / len(self.modalities), device=hidden.device
            )
        elif self.fusion_type == "gated":
            concat = hidden.flatten(start_dim=1)
            logits = self.gate_mlp(concat)
            weights = torch.softmax(logits, dim=-1)
            fused = (hidden * weights.unsqueeze(-1)).sum(dim=1)
        else:  # concat
            concat = hidden.flatten(start_dim=1)
            fused = self.concat_proj(concat)
            weights = torch.softmax(hidden.norm(dim=-1), dim=-1)

        fused = self.output_norm(fused)
        weight_map = {
            name: weights[:, idx]
            for idx, name in enumerate(self.modalities)
        }
        return {
            "fused": fused,
            "weights": weight_map,
        }
