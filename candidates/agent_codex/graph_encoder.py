from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .utils import warn_on_unused_kwargs


class GraphEncoder(nn.Module):
    """Graph encoder using simple message passing layers."""

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        aggregator: str = "mean",
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()
        warn_on_unused_kwargs(kwargs)

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if aggregator not in {"mean", "sum", "max"}:
            raise ValueError("aggregator must be one of {'mean', 'sum', 'max'}")

        self.aggregator = aggregator
        self.dropout = nn.Dropout(dropout)

        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layer = nn.Linear(in_dim, hidden_dim)
            layers.append(layer)
            in_dim = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.activation = nn.ReLU()

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if node_features.dim() != 2:
            raise ValueError("node_features must be 2D (num_nodes, feature_dim)")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape (2, num_edges)")

        num_nodes = node_features.size(0)
        if node_mask is None:
            node_mask = torch.ones(num_nodes, dtype=torch.bool, device=node_features.device)
        if node_mask.size(0) != num_nodes:
            raise ValueError("node_mask must have length equal to num_nodes")

        x = node_features
        for layer, norm in zip(self.layers, self.norms):
            messages = self._aggregate(x, edge_index, num_nodes)
            x = layer(messages)
            x = self.activation(x)
            x = norm(x)
            x = self.dropout(x)

        masked = x[node_mask]
        if masked.numel() == 0:
            graph_embedding = torch.zeros(x.size(1), device=x.device)
        else:
            graph_embedding = masked.mean(dim=0)

        return {
            "node_embeddings": x,
            "graph_embedding": graph_embedding,
        }

    def _aggregate(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        src, dst = edge_index
        if src.numel() == 0:
            return node_embeddings

        messages = node_embeddings[src]
        aggregated = torch.zeros_like(node_embeddings)
        aggregated.index_add_(0, dst, messages)

        if self.aggregator == "mean":
            deg = torch.zeros(num_nodes, device=node_embeddings.device)
            deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32, device=node_embeddings.device))
            deg = deg.clamp(min=1.0).unsqueeze(-1)
            aggregated = aggregated / deg
        elif self.aggregator == "max":
            aggregated = node_embeddings.new_full(
                (num_nodes, node_embeddings.size(1)), float("-inf")
            )
            index = dst.unsqueeze(-1).expand_as(messages)
            if hasattr(aggregated, "scatter_reduce_"):
                aggregated.scatter_reduce_(0, index, messages, reduce="amax", include_self=True)
            else:
                for i in range(messages.size(0)):
                    node_idx = int(dst[i])
                    aggregated[node_idx] = torch.maximum(
                        aggregated[node_idx], messages[i]
                    )
        return aggregated
