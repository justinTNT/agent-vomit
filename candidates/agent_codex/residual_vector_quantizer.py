from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from .utils import warn_on_unused_kwargs


class ResidualVectorQuantizer(nn.Module):
    """Hierarchical residual vector quantization module."""

    def __init__(
        self,
        embedding_dim: int = 128,
        codebook_size: int = 512,
        num_quantizers: int = 4,
        commitment_cost: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__()
        warn_on_unused_kwargs(kwargs)

        if num_quantizers < 1:
            raise ValueError("num_quantizers must be >= 1")
        if codebook_size < 1:
            raise ValueError("codebook_size must be >= 1")

        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers
        self.commitment_cost = commitment_cost

        self.codebooks = nn.ParameterList(
            [
                nn.Parameter(torch.randn(codebook_size, embedding_dim) * 0.1)
                for _ in range(num_quantizers)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        if inputs.dim() != 3 or inputs.size(-1) != self.embedding_dim:
            raise ValueError(
                f"inputs must be (batch, length, {self.embedding_dim})"
            )

        residual = inputs
        quantized_outputs: List[torch.Tensor] = []
        codes: List[torch.Tensor] = []
        loss = inputs.new_tensor(0.0)

        for codebook in self.codebooks:
            distances = self._compute_distances(residual, codebook)
            encoding_indices = torch.argmin(distances, dim=-1)
            codes.append(encoding_indices)

            encodings = torch.nn.functional.one_hot(
                encoding_indices, num_classes=self.codebook_size
            ).type_as(inputs)
            quantized = torch.matmul(encodings, codebook)

            quantized_outputs.append(quantized)

            loss = loss + self._commitment_loss(residual, quantized)
            residual = residual - quantized.detach()

        quantized_total = torch.stack(quantized_outputs, dim=0).sum(dim=0)
        quantized_sum = inputs + (quantized_total - inputs).detach()

        stacked_codes = torch.stack(codes, dim=0)
        return {
            "quantized": quantized_sum,
            "codes": stacked_codes,
            "loss": loss / self.num_quantizers,
        }

    def _compute_distances(self, residual: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        # residual: (batch, length, dim), codebook: (codebook_size, dim)
        residual_sq = residual.pow(2).sum(dim=-1, keepdim=True)
        codebook_sq = codebook.pow(2).sum(dim=1)
        inner = torch.matmul(residual, codebook.t())
        distances = residual_sq + codebook_sq.unsqueeze(0).unsqueeze(0) - 2 * inner
        return distances

    def _commitment_loss(self, residual: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
        return (
            torch.mean((quantized.detach() - residual) ** 2)
            + self.commitment_cost * torch.mean((quantized - residual.detach()) ** 2)
        )
