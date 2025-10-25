from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from .transformer_block import TransformerBlock
from .utils import warn_on_unused_kwargs


class PositionalEncoding(nn.Module):
    """Sine-cosine positional encoding with dynamic length support."""

    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, length: int) -> torch.Tensor:
        if length > self.pe.size(0):
            raise ValueError(
                f"Requested positional encoding length {length} exceeds maximum {self.pe.size(0)}"
            )
        return self.pe[:length]


class SequenceEncoder(nn.Module):
    """Token embedding + transformer encoder stack with pooled output."""

    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = 512,
        max_len: int = 512,
        num_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        pad_idx: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        warn_on_unused_kwargs(kwargs)

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.pad_idx = int(pad_idx)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_idx)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)]
        )
        for _ in range(num_layers - 1):
            self.layers.append(
                TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if tokens.dim() != 2:
            raise ValueError("tokens must have shape (batch, sequence_length)")

        batch, seq_len = tokens.shape
        device = tokens.device
        embeddings = self.embedding(tokens)
        pe = self.positional_encoding(seq_len).to(device)
        embeddings = embeddings + pe.unsqueeze(0)
        embeddings = self.dropout(embeddings)

        if attention_mask is None:
            attention_mask = tokens != self.pad_idx
        encoded = embeddings
        for layer in self.layers:
            encoded = layer(encoded, mask=attention_mask)
        encoded = self.layer_norm(encoded)

        pooled = encoded.masked_fill(~attention_mask.unsqueeze(-1), 0.0).sum(dim=1)
        denom = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = pooled / denom

        return {
            "encoded": encoded,
            "pooled": pooled,
            "mask": attention_mask,
            "embedding": embeddings,
        }
