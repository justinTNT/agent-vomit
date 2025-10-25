from __future__ import annotations

from typing import Optional, Tuple, Dict

import torch
from torch import nn

from .utils import warn_on_unused_kwargs


class TransformerBlock(nn.Module):
    """Standard transformer encoder block with robust mask handling."""

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()
        warn_on_unused_kwargs(kwargs)

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.d_model = d_model
        self.n_heads = n_heads

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"Expected input with 3 dims (batch, seq, feature), got {x.shape}"
            )

        attn_mask: Optional[torch.Tensor] = None
        key_padding_mask: Optional[torch.Tensor] = None
        if mask is not None:
            masks = self._prepare_masks(mask, x)
            attn_mask = masks.get("attn_mask")
            key_padding_mask = masks.get("key_padding_mask")

        attn_output, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        ff_output = self.linear2(self.dropout_ff(self.activation(self.linear1(x))))
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        return x

    def _prepare_masks(
        self,
        mask: torch.Tensor,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        mask = mask.to(device)

        result: Dict[str, torch.Tensor] = {}

        if mask.dtype == torch.bool:
            if mask.dim() == 2:
                if mask.shape != (batch, seq_len):
                    raise ValueError(
                        f"Boolean mask must have shape (batch, seq), got {mask.shape}"
                    )
                result["key_padding_mask"] = ~mask
            elif mask.dim() == 3:
                if mask.shape != (batch, seq_len, seq_len):
                    raise ValueError(
                        "Boolean attention mask must match (batch, seq, seq), "
                        f"got {mask.shape}"
                    )
                additive = torch.zeros_like(mask, dtype=dtype)
                additive = additive.masked_fill(~mask, torch.finfo(dtype).min)
                result["attn_mask"] = self._expand_attn_mask(additive, batch, dtype)
            else:
                raise ValueError(
                    f"Unsupported boolean mask shape {mask.shape}, dims={mask.dim()}"
                )
        else:
            if mask.dim() == 2:
                if mask.shape != (batch, seq_len):
                    raise ValueError(
                        f"Additive mask must have shape (batch, seq), got {mask.shape}"
                    )
                if torch.isneginf(mask).any():
                    key_padding = torch.isneginf(mask)
                else:
                    key_padding = mask <= -1e4
                result["key_padding_mask"] = key_padding.to(torch.bool)
            elif mask.dim() == 3:
                if mask.shape != (batch, seq_len, seq_len):
                    raise ValueError(
                        "Additive attention mask must match (batch, seq, seq), "
                        f"got {mask.shape}"
                    )
                result["attn_mask"] = self._expand_attn_mask(
                    mask.to(dtype), batch, dtype
                )
            else:
                raise ValueError(
                    f"Unsupported mask dimensions {mask.dim()} for additive mask"
                )

        return result

    def _expand_attn_mask(
        self,
        mask: torch.Tensor,
        batch: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        seq_len = mask.size(-1)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).expand(batch, seq_len, seq_len)
        if mask.shape[0] != batch:
            raise ValueError(
                f"Attention mask batch dimension {mask.shape[0]} does not match {batch}"
            )
        mask = mask.reshape(batch, 1, seq_len, seq_len)
        mask = mask.repeat_interleave(self.n_heads, dim=1)
        mask = mask.reshape(batch * self.n_heads, seq_len, seq_len)
        return mask.to(dtype)
