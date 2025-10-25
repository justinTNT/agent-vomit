from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from .utils import warn_on_unused_kwargs


class AttentionDecoder(nn.Module):
    """Autoregressive decoder block with self- and cross-attention."""

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
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if target.dim() != 3:
            raise ValueError(
                f"Expected target shape (batch, seq, feature), got {target.shape}"
            )
        if memory.dim() != 3 or memory.size(-1) != self.d_model:
            raise ValueError(
                "Memory must have shape (batch, seq, feature) with feature=d_model"
            )

        tgt_masks = self._prepare_masks(target, tgt_mask)
        mem_masks = self._prepare_masks(memory, memory_mask)

        attn_output, _ = self.self_attn(
            target,
            target,
            target,
            attn_mask=tgt_masks.get("attn_mask"),
            key_padding_mask=tgt_masks.get("key_padding_mask"),
            need_weights=False,
        )
        target = target + self.dropout(attn_output)
        target = self.norm1(target)

        cross_output, _ = self.cross_attn(
            target,
            memory,
            memory,
            attn_mask=mem_masks.get("attn_mask"),
            key_padding_mask=mem_masks.get("key_padding_mask"),
            need_weights=False,
        )
        target = target + self.dropout2(cross_output)
        target = self.norm2(target)

        ff_output = self.linear2(self.dropout3(self.activation(self.linear1(target))))
        target = target + ff_output
        target = self.norm3(target)
        return target

    def _prepare_masks(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if mask is None:
            return {}
        batch, seq_len, _ = x.shape

        result: Dict[str, torch.Tensor] = {}
        mask = mask.to(x.device)

        if mask.dtype == torch.bool:
            if mask.dim() == 2:
                if mask.shape != (batch, seq_len):
                    raise ValueError(
                        f"Boolean mask must be (batch, seq_len); got {mask.shape}"
                    )
                result["key_padding_mask"] = ~mask
            elif mask.dim() == 3:
                if mask.shape != (batch, seq_len, seq_len):
                    raise ValueError(
                        "Attention mask must be (batch, seq, seq) for boolean input"
                    )
                additive = torch.zeros_like(mask, dtype=x.dtype)
                additive = additive.masked_fill(~mask, torch.finfo(x.dtype).min)
                result["attn_mask"] = self._expand_attn_mask(additive, batch)
            else:
                raise ValueError(
                    f"Unsupported boolean mask with dims={mask.dim()}"
                )
        else:
            if mask.dim() == 2:
                if mask.shape != (batch, seq_len):
                    raise ValueError(
                        f"Additive mask must be (batch, seq); got {mask.shape}"
                    )
                padding = torch.isneginf(mask) | (mask <= -1e4)
                result["key_padding_mask"] = padding
            elif mask.dim() == 3:
                if mask.shape != (batch, seq_len, seq_len):
                    raise ValueError(
                        "Additive attention mask must be (batch, seq, seq)"
                    )
                result["attn_mask"] = self._expand_attn_mask(mask.to(x.dtype), batch)
            else:
                raise ValueError(f"Unsupported additive mask dims={mask.dim()}")
        return result

    def _expand_attn_mask(self, mask: torch.Tensor, batch: int) -> torch.Tensor:
        seq_len = mask.size(-1)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).expand(batch, seq_len, seq_len)
        if mask.shape[0] != batch:
            raise ValueError(
                f"Attention mask batch {mask.shape[0]} mismatch expected {batch}"
            )
        mask = mask.reshape(batch, 1, seq_len, seq_len)
        mask = mask.repeat_interleave(self.n_heads, dim=1)
        return mask.reshape(batch * self.n_heads, seq_len, seq_len)
