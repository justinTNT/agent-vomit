from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from .utils import warn_on_unused_kwargs


class ViTPatchEncoder(nn.Module):
    """Vision Transformer style patch embedding with class token."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        include_cls: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        warn_on_unused_kwargs(kwargs)

        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.include_cls = include_cls

        num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.register_buffer(
            "position_embeddings",
            torch.zeros(num_patches + (1 if include_cls else 0), embed_dim),
        )
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)
        if include_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.register_parameter("cls_token", None)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        if images.dim() != 4:
            raise ValueError("images must have shape (batch, channels, height, width)")
        if images.size(1) != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {images.size(1)}")
        if images.size(2) != self.image_size or images.size(3) != self.image_size:
            raise ValueError(
                f"Expected images with spatial size ({self.image_size}, {self.image_size})"
            )

        patches = self.proj(images)
        batch_size, embed_dim, height, width = patches.shape
        patches = patches.flatten(2).transpose(1, 2)

        if self.include_cls:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            tokens = torch.cat([cls_tokens, patches], dim=1)
        else:
            tokens = patches

        pos_embed = self.position_embeddings[: tokens.size(1)].unsqueeze(0)
        tokens = tokens + pos_embed

        return {
            "tokens": tokens,
            "position_embeddings": pos_embed,
            "cls_token": tokens[:, 0:1] if self.include_cls else None,
        }
