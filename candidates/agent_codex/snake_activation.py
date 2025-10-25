from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .utils import warn_on_unused_kwargs


class SnakeActivation(nn.Module):
    """Snake activation: x + (1/alpha) * sin^2(alpha * x)."""

    def __init__(
        self,
        n_channels: int = 1,
        alpha_init: float = 1.0,
        learnable: bool = True,
        shared_alpha: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        warn_on_unused_kwargs(kwargs)

        if n_channels <= 0:
            raise ValueError("n_channels must be positive")

        self.n_channels = n_channels
        self.learnable = learnable
        self.shared_alpha = shared_alpha

        alpha_shape = (1,) if shared_alpha else (n_channels,)
        alpha_tensor = torch.full(alpha_shape, float(alpha_init), dtype=torch.float32)

        if learnable:
            self.alpha = nn.Parameter(alpha_tensor)
        else:
            self.register_buffer("alpha", alpha_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() not in (2, 3):
            raise ValueError("SnakeActivation expects a 2D or 3D tensor input")

        expected_channels = self.n_channels
        channel_dim = 1 if x.dim() == 3 else 1
        if x.size(channel_dim) != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} channels, got {x.size(channel_dim)}"
            )

        alpha = self.alpha
        if not self.learnable:
            alpha = alpha.detach()
        alpha = alpha.to(dtype=x.dtype, device=x.device)

        if self.shared_alpha:
            if x.dim() == 3:
                alpha = alpha.view(1, 1, 1)
            else:
                alpha = alpha.view(1, 1)
        else:
            if x.dim() == 3:
                alpha = alpha.view(1, -1, 1)
            else:
                alpha = alpha.view(1, -1)

        safe_alpha = torch.clamp(alpha, min=1e-6)
        return x + torch.sin(alpha * x) ** 2 / safe_alpha
