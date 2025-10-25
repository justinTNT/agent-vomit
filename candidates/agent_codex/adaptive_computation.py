from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .utils import warn_on_unused_kwargs


class AdaptiveComputation(nn.Module):
    """Implements Adaptive Computation Time for dynamic-depth processing."""

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        max_steps: int = 6,
        halting_threshold: float = 0.99,
        **kwargs,
    ) -> None:
        super().__init__()
        warn_on_unused_kwargs(kwargs)

        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if halting_threshold <= 0 or halting_threshold > 1:
            raise ValueError("halting_threshold must be in (0, 1]")

        self.max_steps = max_steps
        self.halting_threshold = halting_threshold

        self.layers = nn.ModuleList()
        self.halting_projections = nn.ModuleList()
        for _ in range(max_steps):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                )
            )
            self.halting_projections.append(nn.Linear(hidden_dim, 1))
            input_dim = hidden_dim

        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.dim() != 2:
            raise ValueError("Input must have shape (batch, features)")

        batch_size = x.size(0)
        halting_prob = torch.zeros(batch_size, device=x.device)
        remainders = torch.zeros(batch_size, device=x.device)
        n_updates = torch.zeros(batch_size, device=x.device)
        previous_state = x
        accumulated = torch.zeros_like(previous_state)

        for step, (layer, halting_layer) in enumerate(
            zip(self.layers, self.halting_projections)
        ):
            transformed = layer(previous_state)
            p = torch.sigmoid(halting_layer(transformed)).squeeze(-1)

            still_running = halting_prob < 1.0
            new_halt = halting_prob + p * still_running.float()
            reached = new_halt >= self.halting_threshold

            halting_contrib = torch.where(
                reached,
                (self.halting_threshold - halting_prob).clamp(min=0.0),
                p * still_running.float(),
            )

            halting_prob = halting_prob + halting_contrib
            n_updates = n_updates + still_running.float()

            accumulated = accumulated + halting_contrib.unsqueeze(-1) * transformed
            previous_state = transformed

            if halting_prob.min() >= self.halting_threshold:
                break

        remaining = (1.0 - halting_prob).clamp(min=0.0)
        if remaining.any():
            accumulated = accumulated + remaining.unsqueeze(-1) * previous_state
            halting_prob = halting_prob + remaining

        output = self.final_norm(accumulated)
        return {
            "output": output,
            "halting_prob": halting_prob,
            "updates": n_updates,
        }
