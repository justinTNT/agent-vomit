from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from .utils import warn_on_unused_kwargs


class AutoEncoder(nn.Module):
    """Autoencoder with optional variational latent space."""

    def __init__(
        self,
        input_dim: int = 784,
        latent_dim: int = 64,
        hidden_dims: List[int] | None = None,
        variational: bool = False,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        warn_on_unused_kwargs(kwargs)

        hidden_dims = hidden_dims or [256, 128]
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")

        self.input_dim = input_dim
        encoder_layers: List[nn.Module] = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.extend(
                [nn.Linear(prev_dim, dim), nn.GELU(), nn.Dropout(dropout)]
            )
            prev_dim = dim
        self.encoder = nn.Sequential(*encoder_layers)

        self.variational = variational
        if variational:
            self.latent_mu = nn.Linear(hidden_dims[-1], latent_dim)
            self.latent_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        else:
            self.latent_projection = nn.Linear(hidden_dims[-1], latent_dim)

        decoder_layers: List[nn.Module] = []
        prev_dim = latent_dim
        for dim in reversed(hidden_dims):
            decoder_layers.extend(
                [nn.Linear(prev_dim, dim), nn.GELU(), nn.Dropout(dropout)]
            )
            prev_dim = dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(
                f"Input must have shape (batch, {self.encoder[0].in_features})"
            )

        encoded = self.encoder(x)
        if self.variational:
            mu = self.latent_mu(encoded)
            logvar = self.latent_logvar(encoded)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            latent = mu + eps * std
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        else:
            latent = self.latent_projection(encoded)
            mu = latent
            logvar = torch.zeros_like(latent)
            kl_div = torch.zeros(latent.size(0), device=latent.device)

        reconstruction = self.decoder(latent)
        return {
            "reconstruction": reconstruction,
            "latent": latent,
            "mu": mu,
            "logvar": logvar,
            "kl_loss": kl_div,
        }
