"""
ResidualVectorQuantizer - Hierarchical discrete representation learning.

This module implements multiple codebook levels with residual quantization
and commitment loss computation for learning discrete representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
import warnings
import math
from rave_config_system import RAVEConfig




# TODO: REFACTOR TO CONFIG-FIRST INTERFACE
# New signature: def __init__(self, config: RAVEConfig, dim: int, commitment_weight: float, ema_decay: Optional[float], epsilon: float, kmeans_init: bool, threshold_ema_dead_code: int, **kwargs):
# New assignments:
#         self.dim = dim
        self.codebook_size = config.quantization.codebook_size
        self.num_quantizers = config.quantization.num_quantizers
        self.commitment_weight = commitment_weight
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        self.kmeans_init = kmeans_init
        self.threshold_ema_dead_code = threshold_ema_dead_code
class ResidualVectorQuantizer(nn.Module):
    """
    Hierarchical vector quantization with residual learning.
    
    Args:
        dim: Dimension of the vectors to quantize
        codebook_size: Number of codes in each codebook
        num_quantizers: Number of residual quantization levels
        commitment_weight: Weight for commitment loss
        ema_decay: Decay for exponential moving average (None for no EMA)
        epsilon: Small value for numerical stability
        kmeans_init: Initialize codebooks with k-means
        threshold_ema_dead_code: Threshold for resetting dead codes
        **kwargs: Additional arguments
    """
    
    def __init__(
        self,
        dim: int,
        codebook_size: int = 1024,
        num_quantizers: int = 8,
        commitment_weight: float = 1.0,
        ema_decay: Optional[float] = 0.99,
        epsilon: float = 1e-5,
        kmeans_init: bool = False,
        threshold_ema_dead_code: int = 2,
        **kwargs
    ):
        super().__init__()
        
        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if codebook_size <= 0:
            raise ValueError(f"codebook_size must be positive, got {codebook_size}")
        if num_quantizers <= 0:
            raise ValueError(f"num_quantizers must be positive, got {num_quantizers}")
        if commitment_weight < 0:
            raise ValueError(f"commitment_weight must be non-negative, got {commitment_weight}")
        if ema_decay is not None and (ema_decay <= 0 or ema_decay > 1):
            raise ValueError(f"ema_decay must be in (0, 1], got {ema_decay}")
        
        self.dim = dim
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers
        self.commitment_weight = commitment_weight
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        self.kmeans_init = kmeans_init
        self.threshold_ema_dead_code = threshold_ema_dead_code
        
        # Initialize codebooks
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, dim))
            for _ in range(num_quantizers)
        ])
        
        # Initialize codebooks with unit variance
        for codebook in self.codebooks:
            codebook.data.uniform_(-1/codebook_size, 1/codebook_size)
        
        # EMA parameters if enabled
        if ema_decay is not None:
            self.register_buffer('ema_cluster_size', torch.zeros(num_quantizers, codebook_size))
            self.ema_w = nn.ParameterList([
                nn.Parameter(torch.zeros(codebook_size, dim), requires_grad=False)
                for _ in range(num_quantizers)
            ])
            self._ema_initted = False
        
        # Usage statistics
        self.register_buffer('code_usage', torch.zeros(num_quantizers, codebook_size, dtype=torch.long))
        self.register_buffer('total_quantizations', torch.tensor(0, dtype=torch.long))
        
        # Handle unused kwargs
        if kwargs:
            warnings.warn(f"Unused arguments: {list(kwargs.keys())}")
    
    def _quantize(
        self, 
        x: torch.Tensor, 
        codebook: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Quantize input using nearest neighbor in codebook."""
        # x: (batch, ..., dim)
        # codebook: (codebook_size, dim)
        
        # Flatten spatial dimensions
        x_flat = x.reshape(-1, self.dim)
        
        # Compute L2 distances
        distances = torch.cdist(x_flat, codebook)
        
        # Find nearest codes
        indices = distances.argmin(dim=-1)
        
        # Look up codes
        quantized_flat = F.embedding(indices, codebook)
        
        # Reshape to original shape
        quantized = quantized_flat.view_as(x)
        indices = indices.view(*x.shape[:-1])
        
        return quantized, indices
    
    def _update_ema(
        self, 
        x: torch.Tensor, 
        indices: torch.LongTensor, 
        level: int
    ):
        """Update EMA statistics for codebook."""
        if self.training and self.ema_decay is not None:
            # Initialize EMA on first call
            if not self._ema_initted:
                self.ema_cluster_size.data.copy_(torch.ones_like(self.ema_cluster_size))
                for i, codebook in enumerate(self.codebooks):
                    self.ema_w[i].data.copy_(codebook.data.clone())
                self._ema_initted = True
            
            # Flatten for easier computation
            x_flat = x.reshape(-1, self.dim)
            indices_flat = indices.view(-1)
            
            # One-hot encoding of indices
            onehot = F.one_hot(indices_flat, self.codebook_size).float()
            
            # Update cluster sizes
            cluster_size = onehot.sum(0)
            self.ema_cluster_size[level].mul_(self.ema_decay).add_(
                cluster_size, alpha=1 - self.ema_decay
            )
            
            # Update cluster centers
            sum_x = onehot.t() @ x_flat
            self.ema_w[level].mul_(self.ema_decay).add_(
                sum_x, alpha=1 - self.ema_decay
            )
            
            # Update codebook
            cluster_size = self.ema_cluster_size[level]
            n = cluster_size.sum()
            cluster_size = (cluster_size + self.epsilon) / (n + self.codebook_size * self.epsilon) * n
            
            embed_normalized = self.ema_w[level] / cluster_size.unsqueeze(1)
            self.codebooks[level].data.copy_(embed_normalized)
            
            # Reset dead codes
            dead_codes = cluster_size < self.threshold_ema_dead_code
            if dead_codes.any():
                # Replace dead codes with random vectors from input
                n_dead = dead_codes.sum().item()
                if x_flat.size(0) >= n_dead:
                    indices_to_replace = torch.randperm(x_flat.size(0))[:n_dead]
                    self.codebooks[level].data[dead_codes] = x_flat[indices_to_replace]
    
    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Apply hierarchical residual vector quantization.
        
        Args:
            x: Input tensor of shape (..., dim)
            
        Returns:
            Dictionary containing:
                - quantized: Final quantized output
                - indices: Indices for each quantization level
                - commitment_loss: Commitment loss value
                - quantized_levels: Quantized values at each level
                - residuals: Residual values at each level
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        
        if x.size(-1) != self.dim:
            raise ValueError(f"Expected last dimension {self.dim}, got {x.size(-1)}")
        
        device = x.device
        self.total_quantizations += 1
        
        # Store results for each level
        quantized_levels = []
        indices_list = []
        residuals = []
        
        # Start with input as residual
        residual = x
        quantized_sum = torch.zeros_like(x)
        
        # Apply residual quantization
        for level in range(self.num_quantizers):
            # Quantize current residual
            quantized, indices = self._quantize(residual, self.codebooks[level])
            
            # Update EMA if enabled
            if self.ema_decay is not None:
                self._update_ema(residual, indices, level)
            
            # Store results
            quantized_levels.append(quantized)
            indices_list.append(indices)
            residuals.append(residual.clone())
            
            # Update usage statistics
            if self.training:
                unique_indices, counts = indices.unique(return_counts=True)
                self.code_usage[level, unique_indices] += counts
            
            # Update quantized sum and residual for next level
            quantized_sum = quantized_sum + quantized
            residual = residual - quantized
        
        # Compute commitment loss
        commitment_loss = torch.tensor(0.0, device=device)
        if self.commitment_weight > 0:
            for i, (quantized, residual) in enumerate(zip(quantized_levels, residuals)):
                # Commitment loss encourages the input to be close to the quantized value
                commitment_loss = commitment_loss + F.mse_loss(
                    residual.detach(), 
                    quantized
                ) * self.commitment_weight
            commitment_loss = commitment_loss / self.num_quantizers
        
        # Straight-through estimator for gradients
        quantized_sum = x + (quantized_sum - x).detach()
        
        # Stack indices
        indices_stacked = torch.stack(indices_list, dim=0)  # (num_quantizers, ...)
        
        return {
            'quantized': quantized_sum,
            'indices': indices_stacked,
            'commitment_loss': commitment_loss,
            'quantized_levels': quantized_levels,
            'residuals': residuals
        }
    
    def decode(self, indices: torch.LongTensor) -> torch.Tensor:
        """
        Decode from indices to reconstructed vectors.
        
        Args:
            indices: Tensor of shape (num_quantizers, ...) containing code indices
            
        Returns:
            Decoded tensor of shape (..., dim)
        """
        if indices.size(0) != self.num_quantizers:
            raise ValueError(
                f"Expected first dimension {self.num_quantizers}, got {indices.size(0)}"
            )
        
        # Sum quantized values from all levels
        quantized_sum = None
        
        for level in range(self.num_quantizers):
            level_indices = indices[level]
            quantized = F.embedding(level_indices, self.codebooks[level])
            
            if quantized_sum is None:
                quantized_sum = quantized
            else:
                quantized_sum = quantized_sum + quantized
        
        return quantized_sum
    
    def get_codebook_usage(self) -> Dict[str, Any]:
        """Get statistics about codebook usage."""
        usage_stats = []
        
        for level in range(self.num_quantizers):
            level_usage = self.code_usage[level]
            used_codes = (level_usage > 0).sum().item()
            usage_rate = used_codes / self.codebook_size
            
            stats = {
                'level': level,
                'used_codes': used_codes,
                'total_codes': self.codebook_size,
                'usage_rate': usage_rate,
                'total_usage': level_usage.sum().item()
            }
            
            if level_usage.sum() > 0:
                stats['usage_entropy'] = -(
                    (level_usage.float() / level_usage.sum()) * 
                    (level_usage.float() / level_usage.sum()).log()
                ).nansum().item()
            else:
                stats['usage_entropy'] = 0.0
            
            usage_stats.append(stats)
        
        return {
            'per_level_stats': usage_stats,
            'total_quantizations': self.total_quantizations.item()
        }
    
    def reset_usage_stats(self):
        """Reset usage statistics."""
        self.code_usage.zero_()
        self.total_quantizations.zero_()