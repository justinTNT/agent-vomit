"""
MultiScaleSTFTLoss - Multi-resolution spectral loss for audio.

This module computes losses at multiple STFT resolutions with both
spectral and magnitude components, supporting different window functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Literal
import warnings
import math


class MultiScaleSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss for audio synthesis.
    
    Args:
        fft_sizes: List of FFT sizes for multi-scale analysis
        hop_sizes: List of hop sizes (must match fft_sizes length)
        win_lengths: List of window lengths (None to use fft_sizes)
        window_fn: Window function ('hann', 'hamming', 'blackman', 'bartlett')
        mag_weight: Weight for magnitude loss
        log_weight: Weight for log magnitude loss  
        spectral_weight: Weight for spectral convergence loss
        epsilon: Small value for numerical stability
        log_base: Base for logarithm (2, 10, or 'e')
        reduction: Reduction method ('mean', 'sum', 'none')
        **kwargs: Additional arguments
    """
    
    def __init__(
        self,
        fft_sizes: List[int] = [512, 1024, 2048],
        hop_sizes: Optional[List[int]] = None,
        win_lengths: Optional[List[int]] = None,
        window_fn: Literal['hann', 'hamming', 'blackman', 'bartlett'] = 'hann',
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        spectral_weight: float = 1.0,
        epsilon: float = 1e-8,
        log_base: Literal[2, 10, 'e'] = 'e',
        reduction: Literal['mean', 'sum', 'none'] = 'mean',
        **kwargs
    ):
        super().__init__()
        
        # Validate inputs
        if not fft_sizes:
            raise ValueError("fft_sizes must not be empty")
        for size in fft_sizes:
            if size <= 0 or (size & (size - 1)) != 0:
                raise ValueError(f"FFT size must be positive power of 2, got {size}")
        
        self.fft_sizes = fft_sizes
        self.num_scales = len(fft_sizes)
        
        # Set hop sizes (default to fft_size // 4)
        if hop_sizes is None:
            self.hop_sizes = [size // 4 for size in fft_sizes]
        else:
            if len(hop_sizes) != len(fft_sizes):
                raise ValueError("hop_sizes must match fft_sizes length")
            self.hop_sizes = hop_sizes
        
        # Set window lengths (default to fft_size)
        if win_lengths is None:
            self.win_lengths = fft_sizes.copy()
        else:
            if len(win_lengths) != len(fft_sizes):
                raise ValueError("win_lengths must match fft_sizes length")
            self.win_lengths = win_lengths
        
        # Validate weights
        if mag_weight < 0 or log_weight < 0 or spectral_weight < 0:
            raise ValueError("Weights must be non-negative")
        
        self.window_fn = window_fn
        self.mag_weight = mag_weight
        self.log_weight = log_weight
        self.spectral_weight = spectral_weight
        self.epsilon = epsilon
        self.log_base = log_base
        self.reduction = reduction
        
        # Create windows for each scale
        self.windows = nn.ParameterList()
        for win_length in self.win_lengths:
            window = self._create_window(win_length)
            self.windows.append(nn.Parameter(window, requires_grad=False))
        
        # Statistics
        self.register_buffer('total_batches_processed', torch.tensor(0, dtype=torch.long))
        
        # Handle unused kwargs
        if kwargs:
            warnings.warn(f"Unused arguments: {list(kwargs.keys())}")
    
    def _create_window(self, win_length: int) -> torch.Tensor:
        """Create window function."""
        if self.window_fn == 'hann':
            return torch.hann_window(win_length)
        elif self.window_fn == 'hamming':
            return torch.hamming_window(win_length)
        elif self.window_fn == 'blackman':
            return torch.blackman_window(win_length)
        elif self.window_fn == 'bartlett':
            return torch.bartlett_window(win_length)
        else:
            raise ValueError(f"Unknown window function: {self.window_fn}")
    
    def _stft(
        self, 
        x: torch.Tensor, 
        fft_size: int, 
        hop_size: int, 
        win_length: int, 
        window: torch.Tensor
    ) -> torch.Tensor:
        """Compute STFT with given parameters."""
        # Move window to correct device
        window = window.to(x.device)
        
        # Compute STFT
        stft = torch.stft(
            x,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_length,
            window=window,
            center=True,
            pad_mode='reflect',
            return_complex=True
        )
        
        return stft
    
    def _magnitude_loss(
        self, 
        pred_stft: torch.Tensor, 
        target_stft: torch.Tensor
    ) -> torch.Tensor:
        """Compute magnitude loss."""
        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)
        
        return F.l1_loss(pred_mag, target_mag, reduction='none')
    
    def _log_magnitude_loss(
        self, 
        pred_stft: torch.Tensor, 
        target_stft: torch.Tensor
    ) -> torch.Tensor:
        """Compute log magnitude loss."""
        pred_mag = torch.abs(pred_stft) + self.epsilon
        target_mag = torch.abs(target_stft) + self.epsilon
        
        if self.log_base == 2:
            pred_log_mag = torch.log2(pred_mag)
            target_log_mag = torch.log2(target_mag)
        elif self.log_base == 10:
            pred_log_mag = torch.log10(pred_mag)
            target_log_mag = torch.log10(target_mag)
        else:  # natural log
            pred_log_mag = torch.log(pred_mag)
            target_log_mag = torch.log(target_mag)
        
        return F.l1_loss(pred_log_mag, target_log_mag, reduction='none')
    
    def _spectral_convergence_loss(
        self, 
        pred_stft: torch.Tensor, 
        target_stft: torch.Tensor
    ) -> torch.Tensor:
        """Compute spectral convergence loss."""
        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)
        
        # Frobenius norm of difference over Frobenius norm of target
        diff_norm = torch.norm(pred_mag - target_mag, p='fro', dim=(-2, -1))
        target_norm = torch.norm(target_mag, p='fro', dim=(-2, -1)) + self.epsilon
        
        return diff_norm / target_norm
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-scale STFT loss.
        
        Args:
            pred: Predicted audio of shape (batch, time) or (batch, channels, time)
            target: Target audio of shape (batch, time) or (batch, channels, time)
            
        Returns:
            Dictionary containing:
                - loss: Total loss
                - mag_loss: Magnitude loss component
                - log_mag_loss: Log magnitude loss component
                - spectral_loss: Spectral convergence loss
                - scale_losses: Individual losses per scale
        """
        if not isinstance(pred, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise TypeError("pred and target must be torch.Tensors")
        
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
        
        # Handle multi-channel by flattening
        if pred.dim() == 3:  # (batch, channels, time)
            batch_size, n_channels, length = pred.shape
            pred = pred.reshape(batch_size * n_channels, length)
            target = target.reshape(batch_size * n_channels, length)
        elif pred.dim() == 2:  # (batch, time)
            batch_size = pred.shape[0]
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {pred.dim()}D")
        
        device = pred.device
        self.total_batches_processed += 1
        
        # Compute losses at each scale
        scale_mag_losses = []
        scale_log_mag_losses = []
        scale_spectral_losses = []
        
        for i, (fft_size, hop_size, win_length, window) in enumerate(
            zip(self.fft_sizes, self.hop_sizes, self.win_lengths, self.windows)
        ):
            # Compute STFTs
            pred_stft = self._stft(pred, fft_size, hop_size, win_length, window)
            target_stft = self._stft(target, fft_size, hop_size, win_length, window)
            
            # Compute individual losses
            if self.mag_weight > 0:
                mag_loss = self._magnitude_loss(pred_stft, target_stft)
                mag_loss = mag_loss.mean(dim=(-2, -1))  # Average over freq and time
                scale_mag_losses.append(mag_loss)
            
            if self.log_weight > 0:
                log_mag_loss = self._log_magnitude_loss(pred_stft, target_stft)
                log_mag_loss = log_mag_loss.mean(dim=(-2, -1))
                scale_log_mag_losses.append(log_mag_loss)
            
            if self.spectral_weight > 0:
                spectral_loss = self._spectral_convergence_loss(pred_stft, target_stft)
                scale_spectral_losses.append(spectral_loss)
        
        # Stack and average across scales
        if scale_mag_losses:
            mag_losses = torch.stack(scale_mag_losses).mean(dim=0)
        else:
            mag_losses = torch.zeros(batch_size, device=device)
        
        if scale_log_mag_losses:
            log_mag_losses = torch.stack(scale_log_mag_losses).mean(dim=0)
        else:
            log_mag_losses = torch.zeros(batch_size, device=device)
        
        if scale_spectral_losses:
            spectral_losses = torch.stack(scale_spectral_losses).mean(dim=0)
        else:
            spectral_losses = torch.zeros(batch_size, device=device)
        
        # Combine losses with weights
        total_loss = (
            self.mag_weight * mag_losses +
            self.log_weight * log_mag_losses +
            self.spectral_weight * spectral_losses
        )
        
        # Apply reduction
        if self.reduction == 'mean':
            total_loss = total_loss.mean()
            mag_losses = mag_losses.mean()
            log_mag_losses = log_mag_losses.mean()
            spectral_losses = spectral_losses.mean()
        elif self.reduction == 'sum':
            total_loss = total_loss.sum()
            mag_losses = mag_losses.sum()
            log_mag_losses = log_mag_losses.sum()
            spectral_losses = spectral_losses.sum()
        
        # Prepare scale losses for output
        scale_losses = []
        for i in range(self.num_scales):
            scale_loss = {
                'fft_size': self.fft_sizes[i],
                'mag_loss': scale_mag_losses[i].mean() if scale_mag_losses else torch.tensor(0.0, device=device),
                'log_mag_loss': scale_log_mag_losses[i].mean() if scale_log_mag_losses else torch.tensor(0.0, device=device),
                'spectral_loss': scale_spectral_losses[i].mean() if scale_spectral_losses else torch.tensor(0.0, device=device)
            }
            scale_losses.append(scale_loss)
        
        return {
            'loss': total_loss,
            'mag_loss': mag_losses,
            'log_mag_loss': log_mag_losses,
            'spectral_loss': spectral_losses,
            'scale_losses': scale_losses
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration."""
        return {
            'fft_sizes': self.fft_sizes,
            'hop_sizes': self.hop_sizes,
            'win_lengths': self.win_lengths,
            'window_fn': self.window_fn,
            'mag_weight': self.mag_weight,
            'log_weight': self.log_weight,
            'spectral_weight': self.spectral_weight,
            'num_scales': self.num_scales,
            'total_batches_processed': self.total_batches_processed.item()
        }