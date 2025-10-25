"""
CausalConv1d - Causal convolution for real-time audio processing.

This module implements causal padding with no future information leakage
and supports dilation for multi-scale temporal modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import warnings
import math


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution for real-time processing.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolution kernel
        stride: Stride of the convolution
        dilation: Dilation factor
        groups: Number of groups for grouped convolution
        bias: Whether to use bias
        padding_mode: Padding mode ('zeros', 'reflect', 'replicate')
        **kwargs: Additional arguments
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        **kwargs
    ):
        super().__init__()
        
        # Validate inputs
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if dilation <= 0:
            raise ValueError(f"dilation must be positive, got {dilation}")
        if groups <= 0 or in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError(f"Invalid groups: {groups}")
        if padding_mode not in ['zeros', 'reflect', 'replicate']:
            raise ValueError(f"Invalid padding_mode: {padding_mode}")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        
        # Calculate causal padding
        # For causal convolution, we need to pad such that output at time t
        # only depends on inputs at time t and before
        self.causal_padding = (kernel_size - 1) * dilation
        
        # Create convolution layer without padding (we'll add it manually)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,  # No padding in conv layer
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        
        # Buffer for streaming mode
        self.register_buffer(
            'streaming_buffer', 
            torch.zeros(1, in_channels, self.causal_padding)
        )
        self.streaming_mode = False
        
        # Statistics
        self.register_buffer('total_frames_processed', torch.tensor(0, dtype=torch.long))
        
        # Handle unused kwargs
        if kwargs:
            warnings.warn(f"Unused arguments: {list(kwargs.keys())}")
    
    def _apply_causal_padding(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal padding to input tensor."""
        if self.causal_padding == 0:
            return x
        
        # Pad only on the left side (past) for causality
        if self.padding_mode == 'zeros':
            return F.pad(x, (self.causal_padding, 0), mode='constant', value=0)
        elif self.padding_mode == 'reflect':
            # Reflect padding needs at least as many samples as padding
            if x.size(-1) > self.causal_padding:
                return F.pad(x, (self.causal_padding, 0), mode='reflect')
            else:
                # Fall back to replicate if not enough samples
                return F.pad(x, (self.causal_padding, 0), mode='replicate')
        else:  # replicate
            return F.pad(x, (self.causal_padding, 0), mode='replicate')
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply causal convolution.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
            
        Returns:
            Dictionary containing:
                - output: Convolved output
                - effective_kernel_size: Effective kernel size with dilation
                - receptive_field: Total receptive field size
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor (batch, channels, time), got {x.dim()}D")
        
        if x.size(1) != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {x.size(1)}")
        
        batch_size = x.size(0)
        device = x.device
        
        # Update statistics
        self.total_frames_processed += x.size(-1)
        
        if self.streaming_mode:
            # In streaming mode, use buffer for causality
            # Ensure buffer is correct size
            if self.streaming_buffer.size(0) != batch_size:
                self.streaming_buffer = self.streaming_buffer.expand(
                    batch_size, -1, -1
                ).contiguous()
            
            # Concatenate buffer with input
            x_padded = torch.cat([self.streaming_buffer, x], dim=-1)
            
            # Update buffer with most recent samples
            if x.size(-1) >= self.causal_padding:
                self.streaming_buffer = x[..., -self.causal_padding:].clone()
            else:
                # Shift buffer and append new samples
                shift = x.size(-1)
                self.streaming_buffer[..., :-shift] = self.streaming_buffer[..., shift:].clone()
                self.streaming_buffer[..., -shift:] = x.clone()
        else:
            # Apply causal padding
            x_padded = self._apply_causal_padding(x)
        
        # Apply convolution
        output = self.conv(x_padded)
        
        # Calculate receptive field information
        effective_kernel_size = (self.kernel_size - 1) * self.dilation + 1
        
        # For stacked layers, receptive field grows
        # This is the receptive field of a single layer
        receptive_field = effective_kernel_size
        
        return {
            'output': output,
            'effective_kernel_size': torch.tensor(effective_kernel_size, device=device),
            'receptive_field': torch.tensor(receptive_field, device=device)
        }
    
    def set_streaming_mode(self, enabled: bool = True):
        """Enable or disable streaming mode."""
        self.streaming_mode = enabled
        if not enabled:
            # Reset buffer when disabling streaming
            self.streaming_buffer.zero_()
    
    def reset_streaming_buffer(self):
        """Reset the streaming buffer."""
        self.streaming_buffer.zero_()
    
    def get_delay_samples(self) -> int:
        """Get the delay introduced by the causal convolution in samples."""
        # In causal mode, there's no algorithmic delay (output aligned with input end)
        # But there's a group delay from the filter
        return (self.kernel_size - 1) // 2 * self.dilation
    
    def get_receptive_field_size(self) -> int:
        """Get the total receptive field size."""
        return (self.kernel_size - 1) * self.dilation + 1
    
    def extra_repr(self) -> str:
        """String representation with extra information."""
        s = (
            f'{self.in_channels}, {self.out_channels}, '
            f'kernel_size={self.kernel_size}, stride={self.stride}'
        )
        if self.dilation != 1:
            s += f', dilation={self.dilation}'
        if self.groups != 1:
            s += f', groups={self.groups}'
        s += f', causal_padding={self.causal_padding}'
        s += f', padding_mode={self.padding_mode}'
        return s