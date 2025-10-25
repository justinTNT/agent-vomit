"""
AntiAliasedConv - Anti-aliased convolution with low-pass filtering.

This module applies low-pass filters before downsampling to prevent
aliasing artifacts, supporting different filter types.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Literal, Tuple
import warnings
import math


class AntiAliasedConv(nn.Module):
    """
    Anti-aliased convolution layer with built-in low-pass filtering.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolution kernel
        stride: Stride of the convolution
        padding: Padding to add
        dilation: Dilation factor
        groups: Number of groups
        bias: Whether to use bias
        filter_type: Type of anti-aliasing filter ('box', 'triangle', 'gaussian', 'lanczos')
        filter_size: Size of the anti-aliasing filter (odd number)
        sigma: Standard deviation for Gaussian filter
        dim: Convolution dimension (1, 2, or 3)
        **kwargs: Additional arguments
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        filter_type: Literal['box', 'triangle', 'gaussian', 'lanczos'] = 'gaussian',
        filter_size: Optional[int] = None,
        sigma: Optional[float] = None,
        dim: int = 2,
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
        if dim not in [1, 2, 3]:
            raise ValueError(f"dim must be 1, 2, or 3, got {dim}")
        if filter_type not in ['box', 'triangle', 'gaussian', 'lanczos']:
            raise ValueError(f"Invalid filter_type: {filter_type}")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.filter_type = filter_type
        self.dim = dim
        
        # Only apply anti-aliasing if stride > 1
        self.apply_antialiasing = stride > 1
        
        if self.apply_antialiasing:
            # Determine filter size based on stride if not provided
            if filter_size is None:
                # Nyquist-Shannon theorem: need at least 2*stride-1
                filter_size = 2 * stride + 1
            
            if filter_size % 2 == 0:
                filter_size += 1  # Ensure odd size
            
            self.filter_size = filter_size
            
            # Set sigma for Gaussian if not provided
            if sigma is None and filter_type == 'gaussian':
                # Standard choice: sigma = 0.5 * stride
                sigma = 0.5 * stride
            self.sigma = sigma
            
            # Create anti-aliasing filter
            aa_filter = self._create_filter()
            
            # Register as buffer (no gradients needed)
            self.register_buffer('aa_filter', aa_filter)
            
            # Padding for anti-aliasing filter
            self.aa_padding = (filter_size - 1) // 2
        
        # Select appropriate convolution class
        if dim == 1:
            conv_class = nn.Conv1d
        elif dim == 2:
            conv_class = nn.Conv2d
        else:  # dim == 3
            conv_class = nn.Conv3d
        
        # Main convolution layer
        # If using anti-aliasing, set stride=1 here and apply stride in forward
        conv_stride = 1 if self.apply_antialiasing else stride
        
        self.conv = conv_class(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=conv_stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        
        # Statistics
        self.register_buffer('total_forward_passes', torch.tensor(0, dtype=torch.long))
        
        # Handle unused kwargs
        if kwargs:
            warnings.warn(f"Unused arguments: {list(kwargs.keys())}")
    
    def _create_1d_filter(self) -> torch.Tensor:
        """Create 1D anti-aliasing filter."""
        if self.filter_type == 'box':
            kernel = torch.ones(self.filter_size)
        
        elif self.filter_type == 'triangle':
            # Triangle (linear) filter
            half_size = self.filter_size // 2
            kernel = 1 - torch.abs(torch.linspace(-1, 1, self.filter_size))
        
        elif self.filter_type == 'gaussian':
            # Gaussian filter
            x = torch.arange(self.filter_size) - self.filter_size // 2
            kernel = torch.exp(-0.5 * (x / self.sigma) ** 2)
        
        elif self.filter_type == 'lanczos':
            # Lanczos filter
            x = torch.linspace(-2, 2, self.filter_size)
            # Avoid division by zero
            x_safe = torch.where(x == 0, torch.tensor(1e-8), x)
            kernel = torch.sinc(x) * torch.sinc(x_safe / 2)
            kernel = torch.where(x == 0, torch.tensor(1.0), kernel)
        
        # Normalize
        kernel = kernel / kernel.sum()
        
        return kernel
    
    def _create_filter(self) -> torch.Tensor:
        """Create anti-aliasing filter for the specified dimension."""
        # Create 1D filter
        kernel_1d = self._create_1d_filter()
        
        if self.dim == 1:
            # Shape: (1, 1, filter_size)
            kernel = kernel_1d.view(1, 1, -1)
            # Repeat for each channel
            kernel = kernel.repeat(self.in_channels, 1, 1)
        
        elif self.dim == 2:
            # Create 2D filter by outer product
            kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
            # Shape: (1, 1, filter_size, filter_size)
            kernel = kernel_2d.unsqueeze(0).unsqueeze(0)
            # Repeat for each channel
            kernel = kernel.repeat(self.in_channels, 1, 1, 1)
        
        else:  # dim == 3
            # Create 3D filter by triple outer product
            kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
            kernel_3d = kernel_2d.unsqueeze(0) * kernel_1d.unsqueeze(1).unsqueeze(2)
            # Shape: (1, 1, filter_size, filter_size, filter_size)
            kernel = kernel_3d.unsqueeze(0).unsqueeze(0)
            # Repeat for each channel
            kernel = kernel.repeat(self.in_channels, 1, 1, 1, 1)
        
        return kernel
    
    def _apply_antialiasing(self, x: torch.Tensor) -> torch.Tensor:
        """Apply anti-aliasing filter before downsampling."""
        # Move filter to correct device
        aa_filter = self.aa_filter.to(x.device)
        
        if self.dim == 1:
            # Apply 1D filter
            x_filtered = F.conv1d(
                x, 
                aa_filter, 
                padding=self.aa_padding,
                groups=self.in_channels
            )
            # Downsample
            x_downsampled = x_filtered[..., ::self.stride]
        
        elif self.dim == 2:
            # Apply 2D filter
            x_filtered = F.conv2d(
                x,
                aa_filter,
                padding=self.aa_padding,
                groups=self.in_channels
            )
            # Downsample
            x_downsampled = x_filtered[..., ::self.stride, ::self.stride]
        
        else:  # dim == 3
            # Apply 3D filter
            x_filtered = F.conv3d(
                x,
                aa_filter,
                padding=self.aa_padding,
                groups=self.in_channels
            )
            # Downsample
            x_downsampled = x_filtered[..., ::self.stride, ::self.stride, ::self.stride]
        
        return x_downsampled
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply anti-aliased convolution.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary containing:
                - output: Convolved and anti-aliased output
                - aliasing_prevented: Whether anti-aliasing was applied
                - effective_stride: Actual stride used
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        
        expected_dim = self.dim + 2  # batch + channels + spatial dims
        if x.dim() != expected_dim:
            raise ValueError(f"Expected {expected_dim}D tensor, got {x.dim()}D")
        
        if x.size(1) != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {x.size(1)}")
        
        device = x.device
        self.total_forward_passes += 1
        
        if self.apply_antialiasing:
            # First apply convolution with stride=1
            x_conv = self.conv(x)
            
            # Then apply anti-aliasing and downsampling
            output = self._apply_antialiasing(x_conv)
            
            aliasing_prevented = True
            effective_stride = self.stride
        else:
            # Standard convolution without anti-aliasing
            output = self.conv(x)
            
            aliasing_prevented = False
            effective_stride = self.stride
        
        return {
            'output': output,
            'aliasing_prevented': torch.tensor(aliasing_prevented, device=device),
            'effective_stride': torch.tensor(effective_stride, device=device)
        }
    
    def get_filter_info(self) -> Dict[str, Any]:
        """Get information about the anti-aliasing filter."""
        info = {
            'apply_antialiasing': self.apply_antialiasing,
            'filter_type': self.filter_type,
            'stride': self.stride,
            'dimension': self.dim
        }
        
        if self.apply_antialiasing:
            info.update({
                'filter_size': self.filter_size,
                'filter_shape': list(self.aa_filter.shape),
                'sigma': self.sigma if self.filter_type == 'gaussian' else None
            })
        
        return info
    
    def extra_repr(self) -> str:
        """String representation with extra information."""
        s = (
            f'{self.in_channels}, {self.out_channels}, '
            f'kernel_size={self.kernel_size}, stride={self.stride}'
        )
        if self.padding != 0:
            s += f', padding={self.padding}'
        if self.dilation != 1:
            s += f', dilation={self.dilation}'
        if self.groups != 1:
            s += f', groups={self.groups}'
        if self.apply_antialiasing:
            s += f', filter_type={self.filter_type}, filter_size={self.filter_size}'
        return s