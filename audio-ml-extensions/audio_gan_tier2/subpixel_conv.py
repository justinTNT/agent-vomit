"""
SubPixelConv module for efficient upsampling in neural vocoders.

SubPixel convolution (also known as PixelShuffle) is an efficient method for upsampling
that learns the upsampling filters directly. It's widely used in neural vocoders
like HiFi-GAN and UnivNet for converting low-resolution features to high-resolution audio.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union


class SubPixelConv(nn.Module):
    """
    SubPixel convolution for efficient upsampling.
    
    This module performs convolution followed by pixel shuffling to increase
    spatial/temporal resolution. It's computationally efficient compared to
    transposed convolutions and produces fewer checkerboard artifacts.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels after upsampling
        kernel_size: Size of the convolution kernel
        upsample_factor: Factor by which to upsample (must be int)
        stride: Stride of the convolution (default: 1)
        padding: Padding for the convolution (default: auto)
        activation: Activation function to apply after upsampling
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        upsample_factor: int,
        stride: int = 1,
        padding: Optional[int] = None,
        activation: Optional[str] = 'leaky_relu',
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.upsample_factor = upsample_factor
        self.stride = stride
        
        # Auto-calculate padding if not specified
        if padding is None:
            self.padding = (kernel_size - 1) // 2
        else:
            self.padding = padding
        
        # The conv layer produces upsample_factor * out_channels channels
        # These will be rearranged by pixel shuffle
        self.conv = nn.Conv1d(
            in_channels,
            out_channels * upsample_factor,
            kernel_size,
            stride=stride,
            padding=self.padding
        )
        
        # Activation
        self.activation = self._get_activation(activation)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation: Optional[str]) -> Optional[nn.Module]:
        """Get activation module from string name."""
        if activation is None:
            return None
        elif activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SubPixelConv.
        
        Args:
            x: Input tensor of shape (batch, in_channels, time)
            
        Returns:
            Upsampled tensor of shape (batch, out_channels, time * upsample_factor)
        """
        # Apply convolution
        x = self.conv(x)  # (B, out_channels * upsample_factor, T)
        
        # Reshape for pixel shuffle
        # Split the channel dimension into (out_channels, upsample_factor)
        batch_size = x.size(0)
        time_steps = x.size(2)
        
        x = x.view(batch_size, self.out_channels, self.upsample_factor, time_steps)
        
        # Transpose and reshape to perform the shuffle
        # Move upsample_factor next to time dimension
        x = x.permute(0, 1, 3, 2).contiguous()
        
        # Reshape to merge time and upsample dimensions
        x = x.view(batch_size, self.out_channels, time_steps * self.upsample_factor)
        
        # Apply activation
        if self.activation is not None:
            x = self.activation(x)
        
        return x
    
    def get_output_length(self, input_length: int) -> int:
        """Calculate output length given input length."""
        # First apply convolution formula
        conv_output = (input_length + 2 * self.padding - self.kernel_size) // self.stride + 1
        # Then apply upsampling
        return conv_output * self.upsample_factor


class SubPixelConvTranspose(nn.Module):
    """
    SubPixel convolution with pre-upsampling for better frequency response.
    
    This variant first upsamples with zeros, then applies convolution.
    It can achieve better frequency response at the cost of more computation.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolution kernel
        upsample_factor: Factor by which to upsample
        padding: Padding for the convolution (default: auto)
        activation: Activation function to apply
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        upsample_factor: int,
        padding: Optional[int] = None,
        activation: Optional[str] = 'leaky_relu',
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.upsample_factor = upsample_factor
        
        # Calculate padding to maintain size after upsampling
        if padding is None:
            # We want the convolution to not change the size after upsampling
            self.padding = kernel_size // 2
        else:
            self.padding = padding
        
        # Standard convolution after upsampling
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=self.padding
        )
        
        # Activation
        self.activation = self._get_activation(activation)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation: Optional[str]) -> Optional[nn.Module]:
        """Get activation module from string name."""
        if activation is None:
            return None
        elif activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with pre-upsampling.
        
        Args:
            x: Input tensor of shape (batch, in_channels, time)
            
        Returns:
            Upsampled tensor of shape (batch, out_channels, time * upsample_factor)
        """
        # Upsample by inserting zeros
        batch_size, channels, time_steps = x.shape
        
        # Create output tensor with zeros
        upsampled = torch.zeros(
            batch_size,
            channels,
            time_steps * self.upsample_factor,
            device=x.device,
            dtype=x.dtype
        )
        
        # Fill in the values at every upsample_factor position
        upsampled[:, :, ::self.upsample_factor] = x
        
        # Apply convolution
        output = self.conv(upsampled)
        
        # Apply activation
        if self.activation is not None:
            output = self.activation(output)
        
        return output


class MultiScaleSubPixelConv(nn.Module):
    """
    Multi-scale SubPixel convolution for progressive upsampling.
    
    This module applies SubPixel convolution at multiple scales,
    useful for high upsampling ratios (e.g., 256x for neural vocoders).
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        upsample_factors: List of upsample factors for each stage
        kernel_sizes: List of kernel sizes for each stage (or single value)
        hidden_channels: Number of channels in hidden layers
        activation: Activation function between stages
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_factors: List[int],
        kernel_sizes: Union[int, List[int]] = 7,
        hidden_channels: Optional[int] = None,
        activation: str = 'leaky_relu',
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_factors = upsample_factors
        
        # Handle kernel sizes
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(upsample_factors)
        assert len(kernel_sizes) == len(upsample_factors), \
            "kernel_sizes must match upsample_factors length"
        
        # Default hidden channels
        if hidden_channels is None:
            hidden_channels = max(in_channels, out_channels)
        
        # Build layers
        self.layers = nn.ModuleList()
        current_channels = in_channels
        
        for i, (factor, kernel_size) in enumerate(zip(upsample_factors, kernel_sizes)):
            # Determine output channels for this layer
            if i == len(upsample_factors) - 1:
                # Last layer
                next_channels = out_channels
                layer_activation = None  # No activation on last layer
            else:
                next_channels = hidden_channels
                layer_activation = activation
            
            # Add SubPixel layer
            self.layers.append(
                SubPixelConv(
                    current_channels,
                    next_channels,
                    kernel_size,
                    factor,
                    activation=layer_activation
                )
            )
            
            current_channels = next_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multiple upsampling stages.
        
        Args:
            x: Input tensor of shape (batch, in_channels, time)
            
        Returns:
            Upsampled tensor
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_total_upsample_factor(self) -> int:
        """Get total upsampling factor across all stages."""
        total = 1
        for factor in self.upsample_factors:
            total *= factor
        return total