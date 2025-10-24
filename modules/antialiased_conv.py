import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import numpy as np
from rave_config_system import RAVEConfig




# TODO: REFACTOR TO CONFIG-FIRST INTERFACE
# New signature: def __init__(self, config: RAVEConfig, filter_type: str, cutoff_ratio: Optional[float], filter_size: int, **kwargs):
# New assignments:
#         self.filter_type = filter_type
        self.cutoff_ratio = cutoff_ratio
        self.filter_size = filter_size
        self.stride = config.convolution.stride
class LowPassFilter1d(nn.Module):
    """
    1D low-pass filter for anti-aliasing.
    
    Implements various filter types commonly used in signal processing.
    Default is Lanczos filter which provides good frequency response.
    
    Args:
        filter_type: Type of filter ('lanczos', 'gaussian', 'butterworth', 'box')
        cutoff_ratio: Cutoff frequency as ratio of Nyquist (0.0-1.0)
        filter_size: Size of filter kernel (must be odd)
        stride: Downsampling factor (determines cutoff if not specified)
    """
    
    def __init__(
        self,
        filter_type: str = 'lanczos',
        cutoff_ratio: Optional[float] = None,
        filter_size: int = 5,
        stride: int = 2
    ):
        super().__init__()
        
        self.filter_type = filter_type
        self.filter_size = filter_size
        self.stride = stride
        
        # Determine cutoff based on stride if not specified
        if cutoff_ratio is None:
            cutoff_ratio = 1.0 / stride
        self.cutoff_ratio = min(cutoff_ratio, 1.0)
        
        # Generate filter kernel
        kernel = self._generate_kernel()
        
        # Normalize kernel
        kernel = kernel / kernel.sum()
        
        # Register as buffer (not trainable)
        self.register_buffer('kernel', kernel)
    
    def _generate_kernel(self) -> torch.Tensor:
        """Generate low-pass filter kernel."""
        if self.filter_type == 'lanczos':
            return self._lanczos_kernel()
        elif self.filter_type == 'gaussian':
            return self._gaussian_kernel()
        elif self.filter_type == 'butterworth':
            return self._butterworth_kernel()
        elif self.filter_type == 'box':
            return self._box_kernel()
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")
    
    def _lanczos_kernel(self) -> torch.Tensor:
        """Generate Lanczos filter kernel."""
        # Create coordinate grid
        n = self.filter_size
        x = torch.linspace(-(n//2), n//2, n)
        
        # Lanczos kernel: sinc(x) * sinc(x/a) where a is the filter order
        a = 2  # Lanczos-2 is a good default
        
        # Avoid division by zero
        x_pi = np.pi * x * self.cutoff_ratio
        kernel = torch.where(
            x == 0,
            torch.ones_like(x),
            torch.sin(x_pi) / x_pi * torch.sin(x_pi / a) / (x_pi / a)
        )
        
        # Window to zero at edges
        kernel = torch.where(torch.abs(x) > a, torch.zeros_like(kernel), kernel)
        
        return kernel
    
    def _gaussian_kernel(self) -> torch.Tensor:
        """Generate Gaussian filter kernel."""
        n = self.filter_size
        x = torch.linspace(-(n//2), n//2, n)
        
        # Standard deviation based on cutoff
        sigma = 0.5 / (np.pi * self.cutoff_ratio)
        
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        return kernel
    
    def _butterworth_kernel(self) -> torch.Tensor:
        """Generate Butterworth-like filter kernel (frequency domain approximation)."""
        n = self.filter_size
        x = torch.linspace(-(n//2), n//2, n)
        
        # Approximate Butterworth in spatial domain
        order = 4
        cutoff = self.cutoff_ratio * n / 2
        
        kernel = 1.0 / (1 + (torch.abs(x) / cutoff) ** (2 * order))
        return kernel
    
    def _box_kernel(self) -> torch.Tensor:
        """Generate simple box (averaging) filter."""
        return torch.ones(self.filter_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply low-pass filter.
        
        Args:
            x: Input tensor [batch, channels, length]
            
        Returns:
            Filtered tensor
        """
        batch, channels, length = x.shape
        
        # Reshape kernel for depthwise convolution
        kernel = self.kernel.view(1, 1, -1).repeat(channels, 1, 1)
        
        # Apply padding
        padding = self.filter_size // 2
        
        # Depthwise convolution
        filtered = F.conv1d(
            x,
            kernel,
            padding=padding,
            groups=channels
        )
        
        return filtered



# TODO: REFACTOR TO CONFIG-FIRST INTERFACE
# New signature: def __init__(self, config: RAVEConfig, filter_type: str, filter_size: Optional[int], **kwargs):
# New assignments:
#         self.in_channels = config.convolution.in_channels
        self.out_channels = config.convolution.out_channels
        self.kernel_size = config.convolution.kernel_size
        self.stride = config.convolution.stride
        self.padding = config.convolution.padding
        self.dilation = config.convolution.dilation
        self.groups = config.convolution.groups
        self.bias = config.convolution.bias
        self.filter_type = filter_type
        self.filter_size = filter_size
class AntialiasedConv1d(nn.Module):
    """
    Anti-aliased 1D convolution with downsampling.
    
    Applies convolution followed by low-pass filtering and downsampling
    to prevent aliasing artifacts.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolution kernel
        stride: Downsampling factor
        padding: Convolution padding
        dilation: Convolution dilation
        groups: Number of groups
        bias: Whether to include bias
        filter_type: Type of anti-aliasing filter
        filter_size: Size of anti-aliasing filter
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 2,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        filter_type: str = 'lanczos',
        filter_size: Optional[int] = None
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Standard convolution with stride=1
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,  # We handle downsampling separately
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        
        # Anti-aliasing filter
        if stride > 1:
            # Default filter size based on stride
            if filter_size is None:
                filter_size = 2 * stride + 1
            
            self.filter = LowPassFilter1d(
                filter_type=filter_type,
                filter_size=filter_size,
                stride=stride
            )
        else:
            self.filter = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply anti-aliased convolution.
        
        Args:
            x: Input tensor [batch, channels, length]
            
        Returns:
            Output tensor with downsampling
        """
        # Apply convolution
        x = self.conv(x)
        
        # Apply anti-aliasing and downsampling
        if self.stride > 1 and self.filter is not None:
            x = self.filter(x)
            # Downsample
            x = x[:, :, ::self.stride]
        
        return x
    
    def extra_repr(self) -> str:
        """String representation."""
        s = (f'{self.in_channels}, {self.out_channels}, '
             f'kernel_size={self.kernel_size}, stride={self.stride}')
        if self.padding != 0:
            s += f', padding={self.padding}'
        if self.dilation != 1:
            s += f', dilation={self.dilation}'
        if self.groups != 1:
            s += f', groups={self.groups}'
        if self.conv.bias is None:
            s += ', bias=False'
        if self.filter is not None:
            s += f', filter_type={self.filter.filter_type}'
        return s



# TODO: REFACTOR TO CONFIG-FIRST INTERFACE
# New signature: def __init__(self, config: RAVEConfig, output_padding: int, filter_type: str, filter_size: Optional[int], **kwargs):
# New assignments:
#         self.in_channels = config.convolution.in_channels
        self.out_channels = config.convolution.out_channels
        self.kernel_size = config.convolution.kernel_size
        self.stride = config.convolution.stride
        self.padding = config.convolution.padding
        self.output_padding = output_padding
        self.groups = config.convolution.groups
        self.bias = config.convolution.bias
        self.dilation = config.convolution.dilation
        self.filter_type = filter_type
        self.filter_size = filter_size
class AntialiasedConvTranspose1d(nn.Module):
    """
    Anti-aliased transposed 1D convolution with upsampling.
    
    Applies upsampling followed by low-pass filtering and convolution
    to prevent aliasing artifacts during upsampling.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels  
        kernel_size: Size of convolution kernel
        stride: Upsampling factor
        padding: Convolution padding
        output_padding: Additional output padding
        groups: Number of groups
        bias: Whether to include bias
        dilation: Dilation factor
        filter_type: Type of anti-aliasing filter
        filter_size: Size of anti-aliasing filter
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 2,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        filter_type: str = 'lanczos',
        filter_size: Optional[int] = None
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = dilation
        
        # Anti-aliasing filter (applied after upsampling)
        if stride > 1:
            if filter_size is None:
                filter_size = 2 * stride + 1
            
            self.filter = LowPassFilter1d(
                filter_type=filter_type,
                filter_size=filter_size,
                stride=stride
            )
        else:
            self.filter = None
        
        # Standard transposed convolution
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply anti-aliased transposed convolution.
        
        Args:
            x: Input tensor [batch, channels, length]
            
        Returns:
            Upsampled output tensor
        """
        # Apply transposed convolution (upsampling)
        x = self.conv_transpose(x)
        
        # Apply anti-aliasing filter to remove upsampling artifacts
        if self.filter is not None:
            x = self.filter(x)
        
        return x



# TODO: REFACTOR TO CONFIG-FIRST INTERFACE
# New signature: def __init__(self, config: RAVEConfig, channels: int, filter_type: str, filter_size: int, **kwargs):
# New assignments:
#         self.channels = channels
        self.filter_type = filter_type
        self.filter_size = filter_size
        self.stride = config.convolution.stride
class BlurPool1d(nn.Module):
    """
    Blur pooling layer for anti-aliased downsampling.
    
    Replaces standard pooling with blur (low-pass filter) + subsampling.
    
    Args:
        channels: Number of channels (for depthwise filtering)
        filter_type: Type of blur filter
        filter_size: Size of blur kernel
        stride: Downsampling stride
    """
    
    def __init__(
        self,
        channels: int,
        filter_type: str = 'lanczos',
        filter_size: int = 5,
        stride: int = 2
    ):
        super().__init__()
        
        self.channels = channels
        self.stride = stride
        
        # Create blur filter
        self.blur = LowPassFilter1d(
            filter_type=filter_type,
            filter_size=filter_size,
            stride=stride
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply blur pooling.
        
        Args:
            x: Input tensor [batch, channels, length]
            
        Returns:
            Blurred and downsampled tensor
        """
        # Apply blur
        x = self.blur(x)
        
        # Subsample
        return x[:, :, ::self.stride]