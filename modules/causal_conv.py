import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution for real-time audio processing.
    
    Ensures the output at time t only depends on inputs up to time t (no future lookahead).
    Essential for streaming/real-time audio applications.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolution kernel
        stride: Stride of the convolution
        dilation: Dilation factor
        groups: Number of groups for grouped convolution
        bias: Whether to add learnable bias
        padding_mode: Padding mode for boundaries (default 'zeros')
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
        padding_mode: str = 'zeros'
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        
        # Calculate required padding for causality
        self.causal_padding = (kernel_size - 1) * dilation
        
        # Create the convolution layer with asymmetric padding
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,  # We'll handle padding ourselves
            dilation=dilation,
            groups=groups,
            bias=bias
        )
    
    def forward(self, x: torch.Tensor, return_state: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply causal convolution.
        
        Args:
            x: Input tensor [batch, channels, time]
            return_state: If True, return the state for streaming
        
        Returns:
            Output tensor [batch, channels, time] and optionally the state
        """
        # Apply causal padding (only pad the left side)
        if self.causal_padding > 0:
            if self.padding_mode == 'zeros':
                x_padded = F.pad(x, (self.causal_padding, 0), mode='constant', value=0)
            elif self.padding_mode == 'reflect':
                # For reflect mode, we need at least as many samples as padding
                if x.size(-1) > self.causal_padding:
                    x_padded = F.pad(x, (self.causal_padding, 0), mode='reflect')
                else:
                    # Fall back to zeros if not enough samples
                    x_padded = F.pad(x, (self.causal_padding, 0), mode='constant', value=0)
            elif self.padding_mode == 'replicate':
                x_padded = F.pad(x, (self.causal_padding, 0), mode='replicate')
            else:
                raise ValueError(f"Unsupported padding mode: {self.padding_mode}")
        else:
            x_padded = x
        
        # Apply convolution
        output = self.conv(x_padded)
        
        if return_state:
            # Return the last kernel_size - 1 samples as state for streaming
            receptive_field = self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1)
            state = x[:, :, -receptive_field + 1:] if receptive_field > 1 else None
            return output, state
        else:
            return output
    
    def streaming_forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for streaming/real-time mode.
        
        Args:
            x: Input tensor [batch, channels, time] (can be single frame)
            state: Previous state [batch, channels, receptive_field-1]
        
        Returns:
            Output tensor and updated state
        """
        if state is not None:
            # Concatenate state with new input
            x = torch.cat([state, x], dim=-1)
        
        output = self.forward(x)
        
        # Extract new state
        receptive_field = self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1)
        new_state = x[:, :, -receptive_field + 1:] if receptive_field > 1 else None
        
        return output, new_state
    
    @property
    def receptive_field(self) -> int:
        """Calculate the receptive field of the causal convolution."""
        return self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1)
    
    @property
    def output_delay(self) -> int:
        """Calculate the inherent delay (in samples) introduced by causal convolution."""
        # Causal convolution has zero algorithmic delay
        return 0
    
    def extra_repr(self) -> str:
        """String representation for printing."""
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.dilation != 1:
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if not self.conv.bias:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += f", padding_mode='{padding_mode}'"
        s += f', causal_padding={causal_padding}'
        return s.format(**self.__dict__)


class CausalConvTranspose1d(nn.Module):
    """
    Causal transposed 1D convolution for upsampling in real-time audio.
    
    Maintains causality while upsampling the signal.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolution kernel
        stride: Stride of the transposed convolution (upsampling factor)
        dilation: Dilation factor
        groups: Number of groups
        bias: Whether to add learnable bias
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        
        # For transposed conv, we need to handle the output padding carefully
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        
        # Calculate trimming needed for causality
        self.causal_trim = (kernel_size - 1) * stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply causal transposed convolution.
        
        Args:
            x: Input tensor [batch, channels, time]
        
        Returns:
            Upsampled output tensor
        """
        # Apply transposed convolution
        output = self.conv_transpose(x)
        
        # Trim future samples to maintain causality
        if self.causal_trim > 0:
            output = output[:, :, :-self.causal_trim]
        
        return output
    
    @property
    def receptive_field(self) -> int:
        """Calculate the receptive field."""
        return self.kernel_size
    
    def extra_repr(self) -> str:
        """String representation."""
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.dilation != 1:
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if not self.conv_transpose.bias:
            s += ', bias=False'
        s += f', causal_trim={causal_trim}'
        return s.format(**self.__dict__)


class CausalPaddingMode:
    """Enum-like class for padding modes."""
    ZEROS = 'zeros'
    REFLECT = 'reflect'
    REPLICATE = 'replicate'
    
    @classmethod
    def all_modes(cls):
        return [cls.ZEROS, cls.REFLECT, cls.REPLICATE]