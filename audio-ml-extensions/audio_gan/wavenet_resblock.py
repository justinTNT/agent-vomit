import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union


class WaveNetResBlock(nn.Module):
    """
    WaveNet-style residual block with dilated causal convolutions.
    Used for high-quality neural audio synthesis.
    """
    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 dilation: int = 1,
                 skip_channels: int = 256,
                 residual_channels: int = 256,
                 gate_channels: int = 256,
                 dropout: float = 0.0,
                 bias: bool = True,
                 use_1x1_skip: bool = True,
                 conditioning_channels: Optional[int] = None):
        super().__init__()
        
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout = dropout
        self.use_1x1_skip = use_1x1_skip
        self.conditioning_channels = conditioning_channels
        
        # Calculate padding for causal convolution
        self.padding = (kernel_size - 1) * dilation
        
        # Dilated causal convolution
        # Output channels should be gate_channels + residual_channels
        output_channels = gate_channels + residual_channels
        self.dilated_conv = nn.Conv1d(
            channels,
            output_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
            bias=bias
        )
        
        # Conditioning projection (if using conditioning)
        if conditioning_channels is not None:
            self.cond_conv = nn.Conv1d(
                conditioning_channels,
                gate_channels + residual_channels,
                kernel_size=1,
                bias=bias
            )
            
        # Output projections
        self.residual_conv = nn.Conv1d(
            residual_channels,
            channels,
            kernel_size=1,
            bias=bias
        )
        
        if use_1x1_skip:
            self.skip_conv = nn.Conv1d(
                residual_channels,
                skip_channels,
                kernel_size=1,
                bias=bias
            )
        else:
            self.skip_conv = None
            
        # Dropout
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None
            
    def forward(self, 
                x: torch.Tensor,
                conditioning: Optional[torch.Tensor] = None) -> tuple:
        """
        Forward pass of WaveNet residual block.
        
        Args:
            x: Input tensor [batch, channels, time]
            conditioning: Optional conditioning tensor [batch, cond_channels, time]
            
        Returns:
            residual: Output for next layer [batch, channels, time]
            skip: Skip connection output [batch, skip_channels, time]
        """
        # Store input for residual connection
        residual = x
        
        # Dilated causal convolution
        x = self.dilated_conv(x)
        
        # Remove future samples for causality
        if self.padding > 0:
            x = x[:, :, :-self.padding]
            
        # Add conditioning if provided
        if conditioning is not None and self.conditioning_channels is not None:
            cond = self.cond_conv(conditioning)
            x = x + cond
            
        # Gated activation
        gate_channels = x.size(1) // 2
        gate = torch.sigmoid(x[:, :gate_channels, :])
        filter = torch.tanh(x[:, gate_channels:, :])
        x = gate * filter
        
        # Apply dropout if configured
        if self.dropout_layer is not None:
            x = self.dropout_layer(x)
            
        # Skip connection
        if self.skip_conv is not None:
            skip = self.skip_conv(x)
        else:
            skip = x
            
        # Residual connection
        x = self.residual_conv(x)
        x = x + residual
        
        return x, skip


class WaveNetStack(nn.Module):
    """
    Stack of WaveNet residual blocks with exponentially increasing dilations.
    """
    def __init__(self,
                 in_channels: int,
                 residual_channels: int = 256,
                 skip_channels: int = 256,
                 kernel_size: int = 3,
                 dilations: Optional[List[int]] = None,
                 dropout: float = 0.0,
                 bias: bool = True,
                 conditioning_channels: Optional[int] = None):
        super().__init__()
        
        # Default dilation pattern (exponential growth)
        if dilations is None:
            dilations = [2 ** i for i in range(10)]  # 1, 2, 4, ..., 512
            
        self.dilations = dilations
        self.receptive_field = self._compute_receptive_field(kernel_size, dilations)
        
        # Input projection
        self.input_conv = nn.Conv1d(in_channels, residual_channels, 1, bias=bias)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for dilation in dilations:
            block = WaveNetResBlock(
                channels=residual_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                skip_channels=skip_channels,
                residual_channels=residual_channels,
                gate_channels=residual_channels,
                dropout=dropout,
                bias=bias,
                conditioning_channels=conditioning_channels
            )
            self.residual_blocks.append(block)
            
        # Output projection
        self.output_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_channels, skip_channels, 1, bias=bias),
            nn.ReLU(),
            nn.Conv1d(skip_channels, in_channels, 1, bias=bias)
        )
        
    def _compute_receptive_field(self, kernel_size: int, dilations: List[int]) -> int:
        """Compute the receptive field of the stack."""
        rf = 1
        for dilation in dilations:
            rf += (kernel_size - 1) * dilation
        return rf
        
    def forward(self, 
                x: torch.Tensor,
                conditioning: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through WaveNet stack.
        
        Args:
            x: Input tensor [batch, in_channels, time]
            conditioning: Optional conditioning [batch, cond_channels, time]
            
        Returns:
            output: Synthesized audio [batch, in_channels, time]
        """
        # Input projection
        x = self.input_conv(x)
        
        # Collect skip connections
        skips = []
        
        # Pass through residual blocks
        for block in self.residual_blocks:
            x, skip = block(x, conditioning)
            skips.append(skip)
            
        # Sum skip connections
        skip_sum = torch.zeros_like(skips[0])
        for skip in skips:
            skip_sum = skip_sum + skip
            
        # Output projection
        output = self.output_conv(skip_sum)
        
        return output
        
    def get_receptive_field(self) -> int:
        """Get the receptive field size in samples."""
        return self.receptive_field
        
    def inference(self, 
                  initial_input: torch.Tensor,
                  steps: int,
                  conditioning: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Autoregressive inference mode.
        
        Args:
            initial_input: Initial input [batch, in_channels, initial_time]
            steps: Number of steps to generate
            conditioning: Optional conditioning [batch, cond_channels, total_time]
            
        Returns:
            output: Generated sequence [batch, in_channels, initial_time + steps]
        """
        # This is a simplified version - full WaveNet uses caching for efficiency
        generated = initial_input
        
        for i in range(steps):
            # Use full generated sequence (inefficient but simple)
            output = self.forward(generated, conditioning)
            
            # Take last sample
            next_sample = output[:, :, -1:]
            
            # Append to generated sequence
            generated = torch.cat([generated, next_sample], dim=2)
            
        return generated