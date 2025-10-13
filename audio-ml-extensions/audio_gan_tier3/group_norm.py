"""
GroupNorm module for audio processing.

Group Normalization is particularly useful for audio applications where
batch sizes can be small or variable. It normalizes features across groups
of channels rather than across batches, providing more stable training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union


class GroupNorm1d(nn.Module):
    """
    Group Normalization for 1D sequences (audio).
    
    Normalizes input by dividing channels into groups and normalizing
    within each group. More stable than BatchNorm for small batches.
    
    Args:
        num_channels: Number of input channels
        num_groups: Number of groups to divide channels into
        eps: Small constant for numerical stability
        affine: Whether to learn scale and shift parameters
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        num_channels: int,
        num_groups: int = 32,
        eps: float = 1e-5,
        affine: bool = True,
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps
        self.affine = affine
        
        # Validate group configuration
        if num_channels % num_groups != 0:
            raise ValueError(f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})")
        
        self.channels_per_group = num_channels // num_groups
        
        # Learnable parameters
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply group normalization.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
            
        Returns:
            Normalized tensor of same shape
        """
        batch_size, num_channels, time_steps = x.shape
        
        assert num_channels == self.num_channels, \
            f"Expected {self.num_channels} channels, got {num_channels}"
        
        # Reshape to (batch, num_groups, channels_per_group, time)
        x = x.view(batch_size, self.num_groups, self.channels_per_group, time_steps)
        
        # Compute mean and variance across channels_per_group and time dimensions
        mean = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        
        # Normalize
        x = (x - mean) / torch.sqrt(var + self.eps)
        
        # Reshape back to original shape
        x = x.view(batch_size, num_channels, time_steps)
        
        # Apply affine transformation
        if self.affine:
            x = x * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        
        return x
    
    def extra_repr(self) -> str:
        return f'num_channels={self.num_channels}, num_groups={self.num_groups}, eps={self.eps}, affine={self.affine}'


class AdaptiveGroupNorm(nn.Module):
    """
    Adaptive Group Normalization with conditioning.
    
    Allows the normalization parameters to be modulated by external
    conditioning signals, useful for controllable audio generation.
    
    Args:
        num_channels: Number of input channels
        conditioning_dim: Dimension of conditioning vector
        num_groups: Number of groups (default: 32)
        eps: Small constant for numerical stability
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        num_channels: int,
        conditioning_dim: int,
        num_groups: int = 32,
        eps: float = 1e-5,
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.num_channels = num_channels
        self.conditioning_dim = conditioning_dim
        self.num_groups = num_groups
        self.eps = eps
        
        # Validate group configuration
        if num_channels % num_groups != 0:
            raise ValueError(f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})")
        
        self.channels_per_group = num_channels // num_groups
        
        # Base group norm (without affine parameters)
        self.group_norm = GroupNorm1d(num_channels, num_groups, eps, affine=False)
        
        # Conditioning networks for scale and bias
        self.scale_net = nn.Linear(conditioning_dim, num_channels)
        self.bias_net = nn.Linear(conditioning_dim, num_channels)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        # Initialize to produce identity transformation
        nn.init.zeros_(self.scale_net.weight)
        nn.init.ones_(self.scale_net.bias)
        
        nn.init.zeros_(self.bias_net.weight)
        nn.init.zeros_(self.bias_net.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply adaptive group normalization.
        
        Args:
            x: Input tensor (batch, channels, time)
            conditioning: Conditioning vector (batch, conditioning_dim)
            
        Returns:
            Normalized and conditioned tensor
        """
        # Apply base group normalization
        x = self.group_norm(x)
        
        # Generate adaptive parameters
        scale = self.scale_net(conditioning).view(-1, self.num_channels, 1)
        bias = self.bias_net(conditioning).view(-1, self.num_channels, 1)
        
        # Apply adaptive transformation
        x = x * scale + bias
        
        return x


class LayerNorm1d(nn.Module):
    """
    Layer Normalization for 1D sequences.
    
    Normalizes across the channel dimension for each time step independently.
    Alternative to GroupNorm when you want to normalize across all channels.
    
    Args:
        num_channels: Number of input channels
        eps: Small constant for numerical stability
        affine: Whether to learn scale and shift parameters
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization.
        
        Args:
            x: Input tensor (batch, channels, time)
            
        Returns:
            Normalized tensor
        """
        # Transpose to (batch, time, channels) for layer norm
        x = x.transpose(1, 2)
        
        # Apply layer normalization
        x = F.layer_norm(x, (self.num_channels,), self.weight, self.bias, self.eps)
        
        # Transpose back to (batch, channels, time)
        x = x.transpose(1, 2)
        
        return x


class InstanceNorm1d(nn.Module):
    """
    Instance Normalization for 1D sequences.
    
    Normalizes each channel independently across the time dimension.
    Useful for style transfer and when you want per-instance statistics.
    
    Args:
        num_channels: Number of input channels
        eps: Small constant for numerical stability
        affine: Whether to learn scale and shift parameters
        track_running_stats: Whether to track running statistics
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        track_running_stats: bool = False,
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        if track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_channels))
            self.register_buffer('running_var', torch.ones(num_channels))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
        
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply instance normalization.
        
        Args:
            x: Input tensor (batch, channels, time)
            
        Returns:
            Normalized tensor
        """
        batch_size, num_channels, time_steps = x.shape
        
        assert num_channels == self.num_channels, \
            f"Expected {self.num_channels} channels, got {num_channels}"
        
        # Reshape to (batch * channels, time) for easier processing
        x_reshaped = x.view(batch_size * num_channels, time_steps)
        
        # Compute instance statistics
        if self.training or not self.track_running_stats:
            # Compute statistics per instance
            mean = x_reshaped.mean(dim=1, keepdim=True)
            var = x_reshaped.var(dim=1, keepdim=True, unbiased=False)
            
            if self.track_running_stats:
                # Update running statistics
                with torch.no_grad():
                    batch_mean = mean.view(batch_size, num_channels).mean(dim=0)
                    batch_var = var.view(batch_size, num_channels).mean(dim=0)
                    
                    n = self.num_batches_tracked.item()
                    self.num_batches_tracked += 1
                    
                    # Exponential moving average
                    momentum = 0.1
                    self.running_mean = (1 - momentum) * self.running_mean + momentum * batch_mean
                    self.running_var = (1 - momentum) * self.running_var + momentum * batch_var
        else:
            # Use running statistics
            mean = self.running_mean.repeat(batch_size, 1).view(-1, 1)
            var = self.running_var.repeat(batch_size, 1).view(-1, 1)
        
        # Normalize
        x_norm = (x_reshaped - mean) / torch.sqrt(var + self.eps)
        
        # Reshape back
        x_norm = x_norm.view(batch_size, num_channels, time_steps)
        
        # Apply affine transformation
        if self.affine:
            x_norm = x_norm * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        
        return x_norm


class SwitchableNorm1d(nn.Module):
    """
    Switchable Normalization that learns to combine different normalization methods.
    
    Automatically learns to weight different normalization techniques
    (batch, layer, instance) based on the data.
    
    Args:
        num_channels: Number of input channels
        eps: Small constant for numerical stability
        affine: Whether to learn scale and shift parameters
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        
        # Learnable weights for combining normalizations
        self.weight_bn = nn.Parameter(torch.ones(num_channels))
        self.weight_ln = nn.Parameter(torch.ones(num_channels))
        self.weight_in = nn.Parameter(torch.ones(num_channels))
        
        # Affine parameters
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        # Running statistics for batch norm component
        self.register_buffer('running_mean', torch.zeros(num_channels))
        self.register_buffer('running_var', torch.ones(num_channels))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.ones_(self.weight_bn)
        nn.init.ones_(self.weight_ln)
        nn.init.ones_(self.weight_in)
        
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
        
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply switchable normalization.
        
        Args:
            x: Input tensor (batch, channels, time)
            
        Returns:
            Normalized tensor
        """
        batch_size, num_channels, time_steps = x.shape
        
        # Compute different normalizations
        
        # 1. Batch Norm statistics
        if self.training:
            bn_mean = x.mean(dim=[0, 2])
            bn_var = x.var(dim=[0, 2], unbiased=False)
            
            # Update running statistics
            with torch.no_grad():
                momentum = 0.1
                self.running_mean = (1 - momentum) * self.running_mean + momentum * bn_mean
                self.running_var = (1 - momentum) * self.running_var + momentum * bn_var
        else:
            bn_mean = self.running_mean
            bn_var = self.running_var
        
        # 2. Layer Norm statistics (across channels for each time step)
        ln_mean = x.mean(dim=1, keepdim=True)
        ln_var = x.var(dim=1, keepdim=True, unbiased=False)
        
        # 3. Instance Norm statistics (across time for each channel)
        in_mean = x.mean(dim=2, keepdim=True)
        in_var = x.var(dim=2, keepdim=True, unbiased=False)
        
        # Normalize using each method
        bn_norm = (x - bn_mean.view(1, -1, 1)) / torch.sqrt(bn_var.view(1, -1, 1) + self.eps)
        ln_norm = (x - ln_mean) / torch.sqrt(ln_var + self.eps)
        in_norm = (x - in_mean) / torch.sqrt(in_var + self.eps)
        
        # Compute softmax weights
        weights = torch.stack([self.weight_bn, self.weight_ln, self.weight_in], dim=0)
        weights = F.softmax(weights, dim=0)
        
        # Combine normalizations
        x_norm = (weights[0].view(1, -1, 1) * bn_norm + 
                  weights[1].view(1, -1, 1) * ln_norm + 
                  weights[2].view(1, -1, 1) * in_norm)
        
        # Apply affine transformation
        if self.affine:
            x_norm = x_norm * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        
        return x_norm