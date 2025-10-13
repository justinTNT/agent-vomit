"""
AdaIN (Adaptive Instance Normalization) module for style-based generation.

AdaIN enables style transfer and controllable generation by normalizing
features and applying learned affine transformations based on style/condition
inputs. Widely used in GANs and style transfer networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple


class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization layer.
    
    Normalizes input features and applies style-based affine transformation.
    Unlike batch norm, statistics are computed per-instance, and affine
    parameters come from a style/condition input rather than being fixed.
    
    Args:
        num_features: Number of features to normalize
        style_dim: Dimension of style/conditioning vector
        use_bias: Whether to apply bias (shift) in addition to scale
        eps: Small value for numerical stability
        momentum: Momentum for optional running stats tracking
        track_running_stats: Whether to track running statistics
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        num_features: int,
        style_dim: int,
        use_bias: bool = True,
        eps: float = 1e-5,
        momentum: float = 0.1,
        track_running_stats: bool = False,
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.num_features = num_features
        self.style_dim = style_dim
        self.use_bias = use_bias
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        
        # Style mapping networks
        self.style_scale = nn.Linear(style_dim, num_features)
        if use_bias:
            self.style_bias = nn.Linear(style_dim, num_features)
        else:
            self.style_bias = None
        
        # Optional running statistics (for analysis/debugging)
        if track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None
        
        # Initialize
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters."""
        # Initialize to produce identity transform by default
        nn.init.constant_(self.style_scale.weight, 0)
        nn.init.constant_(self.style_scale.bias, 1)
        
        if self.style_bias is not None:
            nn.init.constant_(self.style_bias.weight, 0)
            nn.init.constant_(self.style_bias.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        style: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply AdaIN to input features.
        
        Args:
            x: Input features of shape (batch, num_features, *spatial)
               where *spatial can be 1D, 2D, or 3D
            style: Style vector of shape (batch, style_dim)
            
        Returns:
            Normalized and style-modulated features, same shape as input
        """
        assert x.size(1) == self.num_features, \
            f"Expected {self.num_features} features, got {x.size(1)}"
        
        # Calculate instance statistics
        # Reshape to (batch, features, -1) for any spatial dims
        batch_size = x.size(0)
        x_reshaped = x.view(batch_size, self.num_features, -1)
        
        # Instance mean and std
        instance_mean = x_reshaped.mean(dim=2, keepdim=True)
        instance_var = x_reshaped.var(dim=2, keepdim=True, unbiased=False)
        instance_std = (instance_var + self.eps).sqrt()
        
        # Normalize
        x_norm = (x_reshaped - instance_mean) / instance_std
        
        # Update running stats if tracking
        if self.training and self.track_running_stats:
            # Detach to avoid backprop through running stats
            mean_batch = instance_mean.mean(dim=0).squeeze()
            var_batch = instance_var.mean(dim=0).squeeze()
            
            self.num_batches_tracked += 1
            if self.momentum is None:
                # Cumulative moving average
                n = self.num_batches_tracked.item()
                self.running_mean = (self.running_mean * (n - 1) + mean_batch) / n
                self.running_var = (self.running_var * (n - 1) + var_batch) / n
            else:
                # Exponential moving average
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_batch
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_batch
        
        # Get style parameters
        scale = self.style_scale(style)  # (batch, features)
        if self.use_bias:
            bias = self.style_bias(style)  # (batch, features)
        else:
            bias = 0
        
        # Apply style modulation
        # Reshape scale and bias to match normalized tensor
        scale = scale.view(batch_size, self.num_features, 1)
        if self.use_bias:
            bias = bias.view(batch_size, self.num_features, 1)
        
        x_modulated = x_norm * scale + bias
        
        # Reshape back to original shape
        x_modulated = x_modulated.view_as(x)
        
        return x_modulated
    
    def extra_repr(self) -> str:
        return (f'num_features={self.num_features}, style_dim={self.style_dim}, '
                f'use_bias={self.use_bias}, eps={self.eps}')


class AdaINResBlock(nn.Module):
    """
    Residual block with AdaIN conditioning.
    
    Combines convolution, AdaIN normalization, and residual connections
    for building style-based generators.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        style_dim: Dimension of style vector
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Convolution padding
        activation: Activation function
        upsample: Whether to upsample in this block
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        activation: str = 'leaky_relu',
        upsample: bool = False,
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = upsample
        
        # Auto padding
        if padding is None:
            padding = (kernel_size - 1) // 2
        
        # Main path
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.adain1 = AdaIN(out_channels, style_dim)
        self.activation1 = self._get_activation(activation)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)
        self.adain2 = AdaIN(out_channels, style_dim)
        
        # Skip connection
        if in_channels != out_channels or stride != 1 or upsample:
            self.skip = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.skip = None
        
        # Final activation
        self.activation2 = self._get_activation(activation)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation module."""
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(
        self,
        x: torch.Tensor,
        style: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with style conditioning.
        
        Args:
            x: Input features (batch, in_channels, time)
            style: Style vector (batch, style_dim)
            
        Returns:
            Output features (batch, out_channels, time')
        """
        # Upsample if needed
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
        
        # Main path
        residual = x
        
        x = self.conv1(x)
        x = self.adain1(x, style)
        x = self.activation1(x)
        
        x = self.conv2(x)
        x = self.adain2(x, style)
        
        # Skip connection
        if self.skip is not None:
            residual = self.skip(residual)
        
        x = x + residual
        x = self.activation2(x)
        
        return x


class StyleMapping(nn.Module):
    """
    Maps latent codes to style vectors for AdaIN.
    
    This network transforms random latent codes into style vectors
    that control the AdaIN layers. Often includes multiple layers
    to increase the expressiveness of the style space.
    
    Args:
        latent_dim: Dimension of input latent code
        style_dim: Dimension of output style vector
        num_layers: Number of mapping layers
        hidden_dim: Hidden dimension (defaults to style_dim)
        activation: Activation function
        normalize_input: Whether to normalize input latents
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        latent_dim: int,
        style_dim: int,
        num_layers: int = 4,
        hidden_dim: Optional[int] = None,
        activation: str = 'leaky_relu',
        normalize_input: bool = True,
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        self.normalize_input = normalize_input
        
        if hidden_dim is None:
            hidden_dim = style_dim
        
        # Build mapping network
        layers = []
        in_features = latent_dim
        
        for i in range(num_layers):
            out_features = style_dim if i == num_layers - 1 else hidden_dim
            
            layers.append(nn.Linear(in_features, out_features))
            
            if i < num_layers - 1:
                # Add activation except for last layer
                if activation == 'relu':
                    layers.append(nn.ReLU(inplace=True))
                elif activation == 'leaky_relu':
                    layers.append(nn.LeakyReLU(0.2, inplace=True))
                else:
                    raise ValueError(f"Unknown activation: {activation}")
            
            in_features = out_features
        
        self.mapping = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Map latent code to style vector.
        
        Args:
            z: Latent code (batch, latent_dim)
            
        Returns:
            Style vector (batch, style_dim)
        """
        # Normalize input if requested
        if self.normalize_input:
            z = F.normalize(z, p=2, dim=1)
        
        # Map to style space
        style = self.mapping(z)
        
        return style


class ConditionalAdaIN(AdaIN):
    """
    Conditional AdaIN that can also use semantic conditioning.
    
    Extends AdaIN to optionally incorporate semantic information
    (e.g., class labels, text embeddings) in addition to style.
    
    Args:
        num_features: Number of features to normalize
        style_dim: Dimension of style vector
        condition_dim: Dimension of condition vector (optional)
        use_bias: Whether to apply bias
        combine_method: How to combine style and condition ('concat', 'add', 'gate')
        **kwargs: Additional arguments passed to AdaIN
    """
    
    def __init__(
        self,
        num_features: int,
        style_dim: int,
        condition_dim: int = 0,
        use_bias: bool = True,
        combine_method: str = 'concat',
        **kwargs
    ):
        self.condition_dim = condition_dim
        self.combine_method = combine_method
        self.base_style_dim = style_dim
        
        # Determine combined dimension based on method
        if condition_dim > 0:
            if combine_method == 'concat':
                combined_dim = style_dim + condition_dim
            elif combine_method in ['add', 'gate']:
                combined_dim = style_dim
            else:
                raise ValueError(f"Unknown combine method: {combine_method}")
        else:
            combined_dim = style_dim
        
        super().__init__(
            num_features=num_features,
            style_dim=combined_dim,
            use_bias=use_bias,
            **kwargs
        )
        
        # Create projection/gate layers after super().__init__
        if condition_dim > 0 and combine_method in ['add', 'gate']:
            self.condition_proj = nn.Linear(condition_dim, style_dim)
            
        if condition_dim > 0 and combine_method == 'gate':
            self.gate = nn.Linear(style_dim + condition_dim, style_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        style: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply conditional AdaIN.
        
        Args:
            x: Input features (batch, num_features, *spatial)
            style: Style vector (batch, style_dim)
            condition: Optional condition vector (batch, condition_dim)
            
        Returns:
            Normalized and modulated features
        """
        # Combine style and condition if provided
        if condition is not None and self.condition_dim > 0:
            if self.combine_method == 'concat':
                style = torch.cat([style, condition], dim=1)
            elif self.combine_method == 'add':
                condition_proj = self.condition_proj(condition)
                style = style + condition_proj
            elif self.combine_method == 'gate':
                condition_proj = self.condition_proj(condition)
                gate = torch.sigmoid(self.gate(torch.cat([style, condition], dim=1)))
                style = style * gate + condition_proj * (1 - gate)
        
        # Apply standard AdaIN
        return super().forward(x, style)