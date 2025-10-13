"""
FiLM (Feature-wise Linear Modulation) module for conditional neural networks.

FiLM allows neural networks to be conditioned on external information by learning
scale and shift parameters that modulate intermediate features. It's widely used
in audio synthesis for conditioning on pitch, timbre, or other control parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation layer.
    
    This layer learns affine transformations (scale and shift) based on
    conditioning information and applies them to input features.
    
    Args:
        num_features: Number of features to modulate
        conditioning_dim: Dimension of conditioning input
        use_bias: Whether to learn shift parameters (default: True)
        init_scale: Initial value for scale parameters (default: 1.0)
        init_bias: Initial value for bias parameters (default: 0.0)
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        num_features: int,
        conditioning_dim: int,
        use_bias: bool = True,
        init_scale: float = 1.0,
        init_bias: float = 0.0,
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.num_features = num_features
        self.conditioning_dim = conditioning_dim
        self.use_bias = use_bias
        
        # Linear layer to generate scale parameters
        self.scale_gen = nn.Linear(conditioning_dim, num_features)
        
        # Linear layer to generate shift parameters (if used)
        if use_bias:
            self.bias_gen = nn.Linear(conditioning_dim, num_features)
        else:
            self.bias_gen = None
        
        # Initialize parameters
        self._initialize_parameters(init_scale, init_bias)
    
    def _initialize_parameters(self, init_scale: float, init_bias: float):
        """Initialize parameters to sensible defaults."""
        # Initialize scale generation to produce values around init_scale
        nn.init.constant_(self.scale_gen.weight, 0.0)
        nn.init.constant_(self.scale_gen.bias, init_scale)
        
        # Initialize bias generation to produce values around init_bias
        if self.bias_gen is not None:
            nn.init.constant_(self.bias_gen.weight, 0.0)
            nn.init.constant_(self.bias_gen.bias, init_bias)
    
    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply FiLM modulation to input features.
        
        Args:
            x: Input features of shape (batch, num_features) or
               (batch, num_features, time) or (batch, num_features, height, width)
            conditioning: Conditioning input of shape (batch, conditioning_dim)
            
        Returns:
            Modulated features with same shape as input
        """
        # Generate scale and bias
        scale = self.scale_gen(conditioning)  # (batch, num_features)
        
        if self.use_bias:
            bias = self.bias_gen(conditioning)  # (batch, num_features)
        else:
            bias = 0
        
        # Reshape scale and bias to match input dimensions
        if x.dim() == 2:
            # (batch, features)
            return x * scale + bias
        elif x.dim() == 3:
            # (batch, features, time)
            scale = scale.unsqueeze(-1)
            if self.use_bias:
                bias = bias.unsqueeze(-1)
            return x * scale + bias
        elif x.dim() == 4:
            # (batch, features, height, width)
            scale = scale.unsqueeze(-1).unsqueeze(-1)
            if self.use_bias:
                bias = bias.unsqueeze(-1).unsqueeze(-1)
            return x * scale + bias
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}")


class FiLMBlock(nn.Module):
    """
    A neural network block with FiLM conditioning.
    
    This combines a standard layer (conv or linear) with FiLM modulation,
    allowing the block to be conditioned on external information.
    
    Args:
        layer_type: Type of layer ('conv1d', 'conv2d', or 'linear')
        in_features: Number of input features/channels
        out_features: Number of output features/channels
        conditioning_dim: Dimension of conditioning input
        kernel_size: Kernel size for conv layers (default: 3)
        stride: Stride for conv layers (default: 1)
        padding: Padding for conv layers (default: 1)
        activation: Activation function (default: 'relu')
        norm_type: Normalization type ('batch', 'layer', 'none')
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        layer_type: str,
        in_features: int,
        out_features: int,
        conditioning_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        activation: str = 'relu',
        norm_type: str = 'none',
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        # Create main layer
        if layer_type == 'conv1d':
            self.layer = nn.Conv1d(
                in_features, out_features, kernel_size,
                stride=stride, padding=padding
            )
        elif layer_type == 'conv2d':
            self.layer = nn.Conv2d(
                in_features, out_features, kernel_size,
                stride=stride, padding=padding
            )
        elif layer_type == 'linear':
            self.layer = nn.Linear(in_features, out_features)
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
        
        # Create normalization layer
        if norm_type == 'batch':
            if layer_type == 'conv1d':
                self.norm = nn.BatchNorm1d(out_features)
            elif layer_type == 'conv2d':
                self.norm = nn.BatchNorm2d(out_features)
            else:
                self.norm = nn.BatchNorm1d(out_features)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(out_features)
        else:
            self.norm = None
        
        # Create FiLM layer
        self.film = FiLMLayer(out_features, conditioning_dim)
        
        # Create activation
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, activation: str) -> Optional[nn.Module]:
        """Get activation module from string name."""
        if activation == 'none':
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
    
    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with FiLM conditioning.
        
        Args:
            x: Input features
            conditioning: Conditioning information
            
        Returns:
            Output features modulated by conditioning
        """
        # Apply main layer
        x = self.layer(x)
        
        # Apply normalization (if used)
        if self.norm is not None:
            x = self.norm(x)
        
        # Apply FiLM modulation
        x = self.film(x, conditioning)
        
        # Apply activation
        if self.activation is not None:
            x = self.activation(x)
        
        return x


class FiLMGenerator(nn.Module):
    """
    A network that generates FiLM parameters from conditioning input.
    
    This is useful when you want to generate multiple sets of FiLM parameters
    from a single conditioning input, for modulating multiple layers.
    
    Args:
        conditioning_dim: Dimension of conditioning input
        num_layers: Number of FiLM parameter sets to generate
        features_per_layer: List of feature dimensions for each layer
        hidden_dim: Hidden dimension of parameter generation network
        num_hidden_layers: Number of hidden layers (default: 2)
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        conditioning_dim: int,
        num_layers: int,
        features_per_layer: List[int],
        hidden_dim: Optional[int] = None,
        num_hidden_layers: int = 2,
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        assert len(features_per_layer) == num_layers, \
            "features_per_layer must have length num_layers"
        
        self.conditioning_dim = conditioning_dim
        self.num_layers = num_layers
        self.features_per_layer = features_per_layer
        
        # Default hidden dimension
        if hidden_dim is None:
            hidden_dim = conditioning_dim * 2
        
        # Build shared encoder
        encoder_layers = []
        current_dim = conditioning_dim
        
        for _ in range(num_hidden_layers):
            encoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ])
            current_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build parameter generators for each layer
        self.scale_generators = nn.ModuleList()
        self.bias_generators = nn.ModuleList()
        
        for features in features_per_layer:
            self.scale_generators.append(
                nn.Linear(hidden_dim, features)
            )
            self.bias_generators.append(
                nn.Linear(hidden_dim, features)
            )
        
        # Initialize
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize to produce identity transformation initially."""
        for scale_gen in self.scale_generators:
            nn.init.zeros_(scale_gen.weight)
            nn.init.ones_(scale_gen.bias)
        
        for bias_gen in self.bias_generators:
            nn.init.zeros_(bias_gen.weight)
            nn.init.zeros_(bias_gen.bias)
    
    def forward(
        self,
        conditioning: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate FiLM parameters for all layers.
        
        Args:
            conditioning: Conditioning input of shape (batch, conditioning_dim)
            
        Returns:
            List of (scale, bias) tuples for each layer
        """
        # Encode conditioning
        hidden = self.encoder(conditioning)
        
        # Generate parameters for each layer
        film_params = []
        for scale_gen, bias_gen in zip(self.scale_generators, self.bias_generators):
            scale = scale_gen(hidden)
            bias = bias_gen(hidden)
            film_params.append((scale, bias))
        
        return film_params


class ConditionalSequential(nn.Module):
    """
    Sequential container that passes conditioning to all FiLM-enabled layers.
    
    This allows building conditional networks by stacking regular layers
    and FiLM-enabled layers, automatically routing conditioning information.
    
    Args:
        *modules: Variable number of modules to chain
    """
    
    def __init__(self, *modules):
        super().__init__()
        self.modules_list = nn.ModuleList(modules)
    
    def forward(
        self,
        x: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass, routing conditioning to FiLM layers.
        
        Args:
            x: Input features
            conditioning: Optional conditioning information
            
        Returns:
            Output features
        """
        for module in self.modules_list:
            # Check if module expects conditioning
            if isinstance(module, (FiLMLayer, FiLMBlock)):
                if conditioning is not None:
                    x = module(x, conditioning)
                else:
                    # Skip FiLM modules if no conditioning provided
                    continue
            else:
                # Regular module
                x = module(x)
        
        return x