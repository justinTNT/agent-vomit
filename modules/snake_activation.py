import torch
import torch.nn as nn
from typing import Optional


class SnakeActivation(nn.Module):
    """
    Snake activation function for audio processing.
    
    A periodic activation function that preserves audio signal characteristics.
    Particularly effective for modeling periodic patterns in audio waveforms.
    
    Formula: snake(x) = x + (1/alpha) * sin^2(alpha * x)
    
    Args:
        channels: Number of channels to apply separate alpha parameters
        alpha_init: Initial value for alpha parameter (frequency control)
        learnable: Whether alpha should be learnable parameter
        shared_alpha: If True, use single alpha for all channels
    
    Reference:
        "Snake: A Flexible Activation Function for Efficient Deep Neural Network Design"
    """
    
    def __init__(
        self,
        channels: int = 1,
        alpha_init: float = 1.0,
        learnable: bool = True,
        shared_alpha: bool = False
    ):
        super().__init__()
        
        self.channels = channels
        self.learnable = learnable
        self.shared_alpha = shared_alpha
        
        # Initialize alpha parameter(s)
        if shared_alpha:
            alpha_shape = (1,)
        else:
            alpha_shape = (channels,)
        
        if learnable:
            self.alpha = nn.Parameter(torch.ones(alpha_shape) * alpha_init)
        else:
            self.register_buffer('alpha', torch.ones(alpha_shape) * alpha_init)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply snake activation.
        
        Args:
            x: Input tensor of shape [batch, channels, ...] or [batch, ...]
        
        Returns:
            Activated tensor of same shape as input
        """
        # Handle different input shapes
        if x.dim() >= 2 and x.size(1) == self.channels and not self.shared_alpha:
            # Standard case: [batch, channels, ...]
            # Reshape alpha for broadcasting
            if x.dim() == 2:
                alpha = self.alpha.view(1, -1)
            elif x.dim() == 3:
                alpha = self.alpha.view(1, -1, 1)
            elif x.dim() == 4:
                alpha = self.alpha.view(1, -1, 1, 1)
            else:
                # General case
                shape = [1] * x.dim()
                shape[1] = -1
                alpha = self.alpha.view(*shape)
        else:
            # Shared alpha or channel mismatch - use broadcasting
            alpha = self.alpha
        
        # Apply snake activation
        return x + (1.0 / alpha) * torch.sin(alpha * x).pow(2)
    
    def extra_repr(self) -> str:
        """String representation for printing."""
        return f'channels={self.channels}, learnable={self.learnable}, shared_alpha={self.shared_alpha}'


class SnakeBeta(nn.Module):
    """
    Extended Snake activation with additional beta parameter.
    
    Formula: snake_beta(x) = x + (1/alpha) * sin^2(alpha * x) + (1/beta) * sin^2(beta * x)
    
    This provides richer frequency modeling capabilities for complex audio signals.
    
    Args:
        channels: Number of channels
        alpha_init: Initial value for alpha parameter
        beta_init: Initial value for beta parameter
        learnable: Whether parameters should be learnable
        shared_params: If True, use single alpha/beta for all channels
    """
    
    def __init__(
        self,
        channels: int = 1,
        alpha_init: float = 1.0,
        beta_init: float = 2.0,
        learnable: bool = True,
        shared_params: bool = False
    ):
        super().__init__()
        
        self.channels = channels
        self.learnable = learnable
        self.shared_params = shared_params
        
        # Initialize parameters
        if shared_params:
            param_shape = (1,)
        else:
            param_shape = (channels,)
        
        if learnable:
            self.alpha = nn.Parameter(torch.ones(param_shape) * alpha_init)
            self.beta = nn.Parameter(torch.ones(param_shape) * beta_init)
        else:
            self.register_buffer('alpha', torch.ones(param_shape) * alpha_init)
            self.register_buffer('beta', torch.ones(param_shape) * beta_init)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply snake-beta activation."""
        # Handle shape broadcasting similar to basic Snake
        if x.dim() >= 2 and x.size(1) == self.channels and not self.shared_params:
            if x.dim() == 2:
                alpha = self.alpha.view(1, -1)
                beta = self.beta.view(1, -1)
            elif x.dim() == 3:
                alpha = self.alpha.view(1, -1, 1)
                beta = self.beta.view(1, -1, 1)
            elif x.dim() == 4:
                alpha = self.alpha.view(1, -1, 1, 1)
                beta = self.beta.view(1, -1, 1, 1)
            else:
                shape = [1] * x.dim()
                shape[1] = -1
                alpha = self.alpha.view(*shape)
                beta = self.beta.view(*shape)
        else:
            alpha = self.alpha
            beta = self.beta
        
        # Apply snake-beta activation
        return x + (1.0 / alpha) * torch.sin(alpha * x).pow(2) + \
               (1.0 / beta) * torch.sin(beta * x).pow(2)


def snake(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Functional interface for Snake activation.
    
    Args:
        x: Input tensor
        alpha: Frequency parameter
    
    Returns:
        Activated tensor
    """
    return x + (1.0 / alpha) * torch.sin(alpha * x).pow(2)