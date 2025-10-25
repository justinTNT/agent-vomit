import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import math


class SnakeActivation(nn.Module):
    def __init__(self,
                 n_channels: int = 1,           # Number of channels
                 alpha_init: float = 1.0,       # Initial alpha value
                 learnable: bool = True,        # Whether alpha is learnable
                 shared_alpha: bool = False,    # Share alpha across channels
                 **kwargs):                     # REQUIRED: Accept extra kwargs
        super().__init__()
        
        # Log unknown parameters
        if kwargs:
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.n_channels = n_channels
        self.alpha_init = alpha_init
        self.learnable = learnable
        self.shared_alpha = shared_alpha
        
        # Initialize alpha parameter(s)
        if shared_alpha:
            # Single alpha for all channels
            alpha_shape = (1,)
        else:
            # Separate alpha for each channel
            alpha_shape = (n_channels,)
        
        if learnable:
            # Learnable parameter
            self.alpha = nn.Parameter(torch.ones(alpha_shape) * alpha_init)
        else:
            # Fixed buffer
            self.register_buffer('alpha', torch.ones(alpha_shape) * alpha_init)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Snake activation: x + (1/alpha) * sin²(alpha * x)
        
        Args:
            x: Input tensor of shape (batch, features) or (batch, channels, time)
        
        Returns:
            Activated tensor of same shape as input
        """
        # Handle different input dimensions
        if x.dim() == 2:
            # 2D input: (batch, features)
            if self.shared_alpha or self.n_channels == 1:
                # Use single alpha
                alpha = self.alpha
            else:
                # Ensure we have the right number of channels
                if x.size(1) != self.n_channels:
                    raise ValueError(
                        f"Expected {self.n_channels} channels, got {x.size(1)}"
                    )
                # Reshape alpha for broadcasting
                alpha = self.alpha.unsqueeze(0)  # (1, n_channels)
        
        elif x.dim() == 3:
            # 3D input: (batch, channels, time)
            if x.size(1) != self.n_channels and not self.shared_alpha:
                raise ValueError(
                    f"Expected {self.n_channels} channels, got {x.size(1)}"
                )
            
            if self.shared_alpha:
                # Use single alpha for all channels
                alpha = self.alpha
            else:
                # Reshape alpha for broadcasting
                alpha = self.alpha.view(1, -1, 1)  # (1, n_channels, 1)
        
        else:
            raise ValueError(
                f"SnakeActivation expects 2D or 3D input, got {x.dim()}D"
            )
        
        # Apply Snake activation
        # snake(x) = x + (1/alpha) * sin²(alpha * x)
        alpha_x = alpha * x
        sine_squared = torch.sin(alpha_x) ** 2
        output = x + (sine_squared / alpha)
        
        return output
    
    def extra_repr(self) -> str:
        """String representation with module details."""
        return (f'n_channels={self.n_channels}, '
                f'alpha_init={self.alpha_init}, '
                f'learnable={self.learnable}, '
                f'shared_alpha={self.shared_alpha}')