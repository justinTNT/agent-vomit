import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from rave_config_system import RAVEConfig




# TODO: REFACTOR TO CONFIG-FIRST INTERFACE
# New signature: def __init__(self, config: RAVEConfig, base_channels: int, num_layers: int, **kwargs):
# New assignments:
#         self.in_channels = config.convolution.in_channels
        self.base_channels = base_channels
        self.num_layers = num_layers
class ConvEncoder(nn.Module):
    def __init__(self,
                 in_channels: int = 3,      # Input channels
                 base_channels: int = 64,   # Base channel count
                 num_layers: int = 4,       # Number of layers
                 **kwargs):                 # REQUIRED: Accept extra kwargs
        super().__init__()
        
        # Log unknown parameters
        if kwargs:
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        # Store configuration
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_layers = num_layers
        
        # Build encoder layers
        layers = []
        current_channels = in_channels
        
        for i in range(num_layers):
            out_channels = base_channels * (2 ** i)
            
            # Conv block: Conv -> BatchNorm -> ReLU
            layers.append(nn.Sequential(
                nn.Conv2d(current_channels, out_channels, kernel_size=3, 
                         stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
            
            current_channels = out_channels
        
        self.encoder = nn.ModuleList(layers)
        self.final_channels = current_channels
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass of the convolutional encoder.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
        
        Returns:
            Dictionary with keys:
                - 'features': Final feature map before pooling
                - 'pooled': Global pooled features (batch, channels)
                - 'shape': Tuple of (height, width) of final feature map
        """
        # Ensure input has correct number of channels
        assert x.size(1) == self.in_channels, \
            f"Expected {self.in_channels} input channels, got {x.size(1)}"
        
        # Progressive encoding with downsampling
        features = x
        for layer in self.encoder:
            features = layer(features)
        
        # Global pooling
        pooled = self.global_pool(features)
        pooled = pooled.view(pooled.size(0), -1)  # Flatten to (batch, channels)
        
        # Return dictionary with all outputs
        return {
            'features': features,
            'pooled': pooled,
            'shape': (features.size(2), features.size(3))
        }