
import torch
import torch.nn as nn
from rave_config_system import RAVEConfig

class ConfigFirstConvEncoder(nn.Module):
    """Convolution Encoder with config-first, over-clockable interface"""
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__()
        
        # All parameters from config - enables extreme architectures
        self.in_channels = config.convolution.in_channels
        self.base_channels = config.model.base_channels
        self.latent_dim = config.model.latent_dim
        self.num_layers = config.model.n_layers
        self.kernel_size = config.convolution.kernel_size
        self.stride = config.convolution.stride
        self.activation = config.model.activation
        self.use_spectral_norm = config.convolution.use_spectral_norm
        self.norm_type = config.convolution.norm_type
        
        # Build over-clockable encoder
        layers = []
        in_ch = self.in_channels
        
        for i in range(self.num_layers):
            out_ch = min(self.base_channels * (2 ** i), config.model.max_channels)
            
            conv = nn.Conv1d(
                in_ch, out_ch, 
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.kernel_size // 2,
                bias=config.convolution.bias
            )
            
            if self.use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            
            layers.append(conv)
            
            # Add normalization
            if self.norm_type == 'batch_norm':
                layers.append(nn.BatchNorm1d(out_ch))
            elif self.norm_type == 'layer_norm':
                layers.append(nn.LayerNorm(out_ch))
            elif self.norm_type == 'group_norm':
                layers.append(nn.GroupNorm(8, out_ch))
            
            # Add activation
            layers.append(self._get_activation())
            
            in_ch = out_ch
        
        # Final projection to latent space
        layers.extend([
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_ch, self.latent_dim)
        ])
        
        self.encoder = nn.Sequential(*layers)
    
    def _get_activation(self):
        """Get activation function from config"""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(), 
            'swish': nn.SiLU(),
            'mish': nn.Mish()
        }
        return activations.get(self.activation, nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> dict:
        """Encode input to latent space"""
        # Ensure correct input shape for 1D conv
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        
        latent = self.encoder(x)
        
        return {
            'latent': latent,
            'shape_info': {
                'input': tuple(x.shape),
                'latent': tuple(latent.shape)
            }
        }
