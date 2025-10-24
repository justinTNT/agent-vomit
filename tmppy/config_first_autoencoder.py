
import torch
import torch.nn as nn
import torch.nn.functional as F
from rave_config_system import RAVEConfig

class ConfigFirstConvDecoder(nn.Module):
    """Config-first convolution decoder"""
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__()
        
        self.latent_dim = config.model.latent_dim
        self.base_channels = config.model.base_channels
        self.num_layers = config.model.n_layers
        self.out_channels = config.convolution.out_channels
        self.kernel_size = config.convolution.kernel_size
        self.stride = config.convolution.stride
        
        # Initial projection to feature maps
        initial_channels = self.base_channels * (2 ** (self.num_layers - 1))
        initial_length = 64  # Start with reasonable length
        
        self.initial_proj = nn.Linear(self.latent_dim, initial_channels * initial_length)
        self.initial_channels = initial_channels
        self.initial_length = initial_length
        
        # Build decoder layers
        layers = []
        in_ch = initial_channels
        
        for i in range(self.num_layers):
            out_ch = self.base_channels * (2 ** max(self.num_layers - i - 2, 0))
            
            if i == self.num_layers - 1:
                # Final layer
                layers.extend([
                    nn.ConvTranspose1d(in_ch, self.out_channels, 
                                     kernel_size=self.kernel_size,
                                     stride=self.stride, 
                                     padding=self.kernel_size // 2),
                    nn.Tanh()
                ])
            else:
                layers.extend([
                    nn.ConvTranspose1d(in_ch, out_ch,
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     padding=self.kernel_size // 2),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU()
                ])
            
            in_ch = out_ch
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to output"""
        # Project and reshape
        x = self.initial_proj(z)
        x = x.view(x.size(0), self.initial_channels, self.initial_length)
        
        # Decode
        x = self.decoder(x)
        
        return x

class ConfigFirstAutoEncoder(nn.Module):
    """AutoEncoder with config-first interface"""
    
    def __init__(self, config: RAVEConfig, encoder=None, decoder=None, **kwargs):
        super().__init__()
        
        self.latent_dim = config.model.latent_dim
        
        # Use provided encoder/decoder or create from config
        if encoder is None:
            from config_first_conv_encoder import ConfigFirstConvEncoder
            self.encoder = ConfigFirstConvEncoder(config)
        else:
            self.encoder = encoder
        
        if decoder is None:
            self.decoder = ConfigFirstConvDecoder(config)
        else:
            self.decoder = decoder
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input"""
        output = self.encoder(x)
        return output['latent'] if isinstance(output, dict) else output
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass"""
        z = self.encode(x)
        x_recon = self.decode(z)
        
        return {
            'reconstruction': x_recon,
            'latent': z
        }
