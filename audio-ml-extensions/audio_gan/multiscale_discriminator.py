import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional


class DiscriminatorBlock(nn.Module):
    """Single discriminator block with downsampling."""
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 15,
                 stride: int = 1,
                 use_spectral_norm: bool = True,
                 activation: str = 'lrelu'):
        super().__init__()
        
        # Padding to maintain reasonable size
        padding = kernel_size // 2
        
        # Convolution layer
        conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        
        # Apply spectral norm if requested
        if use_spectral_norm:
            self.conv = nn.utils.spectral_norm(conv)
        else:
            self.conv = conv
            
        # Activation
        if activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x))


class ScaleDiscriminator(nn.Module):
    """Single scale discriminator."""
    def __init__(self,
                 in_channels: int = 1,
                 channels: List[int] = [16, 64, 256, 1024, 1024],
                 kernel_sizes: List[int] = [15, 41, 41, 41, 5],
                 strides: List[int] = [1, 4, 4, 4, 1],
                 groups: List[int] = [1, 4, 16, 64, 256],
                 use_spectral_norm: bool = True):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # Build discriminator layers
        in_ch = in_channels
        for i, (out_ch, ks, stride, group) in enumerate(zip(channels, kernel_sizes, strides, groups)):
            # Use grouped convolution for efficiency
            conv = nn.Conv1d(in_ch, out_ch, ks, stride, padding=ks//2, groups=min(group, in_ch))
            
            if use_spectral_norm and i < len(channels) - 1:  # Don't apply to final layer
                conv = nn.utils.spectral_norm(conv)
                
            self.layers.append(conv)
            
            # Add activation for all but last layer
            if i < len(channels) - 1:
                self.layers.append(nn.LeakyReLU(0.1))
                
            in_ch = out_ch
            
        # Final layer for binary classification
        self.final = nn.Conv1d(channels[-1], 1, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass returning final output and intermediate features."""
        features = []
        
        for layer in self.layers:
            x = layer(x)
            # Save features from conv layers (skip activations)
            if isinstance(layer, nn.Conv1d):
                features.append(x)
                
        # Final prediction
        output = self.final(x)
        
        return output, features


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator for audio GANs."""
    def __init__(self,
                 num_scales: int = 3,
                 in_channels: int = 1,
                 channels: List[int] = [16, 64, 256, 1024, 1024],
                 kernel_sizes: List[int] = [15, 41, 41, 41, 5],
                 strides: List[int] = [1, 4, 4, 4, 1],
                 groups: List[int] = [1, 4, 16, 64, 256],
                 downsample_factor: int = 4,
                 use_spectral_norm: bool = True):
        super().__init__()
        
        self.num_scales = num_scales
        self.downsample_factor = downsample_factor
        
        # Create discriminators at different scales
        self.discriminators = nn.ModuleList()
        for i in range(num_scales):
            self.discriminators.append(
                ScaleDiscriminator(
                    in_channels=in_channels,
                    channels=channels,
                    kernel_sizes=kernel_sizes,
                    strides=strides,
                    groups=groups,
                    use_spectral_norm=use_spectral_norm
                )
            )
            
        # Downsampling layers
        self.downsamples = nn.ModuleList()
        for i in range(num_scales - 1):
            self.downsamples.append(
                nn.AvgPool1d(kernel_size=downsample_factor, stride=downsample_factor)
            )
            
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Forward pass through all discriminators.
        
        Args:
            x: Input audio [batch, channels, time]
            
        Returns:
            outputs: List of discriminator outputs (one per scale)
            features: List of feature lists (one list per scale)
        """
        outputs = []
        all_features = []
        
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                # Downsample input for this scale
                x = self.downsamples[i - 1](x)
                
            output, features = disc(x)
            outputs.append(output)
            all_features.append(features)
            
        return outputs, all_features
        
    def compute_loss(self, 
                    real_outputs: List[torch.Tensor],
                    fake_outputs: List[torch.Tensor],
                    loss_type: str = 'hinge') -> torch.Tensor:
        """
        Compute discriminator loss.
        
        Args:
            real_outputs: List of discriminator outputs for real data
            fake_outputs: List of discriminator outputs for fake data
            loss_type: 'hinge', 'lsgan', or 'vanilla'
        """
        loss = 0.0
        
        for real_out, fake_out in zip(real_outputs, fake_outputs):
            if loss_type == 'hinge':
                real_loss = torch.mean(torch.relu(1.0 - real_out))
                fake_loss = torch.mean(torch.relu(1.0 + fake_out))
            elif loss_type == 'lsgan':
                real_loss = torch.mean((real_out - 1) ** 2)
                fake_loss = torch.mean(fake_out ** 2)
            else:  # vanilla
                real_loss = torch.mean(F.binary_cross_entropy_with_logits(
                    real_out, torch.ones_like(real_out)
                ))
                fake_loss = torch.mean(F.binary_cross_entropy_with_logits(
                    fake_out, torch.zeros_like(fake_out)
                ))
                
            loss = loss + real_loss + fake_loss
            
        return loss / len(real_outputs)
        
    def compute_generator_loss(self,
                             fake_outputs: List[torch.Tensor],
                             loss_type: str = 'hinge') -> torch.Tensor:
        """
        Compute generator loss from discriminator outputs.
        
        Args:
            fake_outputs: List of discriminator outputs for fake data
            loss_type: 'hinge', 'lsgan', or 'vanilla'
        """
        loss = 0.0
        
        for fake_out in fake_outputs:
            if loss_type == 'hinge':
                loss = loss - torch.mean(fake_out)
            elif loss_type == 'lsgan':
                loss = loss + torch.mean((fake_out - 1) ** 2)
            else:  # vanilla
                loss = loss + torch.mean(F.binary_cross_entropy_with_logits(
                    fake_out, torch.ones_like(fake_out)
                ))
                
        return loss / len(fake_outputs)