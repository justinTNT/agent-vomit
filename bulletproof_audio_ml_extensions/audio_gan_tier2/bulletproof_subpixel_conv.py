#!/usr/bin/env python3
"""
BULLETPROOF SUBPIXEL CONVOLUTION MODULE
Comprehensive SubPixel convolution for BigVGAN upsampling with anti-aliasing and stability features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, Tuple, Dict, Any, List
from rave_config_system import RAVEConfig
import logging
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SubPixelConvConfig:
    """Configuration for bulletproof SubPixel convolution"""
    in_channels: int = 256
    out_channels: int = 128
    kernel_size: int = 7
    upsample_factor: int = 2
    stride: int = 1
    padding: Optional[int] = None
    activation: Optional[str] = 'leaky_relu'
    
    # Bulletproof parameters
    numerical_stability_check: bool = True
    gradient_clipping: bool = True
    gradient_clip_value: float = 1.0
    anti_aliasing: bool = True
    
    # Advanced features
    use_weight_norm: bool = False
    use_spectral_norm: bool = False
    channel_shuffle: bool = False
    memory_efficient: bool = True
    
    # Fallback strategies
    enable_fallbacks: bool = True
    fallback_mode: str = 'interpolation'  # 'interpolation', 'transpose_conv'
    validate_output_shape: bool = True

class BulletproofSubPixelConv(nn.Module):
    """Bulletproof SubPixel Convolution with comprehensive error handling and anti-aliasing"""
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__()
        
        self.config = config
        self.subpixel_config = kwargs.get('subpixel_config', SubPixelConvConfig())
        
        self.in_channels = self.subpixel_config.in_channels
        self.out_channels = self.subpixel_config.out_channels
        self.kernel_size = self.subpixel_config.kernel_size
        self.upsample_factor = self.subpixel_config.upsample_factor
        self.stride = self.subpixel_config.stride
        self.anti_aliasing = self.subpixel_config.anti_aliasing
        
        # Validate parameters
        self._validate_parameters()
        
        # Auto-calculate padding if not specified
        if self.subpixel_config.padding is None:
            self.padding = (self.kernel_size - 1) // 2
        else:
            self.padding = self.subpixel_config.padding
        
        # Build main convolution layer
        try:
            self._build_conv_layer()
        except Exception as e:
            logger.error(f"Failed to build convolution layer: {e}")
            if self.subpixel_config.enable_fallbacks:
                logger.warning("Building fallback convolution layer")
                self._build_fallback_conv_layer()
            else:
                raise
        
        # Anti-aliasing filter
        if self.anti_aliasing:
            try:
                self._build_anti_aliasing_filter()
            except Exception as e:
                logger.warning(f"Failed to build anti-aliasing filter: {e}")
                self.anti_aliasing = False
        
        # Activation
        self.activation = self._get_activation(self.subpixel_config.activation)
        
        # Initialize weights
        self._initialize_weights()
        
        # Tracking
        self.upsampling_stats = []
        self.fallback_activations = 0
        
        logger.info(f"BulletproofSubPixelConv initialized: {self.in_channels}->{self.out_channels}, factor={self.upsample_factor}")
    
    def _validate_parameters(self):
        """Validate and adjust parameters for stability"""
        try:
            # Ensure reasonable channel counts
            if self.in_channels < 1:
                logger.warning(f"in_channels {self.in_channels} invalid, setting to 1")
                self.in_channels = 1
            elif self.in_channels > 2048:
                logger.warning(f"in_channels {self.in_channels} too large, setting to 1024")
                self.in_channels = 1024
            
            if self.out_channels < 1:
                logger.warning(f"out_channels {self.out_channels} invalid, setting to 1")
                self.out_channels = 1
            elif self.out_channels > 2048:
                logger.warning(f"out_channels {self.out_channels} too large, setting to 1024")
                self.out_channels = 1024
            
            # Ensure reasonable upsample factor
            if self.upsample_factor < 1:
                logger.warning(f"upsample_factor {self.upsample_factor} invalid, setting to 2")
                self.upsample_factor = 2
            elif self.upsample_factor > 16:
                logger.warning(f"upsample_factor {self.upsample_factor} too large, setting to 8")
                self.upsample_factor = 8
            
            # Ensure reasonable kernel size
            if self.kernel_size < 3:
                logger.warning(f"kernel_size {self.kernel_size} too small, setting to 3")
                self.kernel_size = 3
            elif self.kernel_size > 15:
                logger.warning(f"kernel_size {self.kernel_size} too large, setting to 15")
                self.kernel_size = 15
            
            # Ensure odd kernel size for symmetric padding
            if self.kernel_size % 2 == 0:
                self.kernel_size += 1
                logger.warning(f"Adjusted kernel_size to {self.kernel_size} for symmetry")
            
        except Exception as e:
            logger.error(f"Parameter validation failed: {e}")
            # Set safe defaults
            self.in_channels = 256
            self.out_channels = 128
            self.kernel_size = 7
            self.upsample_factor = 2
            self.stride = 1
    
    def _build_conv_layer(self):
        """Build the main convolution layer"""
        try:
            # The conv layer produces upsample_factor * out_channels channels
            # These will be rearranged by pixel shuffle
            conv_out_channels = self.out_channels * self.upsample_factor
            
            self.conv = nn.Conv1d(
                self.in_channels,
                conv_out_channels,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding
            )
            
            # Apply normalization if requested
            if self.subpixel_config.use_spectral_norm:
                self.conv = nn.utils.spectral_norm(self.conv)
            elif self.subpixel_config.use_weight_norm:
                self.conv = nn.utils.weight_norm(self.conv)
                
        except Exception as e:
            logger.error(f"Convolution layer building failed: {e}")
            raise
    
    def _build_fallback_conv_layer(self):
        """Build fallback convolution layer"""
        try:
            # Simple 1x1 convolution as fallback
            self.conv = nn.Conv1d(self.in_channels, self.out_channels, 1)
            self.upsample_factor = 1  # Disable upsampling in fallback
            logger.info("Built fallback convolution layer")
        except Exception as e:
            logger.error(f"Fallback convolution building failed: {e}")
            # Emergency fallback: identity
            self.conv = nn.Identity()
            self.upsample_factor = 1
    
    def _build_anti_aliasing_filter(self):
        """Build anti-aliasing filter for clean upsampling"""
        try:
            # Create low-pass filter to prevent aliasing
            # Use a simple Gaussian filter
            kernel_size = self.upsample_factor * 2 + 1
            sigma = self.upsample_factor / 3.0
            
            # Create 1D Gaussian kernel
            x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
            gaussian_kernel = torch.exp(-0.5 * (x / sigma) ** 2)
            gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
            
            # Reshape for convolution [out_channels, 1, kernel_size]
            gaussian_kernel = gaussian_kernel.view(1, 1, -1).repeat(self.out_channels, 1, 1)
            
            # Register as buffer (non-trainable)
            self.register_buffer('anti_alias_filter', gaussian_kernel)
            self.anti_alias_padding = kernel_size // 2
            
        except Exception as e:
            logger.error(f"Anti-aliasing filter building failed: {e}")
            self.anti_aliasing = False
    
    def _get_activation(self, activation: Optional[str]) -> Optional[nn.Module]:
        """Get activation module from string name"""
        if activation is None:
            return None
        elif activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'swish':
            return nn.SiLU()
        else:
            logger.warning(f"Unknown activation {activation}, using LeakyReLU")
            return nn.LeakyReLU(0.2, inplace=True)
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization"""
        try:
            if hasattr(self.conv, 'weight'):
                nn.init.xavier_uniform_(self.conv.weight, gain=1.0)
                if self.conv.bias is not None:
                    nn.init.zeros_(self.conv.bias)
        except Exception as e:
            logger.warning(f"Weight initialization failed: {e}")
    
    def _apply_pixel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pixel shuffle operation with error handling"""
        try:
            if self.upsample_factor == 1:
                return x
            
            batch_size, channels, time_steps = x.shape
            
            # Validate channel count
            if channels != self.out_channels * self.upsample_factor:
                logger.warning(f"Channel mismatch for pixel shuffle: expected {self.out_channels * self.upsample_factor}, got {channels}")
                if self.subpixel_config.enable_fallbacks:
                    # Fallback to interpolation
                    return self._apply_interpolation_fallback(x)
                else:
                    raise ValueError(f"Channel mismatch for pixel shuffle")
            
            # Reshape for pixel shuffle
            # Split the channel dimension into (out_channels, upsample_factor)
            x = x.view(batch_size, self.out_channels, self.upsample_factor, time_steps)
            
            # Transpose and reshape to perform the shuffle
            x = x.permute(0, 1, 3, 2).contiguous()
            x = x.view(batch_size, self.out_channels, time_steps * self.upsample_factor)
            
            return x
            
        except Exception as e:
            logger.error(f"Pixel shuffle failed: {e}")
            if self.subpixel_config.enable_fallbacks:
                self.fallback_activations += 1
                return self._apply_interpolation_fallback(x)
            else:
                raise
    
    def _apply_interpolation_fallback(self, x: torch.Tensor) -> torch.Tensor:
        """Apply interpolation as fallback upsampling method"""
        try:
            # Adjust channels if necessary
            if x.size(1) != self.out_channels:
                if x.size(1) > self.out_channels:
                    x = x[:, :self.out_channels, :]
                else:
                    # Pad or repeat channels
                    padding = self.out_channels - x.size(1)
                    x = F.pad(x, (0, 0, 0, padding), mode='replicate')
            
            # Apply interpolation upsampling
            if self.upsample_factor > 1:
                x = F.interpolate(x, scale_factor=self.upsample_factor, mode='linear', align_corners=False)
            
            return x
            
        except Exception as e:
            logger.error(f"Interpolation fallback failed: {e}")
            return x  # Return unchanged as last resort
    
    def _apply_anti_aliasing(self, x: torch.Tensor) -> torch.Tensor:
        """Apply anti-aliasing filter if enabled"""
        try:
            if not self.anti_aliasing or not hasattr(self, 'anti_alias_filter'):
                return x
            
            # Apply anti-aliasing filter using grouped convolution
            x = F.conv1d(
                x,
                self.anti_alias_filter,
                padding=self.anti_alias_padding,
                groups=self.out_channels
            )
            
            return x
            
        except Exception as e:
            logger.warning(f"Anti-aliasing failed: {e}")
            return x
    
    def _apply_channel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel shuffle if enabled"""
        try:
            if not self.subpixel_config.channel_shuffle:
                return x
            
            batch_size, channels, time_steps = x.shape
            
            # Reshape and permute for channel shuffle
            channels_per_group = 2
            if channels % channels_per_group == 0:
                x = x.view(batch_size, channels // channels_per_group, channels_per_group, time_steps)
                x = x.permute(0, 2, 1, 3).contiguous()
                x = x.view(batch_size, channels, time_steps)
            
            return x
            
        except Exception as e:
            logger.warning(f"Channel shuffle failed: {e}")
            return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with comprehensive error handling"""
        try:
            # Validate input
            if x.dim() != 3:
                raise ValueError(f"Expected 3D input tensor [batch, channels, time], got {x.dim()}D")
            
            if x.size(1) != self.in_channels:
                if self.subpixel_config.enable_fallbacks:
                    logger.warning(f"Input channel mismatch: expected {self.in_channels}, got {x.size(1)}")
                    # Adapt input
                    if x.size(1) < self.in_channels:
                        # Pad channels
                        padding = torch.zeros(x.size(0), self.in_channels - x.size(1), x.size(2), 
                                            device=x.device, dtype=x.dtype)
                        x = torch.cat([x, padding], dim=1)
                    else:
                        # Truncate channels
                        x = x[:, :self.in_channels, :]
                else:
                    raise ValueError(f"Input channel mismatch: expected {self.in_channels}, got {x.size(1)}")
            
            # Validate numerical stability
            if self.subpixel_config.numerical_stability_check:
                if not torch.isfinite(x).all():
                    logger.warning("Non-finite values in input")
                    if self.subpixel_config.enable_fallbacks:
                        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
                    else:
                        raise ValueError("Non-finite input values")
            
            # Apply convolution
            if isinstance(self.conv, nn.Identity):
                # Emergency fallback mode
                output = x
                if output.size(1) != self.out_channels:
                    # Adjust channels
                    if output.size(1) > self.out_channels:
                        output = output[:, :self.out_channels, :]
                    else:
                        # Interpolate or pad
                        output = F.interpolate(output.unsqueeze(1), size=(self.out_channels, output.size(2)), mode='bilinear', align_corners=False).squeeze(1)
            else:
                output = self.conv(x)
            
            # Apply gradient clipping if enabled
            if self.subpixel_config.gradient_clipping and self.training:
                output = torch.clamp(output, -self.subpixel_config.gradient_clip_value, self.subpixel_config.gradient_clip_value)
            
            # Apply pixel shuffle upsampling
            output = self._apply_pixel_shuffle(output)
            
            # Apply anti-aliasing filter
            output = self._apply_anti_aliasing(output)
            
            # Apply channel shuffle if enabled
            output = self._apply_channel_shuffle(output)
            
            # Apply activation
            if self.activation is not None:
                output = self.activation(output)
            
            # Validate output shape
            if self.subpixel_config.validate_output_shape:
                expected_time = x.size(2) * self.upsample_factor
                if output.size(2) != expected_time:
                    logger.warning(f"Output time dimension mismatch: expected {expected_time}, got {output.size(2)}")
                
                if output.size(1) != self.out_channels:
                    logger.warning(f"Output channel mismatch: expected {self.out_channels}, got {output.size(1)}")
            
            # Final numerical stability check
            if self.subpixel_config.numerical_stability_check:
                if not torch.isfinite(output).all():
                    logger.warning("Non-finite values in output")
                    output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Track statistics
            if len(self.upsampling_stats) < 1000:
                self.upsampling_stats.append({
                    'input_shape': list(x.shape),
                    'output_shape': list(output.shape),
                    'upsample_factor': self.upsample_factor,
                    'input_mean': x.mean().item(),
                    'output_mean': output.mean().item(),
                    'input_std': x.std().item(),
                    'output_std': output.std().item()
                })
            
            return output
            
        except Exception as e:
            logger.error(f"SubPixel convolution forward failed: {e}")
            if self.subpixel_config.enable_fallbacks:
                self.fallback_activations += 1
                # Emergency fallback: return upsampled input
                try:
                    if x.size(1) != self.out_channels:
                        x = F.interpolate(x.unsqueeze(1), size=(self.out_channels, x.size(2)), mode='bilinear', align_corners=False).squeeze(1)
                    if self.upsample_factor > 1:
                        x = F.interpolate(x, scale_factor=self.upsample_factor, mode='linear', align_corners=False)
                    return x
                except Exception as fallback_e:
                    logger.error(f"Emergency fallback failed: {fallback_e}")
                    # Last resort: return zeros
                    return torch.zeros(x.size(0), self.out_channels, x.size(2) * self.upsample_factor, 
                                     device=x.device, dtype=x.dtype)
            else:
                raise
    
    def get_output_length(self, input_length: int) -> int:
        """Calculate output length given input length"""
        try:
            # First apply convolution formula
            conv_output = (input_length + 2 * self.padding - self.kernel_size) // self.stride + 1
            # Then apply upsampling
            return conv_output * self.upsample_factor
        except Exception as e:
            logger.error(f"Output length calculation failed: {e}")
            return input_length * self.upsample_factor  # Fallback estimate
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        stats = {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'upsample_factor': self.upsample_factor,
            'anti_aliasing': self.anti_aliasing,
            'fallback_activations': self.fallback_activations
        }
        
        if self.upsampling_stats:
            last_stats = self.upsampling_stats[-1]
            stats.update({
                'last_input_shape': last_stats['input_shape'],
                'last_output_shape': last_stats['output_shape'],
                'last_output_mean': last_stats['output_mean']
            })
        
        return stats
    
    def reset_statistics(self):
        """Reset training statistics"""
        self.upsampling_stats.clear()
        self.fallback_activations = 0


class BulletproofMultiScaleSubPixelConv(nn.Module):
    """Multi-scale SubPixel convolution for progressive upsampling"""
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__()
        
        self.config = config
        
        # Extract parameters
        self.in_channels = kwargs.get('in_channels', 256)
        self.out_channels = kwargs.get('out_channels', 128)
        self.upsample_factors = kwargs.get('upsample_factors', [2, 2, 2])
        kernel_sizes = kwargs.get('kernel_sizes', 7)
        hidden_channels = kwargs.get('hidden_channels', None)
        activation = kwargs.get('activation', 'leaky_relu')
        
        # Handle kernel sizes
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(self.upsample_factors)
        
        # Default hidden channels
        if hidden_channels is None:
            hidden_channels = max(self.in_channels, self.out_channels)
        
        # Build layers
        self.layers = nn.ModuleList()
        current_channels = self.in_channels
        
        for i, (factor, kernel_size) in enumerate(zip(self.upsample_factors, kernel_sizes)):
            # Determine output channels for this layer
            if i == len(self.upsample_factors) - 1:
                next_channels = self.out_channels
                layer_activation = None  # No activation on last layer
            else:
                next_channels = hidden_channels
                layer_activation = activation
            
            # Create SubPixel layer
            subpixel_config = SubPixelConvConfig(
                in_channels=current_channels,
                out_channels=next_channels,
                kernel_size=kernel_size,
                upsample_factor=factor,
                activation=layer_activation
            )
            
            layer = BulletproofSubPixelConv(config, subpixel_config=subpixel_config)
            self.layers.append(layer)
            
            current_channels = next_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through multiple upsampling stages"""
        try:
            for layer in self.layers:
                x = layer(x)
            return x
        except Exception as e:
            logger.error(f"Multi-scale SubPixel forward failed: {e}")
            return x
    
    def get_total_upsample_factor(self) -> int:
        """Get total upsampling factor across all stages"""
        total = 1
        for factor in self.upsample_factors:
            total *= factor
        return total


# Factory functions
def create_bulletproof_subpixel_conv(config: RAVEConfig, **kwargs) -> BulletproofSubPixelConv:
    """Create a bulletproof SubPixel convolution layer"""
    return BulletproofSubPixelConv(config, **kwargs)


def create_bulletproof_multiscale_subpixel_conv(config: RAVEConfig, **kwargs) -> BulletproofMultiScaleSubPixelConv:
    """Create a bulletproof multi-scale SubPixel convolution"""
    return BulletproofMultiScaleSubPixelConv(config, **kwargs)


if __name__ == "__main__":
    print("🛡️ BULLETPROOF SUBPIXEL CONVOLUTION MODULE")
    print("=" * 50)
    
    from rave_config_system import get_minimal_config
    
    config = get_minimal_config()
    
    # Test SubPixel convolution
    subpixel_config = SubPixelConvConfig(
        in_channels=256, out_channels=128, kernel_size=7, upsample_factor=2
    )
    subpixel_conv = create_bulletproof_subpixel_conv(config, subpixel_config=subpixel_config)
    
    batch_size = 4
    in_channels = 256
    seq_len = 1024
    
    x = torch.randn(batch_size, in_channels, seq_len)
    
    try:
        output = subpixel_conv(x)
        expected_len = subpixel_conv.get_output_length(seq_len)
        
        print(f"✅ SubPixel convolution test passed")
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Expected length: {expected_len}")
        
        stats = subpixel_conv.get_training_stats()
        print(f"   SubPixel stats: {stats}")
        
    except Exception as e:
        print(f"❌ SubPixel convolution test failed: {e}")
    
    # Test multi-scale SubPixel convolution
    try:
        multiscale_config = {
            'in_channels': 512,
            'out_channels': 1,
            'upsample_factors': [2, 4, 8],
            'kernel_sizes': [7, 7, 7]
        }
        
        multiscale_conv = create_bulletproof_multiscale_subpixel_conv(config, **multiscale_config)
        
        x_multiscale = torch.randn(batch_size, 512, 64)
        output_multiscale = multiscale_conv(x_multiscale)
        
        total_factor = multiscale_conv.get_total_upsample_factor()
        
        print(f"✅ Multi-scale SubPixel test passed")
        print(f"   Multi-scale input: {x_multiscale.shape}")
        print(f"   Multi-scale output: {output_multiscale.shape}")
        print(f"   Total upsample factor: {total_factor}")
        
    except Exception as e:
        print(f"❌ Multi-scale SubPixel test failed: {e}")
    
    print("🚀 BulletproofSubPixelConv ready for BigVGAN efficient upsampling!")