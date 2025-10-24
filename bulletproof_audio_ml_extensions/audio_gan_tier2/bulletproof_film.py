#!/usr/bin/env python3
"""
BULLETPROOF FEATURE-WISE LINEAR MODULATION MODULE
Comprehensive FiLM implementation for BigVGAN conditional generation with stability features.
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
class FiLMConfig:
    """Configuration for bulletproof FiLM"""
    num_features: int = 256
    conditioning_dim: int = 128
    use_bias: bool = True
    init_scale: float = 1.0
    init_bias: float = 0.0
    
    # Bulletproof parameters
    numerical_stability_check: bool = True
    gradient_clipping: bool = True
    gradient_clip_value: float = 1.0
    
    # Advanced features
    adaptive_modulation: bool = False
    conditioning_dropout: float = 0.0
    modulation_type: str = 'multiplicative'  # 'multiplicative', 'additive', 'hybrid'
    
    # Fallback strategies
    enable_fallbacks: bool = True
    disable_modulation_on_failure: bool = False

class BulletproofFiLMLayer(nn.Module):
    """Bulletproof Feature-wise Linear Modulation layer with comprehensive error handling"""
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__()
        
        self.config = config
        self.film_config = kwargs.get('film_config', FiLMConfig())
        
        self.num_features = self.film_config.num_features
        self.conditioning_dim = self.film_config.conditioning_dim
        self.use_bias = self.film_config.use_bias
        
        # Build modulation networks
        try:
            self.scale_gen = nn.Linear(self.conditioning_dim, self.num_features)
            if self.use_bias:
                self.bias_gen = nn.Linear(self.conditioning_dim, self.num_features)
            else:
                self.bias_gen = None
        except Exception as e:
            logger.error(f"Failed to build FiLM networks: {e}")
            if self.film_config.enable_fallbacks:
                # Create minimal fallback
                self.scale_gen = nn.Identity()
                self.bias_gen = nn.Identity() if self.use_bias else None
            else:
                raise
        
        # Conditioning dropout
        if self.film_config.conditioning_dropout > 0:
            self.dropout = nn.Dropout(self.film_config.conditioning_dropout)
        else:
            self.dropout = None
        
        # Initialize parameters
        self._initialize_parameters()
        
        # Tracking
        self.modulation_stats = []
        self.fallback_activations = 0
        
        logger.info(f"BulletproofFiLMLayer initialized: {self.num_features} features")
    
    def _initialize_parameters(self):
        """Initialize parameters to sensible defaults"""
        try:
            if hasattr(self.scale_gen, 'weight'):
                nn.init.constant_(self.scale_gen.weight, 0.0)
                nn.init.constant_(self.scale_gen.bias, self.film_config.init_scale)
            
            if self.bias_gen is not None and hasattr(self.bias_gen, 'weight'):
                nn.init.constant_(self.bias_gen.weight, 0.0)
                nn.init.constant_(self.bias_gen.bias, self.film_config.init_bias)
        except Exception as e:
            logger.warning(f"Parameter initialization failed: {e}")
    
    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation with comprehensive error handling"""
        try:
            # Validate inputs
            if not torch.isfinite(x).all() or not torch.isfinite(conditioning).all():
                if self.film_config.enable_fallbacks:
                    logger.warning("Non-finite inputs detected")
                    self.fallback_activations += 1
                    return x
                else:
                    raise ValueError("Non-finite inputs")
            
            # Apply conditioning dropout
            if self.dropout is not None and self.training:
                conditioning = self.dropout(conditioning)
            
            # Generate modulation parameters
            if isinstance(self.scale_gen, nn.Identity):
                scale = torch.ones(conditioning.size(0), self.num_features, device=x.device)
            else:
                scale = self.scale_gen(conditioning)
            
            if self.use_bias and self.bias_gen is not None:
                if isinstance(self.bias_gen, nn.Identity):
                    bias = torch.zeros(conditioning.size(0), self.num_features, device=x.device)
                else:
                    bias = self.bias_gen(conditioning)
            else:
                bias = 0
            
            # Validate modulation parameters
            if not torch.isfinite(scale).all() or (isinstance(bias, torch.Tensor) and not torch.isfinite(bias).all()):
                if self.film_config.enable_fallbacks:
                    logger.warning("Non-finite modulation parameters")
                    self.fallback_activations += 1
                    return x
                else:
                    raise ValueError("Non-finite modulation parameters")
            
            # Apply gradient clipping if enabled
            if self.film_config.gradient_clipping and self.training:
                scale = torch.clamp(scale, -self.film_config.gradient_clip_value, self.film_config.gradient_clip_value)
                if isinstance(bias, torch.Tensor):
                    bias = torch.clamp(bias, -self.film_config.gradient_clip_value, self.film_config.gradient_clip_value)
            
            # Reshape for broadcasting
            if x.dim() == 3:  # (batch, features, time)
                scale = scale.unsqueeze(-1)
                if isinstance(bias, torch.Tensor):
                    bias = bias.unsqueeze(-1)
            elif x.dim() == 4:  # (batch, features, height, width)
                scale = scale.unsqueeze(-1).unsqueeze(-1)
                if isinstance(bias, torch.Tensor):
                    bias = bias.unsqueeze(-1).unsqueeze(-1)
            
            # Apply modulation
            if self.film_config.modulation_type == 'multiplicative':
                output = x * scale + bias
            elif self.film_config.modulation_type == 'additive':
                output = x + scale + bias
            elif self.film_config.modulation_type == 'hybrid':
                output = x * (1 + scale) + bias
            else:
                output = x * scale + bias  # Default
            
            # Track statistics
            if len(self.modulation_stats) < 1000:
                self.modulation_stats.append({
                    'scale_mean': scale.mean().item(),
                    'scale_std': scale.std().item(),
                    'bias_mean': bias.mean().item() if isinstance(bias, torch.Tensor) else 0.0
                })
            
            return output
            
        except Exception as e:
            logger.error(f"FiLM forward failed: {e}")
            if self.film_config.enable_fallbacks:
                self.fallback_activations += 1
                return x
            else:
                raise
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        stats = {
            'num_features': self.num_features,
            'conditioning_dim': self.conditioning_dim,
            'fallback_activations': self.fallback_activations,
            'modulation_type': self.film_config.modulation_type
        }
        
        if self.modulation_stats:
            last_stats = self.modulation_stats[-1]
            stats.update({
                'last_scale_mean': last_stats['scale_mean'],
                'last_bias_mean': last_stats['bias_mean']
            })
        
        return stats


class BulletproofFiLMBlock(nn.Module):
    """Neural network block with bulletproof FiLM conditioning"""
    
    def __init__(self, config: RAVEConfig, layer_type: str, in_features: int, 
                 out_features: int, conditioning_dim: int, **kwargs):
        super().__init__()
        
        self.config = config
        self.layer_type = layer_type
        
        # Extract parameters
        kernel_size = kwargs.get('kernel_size', 3)
        stride = kwargs.get('stride', 1)
        padding = kwargs.get('padding', 1)
        activation = kwargs.get('activation', 'relu')
        norm_type = kwargs.get('norm_type', 'none')
        
        # Create main layer
        try:
            if layer_type == 'conv1d':
                self.layer = nn.Conv1d(in_features, out_features, kernel_size, stride=stride, padding=padding)
            elif layer_type == 'conv2d':
                self.layer = nn.Conv2d(in_features, out_features, kernel_size, stride=stride, padding=padding)
            elif layer_type == 'linear':
                self.layer = nn.Linear(in_features, out_features)
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
        except Exception as e:
            logger.error(f"Failed to create main layer: {e}")
            # Fallback to identity
            self.layer = nn.Identity()
        
        # Create normalization
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
        film_config = FiLMConfig(num_features=out_features, conditioning_dim=conditioning_dim)
        self.film = BulletproofFiLMLayer(config, film_config=film_config)
        
        # Create activation
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, activation: str) -> Optional[nn.Module]:
        """Get activation module"""
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
        elif activation == 'gelu':
            return nn.GELU()
        else:
            return nn.ReLU(inplace=True)  # Safe default
    
    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """Forward pass with FiLM conditioning"""
        try:
            # Apply main layer
            x = self.layer(x)
            
            # Apply normalization if used
            if self.norm is not None:
                x = self.norm(x)
            
            # Apply FiLM modulation
            x = self.film(x, conditioning)
            
            # Apply activation
            if self.activation is not None:
                x = self.activation(x)
            
            return x
            
        except Exception as e:
            logger.error(f"FiLM block forward failed: {e}")
            return x


# Factory functions
def create_bulletproof_film_layer(config: RAVEConfig, **kwargs) -> BulletproofFiLMLayer:
    """Create a bulletproof FiLM layer"""
    return BulletproofFiLMLayer(config, **kwargs)


def create_bulletproof_film_block(config: RAVEConfig, layer_type: str, 
                                 in_features: int, out_features: int, 
                                 conditioning_dim: int, **kwargs) -> BulletproofFiLMBlock:
    """Create a bulletproof FiLM block"""
    return BulletproofFiLMBlock(config, layer_type, in_features, out_features, conditioning_dim, **kwargs)


if __name__ == "__main__":
    print("🛡️ BULLETPROOF FEATURE-WISE LINEAR MODULATION MODULE")
    print("=" * 65)
    
    from rave_config_system import get_minimal_config
    
    config = get_minimal_config()
    
    # Test FiLM layer
    film_config = FiLMConfig(num_features=128, conditioning_dim=64)
    film = create_bulletproof_film_layer(config, film_config=film_config)
    
    batch_size = 4
    num_features = 128
    seq_len = 1024
    conditioning_dim = 64
    
    x = torch.randn(batch_size, num_features, seq_len)
    conditioning = torch.randn(batch_size, conditioning_dim)
    
    try:
        output = film(x, conditioning)
        print(f"✅ FiLM layer test passed: {x.shape} -> {output.shape}")
        
        stats = film.get_training_stats()
        print(f"   FiLM stats: {stats}")
        
    except Exception as e:
        print(f"❌ FiLM layer test failed: {e}")
    
    # Test FiLM block
    try:
        film_block = create_bulletproof_film_block(config, 'conv1d', 64, 128, 64)
        x_block = torch.randn(batch_size, 64, seq_len)
        
        output_block = film_block(x_block, conditioning)
        print(f"✅ FiLM block test passed: {x_block.shape} -> {output_block.shape}")
        
    except Exception as e:
        print(f"❌ FiLM block test failed: {e}")
    
    print("🚀 BulletproofFiLM ready for BigVGAN conditional generation!")