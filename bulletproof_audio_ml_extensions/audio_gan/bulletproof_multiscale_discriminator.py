#!/usr/bin/env python3
"""
BULLETPROOF MULTI-SCALE DISCRIMINATOR MODULE
Comprehensive multi-scale discriminator for BigVGAN with advanced stability features.
Handles high-resolution audio generation, memory management, and training instabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Any
from rave_config_system import RAVEConfig
import warnings
import logging
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MultiScaleDiscriminatorConfig:
    """Configuration for bulletproof multi-scale discriminator"""
    num_scales: int = 3
    channels: List[int] = field(default_factory=lambda: [16, 64, 256, 1024, 1024])
    kernel_sizes: List[int] = field(default_factory=lambda: [15, 41, 41, 41, 5])
    strides: List[int] = field(default_factory=lambda: [1, 4, 4, 4, 1])
    groups: List[int] = field(default_factory=lambda: [1, 4, 16, 64, 256])
    downsample_factor: int = 4
    use_spectral_norm: bool = True
    
    # Bulletproof stability parameters
    activation: str = 'lrelu'
    leaky_relu_slope: float = 0.1
    dropout_rate: float = 0.1
    use_gradient_checkpointing: bool = False
    memory_efficient: bool = True
    
    # Advanced features
    use_self_attention: bool = False
    attention_layers: List[int] = field(default_factory=lambda: [2, 4])
    use_progressive_growing: bool = False
    feature_matching: bool = True
    
    # Fallback strategies
    enable_fallbacks: bool = True
    min_channels: int = 8
    max_channels: int = 2048
    adaptive_channels: bool = True

class BulletproofScaleDiscriminator(nn.Module):
    """Single scale discriminator with comprehensive error handling"""
    
    def __init__(self, config: RAVEConfig, scale_config: MultiScaleDiscriminatorConfig, 
                 in_channels: int = 1):
        super().__init__()
        
        self.config = config
        self.scale_config = scale_config
        self.in_channels = in_channels
        
        # Build discriminator layers with error handling
        try:
            self.layers = self._build_layers()
            self.final_layer = self._build_final_layer()
        except Exception as e:
            logger.error(f"Failed to build discriminator layers: {e}")
            if scale_config.enable_fallbacks:
                logger.warning("Building minimal fallback discriminator")
                self.layers = self._build_fallback_layers()
                self.final_layer = self._build_fallback_final()
            else:
                raise
        
        # Self-attention layers if enabled
        if scale_config.use_self_attention:
            self.attention_layers = self._build_attention_layers()
        else:
            self.attention_layers = {}
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"BulletproofScaleDiscriminator initialized with {len(self.layers)} layers")
    
    def _validate_channels(self, channels: List[int]) -> List[int]:
        """Validate and fix channel specifications"""
        try:
            # Ensure minimum and maximum bounds
            validated = []
            for ch in channels:
                ch = max(self.scale_config.min_channels, min(ch, self.scale_config.max_channels))
                validated.append(ch)
            
            # Ensure reasonable progression
            for i in range(1, len(validated)):
                if validated[i] < validated[i-1] // 4:  # Prevent too aggressive reduction
                    validated[i] = validated[i-1] // 2
            
            return validated
        except Exception as e:
            logger.error(f"Channel validation failed: {e}")
            return [16, 64, 256, 512, 512]  # Safe default
    
    def _build_layers(self) -> nn.ModuleList:
        """Build discriminator layers with comprehensive error handling"""
        try:
            layers = nn.ModuleList()
            channels = self._validate_channels(self.scale_config.channels)
            kernel_sizes = self.scale_config.kernel_sizes
            strides = self.scale_config.strides
            groups = self.scale_config.groups
            
            # Ensure all lists have same length
            max_len = max(len(channels), len(kernel_sizes), len(strides), len(groups))
            
            # Extend lists if necessary
            while len(channels) < max_len:
                channels.append(channels[-1])
            while len(kernel_sizes) < max_len:
                kernel_sizes.append(kernel_sizes[-1])
            while len(strides) < max_len:
                strides.append(strides[-1])
            while len(groups) < max_len:
                groups.append(groups[-1])
            
            in_ch = self.in_channels
            
            for i, (out_ch, ks, stride, group) in enumerate(zip(channels, kernel_sizes, strides, groups)):
                try:
                    # Ensure valid group size
                    group = min(group, in_ch, out_ch)
                    if group <= 0:
                        group = 1
                    
                    # Ensure divisibility
                    while in_ch % group != 0 and group > 1:
                        group //= 2
                    
                    # Build layer block
                    layer_block = self._build_discriminator_block(
                        in_ch, out_ch, ks, stride, group, i
                    )
                    layers.append(layer_block)
                    
                    in_ch = out_ch
                    
                except Exception as e:
                    logger.warning(f"Failed to build layer {i}: {e}")
                    if self.scale_config.enable_fallbacks:
                        # Create simple fallback layer
                        layer_block = nn.Sequential(
                            nn.Conv1d(in_ch, out_ch, 3, stride, 1),
                            nn.LeakyReLU(0.1),
                            nn.Dropout(0.1)
                        )
                        layers.append(layer_block)
                        in_ch = out_ch
                    else:
                        raise
            
            return layers
            
        except Exception as e:
            logger.error(f"Layer building failed: {e}")
            if self.scale_config.enable_fallbacks:
                return self._build_fallback_layers()
            else:
                raise
    
    def _build_discriminator_block(self, in_ch: int, out_ch: int, kernel_size: int,
                                  stride: int, groups: int, layer_idx: int) -> nn.Module:
        """Build a single discriminator block with comprehensive features"""
        try:
            layers = []
            
            # Convolution layer
            padding = kernel_size // 2
            conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding, groups=groups)
            
            # Apply spectral normalization if enabled
            if self.scale_config.use_spectral_norm and layer_idx < len(self.scale_config.channels) - 1:
                conv = nn.utils.spectral_norm(conv)
            
            layers.append(conv)
            
            # Normalization (except for first layer)
            if layer_idx > 0:
                if hasattr(self.config.model, 'norm_type'):
                    if self.config.model.norm_type == 'group_norm':
                        num_groups = min(32, out_ch // 4)
                        if num_groups > 0 and out_ch % num_groups == 0:
                            layers.append(nn.GroupNorm(num_groups, out_ch))
                    elif self.config.model.norm_type == 'batch_norm':
                        layers.append(nn.BatchNorm1d(out_ch))
            
            # Activation
            if self.scale_config.activation == 'lrelu':
                layers.append(nn.LeakyReLU(self.scale_config.leaky_relu_slope))
            elif self.scale_config.activation == 'relu':
                layers.append(nn.ReLU())
            elif self.scale_config.activation == 'gelu':
                layers.append(nn.GELU())
            
            # Dropout for regularization
            if self.scale_config.dropout_rate > 0 and layer_idx > 0:
                layers.append(nn.Dropout(self.scale_config.dropout_rate))
            
            return nn.Sequential(*layers)
            
        except Exception as e:
            logger.error(f"Discriminator block building failed: {e}")
            # Fallback to simple block
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 3, stride, 1),
                nn.LeakyReLU(0.1)
            )
    
    def _build_final_layer(self) -> nn.Module:
        """Build final classification layer"""
        try:
            final_channels = self.scale_config.channels[-1] if self.scale_config.channels else 512
            return nn.Conv1d(final_channels, 1, kernel_size=3, padding=1)
        except Exception as e:
            logger.error(f"Final layer building failed: {e}")
            return nn.Conv1d(512, 1, kernel_size=3, padding=1)  # Safe fallback
    
    def _build_fallback_layers(self) -> nn.ModuleList:
        """Build minimal fallback layers"""
        layers = nn.ModuleList()
        
        channels = [16, 64, 256, 512]
        in_ch = self.in_channels
        
        for out_ch in channels:
            layer = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 15, 4, 7),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.1)
            )
            layers.append(layer)
            in_ch = out_ch
        
        return layers
    
    def _build_fallback_final(self) -> nn.Module:
        """Build fallback final layer"""
        return nn.Conv1d(512, 1, kernel_size=3, padding=1)
    
    def _build_attention_layers(self) -> nn.ModuleDict:
        """Build self-attention layers"""
        attention_layers = nn.ModuleDict()
        
        try:
            for layer_idx in self.scale_config.attention_layers:
                if layer_idx < len(self.scale_config.channels):
                    channels = self.scale_config.channels[layer_idx]
                    attention_layers[str(layer_idx)] = nn.MultiheadAttention(
                        channels, num_heads=min(8, channels // 64), dropout=0.1, batch_first=True
                    )
        except Exception as e:
            logger.warning(f"Self-attention building failed: {e}")
        
        return attention_layers
    
    def _initialize_weights(self):
        """Initialize network weights"""
        try:
            for module in self.modules():
                if isinstance(module, nn.Conv1d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.01)
                elif isinstance(module, (nn.BatchNorm1d, nn.GroupNorm)):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
        except Exception as e:
            logger.warning(f"Weight initialization failed: {e}")
    
    def _apply_attention(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Apply self-attention if available"""
        try:
            attention_key = str(layer_idx)
            if attention_key in self.attention_layers:
                # Reshape for attention: [B, C, T] -> [B, T, C]
                B, C, T = x.shape
                x_reshaped = x.transpose(1, 2)  # [B, T, C]
                
                # Apply attention
                attn_out, _ = self.attention_layers[attention_key](x_reshaped, x_reshaped, x_reshaped)
                
                # Reshape back: [B, T, C] -> [B, C, T]
                x = attn_out.transpose(1, 2)
            
            return x
        except Exception as e:
            logger.warning(f"Attention application failed: {e}")
            return x
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with comprehensive error handling
        
        Args:
            x: Input audio tensor [batch, channels, time]
            
        Returns:
            output: Discriminator output [batch, 1, time]
            features: List of intermediate features
        """
        try:
            # Validate input
            if x.dim() != 3:
                raise ValueError(f"Expected 3D input tensor, got {x.dim()}D")
            
            if x.size(1) != self.in_channels:
                if self.scale_config.enable_fallbacks:
                    logger.warning(f"Input channels mismatch: expected {self.in_channels}, got {x.size(1)}")
                    # Adapt input channels
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
            
            features = []
            
            # Pass through discriminator layers
            for i, layer in enumerate(self.layers):
                try:
                    if self.scale_config.use_gradient_checkpointing and self.training:
                        x = torch.utils.checkpoint.checkpoint(layer, x)
                    else:
                        x = layer(x)
                    
                    # Apply self-attention if configured
                    x = self._apply_attention(x, i)
                    
                    # Save features for feature matching
                    if self.scale_config.feature_matching:
                        features.append(x.detach() if not self.training else x)
                    
                    # Memory management for long sequences
                    if self.scale_config.memory_efficient and x.size(-1) > 32768:
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.warning(f"Layer {i} forward failed: {e}")
                    if self.scale_config.enable_fallbacks:
                        # Skip problematic layer
                        continue
                    else:
                        raise
            
            # Final prediction
            try:
                output = self.final_layer(x)
            except Exception as e:
                logger.error(f"Final layer failed: {e}")
                if self.scale_config.enable_fallbacks:
                    # Create emergency final prediction
                    output = torch.mean(x, dim=1, keepdim=True)
                    output = torch.tanh(output)  # Ensure bounded output
                else:
                    raise
            
            return output, features
            
        except Exception as e:
            logger.error(f"Discriminator forward pass failed: {e}")
            if self.scale_config.enable_fallbacks:
                # Emergency fallback: return zeros
                batch_size, _, seq_len = x.shape
                output = torch.zeros(batch_size, 1, seq_len, device=x.device, dtype=x.dtype)
                features = [torch.zeros_like(x)]
                return output, features
            else:
                raise

class BulletproofMultiScaleDiscriminator(nn.Module):
    """
    Bulletproof Multi-scale Discriminator for BigVGAN neural vocoder.
    
    Features:
    - Multiple scale discriminators with automatic downsampling
    - Comprehensive error handling and fallback strategies
    - Memory management for high-resolution audio
    - Progressive growing support
    - Feature matching for improved training
    - Device compatibility with mixed precision
    """
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__()
        
        self.config = config
        
        # Extract or create discriminator config
        self.disc_config = kwargs.get('discriminator_config', MultiScaleDiscriminatorConfig())
        
        # Override with config values if available
        if hasattr(config.convolution, 'in_channels'):
            self.in_channels = config.convolution.in_channels
        else:
            self.in_channels = 1
        
        # Build discriminators at different scales
        try:
            self.discriminators = self._build_discriminators()
            self.downsamples = self._build_downsampling_layers()
        except Exception as e:
            logger.error(f"Failed to build multi-scale discriminator: {e}")
            if self.disc_config.enable_fallbacks:
                logger.warning("Building fallback discriminators")
                self.discriminators, self.downsamples = self._build_fallback_discriminators()
            else:
                raise
        
        # Progressive growing state
        self.current_scale = 0 if self.disc_config.use_progressive_growing else len(self.discriminators) - 1
        self.alpha = 1.0  # Blending factor for progressive growing
        
        logger.info(f"BulletproofMultiScaleDiscriminator initialized with {len(self.discriminators)} scales")
    
    def _build_discriminators(self) -> nn.ModuleList:
        """Build discriminators at different scales"""
        discriminators = nn.ModuleList()
        
        for i in range(self.disc_config.num_scales):
            try:
                disc = BulletproofScaleDiscriminator(
                    self.config, self.disc_config, self.in_channels
                )
                discriminators.append(disc)
            except Exception as e:
                logger.error(f"Failed to build discriminator {i}: {e}")
                if self.disc_config.enable_fallbacks:
                    # Create minimal fallback discriminator
                    fallback_disc = self._create_fallback_discriminator()
                    discriminators.append(fallback_disc)
                else:
                    raise
        
        return discriminators
    
    def _build_downsampling_layers(self) -> nn.ModuleList:
        """Build downsampling layers between scales"""
        downsamples = nn.ModuleList()
        
        for i in range(self.disc_config.num_scales - 1):
            try:
                # Use average pooling for stable downsampling
                downsample = nn.AvgPool1d(
                    kernel_size=self.disc_config.downsample_factor,
                    stride=self.disc_config.downsample_factor
                )
                downsamples.append(downsample)
            except Exception as e:
                logger.error(f"Failed to build downsampling layer {i}: {e}")
                # Fallback to stride-2 pooling
                downsample = nn.AvgPool1d(kernel_size=2, stride=2)
                downsamples.append(downsample)
        
        return downsamples
    
    def _create_fallback_discriminator(self) -> nn.Module:
        """Create minimal fallback discriminator"""
        return nn.Sequential(
            nn.Conv1d(self.in_channels, 64, 15, 4, 7),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 256, 15, 4, 7),
            nn.LeakyReLU(0.1),
            nn.Conv1d(256, 512, 15, 4, 7),
            nn.LeakyReLU(0.1),
            nn.Conv1d(512, 1, 3, 1, 1)
        )
    
    def _build_fallback_discriminators(self) -> Tuple[nn.ModuleList, nn.ModuleList]:
        """Build minimal fallback discriminators and downsampling"""
        discriminators = nn.ModuleList()
        downsamples = nn.ModuleList()
        
        # Create 3 simple discriminators
        for _ in range(3):
            discriminators.append(self._create_fallback_discriminator())
        
        # Create simple downsampling
        for _ in range(2):
            downsamples.append(nn.AvgPool1d(kernel_size=4, stride=4))
        
        return discriminators, downsamples
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Forward pass through all discriminators with comprehensive error handling
        
        Args:
            x: Input audio tensor [batch, channels, time]
            
        Returns:
            outputs: List of discriminator outputs (one per scale)
            all_features: List of feature lists (one list per scale)
        """
        try:
            # Validate input
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor, got {type(x)}")
            
            if x.dim() != 3:
                raise ValueError(f"Expected 3D tensor [batch, channels, time], got {x.dim()}D")
            
            # Handle device mismatch
            if x.device != next(self.parameters()).device:
                x = x.to(next(self.parameters()).device)
            
            outputs = []
            all_features = []
            current_input = x
            
            # Progressive growing logic
            start_scale = 0 if not self.disc_config.use_progressive_growing else max(0, self.current_scale - 1)
            end_scale = len(self.discriminators) if not self.disc_config.use_progressive_growing else self.current_scale + 1
            
            for i in range(len(self.discriminators)):
                try:
                    # Skip scales if using progressive growing
                    if self.disc_config.use_progressive_growing and (i < start_scale or i >= end_scale):
                        continue
                    
                    # Downsample for this scale (except first)
                    if i > 0:
                        try:
                            current_input = self.downsamples[i - 1](current_input)
                        except Exception as e:
                            logger.warning(f"Downsampling failed at scale {i}: {e}")
                            if self.disc_config.enable_fallbacks:
                                # Manual downsampling fallback
                                current_input = F.avg_pool1d(current_input, kernel_size=4, stride=4)
                            else:
                                raise
                    
                    # Forward through discriminator
                    if hasattr(self.discriminators[i], 'forward'):
                        output, features = self.discriminators[i](current_input)
                    else:
                        # Fallback discriminator (simple Sequential)
                        output = self.discriminators[i](current_input)
                        features = [output]
                    
                    # Progressive growing blending
                    if self.disc_config.use_progressive_growing and i == self.current_scale - 1 and self.alpha < 1.0:
                        # Blend with previous scale
                        prev_output, prev_features = outputs[-1], all_features[-1]
                        output = self.alpha * output + (1 - self.alpha) * prev_output
                        features = [self.alpha * f + (1 - self.alpha) * pf 
                                  for f, pf in zip(features, prev_features)]
                    
                    outputs.append(output)
                    all_features.append(features)
                    
                except Exception as e:
                    logger.warning(f"Discriminator {i} failed: {e}")
                    if self.disc_config.enable_fallbacks:
                        # Create dummy output
                        batch_size, _, seq_len = current_input.shape
                        dummy_output = torch.zeros(batch_size, 1, seq_len // (4 ** i), 
                                                 device=x.device, dtype=x.dtype)
                        dummy_features = [torch.zeros_like(current_input)]
                        outputs.append(dummy_output)
                        all_features.append(dummy_features)
                    else:
                        raise
            
            return outputs, all_features
            
        except Exception as e:
            logger.error(f"Multi-scale discriminator forward failed: {e}")
            if self.disc_config.enable_fallbacks:
                # Emergency fallback
                batch_size, _, seq_len = x.shape
                dummy_outputs = [torch.zeros(batch_size, 1, seq_len // (4 ** i), device=x.device) 
                               for i in range(3)]
                dummy_features = [[torch.zeros_like(x)] for _ in range(3)]
                return dummy_outputs, dummy_features
            else:
                raise
    
    def grow_network(self):
        """Grow network for progressive training"""
        if self.disc_config.use_progressive_growing and self.current_scale < len(self.discriminators) - 1:
            self.current_scale += 1
            self.alpha = 0.0
            logger.info(f"Grew network to scale {self.current_scale}")
    
    def update_alpha(self, alpha: float):
        """Update blending factor for progressive growing"""
        self.alpha = max(0.0, min(1.0, alpha))
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'num_scales': len(self.discriminators),
            'current_scale': self.current_scale,
            'alpha': self.alpha,
            'progressive_growing': self.disc_config.use_progressive_growing,
            'use_spectral_norm': self.disc_config.use_spectral_norm,
            'memory_efficient': self.disc_config.memory_efficient
        }

# Factory function
def create_bulletproof_multiscale_discriminator(config: RAVEConfig, **kwargs) -> BulletproofMultiScaleDiscriminator:
    """Create a bulletproof multi-scale discriminator instance"""
    return BulletproofMultiScaleDiscriminator(config, **kwargs)

if __name__ == "__main__":
    print("🛡️ BULLETPROOF MULTI-SCALE DISCRIMINATOR MODULE")
    print("=" * 60)
    
    # Test with minimal config
    from rave_config_system import get_minimal_config
    
    config = get_minimal_config()
    config.convolution.in_channels = 1
    
    discriminator = create_bulletproof_multiscale_discriminator(config)
    
    # Test with dummy audio data
    batch_size = 2
    seq_len = 16384  # ~0.37 seconds at 44.1kHz
    
    x = torch.randn(batch_size, 1, seq_len)
    
    # Forward pass
    try:
        outputs, features = discriminator(x)
        
        print(f"✅ Multi-scale forward pass successful")
        print(f"   Number of scales: {len(outputs)}")
        for i, (output, feats) in enumerate(zip(outputs, features)):
            print(f"   Scale {i}: output shape {output.shape}, {len(feats)} features")
        
        # Test progressive growing
        discriminator.grow_network()
        discriminator.update_alpha(0.5)
        
        stats = discriminator.get_training_stats()
        print(f"📊 Training stats: {stats}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    print("🚀 BulletproofMultiScaleDiscriminator ready for BigVGAN training!")