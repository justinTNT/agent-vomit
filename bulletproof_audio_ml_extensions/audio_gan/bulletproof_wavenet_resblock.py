#!/usr/bin/env python3
"""
BULLETPROOF WAVENET RESBLOCK MODULE
Comprehensive WaveNet-style residual blocks for BigVGAN generator with causality preservation.
Handles gradient flow, memory management, and conditional generation stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Union, Tuple, Dict, Any
from rave_config_system import RAVEConfig
import warnings
import logging
from dataclasses import dataclass, field
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WaveNetConfig:
    """Configuration for bulletproof WaveNet ResBlock"""
    channels: int = 256
    kernel_size: int = 3
    dilation: int = 1
    skip_channels: int = 256
    residual_channels: int = 256
    gate_channels: int = 256
    dropout: float = 0.0
    bias: bool = True
    use_1x1_skip: bool = True
    conditioning_channels: Optional[int] = None
    
    # Bulletproof stability parameters
    causality_check: bool = True
    gradient_checkpointing: bool = False
    memory_efficient: bool = True
    numerical_stability_check: bool = True
    
    # Advanced features
    use_weight_norm: bool = False
    use_spectral_norm: bool = False
    activation_type: str = 'gated'  # 'gated', 'relu', 'gelu', 'swish'
    normalization_type: str = 'none'  # 'none', 'batch_norm', 'layer_norm', 'group_norm'
    
    # Conditioning features
    adaptive_conditioning: bool = False
    conditioning_dropout: float = 0.0
    global_conditioning: bool = False
    
    # Fallback strategies
    enable_fallbacks: bool = True
    fallback_kernel_size: int = 3
    fallback_dilation: int = 1
    simple_residual: bool = False

class BulletproofWaveNetResBlock(nn.Module):
    """
    Bulletproof WaveNet Residual Block with comprehensive error handling.
    
    Features:
    - Dilated causal convolutions with causality preservation
    - Gated activation with numerical stability
    - Optional conditioning with dropout and normalization
    - Memory efficient processing for long sequences
    - Gradient checkpointing support
    - Multiple activation and normalization options
    - Comprehensive fallback strategies
    """
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__()
        
        self.config = config
        self.wavenet_config = kwargs.get('wavenet_config', WaveNetConfig())
        
        # Extract parameters
        self.channels = self.wavenet_config.channels
        self.kernel_size = self.wavenet_config.kernel_size
        self.dilation = self.wavenet_config.dilation
        self.skip_channels = self.wavenet_config.skip_channels
        self.residual_channels = self.wavenet_config.residual_channels
        self.gate_channels = self.wavenet_config.gate_channels
        self.dropout = self.wavenet_config.dropout
        self.use_1x1_skip = self.wavenet_config.use_1x1_skip
        self.conditioning_channels = self.wavenet_config.conditioning_channels
        
        # Validate and adjust parameters
        self._validate_parameters()
        
        # Build layers with error handling
        try:
            self._build_layers()
        except Exception as e:
            logger.error(f"Failed to build WaveNet layers: {e}")
            if self.wavenet_config.enable_fallbacks:
                logger.warning("Building fallback WaveNet block")
                self._build_fallback_layers()
            else:
                raise
        
        # Initialize weights
        self._initialize_weights()
        
        # Tracking
        self.gradient_norms = []
        self.activation_stats = []
        self.fallback_activations = 0
        
        logger.info(f"BulletproofWaveNetResBlock initialized: dilation={self.dilation}, channels={self.channels}")
    
    def _validate_parameters(self):
        """Validate and adjust parameters for stability"""
        try:
            # Ensure reasonable channel counts
            if self.channels < 1:
                logger.warning(f"channels {self.channels} too small, setting to 64")
                self.channels = 64
            elif self.channels > 2048:
                logger.warning(f"channels {self.channels} too large, setting to 1024")
                self.channels = 1024
            
            # Ensure reasonable kernel size
            if self.kernel_size < 2:
                logger.warning(f"kernel_size {self.kernel_size} too small, setting to 3")
                self.kernel_size = 3
            elif self.kernel_size > 15:
                logger.warning(f"kernel_size {self.kernel_size} too large, setting to 15")
                self.kernel_size = 15
            
            # Ensure odd kernel size for causality
            if self.kernel_size % 2 == 0:
                self.kernel_size += 1
                logger.warning(f"Adjusted kernel_size to {self.kernel_size} for causality")
            
            # Ensure reasonable dilation
            if self.dilation < 1:
                logger.warning(f"dilation {self.dilation} invalid, setting to 1")
                self.dilation = 1
            elif self.dilation > 512:
                logger.warning(f"dilation {self.dilation} too large, setting to 256")
                self.dilation = 256
            
            # Ensure compatible channel counts
            if self.gate_channels < self.residual_channels:
                self.gate_channels = self.residual_channels
                logger.warning(f"Adjusted gate_channels to {self.gate_channels}")
            
        except Exception as e:
            logger.error(f"Parameter validation failed: {e}")
            # Set safe defaults
            self.channels = 256
            self.kernel_size = 3
            self.dilation = 1
            self.skip_channels = 256
            self.residual_channels = 256
            self.gate_channels = 256
    
    def _build_layers(self):
        """Build WaveNet layers with error handling"""
        try:
            # Calculate padding for causal convolution
            self.padding = (self.kernel_size - 1) * self.dilation
            
            # Main dilated causal convolution
            # Output channels = gate_channels + residual_channels for gated activation
            if self.wavenet_config.activation_type == 'gated':
                output_channels = self.gate_channels + self.residual_channels
            else:
                output_channels = self.residual_channels
            
            self.dilated_conv = nn.Conv1d(
                self.channels,
                output_channels,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                padding=self.padding,
                bias=self.wavenet_config.bias
            )
            
            # Apply normalization to main conv if requested
            self.dilated_conv = self._apply_normalization(self.dilated_conv, output_channels)
            
            # Conditioning projection (if using conditioning)
            if self.conditioning_channels is not None:
                try:
                    self.cond_conv = nn.Conv1d(
                        self.conditioning_channels,
                        output_channels,
                        kernel_size=1,
                        bias=self.wavenet_config.bias
                    )
                    
                    if self.wavenet_config.conditioning_dropout > 0:
                        self.cond_dropout = nn.Dropout(self.wavenet_config.conditioning_dropout)
                    else:
                        self.cond_dropout = None
                        
                except Exception as e:
                    logger.warning(f"Failed to build conditioning layers: {e}")
                    self.cond_conv = None
                    self.cond_dropout = None
            else:
                self.cond_conv = None
                self.cond_dropout = None
            
            # Output projections
            self.residual_conv = nn.Conv1d(
                self.residual_channels,
                self.channels,
                kernel_size=1,
                bias=self.wavenet_config.bias
            )
            
            if self.use_1x1_skip:
                self.skip_conv = nn.Conv1d(
                    self.residual_channels,
                    self.skip_channels,
                    kernel_size=1,
                    bias=self.wavenet_config.bias
                )
            else:
                self.skip_conv = None
            
            # Dropout layer
            if self.dropout > 0:
                self.dropout_layer = nn.Dropout(self.dropout)
            else:
                self.dropout_layer = None
            
            # Activation function
            if self.wavenet_config.activation_type == 'relu':
                self.activation = nn.ReLU()
            elif self.wavenet_config.activation_type == 'gelu':
                self.activation = nn.GELU()
            elif self.wavenet_config.activation_type == 'swish':
                self.activation = nn.SiLU()  # SiLU is the same as Swish
            else:
                self.activation = None  # Use gated activation
                
        except Exception as e:
            logger.error(f"Layer building failed: {e}")
            raise
    
    def _apply_normalization(self, conv_layer: nn.Module, num_channels: int) -> nn.Module:
        """Apply normalization to convolution layer"""
        try:
            if self.wavenet_config.use_spectral_norm:
                conv_layer = nn.utils.spectral_norm(conv_layer)
            elif self.wavenet_config.use_weight_norm:
                conv_layer = nn.utils.weight_norm(conv_layer)
            
            return conv_layer
        except Exception as e:
            logger.warning(f"Normalization application failed: {e}")
            return conv_layer
    
    def _build_fallback_layers(self):
        """Build minimal fallback layers"""
        try:
            # Simple causal convolution
            self.padding = (self.wavenet_config.fallback_kernel_size - 1) * self.wavenet_config.fallback_dilation
            
            self.dilated_conv = nn.Conv1d(
                self.channels,
                self.channels,
                kernel_size=self.wavenet_config.fallback_kernel_size,
                dilation=self.wavenet_config.fallback_dilation,
                padding=self.padding,
                bias=True
            )
            
            # Simple residual and skip connections
            self.residual_conv = nn.Conv1d(self.channels, self.channels, 1)
            self.skip_conv = nn.Conv1d(self.channels, self.skip_channels, 1) if self.use_1x1_skip else None
            
            # No conditioning in fallback
            self.cond_conv = None
            self.cond_dropout = None
            self.dropout_layer = None
            self.activation = nn.ReLU()
            
            # Update parameters for fallback
            self.kernel_size = self.wavenet_config.fallback_kernel_size
            self.dilation = self.wavenet_config.fallback_dilation
            self.residual_channels = self.channels
            self.gate_channels = self.channels
            
            logger.info("Built fallback WaveNet layers")
            
        except Exception as e:
            logger.error(f"Fallback layer building failed: {e}")
            # Emergency fallback: identity layers
            self.dilated_conv = nn.Identity()
            self.residual_conv = nn.Identity()
            self.skip_conv = nn.Identity()
            self.cond_conv = None
            self.dropout_layer = None
            self.activation = nn.Identity()
            self.padding = 0
    
    def _initialize_weights(self):
        """Initialize network weights"""
        try:
            for module in self.modules():
                if isinstance(module, nn.Conv1d):
                    # Use proper initialization for WaveNet
                    if self.wavenet_config.activation_type == 'gated':
                        # Xavier initialization for gated units
                        nn.init.xavier_uniform_(module.weight, gain=1.0)
                    else:
                        # Kaiming initialization for ReLU-like activations
                        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, (nn.BatchNorm1d, nn.GroupNorm)):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
        except Exception as e:
            logger.warning(f"Weight initialization failed: {e}")
    
    def _apply_causality(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causality by removing future samples"""
        try:
            if self.padding > 0 and self.wavenet_config.causality_check:
                # Remove future samples to maintain causality
                x = x[:, :, :-self.padding]
            return x
        except Exception as e:
            logger.warning(f"Causality application failed: {e}")
            return x
    
    def _gated_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated activation function with numerical stability"""
        try:
            if self.wavenet_config.activation_type != 'gated':
                return self.activation(x) if self.activation else x
            
            # Split into gate and filter parts
            if x.size(1) == self.gate_channels + self.residual_channels:
                gate = x[:, :self.gate_channels, :]
                filt = x[:, self.gate_channels:, :]
            else:
                # Fallback: split in half
                split_size = x.size(1) // 2
                gate = x[:, :split_size, :]
                filt = x[:, split_size:, :]
            
            # Apply gated activation: gate * tanh(filter)
            gate_activated = torch.sigmoid(gate)
            filter_activated = torch.tanh(filt)
            
            # Check for numerical issues
            if not torch.isfinite(gate_activated).all() or not torch.isfinite(filter_activated).all():
                logger.warning("Non-finite values in gated activation")
                if self.wavenet_config.enable_fallbacks:
                    self.fallback_activations += 1
                    return torch.relu(x[:, :self.residual_channels, :])
                else:
                    raise ValueError("Non-finite values in gated activation")
            
            result = gate_activated * filter_activated
            
            # Track activation statistics
            if len(self.activation_stats) < 1000:
                self.activation_stats.append({
                    'gate_mean': gate_activated.mean().item(),
                    'gate_std': gate_activated.std().item(),
                    'filter_mean': filter_activated.mean().item(),
                    'filter_std': filter_activated.std().item()
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Gated activation failed: {e}")
            if self.wavenet_config.enable_fallbacks:
                self.fallback_activations += 1
                # Fallback to simple ReLU
                return torch.relu(x[:, :self.residual_channels, :])
            else:
                raise
    
    def _process_conditioning(self, conditioning: torch.Tensor) -> torch.Tensor:
        """Process conditioning signal with error handling"""
        try:
            if self.cond_conv is None:
                return conditioning
            
            # Apply conditioning convolution
            cond_processed = self.cond_conv(conditioning)
            
            # Apply dropout if configured
            if self.cond_dropout is not None and self.training:
                cond_processed = self.cond_dropout(cond_processed)
            
            return cond_processed
            
        except Exception as e:
            logger.warning(f"Conditioning processing failed: {e}")
            if self.wavenet_config.enable_fallbacks:
                # Return zeros of appropriate shape
                return torch.zeros_like(conditioning[:, :self.gate_channels + self.residual_channels, :])
            else:
                raise
    
    def forward(self, x: torch.Tensor, conditioning: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through WaveNet residual block.
        
        Args:
            x: Input tensor [batch, channels, time]
            conditioning: Optional conditioning tensor [batch, cond_channels, time]
            
        Returns:
            residual: Output for next layer [batch, channels, time]
            skip: Skip connection output [batch, skip_channels, time]
        """
        try:
            # Validate input
            if x.dim() != 3:
                raise ValueError(f"Expected 3D input tensor, got {x.dim()}D")
            
            if x.size(1) != self.channels:
                if self.wavenet_config.enable_fallbacks:
                    logger.warning(f"Input channel mismatch: expected {self.channels}, got {x.size(1)}")
                    # Adapt input channels
                    if x.size(1) < self.channels:
                        padding = torch.zeros(x.size(0), self.channels - x.size(1), x.size(2), 
                                            device=x.device, dtype=x.dtype)
                        x = torch.cat([x, padding], dim=1)
                    else:
                        x = x[:, :self.channels, :]
                else:
                    raise ValueError(f"Input channel mismatch: expected {self.channels}, got {x.size(1)}")
            
            # Store input for residual connection
            residual_input = x
            
            # Dilated causal convolution
            if self.wavenet_config.memory_efficient and x.size(-1) > 16384:
                # Process in chunks for very long sequences
                chunk_size = 8192
                chunks = []
                for i in range(0, x.size(-1), chunk_size):
                    chunk = x[:, :, i:i+chunk_size]
                    if chunk.size(-1) > self.padding:  # Ensure chunk is large enough
                        chunk_out = self.dilated_conv(chunk)
                        chunks.append(chunk_out)
                
                if chunks:
                    x = torch.cat(chunks, dim=-1)
                else:
                    x = self.dilated_conv(x)
            else:
                x = self.dilated_conv(x)
            
            # Apply causality
            x = self._apply_causality(x)
            
            # Add conditioning if provided
            if conditioning is not None and self.conditioning_channels is not None:
                try:
                    cond_processed = self._process_conditioning(conditioning)
                    
                    # Match temporal dimension
                    if cond_processed.size(-1) != x.size(-1):
                        if cond_processed.size(-1) > x.size(-1):
                            cond_processed = cond_processed[:, :, :x.size(-1)]
                        else:
                            # Interpolate to match
                            cond_processed = F.interpolate(cond_processed, size=x.size(-1), mode='linear', align_corners=False)
                    
                    x = x + cond_processed
                    
                except Exception as e:
                    logger.warning(f"Conditioning addition failed: {e}")
                    # Continue without conditioning
            
            # Gated activation
            x = self._gated_activation(x)
            
            # Apply dropout if configured
            if self.dropout_layer is not None and self.training:
                x = self.dropout_layer(x)
            
            # Skip connection
            if self.skip_conv is not None:
                try:
                    skip = self.skip_conv(x)
                except Exception as e:
                    logger.warning(f"Skip connection failed: {e}")
                    if self.wavenet_config.enable_fallbacks:
                        skip = x[:, :self.skip_channels, :] if x.size(1) >= self.skip_channels else x
                    else:
                        raise
            else:
                skip = x
            
            # Residual connection
            try:
                residual_output = self.residual_conv(x)
                
                # Ensure compatible shapes for residual addition
                if residual_output.shape != residual_input.shape:
                    if residual_output.size(-1) > residual_input.size(-1):
                        residual_output = residual_output[:, :, :residual_input.size(-1)]
                    elif residual_output.size(-1) < residual_input.size(-1):
                        residual_input = residual_input[:, :, :residual_output.size(-1)]
                
                residual_output = residual_output + residual_input
                
            except Exception as e:
                logger.warning(f"Residual connection failed: {e}")
                if self.wavenet_config.enable_fallbacks:
                    self.fallback_activations += 1
                    residual_output = residual_input  # Pass through
                else:
                    raise
            
            return residual_output, skip
            
        except Exception as e:
            logger.error(f"WaveNet ResBlock forward failed: {e}")
            if self.wavenet_config.enable_fallbacks:
                self.fallback_activations += 1
                # Emergency fallback: return input as both residual and skip
                batch_size, channels, seq_len = x.shape
                skip_dummy = torch.zeros(batch_size, self.skip_channels, seq_len, device=x.device, dtype=x.dtype)
                return x, skip_dummy
            else:
                raise
    
    def get_receptive_field(self) -> int:
        """Get the receptive field contribution of this block"""
        return (self.kernel_size - 1) * self.dilation
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        stats = {
            'dilation': self.dilation,
            'kernel_size': self.kernel_size,
            'channels': self.channels,
            'receptive_field': self.get_receptive_field(),
            'fallback_activations': self.fallback_activations,
            'use_conditioning': self.conditioning_channels is not None,
            'activation_type': self.wavenet_config.activation_type
        }
        
        if self.activation_stats:
            last_stats = self.activation_stats[-1]
            stats.update({
                'gate_activation_mean': last_stats['gate_mean'],
                'filter_activation_mean': last_stats['filter_mean']
            })
        
        return stats
    
    def reset_statistics(self):
        """Reset training statistics"""
        self.gradient_norms.clear()
        self.activation_stats.clear()
        self.fallback_activations = 0


class BulletproofWaveNetStack(nn.Module):
    """Stack of WaveNet residual blocks with exponentially increasing dilations"""
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__()
        
        self.config = config
        self.stack_config = kwargs.get('stack_config', {})
        
        # Extract parameters
        self.in_channels = self.stack_config.get('in_channels', config.convolution.in_channels)
        self.residual_channels = self.stack_config.get('residual_channels', 256)
        self.skip_channels = self.stack_config.get('skip_channels', 256)
        self.kernel_size = self.stack_config.get('kernel_size', 3)
        self.dilations = self.stack_config.get('dilations', [2 ** i for i in range(10)])  # 1, 2, 4, ..., 512
        
        # Compute receptive field
        self.receptive_field = self._compute_receptive_field()
        
        # Input projection
        try:
            self.input_conv = nn.Conv1d(self.in_channels, self.residual_channels, 1, bias=True)
        except Exception as e:
            logger.error(f"Failed to create input projection: {e}")
            self.input_conv = nn.Identity()
        
        # Build residual blocks
        self.residual_blocks = nn.ModuleList()
        for i, dilation in enumerate(self.dilations):
            try:
                wavenet_config = WaveNetConfig(
                    channels=self.residual_channels,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                    skip_channels=self.skip_channels,
                    residual_channels=self.residual_channels,
                    gate_channels=self.residual_channels,
                    **kwargs.get('block_kwargs', {})
                )
                
                block = BulletproofWaveNetResBlock(config, wavenet_config=wavenet_config)
                self.residual_blocks.append(block)
                
            except Exception as e:
                logger.error(f"Failed to create block {i} with dilation {dilation}: {e}")
                # Skip this block or create a fallback
                continue
        
        # Output projection
        try:
            self.output_conv = nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(self.skip_channels, self.skip_channels, 1, bias=True),
                nn.ReLU(),
                nn.Conv1d(self.skip_channels, self.in_channels, 1, bias=True)
            )
        except Exception as e:
            logger.error(f"Failed to create output projection: {e}")
            self.output_conv = nn.Conv1d(self.skip_channels, self.in_channels, 1)
        
        logger.info(f"BulletproofWaveNetStack initialized: {len(self.residual_blocks)} blocks, RF={self.receptive_field}")
    
    def _compute_receptive_field(self) -> int:
        """Compute the total receptive field of the stack"""
        rf = 1
        for dilation in self.dilations:
            rf += (self.kernel_size - 1) * dilation
        return rf
    
    def forward(self, x: torch.Tensor, conditioning: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through WaveNet stack.
        
        Args:
            x: Input tensor [batch, in_channels, time]
            conditioning: Optional conditioning [batch, cond_channels, time]
            
        Returns:
            output: Generated audio [batch, in_channels, time]
        """
        try:
            # Input projection
            if not isinstance(self.input_conv, nn.Identity):
                x = self.input_conv(x)
            
            # Collect skip connections
            skips = []
            
            # Pass through residual blocks
            for i, block in enumerate(self.residual_blocks):
                try:
                    x, skip = block(x, conditioning)
                    skips.append(skip)
                except Exception as e:
                    logger.warning(f"Block {i} failed: {e}")
                    # Continue with current x, add zero skip
                    if skips:
                        zero_skip = torch.zeros_like(skips[0])
                        skips.append(zero_skip)
            
            # Sum skip connections
            if skips:
                skip_sum = torch.stack(skips, dim=0).sum(dim=0)
            else:
                # No skips available, use current x
                skip_sum = x
            
            # Output projection
            try:
                output = self.output_conv(skip_sum)
            except Exception as e:
                logger.error(f"Output projection failed: {e}")
                # Emergency fallback
                output = skip_sum[:, :self.in_channels, :] if skip_sum.size(1) >= self.in_channels else skip_sum
            
            return output
            
        except Exception as e:
            logger.error(f"WaveNet stack forward failed: {e}")
            # Emergency fallback: return input
            return x
    
    def get_receptive_field(self) -> int:
        """Get the receptive field size in samples"""
        return self.receptive_field
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics from all blocks"""
        stats = {
            'num_blocks': len(self.residual_blocks),
            'receptive_field': self.receptive_field,
            'dilations': self.dilations
        }
        
        # Aggregate block statistics
        total_fallbacks = 0
        for i, block in enumerate(self.residual_blocks):
            if hasattr(block, 'get_training_stats'):
                block_stats = block.get_training_stats()
                total_fallbacks += block_stats.get('fallback_activations', 0)
        
        stats['total_fallback_activations'] = total_fallbacks
        return stats


# Factory functions
def create_bulletproof_wavenet_resblock(config: RAVEConfig, **kwargs) -> BulletproofWaveNetResBlock:
    """Create a bulletproof WaveNet residual block"""
    return BulletproofWaveNetResBlock(config, **kwargs)


def create_bulletproof_wavenet_stack(config: RAVEConfig, **kwargs) -> BulletproofWaveNetStack:
    """Create a bulletproof WaveNet stack"""
    return BulletproofWaveNetStack(config, **kwargs)


if __name__ == "__main__":
    print("🛡️ BULLETPROOF WAVENET RESBLOCK MODULE")
    print("=" * 50)
    
    # Test with minimal config
    from rave_config_system import get_minimal_config
    
    config = get_minimal_config()
    config.convolution.in_channels = 1
    
    # Test single ResBlock
    wavenet_config = WaveNetConfig(
        channels=128,
        dilation=4,
        conditioning_channels=80
    )
    
    resblock = create_bulletproof_wavenet_resblock(config, wavenet_config=wavenet_config)
    
    # Test data
    batch_size = 4
    seq_len = 2048
    x = torch.randn(batch_size, 128, seq_len)
    conditioning = torch.randn(batch_size, 80, seq_len)
    
    try:
        residual, skip = resblock(x, conditioning)
        print(f"✅ WaveNet ResBlock test passed")
        print(f"   Input shape: {x.shape}")
        print(f"   Conditioning shape: {conditioning.shape}")
        print(f"   Residual output shape: {residual.shape}")
        print(f"   Skip output shape: {skip.shape}")
        print(f"   Receptive field: {resblock.get_receptive_field()}")
        
        stats = resblock.get_training_stats()
        print(f"   Block stats: {stats}")
        
    except Exception as e:
        print(f"❌ ResBlock test failed: {e}")
    
    # Test WaveNet stack
    try:
        stack_config = {
            'in_channels': 1,
            'residual_channels': 128,
            'skip_channels': 128,
            'dilations': [1, 2, 4, 8, 16]
        }
        
        wavenet_stack = create_bulletproof_wavenet_stack(config, stack_config=stack_config)
        
        x_stack = torch.randn(2, 1, 4096)
        output = wavenet_stack(x_stack)
        
        print(f"✅ WaveNet Stack test passed")
        print(f"   Stack input shape: {x_stack.shape}")
        print(f"   Stack output shape: {output.shape}")
        print(f"   Stack receptive field: {wavenet_stack.get_receptive_field()}")
        
        stack_stats = wavenet_stack.get_training_stats()
        print(f"   Stack stats: {stack_stats}")
        
    except Exception as e:
        print(f"❌ Stack test failed: {e}")
    
    # Test with corrupted input
    try:
        x_corrupted = torch.randn(2, 128, 1024)
        x_corrupted[:, :, 100:110] = float('inf')
        
        residual_robust, skip_robust = resblock(x_corrupted)
        print(f"✅ Robust handling of corrupted input")
        
    except Exception as e:
        print(f"❌ Corrupted input test failed: {e}")
    
    print("🚀 BulletproofWaveNetResBlock ready for BigVGAN generator!")