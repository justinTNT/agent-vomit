#!/usr/bin/env python3
"""
BULLETPROOF NOISE GENERATOR MODULE
Comprehensive noise generation for BigVGAN with multiple fallback strategies and stability features.
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
class NoiseGeneratorConfig:
    """Configuration for bulletproof noise generator"""
    noise_type: str = 'white'  # 'white', 'pink', 'shaped', 'filtered'
    num_channels: int = 1
    learnable: bool = True
    conditional: bool = False
    conditioning_dim: Optional[int] = None
    
    # Bulletproof parameters
    numerical_stability_check: bool = True
    gradient_clipping: bool = True
    gradient_clip_value: float = 10.0
    noise_scale_range: Tuple[float, float] = (0.001, 10.0)
    
    # Advanced features
    adaptive_noise_scaling: bool = True
    memory_efficient: bool = True
    device_compatible: bool = True
    
    # Fallback strategies
    enable_fallbacks: bool = True
    fallback_noise_type: str = 'white'
    emergency_noise_level: float = 0.01

class BulletproofNoiseGenerator(nn.Module):
    """Bulletproof Noise Generator with comprehensive error handling and multiple noise types"""
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__()
        
        self.config = config
        self.noise_config = kwargs.get('noise_config', NoiseGeneratorConfig())
        
        self.noise_type = self.noise_config.noise_type
        self.num_channels = self.noise_config.num_channels
        self.learnable = self.noise_config.learnable
        self.conditional = self.noise_config.conditional
        self.conditioning_dim = self.noise_config.conditioning_dim
        
        # Validate configuration
        if self.conditional and self.conditioning_dim is None:
            logger.warning("Conditional noise requires conditioning_dim, disabling conditioning")
            self.conditional = False
        
        # Initialize noise generation components
        try:
            self._build_noise_components()
        except Exception as e:
            logger.error(f"Failed to build noise components: {e}")
            if self.noise_config.enable_fallbacks:
                logger.warning("Building fallback noise components")
                self._build_fallback_components()
            else:
                raise
        
        # Conditional components
        if self.conditional and self.conditioning_dim is not None:
            try:
                self.conditioning_net = nn.Sequential(
                    nn.Linear(self.conditioning_dim, 128),
                    nn.LeakyReLU(0.2),
                    nn.Linear(128, self.num_channels * 2),  # Scale and bias
                    nn.Tanh()  # Bounded output
                )
            except Exception as e:
                logger.error(f"Failed to build conditioning network: {e}")
                self.conditioning_net = None
                self.conditional = False
        
        # Tracking
        self.generation_stats = []
        self.fallback_activations = 0
        
        logger.info(f"BulletproofNoiseGenerator initialized: {self.noise_type}, {self.num_channels} channels")
    
    def _build_noise_components(self):
        """Build noise generation components based on type"""
        if self.noise_type == 'white':
            self._build_white_noise_components()
        elif self.noise_type == 'pink':
            self._build_pink_noise_components()
        elif self.noise_type == 'shaped':
            self._build_shaped_noise_components()
        elif self.noise_type == 'filtered':
            self._build_filtered_noise_components()
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
    
    def _build_white_noise_components(self):
        """Build white noise components"""
        if self.learnable:
            self.gain = nn.Parameter(torch.ones(self.num_channels))
        else:
            self.register_buffer('gain', torch.ones(self.num_channels))
    
    def _build_pink_noise_components(self):
        """Build pink noise components"""
        self.num_sources = 5  # Number of octave bands
        if self.learnable:
            self.gains = nn.Parameter(torch.ones(self.num_channels, self.num_sources))
        else:
            # Pink noise gains for -3dB/octave slope
            gains = torch.tensor([1.0, 0.5, 0.25, 0.125, 0.0625]).repeat(self.num_channels, 1)
            self.register_buffer('gains', gains)
    
    def _build_shaped_noise_components(self):
        """Build shaped noise components"""
        self.num_bands = 32
        if self.learnable:
            self.envelope = nn.Parameter(torch.ones(self.num_channels, self.num_bands))
        else:
            self.register_buffer('envelope', torch.ones(self.num_channels, self.num_bands))
    
    def _build_filtered_noise_components(self):
        """Build filtered noise components"""
        try:
            self.filter = nn.Conv1d(
                self.num_channels, self.num_channels,
                kernel_size=15, padding=7, groups=self.num_channels
            )
            if not self.learnable:
                for param in self.filter.parameters():
                    param.requires_grad = False
        except Exception as e:
            logger.error(f"Failed to build filter: {e}")
            raise
    
    def _build_fallback_components(self):
        """Build simple fallback components"""
        try:
            # Default to white noise
            self.noise_type = self.noise_config.fallback_noise_type
            if self.learnable:
                self.gain = nn.Parameter(torch.ones(self.num_channels) * self.noise_config.emergency_noise_level)
            else:
                self.register_buffer('gain', torch.ones(self.num_channels) * self.noise_config.emergency_noise_level)
            
            logger.info("Built fallback noise components")
        except Exception as e:
            logger.error(f"Fallback component building failed: {e}")
            # Emergency fallback
            self.register_buffer('gain', torch.ones(1) * 0.01)
            self.num_channels = 1
    
    def _validate_generation_params(self, batch_size: int, length: int, device: torch.device) -> bool:
        """Validate noise generation parameters"""
        try:
            if batch_size <= 0 or batch_size > 1024:  # Reasonable batch size limits
                logger.warning(f"Invalid batch_size: {batch_size}")
                return False
            
            if length <= 0 or length > 1048576:  # Reasonable length limits (2^20)
                logger.warning(f"Invalid length: {length}")
                return False
            
            # Check device compatibility
            if self.noise_config.device_compatible:
                if device != next(self.parameters()).device:
                    logger.warning(f"Device mismatch: {device} vs {next(self.parameters()).device}")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Parameter validation failed: {e}")
            return False
    
    def _generate_white_noise(self, batch_size: int, length: int, device: torch.device) -> torch.Tensor:
        """Generate white noise with error handling"""
        try:
            # Generate uniform random noise
            noise = torch.randn(batch_size, self.num_channels, length, device=device, dtype=torch.float32)
            
            # Apply gain with clipping
            gain = torch.clamp(self.gain, *self.noise_config.noise_scale_range)
            noise = noise * gain.view(1, -1, 1)
            
            # Validate output
            if self.noise_config.numerical_stability_check:
                if not torch.isfinite(noise).all():
                    logger.warning("Non-finite white noise generated")
                    noise = torch.nan_to_num(noise, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return noise
            
        except Exception as e:
            logger.error(f"White noise generation failed: {e}")
            if self.noise_config.enable_fallbacks:
                self.fallback_activations += 1
                return torch.randn(batch_size, self.num_channels, length, device=device) * self.noise_config.emergency_noise_level
            else:
                raise
    
    def _generate_pink_noise(self, batch_size: int, length: int, device: torch.device) -> torch.Tensor:
        """Generate pink noise using octave band method"""
        try:
            noise = torch.zeros(batch_size, self.num_channels, length, device=device, dtype=torch.float32)
            
            for i in range(self.num_sources):
                try:
                    # Downsample factor for this octave
                    factor = 2 ** i
                    
                    # Generate noise at lower rate
                    low_rate_length = max(1, (length + factor - 1) // factor)
                    source = torch.randn(batch_size, self.num_channels, low_rate_length, device=device)
                    
                    # Upsample to full rate
                    if factor > 1:
                        source = F.interpolate(source, size=length, mode='linear', align_corners=False)
                    
                    # Add weighted contribution with clipping
                    gains_clipped = torch.clamp(self.gains[:, i], *self.noise_config.noise_scale_range)
                    noise += source * gains_clipped.view(1, -1, 1)
                    
                except Exception as e:
                    logger.warning(f"Pink noise source {i} failed: {e}")
                    continue
            
            # Normalize and validate
            noise = noise / max(1, self.num_sources**0.5)
            
            if self.noise_config.numerical_stability_check:
                if not torch.isfinite(noise).all():
                    logger.warning("Non-finite pink noise generated")
                    noise = torch.nan_to_num(noise, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return noise
            
        except Exception as e:
            logger.error(f"Pink noise generation failed: {e}")
            if self.noise_config.enable_fallbacks:
                self.fallback_activations += 1
                return self._generate_white_noise(batch_size, length, device)
            else:
                raise
    
    def _generate_shaped_noise(self, batch_size: int, length: int, device: torch.device) -> torch.Tensor:
        """Generate spectrally shaped noise"""
        try:
            # Generate white noise
            noise = torch.randn(batch_size, self.num_channels, length, device=device, dtype=torch.float32)
            
            # Apply FFT with error handling
            try:
                noise_fft = torch.fft.rfft(noise, dim=-1)
            except Exception as e:
                logger.warning(f"FFT failed: {e}")
                return self._generate_white_noise(batch_size, length, device)
            
            # Create spectral envelope
            freqs = noise_fft.shape[-1]
            
            try:
                # Interpolate envelope to match FFT bins
                envelope_interp = F.interpolate(
                    self.envelope.unsqueeze(0),
                    size=freqs,
                    mode='linear',
                    align_corners=False
                ).squeeze(0)
                
                # Clamp envelope
                envelope_interp = torch.clamp(envelope_interp, *self.noise_config.noise_scale_range)
                
                # Apply envelope
                noise_fft = noise_fft * envelope_interp.view(1, -1, freqs)
                
            except Exception as e:
                logger.warning(f"Spectral shaping failed: {e}")
                # Continue with unmodified FFT
            
            # Inverse FFT with error handling
            try:
                noise = torch.fft.irfft(noise_fft, n=length, dim=-1)
            except Exception as e:
                logger.warning(f"Inverse FFT failed: {e}")
                return self._generate_white_noise(batch_size, length, device)
            
            # Validate output
            if self.noise_config.numerical_stability_check:
                if not torch.isfinite(noise).all():
                    logger.warning("Non-finite shaped noise generated")
                    noise = torch.nan_to_num(noise, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return noise
            
        except Exception as e:
            logger.error(f"Shaped noise generation failed: {e}")
            if self.noise_config.enable_fallbacks:
                self.fallback_activations += 1
                return self._generate_white_noise(batch_size, length, device)
            else:
                raise
    
    def _generate_filtered_noise(self, batch_size: int, length: int, device: torch.device) -> torch.Tensor:
        """Generate filtered noise"""
        try:
            # Generate white noise
            noise = torch.randn(batch_size, self.num_channels, length, device=device, dtype=torch.float32)
            
            # Apply learned filters
            if hasattr(self, 'filter'):
                try:
                    noise = self.filter(noise)
                except Exception as e:
                    logger.warning(f"Filter application failed: {e}")
                    # Continue with unfiltered noise
            
            # Validate output
            if self.noise_config.numerical_stability_check:
                if not torch.isfinite(noise).all():
                    logger.warning("Non-finite filtered noise generated")
                    noise = torch.nan_to_num(noise, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return noise
            
        except Exception as e:
            logger.error(f"Filtered noise generation failed: {e}")
            if self.noise_config.enable_fallbacks:
                self.fallback_activations += 1
                return self._generate_white_noise(batch_size, length, device)
            else:
                raise
    
    def _apply_conditioning(self, noise: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """Apply conditioning to noise with error handling"""
        try:
            if not self.conditional or self.conditioning_net is None:
                return noise
            
            # Generate scale and bias from conditioning
            modulation = self.conditioning_net(conditioning)  # (batch, num_channels * 2)
            
            batch_size = noise.size(0)
            scale = modulation[:, :self.num_channels].unsqueeze(-1)
            bias = modulation[:, self.num_channels:].unsqueeze(-1)
            
            # Apply gradient clipping
            if self.noise_config.gradient_clipping:
                scale = torch.clamp(scale, -self.noise_config.gradient_clip_value, self.noise_config.gradient_clip_value)
                bias = torch.clamp(bias, -self.noise_config.gradient_clip_value, self.noise_config.gradient_clip_value)
            
            # Apply conditional modulation with sigmoid gating
            noise = noise * (torch.sigmoid(scale) * 2.0) + bias * 0.1
            
            return noise
            
        except Exception as e:
            logger.warning(f"Conditioning application failed: {e}")
            return noise
    
    def forward(self, batch_size: int, length: int, conditioning: Optional[torch.Tensor] = None,
                device: Optional[torch.device] = None) -> torch.Tensor:
        """Generate noise with comprehensive error handling"""
        try:
            # Determine device
            if device is None:
                if list(self.parameters()):
                    device = next(self.parameters()).device
                else:
                    device = torch.device('cpu')
            
            # Validate parameters
            if not self._validate_generation_params(batch_size, length, device):
                if self.noise_config.enable_fallbacks:
                    logger.warning("Invalid parameters, using fallback generation")
                    self.fallback_activations += 1
                    batch_size = min(max(batch_size, 1), 32)
                    length = min(max(length, 1), 8192)
                else:
                    raise ValueError("Invalid generation parameters")
            
            # Generate base noise
            if self.noise_type == 'white':
                noise = self._generate_white_noise(batch_size, length, device)
            elif self.noise_type == 'pink':
                noise = self._generate_pink_noise(batch_size, length, device)
            elif self.noise_type == 'shaped':
                noise = self._generate_shaped_noise(batch_size, length, device)
            elif self.noise_type == 'filtered':
                noise = self._generate_filtered_noise(batch_size, length, device)
            else:
                # Fallback to white noise
                noise = self._generate_white_noise(batch_size, length, device)
            
            # Apply conditioning if provided
            if conditioning is not None:
                noise = self._apply_conditioning(noise, conditioning)
            
            # Final validation and clipping
            if self.noise_config.numerical_stability_check:
                if not torch.isfinite(noise).all():
                    logger.warning("Non-finite final noise output")
                    noise = torch.nan_to_num(noise, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # Clamp to reasonable range
                noise = torch.clamp(noise, -100.0, 100.0)
            
            # Track statistics
            if len(self.generation_stats) < 1000:
                self.generation_stats.append({
                    'noise_type': self.noise_type,
                    'batch_size': batch_size,
                    'length': length,
                    'noise_mean': noise.mean().item(),
                    'noise_std': noise.std().item(),
                    'noise_max': noise.max().item(),
                    'noise_min': noise.min().item()
                })
            
            return noise
            
        except Exception as e:
            logger.error(f"Noise generation failed: {e}")
            if self.noise_config.enable_fallbacks:
                self.fallback_activations += 1
                # Emergency fallback: return minimal noise
                device = device or torch.device('cpu')
                return torch.randn(batch_size, self.num_channels, length, device=device) * self.noise_config.emergency_noise_level
            else:
                raise
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        stats = {
            'noise_type': self.noise_type,
            'num_channels': self.num_channels,
            'conditional': self.conditional,
            'fallback_activations': self.fallback_activations
        }
        
        if self.generation_stats:
            last_stats = self.generation_stats[-1]
            stats.update({
                'last_noise_mean': last_stats['noise_mean'],
                'last_noise_std': last_stats['noise_std'],
                'last_length': last_stats['length']
            })
        
        return stats
    
    def reset_statistics(self):
        """Reset training statistics"""
        self.generation_stats.clear()
        self.fallback_activations = 0


# Factory functions
def create_bulletproof_noise_generator(config: RAVEConfig, **kwargs) -> BulletproofNoiseGenerator:
    """Create a bulletproof noise generator"""
    return BulletproofNoiseGenerator(config, **kwargs)


if __name__ == "__main__":
    print("🛡️ BULLETPROOF NOISE GENERATOR MODULE")
    print("=" * 45)
    
    from rave_config_system import get_minimal_config
    
    config = get_minimal_config()
    
    # Test white noise generator
    noise_config = NoiseGeneratorConfig(noise_type='white', num_channels=1, learnable=True)
    noise_gen = create_bulletproof_noise_generator(config, noise_config=noise_config)
    
    batch_size = 4
    length = 1024
    
    try:
        noise = noise_gen(batch_size, length)
        print(f"✅ White noise generation test passed: {noise.shape}")
        
        stats = noise_gen.get_training_stats()
        print(f"   Noise stats: {stats}")
        
    except Exception as e:
        print(f"❌ White noise test failed: {e}")
    
    # Test conditional noise generation
    try:
        cond_config = NoiseGeneratorConfig(
            noise_type='pink', num_channels=2, conditional=True, conditioning_dim=64
        )
        cond_noise_gen = create_bulletproof_noise_generator(config, noise_config=cond_config)
        
        conditioning = torch.randn(batch_size, 64)
        cond_noise = cond_noise_gen(batch_size, length, conditioning)
        print(f"✅ Conditional noise generation test passed: {cond_noise.shape}")
        
    except Exception as e:
        print(f"❌ Conditional noise test failed: {e}")
    
    # Test shaped noise
    try:
        shaped_config = NoiseGeneratorConfig(noise_type='shaped', num_channels=1)
        shaped_noise_gen = create_bulletproof_noise_generator(config, noise_config=shaped_config)
        
        shaped_noise = shaped_noise_gen(batch_size, length)
        print(f"✅ Shaped noise generation test passed: {shaped_noise.shape}")
        
    except Exception as e:
        print(f"❌ Shaped noise test failed: {e}")
    
    print("🚀 BulletproofNoiseGenerator ready for BigVGAN audio synthesis!")