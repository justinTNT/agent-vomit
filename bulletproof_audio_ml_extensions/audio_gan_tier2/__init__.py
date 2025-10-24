#!/usr/bin/env python3
"""
BULLETPROOF AUDIO GAN TIER 2 ADAPTIVE MODULES
Adaptive conditioning and feature processing modules for BigVGAN with stability features.
"""

from .bulletproof_adain import (
    BulletproofAdaIN,
    BulletproofAdaINResBlock,
    create_bulletproof_adain,
    create_bulletproof_adain_resblock
)
from .bulletproof_film import (
    BulletproofFiLMLayer,
    BulletproofFiLMBlock,
    create_bulletproof_film_layer,
    create_bulletproof_film_block
)
from .bulletproof_mel_spectrogram import (
    BulletproofMelSpectrogram,
    BulletproofLogMelSpectrogram,
    create_bulletproof_mel_spectrogram,
    create_bulletproof_log_mel_spectrogram
)
from .bulletproof_noise_generator import (
    BulletproofNoiseGenerator,
    create_bulletproof_noise_generator
)
from .bulletproof_subpixel_conv import (
    BulletproofSubPixelConv,
    BulletproofMultiScaleSubPixelConv,
    create_bulletproof_subpixel_conv,
    create_bulletproof_multiscale_subpixel_conv
)

__all__ = [
    # Adaptive Instance Normalization
    "BulletproofAdaIN",
    "BulletproofAdaINResBlock", 
    "create_bulletproof_adain",
    "create_bulletproof_adain_resblock",
    
    # Feature-wise Linear Modulation
    "BulletproofFiLMLayer",
    "BulletproofFiLMBlock",
    "create_bulletproof_film_layer",
    "create_bulletproof_film_block",
    
    # Mel Spectrogram
    "BulletproofMelSpectrogram",
    "BulletproofLogMelSpectrogram",
    "create_bulletproof_mel_spectrogram",
    "create_bulletproof_log_mel_spectrogram",
    
    # Noise Generator
    "BulletproofNoiseGenerator",
    "create_bulletproof_noise_generator",
    
    # SubPixel Convolution
    "BulletproofSubPixelConv",
    "BulletproofMultiScaleSubPixelConv",
    "create_bulletproof_subpixel_conv",
    "create_bulletproof_multiscale_subpixel_conv"
]