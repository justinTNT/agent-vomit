#!/usr/bin/env python3
"""
BULLETPROOF AUDIO GAN TIER 1 CORE MODULES
Core BigVGAN discriminator and generator components with comprehensive stability features.
"""

from .bulletproof_gan_loss import BulletproofGANLoss, create_bulletproof_gan_loss
from .bulletproof_multiscale_discriminator import (
    BulletproofMultiScaleDiscriminator, 
    BulletproofScaleDiscriminator,
    create_bulletproof_multiscale_discriminator
)
from .bulletproof_pqmf_filterbank import BulletproofPQMFFilterBank, create_bulletproof_pqmf_filterbank
from .bulletproof_spectral_normalization import (
    BulletproofSpectralNormalization,
    BulletproofSNLinear,
    BulletproofSNConv1d, 
    BulletproofSNConv2d,
    create_bulletproof_sn_linear,
    create_bulletproof_sn_conv1d,
    create_bulletproof_sn_conv2d
)
from .bulletproof_wavenet_resblock import (
    BulletproofWaveNetResBlock,
    BulletproofWaveNetStack,
    create_bulletproof_wavenet_resblock,
    create_bulletproof_wavenet_stack
)

__all__ = [
    # GAN Loss
    "BulletproofGANLoss",
    "create_bulletproof_gan_loss",
    
    # Multi-scale Discriminator
    "BulletproofMultiScaleDiscriminator",
    "BulletproofScaleDiscriminator", 
    "create_bulletproof_multiscale_discriminator",
    
    # PQMF Filter Bank
    "BulletproofPQMFFilterBank",
    "create_bulletproof_pqmf_filterbank",
    
    # Spectral Normalization
    "BulletproofSpectralNormalization",
    "BulletproofSNLinear",
    "BulletproofSNConv1d",
    "BulletproofSNConv2d",
    "create_bulletproof_sn_linear",
    "create_bulletproof_sn_conv1d", 
    "create_bulletproof_sn_conv2d",
    
    # WaveNet ResBlock
    "BulletproofWaveNetResBlock",
    "BulletproofWaveNetStack",
    "create_bulletproof_wavenet_resblock",
    "create_bulletproof_wavenet_stack"
]