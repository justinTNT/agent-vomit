#!/usr/bin/env python3
"""
BULLETPROOF BIGVGAN TIER 4 ANALYSIS MODULES
Advanced audio analysis and quality assessment for neural audio generation.
"""

from .bulletproof_chroma_encoder import BulletproofChromaEncoder, create_bulletproof_chroma_encoder
from .bulletproof_onset_detector import BulletproofOnsetDetector, create_bulletproof_onset_detector
from .bulletproof_phase_reconstruction import BulletproofPhaseReconstruction, create_bulletproof_phase_reconstruction
from .bulletproof_wave_gan_discriminator import BulletproofWaveGANDiscriminator, create_bulletproof_wave_gan_discriminator

__all__ = [
    'BulletproofChromaEncoder',
    'BulletproofOnsetDetector', 
    'BulletproofPhaseReconstruction',
    'BulletproofWaveGANDiscriminator',
    'create_bulletproof_chroma_encoder',
    'create_bulletproof_onset_detector',
    'create_bulletproof_phase_reconstruction',
    'create_bulletproof_wave_gan_discriminator'
]