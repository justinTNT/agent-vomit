#!/usr/bin/env python3
"""
BULLETPROOF AUDIO ML EXTENSIONS - BigVGAN Tier 3 Processing Modules
Advanced audio generation capabilities with comprehensive error handling
"""

from .bulletproof_attention import BulletproofAttention, AttentionConfig, create_bulletproof_attention
from .bulletproof_group_norm import BulletproofGroupNorm, GroupNormConfig, create_bulletproof_group_norm  
from .bulletproof_modulation import BulletproofModulation, ModulationConfig, create_bulletproof_modulation
from .bulletproof_pitch_shift import BulletproofPitchShift, PitchShiftConfig, create_bulletproof_pitch_shift
from .bulletproof_time_stretch import BulletproofTimeStretch, TimeStretchConfig, create_bulletproof_time_stretch

__all__ = [
    'BulletproofAttention',
    'AttentionConfig',
    'BulletproofGroupNorm',
    'GroupNormConfig', 
    'BulletproofModulation',
    'ModulationConfig',
    'BulletproofPitchShift',
    'PitchShiftConfig',
    'BulletproofTimeStretch',
    'TimeStretchConfig',
    'create_bulletproof_attention',
    'create_bulletproof_group_norm',
    'create_bulletproof_modulation', 
    'create_bulletproof_pitch_shift',
    'create_bulletproof_time_stretch'
]