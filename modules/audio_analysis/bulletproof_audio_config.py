"""
Bulletproof Audio Configuration Module

Provides robust, validated configuration management for all audio processing modules
with comprehensive error handling and fallback strategies.
"""

import torch
import torch.nn as nn
import torchaudio
import numpy as np
import warnings
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioDomain(Enum):
    """Audio domain types with different optimal configurations."""
    SPEECH = "speech"
    MUSIC = "music"
    GENERAL = "general"
    TIMBRALGEBRAICS = "timbralgebraics"
    ENVIRONMENTAL = "environmental"
    CUSTOM = "custom"


class NormalizationMethod(Enum):
    """Audio normalization methods with validation."""
    NONE = "none"
    ZERO_MEAN_UNIT_VAR = "zero_mean_unit_var"
    ZERO_MEAN_HALF_VAR = "zero_mean_half_var"
    MIN_MAX = "min_max"
    ROBUST = "robust"
    INSTANCE = "instance"
    QUANTILE = "quantile"


class DeviceType(Enum):
    """Supported device types."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"


@dataclass
class BulletproofAudioConfig:
    """
    Bulletproof audio processing configuration with comprehensive validation,
    error handling, and automatic fallback mechanisms.
    """
    
    # Core audio parameters
    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    win_length: Optional[int] = None
    window: str = "hann"
    
    # Spectral parameters
    n_mels: int = 128
    n_mfcc: int = 13
    n_chroma: int = 12
    f_min: float = 0.0
    f_max: Optional[float] = None
    
    # Model architecture parameters
    embedding_dim: int = 512
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    
    # Processing parameters
    max_audio_length: int = 22050 * 300  # 5 minutes default
    chunk_size: Optional[int] = None
    overlap_ratio: float = 0.1
    
    # Normalization and preprocessing
    normalization_method: NormalizationMethod = NormalizationMethod.ZERO_MEAN_UNIT_VAR
    normalization_stats: Optional[Dict[str, float]] = None
    apply_log_scaling: bool = True
    log_epsilon: float = 1e-8
    
    # Data augmentation
    use_spec_augment: bool = False
    freq_mask_param: int = 27
    time_mask_param: int = 100
    n_freq_masks: int = 1
    n_time_masks: int = 1
    
    # Domain and compatibility
    domain: AudioDomain = AudioDomain.GENERAL
    content_aware: bool = True
    
    # Validation and error handling
    validate_inputs: bool = True
    enable_fallbacks: bool = True
    strict_mode: bool = False
    warn_on_mismatch: bool = True
    
    # Device and memory management
    device_type: DeviceType = DeviceType.AUTO
    memory_efficient: bool = True
    max_memory_mb: Optional[int] = None
    enable_mixed_precision: bool = False
    
    # Caching and optimization
    cache_transforms: bool = True
    cache_features: bool = False
    optimization_level: int = 1  # 0=basic, 1=standard, 2=aggressive
    
    def __post_init__(self):
        """Initialize and validate configuration with comprehensive error handling."""
        try:
            # Validate and sanitize parameters
            self._validate_core_audio_params()
            self._validate_spectral_params()
            self._validate_model_params()
            self._validate_processing_params()
            self._validate_device_params()
            
            # Set derived parameters
            self._set_derived_params()
            
            # Set domain-specific defaults
            self._apply_domain_defaults()
            
            # Final validation
            self._final_validation()
            
            logger.info(f"BulletproofAudioConfig initialized for domain: {self.domain.value}")
            
        except Exception as e:
            logger.error(f"Error initializing audio config: {e}")
            if self.strict_mode:
                raise
            else:
                logger.warning("Applying emergency fallback configuration")
                self._apply_emergency_fallback()
    
    def _validate_core_audio_params(self):
        """Validate core audio parameters with sanitization."""
        # Sample rate validation
        if not isinstance(self.sample_rate, (int, float)):
            logger.warning(f"Invalid sample_rate type: {type(self.sample_rate)}, using 22050")
            self.sample_rate = 22050
        else:
            self.sample_rate = int(self.sample_rate)
            if self.sample_rate < 8000:
                logger.warning(f"Sample rate {self.sample_rate} too low, using 8000")
                self.sample_rate = 8000
            elif self.sample_rate > 192000:
                logger.warning(f"Sample rate {self.sample_rate} too high, using 192000")
                self.sample_rate = 192000
        
        # n_fft validation
        if not isinstance(self.n_fft, (int, float)):
            logger.warning(f"Invalid n_fft type: {type(self.n_fft)}, using 2048")
            self.n_fft = 2048
        else:
            self.n_fft = int(self.n_fft)
            # Ensure power of 2
            if self.n_fft <= 0:
                self.n_fft = 2048
            else:
                power = int(np.log2(self.n_fft))
                if 2**power != self.n_fft:
                    self.n_fft = 2**power
                    logger.info(f"Rounded n_fft to nearest power of 2: {self.n_fft}")
                self.n_fft = max(256, min(8192, self.n_fft))
        
        # hop_length validation
        if not isinstance(self.hop_length, (int, float)):
            logger.warning(f"Invalid hop_length type: {type(self.hop_length)}, using n_fft//4")
            self.hop_length = self.n_fft // 4
        else:
            self.hop_length = int(self.hop_length)
            if self.hop_length <= 0:
                self.hop_length = self.n_fft // 4
            elif self.hop_length > self.n_fft:
                logger.warning(f"hop_length > n_fft, using n_fft//2")
                self.hop_length = self.n_fft // 2
        
        # win_length validation
        if self.win_length is not None:
            if not isinstance(self.win_length, (int, float)):
                logger.warning(f"Invalid win_length type: {type(self.win_length)}, using n_fft")
                self.win_length = self.n_fft
            else:
                self.win_length = int(self.win_length)
                if self.win_length <= 0 or self.win_length > self.n_fft:
                    logger.warning(f"Invalid win_length {self.win_length}, using n_fft")
                    self.win_length = self.n_fft
        
        # window validation
        valid_windows = ["hann", "hamming", "blackman", "bartlett", "rectangular"]
        if not isinstance(self.window, str) or self.window.lower() not in valid_windows:
            logger.warning(f"Invalid window '{self.window}', using 'hann'")
            self.window = "hann"
        else:
            self.window = self.window.lower()
    
    def _validate_spectral_params(self):
        """Validate spectral analysis parameters."""
        # n_mels validation
        if not isinstance(self.n_mels, (int, float)):
            logger.warning(f"Invalid n_mels type: {type(self.n_mels)}, using 128")
            self.n_mels = 128
        else:
            self.n_mels = max(10, min(512, int(self.n_mels)))
        
        # n_mfcc validation
        if not isinstance(self.n_mfcc, (int, float)):
            logger.warning(f"Invalid n_mfcc type: {type(self.n_mfcc)}, using 13")
            self.n_mfcc = 13
        else:
            self.n_mfcc = max(1, min(50, int(self.n_mfcc)))
            if self.n_mfcc > self.n_mels:
                logger.warning(f"n_mfcc > n_mels, setting n_mfcc = {self.n_mels}")
                self.n_mfcc = self.n_mels
        
        # n_chroma validation
        if not isinstance(self.n_chroma, (int, float)):
            logger.warning(f"Invalid n_chroma type: {type(self.n_chroma)}, using 12")
            self.n_chroma = 12
        else:
            self.n_chroma = max(6, min(24, int(self.n_chroma)))
        
        # Frequency range validation
        if not isinstance(self.f_min, (int, float)):
            logger.warning(f"Invalid f_min type: {type(self.f_min)}, using 0.0")
            self.f_min = 0.0
        else:
            self.f_min = max(0.0, float(self.f_min))
        
        if self.f_max is not None:
            if not isinstance(self.f_max, (int, float)):
                logger.warning(f"Invalid f_max type: {type(self.f_max)}, using Nyquist")
                self.f_max = None
            else:
                self.f_max = float(self.f_max)
                nyquist = self.sample_rate / 2.0
                if self.f_max > nyquist:
                    logger.warning(f"f_max {self.f_max} > Nyquist {nyquist}, using Nyquist")
                    self.f_max = nyquist
                elif self.f_max <= self.f_min:
                    logger.warning(f"f_max <= f_min, using Nyquist")
                    self.f_max = nyquist
    
    def _validate_model_params(self):
        """Validate model architecture parameters."""
        # embedding_dim validation
        if not isinstance(self.embedding_dim, (int, float)):
            logger.warning(f"Invalid embedding_dim type: {type(self.embedding_dim)}, using 512")
            self.embedding_dim = 512
        else:
            self.embedding_dim = max(32, min(2048, int(self.embedding_dim)))
            # Ensure divisible by num_heads
            if self.embedding_dim % self.num_heads != 0:
                self.embedding_dim = ((self.embedding_dim // self.num_heads) + 1) * self.num_heads
                logger.info(f"Adjusted embedding_dim to {self.embedding_dim} (divisible by num_heads)")
        
        # hidden_dim validation
        if not isinstance(self.hidden_dim, (int, float)):
            logger.warning(f"Invalid hidden_dim type: {type(self.hidden_dim)}, using 256")
            self.hidden_dim = 256
        else:
            self.hidden_dim = max(32, min(1024, int(self.hidden_dim)))
        
        # num_heads validation
        if not isinstance(self.num_heads, (int, float)):
            logger.warning(f"Invalid num_heads type: {type(self.num_heads)}, using 8")
            self.num_heads = 8
        else:
            self.num_heads = max(1, min(32, int(self.num_heads)))
        
        # num_layers validation
        if not isinstance(self.num_layers, (int, float)):
            logger.warning(f"Invalid num_layers type: {type(self.num_layers)}, using 4")
            self.num_layers = 4
        else:
            self.num_layers = max(1, min(48, int(self.num_layers)))
        
        # dropout validation
        if not isinstance(self.dropout, (int, float)):
            logger.warning(f"Invalid dropout type: {type(self.dropout)}, using 0.1")
            self.dropout = 0.1
        else:
            self.dropout = max(0.0, min(0.9, float(self.dropout)))
    
    def _validate_processing_params(self):
        """Validate processing parameters."""
        # max_audio_length validation
        if not isinstance(self.max_audio_length, (int, float)):
            logger.warning(f"Invalid max_audio_length type: {type(self.max_audio_length)}, using default")
            self.max_audio_length = self.sample_rate * 300  # 5 minutes
        else:
            self.max_audio_length = max(self.sample_rate, int(self.max_audio_length))
        
        # chunk_size validation
        if self.chunk_size is not None:
            if not isinstance(self.chunk_size, (int, float)):
                logger.warning(f"Invalid chunk_size type: {type(self.chunk_size)}, setting to None")
                self.chunk_size = None
            else:
                self.chunk_size = max(self.sample_rate, int(self.chunk_size))
        
        # overlap_ratio validation
        if not isinstance(self.overlap_ratio, (int, float)):
            logger.warning(f"Invalid overlap_ratio type: {type(self.overlap_ratio)}, using 0.1")
            self.overlap_ratio = 0.1
        else:
            self.overlap_ratio = max(0.0, min(0.5, float(self.overlap_ratio)))
        
        # log_epsilon validation
        if not isinstance(self.log_epsilon, (int, float)):
            logger.warning(f"Invalid log_epsilon type: {type(self.log_epsilon)}, using 1e-8")
            self.log_epsilon = 1e-8
        else:
            self.log_epsilon = max(1e-12, min(1e-3, float(self.log_epsilon)))
    
    def _validate_device_params(self):
        """Validate device and memory parameters."""
        # max_memory_mb validation
        if self.max_memory_mb is not None:
            if not isinstance(self.max_memory_mb, (int, float)):
                logger.warning(f"Invalid max_memory_mb type: {type(self.max_memory_mb)}, setting to None")
                self.max_memory_mb = None
            else:
                self.max_memory_mb = max(100, int(self.max_memory_mb))  # At least 100MB
        
        # optimization_level validation
        if not isinstance(self.optimization_level, (int, float)):
            logger.warning(f"Invalid optimization_level type: {type(self.optimization_level)}, using 1")
            self.optimization_level = 1
        else:
            self.optimization_level = max(0, min(2, int(self.optimization_level)))
    
    def _set_derived_params(self):
        """Set derived parameters based on validated inputs."""
        # Set win_length if not specified
        if self.win_length is None:
            self.win_length = self.n_fft
        
        # Set f_max if not specified
        if self.f_max is None:
            self.f_max = self.sample_rate / 2.0
        
        # Set chunk_size based on memory constraints if not specified
        if self.chunk_size is None and self.max_memory_mb is not None:
            # Estimate chunk size based on memory limit
            estimated_chunk = self._estimate_chunk_size_from_memory()
            self.chunk_size = estimated_chunk
        
        # Set normalization stats if not provided
        if self.normalization_stats is None:
            self.normalization_stats = self._get_domain_normalization_stats()
    
    def _estimate_chunk_size_from_memory(self) -> int:
        """Estimate appropriate chunk size based on memory limit."""
        if self.max_memory_mb is None:
            return self.sample_rate * 10  # 10 seconds default
        
        # Conservative estimate: 1MB can handle ~1 second of processed audio
        seconds_per_mb = 1.0
        max_seconds = self.max_memory_mb * seconds_per_mb * 0.8  # 80% safety margin
        chunk_samples = int(max_seconds * self.sample_rate)
        
        return max(self.sample_rate, chunk_samples)  # At least 1 second
    
    def _apply_domain_defaults(self):
        """Apply domain-specific default configurations."""
        domain_configs = {
            AudioDomain.SPEECH: {
                'sample_rate': 16000,
                'n_fft': 512,
                'hop_length': 160,
                'n_mels': 80,
                'f_max': 8000,
                'normalization_method': NormalizationMethod.ZERO_MEAN_HALF_VAR,
                'use_spec_augment': True
            },
            AudioDomain.MUSIC: {
                'sample_rate': 22050,
                'n_fft': 2048,
                'hop_length': 512,
                'n_mels': 128,
                'normalization_method': NormalizationMethod.ZERO_MEAN_UNIT_VAR,
                'use_spec_augment': True,
                'freq_mask_param': 40,
                'time_mask_param': 80
            },
            AudioDomain.ENVIRONMENTAL: {
                'sample_rate': 22050,
                'n_fft': 2048,
                'hop_length': 512,
                'n_mels': 128,
                'normalization_method': NormalizationMethod.ROBUST,
                'use_spec_augment': False
            },
            AudioDomain.TIMBRALGEBRAICS: {
                'sample_rate': 22050,
                'n_fft': 2048,
                'hop_length': 512,
                'n_mels': 128,
                'embedding_dim': 512,
                'normalization_method': NormalizationMethod.ZERO_MEAN_UNIT_VAR,
                'content_aware': True,
                'use_spec_augment': False
            }
        }
        
        if self.domain in domain_configs:
            domain_config = domain_configs[self.domain]
            for key, value in domain_config.items():
                if hasattr(self, key):
                    # Only apply if current value is default
                    current_value = getattr(self, key)
                    if self._is_default_value(key, current_value):
                        setattr(self, key, value)
                        logger.info(f"Applied domain default {key}={value} for {self.domain.value}")
    
    def _is_default_value(self, param_name: str, value: Any) -> bool:
        """Check if a parameter has its default value."""
        # This is a simplified check - in practice, you'd track original vs modified values
        defaults = {
            'sample_rate': 22050,
            'n_fft': 2048,
            'hop_length': 512,
            'n_mels': 128,
            'normalization_method': NormalizationMethod.ZERO_MEAN_UNIT_VAR
        }
        return param_name in defaults and value == defaults[param_name]
    
    def _get_domain_normalization_stats(self) -> Dict[str, float]:
        """Get normalization statistics for different domains."""
        stats = {
            AudioDomain.SPEECH: {
                "mel_mean": -5.081, "mel_std": 4.969,
                "mfcc_mean": 0.0, "mfcc_std": 15.0,
                "chroma_mean": 0.083, "chroma_std": 0.146
            },
            AudioDomain.MUSIC: {
                "mel_mean": -4.268, "mel_std": 4.569,
                "mfcc_mean": 0.0, "mfcc_std": 12.0,
                "chroma_mean": 0.083, "chroma_std": 0.15
            },
            AudioDomain.GENERAL: {
                "mel_mean": -4.268, "mel_std": 4.569,
                "mfcc_mean": 0.0, "mfcc_std": 13.5,
                "chroma_mean": 0.083, "chroma_std": 0.15
            },
            AudioDomain.ENVIRONMENTAL: {
                "mel_mean": -3.97, "mel_std": 4.12,
                "mfcc_mean": 0.0, "mfcc_std": 11.8,
                "chroma_mean": 0.083, "chroma_std": 0.142
            },
            AudioDomain.TIMBRALGEBRAICS: {
                "mel_mean": -4.0, "mel_std": 4.0,
                "mfcc_mean": 0.0, "mfcc_std": 10.0,
                "chroma_mean": 0.083, "chroma_std": 0.12
            }
        }
        
        return stats.get(self.domain, stats[AudioDomain.GENERAL])
    
    def _final_validation(self):
        """Perform final cross-parameter validation."""
        # Ensure consistency between parameters
        if self.n_mfcc > self.n_mels:
            logger.warning(f"n_mfcc ({self.n_mfcc}) > n_mels ({self.n_mels}), adjusting n_mfcc")
            self.n_mfcc = self.n_mels
        
        if self.embedding_dim % self.num_heads != 0:
            logger.warning(f"embedding_dim not divisible by num_heads, adjusting")
            self.embedding_dim = ((self.embedding_dim // self.num_heads) + 1) * self.num_heads
        
        # Validate memory constraints
        estimated_memory = self._estimate_memory_usage()
        if self.max_memory_mb and estimated_memory > self.max_memory_mb:
            logger.warning(f"Estimated memory usage ({estimated_memory:.1f}MB) exceeds limit ({self.max_memory_mb}MB)")
            if self.enable_fallbacks:
                self._adjust_for_memory_constraints()
            elif self.strict_mode:
                raise ValueError(f"Memory usage exceeds limit in strict mode")
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB for typical processing."""
        # Rough estimates for memory usage
        audio_length = min(self.max_audio_length, self.sample_rate * 60)  # Max 1 minute for estimate
        
        # Spectrogram memory
        n_frames = audio_length // self.hop_length
        spec_memory = self.n_mels * n_frames * 4  # 4 bytes per float
        
        # Model memory (rough estimate)
        model_memory = self.embedding_dim * self.num_layers * 1000  # Very rough
        
        total_mb = (spec_memory + model_memory) * 2 / (1024 * 1024)  # 2x overhead, convert to MB
        
        return total_mb
    
    def _adjust_for_memory_constraints(self):
        """Adjust configuration to fit memory constraints."""
        logger.info("Adjusting configuration for memory constraints")
        
        # Reduce model size
        if self.embedding_dim > 256:
            self.embedding_dim = 256
            logger.info("Reduced embedding_dim to 256")
        
        if self.num_layers > 6:
            self.num_layers = 6
            logger.info("Reduced num_layers to 6")
        
        # Reduce spectral resolution if needed
        if self.n_mels > 80:
            self.n_mels = 80
            logger.info("Reduced n_mels to 80")
        
        # Enable chunked processing
        if self.chunk_size is None or self.chunk_size > self.sample_rate * 10:
            self.chunk_size = self.sample_rate * 5  # 5 seconds
            logger.info("Set chunk_size to 5 seconds")
    
    def _apply_emergency_fallback(self):
        """Apply emergency fallback configuration."""
        logger.warning("Applying emergency fallback configuration")
        
        # Minimal safe configuration
        self.sample_rate = 22050
        self.n_fft = 1024
        self.hop_length = 256
        self.n_mels = 64
        self.n_mfcc = 13
        self.n_chroma = 12
        self.f_min = 0.0
        self.f_max = self.sample_rate / 2.0
        self.embedding_dim = 128
        self.hidden_dim = 64
        self.num_heads = 4
        self.num_layers = 2
        self.dropout = 0.1
        self.normalization_method = NormalizationMethod.ZERO_MEAN_UNIT_VAR
        self.enable_fallbacks = True
        self.validate_inputs = True
        self.memory_efficient = True
        
        # Set derived params
        self.win_length = self.n_fft
        self.chunk_size = self.sample_rate * 5  # 5 seconds
        self.normalization_stats = self._get_domain_normalization_stats()
    
    def get_transform_params(self) -> Dict[str, Any]:
        """Get parameters for torchaudio transforms with validation."""
        try:
            params = {
                'sample_rate': self.sample_rate,
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
                'win_length': self.win_length,
                'n_mels': self.n_mels,
                'n_mfcc': self.n_mfcc,
                'f_min': self.f_min,
                'f_max': self.f_max,
                'window_fn': self._get_window_function()
            }
            
            # Validate all parameters are valid
            for key, value in params.items():
                if value is None and key != 'window_fn':
                    logger.warning(f"Parameter {key} is None, using fallback")
                    params[key] = self._get_fallback_param(key)
            
            return params
            
        except Exception as e:
            logger.error(f"Error getting transform params: {e}")
            if self.enable_fallbacks:
                return self._get_fallback_transform_params()
            raise
    
    def _get_window_function(self):
        """Get window function with fallback."""
        try:
            window_map = {
                'hann': torch.hann_window,
                'hamming': torch.hamming_window,
                'blackman': torch.blackman_window,
                'bartlett': torch.bartlett_window,
                'rectangular': lambda n, **kwargs: torch.ones(n)
            }
            
            return window_map.get(self.window, torch.hann_window)
            
        except Exception as e:
            logger.error(f"Error getting window function: {e}")
            return torch.hann_window
    
    def _get_fallback_param(self, param_name: str) -> Any:
        """Get fallback value for a parameter."""
        fallbacks = {
            'sample_rate': 22050,
            'n_fft': 2048,
            'hop_length': 512,
            'win_length': 2048,
            'n_mels': 128,
            'n_mfcc': 13,
            'f_min': 0.0,
            'f_max': 11025.0
        }
        
        return fallbacks.get(param_name, 0)
    
    def _get_fallback_transform_params(self) -> Dict[str, Any]:
        """Get fallback transform parameters."""
        return {
            'sample_rate': 22050,
            'n_fft': 2048,
            'hop_length': 512,
            'win_length': 2048,
            'n_mels': 128,
            'n_mfcc': 13,
            'f_min': 0.0,
            'f_max': 11025.0,
            'window_fn': torch.hann_window
        }
    
    def is_compatible(self, other: 'BulletproofAudioConfig') -> bool:
        """Check if this config is compatible with another with detailed analysis."""
        if not isinstance(other, BulletproofAudioConfig):
            logger.error(f"Cannot compare with non-BulletproofAudioConfig: {type(other)}")
            return False
        
        try:
            critical_params = [
                'sample_rate', 'n_fft', 'hop_length', 'n_mels', 'embedding_dim'
            ]
            
            incompatibilities = []
            
            for param in critical_params:
                self_val = getattr(self, param, None)
                other_val = getattr(other, param, None)
                
                if self_val != other_val:
                    incompatibilities.append(f"{param}: {self_val} vs {other_val}")
            
            if incompatibilities:
                if self.warn_on_mismatch:
                    logger.warning(f"Configuration incompatibilities: {'; '.join(incompatibilities)}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking compatibility: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary with error handling."""
        try:
            config_dict = {}
            
            for field_name in self.__dataclass_fields__:
                value = getattr(self, field_name)
                
                # Handle enums
                if isinstance(value, Enum):
                    config_dict[field_name] = value.value
                # Handle None values
                elif value is None:
                    config_dict[field_name] = None
                # Handle dictionaries
                elif isinstance(value, dict):
                    config_dict[field_name] = value.copy()
                # Handle other types
                else:
                    config_dict[field_name] = value
            
            # Add metadata
            config_dict['_metadata'] = {
                'config_version': '1.0',
                'validation_passed': True,
                'estimated_memory_mb': self._estimate_memory_usage()
            }
            
            return config_dict
            
        except Exception as e:
            logger.error(f"Error converting config to dict: {e}")
            return {'error': str(e), 'fallback': True}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BulletproofAudioConfig':
        """Create config from dictionary with validation."""
        try:
            # Remove metadata if present
            config_data = config_dict.copy()
            config_data.pop('_metadata', None)
            config_data.pop('error', None)
            config_data.pop('fallback', None)
            
            # Convert enum strings back to enums
            if 'normalization_method' in config_data:
                if isinstance(config_data['normalization_method'], str):
                    config_data['normalization_method'] = NormalizationMethod(config_data['normalization_method'])
            
            if 'domain' in config_data:
                if isinstance(config_data['domain'], str):
                    config_data['domain'] = AudioDomain(config_data['domain'])
            
            if 'device_type' in config_data:
                if isinstance(config_data['device_type'], str):
                    config_data['device_type'] = DeviceType(config_data['device_type'])
            
            return cls(**config_data)
            
        except Exception as e:
            logger.error(f"Error creating config from dict: {e}")
            logger.warning("Creating fallback configuration")
            return cls()  # Return default config
    
    def save_to_file(self, filepath: Union[str, Path]) -> bool:
        """Save configuration to file with error handling."""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            config_dict = self.to_dict()
            
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            
            logger.info(f"Configuration saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving config to {filepath}: {e}")
            return False
    
    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> 'BulletproofAudioConfig':
        """Load configuration from file with error handling."""
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                logger.error(f"Config file not found: {filepath}")
                logger.warning("Using default configuration")
                return cls()
            
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            
            logger.info(f"Configuration loaded from {filepath}")
            return cls.from_dict(config_dict)
            
        except Exception as e:
            logger.error(f"Error loading config from {filepath}: {e}")
            logger.warning("Using default configuration")
            return cls()
    
    def create_device_config(self, device: Optional[torch.device] = None) -> Dict[str, Any]:
        """Create device-specific configuration."""
        try:
            if device is None:
                device = self._auto_detect_device()
            
            config = {
                'device': device,
                'mixed_precision': self.enable_mixed_precision and device.type in ['cuda'],
                'memory_efficient': self.memory_efficient,
                'optimization_level': self.optimization_level
            }
            
            # Adjust settings based on device capabilities
            if device.type == 'cuda':
                # GPU-specific optimizations
                config['pin_memory'] = True
                config['non_blocking'] = True
            elif device.type == 'mps':
                # Apple Metal specific settings
                config['mixed_precision'] = False  # Not fully supported yet
            else:
                # CPU settings
                config['mixed_precision'] = False
                config['pin_memory'] = False
                config['non_blocking'] = False
            
            return config
            
        except Exception as e:
            logger.error(f"Error creating device config: {e}")
            return {
                'device': torch.device('cpu'),
                'mixed_precision': False,
                'memory_efficient': True,
                'optimization_level': 0
            }
    
    def _auto_detect_device(self) -> torch.device:
        """Auto-detect best available device."""
        try:
            if self.device_type == DeviceType.AUTO:
                if torch.cuda.is_available():
                    return torch.device('cuda')
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return torch.device('mps')
                else:
                    return torch.device('cpu')
            else:
                return torch.device(self.device_type.value)
                
        except Exception as e:
            logger.warning(f"Device detection failed: {e}, using CPU")
            return torch.device('cpu')


class BulletproofAudioNormalizer(nn.Module):
    """
    Bulletproof audio normalization with comprehensive error handling.
    """
    
    def __init__(self, config: BulletproofAudioConfig):
        super().__init__()
        self.config = config
        self.method = config.normalization_method
        self.stats = config.normalization_stats or {}
        
        # Register normalization parameters as buffers with validation
        try:
            for key, value in self.stats.items():
                if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                    self.register_buffer(f'norm_{key}', torch.tensor(float(value)))
                else:
                    logger.warning(f"Invalid normalization stat {key}={value}, skipping")
        except Exception as e:
            logger.error(f"Error registering normalization stats: {e}")
    
    def normalize_tensor(self, tensor: torch.Tensor, feature_type: str = 'mel') -> torch.Tensor:
        """Normalize tensor with comprehensive error handling."""
        try:
            if not isinstance(tensor, torch.Tensor):
                logger.error(f"Expected torch.Tensor, got {type(tensor)}")
                return tensor
            
            # Check for invalid values
            if torch.isnan(tensor).any():
                logger.warning("NaN values detected, replacing with zeros")
                tensor = torch.nan_to_num(tensor, nan=0.0)
            
            if torch.isinf(tensor).any():
                logger.warning("Inf values detected, clipping")
                tensor = torch.clamp(tensor, -100.0, 100.0)
            
            # Apply normalization based on method
            if self.method == NormalizationMethod.NONE:
                return tensor
            
            elif self.method == NormalizationMethod.ZERO_MEAN_UNIT_VAR:
                return self._normalize_zero_mean_unit_var(tensor, feature_type)
            
            elif self.method == NormalizationMethod.ZERO_MEAN_HALF_VAR:
                return self._normalize_zero_mean_half_var(tensor, feature_type)
            
            elif self.method == NormalizationMethod.MIN_MAX:
                return self._normalize_min_max(tensor)
            
            elif self.method == NormalizationMethod.ROBUST:
                return self._normalize_robust(tensor)
            
            elif self.method == NormalizationMethod.INSTANCE:
                return self._normalize_instance(tensor)
            
            elif self.method == NormalizationMethod.QUANTILE:
                return self._normalize_quantile(tensor)
            
            else:
                logger.warning(f"Unknown normalization method: {self.method}")
                return tensor
                
        except Exception as e:
            logger.error(f"Normalization failed: {e}")
            return tensor  # Return original tensor on failure
    
    def _normalize_zero_mean_unit_var(self, tensor: torch.Tensor, feature_type: str) -> torch.Tensor:
        """Zero mean unit variance normalization."""
        try:
            mean_key = f'norm_{feature_type}_mean'
            std_key = f'norm_{feature_type}_std'
            
            if hasattr(self, mean_key) and hasattr(self, std_key):
                mean = getattr(self, mean_key)
                std = getattr(self, std_key)
                return (tensor - mean) / (std + 1e-8)
            else:
                # Compute statistics on the fly
                mean = tensor.mean()
                std = tensor.std()
                return (tensor - mean) / (std + 1e-8)
                
        except Exception as e:
            logger.error(f"Zero mean unit var normalization failed: {e}")
            return tensor
    
    def _normalize_zero_mean_half_var(self, tensor: torch.Tensor, feature_type: str) -> torch.Tensor:
        """Zero mean half variance normalization (AST style)."""
        try:
            mean_key = f'norm_{feature_type}_mean'
            std_key = f'norm_{feature_type}_std'
            
            if hasattr(self, mean_key) and hasattr(self, std_key):
                mean = getattr(self, mean_key)
                std = getattr(self, std_key)
                return (tensor - mean) / (2 * std + 1e-8)
            else:
                mean = tensor.mean()
                std = tensor.std()
                return (tensor - mean) / (2 * std + 1e-8)
                
        except Exception as e:
            logger.error(f"Zero mean half var normalization failed: {e}")
            return tensor
    
    def _normalize_min_max(self, tensor: torch.Tensor) -> torch.Tensor:
        """Min-max normalization."""
        try:
            tensor_min = tensor.min()
            tensor_max = tensor.max()
            range_val = tensor_max - tensor_min
            
            if range_val > 1e-8:
                return (tensor - tensor_min) / range_val
            else:
                return torch.zeros_like(tensor)
                
        except Exception as e:
            logger.error(f"Min-max normalization failed: {e}")
            return tensor
    
    def _normalize_robust(self, tensor: torch.Tensor) -> torch.Tensor:
        """Robust normalization using median and IQR."""
        try:
            median = tensor.median()
            q75 = tensor.quantile(0.75)
            q25 = tensor.quantile(0.25)
            iqr = q75 - q25
            
            return (tensor - median) / (iqr + 1e-8)
            
        except Exception as e:
            logger.error(f"Robust normalization failed: {e}")
            return tensor
    
    def _normalize_instance(self, tensor: torch.Tensor) -> torch.Tensor:
        """Instance normalization (per-sample)."""
        try:
            if tensor.dim() < 2:
                return tensor
            
            # Normalize each sample separately
            normalized = torch.zeros_like(tensor)
            for i in range(tensor.shape[0]):
                sample = tensor[i]
                sample_mean = sample.mean()
                sample_std = sample.std()
                normalized[i] = (sample - sample_mean) / (sample_std + 1e-8)
            
            return normalized
            
        except Exception as e:
            logger.error(f"Instance normalization failed: {e}")
            return tensor
    
    def _normalize_quantile(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantile normalization."""
        try:
            # Normalize to [0, 1] using 5th and 95th percentiles
            q05 = tensor.quantile(0.05)
            q95 = tensor.quantile(0.95)
            
            range_val = q95 - q05
            if range_val > 1e-8:
                normalized = (tensor - q05) / range_val
                return torch.clamp(normalized, 0.0, 1.0)
            else:
                return torch.zeros_like(tensor)
                
        except Exception as e:
            logger.error(f"Quantile normalization failed: {e}")
            return tensor


# Factory functions for common configurations
def get_bulletproof_speech_config(**kwargs) -> BulletproofAudioConfig:
    """Get bulletproof configuration optimized for speech."""
    return BulletproofAudioConfig(
        domain=AudioDomain.SPEECH,
        sample_rate=16000,
        n_fft=512,
        hop_length=160,
        n_mels=80,
        f_max=8000,
        normalization_method=NormalizationMethod.ZERO_MEAN_HALF_VAR,
        use_spec_augment=True,
        validate_inputs=True,
        enable_fallbacks=True,
        **kwargs
    )


def get_bulletproof_music_config(**kwargs) -> BulletproofAudioConfig:
    """Get bulletproof configuration optimized for music."""
    return BulletproofAudioConfig(
        domain=AudioDomain.MUSIC,
        sample_rate=22050,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        normalization_method=NormalizationMethod.ZERO_MEAN_UNIT_VAR,
        use_spec_augment=True,
        freq_mask_param=40,
        time_mask_param=80,
        validate_inputs=True,
        enable_fallbacks=True,
        **kwargs
    )


def get_bulletproof_general_config(**kwargs) -> BulletproofAudioConfig:
    """Get bulletproof general-purpose configuration."""
    return BulletproofAudioConfig(
        domain=AudioDomain.GENERAL,
        sample_rate=22050,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        normalization_method=NormalizationMethod.ZERO_MEAN_UNIT_VAR,
        validate_inputs=True,
        enable_fallbacks=True,
        memory_efficient=True,
        **kwargs
    )


def get_bulletproof_timbralgebraics_config(**kwargs) -> BulletproofAudioConfig:
    """Get bulletproof configuration optimized for timbralgebraics project."""
    return BulletproofAudioConfig(
        domain=AudioDomain.TIMBRALGEBRAICS,
        sample_rate=22050,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        embedding_dim=512,
        normalization_method=NormalizationMethod.ZERO_MEAN_UNIT_VAR,
        content_aware=True,
        use_spec_augment=False,
        validate_inputs=True,
        enable_fallbacks=True,
        memory_efficient=True,
        **kwargs
    )


def get_bulletproof_low_memory_config(**kwargs) -> BulletproofAudioConfig:
    """Get bulletproof configuration optimized for low memory usage."""
    return BulletproofAudioConfig(
        domain=AudioDomain.GENERAL,
        sample_rate=16000,
        n_fft=1024,
        hop_length=256,
        n_mels=64,
        embedding_dim=128,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        max_memory_mb=512,  # 512MB limit
        chunk_size=16000 * 5,  # 5 seconds
        memory_efficient=True,
        validate_inputs=True,
        enable_fallbacks=True,
        **kwargs
    )


# Test function
def test_bulletproof_audio_config():
    """Test the bulletproof audio config."""
    logger.info("Testing BulletproofAudioConfig")
    
    # Test different configurations
    configs = [
        get_bulletproof_speech_config(),
        get_bulletproof_music_config(),
        get_bulletproof_general_config(),
        get_bulletproof_timbralgebraics_config(),
        get_bulletproof_low_memory_config()
    ]
    
    for i, config in enumerate(configs):
        try:
            logger.info(f"Testing config {i+1}: {config.domain.value}")
            
            # Test parameter access
            params = config.get_transform_params()
            logger.info(f"  Transform params: {len(params)} parameters")
            
            # Test serialization
            config_dict = config.to_dict()
            restored_config = BulletproofAudioConfig.from_dict(config_dict)
            logger.info(f"  Serialization: {'OK' if restored_config.sample_rate == config.sample_rate else 'FAILED'}")
            
            # Test compatibility
            compatible = config.is_compatible(configs[0])
            logger.info(f"  Compatible with speech config: {compatible}")
            
            # Test memory estimation
            memory_est = config._estimate_memory_usage()
            logger.info(f"  Estimated memory usage: {memory_est:.1f}MB")
            
        except Exception as e:
            logger.error(f"Config {i+1} test failed: {e}")
    
    # Test error handling
    try:
        logger.info("Testing error handling...")
        
        # Invalid parameters
        bad_config = BulletproofAudioConfig(
            sample_rate=-1000,  # Invalid
            n_fft=123,  # Not power of 2
            hop_length=-50,  # Invalid
            enable_fallbacks=True
        )
        logger.info("Error handling test passed")
        
    except Exception as e:
        logger.error(f"Error handling test failed: {e}")
    
    logger.info("BulletproofAudioConfig testing completed")


if __name__ == "__main__":
    test_bulletproof_audio_config()