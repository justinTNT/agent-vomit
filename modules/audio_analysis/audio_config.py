"""
Standardized audio processing configuration for agent-vomit modules.

Provides consistent parameters, normalization, and compatibility across all audio modules.
Supports multiple domain-specific configurations (speech, music, general).
"""

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings


class AudioDomain(Enum):
    """Audio domain types with different optimal configurations."""
    SPEECH = "speech"
    MUSIC = "music" 
    GENERAL = "general"
    TIMBRALGEBRAICS = "timbralgebraics"  # Custom for your project


class NormalizationMethod(Enum):
    """Audio normalization methods."""
    NONE = "none"
    ZERO_MEAN_UNIT_VAR = "zero_mean_unit_var"  # (x - μ) / σ
    ZERO_MEAN_HALF_VAR = "zero_mean_half_var"   # (x - μ) / (2σ) - AudioSet style
    MIN_MAX = "min_max"                         # (x - min) / (max - min)
    ROBUST = "robust"                           # (x - median) / IQR
    INSTANCE = "instance"                       # Per-sample normalization


@dataclass
class AudioModuleConfig:
    """
    Standardized configuration for all audio processing modules.
    
    Ensures parameter consistency and enables module chaining.
    """
    
    # Core audio parameters
    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    win_length: Optional[int] = None  # Defaults to n_fft
    window: str = "hann"
    
    # Spectral parameters
    n_mels: int = 128
    n_mfcc: int = 13
    n_chroma: int = 12
    f_min: float = 0.0
    f_max: Optional[float] = None  # Defaults to sample_rate // 2
    
    # Model architecture parameters
    embedding_dim: int = 512
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    
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
    
    # Domain-specific settings
    domain: AudioDomain = AudioDomain.GENERAL
    content_aware: bool = True
    
    # Module compatibility
    feature_compatibility_check: bool = True
    auto_convert_sample_rate: bool = True
    warn_on_mismatch: bool = True
    
    def __post_init__(self):
        """Initialize derived parameters and validate configuration."""
        if self.win_length is None:
            self.win_length = self.n_fft
            
        if self.f_max is None:
            self.f_max = self.sample_rate // 2
            
        # Set domain-specific normalization stats
        if self.normalization_stats is None:
            self.normalization_stats = self._get_domain_normalization_stats()
            
        # Validate parameters
        self._validate_config()
        
    def _get_domain_normalization_stats(self) -> Dict[str, float]:
        """Get normalization statistics for different domains."""
        
        # Standard dataset statistics
        stats = {
            AudioDomain.SPEECH: {
                "mel_mean": -5.081, "mel_std": 4.969,      # LibriSpeech-like
                "mfcc_mean": 0.0, "mfcc_std": 15.0,
                "chroma_mean": 0.083, "chroma_std": 0.146
            },
            AudioDomain.MUSIC: {
                "mel_mean": -4.268, "mel_std": 4.569,      # Music dataset stats
                "mfcc_mean": 0.0, "mfcc_std": 12.0,
                "chroma_mean": 0.083, "chroma_std": 0.15
            },
            AudioDomain.GENERAL: {
                "mel_mean": -4.268, "mel_std": 4.569,      # AudioSet stats
                "mfcc_mean": 0.0, "mfcc_std": 13.5,
                "chroma_mean": 0.083, "chroma_std": 0.15
            },
            AudioDomain.TIMBRALGEBRAICS: {
                "mel_mean": -4.0, "mel_std": 4.0,          # Guitar-optimized
                "mfcc_mean": 0.0, "mfcc_std": 10.0,
                "chroma_mean": 0.083, "chroma_std": 0.12
            }
        }
        
        return stats.get(self.domain, stats[AudioDomain.GENERAL])
        
    def _validate_config(self):
        """Validate configuration parameters."""
        assert self.sample_rate > 0, "Sample rate must be positive"
        assert self.n_fft > 0, "n_fft must be positive"
        assert self.hop_length > 0, "hop_length must be positive"
        assert 0 <= self.dropout <= 1, "dropout must be in [0, 1]"
        assert self.f_min >= 0, "f_min must be non-negative"
        assert self.f_max <= self.sample_rate // 2, "f_max must be <= Nyquist frequency"
        
        # Warn about non-standard configurations
        if self.sample_rate not in [16000, 22050, 44100, 48000]:
            warnings.warn(f"Non-standard sample rate: {self.sample_rate}")
            
    def get_transform_params(self) -> Dict[str, Any]:
        """Get parameters for torchaudio transforms."""
        return {
            'sample_rate': self.sample_rate,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'win_length': self.win_length,
            'n_mels': self.n_mels,
            'n_mfcc': self.n_mfcc,
            'f_min': self.f_min,
            'f_max': self.f_max
        }
        
    def is_compatible(self, other: 'AudioModuleConfig') -> bool:
        """Check if this config is compatible with another."""
        critical_params = [
            'sample_rate', 'n_fft', 'hop_length', 'n_mels', 'embedding_dim'
        ]
        
        for param in critical_params:
            if getattr(self, param) != getattr(other, param):
                if self.warn_on_mismatch:
                    warnings.warn(f"Parameter mismatch: {param} = {getattr(self, param)} vs {getattr(other, param)}")
                return False
                
        return True
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'sample_rate': self.sample_rate,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'win_length': self.win_length,
            'n_mels': self.n_mels,
            'n_mfcc': self.n_mfcc,
            'n_chroma': self.n_chroma,
            'f_min': self.f_min,
            'f_max': self.f_max,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'normalization_method': self.normalization_method.value,
            'domain': self.domain.value
        }
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AudioModuleConfig':
        """Create config from dictionary."""
        # Convert enum strings back to enums
        if 'normalization_method' in config_dict:
            config_dict['normalization_method'] = NormalizationMethod(config_dict['normalization_method'])
        if 'domain' in config_dict:
            config_dict['domain'] = AudioDomain(config_dict['domain'])
            
        return cls(**config_dict)


# Pre-defined configurations for different use cases
def get_speech_config() -> AudioModuleConfig:
    """Configuration optimized for speech processing."""
    return AudioModuleConfig(
        sample_rate=16000,
        n_fft=512,
        hop_length=160,
        n_mels=80,
        f_max=8000,
        domain=AudioDomain.SPEECH,
        normalization_method=NormalizationMethod.ZERO_MEAN_HALF_VAR,
        use_spec_augment=True
    )


def get_music_config() -> AudioModuleConfig:
    """Configuration optimized for music processing."""
    return AudioModuleConfig(
        sample_rate=22050,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        domain=AudioDomain.MUSIC,
        normalization_method=NormalizationMethod.ZERO_MEAN_UNIT_VAR,
        use_spec_augment=True,
        freq_mask_param=40,
        time_mask_param=80
    )


def get_general_config() -> AudioModuleConfig:
    """General-purpose configuration."""
    return AudioModuleConfig(
        sample_rate=22050,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        domain=AudioDomain.GENERAL,
        normalization_method=NormalizationMethod.ZERO_MEAN_UNIT_VAR
    )


def get_timbralgebraics_config() -> AudioModuleConfig:
    """Configuration optimized for timbralgebraics project."""
    return AudioModuleConfig(
        sample_rate=22050,  # Match RAVE
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        embedding_dim=512,  # Match RAVE latent dim
        domain=AudioDomain.TIMBRALGEBRAICS,
        normalization_method=NormalizationMethod.ZERO_MEAN_UNIT_VAR,
        content_aware=True,
        use_spec_augment=False  # Preserve fidelity for latent operations
    )


def get_ast_compatible_config() -> AudioModuleConfig:
    """Configuration compatible with original AST implementation."""
    return AudioModuleConfig(
        sample_rate=16000,
        n_fft=512,  # AST uses smaller FFT
        hop_length=160,
        n_mels=128,
        embedding_dim=768,  # AST base model
        domain=AudioDomain.GENERAL,
        normalization_method=NormalizationMethod.ZERO_MEAN_HALF_VAR,  # AST normalization
        use_spec_augment=True
    )


class AudioNormalizer(nn.Module):
    """
    Standardized audio normalization module.
    
    Applies consistent normalization across all audio processing modules.
    """
    
    def __init__(self, config: AudioModuleConfig):
        super().__init__()
        self.config = config
        self.method = config.normalization_method
        self.stats = config.normalization_stats
        
        # Register normalization parameters as buffers
        if self.stats:
            for key, value in self.stats.items():
                self.register_buffer(f'norm_{key}', torch.tensor(value))
                
    def normalize_mel_spectrogram(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Normalize mel-spectrogram according to configuration."""
        if self.config.apply_log_scaling:
            mel_spec = torch.log(mel_spec.clamp(min=self.config.log_epsilon))
            
        if self.method == NormalizationMethod.NONE:
            return mel_spec
            
        elif self.method == NormalizationMethod.ZERO_MEAN_UNIT_VAR:
            if hasattr(self, 'norm_mel_mean') and hasattr(self, 'norm_mel_std'):
                return (mel_spec - self.norm_mel_mean) / self.norm_mel_std
            else:
                return (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
                
        elif self.method == NormalizationMethod.ZERO_MEAN_HALF_VAR:
            if hasattr(self, 'norm_mel_mean') and hasattr(self, 'norm_mel_std'):
                return (mel_spec - self.norm_mel_mean) / (2 * self.norm_mel_std)
            else:
                return (mel_spec - mel_spec.mean()) / (2 * mel_spec.std() + 1e-8)
                
        elif self.method == NormalizationMethod.MIN_MAX:
            mel_min = mel_spec.min()
            mel_max = mel_spec.max()
            return (mel_spec - mel_min) / (mel_max - mel_min + 1e-8)
            
        elif self.method == NormalizationMethod.ROBUST:
            mel_median = mel_spec.median()
            mel_q75 = mel_spec.quantile(0.75)
            mel_q25 = mel_spec.quantile(0.25)
            iqr = mel_q75 - mel_q25
            return (mel_spec - mel_median) / (iqr + 1e-8)
            
        elif self.method == NormalizationMethod.INSTANCE:
            # Per-sample normalization
            batch_size = mel_spec.shape[0]
            normalized = torch.zeros_like(mel_spec)
            for i in range(batch_size):
                sample = mel_spec[i]
                normalized[i] = (sample - sample.mean()) / (sample.std() + 1e-8)
            return normalized
            
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
            
    def normalize_mfcc(self, mfcc: torch.Tensor) -> torch.Tensor:
        """Normalize MFCC features."""
        if self.method == NormalizationMethod.NONE:
            return mfcc
            
        if hasattr(self, 'norm_mfcc_mean') and hasattr(self, 'norm_mfcc_std'):
            return (mfcc - self.norm_mfcc_mean) / self.norm_mfcc_std
        else:
            return (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
            
    def normalize_chroma(self, chroma: torch.Tensor) -> torch.Tensor:
        """Normalize chroma features."""
        if self.method == NormalizationMethod.NONE:
            return chroma
            
        if hasattr(self, 'norm_chroma_mean') and hasattr(self, 'norm_chroma_std'):
            return (chroma - self.norm_chroma_mean) / self.norm_chroma_std
        else:
            return (chroma - chroma.mean()) / (chroma.std() + 1e-8)


class SpecAugment(nn.Module):
    """
    SpecAugment implementation for data augmentation.
    
    Applies frequency and time masking to spectrograms.
    """
    
    def __init__(self, config: AudioModuleConfig):
        super().__init__()
        self.config = config
        self.freq_mask_param = config.freq_mask_param
        self.time_mask_param = config.time_mask_param
        self.n_freq_masks = config.n_freq_masks
        self.n_time_masks = config.n_time_masks
        
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment to spectrogram."""
        if not self.training or not self.config.use_spec_augment:
            return spec
            
        # spec shape: [batch, freq, time]
        batch_size, n_freq, n_time = spec.shape
        
        # Apply frequency masking
        for _ in range(self.n_freq_masks):
            freq_mask_size = torch.randint(0, min(self.freq_mask_param, n_freq), (1,)).item()
            if freq_mask_size > 0:
                freq_mask_start = torch.randint(0, n_freq - freq_mask_size, (1,)).item()
                spec[:, freq_mask_start:freq_mask_start + freq_mask_size, :] = 0
                
        # Apply time masking
        for _ in range(self.n_time_masks):
            time_mask_size = torch.randint(0, min(self.time_mask_param, n_time), (1,)).item()
            if time_mask_size > 0:
                time_mask_start = torch.randint(0, n_time - time_mask_size, (1,)).item()
                spec[:, :, time_mask_start:time_mask_start + time_mask_size] = 0
                
        return spec


class AudioModuleBase(nn.Module):
    """
    Base class for all audio processing modules.
    
    Provides standardized configuration management and compatibility checking.
    """
    
    def __init__(self, config: AudioModuleConfig):
        super().__init__()
        self.config = config
        self.normalizer = AudioNormalizer(config)
        self.spec_augment = SpecAugment(config) if config.use_spec_augment else None
        
        # Store module type for compatibility checking
        self.module_type = self.__class__.__name__
        
    def check_compatibility(self, other_config: AudioModuleConfig) -> bool:
        """Check if this module is compatible with another configuration."""
        if not self.config.feature_compatibility_check:
            return True
            
        return self.config.is_compatible(other_config)
        
    def get_config(self) -> AudioModuleConfig:
        """Get module configuration."""
        return self.config
        
    def extract_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract and normalize mel-spectrogram using standard configuration."""
        # Ensure correct input format
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
            
        # Convert sample rate if needed
        if self.config.auto_convert_sample_rate:
            # This would need torchaudio.functional.resample in practice
            pass
            
        # Extract mel-spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            **self.config.get_transform_params()
        )
        
        mel_spec = mel_transform(waveform)
        
        # Apply normalization
        mel_spec = self.normalizer.normalize_mel_spectrogram(mel_spec)
        
        # Apply SpecAugment if enabled
        if self.spec_augment is not None:
            mel_spec = self.spec_augment(mel_spec)
            
        return mel_spec


# Compatibility checking utilities
def check_module_compatibility(modules: List[nn.Module]) -> bool:
    """Check if a list of audio modules are compatible."""
    if len(modules) < 2:
        return True
        
    reference_config = None
    for module in modules:
        if hasattr(module, 'config'):
            if reference_config is None:
                reference_config = module.config
            else:
                if not reference_config.is_compatible(module.config):
                    return False
                    
    return True


def get_compatibility_report(modules: List[nn.Module]) -> Dict[str, Any]:
    """Get detailed compatibility report for audio modules."""
    report = {
        'compatible': True,
        'issues': [],
        'recommendations': []
    }
    
    configs = []
    for i, module in enumerate(modules):
        if hasattr(module, 'config'):
            configs.append((i, module.__class__.__name__, module.config))
            
    # Check pairwise compatibility
    for i in range(len(configs)):
        for j in range(i + 1, len(configs)):
            idx1, name1, config1 = configs[i]
            idx2, name2, config2 = configs[j]
            
            if not config1.is_compatible(config2):
                report['compatible'] = False
                report['issues'].append(f"Incompatible: {name1} (module {idx1}) vs {name2} (module {idx2})")
                
    # Generate recommendations
    if not report['compatible']:
        report['recommendations'].append("Use a common AudioModuleConfig for all modules")
        report['recommendations'].append("Consider using get_timbralgebraics_config() for consistency")
        
    return report


# Example usage and testing
if __name__ == "__main__":
    # Test different configurations
    speech_config = get_speech_config()
    music_config = get_music_config()
    timbral_config = get_timbralgebraics_config()
    
    print("Speech config:", speech_config.to_dict())
    print("Music config:", music_config.to_dict())
    print("Timbralgebraics config:", timbral_config.to_dict())
    
    # Test compatibility
    print("Speech-Music compatible:", speech_config.is_compatible(music_config))
    print("Music-Timbral compatible:", music_config.is_compatible(timbral_config))
    
    # Test normalization
    normalizer = AudioNormalizer(timbral_config)
    dummy_mel = torch.randn(2, 128, 1000)  # [batch, n_mels, time]
    normalized = normalizer.normalize_mel_spectrogram(dummy_mel)
    
    print(f"Original mel stats: mean={dummy_mel.mean():.3f}, std={dummy_mel.std():.3f}")
    print(f"Normalized mel stats: mean={normalized.mean():.3f}, std={normalized.std():.3f}")
    
    # Test SpecAugment
    spec_aug = SpecAugment(music_config)
    augmented = spec_aug(dummy_mel)
    print(f"Augmented spectrogram shape: {augmented.shape}")