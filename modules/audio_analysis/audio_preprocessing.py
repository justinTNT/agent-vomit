"""
Standardized audio preprocessing pipeline for agent-vomit modules.

Provides reference-compliant feature extraction, normalization, and data augmentation.
Ensures consistency across all audio analysis modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as F_audio
import numpy as np
import librosa
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

from .audio_config import AudioModuleConfig, AudioNormalizer, SpecAugment, NormalizationMethod


class StandardAudioPreprocessor(nn.Module):
    """
    Standardized audio preprocessing pipeline.
    
    Provides reference-compliant feature extraction with proper normalization,
    sample rate conversion, and data augmentation.
    """
    
    def __init__(self, config: AudioModuleConfig):
        super().__init__()
        self.config = config
        
        # Core transforms
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
            window_fn=torch.hann_window,
            power=2.0
        )
        
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=config.sample_rate,
            n_mfcc=config.n_mfcc,
            melkwargs={
                'n_fft': config.n_fft,
                'hop_length': config.hop_length,
                'n_mels': config.n_mels,
                'f_min': config.f_min,
                'f_max': config.f_max
            }
        )
        
        # Chroma filterbank
        self.chroma_filterbank = self._create_chroma_filterbank()
        
        # Normalization and augmentation
        self.normalizer = AudioNormalizer(config)
        self.spec_augment = SpecAugment(config) if config.use_spec_augment else None
        
        # Sample rate conversion
        self.target_sample_rate = config.sample_rate
        
    def _create_chroma_filterbank(self):
        """Create chroma filterbank using librosa."""
        chroma_fb = librosa.filters.chroma(
            sr=self.config.sample_rate,
            n_fft=self.config.n_fft,
            n_chroma=self.config.n_chroma,
            tuning=0.0
        )
        chroma_fb = torch.from_numpy(chroma_fb).float()
        self.register_buffer('chroma_filterbank', chroma_fb)
        return chroma_fb
        
    def convert_sample_rate(self, waveform: torch.Tensor, orig_sample_rate: int) -> torch.Tensor:
        """Convert waveform to target sample rate."""
        if orig_sample_rate == self.target_sample_rate:
            return waveform
            
        if self.config.auto_convert_sample_rate:
            # Use high-quality resampling
            resampled = F_audio.resample(
                waveform, 
                orig_freq=orig_sample_rate,
                new_freq=self.target_sample_rate,
                resampling_method="sinc_interp_hann"
            )
            return resampled
        else:
            warnings.warn(f"Sample rate mismatch: {orig_sample_rate} vs {self.target_sample_rate}")
            return waveform
            
    def ensure_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert stereo to mono if needed."""
        if waveform.dim() == 3:  # [batch, channels, samples]
            if waveform.shape[1] == 1:
                return waveform.squeeze(1)  # Already mono
            else:
                return waveform.mean(dim=1)  # Average channels
        elif waveform.dim() == 2 and waveform.shape[0] > 1:  # [channels, samples]
            return waveform.mean(dim=0, keepdim=True)
        else:
            return waveform
            
    def normalize_audio_amplitude(self, waveform: torch.Tensor, method: str = "rms") -> torch.Tensor:
        """Normalize audio amplitude."""
        if method == "peak":
            # Peak normalization
            peak = torch.max(torch.abs(waveform), dim=-1, keepdim=True)[0]
            return waveform / (peak + 1e-8)
        elif method == "rms":
            # RMS normalization
            rms = torch.sqrt(torch.mean(waveform ** 2, dim=-1, keepdim=True))
            return waveform / (rms + 1e-8) * 0.1  # Scale to reasonable level
        elif method == "lufs":
            # Simplified LUFS-like normalization
            # Real implementation would use ITU-R BS.1770 standard
            energy = torch.mean(waveform ** 2, dim=-1, keepdim=True)
            target_energy = 10 ** (-23 / 10)  # -23 LUFS
            scale = torch.sqrt(target_energy / (energy + 1e-8))
            return waveform * scale.clamp(max=10.0)  # Limit maximum gain
        else:
            return waveform
            
    def extract_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract mel-spectrogram with proper normalization."""
        mel_spec = self.mel_transform(waveform)
        
        # Apply log scaling if configured
        if self.config.apply_log_scaling:
            mel_spec = torch.log(mel_spec.clamp(min=self.config.log_epsilon))
            
        # Apply normalization
        mel_spec = self.normalizer.normalize_mel_spectrogram(mel_spec)
        
        # Apply SpecAugment if enabled and in training mode
        if self.spec_augment is not None:
            mel_spec = self.spec_augment(mel_spec)
            
        return mel_spec
        
    def extract_mfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract MFCC features with normalization."""
        mfcc = self.mfcc_transform(waveform)
        
        # Apply normalization
        mfcc = self.normalizer.normalize_mfcc(mfcc)
        
        return mfcc
        
    def extract_chroma(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract chroma features."""
        # Compute STFT
        stft = torch.stft(
            waveform,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=torch.hann_window(self.config.win_length, device=waveform.device),
            return_complex=True
        )
        
        # Convert to magnitude
        magnitude = torch.abs(stft)
        
        # Apply chroma filterbank
        chroma = torch.matmul(self.chroma_filterbank, magnitude)
        
        # Normalize chroma vectors
        chroma = F.normalize(chroma, p=2, dim=1)
        
        # Apply normalization
        chroma = self.normalizer.normalize_chroma(chroma)
        
        return chroma
        
    def extract_spectral_features(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract comprehensive spectral features."""
        # STFT for spectral analysis
        stft = torch.stft(
            waveform,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=torch.hann_window(self.config.win_length, device=waveform.device),
            return_complex=True
        )
        
        magnitude = torch.abs(stft)
        power_spec = magnitude ** 2
        
        # Frequency bins
        freqs = torch.linspace(0, self.config.sample_rate // 2, 
                              magnitude.shape[1], device=waveform.device)
        freqs = freqs.unsqueeze(0).unsqueeze(-1)  # [1, freq, 1]
        
        # Spectral centroid
        spectral_centroid = torch.sum(magnitude * freqs, dim=1) / (torch.sum(magnitude, dim=1) + 1e-8)
        
        # Spectral bandwidth
        centroid_expanded = spectral_centroid.unsqueeze(1)
        spectral_bandwidth = torch.sqrt(
            torch.sum(magnitude * (freqs - centroid_expanded)**2, dim=1) / 
            (torch.sum(magnitude, dim=1) + 1e-8)
        )
        
        # Spectral rolloff (85th percentile)
        cumsum_mag = torch.cumsum(magnitude, dim=1)
        total_energy = cumsum_mag[:, -1:, :]
        rolloff_threshold = 0.85 * total_energy
        
        # Find rolloff frequency
        rolloff_indices = torch.searchsorted(
            cumsum_mag.transpose(1, 2), 
            rolloff_threshold.transpose(1, 2)
        ).transpose(1, 2).clamp(max=len(freqs.squeeze())-1)
        
        spectral_rolloff = freqs.squeeze()[rolloff_indices.squeeze(1)]
        
        # Spectral flatness (Wiener entropy)
        log_magnitude = torch.log(magnitude + 1e-8)
        geometric_mean = torch.exp(torch.mean(log_magnitude, dim=1))
        arithmetic_mean = torch.mean(magnitude, dim=1)
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-8)
        
        # Zero crossing rate
        zcr = self._compute_zero_crossing_rate(waveform)
        
        return {
            'magnitude_spectrum': magnitude,
            'power_spectrum': power_spec,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_rolloff': spectral_rolloff,
            'spectral_flatness': spectral_flatness,
            'zero_crossing_rate': zcr
        }
        
    def _compute_zero_crossing_rate(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute zero crossing rate."""
        # Frame the signal
        frame_length = self.config.hop_length * 2
        frames = waveform.unfold(-1, frame_length, self.config.hop_length)
        
        # Count zero crossings per frame
        signs = torch.sign(frames)
        sign_changes = torch.abs(torch.diff(signs, dim=-1))
        zcr = torch.sum(sign_changes, dim=-1) / frame_length
        
        return zcr
        
    def forward(
        self, 
        waveform: torch.Tensor,
        orig_sample_rate: Optional[int] = None,
        extract_features: List[str] = ['mel', 'mfcc', 'chroma'],
        normalize_amplitude: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Complete preprocessing pipeline.
        
        Args:
            waveform: Input audio [batch, samples] or [batch, channels, samples]
            orig_sample_rate: Original sample rate (for conversion)
            extract_features: List of features to extract
            normalize_amplitude: Whether to normalize audio amplitude
            
        Returns:
            Dictionary with extracted features
        """
        # Convert sample rate if needed
        if orig_sample_rate is not None and orig_sample_rate != self.config.sample_rate:
            waveform = self.convert_sample_rate(waveform, orig_sample_rate)
            
        # Ensure mono audio
        waveform = self.ensure_mono(waveform)
        
        # Normalize amplitude
        if normalize_amplitude:
            waveform = self.normalize_audio_amplitude(waveform, method="rms")
            
        # Extract requested features
        features = {'waveform': waveform}
        
        if 'mel' in extract_features:
            features['mel_spectrogram'] = self.extract_mel_spectrogram(waveform)
            
        if 'mfcc' in extract_features:
            features['mfcc'] = self.extract_mfcc(waveform)
            
        if 'chroma' in extract_features:
            features['chroma'] = self.extract_chroma(waveform)
            
        if 'spectral' in extract_features:
            spectral_features = self.extract_spectral_features(waveform)
            features.update(spectral_features)
            
        return features


class ReferenceDatasetStats:
    """
    Reference dataset statistics for proper normalization.
    
    Contains pre-computed statistics from standard audio datasets.
    """
    
    # AudioSet statistics (computed on subset)
    AUDIOSET_STATS = {
        'mel_mean': -4.2677393,
        'mel_std': 4.5689974,
        'mfcc_mean': 0.0,
        'mfcc_std': 13.47,
        'chroma_mean': 0.0833,
        'chroma_std': 0.1468
    }
    
    # LibriSpeech statistics
    LIBRISPEECH_STATS = {
        'mel_mean': -5.081,
        'mel_std': 4.969,
        'mfcc_mean': 0.0,
        'mfcc_std': 15.2,
        'chroma_mean': 0.083,
        'chroma_std': 0.146
    }
    
    # Music dataset statistics (FMA/GTZAN-like)
    MUSIC_STATS = {
        'mel_mean': -4.268,
        'mel_std': 4.569,
        'mfcc_mean': 0.0,
        'mfcc_std': 12.34,
        'chroma_mean': 0.083,
        'chroma_std': 0.15
    }
    
    # ESC-50 statistics
    ESC50_STATS = {
        'mel_mean': -3.97,
        'mel_std': 4.12,
        'mfcc_mean': 0.0,
        'mfcc_std': 11.8,
        'chroma_mean': 0.083,
        'chroma_std': 0.142
    }
    
    @classmethod
    def get_stats(cls, dataset: str) -> Dict[str, float]:
        """Get statistics for a specific dataset."""
        stats_map = {
            'audioset': cls.AUDIOSET_STATS,
            'librispeech': cls.LIBRISPEECH_STATS,
            'music': cls.MUSIC_STATS,
            'esc50': cls.ESC50_STATS
        }
        
        return stats_map.get(dataset.lower(), cls.AUDIOSET_STATS)


class MultiScalePreprocessor(nn.Module):
    """
    Multi-scale audio preprocessing for different temporal resolutions.
    
    Useful for models that need features at multiple time scales.
    """
    
    def __init__(self, config: AudioModuleConfig, scales: List[int] = [512, 1024, 2048]):
        super().__init__()
        self.config = config
        self.scales = scales
        
        # Create preprocessors for each scale
        self.preprocessors = nn.ModuleDict()
        for scale in scales:
            scale_config = AudioModuleConfig(
                sample_rate=config.sample_rate,
                n_fft=scale,
                hop_length=scale // 4,
                n_mels=config.n_mels,
                normalization_method=config.normalization_method,
                normalization_stats=config.normalization_stats
            )
            self.preprocessors[f'scale_{scale}'] = StandardAudioPreprocessor(scale_config)
            
    def forward(self, waveform: torch.Tensor, **kwargs) -> Dict[str, Dict[str, torch.Tensor]]:
        """Extract features at multiple scales."""
        multi_scale_features = {}
        
        for scale in self.scales:
            preprocessor = self.preprocessors[f'scale_{scale}']
            features = preprocessor(waveform, **kwargs)
            multi_scale_features[f'scale_{scale}'] = features
            
        return multi_scale_features


class ASTCompatiblePreprocessor(StandardAudioPreprocessor):
    """
    Preprocessor specifically configured for AST compatibility.
    
    Matches the original AST preprocessing pipeline exactly.
    """
    
    def __init__(self):
        # AST-specific configuration
        config = AudioModuleConfig(
            sample_rate=16000,
            n_fft=512,
            hop_length=160,  # 10ms hop
            n_mels=128,
            f_min=0,
            f_max=8000,
            normalization_method=NormalizationMethod.ZERO_MEAN_HALF_VAR,
            normalization_stats=ReferenceDatasetStats.AUDIOSET_STATS,
            use_spec_augment=True,
            freq_mask_param=27,
            time_mask_param=100
        )
        
        super().__init__(config)
        
    def extract_ast_features(self, waveform: torch.Tensor, target_length: int = 1024) -> torch.Tensor:
        """
        Extract features in AST format.
        
        Args:
            waveform: Input audio [batch, samples]
            target_length: Target time dimension (1024 for AST)
            
        Returns:
            Mel-spectrogram [batch, n_mels, target_length]
        """
        mel_spec = self.extract_mel_spectrogram(waveform)
        
        # Resize to target length
        if mel_spec.shape[-1] != target_length:
            mel_spec = F.interpolate(
                mel_spec.unsqueeze(1),  # Add channel dim
                size=(self.config.n_mels, target_length),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
            
        return mel_spec


class TimbralgebraicsPreprocessor(StandardAudioPreprocessor):
    """
    Preprocessor optimized for timbralgebraics project.
    
    Configured to work well with RAVE and BigVGAN models.
    """
    
    def __init__(self):
        # Timbralgebraics-specific configuration
        config = AudioModuleConfig(
            sample_rate=22050,  # Match RAVE
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            embedding_dim=512,  # Match RAVE latent dim
            normalization_method=NormalizationMethod.ZERO_MEAN_UNIT_VAR,
            normalization_stats=ReferenceDatasetStats.MUSIC_STATS,
            use_spec_augment=False,  # Preserve fidelity for latent operations
            content_aware=True
        )
        
        super().__init__(config)
        
    def extract_timbral_features(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features optimized for timbral analysis."""
        features = self.forward(
            waveform,
            extract_features=['mel', 'mfcc', 'chroma', 'spectral']
        )
        
        # Add timbral-specific features
        mel_spec = features['mel_spectrogram']
        
        # Temporal statistics for timbral characterization
        features['mel_mean'] = mel_spec.mean(dim=-1)
        features['mel_std'] = mel_spec.std(dim=-1)
        features['mel_skewness'] = self._compute_skewness(mel_spec)
        features['mel_kurtosis'] = self._compute_kurtosis(mel_spec)
        
        # Harmonic-percussive separation features
        features['harmonic_strength'] = self._compute_harmonic_strength(features['chroma'])
        features['percussive_strength'] = self._compute_percussive_strength(features['spectral_centroid'])
        
        return features
        
    def _compute_skewness(self, x: torch.Tensor) -> torch.Tensor:
        """Compute skewness along time dimension."""
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        normalized = (x - mean) / (std + 1e-8)
        skewness = (normalized ** 3).mean(dim=-1)
        return skewness
        
    def _compute_kurtosis(self, x: torch.Tensor) -> torch.Tensor:
        """Compute kurtosis along time dimension."""
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        normalized = (x - mean) / (std + 1e-8)
        kurtosis = (normalized ** 4).mean(dim=-1) - 3  # Excess kurtosis
        return kurtosis
        
    def _compute_harmonic_strength(self, chroma: torch.Tensor) -> torch.Tensor:
        """Compute harmonic strength from chroma features."""
        # Measure how much energy is in harmonic intervals
        harmonic_template = torch.tensor([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], 
                                       device=chroma.device, dtype=chroma.dtype)
        harmonic_template = harmonic_template.unsqueeze(0).unsqueeze(-1)
        
        harmonic_correlation = F.cosine_similarity(
            chroma, harmonic_template.expand_as(chroma), dim=1
        )
        
        return harmonic_correlation.mean(dim=-1)
        
    def _compute_percussive_strength(self, spectral_centroid: torch.Tensor) -> torch.Tensor:
        """Compute percussive strength from spectral centroid variation."""
        # High variation in spectral centroid indicates percussive content
        centroid_variation = torch.var(spectral_centroid, dim=-1)
        return centroid_variation


# Factory functions
def create_preprocessor(domain: str = "general") -> StandardAudioPreprocessor:
    """Create preprocessor for specific domain."""
    if domain == "speech":
        from .audio_config import get_speech_config
        return StandardAudioPreprocessor(get_speech_config())
    elif domain == "music":
        from .audio_config import get_music_config
        return StandardAudioPreprocessor(get_music_config())
    elif domain == "timbralgebraics":
        return TimbralgebraicsPreprocessor()
    elif domain == "ast":
        return ASTCompatiblePreprocessor()
    else:
        from .audio_config import get_general_config
        return StandardAudioPreprocessor(get_general_config())


# Example usage
if __name__ == "__main__":
    # Test different preprocessors
    general_preprocessor = create_preprocessor("general")
    ast_preprocessor = create_preprocessor("ast")
    timbral_preprocessor = create_preprocessor("timbralgebraics")
    
    # Test audio
    sample_rate = 22050
    duration = 3
    waveform = torch.randn(2, sample_rate * duration)
    
    # General preprocessing
    general_features = general_preprocessor(waveform)
    print("General features:", list(general_features.keys()))
    
    # AST preprocessing
    ast_waveform = F_audio.resample(waveform, sample_rate, 16000)
    ast_features = ast_preprocessor.extract_ast_features(ast_waveform)
    print(f"AST features shape: {ast_features.shape}")
    
    # Timbralgebraics preprocessing
    timbral_features = timbral_preprocessor.extract_timbral_features(waveform)
    print("Timbral features:", list(timbral_features.keys()))
    
    # Multi-scale preprocessing
    from .audio_config import get_music_config
    multi_scale = MultiScalePreprocessor(get_music_config())
    multi_features = multi_scale(waveform)
    print("Multi-scale features:", list(multi_features.keys()))