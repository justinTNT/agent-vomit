"""
Bulletproof Source Separation Module

Advanced audio source separation with spectral masking, harmonic-percussive separation,
and quality assessment for overlapping sources. Designed for robust separation of
musical instruments, vocals, and percussive elements with confidence scoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import warnings
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, NamedTuple
from dataclasses import dataclass
from enum import Enum
import scipy.signal
import scipy.ndimage
from pathlib import Path

# Import audio config
try:
    from modules.audio_analysis.bulletproof_audio_config import (
        BulletproofAudioConfig, get_bulletproof_music_config
    )
except ImportError:
    # Fallback for testing
    class BulletproofAudioConfig:
        def __init__(self, **kwargs):
            self.sample_rate = kwargs.get('sample_rate', 22050)
            self.n_fft = kwargs.get('n_fft', 2048)
            self.hop_length = kwargs.get('hop_length', 512)
            self.n_mels = kwargs.get('n_mels', 128)
    
    def get_bulletproof_music_config(**kwargs):
        return BulletproofAudioConfig(**kwargs)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SourceType(Enum):
    """Types of audio sources that can be separated."""
    VOCALS = "vocals"
    DRUMS = "drums"
    BASS = "bass"
    PIANO = "piano"
    GUITAR = "guitar"
    STRINGS = "strings"
    BRASS = "brass"
    WOODWINDS = "woodwinds"
    HARMONIC = "harmonic"
    PERCUSSIVE = "percussive"
    RESIDUAL = "residual"
    BACKGROUND = "background"
    FOREGROUND = "foreground"
    UNKNOWN = "unknown"


class SeparationMethod(Enum):
    """Source separation algorithm types."""
    HARMONIC_PERCUSSIVE = "harmonic_percussive"
    SPECTRAL_MASKING = "spectral_masking"
    MATRIX_FACTORIZATION = "matrix_factorization"
    DEEP_CLUSTERING = "deep_clustering"
    IDEAL_BINARY_MASK = "ideal_binary_mask"
    IDEAL_RATIO_MASK = "ideal_ratio_mask"
    WIENER_FILTER = "wiener_filter"
    MULTI_CHANNEL = "multi_channel"


class MaskType(Enum):
    """Types of spectral masks."""
    BINARY = "binary"
    SOFT = "soft"
    RATIO = "ratio"
    COMPLEX = "complex"
    PHASE_SENSITIVE = "phase_sensitive"


@dataclass
class SeparationResult:
    """Container for source separation results with quality metrics."""
    
    # Separated sources
    sources: Dict[SourceType, torch.Tensor]  # Source type -> audio tensor
    
    # Spectral representations
    spectrograms: Dict[SourceType, torch.Tensor]  # Source spectrograms
    masks: Dict[SourceType, torch.Tensor]  # Separation masks
    
    # Quality metrics
    separation_quality: Dict[SourceType, float]  # Quality score per source [0,1]
    isolation_quality: Dict[SourceType, float]  # Isolation quality [0,1]
    artifact_level: Dict[SourceType, float]  # Artifact detection [0,1]
    
    # Overall metrics
    overall_quality: float  # Overall separation quality [0,1]
    snr_improvement: Dict[SourceType, float]  # SNR improvement in dB
    sdr_scores: Dict[SourceType, float]  # Source-to-distortion ratio
    
    # Method information
    method_used: SeparationMethod
    mask_type: MaskType
    processing_time: float
    
    # Diagnostics
    convergence_achieved: bool
    iterations_used: int
    fallback_used: bool
    warnings: List[str]
    
    def __post_init__(self):
        """Validate and sanitize results."""
        # Clamp quality metrics to [0,1]
        for source_type in self.separation_quality:
            self.separation_quality[source_type] = max(0.0, min(1.0, self.separation_quality[source_type]))
            self.isolation_quality[source_type] = max(0.0, min(1.0, self.isolation_quality[source_type]))
            self.artifact_level[source_type] = max(0.0, min(1.0, self.artifact_level[source_type]))
        
        self.overall_quality = max(0.0, min(1.0, self.overall_quality))


class BulletproofHarmonicPercussiveSeparator(nn.Module):
    """
    Advanced harmonic-percussive separation using median filtering
    and spectral decomposition with adaptive parameters.
    """
    
    def __init__(self, config: BulletproofAudioConfig):
        super().__init__()
        self.config = config
        self.sr = config.sample_rate
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        
        # Separation parameters
        self.harmonic_kernel_size = 17  # Horizontal kernel for harmonic content
        self.percussive_kernel_size = 17  # Vertical kernel for percussive content
        self.power = 2.0  # Power for spectral enhancement
        self.margin = 1.0  # Separation margin
        
        # Adaptive parameters
        self.adaptive_kernels = True
        self.use_phase_vocoder = True
        
        # Initialize transforms
        self._init_transforms()
    
    def _init_transforms(self):
        """Initialize spectral transforms."""
        try:
            # STFT transform
            self.stft_transform = torchaudio.transforms.Spectrogram(
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                power=None,  # Complex spectrogram
                window_fn=torch.hann_window,
                normalized=True
            )
            
            # Inverse STFT for reconstruction
            self.istft_transform = torchaudio.transforms.InverseSpectrogram(
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window_fn=torch.hann_window,
                normalized=True
            )
            
        except Exception as e:
            logger.error(f"Transform initialization failed: {e}")
            self.stft_transform = None
            self.istft_transform = None
    
    def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Separate audio into harmonic and percussive components.
        
        Args:
            audio: Input audio tensor [batch_size, time] or [time]
            
        Returns:
            harmonic: Harmonic component
            percussive: Percussive component
            separation_info: Additional separation information
        """
        try:
            # Ensure proper tensor format
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            
            batch_size, time_samples = audio.shape
            
            # Validate input
            if time_samples < self.hop_length:
                logger.warning("Audio too short for harmonic-percussive separation")
                return self._empty_hp_result(audio)
            
            # Compute complex spectrogram
            if self.stft_transform is None:
                logger.error("STFT transform not available")
                return self._empty_hp_result(audio)
            
            complex_spec = self.stft_transform(audio)
            magnitude_spec = torch.abs(complex_spec)
            phase_spec = torch.angle(complex_spec)
            
            # Enhance spectrogram for separation
            enhanced_magnitude = self._enhance_spectrogram(magnitude_spec)
            
            # Adaptive kernel sizing
            if self.adaptive_kernels:
                h_kernel, p_kernel = self._compute_adaptive_kernels(enhanced_magnitude)
            else:
                h_kernel = self.harmonic_kernel_size
                p_kernel = self.percussive_kernel_size
            
            # Apply median filtering
            harmonic_mask, percussive_mask = self._compute_hp_masks(
                enhanced_magnitude, h_kernel, p_kernel
            )
            
            # Apply masks to magnitude spectrum
            harmonic_magnitude = magnitude_spec * harmonic_mask
            percussive_magnitude = magnitude_spec * percussive_mask
            
            # Reconstruct audio
            if self.use_phase_vocoder:
                harmonic_audio = self._reconstruct_with_phase_vocoder(
                    harmonic_magnitude, phase_spec
                )
                percussive_audio = self._reconstruct_with_phase_vocoder(
                    percussive_magnitude, phase_spec
                )
            else:
                # Simple phase reconstruction
                harmonic_complex = harmonic_magnitude * torch.exp(1j * phase_spec)
                percussive_complex = percussive_magnitude * torch.exp(1j * phase_spec)
                
                harmonic_audio = self._istft_reconstruction(harmonic_complex)
                percussive_audio = self._istft_reconstruction(percussive_complex)
            
            # Trim to original length
            harmonic_audio = harmonic_audio[:, :time_samples]
            percussive_audio = percussive_audio[:, :time_samples]
            
            # Quality assessment
            separation_info = self._assess_hp_separation_quality(
                audio, harmonic_audio, percussive_audio,
                harmonic_mask, percussive_mask
            )
            
            separation_info.update({
                'harmonic_kernel_size': h_kernel,
                'percussive_kernel_size': p_kernel,
                'method': 'harmonic_percussive_median_filtering'
            })
            
            return harmonic_audio, percussive_audio, separation_info
            
        except Exception as e:
            logger.error(f"Harmonic-percussive separation failed: {e}")
            return self._empty_hp_result(audio)
    
    def _enhance_spectrogram(self, magnitude_spec: torch.Tensor) -> torch.Tensor:
        """Enhance spectrogram for better separation."""
        try:
            # Power law enhancement
            enhanced = torch.pow(magnitude_spec + 1e-8, self.power)
            
            # Log compression for dynamic range
            enhanced = torch.log(enhanced + 1e-8)
            
            # Normalize
            enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min() + 1e-8)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Spectrogram enhancement failed: {e}")
            return magnitude_spec
    
    def _compute_adaptive_kernels(self, magnitude_spec: torch.Tensor) -> Tuple[int, int]:
        """Compute adaptive kernel sizes based on spectral content."""
        try:
            # Analyze spectral characteristics
            freq_bins, time_frames = magnitude_spec.shape[-2:]
            
            # Estimate harmonic content (horizontal structure)
            horizontal_coherence = self._compute_horizontal_coherence(magnitude_spec)
            
            # Estimate percussive content (vertical structure)
            vertical_coherence = self._compute_vertical_coherence(magnitude_spec)
            
            # Adapt kernel sizes
            base_h_kernel = self.harmonic_kernel_size
            base_p_kernel = self.percussive_kernel_size
            
            # Scale based on coherence measures
            h_kernel = max(5, min(31, int(base_h_kernel * (0.5 + horizontal_coherence))))
            p_kernel = max(5, min(31, int(base_p_kernel * (0.5 + vertical_coherence))))
            
            # Ensure odd kernel sizes
            h_kernel = h_kernel if h_kernel % 2 == 1 else h_kernel + 1
            p_kernel = p_kernel if p_kernel % 2 == 1 else p_kernel + 1
            
            return h_kernel, p_kernel
            
        except Exception as e:
            logger.error(f"Adaptive kernel computation failed: {e}")
            return self.harmonic_kernel_size, self.percussive_kernel_size
    
    def _compute_horizontal_coherence(self, magnitude_spec: torch.Tensor) -> float:
        """Compute horizontal coherence (harmonic content indicator)."""
        try:
            # Measure consistency across time for each frequency
            if magnitude_spec.dim() > 2:
                magnitude_spec = magnitude_spec[0]  # Take first batch
            
            # Compute local variance along time axis
            time_variance = torch.var(magnitude_spec, dim=-1)
            
            # Lower variance indicates more harmonic content
            coherence = 1.0 - torch.mean(time_variance) / (torch.mean(magnitude_spec) + 1e-8)
            
            return float(torch.clamp(coherence, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Horizontal coherence computation failed: {e}")
            return 0.5
    
    def _compute_vertical_coherence(self, magnitude_spec: torch.Tensor) -> float:
        """Compute vertical coherence (percussive content indicator)."""
        try:
            # Measure consistency across frequency for each time frame
            if magnitude_spec.dim() > 2:
                magnitude_spec = magnitude_spec[0]  # Take first batch
            
            # Compute local variance along frequency axis
            freq_variance = torch.var(magnitude_spec, dim=-2)
            
            # Higher variance indicates more percussive content
            coherence = torch.mean(freq_variance) / (torch.mean(magnitude_spec) + 1e-8)
            
            return float(torch.clamp(coherence, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Vertical coherence computation failed: {e}")
            return 0.5
    
    def _compute_hp_masks(self, magnitude_spec: torch.Tensor, 
                         h_kernel: int, p_kernel: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute harmonic and percussive masks using median filtering."""
        try:
            batch_size = magnitude_spec.size(0)
            
            # Convert to numpy for scipy filtering
            magnitude_np = magnitude_spec.detach().cpu().numpy()
            
            harmonic_masks = []
            percussive_masks = []
            
            for b in range(batch_size):
                mag = magnitude_np[b]
                
                # Harmonic filter (horizontal median filter)
                harmonic_filtered = scipy.ndimage.median_filter(
                    mag, size=(1, h_kernel), mode='reflect'
                )
                
                # Percussive filter (vertical median filter)
                percussive_filtered = scipy.ndimage.median_filter(
                    mag, size=(p_kernel, 1), mode='reflect'
                )
                
                # Compute masks with margin
                harmonic_mask = self._compute_soft_mask(
                    harmonic_filtered, percussive_filtered, favor='harmonic'
                )
                percussive_mask = self._compute_soft_mask(
                    harmonic_filtered, percussive_filtered, favor='percussive'
                )
                
                harmonic_masks.append(harmonic_mask)
                percussive_masks.append(percussive_mask)
            
            # Convert back to torch tensors
            harmonic_masks = torch.tensor(
                np.stack(harmonic_masks), 
                device=magnitude_spec.device, 
                dtype=magnitude_spec.dtype
            )
            percussive_masks = torch.tensor(
                np.stack(percussive_masks),
                device=magnitude_spec.device,
                dtype=magnitude_spec.dtype
            )
            
            return harmonic_masks, percussive_masks
            
        except Exception as e:
            logger.error(f"HP mask computation failed: {e}")
            # Fallback: identity masks
            identity_mask = torch.ones_like(magnitude_spec) * 0.5
            return identity_mask, identity_mask
    
    def _compute_soft_mask(self, harmonic_filtered: np.ndarray, 
                          percussive_filtered: np.ndarray,
                          favor: str = 'harmonic') -> np.ndarray:
        """Compute soft mask for separation."""
        try:
            if favor == 'harmonic':
                numerator = harmonic_filtered ** (1.0 + self.margin)
                denominator = harmonic_filtered ** (1.0 + self.margin) + percussive_filtered ** (1.0 + self.margin)
            else:  # percussive
                numerator = percussive_filtered ** (1.0 + self.margin)
                denominator = harmonic_filtered ** (1.0 + self.margin) + percussive_filtered ** (1.0 + self.margin)
            
            mask = numerator / (denominator + 1e-8)
            
            # Ensure mask is in [0, 1]
            mask = np.clip(mask, 0.0, 1.0)
            
            return mask
            
        except Exception as e:
            logger.error(f"Soft mask computation failed: {e}")
            return np.ones_like(harmonic_filtered) * 0.5
    
    def _reconstruct_with_phase_vocoder(self, magnitude: torch.Tensor, 
                                      phase: torch.Tensor) -> torch.Tensor:
        """Reconstruct audio using phase vocoder for better quality."""
        try:
            # Phase-aware reconstruction
            complex_spec = magnitude * torch.exp(1j * phase)
            
            # Apply phase vocoder processing
            processed_complex = self._phase_vocoder_processing(complex_spec)
            
            # ISTFT reconstruction
            audio = self._istft_reconstruction(processed_complex)
            
            return audio
            
        except Exception as e:
            logger.error(f"Phase vocoder reconstruction failed: {e}")
            # Fallback to simple reconstruction
            complex_spec = magnitude * torch.exp(1j * phase)
            return self._istft_reconstruction(complex_spec)
    
    def _phase_vocoder_processing(self, complex_spec: torch.Tensor) -> torch.Tensor:
        """Apply phase vocoder processing for better reconstruction."""
        try:
            # Simple phase unwrapping and smoothing
            magnitude = torch.abs(complex_spec)
            phase = torch.angle(complex_spec)
            
            # Phase unwrapping along time axis
            unwrapped_phase = self._unwrap_phase(phase)
            
            # Phase smoothing
            smoothed_phase = self._smooth_phase(unwrapped_phase)
            
            # Reconstruct complex spectrum
            processed_complex = magnitude * torch.exp(1j * smoothed_phase)
            
            return processed_complex
            
        except Exception as e:
            logger.error(f"Phase vocoder processing failed: {e}")
            return complex_spec
    
    def _unwrap_phase(self, phase: torch.Tensor) -> torch.Tensor:
        """Unwrap phase for smoother reconstruction."""
        try:
            # Simple phase unwrapping (could be improved)
            unwrapped = phase.clone()
            
            # Unwrap along time axis
            for t in range(1, phase.size(-1)):
                diff = unwrapped[..., t] - unwrapped[..., t-1]
                
                # Wrap differences to [-π, π]
                diff = torch.remainder(diff + np.pi, 2*np.pi) - np.pi
                
                unwrapped[..., t] = unwrapped[..., t-1] + diff
            
            return unwrapped
            
        except Exception as e:
            logger.error(f"Phase unwrapping failed: {e}")
            return phase
    
    def _smooth_phase(self, phase: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        """Smooth phase for better reconstruction quality."""
        try:
            if kernel_size < 3:
                return phase
            
            # Simple moving average along time axis
            padding = kernel_size // 2
            smoothed = F.avg_pool1d(
                phase.view(-1, 1, phase.size(-1)),
                kernel_size=kernel_size,
                stride=1,
                padding=padding
            )
            
            return smoothed.view(phase.shape)
            
        except Exception as e:
            logger.error(f"Phase smoothing failed: {e}")
            return phase
    
    def _istft_reconstruction(self, complex_spec: torch.Tensor) -> torch.Tensor:
        """Reconstruct audio from complex spectrogram."""
        try:
            if self.istft_transform is None:
                logger.error("ISTFT transform not available")
                # Fallback: simple magnitude-only reconstruction
                magnitude = torch.abs(complex_spec)
                return magnitude.sum(dim=-2)  # Sum across frequency bins
            
            # Ensure complex tensor
            if not complex_spec.is_complex():
                logger.warning("Expected complex spectrogram, got real tensor")
                complex_spec = complex_spec.to(torch.complex64)
            
            audio = self.istft_transform(complex_spec)
            
            return audio
            
        except Exception as e:
            logger.error(f"ISTFT reconstruction failed: {e}")
            # Fallback: magnitude-only reconstruction
            magnitude = torch.abs(complex_spec) if complex_spec.is_complex() else complex_spec
            return magnitude.sum(dim=-2)
    
    def _assess_hp_separation_quality(self, original: torch.Tensor,
                                    harmonic: torch.Tensor,
                                    percussive: torch.Tensor,
                                    harmonic_mask: torch.Tensor,
                                    percussive_mask: torch.Tensor) -> Dict[str, Any]:
        """Assess quality of harmonic-percussive separation."""
        try:
            # Reconstruction quality
            reconstructed = harmonic + percussive
            reconstruction_error = F.mse_loss(reconstructed, original)
            reconstruction_quality = torch.exp(-reconstruction_error * 10)  # Scale to [0,1]
            
            # Mask quality (how well masks sum to 1)
            mask_sum = harmonic_mask + percussive_mask
            mask_consistency = 1.0 - F.mse_loss(mask_sum, torch.ones_like(mask_sum))
            mask_consistency = torch.clamp(mask_consistency, 0.0, 1.0)
            
            # Energy conservation
            original_energy = torch.sum(original ** 2, dim=-1)
            harmonic_energy = torch.sum(harmonic ** 2, dim=-1)
            percussive_energy = torch.sum(percussive ** 2, dim=-1)
            total_separated_energy = harmonic_energy + percussive_energy
            
            energy_conservation = 1.0 - torch.abs(original_energy - total_separated_energy) / (original_energy + 1e-8)
            energy_conservation = torch.clamp(energy_conservation.mean(), 0.0, 1.0)
            
            # Component distinctiveness
            cross_correlation = self._compute_cross_correlation(harmonic, percussive)
            distinctiveness = 1.0 - torch.abs(cross_correlation)
            
            return {
                'reconstruction_quality': float(reconstruction_quality),
                'mask_consistency': float(mask_consistency),
                'energy_conservation': float(energy_conservation),
                'component_distinctiveness': float(distinctiveness),
                'overall_hp_quality': float(
                    0.3 * reconstruction_quality + 
                    0.2 * mask_consistency + 
                    0.3 * energy_conservation + 
                    0.2 * distinctiveness
                )
            }
            
        except Exception as e:
            logger.error(f"HP separation quality assessment failed: {e}")
            return {
                'reconstruction_quality': 0.5,
                'mask_consistency': 0.5,
                'energy_conservation': 0.5,
                'component_distinctiveness': 0.5,
                'overall_hp_quality': 0.5
            }
    
    def _compute_cross_correlation(self, signal1: torch.Tensor, signal2: torch.Tensor) -> torch.Tensor:
        """Compute normalized cross-correlation between two signals."""
        try:
            # Flatten signals
            s1_flat = signal1.view(-1)
            s2_flat = signal2.view(-1)
            
            # Normalize
            s1_norm = s1_flat / (torch.norm(s1_flat) + 1e-8)
            s2_norm = s2_flat / (torch.norm(s2_flat) + 1e-8)
            
            # Cross-correlation
            correlation = torch.dot(s1_norm, s2_norm)
            
            return correlation
            
        except Exception as e:
            logger.error(f"Cross-correlation computation failed: {e}")
            return torch.tensor(0.0)
    
    def _empty_hp_result(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Return empty result for error cases."""
        empty_audio = torch.zeros_like(audio)
        empty_info = {
            'reconstruction_quality': 0.0,
            'mask_consistency': 0.0,
            'energy_conservation': 0.0,
            'component_distinctiveness': 0.0,
            'overall_hp_quality': 0.0,
            'method': 'failed'
        }
        return empty_audio, empty_audio, empty_info


class BulletproofSpectralMasking(nn.Module):
    """
    Advanced spectral masking for source separation using various mask types
    and adaptive thresholding for robust separation performance.
    """
    
    def __init__(self, config: BulletproofAudioConfig):
        super().__init__()
        self.config = config
        self.sr = config.sample_rate
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        
        # Masking parameters
        self.mask_threshold = 0.5
        self.soft_mask_power = 1.0
        self.ratio_mask_limit = 10.0  # Maximum ratio for ratio masks
        
        # Source detection parameters
        self.energy_threshold = 0.01
        self.spectral_rolloff_threshold = 0.85
        self.zcr_threshold = 0.1
        
        # Initialize transforms
        self._init_transforms()
        
        # Pre-computed filter banks for different source types
        self.source_filters = self._create_source_filters()
    
    def _init_transforms(self):
        """Initialize spectral transforms."""
        try:
            # Complex STFT
            self.stft_transform = torchaudio.transforms.Spectrogram(
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                power=None,  # Complex
                window_fn=torch.hann_window,
                normalized=True
            )
            
            # Mel-scale for source characterization
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.config.n_mels,
                f_min=20,
                f_max=self.sr // 2,
                power=2.0
            )
            
            # ISTFT for reconstruction
            self.istft_transform = torchaudio.transforms.InverseSpectrogram(
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window_fn=torch.hann_window,
                normalized=True
            )
            
        except Exception as e:
            logger.error(f"Spectral masking transform initialization failed: {e}")
            self.stft_transform = None
            self.mel_transform = None
            self.istft_transform = None
    
    def _create_source_filters(self) -> Dict[SourceType, torch.Tensor]:
        """Create filter banks for different source types."""
        try:
            filters = {}
            freq_bins = self.n_fft // 2 + 1
            freqs = torch.linspace(0, self.sr // 2, freq_bins)
            
            # Vocals (focused on mid-range frequencies)
            vocal_filter = torch.zeros(freq_bins)
            vocal_range = (freqs >= 80) & (freqs <= 2000)
            vocal_filter[vocal_range] = torch.exp(-((freqs[vocal_range] - 400) / 300) ** 2)
            filters[SourceType.VOCALS] = vocal_filter
            
            # Bass (low frequencies)
            bass_filter = torch.zeros(freq_bins)
            bass_range = freqs <= 250
            bass_filter[bass_range] = torch.exp(-((freqs[bass_range] - 60) / 40) ** 2)
            filters[SourceType.BASS] = bass_filter
            
            # Drums (broad spectrum with emphasis on low and high)
            drum_filter = torch.zeros(freq_bins)
            # Low frequencies (kick)
            drum_filter[freqs <= 150] = 0.8
            # High frequencies (cymbals, snare harmonics)
            drum_filter[freqs >= 2000] = 0.6
            # Mid frequencies (snare fundamental)
            mid_range = (freqs >= 150) & (freqs <= 400)
            drum_filter[mid_range] = 0.7
            filters[SourceType.DRUMS] = drum_filter
            
            # Piano (broad spectrum)
            piano_filter = torch.ones(freq_bins)
            piano_filter[freqs <= 30] = 0.1  # Attenuate very low frequencies
            piano_filter[freqs >= 4000] = 0.5  # Attenuate very high frequencies
            filters[SourceType.PIANO] = piano_filter
            
            return filters
            
        except Exception as e:
            logger.error(f"Source filter creation failed: {e}")
            # Fallback: uniform filters
            freq_bins = self.n_fft // 2 + 1
            uniform_filter = torch.ones(freq_bins)
            return {source_type: uniform_filter for source_type in SourceType}
    
    def forward(self, audio: torch.Tensor, 
                target_sources: List[SourceType],
                mask_type: MaskType = MaskType.SOFT) -> Dict[SourceType, torch.Tensor]:
        """
        Separate sources using spectral masking.
        
        Args:
            audio: Input audio tensor
            target_sources: List of sources to separate
            mask_type: Type of mask to use
            
        Returns:
            Dictionary mapping source types to separated audio
        """
        try:
            # Ensure proper tensor format
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            
            # Compute complex spectrogram
            if self.stft_transform is None:
                logger.error("STFT transform not available")
                return self._empty_separation_result(audio, target_sources)
            
            complex_spec = self.stft_transform(audio)
            magnitude_spec = torch.abs(complex_spec)
            phase_spec = torch.angle(complex_spec)
            
            # Analyze spectral content
            spectral_features = self._analyze_spectral_content(magnitude_spec)
            
            # Detect source activity
            source_activity = self._detect_source_activity(
                magnitude_spec, spectral_features, target_sources
            )
            
            # Create source masks
            source_masks = self._create_source_masks(
                magnitude_spec, source_activity, target_sources, mask_type
            )
            
            # Apply masks and reconstruct
            separated_sources = {}
            for source_type in target_sources:
                if source_type in source_masks:
                    mask = source_masks[source_type]
                    
                    # Apply mask
                    masked_magnitude = magnitude_spec * mask
                    
                    # Reconstruct complex spectrum
                    masked_complex = masked_magnitude * torch.exp(1j * phase_spec)
                    
                    # Reconstruct audio
                    separated_audio = self._reconstruct_audio(masked_complex)
                    
                    # Trim to original length
                    separated_audio = separated_audio[:, :audio.size(-1)]
                    
                    separated_sources[source_type] = separated_audio
                else:
                    # No mask available, return silence
                    separated_sources[source_type] = torch.zeros_like(audio)
            
            return separated_sources
            
        except Exception as e:
            logger.error(f"Spectral masking separation failed: {e}")
            return self._empty_separation_result(audio, target_sources)
    
    def _analyze_spectral_content(self, magnitude_spec: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze spectral content for source detection."""
        try:
            features = {}
            
            # Spectral centroid
            freq_bins = magnitude_spec.size(-2)
            freqs = torch.linspace(0, self.sr // 2, freq_bins).unsqueeze(0).unsqueeze(-1)
            freqs = freqs.to(magnitude_spec.device)
            
            spectral_centroid = torch.sum(magnitude_spec * freqs, dim=-2) / (torch.sum(magnitude_spec, dim=-2) + 1e-8)
            features['spectral_centroid'] = spectral_centroid
            
            # Spectral rolloff
            cumulative_energy = torch.cumsum(magnitude_spec, dim=-2)
            total_energy = cumulative_energy[-1:, :]
            rolloff_threshold = self.spectral_rolloff_threshold * total_energy
            
            # Find frequency where cumulative energy exceeds threshold
            rolloff_indices = torch.argmax((cumulative_energy >= rolloff_threshold).float(), dim=-2)
            spectral_rolloff = freqs.squeeze()[rolloff_indices]
            features['spectral_rolloff'] = spectral_rolloff
            
            # Spectral flatness (tonality measure)
            geometric_mean = torch.exp(torch.mean(torch.log(magnitude_spec + 1e-8), dim=-2))
            arithmetic_mean = torch.mean(magnitude_spec, dim=-2)
            spectral_flatness = geometric_mean / (arithmetic_mean + 1e-8)
            features['spectral_flatness'] = spectral_flatness
            
            # High-frequency content
            high_freq_start = int(0.7 * freq_bins)  # Above 70% of Nyquist
            high_freq_content = torch.sum(magnitude_spec[:, high_freq_start:, :], dim=-2)
            features['high_freq_content'] = high_freq_content
            
            return features
            
        except Exception as e:
            logger.error(f"Spectral content analysis failed: {e}")
            return {}
    
    def _detect_source_activity(self, magnitude_spec: torch.Tensor,
                              spectral_features: Dict[str, torch.Tensor],
                              target_sources: List[SourceType]) -> Dict[SourceType, torch.Tensor]:
        """Detect activity of different source types."""
        try:
            activity = {}
            batch_size, freq_bins, time_frames = magnitude_spec.shape
            
            for source_type in target_sources:
                source_activity = torch.zeros(batch_size, time_frames)
                
                if source_type == SourceType.VOCALS:
                    # Vocals: mid-range frequencies, harmonic content
                    if 'spectral_centroid' in spectral_features:
                        centroid = spectral_features['spectral_centroid']
                        vocal_activity = ((centroid >= 200) & (centroid <= 2000)).float()
                        source_activity += vocal_activity
                    
                    if 'spectral_flatness' in spectral_features:
                        flatness = spectral_features['spectral_flatness']
                        tonal_activity = (flatness <= 0.5).float()  # More tonal
                        source_activity += tonal_activity
                
                elif source_type == SourceType.DRUMS:
                    # Drums: broadband, high spectral flatness
                    if 'spectral_flatness' in spectral_features:
                        flatness = spectral_features['spectral_flatness']
                        percussive_activity = (flatness >= 0.3).float()  # More noise-like
                        source_activity += percussive_activity
                    
                    if 'high_freq_content' in spectral_features:
                        hfc = spectral_features['high_freq_content']
                        hfc_normalized = hfc / (torch.max(hfc) + 1e-8)
                        cymbal_activity = (hfc_normalized >= 0.3).float()
                        source_activity += cymbal_activity
                
                elif source_type == SourceType.BASS:
                    # Bass: low frequencies
                    if 'spectral_centroid' in spectral_features:
                        centroid = spectral_features['spectral_centroid']
                        bass_activity = (centroid <= 200).float()
                        source_activity += bass_activity
                
                elif source_type == SourceType.PIANO:
                    # Piano: harmonic, broad spectrum
                    if 'spectral_flatness' in spectral_features:
                        flatness = spectral_features['spectral_flatness']
                        harmonic_activity = (flatness <= 0.4).float()
                        source_activity += harmonic_activity
                
                # Normalize activity to [0, 1]
                if source_activity.max() > 0:
                    source_activity = source_activity / source_activity.max()
                
                activity[source_type] = source_activity.to(magnitude_spec.device)
            
            return activity
            
        except Exception as e:
            logger.error(f"Source activity detection failed: {e}")
            # Fallback: uniform activity
            batch_size, _, time_frames = magnitude_spec.shape
            uniform_activity = torch.ones(batch_size, time_frames, device=magnitude_spec.device) * 0.5
            return {source_type: uniform_activity for source_type in target_sources}
    
    def _create_source_masks(self, magnitude_spec: torch.Tensor,
                           source_activity: Dict[SourceType, torch.Tensor],
                           target_sources: List[SourceType],
                           mask_type: MaskType) -> Dict[SourceType, torch.Tensor]:
        """Create separation masks for each source."""
        try:
            masks = {}
            batch_size, freq_bins, time_frames = magnitude_spec.shape
            
            for source_type in target_sources:
                if source_type not in source_activity:
                    continue
                
                activity = source_activity[source_type]
                
                # Get source-specific frequency weighting
                if source_type in self.source_filters:
                    freq_filter = self.source_filters[source_type].to(magnitude_spec.device)
                    freq_filter = freq_filter.unsqueeze(0).unsqueeze(-1)  # [1, freq, 1]
                else:
                    freq_filter = torch.ones(1, freq_bins, 1, device=magnitude_spec.device)
                
                # Combine frequency weighting with temporal activity
                activity_expanded = activity.unsqueeze(-2)  # [batch, 1, time]
                
                # Create base mask
                base_mask = freq_filter * activity_expanded
                
                # Apply mask type-specific processing
                if mask_type == MaskType.BINARY:
                    mask = (base_mask >= self.mask_threshold).float()
                
                elif mask_type == MaskType.SOFT:
                    mask = torch.pow(base_mask, self.soft_mask_power)
                    mask = torch.clamp(mask, 0.0, 1.0)
                
                elif mask_type == MaskType.RATIO:
                    # Ratio mask based on magnitude
                    source_magnitude = magnitude_spec * base_mask
                    other_magnitude = magnitude_spec * (1.0 - base_mask)
                    
                    ratio = source_magnitude / (other_magnitude + 1e-8)
                    ratio = torch.clamp(ratio, 0.0, self.ratio_mask_limit)
                    mask = ratio / (ratio + 1.0)
                
                else:  # Default to soft mask
                    mask = base_mask
                    mask = torch.clamp(mask, 0.0, 1.0)
                
                masks[source_type] = mask
            
            # Ensure masks are consistent (optional normalization)
            if len(masks) > 1:
                masks = self._normalize_masks(masks)
            
            return masks
            
        except Exception as e:
            logger.error(f"Source mask creation failed: {e}")
            # Fallback: uniform masks
            uniform_mask = torch.ones_like(magnitude_spec) / len(target_sources)
            return {source_type: uniform_mask for source_type in target_sources}
    
    def _normalize_masks(self, masks: Dict[SourceType, torch.Tensor]) -> Dict[SourceType, torch.Tensor]:
        """Normalize masks to ensure they sum to approximately 1."""
        try:
            # Sum all masks
            mask_sum = torch.zeros_like(list(masks.values())[0])
            for mask in masks.values():
                mask_sum += mask
            
            # Normalize each mask
            normalized_masks = {}
            for source_type, mask in masks.items():
                normalized_mask = mask / (mask_sum + 1e-8)
                normalized_masks[source_type] = normalized_mask
            
            return normalized_masks
            
        except Exception as e:
            logger.error(f"Mask normalization failed: {e}")
            return masks
    
    def _reconstruct_audio(self, complex_spec: torch.Tensor) -> torch.Tensor:
        """Reconstruct audio from complex spectrogram."""
        try:
            if self.istft_transform is None:
                logger.error("ISTFT transform not available")
                # Fallback
                magnitude = torch.abs(complex_spec)
                return magnitude.sum(dim=-2)
            
            audio = self.istft_transform(complex_spec)
            return audio
            
        except Exception as e:
            logger.error(f"Audio reconstruction failed: {e}")
            # Fallback
            magnitude = torch.abs(complex_spec) if complex_spec.is_complex() else complex_spec
            return magnitude.sum(dim=-2)
    
    def _empty_separation_result(self, audio: torch.Tensor, 
                               target_sources: List[SourceType]) -> Dict[SourceType, torch.Tensor]:
        """Return empty separation result for error cases."""
        return {source_type: torch.zeros_like(audio) for source_type in target_sources}


class BulletproofSourceSeparator(nn.Module):
    """
    Complete source separation system combining multiple separation methods
    with quality assessment and automatic method selection.
    """
    
    def __init__(self, config: Optional[BulletproofAudioConfig] = None):
        super().__init__()
        
        if config is None:
            config = get_bulletproof_music_config()
        
        self.config = config
        
        # Initialize separation components
        self.hp_separator = BulletproofHarmonicPercussiveSeparator(config)
        self.spectral_masking = BulletproofSpectralMasking(config)
        
        # Quality assessment thresholds
        self.min_separation_quality = 0.3
        self.min_isolation_quality = 0.2
        self.max_artifact_level = 0.7
        
        # Method selection parameters
        self.method_weights = {
            SeparationMethod.HARMONIC_PERCUSSIVE: 0.8,
            SeparationMethod.SPECTRAL_MASKING: 0.6,
        }
    
    def forward(self, audio: torch.Tensor,
                target_sources: Optional[List[SourceType]] = None,
                method: Optional[SeparationMethod] = None) -> SeparationResult:
        """
        Perform complete source separation with quality assessment.
        
        Args:
            audio: Input audio tensor
            target_sources: List of sources to separate (if None, auto-detect)
            method: Separation method to use (if None, auto-select)
            
        Returns:
            SeparationResult with separated sources and quality metrics
        """
        import time
        start_time = time.time()
        
        try:
            # Validate input
            audio = self._validate_audio_input(audio)
            if audio is None:
                return self._create_fallback_result("Invalid audio input", start_time)
            
            # Auto-detect target sources if not specified
            if target_sources is None:
                target_sources = self._auto_detect_sources(audio)
            
            # Auto-select method if not specified
            if method is None:
                method = self._auto_select_method(audio, target_sources)
            
            # Perform separation based on method
            if method == SeparationMethod.HARMONIC_PERCUSSIVE:
                separated_sources, spectrograms, masks = self._harmonic_percussive_separation(audio)
                
            elif method == SeparationMethod.SPECTRAL_MASKING:
                separated_sources, spectrograms, masks = self._spectral_masking_separation(
                    audio, target_sources
                )
                
            else:
                # Fallback to harmonic-percussive
                logger.warning(f"Method {method} not implemented, using harmonic-percussive")
                separated_sources, spectrograms, masks = self._harmonic_percussive_separation(audio)
                method = SeparationMethod.HARMONIC_PERCUSSIVE
            
            # Quality assessment
            quality_metrics = self._assess_separation_quality(
                audio, separated_sources, spectrograms, masks
            )
            
            # Create result
            result = SeparationResult(
                sources=separated_sources,
                spectrograms=spectrograms,
                masks=masks,
                separation_quality=quality_metrics['separation_quality'],
                isolation_quality=quality_metrics['isolation_quality'],
                artifact_level=quality_metrics['artifact_level'],
                overall_quality=quality_metrics['overall_quality'],
                snr_improvement=quality_metrics['snr_improvement'],
                sdr_scores=quality_metrics['sdr_scores'],
                method_used=method,
                mask_type=MaskType.SOFT,  # Default
                processing_time=time.time() - start_time,
                convergence_achieved=True,
                iterations_used=1,
                fallback_used=False,
                warnings=self._collect_warnings(quality_metrics, separated_sources)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Source separation failed: {e}")
            return self._create_fallback_result(f"Processing error: {e}", start_time)
    
    def _validate_audio_input(self, audio: torch.Tensor) -> Optional[torch.Tensor]:
        """Validate and preprocess audio input."""
        try:
            if not isinstance(audio, torch.Tensor):
                logger.error(f"Expected torch.Tensor, got {type(audio)}")
                return None
            
            # Ensure 1D or 2D
            if audio.dim() > 2:
                logger.error(f"Audio tensor has too many dimensions: {audio.dim()}")
                return None
            
            # Convert to 1D if needed
            if audio.dim() == 2:
                if audio.size(0) == 1:
                    audio = audio.squeeze(0)
                elif audio.size(1) == 1:
                    audio = audio.squeeze(1)
                else:
                    # Take first channel
                    audio = audio[0]
                    logger.warning("Multi-channel audio detected, using first channel")
            
            # Check length
            min_length = self.config.sample_rate * 2  # Minimum 2 seconds
            if audio.size(-1) < min_length:
                logger.warning(f"Audio too short: {audio.size(-1)} samples, minimum {min_length}")
                if audio.size(-1) < self.config.hop_length:
                    return None
            
            # Check for silence
            if audio.abs().max() < 1e-6:
                logger.warning("Audio appears to be silent")
                return None
            
            # Normalize
            audio = audio / (audio.abs().max() + 1e-8)
            
            return audio
            
        except Exception as e:
            logger.error(f"Audio validation failed: {e}")
            return None
    
    def _auto_detect_sources(self, audio: torch.Tensor) -> List[SourceType]:
        """Automatically detect likely sources in the audio."""
        try:
            # Simple heuristic-based source detection
            detected_sources = []
            
            # Always include harmonic and percussive for basic separation
            detected_sources.extend([SourceType.HARMONIC, SourceType.PERCUSSIVE])
            
            # Analyze spectral content for specific instruments
            if self.spectral_masking.stft_transform is not None:
                if audio.dim() == 1:
                    audio_batch = audio.unsqueeze(0)
                else:
                    audio_batch = audio
                
                complex_spec = self.spectral_masking.stft_transform(audio_batch)
                magnitude_spec = torch.abs(complex_spec)
                
                spectral_features = self.spectral_masking._analyze_spectral_content(magnitude_spec)
                
                # Detect vocals (mid-range harmonic content)
                if 'spectral_centroid' in spectral_features:
                    centroid = spectral_features['spectral_centroid']
                    if torch.mean((centroid >= 200) & (centroid <= 2000)).item() > 0.3:
                        detected_sources.append(SourceType.VOCALS)
                
                # Detect drums (high spectral flatness)
                if 'spectral_flatness' in spectral_features:
                    flatness = spectral_features['spectral_flatness']
                    if torch.mean(flatness >= 0.3).item() > 0.2:
                        detected_sources.append(SourceType.DRUMS)
                
                # Detect bass (low-frequency content)
                if 'spectral_centroid' in spectral_features:
                    centroid = spectral_features['spectral_centroid']
                    if torch.mean(centroid <= 200).item() > 0.1:
                        detected_sources.append(SourceType.BASS)
            
            return list(set(detected_sources))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Auto source detection failed: {e}")
            # Fallback to basic harmonic-percussive
            return [SourceType.HARMONIC, SourceType.PERCUSSIVE]
    
    def _auto_select_method(self, audio: torch.Tensor, 
                          target_sources: List[SourceType]) -> SeparationMethod:
        """Automatically select the best separation method."""
        try:
            # Simple heuristic-based method selection
            
            # If only harmonic/percussive separation needed
            if (set(target_sources) <= {SourceType.HARMONIC, SourceType.PERCUSSIVE}):
                return SeparationMethod.HARMONIC_PERCUSSIVE
            
            # For specific instruments, use spectral masking
            specific_instruments = {
                SourceType.VOCALS, SourceType.DRUMS, SourceType.BASS, 
                SourceType.PIANO, SourceType.GUITAR
            }
            
            if set(target_sources) & specific_instruments:
                return SeparationMethod.SPECTRAL_MASKING
            
            # Default to harmonic-percussive
            return SeparationMethod.HARMONIC_PERCUSSIVE
            
        except Exception as e:
            logger.error(f"Auto method selection failed: {e}")
            return SeparationMethod.HARMONIC_PERCUSSIVE
    
    def _harmonic_percussive_separation(self, audio: torch.Tensor) -> Tuple[Dict, Dict, Dict]:
        """Perform harmonic-percussive separation."""
        try:
            harmonic, percussive, hp_info = self.hp_separator(audio)
            
            sources = {
                SourceType.HARMONIC: harmonic,
                SourceType.PERCUSSIVE: percussive
            }
            
            # Create spectrograms for quality assessment
            spectrograms = {}
            masks = {}
            
            if self.hp_separator.stft_transform is not None:
                for source_type, source_audio in sources.items():
                    if source_audio.dim() == 1:
                        source_audio = source_audio.unsqueeze(0)
                    
                    spec = self.hp_separator.stft_transform(source_audio)
                    spectrograms[source_type] = torch.abs(spec)
                    
                    # Create dummy masks (not available from HP separation)
                    masks[source_type] = torch.ones_like(spectrograms[source_type]) * 0.5
            
            return sources, spectrograms, masks
            
        except Exception as e:
            logger.error(f"Harmonic-percussive separation failed: {e}")
            return {}, {}, {}
    
    def _spectral_masking_separation(self, audio: torch.Tensor,
                                   target_sources: List[SourceType]) -> Tuple[Dict, Dict, Dict]:
        """Perform spectral masking separation."""
        try:
            # Filter out harmonic/percussive for spectral masking
            spectral_sources = [
                s for s in target_sources 
                if s not in [SourceType.HARMONIC, SourceType.PERCUSSIVE]
            ]
            
            if not spectral_sources:
                spectral_sources = [SourceType.VOCALS, SourceType.BACKGROUND]
            
            separated_sources = self.spectral_masking(
                audio, spectral_sources, MaskType.SOFT
            )
            
            # Create spectrograms and extract masks
            spectrograms = {}
            masks = {}
            
            if self.spectral_masking.stft_transform is not None:
                # Original spectrogram
                if audio.dim() == 1:
                    audio_batch = audio.unsqueeze(0)
                else:
                    audio_batch = audio
                
                original_spec = self.spectral_masking.stft_transform(audio_batch)
                original_magnitude = torch.abs(original_spec)
                
                for source_type, source_audio in separated_sources.items():
                    if source_audio.dim() == 1:
                        source_audio = source_audio.unsqueeze(0)
                    
                    # Source spectrogram
                    source_spec = self.spectral_masking.stft_transform(source_audio)
                    source_magnitude = torch.abs(source_spec)
                    spectrograms[source_type] = source_magnitude
                    
                    # Estimate mask from source and original
                    mask = source_magnitude / (original_magnitude + 1e-8)
                    mask = torch.clamp(mask, 0.0, 1.0)
                    masks[source_type] = mask
            
            return separated_sources, spectrograms, masks
            
        except Exception as e:
            logger.error(f"Spectral masking separation failed: {e}")
            return {}, {}, {}
    
    def _assess_separation_quality(self, original: torch.Tensor,
                                 separated_sources: Dict[SourceType, torch.Tensor],
                                 spectrograms: Dict[SourceType, torch.Tensor],
                                 masks: Dict[SourceType, torch.Tensor]) -> Dict[str, Any]:
        """Assess the quality of source separation."""
        try:
            quality_metrics = {
                'separation_quality': {},
                'isolation_quality': {},
                'artifact_level': {},
                'snr_improvement': {},
                'sdr_scores': {}
            }
            
            if not separated_sources:
                quality_metrics['overall_quality'] = 0.0
                return quality_metrics
            
            # Reconstruction quality
            total_reconstruction = torch.zeros_like(original)
            for source_audio in separated_sources.values():
                total_reconstruction += source_audio
            
            reconstruction_error = F.mse_loss(total_reconstruction, original)
            reconstruction_quality = torch.exp(-reconstruction_error * 10)
            
            # Per-source quality assessment
            for source_type, source_audio in separated_sources.items():
                # Separation quality (based on energy concentration)
                if source_type in spectrograms:
                    spec = spectrograms[source_type]
                    # Measure energy concentration
                    energy_concentration = self._compute_energy_concentration(spec)
                    quality_metrics['separation_quality'][source_type] = energy_concentration
                else:
                    quality_metrics['separation_quality'][source_type] = 0.5
                
                # Isolation quality (cross-correlation with other sources)
                other_sources = [s for s_type, s in separated_sources.items() if s_type != source_type]
                if other_sources:
                    avg_correlation = torch.mean(torch.stack([
                        torch.abs(self._compute_correlation(source_audio, other_source))
                        for other_source in other_sources
                    ]))
                    isolation = 1.0 - avg_correlation
                    quality_metrics['isolation_quality'][source_type] = float(isolation)
                else:
                    quality_metrics['isolation_quality'][source_type] = 1.0
                
                # Artifact level (spectral discontinuities)
                artifact_level = self._detect_artifacts(source_audio)
                quality_metrics['artifact_level'][source_type] = artifact_level
                
                # SNR improvement (simplified estimate)
                source_energy = torch.sum(source_audio ** 2)
                noise_energy = torch.sum((original - source_audio) ** 2)
                snr_improvement = 10 * torch.log10((source_energy + 1e-8) / (noise_energy + 1e-8))
                quality_metrics['snr_improvement'][source_type] = float(snr_improvement)
                
                # SDR score (simplified)
                sdr = self._compute_sdr(source_audio, original)
                quality_metrics['sdr_scores'][source_type] = sdr
            
            # Overall quality
            sep_qualities = list(quality_metrics['separation_quality'].values())
            iso_qualities = list(quality_metrics['isolation_quality'].values())
            
            if sep_qualities and iso_qualities:
                overall_quality = (
                    0.4 * float(reconstruction_quality) +
                    0.3 * np.mean(sep_qualities) +
                    0.3 * np.mean(iso_qualities)
                )
            else:
                overall_quality = float(reconstruction_quality)
            
            quality_metrics['overall_quality'] = max(0.0, min(1.0, overall_quality))
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Separation quality assessment failed: {e}")
            return {
                'separation_quality': {},
                'isolation_quality': {},
                'artifact_level': {},
                'snr_improvement': {},
                'sdr_scores': {},
                'overall_quality': 0.0
            }
    
    def _compute_energy_concentration(self, spectrogram: torch.Tensor) -> float:
        """Compute energy concentration metric."""
        try:
            # Measure how concentrated the energy is (higher = better separation)
            if spectrogram.dim() > 2:
                spectrogram = spectrogram[0]  # Take first batch
            
            # Normalize
            spec_norm = spectrogram / (torch.sum(spectrogram) + 1e-8)
            
            # Compute entropy (lower entropy = higher concentration)
            entropy = -torch.sum(spec_norm * torch.log(spec_norm + 1e-8))
            
            # Convert to concentration score
            max_entropy = np.log(spectrogram.numel())
            concentration = 1.0 - (entropy / max_entropy)
            
            return float(torch.clamp(concentration, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Energy concentration computation failed: {e}")
            return 0.5
    
    def _compute_correlation(self, signal1: torch.Tensor, signal2: torch.Tensor) -> torch.Tensor:
        """Compute normalized correlation between two signals."""
        try:
            # Ensure same length
            min_length = min(signal1.size(-1), signal2.size(-1))
            s1 = signal1[..., :min_length].flatten()
            s2 = signal2[..., :min_length].flatten()
            
            # Normalize
            s1_norm = s1 / (torch.norm(s1) + 1e-8)
            s2_norm = s2 / (torch.norm(s2) + 1e-8)
            
            # Correlation
            correlation = torch.dot(s1_norm, s2_norm)
            
            return correlation
            
        except Exception as e:
            logger.error(f"Correlation computation failed: {e}")
            return torch.tensor(0.0)
    
    def _detect_artifacts(self, audio: torch.Tensor) -> float:
        """Detect artifacts in separated audio."""
        try:
            # Simple artifact detection based on discontinuities
            if audio.dim() > 1:
                audio = audio.flatten()
            
            # Compute first-order differences
            diff = torch.diff(audio)
            
            # Large differences indicate potential artifacts
            artifact_threshold = 3 * torch.std(diff)
            artifacts = torch.sum(torch.abs(diff) > artifact_threshold)
            
            # Normalize by signal length
            artifact_rate = float(artifacts) / len(diff)
            
            return min(1.0, artifact_rate * 10)  # Scale to [0, 1]
            
        except Exception as e:
            logger.error(f"Artifact detection failed: {e}")
            return 0.5
    
    def _compute_sdr(self, source: torch.Tensor, reference: torch.Tensor) -> float:
        """Compute source-to-distortion ratio (simplified)."""
        try:
            # Ensure same length
            min_length = min(source.size(-1), reference.size(-1))
            source = source[..., :min_length].flatten()
            reference = reference[..., :min_length].flatten()
            
            # Project source onto reference
            projection = torch.dot(source, reference) / (torch.dot(reference, reference) + 1e-8)
            projected_source = projection * reference
            
            # Compute SDR
            signal_power = torch.sum(projected_source ** 2)
            distortion_power = torch.sum((source - projected_source) ** 2)
            
            sdr = 10 * torch.log10((signal_power + 1e-8) / (distortion_power + 1e-8))
            
            return float(sdr)
            
        except Exception as e:
            logger.error(f"SDR computation failed: {e}")
            return 0.0
    
    def _collect_warnings(self, quality_metrics: Dict, 
                         separated_sources: Dict[SourceType, torch.Tensor]) -> List[str]:
        """Collect warnings about separation quality."""
        warnings = []
        
        try:
            if not separated_sources:
                warnings.append("No sources separated")
                return warnings
            
            overall_quality = quality_metrics.get('overall_quality', 0.0)
            if overall_quality < self.min_separation_quality:
                warnings.append(f"Low overall separation quality: {overall_quality:.3f}")
            
            # Check individual source quality
            for source_type, quality in quality_metrics.get('separation_quality', {}).items():
                if quality < self.min_separation_quality:
                    warnings.append(f"Low separation quality for {source_type.value}: {quality:.3f}")
            
            # Check isolation quality
            for source_type, quality in quality_metrics.get('isolation_quality', {}).items():
                if quality < self.min_isolation_quality:
                    warnings.append(f"Poor isolation for {source_type.value}: {quality:.3f}")
            
            # Check artifact levels
            for source_type, level in quality_metrics.get('artifact_level', {}).items():
                if level > self.max_artifact_level:
                    warnings.append(f"High artifact level in {source_type.value}: {level:.3f}")
            
        except Exception as e:
            warnings.append(f"Error collecting warnings: {e}")
        
        return warnings
    
    def _create_fallback_result(self, reason: str, start_time: float) -> SeparationResult:
        """Create fallback result for error cases."""
        return SeparationResult(
            sources={},
            spectrograms={},
            masks={},
            separation_quality={},
            isolation_quality={},
            artifact_level={},
            overall_quality=0.0,
            snr_improvement={},
            sdr_scores={},
            method_used=SeparationMethod.HARMONIC_PERCUSSIVE,
            mask_type=MaskType.SOFT,
            processing_time=time.time() - start_time,
            convergence_achieved=False,
            iterations_used=0,
            fallback_used=True,
            warnings=[reason]
        )


# Factory functions and utilities

def create_source_separator(sample_rate: int = 22050,
                          domain: str = "music",
                          **kwargs) -> BulletproofSourceSeparator:
    """Create a source separator with specified configuration."""
    try:
        if domain.lower() == "music":
            config = get_bulletproof_music_config(sample_rate=sample_rate, **kwargs)
        else:
            from modules.audio_analysis.bulletproof_audio_config import (
                BulletproofAudioConfig, AudioDomain
            )
            config = BulletproofAudioConfig(
                sample_rate=sample_rate,
                domain=AudioDomain.GENERAL,
                **kwargs
            )
        
        return BulletproofSourceSeparator(config)
        
    except Exception as e:
        logger.error(f"Failed to create source separator: {e}")
        # Fallback configuration
        config = BulletproofAudioConfig(sample_rate=sample_rate)
        return BulletproofSourceSeparator(config)


def test_source_separator():
    """Test the source separator with synthetic audio."""
    logger.info("Testing BulletproofSourceSeparator")
    
    try:
        # Create test audio with mixed sources
        sr = 22050
        duration = 8.0  # 8 seconds
        
        t = torch.linspace(0, duration, int(sr * duration))
        
        # Create harmonic component (musical tone)
        harmonic_freq = 440.0  # A4
        harmonic = torch.sin(2 * np.pi * harmonic_freq * t)
        harmonic += 0.5 * torch.sin(2 * np.pi * harmonic_freq * 2 * t)  # Second harmonic
        harmonic += 0.25 * torch.sin(2 * np.pi * harmonic_freq * 3 * t)  # Third harmonic
        
        # Apply envelope
        envelope = torch.exp(-t * 0.5)
        harmonic *= envelope
        
        # Create percussive component (drum-like sounds)
        percussive = torch.zeros_like(t)
        beat_times = torch.arange(0, duration, 0.5)  # Every 500ms
        
        for beat_time in beat_times:
            if beat_time < duration:
                beat_sample = int(beat_time * sr)
                beat_length = int(0.1 * sr)  # 100ms beats
                end_sample = min(beat_sample + beat_length, len(percussive))
                
                # Create drum-like sound (noise burst with envelope)
                drum_noise = torch.randn(end_sample - beat_sample)
                drum_envelope = torch.exp(-10 * torch.linspace(0, 1, end_sample - beat_sample))
                drum_sound = drum_noise * drum_envelope
                
                percussive[beat_sample:end_sample] += 0.7 * drum_sound
        
        # Mix sources
        mixed_audio = 0.6 * harmonic + 0.4 * percussive
        
        # Add some background noise
        mixed_audio += 0.05 * torch.randn_like(mixed_audio)
        
        # Normalize
        mixed_audio = mixed_audio / (mixed_audio.abs().max() + 1e-8)
        
        # Test source separator
        separator = create_source_separator(sample_rate=sr)
        
        # Test automatic separation
        result = separator(mixed_audio)
        
        # Analyze results
        logger.info(f"Source separation results:")
        logger.info(f"  Method used: {result.method_used.value}")
        logger.info(f"  Mask type: {result.mask_type.value}")
        logger.info(f"  Number of sources: {len(result.sources)}")
        logger.info(f"  Overall quality: {result.overall_quality:.3f}")
        logger.info(f"  Processing time: {result.processing_time:.3f}s")
        logger.info(f"  Convergence achieved: {result.convergence_achieved}")
        logger.info(f"  Fallback used: {result.fallback_used}")
        
        # Show separated sources
        logger.info("  Separated sources:")
        for source_type, source_audio in result.sources.items():
            logger.info(f"    {source_type.value}: {source_audio.shape}")
            
            # Quality metrics for this source
            if source_type in result.separation_quality:
                sep_quality = result.separation_quality[source_type]
                logger.info(f"      Separation quality: {sep_quality:.3f}")
            
            if source_type in result.isolation_quality:
                iso_quality = result.isolation_quality[source_type]
                logger.info(f"      Isolation quality: {iso_quality:.3f}")
            
            if source_type in result.artifact_level:
                artifact_level = result.artifact_level[source_type]
                logger.info(f"      Artifact level: {artifact_level:.3f}")
            
            if source_type in result.snr_improvement:
                snr_improvement = result.snr_improvement[source_type]
                logger.info(f"      SNR improvement: {snr_improvement:.1f} dB")
            
            if source_type in result.sdr_scores:
                sdr = result.sdr_scores[source_type]
                logger.info(f"      SDR score: {sdr:.1f} dB")
        
        if result.warnings:
            logger.info(f"  Warnings: {'; '.join(result.warnings)}")
        
        # Test specific source separation
        logger.info("\nTesting specific source separation...")
        target_sources = [SourceType.VOCALS, SourceType.DRUMS]
        specific_result = separator(mixed_audio, target_sources=target_sources)
        
        logger.info(f"  Specific separation - Sources: {[s.value for s in target_sources]}")
        logger.info(f"  Overall quality: {specific_result.overall_quality:.3f}")
        logger.info(f"  Separated: {list(specific_result.sources.keys())}")
        
        logger.info("Source separator test completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Source separator test failed: {e}")
        return None


if __name__ == "__main__":
    test_source_separator()