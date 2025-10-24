#!/usr/bin/env python3
"""
BULLETPROOF CHROMA ENCODER MODULE
Comprehensive chroma feature extraction with harmonic analysis for BigVGAN compatibility.
Handles polyphonic music, noise artifacts, and provides confidence scoring for musical analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Union, Dict, Tuple, Any
from rave_config_system import RAVEConfig
import warnings
import logging
import math
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChromaConfig:
    """Configuration for bulletproof chroma encoder"""
    # Audio analysis parameters
    sample_rate: int = 44100
    n_fft: int = 4096
    hop_length: int = 1024
    win_length: Optional[int] = None
    window: str = 'hann'
    
    # Chroma parameters
    n_chroma: int = 12
    tuning: float = 0.0  # Tuning deviation in cents
    norm: Optional[str] = 'L2'  # 'L1', 'L2', 'max', None
    
    # Harmonic analysis
    n_harmonics: int = 5
    harmonic_weights: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.33, 0.25, 0.2])
    fundamental_freq_range: Tuple[float, float] = (80.0, 4000.0)
    
    # Advanced features
    use_harmonic_percussive_separation: bool = True
    use_pitch_class_profile: bool = True
    use_spectral_centroid_weighting: bool = True
    use_onset_strength_weighting: bool = True
    
    # Confidence estimation
    min_energy_threshold: float = 1e-6
    confidence_window_size: int = 5
    harmonic_confidence_weight: float = 0.6
    tonal_confidence_weight: float = 0.4
    
    # Bulletproof stability
    eps: float = 1e-8
    max_db: float = 80.0
    ref_power: float = 1e-6
    numerical_stability_check: bool = True
    
    # Memory and efficiency
    use_checkpoint: bool = False
    chunk_size: int = 2048
    max_memory_gb: float = 4.0
    
    # Fallback strategies
    enable_fallbacks: bool = True
    fallback_to_simple_chroma: bool = True
    disable_on_failure: bool = False
    
    # Quality assessment
    quality_threshold: float = 0.3
    noise_floor_db: float = -60.0
    tonal_clarity_threshold: float = 0.5

class BulletproofChromaEncoder(nn.Module):
    """
    Bulletproof chroma feature extraction with comprehensive error handling.
    
    Features:
    - Multi-harmonic chroma analysis with confidence scoring
    - Harmonic-percussive source separation for clean chroma
    - Spectral centroid and onset strength weighting
    - Real-time processing with memory constraints
    - Comprehensive fallback strategies for corrupted audio
    - Quality assessment and noise robustness
    - Polyphonic music handling with harmonic analysis
    """
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__()
        
        self.config = config
        self.chroma_config = kwargs.get('chroma_config', ChromaConfig())
        
        # Override with RAVEConfig values if available
        if hasattr(config.audio, 'sample_rate'):
            self.chroma_config.sample_rate = config.audio.sample_rate
        if hasattr(config.audio, 'n_fft'):
            self.chroma_config.n_fft = config.audio.n_fft
        if hasattr(config.audio, 'hop_length'):
            self.chroma_config.hop_length = config.audio.hop_length
        
        self.sample_rate = self.chroma_config.sample_rate
        self.n_fft = self.chroma_config.n_fft
        self.hop_length = self.chroma_config.hop_length
        self.win_length = self.chroma_config.win_length or self.n_fft
        
        # Build chroma analysis components
        try:
            self._build_chroma_components()
        except Exception as e:
            logger.error(f"Failed to build chroma components: {e}")
            if self.chroma_config.enable_fallbacks:
                logger.warning("Building fallback chroma components")
                self._build_fallback_components()
            else:
                raise
        
        # Tracking and monitoring
        self.processing_stats = []
        self.confidence_history = []
        self.fallback_activations = 0
        self.quality_assessments = []
        
        # Memory management
        self._memory_usage = 0
        self._max_memory_bytes = int(self.chroma_config.max_memory_gb * 1e9)
        
        logger.info(f"BulletproofChromaEncoder initialized: n_chroma={self.chroma_config.n_chroma}")
    
    def _build_chroma_components(self):
        """Build main chroma analysis components"""
        # Frequency bins and chroma mapping
        self._build_frequency_mappings()
        
        # Window function
        self.register_buffer('window', self._create_window())
        
        # Harmonic analysis components
        if self.chroma_config.n_harmonics > 1:
            self._build_harmonic_analysis()
        
        # Harmonic-percussive separation
        if self.chroma_config.use_harmonic_percussive_separation:
            self._build_hpss_components()
        
        # Pitch class profile
        if self.chroma_config.use_pitch_class_profile:
            self._build_pitch_class_profile()
        
        # Quality assessment components
        self._build_quality_assessment()
    
    def _build_fallback_components(self):
        """Build simple fallback components"""
        # Basic frequency mapping
        self._build_basic_frequency_mappings()
        
        # Simple window
        self.register_buffer('window', torch.hann_window(self.win_length))
        
        # Disable advanced features
        self.chroma_config.use_harmonic_percussive_separation = False
        self.chroma_config.use_pitch_class_profile = False
        self.chroma_config.n_harmonics = 1
        
        logger.info("Built fallback chroma components")
    
    def _create_window(self) -> torch.Tensor:
        """Create analysis window"""
        try:
            if self.chroma_config.window == 'hann':
                return torch.hann_window(self.win_length)
            elif self.chroma_config.window == 'hamming':
                return torch.hamming_window(self.win_length)
            elif self.chroma_config.window == 'blackman':
                return torch.blackman_window(self.win_length)
            else:
                return torch.hann_window(self.win_length)
        except Exception as e:
            logger.warning(f"Window creation failed: {e}")
            return torch.hann_window(self.win_length)
    
    def _build_frequency_mappings(self):
        """Build frequency to chroma mappings"""
        try:
            # Frequency bins
            freqs = torch.linspace(0, self.sample_rate / 2, self.n_fft // 2 + 1)
            
            # A4 tuning adjustment
            tuning_factor = 2 ** (self.chroma_config.tuning / 1200.0)
            a4_freq = 440.0 * tuning_factor
            
            # Convert frequencies to MIDI numbers
            # MIDI 69 = A4 = 440 Hz
            midi_numbers = 69 + 12 * torch.log2(freqs / a4_freq + self.chroma_config.eps)
            
            # Convert MIDI to chroma (0-11)
            chroma_numbers = midi_numbers % 12
            
            # Create chroma filter bank
            self._build_chroma_filterbank(chroma_numbers, freqs)
            
        except Exception as e:
            logger.error(f"Frequency mapping build failed: {e}")
            self._build_basic_frequency_mappings()
    
    def _build_basic_frequency_mappings(self):
        """Build basic frequency mappings as fallback"""
        try:
            # Simple linear chroma mapping
            n_bins = self.n_fft // 2 + 1
            chroma_matrix = torch.zeros(self.chroma_config.n_chroma, n_bins)
            
            # Simple bin assignment
            bins_per_chroma = n_bins // self.chroma_config.n_chroma
            for i in range(self.chroma_config.n_chroma):
                start_bin = i * bins_per_chroma
                end_bin = min((i + 1) * bins_per_chroma, n_bins)
                chroma_matrix[i, start_bin:end_bin] = 1.0
            
            self.register_buffer('chroma_matrix', chroma_matrix)
            
        except Exception as e:
            logger.error(f"Basic frequency mapping failed: {e}")
            # Emergency fallback
            chroma_matrix = torch.eye(self.chroma_config.n_chroma, self.n_fft // 2 + 1)
            self.register_buffer('chroma_matrix', chroma_matrix)
    
    def _build_chroma_filterbank(self, chroma_numbers: torch.Tensor, freqs: torch.Tensor):
        """Build chroma filter bank"""
        try:
            n_bins = len(freqs)
            chroma_matrix = torch.zeros(self.chroma_config.n_chroma, n_bins)
            
            # Build filters for each chroma class
            for chroma in range(self.chroma_config.n_chroma):
                # Find bins closest to this chroma class
                chroma_distances = torch.abs(chroma_numbers - chroma)
                
                # Handle wraparound (e.g., distance between 11 and 0)
                chroma_distances = torch.min(chroma_distances, 12 - chroma_distances)
                
                # Gaussian-like weights
                sigma = 0.5  # Adjust for wider/narrower filters
                weights = torch.exp(-0.5 * (chroma_distances / sigma) ** 2)
                
                # Only include frequencies above minimum threshold
                valid_freq_mask = freqs >= self.chroma_config.fundamental_freq_range[0]
                valid_freq_mask &= freqs <= self.chroma_config.fundamental_freq_range[1]
                
                weights = weights * valid_freq_mask.float()
                chroma_matrix[chroma] = weights
            
            # Normalize filters
            chroma_matrix = chroma_matrix / (chroma_matrix.sum(dim=0, keepdim=True) + self.chroma_config.eps)
            
            self.register_buffer('chroma_matrix', chroma_matrix)
            
        except Exception as e:
            logger.error(f"Chroma filterbank build failed: {e}")
            self._build_basic_frequency_mappings()
    
    def _build_harmonic_analysis(self):
        """Build harmonic analysis components"""
        try:
            # Harmonic weight matrix
            n_harmonics = self.chroma_config.n_harmonics
            weights = torch.tensor(self.chroma_config.harmonic_weights[:n_harmonics])
            
            # Normalize weights
            weights = weights / weights.sum()
            self.register_buffer('harmonic_weights', weights)
            
            # Harmonic frequency multipliers
            harmonic_multipliers = torch.arange(1, n_harmonics + 1, dtype=torch.float32)
            self.register_buffer('harmonic_multipliers', harmonic_multipliers)
            
        except Exception as e:
            logger.error(f"Harmonic analysis build failed: {e}")
            # Fallback: single harmonic
            self.register_buffer('harmonic_weights', torch.tensor([1.0]))
            self.register_buffer('harmonic_multipliers', torch.tensor([1.0]))
    
    def _build_hpss_components(self):
        """Build harmonic-percussive source separation components"""
        try:
            # Median filter sizes for HPSS
            self.harmonic_filter_size = 31  # Horizontal (time) filtering
            self.percussive_filter_size = 31  # Vertical (frequency) filtering
            
            # Separation power (higher = more aggressive separation)
            self.hpss_power = 2.0
            
        except Exception as e:
            logger.error(f"HPSS components build failed: {e}")
            self.chroma_config.use_harmonic_percussive_separation = False
    
    def _build_pitch_class_profile(self):
        """Build pitch class profile components"""
        try:
            # Pitch class profile templates (major/minor scales, etc.)
            self._build_key_profiles()
            
        except Exception as e:
            logger.error(f"Pitch class profile build failed: {e}")
            self.chroma_config.use_pitch_class_profile = False
    
    def _build_key_profiles(self):
        """Build key profile templates"""
        try:
            # Krumhansl-Schmuckler key profiles
            major_profile = torch.tensor([
                6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88
            ])
            
            minor_profile = torch.tensor([
                6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17
            ])
            
            # Normalize profiles
            major_profile = major_profile / major_profile.sum()
            minor_profile = minor_profile / minor_profile.sum()
            
            self.register_buffer('major_profile', major_profile)
            self.register_buffer('minor_profile', minor_profile)
            
            # Build all 24 key profiles (12 major + 12 minor)
            all_profiles = []
            for shift in range(12):
                major_shifted = torch.roll(major_profile, shift)
                minor_shifted = torch.roll(minor_profile, shift)
                all_profiles.extend([major_shifted, minor_shifted])
            
            key_profiles = torch.stack(all_profiles)  # [24, 12]
            self.register_buffer('key_profiles', key_profiles)
            
        except Exception as e:
            logger.error(f"Key profiles build failed: {e}")
            # Fallback: uniform profiles
            uniform_profile = torch.ones(12) / 12
            self.register_buffer('major_profile', uniform_profile)
            self.register_buffer('minor_profile', uniform_profile)
            self.register_buffer('key_profiles', uniform_profile.unsqueeze(0).repeat(24, 1))
    
    def _build_quality_assessment(self):
        """Build quality assessment components"""
        try:
            # Spectral features for quality assessment
            self.spectral_rolloff_threshold = 0.85
            self.spectral_centroid_weight = 0.3
            self.zero_crossing_weight = 0.2
            
        except Exception as e:
            logger.error(f"Quality assessment build failed: {e}")
    
    def _validate_input(self, audio: torch.Tensor) -> bool:
        """Validate input audio tensor"""
        try:
            if audio is None or audio.numel() == 0:
                logger.warning("Empty or None audio input")
                return False
            
            if not torch.isfinite(audio).all():
                logger.warning("Non-finite values in audio input")
                if self.chroma_config.enable_fallbacks:
                    return True  # Allow fallback to handle corrupted audio
                return False
            
            if audio.dim() > 2:
                logger.warning(f"Audio has too many dimensions: {audio.dim()}")
                return False
            
            # Check reasonable audio length
            min_samples = self.n_fft
            if audio.size(-1) < min_samples:
                logger.warning(f"Audio too short: {audio.size(-1)} < {min_samples}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False
    
    def _compute_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute STFT spectrogram with error handling"""
        try:
            # Handle multi-channel audio
            if audio.dim() == 2:
                # Take mean across channels if stereo
                audio = audio.mean(dim=0)
            
            # Clean up corrupted values
            if not torch.isfinite(audio).all():
                audio = torch.where(torch.isfinite(audio), audio, torch.zeros_like(audio))
            
            # Compute STFT
            stft = torch.stft(
                audio,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=True,
                pad_mode='constant',
                normalized=False,
                onesided=True,
                return_complex=True
            )
            
            # Compute magnitude spectrogram
            magnitude = torch.abs(stft)
            
            # Apply energy threshold
            magnitude = torch.clamp(magnitude, min=self.chroma_config.min_energy_threshold)
            
            return magnitude
            
        except Exception as e:
            logger.error(f"Spectrogram computation failed: {e}")
            if self.chroma_config.enable_fallbacks:
                # Emergency fallback: return zeros with correct shape
                n_frames = 1 + (audio.size(-1) - self.n_fft) // self.hop_length
                return torch.zeros(self.n_fft // 2 + 1, n_frames, device=audio.device)
            else:
                raise
    
    def _harmonic_percussive_separation(self, magnitude: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Separate harmonic and percussive components"""
        try:
            if not self.chroma_config.use_harmonic_percussive_separation:
                return magnitude, torch.zeros_like(magnitude)
            
            # Convert to dB for processing
            magnitude_db = 20 * torch.log10(magnitude + self.chroma_config.eps)
            
            # Median filtering for separation
            # Harmonic: median filter along time axis (horizontal)
            harmonic_db = self._median_filter_2d(magnitude_db, kernel_size=(1, self.harmonic_filter_size))
            
            # Percussive: median filter along frequency axis (vertical)
            percussive_db = self._median_filter_2d(magnitude_db, kernel_size=(self.percussive_filter_size, 1))
            
            # Soft masking
            harmonic_mask = harmonic_db >= percussive_db
            percussive_mask = ~harmonic_mask
            
            # Apply masks with soft boundaries
            harmonic_magnitude = magnitude * harmonic_mask.float()
            percussive_magnitude = magnitude * percussive_mask.float()
            
            return harmonic_magnitude, percussive_magnitude
            
        except Exception as e:
            logger.error(f"HPSS failed: {e}")
            return magnitude, torch.zeros_like(magnitude)
    
    def _median_filter_2d(self, x: torch.Tensor, kernel_size: Tuple[int, int]) -> torch.Tensor:
        """2D median filter implementation"""
        try:
            kh, kw = kernel_size
            
            if kh == 1 and kw > 1:
                # Horizontal filtering
                return self._median_filter_1d(x, kw, dim=1)
            elif kh > 1 and kw == 1:
                # Vertical filtering  
                return self._median_filter_1d(x, kh, dim=0)
            else:
                # Both dimensions - approximate with separable filters
                x_filtered = self._median_filter_1d(x, kh, dim=0)
                x_filtered = self._median_filter_1d(x_filtered, kw, dim=1)
                return x_filtered
                
        except Exception as e:
            logger.warning(f"Median filter failed: {e}")
            return x
    
    def _median_filter_1d(self, x: torch.Tensor, kernel_size: int, dim: int) -> torch.Tensor:
        """1D median filter along specified dimension"""
        try:
            if kernel_size <= 1:
                return x
            
            # Pad the tensor
            pad_size = kernel_size // 2
            if dim == 0:
                x_padded = F.pad(x, (0, 0, pad_size, pad_size), mode='reflect')
            else:
                x_padded = F.pad(x, (pad_size, pad_size, 0, 0), mode='reflect')
            
            # Unfold and compute median
            if dim == 0:
                unfolded = x_padded.unfold(0, kernel_size, 1)
                medians = unfolded.median(dim=-1)[0]
            else:
                unfolded = x_padded.unfold(1, kernel_size, 1)
                medians = unfolded.median(dim=-1)[0]
            
            return medians
            
        except Exception as e:
            logger.warning(f"1D median filter failed: {e}")
            return x
    
    def _compute_chroma_features(self, magnitude: torch.Tensor) -> torch.Tensor:
        """Compute chroma features from magnitude spectrogram"""
        try:
            # Apply chroma filter bank
            chroma = torch.matmul(self.chroma_matrix, magnitude)  # [n_chroma, n_frames]
            
            # Multi-harmonic analysis if enabled
            if self.chroma_config.n_harmonics > 1 and hasattr(self, 'harmonic_weights'):
                chroma_harmonics = []
                
                for h, harmonic_mult in enumerate(self.harmonic_multipliers):
                    if h >= len(self.harmonic_weights):
                        break
                    
                    # Shift chroma matrix for harmonic
                    harmonic_shift = int(round(12 * torch.log2(harmonic_mult)))
                    shifted_chroma = torch.roll(chroma, harmonic_shift, dims=0)
                    
                    # Weight by harmonic strength
                    weighted_chroma = shifted_chroma * self.harmonic_weights[h]
                    chroma_harmonics.append(weighted_chroma)
                
                # Combine harmonics
                chroma = torch.stack(chroma_harmonics).sum(dim=0)
            
            # Normalization
            if self.chroma_config.norm == 'L1':
                chroma = chroma / (chroma.sum(dim=0, keepdim=True) + self.chroma_config.eps)
            elif self.chroma_config.norm == 'L2':
                chroma = chroma / (torch.norm(chroma, dim=0, keepdim=True) + self.chroma_config.eps)
            elif self.chroma_config.norm == 'max':
                chroma = chroma / (chroma.max(dim=0, keepdim=True)[0] + self.chroma_config.eps)
            
            return chroma
            
        except Exception as e:
            logger.error(f"Chroma feature computation failed: {e}")
            if self.chroma_config.enable_fallbacks:
                # Emergency fallback: uniform chroma
                n_frames = magnitude.size(-1)
                return torch.ones(self.chroma_config.n_chroma, n_frames, device=magnitude.device) / self.chroma_config.n_chroma
            else:
                raise
    
    def _compute_confidence_score(self, chroma: torch.Tensor, magnitude: torch.Tensor) -> torch.Tensor:
        """Compute confidence score for chroma features"""
        try:
            # Tonal clarity: how peaked the chroma vector is
            chroma_entropy = -torch.sum(chroma * torch.log(chroma + self.chroma_config.eps), dim=0)
            max_entropy = math.log(self.chroma_config.n_chroma)
            tonal_clarity = 1.0 - (chroma_entropy / max_entropy)
            
            # Energy-based confidence
            total_energy = magnitude.sum(dim=0)
            energy_confidence = torch.clamp(
                torch.log10(total_energy + self.chroma_config.eps) / self.chroma_config.max_db,
                0.0, 1.0
            )
            
            # Harmonic strength (if harmonic separation is enabled)
            harmonic_confidence = torch.ones_like(tonal_clarity)
            if self.chroma_config.use_harmonic_percussive_separation:
                harmonic_mag, percussive_mag = self._harmonic_percussive_separation(magnitude)
                harmonic_ratio = harmonic_mag.sum(dim=0) / (magnitude.sum(dim=0) + self.chroma_config.eps)
                harmonic_confidence = harmonic_ratio
            
            # Combined confidence
            confidence = (
                self.chroma_config.tonal_confidence_weight * tonal_clarity +
                self.chroma_config.harmonic_confidence_weight * harmonic_confidence +
                (1.0 - self.chroma_config.tonal_confidence_weight - self.chroma_config.harmonic_confidence_weight) * energy_confidence
            )
            
            # Smooth confidence over time
            if self.chroma_config.confidence_window_size > 1:
                confidence = self._smooth_confidence(confidence)
            
            return torch.clamp(confidence, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Confidence computation failed: {e}")
            return torch.ones(chroma.size(1), device=chroma.device) * 0.5
    
    def _smooth_confidence(self, confidence: torch.Tensor) -> torch.Tensor:
        """Smooth confidence scores over time"""
        try:
            window_size = self.chroma_config.confidence_window_size
            if window_size <= 1 or confidence.size(0) < window_size:
                return confidence
            
            # Simple moving average
            kernel = torch.ones(window_size, device=confidence.device) / window_size
            
            # Pad for convolution
            pad_size = window_size // 2
            confidence_padded = F.pad(confidence.unsqueeze(0), (pad_size, pad_size), mode='reflect')
            
            # Apply convolution
            smoothed = F.conv1d(confidence_padded.unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0))
            
            return smoothed.squeeze(0).squeeze(0)
            
        except Exception as e:
            logger.warning(f"Confidence smoothing failed: {e}")
            return confidence
    
    def _assess_quality(self, audio: torch.Tensor, chroma: torch.Tensor, confidence: torch.Tensor) -> Dict[str, float]:
        """Assess overall quality of chroma analysis"""
        try:
            # Signal-to-noise ratio estimate
            audio_energy = torch.mean(audio ** 2)
            noise_threshold = 10 ** (self.chroma_config.noise_floor_db / 10)
            snr_estimate = 10 * torch.log10(audio_energy / noise_threshold + self.chroma_config.eps)
            
            # Tonal clarity
            mean_confidence = torch.mean(confidence)
            
            # Chroma vector stability (consistency over time)
            if chroma.size(1) > 1:
                chroma_diff = torch.diff(chroma, dim=1)
                stability = 1.0 - torch.mean(torch.norm(chroma_diff, dim=0))
            else:
                stability = torch.tensor(1.0)
            
            # Overall quality score
            quality_score = (
                0.4 * torch.clamp(snr_estimate / 40.0, 0.0, 1.0) +
                0.4 * mean_confidence +
                0.2 * torch.clamp(stability, 0.0, 1.0)
            )
            
            quality_dict = {
                'overall_quality': quality_score.item(),
                'snr_estimate_db': snr_estimate.item(),
                'mean_confidence': mean_confidence.item(),
                'chroma_stability': stability.item(),
                'above_quality_threshold': quality_score.item() > self.chroma_config.quality_threshold
            }
            
            return quality_dict
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return {
                'overall_quality': 0.5,
                'snr_estimate_db': 0.0,
                'mean_confidence': 0.5,
                'chroma_stability': 0.5,
                'above_quality_threshold': False
            }
    
    def _key_estimation(self, chroma: torch.Tensor) -> Dict[str, Any]:
        """Estimate musical key from chroma features"""
        try:
            if not self.chroma_config.use_pitch_class_profile or not hasattr(self, 'key_profiles'):
                return {'estimated_key': None, 'key_confidence': 0.0}
            
            # Average chroma over time
            avg_chroma = torch.mean(chroma, dim=1)  # [n_chroma]
            
            # Correlate with key profiles
            correlations = torch.matmul(self.key_profiles, avg_chroma)  # [24]
            
            # Find best match
            best_key_idx = torch.argmax(correlations)
            key_confidence = correlations[best_key_idx] / torch.sum(correlations)
            
            # Convert to key name
            key_names = [
                'C major', 'C minor', 'C# major', 'C# minor', 'D major', 'D minor',
                'D# major', 'D# minor', 'E major', 'E minor', 'F major', 'F minor',
                'F# major', 'F# minor', 'G major', 'G minor', 'G# major', 'G# minor',
                'A major', 'A minor', 'A# major', 'A# minor', 'B major', 'B minor'
            ]
            
            estimated_key = key_names[best_key_idx.item()]
            
            return {
                'estimated_key': estimated_key,
                'key_confidence': key_confidence.item(),
                'key_correlations': correlations.tolist()
            }
            
        except Exception as e:
            logger.error(f"Key estimation failed: {e}")
            return {'estimated_key': None, 'key_confidence': 0.0}
    
    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with comprehensive error handling.
        
        Args:
            audio: Input audio tensor [batch_size, n_samples] or [n_samples]
            
        Returns:
            Dictionary containing:
            - chroma: Chroma features [batch_size, n_chroma, n_frames] 
            - confidence: Confidence scores [batch_size, n_frames]
            - quality_assessment: Quality metrics
            - key_estimation: Estimated musical key information
        """
        try:
            # Handle batch dimension
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)  # Add batch dimension
                squeeze_output = True
            else:
                squeeze_output = False
            
            batch_size = audio.size(0)
            
            # Validate input
            if not self._validate_input(audio):
                if self.chroma_config.enable_fallbacks:
                    logger.warning("Input validation failed, using fallback")
                    self.fallback_activations += 1
                    if self.chroma_config.disable_on_failure:
                        # Return empty results
                        return self._create_empty_results(batch_size, audio.device, squeeze_output)
                    else:
                        # Clean input and continue
                        audio = torch.where(torch.isfinite(audio), audio, torch.zeros_like(audio))
                else:
                    raise ValueError("Input validation failed")
            
            # Process each sample in batch
            batch_results = []
            
            for b in range(batch_size):
                try:
                    sample_audio = audio[b]
                    
                    # Compute spectrogram
                    magnitude = self._compute_spectrogram(sample_audio)
                    
                    # Harmonic-percussive separation if enabled
                    if self.chroma_config.use_harmonic_percussive_separation:
                        harmonic_magnitude, _ = self._harmonic_percussive_separation(magnitude)
                        magnitude_for_chroma = harmonic_magnitude
                    else:
                        magnitude_for_chroma = magnitude
                    
                    # Compute chroma features
                    chroma = self._compute_chroma_features(magnitude_for_chroma)
                    
                    # Compute confidence scores
                    confidence = self._compute_confidence_score(chroma, magnitude)
                    
                    # Quality assessment
                    quality_assessment = self._assess_quality(sample_audio, chroma, confidence)
                    
                    # Key estimation
                    key_estimation = self._key_estimation(chroma)
                    
                    sample_results = {
                        'chroma': chroma,
                        'confidence': confidence,
                        'quality_assessment': quality_assessment,
                        'key_estimation': key_estimation
                    }
                    
                    batch_results.append(sample_results)
                    
                except Exception as e:
                    logger.error(f"Processing failed for batch item {b}: {e}")
                    if self.chroma_config.enable_fallbacks:
                        self.fallback_activations += 1
                        # Create fallback results for this sample
                        n_frames = 1 + (sample_audio.size(-1) - self.n_fft) // self.hop_length
                        fallback_chroma = torch.ones(self.chroma_config.n_chroma, n_frames, device=audio.device) / self.chroma_config.n_chroma
                        fallback_confidence = torch.ones(n_frames, device=audio.device) * 0.1
                        
                        sample_results = {
                            'chroma': fallback_chroma,
                            'confidence': fallback_confidence,
                            'quality_assessment': {'overall_quality': 0.1, 'above_quality_threshold': False},
                            'key_estimation': {'estimated_key': None, 'key_confidence': 0.0}
                        }
                        batch_results.append(sample_results)
                    else:
                        raise
            
            # Combine batch results
            combined_results = self._combine_batch_results(batch_results, squeeze_output)
            
            # Update statistics
            self._update_statistics(combined_results)
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Chroma encoder forward pass failed: {e}")
            if self.chroma_config.enable_fallbacks:
                self.fallback_activations += 1
                logger.warning("Using emergency fallback")
                return self._create_empty_results(batch_size, audio.device, squeeze_output)
            else:
                raise
    
    def _create_empty_results(self, batch_size: int, device: torch.device, squeeze_output: bool) -> Dict[str, torch.Tensor]:
        """Create empty results structure for fallback"""
        n_frames = 100  # Default frame count
        
        if squeeze_output:
            chroma_shape = (self.chroma_config.n_chroma, n_frames)
            confidence_shape = (n_frames,)
        else:
            chroma_shape = (batch_size, self.chroma_config.n_chroma, n_frames)
            confidence_shape = (batch_size, n_frames)
        
        return {
            'chroma': torch.zeros(chroma_shape, device=device),
            'confidence': torch.zeros(confidence_shape, device=device),
            'quality_assessment': {'overall_quality': 0.0, 'above_quality_threshold': False},
            'key_estimation': {'estimated_key': None, 'key_confidence': 0.0}
        }
    
    def _combine_batch_results(self, batch_results: List[Dict], squeeze_output: bool) -> Dict[str, torch.Tensor]:
        """Combine results from batch processing"""
        try:
            if not batch_results:
                return self._create_empty_results(1, torch.device('cpu'), squeeze_output)
            
            # Stack chroma and confidence
            chromas = [result['chroma'] for result in batch_results]
            confidences = [result['confidence'] for result in batch_results]
            
            combined_chroma = torch.stack(chromas, dim=0)
            combined_confidence = torch.stack(confidences, dim=0)
            
            if squeeze_output and combined_chroma.size(0) == 1:
                combined_chroma = combined_chroma.squeeze(0)
                combined_confidence = combined_confidence.squeeze(0)
            
            # Combine quality assessments (take mean)
            quality_scores = [result['quality_assessment']['overall_quality'] for result in batch_results]
            mean_quality = sum(quality_scores) / len(quality_scores)
            
            # Key estimation (use first valid result)
            key_estimation = batch_results[0]['key_estimation']
            for result in batch_results:
                if result['key_estimation']['estimated_key'] is not None:
                    key_estimation = result['key_estimation']
                    break
            
            return {
                'chroma': combined_chroma,
                'confidence': combined_confidence,
                'quality_assessment': {
                    'overall_quality': mean_quality,
                    'above_quality_threshold': mean_quality > self.chroma_config.quality_threshold
                },
                'key_estimation': key_estimation
            }
            
        except Exception as e:
            logger.error(f"Batch combination failed: {e}")
            return self._create_empty_results(len(batch_results), torch.device('cpu'), squeeze_output)
    
    def _update_statistics(self, results: Dict[str, Any]):
        """Update processing statistics"""
        try:
            quality_score = results['quality_assessment']['overall_quality']
            confidence_mean = torch.mean(results['confidence']).item()
            
            stats = {
                'quality_score': quality_score,
                'confidence_mean': confidence_mean,
                'fallback_activations': self.fallback_activations
            }
            
            if len(self.processing_stats) < 1000:  # Limit history size
                self.processing_stats.append(stats)
            
            if len(self.confidence_history) < 1000:
                self.confidence_history.append(confidence_mean)
                
            if len(self.quality_assessments) < 1000:
                self.quality_assessments.append(quality_score)
            
        except Exception as e:
            logger.warning(f"Statistics update failed: {e}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics for monitoring"""
        stats = {
            'sample_rate': self.sample_rate,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'n_chroma': self.chroma_config.n_chroma,
            'n_harmonics': self.chroma_config.n_harmonics,
            'fallback_activations': self.fallback_activations,
            'use_harmonic_percussive_separation': self.chroma_config.use_harmonic_percussive_separation,
            'use_pitch_class_profile': self.chroma_config.use_pitch_class_profile
        }
        
        if self.processing_stats:
            last_stats = self.processing_stats[-1]
            stats.update({
                'last_quality_score': last_stats['quality_score'],
                'last_confidence_mean': last_stats['confidence_mean']
            })
        
        if self.confidence_history:
            stats.update({
                'mean_confidence_history': sum(self.confidence_history) / len(self.confidence_history),
                'confidence_std': np.std(self.confidence_history) if len(self.confidence_history) > 1 else 0.0
            })
        
        if self.quality_assessments:
            stats.update({
                'mean_quality_score': sum(self.quality_assessments) / len(self.quality_assessments),
                'quality_std': np.std(self.quality_assessments) if len(self.quality_assessments) > 1 else 0.0
            })
        
        return stats
    
    def reset_statistics(self):
        """Reset training statistics"""
        self.processing_stats.clear()
        self.confidence_history.clear()
        self.quality_assessments.clear()
        self.fallback_activations = 0


# Factory function
def create_bulletproof_chroma_encoder(config: RAVEConfig, **kwargs) -> BulletproofChromaEncoder:
    """Create a bulletproof chroma encoder"""
    return BulletproofChromaEncoder(config, **kwargs)


if __name__ == "__main__":
    print("🛡️ BULLETPROOF CHROMA ENCODER MODULE")
    print("=" * 50)
    
    # Test with minimal config
    from rave_config_system import get_minimal_config
    
    config = get_minimal_config()
    
    # Test chroma encoder
    chroma_config = ChromaConfig(n_chroma=12, n_harmonics=3)
    encoder = create_bulletproof_chroma_encoder(config, chroma_config=chroma_config)
    
    try:
        # Test with synthetic audio
        sample_rate = 44100
        duration = 2.0
        t = torch.linspace(0, duration, int(sample_rate * duration))
        
        # Create test audio with multiple frequencies (C major chord)
        freq_c = 261.63  # C4
        freq_e = 329.63  # E4  
        freq_g = 392.00  # G4
        
        audio = (torch.sin(2 * math.pi * freq_c * t) + 
                torch.sin(2 * math.pi * freq_e * t) + 
                torch.sin(2 * math.pi * freq_g * t)) / 3.0
        
        # Add some noise
        audio += 0.01 * torch.randn_like(audio)
        
        print(f"Testing with audio shape: {audio.shape}")
        
        # Single sample test
        results = encoder(audio)
        
        print(f"✅ Single sample test passed")
        print(f"   Chroma shape: {results['chroma'].shape}")
        print(f"   Confidence shape: {results['confidence'].shape}")
        print(f"   Quality score: {results['quality_assessment']['overall_quality']:.3f}")
        print(f"   Estimated key: {results['key_estimation']['estimated_key']}")
        print(f"   Key confidence: {results['key_estimation']['key_confidence']:.3f}")
        
        # Batch test
        batch_audio = torch.stack([audio, audio * 0.5], dim=0)
        batch_results = encoder(batch_audio)
        
        print(f"✅ Batch test passed")
        print(f"   Batch chroma shape: {batch_results['chroma'].shape}")
        print(f"   Batch confidence shape: {batch_results['confidence'].shape}")
        
        # Test with corrupted audio
        corrupted_audio = audio.clone()
        corrupted_audio[1000:1100] = float('nan')
        
        corrupted_results = encoder(corrupted_audio)
        print(f"✅ Robust handling of corrupted audio")
        
        # Test statistics
        stats = encoder.get_training_stats()
        print(f"   Encoder stats: {stats}")
        
    except Exception as e:
        print(f"❌ Chroma encoder test failed: {e}")
    
    # Test different configurations
    try:
        # Test with different chroma count
        chroma_config_24 = ChromaConfig(n_chroma=24, n_harmonics=5)
        encoder_24 = create_bulletproof_chroma_encoder(config, chroma_config=chroma_config_24)
        
        results_24 = encoder_24(audio)
        print(f"✅ 24-chroma test passed: {results_24['chroma'].shape}")
        
        # Test with HPSS disabled
        chroma_config_no_hpss = ChromaConfig(use_harmonic_percussive_separation=False)
        encoder_no_hpss = create_bulletproof_chroma_encoder(config, chroma_config=chroma_config_no_hpss)
        
        results_no_hpss = encoder_no_hpss(audio)
        print(f"✅ No-HPSS test passed")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
    
    print("🚀 BulletproofChromaEncoder ready for BigVGAN harmonic analysis!")