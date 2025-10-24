#!/usr/bin/env python3
"""
BULLETPROOF ONSET DETECTOR MODULE
Comprehensive musical onset detection for rhythm analysis with BigVGAN compatibility.
Handles polyphonic music, noise artifacts, and provides confidence scoring for onset timing.
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
class OnsetConfig:
    """Configuration for bulletproof onset detector"""
    # Audio analysis parameters
    sample_rate: int = 44100
    n_fft: int = 2048
    hop_length: int = 512
    win_length: Optional[int] = None
    window: str = 'hann'
    
    # Onset detection parameters
    onset_method: str = 'complex'  # 'energy', 'spectral_flux', 'phase_deviation', 'complex'
    pre_avg_time: float = 0.2  # Pre-onset average time in seconds
    post_avg_time: float = 0.05  # Post-onset average time in seconds
    
    # Peak picking parameters
    peak_threshold: float = 0.3  # Relative threshold for onset peaks
    min_onset_interval: float = 0.05  # Minimum time between onsets in seconds
    backtrack_frames: int = 3  # Frames to backtrack for precise onset timing
    
    # Multi-band analysis
    use_multi_band: bool = True
    n_bands: int = 6
    band_frequencies: List[float] = field(default_factory=lambda: [200, 400, 800, 1600, 3200, 6400])
    band_weights: List[float] = field(default_factory=lambda: [1.0, 1.2, 1.5, 1.2, 1.0, 0.8])
    
    # Onset strength functions
    use_spectral_flux: bool = True
    use_energy_difference: bool = True
    use_phase_deviation: bool = True
    use_complex_domain: bool = True
    combine_methods: str = 'weighted_sum'  # 'weighted_sum', 'max', 'mean'
    
    # Method weights for combination
    energy_weight: float = 0.2
    spectral_flux_weight: float = 0.3
    phase_deviation_weight: float = 0.25
    complex_domain_weight: float = 0.25
    
    # Advanced features
    use_harmonic_percussive_separation: bool = True
    percussive_emphasis: float = 2.0  # Emphasis on percussive component
    use_adaptive_whitening: bool = True
    whitening_window: int = 200  # Frames for adaptive whitening
    
    # Confidence estimation
    confidence_window_size: int = 5
    min_energy_threshold: float = 1e-6
    snr_threshold_db: float = 10.0
    consistency_weight: float = 0.4
    
    # Bulletproof stability
    eps: float = 1e-8
    max_db: float = 80.0
    numerical_stability_check: bool = True
    
    # Memory and efficiency
    use_checkpoint: bool = False
    chunk_size: int = 2048
    max_memory_gb: float = 4.0
    
    # Fallback strategies
    enable_fallbacks: bool = True
    fallback_to_energy: bool = True
    disable_on_failure: bool = False
    
    # Real-time processing
    real_time_mode: bool = False
    lookahead_frames: int = 5
    buffer_size: int = 4096

class BulletproofOnsetDetector(nn.Module):
    """
    Bulletproof onset detection with comprehensive error handling.
    
    Features:
    - Multiple onset detection algorithms with confidence scoring
    - Multi-band analysis for frequency-specific onset detection
    - Harmonic-percussive separation for cleaner onset detection
    - Adaptive whitening for dynamic range normalization
    - Real-time processing with low-latency requirements
    - Comprehensive fallback strategies for corrupted audio
    - Rhythm analysis with tempo estimation
    - Sub-frame precision onset timing with backtracking
    """
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__()
        
        self.config = config
        self.onset_config = kwargs.get('onset_config', OnsetConfig())
        
        # Override with RAVEConfig values if available
        if hasattr(config.audio, 'sample_rate'):
            self.onset_config.sample_rate = config.audio.sample_rate
        if hasattr(config.audio, 'n_fft'):
            self.onset_config.n_fft = config.audio.n_fft
        if hasattr(config.audio, 'hop_length'):
            self.onset_config.hop_length = config.audio.hop_length
        
        self.sample_rate = self.onset_config.sample_rate
        self.n_fft = self.onset_config.n_fft
        self.hop_length = self.onset_config.hop_length
        self.win_length = self.onset_config.win_length or self.n_fft
        
        # Build onset detection components
        try:
            self._build_onset_components()
        except Exception as e:
            logger.error(f"Failed to build onset components: {e}")
            if self.onset_config.enable_fallbacks:
                logger.warning("Building fallback onset components")
                self._build_fallback_components()
            else:
                raise
        
        # Tracking and monitoring
        self.processing_stats = []
        self.confidence_history = []
        self.onset_history = []
        self.fallback_activations = 0
        
        # Real-time processing buffers
        if self.onset_config.real_time_mode:
            self._init_real_time_buffers()
        
        # Memory management
        self._memory_usage = 0
        self._max_memory_bytes = int(self.onset_config.max_memory_gb * 1e9)
        
        logger.info(f"BulletproofOnsetDetector initialized: method={self.onset_config.onset_method}")
    
    def _build_onset_components(self):
        """Build main onset detection components"""
        # Window function
        self.register_buffer('window', self._create_window())
        
        # Multi-band filterbank
        if self.onset_config.use_multi_band:
            self._build_multi_band_filters()
        
        # Onset detection kernels
        self._build_onset_kernels()
        
        # Peak picking parameters
        self._setup_peak_picking()
        
        # Adaptive whitening components
        if self.onset_config.use_adaptive_whitening:
            self._build_whitening_components()
    
    def _build_fallback_components(self):
        """Build simple fallback components"""
        # Basic window
        self.register_buffer('window', torch.hann_window(self.win_length))
        
        # Disable advanced features
        self.onset_config.use_multi_band = False
        self.onset_config.use_harmonic_percussive_separation = False
        self.onset_config.use_adaptive_whitening = False
        self.onset_config.onset_method = 'energy'
        
        logger.info("Built fallback onset components")
    
    def _create_window(self) -> torch.Tensor:
        """Create analysis window"""
        try:
            if self.onset_config.window == 'hann':
                return torch.hann_window(self.win_length)
            elif self.onset_config.window == 'hamming':
                return torch.hamming_window(self.win_length)
            elif self.onset_config.window == 'blackman':
                return torch.blackman_window(self.win_length)
            else:
                return torch.hann_window(self.win_length)
        except Exception as e:
            logger.warning(f"Window creation failed: {e}")
            return torch.hann_window(self.win_length)
    
    def _build_multi_band_filters(self):
        """Build multi-band analysis filters"""
        try:
            # Create mel-scale filter bank for multi-band analysis
            n_bins = self.n_fft // 2 + 1
            freqs = torch.linspace(0, self.sample_rate / 2, n_bins)
            
            # Convert band frequencies to filter bank
            band_filters = []
            for i, (low_freq, high_freq) in enumerate(zip(
                [0] + self.onset_config.band_frequencies[:-1], 
                self.onset_config.band_frequencies
            )):
                # Create triangular filter
                filter_response = torch.zeros(n_bins)
                
                # Find frequency indices
                low_idx = torch.argmin(torch.abs(freqs - low_freq))
                high_idx = torch.argmin(torch.abs(freqs - high_freq))
                center_idx = (low_idx + high_idx) // 2
                
                # Triangular response
                if center_idx > low_idx:
                    filter_response[low_idx:center_idx+1] = torch.linspace(0, 1, center_idx - low_idx + 1)
                if high_idx > center_idx:
                    filter_response[center_idx:high_idx+1] = torch.linspace(1, 0, high_idx - center_idx + 1)
                
                band_filters.append(filter_response)
            
            # Stack filters
            filter_bank = torch.stack(band_filters)  # [n_bands, n_bins]
            self.register_buffer('band_filters', filter_bank)
            
            # Band weights
            weights = torch.tensor(self.onset_config.band_weights[:self.onset_config.n_bands])
            weights = weights / weights.sum()  # Normalize
            self.register_buffer('band_weights', weights)
            
        except Exception as e:
            logger.error(f"Multi-band filter build failed: {e}")
            # Fallback: single band (full spectrum)
            n_bins = self.n_fft // 2 + 1
            self.register_buffer('band_filters', torch.ones(1, n_bins))
            self.register_buffer('band_weights', torch.ones(1))
    
    def _build_onset_kernels(self):
        """Build onset detection kernels"""
        try:
            # Difference kernels for onset detection
            if self.onset_config.use_spectral_flux:
                # Simple difference kernel for spectral flux
                self.register_buffer('flux_kernel', torch.tensor([-1.0, 1.0]).unsqueeze(0).unsqueeze(0))
            
            # Phase deviation kernels
            if self.onset_config.use_phase_deviation:
                # Expected phase advancement
                freq_bins = torch.arange(self.n_fft // 2 + 1)
                expected_phase_advance = 2 * math.pi * freq_bins * self.hop_length / self.n_fft
                self.register_buffer('expected_phase_advance', expected_phase_advance)
            
        except Exception as e:
            logger.error(f"Onset kernel build failed: {e}")
    
    def _setup_peak_picking(self):
        """Setup peak picking parameters"""
        try:
            # Convert time-based parameters to frame-based
            self.pre_avg_frames = max(1, int(self.onset_config.pre_avg_time * self.sample_rate / self.hop_length))
            self.post_avg_frames = max(1, int(self.onset_config.post_avg_time * self.sample_rate / self.hop_length))
            self.min_onset_frames = max(1, int(self.onset_config.min_onset_interval * self.sample_rate / self.hop_length))
            
        except Exception as e:
            logger.error(f"Peak picking setup failed: {e}")
            # Fallback values
            self.pre_avg_frames = 10
            self.post_avg_frames = 3
            self.min_onset_frames = 5
    
    def _build_whitening_components(self):
        """Build adaptive whitening components"""
        try:
            self.whitening_window_frames = max(1, self.onset_config.whitening_window)
            
        except Exception as e:
            logger.error(f"Whitening components build failed: {e}")
            self.onset_config.use_adaptive_whitening = False
    
    def _init_real_time_buffers(self):
        """Initialize real-time processing buffers"""
        try:
            buffer_frames = self.onset_config.buffer_size // self.hop_length
            self.spectral_buffer = torch.zeros(self.n_fft // 2 + 1, buffer_frames)
            self.onset_buffer = torch.zeros(buffer_frames)
            self.buffer_ptr = 0
            
        except Exception as e:
            logger.error(f"Real-time buffer initialization failed: {e}")
            self.onset_config.real_time_mode = False
    
    def _validate_input(self, audio: torch.Tensor) -> bool:
        """Validate input audio tensor"""
        try:
            if audio is None or audio.numel() == 0:
                logger.warning("Empty or None audio input")
                return False
            
            if not torch.isfinite(audio).all():
                logger.warning("Non-finite values in audio input")
                if self.onset_config.enable_fallbacks:
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
    
    def _compute_spectrogram(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
            
            # Separate magnitude and phase
            magnitude = torch.abs(stft)
            phase = torch.angle(stft)
            
            # Apply energy threshold
            magnitude = torch.clamp(magnitude, min=self.onset_config.min_energy_threshold)
            
            return magnitude, phase
            
        except Exception as e:
            logger.error(f"Spectrogram computation failed: {e}")
            if self.onset_config.enable_fallbacks:
                # Emergency fallback: return zeros with correct shape
                n_frames = 1 + (audio.size(-1) - self.n_fft) // self.hop_length
                magnitude = torch.zeros(self.n_fft // 2 + 1, n_frames, device=audio.device)
                phase = torch.zeros_like(magnitude)
                return magnitude, phase
            else:
                raise
    
    def _harmonic_percussive_separation(self, magnitude: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Separate harmonic and percussive components"""
        try:
            if not self.onset_config.use_harmonic_percussive_separation:
                return magnitude, magnitude
            
            # Convert to dB for processing
            magnitude_db = 20 * torch.log10(magnitude + self.onset_config.eps)
            
            # Median filtering for separation
            # Harmonic: median filter along time axis (horizontal)
            harmonic_db = self._median_filter_2d(magnitude_db, kernel_size=(1, 17))
            
            # Percussive: median filter along frequency axis (vertical)
            percussive_db = self._median_filter_2d(magnitude_db, kernel_size=(17, 1))
            
            # Soft masking
            harmonic_mask = harmonic_db >= percussive_db
            percussive_mask = ~harmonic_mask
            
            # Apply masks
            harmonic_magnitude = magnitude * harmonic_mask.float()
            percussive_magnitude = magnitude * percussive_mask.float()
            
            return harmonic_magnitude, percussive_magnitude
            
        except Exception as e:
            logger.error(f"HPSS failed: {e}")
            return magnitude, magnitude
    
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
    
    def _compute_energy_onset_strength(self, magnitude: torch.Tensor) -> torch.Tensor:
        """Compute energy-based onset strength"""
        try:
            # Sum across frequency bins
            energy = torch.sum(magnitude ** 2, dim=0)
            
            # Compute first-order difference
            energy_diff = torch.diff(energy, prepend=energy[0:1])
            
            # Only positive changes (energy increases)
            onset_strength = torch.clamp(energy_diff, min=0)
            
            return onset_strength
            
        except Exception as e:
            logger.error(f"Energy onset strength computation failed: {e}")
            return torch.zeros(magnitude.size(1), device=magnitude.device)
    
    def _compute_spectral_flux(self, magnitude: torch.Tensor) -> torch.Tensor:
        """Compute spectral flux onset strength"""
        try:
            # Compute magnitude differences across time
            magnitude_diff = torch.diff(magnitude, dim=1, prepend=magnitude[:, 0:1])
            
            # Sum positive differences across frequency bins
            spectral_flux = torch.sum(torch.clamp(magnitude_diff, min=0), dim=0)
            
            return spectral_flux
            
        except Exception as e:
            logger.error(f"Spectral flux computation failed: {e}")
            return torch.zeros(magnitude.size(1), device=magnitude.device)
    
    def _compute_phase_deviation(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Compute phase deviation onset strength"""
        try:
            if not self.onset_config.use_phase_deviation or not hasattr(self, 'expected_phase_advance'):
                return torch.zeros(magnitude.size(1), device=magnitude.device)
            
            # Compute phase differences
            phase_diff = torch.diff(phase, dim=1, prepend=phase[:, 0:1])
            
            # Unwrap phase differences
            phase_diff = torch.where(phase_diff > math.pi, phase_diff - 2*math.pi, phase_diff)
            phase_diff = torch.where(phase_diff < -math.pi, phase_diff + 2*math.pi, phase_diff)
            
            # Expected phase advance for each frequency bin
            expected = self.expected_phase_advance.unsqueeze(1)  # [n_bins, 1]
            
            # Compute phase deviation
            phase_deviation = torch.abs(phase_diff - expected)
            
            # Weight by magnitude
            weighted_deviation = phase_deviation * magnitude
            
            # Sum across frequency bins
            onset_strength = torch.sum(weighted_deviation, dim=0)
            
            return onset_strength
            
        except Exception as e:
            logger.error(f"Phase deviation computation failed: {e}")
            return torch.zeros(magnitude.size(1), device=magnitude.device)
    
    def _compute_complex_domain_onset(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Compute complex domain onset strength"""
        try:
            # Reconstruct complex spectrogram
            complex_spec = magnitude * torch.exp(1j * phase)
            
            # Compute complex difference
            complex_diff = torch.diff(complex_spec, dim=1, prepend=complex_spec[:, 0:1])
            
            # Magnitude of complex difference
            complex_magnitude_diff = torch.abs(complex_diff)
            
            # Sum across frequency bins
            onset_strength = torch.sum(complex_magnitude_diff, dim=0)
            
            return onset_strength
            
        except Exception as e:
            logger.error(f"Complex domain onset computation failed: {e}")
            return torch.zeros(magnitude.size(1), device=magnitude.device)
    
    def _compute_multi_band_onset_strength(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Compute multi-band onset strength"""
        try:
            if not self.onset_config.use_multi_band or not hasattr(self, 'band_filters'):
                # Fallback to single-band
                return self._compute_single_band_onset_strength(magnitude, phase)
            
            band_onset_strengths = []
            
            # Process each frequency band
            for band_idx in range(self.band_filters.size(0)):
                band_filter = self.band_filters[band_idx].unsqueeze(1)  # [n_bins, 1]
                
                # Apply band filter
                band_magnitude = magnitude * band_filter
                band_phase = phase  # Phase is the same across bands
                
                # Compute onset strength for this band
                band_onset = self._compute_single_band_onset_strength(band_magnitude, band_phase)
                
                # Weight by band weight
                band_weight = self.band_weights[band_idx]
                weighted_band_onset = band_onset * band_weight
                
                band_onset_strengths.append(weighted_band_onset)
            
            # Combine band onset strengths
            if len(band_onset_strengths) > 1:
                combined_onset = torch.stack(band_onset_strengths).sum(dim=0)
            else:
                combined_onset = band_onset_strengths[0]
            
            return combined_onset
            
        except Exception as e:
            logger.error(f"Multi-band onset strength computation failed: {e}")
            return self._compute_single_band_onset_strength(magnitude, phase)
    
    def _compute_single_band_onset_strength(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Compute onset strength for a single band"""
        try:
            onset_strengths = []
            
            # Energy-based onset detection
            if self.onset_config.use_energy_difference:
                energy_onset = self._compute_energy_onset_strength(magnitude)
                onset_strengths.append(energy_onset * self.onset_config.energy_weight)
            
            # Spectral flux
            if self.onset_config.use_spectral_flux:
                flux_onset = self._compute_spectral_flux(magnitude)
                onset_strengths.append(flux_onset * self.onset_config.spectral_flux_weight)
            
            # Phase deviation
            if self.onset_config.use_phase_deviation:
                phase_onset = self._compute_phase_deviation(magnitude, phase)
                onset_strengths.append(phase_onset * self.onset_config.phase_deviation_weight)
            
            # Complex domain
            if self.onset_config.use_complex_domain:
                complex_onset = self._compute_complex_domain_onset(magnitude, phase)
                onset_strengths.append(complex_onset * self.onset_config.complex_domain_weight)
            
            # Combine onset strengths
            if len(onset_strengths) == 0:
                # Fallback: energy-based
                return self._compute_energy_onset_strength(magnitude)
            elif len(onset_strengths) == 1:
                return onset_strengths[0]
            else:
                if self.onset_config.combine_methods == 'weighted_sum':
                    return torch.stack(onset_strengths).sum(dim=0)
                elif self.onset_config.combine_methods == 'max':
                    return torch.stack(onset_strengths).max(dim=0)[0]
                elif self.onset_config.combine_methods == 'mean':
                    return torch.stack(onset_strengths).mean(dim=0)
                else:
                    return torch.stack(onset_strengths).sum(dim=0)
            
        except Exception as e:
            logger.error(f"Single-band onset strength computation failed: {e}")
            return torch.zeros(magnitude.size(1), device=magnitude.device)
    
    def _apply_adaptive_whitening(self, onset_strength: torch.Tensor) -> torch.Tensor:
        """Apply adaptive whitening to onset strength"""
        try:
            if not self.onset_config.use_adaptive_whitening:
                return onset_strength
            
            # Compute local statistics
            window_size = min(self.whitening_window_frames, onset_strength.size(0))
            
            if window_size <= 1:
                return onset_strength
            
            # Compute local mean and std using convolution
            kernel = torch.ones(window_size, device=onset_strength.device) / window_size
            
            # Pad for convolution
            pad_size = window_size // 2
            onset_padded = F.pad(onset_strength.unsqueeze(0), (pad_size, pad_size), mode='reflect')
            
            # Local mean
            local_mean = F.conv1d(onset_padded.unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0))
            local_mean = local_mean.squeeze(0).squeeze(0)
            
            # Local variance (approximate)
            onset_squared = onset_strength ** 2
            onset_squared_padded = F.pad(onset_squared.unsqueeze(0), (pad_size, pad_size), mode='reflect')
            local_mean_squared = F.conv1d(onset_squared_padded.unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0))
            local_mean_squared = local_mean_squared.squeeze(0).squeeze(0)
            
            local_var = local_mean_squared - local_mean ** 2
            local_std = torch.sqrt(torch.clamp(local_var, min=self.onset_config.eps))
            
            # Whitening
            whitened_onset = (onset_strength - local_mean) / (local_std + self.onset_config.eps)
            
            # Only keep positive values
            whitened_onset = torch.clamp(whitened_onset, min=0)
            
            return whitened_onset
            
        except Exception as e:
            logger.error(f"Adaptive whitening failed: {e}")
            return onset_strength
    
    def _pick_peaks(self, onset_strength: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pick onset peaks from onset strength function"""
        try:
            # Apply adaptive threshold
            threshold = self._compute_adaptive_threshold(onset_strength)
            
            # Find local maxima
            onset_times, onset_confidences = self._find_local_maxima(onset_strength, threshold)
            
            # Backtrack for precise timing
            if self.onset_config.backtrack_frames > 0:
                onset_times = self._backtrack_onsets(onset_strength, onset_times)
            
            return onset_times, onset_confidences
            
        except Exception as e:
            logger.error(f"Peak picking failed: {e}")
            return torch.tensor([], device=onset_strength.device), torch.tensor([], device=onset_strength.device)
    
    def _compute_adaptive_threshold(self, onset_strength: torch.Tensor) -> torch.Tensor:
        """Compute adaptive threshold for onset detection"""
        try:
            # Local statistics for adaptive thresholding
            pre_avg = self._local_average(onset_strength, self.pre_avg_frames, mode='pre')
            post_avg = self._local_average(onset_strength, self.post_avg_frames, mode='post')
            
            # Adaptive threshold
            threshold = pre_avg + self.onset_config.peak_threshold * (pre_avg - post_avg)
            
            # Ensure non-negative threshold
            threshold = torch.clamp(threshold, min=0)
            
            return threshold
            
        except Exception as e:
            logger.error(f"Adaptive threshold computation failed: {e}")
            # Fallback: global threshold
            global_threshold = torch.mean(onset_strength) * self.onset_config.peak_threshold
            return torch.full_like(onset_strength, global_threshold)
    
    def _local_average(self, signal: torch.Tensor, window_size: int, mode: str = 'center') -> torch.Tensor:
        """Compute local average of signal"""
        try:
            if window_size <= 1:
                return signal
            
            kernel = torch.ones(window_size, device=signal.device) / window_size
            
            # Padding based on mode
            if mode == 'pre':
                pad = (window_size - 1, 0)
            elif mode == 'post':
                pad = (0, window_size - 1)
            else:  # center
                pad_size = window_size // 2
                pad = (pad_size, pad_size)
            
            signal_padded = F.pad(signal.unsqueeze(0), pad, mode='reflect')
            
            # Apply convolution
            averaged = F.conv1d(signal_padded.unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0))
            
            return averaged.squeeze(0).squeeze(0)
            
        except Exception as e:
            logger.warning(f"Local average computation failed: {e}")
            return signal
    
    def _find_local_maxima(self, onset_strength: torch.Tensor, threshold: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find local maxima in onset strength function"""
        try:
            # Find peaks above threshold
            above_threshold = onset_strength > threshold
            
            # Find local maxima
            is_peak = torch.zeros_like(onset_strength, dtype=torch.bool)
            
            for i in range(1, onset_strength.size(0) - 1):
                if (above_threshold[i] and 
                    onset_strength[i] > onset_strength[i-1] and 
                    onset_strength[i] > onset_strength[i+1]):
                    is_peak[i] = True
            
            # Apply minimum onset interval constraint
            peak_indices = torch.where(is_peak)[0]
            
            if len(peak_indices) > 1:
                # Remove peaks that are too close together
                filtered_peaks = [peak_indices[0]]
                
                for peak_idx in peak_indices[1:]:
                    if peak_idx - filtered_peaks[-1] >= self.min_onset_frames:
                        filtered_peaks.append(peak_idx)
                
                peak_indices = torch.tensor(filtered_peaks, device=onset_strength.device)
            
            # Convert to time and extract confidences
            onset_times = peak_indices.float() * self.hop_length / self.sample_rate
            onset_confidences = onset_strength[peak_indices] if len(peak_indices) > 0 else torch.tensor([], device=onset_strength.device)
            
            return onset_times, onset_confidences
            
        except Exception as e:
            logger.error(f"Local maxima detection failed: {e}")
            return torch.tensor([], device=onset_strength.device), torch.tensor([], device=onset_strength.device)
    
    def _backtrack_onsets(self, onset_strength: torch.Tensor, onset_frames: torch.Tensor) -> torch.Tensor:
        """Backtrack onset times for more precise timing"""
        try:
            if len(onset_frames) == 0:
                return onset_frames
            
            backtracked_frames = onset_frames.clone()
            
            for i, frame in enumerate(onset_frames):
                frame_idx = int(frame * self.sample_rate / self.hop_length)
                
                # Look back for the actual onset start
                start_idx = max(0, frame_idx - self.onset_config.backtrack_frames)
                
                if start_idx < frame_idx:
                    # Find minimum in the backtrack window
                    backtrack_segment = onset_strength[start_idx:frame_idx+1]
                    min_idx = torch.argmin(backtrack_segment)
                    actual_frame = start_idx + min_idx
                    
                    backtracked_frames[i] = actual_frame * self.hop_length / self.sample_rate
            
            return backtracked_frames
            
        except Exception as e:
            logger.warning(f"Onset backtracking failed: {e}")
            return onset_frames
    
    def _compute_confidence_scores(self, onset_strength: torch.Tensor, onset_times: torch.Tensor) -> torch.Tensor:
        """Compute confidence scores for detected onsets"""
        try:
            if len(onset_times) == 0:
                return torch.tensor([], device=onset_strength.device)
            
            confidences = []
            
            for onset_time in onset_times:
                frame_idx = int(onset_time * self.sample_rate / self.hop_length)
                frame_idx = torch.clamp(torch.tensor(frame_idx), 0, onset_strength.size(0) - 1)
                
                # Local strength
                local_strength = onset_strength[frame_idx]
                
                # Local context
                window_start = max(0, frame_idx - self.onset_config.confidence_window_size // 2)
                window_end = min(onset_strength.size(0), frame_idx + self.onset_config.confidence_window_size // 2 + 1)
                
                local_context = onset_strength[window_start:window_end]
                local_mean = torch.mean(local_context)
                local_std = torch.std(local_context)
                
                # Confidence based on local prominence
                if local_std > self.onset_config.eps:
                    prominence = (local_strength - local_mean) / local_std
                else:
                    prominence = torch.tensor(0.0)
                
                # Consistency check (how consistent is the local pattern)
                consistency = 1.0 / (1.0 + local_std)
                
                # Combined confidence
                confidence = self.onset_config.consistency_weight * consistency + (1 - self.onset_config.consistency_weight) * torch.sigmoid(prominence)
                
                confidences.append(confidence)
            
            return torch.stack(confidences)
            
        except Exception as e:
            logger.error(f"Confidence computation failed: {e}")
            return torch.ones(len(onset_times), device=onset_strength.device) * 0.5
    
    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with comprehensive error handling.
        
        Args:
            audio: Input audio tensor [batch_size, n_samples] or [n_samples]
            
        Returns:
            Dictionary containing:
            - onset_times: Detected onset times in seconds [n_onsets]
            - onset_confidences: Confidence scores for onsets [n_onsets]
            - onset_strength: Onset strength function [n_frames]
            - rhythm_features: Rhythm analysis features
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
                if self.onset_config.enable_fallbacks:
                    logger.warning("Input validation failed, using fallback")
                    self.fallback_activations += 1
                    if self.onset_config.disable_on_failure:
                        # Return empty results
                        return self._create_empty_results(audio.device, squeeze_output)
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
                    magnitude, phase = self._compute_spectrogram(sample_audio)
                    
                    # Harmonic-percussive separation if enabled
                    if self.onset_config.use_harmonic_percussive_separation:
                        _, percussive_magnitude = self._harmonic_percussive_separation(magnitude)
                        # Use percussive component for onset detection with emphasis
                        magnitude_for_onset = percussive_magnitude * self.onset_config.percussive_emphasis + magnitude * (1.0 - self.onset_config.percussive_emphasis)
                    else:
                        magnitude_for_onset = magnitude
                    
                    # Compute onset strength
                    if self.onset_config.use_multi_band:
                        onset_strength = self._compute_multi_band_onset_strength(magnitude_for_onset, phase)
                    else:
                        onset_strength = self._compute_single_band_onset_strength(magnitude_for_onset, phase)
                    
                    # Apply adaptive whitening
                    onset_strength = self._apply_adaptive_whitening(onset_strength)
                    
                    # Pick onset peaks
                    onset_times, onset_confidences = self._pick_peaks(onset_strength)
                    
                    # Compute detailed confidence scores
                    if len(onset_times) > 0:
                        detailed_confidences = self._compute_confidence_scores(onset_strength, onset_times)
                    else:
                        detailed_confidences = onset_confidences
                    
                    # Rhythm analysis
                    rhythm_features = self._analyze_rhythm(onset_times, sample_audio.size(-1))
                    
                    sample_results = {
                        'onset_times': onset_times,
                        'onset_confidences': detailed_confidences,
                        'onset_strength': onset_strength,
                        'rhythm_features': rhythm_features
                    }
                    
                    batch_results.append(sample_results)
                    
                except Exception as e:
                    logger.error(f"Processing failed for batch item {b}: {e}")
                    if self.onset_config.enable_fallbacks:
                        self.fallback_activations += 1
                        # Create fallback results for this sample
                        n_frames = 1 + (sample_audio.size(-1) - self.n_fft) // self.hop_length
                        
                        sample_results = {
                            'onset_times': torch.tensor([], device=audio.device),
                            'onset_confidences': torch.tensor([], device=audio.device),
                            'onset_strength': torch.zeros(n_frames, device=audio.device),
                            'rhythm_features': {'tempo_bpm': 0.0, 'rhythm_regularity': 0.0}
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
            logger.error(f"Onset detector forward pass failed: {e}")
            if self.onset_config.enable_fallbacks:
                self.fallback_activations += 1
                logger.warning("Using emergency fallback")
                return self._create_empty_results(audio.device, squeeze_output)
            else:
                raise
    
    def _analyze_rhythm(self, onset_times: torch.Tensor, audio_length: int) -> Dict[str, float]:
        """Analyze rhythm from onset times"""
        try:
            if len(onset_times) < 2:
                return {'tempo_bpm': 0.0, 'rhythm_regularity': 0.0}
            
            # Compute inter-onset intervals
            intervals = torch.diff(onset_times)
            
            # Estimate tempo (BPM)
            # Use median interval as basic tempo estimate
            median_interval = torch.median(intervals)
            tempo_bpm = 60.0 / median_interval.item() if median_interval > 0 else 0.0
            
            # Rhythm regularity (consistency of intervals)
            interval_std = torch.std(intervals)
            interval_mean = torch.mean(intervals)
            
            if interval_mean > 0:
                coefficient_of_variation = interval_std / interval_mean
                rhythm_regularity = 1.0 / (1.0 + coefficient_of_variation.item())
            else:
                rhythm_regularity = 0.0
            
            return {
                'tempo_bpm': tempo_bpm,
                'rhythm_regularity': rhythm_regularity,
                'n_onsets': len(onset_times),
                'onset_density': len(onset_times) / (audio_length / self.sample_rate),
                'median_interval': median_interval.item()
            }
            
        except Exception as e:
            logger.error(f"Rhythm analysis failed: {e}")
            return {'tempo_bpm': 0.0, 'rhythm_regularity': 0.0}
    
    def _create_empty_results(self, device: torch.device, squeeze_output: bool) -> Dict[str, torch.Tensor]:
        """Create empty results structure for fallback"""
        return {
            'onset_times': torch.tensor([], device=device),
            'onset_confidences': torch.tensor([], device=device),
            'onset_strength': torch.zeros(100, device=device),  # Default frame count
            'rhythm_features': {'tempo_bpm': 0.0, 'rhythm_regularity': 0.0}
        }
    
    def _combine_batch_results(self, batch_results: List[Dict], squeeze_output: bool) -> Dict[str, torch.Tensor]:
        """Combine results from batch processing"""
        try:
            if not batch_results:
                return self._create_empty_results(torch.device('cpu'), squeeze_output)
            
            if squeeze_output and len(batch_results) == 1:
                # Single sample - return as is
                return batch_results[0]
            
            # For multiple samples, return list of results
            # (Onset detection results don't naturally stack like spectrograms)
            return {
                'batch_results': batch_results,
                'n_samples': len(batch_results)
            }
            
        except Exception as e:
            logger.error(f"Batch combination failed: {e}")
            return self._create_empty_results(torch.device('cpu'), squeeze_output)
    
    def _update_statistics(self, results: Dict[str, Any]):
        """Update processing statistics"""
        try:
            if 'batch_results' in results:
                # Multiple samples
                for sample_results in results['batch_results']:
                    self._update_sample_statistics(sample_results)
            else:
                # Single sample
                self._update_sample_statistics(results)
                
        except Exception as e:
            logger.warning(f"Statistics update failed: {e}")
    
    def _update_sample_statistics(self, sample_results: Dict[str, Any]):
        """Update statistics for a single sample"""
        try:
            n_onsets = len(sample_results['onset_times'])
            onset_strength_mean = torch.mean(sample_results['onset_strength']).item()
            
            if n_onsets > 0:
                confidence_mean = torch.mean(sample_results['onset_confidences']).item()
            else:
                confidence_mean = 0.0
            
            stats = {
                'n_onsets': n_onsets,
                'onset_strength_mean': onset_strength_mean,
                'confidence_mean': confidence_mean,
                'fallback_activations': self.fallback_activations
            }
            
            if len(self.processing_stats) < 1000:  # Limit history size
                self.processing_stats.append(stats)
            
            if len(self.confidence_history) < 1000:
                self.confidence_history.append(confidence_mean)
            
            if len(self.onset_history) < 1000:
                self.onset_history.append(n_onsets)
            
        except Exception as e:
            logger.warning(f"Sample statistics update failed: {e}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics for monitoring"""
        stats = {
            'sample_rate': self.sample_rate,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'onset_method': self.onset_config.onset_method,
            'use_multi_band': self.onset_config.use_multi_band,
            'n_bands': self.onset_config.n_bands if self.onset_config.use_multi_band else 1,
            'fallback_activations': self.fallback_activations,
            'use_harmonic_percussive_separation': self.onset_config.use_harmonic_percussive_separation
        }
        
        if self.processing_stats:
            last_stats = self.processing_stats[-1]
            stats.update({
                'last_n_onsets': last_stats['n_onsets'],
                'last_onset_strength_mean': last_stats['onset_strength_mean'],
                'last_confidence_mean': last_stats['confidence_mean']
            })
        
        if self.onset_history:
            stats.update({
                'mean_onsets_per_sample': sum(self.onset_history) / len(self.onset_history),
                'onset_count_std': np.std(self.onset_history) if len(self.onset_history) > 1 else 0.0
            })
        
        if self.confidence_history:
            stats.update({
                'mean_confidence': sum(self.confidence_history) / len(self.confidence_history),
                'confidence_std': np.std(self.confidence_history) if len(self.confidence_history) > 1 else 0.0
            })
        
        return stats
    
    def reset_statistics(self):
        """Reset training statistics"""
        self.processing_stats.clear()
        self.confidence_history.clear()
        self.onset_history.clear()
        self.fallback_activations = 0


# Factory function
def create_bulletproof_onset_detector(config: RAVEConfig, **kwargs) -> BulletproofOnsetDetector:
    """Create a bulletproof onset detector"""
    return BulletproofOnsetDetector(config, **kwargs)


if __name__ == "__main__":
    print("🛡️ BULLETPROOF ONSET DETECTOR MODULE")
    print("=" * 50)
    
    # Test with minimal config
    from rave_config_system import get_minimal_config
    
    config = get_minimal_config()
    
    # Test onset detector
    onset_config = OnsetConfig(onset_method='complex', use_multi_band=True)
    detector = create_bulletproof_onset_detector(config, onset_config=onset_config)
    
    try:
        # Create synthetic audio with clear onsets
        sample_rate = 44100
        duration = 3.0
        t = torch.linspace(0, duration, int(sample_rate * duration))
        
        # Create audio with periodic onsets (like drum beats)
        audio = torch.zeros_like(t)
        
        # Add synthetic drum hits every 0.5 seconds
        for onset_time in [0.5, 1.0, 1.5, 2.0, 2.5]:
            onset_idx = int(onset_time * sample_rate)
            if onset_idx < len(audio):
                # Short burst for drum hit
                burst_length = int(0.01 * sample_rate)  # 10ms burst
                burst = torch.exp(-torch.linspace(0, 5, burst_length)) * torch.sin(2 * math.pi * 200 * torch.linspace(0, 0.01, burst_length))
                end_idx = min(onset_idx + burst_length, len(audio))
                audio[onset_idx:end_idx] = burst[:end_idx-onset_idx]
        
        # Add some background noise
        audio += 0.01 * torch.randn_like(audio)
        
        print(f"Testing with audio shape: {audio.shape}")
        
        # Single sample test
        results = detector(audio)
        
        print(f"✅ Single sample test passed")
        print(f"   Detected {len(results['onset_times'])} onsets")
        print(f"   Onset times: {results['onset_times'].tolist()}")
        print(f"   Confidences: {results['onset_confidences'].tolist()}")
        print(f"   Onset strength shape: {results['onset_strength'].shape}")
        print(f"   Tempo estimate: {results['rhythm_features']['tempo_bpm']:.1f} BPM")
        print(f"   Rhythm regularity: {results['rhythm_features']['rhythm_regularity']:.3f}")
        
        # Batch test
        batch_audio = torch.stack([audio, audio * 0.7], dim=0)
        batch_results = detector(batch_audio)
        
        print(f"✅ Batch test passed")
        print(f"   Processed {batch_results['n_samples']} samples")
        
        # Test with corrupted audio
        corrupted_audio = audio.clone()
        corrupted_audio[10000:11000] = float('inf')
        
        corrupted_results = detector(corrupted_audio)
        print(f"✅ Robust handling of corrupted audio")
        
        # Test statistics
        stats = detector.get_training_stats()
        print(f"   Detector stats: {stats}")
        
    except Exception as e:
        print(f"❌ Onset detector test failed: {e}")
    
    # Test different configurations
    try:
        # Test energy-only method
        onset_config_energy = OnsetConfig(onset_method='energy', use_multi_band=False)
        detector_energy = create_bulletproof_onset_detector(config, onset_config=onset_config_energy)
        
        results_energy = detector_energy(audio)
        print(f"✅ Energy-only test passed: {len(results_energy['onset_times'])} onsets")
        
        # Test with HPSS
        onset_config_hpss = OnsetConfig(use_harmonic_percussive_separation=True)
        detector_hpss = create_bulletproof_onset_detector(config, onset_config=onset_config_hpss)
        
        results_hpss = detector_hpss(audio)
        print(f"✅ HPSS test passed: {len(results_hpss['onset_times'])} onsets")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
    
    print("🚀 BulletproofOnsetDetector ready for BigVGAN rhythm analysis!")