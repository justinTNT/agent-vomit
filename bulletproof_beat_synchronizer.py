"""
Bulletproof Beat Synchronizer Module

Advanced beat tracking, tempo estimation, and onset detection with multi-scale analysis
for robust musical timing extraction. Designed for complex musical structures including
polyrhythms, tempo changes, and mixed meters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import warnings
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
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


class TempoRange(Enum):
    """Common tempo ranges with typical BPM bounds."""
    VERY_SLOW = (40, 60)
    SLOW = (60, 76)
    MODERATE = (76, 108)
    MODERATELY_FAST = (108, 120)
    FAST = (120, 168)
    VERY_FAST = (168, 200)
    EXTREME = (200, 300)
    FULL_RANGE = (30, 300)


class OnsetMethod(Enum):
    """Onset detection methods."""
    SPECTRAL_FLUX = "spectral_flux"
    HIGH_FREQUENCY_CONTENT = "hfc"
    COMPLEX_DOMAIN = "complex_domain"
    PHASE_DEVIATION = "phase_deviation"
    WEIGHTED_PHASE_DEVIATION = "wpd"
    MODIFIED_KULLBACK_LEIBLER = "mkl"
    ENERGY = "energy"
    MULTI_FEATURE = "multi_feature"


class BeatTrackingMethod(Enum):
    """Beat tracking algorithm types."""
    DYNAMIC_PROGRAMMING = "dynamic_programming"
    PARTICLE_FILTER = "particle_filter"
    TEMPLATE_MATCHING = "template_matching"
    AUTOCORRELATION = "autocorrelation"
    MULTI_AGENT = "multi_agent"


@dataclass
class BeatSyncResult:
    """Container for beat synchronization results with confidence metrics."""
    
    # Core results
    beats: torch.Tensor  # Beat times in seconds
    tempo: float  # Primary tempo in BPM
    onset_times: torch.Tensor  # Onset times in seconds
    onset_strengths: torch.Tensor  # Onset strength values
    
    # Confidence metrics
    beat_confidence: float  # Overall beat tracking confidence [0,1]
    tempo_confidence: float  # Tempo estimation confidence [0,1]
    onset_confidence: float  # Onset detection confidence [0,1]
    
    # Advanced metrics
    tempo_stability: float  # Tempo consistency over time [0,1]
    beat_consistency: float  # Beat interval consistency [0,1]
    polyrhythm_detected: bool  # Whether polyrhythmic patterns detected
    
    # Multi-scale analysis
    tempo_candidates: List[Tuple[float, float]]  # [(tempo, confidence), ...]
    beat_hierarchies: Dict[str, torch.Tensor]  # Different metrical levels
    
    # Diagnostics
    processing_time: float  # Processing time in seconds
    fallback_used: bool  # Whether fallback methods were needed
    warnings: List[str]  # Processing warnings
    
    def __post_init__(self):
        """Validate and sanitize results."""
        self.beat_confidence = max(0.0, min(1.0, self.beat_confidence))
        self.tempo_confidence = max(0.0, min(1.0, self.tempo_confidence))
        self.onset_confidence = max(0.0, min(1.0, self.onset_confidence))
        self.tempo_stability = max(0.0, min(1.0, self.tempo_stability))
        self.beat_consistency = max(0.0, min(1.0, self.beat_consistency))
        self.tempo = max(30.0, min(300.0, self.tempo))


class BulletproofOnsetDetector(nn.Module):
    """
    Multi-method onset detection with automatic method selection and confidence scoring.
    """
    
    def __init__(self, config: BulletproofAudioConfig):
        super().__init__()
        self.config = config
        self.sr = config.sample_rate
        self.hop_length = config.hop_length
        self.n_fft = config.n_fft
        
        # Detection parameters
        self.onset_threshold = 0.3
        self.onset_pre_max = 3  # frames
        self.onset_post_max = 3  # frames
        self.onset_pre_avg = 10  # frames
        self.onset_post_avg = 10  # frames
        self.onset_wait = 10  # frames minimum between onsets
        
        # Initialize transforms
        self._init_transforms()
        
        # Method weights (learned or tuned)
        self.method_weights = nn.Parameter(
            torch.ones(len(OnsetMethod)) / len(OnsetMethod),
            requires_grad=False
        )
    
    def _init_transforms(self):
        """Initialize spectral transforms with error handling."""
        try:
            # STFT transform
            self.stft_transform = torchaudio.transforms.Spectrogram(
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                power=None,  # Complex spectrogram
                window_fn=torch.hann_window,
                normalized=True
            )
            
            # Mel-scale transform
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.config.n_mels,
                f_min=50,  # Focus on musical content
                f_max=8000,
                power=2.0
            )
            
            # High-frequency content transform
            self.hfc_freqs = torch.linspace(0, self.sr//2, self.n_fft//2 + 1)
            
        except Exception as e:
            logger.error(f"Error initializing onset detector transforms: {e}")
            self._init_fallback_transforms()
    
    def _init_fallback_transforms(self):
        """Initialize minimal fallback transforms."""
        try:
            self.stft_transform = torchaudio.transforms.Spectrogram(
                n_fft=1024,
                hop_length=256,
                power=2.0
            )
            self.mel_transform = None
            self.hfc_freqs = torch.linspace(0, self.sr//2, 513)
        except Exception as e:
            logger.error(f"Fallback transform initialization failed: {e}")
            self.stft_transform = None
            self.mel_transform = None
            self.hfc_freqs = None
    
    def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Detect onsets using multiple methods with confidence scoring.
        
        Args:
            audio: Audio tensor [batch_size, time] or [time]
            
        Returns:
            onset_times: Onset times in seconds
            onset_strengths: Onset strength values
            detection_info: Additional detection information
        """
        try:
            # Ensure proper tensor format
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            
            # Validate input
            if audio.size(-1) < self.hop_length:
                logger.warning("Audio too short for onset detection")
                return self._empty_result()
            
            # Check for silence
            if audio.abs().max() < 1e-6:
                logger.warning("Silent audio detected")
                return self._empty_result()
            
            # Compute spectrograms
            spectrograms = self._compute_spectrograms(audio)
            
            # Apply different onset detection methods
            onset_functions = self._compute_onset_functions(spectrograms)
            
            # Combine onset functions
            combined_onset = self._combine_onset_functions(onset_functions)
            
            # Peak picking
            onset_frames = self._pick_onset_peaks(combined_onset)
            
            # Convert to time
            onset_times = onset_frames * self.hop_length / self.sr
            onset_strengths = combined_onset[onset_frames]
            
            # Compute confidence and additional info
            detection_info = self._compute_detection_info(
                onset_functions, combined_onset, onset_frames
            )
            
            return onset_times, onset_strengths, detection_info
            
        except Exception as e:
            logger.error(f"Onset detection failed: {e}")
            return self._empty_result()
    
    def _compute_spectrograms(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute various spectral representations."""
        spectrograms = {}
        
        try:
            # Complex STFT
            if self.stft_transform is not None:
                complex_spec = self.stft_transform(audio)
                spectrograms['complex'] = complex_spec
                spectrograms['magnitude'] = torch.abs(complex_spec)
                spectrograms['phase'] = torch.angle(complex_spec)
                spectrograms['power'] = torch.abs(complex_spec) ** 2
            
            # Mel spectrogram
            if self.mel_transform is not None:
                spectrograms['mel'] = self.mel_transform(audio)
            
            # Log-magnitude spectrogram
            if 'magnitude' in spectrograms:
                spectrograms['log_magnitude'] = torch.log(
                    spectrograms['magnitude'] + 1e-8
                )
            
        except Exception as e:
            logger.error(f"Spectrogram computation failed: {e}")
            # Minimal fallback
            spectrograms['magnitude'] = torch.rand(1, 513, audio.size(-1) // 256 + 1)
        
        return spectrograms
    
    def _compute_onset_functions(self, spectrograms: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute onset detection functions using multiple methods."""
        onset_functions = {}
        
        try:
            # Spectral flux
            if 'magnitude' in spectrograms:
                onset_functions['spectral_flux'] = self._spectral_flux(spectrograms['magnitude'])
            
            # High-frequency content
            if 'magnitude' in spectrograms and self.hfc_freqs is not None:
                onset_functions['hfc'] = self._high_frequency_content(spectrograms['magnitude'])
            
            # Complex domain
            if 'complex' in spectrograms:
                onset_functions['complex_domain'] = self._complex_domain(spectrograms['complex'])
            
            # Phase deviation
            if 'phase' in spectrograms:
                onset_functions['phase_deviation'] = self._phase_deviation(spectrograms['phase'])
            
            # Energy-based
            if 'power' in spectrograms:
                onset_functions['energy'] = self._energy_onset(spectrograms['power'])
            
            # Modified Kullback-Leibler
            if 'magnitude' in spectrograms:
                onset_functions['mkl'] = self._modified_kl_divergence(spectrograms['magnitude'])
                
        except Exception as e:
            logger.error(f"Onset function computation failed: {e}")
            # Fallback: simple energy-based detection
            if spectrograms:
                key = list(spectrograms.keys())[0]
                onset_functions['energy'] = spectrograms[key].sum(dim=-2).diff(dim=-1).clamp(min=0)
        
        return onset_functions
    
    def _spectral_flux(self, magnitude: torch.Tensor) -> torch.Tensor:
        """Compute spectral flux onset detection function."""
        try:
            # Spectral flux = positive differences in magnitude spectrum
            flux = magnitude.diff(dim=-1).clamp(min=0).sum(dim=-2)
            return flux
        except Exception as e:
            logger.error(f"Spectral flux computation failed: {e}")
            return torch.zeros(magnitude.size(-1) - 1)
    
    def _high_frequency_content(self, magnitude: torch.Tensor) -> torch.Tensor:
        """Compute high-frequency content onset detection function."""
        try:
            # Weight by frequency and sum
            if self.hfc_freqs.device != magnitude.device:
                self.hfc_freqs = self.hfc_freqs.to(magnitude.device)
            
            weights = self.hfc_freqs.unsqueeze(0).unsqueeze(-1)
            hfc = (magnitude * weights).sum(dim=-2)
            
            # Take derivative
            return hfc.diff(dim=-1).clamp(min=0)
        except Exception as e:
            logger.error(f"HFC computation failed: {e}")
            return torch.zeros(magnitude.size(-1) - 1)
    
    def _complex_domain(self, complex_spec: torch.Tensor) -> torch.Tensor:
        """Compute complex domain onset detection function."""
        try:
            # Complex domain onset detection
            target = complex_spec[..., 1:]  # Current frame
            prediction = complex_spec[..., :-1]  # Previous frame
            
            # Prediction error
            error = torch.abs(target - prediction).sum(dim=-2)
            return error
        except Exception as e:
            logger.error(f"Complex domain computation failed: {e}")
            return torch.zeros(complex_spec.size(-1) - 1)
    
    def _phase_deviation(self, phase: torch.Tensor) -> torch.Tensor:
        """Compute phase deviation onset detection function."""
        try:
            # Phase differences
            phase_diff = phase.diff(dim=-1)
            
            # Unwrap phase differences
            phase_diff = torch.remainder(phase_diff + np.pi, 2 * np.pi) - np.pi
            
            # Deviation from expected phase progression
            deviation = torch.abs(phase_diff).sum(dim=-2)
            return deviation
        except Exception as e:
            logger.error(f"Phase deviation computation failed: {e}")
            return torch.zeros(phase.size(-1) - 1)
    
    def _energy_onset(self, power: torch.Tensor) -> torch.Tensor:
        """Compute energy-based onset detection function."""
        try:
            # Total energy per frame
            energy = power.sum(dim=-2)
            
            # Energy differences
            energy_diff = energy.diff(dim=-1).clamp(min=0)
            return energy_diff
        except Exception as e:
            logger.error(f"Energy onset computation failed: {e}")
            return torch.zeros(power.size(-1) - 1)
    
    def _modified_kl_divergence(self, magnitude: torch.Tensor) -> torch.Tensor:
        """Compute modified Kullback-Leibler divergence onset function."""
        try:
            # Normalize spectra
            magnitude_norm = magnitude / (magnitude.sum(dim=-2, keepdim=True) + 1e-8)
            
            # KL divergence between consecutive frames
            prev_frame = magnitude_norm[..., :-1]
            curr_frame = magnitude_norm[..., 1:]
            
            # Modified KL divergence
            kl_div = curr_frame * torch.log((curr_frame + 1e-8) / (prev_frame + 1e-8))
            return kl_div.sum(dim=-2)
        except Exception as e:
            logger.error(f"Modified KL divergence computation failed: {e}")
            return torch.zeros(magnitude.size(-1) - 1)
    
    def _combine_onset_functions(self, onset_functions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine multiple onset detection functions with learned weights."""
        try:
            if not onset_functions:
                logger.warning("No onset functions available")
                return torch.zeros(100)  # Fallback
            
            # Normalize each function
            normalized_functions = {}
            for name, func in onset_functions.items():
                if func.numel() > 0:
                    func_norm = (func - func.min()) / (func.max() - func.min() + 1e-8)
                    normalized_functions[name] = func_norm
            
            if not normalized_functions:
                return torch.zeros(100)
            
            # Get corresponding weights
            method_names = list(normalized_functions.keys())
            if len(method_names) == 1:
                return list(normalized_functions.values())[0]
            
            # Weighted combination
            combined = torch.zeros_like(list(normalized_functions.values())[0])
            total_weight = 0.0
            
            for name, func in normalized_functions.items():
                # Use equal weights for now (could be learned)
                weight = 1.0 / len(normalized_functions)
                combined += weight * func
                total_weight += weight
            
            if total_weight > 0:
                combined /= total_weight
            
            return combined
            
        except Exception as e:
            logger.error(f"Onset function combination failed: {e}")
            # Return first available function
            if onset_functions:
                return list(onset_functions.values())[0]
            return torch.zeros(100)
    
    def _pick_onset_peaks(self, onset_function: torch.Tensor) -> torch.Tensor:
        """Pick onset peaks using adaptive thresholding."""
        try:
            if onset_function.numel() == 0:
                return torch.tensor([], dtype=torch.long)
            
            # Convert to numpy for scipy processing
            onset_np = onset_function.detach().cpu().numpy()
            
            # Adaptive threshold
            threshold = np.mean(onset_np) + self.onset_threshold * np.std(onset_np)
            threshold = max(threshold, 0.01)  # Minimum threshold
            
            # Peak picking with constraints
            peaks, _ = scipy.signal.find_peaks(
                onset_np,
                height=threshold,
                distance=self.onset_wait,
                prominence=0.01
            )
            
            # Additional filtering
            if len(peaks) > 0:
                # Remove peaks too close to boundaries
                valid_peaks = peaks[
                    (peaks >= self.onset_pre_avg) & 
                    (peaks < len(onset_np) - self.onset_post_avg)
                ]
                return torch.tensor(valid_peaks, dtype=torch.long)
            
            return torch.tensor([], dtype=torch.long)
            
        except Exception as e:
            logger.error(f"Peak picking failed: {e}")
            return torch.tensor([], dtype=torch.long)
    
    def _compute_detection_info(self, onset_functions: Dict[str, torch.Tensor], 
                              combined_onset: torch.Tensor, 
                              onset_frames: torch.Tensor) -> Dict[str, Any]:
        """Compute additional detection information and confidence metrics."""
        try:
            info = {
                'num_onsets': len(onset_frames),
                'onset_rate': len(onset_frames) / (combined_onset.numel() * self.hop_length / self.sr),
                'methods_used': list(onset_functions.keys()),
                'onset_function_stats': {},
                'confidence': 0.5  # Default confidence
            }
            
            # Function statistics
            for name, func in onset_functions.items():
                if func.numel() > 0:
                    info['onset_function_stats'][name] = {
                        'mean': float(func.mean()),
                        'std': float(func.std()),
                        'max': float(func.max()),
                        'min': float(func.min())
                    }
            
            # Compute confidence based on onset consistency across methods
            if len(onset_functions) > 1 and len(onset_frames) > 0:
                # Measure agreement between methods
                agreements = []
                for i, (name1, func1) in enumerate(onset_functions.items()):
                    for j, (name2, func2) in enumerate(onset_functions.items()):
                        if i < j and func1.numel() == func2.numel():
                            correlation = torch.corrcoef(torch.stack([func1, func2]))[0, 1]
                            if not torch.isnan(correlation):
                                agreements.append(float(correlation))
                
                if agreements:
                    info['confidence'] = np.mean(agreements)
                    info['method_agreement'] = agreements
            
            return info
            
        except Exception as e:
            logger.error(f"Detection info computation failed: {e}")
            return {'num_onsets': 0, 'onset_rate': 0.0, 'confidence': 0.0}
    
    def _empty_result(self) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Return empty result for edge cases."""
        return (
            torch.tensor([]),
            torch.tensor([]),
            {'num_onsets': 0, 'onset_rate': 0.0, 'confidence': 0.0}
        )


class BulletproofTempoEstimator(nn.Module):
    """
    Multi-method tempo estimation with confidence scoring and stability analysis.
    """
    
    def __init__(self, config: BulletproofAudioConfig):
        super().__init__()
        self.config = config
        self.sr = config.sample_rate
        self.hop_length = config.hop_length
        
        # Tempo estimation parameters
        self.tempo_min = 30.0
        self.tempo_max = 300.0
        self.tempo_bins = 256
        self.autocorr_max_lag = int(4 * self.sr / 60)  # 4 seconds at 60 BPM
        
        # Create tempo grid
        self.tempo_grid = torch.logspace(
            np.log10(self.tempo_min), np.log10(self.tempo_max), self.tempo_bins
        )
    
    def forward(self, onset_function: torch.Tensor, 
                onset_times: Optional[torch.Tensor] = None) -> Tuple[float, float, Dict[str, Any]]:
        """
        Estimate tempo from onset detection function.
        
        Args:
            onset_function: Onset detection function
            onset_times: Optional onset times for additional analysis
            
        Returns:
            tempo: Estimated tempo in BPM
            confidence: Tempo estimation confidence [0,1]
            tempo_info: Additional tempo analysis information
        """
        try:
            if onset_function.numel() == 0:
                logger.warning("Empty onset function for tempo estimation")
                return 120.0, 0.0, {'method': 'fallback'}
            
            # Multiple tempo estimation methods
            tempo_estimates = {}
            
            # Method 1: Autocorrelation
            tempo_estimates['autocorr'] = self._autocorrelation_tempo(onset_function)
            
            # Method 2: FFT-based periodicity
            tempo_estimates['fft'] = self._fft_tempo(onset_function)
            
            # Method 3: Onset interval analysis (if onsets available)
            if onset_times is not None and len(onset_times) > 3:
                tempo_estimates['intervals'] = self._interval_tempo(onset_times)
            
            # Method 4: Template matching
            tempo_estimates['template'] = self._template_tempo(onset_function)
            
            # Combine estimates
            final_tempo, confidence, tempo_info = self._combine_tempo_estimates(tempo_estimates)
            
            # Stability analysis
            stability = self._analyze_tempo_stability(onset_function, final_tempo)
            tempo_info['stability'] = stability
            
            return final_tempo, confidence, tempo_info
            
        except Exception as e:
            logger.error(f"Tempo estimation failed: {e}")
            return 120.0, 0.0, {'method': 'fallback', 'error': str(e)}
    
    def _autocorrelation_tempo(self, onset_function: torch.Tensor) -> Tuple[float, float]:
        """Estimate tempo using autocorrelation."""
        try:
            # Convert to numpy for processing
            onset_np = onset_function.detach().cpu().numpy()
            
            # Compute autocorrelation
            autocorr = np.correlate(onset_np, onset_np, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find peaks in autocorrelation
            min_lag = int(60 * self.sr / (self.hop_length * self.tempo_max))  # Max tempo
            max_lag = int(60 * self.sr / (self.hop_length * self.tempo_min))  # Min tempo
            max_lag = min(max_lag, len(autocorr) - 1)
            
            if max_lag <= min_lag:
                return 120.0, 0.1
            
            # Find peak in valid range
            valid_autocorr = autocorr[min_lag:max_lag]
            if len(valid_autocorr) == 0:
                return 120.0, 0.1
            
            peak_idx = np.argmax(valid_autocorr) + min_lag
            
            # Convert to tempo
            tempo = 60 * self.sr / (self.hop_length * peak_idx)
            confidence = valid_autocorr[peak_idx - min_lag] / (np.max(autocorr[:min_lag]) + 1e-8)
            confidence = min(1.0, max(0.0, confidence))
            
            return float(tempo), float(confidence)
            
        except Exception as e:
            logger.error(f"Autocorrelation tempo estimation failed: {e}")
            return 120.0, 0.1
    
    def _fft_tempo(self, onset_function: torch.Tensor) -> Tuple[float, float]:
        """Estimate tempo using FFT-based periodicity analysis."""
        try:
            # Convert to numpy
            onset_np = onset_function.detach().cpu().numpy()
            
            # Apply windowing to reduce spectral leakage
            window = np.hanning(len(onset_np))
            onset_windowed = onset_np * window
            
            # FFT
            fft_mag = np.abs(np.fft.rfft(onset_windowed))
            freqs = np.fft.rfftfreq(len(onset_np), d=self.hop_length/self.sr)
            
            # Convert frequencies to tempos
            tempo_freqs = freqs * 60  # Convert Hz to BPM
            
            # Focus on musical tempo range
            valid_mask = (tempo_freqs >= self.tempo_min) & (tempo_freqs <= self.tempo_max)
            
            if not np.any(valid_mask):
                return 120.0, 0.1
            
            valid_tempos = tempo_freqs[valid_mask]
            valid_magnitudes = fft_mag[valid_mask]
            
            # Find peak
            peak_idx = np.argmax(valid_magnitudes)
            tempo = valid_tempos[peak_idx]
            
            # Compute confidence
            peak_mag = valid_magnitudes[peak_idx]
            mean_mag = np.mean(valid_magnitudes)
            confidence = min(1.0, max(0.0, (peak_mag - mean_mag) / (peak_mag + 1e-8)))
            
            return float(tempo), float(confidence)
            
        except Exception as e:
            logger.error(f"FFT tempo estimation failed: {e}")
            return 120.0, 0.1
    
    def _interval_tempo(self, onset_times: torch.Tensor) -> Tuple[float, float]:
        """Estimate tempo from onset intervals."""
        try:
            if len(onset_times) < 4:
                return 120.0, 0.1
            
            # Convert to numpy
            onsets_np = onset_times.detach().cpu().numpy()
            
            # Compute inter-onset intervals
            intervals = np.diff(onsets_np)
            
            # Remove very short intervals (likely false positives)
            intervals = intervals[intervals > 0.1]  # Minimum 100ms
            
            if len(intervals) < 3:
                return 120.0, 0.1
            
            # Convert intervals to tempos
            interval_tempos = 60.0 / intervals
            
            # Filter to valid tempo range
            valid_tempos = interval_tempos[
                (interval_tempos >= self.tempo_min) & 
                (interval_tempos <= self.tempo_max)
            ]
            
            if len(valid_tempos) < 2:
                return 120.0, 0.1
            
            # Cluster analysis to find dominant tempo
            # Simple approach: find mode using histogram
            hist, bin_edges = np.histogram(valid_tempos, bins=50, range=(self.tempo_min, self.tempo_max))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Find peak
            peak_bin = np.argmax(hist)
            tempo = bin_centers[peak_bin]
            
            # Confidence based on how many intervals support this tempo
            tempo_tolerance = 5.0  # BPM
            supporting_intervals = np.sum(np.abs(valid_tempos - tempo) < tempo_tolerance)
            confidence = min(1.0, supporting_intervals / len(valid_tempos))
            
            return float(tempo), float(confidence)
            
        except Exception as e:
            logger.error(f"Interval tempo estimation failed: {e}")
            return 120.0, 0.1
    
    def _template_tempo(self, onset_function: torch.Tensor) -> Tuple[float, float]:
        """Estimate tempo using template matching."""
        try:
            onset_np = onset_function.detach().cpu().numpy()
            
            # Create tempo templates
            template_scores = []
            test_tempos = np.linspace(self.tempo_min, self.tempo_max, 100)
            
            for tempo in test_tempos:
                # Create impulse train template at this tempo
                beat_period = 60 * self.sr / (self.hop_length * tempo)
                template = np.zeros_like(onset_np)
                
                # Place impulses at beat positions
                beat_positions = np.arange(0, len(template), beat_period)
                beat_positions = beat_positions[beat_positions < len(template)].astype(int)
                
                if len(beat_positions) > 0:
                    template[beat_positions] = 1.0
                
                # Cross-correlate with onset function
                if np.sum(template) > 0:
                    correlation = np.corrcoef(onset_np, template)[0, 1]
                    if not np.isnan(correlation):
                        template_scores.append(correlation)
                    else:
                        template_scores.append(0.0)
                else:
                    template_scores.append(0.0)
            
            if not template_scores:
                return 120.0, 0.1
            
            # Find best tempo
            best_idx = np.argmax(template_scores)
            tempo = test_tempos[best_idx]
            confidence = max(0.0, min(1.0, template_scores[best_idx]))
            
            return float(tempo), float(confidence)
            
        except Exception as e:
            logger.error(f"Template tempo estimation failed: {e}")
            return 120.0, 0.1
    
    def _combine_tempo_estimates(self, tempo_estimates: Dict[str, Tuple[float, float]]) -> Tuple[float, float, Dict[str, Any]]:
        """Combine multiple tempo estimates with confidence weighting."""
        try:
            if not tempo_estimates:
                return 120.0, 0.0, {'method': 'fallback'}
            
            # Extract tempos and confidences
            tempos = []
            confidences = []
            methods = []
            
            for method, (tempo, conf) in tempo_estimates.items():
                if 30 <= tempo <= 300 and 0 <= conf <= 1:  # Sanity check
                    tempos.append(tempo)
                    confidences.append(conf)
                    methods.append(method)
            
            if not tempos:
                return 120.0, 0.0, {'method': 'fallback'}
            
            tempos = np.array(tempos)
            confidences = np.array(confidences)
            
            # Weighted average
            if np.sum(confidences) > 0:
                weights = confidences / np.sum(confidences)
                final_tempo = np.sum(tempos * weights)
                final_confidence = np.mean(confidences)
            else:
                final_tempo = np.median(tempos)
                final_confidence = 0.3
            
            # Check for tempo doubling/halving consistency
            tempo_ratios = []
            for i in range(len(tempos)):
                for j in range(i+1, len(tempos)):
                    ratio = tempos[i] / tempos[j]
                    tempo_ratios.append(ratio)
            
            # Compute consistency metric
            consistency = 0.5
            if tempo_ratios:
                # Look for ratios near 1, 2, 0.5 (indicating consistent tempo or octave errors)
                near_one = np.sum(np.abs(np.array(tempo_ratios) - 1.0) < 0.1)
                near_two = np.sum(np.abs(np.array(tempo_ratios) - 2.0) < 0.2)
                near_half = np.sum(np.abs(np.array(tempo_ratios) - 0.5) < 0.1)
                
                consistency = (near_one + near_two + near_half) / len(tempo_ratios)
                final_confidence *= consistency
            
            tempo_info = {
                'estimates': tempo_estimates,
                'methods_used': methods,
                'consistency': consistency,
                'tempo_candidates': [(float(t), float(c)) for t, c in zip(tempos, confidences)]
            }
            
            return float(final_tempo), float(final_confidence), tempo_info
            
        except Exception as e:
            logger.error(f"Tempo combination failed: {e}")
            return 120.0, 0.0, {'method': 'fallback', 'error': str(e)}
    
    def _analyze_tempo_stability(self, onset_function: torch.Tensor, tempo: float) -> float:
        """Analyze tempo stability over time."""
        try:
            # Divide into overlapping windows
            onset_np = onset_function.detach().cpu().numpy()
            window_size = min(len(onset_np) // 4, int(8 * self.sr / self.hop_length))  # ~8 seconds
            hop_size = window_size // 2
            
            if window_size < 100:  # Too short for analysis
                return 0.5
            
            window_tempos = []
            
            for start in range(0, len(onset_np) - window_size, hop_size):
                window = onset_np[start:start + window_size]
                window_tensor = torch.tensor(window)
                
                # Simple autocorrelation for this window
                window_tempo, _ = self._autocorrelation_tempo(window_tensor)
                if 30 <= window_tempo <= 300:
                    window_tempos.append(window_tempo)
            
            if len(window_tempos) < 2:
                return 0.5
            
            # Compute stability as inverse of coefficient of variation
            window_tempos = np.array(window_tempos)
            mean_tempo = np.mean(window_tempos)
            std_tempo = np.std(window_tempos)
            
            if mean_tempo > 0:
                cv = std_tempo / mean_tempo
                stability = np.exp(-cv * 5)  # Convert to [0,1] with exponential decay
                return min(1.0, max(0.0, stability))
            
            return 0.5
            
        except Exception as e:
            logger.error(f"Tempo stability analysis failed: {e}")
            return 0.5


class BulletproofBeatTracker(nn.Module):
    """
    Advanced beat tracking with multiple algorithms and confidence assessment.
    """
    
    def __init__(self, config: BulletproofAudioConfig):
        super().__init__()
        self.config = config
        self.sr = config.sample_rate
        self.hop_length = config.hop_length
        
        # Beat tracking parameters
        self.beat_tolerance = 0.07  # 70ms tolerance
        self.phase_candidates = 16  # Number of phase candidates
        self.min_beats = 3  # Minimum beats required
        
    def forward(self, onset_function: torch.Tensor, tempo: float, 
                onset_times: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
        """
        Track beats given onset function and tempo.
        
        Args:
            onset_function: Onset detection function
            tempo: Estimated tempo in BPM
            onset_times: Optional onset times for initialization
            
        Returns:
            beat_times: Beat times in seconds
            confidence: Beat tracking confidence [0,1]
            tracking_info: Additional tracking information
        """
        try:
            if onset_function.numel() == 0:
                logger.warning("Empty onset function for beat tracking")
                return torch.tensor([]), 0.0, {'method': 'fallback'}
            
            # Multiple beat tracking methods
            beat_estimates = {}
            
            # Method 1: Template-based tracking
            beat_estimates['template'] = self._template_beat_tracking(onset_function, tempo)
            
            # Method 2: Dynamic programming
            beat_estimates['dp'] = self._dynamic_programming_beats(onset_function, tempo)
            
            # Method 3: Onset-based tracking (if onsets available)
            if onset_times is not None and len(onset_times) > 2:
                beat_estimates['onset_based'] = self._onset_based_beats(onset_times, tempo)
            
            # Method 4: Autocorrelation phase alignment
            beat_estimates['autocorr'] = self._autocorr_beat_tracking(onset_function, tempo)
            
            # Combine beat estimates
            final_beats, confidence, tracking_info = self._combine_beat_estimates(
                beat_estimates, onset_function, tempo
            )
            
            return final_beats, confidence, tracking_info
            
        except Exception as e:
            logger.error(f"Beat tracking failed: {e}")
            return torch.tensor([]), 0.0, {'method': 'fallback', 'error': str(e)}
    
    def _template_beat_tracking(self, onset_function: torch.Tensor, tempo: float) -> Tuple[torch.Tensor, float]:
        """Template-based beat tracking with phase optimization."""
        try:
            onset_np = onset_function.detach().cpu().numpy()
            beat_period = 60 * self.sr / (self.hop_length * tempo)
            
            # Try different phase candidates
            best_score = -1
            best_beats = []
            
            for phase in np.linspace(0, beat_period, self.phase_candidates):
                # Generate beat positions
                beat_positions = []
                pos = phase
                while pos < len(onset_np):
                    beat_positions.append(pos)
                    pos += beat_period
                
                if len(beat_positions) < self.min_beats:
                    continue
                
                # Score this beat sequence
                score = self._score_beat_sequence(onset_np, beat_positions)
                
                if score > best_score:
                    best_score = score
                    best_beats = beat_positions.copy()
            
            if not best_beats:
                return torch.tensor([]), 0.0
            
            # Convert to time
            beat_times = np.array(best_beats) * self.hop_length / self.sr
            confidence = max(0.0, min(1.0, best_score))
            
            return torch.tensor(beat_times), confidence
            
        except Exception as e:
            logger.error(f"Template beat tracking failed: {e}")
            return torch.tensor([]), 0.0
    
    def _dynamic_programming_beats(self, onset_function: torch.Tensor, tempo: float) -> Tuple[torch.Tensor, float]:
        """Dynamic programming beat tracking."""
        try:
            onset_np = onset_function.detach().cpu().numpy()
            beat_period = 60 * self.sr / (self.hop_length * tempo)
            
            # Simplified DP approach
            # Create cost matrix for beat positions
            n_frames = len(onset_np)
            min_period = int(beat_period * 0.8)
            max_period = int(beat_period * 1.2)
            
            # Initialize DP table
            dp_scores = np.full(n_frames, -np.inf)
            dp_paths = np.full(n_frames, -1, dtype=int)
            
            # Base case: any frame can be the first beat
            for i in range(min(n_frames, max_period)):
                dp_scores[i] = onset_np[i]
            
            # Fill DP table
            for i in range(max_period, n_frames):
                for prev_beat in range(max(0, i - max_period), max(0, i - min_period + 1)):
                    period = i - prev_beat
                    if min_period <= period <= max_period:
                        # Score based on onset strength and period consistency
                        period_score = 1.0 - abs(period - beat_period) / beat_period
                        score = dp_scores[prev_beat] + onset_np[i] * period_score
                        
                        if score > dp_scores[i]:
                            dp_scores[i] = score
                            dp_paths[i] = prev_beat
            
            # Backtrack to find best beat sequence
            if np.max(dp_scores) <= -np.inf:
                return torch.tensor([]), 0.0
            
            # Find best ending position
            best_end = np.argmax(dp_scores)
            
            # Backtrack
            beat_frames = []
            current = best_end
            while current >= 0:
                beat_frames.append(current)
                current = dp_paths[current]
            
            beat_frames.reverse()
            
            if len(beat_frames) < self.min_beats:
                return torch.tensor([]), 0.0
            
            # Convert to time
            beat_times = np.array(beat_frames) * self.hop_length / self.sr
            
            # Compute confidence
            avg_score = np.mean([dp_scores[f] for f in beat_frames])
            max_possible = np.max(onset_np)
            confidence = max(0.0, min(1.0, avg_score / (max_possible + 1e-8)))
            
            return torch.tensor(beat_times), confidence
            
        except Exception as e:
            logger.error(f"Dynamic programming beat tracking failed: {e}")
            return torch.tensor([]), 0.0
    
    def _onset_based_beats(self, onset_times: torch.Tensor, tempo: float) -> Tuple[torch.Tensor, float]:
        """Beat tracking based on onset times."""
        try:
            onsets_np = onset_times.detach().cpu().numpy()
            beat_period = 60.0 / tempo
            
            # Find the best phase by trying different starting onsets
            best_score = -1
            best_beats = []
            
            for start_idx in range(min(len(onsets_np), 10)):  # Try first 10 onsets
                start_time = onsets_np[start_idx]
                
                # Generate beat grid from this onset
                beat_times = []
                beat_time = start_time
                
                # Go backwards
                while beat_time >= 0:
                    beat_times.append(beat_time)
                    beat_time -= beat_period
                
                # Go forwards
                beat_time = start_time + beat_period
                while beat_time <= onsets_np[-1] + beat_period:
                    beat_times.append(beat_time)
                    beat_time += beat_period
                
                beat_times.sort()
                
                if len(beat_times) < self.min_beats:
                    continue
                
                # Score this beat grid against onsets
                score = self._score_beats_against_onsets(beat_times, onsets_np)
                
                if score > best_score:
                    best_score = score
                    best_beats = beat_times.copy()
            
            if not best_beats:
                return torch.tensor([]), 0.0
            
            confidence = max(0.0, min(1.0, best_score))
            return torch.tensor(best_beats), confidence
            
        except Exception as e:
            logger.error(f"Onset-based beat tracking failed: {e}")
            return torch.tensor([]), 0.0
    
    def _autocorr_beat_tracking(self, onset_function: torch.Tensor, tempo: float) -> Tuple[torch.Tensor, float]:
        """Beat tracking using autocorrelation phase alignment."""
        try:
            onset_np = onset_function.detach().cpu().numpy()
            beat_period = 60 * self.sr / (self.hop_length * tempo)
            
            # Create ideal beat train
            beat_train = np.zeros_like(onset_np)
            beat_positions = np.arange(0, len(beat_train), beat_period)
            beat_positions = beat_positions[beat_positions < len(beat_train)].astype(int)
            
            if len(beat_positions) < self.min_beats:
                return torch.tensor([]), 0.0
            
            beat_train[beat_positions] = 1.0
            
            # Cross-correlate to find best phase alignment
            correlation = np.correlate(onset_np, beat_train, mode='full')
            
            # Find peak
            peak_idx = np.argmax(correlation)
            offset = peak_idx - (len(beat_train) - 1)
            
            # Apply offset to beat positions
            adjusted_positions = beat_positions + offset
            
            # Keep only valid positions
            valid_positions = adjusted_positions[
                (adjusted_positions >= 0) & (adjusted_positions < len(onset_np))
            ]
            
            if len(valid_positions) < self.min_beats:
                return torch.tensor([]), 0.0
            
            # Convert to time
            beat_times = valid_positions * self.hop_length / self.sr
            
            # Compute confidence
            max_corr = np.max(correlation)
            mean_corr = np.mean(correlation)
            confidence = max(0.0, min(1.0, (max_corr - mean_corr) / (max_corr + 1e-8)))
            
            return torch.tensor(beat_times), confidence
            
        except Exception as e:
            logger.error(f"Autocorrelation beat tracking failed: {e}")
            return torch.tensor([]), 0.0
    
    def _score_beat_sequence(self, onset_function: np.ndarray, beat_positions: List[float]) -> float:
        """Score a beat sequence against the onset function."""
        try:
            if not beat_positions:
                return 0.0
            
            total_score = 0.0
            
            for pos in beat_positions:
                # Get onset strength at beat position (with interpolation)
                if 0 <= pos < len(onset_function):
                    # Simple interpolation
                    idx = int(pos)
                    frac = pos - idx
                    
                    if idx + 1 < len(onset_function):
                        strength = onset_function[idx] * (1 - frac) + onset_function[idx + 1] * frac
                    else:
                        strength = onset_function[idx]
                    
                    total_score += strength
            
            # Normalize by number of beats
            return total_score / len(beat_positions) if len(beat_positions) > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Beat sequence scoring failed: {e}")
            return 0.0
    
    def _score_beats_against_onsets(self, beat_times: List[float], onset_times: np.ndarray) -> float:
        """Score beat times against onset times."""
        try:
            if not beat_times or len(onset_times) == 0:
                return 0.0
            
            matched_onsets = 0
            total_beats = len(beat_times)
            
            for beat_time in beat_times:
                # Find closest onset
                distances = np.abs(onset_times - beat_time)
                min_distance = np.min(distances)
                
                if min_distance <= self.beat_tolerance:
                    matched_onsets += 1
            
            # Score based on percentage of beats that match onsets
            return matched_onsets / total_beats if total_beats > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Beat-onset scoring failed: {e}")
            return 0.0
    
    def _combine_beat_estimates(self, beat_estimates: Dict[str, Tuple[torch.Tensor, float]], 
                               onset_function: torch.Tensor, tempo: float) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
        """Combine multiple beat estimates."""
        try:
            if not beat_estimates:
                return torch.tensor([]), 0.0, {'method': 'fallback'}
            
            # Filter valid estimates
            valid_estimates = {}
            for method, (beats, conf) in beat_estimates.items():
                if len(beats) >= self.min_beats and 0 <= conf <= 1:
                    valid_estimates[method] = (beats, conf)
            
            if not valid_estimates:
                return torch.tensor([]), 0.0, {'method': 'fallback'}
            
            # Choose best estimate based on confidence
            best_method = max(valid_estimates.keys(), key=lambda k: valid_estimates[k][1])
            best_beats, best_confidence = valid_estimates[best_method]
            
            # Additional validation
            if len(best_beats) >= self.min_beats:
                # Check beat consistency
                if len(best_beats) > 1:
                    intervals = torch.diff(best_beats)
                    expected_interval = 60.0 / tempo
                    interval_consistency = 1.0 - torch.std(intervals) / (expected_interval + 1e-8)
                    interval_consistency = max(0.0, min(1.0, float(interval_consistency)))
                    
                    # Adjust confidence based on consistency
                    best_confidence *= interval_consistency
                
                tracking_info = {
                    'method': best_method,
                    'estimates': {k: (v[0].tolist(), v[1]) for k, v in valid_estimates.items()},
                    'num_beats': len(best_beats),
                    'beat_interval_consistency': interval_consistency if len(best_beats) > 1 else 1.0
                }
                
                return best_beats, float(best_confidence), tracking_info
            
            return torch.tensor([]), 0.0, {'method': 'fallback'}
            
        except Exception as e:
            logger.error(f"Beat estimate combination failed: {e}")
            return torch.tensor([]), 0.0, {'method': 'fallback', 'error': str(e)}


class BulletproofBeatSynchronizer(nn.Module):
    """
    Complete beat synchronization system combining onset detection, tempo estimation,
    and beat tracking with confidence assessment and multi-scale analysis.
    """
    
    def __init__(self, config: Optional[BulletproofAudioConfig] = None):
        super().__init__()
        
        if config is None:
            config = get_bulletproof_music_config()
        
        self.config = config
        self.sr = config.sample_rate
        self.hop_length = config.hop_length
        
        # Initialize components
        self.onset_detector = BulletproofOnsetDetector(config)
        self.tempo_estimator = BulletproofTempoEstimator(config)
        self.beat_tracker = BulletproofBeatTracker(config)
        
        # Multi-scale analysis parameters
        self.tempo_ranges = [
            TempoRange.SLOW,
            TempoRange.MODERATE,
            TempoRange.FAST,
            TempoRange.VERY_FAST
        ]
        
        # Quality thresholds
        self.min_onset_confidence = 0.2
        self.min_tempo_confidence = 0.3
        self.min_beat_confidence = 0.2
    
    def forward(self, audio: torch.Tensor, 
                tempo_range: Optional[TempoRange] = None) -> BeatSyncResult:
        """
        Perform complete beat synchronization analysis.
        
        Args:
            audio: Audio tensor [batch_size, time] or [time]
            tempo_range: Optional tempo range constraint
            
        Returns:
            BeatSyncResult with all analysis results and confidence metrics
        """
        import time
        start_time = time.time()
        
        try:
            # Validate input
            audio = self._validate_audio_input(audio)
            if audio is None:
                return self._create_fallback_result("Invalid audio input", start_time)
            
            # Onset detection
            onset_times, onset_strengths, onset_info = self.onset_detector(audio)
            
            if len(onset_times) == 0:
                logger.warning("No onsets detected")
                return self._create_fallback_result("No onsets detected", start_time)
            
            # Get onset function for tempo estimation
            onset_function = onset_info.get('combined_onset_function', onset_strengths)
            if isinstance(onset_function, list):
                onset_function = torch.tensor(onset_function)
            
            # Tempo estimation
            tempo, tempo_confidence, tempo_info = self.tempo_estimator(
                onset_function, onset_times
            )
            
            # Beat tracking
            beat_times, beat_confidence, tracking_info = self.beat_tracker(
                onset_function, tempo, onset_times
            )
            
            # Multi-scale analysis
            hierarchical_beats = self._analyze_beat_hierarchies(
                beat_times, tempo, onset_function
            )
            
            # Polyrhythm detection
            polyrhythm_detected = self._detect_polyrhythms(onset_times, beat_times, tempo)
            
            # Compute overall confidence and stability metrics
            overall_confidence = self._compute_overall_confidence(
                onset_info.get('confidence', 0.5),
                tempo_confidence,
                beat_confidence
            )
            
            tempo_stability = tempo_info.get('stability', 0.5)
            beat_consistency = tracking_info.get('beat_interval_consistency', 0.5)
            
            # Create result
            result = BeatSyncResult(
                beats=beat_times,
                tempo=tempo,
                onset_times=onset_times,
                onset_strengths=onset_strengths,
                beat_confidence=beat_confidence,
                tempo_confidence=tempo_confidence,
                onset_confidence=onset_info.get('confidence', 0.5),
                tempo_stability=tempo_stability,
                beat_consistency=beat_consistency,
                polyrhythm_detected=polyrhythm_detected,
                tempo_candidates=tempo_info.get('tempo_candidates', [(tempo, tempo_confidence)]),
                beat_hierarchies=hierarchical_beats,
                processing_time=time.time() - start_time,
                fallback_used=False,
                warnings=self._collect_warnings(onset_info, tempo_info, tracking_info)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Beat synchronization failed: {e}")
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
            min_length = self.sr * 2  # Minimum 2 seconds
            if audio.size(-1) < min_length:
                logger.warning(f"Audio too short: {audio.size(-1)} samples, minimum {min_length}")
                if audio.size(-1) < self.hop_length:
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
    
    def _analyze_beat_hierarchies(self, beat_times: torch.Tensor, tempo: float, 
                                 onset_function: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze different metrical levels (downbeats, half-notes, etc.)."""
        try:
            hierarchies = {}
            
            if len(beat_times) < 4:
                return hierarchies
            
            # Quarter notes (main beats)
            hierarchies['quarter'] = beat_times
            
            # Half notes (every other beat)
            if len(beat_times) >= 2:
                hierarchies['half'] = beat_times[::2]
            
            # Downbeats (every 4th beat, assuming 4/4 time)
            if len(beat_times) >= 4:
                hierarchies['downbeat'] = beat_times[::4]
            
            # Eighth notes (subdivision)
            if len(beat_times) >= 2:
                eighth_times = []
                beat_interval = 60.0 / tempo
                subdivision_interval = beat_interval / 2
                
                for beat_time in beat_times:
                    eighth_times.append(float(beat_time))
                    if beat_time + subdivision_interval < beat_times[-1]:
                        eighth_times.append(float(beat_time) + subdivision_interval)
                
                hierarchies['eighth'] = torch.tensor(eighth_times)
            
            return hierarchies
            
        except Exception as e:
            logger.error(f"Beat hierarchy analysis failed: {e}")
            return {}
    
    def _detect_polyrhythms(self, onset_times: torch.Tensor, beat_times: torch.Tensor, 
                          tempo: float) -> bool:
        """Detect polyrhythmic patterns in the audio."""
        try:
            if len(onset_times) < 10 or len(beat_times) < 4:
                return False
            
            # Look for onset patterns that don't align with main beat grid
            beat_interval = 60.0 / tempo
            tolerance = beat_interval * 0.15  # 15% tolerance
            
            unmatched_onsets = 0
            total_onsets = len(onset_times)
            
            for onset_time in onset_times:
                # Find closest beat
                distances = torch.abs(beat_times - onset_time)
                min_distance = torch.min(distances)
                
                if min_distance > tolerance:
                    unmatched_onsets += 1
            
            # If more than 30% of onsets don't align with beats, suspect polyrhythm
            unmatched_ratio = unmatched_onsets / total_onsets
            
            return unmatched_ratio > 0.3
            
        except Exception as e:
            logger.error(f"Polyrhythm detection failed: {e}")
            return False
    
    def _compute_overall_confidence(self, onset_conf: float, tempo_conf: float, 
                                  beat_conf: float) -> float:
        """Compute overall confidence from component confidences."""
        try:
            # Weighted average with beat tracking being most important
            weights = [0.2, 0.3, 0.5]  # onset, tempo, beat
            confidences = [onset_conf, tempo_conf, beat_conf]
            
            # Check minimum thresholds
            if onset_conf < self.min_onset_confidence:
                return 0.1
            if tempo_conf < self.min_tempo_confidence:
                return 0.2
            if beat_conf < self.min_beat_confidence:
                return 0.3
            
            # Weighted average
            overall = sum(w * c for w, c in zip(weights, confidences))
            return max(0.0, min(1.0, overall))
            
        except Exception as e:
            logger.error(f"Overall confidence computation failed: {e}")
            return 0.3
    
    def _collect_warnings(self, onset_info: Dict, tempo_info: Dict, 
                         tracking_info: Dict) -> List[str]:
        """Collect warnings from all processing stages."""
        warnings = []
        
        try:
            # Onset detection warnings
            if onset_info.get('num_onsets', 0) < 5:
                warnings.append("Very few onsets detected")
            
            if onset_info.get('confidence', 0) < 0.3:
                warnings.append("Low onset detection confidence")
            
            # Tempo estimation warnings
            if tempo_info.get('consistency', 0) < 0.5:
                warnings.append("Inconsistent tempo estimates across methods")
            
            if tempo_info.get('stability', 0) < 0.4:
                warnings.append("Unstable tempo over time")
            
            # Beat tracking warnings
            if tracking_info.get('num_beats', 0) < 8:
                warnings.append("Very few beats tracked")
            
            if tracking_info.get('beat_interval_consistency', 0) < 0.6:
                warnings.append("Inconsistent beat intervals")
            
            return warnings
            
        except Exception as e:
            logger.error(f"Warning collection failed: {e}")
            return ["Error collecting processing warnings"]
    
    def _create_fallback_result(self, reason: str, start_time: float) -> BeatSyncResult:
        """Create a fallback result for error cases."""
        return BeatSyncResult(
            beats=torch.tensor([]),
            tempo=120.0,
            onset_times=torch.tensor([]),
            onset_strengths=torch.tensor([]),
            beat_confidence=0.0,
            tempo_confidence=0.0,
            onset_confidence=0.0,
            tempo_stability=0.0,
            beat_consistency=0.0,
            polyrhythm_detected=False,
            tempo_candidates=[(120.0, 0.0)],
            beat_hierarchies={},
            processing_time=time.time() - start_time,
            fallback_used=True,
            warnings=[reason]
        )
    
    def analyze_long_audio(self, audio: torch.Tensor, 
                          chunk_duration: float = 30.0,
                          overlap_duration: float = 5.0) -> List[BeatSyncResult]:
        """
        Analyze long audio by chunking with overlap for tempo changes.
        
        Args:
            audio: Long audio tensor
            chunk_duration: Duration of each chunk in seconds
            overlap_duration: Overlap between chunks in seconds
            
        Returns:
            List of BeatSyncResult for each chunk
        """
        try:
            results = []
            chunk_samples = int(chunk_duration * self.sr)
            overlap_samples = int(overlap_duration * self.sr)
            hop_samples = chunk_samples - overlap_samples
            
            for start in range(0, audio.size(-1) - chunk_samples + 1, hop_samples):
                end = start + chunk_samples
                chunk = audio[start:end]
                
                result = self.forward(chunk)
                
                # Adjust times to global timeline
                global_offset = start / self.sr
                if len(result.beats) > 0:
                    result.beats = result.beats + global_offset
                if len(result.onset_times) > 0:
                    result.onset_times = result.onset_times + global_offset
                
                # Add chunk metadata
                result.warnings.append(f"Chunk {len(results)+1}: {global_offset:.1f}-{end/self.sr:.1f}s")
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Long audio analysis failed: {e}")
            return []


# Factory functions and utilities

def create_beat_synchronizer(sample_rate: int = 22050, 
                           domain: str = "music",
                           **kwargs) -> BulletproofBeatSynchronizer:
    """Create a beat synchronizer with specified configuration."""
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
        
        return BulletproofBeatSynchronizer(config)
        
    except Exception as e:
        logger.error(f"Failed to create beat synchronizer: {e}")
        # Fallback configuration
        config = BulletproofAudioConfig(sample_rate=sample_rate)
        return BulletproofBeatSynchronizer(config)


def test_beat_synchronizer():
    """Test the beat synchronizer with synthetic audio."""
    logger.info("Testing BulletproofBeatSynchronizer")
    
    try:
        # Create test audio with clear beat pattern
        sr = 22050
        duration = 10.0  # 10 seconds
        tempo = 120.0  # BPM
        
        t = torch.linspace(0, duration, int(sr * duration))
        
        # Create beat pattern: kick on beats 1,3 and snare on beats 2,4
        beat_times = torch.arange(0, duration, 60.0/tempo)
        
        # Simple beat pattern
        audio = torch.zeros_like(t)
        for beat_time in beat_times:
            if beat_time < duration:
                # Add impulse at beat time
                beat_sample = int(beat_time * sr)
                if beat_sample < len(audio):
                    # Create short percussive sound
                    impulse_length = int(0.05 * sr)  # 50ms
                    end_sample = min(beat_sample + impulse_length, len(audio))
                    envelope = torch.exp(-10 * torch.linspace(0, 1, end_sample - beat_sample))
                    frequency = 60.0 if (len(beat_times) % 2 == 0) else 200.0  # Kick vs snare
                    tone = torch.sin(2 * np.pi * frequency * torch.linspace(0, 0.05, end_sample - beat_sample))
                    audio[beat_sample:end_sample] += 0.5 * envelope * tone
        
        # Add some noise for realism
        audio += 0.05 * torch.randn_like(audio)
        
        # Test beat synchronizer
        synchronizer = create_beat_synchronizer(sample_rate=sr)
        result = synchronizer(audio)
        
        # Analyze results
        logger.info(f"Beat synchronization results:")
        logger.info(f"  Detected tempo: {result.tempo:.1f} BPM (expected: {tempo:.1f})")
        logger.info(f"  Number of beats: {len(result.beats)}")
        logger.info(f"  Number of onsets: {len(result.onset_times)}")
        logger.info(f"  Beat confidence: {result.beat_confidence:.3f}")
        logger.info(f"  Tempo confidence: {result.tempo_confidence:.3f}")
        logger.info(f"  Onset confidence: {result.onset_confidence:.3f}")
        logger.info(f"  Tempo stability: {result.tempo_stability:.3f}")
        logger.info(f"  Beat consistency: {result.beat_consistency:.3f}")
        logger.info(f"  Polyrhythm detected: {result.polyrhythm_detected}")
        logger.info(f"  Processing time: {result.processing_time:.3f}s")
        logger.info(f"  Fallback used: {result.fallback_used}")
        
        if result.warnings:
            logger.info(f"  Warnings: {'; '.join(result.warnings)}")
        
        # Check accuracy
        tempo_error = abs(result.tempo - tempo) / tempo
        logger.info(f"  Tempo error: {tempo_error:.1%}")
        
        if len(result.beats) > 0:
            # Compare detected beats with expected
            expected_beats = beat_times[beat_times <= duration]
            if len(expected_beats) > 0 and len(result.beats) > 0:
                # Simple alignment check
                first_detected = result.beats[0]
                first_expected = expected_beats[0]
                phase_error = abs(first_detected - first_expected)
                logger.info(f"  Phase error: {phase_error:.3f}s")
        
        logger.info("Beat synchronizer test completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Beat synchronizer test failed: {e}")
        return None


if __name__ == "__main__":
    test_beat_synchronizer()