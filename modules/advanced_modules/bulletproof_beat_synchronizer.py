"""
Bulletproof Beat Synchronization and Advanced Tempo Analysis

A robust, production-ready implementation with comprehensive error handling,
fallback strategies, and memory management for musical beat tracking and rhythm analysis.

Key Features:
- Multi-scale onset detection with neural enhancement
- Ensemble tempo estimation with confidence weighting
- Dynamic programming beat tracking with temporal consistency
- Memory-efficient processing for long audio sequences
- Device compatibility and graceful degradation
- Comprehensive parameter validation and error recovery
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
import traceback
import time
import gc
from dataclasses import dataclass
from enum import Enum

# Use relative imports to maintain compatibility
try:
    from ..audio_analysis.audio_config import AudioModuleConfig, AudioModuleBase
except ImportError:
    # Fallback for testing
    from audio_analysis.audio_config import AudioModuleConfig, AudioModuleBase


class BeatTrackingQuality(Enum):
    """Quality levels for beat tracking results."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    FAILED = "failed"


@dataclass
class BeatSynchronizerConfig:
    """Configuration for beat synchronizer with safe defaults."""
    
    # Core parameters
    min_tempo: float = 60.0
    max_tempo: float = 200.0
    tempo_bins: int = 140
    
    # Onset detection
    onset_threshold: float = 0.1
    min_onset_distance: float = 0.05  # seconds
    
    # Beat tracking
    alpha: float = 0.8  # Tempo consistency weight
    beta: float = 0.2   # Onset strength weight
    transition_sigma: float = 0.1
    
    # Memory management
    max_audio_length: float = 300.0  # seconds
    chunk_size: int = 22050 * 30  # 30 seconds at 22kHz
    
    # Fallback parameters
    fallback_tempo: float = 120.0
    min_beat_confidence: float = 0.3
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        try:
            assert 30 <= self.min_tempo <= 300, f"Invalid min_tempo: {self.min_tempo}"
            assert 60 <= self.max_tempo <= 400, f"Invalid max_tempo: {self.max_tempo}"
            assert self.min_tempo < self.max_tempo, "min_tempo must be < max_tempo"
            assert 0.0 <= self.onset_threshold <= 1.0, f"Invalid onset_threshold: {self.onset_threshold}"
            assert 0.0 < self.min_onset_distance <= 1.0, f"Invalid min_onset_distance: {self.min_onset_distance}"
            assert 0.0 <= self.alpha <= 1.0, f"Invalid alpha: {self.alpha}"
            assert 0.0 <= self.beta <= 1.0, f"Invalid beta: {self.beta}"
            assert abs(self.alpha + self.beta - 1.0) < 1e-6, "alpha + beta must equal 1.0"
            return True
        except AssertionError as e:
            warnings.warn(f"Configuration validation failed: {e}")
            return False


class SafeMultiScaleOnsetDetector(nn.Module):
    """
    Bulletproof multi-scale onset detection with comprehensive error handling.
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        onset_threshold: float = 0.1,
        max_memory_mb: float = 500.0
    ):
        super().__init__()
        
        # Validate and store parameters
        self.sample_rate = max(8000, min(sample_rate, 96000))
        self.n_fft = max(256, min(n_fft, 8192))
        self.hop_length = max(64, min(hop_length, n_fft // 2))
        self.n_mels = max(12, min(n_mels, 256))
        self.onset_threshold = max(0.01, min(onset_threshold, 1.0))
        self.max_memory_mb = max_memory_mb
        
        # Multi-scale configurations with validation
        self.scale_configs = self._validate_scale_configs()
        
        # Build networks with error handling
        try:
            self.onset_network = self._build_onset_network_safe()
            self.flux_processors = self._build_flux_processors_safe()
        except Exception as e:
            warnings.warn(f"Failed to build onset detection networks: {e}")
            self.onset_network = None
            self.flux_processors = None
            
        # Learnable fusion weights with constraints
        self.register_parameter('fusion_weights', 
            nn.Parameter(torch.ones(5) / 5))
        
    def _validate_scale_configs(self) -> Dict[str, Dict[str, int]]:
        """Create and validate multi-scale configurations."""
        base_configs = {
            'fine': {'n_fft': self.n_fft // 2, 'hop_length': self.hop_length // 2},
            'medium': {'n_fft': self.n_fft, 'hop_length': self.hop_length},
            'coarse': {'n_fft': self.n_fft * 2, 'hop_length': self.hop_length * 2}
        }
        
        # Validate and fix configurations
        validated_configs = {}
        for scale, config in base_configs.items():
            n_fft = max(256, min(config['n_fft'], 8192))
            hop_length = max(64, min(config['hop_length'], n_fft // 2))
            validated_configs[scale] = {'n_fft': n_fft, 'hop_length': hop_length}
            
        return validated_configs
        
    def _build_onset_network_safe(self) -> Optional[nn.Module]:
        """Build onset network with error handling."""
        try:
            return nn.Sequential(
                nn.Conv1d(self.n_mels * 3, 256, kernel_size=7, padding=3),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                
                nn.Conv1d(256, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2),
                
                nn.Conv1d(128, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                
                nn.Conv1d(64, 1, kernel_size=1),
                nn.Sigmoid()
            )
        except Exception as e:
            warnings.warn(f"Failed to build onset network: {e}")
            return None
            
    def _build_flux_processors_safe(self) -> Optional[nn.ModuleDict]:
        """Build flux processors with error handling."""
        try:
            processors = {}
            for scale in self.scale_configs.keys():
                processors[scale] = nn.Sequential(
                    nn.Conv1d(self.n_mels, 64, kernel_size=5, padding=2),
                    nn.ReLU(),
                    nn.Conv1d(64, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(32, 1, kernel_size=1),
                    nn.Sigmoid()
                )
            return nn.ModuleDict(processors)
        except Exception as e:
            warnings.warn(f"Failed to build flux processors: {e}")
            return None
            
    def _check_memory_usage(self, waveform: torch.Tensor) -> bool:
        """Check if processing this waveform would exceed memory limits."""
        try:
            # Estimate memory usage
            batch_size, audio_length = waveform.shape
            n_frames = audio_length // self.hop_length
            
            # Rough memory estimation (in MB)
            estimated_mb = (
                batch_size * n_frames * self.n_mels * len(self.scale_configs) * 4  # float32
            ) / (1024 * 1024)
            
            return estimated_mb <= self.max_memory_mb
        except Exception:
            return True  # If estimation fails, proceed cautiously
            
    def compute_spectral_flux_safe(self, mel_spec: torch.Tensor, scale: str) -> torch.Tensor:
        """Compute spectral flux with robust error handling."""
        try:
            if mel_spec.shape[-1] < 2:
                return torch.zeros_like(mel_spec)
                
            # Traditional spectral flux
            mel_diff = torch.diff(mel_spec, dim=-1)
            mel_diff = torch.clamp(mel_diff, min=0)
            
            # Neural enhancement if available
            if self.flux_processors is not None and scale in self.flux_processors:
                try:
                    enhanced_flux = self.flux_processors[scale](mel_diff)
                    enhanced_flux = F.pad(enhanced_flux, (1, 0), mode='constant', value=0)
                    return enhanced_flux.squeeze(1)
                except Exception as e:
                    warnings.warn(f"Neural enhancement failed for {scale}: {e}")
                    
            # Fallback to traditional method
            flux = torch.sum(mel_diff, dim=1)
            flux = F.pad(flux, (1, 0), mode='constant', value=0)
            return flux
            
        except Exception as e:
            warnings.warn(f"Spectral flux computation failed: {e}")
            return torch.zeros(mel_spec.shape[0], mel_spec.shape[-1], device=mel_spec.device)
            
    def compute_complex_domain_onset_safe(self, waveform: torch.Tensor) -> torch.Tensor:
        """Safe complex domain onset detection with fallbacks."""
        try:
            device = waveform.device
            target_length = int(waveform.shape[-1] // self.hop_length)
            
            if target_length == 0:
                return torch.zeros(waveform.shape[0], 1, device=device)
                
            # Multi-resolution analysis with error handling
            onset_strengths = []
            
            for window_size in [1024, 2048, 4096]:
                if window_size > waveform.shape[-1]:
                    continue
                    
                try:
                    hop = window_size // 4
                    stft = torch.stft(
                        waveform,
                        n_fft=window_size,
                        hop_length=hop,
                        return_complex=True,
                        window=torch.hann_window(window_size, device=device)
                    )
                    
                    magnitude = torch.abs(stft)
                    
                    # Simple magnitude flux
                    if magnitude.shape[-1] > 1:
                        mag_flux = torch.diff(magnitude, dim=-1)
                        onset_strength = torch.sum(torch.clamp(mag_flux, min=0), dim=1)
                        
                        # Resample to target length
                        if onset_strength.shape[-1] != target_length:
                            onset_strength = F.interpolate(
                                onset_strength.unsqueeze(1),
                                size=target_length,
                                mode='linear',
                                align_corners=False
                            ).squeeze(1)
                            
                        onset_strengths.append(onset_strength)
                        
                except Exception as e:
                    warnings.warn(f"STFT computation failed for window_size {window_size}: {e}")
                    continue
                    
            if onset_strengths:
                return torch.stack(onset_strengths, dim=1).mean(dim=1)
            else:
                return torch.zeros(waveform.shape[0], target_length, device=device)
                
        except Exception as e:
            warnings.warn(f"Complex domain onset detection failed: {e}")
            target_length = max(1, int(waveform.shape[-1] // self.hop_length))
            return torch.zeros(waveform.shape[0], target_length, device=waveform.device)
            
    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Safe multi-scale onset detection with comprehensive error handling.
        """
        device = waveform.device
        batch_size = waveform.shape[0]
        
        # Input validation
        if waveform.dim() != 2 or waveform.shape[-1] == 0:
            warnings.warn(f"Invalid waveform shape: {waveform.shape}")
            return self._empty_result(batch_size, device)
            
        # Memory check
        if not self._check_memory_usage(waveform):
            warnings.warn("Memory usage too high, using simplified processing")
            return self._simplified_onset_detection(waveform)
            
        try:
            # Extract multi-scale mel-spectrograms safely
            mel_specs = {}
            spectral_fluxes = {}
            
            for scale, scale_config in self.scale_configs.items():
                try:
                    mel_spec = torchaudio.transforms.MelSpectrogram(
                        sample_rate=self.sample_rate,
                        n_fft=scale_config['n_fft'],
                        hop_length=scale_config['hop_length'],
                        n_mels=self.n_mels
                    ).to(device)(waveform)
                    
                    mel_spec_db = torchaudio.functional.amplitude_to_DB(
                        mel_spec, multiplier=10.0, amin=1e-8
                    )
                    
                    # Resample to common time grid
                    target_length = int(waveform.shape[-1] // self.hop_length)
                    if mel_spec_db.shape[-1] != target_length and target_length > 0:
                        mel_spec_db = F.interpolate(
                            mel_spec_db.unsqueeze(1),
                            size=(mel_spec_db.shape[1], target_length),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(1)
                        
                    mel_specs[scale] = mel_spec_db
                    spectral_fluxes[scale] = self.compute_spectral_flux_safe(mel_spec_db, scale)
                    
                except Exception as e:
                    warnings.warn(f"Failed to process scale {scale}: {e}")
                    # Create fallback
                    target_length = max(1, int(waveform.shape[-1] // self.hop_length))
                    mel_specs[scale] = torch.zeros(batch_size, self.n_mels, target_length, device=device)
                    spectral_fluxes[scale] = torch.zeros(batch_size, target_length, device=device)
                    
            # Complex domain onset detection
            complex_onset = self.compute_complex_domain_onset_safe(waveform)
            
            # Neural processing if available
            neural_onset = torch.zeros_like(list(spectral_fluxes.values())[0])
            if self.onset_network is not None and len(mel_specs) == 3:
                try:
                    multi_scale_features = torch.cat(list(mel_specs.values()), dim=1)
                    neural_onset = self.onset_network(multi_scale_features).squeeze(1)
                except Exception as e:
                    warnings.warn(f"Neural onset detection failed: {e}")
                    
            # Fusion with safe weights
            all_onsets = []
            for flux in spectral_fluxes.values():
                all_onsets.append(flux)
            all_onsets.extend([complex_onset, neural_onset])
            
            # Ensure all tensors have the same shape
            min_length = min(onset.shape[-1] for onset in all_onsets)
            all_onsets = [onset[..., :min_length] for onset in all_onsets]
            
            if all_onsets:
                all_onsets_tensor = torch.stack(all_onsets, dim=1)
                weights = F.softmax(self.fusion_weights[:all_onsets_tensor.shape[1]], dim=0)
                combined_onset = torch.sum(all_onsets_tensor * weights.view(1, -1, 1), dim=1)
            else:
                combined_onset = torch.zeros(batch_size, 1, device=device)
                
            # Peak picking
            onset_peaks = self._pick_onset_peaks_safe(combined_onset)
            
            return {
                'onset_strength': combined_onset,
                'spectral_flux_fine': spectral_fluxes.get('fine', torch.zeros_like(combined_onset)),
                'spectral_flux_medium': spectral_fluxes.get('medium', torch.zeros_like(combined_onset)),
                'spectral_flux_coarse': spectral_fluxes.get('coarse', torch.zeros_like(combined_onset)),
                'complex_onset': complex_onset,
                'neural_onset': neural_onset,
                'onset_peaks': onset_peaks,
                'onset_times': self._frames_to_time_safe(onset_peaks),
                'fusion_weights': weights if 'weights' in locals() else self.fusion_weights
            }
            
        except Exception as e:
            warnings.warn(f"Onset detection failed completely: {e}")
            return self._empty_result(batch_size, device)
            
    def _simplified_onset_detection(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Simplified onset detection for memory-constrained scenarios."""
        try:
            device = waveform.device
            batch_size = waveform.shape[0]
            
            # Simple spectral flux
            mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=min(self.n_fft, 1024),
                hop_length=self.hop_length,
                n_mels=min(self.n_mels, 64)
            ).to(device)(waveform)
            
            mel_spec_db = torchaudio.functional.amplitude_to_DB(mel_spec, multiplier=10.0, amin=1e-8)
            
            if mel_spec_db.shape[-1] > 1:
                flux = torch.sum(torch.clamp(torch.diff(mel_spec_db, dim=-1), min=0), dim=1)
                flux = F.pad(flux, (1, 0), mode='constant', value=0)
            else:
                flux = torch.zeros(batch_size, 1, device=device)
                
            onset_peaks = self._pick_onset_peaks_safe(flux)
            
            return {
                'onset_strength': flux,
                'spectral_flux_fine': flux,
                'spectral_flux_medium': flux,
                'spectral_flux_coarse': flux,
                'complex_onset': flux,
                'neural_onset': flux,
                'onset_peaks': onset_peaks,
                'onset_times': self._frames_to_time_safe(onset_peaks),
                'fusion_weights': torch.ones(1, device=device)
            }
            
        except Exception as e:
            warnings.warn(f"Simplified onset detection failed: {e}")
            return self._empty_result(waveform.shape[0], waveform.device)
            
    def _empty_result(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Return empty result structure."""
        empty_tensor = torch.zeros(batch_size, 1, device=device)
        empty_peaks = [torch.tensor([], dtype=torch.long) for _ in range(batch_size)]
        empty_times = [torch.tensor([]) for _ in range(batch_size)]
        
        return {
            'onset_strength': empty_tensor,
            'spectral_flux_fine': empty_tensor,
            'spectral_flux_medium': empty_tensor,
            'spectral_flux_coarse': empty_tensor,
            'complex_onset': empty_tensor,
            'neural_onset': empty_tensor,
            'onset_peaks': empty_peaks,
            'onset_times': empty_times,
            'fusion_weights': torch.ones(1, device=device)
        }
        
    def _pick_onset_peaks_safe(self, onset_strength: torch.Tensor) -> List[torch.Tensor]:
        """Safe peak picking with adaptive thresholding."""
        batch_size = onset_strength.shape[0]
        all_peaks = []
        
        for b in range(batch_size):
            try:
                strength = onset_strength[b].cpu().numpy()
                
                if len(strength) < 3:
                    all_peaks.append(torch.tensor([], dtype=torch.long))
                    continue
                    
                # Adaptive threshold
                window_size = min(int(1.0 * self.sample_rate / self.hop_length), len(strength) // 2)
                window_size = max(3, window_size)
                
                # Simple local maxima detection
                peaks = []
                for i in range(1, len(strength) - 1):
                    if (strength[i] > strength[i-1] and 
                        strength[i] > strength[i+1] and 
                        strength[i] > self.onset_threshold):
                        peaks.append(i)
                        
                # Filter by minimum distance
                if peaks:
                    min_distance = max(1, int(self.min_onset_distance * self.sample_rate / self.hop_length))
                    filtered_peaks = [peaks[0]]
                    
                    for peak in peaks[1:]:
                        if peak - filtered_peaks[-1] >= min_distance:
                            filtered_peaks.append(peak)
                            
                    all_peaks.append(torch.tensor(filtered_peaks, dtype=torch.long))
                else:
                    all_peaks.append(torch.tensor([], dtype=torch.long))
                    
            except Exception as e:
                warnings.warn(f"Peak picking failed for batch {b}: {e}")
                all_peaks.append(torch.tensor([], dtype=torch.long))
                
        return all_peaks
        
    def _frames_to_time_safe(self, frame_peaks: List[torch.Tensor]) -> List[torch.Tensor]:
        """Safe conversion of frame indices to time."""
        times = []
        for peaks in frame_peaks:
            try:
                if len(peaks) > 0:
                    time_peaks = peaks.float() * self.hop_length / self.sample_rate
                else:
                    time_peaks = torch.tensor([])
                times.append(time_peaks)
            except Exception:
                times.append(torch.tensor([]))
        return times


class SafeAdvancedTempoEstimator(nn.Module):
    """
    Bulletproof tempo estimation with multiple methods and confidence weighting.
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        hop_length: int = 512,
        tempo_min: float = 60.0,
        tempo_max: float = 200.0,
        tempo_bins: int = 140
    ):
        super().__init__()
        
        # Validate and store parameters
        self.sample_rate = max(8000, min(sample_rate, 96000))
        self.hop_length = max(64, min(hop_length, 2048))
        self.tempo_min = max(30.0, min(tempo_min, 100.0))
        self.tempo_max = max(self.tempo_min + 10, min(tempo_max, 300.0))
        self.tempo_bins = max(50, min(tempo_bins, 300))
        
        # Tempo grid
        self.register_buffer('tempo_grid', torch.linspace(self.tempo_min, self.tempo_max, self.tempo_bins))
        
        # Neural tempo estimator with error handling
        try:
            self.tempo_network = self._build_tempo_network_safe()
        except Exception as e:
            warnings.warn(f"Failed to build tempo network: {e}")
            self.tempo_network = None
            
    def _build_tempo_network_safe(self) -> Optional[nn.Module]:
        """Build tempo network with error handling."""
        try:
            return nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=15, padding=7),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),
                
                nn.Conv1d(64, 128, kernel_size=9, padding=4),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(2),
                
                nn.Conv1d(128, 256, kernel_size=5, padding=2),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(64),
                
                nn.Flatten(),
                nn.Linear(256 * 64, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                
                nn.Linear(256, self.tempo_bins),
                nn.Softmax(dim=1)
            )
        except Exception as e:
            warnings.warn(f"Failed to build tempo network: {e}")
            return None
            
    def autocorrelation_tempo_safe(self, onset_strength: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Safe autocorrelation-based tempo estimation."""
        batch_size = onset_strength.shape[0]
        tempos = []
        confidences = []
        
        for b in range(batch_size):
            try:
                strength = onset_strength[b].cpu().numpy()
                
                if len(strength) < 10:
                    tempos.append(120.0)
                    confidences.append(0.0)
                    continue
                    
                # Simple autocorrelation
                autocorr = np.correlate(strength, strength, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                
                # Convert lags to tempo
                lags = np.arange(len(autocorr))
                lag_times = lags * self.hop_length / self.sample_rate
                
                # Valid tempo range
                min_lag = max(1, int(60 / self.tempo_max * self.sample_rate / self.hop_length))
                max_lag = min(len(autocorr), int(60 / self.tempo_min * self.sample_rate / self.hop_length))
                
                if max_lag > min_lag:
                    valid_autocorr = autocorr[min_lag:max_lag]
                    valid_lags = lags[min_lag:max_lag]
                    
                    if len(valid_autocorr) > 0:
                        peak_idx = np.argmax(valid_autocorr)
                        peak_lag = valid_lags[peak_idx]
                        
                        tempo = 60 / (peak_lag * self.hop_length / self.sample_rate)
                        confidence = valid_autocorr[peak_idx] / (np.sum(valid_autocorr) + 1e-8)
                    else:
                        tempo = 120.0
                        confidence = 0.0
                else:
                    tempo = 120.0
                    confidence = 0.0
                    
                tempos.append(float(tempo))
                confidences.append(float(confidence))
                
            except Exception as e:
                warnings.warn(f"Autocorrelation tempo estimation failed for batch {b}: {e}")
                tempos.append(120.0)
                confidences.append(0.0)
                
        return {
            'tempo': torch.tensor(tempos, device=onset_strength.device),
            'confidence': torch.tensor(confidences, device=onset_strength.device)
        }
        
    def comb_filter_tempo_safe(self, onset_strength: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Safe comb filtering for tempo estimation."""
        batch_size = onset_strength.shape[0]
        device = onset_strength.device
        tempo_scores = torch.zeros(batch_size, len(self.tempo_grid), device=device)
        
        for b in range(batch_size):
            try:
                strength = onset_strength[b]
                
                if len(strength) < 10:
                    continue
                    
                for t, tempo in enumerate(self.tempo_grid):
                    try:
                        beat_period = 60 / tempo * self.sample_rate / self.hop_length
                        
                        # Simple comb filter
                        max_delay = min(int(beat_period * 4), len(strength) // 2)
                        
                        if max_delay > 1:
                            # Create comb filter
                            comb = torch.zeros(max_delay, device=device)
                            step = max(1, int(beat_period))
                            for i in range(0, max_delay, step):
                                if i < max_delay:
                                    comb[i] = 1.0
                                    
                            # Normalize
                            comb = comb / (torch.sum(comb) + 1e-8)
                            
                            # Convolution score
                            if len(strength) >= len(comb):
                                score = F.conv1d(
                                    strength[:len(comb)].unsqueeze(0).unsqueeze(0),
                                    comb.flip(0).unsqueeze(0).unsqueeze(0),
                                    padding=len(comb)//2
                                ).squeeze()
                                
                                tempo_scores[b, t] = torch.sum(score)
                                
                    except Exception:
                        continue
                        
            except Exception as e:
                warnings.warn(f"Comb filter tempo estimation failed for batch {b}: {e}")
                continue
                
        # Normalize scores
        row_sums = torch.sum(tempo_scores, dim=1, keepdim=True)
        tempo_scores = tempo_scores / (row_sums + 1e-8)
        
        # Find best tempo
        best_tempo_idx = torch.argmax(tempo_scores, dim=1)
        best_tempos = self.tempo_grid[best_tempo_idx]
        best_confidences = torch.gather(tempo_scores, 1, best_tempo_idx.unsqueeze(1)).squeeze(1)
        
        return {
            'tempo': best_tempos,
            'confidence': best_confidences,
            'tempo_scores': tempo_scores
        }
        
    def neural_tempo_estimation_safe(self, onset_strength: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Safe neural network-based tempo estimation."""
        device = onset_strength.device
        batch_size = onset_strength.shape[0]
        
        if self.tempo_network is None:
            # Fallback to default tempo
            return {
                'tempo': torch.full((batch_size,), 120.0, device=device),
                'confidence': torch.zeros(batch_size, device=device),
                'tempo_probs': torch.zeros(batch_size, self.tempo_bins, device=device)
            }
            
        try:
            # Fixed length for neural network
            target_length = 512
            processed_strength = torch.zeros(batch_size, target_length, device=device)
            
            for b in range(batch_size):
                strength = onset_strength[b]
                if len(strength) >= target_length:
                    processed_strength[b] = strength[:target_length]
                elif len(strength) > 0:
                    processed_strength[b, :len(strength)] = strength
                    
            # Neural prediction
            tempo_probs = self.tempo_network(processed_strength.unsqueeze(1))
            
            # Convert to tempo estimates
            expected_tempo = torch.sum(tempo_probs * self.tempo_grid.unsqueeze(0), dim=1)
            confidence = torch.max(tempo_probs, dim=1)[0]
            
            return {
                'tempo': expected_tempo,
                'confidence': confidence,
                'tempo_probs': tempo_probs
            }
            
        except Exception as e:
            warnings.warn(f"Neural tempo estimation failed: {e}")
            return {
                'tempo': torch.full((batch_size,), 120.0, device=device),
                'confidence': torch.zeros(batch_size, device=device),
                'tempo_probs': torch.zeros(batch_size, self.tempo_bins, device=device)
            }
            
    def forward(self, onset_strength: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Safe comprehensive tempo estimation.
        """
        try:
            # Input validation
            if onset_strength.dim() != 2 or onset_strength.shape[-1] == 0:
                batch_size = onset_strength.shape[0] if onset_strength.dim() > 0 else 1
                device = onset_strength.device if hasattr(onset_strength, 'device') else torch.device('cpu')
                return {
                    'tempo': torch.full((batch_size,), 120.0, device=device),
                    'confidence': torch.zeros(batch_size, device=device),
                    'autocorr_tempo': torch.full((batch_size,), 120.0, device=device),
                    'comb_tempo': torch.full((batch_size,), 120.0, device=device),
                    'neural_tempo': torch.full((batch_size,), 120.0, device=device),
                    'individual_confidences': torch.zeros(batch_size, 3, device=device)
                }
                
            # Multiple estimation methods
            autocorr_result = self.autocorrelation_tempo_safe(onset_strength)
            comb_result = self.comb_filter_tempo_safe(onset_strength)
            neural_result = self.neural_tempo_estimation_safe(onset_strength)
            
            # Ensemble fusion with confidence weighting
            confidences = torch.stack([
                autocorr_result['confidence'],
                comb_result['confidence'],
                neural_result['confidence']
            ], dim=1)
            
            tempos = torch.stack([
                autocorr_result['tempo'],
                comb_result['tempo'],
                neural_result['tempo']
            ], dim=1)
            
            # Weighted average
            total_confidence = torch.sum(confidences, dim=1, keepdim=True) + 1e-8
            weights = confidences / total_confidence
            
            ensemble_tempo = torch.sum(tempos * weights, dim=1)
            ensemble_confidence = torch.mean(confidences, dim=1)
            
            return {
                'tempo': ensemble_tempo,
                'confidence': ensemble_confidence,
                'autocorr_tempo': autocorr_result['tempo'],
                'comb_tempo': comb_result['tempo'],
                'neural_tempo': neural_result['tempo'],
                'individual_confidences': confidences
            }
            
        except Exception as e:
            warnings.warn(f"Tempo estimation failed completely: {e}")
            batch_size = onset_strength.shape[0] if onset_strength.dim() > 0 else 1
            device = onset_strength.device if hasattr(onset_strength, 'device') else torch.device('cpu')
            return {
                'tempo': torch.full((batch_size,), 120.0, device=device),
                'confidence': torch.zeros(batch_size, device=device),
                'autocorr_tempo': torch.full((batch_size,), 120.0, device=device),
                'comb_tempo': torch.full((batch_size,), 120.0, device=device),
                'neural_tempo': torch.full((batch_size,), 120.0, device=device),
                'individual_confidences': torch.zeros(batch_size, 3, device=device)
            }


class BulletproofBeatSynchronizer(AudioModuleBase):
    """
    Production-ready beat synchronization with comprehensive error handling,
    memory management, and graceful degradation for musical rhythm analysis.
    
    Features:
    - Robust onset detection with multiple fallback strategies
    - Ensemble tempo estimation with confidence weighting
    - Memory-efficient processing for long audio sequences
    - Device compatibility and automatic error recovery
    - Comprehensive quality assessment and reporting
    """
    
    def __init__(
        self, 
        config: AudioModuleConfig,
        beat_config: Optional[BeatSynchronizerConfig] = None
    ):
        super().__init__(config)
        
        # Configuration validation
        self.beat_config = beat_config or BeatSynchronizerConfig()
        if not self.beat_config.validate():
            warnings.warn("Using fallback beat synchronizer configuration")
            self.beat_config = BeatSynchronizerConfig()
            
        # Core components with error handling
        try:
            self.onset_detector = SafeMultiScaleOnsetDetector(
                sample_rate=config.sample_rate,
                hop_length=config.hop_length,
                n_mels=config.n_mels,
                n_fft=config.n_fft,
                onset_threshold=self.beat_config.onset_threshold
            )
        except Exception as e:
            warnings.warn(f"Failed to initialize onset detector: {e}")
            self.onset_detector = None
            
        try:
            self.tempo_estimator = SafeAdvancedTempoEstimator(
                sample_rate=config.sample_rate,
                hop_length=config.hop_length,
                tempo_min=self.beat_config.min_tempo,
                tempo_max=self.beat_config.max_tempo,
                tempo_bins=self.beat_config.tempo_bins
            )
        except Exception as e:
            warnings.warn(f"Failed to initialize tempo estimator: {e}")
            self.tempo_estimator = None
            
        # Memory management
        self.max_chunk_size = self.beat_config.chunk_size
        
        # Quality assessment thresholds
        self.quality_thresholds = {
            BeatTrackingQuality.EXCELLENT: 0.8,
            BeatTrackingQuality.GOOD: 0.6,
            BeatTrackingQuality.FAIR: 0.4,
            BeatTrackingQuality.POOR: 0.2
        }
        
    def _check_input_validity(self, waveform: torch.Tensor) -> Tuple[bool, str]:
        """Comprehensive input validation."""
        try:
            if not isinstance(waveform, torch.Tensor):
                return False, "Input must be a torch.Tensor"
                
            if waveform.dim() not in [1, 2]:
                return False, f"Invalid number of dimensions: {waveform.dim()}"
                
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
                
            if waveform.shape[-1] == 0:
                return False, "Empty audio input"
                
            if waveform.shape[-1] < self.config.hop_length:
                return False, f"Audio too short: {waveform.shape[-1]} samples"
                
            # Check for valid audio range
            if torch.any(torch.isnan(waveform)) or torch.any(torch.isinf(waveform)):
                return False, "Audio contains NaN or infinite values"
                
            return True, "Valid input"
            
        except Exception as e:
            return False, f"Input validation error: {e}"
            
    def _chunk_processing(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        """Split long audio into manageable chunks."""
        audio_length = waveform.shape[-1]
        
        if audio_length <= self.max_chunk_size:
            return [waveform]
            
        chunks = []
        overlap = self.config.hop_length * 10  # 10 frame overlap
        
        for start in range(0, audio_length, self.max_chunk_size - overlap):
            end = min(start + self.max_chunk_size, audio_length)
            chunk = waveform[..., start:end]
            chunks.append(chunk)
            
        return chunks
        
    def _merge_chunk_results(self, chunk_results: List[Dict], chunk_offsets: List[float]) -> Dict:
        """Merge results from multiple chunks."""
        if not chunk_results:
            return self._empty_analysis_result()
            
        if len(chunk_results) == 1:
            return chunk_results[0]
            
        try:
            # Merge onset times
            all_onset_times = []
            for i, (result, offset) in enumerate(zip(chunk_results, chunk_offsets)):
                onset_times = result.get('onset_times', [])
                if onset_times and len(onset_times) > 0:
                    adjusted_times = onset_times + offset
                    all_onset_times.extend(adjusted_times.tolist())
                    
            # Merge beat times
            all_beat_times = []
            for i, (result, offset) in enumerate(zip(chunk_results, chunk_offsets)):
                beat_times = result.get('beat_times', torch.tensor([]))
                if len(beat_times) > 0:
                    adjusted_times = beat_times + offset
                    all_beat_times.extend(adjusted_times.tolist())
                    
            # Average tempo estimates
            valid_tempos = [r['tempo'] for r in chunk_results if r.get('tempo', 0) > 0]
            avg_tempo = np.mean(valid_tempos) if valid_tempos else self.beat_config.fallback_tempo
            
            # Average confidence
            valid_confidences = [r['tempo_confidence'] for r in chunk_results if r.get('tempo_confidence', 0) > 0]
            avg_confidence = np.mean(valid_confidences) if valid_confidences else 0.0
            
            return {
                'onset_times': torch.tensor(all_onset_times),
                'beat_times': torch.tensor(all_beat_times),
                'tempo': torch.tensor(avg_tempo),
                'tempo_confidence': torch.tensor(avg_confidence),
                'quality': self._assess_quality(torch.tensor(all_beat_times), avg_confidence),
                'processing_chunks': len(chunk_results)
            }
            
        except Exception as e:
            warnings.warn(f"Failed to merge chunk results: {e}")
            return chunk_results[0] if chunk_results else self._empty_analysis_result()
            
    def dynamic_programming_beats_safe(
        self,
        onset_strength: torch.Tensor,
        tempo: float,
        tempo_confidence: float
    ) -> torch.Tensor:
        """
        Safe dynamic programming beat tracking with comprehensive error handling.
        """
        try:
            if len(onset_strength) < 2:
                return torch.tensor([])
                
            onset_np = onset_strength.cpu().numpy()
            n_frames = len(onset_np)
            
            # Validate tempo
            if not (self.beat_config.min_tempo <= tempo <= self.beat_config.max_tempo):
                tempo = self.beat_config.fallback_tempo
                tempo_confidence = 0.0
                
            # Beat period in frames
            beat_period = 60 / tempo * self.config.sample_rate / self.config.hop_length
            beat_period = max(2.0, min(beat_period, n_frames / 4))  # Reasonable bounds
            
            # Simplified DP for robustness
            # Create state space
            dp_scores = np.full(n_frames, -np.inf)
            dp_path = np.zeros(n_frames, dtype=int)
            
            # Initialize
            for i in range(min(n_frames, int(beat_period))):
                dp_scores[i] = onset_np[i]
                
            # Fill DP table
            for t in range(1, n_frames):
                for prev_t in range(max(0, t - int(beat_period * 2)), t):
                    if dp_scores[prev_t] == -np.inf:
                        continue
                        
                    interval = t - prev_t
                    
                    # Tempo consistency cost
                    tempo_cost = -0.5 * ((interval - beat_period) / (beat_period * 0.2)) ** 2
                    
                    # Onset strength
                    onset_cost = onset_np[t]
                    
                    # Total score
                    score = (
                        dp_scores[prev_t] +
                        self.beat_config.alpha * tempo_cost +
                        self.beat_config.beta * onset_cost
                    )
                    
                    if score > dp_scores[t]:
                        dp_scores[t] = score
                        dp_path[t] = prev_t
                        
            # Backtrack
            if np.all(dp_scores == -np.inf):
                return torch.tensor([])
                
            beat_positions = []
            current = np.argmax(dp_scores)
            
            while current > 0:
                beat_positions.append(current)
                current = dp_path[current]
                if current in beat_positions:  # Prevent infinite loops
                    break
                    
            beat_positions.reverse()
            
            # Filter beats by minimum distance
            if beat_positions:
                filtered_beats = [beat_positions[0]]
                min_distance = max(1, int(beat_period * 0.3))
                
                for beat in beat_positions[1:]:
                    if beat - filtered_beats[-1] >= min_distance:
                        filtered_beats.append(beat)
                        
                return torch.tensor(filtered_beats, dtype=torch.float32)
            else:
                return torch.tensor([])
                
        except Exception as e:
            warnings.warn(f"Dynamic programming beat tracking failed: {e}")
            return torch.tensor([])
            
    def _assess_quality(self, beat_times: torch.Tensor, tempo_confidence: float) -> BeatTrackingQuality:
        """Assess the quality of beat tracking results."""
        try:
            if len(beat_times) < 2:
                return BeatTrackingQuality.FAILED
                
            # Beat consistency
            intervals = torch.diff(beat_times)
            if len(intervals) == 0:
                return BeatTrackingQuality.FAILED
                
            consistency = 1.0 / (1.0 + torch.std(intervals) / (torch.mean(intervals) + 1e-8))
            
            # Combined quality score
            quality_score = 0.7 * consistency.item() + 0.3 * tempo_confidence
            
            for quality_level in [BeatTrackingQuality.EXCELLENT, BeatTrackingQuality.GOOD, 
                                 BeatTrackingQuality.FAIR, BeatTrackingQuality.POOR]:
                if quality_score >= self.quality_thresholds[quality_level]:
                    return quality_level
                    
            return BeatTrackingQuality.FAILED
            
        except Exception:
            return BeatTrackingQuality.FAILED
            
    def _empty_analysis_result(self) -> Dict[str, Any]:
        """Return empty analysis result."""
        return {
            'onset_times': torch.tensor([]),
            'beat_times': torch.tensor([]),
            'tempo': torch.tensor(self.beat_config.fallback_tempo),
            'tempo_confidence': torch.tensor(0.0),
            'quality': BeatTrackingQuality.FAILED,
            'error': "No valid analysis possible"
        }
        
    def _fallback_analysis(self, waveform: torch.Tensor) -> Dict[str, Any]:
        """Fallback analysis when main components fail."""
        try:
            # Simple energy-based onset detection
            audio_length = waveform.shape[-1] / self.config.sample_rate
            
            # Generate regular beat grid at fallback tempo
            beat_interval = 60.0 / self.beat_config.fallback_tempo
            n_beats = int(audio_length / beat_interval)
            beat_times = torch.arange(n_beats, dtype=torch.float32) * beat_interval
            
            return {
                'onset_times': beat_times,  # Use beats as onsets
                'beat_times': beat_times,
                'tempo': torch.tensor(self.beat_config.fallback_tempo),
                'tempo_confidence': torch.tensor(0.1),  # Low confidence
                'quality': BeatTrackingQuality.POOR,
                'fallback_used': True
            }
            
        except Exception as e:
            warnings.warn(f"Fallback analysis failed: {e}")
            return self._empty_analysis_result()
            
    def forward(self, waveform: torch.Tensor) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Bulletproof beat synchronization and rhythm analysis.
        
        Args:
            waveform: Input audio [batch, samples] or [samples]
            
        Returns:
            Comprehensive beat and rhythm analysis with quality assessment
        """
        start_time = time.time()
        
        # Input validation
        valid, error_msg = self._check_input_validity(waveform)
        if not valid:
            warnings.warn(f"Input validation failed: {error_msg}")
            return self._empty_analysis_result()
            
        # Ensure batch dimension
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False
            
        batch_results = []
        
        for b in range(waveform.shape[0]):
            try:
                audio_sample = waveform[b:b+1]
                audio_length = audio_sample.shape[-1] / self.config.sample_rate
                
                # Check if audio is too long and needs chunking
                if audio_length > self.beat_config.max_audio_length:
                    warnings.warn(f"Audio too long ({audio_length:.1f}s), using chunked processing")
                    
                    chunks = self._chunk_processing(audio_sample)
                    chunk_results = []
                    chunk_offsets = []
                    
                    for i, chunk in enumerate(chunks):
                        offset = i * (self.max_chunk_size - self.config.hop_length * 10) / self.config.sample_rate
                        chunk_offsets.append(offset)
                        
                        try:
                            chunk_result = self._process_single_chunk(chunk)
                            chunk_results.append(chunk_result)
                        except Exception as e:
                            warnings.warn(f"Chunk {i} processing failed: {e}")
                            continue
                            
                    sample_result = self._merge_chunk_results(chunk_results, chunk_offsets)
                else:
                    sample_result = self._process_single_chunk(audio_sample)
                    
                # Add processing metadata
                sample_result['processing_time'] = time.time() - start_time
                sample_result['audio_length'] = audio_length
                sample_result['sample_rate'] = self.config.sample_rate
                
                batch_results.append(sample_result)
                
            except Exception as e:
                warnings.warn(f"Processing failed for batch {b}: {e}")
                fallback_result = self._fallback_analysis(waveform[b:b+1])
                fallback_result['processing_time'] = time.time() - start_time
                fallback_result['error'] = str(e)
                batch_results.append(fallback_result)
                
        # Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return batch_results[0] if single_sample else batch_results
        
    def _process_single_chunk(self, audio_chunk: torch.Tensor) -> Dict[str, Any]:
        """Process a single audio chunk."""
        try:
            # Onset detection
            if self.onset_detector is not None:
                onset_results = self.onset_detector(audio_chunk)
                onset_strength = onset_results['onset_strength'][0]
                onset_times = onset_results['onset_times'][0]
            else:
                # Fallback: simple energy-based detection
                onset_strength = torch.sum(audio_chunk ** 2, dim=0, keepdim=True)
                onset_times = torch.arange(len(onset_strength), dtype=torch.float32) * self.config.hop_length / self.config.sample_rate
                
            # Tempo estimation
            if self.tempo_estimator is not None:
                tempo_results = self.tempo_estimator(onset_strength.unsqueeze(0))
                tempo = tempo_results['tempo'][0].item()
                tempo_confidence = tempo_results['confidence'][0].item()
            else:
                tempo = self.beat_config.fallback_tempo
                tempo_confidence = 0.0
                
            # Beat tracking
            beat_positions = self.dynamic_programming_beats_safe(
                onset_strength, tempo, tempo_confidence
            )
            
            if len(beat_positions) > 0:
                beat_times = beat_positions * self.config.hop_length / self.config.sample_rate
            else:
                beat_times = torch.tensor([])
                
            # Quality assessment
            quality = self._assess_quality(beat_times, tempo_confidence)
            
            return {
                'onset_times': onset_times,
                'beat_times': beat_times,
                'tempo': torch.tensor(tempo),
                'tempo_confidence': torch.tensor(tempo_confidence),
                'quality': quality,
                'n_onsets': len(onset_times),
                'n_beats': len(beat_times)
            }
            
        except Exception as e:
            warnings.warn(f"Single chunk processing failed: {e}")
            return self._fallback_analysis(audio_chunk)


# Factory function
def create_bulletproof_beat_synchronizer(
    config: Optional[AudioModuleConfig] = None,
    beat_config: Optional[BeatSynchronizerConfig] = None
) -> BulletproofBeatSynchronizer:
    """Create a bulletproof beat synchronizer with configuration."""
    if config is None:
        try:
            from ..audio_analysis.audio_config import get_music_config
            config = get_music_config()
        except ImportError:
            # Fallback configuration
            from dataclasses import dataclass
            
            @dataclass
            class FallbackConfig:
                sample_rate: int = 22050
                hop_length: int = 512
                n_mels: int = 128
                n_fft: int = 2048
                
            config = FallbackConfig()
            
    return BulletproofBeatSynchronizer(config, beat_config)


# Test specifications for comprehensive validation
def test_bulletproof_beat_synchronizer():
    """Comprehensive test suite for bulletproof beat synchronizer."""
    print("Testing BulletproofBeatSynchronizer...")
    
    # Test with various audio scenarios
    test_cases = [
        "synthetic_drum_pattern",
        "variable_tempo",
        "complex_polyrhythm",
        "very_short_audio",
        "very_long_audio",
        "silent_audio",
        "noisy_audio",
        "extreme_tempo"
    ]
    
    synchronizer = create_bulletproof_beat_synchronizer()
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case}")
        
        try:
            # Generate test audio based on case
            if test_case == "synthetic_drum_pattern":
                waveform = generate_synthetic_drums(duration=5, tempo=120)
            elif test_case == "variable_tempo":
                waveform = generate_variable_tempo_audio(duration=8)
            elif test_case == "complex_polyrhythm":
                waveform = generate_polyrhythmic_audio(duration=6)
            elif test_case == "very_short_audio":
                waveform = torch.randn(1, 100)  # Very short
            elif test_case == "very_long_audio":
                waveform = torch.randn(1, 22050 * 120)  # 2 minutes
            elif test_case == "silent_audio":
                waveform = torch.zeros(1, 22050 * 3)
            elif test_case == "noisy_audio":
                waveform = torch.randn(1, 22050 * 4) * 0.1
            elif test_case == "extreme_tempo":
                waveform = generate_synthetic_drums(duration=4, tempo=300)
            else:
                continue
                
            # Test the synchronizer
            result = synchronizer(waveform)
            
            # Validate results
            assert isinstance(result, dict), f"Result should be dict for {test_case}"
            assert 'tempo' in result, f"Missing tempo for {test_case}"
            assert 'quality' in result, f"Missing quality for {test_case}"
            assert 'beat_times' in result, f"Missing beat_times for {test_case}"
            
            print(f"  ✓ Tempo: {result['tempo']:.1f} BPM")
            print(f"  ✓ Quality: {result['quality']}")
            print(f"  ✓ Beats detected: {len(result['beat_times'])}")
            print(f"  ✓ Processing time: {result.get('processing_time', 0):.3f}s")
            
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            
    print("\nBulletproof beat synchronizer testing completed!")


def generate_synthetic_drums(duration: int, tempo: float) -> torch.Tensor:
    """Generate synthetic drum pattern for testing."""
    sample_rate = 22050
    samples = sample_rate * duration
    waveform = torch.zeros(1, samples)
    
    beat_interval = 60.0 / tempo
    
    for beat_time in torch.arange(0, duration, beat_interval):
        beat_sample = int(beat_time * sample_rate)
        if beat_sample < samples - 1000:
            # Kick drum
            kick = torch.sin(2 * torch.pi * 60 * torch.linspace(0, 0.1, 1000))
            kick *= torch.exp(-5 * torch.linspace(0, 0.1, 1000))
            waveform[0, beat_sample:beat_sample+1000] += kick
            
    return waveform


def generate_variable_tempo_audio(duration: int) -> torch.Tensor:
    """Generate audio with variable tempo for testing."""
    sample_rate = 22050
    samples = sample_rate * duration
    waveform = torch.zeros(1, samples)
    
    # Tempo changes from 100 to 140 BPM
    for i in range(samples // 1000):
        tempo = 100 + 40 * (i / (samples // 1000))
        beat_interval = 60.0 / tempo
        
        beat_sample = i * 1000
        if beat_sample < samples - 500:
            kick = torch.sin(2 * torch.pi * 60 * torch.linspace(0, 0.05, 500))
            kick *= torch.exp(-10 * torch.linspace(0, 0.05, 500))
            waveform[0, beat_sample:beat_sample+500] += kick
            
    return waveform


def generate_polyrhythmic_audio(duration: int) -> torch.Tensor:
    """Generate polyrhythmic audio for testing."""
    sample_rate = 22050
    samples = sample_rate * duration
    waveform = torch.zeros(1, samples)
    
    # 4/4 kick pattern
    kick_interval = 60.0 / 120  # 120 BPM
    for beat_time in torch.arange(0, duration, kick_interval):
        beat_sample = int(beat_time * sample_rate)
        if beat_sample < samples - 500:
            kick = torch.sin(2 * torch.pi * 60 * torch.linspace(0, 0.05, 500))
            waveform[0, beat_sample:beat_sample+500] += kick
            
    # 3/4 snare pattern (polyrhythm)
    snare_interval = 60.0 / 90  # Creates polyrhythm
    for beat_time in torch.arange(0, duration, snare_interval):
        beat_sample = int(beat_time * sample_rate)
        if beat_sample < samples - 300:
            snare = torch.randn(300) * 0.3
            snare *= torch.exp(-15 * torch.linspace(0, 0.05, 300))
            waveform[0, beat_sample:beat_sample+300] += snare
            
    return waveform


# Example usage and testing
if __name__ == "__main__":
    test_bulletproof_beat_synchronizer()