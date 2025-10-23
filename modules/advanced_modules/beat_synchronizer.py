"""
Beat Synchronization and Advanced Tempo Analysis

Implements state-of-the-art beat tracking using:
- Dynamic programming beat tracking (Ellis-style)
- Neural onset detection with multi-scale features
- Tempo estimation via autocorrelation and comb filtering
- Beat synchronization and phase alignment for timbral analysis
- Polyrhythmic and complex meter support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import librosa
import scipy.signal

from ..audio_analysis.audio_config import AudioModuleConfig, AudioModuleBase


class MultiScaleOnsetDetector(nn.Module):
    """
    Neural onset detection using multi-scale spectral and temporal features.
    
    Combines traditional spectral flux with learned onset patterns
    across multiple temporal resolutions.
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        onset_threshold: float = 0.1
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.onset_threshold = onset_threshold
        
        # Store config for multi-scale processing
        self.n_mels = n_mels
        
        # Multi-scale configurations
        self.scale_configs = {
            'fine': {'n_fft': 1024, 'hop_length': 256},
            'medium': {'n_fft': n_fft, 'hop_length': hop_length}, 
            'coarse': {'n_fft': 4096, 'hop_length': 1024}
        }
        
        # Neural onset detector with multi-scale fusion
        self.onset_network = self._build_onset_network(n_mels)
        
        # Spectral flux processors for each scale
        self.flux_processors = nn.ModuleDict({
            scale: nn.Sequential(
                nn.Conv1d(n_mels, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 1, kernel_size=1),
                nn.Sigmoid()
            ) for scale in ['fine', 'medium', 'coarse']
        })
        
    def _build_onset_network(self, n_mels: int):
        """Build multi-scale neural onset detector."""
        return nn.Sequential(
            # Input: concatenated multi-scale features
            nn.Conv1d(n_mels * 3, 256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Temporal context layers
            nn.Conv1d(256, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # Onset prediction
            nn.Conv1d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def compute_spectral_flux(self, mel_spec: torch.Tensor, scale: str) -> torch.Tensor:
        """Compute enhanced spectral flux with neural processing."""
        # Traditional spectral flux
        mel_diff = torch.diff(mel_spec, dim=-1)
        mel_diff = torch.clamp(mel_diff, min=0)  # Half-wave rectification
        
        # Neural enhancement
        enhanced_flux = self.flux_processors[scale](mel_diff)
        
        # Pad to original length
        enhanced_flux = F.pad(enhanced_flux, (1, 0), mode='constant', value=0)
        
        return enhanced_flux.squeeze(1)
        
    def compute_complex_domain_onset(self, waveform: torch.Tensor) -> torch.Tensor:
        """Advanced complex domain onset detection."""
        # STFT with different window sizes for multi-resolution analysis
        stfts = []
        
        for window_size in [1024, 2048, 4096]:
            hop = window_size // 4
            stft = torch.stft(
                waveform,
                n_fft=window_size,
                hop_length=hop,
                return_complex=True,
                window=torch.hann_window(window_size, device=waveform.device)
            )
            
            magnitude = torch.abs(stft)
            phase = torch.angle(stft)
            
            # Phase deviation detection
            phase_dev = torch.diff(torch.unwrap(phase, dim=-1), dim=-1)
            
            # Magnitude flux with adaptive thresholding
            mag_flux = torch.diff(magnitude, dim=-1)
            threshold = torch.quantile(mag_flux, 0.8, dim=1, keepdim=True)
            mag_flux = torch.clamp(mag_flux - threshold, min=0)
            
            # Combined onset strength for this resolution
            onset_strength = torch.sum(mag_flux + 0.1 * torch.abs(phase_dev), dim=1)
            
            # Resample to common time grid (using medium resolution)
            if hop != self.hop_length:
                target_length = int(waveform.shape[-1] // self.hop_length)
                onset_strength = F.interpolate(
                    onset_strength.unsqueeze(1),
                    size=target_length,
                    mode='linear',
                    align_corners=False
                ).squeeze(1)
                
            stfts.append(onset_strength)
            
        # Combine multi-resolution onsets
        combined_onset = torch.stack(stfts, dim=1).mean(dim=1)
        
        return combined_onset
        
    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Multi-scale onset detection.
        
        Args:
            waveform: Input audio [batch, samples]
            
        Returns:
            Dictionary with onset detection results
        """
        # Extract multi-scale mel-spectrograms
        mel_specs = {}
        spectral_fluxes = {}
        
        for scale, scale_config in self.scale_configs.items():
            # Use dynamic mel-spectrogram computation with scale-specific params
            mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=scale_config['n_fft'],
                hop_length=scale_config['hop_length'],
                n_mels=self.n_mels
            )(waveform)
            mel_spec_db = torchaudio.functional.amplitude_to_DB(mel_spec, multiplier=10.0, amin=1e-8)
            
            # Resample to common time grid
            target_length = int(waveform.shape[-1] // self.hop_length)
            if mel_spec_db.shape[-1] != target_length:
                mel_spec_db = F.interpolate(
                    mel_spec_db.unsqueeze(1),
                    size=(mel_spec_db.shape[1], target_length),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
                
            mel_specs[scale] = mel_spec_db
            spectral_fluxes[scale] = self.compute_spectral_flux(mel_spec_db, scale)
            
        # Complex domain onset detection
        complex_onset = self.compute_complex_domain_onset(waveform)
        
        # Concatenate multi-scale features for neural processing
        multi_scale_features = torch.cat(list(mel_specs.values()), dim=1)
        neural_onset = self.onset_network(multi_scale_features).squeeze(1)
        
        # Combine onset detectors with learned weights
        all_onsets = torch.stack([
            spectral_fluxes['fine'],
            spectral_fluxes['medium'], 
            spectral_fluxes['coarse'],
            complex_onset,
            neural_onset
        ], dim=1)
        
        # Learnable fusion weights
        if not hasattr(self, 'fusion_weights'):
            self.register_parameter('fusion_weights', 
                nn.Parameter(torch.ones(5) / 5))
            
        weights = F.softmax(self.fusion_weights, dim=0)
        combined_onset = torch.sum(all_onsets * weights.view(1, -1, 1), dim=1)
        
        # Peak picking with adaptive thresholding
        onset_peaks = self._pick_onset_peaks(combined_onset)
        
        return {
            'onset_strength': combined_onset,
            'spectral_flux_fine': spectral_fluxes['fine'],
            'spectral_flux_medium': spectral_fluxes['medium'],
            'spectral_flux_coarse': spectral_fluxes['coarse'],
            'complex_onset': complex_onset,
            'neural_onset': neural_onset,
            'onset_peaks': onset_peaks,
            'onset_times': self._frames_to_time(onset_peaks),
            'fusion_weights': weights
        }
        
    def _pick_onset_peaks(self, onset_strength: torch.Tensor) -> List[torch.Tensor]:
        """Advanced peak picking with adaptive thresholding."""
        batch_size = onset_strength.shape[0]
        all_peaks = []
        
        for b in range(batch_size):
            strength = onset_strength[b].cpu().numpy()
            
            # Adaptive threshold using local statistics
            # Use median-based threshold that adapts to local dynamics
            window_size = int(1.0 * self.sample_rate / self.hop_length)  # 1 second window
            
            # Compute local median threshold
            padded_strength = np.pad(strength, window_size//2, mode='edge')
            local_medians = np.array([
                np.median(padded_strength[i:i+window_size]) 
                for i in range(len(strength))
            ])
            
            # Dynamic threshold
            local_threshold = local_medians + self.onset_threshold * np.std(strength)
            
            # Find peaks above adaptive threshold
            peaks = []
            for i in range(1, len(strength) - 1):
                if (strength[i] > strength[i-1] and 
                    strength[i] > strength[i+1] and 
                    strength[i] > local_threshold[i]):
                    peaks.append(i)
                    
            # Filter peaks by minimum distance
            if peaks:
                min_distance = int(0.05 * self.sample_rate / self.hop_length)  # 50ms
                filtered_peaks = [peaks[0]]
                
                for peak in peaks[1:]:
                    if peak - filtered_peaks[-1] >= min_distance:
                        filtered_peaks.append(peak)
                        
                peaks_tensor = torch.tensor(filtered_peaks, dtype=torch.long)
            else:
                peaks_tensor = torch.tensor([], dtype=torch.long)
                
            all_peaks.append(peaks_tensor)
            
        return all_peaks
        
    def _frames_to_time(self, frame_peaks: List[torch.Tensor]) -> List[torch.Tensor]:
        """Convert frame indices to time in seconds."""
        times = []
        for peaks in frame_peaks:
            if len(peaks) > 0:
                time_peaks = peaks.float() * self.hop_length / self.sample_rate
            else:
                time_peaks = torch.tensor([])
            times.append(time_peaks)
        return times


class AdvancedTempoEstimator(nn.Module):
    """
    Advanced tempo estimation using neural networks and signal processing.
    
    Combines multiple estimation methods with confidence weighting.
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
        
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.tempo_min = tempo_min
        self.tempo_max = tempo_max
        self.tempo_bins = tempo_bins
        
        # Tempo grid
        self.register_buffer('tempo_grid', torch.linspace(tempo_min, tempo_max, tempo_bins))
        
        # Neural tempo estimator
        self.tempo_network = self._build_tempo_network()
        
    def _build_tempo_network(self):
        """Build neural network for tempo estimation."""
        return nn.Sequential(
            # Input: onset strength function
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
            
            # Global features
            nn.Flatten(),
            nn.Linear(256 * 64, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Tempo prediction
            nn.Linear(256, self.tempo_bins),
            nn.Softmax(dim=1)
        )
        
    def autocorrelation_tempo(self, onset_strength: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Enhanced autocorrelation-based tempo estimation."""
        batch_size = onset_strength.shape[0]
        tempos = []
        confidences = []
        autocorr_curves = []
        
        for b in range(batch_size):
            strength = onset_strength[b].cpu().numpy()
            
            # Pre-process onset strength
            # Apply exponential decay to emphasize recent onsets
            decay_factor = 0.99
            decayed_strength = strength * (decay_factor ** np.arange(len(strength)))
            
            # Autocorrelation with zero-padding
            autocorr = np.correlate(decayed_strength, decayed_strength, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Smooth autocorrelation
            from scipy.ndimage import gaussian_filter1d
            autocorr = gaussian_filter1d(autocorr, sigma=2.0)
            
            # Convert lags to tempo
            lags = np.arange(len(autocorr))
            lag_times = lags * self.hop_length / self.sample_rate
            
            # Valid tempo range
            min_lag = max(1, int(60 / self.tempo_max * self.sample_rate / self.hop_length))
            max_lag = min(len(autocorr), int(60 / self.tempo_min * self.sample_rate / self.hop_length))
            
            if max_lag > min_lag:
                valid_autocorr = autocorr[min_lag:max_lag]
                valid_lags = lags[min_lag:max_lag]
                
                # Find multiple peaks for tempo candidates
                peaks, properties = scipy.signal.find_peaks(
                    valid_autocorr,
                    height=np.max(valid_autocorr) * 0.3,
                    distance=int(0.1 * self.sample_rate / self.hop_length)
                )
                
                if len(peaks) > 0:
                    # Take strongest peak
                    strongest_peak_idx = np.argmax(valid_autocorr[peaks])
                    peak_lag = valid_lags[peaks[strongest_peak_idx]]
                    
                    tempo = 60 / (peak_lag * self.hop_length / self.sample_rate)
                    confidence = valid_autocorr[peaks[strongest_peak_idx]] / np.sum(valid_autocorr)
                else:
                    tempo = 120.0
                    confidence = 0.0
            else:
                tempo = 120.0
                confidence = 0.0
                valid_autocorr = np.array([])
                
            tempos.append(tempo)
            confidences.append(confidence)
            autocorr_curves.append(torch.from_numpy(valid_autocorr))
            
        return {
            'tempo': torch.tensor(tempos),
            'confidence': torch.tensor(confidences),
            'autocorr_curves': autocorr_curves
        }
        
    def comb_filter_tempo(self, onset_strength: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Enhanced comb filtering for tempo estimation."""
        batch_size = onset_strength.shape[0]
        tempo_scores = torch.zeros(batch_size, len(self.tempo_grid))
        
        for b in range(batch_size):
            strength = onset_strength[b]
            
            for t, tempo in enumerate(self.tempo_grid):
                # Multiple comb filters for different beat patterns
                beat_period = 60 / tempo * self.sample_rate / self.hop_length
                
                # Standard 4/4 pattern
                scores = []
                for pattern in [1, 2, 3, 4]:  # Different beat subdivisions
                    period = beat_period / pattern
                    max_delay = min(int(period * 8), len(strength) // 2)
                    
                    if max_delay > 1:
                        # Create comb filter
                        comb = torch.zeros(max_delay)
                        for i in range(0, max_delay, max(1, int(period))):
                            if i < max_delay:
                                comb[i] = 1.0
                                
                        # Normalize comb filter
                        comb = comb / (torch.sum(comb) + 1e-8)
                        
                        # Convolution score
                        if len(strength) >= len(comb):
                            conv_result = F.conv1d(
                                strength.unsqueeze(0).unsqueeze(0),
                                comb.flip(0).unsqueeze(0).unsqueeze(0),
                                padding=len(comb)//2
                            ).squeeze()
                            
                            score = torch.sum(conv_result * strength[:len(conv_result)])
                            scores.append(score)
                            
                if scores:
                    tempo_scores[b, t] = torch.max(torch.stack(scores))
                    
        # Normalize scores
        tempo_scores = tempo_scores / (torch.sum(tempo_scores, dim=1, keepdim=True) + 1e-8)
        
        # Find best tempo
        best_tempo_idx = torch.argmax(tempo_scores, dim=1)
        best_tempos = self.tempo_grid[best_tempo_idx]
        best_confidences = torch.gather(tempo_scores, 1, best_tempo_idx.unsqueeze(1)).squeeze(1)
        
        return {
            'tempo': best_tempos,
            'confidence': best_confidences,
            'tempo_scores': tempo_scores
        }
        
    def neural_tempo_estimation(self, onset_strength: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Neural network-based tempo estimation."""
        # Pad or truncate to fixed length for neural network
        target_length = 512  # Fixed input length
        batch_size = onset_strength.shape[0]
        
        processed_strength = torch.zeros(batch_size, target_length, device=onset_strength.device)
        
        for b in range(batch_size):
            strength = onset_strength[b]
            if len(strength) >= target_length:
                processed_strength[b] = strength[:target_length]
            else:
                processed_strength[b, :len(strength)] = strength
                
        # Neural prediction
        tempo_probs = self.tempo_network(processed_strength.unsqueeze(1))
        
        # Convert probabilities to tempo estimates
        expected_tempo = torch.sum(tempo_probs * self.tempo_grid.unsqueeze(0), dim=1)
        confidence = torch.max(tempo_probs, dim=1)[0]
        
        return {
            'tempo': expected_tempo,
            'confidence': confidence,
            'tempo_probs': tempo_probs
        }
        
    def forward(self, onset_strength: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Comprehensive tempo estimation.
        
        Args:
            onset_strength: Onset strength function [batch, time]
            
        Returns:
            Dictionary with tempo estimates from multiple methods
        """
        # Multiple estimation methods
        autocorr_result = self.autocorrelation_tempo(onset_strength)
        comb_result = self.comb_filter_tempo(onset_strength)
        neural_result = self.neural_tempo_estimation(onset_strength)
        
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
            'tempo_scores': comb_result['tempo_scores'],
            'individual_confidences': confidences
        }


class BeatSynchronizer(AudioModuleBase):
    """
    Advanced beat synchronization and rhythm analysis for music processing.
    
    Combines onset detection, tempo estimation, and beat tracking for
    precise rhythmic analysis and synchronization.
    """
    
    def __init__(self, config: AudioModuleConfig):
        super().__init__(config)
        
        # Core components
        self.onset_detector = MultiScaleOnsetDetector(
            sample_rate=config.sample_rate,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            n_fft=config.n_fft
        )
        
        self.tempo_estimator = AdvancedTempoEstimator(
            sample_rate=config.sample_rate,
            hop_length=config.hop_length
        )
        
        # Beat tracking parameters
        self.alpha = 0.8  # Tempo consistency weight
        self.beta = 0.2   # Onset strength weight
        self.transition_sigma = 0.1  # Tempo transition tolerance
        
    def dynamic_programming_beats(
        self,
        onset_strength: torch.Tensor,
        tempo: float,
        tempo_confidence: float
    ) -> torch.Tensor:
        """
        Advanced dynamic programming beat tracking with tempo uncertainty.
        """
        onset_np = onset_strength.cpu().numpy()
        n_frames = len(onset_np)
        
        if n_frames < 2:
            return torch.tensor([])
            
        # Beat period in frames
        beat_period = 60 / tempo * self.config.sample_rate / self.config.hop_length
        
        # Create state space with tempo uncertainty
        tempo_variance = max(0.05, 1.0 - tempo_confidence) * beat_period
        
        # Possible beat positions (every frame is a potential beat)
        n_states = n_frames
        
        # DP table: [time, state]
        dp_score = np.full((n_frames, n_states), -np.inf)
        dp_path = np.zeros((n_frames, n_states), dtype=int)
        
        # Initialize first frame - any position could be first beat
        for s in range(min(n_states, int(beat_period))):
            dp_score[0, s] = onset_np[0]
            
        # Fill DP table
        for t in range(1, n_frames):
            for curr_state in range(n_states):
                # Look at previous beat positions
                for prev_state in range(n_states):
                    interval = curr_state - prev_state
                    
                    if interval <= 0:
                        continue
                        
                    # Transition probability based on expected beat period
                    tempo_cost = -0.5 * ((interval - beat_period) / (self.transition_sigma * beat_period)) ** 2
                    
                    # Onset strength at current position
                    onset_cost = onset_np[curr_state] if curr_state < len(onset_np) else 0
                    
                    # Total score
                    score = (
                        dp_score[t-1, prev_state] +
                        self.alpha * tempo_cost +
                        self.beta * onset_cost
                    )
                    
                    if score > dp_score[t, curr_state]:
                        dp_score[t, curr_state] = score
                        dp_path[t, curr_state] = prev_state
                        
        # Backtrack best path
        best_states = []
        best_final_state = np.argmax(dp_score[-1, :])
        
        current_state = best_final_state
        for t in range(n_frames - 1, -1, -1):
            best_states.append(current_state)
            if t > 0:
                current_state = dp_path[t, current_state]
                
        best_states.reverse()
        
        # Filter to actual beat positions (remove consecutive duplicates)
        beat_positions = []
        last_pos = -1
        
        for pos in best_states:
            if pos != last_pos and (not beat_positions or pos - beat_positions[-1] >= beat_period * 0.5):
                beat_positions.append(pos)
                last_pos = pos
                
        return torch.tensor(beat_positions, dtype=torch.float32)
        
    def synchronize_to_grid(
        self,
        beat_times: torch.Tensor,
        audio_length: float,
        target_tempo: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Synchronize beats to a regular grid for rhythm analysis.
        
        Args:
            beat_times: Detected beat times [n_beats]
            audio_length: Total audio length in seconds
            target_tempo: Optional target tempo for grid
            
        Returns:
            Synchronized beat grid and analysis
        """
        if len(beat_times) < 2:
            return {
                'beat_grid': torch.zeros(0),
                'sync_error': torch.tensor(float('inf')),
                'grid_tempo': torch.tensor(120.0),
                'phase_offset': torch.tensor(0.0)
            }
            
        # Estimate grid tempo from beat intervals
        intervals = torch.diff(beat_times)
        median_interval = torch.median(intervals)
        estimated_tempo = 60.0 / median_interval
        
        grid_tempo = target_tempo if target_tempo is not None else estimated_tempo
        grid_interval = 60.0 / grid_tempo
        
        # Create regular grid
        n_grid_beats = int(audio_length / grid_interval) + 1
        regular_grid = torch.arange(n_grid_beats, dtype=torch.float32) * grid_interval
        
        # Find optimal phase offset
        best_offset = 0.0
        best_error = float('inf')
        
        # Try different phase offsets
        for offset in torch.linspace(0, grid_interval, 50):
            shifted_grid = regular_grid + offset
            
            # Find closest grid points for each detected beat
            errors = []
            for beat in beat_times:
                closest_grid_beat = shifted_grid[torch.argmin(torch.abs(shifted_grid - beat))]
                error = torch.abs(beat - closest_grid_beat)
                if error < grid_interval / 2:  # Only count reasonable matches
                    errors.append(error)
                    
            if errors:
                avg_error = torch.mean(torch.stack(errors))
                if avg_error < best_error:
                    best_error = avg_error
                    best_offset = offset
                    
        # Final synchronized grid
        sync_grid = regular_grid + best_offset
        
        return {
            'beat_grid': sync_grid,
            'sync_error': torch.tensor(best_error),
            'grid_tempo': torch.tensor(grid_tempo.item() if torch.is_tensor(grid_tempo) else grid_tempo),
            'phase_offset': torch.tensor(best_offset),
            'grid_interval': torch.tensor(grid_interval)
        }
        
    def analyze_rhythmic_features(
        self,
        beat_times: torch.Tensor,
        onset_times: torch.Tensor,
        audio_length: float
    ) -> Dict[str, torch.Tensor]:
        """Comprehensive rhythmic feature analysis."""
        if len(beat_times) < 2:
            return {
                'beat_consistency': torch.tensor(0.0),
                'onset_density': torch.tensor(0.0),
                'syncopation': torch.tensor(0.0),
                'groove_deviation': torch.tensor(0.0)
            }
            
        # Beat consistency (regularity)
        intervals = torch.diff(beat_times)
        beat_consistency = 1.0 / (1.0 + torch.std(intervals) / torch.mean(intervals))
        
        # Onset density
        onset_density = len(onset_times) / audio_length
        
        # Syncopation analysis
        median_interval = torch.median(intervals)
        
        # For each onset, find distance to nearest beat
        syncopation_scores = []
        for onset in onset_times:
            distances = torch.abs(beat_times - onset)
            min_distance = torch.min(distances)
            
            # Syncopation is high when onsets are far from beats
            sync_score = min_distance / (median_interval / 2)  # Normalize by half beat
            syncopation_scores.append(torch.clamp(sync_score, 0, 1))
            
        syncopation = torch.mean(torch.stack(syncopation_scores)) if syncopation_scores else torch.tensor(0.0)
        
        # Groove deviation (micro-timing)
        groove_deviations = []
        for i, beat in enumerate(beat_times):
            expected_time = beat_times[0] + i * median_interval
            deviation = torch.abs(beat - expected_time)
            groove_deviations.append(deviation)
            
        groove_deviation = torch.mean(torch.stack(groove_deviations)) if groove_deviations else torch.tensor(0.0)
        
        return {
            'beat_consistency': beat_consistency,
            'onset_density': torch.tensor(onset_density),
            'syncopation': syncopation,
            'groove_deviation': groove_deviation,
            'tempo_stability': 1.0 / (1.0 + torch.std(intervals)),
            'rhythmic_complexity': torch.std(intervals) / torch.mean(intervals)
        }
        
    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Complete beat synchronization and rhythm analysis.
        
        Args:
            waveform: Input audio [batch, samples]
            
        Returns:
            Comprehensive beat and rhythm analysis
        """
        batch_results = []
        audio_length = waveform.shape[-1] / self.config.sample_rate
        
        for b in range(waveform.shape[0]):
            audio_sample = waveform[b:b+1]
            
            # Onset detection
            onset_results = self.onset_detector(audio_sample)
            onset_strength = onset_results['onset_strength'][0]
            onset_times = onset_results['onset_times'][0]
            
            # Tempo estimation
            tempo_results = self.tempo_estimator(onset_strength.unsqueeze(0))
            tempo = tempo_results['tempo'][0].item()
            tempo_confidence = tempo_results['confidence'][0].item()
            
            # Beat tracking
            beat_positions = self.dynamic_programming_beats(
                onset_strength, tempo, tempo_confidence
            )
            
            if len(beat_positions) > 0:
                beat_times = beat_positions * self.config.hop_length / self.config.sample_rate
                
                # Beat synchronization
                sync_results = self.synchronize_to_grid(beat_times, audio_length, tempo)
                
                # Rhythmic analysis
                rhythm_features = self.analyze_rhythmic_features(
                    beat_times, onset_times, audio_length
                )
            else:
                beat_times = torch.tensor([])
                sync_results = {
                    'beat_grid': torch.zeros(0),
                    'sync_error': torch.tensor(float('inf')),
                    'grid_tempo': torch.tensor(tempo),
                    'phase_offset': torch.tensor(0.0)
                }
                rhythm_features = {
                    'beat_consistency': torch.tensor(0.0),
                    'onset_density': torch.tensor(len(onset_times) / audio_length),
                    'syncopation': torch.tensor(0.0),
                    'groove_deviation': torch.tensor(0.0)
                }
                
            # Combine results for this sample
            sample_result = {
                'onset_strength': onset_strength,
                'onset_times': onset_times,
                'tempo': torch.tensor(tempo),
                'tempo_confidence': torch.tensor(tempo_confidence),
                'beat_times': beat_times,
                'sync_results': sync_results,
                'rhythm_features': rhythm_features
            }
            
            batch_results.append(sample_result)
            
        return batch_results


# Factory functions
def create_beat_synchronizer(config: Optional[AudioModuleConfig] = None) -> BeatSynchronizer:
    """Create beat synchronizer with configuration."""
    if config is None:
        from ..audio_analysis.audio_config import get_music_config
        config = get_music_config()
    return BeatSynchronizer(config)


# Example usage
if __name__ == "__main__":
    from ..audio_analysis.audio_config import get_music_config
    
    # Create beat synchronizer
    config = get_music_config()
    synchronizer = create_beat_synchronizer(config)
    
    # Test with synthetic drum pattern
    sample_rate = config.sample_rate
    duration = 8  # seconds
    waveform = torch.zeros(1, sample_rate * duration)
    
    # Add synthetic beats every 0.5 seconds (120 BPM)
    beat_interval = 0.5
    for beat_time in torch.arange(0, duration, beat_interval):
        beat_sample = int(beat_time * sample_rate)
        if beat_sample < waveform.shape[1] - 1000:
            # Add kick drum-like sound
            kick = torch.sin(2 * torch.pi * 60 * torch.linspace(0, 0.1, int(0.1 * sample_rate)))
            kick *= torch.exp(-5 * torch.linspace(0, 0.1, len(kick)))
            waveform[0, beat_sample:beat_sample+len(kick)] += kick
            
    # Add some off-beat snares
    for snare_time in torch.arange(beat_interval/2, duration, beat_interval):
        snare_sample = int(snare_time * sample_rate)
        if snare_sample < waveform.shape[1] - 500:
            # Add snare-like sound
            snare = torch.randn(500) * 0.5 * torch.exp(-10 * torch.linspace(0, 0.1, 500))
            waveform[0, snare_sample:snare_sample+len(snare)] += snare
            
    # Analyze
    results = synchronizer(waveform)
    sample_result = results[0]
    
    print(f"Detected tempo: {sample_result['tempo']:.1f} BPM")
    print(f"Tempo confidence: {sample_result['tempo_confidence']:.3f}")
    print(f"Number of beats detected: {len(sample_result['beat_times'])}")
    print(f"Beat consistency: {sample_result['rhythm_features']['beat_consistency']:.3f}")
    print(f"Syncopation level: {sample_result['rhythm_features']['syncopation']:.3f}")
    print(f"Grid sync error: {sample_result['sync_results']['sync_error']:.3f}s")
    
    if len(sample_result['beat_times']) > 1:
        actual_tempo = 60.0 / torch.median(torch.diff(sample_result['beat_times']))
        print(f"Measured tempo from beats: {actual_tempo:.1f} BPM")