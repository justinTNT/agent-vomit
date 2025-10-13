"""
BeatGridExtractor - Extract precise beat grid from audio segments

This module provides sample-accurate beat detection with confidence scoring,
forming the foundation for all timing-based crossfade decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from typing import NamedTuple, Optional
from dataclasses import dataclass


@dataclass 
class BeatGrid:
    """Beat grid analysis results"""
    bpm: float                          # Detected tempo
    beat_times: torch.Tensor            # Beat positions in samples
    bar_times: torch.Tensor             # Bar positions in samples  
    confidence_curve: torch.Tensor      # Per-beat confidence scores [0,1]
    tempo_stability: float              # How stable is tempo [0,1]
    sample_rate: int                    # Sample rate for beat_times


class BeatGridExtractor(nn.Module):
    """
    Extract precise beat grid from 30s audio segments.
    
    Uses multiple beat tracking algorithms and combines results
    for robust, sample-accurate beat detection.
    """
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 hop_length: int = 512,
                 n_fft: int = 2048,
                 min_bpm: float = 60.0,
                 max_bpm: float = 200.0):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        
        # Spectral features for onset detection
        self.spectral_conv = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=7,
            padding=3,
            bias=False
        )
        
        # Temporal modeling for beat tracking  
        self.temporal_lstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Beat confidence prediction
        self.beat_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Tempo estimation head
        self.tempo_predictor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, audio: torch.Tensor) -> BeatGrid:
        """
        Extract beat grid from audio segment.
        
        Args:
            audio: (batch_size, samples) or (samples,) mono audio
            
        Returns:
            BeatGrid with sample-accurate beat positions and confidence
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # Add batch dimension
            
        batch_size = audio.size(0)
        
        # Convert to numpy for librosa processing
        audio_np = audio.detach().cpu().numpy()
        
        beat_grids = []
        for i in range(batch_size):
            beat_grid = self._extract_single_beat_grid(audio_np[i])
            beat_grids.append(beat_grid)
            
        # Return first if single audio, else batch results
        return beat_grids[0] if batch_size == 1 else beat_grids
    
    def _extract_single_beat_grid(self, audio: np.ndarray) -> BeatGrid:
        """Extract beat grid from single audio segment"""
        
        # Multi-algorithm beat tracking for robustness
        tempo, beats = self._librosa_beat_track(audio)
        neural_beats, neural_confidence = self._neural_beat_track(audio)
        
        # Combine results with confidence weighting
        final_beats, confidence = self._combine_beat_predictions(
            beats, neural_beats, neural_confidence
        )
        
        # Convert to sample positions
        beat_samples = librosa.frames_to_samples(
            final_beats, 
            hop_length=self.hop_length
        )
        
        # Estimate bar positions (assume 4/4 time)
        bar_samples = self._estimate_bars(beat_samples, tempo)
        
        # Calculate tempo stability
        tempo_stability = self._calculate_tempo_stability(beat_samples)
        
        return BeatGrid(
            bpm=float(tempo),
            beat_times=torch.from_numpy(beat_samples).long(),
            bar_times=torch.from_numpy(bar_samples).long(),
            confidence_curve=torch.from_numpy(confidence).float(),
            tempo_stability=float(tempo_stability),
            sample_rate=self.sample_rate
        )
    
    def _librosa_beat_track(self, audio: np.ndarray) -> tuple[float, np.ndarray]:
        """Use librosa for initial beat tracking"""
        tempo, beats = librosa.beat.beat_track(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            start_bpm=120.0,
            tightness=100
        )
        return tempo, beats
    
    def _neural_beat_track(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Use neural network for beat refinement"""
        
        # Spectral features
        stft = librosa.stft(
            audio, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
        magnitude = np.abs(stft)
        
        # Onset strength
        onset_strength = librosa.onset.onset_strength(
            S=magnitude, 
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # Convert to tensor
        features = torch.from_numpy(onset_strength).float().unsqueeze(0).unsqueeze(0)
        
        # Neural processing
        spectral_out = self.spectral_conv(features)  # (1, 32, time)
        spectral_out = spectral_out.transpose(1, 2)  # (1, time, 32)
        
        lstm_out, _ = self.temporal_lstm(spectral_out)  # (1, time, 64)
        
        # Beat prediction
        beat_confidence = self.beat_classifier(lstm_out)  # (1, time, 1)
        beat_confidence = beat_confidence.squeeze().detach().cpu().numpy()
        
        # Find peaks as beat candidates
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(
            beat_confidence, 
            height=0.3,
            distance=int(0.2 * self.sample_rate / self.hop_length)  # Min 200ms apart
        )
        
        return peaks, beat_confidence
    
    def _combine_beat_predictions(self, 
                                 librosa_beats: np.ndarray,
                                 neural_beats: np.ndarray, 
                                 neural_confidence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Combine librosa and neural beat predictions"""
        
        # Simple approach: use librosa beats as base, adjust with neural confidence
        final_beats = librosa_beats.copy()
        
        # Create confidence array for librosa beats
        confidence = np.ones(len(librosa_beats)) * 0.7  # Base confidence
        
        # Boost confidence where neural network agrees
        for i, beat in enumerate(librosa_beats):
            # Find nearest neural beat
            if len(neural_beats) > 0:
                distances = np.abs(neural_beats - beat)
                nearest_idx = np.argmin(distances)
                
                # If neural beat is close, boost confidence
                if distances[nearest_idx] < 3:  # Within 3 frames
                    neural_conf = neural_confidence[neural_beats[nearest_idx]]
                    confidence[i] = min(0.95, confidence[i] + neural_conf * 0.3)
        
        return final_beats, confidence
    
    def _estimate_bars(self, beat_samples: np.ndarray, tempo: float) -> np.ndarray:
        """Estimate bar positions assuming 4/4 time"""
        if len(beat_samples) < 4:
            return np.array([beat_samples[0]] if len(beat_samples) > 0 else [0])
        
        # Take every 4th beat as bar start
        bars = beat_samples[::4]
        return bars
    
    def _calculate_tempo_stability(self, beat_samples: np.ndarray) -> float:
        """Calculate how stable the tempo is"""
        if len(beat_samples) < 3:
            return 0.5
        
        # Calculate inter-beat intervals
        intervals = np.diff(beat_samples)
        
        # Tempo stability is inverse of coefficient of variation
        if np.mean(intervals) == 0:
            return 0.5
            
        cv = np.std(intervals) / np.mean(intervals)
        stability = max(0.0, 1.0 - cv)
        
        return min(1.0, stability)
    
    def get_beat_at_time(self, beat_grid: BeatGrid, sample_pos: int) -> Optional[int]:
        """Find the beat index closest to given sample position"""
        if len(beat_grid.beat_times) == 0:
            return None
            
        distances = torch.abs(beat_grid.beat_times - sample_pos)
        beat_idx = torch.argmin(distances)
        
        return int(beat_idx)
    
    def get_confidence_at_beat(self, beat_grid: BeatGrid, beat_idx: int) -> float:
        """Get confidence score for specific beat"""
        if beat_idx < 0 or beat_idx >= len(beat_grid.confidence_curve):
            return 0.0
            
        return float(beat_grid.confidence_curve[beat_idx])


def create_test_audio(duration: float = 30.0, bpm: float = 120.0, sr: int = 44100) -> torch.Tensor:
    """Create test audio with clear beats for testing"""
    
    # Generate click track
    samples = int(duration * sr)
    audio = torch.zeros(samples)
    
    # Add clicks on beats
    beat_interval = int(60.0 / bpm * sr)  # Samples per beat
    
    for i in range(0, samples, beat_interval):
        if i < samples:
            # Short click
            click_len = min(1000, samples - i)
            click = torch.sin(2 * torch.pi * 1000 * torch.linspace(0, 0.02, click_len))
            click *= torch.exp(-10 * torch.linspace(0, 0.02, click_len))  # Decay
            audio[i:i+click_len] = click
    
    return audio


if __name__ == "__main__":
    # Test the module
    extractor = BeatGridExtractor()
    test_audio = create_test_audio(duration=10.0, bpm=120.0)
    
    print(f"Test audio shape: {test_audio.shape}")
    
    # Extract beat grid
    beat_grid = extractor(test_audio)
    
    print(f"Detected BPM: {beat_grid.bpm:.2f}")
    print(f"Number of beats: {len(beat_grid.beat_times)}")
    print(f"Number of bars: {len(beat_grid.bar_times)}")
    print(f"Tempo stability: {beat_grid.tempo_stability:.3f}")
    print(f"Average confidence: {beat_grid.confidence_curve.mean():.3f}")