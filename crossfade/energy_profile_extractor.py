"""
EnergyProfileExtractor - Extract energy characteristics over time

This module analyzes RMS energy, spectral energy distribution, and dynamic range
for energy-matched crossfade transitions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class EnergyProfile:
    """Energy analysis results"""
    rms_curve: torch.Tensor           # RMS energy over time
    spectral_bands: torch.Tensor      # Energy in low/mid/high bands [3, time]
    dynamics: dict                    # Dynamic range statistics
    energy_slope: torch.Tensor        # Energy trajectory (building/falling)
    peak_positions: torch.Tensor      # Sample positions of energy peaks
    loudness_curve: torch.Tensor      # Perceptual loudness over time
    sample_rate: int                  # For time conversion


class EnergyProfileExtractor(nn.Module):
    """
    Extract comprehensive energy profile from audio segments.
    
    Analyzes multiple aspects of energy for crossfade optimization:
    - RMS energy progression
    - Frequency-band energy distribution  
    - Dynamic range and loudness
    - Energy trajectory analysis
    """
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 hop_length: int = 512,
                 n_fft: int = 2048,
                 frame_length: int = 2048):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.frame_length = frame_length
        
        # Frequency band boundaries (Hz)
        self.low_freq_max = 200
        self.mid_freq_max = 2000
        # High freq is everything above mid_freq_max
        
        # Neural network for perceptual loudness modeling
        self.loudness_predictor = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Energy slope prediction
        self.slope_analyzer = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(8, 1, kernel_size=1),
            nn.Tanh()  # Output range [-1, 1] for falling/rising
        )

    def forward(self, audio: torch.Tensor) -> EnergyProfile:
        """
        Extract energy profile from audio segment.
        
        Args:
            audio: (batch_size, samples) or (samples,) mono audio
            
        Returns:
            EnergyProfile with comprehensive energy analysis
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            
        # Process single audio (extend for batch later)
        audio_np = audio[0].detach().cpu().numpy()
        
        # Extract RMS energy
        rms_curve = self._extract_rms_curve(audio_np)
        
        # Extract spectral band energy
        spectral_bands = self._extract_spectral_bands(audio_np)
        
        # Calculate dynamics
        dynamics = self._calculate_dynamics(rms_curve, spectral_bands)
        
        # Find energy peaks
        peak_positions = self._find_energy_peaks(rms_curve)
        
        # Predict perceptual loudness
        loudness_curve = self._predict_loudness(rms_curve)
        
        # Analyze energy slope
        energy_slope = self._analyze_energy_slope(rms_curve)
        
        return EnergyProfile(
            rms_curve=torch.from_numpy(rms_curve).float(),
            spectral_bands=torch.from_numpy(spectral_bands).float(),
            dynamics=dynamics,
            energy_slope=energy_slope,
            peak_positions=torch.from_numpy(peak_positions).long(),
            loudness_curve=loudness_curve,
            sample_rate=self.sample_rate
        )
    
    def _extract_rms_curve(self, audio: np.ndarray) -> np.ndarray:
        """Extract RMS energy curve over time"""
        
        # Use librosa for consistent frame analysis
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            center=True
        )
        
        # Convert to dB with floor to avoid -inf
        rms_db = librosa.amplitude_to_db(rms + 1e-10, ref=np.max)
        
        return rms_db.flatten()
    
    def _extract_spectral_bands(self, audio: np.ndarray) -> np.ndarray:
        """Extract energy in low/mid/high frequency bands"""
        
        # Compute STFT
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=True
        )
        magnitude = np.abs(stft)
        
        # Frequency bins
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)
        
        # Find frequency band indices
        low_idx = np.where(freqs <= self.low_freq_max)[0][-1]
        mid_idx = np.where(freqs <= self.mid_freq_max)[0][-1]
        
        # Extract band energies
        low_energy = np.sum(magnitude[:low_idx+1, :], axis=0)
        mid_energy = np.sum(magnitude[low_idx+1:mid_idx+1, :], axis=0)
        high_energy = np.sum(magnitude[mid_idx+1:, :], axis=0)
        
        # Stack into bands array
        bands = np.vstack([low_energy, mid_energy, high_energy])
        
        # Convert to dB
        bands_db = librosa.amplitude_to_db(bands + 1e-10, ref=np.max)
        
        return bands_db
    
    def _calculate_dynamics(self, rms_curve: np.ndarray, spectral_bands: np.ndarray) -> dict:
        """Calculate dynamic range statistics"""
        
        # Overall dynamic range
        dynamic_range = np.max(rms_curve) - np.min(rms_curve)
        
        # RMS statistics
        rms_mean = np.mean(rms_curve)
        rms_std = np.std(rms_curve)
        
        # Peak-to-average ratio
        rms_linear = librosa.db_to_amplitude(rms_curve)
        peak_avg_ratio = np.max(rms_linear) / (np.mean(rms_linear) + 1e-10)
        
        # Spectral balance (ratio between bands)
        band_means = np.mean(spectral_bands, axis=1)
        low_mid_ratio = band_means[0] - band_means[1]  # dB difference
        mid_high_ratio = band_means[1] - band_means[2]
        
        # Crest factor (peakiness measure)
        crest_factor = np.max(rms_linear) / np.sqrt(np.mean(rms_linear**2) + 1e-10)
        
        return {
            'dynamic_range_db': float(dynamic_range),
            'rms_mean_db': float(rms_mean),
            'rms_std_db': float(rms_std),
            'peak_avg_ratio': float(peak_avg_ratio),
            'low_mid_ratio_db': float(low_mid_ratio),
            'mid_high_ratio_db': float(mid_high_ratio),
            'crest_factor': float(crest_factor)
        }
    
    def _find_energy_peaks(self, rms_curve: np.ndarray) -> np.ndarray:
        """Find significant energy peaks for 'drop on the 1' detection"""
        
        from scipy.signal import find_peaks
        
        # Find peaks with minimum height and distance
        rms_linear = librosa.db_to_amplitude(rms_curve)
        
        peaks, properties = find_peaks(
            rms_linear,
            height=np.mean(rms_linear) + 0.5 * np.std(rms_linear),
            distance=int(0.5 * self.sample_rate / self.hop_length),  # Min 0.5s apart
            prominence=0.1 * (np.max(rms_linear) - np.min(rms_linear))
        )
        
        # Convert to sample positions
        peak_samples = peaks * self.hop_length
        
        return peak_samples
    
    def _predict_loudness(self, rms_curve: np.ndarray) -> torch.Tensor:
        """Predict perceptual loudness using neural network"""
        
        # Prepare input
        rms_tensor = torch.from_numpy(rms_curve).float()
        rms_tensor = (rms_tensor - rms_tensor.mean()) / (rms_tensor.std() + 1e-8)  # Normalize
        rms_input = rms_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, time)
        
        # Predict loudness
        with torch.no_grad():
            loudness = self.loudness_predictor(rms_input)
            
        return loudness.squeeze()
    
    def _analyze_energy_slope(self, rms_curve: np.ndarray) -> torch.Tensor:
        """Analyze energy trajectory (building up, falling down, stable)"""
        
        # Prepare input
        rms_tensor = torch.from_numpy(rms_curve).float()
        rms_tensor = (rms_tensor - rms_tensor.mean()) / (rms_tensor.std() + 1e-8)
        rms_input = rms_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, time)
        
        # Predict slope
        with torch.no_grad():
            slope = self.slope_analyzer(rms_input)
            
        return slope.squeeze()
    
    def get_energy_at_time(self, profile: EnergyProfile, sample_pos: int) -> dict:
        """Get energy information at specific sample position"""
        
        # Convert sample position to frame
        frame_pos = int(sample_pos // self.hop_length)
        frame_pos = max(0, min(frame_pos, len(profile.rms_curve) - 1))
        
        return {
            'rms_db': float(profile.rms_curve[frame_pos]),
            'low_band_db': float(profile.spectral_bands[0, frame_pos]),
            'mid_band_db': float(profile.spectral_bands[1, frame_pos]),
            'high_band_db': float(profile.spectral_bands[2, frame_pos]),
            'loudness': float(profile.loudness_curve[frame_pos]),
            'energy_slope': float(profile.energy_slope[frame_pos])
        }
    
    def find_quiet_sections(self, profile: EnergyProfile, 
                           threshold_db: float = -40.0,
                           min_duration_ms: float = 500.0) -> list[tuple[int, int]]:
        """Find quiet sections suitable for crossfade entry/exit"""
        
        # Find frames below threshold
        quiet_mask = profile.rms_curve < threshold_db
        
        # Convert minimum duration to frames
        min_frames = int(min_duration_ms * self.sample_rate / 1000 / self.hop_length)
        
        # Find contiguous quiet regions
        quiet_sections = []
        in_quiet = False
        start_frame = 0
        
        for i, is_quiet in enumerate(quiet_mask):
            if is_quiet and not in_quiet:
                # Start of quiet section
                start_frame = i
                in_quiet = True
            elif not is_quiet and in_quiet:
                # End of quiet section
                if i - start_frame >= min_frames:
                    start_sample = start_frame * self.hop_length
                    end_sample = i * self.hop_length
                    quiet_sections.append((start_sample, end_sample))
                in_quiet = False
        
        # Check final section
        if in_quiet and len(quiet_mask) - start_frame >= min_frames:
            start_sample = start_frame * self.hop_length
            end_sample = len(quiet_mask) * self.hop_length
            quiet_sections.append((start_sample, end_sample))
        
        return quiet_sections
    
    def calculate_energy_compatibility(self, profile1: EnergyProfile, 
                                     profile2: EnergyProfile,
                                     pos1: int, pos2: int) -> float:
        """Calculate energy compatibility between two positions"""
        
        energy1 = self.get_energy_at_time(profile1, pos1)
        energy2 = self.get_energy_at_time(profile2, pos2)
        
        # RMS compatibility (prefer similar levels)
        rms_diff = abs(energy1['rms_db'] - energy2['rms_db'])
        rms_compat = max(0.0, 1.0 - rms_diff / 30.0)  # 30dB max difference
        
        # Spectral balance compatibility
        spectral_diff = 0
        for band in ['low_band_db', 'mid_band_db', 'high_band_db']:
            spectral_diff += abs(energy1[band] - energy2[band])
        spectral_compat = max(0.0, 1.0 - spectral_diff / 60.0)  # 60dB max total diff
        
        # Slope compatibility (avoid energy direction conflicts)
        slope_diff = abs(energy1['energy_slope'] - energy2['energy_slope'])
        slope_compat = max(0.0, 1.0 - slope_diff)
        
        # Weighted combination
        compatibility = (
            0.5 * rms_compat +
            0.3 * spectral_compat +
            0.2 * slope_compat
        )
        
        return min(1.0, max(0.0, compatibility))


def create_test_audio_with_energy_pattern(duration: float = 10.0, 
                                        pattern: str = 'buildup',
                                        sr: int = 44100) -> torch.Tensor:
    """Create test audio with specific energy patterns"""
    
    samples = int(duration * sr)
    t = torch.linspace(0, duration, samples)
    
    # Base noise
    audio = 0.1 * torch.randn(samples)
    
    if pattern == 'buildup':
        # Energy builds up over time
        envelope = torch.linspace(0.1, 1.0, samples)
        freq = 440.0 + 100 * t  # Rising frequency
        
    elif pattern == 'drop':
        # Sudden energy drop
        drop_point = int(0.3 * samples)
        envelope = torch.ones(samples)
        envelope[drop_point:] *= 0.2
        freq = 440.0
        
    elif pattern == 'stable':
        # Stable energy
        envelope = 0.5 * torch.ones(samples)
        freq = 440.0
        
    elif pattern == 'fade_out':
        # Gradual fade out
        envelope = torch.linspace(1.0, 0.1, samples)
        freq = 440.0
        
    else:
        envelope = torch.ones(samples)
        freq = 440.0
    
    # Generate tone with envelope
    tone = envelope * torch.sin(2 * torch.pi * freq * t)
    
    # Add some harmonics for spectral content
    tone += 0.3 * envelope * torch.sin(2 * torch.pi * 2 * freq * t)
    tone += 0.1 * envelope * torch.sin(2 * torch.pi * 3 * freq * t)
    
    # Mix with noise
    audio = 0.7 * tone + 0.3 * audio * envelope
    
    return audio


if __name__ == "__main__":
    # Test the module
    extractor = EnergyProfileExtractor()
    
    # Test with buildup pattern
    test_audio = create_test_audio_with_energy_pattern(
        duration=10.0, 
        pattern='buildup'
    )
    print(f"Test audio shape: {test_audio.shape}")
    
    # Extract energy profile
    energy_profile = extractor(test_audio)
    
    print(f"RMS curve length: {len(energy_profile.rms_curve)}")
    print(f"Spectral bands shape: {energy_profile.spectral_bands.shape}")
    print(f"Dynamic range: {energy_profile.dynamics['dynamic_range_db']:.1f} dB")
    print(f"Number of energy peaks: {len(energy_profile.peak_positions)}")
    print(f"Energy slope range: [{energy_profile.energy_slope.min():.2f}, {energy_profile.energy_slope.max():.2f}]")
    
    # Test energy at specific time
    energy_info = extractor.get_energy_at_time(energy_profile, 5 * 44100)  # 5 seconds
    print(f"Energy at 5s: {energy_info['rms_db']:.1f} dB")
    
    # Test quiet section detection
    quiet_sections = extractor.find_quiet_sections(energy_profile)
    print(f"Found {len(quiet_sections)} quiet sections")