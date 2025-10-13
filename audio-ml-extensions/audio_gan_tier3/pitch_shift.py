"""
PitchShift module for audio pitch manipulation.

This module provides pitch shifting capabilities for audio signals,
allowing pitch changes without tempo changes. Implements various
algorithms including phase vocoder and time-domain methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, Tuple


class PhaseVocoderPitchShift(nn.Module):
    """
    Phase vocoder-based pitch shifting.
    
    Combines time stretching with resampling to achieve pitch shifting
    while maintaining duration. Classic approach for pitch manipulation.
    
    Args:
        n_fft: FFT size
        hop_length: Hop length for STFT
        win_length: Window length (default: n_fft)
        window: Window function type
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: str = 'hann',
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4
        self.win_length = win_length if win_length is not None else n_fft
        
        # Create window
        self.register_buffer('window', self._create_window(window))
        
        # Phase advance for each frequency bin
        freq_bins = torch.arange(n_fft // 2 + 1, dtype=torch.float32)
        self.register_buffer('phase_advance', 
                           2 * np.pi * freq_bins * self.hop_length / n_fft)
    
    def _create_window(self, window_type: str) -> torch.Tensor:
        """Create window function."""
        if window_type == 'hann':
            return torch.hann_window(self.win_length)
        elif window_type == 'hamming':
            return torch.hamming_window(self.win_length)
        elif window_type == 'blackman':
            return torch.blackman_window(self.win_length)
        else:
            raise ValueError(f"Unknown window type: {window_type}")
    
    def forward(
        self,
        audio: torch.Tensor,
        pitch_shift: float
    ) -> torch.Tensor:
        """
        Apply pitch shifting to audio.
        
        Args:
            audio: Input audio (batch, time) or (batch, channels, time)
            pitch_shift: Pitch shift in semitones (positive = higher, negative = lower)
            
        Returns:
            Pitch-shifted audio of same length
        """
        # Convert semitones to frequency ratio
        freq_ratio = 2 ** (pitch_shift / 12.0)
        
        # Handle input shape
        original_shape = audio.shape
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # Add channel dimension
        
        batch_size, channels, length = audio.shape
        
        # Process each channel
        shifted_channels = []
        for c in range(channels):
            shifted = self._shift_channel(audio[:, c, :], freq_ratio, length)
            shifted_channels.append(shifted)
        
        result = torch.stack(shifted_channels, dim=1)
        
        # Restore original shape
        if len(original_shape) == 2:
            result = result.squeeze(1)
        
        return result
    
    def _shift_channel(
        self,
        audio: torch.Tensor,
        freq_ratio: float,
        target_length: int
    ) -> torch.Tensor:
        """Pitch shift single channel using phase vocoder."""
        
        # Step 1: Time stretch by 1/freq_ratio
        stretched_audio = self._time_stretch(audio, 1.0 / freq_ratio)
        
        # Step 2: Resample to original length (changes pitch)
        resampled_audio = self._resample(stretched_audio, target_length)
        
        return resampled_audio
    
    def _time_stretch(self, audio: torch.Tensor, stretch_factor: float) -> torch.Tensor:
        """Time stretch using phase vocoder."""
        
        # STFT analysis
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=True
        )
        
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        if stretch_factor == 1.0:
            stretched_stft = stft
        else:
            # Interpolate magnitude
            stretched_magnitude = self._interpolate_time(magnitude, stretch_factor)
            
            # Adjust phase
            stretched_phase = self._adjust_phase(phase, stretch_factor)
            
            # Reconstruct complex spectrogram
            stretched_stft = stretched_magnitude * torch.exp(1j * stretched_phase)
        
        # Synthesis hop length
        syn_hop = int(self.hop_length * stretch_factor)
        
        # Inverse STFT
        audio_stretched = torch.istft(
            stretched_stft,
            n_fft=self.n_fft,
            hop_length=syn_hop,
            win_length=self.win_length,
            window=self.window,
            center=True
        )
        
        return audio_stretched
    
    def _interpolate_time(self, magnitude: torch.Tensor, stretch_factor: float) -> torch.Tensor:
        """Interpolate magnitude spectrogram in time."""
        batch_size, freq_bins, time_frames = magnitude.shape
        target_frames = int(time_frames * stretch_factor)
        
        # Interpolate using linear interpolation
        magnitude_stretched = F.interpolate(
            magnitude.unsqueeze(1),  # Add channel dim
            size=(freq_bins, target_frames),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        
        return magnitude_stretched
    
    def _adjust_phase(self, phase: torch.Tensor, stretch_factor: float) -> torch.Tensor:
        """Adjust phase for time stretching."""
        batch_size, freq_bins, time_frames = phase.shape
        target_frames = int(time_frames * stretch_factor)
        
        # Initialize output phase
        stretched_phase = torch.zeros(
            batch_size, freq_bins, target_frames,
            dtype=phase.dtype, device=phase.device
        )
        
        # Linear phase interpolation with proper unwrapping
        for b in range(batch_size):
            # Compute phase differences
            phase_diff = torch.diff(phase[b], dim=1)
            phase_diff = torch.remainder(phase_diff + np.pi, 2 * np.pi) - np.pi
            
            # Create time grid for interpolation
            time_old = torch.arange(time_frames, dtype=torch.float32, device=phase.device)
            time_new = torch.linspace(0, time_frames - 1, target_frames, device=phase.device)
            
            # Interpolate phase differences
            for f in range(freq_bins):
                # Interpolate phase differences for this frequency
                if target_frames > 1 and time_frames > 1:
                    phase_diff_interp = F.interpolate(
                        phase_diff[f].unsqueeze(0).unsqueeze(0),
                        size=target_frames - 1,
                        mode='linear',
                        align_corners=False
                    ).squeeze()
                else:
                    # Handle edge case with very short sequences
                    phase_diff_interp = phase_diff[f][:target_frames-1] if target_frames > 1 else torch.tensor([])
                
                # Reconstruct phase by cumulative sum
                stretched_phase[b, f, 0] = phase[b, f, 0]
                if target_frames > 1 and len(phase_diff_interp) > 0:
                    stretched_phase[b, f, 1:] = torch.cumsum(phase_diff_interp, dim=0) + phase[b, f, 0]
        
        return stretched_phase
    
    def _resample(self, audio: torch.Tensor, target_length: int) -> torch.Tensor:
        """Resample audio to target length."""
        current_length = audio.shape[-1]
        if current_length == target_length:
            return audio
        
        # Use linear interpolation for resampling
        resampled = F.interpolate(
            audio.unsqueeze(1),  # Add channel dim
            size=target_length,
            mode='linear',
            align_corners=False
        ).squeeze(1)
        
        return resampled


class GranularPitchShift(nn.Module):
    """
    Granular synthesis-based pitch shifting.
    
    Uses granular techniques with pitch shifting applied to each grain.
    Good for smooth pitch changes with controllable artifacts.
    
    Args:
        grain_size: Size of each grain
        overlap: Overlap between grains (0-1)
        window: Window function for grains
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        grain_size: int = 2048,
        overlap: float = 0.75,
        window: str = 'hann',
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.grain_size = grain_size
        self.overlap = overlap
        self.hop_size = int(grain_size * (1 - overlap))
        
        # Create grain window
        self.register_buffer('grain_window', self._create_window(window, grain_size))
    
    def _create_window(self, window_type: str, size: int) -> torch.Tensor:
        """Create window function."""
        if window_type == 'hann':
            return torch.hann_window(size)
        elif window_type == 'hamming':
            return torch.hamming_window(size)
        elif window_type == 'blackman':
            return torch.blackman_window(size)
        else:
            raise ValueError(f"Unknown window type: {window_type}")
    
    def forward(
        self,
        audio: torch.Tensor,
        pitch_shift: float
    ) -> torch.Tensor:
        """
        Apply granular pitch shifting.
        
        Args:
            audio: Input audio (batch, time) or (batch, channels, time)
            pitch_shift: Pitch shift in semitones
            
        Returns:
            Pitch-shifted audio
        """
        # Convert semitones to frequency ratio
        freq_ratio = 2 ** (pitch_shift / 12.0)
        
        # Handle input shape
        original_shape = audio.shape
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        
        batch_size, channels, length = audio.shape
        
        # Process each channel
        shifted_channels = []
        for c in range(channels):
            shifted = self._shift_granular(audio[:, c, :], freq_ratio)
            shifted_channels.append(shifted)
        
        result = torch.stack(shifted_channels, dim=1)
        
        # Restore original shape
        if len(original_shape) == 2:
            result = result.squeeze(1)
        
        return result
    
    def _shift_granular(
        self,
        audio: torch.Tensor,
        freq_ratio: float
    ) -> torch.Tensor:
        """Apply granular pitch shifting to single channel."""
        batch_size, length = audio.shape
        
        # Initialize output
        output = torch.zeros_like(audio)
        
        # Calculate grain hop for reading (pitch shift affects reading rate)
        read_hop = int(self.hop_size / freq_ratio)
        
        # Process each batch
        for b in range(batch_size):
            audio_b = audio[b]
            
            read_pos = 0
            write_pos = 0
            
            while (read_pos + self.grain_size <= length and 
                   write_pos + self.grain_size <= length):
                
                # Extract grain
                grain = audio_b[read_pos:read_pos + self.grain_size]
                
                # Apply window
                windowed_grain = grain * self.grain_window
                
                # Pitch shift grain using resampling
                if freq_ratio != 1.0:
                    # Resample grain to change pitch
                    grain_shifted = F.interpolate(
                        windowed_grain.unsqueeze(0).unsqueeze(0),
                        scale_factor=freq_ratio,
                        mode='linear',
                        align_corners=False
                    ).squeeze()
                    
                    # Trim or pad to original grain size
                    if len(grain_shifted) > self.grain_size:
                        grain_shifted = grain_shifted[:self.grain_size]
                    elif len(grain_shifted) < self.grain_size:
                        grain_shifted = F.pad(grain_shifted, (0, self.grain_size - len(grain_shifted)))
                else:
                    grain_shifted = windowed_grain
                
                # Add to output with overlap-add
                end_pos = min(write_pos + self.grain_size, length)
                grain_len = end_pos - write_pos
                
                output[b, write_pos:end_pos] += grain_shifted[:grain_len]
                
                # Advance positions
                read_pos += read_hop
                write_pos += self.hop_size
        
        return output


class HarmonicPitchShift(nn.Module):
    """
    Harmonic-based pitch shifting.
    
    Analyzes harmonic content and shifts each harmonic independently.
    Preserves harmonic structure better than spectral methods.
    
    Args:
        n_fft: FFT size for harmonic analysis
        hop_length: Hop length
        n_harmonics: Number of harmonics to track
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        n_fft: int = 4096,
        hop_length: Optional[int] = None,
        n_harmonics: int = 16,
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4
        self.n_harmonics = n_harmonics
        
        # Create window
        self.register_buffer('window', torch.hann_window(n_fft))
    
    def forward(
        self,
        audio: torch.Tensor,
        pitch_shift: float,
        fundamental_freq: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply harmonic pitch shifting.
        
        Args:
            audio: Input audio (batch, time)
            pitch_shift: Pitch shift in semitones
            fundamental_freq: Optional fundamental frequency (if known)
            
        Returns:
            Pitch-shifted audio
        """
        freq_ratio = 2 ** (pitch_shift / 12.0)
        
        batch_size, length = audio.shape
        
        # Analyze harmonics
        if fundamental_freq is None:
            fundamental_freq = self._estimate_fundamental(audio)
        
        # STFT analysis
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
            center=True
        )
        
        # Shift harmonics
        shifted_stft = self._shift_harmonics(stft, freq_ratio, fundamental_freq)
        
        # Inverse STFT
        audio_shifted = torch.istft(
            shifted_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            length=length
        )
        
        return audio_shifted
    
    def _estimate_fundamental(self, audio: torch.Tensor) -> torch.Tensor:
        """Estimate fundamental frequency using autocorrelation."""
        batch_size, length = audio.shape
        
        # Simple autocorrelation-based F0 estimation
        min_period = 40  # ~1000 Hz at 44.1 kHz
        max_period = 400  # ~100 Hz at 44.1 kHz
        
        fundamentals = []
        for b in range(batch_size):
            # Compute autocorrelation
            audio_padded = F.pad(audio[b], (0, max_period))
            autocorr = F.conv1d(
                audio_padded.unsqueeze(0).unsqueeze(0),
                audio[b].flip(0).unsqueeze(0).unsqueeze(0),
                padding=0
            ).squeeze()
            
            # Find peak in valid range
            valid_autocorr = autocorr[min_period:max_period + 1]
            peak_idx = torch.argmax(valid_autocorr).item()
            period = min_period + peak_idx
            
            # Convert to frequency (assuming 22050 Hz sample rate)
            fundamental = 22050.0 / period
            fundamentals.append(fundamental)
        
        return torch.tensor(fundamentals, device=audio.device)
    
    def _shift_harmonics(
        self,
        stft: torch.Tensor,
        freq_ratio: float,
        fundamental_freq: torch.Tensor
    ) -> torch.Tensor:
        """Shift harmonic components independently."""
        batch_size, freq_bins, time_frames = stft.shape
        
        # Create output spectrogram
        shifted_stft = torch.zeros_like(stft)
        
        # Frequency bin resolution
        freq_resolution = 22050.0 / self.n_fft  # Assuming 22050 Hz sample rate
        
        for b in range(batch_size):
            f0 = fundamental_freq[b].item()
            
            for h in range(1, self.n_harmonics + 1):
                # Original harmonic frequency
                harmonic_freq = f0 * h
                harmonic_bin = int(harmonic_freq / freq_resolution)
                
                # Target harmonic frequency after shifting
                target_freq = harmonic_freq * freq_ratio
                target_bin = int(target_freq / freq_resolution)
                
                # Copy harmonic content with interpolation
                if (0 <= harmonic_bin < freq_bins and 
                    0 <= target_bin < freq_bins):
                    
                    # Simple copy (could be improved with interpolation)
                    shifted_stft[b, target_bin] += stft[b, harmonic_bin]
        
        return shifted_stft


class PitchShift(nn.Module):
    """
    Unified pitch shifting module with multiple algorithms.
    
    Provides a single interface for different pitch shifting methods,
    automatically selecting the best method based on shift amount.
    
    Args:
        method: Pitch shifting method ('phase_vocoder', 'granular', 'harmonic', 'auto')
        **kwargs: Arguments passed to the specific shifting method
    """
    
    def __init__(
        self,
        method: str = 'auto',
        **kwargs
    ):
        super().__init__()
        
        self.method = method
        
        # Initialize all methods
        self.phase_vocoder = PhaseVocoderPitchShift(**kwargs)
        self.granular = GranularPitchShift(**kwargs)
        self.harmonic = HarmonicPitchShift(**kwargs)
    
    def forward(
        self,
        audio: torch.Tensor,
        pitch_shift: float,
        fundamental_freq: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply pitch shifting with automatic method selection.
        
        Args:
            audio: Input audio
            pitch_shift: Pitch shift in semitones
            fundamental_freq: Optional fundamental frequency (for harmonic method)
            
        Returns:
            Pitch-shifted audio
        """
        if self.method == 'phase_vocoder':
            return self.phase_vocoder(audio, pitch_shift)
        elif self.method == 'granular':
            return self.granular(audio, pitch_shift)
        elif self.method == 'harmonic':
            return self.harmonic(audio, pitch_shift, fundamental_freq)
        elif self.method == 'auto':
            # Automatic method selection based on shift amount
            abs_shift = abs(pitch_shift)
            if abs_shift <= 2.0:
                # Small shifts: use granular for smooth results
                return self.granular(audio, pitch_shift)
            elif abs_shift <= 12.0:
                # Medium shifts: use phase vocoder
                return self.phase_vocoder(audio, pitch_shift)
            else:
                # Large shifts: use harmonic method
                return self.harmonic(audio, pitch_shift, fundamental_freq)
        else:
            raise ValueError(f"Unknown method: {self.method}")