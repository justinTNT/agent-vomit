"""
TimeStretch module for audio time manipulation.

This module provides time-stretching capabilities for audio signals,
allowing tempo changes without pitch changes. Uses phase vocoder
and other time-domain techniques for high-quality stretching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, Tuple


class PhaseVocoder(nn.Module):
    """
    Phase vocoder for time stretching.
    
    Implements the classic phase vocoder algorithm using STFT,
    phase adjustment, and inverse STFT for time stretching.
    
    Args:
        n_fft: FFT size
        hop_length: Hop length for analysis
        win_length: Window length (default: n_fft)
        window: Window function type
        stretch_factor: Time stretch factor (>1 = slower, <1 = faster)
        phase_advance: Phase advance method ('linear', 'identity')
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: str = 'hann',
        stretch_factor: float = 1.0,
        phase_advance: str = 'linear',
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
        self.stretch_factor = stretch_factor
        self.phase_advance = phase_advance
        
        # Create window
        self.register_buffer('window', self._create_window(window))
        
        # Precompute synthesis hop length
        self.syn_hop_length = int(self.hop_length * stretch_factor)
        
        # Phase advance per frame
        if phase_advance == 'linear':
            # Linear phase advance based on frequency bins
            freq_bins = torch.arange(n_fft // 2 + 1, dtype=torch.float32)
            self.register_buffer('phase_advance_per_frame', 
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
        stretch_factor: Optional[float] = None
    ) -> torch.Tensor:
        """
        Apply time stretching to audio.
        
        Args:
            audio: Input audio (batch, time) or (batch, 1, time)
            stretch_factor: Override stretch factor for this forward pass
            
        Returns:
            Time-stretched audio
        """
        if stretch_factor is None:
            stretch_factor = self.stretch_factor
        
        # Handle input shape
        original_shape = audio.shape
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # Add channel dimension
        
        batch_size, channels, length = audio.shape
        
        # Process each channel separately
        stretched_channels = []
        for c in range(channels):
            stretched = self._stretch_channel(audio[:, c, :], stretch_factor)
            stretched_channels.append(stretched)
        
        # Stack channels
        result = torch.stack(stretched_channels, dim=1)
        
        # Remove channel dimension if input was 2D
        if len(original_shape) == 2:
            result = result.squeeze(1)
        
        return result
    
    def _stretch_channel(
        self,
        audio: torch.Tensor,
        stretch_factor: float
    ) -> torch.Tensor:
        """Stretch a single channel using phase vocoder."""
        
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
        
        # Get magnitude and phase
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Time stretch by interpolating magnitude spectrogram
        if stretch_factor != 1.0:
            # Create stretched magnitude spectrogram
            stretched_mag = self._interpolate_magnitude(magnitude, stretch_factor)
            
            # Adjust phase progression
            stretched_phase = self._adjust_phase(phase, stretch_factor)
            
            # Reconstruct complex spectrogram
            stretched_stft = stretched_mag * torch.exp(1j * stretched_phase)
        else:
            stretched_stft = stft
        
        # Inverse STFT
        output_length = int(audio.shape[-1] * stretch_factor)
        
        # Ensure we have valid data for ISTFT
        if stretched_stft.shape[-1] == 0:
            # Edge case: return silence of appropriate length
            return torch.zeros(audio.shape[0], output_length, device=audio.device, dtype=audio.dtype)
        
        audio_stretched = torch.istft(
            stretched_stft,
            n_fft=self.n_fft,
            hop_length=self.syn_hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            length=output_length
        )
        
        # Ensure output has reasonable length
        if audio_stretched.shape[-1] < output_length // 10:
            # ISTFT failed, fallback to simple interpolation
            return F.interpolate(
                audio.unsqueeze(1),
                size=output_length,
                mode='linear',
                align_corners=False
            ).squeeze(1)
        
        return audio_stretched
    
    def _interpolate_magnitude(
        self,
        magnitude: torch.Tensor,
        stretch_factor: float
    ) -> torch.Tensor:
        """Interpolate magnitude spectrogram in time."""
        batch_size, freq_bins, time_frames = magnitude.shape
        
        # Target number of frames
        target_frames = int(time_frames * stretch_factor)
        
        # Interpolate along time axis
        magnitude_stretched = F.interpolate(
            magnitude.unsqueeze(1),  # Add channel dim for interpolation
            size=(freq_bins, target_frames),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        
        return magnitude_stretched
    
    def _adjust_phase(
        self,
        phase: torch.Tensor,
        stretch_factor: float
    ) -> torch.Tensor:
        """Adjust phase for time stretching."""
        if self.phase_advance == 'identity':
            # Simple identity phase (can cause artifacts)
            return self._interpolate_magnitude(phase, stretch_factor)
        
        elif self.phase_advance == 'linear':
            # Linear phase adjustment - simplified approach
            batch_size, freq_bins, time_frames = phase.shape
            target_frames = int(time_frames * stretch_factor)
            
            if target_frames <= 0:
                # Handle edge case
                return torch.zeros(batch_size, freq_bins, 1, dtype=phase.dtype, device=phase.device)
            
            # Simple interpolation of phase
            stretched_phase = F.interpolate(
                phase.unsqueeze(1),  # Add channel dimension
                size=(freq_bins, target_frames),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
            
            return stretched_phase
        
        else:
            raise ValueError(f"Unknown phase advance method: {self.phase_advance}")


class GranularStretch(nn.Module):
    """
    Granular synthesis-based time stretching.
    
    Uses overlapping grains with crossfading for time stretching.
    Generally produces higher quality for moderate stretch factors.
    
    Args:
        grain_size: Size of each grain in samples
        overlap: Overlap between grains (0-1)
        window: Window function for grains
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        grain_size: int = 2048,
        overlap: float = 0.5,
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
        stretch_factor: float
    ) -> torch.Tensor:
        """
        Apply granular time stretching.
        
        Args:
            audio: Input audio (batch, time) or (batch, channels, time)
            stretch_factor: Time stretch factor
            
        Returns:
            Time-stretched audio
        """
        # Handle input shape
        original_shape = audio.shape
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # Add channel dimension
        
        batch_size, channels, length = audio.shape
        output_length = int(length * stretch_factor)
        
        # Process each channel
        stretched_channels = []
        for c in range(channels):
            stretched = self._stretch_granular(audio[:, c, :], stretch_factor, output_length)
            stretched_channels.append(stretched)
        
        result = torch.stack(stretched_channels, dim=1)
        
        # Restore original shape
        if len(original_shape) == 2:
            result = result.squeeze(1)
        
        return result
    
    def _stretch_granular(
        self,
        audio: torch.Tensor,
        stretch_factor: float,
        output_length: int
    ) -> torch.Tensor:
        """Apply granular stretching to single channel."""
        batch_size, input_length = audio.shape
        
        # Initialize output
        output = torch.zeros(batch_size, output_length, device=audio.device, dtype=audio.dtype)
        
        # Calculate grain positions
        input_hop = self.hop_size
        output_hop = int(self.hop_size * stretch_factor)
        
        # Process each batch item
        for b in range(batch_size):
            audio_b = audio[b]
            
            input_pos = 0
            output_pos = 0
            
            while input_pos + self.grain_size <= input_length and output_pos + self.grain_size <= output_length:
                # Extract grain
                grain = audio_b[input_pos:input_pos + self.grain_size]
                
                # Apply window
                windowed_grain = grain * self.grain_window
                
                # Add to output with overlap-add
                end_pos = min(output_pos + self.grain_size, output_length)
                grain_len = end_pos - output_pos
                
                output[b, output_pos:end_pos] += windowed_grain[:grain_len]
                
                # Advance positions
                input_pos += input_hop
                output_pos += output_hop
        
        return output


class PSOLAStretch(nn.Module):
    """
    PSOLA (Pitch Synchronous Overlap and Add) time stretching.
    
    Preserves pitch while changing tempo by using pitch-synchronous analysis.
    Requires fundamental frequency estimation.
    
    Args:
        min_pitch: Minimum expected pitch in Hz
        max_pitch: Maximum expected pitch in Hz
        frame_length: Frame length for pitch estimation
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        min_pitch: float = 80.0,
        max_pitch: float = 400.0,
        frame_length: int = 2048,
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.frame_length = frame_length
    
    def forward(
        self,
        audio: torch.Tensor,
        stretch_factor: float,
        sample_rate: float = 22050
    ) -> torch.Tensor:
        """
        Apply PSOLA time stretching.
        
        Args:
            audio: Input audio (batch, time)
            stretch_factor: Time stretch factor
            sample_rate: Audio sample rate
            
        Returns:
            Time-stretched audio
        """
        # This is a simplified PSOLA implementation
        # In practice, you'd want more sophisticated pitch tracking
        
        batch_size, length = audio.shape
        output_length = int(length * stretch_factor)
        
        # Simple implementation using autocorrelation for pitch estimation
        output = torch.zeros(batch_size, output_length, device=audio.device, dtype=audio.dtype)
        
        for b in range(batch_size):
            audio_b = audio[b]
            
            # Estimate fundamental period using autocorrelation
            period = self._estimate_period(audio_b, sample_rate)
            
            # Apply PSOLA stretching
            output[b] = self._psola_stretch(audio_b, stretch_factor, period, output_length)
        
        return output
    
    def _estimate_period(self, audio: torch.Tensor, sample_rate: float) -> int:
        """Estimate fundamental period using autocorrelation."""
        # Simple autocorrelation-based pitch estimation
        min_period = int(sample_rate / self.max_pitch)
        max_period = int(sample_rate / self.min_pitch)
        
        # Compute autocorrelation
        audio_padded = F.pad(audio, (0, max_period))
        autocorr = F.conv1d(
            audio_padded.unsqueeze(0).unsqueeze(0),
            audio.flip(0).unsqueeze(0).unsqueeze(0),
            padding=0
        ).squeeze()
        
        # Find peak in valid range
        valid_autocorr = autocorr[min_period:max_period + 1]
        peak_idx = torch.argmax(valid_autocorr).item()
        period = min_period + peak_idx
        
        return period
    
    def _psola_stretch(
        self,
        audio: torch.Tensor,
        stretch_factor: float,
        period: int,
        output_length: int
    ) -> torch.Tensor:
        """Apply PSOLA stretching algorithm."""
        length = audio.shape[0]
        output = torch.zeros(output_length, device=audio.device, dtype=audio.dtype)
        
        # Window size (typically 2-3 periods)
        window_size = period * 2
        window = torch.hann_window(window_size, device=audio.device)
        
        # Find pitch marks (simplified)
        pitch_marks = torch.arange(period, length - window_size, period, dtype=torch.long)
        
        # Synthesis
        output_period = int(period * stretch_factor)
        output_pos = 0
        
        for mark in pitch_marks:
            if output_pos + window_size >= output_length:
                break
            
            # Extract windowed segment
            start = max(0, mark - window_size // 2)
            end = min(length, start + window_size)
            segment = audio[start:end]
            
            # Apply window
            if len(segment) == window_size:
                windowed_segment = segment * window
                
                # Overlap-add to output
                output_end = min(output_pos + window_size, output_length)
                seg_len = output_end - output_pos
                output[output_pos:output_end] += windowed_segment[:seg_len]
            
            output_pos += output_period
        
        return output


class TimeStretch(nn.Module):
    """
    Unified time stretching module with multiple algorithms.
    
    Provides a single interface for different time stretching methods,
    automatically selecting the best method based on stretch factor.
    
    Args:
        method: Stretching method ('phase_vocoder', 'granular', 'psola', 'auto')
        **kwargs: Arguments passed to the specific stretching method
    """
    
    def __init__(
        self,
        method: str = 'auto',
        **kwargs
    ):
        super().__init__()
        
        self.method = method
        
        # Initialize all methods
        self.phase_vocoder = PhaseVocoder(**kwargs)
        self.granular = GranularStretch(**kwargs)
        self.psola = PSOLAStretch(**kwargs)
    
    def forward(
        self,
        audio: torch.Tensor,
        stretch_factor: float,
        sample_rate: float = 22050
    ) -> torch.Tensor:
        """
        Apply time stretching with automatic method selection.
        
        Args:
            audio: Input audio
            stretch_factor: Time stretch factor
            sample_rate: Audio sample rate (for PSOLA)
            
        Returns:
            Time-stretched audio
        """
        if self.method == 'phase_vocoder':
            return self.phase_vocoder(audio, stretch_factor)
        elif self.method == 'granular':
            return self.granular(audio, stretch_factor)
        elif self.method == 'psola':
            return self.psola(audio, stretch_factor, sample_rate)
        elif self.method == 'auto':
            # Automatic method selection based on stretch factor
            if 0.8 <= stretch_factor <= 1.25:
                # Small changes: use granular for best quality
                return self.granular(audio, stretch_factor)
            elif stretch_factor > 2.0:
                # Large stretch: use PSOLA to preserve pitch
                return self.psola(audio, stretch_factor, sample_rate)
            else:
                # Medium stretch: use phase vocoder
                return self.phase_vocoder(audio, stretch_factor)
        else:
            raise ValueError(f"Unknown method: {self.method}")