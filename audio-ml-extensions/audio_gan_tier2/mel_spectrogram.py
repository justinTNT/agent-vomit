"""
MelSpectrogram module for audio feature extraction.

This module computes mel-scale spectrograms from raw audio, commonly used
as input features for audio synthesis models and as perceptual loss targets.
Follows the standard librosa/torchaudio implementation patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, Tuple


class MelSpectrogram(nn.Module):
    """
    Mel-scale spectrogram computation module.
    
    Converts raw audio waveforms to mel-scale spectrograms using STFT
    followed by mel filterbank projection. Compatible with common audio
    processing conventions.
    
    Args:
        sample_rate: Audio sample rate
        n_fft: FFT size
        win_length: Window length for STFT (default: n_fft)
        hop_length: Hop length for STFT (default: n_fft // 4)
        f_min: Minimum frequency for mel scale (default: 0.0)
        f_max: Maximum frequency for mel scale (default: sample_rate / 2)
        n_mels: Number of mel bands (default: 80)
        window: Window function ('hann', 'hamming', 'blackman')
        center: Whether to center audio before STFT
        pad_mode: Padding mode for centering
        power: Exponent for magnitude (1 for magnitude, 2 for power)
        normalized: Whether to normalize mel filters by area
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        n_mels: int = 80,
        window: str = 'hann',
        center: bool = True,
        pad_mode: str = 'reflect',
        power: float = 2.0,
        normalized: bool = False,
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate / 2.0
        self.n_mels = n_mels
        self.center = center
        self.pad_mode = pad_mode
        self.power = power
        self.normalized = normalized
        
        # Create window
        self.register_buffer('window', self._create_window(window))
        
        # Create mel filterbank
        self.register_buffer('mel_basis', self._create_mel_filterbank())
    
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
    
    def _create_mel_filterbank(self) -> torch.Tensor:
        """Create mel filterbank matrix."""
        # Following librosa mel filter bank implementation
        n_mels = self.n_mels
        n_fft = self.n_fft
        sample_rate = self.sample_rate
        f_min = self.f_min
        f_max = self.f_max
        
        # Frequency bins
        freqs = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)
        
        # Mel scale points
        mel_min = self._hz_to_mel(f_min)
        mel_max = self._hz_to_mel(f_max)
        mels = torch.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = self._mel_to_hz(mels)
        
        # Create filterbank
        filterbank = torch.zeros(n_mels, n_fft // 2 + 1)
        
        for i in range(n_mels):
            left = hz_points[i]
            center = hz_points[i + 1]
            right = hz_points[i + 2]
            
            # Rising edge
            rise = (freqs - left) / (center - left)
            # Falling edge
            fall = (right - freqs) / (right - center)
            
            # Triangular filter
            filterbank[i] = torch.maximum(
                torch.zeros_like(freqs),
                torch.minimum(rise, fall)
            )
        
        # Normalize filters
        if self.normalized:
            enorm = 2.0 / (hz_points[2:n_mels+2] - hz_points[:n_mels])
            filterbank = filterbank * enorm.unsqueeze(1)
        
        return filterbank
    
    def _hz_to_mel(self, frequencies: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Convert frequency in Hz to mel scale."""
        if isinstance(frequencies, (int, float)):
            frequencies = torch.tensor(frequencies, dtype=torch.float32)
        return 2595.0 * torch.log10(1.0 + frequencies / 700.0)
    
    def _mel_to_hz(self, mels: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Convert mel scale to frequency in Hz."""
        if isinstance(mels, (int, float)):
            mels = torch.tensor(mels, dtype=torch.float32)
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Compute mel spectrogram from audio.
        
        Args:
            audio: Input audio of shape (batch, time) or (batch, 1, time)
            
        Returns:
            Mel spectrogram of shape (batch, n_mels, time_frames)
        """
        # Handle input shape
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # (batch, 1, time)
        
        # Remove channel dimension for STFT
        batch_size, _, length = audio.shape
        audio = audio.squeeze(1)  # (batch, time)
        
        # Apply STFT
        if self.center:
            # Pad audio
            pad = self.n_fft // 2
            audio = F.pad(audio, (pad, pad), mode=self.pad_mode)
        
        # Compute STFT
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,  # Already handled padding
            return_complex=True
        )
        
        # Compute magnitude
        magnitude = torch.abs(stft)
        
        # Apply power
        if self.power != 1.0:
            magnitude = magnitude ** self.power
        
        # Apply mel filterbank
        # magnitude shape: (batch, freq_bins, time_frames)
        # mel_basis shape: (n_mels, freq_bins)
        mel_spec = torch.matmul(self.mel_basis, magnitude)
        
        return mel_spec
    
    def inverse(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Approximate inverse using Griffin-Lim algorithm.
        
        Note: This is an approximation as mel transform is lossy.
        For high-quality inverse, use a neural vocoder.
        
        Args:
            mel_spec: Mel spectrogram of shape (batch, n_mels, time_frames)
            
        Returns:
            Reconstructed audio of shape (batch, time)
        """
        # Pseudo-inverse of mel matrix
        mel_basis_pinv = torch.pinverse(self.mel_basis)
        
        # Reconstruct magnitude spectrogram
        magnitude = torch.matmul(mel_basis_pinv, mel_spec)
        
        # Undo power scaling
        if self.power != 1.0:
            magnitude = magnitude ** (1.0 / self.power)
        
        # Griffin-Lim algorithm for phase reconstruction
        # This is a simplified version - full Griffin-Lim needs iterations
        # For real applications, use a neural vocoder instead
        
        # Random phase
        phase = torch.rand_like(magnitude) * 2 * np.pi - np.pi
        
        # Reconstruct complex spectrogram
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        stft_complex = torch.complex(real, imag)
        
        # Inverse STFT
        audio = torch.istft(
            stft_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False
        )
        
        # Remove padding if centered
        if self.center:
            pad = self.n_fft // 2
            audio = audio[:, pad:-pad]
        
        return audio


class LogMelSpectrogram(MelSpectrogram):
    """
    Log-scale mel spectrogram computation.
    
    Extends MelSpectrogram to output log-scale values, which are often
    more suitable for neural network processing and perceptual loss computation.
    
    Args:
        Same as MelSpectrogram plus:
        log_offset: Small value to add before log to avoid -inf (default: 1e-6)
        **kwargs: Additional arguments passed to MelSpectrogram
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        n_mels: int = 80,
        window: str = 'hann',
        center: bool = True,
        pad_mode: str = 'reflect',
        power: float = 2.0,
        normalized: bool = False,
        log_offset: float = 1e-6,
        **kwargs
    ):
        super().__init__(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            window=window,
            center=center,
            pad_mode=pad_mode,
            power=power,
            normalized=normalized,
            **kwargs
        )
        
        self.log_offset = log_offset
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Compute log mel spectrogram from audio.
        
        Args:
            audio: Input audio of shape (batch, time) or (batch, 1, time)
            
        Returns:
            Log mel spectrogram of shape (batch, n_mels, time_frames)
        """
        # Compute mel spectrogram
        mel_spec = super().forward(audio)
        
        # Apply log scale with offset to avoid -inf
        log_mel_spec = torch.log(mel_spec + self.log_offset)
        
        return log_mel_spec
    
    def inverse(self, log_mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Approximate inverse of log mel spectrogram.
        
        Args:
            log_mel_spec: Log mel spectrogram
            
        Returns:
            Reconstructed audio
        """
        # Undo log scale
        mel_spec = torch.exp(log_mel_spec) - self.log_offset
        mel_spec = torch.clamp(mel_spec, min=0)  # Ensure non-negative
        
        # Use parent's inverse
        return super().inverse(mel_spec)


class MultiScaleMelSpectrogram(nn.Module):
    """
    Multi-scale mel spectrogram for multi-resolution processing.
    
    Computes mel spectrograms at multiple time resolutions, useful for
    hierarchical audio processing and multi-scale discriminators.
    
    Args:
        sample_rate: Audio sample rate
        n_ffts: List of FFT sizes for each scale
        hop_lengths: List of hop lengths for each scale
        n_mels: Number of mel bands (can be list for different per scale)
        **kwargs: Additional arguments passed to each MelSpectrogram
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_ffts: list = [2048, 1024, 512],
        hop_lengths: Optional[list] = None,
        n_mels: Union[int, list] = 80,
        **kwargs
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_ffts = n_ffts
        self.num_scales = len(n_ffts)
        
        # Default hop lengths
        if hop_lengths is None:
            hop_lengths = [n_fft // 4 for n_fft in n_ffts]
        assert len(hop_lengths) == len(n_ffts)
        
        # Handle n_mels
        if isinstance(n_mels, int):
            n_mels = [n_mels] * self.num_scales
        assert len(n_mels) == len(n_ffts)
        
        # Create mel spectrogram modules
        self.mel_specs = nn.ModuleList()
        for n_fft, hop_length, n_mel in zip(n_ffts, hop_lengths, n_mels):
            self.mel_specs.append(
                MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=n_mel,
                    **kwargs
                )
            )
    
    def forward(self, audio: torch.Tensor) -> list:
        """
        Compute multi-scale mel spectrograms.
        
        Args:
            audio: Input audio of shape (batch, time) or (batch, 1, time)
            
        Returns:
            List of mel spectrograms at different scales
        """
        mel_specs = []
        for mel_spec_module in self.mel_specs:
            mel_specs.append(mel_spec_module(audio))
        
        return mel_specs