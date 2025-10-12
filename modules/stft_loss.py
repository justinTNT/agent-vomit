import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
import numpy as np


class SpectralConvergenceLoss(nn.Module):
    """Spectral convergence loss for comparing spectrograms."""
    
    def forward(self, x_mag: torch.Tensor, y_mag: torch.Tensor) -> torch.Tensor:
        """
        Calculate spectral convergence loss.
        
        Args:
            x_mag: Magnitude spectrogram of prediction
            y_mag: Magnitude spectrogram of target
            
        Returns:
            Spectral convergence loss value
        """
        return torch.norm(y_mag - x_mag, p='fro') / torch.norm(y_mag, p='fro')


class LogSTFTMagnitudeLoss(nn.Module):
    """Log STFT magnitude loss (L1 distance in log domain)."""
    
    def forward(self, x_mag: torch.Tensor, y_mag: torch.Tensor) -> torch.Tensor:
        """
        Calculate log STFT magnitude loss.
        
        Args:
            x_mag: Magnitude spectrogram of prediction
            y_mag: Magnitude spectrogram of target
            
        Returns:
            Log magnitude loss value
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-7
        return F.l1_loss(torch.log(x_mag + eps), torch.log(y_mag + eps))


class STFTLoss(nn.Module):
    """
    Single-scale STFT loss combining spectral convergence and log magnitude loss.
    
    Args:
        fft_size: FFT size
        hop_size: Hop size  
        win_length: Window length (defaults to fft_size)
        window: Window type ('hann', 'hamming', etc.)
        normalized: Whether to normalize STFT
        eps: Small epsilon for numerical stability
    """
    
    def __init__(
        self,
        fft_size: int = 1024,
        hop_size: int = 256,
        win_length: Optional[int] = None,
        window: str = 'hann',
        normalized: bool = False,
        eps: float = 1e-7
    ):
        super().__init__()
        
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length or fft_size
        self.normalized = normalized
        self.eps = eps
        
        # Register window as buffer
        self.register_buffer('window', self._get_window(window))
        
        # Loss components
        self.spectral_convergence = SpectralConvergenceLoss()
        self.log_magnitude = LogSTFTMagnitudeLoss()
    
    def _get_window(self, window_type: str) -> torch.Tensor:
        """Get window function."""
        if window_type == 'hann':
            return torch.hann_window(self.win_length)
        elif window_type == 'hamming':
            return torch.hamming_window(self.win_length)
        elif window_type == 'blackman':
            return torch.blackman_window(self.win_length)
        elif window_type == 'bartlett':
            return torch.bartlett_window(self.win_length)
        else:
            raise ValueError(f"Unknown window type: {window_type}")
    
    def forward(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor,
        return_stfts: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Calculate STFT loss.
        
        Args:
            x: Predicted waveform [batch, time] or [batch, channels, time]
            y: Target waveform [batch, time] or [batch, channels, time]
            return_stfts: If True, also return computed STFTs
            
        Returns:
            Combined loss value, and optionally (loss, x_stft, y_stft)
        """
        # Handle multi-channel by flattening
        if x.dim() == 3:
            batch, channels, time = x.shape
            x = x.view(batch * channels, time)
            y = y.view(batch * channels, time)
        
        # Compute STFT
        x_stft = torch.stft(
            x,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=self.window,
            normalized=self.normalized,
            return_complex=True
        )
        
        y_stft = torch.stft(
            y,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=self.window,
            normalized=self.normalized,
            return_complex=True
        )
        
        # Get magnitude
        x_mag = torch.abs(x_stft)
        y_mag = torch.abs(y_stft)
        
        # Calculate losses
        sc_loss = self.spectral_convergence(x_mag, y_mag)
        mag_loss = self.log_magnitude(x_mag, y_mag)
        
        # Combine losses
        loss = sc_loss + mag_loss
        
        if return_stfts:
            return loss, x_stft, y_stft
        else:
            return loss


class MultiScaleSTFTLoss(nn.Module):
    """
    Multi-scale STFT loss for high-quality audio synthesis.
    
    Computes STFT loss at multiple time-frequency resolutions to capture
    both temporal and spectral characteristics at different scales.
    
    Args:
        scales: List of (fft_size, hop_size, win_length) tuples
        window: Window type for all scales
        normalized: Whether to normalize STFT
        loss_weights: Optional weights for each scale
        eps: Small epsilon for numerical stability
    """
    
    def __init__(
        self,
        scales: Optional[List[Tuple[int, int, int]]] = None,
        window: str = 'hann',
        normalized: bool = False,
        loss_weights: Optional[List[float]] = None,
        eps: float = 1e-7
    ):
        super().__init__()
        
        # Default scales if not provided
        if scales is None:
            scales = [
                (512, 128, 512),    # High time resolution
                (1024, 256, 1024),  # Balanced
                (2048, 512, 2048),  # High frequency resolution
            ]
        
        self.scales = scales
        self.loss_weights = loss_weights or [1.0] * len(scales)
        
        # Create STFT loss for each scale
        self.stft_losses = nn.ModuleList([
            STFTLoss(
                fft_size=fft_size,
                hop_size=hop_size,
                win_length=win_length,
                window=window,
                normalized=normalized,
                eps=eps
            )
            for fft_size, hop_size, win_length in scales
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        return_losses_per_scale: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Calculate multi-scale STFT loss.
        
        Args:
            x: Predicted waveform [batch, time] or [batch, channels, time]
            y: Target waveform [batch, time] or [batch, channels, time]
            return_losses_per_scale: If True, return individual scale losses
            
        Returns:
            Combined loss, and optionally list of per-scale losses
        """
        losses = []
        
        for i, stft_loss in enumerate(self.stft_losses):
            loss = stft_loss(x, y)
            losses.append(loss * self.loss_weights[i])
        
        total_loss = sum(losses) / len(losses)
        
        if return_losses_per_scale:
            return total_loss, losses
        else:
            return total_loss


class MelSpectrogramLoss(nn.Module):
    """
    Mel-spectrogram loss for perceptually-weighted audio comparison.
    
    Args:
        sample_rate: Audio sample rate
        n_fft: FFT size
        hop_length: Hop size
        n_mels: Number of mel bands
        f_min: Minimum frequency
        f_max: Maximum frequency
        normalized: Whether to normalize
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        normalized: bool = False
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate / 2
        self.normalized = normalized
        
        # Create mel filterbank
        mel_fb = self._create_mel_filterbank()
        self.register_buffer('mel_filterbank', mel_fb)
        
        # Window
        self.register_buffer('window', torch.hann_window(n_fft))
    
    def _create_mel_filterbank(self) -> torch.Tensor:
        """Create mel filterbank matrix."""
        # This is a simplified version - in practice you might use torchaudio
        n_freqs = self.n_fft // 2 + 1
        
        # Convert frequencies to mel scale
        mel_min = 2595 * np.log10(1 + self.f_min / 700)
        mel_max = 2595 * np.log10(1 + self.f_max / 700)
        
        # Create mel points
        mel_points = torch.linspace(mel_min, mel_max, self.n_mels + 2)
        freq_points = 700 * (10 ** (mel_points / 2595) - 1)
        
        # Convert to FFT bin numbers
        bins = torch.floor((self.n_fft + 1) * freq_points / self.sample_rate).long()
        
        # Create filterbank
        fb = torch.zeros(self.n_mels, n_freqs)
        
        for i in range(self.n_mels):
            low = bins[i].item()
            center = bins[i + 1].item()
            high = bins[i + 2].item()
            
            # Rising edge
            fb[i, low:center] = torch.linspace(0, 1, center - low)
            # Falling edge
            fb[i, center:high] = torch.linspace(1, 0, high - center)
        
        return fb
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculate mel-spectrogram loss.
        
        Args:
            x: Predicted waveform
            y: Target waveform
            
        Returns:
            Mel-spectrogram L1 loss
        """
        # Compute STFT
        x_stft = torch.stft(
            x.squeeze(1) if x.dim() == 3 else x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True
        )
        
        y_stft = torch.stft(
            y.squeeze(1) if y.dim() == 3 else y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True
        )
        
        # Get magnitude
        x_mag = torch.abs(x_stft)
        y_mag = torch.abs(y_stft)
        
        # Apply mel filterbank
        x_mel = torch.matmul(self.mel_filterbank, x_mag)
        y_mel = torch.matmul(self.mel_filterbank, y_mag)
        
        # Convert to log scale
        x_log_mel = torch.log(x_mel + 1e-5)
        y_log_mel = torch.log(y_mel + 1e-5)
        
        return F.l1_loss(x_log_mel, y_log_mel)