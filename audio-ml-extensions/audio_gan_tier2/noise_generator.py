"""
NoiseGenerator module for audio synthesis.

This module provides various noise generation strategies for neural vocoders
and audio synthesis models. Noise is essential for modeling unvoiced sounds,
breathiness, and adding stochastic components to deterministic models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple, List


class NoiseGenerator(nn.Module):
    """
    Flexible noise generation module for audio synthesis.
    
    Supports various noise types (white, pink, shaped) and can be conditioned
    on input features for adaptive noise generation.
    
    Args:
        noise_type: Type of noise ('white', 'pink', 'shaped', 'filtered')
        num_channels: Number of output channels
        learnable: Whether noise parameters are learnable
        conditional: Whether noise is conditioned on input features
        conditioning_dim: Dimension of conditioning features (if conditional)
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        noise_type: str = 'white',
        num_channels: int = 1,
        learnable: bool = True,
        conditional: bool = False,
        conditioning_dim: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.noise_type = noise_type
        self.num_channels = num_channels
        self.learnable = learnable
        self.conditional = conditional
        
        # Validate arguments
        if conditional and conditioning_dim is None:
            raise ValueError("conditioning_dim required when conditional=True")
        
        # Initialize noise generation components
        if noise_type == 'white':
            # White noise: uniform spectrum
            if learnable:
                # Learnable gain per channel
                self.gain = nn.Parameter(torch.ones(num_channels))
            else:
                self.register_buffer('gain', torch.ones(num_channels))
                
        elif noise_type == 'pink':
            # Pink noise: 1/f spectrum
            # We'll use a simple approximation with multiple white noise sources
            self.num_sources = 5  # Number of octave bands
            if learnable:
                self.gains = nn.Parameter(torch.ones(num_channels, self.num_sources))
            else:
                # Pink noise gains for -3dB/octave slope
                gains = torch.tensor([1.0, 0.5, 0.25, 0.125, 0.0625]).repeat(num_channels, 1)
                self.register_buffer('gains', gains)
                
        elif noise_type == 'shaped':
            # Shaped noise: learnable spectral envelope
            self.num_bands = 32  # Number of frequency bands
            if learnable:
                self.envelope = nn.Parameter(torch.ones(num_channels, self.num_bands))
            else:
                self.register_buffer('envelope', torch.ones(num_channels, self.num_bands))
                
        elif noise_type == 'filtered':
            # Filtered noise: learnable filters
            self.filter = nn.Conv1d(
                num_channels, num_channels,
                kernel_size=15, padding=7, groups=num_channels
            )
            if not learnable:
                # Fix filter weights
                for param in self.filter.parameters():
                    param.requires_grad = False
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        # Conditional components
        if conditional:
            self.conditioning_net = nn.Sequential(
                nn.Linear(conditioning_dim, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, num_channels * 2)  # Scale and bias
            )
    
    def forward(
        self,
        batch_size: int,
        length: int,
        conditioning: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Generate noise.
        
        Args:
            batch_size: Batch size
            length: Length of noise sequence to generate
            conditioning: Optional conditioning features (batch, conditioning_dim)
            device: Device to generate noise on
            
        Returns:
            Generated noise of shape (batch, num_channels, length)
        """
        if device is None:
            device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        
        # Generate base noise
        if self.noise_type == 'white':
            noise = self._generate_white_noise(batch_size, length, device)
            
        elif self.noise_type == 'pink':
            noise = self._generate_pink_noise(batch_size, length, device)
            
        elif self.noise_type == 'shaped':
            noise = self._generate_shaped_noise(batch_size, length, device)
            
        elif self.noise_type == 'filtered':
            noise = self._generate_filtered_noise(batch_size, length, device)
        
        # Apply conditioning if provided
        if self.conditional and conditioning is not None:
            # Generate scale and bias from conditioning
            modulation = self.conditioning_net(conditioning)  # (batch, num_channels * 2)
            scale = modulation[:, :self.num_channels].unsqueeze(-1)
            bias = modulation[:, self.num_channels:].unsqueeze(-1)
            
            # Apply conditional modulation
            noise = noise * torch.sigmoid(scale) * 2.0 + bias * 0.1
        
        return noise
    
    def _generate_white_noise(
        self,
        batch_size: int,
        length: int,
        device: torch.device
    ) -> torch.Tensor:
        """Generate white noise."""
        # Generate uniform random noise
        noise = torch.randn(batch_size, self.num_channels, length, device=device)
        
        # Apply gain
        noise = noise * self.gain.view(1, -1, 1)
        
        return noise
    
    def _generate_pink_noise(
        self,
        batch_size: int,
        length: int,
        device: torch.device
    ) -> torch.Tensor:
        """Generate pink noise using octave band method."""
        # Generate white noise sources at different rates
        noise = torch.zeros(batch_size, self.num_channels, length, device=device)
        
        for i in range(self.num_sources):
            # Downsample factor for this octave
            factor = 2 ** i
            
            # Generate noise at lower rate
            low_rate_length = (length + factor - 1) // factor
            source = torch.randn(batch_size, self.num_channels, low_rate_length, device=device)
            
            # Upsample to full rate
            if factor > 1:
                source = F.interpolate(source, size=length, mode='linear', align_corners=False)
            
            # Add weighted contribution
            noise += source * self.gains[:, i].view(1, -1, 1)
        
        # Normalize
        noise = noise / self.num_sources**0.5
        
        return noise
    
    def _generate_shaped_noise(
        self,
        batch_size: int,
        length: int,
        device: torch.device
    ) -> torch.Tensor:
        """Generate spectrally shaped noise."""
        # Generate white noise
        noise = torch.randn(batch_size, self.num_channels, length, device=device)
        
        # Apply FFT
        noise_fft = torch.fft.rfft(noise, dim=-1)
        
        # Create spectral envelope
        freqs = noise_fft.shape[-1]
        # Interpolate envelope to match FFT bins
        envelope_interp = F.interpolate(
            self.envelope.unsqueeze(0),
            size=freqs,
            mode='linear',
            align_corners=False
        ).squeeze(0)
        
        # Apply envelope
        noise_fft = noise_fft * envelope_interp.view(1, -1, freqs)
        
        # Inverse FFT
        noise = torch.fft.irfft(noise_fft, n=length, dim=-1)
        
        return noise
    
    def _generate_filtered_noise(
        self,
        batch_size: int,
        length: int,
        device: torch.device
    ) -> torch.Tensor:
        """Generate filtered noise."""
        # Generate white noise
        noise = torch.randn(batch_size, self.num_channels, length, device=device)
        
        # Apply learned filters
        noise = self.filter(noise)
        
        return noise


class BandedNoiseGenerator(nn.Module):
    """
    Banded noise generator for frequency-specific noise injection.
    
    Useful for models that process audio in frequency bands (like PQMF)
    and need band-specific noise characteristics.
    
    Args:
        num_bands: Number of frequency bands
        channels_per_band: Number of channels per band
        band_noise_types: List of noise types per band (or single type)
        learnable: Whether noise parameters are learnable
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        num_bands: int,
        channels_per_band: Union[int, List[int]] = 1,
        band_noise_types: Union[str, List[str]] = 'white',
        learnable: bool = True,
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.num_bands = num_bands
        
        # Handle channels per band
        if isinstance(channels_per_band, int):
            channels_per_band = [channels_per_band] * num_bands
        assert len(channels_per_band) == num_bands
        self.channels_per_band = channels_per_band
        
        # Handle noise types per band
        if isinstance(band_noise_types, str):
            band_noise_types = [band_noise_types] * num_bands
        assert len(band_noise_types) == num_bands
        
        # Create noise generators for each band
        self.band_generators = nn.ModuleList()
        for channels, noise_type in zip(channels_per_band, band_noise_types):
            self.band_generators.append(
                NoiseGenerator(
                    noise_type=noise_type,
                    num_channels=channels,
                    learnable=learnable,
                    conditional=False
                )
            )
    
    def forward(
        self,
        batch_size: int,
        length: int,
        band_lengths: Optional[List[int]] = None,
        device: Optional[torch.device] = None
    ) -> List[torch.Tensor]:
        """
        Generate banded noise.
        
        Args:
            batch_size: Batch size
            length: Base length (used if band_lengths not provided)
            band_lengths: Optional list of lengths per band
            device: Device to generate on
            
        Returns:
            List of noise tensors for each band
        """
        if band_lengths is None:
            band_lengths = [length] * self.num_bands
        assert len(band_lengths) == self.num_bands
        
        band_noises = []
        for generator, band_length in zip(self.band_generators, band_lengths):
            noise = generator(batch_size, band_length, device=device)
            band_noises.append(noise)
        
        return band_noises


class ResidualNoisePredictor(nn.Module):
    """
    Predicts residual noise from input features.
    
    This is useful for models that want to add learned noise components
    to deterministic predictions, improving naturalness.
    
    Args:
        input_channels: Number of input feature channels
        output_channels: Number of noise channels to predict
        hidden_channels: Hidden layer size
        kernel_size: Kernel size for conv layers
        num_layers: Number of conv layers
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        input_channels: int,
        output_channels: int = 1,
        hidden_channels: int = 128,
        kernel_size: int = 3,
        num_layers: int = 3,
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Build convolutional stack
        layers = []
        in_ch = input_channels
        
        for i in range(num_layers):
            out_ch = hidden_channels if i < num_layers - 1 else output_channels
            
            layers.extend([
                nn.Conv1d(
                    in_ch, out_ch, kernel_size,
                    padding=(kernel_size - 1) // 2
                ),
                nn.LeakyReLU(0.2) if i < num_layers - 1 else nn.Tanh()
            ])
            
            in_ch = out_ch
        
        self.predictor = nn.Sequential(*layers)
        
        # Scale factor for predicted noise
        self.noise_scale = nn.Parameter(torch.tensor(0.01))
    
    def forward(
        self,
        features: torch.Tensor,
        base_signal: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict and optionally add residual noise.
        
        Args:
            features: Input features (batch, input_channels, time)
            base_signal: Optional base signal to add noise to
            
        Returns:
            Noise (if base_signal is None) or signal + noise
        """
        # Predict noise
        noise = self.predictor(features) * self.noise_scale
        
        # If base signal provided, add noise
        if base_signal is not None:
            # Ensure shapes match
            if base_signal.shape[1] != noise.shape[1]:
                # Broadcast noise across channels if needed
                noise = noise.repeat(1, base_signal.shape[1] // noise.shape[1], 1)
            
            return base_signal + noise
        else:
            return noise