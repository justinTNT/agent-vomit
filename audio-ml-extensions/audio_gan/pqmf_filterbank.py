import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class PQMFFilterBank(nn.Module):
    """
    Pseudo Quadrature Mirror Filter Bank for multi-band signal processing.
    Splits signal into multiple frequency bands for efficient processing.
    """
    def __init__(self, 
                 num_bands: int = 4,
                 filter_length: int = 640,
                 beta: float = 9.0):
        super().__init__()
        
        self.num_bands = num_bands
        self.filter_length = filter_length
        self.beta = beta
        
        # Create prototype filter
        prototype = self._create_prototype_filter()
        
        # Create analysis and synthesis filters
        self.register_buffer('analysis_filters', self._create_filters(prototype))
        self.register_buffer('synthesis_filters', self._create_synthesis_filters())
        
        # Padding for perfect reconstruction
        self.pad_length = filter_length - num_bands
        
    def _create_prototype_filter(self) -> np.ndarray:
        """Create the prototype lowpass filter using Kaiser window."""
        # Time indices
        t = np.arange(self.filter_length, dtype=np.float32)
        t = t - (self.filter_length - 1) / 2
        
        # Normalized cutoff frequency
        cutoff = 1.0 / (4.0 * self.num_bands)
        
        # Sinc function
        sinc = np.sin(2 * np.pi * cutoff * t) / (np.pi * t + 1e-8)
        sinc[self.filter_length // 2] = 2 * cutoff  # Fix discontinuity at t=0
        
        # Kaiser window
        window = np.kaiser(self.filter_length, self.beta)
        
        # Windowed sinc
        prototype = sinc * window
        
        # Normalize for perfect reconstruction
        # Each filter should have unit energy
        prototype /= np.sqrt(np.sum(prototype ** 2))
        
        return prototype
        
    def _create_filters(self, prototype: np.ndarray) -> torch.Tensor:
        """Create analysis filter bank from prototype."""
        filters = []
        
        for k in range(self.num_bands):
            # Cosine modulation
            n = np.arange(self.filter_length, dtype=np.float32)
            modulation = 2 * np.cos(
                (2 * k + 1) * np.pi / (2 * self.num_bands) * 
                (n - (self.filter_length - 1) / 2) + 
                (-1) ** k * np.pi / 4
            )
            
            # Modulated filter
            filter_k = prototype * modulation
            filters.append(filter_k)
            
        # Stack filters [num_bands, filter_length]
        filters = np.stack(filters, axis=0)
        
        # Reshape for Conv1d [num_bands, 1, filter_length]
        filters = filters[:, np.newaxis, :]
        
        return torch.from_numpy(filters).float()
        
    def _create_synthesis_filters(self) -> torch.Tensor:
        """Create synthesis filters (time-reversed analysis filters)."""
        # For perfect reconstruction, synthesis filters are time-reversed
        # and scaled versions of analysis filters
        synthesis = self.analysis_filters.flip(dims=[2]) * self.num_bands
        return synthesis
        
    def analysis(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decompose signal into frequency bands.
        
        Args:
            x: Input signal [batch, 1, time]
            
        Returns:
            bands: Decomposed bands [batch, num_bands, time // num_bands]
        """
        batch_size = x.shape[0]
        
        # Pad signal
        x_padded = F.pad(x, (self.pad_length // 2, self.pad_length // 2), mode='reflect')
        
        # Apply analysis filters using grouped convolution
        bands = F.conv1d(
            x_padded,
            self.analysis_filters,
            stride=self.num_bands,
            groups=1
        )
        
        return bands
        
    def synthesis(self, bands: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct signal from frequency bands.
        
        Args:
            bands: Decomposed bands [batch, num_bands, time // num_bands]
            
        Returns:
            x: Reconstructed signal [batch, 1, time]
        """
        # Upsample and filter each band
        x = F.conv_transpose1d(
            bands,
            self.synthesis_filters,
            stride=self.num_bands
        )
        
        # Remove padding
        if self.pad_length > 0:
            x = x[:, :, self.pad_length//2:-self.pad_length//2]
            
        return x
        
    def forward(self, x: torch.Tensor, return_bands: bool = False) -> torch.Tensor:
        """
        Apply analysis and synthesis (perfect reconstruction).
        
        Args:
            x: Input signal [batch, 1, time]
            return_bands: If True, return decomposed bands as well
            
        Returns:
            x_reconstructed: Reconstructed signal
            bands (optional): Decomposed bands if return_bands=True
        """
        # Analysis
        bands = self.analysis(x)
        
        # Synthesis
        x_reconstructed = self.synthesis(bands)
        
        if return_bands:
            return x_reconstructed, bands
        return x_reconstructed
        
    def get_subband_shapes(self, input_length: int) -> list:
        """
        Get output shapes for each subband given input length.
        
        Args:
            input_length: Length of input signal
            
        Returns:
            List of output lengths for each band
        """
        # Account for padding and downsampling
        padded_length = input_length + self.pad_length
        subband_length = padded_length // self.num_bands
        
        return [subband_length] * self.num_bands