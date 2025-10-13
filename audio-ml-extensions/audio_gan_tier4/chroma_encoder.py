"""
ChromaEncoder module for extracting chroma features from audio signals.

This module provides various methods for computing chroma (pitch class) profiles
from audio, essential for music analysis and harmonic content understanding.
Supports different tuning systems and temporal aggregation strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union
import math


class ChromaEncoder(nn.Module):
    """
    Unified chroma encoder supporting multiple extraction methods.
    
    Extracts chroma (pitch class) features from audio signals using various
    established algorithms. Essential for music analysis, chord recognition,
    and harmonic content understanding.
    
    Args:
        method: Extraction method ('stft', 'cqt', 'harmonic', 'neural')
        num_chroma: Number of chroma bins (default: 12 for equal temperament)
        window_size: STFT window size (default: 4096)
        hop_length: STFT hop length (default: 1024)
        tuning_freq: Reference tuning frequency in Hz (default: 440.0)
        octave_range: Range of octaves to consider (default: (0, 8))
        temporal_pooling: How to pool across time ('mean', 'max', 'none')
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        method: str = 'stft',
        num_chroma: int = 12,
        window_size: int = 4096,
        hop_length: int = 1024,
        tuning_freq: float = 440.0,
        octave_range: Tuple[int, int] = (0, 8),
        temporal_pooling: str = 'mean',
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.method = method
        self.num_chroma = num_chroma
        self.window_size = window_size
        self.hop_length = hop_length
        self.tuning_freq = tuning_freq
        self.octave_range = octave_range
        self.temporal_pooling = temporal_pooling
        
        # Validate parameters
        valid_methods = ['stft', 'cqt', 'harmonic', 'neural']
        if method not in valid_methods:
            raise ValueError(f"Unknown method: {method}. Valid options: {valid_methods}")
        
        valid_pooling = ['mean', 'max', 'none']
        if temporal_pooling not in valid_pooling:
            raise ValueError(f"Unknown temporal pooling: {temporal_pooling}. Valid options: {valid_pooling}")
        
        # Create Hann window for STFT
        self.register_buffer('window', torch.hann_window(window_size))
        
        # Initialize method-specific components
        if method == 'stft':
            self.encoder = STFTChromaEncoder(
                num_chroma, window_size, hop_length, tuning_freq, octave_range
            )
        elif method == 'cqt':
            self.encoder = CQTChromaEncoder(
                num_chroma, hop_length, tuning_freq, octave_range
            )
        elif method == 'harmonic':
            self.encoder = HarmonicChromaEncoder(
                num_chroma, window_size, hop_length, tuning_freq, octave_range
            )
        elif method == 'neural':
            self.encoder = NeuralChromaEncoder(
                num_chroma, window_size, hop_length
            )
        
        # Chroma filter bank for mapping frequencies to pitch classes
        self._build_chroma_filter()
    
    def _build_chroma_filter(self):
        """Build filter bank for mapping frequencies to chroma bins."""
        # Frequency bins for STFT
        freq_bins = self.window_size // 2 + 1
        freqs = torch.fft.fftfreq(self.window_size, 1.0)[:freq_bins] * 22050  # Assume 22kHz Nyquist
        
        # Build chroma mapping matrix
        chroma_filter = torch.zeros(self.num_chroma, freq_bins)
        
        # Only process positive frequencies
        for i, freq in enumerate(freqs):
            if freq > 0:
                # Convert frequency to MIDI note
                midi_note = 69 + 12 * math.log2(freq / self.tuning_freq)
                
                # Map to chroma bin (modulo 12)
                chroma_bin = int(midi_note) % self.num_chroma
                
                # Gaussian weighting around the target bin
                for c in range(self.num_chroma):
                    dist = min(abs(c - chroma_bin), 12 - abs(c - chroma_bin))  # Circular distance
                    weight = math.exp(-0.5 * (dist / 0.5) ** 2)  # Gaussian with σ=0.5
                    chroma_filter[c, i] = weight
        
        # Normalize each chroma bin
        chroma_filter = F.normalize(chroma_filter, p=1, dim=1)
        self.register_buffer('chroma_filter', chroma_filter)
    
    def forward(
        self,
        x: torch.Tensor,
        return_temporal: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract chroma features from audio.
        
        Args:
            x: Input audio (batch, time) or (batch, channels, time)
            return_temporal: Whether to return temporal chroma sequence
            
        Returns:
            Chroma features (batch, num_chroma) or with temporal sequence
        """
        # Handle different input shapes
        if x.dim() == 3:
            # Multi-channel: take mean or first channel
            x = x.mean(dim=1)
        elif x.dim() == 1:
            # Single audio signal
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        
        # Apply method-specific encoding
        chroma_temporal = self.encoder(x)
        
        # Apply temporal pooling
        if self.temporal_pooling == 'mean':
            chroma = chroma_temporal.mean(dim=-1)
        elif self.temporal_pooling == 'max':
            chroma = chroma_temporal.max(dim=-1)[0]
        elif self.temporal_pooling == 'none':
            chroma = chroma_temporal
        else:
            chroma = chroma_temporal.mean(dim=-1)  # Fallback
        
        # Normalize chroma features
        chroma = F.normalize(chroma, p=1, dim=-1)
        
        if return_temporal and self.temporal_pooling != 'none':
            return chroma, chroma_temporal
        else:
            return chroma


class STFTChromaEncoder(nn.Module):
    """STFT-based chroma extraction."""
    
    def __init__(self, num_chroma, window_size, hop_length, tuning_freq, octave_range):
        super().__init__()
        self.num_chroma = num_chroma
        self.window_size = window_size
        self.hop_length = hop_length
        self.tuning_freq = tuning_freq
        self.octave_range = octave_range
        
        # Register window
        self.register_buffer('window', torch.hann_window(window_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract chroma using STFT.
        
        Args:
            x: Input audio (batch, time)
            
        Returns:
            Chroma features (batch, num_chroma, time_frames)
        """
        batch_size = x.shape[0]
        
        # Ensure audio is long enough
        if x.shape[-1] < self.window_size:
            pad_length = self.window_size - x.shape[-1]
            x = F.pad(x, (0, pad_length), mode='constant', value=0)
        
        # Compute STFT for each batch item
        chroma_features = []
        
        for b in range(batch_size):
            stft = torch.stft(
                x[b],
                n_fft=self.window_size,
                hop_length=self.hop_length,
                window=self.window,
                return_complex=True
            )
            
            # Get magnitude spectrogram
            magnitude = torch.abs(stft)
            
            # Apply chroma mapping
            chroma = self._magnitude_to_chroma(magnitude)
            chroma_features.append(chroma)
        
        return torch.stack(chroma_features, dim=0)
    
    def _magnitude_to_chroma(self, magnitude: torch.Tensor) -> torch.Tensor:
        """Convert magnitude spectrogram to chroma."""
        # Simple frequency-to-chroma mapping
        freq_bins, time_frames = magnitude.shape
        chroma = torch.zeros(self.num_chroma, time_frames, device=magnitude.device)
        
        # Map each frequency bin to chroma
        for f in range(freq_bins):
            # Convert bin to frequency (approximate)
            freq = f * 22050 / (self.window_size // 2)
            
            if freq > 80:  # Only consider frequencies above 80 Hz
                # Convert to MIDI note
                try:
                    midi_note = 69 + 12 * math.log2(freq / self.tuning_freq)
                    chroma_bin = int(midi_note) % self.num_chroma
                    
                    # Add magnitude to appropriate chroma bin
                    chroma[chroma_bin] += magnitude[f]
                except (ValueError, OverflowError):
                    # Skip problematic frequencies
                    continue
        
        return chroma


class CQTChromaEncoder(nn.Module):
    """Constant-Q Transform based chroma extraction."""
    
    def __init__(self, num_chroma, hop_length, tuning_freq, octave_range):
        super().__init__()
        self.num_chroma = num_chroma
        self.hop_length = hop_length
        self.tuning_freq = tuning_freq
        self.octave_range = octave_range
        
        # Build simplified CQT filter bank
        self._build_cqt_filters()
    
    def _build_cqt_filters(self):
        """Build simplified CQT filter bank."""
        # For simplicity, use a fixed filter bank
        # In practice, would implement proper CQT
        
        # Number of bins per octave
        bins_per_octave = self.num_chroma
        num_octaves = self.octave_range[1] - self.octave_range[0]
        total_bins = bins_per_octave * num_octaves
        
        # Create filter bank (simplified)
        filter_bank = torch.randn(total_bins, 1024)  # Fixed size for simplicity
        self.register_buffer('cqt_filters', filter_bank)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract chroma using CQT.
        
        Args:
            x: Input audio (batch, time)
            
        Returns:
            Chroma features (batch, num_chroma, time_frames)
        """
        batch_size = x.shape[0]
        
        # Simplified CQT implementation
        # In practice, would use proper CQT algorithm
        
        # Compute approximate time frames
        time_frames = max(1, x.shape[-1] // self.hop_length)
        chroma_features = []
        
        for b in range(batch_size):
            # Simple windowed analysis
            chroma_temporal = []
            
            for t in range(time_frames):
                start_idx = t * self.hop_length
                end_idx = min(start_idx + 1024, x.shape[-1])
                
                if end_idx - start_idx > 0:
                    # Extract window
                    window = x[b, start_idx:end_idx]
                    
                    # Pad if necessary
                    if len(window) < 1024:
                        window = F.pad(window, (0, 1024 - len(window)))
                    
                    # Apply filters (simplified)
                    cqt_response = torch.abs(torch.matmul(self.cqt_filters, window))
                    
                    # Fold to chroma
                    chroma_frame = torch.zeros(self.num_chroma, device=x.device)
                    for i, response in enumerate(cqt_response):
                        chroma_bin = i % self.num_chroma
                        chroma_frame[chroma_bin] += response
                    
                    chroma_temporal.append(chroma_frame)
            
            if chroma_temporal:
                chroma = torch.stack(chroma_temporal, dim=1)
            else:
                chroma = torch.zeros(self.num_chroma, 1, device=x.device)
            
            chroma_features.append(chroma)
        
        return torch.stack(chroma_features, dim=0)


class HarmonicChromaEncoder(nn.Module):
    """Harmonic-aware chroma extraction."""
    
    def __init__(self, num_chroma, window_size, hop_length, tuning_freq, octave_range):
        super().__init__()
        self.num_chroma = num_chroma
        self.window_size = window_size
        self.hop_length = hop_length
        self.tuning_freq = tuning_freq
        self.octave_range = octave_range
        
        self.register_buffer('window', torch.hann_window(window_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract chroma using harmonic analysis.
        
        Args:
            x: Input audio (batch, time)
            
        Returns:
            Chroma features (batch, num_chroma, time_frames)
        """
        batch_size = x.shape[0]
        
        # Ensure audio is long enough
        if x.shape[-1] < self.window_size:
            pad_length = self.window_size - x.shape[-1]
            x = F.pad(x, (0, pad_length), mode='constant', value=0)
        
        chroma_features = []
        
        for b in range(batch_size):
            stft = torch.stft(
                x[b],
                n_fft=self.window_size,
                hop_length=self.hop_length,
                window=self.window,
                return_complex=True
            )
            
            magnitude = torch.abs(stft)
            
            # Harmonic processing: enhance harmonic series
            enhanced_magnitude = self._enhance_harmonics(magnitude)
            
            # Convert to chroma
            chroma = self._magnitude_to_chroma(enhanced_magnitude)
            chroma_features.append(chroma)
        
        return torch.stack(chroma_features, dim=0)
    
    def _enhance_harmonics(self, magnitude: torch.Tensor) -> torch.Tensor:
        """Enhance harmonic components."""
        # Simple harmonic enhancement: boost frequencies that are multiples
        enhanced = magnitude.clone()
        
        freq_bins, time_frames = magnitude.shape
        
        # For each potential fundamental frequency
        for f0_bin in range(1, freq_bins // 4):  # Only consider lower frequencies as fundamentals
            # Look for harmonics
            harmonic_strength = 0
            harmonic_count = 0
            
            for harmonic in range(2, 6):  # Check first few harmonics
                harmonic_bin = f0_bin * harmonic
                if harmonic_bin < freq_bins:
                    harmonic_strength += magnitude[harmonic_bin].mean()
                    harmonic_count += 1
            
            if harmonic_count > 0:
                avg_harmonic_strength = harmonic_strength / harmonic_count
                
                # Boost fundamental if harmonics are strong
                if avg_harmonic_strength > magnitude[f0_bin].mean():
                    enhanced[f0_bin] *= 1.5
        
        return enhanced
    
    def _magnitude_to_chroma(self, magnitude: torch.Tensor) -> torch.Tensor:
        """Convert magnitude spectrogram to chroma with harmonic awareness."""
        freq_bins, time_frames = magnitude.shape
        chroma = torch.zeros(self.num_chroma, time_frames, device=magnitude.device)
        
        for f in range(freq_bins):
            freq = f * 22050 / (self.window_size // 2)
            
            if freq > 80:  # Only consider frequencies above 80 Hz
                try:
                    midi_note = 69 + 12 * math.log2(freq / self.tuning_freq)
                    chroma_bin = int(midi_note) % self.num_chroma
                    
                    # Weight by harmonic likelihood
                    harmonic_weight = 1.0
                    
                    # Boost octave frequencies
                    if f % (self.window_size // 12) == 0:  # Rough octave check
                        harmonic_weight = 1.2
                    
                    chroma[chroma_bin] += magnitude[f] * harmonic_weight
                except (ValueError, OverflowError):
                    continue
        
        return chroma


class NeuralChromaEncoder(nn.Module):
    """Neural network-based chroma extraction."""
    
    def __init__(self, num_chroma, window_size, hop_length):
        super().__init__()
        self.num_chroma = num_chroma
        self.window_size = window_size
        self.hop_length = hop_length
        
        # Neural network for chroma extraction
        freq_bins = window_size // 2 + 1
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(freq_bins, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_chroma),
            nn.Softmax(dim=-1)
        )
        
        self.register_buffer('window', torch.hann_window(window_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract chroma using neural network.
        
        Args:
            x: Input audio (batch, time)
            
        Returns:
            Chroma features (batch, num_chroma, time_frames)
        """
        batch_size = x.shape[0]
        
        # Ensure audio is long enough
        if x.shape[-1] < self.window_size:
            pad_length = self.window_size - x.shape[-1]
            x = F.pad(x, (0, pad_length), mode='constant', value=0)
        
        chroma_features = []
        
        for b in range(batch_size):
            stft = torch.stft(
                x[b],
                n_fft=self.window_size,
                hop_length=self.hop_length,
                window=self.window,
                return_complex=True
            )
            
            magnitude = torch.abs(stft)  # (freq_bins, time_frames)
            
            # Apply neural network to each time frame
            time_frames = magnitude.shape[1]
            chroma_temporal = []
            
            for t in range(time_frames):
                frame_magnitude = magnitude[:, t]
                chroma_frame = self.feature_extractor(frame_magnitude)
                chroma_temporal.append(chroma_frame)
            
            chroma = torch.stack(chroma_temporal, dim=1)  # (num_chroma, time_frames)
            chroma_features.append(chroma)
        
        return torch.stack(chroma_features, dim=0)


class ChromaProcessor(nn.Module):
    """
    Advanced chroma processing with smoothing and enhancement.
    
    Post-processes chroma features for better temporal consistency
    and musical relevance.
    
    Args:
        num_chroma: Number of chroma bins (default: 12)
        smoothing_kernel_size: Size of temporal smoothing kernel (default: 3)
        enhancement_method: Enhancement method ('none', 'harmonic', 'template')
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        num_chroma: int = 12,
        smoothing_kernel_size: int = 3,
        enhancement_method: str = 'harmonic',
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.num_chroma = num_chroma
        self.smoothing_kernel_size = smoothing_kernel_size
        self.enhancement_method = enhancement_method
        
        # Smoothing kernel
        if smoothing_kernel_size > 1:
            kernel = torch.ones(1, 1, smoothing_kernel_size) / smoothing_kernel_size
            self.register_buffer('smoothing_kernel', kernel)
        
        # Chord templates for enhancement
        if enhancement_method == 'template':
            self._build_chord_templates()
    
    def _build_chord_templates(self):
        """Build basic chord templates."""
        # Major and minor triads for each root note
        templates = torch.zeros(24, self.num_chroma)  # 12 major + 12 minor
        
        for root in range(self.num_chroma):
            # Major triad: root, major third, fifth
            major_idx = root * 2
            templates[major_idx, root] = 1.0
            templates[major_idx, (root + 4) % self.num_chroma] = 0.8
            templates[major_idx, (root + 7) % self.num_chroma] = 0.6
            
            # Minor triad: root, minor third, fifth
            minor_idx = root * 2 + 1
            templates[minor_idx, root] = 1.0
            templates[minor_idx, (root + 3) % self.num_chroma] = 0.8
            templates[minor_idx, (root + 7) % self.num_chroma] = 0.6
        
        self.register_buffer('chord_templates', templates)
    
    def forward(self, chroma: torch.Tensor) -> torch.Tensor:
        """
        Process chroma features.
        
        Args:
            chroma: Input chroma features (batch, num_chroma, time_frames)
            
        Returns:
            Processed chroma features (batch, num_chroma, time_frames)
        """
        processed = chroma
        
        # Temporal smoothing
        if hasattr(self, 'smoothing_kernel') and processed.shape[-1] > 1:
            # Apply smoothing along time dimension for each chroma bin separately
            batch_size, num_chroma, time_frames = processed.shape
            
            # Reshape to (batch * chroma, 1, time)
            reshaped = processed.view(batch_size * num_chroma, 1, time_frames)
            
            # Apply smoothing
            smoothed = F.conv1d(
                reshaped,
                self.smoothing_kernel,
                padding=self.smoothing_kernel_size // 2
            )
            
            # Reshape back to (batch, chroma, time)
            processed = smoothed.view(batch_size, num_chroma, time_frames)
        
        # Enhancement
        if self.enhancement_method == 'harmonic':
            processed = self._enhance_harmonics(processed)
        elif self.enhancement_method == 'template':
            processed = self._template_enhancement(processed)
        
        # Normalize
        processed = F.normalize(processed, p=1, dim=1)
        
        return processed
    
    def _enhance_harmonics(self, chroma: torch.Tensor) -> torch.Tensor:
        """Enhance harmonic relationships in chroma."""
        enhanced = chroma.clone()
        
        # Boost perfect fifths and octaves
        for root in range(self.num_chroma):
            fifth = (root + 7) % self.num_chroma
            
            # If root is strong, boost fifth
            root_strength = chroma[:, root]
            enhanced[:, fifth] += 0.3 * root_strength
        
        return enhanced
    
    def _template_enhancement(self, chroma: torch.Tensor) -> torch.Tensor:
        """Enhance using chord templates."""
        batch_size, num_chroma, time_frames = chroma.shape
        enhanced = chroma.clone()
        
        # For each time frame, find best matching template
        for t in range(time_frames):
            frame = chroma[:, :, t]  # (batch, num_chroma)
            
            # Compute correlations with templates
            correlations = torch.matmul(frame, self.chord_templates.T)  # (batch, 24)
            
            # Find best template for each batch item
            best_templates = torch.argmax(correlations, dim=1)  # (batch,)
            
            # Apply template enhancement
            for b in range(batch_size):
                template_idx = best_templates[b].item()
                template = self.chord_templates[template_idx]
                
                # Blend original with template
                enhanced[b, :, t] = 0.7 * enhanced[b, :, t] + 0.3 * template
        
        return enhanced


class ChromaShiftAugmentation(nn.Module):
    """
    Chroma augmentation through pitch shifting.
    
    Applies random pitch shifts to chroma features for data augmentation
    or key transposition.
    
    Args:
        max_shift: Maximum shift in semitones (default: 6)
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(self, max_shift: int = 6, **kwargs):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.max_shift = max_shift
    
    def forward(self, chroma: torch.Tensor, shift: Optional[int] = None) -> torch.Tensor:
        """
        Apply pitch shift to chroma features.
        
        Args:
            chroma: Input chroma features (batch, num_chroma, time_frames)
            shift: Shift amount in semitones (random if None)
            
        Returns:
            Shifted chroma features
        """
        if shift is None:
            shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
        
        if shift == 0:
            return chroma
        
        # Circular shift along chroma dimension
        shifted = torch.roll(chroma, shifts=shift, dims=1)
        
        return shifted