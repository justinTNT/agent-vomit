"""
OnsetDetector module for detecting transients and attacks in audio signals.

This module provides various onset detection algorithms commonly used in
music information retrieval and audio analysis. Essential for rhythm-aware
synthesis and temporal audio understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union
import math


class OnsetDetector(nn.Module):
    """
    Unified onset detector supporting multiple detection methods.
    
    Detects onset events (note attacks, transients) in audio signals using
    various established algorithms. Essential for rhythm analysis and
    beat-synchronized audio processing.
    
    Args:
        method: Detection method ('spectral_flux', 'phase_deviation', 'complex_domain', 'hfc')
        threshold_mode: Thresholding approach ('adaptive', 'fixed', 'peak_picking')
        window_size: STFT window size (default: 2048)
        hop_length: STFT hop length (default: 512)
        threshold: Fixed threshold value for 'fixed' mode (default: 0.1)
        median_filter_size: Size for adaptive threshold median filter (default: 3)
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        method: str = 'spectral_flux',
        threshold_mode: str = 'adaptive',
        window_size: int = 2048,
        hop_length: int = 512,
        threshold: float = 0.1,
        median_filter_size: int = 3,
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.method = method
        self.threshold_mode = threshold_mode
        self.window_size = window_size
        self.hop_length = hop_length
        self.threshold = threshold
        self.median_filter_size = median_filter_size
        
        # Validate parameters
        valid_methods = ['spectral_flux', 'phase_deviation', 'complex_domain', 'hfc']
        if method not in valid_methods:
            raise ValueError(f"Unknown method: {method}. Valid options: {valid_methods}")
        
        valid_thresholds = ['adaptive', 'fixed', 'peak_picking']
        if threshold_mode not in valid_thresholds:
            raise ValueError(f"Unknown threshold mode: {threshold_mode}. Valid options: {valid_thresholds}")
        
        # Create Hann window for STFT
        self.register_buffer('window', torch.hann_window(window_size))
        
        # Initialize detection-specific components
        if method == 'spectral_flux':
            self.detector = SpectralFluxDetector()
        elif method == 'phase_deviation':
            self.detector = PhaseDeviationDetector()
        elif method == 'complex_domain':
            self.detector = ComplexDomainDetector()
        elif method == 'hfc':
            self.detector = HighFrequencyContentDetector()
    
    def forward(
        self,
        x: torch.Tensor,
        return_function: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Detect onsets in audio signal.
        
        Args:
            x: Input audio (batch, time) or (batch, channels, time)
            return_function: Whether to return the onset detection function
            
        Returns:
            Onset times (batch, num_onsets) or tuple with detection function
        """
        # Handle different input shapes
        if x.dim() == 3:
            # Multi-channel: take first channel or mean
            x = x.mean(dim=1)
        elif x.dim() == 1:
            # Single audio signal
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        
        # Ensure audio is long enough for STFT
        min_length = self.window_size
        if x.shape[-1] < min_length:
            # Pad audio to minimum length
            pad_length = min_length - x.shape[-1]
            x = F.pad(x, (0, pad_length), mode='constant', value=0)
        
        # Compute STFT for each batch item
        stfts = []
        for b in range(batch_size):
            stft = torch.stft(
                x[b],
                n_fft=self.window_size,
                hop_length=self.hop_length,
                window=self.window,
                return_complex=True
            )
            stfts.append(stft)
        
        # Stack STFTs
        stft = torch.stack(stfts, dim=0)
        
        # Apply detection method
        detection_function = self.detector(stft)
        
        # Apply thresholding to find onsets
        onset_frames = self._apply_threshold(detection_function)
        
        # Convert frame indices to time
        onset_times = self._frames_to_time(onset_frames)
        
        if return_function:
            return onset_times, detection_function
        else:
            return onset_times
    
    def _apply_threshold(self, detection_function: torch.Tensor) -> List[torch.Tensor]:
        """Apply thresholding to detection function."""
        batch_size = detection_function.shape[0]
        onset_frames = []
        
        for b in range(batch_size):
            df = detection_function[b]
            
            if self.threshold_mode == 'fixed':
                # Simple fixed threshold
                peaks = (df > self.threshold).nonzero(as_tuple=True)[0]
                
            elif self.threshold_mode == 'adaptive':
                # Adaptive threshold using median filter
                if len(df) >= self.median_filter_size:
                    # Simple median approximation
                    kernel_size = min(self.median_filter_size, len(df))
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    
                    # Use conv1d for moving average approximation of median
                    df_padded = F.pad(df.unsqueeze(0).unsqueeze(0), 
                                     (kernel_size//2, kernel_size//2), 
                                     mode='reflect')
                    avg_kernel = torch.ones(1, 1, kernel_size, device=df.device) / kernel_size
                    local_avg = F.conv1d(df_padded, avg_kernel).squeeze()
                    
                    # Adaptive threshold
                    adaptive_thresh = local_avg + 0.1 * df.std()
                    peaks = ((df > adaptive_thresh) & (df > 0.01)).nonzero(as_tuple=True)[0]
                else:
                    # Fallback to fixed threshold for short signals
                    peaks = (df > self.threshold).nonzero(as_tuple=True)[0]
                    
            elif self.threshold_mode == 'peak_picking':
                # Local maxima detection
                if len(df) >= 3:
                    # Find local maxima
                    padded = F.pad(df.unsqueeze(0), (1, 1), mode='replicate').squeeze()
                    is_peak = (df > padded[:-2]) & (df > padded[2:]) & (df > self.threshold)
                    peaks = is_peak.nonzero(as_tuple=True)[0]
                else:
                    peaks = torch.tensor([], dtype=torch.long, device=df.device)
            
            # Remove consecutive detections (simple debouncing)
            if len(peaks) > 1:
                min_gap = max(1, self.hop_length // 256)  # Minimum gap between onsets
                filtered_peaks = [peaks[0]]
                for peak in peaks[1:]:
                    if peak - filtered_peaks[-1] >= min_gap:
                        filtered_peaks.append(peak)
                peaks = torch.stack(filtered_peaks)
            
            onset_frames.append(peaks)
        
        return onset_frames
    
    def _frames_to_time(self, onset_frames: List[torch.Tensor]) -> torch.Tensor:
        """Convert frame indices to time values."""
        # For now, return frame indices normalized to [0, 1]
        # In practice, would use sample_rate: frame * hop_length / sample_rate
        max_frames = max(len(frames) if len(frames) > 0 else 1 for frames in onset_frames)
        
        # Pad sequences to same length
        onset_times = []
        for frames in onset_frames:
            if len(frames) == 0:
                # No onsets detected
                times = torch.zeros(1, device=frames.device)
            else:
                times = frames.float()
                
            # Pad to max length
            if len(times) < max_frames:
                pad_size = max_frames - len(times)
                times = F.pad(times, (0, pad_size), value=-1)  # -1 indicates padding
            
            onset_times.append(times)
        
        return torch.stack(onset_times)


class SpectralFluxDetector(nn.Module):
    """Spectral flux onset detection."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, stft: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral flux detection function.
        
        Args:
            stft: Complex STFT (batch, freq, time)
            
        Returns:
            Detection function (batch, time-1)
        """
        # Compute magnitude spectrogram
        magnitude = torch.abs(stft)
        
        # Spectral flux: sum of positive differences between consecutive frames
        if magnitude.shape[-1] > 1:
            diff = magnitude[:, :, 1:] - magnitude[:, :, :-1]
            positive_diff = torch.clamp(diff, min=0)
            flux = positive_diff.sum(dim=1)  # Sum across frequency bins
        else:
            # Single frame case
            flux = torch.zeros(magnitude.shape[0], 1, device=magnitude.device)
        
        return flux


class PhaseDeviationDetector(nn.Module):
    """Phase deviation onset detection."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, stft: torch.Tensor) -> torch.Tensor:
        """
        Compute phase deviation detection function.
        
        Args:
            stft: Complex STFT (batch, freq, time)
            
        Returns:
            Detection function (batch, time-2)
        """
        if stft.shape[-1] < 3:
            # Not enough frames for phase deviation
            return torch.zeros(stft.shape[0], max(1, stft.shape[-1]-1), device=stft.device)
        
        # Extract phase
        phase = torch.angle(stft)
        
        # Compute phase differences
        phase_diff1 = phase[:, :, 1:] - phase[:, :, :-1]
        phase_diff2 = phase_diff1[:, :, 1:] - phase_diff1[:, :, :-1]
        
        # Wrap phase differences to [-π, π]
        phase_diff2 = torch.remainder(phase_diff2 + math.pi, 2 * math.pi) - math.pi
        
        # Magnitude-weighted phase deviation
        magnitude = torch.abs(stft)[:, :, 2:]  # Match time dimension
        weighted_deviation = magnitude * torch.abs(phase_diff2)
        
        # Sum across frequency bins
        deviation = weighted_deviation.sum(dim=1)
        
        return deviation


class ComplexDomainDetector(nn.Module):
    """Complex domain onset detection."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, stft: torch.Tensor) -> torch.Tensor:
        """
        Compute complex domain detection function.
        
        Args:
            stft: Complex STFT (batch, freq, time)
            
        Returns:
            Detection function (batch, time-1)
        """
        if stft.shape[-1] < 2:
            return torch.zeros(stft.shape[0], 1, device=stft.device)
        
        # Predicted next frame based on phase continuity
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Phase difference
        phase_diff = phase[:, :, 1:] - phase[:, :, :-1]
        
        # Predicted magnitude and phase for next frame
        predicted_magnitude = magnitude[:, :, :-1]  # Assume constant magnitude
        predicted_phase = phase[:, :, :-1] + phase_diff
        
        # Construct predicted complex spectrum
        predicted_stft = predicted_magnitude * torch.exp(1j * predicted_phase)
        
        # Actual next frame
        actual_stft = stft[:, :, 1:]
        
        # Complex domain difference
        diff = actual_stft - predicted_stft
        detection_function = torch.abs(diff).sum(dim=1)
        
        return detection_function


class HighFrequencyContentDetector(nn.Module):
    """High-frequency content onset detection."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, stft: torch.Tensor) -> torch.Tensor:
        """
        Compute high-frequency content detection function.
        
        Args:
            stft: Complex STFT (batch, freq, time)
            
        Returns:
            Detection function (batch, time-1)
        """
        magnitude = torch.abs(stft)
        
        # Weight by frequency bin index (higher frequencies get more weight)
        freq_weights = torch.arange(magnitude.shape[1], device=magnitude.device, dtype=magnitude.dtype)
        freq_weights = freq_weights.unsqueeze(0).unsqueeze(-1)  # (1, freq, 1)
        
        # Weighted magnitude
        weighted_magnitude = magnitude * freq_weights
        
        # HFC: sum of weighted magnitudes
        hfc = weighted_magnitude.sum(dim=1)  # Sum across frequency bins
        
        if hfc.shape[-1] > 1:
            # First difference to emphasize changes
            hfc_diff = hfc[:, 1:] - hfc[:, :-1]
            return torch.clamp(hfc_diff, min=0)  # Only positive changes
        else:
            return hfc


class AdaptiveOnsetDetector(nn.Module):
    """
    Adaptive onset detector that combines multiple methods.
    
    Uses multiple detection algorithms and combines their outputs
    for more robust onset detection.
    
    Args:
        methods: List of methods to combine (default: ['spectral_flux', 'phase_deviation'])
        combination: How to combine methods ('mean', 'max', 'weighted')
        weights: Weights for methods if combination='weighted'
        **kwargs: Arguments passed to individual detectors
    """
    
    def __init__(
        self,
        methods: List[str] = None,
        combination: str = 'mean',
        weights: Optional[List[float]] = None,
        **kwargs
    ):
        super().__init__()
        
        if methods is None:
            methods = ['spectral_flux', 'phase_deviation']
        
        self.methods = methods
        self.combination = combination
        
        # Create individual detectors
        self.detectors = nn.ModuleList()
        for method in methods:
            detector = OnsetDetector(method=method, **kwargs)
            self.detectors.append(detector)
        
        # Set up combination weights
        if combination == 'weighted':
            if weights is None:
                weights = [1.0] * len(methods)
            assert len(weights) == len(methods), "Number of weights must match number of methods"
            self.register_buffer('weights', torch.tensor(weights))
        else:
            self.weights = None
    
    def forward(
        self,
        x: torch.Tensor,
        return_individual: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Detect onsets using multiple methods.
        
        Args:
            x: Input audio (batch, time) or (batch, channels, time)
            return_individual: Whether to return individual method results
            
        Returns:
            Combined onset times, optionally with individual results
        """
        individual_results = []
        individual_functions = []
        
        for detector in self.detectors:
            onsets, detection_func = detector(x, return_function=True)
            individual_results.append(onsets)
            individual_functions.append(detection_func)
        
        # Combine detection functions
        if self.combination == 'mean':
            # Pad to same length and average
            max_len = max(df.shape[-1] for df in individual_functions)
            padded_functions = []
            for df in individual_functions:
                if df.shape[-1] < max_len:
                    pad_size = max_len - df.shape[-1]
                    df = F.pad(df, (0, pad_size), value=0)
                padded_functions.append(df)
            
            combined_function = torch.stack(padded_functions).mean(dim=0)
            
        elif self.combination == 'max':
            # Element-wise maximum
            max_len = max(df.shape[-1] for df in individual_functions)
            padded_functions = []
            for df in individual_functions:
                if df.shape[-1] < max_len:
                    pad_size = max_len - df.shape[-1]
                    df = F.pad(df, (0, pad_size), value=0)
                padded_functions.append(df)
            
            combined_function = torch.stack(padded_functions).max(dim=0)[0]
            
        elif self.combination == 'weighted':
            # Weighted combination
            max_len = max(df.shape[-1] for df in individual_functions)
            padded_functions = []
            for df in individual_functions:
                if df.shape[-1] < max_len:
                    pad_size = max_len - df.shape[-1]
                    df = F.pad(df, (0, pad_size), value=0)
                padded_functions.append(df)
            
            weighted_functions = []
            for df, weight in zip(padded_functions, self.weights):
                weighted_functions.append(df * weight)
            
            combined_function = torch.stack(weighted_functions).sum(dim=0)
        
        # Apply thresholding to combined function
        # Use the first detector's thresholding method
        onset_frames = self.detectors[0]._apply_threshold(combined_function)
        combined_onsets = self.detectors[0]._frames_to_time(onset_frames)
        
        if return_individual:
            return combined_onsets, individual_results
        else:
            return combined_onsets


class OnsetTracker(nn.Module):
    """
    Onset tracker that maintains temporal consistency.
    
    Tracks onsets over time with temporal smoothing and
    consistency constraints for stable detection.
    
    Args:
        detector: Underlying onset detector
        temporal_window: Size of temporal smoothing window
        consistency_threshold: Threshold for temporal consistency
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        detector: Optional[nn.Module] = None,
        temporal_window: int = 5,
        consistency_threshold: float = 0.3,
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        if detector is None:
            detector = OnsetDetector()
        
        self.detector = detector
        self.temporal_window = temporal_window
        self.consistency_threshold = consistency_threshold
        
        # Buffer for temporal smoothing
        self.register_buffer('history_buffer', torch.zeros(1, temporal_window))
        self.register_buffer('buffer_index', torch.tensor(0, dtype=torch.long))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Track onsets with temporal consistency.
        
        Args:
            x: Input audio (batch, time) or (batch, channels, time)
            
        Returns:
            Tracked onset times (batch, num_onsets)
        """
        # Get base detection
        onsets, detection_function = self.detector(x, return_function=True)
        
        # Apply temporal smoothing
        batch_size = detection_function.shape[0]
        
        if batch_size == 1:
            # Update history buffer
            current_mean = detection_function.mean()
            
            # Update circular buffer
            idx = self.buffer_index.item()
            self.history_buffer[0, idx] = current_mean
            self.buffer_index.data = (self.buffer_index + 1) % self.temporal_window
            
            # Temporal consistency check
            buffer_std = self.history_buffer.std()
            if buffer_std > self.consistency_threshold:
                # High variance - be more conservative
                onsets = onsets * 0.5  # Reduce sensitivity
        
        return onsets
    
    def reset(self):
        """Reset temporal history."""
        self.history_buffer.zero_()
        self.buffer_index.zero_()