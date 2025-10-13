"""
PhaseReconstruction module for reconstructing phase information from magnitude spectrograms.

This module provides various algorithms for phase reconstruction, essential for
converting magnitude-only representations back to time-domain audio signals.
Critical for vocoder applications and magnitude-based audio processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union
import math


class PhaseReconstruction(nn.Module):
    """
    Unified phase reconstruction supporting multiple algorithms.
    
    Reconstructs phase information from magnitude spectrograms using various
    established algorithms. Essential for converting magnitude representations
    back to high-quality time-domain audio.
    
    Args:
        method: Reconstruction method ('griffin_lim', 'neural', 'iterative', 'heuristic')
        n_fft: FFT size (default: 2048)
        hop_length: Hop length (default: 512)
        num_iterations: Number of iterations for iterative methods (default: 30)
        momentum: Momentum for Griffin-Lim (default: 0.99)
        init_phase: Phase initialization method ('random', 'zero', 'previous')
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        method: str = 'griffin_lim',
        n_fft: int = 2048,
        hop_length: int = 512,
        num_iterations: int = 30,
        momentum: float = 0.99,
        init_phase: str = 'random',
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.method = method
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_iterations = num_iterations
        self.momentum = momentum
        self.init_phase = init_phase
        
        # Validate parameters
        valid_methods = ['griffin_lim', 'neural', 'iterative', 'heuristic']
        if method not in valid_methods:
            raise ValueError(f"Unknown method: {method}. Valid options: {valid_methods}")
        
        valid_init = ['random', 'zero', 'previous']
        if init_phase not in valid_init:
            raise ValueError(f"Unknown init_phase: {init_phase}. Valid options: {valid_init}")
        
        # Create window for STFT
        self.register_buffer('window', torch.hann_window(n_fft))
        
        # Initialize method-specific components
        if method == 'griffin_lim':
            self.reconstructor = GriffinLimReconstructor(
                n_fft, hop_length, num_iterations, momentum
            )
        elif method == 'neural':
            self.reconstructor = NeuralPhaseReconstructor(n_fft, hop_length)
        elif method == 'iterative':
            self.reconstructor = IterativePhaseReconstructor(
                n_fft, hop_length, num_iterations
            )
        elif method == 'heuristic':
            self.reconstructor = HeuristicPhaseReconstructor(n_fft, hop_length)
        
        # Buffer for previous phase (for init_phase='previous')
        self.register_buffer('prev_phase', None)
    
    def forward(
        self,
        magnitude: torch.Tensor,
        target_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Reconstruct audio from magnitude spectrogram.
        
        Args:
            magnitude: Magnitude spectrogram (batch, freq_bins, time_frames)
            target_length: Target audio length (for padding/truncation)
            
        Returns:
            Reconstructed audio (batch, time)
        """
        batch_size, freq_bins, time_frames = magnitude.shape
        
        # Initialize phase
        if self.init_phase == 'random':
            phase = torch.rand_like(magnitude) * 2 * math.pi - math.pi
        elif self.init_phase == 'zero':
            phase = torch.zeros_like(magnitude)
        elif self.init_phase == 'previous' and self.prev_phase is not None:
            # Use previous phase, resizing if necessary
            if self.prev_phase.shape == magnitude.shape:
                phase = self.prev_phase
            else:
                # Resize previous phase to match current magnitude
                phase = F.interpolate(
                    self.prev_phase.unsqueeze(1),
                    size=(freq_bins, time_frames),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
        else:
            # Fallback to random
            phase = torch.rand_like(magnitude) * 2 * math.pi - math.pi
        
        # Apply reconstruction method
        audio = self.reconstructor(magnitude, phase)
        
        # Store phase for next iteration (if using 'previous' init)
        if self.init_phase == 'previous':
            with torch.no_grad():
                # Compute final phase from reconstructed audio
                final_stft = torch.stft(
                    audio.reshape(-1),
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    window=self.window,
                    return_complex=True
                )
                # Reshape to match batch structure
                final_stft = final_stft.view(batch_size, freq_bins, -1)
                self.prev_phase = torch.angle(final_stft).detach()
        
        # Adjust length if specified
        if target_length is not None:
            if audio.shape[-1] > target_length:
                audio = audio[:, :target_length]
            elif audio.shape[-1] < target_length:
                pad_length = target_length - audio.shape[-1]
                audio = F.pad(audio, (0, pad_length), mode='constant', value=0)
        
        return audio
    
    def reset_phase_memory(self):
        """Reset stored phase information."""
        self.prev_phase = None


class GriffinLimReconstructor(nn.Module):
    """Griffin-Lim algorithm for phase reconstruction."""
    
    def __init__(self, n_fft, hop_length, num_iterations, momentum):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_iterations = num_iterations
        self.momentum = momentum
        
        self.register_buffer('window', torch.hann_window(n_fft))
    
    def forward(self, magnitude: torch.Tensor, init_phase: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct audio using Griffin-Lim algorithm.
        
        Args:
            magnitude: Magnitude spectrogram (batch, freq_bins, time_frames)
            init_phase: Initial phase (batch, freq_bins, time_frames)
            
        Returns:
            Reconstructed audio (batch, time)
        """
        batch_size = magnitude.shape[0]
        
        # Initialize complex spectrogram
        complex_spec = magnitude * torch.exp(1j * init_phase)
        
        # Store previous estimate for momentum
        prev_complex_spec = complex_spec.clone()
        
        # Griffin-Lim iterations
        for iteration in range(self.num_iterations):
            # Convert to time domain
            audio_batch = []
            for b in range(batch_size):
                audio = torch.istft(
                    complex_spec[b],
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    window=self.window,
                    return_complex=False
                )
                audio_batch.append(audio)
            
            audio = torch.stack(audio_batch, dim=0)
            
            # Convert back to frequency domain
            new_complex_spec_batch = []
            for b in range(batch_size):
                new_stft = torch.stft(
                    audio[b],
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    window=self.window,
                    return_complex=True
                )
                new_complex_spec_batch.append(new_stft)
            
            new_complex_spec = torch.stack(new_complex_spec_batch, dim=0)
            
            # Maintain magnitude, update phase
            new_phase = torch.angle(new_complex_spec)
            complex_spec = magnitude * torch.exp(1j * new_phase)
            
            # Apply momentum
            if self.momentum > 0 and iteration > 0:
                complex_spec = (self.momentum * prev_complex_spec + 
                               (1 - self.momentum) * complex_spec)
            
            prev_complex_spec = complex_spec.clone()
        
        # Final conversion to time domain
        final_audio_batch = []
        for b in range(batch_size):
            final_audio = torch.istft(
                complex_spec[b],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                return_complex=False
            )
            final_audio_batch.append(final_audio)
        
        return torch.stack(final_audio_batch, dim=0)


class NeuralPhaseReconstructor(nn.Module):
    """Neural network-based phase reconstruction."""
    
    def __init__(self, n_fft, hop_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        freq_bins = n_fft // 2 + 1
        
        # Neural network for phase prediction
        self.phase_net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Tanh()  # Output phase in [-1, 1], scale to [-π, π]
        )
        
        self.register_buffer('window', torch.hann_window(n_fft))
    
    def forward(self, magnitude: torch.Tensor, init_phase: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct audio using neural phase prediction.
        
        Args:
            magnitude: Magnitude spectrogram (batch, freq_bins, time_frames)
            init_phase: Initial phase (ignored for neural method)
            
        Returns:
            Reconstructed audio (batch, time)
        """
        batch_size = magnitude.shape[0]
        
        # Predict phase using neural network
        # Add channel dimension for Conv2D
        magnitude_input = magnitude.unsqueeze(1)  # (batch, 1, freq, time)
        
        # Predict phase
        phase_pred = self.phase_net(magnitude_input)  # (batch, 1, freq, time)
        phase = phase_pred.squeeze(1) * math.pi  # Scale to [-π, π]
        
        # Combine magnitude and predicted phase
        complex_spec = magnitude * torch.exp(1j * phase)
        
        # Convert to time domain
        audio_batch = []
        for b in range(batch_size):
            audio = torch.istft(
                complex_spec[b],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                return_complex=False
            )
            audio_batch.append(audio)
        
        return torch.stack(audio_batch, dim=0)


class IterativePhaseReconstructor(nn.Module):
    """Iterative phase reconstruction with consistency constraints."""
    
    def __init__(self, n_fft, hop_length, num_iterations):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_iterations = num_iterations
        
        self.register_buffer('window', torch.hann_window(n_fft))
    
    def forward(self, magnitude: torch.Tensor, init_phase: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct audio using iterative consistency.
        
        Args:
            magnitude: Magnitude spectrogram (batch, freq_bins, time_frames)
            init_phase: Initial phase (batch, freq_bins, time_frames)
            
        Returns:
            Reconstructed audio (batch, time)
        """
        batch_size = magnitude.shape[0]
        
        # Initialize phase
        phase = init_phase.clone()
        
        # Iterative refinement
        for iteration in range(self.num_iterations):
            # Combine magnitude and phase
            complex_spec = magnitude * torch.exp(1j * phase)
            
            # Convert to time domain and back to enforce consistency
            for b in range(batch_size):
                # To time domain
                audio = torch.istft(
                    complex_spec[b],
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    window=self.window,
                    return_complex=False
                )
                
                # Back to frequency domain
                new_stft = torch.stft(
                    audio,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    window=self.window,
                    return_complex=True
                )
                
                # Update phase while keeping magnitude
                new_phase = torch.angle(new_stft)
                phase[b] = new_phase
        
        # Final reconstruction
        final_complex_spec = magnitude * torch.exp(1j * phase)
        
        audio_batch = []
        for b in range(batch_size):
            audio = torch.istft(
                final_complex_spec[b],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                return_complex=False
            )
            audio_batch.append(audio)
        
        return torch.stack(audio_batch, dim=0)


class HeuristicPhaseReconstructor(nn.Module):
    """Heuristic phase reconstruction using local phase relationships."""
    
    def __init__(self, n_fft, hop_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        self.register_buffer('window', torch.hann_window(n_fft))
    
    def forward(self, magnitude: torch.Tensor, init_phase: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct audio using heuristic phase relationships.
        
        Args:
            magnitude: Magnitude spectrogram (batch, freq_bins, time_frames)
            init_phase: Initial phase (batch, freq_bins, time_frames)
            
        Returns:
            Reconstructed audio (batch, time)
        """
        batch_size, freq_bins, time_frames = magnitude.shape
        
        # Start with initial phase
        phase = init_phase.clone()
        
        # Apply heuristic improvements
        # 1. Smooth phase across frequency for each time frame
        if freq_bins > 2:
            for t in range(time_frames):
                for b in range(batch_size):
                    # Simple phase smoothing
                    phase_frame = phase[b, :, t]
                    
                    # Unwrap phase
                    unwrapped = self._unwrap_phase(phase_frame)
                    
                    # Apply smoothing
                    if len(unwrapped) >= 3:
                        # Simple moving average
                        smoothed = unwrapped.clone()
                        for f in range(1, len(unwrapped) - 1):
                            smoothed[f] = 0.5 * unwrapped[f] + 0.25 * (unwrapped[f-1] + unwrapped[f+1])
                        
                        # Wrap back
                        wrapped = torch.remainder(smoothed + math.pi, 2 * math.pi) - math.pi
                        phase[b, :, t] = wrapped
        
        # 2. Ensure temporal consistency
        if time_frames > 1:
            for f in range(freq_bins):
                for b in range(batch_size):
                    # Phase derivative should be consistent with hop length
                    phase_seq = phase[b, f, :]
                    
                    # Compute expected phase advance
                    expected_advance = 2 * math.pi * f * self.hop_length / self.n_fft
                    
                    # Adjust phase to maintain consistency
                    for t in range(1, time_frames):
                        predicted_phase = phase_seq[t-1] + expected_advance
                        actual_phase = phase_seq[t]
                        
                        # Find closest equivalent phase
                        phase_diff = actual_phase - predicted_phase
                        # Wrap to [-π, π]
                        phase_diff = torch.remainder(phase_diff + math.pi, 2 * math.pi) - math.pi
                        
                        # Apply correction if difference is large
                        if abs(phase_diff) > math.pi / 2:
                            phase[b, f, t] = predicted_phase
        
        # Combine magnitude and corrected phase
        complex_spec = magnitude * torch.exp(1j * phase)
        
        # Convert to time domain
        audio_batch = []
        for b in range(batch_size):
            audio = torch.istft(
                complex_spec[b],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                return_complex=False
            )
            audio_batch.append(audio)
        
        return torch.stack(audio_batch, dim=0)
    
    def _unwrap_phase(self, phase: torch.Tensor) -> torch.Tensor:
        """Simple phase unwrapping."""
        unwrapped = phase.clone()
        
        for i in range(1, len(phase)):
            diff = unwrapped[i] - unwrapped[i-1]
            
            # Wrap difference to [-π, π]
            while diff > math.pi:
                unwrapped[i] -= 2 * math.pi
                diff = unwrapped[i] - unwrapped[i-1]
            
            while diff < -math.pi:
                unwrapped[i] += 2 * math.pi
                diff = unwrapped[i] - unwrapped[i-1]
        
        return unwrapped


class MagnitudeToAudio(nn.Module):
    """
    Complete magnitude-to-audio conversion pipeline.
    
    Combines phase reconstruction with post-processing for high-quality
    audio synthesis from magnitude spectrograms.
    
    Args:
        reconstruction_method: Phase reconstruction method
        post_processing: Whether to apply post-processing
        target_sample_rate: Target sample rate for output
        **kwargs: Arguments passed to PhaseReconstruction
    """
    
    def __init__(
        self,
        reconstruction_method: str = 'griffin_lim',
        post_processing: bool = True,
        target_sample_rate: int = 22050,
        **kwargs
    ):
        super().__init__()
        
        self.post_processing = post_processing
        self.target_sample_rate = target_sample_rate
        
        # Phase reconstructor
        self.phase_reconstructor = PhaseReconstruction(
            method=reconstruction_method,
            **kwargs
        )
        
        # Post-processing components
        if post_processing:
            # Simple high-pass filter to remove DC
            self.register_buffer(
                'highpass_filter',
                torch.tensor([1.0, -0.95], dtype=torch.float32)
            )
    
    def forward(
        self,
        magnitude: torch.Tensor,
        target_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Convert magnitude spectrogram to audio.
        
        Args:
            magnitude: Magnitude spectrogram (batch, freq_bins, time_frames)
            target_length: Target audio length
            
        Returns:
            High-quality audio (batch, time)
        """
        # Phase reconstruction
        audio = self.phase_reconstructor(magnitude, target_length)
        
        # Post-processing
        if self.post_processing:
            audio = self._apply_post_processing(audio)
        
        return audio
    
    def _apply_post_processing(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply post-processing to improve audio quality."""
        # High-pass filter to remove DC component
        # Simple IIR filter: y[n] = x[n] - 0.95 * x[n-1]
        if audio.shape[-1] > 1:
            filtered = audio.clone()
            for b in range(audio.shape[0]):
                for i in range(1, audio.shape[-1]):
                    filtered[b, i] = audio[b, i] - 0.95 * audio[b, i-1]
            audio = filtered
        
        # Normalize to prevent clipping
        if audio.numel() > 0:
            audio_max = torch.max(torch.abs(audio), dim=-1, keepdim=True)[0]
            audio_max = torch.clamp(audio_max, min=1e-8)  # Prevent division by zero
            audio = audio / audio_max * 0.95  # Leave some headroom
        
        return audio


class PhaseAnalyzer(nn.Module):
    """
    Phase analysis tools for evaluating reconstruction quality.
    
    Provides metrics and analysis for phase reconstruction performance.
    
    Args:
        n_fft: FFT size for analysis
        hop_length: Hop length for analysis
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(self, n_fft: int = 2048, hop_length: int = 512, **kwargs):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        self.register_buffer('window', torch.hann_window(n_fft))
    
    def forward(
        self,
        original_audio: torch.Tensor,
        reconstructed_audio: torch.Tensor
    ) -> dict:
        """
        Analyze phase reconstruction quality.
        
        Args:
            original_audio: Original audio (batch, time)
            reconstructed_audio: Reconstructed audio (batch, time)
            
        Returns:
            Dictionary of analysis metrics
        """
        batch_size = original_audio.shape[0]
        metrics = {}
        
        # Ensure same length
        min_length = min(original_audio.shape[-1], reconstructed_audio.shape[-1])
        original = original_audio[:, :min_length]
        reconstructed = reconstructed_audio[:, :min_length]
        
        # Time-domain metrics
        mse = F.mse_loss(reconstructed, original)
        metrics['mse'] = mse.item()
        
        # Spectral metrics
        orig_stfts = []
        recon_stfts = []
        
        for b in range(batch_size):
            orig_stft = torch.stft(
                original[b],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                return_complex=True
            )
            recon_stft = torch.stft(
                reconstructed[b],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                return_complex=True
            )
            orig_stfts.append(orig_stft)
            recon_stfts.append(recon_stft)
        
        orig_stft = torch.stack(orig_stfts, dim=0)
        recon_stft = torch.stack(recon_stfts, dim=0)
        
        # Magnitude error
        orig_mag = torch.abs(orig_stft)
        recon_mag = torch.abs(recon_stft)
        mag_error = F.mse_loss(recon_mag, orig_mag)
        metrics['magnitude_mse'] = mag_error.item()
        
        # Phase error
        orig_phase = torch.angle(orig_stft)
        recon_phase = torch.angle(recon_stft)
        
        # Phase difference (wrapped to [-π, π])
        phase_diff = orig_phase - recon_phase
        phase_diff = torch.remainder(phase_diff + math.pi, 2 * math.pi) - math.pi
        phase_error = torch.mean(torch.abs(phase_diff))
        metrics['phase_error'] = phase_error.item()
        
        # Spectral convergence
        spectral_convergence = torch.norm(orig_stft - recon_stft) / torch.norm(orig_stft)
        metrics['spectral_convergence'] = spectral_convergence.item()
        
        return metrics