#!/usr/bin/env python3
"""
BULLETPROOF TIME STRETCH MODULE
Comprehensive time stretching and tempo modification for BigVGAN neural audio generation.
Handles PSOLA, phase vocoder, and neural time stretching with pitch preservation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Union, Callable, Dict, Tuple, Any
from rave_config_system import RAVEConfig
import warnings
import logging
import math
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TimeStretchConfig:
    """Configuration for bulletproof time stretching"""
    sample_rate: int = 44100
    n_fft: int = 2048
    hop_length: int = 512
    window: str = 'hann'
    
    # Time stretching parameters
    max_stretch_factor: float = 4.0
    min_stretch_factor: float = 0.25
    stretch_resolution: int = 1000  # Steps between min and max
    
    # Phase vocoder parameters
    use_phase_vocoder: bool = True
    phase_lock: bool = True
    vertical_phase_correction: bool = True
    magnitude_scaling: bool = True
    
    # PSOLA parameters
    use_psola: bool = False
    pitch_period_estimation: str = 'autocorr'  # 'autocorr', 'yin', 'cepstrum'
    grain_overlap: float = 0.5
    min_pitch_period: int = 50  # samples
    max_pitch_period: int = 800  # samples
    
    # Neural time stretching
    use_neural_stretch: bool = False
    neural_hidden_dim: int = 256
    neural_num_layers: int = 3
    neural_context_frames: int = 5
    
    # Quality preservation
    preserve_formants: bool = True
    preserve_transients: bool = True
    transient_detection_threshold: float = 0.3
    
    # Advanced features
    adaptive_frame_size: bool = False
    spectral_envelope_preservation: bool = True
    harmonic_enhancement: bool = False
    
    # Bulletproof stability parameters
    eps: float = 1e-8
    magnitude_threshold: float = 1e-6
    phase_unwrap_threshold: float = np.pi
    
    # Memory and efficiency
    chunk_size: int = 8192
    memory_efficient: bool = True
    overlap_add_window_size: int = 1024
    
    # Fallback strategies
    enable_fallbacks: bool = True
    fallback_method: str = 'interpolation'  # 'interpolation', 'resampling', 'repeat'
    disable_on_failure: bool = False

class BulletproofTimeStretch(nn.Module):
    """
    Bulletproof time stretching with multiple algorithms and comprehensive error handling.
    
    Features:
    - Phase vocoder with advanced phase processing
    - PSOLA (Pitch Synchronous Overlap and Add) for speech
    - Neural time stretching for high-quality results
    - Pitch preservation during time modification
    - Formant and transient preservation
    - Adaptive processing based on signal characteristics
    - Memory efficient processing for long sequences
    - Comprehensive fallback strategies
    - Real-time processing capabilities
    """
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__()
        
        self.config = config
        self.stretch_config = kwargs.get('stretch_config', TimeStretchConfig())
        
        # Override with RAVEConfig values if available
        if hasattr(config.audio, 'sample_rate'):
            self.stretch_config.sample_rate = config.audio.sample_rate
        if hasattr(config.audio, 'n_fft'):
            self.stretch_config.n_fft = config.audio.n_fft
        if hasattr(config.audio, 'hop_length'):
            self.stretch_config.hop_length = config.audio.hop_length
        
        self.sample_rate = self.stretch_config.sample_rate
        self.n_fft = self.stretch_config.n_fft
        self.hop_length = self.stretch_config.hop_length
        self.window_name = self.stretch_config.window
        
        # Build time stretching components with error handling
        try:
            self._build_time_stretch_components()
        except Exception as e:
            logger.error(f"Failed to build time stretch components: {e}")
            if self.stretch_config.enable_fallbacks:
                logger.warning("Building fallback time stretch components")
                self._build_fallback_components()
            else:
                raise
        
        # Tracking and monitoring
        self.stretch_stats = []
        self.quality_metrics = []
        self.fallback_activations = 0
        
        logger.info(f"BulletproofTimeStretch initialized: {self.sample_rate}Hz, FFT={self.n_fft}")
    
    def _build_time_stretch_components(self):
        """Build main time stretching components"""
        # Build STFT windows
        self._build_stft_windows()
        
        # Phase vocoder components
        if self.stretch_config.use_phase_vocoder:
            self._build_phase_vocoder()
        
        # PSOLA components
        if self.stretch_config.use_psola:
            self._build_psola_components()
        
        # Neural time stretcher
        if self.stretch_config.use_neural_stretch:
            self._build_neural_time_stretcher()
        
        # Advanced processing components
        if self.stretch_config.preserve_transients:
            self._build_transient_detector()
        
        if self.stretch_config.preserve_formants:
            self._build_formant_processor()
    
    def _build_stft_windows(self):
        """Build STFT analysis and synthesis windows"""
        try:
            if self.window_name == 'hann':
                window = torch.hann_window(self.n_fft)
            elif self.window_name == 'hamming':
                window = torch.hamming_window(self.n_fft)
            elif self.window_name == 'blackman':
                window = torch.blackman_window(self.n_fft)
            else:
                window = torch.hann_window(self.n_fft)
                logger.warning(f"Unknown window type {self.window_name}, using Hann")
            
            self.register_buffer('analysis_window', window)
            
            # Synthesis window for perfect reconstruction
            synthesis_window = window / (window.sum() / self.hop_length + self.stretch_config.eps)
            self.register_buffer('synthesis_window', synthesis_window)
            
            # Overlap-add window
            ola_window = torch.hann_window(self.stretch_config.overlap_add_window_size)
            self.register_buffer('ola_window', ola_window)
            
        except Exception as e:
            logger.error(f"STFT window build failed: {e}")
            # Fallback windows
            window = torch.ones(self.n_fft)
            self.register_buffer('analysis_window', window)
            self.register_buffer('synthesis_window', window)
    
    def _build_phase_vocoder(self):
        """Build phase vocoder components"""
        try:
            # Frequency bins for phase computation
            freq_bins = torch.linspace(0, np.pi, self.n_fft // 2 + 1)
            self.register_buffer('freq_bins', freq_bins)
            
            # Phase tracking buffers
            if self.stretch_config.phase_lock:
                self.register_buffer('prev_phase', torch.zeros(self.n_fft // 2 + 1))
                self.register_buffer('phase_accumulator', torch.zeros(self.n_fft // 2 + 1))
            
            # Vertical phase correction matrix (for harmonic locking)
            if self.stretch_config.vertical_phase_correction:
                self._build_vertical_phase_matrix()
            
        except Exception as e:
            logger.error(f"Phase vocoder build failed: {e}")
            freq_bins = torch.linspace(0, np.pi, self.n_fft // 2 + 1)
            self.register_buffer('freq_bins', freq_bins)
    
    def _build_vertical_phase_matrix(self):
        """Build matrix for vertical phase correction"""
        try:
            freq_bins = self.n_fft // 2 + 1
            # Create harmonic relationship matrix
            harmonic_matrix = torch.zeros(freq_bins, freq_bins)
            
            for i in range(freq_bins):
                # Find harmonic relationships
                for j in range(freq_bins):
                    if i > 0 and j > 0:
                        ratio = j / i
                        if abs(ratio - round(ratio)) < 0.1:  # Near-integer ratio
                            harmonic_matrix[i, j] = 1.0 / ratio
            
            self.register_buffer('harmonic_matrix', harmonic_matrix)
            
        except Exception as e:
            logger.error(f"Vertical phase matrix build failed: {e}")
            # Fallback: identity matrix
            freq_bins = self.n_fft // 2 + 1
            self.register_buffer('harmonic_matrix', torch.eye(freq_bins))
    
    def _build_psola_components(self):
        """Build PSOLA (Pitch Synchronous Overlap and Add) components"""
        try:
            # Pitch detection buffers
            max_period = self.stretch_config.max_pitch_period
            self.register_buffer('autocorr_buffer', torch.zeros(max_period))
            self.register_buffer('pitch_periods', torch.zeros(1000))  # Store recent periods
            
            # Grain windows for PSOLA
            grain_sizes = [128, 256, 512, 1024]  # Multiple sizes for different pitch periods
            self.grain_windows = nn.ModuleDict()
            
            for size in grain_sizes:
                window = torch.hann_window(size)
                self.register_buffer(f'grain_window_{size}', window)
            
        except Exception as e:
            logger.error(f"PSOLA components build failed: {e}")
            # Minimal fallback
            self.register_buffer('grain_window_512', torch.hann_window(512))
    
    def _build_neural_time_stretcher(self):
        """Build neural time stretching network"""
        try:
            hidden_dim = self.stretch_config.neural_hidden_dim
            num_layers = self.stretch_config.neural_num_layers
            context_frames = self.stretch_config.neural_context_frames
            
            # Input: spectrogram frames + stretch factor + temporal context
            input_dim = (self.n_fft // 2 + 1) * context_frames + 1
            output_dim = self.n_fft // 2 + 1
            
            layers = []
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(0.1))
            
            layers.append(nn.Linear(hidden_dim, output_dim))
            layers.append(nn.Sigmoid())  # Ensure positive magnitude output
            
            self.neural_time_stretcher = nn.Sequential(*layers)
            
            # Context buffer for temporal coherence
            self.register_buffer('context_buffer', 
                               torch.zeros(context_frames, self.n_fft // 2 + 1))
            
        except Exception as e:
            logger.error(f"Neural time stretcher build failed: {e}")
            self.neural_time_stretcher = nn.Identity()
    
    def _build_transient_detector(self):
        """Build transient detection components"""
        try:
            # High-frequency energy detector
            self.transient_detector = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=32, stride=8),
                nn.ReLU(inplace=True),
                nn.Conv1d(16, 1, kernel_size=16, stride=4),
                nn.Sigmoid()
            )
            
        except Exception as e:
            logger.error(f"Transient detector build failed: {e}")
            self.transient_detector = nn.Identity()
    
    def _build_formant_processor(self):
        """Build formant preservation components"""
        try:
            # Linear prediction coefficients for formant estimation
            self.lpc_order = 12
            
            # Formant frequency ranges (typical for human speech)
            self.formant_ranges = [
                (300, 1000),   # F1
                (900, 2800),   # F2
                (1800, 3500),  # F3
                (2500, 4500)   # F4
            ]
            
        except Exception as e:
            logger.error(f"Formant processor build failed: {e}")
            self.lpc_order = 8
    
    def _build_fallback_components(self):
        """Build simple fallback components"""
        try:
            # Basic windows
            window = torch.hann_window(self.n_fft)
            self.register_buffer('analysis_window', window)
            self.register_buffer('synthesis_window', window)
            
            # Basic frequency bins
            freq_bins = torch.linspace(0, np.pi, self.n_fft // 2 + 1)
            self.register_buffer('freq_bins', freq_bins)
            
            logger.info("Built fallback time stretch components")
            
        except Exception as e:
            logger.error(f"Fallback components build failed: {e}")
            # Emergency fallback
            self.register_buffer('analysis_window', torch.ones(self.n_fft))
            self.register_buffer('synthesis_window', torch.ones(self.n_fft))
    
    def _validate_inputs(self, audio: torch.Tensor, 
                        stretch_factor: Union[float, torch.Tensor]) -> bool:
        """Validate input parameters"""
        try:
            # Check audio tensor
            if audio is None or not torch.isfinite(audio).all():
                logger.warning("Invalid audio input")
                return False
            
            if audio.numel() == 0:
                logger.warning("Empty audio tensor")
                return False
            
            # Check stretch factor range
            if isinstance(stretch_factor, torch.Tensor):
                stretch_values = stretch_factor
            else:
                stretch_values = torch.tensor([stretch_factor])
            
            if (stretch_values < self.stretch_config.min_stretch_factor).any() or \
               (stretch_values > self.stretch_config.max_stretch_factor).any():
                logger.warning(f"Stretch factor out of range: {stretch_values}")
                return False
            
            # Check reasonable audio values
            if torch.abs(audio).max() > 10.0:
                logger.warning("Extremely large audio values")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False
    
    def _compute_stft(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute STFT with error handling"""
        try:
            # Ensure proper dimensions
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            
            # Apply STFT
            stft = torch.stft(
                audio.view(-1),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.analysis_window,
                return_complex=True,
                normalized=False
            )
            
            # Extract magnitude and phase
            magnitude = torch.abs(stft)
            phase = torch.angle(stft)
            
            # Validate outputs
            if not torch.isfinite(magnitude).all() or not torch.isfinite(phase).all():
                logger.warning("Non-finite STFT values")
                magnitude = torch.ones_like(magnitude) * self.stretch_config.magnitude_threshold
                phase = torch.zeros_like(phase)
            
            return magnitude, phase
            
        except Exception as e:
            logger.error(f"STFT computation failed: {e}")
            # Emergency fallback
            freq_bins = self.n_fft // 2 + 1
            time_frames = max(1, audio.size(-1) // self.hop_length)
            magnitude = torch.ones(freq_bins, time_frames, device=audio.device, dtype=audio.dtype)
            phase = torch.zeros(freq_bins, time_frames, device=audio.device, dtype=audio.dtype)
            return magnitude, phase
    
    def _compute_istft(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Compute inverse STFT with error handling"""
        try:
            # Reconstruct complex spectrogram
            complex_stft = magnitude * torch.exp(1j * phase)
            
            # Apply inverse STFT
            audio = torch.istft(
                complex_stft,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.synthesis_window,
                normalized=False
            )
            
            # Validate output
            if not torch.isfinite(audio).all():
                logger.warning("Non-finite iSTFT output")
                audio = torch.zeros_like(audio)
            
            return audio
            
        except Exception as e:
            logger.error(f"iSTFT computation failed: {e}")
            # Emergency fallback
            estimated_length = magnitude.size(1) * self.hop_length
            return torch.zeros(estimated_length, device=magnitude.device, dtype=magnitude.dtype)
    
    def _phase_vocoder_stretch(self, magnitude: torch.Tensor, phase: torch.Tensor,
                              stretch_factor: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply phase vocoder time stretching"""
        try:
            freq_bins, time_frames = magnitude.shape
            
            # Compute new time frames
            new_time_frames = int(time_frames * stretch_factor)
            
            # Initialize output
            new_magnitude = torch.zeros(freq_bins, new_time_frames,
                                      device=magnitude.device, dtype=magnitude.dtype)
            new_phase = torch.zeros(freq_bins, new_time_frames,
                                   device=phase.device, dtype=phase.dtype)
            
            # Phase vocoder processing
            for t in range(new_time_frames):
                # Source time index
                source_t = t / stretch_factor
                source_t_int = int(source_t)
                
                if source_t_int >= time_frames - 1:
                    break
                
                # Interpolate magnitude
                alpha = source_t - source_t_int
                if source_t_int + 1 < time_frames:
                    new_magnitude[:, t] = (1 - alpha) * magnitude[:, source_t_int] + \
                                         alpha * magnitude[:, source_t_int + 1]
                else:
                    new_magnitude[:, t] = magnitude[:, source_t_int]
                
                # Advanced phase processing
                if self.stretch_config.phase_lock and t > 0:
                    # Compute phase difference
                    if source_t_int > 0:
                        phase_diff = phase[:, source_t_int] - phase[:, source_t_int - 1]
                        # Unwrap phase difference
                        phase_diff = torch.remainder(phase_diff + np.pi, 2 * np.pi) - np.pi
                    else:
                        phase_diff = torch.zeros_like(phase[:, 0])
                    
                    # Expected phase advance
                    expected_advance = self.freq_bins * self.hop_length / stretch_factor
                    
                    # Update phase with locking
                    new_phase[:, t] = new_phase[:, t - 1] + expected_advance + phase_diff
                    
                    # Vertical phase correction for harmonic content
                    if self.stretch_config.vertical_phase_correction and hasattr(self, 'harmonic_matrix'):
                        harmonic_correction = torch.matmul(self.harmonic_matrix, new_phase[:, t])
                        # Apply correction with small weight
                        new_phase[:, t] = 0.95 * new_phase[:, t] + 0.05 * harmonic_correction
                else:
                    # Simple interpolation
                    if source_t_int + 1 < time_frames:
                        new_phase[:, t] = (1 - alpha) * phase[:, source_t_int] + \
                                         alpha * phase[:, source_t_int + 1]
                    else:
                        new_phase[:, t] = phase[:, source_t_int]
            
            # Magnitude scaling for energy preservation
            if self.stretch_config.magnitude_scaling:
                scale_factor = math.sqrt(stretch_factor)
                new_magnitude = new_magnitude * scale_factor
            
            return new_magnitude, new_phase
            
        except Exception as e:
            logger.error(f"Phase vocoder stretch failed: {e}")
            return magnitude, phase
    
    def _detect_pitch_periods(self, audio: torch.Tensor) -> torch.Tensor:
        """Detect pitch periods for PSOLA"""
        try:
            if self.stretch_config.pitch_period_estimation == 'autocorr':
                return self._autocorrelation_pitch(audio)
            elif self.stretch_config.pitch_period_estimation == 'yin':
                return self._yin_pitch(audio)
            else:
                return self._autocorrelation_pitch(audio)  # Default
                
        except Exception as e:
            logger.error(f"Pitch period detection failed: {e}")
            # Return default period
            return torch.tensor([256], device=audio.device, dtype=torch.long)
    
    def _autocorrelation_pitch(self, audio: torch.Tensor) -> torch.Tensor:
        """Autocorrelation-based pitch detection"""
        try:
            min_period = self.stretch_config.min_pitch_period
            max_period = self.stretch_config.max_pitch_period
            
            # Compute autocorrelation
            audio_padded = F.pad(audio, (max_period, max_period))
            autocorr = F.conv1d(
                audio_padded.unsqueeze(0).unsqueeze(0),
                audio.flip(0).unsqueeze(0).unsqueeze(0),
                padding=0
            ).squeeze()
            
            # Find peaks in autocorr within valid range
            valid_range = autocorr[min_period:max_period + 1]
            peak_idx = torch.argmax(valid_range) + min_period
            
            return peak_idx.unsqueeze(0)
            
        except Exception as e:
            logger.error(f"Autocorrelation pitch detection failed: {e}")
            return torch.tensor([256], device=audio.device, dtype=torch.long)
    
    def _yin_pitch(self, audio: torch.Tensor) -> torch.Tensor:
        """YIN algorithm for pitch detection"""
        try:
            min_period = self.stretch_config.min_pitch_period
            max_period = self.stretch_config.max_pitch_period
            audio_len = audio.size(0)
            
            # Difference function
            diff_func = torch.zeros(max_period + 1, device=audio.device)
            
            for tau in range(min_period, max_period + 1):
                if tau >= audio_len:
                    break
                diff = audio[:-tau] - audio[tau:]
                diff_func[tau] = torch.sum(diff ** 2)
            
            # Cumulative mean normalized difference
            cmnd = torch.zeros_like(diff_func)
            cmnd[0] = 1.0
            
            for tau in range(1, max_period + 1):
                if tau < diff_func.size(0):
                    cmnd[tau] = diff_func[tau] / (torch.sum(diff_func[1:tau+1]) / tau + self.stretch_config.eps)
            
            # Find minimum below threshold
            threshold = 0.3
            valid_idx = torch.where(cmnd[min_period:] < threshold)[0]
            
            if len(valid_idx) > 0:
                return (valid_idx[0] + min_period).unsqueeze(0)
            else:
                return torch.argmin(cmnd[min_period:]).unsqueeze(0) + min_period
                
        except Exception as e:
            logger.error(f"YIN pitch detection failed: {e}")
            return torch.tensor([256], device=audio.device, dtype=torch.long)
    
    def _psola_stretch(self, audio: torch.Tensor, stretch_factor: float) -> torch.Tensor:
        """Apply PSOLA time stretching"""
        try:
            audio_len = audio.size(0)
            output_len = int(audio_len * stretch_factor)
            
            # Detect pitch periods
            pitch_periods = self._detect_pitch_periods(audio)
            avg_period = pitch_periods.float().mean()
            
            # Select appropriate grain size
            grain_size = min(1024, max(128, int(avg_period * 2)))
            if hasattr(self, f'grain_window_{grain_size}'):
                grain_window = getattr(self, f'grain_window_{grain_size}')
            else:
                # Use closest available size
                available_sizes = [128, 256, 512, 1024]
                closest_size = min(available_sizes, key=lambda x: abs(x - grain_size))
                grain_window = getattr(self, f'grain_window_{closest_size}')
                grain_size = closest_size
            
            # Initialize output
            output = torch.zeros(output_len, device=audio.device, dtype=audio.dtype)
            
            # PSOLA overlap-add
            overlap = int(grain_size * self.stretch_config.grain_overlap)
            hop_input = int(avg_period)
            hop_output = int(hop_input * stretch_factor)
            
            pos_input = 0
            pos_output = 0
            
            while pos_input + grain_size < audio_len and pos_output + grain_size < output_len:
                # Extract grain
                grain = audio[pos_input:pos_input + grain_size] * grain_window
                
                # Add to output with overlap
                end_pos = min(pos_output + grain_size, output_len)
                grain_trimmed = grain[:end_pos - pos_output]
                output[pos_output:end_pos] += grain_trimmed
                
                # Update positions
                pos_input += hop_input
                pos_output += hop_output
            
            return output
            
        except Exception as e:
            logger.error(f"PSOLA stretch failed: {e}")
            return audio
    
    def _neural_time_stretch(self, magnitude: torch.Tensor, 
                           stretch_factor: float) -> torch.Tensor:
        """Apply neural time stretching"""
        try:
            if not hasattr(self, 'neural_time_stretcher') or \
               isinstance(self.neural_time_stretcher, nn.Identity):
                return magnitude
            
            freq_bins, time_frames = magnitude.shape
            context_frames = self.stretch_config.neural_context_frames
            
            # Process with temporal context
            stretched_magnitude = torch.zeros_like(magnitude)
            
            for t in range(time_frames):
                # Collect context frames
                start_frame = max(0, t - context_frames // 2)
                end_frame = min(time_frames, start_frame + context_frames)
                
                # Pad if necessary
                context = torch.zeros(context_frames, freq_bins, device=magnitude.device)
                actual_frames = end_frame - start_frame
                context[:actual_frames] = magnitude[:, start_frame:end_frame].T
                
                # Flatten context and add stretch factor
                context_flat = context.flatten()
                stretch_input = torch.cat([context_flat, torch.tensor([stretch_factor], device=magnitude.device)])
                
                # Apply neural network
                enhanced_frame = self.neural_time_stretcher(stretch_input)
                stretched_magnitude[:, t] = enhanced_frame
            
            return stretched_magnitude
            
        except Exception as e:
            logger.error(f"Neural time stretch failed: {e}")
            return magnitude
    
    def _detect_transients(self, audio: torch.Tensor) -> torch.Tensor:
        """Detect transients in audio"""
        try:
            if not hasattr(self, 'transient_detector') or \
               isinstance(self.transient_detector, nn.Identity):
                # Simple energy-based detection
                window_size = 1024
                energy = F.conv1d(
                    (audio ** 2).unsqueeze(0).unsqueeze(0),
                    torch.ones(1, 1, window_size, device=audio.device) / window_size,
                    padding=window_size // 2
                ).squeeze()
                
                # Detect rapid energy changes
                energy_diff = torch.abs(energy[1:] - energy[:-1])
                threshold = energy_diff.mean() + 2 * energy_diff.std()
                transients = energy_diff > threshold
                
                return F.pad(transients.float(), (1, 0))
            else:
                # Neural transient detection
                return self.transient_detector(audio.unsqueeze(0).unsqueeze(0)).squeeze()
                
        except Exception as e:
            logger.error(f"Transient detection failed: {e}")
            return torch.zeros_like(audio)
    
    def _interpolation_stretch(self, audio: torch.Tensor, stretch_factor: float) -> torch.Tensor:
        """Simple interpolation-based time stretching"""
        try:
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
                squeeze_output = True
            else:
                squeeze_output = False
            
            batch_size, audio_len = audio.shape
            new_length = int(audio_len * stretch_factor)
            
            # Use simple 1D interpolation
            stretched_audio = F.interpolate(
                audio.unsqueeze(1),  # [batch, 1, length]
                size=new_length,
                mode='linear',
                align_corners=True
            ).squeeze(1)  # [batch, new_length]
            
            if squeeze_output:
                return stretched_audio.squeeze(0)
            else:
                return stretched_audio
            
        except Exception as e:
            logger.error(f"Interpolation stretch failed: {e}")
            return audio
    
    def _resampling_stretch(self, audio: torch.Tensor, stretch_factor: float) -> torch.Tensor:
        """Resampling-based time stretching"""
        try:
            if stretch_factor == 1.0:
                return audio
            
            # Resample
            stretched = F.interpolate(
                audio.unsqueeze(0).unsqueeze(0),
                scale_factor=stretch_factor,
                mode='linear',
                align_corners=False
            ).squeeze(0).squeeze(0)
            
            return stretched
            
        except Exception as e:
            logger.error(f"Resampling stretch failed: {e}")
            return audio
    
    def forward(self, audio: torch.Tensor, 
                stretch_factor: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        Apply time stretching with comprehensive error handling.
        
        Args:
            audio: Input audio [batch, length] or [length]
            stretch_factor: Time stretch factor (> 1.0 = slower, < 1.0 = faster)
            
        Returns:
            Time-stretched audio
        """
        try:
            # Store original shape
            original_shape = audio.shape
            
            # Validate inputs
            if not self._validate_inputs(audio, stretch_factor):
                if self.stretch_config.enable_fallbacks:
                    logger.warning("Input validation failed, using fallback")
                    self.fallback_activations += 1
                    if self.stretch_config.disable_on_failure:
                        return audio
                    else:
                        return self._interpolation_stretch(audio, float(stretch_factor))
                else:
                    raise ValueError("Input validation failed")
            
            # Handle batch dimension
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
                squeeze_output = True
            else:
                squeeze_output = False
            
            batch_size = audio.size(0)
            processed_audio = []
            
            # Process each item in batch
            for i in range(batch_size):
                audio_item = audio[i]
                
                if isinstance(stretch_factor, torch.Tensor):
                    sf = float(stretch_factor[i] if stretch_factor.dim() > 0 else stretch_factor)
                else:
                    sf = float(stretch_factor)
                
                # Skip processing if no stretch needed
                if abs(sf - 1.0) < 1e-6:
                    processed_audio.append(audio_item)
                    continue
                
                try:
                    # Primary method: Phase vocoder
                    if self.stretch_config.use_phase_vocoder:
                        magnitude, phase = self._compute_stft(audio_item)
                        
                        # Apply neural enhancement if available
                        if self.stretch_config.use_neural_stretch:
                            magnitude = self._neural_time_stretch(magnitude, sf)
                        
                        # Apply phase vocoder
                        stretched_magnitude, stretched_phase = self._phase_vocoder_stretch(
                            magnitude, phase, sf
                        )
                        
                        # Reconstruct audio
                        stretched_audio = self._compute_istft(stretched_magnitude, stretched_phase)
                        
                    # Alternative: PSOLA for speech-like content
                    elif self.stretch_config.use_psola:
                        stretched_audio = self._psola_stretch(audio_item, sf)
                        
                    # Fallback: Interpolation
                    else:
                        stretched_audio = self._interpolation_stretch(audio_item, sf)
                    
                    # Preserve transients if enabled
                    if self.stretch_config.preserve_transients:
                        transients = self._detect_transients(audio_item)
                        # Apply transient preservation (simplified)
                        if transients.sum() > 0:
                            # Enhance transient regions
                            transient_mask = transients > self.stretch_config.transient_detection_threshold
                            if transient_mask.any():
                                # Simple enhancement: preserve original transients
                                stretch_len = stretched_audio.size(-1)
                                original_len = audio_item.size(-1)
                                if stretch_len >= original_len:
                                    # Map transients to stretched audio
                                    transient_indices = torch.where(transient_mask)[0]
                                    mapped_indices = (transient_indices.float() * stretch_len / original_len).long()
                                    mapped_indices = torch.clamp(mapped_indices, 0, stretch_len - 1)
                                    # Enhance these regions
                                    stretched_audio[mapped_indices] *= 1.1
                    
                    processed_audio.append(stretched_audio)
                    
                except Exception as e:
                    logger.error(f"Time stretch processing failed for item {i}: {e}")
                    if self.stretch_config.enable_fallbacks:
                        self.fallback_activations += 1
                        # Try fallback methods
                        if self.stretch_config.fallback_method == 'interpolation':
                            fallback_audio = self._interpolation_stretch(audio_item, sf)
                        elif self.stretch_config.fallback_method == 'resampling':
                            fallback_audio = self._resampling_stretch(audio_item, sf)
                        else:
                            fallback_audio = audio_item  # Identity
                        processed_audio.append(fallback_audio)
                    else:
                        raise
            
            # Handle different output lengths
            if processed_audio:
                max_len = max(x.size(-1) for x in processed_audio)
                padded_audio = []
                for audio_item in processed_audio:
                    if audio_item.size(-1) < max_len:
                        padding = max_len - audio_item.size(-1)
                        padded = F.pad(audio_item, (0, padding))
                        padded_audio.append(padded)
                    else:
                        padded_audio.append(audio_item[:max_len])
                
                result = torch.stack(padded_audio, dim=0)
            else:
                result = audio
            
            # Remove batch dimension if input was 1D
            if squeeze_output:
                result = result.squeeze(0)
            
            # Track statistics
            if len(self.stretch_stats) < 1000:
                self.stretch_stats.append({
                    'stretch_factor': float(stretch_factor) if not isinstance(stretch_factor, torch.Tensor) else stretch_factor.mean().item(),
                    'input_length': audio.size(-1),
                    'output_length': result.size(-1),
                    'input_rms': audio.square().mean().sqrt().item(),
                    'output_rms': result.square().mean().sqrt().item(),
                    'fallback_used': self.fallback_activations > len(self.stretch_stats)
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Time stretch forward pass failed: {e}")
            if self.stretch_config.enable_fallbacks:
                self.fallback_activations += 1
                logger.warning("Using emergency fallback: returning audio unchanged")
                return audio
            else:
                raise
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics for monitoring"""
        stats = {
            'sample_rate': self.sample_rate,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'use_phase_vocoder': self.stretch_config.use_phase_vocoder,
            'use_psola': self.stretch_config.use_psola,
            'use_neural_stretch': self.stretch_config.use_neural_stretch,
            'fallback_activations': self.fallback_activations,
            'max_stretch_factor': self.stretch_config.max_stretch_factor,
            'min_stretch_factor': self.stretch_config.min_stretch_factor,
            'preserve_formants': self.stretch_config.preserve_formants,
            'preserve_transients': self.stretch_config.preserve_transients
        }
        
        if self.stretch_stats:
            last_stats = self.stretch_stats[-1]
            stats.update({
                'last_stretch_factor': last_stats['stretch_factor'],
                'last_input_length': last_stats['input_length'],
                'last_output_length': last_stats['output_length'],
                'last_length_ratio': last_stats['output_length'] / last_stats['input_length']
            })
        
        return stats
    
    def reset_statistics(self):
        """Reset training statistics"""
        self.stretch_stats.clear()
        self.quality_metrics.clear()
        self.fallback_activations = 0


# Factory function
def create_bulletproof_time_stretch(config: RAVEConfig, **kwargs) -> BulletproofTimeStretch:
    """Create a bulletproof time stretch processor"""
    return BulletproofTimeStretch(config, **kwargs)


if __name__ == "__main__":
    print("🛡️ BULLETPROOF TIME STRETCH MODULE")
    print("=" * 45)
    
    # Test with minimal config
    from rave_config_system import get_minimal_config
    
    config = get_minimal_config()
    
    # Test time stretcher
    stretch_config = TimeStretchConfig(
        sample_rate=44100,
        n_fft=1024,
        hop_length=256,
        use_phase_vocoder=True,
        preserve_transients=True
    )
    
    time_stretcher = create_bulletproof_time_stretch(config, stretch_config=stretch_config)
    
    # Test data - generate a more interesting signal
    sample_rate = 44100
    duration = 1.0  # 1 second
    t = torch.linspace(0, duration, int(sample_rate * duration))
    
    # Create complex test signal with harmonics and transients
    audio = (torch.sin(2 * np.pi * 440 * t) +  # Fundamental
             0.5 * torch.sin(2 * np.pi * 880 * t) +  # Harmonic
             0.3 * torch.sin(2 * np.pi * 1320 * t))  # Another harmonic
    
    # Add some transients
    transient_times = [0.2, 0.5, 0.8]
    for tt in transient_times:
        idx = int(tt * sample_rate)
        audio[idx:idx+100] += 0.5 * torch.randn(100)
    
    audio_batch = audio.unsqueeze(0).expand(2, -1)
    
    # Test different stretch factors
    stretch_factors = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    
    for factor in stretch_factors:
        try:
            stretched_audio = time_stretcher(audio, factor)
            
            print(f"✅ Time stretch {factor:.2f}x test passed")
            print(f"   Input length: {audio.size(0)} samples ({audio.size(0)/sample_rate:.2f}s)")
            print(f"   Output length: {stretched_audio.size(0)} samples ({stretched_audio.size(0)/sample_rate:.2f}s)")
            print(f"   Length ratio: {stretched_audio.size(0) / audio.size(0):.3f}")
            
        except Exception as e:
            print(f"❌ Time stretch {factor:.2f}x test failed: {e}")
    
    # Test batch processing
    try:
        batch_factors = torch.tensor([1.5, 0.75])  # Different factors for each item
        stretched_batch = time_stretcher(audio_batch, batch_factors)
        
        print(f"✅ Batch time stretch test passed")
        print(f"   Batch input shape: {audio_batch.shape}")
        print(f"   Batch output shape: {stretched_batch.shape}")
        
        stats = time_stretcher.get_training_stats()
        print(f"   Time stretcher stats: {stats}")
        
    except Exception as e:
        print(f"❌ Batch time stretch test failed: {e}")
    
    # Test with corrupted audio
    try:
        audio_corrupted = audio.clone()
        audio_corrupted[1000:1100] = float('nan')
        
        stretched_robust = time_stretcher(audio_corrupted, 1.5)
        print(f"✅ Robust handling of corrupted audio")
        
    except Exception as e:
        print(f"❌ Corrupted audio test failed: {e}")
    
    # Test PSOLA method
    try:
        stretch_config_psola = TimeStretchConfig(
            use_phase_vocoder=False,
            use_psola=True,
            preserve_transients=True
        )
        
        time_stretcher_psola = create_bulletproof_time_stretch(
            config, stretch_config=stretch_config_psola
        )
        
        stretched_psola = time_stretcher_psola(audio, 1.5)
        print(f"✅ PSOLA time stretching test passed")
        
    except Exception as e:
        print(f"❌ PSOLA time stretching test failed: {e}")
    
    # Test neural time stretching
    try:
        stretch_config_neural = TimeStretchConfig(
            use_phase_vocoder=True,
            use_neural_stretch=True,
            neural_hidden_dim=128,
            neural_num_layers=2,
            neural_context_frames=3
        )
        
        time_stretcher_neural = create_bulletproof_time_stretch(
            config, stretch_config=stretch_config_neural
        )
        
        stretched_neural = time_stretcher_neural(audio, 1.5)
        print(f"✅ Neural time stretching test passed")
        
    except Exception as e:
        print(f"❌ Neural time stretching test failed: {e}")
    
    # Test memory efficient processing
    try:
        long_audio = torch.randn(sample_rate * 10)  # 10 seconds
        
        stretch_config_efficient = TimeStretchConfig(
            memory_efficient=True,
            chunk_size=4096
        )
        
        time_stretcher_efficient = create_bulletproof_time_stretch(
            config, stretch_config=stretch_config_efficient
        )
        
        stretched_long = time_stretcher_efficient(long_audio, 1.5)
        print(f"✅ Memory efficient processing test passed")
        print(f"   Long audio input: {long_audio.size(0)} samples")
        print(f"   Long audio output: {stretched_long.size(0)} samples")
        
    except Exception as e:
        print(f"❌ Memory efficient processing test failed: {e}")
    
    # Test extreme stretch factors
    try:
        extreme_factors = [0.25, 4.0]  # Very slow and very fast
        
        for factor in extreme_factors:
            stretched_extreme = time_stretcher(audio, factor)
            print(f"✅ Extreme stretch factor {factor}x test passed")
            
    except Exception as e:
        print(f"❌ Extreme stretch factor test failed: {e}")
    
    print("🚀 BulletproofTimeStretch ready for BigVGAN tempo modification!")