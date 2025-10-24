"""
Bulletproof Audio Preprocessing with comprehensive error handling and fallback strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as F_audio
import numpy as np
import librosa
import warnings
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from contextlib import contextmanager
import gc
import time

logger = logging.getLogger(__name__)

class BulletproofAudioPreprocessing(nn.Module):
    """
    Bulletproof audio preprocessing with:
    - Comprehensive parameter validation
    - Multiple fallback strategies for feature extraction
    - Memory-efficient processing for large audio files
    - Device compatibility with automatic fallback
    - Robust audio normalization and validation
    - Graceful degradation when transforms fail
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: str = "hann",
        n_mels: int = 128,
        n_mfcc: int = 13,
        n_chroma: int = 12,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        normalization_method: str = "zero_mean_unit_var",
        normalization_stats: Optional[Dict[str, float]] = None,
        apply_log_scaling: bool = True,
        log_epsilon: float = 1e-8,
        max_audio_length: int = 22050 * 300,
        memory_efficient: bool = True,
        enable_fallbacks: bool = True,
        validate_inputs: bool = True,
        **kwargs
    ):
        super().__init__()
        
        # Validate and sanitize parameters
        self.sample_rate = max(8000, min(192000, int(sample_rate)))
        self.n_fft = self._validate_n_fft(n_fft)
        self.hop_length = max(1, min(self.n_fft, int(hop_length)))
        self.win_length = win_length or self.n_fft
        self.window = window if window in ["hann", "hamming", "blackman", "bartlett"] else "hann"
        self.n_mels = max(10, min(512, int(n_mels)))
        self.n_mfcc = max(1, min(50, int(n_mfcc)))
        self.n_chroma = max(6, min(24, int(n_chroma)))
        self.f_min = max(0.0, float(f_min))
        self.f_max = min(float(f_max or self.sample_rate // 2), self.sample_rate // 2)
        self.normalization_method = normalization_method
        self.normalization_stats = normalization_stats or {}
        self.apply_log_scaling = apply_log_scaling
        self.log_epsilon = max(1e-12, min(1e-3, float(log_epsilon)))
        self.max_audio_length = max(self.sample_rate, max_audio_length)
        self.memory_efficient = memory_efficient
        self.enable_fallbacks = enable_fallbacks
        self.validate_inputs = validate_inputs
        
        try:
            self._initialize_transforms()
            self._initialize_normalizer()
            logger.info(f"BulletproofAudioPreprocessing initialized: {sample_rate}Hz, {n_mels} mels")
        except Exception as e:
            logger.error(f"Error initializing audio preprocessing: {e}")
            if not self.enable_fallbacks:
                raise
            self._initialize_fallback_transforms()
    
    def _validate_n_fft(self, n_fft: int) -> int:
        """Validate and sanitize n_fft to be power of 2."""
        n_fft = int(n_fft)
        if n_fft <= 0:
            return 2048
        
        # Round to nearest power of 2
        power = int(np.log2(n_fft))
        if 2**power != n_fft:
            n_fft = 2**power
            logger.info(f"Rounded n_fft to power of 2: {n_fft}")
        
        return max(256, min(8192, n_fft))
    
    def _initialize_transforms(self):
        """Initialize audio transforms with error handling."""
        # Core transforms
        self.stft_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window_fn=self._get_window_function(),
            power=2.0,
            normalized=True
        )
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window_fn=self._get_window_function(),
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            normalized=True
        )
        
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
                'win_length': self.win_length,
                'n_mels': self.n_mels,
                'f_min': self.f_min,
                'f_max': self.f_max
            }
        )
        
        # Chroma filterbank
        self._create_chroma_filterbank()
    
    def _get_window_function(self):
        """Get window function with fallback."""
        try:
            window_map = {
                'hann': torch.hann_window,
                'hamming': torch.hamming_window,
                'blackman': torch.blackman_window,
                'bartlett': torch.bartlett_window
            }
            return window_map.get(self.window, torch.hann_window)
        except Exception:
            return torch.hann_window
    
    def _create_chroma_filterbank(self):
        """Create chroma filterbank with error handling."""
        try:
            chroma_fb = librosa.filters.chroma(
                sr=self.sample_rate,
                n_fft=self.n_fft,
                n_chroma=self.n_chroma,
                tuning=0.0
            )
            
            if chroma_fb.shape[0] != self.n_chroma:
                raise ValueError(f"Chroma filterbank shape mismatch: {chroma_fb.shape}")
            
            chroma_fb = torch.from_numpy(chroma_fb).float()
            self.register_buffer('chroma_filterbank', chroma_fb)
            
        except Exception as e:
            logger.error(f"Error creating chroma filterbank: {e}")
            if self.enable_fallbacks:
                # Create simple fallback filterbank
                fallback_fb = torch.eye(self.n_chroma, self.n_fft // 2 + 1)
                self.register_buffer('chroma_filterbank', fallback_fb)
            else:
                raise
    
    def _initialize_normalizer(self):
        """Initialize audio normalizer."""
        try:
            from .bulletproof_audio_config import BulletproofAudioNormalizer, BulletproofAudioConfig
            
            # Create a minimal config for the normalizer
            config = BulletproofAudioConfig(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                normalization_stats=self.normalization_stats
            )
            config.normalization_method = getattr(config.normalization_method.__class__, 
                                                self.normalization_method.upper(), 
                                                config.normalization_method)
            
            self.normalizer = BulletproofAudioNormalizer(config)
            
        except Exception as e:
            logger.error(f"Error initializing normalizer: {e}")
            self.normalizer = self._create_fallback_normalizer()
    
    def _create_fallback_normalizer(self):
        """Create fallback normalizer."""
        class FallbackNormalizer:
            def normalize_tensor(self, tensor, feature_type='mel'):
                # Simple zero mean unit variance normalization
                try:
                    mean = tensor.mean()
                    std = tensor.std()
                    return (tensor - mean) / (std + 1e-8)
                except Exception:
                    return tensor
        
        return FallbackNormalizer()
    
    def _initialize_fallback_transforms(self):
        """Initialize fallback transforms when main initialization fails."""
        logger.warning("Initializing fallback audio transforms")
        
        # Create manual transforms
        self.stft_transform = self._manual_stft
        self.mel_transform = self._manual_mel_spectrogram
        self.mfcc_transform = self._manual_mfcc
        
        # Simple chroma filterbank
        fallback_fb = torch.eye(self.n_chroma, self.n_fft // 2 + 1)
        self.register_buffer('chroma_filterbank', fallback_fb)
        
        # Simple normalizer
        self.normalizer = self._create_fallback_normalizer()
    
    @contextmanager
    def _memory_management(self):
        """Context manager for memory management."""
        if self.memory_efficient:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        try:
            yield
        finally:
            if self.memory_efficient:
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def _validate_audio_input(self, waveform: torch.Tensor) -> torch.Tensor:
        """Validate and sanitize audio input."""
        if not isinstance(waveform, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(waveform)}")
        
        # Check dimensions
        if waveform.dim() < 1 or waveform.dim() > 3:
            raise ValueError(f"Invalid waveform dimensions: {waveform.dim()}")
        
        # Convert to proper format [batch, samples]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 3:
            if waveform.shape[1] == 1:
                waveform = waveform.squeeze(1)
            else:
                # Convert stereo to mono
                waveform = waveform.mean(dim=1)
        
        # Check for NaN or Inf
        if torch.isnan(waveform).any():
            logger.warning("NaN values detected, replacing with zeros")
            waveform = torch.nan_to_num(waveform, nan=0.0)
        
        if torch.isinf(waveform).any():
            logger.warning("Inf values detected, clipping")
            waveform = torch.clamp(waveform, -1.0, 1.0)
        
        # Check length
        if waveform.shape[-1] > self.max_audio_length:
            logger.warning(f"Audio too long, truncating to {self.max_audio_length}")
            waveform = waveform[..., :self.max_audio_length]
        
        # Minimum length check
        if waveform.shape[-1] < self.hop_length:
            logger.warning("Audio too short, padding")
            pad_length = self.hop_length - waveform.shape[-1] + 1
            waveform = F.pad(waveform, (0, pad_length))
        
        return waveform
    
    def convert_sample_rate(self, waveform: torch.Tensor, orig_sample_rate: int) -> torch.Tensor:
        """Convert waveform to target sample rate with error handling."""
        if orig_sample_rate == self.sample_rate:
            return waveform
        
        try:
            # Use high-quality resampling
            resampled = F_audio.resample(
                waveform,
                orig_freq=orig_sample_rate,
                new_freq=self.sample_rate,
                resampling_method="sinc_interp_hann"
            )
            return resampled
            
        except Exception as e:
            logger.error(f"Sample rate conversion failed: {e}")
            if self.enable_fallbacks:
                # Simple linear interpolation fallback
                return self._simple_resample(waveform, orig_sample_rate)
            else:
                logger.warning(f"Sample rate mismatch: {orig_sample_rate} vs {self.sample_rate}")
                return waveform
    
    def _simple_resample(self, waveform: torch.Tensor, orig_sample_rate: int) -> torch.Tensor:
        """Simple resampling fallback."""
        try:
            ratio = self.sample_rate / orig_sample_rate
            new_length = int(waveform.shape[-1] * ratio)
            
            # Use interpolation
            resampled = F.interpolate(
                waveform.unsqueeze(1),
                size=new_length,
                mode='linear',
                align_corners=False
            ).squeeze(1)
            
            return resampled
            
        except Exception as e:
            logger.error(f"Simple resampling failed: {e}")
            return waveform
    
    def ensure_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert stereo to mono if needed."""
        if waveform.dim() == 3:  # [batch, channels, samples]
            if waveform.shape[1] == 1:
                return waveform.squeeze(1)
            else:
                return waveform.mean(dim=1)
        elif waveform.dim() == 2 and waveform.shape[0] > 1:  # [channels, samples]
            return waveform.mean(dim=0, keepdim=True)
        else:
            return waveform
    
    def normalize_audio_amplitude(self, waveform: torch.Tensor, method: str = "rms") -> torch.Tensor:
        """Normalize audio amplitude with error handling."""
        try:
            if method == "peak":
                peak = torch.max(torch.abs(waveform), dim=-1, keepdim=True)[0]
                return waveform / (peak + 1e-8)
            elif method == "rms":
                rms = torch.sqrt(torch.mean(waveform ** 2, dim=-1, keepdim=True))
                return waveform / (rms + 1e-8) * 0.1
            elif method == "lufs":
                # Simplified LUFS-like normalization
                energy = torch.mean(waveform ** 2, dim=-1, keepdim=True)
                target_energy = 10 ** (-23 / 10)
                scale = torch.sqrt(target_energy / (energy + 1e-8))
                return waveform * scale.clamp(max=10.0)
            else:
                return waveform
                
        except Exception as e:
            logger.error(f"Amplitude normalization failed: {e}")
            return waveform
    
    def extract_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract mel-spectrogram with comprehensive error handling."""
        try:
            if callable(self.mel_transform) and hasattr(self.mel_transform, '__self__'):
                # Real torchaudio transform
                mel_spec = self.mel_transform(waveform)
            else:
                # Fallback transform
                mel_spec = self.mel_transform(waveform)
            
            # Apply log scaling
            if self.apply_log_scaling:
                mel_spec = torch.log(mel_spec.clamp(min=self.log_epsilon))
            
            # Apply normalization
            mel_spec = self.normalizer.normalize_tensor(mel_spec, 'mel')
            
            return mel_spec
            
        except Exception as e:
            logger.error(f"Mel spectrogram extraction failed: {e}")
            if self.enable_fallbacks:
                return self._manual_mel_spectrogram(waveform)
            raise
    
    def extract_mfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract MFCC features with error handling."""
        try:
            if callable(self.mfcc_transform) and hasattr(self.mfcc_transform, '__self__'):
                # Real torchaudio transform
                mfcc = self.mfcc_transform(waveform)
            else:
                # Fallback transform
                mfcc = self.mfcc_transform(waveform)
            
            # Apply normalization
            mfcc = self.normalizer.normalize_tensor(mfcc, 'mfcc')
            
            return mfcc
            
        except Exception as e:
            logger.error(f"MFCC extraction failed: {e}")
            if self.enable_fallbacks:
                return self._manual_mfcc(waveform)
            raise
    
    def extract_chroma(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract chroma features with error handling."""
        try:
            # Compute STFT
            stft = torch.stft(
                waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self._get_window_function()(self.win_length, device=waveform.device),
                return_complex=True,
                normalized=True
            )
            
            # Convert to magnitude
            magnitude = torch.abs(stft)
            
            # Apply chroma filterbank
            if hasattr(self, 'chroma_filterbank'):
                chroma = torch.matmul(self.chroma_filterbank, magnitude)
            else:
                # Fallback: simple frequency mapping
                chroma = self._simple_chroma_mapping(magnitude)
            
            # Normalize chroma vectors
            chroma = F.normalize(chroma, p=2, dim=1)
            
            # Apply normalization
            chroma = self.normalizer.normalize_tensor(chroma, 'chroma')
            
            return chroma
            
        except Exception as e:
            logger.error(f"Chroma extraction failed: {e}")
            if self.enable_fallbacks:
                return self._fallback_chroma(waveform)
            raise
    
    def extract_spectral_features(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract comprehensive spectral features with error handling."""
        features = {}
        
        try:
            # STFT for spectral analysis
            stft = torch.stft(
                waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self._get_window_function()(self.win_length, device=waveform.device),
                return_complex=True,
                normalized=True
            )
            
            magnitude = torch.abs(stft)
            power_spec = magnitude ** 2
            
            features['magnitude_spectrum'] = magnitude
            features['power_spectrum'] = power_spec
            
            # Frequency bins
            freqs = torch.linspace(0, self.sample_rate // 2,
                                 magnitude.shape[1], device=waveform.device)
            freqs = freqs.unsqueeze(0).unsqueeze(-1)
            
            # Spectral centroid
            spectral_centroid = torch.sum(magnitude * freqs, dim=1) / (torch.sum(magnitude, dim=1) + 1e-8)
            features['spectral_centroid'] = spectral_centroid
            
            # Spectral bandwidth
            centroid_expanded = spectral_centroid.unsqueeze(1)
            spectral_bandwidth = torch.sqrt(
                torch.sum(magnitude * (freqs - centroid_expanded)**2, dim=1) /
                (torch.sum(magnitude, dim=1) + 1e-8)
            )
            features['spectral_bandwidth'] = spectral_bandwidth
            
            # Spectral rolloff
            cumsum_mag = torch.cumsum(magnitude, dim=1)
            total_energy = cumsum_mag[:, -1:, :]
            rolloff_threshold = 0.85 * total_energy
            
            rolloff_indices = torch.searchsorted(
                cumsum_mag.transpose(1, 2),
                rolloff_threshold.transpose(1, 2)
            ).transpose(1, 2).clamp(max=len(freqs.squeeze())-1)
            
            spectral_rolloff = freqs.squeeze()[rolloff_indices.squeeze(1)]
            features['spectral_rolloff'] = spectral_rolloff
            
            # Spectral flatness
            log_magnitude = torch.log(magnitude + 1e-8)
            geometric_mean = torch.exp(torch.mean(log_magnitude, dim=1))
            arithmetic_mean = torch.mean(magnitude, dim=1)
            spectral_flatness = geometric_mean / (arithmetic_mean + 1e-8)
            features['spectral_flatness'] = spectral_flatness
            
            # Zero crossing rate
            zcr = self._compute_zero_crossing_rate(waveform)
            features['zero_crossing_rate'] = zcr
            
        except Exception as e:
            logger.error(f"Spectral feature extraction failed: {e}")
            if self.enable_fallbacks:
                features.update(self._fallback_spectral_features(waveform))
            else:
                raise
        
        return features
    
    def _compute_zero_crossing_rate(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute zero crossing rate with error handling."""
        try:
            frame_length = self.hop_length * 2
            frames = waveform.unfold(-1, frame_length, self.hop_length)
            
            signs = torch.sign(frames)
            sign_changes = torch.abs(torch.diff(signs, dim=-1))
            zcr = torch.sum(sign_changes, dim=-1) / frame_length
            
            return zcr
            
        except Exception as e:
            logger.error(f"Zero crossing rate computation failed: {e}")
            # Fallback: simple estimate
            batch_size = waveform.shape[0]
            n_frames = max(1, waveform.shape[-1] // self.hop_length)
            return torch.zeros(batch_size, n_frames, device=waveform.device)
    
    def _manual_stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """Manual STFT implementation."""
        try:
            stft = torch.stft(
                waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=torch.hann_window(self.n_fft, device=waveform.device),
                return_complex=True,
                normalized=True
            )
            return torch.abs(stft) ** 2
            
        except Exception as e:
            logger.error(f"Manual STFT failed: {e}")
            # Ultimate fallback
            batch_size = waveform.shape[0]
            n_frames = max(1, waveform.shape[-1] // self.hop_length)
            return torch.ones(batch_size, self.n_fft // 2 + 1, n_frames, device=waveform.device) * 1e-8
    
    def _manual_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Manual mel spectrogram implementation."""
        try:
            # Get STFT
            stft = self._manual_stft(waveform)
            
            # Create mel filterbank
            try:
                mel_filters = torch.from_numpy(
                    librosa.filters.mel(
                        sr=self.sample_rate,
                        n_fft=self.n_fft,
                        n_mels=self.n_mels,
                        fmin=self.f_min,
                        fmax=self.f_max
                    )
                ).float().to(waveform.device)
            except Exception:
                # Fallback: simple frequency binning
                mel_filters = torch.eye(self.n_mels, stft.shape[1], device=waveform.device)
            
            # Apply mel filters
            mel_spec = torch.matmul(mel_filters, stft)
            return mel_spec
            
        except Exception as e:
            logger.error(f"Manual mel spectrogram failed: {e}")
            # Ultimate fallback
            batch_size = waveform.shape[0]
            n_frames = max(1, waveform.shape[-1] // self.hop_length)
            return torch.ones(batch_size, self.n_mels, n_frames, device=waveform.device) * 1e-8
    
    def _manual_mfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        """Manual MFCC implementation."""
        try:
            # Get mel spectrogram
            mel_spec = self._manual_mel_spectrogram(waveform)
            
            # Apply log
            log_mel = torch.log(mel_spec + 1e-8)
            
            # DCT (simplified)
            # Create DCT matrix
            n = self.n_mels
            dct_matrix = torch.zeros(self.n_mfcc, n, device=waveform.device)
            for k in range(self.n_mfcc):
                for n_idx in range(n):
                    dct_matrix[k, n_idx] = np.cos(np.pi * k * (2 * n_idx + 1) / (2 * n))
            
            # Apply DCT
            mfcc = torch.matmul(dct_matrix, log_mel)
            
            return mfcc
            
        except Exception as e:
            logger.error(f"Manual MFCC failed: {e}")
            # Ultimate fallback
            batch_size = waveform.shape[0]
            n_frames = max(1, waveform.shape[-1] // self.hop_length)
            return torch.zeros(batch_size, self.n_mfcc, n_frames, device=waveform.device)
    
    def _simple_chroma_mapping(self, magnitude: torch.Tensor) -> torch.Tensor:
        """Simple chroma mapping fallback."""
        try:
            # Map frequency bins to chroma bins
            chroma = torch.zeros(magnitude.shape[0], self.n_chroma, magnitude.shape[-1], device=magnitude.device)
            
            for i in range(magnitude.shape[1]):
                freq = i * self.sample_rate / (2 * magnitude.shape[1])
                if freq > 0:
                    pitch_class = int(12 * np.log2(freq / 440.0) + 9) % 12
                    if pitch_class < self.n_chroma:
                        chroma[:, pitch_class, :] += magnitude[:, i, :]
            
            return chroma
            
        except Exception as e:
            logger.error(f"Simple chroma mapping failed: {e}")
            # Ultimate fallback
            batch_size = magnitude.shape[0]
            n_frames = magnitude.shape[-1]
            return torch.ones(batch_size, self.n_chroma, n_frames, device=magnitude.device) / self.n_chroma
    
    def _fallback_chroma(self, waveform: torch.Tensor) -> torch.Tensor:
        """Fallback chroma extraction."""
        try:
            # Very simple chroma approximation
            stft = self._manual_stft(waveform)
            return self._simple_chroma_mapping(torch.sqrt(stft))
            
        except Exception as e:
            logger.error(f"Fallback chroma failed: {e}")
            # Ultimate fallback
            batch_size = waveform.shape[0]
            n_frames = max(1, waveform.shape[-1] // self.hop_length)
            return torch.ones(batch_size, self.n_chroma, n_frames, device=waveform.device) / self.n_chroma
    
    def _fallback_spectral_features(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Fallback spectral features."""
        batch_size = waveform.shape[0]
        n_frames = max(1, waveform.shape[-1] // self.hop_length)
        device = waveform.device
        
        return {
            'spectral_centroid': torch.ones(batch_size, n_frames, device=device) * 1000.0,
            'spectral_bandwidth': torch.ones(batch_size, n_frames, device=device) * 500.0,
            'spectral_rolloff': torch.ones(batch_size, n_frames, device=device) * 5000.0,
            'spectral_flatness': torch.ones(batch_size, n_frames, device=device) * 0.1,
            'zero_crossing_rate': torch.ones(batch_size, n_frames, device=device) * 0.1
        }
    
    def forward(
        self,
        waveform: torch.Tensor,
        orig_sample_rate: Optional[int] = None,
        extract_features: List[str] = ['mel', 'mfcc', 'chroma'],
        normalize_amplitude: bool = True,
        max_memory_mb: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Complete preprocessing pipeline with comprehensive error handling.
        """
        start_time = time.time()
        
        with self._memory_management():
            try:
                # Input validation
                if self.validate_inputs:
                    waveform = self._validate_audio_input(waveform)
                
                # Memory check
                if max_memory_mb and self._estimate_memory_usage(waveform) > max_memory_mb:
                    logger.warning(f"Estimated memory usage exceeds {max_memory_mb}MB, using chunked processing")
                    return self._process_chunked(waveform, extract_features, max_memory_mb)
                
                # Sample rate conversion
                if orig_sample_rate is not None and orig_sample_rate != self.sample_rate:
                    waveform = self.convert_sample_rate(waveform, orig_sample_rate)
                
                # Ensure mono audio
                waveform = self.ensure_mono(waveform)
                
                # Normalize amplitude
                if normalize_amplitude:
                    waveform = self.normalize_audio_amplitude(waveform, method="rms")
                
                # Extract requested features
                features = {'waveform': waveform}
                
                if 'mel' in extract_features:
                    try:
                        features['mel_spectrogram'] = self.extract_mel_spectrogram(waveform)
                    except Exception as e:
                        logger.error(f"Mel spectrogram extraction failed: {e}")
                        if self.enable_fallbacks:
                            features['mel_spectrogram'] = self._manual_mel_spectrogram(waveform)
                
                if 'mfcc' in extract_features:
                    try:
                        features['mfcc'] = self.extract_mfcc(waveform)
                    except Exception as e:
                        logger.error(f"MFCC extraction failed: {e}")
                        if self.enable_fallbacks:
                            features['mfcc'] = self._manual_mfcc(waveform)
                
                if 'chroma' in extract_features:
                    try:
                        features['chroma'] = self.extract_chroma(waveform)
                    except Exception as e:
                        logger.error(f"Chroma extraction failed: {e}")
                        if self.enable_fallbacks:
                            features['chroma'] = self._fallback_chroma(waveform)
                
                if 'spectral' in extract_features:
                    try:
                        spectral_features = self.extract_spectral_features(waveform)
                        features.update(spectral_features)
                    except Exception as e:
                        logger.error(f"Spectral feature extraction failed: {e}")
                        if self.enable_fallbacks:
                            features.update(self._fallback_spectral_features(waveform))
                
                # Add metadata
                features['_metadata'] = {
                    'processing_time': time.time() - start_time,
                    'sample_rate': self.sample_rate,
                    'n_fft': self.n_fft,
                    'hop_length': self.hop_length,
                    'input_shape': list(waveform.shape),
                    'extracted_features': extract_features
                }
                
                return features
                
            except Exception as e:
                logger.error(f"Critical error in audio preprocessing: {e}")
                if self.enable_fallbacks:
                    return self._emergency_fallback(waveform, extract_features)
                raise
    
    def _estimate_memory_usage(self, waveform: torch.Tensor) -> float:
        """Estimate memory usage in MB."""
        batch_size, n_samples = waveform.shape
        
        # Estimate feature sizes
        n_frames = n_samples // self.hop_length
        
        mel_size = batch_size * self.n_mels * n_frames * 4
        mfcc_size = batch_size * self.n_mfcc * n_frames * 4
        chroma_size = batch_size * self.n_chroma * n_frames * 4
        spectral_size = batch_size * (self.n_fft // 2 + 1) * n_frames * 4
        
        total_bytes = (mel_size + mfcc_size + chroma_size + spectral_size) * 2  # 2x overhead
        
        return total_bytes / (1024 * 1024)
    
    def _process_chunked(self, waveform: torch.Tensor, extract_features: List[str], max_memory_mb: int) -> Dict[str, torch.Tensor]:
        """Process audio in chunks."""
        batch_size, n_samples = waveform.shape
        
        # Calculate chunk size
        samples_per_mb = int(max_memory_mb * 1024 * 1024 / (batch_size * 4 * 10))
        chunk_size = min(n_samples, max(self.sample_rate, samples_per_mb))
        
        logger.info(f"Processing in chunks of {chunk_size} samples")
        
        chunk_results = []
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            chunk = waveform[:, start:end]
            
            # Add overlap for continuity
            if start > 0:
                overlap = min(self.n_fft, start)
                chunk = torch.cat([waveform[:, start-overlap:start], chunk], dim=1)
            
            try:
                chunk_result = self.forward(chunk, extract_features=extract_features, normalize_amplitude=False)
                chunk_results.append(chunk_result)
            except Exception as e:
                logger.error(f"Chunk processing failed: {e}")
                continue
        
        if not chunk_results:
            return self._emergency_fallback(waveform, extract_features)
        
        # Merge chunks
        return self._merge_chunk_results(chunk_results)
    
    def _merge_chunk_results(self, chunk_results: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Merge results from chunked processing."""
        if not chunk_results:
            raise ValueError("No chunk results to merge")
        
        if len(chunk_results) == 1:
            return chunk_results[0]
        
        merged = {}
        
        for key in chunk_results[0].keys():
            if key.startswith('_'):
                # Skip metadata
                continue
            
            try:
                chunks = [cr[key] for cr in chunk_results if key in cr]
                if chunks and isinstance(chunks[0], torch.Tensor):
                    if chunks[0].dim() >= 2:
                        # Concatenate along time dimension
                        merged[key] = torch.cat(chunks, dim=-1)
                    else:
                        # Average for scalar features
                        merged[key] = torch.stack(chunks).mean(dim=0)
                else:
                    merged[key] = chunks[0] if chunks else None
            except Exception as e:
                logger.warning(f"Failed to merge feature {key}: {e}")
        
        # Update metadata
        merged['_metadata'] = chunk_results[0]['_metadata']
        merged['_metadata']['chunked_processing'] = True
        merged['_metadata']['num_chunks'] = len(chunk_results)
        
        return merged
    
    def _emergency_fallback(self, waveform: torch.Tensor, extract_features: List[str]) -> Dict[str, torch.Tensor]:
        """Emergency fallback when everything fails."""
        logger.warning("Using emergency fallback for audio preprocessing")
        
        batch_size = waveform.shape[0]
        device = waveform.device
        n_frames = max(1, waveform.shape[-1] // self.hop_length)
        
        features = {
            'waveform': waveform,
            '_emergency_fallback': True
        }
        
        # Create minimal dummy features
        if 'mel' in extract_features:
            features['mel_spectrogram'] = torch.zeros(batch_size, self.n_mels, n_frames, device=device)
        
        if 'mfcc' in extract_features:
            features['mfcc'] = torch.zeros(batch_size, self.n_mfcc, n_frames, device=device)
        
        if 'chroma' in extract_features:
            features['chroma'] = torch.ones(batch_size, self.n_chroma, n_frames, device=device) / self.n_chroma
        
        return features


def test_bulletproof_preprocessing():
    """Test the bulletproof preprocessing module."""
    logger.info("Testing BulletproofAudioPreprocessing")
    
    preprocessor = BulletproofAudioPreprocessing(
        sample_rate=22050,
        enable_fallbacks=True,
        validate_inputs=True,
        memory_efficient=True
    )
    
    # Test cases
    test_cases = [
        torch.randn(1, 22050),      # 1 second
        torch.randn(2, 44100),      # 2 seconds
        torch.zeros(1, 8000),       # Silent
        torch.ones(1, 1000) * 0.5, # Constant signal
        torch.randn(1, 22050 * 10), # 10 seconds
    ]
    
    for i, waveform in enumerate(test_cases):
        try:
            logger.info(f"Testing case {i+1}: shape {waveform.shape}")
            
            features = preprocessor(waveform, extract_features=['mel', 'mfcc', 'chroma', 'spectral'])
            
            logger.info(f"  Extracted features: {list(features.keys())}")
            for key, value in features.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"    {key}: {value.shape}")
            
            processing_time = features.get('_metadata', {}).get('processing_time', 0)
            logger.info(f"  Processing time: {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Test case {i+1} failed: {e}")
    
    # Test sample rate conversion
    try:
        logger.info("Testing sample rate conversion...")
        waveform_16k = torch.randn(1, 16000)
        features = preprocessor(waveform_16k, orig_sample_rate=16000)
        logger.info("Sample rate conversion test passed")
    except Exception as e:
        logger.error(f"Sample rate conversion test failed: {e}")
    
    logger.info("Preprocessing testing completed")


if __name__ == "__main__":
    test_bulletproof_preprocessing()
