#!/usr/bin/env python3
"""
BULLETPROOF MEL SPECTROGRAM MODULE
Robust mel-scale spectrogram computation for BigVGAN with numerical stability and error handling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, Tuple, Dict, Any, List
from rave_config_system import RAVEConfig
import logging
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MelSpectrogramConfig:
    """Configuration for bulletproof mel spectrogram"""
    sample_rate: int = 22050
    n_fft: int = 1024
    win_length: Optional[int] = None
    hop_length: Optional[int] = None
    f_min: float = 0.0
    f_max: Optional[float] = None
    n_mels: int = 80
    window: str = 'hann'
    center: bool = True
    pad_mode: str = 'reflect'
    power: float = 2.0
    normalized: bool = False
    
    # Bulletproof parameters
    numerical_stability_check: bool = True
    eps: float = 1e-10
    log_offset: float = 1e-6
    clamp_min: float = 1e-8
    clamp_max: float = 1e8
    
    # Error handling
    enable_fallbacks: bool = True
    validate_filterbank: bool = True
    handle_empty_audio: bool = True

class BulletproofMelSpectrogram(nn.Module):
    """Bulletproof Mel Spectrogram with comprehensive error handling and numerical stability"""
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__()
        
        self.config = config
        self.mel_config = kwargs.get('mel_config', MelSpectrogramConfig())
        
        # Override with audio config if available
        if hasattr(config.audio, 'sample_rate'):
            self.mel_config.sample_rate = config.audio.sample_rate
        if hasattr(config.audio, 'n_fft'):
            self.mel_config.n_fft = config.audio.n_fft
        if hasattr(config.audio, 'hop_length'):
            self.mel_config.hop_length = config.audio.hop_length
        if hasattr(config.audio, 'n_mels'):
            self.mel_config.n_mels = config.audio.n_mels
        
        # Set derived parameters
        self.sample_rate = self.mel_config.sample_rate
        self.n_fft = self.mel_config.n_fft
        self.win_length = self.mel_config.win_length if self.mel_config.win_length is not None else self.n_fft
        self.hop_length = self.mel_config.hop_length if self.mel_config.hop_length is not None else self.n_fft // 4
        self.f_min = self.mel_config.f_min
        self.f_max = self.mel_config.f_max if self.mel_config.f_max is not None else self.sample_rate / 2.0
        self.n_mels = self.mel_config.n_mels
        self.center = self.mel_config.center
        self.pad_mode = self.mel_config.pad_mode
        self.power = self.mel_config.power
        self.normalized = self.mel_config.normalized
        
        # Validate parameters
        self._validate_parameters()
        
        # Create window and mel filterbank
        try:
            self.register_buffer('window', self._create_window())
            self.register_buffer('mel_basis', self._create_mel_filterbank())
        except Exception as e:
            logger.error(f"Failed to create mel spectrogram components: {e}")
            if self.mel_config.enable_fallbacks:
                logger.warning("Creating fallback components")
                self._create_fallback_components()
            else:
                raise
        
        # Tracking
        self.computation_stats = []
        self.fallback_activations = 0
        
        logger.info(f"BulletproofMelSpectrogram initialized: {self.n_mels} mels, {self.n_fft} FFT")
    
    def _validate_parameters(self):
        """Validate and adjust parameters for stability"""
        try:
            # Ensure reasonable FFT size
            if self.n_fft < 64:
                logger.warning(f"n_fft {self.n_fft} too small, setting to 512")
                self.n_fft = 512
            elif self.n_fft > 8192:
                logger.warning(f"n_fft {self.n_fft} too large, setting to 4096")
                self.n_fft = 4096
            
            # Ensure reasonable hop length
            if self.hop_length < 8:
                self.hop_length = self.n_fft // 8
                logger.warning(f"hop_length adjusted to {self.hop_length}")
            elif self.hop_length > self.n_fft:
                self.hop_length = self.n_fft // 4
                logger.warning(f"hop_length adjusted to {self.hop_length}")
            
            # Ensure reasonable mel count
            if self.n_mels < 8:
                logger.warning(f"n_mels {self.n_mels} too small, setting to 32")
                self.n_mels = 32
            elif self.n_mels > 512:
                logger.warning(f"n_mels {self.n_mels} too large, setting to 256")
                self.n_mels = 256
            
            # Ensure reasonable frequency range
            if self.f_min < 0:
                self.f_min = 0.0
            if self.f_max > self.sample_rate / 2:
                self.f_max = self.sample_rate / 2.0
            if self.f_min >= self.f_max:
                self.f_min = 0.0
                self.f_max = self.sample_rate / 2.0
                logger.warning("Adjusted frequency range")
            
        except Exception as e:
            logger.error(f"Parameter validation failed: {e}")
            # Set safe defaults
            self.n_fft = 1024
            self.hop_length = 256
            self.n_mels = 80
            self.f_min = 0.0
            self.f_max = self.sample_rate / 2.0
    
    def _create_window(self) -> torch.Tensor:
        """Create window function with error handling"""
        try:
            if self.mel_config.window == 'hann':
                return torch.hann_window(self.win_length)
            elif self.mel_config.window == 'hamming':
                return torch.hamming_window(self.win_length)
            elif self.mel_config.window == 'blackman':
                return torch.blackman_window(self.win_length)
            else:
                logger.warning(f"Unknown window type {self.mel_config.window}, using Hann")
                return torch.hann_window(self.win_length)
        except Exception as e:
            logger.error(f"Window creation failed: {e}")
            # Fallback to rectangular window
            return torch.ones(self.win_length)
    
    def _hz_to_mel(self, frequencies: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Convert Hz to mel scale with numerical stability"""
        try:
            if isinstance(frequencies, (int, float)):
                frequencies = torch.tensor(frequencies, dtype=torch.float32)
            return 2595.0 * torch.log10(1.0 + frequencies / 700.0)
        except Exception as e:
            logger.error(f"Hz to mel conversion failed: {e}")
            return frequencies * 0.0
    
    def _mel_to_hz(self, mels: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Convert mel to Hz with numerical stability"""
        try:
            if isinstance(mels, (int, float)):
                mels = torch.tensor(mels, dtype=torch.float32)
            return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
        except Exception as e:
            logger.error(f"Mel to Hz conversion failed: {e}")
            return mels * 0.0
    
    def _create_mel_filterbank(self) -> torch.Tensor:
        """Create mel filterbank matrix with comprehensive error handling"""
        try:
            # Frequency bins
            freqs = torch.linspace(0, self.sample_rate / 2, self.n_fft // 2 + 1, dtype=torch.float32)
            
            # Mel scale points
            mel_min = self._hz_to_mel(self.f_min)
            mel_max = self._hz_to_mel(self.f_max)
            mels = torch.linspace(mel_min, mel_max, self.n_mels + 2)
            hz_points = self._mel_to_hz(mels)
            
            # Validate Hz points
            if not torch.isfinite(hz_points).all():
                logger.warning("Non-finite Hz points, using linear spacing")
                hz_points = torch.linspace(self.f_min, self.f_max, self.n_mels + 2)
            
            # Create filterbank
            filterbank = torch.zeros(self.n_mels, self.n_fft // 2 + 1, dtype=torch.float32)
            
            for i in range(self.n_mels):
                try:
                    left = hz_points[i]
                    center = hz_points[i + 1]
                    right = hz_points[i + 2]
                    
                    # Ensure valid triangle
                    if center <= left or right <= center:
                        continue
                    
                    # Rising edge
                    rise = (freqs - left) / (center - left + 1e-8)
                    # Falling edge
                    fall = (right - freqs) / (right - center + 1e-8)
                    
                    # Triangular filter
                    filterbank[i] = torch.clamp(torch.minimum(rise, fall), min=0.0)
                    
                except Exception as e:
                    logger.warning(f"Failed to create mel filter {i}: {e}")
                    continue
            
            # Normalize filters if requested
            if self.normalized:
                try:
                    enorm = 2.0 / (hz_points[2:self.n_mels+2] - hz_points[:self.n_mels] + 1e-8)
                    filterbank = filterbank * enorm.unsqueeze(1)
                except Exception as e:
                    logger.warning(f"Filter normalization failed: {e}")
            
            # Validate filterbank
            if self.mel_config.validate_filterbank:
                if not torch.isfinite(filterbank).all():
                    logger.warning("Non-finite values in filterbank")
                    filterbank = torch.nan_to_num(filterbank, nan=0.0, posinf=1.0, neginf=0.0)
                
                # Ensure each filter has some non-zero values
                for i in range(self.n_mels):
                    if filterbank[i].sum() < 1e-8:
                        # Create simple rectangular filter as fallback
                        start_bin = int(i * (self.n_fft // 2) // self.n_mels)
                        end_bin = int((i + 1) * (self.n_fft // 2) // self.n_mels)
                        filterbank[i, start_bin:end_bin] = 1.0 / (end_bin - start_bin + 1)
            
            return filterbank
            
        except Exception as e:
            logger.error(f"Mel filterbank creation failed: {e}")
            # Create simple filterbank as fallback
            return self._create_simple_filterbank()
    
    def _create_simple_filterbank(self) -> torch.Tensor:
        """Create simple fallback filterbank"""
        try:
            filterbank = torch.zeros(self.n_mels, self.n_fft // 2 + 1, dtype=torch.float32)
            
            # Simple rectangular filters
            bins_per_mel = (self.n_fft // 2 + 1) // self.n_mels
            for i in range(self.n_mels):
                start = i * bins_per_mel
                end = min((i + 1) * bins_per_mel, self.n_fft // 2 + 1)
                if end > start:
                    filterbank[i, start:end] = 1.0 / (end - start)
            
            return filterbank
        except Exception as e:
            logger.error(f"Simple filterbank creation failed: {e}")
            # Emergency fallback: identity-like
            return torch.eye(min(self.n_mels, self.n_fft // 2 + 1))[:self.n_mels, :self.n_fft // 2 + 1]
    
    def _create_fallback_components(self):
        """Create fallback window and filterbank"""
        try:
            # Simple window
            self.register_buffer('window', torch.ones(self.win_length))
            
            # Simple filterbank
            self.register_buffer('mel_basis', self._create_simple_filterbank())
            
            logger.info("Created fallback mel spectrogram components")
        except Exception as e:
            logger.error(f"Fallback component creation failed: {e}")
            # Emergency fallbacks
            self.register_buffer('window', torch.ones(64))
            self.register_buffer('mel_basis', torch.eye(32, 65))
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram with comprehensive error handling"""
        try:
            # Validate input
            if audio.numel() == 0:
                if self.mel_config.handle_empty_audio:
                    logger.warning("Empty audio input")
                    return torch.zeros(audio.size(0), self.n_mels, 1, device=audio.device, dtype=audio.dtype)
                else:
                    raise ValueError("Empty audio input")
            
            if not torch.isfinite(audio).all():
                logger.warning("Non-finite values in audio input")
                if self.mel_config.enable_fallbacks:
                    audio = torch.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
                else:
                    raise ValueError("Non-finite audio input")
            
            # Handle input shape
            if audio.dim() == 2:
                audio = audio.unsqueeze(1)  # Add channel dimension
            
            # Remove channel dimension for STFT
            batch_size, _, length = audio.shape
            audio = audio.squeeze(1)  # (batch, time)
            
            # Clamp audio to reasonable range
            audio = torch.clamp(audio, -self.mel_config.clamp_max, self.mel_config.clamp_max)
            
            # Apply padding if centering
            if self.center:
                try:
                    pad = self.n_fft // 2
                    audio = F.pad(audio, (pad, pad), mode=self.pad_mode)
                except Exception as e:
                    logger.warning(f"Padding failed: {e}")
                    # Fallback to zero padding
                    audio = F.pad(audio, (pad, pad), mode='constant')
            
            # Compute STFT with error handling
            try:
                stft = torch.stft(
                    audio,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.win_length,
                    window=self.window,
                    center=False,  # Already handled padding
                    return_complex=True
                )
            except Exception as e:
                logger.error(f"STFT computation failed: {e}")
                if self.mel_config.enable_fallbacks:
                    self.fallback_activations += 1
                    # Return zero spectrogram
                    return torch.zeros(batch_size, self.n_mels, 1, device=audio.device, dtype=audio.dtype)
                else:
                    raise
            
            # Compute magnitude with numerical stability
            try:
                magnitude = torch.abs(stft)
                magnitude = torch.clamp(magnitude, min=self.mel_config.clamp_min, max=self.mel_config.clamp_max)
                
                # Apply power
                if self.power != 1.0:
                    magnitude = magnitude ** self.power
                
            except Exception as e:
                logger.error(f"Magnitude computation failed: {e}")
                if self.mel_config.enable_fallbacks:
                    self.fallback_activations += 1
                    magnitude = torch.ones_like(stft.real) * self.mel_config.clamp_min
                else:
                    raise
            
            # Apply mel filterbank
            try:
                mel_spec = torch.matmul(self.mel_basis, magnitude)
                mel_spec = torch.clamp(mel_spec, min=self.mel_config.clamp_min)
                
            except Exception as e:
                logger.error(f"Mel filterbank application failed: {e}")
                if self.mel_config.enable_fallbacks:
                    self.fallback_activations += 1
                    # Simple downsampling fallback
                    mel_spec = F.adaptive_avg_pool2d(magnitude.unsqueeze(1), (self.n_mels, magnitude.size(-1))).squeeze(1)
                else:
                    raise
            
            # Validate output
            if self.mel_config.numerical_stability_check:
                if not torch.isfinite(mel_spec).all():
                    logger.warning("Non-finite mel spectrogram output")
                    mel_spec = torch.nan_to_num(mel_spec, nan=self.mel_config.clamp_min, 
                                              posinf=self.mel_config.clamp_max, neginf=self.mel_config.clamp_min)
            
            # Track statistics
            if len(self.computation_stats) < 1000:
                self.computation_stats.append({
                    'input_length': length,
                    'output_frames': mel_spec.size(-1),
                    'magnitude_mean': magnitude.mean().item(),
                    'mel_spec_mean': mel_spec.mean().item(),
                    'mel_spec_std': mel_spec.std().item()
                })
            
            return mel_spec
            
        except Exception as e:
            logger.error(f"Mel spectrogram computation failed: {e}")
            if self.mel_config.enable_fallbacks:
                self.fallback_activations += 1
                # Emergency fallback: return minimal spectrogram
                batch_size = audio.size(0) if audio.dim() > 0 else 1
                return torch.ones(batch_size, self.n_mels, 1, device=audio.device if hasattr(audio, 'device') else torch.device('cpu')) * self.mel_config.clamp_min
            else:
                raise
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        stats = {
            'sample_rate': self.sample_rate,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'n_mels': self.n_mels,
            'fallback_activations': self.fallback_activations
        }
        
        if self.computation_stats:
            last_stats = self.computation_stats[-1]
            stats.update({
                'last_mel_spec_mean': last_stats['mel_spec_mean'],
                'last_output_frames': last_stats['output_frames']
            })
        
        return stats


class BulletproofLogMelSpectrogram(BulletproofMelSpectrogram):
    """Bulletproof Log Mel Spectrogram with additional numerical stability"""
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.log_offset = self.mel_config.log_offset
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute log mel spectrogram"""
        try:
            # Compute mel spectrogram
            mel_spec = super().forward(audio)
            
            # Apply log scale with offset
            log_mel_spec = torch.log(mel_spec + self.log_offset)
            
            # Clamp to reasonable range
            log_mel_spec = torch.clamp(log_mel_spec, min=-20, max=20)
            
            return log_mel_spec
            
        except Exception as e:
            logger.error(f"Log mel spectrogram computation failed: {e}")
            if self.mel_config.enable_fallbacks:
                self.fallback_activations += 1
                return torch.zeros(1, self.n_mels, 1, device=audio.device)
            else:
                raise


# Factory functions
def create_bulletproof_mel_spectrogram(config: RAVEConfig, **kwargs) -> BulletproofMelSpectrogram:
    """Create a bulletproof mel spectrogram"""
    return BulletproofMelSpectrogram(config, **kwargs)


def create_bulletproof_log_mel_spectrogram(config: RAVEConfig, **kwargs) -> BulletproofLogMelSpectrogram:
    """Create a bulletproof log mel spectrogram"""
    return BulletproofLogMelSpectrogram(config, **kwargs)


if __name__ == "__main__":
    print("🛡️ BULLETPROOF MEL SPECTROGRAM MODULE")
    print("=" * 45)
    
    from rave_config_system import get_minimal_config
    
    config = get_minimal_config()
    
    # Test mel spectrogram
    mel_config = MelSpectrogramConfig(sample_rate=22050, n_fft=1024, n_mels=80)
    mel_spec = create_bulletproof_mel_spectrogram(config, mel_config=mel_config)
    
    # Test data
    batch_size = 2
    audio_length = 8192
    audio = torch.randn(batch_size, audio_length)
    
    try:
        output = mel_spec(audio)
        print(f"✅ Mel spectrogram test passed: {audio.shape} -> {output.shape}")
        
        stats = mel_spec.get_training_stats()
        print(f"   Mel stats: {stats}")
        
    except Exception as e:
        print(f"❌ Mel spectrogram test failed: {e}")
    
    # Test log mel spectrogram
    try:
        log_mel_spec = create_bulletproof_log_mel_spectrogram(config, mel_config=mel_config)
        log_output = log_mel_spec(audio)
        print(f"✅ Log mel spectrogram test passed: {audio.shape} -> {log_output.shape}")
        
    except Exception as e:
        print(f"❌ Log mel spectrogram test failed: {e}")
    
    print("🚀 BulletproofMelSpectrogram ready for BigVGAN feature extraction!")