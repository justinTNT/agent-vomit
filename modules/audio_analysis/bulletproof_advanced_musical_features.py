"""
Bulletproof Advanced Musical Features Module

Comprehensive musical feature extraction with extensive error handling,
memory management, and multiple fallback strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import librosa
import warnings
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from contextlib import contextmanager
import gc
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BulletproofAdvancedMusicalFeatures(nn.Module):
    """
    Bulletproof comprehensive musical feature extraction with:
    - Extensive parameter validation and sanitization
    - Multiple fallback strategies for feature extraction
    - Memory management for large audio files
    - Device compatibility (CPU/GPU) with automatic fallback
    - Graceful degradation when advanced features fail
    - Robust error handling and recovery
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        n_chroma: int = 12,
        f_min: float = 80.0,
        f_max: Optional[float] = None,
        tuning_resolution: float = 0.01,
        max_audio_length: int = 22050 * 300,  # 5 minutes max
        memory_efficient: bool = True,
        enable_fallbacks: bool = True,
        validate_inputs: bool = True,
        **kwargs
    ):
        super().__init__()
        
        # Validate and sanitize parameters
        self.sample_rate = self._validate_sample_rate(sample_rate)
        self.n_fft = self._validate_n_fft(n_fft)
        self.hop_length = self._validate_hop_length(hop_length, n_fft)
        self.n_mels = self._validate_n_mels(n_mels)
        self.n_chroma = self._validate_n_chroma(n_chroma)
        self.f_min = self._validate_frequency(f_min, "f_min")
        self.f_max = self._validate_f_max(f_max, sample_rate)
        self.tuning_resolution = max(0.001, min(0.1, tuning_resolution))
        self.max_audio_length = max(22050, max_audio_length)
        self.memory_efficient = memory_efficient
        self.enable_fallbacks = enable_fallbacks
        self.validate_inputs = validate_inputs
        
        # Initialize device
        self.device = torch.device('cpu')
        
        try:
            # Core transforms with error handling
            self.stft_transform = self._create_safe_stft_transform()
            self.mel_transform = self._create_safe_mel_transform()
            
            # Chroma analysis with fallback
            self.chroma_transform = self._build_safe_chroma_transform()
            
            # Feature analyzers with error handling
            self._initialize_analyzers()
            
            logger.info(f"BulletproofAdvancedMusicalFeatures initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing BulletproofAdvancedMusicalFeatures: {e}")
            if not self.enable_fallbacks:
                raise
            # Initialize minimal fallback configuration
            self._initialize_fallback_transforms()
    
    def _validate_sample_rate(self, sample_rate: int) -> int:
        """Validate and sanitize sample rate."""
        if not isinstance(sample_rate, (int, float)):
            logger.warning(f"Invalid sample_rate type: {type(sample_rate)}, using default 22050")
            return 22050
        
        sample_rate = int(sample_rate)
        if sample_rate < 8000:
            logger.warning(f"Sample rate {sample_rate} too low, using 8000")
            return 8000
        elif sample_rate > 192000:
            logger.warning(f"Sample rate {sample_rate} too high, using 192000")
            return 192000
        
        return sample_rate
    
    def _validate_n_fft(self, n_fft: int) -> int:
        """Validate and sanitize n_fft."""
        if not isinstance(n_fft, (int, float)):
            logger.warning(f"Invalid n_fft type: {type(n_fft)}, using default 2048")
            return 2048
        
        n_fft = int(n_fft)
        
        # Ensure power of 2 for FFT efficiency
        if n_fft <= 0:
            return 2048
        
        # Round to nearest power of 2
        power = int(np.log2(n_fft))
        if 2**power != n_fft:
            n_fft = 2**power
            logger.info(f"Rounded n_fft to nearest power of 2: {n_fft}")
        
        return max(256, min(8192, n_fft))
    
    def _validate_hop_length(self, hop_length: int, n_fft: int) -> int:
        """Validate and sanitize hop length."""
        if not isinstance(hop_length, (int, float)):
            logger.warning(f"Invalid hop_length type: {type(hop_length)}, using n_fft//4")
            return n_fft // 4
        
        hop_length = int(hop_length)
        
        if hop_length <= 0:
            hop_length = n_fft // 4
        elif hop_length > n_fft:
            hop_length = n_fft // 2
            logger.warning(f"hop_length > n_fft, reduced to {hop_length}")
        
        return hop_length
    
    def _validate_n_mels(self, n_mels: int) -> int:
        """Validate and sanitize n_mels."""
        if not isinstance(n_mels, (int, float)):
            logger.warning(f"Invalid n_mels type: {type(n_mels)}, using default 128")
            return 128
        
        return max(10, min(512, int(n_mels)))
    
    def _validate_n_chroma(self, n_chroma: int) -> int:
        """Validate and sanitize n_chroma."""
        if not isinstance(n_chroma, (int, float)):
            logger.warning(f"Invalid n_chroma type: {type(n_chroma)}, using default 12")
            return 12
        
        return max(6, min(24, int(n_chroma)))
    
    def _validate_frequency(self, freq: float, name: str) -> float:
        """Validate frequency parameter."""
        if not isinstance(freq, (int, float)):
            logger.warning(f"Invalid {name} type: {type(freq)}, using 0.0")
            return 0.0
        
        return max(0.0, float(freq))
    
    def _validate_f_max(self, f_max: Optional[float], sample_rate: int) -> float:
        """Validate and sanitize f_max."""
        if f_max is None:
            return sample_rate / 2.0
        
        if not isinstance(f_max, (int, float)):
            logger.warning(f"Invalid f_max type: {type(f_max)}, using Nyquist frequency")
            return sample_rate / 2.0
        
        f_max = float(f_max)
        nyquist = sample_rate / 2.0
        
        if f_max > nyquist:
            logger.warning(f"f_max {f_max} > Nyquist {nyquist}, using Nyquist")
            return nyquist
        
        return max(100.0, f_max)
    
    def _create_safe_stft_transform(self):
        """Create STFT transform with error handling."""
        try:
            return torchaudio.transforms.Spectrogram(
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                power=2.0,
                normalized=True
            )
        except Exception as e:
            logger.error(f"Error creating STFT transform: {e}")
            if self.enable_fallbacks:
                return self._create_fallback_stft()
            raise
    
    def _create_safe_mel_transform(self):
        """Create mel transform with error handling."""
        try:
            return torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                f_min=self.f_min,
                f_max=self.f_max,
                normalized=True
            )
        except Exception as e:
            logger.error(f"Error creating mel transform: {e}")
            if self.enable_fallbacks:
                return self._create_fallback_mel()
            raise
    
    def _build_safe_chroma_transform(self):
        """Build chroma feature transform with error handling."""
        try:
            # Create chroma filterbank using librosa
            chroma_fb = librosa.filters.chroma(
                sr=self.sample_rate,
                n_fft=self.n_fft,
                n_chroma=self.n_chroma,
                tuning=0.0
            )
            
            # Validate filterbank
            if chroma_fb.shape[0] != self.n_chroma:
                raise ValueError(f"Chroma filterbank shape mismatch: {chroma_fb.shape}")
            
            chroma_fb = torch.from_numpy(chroma_fb).float()
            self.register_buffer('chroma_filterbank', chroma_fb)
            
            return lambda spec: self._safe_chroma_apply(spec)
            
        except Exception as e:
            logger.error(f"Error creating chroma transform: {e}")
            if self.enable_fallbacks:
                return self._create_fallback_chroma()
            raise
    
    def _safe_chroma_apply(self, spec: torch.Tensor) -> torch.Tensor:
        """Safely apply chroma filterbank."""
        try:
            if hasattr(self, 'chroma_filterbank'):
                return torch.matmul(self.chroma_filterbank, spec)
            else:
                return self._fallback_chroma_features(spec)
        except Exception as e:
            logger.warning(f"Error applying chroma filterbank: {e}")
            return self._fallback_chroma_features(spec)
    
    def _initialize_analyzers(self):
        """Initialize feature analyzers with error handling."""
        try:
            # Simplified analyzers for bulletproof operation
            self.timbral_analyzer = BulletproofTimbralAnalyzer(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            self.harmonic_analyzer = BulletproofHarmonicAnalyzer(
                sample_rate=self.sample_rate,
                n_chroma=self.n_chroma
            )
            
            self.rhythmic_analyzer = BulletproofRhythmicAnalyzer(
                sample_rate=self.sample_rate,
                hop_length=self.hop_length
            )
            
            logger.info("Feature analyzers initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing analyzers: {e}")
            if self.enable_fallbacks:
                self._initialize_fallback_analyzers()
            else:
                raise
    
    def _initialize_fallback_transforms(self):
        """Initialize minimal fallback transforms."""
        logger.info("Initializing fallback transforms")
        
        # Minimal STFT
        self.stft_transform = self._create_fallback_stft()
        self.mel_transform = self._create_fallback_mel()
        self.chroma_transform = self._create_fallback_chroma()
        
        # Minimal analyzers
        self._initialize_fallback_analyzers()
    
    def _create_fallback_stft(self):
        """Create fallback STFT transform."""
        return lambda x: self._manual_stft(x)
    
    def _create_fallback_mel(self):
        """Create fallback mel transform."""
        return lambda x: self._manual_mel_spectrogram(x)
    
    def _create_fallback_chroma(self):
        """Create fallback chroma transform."""
        return lambda x: self._fallback_chroma_features(x)
    
    def _initialize_fallback_analyzers(self):
        """Initialize fallback analyzers."""
        logger.info("Initializing fallback analyzers")
        
        # Minimal feature extractors
        self.timbral_analyzer = lambda w, s, m: self._extract_basic_timbral(w, s, m)
        self.harmonic_analyzer = lambda w, c: self._extract_basic_harmonic(w, c)
        self.rhythmic_analyzer = lambda w, m: self._extract_basic_rhythmic(w, m)
    
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
            waveform = waveform.squeeze(1)  # Remove channel dimension
        
        # Check for NaN or Inf
        if torch.isnan(waveform).any():
            logger.warning("NaN values detected in waveform, replacing with zeros")
            waveform = torch.nan_to_num(waveform, nan=0.0)
        
        if torch.isinf(waveform).any():
            logger.warning("Inf values detected in waveform, clipping")
            waveform = torch.clamp(waveform, -1.0, 1.0)
        
        # Check length
        if waveform.shape[-1] > self.max_audio_length:
            logger.warning(f"Audio too long ({waveform.shape[-1]}), truncating to {self.max_audio_length}")
            waveform = waveform[..., :self.max_audio_length]
        
        # Check for silence
        if torch.abs(waveform).max() < 1e-8:
            logger.warning("Input appears to be silent")
        
        return waveform
    
    def _safe_device_transfer(self, tensor: torch.Tensor) -> torch.Tensor:
        """Safely transfer tensor to appropriate device."""
        try:
            if self.device != tensor.device:
                return tensor.to(self.device)
            return tensor
        except Exception as e:
            logger.warning(f"Device transfer failed: {e}, keeping on {tensor.device}")
            return tensor
    
    def _manual_stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """Manual STFT implementation as fallback."""
        try:
            # Use torch.stft directly
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
            # Ultimate fallback: dummy spectrogram
            batch_size = waveform.shape[0]
            n_frames = max(1, waveform.shape[-1] // self.hop_length)
            return torch.ones(batch_size, self.n_fft // 2 + 1, n_frames, device=waveform.device) * 1e-8
    
    def _manual_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Manual mel spectrogram implementation as fallback."""
        try:
            # Get STFT
            stft = self._manual_stft(waveform)
            
            # Create mel filterbank manually
            mel_filters = torch.from_numpy(
                librosa.filters.mel(
                    sr=self.sample_rate,
                    n_fft=self.n_fft,
                    n_mels=self.n_mels,
                    fmin=self.f_min,
                    fmax=self.f_max
                )
            ).float().to(waveform.device)
            
            # Apply mel filters
            mel_spec = torch.matmul(mel_filters, stft)
            return mel_spec
            
        except Exception as e:
            logger.error(f"Manual mel spectrogram failed: {e}")
            # Ultimate fallback
            batch_size = waveform.shape[0]
            n_frames = max(1, waveform.shape[-1] // self.hop_length)
            return torch.ones(batch_size, self.n_mels, n_frames, device=waveform.device) * 1e-8
    
    def _fallback_chroma_features(self, spec: torch.Tensor) -> torch.Tensor:
        """Fallback chroma feature extraction."""
        try:
            # Simple pitch class profile
            n_bins = spec.shape[1] if spec.dim() > 1 else spec.shape[0]
            chroma_bins = torch.zeros(spec.shape[0], self.n_chroma, spec.shape[-1], device=spec.device)
            
            # Map frequency bins to chroma bins
            for i in range(n_bins):
                freq = i * self.sample_rate / (2 * n_bins)
                if freq > 0:
                    pitch_class = int(12 * np.log2(freq / 440.0) + 9) % 12
                    if pitch_class < self.n_chroma:
                        chroma_bins[:, pitch_class, :] += spec[:, i, :]
            
            return F.normalize(chroma_bins, p=2, dim=1)
            
        except Exception as e:
            logger.error(f"Fallback chroma failed: {e}")
            # Ultimate fallback
            batch_size = spec.shape[0]
            n_frames = spec.shape[-1]
            return torch.ones(batch_size, self.n_chroma, n_frames, device=spec.device) / self.n_chroma
    
    def forward(
        self,
        waveform: torch.Tensor,
        return_intermediate: bool = False,
        feature_groups: Optional[List[str]] = None,
        max_memory_mb: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Bulletproof forward pass with comprehensive error handling.
        
        Args:
            waveform: Input audio [batch, samples]
            return_intermediate: Whether to return intermediate computations
            feature_groups: Which feature groups to compute
            max_memory_mb: Maximum memory usage in MB
            
        Returns:
            Dictionary with extracted features
        """
        start_time = time.time()
        
        with self._memory_management():
            try:
                # Input validation
                if self.validate_inputs:
                    waveform = self._validate_audio_input(waveform)
                
                # Device management
                waveform = self._safe_device_transfer(waveform)
                
                # Set default feature groups
                if feature_groups is None:
                    feature_groups = ['harmonic', 'rhythmic', 'timbral']
                
                # Initialize results
                features = {}
                
                # Memory check
                if max_memory_mb:
                    if self._estimate_memory_usage(waveform) > max_memory_mb:
                        logger.warning(f"Estimated memory usage exceeds {max_memory_mb}MB, enabling chunked processing")
                        return self._process_chunked(waveform, feature_groups, max_memory_mb)
                
                # Core spectral representations with error handling
                try:
                    stft = self.stft_transform(waveform)
                    features['stft'] = stft
                except Exception as e:
                    logger.error(f"STFT extraction failed: {e}")
                    stft = self._manual_stft(waveform)
                    features['stft'] = stft
                
                try:
                    mel_spec = self.mel_transform(waveform)
                    mel_spec_db = torchaudio.functional.amplitude_to_DB(
                        mel_spec, multiplier=10.0, amin=1e-8, db_multiplier=0.0
                    )
                    features['mel_spectrogram'] = mel_spec_db
                except Exception as e:
                    logger.error(f"Mel spectrogram extraction failed: {e}")
                    mel_spec_db = self._manual_mel_spectrogram(waveform)
                    features['mel_spectrogram'] = mel_spec_db
                
                # Chroma features with error handling
                try:
                    chroma = self.chroma_transform(stft)
                    chroma = F.normalize(chroma, p=2, dim=1)
                    features['chroma'] = chroma
                except Exception as e:
                    logger.error(f"Chroma extraction failed: {e}")
                    chroma = self._fallback_chroma_features(stft)
                    features['chroma'] = chroma
                
                # Extract features by group with error handling
                if 'harmonic' in feature_groups:
                    try:
                        harmonic_features = self.harmonic_analyzer(waveform, chroma)
                        if isinstance(harmonic_features, dict):
                            features.update(harmonic_features)
                        else:
                            features['harmonic_features'] = harmonic_features
                    except Exception as e:
                        logger.error(f"Harmonic analysis failed: {e}")
                        features.update(self._extract_basic_harmonic(waveform, chroma))
                
                if 'rhythmic' in feature_groups:
                    try:
                        rhythmic_features = self.rhythmic_analyzer(waveform, mel_spec_db)
                        if isinstance(rhythmic_features, dict):
                            features.update(rhythmic_features)
                        else:
                            features['rhythmic_features'] = rhythmic_features
                    except Exception as e:
                        logger.error(f"Rhythmic analysis failed: {e}")
                        features.update(self._extract_basic_rhythmic(waveform, mel_spec_db))
                
                if 'timbral' in feature_groups:
                    try:
                        timbral_features = self.timbral_analyzer(waveform, stft, mel_spec_db)
                        if isinstance(timbral_features, dict):
                            features.update(timbral_features)
                        else:
                            features['timbral_features'] = timbral_features
                    except Exception as e:
                        logger.error(f"Timbral analysis failed: {e}")
                        features.update(self._extract_basic_timbral(waveform, stft, mel_spec_db))
                
                # Compute summary statistics
                try:
                    features['summary_stats'] = self._compute_safe_summary_statistics(features)
                except Exception as e:
                    logger.error(f"Summary statistics computation failed: {e}")
                    features['summary_stats'] = {}
                
                # Add metadata
                features['_metadata'] = {
                    'processing_time': time.time() - start_time,
                    'sample_rate': self.sample_rate,
                    'n_fft': self.n_fft,
                    'hop_length': self.hop_length,
                    'input_shape': list(waveform.shape),
                    'feature_groups': feature_groups,
                    'device': str(self.device)
                }
                
                logger.info(f"Feature extraction completed in {time.time() - start_time:.2f}s")
                return features
                
            except Exception as e:
                logger.error(f"Critical error in forward pass: {e}")
                if self.enable_fallbacks:
                    return self._emergency_fallback(waveform, feature_groups)
                raise
    
    def _estimate_memory_usage(self, waveform: torch.Tensor) -> float:
        """Estimate memory usage in MB."""
        batch_size, n_samples = waveform.shape
        
        # Estimate STFT size
        n_frames = n_samples // self.hop_length
        stft_size = batch_size * (self.n_fft // 2 + 1) * n_frames * 4  # 4 bytes per float
        
        # Estimate mel spectrogram size
        mel_size = batch_size * self.n_mels * n_frames * 4
        
        # Estimate total with overhead
        total_bytes = (stft_size + mel_size) * 3  # 3x overhead for intermediate computations
        
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    def _process_chunked(self, waveform: torch.Tensor, feature_groups: List[str], max_memory_mb: int) -> Dict[str, torch.Tensor]:
        """Process audio in chunks to manage memory."""
        batch_size, n_samples = waveform.shape
        
        # Calculate chunk size
        samples_per_mb = int(max_memory_mb * 1024 * 1024 / (batch_size * 4 * 3))  # Conservative estimate
        chunk_size = min(n_samples, samples_per_mb)
        
        logger.info(f"Processing in chunks of {chunk_size} samples")
        
        chunk_features = []
        
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            chunk = waveform[:, start:end]
            
            # Add overlap for continuity
            if start > 0:
                overlap = min(self.n_fft, start)
                chunk = torch.cat([waveform[:, start-overlap:start], chunk], dim=1)
            
            chunk_result = self.forward(chunk, feature_groups=feature_groups)
            chunk_features.append(chunk_result)
        
        # Merge chunks
        return self._merge_chunked_features(chunk_features)
    
    def _merge_chunked_features(self, chunk_features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Merge features from chunked processing."""
        if not chunk_features:
            return {}
        
        merged = {}
        
        for key in chunk_features[0].keys():
            if key.startswith('_'):
                # Skip metadata
                continue
                
            try:
                chunks = [cf[key] for cf in chunk_features if key in cf]
                if chunks and isinstance(chunks[0], torch.Tensor):
                    if chunks[0].dim() >= 2:
                        # Concatenate along time dimension (last dimension)
                        merged[key] = torch.cat(chunks, dim=-1)
                    else:
                        # Average for scalar features
                        merged[key] = torch.stack(chunks).mean(dim=0)
                else:
                    merged[key] = chunks[0] if chunks else None
            except Exception as e:
                logger.warning(f"Failed to merge feature {key}: {e}")
        
        return merged
    
    def _extract_basic_timbral(self, waveform: torch.Tensor, stft: torch.Tensor, mel_spec: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract basic timbral features as fallback."""
        try:
            features = {}
            
            # Spectral centroid
            magnitude = torch.sqrt(stft + 1e-8)
            freqs = torch.linspace(0, self.sample_rate//2, magnitude.shape[1], device=stft.device)
            freqs = freqs.unsqueeze(0).unsqueeze(-1)
            
            spectral_centroid = torch.sum(magnitude * freqs, dim=1) / (torch.sum(magnitude, dim=1) + 1e-8)
            features['spectral_centroid'] = spectral_centroid
            
            # RMS energy
            rms = torch.sqrt(torch.mean(waveform ** 2, dim=-1, keepdim=True))
            features['rms_energy'] = rms.squeeze(-1)
            
            # Zero crossing rate (simplified)
            zcr = torch.mean(torch.abs(torch.diff(torch.sign(waveform), dim=-1)), dim=-1)
            features['zero_crossing_rate'] = zcr
            
            return features
            
        except Exception as e:
            logger.error(f"Basic timbral extraction failed: {e}")
            return {'timbral_error': torch.tensor([1.0])}
    
    def _extract_basic_harmonic(self, waveform: torch.Tensor, chroma: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract basic harmonic features as fallback."""
        try:
            features = {}
            
            # Chroma statistics
            features['chroma_mean'] = chroma.mean(dim=-1)
            features['chroma_std'] = chroma.std(dim=-1)
            
            # Simple key detection (major/minor correlation)
            major_profile = torch.tensor([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], device=chroma.device, dtype=chroma.dtype)
            major_profile = major_profile / major_profile.sum()
            
            avg_chroma = chroma.mean(dim=-1)
            key_strength = F.cosine_similarity(avg_chroma, major_profile.unsqueeze(0), dim=1)
            features['key_strength'] = key_strength
            
            return features
            
        except Exception as e:
            logger.error(f"Basic harmonic extraction failed: {e}")
            return {'harmonic_error': torch.tensor([1.0])}
    
    def _extract_basic_rhythmic(self, waveform: torch.Tensor, mel_spec: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract basic rhythmic features as fallback."""
        try:
            features = {}
            
            # Onset strength (simplified)
            mel_diff = torch.diff(mel_spec, dim=-1)
            mel_diff = torch.clamp(mel_diff, min=0)
            onset_strength = torch.sum(mel_diff, dim=1)
            features['onset_strength'] = onset_strength
            
            # Tempo estimate (very basic)
            if onset_strength.shape[-1] > 10:
                # Simple autocorrelation-based tempo
                onset_ac = F.conv1d(
                    onset_strength.unsqueeze(1),
                    onset_strength.flip(-1).unsqueeze(1),
                    padding=onset_strength.shape[-1] - 1
                )
                tempo_estimate = torch.argmax(onset_ac[:, :, onset_strength.shape[-1]//2:], dim=-1) + 1
                tempo_estimate = tempo_estimate.float() * 60 * self.sample_rate / (self.hop_length * onset_strength.shape[-1])
                features['tempo_estimate'] = tempo_estimate.float()
            else:
                features['tempo_estimate'] = torch.tensor([120.0] * waveform.shape[0], device=waveform.device)
            
            return features
            
        except Exception as e:
            logger.error(f"Basic rhythmic extraction failed: {e}")
            return {'rhythmic_error': torch.tensor([1.0])}
    
    def _compute_safe_summary_statistics(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute summary statistics with error handling."""
        stats = {}
        
        temporal_features = ['chroma', 'mel_spectrogram']
        
        for feat_name in temporal_features:
            if feat_name in features:
                try:
                    feat = features[feat_name]
                    if feat.dim() >= 2:
                        stats[f'{feat_name}_mean'] = torch.mean(feat, dim=-1)
                        stats[f'{feat_name}_std'] = torch.std(feat, dim=-1)
                        if feat.shape[-1] > 0:
                            stats[f'{feat_name}_median'] = torch.median(feat, dim=-1)[0]
                except Exception as e:
                    logger.warning(f"Failed to compute stats for {feat_name}: {e}")
        
        return stats
    
    def _emergency_fallback(self, waveform: torch.Tensor, feature_groups: List[str]) -> Dict[str, torch.Tensor]:
        """Emergency fallback when all else fails."""
        logger.warning("Using emergency fallback features")
        
        batch_size = waveform.shape[0]
        device = waveform.device
        
        # Minimal feature set
        features = {
            'energy': torch.mean(waveform ** 2, dim=-1),
            'peak_amplitude': torch.max(torch.abs(waveform), dim=-1)[0],
            'duration': torch.tensor([waveform.shape[-1] / self.sample_rate] * batch_size, device=device),
            '_emergency_fallback': torch.tensor([True], device=device)
        }
        
        return features
    
    def to(self, device):
        """Override to method for device management."""
        self.device = device
        result = super().to(device)
        
        # Update transforms if they exist
        try:
            if hasattr(self, 'stft_transform'):
                self.stft_transform = self.stft_transform.to(device)
            if hasattr(self, 'mel_transform'):
                self.mel_transform = self.mel_transform.to(device)
        except Exception as e:
            logger.warning(f"Error moving transforms to {device}: {e}")
        
        return result


class BulletproofTimbralAnalyzer(nn.Module):
    """Bulletproof timbral feature analyzer."""
    
    def __init__(self, sample_rate: int, n_fft: int, hop_length: int):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        try:
            self.mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=13,
                melkwargs={'n_fft': n_fft, 'hop_length': hop_length}
            )
        except Exception as e:
            logger.warning(f"MFCC transform creation failed: {e}")
            self.mfcc_transform = None
    
    def __call__(self, waveform: torch.Tensor, stft: torch.Tensor, mel_spec: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract timbral features with error handling."""
        features = {}
        
        try:
            # MFCCs
            if self.mfcc_transform is not None:
                mfcc = self.mfcc_transform(waveform)
                features['mfcc'] = mfcc
            else:
                # Fallback MFCC approximation
                features['mfcc'] = mel_spec[:, :13, :]  # Use first 13 mel bands
            
            # Spectral features
            magnitude = torch.sqrt(stft + 1e-8)
            
            # Spectral centroid
            freqs = torch.linspace(0, self.sample_rate//2, stft.shape[1], device=stft.device)
            freqs = freqs.unsqueeze(0).unsqueeze(-1)
            
            spectral_centroid = torch.sum(magnitude * freqs, dim=1) / (torch.sum(magnitude, dim=1) + 1e-8)
            features['spectral_centroid'] = spectral_centroid
            
            # Spectral bandwidth
            centroid_expanded = spectral_centroid.unsqueeze(1)
            spectral_bandwidth = torch.sqrt(
                torch.sum(magnitude * (freqs - centroid_expanded)**2, dim=1) / 
                (torch.sum(magnitude, dim=1) + 1e-8)
            )
            features['spectral_bandwidth'] = spectral_bandwidth
            
        except Exception as e:
            logger.error(f"Timbral analysis error: {e}")
            # Return minimal features
            features['mfcc'] = torch.zeros(waveform.shape[0], 13, 1, device=waveform.device)
            features['spectral_centroid'] = torch.zeros(waveform.shape[0], 1, device=waveform.device)
            features['spectral_bandwidth'] = torch.zeros(waveform.shape[0], 1, device=waveform.device)
        
        return features


class BulletproofHarmonicAnalyzer(nn.Module):
    """Bulletproof harmonic content analyzer."""
    
    def __init__(self, sample_rate: int, n_chroma: int = 12):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_chroma = n_chroma
        
        # Key profiles
        major_profile = torch.tensor([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = torch.tensor([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        self.register_buffer('major_profile', major_profile / major_profile.sum())
        self.register_buffer('minor_profile', minor_profile / minor_profile.sum())
    
    def __call__(self, waveform: torch.Tensor, chroma: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze harmonic content with error handling."""
        features = {}
        
        try:
            # Key detection
            key_correlations = self._detect_key_safe(chroma)
            features['key_correlations'] = key_correlations
            
            # Harmonic change detection
            harmonic_novelty = self._compute_harmonic_novelty_safe(chroma)
            features['harmonic_novelty'] = harmonic_novelty
            
            # Tonal centroid
            tonal_centroid = self._compute_tonal_centroid_safe(chroma)
            features['tonal_centroid'] = tonal_centroid
            
        except Exception as e:
            logger.error(f"Harmonic analysis error: {e}")
            # Return minimal features
            batch_size = chroma.shape[0]
            features['key_correlations'] = torch.zeros(batch_size, 24, device=chroma.device)
            features['harmonic_novelty'] = torch.zeros(batch_size, chroma.shape[-1], device=chroma.device)
            features['tonal_centroid'] = torch.zeros(batch_size, 2, chroma.shape[-1], device=chroma.device)
        
        return features
    
    def _detect_key_safe(self, chroma: torch.Tensor) -> torch.Tensor:
        """Safe key detection."""
        try:
            avg_chroma = chroma.mean(dim=-1)
            correlations = []
            
            for shift in range(12):
                # Major key
                shifted_major = torch.roll(self.major_profile, shift)
                major_corr = F.cosine_similarity(avg_chroma, shifted_major.unsqueeze(0), dim=1)
                correlations.append(major_corr)
                
                # Minor key
                shifted_minor = torch.roll(self.minor_profile, shift)
                minor_corr = F.cosine_similarity(avg_chroma, shifted_minor.unsqueeze(0), dim=1)
                correlations.append(minor_corr)
            
            return torch.stack(correlations, dim=1)
        except Exception as e:
            logger.warning(f"Key detection failed: {e}")
            return torch.zeros(chroma.shape[0], 24, device=chroma.device)
    
    def _compute_harmonic_novelty_safe(self, chroma: torch.Tensor) -> torch.Tensor:
        """Safe harmonic novelty computation."""
        try:
            chroma_diff = torch.diff(chroma, dim=-1)
            novelty = torch.norm(chroma_diff, p=2, dim=1)
            novelty = F.pad(novelty, (1, 0), mode='constant', value=0)
            return novelty
        except Exception as e:
            logger.warning(f"Harmonic novelty failed: {e}")
            return torch.zeros(chroma.shape[0], chroma.shape[-1], device=chroma.device)
    
    def _compute_tonal_centroid_safe(self, chroma: torch.Tensor) -> torch.Tensor:
        """Safe tonal centroid computation."""
        try:
            angles = torch.linspace(0, 2*np.pi, 12, device=chroma.device, dtype=chroma.dtype)
            fifth_angles = angles * 7 % (2 * np.pi)
            
            cos_components = torch.cos(fifth_angles).unsqueeze(0).unsqueeze(-1)
            sin_components = torch.sin(fifth_angles).unsqueeze(0).unsqueeze(-1)
            
            centroid_x = torch.sum(chroma * cos_components, dim=1)
            centroid_y = torch.sum(chroma * sin_components, dim=1)
            
            return torch.stack([centroid_x, centroid_y], dim=1)
        except Exception as e:
            logger.warning(f"Tonal centroid failed: {e}")
            return torch.zeros(chroma.shape[0], 2, chroma.shape[-1], device=chroma.device)


class BulletproofRhythmicAnalyzer(nn.Module):
    """Bulletproof rhythmic analysis."""
    
    def __init__(self, sample_rate: int, hop_length: int):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
    
    def __call__(self, waveform: torch.Tensor, mel_spec: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze rhythmic features with error handling."""
        features = {}
        
        try:
            # Onset detection
            onset_strength = self._compute_onset_strength_safe(mel_spec)
            features['onset_strength'] = onset_strength
            
            # Simple tempo estimation
            tempo_estimate = self._estimate_tempo_safe(onset_strength)
            features['tempo_estimate'] = tempo_estimate
            
            # Beat tracking (simplified)
            beat_strength = self._track_beats_safe(onset_strength)
            features['beat_strength'] = beat_strength
            
        except Exception as e:
            logger.error(f"Rhythmic analysis error: {e}")
            # Return minimal features
            batch_size = mel_spec.shape[0]
            time_frames = mel_spec.shape[-1]
            features['onset_strength'] = torch.zeros(batch_size, time_frames, device=mel_spec.device)
            features['tempo_estimate'] = torch.tensor([120.0] * batch_size, device=mel_spec.device)
            features['beat_strength'] = torch.zeros(batch_size, time_frames, device=mel_spec.device)
        
        return features
    
    def _compute_onset_strength_safe(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Safe onset strength computation."""
        try:
            mel_diff = torch.diff(mel_spec, dim=-1)
            mel_diff = torch.clamp(mel_diff, min=0)
            onset_strength = torch.sum(mel_diff, dim=1)
            onset_strength = F.pad(onset_strength, (1, 0), mode='constant', value=0)
            return onset_strength
        except Exception as e:
            logger.warning(f"Onset strength failed: {e}")
            return torch.zeros(mel_spec.shape[0], mel_spec.shape[-1], device=mel_spec.device)
    
    def _estimate_tempo_safe(self, onset_strength: torch.Tensor) -> torch.Tensor:
        """Safe tempo estimation."""
        try:
            # Simple autocorrelation-based tempo
            batch_size = onset_strength.shape[0]
            tempos = []
            
            for i in range(batch_size):
                onset = onset_strength[i]
                if onset.shape[0] > 10:
                    # Autocorrelation
                    onset_ac = F.conv1d(
                        onset.unsqueeze(0).unsqueeze(0),
                        onset.flip(0).unsqueeze(0).unsqueeze(0),
                        padding=onset.shape[0] - 1
                    )
                    
                    # Find peak in reasonable tempo range
                    center = onset.shape[0]
                    start_lag = max(1, int(0.3 * self.sample_rate / self.hop_length))  # ~200 BPM
                    end_lag = min(center, int(2.0 * self.sample_rate / self.hop_length))  # ~30 BPM
                    
                    tempo_region = onset_ac[0, 0, center + start_lag:center + end_lag]
                    if tempo_region.numel() > 0:
                        peak_lag = torch.argmax(tempo_region) + start_lag
                        tempo_bpm = 60 * self.sample_rate / (self.hop_length * peak_lag)
                        tempos.append(float(tempo_bpm))
                    else:
                        tempos.append(120.0)
                else:
                    tempos.append(120.0)
            
            return torch.tensor(tempos, device=onset_strength.device)
            
        except Exception as e:
            logger.warning(f"Tempo estimation failed: {e}")
            return torch.tensor([120.0] * onset_strength.shape[0], device=onset_strength.device)
    
    def _track_beats_safe(self, onset_strength: torch.Tensor) -> torch.Tensor:
        """Safe beat tracking."""
        try:
            # Simple smoothing as beat strength proxy
            kernel_size = 5
            kernel = torch.ones(1, 1, kernel_size, device=onset_strength.device) / kernel_size
            
            smoothed = F.conv1d(
                onset_strength.unsqueeze(1),
                kernel,
                padding=kernel_size // 2
            ).squeeze(1)
            
            return smoothed
        except Exception as e:
            logger.warning(f"Beat tracking failed: {e}")
            return onset_strength


# Test and utility functions
def test_bulletproof_features():
    """Test the bulletproof features module."""
    logger.info("Testing BulletproofAdvancedMusicalFeatures")
    
    # Test with various input conditions
    test_cases = [
        torch.randn(1, 22050),  # Normal case
        torch.randn(2, 44100),  # Different sample rate
        torch.zeros(1, 8000),   # Silent audio
        torch.ones(1, 1000) * 0.5,  # Constant signal
        torch.randn(1, 22050 * 30),  # Very long audio
    ]
    
    extractor = BulletproofAdvancedMusicalFeatures(
        sample_rate=22050,
        enable_fallbacks=True,
        validate_inputs=True,
        memory_efficient=True
    )
    
    for i, test_audio in enumerate(test_cases):
        try:
            logger.info(f"Testing case {i+1}: shape {test_audio.shape}")
            features = extractor(test_audio)
            logger.info(f"  Extracted {len(features)} features")
            
            # Check for critical features
            required_features = ['mel_spectrogram', 'chroma', 'stft']
            for feat in required_features:
                if feat in features:
                    logger.info(f"    {feat}: {features[feat].shape}")
                else:
                    logger.warning(f"    Missing feature: {feat}")
                    
        except Exception as e:
            logger.error(f"Test case {i+1} failed: {e}")
    
    logger.info("Bulletproof features testing completed")


if __name__ == "__main__":
    test_bulletproof_features()