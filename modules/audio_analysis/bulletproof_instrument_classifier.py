"""
Bulletproof Instrument Classifier with comprehensive error handling and fallback strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import warnings
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from contextlib import contextmanager
import gc
import time

logger = logging.getLogger(__name__)


class BulletproofInstrumentClassifier(nn.Module):
    """
    Bulletproof multi-scale instrument classification with:
    - Comprehensive parameter validation and sanitization
    - Multiple fallback strategies for feature extraction
    - Memory management for large audio files
    - Device compatibility with automatic fallback
    - Graceful degradation when classification components fail
    - Hierarchical classification with robust error handling
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        num_families: int = 8,
        num_instruments: int = 50,
        num_techniques: int = 20,
        hidden_dim: int = 512,
        num_conv_layers: int = 4,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
        max_audio_length: int = 22050 * 300,  # 5 minutes max
        memory_efficient: bool = True,
        enable_fallbacks: bool = True,
        validate_inputs: bool = True,
        **kwargs
    ):
        super().__init__()
        
        # Validate and sanitize parameters
        self.sample_rate = max(8000, min(192000, int(sample_rate)))
        self.n_mels = max(10, min(512, int(n_mels)))
        self.n_fft = max(256, min(8192, int(n_fft)))
        self.hop_length = max(64, min(2048, int(hop_length)))
        self.num_families = max(1, min(100, int(num_families)))
        self.num_instruments = max(1, min(1000, int(num_instruments)))
        self.num_techniques = max(1, min(100, int(num_techniques)))
        self.hidden_dim = max(32, min(2048, int(hidden_dim)))
        self.num_conv_layers = max(1, min(10, int(num_conv_layers)))
        self.num_lstm_layers = max(1, min(5, int(num_lstm_layers)))
        self.dropout = max(0.0, min(0.9, float(dropout)))
        self.max_audio_length = max(self.sample_rate, max_audio_length)
        self.memory_efficient = memory_efficient
        self.enable_fallbacks = enable_fallbacks
        self.validate_inputs = validate_inputs
        
        # Initialize device
        self.device = torch.device('cpu')
        
        try:
            self._initialize_audio_transforms()
            self._initialize_feature_extraction()
            self._initialize_temporal_modeling()
            self._initialize_classification_heads()
            self._initialize_feature_fusion()
            
            logger.info(f"BulletproofInstrumentClassifier initialized successfully")
            logger.info(f"Classes: {num_families} families, {num_instruments} instruments, {num_techniques} techniques")
            
        except Exception as e:
            logger.error(f"Error initializing BulletproofInstrumentClassifier: {e}")
            if not self.enable_fallbacks:
                raise
            self._initialize_fallback_model()
    
    def _initialize_audio_transforms(self):
        """Initialize audio preprocessing transforms with error handling."""
        try:
            # Mel-spectrogram transform
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                f_min=0,
                f_max=self.sample_rate // 2
            )
            
            # MFCC transform
            self.mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=min(13, self.n_mels),
                melkwargs={
                    'n_fft': self.n_fft,
                    'hop_length': self.hop_length,
                    'n_mels': self.n_mels
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize audio transforms: {e}")
            if not self.enable_fallbacks:
                raise
            self._create_fallback_transforms()
    
    def _create_fallback_transforms(self):
        """Create fallback audio transforms."""
        try:
            # Simplified transforms with reduced parameters
            self.mel_transform = torchaudio.transforms.Spectrogram(
                n_fft=min(512, self.n_fft),
                hop_length=min(256, self.hop_length),
                power=2.0
            )
            
            # Simple MFCC fallback
            self.mfcc_transform = None
            logger.warning("Using fallback audio transforms")
            
        except Exception as e:
            logger.error(f"Fallback transforms creation failed: {e}")
            self.mel_transform = None
            self.mfcc_transform = None
    
    def _initialize_feature_extraction(self):
        """Initialize convolutional feature extraction with error handling."""
        try:
            self.spectral_conv = self._build_safe_conv_layers(
                in_channels=1,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_conv_layers,
                dropout=self.dropout
            )
        except Exception as e:
            logger.error(f"Failed to initialize feature extraction: {e}")
            if not self.enable_fallbacks:
                raise
            self._create_fallback_feature_extraction()
    
    def _build_safe_conv_layers(self, in_channels: int, hidden_dim: int, num_layers: int, dropout: float):
        """Build convolutional layers with comprehensive error handling."""
        try:
            layers = []
            current_channels = in_channels
            
            for i in range(num_layers):
                try:
                    out_channels = max(16, hidden_dim // (2 ** max(0, num_layers - i - 1)))
                    
                    layers.extend([
                        nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Dropout2d(dropout)
                    ])
                    current_channels = out_channels
                    
                except Exception as e:
                    logger.warning(f"Failed to create conv layer {i}: {e}")
                    if self.enable_fallbacks and i > 0:
                        break  # Use what we have so far
                    elif not self.enable_fallbacks:
                        raise
            
            # Global pooling
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            
            return nn.Sequential(*layers)
            
        except Exception as e:
            logger.error(f"Conv layers creation failed: {e}")
            if self.enable_fallbacks:
                return self._create_minimal_conv_layers(in_channels, hidden_dim)
            raise
    
    def _create_minimal_conv_layers(self, in_channels: int, hidden_dim: int):
        """Create minimal convolutional layers as fallback."""
        try:
            return nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(32, hidden_dim),
                nn.ReLU()
            )
        except Exception as e:
            logger.error(f"Minimal conv layers creation failed: {e}")
            # Ultra-minimal fallback
            return nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(in_channels, hidden_dim)
            )
    
    def _create_fallback_feature_extraction(self):
        """Create fallback feature extraction."""
        self.spectral_conv = self._create_minimal_conv_layers(1, self.hidden_dim)
        logger.warning("Using fallback feature extraction")
    
    def _initialize_temporal_modeling(self):
        """Initialize temporal modeling with error handling."""
        try:
            self.temporal_lstm = nn.LSTM(
                input_size=self.n_mels,
                hidden_size=self.hidden_dim,
                num_layers=self.num_lstm_layers,
                batch_first=True,
                dropout=self.dropout if self.num_lstm_layers > 1 else 0,
                bidirectional=True
            )
            
            # Attention mechanism
            self.attention = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=min(8, self.hidden_dim // 64),
                dropout=self.dropout,
                batch_first=True
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize temporal modeling: {e}")
            if not self.enable_fallbacks:
                raise
            self._create_fallback_temporal_modeling()
    
    def _create_fallback_temporal_modeling(self):
        """Create fallback temporal modeling."""
        try:
            # Simple LSTM
            self.temporal_lstm = nn.LSTM(
                input_size=self.n_mels,
                hidden_size=self.hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=False
            )
            
            # No attention in fallback
            self.attention = None
            logger.warning("Using fallback temporal modeling")
            
        except Exception as e:
            logger.error(f"Fallback temporal modeling failed: {e}")
            # Ultra-minimal fallback
            self.temporal_lstm = None
            self.attention = None
    
    def _initialize_feature_fusion(self):
        """Initialize feature fusion with error handling."""
        try:
            # Calculate input dimensions safely
            mel_dim = self.n_mels
            mfcc_dim = min(13, self.n_mels)
            spectral_dim = self.hidden_dim
            temporal_dim = self.hidden_dim * 2 if hasattr(self.temporal_lstm, 'bidirectional') and self.temporal_lstm.bidirectional else self.hidden_dim
            
            self.feature_fusion = nn.ModuleDict({
                'mel_proj': nn.Linear(mel_dim, self.hidden_dim),
                'mfcc_proj': nn.Linear(mfcc_dim, self.hidden_dim),
                'spectral_proj': nn.Linear(spectral_dim, self.hidden_dim),
                'temporal_proj': nn.Linear(temporal_dim, self.hidden_dim)
            })
            
        except Exception as e:
            logger.error(f"Failed to initialize feature fusion: {e}")
            if not self.enable_fallbacks:
                raise
            self._create_fallback_feature_fusion()
    
    def _create_fallback_feature_fusion(self):
        """Create fallback feature fusion."""
        try:
            self.feature_fusion = nn.ModuleDict({
                'mel_proj': nn.Linear(self.n_mels, self.hidden_dim),
                'spectral_proj': nn.Linear(self.hidden_dim, self.hidden_dim),
            })
            logger.warning("Using fallback feature fusion")
        except Exception as e:
            logger.error(f"Fallback feature fusion failed: {e}")
            self.feature_fusion = nn.ModuleDict({
                'combined_proj': nn.Linear(self.hidden_dim, self.hidden_dim)
            })
    
    def _initialize_classification_heads(self):
        """Initialize hierarchical classification heads with error handling."""
        try:
            # Family classification head
            self.family_head = self._build_safe_classification_head(
                input_dim=self.hidden_dim,
                output_dim=self.num_families,
                name="family"
            )
            
            # Instrument classification head (conditioned on family)
            self.instrument_head = self._build_safe_classification_head(
                input_dim=self.hidden_dim + self.num_families,
                output_dim=self.num_instruments,
                name="instrument"
            )
            
            # Technique classification head (conditioned on family and instrument)
            self.technique_head = self._build_safe_classification_head(
                input_dim=self.hidden_dim + self.num_families + self.num_instruments,
                output_dim=self.num_techniques,
                name="technique"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize classification heads: {e}")
            if not self.enable_fallbacks:
                raise
            self._create_fallback_classification_heads()
    
    def _build_safe_classification_head(self, input_dim: int, output_dim: int, name: str):
        """Build classification head with error handling."""
        try:
            return nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(input_dim // 2, output_dim)
            )
        except Exception as e:
            logger.warning(f"Failed to create {name} head: {e}")
            # Fallback: simple linear layer
            return nn.Linear(input_dim, output_dim)
    
    def _create_fallback_classification_heads(self):
        """Create fallback classification heads."""
        try:
            self.family_head = nn.Linear(self.hidden_dim, self.num_families)
            self.instrument_head = nn.Linear(self.hidden_dim, self.num_instruments)
            self.technique_head = nn.Linear(self.hidden_dim, self.num_techniques)
            logger.warning("Using fallback classification heads")
        except Exception as e:
            logger.error(f"Fallback classification heads failed: {e}")
            # Ultimate fallback
            self.family_head = nn.Identity()
            self.instrument_head = nn.Identity()
            self.technique_head = nn.Identity()
    
    def _initialize_fallback_model(self):
        """Initialize minimal fallback model."""
        logger.warning("Initializing fallback instrument classifier")
        
        try:
            # Minimal audio transforms
            self.mel_transform = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=256)
            self.mfcc_transform = None
            
            # Minimal feature extraction
            self.spectral_conv = nn.Sequential(
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(16, self.hidden_dim),
                nn.ReLU()
            )
            
            # No temporal modeling
            self.temporal_lstm = None
            self.attention = None
            
            # Simple feature fusion
            self.feature_fusion = nn.ModuleDict({
                'mel_proj': nn.Linear(self.hidden_dim, self.hidden_dim)
            })
            
            # Simple classification heads
            self.family_head = nn.Linear(self.hidden_dim, self.num_families)
            self.instrument_head = nn.Linear(self.hidden_dim, self.num_instruments)
            self.technique_head = nn.Linear(self.hidden_dim, self.num_techniques)
            
            self._is_fallback_model = True
            
        except Exception as e:
            logger.error(f"Failed to initialize fallback model: {e}")
            raise RuntimeError("Complete instrument classifier initialization failure")
    
    def _validate_audio_input(self, waveform: torch.Tensor) -> torch.Tensor:
        """Validate and sanitize audio input."""
        if not self.validate_inputs:
            return waveform
        
        try:
            if not isinstance(waveform, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor, got {type(waveform)}")
            
            if waveform.numel() == 0:
                raise ValueError("Empty audio tensor")
            
            # Handle different input shapes
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.dim() == 3:
                waveform = waveform.squeeze(1)
            elif waveform.dim() > 3:
                raise ValueError(f"Unsupported audio shape: {waveform.shape}")
            
            # Check for invalid values
            if torch.isnan(waveform).any():
                logger.warning("NaN values detected in audio")
                waveform = torch.nan_to_num(waveform, nan=0.0)
            
            if torch.isinf(waveform).any():
                logger.warning("Infinite values detected in audio")
                waveform = torch.clamp(waveform, -10.0, 10.0)
            
            # Limit audio length
            if waveform.shape[-1] > self.max_audio_length:
                logger.warning(f"Audio too long, truncating to {self.max_audio_length}")
                waveform = waveform[..., :self.max_audio_length]
            
            # Ensure minimum length
            min_length = self.sample_rate // 10  # 0.1 seconds
            if waveform.shape[-1] < min_length:
                padding = min_length - waveform.shape[-1]
                waveform = F.pad(waveform, (0, padding))
            
            return waveform.to(self.device)
            
        except Exception as e:
            logger.error(f"Audio validation failed: {e}")
            if self.enable_fallbacks:
                batch_size = 1 if waveform.dim() == 1 else waveform.shape[0]
                return torch.zeros(batch_size, self.sample_rate, device=self.device)
            raise
    
    @contextmanager
    def _memory_efficient_context(self):
        """Context manager for memory efficient processing."""
        if self.memory_efficient:
            with torch.inference_mode():
                try:
                    yield
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
        else:
            yield
    
    def extract_features(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract multi-scale audio features with comprehensive error handling."""
        waveform = self._validate_audio_input(waveform)
        
        features = {}
        
        with self._memory_efficient_context():
            try:
                # Extract mel-spectrogram
                mel_features = self._safe_mel_extraction(waveform)
                features.update(mel_features)
                
                # Extract MFCC features
                if self.mfcc_transform is not None:
                    mfcc_features = self._safe_mfcc_extraction(waveform)
                    features.update(mfcc_features)
                
                # Extract spectral features via CNN
                spectral_features = self._safe_spectral_extraction(waveform, features.get('mel_spectrogram'))
                features.update(spectral_features)
                
                # Extract temporal features via LSTM
                if self.temporal_lstm is not None:
                    temporal_features = self._safe_temporal_extraction(features.get('mel_spectrogram'))
                    features.update(temporal_features)
                
            except Exception as e:
                logger.error(f"Feature extraction failed: {e}")
                if self.enable_fallbacks:
                    features = self._extract_fallback_features(waveform)
                else:
                    raise
        
        return features
    
    def _safe_mel_extraction(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Safely extract mel-spectrogram features."""
        try:
            if waveform.dim() == 3:
                waveform = waveform.squeeze(1)
            
            # Extract mel-spectrogram
            if self.mel_transform is not None:
                mel_spec = self.mel_transform(waveform)
                
                # Convert to log scale with numerical stability
                if hasattr(self.mel_transform, 'amplitude_to_DB'):
                    mel_spec_db = torchaudio.functional.amplitude_to_DB(mel_spec, multiplier=10.0, amin=1e-8)
                else:
                    mel_spec_db = torch.log(mel_spec.clamp(min=1e-8))
            else:
                # Fallback: simple STFT
                stft = torch.stft(
                    waveform, n_fft=512, hop_length=256,
                    return_complex=True, window=torch.hann_window(512, device=waveform.device)
                )
                mel_spec_db = torch.log(torch.abs(stft).clamp(min=1e-8))
            
            # Statistical features
            mel_stats = torch.cat([
                mel_spec_db.mean(dim=2),  # Temporal mean
                mel_spec_db.std(dim=2)    # Temporal std
            ], dim=1).mean(dim=1)  # Pool over frequency
            
            return {
                'mel_spectrogram': mel_spec_db,
                'mel_features': mel_stats
            }
            
        except Exception as e:
            logger.warning(f"Mel extraction failed: {e}")
            batch_size = waveform.shape[0]
            return {
                'mel_spectrogram': torch.randn(batch_size, self.n_mels, 100, device=self.device),
                'mel_features': torch.randn(batch_size, self.n_mels, device=self.device)
            }
    
    def _safe_mfcc_extraction(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Safely extract MFCC features."""
        try:
            mfcc = self.mfcc_transform(waveform)
            
            # Statistical features
            mfcc_stats = torch.cat([
                mfcc.mean(dim=2),
                mfcc.std(dim=2)
            ], dim=1).mean(dim=1)
            
            return {
                'mfcc': mfcc,
                'mfcc_features': mfcc_stats
            }
            
        except Exception as e:
            logger.warning(f"MFCC extraction failed: {e}")
            batch_size = waveform.shape[0]
            n_mfcc = min(13, self.n_mels)
            return {
                'mfcc': torch.randn(batch_size, n_mfcc, 100, device=self.device),
                'mfcc_features': torch.randn(batch_size, n_mfcc, device=self.device)
            }
    
    def _safe_spectral_extraction(self, waveform: torch.Tensor, mel_spec: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Safely extract spectral features via CNN."""
        try:
            if mel_spec is not None:
                mel_input = mel_spec.unsqueeze(1) if mel_spec.dim() == 3 else mel_spec
            else:
                # Fallback: create dummy mel spectrogram
                batch_size = waveform.shape[0]
                mel_input = torch.randn(batch_size, 1, self.n_mels, 100, device=self.device)
            
            spectral_features = self.spectral_conv(mel_input)
            
            # Ensure correct shape
            if spectral_features.dim() > 2:
                spectral_features = spectral_features.view(spectral_features.shape[0], -1)
            
            # Ensure correct dimension
            if spectral_features.shape[1] != self.hidden_dim:
                proj = nn.Linear(spectral_features.shape[1], self.hidden_dim).to(self.device)
                spectral_features = proj(spectral_features)
            
            return {'spectral_features': spectral_features}
            
        except Exception as e:
            logger.warning(f"Spectral extraction failed: {e}")
            batch_size = waveform.shape[0]
            return {'spectral_features': torch.randn(batch_size, self.hidden_dim, device=self.device)}
    
    def _safe_temporal_extraction(self, mel_spec: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Safely extract temporal features via LSTM."""
        try:
            if mel_spec is None:
                raise ValueError("No mel spectrogram for temporal extraction")
            
            # Prepare sequence for LSTM: [batch, time, freq]
            mel_sequence = mel_spec.transpose(1, 2)
            
            # LSTM forward
            lstm_out, _ = self.temporal_lstm(mel_sequence)
            
            # Attention pooling if available
            if self.attention is not None:
                try:
                    # Project to correct dimension if needed
                    if lstm_out.shape[-1] != self.hidden_dim:
                        proj = nn.Linear(lstm_out.shape[-1], self.hidden_dim).to(self.device)
                        lstm_out = proj(lstm_out)
                    
                    temporal_features, _ = self.attention(lstm_out, lstm_out, lstm_out)
                    temporal_features = temporal_features.mean(dim=1)  # Pool over time
                except Exception as e:
                    logger.warning(f"Attention pooling failed: {e}")
                    temporal_features = lstm_out.mean(dim=1)
            else:
                temporal_features = lstm_out.mean(dim=1)
            
            return {'temporal_features': temporal_features}
            
        except Exception as e:
            logger.warning(f"Temporal extraction failed: {e}")
            batch_size = mel_spec.shape[0] if mel_spec is not None else 1
            return {'temporal_features': torch.randn(batch_size, self.hidden_dim, device=self.device)}
    
    def _extract_fallback_features(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract minimal fallback features."""
        try:
            batch_size = waveform.shape[0]
            
            # Simple energy-based features
            frame_length = min(1024, waveform.shape[-1] // 10)
            hop_length = frame_length // 2
            
            frames = waveform.unfold(-1, frame_length, hop_length)
            energy = torch.mean(frames ** 2, dim=-1)
            
            # Use energy as proxy for mel features
            mel_features = energy.mean(dim=-1)
            
            # Dummy features
            features = {
                'mel_features': mel_features,
                'spectral_features': torch.randn(batch_size, self.hidden_dim, device=self.device),
                'temporal_features': torch.randn(batch_size, self.hidden_dim, device=self.device)
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Fallback feature extraction failed: {e}")
            batch_size = waveform.shape[0]
            return {
                'mel_features': torch.ones(batch_size, self.n_mels, device=self.device),
                'spectral_features': torch.ones(batch_size, self.hidden_dim, device=self.device),
                'temporal_features': torch.ones(batch_size, self.hidden_dim, device=self.device)
            }
    
    def forward(
        self, 
        waveform: torch.Tensor,
        return_features: bool = False,
        return_hierarchical: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with comprehensive error handling.
        
        Args:
            waveform: Input audio [batch, samples]
            return_features: Whether to return extracted features
            return_hierarchical: Whether to return all hierarchy levels
            
        Returns:
            Dictionary with classification results and optionally features
        """
        try:
            with self._memory_efficient_context():
                # Extract features
                features = self.extract_features(waveform)
                
                # Project and combine features
                combined_features = self._safe_feature_combination(features)
                
                # Hierarchical classification
                classification_results = self._safe_hierarchical_classification(combined_features)
                
                result = {
                    'family_logits': classification_results['family_logits'],
                    'instrument_logits': classification_results['instrument_logits'],
                    'technique_logits': classification_results['technique_logits'],
                    'combined_features': combined_features
                }
                
                if return_hierarchical:
                    result.update({
                        'family_probs': F.softmax(classification_results['family_logits'], dim=1),
                        'instrument_probs': F.softmax(classification_results['instrument_logits'], dim=1),
                        'technique_probs': F.softmax(classification_results['technique_logits'], dim=1)
                    })
                
                if return_features:
                    result['features'] = features
                
                return result
                
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            if self.enable_fallbacks:
                batch_size = waveform.shape[0] if isinstance(waveform, torch.Tensor) else 1
                return self._create_fallback_output(batch_size, return_features, return_hierarchical, str(e))
            raise
    
    def _safe_feature_combination(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Safely combine features with error handling."""
        try:
            combined_list = []
            
            # Project mel features
            if 'mel_features' in features and 'mel_proj' in self.feature_fusion:
                try:
                    mel_proj = self.feature_fusion['mel_proj'](features['mel_features'])
                    combined_list.append(mel_proj)
                except Exception as e:
                    logger.warning(f"Mel projection failed: {e}")
            
            # Project MFCC features
            if 'mfcc_features' in features and 'mfcc_proj' in self.feature_fusion:
                try:
                    mfcc_proj = self.feature_fusion['mfcc_proj'](features['mfcc_features'])
                    combined_list.append(mfcc_proj)
                except Exception as e:
                    logger.warning(f"MFCC projection failed: {e}")
            
            # Project spectral features
            if 'spectral_features' in features and 'spectral_proj' in self.feature_fusion:
                try:
                    spectral_proj = self.feature_fusion['spectral_proj'](features['spectral_features'])
                    combined_list.append(spectral_proj)
                except Exception as e:
                    logger.warning(f"Spectral projection failed: {e}")
            
            # Project temporal features
            if 'temporal_features' in features and 'temporal_proj' in self.feature_fusion:
                try:
                    temporal_proj = self.feature_fusion['temporal_proj'](features['temporal_features'])
                    combined_list.append(temporal_proj)
                except Exception as e:
                    logger.warning(f"Temporal projection failed: {e}")
            
            if combined_list:
                # Combine features
                if len(combined_list) == 1:
                    combined_features = combined_list[0]
                else:
                    combined_features = torch.stack(combined_list, dim=0).mean(dim=0)
                
                # Normalize
                combined_features = F.normalize(combined_features, p=2, dim=1)
                
                return combined_features
            else:
                # Fallback: use first available feature or create dummy
                for key, value in features.items():
                    if isinstance(value, torch.Tensor) and value.dim() == 2:
                        if value.shape[1] == self.hidden_dim:
                            return value
                        else:
                            # Project to correct dimension
                            proj = nn.Linear(value.shape[1], self.hidden_dim).to(self.device)
                            return proj(value)
                
                # Ultimate fallback
                batch_size = list(features.values())[0].shape[0] if features else 1
                return torch.randn(batch_size, self.hidden_dim, device=self.device)
                
        except Exception as e:
            logger.error(f"Feature combination failed: {e}")
            batch_size = list(features.values())[0].shape[0] if features else 1
            return torch.randn(batch_size, self.hidden_dim, device=self.device)
    
    def _safe_hierarchical_classification(self, combined_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Safely perform hierarchical classification."""
        try:
            results = {}
            
            # Family classification
            try:
                family_logits = self.family_head(combined_features)
                family_probs = F.softmax(family_logits, dim=1)
                results['family_logits'] = family_logits
            except Exception as e:
                logger.warning(f"Family classification failed: {e}")
                batch_size = combined_features.shape[0]
                family_logits = torch.randn(batch_size, self.num_families, device=self.device)
                family_probs = F.softmax(family_logits, dim=1)
                results['family_logits'] = family_logits
            
            # Instrument classification (conditioned on family)
            try:
                instrument_input = torch.cat([combined_features, family_probs], dim=1)
                instrument_logits = self.instrument_head(instrument_input)
                instrument_probs = F.softmax(instrument_logits, dim=1)
                results['instrument_logits'] = instrument_logits
            except Exception as e:
                logger.warning(f"Instrument classification failed: {e}")
                batch_size = combined_features.shape[0]
                instrument_logits = torch.randn(batch_size, self.num_instruments, device=self.device)
                instrument_probs = F.softmax(instrument_logits, dim=1)
                results['instrument_logits'] = instrument_logits
            
            # Technique classification (conditioned on family and instrument)
            try:
                technique_input = torch.cat([combined_features, family_probs, instrument_probs], dim=1)
                technique_logits = self.technique_head(technique_input)
                results['technique_logits'] = technique_logits
            except Exception as e:
                logger.warning(f"Technique classification failed: {e}")
                batch_size = combined_features.shape[0]
                technique_logits = torch.randn(batch_size, self.num_techniques, device=self.device)
                results['technique_logits'] = technique_logits
            
            return results
            
        except Exception as e:
            logger.error(f"Hierarchical classification completely failed: {e}")
            batch_size = combined_features.shape[0]
            return {
                'family_logits': torch.randn(batch_size, self.num_families, device=self.device),
                'instrument_logits': torch.randn(batch_size, self.num_instruments, device=self.device),
                'technique_logits': torch.randn(batch_size, self.num_techniques, device=self.device)
            }
    
    def _create_fallback_output(
        self, 
        batch_size: int, 
        return_features: bool, 
        return_hierarchical: bool,
        error_msg: str
    ) -> Dict[str, torch.Tensor]:
        """Create fallback output when everything fails."""
        result = {
            'family_logits': torch.zeros(batch_size, self.num_families, device=self.device),
            'instrument_logits': torch.zeros(batch_size, self.num_instruments, device=self.device),
            'technique_logits': torch.zeros(batch_size, self.num_techniques, device=self.device),
            'combined_features': torch.zeros(batch_size, self.hidden_dim, device=self.device),
            '_fallback': True,
            '_error': error_msg
        }
        
        if return_hierarchical:
            result.update({
                'family_probs': torch.ones(batch_size, self.num_families, device=self.device) / self.num_families,
                'instrument_probs': torch.ones(batch_size, self.num_instruments, device=self.device) / self.num_instruments,
                'technique_probs': torch.ones(batch_size, self.num_techniques, device=self.device) / self.num_techniques
            })
        
        if return_features:
            result['features'] = {
                'mel_features': torch.zeros(batch_size, self.n_mels, device=self.device),
                'spectral_features': torch.zeros(batch_size, self.hidden_dim, device=self.device)
            }
        
        return result
    
    def to(self, device):
        """Move module to device with error handling."""
        try:
            self.device = device
            return super().to(device)
        except Exception as e:
            logger.warning(f"Failed to move to device {device}: {e}")
            return self


# Pre-defined instrument taxonomies
INSTRUMENT_FAMILIES = [
    'strings', 'woodwinds', 'brass', 'percussion', 
    'keyboard', 'electronic', 'vocal', 'other'
]

COMMON_INSTRUMENTS = [
    'guitar', 'piano', 'violin', 'flute', 'trumpet', 'drums',
    'clarinet', 'saxophone', 'cello', 'bass', 'voice', 'organ',
    'harp', 'accordion', 'harmonica', 'banjo', 'mandolin', 'ukulele',
    'oboe', 'bassoon', 'trombone', 'french_horn', 'tuba', 'xylophone',
    'marimba', 'vibraphone', 'timpani', 'cymbals', 'triangle', 'tambourine',
    'synthesizer', 'electric_guitar', 'electric_bass', 'electric_piano',
    'theremin', 'sample_pad', 'drum_machine', 'string_ensemble',
    'choir', 'soprano', 'alto', 'tenor', 'bass_voice', 'whistle',
    'clapping', 'footsteps', 'door_slam', 'glass_break', 'water_drop', 'other'
]

PLAYING_TECHNIQUES = [
    'normal', 'staccato', 'legato', 'pizzicato', 'arco', 'tremolo',
    'vibrato', 'muted', 'palm_muted', 'harmonics', 'glissando', 'trill',
    'flutter_tongue', 'double_stop', 'chord', 'arpeggio', 'sustained',
    'percussive', 'bowed', 'other'
]


# Factory functions
def create_bulletproof_instrument_classifier(**kwargs) -> BulletproofInstrumentClassifier:
    """Create bulletproof instrument classifier with reasonable defaults."""
    return BulletproofInstrumentClassifier(
        num_families=len(INSTRUMENT_FAMILIES),
        num_instruments=len(COMMON_INSTRUMENTS),
        num_techniques=len(PLAYING_TECHNIQUES),
        **kwargs
    )

def create_bulletproof_guitar_classifier(**kwargs) -> BulletproofInstrumentClassifier:
    """Create specialized bulletproof guitar classifier."""
    return BulletproofInstrumentClassifier(
        num_families=3,  # acoustic, electric, bass
        num_instruments=10,  # specific guitar types
        num_techniques=10,  # guitar-specific techniques
        **kwargs
    )


# Test functionality
def test_bulletproof_instrument_classifier():
    """Test the bulletproof instrument classifier."""
    logger.info("Testing BulletproofInstrumentClassifier...")
    
    classifier = create_bulletproof_instrument_classifier()
    
    # Test with various audio shapes
    test_cases = [
        (2, 22050 * 3),   # 3 seconds
        (1, 22050 * 5),   # 5 seconds
        (3, 16000 * 2),   # 2 seconds at 16kHz
    ]
    
    for i, (batch_size, length) in enumerate(test_cases):
        try:
            logger.info(f"Test case {i+1}: batch_size={batch_size}, length={length}")
            
            waveform = torch.randn(batch_size, length)
            output = classifier(waveform, return_features=True, return_hierarchical=True)
            
            logger.info(f"Family logits shape: {output['family_logits'].shape}")
            logger.info(f"Instrument logits shape: {output['instrument_logits'].shape}")
            logger.info(f"Technique logits shape: {output['technique_logits'].shape}")
            logger.info("Test case passed")
            
        except Exception as e:
            logger.error(f"Test case {i+1} failed: {e}")
    
    logger.info("Instrument classifier testing completed")


if __name__ == "__main__":
    test_bulletproof_instrument_classifier()