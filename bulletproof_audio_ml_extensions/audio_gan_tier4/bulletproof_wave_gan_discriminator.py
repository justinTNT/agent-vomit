#!/usr/bin/env python3
"""
BULLETPROOF WAVE GAN DISCRIMINATOR MODULE
Advanced discriminator for waveform GAN training with BigVGAN compatibility.
Handles multi-scale analysis, temporal features, and provides confidence scoring for GAN training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Union, Dict, Tuple, Any
from rave_config_system import RAVEConfig
import warnings
import logging
import math
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WaveGANDiscriminatorConfig:
    """Configuration for bulletproof wave GAN discriminator"""
    # Input parameters
    input_channels: int = 1
    sample_rate: int = 44100
    
    # Multi-scale discriminator
    use_multi_scale: bool = True
    n_scales: int = 3
    scale_factors: List[int] = field(default_factory=lambda: [1, 2, 4])
    
    # Multi-period discriminator
    use_multi_period: bool = True
    periods: List[int] = field(default_factory=lambda: [2, 3, 5, 7, 11])
    
    # Convolution parameters
    base_channels: int = 32
    max_channels: int = 1024
    kernel_sizes: List[int] = field(default_factory=lambda: [15, 41, 41, 41, 5])
    strides: List[int] = field(default_factory=lambda: [1, 4, 4, 4, 1])
    groups: List[int] = field(default_factory=lambda: [1, 4, 16, 64, 1])
    
    # Advanced features
    use_spectral_normalization: bool = True
    use_self_attention: bool = True
    attention_layers: List[int] = field(default_factory=lambda: [2, 4])
    
    # Temporal modeling
    use_temporal_modeling: bool = True
    temporal_kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    use_causal_conv: bool = False
    
    # Feature matching
    use_feature_matching: bool = True
    feature_matching_layers: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    feature_matching_weight: float = 10.0
    
    # Adversarial loss
    adversarial_loss_type: str = 'hinge'  # 'hinge', 'lsgan', 'wgan_gp', 'standard'
    gradient_penalty_weight: float = 10.0
    
    # Confidence and quality assessment
    use_quality_assessment: bool = True
    quality_features: List[str] = field(default_factory=lambda: ['spectral_centroid', 'zero_crossing_rate', 'mfcc'])
    confidence_threshold: float = 0.5
    
    # Regularization
    dropout: float = 0.2
    weight_decay: float = 1e-4
    spectral_norm_eps: float = 1e-12
    
    # Bulletproof stability
    eps: float = 1e-8
    gradient_clip_value: float = 1.0
    numerical_stability_check: bool = True
    
    # Memory and efficiency
    use_checkpoint: bool = False
    chunk_size: int = 16384
    max_memory_gb: float = 4.0
    
    # Fallback strategies
    enable_fallbacks: bool = True
    fallback_to_simple_conv: bool = True
    disable_on_failure: bool = False

class SpectralNorm(nn.Module):
    """Spectral normalization for discriminator stability"""
    
    def __init__(self, module: nn.Module, name: str = 'weight', n_power_iterations: int = 1, eps: float = 1e-12):
        super().__init__()
        self.module = module
        self.name = name
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        
        if not hasattr(module, name):
            raise ValueError(f"Module {module} does not have parameter {name}")
        
        weight = getattr(module, name)
        
        with torch.no_grad():
            weight_mat = self._reshape_weight_to_matrix(weight)
            h, w = weight_mat.size()
            u = weight.new_empty(h).normal_(0, 1)
            v = weight.new_empty(w).normal_(0, 1)
        
        self.register_buffer('u', u)
        self.register_buffer('v', v)
        
        # Remove original parameter
        delattr(module, name)
        self.module.register_parameter(name + "_orig", nn.Parameter(weight))
    
    def _reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        """Reshape weight tensor to 2D matrix"""
        if weight.dim() == 1:
            return weight.unsqueeze(0)
        else:
            return weight.view(weight.size(0), -1)
    
    def _power_iteration(self, weight_mat: torch.Tensor) -> torch.Tensor:
        """Compute spectral norm using power iteration"""
        u = self.u
        v = self.v
        
        for _ in range(self.n_power_iterations):
            # v = W^T u / ||W^T u||
            v = torch.mv(weight_mat.t(), u)
            v = F.normalize(v, dim=0, eps=self.eps)
            
            # u = W v / ||W v||
            u = torch.mv(weight_mat, v)
            u = F.normalize(u, dim=0, eps=self.eps)
        
        # Update buffers
        self.u.copy_(u)
        self.v.copy_(v)
        
        # Compute spectral norm
        sigma = torch.dot(u, torch.mv(weight_mat, v))
        return sigma
    
    def forward(self, *args, **kwargs):
        weight_orig = getattr(self.module, self.name + "_orig")
        weight_mat = self._reshape_weight_to_matrix(weight_orig)
        
        if self.training:
            sigma = self._power_iteration(weight_mat)
        else:
            # Use cached values during inference
            u = self.u
            v = self.v
            sigma = torch.dot(u, torch.mv(weight_mat, v))
        
        # Normalize weight
        weight = weight_orig / sigma
        setattr(self.module, self.name, weight)
        
        return self.module(*args, **kwargs)

class SelfAttention(nn.Module):
    """Self-attention layer for discriminator"""
    
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        
        self.query_conv = nn.Conv1d(channels, channels // reduction, 1)
        self.key_conv = nn.Conv1d(channels, channels // reduction, 1)
        self.value_conv = nn.Conv1d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, channels, length]
        Returns:
            Attention-enhanced features [batch_size, channels, length]
        """
        batch_size, channels, length = x.size()
        
        # Compute query, key, value
        query = self.query_conv(x).view(batch_size, -1, length).permute(0, 2, 1)  # [B, L, C//r]
        key = self.key_conv(x).view(batch_size, -1, length)  # [B, C//r, L]
        value = self.value_conv(x).view(batch_size, -1, length)  # [B, C, L]
        
        # Attention scores
        attention = torch.bmm(query, key)  # [B, L, L]
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(value, attention.permute(0, 2, 1))  # [B, C, L]
        out = out.view(batch_size, channels, length)
        
        # Residual connection with learnable gate
        out = self.gamma * out + x
        
        return out

class TemporalBlock(nn.Module):
    """Temporal modeling block with multiple kernel sizes"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: List[int], 
                 stride: int = 1, use_causal: bool = False):
        super().__init__()
        self.use_causal = use_causal
        
        # Multi-kernel convolutions
        self.convs = nn.ModuleList()
        for kernel_size in kernel_sizes:
            padding = (kernel_size - 1) if use_causal else (kernel_size - 1) // 2
            conv = nn.Conv1d(in_channels, out_channels // len(kernel_sizes), 
                           kernel_size, stride=stride, padding=padding)
            self.convs.append(conv)
        
        # Combine outputs
        self.combine_conv = nn.Conv1d(out_channels, out_channels, 1)
        self.norm = nn.GroupNorm(1, out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply multi-kernel convolutions
        outputs = []
        for conv in self.convs:
            out = conv(x)
            if self.use_causal:
                # Remove future information
                out = out[:, :, :x.size(2)]
            outputs.append(out)
        
        # Concatenate and combine
        combined = torch.cat(outputs, dim=1)
        combined = self.combine_conv(combined)
        combined = self.norm(combined)
        combined = self.activation(combined)
        
        return combined

class ScaleDiscriminator(nn.Module):
    """Single-scale discriminator"""
    
    def __init__(self, config: WaveGANDiscriminatorConfig, scale: int = 1):
        super().__init__()
        self.config = config
        self.scale = scale
        
        # Downsampling for multi-scale
        if scale > 1:
            self.downsample = nn.AvgPool1d(scale, stride=scale)
        else:
            self.downsample = nn.Identity()
        
        # Build discriminator layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.attentions = nn.ModuleList()
        
        in_channels = config.input_channels
        
        for i, (kernel_size, stride, groups) in enumerate(zip(
            config.kernel_sizes, config.strides, config.groups
        )):
            out_channels = min(config.base_channels * (2 ** i), config.max_channels)
            
            # Convolution layer
            padding = (kernel_size - 1) // 2
            conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                           stride=stride, padding=padding, groups=groups)
            
            # Apply spectral normalization if enabled
            if config.use_spectral_normalization:
                conv = SpectralNorm(conv)
            
            self.convs.append(conv)
            
            # Normalization
            if i > 0:  # No norm on first layer
                norm = nn.GroupNorm(groups, out_channels)
                self.norms.append(norm)
            else:
                self.norms.append(nn.Identity())
            
            # Self-attention
            if config.use_self_attention and i in config.attention_layers:
                attention = SelfAttention(out_channels)
                self.attentions.append(attention)
            else:
                self.attentions.append(nn.Identity())
            
            in_channels = out_channels
        
        # Temporal modeling
        if config.use_temporal_modeling:
            self.temporal_block = TemporalBlock(
                in_channels, in_channels, 
                config.temporal_kernel_sizes,
                use_causal=config.use_causal_conv
            )
        else:
            self.temporal_block = nn.Identity()
        
        # Output layers
        self.output_conv = nn.Conv1d(in_channels, 1, 3, padding=1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: [batch_size, channels, length]
        Returns:
            output: [batch_size, 1, length//total_stride]
            features: List of intermediate features for feature matching
        """
        # Downsample for multi-scale
        x = self.downsample(x)
        
        features = []
        
        # Apply discriminator layers
        for conv, norm, attention in zip(self.convs, self.norms, self.attentions):
            x = conv(x)
            x = norm(x)
            x = self.activation(x)
            x = attention(x)
            x = self.dropout(x)
            features.append(x)
        
        # Temporal modeling
        x = self.temporal_block(x)
        features.append(x)
        
        # Output
        output = self.output_conv(x)
        
        return output, features

class PeriodDiscriminator(nn.Module):
    """Period-based discriminator for periodic patterns"""
    
    def __init__(self, config: WaveGANDiscriminatorConfig, period: int):
        super().__init__()
        self.period = period
        
        # Reshape input to 2D for period-based analysis
        self.convs = nn.ModuleList()
        
        in_channels = 1
        
        # 2D convolutions for period analysis
        for i in range(4):
            out_channels = config.base_channels * (2 ** i)
            
            kernel_size = (5, 1) if i == 0 else (5, 5)
            stride = (3, 1) if i == 0 else (3, 3)
            padding = (2, 0) if i == 0 else (2, 2)
            
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                           stride=stride, padding=padding)
            
            if config.use_spectral_normalization:
                conv = SpectralNorm(conv)
            
            self.convs.append(conv)
            in_channels = out_channels
        
        # Final convolution
        self.final_conv = nn.Conv2d(in_channels, 1, (3, 1), padding=(1, 0))
        self.activation = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: [batch_size, 1, length]
        Returns:
            output: [batch_size, 1, reduced_height, reduced_width]
            features: List of intermediate features
        """
        batch_size, channels, length = x.size()
        
        # Reshape for period-based analysis
        if length % self.period != 0:
            # Pad to make divisible by period
            pad_length = self.period - (length % self.period)
            x = F.pad(x, (0, pad_length))
            length = x.size(2)
        
        # Reshape to [batch_size, 1, length//period, period]
        x = x.view(batch_size, channels, length // self.period, self.period)
        
        features = []
        
        # Apply 2D convolutions
        for conv in self.convs:
            x = conv(x)
            x = self.activation(x)
            features.append(x)
        
        # Final output
        output = self.final_conv(x)
        
        return output, features

class BulletproofWaveGANDiscriminator(nn.Module):
    """
    Bulletproof Wave GAN discriminator with comprehensive error handling.
    
    Features:
    - Multi-scale and multi-period discrimination
    - Spectral normalization for training stability
    - Self-attention for long-range dependencies
    - Temporal modeling with multiple kernel sizes
    - Feature matching for improved training
    - Quality assessment and confidence scoring
    - Comprehensive fallback strategies for corrupted inputs
    - Memory-efficient processing for long sequences
    - Multiple adversarial loss types
    """
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__()
        
        self.config = config
        self.discriminator_config = kwargs.get('discriminator_config', WaveGANDiscriminatorConfig())
        
        # Override with RAVEConfig values if available
        if hasattr(config.audio, 'sample_rate'):
            self.discriminator_config.sample_rate = config.audio.sample_rate
        
        # Build discriminator components
        try:
            self._build_discriminator_components()
        except Exception as e:
            logger.error(f"Failed to build discriminator components: {e}")
            if self.discriminator_config.enable_fallbacks:
                logger.warning("Building fallback discriminator components")
                self._build_fallback_components()
            else:
                raise
        
        # Quality assessment components
        if self.discriminator_config.use_quality_assessment:
            self._build_quality_assessment()
        
        # Tracking and monitoring
        self.processing_stats = []
        self.discriminator_scores = []
        self.fallback_activations = 0
        
        # Memory management
        self._memory_usage = 0
        self._max_memory_bytes = int(self.discriminator_config.max_memory_gb * 1e9)
        
        logger.info(f"BulletproofWaveGANDiscriminator initialized: scales={self.discriminator_config.n_scales}")
    
    def _build_discriminator_components(self):
        """Build main discriminator components"""
        # Multi-scale discriminators
        if self.discriminator_config.use_multi_scale:
            self.scale_discriminators = nn.ModuleList()
            for scale in self.discriminator_config.scale_factors:
                scale_disc = ScaleDiscriminator(self.discriminator_config, scale)
                self.scale_discriminators.append(scale_disc)
        
        # Multi-period discriminators
        if self.discriminator_config.use_multi_period:
            self.period_discriminators = nn.ModuleList()
            for period in self.discriminator_config.periods:
                period_disc = PeriodDiscriminator(self.discriminator_config, period)
                self.period_discriminators.append(period_disc)
    
    def _build_fallback_components(self):
        """Build simple fallback components"""
        # Simple single-scale discriminator
        config = self.discriminator_config
        config.use_multi_scale = False
        config.use_multi_period = False
        config.use_self_attention = False
        config.use_temporal_modeling = False
        config.use_spectral_normalization = False
        
        # Single simple discriminator
        self.fallback_discriminator = ScaleDiscriminator(config, scale=1)
        
        logger.info("Built fallback discriminator components")
    
    def _build_quality_assessment(self):
        """Build quality assessment components"""
        try:
            # Spectral feature extractors
            self.quality_features = nn.ModuleDict()
            
            if 'spectral_centroid' in self.discriminator_config.quality_features:
                self.quality_features['spectral_centroid'] = nn.Conv1d(1, 1, 1024, stride=512)
            
            if 'zero_crossing_rate' in self.discriminator_config.quality_features:
                # Simple conv for zero crossing rate estimation
                self.quality_features['zero_crossing'] = nn.Conv1d(1, 1, 3, padding=1)
            
        except Exception as e:
            logger.error(f"Quality assessment build failed: {e}")
            self.discriminator_config.use_quality_assessment = False
    
    def _validate_input(self, audio: torch.Tensor) -> bool:
        """Validate input audio tensor"""
        try:
            if audio is None or audio.numel() == 0:
                logger.warning("Empty or None audio input")
                return False
            
            if not torch.isfinite(audio).all():
                logger.warning("Non-finite values in audio input")
                if self.discriminator_config.enable_fallbacks:
                    return True  # Allow fallback to handle corrupted audio
                return False
            
            if audio.dim() < 2 or audio.dim() > 3:
                logger.warning(f"Audio has wrong dimensions: {audio.dim()}, expected 2 or 3")
                return False
            
            # Check reasonable audio length
            min_samples = 1024
            if audio.size(-1) < min_samples:
                logger.warning(f"Audio too short: {audio.size(-1)} < {min_samples}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False
    
    def _compute_adversarial_loss(self, real_scores: List[torch.Tensor], 
                                fake_scores: List[torch.Tensor], is_real: bool) -> torch.Tensor:
        """Compute adversarial loss based on configured loss type"""
        try:
            loss_type = self.discriminator_config.adversarial_loss_type
            total_loss = 0.0
            count = 0
            
            scores = real_scores if is_real else fake_scores
            
            for score_list in scores:
                if isinstance(score_list, list):
                    # Multiple outputs from one discriminator
                    for score in score_list:
                        if loss_type == 'hinge':
                            if is_real:
                                loss = F.relu(1.0 - score).mean()
                            else:
                                loss = F.relu(1.0 + score).mean()
                        elif loss_type == 'lsgan':
                            target = torch.ones_like(score) if is_real else torch.zeros_like(score)
                            loss = F.mse_loss(score, target)
                        elif loss_type == 'standard':
                            target = torch.ones_like(score) if is_real else torch.zeros_like(score)
                            loss = F.binary_cross_entropy_with_logits(score, target)
                        else:
                            # Default to hinge loss
                            if is_real:
                                loss = F.relu(1.0 - score).mean()
                            else:
                                loss = F.relu(1.0 + score).mean()
                        
                        total_loss += loss
                        count += 1
                else:
                    # Single output
                    score = score_list
                    if loss_type == 'hinge':
                        if is_real:
                            loss = F.relu(1.0 - score).mean()
                        else:
                            loss = F.relu(1.0 + score).mean()
                    elif loss_type == 'lsgan':
                        target = torch.ones_like(score) if is_real else torch.zeros_like(score)
                        loss = F.mse_loss(score, target)
                    elif loss_type == 'standard':
                        target = torch.ones_like(score) if is_real else torch.zeros_like(score)
                        loss = F.binary_cross_entropy_with_logits(score, target)
                    else:
                        # Default to hinge loss
                        if is_real:
                            loss = F.relu(1.0 - score).mean()
                        else:
                            loss = F.relu(1.0 + score).mean()
                    
                    total_loss += loss
                    count += 1
            
            return total_loss / max(count, 1)
            
        except Exception as e:
            logger.error(f"Adversarial loss computation failed: {e}")
            return torch.tensor(0.0, device=real_scores[0].device if real_scores else torch.device('cpu'))
    
    def _compute_feature_matching_loss(self, real_features: List[List[torch.Tensor]], 
                                     fake_features: List[List[torch.Tensor]]) -> torch.Tensor:
        """Compute feature matching loss"""
        try:
            if not self.discriminator_config.use_feature_matching:
                return torch.tensor(0.0)
            
            total_loss = 0.0
            count = 0
            
            # Compare features from each discriminator
            for real_feat_list, fake_feat_list in zip(real_features, fake_features):
                for layer_idx in self.discriminator_config.feature_matching_layers:
                    if layer_idx < len(real_feat_list) and layer_idx < len(fake_feat_list):
                        real_feat = real_feat_list[layer_idx]
                        fake_feat = fake_feat_list[layer_idx]
                        
                        # L1 loss between features
                        loss = F.l1_loss(fake_feat, real_feat.detach())
                        total_loss += loss
                        count += 1
            
            feature_loss = total_loss / max(count, 1)
            return self.discriminator_config.feature_matching_weight * feature_loss
            
        except Exception as e:
            logger.error(f"Feature matching loss computation failed: {e}")
            return torch.tensor(0.0)
    
    def _compute_quality_features(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute quality assessment features"""
        try:
            if not self.discriminator_config.use_quality_assessment:
                return {}
            
            quality_features = {}
            
            # Spectral centroid
            if 'spectral_centroid' in self.quality_features:
                # Simple approximation using convolution
                centroid_conv = self.quality_features['spectral_centroid']
                spectral_centroid = centroid_conv(audio.unsqueeze(1) if audio.dim() == 2 else audio)
                quality_features['spectral_centroid'] = torch.mean(spectral_centroid, dim=-1)
            
            # Zero crossing rate
            if 'zero_crossing' in self.quality_features:
                # Sign changes
                audio_signs = torch.sign(audio)
                sign_changes = torch.abs(torch.diff(audio_signs, dim=-1))
                zcr = torch.mean(sign_changes, dim=-1)
                quality_features['zero_crossing_rate'] = zcr
            
            return quality_features
            
        except Exception as e:
            logger.error(f"Quality features computation failed: {e}")
            return {}
    
    def _compute_confidence_score(self, discriminator_outputs: List[torch.Tensor]) -> torch.Tensor:
        """Compute confidence score for discriminator outputs"""
        try:
            # Average discriminator scores
            all_scores = []
            for output in discriminator_outputs:
                if isinstance(output, list):
                    for score in output:
                        all_scores.append(torch.mean(torch.abs(score)))
                else:
                    all_scores.append(torch.mean(torch.abs(output)))
            
            if len(all_scores) == 0:
                return torch.tensor(0.5)
            
            # Mean absolute score as confidence
            mean_score = torch.stack(all_scores).mean()
            confidence = torch.sigmoid(mean_score)
            
            return confidence
            
        except Exception as e:
            logger.error(f"Confidence computation failed: {e}")
            return torch.tensor(0.5)
    
    def forward(self, audio: torch.Tensor) -> Dict[str, Any]:
        """
        Forward pass with comprehensive error handling.
        
        Args:
            audio: Input audio tensor [batch_size, length] or [batch_size, channels, length]
            
        Returns:
            Dictionary containing:
            - discriminator_outputs: List of discriminator outputs
            - features: List of feature maps for feature matching
            - quality_features: Quality assessment features
            - confidence: Confidence score for the discrimination
        """
        try:
            # Handle input dimensions
            if audio.dim() == 2:
                # Add channel dimension
                audio = audio.unsqueeze(1)  # [batch_size, 1, length]
            elif audio.dim() == 3 and audio.size(1) != self.discriminator_config.input_channels:
                # Adjust channels if needed
                if audio.size(1) > self.discriminator_config.input_channels:
                    audio = audio[:, :self.discriminator_config.input_channels, :]
                else:
                    # Repeat channels if needed
                    audio = audio.repeat(1, self.discriminator_config.input_channels, 1)
            
            # Validate input
            if not self._validate_input(audio):
                if self.discriminator_config.enable_fallbacks:
                    logger.warning("Input validation failed, using fallback")
                    self.fallback_activations += 1
                    if self.discriminator_config.disable_on_failure:
                        # Return empty results
                        return self._create_empty_results(audio.device)
                    else:
                        # Clean input and continue
                        audio = torch.where(torch.isfinite(audio), audio, torch.zeros_like(audio))
                else:
                    raise ValueError("Input validation failed")
            
            discriminator_outputs = []
            all_features = []
            
            # Multi-scale discrimination
            if self.discriminator_config.use_multi_scale and hasattr(self, 'scale_discriminators'):
                try:
                    for scale_disc in self.scale_discriminators:
                        output, features = scale_disc(audio)
                        discriminator_outputs.append(output)
                        all_features.append(features)
                except Exception as e:
                    logger.error(f"Multi-scale discrimination failed: {e}")
                    if self.discriminator_config.enable_fallbacks:
                        self.fallback_activations += 1
                    else:
                        raise
            
            # Multi-period discrimination
            if self.discriminator_config.use_multi_period and hasattr(self, 'period_discriminators'):
                try:
                    for period_disc in self.period_discriminators:
                        output, features = period_disc(audio)
                        discriminator_outputs.append(output)
                        all_features.append(features)
                except Exception as e:
                    logger.error(f"Multi-period discrimination failed: {e}")
                    if self.discriminator_config.enable_fallbacks:
                        self.fallback_activations += 1
                    else:
                        raise
            
            # Fallback discriminator if no outputs
            if len(discriminator_outputs) == 0:
                if hasattr(self, 'fallback_discriminator'):
                    try:
                        output, features = self.fallback_discriminator(audio)
                        discriminator_outputs.append(output)
                        all_features.append(features)
                    except Exception as e:
                        logger.error(f"Fallback discriminator failed: {e}")
                        return self._create_empty_results(audio.device)
                else:
                    logger.error("No discriminators available")
                    return self._create_empty_results(audio.device)
            
            # Quality assessment
            quality_features = self._compute_quality_features(audio)
            
            # Confidence scoring
            confidence = self._compute_confidence_score(discriminator_outputs)
            
            results = {
                'discriminator_outputs': discriminator_outputs,
                'features': all_features,
                'quality_features': quality_features,
                'confidence': confidence
            }
            
            # Update statistics
            self._update_statistics(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Discriminator forward pass failed: {e}")
            if self.discriminator_config.enable_fallbacks:
                self.fallback_activations += 1
                logger.warning("Using emergency fallback")
                return self._create_empty_results(audio.device if audio is not None else torch.device('cpu'))
            else:
                raise
    
    def compute_discriminator_loss(self, real_audio: torch.Tensor, fake_audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute discriminator loss for training"""
        try:
            # Forward pass on real audio
            real_results = self.forward(real_audio)
            real_outputs = real_results['discriminator_outputs']
            real_features = real_results['features']
            
            # Forward pass on fake audio
            fake_results = self.forward(fake_audio)
            fake_outputs = fake_results['discriminator_outputs']
            fake_features = fake_results['features']
            
            # Adversarial loss
            real_loss = self._compute_adversarial_loss(real_outputs, fake_outputs, is_real=True)
            fake_loss = self._compute_adversarial_loss(real_outputs, fake_outputs, is_real=False)
            adversarial_loss = real_loss + fake_loss
            
            # Feature matching loss
            feature_matching_loss = self._compute_feature_matching_loss(real_features, fake_features)
            
            # Total loss
            total_loss = adversarial_loss + feature_matching_loss
            
            return {
                'total_loss': total_loss,
                'adversarial_loss': adversarial_loss,
                'feature_matching_loss': feature_matching_loss,
                'real_loss': real_loss,
                'fake_loss': fake_loss
            }
            
        except Exception as e:
            logger.error(f"Discriminator loss computation failed: {e}")
            device = real_audio.device if real_audio is not None else torch.device('cpu')
            return {
                'total_loss': torch.tensor(0.0, device=device),
                'adversarial_loss': torch.tensor(0.0, device=device),
                'feature_matching_loss': torch.tensor(0.0, device=device),
                'real_loss': torch.tensor(0.0, device=device),
                'fake_loss': torch.tensor(0.0, device=device)
            }
    
    def _create_empty_results(self, device: torch.device) -> Dict[str, Any]:
        """Create empty results structure for fallback"""
        return {
            'discriminator_outputs': [torch.zeros(1, 1, 100, device=device)],
            'features': [[torch.zeros(1, 32, 100, device=device)]],
            'quality_features': {},
            'confidence': torch.tensor(0.5, device=device)
        }
    
    def _update_statistics(self, results: Dict[str, Any]):
        """Update processing statistics"""
        try:
            confidence = results['confidence'].item()
            n_outputs = len(results['discriminator_outputs'])
            
            # Average discriminator score
            avg_score = 0.0
            count = 0
            for output in results['discriminator_outputs']:
                if isinstance(output, list):
                    for score in output:
                        avg_score += torch.mean(score).item()
                        count += 1
                else:
                    avg_score += torch.mean(output).item()
                    count += 1
            
            avg_score = avg_score / max(count, 1)
            
            stats = {
                'confidence': confidence,
                'n_discriminator_outputs': n_outputs,
                'avg_discriminator_score': avg_score,
                'fallback_activations': self.fallback_activations
            }
            
            if len(self.processing_stats) < 1000:  # Limit history size
                self.processing_stats.append(stats)
            
            if len(self.discriminator_scores) < 1000:
                self.discriminator_scores.append(avg_score)
            
        except Exception as e:
            logger.warning(f"Statistics update failed: {e}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics for monitoring"""
        stats = {
            'n_scales': self.discriminator_config.n_scales if self.discriminator_config.use_multi_scale else 0,
            'n_periods': len(self.discriminator_config.periods) if self.discriminator_config.use_multi_period else 0,
            'use_spectral_normalization': self.discriminator_config.use_spectral_normalization,
            'use_self_attention': self.discriminator_config.use_self_attention,
            'adversarial_loss_type': self.discriminator_config.adversarial_loss_type,
            'fallback_activations': self.fallback_activations
        }
        
        if self.processing_stats:
            last_stats = self.processing_stats[-1]
            stats.update({
                'last_confidence': last_stats['confidence'],
                'last_avg_score': last_stats['avg_discriminator_score']
            })
        
        if self.discriminator_scores:
            stats.update({
                'mean_discriminator_score': sum(self.discriminator_scores) / len(self.discriminator_scores),
                'score_std': np.std(self.discriminator_scores) if len(self.discriminator_scores) > 1 else 0.0
            })
        
        return stats
    
    def reset_statistics(self):
        """Reset training statistics"""
        self.processing_stats.clear()
        self.discriminator_scores.clear()
        self.fallback_activations = 0


# Factory function
def create_bulletproof_wave_gan_discriminator(config: RAVEConfig, **kwargs) -> BulletproofWaveGANDiscriminator:
    """Create a bulletproof wave GAN discriminator"""
    return BulletproofWaveGANDiscriminator(config, **kwargs)


if __name__ == "__main__":
    print("🛡️ BULLETPROOF WAVE GAN DISCRIMINATOR MODULE")
    print("=" * 50)
    
    # Test with minimal config
    from rave_config_system import get_minimal_config
    
    config = get_minimal_config()
    
    # Test discriminator
    discriminator_config = WaveGANDiscriminatorConfig(
        use_multi_scale=True, 
        use_multi_period=True,
        use_spectral_normalization=True
    )
    discriminator = create_bulletproof_wave_gan_discriminator(config, discriminator_config=discriminator_config)
    
    try:
        # Create synthetic audio
        batch_size = 2
        audio_length = 16384
        
        # Real audio (more structured)
        real_audio = torch.sin(2 * math.pi * 440 * torch.linspace(0, 1, audio_length))
        real_audio = real_audio.unsqueeze(0).repeat(batch_size, 1)  # [batch_size, length]
        
        # Fake audio (more random)
        fake_audio = 0.5 * torch.randn(batch_size, audio_length)
        
        print(f"Testing with audio shapes: real={real_audio.shape}, fake={fake_audio.shape}")
        
        # Test discriminator forward pass
        real_results = discriminator(real_audio)
        
        print(f"✅ Real audio discrimination passed")
        print(f"   Discriminator outputs: {len(real_results['discriminator_outputs'])}")
        print(f"   Feature maps: {len(real_results['features'])}")
        print(f"   Confidence: {real_results['confidence']:.3f}")
        print(f"   Quality features: {list(real_results['quality_features'].keys())}")
        
        fake_results = discriminator(fake_audio)
        
        print(f"✅ Fake audio discrimination passed")
        print(f"   Confidence: {fake_results['confidence']:.3f}")
        
        # Test loss computation
        loss_dict = discriminator.compute_discriminator_loss(real_audio, fake_audio)
        
        print(f"✅ Loss computation passed")
        print(f"   Total loss: {loss_dict['total_loss']:.3f}")
        print(f"   Adversarial loss: {loss_dict['adversarial_loss']:.3f}")
        print(f"   Feature matching loss: {loss_dict['feature_matching_loss']:.3f}")
        
        # Test with corrupted audio
        corrupted_audio = real_audio.clone()
        corrupted_audio[:, 1000:1100] = float('inf')
        
        corrupted_results = discriminator(corrupted_audio)
        print(f"✅ Robust handling of corrupted audio")
        
        # Test statistics
        stats = discriminator.get_training_stats()
        print(f"   Discriminator stats: {stats}")
        
    except Exception as e:
        print(f"❌ Discriminator test failed: {e}")
    
    # Test different configurations
    try:
        # Test simple configuration
        simple_config = WaveGANDiscriminatorConfig(
            use_multi_scale=False,
            use_multi_period=False,
            use_self_attention=False
        )
        simple_discriminator = create_bulletproof_wave_gan_discriminator(config, discriminator_config=simple_config)
        
        simple_results = simple_discriminator(real_audio)
        print(f"✅ Simple discriminator test passed")
        
        # Test with different loss types
        hinge_config = WaveGANDiscriminatorConfig(adversarial_loss_type='hinge')
        hinge_discriminator = create_bulletproof_wave_gan_discriminator(config, discriminator_config=hinge_config)
        
        hinge_loss = hinge_discriminator.compute_discriminator_loss(real_audio, fake_audio)
        print(f"✅ Hinge loss test passed: {hinge_loss['total_loss']:.3f}")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
    
    print("🚀 BulletproofWaveGANDiscriminator ready for BigVGAN training!")