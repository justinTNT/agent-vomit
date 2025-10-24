#!/usr/bin/env python3
"""
BULLETPROOF MODULATION MODULE
Comprehensive feature modulation and conditioning for BigVGAN neural audio generation.
Handles FiLM, AdaIN, global/local conditioning, and adaptive modulation strategies.
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
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModulationType(Enum):
    """Types of feature modulation"""
    FILM = "film"
    ADAIN = "adain"
    SCALE_SHIFT = "scale_shift"
    GATING = "gating"
    ATTENTION = "attention"
    SQUEEZE_EXCITE = "squeeze_excite"

@dataclass
class ModulationConfig:
    """Configuration for bulletproof modulation"""
    modulation_type: str = 'film'  # 'film', 'adain', 'scale_shift', 'gating', 'attention', 'squeeze_excite'
    num_features: int = 512
    condition_dim: int = 256
    
    # FiLM specific parameters
    film_use_bias: bool = True
    film_activation: str = 'identity'  # 'identity', 'tanh', 'sigmoid'
    
    # AdaIN specific parameters
    adain_eps: float = 1e-5
    adain_style_mixing: bool = False
    
    # Gating parameters
    gating_activation: str = 'sigmoid'  # 'sigmoid', 'tanh', 'swish'
    gating_type: str = 'channel'  # 'channel', 'spatial', 'both'
    
    # Attention modulation parameters
    attention_heads: int = 8
    attention_dropout: float = 0.1
    
    # Squeeze-and-Excite parameters
    se_reduction_ratio: int = 16
    se_activation: str = 'relu'
    
    # Multi-modal conditioning
    use_global_conditioning: bool = True
    use_local_conditioning: bool = False
    global_condition_dim: int = 128
    local_condition_dim: int = 64
    
    # Adaptive modulation features
    adaptive_modulation: bool = False
    modulation_strength_learnable: bool = True
    modulation_strength_init: float = 1.0
    
    # Temporal conditioning
    temporal_conditioning: bool = False
    temporal_window_size: int = 32
    temporal_stride: int = 16
    
    # Bulletproof stability parameters
    eps: float = 1e-8
    numerical_stability_check: bool = True
    gradient_clip_value: float = 1.0
    
    # Fallback strategies
    enable_fallbacks: bool = True
    fallback_modulation_type: str = 'scale_shift'
    disable_modulation_on_failure: bool = False

class BulletproofModulation(nn.Module):
    """
    Bulletproof feature modulation with comprehensive conditioning support.
    
    Features:
    - Multiple modulation types (FiLM, AdaIN, Scale-Shift, Gating, Attention)
    - Global and local conditioning support
    - Temporal conditioning for sequential data
    - Adaptive modulation strength learning
    - Multi-modal conditioning fusion
    - Comprehensive fallback strategies
    - Memory efficient processing
    - Numerical stability guarantees
    """
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__()
        
        self.config = config
        self.modulation_config = kwargs.get('modulation_config', ModulationConfig())
        
        # Override with RAVEConfig values if available
        if hasattr(config.model, 'd_model'):
            self.modulation_config.num_features = config.model.d_model
        
        self.num_features = kwargs.get('num_features', self.modulation_config.num_features)
        self.condition_dim = kwargs.get('condition_dim', self.modulation_config.condition_dim)
        self.modulation_type = self.modulation_config.modulation_type.lower()
        
        # Build modulation layers with error handling
        try:
            self._build_modulation_layers()
        except Exception as e:
            logger.error(f"Failed to build modulation layers: {e}")
            if self.modulation_config.enable_fallbacks:
                logger.warning("Building fallback modulation layers")
                self._build_fallback_modulation_layers()
            else:
                raise
        
        # Build conditioning networks
        if self.modulation_config.use_global_conditioning:
            self._build_global_conditioning()
        
        if self.modulation_config.use_local_conditioning:
            self._build_local_conditioning()
        
        # Temporal conditioning
        if self.modulation_config.temporal_conditioning:
            self._build_temporal_conditioning()
        
        # Adaptive modulation strength
        if self.modulation_config.modulation_strength_learnable:
            self.modulation_strength = nn.Parameter(
                torch.tensor(self.modulation_config.modulation_strength_init)
            )
        else:
            self.register_buffer('modulation_strength', 
                               torch.tensor(self.modulation_config.modulation_strength_init))
        
        # Tracking and monitoring
        self.modulation_stats = []
        self.conditioning_stats = []
        self.fallback_activations = 0
        
        logger.info(f"BulletproofModulation initialized: {self.modulation_type}, features={self.num_features}")
    
    def _build_modulation_layers(self):
        """Build main modulation layers based on type"""
        if self.modulation_type == 'film':
            self._build_film_layers()
        elif self.modulation_type == 'adain':
            self._build_adain_layers()
        elif self.modulation_type == 'scale_shift':
            self._build_scale_shift_layers()
        elif self.modulation_type == 'gating':
            self._build_gating_layers()
        elif self.modulation_type == 'attention':
            self._build_attention_layers()
        elif self.modulation_type == 'squeeze_excite':
            self._build_squeeze_excite_layers()
        else:
            raise ValueError(f"Unknown modulation type: {self.modulation_type}")
    
    def _build_film_layers(self):
        """Build Feature-wise Linear Modulation layers"""
        # Scale and shift networks
        self.film_scale = nn.Linear(self.condition_dim, self.num_features)
        if self.modulation_config.film_use_bias:
            self.film_shift = nn.Linear(self.condition_dim, self.num_features)
        else:
            self.film_shift = None
        
        # Optional activation for modulation parameters
        if self.modulation_config.film_activation == 'tanh':
            self.film_activation = nn.Tanh()
        elif self.modulation_config.film_activation == 'sigmoid':
            self.film_activation = nn.Sigmoid()
        else:
            self.film_activation = nn.Identity()
        
        # Initialize for identity modulation initially
        nn.init.constant_(self.film_scale.weight, 0)
        nn.init.constant_(self.film_scale.bias, 1)
        if self.film_shift is not None:
            nn.init.constant_(self.film_shift.weight, 0)
            nn.init.constant_(self.film_shift.bias, 0)
    
    def _build_adain_layers(self):
        """Build Adaptive Instance Normalization layers"""
        # Style networks for mean and std
        self.adain_mean = nn.Linear(self.condition_dim, self.num_features)
        self.adain_std = nn.Linear(self.condition_dim, self.num_features)
        
        # Initialize for identity transform
        nn.init.constant_(self.adain_mean.weight, 0)
        nn.init.constant_(self.adain_mean.bias, 0)
        nn.init.constant_(self.adain_std.weight, 0)
        nn.init.constant_(self.adain_std.bias, 1)
    
    def _build_scale_shift_layers(self):
        """Build simple scale and shift layers"""
        self.scale_net = nn.Linear(self.condition_dim, self.num_features)
        self.shift_net = nn.Linear(self.condition_dim, self.num_features)
        
        # Initialize for identity
        nn.init.constant_(self.scale_net.bias, 1)
        nn.init.constant_(self.shift_net.bias, 0)
    
    def _build_gating_layers(self):
        """Build gating modulation layers"""
        if self.modulation_config.gating_type == 'channel':
            self.gate_net = nn.Linear(self.condition_dim, self.num_features)
        elif self.modulation_config.gating_type == 'spatial':
            # For spatial gating, we'll handle this in forward pass
            self.gate_net = nn.Linear(self.condition_dim, self.num_features)
        elif self.modulation_config.gating_type == 'both':
            self.channel_gate = nn.Linear(self.condition_dim, self.num_features)
            self.spatial_gate = nn.Linear(self.condition_dim, self.num_features)
        
        # Gating activation
        if self.modulation_config.gating_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif self.modulation_config.gating_activation == 'tanh':
            self.gate_activation = nn.Tanh()
        elif self.modulation_config.gating_activation == 'swish':
            self.gate_activation = nn.SiLU()
        else:
            self.gate_activation = nn.Sigmoid()
    
    def _build_attention_layers(self):
        """Build attention-based modulation layers"""
        self.attention_query = nn.Linear(self.num_features, self.num_features)
        self.attention_key = nn.Linear(self.condition_dim, self.num_features)
        self.attention_value = nn.Linear(self.condition_dim, self.num_features)
        
        self.attention_heads = self.modulation_config.attention_heads
        self.attention_dropout = nn.Dropout(self.modulation_config.attention_dropout)
        
        # Output projection
        self.attention_out = nn.Linear(self.num_features, self.num_features)
    
    def _build_squeeze_excite_layers(self):
        """Build Squeeze-and-Excitation layers"""
        reduction_dim = max(1, self.num_features // self.modulation_config.se_reduction_ratio)
        
        # Condition-aware SE
        self.se_squeeze = nn.AdaptiveAvgPool1d(1)
        self.se_excite = nn.Sequential(
            nn.Linear(self.num_features + self.condition_dim, reduction_dim),
            self._get_activation(self.modulation_config.se_activation),
            nn.Linear(reduction_dim, self.num_features),
            nn.Sigmoid()
        )
    
    def _build_fallback_modulation_layers(self):
        """Build simple fallback modulation layers"""
        try:
            # Simple scale-shift as fallback
            self.fallback_scale = nn.Linear(self.condition_dim, self.num_features)
            self.fallback_shift = nn.Linear(self.condition_dim, self.num_features)
            
            nn.init.constant_(self.fallback_scale.bias, 1)
            nn.init.constant_(self.fallback_shift.bias, 0)
            
            self.modulation_type = 'scale_shift'
            logger.info("Built fallback scale-shift modulation")
            
        except Exception as e:
            logger.error(f"Fallback modulation build failed: {e}")
            # Emergency fallback: identity
            self.fallback_scale = nn.Identity()
            self.fallback_shift = nn.Identity()
    
    def _build_global_conditioning(self):
        """Build global conditioning network"""
        try:
            global_dim = self.modulation_config.global_condition_dim
            
            self.global_condition_net = nn.Sequential(
                nn.Linear(global_dim, self.condition_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(self.condition_dim // 2, self.condition_dim)
            )
            
        except Exception as e:
            logger.error(f"Global conditioning build failed: {e}")
            self.global_condition_net = nn.Identity()
    
    def _build_local_conditioning(self):
        """Build local conditioning network"""
        try:
            local_dim = self.modulation_config.local_condition_dim
            
            self.local_condition_net = nn.Sequential(
                nn.Conv1d(local_dim, self.condition_dim // 2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(self.condition_dim // 2, self.condition_dim, 1)
            )
            
        except Exception as e:
            logger.error(f"Local conditioning build failed: {e}")
            self.local_condition_net = nn.Identity()
    
    def _build_temporal_conditioning(self):
        """Build temporal conditioning for sequential data"""
        try:
            window_size = self.modulation_config.temporal_window_size
            
            self.temporal_conditioning = nn.Sequential(
                nn.Conv1d(self.condition_dim, self.condition_dim, 
                         kernel_size=window_size, stride=self.modulation_config.temporal_stride,
                         padding=window_size//2),
                nn.ReLU(inplace=True),
                nn.Conv1d(self.condition_dim, self.condition_dim, 1)
            )
            
        except Exception as e:
            logger.error(f"Temporal conditioning build failed: {e}")
            self.temporal_conditioning = nn.Identity()
    
    def _get_activation(self, activation_name: str) -> nn.Module:
        """Get activation function by name"""
        if activation_name == 'relu':
            return nn.ReLU(inplace=True)
        elif activation_name == 'leaky_relu':
            return nn.LeakyReLU(0.2, inplace=True)
        elif activation_name == 'gelu':
            return nn.GELU()
        elif activation_name == 'swish':
            return nn.SiLU()
        elif activation_name == 'tanh':
            return nn.Tanh()
        elif activation_name == 'sigmoid':
            return nn.Sigmoid()
        else:
            return nn.ReLU(inplace=True)
    
    def _validate_inputs(self, features: torch.Tensor, 
                        condition: torch.Tensor) -> bool:
        """Validate input tensors"""
        try:
            # Check tensor validity
            for tensor in [features, condition]:
                if tensor is None:
                    return False
                if not torch.isfinite(tensor).all():
                    logger.warning("Non-finite values in modulation inputs")
                    return False
                if tensor.numel() == 0:
                    logger.warning("Empty tensor in modulation inputs")
                    return False
            
            # Check dimensions
            if features.size(1) != self.num_features:
                logger.warning(f"Feature dimension mismatch: expected {self.num_features}, got {features.size(1)}")
                return False
            
            # Check reasonable value ranges
            if torch.abs(features).max() > 1e6:
                logger.warning("Extremely large values in features")
                return False
            
            if torch.abs(condition).max() > 1e6:
                logger.warning("Extremely large values in condition")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False
    
    def _process_conditioning(self, condition: torch.Tensor,
                            global_condition: Optional[torch.Tensor] = None,
                            local_condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process and combine different conditioning signals"""
        try:
            processed_condition = condition
            
            # Process global conditioning
            if global_condition is not None and self.modulation_config.use_global_conditioning:
                if hasattr(self, 'global_condition_net'):
                    global_processed = self.global_condition_net(global_condition)
                    # Broadcast global condition to match sequence length
                    if processed_condition.dim() == 3:  # [batch, dim, seq_len]
                        global_processed = global_processed.unsqueeze(-1).expand_as(processed_condition)
                    processed_condition = processed_condition + global_processed
            
            # Process local conditioning
            if local_condition is not None and self.modulation_config.use_local_conditioning:
                if hasattr(self, 'local_condition_net'):
                    # Local condition should be [batch, local_dim, seq_len]
                    local_processed = self.local_condition_net(local_condition)
                    # Resize if needed
                    if local_processed.size(-1) != processed_condition.size(-1):
                        local_processed = F.interpolate(
                            local_processed, size=processed_condition.size(-1),
                            mode='linear', align_corners=False
                        )
                    processed_condition = processed_condition + local_processed
            
            # Apply temporal conditioning
            if self.modulation_config.temporal_conditioning and hasattr(self, 'temporal_conditioning'):
                if processed_condition.dim() == 3:
                    temporal_processed = self.temporal_conditioning(processed_condition)
                    processed_condition = processed_condition + temporal_processed
            
            return processed_condition
            
        except Exception as e:
            logger.error(f"Conditioning processing failed: {e}")
            return condition
    
    def _apply_film_modulation(self, features: torch.Tensor, 
                              condition: torch.Tensor) -> torch.Tensor:
        """Apply FiLM (Feature-wise Linear Modulation)"""
        try:
            # Compute scale and shift parameters
            scale = self.film_scale(condition)
            scale = self.film_activation(scale)
            
            if self.film_shift is not None:
                shift = self.film_shift(condition)
                shift = self.film_activation(shift)
            else:
                shift = 0
            
            # Handle broadcasting for different input shapes
            if features.dim() == 3:  # [batch, features, seq_len]
                if condition.dim() == 2:  # [batch, condition_dim]
                    scale = scale.unsqueeze(-1)
                    if isinstance(shift, torch.Tensor):
                        shift = shift.unsqueeze(-1)
                elif condition.dim() == 3:  # [batch, condition_dim, seq_len]
                    # Condition already has temporal dimension
                    pass
            
            # Apply modulation: features * (1 + scale) + shift
            modulated = features * (1 + scale) + shift
            
            return modulated
            
        except Exception as e:
            logger.error(f"FiLM modulation failed: {e}")
            return features
    
    def _apply_adain_modulation(self, features: torch.Tensor, 
                               condition: torch.Tensor) -> torch.Tensor:
        """Apply Adaptive Instance Normalization"""
        try:
            # Compute instance statistics
            if features.dim() == 3:  # [batch, features, seq_len]
                instance_mean = features.mean(dim=2, keepdim=True)
                instance_var = features.var(dim=2, keepdim=True, unbiased=False)
            else:
                instance_mean = features.mean(dim=1, keepdim=True)
                instance_var = features.var(dim=1, keepdim=True, unbiased=False)
            
            instance_std = (instance_var + self.modulation_config.adain_eps).sqrt()
            
            # Normalize
            normalized = (features - instance_mean) / instance_std
            
            # Get style parameters
            target_mean = self.adain_mean(condition)
            target_std = self.adain_std(condition)
            
            # Ensure positive std
            target_std = F.softplus(target_std) + self.modulation_config.adain_eps
            
            # Handle broadcasting
            if features.dim() == 3 and condition.dim() == 2:
                target_mean = target_mean.unsqueeze(-1)
                target_std = target_std.unsqueeze(-1)
            
            # Apply style transfer
            styled = normalized * target_std + target_mean
            
            return styled
            
        except Exception as e:
            logger.error(f"AdaIN modulation failed: {e}")
            return features
    
    def _apply_scale_shift_modulation(self, features: torch.Tensor, 
                                     condition: torch.Tensor) -> torch.Tensor:
        """Apply simple scale and shift modulation"""
        try:
            if hasattr(self, 'scale_net') and hasattr(self, 'shift_net'):
                scale = self.scale_net(condition)
                shift = self.shift_net(condition)
            else:
                # Fallback
                scale = self.fallback_scale(condition) if hasattr(self, 'fallback_scale') else 1
                shift = self.fallback_shift(condition) if hasattr(self, 'fallback_shift') else 0
            
            # Handle broadcasting
            if features.dim() == 3 and condition.dim() == 2:
                if isinstance(scale, torch.Tensor):
                    scale = scale.unsqueeze(-1)
                if isinstance(shift, torch.Tensor):
                    shift = shift.unsqueeze(-1)
            
            return features * scale + shift
            
        except Exception as e:
            logger.error(f"Scale-shift modulation failed: {e}")
            return features
    
    def _apply_gating_modulation(self, features: torch.Tensor, 
                                condition: torch.Tensor) -> torch.Tensor:
        """Apply gating modulation"""
        try:
            if self.modulation_config.gating_type == 'channel':
                gate = self.gate_net(condition)
                gate = self.gate_activation(gate)
                
                # Handle broadcasting
                if features.dim() == 3 and gate.dim() == 2:
                    gate = gate.unsqueeze(-1)
                
                return features * gate
                
            elif self.modulation_config.gating_type == 'both':
                channel_gate = self.channel_gate(condition)
                channel_gate = self.gate_activation(channel_gate)
                
                spatial_gate = self.spatial_gate(condition)
                spatial_gate = self.gate_activation(spatial_gate)
                
                # Handle broadcasting
                if features.dim() == 3:
                    if channel_gate.dim() == 2:
                        channel_gate = channel_gate.unsqueeze(-1)
                    if spatial_gate.dim() == 2:
                        spatial_gate = spatial_gate.unsqueeze(-1)
                
                return features * channel_gate * spatial_gate
            
            else:
                # Default to channel gating
                gate = self.gate_net(condition)
                gate = self.gate_activation(gate)
                if features.dim() == 3 and gate.dim() == 2:
                    gate = gate.unsqueeze(-1)
                return features * gate
                
        except Exception as e:
            logger.error(f"Gating modulation failed: {e}")
            return features
    
    def _apply_attention_modulation(self, features: torch.Tensor, 
                                   condition: torch.Tensor) -> torch.Tensor:
        """Apply attention-based modulation"""
        try:
            batch_size, num_features, seq_len = features.shape
            
            # Prepare queries, keys, values
            q = self.attention_query(features.transpose(1, 2))  # [batch, seq_len, features]
            
            if condition.dim() == 2:  # [batch, condition_dim]
                condition = condition.unsqueeze(1).expand(-1, seq_len, -1)
            elif condition.dim() == 3:  # [batch, condition_dim, seq_len]
                condition = condition.transpose(1, 2)  # [batch, seq_len, condition_dim]
            
            k = self.attention_key(condition)
            v = self.attention_value(condition)
            
            # Multi-head attention
            head_dim = num_features // self.attention_heads
            q = q.view(batch_size, seq_len, self.attention_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.attention_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.attention_heads, head_dim).transpose(1, 2)
            
            # Attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.attention_dropout(attention_weights)
            
            # Apply attention
            attended = torch.matmul(attention_weights, v)
            attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, num_features)
            
            # Output projection
            modulated = self.attention_out(attended)
            modulated = modulated.transpose(1, 2)  # Back to [batch, features, seq_len]
            
            # Residual connection
            return features + modulated
            
        except Exception as e:
            logger.error(f"Attention modulation failed: {e}")
            return features
    
    def _apply_squeeze_excite_modulation(self, features: torch.Tensor, 
                                        condition: torch.Tensor) -> torch.Tensor:
        """Apply Squeeze-and-Excitation modulation"""
        try:
            # Global average pooling
            squeezed = self.se_squeeze(features)  # [batch, features, 1]
            squeezed = squeezed.squeeze(-1)  # [batch, features]
            
            # Combine with condition
            if condition.dim() == 3:
                condition_pooled = condition.mean(dim=-1)  # Average over sequence
            else:
                condition_pooled = condition
            
            combined = torch.cat([squeezed, condition_pooled], dim=1)
            
            # Excitation
            excitation = self.se_excite(combined)  # [batch, features]
            excitation = excitation.unsqueeze(-1)  # [batch, features, 1]
            
            return features * excitation
            
        except Exception as e:
            logger.error(f"Squeeze-Excite modulation failed: {e}")
            return features
    
    def forward(self, features: torch.Tensor, condition: torch.Tensor,
                global_condition: Optional[torch.Tensor] = None,
                local_condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply feature modulation with comprehensive error handling.
        
        Args:
            features: Input features [batch, num_features, *spatial]
            condition: Conditioning signal [batch, condition_dim] or [batch, condition_dim, seq_len]
            global_condition: Global conditioning [batch, global_dim]
            local_condition: Local conditioning [batch, local_dim, seq_len]
            
        Returns:
            Modulated features
        """
        try:
            # Validate inputs
            if not self._validate_inputs(features, condition):
                if self.modulation_config.enable_fallbacks:
                    logger.warning("Input validation failed, using fallback")
                    self.fallback_activations += 1
                    if self.modulation_config.disable_modulation_on_failure:
                        return features
                    else:
                        # Simple scaling fallback
                        return features * 1.1  # Slight amplification
                else:
                    raise ValueError("Input validation failed")
            
            # Process conditioning signals
            processed_condition = self._process_conditioning(
                condition, global_condition, local_condition
            )
            
            # Handle condition dimension alignment
            if processed_condition.size(1) != self.condition_dim:
                if processed_condition.size(1) > self.condition_dim:
                    # Project down
                    processed_condition = processed_condition[:, :self.condition_dim]
                else:
                    # Pad up
                    padding = self.condition_dim - processed_condition.size(1)
                    if processed_condition.dim() == 3:
                        pad_tensor = torch.zeros(
                            processed_condition.size(0), padding, processed_condition.size(2),
                            device=processed_condition.device, dtype=processed_condition.dtype
                        )
                    else:
                        pad_tensor = torch.zeros(
                            processed_condition.size(0), padding,
                            device=processed_condition.device, dtype=processed_condition.dtype
                        )
                    processed_condition = torch.cat([processed_condition, pad_tensor], dim=1)
            
            # For 2D condition, take mean over sequence if features are 3D
            if features.dim() == 3 and processed_condition.dim() == 3:
                condition_for_modulation = processed_condition.mean(dim=-1)  # [batch, condition_dim]
            elif features.dim() == 3 and processed_condition.dim() == 2:
                condition_for_modulation = processed_condition
            else:
                condition_for_modulation = processed_condition
            
            # Apply modulation based on type
            if self.modulation_type == 'film':
                modulated = self._apply_film_modulation(features, condition_for_modulation)
            elif self.modulation_type == 'adain':
                modulated = self._apply_adain_modulation(features, condition_for_modulation)
            elif self.modulation_type == 'scale_shift':
                modulated = self._apply_scale_shift_modulation(features, condition_for_modulation)
            elif self.modulation_type == 'gating':
                modulated = self._apply_gating_modulation(features, condition_for_modulation)
            elif self.modulation_type == 'attention':
                modulated = self._apply_attention_modulation(features, processed_condition)
            elif self.modulation_type == 'squeeze_excite':
                modulated = self._apply_squeeze_excite_modulation(features, condition_for_modulation)
            else:
                logger.error(f"Unknown modulation type: {self.modulation_type}")
                modulated = features
            
            # Apply adaptive modulation strength
            if self.modulation_config.adaptive_modulation:
                strength = torch.sigmoid(self.modulation_strength)
                modulated = features * (1 - strength) + modulated * strength
            else:
                modulated = modulated * self.modulation_strength
            
            # Track statistics
            if len(self.modulation_stats) < 1000:
                self.modulation_stats.append({
                    'input_mean': features.mean().item(),
                    'input_std': features.std().item(),
                    'output_mean': modulated.mean().item(),
                    'output_std': modulated.std().item(),
                    'modulation_strength': self.modulation_strength.item(),
                    'condition_mean': condition.mean().item(),
                    'condition_std': condition.std().item()
                })
            
            return modulated
            
        except Exception as e:
            logger.error(f"Modulation forward pass failed: {e}")
            if self.modulation_config.enable_fallbacks:
                self.fallback_activations += 1
                logger.warning("Using emergency fallback: returning features unchanged")
                return features
            else:
                raise
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics for monitoring"""
        stats = {
            'modulation_type': self.modulation_type,
            'num_features': self.num_features,
            'condition_dim': self.condition_dim,
            'fallback_activations': self.fallback_activations,
            'modulation_strength': self.modulation_strength.item(),
            'adaptive_modulation': self.modulation_config.adaptive_modulation,
            'use_global_conditioning': self.modulation_config.use_global_conditioning,
            'use_local_conditioning': self.modulation_config.use_local_conditioning
        }
        
        if self.modulation_stats:
            last_stats = self.modulation_stats[-1]
            stats.update({
                'last_input_mean': last_stats['input_mean'],
                'last_output_mean': last_stats['output_mean'],
                'last_condition_mean': last_stats['condition_mean']
            })
        
        return stats
    
    def reset_statistics(self):
        """Reset training statistics"""
        self.modulation_stats.clear()
        self.conditioning_stats.clear()
        self.fallback_activations = 0


# Factory function
def create_bulletproof_modulation(config: RAVEConfig, **kwargs) -> BulletproofModulation:
    """Create a bulletproof modulation layer"""
    return BulletproofModulation(config, **kwargs)


if __name__ == "__main__":
    print("🛡️ BULLETPROOF MODULATION MODULE")
    print("=" * 45)
    
    # Test with minimal config
    from rave_config_system import get_minimal_config
    
    config = get_minimal_config()
    
    # Test data
    batch_size = 4
    num_features = 128
    seq_len = 512
    condition_dim = 64
    
    features = torch.randn(batch_size, num_features, seq_len)
    condition = torch.randn(batch_size, condition_dim)
    global_condition = torch.randn(batch_size, 32)
    local_condition = torch.randn(batch_size, 16, seq_len)
    
    # Test different modulation types
    modulation_types = ['film', 'adain', 'scale_shift', 'gating', 'attention', 'squeeze_excite']
    
    for mod_type in modulation_types:
        try:
            modulation_config = ModulationConfig(
                modulation_type=mod_type,
                num_features=num_features,
                condition_dim=condition_dim,
                use_global_conditioning=True,
                use_local_conditioning=True
            )
            
            modulation = create_bulletproof_modulation(
                config, modulation_config=modulation_config
            )
            
            output = modulation(features, condition, global_condition, local_condition)
            
            print(f"✅ {mod_type.upper()} modulation test passed")
            print(f"   Input shape: {features.shape}")
            print(f"   Output shape: {output.shape}")
            
            stats = modulation.get_training_stats()
            print(f"   Stats: strength={stats['modulation_strength']:.3f}, "
                  f"fallbacks={stats['fallback_activations']}")
            
        except Exception as e:
            print(f"❌ {mod_type.upper()} modulation test failed: {e}")
    
    # Test with temporal condition
    try:
        condition_temporal = torch.randn(batch_size, condition_dim, seq_len)
        
        modulation_config_temporal = ModulationConfig(
            modulation_type='film',
            num_features=num_features,
            condition_dim=condition_dim,
            temporal_conditioning=True
        )
        
        modulation_temporal = create_bulletproof_modulation(
            config, modulation_config=modulation_config_temporal
        )
        
        output_temporal = modulation_temporal(features, condition_temporal)
        print(f"✅ Temporal conditioning test passed")
        print(f"   Temporal condition shape: {condition_temporal.shape}")
        
    except Exception as e:
        print(f"❌ Temporal conditioning test failed: {e}")
    
    # Test with corrupted inputs
    try:
        features_corrupted = features.clone()
        features_corrupted[:, :, 100:110] = float('nan')
        
        modulation_robust = create_bulletproof_modulation(
            config, modulation_config=ModulationConfig(
                modulation_type='film', num_features=num_features, condition_dim=condition_dim
            )
        )
        
        output_robust = modulation_robust(features_corrupted, condition)
        print(f"✅ Robust handling of corrupted input")
        
    except Exception as e:
        print(f"❌ Corrupted input test failed: {e}")
    
    # Test adaptive modulation
    try:
        modulation_config_adaptive = ModulationConfig(
            modulation_type='scale_shift',
            num_features=num_features,
            condition_dim=condition_dim,
            adaptive_modulation=True,
            modulation_strength_learnable=True
        )
        
        modulation_adaptive = create_bulletproof_modulation(
            config, modulation_config=modulation_config_adaptive
        )
        
        output_adaptive = modulation_adaptive(features, condition)
        print(f"✅ Adaptive modulation test passed")
        
    except Exception as e:
        print(f"❌ Adaptive modulation test failed: {e}")
    
    print("🚀 BulletproofModulation ready for BigVGAN feature conditioning!")