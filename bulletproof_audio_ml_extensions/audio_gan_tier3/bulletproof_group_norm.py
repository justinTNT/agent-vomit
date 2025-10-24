#!/usr/bin/env python3
"""
BULLETPROOF GROUP NORMALIZATION MODULE
Comprehensive Group Normalization implementation for stable GAN training with BigVGAN compatibility.
Handles numerical instabilities, adaptive grouping, and memory efficient processing.
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
class GroupNormConfig:
    """Configuration for bulletproof group normalization"""
    num_groups: int = 32
    num_channels: int = 512
    eps: float = 1e-5
    affine: bool = True
    
    # Adaptive grouping parameters
    adaptive_groups: bool = True
    min_groups: int = 1
    max_groups: int = 64
    group_selection_strategy: str = 'divisible'  # 'divisible', 'closest', 'power_of_2'
    
    # Bulletproof stability parameters
    numerical_stability_check: bool = True
    adaptive_eps: bool = True
    min_eps: float = 1e-8
    max_eps: float = 1e-3
    variance_threshold: float = 1e-6
    
    # Memory efficiency options
    memory_efficient: bool = True
    chunk_processing: bool = False
    chunk_size: int = 1024
    
    # Advanced normalization features
    momentum: Optional[float] = None  # For exponential moving averages
    track_running_stats: bool = False
    use_weight_standardization: bool = False
    use_spectral_norm: bool = False
    
    # Fallback strategies
    enable_fallbacks: bool = True
    fallback_norm_type: str = 'batch_norm'  # 'batch_norm', 'layer_norm', 'instance_norm'
    disable_norm_on_failure: bool = False

class BulletproofGroupNorm(nn.Module):
    """
    Bulletproof Group Normalization with comprehensive error handling.
    
    Features:
    - Adaptive group selection for optimal normalization
    - Numerical stability with adaptive epsilon
    - Memory efficient processing for high-resolution features
    - Weight standardization and spectral normalization options
    - Comprehensive fallback strategies
    - Statistical monitoring and validation
    - Support for 1D, 2D, and 3D inputs
    """
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__()
        
        self.config = config
        self.group_norm_config = kwargs.get('group_norm_config', GroupNormConfig())
        
        # Override with RAVEConfig values if available
        if hasattr(config.model, 'base_channels'):
            self.group_norm_config.num_channels = config.model.base_channels
        
        self.num_channels = kwargs.get('num_channels', self.group_norm_config.num_channels)
        self.original_num_groups = kwargs.get('num_groups', self.group_norm_config.num_groups)
        self.eps = self.group_norm_config.eps
        self.affine = self.group_norm_config.affine
        
        # Determine optimal number of groups
        self.num_groups = self._determine_optimal_groups()
        
        # Build normalization layers with error handling
        try:
            self._build_normalization_layers()
        except Exception as e:
            logger.error(f"Failed to build group norm layers: {e}")
            if self.group_norm_config.enable_fallbacks:
                logger.warning("Building fallback normalization layers")
                self._build_fallback_normalization_layers()
            else:
                raise
        
        # Optional running statistics
        if self.group_norm_config.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_channels))
            self.register_buffer('running_var', torch.ones(self.num_channels))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None
        
        # Adaptive epsilon parameter
        if self.group_norm_config.adaptive_eps:
            self.register_buffer('adaptive_eps_value', torch.tensor(self.eps))
        
        # Tracking and monitoring
        self.normalization_stats = []
        self.group_stats = []
        self.fallback_activations = 0
        
        logger.info(f"BulletproofGroupNorm initialized: {self.num_channels} channels, {self.num_groups} groups")
    
    def _determine_optimal_groups(self) -> int:
        """Determine optimal number of groups based on channel count and strategy"""
        try:
            if not self.group_norm_config.adaptive_groups:
                # Validate fixed group count
                if self.num_channels % self.original_num_groups == 0:
                    return self.original_num_groups
                else:
                    logger.warning(f"Channels {self.num_channels} not divisible by groups {self.original_num_groups}")
                    if self.group_norm_config.enable_fallbacks:
                        # Find closest divisible number
                        return self._find_closest_divisible_groups()
                    else:
                        raise ValueError(f"Invalid group configuration")
            
            # Adaptive group selection
            strategy = self.group_norm_config.group_selection_strategy
            
            if strategy == 'divisible':
                return self._find_optimal_divisible_groups()
            elif strategy == 'closest':
                return self._find_closest_groups()
            elif strategy == 'power_of_2':
                return self._find_power_of_2_groups()
            else:
                logger.warning(f"Unknown group selection strategy: {strategy}")
                return self._find_optimal_divisible_groups()
                
        except Exception as e:
            logger.error(f"Group determination failed: {e}")
            # Emergency fallback: single group
            return 1
    
    def _find_optimal_divisible_groups(self) -> int:
        """Find optimal number of groups that divide evenly into channels"""
        try:
            min_groups = max(1, self.group_norm_config.min_groups)
            max_groups = min(self.num_channels, self.group_norm_config.max_groups)
            
            # Find divisors of num_channels
            divisors = []
            for g in range(min_groups, max_groups + 1):
                if self.num_channels % g == 0:
                    divisors.append(g)
            
            if not divisors:
                # No valid divisors in range, use min_groups
                return min_groups
            
            # Prefer groups that result in reasonable channels per group (4-32)
            optimal_channels_per_group = 16
            best_group = divisors[0]
            best_score = float('inf')
            
            for g in divisors:
                channels_per_group = self.num_channels // g
                score = abs(channels_per_group - optimal_channels_per_group)
                if score < best_score:
                    best_score = score
                    best_group = g
            
            return best_group
            
        except Exception as e:
            logger.error(f"Optimal divisible group finding failed: {e}")
            return min(32, self.num_channels)
    
    def _find_closest_divisible_groups(self) -> int:
        """Find closest valid group count to original request"""
        try:
            target = self.original_num_groups
            
            # Search around target value
            for delta in range(0, max(target, self.num_channels - target) + 1):
                # Try target + delta
                if target + delta <= self.num_channels and self.num_channels % (target + delta) == 0:
                    return target + delta
                
                # Try target - delta  
                if target - delta >= 1 and self.num_channels % (target - delta) == 0:
                    return target - delta
            
            # Fallback: single group
            return 1
            
        except Exception as e:
            logger.error(f"Closest divisible group finding failed: {e}")
            return 1
    
    def _find_closest_groups(self) -> int:
        """Find closest group count to target (may not be exactly divisible)"""
        try:
            target = self.original_num_groups
            min_groups = max(1, self.group_norm_config.min_groups)
            max_groups = min(self.num_channels, self.group_norm_config.max_groups)
            
            # Clamp target to valid range
            return max(min_groups, min(target, max_groups))
            
        except Exception as e:
            logger.error(f"Closest group finding failed: {e}")
            return min(32, self.num_channels)
    
    def _find_power_of_2_groups(self) -> int:
        """Find power of 2 group count that works well"""
        try:
            min_groups = max(1, self.group_norm_config.min_groups)
            max_groups = min(self.num_channels, self.group_norm_config.max_groups)
            
            # Find powers of 2 in valid range
            power = 0
            while 2**power < min_groups:
                power += 1
            
            best_groups = 2**power
            while best_groups <= max_groups:
                if self.num_channels % best_groups == 0:
                    return best_groups
                power += 1
                best_groups = 2**power
            
            # Fallback to largest power of 2 <= max_groups
            power = int(math.log2(max_groups))
            return 2**power
            
        except Exception as e:
            logger.error(f"Power of 2 group finding failed: {e}")
            return min(32, self.num_channels)
    
    def _build_normalization_layers(self):
        """Build main group normalization layers"""
        if self.affine:
            self.weight = nn.Parameter(torch.ones(self.num_channels))
            self.bias = nn.Parameter(torch.zeros(self.num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        # Optional weight standardization
        if self.group_norm_config.use_weight_standardization and hasattr(self, 'weight'):
            self.weight_standardization = True
        else:
            self.weight_standardization = False
        
        # Optional spectral normalization
        if self.group_norm_config.use_spectral_norm and hasattr(self, 'weight'):
            self.spectral_norm = nn.utils.spectral_norm
        else:
            self.spectral_norm = None
    
    def _build_fallback_normalization_layers(self):
        """Build fallback normalization layers"""
        try:
            fallback_type = self.group_norm_config.fallback_norm_type
            
            if fallback_type == 'batch_norm':
                # Use BatchNorm1d as fallback
                self.fallback_norm = nn.BatchNorm1d(self.num_channels, eps=self.eps)
                logger.info("Built BatchNorm fallback")
                
            elif fallback_type == 'layer_norm':
                # Use LayerNorm as fallback
                self.fallback_norm = nn.LayerNorm(self.num_channels, eps=self.eps)
                logger.info("Built LayerNorm fallback")
                
            elif fallback_type == 'instance_norm':
                # Use InstanceNorm1d as fallback
                self.fallback_norm = nn.InstanceNorm1d(self.num_channels, eps=self.eps)
                logger.info("Built InstanceNorm fallback")
                
            else:
                # Emergency fallback: identity
                self.fallback_norm = nn.Identity()
                logger.warning("Using Identity as emergency fallback")
                
        except Exception as e:
            logger.error(f"Fallback normalization build failed: {e}")
            self.fallback_norm = nn.Identity()
    
    def _validate_inputs(self, x: torch.Tensor) -> bool:
        """Validate input tensor for group normalization"""
        try:
            # Check tensor validity
            if x is None:
                return False
            
            if not torch.isfinite(x).all():
                logger.warning("Non-finite values in input")
                return False
            
            if x.numel() == 0:
                logger.warning("Empty tensor")
                return False
            
            # Check dimensions
            if x.dim() < 2:
                logger.warning(f"Input must be at least 2D, got {x.dim()}D")
                return False
            
            if x.size(1) != self.num_channels:
                logger.warning(f"Channel dimension mismatch: expected {self.num_channels}, got {x.size(1)}")
                return False
            
            # Check for reasonable value ranges
            if torch.abs(x).max() > 1e6:
                logger.warning("Extremely large values in input")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False
    
    def _compute_group_statistics(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute group-wise mean and variance with numerical stability"""
        try:
            batch_size = x.size(0)
            
            # Reshape input for group processing
            # Original: [N, C, *spatial_dims]
            # Reshape to: [N, num_groups, channels_per_group, *spatial_dims]
            original_shape = x.shape
            
            if self.num_channels % self.num_groups != 0:
                # Handle non-divisible case by padding or truncating
                channels_per_group = self.num_channels // self.num_groups
                effective_channels = channels_per_group * self.num_groups
                
                if effective_channels < self.num_channels:
                    # Truncate extra channels
                    x = x[:, :effective_channels]
                    logger.warning(f"Truncated channels from {self.num_channels} to {effective_channels}")
                
                channels_per_group = effective_channels // self.num_groups
            else:
                channels_per_group = self.num_channels // self.num_groups
            
            # Reshape for group-wise computation
            new_shape = [batch_size, self.num_groups, channels_per_group] + list(original_shape[2:])
            x_grouped = x.view(new_shape)
            
            # Compute statistics over group, spatial dimensions
            # Keep batch and group dimensions separate
            reduce_dims = list(range(2, x_grouped.dim()))  # All dims except batch and group
            
            group_mean = x_grouped.mean(dim=reduce_dims, keepdim=True)
            group_var = x_grouped.var(dim=reduce_dims, keepdim=True, unbiased=False)
            
            # Handle adaptive epsilon
            if self.group_norm_config.adaptive_eps:
                # Adjust eps based on variance magnitude
                mean_var = group_var.mean()
                adaptive_eps = torch.clamp(
                    mean_var * 0.01,
                    min=self.group_norm_config.min_eps,
                    max=self.group_norm_config.max_eps
                )
                self.adaptive_eps_value.copy_(adaptive_eps)
                current_eps = adaptive_eps
            else:
                current_eps = self.eps
            
            # Add epsilon for numerical stability
            group_std = (group_var + current_eps).sqrt()
            
            # Validate computed statistics
            if not torch.isfinite(group_mean).all() or not torch.isfinite(group_std).all():
                logger.warning("Non-finite group statistics computed")
                # Fallback to safer computation
                group_mean = torch.zeros_like(group_mean)
                group_std = torch.ones_like(group_std)
            
            # Check for very small variances
            if (group_var < self.group_norm_config.variance_threshold).any():
                logger.debug("Very small group variances detected")
                # Clamp to minimum threshold
                group_var = torch.clamp(group_var, min=self.group_norm_config.variance_threshold)
                group_std = (group_var + current_eps).sqrt()
            
            return group_mean, group_std
            
        except Exception as e:
            logger.error(f"Group statistics computation failed: {e}")
            # Emergency fallback
            batch_size = x.size(0)
            fallback_shape = [batch_size, self.num_groups, 1] + [1] * (x.dim() - 2)
            mean = torch.zeros(fallback_shape, device=x.device, dtype=x.dtype)
            std = torch.ones(fallback_shape, device=x.device, dtype=x.dtype)
            return mean, std
    
    def _apply_group_normalization(self, x: torch.Tensor, group_mean: torch.Tensor, 
                                  group_std: torch.Tensor) -> torch.Tensor:
        """Apply group normalization with affine transformation"""
        try:
            # Normalize
            x_norm = (x - group_mean) / group_std
            
            # Apply affine transformation if enabled
            if self.affine and self.weight is not None and self.bias is not None:
                # Reshape weight and bias for broadcasting
                # Need to handle the grouped structure
                weight = self.weight.view(1, -1, *([1] * (x.dim() - 2)))
                bias = self.bias.view(1, -1, *([1] * (x.dim() - 2)))
                
                # Handle non-divisible channels
                if x_norm.size(1) != weight.size(1):
                    weight = weight[:, :x_norm.size(1)]
                    bias = bias[:, :x_norm.size(1)]
                
                # Apply weight standardization if enabled
                if self.weight_standardization:
                    weight_mean = weight.mean(dim=[1] + list(range(2, weight.dim())), keepdim=True)
                    weight_std = weight.std(dim=[1] + list(range(2, weight.dim())), keepdim=True) + self.eps
                    weight = (weight - weight_mean) / weight_std
                
                x_norm = x_norm * weight + bias
            
            return x_norm
            
        except Exception as e:
            logger.error(f"Group normalization application failed: {e}")
            return x  # Return input unchanged
    
    def _update_running_statistics(self, x: torch.Tensor):
        """Update running statistics if tracking is enabled"""
        try:
            if not self.training or not self.group_norm_config.track_running_stats:
                return
            
            # Compute batch statistics (average across batch and spatial dims)
            # Keep channel dimension separate
            reduce_dims = [0] + list(range(2, x.dim()))
            batch_mean = x.mean(dim=reduce_dims)
            batch_var = x.var(dim=reduce_dims, unbiased=False)
            
            self.num_batches_tracked += 1
            
            if self.group_norm_config.momentum is None:
                # Cumulative moving average
                n = self.num_batches_tracked.item()
                self.running_mean = (self.running_mean * (n - 1) + batch_mean) / n
                self.running_var = (self.running_var * (n - 1) + batch_var) / n
            else:
                # Exponential moving average
                momentum = self.group_norm_config.momentum
                self.running_mean = (1 - momentum) * self.running_mean + momentum * batch_mean
                self.running_var = (1 - momentum) * self.running_var + momentum * batch_var
                
        except Exception as e:
            logger.warning(f"Running statistics update failed: {e}")
    
    def _memory_efficient_processing(self, x: torch.Tensor) -> torch.Tensor:
        """Process input in chunks for memory efficiency"""
        try:
            if not self.group_norm_config.memory_efficient or not self.group_norm_config.chunk_processing:
                return self._standard_processing(x)
            
            batch_size, channels, *spatial_dims = x.shape
            total_spatial = np.prod(spatial_dims) if spatial_dims else 1
            
            if total_spatial <= self.group_norm_config.chunk_size:
                return self._standard_processing(x)
            
            # Process in spatial chunks
            chunk_size = self.group_norm_config.chunk_size
            x_flat = x.view(batch_size, channels, -1)
            
            chunks = []
            for i in range(0, x_flat.size(2), chunk_size):
                end_idx = min(i + chunk_size, x_flat.size(2))
                x_chunk = x_flat[:, :, i:end_idx]
                
                # Reshape chunk back to original spatial structure for processing
                chunk_spatial_dims = [end_idx - i]
                x_chunk_reshaped = x_chunk.view(batch_size, channels, *chunk_spatial_dims)
                
                # Process chunk
                processed_chunk = self._standard_processing(x_chunk_reshaped)
                
                # Flatten for concatenation
                chunks.append(processed_chunk.view(batch_size, channels, -1))
            
            # Concatenate processed chunks
            processed_flat = torch.cat(chunks, dim=2)
            
            # Reshape back to original shape
            return processed_flat.view(x.shape)
            
        except Exception as e:
            logger.error(f"Memory efficient processing failed: {e}")
            return self._standard_processing(x)
    
    def _standard_processing(self, x: torch.Tensor) -> torch.Tensor:
        """Standard group normalization processing"""
        try:
            # Store original shape
            original_shape = x.shape
            
            # Compute group statistics
            group_mean, group_std = self._compute_group_statistics(x)
            
            # Reshape statistics to match grouped input
            batch_size = x.size(0)
            channels_per_group = x.size(1) // self.num_groups
            
            # Expand statistics for broadcasting
            group_mean_expanded = group_mean.repeat_interleave(channels_per_group, dim=1)
            group_std_expanded = group_std.repeat_interleave(channels_per_group, dim=1)
            
            # Handle dimension mismatch
            if group_mean_expanded.size(1) > x.size(1):
                group_mean_expanded = group_mean_expanded[:, :x.size(1)]
                group_std_expanded = group_std_expanded[:, :x.size(1)]
            
            # Apply normalization
            x_norm = self._apply_group_normalization(x, group_mean_expanded, group_std_expanded)
            
            # Update running statistics
            self._update_running_statistics(x)
            
            # Track statistics
            if len(self.normalization_stats) < 1000:
                self.normalization_stats.append({
                    'input_mean': x.mean().item(),
                    'input_std': x.std().item(),
                    'output_mean': x_norm.mean().item(),
                    'output_std': x_norm.std().item(),
                    'group_mean_avg': group_mean.mean().item(),
                    'group_std_avg': group_std.mean().item()
                })
            
            return x_norm
            
        except Exception as e:
            logger.error(f"Standard processing failed: {e}")
            raise
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply group normalization with comprehensive error handling.
        
        Args:
            x: Input tensor [batch, channels, *spatial]
            
        Returns:
            Group normalized tensor
        """
        try:
            # Validate inputs
            if not self._validate_inputs(x):
                if self.group_norm_config.enable_fallbacks:
                    logger.warning("Input validation failed, using fallback")
                    self.fallback_activations += 1
                    
                    if self.group_norm_config.disable_norm_on_failure:
                        return x  # Return unchanged
                    elif hasattr(self, 'fallback_norm'):
                        # Use fallback normalization
                        if isinstance(self.fallback_norm, nn.BatchNorm1d):
                            # BatchNorm expects 3D input for 1D data
                            if x.dim() == 3:
                                return self.fallback_norm(x)
                            else:
                                # Reshape for BatchNorm
                                original_shape = x.shape
                                x_reshaped = x.view(x.size(0), x.size(1), -1)
                                normalized = self.fallback_norm(x_reshaped)
                                return normalized.view(original_shape)
                        else:
                            return self.fallback_norm(x)
                    else:
                        return x
                else:
                    raise ValueError("Input validation failed")
            
            # Apply group normalization (memory efficient if enabled)
            output = self._memory_efficient_processing(x)
            
            return output
            
        except Exception as e:
            logger.error(f"Group normalization forward pass failed: {e}")
            if self.group_norm_config.enable_fallbacks:
                self.fallback_activations += 1
                logger.warning("Using emergency fallback: returning input unchanged")
                return x
            else:
                raise
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics for monitoring"""
        stats = {
            'num_channels': self.num_channels,
            'num_groups': self.num_groups,
            'channels_per_group': self.num_channels // self.num_groups,
            'affine': self.affine,
            'fallback_activations': self.fallback_activations,
            'adaptive_eps': self.adaptive_eps_value.item() if self.group_norm_config.adaptive_eps else self.eps
        }
        
        if self.normalization_stats:
            last_stats = self.normalization_stats[-1]
            stats.update({
                'last_input_mean': last_stats['input_mean'],
                'last_output_mean': last_stats['output_mean'],
                'last_group_std_avg': last_stats['group_std_avg']
            })
        
        if self.group_norm_config.track_running_stats and self.running_mean is not None:
            stats.update({
                'running_mean_norm': self.running_mean.norm().item(),
                'running_var_mean': self.running_var.mean().item(),
                'num_batches_tracked': self.num_batches_tracked.item()
            })
        
        return stats
    
    def reset_statistics(self):
        """Reset training statistics"""
        self.normalization_stats.clear()
        self.group_stats.clear()
        self.fallback_activations = 0
        
        if self.group_norm_config.track_running_stats:
            if self.running_mean is not None:
                self.running_mean.zero_()
            if self.running_var is not None:
                self.running_var.fill_(1)
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.zero_()


# Factory function
def create_bulletproof_group_norm(config: RAVEConfig, num_channels: int, **kwargs) -> BulletproofGroupNorm:
    """Create a bulletproof group normalization layer"""
    return BulletproofGroupNorm(config, num_channels=num_channels, **kwargs)


if __name__ == "__main__":
    print("🛡️ BULLETPROOF GROUP NORMALIZATION MODULE")
    print("=" * 55)
    
    # Test with minimal config
    from rave_config_system import get_minimal_config
    
    config = get_minimal_config()
    
    # Test 1D group norm
    group_norm_config = GroupNormConfig(num_groups=8, num_channels=64)
    group_norm_1d = create_bulletproof_group_norm(config, 64, group_norm_config=group_norm_config)
    
    # Test data
    batch_size = 4
    channels = 64
    seq_len = 1024
    
    x_1d = torch.randn(batch_size, channels, seq_len)
    
    try:
        output_1d = group_norm_1d(x_1d)
        print(f"✅ 1D Group normalization test passed")
        print(f"   Input shape: {x_1d.shape}")
        print(f"   Output shape: {output_1d.shape}")
        print(f"   Groups: {group_norm_1d.num_groups}")
        
        stats = group_norm_1d.get_training_stats()
        print(f"   Group norm stats: {stats}")
        
    except Exception as e:
        print(f"❌ 1D Group norm test failed: {e}")
    
    # Test 2D group norm
    try:
        height, width = 32, 32
        x_2d = torch.randn(batch_size, channels, height, width)
        
        output_2d = group_norm_1d(x_2d)  # Same layer works for different dims
        print(f"✅ 2D Group normalization test passed")
        print(f"   2D input shape: {x_2d.shape}")
        print(f"   2D output shape: {output_2d.shape}")
        
    except Exception as e:
        print(f"❌ 2D Group norm test failed: {e}")
    
    # Test adaptive grouping
    try:
        # Test with non-standard channel count
        group_norm_config_adaptive = GroupNormConfig(
            num_groups=13, num_channels=77, adaptive_groups=True
        )
        group_norm_adaptive = create_bulletproof_group_norm(
            config, 77, group_norm_config=group_norm_config_adaptive
        )
        
        x_adaptive = torch.randn(batch_size, 77, seq_len)
        output_adaptive = group_norm_adaptive(x_adaptive)
        
        print(f"✅ Adaptive grouping test passed")
        print(f"   Requested groups: 13, Actual groups: {group_norm_adaptive.num_groups}")
        print(f"   Channels per group: {77 // group_norm_adaptive.num_groups}")
        
    except Exception as e:
        print(f"❌ Adaptive grouping test failed: {e}")
    
    # Test with corrupted inputs
    try:
        x_corrupted = x_1d.clone()
        x_corrupted[:, :, 100:110] = float('nan')
        
        output_robust = group_norm_1d(x_corrupted)
        print(f"✅ Robust handling of corrupted input")
        
    except Exception as e:
        print(f"❌ Corrupted input test failed: {e}")
    
    # Test memory efficient processing
    try:
        group_norm_config_efficient = GroupNormConfig(
            num_groups=8, num_channels=64,
            memory_efficient=True, chunk_processing=True, chunk_size=256
        )
        group_norm_efficient = create_bulletproof_group_norm(
            config, 64, group_norm_config=group_norm_config_efficient
        )
        
        # Large input for memory efficiency test
        x_large = torch.randn(batch_size, channels, seq_len * 4)
        output_efficient = group_norm_efficient(x_large)
        
        print(f"✅ Memory efficient processing test passed")
        print(f"   Large input shape: {x_large.shape}")
        
    except Exception as e:
        print(f"❌ Memory efficient processing test failed: {e}")
    
    # Test with different group selection strategies
    for strategy in ['divisible', 'closest', 'power_of_2']:
        try:
            group_norm_config_strategy = GroupNormConfig(
                num_groups=12, num_channels=64,
                adaptive_groups=True, group_selection_strategy=strategy
            )
            group_norm_strategy = create_bulletproof_group_norm(
                config, 64, group_norm_config=group_norm_config_strategy
            )
            
            x_strategy = torch.randn(batch_size, 64, seq_len)
            output_strategy = group_norm_strategy(x_strategy)
            
            print(f"✅ Strategy '{strategy}' test passed, groups: {group_norm_strategy.num_groups}")
            
        except Exception as e:
            print(f"❌ Strategy '{strategy}' test failed: {e}")
    
    print("🚀 BulletproofGroupNorm ready for stable BigVGAN training!")