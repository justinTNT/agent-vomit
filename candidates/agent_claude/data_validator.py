"""
DataValidator - Validate data against schemas with statistics.

This module checks tensor shapes, types, ranges, computes and tracks statistics,
and returns validation results.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Union
import warnings


class DataValidator(nn.Module):
    """
    Validate data against schemas with statistics tracking.
    
    Args:
        expected_shape: Expected tensor shape (None for any dimension)
        expected_dtype: Expected data type
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        check_nan: Check for NaN values
        check_inf: Check for infinite values
        track_statistics: Whether to track running statistics
        epsilon: Small value for numerical stability
        **kwargs: Additional arguments
    """
    
    def __init__(
        self,
        expected_shape: Optional[Tuple[Optional[int], ...]] = None,
        expected_dtype: Optional[torch.dtype] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        check_nan: bool = True,
        check_inf: bool = True,
        track_statistics: bool = True,
        epsilon: float = 1e-8,
        **kwargs
    ):
        super().__init__()
        
        self.expected_shape = expected_shape
        self.expected_dtype = expected_dtype
        self.min_value = min_value
        self.max_value = max_value
        self.check_nan = check_nan
        self.check_inf = check_inf
        self.track_statistics = track_statistics
        self.epsilon = epsilon
        
        # Initialize statistics tracking
        if self.track_statistics:
            self.register_buffer('running_mean', torch.tensor(0.0))
            self.register_buffer('running_var', torch.tensor(0.0))
            self.register_buffer('running_min', torch.tensor(float('inf')))
            self.register_buffer('running_max', torch.tensor(float('-inf')))
            self.register_buffer('total_samples', torch.tensor(0, dtype=torch.long))
            self.register_buffer('nan_count', torch.tensor(0, dtype=torch.long))
            self.register_buffer('inf_count', torch.tensor(0, dtype=torch.long))
            self.register_buffer('validation_failures', torch.tensor(0, dtype=torch.long))
        
        # Handle unused kwargs
        if kwargs:
            warnings.warn(f"Unused arguments: {list(kwargs.keys())}")
    
    def forward(self, x: torch.Tensor) -> Dict[str, Union[torch.Tensor, bool, List[str]]]:
        """
        Validate input tensor and compute statistics.
        
        Args:
            x: Input tensor to validate
            
        Returns:
            Dictionary containing:
                - is_valid: Boolean indicating if all validations passed
                - errors: List of validation errors
                - statistics: Dictionary of computed statistics
                - cleaned_data: Input with NaN/Inf replaced if needed
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        
        device = x.device
        errors = []
        is_valid = True
        
        # Shape validation
        if self.expected_shape is not None:
            if len(x.shape) != len(self.expected_shape):
                errors.append(f"Shape mismatch: expected {len(self.expected_shape)} dims, got {len(x.shape)}")
                is_valid = False
            else:
                for i, (expected, actual) in enumerate(zip(self.expected_shape, x.shape)):
                    if expected is not None and expected != actual:
                        errors.append(f"Shape mismatch at dim {i}: expected {expected}, got {actual}")
                        is_valid = False
        
        # Data type validation
        if self.expected_dtype is not None and x.dtype != self.expected_dtype:
            errors.append(f"Dtype mismatch: expected {self.expected_dtype}, got {x.dtype}")
            is_valid = False
        
        # NaN validation
        nan_mask = torch.isnan(x)
        has_nan = nan_mask.any().item()
        if self.check_nan and has_nan:
            nan_count = nan_mask.sum().item()
            errors.append(f"Found {nan_count} NaN values")
            is_valid = False
            if self.track_statistics:
                self.nan_count += nan_count
        
        # Inf validation
        inf_mask = torch.isinf(x)
        has_inf = inf_mask.any().item()
        if self.check_inf and has_inf:
            inf_count = inf_mask.sum().item()
            errors.append(f"Found {inf_count} infinite values")
            is_valid = False
            if self.track_statistics:
                self.inf_count += inf_count
        
        # Create valid mask for range checking
        valid_mask = ~(nan_mask | inf_mask)
        valid_values = x[valid_mask]
        
        # Range validation
        if valid_values.numel() > 0:
            if self.min_value is not None:
                below_min = (valid_values < self.min_value).sum().item()
                if below_min > 0:
                    errors.append(f"Found {below_min} values below minimum {self.min_value}")
                    is_valid = False
            
            if self.max_value is not None:
                above_max = (valid_values > self.max_value).sum().item()
                if above_max > 0:
                    errors.append(f"Found {above_max} values above maximum {self.max_value}")
                    is_valid = False
        
        # Update statistics
        statistics = {}
        if self.track_statistics and valid_values.numel() > 0:
            # Compute current batch statistics
            batch_mean = valid_values.mean()
            batch_var = valid_values.var(unbiased=False)
            batch_min = valid_values.min()
            batch_max = valid_values.max()
            
            # Update running statistics
            n_old = self.total_samples.float()
            n_new = valid_values.numel()
            n_total = n_old + n_new
            
            if n_old == 0:
                self.running_mean.copy_(batch_mean)
                self.running_var.copy_(batch_var)
                self.running_min.copy_(batch_min)
                self.running_max.copy_(batch_max)
            else:
                # Welford's online algorithm for mean and variance
                delta = batch_mean - self.running_mean
                self.running_mean.add_(delta * n_new / n_total)
                
                # Update variance
                m_a = self.running_var * n_old
                m_b = batch_var * n_new
                M2 = m_a + m_b + delta ** 2 * n_old * n_new / n_total
                self.running_var.copy_(M2 / n_total)
                
                # Update min/max
                self.running_min.copy_(torch.min(self.running_min, batch_min))
                self.running_max.copy_(torch.max(self.running_max, batch_max))
            
            self.total_samples += n_new
            
            if not is_valid:
                self.validation_failures += 1
            
            statistics = {
                'mean': self.running_mean.clone(),
                'var': self.running_var.clone(),
                'std': torch.sqrt(self.running_var + self.epsilon),
                'min': self.running_min.clone(),
                'max': self.running_max.clone(),
                'total_samples': self.total_samples.clone(),
                'nan_count': self.nan_count.clone(),
                'inf_count': self.inf_count.clone(),
                'validation_failures': self.validation_failures.clone()
            }
        
        # Clean data by replacing NaN/Inf with zeros if needed
        cleaned_data = x.clone()
        if has_nan or has_inf:
            cleaned_data[nan_mask | inf_mask] = 0.0
        
        return {
            'is_valid': torch.tensor(is_valid, device=device),
            'errors': errors,
            'statistics': statistics,
            'cleaned_data': cleaned_data
        }
    
    def reset_statistics(self):
        """Reset all running statistics."""
        if self.track_statistics:
            self.running_mean.zero_()
            self.running_var.zero_()
            self.running_min.fill_(float('inf'))
            self.running_max.fill_(float('-inf'))
            self.total_samples.zero_()
            self.nan_count.zero_()
            self.inf_count.zero_()
            self.validation_failures.zero_()
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get comprehensive validation summary."""
        summary = {
            'expected_shape': self.expected_shape,
            'expected_dtype': self.expected_dtype,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'check_nan': self.check_nan,
            'check_inf': self.check_inf
        }
        
        if self.track_statistics:
            summary['statistics'] = {
                'mean': self.running_mean.item(),
                'std': torch.sqrt(self.running_var + self.epsilon).item(),
                'min': self.running_min.item(),
                'max': self.running_max.item(),
                'total_samples': self.total_samples.item(),
                'nan_count': self.nan_count.item(),
                'inf_count': self.inf_count.item(),
                'validation_failures': self.validation_failures.item()
            }
        
        return summary