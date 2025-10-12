import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import json


class ValidationRule:
    """Base class for validation rules."""
    def validate(self, data: Any) -> Tuple[bool, Optional[str]]:
        raise NotImplementedError


class DType(ValidationRule):
    """Validates tensor data type."""
    def __init__(self, dtype: torch.dtype):
        self.dtype = dtype
        
    def validate(self, data: torch.Tensor) -> Tuple[bool, Optional[str]]:
        if not isinstance(data, torch.Tensor):
            return False, "Expected torch.Tensor"
        if data.dtype != self.dtype:
            return False, f"Expected dtype {self.dtype}, got {data.dtype}"
        return True, None


class Shape(ValidationRule):
    """Validates tensor shape with wildcards."""
    def __init__(self, shape: Tuple[Union[int, str], ...]):
        self.shape = shape
        
    def validate(self, data: torch.Tensor) -> Tuple[bool, Optional[str]]:
        if not isinstance(data, torch.Tensor):
            return False, "Expected torch.Tensor"
        if len(data.shape) != len(self.shape):
            return False, f"Expected {len(self.shape)} dimensions, got {len(data.shape)}"
        
        for i, (expected, actual) in enumerate(zip(self.shape, data.shape)):
            if expected == "*":  # Wildcard
                continue
            if expected != actual:
                return False, f"Dimension {i}: expected {expected}, got {actual}"
        return True, None


class Range(ValidationRule):
    """Validates numeric ranges."""
    def __init__(self, min_val: Optional[float] = None, max_val: Optional[float] = None):
        self.min_val = min_val
        self.max_val = max_val
        
    def validate(self, data: torch.Tensor) -> Tuple[bool, Optional[str]]:
        if not isinstance(data, torch.Tensor):
            return False, "Expected torch.Tensor"
        
        if self.min_val is not None and data.min().item() < self.min_val:
            return False, f"Values below minimum {self.min_val}"
        if self.max_val is not None and data.max().item() > self.max_val:
            return False, f"Values above maximum {self.max_val}"
        return True, None


class NotNaN(ValidationRule):
    """Validates no NaN values."""
    def validate(self, data: torch.Tensor) -> Tuple[bool, Optional[str]]:
        if not isinstance(data, torch.Tensor):
            return False, "Expected torch.Tensor"
        if torch.isnan(data).any():
            return False, "Contains NaN values"
        return True, None


class NotInf(ValidationRule):
    """Validates no infinite values."""
    def validate(self, data: torch.Tensor) -> Tuple[bool, Optional[str]]:
        if not isinstance(data, torch.Tensor):
            return False, "Expected torch.Tensor"
        if torch.isinf(data).any():
            return False, "Contains infinite values"
        return True, None


class Custom(ValidationRule):
    """Custom validation function."""
    def __init__(self, func: Callable[[Any], bool], error_msg: str = "Custom validation failed"):
        self.func = func
        self.error_msg = error_msg
        
    def validate(self, data: Any) -> Tuple[bool, Optional[str]]:
        try:
            if not self.func(data):
                return False, self.error_msg
            return True, None
        except Exception as e:
            return False, f"Validation error: {str(e)}"


@dataclass
class Schema:
    """Defines validation schema for data."""
    rules: Dict[str, List[ValidationRule]]
    strict: bool = True  # If True, reject extra fields
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary for serialization."""
        return {
            "rules": {
                key: [rule.__class__.__name__ for rule in rules]
                for key, rules in self.rules.items()
            },
            "strict": self.strict
        }


class DataValidator(nn.Module):
    """
    Validates data against defined schemas with distribution tracking.
    Supports schema evolution and validation caching.
    """
    
    def __init__(
        self,
        schema: Optional[Schema] = None,
        track_distributions: bool = True,
        cache_size: int = 1000,
        evolve_schema: bool = False
    ):
        super().__init__()
        self.schema = schema
        self.track_distributions = track_distributions
        self.cache_size = cache_size
        self.evolve_schema = evolve_schema
        
        # Statistics tracking
        self.stats = {
            "mean": {},
            "std": {},
            "min": {},
            "max": {},
            "shape": {},
            "dtype": {},
            "count": {}
        }
        
        # Validation cache
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Schema evolution tracking
        self.schema_violations = {}
        self.suggested_rules = {}
        
    def forward(self, data: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                key: Optional[str] = None) -> Tuple[bool, List[str]]:
        """Validate data against schema."""
        if isinstance(data, torch.Tensor):
            data = {"default": data}
            if key is None:
                key = "default"
                
        errors = []
        
        # Check schema
        if self.schema:
            errors.extend(self._validate_schema(data))
            
        # Track distributions
        if self.track_distributions:
            self._update_statistics(data)
            
        # Check distribution shift
        errors.extend(self._check_distribution_shift(data))
        
        return len(errors) == 0, errors
    
    def _validate_schema(self, data: Dict[str, torch.Tensor]) -> List[str]:
        """Validate data dictionary against schema."""
        errors = []
        
        # Check for extra fields
        if self.schema.strict:
            extra_fields = set(data.keys()) - set(self.schema.rules.keys())
            for field in extra_fields:
                errors.append(f"Unexpected field: {field}")
                # Track for schema evolution
                if self.evolve_schema and field in data:
                    self._track_violation(field, None, data[field])
                
        # Validate each field
        for field, rules in self.schema.rules.items():
            if field not in data:
                errors.append(f"Missing required field: {field}")
                continue
                
            for rule in rules:
                valid, error = rule.validate(data[field])
                if not valid:
                    errors.append(f"{field}: {error}")
                    
                    # Track for schema evolution
                    if self.evolve_schema:
                        self._track_violation(field, rule, data[field])
                        
        return errors
    
    def _update_statistics(self, data: Dict[str, torch.Tensor]) -> None:
        """Update running statistics for distribution tracking."""
        for key, tensor in data.items():
            if key not in self.stats["count"]:
                self.stats["count"][key] = 0
                
            # Update count
            self.stats["count"][key] += 1
            n = self.stats["count"][key]
            
            # Update statistics
            if n == 1:
                self.stats["mean"][key] = tensor.float().mean().item()
                self.stats["std"][key] = tensor.float().std().item()
                self.stats["min"][key] = tensor.min().item()
                self.stats["max"][key] = tensor.max().item()
                self.stats["shape"][key] = list(tensor.shape)
                self.stats["dtype"][key] = str(tensor.dtype)
            else:
                # Running average
                alpha = 1.0 / n
                self.stats["mean"][key] = (1 - alpha) * self.stats["mean"][key] + alpha * tensor.float().mean().item()
                self.stats["std"][key] = (1 - alpha) * self.stats["std"][key] + alpha * tensor.float().std().item()
                self.stats["min"][key] = min(self.stats["min"][key], tensor.min().item())
                self.stats["max"][key] = max(self.stats["max"][key], tensor.max().item())
                
    def _check_distribution_shift(self, data: Dict[str, torch.Tensor]) -> List[str]:
        """Check for significant distribution shifts."""
        errors = []
        
        for key, tensor in data.items():
            if key not in self.stats["count"] or self.stats["count"][key] < 10:
                continue
                
            # Check mean shift (> 3 std devs)
            current_mean = tensor.float().mean().item()
            expected_mean = self.stats["mean"][key]
            expected_std = self.stats["std"][key]
            
            if expected_std > 0 and abs(current_mean - expected_mean) > 3 * expected_std:
                errors.append(f"{key}: Significant mean shift detected")
                
            # Check range violations
            if tensor.min().item() < self.stats["min"][key] * 0.5:
                errors.append(f"{key}: Values significantly below historical minimum")
            if tensor.max().item() > self.stats["max"][key] * 2.0:
                errors.append(f"{key}: Values significantly above historical maximum")
                
        return errors
    
    def _track_violation(self, field: str, rule: Optional[ValidationRule], data: torch.Tensor) -> None:
        """Track schema violations for evolution suggestions."""
        if rule is not None:
            key = f"{field}:{rule.__class__.__name__}"
        else:
            key = f"{field}:new_field"
            
        if key not in self.schema_violations:
            self.schema_violations[key] = 0
        self.schema_violations[key] += 1
        
        # Suggest new rules based on data
        if field not in self.suggested_rules:
            self.suggested_rules[field] = []
            
        # Infer appropriate rules
        if isinstance(data, torch.Tensor):
            self.suggested_rules[field] = [
                Shape(tuple("*" if d > 100 else d for d in data.shape)),
                DType(data.dtype),
                Range(data.min().item(), data.max().item()),
                NotNaN(),
                NotInf()
            ]
    
    def add_rule(self, field: str, rule: ValidationRule) -> None:
        """Add a validation rule to the schema."""
        if not self.schema:
            self.schema = Schema(rules={})
            
        if field not in self.schema.rules:
            self.schema.rules[field] = []
            
        self.schema.rules[field].append(rule)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current distribution statistics."""
        return {
            "statistics": self.stats,
            "cache_stats": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "size": len(self.cache)
            },
            "violations": self.schema_violations,
            "suggestions": self.suggested_rules
        }
    
    def save_schema(self, path: str) -> None:
        """Save schema to file."""
        if self.schema:
            with open(path, 'w') as f:
                json.dump(self.schema.to_dict(), f, indent=2)
                
    def suggest_schema_updates(self) -> Dict[str, List[str]]:
        """Suggest schema updates based on violations."""
        suggestions = {}
        
        for field, rules in self.suggested_rules.items():
            suggestions[field] = [
                f"Consider {rule.__class__.__name__} rule"
                for rule in rules
            ]
            
        return suggestions