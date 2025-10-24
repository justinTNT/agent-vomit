#!/usr/bin/env python3
"""
UNIVERSAL CONFIG FRAMEWORK
Standardized configuration patterns for all bulletproof modules
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import warnings
import time
from typing import Dict, List, Optional, Any, Union, Callable, Type, Tuple, get_type_hints
from dataclasses import dataclass, field, fields, asdict
from abc import ABC, abstractmethod
from enum import Enum
from collections import defaultdict
import inspect
import json
from pathlib import Path

from rave_config_system import RAVEConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParameterValidationError(Exception):
    """Exception raised when parameter validation fails"""
    pass


class ConfigurationError(Exception):
    """Exception raised when configuration is invalid"""
    pass


class FallbackActivated(Warning):
    """Warning raised when fallback configuration is activated"""
    pass


class ConfigPriority(Enum):
    """Configuration priority levels"""
    DEFAULT = 1
    GENERAL_CONFIG = 2
    COMPONENT_CONFIG = 3
    KWARGS = 4  # Highest priority


@dataclass
class ValidationRule:
    """Rule for parameter validation"""
    name: str
    validator: Callable[[Any], bool]
    error_message: str
    severity: str = "error"  # "error", "warning", "info"
    
    def validate(self, value: Any) -> bool:
        """Validate a value against this rule"""
        try:
            return self.validator(value)
        except Exception as e:
            logger.warning(f"Validation rule {self.name} failed with exception: {e}")
            return False


@dataclass 
class ParameterSpec:
    """Specification for a parameter"""
    name: str
    param_type: Type
    default_value: Any = None
    required: bool = False
    validation_rules: List[ValidationRule] = field(default_factory=list)
    description: str = ""
    category: str = "general"
    
    def validate(self, value: Any) -> Tuple[bool, List[str]]:
        """Validate a value against all rules"""
        errors = []
        warnings = []
        
        # Type validation
        if not isinstance(value, self.param_type) and value is not None:
            try:
                # Try to convert
                if self.param_type == bool and isinstance(value, (int, str)):
                    value = bool(value)
                elif self.param_type in (int, float) and isinstance(value, (int, float, str)):
                    value = self.param_type(value)
                elif self.param_type == str:
                    value = str(value)
                else:
                    errors.append(f"Parameter {self.name} must be of type {self.param_type.__name__}, got {type(value).__name__}")
            except (ValueError, TypeError):
                errors.append(f"Cannot convert {self.name}={value} to {self.param_type.__name__}")
        
        # Custom validation rules
        for rule in self.validation_rules:
            if not rule.validate(value):
                if rule.severity == "error":
                    errors.append(f"{self.name}: {rule.error_message}")
                elif rule.severity == "warning":
                    warnings.append(f"{self.name}: {rule.error_message}")
        
        success = len(errors) == 0
        all_messages = errors + warnings
        
        return success, all_messages


@dataclass
class ProcessingResult:
    """Standard result object for all bulletproof modules"""
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    fallback_used: bool = False
    
    def add_warning(self, warning: str):
        """Add a warning message"""
        self.warnings.append(warning)
    
    def add_stat(self, key: str, value: Any):
        """Add a statistic"""
        self.stats[key] = value
    
    def is_valid(self) -> bool:
        """Check if result is valid"""
        return self.success and self.error is None


class ComponentConfigBase(ABC):
    """Base class for component-specific configurations"""
    
    def __init__(self, **kwargs):
        # Set all provided kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    @abstractmethod
    def get_parameter_specs(cls) -> List[ParameterSpec]:
        """Get parameter specifications for this component"""
        pass
    
    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Get default parameters for this component"""
        specs = cls.get_parameter_specs()
        return {spec.name: spec.default_value for spec in specs}
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate all parameters in this config"""
        all_errors = []
        specs = self.get_parameter_specs()
        
        for spec in specs:
            value = getattr(self, spec.name, spec.default_value)
            success, messages = spec.validate(value)
            if not success:
                all_errors.extend(messages)
        
        return len(all_errors) == 0, all_errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {spec.name: getattr(self, spec.name, spec.default_value) 
                for spec in self.get_parameter_specs()}


class UniversalConfigMixin:
    """Mixin class that provides universal configuration capabilities"""
    
    # Class attributes to be overridden by subclasses
    config_section_name: str = None
    component_config_class: Type[ComponentConfigBase] = None
    valid_parameters: List[str] = []
    
    def __init__(self, config: RAVEConfig, **kwargs):
        """Universal initialization pattern"""
        
        # Initialize configuration management
        self.config = config
        self.enable_fallbacks = kwargs.get('enable_fallbacks', True)
        self.strict_mode = kwargs.get('strict_mode', False)
        
        # Statistics tracking
        self.stats = defaultdict(int)
        self.config_source_tracking = {}
        
        # Extract and validate component configuration
        self.component_config = self._extract_component_config(config, **kwargs)
        
        # Validate and sanitize parameters
        self._validate_and_sanitize_params()
        
        # Apply device and dtype settings
        self._apply_device_settings(config)
        
        # Track initialization success
        self.initialization_successful = True
        self.stats['initialization_time'] = time.time()
    
    def _extract_component_config(self, config: RAVEConfig, **kwargs) -> ComponentConfigBase:
        """Extract component-specific config with fallback hierarchy"""
        
        if not self.component_config_class:
            raise ValueError(f"Component {self.__class__.__name__} must define component_config_class")
        
        # Start with default parameters
        params = self.component_config_class.get_default_params()
        self._track_param_sources(params, ConfigPriority.DEFAULT)
        
        # Apply general config settings
        general_params = self._extract_general_config(config)
        params.update(general_params)
        self._track_param_sources(general_params, ConfigPriority.GENERAL_CONFIG)
        
        # Apply component-specific config if available
        if self.config_section_name and hasattr(config, self.config_section_name):
            specific_config = getattr(config, self.config_section_name)
            component_params = self._extract_component_params(specific_config)
            params.update(component_params)
            self._track_param_sources(component_params, ConfigPriority.COMPONENT_CONFIG)
        
        # Apply kwargs overrides (highest priority)
        validated_kwargs = self._validate_kwargs(kwargs)
        params.update(validated_kwargs)
        self._track_param_sources(validated_kwargs, ConfigPriority.KWARGS)
        
        # Create component config
        try:
            return self.component_config_class(**params)
        except Exception as e:
            logger.error(f"Failed to create component config: {e}")
            if self.enable_fallbacks:
                return self._get_universal_fallback()
            else:
                raise ConfigurationError(f"Component config creation failed: {e}")
    
    def _track_param_sources(self, params: Dict[str, Any], priority: ConfigPriority):
        """Track where each parameter came from"""
        for key in params:
            self.config_source_tracking[key] = priority
    
    def _extract_general_config(self, config: RAVEConfig) -> Dict[str, Any]:
        """Extract general configuration parameters"""
        general_params = {}
        
        # Only extract parameters that are valid for the component config
        if self.component_config_class and hasattr(self.component_config_class, 'get_parameter_specs'):
            valid_param_names = {spec.name for spec in self.component_config_class.get_parameter_specs()}
            
            # Common parameters available in RAVEConfig
            if 'device' in valid_param_names and hasattr(config, 'device'):
                general_params['device'] = config.device
            if 'dtype' in valid_param_names and hasattr(config, 'dtype'):
                general_params['dtype'] = config.dtype
            if 'enable_extreme_mode' in valid_param_names and hasattr(config, 'enable_extreme_mode'):
                general_params['enable_extreme_mode'] = config.enable_extreme_mode
        
        return general_params
    
    def _extract_component_params(self, specific_config) -> Dict[str, Any]:
        """Extract parameters from component-specific config"""
        if hasattr(specific_config, '__dict__'):
            return {k: v for k, v in specific_config.__dict__.items() 
                   if not k.startswith('_')}
        elif hasattr(specific_config, 'to_dict'):
            return specific_config.to_dict()
        else:
            return {}
    
    def _validate_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and integrate kwargs with comprehensive validation"""
        
        validated_kwargs = {}
        warning_kwargs = {}
        ignored_kwargs = {}
        
        for key, value in kwargs.items():
            try:
                # Skip special kwargs
                if key in ['enable_fallbacks', 'strict_mode']:
                    continue
                
                # Check if parameter is recognized
                if self.valid_parameters and key in self.valid_parameters:
                    # Validate the parameter if we have specs
                    if self.component_config_class:
                        specs = {spec.name: spec for spec in self.component_config_class.get_parameter_specs()}
                        if key in specs:
                            success, messages = specs[key].validate(value)
                            if success:
                                validated_kwargs[key] = value
                            else:
                                logger.warning(f"Validation failed for {key}={value}: {messages}")
                                if self.enable_fallbacks:
                                    warning_kwargs[key] = value
                                elif self.strict_mode:
                                    raise ParameterValidationError(f"Invalid parameter {key}: {messages}")
                        else:
                            validated_kwargs[key] = value
                    else:
                        validated_kwargs[key] = value
                
                # Check for close matches (typo detection)
                elif self.valid_parameters and self._has_close_match(key):
                    suggested_key = self._get_close_match(key)
                    logger.warning(f"Parameter '{key}' not recognized, did you mean '{suggested_key}'?")
                    warning_kwargs[key] = value
                
                # Handle unknown parameters
                else:
                    if self.strict_mode:
                        raise ValueError(f"Unknown parameter: {key}")
                    else:
                        logger.debug(f"Ignoring unknown parameter: {key}")
                        ignored_kwargs[key] = value
                        
            except Exception as e:
                logger.error(f"Error processing parameter {key}={value}: {e}")
                if self.enable_fallbacks:
                    ignored_kwargs[key] = value
                else:
                    raise
        
        # Log summary
        if warning_kwargs or ignored_kwargs:
            logger.info(f"Kwargs processing: {len(validated_kwargs)} validated, "
                       f"{len(warning_kwargs)} warnings, {len(ignored_kwargs)} ignored")
        
        self.stats['validated_kwargs'] = len(validated_kwargs)
        self.stats['warning_kwargs'] = len(warning_kwargs)
        self.stats['ignored_kwargs'] = len(ignored_kwargs)
        
        return validated_kwargs
    
    def _has_close_match(self, key: str) -> bool:
        """Check if there's a close match for a parameter name"""
        if not self.valid_parameters:
            return False
        
        # Simple edit distance check
        for valid_key in self.valid_parameters:
            if self._edit_distance(key.lower(), valid_key.lower()) <= 2:
                return True
        return False
    
    def _get_close_match(self, key: str) -> str:
        """Get the closest matching parameter name"""
        if not self.valid_parameters:
            return key
        
        best_match = self.valid_parameters[0]
        best_distance = float('inf')
        
        for valid_key in self.valid_parameters:
            distance = self._edit_distance(key.lower(), valid_key.lower())
            if distance < best_distance:
                best_distance = distance
                best_match = valid_key
        
        return best_match
    
    def _edit_distance(self, s1: str, s2: str) -> int:
        """Calculate edit distance between two strings"""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _validate_and_sanitize_params(self):
        """Universal parameter validation with comprehensive error handling"""
        
        try:
            # Validate component config
            if hasattr(self.component_config, 'validate'):
                success, errors = self.component_config.validate()
                
                if not success:
                    if self.enable_fallbacks:
                        logger.warning(f"Parameter validation failed: {errors}, applying fallbacks")
                        self._apply_fallback_config()
                        self.stats['validation_fallbacks'] += 1
                    else:
                        raise ParameterValidationError(f"Parameter validation failed: {errors}")
            
            # Additional custom validation
            self._custom_validation()
            
        except Exception as e:
            logger.error(f"Unexpected validation error: {e}")
            if self.enable_fallbacks:
                self._apply_emergency_fallback()
                self.stats['emergency_fallbacks'] += 1
            else:
                raise
    
    def _custom_validation(self):
        """Override in subclasses for custom validation logic"""
        pass
    
    def _apply_device_settings(self, config: RAVEConfig):
        """Apply device and dtype settings"""
        try:
            # Get device configuration
            if hasattr(config, 'create_device_config'):
                device_config = config.create_device_config()
                self.device = device_config.get('device', torch.device('cpu'))
                self.dtype = getattr(config, 'dtype', torch.float32)
                
                # Apply device-specific settings
                if hasattr(self, 'to'):
                    self.to(self.device)
                
                logger.debug(f"Applied device settings: {self.device}, {self.dtype}")
            
        except Exception as e:
            logger.warning(f"Failed to apply device settings: {e}")
            self.device = torch.device('cpu')
            self.dtype = torch.float32
    
    def _apply_fallback_config(self):
        """Apply fallback configuration when primary config fails"""
        
        logger.warning("Applying fallback configuration")
        
        # Fallback hierarchy:
        # 1. Component-specific fallback
        # 2. Category-specific fallback  
        # 3. Universal minimal fallback
        
        fallback_applied = False
        
        # Try component-specific fallback
        if hasattr(self, '_get_component_fallback'):
            try:
                self.component_config = self._get_component_fallback()
                fallback_applied = True
                self.stats['component_fallbacks'] += 1
                logger.info("Applied component-specific fallback")
            except Exception as e:
                logger.warning(f"Component fallback failed: {e}")
        
        # Try category fallback
        if not fallback_applied and hasattr(self, '_get_category_fallback'):
            try:
                self.component_config = self._get_category_fallback()
                fallback_applied = True
                self.stats['category_fallbacks'] += 1
                logger.info("Applied category-specific fallback")
            except Exception as e:
                logger.warning(f"Category fallback failed: {e}")
        
        # Universal minimal fallback
        if not fallback_applied:
            self.component_config = self._get_universal_fallback()
            self.stats['universal_fallbacks'] += 1
            logger.info("Applied universal fallback")
        
        warnings.warn("Fallback configuration activated", FallbackActivated)
    
    def _get_universal_fallback(self) -> ComponentConfigBase:
        """Get universal minimal fallback configuration"""
        if self.component_config_class:
            # Use defaults from component config class
            return self.component_config_class()
        else:
            # Create a minimal config object
            return type('MinimalConfig', (), {})()
    
    def _apply_emergency_fallback(self):
        """Apply emergency fallback when all else fails"""
        logger.error("Applying emergency fallback configuration")
        
        # Minimal hardcoded fallback
        self.component_config = type('EmergencyConfig', (), {
            'device': torch.device('cpu'),
            'dtype': torch.float32,
            'enable_fallbacks': True,
            'validate': lambda: (True, [])
        })()
        
        self.stats['emergency_fallbacks'] += 1
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration"""
        return {
            'component_config': self.component_config.to_dict() if hasattr(self.component_config, 'to_dict') else str(self.component_config),
            'device': str(getattr(self, 'device', 'unknown')),
            'dtype': str(getattr(self, 'dtype', 'unknown')),
            'config_sources': self.config_source_tracking,
            'stats': dict(self.stats),
            'initialization_successful': getattr(self, 'initialization_successful', False)
        }
    
    def create_processing_result(self, data: Any, **kwargs) -> ProcessingResult:
        """Create a standardized processing result"""
        return ProcessingResult(
            data=data,
            metadata=kwargs.get('metadata', {}),
            success=kwargs.get('success', True),
            error=kwargs.get('error', None),
            warnings=kwargs.get('warnings', []),
            stats=kwargs.get('stats', {}),
            processing_time=kwargs.get('processing_time', 0.0),
            fallback_used=self.stats.get('universal_fallbacks', 0) > 0
        )


class BulletproofModuleBase(nn.Module, UniversalConfigMixin):
    """Base class for all bulletproof modules with universal config support"""
    
    def __init__(self, config: RAVEConfig, **kwargs):
        nn.Module.__init__(self)
        UniversalConfigMixin.__init__(self, config, **kwargs)
        
        # Initialize module-specific components
        self._initialize_module()
    
    @abstractmethod
    def _initialize_module(self):
        """Initialize module-specific components (override in subclasses)"""
        pass
    
    def forward(self, *args, **kwargs) -> ProcessingResult:
        """Standard forward method with error handling"""
        start_time = time.time()
        
        try:
            # Core processing
            result_data = self._process(*args, **kwargs)
            
            # Create standardized result
            processing_time = time.time() - start_time
            return self.create_processing_result(
                data=result_data,
                processing_time=processing_time,
                stats={'forward_calls': self.stats.get('forward_calls', 0) + 1}
            )
            
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            processing_time = time.time() - start_time
            
            if self.enable_fallbacks:
                fallback_data = self._get_processing_fallback(*args, **kwargs)
                return self.create_processing_result(
                    data=fallback_data,
                    success=False,
                    error=str(e),
                    processing_time=processing_time,
                    fallback_used=True
                )
            else:
                raise
    
    @abstractmethod
    def _process(self, *args, **kwargs) -> Any:
        """Core processing logic (override in subclasses)"""
        pass
    
    def _get_processing_fallback(self, *args, **kwargs) -> Any:
        """Get fallback processing result (override in subclasses)"""
        # Default fallback: return zeros with same shape as input
        if args and isinstance(args[0], torch.Tensor):
            return torch.zeros_like(args[0])
        else:
            return None


# Utility functions for creating standardized configurations

def create_validation_rule(name: str, condition: Callable[[Any], bool], 
                         error_msg: str, severity: str = "error") -> ValidationRule:
    """Create a validation rule"""
    return ValidationRule(
        name=name,
        validator=condition,
        error_message=error_msg,
        severity=severity
    )

def range_validator(min_val: float, max_val: float) -> Callable[[Any], bool]:
    """Create a range validation function"""
    def validator(value):
        return min_val <= value <= max_val
    return validator

def positive_validator() -> Callable[[Any], bool]:
    """Create a positive number validation function"""
    def validator(value):
        return value > 0
    return validator

def choices_validator(choices: List[Any]) -> Callable[[Any], bool]:
    """Create a choices validation function"""
    def validator(value):
        return value in choices
    return validator


if __name__ == "__main__":
    print("🔧 UNIVERSAL CONFIG FRAMEWORK")
    print("=" * 50)
    
    # Example usage
    from rave_config_system import get_minimal_config
    
    # Create a sample component config
    @dataclass
    class SampleComponentConfig(ComponentConfigBase):
        """Sample component configuration"""
        
        learning_rate: float = 1e-3
        batch_size: int = 32
        dropout: float = 0.1
        activation: str = "relu"
        
        @classmethod
        def get_parameter_specs(cls) -> List[ParameterSpec]:
            return [
                ParameterSpec(
                    name="learning_rate",
                    param_type=float,
                    default_value=1e-3,
                    validation_rules=[
                        create_validation_rule("positive", positive_validator(), "Learning rate must be positive"),
                        create_validation_rule("reasonable_range", range_validator(1e-6, 1.0), "Learning rate should be between 1e-6 and 1.0")
                    ]
                ),
                ParameterSpec(
                    name="batch_size",
                    param_type=int,
                    default_value=32,
                    validation_rules=[
                        create_validation_rule("positive", positive_validator(), "Batch size must be positive"),
                        create_validation_rule("power_of_2", lambda x: x & (x-1) == 0, "Batch size should be power of 2", "warning")
                    ]
                ),
                ParameterSpec(
                    name="dropout",
                    param_type=float,
                    default_value=0.1,
                    validation_rules=[
                        create_validation_rule("valid_range", range_validator(0.0, 1.0), "Dropout must be between 0 and 1")
                    ]
                ),
                ParameterSpec(
                    name="activation",
                    param_type=str,
                    default_value="relu",
                    validation_rules=[
                        create_validation_rule("valid_choice", choices_validator(["relu", "gelu", "swish", "tanh"]), "Invalid activation function")
                    ]
                )
            ]
    
    # Create a sample bulletproof module
    class SampleBulletproofModule(BulletproofModuleBase):
        config_section_name = "sample"
        component_config_class = SampleComponentConfig
        valid_parameters = ["learning_rate", "batch_size", "dropout", "activation"]
        
        def _initialize_module(self):
            """Initialize the sample module"""
            self.linear = nn.Linear(10, 1)
            logger.info(f"SampleBulletproofModule initialized with config: {self.component_config.to_dict()}")
        
        def _process(self, x: torch.Tensor) -> torch.Tensor:
            """Process input tensor"""
            return self.linear(x)
    
    # Test the framework
    config = get_minimal_config()
    
    # Test with valid parameters
    print("\n✅ Testing with valid parameters...")
    module = SampleBulletproofModule(config, learning_rate=0.001, batch_size=16, dropout=0.2)
    print(f"Config summary: {module.get_config_summary()}")
    
    # Test with invalid parameters (should trigger fallbacks)
    print("\n⚠️ Testing with invalid parameters...")
    module2 = SampleBulletproofModule(config, learning_rate=-0.5, batch_size=0, dropout=2.0, unknown_param="test")
    print(f"Config summary: {module2.get_config_summary()}")
    
    # Test processing
    print("\n🚀 Testing processing...")
    test_input = torch.randn(4, 10)
    result = module(test_input)
    print(f"Processing result: success={result.success}, shape={result.data.shape}")
    
    print("\n✅ Universal Config Framework ready for deployment!")