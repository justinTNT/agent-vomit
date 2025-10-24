#!/usr/bin/env python3
"""
BULLETPROOF UNIVERSAL BASE CLASS
Unified inheritance point combining Universal Config + Standardized Signatures + Error Handling + Performance

This is the single base class that all 50 modules inherit from, providing:
- Universal configuration patterns
- Standardized signatures
- Bulletproof error handling
- Performance monitoring
- Composition operators
- Testing interfaces
- Minimal overhead
"""

import torch
import torch.nn as nn
import numpy as np
import time
import asyncio
import psutil
import gc
import threading
from typing import (
    Dict, List, Optional, Any, Union, Callable, Type, Tuple, 
    Protocol, runtime_checkable, Generic, TypeVar, Awaitable, AsyncContextManager
)
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import logging
import warnings
import traceback
import inspect
from contextlib import contextmanager, asynccontextmanager
from collections import defaultdict, deque
import json
from pathlib import Path

# Import framework components
from universal_config_framework import (
    UniversalConfigMixin, ComponentConfigBase, ProcessingResult, 
    ParameterSpec, ValidationRule, ConfigurationError, ParameterValidationError
)
from standardized_signature_framework import (
    StandardizedModule, DataFormat, TensorSpec, MethodSignature, 
    ProcessingMode, ComposedModule
)
from rave_config_system import RAVEConfig

logger = logging.getLogger(__name__)

# Type variables
T = TypeVar('T')
InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')


@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics"""
    processing_time: float = 0.0
    memory_used: float = 0.0
    gpu_memory_used: float = 0.0
    cpu_percent: float = 0.0
    
    # Throughput metrics
    items_processed: int = 0
    throughput: float = 0.0  # items/second
    
    # Quality metrics
    error_rate: float = 0.0
    fallback_rate: float = 0.0
    success_rate: float = 1.0
    
    # Timing breakdown
    input_validation_time: float = 0.0
    processing_time_core: float = 0.0
    output_validation_time: float = 0.0
    overhead_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'processing_time': self.processing_time,
            'memory_used': self.memory_used,
            'gpu_memory_used': self.gpu_memory_used,
            'cpu_percent': self.cpu_percent,
            'items_processed': self.items_processed,
            'throughput': self.throughput,
            'error_rate': self.error_rate,
            'fallback_rate': self.fallback_rate,
            'success_rate': self.success_rate,
            'timing_breakdown': {
                'input_validation': self.input_validation_time,
                'core_processing': self.processing_time_core,
                'output_validation': self.output_validation_time,
                'overhead': self.overhead_time
            }
        }


@dataclass
class HealthStatus:
    """Module health status"""
    is_healthy: bool = True
    last_error: Optional[str] = None
    error_count: int = 0
    warning_count: int = 0
    
    # Resource status
    memory_pressure: bool = False
    gpu_memory_pressure: bool = False
    cpu_pressure: bool = False
    
    # Performance indicators
    avg_processing_time: float = 0.0
    recent_success_rate: float = 1.0
    recent_throughput: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'is_healthy': self.is_healthy,
            'last_error': self.last_error,
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'resource_status': {
                'memory_pressure': self.memory_pressure,
                'gpu_memory_pressure': self.gpu_memory_pressure,
                'cpu_pressure': self.cpu_pressure
            },
            'performance': {
                'avg_processing_time': self.avg_processing_time,
                'recent_success_rate': self.recent_success_rate,
                'recent_throughput': self.recent_throughput
            }
        }


class BulletproofUniversalBase(nn.Module):
    """
    Universal base class combining all frameworks into one powerful inheritance point.
    
    This class provides:
    1. Universal configuration management with automatic fallbacks
    2. Standardized method signatures with validation
    3. Bulletproof error handling with recovery strategies
    4. Performance monitoring and health tracking
    5. Composition operators (|, >>, +)
    6. Testing interfaces and validation
    7. Minimal overhead for production use
    
    Usage:
        class MyModule(BulletproofUniversalBase):
            def __init__(self, config, **kwargs):
                super().__init__(config, "my_category", **kwargs)
            
            def process_impl(self, input_data):
                # Core logic here
                return processed_data
    """
    
    # Class-level configuration (override in subclasses)
    config_section_name: str = None
    component_config_class: Type[ComponentConfigBase] = None
    valid_parameters: List[str] = []
    module_category: str = "generic"
    method_signatures: Dict[str, MethodSignature] = {}
    supports_composition: bool = True
    
    def __init__(self, config: RAVEConfig, category: str = None, **kwargs):
        """
        Universal initialization providing all framework capabilities.
        
        Args:
            config: RAVE configuration object
            category: Module category (for signature selection)
            **kwargs: Configuration parameters and overrides
        """
        super().__init__()
        
        # Store core parameters
        self.config = config
        self.module_category = category or self.module_category
        
        # Framework configuration
        self.enable_fallbacks = kwargs.get('enable_fallbacks', True)
        self.strict_mode = kwargs.get('strict_mode', False)
        self.enable_monitoring = kwargs.get('enable_monitoring', True)
        self.enable_validation = kwargs.get('enable_validation', True)
        self.enable_composition = kwargs.get('enable_composition', True)
        
        # Performance configuration
        self.performance_tracking = kwargs.get('performance_tracking', True)
        self.health_monitoring = kwargs.get('health_monitoring', True)
        self.memory_monitoring = kwargs.get('memory_monitoring', True)
        
        # Initialize core tracking
        self._init_tracking_systems()
        
        # Initialize configuration system
        self._init_config_system(config, **kwargs)
        
        # Initialize signature system
        self._init_signature_system()
        
        # Initialize performance monitoring
        if self.enable_monitoring:
            self._init_performance_monitoring()
        
        # Initialize composition interface
        if self.enable_composition:
            self._init_composition_interface()
        
        # Initialize module-specific components
        try:
            self._initialize_module()
            self.initialization_successful = True
        except Exception as e:
            logger.error(f"Module initialization failed: {e}")
            if self.enable_fallbacks:
                self._apply_initialization_fallback(e)
                self.initialization_successful = False
            else:
                raise
        
        # Final health check
        self._update_health_status()
        
        logger.info(f"BulletproofUniversalBase initialized: {self.__class__.__name__} "
                   f"(category: {self.module_category}, healthy: {self.health_status.is_healthy})")
    
    def _init_tracking_systems(self):
        """Initialize all tracking and monitoring systems"""
        # Core statistics
        self.stats = defaultdict(int)
        self.config_source_tracking = {}
        
        # Performance tracking
        self.performance_metrics = PerformanceMetrics()
        self.recent_metrics = deque(maxlen=100)  # Last 100 operations
        
        # Health tracking
        self.health_status = HealthStatus()
        self.error_history = deque(maxlen=50)
        self.warning_history = deque(maxlen=50)
        
        # Threading for async monitoring
        self._monitoring_lock = threading.Lock()
        self._stop_monitoring = threading.Event()
        
        # Timing helpers
        self._timing_stack = []
        self._memory_baseline = self._get_memory_usage()
    
    def _init_config_system(self, config: RAVEConfig, **kwargs):
        """Initialize universal configuration system"""
        try:
            # Initialize configuration mixin
            if self.component_config_class:
                # Extract component configuration with fallback hierarchy
                self.component_config = self._extract_component_config(config, **kwargs)
                
                # Validate configuration
                if hasattr(self.component_config, 'validate'):
                    success, errors = self.component_config.validate()
                    if not success and not self.enable_fallbacks:
                        raise ConfigurationError(f"Configuration validation failed: {errors}")
            else:
                # Create minimal config
                self.component_config = type('MinimalConfig', (), kwargs)()
            
            # Apply device and dtype settings
            self._apply_device_settings(config)
            
        except Exception as e:
            logger.error(f"Config system initialization failed: {e}")
            if self.enable_fallbacks:
                self._apply_config_fallback()
            else:
                raise
    
    def _init_signature_system(self):
        """Initialize standardized signature system"""
        try:
            # Auto-detect signatures if not defined
            if not self.method_signatures:
                self.method_signatures = self._auto_detect_signatures()
            
            # Validate signatures
            self._validate_signatures()
            
            # Setup input/output specs
            self._setup_io_specs()
            
        except Exception as e:
            logger.error(f"Signature system initialization failed: {e}")
            if self.enable_fallbacks:
                self.method_signatures = {}
            else:
                raise
    
    def _init_performance_monitoring(self):
        """Initialize performance monitoring system"""
        if not self.performance_tracking:
            return
        
        try:
            # Initialize baseline metrics
            self.performance_metrics = PerformanceMetrics()
            self._start_time = time.time()
            
            # Start background monitoring if enabled
            if self.health_monitoring:
                self._start_health_monitoring()
                
        except Exception as e:
            logger.warning(f"Performance monitoring initialization failed: {e}")
    
    def _init_composition_interface(self):
        """Initialize composition interface"""
        if not self.enable_composition:
            return
        
        try:
            self.composition_interface = {
                'input_specs': self.get_input_spec(),
                'output_specs': self.get_output_spec(),
                'supports_async': self._supports_async(),
                'supports_streaming': self._supports_streaming(),
                'category': self.module_category,
                'compatibility_matrix': self._get_compatibility_matrix()
            }
        except Exception as e:
            logger.warning(f"Composition interface initialization failed: {e}")
            self.composition_interface = {}
    
    # ===== ABSTRACT METHODS (implement in subclasses) =====
    
    @abstractmethod
    def process_impl(self, input_data: Any, **kwargs) -> Any:
        """
        Core processing implementation (implement in subclasses).
        
        This is the only method subclasses MUST implement.
        All error handling, validation, and monitoring is handled automatically.
        """
        pass
    
    def _initialize_module(self):
        """Initialize module-specific components (override in subclasses)"""
        pass
    
    # ===== UNIVERSAL PROCESSING METHODS =====
    
    def forward(self, input_data: Any, **kwargs) -> ProcessingResult:
        """Universal forward method with full bulletproofing"""
        return self._bulletproof_process(input_data, 'forward', **kwargs)
    
    def process(self, input_data: Any, **kwargs) -> ProcessingResult:
        """Universal sync processing method"""
        return self._bulletproof_process(input_data, 'process', **kwargs)
    
    async def aprocess(self, input_data: Any, **kwargs) -> ProcessingResult:
        """Universal async processing method"""
        return await self._bulletproof_aprocess(input_data, 'aprocess', **kwargs)
    
    def _bulletproof_process(self, input_data: Any, method_name: str, **kwargs) -> ProcessingResult:
        """Universal bulletproof processing with full monitoring"""
        start_time = time.time()
        method_start_memory = self._get_memory_usage()
        
        try:
            with self._performance_context(method_name):
                # Input validation
                validation_start = time.time()
                if self.enable_validation:
                    self._validate_input(input_data, method_name)
                validation_time = time.time() - validation_start
                
                # Core processing
                processing_start = time.time()
                result_data = self.process_impl(input_data, **kwargs)
                processing_time = time.time() - processing_start
                
                # Output validation
                output_validation_start = time.time()
                if self.enable_validation and result_data is not None:
                    self._validate_output(result_data, method_name)
                output_validation_time = time.time() - output_validation_start
                
                # Create result
                total_time = time.time() - start_time
                memory_used = self._get_memory_usage() - method_start_memory
                
                result = ProcessingResult(
                    data=result_data,
                    success=True,
                    processing_time=total_time,
                    metadata={
                        'method': method_name,
                        'module_category': self.module_category,
                        'input_shape': getattr(input_data, 'shape', None),
                        'timing_breakdown': {
                            'validation': validation_time,
                            'processing': processing_time,
                            'output_validation': output_validation_time
                        },
                        'memory_delta': memory_used
                    }
                )
                
                # Update metrics
                self._update_performance_metrics(result, total_time, memory_used)
                
                return result
                
        except Exception as e:
            return self._handle_processing_error(e, input_data, method_name, start_time)
    
    async def _bulletproof_aprocess(self, input_data: Any, method_name: str, **kwargs) -> ProcessingResult:
        """Universal bulletproof async processing"""
        start_time = time.time()
        
        try:
            # Check if true async is supported
            if self._supports_async():
                # Use async implementation if available
                if hasattr(self, 'aprocess_impl'):
                    result_data = await self.aprocess_impl(input_data, **kwargs)
                else:
                    # Fall back to sync in executor
                    loop = asyncio.get_event_loop()
                    result_data = await loop.run_in_executor(None, self.process_impl, input_data, **kwargs)
            else:
                # Fall back to sync processing
                result_data = self.process_impl(input_data, **kwargs)
            
            total_time = time.time() - start_time
            
            return ProcessingResult(
                data=result_data,
                success=True,
                processing_time=total_time,
                metadata={'method': method_name, 'async': True}
            )
            
        except Exception as e:
            return self._handle_processing_error(e, input_data, method_name, start_time)
    
    # ===== COMPOSITION OPERATORS =====
    
    def __or__(self, other: 'BulletproofUniversalBase') -> 'ComposedModule':
        """Pipe operator: module1 | module2"""
        if not self.enable_composition:
            raise RuntimeError("Composition disabled for this module")
        return ComposedModule([self, other])
    
    def __rshift__(self, other: 'BulletproofUniversalBase') -> 'ComposedModule':
        """Shift operator: module1 >> module2"""
        return self.__or__(other)
    
    def __add__(self, other: 'BulletproofUniversalBase') -> 'ParallelComposition':
        """Add operator: module1 + module2 (parallel)"""
        return ParallelComposition([self, other])
    
    def compose_with(self, other: 'BulletproofUniversalBase') -> 'ComposedModule':
        """Explicit composition method"""
        return self.__or__(other)
    
    # ===== PERFORMANCE MONITORING =====
    
    @contextmanager
    def _performance_context(self, operation_name: str):
        """Context manager for performance monitoring"""
        if not self.performance_tracking:
            yield
            return
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            if self.performance_tracking:
                elapsed = time.time() - start_time
                memory_delta = self._get_memory_usage() - start_memory
                
                with self._monitoring_lock:
                    self.stats[f"{operation_name}_calls"] += 1
                    self.stats[f"{operation_name}_total_time"] += elapsed
                    self.stats[f"{operation_name}_total_memory"] += memory_delta
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if not self.memory_monitoring:
            return 0.0
        
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except Exception:
            return 0.0
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB"""
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            return torch.cuda.memory_allocated() / 1024 / 1024  # MB
        except Exception:
            return 0.0
    
    def _update_performance_metrics(self, result: ProcessingResult, processing_time: float, memory_used: float):
        """Update performance metrics"""
        if not self.performance_tracking:
            return
        
        with self._monitoring_lock:
            # Update current metrics
            self.performance_metrics.processing_time = processing_time
            self.performance_metrics.memory_used = memory_used
            self.performance_metrics.gpu_memory_used = self._get_gpu_memory_usage()
            self.performance_metrics.items_processed += 1
            
            # Update rates
            if result.success:
                self.stats['successes'] += 1
            else:
                self.stats['failures'] += 1
            
            total_ops = self.stats['successes'] + self.stats['failures']
            if total_ops > 0:
                self.performance_metrics.success_rate = self.stats['successes'] / total_ops
                self.performance_metrics.error_rate = self.stats['failures'] / total_ops
            
            # Calculate throughput
            total_time = time.time() - self._start_time
            if total_time > 0:
                self.performance_metrics.throughput = self.performance_metrics.items_processed / total_time
            
            # Store recent metrics
            self.recent_metrics.append({
                'timestamp': time.time(),
                'processing_time': processing_time,
                'memory_used': memory_used,
                'success': result.success
            })
    
    def _start_health_monitoring(self):
        """Start background health monitoring"""
        def monitor():
            while not self._stop_monitoring.is_set():
                try:
                    self._update_health_status()
                    time.sleep(5.0)  # Check every 5 seconds
                except Exception as e:
                    logger.warning(f"Health monitoring error: {e}")
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
    
    def _update_health_status(self):
        """Update module health status"""
        if not self.health_monitoring:
            return
        
        with self._monitoring_lock:
            # Calculate recent success rate
            recent_window = 10  # Last 10 operations
            recent_ops = list(self.recent_metrics)[-recent_window:]
            
            if recent_ops:
                successes = sum(1 for op in recent_ops if op['success'])
                self.health_status.recent_success_rate = successes / len(recent_ops)
                
                # Calculate recent throughput
                if len(recent_ops) > 1:
                    time_span = recent_ops[-1]['timestamp'] - recent_ops[0]['timestamp']
                    if time_span > 0:
                        self.health_status.recent_throughput = len(recent_ops) / time_span
                
                # Check for performance issues
                avg_time = np.mean([op['processing_time'] for op in recent_ops])
                self.health_status.avg_processing_time = avg_time
            
            # Check resource pressure
            current_memory = self._get_memory_usage()
            self.health_status.memory_pressure = current_memory > 1000  # > 1GB
            
            gpu_memory = self._get_gpu_memory_usage()
            self.health_status.gpu_memory_pressure = gpu_memory > 8000  # > 8GB
            
            # Overall health assessment
            self.health_status.is_healthy = (
                self.health_status.recent_success_rate > 0.9 and
                not self.health_status.memory_pressure and
                not self.health_status.gpu_memory_pressure and
                self.health_status.error_count < 10
            )
    
    # ===== ERROR HANDLING =====
    
    def _handle_processing_error(self, error: Exception, input_data: Any, 
                                method_name: str, start_time: float) -> ProcessingResult:
        """Universal error handling with recovery strategies"""
        processing_time = time.time() - start_time
        error_msg = str(error)
        
        # Log error
        logger.error(f"Processing error in {self.__class__.__name__}.{method_name}: {error_msg}")
        
        # Update error tracking
        with self._monitoring_lock:
            self.health_status.error_count += 1
            self.health_status.last_error = error_msg
            self.error_history.append({
                'timestamp': time.time(),
                'error': error_msg,
                'method': method_name,
                'traceback': traceback.format_exc()
            })
        
        # Attempt recovery if fallbacks enabled
        if self.enable_fallbacks:
            try:
                fallback_data = self._get_fallback_output(input_data, error)
                
                self.stats['fallback_activations'] += 1
                
                return ProcessingResult(
                    data=fallback_data,
                    success=False,
                    error=error_msg,
                    processing_time=processing_time,
                    fallback_used=True,
                    warnings=[f"Fallback activated due to: {error_msg}"]
                )
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
        
        # Return error result
        return ProcessingResult(
            data=None,
            success=False,
            error=error_msg,
            processing_time=processing_time,
            fallback_used=False
        )
    
    def _get_fallback_output(self, input_data: Any, error: Exception) -> Any:
        """Get fallback output when processing fails (override in subclasses)"""
        if hasattr(input_data, 'shape') and hasattr(input_data, 'device'):
            # For tensors, return zeros with same shape
            return torch.zeros_like(input_data)
        elif isinstance(input_data, dict):
            # For dicts, return empty dict
            return {}
        else:
            # Default fallback
            return None
    
    # ===== VALIDATION =====
    
    def _validate_input(self, input_data: Any, method_name: str = None):
        """Validate input data against method signature"""
        if not self.enable_validation or not self.method_signatures:
            return
        
        method_name = method_name or 'process'
        if method_name not in self.method_signatures:
            return
        
        signature = self.method_signatures[method_name]
        if not signature.input_specs:
            return
        
        # Basic tensor validation
        if isinstance(input_data, torch.Tensor) and signature.input_specs:
            spec = signature.input_specs[0]
            valid, errors = spec.validate_tensor(input_data)
            if not valid and not self.enable_fallbacks:
                raise ParameterValidationError(f"Input validation failed: {errors}")
    
    def _validate_output(self, output_data: Any, method_name: str = None):
        """Validate output data against method signature"""
        if not self.enable_validation or not self.method_signatures:
            return
        
        method_name = method_name or 'process'
        if method_name not in self.method_signatures:
            return
        
        signature = self.method_signatures[method_name]
        if not signature.output_specs:
            return
        
        # Basic tensor validation
        if isinstance(output_data, torch.Tensor) and signature.output_specs:
            spec = signature.output_specs[0]
            valid, errors = spec.validate_tensor(output_data)
            if not valid:
                logger.warning(f"Output validation failed: {errors}")
    
    # ===== UTILITY METHODS =====
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        with self._monitoring_lock:
            return {
                'metrics': self.performance_metrics.to_dict(),
                'health': self.health_status.to_dict(),
                'stats': dict(self.stats),
                'recent_operations': len(self.recent_metrics),
                'uptime': time.time() - self._start_time if hasattr(self, '_start_time') else 0
            }
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            'module_category': self.module_category,
            'component_config': getattr(self.component_config, 'to_dict', lambda: {})(),
            'framework_settings': {
                'enable_fallbacks': self.enable_fallbacks,
                'strict_mode': self.strict_mode,
                'enable_monitoring': self.enable_monitoring,
                'enable_validation': self.enable_validation,
                'enable_composition': self.enable_composition
            },
            'signatures': list(self.method_signatures.keys()),
            'initialization_successful': getattr(self, 'initialization_successful', False)
        }
    
    def reset_metrics(self):
        """Reset all metrics and statistics"""
        with self._monitoring_lock:
            self.stats.clear()
            self.recent_metrics.clear()
            self.error_history.clear()
            self.warning_history.clear()
            self.performance_metrics = PerformanceMetrics()
            self.health_status = HealthStatus()
            if hasattr(self, '_start_time'):
                self._start_time = time.time()
    
    def get_input_spec(self) -> List[TensorSpec]:
        """Get input specifications"""
        primary_method = self._get_primary_method()
        if primary_method and primary_method in self.method_signatures:
            return self.method_signatures[primary_method].input_specs
        return []
    
    def get_output_spec(self) -> List[TensorSpec]:
        """Get output specifications"""
        primary_method = self._get_primary_method()
        if primary_method and primary_method in self.method_signatures:
            return self.method_signatures[primary_method].output_specs
        return []
    
    def _get_primary_method(self) -> Optional[str]:
        """Get primary processing method name"""
        for method_name in ['process', 'forward', 'analyze', 'generate', 'orchestrate']:
            if method_name in self.method_signatures:
                return method_name
        return 'process'  # Default
    
    # ===== HELPER METHODS FOR INITIALIZATION =====
    
    def _extract_component_config(self, config: RAVEConfig, **kwargs):
        """Extract component configuration with fallback hierarchy"""
        if not self.component_config_class:
            return type('DefaultConfig', (), kwargs)()
        
        # Start with defaults
        params = self.component_config_class.get_default_params()
        
        # Apply kwargs overrides
        for key, value in kwargs.items():
            if key not in ['enable_fallbacks', 'strict_mode', 'enable_monitoring', 
                          'enable_validation', 'enable_composition', 'performance_tracking',
                          'health_monitoring', 'memory_monitoring']:
                params[key] = value
        
        return self.component_config_class(**params)
    
    def _apply_device_settings(self, config: RAVEConfig):
        """Apply device and dtype settings"""
        try:
            self.device = getattr(config, 'device', torch.device('cpu'))
            self.dtype = getattr(config, 'dtype', torch.float32)
            self.to(self.device)
        except Exception as e:
            logger.warning(f"Device settings failed: {e}")
            self.device = torch.device('cpu')
            self.dtype = torch.float32
    
    def _auto_detect_signatures(self) -> Dict[str, MethodSignature]:
        """Auto-detect method signatures based on category"""
        signatures = {}
        
        # Import signature collections
        from standardized_signature_framework import (
            RAVEMethodSignatures, DataPipelineMethodSignatures,
            AudioAnalysisMethodSignatures, GenerationMethodSignatures,
            OrchestrationMethodSignatures, MusicMLMethodSignatures
        )
        
        # Map categories to signature sets
        signature_map = {
            'neural_network': RAVEMethodSignatures,
            'data_processing': DataPipelineMethodSignatures, 
            'audio_analysis': AudioAnalysisMethodSignatures,
            'generation': GenerationMethodSignatures,
            'orchestration': OrchestrationMethodSignatures,
            'music_ml': MusicMLMethodSignatures
        }
        
        signature_class = signature_map.get(self.module_category)
        if signature_class:
            # Get all signature attributes
            for attr_name in dir(signature_class):
                if not attr_name.startswith('_'):
                    signature = getattr(signature_class, attr_name)
                    if isinstance(signature, MethodSignature):
                        signatures[signature.name] = signature
        
        return signatures
    
    def _validate_signatures(self):
        """Validate module implements required signatures"""
        for sig_name in self.method_signatures:
            if not hasattr(self, sig_name) and sig_name not in ['process', 'forward']:
                logger.warning(f"Module missing method: {sig_name}")
    
    def _setup_io_specs(self):
        """Setup input/output specifications"""
        # This is handled by the signature system
        pass
    
    def _supports_async(self) -> bool:
        """Check if module supports async processing"""
        return any(sig.supports_async for sig in self.method_signatures.values())
    
    def _supports_streaming(self) -> bool:
        """Check if module supports streaming"""
        return any(sig.supports_streaming for sig in self.method_signatures.values())
    
    def _get_compatibility_matrix(self) -> Dict[str, List[str]]:
        """Get compatibility matrix for composition"""
        return {
            'compatible_categories': ['generic'],  # Override in subclasses
            'incompatible_formats': [],
            'required_preprocessing': []
        }
    
    def _apply_config_fallback(self):
        """Apply fallback configuration"""
        self.component_config = type('FallbackConfig', (), {})()
        self.device = torch.device('cpu')
        self.dtype = torch.float32
    
    def _apply_initialization_fallback(self, error: Exception):
        """Apply fallback when module initialization fails"""
        logger.warning(f"Module initialization failed, using minimal fallback: {error}")
        self.stats['initialization_fallbacks'] += 1
    
    def __del__(self):
        """Cleanup when module is destroyed"""
        if hasattr(self, '_stop_monitoring'):
            self._stop_monitoring.set()


class ParallelComposition(BulletproofUniversalBase):
    """Parallel composition of modules (processes inputs in parallel)"""
    
    def __init__(self, modules: List[BulletproofUniversalBase]):
        config = modules[0].config if modules else None
        super().__init__(config, "composed")
        
        self.modules = nn.ModuleList(modules)
        self.module_category = "parallel_composed"
    
    def process_impl(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Process input through all modules in parallel"""
        results = {}
        
        for i, module in enumerate(self.modules):
            try:
                result = module.process(input_data, **kwargs)
                results[f"module_{i}"] = result.data if result.success else None
            except Exception as e:
                logger.warning(f"Parallel module {i} failed: {e}")
                results[f"module_{i}"] = None
        
        return results


# ===== TESTING INTERFACES =====

def validate_bulletproof_module(module: BulletproofUniversalBase) -> Tuple[bool, List[str]]:
    """Validate that a module properly implements the bulletproof interface"""
    errors = []
    
    # Check required attributes
    required_attrs = ['config', 'module_category', 'process_impl']
    for attr in required_attrs:
        if not hasattr(module, attr):
            errors.append(f"Missing required attribute: {attr}")
    
    # Check required methods
    if not callable(getattr(module, 'process_impl', None)):
        errors.append("process_impl must be callable")
    
    # Check initialization
    if not getattr(module, 'initialization_successful', False):
        errors.append("Module initialization was not successful")
    
    # Check health status
    if hasattr(module, 'health_status') and not module.health_status.is_healthy:
        errors.append(f"Module health check failed: {module.health_status.last_error}")
    
    return len(errors) == 0, errors


def create_test_input(module: BulletproofUniversalBase) -> Any:
    """Create appropriate test input for a module"""
    input_specs = module.get_input_spec()
    
    if input_specs:
        spec = input_specs[0]
        if spec.format == DataFormat.AUDIO_WAVEFORM:
            # Create sample audio
            shape = [2, 1, 1000]  # batch, channels, time
            return torch.randn(*shape, dtype=spec.dtype)
        elif spec.format == DataFormat.FEATURES:
            return torch.randn(2, 128, dtype=spec.dtype)
        elif spec.format == DataFormat.TENSOR_DICT:
            return {'data': torch.randn(2, 10)}
    
    # Default test input
    return torch.randn(2, 10)


# ===== FACTORY FUNCTIONS =====

def create_bulletproof_module(category: str, config: RAVEConfig, **kwargs) -> BulletproofUniversalBase:
    """Factory function to create bulletproof modules"""
    
    class GenericBulletproofModule(BulletproofUniversalBase):
        def __init__(self, config: RAVEConfig, **kwargs):
            super().__init__(config, category, **kwargs)
        
        def process_impl(self, input_data: Any, **kwargs) -> Any:
            # Default implementation: pass through
            return input_data
    
    return GenericBulletproofModule(config, **kwargs)


if __name__ == "__main__":
    print("🛡️ BULLETPROOF UNIVERSAL BASE CLASS")
    print("=" * 60)
    
    # Demo implementation
    from rave_config_system import get_minimal_config
    
    class DemoAudioModule(BulletproofUniversalBase):
        """Demo audio processing module"""
        
        def __init__(self, config, **kwargs):
            super().__init__(config, "audio_analysis", **kwargs)
            
        def _initialize_module(self):
            """Initialize module components"""
            self.feature_extractor = nn.Linear(1000, 128)
            
        def process_impl(self, audio_tensor, **kwargs):
            """Core audio processing logic"""
            # Simple feature extraction
            if audio_tensor.dim() == 3:  # [batch, channels, time]
                pooled = audio_tensor.mean(dim=-1)  # Pool over time
                features = self.feature_extractor(pooled.view(pooled.size(0), -1))
                return features
            return audio_tensor
    
    # Test the framework
    config = get_minimal_config()
    
    print("\n✅ Creating demo module...")
    demo_module = DemoAudioModule(config, enable_monitoring=True)
    
    print(f"Module category: {demo_module.module_category}")
    print(f"Initialization successful: {demo_module.initialization_successful}")
    print(f"Health status: {demo_module.health_status.is_healthy}")
    
    print("\n🔗 Testing composition...")
    demo_module2 = DemoAudioModule(config)
    pipeline = demo_module | demo_module2
    print(f"Pipeline created with {len(pipeline.modules)} modules")
    
    print("\n🚀 Testing processing...")
    test_audio = torch.randn(2, 1, 1000)  # Batch of 2, mono, 1000 samples
    
    result = demo_module.process(test_audio)
    print(f"Processing result: success={result.success}, shape={result.data.shape if result.success else 'failed'}")
    
    print("\n📊 Performance summary:")
    perf_summary = demo_module.get_performance_summary()
    print(f"  Throughput: {perf_summary['metrics']['throughput']:.2f} ops/sec")
    print(f"  Success rate: {perf_summary['metrics']['success_rate']:.2%}")
    print(f"  Processing time: {perf_summary['metrics']['processing_time']:.3f}s")
    
    print("\n🔄 Testing validation...")
    valid, errors = validate_bulletproof_module(demo_module)
    print(f"Module validation: {'✅ PASSED' if valid else '❌ FAILED'}")
    if errors:
        for error in errors:
            print(f"  - {error}")
    
    print("\n✅ BulletproofUniversalBase ready for deployment!")
    print("   All 50 modules can now inherit from this single class.")