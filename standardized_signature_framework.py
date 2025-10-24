#!/usr/bin/env python3
"""
STANDARDIZED SIGNATURE FRAMEWORK
Universal method signatures and interfaces for frictionless module composition

This framework standardizes:
1. Method signatures across all module types
2. Input/Output formats and tensor specifications
3. Composition interfaces for seamless chaining
4. Async/Sync operation patterns
5. Pipeline integration standards

Based on the Universal Config Framework, this provides the interface layer
for bulletproof, composable modules.
"""

import torch
import torch.nn as nn
import numpy as np
import asyncio
import inspect
import time
from typing import (
    Dict, List, Optional, Any, Union, Callable, Type, Tuple, 
    Protocol, runtime_checkable, Generic, TypeVar, Awaitable
)
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import logging

from universal_config_framework import ProcessingResult, BulletproofModuleBase
from rave_config_system import RAVEConfig

logger = logging.getLogger(__name__)

# Type variables for generic interfaces
T = TypeVar('T')
InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')


class DataFormat(Enum):
    """Standard data formats for module inputs/outputs"""
    AUDIO_WAVEFORM = "audio_waveform"          # [batch, channels, time]
    SPECTROGRAM = "spectrogram"                # [batch, freq_bins, time_frames]
    MEL_SPECTROGRAM = "mel_spectrogram"        # [batch, mel_bins, time_frames]
    FEATURES = "features"                      # [batch, feature_dim]
    SEQUENCE = "sequence"                      # [batch, seq_len, feature_dim]
    LATENT = "latent"                         # [batch, latent_dim]
    TOKENS = "tokens"                         # [batch, seq_len] (discrete)
    GRAPH = "graph"                           # Special graph structures
    TENSOR_DICT = "tensor_dict"               # Dict[str, torch.Tensor]


@dataclass
class TensorSpec:
    """Specification for tensor inputs/outputs"""
    format: DataFormat
    shape: Tuple[Union[int, str], ...]  # Use "*" for variable dimensions
    dtype: torch.dtype = torch.float32
    device: Optional[str] = None
    name: str = ""
    description: str = ""
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    def validate_tensor(self, tensor: torch.Tensor) -> Tuple[bool, List[str]]:
        """Validate tensor against specification"""
        errors = []
        
        # Shape validation
        if len(tensor.shape) != len(self.shape):
            errors.append(f"Expected {len(self.shape)} dimensions, got {len(tensor.shape)}")
        else:
            for i, (expected, actual) in enumerate(zip(self.shape, tensor.shape)):
                if expected != "*" and expected != actual:
                    errors.append(f"Dimension {i}: expected {expected}, got {actual}")
        
        # Dtype validation
        if tensor.dtype != self.dtype:
            errors.append(f"Expected dtype {self.dtype}, got {tensor.dtype}")
        
        # Device validation
        if self.device and tensor.device.type != self.device:
            errors.append(f"Expected device {self.device}, got {tensor.device}")
        
        # Constraint validation
        for constraint, value in self.constraints.items():
            if constraint == "min_value" and tensor.min() < value:
                errors.append(f"Values below minimum {value}")
            elif constraint == "max_value" and tensor.max() > value:
                errors.append(f"Values above maximum {value}")
            elif constraint == "no_nan" and value and torch.isnan(tensor).any():
                errors.append("Contains NaN values")
            elif constraint == "no_inf" and value and torch.isinf(tensor).any():
                errors.append("Contains infinite values")
        
        return len(errors) == 0, errors


@dataclass
class MethodSignature:
    """Standard method signature specification"""
    name: str
    input_specs: List[TensorSpec]
    output_specs: List[TensorSpec]
    required_kwargs: List[str] = field(default_factory=list)
    optional_kwargs: Dict[str, Any] = field(default_factory=dict)
    supports_async: bool = False
    supports_streaming: bool = False
    supports_batching: bool = True
    description: str = ""
    category: str = ""


class ProcessingMode(Enum):
    """Processing execution modes"""
    SYNC = "sync"                    # Standard synchronous processing
    ASYNC = "async"                  # Asynchronous processing
    STREAMING = "streaming"          # Real-time streaming processing
    BATCH = "batch"                  # Batch processing
    INTERACTIVE = "interactive"      # Interactive/incremental processing


@runtime_checkable
class StandardProcessor(Protocol[InputType, OutputType]):
    """Standard processor interface for all modules"""
    
    def process(self, input_data: InputType, **kwargs) -> ProcessingResult[OutputType]:
        """Standard synchronous processing method"""
        ...
    
    async def aprocess(self, input_data: InputType, **kwargs) -> ProcessingResult[OutputType]:
        """Standard asynchronous processing method"""
        ...
    
    def get_input_spec(self) -> List[TensorSpec]:
        """Get input tensor specifications"""
        ...
    
    def get_output_spec(self) -> List[TensorSpec]:
        """Get output tensor specifications"""
        ...


@runtime_checkable
class Composable(Protocol):
    """Interface for composable modules"""
    
    def compose_with(self, other: 'Composable') -> 'ComposedModule':
        """Compose this module with another"""
        ...
    
    def __or__(self, other: 'Composable') -> 'ComposedModule':
        """Enable |> syntax for composition"""
        ...
    
    def get_composition_interface(self) -> Dict[str, Any]:
        """Get interface information for composition"""
        ...


@runtime_checkable
class StreamingProcessor(Protocol[InputType, OutputType]):
    """Interface for streaming processors"""
    
    def process_chunk(self, chunk: InputType, **kwargs) -> ProcessingResult[OutputType]:
        """Process a single chunk in streaming mode"""
        ...
    
    def reset_state(self) -> None:
        """Reset internal state for new stream"""
        ...
    
    def get_buffer_size(self) -> int:
        """Get required buffer size for streaming"""
        ...


# Standard method signatures for each module category

class RAVEMethodSignatures:
    """Standard method signatures for RAVE neural network modules"""
    
    FORWARD = MethodSignature(
        name="forward",
        input_specs=[
            TensorSpec(
                format=DataFormat.AUDIO_WAVEFORM,
                shape=("*", 1, "*"),
                name="audio_input",
                description="Input audio waveform"
            )
        ],
        output_specs=[
            TensorSpec(
                format=DataFormat.TENSOR_DICT,
                shape=(),  # Variable structure
                name="model_output",
                description="Model outputs with features, reconstruction, etc."
            )
        ],
        required_kwargs=[],
        optional_kwargs={
            "return_latents": False,
            "return_features": False,
            "training": True
        },
        supports_async=True,
        supports_batching=True,
        category="neural_network"
    )
    
    ENCODE = MethodSignature(
        name="encode",
        input_specs=[
            TensorSpec(
                format=DataFormat.AUDIO_WAVEFORM,
                shape=("*", 1, "*"),
                name="audio_input"
            )
        ],
        output_specs=[
            TensorSpec(
                format=DataFormat.LATENT,
                shape=("*", "*"),
                name="encoded_latents"
            )
        ],
        supports_async=True,
        category="neural_network"
    )
    
    DECODE = MethodSignature(
        name="decode", 
        input_specs=[
            TensorSpec(
                format=DataFormat.LATENT,
                shape=("*", "*"),
                name="latent_input"
            )
        ],
        output_specs=[
            TensorSpec(
                format=DataFormat.AUDIO_WAVEFORM,
                shape=("*", 1, "*"),
                name="reconstructed_audio"
            )
        ],
        supports_async=True,
        category="neural_network"
    )


class DataPipelineMethodSignatures:
    """Standard method signatures for data pipeline modules"""
    
    PROCESS = MethodSignature(
        name="process",
        input_specs=[
            TensorSpec(
                format=DataFormat.TENSOR_DICT,
                shape=(),
                name="input_data",
                description="Input data in various formats"
            )
        ],
        output_specs=[
            TensorSpec(
                format=DataFormat.TENSOR_DICT,
                shape=(),
                name="processed_data",
                description="Processed output data"
            )
        ],
        supports_streaming=True,
        supports_async=True,
        category="data_processing"
    )
    
    VALIDATE = MethodSignature(
        name="validate",
        input_specs=[
            TensorSpec(
                format=DataFormat.TENSOR_DICT,
                shape=(),
                name="data_to_validate"
            )
        ],
        output_specs=[
            TensorSpec(
                format=DataFormat.TENSOR_DICT,
                shape=(),
                name="validation_result",
                description="Validation results and metadata"
            )
        ],
        category="data_processing"
    )
    
    TRANSFORM = MethodSignature(
        name="transform",
        input_specs=[
            TensorSpec(
                format=DataFormat.TENSOR_DICT,
                shape=(),
                name="input_data"
            )
        ],
        output_specs=[
            TensorSpec(
                format=DataFormat.TENSOR_DICT,
                shape=(),
                name="transformed_data"
            )
        ],
        supports_streaming=True,
        category="data_processing"
    )


class AudioAnalysisMethodSignatures:
    """Standard method signatures for audio analysis modules"""
    
    ANALYZE = MethodSignature(
        name="analyze",
        input_specs=[
            TensorSpec(
                format=DataFormat.AUDIO_WAVEFORM,
                shape=("*", 1, "*"),
                name="audio_input"
            )
        ],
        output_specs=[
            TensorSpec(
                format=DataFormat.TENSOR_DICT,
                shape=(),
                name="analysis_results",
                description="Analysis results including features, statistics, etc."
            )
        ],
        supports_async=True,
        supports_streaming=True,
        category="audio_analysis"
    )
    
    EXTRACT_FEATURES = MethodSignature(
        name="extract_features",
        input_specs=[
            TensorSpec(
                format=DataFormat.AUDIO_WAVEFORM,
                shape=("*", 1, "*"),
                name="audio_input"
            )
        ],
        output_specs=[
            TensorSpec(
                format=DataFormat.FEATURES,
                shape=("*", "*"),
                name="extracted_features"
            )
        ],
        supports_streaming=True,
        category="audio_analysis"
    )
    
    COMPUTE_SPECTRUM = MethodSignature(
        name="compute_spectrum",
        input_specs=[
            TensorSpec(
                format=DataFormat.AUDIO_WAVEFORM,
                shape=("*", 1, "*"),
                name="audio_input"
            )
        ],
        output_specs=[
            TensorSpec(
                format=DataFormat.SPECTROGRAM,
                shape=("*", "*", "*"),
                name="spectrum_output"
            )
        ],
        category="audio_analysis"
    )


class GenerationMethodSignatures:
    """Standard method signatures for generation modules (BigVGAN, etc.)"""
    
    GENERATE = MethodSignature(
        name="generate",
        input_specs=[
            TensorSpec(
                format=DataFormat.LATENT,
                shape=("*", "*"),
                name="latent_input",
                description="Latent codes or noise for generation"
            )
        ],
        output_specs=[
            TensorSpec(
                format=DataFormat.AUDIO_WAVEFORM,
                shape=("*", 1, "*"),
                name="generated_audio"
            )
        ],
        optional_kwargs={
            "temperature": 1.0,
            "top_k": None,
            "top_p": None,
            "length": None
        },
        supports_async=True,
        category="generation"
    )
    
    DISCRIMINATE = MethodSignature(
        name="discriminate",
        input_specs=[
            TensorSpec(
                format=DataFormat.AUDIO_WAVEFORM,
                shape=("*", 1, "*"),
                name="audio_input"
            )
        ],
        output_specs=[
            TensorSpec(
                format=DataFormat.FEATURES,
                shape=("*", 1),
                name="discrimination_scores",
                description="Real/fake discrimination scores"
            )
        ],
        category="generation"
    )


class OrchestrationMethodSignatures:
    """Standard method signatures for orchestration modules"""
    
    ORCHESTRATE = MethodSignature(
        name="orchestrate",
        input_specs=[
            TensorSpec(
                format=DataFormat.TENSOR_DICT,
                shape=(),
                name="workflow_input",
                description="Input data and workflow parameters"
            )
        ],
        output_specs=[
            TensorSpec(
                format=DataFormat.TENSOR_DICT,
                shape=(),
                name="workflow_output",
                description="Complete workflow results"
            )
        ],
        supports_async=True,
        required_kwargs=["workflow_config"],
        category="orchestration"
    )
    
    EXECUTE_PIPELINE = MethodSignature(
        name="execute_pipeline",
        input_specs=[
            TensorSpec(
                format=DataFormat.TENSOR_DICT,
                shape=(),
                name="pipeline_input"
            )
        ],
        output_specs=[
            TensorSpec(
                format=DataFormat.TENSOR_DICT,
                shape=(),
                name="pipeline_output"
            )
        ],
        supports_async=True,
        category="orchestration"
    )


class MusicMLMethodSignatures:
    """Standard method signatures for advanced music ML modules"""
    
    ANALYZE_MUSIC = MethodSignature(
        name="analyze_music",
        input_specs=[
            TensorSpec(
                format=DataFormat.AUDIO_WAVEFORM,
                shape=("*", 1, "*"),
                name="music_input"
            )
        ],
        output_specs=[
            TensorSpec(
                format=DataFormat.TENSOR_DICT,
                shape=(),
                name="music_analysis",
                description="Musical features, structure, harmony, etc."
            )
        ],
        supports_async=True,
        category="music_ml"
    )
    
    EXTRACT_HARMONY = MethodSignature(
        name="extract_harmony",
        input_specs=[
            TensorSpec(
                format=DataFormat.AUDIO_WAVEFORM,
                shape=("*", 1, "*"),
                name="audio_input"
            )
        ],
        output_specs=[
            TensorSpec(
                format=DataFormat.SEQUENCE,
                shape=("*", "*", "*"),
                name="harmonic_features",
                description="Chord progressions, key signatures, etc."
            )
        ],
        category="music_ml"
    )


class StandardizedModule(BulletproofModuleBase):
    """Base class for standardized modules with universal signatures"""
    
    # Override these in subclasses
    module_category: str = "generic"
    method_signatures: Dict[str, MethodSignature] = {}
    supports_composition: bool = True
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__(config, **kwargs)
        
        # Initialize signature validation
        self._validate_signatures()
        
        # Initialize composition interface
        if self.supports_composition:
            self._setup_composition_interface()
    
    def _validate_signatures(self):
        """Validate that module implements required signatures"""
        for sig_name, signature in self.method_signatures.items():
            if not hasattr(self, sig_name):
                logger.warning(f"Module {self.__class__.__name__} missing method {sig_name}")
    
    def _setup_composition_interface(self):
        """Setup composition interface"""
        self.composition_interface = {
            'input_specs': self.get_input_spec(),
            'output_specs': self.get_output_spec(),
            'supports_async': any(sig.supports_async for sig in self.method_signatures.values()),
            'supports_streaming': any(sig.supports_streaming for sig in self.method_signatures.values()),
            'category': self.module_category
        }
    
    def get_input_spec(self) -> List[TensorSpec]:
        """Get input specifications for primary method"""
        primary_method = self._get_primary_method()
        if primary_method and primary_method in self.method_signatures:
            return self.method_signatures[primary_method].input_specs
        return []
    
    def get_output_spec(self) -> List[TensorSpec]:
        """Get output specifications for primary method"""
        primary_method = self._get_primary_method()
        if primary_method and primary_method in self.method_signatures:
            return self.method_signatures[primary_method].output_specs
        return []
    
    def _get_primary_method(self) -> Optional[str]:
        """Get the primary processing method name"""
        # Priority order for finding primary method
        for method_name in ['process', 'forward', 'analyze', 'generate', 'orchestrate']:
            if method_name in self.method_signatures:
                return method_name
        return None
    
    def validate_input(self, input_data: torch.Tensor, method_name: str = None) -> Tuple[bool, List[str]]:
        """Validate input against method signature"""
        method_name = method_name or self._get_primary_method()
        if not method_name or method_name not in self.method_signatures:
            return True, []  # No validation available
        
        signature = self.method_signatures[method_name]
        if not signature.input_specs:
            return True, []
        
        # Validate against first input spec (primary input)
        input_spec = signature.input_specs[0]
        return input_spec.validate_tensor(input_data)
    
    def validate_output(self, output_data: torch.Tensor, method_name: str = None) -> Tuple[bool, List[str]]:
        """Validate output against method signature"""
        method_name = method_name or self._get_primary_method()
        if not method_name or method_name not in self.method_signatures:
            return True, []
        
        signature = self.method_signatures[method_name]
        if not signature.output_specs:
            return True, []
        
        # Validate against first output spec (primary output)
        output_spec = signature.output_specs[0]
        return output_spec.validate_tensor(output_data)
    
    async def aprocess(self, input_data: torch.Tensor, **kwargs) -> ProcessingResult:
        """Standard async processing method"""
        method_name = self._get_primary_method()
        if not method_name:
            raise NotImplementedError("No primary processing method found")
        
        signature = self.method_signatures.get(method_name)
        if not signature or not signature.supports_async:
            # Fall back to sync processing in executor
            import asyncio
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.process, input_data, **kwargs)
        
        # Native async implementation (override in subclasses)
        return await self._aprocess_impl(input_data, **kwargs)
    
    async def _aprocess_impl(self, input_data: torch.Tensor, **kwargs) -> ProcessingResult:
        """Async implementation (override in subclasses)"""
        return self.process(input_data, **kwargs)
    
    def process(self, input_data: torch.Tensor, **kwargs) -> ProcessingResult:
        """Standard sync processing method"""
        # Use forward as default implementation
        return self.forward(input_data, **kwargs)
    
    def get_composition_interface(self) -> Dict[str, Any]:
        """Get composition interface information"""
        return getattr(self, 'composition_interface', {})
    
    def compose_with(self, other: 'StandardizedModule') -> 'ComposedModule':
        """Compose with another module"""
        return ComposedModule([self, other])
    
    def __or__(self, other: 'StandardizedModule') -> 'ComposedModule':
        """Enable |> syntax: module1 | module2"""
        return self.compose_with(other)


class ComposedModule(StandardizedModule):
    """Module representing composition of multiple modules"""
    
    def __init__(self, modules: List[StandardizedModule]):
        # Use config from first module
        config = modules[0].config if modules else None
        super().__init__(config)
        
        self.modules = nn.ModuleList(modules)
        self.module_category = "composed"
        
        # Validate composition compatibility
        self._validate_composition()
        
        # Setup composed signatures
        self._setup_composed_signatures()
    
    def _validate_composition(self):
        """Validate that modules can be composed"""
        for i in range(len(self.modules) - 1):
            current_module = self.modules[i]
            next_module = self.modules[i + 1]
            
            current_output = current_module.get_output_spec()
            next_input = next_module.get_input_spec()
            
            if current_output and next_input:
                # Check compatibility
                self._check_compatibility(current_output[0], next_input[0])
    
    def _check_compatibility(self, output_spec: TensorSpec, input_spec: TensorSpec):
        """Check if output spec is compatible with input spec"""
        # Basic format compatibility
        if output_spec.format != input_spec.format and input_spec.format != DataFormat.TENSOR_DICT:
            logger.warning(f"Format mismatch: {output_spec.format} -> {input_spec.format}")
        
        # Shape compatibility (allowing flexible dimensions)
        if (len(output_spec.shape) != len(input_spec.shape) and 
            input_spec.format != DataFormat.TENSOR_DICT):
            logger.warning(f"Shape dimension mismatch: {output_spec.shape} -> {input_spec.shape}")
    
    def _setup_composed_signatures(self):
        """Setup signatures for composed module"""
        if self.modules:
            # Input comes from first module
            first_input = self.modules[0].get_input_spec()
            # Output comes from last module
            last_output = self.modules[-1].get_output_spec()
            
            self.method_signatures = {
                'process': MethodSignature(
                    name="process",
                    input_specs=first_input,
                    output_specs=last_output,
                    supports_async=all(m.composition_interface.get('supports_async', False) 
                                     for m in self.modules),
                    supports_streaming=all(m.composition_interface.get('supports_streaming', False) 
                                         for m in self.modules),
                    category="composed"
                )
            }
    
    def _initialize_module(self):
        """Initialize composed module"""
        pass  # Already initialized in __init__
    
    def _process(self, input_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """Process through all modules in sequence"""
        current_data = input_data
        
        for module in self.modules:
            result = module.process(current_data, **kwargs)
            
            if not result.success:
                raise RuntimeError(f"Module {module.__class__.__name__} failed: {result.error}")
            
            current_data = result.data
        
        return current_data
    
    async def _aprocess_impl(self, input_data: torch.Tensor, **kwargs) -> ProcessingResult:
        """Async processing through all modules"""
        current_data = input_data
        
        for module in self.modules:
            result = await module.aprocess(current_data, **kwargs)
            
            if not result.success:
                raise RuntimeError(f"Module {module.__class__.__name__} failed: {result.error}")
            
            current_data = result.data
        
        return self.create_processing_result(current_data)


# Utility functions for signature management

def register_method_signature(module_class: Type, method_name: str, signature: MethodSignature):
    """Register a method signature for a module class"""
    if not hasattr(module_class, 'method_signatures'):
        module_class.method_signatures = {}
    module_class.method_signatures[method_name] = signature


def validate_module_signatures(module: StandardizedModule) -> Tuple[bool, List[str]]:
    """Validate that a module implements all required signatures"""
    errors = []
    
    for sig_name, signature in module.method_signatures.items():
        if not hasattr(module, sig_name):
            errors.append(f"Missing method: {sig_name}")
            continue
        
        method = getattr(module, sig_name)
        if not callable(method):
            errors.append(f"Method {sig_name} is not callable")
            continue
        
        # Check method signature compatibility
        sig = inspect.signature(method)
        # Additional signature validation can be added here
    
    return len(errors) == 0, errors


def create_pipeline(*modules: StandardizedModule) -> ComposedModule:
    """Create a processing pipeline from multiple modules"""
    return ComposedModule(list(modules))


# Export main components
__all__ = [
    'DataFormat',
    'TensorSpec', 
    'MethodSignature',
    'ProcessingMode',
    'StandardProcessor',
    'Composable',
    'StreamingProcessor',
    'StandardizedModule',
    'ComposedModule',
    'RAVEMethodSignatures',
    'DataPipelineMethodSignatures', 
    'AudioAnalysisMethodSignatures',
    'GenerationMethodSignatures',
    'OrchestrationMethodSignatures',
    'MusicMLMethodSignatures',
    'register_method_signature',
    'validate_module_signatures',
    'create_pipeline'
]


if __name__ == "__main__":
    print("🔧 STANDARDIZED SIGNATURE FRAMEWORK")
    print("=" * 60)
    
    # Example usage demonstrating the framework
    from rave_config_system import get_minimal_config
    
    # Create a sample standardized module
    class SampleAudioProcessor(StandardizedModule):
        module_category = "audio_analysis"
        method_signatures = {
            'process': AudioAnalysisMethodSignatures.ANALYZE,
            'extract_features': AudioAnalysisMethodSignatures.EXTRACT_FEATURES
        }
        
        def _initialize_module(self):
            self.feature_extractor = nn.Linear(1, 128)
        
        def _process(self, input_data: torch.Tensor, **kwargs) -> torch.Tensor:
            # Simple feature extraction
            batch_size, channels, time_steps = input_data.shape
            pooled = input_data.mean(dim=-1)  # [batch, channels]
            features = self.feature_extractor(pooled)  # [batch, 128]
            
            return {
                'features': features,
                'input_shape': input_data.shape,
                'processing_time': time.time()
            }
    
    class SampleNeuralModule(StandardizedModule):
        module_category = "neural_network"
        method_signatures = {
            'forward': RAVEMethodSignatures.FORWARD,
            'encode': RAVEMethodSignatures.ENCODE
        }
        
        def _initialize_module(self):
            self.encoder = nn.Linear(128, 64)
            self.decoder = nn.Linear(64, 128)
        
        def _process(self, input_data: torch.Tensor, **kwargs) -> torch.Tensor:
            if isinstance(input_data, dict) and 'features' in input_data:
                features = input_data['features']
            else:
                features = input_data
            
            encoded = self.encoder(features)
            decoded = self.decoder(encoded)
            
            return {
                'encoded': encoded,
                'reconstructed': decoded,
                'original_features': features
            }
    
    # Test the framework
    config = get_minimal_config()
    
    print("\n✅ Creating standardized modules...")
    audio_processor = SampleAudioProcessor(config)
    neural_module = SampleNeuralModule(config)
    
    print(f"Audio processor category: {audio_processor.module_category}")
    print(f"Neural module category: {neural_module.module_category}")
    
    print("\n🔗 Testing module composition...")
    pipeline = audio_processor | neural_module
    print(f"Composed pipeline category: {pipeline.module_category}")
    print(f"Pipeline modules: {len(pipeline.modules)}")
    
    print("\n🚀 Testing pipeline processing...")
    test_audio = torch.randn(2, 1, 1000)  # Batch of 2, mono audio, 1000 samples
    
    # Validate input
    valid, errors = audio_processor.validate_input(test_audio, 'process')
    print(f"Input validation: {'✅ PASSED' if valid else '❌ FAILED'}")
    if errors:
        print(f"Errors: {errors}")
    
    # Process through pipeline
    result = pipeline.process(test_audio)
    print(f"Processing result: success={result.success}")
    if result.success:
        print(f"Output keys: {list(result.data.keys())}")
        print(f"Encoded shape: {result.data['encoded'].shape}")
    
    print("\n🔄 Testing async processing...")
    async def test_async():
        async_result = await pipeline.aprocess(test_audio)
        print(f"Async processing: success={async_result.success}")
        return async_result
    
    # Run async test
    import asyncio
    async_result = asyncio.run(test_async())
    
    print("\n📊 Getting composition interface...")
    interface = pipeline.get_composition_interface()
    print(f"Interface: {interface}")
    
    print("\n✅ Standardized Signature Framework ready for deployment!")