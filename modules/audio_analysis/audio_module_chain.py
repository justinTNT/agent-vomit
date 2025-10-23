"""
Audio Module Chaining and Compatibility Layer

Enables seamless chaining of audio analysis modules with automatic compatibility checking,
feature dimension matching, and standardized interfaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
from enum import Enum

from .audio_config import AudioModuleConfig, check_module_compatibility
from .audio_preprocessing import StandardAudioPreprocessor


class ModuleChainMode(Enum):
    """Different modes for chaining modules."""
    STRICT = "strict"           # Require exact compatibility
    ADAPTIVE = "adaptive"       # Auto-adapt incompatible modules  
    PERMISSIVE = "permissive"   # Allow incompatible modules with warnings


@dataclass
class ChainResult:
    """Result from a module chain execution."""
    output: Dict[str, torch.Tensor]
    intermediate_outputs: List[Dict[str, torch.Tensor]]
    compatibility_issues: List[str]
    execution_time: float
    module_names: List[str]


class FeatureDimensionAdapter(nn.Module):
    """
    Adapts feature dimensions between incompatible modules.
    
    Handles mismatches in:
    - Embedding dimensions
    - Temporal dimensions
    - Frequency dimensions
    - Sample rates
    """
    
    def __init__(
        self,
        input_config: AudioModuleConfig,
        output_config: AudioModuleConfig
    ):
        super().__init__()
        self.input_config = input_config
        self.output_config = output_config
        
        # Build adaptation layers
        self.adapters = nn.ModuleDict()
        
        # Embedding dimension adaptation
        if input_config.embedding_dim != output_config.embedding_dim:
            self.adapters['embedding'] = nn.Linear(
                input_config.embedding_dim,
                output_config.embedding_dim
            )
            
        # Sample rate adaptation
        if input_config.sample_rate != output_config.sample_rate:
            self.sample_rate_ratio = output_config.sample_rate / input_config.sample_rate
            self.adapters['sample_rate'] = True  # Flag for resampling
            
        # Spectral dimension adaptation
        if input_config.n_mels != output_config.n_mels:
            self.adapters['mel_dims'] = nn.Linear(
                input_config.n_mels,
                output_config.n_mels
            )
            
        # Temporal dimension adaptation
        if input_config.hop_length != output_config.hop_length:
            self.temporal_scale_factor = output_config.hop_length / input_config.hop_length
            self.adapters['temporal'] = True  # Flag for temporal resampling
            
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Adapt features to match target configuration."""
        adapted_features = {}
        
        for key, value in features.items():
            adapted_value = value
            
            # Adapt different feature types
            if 'embedding' in self.adapters and 'embedding' in key:
                adapted_value = self.adapters['embedding'](adapted_value)
                
            elif 'mel_dims' in self.adapters and 'mel' in key:
                # Adapt mel spectrogram frequency dimension
                if adapted_value.dim() == 3:  # [batch, freq, time]
                    adapted_value = adapted_value.transpose(1, 2)  # [batch, time, freq]
                    adapted_value = self.adapters['mel_dims'](adapted_value)
                    adapted_value = adapted_value.transpose(1, 2)  # [batch, freq, time]
                    
            elif 'temporal' in self.adapters and adapted_value.dim() >= 2:
                # Adapt temporal dimension using interpolation
                if adapted_value.dim() == 3:  # [batch, feat, time]
                    target_length = int(adapted_value.shape[-1] * self.temporal_scale_factor)
                    adapted_value = F.interpolate(
                        adapted_value,
                        size=target_length,
                        mode='linear',
                        align_corners=False
                    )
                    
            adapted_features[key] = adapted_value
            
        return adapted_features
        
    def is_adaptation_needed(self) -> bool:
        """Check if any adaptation is needed."""
        return len(self.adapters) > 0


class AudioModuleChain(nn.Module):
    """
    Chain multiple audio analysis modules with automatic compatibility handling.
    
    Supports:
    - Automatic compatibility checking
    - Feature dimension adaptation
    - Intermediate result caching
    - Error handling and recovery
    - Performance monitoring
    """
    
    def __init__(
        self,
        modules: List[nn.Module],
        mode: ModuleChainMode = ModuleChainMode.ADAPTIVE,
        cache_intermediates: bool = True,
        validate_outputs: bool = True
    ):
        super().__init__()
        
        self.mode = mode
        self.cache_intermediates = cache_intermediates
        self.validate_outputs = validate_outputs
        
        # Store modules and their configs
        self.module_configs = []
        self.module_names = []
        
        for i, module in enumerate(modules):
            if hasattr(module, 'config'):
                self.module_configs.append(module.config)
            else:
                # Create default config for modules without explicit config
                self.module_configs.append(AudioModuleConfig())
                warnings.warn(f"Module {i} ({type(module).__name__}) has no config, using default")
                
            self.module_names.append(type(module).__name__)
            
        # Register modules
        self.modules = nn.ModuleList(modules)
        
        # Build adaptation layers
        self.adapters = nn.ModuleList()
        self.compatibility_issues = []
        
        for i in range(len(modules) - 1):
            current_config = self.module_configs[i]
            next_config = self.module_configs[i + 1]
            
            if not current_config.is_compatible(next_config):
                issue = f"Incompatible modules: {self.module_names[i]} -> {self.module_names[i+1]}"
                self.compatibility_issues.append(issue)
                
                if self.mode == ModuleChainMode.STRICT:
                    raise ValueError(f"Strict mode: {issue}")
                elif self.mode == ModuleChainMode.ADAPTIVE:
                    adapter = FeatureDimensionAdapter(current_config, next_config)
                    self.adapters.append(adapter)
                else:  # PERMISSIVE
                    warnings.warn(f"Permissive mode: {issue}")
                    self.adapters.append(nn.Identity())  # No adaptation
            else:
                self.adapters.append(nn.Identity())  # No adaptation needed
                
    def forward(
        self,
        waveform: torch.Tensor,
        return_intermediates: bool = None,
        module_subset: Optional[List[int]] = None
    ) -> Union[Dict[str, torch.Tensor], ChainResult]:
        """
        Execute the module chain.
        
        Args:
            waveform: Input audio [batch, samples]
            return_intermediates: Whether to return intermediate results
            module_subset: Indices of specific modules to execute
            
        Returns:
            Final output or ChainResult with detailed information
        """
        if return_intermediates is None:
            return_intermediates = self.cache_intermediates
            
        if module_subset is None:
            module_subset = list(range(len(self.modules)))
            
        import time
        start_time = time.time()
        
        # Initialize
        current_output = None
        intermediate_outputs = []
        execution_issues = []
        
        # Execute modules in sequence
        for i, module_idx in enumerate(module_subset):
            try:
                module = self.modules[module_idx]
                
                if i == 0:
                    # First module processes raw waveform
                    if hasattr(module, 'forward') and 'waveform' in module.forward.__code__.co_varnames:
                        current_output = module(waveform)
                    else:
                        # Try standard call
                        current_output = module(waveform)
                else:
                    # Subsequent modules process previous output
                    if module_idx < len(self.adapters):
                        adapter = self.adapters[module_idx - 1]
                        if adapter.is_adaptation_needed() if hasattr(adapter, 'is_adaptation_needed') else False:
                            current_output = adapter(current_output)
                            
                    # Call next module
                    if isinstance(current_output, dict):
                        # Try to extract main feature
                        main_feature = self._extract_main_feature(current_output)
                        current_output = module(main_feature)
                    else:
                        current_output = module(current_output)
                        
                # Validate output
                if self.validate_outputs:
                    self._validate_module_output(current_output, module_idx)
                    
                # Cache intermediate result
                if return_intermediates:
                    intermediate_outputs.append({
                        'module_idx': module_idx,
                        'module_name': self.module_names[module_idx],
                        'output': current_output.copy() if isinstance(current_output, dict) else current_output.clone()
                    })
                    
            except Exception as e:
                error_msg = f"Error in module {module_idx} ({self.module_names[module_idx]}): {str(e)}"
                execution_issues.append(error_msg)
                
                if self.mode == ModuleChainMode.STRICT:
                    raise RuntimeError(error_msg)
                else:
                    warnings.warn(error_msg)
                    # Continue with previous output
                    
        execution_time = time.time() - start_time
        
        # Return results
        if return_intermediates:
            return ChainResult(
                output=current_output,
                intermediate_outputs=intermediate_outputs,
                compatibility_issues=self.compatibility_issues + execution_issues,
                execution_time=execution_time,
                module_names=[self.module_names[i] for i in module_subset]
            )
        else:
            return current_output
            
    def _extract_main_feature(self, output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract the main feature tensor from module output."""
        # Priority order for main features
        priority_keys = [
            'features', 'embeddings', 'output', 'logits', 
            'mel_spectrogram', 'waveform', 'combined_features'
        ]
        
        for key in priority_keys:
            if key in output:
                return output[key]
                
        # If no priority key found, use first tensor
        for value in output.values():
            if isinstance(value, torch.Tensor):
                return value
                
        raise ValueError("No tensor found in module output")
        
    def _validate_module_output(self, output: Any, module_idx: int):
        """Validate module output format."""
        if output is None:
            raise ValueError(f"Module {module_idx} returned None")
            
        if isinstance(output, dict):
            if len(output) == 0:
                warnings.warn(f"Module {module_idx} returned empty dict")
        elif isinstance(output, torch.Tensor):
            if output.numel() == 0:
                warnings.warn(f"Module {module_idx} returned empty tensor")
        else:
            warnings.warn(f"Module {module_idx} returned unexpected type: {type(output)}")
            
    def get_compatibility_report(self) -> Dict[str, Any]:
        """Get detailed compatibility report."""
        return {
            'compatible': len(self.compatibility_issues) == 0,
            'issues': self.compatibility_issues,
            'num_modules': len(self.modules),
            'module_names': self.module_names,
            'mode': self.mode.value,
            'adapters_needed': sum(1 for adapter in self.adapters 
                                 if hasattr(adapter, 'is_adaptation_needed') and adapter.is_adaptation_needed())
        }
        
    def execute_subset(self, module_indices: List[int], waveform: torch.Tensor) -> ChainResult:
        """Execute only a subset of modules."""
        return self.forward(waveform, return_intermediates=True, module_subset=module_indices)
        
    def benchmark_modules(self, waveform: torch.Tensor, num_runs: int = 5) -> Dict[str, float]:
        """Benchmark individual module execution times."""
        import time
        
        benchmarks = {}
        
        for i, module in enumerate(self.modules):
            times = []
            
            for _ in range(num_runs):
                if i == 0:
                    input_data = waveform
                else:
                    # Use dummy input for subsequent modules
                    input_data = torch.randn(waveform.shape[0], 512)  # Dummy embedding
                    
                start_time = time.time()
                try:
                    with torch.no_grad():
                        _ = module(input_data)
                    end_time = time.time()
                    times.append(end_time - start_time)
                except:
                    times.append(float('inf'))  # Mark failed runs
                    
            benchmarks[self.module_names[i]] = {
                'mean_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times)
            }
            
        return benchmarks


class AudioPipeline(nn.Module):
    """
    High-level audio processing pipeline with predefined module combinations.
    
    Provides common workflows for audio analysis tasks.
    """
    
    def __init__(self, config: AudioModuleConfig):
        super().__init__()
        self.config = config
        
    @classmethod
    def create_classification_pipeline(
        cls,
        classifier_type: str = "ast",
        config: Optional[AudioModuleConfig] = None
    ) -> AudioModuleChain:
        """Create pipeline for audio classification."""
        if config is None:
            from .audio_config import get_general_config
            config = get_general_config()
            
        modules = []
        
        # Preprocessing
        preprocessor = StandardAudioPreprocessor(config)
        modules.append(preprocessor)
        
        # Classifier
        if classifier_type == "ast":
            from .ast_reference_compatible import ast_base384
            classifier = ast_base384(label_dim=50)
        elif classifier_type == "instrument":
            from .instrument_classifier import create_basic_instrument_classifier
            classifier = create_basic_instrument_classifier()
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
            
        modules.append(classifier)
        
        return AudioModuleChain(modules, mode=ModuleChainMode.ADAPTIVE)
        
    @classmethod  
    def create_feature_extraction_pipeline(
        cls,
        feature_types: List[str] = ['spectral', 'harmonic', 'timbral'],
        config: Optional[AudioModuleConfig] = None
    ) -> AudioModuleChain:
        """Create pipeline for comprehensive feature extraction."""
        if config is None:
            from .audio_config import get_music_config
            config = get_music_config()
            
        modules = []
        
        # Preprocessing
        preprocessor = StandardAudioPreprocessor(config)
        modules.append(preprocessor)
        
        # Feature extractors
        if 'musical' in feature_types:
            from .advanced_musical_features import AdvancedMusicalFeatures
            musical_analyzer = AdvancedMusicalFeatures(sample_rate=config.sample_rate)
            modules.append(musical_analyzer)
            
        return AudioModuleChain(modules, mode=ModuleChainMode.ADAPTIVE)
        
    @classmethod
    def create_similarity_pipeline(
        cls,
        similarity_metrics: List[str] = ['spectral', 'harmonic'],
        config: Optional[AudioModuleConfig] = None
    ) -> AudioModuleChain:
        """Create pipeline for audio similarity analysis."""
        if config is None:
            from .audio_config import get_music_config
            config = get_music_config()
            
        modules = []
        
        # Preprocessing
        preprocessor = StandardAudioPreprocessor(config)
        modules.append(preprocessor)
        
        # Similarity matcher
        from .audio_similarity_matcher import AudioSimilarityMatcher
        similarity_matcher = AudioSimilarityMatcher(
            sample_rate=config.sample_rate,
            similarity_metrics=similarity_metrics
        )
        modules.append(similarity_matcher)
        
        return AudioModuleChain(modules, mode=ModuleChainMode.ADAPTIVE)
        
    @classmethod
    def create_timbralgebraics_pipeline(cls) -> AudioModuleChain:
        """Create pipeline optimized for timbralgebraics project."""
        from .audio_config import get_timbralgebraics_config
        from .audio_preprocessing import TimbralgebraicsPreprocessor
        from .advanced_musical_features import AdvancedMusicalFeatures
        from .audio_similarity_matcher import AudioSimilarityMatcher
        from .perceptual_quality_assessor import create_music_quality_assessor
        
        config = get_timbralgebraics_config()
        
        modules = [
            TimbralgebraicsPreprocessor(),
            AdvancedMusicalFeatures(sample_rate=config.sample_rate),
            AudioSimilarityMatcher(
                sample_rate=config.sample_rate,
                similarity_metrics=['spectral', 'harmonic', 'learned']
            ),
            create_music_quality_assessor()
        ]
        
        return AudioModuleChain(modules, mode=ModuleChainMode.ADAPTIVE)


# Utility functions
def chain_modules(
    modules: List[nn.Module],
    mode: ModuleChainMode = ModuleChainMode.ADAPTIVE
) -> AudioModuleChain:
    """Convenience function to create module chain."""
    return AudioModuleChain(modules, mode=mode)


def create_adaptive_chain(modules: List[nn.Module]) -> AudioModuleChain:
    """Create chain with automatic adaptation."""
    return AudioModuleChain(modules, mode=ModuleChainMode.ADAPTIVE)


def create_strict_chain(modules: List[nn.Module]) -> AudioModuleChain:
    """Create chain requiring exact compatibility."""
    return AudioModuleChain(modules, mode=ModuleChainMode.STRICT)


# Example usage
if __name__ == "__main__":
    # Test module chaining
    from .audio_config import get_music_config
    
    # Create a sample pipeline
    pipeline = AudioPipeline.create_classification_pipeline("instrument")
    
    # Test with dummy audio
    sample_rate = 22050
    duration = 3
    waveform = torch.randn(2, sample_rate * duration)
    
    # Execute pipeline
    result = pipeline(waveform, return_intermediates=True)
    
    print(f"Pipeline executed in {result.execution_time:.3f}s")
    print(f"Compatibility issues: {len(result.compatibility_issues)}")
    print(f"Intermediate outputs: {len(result.intermediate_outputs)}")
    
    # Get compatibility report
    report = pipeline.get_compatibility_report()
    print(f"Pipeline compatibility: {report}")
    
    # Benchmark modules
    benchmarks = pipeline.benchmark_modules(waveform[:1])  # Single sample for speed
    print("Module benchmarks:")
    for name, times in benchmarks.items():
        print(f"  {name}: {times['mean_time']*1000:.1f}ms")