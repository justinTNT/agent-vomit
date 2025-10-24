"""
Bulletproof Audio Module Chain with comprehensive error handling and compatibility management.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager
import time
import gc

logger = logging.getLogger(__name__)

class ModuleChainMode(Enum):
    """Different modes for chaining modules."""
    STRICT = "strict"
    ADAPTIVE = "adaptive"
    PERMISSIVE = "permissive"

@dataclass
class ChainResult:
    """Result from a module chain execution."""
    output: Dict[str, torch.Tensor]
    intermediate_outputs: List[Dict[str, torch.Tensor]]
    compatibility_issues: List[str]
    execution_time: float
    module_names: List[str]
    success: bool = True
    error_message: Optional[str] = None

class BulletproofAudioModuleChain(nn.Module):
    """
    Bulletproof chain for audio analysis modules with:
    - Comprehensive compatibility checking
    - Automatic feature dimension adaptation
    - Memory-efficient processing
    - Error recovery and fallback strategies
    - Performance monitoring and optimization
    """
    
    def __init__(
        self,
        modules: List[nn.Module],
        mode: ModuleChainMode = ModuleChainMode.ADAPTIVE,
        cache_intermediates: bool = True,
        validate_outputs: bool = True,
        memory_efficient: bool = True,
        max_memory_mb: Optional[int] = None,
        enable_fallbacks: bool = True,
        validate_inputs: bool = True
    ):
        super().__init__()
        
        if not modules:
            raise ValueError("At least one module must be provided")
        
        self.mode = mode
        self.cache_intermediates = cache_intermediates
        self.validate_outputs = validate_outputs
        self.memory_efficient = memory_efficient
        self.max_memory_mb = max_memory_mb
        self.enable_fallbacks = enable_fallbacks
        self.validate_inputs = validate_inputs
        
        # Store modules and their metadata
        self.module_configs = []
        self.module_names = []
        self.module_types = []
        
        try:
            self._analyze_modules(modules)
            self.modules = nn.ModuleList(modules)
            
            # Build adaptation layers
            self.adapters = self._build_adaptation_layers()
            self.compatibility_issues = self._check_compatibility()
            
            logger.info(f"BulletproofAudioModuleChain initialized with {len(modules)} modules")
            
        except Exception as e:
            logger.error(f"Error initializing module chain: {e}")
            if not self.enable_fallbacks:
                raise
            self._initialize_fallback_chain(modules)
    
    def _analyze_modules(self, modules: List[nn.Module]):
        """Analyze modules and extract metadata."""
        for i, module in enumerate(modules):
            # Extract configuration if available
            if hasattr(module, 'config'):
                self.module_configs.append(module.config)
            elif hasattr(module, 'get_config'):
                try:
                    self.module_configs.append(module.get_config())
                except Exception:
                    self.module_configs.append(None)
            else:
                self.module_configs.append(None)
            
            # Store module metadata
            self.module_names.append(type(module).__name__)
            self.module_types.append(self._classify_module_type(module))
            
            if self.module_configs[-1] is None:
                logger.warning(f"Module {i} ({self.module_names[-1]}) has no config")
    
    def _classify_module_type(self, module: nn.Module) -> str:
        """Classify module type for better handling."""
        module_name = type(module).__name__.lower()
        
        if 'preprocess' in module_name:
            return 'preprocessor'
        elif 'feature' in module_name or 'extract' in module_name:
            return 'feature_extractor'
        elif 'classifier' in module_name or 'detect' in module_name:
            return 'classifier'
        elif 'transform' in module_name or 'ast' in module_name:
            return 'transformer'
        elif 'similar' in module_name or 'match' in module_name:
            return 'similarity_matcher'
        elif 'quality' in module_name or 'assess' in module_name:
            return 'quality_assessor'
        else:
            return 'unknown'
    
    def _build_adaptation_layers(self) -> nn.ModuleList:
        """Build adaptation layers between incompatible modules."""
        adapters = nn.ModuleList()
        
        for i in range(len(self.modules) - 1):
            current_config = self.module_configs[i]
            next_config = self.module_configs[i + 1]
            
            if current_config and next_config:
                if hasattr(current_config, 'is_compatible') and hasattr(next_config, 'is_compatible'):
                    if not current_config.is_compatible(next_config):
                        adapter = self._create_adapter(current_config, next_config)
                        adapters.append(adapter)
                    else:
                        adapters.append(nn.Identity())
                else:
                    # No compatibility check available
                    adapters.append(nn.Identity())
            else:
                # No configs available
                adapters.append(nn.Identity())
        
        return adapters
    
    def _create_adapter(self, input_config, output_config) -> nn.Module:
        """Create adapter between incompatible configurations."""
        try:
            return FeatureDimensionAdapter(input_config, output_config)
        except Exception as e:
            logger.warning(f"Failed to create adapter: {e}")
            return nn.Identity()
    
    def _check_compatibility(self) -> List[str]:
        """Check compatibility between modules."""
        issues = []
        
        for i in range(len(self.modules) - 1):
            current_name = self.module_names[i]
            next_name = self.module_names[i + 1]
            current_config = self.module_configs[i]
            next_config = self.module_configs[i + 1]
            
            if current_config and next_config:
                if hasattr(current_config, 'is_compatible'):
                    if not current_config.is_compatible(next_config):
                        issue = f"Incompatible: {current_name} -> {next_name}"
                        issues.append(issue)
                        
                        if self.mode == ModuleChainMode.STRICT:
                            raise ValueError(f"Strict mode: {issue}")
        
        return issues
    
    def _initialize_fallback_chain(self, modules: List[nn.Module]):
        """Initialize fallback chain when main initialization fails."""
        logger.warning("Initializing fallback module chain")
        
        # Store modules without advanced features
        self.modules = nn.ModuleList(modules)
        self.adapters = nn.ModuleList([nn.Identity() for _ in range(len(modules) - 1)])
        self.compatibility_issues = []
        
        # Simplified metadata
        for i, module in enumerate(modules):
            self.module_names.append(f"Module_{i}")
            self.module_types.append('unknown')
            self.module_configs.append(None)
    
    @contextmanager
    def _memory_management(self):
        """Context manager for memory management."""
        if self.memory_efficient:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        try:
            yield
        finally:
            if self.memory_efficient:
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def _validate_input(self, input_data: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Validate chain input."""
        if isinstance(input_data, torch.Tensor):
            # Check for NaN/Inf
            if torch.isnan(input_data).any():
                logger.warning("NaN values in input, replacing with zeros")
                input_data = torch.nan_to_num(input_data, nan=0.0)
            
            if torch.isinf(input_data).any():
                logger.warning("Inf values in input, clipping")
                input_data = torch.clamp(input_data, -100.0, 100.0)
        
        elif isinstance(input_data, dict):
            # Validate dictionary inputs
            for key, value in input_data.items():
                if isinstance(value, torch.Tensor):
                    if torch.isnan(value).any():
                        logger.warning(f"NaN values in {key}, replacing with zeros")
                        input_data[key] = torch.nan_to_num(value, nan=0.0)
                    
                    if torch.isinf(value).any():
                        logger.warning(f"Inf values in {key}, clipping")
                        input_data[key] = torch.clamp(value, -100.0, 100.0)
        
        return input_data
    
    def _estimate_memory_usage(self, input_data: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> float:
        """Estimate memory usage for processing."""
        if isinstance(input_data, torch.Tensor):
            base_size = input_data.numel() * 4  # 4 bytes per float
        elif isinstance(input_data, dict):
            base_size = sum(v.numel() * 4 for v in input_data.values() if isinstance(v, torch.Tensor))
        else:
            base_size = 1024 * 1024  # 1MB default
        
        # Estimate processing overhead (5x the input size)
        total_bytes = base_size * 5
        
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    def forward(
        self,
        input_data: Union[torch.Tensor, Dict[str, torch.Tensor]],
        return_intermediates: bool = None,
        module_subset: Optional[List[int]] = None,
        max_memory_mb: Optional[int] = None
    ) -> Union[Dict[str, torch.Tensor], ChainResult]:
        """
        Execute the module chain with comprehensive error handling.
        """
        if return_intermediates is None:
            return_intermediates = self.cache_intermediates
        
        if module_subset is None:
            module_subset = list(range(len(self.modules)))
        
        max_memory = max_memory_mb or self.max_memory_mb
        
        start_time = time.time()
        
        with self._memory_management():
            try:
                # Input validation
                if self.validate_inputs:
                    input_data = self._validate_input(input_data)
                
                # Memory check
                if max_memory and self._estimate_memory_usage(input_data) > max_memory:
                    logger.warning(f"Estimated memory usage exceeds {max_memory}MB")
                    if self.enable_fallbacks:
                        return self._process_memory_efficient(input_data, module_subset, return_intermediates)
                
                # Execute chain
                return self._execute_chain(input_data, module_subset, return_intermediates, start_time)
                
            except Exception as e:
                logger.error(f"Critical error in module chain: {e}")
                if self.enable_fallbacks:
                    return self._emergency_fallback(input_data, module_subset, return_intermediates, start_time, str(e))
                raise
    
    def _execute_chain(
        self,
        input_data: Union[torch.Tensor, Dict[str, torch.Tensor]],
        module_subset: List[int],
        return_intermediates: bool,
        start_time: float
    ) -> ChainResult:
        """Execute the module chain."""
        current_output = input_data
        intermediate_outputs = []
        execution_issues = []
        
        for i, module_idx in enumerate(module_subset):
            try:
                module = self.modules[module_idx]
                module_name = self.module_names[module_idx]
                
                # Apply adapter if needed
                if i > 0 and module_idx - 1 < len(self.adapters):
                    adapter = self.adapters[module_idx - 1]
                    if not isinstance(adapter, nn.Identity):
                        try:
                            current_output = adapter(current_output)
                        except Exception as e:
                            logger.warning(f"Adapter failed for {module_name}: {e}")
                
                # Process through module
                processed_output = self._process_module(module, current_output, module_name)
                
                # Validate output
                if self.validate_outputs:
                    self._validate_module_output(processed_output, module_idx, module_name)
                
                current_output = processed_output
                
                # Cache intermediate result
                if return_intermediates:
                    intermediate_outputs.append({
                        'module_idx': module_idx,
                        'module_name': module_name,
                        'output': self._safe_copy_output(current_output)
                    })
                
            except Exception as e:
                error_msg = f"Error in module {module_idx} ({self.module_names[module_idx]}): {str(e)}"
                execution_issues.append(error_msg)
                logger.error(error_msg)
                
                if self.mode == ModuleChainMode.STRICT:
                    raise RuntimeError(error_msg)
                elif self.enable_fallbacks:
                    # Try to continue with previous output or create dummy output
                    current_output = self._create_fallback_output(current_output, module_idx)
                else:
                    raise
        
        execution_time = time.time() - start_time
        
        return ChainResult(
            output=current_output,
            intermediate_outputs=intermediate_outputs,
            compatibility_issues=self.compatibility_issues + execution_issues,
            execution_time=execution_time,
            module_names=[self.module_names[i] for i in module_subset],
            success=len(execution_issues) == 0,
            error_message="; ".join(execution_issues) if execution_issues else None
        )
    
    def _process_module(self, module: nn.Module, input_data: Any, module_name: str) -> Any:
        """Process data through a single module with error handling."""
        try:
            # Determine how to call the module based on its type
            if hasattr(module, 'forward'):
                # Check if module expects specific input format
                if isinstance(input_data, dict):
                    # Try to extract the main feature for modules that expect tensors
                    main_feature = self._extract_main_feature(input_data)
                    
                    # Try calling with dictionary first, fallback to main feature
                    try:
                        return module(input_data)
                    except (TypeError, RuntimeError):
                        try:
                            return module(main_feature)
                        except Exception:
                            # Try with waveform parameter if it's an audio module
                            if 'waveform' in str(module.forward.__code__.co_varnames):
                                return module(waveform=main_feature)
                            else:
                                raise
                else:
                    # Direct tensor input
                    return module(input_data)
            else:
                # Callable but not a standard module
                return module(input_data)
                
        except Exception as e:
            logger.error(f"Module {module_name} processing failed: {e}")
            raise
    
    def _extract_main_feature(self, output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract the main feature tensor from module output."""
        # Priority order for main features
        priority_keys = [
            'waveform', 'features', 'embeddings', 'output', 'logits',
            'mel_spectrogram', 'combined_features', 'cls_features'
        ]
        
        for key in priority_keys:
            if key in output and isinstance(output[key], torch.Tensor):
                return output[key]
        
        # If no priority key found, use first tensor
        for value in output.values():
            if isinstance(value, torch.Tensor):
                return value
        
        raise ValueError("No tensor found in module output")
    
    def _validate_module_output(self, output: Any, module_idx: int, module_name: str):
        """Validate module output."""
        if output is None:
            raise ValueError(f"Module {module_name} returned None")
        
        if isinstance(output, dict):
            if len(output) == 0:
                logger.warning(f"Module {module_name} returned empty dict")
            # Check for tensors with invalid values
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    if torch.isnan(value).any():
                        logger.warning(f"Module {module_name} output '{key}' contains NaN")
                    if torch.isinf(value).any():
                        logger.warning(f"Module {module_name} output '{key}' contains Inf")
        
        elif isinstance(output, torch.Tensor):
            if output.numel() == 0:
                logger.warning(f"Module {module_name} returned empty tensor")
            if torch.isnan(output).any():
                logger.warning(f"Module {module_name} output contains NaN")
            if torch.isinf(output).any():
                logger.warning(f"Module {module_name} output contains Inf")
    
    def _safe_copy_output(self, output: Any) -> Any:
        """Safely copy output for caching."""
        try:
            if isinstance(output, dict):
                return {k: v.detach().clone() if isinstance(v, torch.Tensor) else v for k, v in output.items()}
            elif isinstance(output, torch.Tensor):
                return output.detach().clone()
            else:
                return output
        except Exception as e:
            logger.warning(f"Failed to copy output: {e}")
            return output
    
    def _create_fallback_output(self, current_output: Any, failed_module_idx: int) -> Any:
        """Create fallback output when a module fails."""
        logger.warning(f"Creating fallback output for failed module {failed_module_idx}")
        
        if isinstance(current_output, dict):
            # Return current output as-is
            return current_output
        elif isinstance(current_output, torch.Tensor):
            # Wrap tensor in dictionary
            return {'features': current_output, '_fallback': True}
        else:
            # Create minimal fallback
            return {'_fallback': True, '_error': f'Module {failed_module_idx} failed'}
    
    def _process_memory_efficient(
        self,
        input_data: Union[torch.Tensor, Dict[str, torch.Tensor]],
        module_subset: List[int],
        return_intermediates: bool
    ) -> ChainResult:
        """Process with memory efficiency optimizations."""
        logger.info("Using memory-efficient processing")
        
        # Disable intermediate caching to save memory
        original_cache = self.cache_intermediates
        self.cache_intermediates = False
        
        try:
            # Process with reduced batch size if possible
            if isinstance(input_data, torch.Tensor) and input_data.shape[0] > 1:
                return self._process_batched(input_data, module_subset, return_intermediates)
            else:
                return self._execute_chain(input_data, module_subset, False, time.time())
        finally:
            self.cache_intermediates = original_cache
    
    def _process_batched(
        self,
        input_data: torch.Tensor,
        module_subset: List[int],
        return_intermediates: bool
    ) -> ChainResult:
        """Process large batches by splitting them."""
        batch_size = input_data.shape[0]
        max_batch_size = max(1, batch_size // 2)  # Process half at a time
        
        results = []
        
        for start_idx in range(0, batch_size, max_batch_size):
            end_idx = min(start_idx + max_batch_size, batch_size)
            batch_data = input_data[start_idx:end_idx]
            
            try:
                result = self._execute_chain(batch_data, module_subset, False, time.time())
                results.append(result)
            except Exception as e:
                logger.error(f"Batch {start_idx}-{end_idx} failed: {e}")
                continue
        
        if not results:
            raise RuntimeError("All batches failed")
        
        # Merge results
        return self._merge_batch_results(results, return_intermediates)
    
    def _merge_batch_results(self, results: List[ChainResult], return_intermediates: bool) -> ChainResult:
        """Merge results from batch processing."""
        if len(results) == 1:
            return results[0]
        
        # Merge outputs
        merged_output = {}
        for key in results[0].output.keys():
            values = [r.output[key] for r in results if key in r.output]
            if values and isinstance(values[0], torch.Tensor):
                merged_output[key] = torch.cat(values, dim=0)
            else:
                merged_output[key] = values[0]  # Take first value for non-tensors
        
        # Merge metadata
        total_time = sum(r.execution_time for r in results)
        all_issues = []
        for r in results:
            all_issues.extend(r.compatibility_issues)
        
        return ChainResult(
            output=merged_output,
            intermediate_outputs=[] if not return_intermediates else results[0].intermediate_outputs,
            compatibility_issues=all_issues,
            execution_time=total_time,
            module_names=results[0].module_names,
            success=all(r.success for r in results)
        )
    
    def _emergency_fallback(
        self,
        input_data: Union[torch.Tensor, Dict[str, torch.Tensor]],
        module_subset: List[int],
        return_intermediates: bool,
        start_time: float,
        error_message: str
    ) -> ChainResult:
        """Emergency fallback when everything fails."""
        logger.warning("Using emergency fallback for module chain")
        
        # Create minimal output based on input
        if isinstance(input_data, torch.Tensor):
            emergency_output = {
                'features': input_data,
                '_emergency_fallback': True,
                '_original_error': error_message
            }
        elif isinstance(input_data, dict):
            emergency_output = input_data.copy()
            emergency_output['_emergency_fallback'] = True
            emergency_output['_original_error'] = error_message
        else:
            emergency_output = {
                '_emergency_fallback': True,
                '_original_error': error_message
            }
        
        return ChainResult(
            output=emergency_output,
            intermediate_outputs=[],
            compatibility_issues=[f"Emergency fallback: {error_message}"],
            execution_time=time.time() - start_time,
            module_names=[self.module_names[i] for i in module_subset],
            success=False,
            error_message=error_message
        )
    
    def get_compatibility_report(self) -> Dict[str, Any]:
        """Get detailed compatibility report."""
        return {
            'compatible': len(self.compatibility_issues) == 0,
            'issues': self.compatibility_issues,
            'num_modules': len(self.modules),
            'module_names': self.module_names,
            'module_types': self.module_types,
            'mode': self.mode.value,
            'adapters_needed': sum(1 for adapter in self.adapters if not isinstance(adapter, nn.Identity)),
            'memory_efficient': self.memory_efficient,
            'max_memory_mb': self.max_memory_mb
        }


class FeatureDimensionAdapter(nn.Module):
    """Adapter for feature dimension mismatches."""
    
    def __init__(self, input_config, output_config):
        super().__init__()
        self.input_config = input_config
        self.output_config = output_config
        
        # Build adaptation layers based on config differences
        self.adapters = nn.ModuleDict()
        
        try:
            # Embedding dimension adaptation
            if hasattr(input_config, 'embedding_dim') and hasattr(output_config, 'embedding_dim'):
                if input_config.embedding_dim != output_config.embedding_dim:
                    self.adapters['embedding'] = nn.Linear(
                        input_config.embedding_dim,
                        output_config.embedding_dim
                    )
            
            # Sample rate adaptation (placeholder - would need actual resampling)
            if hasattr(input_config, 'sample_rate') and hasattr(output_config, 'sample_rate'):
                if input_config.sample_rate != output_config.sample_rate:
                    self.sample_rate_ratio = output_config.sample_rate / input_config.sample_rate
                    self.adapters['sample_rate'] = True
            
            # Spectral dimension adaptation
            if hasattr(input_config, 'n_mels') and hasattr(output_config, 'n_mels'):
                if input_config.n_mels != output_config.n_mels:
                    self.adapters['mel_dims'] = nn.Linear(
                        input_config.n_mels,
                        output_config.n_mels
                    )
        
        except Exception as e:
            logger.warning(f"Error building adapter: {e}")
    
    def forward(self, features: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Adapt features to match target configuration."""
        try:
            if isinstance(features, dict):
                adapted_features = {}
                for key, value in features.items():
                    adapted_features[key] = self._adapt_tensor(value, key)
                return adapted_features
            else:
                return self._adapt_tensor(features, 'default')
        except Exception as e:
            logger.warning(f"Feature adaptation failed: {e}")
            return features
    
    def _adapt_tensor(self, tensor: torch.Tensor, key: str) -> torch.Tensor:
        """Adapt a single tensor."""
        if not isinstance(tensor, torch.Tensor):
            return tensor
        
        adapted = tensor
        
        try:
            # Apply relevant adaptations
            if 'embedding' in self.adapters and ('embedding' in key or 'feature' in key):
                if adapted.shape[-1] == self.input_config.embedding_dim:
                    adapted = self.adapters['embedding'](adapted)
            
            elif 'mel_dims' in self.adapters and 'mel' in key:
                if adapted.dim() == 3 and adapted.shape[1] == self.input_config.n_mels:
                    # [batch, freq, time] -> [batch, time, freq] -> adapt -> [batch, freq, time]
                    adapted = adapted.transpose(1, 2)
                    adapted = self.adapters['mel_dims'](adapted)
                    adapted = adapted.transpose(1, 2)
        
        except Exception as e:
            logger.warning(f"Tensor adaptation failed for {key}: {e}")
        
        return adapted


def test_bulletproof_module_chain():
    """Test the bulletproof module chain."""
    logger.info("Testing BulletproofAudioModuleChain")
    
    # Create dummy modules for testing
    class DummyPreprocessor(nn.Module):
        def forward(self, waveform):
            return {'mel_spectrogram': torch.randn(waveform.shape[0], 128, 100)}
    
    class DummyFeatureExtractor(nn.Module):
        def forward(self, features):
            if isinstance(features, dict):
                mel = features['mel_spectrogram']
                return {'features': torch.randn(mel.shape[0], 256)}
            return {'features': torch.randn(features.shape[0], 256)}
    
    class DummyClassifier(nn.Module):
        def forward(self, features):
            if isinstance(features, dict):
                feat = features['features']
                return {'logits': torch.randn(feat.shape[0], 10)}
            return {'logits': torch.randn(features.shape[0], 10)}
    
    # Create chain
    modules = [DummyPreprocessor(), DummyFeatureExtractor(), DummyClassifier()]
    chain = BulletproofAudioModuleChain(
        modules,
        mode=ModuleChainMode.ADAPTIVE,
        enable_fallbacks=True,
        memory_efficient=True
    )
    
    # Test with dummy audio
    waveform = torch.randn(2, 16000)
    
    try:
        result = chain(waveform, return_intermediates=True)
        logger.info(f"Chain execution successful: {result.success}")
        logger.info(f"Output keys: {list(result.output.keys())}")
        logger.info(f"Execution time: {result.execution_time:.3f}s")
        logger.info(f"Intermediate outputs: {len(result.intermediate_outputs)}")
        
        # Test compatibility report
        report = chain.get_compatibility_report()
        logger.info(f"Compatibility report: {report}")
        
    except Exception as e:
        logger.error(f"Chain test failed: {e}")
    
    logger.info("Module chain testing completed")


if __name__ == "__main__":
    test_bulletproof_module_chain()
