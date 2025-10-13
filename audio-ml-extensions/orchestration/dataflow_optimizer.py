import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable, Union, Tuple, Set
import numpy as np
from collections import defaultdict, deque
import warnings
import gc


class MemoryProfile:
    """Track memory usage patterns."""
    def __init__(self):
        self.peak_usage = 0
        self.current_usage = 0
        self.allocations = defaultdict(int)
        self.deallocations = defaultdict(int)
        
    def record_allocation(self, tensor_id: str, size: int):
        self.allocations[tensor_id] = size
        self.current_usage += size
        self.peak_usage = max(self.peak_usage, self.current_usage)
        
    def record_deallocation(self, tensor_id: str):
        if tensor_id in self.allocations:
            size = self.allocations[tensor_id]
            self.deallocations[tensor_id] = size
            self.current_usage -= size
            

class BatchConfig:
    """Configuration for dynamic batching."""
    def __init__(self, 
                 min_batch_size: int = 1,
                 max_batch_size: int = 128,
                 timeout_ms: float = 10.0,
                 adaptive: bool = True):
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.adaptive = adaptive
        self.current_batch_size = min_batch_size
        
        # Adaptive batching stats
        self.latency_history = deque(maxlen=100)
        self.throughput_history = deque(maxlen=100)
        

class DataflowOptimizer(nn.Module):
    def __init__(self,
                 batch_optimization: bool = True,
                 memory_optimization: bool = True,
                 gradient_checkpointing: bool = False,
                 mixed_precision: bool = False,
                 prefetch_factor: int = 2,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.batch_optimization = batch_optimization
        self.memory_optimization = memory_optimization
        self.gradient_checkpointing = gradient_checkpointing
        self.mixed_precision = mixed_precision
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.device = device
        
        # Optimization state
        self.memory_profile = MemoryProfile()
        self.batch_configs = {}
        self.tensor_cache = {}
        self.recompute_map = {}
        
        # Mixed precision
        if self.mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
            
    def optimize_batch_size(self, 
                          module: nn.Module,
                          input_shape: Tuple[int, ...],
                          target_memory_usage: float = 0.9) -> int:
        """Find optimal batch size for given module and input."""
        if not self.batch_optimization:
            return 32  # Default
            
        # Binary search for optimal batch size
        min_batch = 1
        max_batch = 512
        optimal_batch = 1
        
        device = next(module.parameters()).device
        
        while min_batch <= max_batch:
            batch_size = (min_batch + max_batch) // 2
            
            try:
                # Test forward pass
                test_input = torch.randn(batch_size, *input_shape[1:], device=device)
                
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    
                with torch.no_grad():
                    _ = module(test_input)
                    
                if torch.cuda.is_available():
                    memory_used = torch.cuda.max_memory_allocated() / torch.cuda.get_device_properties(0).total_memory
                    
                    if memory_used < target_memory_usage:
                        optimal_batch = batch_size
                        min_batch = batch_size + 1
                    else:
                        max_batch = batch_size - 1
                else:
                    # CPU: use simple heuristic
                    optimal_batch = batch_size
                    min_batch = batch_size + 1
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    max_batch = batch_size - 1
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise e
                    
        return optimal_batch
        
    def create_batch_config(self, name: str, **kwargs) -> BatchConfig:
        """Create named batch configuration."""
        config = BatchConfig(**kwargs)
        self.batch_configs[name] = config
        return config
        
    def optimize_memory_allocation(self, 
                                 modules: List[nn.Module],
                                 execution_order: List[int]) -> Dict[str, Any]:
        """Optimize memory allocation across modules."""
        if not self.memory_optimization:
            return {}
            
        optimization_plan = {
            'recompute': [],
            'inplace': [],
            'buffer_sharing': {}
        }
        
        # Analyze memory requirements
        module_memory = {}
        for i, module in enumerate(modules):
            params_memory = sum(p.numel() * p.element_size() for p in module.parameters())
            # Estimate activation memory (heuristic)
            activation_memory = params_memory * 2
            module_memory[i] = {
                'params': params_memory,
                'activations': activation_memory,
                'total': params_memory + activation_memory
            }
            
        # Identify recomputation candidates
        if self.gradient_checkpointing:
            total_memory = sum(m['total'] for m in module_memory.values())
            cumulative_memory = 0
            
            for idx in execution_order:
                cumulative_memory += module_memory[idx]['total']
                # Recompute heavy modules in the middle of the graph
                if cumulative_memory > total_memory * 0.3 and cumulative_memory < total_memory * 0.7:
                    if module_memory[idx]['activations'] > module_memory[idx]['params']:
                        optimization_plan['recompute'].append(idx)
                        
        # Identify inplace operations
        for i, module in enumerate(modules):
            # Check if module supports inplace operations
            if hasattr(module, 'inplace'):
                optimization_plan['inplace'].append(i)
                
        return optimization_plan
        
    def optimize_dataflow(self, 
                        data_loader: torch.utils.data.DataLoader) -> torch.utils.data.DataLoader:
        """Optimize data loading pipeline."""
        # Create optimized data loader
        optimized_loader = torch.utils.data.DataLoader(
            data_loader.dataset,
            batch_size=data_loader.batch_size,
            shuffle=isinstance(data_loader.sampler, torch.utils.data.RandomSampler),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else 2,
            persistent_workers=self.num_workers > 0
        )
        
        return optimized_loader
        
    def forward(self, 
                module: nn.Module,
                inputs: torch.Tensor,
                optimize_batch: bool = True,
                optimize_memory: bool = True) -> torch.Tensor:
        """Execute module with dataflow optimizations."""
        
        # Batch size optimization
        if optimize_batch and self.batch_optimization:
            optimal_batch = self.optimize_batch_size(module, inputs.shape)
            if inputs.size(0) > optimal_batch:
                # Process in smaller batches
                return self._forward_batched(module, inputs, optimal_batch)
                
        # Memory optimization
        if optimize_memory and self.memory_optimization:
            return self._forward_memory_optimized(module, inputs)
            
        # Mixed precision
        if self.mixed_precision and inputs.device.type == 'cuda':
            with torch.cuda.amp.autocast():
                return module(inputs)
                
        return module(inputs)
        
    def _forward_batched(self, 
                        module: nn.Module,
                        inputs: torch.Tensor,
                        batch_size: int) -> torch.Tensor:
        """Process inputs in batches."""
        outputs = []
        
        for i in range(0, inputs.size(0), batch_size):
            batch = inputs[i:i + batch_size]
            
            if self.mixed_precision and batch.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    output = module(batch)
            else:
                output = module(batch)
                
            outputs.append(output)
            
        return torch.cat(outputs, dim=0)
        
    def _forward_memory_optimized(self,
                                module: nn.Module,
                                inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass with memory optimizations."""
        # Track memory usage
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        # Check if we should checkpoint this module
        module_id = id(module)
        if self.gradient_checkpointing and module_id in self.recompute_map:
            # Use gradient checkpointing
            return torch.utils.checkpoint.checkpoint(module, inputs)
            
        # Normal forward
        output = module(inputs)
        
        # Record memory usage
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated()
            self.memory_profile.record_allocation(f"module_{module_id}", peak_memory)
            
        return output
        
    def enable_mixed_precision(self):
        """Enable mixed precision training."""
        if torch.cuda.is_available():
            self.mixed_precision = True
            if self.scaler is None:
                self.scaler = torch.cuda.amp.GradScaler()
        else:
            warnings.warn("Mixed precision requires CUDA")
            
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory optimization statistics."""
        stats = {
            'peak_memory': self.memory_profile.peak_usage,
            'current_memory': self.memory_profile.current_usage,
            'num_allocations': len(self.memory_profile.allocations),
            'num_deallocations': len(self.memory_profile.deallocations)
        }
        
        if torch.cuda.is_available():
            stats['gpu_memory_allocated'] = torch.cuda.memory_allocated()
            stats['gpu_memory_reserved'] = torch.cuda.memory_reserved()
            
        return stats
        
    def optimize_pipeline(self, 
                        pipeline: List[nn.Module],
                        sample_input: torch.Tensor) -> Dict[str, Any]:
        """Optimize entire pipeline."""
        optimizations = {
            'batch_sizes': {},
            'memory_plan': {},
            'execution_order': list(range(len(pipeline))),
            'estimated_memory': 0,
            'estimated_latency': 0
        }
        
        # Profile each module
        for i, module in enumerate(pipeline):
            # Find optimal batch size
            if self.batch_optimization:
                optimal_batch = self.optimize_batch_size(module, sample_input.shape)
                optimizations['batch_sizes'][i] = optimal_batch
                
            # Estimate memory usage
            if self.memory_optimization:
                with torch.no_grad():
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                    _ = module(sample_input)
                    if torch.cuda.is_available():
                        peak_memory = torch.cuda.max_memory_allocated()
                        optimizations['estimated_memory'] += peak_memory
                        
        # Memory allocation plan
        if self.memory_optimization:
            optimizations['memory_plan'] = self.optimize_memory_allocation(
                pipeline, optimizations['execution_order']
            )
            
        return optimizations
        
    def create_optimized_module(self, module: nn.Module) -> nn.Module:
        """Wrap module with optimizations."""
        
        class OptimizedModule(nn.Module):
            def __init__(self, base_module, optimizer):
                super().__init__()
                self.base_module = base_module
                self.optimizer = optimizer
                
            def forward(self, *args, **kwargs):
                return self.optimizer.forward(self.base_module, *args, **kwargs)
                
        return OptimizedModule(module, self)