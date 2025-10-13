import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
import time
import numpy as np
from collections import defaultdict
import warnings
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
import json


class LayerProfile:
    """Profile information for a single layer."""
    def __init__(self, name: str, module_type: str):
        self.name = name
        self.module_type = module_type
        self.forward_time = []
        self.backward_time = []
        self.memory_usage = []
        self.input_shapes = []
        self.output_shapes = []
        self.parameter_count = 0
        self.flops = 0
        
    def add_forward_time(self, time_ms: float):
        self.forward_time.append(time_ms)
        
    def add_backward_time(self, time_ms: float):
        self.backward_time.append(time_ms)
        
    def add_memory_usage(self, memory_bytes: int):
        self.memory_usage.append(memory_bytes)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics."""
        return {
            'name': self.name,
            'type': self.module_type,
            'parameter_count': self.parameter_count,
            'flops': self.flops,
            'forward_time_ms': {
                'mean': np.mean(self.forward_time) if self.forward_time else 0,
                'std': np.std(self.forward_time) if self.forward_time else 0,
                'min': min(self.forward_time) if self.forward_time else 0,
                'max': max(self.forward_time) if self.forward_time else 0
            },
            'backward_time_ms': {
                'mean': np.mean(self.backward_time) if self.backward_time else 0,
                'std': np.std(self.backward_time) if self.backward_time else 0,
                'min': min(self.backward_time) if self.backward_time else 0,
                'max': max(self.backward_time) if self.backward_time else 0
            },
            'memory_mb': {
                'mean': np.mean(self.memory_usage) / 1024 / 1024 if self.memory_usage else 0,
                'max': max(self.memory_usage) / 1024 / 1024 if self.memory_usage else 0
            }
        }


class HardwareInfo:
    """Collect hardware information."""
    def __init__(self):
        self.cpu_info = self._get_cpu_info()
        self.gpu_info = self._get_gpu_info()
        self.memory_info = self._get_memory_info()
        
    def _get_cpu_info(self) -> Dict[str, Any]:
        if PSUTIL_AVAILABLE:
            return {
                'count': psutil.cpu_count(logical=False),
                'count_logical': psutil.cpu_count(logical=True),
                'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
                'percent': psutil.cpu_percent(interval=1)
            }
        else:
            return {
                'count': 1,
                'count_logical': 1,
                'frequency': 0,
                'percent': 0
            }
        
    def _get_gpu_info(self) -> List[Dict[str, Any]]:
        if not torch.cuda.is_available():
            return []
            
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append({
                'name': props.name,
                'compute_capability': f"{props.major}.{props.minor}",
                'memory_total_mb': props.total_memory / 1024 / 1024,
                'memory_allocated_mb': torch.cuda.memory_allocated(i) / 1024 / 1024,
                'memory_reserved_mb': torch.cuda.memory_reserved(i) / 1024 / 1024
            })
        return gpu_info
        
    def _get_memory_info(self) -> Dict[str, Any]:
        if PSUTIL_AVAILABLE:
            vm = psutil.virtual_memory()
            return {
                'total_mb': vm.total / 1024 / 1024,
                'available_mb': vm.available / 1024 / 1024,
                'used_percent': vm.percent
            }
        else:
            return {
                'total_mb': 0,
                'available_mb': 0,
                'used_percent': 0
            }


class ModelProfiler(nn.Module):
    def __init__(self,
                 profile_memory: bool = True,
                 profile_time: bool = True,
                 profile_flops: bool = True,
                 warmup_runs: int = 3,
                 profile_runs: int = 10,
                 detailed: bool = False,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.profile_memory = profile_memory
        self.profile_time = profile_time
        self.profile_flops = profile_flops
        self.warmup_runs = warmup_runs
        self.profile_runs = profile_runs
        self.detailed = detailed
        self.device = device
        
        # Profiling state
        self.layer_profiles = {}
        self.hooks = []
        self.hardware_info = HardwareInfo()
        self.is_profiling = False
        
    def profile_model(self, 
                     model: nn.Module,
                     input_data: Union[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]],
                     target_layers: Optional[List[str]] = None) -> Dict[str, Any]:
        """Profile the model with given input."""
        model = model.to(self.device)
        
        # Move input to device
        input_data = self._move_to_device(input_data)
        
        # Setup profiling hooks
        self._register_hooks(model, target_layers)
        
        try:
            # Warmup runs
            self._warmup(model, input_data)
            
            # Profile runs
            self._profile_forward(model, input_data)
            
            # Profile backward if model is in training mode
            if model.training and self.profile_time:
                self._profile_backward(model, input_data)
                
            # Compute FLOPs if requested
            if self.profile_flops:
                self._compute_flops(model, input_data)
                
            # Generate report
            report = self._generate_report()
            
        finally:
            # Cleanup hooks
            self._remove_hooks()
            
        return report
        
    def _move_to_device(self, data: Any) -> Any:
        """Move data to target device."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, dict):
            return {k: self._move_to_device(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self._move_to_device(v) for v in data)
        else:
            return data
            
    def _register_hooks(self, model: nn.Module, target_layers: Optional[List[str]]):
        """Register forward and backward hooks."""
        for name, module in model.named_modules():
            if target_layers is None or name in target_layers:
                # Skip container modules
                if len(list(module.children())) > 0 and not self.detailed:
                    continue
                    
                # Create layer profile
                profile = LayerProfile(name, module.__class__.__name__)
                profile.parameter_count = sum(p.numel() for p in module.parameters())
                self.layer_profiles[name] = profile
                
                # Register hooks
                if self.profile_time:
                    handle = module.register_forward_pre_hook(self._forward_pre_hook(name))
                    self.hooks.append(handle)
                    handle = module.register_forward_hook(self._forward_hook(name))
                    self.hooks.append(handle)
                    
                    if model.training:
                        handle = module.register_full_backward_hook(self._backward_hook(name))
                        self.hooks.append(handle)
                        
    def _forward_pre_hook(self, layer_name: str):
        """Hook to record forward pass start time."""
        def hook(module, input):
            if self.is_profiling:
                self.layer_profiles[layer_name]._forward_start = time.perf_counter()
                
                # Record input shapes
                if isinstance(input, tuple) and len(input) > 0:
                    if isinstance(input[0], torch.Tensor):
                        self.layer_profiles[layer_name].input_shapes.append(input[0].shape)
                        
                # Record memory before forward
                if self.profile_memory and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    self.layer_profiles[layer_name]._memory_before = torch.cuda.memory_allocated()
                    
        return hook
        
    def _forward_hook(self, layer_name: str):
        """Hook to record forward pass end time."""
        def hook(module, input, output):
            if self.is_profiling:
                # Record time
                if hasattr(self.layer_profiles[layer_name], '_forward_start'):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - self.layer_profiles[layer_name]._forward_start) * 1000
                    self.layer_profiles[layer_name].add_forward_time(elapsed)
                    
                # Record output shapes
                if isinstance(output, torch.Tensor):
                    self.layer_profiles[layer_name].output_shapes.append(output.shape)
                    
                # Record memory usage
                if self.profile_memory and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    memory_after = torch.cuda.memory_allocated()
                    if hasattr(self.layer_profiles[layer_name], '_memory_before'):
                        memory_used = memory_after - self.layer_profiles[layer_name]._memory_before
                        self.layer_profiles[layer_name].add_memory_usage(memory_used)
                        
        return hook
        
    def _backward_hook(self, layer_name: str):
        """Hook to record backward pass time."""
        def hook(module, grad_input, grad_output):
            if self.is_profiling:
                # Simple timing for backward pass
                # Note: This is approximate as backward hooks have limitations
                pass
                
        return hook
        
    def _warmup(self, model: nn.Module, input_data: Any):
        """Perform warmup runs."""
        model.eval()
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                if isinstance(input_data, dict):
                    _ = model(**input_data)
                elif isinstance(input_data, (tuple, list)):
                    _ = model(*input_data)
                else:
                    _ = model(input_data)
                    
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    
    def _profile_forward(self, model: nn.Module, input_data: Any):
        """Profile forward passes."""
        model.eval()
        self.is_profiling = True
        
        total_time = 0
        with torch.no_grad():
            for _ in range(self.profile_runs):
                start = time.perf_counter()
                
                if isinstance(input_data, dict):
                    _ = model(**input_data)
                elif isinstance(input_data, (tuple, list)):
                    _ = model(*input_data)
                else:
                    _ = model(input_data)
                    
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    
                total_time += (time.perf_counter() - start) * 1000
                
        self.is_profiling = False
        self.avg_forward_time = total_time / self.profile_runs
        
    def _profile_backward(self, model: nn.Module, input_data: Any):
        """Profile backward passes."""
        model.train()
        self.is_profiling = True
        
        total_time = 0
        for _ in range(self.profile_runs):
            # Forward
            if isinstance(input_data, dict):
                output = model(**input_data)
            elif isinstance(input_data, (tuple, list)):
                output = model(*input_data)
            else:
                output = model(input_data)
                
            # Create dummy loss
            if isinstance(output, dict):
                loss = sum(v.mean() for v in output.values() if isinstance(v, torch.Tensor))
            else:
                loss = output.mean()
                
            # Backward
            start = time.perf_counter()
            loss.backward()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            total_time += (time.perf_counter() - start) * 1000
            
            # Clear gradients
            model.zero_grad()
            
        self.is_profiling = False
        self.avg_backward_time = total_time / self.profile_runs
        
    def _compute_flops(self, model: nn.Module, input_data: Any):
        """Estimate FLOPs for the model."""
        # Simple FLOP estimation based on layer types
        total_flops = 0
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # FLOPs = 2 * input_size * output_size (multiply-add)
                if hasattr(module, 'in_features'):
                    flops = 2 * module.in_features * module.out_features
                    if name in self.layer_profiles:
                        self.layer_profiles[name].flops = flops
                    total_flops += flops
                    
            elif isinstance(module, nn.Conv2d):
                # FLOPs = 2 * kernel_size * in_channels * out_channels * output_spatial_size
                if hasattr(module, 'kernel_size'):
                    kernel_flops = np.prod(module.kernel_size)
                    channel_flops = module.in_channels * module.out_channels
                    # Estimate output size (this is approximate)
                    if name in self.layer_profiles and self.layer_profiles[name].output_shapes:
                        output_shape = self.layer_profiles[name].output_shapes[0]
                        spatial_flops = np.prod(output_shape[2:])
                        flops = 2 * kernel_flops * channel_flops * spatial_flops
                        self.layer_profiles[name].flops = flops
                        total_flops += flops
                        
        self.total_flops = total_flops
        
    def _remove_hooks(self):
        """Remove all profiling hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def _generate_report(self) -> Dict[str, Any]:
        """Generate profiling report."""
        report = {
            'summary': {
                'total_parameters': sum(p.parameter_count for p in self.layer_profiles.values()),
                'total_flops': getattr(self, 'total_flops', 0),
                'avg_forward_time_ms': getattr(self, 'avg_forward_time', 0),
                'avg_backward_time_ms': getattr(self, 'avg_backward_time', 0),
                'device': self.device
            },
            'hardware': {
                'cpu': self.hardware_info.cpu_info,
                'gpu': self.hardware_info.gpu_info,
                'memory': self.hardware_info.memory_info
            },
            'layers': {}
        }
        
        # Add layer-wise statistics
        for name, profile in self.layer_profiles.items():
            report['layers'][name] = profile.get_stats()
            
        # Identify bottlenecks
        report['bottlenecks'] = self._identify_bottlenecks()
        
        # Add optimization hints
        report['optimization_hints'] = self._generate_optimization_hints()
        
        return report
        
    def _identify_bottlenecks(self) -> Dict[str, List[str]]:
        """Identify performance bottlenecks."""
        bottlenecks = {
            'time': [],
            'memory': [],
            'compute': []
        }
        
        if not self.layer_profiles:
            return bottlenecks
            
        # Time bottlenecks (top 5 slowest layers)
        time_sorted = sorted(
            [(name, np.mean(p.forward_time)) for name, p in self.layer_profiles.items() if p.forward_time],
            key=lambda x: x[1],
            reverse=True
        )
        bottlenecks['time'] = [name for name, _ in time_sorted[:5]]
        
        # Memory bottlenecks (top 5 memory consuming layers)
        memory_sorted = sorted(
            [(name, max(p.memory_usage) if p.memory_usage else 0) for name, p in self.layer_profiles.items()],
            key=lambda x: x[1],
            reverse=True
        )
        bottlenecks['memory'] = [name for name, mem in memory_sorted[:5] if mem > 0]
        
        # Compute bottlenecks (highest FLOP layers)
        flops_sorted = sorted(
            [(name, p.flops) for name, p in self.layer_profiles.items()],
            key=lambda x: x[1],
            reverse=True
        )
        bottlenecks['compute'] = [name for name, flops in flops_sorted[:5] if flops > 0]
        
        return bottlenecks
        
    def _generate_optimization_hints(self) -> List[str]:
        """Generate optimization suggestions based on profiling."""
        hints = []
        
        # Check for inefficient memory usage
        total_params = sum(p.parameter_count for p in self.layer_profiles.values())
        if self.profile_memory and torch.cuda.is_available():
            total_memory = sum(max(p.memory_usage) if p.memory_usage else 0 for p in self.layer_profiles.values())
            param_memory = total_params * 4  # Assuming float32
            
            if total_memory > param_memory * 10:
                hints.append("High activation memory usage. Consider gradient checkpointing.")
                
        # Check for imbalanced computation
        if self.layer_profiles:
            forward_times = [np.mean(p.forward_time) for p in self.layer_profiles.values() if p.forward_time]
            if forward_times:
                time_variance = np.var(forward_times)
                if time_variance > np.mean(forward_times) ** 2:
                    hints.append("Highly imbalanced layer execution times. Consider layer fusion.")
                    
        # Check for small batch sizes
        for profile in self.layer_profiles.values():
            if profile.input_shapes and profile.input_shapes[0][0] < 32:
                hints.append(f"Small batch size detected ({profile.input_shapes[0][0]}). Consider increasing for better GPU utilization.")
                break
                
        # Mixed precision suggestion
        if torch.cuda.is_available() and not getattr(self, 'uses_mixed_precision', False):
            hints.append("Consider using mixed precision training for faster execution.")
            
        return hints
        
    def save_report(self, report: Dict[str, Any], path: str):
        """Save profiling report to file."""
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
            
    def compare_models(self, 
                      models: List[nn.Module],
                      names: List[str],
                      input_data: Any) -> Dict[str, Any]:
        """Compare multiple models."""
        comparison = {
            'models': {}
        }
        
        for model, name in zip(models, names):
            report = self.profile_model(model, input_data)
            comparison['models'][name] = report['summary']
            
        # Add relative comparisons
        if len(models) > 1:
            base_name = names[0]
            base_stats = comparison['models'][base_name]
            
            for name in names[1:]:
                stats = comparison['models'][name]
                stats['relative_to_' + base_name] = {
                    'parameters': stats['total_parameters'] / (base_stats['total_parameters'] + 1e-9),
                    'flops': stats['total_flops'] / (base_stats['total_flops'] + 1e-9),
                    'forward_time': stats['avg_forward_time_ms'] / (base_stats['avg_forward_time_ms'] + 1e-9)
                }
                
        return comparison