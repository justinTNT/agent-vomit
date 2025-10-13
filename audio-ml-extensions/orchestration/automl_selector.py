import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable, Union, Tuple, Type
import numpy as np
from collections import defaultdict
import warnings
import json
import time


class TaskProfile:
    """Profile of a machine learning task."""
    def __init__(self):
        self.input_shape = None
        self.output_shape = None
        self.task_type = None  # 'classification', 'regression', 'generation', etc.
        self.data_modality = None  # 'image', 'text', 'audio', 'tabular', 'graph'
        self.dataset_size = None
        self.constraints = {}
        self.performance_targets = {}
        
    def analyze_data(self, data_sample: Any) -> 'TaskProfile':
        """Analyze data to determine task characteristics."""
        if isinstance(data_sample, torch.Tensor):
            self.input_shape = data_sample.shape
            
            # Infer modality from shape
            if len(data_sample.shape) == 4:  # Batch, Channels, Height, Width
                self.data_modality = 'image'
            elif len(data_sample.shape) == 3:  # Batch, Sequence, Features
                if data_sample.shape[-1] > 100:
                    self.data_modality = 'text'
                else:
                    self.data_modality = 'sequence'
            elif len(data_sample.shape) == 2:  # Batch, Features
                self.data_modality = 'tabular'
                
        elif isinstance(data_sample, dict):
            # Handle dictionary inputs
            if 'edge_index' in data_sample:
                self.data_modality = 'graph'
            elif 'input_ids' in data_sample:
                self.data_modality = 'text'
                
        return self


class ModuleCandidate:
    """Candidate module with compatibility scores."""
    def __init__(self, 
                 module_class: Type[nn.Module],
                 config: Dict[str, Any],
                 score: float = 0.0):
        self.module_class = module_class
        self.config = config
        self.score = score
        self.compatibility_reasons = []
        self.estimated_performance = {}
        

class AutoMLSelector(nn.Module):
    def __init__(self,
                 module_registry: Optional[Dict[str, Type[nn.Module]]] = None,
                 selection_strategy: str = 'greedy',  # 'greedy', 'ensemble', 'adaptive'
                 performance_predictor: Optional[Callable] = None,
                 compatibility_rules: Optional[Dict[str, Callable]] = None,
                 search_budget: int = 10,
                 ensemble_size: int = 3):
        super().__init__()
        self.module_registry = module_registry or self._get_default_registry()
        self.selection_strategy = selection_strategy
        self.performance_predictor = performance_predictor
        self.compatibility_rules = compatibility_rules or self._get_default_rules()
        self.search_budget = search_budget
        self.ensemble_size = ensemble_size
        
        # Selection history
        self.selection_history = []
        self.performance_cache = {}
        
    def _get_default_registry(self) -> Dict[str, Type[nn.Module]]:
        """Get default module registry."""
        registry = {
            # Basic layers
            'linear': nn.Linear,
            'conv2d': nn.Conv2d,
            'lstm': nn.LSTM,
            'transformer': nn.Transformer,
            
            # Activation functions
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh
        }
        
        # Try to import vision models if available
        try:
            from torchvision.models import resnet18, resnet50, vgg16
            registry.update({
                'resnet18': resnet18,
                'resnet50': resnet50,
                'vgg16': vgg16,
            })
        except ImportError:
            pass
        
        return registry
        
    def _get_default_rules(self) -> Dict[str, Callable]:
        """Get default compatibility rules."""
        rules = {
            'image_requires_conv': lambda task, module: (
                task.data_modality != 'image' or 
                'conv' in module.__name__.lower() or
                'resnet' in module.__name__.lower()
            ),
            'sequence_requires_recurrent': lambda task, module: (
                task.data_modality != 'sequence' or
                any(x in module.__name__.lower() for x in ['lstm', 'gru', 'transformer'])
            ),
            'graph_requires_gnn': lambda task, module: (
                task.data_modality != 'graph' or
                'graph' in module.__name__.lower()
            )
        }
        
        return rules
        
    def analyze_task(self, 
                    train_data: Any,
                    target_data: Optional[Any] = None,
                    task_type: Optional[str] = None) -> TaskProfile:
        """Analyze the ML task from data."""
        profile = TaskProfile()
        profile.analyze_data(train_data)
        
        # Infer task type if not provided
        if task_type:
            profile.task_type = task_type
        elif target_data is not None:
            if isinstance(target_data, torch.Tensor):
                if target_data.dtype in [torch.long, torch.int]:
                    profile.task_type = 'classification'
                else:
                    profile.task_type = 'regression'
                profile.output_shape = target_data.shape
                
        return profile
        
    def select_modules(self, 
                      task_profile: TaskProfile,
                      constraints: Optional[Dict[str, Any]] = None) -> List[ModuleCandidate]:
        """Select best modules for the task."""
        candidates = []
        
        # Generate candidates
        for name, module_class in self.module_registry.items():
            # Check compatibility
            if self._is_compatible(module_class, task_profile):
                # Generate configs
                configs = self._generate_configs(module_class, task_profile)
                
                for config in configs:
                    candidate = ModuleCandidate(module_class, config)
                    
                    # Score candidate
                    candidate.score = self._score_candidate(candidate, task_profile)
                    
                    # Estimate performance
                    if self.performance_predictor:
                        candidate.estimated_performance = self.performance_predictor(
                            candidate, task_profile
                        )
                        
                    candidates.append(candidate)
                    
                    if len(candidates) >= self.search_budget:
                        break
                        
            if len(candidates) >= self.search_budget:
                break
                
        # Sort by score
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        # Apply selection strategy
        if self.selection_strategy == 'greedy':
            selected = candidates[:1]
        elif self.selection_strategy == 'ensemble':
            selected = self._select_ensemble(candidates, self.ensemble_size)
        elif self.selection_strategy == 'adaptive':
            selected = self._select_adaptive(candidates, task_profile)
        else:
            selected = candidates[:1]
            
        # Record selection
        self.selection_history.append({
            'task_profile': task_profile,
            'candidates': selected,
            'timestamp': time.time()
        })
        
        return selected
        
    def _is_compatible(self, module_class: Type[nn.Module], task_profile: TaskProfile) -> bool:
        """Check if module is compatible with task."""
        # Apply compatibility rules
        for rule_name, rule_fn in self.compatibility_rules.items():
            if not rule_fn(task_profile, module_class):
                return False
                
        return True
        
    def _generate_configs(self, 
                         module_class: Type[nn.Module],
                         task_profile: TaskProfile) -> List[Dict[str, Any]]:
        """Generate module configurations."""
        configs = []
        
        # Get module signature
        import inspect
        sig = inspect.signature(module_class.__init__)
        params = sig.parameters
        
        # Generate base config
        base_config = {}
        
        # Handle common parameters
        if 'in_features' in params and task_profile.input_shape:
            base_config['in_features'] = task_profile.input_shape[-1]
        if 'in_channels' in params and task_profile.input_shape and len(task_profile.input_shape) > 2:
            base_config['in_channels'] = task_profile.input_shape[1]
        if 'input_size' in params and task_profile.input_shape:
            base_config['input_size'] = task_profile.input_shape[-1]
            
        # Generate variations
        if module_class == nn.Linear:
            # Linear layer configs
            for hidden_size in [64, 128, 256, 512]:
                config = base_config.copy()
                config['out_features'] = hidden_size
                configs.append(config)
        elif module_class == nn.Conv2d:
            # Conv layer configs
            for out_channels in [32, 64, 128]:
                for kernel_size in [3, 5]:
                    config = base_config.copy()
                    config.update({
                        'out_channels': out_channels,
                        'kernel_size': kernel_size,
                        'padding': kernel_size // 2
                    })
                    configs.append(config)
        elif module_class == nn.LSTM:
            # LSTM configs
            for hidden_size in [128, 256]:
                for num_layers in [1, 2]:
                    config = base_config.copy()
                    config.update({
                        'hidden_size': hidden_size,
                        'num_layers': num_layers,
                        'batch_first': True
                    })
                    configs.append(config)
        else:
            # Default config
            configs.append(base_config)
            
        return configs[:3]  # Limit number of configs
        
    def _score_candidate(self, candidate: ModuleCandidate, task_profile: TaskProfile) -> float:
        """Score a candidate module."""
        score = 0.0
        
        # Modality match
        module_name = candidate.module_class.__name__.lower()
        if task_profile.data_modality == 'image' and 'conv' in module_name:
            score += 0.3
            candidate.compatibility_reasons.append("Convolutional layer for image data")
        elif task_profile.data_modality == 'sequence' and any(x in module_name for x in ['lstm', 'gru']):
            score += 0.3
            candidate.compatibility_reasons.append("Recurrent layer for sequence data")
        elif task_profile.data_modality == 'text' and 'transformer' in module_name:
            score += 0.3
            candidate.compatibility_reasons.append("Transformer for text data")
            
        # Parameter efficiency
        if hasattr(candidate.module_class, '__name__'):
            # Estimate parameter count
            if 'efficientnet' in module_name or 'mobilenet' in module_name:
                score += 0.2
                candidate.compatibility_reasons.append("Parameter efficient architecture")
                
        # Task type match
        if task_profile.task_type == 'classification':
            if any(x in module_name for x in ['classifier', 'resnet', 'vgg']):
                score += 0.2
                candidate.compatibility_reasons.append("Classification-oriented architecture")
                
        # Performance history
        cache_key = f"{module_name}_{task_profile.data_modality}"
        if cache_key in self.performance_cache:
            historical_score = self.performance_cache[cache_key]
            score += 0.3 * historical_score
            candidate.compatibility_reasons.append(f"Historical performance: {historical_score:.2f}")
            
        return score
        
    def _select_ensemble(self, 
                        candidates: List[ModuleCandidate],
                        ensemble_size: int) -> List[ModuleCandidate]:
        """Select diverse ensemble of modules."""
        selected = []
        remaining = candidates.copy()
        
        # Select highest scoring first
        if remaining:
            selected.append(remaining.pop(0))
            
        # Select diverse candidates
        while len(selected) < ensemble_size and remaining:
            # Find most diverse candidate
            max_diversity = -1
            most_diverse = None
            most_diverse_idx = -1
            
            for i, candidate in enumerate(remaining):
                diversity = self._compute_diversity(candidate, selected)
                if diversity > max_diversity:
                    max_diversity = diversity
                    most_diverse = candidate
                    most_diverse_idx = i
                    
            if most_diverse:
                selected.append(most_diverse)
                remaining.pop(most_diverse_idx)
                
        return selected
        
    def _compute_diversity(self, 
                          candidate: ModuleCandidate,
                          selected: List[ModuleCandidate]) -> float:
        """Compute diversity score between candidate and selected modules."""
        if not selected:
            return 1.0
            
        diversity = 0.0
        
        # Architecture diversity
        candidate_type = candidate.module_class.__name__
        for s in selected:
            if s.module_class.__name__ != candidate_type:
                diversity += 0.5
                
        # Configuration diversity
        for s in selected:
            if s.config != candidate.config:
                diversity += 0.3
                
        # Normalize
        diversity /= len(selected)
        
        return diversity
        
    def _select_adaptive(self, 
                        candidates: List[ModuleCandidate],
                        task_profile: TaskProfile) -> List[ModuleCandidate]:
        """Adaptively select modules based on task complexity."""
        # Estimate task complexity
        complexity = self._estimate_task_complexity(task_profile)
        
        if complexity < 0.3:
            # Simple task - single module
            return candidates[:1]
        elif complexity < 0.7:
            # Medium complexity - small ensemble
            return self._select_ensemble(candidates, 2)
        else:
            # Complex task - larger ensemble
            return self._select_ensemble(candidates, self.ensemble_size)
            
    def _estimate_task_complexity(self, task_profile: TaskProfile) -> float:
        """Estimate task complexity from profile."""
        complexity = 0.0
        
        # Data complexity
        if task_profile.input_shape:
            # Higher dimensional data is more complex
            complexity += len(task_profile.input_shape) * 0.1
            
            # Larger inputs are more complex
            input_size = np.prod(task_profile.input_shape[1:])
            complexity += min(input_size / 10000, 0.3)
            
        # Modality complexity
        modality_complexity = {
            'tabular': 0.1,
            'sequence': 0.3,
            'text': 0.4,
            'image': 0.3,
            'graph': 0.5
        }
        if task_profile.data_modality:
            complexity += modality_complexity.get(task_profile.data_modality, 0.2)
            
        return min(complexity, 1.0)
        
    def build_pipeline(self, 
                      selected_modules: List[ModuleCandidate],
                      task_profile: TaskProfile) -> nn.Module:
        """Build complete pipeline from selected modules."""
        
        class AutoMLPipeline(nn.Module):
            def __init__(self, modules):
                super().__init__()
                self.modules_list = nn.ModuleList()
                
                for i, candidate in enumerate(modules):
                    try:
                        module = candidate.module_class(**candidate.config)
                        self.modules_list.append(module)
                    except Exception as e:
                        warnings.warn(f"Failed to instantiate {candidate.module_class}: {e}")
                        
            def forward(self, x):
                for module in self.modules_list:
                    x = module(x)
                return x
                
        return AutoMLPipeline(selected_modules)
        
    def update_performance(self, 
                          module_name: str,
                          task_modality: str,
                          performance_score: float):
        """Update performance cache with observed results."""
        cache_key = f"{module_name}_{task_modality}"
        
        if cache_key in self.performance_cache:
            # Running average
            old_score = self.performance_cache[cache_key]
            self.performance_cache[cache_key] = 0.7 * old_score + 0.3 * performance_score
        else:
            self.performance_cache[cache_key] = performance_score
            
    def export_selection_rules(self) -> Dict[str, Any]:
        """Export learned selection rules."""
        return {
            'performance_cache': self.performance_cache,
            'selection_history': [
                {
                    'task_modality': h['task_profile'].data_modality,
                    'selected_modules': [c.module_class.__name__ for c in h['candidates']]
                }
                for h in self.selection_history[-100:]  # Last 100 selections
            ]
        }
        
    def forward(self, 
                train_data: Any,
                target_data: Optional[Any] = None,
                constraints: Optional[Dict[str, Any]] = None) -> nn.Module:
        """Full AutoML pipeline: analyze task and build model."""
        # Analyze task
        task_profile = self.analyze_task(train_data, target_data)
        
        # Select modules
        selected = self.select_modules(task_profile, constraints)
        
        # Build pipeline
        model = self.build_pipeline(selected, task_profile)
        
        return model