"""
Simple tests for orchestration modules - one at a time
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'orchestration'))

# Test PipelineOrchestrator basic functionality
try:
    from pipeline_orchestrator import PipelineOrchestrator
    
    class TestModule(nn.Module):
        def __init__(self, in_dim=10, out_dim=5):
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)
        def forward(self, x):
            return self.linear(x)
    
    # Create simple pipeline
    orchestrator = PipelineOrchestrator(parallel_execution=False)  # Sequential first
    orchestrator.add_stage('stage1', TestModule, {'in_dim': 10, 'out_dim': 20})
    orchestrator.add_stage('stage2', TestModule, {'in_dim': 20, 'out_dim': 5}, dependencies=['stage1'])
    
    # Test forward
    inputs = {'x': torch.randn(4, 10)}
    outputs = orchestrator(inputs)
    
    print("✓ PipelineOrchestrator: Basic sequential execution works")
    print(f"  Output shape: {outputs['stage2'].shape}")
    
except Exception as e:
    print(f"✗ PipelineOrchestrator failed: {e}")

# Test HyperparameterOptimizer
try:
    from hyperparameter_optimizer import HyperparameterOptimizer, ParameterSpace
    
    # Simple optimization
    param_space = [
        ParameterSpace('x', 'float', bounds=(-5, 5)),
        ParameterSpace('y', 'float', bounds=(-5, 5))
    ]
    
    optimizer = HyperparameterOptimizer(
        parameter_space=param_space,
        objective='minimize',
        n_trials=10
    )
    
    # Run a few iterations
    for i in range(5):
        params = optimizer.suggest()
        # Simple quadratic function
        loss = (params['x'] - 2)**2 + (params['y'] + 1)**2
        optimizer.update(params, loss)
    
    best = optimizer.get_best_parameters()
    print("✓ HyperparameterOptimizer: Basic optimization works")
    print(f"  Best params: x={best['x']:.2f}, y={best['y']:.2f}")
    
except Exception as e:
    print(f"✗ HyperparameterOptimizer failed: {e}")

# Test DataflowOptimizer
try:
    from dataflow_optimizer import DataflowOptimizer
    
    optimizer = DataflowOptimizer(batch_optimization=True)
    
    # Test batch size optimization
    module = nn.Linear(100, 50)
    optimal_batch = optimizer.optimize_batch_size(module, (32, 100))
    
    print("✓ DataflowOptimizer: Batch optimization works")
    print(f"  Optimal batch size: {optimal_batch}")
    
except Exception as e:
    print(f"✗ DataflowOptimizer failed: {e}")

# Test ModelProfiler
try:
    from model_profiler import ModelProfiler
    
    profiler = ModelProfiler(warmup_runs=1, profile_runs=2)
    
    # Simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    report = profiler.profile_model(model, torch.randn(4, 10))
    
    print("✓ ModelProfiler: Basic profiling works")
    print(f"  Total parameters: {report['summary']['total_parameters']}")
    print(f"  Number of layers profiled: {len(report['layers'])}")
    
except Exception as e:
    print(f"✗ ModelProfiler failed: {e}")

# Test ExperimentTracker
try:
    from experiment_tracker import ExperimentTracker
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        tracker = ExperimentTracker('test_project', base_dir=temp_dir)
        
        # Create and log
        exp_id = tracker.create_experiment('test_exp', config={'lr': 0.01})
        tracker.log_metrics({'loss': 0.5, 'acc': 0.8})
        tracker.end_experiment()
        
        print("✓ ExperimentTracker: Basic tracking works")
        print(f"  Experiment ID: {exp_id}")
        
except Exception as e:
    print(f"✗ ExperimentTracker failed: {e}")

# Test AutoMLSelector
try:
    from automl_selector import AutoMLSelector
    
    selector = AutoMLSelector()
    
    # Analyze task
    data = torch.randn(32, 3, 64, 64)  # Image-like data
    profile = selector.analyze_task(data)
    
    # Select modules
    candidates = selector.select_modules(profile)
    
    print("✓ AutoMLSelector: Basic selection works")
    print(f"  Detected modality: {profile.data_modality}")
    print(f"  Selected modules: {len(candidates)}")
    
except Exception as e:
    print(f"✗ AutoMLSelector failed: {e}")

print("\nAll basic tests completed!")