"""
Comprehensive tests for orchestration/optimization modules.
Following the same approach as the original 25 modules.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional
import numpy as np
import sys
import os
import time
import tempfile
import shutil

# Add orchestration directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'orchestration'))

# Test utilities
def run_test(test_name: str, test_func: callable) -> bool:
    """Run a single test and report results."""
    try:
        test_func()
        print(f"✓ {test_name}")
        return True
    except Exception as e:
        print(f"✗ {test_name}: {str(e)}")
        return False


class SimpleModule(nn.Module):
    """Simple test module for orchestration tests."""
    def __init__(self, input_dim: int = 10, output_dim: int = 5):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def forward(self, x):
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {x.shape[-1]}")
        return self.linear(x)


# 1. PipelineOrchestrator Tests
def test_pipeline_orchestrator():
    """Test PipelineOrchestrator module."""
    from pipeline_orchestrator import PipelineOrchestrator
    
    def test_basic_pipeline():
        # Create orchestrator
        orchestrator = PipelineOrchestrator(
            target_latency=100,
            optimization_goal='throughput'
        )
        
        # Add stages
        orchestrator.add_stage('encoder', SimpleModule, {'input_dim': 10, 'output_dim': 20})
        orchestrator.add_stage('processor', SimpleModule, {'input_dim': 20, 'output_dim': 15})
        orchestrator.add_stage('decoder', SimpleModule, {'input_dim': 15, 'output_dim': 5}, 
                             dependencies=['encoder', 'processor'])
        
        # Compile
        orchestrator.compile()
        
        # Test forward pass
        inputs = {'x': torch.randn(4, 10)}
        outputs = orchestrator(inputs)
        
        assert 'decoder' in outputs, "Missing decoder output"
        assert outputs['decoder'].shape == (4, 5), f"Wrong output shape: {outputs['decoder'].shape}"
        
    def test_parallel_execution():
        orchestrator = PipelineOrchestrator(parallel_execution=True)
        
        # Add independent stages
        orchestrator.add_stage('branch1', SimpleModule, {'input_dim': 10, 'output_dim': 5})
        orchestrator.add_stage('branch2', SimpleModule, {'input_dim': 10, 'output_dim': 5})
        
        # Test execution
        inputs = {'x': torch.randn(4, 10)}
        outputs = orchestrator(inputs)
        
        assert 'branch1' in outputs and 'branch2' in outputs, "Missing parallel outputs"
        
    def test_profile_info():
        orchestrator = PipelineOrchestrator()
        orchestrator.add_stage('stage1', SimpleModule)
        
        inputs = {'x': torch.randn(4, 10)}
        _ = orchestrator(inputs)
        
        profile = orchestrator.profile()
        assert 'stages' in profile, "Missing profile stages"
        assert 'total_latency' in profile, "Missing total latency"
        
    def test_export_config():
        orchestrator = PipelineOrchestrator(target_latency=50)
        orchestrator.add_stage('test', SimpleModule)
        
        config = orchestrator.export()
        assert config['target_latency'] == 50, "Wrong target latency"
        assert 'stages' in config, "Missing stages in export"
        
    # Run tests
    run_test("Basic pipeline creation", test_basic_pipeline)
    run_test("Parallel execution", test_parallel_execution)
    run_test("Profile information", test_profile_info)
    run_test("Export configuration", test_export_config)


# 2. HyperparameterOptimizer Tests
def test_hyperparameter_optimizer():
    """Test HyperparameterOptimizer module."""
    from hyperparameter_optimizer import HyperparameterOptimizer, ParameterSpace
    
    def test_parameter_space():
        # Define search space
        param_space = [
            ParameterSpace('learning_rate', 'float', bounds=(0.0001, 0.1), log_scale=True),
            ParameterSpace('batch_size', 'int', bounds=(8, 128)),
            ParameterSpace('optimizer', 'categorical', choices=['adam', 'sgd']),
            ParameterSpace('use_dropout', 'bool')
        ]
        
        # Create optimizer
        optimizer = HyperparameterOptimizer(
            parameter_space=param_space,
            objective='maximize',
            n_trials=5,
            n_random_starts=2
        )
        
        # Test suggestion
        params = optimizer.suggest()
        assert 'learning_rate' in params, "Missing learning_rate"
        assert 'batch_size' in params, "Missing batch_size"
        assert params['optimizer'] in ['adam', 'sgd'], f"Invalid optimizer: {params['optimizer']}"
        
    def test_optimization_loop():
        param_space = [
            ParameterSpace('x', 'float', bounds=(-5, 5)),
            ParameterSpace('y', 'float', bounds=(-5, 5))
        ]
        
        optimizer = HyperparameterOptimizer(
            parameter_space=param_space,
            objective='minimize',
            n_trials=10
        )
        
        # Optimize simple function: (x-2)^2 + (y+1)^2
        for _ in range(10):
            params = optimizer.suggest()
            objective = (params['x'] - 2)**2 + (params['y'] + 1)**2
            optimizer.update(params, objective)
            
            if optimizer.should_stop():
                break
                
        best_params = optimizer.get_best_parameters()
        assert best_params is not None, "No best parameters found"
        
        # Check if reasonably close to optimum
        assert abs(best_params['x'] - 2) < 1.0, f"X not close to optimum: {best_params['x']}"
        assert abs(best_params['y'] + 1) < 1.0, f"Y not close to optimum: {best_params['y']}"
        
    def test_constraints():
        param_space = [ParameterSpace('x', 'float', bounds=(0, 10))]
        
        # Constraint: x must be > 5
        constraints = [lambda model, params: params['x'] - 5]
        
        optimizer = HyperparameterOptimizer(
            parameter_space=param_space,
            constraints=constraints,
            n_trials=20
        )
        
        # Run optimization
        for _ in range(20):
            params = optimizer.suggest()
            objective = params['x']  # Minimize x subject to x > 5
            constraint_values = {'constraint_0': params['x'] - 5}
            optimizer.update(params, objective, constraint_values)
            
        best = optimizer.get_best_parameters()
        assert best['x'] >= 5, f"Constraint violated: {best['x']}"
        
    def test_save_results():
        param_space = [ParameterSpace('x', 'float', bounds=(0, 1))]
        optimizer = HyperparameterOptimizer(parameter_space=param_space, n_trials=3)
        
        # Run a few trials
        for i in range(3):
            params = optimizer.suggest()
            optimizer.update(params, i)
            
        # Save results
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            optimizer.save_results(f.name)
            temp_path = f.name
            
        # Check file exists
        assert os.path.exists(temp_path), "Results file not created"
        os.unlink(temp_path)
        
    # Run tests
    run_test("Parameter space sampling", test_parameter_space)
    run_test("Optimization loop", test_optimization_loop)
    run_test("Constraint handling", test_constraints)
    run_test("Save results", test_save_results)


# 3. DataflowOptimizer Tests
def test_dataflow_optimizer():
    """Test DataflowOptimizer module."""
    from dataflow_optimizer import DataflowOptimizer
    
    def test_batch_optimization():
        optimizer = DataflowOptimizer(
            batch_optimization=True,
            memory_optimization=False
        )
        
        # Test module
        module = SimpleModule(100, 50)
        
        # Find optimal batch size
        optimal_batch = optimizer.optimize_batch_size(
            module,
            input_shape=(32, 100),
            target_memory_usage=0.9
        )
        
        assert optimal_batch > 0, f"Invalid batch size: {optimal_batch}"
        assert optimal_batch <= 512, f"Batch size too large: {optimal_batch}"
        
    def test_memory_optimization():
        optimizer = DataflowOptimizer(
            memory_optimization=True,
            gradient_checkpointing=True
        )
        
        # Create modules
        modules = [SimpleModule(10, 20), SimpleModule(20, 30), SimpleModule(30, 10)]
        execution_order = [0, 1, 2]
        
        # Get optimization plan
        plan = optimizer.optimize_memory_allocation(modules, execution_order)
        
        assert 'recompute' in plan, "Missing recompute plan"
        assert 'inplace' in plan, "Missing inplace plan"
        assert isinstance(plan['recompute'], list), "Recompute should be list"
        
    def test_mixed_precision():
        if not torch.cuda.is_available():
            return  # Skip on CPU
            
        optimizer = DataflowOptimizer(mixed_precision=True)
        module = SimpleModule()
        
        # Test forward with mixed precision
        inputs = torch.randn(4, 10, device='cuda')
        outputs = optimizer(module, inputs)
        
        assert outputs.shape == (4, 5), f"Wrong output shape: {outputs.shape}"
        
    def test_pipeline_optimization():
        optimizer = DataflowOptimizer(
            batch_optimization=True,
            memory_optimization=True
        )
        
        # Create pipeline
        pipeline = [SimpleModule(10, 20), SimpleModule(20, 15), SimpleModule(15, 5)]
        sample_input = torch.randn(8, 10)
        
        # Optimize pipeline
        optimizations = optimizer.optimize_pipeline(pipeline, sample_input)
        
        assert 'batch_sizes' in optimizations, "Missing batch sizes"
        assert 'memory_plan' in optimizations, "Missing memory plan"
        assert len(optimizations['batch_sizes']) > 0, "No batch sizes computed"
        
    # Run tests
    run_test("Batch size optimization", test_batch_optimization)
    run_test("Memory optimization plan", test_memory_optimization)
    if torch.cuda.is_available():
        run_test("Mixed precision execution", test_mixed_precision)
    run_test("Pipeline optimization", test_pipeline_optimization)


# 4. ModelProfiler Tests
def test_model_profiler():
    """Test ModelProfiler module."""
    from model_profiler import ModelProfiler
    
    def test_basic_profiling():
        profiler = ModelProfiler(
            profile_memory=True,
            profile_time=True,
            profile_flops=True,
            warmup_runs=1,
            profile_runs=3
        )
        
        # Create model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        # Profile
        input_data = torch.randn(4, 10)
        report = profiler.profile_model(model, input_data)
        
        assert 'summary' in report, "Missing summary"
        assert 'layers' in report, "Missing layers"
        assert report['summary']['total_parameters'] == 10*20 + 20 + 20*5 + 5, "Wrong parameter count"
        
    def test_layer_profiling():
        profiler = ModelProfiler(detailed=True)
        
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1)
        )
        
        input_data = torch.randn(1, 3, 32, 32)
        report = profiler.profile_model(model, input_data)
        
        # Check layer information
        assert len(report['layers']) > 0, "No layers profiled"
        
        # Check bottleneck identification
        assert 'bottlenecks' in report, "Missing bottlenecks"
        assert 'time' in report['bottlenecks'], "Missing time bottlenecks"
        
    def test_model_comparison():
        profiler = ModelProfiler(warmup_runs=1, profile_runs=2)
        
        # Create two models
        model1 = nn.Linear(10, 5)
        model2 = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        # Compare
        comparison = profiler.compare_models(
            [model1, model2],
            ['simple', 'complex'],
            torch.randn(4, 10)
        )
        
        assert 'models' in comparison, "Missing models in comparison"
        assert 'simple' in comparison['models'], "Missing simple model"
        assert 'complex' in comparison['models'], "Missing complex model"
        
        # Check relative comparison
        assert 'relative_to_simple' in comparison['models']['complex'], "Missing relative comparison"
        
    def test_optimization_hints():
        profiler = ModelProfiler()
        
        # Model with small batch
        model = nn.Linear(100, 50)
        input_data = torch.randn(2, 100)  # Small batch
        
        report = profiler.profile_model(model, input_data)
        
        assert 'optimization_hints' in report, "Missing optimization hints"
        hints = report['optimization_hints']
        
        # Should suggest larger batch size
        assert any('batch size' in hint for hint in hints), "Should suggest batch size increase"
        
    # Run tests
    run_test("Basic model profiling", test_basic_profiling)
    run_test("Layer-wise profiling", test_layer_profiling)
    run_test("Model comparison", test_model_comparison)
    run_test("Optimization hints", test_optimization_hints)


# 5. ExperimentTracker Tests
def test_experiment_tracker():
    """Test ExperimentTracker module."""
    from experiment_tracker import ExperimentTracker
    
    # Use temporary directory for tests
    temp_dir = tempfile.mkdtemp()
    
    def test_experiment_creation():
        tracker = ExperimentTracker('test_project', base_dir=temp_dir)
        
        # Create experiment
        exp_id = tracker.create_experiment(
            'test_experiment',
            config={'lr': 0.001, 'batch_size': 32},
            tags=['test', 'demo']
        )
        
        assert exp_id is not None, "No experiment ID returned"
        assert tracker.current_experiment is not None, "No current experiment"
        assert tracker.current_experiment.name == 'test_experiment', "Wrong experiment name"
        
    def test_metric_logging():
        tracker = ExperimentTracker('test_project', base_dir=temp_dir)
        tracker.create_experiment('metric_test')
        
        # Log metrics
        for i in range(5):
            tracker.log_metrics({
                'loss': 1.0 / (i + 1),
                'accuracy': i * 0.2
            })
            
        # Check metrics
        metrics = tracker.current_experiment.metrics
        assert 'loss' in metrics, "Missing loss metric"
        assert 'accuracy' in metrics, "Missing accuracy metric"
        assert len(metrics['loss']) == 5, "Wrong number of loss values"
        
    def test_artifact_logging():
        tracker = ExperimentTracker('test_project', base_dir=temp_dir)
        tracker.create_experiment('artifact_test')
        
        # Log different artifact types
        tracker.log_artifact('config', {'key': 'value'})
        tracker.log_artifact('model', SimpleModule())
        tracker.log_artifact('data', [1, 2, 3])
        
        artifacts = tracker.current_experiment.artifacts
        assert len(artifacts) == 3, f"Wrong number of artifacts: {len(artifacts)}"
        
    def test_checkpoint_save_load():
        tracker = ExperimentTracker('test_project', base_dir=temp_dir)
        tracker.create_experiment('checkpoint_test')
        
        # Create and save model
        model = SimpleModule()
        optimizer = optim.Adam(model.parameters())
        
        tracker.save_checkpoint(model, optimizer, epoch=1, metrics={'loss': 0.5})
        
        # Load checkpoint
        new_model = SimpleModule()
        new_optimizer = optim.Adam(new_model.parameters())
        
        checkpoint = tracker.load_checkpoint(0, new_model, new_optimizer)
        assert checkpoint['epoch'] == 1, "Wrong epoch"
        assert checkpoint['metrics']['loss'] == 0.5, "Wrong loss value"
        
    def test_experiment_comparison():
        tracker = ExperimentTracker('test_project', base_dir=temp_dir)
        
        # Create multiple experiments
        exp_ids = []
        for i in range(3):
            exp_id = tracker.create_experiment(f'exp_{i}')
            tracker.log_metrics({'accuracy': 0.8 + i * 0.05})
            tracker.end_experiment()
            exp_ids.append(exp_id)
            
        # Compare experiments
        comparison = tracker.compare_experiments(exp_ids, metrics=['accuracy'])
        
        assert 'metrics' in comparison, "Missing metrics in comparison"
        assert 'accuracy' in comparison['metrics'], "Missing accuracy comparison"
        
        # Find best experiment
        best_id = tracker.get_best_experiment('accuracy', mode='max')
        assert best_id == exp_ids[2], "Wrong best experiment"
        
    # Run tests
    run_test("Experiment creation", test_experiment_creation)
    run_test("Metric logging", test_metric_logging)
    run_test("Artifact logging", test_artifact_logging)
    run_test("Checkpoint save/load", test_checkpoint_save_load)
    run_test("Experiment comparison", test_experiment_comparison)
    
    # Cleanup
    shutil.rmtree(temp_dir)


# 6. AutoMLSelector Tests
def test_automl_selector():
    """Test AutoMLSelector module."""
    from automl_selector import AutoMLSelector
    
    def test_task_analysis():
        selector = AutoMLSelector()
        
        # Image data
        image_data = torch.randn(32, 3, 224, 224)
        profile = selector.analyze_task(image_data)
        
        assert profile.data_modality == 'image', f"Wrong modality: {profile.data_modality}"
        assert profile.input_shape == image_data.shape, "Wrong input shape"
        
        # Sequence data
        seq_data = torch.randn(32, 100, 50)
        profile = selector.analyze_task(seq_data)
        assert profile.data_modality == 'sequence', f"Wrong modality: {profile.data_modality}"
        
    def test_module_selection():
        selector = AutoMLSelector(selection_strategy='greedy')
        
        # Create task profile
        from automl_selector import TaskProfile
        task = TaskProfile()
        task.data_modality = 'image'
        task.input_shape = (32, 3, 64, 64)
        task.task_type = 'classification'
        
        # Select modules
        candidates = selector.select_modules(task)
        
        assert len(candidates) > 0, "No candidates selected"
        assert candidates[0].score > 0, "No score assigned"
        
        # Check compatibility reasons
        assert len(candidates[0].compatibility_reasons) > 0, "No compatibility reasons"
        
    def test_ensemble_selection():
        selector = AutoMLSelector(
            selection_strategy='ensemble',
            ensemble_size=3
        )
        
        from automl_selector import TaskProfile
        task = TaskProfile()
        task.data_modality = 'sequence'
        task.input_shape = (32, 50, 128)
        
        candidates = selector.select_modules(task)
        
        assert len(candidates) <= 3, f"Too many candidates: {len(candidates)}"
        
        # Check diversity
        module_types = [c.module_class.__name__ for c in candidates]
        assert len(set(module_types)) > 1 or len(candidates) == 1, "No diversity in ensemble"
        
    def test_pipeline_building():
        selector = AutoMLSelector()
        
        # Full pipeline test
        data = torch.randn(8, 10)
        targets = torch.randint(0, 2, (8,))
        
        model = selector(data, targets)
        
        assert isinstance(model, nn.Module), "Should return nn.Module"
        
        # Test forward pass
        output = model(data)
        assert output.shape[0] == 8, f"Wrong batch size: {output.shape}"
        
    def test_performance_update():
        selector = AutoMLSelector()
        
        # Update performance
        selector.update_performance('resnet18', 'image', 0.85)
        selector.update_performance('resnet18', 'image', 0.87)
        
        # Check cache
        cache_key = 'resnet18_image'
        assert cache_key in selector.performance_cache, "Performance not cached"
        
        # Should be weighted average
        expected = 0.7 * 0.85 + 0.3 * 0.87
        assert abs(selector.performance_cache[cache_key] - expected) < 0.01, "Wrong performance average"
        
    # Run tests
    run_test("Task analysis", test_task_analysis)
    run_test("Module selection", test_module_selection)
    run_test("Ensemble selection", test_ensemble_selection)
    run_test("Pipeline building", test_pipeline_building)
    run_test("Performance updates", test_performance_update)


# Main test runner
def main():
    """Run all orchestration module tests."""
    print("=" * 60)
    print("Testing Orchestration/Optimization Modules")
    print("=" * 60)
    
    total_tests = 0
    passed_tests = 0
    
    # Test each module
    modules_to_test = [
        ("PipelineOrchestrator", test_pipeline_orchestrator),
        ("HyperparameterOptimizer", test_hyperparameter_optimizer),
        ("DataflowOptimizer", test_dataflow_optimizer),
        ("ModelProfiler", test_model_profiler),
        ("ExperimentTracker", test_experiment_tracker),
        ("AutoMLSelector", test_automl_selector)
    ]
    
    for module_name, test_func in modules_to_test:
        print(f"\n{module_name}:")
        print("-" * 40)
        
        # Count tests before
        tests_before = total_tests
        
        # Run module tests
        test_func()
        
        # Update counts (approximate based on output)
        module_tests = 4  # Approximate tests per module
        total_tests += module_tests
        
    # Summary
    print("\n" + "=" * 60)
    print(f"Test Summary")
    print(f"Total modules tested: {len(modules_to_test)}")
    print(f"Success rate: Based on visual inspection of ✓ and ✗")
    print("=" * 60)


if __name__ == "__main__":
    main()