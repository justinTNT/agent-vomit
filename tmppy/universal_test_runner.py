#!/usr/bin/env python3
"""
Universal Test Runner - Executes standardized tests from JSON configs

This is the core component that enables running the same tests 
against both Python and F# implementations.
"""

import json
import torch
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from abc import ABC, abstractmethod
import traceback
from collections import defaultdict

# Import existing test utilities
from test_utils import (
    init_with_variations,
    extract_output, 
    validate_shape_behavior,
    test_mask_support,
    check_gradient_flow,
    find_method,
    test_callable_interface,
    PARAMETER_VARIATIONS
)

from test_utils_v2 import (
    init_with_combinations,
    test_with_smart_init,
    flexible_module_test_v2
)

@dataclass
class TestResult:
    """Enhanced test result with cross-language comparison data"""
    name: str
    module_name: str  
    class_name: str
    test_type: str
    language: str  # 'python' or 'fsharp'
    passed: bool
    error_message: Optional[str] = None
    execution_time: float = 0.0
    memory_usage: Optional[float] = None
    output_shape: Optional[List[int]] = None
    output_dtype: Optional[str] = None
    gradient_norm: Optional[float] = None
    numerical_summary: Optional[Dict[str, float]] = None  # For cross-validation

@dataclass  
class ComparisonResult:
    """Result of comparing Python vs F# test results"""
    test_name: str
    both_passed: bool
    python_result: TestResult
    fsharp_result: TestResult
    numerical_difference: Optional[float] = None
    shape_matches: bool = True
    within_tolerance: bool = True
    notes: str = ""

class LanguageTestInterface(ABC):
    """Abstract interface for language-specific test implementations"""
    
    @abstractmethod
    def initialize_module(self, module_name: str, class_name: str, init_params: Dict[str, Any]) -> Any:
        """Initialize module from standardized config"""
        pass
    
    @abstractmethod
    def execute_forward_test(self, module: Any, test_config: Dict[str, Any]) -> TestResult:
        """Execute forward pass test"""
        pass
    
    @abstractmethod
    def execute_shape_test(self, module: Any, test_config: Dict[str, Any]) -> TestResult:
        """Execute shape behavior test"""  
        pass
    
    @abstractmethod
    def execute_gradient_test(self, module: Any, test_config: Dict[str, Any]) -> TestResult:
        """Execute gradient flow test"""
        pass
    
    @abstractmethod
    def execute_method_test(self, module: Any, test_config: Dict[str, Any]) -> TestResult:
        """Execute method existence test"""
        pass
    
    @abstractmethod
    def execute_callable_test(self, module: Any, test_config: Dict[str, Any]) -> TestResult:
        """Execute callable interface test"""
        pass

class EnhancedPythonTestImplementation(LanguageTestInterface):
    """Enhanced Python implementation using existing test infrastructure"""
    
    def __init__(self):
        self.language = "python"
        self.module_cache = {}
    
    def initialize_module(self, module_name: str, class_name: str, init_params: Dict[str, Any]) -> Any:
        """Initialize Python module using enhanced test utilities"""
        cache_key = f"{module_name}.{class_name}"
        
        if cache_key in self.module_cache:
            return self.module_cache[cache_key]
        
        try:
            # Import the module
            if module_name.startswith('modules.'):
                module_path = module_name
            else:
                module_path = f"modules.{module_name}"
            
            module = __import__(module_path, fromlist=[class_name])
            
            # Handle functions vs classes
            target = getattr(module, class_name)
            if callable(target) and not hasattr(target, '__init__'):
                # It's a function, return directly
                instance = target
            else:
                # It's a class, initialize with parameter variations
                try:
                    instance = init_with_variations(target, init_params)
                except Exception:
                    # Try parameter combinations if variations fail
                    instance = init_with_combinations(target, init_params)
            
            self.module_cache[cache_key] = instance
            return instance
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize {module_name}.{class_name}: {str(e)}")
    
    def execute_forward_test(self, module: Any, test_config: Dict[str, Any]) -> TestResult:
        """Execute forward pass test with comprehensive metrics"""
        start_time = time.time()
        result = TestResult(
            name=test_config['name'],
            module_name=test_config.get('module_name', 'unknown'),
            class_name=test_config.get('class_name', 'unknown'),
            test_type='forward',
            language=self.language,
            passed=False
        )
        
        try:
            # Prepare inputs
            inputs = self._prepare_test_inputs(test_config)
            
            # Execute forward pass
            if len(inputs) == 1:
                output = module(inputs[0])
            else:
                output = module(*inputs)
            
            # Extract output information
            result.output_shape = list(output.shape) if hasattr(output, 'shape') else None
            result.output_dtype = str(output.dtype) if hasattr(output, 'dtype') else None
            
            # Compute numerical summary for cross-validation
            if hasattr(output, 'detach'):
                output_np = output.detach().cpu().numpy()
                result.numerical_summary = {
                    'mean': float(output_np.mean()),
                    'std': float(output_np.std()),
                    'min': float(output_np.min()),
                    'max': float(output_np.max()),
                    'norm': float((output_np ** 2).sum() ** 0.5)
                }
            
            result.passed = True
            
        except Exception as e:
            result.error_message = str(e)
        
        result.execution_time = time.time() - start_time
        return result
    
    def execute_shape_test(self, module: Any, test_config: Dict[str, Any]) -> TestResult:
        """Execute shape behavior test using existing utilities"""
        start_time = time.time()
        result = TestResult(
            name=test_config['name'],
            module_name=test_config.get('module_name', 'unknown'),
            class_name=test_config.get('class_name', 'unknown'),
            test_type='shape',
            language=self.language,
            passed=False
        )
        
        try:
            # Get input tensor
            input_tensor = self._prepare_test_inputs(test_config)[0]
            input_shape = list(input_tensor.shape)
            
            # Execute forward pass
            output = module(input_tensor)
            output_shape = list(output.shape) if hasattr(output, 'shape') else None
            
            result.output_shape = output_shape
            
            # Validate shape behavior using existing utility
            expected_behavior = test_config.get('expected_shape', 'preserve')
            success, message = validate_shape_behavior(module, input_tensor, expected_behavior)
            
            result.passed = success
            if not success:
                result.error_message = message
                
        except Exception as e:
            result.error_message = str(e)
        
        result.execution_time = time.time() - start_time
        return result
    
    def execute_gradient_test(self, module: Any, test_config: Dict[str, Any]) -> TestResult:
        """Execute gradient flow test using existing utilities"""
        start_time = time.time()
        result = TestResult(
            name=test_config['name'],
            module_name=test_config.get('module_name', 'unknown'),
            class_name=test_config.get('class_name', 'unknown'),
            test_type='gradient',
            language=self.language,
            passed=False
        )
        
        try:
            # Get input tensor with gradients enabled
            input_tensor = self._prepare_test_inputs(test_config)[0]
            if not input_tensor.requires_grad:
                input_tensor = input_tensor.requires_grad_(True)
            
            # Check gradient flow using existing utility
            grad_norm = check_gradient_flow(module, input_tensor)
            result.gradient_norm = grad_norm
            
            # Pass if gradients are flowing (non-zero norm)
            result.passed = grad_norm > 1e-8
            if not result.passed:
                result.error_message = f"Gradient norm too small: {grad_norm}"
                
        except Exception as e:
            result.error_message = str(e)
        
        result.execution_time = time.time() - start_time
        return result
    
    def execute_method_test(self, module: Any, test_config: Dict[str, Any]) -> TestResult:
        """Execute method existence test using existing utilities"""
        start_time = time.time()
        result = TestResult(
            name=test_config['name'],
            module_name=test_config.get('module_name', 'unknown'),
            class_name=test_config.get('class_name', 'unknown'),
            test_type='method',
            language=self.language,
            passed=False
        )
        
        try:
            method_name = test_config.get('method_name', test_config.get('method'))
            if not method_name:
                result.error_message = "No method name specified"
                return result
            
            # Find method using existing utility
            method = find_method(module, method_name)
            if method is None:
                result.error_message = f"Method '{method_name}' not found"
            else:
                result.passed = True
                
        except Exception as e:
            result.error_message = str(e)
        
        result.execution_time = time.time() - start_time
        return result
    
    def execute_callable_test(self, module: Any, test_config: Dict[str, Any]) -> TestResult:
        """Execute callable interface test using existing utilities"""
        start_time = time.time()
        result = TestResult(
            name=test_config['name'],
            module_name=test_config.get('module_name', 'unknown'),
            class_name=test_config.get('class_name', 'unknown'),
            test_type='callable',
            language=self.language,
            passed=False
        )
        
        try:
            # Test callable interface using existing utility
            inputs = self._prepare_test_inputs(test_config)
            success, message = test_callable_interface(module, inputs)
            
            result.passed = success
            if not success:
                result.error_message = message
                
        except Exception as e:
            result.error_message = str(e)
        
        result.execution_time = time.time() - start_time
        return result
    
    def _prepare_test_inputs(self, test_config: Dict[str, Any]) -> List[torch.Tensor]:
        """Prepare test inputs from config"""
        inputs = []
        
        # Handle args
        if 'args' in test_config and test_config['args']:
            for arg in test_config['args']:
                if isinstance(arg, dict) and arg.get('type') == 'tensor':
                    # Reconstruct tensor from metadata
                    shape = arg['shape']
                    dtype = getattr(torch, arg.get('dtype', 'float32').split('.')[-1])
                    requires_grad = arg.get('requires_grad', False)
                    
                    tensor = torch.randn(shape, dtype=dtype, requires_grad=requires_grad)
                    inputs.append(tensor)
                else:
                    inputs.append(arg)
        
        # Handle input_tensor
        if 'input_tensor' in test_config and test_config['input_tensor']:
            tensor_config = test_config['input_tensor']
            if isinstance(tensor_config, dict) and tensor_config.get('type') == 'tensor':
                shape = tensor_config['shape']
                dtype = getattr(torch, tensor_config.get('dtype', 'float32').split('.')[-1])
                requires_grad = tensor_config.get('requires_grad', False)
                
                tensor = torch.randn(shape, dtype=dtype, requires_grad=requires_grad)
                inputs.append(tensor)
        
        # Default input if none specified
        if not inputs:
            inputs = [torch.randn(2, 10)]
        
        return inputs

class UniversalTestRunner:
    """Universal test runner that executes standardized tests"""
    
    def __init__(self, implementation: LanguageTestInterface):
        self.implementation = implementation
        self.results = []
    
    def run_from_json_config(self, config_path: str) -> List[TestResult]:
        """Run tests from standardized JSON configuration"""
        
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        configs = data['modules']
        all_results = []
        
        print(f"Running tests for {len(configs)} modules using {self.implementation.language} implementation...")
        print("=" * 80)
        
        for module_config in configs:
            module_results = self.run_module_tests(module_config)
            all_results.extend(module_results)
        
        self.results = all_results
        return all_results
    
    def run_module_tests(self, module_config: Dict[str, Any]) -> List[TestResult]:
        """Run all tests for a single module"""
        
        module_name = module_config['module_name']
        class_name = module_config['class_name']
        init_params = module_config.get('init_params', {})
        tests = module_config.get('tests', [])
        
        results = []
        
        try:
            # Initialize module
            module = self.implementation.initialize_module(module_name, class_name, init_params)
            
            print(f"\n📦 {module_name}.{class_name}")
            print("-" * 40)
            
            # Run each test
            for test_config in tests:
                test_config['module_name'] = module_name
                test_config['class_name'] = class_name
                
                result = self.run_single_test(module, test_config)
                results.append(result)
                
                # Print result
                status = "✅ PASS" if result.passed else "❌ FAIL"
                print(f"  {status} | {result.name} ({result.test_type})")
                
                if result.error_message:
                    print(f"      Error: {result.error_message}")
                if result.output_shape:
                    print(f"      Shape: {result.output_shape}")
                if result.execution_time > 0:
                    print(f"      Time: {result.execution_time:.3f}s")
                if result.gradient_norm:
                    print(f"      Gradient norm: {result.gradient_norm:.6f}")
        
        except Exception as e:
            # If module initialization fails, mark all tests as failed
            for test_config in tests:
                failed_result = TestResult(
                    name=test_config['name'],
                    module_name=module_name,
                    class_name=class_name,
                    test_type=test_config['test_type'],
                    language=self.implementation.language,
                    passed=False,
                    error_message=f"Module initialization failed: {str(e)}"
                )
                results.append(failed_result)
                print(f"  ❌ FAIL | {failed_result.name} (init failure)")
        
        return results
    
    def run_single_test(self, module: Any, test_config: Dict[str, Any]) -> TestResult:
        """Run a single test"""
        
        test_type = test_config['test_type']
        
        if test_type == 'forward':
            return self.implementation.execute_forward_test(module, test_config)
        elif test_type == 'shape':
            return self.implementation.execute_shape_test(module, test_config)
        elif test_type == 'gradient':
            return self.implementation.execute_gradient_test(module, test_config)
        elif test_type == 'method':
            return self.implementation.execute_method_test(module, test_config)
        elif test_type == 'callable':
            return self.implementation.execute_callable_test(module, test_config)
        else:
            return TestResult(
                name=test_config['name'],
                module_name=test_config.get('module_name', 'unknown'),
                class_name=test_config.get('class_name', 'unknown'),
                test_type=test_type,
                language=self.implementation.language,
                passed=False,
                error_message=f"Unknown test type: {test_type}"
            )
    
    def print_summary(self):
        """Print test summary"""
        if not self.results:
            print("No test results available")
            return
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        
        print("\n" + "=" * 80)
        print(f"UNIVERSAL TEST RUNNER SUMMARY ({self.implementation.language.upper()})")
        print("=" * 80)
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success rate: {100 * passed_tests / total_tests:.1f}%")
        
        # Group failures by module
        failures = [r for r in self.results if not r.passed]
        if failures:
            print(f"\nFailed tests ({len(failures)}):")
            by_module = defaultdict(list)
            for failure in failures:
                by_module[f"{failure.module_name}.{failure.class_name}"].append(failure)
            
            for module, module_failures in by_module.items():
                print(f"  📦 {module}: {len(module_failures)} failures")
                for failure in module_failures[:3]:  # Show first 3
                    print(f"    ❌ {failure.name}: {failure.error_message}")
                if len(module_failures) > 3:
                    print(f"    ... and {len(module_failures) - 3} more")

if __name__ == "__main__":
    # Test the Universal Test Runner with Python implementation
    print("Testing Universal Test Runner with Python implementation...")
    
    implementation = EnhancedPythonTestImplementation()
    runner = UniversalTestRunner(implementation)
    
    # Run tests from standardized config
    config_path = "standardized_tests/all_modules_standardized.json"
    if Path(config_path).exists():
        results = runner.run_from_json_config(config_path)
        runner.print_summary()
        
        # Save results for cross-language comparison
        results_path = "test_results/python_universal_test_results.json"
        Path("test_results").mkdir(exist_ok=True)
        
        with open(results_path, 'w') as f:
            serializable_results = [asdict(r) for r in results]
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n📁 Results saved to: {results_path}")
        print("🎯 This framework is now ready for F# wrapper implementation!")
    else:
        print(f"❌ Config file not found: {config_path}")
        print("   Run: python standardize_existing_tests.py first")