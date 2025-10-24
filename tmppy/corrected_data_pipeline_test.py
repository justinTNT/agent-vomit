#!/usr/bin/env python3
"""
Corrected test for data pipeline modules with proper parameter names
"""

import torch
import time
import json
from typing import Dict, List, Any
import traceback

def test_module_directly(module_class, init_params: Dict, test_inputs: List) -> Dict[str, Any]:
    """Test a module directly with given parameters and inputs"""
    try:
        # Initialize module
        module = module_class(**init_params)
        module.eval()
        
        results = []
        
        for i, test_input in enumerate(test_inputs):
            test_name = f"test_{i+1}"
            
            try:
                start_time = time.time()
                
                # Forward pass - handle different input types
                if hasattr(test_input, 'items'):  # Dictionary (method call)
                    method_name = test_input.get('method')
                    args = test_input.get('args', [])
                    if method_name:
                        method = getattr(module, method_name)
                        output = method(*args)
                    else:
                        output = module(**test_input)
                elif isinstance(test_input, tuple):
                    output = module(*test_input)
                elif isinstance(test_input, torch.Tensor):
                    output = module(test_input)
                else:
                    output = module(test_input)
                
                execution_time = time.time() - start_time
                
                # Extract output info
                if isinstance(output, torch.Tensor):
                    output_shape = list(output.shape)
                    output_dtype = str(output.dtype)
                    numerical_summary = {
                        "mean": float(output.mean()),
                        "std": float(output.std()),
                        "min": float(output.min()),
                        "max": float(output.max()),
                        "norm": float(output.norm())
                    }
                elif isinstance(output, str):
                    output_shape = "string"
                    output_dtype = "string"
                    numerical_summary = f"string_length_{len(output)}"
                elif output is None:
                    output_shape = "none"
                    output_dtype = "none"
                    numerical_summary = "none"
                else:
                    output_shape = str(type(output))
                    output_dtype = str(type(output))
                    numerical_summary = str(output)[:100]
                
                results.append({
                    "test_name": test_name,
                    "passed": True,
                    "execution_time": execution_time,
                    "output_shape": output_shape,
                    "output_dtype": output_dtype,
                    "numerical_summary": numerical_summary,
                    "error": None
                })
                
            except Exception as e:
                results.append({
                    "test_name": test_name,
                    "passed": False,
                    "execution_time": 0,
                    "output_shape": None,
                    "output_dtype": None,
                    "numerical_summary": None,
                    "error": str(e)
                })
        
        return {
            "module_passed": all(r["passed"] for r in results),
            "results": results,
            "init_error": None
        }
        
    except Exception as e:
        return {
            "module_passed": False,
            "results": [],
            "init_error": str(e)
        }

def main():
    """Test data pipeline modules with corrected parameters"""
    
    print("🔧 Corrected Data Pipeline Module Tests")
    print("=" * 60)
    print()
    
    # Corrected test configurations based on actual module code
    test_configs = [
        {
            "name": "DataVersioner",
            "module_path": "modules.data_versioner",
            "class_name": "DataVersioner",
            "init_params": {
                "storage_path": "/tmp/test_versions",
                "chunk_size": 1000,
                "track_deltas": True,
                "compression": False
            },
            "test_inputs": [
                # Test forward method (commit data)
                (torch.randn(2, 10, 512), "test commit"),
                # Test load method
                {"method": "load", "args": ["test_version_id"]},
                # Test diff method  
                {"method": "diff", "args": ["version1", "version2"]},
                # Test branch method
                {"method": "branch", "args": ["test_branch"]},
                # Test get_statistics method
                {"method": "get_statistics", "args": []}
            ]
        },
        {
            "name": "StreamProcessor", 
            "module_path": "modules.stream_processor",
            "class_name": "StreamProcessor",
            "init_params": {
                "window_type": "tumbling",
                "window_size": 1000,
                "aggregation": "mean",
                "buffer_size": 10000
            },
            "test_inputs": [
                # Test forward method (process stream element)
                torch.randn(2, 512),
                # Test flush method
                {"method": "flush", "args": []},
                # Test get_metrics method
                {"method": "get_metrics", "args": []}
            ]
        },
        {
            "name": "FeatureStore",
            "module_path": "modules.feature_store", 
            "class_name": "FeatureStore",
            "init_params": {
                "storage_path": "/tmp/test_features",
                "cache_size": 1000,
                "enable_versioning": True,
                "enable_lineage": True
            },
            "test_inputs": [
                # Test forward method (get features)
                {"method": "forward", "args": [["dummy_feature"], {"input": torch.randn(2, 10)}]},
                # Test get_statistics method
                {"method": "get_statistics", "args": []},
                # Test clear_cache method
                {"method": "clear_cache", "args": []}
            ]
        }
    ]
    
    all_results = []
    total_modules = len(test_configs)
    passed_modules = 0
    total_tests = 0
    passed_tests = 0
    
    for config in test_configs:
        print(f"📦 Testing {config['name']}")
        print("-" * 50)
        
        try:
            # Import module
            module_path = config["module_path"]
            class_name = config["class_name"]
            
            # Dynamic import
            module = __import__(module_path, fromlist=[class_name])
            module_class = getattr(module, class_name)
            
            # Test module
            result = test_module_directly(
                module_class, 
                config["init_params"], 
                config["test_inputs"]
            )
            
            if result["init_error"]:
                print(f"❌ INIT FAILED: {result['init_error']}")
            else:
                test_results = result["results"]
                module_passed_tests = sum(1 for r in test_results if r["passed"])
                module_total_tests = len(test_results)
                
                total_tests += module_total_tests
                passed_tests += module_passed_tests
                
                if result["module_passed"]:
                    passed_modules += 1
                    print(f"✅ ALL TESTS PASSED ({module_passed_tests}/{module_total_tests})")
                else:
                    print(f"⚠️  SOME TESTS PASSED ({module_passed_tests}/{module_total_tests})")
                
                # Show individual test results
                for test_result in test_results:
                    status = "✅" if test_result["passed"] else "❌"
                    print(f"  {status} {test_result['test_name']}: ", end="")
                    
                    if test_result["passed"]:
                        print(f"Time: {test_result['execution_time']:.3f}s, Output: {test_result['output_shape']}")
                        if isinstance(test_result["numerical_summary"], dict):
                            ns = test_result["numerical_summary"]
                            print(f"    Stats: mean={ns['mean']:.6f}, std={ns['std']:.6f}, norm={ns['norm']:.6f}")
                        elif test_result["numerical_summary"]:
                            print(f"    Result: {test_result['numerical_summary']}")
                    else:
                        print(f"Error: {test_result['error']}")
            
            all_results.append({
                "module": config["name"],
                "result": result
            })
            
        except Exception as e:
            print(f"❌ IMPORT/SETUP FAILED: {str(e)}")
            all_results.append({
                "module": config["name"],
                "result": {"module_passed": False, "results": [], "init_error": str(e)}
            })
        
        print()
    
    # Summary
    print("=" * 60)
    print("DATA PIPELINE TEST SUMMARY")
    print("=" * 60)
    print(f"Modules tested: {total_modules}")
    print(f"Modules fully working: {passed_modules}")
    print(f"Module success rate: {passed_modules/total_modules*100:.1f}%")
    print()
    print(f"Total individual tests: {total_tests}")
    print(f"Tests passed: {passed_tests}")
    print(f"Test success rate: {passed_tests/total_tests*100:.1f}%")
    print()
    
    if passed_modules / total_modules >= 0.8:
        print("🎉 EXCELLENT: 80%+ data pipeline modules working!")
        print("✅ Data pipeline infrastructure is solid!")
    elif passed_modules / total_modules >= 0.6:
        print("✅ GOOD: 60%+ data pipeline modules working!")
        print("🔧 Minor fixes needed for remaining modules")
    else:
        print("⚠️  Data pipeline modules need attention")
        print("🛠️  Focus on parameter mapping and method interfaces")
    
    # Save results
    with open("data_pipeline_corrected_results.json", "w") as f:
        json.dump({
            "metadata": {
                "test_type": "corrected_data_pipeline_modules",
                "total_modules": total_modules,
                "passed_modules": passed_modules,
                "module_success_rate": passed_modules/total_modules*100,
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "test_success_rate": passed_tests/total_tests*100 if total_tests > 0 else 0
            },
            "all_results": all_results
        }, f, indent=2)
    
    print(f"📁 Results saved to: data_pipeline_corrected_results.json")
    
    # Analysis of issues found
    failed_modules = [r for r in all_results if not r["result"]["module_passed"]]
    if failed_modules:
        print("\n" + "=" * 60)
        print("FAILURE ANALYSIS")
        print("=" * 60)
        for result in failed_modules:
            module_name = result["module"]
            module_result = result["result"]
            
            print(f"\n📋 {module_name}:")
            if module_result["init_error"]:
                print(f"  Init Error: {module_result['init_error']}")
            else:
                for test in module_result["results"]:
                    if not test["passed"]:
                        print(f"  ❌ {test['test_name']}: {test['error']}")

if __name__ == "__main__":
    main()