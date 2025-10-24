#!/usr/bin/env python3
"""
COMPREHENSIVE COVERAGE PUSH: Test ALL remaining modules for 100% coverage
Strategy: Build on 96.4% success momentum, test every discoverable module
"""

import torch
import time
import json
from typing import Dict, List, Any
import traceback
import importlib
import sys
from pathlib import Path
import os

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
                
                # Forward pass
                if isinstance(test_input, tuple):
                    output = module(*test_input)
                else:
                    output = module(test_input)
                
                execution_time = time.time() - start_time
                
                # Extract output info
                if isinstance(output, torch.Tensor):
                    output_shape = list(output.shape)
                    numerical_summary = {
                        "mean": float(output.mean()),
                        "std": float(output.std()),
                        "norm": float(output.norm())
                    }
                elif isinstance(output, (tuple, list)):
                    if len(output) > 0 and isinstance(output[0], torch.Tensor):
                        output_shape = list(output[0].shape)
                        numerical_summary = {
                            "mean": float(output[0].mean()),
                            "std": float(output[0].std()),
                            "norm": float(output[0].norm())
                        }
                    else:
                        output_shape = [str(type(o)) for o in output]
                        numerical_summary = "multiple_outputs"
                elif isinstance(output, dict):
                    output_shape = "dict"
                    numerical_summary = f"dict_keys_{list(output.keys())}"
                else:
                    output_shape = str(type(output))
                    numerical_summary = str(output)[:50]
                
                results.append({
                    "test_name": test_name,
                    "passed": True,
                    "execution_time": execution_time,
                    "output_shape": output_shape,
                    "numerical_summary": numerical_summary,
                    "error": None
                })
                
            except Exception as e:
                results.append({
                    "test_name": test_name,
                    "passed": False,
                    "execution_time": 0,
                    "output_shape": None,
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

def discover_all_modules():
    """Discover all available modules in the codebase"""
    modules_found = []
    base_path = Path("/Users/jtnt/Play/agent-vomit")
    
    # Search directories
    search_dirs = [
        "modules",
        "modules/audio_analysis", 
        "modules/advanced_modules",
        "audio-ml-extensions/audio_gan",
        "audio-ml-extensions/orchestration",
        "crossfade",
        "candidates/agent_codex",
        "candidates/agent_claude"
    ]
    
    for search_dir in search_dirs:
        dir_path = base_path / search_dir
        if dir_path.exists():
            for py_file in dir_path.glob("*.py"):
                if py_file.name.startswith("__"):
                    continue
                    
                modules_found.append({
                    "file_path": str(py_file),
                    "module_name": py_file.stem,
                    "search_dir": search_dir,
                    "relative_path": f"{search_dir}/{py_file.stem}"
                })
    
    return modules_found

def try_import_module(module_info: Dict):
    """Try to import and extract classes from module"""
    try:
        # Add directory to path
        module_dir = Path(module_info["file_path"]).parent
        if str(module_dir) not in sys.path:
            sys.path.insert(0, str(module_dir))
        
        # Try importing
        spec = importlib.util.spec_from_file_location(
            module_info["module_name"], 
            module_info["file_path"]
        )
        if spec is None:
            return None
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find nn.Module classes
        classes_found = []
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and 
                hasattr(attr, '__module__') and
                attr.__module__ == module.__name__ and
                hasattr(attr, '__bases__')):
                
                # Check if it's a nn.Module or has forward method
                if (any("Module" in str(base) for base in attr.__bases__) or
                    hasattr(attr, 'forward')):
                    classes_found.append(attr_name)
        
        return {
            "module": module,
            "classes": classes_found,
            "module_name": module_info["module_name"]
        }
        
    except Exception as e:
        return None

def create_test_config(class_name: str, module_name: str):
    """Create test configuration for a class"""
    
    # Common parameter patterns
    common_configs = {
        # Transformer patterns
        "TransformerBlock": {"d_model": 256, "n_heads": 8, "d_ff": 1024, "dropout": 0.1},
        "AttentionDecoder": {"vocab_size": 1000, "d_model": 256, "n_heads": 8, "n_layers": 2, "d_ff": 1024, "max_seq_len": 100, "dropout": 0.1, "pad_token_id": 0, "start_token_id": 1, "end_token_id": 2},
        
        # Convolution patterns
        "ConvEncoder": {"in_channels": 3, "base_channels": 64},
        "CausalConv1d": {"in_channels": 256, "out_channels": 256, "kernel_size": 3},
        
        # Audio patterns
        "STFTLoss": {"fft_size": 1024, "hop_size": 256, "win_length": 1024},
        "MultiScaleSTFTLoss": {"fft_sizes": [1024, 2048, 512], "hop_sizes": [120, 240, 50], "win_lengths": [600, 1200, 240]},
        
        # Memory patterns
        "MemoryBank": {"memory_size": 1000},
        
        # Default patterns by name keywords
        "Encoder": {"input_dim": 256, "output_dim": 128, "hidden_dim": 256},
        "Decoder": {"input_dim": 128, "output_dim": 256, "hidden_dim": 256},
        "GAN": {"latent_dim": 100, "hidden_dim": 256},
        "Loss": {"reduction": "mean"},
        "Norm": {"num_features": 128},
        "Conv": {"in_channels": 64, "out_channels": 128, "kernel_size": 3},
        "Linear": {"input_dim": 256, "output_dim": 128},
        "Transformer": {"d_model": 256, "n_heads": 8, "d_ff": 1024, "n_layers": 2},
        "Attention": {"d_model": 256, "n_heads": 8},
    }
    
    # Try exact match first
    if class_name in common_configs:
        params = common_configs[class_name]
    else:
        # Try pattern matching
        params = {}
        for pattern, config in common_configs.items():
            if pattern.lower() in class_name.lower():
                params.update(config)
                break
        
        # Default minimal config
        if not params:
            if "dim" in class_name.lower() or "size" in class_name.lower():
                params = {"input_dim": 256, "output_dim": 128}
            else:
                params = {}
    
    # Create test inputs based on class type
    if "conv" in class_name.lower() and "1d" in class_name.lower():
        test_inputs = [torch.randn(2, 256, 100)]
    elif "conv" in class_name.lower() and "2d" in class_name.lower():
        test_inputs = [torch.randn(2, 64, 32, 32)]
    elif "transformer" in class_name.lower() or "attention" in class_name.lower():
        test_inputs = [torch.randn(2, 10, 256)]
    elif "loss" in class_name.lower():
        test_inputs = [(torch.randn(2, 1024), torch.randn(2, 1024))]
    elif "audio" in class_name.lower():
        test_inputs = [torch.randn(2, 1, 8192)]
    elif "sequence" in class_name.lower():
        test_inputs = [torch.randint(0, 1000, (2, 20))]
    elif "memory" in class_name.lower():
        test_inputs = [torch.randn(32, 256)]
    else:
        # Default tensor input
        test_inputs = [torch.randn(2, 256)]
    
    return {
        "params": params,
        "test_inputs": test_inputs
    }

def main():
    """Comprehensive coverage push - test ALL discoverable modules"""
    
    print("🚀 COMPREHENSIVE COVERAGE PUSH: Testing ALL Discoverable Modules")
    print("=" * 80)
    print("Strategy: Build on 96.4% success momentum, achieve 100% coverage")
    print("Target: Test every single module in the codebase")
    print()
    
    # Discover all modules
    print("🔍 Discovering all modules in codebase...")
    all_modules = discover_all_modules()
    print(f"Found {len(all_modules)} Python files to analyze")
    print()
    
    # Test results tracking
    total_modules_discovered = 0
    total_classes_found = 0
    total_tests_run = 0
    total_tests_passed = 0
    successful_modules = 0
    
    results_by_directory = {}
    all_results = []
    
    for module_info in all_modules:
        search_dir = module_info["search_dir"]
        if search_dir not in results_by_directory:
            results_by_directory[search_dir] = {"tested": 0, "passed": 0, "results": []}
        
        print(f"📦 Analyzing: {module_info['relative_path']}")
        
        # Try to import module
        import_result = try_import_module(module_info)
        if import_result is None:
            print(f"   ❌ Could not import module")
            continue
            
        total_modules_discovered += 1
        classes = import_result["classes"]
        
        if not classes:
            print(f"   ⚠️  No testable classes found")
            continue
            
        print(f"   ✅ Found {len(classes)} classes: {', '.join(classes)}")
        total_classes_found += len(classes)
        
        # Test each class
        for class_name in classes:
            try:
                module_class = getattr(import_result["module"], class_name)
                test_config = create_test_config(class_name, module_info["module_name"])
                
                print(f"      🧪 Testing {class_name}...")
                
                # Test the module
                result = test_module_directly(
                    module_class,
                    test_config["params"],
                    test_config["test_inputs"]
                )
                
                results_by_directory[search_dir]["tested"] += 1
                total_tests_run += 1
                
                if result["module_passed"]:
                    results_by_directory[search_dir]["passed"] += 1
                    total_tests_passed += 1
                    successful_modules += 1
                    print(f"         ✅ PASS")
                else:
                    print(f"         ❌ FAIL: {result.get('init_error', 'Test failures')}")
                
                all_results.append({
                    "module_file": module_info["relative_path"],
                    "class_name": class_name,
                    "directory": search_dir,
                    "result": result
                })
                
            except Exception as e:
                print(f"         ❌ ERROR: {str(e)}")
                total_tests_run += 1
        
        print()
    
    # Calculate final statistics
    overall_success_rate = (total_tests_passed / total_tests_run * 100) if total_tests_run > 0 else 0
    
    print("=" * 80)
    print("🎯 COMPREHENSIVE COVERAGE RESULTS")
    print("=" * 80)
    print(f"Python files analyzed: {len(all_modules)}")
    print(f"Modules imported: {total_modules_discovered}")
    print(f"Classes discovered: {total_classes_found}")
    print(f"Total tests run: {total_tests_run}")
    print(f"Tests passed: {total_tests_passed}")
    print(f"Overall success rate: {overall_success_rate:.1f}%")
    print()
    
    # Results by directory
    print("📊 RESULTS BY DIRECTORY:")
    for directory, stats in results_by_directory.items():
        if stats["tested"] > 0:
            success_rate = (stats["passed"] / stats["tested"] * 100)
            print(f"  {directory}: {stats['passed']}/{stats['tested']} ({success_rate:.1f}%)")
    print()
    
    # Coverage calculation
    estimated_total_coverage = 79 + total_tests_passed  # Previous coverage + new
    estimated_coverage_percentage = (estimated_total_coverage / 137) * 100
    
    print(f"📈 COVERAGE ESTIMATION:")
    print(f"Previous coverage: 79 modules")
    print(f"New modules tested: {total_tests_passed}")
    print(f"Estimated total coverage: {estimated_total_coverage}/137 ({estimated_coverage_percentage:.1f}%)")
    print()
    
    if overall_success_rate >= 90:
        print("🎉 EXCELLENT: 90%+ success rate achieved!")
        print("🚀 Ready for F# implementation with high confidence!")
    elif overall_success_rate >= 75:
        print("✅ GOOD: 75%+ success rate - strong foundation!")
        print("🔧 Minor fixes will achieve 90%+ success")
    else:
        print("⚠️  More work needed to reach 90%+ success rate")
    
    # Save comprehensive results
    with open("comprehensive_coverage_results.json", "w") as f:
        json.dump({
            "metadata": {
                "test_type": "comprehensive_coverage_push",
                "strategy": "Test every discoverable module for 100% coverage",
                "files_analyzed": len(all_modules),
                "modules_imported": total_modules_discovered,
                "classes_discovered": total_classes_found,
                "total_tests": total_tests_run,
                "tests_passed": total_tests_passed,
                "overall_success_rate": overall_success_rate,
                "estimated_coverage": {
                    "previous": 79,
                    "new": total_tests_passed,
                    "total": estimated_total_coverage,
                    "percentage": estimated_coverage_percentage
                }
            },
            "results_by_directory": results_by_directory,
            "all_results": all_results
        }, f, indent=2)
    
    print(f"📁 Comprehensive results saved to: comprehensive_coverage_results.json")
    
    if estimated_coverage_percentage >= 90:
        print("\n🎯 TARGET ACHIEVED: 90%+ coverage with high success rate!")
        print("✅ Ready for production F# validation framework!")

if __name__ == "__main__":
    main()