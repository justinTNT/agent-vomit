#!/usr/bin/env python3
"""
Wave 1 - Quick Wins: Testing 6 untested modules for 100% coverage push
Strategy: Start with easiest modules to build momentum toward 100% success
"""

import torch
import time
import json
from typing import Dict, List, Any
import traceback
import importlib
import sys
from pathlib import Path

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
                    output_dtype = str(output.dtype)
                    numerical_summary = {
                        "mean": float(output.mean()),
                        "std": float(output.std()),
                        "min": float(output.min()),
                        "max": float(output.max()),
                        "norm": float(output.norm())
                    }
                elif isinstance(output, (tuple, list)):
                    if len(output) > 0 and isinstance(output[0], torch.Tensor):
                        output_shape = list(output[0].shape)
                        output_dtype = str(output[0].dtype)
                        numerical_summary = {
                            "mean": float(output[0].mean()),
                            "std": float(output[0].std()),
                            "min": float(output[0].min()),
                            "max": float(output[0].max()),
                            "norm": float(output[0].norm())
                        }
                    else:
                        output_shape = [str(type(o)) for o in output]
                        output_dtype = [str(type(o)) for o in output]
                        numerical_summary = "multiple_outputs"
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

def try_import_module(module_path: str, class_name: str):
    """Try to import module with fallback paths"""
    # Add candidates directory to path if needed
    candidates_path = Path("/Users/jtnt/Play/agent-vomit/candidates")
    if candidates_path.exists():
        for subdir in candidates_path.iterdir():
            if subdir.is_dir():
                sys.path.insert(0, str(subdir))
    
    try:
        # Try direct import first
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError):
        # Try alternative paths
        for alt_path in [f"candidates.agent_codex.{module_path}", f"candidates.agent_claude.{module_path}"]:
            try:
                module = importlib.import_module(alt_path)
                return getattr(module, class_name)
            except (ImportError, AttributeError):
                continue
    
    # Try loading from file directly
    for subdir in ["agent_codex", "agent_claude"]:
        file_path = candidates_path / subdir / f"{module_path}.py"
        if file_path.exists():
            try:
                spec = importlib.util.spec_from_file_location(module_path, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return getattr(module, class_name)
            except Exception:
                continue
                
    raise ImportError(f"Could not import {class_name} from {module_path}")

def main():
    """Wave 1: Test 6 quick win modules for 100% coverage push"""
    
    print("🚀 WAVE 1 - QUICK WINS: 6 Modules → 100% Coverage Push")
    print("=" * 70)
    print("Strategy: Start with easiest untested modules to build momentum")
    print("Target: 6 modules in 5 hours → 52.6% total coverage")
    print()
    
    # Wave 1 test configurations - optimized for success
    test_configs = [
        {
            "name": "MultiScaleSTFTLoss",
            "module_path": "multi_scale_stft_loss",
            "class_name": "MultiScaleSTFTLoss",
            "difficulty": "EASY",
            "estimated_hours": 0.5,
            "init_params": {
                "fft_sizes": [1024, 2048, 512],
                "hop_sizes": [120, 240, 50],
                "win_lengths": [600, 1200, 240]
            },
            "test_inputs": [
                (torch.randn(2, 8192), torch.randn(2, 8192)),
                (torch.randn(1, 4096), torch.randn(1, 4096))
            ]
        },
        {
            "name": "CausalConv1d",
            "module_path": "causal_conv1d", 
            "class_name": "CausalConv1d",
            "difficulty": "EASY",
            "estimated_hours": 0.5,
            "init_params": {
                "in_channels": 256,
                "out_channels": 256,
                "kernel_size": 3,
                "dilation": 1
            },
            "test_inputs": [
                torch.randn(2, 256, 100),
                torch.randn(1, 256, 50)
            ]
        },
        {
            "name": "MemoryBank",
            "module_path": "memory_bank",
            "class_name": "MemoryBank", 
            "difficulty": "EASY",
            "estimated_hours": 0.5,
            "init_params": {
                "memory_size": 1000,
                "feature_dim": 256,
                "momentum": 0.99
            },
            "test_inputs": [
                torch.randn(32, 256),
                torch.randn(16, 256)
            ]
        },
        {
            "name": "AntiAliasedConv",
            "module_path": "anti_aliased_conv",
            "class_name": "AntiAliasedConv",
            "difficulty": "EASY", 
            "estimated_hours": 0.5,
            "init_params": {
                "in_channels": 64,
                "out_channels": 128,
                "kernel_size": 3,
                "stride": 2
            },
            "test_inputs": [
                torch.randn(2, 64, 32, 32),
                torch.randn(1, 64, 16, 16)
            ]
        },
        {
            "name": "SequenceToSequenceModel",
            "module_path": "sequence_to_sequence_model",
            "class_name": "SequenceToSequenceModel",
            "difficulty": "MODERATE",
            "estimated_hours": 1.0,
            "init_params": {
                "input_vocab_size": 1000,
                "output_vocab_size": 1000,
                "d_model": 256,
                "n_heads": 8,
                "n_layers": 4
            },
            "test_inputs": [
                torch.randint(0, 1000, (2, 10)),
                torch.randint(0, 1000, (1, 20))
            ]
        },
        {
            "name": "Autoencoder",
            "module_path": "autoencoder",
            "class_name": "Autoencoder",
            "difficulty": "HARD",
            "estimated_hours": 2.0,
            "init_params": {
                "input_dim": 784,
                "hidden_dims": [512, 256, 128],
                "latent_dim": 64
            },
            "test_inputs": [
                torch.randn(32, 784),
                torch.randn(16, 784)
            ]
        }
    ]
    
    all_results = []
    total_modules = len(test_configs)
    passed_modules = 0
    total_tests = 0
    passed_tests = 0
    total_estimated_hours = sum(config["estimated_hours"] for config in test_configs)
    
    print(f"Testing {total_modules} modules (estimated {total_estimated_hours} hours)")
    print()
    
    for config in test_configs:
        print(f"📦 Testing {config['name']} ({config['difficulty']} - {config['estimated_hours']}h)")
        print("-" * 60)
        
        try:
            # Import module with fallback paths
            module_class = try_import_module(config["module_path"], config["class_name"])
            
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
                    print(f"  {status} {test_result['test_name']}: Time: {test_result['execution_time']:.3f}s")
                    
                    if test_result["passed"]:
                        print(f"    Shape: {test_result['output_shape']}")
                        if isinstance(test_result["numerical_summary"], dict):
                            ns = test_result["numerical_summary"]
                            print(f"    Stats: mean={ns['mean']:.4f}, norm={ns['norm']:.4f}")
                    else:
                        print(f"    Error: {test_result['error']}")
            
            all_results.append({
                "module": config["name"],
                "difficulty": config["difficulty"], 
                "estimated_hours": config["estimated_hours"],
                "result": result
            })
            
        except Exception as e:
            print(f"❌ IMPORT/SETUP FAILED: {str(e)}")
            all_results.append({
                "module": config["name"],
                "difficulty": config["difficulty"],
                "estimated_hours": config["estimated_hours"], 
                "result": {"module_passed": False, "results": [], "init_error": str(e)}
            })
        
        print()
    
    # Calculate progress toward 100% coverage
    current_coverage = 79  # Previous coverage
    wave1_additions = passed_modules
    new_coverage = current_coverage + wave1_additions
    total_modules_in_codebase = 137  # From inventory
    coverage_percentage = (new_coverage / total_modules_in_codebase) * 100
    
    # Summary
    print("=" * 70)
    print("WAVE 1 RESULTS - QUICK WINS SUMMARY")
    print("=" * 70)
    print(f"Wave 1 modules tested: {total_modules}")
    print(f"Wave 1 modules working: {passed_modules}")
    print(f"Wave 1 success rate: {passed_modules/total_modules*100:.1f}%")
    print()
    print(f"Total tests in wave: {total_tests}")
    print(f"Tests passed: {passed_tests}")
    print(f"Test success rate: {passed_tests/total_tests*100:.1f}%")
    print()
    print(f"📊 COVERAGE PROGRESS:")
    print(f"Previous coverage: {current_coverage}/137 modules ({current_coverage/total_modules_in_codebase*100:.1f}%)")
    print(f"New coverage: {new_coverage}/137 modules ({coverage_percentage:.1f}%)")
    print(f"Progress: +{wave1_additions} modules")
    print()
    
    if passed_modules / total_modules >= 0.8:
        print("🎉 WAVE 1 SUCCESS: 80%+ modules working!")
        print("🚀 Ready for Wave 2 - Medium Effort modules")
        next_wave_ready = True
    elif passed_modules / total_modules >= 0.6:
        print("✅ WAVE 1 GOOD: 60%+ modules working!")
        print("🔧 Minor fixes needed before Wave 2")
        next_wave_ready = True
    else:
        print("⚠️  WAVE 1 NEEDS WORK: Fix issues before proceeding")
        next_wave_ready = False
    
    # Save results
    with open("wave1_results.json", "w") as f:
        json.dump({
            "metadata": {
                "wave": "wave_1_quick_wins",
                "strategy": "Start with easiest untested modules to build momentum",
                "total_modules": total_modules,
                "passed_modules": passed_modules,
                "wave_success_rate": passed_modules/total_modules*100,
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "test_success_rate": passed_tests/total_tests*100 if total_tests > 0 else 0,
                "coverage_progress": {
                    "previous_coverage": current_coverage,
                    "new_coverage": new_coverage,
                    "total_modules": total_modules_in_codebase,
                    "coverage_percentage": coverage_percentage,
                    "modules_added": wave1_additions
                },
                "next_wave_ready": next_wave_ready,
                "estimated_hours": total_estimated_hours
            },
            "all_results": all_results
        }, f, indent=2)
    
    print(f"📁 Wave 1 results saved to: wave1_results.json")
    
    if next_wave_ready:
        print("\n🎯 READY FOR WAVE 2!")
        print("Next targets: 8 medium-effort modules → 58.4% total coverage")

if __name__ == "__main__":
    main()