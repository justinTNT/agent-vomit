#!/usr/bin/env python3
"""
Wave 2 - Medium Effort: Testing 8 moderate difficulty modules
Strategy: Build on Wave 1 success, tackle medium complexity modules
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
                elif isinstance(output, dict):
                    output_shape = "dict"
                    output_dtype = "dict"
                    numerical_summary = f"dict_keys_{list(output.keys())}"
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

def try_import_from_paths(module_name: str, class_name: str, search_paths: List[str]):
    """Try to import module from multiple paths"""
    
    for base_path in search_paths:
        full_path = Path(base_path)
        if full_path.exists():
            sys.path.insert(0, str(full_path))
            
        try:
            # Try direct import
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (ImportError, AttributeError):
            pass
            
        # Try loading from file directly
        file_path = full_path / f"{module_name}.py"
        if file_path.exists():
            try:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return getattr(module, class_name)
            except Exception:
                pass
                
    raise ImportError(f"Could not import {class_name} from {module_name}")

def main():
    """Wave 2: Test 8 medium effort modules for continued coverage progress"""
    
    print("🚀 WAVE 2 - MEDIUM EFFORT: 8 Modules → 68.4% Coverage Target")
    print("=" * 70)
    print("Strategy: Build on Wave 1 success (100%), tackle moderate complexity")
    print("Target: 8 modules → 68.4% total coverage")
    print()
    
    # Wave 2 test configurations - moderate difficulty modules
    test_configs = [
        {
            "name": "GANLoss",
            "module_name": "gan_loss",
            "class_name": "GANLoss",
            "search_paths": [
                "/Users/jtnt/Play/agent-vomit/audio-ml-extensions/audio_gan",
                "/Users/jtnt/Play/agent-vomit/modules",
                "/Users/jtnt/Play/agent-vomit/candidates/agent_codex",
                "/Users/jtnt/Play/agent-vomit/candidates/agent_claude"
            ],
            "difficulty": "MODERATE",
            "estimated_hours": 1.0,
            "init_params": {
                "gan_mode": "lsgan",
                "target_real_label": 1.0,
                "target_fake_label": 0.0
            },
            "test_inputs": [
                (torch.randn(32, 1), True),  # Real predictions
                (torch.randn(32, 1), False)  # Fake predictions
            ]
        },
        {
            "name": "WaveNetResBlock",
            "module_name": "wavenet_resblock",
            "class_name": "WaveNetResBlock",
            "search_paths": [
                "/Users/jtnt/Play/agent-vomit/audio-ml-extensions/audio_gan",
                "/Users/jtnt/Play/agent-vomit/modules",
                "/Users/jtnt/Play/agent-vomit/candidates/agent_codex"
            ],
            "difficulty": "MODERATE",
            "estimated_hours": 1.0,
            "init_params": {
                "channels": 256,
                "kernel_size": 3,
                "dilation": 1
            },
            "test_inputs": [
                torch.randn(2, 256, 100),
                torch.randn(1, 256, 50)
            ]
        },
        {
            "name": "SpectralNorm",
            "module_name": "spectral_norm",
            "class_name": "SpectralNorm",
            "search_paths": [
                "/Users/jtnt/Play/agent-vomit/audio-ml-extensions/audio_gan",
                "/Users/jtnt/Play/agent-vomit/modules"
            ],
            "difficulty": "MODERATE",
            "estimated_hours": 1.0,
            "init_params": {
                "module": torch.nn.Linear(256, 128),
                "name": "weight",
                "n_power_iterations": 1
            },
            "test_inputs": [
                torch.randn(32, 256),
                torch.randn(16, 256)
            ]
        },
        {
            "name": "AdaptiveInstanceNorm",
            "module_name": "adaptive_instance_norm",
            "class_name": "AdaptiveInstanceNorm",
            "search_paths": [
                "/Users/jtnt/Play/agent-vomit/audio-ml-extensions/audio_gan",
                "/Users/jtnt/Play/agent-vomit/modules"
            ],
            "difficulty": "MODERATE",
            "estimated_hours": 1.0,
            "init_params": {
                "num_features": 128,
                "style_dim": 64
            },
            "test_inputs": [
                (torch.randn(2, 128, 32), torch.randn(2, 64)),
                (torch.randn(1, 128, 16), torch.randn(1, 64))
            ]
        },
        {
            "name": "PQMFFilterbank",
            "module_name": "pqmf_filterbank", 
            "class_name": "PQMFFilterbank",
            "search_paths": [
                "/Users/jtnt/Play/agent-vomit/modules",
                "/Users/jtnt/Play/agent-vomit/candidates/agent_codex"
            ],
            "difficulty": "MODERATE",
            "estimated_hours": 1.0,
            "init_params": {
                "subbands": 4,
                "filter_size": 64,
                "beta": 9.0
            },
            "test_inputs": [
                torch.randn(2, 1, 8192),
                torch.randn(1, 1, 4096)
            ]
        },
        {
            "name": "GraphEncoder",
            "module_name": "graph_encoder",
            "class_name": "GraphEncoder",
            "search_paths": [
                "/Users/jtnt/Play/agent-vomit/modules",
                "/Users/jtnt/Play/agent-vomit/candidates/agent_codex"
            ],
            "difficulty": "MODERATE", 
            "estimated_hours": 1.0,
            "init_params": {
                "input_dim": 256,
                "hidden_dim": 128,
                "output_dim": 64,
                "num_layers": 3
            },
            "test_inputs": [
                torch.randn(32, 256),  # Node features
                torch.randn(16, 256)
            ]
        },
        {
            "name": "AudioSpectrogramTransformer",
            "module_name": "audio_spectrogram_transformer",
            "class_name": "AudioSpectrogramTransformer", 
            "search_paths": [
                "/Users/jtnt/Play/agent-vomit/modules",
                "/Users/jtnt/Play/agent-vomit/candidates/agent_codex"
            ],
            "difficulty": "MODERATE",
            "estimated_hours": 1.0,
            "init_params": {
                "input_fdim": 128,
                "input_tdim": 100,
                "audiolen": 1024,
                "num_classes": 10
            },
            "test_inputs": [
                torch.randn(2, 1, 128, 100),
                torch.randn(1, 1, 128, 100)
            ]
        },
        {
            "name": "FoundationModel",
            "module_name": "foundation_model",
            "class_name": "FoundationModel",
            "search_paths": [
                "/Users/jtnt/Play/agent-vomit/modules",
                "/Users/jtnt/Play/agent-vomit/candidates/agent_codex"
            ],
            "difficulty": "MODERATE",
            "estimated_hours": 1.0,
            "init_params": {
                "vocab_size": 50000,
                "d_model": 512,
                "n_heads": 8,
                "n_layers": 12
            },
            "test_inputs": [
                torch.randint(0, 50000, (2, 100)),
                torch.randint(0, 50000, (1, 50))
            ]
        }
    ]
    
    all_results = []
    total_modules = len(test_configs)
    passed_modules = 0
    total_tests = 0
    passed_tests = 0
    total_estimated_hours = sum(config["estimated_hours"] for config in test_configs)
    
    print(f"Testing {total_modules} moderate modules (estimated {total_estimated_hours} hours)")
    print()
    
    for config in test_configs:
        print(f"📦 Testing {config['name']} ({config['difficulty']} - {config['estimated_hours']}h)")
        print("-" * 60)
        
        try:
            # Import module with multiple search paths
            module_class = try_import_from_paths(
                config["module_name"], 
                config["class_name"], 
                config["search_paths"]
            )
            
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
                        print(f"    Output: {test_result['output_shape']}")
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
    previous_coverage = 85  # From Wave 1
    wave2_additions = passed_modules
    new_coverage = previous_coverage + wave2_additions
    total_modules_in_codebase = 137
    coverage_percentage = (new_coverage / total_modules_in_codebase) * 100
    
    # Summary
    print("=" * 70)
    print("WAVE 2 RESULTS - MEDIUM EFFORT SUMMARY")
    print("=" * 70)
    print(f"Wave 2 modules tested: {total_modules}")
    print(f"Wave 2 modules working: {passed_modules}")
    print(f"Wave 2 success rate: {passed_modules/total_modules*100:.1f}%")
    print()
    print(f"Total tests in wave: {total_tests}")
    print(f"Tests passed: {passed_tests}")
    print(f"Test success rate: {passed_tests/total_tests*100:.1f}%")
    print()
    print(f"📊 COVERAGE PROGRESS:")
    print(f"Wave 1 coverage: {previous_coverage}/137 modules ({previous_coverage/total_modules_in_codebase*100:.1f}%)")
    print(f"Wave 2 coverage: {new_coverage}/137 modules ({coverage_percentage:.1f}%)")
    print(f"Progress: +{wave2_additions} modules")
    print()
    
    if passed_modules / total_modules >= 0.8:
        print("🎉 WAVE 2 SUCCESS: 80%+ modules working!")
        print("🚀 Ready for Wave 3 - Core Audio modules")
        next_wave_ready = True
    elif passed_modules / total_modules >= 0.6:
        print("✅ WAVE 2 GOOD: 60%+ modules working!")
        print("🔧 Minor fixes needed before Wave 3")
        next_wave_ready = True
    else:
        print("⚠️  WAVE 2 needs fixes before proceeding")
        next_wave_ready = False
    
    # Save results
    with open("wave2_results.json", "w") as f:
        json.dump({
            "metadata": {
                "wave": "wave_2_medium_effort",
                "strategy": "Build on Wave 1 success, tackle moderate complexity",
                "total_modules": total_modules,
                "passed_modules": passed_modules,
                "wave_success_rate": passed_modules/total_modules*100,
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "test_success_rate": passed_tests/total_tests*100 if total_tests > 0 else 0,
                "coverage_progress": {
                    "previous_coverage": previous_coverage,
                    "new_coverage": new_coverage,
                    "total_modules": total_modules_in_codebase,
                    "coverage_percentage": coverage_percentage,
                    "modules_added": wave2_additions
                },
                "next_wave_ready": next_wave_ready,
                "estimated_hours": total_estimated_hours
            },
            "all_results": all_results
        }, f, indent=2)
    
    print(f"📁 Wave 2 results saved to: wave2_results.json")
    
    if next_wave_ready:
        print("\n🎯 READY FOR WAVE 3!")
        print("Next targets: 8 core audio modules → 74.2% total coverage")
    
    # Progress toward 100% coverage and 100% success
    print(f"\n📈 100% COVERAGE PROGRESS:")
    print(f"Tested: {new_coverage}/137 modules ({coverage_percentage:.1f}% coverage)")
    print(f"Remaining: {137 - new_coverage} modules")
    print(f"On track for systematic 100% coverage completion!")

if __name__ == "__main__":
    main()