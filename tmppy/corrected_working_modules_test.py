#!/usr/bin/env python3
"""
Corrected test with proper parameter names for proven working modules
"""

import torch
import time
import json
from typing import Dict, List, Any
import traceback

def test_module_directly(module_class, init_params: Dict, test_inputs: List[torch.Tensor]) -> Dict[str, Any]:
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
                        # Take first tensor for shape/dtype
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
                    numerical_summary = str(output)
                
                # Test gradient flow if input requires grad
                gradient_norm = None
                if hasattr(test_input, 'requires_grad') and test_input.requires_grad:
                    try:
                        if isinstance(output, torch.Tensor) and output.requires_grad:
                            loss = output.sum()
                            loss.backward()
                            if test_input.grad is not None:
                                gradient_norm = float(test_input.grad.norm())
                        elif isinstance(output, (tuple, list)) and len(output) > 0:
                            # Handle multiple outputs
                            if isinstance(output[0], torch.Tensor) and output[0].requires_grad:
                                loss = output[0].sum()
                                loss.backward()
                                if test_input.grad is not None:
                                    gradient_norm = float(test_input.grad.norm())
                        test_input.grad = None  # Clear gradients
                    except Exception:
                        pass
                
                results.append({
                    "test_name": test_name,
                    "passed": True,
                    "execution_time": execution_time,
                    "output_shape": output_shape,
                    "output_dtype": output_dtype,
                    "numerical_summary": numerical_summary,
                    "gradient_norm": gradient_norm,
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
                    "gradient_norm": None,
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
    """Test all modules with corrected parameter names"""
    
    print("🎯 Corrected Test of Working Modules with Proper Parameters")
    print("=" * 70)
    print()
    
    # Corrected test configurations
    test_configs = [
        {
            "name": "TransformerBlock",
            "module_path": "modules.transformer_block",
            "class_name": "TransformerBlock",
            "init_params": {
                "d_model": 512,
                "n_heads": 8,
                "d_ff": 2048,
                "dropout": 0.1
            },
            "test_inputs": [
                torch.randn(2, 10, 512),
                torch.randn(2, 10, 512, requires_grad=True)
            ]
        },
        {
            "name": "MultiHeadAttention",
            "module_path": "modules.transformer_block",
            "class_name": "MultiHeadAttention",
            "init_params": {
                "d_model": 256,
                "n_heads": 8
            },
            "test_inputs": [
                torch.randn(2, 10, 256),
                torch.randn(2, 10, 256, requires_grad=True)
            ]
        },
        {
            "name": "FeedForwardNetwork",
            "module_path": "modules.transformer_block",
            "class_name": "FeedForwardNetwork",
            "init_params": {
                "d_model": 256,
                "d_ff": 1024
            },
            "test_inputs": [
                torch.randn(2, 10, 256),
                torch.randn(2, 10, 256, requires_grad=True)
            ]
        },
        {
            "name": "SnakeActivation",
            "module_path": "modules.snake_activation",
            "class_name": "SnakeActivation",
            "init_params": {
                "channels": 64,  # Fixed: was n_channels
                "alpha_init": 1.0,
                "learnable": True
            },
            "test_inputs": [
                torch.randn(2, 64, 100),
                torch.randn(2, 64, 100, requires_grad=True)
            ]
        },
        {
            "name": "ConvEncoder",
            "module_path": "modules.conv_encoder",
            "class_name": "ConvEncoder",
            "init_params": {
                "in_channels": 3,
                "base_channels": 64
            },
            "test_inputs": [
                torch.randn(2, 3, 64, 64),  # Fixed: 4D tensor for conv
                torch.randn(2, 3, 64, 64, requires_grad=True)
            ]
        },
        {
            "name": "ResidualVectorQuantizer",
            "module_path": "modules.residual_vector_quantizer",
            "class_name": "ResidualVectorQuantizer",
            "init_params": {
                "num_quantizers": 4,  # Fixed: correct param names
                "num_embeddings": 1024,
                "embedding_dim": 256,
                "commitment_cost": 0.25
            },
            "test_inputs": [
                torch.randn(2, 10, 256),
                torch.randn(2, 10, 256, requires_grad=True)
            ]
        },
        {
            "name": "Antialiased2dConv",  # Check actual class name
            "module_path": "modules.antialiased_conv",
            "class_name": "Antialiased2dConv",
            "init_params": {
                "in_channels": 64,
                "out_channels": 128,
                "kernel_size": 3,
                "stride": 2
            },
            "test_inputs": [
                torch.randn(2, 64, 32, 32),
                torch.randn(2, 64, 32, 32, requires_grad=True)
            ]
        },
        {
            "name": "MultiScaleSTFTLoss",
            "module_path": "modules.stft_loss",
            "class_name": "MultiScaleSTFTLoss",
            "init_params": {
                "scales": [  # Fixed: using 'scales' instead of individual lists
                    {"fft_size": 1024, "hop_size": 120, "win_length": 600},
                    {"fft_size": 2048, "hop_size": 240, "win_length": 1200},
                    {"fft_size": 512, "hop_size": 50, "win_length": 240}
                ]
            },
            "test_inputs": [
                (torch.randn(2, 8192), torch.randn(2, 8192))
            ]
        },
        {
            "name": "CausalConv1d",
            "module_path": "modules.causal_conv",
            "class_name": "CausalConv1d",
            "init_params": {
                "in_channels": 512,
                "out_channels": 512,
                "kernel_size": 3
            },
            "test_inputs": [
                torch.randn(2, 512, 100),
                torch.randn(2, 512, 100, requires_grad=True)
            ]
        },
        {
            "name": "TimeSeriesEncoder",
            "module_path": "modules.time_series_encoder",
            "class_name": "TimeSeriesEncoder",
            "init_params": {
                "input_dim": 256,  # Fixed: correct param names  
                "output_dim": 128,
                "num_layers": 2
            },
            "test_inputs": [
                torch.randn(2, 10, 256),
                torch.randn(2, 10, 256, requires_grad=True)
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
                    print(f"❌ SOME TESTS FAILED ({module_passed_tests}/{module_total_tests})")
                
                # Show individual test results
                for test_result in test_results:
                    status = "✅" if test_result["passed"] else "❌"
                    print(f"  {status} {test_result['test_name']}: ", end="")
                    
                    if test_result["passed"]:
                        print(f"Time: {test_result['execution_time']:.3f}s, Shape: {test_result['output_shape']}")
                        if test_result["gradient_norm"]:
                            print(f"    Gradient norm: {test_result['gradient_norm']:.6f}")
                        # Show numerical summary for verification
                        if isinstance(test_result["numerical_summary"], dict):
                            ns = test_result["numerical_summary"]
                            print(f"    Stats: mean={ns['mean']:.6f}, std={ns['std']:.6f}, norm={ns['norm']:.6f}")
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
    print("=" * 70)
    print("CORRECTED TEST SUMMARY")
    print("=" * 70)
    print(f"Modules tested: {total_modules}")
    print(f"Modules fully working: {passed_modules}")
    print(f"Module success rate: {passed_modules/total_modules*100:.1f}%")
    print()
    print(f"Total individual tests: {total_tests}")
    print(f"Tests passed: {passed_tests}")
    print(f"Test success rate: {passed_tests/total_tests*100:.1f}%")
    print()
    
    if passed_modules / total_modules >= 0.7:
        print("🎉 TARGET ACHIEVED: 70%+ modules fully working!")
        print("🚀 Ready for F# implementation with high confidence!")
    elif passed_modules / total_modules >= 0.6:
        print("🔥 EXCELLENT: 60%+ modules fully working!")
        print("✅ Strong foundation for F# validation!")
    elif passed_modules / total_modules >= 0.5:
        print("✅ GOOD: 50%+ modules fully working!")
        print("🛠️ Solid base with room for expansion!")
    else:
        print("⚠️  Making progress but need more working modules")
    
    # Save results for F# validation baseline
    with open("production_ready_baseline.json", "w") as f:
        json.dump({
            "metadata": {
                "test_type": "production_ready_modules",
                "framework_status": "CORRECTED_PARAMETERS", 
                "total_modules": total_modules,
                "passed_modules": passed_modules,
                "module_success_rate": passed_modules/total_modules*100,
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "test_success_rate": passed_tests/total_tests*100 if total_tests > 0 else 0,
                "f_sharp_ready": passed_modules >= total_modules * 0.7
            },
            "working_modules": [
                r for r in all_results if r["result"]["module_passed"]
            ],
            "all_results": all_results
        }, f, indent=2)
    
    print(f"📁 Production baseline saved to: production_ready_baseline.json")
    
    if passed_modules / total_modules >= 0.7:
        print("🎯 SUCCESS: Framework ready for F# cross-language validation!")

if __name__ == "__main__":
    main()