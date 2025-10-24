#!/usr/bin/env python3
"""
Test Critical Advanced Modules

Tests the 5 most important advanced audio ML modules with enhanced parameter mapping:
1. Foundation Models (AudioMAE, Data2Vec, WavLM)  
2. Audio Spectrogram Transformer
3. Spectral Normalization
4. PQMF FilterBank
5. WaveNet ResBlock

These represent state-of-the-art audio ML for F# validation framework.
"""

import json
import torch
import time
from pathlib import Path
from typing import Dict, List, Any
from enhanced_universal_test_runner import (
    EnhancedPythonTestImplementation, 
    EnhancedUniversalTestRunner,
    TestResult
)

def create_critical_module_configs() -> List[Dict[str, Any]]:
    """Create test configurations for the 5 critical advanced modules"""
    
    configs = []
    
    # 1. Foundation Models - AudioMAE
    configs.append({
        "module_name": "advanced_modules.foundation_models",
        "class_name": "AudioMAE", 
        "init_params": {
            "model_type": "audio_mae",
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "dropout": 0.1,
            "intermediate_size": 3072
        },
        "tests": [
            {
                "name": "forward_pass_waveform",
                "test_type": "forward",
                "args": [{
                    "type": "tensor",
                    "shape": [2, 48000],  # 3 seconds at 16kHz
                    "dtype": "float32",
                    "requires_grad": False
                }]
            },
            {
                "name": "shape_consistency",
                "test_type": "shape",
                "input_tensor": {
                    "type": "tensor", 
                    "shape": [1, 32000],
                    "dtype": "float32"
                },
                "expected_shape": "reduce"
            },
            {
                "name": "gradient_flow",
                "test_type": "gradient",
                "input_tensor": {
                    "type": "tensor",
                    "shape": [1, 16000],
                    "dtype": "float32",
                    "requires_grad": True
                }
            }
        ]
    })
    
    # 2. Foundation Models - Data2VecAudio
    configs.append({
        "module_name": "advanced_modules.foundation_models",
        "class_name": "Data2VecAudio",
        "init_params": {
            "model_type": "data2vec_audio", 
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "dropout": 0.1
        },
        "tests": [
            {
                "name": "forward_pass_contrastive",
                "test_type": "forward",
                "args": [{
                    "type": "tensor",
                    "shape": [2, 48000],
                    "dtype": "float32"
                }]
            },
            {
                "name": "self_supervised_loss",
                "test_type": "callable",
                "input_tensor": {
                    "type": "tensor",
                    "shape": [1, 32000],
                    "dtype": "float32"
                }
            }
        ]
    })
    
    # 3. Foundation Models - WavLM  
    configs.append({
        "module_name": "advanced_modules.foundation_models",
        "class_name": "WavLM",
        "init_params": {
            "model_type": "wavlm",
            "hidden_size": 768, 
            "num_layers": 12,
            "num_heads": 12,
            "mask_prob": 0.15,
            "mask_length": 10
        },
        "tests": [
            {
                "name": "forward_pass_masked",
                "test_type": "forward",
                "args": [{
                    "type": "tensor",
                    "shape": [2, 48000],
                    "dtype": "float32"
                }]
            },
            {
                "name": "contrastive_learning",
                "test_type": "gradient",
                "input_tensor": {
                    "type": "tensor",
                    "shape": [1, 32000],
                    "dtype": "float32",
                    "requires_grad": True
                }
            }
        ]
    })
    
    # 4. Audio Spectrogram Transformer
    configs.append({
        "module_name": "audio_analysis.audio_spectrogram_transformer", 
        "class_name": "AudioSpectrogramTransformer",
        "init_params": {
            "img_size": [1024, 128],
            "patch_size": [16, 16],
            "num_classes": 527,
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "sample_rate": 16000,
            "n_mels": 128
        },
        "tests": [
            {
                "name": "forward_pass_classification",
                "test_type": "forward",
                "args": [{
                    "type": "tensor",
                    "shape": [2, 48000],  # Raw waveform
                    "dtype": "float32"
                }]
            },
            {
                "name": "feature_extraction",
                "test_type": "shape",
                "input_tensor": {
                    "type": "tensor",
                    "shape": [1, 32000],
                    "dtype": "float32"
                },
                "expected_shape": "reduce"
            },
            {
                "name": "attention_gradients", 
                "test_type": "gradient",
                "input_tensor": {
                    "type": "tensor",
                    "shape": [1, 16000],
                    "dtype": "float32",
                    "requires_grad": True
                }
            }
        ]
    })
    
    # 5. Spectral Normalization
    configs.append({
        "module_name": "audio_gan.spectral_normalization",
        "class_name": "SpectralNormalization",
        "init_params": {
            "input_dim": 64,
            "output_dim": 32,
            "power_iterations": 1,
            "eps": 1e-12
        },
        "tests": [
            {
                "name": "forward_pass_normalized",
                "test_type": "forward", 
                "args": [{
                    "type": "tensor",
                    "shape": [8, 64],
                    "dtype": "float32"
                }]
            },
            {
                "name": "spectral_norm_constraint",
                "test_type": "gradient",
                "input_tensor": {
                    "type": "tensor",
                    "shape": [4, 64], 
                    "dtype": "float32",
                    "requires_grad": True
                }
            }
        ]
    })
    
    # 6. PQMF FilterBank
    configs.append({
        "module_name": "audio_gan.pqmf_filterbank",
        "class_name": "PQMFFilterBank",
        "init_params": {
            "num_bands": 4,
            "filter_length": 640,
            "beta": 9.0
        },
        "tests": [
            {
                "name": "forward_pass_multiband",
                "test_type": "forward",
                "args": [{
                    "type": "tensor", 
                    "shape": [2, 1, 8192],  # [batch, channels, time]
                    "dtype": "float32"
                }]
            },
            {
                "name": "perfect_reconstruction",
                "test_type": "shape",
                "input_tensor": {
                    "type": "tensor",
                    "shape": [1, 1, 4096],
                    "dtype": "float32"
                },
                "expected_shape": "preserve"
            },
            {
                "name": "analysis_synthesis",
                "test_type": "callable",
                "input_tensor": {
                    "type": "tensor",
                    "shape": [1, 1, 2048],
                    "dtype": "float32"
                }
            }
        ]
    })
    
    # 7. WaveNet ResBlock
    configs.append({
        "module_name": "audio_gan.wavenet_resblock",
        "class_name": "WaveNetResBlock",
        "init_params": {
            "channels": 256,
            "kernel_size": 3,
            "dilation": 1,
            "skip_channels": 256, 
            "residual_channels": 256,
            "gate_channels": 256,
            "dropout": 0.0
        },
        "tests": [
            {
                "name": "forward_pass_causal",
                "test_type": "forward",
                "args": [{
                    "type": "tensor",
                    "shape": [2, 256, 1024],  # [batch, channels, time]
                    "dtype": "float32"
                }]
            },
            {
                "name": "residual_skip_outputs",
                "test_type": "shape", 
                "input_tensor": {
                    "type": "tensor",
                    "shape": [1, 256, 512],
                    "dtype": "float32"
                },
                "expected_shape": "preserve"
            },
            {
                "name": "dilated_conv_gradients",
                "test_type": "gradient",
                "input_tensor": {
                    "type": "tensor",
                    "shape": [1, 256, 256],
                    "dtype": "float32",
                    "requires_grad": True
                }
            }
        ]
    })
    
    return configs

def test_critical_modules():
    """Test the 5 critical advanced modules with enhanced parameter mapping"""
    
    print("=" * 80)
    print("TESTING CRITICAL ADVANCED AUDIO ML MODULES")
    print("=" * 80)
    print("Testing 5 critical modules for F# validation framework:")
    print("1. Foundation Models (AudioMAE, Data2Vec, WavLM)")
    print("2. Audio Spectrogram Transformer") 
    print("3. Spectral Normalization")
    print("4. PQMF FilterBank")
    print("5. WaveNet ResBlock")
    print()
    
    # Create test configuration
    configs = create_critical_module_configs()
    
    # Save test configuration
    test_config = {"modules": configs}
    config_path = "critical_modules_test_config.json"
    
    with open(config_path, 'w') as f:
        json.dump(test_config, f, indent=2)
    
    print(f"📝 Test config saved to: {config_path}")
    
    # Run tests
    implementation = EnhancedPythonTestImplementation()
    runner = EnhancedUniversalTestRunner(implementation)
    
    start_time = time.time()
    results = runner.run_from_json_config(config_path)
    test_duration = time.time() - start_time
    
    # Print summary
    runner.print_summary()
    
    # Analyze results by module category
    print("\n" + "=" * 60)
    print("CRITICAL MODULE ANALYSIS")
    print("=" * 60)
    
    foundation_results = [r for r in results if r.class_name in ['AudioMAE', 'Data2VecAudio', 'WavLM']]
    ast_results = [r for r in results if r.class_name == 'AudioSpectrogramTransformer']
    sn_results = [r for r in results if r.class_name == 'SpectralNormalization']
    pqmf_results = [r for r in results if r.class_name == 'PQMFFilterBank']
    wavenet_results = [r for r in results if r.class_name == 'WaveNetResBlock']
    
    categories = [
        ("Foundation Models", foundation_results),
        ("Audio Spectrogram Transformer", ast_results),
        ("Spectral Normalization", sn_results), 
        ("PQMF FilterBank", pqmf_results),
        ("WaveNet ResBlock", wavenet_results)
    ]
    
    for category_name, category_results in categories:
        if category_results:
            passed = sum(1 for r in category_results if r.passed)
            total = len(category_results)
            rate = 100 * passed / total if total > 0 else 0
            
            status = "✅" if rate >= 75 else "⚠️" if rate >= 50 else "❌"
            print(f"{status} {category_name}: {passed}/{total} ({rate:.1f}%)")
            
            # Show specific failures
            failures = [r for r in category_results if not r.passed]
            if failures:
                for failure in failures[:2]:  # Show first 2 failures
                    print(f"    Failed: {failure.name} - {failure.error_message}")
    
    # Save detailed results
    results_path = "test_results/critical_modules_results.json"
    Path("test_results").mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        serializable_results = []
        for r in results:
            result_dict = {
                'name': r.name,
                'module_name': r.module_name,
                'class_name': r.class_name,
                'test_type': r.test_type,
                'language': r.language,
                'passed': r.passed,
                'error_message': r.error_message,
                'execution_time': r.execution_time,
                'output_shape': r.output_shape,
                'output_dtype': r.output_dtype,
                'gradient_norm': r.gradient_norm,
                'numerical_summary': r.numerical_summary
            }
            serializable_results.append(result_dict)
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n📁 Detailed results saved to: {results_path}")
    print(f"⏱️  Total test duration: {test_duration:.2f} seconds")
    
    # Success criteria for F# validation
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.passed)
    success_rate = 100 * passed_tests / total_tests if total_tests > 0 else 0
    
    print(f"\n🎯 F# VALIDATION READINESS:")
    if success_rate >= 75:
        print(f"✅ EXCELLENT ({success_rate:.1f}%) - Ready for F# cross-validation!")
        print("   Advanced modules are working well with enhanced parameter mapping.")
    elif success_rate >= 50:
        print(f"⚠️  GOOD ({success_rate:.1f}%) - Mostly ready for F# validation")
        print("   Some parameter mapping issues remain but core functionality works.")
    else:
        print(f"❌ NEEDS WORK ({success_rate:.1f}%) - More parameter fixes needed")
        print("   Significant issues with parameter mapping for advanced modules.")
    
    return results

if __name__ == "__main__":
    results = test_critical_modules()