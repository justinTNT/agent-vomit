#!/usr/bin/env python3
"""
Comprehensive test script that tests all 25 modules for both agent_claude and agent_codex.
Uses flexible testing utilities to maximize success rates.
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime
import traceback
from collections import defaultdict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from test_utils import (
    init_with_variations,
    extract_output,
    validate_shape_behavior,
    test_mask_support,
    check_gradient_flow,
    find_method,
    test_callable_interface
)

from test_utils_v2 import (
    init_with_combinations,
    test_with_smart_init,
    flexible_module_test_v2
)

import torch
import torch.nn as nn


# Complete test configuration for all 25 modules
ALL_MODULES_CONFIG = [
    # 1. Transformer Block
    {
        'module': 'transformer_block',
        'class': 'TransformerBlock',
        'init_params': {
            'd_model': 512,
            'n_heads': 8,
            'd_ff': 2048,
            'dropout': 0.1
        },
        'tests': [
            {'name': 'forward', 'type': 'forward', 'args': [torch.randn(2, 10, 512)]},
            {'name': 'shape_preserve', 'type': 'shape', 'input': torch.randn(2, 10, 512)},
            {'name': 'gradient', 'type': 'gradient', 'input': torch.randn(2, 10, 512)}
        ]
    },
    
    # 2. Conv Encoder
    {
        'module': 'conv_encoder',
        'class': 'ConvEncoder',
        'init_params': {
            'in_channels': 3,
            'base_channels': 64,
            'num_layers': 4,
            'latent_dim': 512
        },
        'tests': [
            {'name': 'forward', 'type': 'forward', 'args': [torch.randn(2, 3, 64, 64)]},
            {'name': 'shape_reduce', 'type': 'shape', 'input': torch.randn(2, 3, 64, 64), 'expected': 'reduce'},
            {'name': 'gradient', 'type': 'gradient', 'input': torch.randn(2, 3, 64, 64)}
        ]
    },
    
    # 3. Snake Activation
    {
        'module': 'snake_activation',
        'class': 'SnakeActivation',
        'class_variations': ['SnakeActivation', 'Snake'],
        'init_params': {
            'n_channels': 64,
            'alpha_init': 1.0
        },
        'tests': [
            {'name': 'forward', 'type': 'forward', 'args': [torch.randn(2, 64, 100)]},
            {'name': 'shape_preserve', 'type': 'shape', 'input': torch.randn(2, 64, 100)},
            {'name': 'gradient', 'type': 'gradient', 'input': torch.randn(2, 64, 100)}
        ]
    },
    
    # 4. Residual Vector Quantizer
    {
        'module': 'residual_vector_quantizer',
        'class': 'ResidualVectorQuantizer',
        'class_variations': ['ResidualVectorQuantizer', 'ResidualVQ'],
        'init_params': {
            'dim': 256,
            'num_quantizers': 8,
            'codebook_size': 1024,
            'decay': 0.8,
            'commitment_weight': 1.0
        },
        'tests': [
            {'name': 'forward', 'type': 'forward', 'args': [torch.randn(2, 10, 256)]},
            {'name': 'shape_preserve', 'type': 'shape', 'input': torch.randn(2, 10, 256)},
            {'name': 'gradient', 'type': 'gradient', 'input': torch.randn(2, 10, 256)}
        ]
    },
    
    # 5. Antialiased Conv
    {
        'module': 'antialiased_conv',
        'class': 'AntialiasedConv',
        'class_variations': ['AntialiasedConv', 'AntiAliasedConv', 'AAConv'],
        'init_params': {
            'in_channels': 64,
            'out_channels': 128,
            'kernel_size': 3,
            'stride': 2
        },
        'tests': [
            {'name': 'forward', 'type': 'forward', 'args': [torch.randn(2, 64, 32, 32)]},
            {'name': 'shape_reduce', 'type': 'shape', 'input': torch.randn(2, 64, 32, 32), 'expected': 'reduce'}
        ]
    },
    
    # 6. Multi-Scale STFT Loss
    {
        'module': 'stft_loss',
        'class': 'MultiScaleSTFTLoss',
        'class_variations': ['MultiScaleSTFTLoss', 'STFTLoss', 'multi_scale_stft_loss'],
        'init_params': {
            'fft_sizes': [2048, 1024, 512],
            'hop_sizes': [512, 256, 128],
            'win_lengths': [2048, 1024, 512]
        },
        'tests': [
            {'name': 'forward', 'type': 'forward', 'args': [torch.randn(2, 16000), torch.randn(2, 16000)]},
            {'name': 'callable', 'type': 'callable', 'args': (torch.randn(2, 16000), torch.randn(2, 16000))}
        ]
    },
    
    # 7. Causal Conv1D
    {
        'module': 'causal_conv',
        'class': 'CausalConv1d',
        'class_variations': ['CausalConv1d', 'CausalConv'],
        'init_params': {
            'in_channels': 256,
            'out_channels': 512,
            'kernel_size': 3,
            'dilation': 1,
            'groups': 1
        },
        'tests': [
            {'name': 'forward', 'type': 'forward', 'args': [torch.randn(2, 256, 100)]},
            {'name': 'shape_preserve', 'type': 'shape', 'input': torch.randn(2, 256, 100)},
            {'name': 'gradient', 'type': 'gradient', 'input': torch.randn(2, 256, 100)}
        ]
    },
    
    # 8. Stream Joiner
    {
        'module': 'stream_joiner',
        'class': 'StreamJoiner',
        'init_params': {
            'join_type': 'inner',
            'time_window': 1.0,
            'buffer_size': 100
        },
        'tests': [
            {'name': 'forward', 'type': 'forward', 'args': ["stream1", torch.randn(4), 1.0]},
            {'name': 'add_method', 'type': 'method', 'method': 'add'},
            {'name': 'join_method', 'type': 'method', 'method': 'join'}
        ]
    },
    
    # 9. Data Versioner
    {
        'module': 'data_versioner',
        'class': 'DataVersioner',
        'init_params': {},
        'tests': [
            {'name': 'callable', 'type': 'callable', 'args': (torch.randn(10, 10), "test")},
            {'name': 'save_method', 'type': 'method', 'method': 'save'},
            {'name': 'load_method', 'type': 'method', 'method': 'load'}
        ]
    },
    
    # 10. Data Sampler
    {
        'module': 'data_sampler',
        'class': 'DataSampler',
        'init_params': {
            'dataset_size': 1000,
            'batch_size': 32,
            'num_replicas': 1,
            'rank': 0,
            'shuffle': True,
            'seed': 42,
            'drop_last': False
        },
        'tests': [
            {'name': 'callable', 'type': 'callable'},
            {'name': 'iter_method', 'type': 'method', 'method': '__iter__'},
            {'name': 'len_method', 'type': 'method', 'method': '__len__'}
        ]
    },
    
    # 11. Feature Store
    {
        'module': 'feature_store',
        'class': 'FeatureStore',
        'init_params': {
            'dim': 256,
            'max_size': 10000,
            'replacement': 'fifo'
        },
        'tests': [
            {'name': 'add_method', 'type': 'method', 'method': 'add'},
            {'name': 'query_method', 'type': 'method', 'method': 'query'},
            {'name': 'clear_method', 'type': 'method', 'method': 'clear'}
        ]
    },
    
    # 12. Data Validator
    {
        'module': 'data_validator',
        'class': 'DataValidator',
        'init_params': {
            'schema': {
                'type': 'tensor',
                'shape': [None, 10],
                'dtype': 'float32'
            }
        },
        'tests': [
            {'name': 'validate_method', 'type': 'method', 'method': 'validate'},
            {'name': 'callable', 'type': 'callable', 'args': (torch.randn(5, 10),)}
        ]
    },
    
    # 13. Stream Processor
    {
        'module': 'stream_processor',
        'class': 'StreamProcessor',
        'init_params': {
            'window_size': 1000,
            'step_size': 100,
            'transform': 'fft'
        },
        'tests': [
            {'name': 'process_method', 'type': 'method', 'method': 'process'},
            {'name': 'reset_method', 'type': 'method', 'method': 'reset'}
        ]
    },
    
    # 14. Adaptive Computation
    {
        'module': 'adaptive_computation',
        'class': 'AdaptiveComputation',
        'init_params': {
            'd_model': 512,
            'max_steps': 10,
            'threshold': 0.01
        },
        'tests': [
            {'name': 'forward', 'type': 'forward', 'args': [torch.randn(2, 10, 512)]},
            {'name': 'shape_preserve', 'type': 'shape', 'input': torch.randn(2, 10, 512)},
            {'name': 'gradient', 'type': 'gradient', 'input': torch.randn(2, 10, 512)}
        ]
    },
    
    # 15. Memory Bank Retriever
    {
        'module': 'memory_bank_retriever',
        'class': 'MemoryBankRetriever',
        'init_params': {
            'dim': 256,
            'num_slots': 1024,
            'num_heads': 8
        },
        'tests': [
            {'name': 'forward', 'type': 'forward', 'args': [torch.randn(2, 10, 256)]},
            {'name': 'shape_preserve', 'type': 'shape', 'input': torch.randn(2, 10, 256)},
            {'name': 'update_method', 'type': 'method', 'method': 'update'}
        ]
    },
    
    # 16. Graph Encoder
    {
        'module': 'graph_encoder',
        'class': 'GraphEncoder',
        'init_params': {
            'in_features': 128,
            'hidden_features': 256,
            'out_features': 512,
            'num_layers': 3,
            'dropout': 0.1
        },
        'tests': [
            {'name': 'forward', 'type': 'forward', 'args': [
                torch.randn(10, 128), 
                torch.randint(0, 10, (2, 20))
            ]},
            {'name': 'gradient', 'type': 'gradient', 'input': torch.randn(10, 128)}
        ]
    },
    
    # 17. Sequence to Sequence
    {
        'module': 'sequence_to_sequence',
        'class': 'SequenceToSequenceModel',
        'class_variations': ['SequenceToSequenceModel', 'Seq2Seq'],
        'init_params': {
            'input_dim': 100,
            'hidden_dim': 256,
            'output_dim': 100,
            'num_layers': 2,
            'dropout': 0.1
        },
        'tests': [
            {'name': 'forward', 'type': 'forward', 'args': [
                torch.randn(2, 10, 100), 
                torch.randn(2, 15, 100)
            ]},
            {'name': 'encode_method', 'type': 'method', 'method': 'encode'},
            {'name': 'decode_method', 'type': 'method', 'method': 'decode'}
        ]
    },
    
    # 18. Autoencoder VAE
    {
        'module': 'autoencoder_vae',
        'class': 'AutoencoderVAE',
        'class_variations': ['AutoencoderVAE', 'VAE', 'VariationalAutoencoder'],
        'init_params': {
            'input_dim': 784,
            'hidden_dim': 400,
            'latent_dim': 20
        },
        'tests': [
            {'name': 'forward', 'type': 'forward', 'args': [torch.randn(2, 784)]},
            {'name': 'encode_method', 'type': 'method', 'method': 'encode'},
            {'name': 'decode_method', 'type': 'method', 'method': 'decode'},
            {'name': 'gradient', 'type': 'gradient', 'input': torch.randn(2, 784)}
        ]
    },
    
    # 19. Contrastive Learner
    {
        'module': 'contrastive_learner',
        'class': 'ContrastiveLearner',
        'init_params': {
            'encoder_dim': 512,
            'projection_dim': 128,
            'temperature': 0.07
        },
        'tests': [
            {'name': 'forward', 'type': 'forward', 'args': [
                torch.randn(8, 3, 224, 224),
                torch.randn(8, 3, 224, 224)
            ]},
            {'name': 'compute_loss_method', 'type': 'method', 'method': 'compute_loss'}
        ]
    },
    
    # 20. Set Encoder
    {
        'module': 'set_encoder',
        'class': 'SetEncoder',
        'init_params': {
            'in_features': 128,
            'hidden_features': 256,
            'out_features': 512,
            'num_seeds': 1,
            'num_heads': 8
        },
        'tests': [
            {'name': 'forward', 'type': 'forward', 'args': [torch.randn(2, 10, 128)]},
            {'name': 'shape_reduce', 'type': 'shape', 'input': torch.randn(2, 10, 128), 'expected': 'reduce'},
            {'name': 'gradient', 'type': 'gradient', 'input': torch.randn(2, 10, 128)}
        ]
    },
    
    # 21. Time Series Encoder
    {
        'module': 'time_series_encoder',
        'class': 'TimeSeriesEncoder',
        'init_params': {
            'input_dim': 10,
            'hidden_dim': 128,
            'num_layers': 3,
            'dropout': 0.1,
            'bidirectional': True
        },
        'tests': [
            {'name': 'forward', 'type': 'forward', 'args': [torch.randn(2, 50, 10)]},
            {'name': 'shape_preserve', 'type': 'shape', 'input': torch.randn(2, 50, 10)},
            {'name': 'gradient', 'type': 'gradient', 'input': torch.randn(2, 50, 10)}
        ]
    },
    
    # 22. Cross Modal Fusion
    {
        'module': 'cross_modal_fusion',
        'class': 'CrossModalFusion',
        'init_params': {
            'dim_a': 512,
            'dim_b': 768,
            'hidden_dim': 512,
            'output_dim': 256,
            'num_heads': 8
        },
        'tests': [
            {'name': 'forward', 'type': 'forward', 'args': [
                torch.randn(2, 10, 512),
                torch.randn(2, 20, 768)
            ]},
            {'name': 'gradient', 'type': 'gradient', 'input': torch.randn(2, 10, 512)}
        ]
    },
    
    # 23. ViT Patch Encoder
    {
        'module': 'vit_patch_encoder',
        'class': 'ViTPatchEncoder',
        'class_variations': ['ViTPatchEncoder', 'VitPatchEncoder', 'PatchEncoder'],
        'init_params': {
            'image_size': 224,
            'patch_size': 16,
            'in_channels': 3,
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'mlp_ratio': 4.0,
            'num_classes': 1000
        },
        'tests': [
            {'name': 'forward', 'type': 'forward', 'args': [torch.randn(2, 3, 224, 224)]},
            {'name': 'gradient', 'type': 'gradient', 'input': torch.randn(2, 3, 224, 224)}
        ]
    },
    
    # 24. Attention Decoder
    {
        'module': 'attention_decoder',
        'class': 'AttentionDecoder',
        'init_params': {
            'd_model': 512,
            'vocab_size': 32000,
            'max_length': 100,
            'num_layers': 6,
            'num_heads': 8,
            'dropout': 0.1
        },
        'tests': [
            {'name': 'forward', 'type': 'forward', 'args': [
                torch.randint(0, 32000, (2, 20)),
                torch.randn(2, 10, 512)
            ]},
            {'name': 'generate_method', 'type': 'method', 'method': 'generate'}
        ]
    },
    
    # 25. Sequence Encoder
    {
        'module': 'sequence_encoder',
        'class': 'SequenceEncoder',
        'init_params': {
            'vocab_size': 32000,
            'd_model': 512,
            'num_layers': 6,
            'num_heads': 8,
            'max_length': 1000,
            'dropout': 0.1
        },
        'tests': [
            {'name': 'forward', 'type': 'forward', 'args': [torch.randint(0, 32000, (2, 50))]},
            {'name': 'shape_preserve', 'type': 'shape', 'input': torch.randint(0, 32000, (2, 50))},
            {'name': 'gradient', 'type': 'gradient', 'input': torch.randint(0, 32000, (2, 50))}
        ]
    }
]


def test_module_comprehensive(module_path, module_name, class_name, test_config):
    """Test a module with comprehensive error handling and flexible initialization."""
    results = {
        'module': module_name,
        'class': class_name,
        'tests_passed': [],
        'tests_failed': [],
        'status': 'unknown',
        'initialization_strategy': None,
        'error_details': None
    }
    
    # Check if module file exists
    if not os.path.exists(module_path):
        results['status'] = 'missing'
        results['error_details'] = f"Module file not found: {module_path}"
        return results
    
    # Temporarily add module to path
    original_module_path = Path('modules') / f"{module_name}.py"
    backup_path = None
    
    try:
        # Backup original if it exists
        if original_module_path.exists():
            backup_path = original_module_path.with_suffix('.py.backup')
            original_module_path.rename(backup_path)
        
        # Symlink candidate module
        os.symlink(Path(module_path).absolute(), original_module_path)
        
        # Import module
        module = __import__(f'modules.{module_name}', fromlist=[class_name])
        
        # Try to get class with variations
        ModuleClass = None
        class_variations = test_config.get('class_variations', [class_name])
        for class_variant in class_variations:
            if hasattr(module, class_variant):
                ModuleClass = getattr(module, class_variant)
                results['class'] = class_variant  # Update to actual class name used
                break
        
        if ModuleClass is None:
            results['status'] = 'error'
            results['error_details'] = f"Class {class_name} not found. Available: {[x for x in dir(module) if not x.startswith('_')]}"
            return results
        
        # Try comprehensive initialization strategies
        model = None
        init_strategy = None
        
        # Strategy 1: Original parameters
        try:
            model = ModuleClass(**test_config['init_params'])
            init_strategy = 'original'
            results['tests_passed'].append('initialization')
        except Exception as e1:
            # Strategy 2: Simple variations
            try:
                model = init_with_variations(ModuleClass, test_config['init_params'])
                init_strategy = 'variations'
                results['tests_passed'].append('initialization_with_variations')
            except Exception as e2:
                # Strategy 3: Combinations
                try:
                    model = init_with_combinations(
                        ModuleClass, test_config['init_params'],
                        max_combinations=100
                    )
                    init_strategy = 'combinations'
                    results['tests_passed'].append('initialization_with_combinations')
                except Exception as e3:
                    results['status'] = 'failed'
                    results['tests_failed'].append(('initialization', str(e1)))
                    results['error_details'] = {
                        'original_error': str(e1),
                        'variations_error': str(e2),
                        'combinations_error': str(e3)
                    }
                    return results
        
        results['initialization_strategy'] = init_strategy
        
        # Run test cases
        for test_case in test_config.get('tests', []):
            test_name = test_case['name']
            try:
                if test_case['type'] == 'forward':
                    output = model(*test_case.get('args', []), **test_case.get('kwargs', {}))
                    output_dict = extract_output(output)
                    if output_dict:
                        results['tests_passed'].append(test_name)
                    else:
                        results['tests_failed'].append((test_name, "No output"))
                        
                elif test_case['type'] == 'shape':
                    success, msg = validate_shape_behavior(
                        model, test_case['input'], test_case.get('expected', 'preserve')
                    )
                    if success:
                        results['tests_passed'].append(test_name)
                    else:
                        results['tests_failed'].append((test_name, msg))
                        
                elif test_case['type'] == 'mask':
                    success, msg = test_mask_support(model, test_case['input'])
                    if success:
                        results['tests_passed'].append(test_name)
                    else:
                        results['tests_failed'].append((test_name, msg))
                        
                elif test_case['type'] == 'gradient':
                    success, msg = check_gradient_flow(model, test_case['input'])
                    if success:
                        results['tests_passed'].append(test_name)
                    else:
                        results['tests_failed'].append((test_name, msg))
                        
                elif test_case['type'] == 'method':
                    method = find_method(model, test_case['method'])
                    if method:
                        results['tests_passed'].append(test_name)
                    else:
                        results['tests_failed'].append((test_name, f"Method '{test_case['method']}' not found"))
                        
                elif test_case['type'] == 'callable':
                    is_callable, result, msg = test_callable_interface(
                        model, test_case.get('args', ()), test_case.get('kwargs', {})
                    )
                    if is_callable:
                        results['tests_passed'].append(test_name)
                    else:
                        results['tests_failed'].append((test_name, msg))
                        
            except Exception as e:
                results['tests_failed'].append((test_name, f"{type(e).__name__}: {str(e)}"))
        
        # Determine overall status
        if len(results['tests_passed']) > 0 and len(results['tests_failed']) == 0:
            results['status'] = 'passed'
        elif len(results['tests_passed']) > 0:
            results['status'] = 'partial'
        else:
            results['status'] = 'failed'
            
    except Exception as e:
        results['status'] = 'error'
        results['error_details'] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        
    finally:
        # Restore original module
        if original_module_path.exists() or original_module_path.is_symlink():
            try:
                original_module_path.unlink()
            except:
                pass
        if backup_path and backup_path.exists():
            backup_path.rename(original_module_path)
    
    return results


def test_candidate_comprehensive(candidate_dir, modules_config):
    """Test all modules for a candidate with comprehensive reporting."""
    results = {
        'candidate': candidate_dir,
        'timestamp': datetime.now().isoformat(),
        'modules': {},
        'summary': {
            'total': 0,
            'passed': 0,
            'partial': 0,
            'failed': 0,
            'missing': 0,
            'error': 0
        },
        'patterns': {
            'by_initialization_strategy': defaultdict(int),
            'by_error_type': defaultdict(int),
            'by_test_type_failure': defaultdict(int)
        }
    }
    
    print(f"\n{'='*80}")
    print(f"Testing candidate: {candidate_dir}")
    print(f"{'='*80}")
    
    for i, module_config in enumerate(modules_config):
        module_name = module_config['module']
        class_name = module_config['class']
        
        # Handle different file naming conventions
        module_files = [
            f"{module_name}.py",
            f"{module_name.replace('_', '')}.py",
            f"{'_'.join(module_name.split('_')[:-1]) if '_' in module_name else module_name}.py"
        ]
        
        module_path = None
        for filename in module_files:
            candidate_file = Path(candidate_dir) / filename
            if candidate_file.exists():
                module_path = candidate_file
                break
        
        if not module_path:
            module_path = Path(candidate_dir) / f"{module_name}.py"
        
        print(f"\n[{i+1}/25] Testing {module_name} ({class_name})...")
        
        result = test_module_comprehensive(
            module_path, 
            module_name,
            class_name,
            module_config
        )
        
        results['modules'][module_name] = result
        results['summary']['total'] += 1
        results['summary'][result['status']] += 1
        
        # Track patterns
        if result.get('initialization_strategy'):
            results['patterns']['by_initialization_strategy'][result['initialization_strategy']] += 1
        
        # Track error types
        if result['status'] in ['failed', 'error'] and result.get('error_details'):
            if isinstance(result['error_details'], dict):
                for error_key, error_msg in result['error_details'].items():
                    if 'TypeError' in str(error_msg):
                        results['patterns']['by_error_type']['TypeError'] += 1
                    elif 'AttributeError' in str(error_msg):
                        results['patterns']['by_error_type']['AttributeError'] += 1
                    elif 'ValueError' in str(error_msg):
                        results['patterns']['by_error_type']['ValueError'] += 1
            else:
                error_msg = str(result['error_details'])
                if 'TypeError' in error_msg:
                    results['patterns']['by_error_type']['TypeError'] += 1
                elif 'AttributeError' in error_msg:
                    results['patterns']['by_error_type']['AttributeError'] += 1
                elif 'ValueError' in error_msg:
                    results['patterns']['by_error_type']['ValueError'] += 1
        
        # Track test type failures
        for test_name, error in result.get('tests_failed', []):
            for test_config in module_config.get('tests', []):
                if test_config.get('name') == test_name:
                    results['patterns']['by_test_type_failure'][test_config['type']] += 1
                    break
        
        # Print result
        status_emoji = {
            'passed': '✅',
            'partial': '⚠️',
            'failed': '❌',
            'missing': '📭',
            'error': '🚨'
        }
        
        print(f"  {status_emoji.get(result['status'], '❓')} {result['status'].upper()}", end='')
        
        if result['status'] == 'passed':
            print(f" - All {len(result['tests_passed'])} tests passed")
            if result.get('initialization_strategy') != 'original':
                print(f"     (initialized with: {result['initialization_strategy']})")
        elif result['status'] == 'partial':
            print(f" - Passed {len(result['tests_passed'])}, Failed {len(result['tests_failed'])}")
            if result.get('initialization_strategy'):
                print(f"     (initialized with: {result['initialization_strategy']})")
            for test, error in result['tests_failed'][:3]:  # Show first 3 failures
                print(f"     ❌ {test}: {error[:60]}...")
        elif result['status'] == 'missing':
            print(f" - Module file not found")
        else:
            error_msg = str(result.get('error_details', 'Unknown error'))
            if len(error_msg) > 100:
                error_msg = error_msg[:100] + "..."
            print(f" - {error_msg}")
    
    return results


def analyze_patterns(all_results):
    """Analyze patterns across all test results."""
    print("\n" + "="*80)
    print("PATTERN ANALYSIS")
    print("="*80)
    
    for candidate_results in all_results:
        candidate = candidate_results['candidate']
        patterns = candidate_results['patterns']
        
        print(f"\n{candidate}:")
        
        print("\n  Initialization Strategies Used:")
        for strategy, count in patterns['by_initialization_strategy'].items():
            print(f"    - {strategy}: {count} modules")
        
        print("\n  Common Error Types:")
        for error_type, count in sorted(patterns['by_error_type'].items(), key=lambda x: x[1], reverse=True):
            print(f"    - {error_type}: {count} occurrences")
        
        print("\n  Test Type Failures:")
        for test_type, count in sorted(patterns['by_test_type_failure'].items(), key=lambda x: x[1], reverse=True):
            print(f"    - {test_type}: {count} failures")


def generate_summary_report(all_results, output_file):
    """Generate a comprehensive summary report."""
    with open(output_file, 'w') as f:
        f.write("# Comprehensive Module Testing Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall summary
        f.write("## Overall Summary\n\n")
        f.write("| Candidate | Total | ✅ Passed | ⚠️ Partial | ❌ Failed | 📭 Missing | 🚨 Error | Success Rate |\n")
        f.write("|-----------|-------|-----------|------------|-----------|-----------|----------|-------------|\n")
        
        for results in all_results:
            summary = results['summary']
            total = summary['total']
            passed = summary['passed']
            partial = summary['partial']
            success_rate = (passed + partial) / total * 100 if total > 0 else 0
            
            f.write(f"| {results['candidate']} | {total} | {passed} | {partial} | "
                   f"{summary['failed']} | {summary['missing']} | {summary['error']} | {success_rate:.1f}% |\n")
        
        # Detailed results by module
        f.write("\n## Detailed Results by Module\n\n")
        
        # Create comparison table
        module_names = sorted(ALL_MODULES_CONFIG, key=lambda x: x['module'])
        
        f.write("| Module | Agent Claude | Agent Codex |\n")
        f.write("|--------|-------------|-------------|\n")
        
        for module_config in module_names:
            module_name = module_config['module']
            f.write(f"| {module_name} |")
            
            for results in all_results:
                if 'claude' in results['candidate']:
                    module_result = results['modules'].get(module_name, {})
                    status = module_result.get('status', 'unknown')
                    f.write(f" {status} |")
                else:
                    f.write("")  # Leave empty, will be filled by codex
            
            for results in all_results:
                if 'codex' in results['candidate']:
                    module_result = results['modules'].get(module_name, {})
                    status = module_result.get('status', 'unknown')
                    f.write(f" {status} |")
            
            f.write("\n")
        
        # Pattern analysis
        f.write("\n## Pattern Analysis\n\n")
        
        for results in all_results:
            f.write(f"### {results['candidate']}\n\n")
            
            patterns = results['patterns']
            
            f.write("**Initialization Strategies:**\n")
            for strategy, count in patterns['by_initialization_strategy'].items():
                f.write(f"- {strategy}: {count} modules\n")
            
            f.write("\n**Common Error Types:**\n")
            for error_type, count in sorted(patterns['by_error_type'].items(), key=lambda x: x[1], reverse=True):
                f.write(f"- {error_type}: {count} occurrences\n")
            
            f.write("\n**Test Type Failures:**\n")
            for test_type, count in sorted(patterns['by_test_type_failure'].items(), key=lambda x: x[1], reverse=True):
                f.write(f"- {test_type}: {count} failures\n")
            
            f.write("\n")


def main():
    """Main test execution function."""
    print("🧪 COMPREHENSIVE MODULE TESTING - ALL 25 MODULES")
    print("Testing both agent_claude and agent_codex with flexible utilities\n")
    
    # Test each candidate
    candidates = [
        'candidates/agent_claude',
        'candidates/agent_codex'
    ]
    
    all_results = []
    
    for candidate_dir in candidates:
        if os.path.exists(candidate_dir):
            results = test_candidate_comprehensive(candidate_dir, ALL_MODULES_CONFIG)
            all_results.append(results)
            
            # Save detailed results
            results_dir = Path('test_results')
            results_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            candidate_name = Path(candidate_dir).name
            results_file = results_dir / f"{candidate_name}_comprehensive_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\nDetailed results saved to: {results_file}")
    
    # Analyze patterns
    analyze_patterns(all_results)
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    for results in all_results:
        candidate = results['candidate']
        summary = results['summary']
        total = summary['total']
        passed = summary['passed']
        partial = summary['partial']
        success = passed + partial
        
        print(f"\n{candidate}:")
        print(f"  Total modules tested: {total}")
        print(f"  ✅ Fully passed: {passed} ({passed/total*100:.1f}%)")
        print(f"  ⚠️  Partially passed: {partial} ({partial/total*100:.1f}%)")
        print(f"  ❌ Failed: {summary['failed']} ({summary['failed']/total*100:.1f}%)")
        print(f"  📭 Missing: {summary['missing']} ({summary['missing']/total*100:.1f}%)")
        print(f"  🚨 Error: {summary['error']} ({summary['error']/total*100:.1f}%)")
        print(f"  Overall success rate: {success/total*100:.1f}%")
    
    # Generate summary report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = Path('test_results') / f"comprehensive_summary_{timestamp}.md"
    generate_summary_report(all_results, summary_file)
    print(f"\nSummary report saved to: {summary_file}")
    
    # Compare candidates
    if len(all_results) == 2:
        print("\n" + "="*80)
        print("HEAD-TO-HEAD COMPARISON")
        print("="*80)
        
        claude_results = all_results[0]
        codex_results = all_results[1]
        
        # Find modules where one succeeded and the other failed
        claude_better = []
        codex_better = []
        both_good = []
        both_bad = []
        
        for module_name in claude_results['modules']:
            claude_status = claude_results['modules'][module_name]['status']
            codex_status = codex_results['modules'].get(module_name, {}).get('status', 'missing')
            
            if claude_status in ['passed', 'partial'] and codex_status in ['failed', 'missing', 'error']:
                claude_better.append(module_name)
            elif codex_status in ['passed', 'partial'] and claude_status in ['failed', 'missing', 'error']:
                codex_better.append(module_name)
            elif claude_status in ['passed', 'partial'] and codex_status in ['passed', 'partial']:
                both_good.append(module_name)
            else:
                both_bad.append(module_name)
        
        print(f"\nModules where Agent Claude performs better: {len(claude_better)}")
        for module in claude_better[:5]:  # Show first 5
            print(f"  - {module}")
        if len(claude_better) > 5:
            print(f"  ... and {len(claude_better) - 5} more")
        
        print(f"\nModules where Agent Codex performs better: {len(codex_better)}")
        for module in codex_better[:5]:  # Show first 5
            print(f"  - {module}")
        if len(codex_better) > 5:
            print(f"  ... and {len(codex_better) - 5} more")
        
        print(f"\nModules where both succeed: {len(both_good)}")
        print(f"Modules where both fail: {len(both_bad)}")


if __name__ == "__main__":
    main()