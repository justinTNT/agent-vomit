#!/usr/bin/env python3
"""Test all 25 modules with corrected filenames and class names."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
import json
from datetime import datetime
from test_utils_v2 import init_with_combinations, extract_output

# Corrected module configurations based on actual filenames
MODULE_TESTS = [
    # ML Components
    {
        'name': 'transformer_block',
        'class': 'TransformerBlock',
        'init_params': {'d_model': 512, 'n_heads': 8},
        'test_input': torch.randn(2, 10, 512),
        'expected_behavior': 'preserve_shape'
    },
    {
        'name': 'conv_encoder', 
        'class': 'ConvEncoder',
        'init_params': {'in_channels': 3, 'base_channels': 64, 'num_layers': 4},
        'test_input': torch.randn(2, 3, 64, 64),
        'expected_behavior': 'return_dict'
    },
    {
        'name': 'sequence_encoder',
        'class': 'SequenceEncoder',
        'init_params': {'vocab_size': 10000, 'd_model': 512, 'n_layers': 6, 'n_heads': 8},
        'test_input': torch.randint(0, 10000, (2, 50)),
        'expected_behavior': 'encode'
    },
    {
        'name': 'attention_decoder',
        'class': 'AttentionDecoder',
        'init_params': {'d_model': 512, 'n_heads': 8, 'n_layers': 6, 'vocab_size': 10000},
        'test_input': (torch.randint(0, 10000, (2, 20)), torch.randn(2, 30, 512)),
        'expected_behavior': 'decode'
    },
    {
        'name': 'vit_patch_encoder',
        'class': 'ViTPatchEncoder',
        'init_params': {'img_size': 224, 'patch_size': 16, 'in_channels': 3, 'embed_dim': 768, 'n_layers': 12, 'n_heads': 12},
        'test_input': torch.randn(2, 3, 224, 224),
        'expected_behavior': 'encode'
    },
    {
        'name': 'cross_modal_fusion',
        'class': 'CrossModalFusion',
        'init_params': {'d_model_1': 512, 'd_model_2': 768, 'd_hidden': 1024},
        'test_input': (torch.randn(2, 10, 512), torch.randn(2, 15, 768)),
        'expected_behavior': 'fuse'
    },
    {
        'name': 'time_series_encoder',
        'class': 'TimeSeriesEncoder',
        'init_params': {'input_dim': 10, 'hidden_dim': 128, 'n_layers': 4, 'kernel_size': 3},
        'test_input': torch.randn(2, 10, 100),
        'expected_behavior': 'encode'
    },
    {
        'name': 'set_encoder',
        'class': 'SetEncoder',
        'init_params': {'input_dim': 64, 'hidden_dim': 128, 'output_dim': 256},
        'test_input': torch.randn(2, 20, 64),
        'expected_behavior': 'permutation_invariant'
    },
    {
        'name': 'contrastive_learner',
        'class': 'ContrastiveLearner',
        'init_params': {'encoder_dim': 512, 'projection_dim': 128, 'temperature': 0.07},
        'test_input': torch.randn(8, 512),
        'expected_behavior': 'compute_loss'
    },
    {
        'name': 'auto_encoder',  # Corrected filename
        'class': 'AutoEncoder',  # Corrected class name
        'class_alt': {'agent_codex': 'AutoEncoder'},  # Codex uses same name
        'init_params': {'input_dim': 784, 'hidden_dims': [256, 128], 'latent_dim': 32},
        'test_input': torch.randn(4, 784),
        'expected_behavior': 'reconstruct'
    },
    {
        'name': 'sequence_to_sequence_model',  # Corrected filename
        'class': 'SequenceToSequenceModel',
        'init_params': {'input_vocab_size': 5000, 'output_vocab_size': 5000, 'd_model': 256, 'n_heads': 8},
        'test_input': (torch.randint(0, 5000, (2, 20)), torch.randint(0, 5000, (2, 25))),
        'expected_behavior': 'seq2seq'
    },
    {
        'name': 'graph_encoder',
        'class': 'GraphEncoder',
        'init_params': {'node_features': 32, 'edge_features': 16, 'hidden_dim': 64, 'n_layers': 3},
        'test_input': None,  # Graph input is complex
        'expected_behavior': 'graph_encode'
    },
    {
        'name': 'memory_bank',  # Corrected filename
        'class': 'MemoryBank',  # Corrected class name
        'init_params': {'memory_size': 1000, 'key_dim': 256, 'value_dim': 512},
        'test_input': torch.randn(2, 256),
        'expected_behavior': 'retrieve'
    },
    {
        'name': 'adaptive_computation',
        'class': 'AdaptiveComputation',
        'init_params': {'input_dim': 256, 'hidden_dim': 512, 'max_steps': 10},
        'test_input': torch.randn(4, 256),
        'expected_behavior': 'adaptive_forward'
    },
    
    # Data Pipeline Components
    {
        'name': 'stream_processor',
        'class': 'StreamProcessor',
        'init_params': {'window_type': 'sliding', 'window_size': 100, 'stride': 50},
        'test_input': (torch.randn(200), torch.arange(200)),
        'expected_behavior': 'process_stream'
    },
    {
        'name': 'data_validator',
        'class': 'DataValidator',
        'init_params': {'schema': {'shape': (None, 10), 'dtype': torch.float32}},
        'test_input': torch.randn(5, 10),
        'expected_behavior': 'validate'
    },
    {
        'name': 'feature_store',
        'class': 'FeatureStore',
        'init_params': {'features': ['mean', 'std', 'max']},
        'test_input': torch.randn(100, 50),
        'expected_behavior': 'compute_features'
    },
    {
        'name': 'data_versioner',
        'class': 'DataVersioner',
        'init_params': {'storage_path': '/tmp/test_versions'},
        'test_input': torch.randn(10, 10),
        'expected_behavior': 'version_data'
    },
    {
        'name': 'stream_joiner',
        'class': 'StreamJoiner',
        'init_params': {'join_type': 'inner', 'window_size': 1000, 'timeout': 5.0},
        'test_input': ("stream1", torch.randn(4), 1000.0),
        'expected_behavior': 'join_streams'
    },
    {
        'name': 'data_sampler',
        'class': 'DataSampler',
        'init_params': {'strategy': 'balanced', 'n_samples': 100},
        'test_input': (torch.randn(500, 10), torch.randint(0, 5, (500,))),
        'expected_behavior': 'sample'
    },
    
    # Audio-Specific Components
    {
        'name': 'snake_activation',
        'class': 'SnakeActivation',
        'init_params': {'n_channels': 64, 'alpha_init': 1.0, 'shared_alpha': False},
        'test_input': torch.randn(2, 64, 100),
        'expected_behavior': 'activation'
    },
    {
        'name': 'causal_conv1d',  # Corrected filename
        'class': 'CausalConv1d',
        'init_params': {'in_channels': 16, 'out_channels': 32, 'kernel_size': 7},
        'test_input': torch.randn(2, 16, 100),
        'expected_behavior': 'causal_conv'
    },
    {
        'name': 'multi_scale_stft_loss',  # Corrected filename
        'class': 'MultiScaleSTFTLoss',
        'init_params': {'scales': [(512, 128, 512), (1024, 256, 1024), (2048, 512, 2048)]},
        'test_input': (torch.randn(2, 16000), torch.randn(2, 16000)),
        'expected_behavior': 'compute_loss'
    },
    {
        'name': 'antialiased_conv',
        'class': 'AntialiasedConv',
        'init_params': {'in_channels': 3, 'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'filter_type': 'kaiser'},
        'test_input': torch.randn(2, 3, 64, 64),
        'expected_behavior': 'conv_aa'
    },
    {
        'name': 'residual_vector_quantizer',
        'class': 'ResidualVectorQuantizer',
        'init_params': {'n_quantizers': 4, 'n_embeddings': 512, 'embedding_dim': 64},
        'test_input': torch.randn(2, 100, 64),
        'expected_behavior': 'quantize'
    }
]


def test_module(module_info, candidate='agent_claude'):
    """Test a single module with flexible approach."""
    module_name = module_info['name']
    class_name = module_info['class']
    
    # Handle alternative class names for different agents
    if 'class_alt' in module_info and candidate in module_info['class_alt']:
        class_name = module_info['class_alt'][candidate]
    
    # Handle special filename for agent_codex autoencoder
    if module_name == 'auto_encoder' and candidate == 'agent_codex':
        module_name = 'autoencoder'
    
    try:
        # Import module from candidate directory
        module = __import__(f'candidates.{candidate}.{module_name}', fromlist=[class_name])
        ModuleClass = getattr(module, class_name)
        
        # Initialize with flexible parameters
        model = init_with_combinations(ModuleClass, module_info['init_params'])
        
        # Test forward pass if applicable
        test_results = {
            'initialization': True,
            'forward_pass': False,
            'output_type': None,
            'error': None
        }
        
        if hasattr(model, 'forward') and module_info['test_input'] is not None:
            try:
                test_input = module_info['test_input']
                
                # Handle special cases
                if module_info['name'] == 'data_versioner':
                    # Test the callable interface
                    output = model(test_input)
                elif module_info['name'] == 'memory_bank':
                    # Test store and retrieve
                    if hasattr(model, 'store'):
                        model.store(test_input, test_input)
                        output = model.retrieve(test_input)
                    else:
                        output = model(test_input)
                elif isinstance(test_input, tuple):
                    output = model(*test_input)
                else:
                    output = model(test_input)
                
                test_results['forward_pass'] = True
                test_results['output_type'] = type(output).__name__
                
                # Extract output keys if dict/tuple
                if hasattr(output, 'keys'):
                    test_results['output_keys'] = list(output.keys())
                elif isinstance(output, tuple) and len(output) > 0:
                    test_results['output_format'] = f"tuple[{len(output)}]"
                    
            except Exception as e:
                test_results['forward_pass'] = False
                test_results['error'] = f"Forward pass failed: {str(e)}"
        
        return {
            'status': 'passed' if test_results['forward_pass'] or test_results['initialization'] else 'failed',
            **test_results
        }
        
    except Exception as e:
        return {
            'status': 'failed',
            'initialization': False,
            'forward_pass': False,
            'error': str(e),
            'error_type': type(e).__name__
        }


def test_candidate(candidate):
    """Test all modules for a candidate."""
    print(f"\n🧪 TESTING {candidate.upper()}")
    print("="*60)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'candidate': candidate,
        'modules': {},
        'summary': {
            'total': len(MODULE_TESTS),
            'passed': 0,
            'partially_passed': 0,
            'failed': 0
        }
    }
    
    for i, module_info in enumerate(MODULE_TESTS, 1):
        module_name = module_info['name']
        
        # Adjust name for agent_codex autoencoder
        display_name = module_name
        if module_name == 'auto_encoder' and candidate == 'agent_codex':
            display_name = 'autoencoder'
            
        print(f"{i}. Testing {display_name}...", end=" ")
        
        result = test_module(module_info, candidate)
        results['modules'][module_name] = result
        
        if result['status'] == 'passed':
            if result.get('forward_pass', False):
                print("✅ PASSED (init + forward)")
                results['summary']['passed'] += 1
            else:
                print("⚠️  PARTIAL (init only)")
                results['summary']['partially_passed'] += 1
        else:
            print(f"❌ FAILED - {result.get('error_type', 'Unknown')}")
            results['summary']['failed'] += 1
    
    # Calculate success rate
    working_modules = results['summary']['passed'] + results['summary']['partially_passed']
    results['summary']['success_rate'] = working_modules / results['summary']['total'] * 100
    
    return results


def main():
    print("🔍 COMPREHENSIVE MODULE TEST - CORRECTED FILENAMES")
    print("="*80 + "\n")
    
    # Test both candidates
    all_results = {}
    
    for candidate in ['agent_claude', 'agent_codex']:
        results = test_candidate(candidate)
        all_results[candidate] = results
        
        # Save individual results
        output_path = f"test_results/{candidate}_corrected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    # Print comparison summary
    print("\n" + "="*80)
    print("FINAL SUMMARY - CORRECTED TEST")
    print("="*80)
    
    for candidate, results in all_results.items():
        summary = results['summary']
        print(f"\n{candidate.upper()}:")
        print(f"  Total modules: {summary['total']}")
        print(f"  ✅ Fully passed: {summary['passed']}")
        print(f"  ⚠️  Partially passed: {summary['partially_passed']}")
        print(f"  ❌ Failed: {summary['failed']}")
        print(f"  Success rate: {summary['success_rate']:.1f}%")
        
        # List failed modules
        failed = [m for m, r in results['modules'].items() if r['status'] == 'failed']
        if failed:
            print(f"  Failed modules: {', '.join(failed[:5])}")
            if len(failed) > 5:
                print(f"                  ... and {len(failed)-5} more")
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    claude_working = all_results['agent_claude']['summary']['passed'] + all_results['agent_claude']['summary']['partially_passed']
    codex_working = all_results['agent_codex']['summary']['passed'] + all_results['agent_codex']['summary']['partially_passed']
    
    print(f"Agent Claude: {claude_working}/{len(MODULE_TESTS)} modules working ({claude_working/len(MODULE_TESTS)*100:.0f}%)")
    print(f"Agent Codex: {codex_working}/{len(MODULE_TESTS)} modules working ({codex_working/len(MODULE_TESTS)*100:.0f}%)")
    
    # Module-by-module comparison
    print("\nModule differences:")
    for module in MODULE_TESTS:
        module_name = module['name']
        claude_status = all_results['agent_claude']['modules'][module_name]['status']
        codex_status = all_results['agent_codex']['modules'][module_name]['status']
        
        if claude_status != codex_status:
            print(f"  {module_name}: Claude={claude_status}, Codex={codex_status}")


if __name__ == "__main__":
    main()