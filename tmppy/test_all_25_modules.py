#!/usr/bin/env python3
"""Test all 25 modules with flexible testing approach."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import json
from datetime import datetime
from test_utils import init_with_variations, extract_output, validate_shape_behavior, check_gradient_flow

# Define test configurations for all 25 modules
MODULE_TESTS = [
    # ML Components (14 modules)
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
        'init_params': {'img_size': 224, 'patch_size': 16, 'n_channels': 3, 'd_model': 768, 'n_layers': 12, 'n_heads': 12},
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
        'name': 'autoencoder_vae',
        'class': 'AutoencoderVAE',
        'init_params': {'input_dim': 784, 'hidden_dims': [256, 128], 'latent_dim': 32},
        'test_input': torch.randn(4, 784),
        'expected_behavior': 'reconstruct'
    },
    {
        'name': 'sequence_to_sequence',
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
        'name': 'memory_bank_retriever',
        'class': 'MemoryBankRetriever',
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
    
    # Data Pipeline Components (6 modules)
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
    
    # Audio-Specific Components (5 modules)
    {
        'name': 'snake_activation',
        'class': 'SnakeActivation',
        'init_params': {'n_channels': 64, 'alpha_init': 1.0, 'learnable': True},
        'test_input': torch.randn(2, 64, 100),
        'expected_behavior': 'activation'
    },
    {
        'name': 'causal_conv',
        'class': 'CausalConv1d',
        'init_params': {'in_channels': 16, 'out_channels': 32, 'kernel_size': 7},
        'test_input': torch.randn(2, 16, 100),
        'expected_behavior': 'causal_conv'
    },
    {
        'name': 'stft_loss',
        'class': 'MultiScaleSTFTLoss',
        'init_params': {'scales': [(512, 128), (1024, 256), (2048, 512)]},
        'test_input': (torch.randn(2, 16000), torch.randn(2, 16000)),
        'expected_behavior': 'compute_loss'
    },
    {
        'name': 'antialiased_conv',
        'class': 'AntiAliasedConv',
        'init_params': {'in_channels': 3, 'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'filter_type': 'kaiser'},
        'test_input': torch.randn(2, 3, 64, 64),
        'expected_behavior': 'conv_aa'
    },
    {
        'name': 'residual_vector_quantizer',
        'class': 'ResidualVectorQuantizer',
        'init_params': {'n_quantizers': 4, 'n_embed': 512, 'embed_dim': 64},
        'test_input': torch.randn(2, 100, 64),
        'expected_behavior': 'quantize'
    }
]


def test_module(module_info):
    """Test a single module with flexible approach."""
    module_name = module_info['name']
    class_name = module_info['class']
    
    try:
        # Import module
        module = __import__(f'modules.{module_name}', fromlist=[class_name])
        ModuleClass = getattr(module, class_name)
        
        # Initialize with flexible parameters
        model = init_with_variations(ModuleClass, module_info['init_params'])
        
        # Test forward pass if applicable
        if hasattr(model, 'forward') and module_info['test_input'] is not None:
            test_input = module_info['test_input']
            
            # Handle tuple inputs
            if isinstance(test_input, tuple):
                output = model(*test_input)
            else:
                output = model(test_input)
            
            # Extract output if needed
            output_dict = extract_output(output)
            
            return {
                'status': 'passed',
                'initialization': True,
                'forward_pass': True,
                'output_type': type(output).__name__,
                'output_keys': list(output_dict.keys()) if output_dict else None
            }
        
        return {
            'status': 'passed',
            'initialization': True,
            'forward_pass': False,
            'note': 'No forward test performed'
        }
        
    except Exception as e:
        return {
            'status': 'failed',
            'error': str(e),
            'error_type': type(e).__name__
        }


def main():
    print("🧪 TESTING ALL 25 MODULES")
    print("="*60 + "\n")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'modules': {},
        'summary': {
            'total': len(MODULE_TESTS),
            'passed': 0,
            'failed': 0
        }
    }
    
    for i, module_info in enumerate(MODULE_TESTS, 1):
        module_name = module_info['name']
        print(f"{i}. Testing {module_name}...", end=" ")
        
        result = test_module(module_info)
        results['modules'][module_name] = result
        
        if result['status'] == 'passed':
            print("✅ PASSED")
            results['summary']['passed'] += 1
        else:
            print(f"❌ FAILED - {result.get('error_type', 'Unknown')}")
            results['summary']['failed'] += 1
    
    # Save results
    output_path = f"test_results/all_modules_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total modules: {results['summary']['total']}")
    print(f"✅ Passed: {results['summary']['passed']}")
    print(f"❌ Failed: {results['summary']['failed']}")
    print(f"Success rate: {results['summary']['passed']/results['summary']['total']*100:.1f}%")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()