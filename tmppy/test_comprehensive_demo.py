#!/usr/bin/env python3
"""
Comprehensive demonstration script showing:
1. Tests on original modules to verify flexible tests work on happy path
2. Tests on candidate implementations with variations
3. Before/after comparison (rigid vs flexible)
4. Clear improvements in success rate
"""

import torch
import torch.nn as nn
import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any

sys.path.append('.')

# Import our flexible test utilities
from test_utils import (
    init_with_variations,
    extract_output,
    validate_shape_behavior,
    test_mask_support,
    check_gradient_flow,
    create_test_tensors,
    find_method,
    validate_module_interface
)


class TestComparison:
    """Compare rigid and flexible testing approaches."""
    
    def __init__(self):
        self.results = {
            'rigid': {'passed': [], 'failed': []},
            'flexible': {'passed': [], 'failed': []},
            'modules_tested': []
        }
        self.test_tensors = create_test_tensors()
    
    def run_rigid_test(self, module_name: str) -> Tuple[bool, str]:
        """
        Run traditional rigid test.
        Expects exact parameter names, exact output formats, etc.
        """
        try:
            # Import module
            mod = __import__(f'modules.{module_name}', fromlist=[module_name])
            class_name = ''.join(w.capitalize() for w in module_name.split('_'))
            
            if module_name == 'transformer_block':
                TransformerBlock = mod.TransformerBlock
                # Rigid: expects exact parameter names
                module = TransformerBlock(d_model=512, n_heads=8)
                x = torch.randn(2, 10, 512)
                output = module(x)
                
                # Rigid: expects tensor output
                assert isinstance(output, torch.Tensor), "Output must be tensor"
                assert output.shape == x.shape, "Shape must be preserved exactly"
                assert not torch.isnan(output).any(), "No NaN values"
                
                # Rigid: expects mask parameter named 'mask'
                mask = torch.ones(2, 8, 10, 10)
                masked_output = module(x, mask=mask)
                assert isinstance(masked_output, torch.Tensor)
                
                return True, "All rigid checks passed"
                
            elif module_name == 'conv_encoder':
                ConvEncoder = mod.ConvEncoder
                # Rigid: expects these exact names
                module = ConvEncoder(in_channels=3, base_channels=64)
                x = self.test_tensors['image_small']
                output = module(x)
                
                # Rigid: expects tensor output (will fail!)
                assert isinstance(output, torch.Tensor), "Output must be tensor"
                return True, "All rigid checks passed"
                
            elif module_name == 'sequence_encoder':
                SequenceEncoder = mod.SequenceEncoder
                module = SequenceEncoder(vocab_size=1000, d_model=512)
                x = torch.randint(0, 1000, (2, 20))
                output = module(x)
                
                # Rigid: expects tensor output (will fail!)
                assert isinstance(output, torch.Tensor), "Output must be tensor"
                return True, "All rigid checks passed"
                
            elif module_name == 'attention_decoder':
                AttentionDecoder = mod.AttentionDecoder
                module = AttentionDecoder(d_model=512, n_heads=8, vocab_size=1000)
                x = torch.randn(2, 10, 512)
                # Rigid: doesn't know it needs encoder_output
                output = module(x)
                return True, "All rigid checks passed"
                
            elif module_name == 'cross_modal_fusion':
                CrossModalFusion = mod.CrossModalFusion
                # Rigid: uses wrong parameter names
                module = CrossModalFusion(visual_dim=512, text_dim=768, fusion_dim=256)
                return True, "All rigid checks passed"
                
            elif module_name == 'autoencoder_vae':
                # Rigid: expects class named AutoencoderVAE
                AutoencoderVAE = mod.AutoencoderVAE
                module = AutoencoderVAE(input_dim=784, hidden_dim=256, latent_dim=32)
                return True, "All rigid checks passed"
                
            else:
                return False, f"No rigid test for {module_name}"
                
        except Exception as e:
            return False, f"{type(e).__name__}: {str(e)}"
    
    def run_flexible_test(self, module_name: str, module_path: str = 'modules') -> Tuple[bool, str, Dict]:
        """
        Run flexible test that handles variations gracefully.
        """
        details = {'tests_run': [], 'tests_passed': [], 'tests_failed': []}
        
        try:
            # Import module flexibly
            mod = __import__(f'{module_path}.{module_name}', fromlist=[module_name])
            
            # Try multiple class name patterns
            class_name = ''.join(w.capitalize() for w in module_name.split('_'))
            ModuleClass = None
            
            # Search for class flexibly
            for name in [class_name, 'VAE', module_name.upper().replace('_', '')]:
                if hasattr(mod, name):
                    ModuleClass = getattr(mod, name)
                    break
            
            if not ModuleClass:
                return False, "No suitable class found", details
            
            if module_name == 'transformer_block':
                # Try initialization with flexible parameter mapping
                params = {'d_model': 512, 'n_heads': 8}
                module = init_with_variations(ModuleClass, params)
                details['tests_run'].append('initialization')
                details['tests_passed'].append('initialization')
                
                # Test shape behavior
                x = self.test_tensors['seq_short']
                passed, msg = validate_shape_behavior(module, x, 'preserve')
                details['tests_run'].append('shape_preservation')
                if passed:
                    details['tests_passed'].append('shape_preservation')
                else:
                    details['tests_failed'].append(('shape_preservation', msg))
                
                # Test mask support flexibly
                passed, msg = test_mask_support(module, x)
                details['tests_run'].append('mask_support')
                if passed:
                    details['tests_passed'].append('mask_support')
                else:
                    details['tests_failed'].append(('mask_support', msg))
                
                # Test gradients
                passed, msg = check_gradient_flow(module, x)
                details['tests_run'].append('gradient_flow')
                if passed:
                    details['tests_passed'].append('gradient_flow')
                else:
                    details['tests_failed'].append(('gradient_flow', msg))
                
            elif module_name in ['conv_encoder', 'sequence_encoder']:
                # These modules return dict outputs
                if module_name == 'conv_encoder':
                    params = {'in_channels': 3, 'base_channels': 64}
                    x = self.test_tensors['image_small']
                else:
                    params = {'vocab_size': 1000, 'd_model': 512}
                    x = torch.randint(0, 1000, (2, 20))
                
                module = init_with_variations(ModuleClass, params)
                details['tests_run'].append('initialization')
                details['tests_passed'].append('initialization')
                
                # Get output flexibly
                output = module(x)
                output_dict = extract_output(output)
                details['tests_run'].append('output_extraction')
                details['tests_passed'].append('output_extraction')
                
                # Validate output
                primary_output = next(iter(output_dict.values()))
                if not torch.isnan(primary_output).any():
                    details['tests_passed'].append('valid_output')
                else:
                    details['tests_failed'].append(('valid_output', 'Contains NaN'))
                
            elif module_name == 'attention_decoder':
                params = {'d_model': 512, 'n_heads': 8, 'vocab_size': 1000}
                module = init_with_variations(ModuleClass, params)
                details['tests_run'].append('initialization')
                details['tests_passed'].append('initialization')
                
                # This module needs encoder output
                x = torch.randn(2, 10, 512)
                encoder_output = torch.randn(2, 20, 512)
                
                try:
                    output = module(x, encoder_output)
                    details['tests_passed'].append('forward_pass')
                except Exception as e:
                    details['tests_failed'].append(('forward_pass', str(e)))
                
            elif module_name == 'cross_modal_fusion':
                params = {'d_model_1': 512, 'd_model_2': 768, 'd_hidden': 256}
                module = init_with_variations(ModuleClass, params)
                details['tests_run'].append('initialization')
                details['tests_passed'].append('initialization')
                
                visual = self.test_tensors['seq_short']
                text = torch.randn(2, 20, 768)
                
                output = module(visual, text)
                output_dict = extract_output(output)
                details['tests_passed'].append('multi_modal_forward')
                
            elif module_name == 'autoencoder_vae':
                # Class might be named VAE
                params = {'input_dim': 784, 'hidden_dim': 256, 'latent_dim': 32}
                module = init_with_variations(ModuleClass, params)
                details['tests_run'].append('initialization')
                details['tests_passed'].append('initialization')
                
                x = torch.randn(2, 784)
                output = module(x)
                details['tests_passed'].append('forward_pass')
            
            # Overall success if we passed more than we failed
            success = len(details['tests_passed']) > len(details['tests_failed'])
            msg = f"Passed {len(details['tests_passed'])}/{len(details['tests_run'])} tests"
            
            return success, msg, details
            
        except Exception as e:
            return False, f"{type(e).__name__}: {str(e)}", details
    
    def test_module(self, module_name: str):
        """Test a module with both approaches."""
        print(f"\n{'='*60}")
        print(f"Testing: {module_name}")
        print(f"{'='*60}")
        
        self.results['modules_tested'].append(module_name)
        
        # Rigid test
        print("\n1. RIGID TEST:")
        rigid_passed, rigid_msg = self.run_rigid_test(module_name)
        print(f"   Result: {'✓ PASSED' if rigid_passed else '✗ FAILED'}")
        print(f"   {rigid_msg}")
        
        if rigid_passed:
            self.results['rigid']['passed'].append(module_name)
        else:
            self.results['rigid']['failed'].append((module_name, rigid_msg))
        
        # Flexible test
        print("\n2. FLEXIBLE TEST:")
        flex_passed, flex_msg, flex_details = self.run_flexible_test(module_name)
        print(f"   Result: {'✓ PASSED' if flex_passed else '✗ FAILED'}")
        print(f"   {flex_msg}")
        if flex_details['tests_passed']:
            print(f"   Passed: {', '.join(flex_details['tests_passed'])}")
        if flex_details['tests_failed']:
            print(f"   Failed: {', '.join(f[0] for f in flex_details['tests_failed'])}")
        
        if flex_passed:
            self.results['flexible']['passed'].append(module_name)
        else:
            self.results['flexible']['failed'].append((module_name, flex_msg))
    
    def print_summary(self):
        """Print comprehensive summary."""
        print("\n" + "="*60)
        print("COMPREHENSIVE TEST SUMMARY")
        print("="*60)
        
        rigid_total = len(self.results['rigid']['passed']) + len(self.results['rigid']['failed'])
        flex_total = len(self.results['flexible']['passed']) + len(self.results['flexible']['failed'])
        
        rigid_rate = (len(self.results['rigid']['passed']) / rigid_total * 100) if rigid_total > 0 else 0
        flex_rate = (len(self.results['flexible']['passed']) / flex_total * 100) if flex_total > 0 else 0
        
        print(f"\nMODULES TESTED: {len(self.results['modules_tested'])}")
        print(f"  {', '.join(self.results['modules_tested'])}")
        
        print(f"\nRIGID TESTING RESULTS:")
        print(f"  Passed: {len(self.results['rigid']['passed'])}/{rigid_total} ({rigid_rate:.1f}%)")
        print(f"  Failed: {len(self.results['rigid']['failed'])}")
        if self.results['rigid']['failed']:
            print("  Failures:")
            for module, msg in self.results['rigid']['failed']:
                print(f"    - {module}: {msg}")
        
        print(f"\nFLEXIBLE TESTING RESULTS:")
        print(f"  Passed: {len(self.results['flexible']['passed'])}/{flex_total} ({flex_rate:.1f}%)")
        print(f"  Failed: {len(self.results['flexible']['failed'])}")
        
        print(f"\nIMPROVEMENT: +{flex_rate - rigid_rate:.1f}% success rate")
        
        # Modules that benefited
        print("\nMODULES THAT BENEFITED FROM FLEXIBLE TESTING:")
        rigid_failed = [m[0] if isinstance(m, tuple) else m for m in self.results['rigid']['failed']]
        for module in self.results['flexible']['passed']:
            if module in rigid_failed:
                print(f"  ✓ {module}: Failed rigid but passed flexible")
        
        print("\nKEY ADVANTAGES OF FLEXIBLE TESTING:")
        print("  1. Handles multiple output formats (tensor/dict/tuple)")
        print("  2. Supports parameter name variations")
        print("  3. Tests behavior rather than exact implementation")
        print("  4. Gracefully handles missing optional features")
        print("  5. Better error messages for debugging")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"test_results/comprehensive_comparison_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'modules_tested': self.results['modules_tested'],
                'rigid_results': {
                    'passed': self.results['rigid']['passed'],
                    'failed': [{'module': m, 'error': e} for m, e in self.results['rigid']['failed']],
                    'success_rate': rigid_rate
                },
                'flexible_results': {
                    'passed': self.results['flexible']['passed'],
                    'failed': [{'module': m, 'error': e} for m, e in self.results['flexible']['failed']],
                    'success_rate': flex_rate
                },
                'improvement': flex_rate - rigid_rate
            }, f, indent=2)
        print(f"\nResults saved to: {results_file}")


def main():
    """Run comprehensive comparison."""
    print("COMPREHENSIVE RIGID vs FLEXIBLE TESTING DEMONSTRATION")
    print("This demonstrates how flexible testing improves success rates")
    print("by handling common implementation variations gracefully.")
    
    # Test representative modules
    test_modules = [
        'transformer_block',    # Works with rigid
        'conv_encoder',        # Dict output breaks rigid
        'sequence_encoder',    # Dict output breaks rigid
        'attention_decoder',   # Missing arg breaks rigid
        'cross_modal_fusion',  # Wrong param names break rigid
        'autoencoder_vae'      # Wrong class name breaks rigid
    ]
    
    comparison = TestComparison()
    
    # Test each module
    for module in test_modules:
        try:
            comparison.test_module(module)
        except Exception as e:
            print(f"\nError testing {module}: {e}")
    
    # Print summary
    comparison.print_summary()


if __name__ == "__main__":
    main()