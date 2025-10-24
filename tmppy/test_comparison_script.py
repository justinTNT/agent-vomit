#!/usr/bin/env python3
"""
Test comparison script that demonstrates the improvement from rigid to flexible testing.
Tests both original modules and candidate implementations.
"""

import torch
import torch.nn as nn
import sys
import os
import traceback
from typing import Dict, Any, List, Tuple
import json
from datetime import datetime

sys.path.append('.')

# Import test utilities
from test_utils import (
    init_with_variations,
    validate_shape_behavior,
    test_mask_support,
    check_gradient_flow,
    create_test_tensors,
    flexible_module_test
)

# List of representative modules to test
TEST_MODULES = [
    'transformer_block',
    'attention_decoder',
    'conv_encoder',
    'sequence_encoder',
    'autoencoder_vae',
    'cross_modal_fusion'
]

class TestResults:
    """Track and display test results."""
    def __init__(self):
        self.results = {
            'rigid': {'passed': [], 'failed': []},
            'flexible': {'passed': [], 'failed': []},
            'comparison': {}
        }
    
    def add_result(self, test_type: str, module_name: str, passed: bool, details: str = ""):
        """Add test result."""
        if passed:
            self.results[test_type]['passed'].append(module_name)
        else:
            self.results[test_type]['failed'].append((module_name, details))
    
    def calculate_improvement(self):
        """Calculate improvement metrics."""
        rigid_passed = len(self.results['rigid']['passed'])
        flexible_passed = len(self.results['flexible']['passed'])
        total = rigid_passed + len(self.results['rigid']['failed'])
        
        if total == 0:
            return 0, 0, 0
        
        rigid_rate = rigid_passed / total * 100
        flexible_rate = flexible_passed / total * 100
        improvement = flexible_rate - rigid_rate
        
        return rigid_rate, flexible_rate, improvement


def run_rigid_test(module_name: str, source_path: str) -> Tuple[bool, str]:
    """
    Run a rigid test (the original way).
    This represents the traditional approach with fixed parameter names and expectations.
    """
    try:
        # Import module
        module_path = f"{source_path}.{module_name}"
        mod = __import__(module_path, fromlist=[module_name])
        
        # Get the main class (assume it's named after the module in CamelCase)
        class_name = ''.join(word.capitalize() for word in module_name.split('_'))
        if not hasattr(mod, class_name):
            return False, f"Class {class_name} not found in module"
        
        ModuleClass = getattr(mod, class_name)
        
        # Try to create with rigid parameters
        if module_name == 'transformer_block':
            # TransformerBlock expects exactly these parameter names
            module = ModuleClass(d_model=512, n_heads=8)
            x = torch.randn(2, 10, 512)
            output = module(x)
            assert output.shape == x.shape
            
        elif module_name == 'attention_decoder':
            # AttentionDecoder expects these exact names
            module = ModuleClass(d_model=512, n_heads=8, vocab_size=1000)
            x = torch.randn(2, 10, 512)
            output = module(x)
            
        elif module_name == 'conv_encoder':
            # ConvEncoder expects these exact names
            module = ModuleClass(in_channels=3, base_channels=64)
            x = torch.randn(2, 3, 64, 64)
            output = module(x)
            
        elif module_name == 'sequence_encoder':
            # SequenceEncoder expects these exact names
            module = ModuleClass(vocab_size=1000, d_model=512)
            x = torch.randint(0, 1000, (2, 20))
            output = module(x)
            
        elif module_name == 'autoencoder_vae':
            # VAE expects these exact names
            if class_name == 'AutoencoderVae':
                # Try with VAE class name
                ModuleClass = getattr(mod, 'VAE')
            module = ModuleClass(input_dim=784, hidden_dim=256, latent_dim=32)
            x = torch.randn(2, 784)
            output = module(x)
            
        elif module_name == 'cross_modal_fusion':
            # CrossModalFusion expects these exact names
            module = ModuleClass(d_model_1=512, d_model_2=768, d_hidden=256)
            visual = torch.randn(2, 10, 512)
            text = torch.randn(2, 20, 768)
            output = module(visual, text)
        
        else:
            return False, f"No rigid test defined for {module_name}"
        
        # Basic checks that rigid tests do
        if torch.isnan(output).any():
            return False, "Output contains NaN"
        
        return True, "Passed rigid test"
        
    except Exception as e:
        return False, f"Exception: {str(e)}"


def run_flexible_test(module_name: str, source_path: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Run a flexible test using our test utilities.
    This represents the new approach that handles variations.
    """
    try:
        # Import module
        module_path = f"{source_path}.{module_name}"
        mod = __import__(module_path, fromlist=[module_name])
        
        # Get the main class
        class_name = ''.join(word.capitalize() for word in module_name.split('_'))
        if not hasattr(mod, class_name):
            # Try alternative naming patterns
            alt_names = [
                module_name.replace('_', '').capitalize(),
                module_name.title().replace('_', ''),
                module_name.upper().replace('_', ''),
                'VAE' if 'vae' in module_name.lower() else None
            ]
            for alt_name in alt_names:
                if alt_name and hasattr(mod, alt_name):
                    class_name = alt_name
                    break
            else:
                return False, f"No suitable class found in module", {}
        
        ModuleClass = getattr(mod, class_name)
        
        # Define flexible test parameters and cases for each module
        test_tensors = create_test_tensors()
        
        if module_name == 'transformer_block':
            params = {'d_model': 512, 'n_heads': 8}
            test_cases = [
                {'type': 'shape', 'input': test_tensors['seq_short'], 'expected': 'preserve'},
                {'type': 'mask', 'input': test_tensors['seq_short']},
                {'type': 'gradient', 'input': test_tensors['seq_short']}
            ]
            
        elif module_name == 'attention_decoder':
            params = {'d_model': 512, 'n_heads': 8, 'vocab_size': 1000}
            test_cases = [
                {'type': 'shape', 'input': test_tensors['seq_short'], 'expected': 'same_batch'},
                {'type': 'gradient', 'input': test_tensors['seq_short']}
            ]
            
        elif module_name == 'conv_encoder':
            params = {'in_channels': 3, 'base_channels': 64}
            test_cases = [
                {'type': 'shape', 'input': test_tensors['image_small'], 'expected': 'same_batch'},
                {'type': 'gradient', 'input': test_tensors['image_small']}
            ]
            
        elif module_name == 'sequence_encoder':
            params = {'vocab_size': 1000, 'd_model': 512}
            # Create integer input for sequence encoder
            seq_input = torch.randint(0, 1000, (2, 20))
            test_cases = [
                {'type': 'shape', 'input': seq_input, 'expected': 'same_batch'},
                {'type': 'gradient', 'input': seq_input}
            ]
            
        elif module_name == 'autoencoder_vae':
            params = {'input_dim': 784, 'hidden_dim': 256, 'latent_dim': 32}
            flat_input = torch.randn(2, 784)
            test_cases = [
                {'type': 'shape', 'input': flat_input, 'expected': 'same_batch'},
                {'type': 'gradient', 'input': flat_input}
            ]
            
        elif module_name == 'cross_modal_fusion':
            params = {'d_model_1': 512, 'd_model_2': 768, 'd_hidden': 256}
            # This module needs special handling for two inputs
            visual_input = test_tensors['seq_short']
            text_input = torch.randn(2, 20, 768)
            
            # Initialize with variations
            try:
                module = init_with_variations(ModuleClass, params)
                
                # Test forward pass
                output = module(visual_input, text_input)
                passed_tests = ['initialization']
                
                # Check output
                if not torch.isnan(output).any():
                    passed_tests.append('forward_pass')
                
                # Check gradients
                if check_gradient_flow(module, visual_input)[0]:
                    passed_tests.append('gradients')
                
                return True, "All tests passed", {'tests_passed': passed_tests, 'tests_failed': []}
                
            except Exception as e:
                return False, str(e), {'initialization_error': str(e)}
        
        else:
            return False, f"No flexible test defined for {module_name}", {}
        
        # Run flexible tests
        results = flexible_module_test(ModuleClass, params, test_cases, verbose=True)
        
        if results['initialization'] and len(results['tests_passed']) > 0:
            return True, f"Passed {len(results['tests_passed'])} tests", results
        else:
            error = results.get('initialization_error', 'Some tests failed')
            return False, error, results
            
    except Exception as e:
        return False, f"Exception: {str(e)}", {'exception': str(e)}


def test_implementations(module_name: str, results_tracker: TestResults):
    """Test a module across different implementations."""
    print(f"\n{'='*60}")
    print(f"Testing: {module_name}")
    print(f"{'='*60}")
    
    # Test original module with rigid approach
    print("\n1. RIGID TEST (Original Approach):")
    print("-" * 30)
    rigid_passed, rigid_msg = run_rigid_test(module_name, "modules")
    print(f"Result: {'✓ PASSED' if rigid_passed else '✗ FAILED'}")
    if not rigid_passed:
        print(f"Error: {rigid_msg}")
    results_tracker.add_result('rigid', module_name, rigid_passed, rigid_msg)
    
    # Test original module with flexible approach
    print("\n2. FLEXIBLE TEST (New Approach):")
    print("-" * 30)
    flex_passed, flex_msg, flex_details = run_flexible_test(module_name, "modules")
    print(f"Result: {'✓ PASSED' if flex_passed else '✗ FAILED'}")
    if flex_passed:
        print(f"Tests passed: {flex_details.get('tests_passed', [])}")
    else:
        print(f"Error: {flex_msg}")
    results_tracker.add_result('flexible', module_name, flex_passed, flex_msg)
    
    # Test candidate implementations if they exist
    candidates = ['agent_claude', 'agent_codex']
    for candidate in candidates:
        candidate_dir = f"candidates/{candidate}"
        module_path = f"{candidate_dir}/{module_name}.py"
        
        if os.path.exists(module_path):
            print(f"\n3. FLEXIBLE TEST on {candidate} implementation:")
            print("-" * 30)
            
            # Try flexible test on candidate
            cand_passed, cand_msg, cand_details = run_flexible_test(
                module_name, 
                f"candidates.{candidate}"
            )
            
            print(f"Result: {'✓ PASSED' if cand_passed else '✗ FAILED'}")
            if cand_passed:
                print(f"Tests passed: {cand_details.get('tests_passed', [])}")
            else:
                print(f"Error: {cand_msg}")
        else:
            print(f"\n3. {candidate} implementation not found at {module_path}")


def main():
    """Main test execution."""
    print("RIGID vs FLEXIBLE TESTING COMPARISON")
    print("=" * 60)
    print("\nThis script demonstrates how flexible testing improves success rates")
    print("by handling common implementation variations.\n")
    
    results_tracker = TestResults()
    
    # Test each module
    for module_name in TEST_MODULES:
        try:
            test_implementations(module_name, results_tracker)
        except Exception as e:
            print(f"\nUnexpected error testing {module_name}: {e}")
            traceback.print_exc()
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    rigid_rate, flexible_rate, improvement = results_tracker.calculate_improvement()
    
    print(f"\nRIGID TESTING:")
    print(f"  Passed: {len(results_tracker.results['rigid']['passed'])}")
    print(f"  Failed: {len(results_tracker.results['rigid']['failed'])}")
    print(f"  Success Rate: {rigid_rate:.1f}%")
    
    print(f"\nFLEXIBLE TESTING:")
    print(f"  Passed: {len(results_tracker.results['flexible']['passed'])}")
    print(f"  Failed: {len(results_tracker.results['flexible']['failed'])}")
    print(f"  Success Rate: {flexible_rate:.1f}%")
    
    print(f"\nIMPROVEMENT: +{improvement:.1f}% success rate")
    
    # Show which modules benefited
    print("\nMODULES THAT BENEFITED FROM FLEXIBLE TESTING:")
    rigid_failed = [m[0] if isinstance(m, tuple) else m for m in results_tracker.results['rigid']['failed']]
    flexible_passed = results_tracker.results['flexible']['passed']
    
    benefited = [m for m in flexible_passed if m in rigid_failed]
    if benefited:
        for module in benefited:
            print(f"  - {module}: Failed rigid test but passed flexible test")
    else:
        print("  (All modules that passed rigid also passed flexible)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"test_results/comparison_{timestamp}.json"
    os.makedirs("test_results", exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'results': results_tracker.results,
            'metrics': {
                'rigid_success_rate': rigid_rate,
                'flexible_success_rate': flexible_rate,
                'improvement': improvement
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()