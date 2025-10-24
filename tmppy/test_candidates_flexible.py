#!/usr/bin/env python3
"""
Test candidate implementations using flexible testing utilities.
This shows real-world results with external implementations.
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime

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

import torch
import torch.nn as nn


def test_module_flexible(module_path, module_name, class_name, test_config):
    """Test a module flexibly."""
    results = {
        'module': module_name,
        'tests_passed': [],
        'tests_failed': [],
        'status': 'unknown'
    }
    
    # Check if module file exists
    if not os.path.exists(module_path):
        results['status'] = 'missing'
        results['error'] = f"Module file not found: {module_path}"
        return results
    
    # Temporarily add module to path
    original_module = None
    backup_path = None
    try:
        # Backup original if it exists
        original_module_path = Path('modules') / f"{module_name}.py"
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
                break
        
        if ModuleClass is None:
            results['status'] = 'error'
            results['error'] = f"Class {class_name} not found. Available: {dir(module)}"
            return results
        
        # Initialize with flexible parameters
        try:
            model = init_with_variations(ModuleClass, test_config['init_params'])
            results['tests_passed'].append('initialization')
        except Exception as e:
            results['status'] = 'failed'
            results['tests_failed'].append(('initialization', str(e)))
            return results
        
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
                        results['tests_failed'].append((test_name, "Method not found"))
                        
                elif test_case['type'] == 'callable':
                    is_callable, result, msg = test_callable_interface(
                        model, test_case.get('args', ()), test_case.get('kwargs', {})
                    )
                    if is_callable:
                        results['tests_passed'].append(test_name)
                    else:
                        results['tests_failed'].append((test_name, msg))
                        
            except Exception as e:
                results['tests_failed'].append((test_name, str(e)))
        
        # Determine overall status
        if len(results['tests_passed']) > 0 and len(results['tests_failed']) == 0:
            results['status'] = 'passed'
        elif len(results['tests_passed']) > 0:
            results['status'] = 'partial'
        else:
            results['status'] = 'failed'
            
    except Exception as e:
        results['status'] = 'error'
        results['error'] = str(e)
        
    finally:
        # Restore original module
        if original_module_path.exists() or original_module_path.is_symlink():
            original_module_path.unlink()
        if backup_path and backup_path.exists():
            backup_path.rename(original_module_path)
    
    return results


def test_candidate(candidate_dir, modules_to_test):
    """Test a candidate's implementations."""
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
        }
    }
    
    print(f"\n{'='*60}")
    print(f"Testing candidate: {candidate_dir}")
    print(f"{'='*60}")
    
    for module_config in modules_to_test:
        module_name = module_config['module']
        module_path = Path(candidate_dir) / f"{module_name}.py"
        
        print(f"\nTesting {module_name}...")
        result = test_module_flexible(
            module_path, 
            module_name,
            module_config['class'],
            module_config
        )
        
        results['modules'][module_name] = result
        results['summary']['total'] += 1
        results['summary'][result['status']] += 1
        
        # Print result
        if result['status'] == 'passed':
            print(f"  ✅ PASSED - All tests passed")
        elif result['status'] == 'partial':
            print(f"  ⚠️  PARTIAL - Passed {len(result['tests_passed'])}, Failed {len(result['tests_failed'])}")
        elif result['status'] == 'failed':
            print(f"  ❌ FAILED - {result.get('error', 'All tests failed')}")
        elif result['status'] == 'missing':
            print(f"  📭 MISSING - Module file not found")
        else:
            print(f"  🚨 ERROR - {result.get('error', 'Unknown error')}")
    
    return results


# Test configurations for key modules
MODULES_TO_TEST = [
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
            {'name': 'shape_reduce', 'type': 'shape', 'input': torch.randn(2, 3, 64, 64), 'expected': 'reduce'}
        ]
    },
    {
        'module': 'data_versioner',
        'class': 'DataVersioner',
        'init_params': {},
        'tests': [
            {'name': 'callable', 'type': 'callable', 'args': (torch.randn(10, 10), "test")},
            {'name': 'load_method', 'type': 'method', 'method': 'load'}
        ]
    },
    {
        'module': 'stream_joiner',
        'class': 'StreamJoiner',
        'init_params': {
            'join_type': 'inner',
            'time_window': 1.0,
            'buffer_size': 100
        },
        'tests': [
            {'name': 'forward', 'type': 'forward', 'args': ["stream1", torch.randn(4), 1.0]}
        ]
    },
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
            {'name': 'gradient', 'type': 'gradient', 'input': torch.randn(2, 64, 100)}
        ]
    }
]


def main():
    print("🧪 FLEXIBLE CANDIDATE TESTING")
    print("Testing external implementations with flexible utilities\n")
    
    # Test each candidate
    candidates = [
        'candidates/agent_claude',
        'candidates/agent_codex'
    ]
    
    all_results = []
    
    for candidate_dir in candidates:
        if os.path.exists(candidate_dir):
            results = test_candidate(candidate_dir, MODULES_TO_TEST)
            all_results.append(results)
            
            # Save results
            results_dir = Path('test_results')
            results_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            candidate_name = Path(candidate_dir).name
            results_file = results_dir / f"{candidate_name}_flexible_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\nResults saved to: {results_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    
    for results in all_results:
        candidate = results['candidate']
        summary = results['summary']
        total = summary['total']
        passed = summary['passed']
        partial = summary['partial']
        success = passed + partial
        
        print(f"\n{candidate}:")
        print(f"  Total modules: {total}")
        print(f"  ✅ Passed: {passed}")
        print(f"  ⚠️  Partial: {partial}")
        print(f"  ❌ Failed: {summary['failed']}")
        print(f"  📭 Missing: {summary['missing']}")
        print(f"  🚨 Error: {summary['error']}")
        print(f"  Success rate: {success/total*100:.1f}% (vs 0% with rigid tests)")


if __name__ == "__main__":
    main()