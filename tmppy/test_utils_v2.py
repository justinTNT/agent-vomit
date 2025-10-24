"""
Enhanced test utilities that handle parameter combinations.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Union
import itertools
from test_utils import *  # Import everything from original test_utils


def init_with_combinations(
    ModuleClass: type,
    params: Dict[str, Any],
    variations: Dict[str, List[str]] = None,
    max_combinations: int = 100,
    verbose: bool = False
) -> nn.Module:
    """
    Initialize module by trying combinations of parameter variations.
    
    This is more powerful than init_with_variations which only tries
    one parameter variation at a time.
    """
    variations = variations or PARAMETER_VARIATIONS
    
    # First try original
    try:
        if verbose:
            print(f"Trying original params: {params}")
        return ModuleClass(**params)
    except TypeError as e:
        original_error = str(e)
    
    # Get all parameters that have variations
    params_with_variations = []
    for param_name in params:
        if param_name in variations:
            params_with_variations.append((param_name, variations[param_name]))
    
    if not params_with_variations:
        raise TypeError(f"No parameter variations available. {original_error}")
    
    # Generate combinations
    param_names = [p[0] for p in params_with_variations]
    param_options = [p[1] for p in params_with_variations]
    
    # Try each combination
    attempts = 0
    for combination in itertools.product(*param_options):
        if attempts >= max_combinations:
            break
        attempts += 1
        
        # Build parameter dict with this combination
        test_params = params.copy()
        for i, (orig_name, new_name) in enumerate(zip(param_names, combination)):
            if orig_name in test_params and new_name != orig_name:
                test_params[new_name] = test_params.pop(orig_name)
        
        # Try this combination
        try:
            if verbose and attempts <= 10:  # Only print first 10
                print(f"Trying combination {attempts}: {list(test_params.keys())}")
            return ModuleClass(**test_params)
        except TypeError:
            continue
    
    raise TypeError(
        f"Could not initialize {ModuleClass.__name__} after {attempts} combinations.\n"
        f"Original error: {original_error}"
    )


def test_with_smart_init(ModuleClass, canonical_params, test_fn, verbose=False):
    """
    High-level test function that tries increasingly sophisticated initialization.
    """
    # Strategy 1: Try original params
    try:
        model = ModuleClass(**canonical_params)
        if verbose:
            print("✅ Initialized with original params")
        return test_fn(model), "original"
    except:
        pass
    
    # Strategy 2: Try simple variations (one at a time)
    try:
        model = init_with_variations(ModuleClass, canonical_params, verbose=False)
        if verbose:
            print("✅ Initialized with simple variations")
        return test_fn(model), "variations"
    except:
        pass
    
    # Strategy 3: Try parameter combinations
    try:
        model = init_with_combinations(
            ModuleClass, canonical_params, 
            max_combinations=50, verbose=verbose
        )
        if verbose:
            print("✅ Initialized with parameter combinations")
        return test_fn(model), "combinations"
    except Exception as e:
        if verbose:
            print(f"❌ All initialization strategies failed: {e}")
        raise


def flexible_module_test_v2(
    ModuleClass: type,
    canonical_params: Dict[str, Any],
    test_cases: List[Dict[str, Any]],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Enhanced version that uses smart initialization strategies.
    """
    results = {
        'initialization': False,
        'init_strategy': None,
        'tests_passed': [],
        'tests_failed': [],
        'module': None
    }
    
    # Try initialization with increasing sophistication
    try:
        def run_tests(model):
            results['module'] = model
            results['initialization'] = True
            
            # Run test cases
            for test_case in test_cases:
                test_name = test_case.get('name', 'unnamed')
                try:
                    if test_case['type'] == 'forward':
                        output = model(*test_case.get('args', []))
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
                    
                    elif test_case['type'] == 'gradient':
                        success, msg = check_gradient_flow(model, test_case['input'])
                        if success:
                            results['tests_passed'].append(test_name)
                        else:
                            results['tests_failed'].append((test_name, msg))
                            
                except Exception as e:
                    results['tests_failed'].append((test_name, str(e)))
            
            return results
        
        result, strategy = test_with_smart_init(
            ModuleClass, canonical_params, run_tests, verbose
        )
        results.update(result)
        results['init_strategy'] = strategy
        
    except Exception as e:
        results['initialization_error'] = str(e)
    
    return results