"""
Flexible test utilities that handle common implementation variations.
These utilities allow tests to validate behavior while being resilient to 
reasonable differences in API design.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
import inspect
import warnings


# Common parameter name variations
PARAMETER_VARIATIONS = {
    # Attention/Transformer parameters
    'n_heads': ['n_heads', 'num_heads', 'heads', 'n_head', 'attention_heads'],
    'd_model': ['d_model', 'hidden_dim', 'embed_dim', 'hidden_size', 'model_dim'],
    'd_model_1': ['d_model_1', 'visual_dim', 'modal1_dim', 'input_dim_1'],
    'd_model_2': ['d_model_2', 'text_dim', 'modal2_dim', 'input_dim_2'],
    'd_hidden': ['d_hidden', 'fusion_dim', 'hidden_dim', 'proj_dim'],
    'd_ff': ['d_ff', 'ff_dim', 'feedforward_dim', 'mlp_dim', 'intermediate_size'],
    'n_layers': ['n_layers', 'num_layers', 'layers', 'depth'],
    
    # Vision parameters
    'img_size': ['img_size', 'image_size', 'input_size', 'resolution'],
    'patch_size': ['patch_size', 'patch_dim', 'patch_resolution'],
    'in_channels': ['in_channels', 'in_chans', 'input_channels', 'channels'],
    'base_channels': ['base_channels', 'base_filters', 'base_ch', 'base_dim'],
    'out_channels': ['out_channels', 'out_chans', 'output_channels'],
    
    # Sequence parameters
    'seq_len': ['seq_len', 'sequence_length', 'max_len', 'max_length'],
    'vocab_size': ['vocab_size', 'n_vocab', 'vocabulary_size'],
    
    # General parameters
    'batch_size': ['batch_size', 'bs', 'batch'],
    'dropout': ['dropout', 'dropout_rate', 'drop_rate', 'p_dropout'],
    'lr': ['lr', 'learning_rate'],
    
    # Snake activation parameters
    'channels': ['channels', 'n_channels', 'num_channels'],
    'alpha': ['alpha', 'alpha_init', 'alpha_initial'],
}

# Common method name variations
METHOD_VARIATIONS = {
    'load': ['load', 'get', 'retrieve', 'fetch', 'read'],
    'save': ['save', 'store', 'put', 'write', 'persist'],
    'create': ['create', 'make', 'new', 'build', 'generate'],
    'reset': ['reset', 'clear', 'flush', 'initialize'],
    'update': ['update', 'refresh', 'modify', 'set'],
}


def init_with_variations(
    ModuleClass: type,
    params: Dict[str, Any],
    variations: Dict[str, List[str]] = None,
    verbose: bool = False
) -> nn.Module:
    """
    Initialize module with parameter name variations.
    
    Args:
        ModuleClass: The module class to instantiate
        params: Dictionary of parameters with canonical names
        variations: Optional custom parameter variations (uses defaults if None)
        verbose: Print attempted variations for debugging
    
    Returns:
        Initialized module
        
    Raises:
        TypeError: If no parameter combination works
    """
    variations = variations or PARAMETER_VARIATIONS
    
    # First try with original parameters
    try:
        if verbose:
            print(f"Trying original params: {params}")
        return ModuleClass(**params)
    except TypeError as e:
        original_error = str(e)
        
    # Try parameter variations
    attempted = []
    for canonical_name, values in params.items():
        if canonical_name in variations:
            for variant in variations[canonical_name]:
                if variant != canonical_name:
                    try:
                        alt_params = params.copy()
                        alt_params[variant] = alt_params.pop(canonical_name)
                        if verbose:
                            print(f"Trying variant: {canonical_name} -> {variant}")
                        attempted.append(alt_params)
                        return ModuleClass(**alt_params)
                    except TypeError:
                        continue
    
    # If we get here, nothing worked
    error_msg = f"Could not initialize {ModuleClass.__name__} with any parameter variation.\n"
    error_msg += f"Original error: {original_error}\n"
    error_msg += f"Attempted variations: {len(attempted)}"
    raise TypeError(error_msg)


def extract_output(
    output: Union[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]],
    expected_keys: List[str] = None,
    primary_key: str = 'output'
) -> Dict[str, torch.Tensor]:
    """
    Convert module output to dictionary format.
    
    Args:
        output: Module output (tensor, dict, or tuple)
        expected_keys: List of expected dictionary keys
        primary_key: Key to use for single tensor output
        
    Returns:
        Dictionary with standardized keys
    """
    # Already a dictionary
    if isinstance(output, dict):
        return output
    
    # Single tensor
    if isinstance(output, torch.Tensor):
        return {primary_key: output}
    
    # Tuple/List of tensors
    if isinstance(output, (tuple, list)):
        if expected_keys and len(expected_keys) == len(output):
            # Map to expected keys
            return {key: val for key, val in zip(expected_keys, output)}
        else:
            # Generic mapping
            result = {primary_key: output[0]}
            for i, val in enumerate(output[1:], 1):
                result[f'output_{i}'] = val
            return result
    
    # Unknown type
    raise ValueError(f"Cannot extract output from type {type(output)}")


def find_method(
    obj: object,
    method_name: str,
    variations: List[str] = None,
    return_name: bool = False
) -> Union[Callable, Tuple[Callable, str], None]:
    """
    Find method by trying common name variations.
    
    Args:
        obj: Object to search for method
        method_name: Canonical method name
        variations: List of name variations to try
        return_name: If True, return (method, actual_name) tuple
        
    Returns:
        Method if found, None otherwise
        If return_name=True, returns (method, name) or (None, None)
    """
    variations = variations or METHOD_VARIATIONS.get(method_name, [method_name])
    
    for name in variations:
        if hasattr(obj, name):
            method = getattr(obj, name)
            if callable(method):
                return (method, name) if return_name else method
    
    return (None, None) if return_name else None


def validate_shape_behavior(
    module: nn.Module,
    input_tensor: torch.Tensor,
    expected_behavior: str = 'preserve',
    dim: Optional[int] = None
) -> Tuple[bool, str]:
    """
    Validate output shape behavior.
    
    Args:
        module: Module to test
        input_tensor: Input tensor
        expected_behavior: 'preserve', 'reduce', 'expand', 'same_batch'
        dim: Specific dimension to check (None checks all)
        
    Returns:
        (success, message) tuple
    """
    output = module(input_tensor)
    if isinstance(output, dict):
        output = next(iter(output.values()))
    
    in_shape = input_tensor.shape
    out_shape = output.shape
    
    if expected_behavior == 'preserve':
        if in_shape == out_shape:
            return True, "Shape preserved"
        return False, f"Shape changed: {in_shape} -> {out_shape}"
    
    elif expected_behavior == 'same_batch':
        if in_shape[0] == out_shape[0]:
            return True, "Batch dimension preserved"
        return False, f"Batch dimension changed: {in_shape[0]} -> {out_shape[0]}"
    
    elif expected_behavior == 'reduce':
        if dim is not None:
            if out_shape[dim] < in_shape[dim]:
                return True, f"Dimension {dim} reduced"
            return False, f"Dimension {dim} not reduced: {in_shape[dim]} -> {out_shape[dim]}"
        else:
            if len(out_shape) < len(in_shape) or any(o < i for o, i in zip(out_shape, in_shape)):
                return True, "Shape reduced"
            return False, "Shape not reduced"
    
    return True, "No specific shape requirement"


def test_mask_support(
    module: nn.Module,
    input_tensor: torch.Tensor,
    mask_type: str = 'auto'
) -> Tuple[bool, str]:
    """
    Test if module properly supports masking.
    
    Args:
        module: Module to test
        input_tensor: Input tensor
        mask_type: 'boolean', 'additive', 'auto' (tries both)
        
    Returns:
        (supports_mask, message) tuple
    """
    # Get baseline output
    baseline_output = module(input_tensor)
    if isinstance(baseline_output, dict):
        baseline_output = next(iter(baseline_output.values()))
    
    # Try to find mask parameter
    sig = inspect.signature(module.forward)
    mask_params = [p for p in sig.parameters if 'mask' in p.lower()]
    
    if not mask_params:
        return False, "No mask parameter found"
    
    mask_param = mask_params[0]
    
    # Create masks
    if len(input_tensor.shape) == 3:  # [batch, seq, dim]
        boolean_mask = torch.ones(input_tensor.shape[:2], dtype=torch.bool)
        boolean_mask[:, -2:] = False
        additive_mask = torch.zeros(input_tensor.shape[:2])
        additive_mask[:, -2:] = float('-inf')
    else:
        return False, "Cannot create mask for input shape"
    
    # Test masks
    masks_to_try = []
    if mask_type in ['boolean', 'auto']:
        masks_to_try.append(('boolean', boolean_mask))
    if mask_type in ['additive', 'auto']:
        masks_to_try.append(('additive', additive_mask))
    
    for mask_name, mask in masks_to_try:
        try:
            masked_output = module(input_tensor, **{mask_param: mask})
            if isinstance(masked_output, dict):
                masked_output = next(iter(masked_output.values()))
            
            # Check if mask had effect
            if not torch.allclose(baseline_output, masked_output, atol=1e-6):
                return True, f"Supports {mask_name} masking"
        except Exception:
            continue
    
    return False, "Mask parameter exists but has no effect"


def test_callable_interface(
    obj: object,
    args: tuple,
    kwargs: dict = None
) -> Tuple[bool, Any, str]:
    """
    Test if object supports callable interface.
    
    Returns:
        (is_callable, result, message) tuple
    """
    kwargs = kwargs or {}
    
    try:
        result = obj(*args, **kwargs)
        return True, result, "Object is callable"
    except TypeError as e:
        if "object is not callable" in str(e):
            return False, None, "Object is not callable"
        else:
            return False, None, f"Callable with wrong signature: {e}"


def validate_module_interface(
    module: nn.Module,
    expected_methods: List[str] = None,
    expected_attributes: List[str] = None
) -> Tuple[bool, List[str]]:
    """
    Validate module has expected interface.
    
    Returns:
        (all_present, missing_items) tuple
    """
    missing = []
    
    if expected_methods:
        for method in expected_methods:
            found_method = find_method(module, method)
            if not found_method:
                missing.append(f"method:{method}")
    
    if expected_attributes:
        for attr in expected_attributes:
            if not hasattr(module, attr):
                missing.append(f"attribute:{attr}")
    
    return len(missing) == 0, missing


def create_test_tensors(
    batch_size: int = 2,
    shapes: Dict[str, tuple] = None
) -> Dict[str, torch.Tensor]:
    """
    Create common test tensors.
    
    Args:
        batch_size: Batch size for all tensors
        shapes: Dict of name -> shape (without batch dim)
        
    Returns:
        Dictionary of test tensors
    """
    if shapes is None:
        shapes = {
            'seq_short': (10, 512),      # [batch, seq, dim]
            'seq_long': (100, 512),      # [batch, seq, dim]
            'image_small': (3, 64, 64),  # [batch, channels, h, w]
            'image_large': (3, 224, 224),
            'features': (512,),          # [batch, features]
            'graph_nodes': (20, 64),     # [batch * nodes, features]
        }
    
    tensors = {}
    for name, shape in shapes.items():
        full_shape = (batch_size,) + shape
        tensors[name] = torch.randn(full_shape)
    
    return tensors


def check_gradient_flow(
    module: nn.Module,
    input_tensor: torch.Tensor,
    target_tensor: Optional[torch.Tensor] = None
) -> Tuple[bool, str]:
    """
    Check if gradients flow through module.
    
    Returns:
        (has_gradients, message) tuple
    """
    # Ensure input requires grad
    input_tensor = input_tensor.detach().requires_grad_(True)
    
    # Forward pass
    output = module(input_tensor)
    if isinstance(output, dict):
        output = next(iter(output.values()))
    
    # Create loss
    if target_tensor is None:
        loss = output.mean()
    else:
        loss = nn.functional.mse_loss(output, target_tensor)
    
    # Backward pass
    try:
        loss.backward()
        
        # Check if input has gradients
        if input_tensor.grad is not None and input_tensor.grad.abs().sum() > 0:
            # Check if module parameters have gradients
            param_grads = []
            for name, param in module.named_parameters():
                if param.grad is not None:
                    param_grads.append((name, param.grad.abs().sum().item()))
            
            if param_grads:
                return True, f"Gradients flow through {len(param_grads)} parameters"
            else:
                return True, "Gradients flow (no trainable parameters)"
        else:
            return False, "No gradients on input"
            
    except Exception as e:
        return False, f"Gradient computation failed: {e}"


# Convenience function for common test pattern
def flexible_module_test(
    ModuleClass: type,
    test_params: Dict[str, Any],
    test_cases: List[Dict[str, Any]],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run a standard set of flexible tests on a module.
    
    Args:
        ModuleClass: Module class to test
        test_params: Parameters for initialization
        test_cases: List of test cases, each with 'input', 'test_type', etc.
        verbose: Print detailed information
        
    Returns:
        Dictionary of test results
    """
    results = {
        'initialization': False,
        'tests_passed': [],
        'tests_failed': [],
        'module': None
    }
    
    # Try to initialize
    try:
        module = init_with_variations(ModuleClass, test_params, verbose=verbose)
        results['initialization'] = True
        results['module'] = module
    except Exception as e:
        results['initialization_error'] = str(e)
        return results
    
    # Run test cases
    for i, test_case in enumerate(test_cases):
        test_name = test_case.get('name', f'test_{i}')
        try:
            if test_case['type'] == 'shape':
                success, msg = validate_shape_behavior(
                    module, 
                    test_case['input'],
                    test_case.get('expected', 'preserve')
                )
            elif test_case['type'] == 'mask':
                success, msg = test_mask_support(
                    module,
                    test_case['input'],
                    test_case.get('mask_type', 'auto')
                )
            elif test_case['type'] == 'gradient':
                success, msg = check_gradient_flow(
                    module,
                    test_case['input'],
                    test_case.get('target')
                )
            else:
                success, msg = False, f"Unknown test type: {test_case['type']}"
            
            if success:
                results['tests_passed'].append(test_name)
            else:
                results['tests_failed'].append((test_name, msg))
                
        except Exception as e:
            results['tests_failed'].append((test_name, f"Exception: {e}"))
    
    return results