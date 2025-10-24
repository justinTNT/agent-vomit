"""
Examples of how tests could be more adaptive to common implementation variations.
These patterns would dramatically improve success rates.
"""

import torch
import inspect
from typing import Dict, Any, Optional

def adaptive_model_init(ModelClass, primary_params: Dict[str, Any], 
                       param_aliases: Dict[str, list] = None) -> torch.nn.Module:
    """
    Try to initialize model with common parameter name variations.
    
    Args:
        ModelClass: The model class to initialize
        primary_params: Primary parameter names and values
        param_aliases: Dict mapping primary names to list of aliases
    """
    param_aliases = param_aliases or {
        'n_heads': ['num_heads', 'heads', 'n_head'],
        'd_model': ['hidden_size', 'dim_model', 'model_dim'],
        'd_ff': ['feedforward_dim', 'ff_dim', 'mlp_dim'],
        'img_size': ['image_size', 'input_size'],
        'batch_size': ['bs', 'batch'],
    }
    
    # Try primary parameters first
    try:
        return ModelClass(**primary_params)
    except TypeError as e:
        # Extract which parameter caused the error
        error_msg = str(e)
        
        # Try aliases
        for param, aliases in param_aliases.items():
            if param in error_msg and param in primary_params:
                for alias in aliases:
                    try:
                        alt_params = primary_params.copy()
                        alt_params[alias] = alt_params.pop(param)
                        return ModelClass(**alt_params)
                    except TypeError:
                        continue
        raise e


def adaptive_forward_call(model, *args, **kwargs):
    """
    Adaptively call model forward with different signatures.
    """
    # Try standard forward
    try:
        return model(*args, **kwargs)
    except TypeError as e:
        # Check if timestamp should be positional vs kwarg
        if 'timestamp' in str(e) and 'timestamp' in kwargs:
            # Try as positional
            return model(*args, kwargs.pop('timestamp'), **kwargs)
        raise e


def adaptive_return_extraction(output, expected_keys: list) -> Dict[str, Any]:
    """
    Extract expected values from model output regardless of format.
    """
    if isinstance(output, dict):
        return output
    
    # If tensor, wrap in dict with expected keys
    if torch.is_tensor(output):
        if len(expected_keys) == 1:
            return {expected_keys[0]: output}
        else:
            # For multi-output, try to split or just use first key
            return {expected_keys[0]: output}
    
    # If tuple/list, map to expected keys
    if isinstance(output, (tuple, list)):
        return {key: output[i] for i, key in enumerate(expected_keys) 
                if i < len(output)}
    
    return {}


def adaptive_import(module_path: str, required_names: list, 
                   optional_names: list = None) -> Dict[str, Any]:
    """
    Import with fallbacks for missing classes/functions.
    """
    optional_names = optional_names or []
    imports = {}
    
    # Dynamic import
    module = __import__(module_path, fromlist=[''])
    
    # Get required imports
    for name in required_names:
        if hasattr(module, name):
            imports[name] = getattr(module, name)
        else:
            # Could provide a dummy/mock for testing
            raise ImportError(f"Required {name} not found in {module_path}")
    
    # Get optional imports
    for name in optional_names:
        imports[name] = getattr(module, name, None)
    
    return imports


def check_method_signature(obj, method_name: str, expected_params: list) -> bool:
    """
    Check if method accepts expected parameters.
    """
    if not hasattr(obj, method_name):
        return False
    
    method = getattr(obj, method_name)
    sig = inspect.signature(method)
    params = list(sig.parameters.keys())
    
    # Remove 'self' if present
    if 'self' in params:
        params.remove('self')
    
    # Check if all expected params are present
    return all(param in params for param in expected_params)


# Example: Adaptive TransformerBlock test
def test_transformer_adaptive():
    from modules.transformer_block import TransformerBlock
    
    # Try multiple initialization patterns
    model = adaptive_model_init(
        TransformerBlock,
        {'d_model': 512, 'n_heads': 8},
        param_aliases={'n_heads': ['num_heads', 'heads']}
    )
    
    # Test forward with adaptive return handling
    x = torch.randn(2, 10, 512)
    output = model(x)
    
    # Handle both dict and tensor returns
    result = adaptive_return_extraction(output, ['output', 'attention_weights'])
    
    if 'output' in result:
        assert result['output'].shape == x.shape
    else:
        # Fallback: assume output is the tensor itself
        assert output.shape == x.shape


# Example: Adaptive DataVersioner test  
def test_versioner_adaptive():
    from modules.data_versioner import DataVersioner
    
    versioner = DataVersioner()
    data = torch.randn(10, 10)
    
    # Try callable vs method
    try:
        version_id = versioner(data, "test")
    except TypeError:
        # Try method calls
        if hasattr(versioner, 'create_version'):
            version_id = versioner.create_version(data, "test")
        elif hasattr(versioner, 'version'):
            version_id = versioner.version(data, "test")
        else:
            raise
    
    # Try different load patterns
    if hasattr(versioner, 'load'):
        loaded = versioner.load(version_id)
    elif hasattr(versioner, 'get'):
        loaded = versioner.get(version_id)
    elif hasattr(versioner, 'retrieve'):
        loaded = versioner.retrieve(version_id)