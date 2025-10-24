"""
Example of generalized tests that work with explicit guidelines.
Tests validate behavior while respecting guideline specifications.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
import inspect

class FlexibleModuleTester:
    """Base class for generalized testing that respects guidelines."""
    
    def __init__(self, module_class, guideline_spec: Dict[str, Any]):
        self.module_class = module_class
        self.spec = guideline_spec
    
    def test_initialization(self):
        """Test module can be initialized with specified parameters."""
        required_params = self.spec['parameters']
        
        # Extract required params without defaults
        required_only = {k: v for k, v in required_params.items() 
                        if not isinstance(v, tuple) or v[1] is None}
        
        # Try to initialize with minimal params
        try:
            module = self.module_class(**required_only)
            assert isinstance(module, nn.Module)
            return True, module
        except Exception as e:
            return False, f"Initialization failed: {e}"
    
    def test_forward_signature(self, module):
        """Test forward method matches specification."""
        forward_spec = self.spec.get('forward_signature', {})
        sig = inspect.signature(module.forward)
        
        # Check required parameters exist
        for param_name in forward_spec.get('required', []):
            if param_name not in sig.parameters:
                return False, f"Missing required parameter: {param_name}"
        
        return True, "Forward signature matches"
    
    def test_return_format(self, module, sample_input):
        """Test output matches specified format."""
        output = module(sample_input)
        expected_format = self.spec.get('return_format')
        
        if expected_format == 'Tensor':
            if not torch.is_tensor(output):
                return False, "Expected Tensor output"
                
        elif expected_format == 'Dict[str, Tensor]':
            if not isinstance(output, dict):
                return False, "Expected dict output"
            
            # Check required keys
            required_keys = self.spec.get('return_keys', [])
            for key in required_keys:
                if key not in output:
                    return False, f"Missing required output key: {key}"
        
        return True, "Return format matches"
    
    def test_behavioral_properties(self, module, sample_input):
        """Test module exhibits expected behaviors."""
        behaviors = self.spec.get('behaviors', {})
        output = module(sample_input)
        
        # Shape preservation
        if behaviors.get('preserves_shape'):
            if output.shape != sample_input.shape:
                return False, "Should preserve input shape"
        
        # Batch dimension preservation
        if behaviors.get('preserves_batch'):
            if output.shape[0] != sample_input.shape[0]:
                return False, "Should preserve batch dimension"
        
        # Differentiability
        if behaviors.get('differentiable', True):
            try:
                loss = output.sum()
                loss.backward()
            except:
                return False, "Should be differentiable"
        
        return True, "Behavioral properties satisfied"


# Example: TransformerBlock test using guidelines
def test_transformer_block_generalized():
    from modules.transformer_block import TransformerBlock
    
    # Specification from guidelines
    spec = {
        'parameters': {
            'd_model': int,
            'n_heads': int,
            'd_ff': int,
            'dropout': (float, 0.1),
            'activation': (str, 'gelu'),
        },
        'return_format': 'Tensor',
        'forward_signature': {
            'required': ['x'],
            'optional': ['mask'],
        },
        'behaviors': {
            'preserves_shape': True,
            'preserves_batch': True,
            'differentiable': True,
            'supports_masking': True,
        }
    }
    
    tester = FlexibleModuleTester(TransformerBlock, spec)
    
    # Test 1: Initialization with guidelines params
    success, module = tester.test_initialization()
    assert success, f"Initialization failed: {module}"
    
    # Test 2: Forward signature
    success, msg = tester.test_forward_signature(module)
    assert success, msg
    
    # Test 3: Return format
    sample_input = torch.randn(2, 10, spec['parameters']['d_model'])
    success, msg = tester.test_return_format(module, sample_input)
    assert success, msg
    
    # Test 4: Behavioral properties
    success, msg = tester.test_behavioral_properties(module, sample_input)
    assert success, msg
    
    # Test 5: Specific transformer behaviors
    # Multi-head attention should work
    output = module(sample_input)
    assert output.shape == sample_input.shape
    
    # Should support masking
    mask = torch.ones(2, 10, 10)
    output_masked = module(sample_input, mask=mask)
    assert output_masked.shape == sample_input.shape
    
    print("✓ TransformerBlock passes all generalized tests")


# Example: DataVersioner test using guidelines  
def test_data_versioner_generalized():
    from modules.data_versioner import DataVersioner
    
    spec = {
        'parameters': {
            'storage_path': (str, './versions'),
            'deduplicate': (bool, True),
            'compression': (bool, False),
        },
        'callable_interface': True,
        'required_methods': ['load', 'list_versions'],
        'behaviors': {
            'stores_data': True,
            'retrieves_data': True,
            'deduplicates': True,
        }
    }
    
    versioner = DataVersioner()
    
    # Test callable interface (from guidelines)
    data = torch.randn(10, 10)
    version_id = versioner(data, "test version")
    assert isinstance(version_id, str)
    
    # Test required methods exist
    assert hasattr(versioner, 'load')
    assert hasattr(versioner, 'list_versions')
    
    # Test behavioral properties
    loaded = versioner.load(version_id)
    assert torch.allclose(loaded, data)
    
    # Test deduplication behavior (if enabled in guidelines)
    if spec['parameters']['deduplicate'][1]:
        version_id2 = versioner(data, "duplicate")
        assert version_id2 == version_id  # Same data = same version
    
    print("✓ DataVersioner passes all generalized tests")


# Guideline validation helper
def validate_module_against_guidelines(module_name: str, guidelines: Dict):
    """Ensures module implementation matches guidelines."""
    module_spec = guidelines.get(module_name)
    if not module_spec:
        return True, "No specification found"
    
    # Import module
    module_class = __import__(f'modules.{module_name}', fromlist=[module_name])
    
    # Run generalized tests
    tester = FlexibleModuleTester(module_class, module_spec)
    
    results = []
    results.append(tester.test_initialization())
    # ... other tests
    
    return all(r[0] for r in results), results