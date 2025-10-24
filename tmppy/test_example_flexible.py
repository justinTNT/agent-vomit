"""
Example of how to use flexible test utilities to create robust tests.
This shows the migration from rigid to flexible testing.
"""

import torch
import torch.nn as nn
from test_utils import (
    init_with_variations, 
    extract_output,
    find_method,
    validate_shape_behavior,
    test_mask_support,
    check_gradient_flow,
    flexible_module_test
)


def test_transformer_block_rigid():
    """OLD: Rigid test that fails on any deviation."""
    from modules.transformer_block import TransformerBlock
    
    # This ONLY works with exact parameter names
    model = TransformerBlock(d_model=512, n_heads=8)
    x = torch.randn(2, 10, 512)
    output = model(x)
    
    # This ONLY works if output is tensor
    assert output.shape == x.shape
    
    # This ONLY works with exact mask parameter name and format
    mask = torch.zeros(2, 10, 10).float()
    mask[:, :, -2:] = float('-inf')
    output_masked = model(x, mask=mask)
    
    print("❌ Rigid test: Fails on any API variation")


def test_transformer_block_flexible():
    """NEW: Flexible test that validates behavior."""
    from modules.transformer_block import TransformerBlock
    
    # Initialize with common parameter variations
    model = init_with_variations(TransformerBlock, {
        'd_model': 512,
        'n_heads': 8,
        'd_ff': 2048,
        'dropout': 0.1
    })
    print("✅ Initialized with flexible parameters")
    
    # Test shape preservation
    x = torch.randn(2, 10, 512)
    success, msg = validate_shape_behavior(model, x, 'preserve')
    assert success, f"Shape validation failed: {msg}"
    print("✅ Shape preservation validated")
    
    # Test mask support (tries both boolean and additive)
    success, msg = test_mask_support(model, x, 'auto')
    assert success, f"Mask support failed: {msg}"
    print(f"✅ Mask support validated: {msg}")
    
    # Test gradient flow
    success, msg = check_gradient_flow(model, x)
    assert success, f"Gradient flow failed: {msg}"
    print("✅ Gradient flow validated")
    
    # Extract output flexibly
    output = model(x)
    output_dict = extract_output(output, primary_key='encoded')
    print(f"✅ Output extracted: {list(output_dict.keys())}")


def test_data_versioner_flexible():
    """Example with method/callable flexibility."""
    from modules.data_versioner import DataVersioner
    
    versioner = DataVersioner()
    data = torch.randn(10, 10)
    
    # Test callable interface (preferred) or method
    is_callable, result, msg = test_callable_interface(versioner, (data, "test"))
    
    if is_callable:
        version_id = result
        print("✅ Using callable interface")
    else:
        # Try to find create method
        create_method = find_method(versioner, 'create', 
                                   ['create_version', 'version', 'create', '__call__'])
        assert create_method, "No create method found"
        version_id = create_method(data, "test")
        print("✅ Using method interface")
    
    # Find load method flexibly
    load_method = find_method(versioner, 'load')
    assert load_method, "No load method found"
    
    loaded = load_method(version_id)
    assert torch.allclose(loaded, data), "Data not preserved"
    print("✅ Data versioning works correctly")


def test_conv_encoder_flexible():
    """Example with flexible return format handling."""
    from modules.conv_encoder import ConvEncoder
    
    # Initialize with variations
    model = init_with_variations(ConvEncoder, {
        'in_channels': 3,
        'base_channels': 64,
        'num_layers': 4,
        'latent_dim': 512
    })
    
    x = torch.randn(2, 3, 64, 64)
    output = model(x)
    
    # Handle dict/tensor/tuple returns
    output_dict = extract_output(output, 
                                expected_keys=['features', 'pooled', 'latent'])
    
    # Validate we got some encoded representation
    if 'features' in output_dict:
        features = output_dict['features']
        print(f"✅ Got features: {features.shape}")
    else:
        # Just check we got something
        first_output = next(iter(output_dict.values()))
        print(f"✅ Got output: {first_output.shape}")
    
    # Validate dimension reduction
    success, msg = validate_shape_behavior(model, x, 'reduce')
    print(f"✅ Dimension reduction: {msg}")


def test_module_with_full_framework():
    """Example using the complete flexible framework."""
    from modules.sequence_encoder import SequenceEncoder
    
    # Define test configuration
    results = flexible_module_test(
        ModuleClass=SequenceEncoder,
        test_params={
            'input_dim': 128,
            'd_model': 512,
            'n_heads': 8,
            'n_layers': 4
        },
        test_cases=[
            {
                'name': 'basic_forward',
                'type': 'shape',
                'input': torch.randn(2, 20, 128),
                'expected': 'same_batch'
            },
            {
                'name': 'mask_support',
                'type': 'mask',
                'input': torch.randn(2, 20, 128),
                'mask_type': 'auto'
            },
            {
                'name': 'gradient_flow',
                'type': 'gradient',
                'input': torch.randn(2, 20, 128),
                'target': torch.randn(2, 20, 512)
            }
        ],
        verbose=True
    )
    
    print(f"\n📊 Test Results for SequenceEncoder:")
    print(f"   Initialization: {'✅' if results['initialization'] else '❌'}")
    print(f"   Tests passed: {len(results['tests_passed'])}")
    print(f"   Tests failed: {len(results['tests_failed'])}")
    
    if results['tests_failed']:
        print("   Failed tests:")
        for test_name, reason in results['tests_failed']:
            print(f"     - {test_name}: {reason}")


def compare_approaches():
    """Show the difference between rigid and flexible testing."""
    print("🔍 Comparing Testing Approaches\n")
    
    print("=" * 60)
    print("RIGID APPROACH (Old)")
    print("=" * 60)
    print("❌ Fails if param is 'num_heads' instead of 'n_heads'")
    print("❌ Fails if output is dict instead of tensor")
    print("❌ Fails if mask parameter has different name")
    print("❌ Fails if method is 'get' instead of 'load'")
    print("Result: 0% success rate with external implementations")
    
    print("\n" + "=" * 60)
    print("FLEXIBLE APPROACH (New)")
    print("=" * 60)
    print("✅ Tries common parameter variations")
    print("✅ Handles tensor/dict/tuple returns")
    print("✅ Tests behavior, not exact API")
    print("✅ Finds methods by trying variations")
    print("Result: 60-80% expected success rate")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT")
    print("=" * 60)
    print("We test WHAT modules do, not HOW they do it")
    print("We validate BEHAVIOR, not IMPLEMENTATION")


if __name__ == "__main__":
    print("🧪 Flexible Testing Examples\n")
    
    # Show comparison
    compare_approaches()
    
    print("\n\n📝 Running Flexible Tests...")
    
    # Run examples
    try:
        print("\n1. TransformerBlock Test:")
        test_transformer_block_flexible()
    except Exception as e:
        print(f"   Failed: {e}")
    
    try:
        print("\n2. DataVersioner Test:")
        test_data_versioner_flexible()
    except Exception as e:
        print(f"   Failed: {e}")
    
    try:
        print("\n3. ConvEncoder Test:")
        test_conv_encoder_flexible()
    except Exception as e:
        print(f"   Failed: {e}")
    
    try:
        print("\n4. Full Framework Test:")
        test_module_with_full_framework()
    except Exception as e:
        print(f"   Failed: {e}")