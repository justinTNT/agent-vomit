#!/usr/bin/env python3
"""
Focused test comparison showing rigid vs flexible testing with clear examples.
"""

import torch
import torch.nn as nn
import sys
sys.path.append('.')

from test_utils import init_with_variations, PARAMETER_VARIATIONS


def rigid_test_example():
    """Show how rigid tests fail with parameter variations."""
    print("RIGID TESTING EXAMPLE")
    print("=" * 50)
    
    # Import the original module
    from modules.transformer_block import TransformerBlock
    
    print("\n1. Testing with expected parameter names:")
    print("   TransformerBlock(d_model=512, n_heads=8)")
    try:
        module = TransformerBlock(d_model=512, n_heads=8)
        print("   ✓ SUCCESS: Module created")
    except TypeError as e:
        print(f"   ✗ FAILED: {e}")
    
    print("\n2. Testing with alternative parameter names:")
    print("   TransformerBlock(hidden_dim=512, num_heads=8)")
    try:
        module = TransformerBlock(hidden_dim=512, num_heads=8)
        print("   ✓ SUCCESS: Module created")
    except TypeError as e:
        print(f"   ✗ FAILED: {e}")
    
    print("\n3. Testing with another variation:")
    print("   TransformerBlock(embed_dim=512, heads=8)")
    try:
        module = TransformerBlock(embed_dim=512, heads=8)
        print("   ✓ SUCCESS: Module created")
    except TypeError as e:
        print(f"   ✗ FAILED: {e}")


def flexible_test_example():
    """Show how flexible tests handle parameter variations."""
    print("\n\nFLEXIBLE TESTING EXAMPLE")
    print("=" * 50)
    
    from modules.transformer_block import TransformerBlock
    
    # Test cases with different parameter names
    test_cases = [
        {'d_model': 512, 'n_heads': 8},
        {'hidden_dim': 512, 'num_heads': 8},
        {'embed_dim': 512, 'heads': 8},
        {'model_dim': 512, 'attention_heads': 8}
    ]
    
    for i, params in enumerate(test_cases, 1):
        print(f"\n{i}. Testing with parameters: {params}")
        try:
            # Use flexible initialization
            module = init_with_variations(TransformerBlock, params, verbose=True)
            print("   ✓ SUCCESS: Module created with flexible initialization")
            
            # Test it works
            x = torch.randn(2, 10, 512)
            output = module(x)
            print(f"   ✓ Forward pass successful, output shape: {output.shape}")
        except Exception as e:
            print(f"   ✗ FAILED: {e}")


def test_dict_output_handling():
    """Show how flexible testing handles different output formats."""
    print("\n\nOUTPUT FORMAT HANDLING")
    print("=" * 50)
    
    from modules.conv_encoder import ConvEncoder
    from test_utils import extract_output
    
    print("\n1. ConvEncoder returns a dictionary output")
    module = ConvEncoder(in_channels=3, base_channels=64)
    x = torch.randn(2, 3, 64, 64)
    output = module(x)
    
    print(f"   Output type: {type(output)}")
    if isinstance(output, dict):
        print(f"   Output keys: {list(output.keys())}")
    
    print("\n2. Rigid test expects tensor output:")
    print("   torch.isnan(output).any()  # This fails with dict!")
    
    print("\n3. Flexible test uses extract_output:")
    standardized = extract_output(output)
    print(f"   Standardized output keys: {list(standardized.keys())}")
    print(f"   Can now check: torch.isnan(standardized['output']).any()")


def show_improvement_summary():
    """Show the overall improvement."""
    print("\n\nSUMMARY: WHY FLEXIBLE TESTING IMPROVES SUCCESS RATES")
    print("=" * 50)
    
    print("\nRigid Testing Problems:")
    print("  1. Expects exact parameter names (d_model, n_heads)")
    print("  2. Fails if output is dict instead of tensor")
    print("  3. Can't handle alternative method names")
    print("  4. No tolerance for reasonable implementation variations")
    
    print("\nFlexible Testing Solutions:")
    print("  1. Tries multiple parameter name variations")
    print("  2. Handles different output formats (tensor/dict/tuple)")
    print("  3. Searches for alternative method names")
    print("  4. Tests behavior rather than exact implementation")
    
    print("\nResult: Much higher success rate with diverse implementations!")


def main():
    """Run all examples."""
    rigid_test_example()
    flexible_test_example()
    test_dict_output_handling()
    show_improvement_summary()


if __name__ == "__main__":
    main()