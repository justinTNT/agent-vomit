"""
Example of migrating an actual test to use flexible utilities.
Shows before/after for the transformer_block test.
"""

import torch
import sys
sys.path.append('/Users/jtnt/Play/agent-vomit')
from test_utils import (
    init_with_variations,
    extract_output,
    validate_shape_behavior,
    test_mask_support,
    check_gradient_flow
)


def test_transformer_block_original():
    """ORIGINAL: From test_transformer_block.py"""
    from modules.transformer_block import TransformerBlock
    
    # Fixed parameters - fails if module uses different names
    d_model = 512
    n_heads = 8
    model = TransformerBlock(d_model=d_model, n_heads=n_heads)
    
    # Fixed batch size and sequence length
    batch_size = 4
    seq_length = 10
    x = torch.randn(batch_size, seq_length, d_model)
    
    # Basic forward pass - assumes tensor output
    output = model(x)
    assert output.shape == (batch_size, seq_length, d_model), \
        f"Expected {(batch_size, seq_length, d_model)}, got {output.shape}"
    
    # Test with mask - assumes specific mask format and parameter name
    mask = torch.ones(batch_size, seq_length, seq_length)
    mask[:, :, 5:] = 0
    mask = mask.float().masked_fill(mask == 0, float('-inf'))
    
    masked_output = model(x, mask=mask)
    assert masked_output.shape == output.shape
    
    # Very rigid - any deviation fails


def test_transformer_block_flexible():
    """MIGRATED: Using flexible utilities"""
    from modules.transformer_block import TransformerBlock
    
    # Flexible initialization - handles parameter variations
    model = init_with_variations(TransformerBlock, {
        'd_model': 512,
        'n_heads': 8,  # Will also try 'num_heads', 'heads', etc.
        'd_ff': 2048,
        'dropout': 0.1
    })
    
    # Test data
    batch_size = 4
    seq_length = 10
    x = torch.randn(batch_size, seq_length, 512)
    
    # Test 1: Forward pass with flexible output handling
    output = model(x)
    output_dict = extract_output(output, primary_key='output')
    
    # Validate behavior, not exact format
    primary_output = output_dict[list(output_dict.keys())[0]]
    assert primary_output.shape[0] == batch_size, "Batch size not preserved"
    assert primary_output.shape[1] == seq_length, "Sequence length not preserved"
    
    # Test 2: Shape preservation (behavioral test)
    success, msg = validate_shape_behavior(model, x, 'preserve')
    assert success, f"TransformerBlock should preserve shape: {msg}"
    
    # Test 3: Mask support (tries multiple formats)
    success, msg = test_mask_support(model, x)
    assert success, f"TransformerBlock should support masking: {msg}"
    
    # Test 4: Gradient flow
    success, msg = check_gradient_flow(model, x)
    assert success, f"TransformerBlock should be differentiable: {msg}"
    
    print("✅ All flexible tests passed!")


def show_migration_benefits():
    """Demonstrate the benefits of migration."""
    print("🔄 Migration Benefits:\n")
    
    print("BEFORE (Rigid):")
    print("- ❌ Fails if module uses 'num_heads' instead of 'n_heads'")
    print("- ❌ Fails if output is wrapped in dict")
    print("- ❌ Fails if mask parameter is named differently")
    print("- ❌ Only tests one specific mask format")
    
    print("\nAFTER (Flexible):")
    print("- ✅ Automatically tries parameter variations")
    print("- ✅ Handles tensor/dict/tuple outputs")
    print("- ✅ Tests mask functionality, not format")
    print("- ✅ Validates actual behavior")
    
    print("\nCode Comparison:")
    print("- Old: model = TransformerBlock(d_model=512, n_heads=8)  # Exact only")
    print("- New: model = init_with_variations(...)  # Tries variations")
    print("\n- Old: assert output.shape == x.shape  # Exact format")
    print("- New: validate_shape_behavior(model, x, 'preserve')  # Behavior")


def create_migration_template():
    """Template for migrating tests."""
    template = '''
# Migration Template for test_[module_name].py

import torch
from test_utils import (
    init_with_variations,
    extract_output,
    validate_shape_behavior,
    test_mask_support,
    check_gradient_flow,
    find_method,
    test_callable_interface
)

def test_[module_name]():
    from modules.[module_name] import [ModuleClass]
    
    # Step 1: Flexible initialization
    model = init_with_variations([ModuleClass], {
        # Use canonical parameter names
        # Utility will try variations
    })
    
    # Step 2: Create test data
    x = torch.randn(...)
    
    # Step 3: Test forward pass flexibly
    output = model(x)
    output_dict = extract_output(output)
    
    # Step 4: Validate behavior (not format)
    # Instead of: assert output.shape == x.shape
    success, msg = validate_shape_behavior(model, x, 'preserve')
    assert success, msg
    
    # Step 5: Test capabilities
    if supports_masking:
        success, msg = test_mask_support(model, x)
        assert success, msg
    
    # Step 6: Test methods flexibly
    if has_methods:
        method = find_method(model, 'canonical_name')
        assert method is not None
        result = method(...)
'''
    
    print("\n📝 Migration Template:")
    print(template)


if __name__ == "__main__":
    print("🔄 Test Migration Example\n")
    
    # Show benefits
    show_migration_benefits()
    
    print("\n" + "="*60 + "\n")
    
    # Run the flexible version
    print("Running flexible transformer test...")
    try:
        test_transformer_block_flexible()
    except ImportError:
        print("(Skipping actual test - modules not available)")
    
    print("\n" + "="*60 + "\n")
    
    # Show template
    create_migration_template()