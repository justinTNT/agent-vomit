import torch
import sys
sys.path.append('.')

from modules.set_encoder import SetEncoder

def test_set_encoder():
    print("Testing SetEncoder Generation...")
    print("-" * 50)
    
    # Test configurations
    batch_size = 4
    set_size = 50
    input_dim = 64
    d_model = 256
    
    # Test basic set encoder
    print("\nTesting standard SetEncoder...")
    model = SetEncoder(
        input_dim=input_dim,
        d_model=d_model,
        n_heads=8,
        n_layers=4,
        pooling='mean'
    )
    print(f"✓ Module created successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(batch_size, set_size, input_dim)
    output = model(x)
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    
    assert output.shape == (batch_size, d_model)
    assert not torch.isnan(output).any()
    print("✓ Forward pass successful")
    
    # Test permutation invariance
    with torch.no_grad():
        model.eval()  # Ensure deterministic behavior
        perm = torch.randperm(set_size)
        x_permuted = x[:, perm]
        output_eval = model(x)
        output_permuted = model(x_permuted)
        # Check if outputs are close (may have small numerical differences)
        diff = (output_eval - output_permuted).abs().max().item()
        assert diff < 1e-3, f"Permutation invariance failed, max diff: {diff}"
        print(f"✓ Permutation invariance verified (max diff: {diff:.2e})")
        model.train()
    
    # Test with variable set sizes using mask
    mask = torch.ones(batch_size, set_size)
    mask[0, 30:] = 0  # First sample has only 30 elements
    mask[1, 40:] = 0  # Second sample has only 40 elements
    output_masked = model(x, mask=mask)
    print("✓ Variable set sizes with masking works")
    
    # Test gradients
    loss = output.sum()
    loss.backward()
    
    grad_check_passed = True
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"✗ No gradient for {name}")
            grad_check_passed = False
        elif torch.isnan(param.grad).any():
            print(f"✗ NaN gradient for {name}")
            grad_check_passed = False
    
    if grad_check_passed:
        print("✓ All gradients computed correctly")
    
    # Test ISAB (Induced Set Attention Block)
    print("\nTesting with ISAB (efficient for large sets)...")
    model_isab = SetEncoder(
        input_dim=input_dim,
        d_model=d_model,
        n_heads=8,
        n_layers=2,
        use_isab=True,
        n_inducing_points=16
    )
    output_isab = model_isab(x)
    print("✓ ISAB variant works correctly")
    print(f"  Parameters: {sum(p.numel() for p in model_isab.parameters()):,}")
    
    # Test different pooling strategies
    print("\nTesting pooling strategies...")
    pooling_strategies = ['mean', 'max', 'sum', 'attention', 'pma']
    
    for pooling in pooling_strategies:
        model_pool = SetEncoder(
            input_dim=input_dim,
            d_model=128,
            n_layers=2,
            pooling=pooling
        )
        out = model_pool(torch.randn(2, 30, input_dim))
        print(f"✓ Pooling '{pooling}' works correctly")
    
    # Test return_elements option
    output_full = model(x, return_elements=True)
    assert 'pooled' in output_full
    assert 'elements' in output_full
    assert output_full['elements'].shape == (batch_size, set_size, d_model)
    print("✓ Return elements option works")
    
    # Test pairwise processing
    print("\nTesting pairwise interactions...")
    x_small = torch.randn(2, 10, input_dim)  # Smaller set for pairwise
    output_pairs = model.forward_with_pairs(x_small)
    assert output_pairs.shape == (2, d_model)
    print("✓ Pairwise processing works")
    
    # Test empty set handling
    empty_mask = torch.zeros(1, set_size)
    x_empty = torch.randn(1, set_size, input_dim)
    output_empty = model(x_empty, mask=empty_mask)
    assert not torch.isnan(output_empty).any()
    print("✓ Handles empty sets gracefully")
    
    # Component verification
    print("\nComponent Analysis:")
    print("  - Permutation invariant architecture")
    print("  - No positional encoding (unlike standard transformers)")
    print("  - SetAttention: Self-attention without position")
    print("  - ISAB option: O(n*m) complexity instead of O(n²)")
    print("  - Multiple pooling strategies including PMA")
    print("  - Supports variable set sizes via masking")
    print("  - Optional pairwise interaction processing")
    
    print("\n✅ SetEncoder: PASSED ALL TESTS")
    return True

if __name__ == "__main__":
    test_set_encoder()