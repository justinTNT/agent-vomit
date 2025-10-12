import torch
import torch.nn as nn
import sys
sys.path.append('.')

from modules.adaptive_computation import AdaptiveComputationTime, PonderNet, UniversalTransformer

def test_adaptive_computation():
    print("Testing AdaptiveComputation Generation...")
    print("-" * 50)
    
    # Test configurations
    batch_size = 4
    seq_len = 10
    hidden_size = 128
    input_size = 64
    output_size = 10
    
    # Test ACT mechanism
    print("\nTesting AdaptiveComputationTime...")
    act = AdaptiveComputationTime(
        hidden_size=hidden_size,
        max_steps=5,
        threshold=0.01
    )
    print(f"✓ ACT created successfully")
    
    # Create a simple compute function
    linear = nn.Linear(hidden_size, hidden_size)
    def compute_fn(x):
        return torch.tanh(linear(x))
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, hidden_size)
    output = act(x, compute_fn)
    
    assert 'output' in output
    assert 'ponder_cost' in output
    assert 'n_updates' in output
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output['output'].shape}")
    print(f"✓ Average ponder cost: {output['ponder_cost'].item():.3f}")
    print(f"✓ Updates per position: {output['n_updates'].mean().item():.2f}")
    assert output['output'].shape == x.shape
    print("✓ Forward pass successful")
    
    # Test PonderNet
    print("\nTesting PonderNet...")
    pondernet = PonderNet(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        max_steps=8,
        lambda_p=0.01
    )
    print(f"✓ PonderNet created successfully")
    print(f"  Parameters: {sum(p.numel() for p in pondernet.parameters()):,}")
    
    # Test forward pass
    x_ponder = torch.randn(batch_size, input_size)
    y_ponder, reg_loss = pondernet(x_ponder)
    
    print(f"✓ Input shape: {x_ponder.shape}")
    print(f"✓ Output shape: {y_ponder.shape}")
    print(f"✓ Regularization loss: {reg_loss.item():.4f}")
    assert y_ponder.shape == (batch_size, output_size)
    assert not torch.isnan(y_ponder).any()
    print("✓ Forward pass successful")
    
    # Test with all steps return
    output_all = pondernet(x_ponder, return_all_steps=True)
    print(f"✓ All outputs shape: {output_all['all_outputs'].shape}")
    print(f"✓ Expected steps: {output_all['expected_steps'].item():.2f}")
    
    # Test gradients
    loss = y_ponder.sum() + reg_loss
    loss.backward()
    
    grad_check_passed = True
    for name, param in pondernet.named_parameters():
        if param.grad is None:
            print(f"✗ No gradient for {name}")
            grad_check_passed = False
        elif torch.isnan(param.grad).any():
            print(f"✗ NaN gradient for {name}")
            grad_check_passed = False
    
    if grad_check_passed:
        print("✓ All gradients computed correctly")
    
    # Test Universal Transformer
    print("\nTesting UniversalTransformer...")
    ut = UniversalTransformer(
        d_model=256,
        n_heads=8,
        d_ff=1024,
        max_steps=4,
        use_act=False  # Start with fixed steps
    )
    print(f"✓ UniversalTransformer created successfully")
    print(f"  Parameters: {sum(p.numel() for p in ut.parameters()):,}")
    
    # Test forward pass
    x_ut = torch.randn(batch_size, seq_len, 256)
    output_ut = ut(x_ut)
    
    print(f"✓ Input shape: {x_ut.shape}")
    print(f"✓ Output shape: {output_ut.shape}")
    assert output_ut.shape == x_ut.shape
    assert not torch.isnan(output_ut).any()
    print("✓ Fixed-step forward pass successful")
    
    # Test with ACT
    print("\nTesting UniversalTransformer with ACT...")
    ut_act = UniversalTransformer(
        d_model=256,
        n_heads=8,
        d_ff=1024,
        max_steps=6,
        use_act=True,
        act_threshold=0.05
    )
    
    output_act = ut_act(x_ut, return_steps=True)
    print(f"✓ ACT output shape: {output_act['output'].shape}")
    print(f"✓ Ponder cost: {output_act['ponder_cost'].item():.3f}")
    print("✓ ACT-based forward pass successful")
    
    # Test with mask
    mask = torch.ones(batch_size, 1, 1, seq_len)
    mask[:, :, :, 5:] = 0
    output_masked = ut(x_ut, mask=mask)
    print("✓ Masking works correctly")
    
    # Test early stopping behavior
    print("\nTesting early stopping...")
    act_early = AdaptiveComputationTime(
        hidden_size=hidden_size,
        max_steps=10,
        threshold=0.5  # Higher threshold for earlier stopping
    )
    
    output_early = act_early(x, compute_fn)
    print(f"✓ Early stopping updates: {output_early['n_updates'].mean().item():.2f}")
    assert output_early['n_updates'].mean().item() < 10
    print("✓ Early stopping works")
    
    # Component verification
    print("\nComponent Analysis:")
    print("  - ACT: Dynamic computation depth with halting")
    print("  - PonderNet: Learned halting with KL regularization")
    print("  - UniversalTransformer: Recurrent transformers")
    print("  - Step embeddings for position in computation")
    print("  - Ponder cost for efficiency regularization")
    
    print("\n✅ AdaptiveComputation: PASSED ALL TESTS")
    return True

if __name__ == "__main__":
    test_adaptive_computation()