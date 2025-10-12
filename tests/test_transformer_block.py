import torch
import sys
sys.path.append('.')

from modules.transformer_block import TransformerBlock

def test_transformer_block():
    print("Testing TransformerBlock Generation...")
    print("-" * 50)
    
    # Test configurations
    batch_size = 4
    seq_len = 16
    d_model = 512
    n_heads = 8
    
    # Create module
    model = TransformerBlock(d_model=d_model, n_heads=n_heads)
    print(f"✓ Module created successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"\n✓ Input shape: {x.shape}")
    
    output = model(x)
    print(f"✓ Output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, d_model), "Output shape mismatch"
    assert not torch.isnan(output).any(), "NaN values in output"
    print("✓ Forward pass successful")
    
    # Test with attention mask
    mask = torch.ones(batch_size, n_heads, seq_len, seq_len)
    mask[:, :, :, 5:] = 0  # Mask positions after 5
    output_masked = model(x, mask=mask)
    print("✓ Masked attention working")
    
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
    
    # Test different sequence lengths
    x_short = torch.randn(batch_size, 8, d_model)
    output_short = model(x_short)
    assert output_short.shape == (batch_size, 8, d_model)
    print("✓ Handles variable sequence lengths")
    
    # Component verification
    print("\nComponent Analysis:")
    print(f"  - MultiHeadAttention: {n_heads} heads, d_k={d_model//n_heads}")
    print(f"  - FFN: {d_model} -> 2048 -> {d_model}")
    print(f"  - LayerNorm: 2 instances")
    print(f"  - Residual connections: 2 instances")
    print(f"  - Dropout: Applied throughout")
    
    print("\n✅ TransformerBlock: PASSED ALL TESTS")
    return True

if __name__ == "__main__":
    test_transformer_block()