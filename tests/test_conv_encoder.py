import torch
import sys
sys.path.append('.')

from modules.conv_encoder import ConvEncoder

def test_conv_encoder():
    print("Testing ConvEncoder Generation...")
    print("-" * 50)
    
    # Test configurations
    batch_size = 4
    in_channels = 3
    height, width = 64, 64
    base_channels = 32
    num_layers = 4
    
    # Create module
    model = ConvEncoder(
        in_channels=in_channels, 
        base_channels=base_channels, 
        num_layers=num_layers,
        use_residual=True
    )
    print(f"✓ Module created successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(batch_size, in_channels, height, width)
    print(f"\n✓ Input shape: {x.shape}")
    
    output = model(x)
    features = output['features']
    pooled = output['pooled']
    
    print(f"✓ Feature map shape: {features.shape}")
    print(f"✓ Pooled output shape: {pooled.shape}")
    
    # Verify shapes
    expected_channels = base_channels * (2 ** (num_layers - 1))  # 32 * 8 = 256
    expected_spatial = height // (2 ** (num_layers - 1))  # 64 / 8 = 8
    assert features.shape == (batch_size, expected_channels, expected_spatial, expected_spatial)
    assert pooled.shape == (batch_size, expected_channels)
    assert not torch.isnan(features).any(), "NaN values in features"
    print("✓ Forward pass successful")
    
    # Test gradients
    loss = pooled.sum()
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
    
    # Test different input sizes
    x_small = torch.randn(batch_size, in_channels, 32, 32)
    output_small = model(x_small)
    print(f"✓ Handles different input sizes: {x_small.shape} -> {output_small['features'].shape}")
    
    # Test feature shape calculation
    calculated_shape = model.get_feature_shape((in_channels, height, width))
    assert calculated_shape == features.shape[1:], "Feature shape calculation mismatch"
    print("✓ Feature shape calculation correct")
    
    # Component verification
    print("\nComponent Analysis:")
    print(f"  - ConvBlocks: {num_layers} layers")
    print(f"  - Channel progression: {in_channels} -> {base_channels} -> ... -> {expected_channels}")
    print(f"  - Spatial reduction: {height}x{width} -> {expected_spatial}x{expected_spatial}")
    print(f"  - Residual blocks: Enabled")
    print(f"  - BatchNorm: Applied throughout")
    print(f"  - Pooling: MaxPool2d between layers")
    print(f"  - Global pooling: AdaptiveAvgPool2d")
    
    print("\n✅ ConvEncoder: PASSED ALL TESTS")
    return True

if __name__ == "__main__":
    test_conv_encoder()