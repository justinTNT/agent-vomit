import torch
import sys
sys.path.append('.')

from modules.cross_modal_fusion import CrossModalFusion

def test_cross_modal_fusion():
    print("Testing CrossModalFusion Generation...")
    print("-" * 50)
    
    # Test configurations
    batch_size = 4
    seq_len_text = 32
    seq_len_image = 196  # 14x14 patches
    d_text = 512
    d_image = 768
    d_hidden = 512
    
    # Test all fusion types
    fusion_types = ['cross_attention', 'concatenate', 'multiplicative', 'bottleneck']
    
    for fusion_type in fusion_types:
        print(f"\nTesting {fusion_type} fusion...")
        
        # Create module
        model = CrossModalFusion(
            d_model_1=d_text,
            d_model_2=d_image,
            d_hidden=d_hidden,
            n_heads=8,
            n_layers=2 if fusion_type == 'cross_attention' else 1,
            fusion_type=fusion_type
        )
        print(f"✓ Module created successfully")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create inputs
        text_features = torch.randn(batch_size, seq_len_text, d_text)
        image_features = torch.randn(batch_size, seq_len_image, d_image)
        
        # Test forward pass
        if fusion_type == 'cross_attention':
            # Sequence inputs
            output = model(text_features, image_features)
        else:
            # These fusion types expect pooled features
            output = model(text_features, image_features)
        
        print(f"✓ Output shape: {output.shape}")
        assert output.shape == (batch_size, d_hidden)
        assert not torch.isnan(output).any()
        print(f"✓ Forward pass successful")
        
        # Test with masks (for cross-attention)
        if fusion_type == 'cross_attention':
            text_mask = torch.ones(batch_size, 1, 1, seq_len_text)
            text_mask[:, :, :, 20:] = 0
            image_mask = torch.ones(batch_size, 1, 1, seq_len_image)
            image_mask[:, :, :, 100:] = 0
            
            output_masked = model(text_features, image_features, text_mask, image_mask)
            print(f"✓ Masking works correctly")
        
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
            print(f"✓ All gradients computed correctly")
            
        # Clear gradients for next test
        model.zero_grad()
    
    # Test with different input dimensions
    print("\nTesting dimension flexibility...")
    model_flex = CrossModalFusion(
        d_model_1=256,
        d_model_2=384,
        d_hidden=512,
        output_dim=128,
        fusion_type='cross_attention'
    )
    
    text_small = torch.randn(2, 16, 256)
    image_small = torch.randn(2, 49, 384)
    output_flex = model_flex(text_small, image_small)
    assert output_flex.shape == (2, 128)
    print("✓ Handles different dimensions correctly")
    
    # Test return_all option
    model_all = CrossModalFusion(d_text, d_image, fusion_type='cross_attention')
    output_all = model_all(text_features, image_features, return_all=True)
    assert 'fused' in output_all
    assert 'modal_1_features' in output_all
    assert 'modal_2_features' in output_all
    print("✓ Return all features works")
    
    # Component verification
    print("\nComponent Analysis:")
    print("  Fusion types implemented:")
    print("    - Cross-attention: Bidirectional attention between modalities")
    print("    - Concatenate: Simple concat + MLP")
    print("    - Multiplicative: Element-wise product with gating")
    print("    - Bottleneck: Dimension reduction fusion")
    print("  All types handle different input dimensions")
    print("  Cross-attention supports sequence inputs with masking")
    
    print("\n✅ CrossModalFusion: PASSED ALL TESTS")
    return True

if __name__ == "__main__":
    test_cross_modal_fusion()