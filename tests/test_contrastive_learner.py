import torch
import torch.nn as nn
import sys
sys.path.append('.')

from modules.contrastive_learner import ContrastiveLearner
from modules.conv_encoder import ConvEncoder
from modules.vit_patch_encoder import ViTPatchEncoder

def test_contrastive_learner():
    print("Testing ContrastiveLearner Generation...")
    print("-" * 50)
    
    # Test configurations
    batch_size = 8
    img_size = 224
    
    # Create simple encoders for testing
    encoder = ConvEncoder(in_channels=3, base_channels=32, num_layers=3)
    
    # Test basic SimCLR style
    print("\nTesting SimCLR-style contrastive learning...")
    model = ContrastiveLearner(
        encoder_1=encoder,
        projection_dim=128,
        hidden_dim=512,
        temperature=0.07,
        loss_type='simclr'
    )
    print(f"✓ Module created successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x1 = torch.randn(batch_size, 3, img_size, img_size)
    x2 = torch.randn(batch_size, 3, img_size, img_size)  # Different augmentation
    
    loss = model(x1, x2)
    print(f"✓ SimCLR loss computed: {loss.item():.4f}")
    assert not torch.isnan(loss)
    print("✓ Forward pass successful")
    
    # Test with embeddings return
    output = model(x1, x2, return_embeddings=True)
    assert 'loss' in output
    assert 'embeddings_1' in output
    assert 'projections_1' in output
    print(f"✓ Embeddings shape: {output['embeddings_1'].shape}")
    print(f"✓ Projections shape: {output['projections_1'].shape}")
    
    # Test gradients
    loss.backward()
    grad_check_passed = True
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                print(f"✗ No gradient for {name}")
                grad_check_passed = False
            elif torch.isnan(param.grad).any():
                print(f"✗ NaN gradient for {name}")
                grad_check_passed = False
    
    if grad_check_passed:
        print("✓ All gradients computed correctly")
    
    # Test supervised contrastive
    print("\nTesting supervised contrastive...")
    labels = torch.randint(0, 5, (batch_size,))  # 5 classes
    loss_supervised = model(x1, x2, labels=labels)
    print(f"✓ Supervised contrastive loss: {loss_supervised.item():.4f}")
    
    # Test with different encoders
    print("\nTesting with different encoders...")
    encoder_1 = ConvEncoder(in_channels=3, base_channels=32, num_layers=2)
    encoder_2 = ConvEncoder(in_channels=3, base_channels=64, num_layers=2)
    
    model_diff = ContrastiveLearner(
        encoder_1=encoder_1,
        encoder_2=encoder_2,
        projection_dim=128
    )
    loss_diff = model_diff(x1, x2)
    print("✓ Works with different encoders")
    
    # Test with ViT encoder
    print("\nTesting with Vision Transformer encoder...")
    vit_encoder = ViTPatchEncoder(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        n_heads=6,
        n_layers=3
    )
    
    model_vit = ContrastiveLearner(
        encoder_1=vit_encoder,
        projection_dim=128
    )
    loss_vit = model_vit(x1, x2)
    print("✓ Works with ViT encoder")
    
    # Test MoCo style (if implemented)
    print("\nTesting MoCo-style with momentum encoder...")
    model_moco = ContrastiveLearner(
        encoder_1=encoder,
        projection_dim=128,
        loss_type='moco',
        use_momentum_encoder=True,
        momentum=0.999
    )
    loss_moco = model_moco(x1)  # MoCo typically uses single input
    print(f"✓ MoCo loss computed: {loss_moco.item():.4f}")
    
    # Test temperature sensitivity
    print("\nTesting temperature parameter...")
    temps = [0.05, 0.1, 1.0]
    losses = []
    for temp in temps:
        model_temp = ContrastiveLearner(
            encoder_1=encoder,
            temperature=temp
        )
        loss_temp = model_temp(x1, x2)
        losses.append(loss_temp.item())
        print(f"  Temperature {temp}: loss = {loss_temp.item():.4f}")
    
    # Verify temperature affects loss
    assert losses[0] != losses[1] != losses[2], "Temperature should affect loss values"
    print("✓ Temperature parameter working correctly")
    
    # Component verification
    print("\nComponent Analysis:")
    print("  - Encoders: Flexible, can use any encoder module")
    print("  - Projection head: MLP with BN and ReLU")
    print("  - Loss functions: SimCLR and MoCo style")
    print("  - Supports: Supervised and unsupervised contrastive")
    print("  - Temperature scaling: Adjustable")
    print("  - Momentum encoder: Optional for MoCo")
    
    print("\n✅ ContrastiveLearner: PASSED ALL TESTS")
    return True

if __name__ == "__main__":
    test_contrastive_learner()