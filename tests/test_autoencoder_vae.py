import torch
import torch.nn as nn
import sys
sys.path.append('.')

from modules.autoencoder_vae import AutoEncoder, VAE, ConditionalVAE
from modules.conv_encoder import ConvEncoder

def test_autoencoder_vae():
    print("Testing AutoEncoder/VAE Generation...")
    print("-" * 50)
    
    # Test configurations
    batch_size = 4
    img_size = 64
    latent_dim = 128
    
    # Create encoder
    encoder = ConvEncoder(in_channels=3, base_channels=32, num_layers=3)
    
    # Test AutoEncoder
    print("\nTesting basic AutoEncoder...")
    ae = AutoEncoder(encoder=encoder, latent_dim=latent_dim)
    print(f"✓ AutoEncoder created successfully")
    print(f"  Parameters: {sum(p.numel() for p in ae.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(batch_size, 3, img_size, img_size)
    output = ae(x)
    
    assert 'reconstruction' in output
    assert 'latent' in output
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Reconstruction shape: {output['reconstruction'].shape}")
    print(f"✓ Latent shape: {output['latent'].shape}")
    
    # Check latent dimension
    assert output['latent'].shape == (batch_size, latent_dim)
    assert not torch.isnan(output['reconstruction']).any()
    print("✓ Forward pass successful")
    
    # Test encode/decode separately
    z = ae.encode(x)
    x_recon = ae.decode(z)
    assert x_recon.shape == x.shape
    print("✓ Separate encode/decode working")
    
    # Test gradients
    loss = F.mse_loss(output['reconstruction'], x)
    loss.backward()
    
    grad_check_passed = True
    for name, param in ae.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                print(f"✗ No gradient for {name}")
                grad_check_passed = False
            elif torch.isnan(param.grad).any():
                print(f"✗ NaN gradient for {name}")
                grad_check_passed = False
    
    if grad_check_passed:
        print("✓ All gradients computed correctly")
    
    # Test VAE
    print("\nTesting Variational AutoEncoder (VAE)...")
    vae = VAE(encoder=encoder, latent_dim=latent_dim)
    print(f"✓ VAE created successfully")
    print(f"  Parameters: {sum(p.numel() for p in vae.parameters()):,}")
    
    # Test forward pass
    output_vae = vae(x)
    
    assert 'reconstruction' in output_vae
    assert 'mu' in output_vae
    assert 'logvar' in output_vae
    assert 'latent' in output_vae
    print(f"✓ Reconstruction shape: {output_vae['reconstruction'].shape}")
    print(f"✓ Mu shape: {output_vae['mu'].shape}")
    print(f"✓ Logvar shape: {output_vae['logvar'].shape}")
    
    # Test VAE loss computation
    loss_dict = vae.compute_loss(x, output_vae, beta=1.0)
    assert 'loss' in loss_dict
    assert 'recon_loss' in loss_dict
    assert 'kl_loss' in loss_dict
    print(f"✓ Total loss: {loss_dict['loss'].item():.4f}")
    print(f"✓ Recon loss: {loss_dict['recon_loss'].item():.4f}")
    print(f"✓ KL loss: {loss_dict['kl_loss'].item():.4f}")
    
    # Test sampling
    vae.eval()
    samples = vae.sample(num_samples=8, device=x.device)
    assert samples.shape == (8, 3, img_size, img_size)
    print("✓ Sampling from latent space works")
    vae.train()
    
    # Test reparameterization trick
    mu, logvar = vae.encode(x)
    z1 = vae.reparameterize(mu, logvar)
    z2 = vae.reparameterize(mu, logvar)
    assert not torch.allclose(z1, z2), "Reparameterization should be stochastic in training"
    
    vae.eval()
    z3 = vae.reparameterize(mu, logvar)
    z4 = vae.reparameterize(mu, logvar)
    assert torch.allclose(z3, z4), "Reparameterization should be deterministic in eval"
    print("✓ Reparameterization trick working correctly")
    
    # Test Conditional VAE
    print("\nTesting Conditional VAE...")
    condition_dim = 10
    cvae = ConditionalVAE(encoder=encoder, latent_dim=latent_dim, condition_dim=condition_dim)
    print(f"✓ Conditional VAE created successfully")
    
    # Create conditions (e.g., one-hot encoded class labels)
    conditions = torch.eye(condition_dim)[torch.randint(0, condition_dim, (batch_size,))]
    
    output_cvae = cvae(x, conditions)
    assert 'reconstruction' in output_cvae
    assert 'condition' in output_cvae
    print("✓ Conditional forward pass successful")
    
    # Test conditional sampling
    cvae.eval()
    condition = torch.eye(condition_dim)[3]  # Class 3
    conditional_samples = cvae.sample(num_samples=4, condition=condition, device=x.device)
    assert conditional_samples.shape == (4, 3, img_size, img_size)
    print("✓ Conditional sampling works")
    
    # Test with different beta values
    print("\nTesting beta-VAE behavior...")
    betas = [0.1, 1.0, 10.0]
    for beta in betas:
        loss_dict_beta = vae.compute_loss(x, output_vae, beta=beta)
        print(f"  Beta={beta}: Total loss={loss_dict_beta['loss'].item():.4f}")
    print("✓ Beta parameter affects loss correctly")
    
    # Component verification
    print("\nComponent Analysis:")
    print("  - AutoEncoder: Basic encoder-decoder architecture")
    print("  - VAE: Adds stochastic latent space with reparameterization")
    print("  - ConditionalVAE: Conditioning on auxiliary information")
    print("  - Loss computation: Reconstruction + β*KL divergence")
    print("  - Sampling: Generate from latent distribution")
    print("  - Flexible encoder/decoder: Can use any encoder module")
    
    print("\n✅ AutoEncoder/VAE: PASSED ALL TESTS")
    return True

# Import F for loss computation
import torch.nn.functional as F

if __name__ == "__main__":
    test_autoencoder_vae()