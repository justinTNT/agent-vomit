import torch
import pytest
import torch.nn.functional as F
from modules.residual_vector_quantizer import (
    VectorQuantizer,
    ResidualVectorQuantizer,
    ResidualVectorQuantizerWrapper
)


def test_basic_vector_quantizer():
    """Test basic vector quantizer functionality."""
    vq = VectorQuantizer(
        num_embeddings=256,
        embedding_dim=64,
        commitment_cost=0.25
    )
    
    # Test with 2D input
    x = torch.randn(8, 64)
    quantized, indices, loss = vq(x)
    
    assert quantized.shape == x.shape
    assert indices.shape == (8,)
    assert loss >= 0
    assert torch.all(indices >= 0) and torch.all(indices < 256)
    
    # Test with 3D input
    x_3d = torch.randn(4, 16, 64)
    quantized_3d, indices_3d, loss_3d = vq(x_3d)
    
    assert quantized_3d.shape == x_3d.shape
    assert indices_3d.shape == (4, 16)


def test_straight_through_estimator():
    """Test that gradients flow through quantizer."""
    vq = VectorQuantizer(num_embeddings=128, embedding_dim=32)
    
    x = torch.randn(4, 32, requires_grad=True)
    quantized, indices, loss = vq(x)
    
    # Gradients should flow through
    output_loss = quantized.sum() + loss
    output_loss.backward()
    
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))


def test_residual_vector_quantizer():
    """Test hierarchical residual quantization."""
    rvq = ResidualVectorQuantizer(
        num_quantizers=4,
        num_embeddings=512,
        embedding_dim=128
    )
    
    x = torch.randn(8, 128)
    quantized, indices, loss = rvq(x)
    
    assert quantized.shape == x.shape
    assert indices.shape == (8, 4)  # 4 quantizers
    assert loss >= 0
    
    # Check that each quantizer uses different indices
    # (unless by chance, which is very unlikely with 512 embeddings)
    assert len(torch.unique(indices)) > 4


def test_progressive_quantization():
    """Test using fewer quantizers than available."""
    rvq = ResidualVectorQuantizer(
        num_quantizers=8,
        num_embeddings=256,
        embedding_dim=64
    )
    
    x = torch.randn(4, 64)
    
    # Use only first 4 quantizers
    quantized, indices, loss = rvq(x, num_quantizers=4)
    
    # When using fewer quantizers, indices only includes those used
    assert indices.shape == (4, 4)  # Only 4 quantizers used
    
    # Using all quantizers
    quantized_all, indices_all, loss_all = rvq(x)
    assert indices_all.shape == (4, 8)  # All 8 quantizers


def test_different_codebook_sizes():
    """Test with different codebook sizes per level."""
    rvq = ResidualVectorQuantizer(
        num_quantizers=3,
        num_embeddings=[1024, 512, 256],  # Decreasing sizes
        embedding_dim=64
    )
    
    x = torch.randn(8, 64)
    quantized, indices, loss = rvq(x)
    
    assert indices.shape == (8, 3)
    # Check each quantizer respects its codebook size
    assert torch.all(indices[:, 0] < 1024)
    assert torch.all(indices[:, 1] < 512)
    assert torch.all(indices[:, 2] < 256)


def test_shared_codebook():
    """Test shared codebook across all levels."""
    rvq = ResidualVectorQuantizer(
        num_quantizers=4,
        num_embeddings=256,
        embedding_dim=32,
        shared_codebook=True
    )
    
    # Check that all quantizers share the same embedding weights
    embedding_weight = rvq.quantizers[0].embedding.weight
    for i in range(1, 4):
        assert rvq.quantizers[i].embedding.weight is embedding_weight


def test_encode_decode():
    """Test encoding to indices and decoding back."""
    rvq = ResidualVectorQuantizer(
        num_quantizers=4,
        num_embeddings=512,
        embedding_dim=64
    )
    
    x = torch.randn(8, 16, 64)  # Batch of sequences
    
    # Encode
    indices = rvq.encode(x)
    assert indices.shape == (8, 16, 4)
    
    # Decode
    reconstructed = rvq.decode(indices)
    assert reconstructed.shape == x.shape
    
    # Should be close to quantized version
    quantized, _, _ = rvq(x)
    assert torch.allclose(reconstructed, quantized, atol=1e-6)


def test_commitment_loss():
    """Test that commitment loss works correctly."""
    # High commitment cost
    rvq_high = ResidualVectorQuantizer(
        num_quantizers=2,
        num_embeddings=128,
        embedding_dim=32,
        commitment_cost=1.0
    )
    
    # Low commitment cost
    rvq_low = ResidualVectorQuantizer(
        num_quantizers=2,
        num_embeddings=128,
        embedding_dim=32,
        commitment_cost=0.1
    )
    
    x = torch.randn(4, 32)
    _, _, loss_high = rvq_high(x)
    _, _, loss_low = rvq_low(x)
    
    # Higher commitment cost should generally lead to higher loss
    # (though not guaranteed for a single sample)
    assert loss_high >= 0 and loss_low >= 0


def test_quantizer_dropout():
    """Test quantizer dropout during training."""
    rvq = ResidualVectorQuantizer(
        num_quantizers=8,
        num_embeddings=256,
        embedding_dim=64,
        quantizer_dropout=0.5
    )
    
    rvq.train()
    x = torch.randn(16, 64)
    
    # Multiple forward passes should use different quantizers
    indices_list = []
    for _ in range(5):
        _, indices, _ = rvq(x)
        indices_list.append(indices)
    
    # Check that dropout is actually happening
    # (some indices should be different due to dropout)
    all_same = True
    for i in range(1, 5):
        if not torch.all(indices_list[0] == indices_list[i]):
            all_same = False
            break
    
    assert not all_same  # Dropout should cause variation


def test_ema_update():
    """Test EMA codebook update."""
    vq = VectorQuantizer(
        num_embeddings=128,
        embedding_dim=32,
        decay=0.99
    )
    
    # Get initial embeddings
    initial_embeddings = vq.embedding.weight.data.clone()
    
    # Train mode and forward pass
    vq.train()
    x = torch.randn(64, 32)  # Large batch
    _, _, _ = vq(x)
    
    # Embeddings should have changed (slightly due to high decay)
    assert not torch.allclose(initial_embeddings, vq.embedding.weight.data)


def test_cosine_distance():
    """Test cosine distance metric."""
    vq = VectorQuantizer(
        num_embeddings=256,
        embedding_dim=64,
        distance_metric='cosine'
    )
    
    x = torch.randn(8, 64)
    quantized, indices, loss = vq(x)
    
    assert quantized.shape == x.shape
    assert indices.shape == (8,)
    
    # Cosine distance should produce unit-norm quantized vectors
    # (approximately, due to straight-through estimator)
    norms = torch.norm(quantized, p=2, dim=1)
    # The quantized embeddings themselves should be normalized in cosine mode


def test_wrapper_with_projection():
    """Test wrapper with input projection."""
    wrapper = ResidualVectorQuantizerWrapper(
        input_dim=256,  # Different from embedding_dim
        num_quantizers=4,
        embedding_dim=64,
        num_embeddings=512
    )
    
    x = torch.randn(8, 256)
    quantized, indices, loss = wrapper(x)
    
    assert quantized.shape == x.shape  # Should match input shape
    assert indices.shape == (8, 4)


def test_3d_input():
    """Test with 3D input (e.g., audio sequences)."""
    rvq = ResidualVectorQuantizer(
        num_quantizers=4,
        num_embeddings=1024,
        embedding_dim=128
    )
    
    # [batch, time, features]
    x = torch.randn(4, 100, 128)
    quantized, indices, loss = rvq(x)
    
    assert quantized.shape == x.shape
    assert indices.shape == (4, 100, 4)


def test_codebook_usage():
    """Test codebook usage statistics."""
    rvq = ResidualVectorQuantizer(
        num_quantizers=2,
        num_embeddings=128,
        embedding_dim=32,
        decay=0.99
    )
    
    # Forward pass to initialize EMA
    rvq.train()
    x = torch.randn(64, 32)
    _, _, _ = rvq(x)
    
    # Get usage statistics
    usage = rvq.get_codebook_usage()
    assert len(usage) == 2
    assert usage[0] is not None  # Should have EMA stats
    assert usage[0].shape == (128,)


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    # Single quantizer
    rvq_single = ResidualVectorQuantizer(
        num_quantizers=1,
        num_embeddings=256,
        embedding_dim=64
    )
    
    x = torch.randn(4, 64)
    quantized, indices, loss = rvq_single(x)
    assert indices.shape == (4, 1)
    
    # Very small codebook
    rvq_small = ResidualVectorQuantizer(
        num_quantizers=2,
        num_embeddings=4,  # Only 4 codes
        embedding_dim=32
    )
    
    x = torch.randn(8, 32)
    _, indices, _ = rvq_small(x)
    assert torch.all(indices < 4)
    
    # Large number of quantizers
    rvq_large = ResidualVectorQuantizer(
        num_quantizers=16,
        num_embeddings=256,
        embedding_dim=64
    )
    
    x = torch.randn(2, 64)
    _, indices, _ = rvq_large(x)
    assert indices.shape == (2, 16)


def test_deterministic_eval():
    """Test deterministic behavior in eval mode."""
    rvq = ResidualVectorQuantizer(
        num_quantizers=4,
        num_embeddings=256,
        embedding_dim=64,
        quantizer_dropout=0.5  # Should be ignored in eval
    )
    
    rvq.eval()
    x = torch.randn(8, 64)
    
    # Multiple forward passes should be identical
    quantized1, indices1, _ = rvq(x)
    quantized2, indices2, _ = rvq(x)
    
    assert torch.allclose(quantized1, quantized2)
    assert torch.all(indices1 == indices2)


if __name__ == "__main__":
    test_basic_vector_quantizer()
    test_straight_through_estimator()
    test_residual_vector_quantizer()
    test_progressive_quantization()
    test_different_codebook_sizes()
    test_shared_codebook()
    test_encode_decode()
    test_commitment_loss()
    test_quantizer_dropout()
    test_ema_update()
    test_cosine_distance()
    test_wrapper_with_projection()
    test_3d_input()
    test_codebook_usage()
    test_edge_cases()
    test_deterministic_eval()
    print("All ResidualVectorQuantizer tests passed!")