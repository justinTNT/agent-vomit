"""
Test suite for WaveGANDiscriminator module.

Tests basic functionality, multi-scale operation, feature matching,
and conditional discrimination for time-domain audio processing.
"""

import torch
import torch.nn as nn
import pytest
import sys
import os

# Add the audio_gan_tier4 directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'audio_gan_tier4'))

from wave_gan_discriminator import (
    WaveGANDiscriminator,
    MultiScaleWaveDiscriminator,
    FeatureMatchingLoss,
    ConditionalWaveDiscriminator
)


class TestWaveGANDiscriminator:
    """Test WaveGANDiscriminator functionality."""
    
    def test_basic_discriminator(self):
        """Test basic WaveGAN discriminator functionality."""
        discriminator = WaveGANDiscriminator(
            in_channels=1,
            base_channels=32,
            num_layers=3
        )
        
        # Test input shapes
        batch_size = 2
        audio_length = 8192
        x = torch.randn(batch_size, 1, audio_length)
        
        # Basic forward pass
        output = discriminator(x)
        
        assert output.dim() >= 2, "Output should have at least 2 dimensions"
        assert output.shape[0] == batch_size, "Batch size should be preserved"
        assert output.requires_grad, "Output should require gradients"
        
        # Test with feature extraction
        output, features = discriminator(x, return_features=True)
        
        assert isinstance(features, list), "Features should be a list"
        assert len(features) > 0, "Should extract some features"
        assert all(f.shape[0] == batch_size for f in features), "All features should preserve batch size"
    
    def test_receptive_field_calculation(self):
        """Test receptive field calculation."""
        discriminator = WaveGANDiscriminator(
            num_layers=4,
            kernel_sizes=[15, 41, 41, 41],
            strides=[1, 2, 2, 4]
        )
        
        receptive_field = discriminator.get_receptive_field()
        
        assert isinstance(receptive_field, int), "Receptive field should be integer"
        assert receptive_field > 1, "Receptive field should be greater than 1"
        assert receptive_field < 1000, "Receptive field should be reasonable"
    
    def test_spectral_normalization(self):
        """Test spectral normalization application."""
        # With spectral norm
        disc_with_sn = WaveGANDiscriminator(use_spectral_norm=True)
        
        # Without spectral norm
        disc_without_sn = WaveGANDiscriminator(use_spectral_norm=False)
        
        x = torch.randn(2, 1, 4096)
        
        # Both should work
        out_with = disc_with_sn(x)
        out_without = disc_without_sn(x)
        
        assert out_with.shape == out_without.shape, "Spectral norm shouldn't change output shape"
    
    def test_kwargs_acceptance(self):
        """Test that discriminator accepts unknown kwargs."""
        # Should not raise error with unknown parameters
        discriminator = WaveGANDiscriminator(
            unknown_param=123,
            another_param="test"
        )
        
        x = torch.randn(1, 1, 2048)
        output = discriminator(x)
        
        assert output is not None, "Should work with unknown kwargs"


class TestMultiScaleWaveDiscriminator:
    """Test MultiScaleWaveDiscriminator functionality."""
    
    def test_multi_scale_operation(self):
        """Test multi-scale discriminator operation."""
        discriminator = MultiScaleWaveDiscriminator(
            num_scales=3,
            scale_factors=[1, 2, 4],
            base_channels=32
        )
        
        batch_size = 2
        audio_length = 8192
        x = torch.randn(batch_size, 1, audio_length)
        
        # Basic forward pass
        outputs = discriminator(x)
        
        assert isinstance(outputs, list), "Should return list of outputs"
        assert len(outputs) == 3, "Should have 3 scale outputs"
        assert all(out.shape[0] == batch_size for out in outputs), "All outputs should preserve batch size"
        
        # Test with features
        outputs, all_features = discriminator(x, return_features=True)
        
        assert isinstance(all_features, list), "Features should be a list"
        assert len(all_features) == 3, "Should have features for all scales"
        assert all(isinstance(features, list) for features in all_features), "Each scale should have feature list"
    
    def test_loss_computation(self):
        """Test discriminator loss computation."""
        discriminator = MultiScaleWaveDiscriminator(num_scales=2)
        
        batch_size = 2
        x_real = torch.randn(batch_size, 1, 4096)
        x_fake = torch.randn(batch_size, 1, 4096)
        
        real_outputs = discriminator(x_real)
        fake_outputs = discriminator(x_fake)
        
        # Test different loss types
        for loss_type in ['hinge', 'vanilla', 'lsgan']:
            loss = discriminator.compute_loss(real_outputs, fake_outputs, loss_type)
            
            assert isinstance(loss, torch.Tensor), f"Loss should be tensor for {loss_type}"
            assert loss.dim() == 0, f"Loss should be scalar for {loss_type}"
            assert loss.requires_grad, f"Loss should require gradients for {loss_type}"
    
    def test_generator_loss(self):
        """Test generator loss computation."""
        discriminator = MultiScaleWaveDiscriminator(num_scales=2)
        
        x_fake = torch.randn(2, 1, 4096)
        fake_outputs = discriminator(x_fake)
        
        # Test different loss types
        for loss_type in ['hinge', 'vanilla', 'lsgan']:
            gen_loss = discriminator.compute_generator_loss(fake_outputs, loss_type)
            
            assert isinstance(gen_loss, torch.Tensor), f"Gen loss should be tensor for {loss_type}"
            assert gen_loss.dim() == 0, f"Gen loss should be scalar for {loss_type}"
            assert gen_loss.requires_grad, f"Gen loss should require gradients for {loss_type}"


class TestFeatureMatchingLoss:
    """Test FeatureMatchingLoss functionality."""
    
    def test_feature_matching_computation(self):
        """Test feature matching loss computation."""
        loss_fn = FeatureMatchingLoss(normalize_features=True)
        
        # Create mock features: [scale][layer]
        real_features = [
            [torch.randn(2, 64, 100), torch.randn(2, 128, 50)],  # Scale 0
            [torch.randn(2, 64, 50), torch.randn(2, 128, 25)]    # Scale 1
        ]
        
        fake_features = [
            [torch.randn(2, 64, 100), torch.randn(2, 128, 50)],  # Scale 0
            [torch.randn(2, 64, 50), torch.randn(2, 128, 25)]    # Scale 1
        ]
        
        loss = loss_fn(real_features, fake_features)
        
        assert isinstance(loss, torch.Tensor), "Loss should be tensor"
        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Feature matching loss should be non-negative"
        assert loss.requires_grad, "Loss should require gradients"
    
    def test_normalization_effect(self):
        """Test feature normalization effect."""
        loss_with_norm = FeatureMatchingLoss(normalize_features=True)
        loss_without_norm = FeatureMatchingLoss(normalize_features=False)
        
        # Create features with different scales
        real_features = [[torch.randn(1, 32, 50) * 10]]  # Large scale
        fake_features = [[torch.randn(1, 32, 50) * 0.1]]  # Small scale
        
        loss_norm = loss_with_norm(real_features, fake_features)
        loss_no_norm = loss_without_norm(real_features, fake_features)
        
        # Both should be finite
        assert torch.isfinite(loss_norm), "Normalized loss should be finite"
        assert torch.isfinite(loss_no_norm), "Non-normalized loss should be finite"
    
    def test_kwargs_acceptance(self):
        """Test that loss accepts unknown kwargs."""
        loss_fn = FeatureMatchingLoss(
            unknown_param=123,
            another_param="test"
        )
        
        real_features = [[torch.randn(1, 16, 20)]]
        fake_features = [[torch.randn(1, 16, 20)]]
        
        loss = loss_fn(real_features, fake_features)
        assert loss is not None, "Should work with unknown kwargs"


class TestConditionalWaveDiscriminator:
    """Test ConditionalWaveDiscriminator functionality."""
    
    def test_concat_conditioning(self):
        """Test concatenation-based conditioning."""
        discriminator = ConditionalWaveDiscriminator(
            conditioning_dim=64,
            conditioning_method='concat',
            base_channels=32
        )
        
        batch_size = 2
        x = torch.randn(batch_size, 1, 4096)
        conditioning = torch.randn(batch_size, 64)
        
        output = discriminator(x, conditioning)
        
        assert output.shape[0] == batch_size, "Should preserve batch size"
        assert output.requires_grad, "Output should require gradients"
        
        # Test with features
        output, features = discriminator(x, conditioning, return_features=True)
        assert isinstance(features, list), "Should return features"
    
    def test_film_conditioning(self):
        """Test FiLM-based conditioning."""
        discriminator = ConditionalWaveDiscriminator(
            conditioning_dim=64,
            conditioning_method='film',
            base_channels=32,
            num_layers=3
        )
        
        batch_size = 2
        x = torch.randn(batch_size, 1, 4096)
        conditioning = torch.randn(batch_size, 64)
        
        output = discriminator(x, conditioning)
        
        assert output.shape[0] == batch_size, "Should preserve batch size"
        assert output.requires_grad, "Output should require gradients"
    
    def test_embed_conditioning(self):
        """Test embedding-based conditioning."""
        discriminator = ConditionalWaveDiscriminator(
            conditioning_dim=32,
            conditioning_method='embed',
            embed_dim=64,
            base_channels=32
        )
        
        batch_size = 2
        x = torch.randn(batch_size, 1, 4096)
        conditioning = torch.randn(batch_size, 32)
        
        output = discriminator(x, conditioning)
        
        assert output.shape[0] == batch_size, "Should preserve batch size"
        assert output.requires_grad, "Output should require gradients"
    
    def test_conditioning_dimension_mismatch(self):
        """Test handling of conditioning dimension mismatches."""
        discriminator = ConditionalWaveDiscriminator(
            conditioning_dim=64,
            conditioning_method='concat'
        )
        
        x = torch.randn(2, 1, 4096)
        wrong_conditioning = torch.randn(2, 32)  # Wrong dimension
        
        # Should handle gracefully or raise clear error
        try:
            output = discriminator(x, wrong_conditioning)
            # If it works, that's fine (broadcasting might handle it)
            assert output is not None
        except RuntimeError as e:
            # If it fails, should be a clear error about dimensions
            assert "size" in str(e).lower() or "dimension" in str(e).lower()


def run_all_tests():
    """Run all test classes."""
    test_classes = [
        TestWaveGANDiscriminator,
        TestMultiScaleWaveDiscriminator,
        TestFeatureMatchingLoss,
        TestConditionalWaveDiscriminator
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n=== Running {test_class.__name__} ===")
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            try:
                # Create instance and run test
                test_instance = test_class()
                getattr(test_instance, test_method)()
                print(f"✅ {test_method}")
                passed_tests += 1
            except Exception as e:
                print(f"❌ {test_method}: {e}")
    
    print(f"\n=== WaveGANDiscriminator Test Results ===")
    print(f"Passed: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    return passed_tests, total_tests


if __name__ == "__main__":
    run_all_tests()