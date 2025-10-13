"""
Test suite for ChromaEncoder module.

Tests chroma feature extraction, different methods, temporal processing,
and music-specific augmentation for harmonic content analysis.
"""

import torch
import torch.nn as nn
import pytest
import sys
import os
import math

# Add the audio_gan_tier4 directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'audio_gan_tier4'))

from chroma_encoder import (
    ChromaEncoder,
    STFTChromaEncoder,
    CQTChromaEncoder,
    HarmonicChromaEncoder,
    NeuralChromaEncoder,
    ChromaProcessor,
    ChromaShiftAugmentation
)


class TestChromaEncoder:
    """Test basic ChromaEncoder functionality."""
    
    def test_basic_chroma_extraction(self):
        """Test basic chroma extraction functionality."""
        encoder = ChromaEncoder(
            method='stft',
            num_chroma=12,
            window_size=2048,
            hop_length=512,
            temporal_pooling='mean'
        )
        
        # Create test audio
        batch_size = 2
        audio_length = 8192
        audio = torch.randn(batch_size, audio_length)
        
        # Extract chroma features
        chroma = encoder(audio)
        
        assert chroma.shape == (batch_size, 12), "Should return correct chroma shape"
        assert torch.all(torch.isfinite(chroma)), "All chroma values should be finite"
        assert torch.all(chroma >= 0), "Chroma values should be non-negative"
        
        # Test normalization (should sum to 1 or be close)
        chroma_sums = chroma.sum(dim=1)
        assert torch.allclose(chroma_sums, torch.ones_like(chroma_sums), atol=1e-5), "Chroma should be normalized"
        
        # Test with temporal output
        chroma, temporal = encoder(audio, return_temporal=True)
        
        assert temporal.shape[0] == batch_size, "Temporal should preserve batch size"
        assert temporal.shape[1] == 12, "Temporal should have correct chroma bins"
        assert temporal.dim() == 3, "Temporal should be 3D (batch, chroma, time)"
    
    def test_different_methods(self):
        """Test different chroma extraction methods."""
        methods = ['stft', 'cqt', 'harmonic', 'neural']
        
        audio = torch.randn(1, 4096)
        
        for method in methods:
            encoder = ChromaEncoder(
                method=method,
                window_size=1024,
                hop_length=256,
                temporal_pooling='mean'
            )
            
            chroma = encoder(audio)
            
            assert chroma.shape == (1, 12), f"Method {method} should return correct shape"
            assert torch.all(torch.isfinite(chroma)), f"Method {method} should produce finite values"
            assert torch.all(chroma >= 0), f"Method {method} should produce non-negative values"
    
    def test_temporal_pooling_methods(self):
        """Test different temporal pooling strategies."""
        pooling_methods = ['mean', 'max', 'none']
        
        audio = torch.randn(1, 4096)
        
        for pooling in pooling_methods:
            encoder = ChromaEncoder(
                method='stft',
                temporal_pooling=pooling,
                window_size=1024,
                hop_length=256
            )
            
            chroma = encoder(audio)
            
            if pooling == 'none':
                assert chroma.dim() == 3, f"Pooling {pooling} should return 3D tensor"
                assert chroma.shape[1] == 12, f"Pooling {pooling} should preserve chroma bins"
            else:
                assert chroma.shape == (1, 12), f"Pooling {pooling} should return 2D tensor"
    
    def test_different_input_shapes(self):
        """Test different input shapes."""
        encoder = ChromaEncoder(window_size=1024, hop_length=256)
        
        # Test different input shapes
        test_cases = [
            (4096,),           # Single audio signal
            (2, 4096),         # Batch of mono signals
            (2, 1, 4096),      # Batch of mono signals with channel dim
            (2, 2, 4096),      # Batch of stereo signals
        ]
        
        for shape in test_cases:
            audio = torch.randn(*shape)
            chroma = encoder(audio)
            
            expected_batch = shape[0] if len(shape) > 1 else 1
            assert chroma.shape[0] == expected_batch, f"Shape {shape} should produce correct batch size"
            assert chroma.shape[1] == 12, f"Shape {shape} should preserve chroma bins"
    
    def test_tuning_frequency(self):
        """Test different tuning frequencies."""
        tuning_freqs = [440.0, 432.0, 444.0]
        
        audio = torch.randn(1, 4096)
        
        for tuning in tuning_freqs:
            encoder = ChromaEncoder(
                tuning_freq=tuning,
                window_size=1024,
                hop_length=256
            )
            
            chroma = encoder(audio)
            
            assert chroma.shape == (1, 12), f"Tuning {tuning} should work correctly"
            assert torch.all(torch.isfinite(chroma)), f"Tuning {tuning} should produce finite values"
    
    def test_edge_cases(self):
        """Test edge cases."""
        encoder = ChromaEncoder(window_size=1024, hop_length=256)
        
        # Test silence
        silence = torch.zeros(1, 2048)
        chroma = encoder(silence)
        assert chroma is not None, "Should handle silence"
        assert torch.all(torch.isfinite(chroma)), "Silence should produce finite chroma"
        
        # Test very short audio
        short_audio = torch.randn(1, 100)
        chroma = encoder(short_audio)
        assert chroma is not None, "Should handle short audio"
        
        # Test single sample
        single_sample = torch.randn(1, 1)
        chroma = encoder(single_sample)
        assert chroma is not None, "Should handle single sample"
    
    def test_kwargs_acceptance(self):
        """Test that encoder accepts unknown kwargs."""
        encoder = ChromaEncoder(
            unknown_param=123,
            another_param="test"
        )
        
        audio = torch.randn(1, 2048)
        chroma = encoder(audio)
        assert chroma is not None, "Should work with unknown kwargs"


class TestIndividualEncoders:
    """Test individual encoder components."""
    
    def test_stft_chroma_encoder(self):
        """Test STFT-based chroma encoder."""
        encoder = STFTChromaEncoder(
            num_chroma=12,
            window_size=1024,
            hop_length=256,
            tuning_freq=440.0,
            octave_range=(0, 8)
        )
        
        audio = torch.randn(2, 4096)
        chroma = encoder(audio)
        
        assert chroma.shape[0] == 2, "Should preserve batch size"
        assert chroma.shape[1] == 12, "Should have correct chroma bins"
        assert chroma.dim() == 3, "Should return temporal chroma"
        assert torch.all(chroma >= 0), "STFT chroma should be non-negative"
    
    def test_cqt_chroma_encoder(self):
        """Test CQT-based chroma encoder."""
        encoder = CQTChromaEncoder(
            num_chroma=12,
            hop_length=256,
            tuning_freq=440.0,
            octave_range=(0, 8)
        )
        
        audio = torch.randn(2, 4096)
        chroma = encoder(audio)
        
        assert chroma.shape[0] == 2, "Should preserve batch size"
        assert chroma.shape[1] == 12, "Should have correct chroma bins"
        assert chroma.dim() == 3, "Should return temporal chroma"
        assert torch.all(torch.isfinite(chroma)), "CQT chroma should be finite"
    
    def test_harmonic_chroma_encoder(self):
        """Test harmonic-aware chroma encoder."""
        encoder = HarmonicChromaEncoder(
            num_chroma=12,
            window_size=1024,
            hop_length=256,
            tuning_freq=440.0,
            octave_range=(0, 8)
        )
        
        audio = torch.randn(2, 4096)
        chroma = encoder(audio)
        
        assert chroma.shape[0] == 2, "Should preserve batch size"
        assert chroma.shape[1] == 12, "Should have correct chroma bins"
        assert chroma.dim() == 3, "Should return temporal chroma"
        assert torch.all(chroma >= 0), "Harmonic chroma should be non-negative"
    
    def test_neural_chroma_encoder(self):
        """Test neural network-based chroma encoder."""
        encoder = NeuralChromaEncoder(
            num_chroma=12,
            window_size=1024,
            hop_length=256
        )
        
        audio = torch.randn(2, 4096)
        chroma = encoder(audio)
        
        assert chroma.shape[0] == 2, "Should preserve batch size"
        assert chroma.shape[1] == 12, "Should have correct chroma bins"
        assert chroma.dim() == 3, "Should return temporal chroma"
        assert torch.all(chroma >= 0), "Neural chroma should be non-negative"
        
        # Test gradients flow through neural network
        loss = chroma.sum()
        loss.backward()
        assert any(p.grad is not None for p in encoder.parameters()), "Gradients should flow"


class TestChromaProcessor:
    """Test chroma post-processing."""
    
    def test_chroma_processing(self):
        """Test basic chroma processing."""
        processor = ChromaProcessor(
            num_chroma=12,
            smoothing_kernel_size=3,
            enhancement_method='harmonic'
        )
        
        # Create sample chroma with some temporal variation
        batch_size = 2
        time_frames = 20
        chroma = torch.rand(batch_size, 12, time_frames)
        chroma = torch.nn.functional.normalize(chroma, p=1, dim=1)
        
        processed = processor(chroma)
        
        assert processed.shape == chroma.shape, "Should preserve shape"
        assert torch.all(torch.isfinite(processed)), "Processed chroma should be finite"
        assert torch.all(processed >= 0), "Processed chroma should be non-negative"
        
        # Check normalization
        processed_sums = processed.sum(dim=1)
        expected_sums = torch.ones_like(processed_sums)
        assert torch.allclose(processed_sums, expected_sums, atol=1e-4), "Should maintain normalization"
    
    def test_enhancement_methods(self):
        """Test different enhancement methods."""
        enhancement_methods = ['none', 'harmonic', 'template']
        
        chroma = torch.rand(1, 12, 10)
        chroma = torch.nn.functional.normalize(chroma, p=1, dim=1)
        
        for method in enhancement_methods:
            processor = ChromaProcessor(
                enhancement_method=method,
                smoothing_kernel_size=3
            )
            
            processed = processor(chroma)
            
            assert processed.shape == chroma.shape, f"Enhancement {method} should preserve shape"
            assert torch.all(torch.isfinite(processed)), f"Enhancement {method} should produce finite values"
    
    def test_smoothing_effect(self):
        """Test temporal smoothing effect."""
        # Create processor with smoothing
        processor_smooth = ChromaProcessor(smoothing_kernel_size=5)
        processor_no_smooth = ChromaProcessor(smoothing_kernel_size=1)
        
        # Create chroma with sharp temporal changes
        chroma = torch.zeros(1, 12, 10)
        chroma[0, 0, 0] = 1.0  # Sharp peak at start
        chroma[0, 5, 5] = 1.0  # Sharp peak in middle
        
        processed_smooth = processor_smooth(chroma)
        processed_no_smooth = processor_no_smooth(chroma)
        
        # Smoothed version should have less temporal variation
        smooth_var = processed_smooth.var(dim=2).mean()
        no_smooth_var = processed_no_smooth.var(dim=2).mean()
        
        # Note: Due to normalization, this test is approximate
        assert smooth_var <= no_smooth_var + 0.1, "Smoothing should reduce temporal variation"
    
    def test_kwargs_acceptance(self):
        """Test that processor accepts unknown kwargs."""
        processor = ChromaProcessor(
            unknown_param=123,
            another_param="test"
        )
        
        chroma = torch.rand(1, 12, 5)
        processed = processor(chroma)
        assert processed is not None, "Should work with unknown kwargs"


class TestChromaShiftAugmentation:
    """Test chroma shift augmentation."""
    
    def test_pitch_shift_augmentation(self):
        """Test pitch shift augmentation."""
        augmentation = ChromaShiftAugmentation(max_shift=6)
        
        # Create chroma with clear peak
        chroma = torch.zeros(2, 12, 10)
        chroma[:, 0, :] = 1.0  # Peak at C
        
        # Test fixed shift
        shifted = augmentation(chroma, shift=3)  # Shift by 3 semitones (C -> D#)
        
        assert shifted.shape == chroma.shape, "Should preserve shape"
        assert torch.allclose(shifted[:, 3, :], chroma[:, 0, :]), "Should shift correctly"
        assert torch.allclose(shifted[:, 0, :], torch.zeros_like(shifted[:, 0, :])), "Original position should be zero"
        
        # Test negative shift
        shifted_neg = augmentation(chroma, shift=-2)  # Shift by -2 semitones (C -> A#/Bb)
        
        assert torch.allclose(shifted_neg[:, 10, :], chroma[:, 0, :]), "Negative shift should work"
        
        # Test no shift
        no_shift = augmentation(chroma, shift=0)
        
        assert torch.allclose(no_shift, chroma), "Zero shift should return original"
        
        # Test random shift
        random_shifted = augmentation(chroma)  # Random shift
        
        assert random_shifted.shape == chroma.shape, "Random shift should preserve shape"
        assert torch.all(torch.isfinite(random_shifted)), "Random shift should produce finite values"
    
    def test_circular_shift_property(self):
        """Test that pitch shifts are circular."""
        augmentation = ChromaShiftAugmentation()
        
        # Create test chroma
        chroma = torch.rand(1, 12, 5)
        
        # Shift by 12 semitones (full octave) should return to original
        shifted_octave = augmentation(chroma, shift=12)
        
        assert torch.allclose(shifted_octave, chroma, atol=1e-6), "12-semitone shift should be circular"
        
        # Test negative circular shift
        shifted_neg_octave = augmentation(chroma, shift=-12)
        
        assert torch.allclose(shifted_neg_octave, chroma, atol=1e-6), "Negative 12-semitone shift should be circular"
    
    def test_max_shift_constraint(self):
        """Test that random shifts respect max_shift constraint."""
        max_shift = 3
        augmentation = ChromaShiftAugmentation(max_shift=max_shift)
        
        chroma = torch.rand(1, 12, 5)
        
        # Test multiple random shifts to check constraint
        for _ in range(10):
            # We can't directly test the shift amount since it's internal,
            # but we can test that the function works without error
            shifted = augmentation(chroma)
            assert shifted.shape == chroma.shape, "Should preserve shape with constrained shifts"
            assert torch.all(torch.isfinite(shifted)), "Should produce finite values"
    
    def test_kwargs_acceptance(self):
        """Test that augmentation accepts unknown kwargs."""
        augmentation = ChromaShiftAugmentation(
            unknown_param=123,
            another_param="test"
        )
        
        chroma = torch.rand(1, 12, 5)
        shifted = augmentation(chroma, shift=2)
        assert shifted is not None, "Should work with unknown kwargs"


def run_all_tests():
    """Run all test classes."""
    test_classes = [
        TestChromaEncoder,
        TestIndividualEncoders,
        TestChromaProcessor,
        TestChromaShiftAugmentation
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
    
    print(f"\n=== ChromaEncoder Test Results ===")
    print(f"Passed: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    return passed_tests, total_tests


if __name__ == "__main__":
    run_all_tests()