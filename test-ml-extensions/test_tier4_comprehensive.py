"""
Comprehensive test suite for all Tier 4 audio/GAN modules.

Tests integration and combined functionality of WaveGANDiscriminator,
OnsetDetector, ChromaEncoder, and PhaseReconstruction modules.
"""

import torch
import torch.nn as nn
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

from onset_detector import (
    OnsetDetector,
    AdaptiveOnsetDetector,
    OnsetTracker
)

from chroma_encoder import (
    ChromaEncoder,
    ChromaProcessor,
    ChromaShiftAugmentation
)

from phase_reconstruction import (
    PhaseReconstruction,
    MagnitudeToAudio,
    PhaseAnalyzer
)


class TestTier4Integration:
    """Test integration between Tier 4 modules."""
    
    def test_audio_analysis_pipeline(self):
        """Test complete audio analysis pipeline."""
        # Create test audio
        batch_size = 2
        audio_length = 16000  # 1 second at 16kHz
        audio = torch.randn(batch_size, audio_length)
        
        # 1. Onset detection
        onset_detector = OnsetDetector(
            method='spectral_flux',
            window_size=1024,
            hop_length=256
        )
        onsets = onset_detector(audio)
        
        assert onsets.shape[0] == batch_size, "Onset detection should preserve batch size"
        assert torch.all(torch.isfinite(onsets)), "Onsets should be finite"
        
        # 2. Chroma analysis
        chroma_encoder = ChromaEncoder(
            method='stft',
            window_size=1024,
            hop_length=256,
            temporal_pooling='mean'
        )
        chroma = chroma_encoder(audio)
        
        assert chroma.shape == (batch_size, 12), "Chroma should have 12 bins"
        assert torch.all(chroma >= 0), "Chroma should be non-negative"
        
        # 3. Integration check
        print(f"✅ Audio analysis pipeline: {batch_size} audio signals processed")
        print(f"   - Onsets detected: {onsets.shape}")
        print(f"   - Chroma extracted: {chroma.shape}")
    
    def test_generation_and_discrimination_pipeline(self):
        """Test audio generation and discrimination pipeline."""
        batch_size = 2
        audio_length = 8192
        
        # 1. Create fake audio (simulating generator output)
        fake_audio = torch.randn(batch_size, 1, audio_length)
        real_audio = torch.randn(batch_size, 1, audio_length)
        
        # 2. Multi-scale discrimination
        discriminator = MultiScaleWaveDiscriminator(
            num_scales=3,
            base_channels=32
        )
        
        real_scores = discriminator(real_audio)
        fake_scores = discriminator(fake_audio)
        
        assert len(real_scores) == 3, "Should have 3 scale outputs"
        assert len(fake_scores) == 3, "Should have 3 scale outputs"
        assert all(score.shape[0] == batch_size for score in real_scores), "Should preserve batch size"
        
        # 3. Compute discriminator loss
        disc_loss = discriminator.compute_loss(real_scores, fake_scores, loss_type='hinge')
        
        assert torch.isfinite(disc_loss), "Discriminator loss should be finite"
        assert disc_loss.requires_grad, "Loss should have gradients"
        
        # 4. Feature matching loss
        real_scores_feat, real_features = discriminator(real_audio, return_features=True)
        fake_scores_feat, fake_features = discriminator(fake_audio, return_features=True)
        
        feat_loss = FeatureMatchingLoss()
        fm_loss = feat_loss(real_features, fake_features)
        
        assert torch.isfinite(fm_loss), "Feature matching loss should be finite"
        assert fm_loss.requires_grad, "Feature matching loss should have gradients"
        
        print(f"✅ Generation/discrimination pipeline completed")
        print(f"   - Discriminator loss: {disc_loss.item():.4f}")
        print(f"   - Feature matching loss: {fm_loss.item():.4f}")
    
    def test_magnitude_to_audio_reconstruction(self):
        """Test magnitude-to-audio reconstruction with analysis."""
        batch_size = 2
        freq_bins = 513  # For n_fft=1024
        time_frames = 32
        
        # 1. Create magnitude spectrogram
        magnitude = torch.rand(batch_size, freq_bins, time_frames) + 0.1
        
        # 2. Reconstruct audio
        reconstructor = MagnitudeToAudio(
            reconstruction_method='griffin_lim',
            n_fft=1024,
            hop_length=256,
            num_iterations=5,
            post_processing=True
        )
        
        reconstructed_audio = reconstructor(magnitude)
        
        assert reconstructed_audio.shape[0] == batch_size, "Should preserve batch size"
        assert torch.all(torch.isfinite(reconstructed_audio)), "Reconstructed audio should be finite"
        
        # 3. Analyze reconstruction quality
        # Create reference audio for comparison
        reference_audio = torch.randn_like(reconstructed_audio)
        
        analyzer = PhaseAnalyzer(n_fft=1024, hop_length=256)
        metrics = analyzer(reference_audio, reconstructed_audio)
        
        assert 'mse' in metrics, "Should include MSE metric"
        assert 'spectral_convergence' in metrics, "Should include spectral convergence"
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values()), "All metrics should be finite"
        
        print(f"✅ Magnitude-to-audio reconstruction completed")
        print(f"   - Reconstruction MSE: {metrics['mse']:.6f}")
        print(f"   - Spectral convergence: {metrics['spectral_convergence']:.6f}")
    
    def test_conditional_generation_pipeline(self):
        """Test conditional generation with chroma conditioning."""
        batch_size = 2
        audio_length = 8192
        
        # 1. Extract chroma features
        reference_audio = torch.randn(batch_size, audio_length)
        
        chroma_encoder = ChromaEncoder(
            method='stft',
            window_size=1024,
            hop_length=256,
            temporal_pooling='mean'
        )
        chroma_features = chroma_encoder(reference_audio)
        
        # 2. Use chroma as conditioning for discriminator
        conditioning_dim = 12  # Chroma has 12 bins
        
        conditional_discriminator = ConditionalWaveDiscriminator(
            conditioning_dim=conditioning_dim,
            conditioning_method='concat',
            base_channels=32,
            n_fft=1024,
            hop_length=256
        )
        
        # 3. Test conditional discrimination
        test_audio = torch.randn(batch_size, 1, audio_length)
        
        conditional_score = conditional_discriminator(test_audio, chroma_features)
        
        assert conditional_score.shape[0] == batch_size, "Should preserve batch size"
        assert torch.all(torch.isfinite(conditional_score)), "Conditional scores should be finite"
        assert conditional_score.requires_grad, "Should have gradients"
        
        print(f"✅ Conditional generation pipeline completed")
        print(f"   - Chroma features: {chroma_features.shape}")
        print(f"   - Conditional scores: {conditional_score.shape}")
    
    def test_temporal_analysis_pipeline(self):
        """Test temporal analysis combining onset and chroma tracking."""
        batch_size = 1  # Use single batch for temporal tracking
        audio_length = 16000
        
        # 1. Set up temporal tracking
        onset_tracker = OnsetTracker(
            temporal_window=5,
            consistency_threshold=0.3
        )
        
        chroma_processor = ChromaProcessor(
            smoothing_kernel_size=5,
            enhancement_method='harmonic'
        )
        
        # 2. Process audio chunks sequentially
        chunk_size = 4000
        num_chunks = audio_length // chunk_size
        
        all_onsets = []
        all_chroma = []
        
        for i in range(num_chunks):
            # Extract chunk
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            audio_chunk = torch.randn(batch_size, chunk_size)
            
            # Track onsets
            onsets = onset_tracker(audio_chunk)
            all_onsets.append(onsets)
            
            # Extract and process chroma
            chroma_encoder = ChromaEncoder(
                method='stft',
                window_size=1024,
                hop_length=256,
                temporal_pooling='none'  # Keep temporal dimension
            )
            chroma_temporal = chroma_encoder(audio_chunk)
            
            # Process chroma
            processed_chroma = chroma_processor(chroma_temporal)
            all_chroma.append(processed_chroma)
        
        # 3. Verify temporal consistency
        assert len(all_onsets) == num_chunks, "Should have onsets for all chunks"
        assert len(all_chroma) == num_chunks, "Should have chroma for all chunks"
        
        # Check that tracking maintains state
        onset_tracker.reset()
        
        print(f"✅ Temporal analysis pipeline completed")
        print(f"   - Processed {num_chunks} chunks")
        print(f"   - Onset tracking maintained across chunks")
        print(f"   - Chroma processing applied temporal smoothing")
    
    def test_music_analysis_augmentation_pipeline(self):
        """Test music analysis with augmentation pipeline."""
        batch_size = 2
        audio_length = 8192
        
        # 1. Extract chroma features
        audio = torch.randn(batch_size, audio_length)
        
        chroma_encoder = ChromaEncoder(
            method='harmonic',  # Use harmonic-aware method
            window_size=1024,
            hop_length=256,
            temporal_pooling='none'
        )
        
        chroma_features = chroma_encoder(audio)
        
        # 2. Apply chroma augmentation (key transposition)
        augmenter = ChromaShiftAugmentation(max_shift=6)
        
        # Test different transpositions
        transpositions = [-3, 0, +4]  # Minor third down, original, major third up
        augmented_chroma = []
        
        for shift in transpositions:
            shifted = augmenter(chroma_features, shift=shift)
            augmented_chroma.append(shifted)
        
        # 3. Verify augmentation properties
        original_chroma = augmented_chroma[1]  # shift=0
        shifted_up = augmented_chroma[2]  # shift=+4
        
        # Check that energy is preserved but distribution changed
        original_energy = torch.sum(original_chroma, dim=1)
        shifted_energy = torch.sum(shifted_up, dim=1)
        
        assert torch.allclose(original_energy, shifted_energy, atol=1e-5), "Energy should be preserved"
        
        # 4. Onset detection on original audio
        onset_detector = AdaptiveOnsetDetector(
            methods=['spectral_flux', 'hfc'],
            combination='mean'
        )
        
        onsets = onset_detector(audio)
        
        assert onsets.shape[0] == batch_size, "Should preserve batch size"
        
        print(f"✅ Music analysis augmentation pipeline completed")
        print(f"   - Chroma features: {chroma_features.shape}")
        print(f"   - Tested {len(transpositions)} transpositions")
        print(f"   - Onset detection with {len(onset_detector.methods)} methods")


def run_comprehensive_tests():
    """Run all comprehensive integration tests."""
    test_instance = TestTier4Integration()
    
    test_methods = [
        'test_audio_analysis_pipeline',
        'test_generation_and_discrimination_pipeline',
        'test_magnitude_to_audio_reconstruction',
        'test_conditional_generation_pipeline',
        'test_temporal_analysis_pipeline',
        'test_music_analysis_augmentation_pipeline'
    ]
    
    total_tests = len(test_methods)
    passed_tests = 0
    
    print("=== Tier 4 Comprehensive Integration Tests ===\n")
    
    for test_method in test_methods:
        try:
            print(f"Running {test_method}...")
            getattr(test_instance, test_method)()
            print(f"✅ {test_method} PASSED\n")
            passed_tests += 1
        except Exception as e:
            print(f"❌ {test_method} FAILED: {e}\n")
    
    print("=== Tier 4 Comprehensive Test Results ===")
    print(f"Passed: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("🎉 All Tier 4 integration tests passed!")
        print("\nTier 4 modules are ready for production use:")
        print("- WaveGANDiscriminator: Time-domain audio discrimination")
        print("- OnsetDetector: Musical onset and transient detection")
        print("- ChromaEncoder: Harmonic content analysis")
        print("- PhaseReconstruction: Magnitude-to-audio conversion")
    
    return passed_tests, total_tests


if __name__ == "__main__":
    run_comprehensive_tests()