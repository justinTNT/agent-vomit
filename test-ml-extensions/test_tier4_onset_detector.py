"""
Test suite for OnsetDetector module.

Tests onset detection algorithms, thresholding methods, adaptive detection,
and temporal tracking for music information retrieval.
"""

import torch
import torch.nn as nn
import pytest
import sys
import os
import math

# Add the audio_gan_tier4 directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'audio_gan_tier4'))

from onset_detector import (
    OnsetDetector,
    SpectralFluxDetector,
    PhaseDeviationDetector,
    ComplexDomainDetector,
    HighFrequencyContentDetector,
    AdaptiveOnsetDetector,
    OnsetTracker
)


class TestOnsetDetector:
    """Test basic OnsetDetector functionality."""
    
    def test_basic_onset_detection(self):
        """Test basic onset detection functionality."""
        detector = OnsetDetector(
            method='spectral_flux',
            threshold_mode='fixed',
            window_size=1024,
            hop_length=256
        )
        
        # Create test audio with clear onset (impulse)
        batch_size = 2
        audio_length = 8192
        audio = torch.zeros(batch_size, audio_length)
        
        # Add impulse onsets
        audio[0, 1000] = 1.0  # Sharp onset
        audio[1, 2000] = 0.8  # Another onset
        
        # Detect onsets
        onsets = detector(audio)
        
        assert onsets.shape[0] == batch_size, "Should preserve batch size"
        assert onsets.dim() == 2, "Should return 2D tensor (batch, onsets)"
        assert torch.all(torch.isfinite(onsets)), "All onsets should be finite"
        
        # Test with return_function
        onsets, detection_func = detector(audio, return_function=True)
        
        assert detection_func.shape[0] == batch_size, "Detection function should preserve batch size"
        assert detection_func.dim() == 2, "Detection function should be 2D"
        assert torch.all(detection_func >= 0), "Detection function should be non-negative"
    
    def test_different_methods(self):
        """Test different onset detection methods."""
        methods = ['spectral_flux', 'phase_deviation', 'complex_domain', 'hfc']
        
        audio = torch.randn(1, 4096)
        
        for method in methods:
            detector = OnsetDetector(method=method, window_size=512, hop_length=128)
            
            onsets = detector(audio)
            
            assert onsets.shape[0] == 1, f"Method {method} should preserve batch size"
            assert torch.all(torch.isfinite(onsets)), f"Method {method} should produce finite onsets"
    
    def test_threshold_modes(self):
        """Test different thresholding modes."""
        threshold_modes = ['fixed', 'adaptive', 'peak_picking']
        
        audio = torch.randn(1, 4096)
        
        for mode in threshold_modes:
            detector = OnsetDetector(
                method='spectral_flux',
                threshold_mode=mode,
                window_size=512
            )
            
            onsets = detector(audio)
            
            assert onsets.shape[0] == 1, f"Threshold mode {mode} should preserve batch size"
            assert torch.all(torch.isfinite(onsets)), f"Threshold mode {mode} should produce finite onsets"
    
    def test_input_shapes(self):
        """Test different input shapes."""
        detector = OnsetDetector(window_size=512, hop_length=128)
        
        # Test different input shapes
        test_cases = [
            (4096,),           # Single audio signal
            (2, 4096),         # Batch of mono signals
            (2, 1, 4096),      # Batch of mono signals with channel dim
            (2, 2, 4096),      # Batch of stereo signals
        ]
        
        for shape in test_cases:
            audio = torch.randn(*shape)
            onsets = detector(audio)
            
            expected_batch = shape[0] if len(shape) > 1 else 1
            assert onsets.shape[0] == expected_batch, f"Shape {shape} should produce correct batch size"
    
    def test_edge_cases(self):
        """Test edge cases."""
        detector = OnsetDetector(window_size=512, hop_length=128)
        
        # Test silence
        silence = torch.zeros(1, 2048)
        onsets = detector(silence)
        assert onsets is not None, "Should handle silence"
        
        # Test very short audio
        short_audio = torch.randn(1, 100)
        onsets = detector(short_audio)
        assert onsets is not None, "Should handle short audio"
        
        # Test single sample
        single_sample = torch.randn(1, 1)
        onsets = detector(single_sample)
        assert onsets is not None, "Should handle single sample"
    
    def test_kwargs_acceptance(self):
        """Test that detector accepts unknown kwargs."""
        detector = OnsetDetector(
            unknown_param=123,
            another_param="test"
        )
        
        audio = torch.randn(1, 2048)
        onsets = detector(audio)
        assert onsets is not None, "Should work with unknown kwargs"


class TestIndividualDetectors:
    """Test individual detector components."""
    
    def test_spectral_flux_detector(self):
        """Test spectral flux detector."""
        detector = SpectralFluxDetector()
        
        # Create mock STFT
        batch_size = 2
        freq_bins = 513
        time_frames = 32
        stft = torch.randn(batch_size, freq_bins, time_frames, dtype=torch.complex64)
        
        flux = detector(stft)
        
        assert flux.shape[0] == batch_size, "Should preserve batch size"
        assert flux.shape[1] == time_frames - 1, "Should have time_frames - 1 output"
        assert torch.all(flux >= 0), "Spectral flux should be non-negative"
    
    def test_phase_deviation_detector(self):
        """Test phase deviation detector."""
        detector = PhaseDeviationDetector()
        
        batch_size = 2
        freq_bins = 513
        time_frames = 10
        stft = torch.randn(batch_size, freq_bins, time_frames, dtype=torch.complex64)
        
        deviation = detector(stft)
        
        assert deviation.shape[0] == batch_size, "Should preserve batch size"
        expected_frames = max(1, time_frames - 2)
        assert deviation.shape[1] == expected_frames, "Should handle time dimension correctly"
        assert torch.all(torch.isfinite(deviation)), "Phase deviation should be finite"
    
    def test_complex_domain_detector(self):
        """Test complex domain detector."""
        detector = ComplexDomainDetector()
        
        batch_size = 2
        freq_bins = 513
        time_frames = 10
        stft = torch.randn(batch_size, freq_bins, time_frames, dtype=torch.complex64)
        
        complex_diff = detector(stft)
        
        assert complex_diff.shape[0] == batch_size, "Should preserve batch size"
        expected_frames = max(1, time_frames - 1)
        assert complex_diff.shape[1] == expected_frames, "Should handle time dimension correctly"
        assert torch.all(complex_diff >= 0), "Complex domain difference should be non-negative"
    
    def test_hfc_detector(self):
        """Test high-frequency content detector."""
        detector = HighFrequencyContentDetector()
        
        batch_size = 2
        freq_bins = 513
        time_frames = 10
        stft = torch.randn(batch_size, freq_bins, time_frames, dtype=torch.complex64)
        
        hfc = detector(stft)
        
        assert hfc.shape[0] == batch_size, "Should preserve batch size"
        assert torch.all(torch.isfinite(hfc)), "HFC should be finite"
        assert torch.all(hfc >= 0), "HFC should be non-negative"


class TestAdaptiveOnsetDetector:
    """Test adaptive onset detector."""
    
    def test_adaptive_detection(self):
        """Test adaptive onset detection."""
        detector = AdaptiveOnsetDetector(
            methods=['spectral_flux', 'hfc'],
            combination='mean',
            window_size=512,
            hop_length=128
        )
        
        audio = torch.randn(2, 4096)
        onsets = detector(audio)
        
        assert onsets.shape[0] == 2, "Should preserve batch size"
        assert torch.all(torch.isfinite(onsets)), "Combined onsets should be finite"
        
        # Test with individual results
        combined_onsets, individual_results = detector(audio, return_individual=True)
        
        assert len(individual_results) == 2, "Should return results for both methods"
        assert all(result.shape[0] == 2 for result in individual_results), "All methods should preserve batch size"
    
    def test_combination_methods(self):
        """Test different combination methods."""
        methods = ['spectral_flux', 'hfc']
        combinations = ['mean', 'max', 'weighted']
        
        audio = torch.randn(1, 2048)
        
        for combination in combinations:
            if combination == 'weighted':
                detector = AdaptiveOnsetDetector(
                    methods=methods,
                    combination=combination,
                    weights=[0.6, 0.4],
                    window_size=512
                )
            else:
                detector = AdaptiveOnsetDetector(
                    methods=methods,
                    combination=combination,
                    window_size=512
                )
            
            onsets = detector(audio)
            
            assert onsets.shape[0] == 1, f"Combination {combination} should preserve batch size"
            assert torch.all(torch.isfinite(onsets)), f"Combination {combination} should produce finite onsets"
    
    def test_single_method_adaptive(self):
        """Test adaptive detector with single method."""
        detector = AdaptiveOnsetDetector(
            methods=['spectral_flux'],
            combination='mean',
            window_size=512
        )
        
        audio = torch.randn(1, 2048)
        onsets = detector(audio)
        
        assert onsets.shape[0] == 1, "Single method should work"
        assert torch.all(torch.isfinite(onsets)), "Single method should produce finite onsets"


class TestOnsetTracker:
    """Test onset tracker with temporal consistency."""
    
    def test_onset_tracking(self):
        """Test basic onset tracking."""
        base_detector = OnsetDetector(window_size=512, hop_length=128)
        tracker = OnsetTracker(
            detector=base_detector,
            temporal_window=3,
            consistency_threshold=0.2
        )
        
        audio = torch.randn(1, 2048)
        tracked_onsets = tracker(audio)
        
        assert tracked_onsets.shape[0] == 1, "Should preserve batch size"
        assert torch.all(torch.isfinite(tracked_onsets)), "Tracked onsets should be finite"
    
    def test_tracker_reset(self):
        """Test tracker reset functionality."""
        tracker = OnsetTracker(temporal_window=5)
        
        # Process some audio
        audio1 = torch.randn(1, 2048)
        tracker(audio1)
        
        # Check history is non-zero
        assert not torch.allclose(tracker.history_buffer, torch.zeros_like(tracker.history_buffer)), \
            "History should be updated"
        
        # Reset and check
        tracker.reset()
        assert torch.allclose(tracker.history_buffer, torch.zeros_like(tracker.history_buffer)), \
            "History should be reset"
        assert tracker.buffer_index.item() == 0, "Buffer index should be reset"
    
    def test_temporal_consistency(self):
        """Test temporal consistency behavior."""
        tracker = OnsetTracker(
            temporal_window=3,
            consistency_threshold=0.1  # Low threshold for more conservative behavior
        )
        
        # Process multiple frames
        for _ in range(5):
            audio = torch.randn(1, 2048)
            onsets = tracker(audio)
            assert onsets is not None, "Should produce onsets"
    
    def test_default_detector(self):
        """Test tracker with default detector."""
        tracker = OnsetTracker()  # Should create default detector
        
        audio = torch.randn(1, 2048)
        onsets = tracker(audio)
        
        assert onsets is not None, "Should work with default detector"
        assert onsets.shape[0] == 1, "Should preserve batch size"
    
    def test_kwargs_acceptance(self):
        """Test that tracker accepts unknown kwargs."""
        tracker = OnsetTracker(
            unknown_param=123,
            another_param="test"
        )
        
        audio = torch.randn(1, 2048)
        onsets = tracker(audio)
        assert onsets is not None, "Should work with unknown kwargs"


def run_all_tests():
    """Run all test classes."""
    test_classes = [
        TestOnsetDetector,
        TestIndividualDetectors,
        TestAdaptiveOnsetDetector,
        TestOnsetTracker
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
    
    print(f"\n=== OnsetDetector Test Results ===")
    print(f"Passed: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    return passed_tests, total_tests


if __name__ == "__main__":
    run_all_tests()