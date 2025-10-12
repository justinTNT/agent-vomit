import torch
import pytest
import numpy as np
from modules.antialiased_conv import (
    LowPassFilter1d,
    AntialiasedConv1d, 
    AntialiasedConvTranspose1d,
    BlurPool1d
)


def test_low_pass_filter():
    """Test low-pass filter functionality."""
    # Test different filter types
    for filter_type in ['lanczos', 'gaussian', 'butterworth', 'box']:
        lpf = LowPassFilter1d(filter_type=filter_type, filter_size=5, stride=2)
        
        # Test on simple signal
        x = torch.randn(2, 3, 100)
        y = lpf(x)
        
        assert y.shape == x.shape
        assert torch.all(torch.isfinite(y))
        
        # Check that kernel sums to 1 (normalized)
        assert torch.allclose(lpf.kernel.sum(), torch.tensor(1.0), atol=1e-6)


def test_antialiased_conv_basic():
    """Test basic anti-aliased convolution."""
    conv = AntialiasedConv1d(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        stride=2,
        padding=1
    )
    
    x = torch.randn(2, 3, 100)
    y = conv(x)
    
    # Output should be downsampled by stride
    expected_length = 50  # 100 / 2
    assert y.shape == (2, 16, expected_length)
    assert torch.all(torch.isfinite(y))


def test_antialiasing_effect():
    """Test that anti-aliasing actually reduces aliasing."""
    # Create a high-frequency signal that would alias
    t = torch.linspace(0, 1, 1000)
    # Signal with frequency above Nyquist for stride=4
    high_freq = torch.sin(2 * np.pi * 100 * t)  
    x = high_freq.unsqueeze(0).unsqueeze(0)  # [1, 1, 1000]
    
    # Regular strided conv (will alias)
    regular_conv = torch.nn.Conv1d(1, 1, kernel_size=1, stride=4)
    regular_conv.weight.data.fill_(1.0)
    regular_conv.bias.data.fill_(0.0)
    
    # Anti-aliased conv
    aa_conv = AntialiasedConv1d(1, 1, kernel_size=1, stride=4)
    aa_conv.conv.weight.data.fill_(1.0)
    aa_conv.conv.bias.data.fill_(0.0)
    
    y_regular = regular_conv(x)
    y_aa = aa_conv(x)
    
    # Anti-aliased version should have lower high-frequency content
    # (Lower standard deviation indicates smoother signal)
    assert y_aa.std() < y_regular.std()


def test_no_antialiasing_stride_1():
    """Test that stride=1 doesn't apply anti-aliasing."""
    conv = AntialiasedConv1d(
        in_channels=2,
        out_channels=4,
        kernel_size=3,
        stride=1,  # No downsampling
        padding=1
    )
    
    # Should not have filter
    assert conv.filter is None
    
    x = torch.randn(1, 2, 50)
    y = conv(x)
    
    # Output length should be preserved
    assert y.shape == (1, 4, 50)


def test_different_filter_types():
    """Test different anti-aliasing filter types."""
    x = torch.randn(2, 3, 100)
    
    outputs = {}
    for filter_type in ['lanczos', 'gaussian', 'butterworth', 'box']:
        conv = AntialiasedConv1d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            stride=2,
            padding=1,  # Add padding to maintain size calculation
            filter_type=filter_type
        )
        outputs[filter_type] = conv(x)
    
    # All should produce same output shape
    for filter_type, out in outputs.items():
        assert out.shape == (2, 8, 50)
    
    # But different filter types should produce different results
    # (Box filter should be most different from others)
    assert not torch.allclose(outputs['lanczos'], outputs['box'])


def test_antialiased_conv_transpose():
    """Test anti-aliased transposed convolution."""
    conv_t = AntialiasedConvTranspose1d(
        in_channels=16,
        out_channels=8,
        kernel_size=4,
        stride=2,
        padding=1
    )
    
    x = torch.randn(2, 16, 50)
    y = conv_t(x)
    
    # Output should be upsampled
    assert y.shape[0] == 2  # Batch preserved
    assert y.shape[1] == 8  # Output channels
    assert y.shape[2] > 50  # Upsampled


def test_blur_pool():
    """Test blur pooling layer."""
    pool = BlurPool1d(channels=3, stride=2, filter_size=5)
    
    x = torch.randn(2, 3, 100)
    y = pool(x)
    
    # Should downsample by stride
    assert y.shape == (2, 3, 50)
    
    # Should be smoother than simple subsampling
    y_subsample = x[:, :, ::2]
    # Blur pool should have less high-frequency content
    assert torch.diff(y).abs().mean() < torch.diff(y_subsample).abs().mean()


def test_custom_filter_size():
    """Test custom filter sizes."""
    # Larger filter for stronger anti-aliasing
    conv_large = AntialiasedConv1d(
        1, 1, kernel_size=3, stride=4,
        filter_size=9  # Larger than default
    )
    
    # Smaller filter
    conv_small = AntialiasedConv1d(
        1, 1, kernel_size=3, stride=4,
        filter_size=3
    )
    
    x = torch.randn(1, 1, 200)
    y_large = conv_large(x)
    y_small = conv_small(x)
    
    # Both should produce same shape
    assert y_large.shape == y_small.shape == (1, 1, 50)
    
    # But results should differ
    assert not torch.allclose(y_large, y_small)


def test_groups():
    """Test grouped convolution with anti-aliasing."""
    conv = AntialiasedConv1d(
        in_channels=4,
        out_channels=8,
        kernel_size=3,
        stride=2,
        padding=1,
        groups=2
    )
    
    x = torch.randn(2, 4, 100)
    y = conv(x)
    
    assert y.shape == (2, 8, 50)


def test_dilation():
    """Test dilated convolution with anti-aliasing."""
    conv = AntialiasedConv1d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        stride=2,
        dilation=2,
        padding=2
    )
    
    x = torch.randn(1, 1, 100)
    y = conv(x)
    
    assert y.shape[2] == 50


def test_gradient_flow():
    """Test gradient flow through anti-aliased layers."""
    conv = AntialiasedConv1d(3, 8, kernel_size=3, stride=2)
    
    x = torch.randn(2, 3, 64, requires_grad=True)
    y = conv(x)
    loss = y.mean()
    loss.backward()
    
    # Check gradients
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))
    assert conv.conv.weight.grad is not None
    assert torch.all(torch.isfinite(conv.conv.weight.grad))


def test_cutoff_ratio():
    """Test custom cutoff ratio for filter."""
    # Lower cutoff = stronger filtering
    lpf_low = LowPassFilter1d(cutoff_ratio=0.25, stride=2)
    lpf_high = LowPassFilter1d(cutoff_ratio=0.5, stride=2)
    
    x = torch.randn(1, 1, 100)
    y_low = lpf_low(x)
    y_high = lpf_high(x)
    
    # Lower cutoff should produce smoother output
    assert torch.diff(y_low).abs().mean() < torch.diff(y_high).abs().mean()


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    # Very short sequence
    conv = AntialiasedConv1d(1, 1, kernel_size=3, stride=2, padding=1)
    x_short = torch.randn(1, 1, 4)
    y_short = conv(x_short)
    assert y_short.shape[2] == 2
    
    # Single sample (edge case)
    x_single = torch.randn(1, 1, 1)
    y_single = conv(x_single)
    assert y_single.shape[2] >= 0  # Should handle gracefully
    
    # Large stride
    conv_large = AntialiasedConv1d(1, 1, kernel_size=3, stride=8)
    x = torch.randn(1, 1, 64)
    y = conv_large(x)
    assert y.shape[2] == 8


def test_audio_processing_scenario():
    """Test in realistic audio processing context."""
    # Downsample 48kHz to 16kHz (stride=3)
    downsample = AntialiasedConv1d(
        in_channels=1,
        out_channels=1,
        kernel_size=1,  # Just filtering, no feature extraction
        stride=3,
        filter_type='lanczos',
        filter_size=15  # Strong anti-aliasing
    )
    
    # Make it act as pure downsampler
    downsample.conv.weight.data.fill_(1.0)
    downsample.conv.bias.data.fill_(0.0)
    
    # Simulate 1 second of audio at 48kHz
    audio_48k = torch.randn(1, 1, 48000) * 0.5
    audio_16k = downsample(audio_48k)
    
    assert audio_16k.shape == (1, 1, 16000)
    
    # Check that high frequencies are attenuated
    # (This is a simple check - in practice you'd use FFT)
    assert audio_16k.std() < audio_48k.std()


def test_sequential_downsampling():
    """Test sequential application of anti-aliased layers."""
    # Build a downsampling pyramid
    layer1 = AntialiasedConv1d(1, 16, kernel_size=7, stride=2, padding=3)
    layer2 = AntialiasedConv1d(16, 32, kernel_size=5, stride=2, padding=2)
    layer3 = AntialiasedConv1d(32, 64, kernel_size=3, stride=2, padding=1)
    
    x = torch.randn(2, 1, 256)
    
    # Apply layers
    h1 = torch.relu(layer1(x))      # 256 -> 128
    h2 = torch.relu(layer2(h1))     # 128 -> 64
    h3 = torch.relu(layer3(h2))     # 64 -> 32
    
    assert h1.shape == (2, 16, 128)
    assert h2.shape == (2, 32, 64)
    assert h3.shape == (2, 64, 32)


if __name__ == "__main__":
    test_low_pass_filter()
    test_antialiased_conv_basic()
    test_antialiasing_effect()
    test_no_antialiasing_stride_1()
    test_different_filter_types()
    test_antialiased_conv_transpose()
    test_blur_pool()
    test_custom_filter_size()
    test_groups()
    test_dilation()
    test_gradient_flow()
    test_cutoff_ratio()
    test_edge_cases()
    test_audio_processing_scenario()
    test_sequential_downsampling()
    print("All AntialiasedConv tests passed!")