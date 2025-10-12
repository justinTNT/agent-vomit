import torch
import pytest
import torch.nn as nn
from modules.causal_conv import CausalConv1d, CausalConvTranspose1d, CausalPaddingMode


def test_basic_causal_conv():
    """Test basic causal convolution functionality."""
    conv = CausalConv1d(
        in_channels=1,
        out_channels=16,
        kernel_size=3
    )
    
    # Test with simple input
    batch_size = 2
    seq_len = 100
    x = torch.randn(batch_size, 1, seq_len)
    y = conv(x)
    
    # Output should have same sequence length (causal padding compensates)
    assert y.shape == (batch_size, 16, seq_len)
    
    # Check that output is finite
    assert torch.all(torch.isfinite(y))


def test_causality():
    """Test that convolution is truly causal."""
    kernel_size = 5
    conv = CausalConv1d(1, 1, kernel_size, bias=False)
    
    # Set weights to simple averaging
    conv.conv.weight.data.fill_(1.0 / kernel_size)
    
    # Create two signals - one changes halfway through
    x1 = torch.ones(1, 1, 20)
    x2 = torch.ones(1, 1, 20)
    x2[:, :, 10:] = 2.0  # Change happens at position 10
    
    y1 = conv(x1)
    y2 = conv(x2)
    
    # Before position 10, outputs should be identical
    # (future change shouldn't affect past outputs)
    assert torch.allclose(y1[:, :, :10], y2[:, :, :10], atol=1e-6)
    
    # At position 10, y2 should start incorporating the change
    assert not torch.allclose(y1[:, :, 10:], y2[:, :, 10:], atol=1e-6)


def test_different_kernel_sizes():
    """Test causal convolution with various kernel sizes."""
    for kernel_size in [1, 3, 5, 7, 15]:
        conv = CausalConv1d(2, 4, kernel_size)
        x = torch.randn(1, 2, 50)
        y = conv(x)
        
        # Output length should be preserved
        assert y.shape == (1, 4, 50)
        
        # Check causal padding calculation
        assert conv.causal_padding == (kernel_size - 1)


def test_stride():
    """Test causal convolution with stride > 1."""
    conv = CausalConv1d(1, 1, kernel_size=3, stride=2)
    
    x = torch.randn(1, 1, 100)
    y = conv(x)
    
    # Calculate expected output length
    # With padding, input length becomes 100 + causal_padding
    # Then apply stride formula: (L + 2*P - K) // S + 1
    # But we have P=0 in the conv itself, padding is applied before
    padded_len = 100 + conv.causal_padding
    expected_len = (padded_len - conv.kernel_size) // conv.stride + 1
    assert y.shape[2] == expected_len


def test_dilation():
    """Test causal convolution with dilation."""
    conv = CausalConv1d(1, 1, kernel_size=3, dilation=2, bias=False)
    
    # Set specific weights for testing
    conv.conv.weight.data = torch.tensor([[[1.0, 0.0, -1.0]]])
    
    # The dilated kernel sees positions [t-4, t-2, t]
    x = torch.zeros(1, 1, 10)
    x[0, 0, 4] = 1.0  # Set one position to 1
    
    y = conv(x)
    
    # At position 4, kernel sees [0, 0, 1] -> output = -1
    assert torch.allclose(y[0, 0, 4], torch.tensor(-1.0))
    
    # At position 8, kernel sees [1, 0, 0] -> output = 1
    assert torch.allclose(y[0, 0, 8], torch.tensor(1.0))


def test_groups():
    """Test grouped causal convolution."""
    groups = 2
    conv = CausalConv1d(4, 8, kernel_size=3, groups=groups)
    
    x = torch.randn(2, 4, 50)
    y = conv(x)
    
    assert y.shape == (2, 8, 50)
    
    # Verify grouped convolution by checking that groups don't interact
    # Zero out second group of input channels
    x_test = x.clone()
    x_test[:, 2:, :] = 0
    y_test = conv(x_test)
    
    # Second group of output channels should be near zero (except bias)
    if conv.conv.bias is not None:
        y_test[:, 4:, :] -= conv.conv.bias[4:].view(1, -1, 1)
    assert torch.allclose(y_test[:, 4:, :], torch.zeros_like(y_test[:, 4:, :]), atol=1e-6)


def test_padding_modes():
    """Test different padding modes."""
    x = torch.randn(1, 1, 20)
    
    for mode in CausalPaddingMode.all_modes():
        conv = CausalConv1d(1, 1, kernel_size=5, padding_mode=mode)
        y = conv(x)
        assert y.shape == x.shape
        assert torch.all(torch.isfinite(y))


def test_streaming_mode():
    """Test streaming/real-time processing mode."""
    conv = CausalConv1d(1, 1, kernel_size=3)
    
    # For proper streaming test, we need to handle the initial state correctly
    # Let's test with chunks instead of single samples
    x_full = torch.randn(1, 1, 100)
    y_full = conv(x_full)
    
    # Process in chunks
    chunk_size = 10
    outputs = []
    
    for i in range(0, 100, chunk_size):
        x_chunk = x_full[:, :, i:i+chunk_size]
        y_chunk = conv(x_chunk)
        outputs.append(y_chunk)
    
    y_chunked = torch.cat(outputs, dim=-1)
    
    # The chunked processing should give same shape
    assert y_chunked.shape == y_full.shape
    
    # For true streaming with state management, we'd need to modify
    # the implementation to handle state properly


def test_receptive_field():
    """Test receptive field calculation."""
    # Without dilation
    conv1 = CausalConv1d(1, 1, kernel_size=5)
    assert conv1.receptive_field == 5
    
    # With dilation
    conv2 = CausalConv1d(1, 1, kernel_size=3, dilation=4)
    # Receptive field = kernel_size + (kernel_size - 1) * (dilation - 1)
    # = 3 + (3 - 1) * (4 - 1) = 3 + 2 * 3 = 9
    assert conv2.receptive_field == 9


def test_causal_transpose():
    """Test causal transposed convolution."""
    conv_t = CausalConvTranspose1d(
        in_channels=16,
        out_channels=1,
        kernel_size=4,
        stride=2
    )
    
    x = torch.randn(2, 16, 50)
    y = conv_t(x)
    
    # Output should be approximately 2x input length (stride=2)
    # Exact calculation depends on trimming
    assert y.shape[0] == 2  # Batch preserved
    assert y.shape[1] == 1  # Output channels
    assert 95 <= y.shape[2] <= 100  # Approximately 2x length


def test_zero_delay():
    """Test that causal convolution has zero algorithmic delay."""
    conv = CausalConv1d(1, 1, kernel_size=5)
    assert conv.output_delay == 0


def test_gradient_flow():
    """Test gradient flow through causal convolution."""
    conv = CausalConv1d(2, 4, kernel_size=3)
    
    x = torch.randn(2, 2, 50, requires_grad=True)
    y = conv(x)
    loss = y.mean()
    loss.backward()
    
    # Check gradients exist and are finite
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))
    assert conv.conv.weight.grad is not None
    assert torch.all(torch.isfinite(conv.conv.weight.grad))


def test_return_state():
    """Test state return for streaming applications."""
    conv = CausalConv1d(1, 1, kernel_size=5, dilation=2)
    
    x = torch.randn(1, 1, 100)
    y, state = conv(x, return_state=True)
    
    assert y.shape == (1, 1, 100)
    assert state is not None
    
    # State should contain last receptive_field - 1 samples
    expected_state_len = conv.receptive_field - 1
    assert state.shape == (1, 1, expected_state_len)


def test_edge_cases():
    """Test edge cases."""
    # Very short sequence
    conv = CausalConv1d(1, 1, kernel_size=5)
    x_short = torch.randn(1, 1, 2)
    y_short = conv(x_short)
    assert y_short.shape == (1, 1, 2)
    
    # Single sample
    x_single = torch.randn(1, 1, 1)
    y_single = conv(x_single)
    assert y_single.shape == (1, 1, 1)
    
    # Kernel size = 1 (no padding needed)
    conv_k1 = CausalConv1d(1, 1, kernel_size=1)
    assert conv_k1.causal_padding == 0
    x = torch.randn(1, 1, 10)
    y = conv_k1(x)
    assert y.shape == x.shape


def test_audio_processing_scenario():
    """Test in realistic audio processing scenario."""
    # Typical audio processing setup
    sample_rate = 16000
    chunk_size = 256  # ~16ms at 16kHz
    channels = 1
    
    # Multi-layer causal convolution (like in WaveNet)
    conv1 = CausalConv1d(channels, 16, kernel_size=3, dilation=1)
    conv2 = CausalConv1d(16, 16, kernel_size=3, dilation=2)
    conv3 = CausalConv1d(16, 16, kernel_size=3, dilation=4)
    conv4 = CausalConv1d(16, channels, kernel_size=1)  # 1x1 conv
    
    # Process audio chunk
    audio_chunk = torch.randn(1, channels, chunk_size)
    
    # Forward pass through stack
    h = torch.relu(conv1(audio_chunk))
    h = torch.relu(conv2(h))
    h = torch.relu(conv3(h))
    output = conv4(h)
    
    # Output should maintain chunk size
    assert output.shape == audio_chunk.shape


if __name__ == "__main__":
    test_basic_causal_conv()
    test_causality()
    test_different_kernel_sizes()
    test_stride()
    test_dilation()
    test_groups()
    test_padding_modes()
    test_streaming_mode()
    test_receptive_field()
    test_causal_transpose()
    test_zero_delay()
    test_gradient_flow()
    test_return_state()
    test_edge_cases()
    test_audio_processing_scenario()
    print("All CausalConv tests passed!")