import torch
import pytest
import numpy as np
from modules.stft_loss import (
    SpectralConvergenceLoss, 
    LogSTFTMagnitudeLoss,
    STFTLoss,
    MultiScaleSTFTLoss,
    MelSpectrogramLoss
)


def test_spectral_convergence_loss():
    """Test spectral convergence loss calculation."""
    sc_loss = SpectralConvergenceLoss()
    
    # Test with identical spectrograms
    x_mag = torch.randn(2, 513, 100).abs()
    y_mag = x_mag.clone()
    loss = sc_loss(x_mag, y_mag)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)
    
    # Test with different spectrograms
    y_mag_diff = torch.randn_like(x_mag).abs()
    loss_diff = sc_loss(x_mag, y_mag_diff)
    assert loss_diff > 0
    assert loss_diff <= 2.0  # Theoretical max is 2 for completely different


def test_log_magnitude_loss():
    """Test log STFT magnitude loss."""
    log_loss = LogSTFTMagnitudeLoss()
    
    # Test with identical spectrograms
    x_mag = torch.randn(2, 513, 100).abs() + 0.1
    y_mag = x_mag.clone()
    loss = log_loss(x_mag, y_mag)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)
    
    # Test with different spectrograms
    y_mag_diff = torch.randn_like(x_mag).abs() + 0.1
    loss_diff = log_loss(x_mag, y_mag_diff)
    assert loss_diff > 0


def test_single_stft_loss():
    """Test single-scale STFT loss."""
    stft_loss = STFTLoss(fft_size=512, hop_size=128)
    
    # Create test signals
    batch_size = 2
    length = 16000
    
    # Test with identical signals
    x = torch.randn(batch_size, length)
    y = x.clone()
    loss = stft_loss(x, y)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)
    
    # Test with different signals
    y_diff = torch.randn_like(x)
    loss_diff = stft_loss(x, y_diff)
    assert loss_diff > 0
    
    # Test return_stfts option
    loss, x_stft, y_stft = stft_loss(x, y, return_stfts=True)
    assert x_stft.shape[0] == batch_size
    assert x_stft.shape[1] == 512 // 2 + 1  # Frequency bins
    assert y_stft.shape == x_stft.shape


def test_multi_channel_stft_loss():
    """Test STFT loss with multi-channel audio."""
    stft_loss = STFTLoss()
    
    # Multi-channel input
    x = torch.randn(2, 2, 8000)  # [batch, channels, time]
    y = torch.randn_like(x)
    
    loss = stft_loss(x, y)
    assert loss.ndim == 0  # Scalar loss
    assert loss > 0


def test_multi_scale_stft_loss():
    """Test multi-scale STFT loss."""
    # Custom scales
    scales = [
        (256, 64, 256),
        (512, 128, 512),
        (1024, 256, 1024),
    ]
    
    ms_loss = MultiScaleSTFTLoss(scales=scales)
    
    # Test signals
    x = torch.randn(2, 16000)
    y = x.clone()
    
    # Identical signals should have near-zero loss
    loss = ms_loss(x, y)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)
    
    # Different signals
    y_diff = torch.randn_like(x)
    loss_diff = ms_loss(x, y_diff)
    assert loss_diff > 0
    
    # Test per-scale losses
    total_loss, scale_losses = ms_loss(x, y_diff, return_losses_per_scale=True)
    assert len(scale_losses) == len(scales)
    assert all(l > 0 for l in scale_losses)
    
    # Verify weighted average
    avg_loss = sum(scale_losses) / len(scale_losses)
    assert torch.allclose(total_loss, avg_loss, atol=1e-6)


def test_loss_weights():
    """Test multi-scale STFT loss with custom weights."""
    scales = [(512, 128, 512), (1024, 256, 1024)]
    weights = [0.5, 2.0]
    
    ms_loss = MultiScaleSTFTLoss(scales=scales, loss_weights=weights)
    
    x = torch.randn(1, 8000)
    y = torch.randn_like(x)
    
    total_loss, scale_losses = ms_loss(x, y, return_losses_per_scale=True)
    
    # Manual calculation
    expected = (scale_losses[0] + scale_losses[1]) / 2
    assert torch.allclose(total_loss, expected, atol=1e-6)


def test_different_window_types():
    """Test STFT loss with different window types."""
    windows = ['hann', 'hamming', 'blackman', 'bartlett']
    
    x = torch.randn(1, 4096)
    y = torch.randn_like(x)
    
    losses = []
    for window in windows:
        stft_loss = STFTLoss(window=window)
        loss = stft_loss(x, y)
        losses.append(loss)
        assert loss > 0
        assert torch.isfinite(loss)
    
    # Different windows should give slightly different results
    assert not all(torch.allclose(losses[0], l) for l in losses[1:])


def test_mel_spectrogram_loss():
    """Test mel-spectrogram loss."""
    mel_loss = MelSpectrogramLoss(
        sample_rate=16000,
        n_fft=1024,
        hop_length=256,
        n_mels=80
    )
    
    # Test signals
    x = torch.randn(2, 16000)
    y = x.clone()
    
    # Identical signals
    loss = mel_loss(x, y)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)
    
    # Different signals
    y_diff = torch.randn_like(x)
    loss_diff = mel_loss(x, y_diff)
    assert loss_diff > 0


def test_gradient_flow():
    """Test gradient flow through losses."""
    stft_loss = STFTLoss()
    
    x = torch.randn(2, 8000, requires_grad=True)
    y = torch.randn_like(x)
    
    loss = stft_loss(x, y)
    loss.backward()
    
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))
    assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


def test_numerical_stability():
    """Test numerical stability with edge cases."""
    stft_loss = STFTLoss()
    
    # Very small signals
    x_small = torch.randn(1, 1024) * 1e-6
    y_small = torch.randn_like(x_small) * 1e-6
    loss_small = stft_loss(x_small, y_small)
    assert torch.isfinite(loss_small)
    
    # Silent signal vs non-silent
    x_zero = torch.zeros(1, 1024)
    y_nonzero = torch.randn(1, 1024)
    loss_zero = stft_loss(x_zero, y_nonzero)
    assert torch.isfinite(loss_zero)
    assert loss_zero > 0


def test_different_lengths():
    """Test STFT loss with different signal lengths."""
    stft_loss = STFTLoss(fft_size=256, hop_size=64)
    
    for length in [256, 512, 1024, 2048, 3000]:
        x = torch.randn(1, length)
        y = torch.randn_like(x)
        loss = stft_loss(x, y)
        assert torch.isfinite(loss)
        assert loss > 0


def test_normalized_stft():
    """Test STFT loss with normalization."""
    stft_loss_norm = STFTLoss(normalized=True)
    stft_loss_no_norm = STFTLoss(normalized=False)
    
    x = torch.randn(1, 4096)
    y = torch.randn_like(x)
    
    loss_norm = stft_loss_norm(x, y)
    loss_no_norm = stft_loss_no_norm(x, y)
    
    # Both should be valid
    assert torch.isfinite(loss_norm)
    assert torch.isfinite(loss_no_norm)
    # Normalization affects the scale but both are valid losses


def test_real_audio_scenario():
    """Test with realistic audio parameters."""
    # Typical setup for 24kHz audio
    ms_loss = MultiScaleSTFTLoss(
        scales=[
            (512, 120, 480),    # ~20ms window, ~5ms hop
            (1024, 240, 960),   # ~40ms window, ~10ms hop  
            (2048, 480, 1920),  # ~80ms window, ~20ms hop
        ]
    )
    
    # Simulate batch of audio
    batch_size = 4
    duration = 1.0  # seconds
    sample_rate = 24000
    samples = int(duration * sample_rate)
    
    x = torch.randn(batch_size, samples) * 0.5  # Reasonable amplitude
    y = x + torch.randn_like(x) * 0.01  # Small noise
    
    loss = ms_loss(x, y)
    assert loss > 0
    assert loss < 1.0  # Should be small for similar signals


if __name__ == "__main__":
    test_spectral_convergence_loss()
    test_log_magnitude_loss()
    test_single_stft_loss()
    test_multi_channel_stft_loss()
    test_multi_scale_stft_loss()
    test_loss_weights()
    test_different_window_types()
    test_mel_spectrogram_loss()
    test_gradient_flow()
    test_numerical_stability()
    test_different_lengths()
    test_normalized_stft()
    test_real_audio_scenario()
    print("All STFT loss tests passed!")