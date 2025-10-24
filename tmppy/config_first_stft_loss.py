
import torch
import torch.nn as nn
import torch.nn.functional as F
from rave_config_system import RAVEConfig

class ConfigFirstSTFTLoss(nn.Module):
    """STFT Loss with config-first, over-clockable interface"""
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__()
        
        # All parameters come from config - enables over-clocking
        self.n_fft = config.audio.n_fft
        self.hop_length = config.audio.hop_length  
        self.win_length = config.audio.win_length
        self.window = config.audio.window
        self.normalized = config.audio.normalized
        self.reduction = config.loss.reduction
        
        # Register window buffer
        window_fn = getattr(torch, f'{self.window}_window')
        window = window_fn(self.win_length)
        self.register_buffer('window', window)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute STFT loss with configurable parameters"""
        
        # Ensure correct input shape (B, C, T) -> (B*C, T)
        if pred.dim() == 3:
            pred = pred.reshape(-1, pred.size(-1))
        if target.dim() == 3:
            target = target.reshape(-1, target.size(-1))
        
        # Compute STFT with config parameters
        pred_stft = torch.stft(
            pred, 
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            normalized=self.normalized,
            return_complex=True
        )
        
        target_stft = torch.stft(
            target,
            n_fft=self.n_fft, 
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            normalized=self.normalized,
            return_complex=True
        )
        
        # Magnitude loss
        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)
        
        magnitude_loss = F.l1_loss(pred_mag, target_mag, reduction=self.reduction)
        
        # Spectral convergence loss  
        convergence_loss = torch.norm(pred_mag - target_mag, p='fro') / (torch.norm(target_mag, p='fro') + 1e-8)
        
        return magnitude_loss + convergence_loss

class ConfigFirstMelSpectrogramLoss(nn.Module):
    """Mel Spectrogram Loss with config-first interface"""
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__()
        
        self.sample_rate = config.audio.sample_rate
        self.n_fft = config.audio.n_fft
        self.hop_length = config.audio.hop_length
        self.n_mels = config.audio.n_mels
        self.f_min = config.audio.f_min
        self.f_max = config.audio.f_max
        self.normalized = config.audio.normalized
        self.reduction = config.loss.reduction
        
        # Create mel transform
        self.mel_transform = nn.modules.utils.spectral_ops.MelScale(
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            f_min=self.f_min,
            f_max=self.f_max,
            n_stft=self.n_fft // 2 + 1
        )
        
        # Register window
        window_fn = getattr(torch, f'{config.audio.window}_window')
        window = window_fn(config.audio.win_length)
        self.register_buffer('window', window)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram loss"""
        
        # Ensure correct input shape
        if pred.dim() == 3:
            pred = pred.reshape(-1, pred.size(-1))
        if target.dim() == 3:
            target = target.reshape(-1, target.size(-1))
        
        # Compute spectrograms
        pred_spec = torch.stft(
            pred, n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, normalized=self.normalized, return_complex=True
        )
        target_spec = torch.stft(
            target, n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, normalized=self.normalized, return_complex=True
        )
        
        # Convert to mel scale
        pred_mag = torch.abs(pred_spec)
        target_mag = torch.abs(target_spec)
        
        pred_mel = self.mel_transform(pred_mag)
        target_mel = self.mel_transform(target_mag)
        
        return F.l1_loss(pred_mel, target_mel, reduction=self.reduction)
