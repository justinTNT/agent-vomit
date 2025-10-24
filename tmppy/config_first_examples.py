
from rave_config_system import RAVEConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        # Over-clockable: can use extreme values from config
        self.epsilon = 1e-8
        
        # Register window buffer
        window_fn = getattr(torch, f'{self.window}_window')
        window = window_fn(self.win_length)
        self.register_buffer('window', window)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute STFT loss with configurable parameters"""
        
        # Compute STFT with config parameters
        pred_stft = torch.stft(
            pred.squeeze(1), 
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            normalized=self.normalized,
            return_complex=True
        )
        
        target_stft = torch.stft(
            target.squeeze(1),
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
        convergence_loss = torch.norm(pred_mag - target_mag, p='fro') / torch.norm(target_mag, p='fro')
        
        return magnitude_loss + convergence_loss

class ConfigFirstConvEncoder(nn.Module):
    """Convolution Encoder with config-first, over-clockable interface"""
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__()
        
        # All parameters from config - enables extreme architectures
        self.in_channels = config.convolution.in_channels
        self.base_channels = config.model.base_channels
        self.latent_dim = config.model.latent_dim
        self.num_layers = config.model.n_layers
        self.kernel_size = config.convolution.kernel_size
        self.stride = config.convolution.stride
        self.activation = config.model.activation
        self.use_spectral_norm = config.convolution.use_spectral_norm
        
        # Build over-clockable encoder
        layers = []
        in_ch = self.in_channels
        
        for i in range(self.num_layers):
            out_ch = min(self.base_channels * (2 ** i), config.model.max_channels)
            
            conv = nn.Conv1d(
                in_ch, out_ch, 
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.kernel_size // 2,
                bias=config.convolution.bias
            )
            
            if self.use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            
            layers.extend([
                conv,
                nn.BatchNorm1d(out_ch),
                self._get_activation()
            ])
            
            in_ch = out_ch
        
        # Final projection to latent space
        layers.append(nn.AdaptiveAvgPool1d(1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_ch, self.latent_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def _get_activation(self):
        """Get activation function from config"""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(), 
            'swish': nn.SiLU(),
            'mish': nn.Mish()
        }
        return activations.get(self.activation, nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space"""
        return self.encoder(x)

class ConfigFirstResidualVectorQuantizer(nn.Module):
    """RVQ with config-first, over-clockable interface"""
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__()
        
        # All quantization parameters from config
        self.num_quantizers = config.quantization.num_quantizers
        self.codebook_size = config.quantization.codebook_size
        self.codebook_dim = config.quantization.codebook_dim
        self.commitment_cost = config.quantization.commitment_cost
        self.ema_decay = config.quantization.ema_decay
        
        # Create quantizer layers - over-clockable
        self.quantizers = nn.ModuleList([
            VectorQuantizer(
                self.codebook_size,
                self.codebook_dim,
                self.commitment_cost,
                self.ema_decay
            ) for _ in range(self.num_quantizers)
        ])
    
    def forward(self, x: torch.Tensor) -> dict:
        """Residual vector quantization"""
        residual = x
        quantized = torch.zeros_like(x)
        indices = []
        total_loss = 0.0
        
        for quantizer in self.quantizers:
            quant_out = quantizer(residual)
            quantized += quant_out['quantized']
            residual = residual - quant_out['quantized']
            indices.append(quant_out['indices'])
            total_loss += quant_out['loss']
        
        return {
            'quantized': quantized,
            'indices': torch.stack(indices, dim=1),
            'loss': total_loss,
            'residual': residual
        }
