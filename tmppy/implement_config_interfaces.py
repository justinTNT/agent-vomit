#!/usr/bin/env python3
"""
IMPLEMENT CONFIG INTERFACES: Actually implement the config-first interfaces
Strategy: Replace key modules with config-driven implementations for testing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from rave_config_system import RAVEConfig, get_standard_config
from pathlib import Path

# Create config-first implementations
def create_config_first_modules():
    """Create actual config-first module implementations"""
    
    # 1. Config-First STFT Loss
    stft_loss_code = '''
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
'''
    
    # 2. Config-First Conv Encoder
    conv_encoder_code = '''
import torch
import torch.nn as nn
from rave_config_system import RAVEConfig

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
        self.norm_type = config.convolution.norm_type
        
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
            
            layers.append(conv)
            
            # Add normalization
            if self.norm_type == 'batch_norm':
                layers.append(nn.BatchNorm1d(out_ch))
            elif self.norm_type == 'layer_norm':
                layers.append(nn.LayerNorm(out_ch))
            elif self.norm_type == 'group_norm':
                layers.append(nn.GroupNorm(8, out_ch))
            
            # Add activation
            layers.append(self._get_activation())
            
            in_ch = out_ch
        
        # Final projection to latent space
        layers.extend([
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_ch, self.latent_dim)
        ])
        
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
    
    def forward(self, x: torch.Tensor) -> dict:
        """Encode input to latent space"""
        # Ensure correct input shape for 1D conv
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        
        latent = self.encoder(x)
        
        return {
            'latent': latent,
            'shape_info': {
                'input': tuple(x.shape),
                'latent': tuple(latent.shape)
            }
        }
'''
    
    # 3. Config-First AutoEncoder
    autoencoder_code = '''
import torch
import torch.nn as nn
import torch.nn.functional as F
from rave_config_system import RAVEConfig

class ConfigFirstConvDecoder(nn.Module):
    """Config-first convolution decoder"""
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__()
        
        self.latent_dim = config.model.latent_dim
        self.base_channels = config.model.base_channels
        self.num_layers = config.model.n_layers
        self.out_channels = config.convolution.out_channels
        self.kernel_size = config.convolution.kernel_size
        self.stride = config.convolution.stride
        
        # Initial projection to feature maps
        initial_channels = self.base_channels * (2 ** (self.num_layers - 1))
        initial_length = 64  # Start with reasonable length
        
        self.initial_proj = nn.Linear(self.latent_dim, initial_channels * initial_length)
        self.initial_channels = initial_channels
        self.initial_length = initial_length
        
        # Build decoder layers
        layers = []
        in_ch = initial_channels
        
        for i in range(self.num_layers):
            out_ch = self.base_channels * (2 ** max(self.num_layers - i - 2, 0))
            
            if i == self.num_layers - 1:
                # Final layer
                layers.extend([
                    nn.ConvTranspose1d(in_ch, self.out_channels, 
                                     kernel_size=self.kernel_size,
                                     stride=self.stride, 
                                     padding=self.kernel_size // 2),
                    nn.Tanh()
                ])
            else:
                layers.extend([
                    nn.ConvTranspose1d(in_ch, out_ch,
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     padding=self.kernel_size // 2),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU()
                ])
            
            in_ch = out_ch
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to output"""
        # Project and reshape
        x = self.initial_proj(z)
        x = x.view(x.size(0), self.initial_channels, self.initial_length)
        
        # Decode
        x = self.decoder(x)
        
        return x

class ConfigFirstAutoEncoder(nn.Module):
    """AutoEncoder with config-first interface"""
    
    def __init__(self, config: RAVEConfig, encoder=None, decoder=None, **kwargs):
        super().__init__()
        
        self.latent_dim = config.model.latent_dim
        
        # Use provided encoder/decoder or create from config
        if encoder is None:
            from config_first_conv_encoder import ConfigFirstConvEncoder
            self.encoder = ConfigFirstConvEncoder(config)
        else:
            self.encoder = encoder
        
        if decoder is None:
            self.decoder = ConfigFirstConvDecoder(config)
        else:
            self.decoder = decoder
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input"""
        output = self.encoder(x)
        return output['latent'] if isinstance(output, dict) else output
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass"""
        z = self.encode(x)
        x_recon = self.decode(z)
        
        return {
            'reconstruction': x_recon,
            'latent': z
        }
'''
    
    # 4. Config-First Vector Quantizer
    quantizer_code = '''
import torch
import torch.nn as nn
import torch.nn.functional as F
from rave_config_system import RAVEConfig

class ConfigFirstVectorQuantizer(nn.Module):
    """Vector Quantizer with config-first interface"""
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__()
        
        self.num_embeddings = config.quantization.codebook_size
        self.embedding_dim = config.quantization.codebook_dim
        self.commitment_cost = config.quantization.commitment_cost
        self.epsilon = config.quantization.epsilon
        
        # Initialize codebook
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
    
    def forward(self, inputs: torch.Tensor) -> dict:
        """Vector quantization forward pass"""
        
        # Flatten input for quantization
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # Get indices of closest embedding vectors
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        # One-hot encoding
        device = inputs.device
        encoding_one_hot = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=device)
        encoding_one_hot.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = torch.matmul(encoding_one_hot, self.embedding.weight)
        quantized = quantized.view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Perplexity
        avg_probs = torch.mean(encoding_one_hot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.epsilon)))
        
        return {
            'quantized': quantized,
            'loss': loss,
            'indices': encoding_indices.view(input_shape[:-1]),
            'perplexity': perplexity
        }

class ConfigFirstResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer with config-first interface"""
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__()
        
        self.num_quantizers = config.quantization.num_quantizers
        self.codebook_size = config.quantization.codebook_size
        self.codebook_dim = config.quantization.codebook_dim
        
        # Create quantizer layers
        self.quantizers = nn.ModuleList([
            ConfigFirstVectorQuantizer(config)
            for _ in range(self.num_quantizers)
        ])
    
    def forward(self, x: torch.Tensor) -> dict:
        """Residual vector quantization"""
        residual = x
        quantized = torch.zeros_like(x)
        indices = []
        total_loss = 0.0
        total_perplexity = 0.0
        
        for quantizer in self.quantizers:
            quant_out = quantizer(residual)
            quantized = quantized + quant_out['quantized']
            residual = residual - quant_out['quantized']
            indices.append(quant_out['indices'])
            total_loss = total_loss + quant_out['loss']
            total_perplexity = total_perplexity + quant_out['perplexity']
        
        return {
            'quantized': quantized,
            'indices': torch.stack(indices, dim=1),
            'loss': total_loss,
            'perplexity': total_perplexity / self.num_quantizers,
            'residual': residual
        }
'''
    
    return {
        'stft_loss': stft_loss_code,
        'conv_encoder': conv_encoder_code,
        'autoencoder': autoencoder_code,
        'quantizer': quantizer_code
    }

def write_config_first_modules():
    """Write config-first modules to files"""
    
    modules = create_config_first_modules()
    
    files_created = []
    
    # Write STFT Loss
    with open('config_first_stft_loss.py', 'w') as f:
        f.write(modules['stft_loss'])
    files_created.append('config_first_stft_loss.py')
    
    # Write Conv Encoder
    with open('config_first_conv_encoder.py', 'w') as f:
        f.write(modules['conv_encoder'])
    files_created.append('config_first_conv_encoder.py')
    
    # Write AutoEncoder
    with open('config_first_autoencoder.py', 'w') as f:
        f.write(modules['autoencoder'])
    files_created.append('config_first_autoencoder.py')
    
    # Write Quantizer
    with open('config_first_quantizer.py', 'w') as f:
        f.write(modules['quantizer'])
    files_created.append('config_first_quantizer.py')
    
    return files_created

def test_config_first_modules():
    """Test all config-first modules with different configurations"""
    
    print("🧪 TESTING CONFIG-FIRST MODULES")
    print("=" * 60)
    
    configs = {
        'minimal': get_standard_config(),
        'extreme': get_standard_config().get_overclocked_config()
    }
    
    # Update minimal config to be actually minimal
    configs['minimal'].model.d_model = 128
    configs['minimal'].model.latent_dim = 64
    configs['minimal'].model.n_heads = 4
    configs['minimal'].model.n_layers = 3
    configs['minimal'].audio.n_fft = 512
    
    test_results = {}
    
    for config_name, config in configs.items():
        print(f"\n🎯 Testing {config_name.upper()} config:")
        print(f"   d_model: {config.model.d_model}")
        print(f"   latent_dim: {config.model.latent_dim}")
        print(f"   n_fft: {config.audio.n_fft}")
        print(f"   quantizers: {config.quantization.num_quantizers}")
        
        results = {}
        
        # Test STFT Loss
        try:
            exec(open('config_first_stft_loss.py').read())
            stft_loss = eval('ConfigFirstSTFTLoss(config)')
            
            # Test with audio input
            audio_input = torch.randn(2, 1, 8192)
            audio_target = torch.randn(2, 1, 8192)
            loss = stft_loss(audio_input, audio_target)
            
            results['stft_loss'] = {
                'status': 'success',
                'loss_value': float(loss),
                'input_shape': list(audio_input.shape)
            }
            print(f"   ✅ STFT Loss: {float(loss):.4f}")
            
        except Exception as e:
            results['stft_loss'] = {'status': 'failed', 'error': str(e)}
            print(f"   ❌ STFT Loss: {str(e)}")
        
        # Test Conv Encoder
        try:
            exec(open('config_first_conv_encoder.py').read())
            encoder = eval('ConfigFirstConvEncoder(config)')
            
            # Test with audio input
            audio_input = torch.randn(2, 1, 8192)
            output = encoder(audio_input)
            
            results['conv_encoder'] = {
                'status': 'success',
                'output_shape': list(output['latent'].shape),
                'input_shape': list(audio_input.shape)
            }
            print(f"   ✅ Conv Encoder: {output['latent'].shape}")
            
        except Exception as e:
            results['conv_encoder'] = {'status': 'failed', 'error': str(e)}
            print(f"   ❌ Conv Encoder: {str(e)}")
        
        # Test AutoEncoder
        try:
            exec(open('config_first_autoencoder.py').read())
            autoencoder = eval('ConfigFirstAutoEncoder(config)')
            
            # Test with audio input
            audio_input = torch.randn(2, 1, 8192)
            output = autoencoder(audio_input)
            
            results['autoencoder'] = {
                'status': 'success',
                'reconstruction_shape': list(output['reconstruction'].shape),
                'latent_shape': list(output['latent'].shape),
                'input_shape': list(audio_input.shape)
            }
            print(f"   ✅ AutoEncoder: recon {output['reconstruction'].shape}, latent {output['latent'].shape}")
            
        except Exception as e:
            results['autoencoder'] = {'status': 'failed', 'error': str(e)}
            print(f"   ❌ AutoEncoder: {str(e)}")
        
        # Test Vector Quantizer
        try:
            exec(open('config_first_quantizer.py').read())
            quantizer = eval('ConfigFirstResidualVectorQuantizer(config)')
            
            # Test with latent input
            latent_input = torch.randn(2, 64, config.quantization.codebook_dim)
            output = quantizer(latent_input)
            
            results['quantizer'] = {
                'status': 'success',
                'quantized_shape': list(output['quantized'].shape),
                'indices_shape': list(output['indices'].shape),
                'loss_value': float(output['loss']),
                'perplexity': float(output['perplexity'])
            }
            print(f"   ✅ RVQ: loss {float(output['loss']):.4f}, perplexity {float(output['perplexity']):.2f}")
            
        except Exception as e:
            results['quantizer'] = {'status': 'failed', 'error': str(e)}
            print(f"   ❌ RVQ: {str(e)}")
        
        test_results[config_name] = results
    
    return test_results

def main():
    """Implement and test config-first interfaces"""
    
    print("🔧 IMPLEMENTING CONFIG-FIRST INTERFACES")
    print("=" * 70)
    print("Strategy: Create production-ready, over-clockable module implementations")
    print()
    
    # Create config-first modules
    print("📝 Creating config-first module files...")
    files_created = write_config_first_modules()
    
    for file_name in files_created:
        print(f"   ✅ Created: {file_name}")
    
    print()
    
    # Test the modules
    test_results = test_config_first_modules()
    
    print("\n" + "=" * 70)
    print("🎯 CONFIG-FIRST IMPLEMENTATION RESULTS")
    print("=" * 70)
    
    # Calculate success rates
    for config_name, results in test_results.items():
        total_modules = len(results)
        successful_modules = len([r for r in results.values() if r['status'] == 'success'])
        success_rate = (successful_modules / total_modules * 100) if total_modules > 0 else 0
        
        print(f"\n{config_name.upper()} CONFIG:")
        print(f"   Success rate: {successful_modules}/{total_modules} ({success_rate:.1f}%)")
        
        for module_name, result in results.items():
            if result['status'] == 'success':
                print(f"   ✅ {module_name}")
            else:
                print(f"   ❌ {module_name}: {result.get('error', 'Unknown error')}")
    
    # Overall assessment
    all_results = [r for results in test_results.values() for r in results.values()]
    total_tests = len(all_results)
    successful_tests = len([r for r in all_results if r['status'] == 'success'])
    overall_success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n📊 OVERALL CONFIG-FIRST SUCCESS RATE:")
    print(f"   {successful_tests}/{total_tests} ({overall_success_rate:.1f}%)")
    
    if overall_success_rate >= 90:
        print("🎉 EXCELLENT: 90%+ config-first success rate!")
        print("✅ Ready for over-clockable RAVE experimentation")
    elif overall_success_rate >= 75:
        print("🔥 GOOD: 75%+ config-first success rate!")
        print("🔧 Minor fixes for 90%+ target")
    else:
        print("⚠️  More work needed for config-first interfaces")
    
    print(f"\n🚀 NEXT STEPS:")
    print("1. Replace legacy modules with config-first versions")
    print("2. Run comprehensive validation with over-clocked configs")
    print("3. Measure improvement in overall test success rate")
    print("4. Begin extreme RAVE experimentation")
    
    return test_results

if __name__ == "__main__":
    main()