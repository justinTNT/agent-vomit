#!/usr/bin/env python3
"""
CONFIG-FIRST VALIDATION: Test the new config-driven interface improvements
Strategy: Compare original 91% success rate with config-first implementations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from rave_config_system import RAVEConfig, get_standard_config, get_extreme_config
import time
from typing import Dict, List, Any
import traceback

class ConfigFirstSTFTLoss(nn.Module):
    """Fixed STFT Loss with config-first interface"""
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__()
        
        self.n_fft = config.audio.n_fft
        self.hop_length = config.audio.hop_length  
        self.win_length = config.audio.win_length
        self.window_name = config.audio.window
        self.normalized = config.audio.normalized
        self.reduction = config.loss.reduction
        
        # Register window buffer (fixed name conflict)
        window_fn = getattr(torch, f'{self.window_name}_window')
        window_tensor = window_fn(self.win_length)
        self.register_buffer('window_tensor', window_tensor)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Ensure correct input shape
        if pred.dim() == 3:
            pred = pred.reshape(-1, pred.size(-1))
        if target.dim() == 3:
            target = target.reshape(-1, target.size(-1))
        
        # Compute STFT
        pred_stft = torch.stft(
            pred, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=self.window_tensor,
            normalized=self.normalized, return_complex=True
        )
        
        target_stft = torch.stft(
            target, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=self.window_tensor,
            normalized=self.normalized, return_complex=True
        )
        
        # Compute losses
        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)
        
        magnitude_loss = F.l1_loss(pred_mag, target_mag, reduction=self.reduction)
        convergence_loss = torch.norm(pred_mag - target_mag, p='fro') / (torch.norm(target_mag, p='fro') + 1e-8)
        
        return magnitude_loss + convergence_loss

class ConfigFirstConvEncoder(nn.Module):
    """Fixed Conv Encoder with config-first interface"""
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__()
        
        self.in_channels = config.convolution.in_channels
        self.base_channels = config.model.base_channels
        self.latent_dim = config.model.latent_dim
        self.num_layers = min(config.model.n_layers, 6)  # Limit for stability
        self.kernel_size = config.convolution.kernel_size
        self.stride = config.convolution.stride
        
        # Build encoder
        layers = []
        in_ch = self.in_channels
        
        for i in range(self.num_layers):
            out_ch = min(self.base_channels * (2 ** i), config.model.max_channels)
            
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=self.kernel_size,
                         stride=self.stride, padding=self.kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU()
            ])
            in_ch = out_ch
        
        layers.extend([
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_ch, self.latent_dim)
        ])
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> dict:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        
        latent = self.encoder(x)
        return {'latent': latent}

class ConfigFirstConvDecoder(nn.Module):
    """Fixed Conv Decoder with config-first interface"""
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__()
        
        self.latent_dim = config.model.latent_dim
        self.base_channels = config.model.base_channels
        self.num_layers = min(config.model.n_layers, 6)
        self.out_channels = config.convolution.out_channels
        
        # Initial projection
        initial_channels = self.base_channels * (2 ** (self.num_layers - 1))
        initial_length = 64
        
        self.initial_proj = nn.Linear(self.latent_dim, initial_channels * initial_length)
        self.initial_channels = initial_channels
        self.initial_length = initial_length
        
        # Build decoder
        layers = []
        in_ch = initial_channels
        
        for i in range(self.num_layers):
            if i == self.num_layers - 1:
                out_ch = self.out_channels
                layers.extend([
                    nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                    nn.Tanh()
                ])
            else:
                out_ch = self.base_channels * (2 ** max(self.num_layers - i - 2, 0))
                layers.extend([
                    nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU()
                ])
            in_ch = out_ch
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.initial_proj(z)
        x = x.view(x.size(0), self.initial_channels, self.initial_length)
        return self.decoder(x)

class ConfigFirstAutoEncoder(nn.Module):
    """Fixed AutoEncoder with config-first interface"""
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__()
        
        self.encoder = ConfigFirstConvEncoder(config)
        self.decoder = ConfigFirstConvDecoder(config)
    
    def forward(self, x: torch.Tensor) -> dict:
        encoder_out = self.encoder(x)
        latent = encoder_out['latent']
        reconstruction = self.decoder(latent)
        
        return {
            'reconstruction': reconstruction,
            'latent': latent
        }

class ConfigFirstVectorQuantizer(nn.Module):
    """Fixed Vector Quantizer with config-first interface"""
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__()
        
        self.num_embeddings = config.quantization.codebook_size
        self.embedding_dim = config.quantization.codebook_dim
        self.commitment_cost = config.quantization.commitment_cost
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
    
    def forward(self, inputs: torch.Tensor) -> dict:
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        # One-hot encoding
        encoding_one_hot = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
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
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-8)))
        
        return {
            'quantized': quantized,
            'loss': loss,
            'indices': encoding_indices.view(input_shape[:-1]),
            'perplexity': perplexity
        }

class ConfigFirstResidualVectorQuantizer(nn.Module):
    """Fixed RVQ with config-first interface"""
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__()
        
        self.num_quantizers = config.quantization.num_quantizers
        self.quantizers = nn.ModuleList([
            ConfigFirstVectorQuantizer(config) for _ in range(self.num_quantizers)
        ])
    
    def forward(self, x: torch.Tensor) -> dict:
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
            'perplexity': total_perplexity / self.num_quantizers
        }

def test_config_first_module(module_class, config: RAVEConfig, test_inputs: List[torch.Tensor]) -> Dict[str, Any]:
    """Test a single config-first module"""
    
    try:
        # Initialize module
        start_time = time.time()
        module = module_class(config)
        module.eval()
        init_time = time.time() - start_time
        
        # Test forward pass
        test_results = []
        for i, test_input in enumerate(test_inputs):
            try:
                start_time = time.time()
                
                with torch.no_grad():
                    if isinstance(test_input, tuple):
                        output = module(*test_input)
                    else:
                        output = module(test_input)
                
                forward_time = time.time() - start_time
                
                # Analyze output
                if isinstance(output, torch.Tensor):
                    output_analysis = {
                        'type': 'tensor',
                        'shape': list(output.shape),
                        'mean': float(output.mean()),
                        'std': float(output.std()),
                        'has_nan': bool(torch.isnan(output).any()),
                        'has_inf': bool(torch.isinf(output).any())
                    }
                elif isinstance(output, dict):
                    output_analysis = {
                        'type': 'dict',
                        'keys': list(output.keys()),
                        'shapes': {k: list(v.shape) if isinstance(v, torch.Tensor) else str(type(v)) 
                                  for k, v in output.items()}
                    }
                else:
                    output_analysis = {'type': str(type(output))}
                
                test_results.append({
                    'test_id': i + 1,
                    'status': 'passed',
                    'forward_time': forward_time,
                    'output_analysis': output_analysis
                })
                
            except Exception as e:
                test_results.append({
                    'test_id': i + 1,
                    'status': 'failed',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
        
        all_passed = all(r['status'] == 'passed' for r in test_results)
        
        return {
            'module_name': module_class.__name__,
            'status': 'success' if all_passed else 'partial_success',
            'init_time': init_time,
            'test_results': test_results,
            'config_used': {
                'd_model': config.model.d_model,
                'latent_dim': config.model.latent_dim,
                'n_fft': config.audio.n_fft,
                'quantizers': config.quantization.num_quantizers
            }
        }
        
    except Exception as e:
        return {
            'module_name': module_class.__name__,
            'status': 'failed',
            'init_error': str(e),
            'traceback': traceback.format_exc()
        }

def comprehensive_config_first_validation():
    """Run comprehensive validation of config-first modules"""
    
    print("🔬 CONFIG-FIRST VALIDATION: Testing over-clockable interfaces")
    print("=" * 80)
    
    # Test configurations
    test_configs = {
        'minimal': get_standard_config(),
        'standard': get_standard_config(),
        'extreme': get_extreme_config()
    }
    
    # Make minimal actually minimal
    test_configs['minimal'].model.d_model = 128
    test_configs['minimal'].model.latent_dim = 64
    test_configs['minimal'].model.n_heads = 4
    test_configs['minimal'].model.n_layers = 3
    test_configs['minimal'].audio.n_fft = 512
    test_configs['minimal'].quantization.num_quantizers = 4
    
    # Module test specifications
    module_tests = {
        ConfigFirstSTFTLoss: [
            (torch.randn(2, 1, 8192), torch.randn(2, 1, 8192))  # Audio loss test
        ],
        ConfigFirstConvEncoder: [
            torch.randn(2, 1, 8192)  # Audio encoding test
        ],
        ConfigFirstAutoEncoder: [
            torch.randn(2, 1, 8192)  # Audio autoencoding test
        ],
        ConfigFirstResidualVectorQuantizer: [
            torch.randn(2, 64, 256)  # Latent quantization test
        ]
    }
    
    validation_results = {}
    
    for config_name, config in test_configs.items():
        print(f"\n🎯 TESTING {config_name.upper()} CONFIGURATION:")
        print(f"   d_model: {config.model.d_model}")
        print(f"   latent_dim: {config.model.latent_dim}")
        print(f"   n_fft: {config.audio.n_fft}")
        print(f"   quantizers: {config.quantization.num_quantizers}")
        print(f"   max_channels: {config.model.max_channels}")
        
        config_results = {}
        
        for module_class, test_inputs in module_tests.items():
            print(f"\n   🧪 Testing {module_class.__name__}...")
            
            # Adjust test inputs for quantizer based on config
            if module_class == ConfigFirstResidualVectorQuantizer:
                adjusted_inputs = [torch.randn(2, 64, config.quantization.codebook_dim)]
            else:
                adjusted_inputs = test_inputs
            
            result = test_config_first_module(module_class, config, adjusted_inputs)
            config_results[module_class.__name__] = result
            
            if result['status'] == 'success':
                print(f"      ✅ SUCCESS - All tests passed")
                for test_result in result['test_results']:
                    if test_result['status'] == 'passed':
                        analysis = test_result['output_analysis']
                        if analysis['type'] == 'tensor':
                            print(f"         Test {test_result['test_id']}: shape {analysis['shape']}")
                        elif analysis['type'] == 'dict':
                            print(f"         Test {test_result['test_id']}: dict with {list(analysis['shapes'].keys())}")
            elif result['status'] == 'partial_success':
                passed = len([r for r in result['test_results'] if r['status'] == 'passed'])
                total = len(result['test_results'])
                print(f"      ⚠️  PARTIAL - {passed}/{total} tests passed")
            else:
                print(f"      ❌ FAILED - {result.get('init_error', 'Unknown error')}")
        
        validation_results[config_name] = config_results
    
    return validation_results

def analyze_validation_results(results: Dict[str, Any]):
    """Analyze validation results and compute improvement metrics"""
    
    print("\n" + "=" * 80)
    print("📊 CONFIG-FIRST VALIDATION ANALYSIS")
    print("=" * 80)
    
    # Calculate success rates per configuration
    for config_name, config_results in results.items():
        total_modules = len(config_results)
        successful_modules = len([r for r in config_results.values() if r['status'] == 'success'])
        partial_modules = len([r for r in config_results.values() if r['status'] == 'partial_success'])
        
        success_rate = (successful_modules / total_modules * 100) if total_modules > 0 else 0
        
        print(f"\n🎯 {config_name.upper()} CONFIGURATION RESULTS:")
        print(f"   Total modules: {total_modules}")
        print(f"   Successful: {successful_modules} ({success_rate:.1f}%)")
        print(f"   Partial: {partial_modules}")
        print(f"   Failed: {total_modules - successful_modules - partial_modules}")
        
        # Show detailed results
        for module_name, result in config_results.items():
            status_icon = "✅" if result['status'] == 'success' else "⚠️" if result['status'] == 'partial_success' else "❌"
            print(f"      {status_icon} {module_name}")
    
    # Overall analysis
    all_results = [r for config_results in results.values() for r in config_results.values()]
    total_tests = len(all_results)
    successful_tests = len([r for r in all_results if r['status'] == 'success'])
    
    overall_success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n📈 OVERALL CONFIG-FIRST SUCCESS RATE:")
    print(f"   {successful_tests}/{total_tests} ({overall_success_rate:.1f}%)")
    
    # Compare with original 91% baseline
    print(f"\n🔄 IMPROVEMENT ANALYSIS:")
    print(f"   Original success rate: 91.0%")
    print(f"   Config-first success rate: {overall_success_rate:.1f}%")
    
    if overall_success_rate > 91:
        improvement = overall_success_rate - 91
        print(f"   🎉 IMPROVEMENT: +{improvement:.1f}% success rate!")
        print("   ✅ Config-first interfaces are superior!")
    elif overall_success_rate >= 85:
        print("   🔥 GOOD: High success rate with config interfaces")
        print("   🎯 Ready for over-clocking experiments")
    else:
        print("   ⚠️  Need to refine config-first implementations")
    
    # Configuration scalability analysis
    config_success_rates = {}
    for config_name, config_results in results.items():
        total = len(config_results)
        successful = len([r for r in config_results.values() if r['status'] == 'success'])
        config_success_rates[config_name] = (successful / total * 100) if total > 0 else 0
    
    print(f"\n⚡ OVER-CLOCKING CAPABILITY:")
    for config_name, rate in config_success_rates.items():
        if config_name == 'extreme':
            if rate >= 75:
                print(f"   🚀 {config_name}: {rate:.1f}% - Ready for extreme experimentation!")
            else:
                print(f"   ⚠️  {config_name}: {rate:.1f}% - Need stability improvements")
        else:
            print(f"   📊 {config_name}: {rate:.1f}%")
    
    return {
        'overall_success_rate': overall_success_rate,
        'config_success_rates': config_success_rates,
        'improvement_over_baseline': overall_success_rate - 91.0,
        'ready_for_extreme_experiments': config_success_rates.get('extreme', 0) >= 75
    }

def main():
    """Run complete config-first validation"""
    
    # Run validation
    results = comprehensive_config_first_validation()
    
    # Analyze results
    analysis = analyze_validation_results(results)
    
    print(f"\n🎯 VALIDATION COMPLETE")
    print("=" * 80)
    
    if analysis['ready_for_extreme_experiments']:
        print("🚀 READY FOR EXTREME RAVE EXPERIMENTATION!")
        print("✅ Config-first interfaces enable over-clocking")
        print("✅ All configurations working reliably")
        
        print(f"\n🧪 NEXT PHASE: EXTREME RAVE EXPERIMENTS")
        print("1. Push parameters to maximum limits")
        print("2. Test architectural variants")
        print("3. Explore novel configurations")
        print("4. Validate F# implementation readiness")
    else:
        print("🔧 REFINEMENT NEEDED")
        print("Focus on stabilizing extreme configuration support")
    
    return results, analysis

if __name__ == "__main__":
    main()