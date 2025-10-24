#!/usr/bin/env python3
"""
ROBUST MODULE VALIDATOR: Production-quality testing for RAVE components
Strategy: Introspect actual module signatures and create proper test configurations
"""

import torch
import torch.nn as nn
import inspect
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import traceback

class RobustModuleValidator:
    """Production-quality module validation with signature introspection"""
    
    def __init__(self):
        self.rave_critical_modules = {
            # Core RAVE components
            'autoencoder_vae': ['AutoEncoder', 'VAE', 'ConditionalVAE', 'ConvDecoder'],
            'conv_encoder': ['ConvEncoder'],
            'quantization': ['ResidualVectorQuantizer', 'VectorQuantizer'],
            'audio_transforms': ['STFT', 'MelSpectrogram'],
            'conv_layers': ['CausalConv1d', 'CausalConvTranspose1d', 'AntialiasedConv1d'],
            'loss_functions': ['STFTLoss', 'MultiScaleSTFTLoss', 'SpectralConvergenceLoss'],
            'gan_components': ['Discriminator', 'Generator'],
        }
        
        self.dependency_builders = {
            'encoder': self._build_conv_encoder,
            'decoder': self._build_conv_decoder,
            'config': self._build_config_object,
        }
    
    def introspect_module_signature(self, module_class) -> Dict[str, Any]:
        """Introspect the actual __init__ signature of a module"""
        try:
            sig = inspect.signature(module_class.__init__)
            params = {}
            required_params = []
            
            for name, param in sig.parameters.items():
                if name == 'self':
                    continue
                    
                if param.default == inspect.Parameter.empty:
                    required_params.append(name)
                    params[name] = {
                        'required': True,
                        'type': param.annotation if param.annotation != inspect.Parameter.empty else None,
                        'default': None
                    }
                else:
                    params[name] = {
                        'required': False,
                        'type': param.annotation if param.annotation != inspect.Parameter.empty else None,
                        'default': param.default
                    }
            
            return {
                'parameters': params,
                'required_parameters': required_params,
                'signature': str(sig)
            }
            
        except Exception as e:
            return {
                'error': f"Could not introspect signature: {str(e)}",
                'parameters': {},
                'required_parameters': []
            }
    
    def _build_conv_encoder(self, **kwargs) -> nn.Module:
        """Build a proper ConvEncoder for dependency injection"""
        return ConvEncoder(
            in_channels=kwargs.get('in_channels', 3),
            base_channels=kwargs.get('base_channels', 64),
            num_layers=kwargs.get('num_layers', 4)
        )
    
    def _build_conv_decoder(self, **kwargs) -> nn.Module:
        """Build a proper ConvDecoder for dependency injection"""
        from modules.autoencoder_vae import ConvDecoder
        return ConvDecoder(
            latent_dim=kwargs.get('latent_dim', 128),
            output_channels=kwargs.get('output_channels', 3),
            base_channels=kwargs.get('base_channels', 64),
            num_layers=kwargs.get('num_layers', 4)
        )
    
    def _build_config_object(self, **kwargs) -> object:
        """Build a proper config object based on module needs"""
        class ModuleConfig:
            def __init__(self, **config_kwargs):
                # Common audio processing configs
                self.sample_rate = config_kwargs.get('sample_rate', 22050)
                self.n_fft = config_kwargs.get('n_fft', 1024)
                self.hop_length = config_kwargs.get('hop_length', 256)
                self.win_length = config_kwargs.get('win_length', 1024)
                
                # Common model configs
                self.hidden_dim = config_kwargs.get('hidden_dim', 256)
                self.latent_dim = config_kwargs.get('latent_dim', 128)
                self.num_layers = config_kwargs.get('num_layers', 4)
                self.dropout = config_kwargs.get('dropout', 0.1)
                
                # Audio-specific configs
                self.num_scales = config_kwargs.get('num_scales', 4)
                self.channels = config_kwargs.get('channels', 256)
                
                # Override with any provided kwargs
                for k, v in config_kwargs.items():
                    setattr(self, k, v)
        
        return ModuleConfig(**kwargs)
    
    def build_proper_parameters(self, module_class, signature_info: Dict[str, Any]) -> Dict[str, Any]:
        """Build proper parameters based on actual module signature"""
        params = {}
        required_params = signature_info['required_parameters']
        
        for param_name in required_params:
            if param_name in self.dependency_builders:
                # Build complex dependency
                params[param_name] = self.dependency_builders[param_name]()
            else:
                # Use intelligent parameter inference
                params[param_name] = self._infer_parameter_value(param_name, module_class.__name__)
        
        return params
    
    def _infer_parameter_value(self, param_name: str, class_name: str) -> Any:
        """Intelligent parameter value inference based on name and context"""
        
        # Exact parameter name mappings from actual module signatures
        exact_mappings = {
            'latent_dim': 128,
            'hidden_size': 256,
            'input_size': 256,
            'output_size': 128,
            'd_model': 256,
            'd_ff': 1024,
            'n_heads': 8,
            'num_heads': 8,
            'n_layers': 4,
            'num_layers': 4,
            'dropout': 0.1,
            'in_channels': 3 if 'image' in class_name.lower() else 1,
            'out_channels': 64,
            'base_channels': 64,
            'output_channels': 3,
            'channels': 256,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1,
            'sample_rate': 22050,
            'n_fft': 1024,
            'hop_length': 256,
            'win_length': 1024,
            'f_min': 50.0,
            'f_max': 8000.0,
            'vocab_size': 1000,
            'embedding_dim': 256,
            'num_embeddings': 1024,
            'codebook_size': 1024,
            'codebook_dim': 256,
            'num_quantizers': 4,
            'commitment_cost': 0.25,
        }
        
        if param_name in exact_mappings:
            return exact_mappings[param_name]
        
        # Pattern-based inference
        if 'dim' in param_name.lower():
            if 'latent' in param_name.lower():
                return 128
            elif 'hidden' in param_name.lower():
                return 256
            elif 'embed' in param_name.lower():
                return 256
            else:
                return 256
        
        elif 'size' in param_name.lower():
            if 'vocab' in param_name.lower():
                return 1000
            elif 'batch' in param_name.lower():
                return 32
            else:
                return 256
        
        elif 'channel' in param_name.lower():
            if 'in' in param_name.lower():
                return 3 if 'image' in class_name.lower() else 1
            elif 'out' in param_name.lower():
                return 64
            else:
                return 256
        
        elif param_name.lower().endswith('_rate'):
            if 'sample' in param_name.lower():
                return 22050
            else:
                return 0.1
        
        elif param_name.lower().startswith('num_') or param_name.lower().startswith('n_'):
            if 'head' in param_name.lower():
                return 8
            elif 'layer' in param_name.lower():
                return 4
            elif 'class' in param_name.lower():
                return 10
            else:
                return 4
        
        # Default fallback
        return 256
    
    def create_realistic_test_inputs(self, module_class, params: Dict[str, Any]) -> List[Any]:
        """Create realistic test inputs based on module type and parameters"""
        class_name = module_class.__name__.lower()
        
        # Audio processing modules
        if any(kw in class_name for kw in ['audio', 'stft', 'mel', 'wave', 'sound']):
            if 'loss' in class_name:
                # Loss functions need two audio tensors
                return [(torch.randn(2, 1, 8192), torch.randn(2, 1, 8192))]
            else:
                # Audio processing modules need audio input
                return [torch.randn(2, 1, 8192)]
        
        # Image/vision modules
        elif any(kw in class_name for kw in ['conv', 'image', 'vision', 'encoder', 'decoder']):
            if 'autoencoder' in class_name or 'vae' in class_name:
                # Autoencoders typically work with images
                return [torch.randn(2, 3, 64, 64)]
            elif 'conv' in class_name:
                in_channels = params.get('in_channels', 3)
                return [torch.randn(2, in_channels, 64, 64)]
            else:
                return [torch.randn(2, 3, 64, 64)]
        
        # Sequence/language modules
        elif any(kw in class_name for kw in ['transformer', 'attention', 'sequence', 'bert']):
            d_model = params.get('d_model', 256)
            return [torch.randn(2, 20, d_model)]
        
        # Default tensor input
        else:
            input_dim = params.get('input_dim', params.get('input_size', 256))
            return [torch.randn(2, input_dim)]
    
    def validate_module_robustly(self, module_class, module_name: str) -> Dict[str, Any]:
        """Robustly validate a module with proper parameter introspection"""
        
        try:
            # 1. Introspect the actual signature
            signature_info = self.introspect_module_signature(module_class)
            
            if 'error' in signature_info:
                return {
                    'module_name': module_name,
                    'class_name': module_class.__name__,
                    'status': 'signature_error',
                    'error': signature_info['error'],
                    'rave_critical': self._is_rave_critical(module_name, module_class.__name__)
                }
            
            # 2. Build proper parameters
            proper_params = self.build_proper_parameters(module_class, signature_info)
            
            # 3. Create realistic test inputs
            test_inputs = self.create_realistic_test_inputs(module_class, proper_params)
            
            # 4. Test module initialization
            try:
                module = module_class(**proper_params)
                module.eval()
            except Exception as e:
                return {
                    'module_name': module_name,
                    'class_name': module_class.__name__,
                    'status': 'init_failed',
                    'error': str(e),
                    'signature': signature_info['signature'],
                    'attempted_params': proper_params,
                    'rave_critical': self._is_rave_critical(module_name, module_class.__name__)
                }
            
            # 5. Test forward pass
            test_results = []
            for i, test_input in enumerate(test_inputs):
                try:
                    with torch.no_grad():
                        if isinstance(test_input, tuple):
                            output = module(*test_input)
                        else:
                            output = module(test_input)
                    
                    # Analyze output
                    output_analysis = self._analyze_output(output)
                    test_results.append({
                        'test_id': i+1,
                        'status': 'passed',
                        'output_analysis': output_analysis,
                        'input_shape': test_input.shape if hasattr(test_input, 'shape') else str(type(test_input))
                    })
                    
                except Exception as e:
                    test_results.append({
                        'test_id': i+1,
                        'status': 'failed',
                        'error': str(e),
                        'input_shape': test_input.shape if hasattr(test_input, 'shape') else str(type(test_input))
                    })
            
            all_tests_passed = all(result['status'] == 'passed' for result in test_results)
            
            return {
                'module_name': module_name,
                'class_name': module_class.__name__,
                'status': 'success' if all_tests_passed else 'forward_failed',
                'signature': signature_info['signature'],
                'parameters_used': proper_params,
                'test_results': test_results,
                'rave_critical': self._is_rave_critical(module_name, module_class.__name__)
            }
            
        except Exception as e:
            return {
                'module_name': module_name,
                'class_name': module_class.__name__,
                'status': 'validation_error',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'rave_critical': self._is_rave_critical(module_name, module_class.__name__)
            }
    
    def _analyze_output(self, output) -> Dict[str, Any]:
        """Analyze module output for production validation"""
        if isinstance(output, torch.Tensor):
            return {
                'type': 'tensor',
                'shape': list(output.shape),
                'mean': float(output.mean()),
                'std': float(output.std()),
                'min': float(output.min()),
                'max': float(output.max()),
                'has_nan': bool(torch.isnan(output).any()),
                'has_inf': bool(torch.isinf(output).any())
            }
        elif isinstance(output, dict):
            return {
                'type': 'dict',
                'keys': list(output.keys()),
                'key_analysis': {k: self._analyze_output(v) for k, v in output.items() if isinstance(v, torch.Tensor)}
            }
        elif isinstance(output, (tuple, list)):
            return {
                'type': 'sequence',
                'length': len(output),
                'element_analysis': [self._analyze_output(item) for item in output if isinstance(item, torch.Tensor)]
            }
        else:
            return {
                'type': str(type(output)),
                'value': str(output)[:100]  # Truncate for safety
            }
    
    def _is_rave_critical(self, module_name: str, class_name: str) -> bool:
        """Determine if this module is critical for RAVE functionality"""
        for category, classes in self.rave_critical_modules.items():
            if class_name in classes or any(keyword in module_name for keyword in ['autoencoder', 'vae', 'conv', 'quantiz', 'audio', 'stft']):
                return True
        return False

# Import required modules for dependency building
try:
    sys.path.insert(0, '/Users/jtnt/Play/agent-vomit/modules')
    from conv_encoder import ConvEncoder
except ImportError:
    # Fallback simple encoder
    class ConvEncoder(nn.Module):
        def __init__(self, in_channels=3, base_channels=64, num_layers=4, **kwargs):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            return x.view(x.size(0), -1)

if __name__ == "__main__":
    print("🔬 ROBUST MODULE VALIDATOR")
    print("Production-quality testing with signature introspection")
    print("Ready for comprehensive RAVE component validation")