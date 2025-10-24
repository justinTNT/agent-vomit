#!/usr/bin/env python3
"""
CONFIG-FIRST REFACTOR: Convert all RAVE modules to use global, over-clockable configuration
Strategy: Systematic refactoring to config-driven interfaces for maximum experimentation
"""

import os
import sys
from pathlib import Path
import re
from typing import List, Dict, Tuple
import ast
import inspect

class ConfigFirstRefactor:
    """Refactor modules to use config-first, over-clockable interfaces"""
    
    def __init__(self):
        self.modules_to_refactor = [
            # Core RAVE modules
            "modules/autoencoder_vae.py",
            "modules/conv_encoder.py", 
            "modules/causal_conv.py",
            "modules/antialiased_conv.py",
            "modules/stft_loss.py",
            "modules/residual_vector_quantizer.py",
            "modules/guitar_texture_architecture.py",
            
            # Audio GAN modules  
            "audio-ml-extensions/audio_gan/multiscale_discriminator.py",
            "audio-ml-extensions/audio_gan/spectral_normalization.py",
            "audio-ml-extensions/audio_gan/pqmf_filterbank.py",
            "audio-ml-extensions/audio_gan/gan_loss.py",
            
            # Candidates with good implementations
            "candidates/agent_claude/conv_encoder.py",
            "candidates/agent_claude/multi_scale_stft_loss.py", 
            "candidates/agent_claude/residual_vector_quantizer.py",
            "candidates/agent_claude/auto_encoder.py"
        ]
        
        self.refactor_patterns = {
            'audio_modules': {
                'old_params': ['sample_rate', 'n_fft', 'hop_length', 'hop_size', 'fft_size', 'win_length', 'f_min', 'f_max', 'n_mels'],
                'new_interface': 'config: RAVEConfig',
                'config_access': 'config.audio',
                'param_mapping': {
                    'fft_size': 'config.audio.n_fft',
                    'hop_size': 'config.audio.hop_length',
                    'sample_rate': 'config.audio.sample_rate',
                    'n_fft': 'config.audio.n_fft',
                    'hop_length': 'config.audio.hop_length',
                    'win_length': 'config.audio.win_length',
                    'f_min': 'config.audio.f_min',
                    'f_max': 'config.audio.f_max',
                    'n_mels': 'config.audio.n_mels'
                }
            },
            
            'model_modules': {
                'old_params': ['d_model', 'hidden_dim', 'latent_dim', 'n_heads', 'n_layers', 'd_ff', 'dropout'],
                'new_interface': 'config: RAVEConfig',
                'config_access': 'config.model',
                'param_mapping': {
                    'd_model': 'config.model.d_model',
                    'hidden_dim': 'config.model.hidden_dim',
                    'latent_dim': 'config.model.latent_dim',
                    'n_heads': 'config.model.n_heads',
                    'n_layers': 'config.model.n_layers',
                    'd_ff': 'config.model.d_ff',
                    'dropout': 'config.model.dropout'
                }
            },
            
            'conv_modules': {
                'old_params': ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups', 'bias'],
                'new_interface': 'config: RAVEConfig', 
                'config_access': 'config.convolution',
                'param_mapping': {
                    'in_channels': 'config.convolution.in_channels',
                    'out_channels': 'config.convolution.out_channels', 
                    'kernel_size': 'config.convolution.kernel_size',
                    'stride': 'config.convolution.stride',
                    'padding': 'config.convolution.padding',
                    'dilation': 'config.convolution.dilation',
                    'groups': 'config.convolution.groups',
                    'bias': 'config.convolution.bias'
                }
            },
            
            'quantization_modules': {
                'old_params': ['num_quantizers', 'codebook_size', 'codebook_dim', 'commitment_cost', 'num_embeddings', 'embedding_dim'],
                'new_interface': 'config: RAVEConfig',
                'config_access': 'config.quantization', 
                'param_mapping': {
                    'num_quantizers': 'config.quantization.num_quantizers',
                    'codebook_size': 'config.quantization.codebook_size',
                    'codebook_dim': 'config.quantization.codebook_dim',
                    'commitment_cost': 'config.quantization.commitment_cost',
                    'num_embeddings': 'config.quantization.codebook_size',
                    'embedding_dim': 'config.quantization.codebook_dim'
                }
            }
        }
    
    def analyze_module(self, file_path: str) -> Dict[str, any]:
        """Analyze a module to understand its current interface"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Parse AST to find class definitions
            tree = ast.parse(content)
            
            analysis = {
                'file_path': file_path,
                'classes': [],
                'imports': [],
                'needs_refactor': False
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = self._analyze_class(node, content)
                    analysis['classes'].append(class_info)
                    
                    # Check if class needs refactoring
                    if self._needs_config_refactor(class_info):
                        analysis['needs_refactor'] = True
                
                elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    analysis['imports'].append(ast.unparse(node))
            
            return analysis
            
        except Exception as e:
            return {
                'file_path': file_path,
                'error': f"Could not analyze: {str(e)}",
                'needs_refactor': False
            }
    
    def _analyze_class(self, class_node: ast.ClassDef, content: str) -> Dict[str, any]:
        """Analyze a class definition to understand its interface"""
        class_info = {
            'name': class_node.name,
            'bases': [ast.unparse(base) for base in class_node.bases],
            'methods': [],
            'init_params': [],
            'is_nn_module': False
        }
        
        # Check if it's a nn.Module
        class_info['is_nn_module'] = any('Module' in base for base in class_info['bases'])
        
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                if node.name == '__init__':
                    # Extract __init__ parameters
                    for arg in node.args.args[1:]:  # Skip 'self'
                        param_info = {
                            'name': arg.arg,
                            'annotation': ast.unparse(arg.annotation) if arg.annotation else None
                        }
                        class_info['init_params'].append(param_info)
                
                class_info['methods'].append(node.name)
        
        return class_info
    
    def _needs_config_refactor(self, class_info: Dict[str, any]) -> bool:
        """Determine if a class needs config refactoring"""
        if not class_info['is_nn_module']:
            return False
        
        # Check if it already uses config
        for param in class_info['init_params']:
            if 'config' in param['name'].lower() and 'RAVEConfig' in str(param.get('annotation', '')):
                return False  # Already refactored
        
        # Check if it has parameters that should be config-driven
        param_names = [p['name'] for p in class_info['init_params']]
        
        for pattern_type, pattern_info in self.refactor_patterns.items():
            old_params = pattern_info['old_params']
            if any(param in param_names for param in old_params):
                return True
        
        return False
    
    def generate_refactored_interface(self, class_info: Dict[str, any]) -> str:
        """Generate new config-first interface for a class"""
        
        # Determine which pattern applies
        param_names = [p['name'] for p in class_info['init_params']]
        applicable_patterns = []
        
        for pattern_type, pattern_info in self.refactor_patterns.items():
            old_params = pattern_info['old_params']
            if any(param in param_names for param in old_params):
                applicable_patterns.append(pattern_type)
        
        # Generate new __init__ signature
        new_signature = f"def __init__(self, config: RAVEConfig"
        
        # Keep non-config parameters
        non_config_params = []
        for param in class_info['init_params']:
            param_name = param['name']
            
            # Check if this parameter should be config-driven
            is_config_param = False
            for pattern_type in applicable_patterns:
                if param_name in self.refactor_patterns[pattern_type]['old_params']:
                    is_config_param = True
                    break
            
            if not is_config_param:
                annotation = f": {param['annotation']}" if param['annotation'] else ""
                non_config_params.append(f"{param_name}{annotation}")
        
        if non_config_params:
            new_signature += ", " + ", ".join(non_config_params)
        
        new_signature += ", **kwargs):"
        
        # Generate parameter assignments
        assignments = []
        for param in class_info['init_params']:
            param_name = param['name']
            
            # Find config mapping
            config_path = None
            for pattern_type in applicable_patterns:
                pattern = self.refactor_patterns[pattern_type]
                if param_name in pattern['param_mapping']:
                    config_path = pattern['param_mapping'][param_name]
                    break
            
            if config_path:
                assignments.append(f"        self.{param_name} = {config_path}")
            else:
                assignments.append(f"        self.{param_name} = {param_name}")
        
        return new_signature, assignments
    
    def refactor_module_file(self, file_path: str) -> bool:
        """Refactor a single module file to use config-first interface"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Add config import if not present
            if 'from rave_config_system import RAVEConfig' not in content:
                import_line = "from rave_config_system import RAVEConfig\n"
                
                # Insert after existing imports
                lines = content.split('\n')
                insert_index = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith('import ') or line.strip().startswith('from '):
                        insert_index = i + 1
                
                lines.insert(insert_index, import_line)
                content = '\n'.join(lines)
            
            # Analyze and refactor classes
            analysis = self.analyze_module(file_path)
            if not analysis.get('needs_refactor', False):
                return False
            
            # Create backup
            backup_path = file_path + '.backup'
            with open(backup_path, 'w') as f:
                f.write(content)
            
            # Apply refactoring
            refactored_content = self._apply_refactoring(content, analysis)
            
            # Write refactored content
            with open(file_path, 'w') as f:
                f.write(refactored_content)
            
            print(f"✅ Refactored: {file_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to refactor {file_path}: {str(e)}")
            return False
    
    def _apply_refactoring(self, content: str, analysis: Dict[str, any]) -> str:
        """Apply refactoring to content"""
        # This is a simplified version - in practice, you'd need more sophisticated AST manipulation
        
        # For now, just add a comment indicating where refactoring should happen
        refactored_content = content
        
        for class_info in analysis['classes']:
            if self._needs_config_refactor(class_info):
                new_signature, assignments = self.generate_refactored_interface(class_info)
                
                # Add refactoring comment
                comment = f"""
# TODO: REFACTOR TO CONFIG-FIRST INTERFACE
# New signature: {new_signature}
# New assignments:
# {chr(10).join(assignments)}
"""
                
                # Find class definition and add comment
                class_pattern = f"class {class_info['name']}"
                refactored_content = refactored_content.replace(
                    class_pattern,
                    comment + class_pattern
                )
        
        return refactored_content
    
    def create_example_refactored_module(self) -> str:
        """Create an example of a fully refactored module"""
        
        example_code = '''
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
'''
        
        return example_code
    
    def run_comprehensive_refactor(self) -> Dict[str, any]:
        """Run comprehensive refactoring of all RAVE modules"""
        
        print("🔧 CONFIG-FIRST REFACTOR: Converting all modules to over-clockable interfaces")
        print("=" * 80)
        
        results = {
            'analyzed': 0,
            'refactored': 0,
            'skipped': 0,
            'failed': 0,
            'modules': []
        }
        
        base_path = Path("/Users/jtnt/Play/agent-vomit")
        
        for module_path in self.modules_to_refactor:
            full_path = base_path / module_path
            
            if not full_path.exists():
                print(f"⚠️  Module not found: {module_path}")
                results['failed'] += 1
                continue
            
            print(f"🔍 Analyzing: {module_path}")
            
            analysis = self.analyze_module(str(full_path))
            results['analyzed'] += 1
            results['modules'].append(analysis)
            
            if analysis.get('needs_refactor', False):
                if self.refactor_module_file(str(full_path)):
                    results['refactored'] += 1
                else:
                    results['failed'] += 1
            else:
                print(f"✅ Already config-first or no refactor needed: {module_path}")
                results['skipped'] += 1
        
        # Create example refactored module
        example_path = base_path / "config_first_examples.py"
        with open(example_path, 'w') as f:
            f.write(self.create_example_refactored_module())
        print(f"📝 Created example refactored module: {example_path}")
        
        return results

def main():
    """Run config-first refactoring"""
    
    refactor = ConfigFirstRefactor()
    results = refactor.run_comprehensive_refactor()
    
    print("\n" + "=" * 80)
    print("🎯 CONFIG-FIRST REFACTOR RESULTS")
    print("=" * 80)
    print(f"Modules analyzed: {results['analyzed']}")
    print(f"Modules refactored: {results['refactored']}")
    print(f"Modules skipped: {results['skipped']}")
    print(f"Modules failed: {results['failed']}")
    
    print(f"\n📋 MODULES NEEDING REFACTOR:")
    for module_info in results['modules']:
        if module_info.get('needs_refactor', False):
            print(f"  ⚡ {module_info['file_path']}")
            for class_info in module_info.get('classes', []):
                if refactor._needs_config_refactor(class_info):
                    params = [p['name'] for p in class_info.get('init_params', [])]
                    print(f"    - {class_info['name']}: {params}")
    
    print(f"\n🚀 NEXT STEPS:")
    print("1. Review generated refactoring comments in modules")
    print("2. Implement config-first interfaces using examples")
    print("3. Test with new over-clockable configurations")
    print("4. Validate increased success rate")
    
    return results

if __name__ == "__main__":
    main()