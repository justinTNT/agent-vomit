#!/usr/bin/env python3
"""
INTERFACE ANALYZER: Analyze module interfaces for consistency across RAVE components
Strategy: Extract signatures, input/output shapes, and identify interface patterns
"""

import torch
import torch.nn as nn
import inspect
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import json
from collections import defaultdict
import ast

class InterfaceAnalyzer:
    """Analyze module interfaces for consistency and design patterns"""
    
    def __init__(self):
        self.rave_modules = {
            'encoders': ['ConvEncoder', 'SequenceEncoder', 'AudioEncoder'],
            'decoders': ['ConvDecoder', 'ConditionalDecoder', 'ConditionalConvDecoder'],
            'autoencoders': ['AutoEncoder', 'VAE', 'ConditionalVAE'],
            'quantizers': ['VectorQuantizer', 'ResidualVectorQuantizer'],
            'discriminators': ['MultiScaleDiscriminator', 'ScaleDiscriminator'],
            'losses': ['STFTLoss', 'MultiScaleSTFTLoss', 'SpectralConvergenceLoss', 'MelSpectrogramLoss'],
            'conv_layers': ['CausalConv1d', 'CausalConvTranspose1d', 'AntialiasedConv1d'],
            'audio_transforms': ['STFT', 'MelSpectrogram', 'PQMFFilterBank'],
        }
        
        self.interface_patterns = defaultdict(list)
        self.shape_expectations = defaultdict(list)
        self.parameter_patterns = defaultdict(set)
    
    def extract_module_signature(self, module_class) -> Dict[str, Any]:
        """Extract detailed signature information from a module"""
        try:
            sig = inspect.signature(module_class.__init__)
            
            signature_info = {
                'class_name': module_class.__name__,
                'module': module_class.__module__,
                'signature_str': str(sig),
                'parameters': {},
                'required_params': [],
                'optional_params': [],
                'type_hints': {},
                'defaults': {}
            }
            
            for name, param in sig.parameters.items():
                if name == 'self':
                    continue
                
                param_info = {
                    'name': name,
                    'annotation': param.annotation if param.annotation != inspect.Parameter.empty else None,
                    'default': param.default if param.default != inspect.Parameter.empty else None,
                    'required': param.default == inspect.Parameter.empty
                }
                
                signature_info['parameters'][name] = param_info
                
                if param_info['required']:
                    signature_info['required_params'].append(name)
                else:
                    signature_info['optional_params'].append(name)
                    signature_info['defaults'][name] = param_info['default']
                
                if param_info['annotation']:
                    signature_info['type_hints'][name] = str(param_info['annotation'])
            
            return signature_info
            
        except Exception as e:
            return {
                'class_name': module_class.__name__,
                'error': f"Could not extract signature: {str(e)}"
            }
    
    def analyze_forward_method(self, module_class) -> Dict[str, Any]:
        """Analyze the forward method to understand input/output patterns"""
        try:
            if not hasattr(module_class, 'forward'):
                return {'error': 'No forward method found'}
            
            forward_sig = inspect.signature(module_class.forward)
            source_lines = inspect.getsourcelines(module_class.forward)[0]
            source_code = ''.join(source_lines)
            
            forward_info = {
                'signature': str(forward_sig),
                'parameters': [],
                'source_preview': source_code[:500],  # First 500 chars
                'input_patterns': [],
                'output_patterns': []
            }
            
            # Extract parameter info
            for name, param in forward_sig.parameters.items():
                if name in ['self']:
                    continue
                forward_info['parameters'].append({
                    'name': name,
                    'annotation': str(param.annotation) if param.annotation != inspect.Parameter.empty else None
                })
            
            # Simple pattern detection in source
            if 'return' in source_code:
                if 'dict' in source_code or '{' in source_code:
                    forward_info['output_patterns'].append('dict_output')
                if 'tuple' in source_code or ',' in source_code:
                    forward_info['output_patterns'].append('tuple_output')
                if 'torch.tensor' in source_code.lower():
                    forward_info['output_patterns'].append('tensor_output')
            
            return forward_info
            
        except Exception as e:
            return {'error': f"Could not analyze forward method: {str(e)}"}
    
    def categorize_parameter_patterns(self, signature_info: Dict[str, Any]) -> Dict[str, List[str]]:
        """Categorize parameters by their semantic patterns"""
        patterns = {
            'dimensions': [],
            'channels': [],
            'audio_params': [],
            'model_config': [],
            'training_params': [],
            'dependencies': []
        }
        
        for param_name in signature_info.get('parameters', {}):
            param_lower = param_name.lower()
            
            # Dimension patterns
            if any(kw in param_lower for kw in ['dim', 'size', 'length']):
                patterns['dimensions'].append(param_name)
            
            # Channel patterns
            elif any(kw in param_lower for kw in ['channel', 'band']):
                patterns['channels'].append(param_name)
            
            # Audio-specific patterns
            elif any(kw in param_lower for kw in ['sample_rate', 'n_fft', 'hop_length', 'f_min', 'f_max', 'mel', 'stft']):
                patterns['audio_params'].append(param_name)
            
            # Model configuration
            elif any(kw in param_lower for kw in ['n_heads', 'n_layers', 'dropout', 'activation']):
                patterns['model_config'].append(param_name)
            
            # Training parameters
            elif any(kw in param_lower for kw in ['lr', 'learning_rate', 'momentum', 'weight_decay', 'beta']):
                patterns['training_params'].append(param_name)
            
            # Dependencies (complex objects)
            elif any(kw in param_lower for kw in ['encoder', 'decoder', 'config', 'module']):
                patterns['dependencies'].append(param_name)
        
        return patterns
    
    def detect_interface_inconsistencies(self, module_signatures: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect inconsistencies across module interfaces"""
        inconsistencies = {
            'parameter_naming': {},
            'type_conflicts': {},
            'missing_standards': {},
            'dimension_patterns': {}
        }
        
        # Group by module category
        param_usage = defaultdict(lambda: defaultdict(list))
        type_usage = defaultdict(lambda: defaultdict(set))
        
        for sig_info in module_signatures:
            if 'error' in sig_info:
                continue
                
            class_name = sig_info['class_name']
            
            for param_name, param_info in sig_info.get('parameters', {}).items():
                param_usage[param_name]['classes'].append(class_name)
                param_usage[param_name]['required'].append(param_info.get('required', False))
                
                if param_info.get('annotation'):
                    type_usage[param_name]['types'].add(str(param_info['annotation']))
        
        # Detect naming inconsistencies
        similar_params = {}
        for param in param_usage.keys():
            for other_param in param_usage.keys():
                if param != other_param:
                    # Check for similar but different parameter names
                    if (self._params_similar(param, other_param) and 
                        len(set(param_usage[param]['classes']) & set(param_usage[other_param]['classes'])) == 0):
                        key = tuple(sorted([param, other_param]))
                        if key not in similar_params:
                            similar_params[key] = {
                                'params': [param, other_param],
                                'classes': [param_usage[param]['classes'], param_usage[other_param]['classes']]
                            }
        
        inconsistencies['parameter_naming'] = similar_params
        
        # Detect type conflicts
        for param_name, type_info in type_usage.items():
            if len(type_info['types']) > 1:
                inconsistencies['type_conflicts'][param_name] = {
                    'types_found': list(type_info['types']),
                    'classes_using': param_usage[param_name]['classes']
                }
        
        return inconsistencies
    
    def _params_similar(self, param1: str, param2: str) -> bool:
        """Check if two parameter names are semantically similar"""
        # Simple similarity check
        common_variants = [
            ('dim', 'size', 'dimension'),
            ('n_fft', 'fft_size'),
            ('hop_length', 'hop_size'),
            ('in_channels', 'input_channels'),
            ('out_channels', 'output_channels'),
            ('d_model', 'embed_dim', 'embedding_dim'),
            ('n_heads', 'num_heads'),
            ('n_layers', 'num_layers')
        ]
        
        for variants in common_variants:
            if param1 in variants and param2 in variants:
                return True
        
        return False
    
    def propose_interface_standards(self, module_signatures: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Propose standardized interfaces based on analysis"""
        
        # Collect all parameter usage
        param_stats = defaultdict(lambda: {
            'frequency': 0,
            'classes': [],
            'types': set(),
            'defaults': set(),
            'required_frequency': 0
        })
        
        for sig_info in module_signatures:
            if 'error' in sig_info:
                continue
                
            for param_name, param_info in sig_info.get('parameters', {}).items():
                stats = param_stats[param_name]
                stats['frequency'] += 1
                stats['classes'].append(sig_info['class_name'])
                
                if param_info.get('annotation'):
                    stats['types'].add(str(param_info['annotation']))
                
                if param_info.get('default') is not None:
                    stats['defaults'].add(str(param_info['default']))
                
                if param_info.get('required', False):
                    stats['required_frequency'] += 1
        
        # Propose standards
        standards = {
            'common_parameters': {},
            'recommended_types': {},
            'suggested_defaults': {},
            'naming_conventions': {}
        }
        
        # Common parameters (used by 3+ modules)
        for param_name, stats in param_stats.items():
            if stats['frequency'] >= 3:
                standards['common_parameters'][param_name] = {
                    'frequency': stats['frequency'],
                    'classes': stats['classes'],
                    'should_be_required': stats['required_frequency'] > stats['frequency'] / 2
                }
        
        # Type recommendations
        for param_name, stats in param_stats.items():
            if len(stats['types']) == 1:
                standards['recommended_types'][param_name] = list(stats['types'])[0]
            elif len(stats['types']) > 1:
                standards['recommended_types'][param_name] = f"CONFLICT: {list(stats['types'])}"
        
        # Default value suggestions
        for param_name, stats in param_stats.items():
            if len(stats['defaults']) == 1:
                standards['suggested_defaults'][param_name] = list(stats['defaults'])[0]
            elif len(stats['defaults']) > 1:
                standards['suggested_defaults'][param_name] = f"VARIES: {list(stats['defaults'])}"
        
        return standards

def import_rave_modules():
    """Import all RAVE-related modules for analysis"""
    modules_found = []
    base_path = Path("/Users/jtnt/Play/agent-vomit")
    
    # Key RAVE module files
    rave_files = [
        "modules/autoencoder_vae.py",
        "modules/conv_encoder.py", 
        "modules/causal_conv.py",
        "modules/antialiased_conv.py",
        "modules/stft_loss.py",
        "modules/residual_vector_quantizer.py",
        "modules/guitar_texture_architecture.py",
        "audio-ml-extensions/audio_gan/multiscale_discriminator.py",
        "audio-ml-extensions/audio_gan/spectral_normalization.py",
        "audio-ml-extensions/audio_gan/pqmf_filterbank.py",
        "candidates/agent_claude/conv_encoder.py",
        "candidates/agent_claude/multi_scale_stft_loss.py",
        "candidates/agent_claude/residual_vector_quantizer.py"
    ]
    
    for file_path in rave_files:
        full_path = base_path / file_path
        if full_path.exists():
            try:
                # Add directory to path
                module_dir = full_path.parent
                if str(module_dir) not in sys.path:
                    sys.path.insert(0, str(module_dir))
                
                spec = importlib.util.spec_from_file_location(
                    full_path.stem,
                    str(full_path)
                )
                
                if spec is not None:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Extract classes
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and 
                            hasattr(attr, '__module__') and
                            attr.__module__ == module.__name__ and
                            hasattr(attr, '__bases__')):
                            
                            if any("Module" in str(base) for base in attr.__bases__):
                                modules_found.append({
                                    'class': attr,
                                    'file_path': str(file_path),
                                    'class_name': attr_name
                                })
                                
            except Exception as e:
                print(f"Could not import {file_path}: {str(e)}")
    
    return modules_found

def main():
    """Analyze RAVE module interfaces for consistency"""
    
    print("🔍 INTERFACE ANALYZER: RAVE Module Interface Consistency")
    print("=" * 70)
    print("Goal: Identify interface inconsistencies and propose standards")
    print()
    
    # Import RAVE modules
    print("📦 Importing RAVE modules...")
    rave_modules = import_rave_modules()
    print(f"Found {len(rave_modules)} RAVE modules to analyze")
    print()
    
    # Initialize analyzer
    analyzer = InterfaceAnalyzer()
    
    # Analyze each module
    module_signatures = []
    forward_analyses = []
    
    print("🔬 Analyzing module interfaces...")
    for module_info in rave_modules:
        module_class = module_info['class']
        
        # Extract signature
        sig_info = analyzer.extract_module_signature(module_class)
        sig_info['file_path'] = module_info['file_path']
        module_signatures.append(sig_info)
        
        # Analyze forward method
        forward_info = analyzer.analyze_forward_method(module_class)
        forward_info['class_name'] = module_class.__name__
        forward_info['file_path'] = module_info['file_path']
        forward_analyses.append(forward_info)
    
    print(f"Analyzed {len(module_signatures)} module signatures")
    print()
    
    # Detect inconsistencies
    print("🚨 Detecting interface inconsistencies...")
    inconsistencies = analyzer.detect_interface_inconsistencies(module_signatures)
    
    # Propose standards
    print("📋 Proposing interface standards...")
    standards = analyzer.propose_interface_standards(module_signatures)
    
    # Generate report
    print("=" * 70)
    print("🎯 INTERFACE ANALYSIS RESULTS")
    print("=" * 70)
    
    # Show parameter naming inconsistencies
    if inconsistencies['parameter_naming']:
        print("⚠️  PARAMETER NAMING INCONSISTENCIES:")
        for params, info in inconsistencies['parameter_naming'].items():
            print(f"  Similar parameters: {params}")
            for i, param in enumerate(info['params']):
                print(f"    {param}: used by {info['classes'][i]}")
        print()
    
    # Show type conflicts
    if inconsistencies['type_conflicts']:
        print("🔥 TYPE CONFLICTS:")
        for param, info in inconsistencies['type_conflicts'].items():
            print(f"  {param}: conflicting types {info['types_found']}")
            print(f"    Used by: {info['classes_using']}")
        print()
    
    # Show common parameters
    common_params = standards['common_parameters']
    print(f"📊 COMMON PARAMETERS (used by 3+ modules):")
    for param, info in sorted(common_params.items(), key=lambda x: x[1]['frequency'], reverse=True):
        required_note = " (should be required)" if info['should_be_required'] else ""
        print(f"  {param}: used by {info['frequency']} modules{required_note}")
    print()
    
    # Show suggested standards
    print("📝 SUGGESTED INTERFACE STANDARDS:")
    print("-" * 50)
    
    # Audio processing standard
    audio_params = [p for p in common_params.keys() if any(kw in p.lower() for kw in ['sample_rate', 'n_fft', 'hop', 'mel'])]
    if audio_params:
        print("🎵 Audio Processing Interface:")
        for param in audio_params:
            default = standards['suggested_defaults'].get(param, 'No default')
            type_hint = standards['recommended_types'].get(param, 'Any')
            print(f"  {param}: {type_hint} = {default}")
        print()
    
    # Model architecture standard
    model_params = [p for p in common_params.keys() if any(kw in p.lower() for kw in ['dim', 'size', 'channel', 'layer', 'head'])]
    if model_params:
        print("🏗️  Model Architecture Interface:")
        for param in model_params:
            default = standards['suggested_defaults'].get(param, 'No default')
            type_hint = standards['recommended_types'].get(param, 'Any')
            print(f"  {param}: {type_hint} = {default}")
        print()
    
    # Save detailed analysis
    analysis_results = {
        'module_signatures': module_signatures,
        'forward_analyses': forward_analyses,
        'inconsistencies': inconsistencies,
        'standards': standards,
        'summary': {
            'total_modules_analyzed': len(module_signatures),
            'parameter_naming_issues': len(inconsistencies['parameter_naming']),
            'type_conflicts': len(inconsistencies['type_conflicts']),
            'common_parameters': len(common_params)
        }
    }
    
    with open('interface_analysis_results.json', 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    print(f"📁 Detailed analysis saved to: interface_analysis_results.json")
    print()
    print("🎯 INTERFACE ANALYSIS COMPLETE")
    print("Next: Review inconsistencies and implement standardized interfaces")

if __name__ == "__main__":
    main()