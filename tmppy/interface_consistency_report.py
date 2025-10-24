#!/usr/bin/env python3
"""
INTERFACE CONSISTENCY REPORT: Focus on critical RAVE interface issues
Strategy: Document key inconsistencies and propose concrete fixes
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Union, Optional

def analyze_interface_inconsistencies():
    """Analyze the key interface inconsistencies found in RAVE modules"""
    
    print("🔍 RAVE INTERFACE CONSISTENCY ANALYSIS")
    print("=" * 70)
    print("Critical inconsistencies affecting module interoperability")
    print()
    
    # Key findings from the analysis
    critical_issues = {
        'parameter_naming': {
            'fft_size vs n_fft': {
                'problem': 'Audio processing modules use different names for FFT size',
                'affected_modules': ['STFTLoss', 'MelSpectrogramLoss'],
                'impact': 'Cannot interchangeably use STFT-based modules',
                'recommendation': 'Standardize on n_fft (follows librosa/torchaudio convention)'
            },
            'hop_length vs hop_size': {
                'problem': 'Audio processing modules use different names for hop length', 
                'affected_modules': ['STFTLoss', 'MelSpectrogramLoss'],
                'impact': 'Inconsistent audio parameter naming',
                'recommendation': 'Standardize on hop_length (follows librosa convention)'
            },
            'out_channels vs output_channels': {
                'problem': 'Convolution modules use different channel naming',
                'affected_modules': ['ConvDecoder vs CausalConv1d'],
                'impact': 'Confusing when connecting encoder/decoder chains',
                'recommendation': 'Standardize on out_channels (follows PyTorch convention)'
            }
        },
        
        'type_conflicts': {
            'kernel_size': {
                'problem': 'Some modules expect int, others Union[int, tuple]',
                'impact': 'Type checking failures and inconsistent APIs',
                'recommendation': 'Use Union[int, Tuple[int, ...]] consistently'
            },
            'stride': {
                'problem': 'Mixed int vs Union[int, tuple] expectations',
                'impact': 'Parameter passing failures between modules',
                'recommendation': 'Use Union[int, Tuple[int, ...]] consistently'  
            },
            'channels': {
                'problem': 'Some expect int, others List[int]',
                'affected_modules': ['BlurPool1d vs MultiScaleDiscriminator'],
                'impact': 'Cannot use modules interchangeably',
                'recommendation': 'Context-specific: int for single-scale, List[int] for multi-scale'
            }
        },
        
        'shape_expectations': {
            'audio_input': {
                'problem': 'Modules expect different audio input shapes',
                'variations': ['(B, 1, T)', '(B, C, T)', '(B, T)', '(B, T, C)'],
                'impact': 'Shape mismatches in audio processing pipeline',
                'recommendation': 'Standardize on (B, C, T) for consistency with PyTorch conv1d'
            },
            'latent_representations': {
                'problem': 'VAE/encoder outputs vary in structure',
                'variations': ['tensor', 'dict with keys', 'tuple'],
                'impact': 'Complex handling of encoded representations',
                'recommendation': 'Standardize on dict with "latent" and optional "params" keys'
            }
        }
    }
    
    # Print detailed analysis
    for category, issues in critical_issues.items():
        print(f"🚨 {category.upper().replace('_', ' ')} ISSUES:")
        print("-" * 50)
        
        for issue_name, details in issues.items():
            print(f"\n📍 {issue_name}:")
            print(f"   Problem: {details['problem']}")
            if 'affected_modules' in details:
                print(f"   Affects: {details['affected_modules']}")
            if 'variations' in details:
                print(f"   Variations: {details['variations']}")
            print(f"   Impact: {details['impact']}")
            print(f"   Fix: {details['recommendation']}")
        
        print()
    
    return critical_issues

def propose_rave_interface_standard():
    """Propose a standardized interface for RAVE components"""
    
    print("📋 PROPOSED RAVE INTERFACE STANDARD")
    print("=" * 70)
    
    standard = {
        'audio_processing': {
            'description': 'Standard parameters for audio processing modules',
            'parameters': {
                'sample_rate': 'int = 22050',
                'n_fft': 'int = 1024', 
                'hop_length': 'int = 256',
                'win_length': 'int = 1024',
                'f_min': 'float = 0.0',
                'f_max': 'Optional[float] = None',
                'n_mels': 'int = 80'
            },
            'input_shape': '(batch_size, channels, time_steps)',
            'output_shape': '(batch_size, freq_bins, time_frames) or dict'
        },
        
        'convolution_layers': {
            'description': 'Standard parameters for convolution modules',
            'parameters': {
                'in_channels': 'int',
                'out_channels': 'int', 
                'kernel_size': 'Union[int, Tuple[int, ...]]',
                'stride': 'Union[int, Tuple[int, ...]] = 1',
                'padding': 'Union[int, Tuple[int, ...]] = 0',
                'dilation': 'Union[int, Tuple[int, ...]] = 1',
                'groups': 'int = 1',
                'bias': 'bool = True'
            },
            'input_shape': '(batch_size, in_channels, *spatial_dims)',
            'output_shape': '(batch_size, out_channels, *spatial_dims)'
        },
        
        'encoder_decoder': {
            'description': 'Standard interface for encoders and decoders',
            'encoder_output': {
                'type': 'dict',
                'required_keys': ['latent'],
                'optional_keys': ['mu', 'logvar', 'attention_weights'],
                'latent_shape': '(batch_size, latent_dim)'
            },
            'decoder_input': {
                'latent': '(batch_size, latent_dim)',
                'conditioning': 'Optional[(batch_size, condition_dim)]'
            }
        },
        
        'quantization': {
            'description': 'Standard interface for quantization modules',
            'parameters': {
                'num_quantizers': 'int',
                'codebook_size': 'int = 1024',
                'codebook_dim': 'int = 256', 
                'commitment_cost': 'float = 0.25',
                'epsilon': 'float = 1e-5'
            },
            'output': {
                'type': 'dict',
                'keys': ['quantized', 'indices', 'loss', 'perplexity']
            }
        },
        
        'loss_functions': {
            'description': 'Standard interface for loss functions',
            'parameters': {
                'reduction': "Literal['mean', 'sum', 'none'] = 'mean'"
            },
            'input': 'Tuple[Tensor, Tensor] (prediction, target)',
            'output': 'Tensor (scalar loss)'
        }
    }
    
    for category, spec in standard.items():
        print(f"🎯 {category.upper().replace('_', ' ')}:")
        print(f"   {spec['description']}")
        print()
        
        if 'parameters' in spec:
            print("   Parameters:")
            for param, type_hint in spec['parameters'].items():
                print(f"     {param}: {type_hint}")
            print()
        
        if 'input_shape' in spec:
            print(f"   Input: {spec['input_shape']}")
        
        if 'output_shape' in spec:
            print(f"   Output: {spec['output_shape']}")
        
        if 'encoder_output' in spec:
            eo = spec['encoder_output']
            print(f"   Output type: {eo['type']}")
            print(f"   Required keys: {eo['required_keys']}")
            print(f"   Optional keys: {eo['optional_keys']}")
        
        if 'output' in spec:
            output = spec['output']
            print(f"   Output type: {output['type']}")
            print(f"   Keys: {output['keys']}")
        
        print()
    
    return standard

def generate_interface_migration_plan():
    """Generate a concrete plan for fixing interface inconsistencies"""
    
    print("🛠️  INTERFACE MIGRATION PLAN")
    print("=" * 70)
    
    migration_plan = {
        'phase_1_critical_fixes': {
            'priority': 'HIGH',
            'timeline': '1-2 days',
            'tasks': [
                {
                    'task': 'Standardize audio parameter naming',
                    'files': ['modules/stft_loss.py'],
                    'changes': [
                        'Change fft_size → n_fft',
                        'Change hop_size → hop_length'
                    ],
                    'impact': 'Enables interoperability between STFT modules'
                },
                {
                    'task': 'Fix convolution parameter types',
                    'files': ['modules/causal_conv.py', 'modules/antialiased_conv.py'],
                    'changes': [
                        'Use Union[int, Tuple[int, ...]] for kernel_size',
                        'Use Union[int, Tuple[int, ...]] for stride, padding, dilation'
                    ],
                    'impact': 'Prevents type errors in conv layer chains'
                }
            ]
        },
        
        'phase_2_shape_standardization': {
            'priority': 'HIGH',
            'timeline': '2-3 days', 
            'tasks': [
                {
                    'task': 'Standardize audio input shapes',
                    'files': ['modules/stft_loss.py', 'modules/audio_analysis/*'],
                    'changes': [
                        'Ensure all audio modules expect (B, C, T)',
                        'Add automatic reshaping for legacy (B, T) inputs'
                    ],
                    'impact': 'Eliminates shape mismatches in audio pipeline'
                },
                {
                    'task': 'Standardize encoder/decoder outputs',
                    'files': ['modules/autoencoder_vae.py', 'modules/conv_encoder.py'],
                    'changes': [
                        'Return dict with "latent" key from encoders',
                        'Accept dict input in decoders'
                    ],
                    'impact': 'Enables clean encoder→decoder chaining'
                }
            ]
        },
        
        'phase_3_api_consistency': {
            'priority': 'MEDIUM',
            'timeline': '3-5 days',
            'tasks': [
                {
                    'task': 'Implement base classes for each component type',
                    'files': ['modules/base_classes.py (new)'],
                    'changes': [
                        'Create BaseEncoder, BaseDecoder, BaseQuantizer',
                        'Define abstract methods and standard interfaces'
                    ],
                    'impact': 'Enforces consistent APIs across implementations'
                },
                {
                    'task': 'Add interface validation utilities',
                    'files': ['modules/interface_utils.py (new)'],
                    'changes': [
                        'Shape validation helpers',
                        'Parameter compatibility checkers'
                    ],
                    'impact': 'Catches interface mismatches at runtime'
                }
            ]
        }
    }
    
    for phase, details in migration_plan.items():
        print(f"🎯 {phase.upper().replace('_', ' ')}:")
        print(f"   Priority: {details['priority']}")
        print(f"   Timeline: {details['timeline']}")
        print()
        
        for i, task in enumerate(details['tasks'], 1):
            print(f"   {i}. {task['task']}")
            print(f"      Files: {task['files']}")
            print("      Changes:")
            for change in task['changes']:
                print(f"        - {change}")
            print(f"      Impact: {task['impact']}")
            print()
    
    return migration_plan

def main():
    """Generate comprehensive interface consistency report"""
    
    # Analyze inconsistencies
    critical_issues = analyze_interface_inconsistencies()
    
    print()
    
    # Propose standard
    standard = propose_rave_interface_standard()
    
    print()
    
    # Generate migration plan  
    migration_plan = generate_interface_migration_plan()
    
    print()
    print("🎯 SUMMARY")
    print("=" * 70)
    print("✅ Identified critical interface inconsistencies")
    print("✅ Proposed standardized RAVE interfaces")
    print("✅ Created concrete migration plan")
    print()
    print("🚀 NEXT STEPS:")
    print("1. Implement Phase 1 critical fixes (1-2 days)")
    print("2. Validate fixes with robust testing")
    print("3. Move to Phase 2 shape standardization")
    print("4. Build consistent RAVE pipeline")

if __name__ == "__main__":
    main()