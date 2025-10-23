#!/usr/bin/env python3
"""
Comprehensive test suite for advanced audio modules.

Tests all advanced modules including:
- Music-specific modules (beat sync, chord modeling, source separation)
- Foundation models (AudioMAE, Data2Vec, WavLM)
- Multimodal models (CLAP, ImageBind, captioning)

Follows patterns from existing test_all_25_modules_comprehensive.py.
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime
import traceback
from collections import defaultdict
import warnings

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
import torchaudio
import numpy as np

# Test utilities (reuse existing patterns)
try:
    from test_utils import (
        extract_output,
        validate_shape_behavior,
        check_gradient_flow,
        find_method,
        test_callable_interface
    )
except ImportError:
    # Fallback implementations if test_utils not available
    def extract_output(model, input_tensor):
        """Extract output from model with various calling patterns."""
        try:
            return model(input_tensor)
        except Exception as e:
            # Try different calling patterns
            if hasattr(model, 'forward'):
                return model.forward(input_tensor)
            raise e
    
    def validate_shape_behavior(model, input_tensor, expected_behavior='preserve'):
        """Validate output shape behavior."""
        output = extract_output(model, input_tensor)
        if expected_behavior == 'preserve':
            return input_tensor.shape == output.shape
        elif expected_behavior == 'reduce':
            return len(output.shape) <= len(input_tensor.shape)
        return True
    
    def check_gradient_flow(model, input_tensor):
        """Check if gradients flow properly."""
        if not input_tensor.requires_grad:
            input_tensor = input_tensor.requires_grad_(True)
        output = extract_output(model, input_tensor)
        if isinstance(output, dict):
            loss = sum(v.sum() for v in output.values() if torch.is_tensor(v))
        else:
            loss = output.sum()
        loss.backward()
        return input_tensor.grad is not None


class AdvancedModuleTestRunner:
    """Test runner for advanced audio modules."""
    
    def __init__(self):
        self.results = defaultdict(lambda: defaultdict(dict))
        self.summary = defaultdict(int)
        self.errors = []
        
    def log_result(self, module_name, test_name, success, details=None, error=None):
        """Log test result."""
        self.results[module_name][test_name] = {
            'success': success,
            'details': details,
            'error': str(error) if error else None,
            'timestamp': datetime.now().isoformat()
        }
        
        status = 'PASS' if success else 'FAIL'
        self.summary[status] += 1
        
        if error:
            self.errors.append(f"{module_name}.{test_name}: {error}")
            
        print(f"  {'✅' if success else '❌'} {test_name}: {status}")
        if error and len(str(error)) < 100:
            print(f"    Error: {error}")
    
    def create_test_audio(self, duration=3, sample_rate=22050, batch_size=2):
        """Create synthetic test audio."""
        # Generate a mix of tonal and percussive content
        t = torch.linspace(0, duration, sample_rate * duration)
        
        # Harmonic content (chord progression)
        harmonic = (
            0.3 * torch.sin(2 * torch.pi * 262 * t) +  # C4
            0.2 * torch.sin(2 * torch.pi * 330 * t) +  # E4
            0.15 * torch.sin(2 * torch.pi * 392 * t)   # G4
        )
        
        # Percussive content (beats)
        percussive = torch.zeros_like(t)
        beat_interval = 0.5  # 120 BPM
        for beat_time in torch.arange(0, duration, beat_interval):
            beat_sample = int(beat_time * sample_rate)
            if beat_sample < len(percussive) - 1000:
                # Sharp attack, exponential decay
                burst_length = 1000
                burst = torch.randn(burst_length) * 0.4
                decay = torch.exp(-10 * torch.linspace(0, 0.1, burst_length))
                percussive[beat_sample:beat_sample + burst_length] += burst * decay
        
        # Combine and add to batch
        mixed = harmonic + percussive
        waveform = mixed.unsqueeze(0).expand(batch_size, -1)
        
        return waveform
    
    def create_test_text(self, batch_size=2):
        """Create test text data for multimodal models."""
        # Simple tokenized text (dummy tokens)
        seq_length = 20
        vocab_size = 1000
        
        input_ids = torch.randint(1, vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)
        
        return input_ids, attention_mask
    
    def test_basic_functionality(self, model, module_name, input_data):
        """Test basic forward pass and gradient flow."""
        tests_passed = 0
        total_tests = 3
        
        try:
            # Test 1: Forward pass
            try:
                if isinstance(input_data, tuple):
                    output = model(*input_data)
                else:
                    output = model(input_data)
                self.log_result(module_name, 'forward_pass', True, 
                              f"Output type: {type(output)}")
                tests_passed += 1
            except Exception as e:
                self.log_result(module_name, 'forward_pass', False, error=e)
            
            # Test 2: Output structure validation
            try:
                if isinstance(input_data, tuple):
                    output = model(*input_data)
                else:
                    output = model(input_data)
                
                # Check output is reasonable
                is_valid = True
                details = []
                
                if isinstance(output, dict):
                    details.append(f"Dict with keys: {list(output.keys())}")
                    for key, value in output.items():
                        if torch.is_tensor(value):
                            if torch.isnan(value).any() or torch.isinf(value).any():
                                is_valid = False
                                details.append(f"NaN/Inf in {key}")
                elif isinstance(output, list):
                    details.append(f"List with {len(output)} items")
                    for i, item in enumerate(output):
                        if isinstance(item, dict) and torch.is_tensor(list(item.values())[0]):
                            if any(torch.isnan(v).any() for v in item.values() if torch.is_tensor(v)):
                                is_valid = False
                                details.append(f"NaN in list item {i}")
                elif torch.is_tensor(output):
                    details.append(f"Tensor shape: {output.shape}")
                    if torch.isnan(output).any() or torch.isinf(output).any():
                        is_valid = False
                        details.append("NaN/Inf in output")
                else:
                    details.append(f"Unexpected type: {type(output)}")
                
                self.log_result(module_name, 'output_validation', is_valid, 
                              "; ".join(details))
                if is_valid:
                    tests_passed += 1
            except Exception as e:
                self.log_result(module_name, 'output_validation', False, error=e)
            
            # Test 3: Parameter count and model size
            try:
                param_count = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                details = f"Total: {param_count:,}, Trainable: {trainable_params:,}"
                self.log_result(module_name, 'parameter_count', True, details)
                tests_passed += 1
            except Exception as e:
                self.log_result(module_name, 'parameter_count', False, error=e)
                
        except Exception as e:
            self.log_result(module_name, 'basic_functionality', False, error=e)
        
        return tests_passed, total_tests
    
    def test_gradient_flow(self, model, module_name, input_data):
        """Test gradient flow through model."""
        try:
            model.train()
            
            # Ensure input requires grad
            if isinstance(input_data, tuple):
                processed_input = tuple(
                    x.requires_grad_(True) if torch.is_tensor(x) else x 
                    for x in input_data
                )
            else:
                processed_input = input_data.requires_grad_(True)
            
            # Forward pass
            if isinstance(processed_input, tuple):
                output = model(*processed_input)
            else:
                output = model(processed_input)
            
            # Compute loss for backprop
            if isinstance(output, dict):
                # Sum all tensor values in dict
                loss = torch.tensor(0.0, requires_grad=True)
                for key, value in output.items():
                    if torch.is_tensor(value) and value.requires_grad:
                        loss = loss + value.sum()
            elif isinstance(output, list):
                # Handle list of dicts (like beat synchronizer)
                loss = torch.tensor(0.0, requires_grad=True)
                for item in output:
                    if isinstance(item, dict):
                        for key, value in item.items():
                            if torch.is_tensor(value) and value.requires_grad:
                                loss = loss + value.sum()
                    elif torch.is_tensor(item) and item.requires_grad:
                        loss = loss + item.sum()
            else:
                loss = output.sum()
            
            # Backward pass
            loss.backward()
            
            # Check if gradients exist
            has_gradients = False
            grad_info = []
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        has_gradients = True
                        grad_norm = param.grad.norm().item()
                        grad_info.append(f"{name}: {grad_norm:.6f}")
                    else:
                        grad_info.append(f"{name}: None")
            
            details = f"Gradients found: {has_gradients}. Sample norms: {grad_info[:3]}"
            self.log_result(module_name, 'gradient_flow', has_gradients, details)
            
            return has_gradients
            
        except Exception as e:
            self.log_result(module_name, 'gradient_flow', False, error=e)
            return False
    
    def test_config_compatibility(self, factory_func, module_name):
        """Test configuration compatibility."""
        try:
            from modules.audio_analysis.audio_config import (
                get_music_config, get_speech_config, get_timbralgebraics_config
            )
            
            configs_tested = 0
            configs_passed = 0
            
            # Test different standard configs
            for config_name, config_func in [
                ('music', get_music_config),
                ('speech', get_speech_config), 
                ('timbralgebraics', get_timbralgebraics_config)
            ]:
                try:
                    config = config_func()
                    model = factory_func(config)
                    
                    # Quick forward test
                    test_audio = self.create_test_audio(
                        duration=1, 
                        sample_rate=config.sample_rate,
                        batch_size=1
                    )
                    
                    if 'multimodal' in module_name.lower() or 'clap' in module_name.lower():
                        # Multimodal needs text input too
                        input_ids, attention_mask = self.create_test_text(batch_size=1)
                        output = model(test_audio, input_ids, attention_mask)
                    else:
                        output = model(test_audio)
                    
                    configs_passed += 1
                    
                except Exception as e:
                    self.log_result(module_name, f'config_{config_name}', False, error=e)
                
                configs_tested += 1
            
            success = configs_passed > 0
            details = f"Passed {configs_passed}/{configs_tested} configs"
            self.log_result(module_name, 'config_compatibility', success, details)
            
            return success
            
        except Exception as e:
            self.log_result(module_name, 'config_compatibility', False, error=e)
            return False
    
    def test_memory_efficiency(self, model, module_name, input_data):
        """Test memory usage and efficiency."""
        try:
            if not torch.cuda.is_available():
                self.log_result(module_name, 'memory_efficiency', True, 
                              "Skipped - no CUDA available")
                return True
            
            device = torch.device('cuda')
            model = model.to(device)
            
            if isinstance(input_data, tuple):
                input_data = tuple(x.to(device) if torch.is_tensor(x) else x for x in input_data)
            else:
                input_data = input_data.to(device)
            
            # Measure memory before
            torch.cuda.empty_cache()
            mem_before = torch.cuda.memory_allocated()
            
            # Forward pass
            if isinstance(input_data, tuple):
                output = model(*input_data)
            else:
                output = model(input_data)
            
            # Measure memory after
            mem_after = torch.cuda.memory_allocated()
            mem_used = (mem_after - mem_before) / (1024 ** 2)  # MB
            
            # Check if reasonable (less than 2GB for test)
            reasonable = mem_used < 2048
            details = f"Memory used: {mem_used:.1f} MB"
            
            self.log_result(module_name, 'memory_efficiency', reasonable, details)
            
            # Clean up
            model = model.cpu()
            torch.cuda.empty_cache()
            
            return reasonable
            
        except Exception as e:
            self.log_result(module_name, 'memory_efficiency', False, error=e)
            return False
    
    def run_module_test(self, module_name, factory_func, test_config):
        """Run complete test suite for a module."""
        print(f"\n{'='*60}")
        print(f"🧪 TESTING: {module_name}")
        print(f"{'='*60}")
        
        total_passed = 0
        total_tests = 0
        
        try:
            # Create model
            model = factory_func()
            print(f"✅ Model created successfully")
            
            # Prepare test data
            if test_config.get('needs_text', False):
                # Multimodal model
                audio_data = self.create_test_audio(
                    sample_rate=test_config.get('sample_rate', 22050)
                )
                input_ids, attention_mask = self.create_test_text()
                test_data = (audio_data, input_ids, attention_mask)
            else:
                # Audio-only model
                test_data = self.create_test_audio(
                    sample_rate=test_config.get('sample_rate', 22050)
                )
            
            # Run basic functionality tests
            passed, total = self.test_basic_functionality(model, module_name, test_data)
            total_passed += passed
            total_tests += total
            
            # Test gradient flow
            if self.test_gradient_flow(model, module_name, test_data):
                total_passed += 1
            total_tests += 1
            
            # Test configuration compatibility
            if self.test_config_compatibility(factory_func, module_name):
                total_passed += 1
            total_tests += 1
            
            # Test memory efficiency (optional)
            if self.test_memory_efficiency(model, module_name, test_data):
                total_passed += 1
            total_tests += 1
            
            # Module-specific tests
            if 'beat' in module_name.lower():
                self.test_beat_synchronizer_specifics(model, module_name, test_data)
                total_tests += 2  # Assume 2 specific tests
            elif 'chord' in module_name.lower():
                self.test_chord_modeler_specifics(model, module_name, test_data)
                total_tests += 2
            elif 'separation' in module_name.lower():
                self.test_source_separation_specifics(model, module_name, test_data)
                total_tests += 2
            
        except Exception as e:
            print(f"❌ Failed to create model: {e}")
            self.log_result(module_name, 'model_creation', False, error=e)
            total_tests += 1
        
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        print(f"\n📊 {module_name} Results: {total_passed}/{total_tests} ({success_rate:.1f}%)")
        
        return total_passed, total_tests
    
    def test_beat_synchronizer_specifics(self, model, module_name, test_data):
        """Test beat synchronizer specific functionality."""
        try:
            output = model(test_data)
            
            # Check for beat detection results
            if isinstance(output, list) and len(output) > 0:
                result = output[0]  # First batch item
                
                # Check required keys
                required_keys = ['beat_times', 'tempo', 'onset_times']
                has_keys = all(key in result for key in required_keys)
                
                self.log_result(module_name, 'beat_output_structure', has_keys,
                              f"Has keys: {list(result.keys())}")
                
                # Check tempo is reasonable (60-200 BPM)
                tempo = result.get('tempo', 0)
                reasonable_tempo = 60 <= tempo <= 200 if torch.is_tensor(tempo) else 60 <= tempo <= 200
                self.log_result(module_name, 'tempo_range', reasonable_tempo,
                              f"Detected tempo: {tempo}")
            else:
                self.log_result(module_name, 'beat_output_structure', False,
                              "Unexpected output format")
                self.log_result(module_name, 'tempo_range', False,
                              "No tempo detected")
                
        except Exception as e:
            self.log_result(module_name, 'beat_output_structure', False, error=e)
            self.log_result(module_name, 'tempo_range', False, error=e)
    
    def test_chord_modeler_specifics(self, model, module_name, test_data):
        """Test chord modeler specific functionality."""
        try:
            output = model(test_data)
            
            if isinstance(output, list) and len(output) > 0:
                result = output[0]
                
                # Check for chord analysis results
                required_keys = ['chord_symbols', 'key_root', 'harmonic_functions']
                has_keys = all(key in result for key in required_keys)
                
                self.log_result(module_name, 'chord_output_structure', has_keys,
                              f"Has keys: {list(result.keys())}")
                
                # Check chord symbols are strings
                symbols = result.get('chord_symbols', [])
                valid_symbols = all(isinstance(s, str) for s in symbols)
                self.log_result(module_name, 'chord_symbols_format', valid_symbols,
                              f"Sample symbols: {symbols[:3] if symbols else 'None'}")
            else:
                self.log_result(module_name, 'chord_output_structure', False,
                              "Unexpected output format")
                self.log_result(module_name, 'chord_symbols_format', False,
                              "No chord symbols found")
                
        except Exception as e:
            self.log_result(module_name, 'chord_output_structure', False, error=e)
            self.log_result(module_name, 'chord_symbols_format', False, error=e)
    
    def test_source_separation_specifics(self, model, module_name, test_data):
        """Test source separation specific functionality."""
        try:
            from modules.advanced_modules.source_separation import SeparationType
            
            # Test harmonic-percussive separation
            output = model(test_data, SeparationType.HARMONIC_PERCUSSIVE)
            
            # Check separation result structure
            required_attrs = ['separated_sources', 'source_masks', 'confidence_scores']
            has_attrs = all(hasattr(output, attr) for attr in required_attrs)
            
            self.log_result(module_name, 'separation_output_structure', has_attrs,
                          f"Has attributes: {[attr for attr in required_attrs if hasattr(output, attr)]}")
            
            # Check separated sources
            if hasattr(output, 'separated_sources'):
                sources = output.separated_sources
                expected_sources = ['harmonic', 'percussive']
                has_sources = all(source in sources for source in expected_sources)
                
                self.log_result(module_name, 'separation_sources', has_sources,
                              f"Sources: {list(sources.keys())}")
            else:
                self.log_result(module_name, 'separation_sources', False,
                              "No separated sources found")
                
        except Exception as e:
            self.log_result(module_name, 'separation_output_structure', False, error=e)
            self.log_result(module_name, 'separation_sources', False, error=e)
    
    def print_summary(self):
        """Print test summary."""
        print(f"\n{'='*80}")
        print(f"🎯 ADVANCED MODULES TEST SUMMARY")
        print(f"{'='*80}")
        
        total_pass = self.summary['PASS']
        total_fail = self.summary['FAIL']
        total_tests = total_pass + total_fail
        
        if total_tests > 0:
            success_rate = (total_pass / total_tests) * 100
            print(f"Overall Success Rate: {total_pass}/{total_tests} ({success_rate:.1f}%)")
        else:
            print("No tests completed")
        
        print(f"✅ Passed: {total_pass}")
        print(f"❌ Failed: {total_fail}")
        
        # Module breakdown
        print(f"\n📋 Results by Module:")
        for module_name, tests in self.results.items():
            passed = sum(1 for test in tests.values() if test['success'])
            total = len(tests)
            rate = (passed / total * 100) if total > 0 else 0
            print(f"  {module_name}: {passed}/{total} ({rate:.1f}%)")
        
        # Show critical errors
        if self.errors:
            print(f"\n⚠️  Critical Errors ({len(self.errors)}):")
            for error in self.errors[:5]:  # Show first 5
                print(f"  • {error}")
            if len(self.errors) > 5:
                print(f"  ... and {len(self.errors) - 5} more")
        
        return success_rate if total_tests > 0 else 0


def main():
    """Run comprehensive tests for all advanced modules."""
    print("🚀 Starting Advanced Modules Test Suite")
    print("="*80)
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Initialize test runner
    runner = AdvancedModuleTestRunner()
    
    # Test configuration for all modules
    test_modules = [
        # Music-specific modules
        {
            'name': 'BeatSynchronizer',
            'factory': lambda: __import__('modules.advanced_modules', fromlist=['create_beat_synchronizer']).create_beat_synchronizer(),
            'config': {'sample_rate': 22050, 'needs_text': False}
        },
        {
            'name': 'ChordSequenceModeler', 
            'factory': lambda: __import__('modules.advanced_modules', fromlist=['create_chord_sequence_modeler']).create_chord_sequence_modeler(),
            'config': {'sample_rate': 22050, 'needs_text': False}
        },
        {
            'name': 'SourceSeparationSystem',
            'factory': lambda: __import__('modules.advanced_modules', fromlist=['create_source_separator']).create_source_separator(),
            'config': {'sample_rate': 22050, 'needs_text': False}
        },
        
        # Foundation models
        {
            'name': 'AudioMAE',
            'factory': lambda: __import__('modules.advanced_modules', fromlist=['create_audio_mae']).create_audio_mae(),
            'config': {'sample_rate': 16000, 'needs_text': False}
        },
        {
            'name': 'Data2VecAudio',
            'factory': lambda: __import__('modules.advanced_modules', fromlist=['create_data2vec_audio']).create_data2vec_audio(),
            'config': {'sample_rate': 16000, 'needs_text': False}
        },
        {
            'name': 'WavLM',
            'factory': lambda: __import__('modules.advanced_modules', fromlist=['create_wavlm']).create_wavlm(),
            'config': {'sample_rate': 16000, 'needs_text': False}
        },
        
        # Multimodal models
        {
            'name': 'CLAP',
            'factory': lambda: __import__('modules.advanced_modules', fromlist=['create_clap']).create_clap(),
            'config': {'sample_rate': 22050, 'needs_text': True}
        },
        {
            'name': 'ImageBindAudio',
            'factory': lambda: __import__('modules.advanced_modules', fromlist=['create_imagebind_audio']).create_imagebind_audio(),
            'config': {'sample_rate': 22050, 'needs_text': False}
        },
        {
            'name': 'AudioCaptioningModel',
            'factory': lambda: __import__('modules.advanced_modules', fromlist=['create_audio_captioning_model']).create_audio_captioning_model(),
            'config': {'sample_rate': 22050, 'needs_text': False}
        }
    ]
    
    total_passed = 0
    total_tests = 0
    
    # Run tests for each module
    for test_config in test_modules:
        try:
            passed, tests = runner.run_module_test(
                test_config['name'],
                test_config['factory'],
                test_config['config']
            )
            total_passed += passed
            total_tests += tests
        except Exception as e:
            print(f"❌ Critical failure testing {test_config['name']}: {e}")
            runner.log_result(test_config['name'], 'critical_failure', False, error=e)
            total_tests += 1
    
    # Print final summary
    success_rate = runner.print_summary()
    
    # Save detailed results
    results_file = f"advanced_modules_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(results_file, 'w') as f:
            json.dump(dict(runner.results), f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to: {results_file}")
    except Exception as e:
        print(f"⚠️  Could not save results file: {e}")
    
    # Exit with appropriate code
    if success_rate >= 70:  # 70% success threshold
        print(f"\n🎉 Test suite PASSED with {success_rate:.1f}% success rate!")
        return 0
    else:
        print(f"\n💥 Test suite FAILED with {success_rate:.1f}% success rate")
        return 1


if __name__ == "__main__":
    sys.exit(main())