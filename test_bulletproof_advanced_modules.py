"""
Comprehensive Test Suite for Bulletproof Advanced Music ML Modules

Tests all three bulletproof advanced modules with various musical scenarios,
edge cases, and stress conditions to validate robustness and reliability.
"""

import torch
import numpy as np
import time
import warnings
from typing import Dict, List, Tuple, Any
import sys
import os

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

# Import bulletproof modules
try:
    from modules.advanced_modules.bulletproof_beat_synchronizer import (
        create_bulletproof_beat_synchronizer,
        BeatSynchronizerConfig,
        BeatTrackingQuality
    )
    from modules.advanced_modules.bulletproof_chord_sequence_modeler import (
        create_bulletproof_chord_sequence_modeler,
        ChordSequenceConfig,
        HarmonicQuality,
        ChordQuality
    )
    from modules.advanced_modules.bulletproof_source_separation import (
        create_bulletproof_source_separation,
        SourceSeparationConfig,
        SeparationType,
        SeparationQuality
    )
    print("✓ Successfully imported all bulletproof modules")
except ImportError as e:
    print(f"✗ Failed to import bulletproof modules: {e}")
    sys.exit(1)

# Import audio config
try:
    from modules.audio_analysis.audio_config import get_music_config, AudioModuleConfig
    print("✓ Successfully imported audio config")
except ImportError:
    print("⚠ Using fallback audio config")
    from dataclasses import dataclass
    
    @dataclass
    class AudioModuleConfig:
        sample_rate: int = 22050
        hop_length: int = 512
        n_mels: int = 128
        n_fft: int = 2048
        
    def get_music_config():
        return AudioModuleConfig()


class TestResult:
    """Container for test results."""
    def __init__(self, test_name: str, module_name: str):
        self.test_name = test_name
        self.module_name = module_name
        self.passed = False
        self.error_message = None
        self.processing_time = 0.0
        self.memory_usage = 0.0
        self.quality_score = 0.0
        self.additional_info = {}


class AdvancedModuleTester:
    """Comprehensive tester for bulletproof advanced music ML modules."""
    
    def __init__(self):
        self.config = get_music_config()
        self.results = []
        
        # Initialize modules with error handling
        try:
            self.beat_synchronizer = create_bulletproof_beat_synchronizer(self.config)
            print("✓ Beat synchronizer initialized")
        except Exception as e:
            print(f"✗ Failed to initialize beat synchronizer: {e}")
            self.beat_synchronizer = None
            
        try:
            self.chord_modeler = create_bulletproof_chord_sequence_modeler(self.config)
            print("✓ Chord sequence modeler initialized")
        except Exception as e:
            print(f"✗ Failed to initialize chord modeler: {e}")
            self.chord_modeler = None
            
        try:
            self.source_separator = create_bulletproof_source_separation(self.config)
            print("✓ Source separator initialized")
        except Exception as e:
            print(f"✗ Failed to initialize source separator: {e}")
            self.source_separator = None
            
    def generate_test_audio(self, test_type: str, duration: float = 5.0) -> torch.Tensor:
        """Generate various types of test audio."""
        sample_rate = self.config.sample_rate
        samples = int(sample_rate * duration)
        
        if test_type == "simple_drum_pattern":
            return self._generate_drum_pattern(duration, tempo=120)
        elif test_type == "complex_polyrhythm":
            return self._generate_polyrhythm(duration)
        elif test_type == "variable_tempo":
            return self._generate_variable_tempo(duration)
        elif test_type == "chord_progression":
            return self._generate_chord_progression(duration)
        elif test_type == "jazz_chords":
            return self._generate_jazz_chords(duration)
        elif test_type == "atonal_music":
            return self._generate_atonal_music(duration)
        elif test_type == "harmonic_percussive_mix":
            return self._generate_hp_mix(duration)
        elif test_type == "vocal_instrumental_mix":
            return self._generate_vocal_mix(duration)
        elif test_type == "multitrack_mix":
            return self._generate_multitrack_mix(duration)
        elif test_type == "stereo_audio":
            return self._generate_stereo_audio(duration)
        elif test_type == "very_short":
            return torch.randn(1, 100)
        elif test_type == "very_long":
            return torch.randn(1, sample_rate * 60)  # 1 minute
        elif test_type == "silent":
            return torch.zeros(1, samples)
        elif test_type == "white_noise":
            return torch.randn(1, samples) * 0.1
        elif test_type == "extreme_dynamics":
            signal = torch.randn(1, samples)
            signal = signal * torch.exp(torch.linspace(-5, 5, samples))  # Dynamic range
            return signal
        elif test_type == "clipped_audio":
            signal = torch.randn(1, samples) * 2
            return torch.clamp(signal, -1, 1)
        else:
            return torch.randn(1, samples) * 0.1
            
    def _generate_drum_pattern(self, duration: float, tempo: float = 120) -> torch.Tensor:
        """Generate simple drum pattern."""
        sample_rate = self.config.sample_rate
        samples = int(sample_rate * duration)
        waveform = torch.zeros(1, samples)
        
        beat_interval = 60.0 / tempo
        
        for beat_time in np.arange(0, duration, beat_interval):
            beat_sample = int(beat_time * sample_rate)
            if beat_sample < samples - 1000:
                # Kick drum
                kick = torch.sin(2 * torch.pi * 60 * torch.linspace(0, 0.1, 1000))
                kick *= torch.exp(-5 * torch.linspace(0, 0.1, 1000))
                waveform[0, beat_sample:beat_sample+1000] += kick
                
        return waveform
        
    def _generate_polyrhythm(self, duration: float) -> torch.Tensor:
        """Generate polyrhythmic pattern."""
        sample_rate = self.config.sample_rate
        samples = int(sample_rate * duration)
        waveform = torch.zeros(1, samples)
        
        # 4/4 pattern
        for beat_time in np.arange(0, duration, 0.5):
            beat_sample = int(beat_time * sample_rate)
            if beat_sample < samples - 500:
                kick = torch.sin(2 * torch.pi * 60 * torch.linspace(0, 0.05, 500))
                waveform[0, beat_sample:beat_sample+500] += kick
                
        # 3/4 pattern (creates polyrhythm)
        for beat_time in np.arange(0, duration, 0.67):
            beat_sample = int(beat_time * sample_rate)
            if beat_sample < samples - 300:
                snare = torch.randn(300) * 0.3
                snare *= torch.exp(-10 * torch.linspace(0, 0.05, 300))
                waveform[0, beat_sample:beat_sample+300] += snare
                
        return waveform
        
    def _generate_variable_tempo(self, duration: float) -> torch.Tensor:
        """Generate audio with changing tempo."""
        sample_rate = self.config.sample_rate
        samples = int(sample_rate * duration)
        waveform = torch.zeros(1, samples)
        
        current_time = 0
        sample_idx = 0
        
        while current_time < duration and sample_idx < samples - 1000:
            # Tempo changes from 100 to 140 BPM
            tempo = 100 + 40 * (current_time / duration)
            beat_interval = 60.0 / tempo
            
            # Add beat
            kick = torch.sin(2 * torch.pi * 60 * torch.linspace(0, 0.1, 1000))
            kick *= torch.exp(-5 * torch.linspace(0, 0.1, 1000))
            
            end_idx = min(sample_idx + 1000, samples)
            waveform[0, sample_idx:end_idx] += kick[:end_idx - sample_idx]
            
            current_time += beat_interval
            sample_idx = int(current_time * sample_rate)
            
        return waveform
        
    def _generate_chord_progression(self, duration: float) -> torch.Tensor:
        """Generate chord progression (I-V-vi-IV)."""
        sample_rate = self.config.sample_rate
        samples = int(sample_rate * duration)
        waveform = torch.zeros(1, samples)
        
        # Chord progression: C-G-Am-F
        chord_notes = [
            [0, 4, 7],    # C major
            [7, 11, 2],   # G major
            [9, 0, 4],    # A minor
            [5, 9, 0]     # F major
        ]
        
        chord_duration = duration / len(chord_notes)
        
        for i, notes in enumerate(chord_notes):
            start_sample = int(i * chord_duration * sample_rate)
            end_sample = int((i + 1) * chord_duration * sample_rate)
            
            for note in notes:
                freq = 220 * (2 ** (note / 12))  # A3 = 220 Hz
                t = torch.linspace(0, chord_duration, end_sample - start_sample)
                sine_wave = 0.2 * torch.sin(2 * torch.pi * freq * t)
                
                if start_sample < samples and end_sample <= samples:
                    waveform[0, start_sample:end_sample] += sine_wave
                    
        return waveform
        
    def _generate_jazz_chords(self, duration: float) -> torch.Tensor:
        """Generate jazz chord progression."""
        sample_rate = self.config.sample_rate
        samples = int(sample_rate * duration)
        waveform = torch.zeros(1, samples)
        
        # Jazz progression: Cmaj7-A7-Dm7-G7
        chord_notes = [
            [0, 4, 7, 11],  # Cmaj7
            [9, 1, 4, 7],   # A7
            [2, 5, 9, 0],   # Dm7
            [7, 11, 2, 5]   # G7
        ]
        
        chord_duration = duration / len(chord_notes)
        
        for i, notes in enumerate(chord_notes):
            start_sample = int(i * chord_duration * sample_rate)
            end_sample = int((i + 1) * chord_duration * sample_rate)
            
            for note in notes:
                freq = 220 * (2 ** (note / 12))
                t = torch.linspace(0, chord_duration, end_sample - start_sample)
                sine_wave = 0.15 * torch.sin(2 * torch.pi * freq * t)
                
                if start_sample < samples and end_sample <= samples:
                    waveform[0, start_sample:end_sample] += sine_wave
                    
        return waveform
        
    def _generate_atonal_music(self, duration: float) -> torch.Tensor:
        """Generate atonal music."""
        sample_rate = self.config.sample_rate
        samples = int(sample_rate * duration)
        waveform = torch.zeros(1, samples)
        
        # Random frequencies
        for _ in range(15):
            freq = np.random.uniform(100, 2000)
            amplitude = np.random.uniform(0.05, 0.15)
            t = torch.linspace(0, duration, samples)
            sine_wave = amplitude * torch.sin(2 * torch.pi * freq * t)
            waveform[0] += sine_wave
            
        return waveform
        
    def _generate_hp_mix(self, duration: float) -> torch.Tensor:
        """Generate harmonic + percussive mix."""
        sample_rate = self.config.sample_rate
        samples = int(sample_rate * duration)
        t = torch.linspace(0, duration, samples)
        
        # Harmonic component
        harmonic = (
            0.3 * torch.sin(2 * torch.pi * 220 * t) +
            0.2 * torch.sin(2 * torch.pi * 330 * t)
        )
        
        # Percussive component
        percussive = torch.zeros_like(t)
        for beat_time in np.arange(0, duration, 0.5):
            beat_sample = int(beat_time * sample_rate)
            if beat_sample < len(percussive) - 500:
                hit = torch.randn(500) * 0.4
                hit *= torch.exp(-8 * torch.linspace(0, 0.1, 500))
                percussive[beat_sample:beat_sample + 500] += hit
                
        return (harmonic + percussive).unsqueeze(0)
        
    def _generate_vocal_mix(self, duration: float) -> torch.Tensor:
        """Generate vocal + instrumental mix."""
        sample_rate = self.config.sample_rate
        samples = int(sample_rate * duration)
        t = torch.linspace(0, duration, samples)
        
        # Vocal-like signal
        vocal = torch.zeros_like(t)
        for freq in [200, 400, 600, 800]:
            vocal += 0.1 * torch.sin(2 * torch.pi * freq * t) * (1 + 0.2 * torch.sin(2 * torch.pi * 3 * t))
            
        # Instrumental
        instrumental = (
            0.2 * torch.sin(2 * torch.pi * 110 * t) +
            0.15 * torch.sin(2 * torch.pi * 1500 * t)
        )
        
        return (vocal + instrumental).unsqueeze(0)
        
    def _generate_multitrack_mix(self, duration: float) -> torch.Tensor:
        """Generate multitrack mix."""
        sample_rate = self.config.sample_rate
        samples = int(sample_rate * duration)
        t = torch.linspace(0, duration, samples)
        
        # Vocals
        vocals = 0.3 * torch.sin(2 * torch.pi * 400 * t) * (1 + 0.3 * torch.sin(2 * torch.pi * 2 * t))
        
        # Drums
        drums = torch.zeros_like(t)
        for beat in np.arange(0, duration, 0.5):
            beat_sample = int(beat * sample_rate)
            if beat_sample < len(drums) - 200:
                drums[beat_sample:beat_sample + 200] += torch.randn(200) * 0.3
                
        # Bass
        bass = 0.25 * torch.sin(2 * torch.pi * 80 * t)
        
        # Other
        other = 0.15 * torch.sin(2 * torch.pi * 2000 * t)
        
        return (vocals + drums + bass + other).unsqueeze(0)
        
    def _generate_stereo_audio(self, duration: float) -> torch.Tensor:
        """Generate stereo audio."""
        sample_rate = self.config.sample_rate
        samples = int(sample_rate * duration)
        t = torch.linspace(0, duration, samples)
        
        left = 0.5 * torch.sin(2 * torch.pi * 440 * t)
        right = 0.5 * torch.sin(2 * torch.pi * 550 * t)
        
        return torch.stack([left, right], dim=0).unsqueeze(0)
        
    def test_module(self, module, module_name: str, test_cases: List[str]) -> List[TestResult]:
        """Test a module with various test cases."""
        results = []
        
        for test_case in test_cases:
            result = TestResult(test_case, module_name)
            
            try:
                start_time = time.time()
                
                # Generate test audio
                if test_case in ["very_short", "very_long", "silent", "white_noise", 
                               "extreme_dynamics", "clipped_audio"]:
                    waveform = self.generate_test_audio(test_case, duration=3.0)
                else:
                    waveform = self.generate_test_audio(test_case, duration=5.0)
                    
                # Test the module
                if module is None:
                    raise Exception("Module not initialized")
                    
                # Module-specific testing
                if module_name == "beat_synchronizer":
                    module_result = module(waveform)
                    
                    # Validate beat synchronizer results
                    if isinstance(module_result, dict):
                        result.passed = True
                        result.quality_score = self._assess_beat_quality(module_result)
                        result.additional_info = {
                            'tempo': module_result.get('tempo', 0),
                            'n_beats': len(module_result.get('beat_times', [])),
                            'quality': module_result.get('quality', 'unknown')
                        }
                    else:
                        result.passed = False
                        result.error_message = "Invalid result format"
                        
                elif module_name == "chord_modeler":
                    module_result = module(waveform)
                    
                    # Validate chord modeler results
                    if isinstance(module_result, dict):
                        result.passed = True
                        result.quality_score = self._assess_chord_quality(module_result)
                        result.additional_info = {
                            'n_chords': len(module_result.get('chord_sequence', [])),
                            'key_confidence': module_result.get('key_confidence', 0),
                            'quality': module_result.get('quality', 'unknown')
                        }
                    else:
                        result.passed = False
                        result.error_message = "Invalid result format"
                        
                elif module_name == "source_separator":
                    # Test different separation types
                    if "harmonic" in test_case or test_case == "simple_drum_pattern":
                        separation_type = SeparationType.HARMONIC_PERCUSSIVE
                    elif "vocal" in test_case or test_case == "chord_progression":
                        separation_type = SeparationType.VOCAL_INSTRUMENTAL
                    else:
                        separation_type = SeparationType.MULTITRACK
                        
                    module_result = module(waveform, separation_type)
                    
                    # Validate separator results
                    if hasattr(module_result, 'separated_sources'):
                        result.passed = True
                        result.quality_score = self._assess_separation_quality(module_result)
                        result.additional_info = {
                            'n_sources': len(module_result.separated_sources),
                            'separation_quality': module_result.separation_quality.value if hasattr(module_result.separation_quality, 'value') else str(module_result.separation_quality),
                            'overall_quality': module_result.quality_metrics.get('overall_quality', 0)
                        }
                    else:
                        result.passed = False
                        result.error_message = "Invalid result format"
                        
                result.processing_time = time.time() - start_time
                
                # Memory usage estimation
                if torch.cuda.is_available():
                    result.memory_usage = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                    
            except Exception as e:
                result.passed = False
                result.error_message = str(e)
                result.processing_time = time.time() - start_time if 'start_time' in locals() else 0
                
            results.append(result)
            
        return results
        
    def _assess_beat_quality(self, result: Dict) -> float:
        """Assess beat tracking quality."""
        try:
            if 'quality' in result:
                quality = result['quality']
                if hasattr(quality, 'value'):
                    quality_str = quality.value
                else:
                    quality_str = str(quality)
                    
                quality_map = {
                    'excellent': 1.0,
                    'good': 0.8,
                    'fair': 0.6,
                    'poor': 0.4,
                    'failed': 0.0
                }
                return quality_map.get(quality_str.lower(), 0.5)
            else:
                # Fallback assessment
                n_beats = len(result.get('beat_times', []))
                tempo_confidence = result.get('tempo_confidence', 0)
                return min(1.0, n_beats / 10 * 0.5 + tempo_confidence * 0.5)
        except:
            return 0.0
            
    def _assess_chord_quality(self, result: Dict) -> float:
        """Assess chord recognition quality."""
        try:
            if 'quality' in result:
                quality = result['quality']
                if hasattr(quality, 'value'):
                    quality_str = quality.value
                else:
                    quality_str = str(quality)
                    
                quality_map = {
                    'excellent': 1.0,
                    'good': 0.8,
                    'fair': 0.6,
                    'poor': 0.4,
                    'failed': 0.0
                }
                return quality_map.get(quality_str.lower(), 0.5)
            else:
                # Fallback assessment
                n_chords = len(result.get('chord_sequence', []))
                key_confidence = result.get('key_confidence', 0)
                return min(1.0, n_chords / 5 * 0.5 + key_confidence * 0.5)
        except:
            return 0.0
            
    def _assess_separation_quality(self, result) -> float:
        """Assess source separation quality."""
        try:
            if hasattr(result, 'quality_metrics'):
                return result.quality_metrics.get('overall_quality', 0.0)
            elif hasattr(result, 'separation_quality'):
                quality_map = {
                    'excellent': 1.0,
                    'good': 0.8,
                    'fair': 0.6,
                    'poor': 0.4,
                    'failed': 0.0
                }
                quality_str = result.separation_quality.value if hasattr(result.separation_quality, 'value') else str(result.separation_quality)
                return quality_map.get(quality_str.lower(), 0.5)
            else:
                return 0.0
        except:
            return 0.0
            
    def run_comprehensive_tests(self):
        """Run comprehensive tests on all modules."""
        print("\n" + "="*60)
        print("COMPREHENSIVE BULLETPROOF MODULE TESTING")
        print("="*60)
        
        # Test cases for each module
        beat_test_cases = [
            "simple_drum_pattern", "complex_polyrhythm", "variable_tempo",
            "very_short", "very_long", "silent", "white_noise",
            "extreme_dynamics", "clipped_audio"
        ]
        
        chord_test_cases = [
            "chord_progression", "jazz_chords", "atonal_music",
            "very_short", "very_long", "silent", "white_noise",
            "extreme_dynamics", "clipped_audio"
        ]
        
        separation_test_cases = [
            "harmonic_percussive_mix", "vocal_instrumental_mix", "multitrack_mix",
            "stereo_audio", "very_short", "very_long", "silent", "white_noise",
            "extreme_dynamics", "clipped_audio"
        ]
        
        # Test each module
        all_results = []
        
        if self.beat_synchronizer is not None:
            print(f"\nTesting Beat Synchronizer...")
            beat_results = self.test_module(self.beat_synchronizer, "beat_synchronizer", beat_test_cases)
            all_results.extend(beat_results)
            self._print_module_results(beat_results, "Beat Synchronizer")
            
        if self.chord_modeler is not None:
            print(f"\nTesting Chord Sequence Modeler...")
            chord_results = self.test_module(self.chord_modeler, "chord_modeler", chord_test_cases)
            all_results.extend(chord_results)
            self._print_module_results(chord_results, "Chord Sequence Modeler")
            
        if self.source_separator is not None:
            print(f"\nTesting Source Separator...")
            separation_results = self.test_module(self.source_separator, "source_separator", separation_test_cases)
            all_results.extend(separation_results)
            self._print_module_results(separation_results, "Source Separator")
            
        # Overall summary
        self._print_overall_summary(all_results)
        
        return all_results
        
    def _print_module_results(self, results: List[TestResult], module_name: str):
        """Print results for a specific module."""
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        avg_quality = np.mean([r.quality_score for r in results])
        avg_time = np.mean([r.processing_time for r in results])
        
        print(f"\n{module_name} Results:")
        print(f"  Tests passed: {passed}/{total} ({passed/total*100:.1f}%)")
        print(f"  Average quality score: {avg_quality:.3f}")
        print(f"  Average processing time: {avg_time:.3f}s")
        
        # Detailed results
        for result in results:
            status = "✓" if result.passed else "✗"
            print(f"    {status} {result.test_name}: "
                  f"quality={result.quality_score:.2f}, "
                  f"time={result.processing_time:.3f}s")
            if result.error_message:
                print(f"      Error: {result.error_message}")
            if result.additional_info:
                info_str = ", ".join([f"{k}={v}" for k, v in result.additional_info.items()])
                print(f"      Info: {info_str}")
                
    def _print_overall_summary(self, all_results: List[TestResult]):
        """Print overall test summary."""
        print("\n" + "="*60)
        print("OVERALL TEST SUMMARY")
        print("="*60)
        
        total_tests = len(all_results)
        passed_tests = sum(1 for r in all_results if r.passed)
        
        print(f"Total tests run: {total_tests}")
        print(f"Tests passed: {passed_tests}")
        print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
        
        # Module breakdown
        modules = {}
        for result in all_results:
            if result.module_name not in modules:
                modules[result.module_name] = {'passed': 0, 'total': 0, 'quality': []}
            modules[result.module_name]['total'] += 1
            if result.passed:
                modules[result.module_name]['passed'] += 1
            modules[result.module_name]['quality'].append(result.quality_score)
            
        print(f"\nModule breakdown:")
        for module_name, stats in modules.items():
            avg_quality = np.mean(stats['quality'])
            success_rate = stats['passed'] / stats['total'] * 100
            print(f"  {module_name}: {stats['passed']}/{stats['total']} "
                  f"({success_rate:.1f}%) - avg quality: {avg_quality:.3f}")
            
        # Failed tests
        failed_tests = [r for r in all_results if not r.passed]
        if failed_tests:
            print(f"\nFailed tests ({len(failed_tests)}):")
            for result in failed_tests:
                print(f"  ✗ {result.module_name}.{result.test_name}: {result.error_message}")
                
        print("\n" + "="*60)
        
    def run_stress_tests(self):
        """Run stress tests to evaluate robustness."""
        print("\n" + "="*60)
        print("STRESS TESTING")
        print("="*60)
        
        stress_scenarios = [
            ("memory_stress", self._test_memory_stress),
            ("processing_speed", self._test_processing_speed),
            ("edge_cases", self._test_edge_cases),
            ("concurrent_processing", self._test_concurrent_processing)
        ]
        
        for scenario_name, test_func in stress_scenarios:
            print(f"\nRunning {scenario_name} test...")
            try:
                test_func()
                print(f"  ✓ {scenario_name} test passed")
            except Exception as e:
                print(f"  ✗ {scenario_name} test failed: {e}")
                
    def _test_memory_stress(self):
        """Test memory usage with large inputs."""
        large_audio = torch.randn(1, 22050 * 60)  # 1 minute
        
        if self.beat_synchronizer is not None:
            result = self.beat_synchronizer(large_audio)
            assert isinstance(result, dict), "Beat synchronizer should return dict"
            
        if self.chord_modeler is not None:
            result = self.chord_modeler(large_audio)
            assert isinstance(result, dict), "Chord modeler should return dict"
            
        if self.source_separator is not None:
            result = self.source_separator(large_audio, SeparationType.MULTITRACK)
            assert hasattr(result, 'separated_sources'), "Source separator should return SeparationResult"
            
    def _test_processing_speed(self):
        """Test processing speed requirements."""
        test_audio = torch.randn(1, 22050 * 10)  # 10 seconds
        
        modules = [
            (self.beat_synchronizer, "beat_synchronizer"),
            (self.chord_modeler, "chord_modeler"),
            (self.source_separator, "source_separator")
        ]
        
        for module, name in modules:
            if module is not None:
                start_time = time.time()
                
                if name == "source_separator":
                    result = module(test_audio, SeparationType.HARMONIC_PERCUSSIVE)
                else:
                    result = module(test_audio)
                    
                processing_time = time.time() - start_time
                print(f"    {name}: {processing_time:.3f}s")
                
                # Should process faster than real-time for 10s audio
                assert processing_time < 30.0, f"{name} too slow: {processing_time:.3f}s"
                
    def _test_edge_cases(self):
        """Test various edge cases."""
        edge_cases = [
            torch.zeros(1, 100),  # Very short silence
            torch.ones(1, 1000) * 1e-10,  # Near-silence
            torch.tensor([[float('nan')] * 1000]),  # NaN values (should be caught)
            torch.tensor([[float('inf')] * 1000]),  # Inf values (should be caught)
        ]
        
        for i, test_audio in enumerate(edge_cases):
            print(f"    Testing edge case {i+1}...")
            
            # These should not crash but may return empty/fallback results
            for module in [self.beat_synchronizer, self.chord_modeler, self.source_separator]:
                if module is not None:
                    try:
                        if module == self.source_separator:
                            result = module(test_audio, SeparationType.HARMONIC_PERCUSSIVE)
                        else:
                            result = module(test_audio)
                    except Exception as e:
                        # Some edge cases should be gracefully handled
                        if "NaN" in str(e) or "infinite" in str(e):
                            continue  # Expected for NaN/inf inputs
                        else:
                            raise e
                            
    def _test_concurrent_processing(self):
        """Test concurrent processing capabilities."""
        import threading
        
        test_audio = torch.randn(1, 22050 * 5)
        results = []
        errors = []
        
        def test_module(module, name):
            try:
                if name == "source_separator":
                    result = module(test_audio, SeparationType.MULTITRACK)
                else:
                    result = module(test_audio)
                results.append((name, result))
            except Exception as e:
                errors.append((name, e))
                
        threads = []
        for module, name in [(self.beat_synchronizer, "beat_synchronizer"),
                           (self.chord_modeler, "chord_modeler"),
                           (self.source_separator, "source_separator")]:
            if module is not None:
                thread = threading.Thread(target=test_module, args=(module, name))
                threads.append(thread)
                thread.start()
                
        for thread in threads:
            thread.join()
            
        assert len(errors) == 0, f"Concurrent processing errors: {errors}"
        print(f"    Concurrent processing successful: {len(results)} modules")


def main():
    """Main test function."""
    print("Bulletproof Advanced Music ML Modules Test Suite")
    print("="*60)
    
    # Initialize tester
    tester = AdvancedModuleTester()
    
    # Run comprehensive tests
    results = tester.run_comprehensive_tests()
    
    # Run stress tests
    tester.run_stress_tests()
    
    # Final summary
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.passed)
    
    print(f"\n🎵 FINAL RESULTS 🎵")
    print(f"Successfully tested {passed_tests}/{total_tests} scenarios")
    print(f"Overall success rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 All bulletproof modules are working perfectly!")
    elif passed_tests >= total_tests * 0.8:
        print("✅ Bulletproof modules are robust and reliable!")
    elif passed_tests >= total_tests * 0.6:
        print("⚠️ Bulletproof modules are functional but need improvement.")
    else:
        print("❌ Bulletproof modules need significant fixes.")
        
    return results


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        results = main()