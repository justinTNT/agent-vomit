#!/usr/bin/env python3
"""
Focused tests for music-specific advanced modules:
- Beat Synchronizer
- Chord Sequence Modeler  
- Source Separation System

Tests both functionality and musical accuracy.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import torchaudio
import numpy as np
import librosa
from typing import Dict, List, Tuple

def create_synthetic_music(duration=5, sample_rate=22050):
    """Create synthetic music with known musical properties."""
    t = torch.linspace(0, duration, sample_rate * duration)
    
    # Chord progression: C - Am - F - G (common pop progression)
    chord_duration = duration / 4
    chords = [
        [261.63, 329.63, 392.00],  # C major (C, E, G)
        [220.00, 261.63, 329.63],  # A minor (A, C, E)  
        [174.61, 220.00, 261.63],  # F major (F, A, C)
        [196.00, 246.94, 293.66]   # G major (G, B, D)
    ]
    
    harmonic = torch.zeros_like(t)
    for i, chord_freqs in enumerate(chords):
        start_time = i * chord_duration
        end_time = (i + 1) * chord_duration
        
        start_idx = int(start_time * sample_rate)
        end_idx = int(end_time * sample_rate)
        
        # Generate chord
        chord_audio = torch.zeros(end_idx - start_idx)
        for freq in chord_freqs:
            chord_audio += 0.3 * torch.sin(2 * torch.pi * freq * t[start_idx:end_idx])
        
        harmonic[start_idx:end_idx] = chord_audio
    
    # Rhythmic component (120 BPM = 0.5s per beat)
    beat_interval = 0.5
    percussive = torch.zeros_like(t)
    
    for beat_time in torch.arange(0, duration, beat_interval):
        beat_idx = int(beat_time * sample_rate)
        if beat_idx < len(percussive) - 2000:
            # Kick drum on beats 1 and 3
            if int(beat_time / beat_interval) % 2 == 0:
                kick_duration = 1000
                kick = torch.sin(2 * torch.pi * 60 * torch.linspace(0, 0.1, kick_duration))
                kick *= torch.exp(-8 * torch.linspace(0, 0.1, kick_duration))
                percussive[beat_idx:beat_idx + kick_duration] += kick * 0.8
            
            # Hi-hat on off-beats
            hihat_start = beat_idx + int(beat_interval * sample_rate // 2)
            if hihat_start < len(percussive) - 500:
                hihat_duration = 500
                hihat = torch.randn(hihat_duration) * 0.3
                hihat *= torch.exp(-20 * torch.linspace(0, 0.05, hihat_duration))
                percussive[hihat_start:hihat_start + hihat_duration] += hihat
    
    # Combine components
    mixed = harmonic + percussive
    
    return mixed.unsqueeze(0), harmonic.unsqueeze(0), percussive.unsqueeze(0)


def test_beat_synchronizer():
    """Test beat synchronizer with known rhythm."""
    print("🥁 Testing Beat Synchronizer")
    
    try:
        from modules.advanced_modules import create_beat_synchronizer
        
        # Create test audio with known properties
        mixed_audio, _, _ = create_synthetic_music(duration=8, sample_rate=22050)
        
        # Expected: 120 BPM, beats every 0.5 seconds
        expected_tempo = 120.0
        expected_beat_times = torch.arange(0, 8, 0.5)
        
        # Test beat synchronizer
        beat_sync = create_beat_synchronizer()
        results = beat_sync(mixed_audio)
        
        if isinstance(results, list) and len(results) > 0:
            result = results[0]
            
            # Check tempo detection
            detected_tempo = result['tempo'].item() if torch.is_tensor(result['tempo']) else result['tempo']
            tempo_error = abs(detected_tempo - expected_tempo) / expected_tempo
            
            print(f"  Expected tempo: {expected_tempo} BPM")
            print(f"  Detected tempo: {detected_tempo:.1f} BPM") 
            print(f"  Tempo error: {tempo_error:.1%}")
            
            tempo_ok = tempo_error < 0.2  # Within 20%
            print(f"  ✅ Tempo detection: {'PASS' if tempo_ok else 'FAIL'}")
            
            # Check beat detection
            beat_times = result.get('beat_times', torch.tensor([]))
            if len(beat_times) > 0:
                # Compare with expected beats (allow some tolerance)
                beat_errors = []
                for expected_beat in expected_beat_times:
                    if len(beat_times) > 0:
                        closest_beat = beat_times[torch.argmin(torch.abs(beat_times - expected_beat))]
                        error = abs(closest_beat - expected_beat).item()
                        beat_errors.append(error)
                
                if beat_errors:
                    avg_beat_error = np.mean(beat_errors)
                    print(f"  Average beat error: {avg_beat_error:.3f} seconds")
                    
                    beat_ok = avg_beat_error < 0.1  # Within 100ms
                    print(f"  ✅ Beat detection: {'PASS' if beat_ok else 'FAIL'}")
                else:
                    print(f"  ❌ Beat detection: FAIL (no beats to compare)")
                    beat_ok = False
            else:
                print(f"  ❌ Beat detection: FAIL (no beats detected)")
                beat_ok = False
                
            # Test rhythm features
            rhythm_features = result.get('rhythm_features', {})
            if rhythm_features:
                consistency = rhythm_features.get('beat_consistency', 0)
                syncopation = rhythm_features.get('syncopation', 0)
                
                print(f"  Beat consistency: {consistency:.3f}")
                print(f"  Syncopation level: {syncopation:.3f}")
                
                features_ok = consistency > 0.7  # Should be fairly consistent
                print(f"  ✅ Rhythm features: {'PASS' if features_ok else 'FAIL'}")
            else:
                print(f"  ❌ Rhythm features: FAIL (no features extracted)")
                features_ok = False
            
            overall_pass = tempo_ok and beat_ok and features_ok
            print(f"  🎯 Overall: {'PASS' if overall_pass else 'FAIL'}")
            
            return overall_pass
        else:
            print("  ❌ Invalid output format")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_chord_sequence_modeler():
    """Test chord sequence modeler with known chord progression."""
    print("\n🎼 Testing Chord Sequence Modeler")
    
    try:
        from modules.advanced_modules import create_chord_sequence_modeler
        
        # Create test audio with known chord progression: C - Am - F - G
        mixed_audio, harmonic_audio, _ = create_synthetic_music(duration=8, sample_rate=22050)
        
        expected_chords = ['C', 'Am', 'F', 'G']  # Simplified chord names
        expected_key = 'C'  # Key of C major
        
        # Test chord modeler
        chord_modeler = create_chord_sequence_modeler()
        results = chord_modeler(harmonic_audio)  # Use harmonic component for cleaner results
        
        if isinstance(results, list) and len(results) > 0:
            result = results[0]
            
            # Check key detection
            key_root = result.get('key_root', -1)
            is_major = result.get('is_major', False)
            
            # Convert key root to note name
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            detected_key = note_names[key_root] if 0 <= key_root < 12 else 'Unknown'
            detected_mode = 'major' if is_major else 'minor'
            
            print(f"  Expected key: {expected_key} major")
            print(f"  Detected key: {detected_key} {detected_mode}")
            
            key_ok = detected_key == expected_key and is_major
            print(f"  ✅ Key detection: {'PASS' if key_ok else 'FAIL'}")
            
            # Check chord sequence
            chord_symbols = result.get('chord_symbols', [])
            print(f"  Detected chords: {chord_symbols}")
            
            if chord_symbols:
                # Count how many expected chords were detected
                detected_count = 0
                for expected_chord in expected_chords:
                    for detected_chord in chord_symbols:
                        if expected_chord in detected_chord:
                            detected_count += 1
                            break
                
                chord_accuracy = detected_count / len(expected_chords)
                print(f"  Chord accuracy: {chord_accuracy:.1%}")
                
                chord_ok = chord_accuracy >= 0.5  # At least 50% correct
                print(f"  ✅ Chord detection: {'PASS' if chord_ok else 'FAIL'}")
            else:
                print(f"  ❌ Chord detection: FAIL (no chords detected)")
                chord_ok = False
            
            # Check harmonic functions
            harmonic_functions = result.get('harmonic_functions', [])
            if harmonic_functions:
                print(f"  Harmonic functions: {harmonic_functions}")
                
                # Should contain typical functions for C major
                expected_functions = ['I', 'vi', 'IV', 'V']
                function_matches = sum(1 for ef in expected_functions 
                                     if any(ef in hf for hf in harmonic_functions))
                
                function_accuracy = function_matches / len(expected_functions)
                function_ok = function_accuracy >= 0.5
                print(f"  ✅ Harmonic analysis: {'PASS' if function_ok else 'FAIL'}")
            else:
                print(f"  ❌ Harmonic analysis: FAIL (no functions detected)")
                function_ok = False
            
            overall_pass = key_ok and chord_ok and function_ok
            print(f"  🎯 Overall: {'PASS' if overall_pass else 'FAIL'}")
            
            return overall_pass
        else:
            print("  ❌ Invalid output format")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_source_separation():
    """Test source separation with known harmonic/percussive content."""
    print("\n🔀 Testing Source Separation")
    
    try:
        from modules.advanced_modules import create_source_separator
        from modules.advanced_modules.source_separation import SeparationType
        
        # Create test audio with separate harmonic and percussive components
        mixed_audio, harmonic_audio, percussive_audio = create_synthetic_music(duration=5, sample_rate=22050)
        
        separator = create_source_separator()
        
        # Test harmonic-percussive separation
        separation_result = separator(mixed_audio, SeparationType.HARMONIC_PERCUSSIVE)
        
        if hasattr(separation_result, 'separated_sources'):
            separated_sources = separation_result.separated_sources
            
            print(f"  Separated sources: {list(separated_sources.keys())}")
            
            if 'harmonic' in separated_sources and 'percussive' in separated_sources:
                harmonic_sep = separated_sources['harmonic']
                percussive_sep = separated_sources['percussive']
                
                # Compare separated components with originals using correlation
                # Flatten for correlation calculation
                harmonic_orig_flat = harmonic_audio.flatten()
                percussive_orig_flat = percussive_audio.flatten()
                harmonic_sep_flat = harmonic_sep.flatten()
                percussive_sep_flat = percussive_sep.flatten()
                
                # Normalize for correlation
                def normalize(x):
                    return (x - x.mean()) / (x.std() + 1e-8)
                
                harmonic_orig_norm = normalize(harmonic_orig_flat)
                percussive_orig_norm = normalize(percussive_orig_flat)
                harmonic_sep_norm = normalize(harmonic_sep_flat)
                percussive_sep_norm = normalize(percussive_sep_flat)
                
                # Compute correlations
                harmonic_corr = torch.corrcoef(torch.stack([harmonic_orig_norm, harmonic_sep_norm]))[0, 1]
                percussive_corr = torch.corrcoef(torch.stack([percussive_orig_norm, percussive_sep_norm]))[0, 1]
                
                print(f"  Harmonic correlation: {harmonic_corr:.3f}")
                print(f"  Percussive correlation: {percussive_corr:.3f}")
                
                # Check separation quality
                harmonic_ok = harmonic_corr > 0.3  # Moderate correlation expected
                percussive_ok = percussive_corr > 0.3
                
                print(f"  ✅ Harmonic separation: {'PASS' if harmonic_ok else 'FAIL'}")
                print(f"  ✅ Percussive separation: {'PASS' if percussive_ok else 'FAIL'}")
                
                # Check reconstruction quality
                reconstructed = harmonic_sep + percussive_sep
                reconstruction_error = torch.nn.functional.mse_loss(reconstructed, mixed_audio).item()
                
                print(f"  Reconstruction MSE: {reconstruction_error:.6f}")
                
                reconstruction_ok = reconstruction_error < 0.1  # Low reconstruction error
                print(f"  ✅ Reconstruction: {'PASS' if reconstruction_ok else 'FAIL'}")
                
                # Check confidence scores
                confidence_scores = separation_result.confidence_scores
                if confidence_scores:
                    harmonic_conf = confidence_scores.get('harmonic', 0)
                    percussive_conf = confidence_scores.get('percussive', 0)
                    
                    print(f"  Harmonic confidence: {harmonic_conf:.3f}")
                    print(f"  Percussive confidence: {percussive_conf:.3f}")
                    
                    confidence_ok = harmonic_conf > 0.5 and percussive_conf > 0.5
                    print(f"  ✅ Confidence scores: {'PASS' if confidence_ok else 'FAIL'}")
                else:
                    confidence_ok = False
                    print(f"  ❌ Confidence scores: FAIL (no scores available)")
                
                overall_pass = harmonic_ok and percussive_ok and reconstruction_ok and confidence_ok
                print(f"  🎯 Overall: {'PASS' if overall_pass else 'FAIL'}")
                
                return overall_pass
            else:
                print("  ❌ Missing expected separation sources")
                return False
        else:
            print("  ❌ Invalid separation result format")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_integration():
    """Test integration between music modules."""
    print("\n🔗 Testing Module Integration")
    
    try:
        # Create consistent test audio
        mixed_audio, _, _ = create_synthetic_music(duration=6, sample_rate=22050)
        
        # Import all modules
        from modules.advanced_modules import (
            create_beat_synchronizer,
            create_chord_sequence_modeler, 
            create_source_separator
        )
        from modules.advanced_modules.source_separation import SeparationType
        
        # Test workflow: separation -> analysis
        separator = create_source_separator()
        beat_sync = create_beat_synchronizer()
        chord_modeler = create_chord_sequence_modeler()
        
        print("  Step 1: Source separation")
        sep_result = separator(mixed_audio, SeparationType.HARMONIC_PERCUSSIVE)
        
        if hasattr(sep_result, 'separated_sources'):
            harmonic_component = sep_result.separated_sources['harmonic']
            percussive_component = sep_result.separated_sources['percussive']
            
            print("  Step 2: Beat analysis on percussive component")
            beat_result = beat_sync(percussive_component)
            
            print("  Step 3: Chord analysis on harmonic component")
            chord_result = chord_modeler(harmonic_component)
            
            # Check results are coherent
            if (isinstance(beat_result, list) and len(beat_result) > 0 and
                isinstance(chord_result, list) and len(chord_result) > 0):
                
                beat_data = beat_result[0]
                chord_data = chord_result[0]
                
                # Extract key metrics
                tempo = beat_data.get('tempo', 0)
                chord_count = len(chord_data.get('chord_symbols', []))
                
                print(f"  Detected tempo: {tempo:.1f} BPM")
                print(f"  Detected chords: {chord_count}")
                
                # Check if results are reasonable
                tempo_ok = 60 <= tempo <= 200
                chord_ok = chord_count > 0
                
                integration_ok = tempo_ok and chord_ok
                print(f"  ✅ Integration test: {'PASS' if integration_ok else 'FAIL'}")
                
                return integration_ok
            else:
                print("  ❌ Invalid analysis results")
                return False
        else:
            print("  ❌ Separation failed")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    """Run all music module tests."""
    print("🎵 Music Modules Test Suite")
    print("=" * 50)
    
    results = {}
    
    # Run individual module tests
    results['beat_synchronizer'] = test_beat_synchronizer()
    results['chord_modeler'] = test_chord_sequence_modeler()
    results['source_separation'] = test_source_separation()
    results['integration'] = test_integration()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 MUSIC MODULES TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    success_rate = (passed / total) * 100
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate >= 75:
        print("🎉 Music modules test suite PASSED!")
        return 0
    else:
        print("💥 Music modules test suite FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())