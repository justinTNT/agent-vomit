#!/usr/bin/env python3
"""
Integration tests for advanced audio modules ecosystem.

Tests module interactions, configuration compatibility, and real-world workflows.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn.functional as F
import numpy as np
import warnings
from typing import Dict, List, Tuple, Any

def create_realistic_audio_content(duration=10, sample_rate=22050):
    """Create realistic mixed audio content for comprehensive testing."""
    t = torch.linspace(0, duration, sample_rate * duration)
    
    # Musical composition: intro + verse + chorus structure
    section_duration = duration / 3
    
    # Intro (ambient pad)
    intro_end = int(section_duration * sample_rate)
    intro = 0.2 * torch.sin(2 * torch.pi * 55 * t[:intro_end])  # Low A
    intro += 0.15 * torch.sin(2 * torch.pi * 110 * t[:intro_end])  # A
    intro *= torch.linspace(0, 1, intro_end)  # Fade in
    
    # Verse (chord progression with rhythm)
    verse_start = intro_end
    verse_end = int(2 * section_duration * sample_rate)
    verse_t = t[verse_start:verse_end]
    
    # Chord progression: Am - F - C - G
    chord_duration = len(verse_t) / 4
    verse = torch.zeros_like(verse_t)
    
    chords = [
        [220, 261.63, 329.63],  # Am
        [174.61, 220, 261.63],  # F  
        [261.63, 329.63, 392],  # C
        [196, 246.94, 293.66]   # G
    ]
    
    for i, chord_freqs in enumerate(chords):
        start_idx = int(i * chord_duration)
        end_idx = int((i + 1) * chord_duration)
        
        for freq in chord_freqs:
            verse[start_idx:end_idx] += 0.2 * torch.sin(2 * torch.pi * freq * verse_t[start_idx:end_idx])
    
    # Add rhythm section to verse
    beat_interval = 0.5  # 120 BPM
    for beat_time in torch.arange(0, len(verse_t) / sample_rate, beat_interval):
        beat_idx = int(beat_time * sample_rate)
        if beat_idx < len(verse) - 1000:
            # Kick on 1 and 3
            if int(beat_time / beat_interval) % 2 == 0:
                kick = 0.4 * torch.exp(-10 * torch.linspace(0, 0.1, 1000)) * torch.sin(2 * torch.pi * 60 * torch.linspace(0, 0.1, 1000))
                verse[beat_idx:beat_idx + 1000] += kick
            
            # Snare on 2 and 4
            snare_beat = beat_idx + int(beat_interval * sample_rate / 2)
            if snare_beat < len(verse) - 500:
                snare = 0.3 * torch.randn(500) * torch.exp(-15 * torch.linspace(0, 0.05, 500))
                verse[snare_beat:snare_beat + 500] += snare
    
    # Chorus (more energetic, add lead)
    chorus_start = verse_end
    chorus_t = t[chorus_start:]
    
    # Base from verse but louder
    chorus_base = verse[-len(chorus_t):] * 1.3
    
    # Add lead melody
    melody_freqs = [523.25, 587.33, 659.25, 698.46, 783.99]  # C5 major pentatonic
    melody = torch.zeros_like(chorus_t)
    
    note_duration = len(chorus_t) / len(melody_freqs)
    for i, freq in enumerate(melody_freqs):
        start_idx = int(i * note_duration)
        end_idx = int((i + 1) * note_duration)
        envelope = torch.exp(-3 * torch.linspace(0, 1, end_idx - start_idx))
        melody[start_idx:end_idx] = 0.3 * torch.sin(2 * torch.pi * freq * chorus_t[start_idx:end_idx]) * envelope
    
    chorus = chorus_base + melody
    
    # Combine sections
    full_audio = torch.cat([intro, verse, chorus])
    
    # Add some realistic audio artifacts
    # Slight DC offset
    full_audio += 0.001
    
    # Gentle compression
    full_audio = torch.tanh(full_audio * 2) / 2
    
    # Normalize
    full_audio = full_audio / (torch.max(torch.abs(full_audio)) + 1e-8) * 0.9
    
    return full_audio.unsqueeze(0)  # Add batch dimension


def test_end_to_end_music_analysis():
    """Test complete music analysis pipeline."""
    print("🎼 Testing End-to-End Music Analysis Pipeline")
    
    try:
        # Import all required modules
        from modules.advanced_modules import (
            create_beat_synchronizer,
            create_chord_sequence_modeler,
            create_source_separator
        )
        from modules.advanced_modules.source_separation import SeparationType
        from modules.audio_analysis.audio_config import get_music_config
        
        # Create consistent configuration
        config = get_music_config()
        print(f"  Using config: {config.sample_rate}Hz, {config.n_fft}pt FFT")
        
        # Create models with shared config
        beat_sync = create_beat_synchronizer(config)
        chord_modeler = create_chord_sequence_modeler(config)
        separator = create_source_separator(config)
        
        # Create realistic test audio
        audio = create_realistic_audio_content(duration=12, sample_rate=config.sample_rate)
        print(f"  Test audio: {audio.shape[1]/config.sample_rate:.1f}s at {config.sample_rate}Hz")
        
        # Step 1: Source separation
        print("  Step 1: Source separation")
        sep_result = separator(audio, SeparationType.HARMONIC_PERCUSSIVE)
        
        if not hasattr(sep_result, 'separated_sources'):
            print("  ❌ Source separation failed")
            return False
        
        harmonic = sep_result.separated_sources['harmonic']
        percussive = sep_result.separated_sources['percussive']
        separation_quality = sep_result.separation_quality
        
        print(f"    Separation quality: {separation_quality:.3f}")
        
        # Step 2: Rhythm analysis on percussive component
        print("  Step 2: Rhythm analysis")
        rhythm_result = beat_sync(percussive)
        
        if not isinstance(rhythm_result, list) or len(rhythm_result) == 0:
            print("  ❌ Rhythm analysis failed")
            return False
        
        rhythm_data = rhythm_result[0]
        tempo = rhythm_data.get('tempo', 0)
        beat_times = rhythm_data.get('beat_times', torch.tensor([]))
        
        print(f"    Detected tempo: {tempo:.1f} BPM")
        print(f"    Number of beats: {len(beat_times)}")
        
        # Step 3: Harmonic analysis on harmonic component
        print("  Step 3: Harmonic analysis")
        chord_result = chord_modeler(harmonic)
        
        if not isinstance(chord_result, list) or len(chord_result) == 0:
            print("  ❌ Harmonic analysis failed")
            return False
        
        chord_data = chord_result[0]
        chord_symbols = chord_data.get('chord_symbols', [])
        key_root = chord_data.get('key_root', -1)
        is_major = chord_data.get('is_major', False)
        
        print(f"    Detected chords: {chord_symbols}")
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key_name = note_names[key_root] if 0 <= key_root < 12 else 'Unknown'
        print(f"    Detected key: {key_name} {'major' if is_major else 'minor'}")
        
        # Step 4: Validate results coherence
        print("  Step 4: Results validation")
        
        # Tempo should be reasonable
        tempo_ok = 60 <= tempo <= 200
        print(f"    Tempo range: {'✅' if tempo_ok else '❌'} ({tempo:.1f} BPM)")
        
        # Should detect multiple chords
        chord_count_ok = len(chord_symbols) >= 2
        print(f"    Chord variety: {'✅' if chord_count_ok else '❌'} ({len(chord_symbols)} chords)")
        
        # Beat timing should be consistent
        if len(beat_times) > 2:
            beat_intervals = torch.diff(beat_times)
            beat_consistency = 1.0 / (1.0 + torch.std(beat_intervals) / torch.mean(beat_intervals))
            beat_consistent_ok = beat_consistency > 0.5
            print(f"    Beat consistency: {'✅' if beat_consistent_ok else '❌'} ({beat_consistency:.3f})")
        else:
            beat_consistent_ok = False
            print(f"    Beat consistency: ❌ (too few beats)")
        
        # Separation quality should be reasonable
        sep_quality_ok = separation_quality > 0.3
        print(f"    Separation quality: {'✅' if sep_quality_ok else '❌'} ({separation_quality:.3f})")
        
        overall_pass = tempo_ok and chord_count_ok and beat_consistent_ok and sep_quality_ok
        print(f"  🎯 Pipeline result: {'PASS' if overall_pass else 'FAIL'}")
        
        return overall_pass
        
    except Exception as e:
        print(f"  ❌ Pipeline error: {e}")
        return False


def test_foundation_model_workflow():
    """Test foundation model feature extraction workflow."""
    print("\n🏗️ Testing Foundation Model Workflow")
    
    try:
        from modules.advanced_modules import create_audio_mae, create_data2vec_audio, create_wavlm
        from modules.advanced_modules.foundation_models import FoundationModelFactory, FoundationModelType
        
        # Create speech-like audio for foundation models
        sample_rate = 16000  # Foundation models typically use 16kHz
        duration = 4
        t = torch.linspace(0, duration, sample_rate * duration)
        
        # Create speech-like formant structure
        speech_audio = torch.zeros_like(t)
        formants = [800, 1200, 2400]
        for freq in formants:
            speech_audio += 0.2 * torch.sin(2 * torch.pi * freq * t) * torch.sin(2 * torch.pi * 150 * t)
        
        # Add realistic speech envelope
        envelope = torch.abs(torch.sin(2 * torch.pi * 3 * t)) ** 0.3
        speech_audio = speech_audio * envelope
        speech_audio = speech_audio.unsqueeze(0)
        
        print(f"  Test audio: {speech_audio.shape} at {sample_rate}Hz")
        
        # Test each foundation model
        models = {
            'AudioMAE': create_audio_mae(),
            'Data2Vec': create_data2vec_audio(), 
            'WavLM': create_wavlm()
        }
        
        features = {}
        
        for name, model in models.items():
            print(f"  Testing {name}...")
            
            try:
                model.eval()
                with torch.no_grad():
                    if name == 'AudioMAE':
                        result = model(speech_audio, mask_ratio=0.0, return_loss=False)
                        feature = result['encoded_features']
                    elif name == 'Data2Vec':
                        result = model(speech_audio, mask_prob=0.0, update_teacher=False)
                        feature = result['teacher_features']  # Use stable teacher features
                    else:  # WavLM
                        result = model(speech_audio, mask_prob=0.0)
                        feature = result['contextualized_features']
                    
                    features[name] = feature
                    print(f"    Features: {feature.shape}")
                    
                    # Check feature quality
                    feature_var = feature.var().item()
                    has_nan = torch.isnan(feature).any()
                    has_inf = torch.isinf(feature).any()
                    
                    quality_ok = feature_var > 1e-6 and not has_nan and not has_inf
                    print(f"    Quality: {'✅' if quality_ok else '❌'} (var={feature_var:.6f})")
                    
                    if not quality_ok:
                        return False
                        
            except Exception as e:
                print(f"    ❌ {name} failed: {e}")
                return False
        
        # Test feature compatibility for downstream tasks
        print("  Testing downstream compatibility...")
        
        # All features should be usable for classification
        try:
            num_classes = 10
            
            for name, feature in features.items():
                # Create simple classifier head
                feature_dim = feature.shape[-1]
                classifier = torch.nn.Linear(feature_dim, num_classes)
                
                # Test classification
                if len(feature.shape) == 3:  # [batch, seq, dim]
                    pooled_feature = feature.mean(dim=1)  # Global average pooling
                else:
                    pooled_feature = feature
                
                logits = classifier(pooled_feature)
                
                expected_shape = (speech_audio.shape[0], num_classes)
                shape_ok = logits.shape == expected_shape
                
                print(f"    {name} classifier: {'✅' if shape_ok else '❌'} {logits.shape}")
                
                if not shape_ok:
                    return False
                    
        except Exception as e:
            print(f"    ❌ Downstream test failed: {e}")
            return False
        
        print(f"  🎯 Foundation workflow: PASS")
        return True
        
    except Exception as e:
        print(f"  ❌ Foundation workflow error: {e}")
        return False


def test_multimodal_alignment():
    """Test multimodal model alignment and retrieval."""
    print("\n🎭 Testing Multimodal Alignment")
    
    try:
        from modules.advanced_modules import create_clap, create_imagebind_audio
        
        # Create models
        clap = create_clap()
        imagebind = create_imagebind_audio()
        
        # Create audio content with different characteristics
        sample_rate = 22050
        duration = 3
        
        audio_types = []
        descriptions = []
        
        # Type 1: Piano melody
        t = torch.linspace(0, duration, sample_rate * duration)
        piano = 0.3 * torch.sin(2 * torch.pi * 261.63 * t) + 0.2 * torch.sin(2 * torch.pi * 329.63 * t)
        piano *= torch.exp(-0.5 * t)
        audio_types.append(piano)
        descriptions.append("piano melody")
        
        # Type 2: Drum pattern
        drums = torch.zeros_like(t)
        for beat_time in torch.arange(0, duration, 0.5):
            beat_idx = int(beat_time * sample_rate)
            if beat_idx < len(drums) - 1000:
                drums[beat_idx:beat_idx + 1000] += torch.randn(1000) * torch.exp(-8 * torch.linspace(0, 0.1, 1000))
        audio_types.append(drums)
        descriptions.append("drum beats")
        
        # Type 3: Synthesizer
        synth = 0.4 * torch.sin(2 * torch.pi * 440 * t * (1 + 0.1 * torch.sin(2 * torch.pi * 3 * t)))
        audio_types.append(synth)
        descriptions.append("synthesizer")
        
        # Stack into batch
        audio_batch = torch.stack(audio_types)
        
        # Create dummy text tokens (in real use, these would be proper tokenized descriptions)
        vocab_size = 1000
        seq_length = 8
        batch_size = len(descriptions)
        
        input_ids = torch.randint(1, vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)
        
        print(f"  Test data: {batch_size} audio-text pairs")
        
        # Test CLAP alignment
        print("  Testing CLAP audio-text alignment...")
        
        with torch.no_grad():
            clap_result = clap(audio_batch, input_ids, attention_mask)
            
            audio_emb = clap_result['audio_embeddings']
            text_emb = clap_result['text_embeddings']
            
            print(f"    Audio embeddings: {audio_emb.shape}")
            print(f"    Text embeddings: {text_emb.shape}")
            
            # Compute similarity matrix
            similarities = torch.matmul(audio_emb, text_emb.t())
            print(f"    Similarity matrix:\n{similarities}")
            
            # Diagonal should have highest similarities (matching pairs)
            diagonal = torch.diag(similarities)
            off_diagonal = similarities[~torch.eye(batch_size, dtype=bool)]
            
            avg_diagonal = diagonal.mean()
            avg_off_diagonal = off_diagonal.mean()
            
            print(f"    Diagonal similarity: {avg_diagonal:.3f}")
            print(f"    Off-diagonal similarity: {avg_off_diagonal:.3f}")
            
            # Diagonal should be higher than off-diagonal
            alignment_ok = avg_diagonal > avg_off_diagonal
            print(f"    Alignment quality: {'✅' if alignment_ok else '❌'}")
        
        # Test ImageBind feature consistency
        print("  Testing ImageBind feature consistency...")
        
        with torch.no_grad():
            imagebind_result = imagebind(audio_batch)
            imagebind_features = imagebind_result['final_embedding']
            
            print(f"    ImageBind features: {imagebind_features.shape}")
            
            # Check feature diversity (different audio should give different features)
            pairwise_sims = []
            for i in range(batch_size):
                for j in range(i + 1, batch_size):
                    sim = F.cosine_similarity(imagebind_features[i], imagebind_features[j], dim=0)
                    pairwise_sims.append(sim.item())
            
            avg_similarity = np.mean(pairwise_sims)
            print(f"    Average pairwise similarity: {avg_similarity:.3f}")
            
            # Features should be somewhat different (not too similar)
            diversity_ok = avg_similarity < 0.8
            print(f"    Feature diversity: {'✅' if diversity_ok else '❌'}")
        
        # Test cross-model feature alignment (if same dimensions)
        print("  Testing cross-model alignment...")
        
        if audio_emb.shape[1] == imagebind_features.shape[1]:
            cross_similarities = F.cosine_similarity(audio_emb, imagebind_features, dim=1)
            avg_cross_sim = cross_similarities.mean()
            
            print(f"    Cross-model similarity: {avg_cross_sim:.3f}")
            
            # Should have moderate correlation (same audio content)
            cross_align_ok = 0.1 < avg_cross_sim < 0.9
            print(f"    Cross-model alignment: {'✅' if cross_align_ok else '❌'}")
        else:
            print(f"    Cross-model alignment: ⚠️ SKIP (different dimensions)")
            cross_align_ok = True
        
        overall_pass = alignment_ok and diversity_ok and cross_align_ok
        print(f"  🎯 Multimodal alignment: {'PASS' if overall_pass else 'FAIL'}")
        
        return overall_pass
        
    except Exception as e:
        print(f"  ❌ Multimodal alignment error: {e}")
        return False


def test_configuration_consistency():
    """Test configuration system consistency across all modules."""
    print("\n⚙️ Testing Configuration Consistency")
    
    try:
        from modules.audio_analysis.audio_config import (
            get_music_config, get_speech_config, get_timbralgebraics_config
        )
        from modules.advanced_modules import (
            create_beat_synchronizer, create_chord_sequence_modeler, create_source_separator,
            create_audio_mae, create_data2vec_audio, create_wavlm,
            create_clap, create_imagebind_audio
        )
        
        configs = {
            'music': get_music_config(),
            'speech': get_speech_config(),
            'timbralgebraics': get_timbralgebraics_config()
        }
        
        # Test each config with compatible modules
        results = {}
        
        for config_name, config in configs.items():
            print(f"  Testing {config_name} config ({config.sample_rate}Hz)...")
            
            config_results = {}
            
            # Music-specific modules (work with all configs)
            if config_name in ['music', 'timbralgebraics']:  # Skip speech for music modules
                try:
                    beat_sync = create_beat_synchronizer(config)
                    chord_model = create_chord_sequence_modeler(config)
                    separator = create_source_separator(config)
                    
                    config_results['music_modules'] = True
                    print(f"    Music modules: ✅")
                except Exception as e:
                    config_results['music_modules'] = False
                    print(f"    Music modules: ❌ ({e})")
            else:
                config_results['music_modules'] = True  # Skip for speech
            
            # Foundation models (prefer 16kHz but should adapt)
            try:
                # Create foundation config based on current config
                from modules.advanced_modules.foundation_models import FoundationModelConfig, FoundationModelType
                
                foundation_config = FoundationModelConfig(
                    model_type=FoundationModelType.AUDIO_MAE,
                    sample_rate=16000,  # Foundation models use 16kHz
                    n_fft=config.n_fft,
                    hop_length=config.hop_length,
                    n_mels=config.n_mels
                )
                
                audio_mae = create_audio_mae(foundation_config)
                config_results['foundation_models'] = True
                print(f"    Foundation models: ✅")
                
            except Exception as e:
                config_results['foundation_models'] = False
                print(f"    Foundation models: ❌ ({e})")
            
            # Multimodal models
            try:
                from modules.advanced_modules.multimodal_models import MultimodalConfig, MultimodalModelType
                
                multimodal_config = MultimodalConfig(
                    model_type=MultimodalModelType.CLAP,
                    sample_rate=config.sample_rate,
                    n_fft=config.n_fft,
                    hop_length=config.hop_length,
                    n_mels=config.n_mels
                )
                
                clap = create_clap(multimodal_config)
                config_results['multimodal_models'] = True
                print(f"    Multimodal models: ✅")
                
            except Exception as e:
                config_results['multimodal_models'] = False
                print(f"    Multimodal models: ❌ ({e})")
            
            results[config_name] = config_results
        
        # Check overall consistency
        all_passed = all(
            all(test_results.values()) 
            for test_results in results.values()
        )
        
        print(f"  🎯 Configuration consistency: {'PASS' if all_passed else 'FAIL'}")
        
        # Print summary
        for config_name, test_results in results.items():
            passed = sum(test_results.values())
            total = len(test_results)
            print(f"    {config_name}: {passed}/{total}")
        
        return all_passed
        
    except Exception as e:
        print(f"  ❌ Configuration consistency error: {e}")
        return False


def test_memory_and_performance():
    """Test memory usage and performance characteristics."""
    print("\n⚡ Testing Memory and Performance")
    
    try:
        import time
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Test with different audio lengths
        durations = [1, 5, 10]  # seconds
        sample_rate = 22050
        
        from modules.advanced_modules import create_beat_synchronizer
        
        model = create_beat_synchronizer()
        
        performance_data = []
        
        for duration in durations:
            print(f"  Testing {duration}s audio...")
            
            # Create test audio
            audio = torch.randn(1, sample_rate * duration)
            
            # Measure memory before
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Time the forward pass
            start_time = time.time()
            
            with torch.no_grad():
                result = model(audio)
            
            end_time = time.time()
            
            # Measure memory after
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            
            processing_time = end_time - start_time
            memory_used = mem_after - mem_before
            
            performance_data.append({
                'duration': duration,
                'processing_time': processing_time,
                'memory_used': memory_used,
                'real_time_factor': processing_time / duration
            })
            
            print(f"    Processing time: {processing_time:.3f}s")
            print(f"    Memory used: {memory_used:.1f}MB")
            print(f"    Real-time factor: {processing_time/duration:.2f}x")
        
        # Check performance criteria
        # Should process faster than real-time for shorter audio
        fast_enough = any(data['real_time_factor'] < 2.0 for data in performance_data)
        
        # Memory usage should be reasonable (< 500MB for test)
        memory_ok = all(data['memory_used'] < 500 for data in performance_data)
        
        # Processing time should scale reasonably with duration
        times = [data['processing_time'] for data in performance_data]
        durations_list = [data['duration'] for data in performance_data]
        
        # Check if processing time increases with duration (should be roughly linear)
        if len(times) > 1:
            scaling_ok = times[-1] > times[0]  # Longer audio takes more time
        else:
            scaling_ok = True
        
        print(f"    Performance: {'✅' if fast_enough else '❌'}")
        print(f"    Memory usage: {'✅' if memory_ok else '❌'}")
        print(f"    Scaling: {'✅' if scaling_ok else '❌'}")
        
        overall_perf = fast_enough and memory_ok and scaling_ok
        print(f"  🎯 Performance: {'PASS' if overall_perf else 'FAIL'}")
        
        return overall_perf
        
    except Exception as e:
        print(f"  ❌ Performance test error: {e}")
        return False


def main():
    """Run comprehensive integration tests."""
    print("🔬 Advanced Modules Integration Test Suite")
    print("=" * 60)
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    
    results = {}
    
    # Run integration tests
    results['music_analysis_pipeline'] = test_end_to_end_music_analysis()
    results['foundation_workflow'] = test_foundation_model_workflow() 
    results['multimodal_alignment'] = test_multimodal_alignment()
    results['configuration_consistency'] = test_configuration_consistency()
    results['performance'] = test_memory_and_performance()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    success_rate = (passed / total) * 100
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall Integration: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 Advanced modules integration PASSED!")
        print("\n✨ The advanced modules ecosystem is ready for production use!")
        return 0
    else:
        print("💥 Advanced modules integration FAILED")
        print("\n⚠️  Some integration issues need to be resolved.")
        return 1


if __name__ == "__main__":
    sys.exit(main())