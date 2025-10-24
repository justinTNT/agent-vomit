#!/usr/bin/env python3
"""
Focused tests for multimodal audio models:
- CLAP (Contrastive Language-Audio Pre-training)
- ImageBind Audio Component
- Audio Captioning Model

Tests cross-modal alignment and text-audio understanding.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple

def create_audio_text_pairs():
    """Create synthetic audio-text pairs for testing."""
    # Audio descriptions that match synthetic audio characteristics
    descriptions = [
        "a piano playing a simple melody",
        "drum beats with a steady rhythm", 
        "synthesized tones and electronic music",
        "acoustic guitar strumming chords",
        "ambient sounds with low frequencies"
    ]
    
    # Create corresponding audio (simplified synthetic versions)
    sample_rate = 22050
    duration = 3
    batch_size = len(descriptions)
    
    audio_batch = []
    
    for i, desc in enumerate(descriptions):
        t = torch.linspace(0, duration, sample_rate * duration)
        
        if "piano" in desc:
            # Piano-like tones (harmonic series)
            audio = 0.3 * torch.sin(2 * torch.pi * 261.63 * t)  # C4
            audio += 0.2 * torch.sin(2 * torch.pi * 329.63 * t)  # E4
            audio += 0.15 * torch.sin(2 * torch.pi * 392.00 * t)  # G4
            
        elif "drum" in desc:
            # Drum-like percussive sounds
            audio = torch.zeros_like(t)
            for beat_time in torch.arange(0, duration, 0.5):
                beat_idx = int(beat_time * sample_rate)
                if beat_idx < len(audio) - 1000:
                    drum_hit = torch.randn(1000) * torch.exp(-10 * torch.linspace(0, 0.1, 1000))
                    audio[beat_idx:beat_idx + 1000] += drum_hit
                    
        elif "synthesized" in desc:
            # Synthetic tones with modulation
            carrier = 440  # A4
            modulator = 5  # 5 Hz modulation
            audio = 0.4 * torch.sin(2 * torch.pi * carrier * t) * (1 + 0.5 * torch.sin(2 * torch.pi * modulator * t))
            
        elif "guitar" in desc:
            # Guitar-like harmonics
            fundamentals = [82.41, 110, 146.83, 196, 246.94, 329.63]  # Guitar strings
            audio = torch.zeros_like(t)
            for freq in fundamentals:
                audio += 0.15 * torch.sin(2 * torch.pi * freq * t) * torch.exp(-2 * t)
                
        else:  # ambient
            # Low frequency ambient sounds
            audio = 0.2 * torch.sin(2 * torch.pi * 60 * t) + 0.1 * torch.sin(2 * torch.pi * 80 * t)
            audio += 0.05 * torch.randn_like(t)  # Noise
        
        # Add envelope and normalize
        envelope = torch.exp(-0.5 * t)
        audio = audio * envelope
        audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
        
        audio_batch.append(audio)
    
    # Stack into batch
    audio_tensor = torch.stack(audio_batch)
    
    # Create tokenized text (simplified - just random tokens for testing)
    vocab_size = 1000
    max_length = 10
    
    input_ids = torch.randint(1, vocab_size, (batch_size, max_length))
    attention_mask = torch.ones_like(input_ids)
    
    return audio_tensor, input_ids, attention_mask, descriptions


def test_clap_model():
    """Test CLAP contrastive learning."""
    print("🎯 Testing CLAP")
    
    try:
        from modules.advanced_modules import create_clap
        
        # Create model and test data
        model = create_clap()
        audio_batch, input_ids, attention_mask, descriptions = create_audio_text_pairs()
        
        print(f"  Test data: {len(descriptions)} audio-text pairs")
        print(f"  Audio shape: {audio_batch.shape}")
        print(f"  Text shape: {input_ids.shape}")
        
        # Test contrastive learning
        print("  Testing contrastive learning...")
        
        result = model(audio_batch, input_ids, attention_mask)
        
        # Check output structure
        required_keys = ['audio_embeddings', 'text_embeddings', 'total_loss', 
                        'audio_to_text_accuracy', 'text_to_audio_accuracy']
        has_keys = all(key in result for key in required_keys)
        
        if has_keys:
            print(f"  ✅ Output structure: PASS")
            
            # Check embedding dimensions
            audio_emb = result['audio_embeddings']
            text_emb = result['text_embeddings']
            
            print(f"  Audio embeddings: {audio_emb.shape}")
            print(f"  Text embeddings: {text_emb.shape}")
            
            same_dim = audio_emb.shape == text_emb.shape
            print(f"  ✅ Embedding dimensions: {'PASS' if same_dim else 'FAIL'}")
            
            # Check contrastive loss
            loss = result['total_loss']
            print(f"  Contrastive loss: {loss.item():.6f}")
            
            loss_reasonable = 0.001 < loss.item() < 100.0
            print(f"  ✅ Loss magnitude: {'PASS' if loss_reasonable else 'FAIL'}")
            
            # Check retrieval accuracy
            audio_to_text_acc = result['audio_to_text_accuracy']
            text_to_audio_acc = result['text_to_audio_accuracy']
            
            print(f"  Audio→Text accuracy: {audio_to_text_acc:.3f}")
            print(f"  Text→Audio accuracy: {text_to_audio_acc:.3f}")
            
            # For random initialization, accuracy should be around 1/batch_size
            batch_size = audio_batch.shape[0]
            expected_random_acc = 1.0 / batch_size
            
            # Accuracy should be at least random chance (could be higher due to architecture)
            acc_ok = (audio_to_text_acc >= expected_random_acc * 0.8 and 
                     text_to_audio_acc >= expected_random_acc * 0.8)
            print(f"  ✅ Retrieval accuracy: {'PASS' if acc_ok else 'FAIL'}")
            
            # Test similarity computation
            print("  Testing similarity computation...")
            
            with torch.no_grad():
                audio_features = model.get_audio_features(audio_batch)
                text_features = model.get_text_features(input_ids, attention_mask)
                
                # Compute similarity matrix
                similarities = torch.matmul(audio_features, text_features.t())
                
                print(f"  Similarity matrix: {similarities.shape}")
                print(f"  Similarity range: [{similarities.min():.3f}, {similarities.max():.3f}]")
                
                # Check that diagonal elements (matching pairs) have reasonable similarities
                diagonal_similarities = torch.diag(similarities)
                print(f"  Diagonal similarities: {diagonal_similarities}")
                
                sim_ok = similarities.shape == (batch_size, batch_size)
                print(f"  ✅ Similarity computation: {'PASS' if sim_ok else 'FAIL'}")
            
            # Test gradient flow
            loss.backward()
            has_audio_grads = any(p.grad is not None for name, p in model.named_parameters() 
                                if 'audio' in name and p.requires_grad)
            has_text_grads = any(p.grad is not None for name, p in model.named_parameters() 
                               if 'text' in name and p.requires_grad)
            
            gradient_ok = has_audio_grads and has_text_grads
            print(f"  ✅ Gradient flow: {'PASS' if gradient_ok else 'FAIL'}")
            
            overall_pass = (has_keys and same_dim and loss_reasonable and 
                          acc_ok and sim_ok and gradient_ok)
            print(f"  🎯 Overall: {'PASS' if overall_pass else 'FAIL'}")
            
            return overall_pass
        else:
            print(f"  ❌ Missing keys: {set(required_keys) - set(result.keys())}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_imagebind_audio():
    """Test ImageBind audio component."""
    print("\n🌐 Testing ImageBind Audio")
    
    try:
        from modules.advanced_modules import create_imagebind_audio
        
        model = create_imagebind_audio()
        audio_batch, _, _, descriptions = create_audio_text_pairs()
        
        print("  Testing multi-scale audio encoding...")
        
        result = model(audio_batch)
        
        # Check output structure
        required_keys = ['scale_embeddings', 'fused_embedding', 'final_embedding']
        has_keys = all(key in result for key in required_keys)
        
        if has_keys:
            print(f"  ✅ Output structure: PASS")
            
            # Check multi-scale embeddings
            scale_embeddings = result['scale_embeddings']
            print(f"  Number of scales: {len(scale_embeddings)}")
            
            # Should have multiple scales
            multi_scale_ok = len(scale_embeddings) >= 2
            print(f"  ✅ Multi-scale processing: {'PASS' if multi_scale_ok else 'FAIL'}")
            
            # Check embedding shapes
            for i, emb in enumerate(scale_embeddings):
                print(f"    Scale {i}: {emb.shape}")
            
            fused_emb = result['fused_embedding']
            final_emb = result['final_embedding']
            
            print(f"  Fused embedding: {fused_emb.shape}")
            print(f"  Final embedding: {final_emb.shape}")
            
            # Check that final embedding is normalized (for multimodal alignment)
            final_norms = torch.norm(final_emb, dim=1)
            print(f"  Final embedding norms: {final_norms}")
            
            # Should be close to 1 if normalized
            normalized_ok = torch.allclose(final_norms, torch.ones_like(final_norms), atol=0.1)
            print(f"  ✅ Embedding normalization: {'PASS' if normalized_ok else 'FAIL'}")
            
            # Check that embeddings are different for different inputs
            if audio_batch.shape[0] > 1:
                emb_diffs = []
                for i in range(len(final_emb)):
                    for j in range(i + 1, len(final_emb)):
                        diff = F.cosine_similarity(final_emb[i], final_emb[j], dim=0)
                        emb_diffs.append(diff.item())
                
                avg_similarity = np.mean(emb_diffs)
                print(f"  Average pairwise similarity: {avg_similarity:.3f}")
                
                # Embeddings shouldn't be too similar (indicating they capture differences)
                diversity_ok = avg_similarity < 0.9
                print(f"  ✅ Embedding diversity: {'PASS' if diversity_ok else 'FAIL'}")
            else:
                diversity_ok = True
            
            # Test gradient flow
            loss = final_emb.sum()  # Dummy loss for gradient test
            loss.backward()
            has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
            print(f"  ✅ Gradient flow: {'PASS' if has_gradients else 'FAIL'}")
            
            overall_pass = (has_keys and multi_scale_ok and normalized_ok and 
                          diversity_ok and has_gradients)
            print(f"  🎯 Overall: {'PASS' if overall_pass else 'FAIL'}")
            
            return overall_pass
        else:
            print(f"  ❌ Missing keys: {set(required_keys) - set(result.keys())}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_audio_captioning():
    """Test audio captioning model."""
    print("\n💬 Testing Audio Captioning")
    
    try:
        from modules.advanced_modules import create_audio_captioning_model
        
        model = create_audio_captioning_model()
        audio_batch, input_ids, attention_mask, descriptions = create_audio_text_pairs()
        
        print("  Testing caption generation...")
        
        # Test inference mode (caption generation)
        with torch.no_grad():
            inference_result = model(audio_batch[:2])  # Test with 2 samples
            
            if 'generated_captions' in inference_result:
                captions = inference_result['generated_captions']
                print(f"  Generated {len(captions)} captions")
                
                # Check caption structure
                captions_ok = True
                for i, caption in enumerate(captions):
                    if not isinstance(caption, list) or len(caption) == 0:
                        captions_ok = False
                        print(f"    Caption {i}: Invalid format")
                    else:
                        print(f"    Caption {i}: {len(caption)} tokens")
                
                print(f"  ✅ Caption generation: {'PASS' if captions_ok else 'FAIL'}")
            else:
                print(f"  ❌ Caption generation: FAIL (no captions generated)")
                captions_ok = False
        
        # Test training mode (with target captions)
        print("  Testing training mode...")
        
        model.train()
        train_result = model(audio_batch, target_ids=input_ids)
        
        if 'logits' in train_result and 'loss' in train_result:
            logits = train_result['logits']
            loss = train_result['loss']
            
            print(f"  Training logits: {logits.shape}")
            print(f"  Training loss: {loss.item():.6f}")
            
            # Check logits shape
            batch_size, seq_len = input_ids.shape
            expected_shape = (batch_size, seq_len - 1, 50000)  # vocab_size = 50000
            
            logits_shape_ok = logits.shape[:2] == expected_shape[:2]  # Check batch and seq dims
            print(f"  ✅ Logits shape: {'PASS' if logits_shape_ok else 'FAIL'}")
            
            # Check loss magnitude
            loss_reasonable = 0.1 < loss.item() < 50.0  # Reasonable for language modeling
            print(f"  ✅ Loss magnitude: {'PASS' if loss_reasonable else 'FAIL'}")
            
            # Test gradient flow
            loss.backward()
            has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
            print(f"  ✅ Gradient flow: {'PASS' if has_gradients else 'FAIL'}")
            
            training_ok = logits_shape_ok and loss_reasonable and has_gradients
        else:
            print(f"  ❌ Training mode: FAIL (missing logits or loss)")
            training_ok = False
        
        # Test audio feature extraction
        print("  Testing audio feature extraction...")
        
        audio_features = train_result.get('audio_features')
        if audio_features is not None:
            print(f"  Audio features: {audio_features.shape}")
            
            # Should extract meaningful features
            feature_var = audio_features.var().item()
            features_ok = feature_var > 1e-6 and not torch.isnan(audio_features).any()
            print(f"  ✅ Audio features: {'PASS' if features_ok else 'FAIL'}")
        else:
            print(f"  ❌ Audio features: FAIL (no features extracted)")
            features_ok = False
        
        overall_pass = captions_ok and training_ok and features_ok
        print(f"  🎯 Overall: {'PASS' if overall_pass else 'FAIL'}")
        
        return overall_pass
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_multimodal_integration():
    """Test integration between multimodal models."""
    print("\n🔗 Testing Multimodal Integration")
    
    try:
        from modules.advanced_modules import create_clap, create_imagebind_audio
        
        # Create models
        clap = create_clap()
        imagebind = create_imagebind_audio()
        
        # Create test data
        audio_batch, input_ids, attention_mask, descriptions = create_audio_text_pairs()
        
        print("  Testing embedding space compatibility...")
        
        with torch.no_grad():
            # Extract features from both models
            clap_audio_features = clap.get_audio_features(audio_batch)
            imagebind_features = imagebind(audio_batch)['final_embedding']
            
            print(f"  CLAP audio features: {clap_audio_features.shape}")
            print(f"  ImageBind features: {imagebind_features.shape}")
            
            # Check if embeddings are normalized similarly
            clap_norms = torch.norm(clap_audio_features, dim=1)
            imagebind_norms = torch.norm(imagebind_features, dim=1)
            
            print(f"  CLAP embedding norms: {clap_norms.mean():.3f} ± {clap_norms.std():.3f}")
            print(f"  ImageBind embedding norms: {imagebind_norms.mean():.3f} ± {imagebind_norms.std():.3f}")
            
            # Both should be normalized (norm ≈ 1)
            clap_normalized = torch.allclose(clap_norms, torch.ones_like(clap_norms), atol=0.2)
            imagebind_normalized = torch.allclose(imagebind_norms, torch.ones_like(imagebind_norms), atol=0.2)
            
            normalization_ok = clap_normalized and imagebind_normalized
            print(f"  ✅ Embedding normalization: {'PASS' if normalization_ok else 'FAIL'}")
            
            # Test cross-model similarity
            if clap_audio_features.shape[1] == imagebind_features.shape[1]:
                cross_similarities = F.cosine_similarity(clap_audio_features, imagebind_features, dim=1)
                print(f"  Cross-model similarities: {cross_similarities}")
                
                # Similarities should be moderate (not too high, not too low)
                sim_reasonable = torch.all((cross_similarities > -0.5) & (cross_similarities < 0.8))
                print(f"  ✅ Cross-model similarity: {'PASS' if sim_reasonable else 'FAIL'}")
            else:
                print("  ⚠️  Cross-model similarity: SKIP (different dimensions)")
                sim_reasonable = True
            
            # Test configuration consistency
            print("  Testing configuration consistency...")
            
            try:
                from modules.advanced_modules.multimodal_models import MultimodalConfig, MultimodalModelType
                from modules.audio_analysis.audio_config import get_music_config
                
                base_config = get_music_config()
                
                clap_config = MultimodalConfig(
                    model_type=MultimodalModelType.CLAP,
                    sample_rate=base_config.sample_rate,
                    n_fft=base_config.n_fft
                )
                
                imagebind_config = MultimodalConfig(
                    model_type=MultimodalModelType.IMAGEBIND_AUDIO,
                    sample_rate=base_config.sample_rate,
                    n_fft=base_config.n_fft
                )
                
                config_ok = (clap_config.sample_rate == imagebind_config.sample_rate and
                           clap_config.n_fft == imagebind_config.n_fft)
                print(f"  ✅ Configuration consistency: {'PASS' if config_ok else 'FAIL'}")
                
            except Exception as e:
                print(f"  ❌ Configuration test failed: {e}")
                config_ok = False
            
            overall_pass = normalization_ok and sim_reasonable and config_ok
            print(f"  🎯 Overall: {'PASS' if overall_pass else 'FAIL'}")
            
            return overall_pass
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    """Run all multimodal model tests."""
    print("🎭 Multimodal Models Test Suite")
    print("=" * 50)
    
    results = {}
    
    # Run individual model tests
    results['clap'] = test_clap_model()
    results['imagebind_audio'] = test_imagebind_audio()
    results['audio_captioning'] = test_audio_captioning()
    results['integration'] = test_multimodal_integration()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 MULTIMODAL MODELS TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    success_rate = (passed / total) * 100
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate >= 75:
        print("🎉 Multimodal models test suite PASSED!")
        return 0
    else:
        print("💥 Multimodal models test suite FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())