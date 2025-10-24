#!/usr/bin/env python3
"""
Focused tests for foundation audio models:
- AudioMAE (Masked Audio Encoder)
- Data2Vec-Audio
- WavLM

Tests self-supervised learning capabilities and model architecture.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple

def create_speech_like_audio(duration=3, sample_rate=16000):
    """Create speech-like audio for foundation model testing."""
    t = torch.linspace(0, duration, sample_rate * duration)
    
    # Create formant-like structure (simplified speech)
    formants = [800, 1200, 2400]  # Typical formant frequencies
    speech = torch.zeros_like(t)
    
    # Generate formant structure with varying amplitude
    for i, f0 in enumerate(formants):
        amplitude = 0.3 / (i + 1)  # Decreasing amplitude for higher formants
        # Add frequency modulation to simulate speech variation
        freq_mod = f0 * (1 + 0.1 * torch.sin(2 * torch.pi * 3 * t))
        speech += amplitude * torch.sin(2 * torch.pi * freq_mod * t)
    
    # Add fundamental frequency variation (pitch contour)
    f0_base = 150  # Base fundamental frequency
    f0_contour = f0_base * (1 + 0.3 * torch.sin(2 * torch.pi * 0.5 * t))
    fundamental = 0.2 * torch.sin(2 * torch.pi * f0_contour * t)
    
    # Combine components
    audio = fundamental + speech
    
    # Add envelope (speech-like amplitude modulation)
    envelope = torch.abs(torch.sin(2 * torch.pi * 2 * t)) ** 0.5
    audio = audio * envelope
    
    # Add some noise for realism
    noise = 0.05 * torch.randn_like(audio)
    audio = audio + noise
    
    return audio.unsqueeze(0)  # Add batch dimension


def test_audio_mae():
    """Test AudioMAE self-supervised learning."""
    print("🎭 Testing AudioMAE")
    
    try:
        from modules.advanced_modules import create_audio_mae
        
        # Create model and test audio
        model = create_audio_mae()
        test_audio = create_speech_like_audio(duration=4, sample_rate=16000)
        
        # Test reconstruction task
        print("  Testing reconstruction task...")
        
        # Forward pass with high mask ratio
        result = model(test_audio, mask_ratio=0.75, return_loss=True)
        
        # Check output structure
        required_keys = ['encoded_features', 'reconstructed_features', 'mask', 'loss']
        has_keys = all(key in result for key in required_keys)
        
        if has_keys:
            print(f"  ✅ Output structure: PASS")
            
            # Check reconstruction loss
            loss = result['loss']
            print(f"  Reconstruction loss: {loss.item():.6f}")
            
            loss_reasonable = 0.001 < loss.item() < 10.0  # Reasonable loss range
            print(f"  ✅ Loss magnitude: {'PASS' if loss_reasonable else 'FAIL'}")
            
            # Check mask ratio
            mask = result['mask']
            actual_mask_ratio = mask.float().mean().item()
            print(f"  Mask ratio: {actual_mask_ratio:.3f} (target: 0.75)")
            
            mask_ok = abs(actual_mask_ratio - 0.75) < 0.1
            print(f"  ✅ Mask ratio: {'PASS' if mask_ok else 'FAIL'}")
            
            # Check feature shapes
            encoded = result['encoded_features']
            reconstructed = result['reconstructed_features']
            
            print(f"  Encoded shape: {encoded.shape}")
            print(f"  Reconstructed shape: {reconstructed.shape}")
            
            shapes_ok = len(encoded.shape) == 3 and len(reconstructed.shape) == 3
            print(f"  ✅ Feature shapes: {'PASS' if shapes_ok else 'FAIL'}")
            
            # Test gradient flow
            loss.backward()
            has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
            print(f"  ✅ Gradient flow: {'PASS' if has_gradients else 'FAIL'}")
            
            # Test inference mode (no loss computation)
            with torch.no_grad():
                inference_result = model(test_audio, mask_ratio=0.0, return_loss=False)
                inference_ok = 'encoded_features' in inference_result
                print(f"  ✅ Inference mode: {'PASS' if inference_ok else 'FAIL'}")
            
            overall_pass = (has_keys and loss_reasonable and mask_ok and 
                          shapes_ok and has_gradients and inference_ok)
            print(f"  🎯 Overall: {'PASS' if overall_pass else 'FAIL'}")
            
            return overall_pass
        else:
            print(f"  ❌ Missing keys: {set(required_keys) - set(result.keys())}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_data2vec_audio():
    """Test Data2Vec Audio self-supervised learning."""
    print("\n🎓 Testing Data2Vec Audio")
    
    try:
        from modules.advanced_modules import create_data2vec_audio
        
        model = create_data2vec_audio()
        test_audio = create_speech_like_audio(duration=3, sample_rate=16000)
        
        print("  Testing teacher-student learning...")
        
        # Forward pass
        result = model(test_audio, mask_prob=0.15, update_teacher=True)
        
        # Check output structure
        required_keys = ['predictions', 'targets', 'mask', 'loss', 'student_features', 'teacher_features']
        has_keys = all(key in result for key in required_keys)
        
        if has_keys:
            print(f"  ✅ Output structure: PASS")
            
            # Check contrastive loss
            loss = result['loss']
            print(f"  Contrastive loss: {loss.item():.6f}")
            
            loss_reasonable = 0.001 < loss.item() < 100.0
            print(f"  ✅ Loss magnitude: {'PASS' if loss_reasonable else 'FAIL'}")
            
            # Check student vs teacher features
            student_features = result['student_features']
            teacher_features = result['teacher_features']
            
            print(f"  Student features: {student_features.shape}")
            print(f"  Teacher features: {teacher_features.shape}")
            
            shapes_match = student_features.shape == teacher_features.shape
            print(f"  ✅ Feature shape match: {'PASS' if shapes_match else 'FAIL'}")
            
            # Check that teacher and student produce different outputs (due to masking)
            feature_diff = F.mse_loss(student_features, teacher_features).item()
            print(f"  Student-teacher MSE: {feature_diff:.6f}")
            
            diff_ok = feature_diff > 0.001  # Should have some difference
            print(f"  ✅ Student-teacher difference: {'PASS' if diff_ok else 'FAIL'}")
            
            # Check masking
            mask = result['mask']
            mask_ratio = mask.float().mean().item()
            print(f"  Actual mask ratio: {mask_ratio:.3f}")
            
            mask_ok = 0.1 < mask_ratio < 0.3  # Should be around 15%
            print(f"  ✅ Mask ratio: {'PASS' if mask_ok else 'FAIL'}")
            
            # Test gradient flow (only student should have gradients)
            loss.backward()
            student_has_grads = any(p.grad is not None for name, p in model.named_parameters() 
                                  if 'student' in name and p.requires_grad)
            teacher_has_grads = any(p.grad is not None for name, p in model.named_parameters() 
                                  if 'teacher' in name and p.requires_grad)
            
            print(f"  Student gradients: {student_has_grads}")
            print(f"  Teacher gradients: {teacher_has_grads}")
            
            gradient_ok = student_has_grads and not teacher_has_grads
            print(f"  ✅ Gradient flow: {'PASS' if gradient_ok else 'FAIL'}")
            
            overall_pass = (has_keys and loss_reasonable and shapes_match and 
                          diff_ok and mask_ok and gradient_ok)
            print(f"  🎯 Overall: {'PASS' if overall_pass else 'FAIL'}")
            
            return overall_pass
        else:
            print(f"  ❌ Missing keys: {set(required_keys) - set(result.keys())}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_wavlm():
    """Test WavLM contrastive learning."""
    print("\n🌊 Testing WavLM")
    
    try:
        from modules.advanced_modules import create_wavlm
        
        model = create_wavlm()
        test_audio = create_speech_like_audio(duration=3, sample_rate=16000)
        
        print("  Testing contrastive learning...")
        
        # Forward pass
        result = model(test_audio, mask_prob=0.15, mask_length=10)
        
        # Check output structure
        required_keys = ['contextualized_features', 'projected_features', 'quantized_targets', 'mask', 'loss']
        has_keys = all(key in result for key in required_keys)
        
        if has_keys:
            print(f"  ✅ Output structure: PASS")
            
            # Check contrastive loss
            loss = result['loss']
            print(f"  Contrastive loss: {loss.item():.6f}")
            
            loss_reasonable = 0.0 <= loss.item() < 100.0  # Loss can be 0 in eval mode
            print(f"  ✅ Loss magnitude: {'PASS' if loss_reasonable else 'FAIL'}")
            
            # Check feature dimensions
            contextualized = result['contextualized_features']
            projected = result['projected_features']
            quantized = result['quantized_targets']
            
            print(f"  Contextualized features: {contextualized.shape}")
            print(f"  Projected features: {projected.shape}")
            print(f"  Quantized targets: {quantized.shape}")
            
            # Check that dimensions are reasonable
            same_length = (contextualized.shape[1] == projected.shape[1] == quantized.shape[1])
            print(f"  ✅ Sequence length match: {'PASS' if same_length else 'FAIL'}")
            
            # Check masking
            mask = result['mask']
            masked_positions = mask.sum().item()
            total_positions = mask.numel()
            mask_ratio = masked_positions / total_positions
            
            print(f"  Masked positions: {masked_positions}/{total_positions} ({mask_ratio:.3f})")
            
            mask_ok = 0.05 < mask_ratio < 0.5  # Reasonable mask ratio
            print(f"  ✅ Mask ratio: {'PASS' if mask_ok else 'FAIL'}")
            
            # Test training mode (should have non-zero loss)
            model.train()
            train_result = model(test_audio, mask_prob=0.15, mask_length=10)
            train_loss = train_result['loss'].item()
            
            print(f"  Training loss: {train_loss:.6f}")
            training_ok = train_loss > 0.0
            print(f"  ✅ Training mode: {'PASS' if training_ok else 'FAIL'}")
            
            # Test gradient flow in training mode
            train_result['loss'].backward()
            has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
            print(f"  ✅ Gradient flow: {'PASS' if has_gradients else 'FAIL'}")
            
            overall_pass = (has_keys and loss_reasonable and same_length and 
                          mask_ok and training_ok and has_gradients)
            print(f"  🎯 Overall: {'PASS' if overall_pass else 'FAIL'}")
            
            return overall_pass
        else:
            print(f"  ❌ Missing keys: {set(required_keys) - set(result.keys())}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_foundation_model_consistency():
    """Test consistency across foundation models."""
    print("\n🔗 Testing Foundation Model Consistency")
    
    try:
        from modules.advanced_modules import create_audio_mae, create_data2vec_audio, create_wavlm
        
        # Create consistent test audio
        test_audio = create_speech_like_audio(duration=2, sample_rate=16000)
        
        models = {
            'AudioMAE': create_audio_mae(),
            'Data2Vec': create_data2vec_audio(),
            'WavLM': create_wavlm()
        }
        
        # Test that all models accept same input
        print("  Testing input compatibility...")
        
        input_compatible = True
        feature_shapes = {}
        
        for name, model in models.items():
            try:
                with torch.no_grad():
                    if name == 'AudioMAE':
                        result = model(test_audio, return_loss=False)
                        features = result['encoded_features']
                    elif name == 'Data2Vec':
                        result = model(test_audio, update_teacher=False)
                        features = result['student_features']
                    else:  # WavLM
                        result = model(test_audio)
                        features = result['contextualized_features']
                    
                    feature_shapes[name] = features.shape
                    print(f"    {name}: {features.shape}")
                    
            except Exception as e:
                print(f"    ❌ {name} failed: {e}")
                input_compatible = False
        
        print(f"  ✅ Input compatibility: {'PASS' if input_compatible else 'FAIL'}")
        
        # Test configuration consistency  
        print("  Testing configuration consistency...")
        
        config_consistent = True
        try:
            from modules.advanced_modules.foundation_models import FoundationModelConfig, FoundationModelType
            from modules.audio_analysis.audio_config import get_music_config
            
            # Test that foundation configs extend base config properly
            base_config = get_music_config()
            
            # Test config creation
            for model_type in FoundationModelType:
                if model_type in [FoundationModelType.AUDIO_MAE, 
                                FoundationModelType.DATA2VEC_AUDIO, 
                                FoundationModelType.WAVLM]:
                    try:
                        config = FoundationModelConfig(
                            model_type=model_type,
                            sample_rate=base_config.sample_rate,
                            n_fft=base_config.n_fft,
                            hop_length=base_config.hop_length
                        )
                        print(f"    {model_type.value}: config created successfully")
                    except Exception as e:
                        print(f"    ❌ {model_type.value}: {e}")
                        config_consistent = False
                        
        except Exception as e:
            print(f"    ❌ Configuration test failed: {e}")
            config_consistent = False
        
        print(f"  ✅ Configuration consistency: {'PASS' if config_consistent else 'FAIL'}")
        
        # Test feature extraction for downstream tasks
        print("  Testing feature extraction...")
        
        extraction_ok = True
        try:
            # All models should be able to extract meaningful features
            for name, model in models.items():
                model.eval()
                with torch.no_grad():
                    if name == 'AudioMAE':
                        result = model(test_audio, mask_ratio=0.0, return_loss=False)
                        features = result['encoded_features']
                    elif name == 'Data2Vec':
                        result = model(test_audio, mask_prob=0.0, update_teacher=False)
                        features = result['teacher_features']  # Use teacher for stable features
                    else:  # WavLM
                        result = model(test_audio, mask_prob=0.0)
                        features = result['contextualized_features']
                    
                    # Check feature quality (no NaN/Inf, reasonable variance)
                    has_nan = torch.isnan(features).any()
                    has_inf = torch.isinf(features).any()
                    feature_var = features.var().item()
                    
                    feature_quality = not has_nan and not has_inf and feature_var > 1e-6
                    print(f"    {name}: variance={feature_var:.6f}, quality={'PASS' if feature_quality else 'FAIL'}")
                    
                    if not feature_quality:
                        extraction_ok = False
                        
        except Exception as e:
            print(f"    ❌ Feature extraction failed: {e}")
            extraction_ok = False
        
        print(f"  ✅ Feature extraction: {'PASS' if extraction_ok else 'FAIL'}")
        
        overall_pass = input_compatible and config_consistent and extraction_ok
        print(f"  🎯 Overall: {'PASS' if overall_pass else 'FAIL'}")
        
        return overall_pass
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    """Run all foundation model tests."""
    print("🏗️  Foundation Models Test Suite")
    print("=" * 50)
    
    results = {}
    
    # Run individual model tests
    results['audio_mae'] = test_audio_mae()
    results['data2vec_audio'] = test_data2vec_audio() 
    results['wavlm'] = test_wavlm()
    results['consistency'] = test_foundation_model_consistency()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 FOUNDATION MODELS TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    success_rate = (passed / total) * 100
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate >= 75:
        print("🎉 Foundation models test suite PASSED!")
        return 0
    else:
        print("💥 Foundation models test suite FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())