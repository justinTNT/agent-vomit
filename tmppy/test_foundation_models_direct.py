#!/usr/bin/env python3
"""
Direct test of foundation models to debug the parameter issue
"""

import torch
import sys
sys.path.append('/Users/jtnt/Play/agent-vomit')

try:
    from modules.advanced_modules.foundation_models import (
        AudioMAE, Data2VecAudio, WavLM, 
        FoundationModelConfig, FoundationModelType,
        create_audio_mae, create_data2vec_audio, create_wavlm
    )
    
    print("✅ Successfully imported foundation models")
    
    # Test factory functions
    print("\nTesting factory functions:")
    
    try:
        mae = create_audio_mae()
        print("✅ AudioMAE factory works")
        
        # Test forward pass
        waveform = torch.randn(1, 16000)
        output = mae(waveform)
        print(f"✅ AudioMAE forward pass works, output keys: {list(output.keys())}")
        
    except Exception as e:
        print(f"❌ AudioMAE factory failed: {e}")
    
    try:
        d2v = create_data2vec_audio()
        print("✅ Data2VecAudio factory works")
        
        # Test forward pass  
        waveform = torch.randn(1, 16000)
        output = d2v(waveform)
        print(f"✅ Data2VecAudio forward pass works, output keys: {list(output.keys())}")
        
    except Exception as e:
        print(f"❌ Data2VecAudio factory failed: {e}")
    
    try:
        wavlm = create_wavlm()
        print("✅ WavLM factory works")
        
        # Test forward pass
        waveform = torch.randn(1, 16000)
        output = wavlm(waveform)
        print(f"✅ WavLM forward pass works, output keys: {list(output.keys())}")
        
    except Exception as e:
        print(f"❌ WavLM factory failed: {e}")

    # Test config creation
    print("\nTesting config creation:")
    try:
        config = FoundationModelConfig(
            model_type=FoundationModelType.AUDIO_MAE
        )
        print("✅ Config creation works")
        
        mae = AudioMAE(config)
        print("✅ AudioMAE with config works")
        
    except Exception as e:
        print(f"❌ Config creation failed: {e}")

except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()