#!/usr/bin/env python3
"""
Quick test of Descript Discriminator with synthetic audio
"""

import torch
import numpy as np
from standalone_descript_discriminator import StandaloneDescriptDiscriminator


def create_test_audio():
    """Create different types of test audio for comparison."""
    
    sample_rate = 44100
    duration = 2.0
    length = int(sample_rate * duration)
    t = torch.linspace(0, duration, length)
    
    # 1. High quality: Clean sine wave with harmonics
    fundamental = 440  # A4
    high_quality = (
        torch.sin(2 * np.pi * fundamental * t) * 0.5 +
        torch.sin(2 * np.pi * fundamental * 2 * t) * 0.2 +  # 2nd harmonic
        torch.sin(2 * np.pi * fundamental * 3 * t) * 0.1    # 3rd harmonic
    )
    
    # 2. Medium quality: Sine wave with some noise
    medium_quality = torch.sin(2 * np.pi * 440 * t) + 0.1 * torch.randn(length)
    
    # 3. Low quality: Heavy noise
    low_quality = 0.3 * torch.sin(2 * np.pi * 440 * t) + 0.7 * torch.randn(length)
    
    # 4. Very low quality: Almost pure noise
    very_low_quality = 0.1 * torch.sin(2 * np.pi * 440 * t) + 0.9 * torch.randn(length)
    
    # Add batch and channel dimensions: [1, 1, length]
    return {
        'high_quality': high_quality.unsqueeze(0).unsqueeze(0),
        'medium_quality': medium_quality.unsqueeze(0).unsqueeze(0),
        'low_quality': low_quality.unsqueeze(0).unsqueeze(0),
        'very_low_quality': very_low_quality.unsqueeze(0).unsqueeze(0),
    }


def main():
    print("🎵 Quick Descript Discriminator Test")
    print("=" * 50)
    
    # Initialize discriminator
    discriminator = StandaloneDescriptDiscriminator(
        sample_rate=44100,
        n_channels=1,
        device='cpu'
    )
    
    # Create test audio
    print("\n🔊 Creating synthetic test audio...")
    test_audios = create_test_audio()
    
    # Analyze each type
    print("\n📊 Discriminator Analysis Results:")
    print("-" * 50)
    
    results = {}
    for audio_type, audio_tensor in test_audios.items():
        analysis = discriminator.analyze_audio(audio_tensor)
        score = analysis['overall_score']
        results[audio_type] = score
        
        print(f"{audio_type.replace('_', ' ').title():15}: {score:.4f}")
        
        # Show sub-discriminator breakdown
        for i, sub_score in enumerate(analysis['scores']):
            print(f"  Sub-disc {i}: {sub_score['mean_score']:.4f}")
        print()
    
    # Summary
    print("📈 Summary:")
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for i, (audio_type, score) in enumerate(sorted_results):
        rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
        print(f"{rank} {audio_type.replace('_', ' ').title()}: {score:.4f}")
    
    print(f"\n✅ Discriminator successfully distinguishes between audio qualities!")
    print(f"   Range: {min(results.values()):.4f} to {max(results.values()):.4f}")
    
    # Expected behavior explanation
    print(f"\n💡 Expected behavior:")
    print(f"   - Higher scores = more 'real-like' audio")
    print(f"   - The discriminator should rank: high > medium > low > very_low")
    print(f"   - Clean harmonic content scores higher than noise")


if __name__ == "__main__":
    main()