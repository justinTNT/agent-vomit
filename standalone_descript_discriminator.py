#!/usr/bin/env python3
"""
Standalone Descript Discriminator for External Projects
Extract and use the state-of-the-art Descript discriminator independently from RAVE.
"""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
import torchaudio
import numpy as np
from typing import List, Tuple, Union

# Try to import RAVE discriminator from multiple possible locations
DESCRIPT_AVAILABLE = False

# Option 1: Try direct import (if RAVE is installed or in PYTHONPATH)
try:
    from rave.descript_discriminator import DescriptDiscriminator
    DESCRIPT_AVAILABLE = True
    print("✅ Found RAVE discriminator via direct import")
except ImportError:
    pass

# Option 2: Try relative path (if in timbralgebraics project structure)
if not DESCRIPT_AVAILABLE:
    try:
        sys.path.append('../timbralgebraics/external/rave')
        from rave.descript_discriminator import DescriptDiscriminator
        DESCRIPT_AVAILABLE = True
        print("✅ Found RAVE discriminator via relative path")
    except ImportError:
        pass

# Option 3: Try environment variable
if not DESCRIPT_AVAILABLE:
    rave_path = os.environ.get('RAVE_PATH')
    if rave_path:
        try:
            sys.path.append(rave_path)
            from rave.descript_discriminator import DescriptDiscriminator
            DESCRIPT_AVAILABLE = True
            print(f"✅ Found RAVE discriminator via RAVE_PATH: {rave_path}")
        except ImportError:
            pass

# Final check
if not DESCRIPT_AVAILABLE:
    print("❌ Warning: DescriptDiscriminator not available.")
    print("   Options to fix:")
    print("   1. Install RAVE: pip install rave-audio")
    print("   2. Set RAVE_PATH environment variable")
    print("   3. Place this file in a project with RAVE available")
    print("   4. Manually add RAVE to Python path before importing")


class StandaloneDescriptDiscriminator:
    """
    Wrapper for using Descript discriminator independently.
    
    Features:
    - Multi-Period Discriminator (temporal patterns)
    - Multi-Scale Discriminator (multi-resolution analysis)  
    - Multi-Resolution Discriminator (frequency-domain analysis)
    """
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 n_channels: int = 1,
                 device: str = 'cpu'):
        """
        Initialize standalone Descript discriminator.
        
        Args:
            sample_rate: Audio sample rate (default 44100)
            n_channels: Number of audio channels (1=mono, 2=stereo)
            device: 'cpu' or 'cuda'
        """
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        self.device = device
        
        if not DESCRIPT_AVAILABLE:
            raise ImportError("DescriptDiscriminator not available. Check RAVE installation.")
        
        # Initialize discriminator
        self.discriminator = DescriptDiscriminator(n_channels=n_channels)
        self.discriminator.to(device)
        self.discriminator.eval()
        
        print(f"✅ Initialized Descript Discriminator:")
        print(f"   Sample rate: {sample_rate} Hz")
        print(f"   Channels: {n_channels}")
        print(f"   Device: {device}")
    
    def load_audio(self, file_path: Union[str, Path]) -> torch.Tensor:
        """
        Load audio file and prepare for discriminator.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Audio tensor shaped [1, n_channels, length]
        """
        # Load audio
        audio, sr = torchaudio.load(file_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
        
        # Handle channel count
        if audio.shape[0] != self.n_channels:
            if self.n_channels == 1 and audio.shape[0] > 1:
                # Convert to mono
                audio = audio.mean(dim=0, keepdim=True)
            elif self.n_channels == 2 and audio.shape[0] == 1:
                # Convert mono to stereo
                audio = audio.repeat(2, 1)
            else:
                raise ValueError(f"Cannot convert {audio.shape[0]} channels to {self.n_channels}")
        
        # Add batch dimension
        audio = audio.unsqueeze(0)  # [1, n_channels, length]
        
        return audio.to(self.device)
    
    def analyze_audio(self, audio: torch.Tensor) -> dict:
        """
        Run discriminator analysis on audio.
        
        Args:
            audio: Audio tensor [batch, channels, length]
            
        Returns:
            Dictionary with discriminator outputs and analysis
        """
        with torch.no_grad():
            # Get discriminator outputs
            # Descript discriminator returns list of (logits, features) for each sub-discriminator
            outputs = self.discriminator(audio)
            
            analysis = {
                'discriminator_outputs': outputs,
                'num_sub_discriminators': len(outputs),
                'audio_shape': audio.shape,
            }
            
            # Extract scores from each sub-discriminator
            scores = []
            for i, sub_discriminator_outputs in enumerate(outputs):
                # The last tensor in each sub-discriminator output is the final score
                final_logits = sub_discriminator_outputs[-1]  # Last tensor is the score
                score = torch.sigmoid(final_logits).cpu().numpy()
                scores.append({
                    'sub_discriminator': i,
                    'score': score,
                    'mean_score': float(score.mean()),
                    'logits_shape': final_logits.shape,
                    'num_features': len(sub_discriminator_outputs) - 1  # All except final score
                })
            
            analysis['scores'] = scores
            analysis['overall_score'] = np.mean([s['mean_score'] for s in scores])
            
            return analysis
    
    def analyze_file(self, file_path: Union[str, Path]) -> dict:
        """
        Complete analysis pipeline for a single audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with file info and discriminator analysis
        """
        file_path = Path(file_path)
        
        try:
            # Load audio
            audio = self.load_audio(file_path)
            
            # Analyze
            analysis = self.analyze_audio(audio)
            
            # Add file metadata
            analysis.update({
                'file_path': str(file_path),
                'file_name': file_path.name,
                'audio_length_seconds': audio.shape[-1] / self.sample_rate,
                'success': True
            })
            
            return analysis
            
        except Exception as e:
            return {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'success': False,
                'error': str(e)
            }
    
    def analyze_directory(self, 
                         directory: Union[str, Path], 
                         audio_extensions: List[str] = None) -> List[dict]:
        """
        Analyze all audio files in a directory.
        
        Args:
            directory: Path to directory containing audio files
            audio_extensions: List of file extensions to process (default: common audio formats)
            
        Returns:
            List of analysis results for each file
        """
        if audio_extensions is None:
            audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg']
        
        directory = Path(directory)
        audio_files = []
        
        # Find all audio files
        for ext in audio_extensions:
            audio_files.extend(directory.glob(f"*{ext}"))
            audio_files.extend(directory.glob(f"*{ext.upper()}"))
        
        print(f"Found {len(audio_files)} audio files in {directory}")
        
        results = []
        for i, file_path in enumerate(audio_files):
            print(f"Processing {i+1}/{len(audio_files)}: {file_path.name}")
            result = self.analyze_file(file_path)
            results.append(result)
        
        return results
    
    def compare_audio_quality(self, 
                             real_audio: torch.Tensor, 
                             generated_audio: torch.Tensor) -> dict:
        """
        Compare real vs generated audio using discriminator.
        Higher scores = more "real-like"
        
        Args:
            real_audio: Real audio tensor [batch, channels, length]
            generated_audio: Generated audio tensor [batch, channels, length]
            
        Returns:
            Comparison analysis
        """
        real_analysis = self.analyze_audio(real_audio)
        generated_analysis = self.analyze_audio(generated_audio)
        
        return {
            'real_score': real_analysis['overall_score'],
            'generated_score': generated_analysis['overall_score'],
            'quality_ratio': generated_analysis['overall_score'] / (real_analysis['overall_score'] + 1e-8),
            'real_analysis': real_analysis,
            'generated_analysis': generated_analysis
        }


def demo_usage():
    """Demonstrate how to use the standalone discriminator."""
    
    print("🎵 Descript Discriminator Demo")
    print("=" * 50)
    
    # Initialize discriminator
    discriminator = StandaloneDescriptDiscriminator(
        sample_rate=44100,
        n_channels=1,
        device='cpu'  # Change to 'cuda' if you have GPU
    )
    
    # Example 1: Analyze a single file
    print("\n📁 Single file analysis:")
    # Replace with actual audio file path
    # result = discriminator.analyze_file("/path/to/your/audio.wav")
    # print(f"Overall quality score: {result['overall_score']:.3f}")
    
    # Example 2: Analyze directory
    print("\n📂 Directory analysis:")
    # results = discriminator.analyze_directory("/path/to/audio/directory")
    # for result in results:
    #     if result['success']:
    #         print(f"{result['file_name']}: {result['overall_score']:.3f}")
    
    # Example 3: Create synthetic audio for testing
    print("\n🧪 Synthetic audio test:")
    
    # Create some test audio
    sample_rate = 44100
    duration = 2.0  # seconds
    length = int(sample_rate * duration)
    
    # Real-ish audio (sine wave with some noise)
    t = torch.linspace(0, duration, length)
    real_audio = torch.sin(2 * np.pi * 440 * t) + 0.1 * torch.randn(length)
    real_audio = real_audio.unsqueeze(0).unsqueeze(0)  # [1, 1, length]
    
    # Fake audio (just noise)
    fake_audio = torch.randn(1, 1, length)
    
    # Compare
    comparison = discriminator.compare_audio_quality(real_audio, fake_audio)
    print(f"Real audio score: {comparison['real_score']:.3f}")
    print(f"Fake audio score: {comparison['generated_score']:.3f}")
    print(f"Quality ratio: {comparison['quality_ratio']:.3f}")


if __name__ == "__main__":
    demo_usage()