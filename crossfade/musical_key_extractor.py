"""
MusicalKeyExtractor - Extract musical key with confidence and stability

This module detects the musical key of audio segments and determines
harmonic relevance for crossfade processing decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum


class Mode(Enum):
    """Musical modes"""
    MAJOR = "major"
    MINOR = "minor"
    ATONAL = "atonal"


@dataclass
class KeyProfile:
    """Musical key analysis results"""
    key: str                    # Key name (C, C#, D, etc.)
    mode: Mode                  # Major, minor, or atonal
    confidence: float           # Key detection confidence [0,1]
    stability_zones: List[tuple[int, int]]  # Sample ranges where key is stable
    harmonic_relevance: float   # How much does pitch matter [0,1]
    chroma_vector: torch.Tensor # 12-dimensional chroma features
    key_strength: torch.Tensor  # Strength of each key candidate [24,]


class MusicalKeyExtractor(nn.Module):
    """
    Extract musical key with confidence and harmonic relevance.
    
    Uses chromagram analysis combined with neural key detection
    for robust key identification across different musical styles.
    """
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 hop_length: int = 512,
                 n_fft: int = 4096,
                 n_chroma: int = 12):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_chroma = n_chroma
        
        # Key names for output
        self.key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 
                         'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Krumhansl-Schmuckler key profiles
        self.major_profile = torch.tensor([
            6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 
            5.19, 2.39, 3.66, 2.29, 2.88
        ], dtype=torch.float32)
        
        self.minor_profile = torch.tensor([
            6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54,
            4.75, 3.98, 2.69, 3.34, 3.17
        ], dtype=torch.float32)
        
        # Neural network for harmonic relevance detection
        self.relevance_encoder = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Neural key classifier for confidence estimation
        self.key_classifier = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 24)  # 12 major + 12 minor keys
        )

    def forward(self, audio: torch.Tensor) -> KeyProfile:
        """
        Extract musical key from audio segment.
        
        Args:
            audio: (batch_size, samples) or (samples,) mono audio
            
        Returns:
            KeyProfile with key, confidence, and harmonic relevance
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            
        batch_size = audio.size(0)
        
        # Process single audio for now (extend for batch later)
        audio_np = audio[0].detach().cpu().numpy()
        
        # Extract chromagram
        chroma = self._extract_chroma(audio_np)
        chroma_tensor = torch.from_numpy(chroma).float()
        
        # Average chroma over time for key detection
        avg_chroma = torch.mean(chroma_tensor, dim=1)
        
        # Traditional key detection using Krumhansl-Schmuckler
        key, mode, confidence = self._krumhansl_schmuckler(avg_chroma)
        
        # Neural key classification for confidence refinement
        key_strengths = self.key_classifier(avg_chroma.unsqueeze(0))
        key_strengths = F.softmax(key_strengths, dim=1).squeeze()
        
        # Harmonic relevance detection
        harmonic_relevance = self._calculate_harmonic_relevance(chroma_tensor)
        
        # Key stability analysis
        stability_zones = self._analyze_key_stability(chroma_tensor)
        
        return KeyProfile(
            key=key,
            mode=mode,
            confidence=float(confidence),
            stability_zones=stability_zones,
            harmonic_relevance=float(harmonic_relevance),
            chroma_vector=avg_chroma,
            key_strength=key_strengths
        )
    
    def _extract_chroma(self, audio: np.ndarray) -> np.ndarray:
        """Extract chromagram from audio"""
        
        # Use CQT-based chroma for better harmonic resolution
        chroma = librosa.feature.chroma_cqt(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_chroma=self.n_chroma,
            norm=2
        )
        
        # Smooth chroma to reduce noise using scipy
        from scipy import ndimage
        chroma = ndimage.uniform_filter1d(chroma, size=5, axis=1)
        
        return chroma
    
    def _krumhansl_schmuckler(self, chroma: torch.Tensor) -> tuple[str, Mode, float]:
        """Traditional Krumhansl-Schmuckler key detection"""
        
        # Normalize chroma
        chroma_norm = F.normalize(chroma, p=2, dim=0)
        
        # Test all 24 keys (12 major + 12 minor)
        correlations = []
        
        # Major keys
        for shift in range(12):
            profile = torch.roll(self.major_profile, shift)
            profile_norm = F.normalize(profile, p=2, dim=0)
            corr = torch.dot(chroma_norm, profile_norm)
            correlations.append(('major', shift, corr.item()))
        
        # Minor keys
        for shift in range(12):
            profile = torch.roll(self.minor_profile, shift)
            profile_norm = F.normalize(profile, p=2, dim=0)
            corr = torch.dot(chroma_norm, profile_norm)
            correlations.append(('minor', shift, corr.item()))
        
        # Find best match
        best_mode, best_shift, best_corr = max(correlations, key=lambda x: x[2])
        
        # Convert to key name and mode
        key_name = self.key_names[best_shift]
        mode = Mode.MAJOR if best_mode == 'major' else Mode.MINOR
        
        # Confidence is correlation strength
        confidence = min(1.0, max(0.0, best_corr))
        
        # Check for atonality (low confidence)
        if confidence < 0.3:
            mode = Mode.ATONAL
            
        return key_name, mode, confidence
    
    def _calculate_harmonic_relevance(self, chroma: torch.Tensor) -> float:
        """Determine how much harmonic content matters"""
        
        # Use neural network for relevance detection
        chroma_input = chroma.unsqueeze(0)  # Add batch dim
        relevance = self.relevance_encoder(chroma_input)
        
        # Also use traditional measures
        # High harmonic relevance if:
        # 1. Strong tonal peaks in chroma
        # 2. Clear harmonic relationships
        # 3. Low noise/atonality
        
        # Measure chroma sparsity (fewer active notes = more tonal)
        chroma_mean = torch.mean(chroma, dim=1)
        sparsity = 1.0 - torch.sum(chroma_mean > 0.1) / 12.0
        
        # Measure peak prominence
        chroma_max = torch.max(chroma_mean)
        chroma_mean_val = torch.mean(chroma_mean)
        prominence = (chroma_max - chroma_mean_val) / (chroma_max + 1e-8)
        
        # Combine neural and traditional measures
        traditional_relevance = 0.6 * sparsity + 0.4 * prominence
        neural_relevance = relevance.item()
        
        final_relevance = 0.7 * neural_relevance + 0.3 * traditional_relevance
        
        return min(1.0, max(0.0, final_relevance))
    
    def _analyze_key_stability(self, chroma: torch.Tensor) -> List[tuple[int, int]]:
        """Analyze where the key is stable over time"""
        
        time_frames = chroma.size(1)
        window_size = max(10, time_frames // 10)  # Analyze in windows
        
        stability_zones = []
        current_key = None
        zone_start = 0
        
        for i in range(0, time_frames, window_size):
            end_idx = min(i + window_size, time_frames)
            window_chroma = torch.mean(chroma[:, i:end_idx], dim=1)
            
            # Detect key in this window
            key, mode, confidence = self._krumhansl_schmuckler(window_chroma)
            window_key = f"{key}_{mode.value}"
            
            # Check if key changed
            if current_key is None:
                current_key = window_key
                zone_start = i
            elif window_key != current_key or confidence < 0.4:
                # Key changed or became unstable, end current zone
                if i - zone_start > window_size:  # Only add significant zones
                    zone_end = i * self.hop_length
                    zone_start_samples = zone_start * self.hop_length
                    stability_zones.append((zone_start_samples, zone_end))
                
                current_key = window_key if confidence >= 0.4 else None
                zone_start = i
        
        # Add final zone
        if current_key is not None and time_frames - zone_start > window_size:
            zone_end = time_frames * self.hop_length
            zone_start_samples = zone_start * self.hop_length
            stability_zones.append((zone_start_samples, zone_end))
        
        return stability_zones
    
    def get_key_distance(self, key1: str, mode1: Mode, key2: str, mode2: Mode) -> int:
        """Calculate distance between two keys on circle of fifths"""
        
        # Circle of fifths order
        circle_of_fifths = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F']
        
        try:
            pos1 = circle_of_fifths.index(key1)
            pos2 = circle_of_fifths.index(key2)
        except ValueError:
            return 6  # Maximum distance for unknown keys
        
        # Calculate circular distance
        distance = min(abs(pos1 - pos2), 12 - abs(pos1 - pos2))
        
        # Adjust for mode differences
        if mode1 != mode2:
            distance += 1  # Minor penalty for mode mismatch
            
        return min(6, distance)  # Cap at maximum distance
    
    def is_harmonic_clash(self, key1: str, mode1: Mode, key2: str, mode2: Mode, 
                         threshold: int = 1) -> bool:
        """Determine if two keys will clash harmonically"""
        
        distance = self.get_key_distance(key1, mode1, key2, mode2)
        return distance <= threshold
    
    def calculate_pitch_shift_semitones(self, from_key: str, to_key: str) -> int:
        """Calculate semitone shift needed to go from one key to another"""
        
        from_idx = self.key_names.index(from_key)
        to_idx = self.key_names.index(to_key)
        
        # Calculate shortest path (could be up or down)
        shift = to_idx - from_idx
        
        # Normalize to [-6, 6] range
        if shift > 6:
            shift -= 12
        elif shift < -6:
            shift += 12
            
        return shift


def create_test_audio_with_key(duration: float = 10.0, 
                              key: str = 'C', 
                              mode: str = 'major',
                              sr: int = 44100) -> torch.Tensor:
    """Create test audio in specific key for testing"""
    
    # Major and minor scale intervals (semitones from root)
    major_scale = [0, 2, 4, 5, 7, 9, 11]
    minor_scale = [0, 2, 3, 5, 7, 8, 10]
    
    scale = major_scale if mode == 'major' else minor_scale
    
    # Root note MIDI number
    root_notes = {'C': 60, 'C#': 61, 'D': 62, 'D#': 63, 'E': 64, 'F': 65,
                 'F#': 66, 'G': 67, 'G#': 68, 'A': 69, 'A#': 70, 'B': 71}
    root_midi = root_notes[key]
    
    # Generate chord progression in the key
    samples = int(duration * sr)
    audio = torch.zeros(samples)
    
    chord_duration = int(2.0 * sr)  # 2-second chords
    
    for i in range(0, samples, chord_duration):
        # Simple I-V-vi-IV progression
        chord_degrees = [0, 4, 5, 3]  # I, V, vi, IV
        chord_idx = (i // chord_duration) % 4
        chord_root = scale[chord_degrees[chord_idx]]
        
        # Generate triad
        chord_notes = [chord_root, chord_root + 4, chord_root + 7]  # Major triad
        
        for note_offset in chord_notes:
            note_midi = root_midi + note_offset
            freq = 440.0 * (2 ** ((note_midi - 69) / 12))
            
            # Generate sine wave
            duration_samples = min(chord_duration, samples - i)
            t = torch.linspace(0, duration_samples / sr, duration_samples)
            note_audio = 0.3 * torch.sin(2 * torch.pi * freq * t)
            
            # Add envelope
            envelope = torch.exp(-2 * t)
            note_audio *= envelope
            
            audio[i:i+duration_samples] += note_audio
    
    return audio


if __name__ == "__main__":
    # Test the module
    extractor = MusicalKeyExtractor()
    
    # Test with audio in C major
    test_audio = create_test_audio_with_key(duration=10.0, key='C', mode='major')
    print(f"Test audio shape: {test_audio.shape}")
    
    # Extract key
    key_profile = extractor(test_audio)
    
    print(f"Detected key: {key_profile.key} {key_profile.mode.value}")
    print(f"Confidence: {key_profile.confidence:.3f}")
    print(f"Harmonic relevance: {key_profile.harmonic_relevance:.3f}")
    print(f"Stability zones: {len(key_profile.stability_zones)}")
    
    # Test key distance calculation
    distance = extractor.get_key_distance('C', Mode.MAJOR, 'G', Mode.MAJOR)
    print(f"Distance C major to G major: {distance}")
    
    # Test pitch shift calculation
    shift = extractor.calculate_pitch_shift_semitones('C', 'G')
    print(f"Semitone shift C to G: {shift}")