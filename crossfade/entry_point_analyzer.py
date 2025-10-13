"""
EntryPointAnalyzer - Find optimal entry points in Track B intro

This module identifies the best places to enter Track B, particularly
looking for "drop on the 1" opportunities and build-up detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from dataclasses import dataclass

from .beat_grid_extractor import BeatGrid
from .energy_profile_extractor import EnergyProfile
from .musical_key_extractor import KeyProfile


@dataclass
class EntryCandidate:
    """Single entry point candidate"""
    sample_position: int        # Sample position for entry
    drop_score: float          # "Drop on the 1" quality [0,1]
    energy_level: float        # Energy level at entry [0,1]
    buildup_score: float       # Build-up leading to entry [0,1]
    phrase_start_score: float  # Phrase beginning quality [0,1]
    beat_alignment: float      # Beat alignment quality [0,1]
    overall_score: float       # Combined quality score [0,1]


@dataclass
class EntryCandidates:
    """Collection of entry point options"""
    positions: List[EntryCandidate]     # Ranked entry candidates
    best_entry: Optional[EntryCandidate]  # Highest scoring option
    drop_opportunities: List[EntryCandidate]  # "Drop on the 1" candidates
    intro_character: str                # Character classification


class EntryPointAnalyzer(nn.Module):
    """
    Find optimal entry points in Track B intro.
    
    Focuses on finding "drop on the 1" opportunities and
    build-up detection for energy-matched transitions.
    """
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 hop_length: int = 512,
                 min_entry_candidates: int = 3,
                 max_entry_candidates: int = 10):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.min_entry_candidates = min_entry_candidates
        self.max_entry_candidates = max_entry_candidates
        
        # Neural network for "drop" detection
        self.drop_detector = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=11, padding=5),  # Energy + onset features
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Build-up pattern detection
        self.buildup_detector = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, padding=7),  # Longer context for build-ups
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Entry quality assessment
        self.entry_assessor = nn.Sequential(
            nn.Linear(6, 32),  # 6 input features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, 
                audio: torch.Tensor,
                beat_grid: BeatGrid,
                energy_profile: EnergyProfile,
                key_profile: KeyProfile) -> EntryCandidates:
        """
        Find optimal entry points in Track B intro.
        
        Args:
            audio: Track B intro audio (30s)
            beat_grid: Beat detection results
            energy_profile: Energy analysis results
            key_profile: Key detection results
            
        Returns:
            EntryCandidates with ranked entry options
        """
        
        # Find structural entry candidates
        structural_candidates = self._find_structural_candidates(
            beat_grid, energy_profile
        )
        
        # Detect "drop on the 1" opportunities
        drop_scores = self._detect_drop_opportunities(audio, energy_profile)
        
        # Detect build-up patterns
        buildup_scores = self._detect_buildup_patterns(energy_profile)
        
        # Score all candidate positions
        scored_candidates = []
        for candidate_pos in structural_candidates:
            candidate = self._score_entry_candidate(
                candidate_pos, audio, beat_grid, energy_profile,
                key_profile, drop_scores, buildup_scores
            )
            if candidate.overall_score > 0.1:  # Filter very low quality
                scored_candidates.append(candidate)
        
        # Sort by overall score
        scored_candidates.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Limit candidates
        final_candidates = scored_candidates[:self.max_entry_candidates]
        
        # Generate backup candidates if needed
        if len(final_candidates) < self.min_entry_candidates:
            backup_candidates = self._generate_backup_candidates(
                beat_grid, energy_profile
            )
            final_candidates.extend(backup_candidates)
            final_candidates = final_candidates[:self.max_entry_candidates]
        
        # Find best drop opportunities
        drop_opportunities = [
            c for c in final_candidates if c.drop_score > 0.6
        ]
        drop_opportunities.sort(key=lambda x: x.drop_score, reverse=True)
        
        # Classify intro character
        intro_character = self._classify_intro_character(energy_profile, drop_opportunities)
        
        best_entry = final_candidates[0] if final_candidates else None
        
        return EntryCandidates(
            positions=final_candidates,
            best_entry=best_entry,
            drop_opportunities=drop_opportunities,
            intro_character=intro_character
        )
    
    def _find_structural_candidates(self, 
                                   beat_grid: BeatGrid,
                                   energy_profile: EnergyProfile) -> List[int]:
        """Find candidate positions based on musical structure"""
        
        candidates = []
        
        # Downbeat positions (strongest entry points)
        if len(beat_grid.bar_times) > 0:
            for bar_pos in beat_grid.bar_times:
                candidates.append(int(bar_pos))
        
        # Strong beat positions (1st and 3rd beats in 4/4)
        if len(beat_grid.beat_times) > 0:
            for i, beat_pos in enumerate(beat_grid.beat_times):
                beat_in_bar = i % 4
                if beat_in_bar in [0, 2]:  # 1st and 3rd beats
                    candidates.append(int(beat_pos))
        
        # Energy peaks (potential drop points)
        for peak_pos in energy_profile.peak_positions:
            candidates.append(int(peak_pos))
        
        # Start of intro (first few seconds)
        intro_start_samples = int(2.0 * self.sample_rate)  # First 2 seconds
        if len(beat_grid.beat_times) > 0:
            early_beats = [b for b in beat_grid.beat_times if int(b) < intro_start_samples]
            candidates.extend([int(b) for b in early_beats])
        
        # Remove duplicates and sort
        candidates = sorted(list(set(candidates)))
        
        # Filter to intro region (first 75% of track)
        audio_length = len(energy_profile.rms_curve) * self.hop_length
        intro_end = int(0.75 * audio_length)
        candidates = [c for c in candidates if c <= intro_end]
        
        return candidates
    
    def _detect_drop_opportunities(self, 
                                  audio: torch.Tensor,
                                  energy_profile: EnergyProfile) -> torch.Tensor:
        """Detect "drop on the 1" opportunities using neural network"""
        
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        audio_np = audio[0].detach().cpu().numpy()
        
        # Extract onset strength
        import librosa
        onset_strength = librosa.onset.onset_strength(
            y=audio_np,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # Combine energy and onset features
        energy_curve = energy_profile.rms_curve.numpy()
        
        # Ensure same length
        min_len = min(len(onset_strength), len(energy_curve))
        onset_strength = onset_strength[:min_len]
        energy_curve = energy_curve[:min_len]
        
        # Normalize features
        onset_norm = (onset_strength - np.mean(onset_strength)) / (np.std(onset_strength) + 1e-8)
        energy_norm = (energy_curve - np.mean(energy_curve)) / (np.std(energy_curve) + 1e-8)
        
        # Stack features
        features = torch.from_numpy(np.vstack([energy_norm, onset_norm])).float()
        features = features.unsqueeze(0)  # Add batch dimension
        
        # Detect drops
        with torch.no_grad():
            drop_scores = self.drop_detector(features)
            
        return drop_scores.squeeze()
    
    def _detect_buildup_patterns(self, energy_profile: EnergyProfile) -> torch.Tensor:
        """Detect build-up patterns that lead to good entry points"""
        
        # Use energy slope as input feature
        energy_slope = energy_profile.energy_slope
        
        # Normalize
        slope_norm = (energy_slope - energy_slope.mean()) / (energy_slope.std() + 1e-8)
        slope_input = slope_norm.unsqueeze(0).unsqueeze(0)  # (1, 1, time)
        
        # Detect build-ups
        with torch.no_grad():
            buildup_scores = self.buildup_detector(slope_input)
            
        return buildup_scores.squeeze()
    
    def _score_entry_candidate(self, 
                              sample_pos: int,
                              audio: torch.Tensor,
                              beat_grid: BeatGrid,
                              energy_profile: EnergyProfile,
                              key_profile: KeyProfile,
                              drop_scores: torch.Tensor,
                              buildup_scores: torch.Tensor) -> EntryCandidate:
        """Score a single entry candidate position"""
        
        # Convert to frame position
        frame_pos = int(sample_pos // self.hop_length)
        frame_pos = max(0, min(frame_pos, len(drop_scores) - 1))
        
        # Drop score
        drop_score = float(drop_scores[frame_pos])
        
        # Build-up score
        buildup_score = float(buildup_scores[frame_pos])
        
        # Beat alignment score
        beat_alignment = self._calculate_beat_alignment(sample_pos, beat_grid)
        
        # Energy level appropriateness
        energy_level = self._calculate_energy_appropriateness(
            frame_pos, energy_profile
        )
        
        # Phrase start score
        phrase_start_score = self._calculate_phrase_start_score(
            frame_pos, energy_profile, buildup_score
        )
        
        # Intro position score (prefer earlier positions)
        intro_position_score = self._calculate_intro_position_score(
            sample_pos, len(audio[0]) if audio.dim() > 1 else len(audio)
        )
        
        # Combine scores using neural network
        features = torch.tensor([
            drop_score,
            buildup_score,
            beat_alignment,
            energy_level,
            phrase_start_score,
            intro_position_score
        ], dtype=torch.float32)
        
        with torch.no_grad():
            overall_score = self.entry_assessor(features.unsqueeze(0))
            overall_score = float(overall_score.squeeze())
        
        return EntryCandidate(
            sample_position=sample_pos,
            drop_score=drop_score,
            energy_level=energy_level,
            buildup_score=buildup_score,
            phrase_start_score=phrase_start_score,
            beat_alignment=beat_alignment,
            overall_score=overall_score
        )
    
    def _calculate_beat_alignment(self, sample_pos: int, beat_grid: BeatGrid) -> float:
        """Calculate how well position aligns with strong beats"""
        
        if len(beat_grid.beat_times) == 0:
            return 0.5
        
        # Find nearest beat
        distances = torch.abs(beat_grid.beat_times - sample_pos)
        min_distance_idx = torch.argmin(distances)
        min_distance = distances[min_distance_idx]
        
        # Check if this is a strong beat (downbeat or beat 1 and 3)
        beat_index = int(min_distance_idx)
        beat_strength = 1.0  # Default strength
        
        if len(beat_grid.bar_times) > 0:
            # Check if this beat is a downbeat
            bar_distances = torch.abs(beat_grid.bar_times - beat_grid.beat_times[beat_index])
            if torch.min(bar_distances) < self.hop_length:  # Very close to a bar
                beat_strength = 1.0  # Strongest
        else:
            # Use beat position in assumed 4/4 pattern
            beat_in_bar = beat_index % 4
            if beat_in_bar == 0:
                beat_strength = 1.0  # Downbeat
            elif beat_in_bar == 2:
                beat_strength = 0.8  # Beat 3
            else:
                beat_strength = 0.6  # Weak beats
        
        # Distance-based alignment score
        tolerance_samples = int(0.05 * self.sample_rate)  # 50ms tolerance
        if min_distance <= tolerance_samples:
            distance_score = 1.0
        else:
            max_distance = tolerance_samples * 4
            distance_score = max(0.0, 1.0 - float(min_distance) / max_distance)
        
        return beat_strength * distance_score
    
    def _calculate_energy_appropriateness(self, 
                                        frame_pos: int,
                                        energy_profile: EnergyProfile) -> float:
        """Calculate energy appropriateness for entry"""
        
        if frame_pos >= len(energy_profile.rms_curve):
            return 0.0
        
        current_energy = energy_profile.rms_curve[frame_pos]
        
        # For entries, we prefer higher energy (opposite of exits)
        max_energy = torch.max(energy_profile.rms_curve)
        min_energy = torch.min(energy_profile.rms_curve)
        
        if max_energy > min_energy:
            normalized_energy = (current_energy - min_energy) / (max_energy - min_energy)
        else:
            normalized_energy = 0.5
        
        # Prefer higher energy for entries, but not exclusively
        energy_score = float(normalized_energy) * 0.8 + 0.2  # Minimum score of 0.2
        
        return min(1.0, max(0.0, energy_score))
    
    def _calculate_phrase_start_score(self, 
                                    frame_pos: int,
                                    energy_profile: EnergyProfile,
                                    buildup_score: float) -> float:
        """Calculate phrase start appropriateness"""
        
        # Use energy slope to detect phrase starts
        if frame_pos >= len(energy_profile.energy_slope):
            return 0.5
        
        energy_slope = energy_profile.energy_slope[frame_pos]
        
        # Phrase starts often have positive energy slope (building energy)
        slope_score = max(0.0, float(energy_slope)) * 0.5 + 0.3  # Base score 0.3
        
        # Combine with build-up score
        phrase_score = 0.6 * slope_score + 0.4 * buildup_score
        
        return min(1.0, max(0.0, phrase_score))
    
    def _calculate_intro_position_score(self, sample_pos: int, total_length: int) -> float:
        """Score based on position in intro (prefer earlier positions)"""
        
        relative_pos = sample_pos / total_length
        
        # Prefer positions in first 60% of track
        if relative_pos < 0.4:
            return 1.0  # Perfect for early intro
        elif relative_pos < 0.6:
            # Gradual decrease
            return 1.0 - (relative_pos - 0.4) / 0.2 * 0.5
        else:
            # Lower preference for later positions
            return 0.5 - (relative_pos - 0.6) / 0.4 * 0.3
    
    def _generate_backup_candidates(self, 
                                   beat_grid: BeatGrid,
                                   energy_profile: EnergyProfile) -> List[EntryCandidate]:
        """Generate backup candidates if not enough found"""
        
        backup_candidates = []
        
        # Use early beats as backup
        if len(beat_grid.beat_times) > 0:
            audio_length = len(energy_profile.rms_curve) * self.hop_length
            intro_end = int(0.5 * audio_length)
            
            intro_beats = [b for b in beat_grid.beat_times if int(b) <= intro_end]
            
            for beat_pos in intro_beats[:3]:
                backup_candidates.append(EntryCandidate(
                    sample_position=int(beat_pos),
                    drop_score=0.3,
                    energy_level=0.5,
                    buildup_score=0.3,
                    phrase_start_score=0.4,
                    beat_alignment=0.8,  # Good beat alignment
                    overall_score=0.4
                ))
        
        return backup_candidates
    
    def _classify_intro_character(self, 
                                 energy_profile: EnergyProfile,
                                 drop_opportunities: List[EntryCandidate]) -> str:
        """Classify the character of the intro"""
        
        # Analyze energy pattern in first half
        intro_length = len(energy_profile.energy_slope) // 2
        intro_slope = energy_profile.energy_slope[:intro_length]
        
        avg_slope = torch.mean(intro_slope)
        
        # Check for drops
        has_strong_drops = len(drop_opportunities) > 0 and drop_opportunities[0].drop_score > 0.7
        
        if has_strong_drops:
            return 'drop_heavy'
        elif avg_slope > 0.4:
            return 'building'
        elif avg_slope < -0.2:
            return 'fade_in'
        else:
            # Check energy level
            avg_energy = torch.mean(energy_profile.rms_curve[:intro_length])
            total_avg_energy = torch.mean(energy_profile.rms_curve)
            
            if avg_energy > total_avg_energy:
                return 'high_energy'
            else:
                return 'steady'


def create_test_audio_with_drop(duration: float = 30.0, 
                               drop_time: float = 8.0,
                               sr: int = 44100) -> torch.Tensor:
    """Create test audio with a clear drop for testing"""
    
    samples = int(duration * sr)
    drop_sample = int(drop_time * sr)
    
    # Build-up phase (increasing energy and frequency)
    buildup_samples = drop_sample
    t_buildup = torch.linspace(0, drop_time, buildup_samples)
    
    # Rising frequency and amplitude
    freq_buildup = 200 + 300 * t_buildup / drop_time  # 200Hz to 500Hz
    amp_buildup = 0.3 * t_buildup / drop_time  # Increasing amplitude
    
    buildup_audio = amp_buildup * torch.sin(2 * torch.pi * freq_buildup * t_buildup)
    
    # Add some noise for texture
    buildup_audio += 0.1 * torch.randn(buildup_samples) * amp_buildup
    
    # Drop phase (sudden energy jump)
    drop_samples = samples - drop_sample
    t_drop = torch.linspace(0, (duration - drop_time), drop_samples)
    
    # Strong, consistent energy after drop
    drop_freq = 440.0  # Clear fundamental
    drop_audio = 0.8 * torch.sin(2 * torch.pi * drop_freq * t_drop)
    
    # Add harmonics for richness
    drop_audio += 0.3 * torch.sin(2 * torch.pi * 2 * drop_freq * t_drop)
    drop_audio += 0.1 * torch.sin(2 * torch.pi * 3 * drop_freq * t_drop)
    
    # Add some decay over time
    decay = torch.exp(-0.5 * t_drop)
    drop_audio *= decay
    
    # Combine buildup and drop
    audio = torch.zeros(samples)
    audio[:buildup_samples] = buildup_audio
    audio[buildup_samples:] = drop_audio
    
    return audio


if __name__ == "__main__":
    # Test with dummy data
    from .beat_grid_extractor import BeatGridExtractor
    from .energy_profile_extractor import EnergyProfileExtractor
    from .musical_key_extractor import MusicalKeyExtractor
    
    # Create test audio with clear drop
    test_audio = create_test_audio_with_drop(duration=30.0, drop_time=8.0)
    print(f"Test audio shape: {test_audio.shape}")
    
    # Create analyzers
    beat_extractor = BeatGridExtractor()
    energy_extractor = EnergyProfileExtractor()
    key_extractor = MusicalKeyExtractor()
    entry_analyzer = EntryPointAnalyzer()
    
    # Extract features
    beat_grid = beat_extractor(test_audio)
    energy_profile = energy_extractor(test_audio)
    key_profile = key_extractor(test_audio)
    
    # Find entry points
    entry_candidates = entry_analyzer(test_audio, beat_grid, energy_profile, key_profile)
    
    print(f"Found {len(entry_candidates.positions)} entry candidates")
    print(f"Found {len(entry_candidates.drop_opportunities)} drop opportunities")
    print(f"Intro character: {entry_candidates.intro_character}")
    
    if entry_candidates.best_entry:
        print(f"Best entry at sample {entry_candidates.best_entry.sample_position}")
        print(f"Best entry drop score: {entry_candidates.best_entry.drop_score:.3f}")
        print(f"Best entry overall score: {entry_candidates.best_entry.overall_score:.3f}")
    
    if entry_candidates.drop_opportunities:
        best_drop = entry_candidates.drop_opportunities[0]
        print(f"Best drop at sample {best_drop.sample_position}")
        print(f"Drop score: {best_drop.drop_score:.3f}")