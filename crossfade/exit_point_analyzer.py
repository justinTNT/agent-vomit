"""
ExitPointAnalyzer - Find optimal exit points in Track A outro

This module identifies musically sensible places to exit Track A,
considering phrase boundaries, energy levels, and beat alignment.
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
class ExitCandidate:
    """Single exit point candidate"""
    sample_position: int        # Sample position for exit
    musical_score: float        # Musical appropriateness [0,1]
    energy_level: float         # Energy level at exit point [0,1]
    beat_alignment: float       # How well aligned with beat [0,1]
    phrase_completion: float    # Phrase boundary score [0,1]
    overall_score: float        # Combined quality score [0,1]


@dataclass
class ExitCandidates:
    """Collection of exit point options"""
    positions: List[ExitCandidate]    # Ranked exit candidates
    best_exit: Optional[ExitCandidate]  # Highest scoring option
    musical_context: dict              # Context for B track matching


class ExitPointAnalyzer(nn.Module):
    """
    Find optimal exit points in Track A outro.
    
    Analyzes musical structure, energy patterns, and beat grid
    to identify natural, musically appropriate exit points.
    """
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 hop_length: int = 512,
                 min_exit_candidates: int = 3,
                 max_exit_candidates: int = 10):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.min_exit_candidates = min_exit_candidates
        self.max_exit_candidates = max_exit_candidates
        
        # Neural network for phrase boundary detection
        self.phrase_detector = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Musical quality assessment network
        self.quality_assessor = nn.Sequential(
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
                key_profile: KeyProfile) -> ExitCandidates:
        """
        Find optimal exit points in Track A outro.
        
        Args:
            audio: Track A outro audio (30s)
            beat_grid: Beat detection results
            energy_profile: Energy analysis results  
            key_profile: Key detection results
            
        Returns:
            ExitCandidates with ranked exit options
        """
        
        # Find candidate positions based on musical structure
        structural_candidates = self._find_structural_candidates(
            beat_grid, energy_profile
        )
        
        # Analyze phrase boundaries
        phrase_boundaries = self._detect_phrase_boundaries(audio)
        
        # Score all candidate positions
        scored_candidates = []
        for candidate_pos in structural_candidates:
            candidate = self._score_exit_candidate(
                candidate_pos, audio, beat_grid, energy_profile, 
                key_profile, phrase_boundaries
            )
            if candidate.overall_score > 0.1:  # Filter very low quality
                scored_candidates.append(candidate)
        
        # Sort by overall score
        scored_candidates.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Limit to max candidates and ensure minimum
        final_candidates = scored_candidates[:self.max_exit_candidates]
        
        if len(final_candidates) < self.min_exit_candidates:
            # Add backup candidates if needed
            backup_candidates = self._generate_backup_candidates(
                beat_grid, energy_profile
            )
            final_candidates.extend(backup_candidates)
            final_candidates = final_candidates[:self.max_exit_candidates]
        
        # Extract musical context
        musical_context = self._extract_musical_context(
            key_profile, energy_profile, beat_grid
        )
        
        best_exit = final_candidates[0] if final_candidates else None
        
        return ExitCandidates(
            positions=final_candidates,
            best_exit=best_exit,
            musical_context=musical_context
        )
    
    def _find_structural_candidates(self, 
                                   beat_grid: BeatGrid,
                                   energy_profile: EnergyProfile) -> List[int]:
        """Find candidate positions based on musical structure"""
        
        candidates = []
        
        # Bar boundaries (strong structural points)
        if len(beat_grid.bar_times) > 0:
            for bar_pos in beat_grid.bar_times:
                candidates.append(int(bar_pos))
        
        # Beat boundaries (weaker but still valid)
        if len(beat_grid.beat_times) > 0:
            # Take every 4th beat if no bars detected
            if len(beat_grid.bar_times) == 0:
                for i in range(0, len(beat_grid.beat_times), 4):
                    candidates.append(int(beat_grid.beat_times[i]))
        
        # Energy-based candidates (quiet sections)
        quiet_sections = self._find_quiet_sections_for_exit(energy_profile)
        for start, end in quiet_sections:
            candidates.append(start)
            candidates.append(end)
        
        # Remove duplicates and sort
        candidates = sorted(list(set(candidates)))
        
        # Filter to outro region (last 75% of track)
        audio_length = len(energy_profile.rms_curve) * self.hop_length
        outro_start = int(0.25 * audio_length)
        candidates = [c for c in candidates if c >= outro_start]
        
        return candidates
    
    def _detect_phrase_boundaries(self, audio: torch.Tensor) -> torch.Tensor:
        """Detect phrase boundaries using neural network"""
        
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Extract onset strength as input feature
        audio_np = audio[0].detach().cpu().numpy()
        import librosa
        
        onset_strength = librosa.onset.onset_strength(
            y=audio_np,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # Prepare input for neural network
        onset_tensor = torch.from_numpy(onset_strength).float()
        onset_tensor = (onset_tensor - onset_tensor.mean()) / (onset_tensor.std() + 1e-8)
        onset_input = onset_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, time)
        
        # Predict phrase boundaries
        with torch.no_grad():
            phrase_scores = self.phrase_detector(onset_input)
            
        return phrase_scores.squeeze()
    
    def _score_exit_candidate(self, 
                             sample_pos: int,
                             audio: torch.Tensor,
                             beat_grid: BeatGrid, 
                             energy_profile: EnergyProfile,
                             key_profile: KeyProfile,
                             phrase_boundaries: torch.Tensor) -> ExitCandidate:
        """Score a single exit candidate position"""
        
        # Convert sample position to frame for analysis
        frame_pos = int(sample_pos // self.hop_length)
        frame_pos = max(0, min(frame_pos, len(phrase_boundaries) - 1))
        
        # Beat alignment score
        beat_alignment = self._calculate_beat_alignment(sample_pos, beat_grid)
        
        # Energy appropriateness
        energy_level = self._calculate_energy_appropriateness(
            frame_pos, energy_profile
        )
        
        # Phrase completion score
        phrase_completion = float(phrase_boundaries[frame_pos])
        
        # Musical context score
        musical_score = self._calculate_musical_score(
            sample_pos, key_profile, energy_profile
        )
        
        # Zero-crossing score (for clean cuts)
        zero_crossing_score = self._calculate_zero_crossing_score(
            sample_pos, audio
        )
        
        # Outro position score (prefer later in track)
        outro_score = self._calculate_outro_position_score(
            sample_pos, len(audio[0]) if audio.dim() > 1 else len(audio)
        )
        
        # Combine scores using neural network
        features = torch.tensor([
            beat_alignment,
            energy_level, 
            phrase_completion,
            musical_score,
            zero_crossing_score,
            outro_score
        ], dtype=torch.float32)
        
        with torch.no_grad():
            overall_score = self.quality_assessor(features.unsqueeze(0))
            overall_score = float(overall_score.squeeze())
        
        return ExitCandidate(
            sample_position=sample_pos,
            musical_score=musical_score,
            energy_level=energy_level,
            beat_alignment=beat_alignment,
            phrase_completion=phrase_completion,
            overall_score=overall_score
        )
    
    def _calculate_beat_alignment(self, sample_pos: int, beat_grid: BeatGrid) -> float:
        """Calculate how well position aligns with beats"""
        
        if len(beat_grid.beat_times) == 0:
            return 0.5  # Neutral if no beats detected
        
        # Find nearest beat
        distances = torch.abs(beat_grid.beat_times - sample_pos)
        min_distance = torch.min(distances)
        
        # Score based on distance (closer = better)
        # Allow 50ms tolerance for "perfect" alignment
        tolerance_samples = int(0.05 * self.sample_rate)
        
        if min_distance <= tolerance_samples:
            return 1.0
        else:
            # Decay score with distance
            max_distance = tolerance_samples * 4  # 200ms max
            score = max(0.0, 1.0 - float(min_distance) / max_distance)
            return score
    
    def _calculate_energy_appropriateness(self, 
                                        frame_pos: int,
                                        energy_profile: EnergyProfile) -> float:
        """Calculate energy appropriateness for exit"""
        
        if frame_pos >= len(energy_profile.rms_curve):
            return 0.0
        
        # Get energy at position
        current_energy = energy_profile.rms_curve[frame_pos]
        
        # Prefer moderate to low energy for exit
        # Convert from dB to linear scale
        energy_linear = torch.pow(10.0, current_energy / 20.0)
        
        # Score: prefer energy in lower 60% range
        max_energy = torch.max(energy_profile.rms_curve)
        min_energy = torch.min(energy_profile.rms_curve)
        
        # Normalize to [0, 1]
        if max_energy > min_energy:
            normalized_energy = (current_energy - min_energy) / (max_energy - min_energy)
        else:
            normalized_energy = 0.5
        
        # Prefer lower energy for exits (inverted score)
        energy_score = 1.0 - float(normalized_energy) * 0.7  # Don't penalize too heavily
        
        return max(0.0, min(1.0, energy_score))
    
    def _calculate_musical_score(self, 
                               sample_pos: int,
                               key_profile: KeyProfile,
                               energy_profile: EnergyProfile) -> float:
        """Calculate overall musical appropriateness"""
        
        # Key stability score
        key_stable = False
        for start, end in key_profile.stability_zones:
            if start <= sample_pos <= end:
                key_stable = True
                break
        
        stability_score = 0.8 if key_stable else 0.4
        
        # Harmonic relevance (if low, position matters less)
        relevance_weight = key_profile.harmonic_relevance
        
        # Combined musical score
        musical_score = stability_score * relevance_weight + (1 - relevance_weight) * 0.6
        
        return min(1.0, max(0.0, musical_score))
    
    def _calculate_zero_crossing_score(self, sample_pos: int, audio: torch.Tensor) -> float:
        """Score based on zero-crossing for clean cuts"""
        
        if audio.dim() > 1:
            audio = audio[0]
        
        if sample_pos >= len(audio) - 1 or sample_pos <= 0:
            return 0.0
        
        # Check for zero crossing within small window
        window = min(100, len(audio) - sample_pos - 1)  # ~2ms window
        start_idx = max(0, sample_pos - window // 2)
        end_idx = min(len(audio), sample_pos + window // 2)
        
        audio_segment = audio[start_idx:end_idx]
        
        # Find zero crossings
        zero_crossings = torch.where(
            torch.diff(torch.sign(audio_segment)) != 0
        )[0]
        
        if len(zero_crossings) > 0:
            # Find closest zero crossing to our position
            target_pos = sample_pos - start_idx
            distances = torch.abs(zero_crossings - target_pos)
            min_distance = torch.min(distances)
            
            # Score based on proximity to zero crossing
            score = max(0.0, 1.0 - float(min_distance) / window)
            return score
        
        return 0.2  # Low score if no zero crossings found
    
    def _calculate_outro_position_score(self, sample_pos: int, total_length: int) -> float:
        """Score based on position in outro (prefer later positions)"""
        
        # Calculate relative position
        relative_pos = sample_pos / total_length
        
        # Prefer positions in last 50% of track, with peak preference at 75%
        if relative_pos < 0.5:
            return 0.2
        elif relative_pos < 0.75:
            # Linear increase from 0.5 to 0.75
            return 0.2 + 0.6 * (relative_pos - 0.5) / 0.25
        elif relative_pos < 0.9:
            # Slight decrease after 75%
            return 0.8 + 0.2 * (0.9 - relative_pos) / 0.15
        else:
            # Don't cut too close to the end
            return 0.6
    
    def _find_quiet_sections_for_exit(self, energy_profile: EnergyProfile) -> List[tuple[int, int]]:
        """Find quiet sections suitable for exits"""
        
        # Use energy profile's quiet section detection
        # This is a simplified version - could import from energy_profile_extractor
        threshold_db = -30.0  # Stricter threshold for exits
        min_duration_ms = 200.0  # Shorter minimum for exit points
        
        quiet_mask = energy_profile.rms_curve < threshold_db
        min_frames = int(min_duration_ms * self.sample_rate / 1000 / self.hop_length)
        
        quiet_sections = []
        in_quiet = False
        start_frame = 0
        
        for i, is_quiet in enumerate(quiet_mask):
            if is_quiet and not in_quiet:
                start_frame = i
                in_quiet = True
            elif not is_quiet and in_quiet:
                if i - start_frame >= min_frames:
                    start_sample = start_frame * self.hop_length
                    end_sample = i * self.hop_length
                    quiet_sections.append((start_sample, end_sample))
                in_quiet = False
        
        # Check final section
        if in_quiet and len(quiet_mask) - start_frame >= min_frames:
            start_sample = start_frame * self.hop_length
            end_sample = len(quiet_mask) * self.hop_length
            quiet_sections.append((start_sample, end_sample))
        
        return quiet_sections
    
    def _generate_backup_candidates(self, 
                                   beat_grid: BeatGrid,
                                   energy_profile: EnergyProfile) -> List[ExitCandidate]:
        """Generate backup candidates if not enough found"""
        
        backup_candidates = []
        
        # Use regular beat positions as backup
        if len(beat_grid.beat_times) > 0:
            # Take beats in outro region
            audio_length = len(energy_profile.rms_curve) * self.hop_length
            outro_start = int(0.5 * audio_length)
            
            outro_beats = [b for b in beat_grid.beat_times if int(b) >= outro_start]
            
            for beat_pos in outro_beats[:3]:  # Max 3 backups
                backup_candidates.append(ExitCandidate(
                    sample_position=int(beat_pos),
                    musical_score=0.5,
                    energy_level=0.5, 
                    beat_alignment=1.0,  # Perfect beat alignment
                    phrase_completion=0.3,
                    overall_score=0.4  # Low but acceptable
                ))
        
        return backup_candidates
    
    def _extract_musical_context(self, 
                               key_profile: KeyProfile,
                               energy_profile: EnergyProfile,
                               beat_grid: BeatGrid) -> dict:
        """Extract musical context for Track B matching"""
        
        return {
            'key': key_profile.key,
            'mode': key_profile.mode.value,
            'key_confidence': key_profile.confidence,
            'harmonic_relevance': key_profile.harmonic_relevance,
            'bpm': beat_grid.bpm,
            'tempo_stability': beat_grid.tempo_stability,
            'avg_energy_db': float(torch.mean(energy_profile.rms_curve)),
            'energy_range_db': float(torch.max(energy_profile.rms_curve) - torch.min(energy_profile.rms_curve)),
            'outro_character': self._classify_outro_character(energy_profile)
        }
    
    def _classify_outro_character(self, energy_profile: EnergyProfile) -> str:
        """Classify the character of the outro"""
        
        # Analyze energy slope in outro region
        outro_length = len(energy_profile.energy_slope) // 4  # Last 25%
        outro_slope = energy_profile.energy_slope[-outro_length:]
        
        avg_slope = torch.mean(outro_slope)
        
        if avg_slope > 0.3:
            return 'building'
        elif avg_slope < -0.3:
            return 'fading'
        else:
            return 'stable'


if __name__ == "__main__":
    # Test with dummy data
    from .beat_grid_extractor import BeatGridExtractor, create_test_audio
    from .energy_profile_extractor import EnergyProfileExtractor
    from .musical_key_extractor import MusicalKeyExtractor
    
    # Create test audio
    test_audio = create_test_audio(duration=30.0, bpm=120.0)
    print(f"Test audio shape: {test_audio.shape}")
    
    # Create analyzers
    beat_extractor = BeatGridExtractor()
    energy_extractor = EnergyProfileExtractor()
    key_extractor = MusicalKeyExtractor()
    exit_analyzer = ExitPointAnalyzer()
    
    # Extract features
    beat_grid = beat_extractor(test_audio)
    energy_profile = energy_extractor(test_audio)
    key_profile = key_extractor(test_audio)
    
    # Find exit points
    exit_candidates = exit_analyzer(test_audio, beat_grid, energy_profile, key_profile)
    
    print(f"Found {len(exit_candidates.positions)} exit candidates")
    if exit_candidates.best_exit:
        print(f"Best exit at sample {exit_candidates.best_exit.sample_position}")
        print(f"Best exit score: {exit_candidates.best_exit.overall_score:.3f}")
    
    print(f"Musical context: {exit_candidates.musical_context['outro_character']} outro")