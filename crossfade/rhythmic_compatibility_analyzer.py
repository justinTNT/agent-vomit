"""
RhythmicCompatibilityAnalyzer - Analyze tempo compatibility between tracks

This module determines rhythmic compatibility and calculates required
tempo adjustments based on BPM analysis and beat alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .beat_grid_extractor import BeatGrid


class TempoCompatibility(Enum):
    """Tempo compatibility levels"""
    PERFECT = "perfect"         # Same or very close BPM
    EXCELLENT = "excellent"     # Within easy correction range
    GOOD = "good"              # Within acceptable correction range
    MARGINAL = "marginal"      # At limit of correction range
    INCOMPATIBLE = "incompatible"  # Exceeds correction limits


@dataclass
class RhythmicMatch:
    """Rhythmic compatibility analysis results"""
    compatibility_score: float      # Overall compatibility [0,1]
    compatibility_level: TempoCompatibility  # Qualitative assessment
    required_rate_change: float     # Tempo ratio for B (1.0 = no change)
    phase_offset: float            # Beat phase alignment offset [0,1]
    bpm_difference: float          # Raw BPM difference
    tempo_stability_factor: float  # How stable both tempos are [0,1]
    correction_difficulty: float   # Processing difficulty [0,1]
    processing_recommendation: str  # Recommended action


class RhythmicCompatibilityAnalyzer(nn.Module):
    """
    Analyze rhythmic compatibility between Track A and Track B.
    
    Determines optimal tempo corrections and assesses the feasibility
    of beat matching for crossfade decisions.
    """
    
    def __init__(self, 
                 tempo_tolerance_percent: float = 5.0,
                 min_bpm: float = 60.0,
                 max_bpm: float = 200.0,
                 phase_tolerance: float = 0.1):
        super().__init__()
        
        self.tempo_tolerance = tempo_tolerance_percent / 100.0  # Convert to ratio
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.phase_tolerance = phase_tolerance
        
        # Neural network for tempo stability analysis
        self.stability_analyzer = nn.Sequential(
            nn.Linear(4, 32),  # bpm, stability, confidence, variance
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Processing difficulty predictor
        self.difficulty_predictor = nn.Sequential(
            nn.Linear(5, 32),  # rate_change, bpm_diff, stability_a, stability_b, phase_offset
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, 
                beat_grid_a: BeatGrid,
                beat_grid_b: BeatGrid) -> RhythmicMatch:
        """
        Analyze rhythmic compatibility between two tracks.
        
        Args:
            beat_grid_a: Track A beat analysis
            beat_grid_b: Track B beat analysis
            
        Returns:
            RhythmicMatch with compatibility analysis and recommendations
        """
        
        # Calculate BPM difference
        bpm_difference = beat_grid_b.bpm - beat_grid_a.bpm
        
        # Calculate required rate change for Track B
        required_rate_change = beat_grid_a.bpm / beat_grid_b.bpm if beat_grid_b.bpm > 0 else 1.0
        
        # Check if within tolerance
        within_tolerance = abs(required_rate_change - 1.0) <= self.tempo_tolerance
        
        # Calculate phase offset
        phase_offset = self._calculate_phase_offset(beat_grid_a, beat_grid_b)
        
        # Assess tempo stability
        tempo_stability_factor = self._calculate_stability_factor(beat_grid_a, beat_grid_b)
        
        # Calculate compatibility score
        compatibility_score = self._calculate_compatibility_score(
            beat_grid_a, beat_grid_b, required_rate_change, phase_offset, tempo_stability_factor
        )
        
        # Determine compatibility level
        compatibility_level = self._determine_compatibility_level(
            required_rate_change, within_tolerance, tempo_stability_factor
        )
        
        # Predict correction difficulty
        correction_difficulty = self._predict_correction_difficulty(
            required_rate_change, abs(bpm_difference), 
            beat_grid_a.tempo_stability, beat_grid_b.tempo_stability, phase_offset
        )
        
        # Generate processing recommendation
        processing_recommendation = self._generate_processing_recommendation(
            compatibility_level, required_rate_change, tempo_stability_factor
        )
        
        return RhythmicMatch(
            compatibility_score=float(compatibility_score),
            compatibility_level=compatibility_level,
            required_rate_change=float(required_rate_change),
            phase_offset=float(phase_offset),
            bpm_difference=float(bpm_difference),
            tempo_stability_factor=float(tempo_stability_factor),
            correction_difficulty=float(correction_difficulty),
            processing_recommendation=processing_recommendation
        )
    
    def _calculate_phase_offset(self, beat_grid_a: BeatGrid, beat_grid_b: BeatGrid) -> float:
        """Calculate beat phase alignment offset"""
        
        if len(beat_grid_a.beat_times) == 0 or len(beat_grid_b.beat_times) == 0:
            return 0.5  # Neutral phase offset
        
        # Find average beat interval for each track
        if len(beat_grid_a.beat_times) > 1:
            avg_interval_a = torch.mean(torch.diff(beat_grid_a.beat_times.float()))
        else:
            avg_interval_a = 60.0 / beat_grid_a.bpm * beat_grid_a.sample_rate
        
        if len(beat_grid_b.beat_times) > 1:
            avg_interval_b = torch.mean(torch.diff(beat_grid_b.beat_times.float()))
        else:
            avg_interval_b = 60.0 / beat_grid_b.bpm * beat_grid_b.sample_rate
        
        # Calculate phase offset as fraction of beat interval
        # Use first beats as reference
        phase_diff = (beat_grid_b.beat_times[0] % avg_interval_b) - (beat_grid_a.beat_times[0] % avg_interval_a)
        phase_offset = abs(phase_diff) / avg_interval_a
        
        # Normalize to [0, 0.5] (0.5 is maximum offset)
        phase_offset = min(0.5, phase_offset % 1.0)
        if phase_offset > 0.5:
            phase_offset = 1.0 - phase_offset
            
        return float(phase_offset)
    
    def _calculate_stability_factor(self, beat_grid_a: BeatGrid, beat_grid_b: BeatGrid) -> float:
        """Calculate combined tempo stability factor"""
        
        features = torch.tensor([
            beat_grid_a.bpm,
            beat_grid_a.tempo_stability,
            torch.mean(beat_grid_a.confidence_curve) if len(beat_grid_a.confidence_curve) > 0 else 0.5,
            torch.std(beat_grid_a.confidence_curve) if len(beat_grid_a.confidence_curve) > 0 else 0.5
        ], dtype=torch.float32)
        
        # Normalize BPM feature
        features[0] = (features[0] - 120.0) / 60.0  # Normalize around 120 BPM
        
        try:
            with torch.no_grad():
                stability_a = self.stability_analyzer(features.unsqueeze(0))
                stability_a = float(stability_a.squeeze())
        except:
            stability_a = beat_grid_a.tempo_stability
        
        # Same for track B
        features_b = torch.tensor([
            beat_grid_b.bpm,
            beat_grid_b.tempo_stability,
            torch.mean(beat_grid_b.confidence_curve) if len(beat_grid_b.confidence_curve) > 0 else 0.5,
            torch.std(beat_grid_b.confidence_curve) if len(beat_grid_b.confidence_curve) > 0 else 0.5
        ], dtype=torch.float32)
        
        features_b[0] = (features_b[0] - 120.0) / 60.0
        
        try:
            with torch.no_grad():
                stability_b = self.stability_analyzer(features_b.unsqueeze(0))
                stability_b = float(stability_b.squeeze())
        except:
            stability_b = beat_grid_b.tempo_stability
        
        # Combined stability (geometric mean)
        combined_stability = (stability_a * stability_b) ** 0.5
        
        return combined_stability
    
    def _calculate_compatibility_score(self, 
                                     beat_grid_a: BeatGrid,
                                     beat_grid_b: BeatGrid,
                                     required_rate_change: float,
                                     phase_offset: float,
                                     tempo_stability_factor: float) -> float:
        """Calculate overall rhythmic compatibility score"""
        
        # Base score from rate change requirement
        rate_change_diff = abs(required_rate_change - 1.0)
        
        if rate_change_diff <= self.tempo_tolerance:
            # Within tolerance - score based on how close to perfect
            rate_score = 1.0 - (rate_change_diff / self.tempo_tolerance) * 0.3
        else:
            # Outside tolerance - score based on how far beyond
            excess = rate_change_diff - self.tempo_tolerance
            max_excess = 0.5  # Beyond 50% rate change is essentially impossible
            rate_score = max(0.0, 0.7 - (excess / max_excess) * 0.7)
        
        # Phase alignment score
        phase_score = 1.0 - (phase_offset / 0.5) * 0.2  # Max 20% penalty for phase
        
        # Tempo stability bonus
        stability_bonus = tempo_stability_factor * 0.2
        
        # Beat detection confidence
        conf_a = torch.mean(beat_grid_a.confidence_curve) if len(beat_grid_a.confidence_curve) > 0 else 0.7
        conf_b = torch.mean(beat_grid_b.confidence_curve) if len(beat_grid_b.confidence_curve) > 0 else 0.7
        confidence_factor = (conf_a + conf_b) / 2.0 * 0.1
        
        # Combine scores
        compatibility = (
            rate_score * 0.6 +
            phase_score * 0.2 +
            stability_bonus +
            confidence_factor
        )
        
        return max(0.0, min(1.0, compatibility))
    
    def _determine_compatibility_level(self, 
                                     required_rate_change: float,
                                     within_tolerance: bool,
                                     tempo_stability_factor: float) -> TempoCompatibility:
        """Determine qualitative compatibility level"""
        
        rate_change_diff = abs(required_rate_change - 1.0)
        
        if rate_change_diff < 0.01:  # Less than 1% difference
            return TempoCompatibility.PERFECT
        elif within_tolerance and tempo_stability_factor > 0.7:
            return TempoCompatibility.EXCELLENT
        elif within_tolerance:
            return TempoCompatibility.GOOD
        elif rate_change_diff <= self.tempo_tolerance * 1.5:  # 1.5x tolerance
            return TempoCompatibility.MARGINAL
        else:
            return TempoCompatibility.INCOMPATIBLE
    
    def _predict_correction_difficulty(self, 
                                     required_rate_change: float,
                                     bpm_difference: float,
                                     stability_a: float,
                                     stability_b: float,
                                     phase_offset: float) -> float:
        """Predict difficulty of tempo correction"""
        
        features = torch.tensor([
            abs(required_rate_change - 1.0),
            abs(bpm_difference) / 60.0,  # Normalize
            stability_a,
            stability_b,
            phase_offset
        ], dtype=torch.float32)
        
        try:
            with torch.no_grad():
                difficulty = self.difficulty_predictor(features.unsqueeze(0))
                difficulty = float(difficulty.squeeze())
        except:
            # Fallback heuristic
            rate_difficulty = abs(required_rate_change - 1.0) / self.tempo_tolerance
            stability_difficulty = 1.0 - (stability_a + stability_b) / 2.0
            phase_difficulty = phase_offset / 0.5
            
            difficulty = (rate_difficulty * 0.5 + stability_difficulty * 0.3 + phase_difficulty * 0.2)
            difficulty = min(1.0, max(0.0, difficulty))
        
        return difficulty
    
    def _generate_processing_recommendation(self, 
                                          compatibility_level: TempoCompatibility,
                                          required_rate_change: float,
                                          tempo_stability_factor: float) -> str:
        """Generate processing recommendation"""
        
        rate_change_percent = (required_rate_change - 1.0) * 100
        
        if compatibility_level == TempoCompatibility.PERFECT:
            return "no_tempo_correction_needed"
        elif compatibility_level == TempoCompatibility.EXCELLENT:
            return f"light_tempo_correction_{rate_change_percent:+.1f}_percent"
        elif compatibility_level == TempoCompatibility.GOOD:
            return f"moderate_tempo_correction_{rate_change_percent:+.1f}_percent"
        elif compatibility_level == TempoCompatibility.MARGINAL:
            if tempo_stability_factor > 0.6:
                return f"aggressive_tempo_correction_{rate_change_percent:+.1f}_percent"
            else:
                return "hard_cut_recommended_unstable_tempo"
        else:  # INCOMPATIBLE
            return "hard_cut_required_tempo_mismatch"
    
    def is_tempo_correction_recommended(self, rhythmic_match: RhythmicMatch) -> bool:
        """Determine if tempo correction is recommended"""
        
        return not rhythmic_match.processing_recommendation.startswith("hard_cut")
    
    def get_hard_cut_timing_recommendation(self, 
                                         rhythmic_match: RhythmicMatch,
                                         beat_grid_a: BeatGrid) -> dict:
        """Get timing recommendation for hard cut strategy"""
        
        # When tempo correction isn't viable, recommend optimal hard cut timing
        # Wait for A beat to finish, start B where A's next downbeat would be
        
        if len(beat_grid_a.beat_times) == 0:
            return {
                'strategy': 'immediate_cut',
                'timing_offset': 0,
                'confidence': 0.5
            }
        
        # Find next downbeat timing
        if len(beat_grid_a.bar_times) > 0:
            # Use actual bar detection
            next_downbeat_interval = torch.mean(torch.diff(beat_grid_a.bar_times.float()))
            timing_strategy = 'bar_aligned_cut'
            confidence = 0.8
        else:
            # Estimate 4/4 time from beats
            beat_interval = torch.mean(torch.diff(beat_grid_a.beat_times.float()))
            next_downbeat_interval = beat_interval * 4  # Assume 4/4
            timing_strategy = 'estimated_bar_aligned_cut'
            confidence = 0.6
        
        return {
            'strategy': timing_strategy,
            'timing_offset': int(next_downbeat_interval),  # Samples to wait
            'confidence': confidence,
            'beat_interval': int(torch.mean(torch.diff(beat_grid_a.beat_times.float()))) if len(beat_grid_a.beat_times) > 1 else None
        }
    
    def calculate_processing_quality_prediction(self, rhythmic_match: RhythmicMatch) -> dict:
        """Predict quality of tempo processing"""
        
        if not self.is_tempo_correction_recommended(rhythmic_match):
            return {
                'expected_quality': 1.0,  # No processing = no artifacts
                'confidence': 0.9,
                'artifact_risk': 0.0,
                'recommended': False
            }
        
        # Quality decreases with larger rate changes
        rate_change_magnitude = abs(rhythmic_match.required_rate_change - 1.0)
        
        # High-quality time stretching algorithms can handle up to ~10% pretty well
        if rate_change_magnitude <= 0.05:  # ±5%
            expected_quality = 0.95
            artifact_risk = 0.05
        elif rate_change_magnitude <= 0.10:  # ±10%
            expected_quality = 0.85
            artifact_risk = 0.15
        else:  # Beyond ±10%
            expected_quality = max(0.5, 0.85 - (rate_change_magnitude - 0.1) * 2.0)
            artifact_risk = min(0.8, 0.15 + (rate_change_magnitude - 0.1) * 2.0)
        
        # Adjust for tempo stability
        stability_adjustment = rhythmic_match.tempo_stability_factor * 0.1
        expected_quality = min(1.0, expected_quality + stability_adjustment)
        artifact_risk = max(0.0, artifact_risk - stability_adjustment)
        
        return {
            'expected_quality': expected_quality,
            'confidence': rhythmic_match.tempo_stability_factor * 0.7 + 0.3,
            'artifact_risk': artifact_risk,
            'recommended': rhythmic_match.compatibility_level != TempoCompatibility.INCOMPATIBLE
        }


if __name__ == "__main__":
    # Test the module
    from .beat_grid_extractor import BeatGrid
    
    # Create test beat grids
    beat_grid_120 = BeatGrid(
        bpm=120.0,
        beat_times=torch.tensor([0, 22050, 44100, 66150, 88200]),  # 120 BPM at 44.1kHz
        bar_times=torch.tensor([0, 88200]),
        confidence_curve=torch.tensor([0.9, 0.8, 0.9, 0.8, 0.9]),
        tempo_stability=0.9,
        sample_rate=44100
    )
    
    beat_grid_126 = BeatGrid(
        bpm=126.0,
        beat_times=torch.tensor([0, 21000, 42000, 63000, 84000]),  # 126 BPM
        bar_times=torch.tensor([0, 84000]),
        confidence_curve=torch.tensor([0.8, 0.9, 0.8, 0.9, 0.8]),
        tempo_stability=0.8,
        sample_rate=44100
    )
    
    beat_grid_140 = BeatGrid(
        bpm=140.0,
        beat_times=torch.tensor([0, 18900, 37800, 56700, 75600]),  # 140 BPM
        bar_times=torch.tensor([0, 75600]),
        confidence_curve=torch.tensor([0.7, 0.8, 0.7, 0.8, 0.7]),
        tempo_stability=0.7,
        sample_rate=44100
    )
    
    analyzer = RhythmicCompatibilityAnalyzer()
    
    # Test compatible tempos (120 to 126 BPM = 5% difference)
    print("=== 120 BPM to 126 BPM (5% difference) ===")
    match1 = analyzer(beat_grid_120, beat_grid_126)
    print(f"Compatibility: {match1.compatibility_level.value}")
    print(f"Score: {match1.compatibility_score:.3f}")
    print(f"Required rate change: {match1.required_rate_change:.3f}")
    print(f"Rate change %: {(match1.required_rate_change - 1.0) * 100:+.1f}%")
    print(f"Recommendation: {match1.processing_recommendation}")
    
    # Test incompatible tempos (120 to 140 BPM = 16.7% difference)
    print("\n=== 120 BPM to 140 BPM (16.7% difference) ===")
    match2 = analyzer(beat_grid_120, beat_grid_140)
    print(f"Compatibility: {match2.compatibility_level.value}")
    print(f"Score: {match2.compatibility_score:.3f}")
    print(f"Required rate change: {match2.required_rate_change:.3f}")
    print(f"Rate change %: {(match2.required_rate_change - 1.0) * 100:+.1f}%")
    print(f"Recommendation: {match2.processing_recommendation}")
    
    # Test hard cut timing recommendation
    if not analyzer.is_tempo_correction_recommended(match2):
        hard_cut_rec = analyzer.get_hard_cut_timing_recommendation(match2, beat_grid_120)
        print(f"Hard cut strategy: {hard_cut_rec['strategy']}")
        print(f"Timing offset: {hard_cut_rec['timing_offset']} samples")
    
    # Test quality prediction
    quality_pred = analyzer.calculate_processing_quality_prediction(match1)
    print(f"\nQuality prediction for 5% correction: {quality_pred}")