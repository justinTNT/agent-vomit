"""
EnergyCompatibilityAnalyzer - Analyze energy compatibility between tracks

This module evaluates energy level matching, spectral balance compatibility,
and perceptual loudness matching for smooth crossfade transitions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .energy_profile_extractor import EnergyProfile


class EnergyCompatibilityLevel(Enum):
    """Energy compatibility levels"""
    EXCELLENT = "excellent"     # Very close energy levels
    GOOD = "good"              # Compatible energy levels  
    ACCEPTABLE = "acceptable"   # Some adjustment needed
    POOR = "poor"              # Significant mismatch
    INCOMPATIBLE = "incompatible"  # Cannot be matched


@dataclass
class EnergyMatch:
    """Energy compatibility analysis results"""
    compatibility_score: float      # Overall energy compatibility [0,1]
    compatibility_level: EnergyCompatibilityLevel  # Qualitative assessment
    level_difference_db: float      # RMS level difference in dB
    spectral_balance_score: float   # Frequency balance compatibility [0,1]
    loudness_adjustment_db: float   # Recommended loudness adjustment
    dynamic_range_compatibility: float  # Dynamic range match [0,1]
    energy_flow_score: float        # Smoothness of energy transition [0,1]
    processing_recommendation: str   # Recommended action


class EnergyCompatibilityAnalyzer(nn.Module):
    """
    Analyze energy compatibility between Track A exit and Track B entry points.
    
    Evaluates multiple aspects of energy matching for natural transitions:
    - RMS energy level matching
    - Spectral balance (frequency distribution)
    - Perceptual loudness matching
    - Dynamic range compatibility
    """
    
    def __init__(self, 
                 max_level_difference_db: float = 12.0,
                 spectral_tolerance_db: float = 6.0,
                 loudness_tolerance_db: float = 3.0):
        super().__init__()
        
        self.max_level_difference = max_level_difference_db
        self.spectral_tolerance = spectral_tolerance_db
        self.loudness_tolerance = loudness_tolerance_db
        
        # Neural network for perceptual energy matching
        self.perceptual_matcher = nn.Sequential(
            nn.Linear(8, 64),  # Energy features from both tracks
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Dynamic range compatibility predictor
        self.dynamics_assessor = nn.Sequential(
            nn.Linear(6, 32),  # Dynamic range features
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Energy flow quality predictor
        self.flow_analyzer = nn.Sequential(
            nn.Linear(4, 32),  # Energy slope and transition features
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, 
                energy_profile_a: EnergyProfile,
                energy_profile_b: EnergyProfile,
                exit_sample_a: int,
                entry_sample_b: int) -> EnergyMatch:
        """
        Analyze energy compatibility between specific exit and entry points.
        
        Args:
            energy_profile_a: Track A energy analysis
            energy_profile_b: Track B energy analysis
            exit_sample_a: Sample position for A exit
            entry_sample_b: Sample position for B entry
            
        Returns:
            EnergyMatch with compatibility analysis and recommendations
        """
        
        # Get energy information at specific positions
        energy_a = self._get_energy_at_position(energy_profile_a, exit_sample_a)
        energy_b = self._get_energy_at_position(energy_profile_b, entry_sample_b)
        
        # Calculate level difference
        level_difference = energy_b['rms_db'] - energy_a['rms_db']
        
        # Calculate spectral balance compatibility
        spectral_balance_score = self._calculate_spectral_balance(energy_a, energy_b)
        
        # Calculate recommended loudness adjustment
        loudness_adjustment = self._calculate_loudness_adjustment(energy_a, energy_b)
        
        # Assess dynamic range compatibility
        dynamic_range_compatibility = self._assess_dynamic_range_compatibility(
            energy_profile_a, energy_profile_b
        )
        
        # Analyze energy flow quality
        energy_flow_score = self._analyze_energy_flow(
            energy_profile_a, energy_profile_b, exit_sample_a, entry_sample_b
        )
        
        # Calculate overall compatibility score
        compatibility_score = self._calculate_overall_compatibility(
            level_difference, spectral_balance_score, loudness_adjustment,
            dynamic_range_compatibility, energy_flow_score, energy_a, energy_b
        )
        
        # Determine compatibility level
        compatibility_level = self._determine_compatibility_level(
            compatibility_score, abs(level_difference)
        )
        
        # Generate processing recommendation
        processing_recommendation = self._generate_processing_recommendation(
            compatibility_level, level_difference, loudness_adjustment, spectral_balance_score
        )
        
        return EnergyMatch(
            compatibility_score=float(compatibility_score),
            compatibility_level=compatibility_level,
            level_difference_db=float(level_difference),
            spectral_balance_score=float(spectral_balance_score),
            loudness_adjustment_db=float(loudness_adjustment),
            dynamic_range_compatibility=float(dynamic_range_compatibility),
            energy_flow_score=float(energy_flow_score),
            processing_recommendation=processing_recommendation
        )
    
    def _get_energy_at_position(self, energy_profile: EnergyProfile, sample_pos: int) -> dict:
        """Extract energy information at specific sample position"""
        
        # Convert to frame position
        frame_pos = int(sample_pos // (energy_profile.sample_rate / len(energy_profile.rms_curve)))
        frame_pos = max(0, min(frame_pos, len(energy_profile.rms_curve) - 1))
        
        return {
            'rms_db': float(energy_profile.rms_curve[frame_pos]),
            'low_band_db': float(energy_profile.spectral_bands[0, frame_pos]),
            'mid_band_db': float(energy_profile.spectral_bands[1, frame_pos]),
            'high_band_db': float(energy_profile.spectral_bands[2, frame_pos]),
            'loudness': float(energy_profile.loudness_curve[frame_pos]),
            'energy_slope': float(energy_profile.energy_slope[frame_pos])
        }
    
    def _calculate_spectral_balance(self, energy_a: dict, energy_b: dict) -> float:
        """Calculate spectral balance compatibility score"""
        
        # Compare frequency band distributions
        low_diff = abs(energy_a['low_band_db'] - energy_b['low_band_db'])
        mid_diff = abs(energy_a['mid_band_db'] - energy_b['mid_band_db'])
        high_diff = abs(energy_a['high_band_db'] - energy_b['high_band_db'])
        
        # Weight low frequencies more heavily (bass is most noticeable in crossfades)
        weighted_diff = (low_diff * 0.5 + mid_diff * 0.3 + high_diff * 0.2)
        
        # Convert to compatibility score
        spectral_score = max(0.0, 1.0 - weighted_diff / self.spectral_tolerance)
        
        return spectral_score
    
    def _calculate_loudness_adjustment(self, energy_a: dict, energy_b: dict) -> float:
        """Calculate recommended loudness adjustment in dB"""
        
        # Use perceptual loudness rather than raw RMS
        loudness_diff = energy_a['loudness'] - energy_b['loudness']
        
        # Convert loudness difference to approximate dB adjustment
        # This is a simplified mapping - real loudness matching is more complex
        loudness_adjustment = loudness_diff * 6.0  # Rough scaling factor
        
        # Clamp to reasonable range
        loudness_adjustment = max(-self.max_level_difference, 
                                min(self.max_level_difference, loudness_adjustment))
        
        return loudness_adjustment
    
    def _assess_dynamic_range_compatibility(self, 
                                          energy_profile_a: EnergyProfile,
                                          energy_profile_b: EnergyProfile) -> float:
        """Assess compatibility of dynamic range characteristics"""
        
        dynamics_a = energy_profile_a.dynamics
        dynamics_b = energy_profile_b.dynamics
        
        # Compare key dynamic characteristics
        features = torch.tensor([
            dynamics_a['dynamic_range_db'],
            dynamics_b['dynamic_range_db'],
            dynamics_a['peak_avg_ratio'],
            dynamics_b['peak_avg_ratio'],
            dynamics_a['crest_factor'],
            dynamics_b['crest_factor']
        ], dtype=torch.float32)
        
        # Normalize features
        features = torch.clamp(features / 30.0, 0.0, 1.0)  # Rough normalization
        
        try:
            with torch.no_grad():
                compatibility = self.dynamics_assessor(features.unsqueeze(0))
                compatibility = float(compatibility.squeeze())
        except:
            # Fallback heuristic
            range_diff = abs(dynamics_a['dynamic_range_db'] - dynamics_b['dynamic_range_db'])
            peak_diff = abs(dynamics_a['peak_avg_ratio'] - dynamics_b['peak_avg_ratio'])
            crest_diff = abs(dynamics_a['crest_factor'] - dynamics_b['crest_factor'])
            
            # Score based on similarity
            range_score = max(0.0, 1.0 - range_diff / 20.0)  # 20dB max difference
            peak_score = max(0.0, 1.0 - peak_diff / 5.0)     # 5x max ratio difference
            crest_score = max(0.0, 1.0 - crest_diff / 10.0)   # 10x max crest difference
            
            compatibility = (range_score + peak_score + crest_score) / 3.0
        
        return compatibility
    
    def _analyze_energy_flow(self, 
                           energy_profile_a: EnergyProfile,
                           energy_profile_b: EnergyProfile,
                           exit_sample_a: int,
                           entry_sample_b: int) -> float:
        """Analyze smoothness of energy flow through transition"""
        
        # Get energy slopes at transition points
        energy_a = self._get_energy_at_position(energy_profile_a, exit_sample_a)
        energy_b = self._get_energy_at_position(energy_profile_b, entry_sample_b)
        
        # Features for energy flow analysis
        features = torch.tensor([
            energy_a['energy_slope'],
            energy_b['energy_slope'],
            energy_a['rms_db'] - energy_b['rms_db'],  # Level jump
            abs(energy_a['energy_slope'] - energy_b['energy_slope'])  # Slope discontinuity
        ], dtype=torch.float32)
        
        try:
            with torch.no_grad():
                flow_score = self.flow_analyzer(features.unsqueeze(0))
                flow_score = float(flow_score.squeeze())
        except:
            # Fallback heuristic
            # Prefer minimal slope discontinuity and level jumps
            slope_continuity = 1.0 - abs(energy_a['energy_slope'] - energy_b['energy_slope'])
            level_continuity = 1.0 - min(1.0, abs(energy_a['rms_db'] - energy_b['rms_db']) / 12.0)
            
            flow_score = (slope_continuity * 0.4 + level_continuity * 0.6)
        
        return max(0.0, min(1.0, flow_score))
    
    def _calculate_overall_compatibility(self, 
                                       level_difference: float,
                                       spectral_balance_score: float,
                                       loudness_adjustment: float,
                                       dynamic_range_compatibility: float,
                                       energy_flow_score: float,
                                       energy_a: dict,
                                       energy_b: dict) -> float:
        """Calculate overall energy compatibility score"""
        
        # Level difference score (prefer smaller differences)
        level_score = max(0.0, 1.0 - abs(level_difference) / self.max_level_difference)
        
        # Loudness adjustment score (prefer smaller adjustments)
        loudness_score = max(0.0, 1.0 - abs(loudness_adjustment) / self.loudness_tolerance)
        
        # Use neural network for perceptual matching
        features = torch.tensor([
            energy_a['rms_db'] / 60.0,          # Normalize roughly
            energy_b['rms_db'] / 60.0,
            energy_a['loudness'],
            energy_b['loudness'],
            energy_a['low_band_db'] / 60.0,
            energy_b['low_band_db'] / 60.0,
            energy_a['energy_slope'],
            energy_b['energy_slope']
        ], dtype=torch.float32)
        
        try:
            with torch.no_grad():
                perceptual_score = self.perceptual_matcher(features.unsqueeze(0))
                perceptual_score = float(perceptual_score.squeeze())
        except:
            perceptual_score = (level_score + loudness_score) / 2.0
        
        # Weighted combination of all scores
        overall_score = (
            level_score * 0.25 +
            spectral_balance_score * 0.25 +
            loudness_score * 0.15 +
            dynamic_range_compatibility * 0.10 +
            energy_flow_score * 0.15 +
            perceptual_score * 0.10
        )
        
        return max(0.0, min(1.0, overall_score))
    
    def _determine_compatibility_level(self, 
                                     compatibility_score: float,
                                     level_difference_abs: float) -> EnergyCompatibilityLevel:
        """Determine qualitative energy compatibility level"""
        
        if level_difference_abs > self.max_level_difference:
            return EnergyCompatibilityLevel.INCOMPATIBLE
        elif compatibility_score >= 0.8 and level_difference_abs <= 3.0:
            return EnergyCompatibilityLevel.EXCELLENT
        elif compatibility_score >= 0.6:
            return EnergyCompatibilityLevel.GOOD
        elif compatibility_score >= 0.4:
            return EnergyCompatibilityLevel.ACCEPTABLE
        else:
            return EnergyCompatibilityLevel.POOR
    
    def _generate_processing_recommendation(self, 
                                          compatibility_level: EnergyCompatibilityLevel,
                                          level_difference: float,
                                          loudness_adjustment: float,
                                          spectral_balance_score: float) -> str:
        """Generate energy processing recommendation"""
        
        if compatibility_level == EnergyCompatibilityLevel.INCOMPATIBLE:
            return "reject_excessive_energy_mismatch"
        elif compatibility_level == EnergyCompatibilityLevel.EXCELLENT:
            if abs(level_difference) < 1.0:
                return "no_energy_adjustment_needed"
            else:
                return f"minimal_level_adjustment_{level_difference:+.1f}_db"
        elif abs(loudness_adjustment) <= self.loudness_tolerance:
            adjustment_type = "level" if abs(level_difference) > abs(loudness_adjustment) else "loudness"
            adjustment_value = level_difference if adjustment_type == "level" else loudness_adjustment
            return f"moderate_{adjustment_type}_adjustment_{adjustment_value:+.1f}_db"
        elif spectral_balance_score < 0.5:
            return f"spectral_eq_and_level_adjustment_{level_difference:+.1f}_db"
        else:
            return f"aggressive_level_adjustment_{level_difference:+.1f}_db"
    
    def calculate_crossfade_energy_envelope(self, 
                                          energy_match: EnergyMatch,
                                          energy_profile_a: EnergyProfile,
                                          energy_profile_b: EnergyProfile,
                                          exit_sample_a: int,
                                          entry_sample_b: int,
                                          crossfade_duration_samples: int) -> dict:
        """Calculate energy-matched crossfade envelope"""
        
        # Get energy context around transition points
        energy_a = self._get_energy_at_position(energy_profile_a, exit_sample_a)
        energy_b = self._get_energy_at_position(energy_profile_b, entry_sample_b)
        
        # Determine crossfade strategy based on energy compatibility
        if energy_match.compatibility_level == EnergyCompatibilityLevel.EXCELLENT:
            strategy = 'equal_power'
            adjustment_needed = False
        elif energy_match.spectral_balance_score > 0.7:
            strategy = 'equal_power'
            adjustment_needed = True
        else:
            strategy = 'frequency_selective'
            adjustment_needed = True
        
        # Calculate pre-crossfade adjustments
        adjustments = {}
        if adjustment_needed:
            adjustments['track_b_gain_db'] = -energy_match.level_difference_db
            
            if energy_match.spectral_balance_score < 0.6:
                # Calculate simple EQ adjustments
                energy_a_pos = self._get_energy_at_position(energy_profile_a, exit_sample_a)
                energy_b_pos = self._get_energy_at_position(energy_profile_b, entry_sample_b)
                
                adjustments['track_b_low_eq_db'] = energy_a_pos['low_band_db'] - energy_b_pos['low_band_db']
                adjustments['track_b_mid_eq_db'] = energy_a_pos['mid_band_db'] - energy_b_pos['mid_band_db']
                adjustments['track_b_high_eq_db'] = energy_a_pos['high_band_db'] - energy_b_pos['high_band_db']
        
        return {
            'strategy': strategy,
            'adjustments': adjustments,
            'confidence': energy_match.compatibility_score,
            'expected_smoothness': energy_match.energy_flow_score
        }
    
    def get_energy_matching_quality_prediction(self, energy_match: EnergyMatch) -> dict:
        """Predict quality of energy matching"""
        
        return {
            'transition_smoothness': energy_match.energy_flow_score,
            'perceptual_matching': energy_match.compatibility_score,
            'adjustment_artifacts': max(0.0, (abs(energy_match.level_difference_db) - 3.0) / 9.0),
            'recommended': energy_match.compatibility_level != EnergyCompatibilityLevel.INCOMPATIBLE,
            'confidence': min(energy_match.spectral_balance_score, energy_match.dynamic_range_compatibility)
        }


if __name__ == "__main__":
    # Test the module
    from .energy_profile_extractor import EnergyProfileExtractor, create_test_audio_with_energy_pattern
    
    # Create test audio with different energy patterns
    audio_stable = create_test_audio_with_energy_pattern(duration=30.0, pattern='stable')
    audio_buildup = create_test_audio_with_energy_pattern(duration=30.0, pattern='buildup')
    
    # Extract energy profiles
    extractor = EnergyProfileExtractor()
    profile_stable = extractor(audio_stable)
    profile_buildup = extractor(audio_buildup)
    
    # Test energy compatibility
    analyzer = EnergyCompatibilityAnalyzer()
    
    # Test stable to stable (should be compatible)
    print("=== Stable to Stable Energy ===")
    match1 = analyzer(profile_stable, profile_stable, 20*44100, 5*44100)  # 20s to 5s
    print(f"Compatibility: {match1.compatibility_level.value}")
    print(f"Score: {match1.compatibility_score:.3f}")
    print(f"Level difference: {match1.level_difference_db:+.1f} dB")
    print(f"Spectral balance: {match1.spectral_balance_score:.3f}")
    print(f"Recommendation: {match1.processing_recommendation}")
    
    # Test stable to buildup (different energy characteristics)
    print("\n=== Stable to Buildup Energy ===")
    match2 = analyzer(profile_stable, profile_buildup, 20*44100, 5*44100)
    print(f"Compatibility: {match2.compatibility_level.value}")
    print(f"Score: {match2.compatibility_score:.3f}")
    print(f"Level difference: {match2.level_difference_db:+.1f} dB")
    print(f"Energy flow score: {match2.energy_flow_score:.3f}")
    print(f"Recommendation: {match2.processing_recommendation}")
    
    # Test crossfade envelope calculation
    envelope_info = analyzer.calculate_crossfade_energy_envelope(
        match2, profile_stable, profile_buildup, 20*44100, 5*44100, 2*44100
    )
    print(f"\nCrossfade strategy: {envelope_info['strategy']}")
    print(f"Adjustments needed: {len(envelope_info['adjustments']) > 0}")
    
    # Test quality prediction
    quality_pred = analyzer.get_energy_matching_quality_prediction(match2)
    print(f"Quality prediction: {quality_pred}")
