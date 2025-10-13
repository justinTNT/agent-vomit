"""
HarmonicCompatibilityAnalyzer - Analyze key compatibility between tracks

This module determines harmonic compatibility and calculates required
pitch adjustments based on musical key relationships.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .musical_key_extractor import KeyProfile, Mode


class CompatibilityLevel(Enum):
    """Harmonic compatibility levels"""
    EXCELLENT = "excellent"     # Same key, perfect matches
    GOOD = "good"              # Related keys, minimal adjustment
    ACCEPTABLE = "acceptable"   # Slight clash, correctable
    POOR = "poor"              # Strong clash, major adjustment needed
    INCOMPATIBLE = "incompatible"  # Cannot be corrected within limits


@dataclass
class HarmonicMatch:
    """Harmonic compatibility analysis results"""
    compatibility_score: float      # Overall compatibility [0,1]
    compatibility_level: CompatibilityLevel  # Qualitative assessment
    required_pitch_shift: float     # Semitones to shift B (+/- 12)
    dissonance_risk: float         # Risk of artifacts [0,1]
    key_relationship: str          # Description of key relationship
    correction_confidence: float    # Confidence in correction [0,1]
    harmonic_tension: float        # Musical tension level [0,1]
    processing_recommendation: str  # Recommended action


class HarmonicCompatibilityAnalyzer(nn.Module):
    """
    Analyze harmonic compatibility between Track A and Track B keys.
    
    Determines optimal pitch corrections and assesses the musical
    quality of key relationships for crossfade decisions.
    """
    
    def __init__(self, 
                 max_pitch_shift: float = 2.0,
                 clash_threshold: float = 0.3,
                 harmonic_relevance_threshold: float = 0.4):
        super().__init__()
        
        self.max_pitch_shift = max_pitch_shift  # ±2 semitones limit
        self.clash_threshold = clash_threshold  # When to consider correction
        self.harmonic_relevance_threshold = harmonic_relevance_threshold
        
        # Circle of fifths for key relationships
        self.circle_of_fifths = [
            'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F'
        ]
        
        # Key compatibility matrix (based on music theory)
        self.compatibility_matrix = self._build_compatibility_matrix()
        
        # Neural network for dissonance prediction
        self.dissonance_predictor = nn.Sequential(
            nn.Linear(4, 32),  # key1, mode1, key2, mode2 encoded
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Correction confidence estimator
        self.correction_assessor = nn.Sequential(
            nn.Linear(6, 32),  # Multiple factors
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, 
                key_profile_a: KeyProfile,
                key_profile_b: KeyProfile) -> HarmonicMatch:
        """
        Analyze harmonic compatibility between two tracks.
        
        Args:
            key_profile_a: Track A key analysis
            key_profile_b: Track B key analysis
            
        Returns:
            HarmonicMatch with compatibility analysis and recommendations
        """
        
        # Check if harmonic analysis is relevant
        if (key_profile_a.harmonic_relevance < self.harmonic_relevance_threshold and
            key_profile_b.harmonic_relevance < self.harmonic_relevance_threshold):
            # Low harmonic relevance - skip detailed analysis
            return self._create_low_relevance_match(key_profile_a, key_profile_b)
        
        # Calculate key distance and relationship
        key_distance = self._calculate_key_distance(
            key_profile_a.key, key_profile_a.mode,
            key_profile_b.key, key_profile_b.mode
        )
        
        key_relationship = self._describe_key_relationship(
            key_profile_a.key, key_profile_a.mode,
            key_profile_b.key, key_profile_b.mode, key_distance
        )
        
        # Calculate required pitch shift
        required_shift = self._calculate_optimal_pitch_shift(
            key_profile_a, key_profile_b
        )
        
        # Assess compatibility
        compatibility_score = self._calculate_compatibility_score(
            key_distance, key_profile_a, key_profile_b, required_shift
        )
        
        compatibility_level = self._determine_compatibility_level(
            compatibility_score, abs(required_shift)
        )
        
        # Predict dissonance risk
        dissonance_risk = self._predict_dissonance_risk(
            key_profile_a, key_profile_b, required_shift
        )
        
        # Calculate harmonic tension
        harmonic_tension = self._calculate_harmonic_tension(
            key_distance, key_profile_a.mode, key_profile_b.mode
        )
        
        # Assess correction confidence
        correction_confidence = self._assess_correction_confidence(
            key_profile_a, key_profile_b, required_shift, dissonance_risk
        )
        
        # Generate processing recommendation
        processing_recommendation = self._generate_processing_recommendation(
            compatibility_level, required_shift, key_profile_a, key_profile_b
        )
        
        return HarmonicMatch(
            compatibility_score=float(compatibility_score),
            compatibility_level=compatibility_level,
            required_pitch_shift=float(required_shift),
            dissonance_risk=float(dissonance_risk),
            key_relationship=key_relationship,
            correction_confidence=float(correction_confidence),
            harmonic_tension=float(harmonic_tension),
            processing_recommendation=processing_recommendation
        )
    
    def _build_compatibility_matrix(self) -> dict:
        """Build key compatibility matrix based on music theory"""
        
        matrix = {}
        
        for i, key1 in enumerate(self.circle_of_fifths):
            matrix[key1] = {}
            
            for j, key2 in enumerate(self.circle_of_fifths):
                # Distance on circle of fifths
                distance = min(abs(i - j), 12 - abs(i - j))
                
                # Compatibility based on distance
                if distance == 0:
                    compatibility = 1.0  # Same key
                elif distance == 1:
                    compatibility = 0.9  # Adjacent on circle (perfect fifth)
                elif distance == 2:
                    compatibility = 0.7  # Two steps (major second)
                elif distance == 3:
                    compatibility = 0.5  # Three steps 
                elif distance == 4:
                    compatibility = 0.3  # Four steps
                elif distance == 5:
                    compatibility = 0.2  # Five steps
                else:  # distance == 6 (tritone)
                    compatibility = 0.1  # Maximum dissonance
                
                matrix[key1][key2] = compatibility
        
        return matrix
    
    def _calculate_key_distance(self, 
                               key1: str, mode1: Mode,
                               key2: str, mode2: Mode) -> int:
        """Calculate distance between keys on circle of fifths"""
        
        try:
            pos1 = self.circle_of_fifths.index(key1)
            pos2 = self.circle_of_fifths.index(key2)
        except ValueError:
            return 6  # Maximum distance for unknown keys
        
        # Calculate circular distance
        distance = min(abs(pos1 - pos2), 12 - abs(pos1 - pos2))
        
        # Adjust for mode differences
        if mode1 != mode2 and mode1 != Mode.ATONAL and mode2 != Mode.ATONAL:
            distance += 1  # Minor penalty for mode mismatch
            
        return min(6, distance)
    
    def _describe_key_relationship(self, 
                                  key1: str, mode1: Mode,
                                  key2: str, mode2: Mode, 
                                  distance: int) -> str:
        """Describe the musical relationship between keys"""
        
        if key1 == key2 and mode1 == mode2:
            return "identical"
        elif key1 == key2:
            return f"parallel_{mode1.value}_to_{mode2.value}"
        elif distance == 1:
            return "perfect_fifth_related"
        elif distance == 2:
            return "whole_tone_related"
        elif distance == 3:
            return "minor_third_related"
        elif distance == 6:
            return "tritone_related"
        else:
            return f"circle_distance_{distance}"
    
    def _calculate_optimal_pitch_shift(self, 
                                      key_profile_a: KeyProfile,
                                      key_profile_b: KeyProfile) -> float:
        """Calculate optimal pitch shift for Track B"""
        
        # Convert keys to semitone values
        key_semitones = {
            'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
            'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
        }
        
        try:
            a_semitone = key_semitones[key_profile_a.key]
            b_semitone = key_semitones[key_profile_b.key]
        except KeyError:
            return 0.0  # Unknown keys, no shift
        
        # Calculate shift needed to match A
        shift = a_semitone - b_semitone
        
        # Normalize to [-6, 6] range (shortest path)
        if shift > 6:
            shift -= 12
        elif shift < -6:
            shift += 12
        
        # Consider mode relationships
        if (key_profile_a.mode == Mode.MAJOR and key_profile_b.mode == Mode.MINOR):
            # Major to minor: might prefer relative minor (3 semitones up)
            relative_shift = shift + 3
            if abs(relative_shift) < abs(shift) and abs(relative_shift) <= self.max_pitch_shift:
                shift = relative_shift
        elif (key_profile_a.mode == Mode.MINOR and key_profile_b.mode == Mode.MAJOR):
            # Minor to major: might prefer relative major (3 semitones down)  
            relative_shift = shift - 3
            if abs(relative_shift) < abs(shift) and abs(relative_shift) <= self.max_pitch_shift:
                shift = relative_shift
        
        # Clamp to maximum allowed shift
        shift = max(-self.max_pitch_shift, min(self.max_pitch_shift, shift))
        
        return float(shift)
    
    def _calculate_compatibility_score(self, 
                                     key_distance: int,
                                     key_profile_a: KeyProfile,
                                     key_profile_b: KeyProfile,
                                     required_shift: float) -> float:
        """Calculate overall compatibility score"""
        
        # Base compatibility from key distance
        if key_profile_a.key in self.compatibility_matrix and key_profile_b.key in self.compatibility_matrix[key_profile_a.key]:
            base_compatibility = self.compatibility_matrix[key_profile_a.key][key_profile_b.key]
        else:
            base_compatibility = 0.5
        
        # Adjust for mode compatibility
        if key_profile_a.mode == key_profile_b.mode:
            mode_bonus = 0.1
        elif key_profile_a.mode == Mode.ATONAL or key_profile_b.mode == Mode.ATONAL:
            mode_bonus = 0.0  # No bonus/penalty for atonal
        else:
            mode_bonus = -0.1  # Small penalty for mode mismatch
        
        # Adjust for confidence levels
        confidence_factor = (key_profile_a.confidence + key_profile_b.confidence) / 2.0
        
        # Penalty for large required shifts
        shift_penalty = min(0.3, abs(required_shift) / self.max_pitch_shift * 0.3)
        
        # Adjust for harmonic relevance
        relevance_factor = (key_profile_a.harmonic_relevance + key_profile_b.harmonic_relevance) / 2.0
        
        # Combine factors
        compatibility = (
            base_compatibility * 0.5 +
            mode_bonus +
            confidence_factor * 0.2 -
            shift_penalty +
            relevance_factor * 0.2
        )
        
        return max(0.0, min(1.0, compatibility))
    
    def _determine_compatibility_level(self, 
                                     compatibility_score: float,
                                     abs_pitch_shift: float) -> CompatibilityLevel:
        """Determine qualitative compatibility level"""
        
        if abs_pitch_shift > self.max_pitch_shift:
            return CompatibilityLevel.INCOMPATIBLE
        elif compatibility_score >= 0.8:
            return CompatibilityLevel.EXCELLENT
        elif compatibility_score >= 0.6:
            return CompatibilityLevel.GOOD
        elif compatibility_score >= 0.4:
            return CompatibilityLevel.ACCEPTABLE
        else:
            return CompatibilityLevel.POOR
    
    def _predict_dissonance_risk(self, 
                               key_profile_a: KeyProfile,
                               key_profile_b: KeyProfile,
                               required_shift: float) -> float:
        """Predict risk of dissonance after correction"""
        
        # Encode keys and modes for neural network
        key_encoding = {key: i for i, key in enumerate(self.circle_of_fifths)}
        mode_encoding = {Mode.MAJOR: 0, Mode.MINOR: 1, Mode.ATONAL: 2}
        
        try:
            features = torch.tensor([
                key_encoding.get(key_profile_a.key, 0),
                mode_encoding.get(key_profile_a.mode, 2),
                key_encoding.get(key_profile_b.key, 0),
                mode_encoding.get(key_profile_b.mode, 2)
            ], dtype=torch.float32)
            
            # Normalize
            features = features / 12.0
            
            with torch.no_grad():
                risk = self.dissonance_predictor(features.unsqueeze(0))
                risk = float(risk.squeeze())
        except:
            # Fallback to heuristic
            risk = min(1.0, abs(required_shift) / self.max_pitch_shift * 0.5)
        
        # Adjust for processing artifacts risk
        artifact_risk = min(0.5, abs(required_shift) / self.max_pitch_shift * 0.3)
        
        total_risk = min(1.0, risk + artifact_risk)
        
        return total_risk
    
    def _calculate_harmonic_tension(self, 
                                  key_distance: int,
                                  mode_a: Mode, mode_b: Mode) -> float:
        """Calculate harmonic tension level"""
        
        # Base tension from key distance
        distance_tension = key_distance / 6.0  # Normalize to [0,1]
        
        # Mode tension
        if mode_a == Mode.ATONAL or mode_b == Mode.ATONAL:
            mode_tension = 0.5  # Neutral for atonal
        elif mode_a != mode_b:
            mode_tension = 0.3  # Some tension for mode mismatch
        else:
            mode_tension = 0.0  # No tension for same mode
        
        total_tension = min(1.0, distance_tension * 0.7 + mode_tension)
        
        return total_tension
    
    def _assess_correction_confidence(self, 
                                    key_profile_a: KeyProfile,
                                    key_profile_b: KeyProfile,
                                    required_shift: float,
                                    dissonance_risk: float) -> float:
        """Assess confidence in the correction approach"""
        
        features = torch.tensor([
            key_profile_a.confidence,
            key_profile_b.confidence,
            key_profile_a.harmonic_relevance,
            key_profile_b.harmonic_relevance,
            abs(required_shift) / self.max_pitch_shift,
            dissonance_risk
        ], dtype=torch.float32)
        
        try:
            with torch.no_grad():
                confidence = self.correction_assessor(features.unsqueeze(0))
                confidence = float(confidence.squeeze())
        except:
            # Fallback heuristic
            key_conf = (key_profile_a.confidence + key_profile_b.confidence) / 2.0
            shift_conf = 1.0 - abs(required_shift) / self.max_pitch_shift
            risk_conf = 1.0 - dissonance_risk
            confidence = (key_conf + shift_conf + risk_conf) / 3.0
        
        return confidence
    
    def _generate_processing_recommendation(self, 
                                          compatibility_level: CompatibilityLevel,
                                          required_shift: float,
                                          key_profile_a: KeyProfile,
                                          key_profile_b: KeyProfile) -> str:
        """Generate processing recommendation"""
        
        low_relevance = (key_profile_a.harmonic_relevance < self.harmonic_relevance_threshold or
                        key_profile_b.harmonic_relevance < self.harmonic_relevance_threshold)
        
        if low_relevance:
            return "skip_pitch_correction_low_relevance"
        elif compatibility_level == CompatibilityLevel.INCOMPATIBLE:
            return "reject_incompatible_keys"
        elif compatibility_level == CompatibilityLevel.EXCELLENT:
            if abs(required_shift) < 0.5:
                return "no_correction_needed"
            else:
                return f"minimal_correction_{required_shift:+.1f}_semitones"
        elif abs(required_shift) <= 1.0:
            return f"light_correction_{required_shift:+.1f}_semitones"
        elif abs(required_shift) <= self.max_pitch_shift:
            return f"moderate_correction_{required_shift:+.1f}_semitones"
        else:
            return "correction_exceeds_limits"
    
    def _create_low_relevance_match(self, 
                                   key_profile_a: KeyProfile,
                                   key_profile_b: KeyProfile) -> HarmonicMatch:
        """Create match result for low harmonic relevance content"""
        
        return HarmonicMatch(
            compatibility_score=0.8,  # High score since harmony doesn't matter
            compatibility_level=CompatibilityLevel.GOOD,
            required_pitch_shift=0.0,
            dissonance_risk=0.1,
            key_relationship="harmonic_irrelevant",
            correction_confidence=0.9,
            harmonic_tension=0.0,
            processing_recommendation="skip_pitch_correction_low_relevance"
        )
    
    def is_correction_worthwhile(self, harmonic_match: HarmonicMatch) -> bool:
        """Determine if pitch correction is worthwhile"""
        
        if harmonic_match.processing_recommendation.startswith("skip"):
            return False
        elif harmonic_match.processing_recommendation.startswith("reject"):
            return False
        elif harmonic_match.processing_recommendation.startswith("no_correction"):
            return False
        else:
            return True
    
    def get_correction_quality_prediction(self, harmonic_match: HarmonicMatch) -> dict:
        """Predict the quality of pitch correction"""
        
        return {
            'expected_quality': 1.0 - harmonic_match.dissonance_risk,
            'confidence': harmonic_match.correction_confidence,
            'recommended': self.is_correction_worthwhile(harmonic_match),
            'risk_level': 'low' if harmonic_match.dissonance_risk < 0.3 else
                         'medium' if harmonic_match.dissonance_risk < 0.7 else 'high'
        }


if __name__ == "__main__":
    # Test the module
    from .musical_key_extractor import KeyProfile, Mode
    
    # Create test key profiles
    key_profile_c_major = KeyProfile(
        key='C',
        mode=Mode.MAJOR,
        confidence=0.9,
        stability_zones=[(0, 100000)],
        harmonic_relevance=0.8,
        chroma_vector=torch.zeros(12),
        key_strength=torch.zeros(24)
    )
    
    key_profile_g_major = KeyProfile(
        key='G',
        mode=Mode.MAJOR,
        confidence=0.8,
        stability_zones=[(0, 100000)],
        harmonic_relevance=0.7,
        chroma_vector=torch.zeros(12),
        key_strength=torch.zeros(24)
    )
    
    key_profile_c_sharp_minor = KeyProfile(
        key='C#',
        mode=Mode.MINOR,
        confidence=0.7,
        stability_zones=[(0, 100000)],
        harmonic_relevance=0.6,
        chroma_vector=torch.zeros(12),
        key_strength=torch.zeros(24)
    )
    
    analyzer = HarmonicCompatibilityAnalyzer()
    
    # Test compatible keys (C major to G major - perfect fifth)
    print("=== C Major to G Major ===")
    match1 = analyzer(key_profile_c_major, key_profile_g_major)
    print(f"Compatibility: {match1.compatibility_level.value}")
    print(f"Score: {match1.compatibility_score:.3f}")
    print(f"Required shift: {match1.required_pitch_shift:+.1f} semitones")
    print(f"Recommendation: {match1.processing_recommendation}")
    
    # Test more distant keys (C major to C# minor)
    print("\n=== C Major to C# Minor ===")
    match2 = analyzer(key_profile_c_major, key_profile_c_sharp_minor)
    print(f"Compatibility: {match2.compatibility_level.value}")
    print(f"Score: {match2.compatibility_score:.3f}")
    print(f"Required shift: {match2.required_pitch_shift:+.1f} semitones")
    print(f"Dissonance risk: {match2.dissonance_risk:.3f}")
    print(f"Recommendation: {match2.processing_recommendation}")
    
    # Test correction quality prediction
    quality_pred = analyzer.get_correction_quality_prediction(match2)
    print(f"Quality prediction: {quality_pred}")