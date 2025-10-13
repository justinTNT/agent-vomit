"""
TransitionSmoothnessPredictions - Predict crossfade quality using audio analysis

This module uses established audio analysis techniques to predict how smooth
a crossfade transition will sound, helping optimize parameters before processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .energy_profile_extractor import EnergyProfile
from .beat_grid_extractor import BeatGrid
from .musical_key_extractor import KeyProfile
from .processing_parameter_calculator import ProcessingParams
from .crossfade_envelope_designer import CrossfadeEnvelope


class SmoothnessFactor(Enum):
    """Factors affecting transition smoothness"""
    TEMPO_CONTINUITY = "tempo_continuity"
    ENERGY_FLOW = "energy_flow"
    HARMONIC_COMPATIBILITY = "harmonic_compatibility"
    RHYTHMIC_ALIGNMENT = "rhythmic_alignment"
    SPECTRAL_MATCHING = "spectral_matching"
    PROCESSING_ARTIFACTS = "processing_artifacts"


class TransitionQuality(Enum):
    """Overall transition quality levels"""
    SEAMLESS = "seamless"        # Perfect transition, undetectable
    SMOOTH = "smooth"            # Good transition, minimal artifacts
    ACCEPTABLE = "acceptable"    # Decent transition, some artifacts
    ROUGH = "rough"              # Noticeable transition, artifacts present
    JARRING = "jarring"          # Poor transition, obvious artifacts


@dataclass
class SmoothnessPrediction:
    """Transition smoothness prediction results"""
    overall_quality: TransitionQuality     # Overall predicted quality
    smoothness_score: float                # Overall smoothness [0,1]
    factor_scores: Dict[SmoothnessFactor, float]  # Individual factor scores
    predicted_artifacts: List[str]         # Expected artifacts/issues
    confidence: float                      # Prediction confidence [0,1]
    improvement_suggestions: List[str]     # How to improve the transition
    quality_breakdown: Dict[str, float]    # Detailed quality metrics


class TransitionSmoothnessPredictions(nn.Module):
    """
    Predict crossfade transition smoothness using audio analysis.
    
    Uses established techniques from audio engineering and perceptual
    audio quality assessment to predict transition quality.
    """
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 smoothness_threshold: float = 0.7,
                 artifact_detection_sensitivity: float = 0.8):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.smoothness_threshold = smoothness_threshold
        self.artifact_sensitivity = artifact_detection_sensitivity
        
        # Neural network for overall smoothness prediction
        self.smoothness_predictor = nn.Sequential(
            nn.Linear(15, 128),  # Multi-factor input
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Factor-specific analysis networks
        self.tempo_analyzer = nn.Sequential(
            nn.Linear(4, 32),   # Tempo-related features
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        self.energy_analyzer = nn.Sequential(
            nn.Linear(6, 32),   # Energy flow features
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Artifact detection network
        self.artifact_detector = nn.Sequential(
            nn.Linear(8, 64),   # Processing-related features
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6)    # 6 types of artifacts
        )

    def forward(self, 
                beat_grid_a: BeatGrid,
                beat_grid_b: BeatGrid,
                energy_profile_a: EnergyProfile,
                energy_profile_b: EnergyProfile,
                key_profile_a: Optional[KeyProfile],
                key_profile_b: Optional[KeyProfile],
                processing_params: ProcessingParams,
                crossfade_envelope: CrossfadeEnvelope,
                crossfade_start_a: int,
                crossfade_start_b: int) -> SmoothnessPrediction:
        """
        Predict transition smoothness based on all available analysis.
        
        Args:
            beat_grid_a: Track A beat analysis
            beat_grid_b: Track B beat analysis
            energy_profile_a: Track A energy analysis
            energy_profile_b: Track B energy analysis
            key_profile_a: Track A key analysis (optional)
            key_profile_b: Track B key analysis (optional)
            processing_params: Processing parameters to be applied
            crossfade_envelope: Crossfade envelope design
            crossfade_start_a: Crossfade start in A (samples)
            crossfade_start_b: Crossfade start in B (samples)
            
        Returns:
            SmoothnessPrediction with quality assessment
        """
        
        # Analyze individual smoothness factors
        factor_scores = {}
        
        # Tempo continuity analysis
        factor_scores[SmoothnessFactor.TEMPO_CONTINUITY] = self._analyze_tempo_continuity(
            beat_grid_a, beat_grid_b, processing_params
        )
        
        # Energy flow analysis
        factor_scores[SmoothnessFactor.ENERGY_FLOW] = self._analyze_energy_flow(
            energy_profile_a, energy_profile_b, crossfade_start_a, crossfade_start_b, crossfade_envelope
        )
        
        # Harmonic compatibility (if key data available)
        if key_profile_a and key_profile_b:
            factor_scores[SmoothnessFactor.HARMONIC_COMPATIBILITY] = self._analyze_harmonic_compatibility(
                key_profile_a, key_profile_b, processing_params
            )
        else:
            factor_scores[SmoothnessFactor.HARMONIC_COMPATIBILITY] = 0.5  # Neutral if no key data
        
        # Rhythmic alignment analysis
        factor_scores[SmoothnessFactor.RHYTHMIC_ALIGNMENT] = self._analyze_rhythmic_alignment(
            beat_grid_a, beat_grid_b, crossfade_start_a, crossfade_start_b
        )
        
        # Spectral matching analysis (using energy profiles)
        factor_scores[SmoothnessFactor.SPECTRAL_MATCHING] = self._analyze_spectral_matching(
            energy_profile_a, energy_profile_b, crossfade_start_a, crossfade_start_b
        )
        
        # Processing artifacts prediction
        factor_scores[SmoothnessFactor.PROCESSING_ARTIFACTS] = self._predict_processing_artifacts(
            processing_params, beat_grid_a, beat_grid_b
        )
        
        # Overall smoothness prediction using neural network
        overall_smoothness = self._predict_overall_smoothness(factor_scores, processing_params)
        
        # Quality classification
        overall_quality = self._classify_quality(overall_smoothness)
        
        # Predict specific artifacts
        predicted_artifacts = self._predict_specific_artifacts(factor_scores, processing_params)
        
        # Calculate prediction confidence
        confidence = self._calculate_prediction_confidence(factor_scores, overall_smoothness)
        
        # Generate improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(factor_scores, processing_params)
        
        # Create quality breakdown
        quality_breakdown = self._create_quality_breakdown(factor_scores, overall_smoothness)
        
        return SmoothnessPrediction(
            overall_quality=overall_quality,
            smoothness_score=float(overall_smoothness),
            factor_scores=factor_scores,
            predicted_artifacts=predicted_artifacts,
            confidence=float(confidence),
            improvement_suggestions=improvement_suggestions,
            quality_breakdown=quality_breakdown
        )
    
    def _analyze_tempo_continuity(self, 
                                 beat_grid_a: BeatGrid,
                                 beat_grid_b: BeatGrid,
                                 processing_params: ProcessingParams) -> float:
        """Analyze tempo continuity smoothness"""
        
        # Original tempo difference
        tempo_ratio = beat_grid_b.bpm / (beat_grid_a.bpm + 1e-8)
        original_tempo_difference = abs(1.0 - tempo_ratio)
        
        # After processing tempo difference
        if processing_params.strategy.value in ['tempo_only', 'pitch_and_tempo']:
            processed_tempo_difference = abs(processing_params.tempo_adjustment - 1.0)
        else:
            processed_tempo_difference = original_tempo_difference
        
        # Tempo stability factors
        avg_stability = (beat_grid_a.tempo_stability + beat_grid_b.tempo_stability) / 2
        
        # Create feature vector
        features = torch.tensor([
            processed_tempo_difference / 0.1,  # Normalize by 10% difference
            original_tempo_difference / 0.1,
            avg_stability,
            float(processing_params.strategy.value == 'tempo_only')
        ], dtype=torch.float32)
        
        try:
            with torch.no_grad():
                tempo_score = self.tempo_analyzer(features.unsqueeze(0)).item()
        except:
            # Fallback calculation
            tempo_score = max(0.0, 1.0 - processed_tempo_difference * 5.0) * avg_stability
            tempo_score = max(0.0, min(1.0, tempo_score))
        
        return tempo_score
    
    def _analyze_energy_flow(self, 
                           energy_profile_a: EnergyProfile,
                           energy_profile_b: EnergyProfile,
                           crossfade_start_a: int,
                           crossfade_start_b: int,
                           crossfade_envelope: CrossfadeEnvelope) -> float:
        """Analyze energy flow continuity"""
        
        # Extract energy at crossfade points
        hop_length = 512  # Assume standard hop length
        frame_start_a = crossfade_start_a // hop_length
        frame_start_b = crossfade_start_b // hop_length
        
        # Clamp to available frames
        frame_start_a = max(0, min(frame_start_a, len(energy_profile_a.rms_curve) - 1))
        frame_start_b = max(0, min(frame_start_b, len(energy_profile_b.rms_curve) - 1))
        
        # Energy levels at crossfade points
        energy_a_start = energy_profile_a.rms_curve[frame_start_a]
        energy_b_start = energy_profile_b.rms_curve[frame_start_b]
        
        # Energy difference
        energy_difference = abs(energy_a_start - energy_b_start)
        
        # Energy slope analysis
        slope_a = energy_profile_a.energy_slope[min(frame_start_a, len(energy_profile_a.energy_slope) - 1)]
        slope_b = energy_profile_b.energy_slope[min(frame_start_b, len(energy_profile_b.energy_slope) - 1)]
        slope_compatibility = 1.0 - abs(slope_a - slope_b) / 2.0
        
        # Dynamic range compatibility
        range_a = energy_profile_a.dynamics['dynamic_range_db']
        range_b = energy_profile_b.dynamics['dynamic_range_db']
        range_compatibility = 1.0 - abs(range_a - range_b) / 40.0  # 40dB max difference
        
        # Create feature vector
        features = torch.tensor([
            energy_difference / 20.0,  # Normalize by 20dB
            slope_compatibility,
            range_compatibility,
            float(energy_a_start) / 60.0,  # Normalize
            float(energy_b_start) / 60.0,
            float(crossfade_envelope.duration) / self.sample_rate  # Duration in seconds
        ], dtype=torch.float32)
        
        try:
            with torch.no_grad():
                energy_score = self.energy_analyzer(features.unsqueeze(0)).item()
        except:
            # Fallback calculation
            energy_score = (slope_compatibility + range_compatibility) / 2
            energy_score *= max(0.0, 1.0 - energy_difference / 15.0)  # Penalize large energy differences
            energy_score = max(0.0, min(1.0, energy_score))
        
        return energy_score
    
    def _analyze_harmonic_compatibility(self, 
                                      key_profile_a: KeyProfile,
                                      key_profile_b: KeyProfile,
                                      processing_params: ProcessingParams) -> float:
        """Analyze harmonic compatibility"""
        
        # Key compatibility (circle of fifths distance)
        key_distance = self._calculate_key_distance(
            key_profile_a.key, key_profile_b.key
        )
        
        # Mode compatibility
        mode_compatibility = 1.0 if key_profile_a.mode == key_profile_b.mode else 0.7
        
        # Pitch processing effects
        pitch_shift_penalty = 0.0
        if processing_params.pitch_adjustment != 0:
            # Penalty increases with pitch shift amount
            pitch_shift_penalty = abs(processing_params.pitch_adjustment) / 2.0  # 2 semitones = 100% penalty
        
        # Harmonic relevance (how much does key matter)
        relevance_factor = (key_profile_a.harmonic_relevance + key_profile_b.harmonic_relevance) / 2
        
        # Base harmonic compatibility
        base_compatibility = max(0.0, 1.0 - key_distance / 6.0)  # 6 is max distance
        base_compatibility *= mode_compatibility
        
        # Apply pitch processing penalty
        processed_compatibility = base_compatibility * (1.0 - pitch_shift_penalty)
        
        # Weight by harmonic relevance
        final_compatibility = (
            processed_compatibility * relevance_factor +
            0.8 * (1.0 - relevance_factor)  # If harmony doesn't matter much, assume good compatibility
        )
        
        return max(0.0, min(1.0, final_compatibility))
    
    def _analyze_rhythmic_alignment(self, 
                                  beat_grid_a: BeatGrid,
                                  beat_grid_b: BeatGrid,
                                  crossfade_start_a: int,
                                  crossfade_start_b: int) -> float:
        """Analyze rhythmic alignment quality"""
        
        # Find nearest beats to crossfade points
        if len(beat_grid_a.beat_times) == 0 or len(beat_grid_b.beat_times) == 0:
            return 0.5  # Neutral if no beat data
        
        # Distance to nearest beats
        distances_a = torch.abs(beat_grid_a.beat_times - crossfade_start_a)
        distances_b = torch.abs(beat_grid_b.beat_times - crossfade_start_b)
        
        nearest_beat_dist_a = torch.min(distances_a).item()
        nearest_beat_dist_b = torch.min(distances_b).item()
        
        # Beat interval calculations
        beat_interval_a = 60.0 * self.sample_rate / beat_grid_a.bpm
        beat_interval_b = 60.0 * self.sample_rate / beat_grid_b.bpm
        
        # Alignment scores (closer to beat = better)
        alignment_a = 1.0 - min(1.0, nearest_beat_dist_a / (beat_interval_a * 0.25))  # Within quarter beat
        alignment_b = 1.0 - min(1.0, nearest_beat_dist_b / (beat_interval_b * 0.25))
        
        # Beat confidence factor
        if len(beat_grid_a.confidence_curve) > 0 and len(beat_grid_b.confidence_curve) > 0:
            avg_confidence = (torch.mean(beat_grid_a.confidence_curve) + 
                            torch.mean(beat_grid_b.confidence_curve)) / 2
        else:
            avg_confidence = 0.5
        
        # Tempo stability factor
        tempo_stability = (beat_grid_a.tempo_stability + beat_grid_b.tempo_stability) / 2
        
        # Combined rhythmic alignment score
        alignment_score = (alignment_a + alignment_b) / 2
        alignment_score *= float(avg_confidence)
        alignment_score *= tempo_stability
        
        return max(0.0, min(1.0, alignment_score))
    
    def _analyze_spectral_matching(self, 
                                 energy_profile_a: EnergyProfile,
                                 energy_profile_b: EnergyProfile,
                                 crossfade_start_a: int,
                                 crossfade_start_b: int) -> float:
        """Analyze spectral content matching"""
        
        # Extract spectral content at crossfade points
        hop_length = 512
        frame_start_a = max(0, min(crossfade_start_a // hop_length, energy_profile_a.spectral_bands.shape[1] - 1))
        frame_start_b = max(0, min(crossfade_start_b // hop_length, energy_profile_b.spectral_bands.shape[1] - 1))
        
        # Get spectral bands at crossfade points
        spectral_a = energy_profile_a.spectral_bands[:, frame_start_a]
        spectral_b = energy_profile_b.spectral_bands[:, frame_start_b]
        
        # Calculate spectral difference for each band
        spectral_differences = torch.abs(spectral_a - spectral_b)
        
        # Weight bands by importance (bass more important for crossfades)
        band_weights = torch.tensor([0.4, 0.35, 0.25])  # Low, Mid, High
        weighted_difference = torch.sum(spectral_differences * band_weights)
        
        # Convert to compatibility score
        spectral_compatibility = max(0.0, 1.0 - weighted_difference / 20.0)  # 20dB max difference
        
        return spectral_compatibility
    
    def _predict_processing_artifacts(self, 
                                    processing_params: ProcessingParams,
                                    beat_grid_a: BeatGrid,
                                    beat_grid_b: BeatGrid) -> float:
        """Predict processing-related artifacts"""
        
        artifact_score = 1.0  # Start with no artifacts
        
        # Pitch shift artifacts
        if processing_params.pitch_adjustment != 0:
            pitch_penalty = abs(processing_params.pitch_adjustment) / 2.0  # 2 semitones = 100% penalty
            artifact_score *= (1.0 - pitch_penalty * 0.3)  # 30% penalty for pitch shifts
        
        # Tempo change artifacts
        if processing_params.tempo_adjustment != 1.0:
            tempo_penalty = abs(processing_params.tempo_adjustment - 1.0) / 0.05  # 5% = 100% penalty
            artifact_score *= (1.0 - tempo_penalty * 0.2)  # 20% penalty for tempo changes
        
        # Gain adjustment artifacts (usually minimal)
        if abs(processing_params.gain_adjustment) > 6.0:  # Large gain changes
            gain_penalty = (abs(processing_params.gain_adjustment) - 6.0) / 12.0  # 12dB = 100% penalty
            artifact_score *= (1.0 - gain_penalty * 0.1)  # 10% penalty for large gain changes
        
        # Processing quality factor
        if hasattr(processing_params, 'expected_quality'):
            quality_map = {
                'excellent': 1.0,
                'good': 0.9,
                'acceptable': 0.7,
                'poor': 0.4
            }
            quality_factor = quality_map.get(processing_params.expected_quality.value, 0.5)
            artifact_score *= quality_factor
        
        return max(0.0, min(1.0, artifact_score))
    
    def _predict_overall_smoothness(self, 
                                  factor_scores: Dict[SmoothnessFactor, float],
                                  processing_params: ProcessingParams) -> float:
        """Predict overall transition smoothness"""
        
        # Create feature vector for neural network
        features = []
        for factor in SmoothnessFactor:
            features.append(factor_scores[factor])
        
        # Add processing-specific features
        features.extend([
            abs(processing_params.pitch_adjustment) / 2.0,  # Normalize
            abs(processing_params.tempo_adjustment - 1.0) / 0.1,
            processing_params.gain_adjustment / 12.0,
            float(processing_params.strategy.value == 'none'),
            float(processing_params.strategy.value == 'pitch_only'),
            float(processing_params.strategy.value == 'tempo_only'),
            float(processing_params.strategy.value == 'pitch_and_tempo'),
            processing_params.confidence,
            processing_params.artifact_likelihood
        ])
        
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        try:
            with torch.no_grad():
                overall_smoothness = self.smoothness_predictor(features_tensor.unsqueeze(0)).item()
        except:
            # Fallback: weighted average of factor scores
            weights = {
                SmoothnessFactor.TEMPO_CONTINUITY: 0.25,
                SmoothnessFactor.ENERGY_FLOW: 0.25,
                SmoothnessFactor.HARMONIC_COMPATIBILITY: 0.15,
                SmoothnessFactor.RHYTHMIC_ALIGNMENT: 0.15,
                SmoothnessFactor.SPECTRAL_MATCHING: 0.1,
                SmoothnessFactor.PROCESSING_ARTIFACTS: 0.1
            }
            
            overall_smoothness = sum(factor_scores[factor] * weight 
                                   for factor, weight in weights.items())
        
        return max(0.0, min(1.0, overall_smoothness))
    
    def _classify_quality(self, smoothness_score: float) -> TransitionQuality:
        """Classify overall transition quality"""
        
        if smoothness_score >= 0.9:
            return TransitionQuality.SEAMLESS
        elif smoothness_score >= 0.7:
            return TransitionQuality.SMOOTH
        elif smoothness_score >= 0.5:
            return TransitionQuality.ACCEPTABLE
        elif smoothness_score >= 0.3:
            return TransitionQuality.ROUGH
        else:
            return TransitionQuality.JARRING
    
    def _predict_specific_artifacts(self, 
                                  factor_scores: Dict[SmoothnessFactor, float],
                                  processing_params: ProcessingParams) -> List[str]:
        """Predict specific artifacts that may occur"""
        
        artifacts = []
        
        # Tempo-related artifacts
        if factor_scores[SmoothnessFactor.TEMPO_CONTINUITY] < 0.6:
            if abs(processing_params.tempo_adjustment - 1.0) > 0.03:
                artifacts.append("Time-stretching artifacts (formant shifts)")
            artifacts.append("Tempo discontinuity audible")
        
        # Energy-related artifacts
        if factor_scores[SmoothnessFactor.ENERGY_FLOW] < 0.6:
            artifacts.append("Energy level jump")
            artifacts.append("Dynamic range mismatch")
        
        # Harmonic artifacts
        if factor_scores[SmoothnessFactor.HARMONIC_COMPATIBILITY] < 0.5:
            if processing_params.pitch_adjustment != 0:
                artifacts.append("Pitch-shifting artifacts")
            artifacts.append("Harmonic clashing")
        
        # Rhythmic artifacts
        if factor_scores[SmoothnessFactor.RHYTHMIC_ALIGNMENT] < 0.6:
            artifacts.append("Beat misalignment")
            artifacts.append("Rhythmic discontinuity")
        
        # Spectral artifacts
        if factor_scores[SmoothnessFactor.SPECTRAL_MATCHING] < 0.6:
            artifacts.append("Frequency content mismatch")
            artifacts.append("Timbral discontinuity")
        
        # Processing artifacts
        if factor_scores[SmoothnessFactor.PROCESSING_ARTIFACTS] < 0.7:
            artifacts.append("Digital processing artifacts")
            if abs(processing_params.pitch_adjustment) > 1:
                artifacts.append("Pitch-shift quality degradation")
        
        return artifacts
    
    def _calculate_prediction_confidence(self, 
                                       factor_scores: Dict[SmoothnessFactor, float],
                                       overall_smoothness: float) -> float:
        """Calculate confidence in the prediction"""
        
        # High confidence when all factors are either very high or very low (clear cases)
        # Low confidence when factors are mixed (uncertain cases)
        
        scores = list(factor_scores.values())
        score_variance = np.var(scores)
        
        # Low variance = consistent scores = high confidence
        consistency_confidence = max(0.0, 1.0 - score_variance * 2.0)
        
        # Extreme scores (very good or very bad) = high confidence
        extreme_confidence = max(0.0, abs(overall_smoothness - 0.5) * 2.0)
        
        # Combined confidence
        confidence = (consistency_confidence + extreme_confidence) / 2
        
        # Minimum confidence based on data availability
        base_confidence = 0.6  # Reasonable baseline
        
        final_confidence = base_confidence + (1.0 - base_confidence) * confidence
        
        return max(0.0, min(1.0, final_confidence))
    
    def _generate_improvement_suggestions(self, 
                                        factor_scores: Dict[SmoothnessFactor, float],
                                        processing_params: ProcessingParams) -> List[str]:
        """Generate suggestions to improve transition quality"""
        
        suggestions = []
        
        # Tempo continuity improvements
        if factor_scores[SmoothnessFactor.TEMPO_CONTINUITY] < 0.6:
            if abs(processing_params.tempo_adjustment - 1.0) > 0.05:
                suggestions.append("Consider hard cut instead of tempo correction")
            suggestions.append("Find better matching tempo sections")
        
        # Energy flow improvements
        if factor_scores[SmoothnessFactor.ENERGY_FLOW] < 0.6:
            suggestions.append("Adjust crossfade curve to match energy slopes")
            suggestions.append("Apply pre-gain to match energy levels")
            suggestions.append("Use longer crossfade duration")
        
        # Harmonic compatibility improvements
        if factor_scores[SmoothnessFactor.HARMONIC_COMPATIBILITY] < 0.5:
            if processing_params.pitch_adjustment == 0:
                suggestions.append("Apply pitch correction for better key matching")
            else:
                suggestions.append("Reduce pitch shift amount if possible")
            suggestions.append("Find sections in more compatible keys")
        
        # Rhythmic alignment improvements
        if factor_scores[SmoothnessFactor.RHYTHMIC_ALIGNMENT] < 0.6:
            suggestions.append("Align crossfade to beat boundaries")
            suggestions.append("Adjust crossfade start positions")
            suggestions.append("Use beat-synchronized crossfade duration")
        
        # Spectral matching improvements
        if factor_scores[SmoothnessFactor.SPECTRAL_MATCHING] < 0.6:
            suggestions.append("Apply EQ to match spectral content")
            suggestions.append("Use frequency-selective crossfade")
            suggestions.append("Find sections with similar timbre")
        
        # Processing artifact improvements
        if factor_scores[SmoothnessFactor.PROCESSING_ARTIFACTS] < 0.7:
            suggestions.append("Reduce processing intensity")
            suggestions.append("Use higher quality processing algorithms")
            if abs(processing_params.pitch_adjustment) > 1.5:
                suggestions.append("Avoid large pitch shifts")
        
        return suggestions
    
    def _create_quality_breakdown(self, 
                                factor_scores: Dict[SmoothnessFactor, float],
                                overall_smoothness: float) -> Dict[str, float]:
        """Create detailed quality breakdown"""
        
        breakdown = {
            'overall_smoothness': overall_smoothness,
            'tempo_continuity': factor_scores[SmoothnessFactor.TEMPO_CONTINUITY],
            'energy_flow': factor_scores[SmoothnessFactor.ENERGY_FLOW],
            'harmonic_compatibility': factor_scores[SmoothnessFactor.HARMONIC_COMPATIBILITY],
            'rhythmic_alignment': factor_scores[SmoothnessFactor.RHYTHMIC_ALIGNMENT],
            'spectral_matching': factor_scores[SmoothnessFactor.SPECTRAL_MATCHING],
            'processing_artifacts': factor_scores[SmoothnessFactor.PROCESSING_ARTIFACTS]
        }
        
        # Add composite scores
        musical_compatibility = (
            factor_scores[SmoothnessFactor.HARMONIC_COMPATIBILITY] * 0.4 +
            factor_scores[SmoothnessFactor.RHYTHMIC_ALIGNMENT] * 0.6
        )
        
        technical_quality = (
            factor_scores[SmoothnessFactor.SPECTRAL_MATCHING] * 0.4 +
            factor_scores[SmoothnessFactor.PROCESSING_ARTIFACTS] * 0.6
        )
        
        flow_quality = (
            factor_scores[SmoothnessFactor.TEMPO_CONTINUITY] * 0.5 +
            factor_scores[SmoothnessFactor.ENERGY_FLOW] * 0.5
        )
        
        breakdown.update({
            'musical_compatibility': musical_compatibility,
            'technical_quality': technical_quality,
            'flow_quality': flow_quality
        })
        
        return breakdown
    
    def _calculate_key_distance(self, key1: str, key2: str) -> int:
        """Calculate distance between two keys on circle of fifths"""
        
        circle_of_fifths = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F']
        
        try:
            pos1 = circle_of_fifths.index(key1)
            pos2 = circle_of_fifths.index(key2)
        except ValueError:
            return 3  # Default moderate distance for unknown keys
        
        # Calculate circular distance
        distance = min(abs(pos1 - pos2), 12 - abs(pos1 - pos2))
        return distance
    
    def get_quality_summary(self, prediction: SmoothnessPrediction) -> str:
        """Generate human-readable quality summary"""
        
        summary = f"Transition Quality: {prediction.overall_quality.value.title()}\n"
        summary += f"Smoothness Score: {prediction.smoothness_score:.3f}\n"
        summary += f"Prediction Confidence: {prediction.confidence:.3f}\n\n"
        
        summary += "Factor Analysis:\n"
        for factor, score in prediction.factor_scores.items():
            factor_name = factor.value.replace('_', ' ').title()
            summary += f"  {factor_name}: {score:.3f}\n"
        
        if prediction.predicted_artifacts:
            summary += f"\nExpected Artifacts:\n"
            for artifact in prediction.predicted_artifacts:
                summary += f"  - {artifact}\n"
        
        if prediction.improvement_suggestions:
            summary += f"\nImprovement Suggestions:\n"
            for suggestion in prediction.improvement_suggestions:
                summary += f"  - {suggestion}\n"
        
        return summary


if __name__ == "__main__":
    # Test the module
    from .beat_grid_extractor import BeatGridExtractor, create_test_audio
    from .energy_profile_extractor import EnergyProfileExtractor
    from .musical_key_extractor import MusicalKeyExtractor, create_test_audio_with_key
    from .processing_parameter_calculator import ProcessingParameterCalculator
    from .crossfade_envelope_designer import CrossfadeEnvelopeDesigner
    
    # Create test audio
    audio_a = create_test_audio(10.0, 120.0)
    audio_b_compatible = create_test_audio(10.0, 123.0)  # Compatible tempo
    audio_b_incompatible = create_test_audio(10.0, 140.0)  # Incompatible tempo
    
    # Extract features
    beat_extractor = BeatGridExtractor()
    energy_extractor = EnergyProfileExtractor()
    key_extractor = MusicalKeyExtractor()
    param_calculator = ProcessingParameterCalculator()
    envelope_designer = CrossfadeEnvelopeDesigner()
    
    beat_grid_a = beat_extractor(audio_a)
    beat_grid_b_good = beat_extractor(audio_b_compatible)
    beat_grid_b_bad = beat_extractor(audio_b_incompatible)
    
    energy_a = energy_extractor(audio_a)
    energy_b_good = energy_extractor(audio_b_compatible)
    energy_b_bad = energy_extractor(audio_b_incompatible)
    
    key_a = key_extractor(audio_a)
    key_b_good = key_extractor(audio_b_compatible)
    key_b_bad = key_extractor(audio_b_incompatible)
    
    # Test processing parameters
    params_good = param_calculator(beat_grid_a, beat_grid_b_good, key_a, key_b_good, energy_a, energy_b_good)
    params_bad = param_calculator(beat_grid_a, beat_grid_b_bad, key_a, key_b_bad, energy_a, energy_b_bad)
    
    # Test crossfade envelope
    envelope = envelope_designer(
        beat_grid_a, beat_grid_b_good, energy_a, energy_b_good,
        5*44100, 2*44100, 2*44100  # Crossfade specs
    )
    
    # Test smoothness prediction
    predictor = TransitionSmoothnessPredictions()
    
    # Test good transition
    prediction_good = predictor(
        beat_grid_a, beat_grid_b_good, energy_a, energy_b_good,
        key_a, key_b_good, params_good, envelope,
        5*44100, 2*44100
    )
    
    print("=== Good Transition Prediction ===")
    print(predictor.get_quality_summary(prediction_good))
    
    # Test bad transition
    prediction_bad = predictor(
        beat_grid_a, beat_grid_b_bad, energy_a, energy_b_bad,
        key_a, key_b_bad, params_bad, envelope,
        5*44100, 2*44100
    )
    
    print("\n=== Bad Transition Prediction ===")
    print(predictor.get_quality_summary(prediction_bad))