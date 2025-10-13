"""
ProcessingParameterCalculator - Calculate exact processing parameters for Track B

This module determines the precise pitch shift, rate change, and timing alignment
parameters needed to transform Track B for optimal crossfade compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from .splice_point_optimizer import OptimalSplice
from .harmonic_compatibility_analyzer import HarmonicMatch
from .rhythmic_compatibility_analyzer import RhythmicMatch
from .energy_compatibility_analyzer import EnergyMatch


class ProcessingStrategy(Enum):
    """Processing strategy options"""
    NO_PROCESSING = "no_processing"
    PITCH_ONLY = "pitch_only"  
    TEMPO_ONLY = "tempo_only"
    PITCH_AND_TEMPO = "pitch_and_tempo"
    HARD_CUT = "hard_cut"


@dataclass
class ProcessingParams:
    """Processing parameter specification"""
    pitch_shift_semitones: float     # Semitone adjustment for B (+/- 12)
    rate_change_ratio: float         # Tempo ratio for B (1.0 = no change)
    timing_offset_samples: int       # Sample alignment offset
    gain_adjustment_db: float        # Level adjustment for B
    strategy: ProcessingStrategy     # Processing approach
    artifact_prediction: float      # Expected artifact level [0,1]
    quality_score: float            # Expected result quality [0,1]
    processing_confidence: float     # Confidence in parameters [0,1]
    fallback_params: Optional[Dict] # Alternative parameters if primary fails


class ProcessingParameterCalculator(nn.Module):
    """
    Calculate exact processing parameters for Track B transformation.
    
    Takes compatibility analysis results and optimal splice selection
    to determine precise audio processing parameters.
    """
    
    def __init__(self, 
                 max_pitch_shift: float = 2.0,
                 max_rate_change: float = 0.05,
                 artifact_threshold: float = 0.3,
                 quality_threshold: float = 0.5):
        super().__init__()
        
        self.max_pitch_shift = max_pitch_shift
        self.max_rate_change = max_rate_change  # ±5%
        self.artifact_threshold = artifact_threshold
        self.quality_threshold = quality_threshold
        
        # Neural network for artifact prediction
        self.artifact_predictor = nn.Sequential(
            nn.Linear(8, 64),  # Processing parameters + audio features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Quality assessment network
        self.quality_assessor = nn.Sequential(
            nn.Linear(6, 32),  # Quality factors
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Parameter optimization network
        self.parameter_optimizer = nn.Sequential(
            nn.Linear(10, 64),  # Compatibility scores + constraints
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)   # pitch, rate, timing, gain adjustments
        )

    def forward(self, 
                optimal_splice: OptimalSplice,
                harmonic_match: HarmonicMatch,
                rhythmic_match: RhythmicMatch,
                energy_match: EnergyMatch) -> ProcessingParams:
        """
        Calculate exact processing parameters for Track B.
        
        Args:
            optimal_splice: Selected splice point combination
            harmonic_match: Harmonic compatibility analysis
            rhythmic_match: Rhythmic compatibility analysis
            energy_match: Energy compatibility analysis
            
        Returns:
            ProcessingParams with exact transformation specifications
        """
        
        # Determine processing strategy
        strategy = self._determine_processing_strategy(
            harmonic_match, rhythmic_match, energy_match
        )
        
        if strategy == ProcessingStrategy.HARD_CUT:
            return self._create_hard_cut_params(optimal_splice, rhythmic_match)
        
        # Calculate base parameters from compatibility analysis
        pitch_shift = self._calculate_pitch_shift(harmonic_match, strategy)
        rate_change = self._calculate_rate_change(rhythmic_match, strategy)
        timing_offset = self._calculate_timing_offset(optimal_splice, rhythmic_match)
        gain_adjustment = self._calculate_gain_adjustment(energy_match)
        
        # Optimize parameters using neural network
        optimized_params = self._optimize_parameters_neural(
            pitch_shift, rate_change, timing_offset, gain_adjustment,
            harmonic_match, rhythmic_match, energy_match
        )
        
        # Predict artifacts and quality
        artifact_prediction = self._predict_artifacts(
            optimized_params, harmonic_match, rhythmic_match, energy_match
        )
        
        quality_score = self._assess_processing_quality(
            optimized_params, artifact_prediction, 
            harmonic_match, rhythmic_match, energy_match
        )
        
        # Calculate confidence
        processing_confidence = self._calculate_processing_confidence(
            optimized_params, artifact_prediction, quality_score,
            harmonic_match, rhythmic_match, energy_match
        )
        
        # Generate fallback parameters
        fallback_params = self._generate_fallback_params(
            optimized_params, quality_score, artifact_prediction
        )
        
        return ProcessingParams(
            pitch_shift_semitones=optimized_params['pitch_shift'],
            rate_change_ratio=optimized_params['rate_change'],
            timing_offset_samples=optimized_params['timing_offset'],
            gain_adjustment_db=optimized_params['gain_adjustment'],
            strategy=strategy,
            artifact_prediction=float(artifact_prediction),
            quality_score=float(quality_score),
            processing_confidence=float(processing_confidence),
            fallback_params=fallback_params
        )
    
    def _determine_processing_strategy(self, 
                                     harmonic_match: HarmonicMatch,
                                     rhythmic_match: RhythmicMatch,
                                     energy_match: EnergyMatch) -> ProcessingStrategy:
        """Determine optimal processing strategy"""
        
        # Check if hard cut is recommended
        if (rhythmic_match.processing_recommendation.startswith("hard_cut") or
            harmonic_match.processing_recommendation.startswith("reject") or
            energy_match.processing_recommendation.startswith("reject")):
            return ProcessingStrategy.HARD_CUT
        
        needs_pitch = (abs(harmonic_match.required_pitch_shift) > 0.5 and 
                      harmonic_match.processing_recommendation != "skip_pitch_correction_low_relevance")
        
        needs_tempo = (abs(rhythmic_match.required_rate_change - 1.0) > 0.01 and
                      not rhythmic_match.processing_recommendation.startswith("no_tempo"))
        
        if needs_pitch and needs_tempo:
            return ProcessingStrategy.PITCH_AND_TEMPO
        elif needs_pitch:
            return ProcessingStrategy.PITCH_ONLY
        elif needs_tempo:
            return ProcessingStrategy.TEMPO_ONLY
        else:
            return ProcessingStrategy.NO_PROCESSING
    
    def _calculate_pitch_shift(self, 
                             harmonic_match: HarmonicMatch,
                             strategy: ProcessingStrategy) -> float:
        """Calculate pitch shift parameter"""
        
        if strategy in [ProcessingStrategy.PITCH_ONLY, ProcessingStrategy.PITCH_AND_TEMPO]:
            # Use recommended shift, but clamp to limits
            pitch_shift = harmonic_match.required_pitch_shift
            pitch_shift = max(-self.max_pitch_shift, min(self.max_pitch_shift, pitch_shift))
            return float(pitch_shift)
        else:
            return 0.0
    
    def _calculate_rate_change(self, 
                             rhythmic_match: RhythmicMatch,
                             strategy: ProcessingStrategy) -> float:
        """Calculate tempo rate change parameter"""
        
        if strategy in [ProcessingStrategy.TEMPO_ONLY, ProcessingStrategy.PITCH_AND_TEMPO]:
            # Use recommended rate change, but clamp to limits
            rate_change = rhythmic_match.required_rate_change
            max_rate = 1.0 + self.max_rate_change
            min_rate = 1.0 - self.max_rate_change
            rate_change = max(min_rate, min(max_rate, rate_change))
            return float(rate_change)
        else:
            return 1.0
    
    def _calculate_timing_offset(self, 
                               optimal_splice: OptimalSplice,
                               rhythmic_match: RhythmicMatch) -> int:
        """Calculate sample-accurate timing offset"""
        
        # Basic timing offset based on splice points
        # This would be refined based on beat grid analysis
        base_offset = 0
        
        # Adjust for phase alignment
        phase_adjustment = int(rhythmic_match.phase_offset * 1000)  # Convert to samples
        
        # Final timing offset
        timing_offset = base_offset + phase_adjustment
        
        return timing_offset
    
    def _calculate_gain_adjustment(self, energy_match: EnergyMatch) -> float:
        """Calculate gain adjustment parameter"""
        
        # Use recommended loudness adjustment
        gain_adjustment = energy_match.loudness_adjustment_db
        
        # Clamp to reasonable range
        gain_adjustment = max(-12.0, min(12.0, gain_adjustment))
        
        return float(gain_adjustment)
    
    def _optimize_parameters_neural(self, 
                                  pitch_shift: float,
                                  rate_change: float,
                                  timing_offset: int,
                                  gain_adjustment: float,
                                  harmonic_match: HarmonicMatch,
                                  rhythmic_match: RhythmicMatch,
                                  energy_match: EnergyMatch) -> Dict[str, float]:
        """Use neural network to optimize parameters"""
        
        features = torch.tensor([
            pitch_shift / self.max_pitch_shift,  # Normalize
            (rate_change - 1.0) / self.max_rate_change,
            timing_offset / 10000.0,  # Rough normalization
            gain_adjustment / 12.0,
            harmonic_match.compatibility_score,
            rhythmic_match.compatibility_score,
            energy_match.compatibility_score,
            harmonic_match.dissonance_risk,
            rhythmic_match.correction_difficulty,
            energy_match.artifact_prediction if hasattr(energy_match, 'artifact_prediction') else 0.2
        ], dtype=torch.float32)
        
        try:
            with torch.no_grad():
                adjustments = self.parameter_optimizer(features.unsqueeze(0))
                adjustments = adjustments.squeeze()
                
                # Apply adjustments to base parameters
                optimized_pitch = pitch_shift + float(adjustments[0]) * 0.2  # Small adjustment
                optimized_rate = rate_change + float(adjustments[1]) * 0.01
                optimized_timing = timing_offset + int(adjustments[2] * 1000)
                optimized_gain = gain_adjustment + float(adjustments[3]) * 2.0
        except:
            # Fallback to original parameters
            optimized_pitch = pitch_shift
            optimized_rate = rate_change
            optimized_timing = timing_offset
            optimized_gain = gain_adjustment
        
        # Ensure parameters stay within limits
        optimized_pitch = max(-self.max_pitch_shift, min(self.max_pitch_shift, optimized_pitch))
        optimized_rate = max(1.0 - self.max_rate_change, min(1.0 + self.max_rate_change, optimized_rate))
        optimized_gain = max(-12.0, min(12.0, optimized_gain))
        
        return {
            'pitch_shift': optimized_pitch,
            'rate_change': optimized_rate,
            'timing_offset': optimized_timing,
            'gain_adjustment': optimized_gain
        }
    
    def _predict_artifacts(self, 
                         params: Dict[str, float],
                         harmonic_match: HarmonicMatch,
                         rhythmic_match: RhythmicMatch,
                         energy_match: EnergyMatch) -> float:
        """Predict artifact level from processing parameters"""
        
        features = torch.tensor([
            abs(params['pitch_shift']) / self.max_pitch_shift,
            abs(params['rate_change'] - 1.0) / self.max_rate_change,
            abs(params['gain_adjustment']) / 12.0,
            harmonic_match.dissonance_risk,
            rhythmic_match.correction_difficulty,
            harmonic_match.compatibility_score,
            rhythmic_match.tempo_stability_factor,
            energy_match.compatibility_score
        ], dtype=torch.float32)
        
        try:
            with torch.no_grad():
                artifact_level = self.artifact_predictor(features.unsqueeze(0))
                artifact_level = float(artifact_level.squeeze())
        except:
            # Fallback heuristic
            pitch_artifacts = abs(params['pitch_shift']) / self.max_pitch_shift * 0.3
            tempo_artifacts = abs(params['rate_change'] - 1.0) / self.max_rate_change * 0.2
            gain_artifacts = abs(params['gain_adjustment']) / 12.0 * 0.1
            
            artifact_level = pitch_artifacts + tempo_artifacts + gain_artifacts
            artifact_level = min(1.0, artifact_level)
        
        return artifact_level
    
    def _assess_processing_quality(self, 
                                 params: Dict[str, float],
                                 artifact_prediction: float,
                                 harmonic_match: HarmonicMatch,
                                 rhythmic_match: RhythmicMatch,
                                 energy_match: EnergyMatch) -> float:
        """Assess overall processing quality"""
        
        features = torch.tensor([
            1.0 - artifact_prediction,
            harmonic_match.compatibility_score,
            rhythmic_match.compatibility_score,
            energy_match.compatibility_score,
            harmonic_match.correction_confidence,
            rhythmic_match.tempo_stability_factor
        ], dtype=torch.float32)
        
        try:
            with torch.no_grad():
                quality = self.quality_assessor(features.unsqueeze(0))
                quality = float(quality.squeeze())
        except:
            # Fallback calculation
            compatibility_avg = (harmonic_match.compatibility_score + 
                               rhythmic_match.compatibility_score + 
                               energy_match.compatibility_score) / 3.0
            
            quality = compatibility_avg * (1.0 - artifact_prediction) * 0.8 + 0.2
        
        return quality
    
    def _calculate_processing_confidence(self, 
                                       params: Dict[str, float],
                                       artifact_prediction: float,
                                       quality_score: float,
                                       harmonic_match: HarmonicMatch,
                                       rhythmic_match: RhythmicMatch,
                                       energy_match: EnergyMatch) -> float:
        """Calculate confidence in processing parameters"""
        
        # Base confidence from compatibility analyses
        base_confidence = (
            harmonic_match.correction_confidence * 0.4 +
            rhythmic_match.tempo_stability_factor * 0.3 +
            energy_match.compatibility_score * 0.3
        )
        
        # Adjust for parameter extremes
        param_confidence = 1.0 - (
            abs(params['pitch_shift']) / self.max_pitch_shift * 0.2 +
            abs(params['rate_change'] - 1.0) / self.max_rate_change * 0.2
        )
        
        # Adjust for predicted quality
        quality_confidence = quality_score
        
        # Combined confidence
        confidence = (
            base_confidence * 0.4 +
            param_confidence * 0.3 +
            quality_confidence * 0.3
        )
        
        return max(0.0, min(1.0, confidence))
    
    def _generate_fallback_params(self, 
                                params: Dict[str, float],
                                quality_score: float,
                                artifact_prediction: float) -> Optional[Dict]:
        """Generate fallback parameters if primary processing fails"""
        
        if quality_score < self.quality_threshold or artifact_prediction > self.artifact_threshold:
            # Generate more conservative parameters
            fallback = {
                'pitch_shift_semitones': params['pitch_shift'] * 0.5,  # Reduce pitch shift
                'rate_change_ratio': 1.0 + (params['rate_change'] - 1.0) * 0.5,  # Reduce tempo change
                'timing_offset_samples': params['timing_offset'],  # Keep timing
                'gain_adjustment_db': params['gain_adjustment'] * 0.7,  # Reduce gain change
                'strategy': 'conservative',
                'reason': 'primary_parameters_risky'
            }
            return fallback
        
        return None
    
    def _create_hard_cut_params(self, 
                              optimal_splice: OptimalSplice,
                              rhythmic_match: RhythmicMatch) -> ProcessingParams:
        """Create parameters for hard cut strategy"""
        
        # No processing, just timing optimization
        return ProcessingParams(
            pitch_shift_semitones=0.0,
            rate_change_ratio=1.0,
            timing_offset_samples=0,  # Would be calculated from beat grid
            gain_adjustment_db=0.0,
            strategy=ProcessingStrategy.HARD_CUT,
            artifact_prediction=0.0,
            quality_score=0.8,  # High quality since no processing artifacts
            processing_confidence=0.9,
            fallback_params=None
        )
    
    def validate_parameters(self, params: ProcessingParams) -> Dict[str, Any]:
        """Validate processing parameters for safety"""
        
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check parameter limits
        if abs(params.pitch_shift_semitones) > self.max_pitch_shift:
            validation_results['errors'].append(
                f"Pitch shift {params.pitch_shift_semitones:.1f} exceeds limit ±{self.max_pitch_shift}"
            )
            validation_results['valid'] = False
        
        if abs(params.rate_change_ratio - 1.0) > self.max_rate_change:
            validation_results['errors'].append(
                f"Rate change {params.rate_change_ratio:.3f} exceeds limit ±{self.max_rate_change}"
            )
            validation_results['valid'] = False
        
        # Check for quality warnings
        if params.artifact_prediction > self.artifact_threshold:
            validation_results['warnings'].append(
                f"High artifact prediction: {params.artifact_prediction:.2f}"
            )
        
        if params.quality_score < self.quality_threshold:
            validation_results['warnings'].append(
                f"Low quality score: {params.quality_score:.2f}"
            )
        
        if params.processing_confidence < 0.5:
            validation_results['warnings'].append(
                f"Low processing confidence: {params.processing_confidence:.2f}"
            )
        
        return validation_results
    
    def get_processing_summary(self, params: ProcessingParams) -> str:
        """Generate human-readable processing summary"""
        
        if params.strategy == ProcessingStrategy.NO_PROCESSING:
            return "No processing required - tracks are already compatible"
        elif params.strategy == ProcessingStrategy.HARD_CUT:
            return "Hard cut recommended - no audio processing applied"
        else:
            summary = f"Track B processing: "
            
            if abs(params.pitch_shift_semitones) > 0.1:
                summary += f"{params.pitch_shift_semitones:+.1f} semitones pitch shift, "
            
            if abs(params.rate_change_ratio - 1.0) > 0.01:
                rate_percent = (params.rate_change_ratio - 1.0) * 100
                summary += f"{rate_percent:+.1f}% tempo change, "
            
            if abs(params.gain_adjustment_db) > 0.1:
                summary += f"{params.gain_adjustment_db:+.1f}dB gain adjustment, "
            
            # Remove trailing comma and space
            summary = summary.rstrip(", ")
            
            summary += f" (quality: {params.quality_score:.2f}, confidence: {params.processing_confidence:.2f})"
            
            return summary


if __name__ == "__main__":
    # Test the module with mock data
    from .splice_point_optimizer import OptimalSplice
    from .harmonic_compatibility_analyzer import HarmonicMatch, CompatibilityLevel
    from .rhythmic_compatibility_analyzer import RhythmicMatch, TempoCompatibility
    from .energy_compatibility_analyzer import EnergyMatch, EnergyCompatibilityLevel
    
    # Create mock data
    optimal_splice = OptimalSplice(
        a_exit_sample=100000,
        b_entry_sample=50000,
        quality_score=0.8,
        musical_compatibility=0.75,
        processing_quality=0.8,
        confidence=0.85,
        optimization_details={},
        fallback_strategy=None
    )
    
    harmonic_match = HarmonicMatch(
        compatibility_score=0.7,
        compatibility_level=CompatibilityLevel.GOOD,
        required_pitch_shift=1.5,
        dissonance_risk=0.2,
        key_relationship="perfect_fifth_related",
        correction_confidence=0.8,
        harmonic_tension=0.3,
        processing_recommendation="moderate_correction_+1.5_semitones"
    )
    
    rhythmic_match = RhythmicMatch(
        compatibility_score=0.8,
        compatibility_level=TempoCompatibility.EXCELLENT,
        required_rate_change=1.03,
        phase_offset=0.1,
        bpm_difference=3.0,
        tempo_stability_factor=0.9,
        correction_difficulty=0.2,
        processing_recommendation="light_tempo_correction_+3.0_percent"
    )
    
    energy_match = EnergyMatch(
        compatibility_score=0.75,
        compatibility_level=EnergyCompatibilityLevel.GOOD,
        level_difference_db=-2.5,
        spectral_balance_score=0.8,
        loudness_adjustment_db=-2.0,
        dynamic_range_compatibility=0.7,
        energy_flow_score=0.8,
        processing_recommendation="moderate_level_adjustment_-2.5_db"
    )
    
    # Test calculator
    calculator = ProcessingParameterCalculator()
    
    params = calculator(optimal_splice, harmonic_match, rhythmic_match, energy_match)
    
    print("=== Processing Parameters ===")
    print(f"Strategy: {params.strategy.value}")
    print(f"Pitch shift: {params.pitch_shift_semitones:+.2f} semitones")
    print(f"Rate change: {params.rate_change_ratio:.4f} ({(params.rate_change_ratio-1)*100:+.1f}%)")
    print(f"Timing offset: {params.timing_offset_samples} samples")
    print(f"Gain adjustment: {params.gain_adjustment_db:+.1f} dB")
    print(f"Artifact prediction: {params.artifact_prediction:.3f}")
    print(f"Quality score: {params.quality_score:.3f}")
    print(f"Confidence: {params.processing_confidence:.3f}")
    
    # Test validation
    validation = calculator.validate_parameters(params)
    print(f"\nValidation: {'PASS' if validation['valid'] else 'FAIL'}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")
    if validation['errors']:
        print(f"Errors: {validation['errors']}")
    
    # Test summary
    summary = calculator.get_processing_summary(params)
    print(f"\nSummary: {summary}")