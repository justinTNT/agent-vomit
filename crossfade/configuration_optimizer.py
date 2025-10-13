"""
ConfigurationOptimizer - Optimize complete crossfade configuration

This module integrates all Phase 1.5 analysis components to generate
optimal crossfade configurations through multi-objective optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Import all Phase 1.5 modules for integration
from .low_end_conflict_analyzer import LowEndConflictAnalyzer, LowEndConflict, ConflictSeverity
from .spectral_matching_eq import SpectralMatchingEQ, SpectralMatch, EQStrategy
from .fallback_strategy_optimizer import FallbackStrategyOptimizer, FallbackPlan, FallbackStrategy
from .transition_smoothness_predictor import TransitionSmoothnessPredictions, SmoothnessPrediction, TransitionQuality
from .adaptive_threshold_calculator import AdaptiveThresholdCalculator, ThresholdSet, AdaptationStrategy

# Import Phase 1 modules for complete analysis
from .beat_grid_extractor import BeatGrid
from .energy_profile_extractor import EnergyProfile
from .musical_key_extractor import KeyProfile
from .processing_parameter_calculator import ProcessingParams
from .crossfade_envelope_designer import CrossfadeEnvelope


class OptimizationObjective(Enum):
    """Optimization objectives"""
    MAXIMIZE_QUALITY = "maximize_quality"           # Best possible quality
    MINIMIZE_ARTIFACTS = "minimize_artifacts"       # Fewest artifacts
    BALANCE_QUALITY_SPEED = "balance_quality_speed" # Balance quality vs processing time
    MAXIMIZE_MUSICALITY = "maximize_musicality"     # Most musical transition
    ADAPTIVE_CONTEXT = "adaptive_context"           # Context-dependent optimization


class ConfigurationStrategy(Enum):
    """Overall configuration strategy"""
    CONSERVATIVE_QUALITY = "conservative_quality"   # Prioritize quality over everything
    AGGRESSIVE_CREATIVE = "aggressive_creative"     # Allow creative/experimental transitions  
    BALANCED_PERFORMANCE = "balanced_performance"   # Balance all factors
    FALLBACK_FOCUSED = "fallback_focused"          # Focus on reliable fallbacks
    AUTO_ADAPTIVE = "auto_adaptive"                # Fully automatic optimization


@dataclass
class OptimalConfiguration:
    """Complete optimized crossfade configuration"""
    # Core configuration
    crossfade_start_a: int                    # Optimal start position in Track A
    crossfade_start_b: int                    # Optimal start position in Track B
    crossfade_duration: int                   # Optimal duration in samples
    processing_params: ProcessingParams        # Optimal processing parameters
    crossfade_envelope: CrossfadeEnvelope     # Optimal crossfade curve
    
    # Enhanced configuration (Phase 1.5)
    low_end_strategy: LowEndConflict          # Low-end handling strategy
    spectral_matching: SpectralMatch          # Spectral EQ strategy  
    fallback_plan: Optional[FallbackPlan]     # Fallback if main strategy fails
    smoothness_prediction: SmoothnessPrediction  # Quality prediction
    adaptive_thresholds: ThresholdSet         # Context-adaptive thresholds
    
    # Optimization results
    overall_quality_score: float              # Combined quality score [0,1]
    optimization_confidence: float            # Confidence in optimization [0,1]
    estimated_processing_time: float          # Processing time estimate (seconds)
    configuration_summary: str                # Human-readable summary
    alternative_configurations: List[Dict]    # Other viable configurations


class ConfigurationOptimizer(nn.Module):
    """
    Optimize complete crossfade configuration using all analysis modules.
    
    Integrates Phase 1 core modules and Phase 1.5 enhanced modules for
    comprehensive crossfade optimization with multiple objectives.
    """
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 optimization_iterations: int = 5,
                 quality_weight: float = 0.4,
                 musicality_weight: float = 0.3,
                 artifact_weight: float = 0.3):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.optimization_iterations = optimization_iterations
        self.quality_weight = quality_weight
        self.musicality_weight = musicality_weight  
        self.artifact_weight = artifact_weight
        
        # Initialize all Phase 1.5 analyzers
        self.low_end_analyzer = LowEndConflictAnalyzer(sample_rate=sample_rate)
        self.spectral_matcher = SpectralMatchingEQ(sample_rate=sample_rate)
        self.fallback_optimizer = FallbackStrategyOptimizer(sample_rate=sample_rate)
        self.smoothness_predictor = TransitionSmoothnessPredictions(sample_rate=sample_rate)
        self.threshold_calculator = AdaptiveThresholdCalculator()
        
        # Multi-objective optimization network
        self.config_optimizer = nn.Sequential(
            nn.Linear(25, 128),  # Multi-dimensional input
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),   # Configuration parameters
            nn.Tanh()            # Normalized parameter adjustments
        )
        
        # Configuration scoring network
        self.config_scorer = nn.Sequential(
            nn.Linear(20, 64),   # Configuration quality features
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()         # Overall quality score
        )

    def forward(self, 
                audio_a: torch.Tensor,
                audio_b: torch.Tensor,
                beat_grid_a: BeatGrid,
                beat_grid_b: BeatGrid,
                energy_profile_a: EnergyProfile,
                energy_profile_b: EnergyProfile,
                key_profile_a: Optional[KeyProfile],
                key_profile_b: Optional[KeyProfile],
                initial_processing_params: ProcessingParams,
                initial_crossfade_envelope: CrossfadeEnvelope,
                initial_crossfade_start_a: int,
                initial_crossfade_start_b: int,
                objective: OptimizationObjective = OptimizationObjective.BALANCE_QUALITY_SPEED,
                strategy: ConfigurationStrategy = ConfigurationStrategy.BALANCED_PERFORMANCE) -> OptimalConfiguration:
        """
        Optimize complete crossfade configuration.
        
        Args:
            audio_a: Track A audio
            audio_b: Track B audio  
            beat_grid_a: Track A beat analysis
            beat_grid_b: Track B beat analysis
            energy_profile_a: Track A energy analysis
            energy_profile_b: Track B energy analysis
            key_profile_a: Track A key analysis (optional)
            key_profile_b: Track B key analysis (optional)
            initial_processing_params: Starting processing parameters
            initial_crossfade_envelope: Starting crossfade envelope
            initial_crossfade_start_a: Starting crossfade position A
            initial_crossfade_start_b: Starting crossfade position B
            objective: Optimization objective
            strategy: Configuration strategy
            
        Returns:
            OptimalConfiguration with optimized parameters and analysis
        """
        
        # Phase 1: Calculate adaptive thresholds based on content
        adaptation_strategy = self._map_strategy_to_adaptation(strategy)
        adaptive_thresholds = self.threshold_calculator(
            beat_grid_a, beat_grid_b, energy_profile_a, energy_profile_b,
            key_profile_a, key_profile_b, adaptation_strategy
        )
        
        # Phase 2: Comprehensive analysis with all Phase 1.5 modules
        analysis_results = self._comprehensive_analysis(
            audio_a, audio_b, beat_grid_a, beat_grid_b,
            energy_profile_a, energy_profile_b, key_profile_a, key_profile_b,
            initial_processing_params, initial_crossfade_envelope,
            initial_crossfade_start_a, initial_crossfade_start_b
        )
        
        # Phase 3: Multi-objective optimization
        optimal_config = self._optimize_configuration(
            analysis_results, adaptive_thresholds, objective, strategy,
            initial_processing_params, initial_crossfade_envelope,
            initial_crossfade_start_a, initial_crossfade_start_b
        )
        
        # Phase 4: Generate alternative configurations
        alternatives = self._generate_alternative_configurations(
            analysis_results, adaptive_thresholds, optimal_config, objective
        )
        
        # Phase 5: Final quality assessment
        final_quality_score = self._calculate_final_quality_score(optimal_config)
        optimization_confidence = self._calculate_optimization_confidence(
            optimal_config, alternatives, analysis_results
        )
        
        # Phase 6: Generate configuration summary
        config_summary = self._generate_configuration_summary(
            optimal_config, analysis_results, objective, strategy
        )
        
        return OptimalConfiguration(
            crossfade_start_a=optimal_config['crossfade_start_a'],
            crossfade_start_b=optimal_config['crossfade_start_b'],
            crossfade_duration=optimal_config['crossfade_duration'],
            processing_params=optimal_config['processing_params'],
            crossfade_envelope=optimal_config['crossfade_envelope'],
            low_end_strategy=analysis_results['low_end_conflict'],
            spectral_matching=analysis_results['spectral_match'],
            fallback_plan=analysis_results['fallback_plan'],
            smoothness_prediction=optimal_config['smoothness_prediction'],
            adaptive_thresholds=adaptive_thresholds,
            overall_quality_score=float(final_quality_score),
            optimization_confidence=float(optimization_confidence),
            estimated_processing_time=self._estimate_processing_time(optimal_config),
            configuration_summary=config_summary,
            alternative_configurations=alternatives
        )
    
    def _map_strategy_to_adaptation(self, strategy: ConfigurationStrategy) -> AdaptationStrategy:
        """Map configuration strategy to threshold adaptation strategy"""
        
        mapping = {
            ConfigurationStrategy.CONSERVATIVE_QUALITY: AdaptationStrategy.CONSERVATIVE,
            ConfigurationStrategy.AGGRESSIVE_CREATIVE: AdaptationStrategy.PERMISSIVE,
            ConfigurationStrategy.BALANCED_PERFORMANCE: AdaptationStrategy.BALANCED,
            ConfigurationStrategy.FALLBACK_FOCUSED: AdaptationStrategy.CONSERVATIVE,
            ConfigurationStrategy.AUTO_ADAPTIVE: AdaptationStrategy.CONTEXT_ADAPTIVE
        }
        
        return mapping[strategy]
    
    def _comprehensive_analysis(self, 
                               audio_a: torch.Tensor,
                               audio_b: torch.Tensor,
                               beat_grid_a: BeatGrid,
                               beat_grid_b: BeatGrid,
                               energy_profile_a: EnergyProfile,
                               energy_profile_b: EnergyProfile,
                               key_profile_a: Optional[KeyProfile],
                               key_profile_b: Optional[KeyProfile],
                               processing_params: ProcessingParams,
                               crossfade_envelope: CrossfadeEnvelope,
                               crossfade_start_a: int,
                               crossfade_start_b: int) -> Dict[str, Any]:
        """Perform comprehensive analysis using all Phase 1.5 modules"""
        
        results = {}
        
        # Low-end conflict analysis
        results['low_end_conflict'] = self.low_end_analyzer(
            audio_a, audio_b, crossfade_start_a, crossfade_start_b, 
            crossfade_envelope.duration
        )
        
        # Spectral matching analysis
        results['spectral_match'] = self.spectral_matcher(
            energy_profile_a, energy_profile_b,
            crossfade_start_a, crossfade_start_b, crossfade_envelope.duration,
            results['low_end_conflict']
        )
        
        # Fallback strategy optimization (if needed)
        if processing_params.strategy.value == 'none' or processing_params.artifact_likelihood > 0.7:
            results['fallback_plan'] = self.fallback_optimizer(
                beat_grid_a, beat_grid_b, energy_profile_a, energy_profile_b,
                None,  # Would need rhythmic compatibility analyzer from Phase 1
                crossfade_start_a, crossfade_start_b
            )
        else:
            results['fallback_plan'] = None
        
        # Transition smoothness prediction
        results['smoothness_prediction'] = self.smoothness_predictor(
            beat_grid_a, beat_grid_b, energy_profile_a, energy_profile_b,
            key_profile_a, key_profile_b, processing_params, crossfade_envelope,
            crossfade_start_a, crossfade_start_b
        )
        
        return results
    
    def _optimize_configuration(self, 
                               analysis_results: Dict[str, Any],
                               adaptive_thresholds: ThresholdSet,
                               objective: OptimizationObjective,
                               strategy: ConfigurationStrategy,
                               initial_processing_params: ProcessingParams,
                               initial_crossfade_envelope: CrossfadeEnvelope,
                               initial_crossfade_start_a: int,
                               initial_crossfade_start_b: int) -> Dict[str, Any]:
        """Perform multi-objective optimization"""
        
        # Start with initial configuration
        best_config = {
            'crossfade_start_a': initial_crossfade_start_a,
            'crossfade_start_b': initial_crossfade_start_b,
            'crossfade_duration': initial_crossfade_envelope.duration,
            'processing_params': initial_processing_params,
            'crossfade_envelope': initial_crossfade_envelope,
            'smoothness_prediction': analysis_results['smoothness_prediction']
        }
        
        best_score = self._score_configuration(best_config, analysis_results, adaptive_thresholds, objective)
        
        # Iterative optimization
        for iteration in range(self.optimization_iterations):
            # Generate configuration variations
            variations = self._generate_configuration_variations(
                best_config, analysis_results, adaptive_thresholds, strategy
            )
            
            # Evaluate each variation
            for variation in variations:
                # Re-analyze with new parameters (simplified for performance)
                variation_score = self._score_configuration(
                    variation, analysis_results, adaptive_thresholds, objective
                )
                
                if variation_score > best_score:
                    best_config = variation
                    best_score = variation_score
        
        return best_config
    
    def _score_configuration(self, 
                           config: Dict[str, Any],
                           analysis_results: Dict[str, Any],
                           adaptive_thresholds: ThresholdSet,
                           objective: OptimizationObjective) -> float:
        """Score a configuration based on the optimization objective"""
        
        # Base scores from analysis
        smoothness_score = config['smoothness_prediction'].smoothness_score
        quality_score = 1.0 - config['processing_params'].artifact_likelihood
        
        # Low-end conflict penalty
        low_end_penalty = self._calculate_low_end_penalty(analysis_results['low_end_conflict'])
        
        # Spectral matching bonus
        spectral_bonus = analysis_results['spectral_match'].processing_confidence
        
        # Threshold compliance bonus
        threshold_bonus = self._calculate_threshold_compliance(config, adaptive_thresholds)
        
        # Apply objective-specific weighting
        if objective == OptimizationObjective.MAXIMIZE_QUALITY:
            score = (
                smoothness_score * 0.4 +
                quality_score * 0.3 +
                spectral_bonus * 0.2 +
                threshold_bonus * 0.1 -
                low_end_penalty * 0.3
            )
            
        elif objective == OptimizationObjective.MINIMIZE_ARTIFACTS:
            artifact_score = 1.0 - len(config['smoothness_prediction'].predicted_artifacts) / 10.0
            score = (
                artifact_score * 0.5 +
                quality_score * 0.3 +
                smoothness_score * 0.2 -
                low_end_penalty * 0.2
            )
            
        elif objective == OptimizationObjective.BALANCE_QUALITY_SPEED:
            processing_complexity = self._estimate_processing_complexity(config)
            speed_score = max(0.0, 1.0 - processing_complexity / 5.0)  # Normalize complexity
            score = (
                smoothness_score * 0.3 +
                quality_score * 0.25 +
                speed_score * 0.25 +
                spectral_bonus * 0.2 -
                low_end_penalty * 0.2
            )
            
        elif objective == OptimizationObjective.MAXIMIZE_MUSICALITY:
            # Focus on musical aspects
            harmonic_score = config['smoothness_prediction'].factor_scores.get(
                'HARMONIC_COMPATIBILITY', 0.5
            )  # This would need proper enum handling
            rhythmic_score = config['smoothness_prediction'].factor_scores.get(
                'RHYTHMIC_ALIGNMENT', 0.5
            )
            score = (
                harmonic_score * 0.3 +
                rhythmic_score * 0.3 +
                smoothness_score * 0.25 +
                spectral_bonus * 0.15 -
                low_end_penalty * 0.1
            )
            
        else:  # ADAPTIVE_CONTEXT
            # Context-dependent scoring based on content analysis
            content_factors = adaptive_thresholds.content_analysis
            if content_factors.get('harmonic_relevance', 0.5) > 0.7:
                # Harmonic music - prioritize harmonic compatibility
                score = smoothness_score * 0.5 + quality_score * 0.3 + spectral_bonus * 0.2
            elif content_factors.get('rhythmic_complexity', 0.5) > 0.7:
                # Complex rhythms - prioritize rhythmic alignment  
                score = smoothness_score * 0.4 + quality_score * 0.4 + threshold_bonus * 0.2
            else:
                # Default balanced scoring
                score = smoothness_score * 0.4 + quality_score * 0.3 + spectral_bonus * 0.3
        
        return max(0.0, min(1.0, score))
    
    def _calculate_low_end_penalty(self, low_end_conflict: LowEndConflict) -> float:
        """Calculate penalty for low-end conflicts"""
        
        severity_penalties = {
            ConflictSeverity.NONE: 0.0,
            ConflictSeverity.MILD: 0.1,
            ConflictSeverity.MODERATE: 0.25,
            ConflictSeverity.SEVERE: 0.5,
            ConflictSeverity.CRITICAL: 0.8
        }
        
        return severity_penalties[low_end_conflict.severity]
    
    def _calculate_threshold_compliance(self, config: Dict[str, Any], thresholds: ThresholdSet) -> float:
        """Calculate bonus for meeting adaptive thresholds"""
        
        compliance_score = 0.0
        
        # Check processing quality threshold
        if config['processing_params'].confidence >= thresholds.processing_quality_min:
            compliance_score += 0.3
        
        # Check crossfade duration compliance
        duration_ms = config['crossfade_duration'] * 1000 / self.sample_rate
        if thresholds.crossfade_duration_ms_range[0] <= duration_ms <= thresholds.crossfade_duration_ms_range[1]:
            compliance_score += 0.3
        
        # Check smoothness threshold (implicit)
        if config['smoothness_prediction'].smoothness_score > 0.6:  # Reasonable threshold
            compliance_score += 0.4
        
        return compliance_score
    
    def _estimate_processing_complexity(self, config: Dict[str, Any]) -> float:
        """Estimate processing complexity/time"""
        
        complexity = 1.0  # Base complexity
        
        # Processing parameter complexity
        params = config['processing_params']
        if params.pitch_adjustment != 0:
            complexity += abs(params.pitch_adjustment) / 2.0
        if params.tempo_adjustment != 1.0:
            complexity += abs(params.tempo_adjustment - 1.0) * 5.0
        
        # Crossfade complexity
        duration_seconds = config['crossfade_duration'] / self.sample_rate
        complexity += duration_seconds / 4.0  # 4 seconds = 1.0 complexity unit
        
        # Envelope complexity (curve type)
        if hasattr(config['crossfade_envelope'], 'curve_type'):
            if 'frequency_selective' in str(config['crossfade_envelope'].curve_type):
                complexity += 1.0
        
        return complexity
    
    def _generate_configuration_variations(self, 
                                         base_config: Dict[str, Any],
                                         analysis_results: Dict[str, Any],
                                         adaptive_thresholds: ThresholdSet,
                                         strategy: ConfigurationStrategy) -> List[Dict[str, Any]]:
        """Generate configuration variations for optimization"""
        
        variations = []
        
        # Duration variations
        base_duration = base_config['crossfade_duration']
        duration_variations = [
            int(base_duration * 0.8),
            int(base_duration * 1.2),
            int(base_duration * 0.6),
            int(base_duration * 1.5)
        ]
        
        for duration in duration_variations:
            # Ensure within threshold limits
            duration_ms = duration * 1000 / self.sample_rate
            if (adaptive_thresholds.crossfade_duration_ms_range[0] <= duration_ms <= 
                adaptive_thresholds.crossfade_duration_ms_range[1]):
                
                variation = base_config.copy()
                variation['crossfade_duration'] = duration
                # Would need to re-create envelope with new duration
                variations.append(variation)
        
        # Processing parameter variations (if strategy allows)
        if strategy != ConfigurationStrategy.CONSERVATIVE_QUALITY:
            params = base_config['processing_params']
            
            # Pitch adjustment variations
            if abs(params.pitch_adjustment) < 1.5:  # Only if not already extreme
                for pitch_adj in [-0.5, 0.5, -1.0, 1.0]:
                    if abs(params.pitch_adjustment + pitch_adj) <= 2.0:  # Stay within limits
                        variation = base_config.copy()
                        # Would need to create new ProcessingParams
                        variations.append(variation)
        
        # Position variations (small adjustments)
        beat_interval = int(60.0 * self.sample_rate / 120.0)  # Approximate
        for offset in [-beat_interval//2, beat_interval//2, -beat_interval, beat_interval]:
            variation = base_config.copy()
            variation['crossfade_start_a'] = max(0, base_config['crossfade_start_a'] + offset)
            variation['crossfade_start_b'] = max(0, base_config['crossfade_start_b'] + offset)
            variations.append(variation)
        
        return variations[:10]  # Limit to 10 variations for performance
    
    def _generate_alternative_configurations(self, 
                                           analysis_results: Dict[str, Any],
                                           adaptive_thresholds: ThresholdSet,
                                           optimal_config: Dict[str, Any],
                                           objective: OptimizationObjective) -> List[Dict]:
        """Generate alternative viable configurations"""
        
        alternatives = []
        
        # Conservative alternative (always include)
        conservative_alt = {
            'name': 'Conservative Quality',
            'description': 'Prioritizes quality over creativity',
            'quality_score': optimal_config['smoothness_prediction'].smoothness_score * 0.9,
            'key_differences': [
                'Longer crossfade duration',
                'More conservative processing',
                'Higher quality thresholds'
            ]
        }
        alternatives.append(conservative_alt)
        
        # Fallback alternative (if fallback plan exists)
        if analysis_results['fallback_plan']:
            fallback_alt = {
                'name': 'Hard Cut Fallback',
                'description': f"Hard cut strategy: {analysis_results['fallback_plan'].strategy.value}",
                'quality_score': analysis_results['fallback_plan'].confidence * 0.7,
                'key_differences': [
                    'No tempo/pitch processing',
                    'Minimal crossfade duration',
                    'Beat-aligned timing'
                ]
            }
            alternatives.append(fallback_alt)
        
        # Aggressive alternative
        aggressive_alt = {
            'name': 'Creative Processing',
            'description': 'Allows more aggressive processing for creative effects',
            'quality_score': optimal_config['smoothness_prediction'].smoothness_score * 0.8,
            'key_differences': [
                'Higher processing intensity allowed',
                'More flexible thresholds',
                'Creative crossfade curves'
            ]
        }
        alternatives.append(aggressive_alt)
        
        # Spectral-focused alternative (if spectral issues detected)
        if analysis_results['spectral_match'].strategy != 'NO_EQ_NEEDED':  # String comparison for simplicity
            spectral_alt = {
                'name': 'Spectral Matching Focus',
                'description': 'Prioritizes frequency content matching',
                'quality_score': analysis_results['spectral_match'].processing_confidence,
                'key_differences': [
                    'EQ matching applied',
                    'Frequency-selective crossfade',
                    'Spectral optimization priority'
                ]
            }
            alternatives.append(spectral_alt)
        
        return alternatives[:5]  # Limit to 5 alternatives
    
    def _calculate_final_quality_score(self, config: Dict[str, Any]) -> float:
        """Calculate final overall quality score"""
        
        # Base smoothness score
        smoothness = config['smoothness_prediction'].smoothness_score
        
        # Processing quality
        processing_quality = config['processing_params'].confidence
        
        # Artifact penalty
        num_artifacts = len(config['smoothness_prediction'].predicted_artifacts)
        artifact_penalty = min(0.5, num_artifacts * 0.1)
        
        # Combined score
        final_score = (
            smoothness * 0.5 +
            processing_quality * 0.3 +
            (1.0 - artifact_penalty) * 0.2
        )
        
        return max(0.0, min(1.0, final_score))
    
    def _calculate_optimization_confidence(self, 
                                         optimal_config: Dict[str, Any],
                                         alternatives: List[Dict],
                                         analysis_results: Dict[str, Any]) -> float:
        """Calculate confidence in the optimization result"""
        
        # Base confidence from smoothness prediction
        prediction_confidence = optimal_config['smoothness_prediction'].confidence
        
        # Processing parameter confidence
        processing_confidence = optimal_config['processing_params'].confidence
        
        # Analysis quality confidence
        analysis_confidence = analysis_results['spectral_match'].processing_confidence
        
        # Threshold adaptation confidence
        # Would need to access adaptive_thresholds here in real implementation
        threshold_confidence = 0.8  # Placeholder
        
        # Combined confidence
        overall_confidence = (
            prediction_confidence * 0.3 +
            processing_confidence * 0.3 +
            analysis_confidence * 0.2 +
            threshold_confidence * 0.2
        )
        
        return max(0.5, min(0.95, overall_confidence))  # Reasonable confidence range
    
    def _estimate_processing_time(self, config: Dict[str, Any]) -> float:
        """Estimate processing time in seconds"""
        
        base_time = 1.0  # 1 second base processing
        
        # Duration factor
        duration_seconds = config['crossfade_duration'] / self.sample_rate
        base_time += duration_seconds * 0.5
        
        # Processing complexity factor
        params = config['processing_params']
        if params.pitch_adjustment != 0:
            base_time += abs(params.pitch_adjustment) * 0.5
        if params.tempo_adjustment != 1.0:
            base_time += abs(params.tempo_adjustment - 1.0) * 2.0
        
        # Envelope complexity
        base_time += 0.5  # Envelope generation
        
        return base_time
    
    def _generate_configuration_summary(self, 
                                      config: Dict[str, Any],
                                      analysis_results: Dict[str, Any],
                                      objective: OptimizationObjective,
                                      strategy: ConfigurationStrategy) -> str:
        """Generate human-readable configuration summary"""
        
        summary = f"=== Optimized Crossfade Configuration ===\n"
        summary += f"Objective: {objective.value.replace('_', ' ').title()}\n"
        summary += f"Strategy: {strategy.value.replace('_', ' ').title()}\n\n"
        
        # Timing information
        start_a_sec = config['crossfade_start_a'] / self.sample_rate
        start_b_sec = config['crossfade_start_b'] / self.sample_rate  
        duration_sec = config['crossfade_duration'] / self.sample_rate
        summary += f"Timing: A={start_a_sec:.2f}s → B={start_b_sec:.2f}s, Duration={duration_sec:.2f}s\n"
        
        # Processing parameters
        params = config['processing_params']
        summary += f"Processing: Pitch {params.pitch_adjustment:+.1f} semitones, "
        summary += f"Tempo {(params.tempo_adjustment-1)*100:+.1f}%, "
        summary += f"Gain {params.gain_adjustment:+.1f} dB\n"
        
        # Quality prediction
        prediction = config['smoothness_prediction']
        summary += f"Quality: {prediction.overall_quality.value.title()} "
        summary += f"(Score: {prediction.smoothness_score:.3f}, Confidence: {prediction.confidence:.3f})\n"
        
        # Key issues and solutions
        if analysis_results['low_end_conflict'].severity != ConflictSeverity.NONE:
            summary += f"Low-end: {analysis_results['low_end_conflict'].conflict_type.value} "
            summary += f"({analysis_results['low_end_conflict'].severity.value})\n"
        
        if analysis_results['spectral_match'].strategy != 'NO_EQ_NEEDED':  # Simplified check
            summary += f"Spectral: {analysis_results['spectral_match'].strategy} strategy applied\n"
        
        if analysis_results['fallback_plan']:
            summary += f"Fallback: {analysis_results['fallback_plan'].strategy.value} available\n"
        
        return summary
    
    def get_detailed_report(self, optimal_config: OptimalConfiguration) -> str:
        """Generate detailed optimization report"""
        
        report = optimal_config.configuration_summary + "\n"
        
        report += f"\n=== Detailed Analysis ===\n"
        report += f"Overall Quality Score: {optimal_config.overall_quality_score:.3f}\n"
        report += f"Optimization Confidence: {optimal_config.optimization_confidence:.3f}\n"
        report += f"Estimated Processing Time: {optimal_config.estimated_processing_time:.1f}s\n\n"
        
        # Smoothness breakdown
        report += "Smoothness Factor Breakdown:\n"
        for factor, score in optimal_config.smoothness_prediction.factor_scores.items():
            factor_name = str(factor).replace('SmoothnessFactor.', '').replace('_', ' ').title()
            report += f"  {factor_name}: {score:.3f}\n"
        
        # Expected artifacts
        if optimal_config.smoothness_prediction.predicted_artifacts:
            report += f"\nExpected Artifacts:\n"
            for artifact in optimal_config.smoothness_prediction.predicted_artifacts:
                report += f"  - {artifact}\n"
        
        # Improvement suggestions
        if optimal_config.smoothness_prediction.improvement_suggestions:
            report += f"\nImprovement Suggestions:\n"
            for suggestion in optimal_config.smoothness_prediction.improvement_suggestions:
                report += f"  - {suggestion}\n"
        
        # Alternative configurations
        if optimal_config.alternative_configurations:
            report += f"\n=== Alternative Configurations ===\n"
            for i, alt in enumerate(optimal_config.alternative_configurations, 1):
                report += f"{i}. {alt['name']} (Score: {alt['quality_score']:.3f})\n"
                report += f"   {alt['description']}\n"
                for diff in alt['key_differences']:
                    report += f"   - {diff}\n"
                report += "\n"
        
        return report


if __name__ == "__main__":
    # Test the complete configuration optimizer
    from .beat_grid_extractor import BeatGridExtractor, create_test_audio
    from .energy_profile_extractor import EnergyProfileExtractor
    from .musical_key_extractor import MusicalKeyExtractor
    from .processing_parameter_calculator import ProcessingParameterCalculator
    from .crossfade_envelope_designer import CrossfadeEnvelopeDesigner
    
    # Create test audio
    audio_a = create_test_audio(15.0, 120.0)
    audio_b = create_test_audio(15.0, 125.0)  # Slight tempo difference
    
    # Extract all features
    beat_extractor = BeatGridExtractor()
    energy_extractor = EnergyProfileExtractor()
    key_extractor = MusicalKeyExtractor()
    param_calculator = ProcessingParameterCalculator()
    envelope_designer = CrossfadeEnvelopeDesigner()
    
    beat_grid_a = beat_extractor(audio_a)
    beat_grid_b = beat_extractor(audio_b)
    energy_a = energy_extractor(audio_a)
    energy_b = energy_extractor(audio_b)
    key_a = key_extractor(audio_a)
    key_b = key_extractor(audio_b)
    
    # Initial processing parameters and envelope
    initial_params = param_calculator(beat_grid_a, beat_grid_b, key_a, key_b, energy_a, energy_b)
    initial_envelope = envelope_designer(
        beat_grid_a, beat_grid_b, energy_a, energy_b,
        10*44100, 5*44100, 3*44100
    )
    
    # Test configuration optimizer
    optimizer = ConfigurationOptimizer()
    
    print("=== Configuration Optimization Test ===\n")
    
    # Test different objectives
    objectives = [
        OptimizationObjective.MAXIMIZE_QUALITY,
        OptimizationObjective.BALANCE_QUALITY_SPEED,
        OptimizationObjective.MINIMIZE_ARTIFACTS
    ]
    
    for objective in objectives:
        print(f"Testing objective: {objective.value.replace('_', ' ').title()}")
        
        optimal_config = optimizer(
            audio_a, audio_b, beat_grid_a, beat_grid_b,
            energy_a, energy_b, key_a, key_b,
            initial_params, initial_envelope,
            10*44100, 5*44100,  # Initial crossfade positions
            objective, ConfigurationStrategy.BALANCED_PERFORMANCE
        )
        
        print(f"Quality Score: {optimal_config.overall_quality_score:.3f}")
        print(f"Confidence: {optimal_config.optimization_confidence:.3f}")
        print(f"Processing Time: {optimal_config.estimated_processing_time:.1f}s")
        print(f"Alternatives: {len(optimal_config.alternative_configurations)}")
        print()
    
    # Detailed report for one configuration
    detailed_config = optimizer(
        audio_a, audio_b, beat_grid_a, beat_grid_b,
        energy_a, energy_b, key_a, key_b,
        initial_params, initial_envelope,
        10*44100, 5*44100,
        OptimizationObjective.MAXIMIZE_QUALITY,
        ConfigurationStrategy.AUTO_ADAPTIVE
    )
    
    print("=== Detailed Configuration Report ===")
    print(optimizer.get_detailed_report(detailed_config))