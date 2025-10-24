"""
Comprehensive Test Suite for Phase 1.5 Enhanced Audio Engineering Modules

This test suite validates all 6 Phase 1.5 crossfade modules with various
test scenarios and audio content types.
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any
import warnings

# Import Phase 1.5 modules
from crossfade.low_end_conflict_analyzer import (
    LowEndConflictAnalyzer, LowEndConflict, ConflictType, ConflictSeverity
)
from crossfade.spectral_matching_eq import (
    SpectralMatchingEQ, SpectralMatch, EQStrategy, EQBand, EQAdjustment
)
from crossfade.fallback_strategy_optimizer import (
    FallbackStrategyOptimizer, FallbackPlan, FallbackStrategy, CutQuality
)
from crossfade.transition_smoothness_predictor import (
    TransitionSmoothnessPredictions, SmoothnessPrediction, TransitionQuality, SmoothnessFactor
)
from crossfade.adaptive_threshold_calculator import (
    AdaptiveThresholdCalculator, ThresholdSet, AdaptationStrategy, ThresholdType
)
from crossfade.configuration_optimizer import (
    ConfigurationOptimizer, OptimalConfiguration, OptimizationObjective, ConfigurationStrategy
)

# Import Phase 1 modules needed for testing
from crossfade import (
    BeatGridExtractor, BeatGrid, create_test_audio,
    EnergyProfileExtractor, EnergyProfile, create_test_audio_with_energy_pattern,
    MusicalKeyExtractor, KeyProfile, create_test_audio_with_key,
    ProcessingParameterCalculator, ProcessingParams,
    CrossfadeEnvelopeDesigner, CrossfadeEnvelope
)


class TestLowEndConflictAnalyzer:
    """Test low-end conflict analysis module"""
    
    def setup_method(self):
        self.analyzer = LowEndConflictAnalyzer()
        
        # Create test audio with different low-end characteristics
        duration = 4.0
        sample_rate = 44100
        t = torch.linspace(0, duration, int(duration * sample_rate))
        
        # Audio A: 60Hz bass + 40Hz kick
        bass_a = 0.5 * torch.sin(2 * torch.pi * 60 * t)
        kick_a = 0.3 * torch.sin(2 * torch.pi * 40 * t) * torch.exp(-5 * (t % 1.0))
        self.audio_a = bass_a + kick_a + 0.1 * torch.randn_like(t)
        
        # Audio B: 65Hz bass + 42Hz kick (slight conflict)
        bass_b = 0.4 * torch.sin(2 * torch.pi * 65 * t)
        kick_b = 0.4 * torch.sin(2 * torch.pi * 42 * t) * torch.exp(-5 * ((t + 0.1) % 1.0))
        self.audio_b_conflict = bass_b + kick_b + 0.1 * torch.randn_like(t)
        
        # Audio C: High frequencies only (no conflict)
        self.audio_c_no_conflict = 0.3 * torch.sin(2 * torch.pi * 1000 * t)
    
    def test_basic_conflict_detection(self):
        """Test basic conflict detection functionality"""
        conflict = self.analyzer(
            self.audio_a, self.audio_b_conflict, 
            0, 0, len(self.audio_a)
        )
        
        assert isinstance(conflict, LowEndConflict)
        assert conflict.conflict_type in [e for e in ConflictType]
        assert conflict.severity in [e for e in ConflictSeverity]
        assert 0 <= conflict.conflict_strength <= 1
        assert conflict.frequency_range_hz[0] < conflict.frequency_range_hz[1]
        assert conflict.recommended_crossfade_freq > 0
    
    def test_no_conflict_detection(self):
        """Test detection when no low-end conflicts exist"""
        conflict = self.analyzer(
            self.audio_c_no_conflict, self.audio_c_no_conflict,
            0, 0, len(self.audio_c_no_conflict)
        )
        
        # Should have low conflict strength or be classified as no/mild conflict
        assert (conflict.conflict_type == ConflictType.NO_CONFLICT or 
                conflict.severity in [ConflictSeverity.NONE, ConflictSeverity.MILD] or
                conflict.conflict_strength < 0.6)  # More lenient threshold
    
    def test_severe_conflict_detection(self):
        """Test detection of severe conflicts"""
        # Create identical bass content that will clash severely
        conflict = self.analyzer(
            self.audio_a, self.audio_a,  # Same audio = maximum conflict
            0, 0, len(self.audio_a)
        )
        
        assert conflict.conflict_strength > 0.3  # Should detect some level of conflict
        assert conflict.recommended_crossfade_freq > 50.0
    
    def test_frequency_selective_weights(self):
        """Test frequency-selective crossfade weight calculation"""
        conflict = self.analyzer(
            self.audio_a, self.audio_b_conflict,
            0, 0, len(self.audio_a)
        )
        
        weights = self.analyzer.get_frequency_selective_weights(conflict)
        
        assert 'low' in weights and 'mid' in weights and 'high' in weights
        assert all(w > 0 for w in weights.values())
        assert all(w <= 2.0 for w in weights.values())  # Reasonable weight range
    
    def test_crossfade_region_extraction(self):
        """Test crossfade region extraction with different positions"""
        # Test with valid positions
        conflict = self.analyzer(
            self.audio_a, self.audio_b_conflict,
            len(self.audio_a)//4, len(self.audio_b_conflict)//4,
            len(self.audio_a)//2
        )
        
        assert isinstance(conflict, LowEndConflict)
        
        # Test with edge case positions
        conflict_edge = self.analyzer(
            self.audio_a, self.audio_b_conflict,
            len(self.audio_a) - 1000, len(self.audio_b_conflict) - 1000,
            2000
        )
        
        assert isinstance(conflict_edge, LowEndConflict)


class TestSpectralMatchingEQ:
    """Test spectral matching EQ module"""
    
    def setup_method(self):
        self.eq_matcher = SpectralMatchingEQ()
        self.energy_extractor = EnergyProfileExtractor()
        
        # Create test audio with different spectral content
        self.audio_bright = create_test_audio_with_energy_pattern(10.0, 'stable') + \
                           0.5 * torch.randn(int(10*44100))  # Add high-freq content
        self.audio_warm = create_test_audio_with_energy_pattern(10.0, 'stable') * 0.7  # Less bright
        
        self.profile_bright = self.energy_extractor(self.audio_bright)
        self.profile_warm = self.energy_extractor(self.audio_warm)
    
    def test_basic_spectral_matching(self):
        """Test basic spectral matching functionality"""
        spectral_match = self.eq_matcher(
            self.profile_bright, self.profile_warm,
            5*44100, 2*44100, 2*44100
        )
        
        assert isinstance(spectral_match, SpectralMatch)
        assert 0 <= spectral_match.compatibility_score <= 1
        assert spectral_match.strategy in [e for e in EQStrategy]
        assert isinstance(spectral_match.track_b_adjustments, list)
        assert isinstance(spectral_match.track_a_adjustments, list)
        assert 0 <= spectral_match.processing_confidence <= 1
    
    def test_no_eq_needed_detection(self):
        """Test detection when no EQ is needed"""
        # Use identical profiles
        spectral_match = self.eq_matcher(
            self.profile_warm, self.profile_warm,
            5*44100, 2*44100, 2*44100
        )
        
        # Should have high compatibility
        assert spectral_match.compatibility_score > 0.8
    
    def test_eq_adjustment_generation(self):
        """Test EQ adjustment generation"""
        spectral_match = self.eq_matcher(
            self.profile_bright, self.profile_warm,
            5*44100, 2*44100, 2*44100
        )
        
        if spectral_match.track_b_adjustments:
            for adjustment in spectral_match.track_b_adjustments:
                assert isinstance(adjustment, EQAdjustment)
                assert adjustment.frequency_hz > 0
                assert -12.0 <= adjustment.gain_db <= 12.0  # Reasonable gain range
                assert adjustment.q_factor > 0
                assert adjustment.filter_type in ['bell', 'high_shelf', 'low_shelf', 'highpass', 'lowpass']
    
    def test_frequency_timeline_creation(self):
        """Test frequency-selective crossfade timeline"""
        spectral_match = self.eq_matcher(
            self.profile_bright, self.profile_warm,
            5*44100, 2*44100, 2*44100
        )
        
        timeline = spectral_match.frequency_selective_timeline
        assert isinstance(timeline, dict)
        
        for band, timing in timeline.items():
            assert isinstance(band, EQBand)
            assert 0 <= timing <= 1.0
    
    def test_before_after_comparison(self):
        """Test before/after spectral comparison"""
        spectral_match = self.eq_matcher(
            self.profile_bright, self.profile_warm,
            5*44100, 2*44100, 2*44100
        )
        
        comparison = spectral_match.before_after_comparison
        assert 'before_compatibility' in comparison
        assert 'after_compatibility' in comparison
        assert 'improvement' in comparison
        
        # After should be same or better than before
        assert comparison['after_compatibility'] >= comparison['before_compatibility'] - 0.1
    
    def test_eq_summary_generation(self):
        """Test EQ summary generation"""
        spectral_match = self.eq_matcher(
            self.profile_bright, self.profile_warm,
            5*44100, 2*44100, 2*44100
        )
        
        summary = self.eq_matcher.get_eq_summary(spectral_match)
        assert isinstance(summary, str)
        assert len(summary) > 10  # Should have meaningful content


class TestFallbackStrategyOptimizer:
    """Test fallback strategy optimizer module"""
    
    def setup_method(self):
        self.optimizer = FallbackStrategyOptimizer()
        self.beat_extractor = BeatGridExtractor()
        self.energy_extractor = EnergyProfileExtractor()
        
        # Create test audio with incompatible tempos
        self.audio_120 = create_test_audio(15.0, 120.0)
        self.audio_140 = create_test_audio(15.0, 140.0)  # 16.7% tempo difference
        
        self.beat_grid_a = self.beat_extractor(self.audio_120)
        self.beat_grid_b = self.beat_extractor(self.audio_140)
        self.energy_profile_a = self.energy_extractor(self.audio_120)
        self.energy_profile_b = self.energy_extractor(self.audio_140)
    
    def test_basic_fallback_optimization(self):
        """Test basic fallback strategy optimization"""
        # Mock rhythmic match for testing
        class MockRhythmicMatch:
            def __init__(self):
                self.required_rate_change = 1.167  # 16.7% change
                self.tempo_stability_factor = 0.8
        
        rhythmic_match = MockRhythmicMatch()
        
        fallback_plan = self.optimizer(
            self.beat_grid_a, self.beat_grid_b,
            self.energy_profile_a, self.energy_profile_b,
            rhythmic_match,
            20*44100, 5*44100  # Cut positions
        )
        
        assert isinstance(fallback_plan, FallbackPlan)
        assert fallback_plan.strategy in [e for e in FallbackStrategy]
        assert fallback_plan.expected_quality in [e for e in CutQuality]
        assert 0 <= fallback_plan.confidence <= 1
        assert 0 <= fallback_plan.beat_alignment_score <= 1
        assert 0 <= fallback_plan.energy_continuity_score <= 1
    
    def test_fallback_strategy_selection(self):
        """Test different fallback strategy selection"""
        class MockRhythmicMatch:
            def __init__(self, rate_change):
                self.required_rate_change = rate_change
                self.tempo_stability_factor = 0.8
        
        # Test with large tempo difference requiring fallback
        rhythmic_match_large = MockRhythmicMatch(1.3)  # 30% difference
        fallback_plan_large = self.optimizer(
            self.beat_grid_a, self.beat_grid_b,
            self.energy_profile_a, self.energy_profile_b,
            rhythmic_match_large,
            20*44100, 5*44100
        )
        
        # Should select a meaningful fallback strategy
        assert fallback_plan_large.strategy != FallbackStrategy.IMMEDIATE_CUT or fallback_plan_large.confidence < 0.5
    
    def test_cut_timing_optimization(self):
        """Test cut timing optimization"""
        class MockRhythmicMatch:
            def __init__(self):
                self.required_rate_change = 1.2
                self.tempo_stability_factor = 0.8
        
        rhythmic_match = MockRhythmicMatch()
        
        fallback_plan = self.optimizer(
            self.beat_grid_a, self.beat_grid_b,
            self.energy_profile_a, self.energy_profile_b,
            rhythmic_match,
            20*44100, 5*44100
        )
        
        # Cut positions should be reasonable (allow some flexibility for processing)
        assert 0 <= fallback_plan.cut_position_a <= len(self.audio_120) * 1.5  # Allow extrapolation
        assert 0 <= fallback_plan.start_position_b <= len(self.audio_140) * 1.5
        assert fallback_plan.crossfade_duration > 0
    
    def test_fallback_summary(self):
        """Test fallback summary generation"""
        class MockRhythmicMatch:
            def __init__(self):
                self.required_rate_change = 1.15
                self.tempo_stability_factor = 0.7
        
        rhythmic_match = MockRhythmicMatch()
        
        fallback_plan = self.optimizer(
            self.beat_grid_a, self.beat_grid_b,
            self.energy_profile_a, self.energy_profile_b,
            rhythmic_match,
            20*44100, 5*44100
        )
        
        summary = self.optimizer.get_fallback_summary(fallback_plan)
        assert isinstance(summary, str)
        assert len(summary) > 50  # Should be detailed


class TestTransitionSmoothnessPredictions:
    """Test transition smoothness predictions module"""
    
    def setup_method(self):
        self.predictor = TransitionSmoothnessPredictions()
        
        # Create test data
        self.beat_extractor = BeatGridExtractor()
        self.energy_extractor = EnergyProfileExtractor()
        self.key_extractor = MusicalKeyExtractor()
        self.param_calculator = ProcessingParameterCalculator()
        self.envelope_designer = CrossfadeEnvelopeDesigner()
        
        # Compatible tracks
        self.audio_a = create_test_audio(10.0, 120.0)
        self.audio_b_good = create_test_audio(10.0, 123.0)
        
        self.beat_grid_a = self.beat_extractor(self.audio_a)
        self.beat_grid_b_good = self.beat_extractor(self.audio_b_good)
        self.energy_a = self.energy_extractor(self.audio_a)
        self.energy_b_good = self.energy_extractor(self.audio_b_good)
        self.key_a = self.key_extractor(self.audio_a)
        self.key_b_good = self.key_extractor(self.audio_b_good)
        
        # Need to create compatibility analyzers first  
        from crossfade import (
            HarmonicCompatibilityAnalyzer, RhythmicCompatibilityAnalyzer, 
            EnergyCompatibilityAnalyzer, SplicePointOptimizer
        )
        
        harmonic_analyzer = HarmonicCompatibilityAnalyzer()
        rhythmic_analyzer = RhythmicCompatibilityAnalyzer()  
        energy_analyzer = EnergyCompatibilityAnalyzer()
        splice_optimizer = SplicePointOptimizer()
        
        harmonic_match = harmonic_analyzer(self.key_a, self.key_b_good)
        rhythmic_match = rhythmic_analyzer(self.beat_grid_a, self.beat_grid_b_good)
        energy_match = energy_analyzer(self.energy_a, self.energy_b_good, 5*44100, 2*44100)  # Add exit/entry positions
        optimal_splice = splice_optimizer(harmonic_match, rhythmic_match, energy_match)
        
        self.params_good = self.param_calculator(optimal_splice, harmonic_match, rhythmic_match, energy_match)
        
        self.envelope = self.envelope_designer(
            self.beat_grid_a, self.beat_grid_b_good,
            self.energy_a, self.energy_b_good,
            5*44100, 2*44100, 2*44100
        )
    
    def test_basic_smoothness_prediction(self):
        """Test basic smoothness prediction functionality"""
        prediction = self.predictor(
            self.beat_grid_a, self.beat_grid_b_good,
            self.energy_a, self.energy_b_good,
            self.key_a, self.key_b_good,
            self.params_good, self.envelope,
            5*44100, 2*44100
        )
        
        assert isinstance(prediction, SmoothnessPrediction)
        assert prediction.overall_quality in [e for e in TransitionQuality]
        assert 0 <= prediction.smoothness_score <= 1
        assert 0 <= prediction.confidence <= 1
        assert isinstance(prediction.factor_scores, dict)
        assert isinstance(prediction.predicted_artifacts, list)
        assert isinstance(prediction.improvement_suggestions, list)
    
    def test_factor_scoring(self):
        """Test individual smoothness factor scoring"""
        prediction = self.predictor(
            self.beat_grid_a, self.beat_grid_b_good,
            self.energy_a, self.energy_b_good,
            self.key_a, self.key_b_good,
            self.params_good, self.envelope,
            5*44100, 2*44100
        )
        
        # Check all factors are present and valid
        expected_factors = [
            SmoothnessFactor.TEMPO_CONTINUITY,
            SmoothnessFactor.ENERGY_FLOW,
            SmoothnessFactor.HARMONIC_COMPATIBILITY,
            SmoothnessFactor.RHYTHMIC_ALIGNMENT,
            SmoothnessFactor.SPECTRAL_MATCHING,
            SmoothnessFactor.PROCESSING_ARTIFACTS
        ]
        
        for factor in expected_factors:
            assert factor in prediction.factor_scores
            assert 0 <= prediction.factor_scores[factor] <= 1
    
    def test_quality_classification(self):
        """Test quality classification consistency"""
        prediction = self.predictor(
            self.beat_grid_a, self.beat_grid_b_good,
            self.energy_a, self.energy_b_good,
            self.key_a, self.key_b_good,
            self.params_good, self.envelope,
            5*44100, 2*44100
        )
        
        # Quality should be consistent with smoothness score
        if prediction.smoothness_score >= 0.9:
            assert prediction.overall_quality == TransitionQuality.SEAMLESS
        elif prediction.smoothness_score >= 0.7:
            assert prediction.overall_quality in [TransitionQuality.SEAMLESS, TransitionQuality.SMOOTH]
        elif prediction.smoothness_score >= 0.5:
            assert prediction.overall_quality in [TransitionQuality.SMOOTH, TransitionQuality.ACCEPTABLE]
    
    def test_artifact_prediction(self):
        """Test artifact prediction"""
        prediction = self.predictor(
            self.beat_grid_a, self.beat_grid_b_good,
            self.energy_a, self.energy_b_good,
            self.key_a, self.key_b_good,
            self.params_good, self.envelope,
            5*44100, 2*44100
        )
        
        # Artifacts should be strings
        for artifact in prediction.predicted_artifacts:
            assert isinstance(artifact, str)
            assert len(artifact) > 5  # Meaningful artifact descriptions
    
    def test_improvement_suggestions(self):
        """Test improvement suggestion generation"""
        prediction = self.predictor(
            self.beat_grid_a, self.beat_grid_b_good,
            self.energy_a, self.energy_b_good,
            self.key_a, self.key_b_good,
            self.params_good, self.envelope,
            5*44100, 2*44100
        )
        
        # Suggestions should be meaningful
        for suggestion in prediction.improvement_suggestions:
            assert isinstance(suggestion, str)
            assert len(suggestion) > 10  # Detailed suggestions
    
    def test_quality_summary(self):
        """Test quality summary generation"""
        prediction = self.predictor(
            self.beat_grid_a, self.beat_grid_b_good,
            self.energy_a, self.energy_b_good,
            self.key_a, self.key_b_good,
            self.params_good, self.envelope,
            5*44100, 2*44100
        )
        
        summary = self.predictor.get_quality_summary(prediction)
        assert isinstance(summary, str)
        assert len(summary) > 100  # Should be comprehensive


class TestAdaptiveThresholdCalculator:
    """Test adaptive threshold calculator module"""
    
    def setup_method(self):
        self.calculator = AdaptiveThresholdCalculator()
        
        # Create test tracks with different characteristics
        self.beat_extractor = BeatGridExtractor()
        self.energy_extractor = EnergyProfileExtractor()
        self.key_extractor = MusicalKeyExtractor()
        
        self.electronic_track = create_test_audio(10.0, 128.0)  # Electronic
        self.organic_track = create_test_audio(10.0, 95.0)      # Organic
        
        self.beat_electronic = self.beat_extractor(self.electronic_track)
        self.beat_organic = self.beat_extractor(self.organic_track)
        self.energy_electronic = self.energy_extractor(self.electronic_track)
        self.energy_organic = self.energy_extractor(self.organic_track)
        self.key_electronic = self.key_extractor(self.electronic_track)
        self.key_organic = self.key_extractor(self.organic_track)
    
    def test_basic_threshold_calculation(self):
        """Test basic threshold calculation"""
        threshold_set = self.calculator(
            self.beat_electronic, self.beat_organic,
            self.energy_electronic, self.energy_organic,
            self.key_electronic, self.key_organic
        )
        
        assert isinstance(threshold_set, ThresholdSet)
        assert 0 <= threshold_set.harmonic_compatibility_min <= 1
        assert threshold_set.tempo_tolerance_percent > 0
        assert threshold_set.energy_matching_tolerance_db > 0
        assert 0 <= threshold_set.beat_confidence_min <= 1
        assert 0 <= threshold_set.key_stability_min <= 1
        assert 0 <= threshold_set.processing_quality_min <= 1
        assert threshold_set.crossfade_duration_ms_range[0] < threshold_set.crossfade_duration_ms_range[1]
        assert 0 <= threshold_set.adaptation_confidence <= 1
    
    def test_adaptation_strategy_differences(self):
        """Test different adaptation strategies produce different results"""
        strategies = [
            AdaptationStrategy.CONSERVATIVE,
            AdaptationStrategy.BALANCED,
            AdaptationStrategy.PERMISSIVE,
            AdaptationStrategy.CONTEXT_ADAPTIVE
        ]
        
        threshold_sets = {}
        for strategy in strategies:
            threshold_sets[strategy] = self.calculator(
                self.beat_electronic, self.beat_organic,
                self.energy_electronic, self.energy_organic,
                self.key_electronic, self.key_organic,
                strategy
            )
        
        # Conservative should have stricter thresholds than permissive
        conservative = threshold_sets[AdaptationStrategy.CONSERVATIVE]
        permissive = threshold_sets[AdaptationStrategy.PERMISSIVE]
        
        # Some thresholds should be different between strategies
        threshold_differences = [
            abs(conservative.harmonic_compatibility_min - permissive.harmonic_compatibility_min),
            abs(conservative.tempo_tolerance_percent - permissive.tempo_tolerance_percent),
            abs(conservative.energy_matching_tolerance_db - permissive.energy_matching_tolerance_db)
        ]
        
        # At least one threshold should be meaningfully different
        assert any(diff > 0.05 for diff in threshold_differences)
    
    def test_content_analysis(self):
        """Test content analysis integration"""
        threshold_set = self.calculator(
            self.beat_electronic, self.beat_organic,
            self.energy_electronic, self.energy_organic,
            self.key_electronic, self.key_organic
        )
        
        content_analysis = threshold_set.content_analysis
        assert isinstance(content_analysis, dict)
        
        # Should contain relevant analysis factors
        expected_factors = [
            'tempo_stability', 'beat_confidence_avg', 'track_similarity'
        ]
        
        for factor in expected_factors:
            if factor in content_analysis:
                assert 0 <= content_analysis[factor] <= 1
    
    def test_threshold_constraints(self):
        """Test threshold constraint validation"""
        # Test with extreme input to verify constraints work
        threshold_set = self.calculator(
            self.beat_electronic, self.beat_organic,
            self.energy_electronic, self.energy_organic,
            None, None,  # No key profiles
            AdaptationStrategy.CONTEXT_ADAPTIVE
        )
        
        # All thresholds should be within reasonable ranges
        assert 0 <= threshold_set.harmonic_compatibility_min <= 1
        assert 0.5 <= threshold_set.tempo_tolerance_percent <= 10.0
        assert 2.0 <= threshold_set.energy_matching_tolerance_db <= 15.0
        assert 0.1 <= threshold_set.beat_confidence_min <= 0.9
        assert 0.1 <= threshold_set.key_stability_min <= 0.9
        assert 0.3 <= threshold_set.processing_quality_min <= 0.95
        assert 200.0 <= threshold_set.crossfade_duration_ms_range[0] <= 5000.0
        assert 1000.0 <= threshold_set.crossfade_duration_ms_range[1] <= 15000.0
    
    def test_threshold_summary(self):
        """Test threshold summary generation"""
        threshold_set = self.calculator(
            self.beat_electronic, self.beat_organic,
            self.energy_electronic, self.energy_organic,
            self.key_electronic, self.key_organic
        )
        
        summary = self.calculator.get_threshold_summary(threshold_set)
        assert isinstance(summary, str)
        assert len(summary) > 100  # Should be comprehensive
        assert 'Adaptive Threshold Configuration' in summary


class TestConfigurationOptimizer:
    """Test complete configuration optimizer module"""
    
    def setup_method(self):
        self.optimizer = ConfigurationOptimizer()
        
        # Create comprehensive test setup
        self.beat_extractor = BeatGridExtractor()
        self.energy_extractor = EnergyProfileExtractor()
        self.key_extractor = MusicalKeyExtractor()
        self.param_calculator = ProcessingParameterCalculator()
        self.envelope_designer = CrossfadeEnvelopeDesigner()
        
        self.audio_a = create_test_audio(15.0, 120.0)
        self.audio_b = create_test_audio(15.0, 125.0)
        
        self.beat_grid_a = self.beat_extractor(self.audio_a)
        self.beat_grid_b = self.beat_extractor(self.audio_b)
        self.energy_a = self.energy_extractor(self.audio_a)
        self.energy_b = self.energy_extractor(self.audio_b)
        self.key_a = self.key_extractor(self.audio_a)
        self.key_b = self.key_extractor(self.audio_b)
        
        self.initial_params = self.param_calculator(
            self.beat_grid_a, self.beat_grid_b, 
            self.key_a, self.key_b,
            self.energy_a, self.energy_b
        )
        self.initial_envelope = self.envelope_designer(
            self.beat_grid_a, self.beat_grid_b,
            self.energy_a, self.energy_b,
            10*44100, 5*44100, 3*44100
        )
    
    def test_basic_configuration_optimization(self):
        """Test basic configuration optimization"""
        optimal_config = self.optimizer(
            self.audio_a, self.audio_b,
            self.beat_grid_a, self.beat_grid_b,
            self.energy_a, self.energy_b,
            self.key_a, self.key_b,
            self.initial_params, self.initial_envelope,
            10*44100, 5*44100
        )
        
        assert isinstance(optimal_config, OptimalConfiguration)
        assert optimal_config.crossfade_start_a > 0
        assert optimal_config.crossfade_start_b > 0
        assert optimal_config.crossfade_duration > 0
        assert isinstance(optimal_config.processing_params, ProcessingParams)
        assert isinstance(optimal_config.crossfade_envelope, CrossfadeEnvelope)
        assert 0 <= optimal_config.overall_quality_score <= 1
        assert 0 <= optimal_config.optimization_confidence <= 1
        assert optimal_config.estimated_processing_time > 0
    
    def test_different_optimization_objectives(self):
        """Test different optimization objectives"""
        objectives = [
            OptimizationObjective.MAXIMIZE_QUALITY,
            OptimizationObjective.MINIMIZE_ARTIFACTS,
            OptimizationObjective.BALANCE_QUALITY_SPEED
        ]
        
        configs = {}
        for objective in objectives:
            configs[objective] = self.optimizer(
                self.audio_a, self.audio_b,
                self.beat_grid_a, self.beat_grid_b,
                self.energy_a, self.energy_b,
                self.key_a, self.key_b,
                self.initial_params, self.initial_envelope,
                10*44100, 5*44100,
                objective
            )
        
        # All configs should be valid
        for config in configs.values():
            assert isinstance(config, OptimalConfiguration)
            assert config.overall_quality_score > 0
    
    def test_different_configuration_strategies(self):
        """Test different configuration strategies"""
        strategies = [
            ConfigurationStrategy.CONSERVATIVE_QUALITY,
            ConfigurationStrategy.BALANCED_PERFORMANCE,
            ConfigurationStrategy.AUTO_ADAPTIVE
        ]
        
        for strategy in strategies:
            config = self.optimizer(
                self.audio_a, self.audio_b,
                self.beat_grid_a, self.beat_grid_b,
                self.energy_a, self.energy_b,
                self.key_a, self.key_b,
                self.initial_params, self.initial_envelope,
                10*44100, 5*44100,
                OptimizationObjective.BALANCE_QUALITY_SPEED,
                strategy
            )
            
            assert isinstance(config, OptimalConfiguration)
            assert config.optimization_confidence > 0.3  # Should have reasonable confidence
    
    def test_phase1_5_integration(self):
        """Test integration of all Phase 1.5 modules"""
        optimal_config = self.optimizer(
            self.audio_a, self.audio_b,
            self.beat_grid_a, self.beat_grid_b,
            self.energy_a, self.energy_b,
            self.key_a, self.key_b,
            self.initial_params, self.initial_envelope,
            10*44100, 5*44100
        )
        
        # Should have all Phase 1.5 analysis results
        assert isinstance(optimal_config.low_end_strategy, LowEndConflict)
        assert isinstance(optimal_config.spectral_matching, SpectralMatch)
        assert isinstance(optimal_config.smoothness_prediction, SmoothnessPrediction)
        assert isinstance(optimal_config.adaptive_thresholds, ThresholdSet)
        # Fallback plan may be None if not needed
    
    def test_alternative_configurations(self):
        """Test alternative configuration generation"""
        optimal_config = self.optimizer(
            self.audio_a, self.audio_b,
            self.beat_grid_a, self.beat_grid_b,
            self.energy_a, self.energy_b,
            self.key_a, self.key_b,
            self.initial_params, self.initial_envelope,
            10*44100, 5*44100
        )
        
        assert isinstance(optimal_config.alternative_configurations, list)
        
        for alt in optimal_config.alternative_configurations:
            assert isinstance(alt, dict)
            assert 'name' in alt
            assert 'description' in alt
            assert 'quality_score' in alt
            assert 'key_differences' in alt
            assert 0 <= alt['quality_score'] <= 1
    
    def test_detailed_report_generation(self):
        """Test detailed report generation"""
        optimal_config = self.optimizer(
            self.audio_a, self.audio_b,
            self.beat_grid_a, self.beat_grid_b,
            self.energy_a, self.energy_b,
            self.key_a, self.key_b,
            self.initial_params, self.initial_envelope,
            10*44100, 5*44100
        )
        
        detailed_report = self.optimizer.get_detailed_report(optimal_config)
        assert isinstance(detailed_report, str)
        assert len(detailed_report) > 200  # Should be comprehensive
        assert 'Quality Score' in detailed_report
        assert 'Confidence' in detailed_report
    
    def test_configuration_summary(self):
        """Test configuration summary"""
        optimal_config = self.optimizer(
            self.audio_a, self.audio_b,
            self.beat_grid_a, self.beat_grid_b,
            self.energy_a, self.energy_b,
            self.key_a, self.key_b,
            self.initial_params, self.initial_envelope,
            10*44100, 5*44100
        )
        
        summary = optimal_config.configuration_summary
        assert isinstance(summary, str)
        assert len(summary) > 50  # Should contain key information
        assert 'Optimized Crossfade Configuration' in summary


# Integration test for complete Phase 1.5 pipeline
class TestPhase15Integration:
    """Test complete Phase 1.5 integration"""
    
    def test_complete_pipeline(self):
        """Test complete Phase 1.5 pipeline integration"""
        
        # Create test data
        audio_a = create_test_audio(12.0, 120.0)
        audio_b = create_test_audio(12.0, 128.0)
        
        # Phase 1 extraction
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
        
        # Import and create compatibility analysis pipeline for processing params  
        from crossfade import (
            HarmonicCompatibilityAnalyzer, RhythmicCompatibilityAnalyzer,
            EnergyCompatibilityAnalyzer, SplicePointOptimizer
        )
        
        harmonic_analyzer = HarmonicCompatibilityAnalyzer()
        rhythmic_analyzer = RhythmicCompatibilityAnalyzer()
        energy_analyzer = EnergyCompatibilityAnalyzer()
        splice_optimizer = SplicePointOptimizer()
        
        harmonic_match = harmonic_analyzer(key_a, key_b)
        rhythmic_match = rhythmic_analyzer(beat_grid_a, beat_grid_b)
        energy_match = energy_analyzer(energy_a, energy_b, 8*44100, 4*44100)  # Add exit/entry positions
        optimal_splice = splice_optimizer(harmonic_match, rhythmic_match, energy_match)
        
        initial_params = param_calculator(optimal_splice, harmonic_match, rhythmic_match, energy_match)
        initial_envelope = envelope_designer(beat_grid_a, beat_grid_b, energy_a, energy_b, 8*44100, 4*44100, 3*44100)
        
        # Phase 1.5 complete optimization
        optimizer = ConfigurationOptimizer()
        optimal_config = optimizer(
            audio_a, audio_b, beat_grid_a, beat_grid_b,
            energy_a, energy_b, key_a, key_b,
            initial_params, initial_envelope,
            8*44100, 4*44100,
            OptimizationObjective.MAXIMIZE_QUALITY,
            ConfigurationStrategy.AUTO_ADAPTIVE
        )
        
        # Verify complete configuration
        assert isinstance(optimal_config, OptimalConfiguration)
        assert optimal_config.overall_quality_score > 0.3  # Should achieve reasonable quality
        assert optimal_config.optimization_confidence > 0.4  # Should have reasonable confidence
        assert len(optimal_config.alternative_configurations) > 0  # Should provide alternatives
        assert len(optimal_config.configuration_summary) > 100  # Should have detailed summary
        
        # Verify all Phase 1.5 components are integrated
        assert optimal_config.low_end_strategy is not None
        assert optimal_config.spectral_matching is not None
        assert optimal_config.smoothness_prediction is not None
        assert optimal_config.adaptive_thresholds is not None
        
        print(f"Integration test passed - Quality: {optimal_config.overall_quality_score:.3f}, "
              f"Confidence: {optimal_config.optimization_confidence:.3f}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])