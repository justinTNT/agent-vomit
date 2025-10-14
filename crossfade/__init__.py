"""
Crossfade System - Phase 1 Core + Phase 1.5 Enhanced Modules

This package provides the core crossfade planning system modules
for intelligent audio crossfade analysis and optimization.
"""

# Phase 1 Core Modules
from .beat_grid_extractor import BeatGridExtractor, BeatGrid, create_test_audio
from .musical_key_extractor import MusicalKeyExtractor, KeyProfile, Mode, create_test_audio_with_key
from .energy_profile_extractor import EnergyProfileExtractor, EnergyProfile, create_test_audio_with_energy_pattern
from .exit_point_analyzer import ExitPointAnalyzer, ExitCandidates, ExitCandidate
from .entry_point_analyzer import EntryPointAnalyzer, EntryCandidates, EntryCandidate, create_test_audio_with_drop
from .harmonic_compatibility_analyzer import HarmonicCompatibilityAnalyzer, HarmonicMatch, CompatibilityLevel
from .rhythmic_compatibility_analyzer import RhythmicCompatibilityAnalyzer, RhythmicMatch, TempoCompatibility
from .energy_compatibility_analyzer import EnergyCompatibilityAnalyzer, EnergyMatch, EnergyCompatibilityLevel
from .splice_point_optimizer import SplicePointOptimizer, OptimalSplice
from .processing_parameter_calculator import ProcessingParameterCalculator, ProcessingParams, ProcessingStrategy
from .crossfade_envelope_designer import CrossfadeEnvelopeDesigner, CrossfadeEnvelope, CrossfadeCurveType

# Phase 1.5 Enhanced Audio Engineering Modules
from .low_end_conflict_analyzer import LowEndConflictAnalyzer, LowEndConflict, ConflictType, ConflictSeverity
from .spectral_matching_eq import SpectralMatchingEQ, SpectralMatch, EQStrategy, EQBand, EQAdjustment
from .fallback_strategy_optimizer import FallbackStrategyOptimizer, FallbackPlan, FallbackStrategy, CutQuality
from .transition_smoothness_predictor import TransitionSmoothnessPredictions, SmoothnessPrediction, TransitionQuality, SmoothnessFactor
from .adaptive_threshold_calculator import AdaptiveThresholdCalculator, ThresholdSet, AdaptationStrategy, ThresholdType
from .configuration_optimizer import ConfigurationOptimizer, OptimalConfiguration, OptimizationObjective, ConfigurationStrategy

__version__ = "1.0.0"
__author__ = "Crossfade AI System"

__all__ = [
    # Phase 1 Core analysis modules
    "BeatGridExtractor", "BeatGrid", "create_test_audio",
    "MusicalKeyExtractor", "KeyProfile", "Mode", "create_test_audio_with_key", 
    "EnergyProfileExtractor", "EnergyProfile", "create_test_audio_with_energy_pattern",
    
    # Splice point analysis
    "ExitPointAnalyzer", "ExitCandidates", "ExitCandidate",
    "EntryPointAnalyzer", "EntryCandidates", "EntryCandidate", "create_test_audio_with_drop",
    
    # Compatibility analysis  
    "HarmonicCompatibilityAnalyzer", "HarmonicMatch", "CompatibilityLevel",
    "RhythmicCompatibilityAnalyzer", "RhythmicMatch", "TempoCompatibility",
    "EnergyCompatibilityAnalyzer", "EnergyMatch", "EnergyCompatibilityLevel",
    
    # Optimization and output
    "SplicePointOptimizer", "OptimalSplice",
    "ProcessingParameterCalculator", "ProcessingParams", "ProcessingStrategy",
    "CrossfadeEnvelopeDesigner", "CrossfadeEnvelope", "CrossfadeCurveType",
    
    # Phase 1.5 Enhanced Audio Engineering Modules
    "LowEndConflictAnalyzer", "LowEndConflict", "ConflictType", "ConflictSeverity",
    "SpectralMatchingEQ", "SpectralMatch", "EQStrategy", "EQBand", "EQAdjustment", 
    "FallbackStrategyOptimizer", "FallbackPlan", "FallbackStrategy", "CutQuality",
    "TransitionSmoothnessPredictions", "SmoothnessPrediction", "TransitionQuality", "SmoothnessFactor",
    "AdaptiveThresholdCalculator", "ThresholdSet", "AdaptationStrategy", "ThresholdType",
    "ConfigurationOptimizer", "OptimalConfiguration", "OptimizationObjective", "ConfigurationStrategy"
]