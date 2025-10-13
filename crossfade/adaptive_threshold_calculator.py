"""
AdaptiveThresholdCalculator - Dynamic threshold adjustment for musical context

This module adapts processing thresholds based on track characteristics,
improving crossfade decisions through context-aware parameter adjustment.
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


class ThresholdType(Enum):
    """Types of adaptive thresholds"""
    HARMONIC_COMPATIBILITY = "harmonic_compatibility"
    TEMPO_TOLERANCE = "tempo_tolerance"
    ENERGY_MATCHING = "energy_matching"
    BEAT_CONFIDENCE = "beat_confidence"
    KEY_STABILITY = "key_stability"
    PROCESSING_QUALITY = "processing_quality"
    CROSSFADE_DURATION = "crossfade_duration"


class AdaptationStrategy(Enum):
    """Threshold adaptation strategies"""
    CONSERVATIVE = "conservative"      # Stricter thresholds for quality
    BALANCED = "balanced"             # Default balanced approach
    PERMISSIVE = "permissive"         # Looser thresholds for flexibility
    CONTEXT_ADAPTIVE = "context_adaptive"  # Fully adaptive based on content


@dataclass
class ThresholdSet:
    """Set of adaptive thresholds for crossfade processing"""
    harmonic_compatibility_min: float      # Minimum harmonic compatibility [0,1]
    tempo_tolerance_percent: float          # Maximum tempo difference (%)
    energy_matching_tolerance_db: float     # Energy level tolerance (dB)
    beat_confidence_min: float              # Minimum beat detection confidence [0,1]
    key_stability_min: float                # Minimum key stability requirement [0,1]
    processing_quality_min: float           # Minimum processing quality [0,1]
    crossfade_duration_ms_range: Tuple[float, float]  # Min/max crossfade duration (ms)
    adaptation_confidence: float            # Confidence in threshold adaptation [0,1]
    content_analysis: Dict[str, float]      # Content characteristics that influenced thresholds


class AdaptiveThresholdCalculator(nn.Module):
    """
    Calculate adaptive thresholds based on musical content characteristics.
    
    Adjusts processing thresholds dynamically to optimize crossfade quality
    for different music styles, energy levels, and harmonic complexity.
    """
    
    def __init__(self, 
                 base_thresholds: Optional[Dict[str, float]] = None,
                 adaptation_sensitivity: float = 0.8,
                 learning_rate: float = 0.1):
        super().__init__()
        
        self.adaptation_sensitivity = adaptation_sensitivity
        self.learning_rate = learning_rate
        
        # Default base thresholds (conservative defaults)
        self.base_thresholds = base_thresholds or {
            'harmonic_compatibility_min': 0.6,
            'tempo_tolerance_percent': 3.0,
            'energy_matching_tolerance_db': 6.0,
            'beat_confidence_min': 0.5,
            'key_stability_min': 0.4,
            'processing_quality_min': 0.7,
            'crossfade_duration_ms_min': 1000.0,
            'crossfade_duration_ms_max': 8000.0
        }
        
        # Neural networks for different types of adaptation
        self.content_analyzer = nn.Sequential(
            nn.Linear(12, 64),   # Content characteristics
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)     # Content complexity factors
        )
        
        # Threshold adjustment network
        self.threshold_adjuster = nn.Sequential(
            nn.Linear(16, 64),   # Content factors + base thresholds
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 7),    # 7 threshold adjustments
            nn.Tanh()            # Adjustment factors [-1, 1]
        )
        
        # Genre/style classifier for context-specific thresholds
        self.style_classifier = nn.Sequential(
            nn.Linear(10, 32),   # Style-indicative features
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 6)     # 6 music style categories
        )

    def forward(self, 
                beat_grid_a: BeatGrid,
                beat_grid_b: BeatGrid,
                energy_profile_a: EnergyProfile,
                energy_profile_b: EnergyProfile,
                key_profile_a: Optional[KeyProfile] = None,
                key_profile_b: Optional[KeyProfile] = None,
                adaptation_strategy: AdaptationStrategy = AdaptationStrategy.BALANCED,
                crossfade_context: Optional[Dict] = None) -> ThresholdSet:
        """
        Calculate adaptive thresholds based on track characteristics.
        
        Args:
            beat_grid_a: Track A beat analysis
            beat_grid_b: Track B beat analysis
            energy_profile_a: Track A energy analysis
            energy_profile_b: Track B energy analysis
            key_profile_a: Track A key analysis (optional)
            key_profile_b: Track B key analysis (optional)
            adaptation_strategy: Threshold adaptation approach
            crossfade_context: Additional context information
            
        Returns:
            ThresholdSet with adapted thresholds and analysis
        """
        
        # Analyze content characteristics
        content_characteristics = self._analyze_content_characteristics(
            beat_grid_a, beat_grid_b, energy_profile_a, energy_profile_b,
            key_profile_a, key_profile_b
        )
        
        # Determine music style/genre influences
        style_factors = self._analyze_style_factors(
            beat_grid_a, beat_grid_b, energy_profile_a, energy_profile_b
        )
        
        # Apply strategy-specific adjustments
        strategy_adjustments = self._apply_strategy_adjustments(
            adaptation_strategy, content_characteristics, style_factors
        )
        
        # Calculate context-specific thresholds
        adaptive_thresholds = self._calculate_adaptive_thresholds(
            content_characteristics, style_factors, strategy_adjustments, crossfade_context
        )
        
        # Validate and constrain thresholds
        final_thresholds = self._validate_and_constrain_thresholds(adaptive_thresholds)
        
        # Calculate adaptation confidence
        adaptation_confidence = self._calculate_adaptation_confidence(
            content_characteristics, style_factors
        )
        
        return ThresholdSet(
            harmonic_compatibility_min=final_thresholds['harmonic_compatibility_min'],
            tempo_tolerance_percent=final_thresholds['tempo_tolerance_percent'],
            energy_matching_tolerance_db=final_thresholds['energy_matching_tolerance_db'],
            beat_confidence_min=final_thresholds['beat_confidence_min'],
            key_stability_min=final_thresholds['key_stability_min'],
            processing_quality_min=final_thresholds['processing_quality_min'],
            crossfade_duration_ms_range=(
                final_thresholds['crossfade_duration_ms_min'],
                final_thresholds['crossfade_duration_ms_max']
            ),
            adaptation_confidence=float(adaptation_confidence),
            content_analysis=content_characteristics
        )
    
    def _analyze_content_characteristics(self, 
                                       beat_grid_a: BeatGrid,
                                       beat_grid_b: BeatGrid,
                                       energy_profile_a: EnergyProfile,
                                       energy_profile_b: EnergyProfile,
                                       key_profile_a: Optional[KeyProfile],
                                       key_profile_b: Optional[KeyProfile]) -> Dict[str, float]:
        """Analyze content characteristics that affect threshold requirements"""
        
        characteristics = {}
        
        # Tempo characteristics
        tempo_stability = (beat_grid_a.tempo_stability + beat_grid_b.tempo_stability) / 2
        tempo_difference = abs(beat_grid_a.bpm - beat_grid_b.bpm) / max(beat_grid_a.bpm, beat_grid_b.bpm)
        characteristics['tempo_stability'] = float(tempo_stability)
        characteristics['tempo_difference'] = float(tempo_difference)
        
        # Energy characteristics
        energy_variance_a = float(torch.var(energy_profile_a.rms_curve))
        energy_variance_b = float(torch.var(energy_profile_b.rms_curve))
        avg_energy_variance = (energy_variance_a + energy_variance_b) / 2
        
        dynamic_range_avg = (energy_profile_a.dynamics['dynamic_range_db'] + energy_profile_b.dynamics['dynamic_range_db']) / 2
        characteristics['energy_variance'] = avg_energy_variance
        characteristics['dynamic_range'] = float(dynamic_range_avg)
        
        # Beat detection characteristics
        if len(beat_grid_a.confidence_curve) > 0 and len(beat_grid_b.confidence_curve) > 0:
            beat_confidence_avg = (
                float(torch.mean(beat_grid_a.confidence_curve)) + 
                float(torch.mean(beat_grid_b.confidence_curve))
            ) / 2
        else:
            beat_confidence_avg = 0.5
        characteristics['beat_confidence_avg'] = beat_confidence_avg
        
        # Harmonic characteristics (if available)
        if key_profile_a and key_profile_b:
            harmonic_relevance_avg = (key_profile_a.harmonic_relevance + key_profile_b.harmonic_relevance) / 2
            key_confidence_avg = (key_profile_a.confidence + key_profile_b.confidence) / 2
            characteristics['harmonic_relevance'] = harmonic_relevance_avg
            characteristics['key_confidence'] = key_confidence_avg
        else:
            characteristics['harmonic_relevance'] = 0.5  # Neutral
            characteristics['key_confidence'] = 0.5
        
        # Spectral characteristics
        spectral_complexity_a = self._calculate_spectral_complexity(energy_profile_a)
        spectral_complexity_b = self._calculate_spectral_complexity(energy_profile_b)
        characteristics['spectral_complexity'] = (spectral_complexity_a + spectral_complexity_b) / 2
        
        # Rhythmic complexity
        rhythmic_complexity_a = self._calculate_rhythmic_complexity(beat_grid_a)
        rhythmic_complexity_b = self._calculate_rhythmic_complexity(beat_grid_b)
        characteristics['rhythmic_complexity'] = (rhythmic_complexity_a + rhythmic_complexity_b) / 2
        
        # Overall track similarity
        characteristics['track_similarity'] = self._calculate_track_similarity(
            beat_grid_a, beat_grid_b, energy_profile_a, energy_profile_b
        )
        
        # Processing difficulty prediction
        characteristics['processing_difficulty'] = self._predict_processing_difficulty(
            tempo_difference, avg_energy_variance, beat_confidence_avg
        )
        
        return characteristics
    
    def _calculate_spectral_complexity(self, energy_profile: EnergyProfile) -> float:
        """Calculate spectral complexity measure"""
        
        # Use spectral band variations as complexity indicator
        if energy_profile.spectral_bands.shape[1] > 1:
            spectral_variance = torch.var(energy_profile.spectral_bands, dim=1)
            complexity = float(torch.mean(spectral_variance))
        else:
            complexity = 0.5  # Default moderate complexity
        
        return min(1.0, max(0.0, complexity / 10.0))  # Normalize
    
    def _calculate_rhythmic_complexity(self, beat_grid: BeatGrid) -> float:
        """Calculate rhythmic complexity measure"""
        
        # Use tempo stability and beat confidence variations
        if len(beat_grid.confidence_curve) > 2:
            confidence_variance = float(torch.var(beat_grid.confidence_curve))
            tempo_factor = 1.0 - beat_grid.tempo_stability
            complexity = (confidence_variance + tempo_factor) / 2
        else:
            complexity = 0.5  # Default moderate complexity
        
        return min(1.0, max(0.0, complexity))
    
    def _calculate_track_similarity(self, 
                                  beat_grid_a: BeatGrid,
                                  beat_grid_b: BeatGrid,
                                  energy_profile_a: EnergyProfile,
                                  energy_profile_b: EnergyProfile) -> float:
        """Calculate overall similarity between tracks"""
        
        # Tempo similarity
        tempo_similarity = 1.0 - abs(beat_grid_a.bpm - beat_grid_b.bpm) / max(beat_grid_a.bpm, beat_grid_b.bpm)
        
        # Energy similarity
        energy_a_mean = float(torch.mean(energy_profile_a.rms_curve))
        energy_b_mean = float(torch.mean(energy_profile_b.rms_curve))
        energy_similarity = 1.0 - abs(energy_a_mean - energy_b_mean) / 20.0  # 20dB range
        energy_similarity = max(0.0, min(1.0, energy_similarity))
        
        # Dynamic range similarity
        range_similarity = 1.0 - abs(energy_profile_a.dynamics['dynamic_range_db'] - energy_profile_b.dynamics['dynamic_range_db']) / 30.0
        range_similarity = max(0.0, min(1.0, range_similarity))
        
        # Beat stability similarity
        stability_similarity = 1.0 - abs(beat_grid_a.tempo_stability - beat_grid_b.tempo_stability)
        
        # Combined similarity
        overall_similarity = (
            tempo_similarity * 0.3 +
            energy_similarity * 0.3 +
            range_similarity * 0.2 +
            stability_similarity * 0.2
        )
        
        return max(0.0, min(1.0, overall_similarity))
    
    def _predict_processing_difficulty(self, 
                                     tempo_difference: float,
                                     energy_variance: float,
                                     beat_confidence: float) -> float:
        """Predict how difficult processing will be"""
        
        # High tempo difference = harder processing
        tempo_difficulty = tempo_difference * 2.0
        
        # High energy variance = more complex crossfade
        energy_difficulty = min(1.0, energy_variance / 5.0)
        
        # Low beat confidence = harder alignment
        beat_difficulty = 1.0 - beat_confidence
        
        # Combined difficulty
        overall_difficulty = (tempo_difficulty + energy_difficulty + beat_difficulty) / 3
        
        return min(1.0, max(0.0, overall_difficulty))
    
    def _analyze_style_factors(self, 
                              beat_grid_a: BeatGrid,
                              beat_grid_b: BeatGrid,
                              energy_profile_a: EnergyProfile,
                              energy_profile_b: EnergyProfile) -> Dict[str, float]:
        """Analyze style/genre factors that influence thresholds"""
        
        style_factors = {}
        
        # Tempo-based style indicators
        avg_bpm = (beat_grid_a.bpm + beat_grid_b.bpm) / 2
        if avg_bpm < 100:
            style_factors['slow_genre_factor'] = 1.0 - (avg_bpm - 60) / 40  # Ballad, ambient
        elif avg_bpm > 140:
            style_factors['fast_genre_factor'] = (avg_bpm - 140) / 60  # EDM, metal, punk
        else:
            style_factors['moderate_genre_factor'] = 1.0 - abs(avg_bpm - 120) / 20  # Pop, rock
        
        # Energy-based style indicators
        avg_energy_a = float(torch.mean(energy_profile_a.rms_curve))
        avg_energy_b = float(torch.mean(energy_profile_b.rms_curve))
        avg_energy = (avg_energy_a + avg_energy_b) / 2
        
        if avg_energy > -10:  # High energy
            style_factors['high_energy_factor'] = min(1.0, (avg_energy + 30) / 20)
        else:  # Low energy
            style_factors['low_energy_factor'] = min(1.0, (-avg_energy - 10) / 20)
        
        # Dynamic range style indicators
        avg_dynamic_range = (energy_profile_a.dynamics['dynamic_range_db'] + energy_profile_b.dynamics['dynamic_range_db']) / 2
        if avg_dynamic_range > 20:
            style_factors['dynamic_music_factor'] = min(1.0, (avg_dynamic_range - 20) / 20)  # Classical, jazz
        else:
            style_factors['compressed_music_factor'] = min(1.0, (20 - avg_dynamic_range) / 15)  # Pop, EDM
        
        # Beat stability style indicators
        avg_stability = (beat_grid_a.tempo_stability + beat_grid_b.tempo_stability) / 2
        if avg_stability > 0.8:
            style_factors['electronic_factor'] = avg_stability  # Electronic music
        else:
            style_factors['organic_factor'] = 1.0 - avg_stability  # Live, acoustic music
        
        return style_factors
    
    def _apply_strategy_adjustments(self, 
                                  strategy: AdaptationStrategy,
                                  content_chars: Dict[str, float],
                                  style_factors: Dict[str, float]) -> Dict[str, float]:
        """Apply strategy-specific threshold adjustments"""
        
        adjustments = {}
        
        if strategy == AdaptationStrategy.CONSERVATIVE:
            # Stricter thresholds for higher quality
            adjustments = {
                'harmonic_compatibility_adjust': 0.1,    # Raise minimum
                'tempo_tolerance_adjust': -0.5,          # Tighter tolerance
                'energy_tolerance_adjust': -1.0,         # Tighter energy matching
                'beat_confidence_adjust': 0.1,           # Higher confidence required
                'key_stability_adjust': 0.1,             # More stable keys required
                'processing_quality_adjust': 0.1,        # Higher quality threshold
                'crossfade_duration_adjust': 1.2         # Longer crossfades
            }
            
        elif strategy == AdaptationStrategy.PERMISSIVE:
            # Looser thresholds for more flexibility
            adjustments = {
                'harmonic_compatibility_adjust': -0.1,   # Lower minimum
                'tempo_tolerance_adjust': 1.0,           # Looser tolerance
                'energy_tolerance_adjust': 2.0,          # Looser energy matching
                'beat_confidence_adjust': -0.1,          # Lower confidence OK
                'key_stability_adjust': -0.1,            # Less stable keys OK
                'processing_quality_adjust': -0.1,       # Lower quality threshold
                'crossfade_duration_adjust': 0.8         # Shorter crossfades
            }
            
        elif strategy == AdaptationStrategy.BALANCED:
            # Default balanced approach
            adjustments = {k: 0.0 for k in [
                'harmonic_compatibility_adjust', 'tempo_tolerance_adjust', 'energy_tolerance_adjust',
                'beat_confidence_adjust', 'key_stability_adjust', 'processing_quality_adjust'
            ]}
            adjustments['crossfade_duration_adjust'] = 1.0
            
        elif strategy == AdaptationStrategy.CONTEXT_ADAPTIVE:
            # Fully adaptive based on content
            adjustments = self._calculate_context_adaptive_adjustments(content_chars, style_factors)
        
        return adjustments
    
    def _calculate_context_adaptive_adjustments(self, 
                                              content_chars: Dict[str, float],
                                              style_factors: Dict[str, float]) -> Dict[str, float]:
        """Calculate fully adaptive adjustments based on content analysis"""
        
        adjustments = {}
        
        # Harmonic compatibility adjustment
        harmonic_relevance = content_chars.get('harmonic_relevance', 0.5)
        key_confidence = content_chars.get('key_confidence', 0.5)
        if harmonic_relevance < 0.3 or key_confidence < 0.4:
            adjustments['harmonic_compatibility_adjust'] = -0.15  # Less important for non-harmonic music
        else:
            adjustments['harmonic_compatibility_adjust'] = 0.1   # More important for harmonic music
        
        # Tempo tolerance adjustment
        tempo_stability = content_chars.get('tempo_stability', 0.5)
        if tempo_stability > 0.8:  # Very stable tempo (electronic)
            adjustments['tempo_tolerance_adjust'] = -0.5  # Tighter tolerance
        elif tempo_stability < 0.5:  # Variable tempo (live, organic)
            adjustments['tempo_tolerance_adjust'] = 1.0   # Looser tolerance
        else:
            adjustments['tempo_tolerance_adjust'] = 0.0
        
        # Energy matching based on dynamic range
        dynamic_range = content_chars.get('dynamic_range', 20.0)
        if dynamic_range > 25:  # High dynamic range music
            adjustments['energy_tolerance_adjust'] = 2.0  # Allow more energy variation
        elif dynamic_range < 15:  # Compressed music
            adjustments['energy_tolerance_adjust'] = -1.0  # Tighter energy matching
        else:
            adjustments['energy_tolerance_adjust'] = 0.0
        
        # Beat confidence based on rhythmic complexity
        rhythmic_complexity = content_chars.get('rhythmic_complexity', 0.5)
        if rhythmic_complexity > 0.7:  # Complex rhythms
            adjustments['beat_confidence_adjust'] = -0.1  # Lower confidence acceptable
        else:
            adjustments['beat_confidence_adjust'] = 0.0
        
        # Processing quality based on processing difficulty
        processing_difficulty = content_chars.get('processing_difficulty', 0.5)
        if processing_difficulty > 0.7:  # Difficult to process
            adjustments['processing_quality_adjust'] = -0.1  # Accept lower quality
        else:
            adjustments['processing_quality_adjust'] = 0.05  # Expect higher quality
        
        # Crossfade duration based on track similarity
        track_similarity = content_chars.get('track_similarity', 0.5)
        if track_similarity > 0.8:  # Very similar tracks
            adjustments['crossfade_duration_adjust'] = 0.7  # Shorter crossfades OK
        elif track_similarity < 0.3:  # Very different tracks
            adjustments['crossfade_duration_adjust'] = 1.5  # Longer crossfades needed
        else:
            adjustments['crossfade_duration_adjust'] = 1.0
        
        # Key stability based on harmonic relevance
        adjustments['key_stability_adjust'] = adjustments['harmonic_compatibility_adjust']
        
        return adjustments
    
    def _calculate_adaptive_thresholds(self, 
                                     content_chars: Dict[str, float],
                                     style_factors: Dict[str, float],
                                     adjustments: Dict[str, float],
                                     context: Optional[Dict] = None) -> Dict[str, float]:
        """Calculate final adaptive thresholds"""
        
        # Start with base thresholds
        thresholds = self.base_thresholds.copy()
        
        # Apply strategy adjustments
        thresholds['harmonic_compatibility_min'] += adjustments.get('harmonic_compatibility_adjust', 0.0)
        thresholds['tempo_tolerance_percent'] += adjustments.get('tempo_tolerance_adjust', 0.0)
        thresholds['energy_matching_tolerance_db'] += adjustments.get('energy_tolerance_adjust', 0.0)
        thresholds['beat_confidence_min'] += adjustments.get('beat_confidence_adjust', 0.0)
        thresholds['key_stability_min'] += adjustments.get('key_stability_adjust', 0.0)
        thresholds['processing_quality_min'] += adjustments.get('processing_quality_adjust', 0.0)
        
        # Apply crossfade duration adjustments
        duration_factor = adjustments.get('crossfade_duration_adjust', 1.0)
        thresholds['crossfade_duration_ms_min'] *= duration_factor
        thresholds['crossfade_duration_ms_max'] *= duration_factor
        
        # Apply style-specific modifications
        for style, factor in style_factors.items():
            if style == 'electronic_factor' and factor > 0.7:
                thresholds['tempo_tolerance_percent'] *= 0.8  # Electronic music needs tighter tempo
                thresholds['beat_confidence_min'] += 0.1 * factor
                
            elif style == 'organic_factor' and factor > 0.7:
                thresholds['tempo_tolerance_percent'] *= 1.3  # Organic music allows looser tempo
                thresholds['beat_confidence_min'] -= 0.1 * factor
                
            elif style == 'high_energy_factor' and factor > 0.7:
                thresholds['energy_matching_tolerance_db'] *= 1.2  # High energy allows more variation
                thresholds['crossfade_duration_ms_min'] *= 0.8  # Faster crossfades
                
            elif style == 'dynamic_music_factor' and factor > 0.7:
                thresholds['energy_matching_tolerance_db'] *= 1.4  # Dynamic music needs looser energy matching
                thresholds['crossfade_duration_ms_max'] *= 1.3  # Allow longer crossfades
        
        # Apply neural network refinement
        try:
            features = torch.tensor([
                content_chars.get('tempo_stability', 0.5),
                content_chars.get('energy_variance', 0.5),
                content_chars.get('beat_confidence_avg', 0.5),
                content_chars.get('harmonic_relevance', 0.5),
                content_chars.get('key_confidence', 0.5),
                content_chars.get('spectral_complexity', 0.5),
                content_chars.get('rhythmic_complexity', 0.5),
                content_chars.get('track_similarity', 0.5),
                content_chars.get('processing_difficulty', 0.5),
                sum(style_factors.values()),  # Total style influence
                thresholds['harmonic_compatibility_min'],
                thresholds['tempo_tolerance_percent'] / 10.0,  # Normalize
                thresholds['energy_matching_tolerance_db'] / 10.0,
                thresholds['beat_confidence_min'],
                thresholds['key_stability_min'],
                thresholds['processing_quality_min']
            ], dtype=torch.float32)
            
            with torch.no_grad():
                threshold_adjustments = self.threshold_adjuster(features.unsqueeze(0)).squeeze()
                
                # Apply neural adjustments (small refinements)
                adjustment_scale = 0.1 * self.adaptation_sensitivity
                thresholds['harmonic_compatibility_min'] += threshold_adjustments[0].item() * adjustment_scale
                thresholds['tempo_tolerance_percent'] += threshold_adjustments[1].item() * adjustment_scale * 2.0
                thresholds['energy_matching_tolerance_db'] += threshold_adjustments[2].item() * adjustment_scale * 3.0
                thresholds['beat_confidence_min'] += threshold_adjustments[3].item() * adjustment_scale
                thresholds['key_stability_min'] += threshold_adjustments[4].item() * adjustment_scale
                thresholds['processing_quality_min'] += threshold_adjustments[5].item() * adjustment_scale
                
                duration_adjustment = threshold_adjustments[6].item() * adjustment_scale * 0.3 + 1.0
                thresholds['crossfade_duration_ms_min'] *= duration_adjustment
                thresholds['crossfade_duration_ms_max'] *= duration_adjustment
        
        except:
            pass  # Use calculated thresholds without neural refinement
        
        return thresholds
    
    def _validate_and_constrain_thresholds(self, thresholds: Dict[str, float]) -> Dict[str, float]:
        """Validate and apply constraints to threshold values"""
        
        # Apply reasonable constraints
        constraints = {
            'harmonic_compatibility_min': (0.0, 1.0),
            'tempo_tolerance_percent': (0.5, 10.0),
            'energy_matching_tolerance_db': (2.0, 15.0),
            'beat_confidence_min': (0.1, 0.9),
            'key_stability_min': (0.1, 0.9),
            'processing_quality_min': (0.3, 0.95),
            'crossfade_duration_ms_min': (200.0, 5000.0),
            'crossfade_duration_ms_max': (1000.0, 15000.0)
        }
        
        constrained = {}
        for key, (min_val, max_val) in constraints.items():
            if key in thresholds:
                constrained[key] = max(min_val, min(max_val, thresholds[key]))
            else:
                constrained[key] = self.base_thresholds.get(key, (min_val + max_val) / 2)
        
        # Ensure min duration < max duration
        if constrained['crossfade_duration_ms_min'] >= constrained['crossfade_duration_ms_max']:
            constrained['crossfade_duration_ms_max'] = constrained['crossfade_duration_ms_min'] * 2
        
        return constrained
    
    def _calculate_adaptation_confidence(self, 
                                       content_chars: Dict[str, float],
                                       style_factors: Dict[str, float]) -> float:
        """Calculate confidence in threshold adaptation"""
        
        # High confidence when content characteristics are clear
        characteristics_clarity = sum([
            abs(content_chars.get('tempo_stability', 0.5) - 0.5) * 2,
            abs(content_chars.get('beat_confidence_avg', 0.5) - 0.5) * 2,
            abs(content_chars.get('harmonic_relevance', 0.5) - 0.5) * 2,
            abs(content_chars.get('track_similarity', 0.5) - 0.5) * 2
        ]) / 4
        
        # High confidence when style factors are pronounced
        style_clarity = max(style_factors.values()) if style_factors else 0.5
        
        # Combined confidence
        adaptation_confidence = (characteristics_clarity + style_clarity) / 2
        
        # Boost confidence if we have good data quality
        data_quality = content_chars.get('beat_confidence_avg', 0.5)
        adaptation_confidence = adaptation_confidence * 0.7 + data_quality * 0.3
        
        return max(0.3, min(0.95, adaptation_confidence))  # Reasonable confidence range
    
    def get_threshold_summary(self, threshold_set: ThresholdSet) -> str:
        """Generate human-readable threshold summary"""
        
        summary = "=== Adaptive Threshold Configuration ===\n"
        summary += f"Harmonic Compatibility Min: {threshold_set.harmonic_compatibility_min:.3f}\n"
        summary += f"Tempo Tolerance: ±{threshold_set.tempo_tolerance_percent:.1f}%\n"
        summary += f"Energy Matching Tolerance: ±{threshold_set.energy_matching_tolerance_db:.1f} dB\n"
        summary += f"Beat Confidence Min: {threshold_set.beat_confidence_min:.3f}\n"
        summary += f"Key Stability Min: {threshold_set.key_stability_min:.3f}\n"
        summary += f"Processing Quality Min: {threshold_set.processing_quality_min:.3f}\n"
        summary += f"Crossfade Duration: {threshold_set.crossfade_duration_ms_range[0]:.0f}-{threshold_set.crossfade_duration_ms_range[1]:.0f} ms\n"
        summary += f"Adaptation Confidence: {threshold_set.adaptation_confidence:.3f}\n"
        
        summary += "\nContent Analysis Factors:\n"
        for factor, value in threshold_set.content_analysis.items():
            summary += f"  {factor.replace('_', ' ').title()}: {value:.3f}\n"
        
        return summary


if __name__ == "__main__":
    # Test the module
    from .beat_grid_extractor import BeatGridExtractor, create_test_audio
    from .energy_profile_extractor import EnergyProfileExtractor
    from .musical_key_extractor import MusicalKeyExtractor
    
    # Create test tracks with different characteristics
    electronic_track = create_test_audio(10.0, 128.0)  # Electronic music
    organic_track = create_test_audio(10.0, 95.0)      # Slower, more organic
    
    # Extract features
    beat_extractor = BeatGridExtractor()
    energy_extractor = EnergyProfileExtractor()
    key_extractor = MusicalKeyExtractor()
    
    beat_grid_electronic = beat_extractor(electronic_track)
    beat_grid_organic = beat_extractor(organic_track)
    
    energy_electronic = energy_extractor(electronic_track)
    energy_organic = energy_extractor(organic_track)
    
    key_electronic = key_extractor(electronic_track)
    key_organic = key_extractor(organic_track)
    
    # Test adaptive threshold calculation
    calculator = AdaptiveThresholdCalculator()
    
    # Test different adaptation strategies
    strategies = [
        AdaptationStrategy.CONSERVATIVE,
        AdaptationStrategy.BALANCED,
        AdaptationStrategy.PERMISSIVE,
        AdaptationStrategy.CONTEXT_ADAPTIVE
    ]
    
    print("=== Adaptive Threshold Calculator Test ===\n")
    
    for strategy in strategies:
        threshold_set = calculator(
            beat_grid_electronic, beat_grid_organic,
            energy_electronic, energy_organic,
            key_electronic, key_organic,
            strategy
        )
        
        print(f"Strategy: {strategy.value.upper()}")
        print(f"Harmonic threshold: {threshold_set.harmonic_compatibility_min:.3f}")
        print(f"Tempo tolerance: ±{threshold_set.tempo_tolerance_percent:.1f}%")
        print(f"Energy tolerance: ±{threshold_set.energy_matching_tolerance_db:.1f} dB")
        print(f"Confidence: {threshold_set.adaptation_confidence:.3f}")
        print()
    
    # Detailed summary for context-adaptive strategy
    context_adaptive_thresholds = calculator(
        beat_grid_electronic, beat_grid_organic,
        energy_electronic, energy_organic,
        key_electronic, key_organic,
        AdaptationStrategy.CONTEXT_ADAPTIVE
    )
    
    print(calculator.get_threshold_summary(context_adaptive_thresholds))