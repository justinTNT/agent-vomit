"""
FallbackStrategyOptimizer - Enhanced hard cut timing when tempo outside 5% tolerance

This module optimizes hard cut timing strategies for when tempo correction
isn't viable, using beat grid analysis for optimal transition timing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .beat_grid_extractor import BeatGrid
from .energy_profile_extractor import EnergyProfile
from .rhythmic_compatibility_analyzer import RhythmicMatch, TempoCompatibility


class FallbackStrategy(Enum):
    """Hard cut fallback strategies"""
    IMMEDIATE_CUT = "immediate_cut"                    # Cut immediately at optimal point
    BEAT_ALIGNED_CUT = "beat_aligned_cut"             # Wait for next beat boundary  
    BAR_ALIGNED_CUT = "bar_aligned_cut"               # Wait for next bar boundary
    PHRASE_ALIGNED_CUT = "phrase_aligned_cut"         # Wait for phrase completion
    ENERGY_OPTIMIZED_CUT = "energy_optimized_cut"     # Cut at energy minimum
    SYNCOPATED_CUT = "syncopated_cut"                 # Cut on off-beat for effect
    CROSSBEAT_TRANSITION = "crossbeat_transition"     # Transition across beat patterns


class CutQuality(Enum):
    """Quality levels for hard cuts"""
    EXCELLENT = "excellent"    # Perfect timing, no artifacts
    GOOD = "good"             # Good timing, minimal artifacts
    ACCEPTABLE = "acceptable" # Acceptable timing, some artifacts
    POOR = "poor"            # Poor timing, noticeable artifacts
    UNACCEPTABLE = "unacceptable"  # Unacceptable artifacts


@dataclass
class FallbackPlan:
    """Hard cut fallback plan specification"""
    strategy: FallbackStrategy           # Selected fallback strategy
    cut_position_a: int                 # Sample position to cut Track A
    start_position_b: int               # Sample position to start Track B
    timing_offset_samples: int          # Additional timing adjustment
    crossfade_duration: int             # Minimal crossfade for click removal
    expected_quality: CutQuality        # Expected cut quality
    confidence: float                   # Confidence in strategy [0,1]
    beat_alignment_score: float         # How well aligned with beats [0,1]
    energy_continuity_score: float     # Energy flow score [0,1]
    processing_notes: str               # Human-readable explanation


class FallbackStrategyOptimizer(nn.Module):
    """
    Optimize hard cut timing when tempo correction isn't viable.
    
    Uses beat grid analysis and energy profiling to find optimal
    hard cut positions that minimize artifacts and maximize musicality.
    """
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 min_crossfade_ms: float = 10.0,    # Minimal crossfade for click removal
                 max_crossfade_ms: float = 50.0,    # Maximum "hard cut" crossfade
                 tempo_tolerance: float = 0.05):     # 5% tempo tolerance
        super().__init__()
        
        self.sample_rate = sample_rate
        self.min_crossfade_samples = int(min_crossfade_ms * sample_rate / 1000)
        self.max_crossfade_samples = int(max_crossfade_ms * sample_rate / 1000)
        self.tempo_tolerance = tempo_tolerance
        
        # Neural network for strategy selection
        self.strategy_selector = nn.Sequential(
            nn.Linear(10, 64),  # Beat grid + energy features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 7)    # 7 fallback strategies
        )
        
        # Cut quality predictor
        self.quality_predictor = nn.Sequential(
            nn.Linear(8, 32),   # Timing and alignment features
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 5)    # 5 quality levels
        )
        
        # Timing optimizer
        self.timing_optimizer = nn.Sequential(
            nn.Linear(12, 64),  # Beat and energy features
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # Output offset as fraction of beat interval
        )

    def forward(self, 
                beat_grid_a: BeatGrid,
                beat_grid_b: BeatGrid,
                energy_profile_a: EnergyProfile,
                energy_profile_b: EnergyProfile,
                rhythmic_match: RhythmicMatch,
                target_cut_position_a: int,
                target_start_position_b: int) -> FallbackPlan:
        """
        Generate optimal hard cut fallback plan.
        
        Args:
            beat_grid_a: Track A beat analysis
            beat_grid_b: Track B beat analysis
            energy_profile_a: Track A energy analysis
            energy_profile_b: Track B energy analysis  
            rhythmic_match: Rhythmic compatibility analysis
            target_cut_position_a: Desired cut position in A
            target_start_position_b: Desired start position in B
            
        Returns:
            FallbackPlan with optimized hard cut strategy
        """
        
        # Determine if fallback is actually needed
        if self._is_tempo_correction_viable(rhythmic_match):
            return self._create_minimal_fallback(target_cut_position_a, target_start_position_b)
        
        # Analyze timing context around target positions
        timing_context_a = self._analyze_timing_context(
            beat_grid_a, energy_profile_a, target_cut_position_a
        )
        
        timing_context_b = self._analyze_timing_context(
            beat_grid_b, energy_profile_b, target_start_position_b
        )
        
        # Select optimal fallback strategy
        strategy = self._select_fallback_strategy(
            timing_context_a, timing_context_b, rhythmic_match
        )
        
        # Optimize cut timing based on strategy
        optimized_positions = self._optimize_cut_timing(
            strategy, timing_context_a, timing_context_b,
            target_cut_position_a, target_start_position_b
        )
        
        # Calculate crossfade duration
        crossfade_duration = self._calculate_crossfade_duration(
            strategy, optimized_positions, timing_context_a, timing_context_b
        )
        
        # Predict cut quality
        quality = self._predict_cut_quality(
            strategy, optimized_positions, timing_context_a, timing_context_b
        )
        
        # Calculate confidence and scores
        confidence = self._calculate_strategy_confidence(
            strategy, quality, timing_context_a, timing_context_b
        )
        
        beat_alignment_score = self._calculate_beat_alignment_score(
            optimized_positions, timing_context_a, timing_context_b
        )
        
        energy_continuity_score = self._calculate_energy_continuity_score(
            optimized_positions, timing_context_a, timing_context_b
        )
        
        # Generate processing notes
        processing_notes = self._generate_processing_notes(
            strategy, quality, rhythmic_match
        )
        
        return FallbackPlan(
            strategy=strategy,
            cut_position_a=optimized_positions['cut_a'],
            start_position_b=optimized_positions['start_b'],
            timing_offset_samples=optimized_positions['offset'],
            crossfade_duration=crossfade_duration,
            expected_quality=quality,
            confidence=float(confidence),
            beat_alignment_score=float(beat_alignment_score),
            energy_continuity_score=float(energy_continuity_score),
            processing_notes=processing_notes
        )
    
    def _is_tempo_correction_viable(self, rhythmic_match: RhythmicMatch) -> bool:
        """Check if tempo correction is still viable"""
        
        rate_change_needed = abs(rhythmic_match.required_rate_change - 1.0)
        return rate_change_needed <= self.tempo_tolerance
    
    def _analyze_timing_context(self, 
                               beat_grid: BeatGrid,
                               energy_profile: EnergyProfile,
                               target_position: int) -> Dict:
        """Analyze timing context around target position"""
        
        # Find nearest beats
        if len(beat_grid.beat_times) > 0:
            beat_distances = torch.abs(beat_grid.beat_times - target_position)
            nearest_beat_idx = torch.argmin(beat_distances)
            nearest_beat_position = int(beat_grid.beat_times[nearest_beat_idx])
            beat_confidence = beat_grid.confidence_curve[nearest_beat_idx] if nearest_beat_idx < len(beat_grid.confidence_curve) else 0.5
        else:
            nearest_beat_position = target_position
            beat_confidence = 0.5
        
        # Find nearest bar
        if len(beat_grid.bar_times) > 0:
            bar_distances = torch.abs(beat_grid.bar_times - target_position)
            nearest_bar_idx = torch.argmin(bar_distances)
            nearest_bar_position = int(beat_grid.bar_times[nearest_bar_idx])
        else:
            # Estimate bar position (assume 4/4 time)
            if len(beat_grid.beat_times) >= 4:
                beat_interval = torch.mean(torch.diff(beat_grid.beat_times.float()))
                bar_interval = beat_interval * 4
                estimated_bar = int(nearest_beat_position + (nearest_beat_idx % 4) * beat_interval)
                nearest_bar_position = estimated_bar
            else:
                nearest_bar_position = nearest_beat_position
        
        # Energy analysis
        frame_position = int(target_position // (self.sample_rate / len(energy_profile.rms_curve)))
        frame_position = max(0, min(frame_position, len(energy_profile.rms_curve) - 1))
        
        current_energy = energy_profile.rms_curve[frame_position]
        energy_slope = energy_profile.energy_slope[frame_position] if frame_position < len(energy_profile.energy_slope) else 0.0
        
        # Find local energy minimum (good for cuts)
        window_size = min(20, len(energy_profile.rms_curve) // 10)
        start_frame = max(0, frame_position - window_size // 2)
        end_frame = min(len(energy_profile.rms_curve), frame_position + window_size // 2)
        
        local_energy = energy_profile.rms_curve[start_frame:end_frame]
        if len(local_energy) > 0:
            local_min_idx = torch.argmin(local_energy) + start_frame
            local_min_position = int(local_min_idx * self.sample_rate / len(energy_profile.rms_curve))
        else:
            local_min_position = target_position
        
        return {
            'target_position': target_position,
            'nearest_beat_position': nearest_beat_position,
            'nearest_bar_position': nearest_bar_position,
            'beat_confidence': float(beat_confidence),
            'current_energy_db': float(current_energy),
            'energy_slope': float(energy_slope),
            'local_min_position': local_min_position,
            'bpm': beat_grid.bpm,
            'tempo_stability': beat_grid.tempo_stability
        }
    
    def _select_fallback_strategy(self, 
                                 context_a: Dict,
                                 context_b: Dict,
                                 rhythmic_match: RhythmicMatch) -> FallbackStrategy:
        """Select optimal fallback strategy"""
        
        # Feature vector for neural selection
        features = torch.tensor([
            context_a['beat_confidence'],
            context_b['beat_confidence'],
            context_a['current_energy_db'] / 60.0,  # Normalize
            context_b['current_energy_db'] / 60.0,
            context_a['energy_slope'],
            context_b['energy_slope'],
            context_a['tempo_stability'],
            context_b['tempo_stability'],
            rhythmic_match.tempo_stability_factor,
            abs(rhythmic_match.required_rate_change - 1.0) / 0.5  # Normalize
        ], dtype=torch.float32)
        
        try:
            with torch.no_grad():
                strategy_logits = self.strategy_selector(features.unsqueeze(0))
                strategy_probs = F.softmax(strategy_logits, dim=1).squeeze()
                
                strategies = [
                    FallbackStrategy.BEAT_ALIGNED_CUT,
                    FallbackStrategy.BAR_ALIGNED_CUT,
                    FallbackStrategy.ENERGY_OPTIMIZED_CUT,
                    FallbackStrategy.PHRASE_ALIGNED_CUT,
                    FallbackStrategy.CROSSBEAT_TRANSITION,
                    FallbackStrategy.SYNCOPATED_CUT,
                    FallbackStrategy.IMMEDIATE_CUT
                ]
                
                selected_strategy = strategies[torch.argmax(strategy_probs)]
                
        except:
            # Fallback to heuristic selection
            selected_strategy = self._heuristic_strategy_selection(context_a, context_b, rhythmic_match)
        
        return selected_strategy
    
    def _heuristic_strategy_selection(self, 
                                    context_a: Dict,
                                    context_b: Dict,
                                    rhythmic_match: RhythmicMatch) -> FallbackStrategy:
        """Heuristic strategy selection fallback"""
        
        # High beat confidence -> use beat alignment
        if context_a['beat_confidence'] > 0.8 and context_b['beat_confidence'] > 0.8:
            return FallbackStrategy.BEAT_ALIGNED_CUT
        
        # High tempo stability -> use bar alignment
        if context_a['tempo_stability'] > 0.8:
            return FallbackStrategy.BAR_ALIGNED_CUT
        
        # Low energy -> use energy optimization
        if context_a['current_energy_db'] < -20 or context_b['current_energy_db'] < -20:
            return FallbackStrategy.ENERGY_OPTIMIZED_CUT
        
        # Default to beat alignment
        return FallbackStrategy.BEAT_ALIGNED_CUT
    
    def _optimize_cut_timing(self, 
                            strategy: FallbackStrategy,
                            context_a: Dict,
                            context_b: Dict,
                            target_cut_a: int,
                            target_start_b: int) -> Dict[str, int]:
        """Optimize cut timing based on strategy"""
        
        if strategy == FallbackStrategy.IMMEDIATE_CUT:
            # Use target positions directly
            return {
                'cut_a': target_cut_a,
                'start_b': target_start_b,
                'offset': 0
            }
        
        elif strategy == FallbackStrategy.BEAT_ALIGNED_CUT:
            # Align to nearest beats
            cut_a = context_a['nearest_beat_position']
            start_b = context_b['nearest_beat_position']
            offset = 0
            
        elif strategy == FallbackStrategy.BAR_ALIGNED_CUT:
            # Align to nearest bars
            cut_a = context_a['nearest_bar_position']
            start_b = context_b['nearest_bar_position']
            offset = 0
            
        elif strategy == FallbackStrategy.ENERGY_OPTIMIZED_CUT:
            # Use local energy minima
            cut_a = context_a['local_min_position']
            start_b = context_b['local_min_position']
            offset = 0
            
        elif strategy == FallbackStrategy.PHRASE_ALIGNED_CUT:
            # Align to estimated phrase boundaries (simplified)
            # Real implementation would use phrase detection
            cut_a = context_a['nearest_bar_position']
            start_b = context_b['nearest_bar_position']
            offset = 0
            
        elif strategy == FallbackStrategy.SYNCOPATED_CUT:
            # Cut on off-beat for stylistic effect
            beat_interval = int(60.0 * self.sample_rate / context_a['bpm'])
            cut_a = context_a['nearest_beat_position'] + beat_interval // 2
            start_b = context_b['nearest_beat_position'] + beat_interval // 2
            offset = 0
            
        elif strategy == FallbackStrategy.CROSSBEAT_TRANSITION:
            # Transition across different beat patterns
            cut_a = context_a['nearest_beat_position']
            start_b = context_b['nearest_beat_position']
            # Add small offset to avoid exact alignment
            beat_interval_b = int(60.0 * self.sample_rate / context_b['bpm'])
            offset = beat_interval_b // 4  # Quarter beat offset
            
        else:
            # Default to target positions
            cut_a = target_cut_a
            start_b = target_start_b
            offset = 0
        
        # Apply neural timing optimization
        try:
            timing_features = torch.tensor([
                (cut_a - target_cut_a) / self.sample_rate,
                (start_b - target_start_b) / self.sample_rate,
                context_a['beat_confidence'],
                context_b['beat_confidence'],
                context_a['current_energy_db'] / 60.0,
                context_b['current_energy_db'] / 60.0,
                context_a['energy_slope'],
                context_b['energy_slope'],
                context_a['tempo_stability'],
                context_b['tempo_stability'],
                float(strategy.value == 'beat_aligned_cut'),
                float(strategy.value == 'bar_aligned_cut')
            ], dtype=torch.float32)
            
            with torch.no_grad():
                timing_adjustment = self.timing_optimizer(timing_features.unsqueeze(0))
                timing_adjustment = timing_adjustment.item()
                
                # Apply adjustment (as fraction of beat interval)
                beat_interval = int(60.0 * self.sample_rate / context_a['bpm'])
                adjustment_samples = int(timing_adjustment * beat_interval * 0.1)  # Max 10% of beat
                
                cut_a += adjustment_samples
                start_b += adjustment_samples
        
        except:
            pass  # Use positions as calculated above
        
        return {
            'cut_a': cut_a,
            'start_b': start_b,
            'offset': offset
        }
    
    def _calculate_crossfade_duration(self, 
                                    strategy: FallbackStrategy,
                                    positions: Dict[str, int],
                                    context_a: Dict,
                                    context_b: Dict) -> int:
        """Calculate minimal crossfade duration for click removal"""
        
        # Base duration for click removal
        base_duration = self.min_crossfade_samples
        
        # Strategy-specific adjustments
        if strategy == FallbackStrategy.IMMEDIATE_CUT:
            # Minimal crossfade
            duration = base_duration
            
        elif strategy in [FallbackStrategy.BEAT_ALIGNED_CUT, FallbackStrategy.BAR_ALIGNED_CUT]:
            # Slightly longer for musical alignment
            duration = int(base_duration * 1.5)
            
        elif strategy == FallbackStrategy.ENERGY_OPTIMIZED_CUT:
            # Medium duration for energy matching
            duration = int(base_duration * 2.0)
            
        elif strategy == FallbackStrategy.SYNCOPATED_CUT:
            # Longer for stylistic effect
            duration = int(base_duration * 3.0)
            
        else:
            duration = base_duration
        
        # Clamp to maximum
        duration = min(duration, self.max_crossfade_samples)
        
        return duration
    
    def _predict_cut_quality(self, 
                           strategy: FallbackStrategy,
                           positions: Dict[str, int],
                           context_a: Dict,
                           context_b: Dict) -> CutQuality:
        """Predict quality of the cut"""
        
        # Feature vector for quality prediction
        features = torch.tensor([
            context_a['beat_confidence'],
            context_b['beat_confidence'],
            abs(context_a['current_energy_db'] - context_b['current_energy_db']) / 20.0,
            abs(context_a['energy_slope'] - context_b['energy_slope']),
            float(strategy.value in ['beat_aligned_cut', 'bar_aligned_cut']),
            context_a['tempo_stability'],
            context_b['tempo_stability'],
            float(positions['offset']) / self.sample_rate
        ], dtype=torch.float32)
        
        try:
            with torch.no_grad():
                quality_logits = self.quality_predictor(features.unsqueeze(0))
                quality_probs = F.softmax(quality_logits, dim=1).squeeze()
                
                qualities = [
                    CutQuality.EXCELLENT,
                    CutQuality.GOOD,
                    CutQuality.ACCEPTABLE,
                    CutQuality.POOR,
                    CutQuality.UNACCEPTABLE
                ]
                
                predicted_quality = qualities[torch.argmax(quality_probs)]
        
        except:
            # Fallback heuristic
            predicted_quality = self._heuristic_quality_prediction(strategy, context_a, context_b)
        
        return predicted_quality
    
    def _heuristic_quality_prediction(self, 
                                    strategy: FallbackStrategy,
                                    context_a: Dict,
                                    context_b: Dict) -> CutQuality:
        """Heuristic quality prediction fallback"""
        
        # Strategy quality mapping
        strategy_quality = {
            FallbackStrategy.BAR_ALIGNED_CUT: CutQuality.EXCELLENT,
            FallbackStrategy.BEAT_ALIGNED_CUT: CutQuality.GOOD,
            FallbackStrategy.PHRASE_ALIGNED_CUT: CutQuality.GOOD,
            FallbackStrategy.ENERGY_OPTIMIZED_CUT: CutQuality.ACCEPTABLE,
            FallbackStrategy.CROSSBEAT_TRANSITION: CutQuality.ACCEPTABLE,
            FallbackStrategy.SYNCOPATED_CUT: CutQuality.ACCEPTABLE,
            FallbackStrategy.IMMEDIATE_CUT: CutQuality.POOR
        }
        
        base_quality = strategy_quality[strategy]
        
        # Adjust based on beat confidence
        avg_confidence = (context_a['beat_confidence'] + context_b['beat_confidence']) / 2
        if avg_confidence < 0.3:
            # Downgrade quality for poor beat detection
            quality_levels = list(CutQuality)
            current_idx = quality_levels.index(base_quality)
            downgraded_idx = min(current_idx + 1, len(quality_levels) - 1)
            base_quality = quality_levels[downgraded_idx]
        
        return base_quality
    
    def _calculate_strategy_confidence(self, 
                                     strategy: FallbackStrategy,
                                     quality: CutQuality,
                                     context_a: Dict,
                                     context_b: Dict) -> float:
        """Calculate confidence in strategy selection"""
        
        # Base confidence from quality
        quality_confidence = {
            CutQuality.EXCELLENT: 0.9,
            CutQuality.GOOD: 0.8,
            CutQuality.ACCEPTABLE: 0.6,
            CutQuality.POOR: 0.4,
            CutQuality.UNACCEPTABLE: 0.2
        }[quality]
        
        # Beat confidence factor
        beat_confidence = (context_a['beat_confidence'] + context_b['beat_confidence']) / 2
        
        # Tempo stability factor
        tempo_confidence = (context_a['tempo_stability'] + context_b['tempo_stability']) / 2
        
        # Combined confidence
        confidence = (
            quality_confidence * 0.5 +
            beat_confidence * 0.3 +
            tempo_confidence * 0.2
        )
        
        return confidence
    
    def _calculate_beat_alignment_score(self, 
                                      positions: Dict[str, int],
                                      context_a: Dict,
                                      context_b: Dict) -> float:
        """Calculate beat alignment score"""
        
        # Distance from nearest beats
        beat_dist_a = abs(positions['cut_a'] - context_a['nearest_beat_position'])
        beat_dist_b = abs(positions['start_b'] - context_b['nearest_beat_position'])
        
        # Convert to beat fractions
        beat_interval_a = 60.0 * self.sample_rate / context_a['bpm']
        beat_interval_b = 60.0 * self.sample_rate / context_b['bpm']
        
        alignment_a = 1.0 - min(1.0, beat_dist_a / (beat_interval_a * 0.25))  # Within quarter beat
        alignment_b = 1.0 - min(1.0, beat_dist_b / (beat_interval_b * 0.25))
        
        return (alignment_a + alignment_b) / 2
    
    def _calculate_energy_continuity_score(self, 
                                         positions: Dict[str, int],
                                         context_a: Dict,
                                         context_b: Dict) -> float:
        """Calculate energy continuity score"""
        
        # Energy level difference
        energy_diff = abs(context_a['current_energy_db'] - context_b['current_energy_db'])
        energy_score = max(0.0, 1.0 - energy_diff / 20.0)  # 20dB max difference
        
        # Energy slope compatibility
        slope_diff = abs(context_a['energy_slope'] - context_b['energy_slope'])
        slope_score = max(0.0, 1.0 - slope_diff)
        
        return (energy_score + slope_score) / 2
    
    def _generate_processing_notes(self, 
                                 strategy: FallbackStrategy,
                                 quality: CutQuality,
                                 rhythmic_match: RhythmicMatch) -> str:
        """Generate human-readable processing notes"""
        
        tempo_diff = abs(rhythmic_match.required_rate_change - 1.0) * 100
        
        notes = f"Hard cut strategy selected due to {tempo_diff:.1f}% tempo difference (>{self.tempo_tolerance*100:.0f}% tolerance). "
        notes += f"Using {strategy.value.replace('_', ' ')} approach. "
        notes += f"Expected quality: {quality.value}. "
        
        if quality in [CutQuality.POOR, CutQuality.UNACCEPTABLE]:
            notes += "Consider manual timing adjustment or alternative track selection."
        elif quality == CutQuality.ACCEPTABLE:
            notes += "Cut quality acceptable but monitor for artifacts."
        else:
            notes += "Cut should be clean with minimal artifacts."
        
        return notes
    
    def _create_minimal_fallback(self, cut_pos_a: int, start_pos_b: int) -> FallbackPlan:
        """Create minimal fallback when tempo correction is viable"""
        
        return FallbackPlan(
            strategy=FallbackStrategy.IMMEDIATE_CUT,
            cut_position_a=cut_pos_a,
            start_position_b=start_pos_b,
            timing_offset_samples=0,
            crossfade_duration=self.min_crossfade_samples,
            expected_quality=CutQuality.GOOD,
            confidence=0.9,
            beat_alignment_score=0.7,
            energy_continuity_score=0.7,
            processing_notes="Tempo correction viable, minimal fallback applied."
        )
    
    def get_fallback_summary(self, fallback_plan: FallbackPlan) -> str:
        """Generate human-readable fallback summary"""
        
        cut_time_a = fallback_plan.cut_position_a / self.sample_rate
        start_time_b = fallback_plan.start_position_b / self.sample_rate
        crossfade_ms = fallback_plan.crossfade_duration * 1000 / self.sample_rate
        
        summary = f"Fallback Strategy: {fallback_plan.strategy.value.replace('_', ' ').title()}\n"
        summary += f"Cut A at: {cut_time_a:.3f}s, Start B at: {start_time_b:.3f}s\n"
        summary += f"Crossfade: {crossfade_ms:.1f}ms (click removal)\n"
        summary += f"Quality: {fallback_plan.expected_quality.value} (confidence: {fallback_plan.confidence:.2f})\n"
        summary += f"Beat alignment: {fallback_plan.beat_alignment_score:.2f}, "
        summary += f"Energy continuity: {fallback_plan.energy_continuity_score:.2f}\n"
        summary += f"Notes: {fallback_plan.processing_notes}"
        
        return summary


if __name__ == "__main__":
    # Test the module
    from .beat_grid_extractor import BeatGridExtractor, create_test_audio
    from .energy_profile_extractor import EnergyProfileExtractor
    from .rhythmic_compatibility_analyzer import RhythmicCompatibilityAnalyzer
    
    # Create test audio with incompatible tempos
    audio_120 = create_test_audio(30.0, 120.0)
    audio_140 = create_test_audio(30.0, 140.0)  # 16.7% tempo difference
    
    # Extract features
    beat_extractor = BeatGridExtractor()
    energy_extractor = EnergyProfileExtractor()
    rhythm_analyzer = RhythmicCompatibilityAnalyzer()
    
    beat_grid_a = beat_extractor(audio_120)
    beat_grid_b = beat_extractor(audio_140)
    energy_profile_a = energy_extractor(audio_120)
    energy_profile_b = energy_extractor(audio_140)
    
    rhythmic_match = rhythm_analyzer(beat_grid_a, beat_grid_b)
    
    # Test fallback optimizer
    optimizer = FallbackStrategyOptimizer()
    
    fallback_plan = optimizer(
        beat_grid_a, beat_grid_b,
        energy_profile_a, energy_profile_b,
        rhythmic_match,
        20*44100,  # Cut at 20s
        5*44100    # Start B at 5s
    )
    
    print("=== Fallback Strategy Optimization ===")
    print(f"Strategy: {fallback_plan.strategy.value}")
    print(f"Cut A position: {fallback_plan.cut_position_a} samples ({fallback_plan.cut_position_a/44100:.2f}s)")
    print(f"Start B position: {fallback_plan.start_position_b} samples ({fallback_plan.start_position_b/44100:.2f}s)")
    print(f"Expected quality: {fallback_plan.expected_quality.value}")
    print(f"Confidence: {fallback_plan.confidence:.3f}")
    print(f"Beat alignment: {fallback_plan.beat_alignment_score:.3f}")
    print(f"Energy continuity: {fallback_plan.energy_continuity_score:.3f}")
    
    # Test summary
    summary = optimizer.get_fallback_summary(fallback_plan)
    print(f"\nSummary:\n{summary}")