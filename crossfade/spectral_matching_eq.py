"""
SpectralMatchingEQ - Frequency content matching for smooth spectral transitions

This module implements established EQ matching techniques to minimize spectral
discontinuities during crossfades, supporting the "first lows then highs" approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from .energy_profile_extractor import EnergyProfile
from .low_end_conflict_analyzer import LowEndConflict, ConflictSeverity


class EQBand(Enum):
    """EQ frequency bands"""
    SUB_BASS = "sub_bass"      # 20-60 Hz
    BASS = "bass"              # 60-200 Hz  
    LOW_MID = "low_mid"        # 200-500 Hz
    MID = "mid"                # 500-2000 Hz
    HIGH_MID = "high_mid"      # 2-6 kHz
    HIGH = "high"              # 6-20 kHz


class EQStrategy(Enum):
    """EQ processing strategies"""
    MATCH_A_TO_B = "match_a_to_b"          # Make A sound like B
    MATCH_B_TO_A = "match_b_to_a"          # Make B sound like A (preferred)
    BIDIRECTIONAL = "bidirectional"        # Adjust both tracks
    FREQUENCY_SELECTIVE = "frequency_selective"  # Different strategies per band
    NO_EQ_NEEDED = "no_eq_needed"          # Spectral content already compatible


@dataclass
class EQAdjustment:
    """EQ adjustment specification for a frequency band"""
    frequency_hz: float        # Center frequency
    gain_db: float            # Gain adjustment (+/-)
    q_factor: float           # Filter Q factor
    filter_type: str          # 'bell', 'high_shelf', 'low_shelf', 'highpass', 'lowpass'


@dataclass
class SpectralMatch:
    """Spectral matching analysis and EQ recommendations"""
    compatibility_score: float              # Spectral compatibility [0,1]
    strategy: EQStrategy                    # Recommended EQ strategy
    track_b_adjustments: List[EQAdjustment] # EQ adjustments for Track B
    track_a_adjustments: List[EQAdjustment] # EQ adjustments for Track A (usually empty)
    frequency_selective_timeline: Dict[EQBand, float]  # Timeline for freq-selective crossfade
    processing_confidence: float            # Confidence in EQ solution [0,1]
    before_after_comparison: Dict           # Expected spectral changes
    crossfade_strategy_override: Optional[str]  # Special crossfade instructions


class SpectralMatchingEQ(nn.Module):
    """
    Implement spectral matching EQ for smooth crossfade transitions.
    
    Uses established audio engineering techniques for frequency content
    matching and supports frequency-selective crossfading strategies.
    """
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 n_fft: int = 4096,
                 hop_length: int = 512,
                 smoothing_factor: float = 0.8,
                 max_eq_gain_db: float = 6.0):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.smoothing_factor = smoothing_factor
        self.max_eq_gain_db = max_eq_gain_db
        
        # Define frequency bands
        self.freq_bands = {
            EQBand.SUB_BASS: (20, 60),
            EQBand.BASS: (60, 200), 
            EQBand.LOW_MID: (200, 500),
            EQBand.MID: (500, 2000),
            EQBand.HIGH_MID: (2000, 6000),
            EQBand.HIGH: (6000, 20000)
        }
        
        # Create frequency array
        self.freqs = np.fft.fftfreq(n_fft, 1/sample_rate)[:n_fft//2 + 1]
        
        # Create frequency band indices
        self.band_indices = {}
        for band, (low, high) in self.freq_bands.items():
            indices = np.where((self.freqs >= low) & (self.freqs <= high))[0]
            self.band_indices[band] = indices
        
        # Neural network for EQ strategy selection
        self.strategy_selector = nn.Sequential(
            nn.Linear(12, 32),  # 6 bands x 2 tracks
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 5)    # 5 strategy options
        )
        
        # EQ parameter optimizer
        self.eq_optimizer = nn.Sequential(
            nn.Linear(8, 64),   # Band differences + context
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)    # gain, Q, type
        )

    def forward(self, 
                energy_profile_a: EnergyProfile,
                energy_profile_b: EnergyProfile,
                crossfade_start_a: int,
                crossfade_start_b: int,
                crossfade_duration: int,
                low_end_conflict: Optional[LowEndConflict] = None) -> SpectralMatch:
        """
        Analyze spectral content and generate EQ matching recommendations.
        
        Args:
            energy_profile_a: Track A energy analysis
            energy_profile_b: Track B energy analysis  
            crossfade_start_a: Crossfade start in A (samples)
            crossfade_start_b: Crossfade start in B (samples)
            crossfade_duration: Crossfade duration (samples)
            low_end_conflict: Low-end conflict analysis (optional)
            
        Returns:
            SpectralMatch with EQ recommendations and strategy
        """
        
        # Extract spectral profiles at crossfade points
        spectral_profile_a = self._extract_spectral_profile(
            energy_profile_a, crossfade_start_a, crossfade_duration
        )
        
        spectral_profile_b = self._extract_spectral_profile(
            energy_profile_b, crossfade_start_b, crossfade_duration
        )
        
        # Calculate spectral compatibility
        compatibility_score = self._calculate_spectral_compatibility(
            spectral_profile_a, spectral_profile_b
        )
        
        # Determine EQ strategy
        strategy = self._determine_eq_strategy(
            spectral_profile_a, spectral_profile_b, compatibility_score, low_end_conflict
        )
        
        # Generate EQ adjustments
        track_b_adjustments, track_a_adjustments = self._generate_eq_adjustments(
            spectral_profile_a, spectral_profile_b, strategy, low_end_conflict
        )
        
        # Create frequency-selective crossfade timeline
        freq_selective_timeline = self._create_frequency_timeline(
            strategy, low_end_conflict, compatibility_score
        )
        
        # Calculate processing confidence
        processing_confidence = self._calculate_processing_confidence(
            compatibility_score, strategy, track_b_adjustments
        )
        
        # Generate before/after comparison
        before_after = self._generate_before_after_comparison(
            spectral_profile_a, spectral_profile_b, track_b_adjustments
        )
        
        # Check for crossfade strategy overrides
        crossfade_override = self._determine_crossfade_override(
            low_end_conflict, compatibility_score, strategy
        )
        
        return SpectralMatch(
            compatibility_score=float(compatibility_score),
            strategy=strategy,
            track_b_adjustments=track_b_adjustments,
            track_a_adjustments=track_a_adjustments,
            frequency_selective_timeline=freq_selective_timeline,
            processing_confidence=float(processing_confidence),
            before_after_comparison=before_after,
            crossfade_strategy_override=crossfade_override
        )
    
    def _extract_spectral_profile(self, 
                                 energy_profile: EnergyProfile,
                                 start_sample: int,
                                 duration_samples: int) -> Dict[EQBand, float]:
        """Extract averaged spectral profile for crossfade region"""
        
        # Convert sample positions to frame positions
        start_frame = int(start_sample / self.hop_length)
        duration_frames = int(duration_samples / self.hop_length)
        end_frame = start_frame + duration_frames
        
        # Clamp to available frames
        start_frame = max(0, min(start_frame, energy_profile.spectral_bands.shape[1] - 1))
        end_frame = max(start_frame + 1, min(end_frame, energy_profile.spectral_bands.shape[1]))
        
        # Extract region from spectral bands (already has low/mid/high)
        # Map to our more detailed frequency bands
        region_spectral = energy_profile.spectral_bands[:, start_frame:end_frame]
        avg_spectral = torch.mean(region_spectral, dim=1)
        
        # Map from 3-band to 6-band approximation
        # Note: This is a simplified mapping - real implementation would need full FFT analysis
        spectral_profile = {
            EQBand.SUB_BASS: float(avg_spectral[0]) * 0.6,      # Lower part of low band
            EQBand.BASS: float(avg_spectral[0]) * 0.4,          # Upper part of low band  
            EQBand.LOW_MID: float(avg_spectral[1]) * 0.4,       # Lower part of mid band
            EQBand.MID: float(avg_spectral[1]) * 0.6,           # Upper part of mid band
            EQBand.HIGH_MID: float(avg_spectral[2]) * 0.6,      # Lower part of high band
            EQBand.HIGH: float(avg_spectral[2]) * 0.4           # Upper part of high band
        }
        
        return spectral_profile
    
    def _calculate_spectral_compatibility(self, 
                                        profile_a: Dict[EQBand, float],
                                        profile_b: Dict[EQBand, float]) -> float:
        """Calculate overall spectral compatibility score"""
        
        total_difference = 0.0
        weights = {
            EQBand.SUB_BASS: 0.2,      # Sub-bass important but not dominant
            EQBand.BASS: 0.25,         # Bass very important for crossfades
            EQBand.LOW_MID: 0.2,       # Low-mid important for warmth
            EQBand.MID: 0.15,          # Mid-range moderately important
            EQBand.HIGH_MID: 0.1,      # High-mid less critical in crossfades
            EQBand.HIGH: 0.1           # High frequencies least critical
        }
        
        for band in EQBand:
            # Calculate dB difference
            level_a = profile_a[band]
            level_b = profile_b[band]
            
            # Convert to approximate dB (energy profiles are already in dB)
            difference = abs(level_a - level_b)
            weighted_diff = difference * weights[band]
            total_difference += weighted_diff
        
        # Convert to compatibility score (0 = incompatible, 1 = perfect match)
        # Assume 30dB total difference is completely incompatible
        compatibility = max(0.0, 1.0 - total_difference / 30.0)
        
        return compatibility
    
    def _determine_eq_strategy(self, 
                              profile_a: Dict[EQBand, float],
                              profile_b: Dict[EQBand, float],
                              compatibility_score: float,
                              low_end_conflict: Optional[LowEndConflict]) -> EQStrategy:
        """Determine optimal EQ strategy"""
        
        # If already highly compatible, no EQ needed
        if compatibility_score > 0.85:
            return EQStrategy.NO_EQ_NEEDED
        
        # If severe low-end conflicts, use frequency-selective approach
        if (low_end_conflict and 
            low_end_conflict.severity in [ConflictSeverity.SEVERE, ConflictSeverity.CRITICAL]):
            return EQStrategy.FREQUENCY_SELECTIVE
        
        # Use neural network for strategy selection
        features = []
        for band in EQBand:
            features.append(profile_a[band])
            features.append(profile_b[band])
        
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        try:
            with torch.no_grad():
                strategy_logits = self.strategy_selector(features_tensor.unsqueeze(0))
                strategy_probs = F.softmax(strategy_logits, dim=1).squeeze()
                
                strategies = [
                    EQStrategy.MATCH_B_TO_A,        # Preferred (Track A sacred)
                    EQStrategy.FREQUENCY_SELECTIVE,
                    EQStrategy.BIDIRECTIONAL,
                    EQStrategy.MATCH_A_TO_B,        # Less preferred
                    EQStrategy.NO_EQ_NEEDED
                ]
                
                return strategies[torch.argmax(strategy_probs)]
                
        except:
            # Fallback to heuristic selection
            # Default to matching B to A (Track A sacred principle)
            if compatibility_score < 0.4:
                return EQStrategy.FREQUENCY_SELECTIVE
            else:
                return EQStrategy.MATCH_B_TO_A
    
    def _generate_eq_adjustments(self, 
                                profile_a: Dict[EQBand, float],
                                profile_b: Dict[EQBand, float],
                                strategy: EQStrategy,
                                low_end_conflict: Optional[LowEndConflict]) -> Tuple[List[EQAdjustment], List[EQAdjustment]]:
        """Generate specific EQ adjustments"""
        
        track_b_adjustments = []
        track_a_adjustments = []  # Usually empty (Track A sacred)
        
        if strategy == EQStrategy.NO_EQ_NEEDED:
            return track_b_adjustments, track_a_adjustments
        
        # Calculate required adjustments for each band
        for band in EQBand:
            level_a = profile_a[band]
            level_b = profile_b[band]
            difference = level_a - level_b  # Positive = A louder, B needs boost
            
            # Skip small differences
            if abs(difference) < 1.0:  # Less than 1dB difference
                continue
            
            # Determine if this band should be adjusted based on strategy
            should_adjust = self._should_adjust_band(band, strategy, low_end_conflict)
            
            if should_adjust:
                # Generate EQ adjustment for Track B
                eq_adjustment = self._create_eq_adjustment(
                    band, difference, strategy, low_end_conflict
                )
                
                if eq_adjustment:
                    track_b_adjustments.append(eq_adjustment)
        
        return track_b_adjustments, track_a_adjustments
    
    def _should_adjust_band(self, 
                           band: EQBand,
                           strategy: EQStrategy,
                           low_end_conflict: Optional[LowEndConflict]) -> bool:
        """Determine if a frequency band should be adjusted"""
        
        if strategy == EQStrategy.NO_EQ_NEEDED:
            return False
        
        if strategy == EQStrategy.FREQUENCY_SELECTIVE:
            # Only adjust problematic bands
            if low_end_conflict:
                # Prioritize low-end if there are conflicts
                return band in [EQBand.SUB_BASS, EQBand.BASS, EQBand.LOW_MID]
            else:
                # Adjust all bands
                return True
        
        # For other strategies, adjust all bands
        return True
    
    def _create_eq_adjustment(self, 
                             band: EQBand,
                             level_difference: float,
                             strategy: EQStrategy,
                             low_end_conflict: Optional[LowEndConflict]) -> Optional[EQAdjustment]:
        """Create specific EQ adjustment for a band"""
        
        # Get band center frequency
        freq_range = self.freq_bands[band]
        center_freq = (freq_range[0] + freq_range[1]) / 2
        
        # Calculate gain (clamp to maximum)
        gain_db = max(-self.max_eq_gain_db, min(self.max_eq_gain_db, level_difference))
        
        # Skip very small adjustments
        if abs(gain_db) < 0.5:
            return None
        
        # Determine filter type and Q based on band and strategy
        if band in [EQBand.SUB_BASS, EQBand.BASS]:
            # Low frequencies: shelf filters or bell with lower Q
            if abs(gain_db) > 3.0:
                filter_type = "low_shelf"
                q_factor = 0.7
            else:
                filter_type = "bell"
                q_factor = 1.0
                
        elif band in [EQBand.HIGH_MID, EQBand.HIGH]:
            # High frequencies: shelf filters or bell with lower Q
            if abs(gain_db) > 3.0:
                filter_type = "high_shelf"
                q_factor = 0.7
            else:
                filter_type = "bell"
                q_factor = 1.0
                
        else:
            # Mid frequencies: bell filters with moderate Q
            filter_type = "bell"
            q_factor = 1.4
        
        # Adjust Q based on conflict severity
        if low_end_conflict and low_end_conflict.severity == ConflictSeverity.CRITICAL:
            q_factor *= 0.7  # Wider filters for severe conflicts
        
        return EQAdjustment(
            frequency_hz=center_freq,
            gain_db=gain_db,
            q_factor=q_factor,
            filter_type=filter_type
        )
    
    def _create_frequency_timeline(self, 
                                  strategy: EQStrategy,
                                  low_end_conflict: Optional[LowEndConflict],
                                  compatibility_score: float) -> Dict[EQBand, float]:
        """Create timeline for frequency-selective crossfading"""
        
        # Default timeline (all bands crossfade simultaneously)
        timeline = {band: 0.5 for band in EQBand}
        
        if strategy == EQStrategy.FREQUENCY_SELECTIVE or (
            low_end_conflict and low_end_conflict.severity in [ConflictSeverity.MODERATE, ConflictSeverity.SEVERE]
        ):
            # Implement "first lows then highs" approach
            timeline = {
                EQBand.SUB_BASS: 0.2,    # Start crossfade early (20% into crossfade)
                EQBand.BASS: 0.3,        # Bass follows shortly after (30%)
                EQBand.LOW_MID: 0.4,     # Low-mid next (40%)
                EQBand.MID: 0.5,         # Mid-range at center (50%)
                EQBand.HIGH_MID: 0.6,    # High-mid later (60%)
                EQBand.HIGH: 0.7         # High frequencies last (70%)
            }
        
        return timeline
    
    def _calculate_processing_confidence(self, 
                                       compatibility_score: float,
                                       strategy: EQStrategy,
                                       adjustments: List[EQAdjustment]) -> float:
        """Calculate confidence in EQ processing"""
        
        # Base confidence from compatibility
        base_confidence = compatibility_score
        
        # Strategy confidence
        strategy_confidence = {
            EQStrategy.NO_EQ_NEEDED: 0.95,
            EQStrategy.MATCH_B_TO_A: 0.85,
            EQStrategy.FREQUENCY_SELECTIVE: 0.80,
            EQStrategy.BIDIRECTIONAL: 0.70,
            EQStrategy.MATCH_A_TO_B: 0.75
        }[strategy]
        
        # Adjustment complexity penalty
        num_adjustments = len(adjustments)
        complexity_penalty = min(0.2, num_adjustments * 0.03)
        
        # Large gain penalty
        max_gain = max([abs(adj.gain_db) for adj in adjustments] + [0])
        gain_penalty = min(0.3, max_gain / self.max_eq_gain_db * 0.2)
        
        confidence = (
            base_confidence * 0.4 +
            strategy_confidence * 0.4 +
            (1.0 - complexity_penalty - gain_penalty) * 0.2
        )
        
        return max(0.0, min(1.0, confidence))
    
    def _generate_before_after_comparison(self, 
                                        profile_a: Dict[EQBand, float],
                                        profile_b: Dict[EQBand, float],
                                        adjustments: List[EQAdjustment]) -> Dict:
        """Generate before/after spectral comparison"""
        
        # Calculate expected profile after EQ
        adjusted_profile_b = profile_b.copy()
        
        for adjustment in adjustments:
            # Find which band this adjustment affects
            for band, (low, high) in self.freq_bands.items():
                if low <= adjustment.frequency_hz <= high:
                    adjusted_profile_b[band] += adjustment.gain_db
                    break
        
        # Calculate improvement metrics
        before_compatibility = self._calculate_spectral_compatibility(profile_a, profile_b)
        after_compatibility = self._calculate_spectral_compatibility(profile_a, adjusted_profile_b)
        improvement = after_compatibility - before_compatibility
        
        return {
            'before_compatibility': float(before_compatibility),
            'after_compatibility': float(after_compatibility),
            'improvement': float(improvement),
            'original_profile_b': profile_b,
            'adjusted_profile_b': adjusted_profile_b
        }
    
    def _determine_crossfade_override(self, 
                                    low_end_conflict: Optional[LowEndConflict],
                                    compatibility_score: float,
                                    strategy: EQStrategy) -> Optional[str]:
        """Determine if crossfade strategy needs override"""
        
        if strategy == EQStrategy.FREQUENCY_SELECTIVE:
            return "use_frequency_selective_timeline"
        
        if (low_end_conflict and 
            low_end_conflict.severity in [ConflictSeverity.SEVERE, ConflictSeverity.CRITICAL]):
            return "prioritize_low_frequency_crossfade"
        
        if compatibility_score < 0.3:
            return "use_longer_crossfade_duration"
        
        return None
    
    def apply_eq_to_audio(self, 
                         audio: torch.Tensor,
                         adjustments: List[EQAdjustment],
                         sample_rate: int) -> torch.Tensor:
        """Apply EQ adjustments to audio (simplified implementation)"""
        
        if not adjustments:
            return audio
        
        # This is a simplified EQ implementation
        # Real implementation would use proper biquad filters
        processed_audio = audio.clone()
        
        for adjustment in adjustments:
            # Simple gain adjustment (not proper EQ, just for demonstration)
            # Real implementation would use biquad filters or FFT-based processing
            if adjustment.filter_type in ["low_shelf", "high_shelf"]:
                # Apply shelf filter (simplified)
                processed_audio = processed_audio * (10 ** (adjustment.gain_db / 20))
            else:
                # Apply bell filter (simplified)
                processed_audio = processed_audio * (10 ** (adjustment.gain_db / 40))
        
        return processed_audio
    
    def get_eq_summary(self, spectral_match: SpectralMatch) -> str:
        """Generate human-readable EQ summary"""
        
        if spectral_match.strategy == EQStrategy.NO_EQ_NEEDED:
            return "No EQ adjustments needed - spectral content already compatible"
        
        summary = f"EQ Strategy: {spectral_match.strategy.value}\n"
        summary += f"Compatibility: {spectral_match.compatibility_score:.2f} -> {spectral_match.before_after_comparison['after_compatibility']:.2f}\n"
        
        if spectral_match.track_b_adjustments:
            summary += "Track B Adjustments:\n"
            for adj in spectral_match.track_b_adjustments:
                summary += f"  {adj.frequency_hz:.0f}Hz: {adj.gain_db:+.1f}dB ({adj.filter_type})\n"
        
        if spectral_match.crossfade_strategy_override:
            summary += f"Crossfade Override: {spectral_match.crossfade_strategy_override}\n"
        
        return summary


if __name__ == "__main__":
    # Test the module
    from .energy_profile_extractor import EnergyProfileExtractor, create_test_audio_with_energy_pattern
    from .low_end_conflict_analyzer import LowEndConflictAnalyzer
    
    # Create test audio with different spectral content
    audio_bright = create_test_audio_with_energy_pattern(10.0, 'stable') + 0.5 * torch.randn(int(10*44100))
    audio_warm = create_test_audio_with_energy_pattern(10.0, 'stable') * 0.7  # Less high-freq content
    
    # Extract energy profiles
    extractor = EnergyProfileExtractor()
    profile_bright = extractor(audio_bright)
    profile_warm = extractor(audio_warm)
    
    # Analyze low-end conflicts
    conflict_analyzer = LowEndConflictAnalyzer()
    low_end_conflict = conflict_analyzer(audio_bright, audio_warm, 0, 0, len(audio_bright))
    
    # Test spectral matching
    eq_matcher = SpectralMatchingEQ()
    
    spectral_match = eq_matcher(
        profile_bright, profile_warm, 
        5*44100, 2*44100, 2*44100,  # 5s to 2s, 2s duration
        low_end_conflict
    )
    
    print("=== Spectral Matching EQ ===")
    print(f"Compatibility: {spectral_match.compatibility_score:.3f}")
    print(f"Strategy: {spectral_match.strategy.value}")
    print(f"Track B adjustments: {len(spectral_match.track_b_adjustments)}")
    print(f"Processing confidence: {spectral_match.processing_confidence:.3f}")
    
    if spectral_match.track_b_adjustments:
        print("EQ Adjustments:")
        for adj in spectral_match.track_b_adjustments:
            print(f"  {adj.frequency_hz:.0f}Hz: {adj.gain_db:+.1f}dB ({adj.filter_type}, Q={adj.q_factor:.1f})")
    
    # Test summary
    summary = eq_matcher.get_eq_summary(spectral_match)
    print(f"\nSummary:\n{summary}")