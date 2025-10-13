"""
CrossfadeEnvelopeDesigner - Design optimal crossfade curves

This module creates crossfade envelopes optimized for the musical content,
supporting equal-power, frequency-selective, and content-adaptive curves.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, List, Tuple, Dict, Union, Any
from dataclasses import dataclass
from enum import Enum

from .energy_profile_extractor import EnergyProfile
from .processing_parameter_calculator import ProcessingParams
from .energy_compatibility_analyzer import EnergyMatch


class CrossfadeCurveType(Enum):
    """Crossfade curve types"""
    EQUAL_POWER = "equal_power"
    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"
    S_CURVE = "s_curve"
    FREQUENCY_SELECTIVE = "frequency_selective"
    CONTENT_ADAPTIVE = "content_adaptive"


class FrequencyBand(Enum):
    """Frequency bands for selective crossfading"""
    LOW = "low"      # 0-200 Hz
    MID = "mid"      # 200-2000 Hz  
    HIGH = "high"    # 2000+ Hz


@dataclass
class CrossfadeEnvelope:
    """Crossfade envelope specification"""
    curve_type: CrossfadeCurveType     # Type of crossfade curve
    duration_samples: int              # Length in samples
    track_a_curve: torch.Tensor        # Fade-out curve for A [duration_samples]
    track_b_curve: torch.Tensor        # Fade-in curve for B [duration_samples]
    frequency_weights: Optional[Dict[FrequencyBand, torch.Tensor]]  # Per-band curves
    sample_rate: int                   # Sample rate for timing
    musical_bars: float               # Duration in musical bars
    confidence: float                 # Confidence in envelope design [0,1]


class CrossfadeEnvelopeDesigner(nn.Module):
    """
    Design optimal crossfade curves based on musical content analysis.
    
    Creates envelopes that match the energy characteristics and spectral
    content of the tracks for smooth, natural transitions.
    """
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 default_duration_ms: float = 2000.0,
                 min_duration_ms: float = 500.0,
                 max_duration_ms: float = 8000.0):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.default_duration_samples = int(default_duration_ms * sample_rate / 1000)
        self.min_duration_samples = int(min_duration_ms * sample_rate / 1000)
        self.max_duration_samples = int(max_duration_ms * sample_rate / 1000)
        
        # Neural network for duration optimization
        self.duration_optimizer = nn.Sequential(
            nn.Linear(8, 32),  # Energy and compatibility features
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Curve shape optimizer
        self.curve_optimizer = nn.Sequential(
            nn.Linear(6, 32),  # Musical and energy features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4)   # Parameters for curve shaping
        )
        
        # Frequency-selective crossfade predictor
        self.frequency_selector = nn.Sequential(
            nn.Linear(9, 32),  # Spectral features
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),  # Low/Mid/High band weights
            nn.Softmax(dim=1)
        )

    def forward(self, 
                energy_profile_a: EnergyProfile,
                energy_profile_b: EnergyProfile,
                processing_params: ProcessingParams,
                energy_match: EnergyMatch,
                exit_sample_a: int,
                entry_sample_b: int,
                bpm: float = 120.0) -> CrossfadeEnvelope:
        """
        Design optimal crossfade envelope for the tracks.
        
        Args:
            energy_profile_a: Track A energy analysis
            energy_profile_b: Track B energy analysis
            processing_params: B processing parameters
            energy_match: Energy compatibility analysis
            exit_sample_a: A exit sample position
            entry_sample_b: B entry sample position
            bpm: Tempo for musical timing calculations
            
        Returns:
            CrossfadeEnvelope with optimized curves
        """
        
        # Determine optimal crossfade curve type
        curve_type = self._determine_curve_type(
            energy_profile_a, energy_profile_b, processing_params, energy_match
        )
        
        # Calculate optimal duration
        duration_samples = self._calculate_optimal_duration(
            energy_profile_a, energy_profile_b, energy_match, bpm
        )
        
        # Generate base crossfade curves
        if curve_type == CrossfadeCurveType.FREQUENCY_SELECTIVE:
            envelope = self._design_frequency_selective_crossfade(
                energy_profile_a, energy_profile_b, duration_samples,
                exit_sample_a, entry_sample_b
            )
        else:
            envelope = self._design_standard_crossfade(
                curve_type, duration_samples, energy_profile_a, energy_profile_b,
                exit_sample_a, entry_sample_b
            )
        
        # Calculate musical timing
        musical_bars = duration_samples * bpm / (60.0 * self.sample_rate * 4)  # Assume 4/4 time
        
        # Calculate confidence
        confidence = self._calculate_envelope_confidence(
            envelope, energy_match, processing_params
        )
        
        return CrossfadeEnvelope(
            curve_type=curve_type,
            duration_samples=duration_samples,
            track_a_curve=envelope['track_a'],
            track_b_curve=envelope['track_b'],
            frequency_weights=envelope.get('frequency_weights'),
            sample_rate=self.sample_rate,
            musical_bars=musical_bars,
            confidence=confidence
        )
    
    def _determine_curve_type(self, 
                            energy_profile_a: EnergyProfile,
                            energy_profile_b: EnergyProfile,
                            processing_params: ProcessingParams,
                            energy_match: EnergyMatch) -> CrossfadeCurveType:
        """Determine optimal crossfade curve type"""
        
        # Check if frequency-selective is needed
        if energy_match.spectral_balance_score < 0.6:
            return CrossfadeCurveType.FREQUENCY_SELECTIVE
        
        # Check energy characteristics
        dynamic_range_a = energy_profile_a.dynamics['dynamic_range_db']
        dynamic_range_b = energy_profile_b.dynamics['dynamic_range_db']
        
        # For high dynamic range content, use content-adaptive
        if dynamic_range_a > 20 or dynamic_range_b > 20:
            return CrossfadeCurveType.CONTENT_ADAPTIVE
        
        # For well-matched energy, use equal-power (standard)
        if energy_match.compatibility_score > 0.7:
            return CrossfadeCurveType.EQUAL_POWER
        
        # For mismatched energy, use S-curve for smoother transition
        if abs(energy_match.level_difference_db) > 6.0:
            return CrossfadeCurveType.S_CURVE
        
        # Default to equal-power
        return CrossfadeCurveType.EQUAL_POWER
    
    def _calculate_optimal_duration(self, 
                                  energy_profile_a: EnergyProfile,
                                  energy_profile_b: EnergyProfile,
                                  energy_match: EnergyMatch,
                                  bpm: float) -> int:
        """Calculate optimal crossfade duration"""
        
        # Base duration from tempo (prefer musical bars)
        beat_duration_ms = 60000 / bpm  # ms per beat
        bar_duration_ms = beat_duration_ms * 4  # 4/4 time
        
        # Prefer power-of-2 bar durations: 1, 2, 4, 8 bars
        bar_options = [1, 2, 4, 8]
        
        # Features for neural network optimization
        features = torch.tensor([
            energy_match.compatibility_score,
            abs(energy_match.level_difference_db) / 12.0,
            energy_match.spectral_balance_score,
            energy_match.energy_flow_score,
            energy_profile_a.dynamics['dynamic_range_db'] / 30.0,
            energy_profile_b.dynamics['dynamic_range_db'] / 30.0,
            bpm / 140.0,  # Normalize around common tempo
            energy_match.dynamic_range_compatibility
        ], dtype=torch.float32)
        
        try:
            with torch.no_grad():
                duration_factor = self.duration_optimizer(features.unsqueeze(0))
                duration_factor = float(duration_factor.squeeze())
        except:
            # Fallback heuristic
            if energy_match.compatibility_score > 0.8:
                duration_factor = 0.3  # Shorter for good matches
            elif energy_match.compatibility_score > 0.5:
                duration_factor = 0.5  # Medium duration
            else:
                duration_factor = 0.8  # Longer for difficult matches
        
        # Map to bar duration
        bar_index = int(duration_factor * len(bar_options))
        bar_index = min(bar_index, len(bar_options) - 1)
        
        selected_bars = bar_options[bar_index]
        duration_ms = selected_bars * bar_duration_ms
        
        # Convert to samples and clamp to limits
        duration_samples = int(duration_ms * self.sample_rate / 1000)
        duration_samples = max(self.min_duration_samples, 
                             min(self.max_duration_samples, duration_samples))
        
        return duration_samples
    
    def _design_standard_crossfade(self, 
                                 curve_type: CrossfadeCurveType,
                                 duration_samples: int,
                                 energy_profile_a: EnergyProfile,
                                 energy_profile_b: EnergyProfile,
                                 exit_sample_a: int,
                                 entry_sample_b: int) -> Dict[str, torch.Tensor]:
        """Design standard crossfade curves"""
        
        t = torch.linspace(0, 1, duration_samples)
        
        if curve_type == CrossfadeCurveType.EQUAL_POWER:
            # Equal power crossfade (constant power sum)
            track_a_curve = torch.cos(t * math.pi / 2)
            track_b_curve = torch.sin(t * math.pi / 2)
            
        elif curve_type == CrossfadeCurveType.LINEAR:
            # Linear crossfade
            track_a_curve = 1.0 - t
            track_b_curve = t
            
        elif curve_type == CrossfadeCurveType.LOGARITHMIC:
            # Logarithmic fade (more natural for some content)
            track_a_curve = torch.pow(1.0 - t, 2)
            track_b_curve = torch.pow(t, 2)
            
        elif curve_type == CrossfadeCurveType.S_CURVE:
            # S-curve (smooth acceleration/deceleration)
            track_a_curve = 0.5 * (1 + torch.cos(t * math.pi))
            track_b_curve = 0.5 * (1 - torch.cos(t * math.pi))
            
        elif curve_type == CrossfadeCurveType.CONTENT_ADAPTIVE:
            # Adaptive curve based on energy content
            track_a_curve, track_b_curve = self._design_content_adaptive_curves(
                duration_samples, energy_profile_a, energy_profile_b,
                exit_sample_a, entry_sample_b
            )
            
        else:
            # Fallback to equal power
            track_a_curve = torch.cos(t * math.pi / 2)
            track_b_curve = torch.sin(t * math.pi / 2)
        
        return {
            'track_a': track_a_curve,
            'track_b': track_b_curve
        }
    
    def _design_frequency_selective_crossfade(self, 
                                            energy_profile_a: EnergyProfile,
                                            energy_profile_b: EnergyProfile,
                                            duration_samples: int,
                                            exit_sample_a: int,
                                            entry_sample_b: int) -> Dict[str, Union[torch.Tensor, Dict]]:
        """Design frequency-selective crossfade curves"""
        
        # Get spectral information at crossfade points
        frame_a = int(exit_sample_a // (self.sample_rate / len(energy_profile_a.rms_curve)))
        frame_b = int(entry_sample_b // (self.sample_rate / len(energy_profile_b.rms_curve)))
        
        frame_a = max(0, min(frame_a, energy_profile_a.spectral_bands.shape[1] - 1))
        frame_b = max(0, min(frame_b, energy_profile_b.spectral_bands.shape[1] - 1))
        
        # Extract spectral features
        spectral_a = energy_profile_a.spectral_bands[:, frame_a]
        spectral_b = energy_profile_b.spectral_bands[:, frame_b]
        
        # Features for frequency-selective decision
        features = torch.tensor([
            spectral_a[0], spectral_b[0],  # Low band
            spectral_a[1], spectral_b[1],  # Mid band  
            spectral_a[2], spectral_b[2],  # High band
            torch.mean(spectral_a),
            torch.mean(spectral_b),
            torch.std(spectral_a - spectral_b)
        ], dtype=torch.float32)
        
        # Normalize features (rough)
        features = features / 60.0  # dB normalization
        
        try:
            with torch.no_grad():
                band_weights = self.frequency_selector(features.unsqueeze(0))
                band_weights = band_weights.squeeze()
        except:
            # Fallback: emphasize low frequencies
            band_weights = torch.tensor([0.5, 0.3, 0.2])
        
        # Create base equal-power curves
        t = torch.linspace(0, 1, duration_samples)
        base_a_curve = torch.cos(t * math.pi / 2)
        base_b_curve = torch.sin(t * math.pi / 2)
        
        # Create frequency-specific curves
        frequency_weights = {}
        
        # Low frequencies: earlier transition (bass is most noticeable)
        low_t = torch.clamp(t * 1.2, 0, 1)  # Faster transition
        frequency_weights[FrequencyBand.LOW] = {
            'track_a': torch.cos(low_t * math.pi / 2),
            'track_b': torch.sin(low_t * math.pi / 2)
        }
        
        # Mid frequencies: standard transition
        frequency_weights[FrequencyBand.MID] = {
            'track_a': base_a_curve,
            'track_b': base_b_curve
        }
        
        # High frequencies: later transition (less critical)
        high_t = torch.clamp(t * 0.8, 0, 1)  # Slower transition
        frequency_weights[FrequencyBand.HIGH] = {
            'track_a': torch.cos(high_t * math.pi / 2),
            'track_b': torch.sin(high_t * math.pi / 2)
        }
        
        return {
            'track_a': base_a_curve,  # Full-spectrum fallback
            'track_b': base_b_curve,
            'frequency_weights': frequency_weights
        }
    
    def _design_content_adaptive_curves(self, 
                                      duration_samples: int,
                                      energy_profile_a: EnergyProfile,
                                      energy_profile_b: EnergyProfile,
                                      exit_sample_a: int,
                                      entry_sample_b: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Design content-adaptive crossfade curves"""
        
        # Get energy context around crossfade points
        fade_duration_frames = duration_samples // (self.sample_rate // len(energy_profile_a.rms_curve))
        
        start_frame_a = max(0, int(exit_sample_a // (self.sample_rate / len(energy_profile_a.rms_curve))))
        end_frame_a = min(len(energy_profile_a.rms_curve), start_frame_a + fade_duration_frames)
        
        start_frame_b = max(0, int(entry_sample_b // (self.sample_rate / len(energy_profile_b.rms_curve))))
        end_frame_b = min(len(energy_profile_b.rms_curve), start_frame_b + fade_duration_frames)
        
        # Extract energy curves during crossfade
        energy_curve_a = energy_profile_a.rms_curve[start_frame_a:end_frame_a]
        energy_curve_b = energy_profile_b.rms_curve[start_frame_b:end_frame_b]
        
        # Align lengths
        min_len = min(len(energy_curve_a), len(energy_curve_b), duration_samples // 10)
        if min_len > 0:
            energy_curve_a = energy_curve_a[:min_len]
            energy_curve_b = energy_curve_b[:min_len]
            
            # Interpolate to crossfade duration
            energy_curve_a = F.interpolate(
                energy_curve_a.unsqueeze(0).unsqueeze(0),
                size=duration_samples, mode='linear', align_corners=True
            ).squeeze()
            
            energy_curve_b = F.interpolate(
                energy_curve_b.unsqueeze(0).unsqueeze(0), 
                size=duration_samples, mode='linear', align_corners=True
            ).squeeze()
            
            # Create adaptive curves based on energy content
            t = torch.linspace(0, 1, duration_samples)
            
            # Normalize energy curves to [0, 1] range
            energy_a_norm = (energy_curve_a - energy_curve_a.min()) / (energy_curve_a.max() - energy_curve_a.min() + 1e-8)
            energy_b_norm = (energy_curve_b - energy_curve_b.min()) / (energy_curve_b.max() - energy_curve_b.min() + 1e-8)
            
            # Use energy content to modulate crossfade curves
            base_a_curve = torch.cos(t * math.pi / 2)
            base_b_curve = torch.sin(t * math.pi / 2)
            
            # Modulate based on energy content (subtle effect)
            track_a_curve = base_a_curve * (0.8 + 0.2 * energy_a_norm)
            track_b_curve = base_b_curve * (0.8 + 0.2 * energy_b_norm)
            
            # Renormalize to maintain power
            total_power = track_a_curve**2 + track_b_curve**2
            track_a_curve = track_a_curve / torch.sqrt(total_power)
            track_b_curve = track_b_curve / torch.sqrt(total_power)
            
        else:
            # Fallback to equal power
            t = torch.linspace(0, 1, duration_samples)
            track_a_curve = torch.cos(t * math.pi / 2)
            track_b_curve = torch.sin(t * math.pi / 2)
        
        return track_a_curve, track_b_curve
    
    def _calculate_envelope_confidence(self, 
                                     envelope: Dict,
                                     energy_match: EnergyMatch,
                                     processing_params: ProcessingParams) -> float:
        """Calculate confidence in envelope design"""
        
        # Base confidence from energy compatibility
        base_confidence = energy_match.compatibility_score
        
        # Adjust for processing complexity
        if processing_params.strategy.value == "no_processing":
            processing_confidence = 1.0
        elif processing_params.strategy.value == "hard_cut":
            processing_confidence = 0.9
        else:
            processing_confidence = processing_params.processing_confidence
        
        # Adjust for envelope complexity
        if 'frequency_weights' in envelope:
            complexity_penalty = 0.1  # Slight penalty for complexity
        else:
            complexity_penalty = 0.0
        
        # Combined confidence
        confidence = (
            base_confidence * 0.5 +
            processing_confidence * 0.4 +
            energy_match.energy_flow_score * 0.1 -
            complexity_penalty
        )
        
        return max(0.0, min(1.0, confidence))
    
    def create_envelope_for_hard_cut(self, 
                                   duration_samples: int,
                                   bpm: float = 120.0) -> CrossfadeEnvelope:
        """Create envelope for hard cut (no crossfade)"""
        
        # Minimal crossfade for click removal
        minimal_duration = min(duration_samples, int(0.01 * self.sample_rate))  # 10ms max
        
        t = torch.linspace(0, 1, minimal_duration)
        track_a_curve = 1.0 - t
        track_b_curve = t
        
        musical_bars = minimal_duration * bpm / (60.0 * self.sample_rate * 4)
        
        return CrossfadeEnvelope(
            curve_type=CrossfadeCurveType.LINEAR,
            duration_samples=minimal_duration,
            track_a_curve=track_a_curve,
            track_b_curve=track_b_curve,
            frequency_weights=None,
            sample_rate=self.sample_rate,
            musical_bars=musical_bars,
            confidence=0.95  # High confidence - minimal processing
        )
    
    def apply_envelope_to_audio(self, 
                              envelope: CrossfadeEnvelope,
                              audio_a: torch.Tensor,
                              audio_b: torch.Tensor,
                              start_pos_a: int,
                              start_pos_b: int) -> torch.Tensor:
        """Apply crossfade envelope to audio segments"""
        
        if audio_a.dim() == 1:
            audio_a = audio_a.unsqueeze(0)
        if audio_b.dim() == 1:
            audio_b = audio_b.unsqueeze(0)
        
        channels = max(audio_a.size(0), audio_b.size(0))
        duration = envelope.duration_samples
        
        # Extract audio segments for crossfade
        seg_a = audio_a[:, start_pos_a:start_pos_a + duration]
        seg_b = audio_b[:, start_pos_b:start_pos_b + duration]
        
        # Pad if necessary
        if seg_a.size(1) < duration:
            pad_a = torch.zeros(channels, duration - seg_a.size(1))
            seg_a = torch.cat([seg_a, pad_a], dim=1)
        
        if seg_b.size(1) < duration:
            pad_b = torch.zeros(channels, duration - seg_b.size(1))
            seg_b = torch.cat([seg_b, pad_b], dim=1)
        
        # Apply crossfade curves
        if envelope.frequency_weights is not None:
            # Frequency-selective crossfade (simplified - would need proper filtering)
            crossfaded = seg_a * envelope.track_a_curve + seg_b * envelope.track_b_curve
        else:
            # Standard crossfade
            crossfaded = seg_a * envelope.track_a_curve + seg_b * envelope.track_b_curve
        
        return crossfaded
    
    def get_envelope_summary(self, envelope: CrossfadeEnvelope) -> Dict[str, Any]:
        """Get human-readable envelope summary"""
        
        duration_ms = envelope.duration_samples * 1000 / envelope.sample_rate
        
        return {
            'curve_type': envelope.curve_type.value,
            'duration_ms': round(duration_ms, 1),
            'duration_bars': round(envelope.musical_bars, 2),
            'frequency_selective': envelope.frequency_weights is not None,
            'confidence': round(envelope.confidence, 3),
            'power_sum_check': self._verify_power_conservation(envelope)
        }
    
    def _verify_power_conservation(self, envelope: CrossfadeEnvelope) -> bool:
        """Verify power conservation for equal-power crossfades"""
        
        if envelope.curve_type != CrossfadeCurveType.EQUAL_POWER:
            return True  # Only check equal-power crossfades
        
        power_sum = envelope.track_a_curve**2 + envelope.track_b_curve**2
        power_deviation = torch.abs(power_sum - 1.0).max()
        
        return float(power_deviation) < 0.01  # 1% tolerance


if __name__ == "__main__":
    # Test the module
    from .energy_profile_extractor import EnergyProfileExtractor, create_test_audio_with_energy_pattern
    from .processing_parameter_calculator import ProcessingParams, ProcessingStrategy
    from .energy_compatibility_analyzer import EnergyMatch, EnergyCompatibilityLevel
    
    # Create test energy profiles
    audio_a = create_test_audio_with_energy_pattern(30.0, 'stable')
    audio_b = create_test_audio_with_energy_pattern(30.0, 'buildup')
    
    extractor = EnergyProfileExtractor()
    profile_a = extractor(audio_a)
    profile_b = extractor(audio_b)
    
    # Mock processing parameters and energy match
    processing_params = ProcessingParams(
        pitch_shift_semitones=1.0,
        rate_change_ratio=1.02,
        timing_offset_samples=100,
        gain_adjustment_db=-2.0,
        strategy=ProcessingStrategy.PITCH_AND_TEMPO,
        artifact_prediction=0.2,
        quality_score=0.8,
        processing_confidence=0.85,
        fallback_params=None
    )
    
    energy_match = EnergyMatch(
        compatibility_score=0.75,
        compatibility_level=EnergyCompatibilityLevel.GOOD,
        level_difference_db=-2.0,
        spectral_balance_score=0.8,
        loudness_adjustment_db=-1.5,
        dynamic_range_compatibility=0.7,
        energy_flow_score=0.8,
        processing_recommendation="moderate_level_adjustment_-2.0_db"
    )
    
    # Test envelope designer
    designer = CrossfadeEnvelopeDesigner()
    
    envelope = designer(
        profile_a, profile_b, processing_params, energy_match,
        20*44100, 5*44100, bpm=120.0
    )
    
    print("=== Crossfade Envelope ===")
    print(f"Curve type: {envelope.curve_type.value}")
    print(f"Duration: {envelope.duration_samples} samples ({envelope.duration_samples/44100*1000:.1f} ms)")
    print(f"Musical duration: {envelope.musical_bars:.2f} bars")
    print(f"Confidence: {envelope.confidence:.3f}")
    print(f"Frequency selective: {envelope.frequency_weights is not None}")
    
    # Test envelope summary
    summary = designer.get_envelope_summary(envelope)
    print(f"Summary: {summary}")
    
    # Test hard cut envelope
    hard_cut = designer.create_envelope_for_hard_cut(2*44100, bpm=120.0)
    print(f"\nHard cut envelope: {hard_cut.duration_samples} samples, confidence: {hard_cut.confidence:.3f}")
    
    # Test power conservation
    if envelope.curve_type == CrossfadeCurveType.EQUAL_POWER:
        power_sum = envelope.track_a_curve**2 + envelope.track_b_curve**2
        print(f"Power conservation check: mean={power_sum.mean():.4f}, std={power_sum.std():.4f}")