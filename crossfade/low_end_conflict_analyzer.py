"""
LowEndConflictAnalyzer - Detect bass/kick conflicts using standard spectral analysis

This module analyzes low-frequency content to detect kick drum conflicts and 
bass line handoffs for frequency-selective crossfading strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .energy_profile_extractor import EnergyProfile


class ConflictType(Enum):
    """Types of low-end conflicts"""
    KICK_CLASH = "kick_clash"           # Overlapping kick drums
    BASS_COMPETITION = "bass_competition"  # Competing bass lines
    PHASE_CANCELLATION = "phase_cancellation"  # Phase issues in low end
    MUDDY_BUILDUP = "muddy_buildup"     # Frequency masking/buildup
    NO_CONFLICT = "no_conflict"         # Clean low-end transition


class ConflictSeverity(Enum):
    """Severity levels for conflicts"""
    NONE = "none"           # No significant conflict
    MILD = "mild"           # Slight overlap, manageable
    MODERATE = "moderate"   # Noticeable conflict, needs attention
    SEVERE = "severe"       # Major conflict, requires intervention
    CRITICAL = "critical"   # Unacceptable conflict level


@dataclass
class LowEndConflict:
    """Low-end conflict analysis results"""
    conflict_type: ConflictType        # Type of detected conflict
    severity: ConflictSeverity         # Severity level
    frequency_range_hz: Tuple[float, float]  # Affected frequency range
    conflict_strength: float           # Conflict strength [0,1]
    recommended_crossfade_freq: float  # Suggested crossfade frequency (Hz)
    bass_handoff_timing: Optional[float]  # Optimal bass handoff time (seconds)
    kick_alignment_offset: Optional[float]  # Kick timing adjustment (seconds)
    processing_recommendation: str      # Recommended processing approach


class LowEndConflictAnalyzer(nn.Module):
    """
    Analyze low-frequency conflicts using standard spectral analysis techniques.
    
    Implements proven audio engineering methods for detecting bass/kick conflicts
    and recommending frequency-selective crossfading strategies.
    """
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 hop_length: int = 512,
                 n_fft: int = 4096,
                 low_freq_max: float = 120.0,      # Sub-bass cutoff
                 bass_freq_max: float = 250.0,     # Bass range cutoff
                 kick_freq_max: float = 80.0):     # Kick fundamental range
        super().__init__()
        
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.low_freq_max = low_freq_max
        self.bass_freq_max = bass_freq_max
        self.kick_freq_max = kick_freq_max
        
        # Frequency bin calculations
        self.freqs = np.fft.fftfreq(n_fft, 1/sample_rate)[:n_fft//2 + 1]
        self.sub_bass_bins = np.where((self.freqs >= 20) & (self.freqs <= low_freq_max))[0]
        self.bass_bins = np.where((self.freqs > low_freq_max) & (self.freqs <= bass_freq_max))[0]
        self.kick_bins = np.where((self.freqs >= 30) & (self.freqs <= kick_freq_max))[0]
        
        # Neural network for conflict classification
        self.conflict_classifier = nn.Sequential(
            nn.Linear(12, 64),  # Spectral features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 5)  # 5 conflict types
        )
        
        # Severity assessment network
        self.severity_assessor = nn.Sequential(
            nn.Linear(8, 32),   # Conflict features
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, 
                audio_a: torch.Tensor,
                audio_b: torch.Tensor,
                crossfade_start_a: int,
                crossfade_start_b: int,
                crossfade_duration: int) -> LowEndConflict:
        """
        Analyze low-end conflicts during crossfade region.
        
        Args:
            audio_a: Track A audio (full track)
            audio_b: Track B audio (full track)  
            crossfade_start_a: Sample position where A crossfade begins
            crossfade_start_b: Sample position where B crossfade begins
            crossfade_duration: Crossfade duration in samples
            
        Returns:
            LowEndConflict with analysis results and recommendations
        """
        
        # Extract crossfade regions
        crossfade_a = self._extract_crossfade_region(
            audio_a, crossfade_start_a, crossfade_duration
        )
        crossfade_b = self._extract_crossfade_region(
            audio_b, crossfade_start_b, crossfade_duration
        )
        
        # Analyze low-frequency content
        low_freq_analysis_a = self._analyze_low_frequencies(crossfade_a)
        low_freq_analysis_b = self._analyze_low_frequencies(crossfade_b)
        
        # Detect specific conflict types
        kick_conflict = self._detect_kick_conflicts(
            low_freq_analysis_a, low_freq_analysis_b
        )
        
        bass_conflict = self._detect_bass_conflicts(
            low_freq_analysis_a, low_freq_analysis_b
        )
        
        phase_conflict = self._detect_phase_conflicts(
            crossfade_a, crossfade_b
        )
        
        # Determine primary conflict type and severity
        conflict_type, severity, conflict_strength = self._classify_primary_conflict(
            kick_conflict, bass_conflict, phase_conflict,
            low_freq_analysis_a, low_freq_analysis_b
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            conflict_type, severity, conflict_strength,
            low_freq_analysis_a, low_freq_analysis_b
        )
        
        return LowEndConflict(
            conflict_type=conflict_type,
            severity=severity,
            frequency_range_hz=recommendations['frequency_range'],
            conflict_strength=float(conflict_strength),
            recommended_crossfade_freq=recommendations['crossfade_freq'],
            bass_handoff_timing=recommendations.get('bass_handoff_timing'),
            kick_alignment_offset=recommendations.get('kick_alignment_offset'),
            processing_recommendation=recommendations['processing_recommendation']
        )
    
    def _extract_crossfade_region(self, 
                                 audio: torch.Tensor, 
                                 start_pos: int, 
                                 duration: int) -> torch.Tensor:
        """Extract crossfade region from audio"""
        
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        end_pos = min(start_pos + duration, audio.size(1))
        
        if start_pos >= audio.size(1):
            # Return silence if start position is beyond audio
            return torch.zeros(1, duration)
        
        region = audio[:, start_pos:end_pos]
        
        # Pad with silence if needed
        if region.size(1) < duration:
            padding = torch.zeros(region.size(0), duration - region.size(1))
            region = torch.cat([region, padding], dim=1)
        
        return region[0]  # Return mono
    
    def _analyze_low_frequencies(self, audio: torch.Tensor) -> Dict:
        """Analyze low-frequency content using STFT"""
        
        # Ensure audio is long enough for STFT analysis
        min_length = self.n_fft * 2
        if len(audio) < min_length:
            # Pad with zeros if too short
            padding_needed = min_length - len(audio)
            audio = torch.cat([audio, torch.zeros(padding_needed)])
        
        # Compute STFT
        stft = torch.stft(
            audio, 
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft),
            return_complex=True
        )
        
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Extract frequency bands
        sub_bass_energy = torch.mean(magnitude[self.sub_bass_bins, :], dim=0)
        bass_energy = torch.mean(magnitude[self.bass_bins, :], dim=0)
        kick_energy = torch.mean(magnitude[self.kick_bins, :], dim=0)
        
        # Calculate temporal characteristics
        sub_bass_rms = torch.sqrt(torch.mean(sub_bass_energy**2))
        bass_rms = torch.sqrt(torch.mean(bass_energy**2))
        kick_rms = torch.sqrt(torch.mean(kick_energy**2))
        
        # Detect transients (kick drums)
        kick_transients = self._detect_transients(kick_energy)
        
        # Calculate spectral centroid in low range
        low_range_bins = np.concatenate([self.sub_bass_bins, self.bass_bins])
        low_range_spectrum = torch.mean(magnitude[low_range_bins, :], dim=1)
        spectral_centroid = torch.sum(
            torch.from_numpy(self.freqs[low_range_bins]).float() * low_range_spectrum
        ) / (torch.sum(low_range_spectrum) + 1e-8)
        
        return {
            'sub_bass_energy': sub_bass_energy,
            'bass_energy': bass_energy,
            'kick_energy': kick_energy,
            'sub_bass_rms': sub_bass_rms,
            'bass_rms': bass_rms,
            'kick_rms': kick_rms,
            'kick_transients': kick_transients,
            'spectral_centroid': spectral_centroid,
            'magnitude_spectrum': magnitude,
            'phase_spectrum': phase
        }
    
    def _detect_transients(self, energy_curve: torch.Tensor) -> List[int]:
        """Detect transient events (kick drums) in energy curve"""
        
        # Calculate onset strength
        if len(energy_curve) > 1:
            diff = torch.diff(energy_curve, prepend=energy_curve[0:1])  # Use slice to maintain dimension
        else:
            diff = torch.zeros_like(energy_curve)
        onset_strength = torch.clamp(diff, min=0)
        
        # Find peaks
        threshold = torch.mean(onset_strength) + torch.std(onset_strength)
        peaks = []
        
        for i in range(1, len(onset_strength) - 1):
            if (onset_strength[i] > threshold and
                onset_strength[i] > onset_strength[i-1] and
                onset_strength[i] > onset_strength[i+1]):
                peaks.append(i)
        
        return peaks
    
    def _detect_kick_conflicts(self, 
                              analysis_a: Dict, 
                              analysis_b: Dict) -> Dict:
        """Detect kick drum conflicts"""
        
        kick_a_transients = analysis_a['kick_transients']
        kick_b_transients = analysis_b['kick_transients']
        
        # Check for overlapping kicks
        overlap_count = 0
        total_conflicts = 0
        
        frame_tolerance = 2  # Allow 2-frame tolerance for alignment
        
        for kick_a_frame in kick_a_transients:
            for kick_b_frame in kick_b_transients:
                if abs(kick_a_frame - kick_b_frame) <= frame_tolerance:
                    overlap_count += 1
                total_conflicts += 1
        
        if total_conflicts == 0:
            conflict_ratio = 0.0
        else:
            conflict_ratio = overlap_count / total_conflicts
        
        # Energy competition
        kick_energy_ratio = (analysis_a['kick_rms'] / (analysis_b['kick_rms'] + 1e-8)).item()
        
        return {
            'conflict_ratio': conflict_ratio,
            'energy_ratio': kick_energy_ratio,
            'overlapping_kicks': overlap_count,
            'total_kick_events': len(kick_a_transients) + len(kick_b_transients)
        }
    
    def _detect_bass_conflicts(self, 
                              analysis_a: Dict, 
                              analysis_b: Dict) -> Dict:
        """Detect bass line conflicts"""
        
        # Energy competition in bass range
        bass_energy_a = analysis_a['bass_rms']
        bass_energy_b = analysis_b['bass_rms']
        
        # Sub-bass competition
        sub_bass_energy_a = analysis_a['sub_bass_rms']
        sub_bass_energy_b = analysis_b['sub_bass_rms']
        
        # Spectral centroid difference (indicates different bass characteristics)
        centroid_diff = abs(analysis_a['spectral_centroid'] - analysis_b['spectral_centroid'])
        
        # Total low-end energy
        total_low_energy = bass_energy_a + bass_energy_b + sub_bass_energy_a + sub_bass_energy_b
        
        # Competition metrics
        bass_competition = min(bass_energy_a, bass_energy_b) / (max(bass_energy_a, bass_energy_b) + 1e-8)
        sub_competition = min(sub_bass_energy_a, sub_bass_energy_b) / (max(sub_bass_energy_a, sub_bass_energy_b) + 1e-8)
        
        return {
            'bass_competition': float(bass_competition),
            'sub_bass_competition': float(sub_competition),
            'centroid_difference_hz': float(centroid_diff),
            'total_low_energy': float(total_low_energy),
            'bass_energy_ratio': float(bass_energy_a / (bass_energy_b + 1e-8))
        }
    
    def _detect_phase_conflicts(self, 
                               audio_a: torch.Tensor, 
                               audio_b: torch.Tensor) -> Dict:
        """Detect phase cancellation issues in low frequencies"""
        
        # Low-pass filter both signals
        cutoff = 200.0  # Hz
        nyquist = self.sample_rate / 2
        cutoff_norm = cutoff / nyquist
        
        # Simple Butterworth filter approximation
        b = cutoff_norm
        a = 1 - b
        
        # Apply simple low-pass filter
        filtered_a = torch.zeros_like(audio_a)
        filtered_b = torch.zeros_like(audio_b)
        
        filtered_a[0] = audio_a[0] * b
        filtered_b[0] = audio_b[0] * b
        
        for i in range(1, len(audio_a)):
            filtered_a[i] = a * filtered_a[i-1] + b * audio_a[i]
            filtered_b[i] = a * filtered_b[i-1] + b * audio_b[i]
        
        # Calculate correlation and phase relationship
        correlation = torch.corrcoef(torch.stack([filtered_a, filtered_b]))[0, 1]
        
        # Sum and check for cancellation
        summed = filtered_a + filtered_b
        individual_energy = torch.mean(filtered_a**2) + torch.mean(filtered_b**2)
        summed_energy = torch.mean(summed**2)
        
        # Phase cancellation metric
        cancellation_ratio = 1.0 - (summed_energy / (individual_energy + 1e-8))
        
        return {
            'correlation': float(correlation),
            'cancellation_ratio': float(cancellation_ratio),
            'phase_alignment': float(torch.abs(correlation))
        }
    
    def _classify_primary_conflict(self, 
                                  kick_conflict: Dict,
                                  bass_conflict: Dict, 
                                  phase_conflict: Dict,
                                  analysis_a: Dict,
                                  analysis_b: Dict) -> Tuple[ConflictType, ConflictSeverity, float]:
        """Classify the primary conflict type and severity"""
        
        # Feature vector for neural classification
        features = torch.tensor([
            kick_conflict['conflict_ratio'],
            kick_conflict['energy_ratio'],
            bass_conflict['bass_competition'],
            bass_conflict['sub_bass_competition'],
            bass_conflict['centroid_difference_hz'] / 50.0,  # Normalize
            phase_conflict['correlation'],
            phase_conflict['cancellation_ratio'],
            float(analysis_a['kick_rms']),
            float(analysis_b['kick_rms']),
            float(analysis_a['bass_rms']),
            float(analysis_b['bass_rms']),
            phase_conflict['phase_alignment']
        ], dtype=torch.float32)
        
        try:
            # Neural classification
            with torch.no_grad():
                conflict_logits = self.conflict_classifier(features.unsqueeze(0))
                conflict_probs = F.softmax(conflict_logits, dim=1).squeeze()
                
                # Map to conflict types
                conflict_types = [
                    ConflictType.KICK_CLASH,
                    ConflictType.BASS_COMPETITION, 
                    ConflictType.PHASE_CANCELLATION,
                    ConflictType.MUDDY_BUILDUP,
                    ConflictType.NO_CONFLICT
                ]
                
                primary_conflict = conflict_types[torch.argmax(conflict_probs)]
                conflict_strength = torch.max(conflict_probs).item()
        
        except:
            # Fallback to heuristic classification
            primary_conflict, conflict_strength = self._heuristic_conflict_classification(
                kick_conflict, bass_conflict, phase_conflict
            )
        
        # Assess severity
        severity = self._assess_severity(conflict_strength, primary_conflict, kick_conflict, bass_conflict, phase_conflict)
        
        return primary_conflict, severity, conflict_strength
    
    def _heuristic_conflict_classification(self, 
                                         kick_conflict: Dict,
                                         bass_conflict: Dict,
                                         phase_conflict: Dict) -> Tuple[ConflictType, float]:
        """Heuristic conflict classification fallback"""
        
        # Check for kick conflicts
        if kick_conflict['conflict_ratio'] > 0.3:
            return ConflictType.KICK_CLASH, kick_conflict['conflict_ratio']
        
        # Check for phase cancellation
        if phase_conflict['cancellation_ratio'] > 0.3:
            return ConflictType.PHASE_CANCELLATION, phase_conflict['cancellation_ratio']
        
        # Check for bass competition
        if (bass_conflict['bass_competition'] > 0.5 and 
            bass_conflict['total_low_energy'] > 0.1):
            return ConflictType.BASS_COMPETITION, bass_conflict['bass_competition']
        
        # Check for muddy buildup
        if bass_conflict['total_low_energy'] > 0.3:
            return ConflictType.MUDDY_BUILDUP, bass_conflict['total_low_energy']
        
        return ConflictType.NO_CONFLICT, 0.1
    
    def _assess_severity(self, 
                        conflict_strength: float,
                        conflict_type: ConflictType,
                        kick_conflict: Dict,
                        bass_conflict: Dict,
                        phase_conflict: Dict) -> ConflictSeverity:
        """Assess conflict severity"""
        
        # Use neural network for severity assessment
        severity_features = torch.tensor([
            conflict_strength,
            kick_conflict['conflict_ratio'],
            bass_conflict['bass_competition'], 
            phase_conflict['cancellation_ratio'],
            kick_conflict.get('overlapping_kicks', 0) / 10.0,  # Normalize
            bass_conflict['total_low_energy'],
            float(conflict_type == ConflictType.KICK_CLASH),
            float(conflict_type == ConflictType.PHASE_CANCELLATION)
        ], dtype=torch.float32)
        
        try:
            with torch.no_grad():
                severity_score = self.severity_assessor(severity_features.unsqueeze(0)).item()
        except:
            # Fallback to simple thresholding
            severity_score = conflict_strength
        
        # Map to severity levels
        if severity_score < 0.2:
            return ConflictSeverity.NONE
        elif severity_score < 0.4:
            return ConflictSeverity.MILD
        elif severity_score < 0.6:
            return ConflictSeverity.MODERATE
        elif severity_score < 0.8:
            return ConflictSeverity.SEVERE
        else:
            return ConflictSeverity.CRITICAL
    
    def _generate_recommendations(self, 
                                 conflict_type: ConflictType,
                                 severity: ConflictSeverity,
                                 conflict_strength: float,
                                 analysis_a: Dict,
                                 analysis_b: Dict) -> Dict:
        """Generate processing recommendations"""
        
        recommendations = {
            'frequency_range': (20.0, self.bass_freq_max),
            'crossfade_freq': 120.0,  # Default bass crossfade frequency
            'processing_recommendation': 'standard_crossfade'
        }
        
        if conflict_type == ConflictType.KICK_CLASH:
            recommendations.update({
                'crossfade_freq': 80.0,  # Lower crossfade for kick conflicts
                'kick_alignment_offset': 0.0,  # Could be calculated from timing analysis
                'processing_recommendation': 'frequency_selective_kick_priority'
            })
            
        elif conflict_type == ConflictType.BASS_COMPETITION:
            recommendations.update({
                'crossfade_freq': 150.0,  # Higher for bass conflicts
                'bass_handoff_timing': self._calculate_bass_handoff_timing(analysis_a, analysis_b),
                'processing_recommendation': 'frequency_selective_bass_handoff'
            })
            
        elif conflict_type == ConflictType.PHASE_CANCELLATION:
            recommendations.update({
                'crossfade_freq': 200.0,  # Higher crossfade to avoid phase issues
                'processing_recommendation': 'phase_aligned_crossfade'
            })
            
        elif conflict_type == ConflictType.MUDDY_BUILDUP:
            recommendations.update({
                'crossfade_freq': 100.0,  # Lower to clear mud
                'processing_recommendation': 'high_pass_filter_crossfade'
            })
            
        else:  # NO_CONFLICT
            recommendations.update({
                'processing_recommendation': 'standard_equal_power_crossfade'
            })
        
        # Adjust based on severity
        if severity in [ConflictSeverity.SEVERE, ConflictSeverity.CRITICAL]:
            # More aggressive processing
            recommendations['crossfade_freq'] *= 0.8  # Lower crossfade frequency
            recommendations['processing_recommendation'] += '_aggressive'
        
        return recommendations
    
    def _calculate_bass_handoff_timing(self, analysis_a: Dict, analysis_b: Dict) -> float:
        """Calculate optimal bass handoff timing"""
        
        # Find quiet moments in bass energy for handoff
        bass_energy_a = analysis_a['bass_energy']
        bass_energy_b = analysis_b['bass_energy']
        
        # Find minimum energy point for handoff
        combined_energy = bass_energy_a + bass_energy_b
        min_energy_frame = torch.argmin(combined_energy).item()
        
        # Convert to time
        handoff_time = min_energy_frame * self.hop_length / self.sample_rate
        
        return handoff_time
    
    def get_frequency_selective_weights(self, conflict: LowEndConflict) -> Dict[str, float]:
        """Get frequency-selective crossfade weights"""
        
        if conflict.conflict_type == ConflictType.NO_CONFLICT:
            return {'low': 1.0, 'mid': 1.0, 'high': 1.0}
        
        # Calculate weights based on conflict type and severity
        severity_factor = {
            ConflictSeverity.NONE: 0.0,
            ConflictSeverity.MILD: 0.2,
            ConflictSeverity.MODERATE: 0.5,
            ConflictSeverity.SEVERE: 0.8,
            ConflictSeverity.CRITICAL: 1.0
        }[conflict.severity]
        
        # Base weights
        weights = {'low': 1.0, 'mid': 1.0, 'high': 1.0}
        
        if conflict.conflict_type in [ConflictType.KICK_CLASH, ConflictType.BASS_COMPETITION]:
            # Earlier crossfade in low frequencies
            weights['low'] = 1.0 + severity_factor * 0.5  # Faster fade
            weights['mid'] = 1.0
            weights['high'] = 1.0 - severity_factor * 0.2  # Slower fade
            
        elif conflict.conflict_type == ConflictType.MUDDY_BUILDUP:
            # Aggressive low-end crossfade
            weights['low'] = 1.0 + severity_factor * 0.8
            weights['mid'] = 1.0 + severity_factor * 0.3
            weights['high'] = 1.0
        
        return weights


if __name__ == "__main__":
    # Test the module
    analyzer = LowEndConflictAnalyzer()
    
    # Create test audio with low-end content
    duration = 4.0
    sample_rate = 44100
    t = torch.linspace(0, duration, int(duration * sample_rate))
    
    # Track A: 60Hz bass + 40Hz kick
    bass_a = 0.5 * torch.sin(2 * torch.pi * 60 * t)
    kick_a = 0.3 * torch.sin(2 * torch.pi * 40 * t) * torch.exp(-5 * (t % 1.0))
    audio_a = bass_a + kick_a
    
    # Track B: 65Hz bass + 42Hz kick (slightly different)
    bass_b = 0.4 * torch.sin(2 * torch.pi * 65 * t)
    kick_b = 0.4 * torch.sin(2 * torch.pi * 42 * t) * torch.exp(-5 * ((t + 0.1) % 1.0))
    audio_b = bass_b + kick_b
    
    # Analyze conflict
    conflict = analyzer(audio_a, audio_b, 0, 0, len(t))
    
    print("=== Low-End Conflict Analysis ===")
    print(f"Conflict type: {conflict.conflict_type.value}")
    print(f"Severity: {conflict.severity.value}")
    print(f"Conflict strength: {conflict.conflict_strength:.3f}")
    print(f"Recommended crossfade freq: {conflict.recommended_crossfade_freq:.1f} Hz")
    print(f"Processing recommendation: {conflict.processing_recommendation}")
    
    # Test frequency-selective weights
    weights = analyzer.get_frequency_selective_weights(conflict)
    print(f"Frequency weights: {weights}")