import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import librosa


class AdvancedMusicalFeatures(nn.Module):
    """
    Comprehensive musical feature extraction including:
    - Harmonic analysis (chroma, key detection, chord progression)
    - Rhythmic analysis (tempo, beat tracking, rhythmic complexity)
    - Timbral analysis (spectral features, formants, brightness)
    - Melodic analysis (pitch tracking, melodic contour)
    - Structural analysis (segmentation, repetition, form)
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        n_chroma: int = 12,
        f_min: float = 80.0,
        f_max: Optional[float] = None,
        tuning_resolution: float = 0.01,
        **kwargs
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_chroma = n_chroma
        self.f_max = f_max or sample_rate // 2
        
        # Core transforms
        self.stft_transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0
        )
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=self.f_max
        )
        
        # Chroma analysis
        self.chroma_transform = self._build_chroma_transform()
        
        # Timbral feature extractors
        self.timbral_analyzer = TimbralAnalyzer(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        # Harmonic analysis
        self.harmonic_analyzer = HarmonicAnalyzer(
            sample_rate=sample_rate,
            n_chroma=n_chroma,
            tuning_resolution=tuning_resolution
        )
        
        # Rhythmic analysis  
        self.rhythmic_analyzer = RhythmicAnalyzer(
            sample_rate=sample_rate,
            hop_length=hop_length
        )
        
        # Melodic analysis
        self.melodic_analyzer = MelodicAnalyzer(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=self.f_max
        )
        
        # Structural analysis
        self.structural_analyzer = StructuralAnalyzer(
            sample_rate=sample_rate,
            hop_length=hop_length
        )
        
    def _build_chroma_transform(self):
        """Build chroma feature transform."""
        # Create chroma filterbank
        chroma_fb = librosa.filters.chroma(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_chroma=self.n_chroma,
            tuning=0.0
        )
        
        # Convert to torch tensor
        chroma_fb = torch.from_numpy(chroma_fb).float()
        self.register_buffer('chroma_filterbank', chroma_fb)
        
        return lambda spec: torch.matmul(self.chroma_filterbank, spec)
        
    def forward(
        self,
        waveform: torch.Tensor,
        return_intermediate: bool = False,
        feature_groups: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract comprehensive musical features.
        
        Args:
            waveform: Input audio [batch, samples]
            return_intermediate: Whether to return intermediate computations
            feature_groups: Which feature groups to compute ['harmonic', 'rhythmic', 'timbral', 'melodic', 'structural']
            
        Returns:
            Dictionary with extracted features
        """
        if feature_groups is None:
            feature_groups = ['harmonic', 'rhythmic', 'timbral', 'melodic', 'structural']
            
        # Ensure mono audio
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
            
        batch_size = waveform.shape[0]
        
        # Core spectral representations
        stft = self.stft_transform(waveform)
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = torchaudio.functional.amplitude_to_DB(mel_spec, multiplier=10.0, amin=1e-8)
        
        # Chroma features
        chroma = self.chroma_transform(stft)
        chroma = F.normalize(chroma, p=2, dim=1)  # L2 normalize
        
        features = {
            'stft': stft,
            'mel_spectrogram': mel_spec_db,
            'chroma': chroma
        }
        
        # Extract features by group
        if 'harmonic' in feature_groups:
            harmonic_features = self.harmonic_analyzer(waveform, chroma)
            features.update(harmonic_features)
            
        if 'rhythmic' in feature_groups:
            rhythmic_features = self.rhythmic_analyzer(waveform, mel_spec_db)
            features.update(rhythmic_features)
            
        if 'timbral' in feature_groups:
            timbral_features = self.timbral_analyzer(waveform, stft, mel_spec)
            features.update(timbral_features)
            
        if 'melodic' in feature_groups:
            melodic_features = self.melodic_analyzer(waveform, stft)
            features.update(melodic_features)
            
        if 'structural' in feature_groups:
            structural_features = self.structural_analyzer(waveform, chroma, mel_spec_db)
            features.update(structural_features)
            
        # Aggregate statistics
        features['summary_stats'] = self._compute_summary_statistics(features)
        
        return features
        
    def _compute_summary_statistics(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute summary statistics across all features."""
        stats = {}
        
        # Temporal statistics for time-varying features
        temporal_features = ['chroma', 'mel_spectrogram']
        
        for feat_name in temporal_features:
            if feat_name in features:
                feat = features[feat_name]
                if feat.dim() >= 2:
                    # Statistics over time
                    stats[f'{feat_name}_mean'] = feat.mean(dim=-1)
                    stats[f'{feat_name}_std'] = feat.std(dim=-1)
                    stats[f'{feat_name}_median'] = feat.median(dim=-1)[0]
                    
        return stats


class TimbralAnalyzer(nn.Module):
    """Extract timbral characteristics and spectral features."""
    
    def __init__(self, sample_rate: int, n_fft: int, hop_length: int):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # MFCC extraction
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=13,
            melkwargs={'n_fft': n_fft, 'hop_length': hop_length}
        )
        
    def forward(
        self, 
        waveform: torch.Tensor, 
        stft: torch.Tensor, 
        mel_spec: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Extract timbral features."""
        # MFCCs
        mfcc = self.mfcc_transform(waveform)
        
        # Spectral features from magnitude spectrum
        magnitude = torch.sqrt(stft + 1e-8)
        
        # Spectral centroid
        freqs = torch.linspace(0, self.sample_rate//2, stft.shape[1], device=stft.device)
        freqs = freqs.unsqueeze(0).unsqueeze(-1)  # [1, freq, 1]
        
        spectral_centroid = torch.sum(magnitude * freqs, dim=1) / (torch.sum(magnitude, dim=1) + 1e-8)
        
        # Spectral bandwidth
        centroid_expanded = spectral_centroid.unsqueeze(1)  # [batch, 1, time]
        spectral_bandwidth = torch.sqrt(
            torch.sum(magnitude * (freqs - centroid_expanded)**2, dim=1) / 
            (torch.sum(magnitude, dim=1) + 1e-8)
        )
        
        # Spectral rolloff (85th percentile)
        cumsum_mag = torch.cumsum(magnitude, dim=1)
        total_energy = cumsum_mag[:, -1:, :]  # Last value is total
        rolloff_threshold = 0.85 * total_energy
        
        # Find rolloff frequency
        rolloff_indices = torch.searchsorted(cumsum_mag.transpose(1, 2), rolloff_threshold.transpose(1, 2))
        rolloff_indices = rolloff_indices.transpose(1, 2).clamp(max=len(freqs)-1)
        spectral_rolloff = freqs.squeeze()[rolloff_indices.squeeze(1)]
        
        # Zero crossing rate
        zero_crossings = self._compute_zero_crossing_rate(waveform)
        
        # Spectral flatness (Wiener entropy)
        spectral_flatness = self._compute_spectral_flatness(magnitude)
        
        return {
            'mfcc': mfcc,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_rolloff': spectral_rolloff,
            'zero_crossing_rate': zero_crossings,
            'spectral_flatness': spectral_flatness
        }
        
    def _compute_zero_crossing_rate(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute zero crossing rate."""
        # Frame the signal
        frame_length = self.hop_length * 2
        frames = waveform.unfold(-1, frame_length, self.hop_length)
        
        # Count zero crossings per frame
        signs = torch.sign(frames)
        sign_changes = torch.abs(torch.diff(signs, dim=-1))
        zcr = torch.sum(sign_changes, dim=-1) / frame_length
        
        return zcr
        
    def _compute_spectral_flatness(self, magnitude: torch.Tensor) -> torch.Tensor:
        """Compute spectral flatness (Wiener entropy)."""
        # Geometric mean / Arithmetic mean
        log_magnitude = torch.log(magnitude + 1e-8)
        geometric_mean = torch.exp(torch.mean(log_magnitude, dim=1))
        arithmetic_mean = torch.mean(magnitude, dim=1)
        
        flatness = geometric_mean / (arithmetic_mean + 1e-8)
        return flatness


class HarmonicAnalyzer(nn.Module):
    """Analyze harmonic content, key, and chord progressions."""
    
    def __init__(self, sample_rate: int, n_chroma: int = 12, tuning_resolution: float = 0.01):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_chroma = n_chroma
        
        # Key profiles (Krumhansl-Schmuckler)
        major_profile = torch.tensor([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = torch.tensor([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        self.register_buffer('major_profile', major_profile / major_profile.sum())
        self.register_buffer('minor_profile', minor_profile / minor_profile.sum())
        
    def forward(self, waveform: torch.Tensor, chroma: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze harmonic content."""
        # Key detection using template matching
        key_correlations = self._detect_key(chroma)
        
        # Chord detection (simplified)
        chord_probabilities = self._detect_chords(chroma)
        
        # Harmonic change detection
        harmonic_novelty = self._compute_harmonic_novelty(chroma)
        
        # Tonal centroid features
        tonal_centroid = self._compute_tonal_centroid(chroma)
        
        return {
            'key_correlations': key_correlations,
            'chord_probabilities': chord_probabilities,
            'harmonic_novelty': harmonic_novelty,
            'tonal_centroid': tonal_centroid
        }
        
    def _detect_key(self, chroma: torch.Tensor) -> torch.Tensor:
        """Detect musical key using Krumhansl-Schmuckler profiles."""
        # Average chroma over time
        avg_chroma = chroma.mean(dim=-1)  # [batch, n_chroma]
        
        # Correlate with key profiles for all 24 keys
        correlations = []
        
        for shift in range(12):
            # Major key
            shifted_major = torch.roll(self.major_profile, shift)
            major_corr = F.cosine_similarity(avg_chroma, shifted_major.unsqueeze(0), dim=1)
            correlations.append(major_corr)
            
            # Minor key  
            shifted_minor = torch.roll(self.minor_profile, shift)
            minor_corr = F.cosine_similarity(avg_chroma, shifted_minor.unsqueeze(0), dim=1)
            correlations.append(minor_corr)
            
        return torch.stack(correlations, dim=1)  # [batch, 24]
        
    def _detect_chords(self, chroma: torch.Tensor) -> torch.Tensor:
        """Simplified chord detection using chroma templates."""
        # Basic chord templates (major, minor, diminished triads)
        chord_templates = self._get_chord_templates()
        
        # Compute correlations
        batch_size, n_chroma, n_frames = chroma.shape
        n_chords = chord_templates.shape[0]
        
        chord_probs = torch.zeros(batch_size, n_chords, n_frames, device=chroma.device)
        
        for i, template in enumerate(chord_templates):
            for frame in range(n_frames):
                chord_probs[:, i, frame] = F.cosine_similarity(
                    chroma[:, :, frame], 
                    template.unsqueeze(0), 
                    dim=1
                )
                
        return chord_probs
        
    def _get_chord_templates(self) -> torch.Tensor:
        """Get basic chord templates."""
        # Major triads (root, third, fifth)
        major_intervals = [0, 4, 7]
        # Minor triads
        minor_intervals = [0, 3, 7]
        # Diminished triads
        dim_intervals = [0, 3, 6]
        
        templates = []
        
        for root in range(12):
            # Major
            template = torch.zeros(12)
            for interval in major_intervals:
                template[(root + interval) % 12] = 1.0
            templates.append(template)
            
            # Minor
            template = torch.zeros(12)
            for interval in minor_intervals:
                template[(root + interval) % 12] = 1.0
            templates.append(template)
            
            # Diminished
            template = torch.zeros(12)
            for interval in dim_intervals:
                template[(root + interval) % 12] = 1.0
            templates.append(template)
            
        return torch.stack(templates)
        
    def _compute_harmonic_novelty(self, chroma: torch.Tensor) -> torch.Tensor:
        """Compute harmonic change detection."""
        # First-order difference in chroma
        chroma_diff = torch.diff(chroma, dim=-1)
        novelty = torch.norm(chroma_diff, p=2, dim=1)
        
        # Pad to match original length
        novelty = F.pad(novelty, (1, 0), mode='constant', value=0)
        
        return novelty
        
    def _compute_tonal_centroid(self, chroma: torch.Tensor) -> torch.Tensor:
        """Compute tonal centroid features (Harte et al.)."""
        # Transform chroma to tonal space
        angles = torch.linspace(0, 2*np.pi, 12, device=chroma.device, dtype=chroma.dtype)
        
        # Circle of fifths transformation
        fifth_angles = angles * 7 % (2 * np.pi)
        
        cos_components = torch.cos(fifth_angles).unsqueeze(0).unsqueeze(-1)
        sin_components = torch.sin(fifth_angles).unsqueeze(0).unsqueeze(-1)
        
        centroid_x = torch.sum(chroma * cos_components, dim=1)
        centroid_y = torch.sum(chroma * sin_components, dim=1)
        
        return torch.stack([centroid_x, centroid_y], dim=1)


class RhythmicAnalyzer(nn.Module):
    """Analyze rhythmic patterns, tempo, and beat structure."""
    
    def __init__(self, sample_rate: int, hop_length: int):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        
    def forward(self, waveform: torch.Tensor, mel_spec: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze rhythmic features."""
        # Onset detection
        onset_strength = self._compute_onset_strength(mel_spec)
        
        # Tempo estimation
        tempo_features = self._estimate_tempo(onset_strength)
        
        # Beat tracking (simplified)
        beat_features = self._track_beats(onset_strength)
        
        # Rhythmic complexity
        rhythmic_complexity = self._compute_rhythmic_complexity(onset_strength)
        
        return {
            'onset_strength': onset_strength,
            'tempo_estimate': tempo_features['tempo'],
            'tempo_confidence': tempo_features['confidence'],
            'beat_strength': beat_features,
            'rhythmic_complexity': rhythmic_complexity
        }
        
    def _compute_onset_strength(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Compute onset strength function."""
        # Spectral flux
        mel_diff = torch.diff(mel_spec, dim=-1)
        mel_diff = torch.clamp(mel_diff, min=0)  # Half-wave rectification
        onset_strength = torch.sum(mel_diff, dim=1)
        
        # Pad to match original length
        onset_strength = F.pad(onset_strength, (1, 0), mode='constant', value=0)
        
        return onset_strength
        
    def _estimate_tempo(self, onset_strength: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Estimate tempo using autocorrelation."""
        # Convert to numpy for librosa processing
        onset_np = onset_strength.detach().cpu().numpy()
        
        tempos = []
        confidences = []
        
        for i in range(onset_np.shape[0]):
            try:
                tempo, beats = librosa.beat.beat_track(
                    onset_envelope=onset_np[i],
                    sr=self.sample_rate,
                    hop_length=self.hop_length,
                    units='time'
                )
                
                # Compute confidence as beat consistency
                if len(beats) > 1:
                    beat_intervals = np.diff(beats)
                    confidence = 1.0 / (1.0 + np.std(beat_intervals))
                else:
                    confidence = 0.0
                    
                tempos.append(tempo)
                confidences.append(confidence)
                
            except:
                tempos.append(120.0)  # Default tempo
                confidences.append(0.0)
                
        return {
            'tempo': torch.tensor(tempos, device=onset_strength.device),
            'confidence': torch.tensor(confidences, device=onset_strength.device)
        }
        
    def _track_beats(self, onset_strength: torch.Tensor) -> torch.Tensor:
        """Simple beat tracking using dynamic programming."""
        # This is a simplified version - full beat tracking is complex
        # Use onset strength as proxy for beat strength
        
        # Apply smoothing
        kernel_size = 5
        kernel = torch.ones(1, 1, kernel_size, device=onset_strength.device) / kernel_size
        smoothed = F.conv1d(
            onset_strength.unsqueeze(1), 
            kernel, 
            padding=kernel_size//2
        ).squeeze(1)
        
        return smoothed
        
    def _compute_rhythmic_complexity(self, onset_strength: torch.Tensor) -> torch.Tensor:
        """Compute rhythmic complexity measure."""
        # Use entropy of onset timing distribution
        # Quantize to rhythmic grid
        grid_size = 16  # 16th note grid
        
        complexity_scores = []
        
        for i in range(onset_strength.shape[0]):
            strength = onset_strength[i]
            
            # Find peaks (onsets)
            threshold = torch.quantile(strength, 0.75)
            peaks = (strength > threshold).float()
            
            # Quantize to grid
            frame_to_beat = len(peaks) / grid_size
            quantized = torch.zeros(grid_size)
            
            for j, peak in enumerate(peaks):
                if peak > 0:
                    grid_pos = int(j / frame_to_beat) % grid_size
                    quantized[grid_pos] = 1.0
                    
            # Compute entropy
            probs = quantized / (quantized.sum() + 1e-8)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
            complexity_scores.append(entropy)
            
        return torch.tensor(complexity_scores, device=onset_strength.device)


class MelodicAnalyzer(nn.Module):
    """Analyze melodic content and pitch relationships."""
    
    def __init__(self, sample_rate: int, n_fft: int, hop_length: int, f_min: float, f_max: float):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        
    def forward(self, waveform: torch.Tensor, stft: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze melodic features."""
        # Fundamental frequency estimation
        f0_features = self._estimate_f0(stft)
        
        # Melodic contour
        contour_features = self._analyze_contour(f0_features['f0'])
        
        # Pitch stability
        pitch_stability = self._compute_pitch_stability(f0_features['f0'])
        
        return {
            'f0': f0_features['f0'],
            'f0_confidence': f0_features['confidence'],
            'melodic_contour': contour_features,
            'pitch_stability': pitch_stability
        }
        
    def _estimate_f0(self, stft: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Estimate fundamental frequency using spectral peaks."""
        magnitude = torch.sqrt(stft + 1e-8)
        
        # Find spectral peaks for each frame
        f0_estimates = []
        confidences = []
        
        for batch_idx in range(magnitude.shape[0]):
            batch_f0 = []
            batch_conf = []
            
            for frame_idx in range(magnitude.shape[2]):
                frame_mag = magnitude[batch_idx, :, frame_idx]
                
                # Find peaks
                peaks = self._find_spectral_peaks(frame_mag)
                
                if len(peaks) > 0:
                    # Use strongest peak as F0 estimate
                    strongest_peak = peaks[torch.argmax(frame_mag[peaks])]
                    f0_hz = strongest_peak * self.sample_rate / self.n_fft
                    confidence = frame_mag[strongest_peak] / torch.sum(frame_mag)
                    
                    batch_f0.append(f0_hz)
                    batch_conf.append(confidence)
                else:
                    batch_f0.append(0.0)
                    batch_conf.append(0.0)
                    
            f0_estimates.append(torch.tensor(batch_f0))
            confidences.append(torch.tensor(batch_conf))
            
        return {
            'f0': torch.stack(f0_estimates).to(stft.device),
            'confidence': torch.stack(confidences).to(stft.device)
        }
        
    def _find_spectral_peaks(self, magnitude: torch.Tensor, prominence: float = 0.1) -> torch.Tensor:
        """Find peaks in magnitude spectrum."""
        # Simple peak finding - can be improved with scipy.signal.find_peaks equivalent
        peaks = []
        
        for i in range(1, len(magnitude) - 1):
            if (magnitude[i] > magnitude[i-1] and 
                magnitude[i] > magnitude[i+1] and 
                magnitude[i] > prominence):
                peaks.append(i)
                
        return torch.tensor(peaks, dtype=torch.long) if peaks else torch.tensor([], dtype=torch.long)
        
    def _analyze_contour(self, f0: torch.Tensor) -> torch.Tensor:
        """Analyze melodic contour direction."""
        # Compute first-order differences
        f0_diff = torch.diff(f0, dim=-1)
        
        # Quantize to direction (up, down, stable)
        threshold = 10.0  # Hz threshold for stability
        contour = torch.zeros_like(f0_diff)
        contour[f0_diff > threshold] = 1.0   # Rising
        contour[f0_diff < -threshold] = -1.0  # Falling
        
        return contour
        
    def _compute_pitch_stability(self, f0: torch.Tensor) -> torch.Tensor:
        """Compute pitch stability measure."""
        # Remove silent frames (f0 == 0)
        voiced_frames = f0 > 0
        
        stability_scores = []
        
        for i in range(f0.shape[0]):
            voiced = voiced_frames[i]
            if torch.sum(voiced) > 1:
                voiced_f0 = f0[i][voiced]
                stability = 1.0 / (1.0 + torch.std(voiced_f0))
            else:
                stability = 0.0
                
            stability_scores.append(stability)
            
        return torch.tensor(stability_scores, device=f0.device)


class StructuralAnalyzer(nn.Module):
    """Analyze musical structure and form."""
    
    def __init__(self, sample_rate: int, hop_length: int):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        
    def forward(
        self, 
        waveform: torch.Tensor, 
        chroma: torch.Tensor, 
        mel_spec: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Analyze structural features."""
        # Self-similarity matrix
        similarity_matrix = self._compute_similarity_matrix(chroma)
        
        # Novelty detection
        novelty = self._compute_novelty(similarity_matrix)
        
        # Repetition analysis
        repetition_features = self._analyze_repetition(similarity_matrix)
        
        return {
            'self_similarity': similarity_matrix,
            'structural_novelty': novelty,
            'repetition_strength': repetition_features
        }
        
    def _compute_similarity_matrix(self, chroma: torch.Tensor) -> torch.Tensor:
        """Compute self-similarity matrix from chroma features."""
        # Normalize chroma vectors
        chroma_norm = F.normalize(chroma, p=2, dim=1)
        
        # Compute cosine similarity
        batch_size, n_chroma, n_frames = chroma_norm.shape
        similarity_matrices = []
        
        for i in range(batch_size):
            chroma_batch = chroma_norm[i].transpose(0, 1)  # [n_frames, n_chroma]
            similarity = torch.mm(chroma_batch, chroma_batch.transpose(0, 1))
            similarity_matrices.append(similarity)
            
        return torch.stack(similarity_matrices)
        
    def _compute_novelty(self, similarity_matrix: torch.Tensor) -> torch.Tensor:
        """Compute structural novelty from self-similarity."""
        # Use checkerboard kernel for novelty detection
        kernel_size = 16
        
        novelty_scores = []
        
        for i in range(similarity_matrix.shape[0]):
            sim = similarity_matrix[i]
            n_frames = sim.shape[0]
            
            novelty = torch.zeros(n_frames)
            
            for t in range(kernel_size, n_frames - kernel_size):
                # Extract local region
                region = sim[t-kernel_size:t+kernel_size, t-kernel_size:t+kernel_size]
                
                # Compute novelty as change in self-similarity
                upper_tri = torch.triu(region, diagonal=1)
                lower_tri = torch.tril(region, diagonal=-1)
                
                novelty[t] = torch.abs(torch.mean(upper_tri) - torch.mean(lower_tri))
                
            novelty_scores.append(novelty)
            
        return torch.stack(novelty_scores).to(similarity_matrix.device)
        
    def _analyze_repetition(self, similarity_matrix: torch.Tensor) -> torch.Tensor:
        """Analyze repetitive structure."""
        # Look for diagonal stripes in similarity matrix
        repetition_scores = []
        
        for i in range(similarity_matrix.shape[0]):
            sim = similarity_matrix[i]
            
            # Sum diagonals at different lags
            n_frames = sim.shape[0]
            max_lag = min(50, n_frames // 4)  # Maximum lag to consider
            
            diagonal_sums = []
            for lag in range(1, max_lag):
                diagonal = torch.diagonal(sim, offset=lag)
                diagonal_sums.append(torch.mean(diagonal))
                
            if diagonal_sums:
                repetition_strength = torch.max(torch.tensor(diagonal_sums))
            else:
                repetition_strength = torch.tensor(0.0)
                
            repetition_scores.append(repetition_strength)
            
        return torch.stack(repetition_scores).to(similarity_matrix.device)


# Example usage and factory functions
def extract_comprehensive_features(
    waveform: torch.Tensor,
    sample_rate: int = 22050,
    feature_groups: Optional[List[str]] = None
) -> Dict[str, torch.Tensor]:
    """Convenience function for comprehensive feature extraction."""
    analyzer = AdvancedMusicalFeatures(sample_rate=sample_rate)
    return analyzer(waveform, feature_groups=feature_groups)


if __name__ == "__main__":
    # Test comprehensive feature extraction
    sample_rate = 22050
    duration = 10  # seconds
    waveform = torch.randn(1, sample_rate * duration)
    
    # Extract all features
    features = extract_comprehensive_features(waveform, sample_rate)
    
    print("Extracted features:")
    for key, value in features.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, dict):
            print(f"  {key}: {list(value.keys())}")
            
    # Extract specific feature groups
    harmonic_features = extract_comprehensive_features(
        waveform, sample_rate, feature_groups=['harmonic', 'timbral']
    )
    
    print(f"\nHarmonic + timbral features: {list(harmonic_features.keys())}")