import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import librosa


class PerceptualQualityAssessor(nn.Module):
    """
    Comprehensive perceptual audio quality assessment combining:
    - Psychoacoustic models (masking, critical bands)
    - Perceptual distance metrics (PESQ-like, STOI-like)
    - Multi-scale spectral losses
    - Learned quality assessment
    - Content-aware quality metrics
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        n_bark_bands: int = 24,
        quality_metrics: List[str] = ['spectral', 'psychoacoustic', 'temporal', 'learned'],
        content_aware: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_bark_bands = n_bark_bands
        self.quality_metrics = quality_metrics
        self.content_aware = content_aware
        
        # Core transforms
        self.stft_transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0
        )
        
        # Bark scale filterbank for psychoacoustic modeling
        self.bark_filterbank = self._create_bark_filterbank()
        
        # Multi-scale STFT for different resolutions
        self.multi_scale_stft = nn.ModuleDict({
            'fine': torchaudio.transforms.Spectrogram(n_fft=512, hop_length=128, power=2.0),
            'medium': torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256, power=2.0),
            'coarse': torchaudio.transforms.Spectrogram(n_fft=2048, hop_length=512, power=2.0)
        })
        
        # Learned quality network
        if 'learned' in quality_metrics:
            self.quality_network = self._build_quality_network()
            
        # Content classifier for content-aware assessment
        if content_aware:
            self.content_classifier = self._build_content_classifier()
            
        # Psychoacoustic model components
        self.masking_model = PsychoacousticMaskingModel(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_bark_bands=n_bark_bands
        )
        
        # Temporal quality analyzer
        self.temporal_analyzer = TemporalQualityAnalyzer(
            sample_rate=sample_rate,
            hop_length=hop_length
        )
        
    def _create_bark_filterbank(self):
        """Create Bark scale filterbank for psychoacoustic modeling."""
        # Bark scale frequency edges
        bark_edges = np.array([
            0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720,
            2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500
        ])
        
        # Convert to bin indices
        freq_bins = np.linspace(0, self.sample_rate // 2, self.n_fft // 2 + 1)
        bark_bins = np.interp(bark_edges, freq_bins, np.arange(len(freq_bins)))
        
        # Create filterbank
        filterbank = np.zeros((self.n_bark_bands, self.n_fft // 2 + 1))
        
        for i in range(self.n_bark_bands):
            start_bin = int(bark_bins[i])
            end_bin = int(bark_bins[i + 1])
            
            if end_bin > start_bin:
                # Triangular filter
                filterbank[i, start_bin:end_bin] = np.linspace(0, 1, end_bin - start_bin)
                if i < self.n_bark_bands - 1:
                    next_end = int(bark_bins[i + 2]) if i + 2 < len(bark_bins) else len(freq_bins)
                    filterbank[i, end_bin:min(next_end, len(freq_bins))] = np.linspace(1, 0, 
                        min(next_end, len(freq_bins)) - end_bin)
                        
        filterbank = torch.from_numpy(filterbank).float()
        self.register_buffer('bark_filterbank', filterbank)
        
        return filterbank
        
    def _build_quality_network(self):
        """Build learned quality assessment network."""
        return nn.Sequential(
            # Multi-scale feature extraction
            QualityFeatureExtractor(
                input_channels=1,
                hidden_dim=128,
                num_scales=3
            ),
            
            # Quality regression head
            nn.Linear(128 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Quality score 0-1
        )
        
    def _build_content_classifier(self):
        """Build content classifier for content-aware quality assessment."""
        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 4),  # speech, music, noise, silence
            nn.Softmax(dim=1)
        )
        
    def extract_spectral_features(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract multi-scale spectral features."""
        features = {}
        
        # Multi-scale STFT
        for scale_name, stft_transform in self.multi_scale_stft.items():
            stft_mag = stft_transform(waveform)
            stft_db = 10 * torch.log10(stft_mag + 1e-8)
            features[f'stft_{scale_name}'] = stft_db
            
        # Bark scale representation
        primary_stft = features['stft_medium']
        bark_spectrum = torch.matmul(self.bark_filterbank, primary_stft)
        features['bark_spectrum'] = bark_spectrum
        
        return features
        
    def compute_spectral_quality(
        self, 
        reference: torch.Tensor, 
        degraded: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute spectral quality metrics."""
        # Extract features for both signals
        ref_features = self.extract_spectral_features(reference)
        deg_features = self.extract_spectral_features(degraded)
        
        quality_scores = {}
        
        # Multi-scale spectral convergence
        for scale in ['fine', 'medium', 'coarse']:
            ref_spec = ref_features[f'stft_{scale}']
            deg_spec = deg_features[f'stft_{scale}']
            
            # Align temporal dimensions
            min_time = min(ref_spec.shape[-1], deg_spec.shape[-1])
            ref_spec = ref_spec[..., :min_time]
            deg_spec = deg_spec[..., :min_time]
            
            # Spectral convergence
            spectral_conv = torch.norm(ref_spec - deg_spec, p='fro') / torch.norm(ref_spec, p='fro')
            quality_scores[f'spectral_convergence_{scale}'] = 1 - spectral_conv.clamp(0, 1)
            
            # Log magnitude distance
            log_mag_dist = F.l1_loss(ref_spec, deg_spec)
            quality_scores[f'log_magnitude_distance_{scale}'] = torch.exp(-log_mag_dist)
            
        # Bark scale quality
        ref_bark = ref_features['bark_spectrum']
        deg_bark = deg_features['bark_spectrum']
        
        min_time = min(ref_bark.shape[-1], deg_bark.shape[-1])
        ref_bark = ref_bark[..., :min_time]
        deg_bark = deg_bark[..., :min_time]
        
        bark_distance = F.mse_loss(ref_bark, deg_bark)
        quality_scores['bark_quality'] = torch.exp(-bark_distance)
        
        return quality_scores
        
    def compute_psychoacoustic_quality(
        self, 
        reference: torch.Tensor, 
        degraded: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute psychoacoustic quality using masking models."""
        # Use psychoacoustic masking model
        masking_results = self.masking_model(reference, degraded)
        
        return {
            'psychoacoustic_quality': masking_results['overall_quality'],
            'masking_threshold_error': masking_results['masking_error'],
            'loudness_error': masking_results['loudness_error']
        }
        
    def compute_temporal_quality(
        self, 
        reference: torch.Tensor, 
        degraded: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute temporal quality metrics."""
        return self.temporal_analyzer(reference, degraded)
        
    def compute_learned_quality(
        self, 
        reference: torch.Tensor, 
        degraded: torch.Tensor
    ) -> torch.Tensor:
        """Compute quality using learned network."""
        # Compute difference signal
        min_len = min(reference.shape[-1], degraded.shape[-1])
        ref_aligned = reference[..., :min_len]
        deg_aligned = degraded[..., :min_len]
        
        # Extract spectral features
        ref_spec = self.stft_transform(ref_aligned)
        deg_spec = self.stft_transform(deg_aligned)
        
        # Compute difference in log-magnitude domain
        ref_log = torch.log(ref_spec + 1e-8)
        deg_log = torch.log(deg_spec + 1e-8)
        
        diff_spec = torch.abs(ref_log - deg_log).unsqueeze(1)  # Add channel dim
        
        # Pass through quality network
        quality_score = self.quality_network(diff_spec)
        
        return quality_score.squeeze(-1)
        
    def forward(
        self, 
        reference: torch.Tensor, 
        degraded: torch.Tensor,
        return_individual_metrics: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive perceptual quality assessment.
        
        Args:
            reference: Reference audio [batch, samples]
            degraded: Degraded audio [batch, samples]
            return_individual_metrics: Whether to return individual metric scores
            
        Returns:
            Dictionary with quality scores
        """
        # Ensure mono audio
        if reference.dim() == 3:
            reference = reference.squeeze(1)
        if degraded.dim() == 3:
            degraded = degraded.squeeze(1)
            
        quality_metrics = {}
        
        # Spectral quality
        if 'spectral' in self.quality_metrics:
            spectral_quality = self.compute_spectral_quality(reference, degraded)
            quality_metrics.update(spectral_quality)
            
        # Psychoacoustic quality
        if 'psychoacoustic' in self.quality_metrics:
            psychoacoustic_quality = self.compute_psychoacoustic_quality(reference, degraded)
            quality_metrics.update(psychoacoustic_quality)
            
        # Temporal quality
        if 'temporal' in self.quality_metrics:
            temporal_quality = self.compute_temporal_quality(reference, degraded)
            quality_metrics.update(temporal_quality)
            
        # Learned quality
        if 'learned' in self.quality_metrics:
            learned_quality = self.compute_learned_quality(reference, degraded)
            quality_metrics['learned_quality'] = learned_quality
            
        # Content-aware weighting
        if self.content_aware:
            content_weights = self._compute_content_weights(reference)
            quality_metrics['content_weights'] = content_weights
            
        # Compute overall quality score
        overall_quality = self._compute_overall_quality(quality_metrics)
        quality_metrics['overall_quality'] = overall_quality
        
        result = {'quality_scores': quality_metrics}
        
        if return_individual_metrics:
            result['individual_metrics'] = quality_metrics
            
        return result
        
    def _compute_content_weights(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute content-based weights for quality metrics."""
        # Extract spectrogram for content classification
        spec = self.stft_transform(audio)
        spec_db = 10 * torch.log10(spec + 1e-8)
        spec_input = spec_db.unsqueeze(1)  # Add channel dimension
        
        # Classify content
        content_probs = self.content_classifier(spec_input)
        
        return content_probs
        
    def _compute_overall_quality(self, metrics: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute weighted overall quality score."""
        # Extract numeric quality scores
        quality_scores = []
        weights = []
        
        # Spectral metrics
        spectral_metrics = ['spectral_convergence_medium', 'bark_quality']
        for metric in spectral_metrics:
            if metric in metrics:
                quality_scores.append(metrics[metric])
                weights.append(0.3)
                
        # Psychoacoustic metrics
        if 'psychoacoustic_quality' in metrics:
            quality_scores.append(metrics['psychoacoustic_quality'])
            weights.append(0.3)
            
        # Temporal metrics
        if 'temporal_quality' in metrics:
            quality_scores.append(metrics['temporal_quality'])
            weights.append(0.2)
            
        # Learned metrics
        if 'learned_quality' in metrics:
            quality_scores.append(metrics['learned_quality'])
            weights.append(0.2)
            
        if quality_scores:
            # Normalize weights
            weights = torch.tensor(weights, device=quality_scores[0].device)
            weights = weights / weights.sum()
            
            # Weighted average
            stacked_scores = torch.stack(quality_scores, dim=1)
            overall = torch.sum(stacked_scores * weights.unsqueeze(0), dim=1)
            
            return overall
        else:
            # Fallback to neutral score
            return torch.ones(metrics[list(metrics.keys())[0]].shape[0])
            
    def assess_single_audio_quality(
        self, 
        audio: torch.Tensor,
        reference_type: str = 'clean'
    ) -> Dict[str, torch.Tensor]:
        """
        Assess quality of single audio without reference (blind quality assessment).
        
        Args:
            audio: Input audio [batch, samples]
            reference_type: Type of expected reference ('clean', 'music', 'speech')
            
        Returns:
            Dictionary with quality assessment
        """
        # Extract features
        features = self.extract_spectral_features(audio)
        
        # Compute intrinsic quality indicators
        quality_indicators = {}
        
        # Signal-to-noise ratio estimation
        snr_estimate = self._estimate_snr(audio)
        quality_indicators['estimated_snr'] = snr_estimate
        
        # Spectral flatness (indication of noise)
        spectral_flatness = self._compute_spectral_flatness(features['stft_medium'])
        quality_indicators['spectral_flatness'] = spectral_flatness
        
        # Crest factor (dynamic range indicator)
        crest_factor = self._compute_crest_factor(audio)
        quality_indicators['crest_factor'] = crest_factor
        
        # Harmonic distortion estimate
        thd_estimate = self._estimate_thd(features['stft_medium'])
        quality_indicators['thd_estimate'] = thd_estimate
        
        # Temporal discontinuities
        temporal_discontinuities = self._detect_temporal_discontinuities(audio)
        quality_indicators['temporal_discontinuities'] = temporal_discontinuities
        
        # Content-based quality expectations
        if self.content_aware:
            content_weights = self._compute_content_weights(audio)
            quality_indicators['content_type'] = content_weights
            
        # Overall blind quality score
        blind_quality = self._compute_blind_quality_score(quality_indicators)
        quality_indicators['blind_quality_score'] = blind_quality
        
        return quality_indicators
        
    def _estimate_snr(self, audio: torch.Tensor) -> torch.Tensor:
        """Estimate signal-to-noise ratio."""
        # Simple energy-based SNR estimation
        # Assume noise is in the quietest 10% of frames
        
        frame_length = 1024
        hop_length = 512
        
        # Frame the signal
        frames = audio.unfold(-1, frame_length, hop_length)
        frame_energy = torch.mean(frames ** 2, dim=-1)
        
        # Estimate noise floor from quietest frames
        noise_threshold = torch.quantile(frame_energy, 0.1, dim=-1, keepdim=True)
        signal_energy = torch.mean(frame_energy, dim=-1)
        
        # SNR in dB
        snr_db = 10 * torch.log10(signal_energy / (noise_threshold.squeeze(-1) + 1e-8))
        
        return snr_db
        
    def _compute_spectral_flatness(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Compute spectral flatness (Wiener entropy)."""
        # Geometric mean / Arithmetic mean
        log_spec = torch.log(spectrogram + 1e-8)
        geometric_mean = torch.exp(torch.mean(log_spec, dim=1))
        arithmetic_mean = torch.mean(spectrogram, dim=1)
        
        flatness = geometric_mean / (arithmetic_mean + 1e-8)
        
        return torch.mean(flatness, dim=-1)  # Average over time
        
    def _compute_crest_factor(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute crest factor (peak-to-RMS ratio)."""
        peak_value = torch.max(torch.abs(audio), dim=-1)[0]
        rms_value = torch.sqrt(torch.mean(audio ** 2, dim=-1))
        
        crest_factor = peak_value / (rms_value + 1e-8)
        
        return crest_factor
        
    def _estimate_thd(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Estimate total harmonic distortion."""
        # Find fundamental frequency and harmonics
        # This is a simplified version
        
        # Find peak frequency in each frame
        peak_bins = torch.argmax(spectrogram, dim=1)
        
        # Estimate THD as ratio of harmonic energy to fundamental
        total_energy = torch.sum(spectrogram, dim=1)
        fundamental_energy = torch.gather(spectrogram, 1, peak_bins.unsqueeze(1)).squeeze(1)
        
        thd_estimate = 1 - (fundamental_energy / (total_energy + 1e-8))
        
        return torch.mean(thd_estimate, dim=-1)  # Average over time
        
    def _detect_temporal_discontinuities(self, audio: torch.Tensor) -> torch.Tensor:
        """Detect temporal discontinuities (clicks, pops)."""
        # High-pass filter to emphasize discontinuities
        high_freq = F.conv1d(
            audio.unsqueeze(1),
            torch.tensor([[-1, 1]], device=audio.device, dtype=audio.dtype).unsqueeze(1),
            padding=1
        ).squeeze(1)
        
        # Count frames with high derivative
        frame_length = 1024
        hop_length = 512
        
        frames = high_freq.unfold(-1, frame_length, hop_length)
        frame_discontinuity = torch.max(torch.abs(frames), dim=-1)[0]
        
        # Threshold for discontinuity detection
        threshold = torch.quantile(frame_discontinuity, 0.95, dim=-1, keepdim=True)
        discontinuity_count = torch.sum(frame_discontinuity > threshold, dim=-1).float()
        
        # Normalize by number of frames
        discontinuity_ratio = discontinuity_count / frame_discontinuity.shape[-1]
        
        return discontinuity_ratio
        
    def _compute_blind_quality_score(self, indicators: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute overall blind quality score from indicators."""
        # Simple heuristic combination
        # In practice, this could be learned from data
        
        quality_score = torch.ones_like(indicators['estimated_snr'])
        
        # SNR contribution (higher is better)
        snr_normalized = torch.sigmoid((indicators['estimated_snr'] - 20) / 10)
        quality_score *= snr_normalized
        
        # Spectral flatness (lower is better for non-noise signals)
        flatness_penalty = torch.sigmoid((0.5 - indicators['spectral_flatness']) * 10)
        quality_score *= flatness_penalty
        
        # Crest factor (moderate values are better)
        crest_penalty = torch.sigmoid(-(torch.abs(indicators['crest_factor'] - 4) - 2))
        quality_score *= crest_penalty
        
        # THD (lower is better)
        thd_penalty = torch.sigmoid((0.1 - indicators['thd_estimate']) * 20)
        quality_score *= thd_penalty
        
        # Temporal discontinuities (lower is better)
        discontinuity_penalty = torch.sigmoid((0.01 - indicators['temporal_discontinuities']) * 100)
        quality_score *= discontinuity_penalty
        
        return quality_score


class PsychoacousticMaskingModel(nn.Module):
    """Psychoacoustic masking model for perceptual quality assessment."""
    
    def __init__(self, sample_rate: int, n_fft: int, n_bark_bands: int):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_bark_bands = n_bark_bands
        
    def forward(self, reference: torch.Tensor, degraded: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute psychoacoustic quality metrics."""
        # Simplified psychoacoustic model
        # In practice, this would implement full ISO/IEC standards
        
        # Compute masking thresholds
        ref_masking = self._compute_masking_threshold(reference)
        deg_masking = self._compute_masking_threshold(degraded)
        
        # Masking threshold error
        masking_error = F.mse_loss(ref_masking, deg_masking)
        
        # Loudness computation (simplified)
        ref_loudness = self._compute_loudness(reference)
        deg_loudness = self._compute_loudness(degraded)
        loudness_error = F.mse_loss(ref_loudness, deg_loudness)
        
        # Overall psychoacoustic quality
        overall_quality = torch.exp(-(masking_error + loudness_error))
        
        return {
            'overall_quality': overall_quality,
            'masking_error': masking_error,
            'loudness_error': loudness_error
        }
        
    def _compute_masking_threshold(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute psychoacoustic masking threshold."""
        # Simplified version - full implementation would follow ISO standards
        
        # STFT
        stft = torch.stft(audio, n_fft=self.n_fft, hop_length=self.n_fft//4, 
                         return_complex=True, window=torch.hann_window(self.n_fft, device=audio.device))
        magnitude = torch.abs(stft)
        
        # Convert to dB
        magnitude_db = 20 * torch.log10(magnitude + 1e-8)
        
        # Apply simple spreading function (convolution with spreading kernel)
        # This is a gross simplification of actual psychoacoustic models
        kernel = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1], device=audio.device).unsqueeze(0).unsqueeze(0)
        
        masking_threshold = F.conv1d(
            magnitude_db.transpose(1, 2),
            kernel,
            padding=2
        ).transpose(1, 2)
        
        return masking_threshold
        
    def _compute_loudness(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loudness."""
        # Simplified loudness computation
        # Real implementation would use standards like ITU-R BS.1770
        
        # RMS energy as proxy for loudness
        frame_length = 1024
        hop_length = 512
        
        frames = audio.unfold(-1, frame_length, hop_length)
        rms_loudness = torch.sqrt(torch.mean(frames ** 2, dim=-1))
        
        # Convert to dB
        loudness_db = 20 * torch.log10(rms_loudness + 1e-8)
        
        return loudness_db


class TemporalQualityAnalyzer(nn.Module):
    """Analyze temporal quality aspects."""
    
    def __init__(self, sample_rate: int, hop_length: int):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        
    def forward(self, reference: torch.Tensor, degraded: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze temporal quality."""
        # Temporal correlation
        temporal_correlation = self._compute_temporal_correlation(reference, degraded)
        
        # Envelope correlation
        envelope_correlation = self._compute_envelope_correlation(reference, degraded)
        
        # Phase coherence
        phase_coherence = self._compute_phase_coherence(reference, degraded)
        
        # Overall temporal quality
        temporal_quality = (temporal_correlation + envelope_correlation + phase_coherence) / 3
        
        return {
            'temporal_quality': temporal_quality,
            'temporal_correlation': temporal_correlation,
            'envelope_correlation': envelope_correlation,
            'phase_coherence': phase_coherence
        }
        
    def _compute_temporal_correlation(self, ref: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
        """Compute temporal correlation between signals."""
        min_len = min(ref.shape[-1], deg.shape[-1])
        ref_aligned = ref[..., :min_len]
        deg_aligned = deg[..., :min_len]
        
        # Normalize signals
        ref_norm = F.normalize(ref_aligned, p=2, dim=-1)
        deg_norm = F.normalize(deg_aligned, p=2, dim=-1)
        
        # Compute correlation
        correlation = torch.sum(ref_norm * deg_norm, dim=-1)
        
        return correlation
        
    def _compute_envelope_correlation(self, ref: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
        """Compute envelope correlation."""
        # Extract envelopes using Hilbert transform approximation
        ref_env = self._extract_envelope(ref)
        deg_env = self._extract_envelope(deg)
        
        min_len = min(ref_env.shape[-1], deg_env.shape[-1])
        ref_env = ref_env[..., :min_len]
        deg_env = deg_env[..., :min_len]
        
        # Compute correlation
        ref_env_norm = F.normalize(ref_env, p=2, dim=-1)
        deg_env_norm = F.normalize(deg_env, p=2, dim=-1)
        
        envelope_corr = torch.sum(ref_env_norm * deg_env_norm, dim=-1)
        
        return envelope_corr
        
    def _compute_phase_coherence(self, ref: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
        """Compute phase coherence."""
        # STFT for phase analysis
        ref_stft = torch.stft(ref, n_fft=512, hop_length=128, return_complex=True,
                             window=torch.hann_window(512, device=ref.device))
        deg_stft = torch.stft(deg, n_fft=512, hop_length=128, return_complex=True,
                             window=torch.hann_window(512, device=deg.device))
        
        # Align dimensions
        min_time = min(ref_stft.shape[-1], deg_stft.shape[-1])
        ref_stft = ref_stft[..., :min_time]
        deg_stft = deg_stft[..., :min_time]
        
        # Phase difference
        ref_phase = torch.angle(ref_stft)
        deg_phase = torch.angle(deg_stft)
        
        phase_diff = torch.abs(ref_phase - deg_phase)
        phase_diff = torch.min(phase_diff, 2 * np.pi - phase_diff)  # Wrap to [0, π]
        
        # Phase coherence (higher is better)
        phase_coherence = 1 - torch.mean(phase_diff, dim=(1, 2)) / np.pi
        
        return phase_coherence
        
    def _extract_envelope(self, signal: torch.Tensor) -> torch.Tensor:
        """Extract signal envelope using moving average."""
        # Simple envelope extraction using absolute value and smoothing
        abs_signal = torch.abs(signal)
        
        # Moving average filter
        kernel_size = 64
        kernel = torch.ones(1, 1, kernel_size, device=signal.device) / kernel_size
        
        envelope = F.conv1d(
            abs_signal.unsqueeze(1),
            kernel,
            padding=kernel_size // 2
        ).squeeze(1)
        
        return envelope


class QualityFeatureExtractor(nn.Module):
    """Multi-scale feature extractor for quality assessment."""
    
    def __init__(self, input_channels: int, hidden_dim: int, num_scales: int):
        super().__init__()
        
        self.scale_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_channels, hidden_dim // (2**i), 
                         kernel_size=(3, 3), padding=(1, 1)),
                nn.ReLU(),
                nn.MaxPool2d((2**(i+1), 2**(i+1))),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear((hidden_dim // (2**i)) * 16, hidden_dim)
            ) for i in range(num_scales)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        for extractor in self.scale_extractors:
            feat = extractor(x)
            features.append(feat)
            
        return torch.cat(features, dim=1)


# Factory functions
def create_speech_quality_assessor(**kwargs) -> PerceptualQualityAssessor:
    """Create quality assessor optimized for speech."""
    return PerceptualQualityAssessor(
        sample_rate=16000,
        quality_metrics=['spectral', 'psychoacoustic', 'temporal'],
        **kwargs
    )

def create_music_quality_assessor(**kwargs) -> PerceptualQualityAssessor:
    """Create quality assessor optimized for music."""
    return PerceptualQualityAssessor(
        sample_rate=44100,
        n_fft=2048,
        hop_length=512,
        quality_metrics=['spectral', 'temporal', 'learned'],
        **kwargs
    )

def create_general_quality_assessor(**kwargs) -> PerceptualQualityAssessor:
    """Create general-purpose quality assessor."""
    return PerceptualQualityAssessor(
        sample_rate=22050,
        quality_metrics=['spectral', 'psychoacoustic', 'temporal', 'learned'],
        content_aware=True,
        **kwargs
    )


# Example usage
if __name__ == "__main__":
    # Create quality assessor
    assessor = create_general_quality_assessor()
    
    # Test with dummy audio
    sample_rate = 22050
    duration = 3  # seconds
    
    reference = torch.randn(2, sample_rate * duration)
    degraded = reference + 0.1 * torch.randn_like(reference)  # Add some noise
    
    # Assess quality
    quality_result = assessor(reference, degraded, return_individual_metrics=True)
    
    print("Quality assessment results:")
    for metric, score in quality_result['quality_scores'].items():
        if isinstance(score, torch.Tensor):
            print(f"  {metric}: {score.mean().item():.3f}")
            
    # Blind quality assessment
    blind_quality = assessor.assess_single_audio_quality(degraded)
    
    print(f"\nBlind quality assessment:")
    print(f"  Estimated SNR: {blind_quality['estimated_snr'].mean().item():.1f} dB")
    print(f"  Blind quality score: {blind_quality['blind_quality_score'].mean().item():.3f}")
    print(f"  Spectral flatness: {blind_quality['spectral_flatness'].mean().item():.3f}")
    print(f"  Crest factor: {blind_quality['crest_factor'].mean().item():.2f}")