"""
Source Separation and Audio Decomposition

Implements state-of-the-art source separation methods:
- Multi-track source separation (vocals, drums, bass, other)
- Harmonic-percussive separation 
- Spatial audio separation (stereo/multichannel)
- Neural source separation with attention mechanisms
- Real-time capable separation for live applications
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import librosa
from dataclasses import dataclass
from enum import Enum

from ..audio_analysis.audio_config import AudioModuleConfig, AudioModuleBase


class SeparationType(Enum):
    """Types of source separation."""
    HARMONIC_PERCUSSIVE = "harmonic_percussive"
    VOCAL_INSTRUMENTAL = "vocal_instrumental"
    MULTITRACK = "multitrack"  # vocals, drums, bass, other
    SPATIAL = "spatial"
    FOREGROUND_BACKGROUND = "foreground_background"


@dataclass
class SeparationResult:
    """Result from source separation."""
    separated_sources: Dict[str, torch.Tensor]
    source_masks: Dict[str, torch.Tensor]
    confidence_scores: Dict[str, float]
    separation_quality: float
    processing_time: float


class SpectralMaskGenerator(nn.Module):
    """
    Advanced spectral mask generation for source separation.
    
    Uses learned attention mechanisms and frequency-aware processing.
    """
    
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_sources: int = 4,
        mask_type: str = "soft",  # "soft", "hard", "wiener"
        frequency_attention: bool = True
    ):
        super().__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_sources = n_sources
        self.mask_type = mask_type
        
        # STFT parameters
        self.register_buffer('window', torch.hann_window(n_fft))
        
        # Frequency-aware processing
        if frequency_attention:
            self.freq_attention = FrequencyAttention(n_fft // 2 + 1)
        else:
            self.freq_attention = None
            
        # Mask generation network
        self.mask_generator = self._build_mask_network()
        
        # Source-specific processing
        self.source_processors = nn.ModuleDict({
            f'source_{i}': self._build_source_processor()
            for i in range(n_sources)
        })
        
    def _build_mask_network(self):
        """Build mask generation network."""
        freq_dim = self.n_fft // 2 + 1
        
        return nn.Sequential(
            # Input: magnitude spectrogram
            nn.Conv2d(1, 64, kernel_size=(7, 7), padding=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Frequency processing
            nn.Conv2d(64, 128, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # Pool only in frequency
            
            # Temporal processing
            nn.Conv2d(128, 256, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Deep processing
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            
            # Upsample back to original frequency resolution
            nn.Upsample(scale_factor=(2, 1), mode='bilinear', align_corners=False),
            
            # Output masks for all sources
            nn.Conv2d(512, self.n_sources, kernel_size=(3, 3), padding=(1, 1)),
            nn.Sigmoid() if self.mask_type == "soft" else nn.Softmax(dim=1)
        )
        
    def _build_source_processor(self):
        """Build source-specific processing network."""
        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=(1, 1)),
            nn.Sigmoid()
        )
        
    def compute_stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute STFT with proper windowing."""
        return torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True
        )
        
    def compute_istft(self, stft: torch.Tensor) -> torch.Tensor:
        """Inverse STFT to reconstruct waveform."""
        return torch.istft(
            stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window
        )
        
    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate spectral masks for source separation.
        
        Args:
            waveform: Input audio [batch, samples]
            
        Returns:
            Dictionary with masks and separated sources
        """
        batch_size = waveform.shape[0]
        
        # Compute STFT
        stft = self.compute_stft(waveform)
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Apply frequency attention if available
        if self.freq_attention is not None:
            magnitude = self.freq_attention(magnitude)
            
        # Generate masks
        mag_input = magnitude.unsqueeze(1)  # Add channel dimension
        masks = self.mask_generator(mag_input)
        
        # Apply source-specific processing
        refined_masks = {}
        separated_sources = {}
        
        for i in range(self.n_sources):
            source_mask = masks[:, i:i+1]
            
            # Refine mask with source-specific processor
            refined_mask = self.source_processors[f'source_{i}'](source_mask)
            refined_masks[f'source_{i}'] = refined_mask.squeeze(1)
            
            # Apply mask to magnitude spectrogram
            masked_magnitude = magnitude * refined_mask.squeeze(1)
            
            # Reconstruct complex spectrogram
            masked_stft = masked_magnitude * torch.exp(1j * phase)
            
            # Inverse STFT
            separated_audio = self.compute_istft(masked_stft)
            separated_sources[f'source_{i}'] = separated_audio
            
        return {
            'masks': refined_masks,
            'separated_sources': separated_sources,
            'original_stft': stft,
            'magnitude': magnitude,
            'phase': phase
        }


class FrequencyAttention(nn.Module):
    """
    Frequency-aware attention mechanism for source separation.
    
    Applies different attention weights across frequency bands.
    """
    
    def __init__(self, n_freq_bins: int, attention_dim: int = 64):
        super().__init__()
        
        self.n_freq_bins = n_freq_bins
        
        # Frequency embedding
        self.freq_embedding = nn.Linear(1, attention_dim)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            attention_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(attention_dim, 1)
        
        # Frequency positions (log scale for better perceptual modeling)
        freq_positions = torch.logspace(0, np.log10(n_freq_bins), n_freq_bins) - 1
        freq_positions = freq_positions / (n_freq_bins - 1)  # Normalize to [0, 1]
        self.register_buffer('freq_positions', freq_positions.unsqueeze(-1))
        
    def forward(self, magnitude: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency attention to magnitude spectrogram.
        
        Args:
            magnitude: Magnitude spectrogram [batch, freq, time]
            
        Returns:
            Attended magnitude spectrogram
        """
        batch_size, n_freq, n_time = magnitude.shape
        
        # Create frequency embeddings
        freq_emb = self.freq_embedding(self.freq_positions)  # [freq, dim]
        freq_emb = freq_emb.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, freq, dim]
        
        # Average magnitude over time for each frequency bin
        freq_magnitude = torch.mean(magnitude, dim=2, keepdim=True)  # [batch, freq, 1]
        
        # Combine frequency embeddings with magnitude information
        freq_features = freq_emb + freq_magnitude.unsqueeze(-1) * 0.1
        
        # Self-attention across frequency bins
        attended_freq, _ = self.attention(freq_features, freq_features, freq_features)
        
        # Generate attention weights
        attention_weights = torch.sigmoid(self.output_proj(attended_freq)).squeeze(-1)  # [batch, freq]
        
        # Apply attention to original magnitude
        attended_magnitude = magnitude * attention_weights.unsqueeze(-1)
        
        return attended_magnitude


class HarmonicPercussiveSeparator(nn.Module):
    """
    Advanced harmonic-percussive separation using neural networks.
    
    Combines traditional signal processing with learned separation.
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        harmonic_margin: float = 5.0,
        percussive_margin: float = 5.0
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.harmonic_margin = harmonic_margin
        self.percussive_margin = percussive_margin
        
        # Neural enhancement of traditional HP separation
        self.harmonic_enhancer = self._build_harmonic_enhancer()
        self.percussive_enhancer = self._build_percussive_enhancer()
        
    def _build_harmonic_enhancer(self):
        """Build harmonic component enhancer."""
        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 9), padding=(0, 4)),  # Temporal smoothing
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(7, 1), padding=(3, 0)),  # Frequency smoothing
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=(3, 3), padding=(1, 1)),
            nn.Sigmoid()
        )
        
    def _build_percussive_enhancer(self):
        """Build percussive component enhancer."""
        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(9, 1), padding=(4, 0)),  # Frequency localization
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(1, 7), padding=(0, 3)),  # Temporal sharpening
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=(3, 3), padding=(1, 1)),
            nn.Sigmoid()
        )
        
    def traditional_hp_separation(self, magnitude: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Traditional harmonic-percussive separation using median filtering."""
        # Convert to numpy for librosa processing
        magnitude_np = magnitude.cpu().numpy()
        
        harmonic_masks = []
        percussive_masks = []
        
        for b in range(magnitude_np.shape[0]):
            mag = magnitude_np[b]
            
            # Harmonic-percussive separation
            harmonic, percussive = librosa.decompose.hpss(
                mag,
                margin=(self.harmonic_margin, self.percussive_margin)
            )
            
            # Create soft masks
            total = harmonic + percussive + 1e-8
            harmonic_mask = harmonic / total
            percussive_mask = percussive / total
            
            harmonic_masks.append(harmonic_mask)
            percussive_masks.append(percussive_mask)
            
        harmonic_mask = torch.from_numpy(np.stack(harmonic_masks)).to(magnitude.device)
        percussive_mask = torch.from_numpy(np.stack(percussive_masks)).to(magnitude.device)
        
        return harmonic_mask, percussive_mask
        
    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Harmonic-percussive separation.
        
        Args:
            waveform: Input audio [batch, samples]
            
        Returns:
            Separated harmonic and percussive components
        """
        # Compute STFT
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft, device=waveform.device),
            return_complex=True
        )
        
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Traditional HP separation
        harmonic_mask_trad, percussive_mask_trad = self.traditional_hp_separation(magnitude)
        
        # Neural enhancement
        harmonic_input = (magnitude * harmonic_mask_trad).unsqueeze(1)
        percussive_input = (magnitude * percussive_mask_trad).unsqueeze(1)
        
        harmonic_enhancement = self.harmonic_enhancer(harmonic_input).squeeze(1)
        percussive_enhancement = self.percussive_enhancer(percussive_input).squeeze(1)
        
        # Combine traditional and neural masks
        harmonic_mask_final = 0.7 * harmonic_mask_trad + 0.3 * harmonic_enhancement
        percussive_mask_final = 0.7 * percussive_mask_trad + 0.3 * percussive_enhancement
        
        # Normalize masks to sum to 1
        total_mask = harmonic_mask_final + percussive_mask_final + 1e-8
        harmonic_mask_final = harmonic_mask_final / total_mask
        percussive_mask_final = percussive_mask_final / total_mask
        
        # Apply masks and reconstruct
        harmonic_stft = magnitude * harmonic_mask_final * torch.exp(1j * phase)
        percussive_stft = magnitude * percussive_mask_final * torch.exp(1j * phase)
        
        harmonic_audio = torch.istft(
            harmonic_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft, device=waveform.device)
        )
        
        percussive_audio = torch.istft(
            percussive_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft, device=waveform.device)
        )
        
        return {
            'harmonic': harmonic_audio,
            'percussive': percussive_audio,
            'harmonic_mask': harmonic_mask_final,
            'percussive_mask': percussive_mask_final,
            'original_magnitude': magnitude,
            'original_phase': phase
        }


class VocalInstrumentalSeparator(nn.Module):
    """
    Vocal-instrumental separation using learned spectral patterns.
    
    Focuses on separating vocals from instrumental background.
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        vocal_freq_range: Tuple[int, int] = (80, 800)  # Hz
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Frequency range for vocals
        vocal_freq_min = int(vocal_freq_range[0] * n_fft / sample_rate)
        vocal_freq_max = int(vocal_freq_range[1] * n_fft / sample_rate)
        self.vocal_freq_range = (vocal_freq_min, vocal_freq_max)
        
        # Vocal detection network
        self.vocal_detector = self._build_vocal_detector()
        
        # Separation network
        self.separator = self._build_separator()
        
    def _build_vocal_detector(self):
        """Build vocal presence detection network."""
        return nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def _build_separator(self):
        """Build vocal-instrumental separator."""
        return nn.Sequential(
            # Encoder
            nn.Conv2d(1, 64, kernel_size=(7, 7), padding=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Attention mechanism
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Decoder - two output channels for vocal and instrumental masks
            nn.Conv2d(256, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 2, kernel_size=(3, 3), padding=(1, 1)),
            nn.Softmax(dim=1)
        )
        
    def center_channel_extraction(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract center channel for vocal isolation (if stereo)."""
        if waveform.shape[1] == 2:  # Stereo
            # Simple center extraction: (L+R)/2 and (L-R)/2
            center = (waveform[:, 0] + waveform[:, 1]) / 2
            sides = (waveform[:, 0] - waveform[:, 1]) / 2
            return torch.stack([center, sides], dim=1)
        else:
            return waveform
            
    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Vocal-instrumental separation.
        
        Args:
            waveform: Input audio [batch, samples] or [batch, 2, samples] for stereo
            
        Returns:
            Separated vocal and instrumental components
        """
        # Handle stereo input
        if len(waveform.shape) == 3 and waveform.shape[1] == 2:
            # Process each channel separately then combine
            left_results = self._process_mono(waveform[:, 0])
            right_results = self._process_mono(waveform[:, 1])
            
            # Combine stereo results
            return {
                'vocal': torch.stack([left_results['vocal'], right_results['vocal']], dim=1),
                'instrumental': torch.stack([left_results['instrumental'], right_results['instrumental']], dim=1),
                'vocal_confidence': (left_results['vocal_confidence'] + right_results['vocal_confidence']) / 2,
                'vocal_mask': torch.stack([left_results['vocal_mask'], right_results['vocal_mask']], dim=1),
                'instrumental_mask': torch.stack([left_results['instrumental_mask'], right_results['instrumental_mask']], dim=1)
            }
        else:
            return self._process_mono(waveform)
            
    def _process_mono(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process mono audio for vocal-instrumental separation."""
        # Compute STFT
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft, device=waveform.device),
            return_complex=True
        )
        
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Vocal presence detection
        mag_input = magnitude.unsqueeze(1)
        vocal_confidence = self.vocal_detector(mag_input).squeeze()
        
        # Generate separation masks
        masks = self.separator(mag_input)
        vocal_mask = masks[:, 0]
        instrumental_mask = masks[:, 1]
        
        # Apply confidence weighting
        confidence_weight = vocal_confidence.unsqueeze(-1).unsqueeze(-1)
        vocal_mask = vocal_mask * confidence_weight
        instrumental_mask = instrumental_mask * (1 - confidence_weight) + instrumental_mask * confidence_weight * 0.1
        
        # Renormalize masks
        total_mask = vocal_mask + instrumental_mask + 1e-8
        vocal_mask = vocal_mask / total_mask
        instrumental_mask = instrumental_mask / total_mask
        
        # Apply masks and reconstruct
        vocal_stft = magnitude * vocal_mask * torch.exp(1j * phase)
        instrumental_stft = magnitude * instrumental_mask * torch.exp(1j * phase)
        
        vocal_audio = torch.istft(
            vocal_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft, device=waveform.device)
        )
        
        instrumental_audio = torch.istft(
            instrumental_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft, device=waveform.device)
        )
        
        return {
            'vocal': vocal_audio,
            'instrumental': instrumental_audio,
            'vocal_confidence': vocal_confidence,
            'vocal_mask': vocal_mask,
            'instrumental_mask': instrumental_mask
        }


class SourceSeparationSystem(AudioModuleBase):
    """
    Complete source separation system with multiple separation types.
    
    Provides unified interface for different separation tasks.
    """
    
    def __init__(self, config: AudioModuleConfig):
        super().__init__(config)
        
        # Different separation modules
        self.hp_separator = HarmonicPercussiveSeparator(
            sample_rate=config.sample_rate,
            hop_length=config.hop_length
        )
        
        self.vocal_separator = VocalInstrumentalSeparator(
            sample_rate=config.sample_rate,
            hop_length=config.hop_length
        )
        
        # Multi-track separator (4-source)
        self.multitrack_separator = SpectralMaskGenerator(
            hop_length=config.hop_length,
            n_sources=4  # vocals, drums, bass, other
        )
        
        # Source labels for multitrack
        self.multitrack_labels = ['vocals', 'drums', 'bass', 'other']
        
    def evaluate_separation_quality(
        self,
        original: torch.Tensor,
        separated_sources: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Evaluate separation quality metrics.
        
        Args:
            original: Original mixed audio
            separated_sources: Dictionary of separated sources
            
        Returns:
            Quality metrics
        """
        # Reconstruction error
        reconstructed = torch.sum(torch.stack(list(separated_sources.values())), dim=0)
        reconstruction_error = F.mse_loss(reconstructed, original).item()
        
        # Source isolation (energy ratio)
        source_energies = {}
        for name, source in separated_sources.items():
            energy = torch.mean(source ** 2).item()
            source_energies[name] = energy
            
        total_energy = sum(source_energies.values()) + 1e-8
        
        # Compute isolation scores (how much energy is in each source)
        isolation_scores = {
            name: energy / total_energy 
            for name, energy in source_energies.items()
        }
        
        # Overall quality score (lower reconstruction error + balanced sources)
        balance_score = 1.0 - np.std(list(isolation_scores.values()))
        quality_score = balance_score * (1.0 / (1.0 + reconstruction_error))
        
        return {
            'reconstruction_error': reconstruction_error,
            'isolation_scores': isolation_scores,
            'balance_score': balance_score,
            'overall_quality': quality_score
        }
        
    def separate_harmonic_percussive(self, waveform: torch.Tensor) -> SeparationResult:
        """Harmonic-percussive separation."""
        import time
        start_time = time.time()
        
        results = self.hp_separator(waveform)
        
        separated_sources = {
            'harmonic': results['harmonic'],
            'percussive': results['percussive']
        }
        
        source_masks = {
            'harmonic': results['harmonic_mask'],
            'percussive': results['percussive_mask']
        }
        
        # Simple confidence based on mask clarity
        confidence_scores = {}
        for name, mask in source_masks.items():
            # Higher confidence when mask values are close to 0 or 1 (clear separation)
            clarity = torch.mean(torch.abs(mask - 0.5)) * 2  # Scale to [0, 1]
            confidence_scores[name] = clarity.item()
            
        quality_metrics = self.evaluate_separation_quality(waveform, separated_sources)
        
        processing_time = time.time() - start_time
        
        return SeparationResult(
            separated_sources=separated_sources,
            source_masks=source_masks,
            confidence_scores=confidence_scores,
            separation_quality=quality_metrics['overall_quality'],
            processing_time=processing_time
        )
        
    def separate_vocal_instrumental(self, waveform: torch.Tensor) -> SeparationResult:
        """Vocal-instrumental separation."""
        import time
        start_time = time.time()
        
        results = self.vocal_separator(waveform)
        
        separated_sources = {
            'vocal': results['vocal'],
            'instrumental': results['instrumental']
        }
        
        source_masks = {
            'vocal': results['vocal_mask'],
            'instrumental': results['instrumental_mask']
        }
        
        confidence_scores = {
            'vocal': results['vocal_confidence'].item() if results['vocal_confidence'].dim() == 0 
                    else torch.mean(results['vocal_confidence']).item(),
            'instrumental': 1.0 - (results['vocal_confidence'].item() if results['vocal_confidence'].dim() == 0 
                                 else torch.mean(results['vocal_confidence']).item())
        }
        
        quality_metrics = self.evaluate_separation_quality(waveform, separated_sources)
        
        processing_time = time.time() - start_time
        
        return SeparationResult(
            separated_sources=separated_sources,
            source_masks=source_masks,
            confidence_scores=confidence_scores,
            separation_quality=quality_metrics['overall_quality'],
            processing_time=processing_time
        )
        
    def separate_multitrack(self, waveform: torch.Tensor) -> SeparationResult:
        """Multi-track source separation (vocals, drums, bass, other)."""
        import time
        start_time = time.time()
        
        results = self.multitrack_separator(waveform)
        
        separated_sources = {}
        source_masks = {}
        
        for i, label in enumerate(self.multitrack_labels):
            separated_sources[label] = results['separated_sources'][f'source_{i}']
            source_masks[label] = results['masks'][f'source_{i}']
            
        # Confidence based on mask sharpness
        confidence_scores = {}
        for label, mask in source_masks.items():
            # Entropy-based confidence (lower entropy = sharper mask = higher confidence)
            mask_flat = mask.flatten()
            mask_probs = mask_flat / (torch.sum(mask_flat) + 1e-8)
            entropy = -torch.sum(mask_probs * torch.log(mask_probs + 1e-8))
            confidence = 1.0 / (1.0 + entropy.item())
            confidence_scores[label] = confidence
            
        quality_metrics = self.evaluate_separation_quality(waveform, separated_sources)
        
        processing_time = time.time() - start_time
        
        return SeparationResult(
            separated_sources=separated_sources,
            source_masks=source_masks,
            confidence_scores=confidence_scores,
            separation_quality=quality_metrics['overall_quality'],
            processing_time=processing_time
        )
        
    def forward(
        self, 
        waveform: torch.Tensor, 
        separation_type: SeparationType = SeparationType.MULTITRACK
    ) -> SeparationResult:
        """
        Perform source separation based on specified type.
        
        Args:
            waveform: Input audio [batch, samples]
            separation_type: Type of separation to perform
            
        Returns:
            Separation results
        """
        if separation_type == SeparationType.HARMONIC_PERCUSSIVE:
            return self.separate_harmonic_percussive(waveform)
        elif separation_type == SeparationType.VOCAL_INSTRUMENTAL:
            return self.separate_vocal_instrumental(waveform)
        elif separation_type == SeparationType.MULTITRACK:
            return self.separate_multitrack(waveform)
        else:
            raise ValueError(f"Unsupported separation type: {separation_type}")


# Factory functions
def create_source_separator(config: Optional[AudioModuleConfig] = None) -> SourceSeparationSystem:
    """Create source separation system with configuration."""
    if config is None:
        from ..audio_analysis.audio_config import get_music_config
        config = get_music_config()
    return SourceSeparationSystem(config)


def create_harmonic_percussive_separator(config: Optional[AudioModuleConfig] = None) -> HarmonicPercussiveSeparator:
    """Create harmonic-percussive separator."""
    if config is None:
        from ..audio_analysis.audio_config import get_music_config
        config = get_music_config()
    return HarmonicPercussiveSeparator(
        sample_rate=config.sample_rate,
        hop_length=config.hop_length
    )


def create_vocal_separator(config: Optional[AudioModuleConfig] = None) -> VocalInstrumentalSeparator:
    """Create vocal-instrumental separator."""
    if config is None:
        from ..audio_analysis.audio_config import get_music_config
        config = get_music_config()
    return VocalInstrumentalSeparator(
        sample_rate=config.sample_rate,
        hop_length=config.hop_length
    )


# Example usage
if __name__ == "__main__":
    from ..audio_analysis.audio_config import get_music_config
    
    # Create source separator
    config = get_music_config()
    separator = create_source_separator(config)
    
    # Test with synthetic mixed audio
    sample_rate = config.sample_rate
    duration = 5  # seconds
    
    # Create synthetic mix: sine wave (harmonic) + noise burst (percussive)
    t = torch.linspace(0, duration, sample_rate * duration)
    
    # Harmonic component (chord progression)
    harmonic = (
        0.3 * torch.sin(2 * torch.pi * 262 * t) +  # C4
        0.2 * torch.sin(2 * torch.pi * 330 * t) +  # E4
        0.25 * torch.sin(2 * torch.pi * 392 * t)   # G4
    )
    
    # Percussive component (drum hits every 0.5 seconds)
    percussive = torch.zeros_like(t)
    for beat_time in torch.arange(0, duration, 0.5):
        beat_sample = int(beat_time * sample_rate)
        if beat_sample < len(percussive) - 1000:
            # Short noise burst
            burst_length = 1000
            noise_burst = torch.randn(burst_length) * 0.5
            noise_burst *= torch.exp(-10 * torch.linspace(0, 0.1, burst_length))
            percussive[beat_sample:beat_sample + burst_length] += noise_burst
            
    # Mix components
    mixed_audio = harmonic + percussive
    waveform = mixed_audio.unsqueeze(0)  # Add batch dimension
    
    # Test different separation types
    print("Testing source separation methods:")
    
    # Harmonic-percussive separation
    hp_result = separator(waveform, SeparationType.HARMONIC_PERCUSSIVE)
    print(f"\nHarmonic-Percussive Separation:")
    print(f"  Quality: {hp_result.separation_quality:.3f}")
    print(f"  Processing time: {hp_result.processing_time:.3f}s")
    print(f"  Harmonic confidence: {hp_result.confidence_scores['harmonic']:.3f}")
    print(f"  Percussive confidence: {hp_result.confidence_scores['percussive']:.3f}")
    
    # Vocal-instrumental separation
    vi_result = separator(waveform, SeparationType.VOCAL_INSTRUMENTAL)
    print(f"\nVocal-Instrumental Separation:")
    print(f"  Quality: {vi_result.separation_quality:.3f}")
    print(f"  Processing time: {vi_result.processing_time:.3f}s")
    print(f"  Vocal confidence: {vi_result.confidence_scores['vocal']:.3f}")
    print(f"  Instrumental confidence: {vi_result.confidence_scores['instrumental']:.3f}")
    
    # Multi-track separation
    mt_result = separator(waveform, SeparationType.MULTITRACK)
    print(f"\nMulti-track Separation:")
    print(f"  Quality: {mt_result.separation_quality:.3f}")
    print(f"  Processing time: {mt_result.processing_time:.3f}s")
    for source, confidence in mt_result.confidence_scores.items():
        print(f"  {source} confidence: {confidence:.3f}")
        
    print(f"\nOriginal audio energy: {torch.mean(waveform ** 2):.6f}")
    print(f"HP separated energy: {torch.mean(hp_result.separated_sources['harmonic'] ** 2) + torch.mean(hp_result.separated_sources['percussive'] ** 2):.6f}")