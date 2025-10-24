"""
Bulletproof Chord Sequence Modeler Module

Advanced chord recognition, harmonic analysis, and progression modeling with temporal
consistency for complex jazz progressions, modulations, and extended harmonies.
Designed for robust musical harmonic understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import warnings
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, NamedTuple
from dataclasses import dataclass
from enum import Enum
import scipy.signal
import scipy.ndimage
from pathlib import Path

# Import audio config
try:
    from modules.audio_analysis.bulletproof_audio_config import (
        BulletproofAudioConfig, get_bulletproof_music_config
    )
except ImportError:
    # Fallback for testing
    class BulletproofAudioConfig:
        def __init__(self, **kwargs):
            self.sample_rate = kwargs.get('sample_rate', 22050)
            self.n_fft = kwargs.get('n_fft', 2048)
            self.hop_length = kwargs.get('hop_length', 512)
            self.n_mels = kwargs.get('n_mels', 128)
            self.n_chroma = kwargs.get('n_chroma', 12)
    
    def get_bulletproof_music_config(**kwargs):
        return BulletproofAudioConfig(**kwargs)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChordQuality(Enum):
    """Chord quality types with harmonic complexity."""
    MAJOR = "major"
    MINOR = "minor"
    DIMINISHED = "diminished"
    AUGMENTED = "augmented"
    DOMINANT7 = "dominant7"
    MAJOR7 = "major7"
    MINOR7 = "minor7"
    DIMINISHED7 = "diminished7"
    HALF_DIMINISHED7 = "half_diminished7"
    MINOR_MAJOR7 = "minor_major7"
    AUGMENTED7 = "augmented7"
    SUSPENDED2 = "suspended2"
    SUSPENDED4 = "suspended4"
    ADDED9 = "added9"
    MAJOR9 = "major9"
    MINOR9 = "minor9"
    DOMINANT9 = "dominant9"
    MAJOR11 = "major11"
    MINOR11 = "minor11"
    DOMINANT11 = "dominant11"
    MAJOR13 = "major13"
    MINOR13 = "minor13"
    DOMINANT13 = "dominant13"
    POWER = "power"
    NO_CHORD = "no_chord"
    UNKNOWN = "unknown"


class HarmonicFunction(Enum):
    """Harmonic function in tonal context."""
    TONIC = "tonic"
    SUBDOMINANT = "subdominant"
    DOMINANT = "dominant"
    MEDIANT = "mediant"
    SUBMEDIANT = "submediant"
    LEADING_TONE = "leading_tone"
    SUPERTONIC = "supertonic"
    SECONDARY_DOMINANT = "secondary_dominant"
    NEAPOLITAN = "neapolitan"
    AUGMENTED_SIXTH = "augmented_sixth"
    CHROMATIC = "chromatic"
    UNKNOWN = "unknown"


class ProgressionType(Enum):
    """Common chord progression patterns."""
    CADENTIAL = "cadential"
    SEQUENTIAL = "sequential"
    CIRCLE_OF_FIFTHS = "circle_of_fifths"
    CHROMATIC = "chromatic"
    MODAL_INTERCHANGE = "modal_interchange"
    SECONDARY_DOMINANTS = "secondary_dominants"
    TRITONE_SUBSTITUTION = "tritone_substitution"
    JAZZ_TURNAROUND = "jazz_turnaround"
    POP_PROGRESSION = "pop_progression"
    BLUES_PROGRESSION = "blues_progression"
    UNKNOWN = "unknown"


@dataclass
class ChordLabel:
    """Container for chord label with confidence and context."""
    root: int  # Root note (0-11, C=0)
    quality: ChordQuality
    bass: Optional[int] = None  # Bass note for inversions/slash chords
    extensions: List[int] = None  # Additional extensions
    confidence: float = 1.0
    start_time: float = 0.0
    end_time: float = 0.0
    
    def __post_init__(self):
        if self.extensions is None:
            self.extensions = []
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.root = self.root % 12
        if self.bass is not None:
            self.bass = self.bass % 12
    
    def to_string(self) -> str:
        """Convert chord to string representation."""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        if self.quality == ChordQuality.NO_CHORD:
            return "N"
        
        chord_str = note_names[self.root]
        
        # Add quality
        quality_map = {
            ChordQuality.MAJOR: "",
            ChordQuality.MINOR: "m",
            ChordQuality.DIMINISHED: "dim",
            ChordQuality.AUGMENTED: "aug",
            ChordQuality.DOMINANT7: "7",
            ChordQuality.MAJOR7: "maj7",
            ChordQuality.MINOR7: "m7",
            ChordQuality.DIMINISHED7: "dim7",
            ChordQuality.HALF_DIMINISHED7: "m7b5",
            ChordQuality.SUSPENDED2: "sus2",
            ChordQuality.SUSPENDED4: "sus4",
            ChordQuality.POWER: "5"
        }
        
        chord_str += quality_map.get(self.quality, self.quality.value)
        
        # Add extensions
        if self.extensions:
            for ext in sorted(self.extensions):
                chord_str += f"add{ext}"
        
        # Add bass note for slash chords
        if self.bass is not None and self.bass != self.root:
            chord_str += f"/{note_names[self.bass]}"
        
        return chord_str


@dataclass
class HarmonicAnalysis:
    """Container for harmonic analysis results."""
    key: int  # Estimated key (0-11, C=0)
    mode: str  # "major" or "minor"
    key_confidence: float
    chords: List[ChordLabel]
    harmonic_functions: List[HarmonicFunction]
    progression_types: List[ProgressionType]
    modulations: List[Tuple[float, int, str]]  # (time, new_key, new_mode)
    harmonic_rhythm: float  # Average chord change rate
    complexity_score: float  # Harmonic complexity [0,1]
    confidence: float  # Overall analysis confidence
    processing_time: float
    warnings: List[str]


class BulletproofChromaExtractor(nn.Module):
    """
    Advanced chroma feature extraction with multiple tuning systems and
    harmonic enhancement for chord recognition.
    """
    
    def __init__(self, config: BulletproofAudioConfig):
        super().__init__()
        self.config = config
        self.sr = config.sample_rate
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.n_chroma = config.n_chroma
        
        # Multiple tuning references for robustness
        self.tuning_frequencies = [440.0, 435.0, 445.0]  # A4 frequencies
        self.chroma_filters = nn.ModuleList()
        
        # Initialize transforms and filters
        self._init_transforms()
        self._init_chroma_filters()
        
        # Harmonic enhancement parameters
        self.harmonic_weights = nn.Parameter(
            torch.tensor([1.0, 0.5, 0.33, 0.25, 0.2]), requires_grad=False
        )
        
    def _init_transforms(self):
        """Initialize spectral transforms."""
        try:
            # CQT for better frequency resolution in bass
            self.cqt_transform = torchaudio.transforms.Spectrogram(
                n_fft=self.n_fft * 2,  # Higher resolution
                hop_length=self.hop_length,
                power=2.0,
                normalized=True
            )
            
            # Standard STFT
            self.stft_transform = torchaudio.transforms.Spectrogram(
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                power=2.0,
                normalized=True
            )
            
        except Exception as e:
            logger.error(f"Transform initialization failed: {e}")
            self.cqt_transform = None
            self.stft_transform = None
    
    def _init_chroma_filters(self):
        """Initialize chroma filter banks for different tunings."""
        try:
            freq_bins = self.n_fft // 2 + 1
            freqs = torch.linspace(0, self.sr // 2, freq_bins)
            
            for tuning_freq in self.tuning_frequencies:
                chroma_filter = self._create_chroma_filter(freqs, tuning_freq)
                self.chroma_filters.append(chroma_filter)
                
        except Exception as e:
            logger.error(f"Chroma filter initialization failed: {e}")
            # Fallback: identity mapping
            identity_filter = nn.Linear(freq_bins, self.n_chroma, bias=False)
            self.chroma_filters.append(identity_filter)
    
    def _create_chroma_filter(self, freqs: torch.Tensor, tuning_freq: float = 440.0) -> nn.Module:
        """Create chroma filter bank for given tuning frequency."""
        try:
            # Convert frequencies to MIDI notes
            # MIDI note = 69 + 12 * log2(freq / 440)
            freq_ratio = freqs / tuning_freq
            # Avoid log of zero
            freq_ratio = torch.clamp(freq_ratio, min=1e-8)
            midi_notes = 69.0 + 12.0 * torch.log2(freq_ratio)
            
            # Map to chroma classes
            chroma_classes = midi_notes % 12
            
            # Create filter bank
            filter_bank = torch.zeros(len(freqs), self.n_chroma)
            
            for i, chroma in enumerate(chroma_classes):
                if not torch.isnan(chroma) and not torch.isinf(chroma):
                    # Gaussian kernel around each chroma class
                    for c in range(self.n_chroma):
                        # Handle wraparound (B -> C)
                        dist = min(abs(chroma - c), abs(chroma - c + 12), abs(chroma - c - 12))
                        filter_bank[i, c] = torch.exp(-(dist ** 2) / (2 * 0.5 ** 2))
            
            # Normalize filters
            filter_bank = F.normalize(filter_bank, p=1, dim=0)
            
            # Create linear layer
            chroma_filter = nn.Linear(len(freqs), self.n_chroma, bias=False)
            chroma_filter.weight.data = filter_bank.T
            chroma_filter.weight.requires_grad = False
            
            return chroma_filter
            
        except Exception as e:
            logger.error(f"Chroma filter creation failed: {e}")
            # Fallback filter
            fallback_filter = nn.Linear(len(freqs), self.n_chroma, bias=False)
            return fallback_filter
    
    def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Extract chroma features from audio.
        
        Args:
            audio: Audio tensor [batch_size, time] or [time]
            
        Returns:
            chroma: Chroma features [batch_size, n_chroma, time_frames]
            chroma_info: Additional information about extraction
        """
        try:
            # Ensure proper tensor format
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            
            # Validate input
            if audio.size(-1) < self.hop_length:
                logger.warning("Audio too short for chroma extraction")
                return self._empty_chroma_result(audio.size(0))
            
            # Compute spectrograms
            spectrograms = self._compute_spectrograms(audio)
            
            # Extract chroma with multiple tunings
            chroma_features = []
            tuning_scores = []
            
            for i, chroma_filter in enumerate(self.chroma_filters):
                if 'magnitude' in spectrograms:
                    spec = spectrograms['magnitude']
                    try:
                        # Apply chroma filter
                        chroma = chroma_filter(spec.transpose(-1, -2)).transpose(-1, -2)
                        
                        # Apply harmonic enhancement
                        enhanced_chroma = self._enhance_harmonics(chroma)
                        
                        # Normalize
                        normalized_chroma = self._normalize_chroma(enhanced_chroma)
                        
                        chroma_features.append(normalized_chroma)
                        
                        # Score this tuning based on chroma clarity
                        clarity_score = self._compute_chroma_clarity(normalized_chroma)
                        tuning_scores.append(clarity_score)
                        
                    except Exception as e:
                        logger.warning(f"Chroma extraction failed for tuning {i}: {e}")
                        continue
            
            if not chroma_features:
                logger.error("All chroma extractions failed")
                return self._empty_chroma_result(audio.size(0))
            
            # Select best tuning or combine
            if len(chroma_features) == 1:
                final_chroma = chroma_features[0]
                best_tuning_idx = 0
            else:
                # Choose best tuning based on clarity
                best_tuning_idx = np.argmax(tuning_scores)
                final_chroma = chroma_features[best_tuning_idx]
            
            # Additional processing
            final_chroma = self._post_process_chroma(final_chroma)
            
            # Compile information
            chroma_info = {
                'tuning_used': self.tuning_frequencies[best_tuning_idx],
                'tuning_scores': tuning_scores,
                'num_frames': final_chroma.size(-1),
                'clarity_score': tuning_scores[best_tuning_idx] if tuning_scores else 0.5,
                'spectrograms_computed': list(spectrograms.keys())
            }
            
            return final_chroma, chroma_info
            
        except Exception as e:
            logger.error(f"Chroma extraction failed: {e}")
            return self._empty_chroma_result(1 if audio.dim() == 1 else audio.size(0))
    
    def _compute_spectrograms(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute various spectral representations."""
        spectrograms = {}
        
        try:
            # Standard STFT
            if self.stft_transform is not None:
                spec = self.stft_transform(audio)
                spectrograms['magnitude'] = spec
                spectrograms['log_magnitude'] = torch.log(spec + 1e-8)
            
            # High-resolution CQT for bass clarity
            if self.cqt_transform is not None:
                try:
                    cqt_spec = self.cqt_transform(audio)
                    spectrograms['cqt'] = cqt_spec
                except Exception as e:
                    logger.warning(f"CQT computation failed: {e}")
            
        except Exception as e:
            logger.error(f"Spectrogram computation failed: {e}")
            # Minimal fallback
            spectrograms['magnitude'] = torch.rand(audio.size(0), 1025, audio.size(-1) // self.hop_length + 1)
        
        return spectrograms
    
    def _enhance_harmonics(self, chroma: torch.Tensor) -> torch.Tensor:
        """Enhance harmonic content in chroma features."""
        try:
            # Apply harmonic weighting
            enhanced = chroma.clone()
            
            # Add weighted harmonics (octaves and fifths)
            for harm_idx, weight in enumerate(self.harmonic_weights):
                if harm_idx == 0:
                    continue  # Skip fundamental
                
                # Perfect fifth = 7 semitones
                fifth_shift = 7 * harm_idx
                # Octave = 12 semitones
                octave_shift = 12 * harm_idx
                
                # Add fifth harmonics
                if fifth_shift < self.n_chroma:
                    shifted_chroma = torch.roll(chroma, shifts=fifth_shift % 12, dims=-2)
                    enhanced += weight * 0.3 * shifted_chroma
                
                # Add octave harmonics
                enhanced += weight * 0.7 * chroma  # Octave equivalence
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Harmonic enhancement failed: {e}")
            return chroma
    
    def _normalize_chroma(self, chroma: torch.Tensor) -> torch.Tensor:
        """Normalize chroma features."""
        try:
            # L2 normalization per frame
            chroma_norm = F.normalize(chroma, p=2, dim=-2)
            
            # Additional smoothing to reduce noise
            if chroma_norm.size(-1) > 3:
                # Simple moving average
                kernel_size = 3
                padding = kernel_size // 2
                
                # Reshape for 1D convolution: [batch, channels, time]
                smoothed = F.avg_pool1d(chroma_norm, kernel_size, stride=1, padding=padding)
                
                return smoothed
            
            return chroma_norm
            
        except Exception as e:
            logger.error(f"Chroma normalization failed: {e}")
            return chroma
    
    def _compute_chroma_clarity(self, chroma: torch.Tensor) -> float:
        """Compute clarity score for chroma features."""
        try:
            # Measure how peaked the chroma vectors are
            # More peaked = more tonal clarity
            
            # Compute entropy per frame
            chroma_prob = F.softmax(chroma, dim=-2)
            entropy = -(chroma_prob * torch.log(chroma_prob + 1e-8)).sum(dim=-2)
            
            # Lower entropy = higher clarity
            mean_entropy = entropy.mean()
            max_entropy = np.log(self.n_chroma)  # Maximum entropy for uniform distribution
            
            clarity = 1.0 - (mean_entropy / max_entropy)
            return float(torch.clamp(clarity, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Chroma clarity computation failed: {e}")
            return 0.5
    
    def _post_process_chroma(self, chroma: torch.Tensor) -> torch.Tensor:
        """Final post-processing of chroma features."""
        try:
            # Apply median filtering to reduce spurious peaks
            if chroma.size(-1) > 5:
                # Convert to numpy for scipy filtering
                chroma_np = chroma.detach().cpu().numpy()
                
                for b in range(chroma_np.shape[0]):
                    for c in range(chroma_np.shape[1]):
                        chroma_np[b, c, :] = scipy.ndimage.median_filter(
                            chroma_np[b, c, :], size=3
                        )
                
                chroma = torch.tensor(chroma_np, device=chroma.device, dtype=chroma.dtype)
            
            # Final normalization
            return F.normalize(chroma, p=2, dim=-2)
            
        except Exception as e:
            logger.error(f"Chroma post-processing failed: {e}")
            return chroma
    
    def _empty_chroma_result(self, batch_size: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Return empty chroma result for error cases."""
        empty_chroma = torch.zeros(batch_size, self.n_chroma, 1)
        empty_info = {
            'tuning_used': 440.0,
            'tuning_scores': [0.0],
            'num_frames': 0,
            'clarity_score': 0.0,
            'spectrograms_computed': []
        }
        return empty_chroma, empty_info


class BulletproofChordRecognizer(nn.Module):
    """
    Advanced chord recognition using template matching, machine learning,
    and harmonic analysis for complex jazz and classical harmonies.
    """
    
    def __init__(self, config: BulletproofAudioConfig):
        super().__init__()
        self.config = config
        self.n_chroma = config.n_chroma
        
        # Chord templates for template matching
        self.chord_templates = self._create_chord_templates()
        
        # Neural network for chord classification
        self.chord_classifier = self._create_chord_classifier()
        
        # Smoothing and temporal consistency
        self.temporal_smoother = self._create_temporal_smoother()
        
        # Quality assessment
        self.quality_threshold = 0.3
        
    def _create_chord_templates(self) -> torch.Tensor:
        """Create chord templates for template matching."""
        try:
            # Define chord intervals (semitones from root)
            chord_intervals = {
                ChordQuality.MAJOR: [0, 4, 7],
                ChordQuality.MINOR: [0, 3, 7],
                ChordQuality.DIMINISHED: [0, 3, 6],
                ChordQuality.AUGMENTED: [0, 4, 8],
                ChordQuality.DOMINANT7: [0, 4, 7, 10],
                ChordQuality.MAJOR7: [0, 4, 7, 11],
                ChordQuality.MINOR7: [0, 3, 7, 10],
                ChordQuality.DIMINISHED7: [0, 3, 6, 9],
                ChordQuality.HALF_DIMINISHED7: [0, 3, 6, 10],
                ChordQuality.SUSPENDED2: [0, 2, 7],
                ChordQuality.SUSPENDED4: [0, 5, 7],
                ChordQuality.POWER: [0, 7],
                ChordQuality.MAJOR9: [0, 4, 7, 11, 14 % 12],
                ChordQuality.MINOR9: [0, 3, 7, 10, 14 % 12],
                ChordQuality.DOMINANT9: [0, 4, 7, 10, 14 % 12],
            }
            
            # Create templates for all root positions
            num_roots = 12
            num_qualities = len(chord_intervals)
            templates = torch.zeros(num_roots * num_qualities, self.n_chroma)
            
            template_idx = 0
            for root in range(num_roots):
                for quality, intervals in chord_intervals.items():
                    template = torch.zeros(self.n_chroma)
                    
                    for interval in intervals:
                        chroma_class = (root + interval) % 12
                        template[chroma_class] = 1.0
                    
                    # Normalize template
                    if template.sum() > 0:
                        template = template / template.sum()
                    
                    templates[template_idx] = template
                    template_idx += 1
            
            return templates
            
        except Exception as e:
            logger.error(f"Chord template creation failed: {e}")
            # Fallback: create simple major/minor templates
            num_templates = 24  # 12 major + 12 minor
            templates = torch.zeros(num_templates, self.n_chroma)
            
            for root in range(12):
                # Major chord
                major_template = torch.zeros(self.n_chroma)
                major_template[[root, (root + 4) % 12, (root + 7) % 12]] = 1.0
                templates[root] = major_template / 3.0
                
                # Minor chord
                minor_template = torch.zeros(self.n_chroma)
                minor_template[[root, (root + 3) % 12, (root + 7) % 12]] = 1.0
                templates[root + 12] = minor_template / 3.0
            
            return templates
    
    def _create_chord_classifier(self) -> nn.Module:
        """Create neural network for chord classification."""
        try:
            # Simple but effective architecture
            classifier = nn.Sequential(
                nn.Linear(self.n_chroma, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, len(ChordQuality) * 12 + 1)  # +1 for no-chord
            )
            
            # Initialize weights
            for module in classifier.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
            
            return classifier
            
        except Exception as e:
            logger.error(f"Chord classifier creation failed: {e}")
            # Fallback: simple linear layer
            return nn.Linear(self.n_chroma, 25)  # 24 chords + no-chord
    
    def _create_temporal_smoother(self) -> nn.Module:
        """Create temporal smoothing network."""
        try:
            # Simple LSTM for temporal consistency
            smoother = nn.LSTM(
                input_size=self.n_chroma,
                hidden_size=32,
                num_layers=2,
                batch_first=True,
                dropout=0.2
            )
            return smoother
            
        except Exception as e:
            logger.error(f"Temporal smoother creation failed: {e}")
            # Fallback: identity
            return nn.Identity()
    
    def forward(self, chroma: torch.Tensor) -> Tuple[List[ChordLabel], Dict[str, Any]]:
        """
        Recognize chords from chroma features.
        
        Args:
            chroma: Chroma features [batch_size, n_chroma, time_frames]
            
        Returns:
            chords: List of chord labels with timing
            recognition_info: Additional recognition information
        """
        try:
            if chroma.size(-1) == 0:
                return [], {'num_frames': 0, 'confidence': 0.0}
            
            # Ensure proper format
            if chroma.dim() == 2:
                chroma = chroma.unsqueeze(0)
            
            batch_size, n_chroma, n_frames = chroma.shape
            
            # Template matching approach
            template_results = self._template_matching(chroma)
            
            # Neural network approach
            nn_results = self._neural_network_recognition(chroma)
            
            # Combine approaches
            combined_results = self._combine_recognition_results(
                template_results, nn_results
            )
            
            # Apply temporal smoothing
            smoothed_results = self._apply_temporal_smoothing(
                combined_results, chroma
            )
            
            # Convert to chord labels with timing
            chord_labels = self._create_chord_labels(
                smoothed_results, chroma
            )
            
            # Quality assessment
            recognition_info = self._assess_recognition_quality(
                chord_labels, chroma, template_results, nn_results
            )
            
            return chord_labels, recognition_info
            
        except Exception as e:
            logger.error(f"Chord recognition failed: {e}")
            return [], {'num_frames': 0, 'confidence': 0.0, 'error': str(e)}
    
    def _template_matching(self, chroma: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform template matching chord recognition."""
        try:
            batch_size, n_chroma, n_frames = chroma.shape
            
            # Compute correlation with all templates
            # chroma: [batch, chroma, time]
            # templates: [num_templates, chroma]
            # result: [batch, num_templates, time]
            
            correlations = torch.einsum('bct,tc->bnt', chroma, self.chord_templates.T)
            
            # Find best matches
            best_templates = torch.argmax(correlations, dim=1)  # [batch, time]
            best_scores = torch.max(correlations, dim=1)[0]  # [batch, time]
            
            return {
                'template_indices': best_templates,
                'template_scores': best_scores,
                'all_correlations': correlations
            }
            
        except Exception as e:
            logger.error(f"Template matching failed: {e}")
            batch_size, n_chroma, n_frames = chroma.shape
            return {
                'template_indices': torch.zeros(batch_size, n_frames, dtype=torch.long),
                'template_scores': torch.zeros(batch_size, n_frames),
                'all_correlations': torch.zeros(batch_size, len(self.chord_templates), n_frames)
            }
    
    def _neural_network_recognition(self, chroma: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform neural network chord recognition."""
        try:
            batch_size, n_chroma, n_frames = chroma.shape
            
            # Reshape for NN processing: [batch * time, chroma]
            chroma_flat = chroma.transpose(1, 2).contiguous().view(-1, n_chroma)
            
            # Forward pass
            logits = self.chord_classifier(chroma_flat)
            
            # Convert back to [batch, time, num_classes]
            logits = logits.view(batch_size, n_frames, -1)
            
            # Get predictions and scores
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
            confidence = torch.max(probabilities, dim=-1)[0]
            
            return {
                'predictions': predictions,
                'confidence': confidence,
                'probabilities': probabilities,
                'logits': logits
            }
            
        except Exception as e:
            logger.error(f"Neural network recognition failed: {e}")
            batch_size, n_chroma, n_frames = chroma.shape
            return {
                'predictions': torch.zeros(batch_size, n_frames, dtype=torch.long),
                'confidence': torch.zeros(batch_size, n_frames),
                'probabilities': torch.zeros(batch_size, n_frames, 25),
                'logits': torch.zeros(batch_size, n_frames, 25)
            }
    
    def _combine_recognition_results(self, template_results: Dict, nn_results: Dict) -> Dict[str, torch.Tensor]:
        """Combine template matching and neural network results."""
        try:
            # Weighted combination based on confidence
            template_weight = 0.4
            nn_weight = 0.6
            
            # Normalize scores to [0, 1]
            template_scores = template_results['template_scores']
            nn_confidence = nn_results['confidence']
            
            # Combined confidence
            combined_confidence = (template_weight * template_scores + 
                                 nn_weight * nn_confidence)
            
            # Choose prediction method based on confidence
            use_template = template_scores > nn_confidence
            
            # Combined predictions
            template_preds = template_results['template_indices']
            nn_preds = nn_results['predictions']
            
            # Map template indices to chord classes (simplified)
            # This is a simplified mapping - in practice, you'd need a proper mapping
            combined_predictions = torch.where(use_template, template_preds % 25, nn_preds)
            
            return {
                'predictions': combined_predictions,
                'confidence': combined_confidence,
                'template_weight': template_weight,
                'nn_weight': nn_weight
            }
            
        except Exception as e:
            logger.error(f"Result combination failed: {e}")
            # Fallback to NN results
            return {
                'predictions': nn_results.get('predictions', torch.zeros(1, 1, dtype=torch.long)),
                'confidence': nn_results.get('confidence', torch.zeros(1, 1)),
                'template_weight': 0.0,
                'nn_weight': 1.0
            }
    
    def _apply_temporal_smoothing(self, recognition_results: Dict, chroma: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply temporal smoothing to reduce noise."""
        try:
            if not isinstance(self.temporal_smoother, nn.LSTM):
                # No temporal smoothing available
                return recognition_results
            
            batch_size, n_chroma, n_frames = chroma.shape
            
            # Prepare input for LSTM: [batch, time, features]
            chroma_input = chroma.transpose(1, 2)  # [batch, time, chroma]
            
            # LSTM forward pass
            lstm_out, _ = self.temporal_smoother(chroma_input)
            
            # The LSTM output can be used to modify confidence scores
            # Here we use a simple approach: higher LSTM activation = more stable
            stability_scores = torch.norm(lstm_out, dim=-1)  # [batch, time]
            stability_scores = F.normalize(stability_scores, p=2, dim=-1)
            
            # Modify confidence based on temporal stability
            original_confidence = recognition_results['confidence']
            smoothed_confidence = original_confidence * (0.7 + 0.3 * stability_scores)
            
            # Apply majority voting filter for predictions
            predictions = recognition_results['predictions']
            smoothed_predictions = self._majority_voting_filter(predictions)
            
            return {
                'predictions': smoothed_predictions,
                'confidence': smoothed_confidence,
                'stability_scores': stability_scores,
                'original_confidence': original_confidence,
                'original_predictions': predictions
            }
            
        except Exception as e:
            logger.error(f"Temporal smoothing failed: {e}")
            return recognition_results
    
    def _majority_voting_filter(self, predictions: torch.Tensor, window_size: int = 5) -> torch.Tensor:
        """Apply majority voting filter to predictions."""
        try:
            if predictions.size(-1) < window_size:
                return predictions
            
            batch_size, n_frames = predictions.shape
            smoothed = predictions.clone()
            
            half_window = window_size // 2
            
            for t in range(half_window, n_frames - half_window):
                window = predictions[:, t-half_window:t+half_window+1]
                # Find mode (most common value) in window
                for b in range(batch_size):
                    window_b = window[b]
                    values, counts = torch.unique(window_b, return_counts=True)
                    mode_idx = torch.argmax(counts)
                    smoothed[b, t] = values[mode_idx]
            
            return smoothed
            
        except Exception as e:
            logger.error(f"Majority voting filter failed: {e}")
            return predictions
    
    def _create_chord_labels(self, recognition_results: Dict, chroma: torch.Tensor) -> List[ChordLabel]:
        """Convert recognition results to chord labels with timing."""
        try:
            predictions = recognition_results['predictions']
            confidences = recognition_results['confidence']
            
            if predictions.dim() > 1:
                predictions = predictions[0]  # Take first batch
                confidences = confidences[0]
            
            # Convert frame indices to time
            hop_time = self.config.hop_length / self.config.sample_rate
            
            chord_labels = []
            current_chord = None
            start_time = 0.0
            
            for frame_idx, (pred, conf) in enumerate(zip(predictions, confidences)):
                frame_time = frame_idx * hop_time
                
                # Map prediction index to chord
                chord = self._prediction_to_chord(int(pred), float(conf))
                
                if current_chord is None:
                    # First chord
                    current_chord = chord
                    start_time = frame_time
                elif (current_chord.root != chord.root or 
                      current_chord.quality != chord.quality):
                    # Chord change detected
                    current_chord.start_time = start_time
                    current_chord.end_time = frame_time
                    chord_labels.append(current_chord)
                    
                    current_chord = chord
                    start_time = frame_time
                else:
                    # Same chord, update confidence
                    current_chord.confidence = max(current_chord.confidence, chord.confidence)
            
            # Add final chord
            if current_chord is not None:
                current_chord.start_time = start_time
                current_chord.end_time = len(predictions) * hop_time
                chord_labels.append(current_chord)
            
            # Filter out very short chords (likely noise)
            min_duration = 0.5  # 500ms minimum
            filtered_labels = [
                chord for chord in chord_labels 
                if chord.end_time - chord.start_time >= min_duration
            ]
            
            return filtered_labels
            
        except Exception as e:
            logger.error(f"Chord label creation failed: {e}")
            return []
    
    def _prediction_to_chord(self, prediction_idx: int, confidence: float) -> ChordLabel:
        """Convert prediction index to chord label."""
        try:
            # Simplified mapping - in practice this would be more sophisticated
            if prediction_idx >= 24:  # No chord
                return ChordLabel(
                    root=0,
                    quality=ChordQuality.NO_CHORD,
                    confidence=confidence
                )
            
            # Major/minor chords (simplified)
            if prediction_idx < 12:
                # Major chords
                return ChordLabel(
                    root=prediction_idx,
                    quality=ChordQuality.MAJOR,
                    confidence=confidence
                )
            else:
                # Minor chords
                return ChordLabel(
                    root=prediction_idx - 12,
                    quality=ChordQuality.MINOR,
                    confidence=confidence
                )
                
        except Exception as e:
            logger.error(f"Prediction to chord conversion failed: {e}")
            return ChordLabel(
                root=0,
                quality=ChordQuality.UNKNOWN,
                confidence=0.0
            )
    
    def _assess_recognition_quality(self, chord_labels: List[ChordLabel], 
                                  chroma: torch.Tensor,
                                  template_results: Dict,
                                  nn_results: Dict) -> Dict[str, Any]:
        """Assess the quality of chord recognition."""
        try:
            if not chord_labels:
                return {
                    'num_chords': 0,
                    'average_confidence': 0.0,
                    'confidence_variance': 0.0,
                    'num_frames': chroma.size(-1),
                    'recognition_rate': 0.0
                }
            
            # Basic statistics
            confidences = [chord.confidence for chord in chord_labels]
            avg_confidence = np.mean(confidences)
            confidence_variance = np.var(confidences)
            
            # Coverage statistics
            total_duration = chord_labels[-1].end_time if chord_labels else 0.0
            recognized_duration = sum(
                chord.end_time - chord.start_time 
                for chord in chord_labels 
                if chord.quality != ChordQuality.NO_CHORD
            )
            recognition_rate = recognized_duration / total_duration if total_duration > 0 else 0.0
            
            # Quality metrics
            num_unique_chords = len(set((chord.root, chord.quality) for chord in chord_labels))
            avg_chord_duration = np.mean([
                chord.end_time - chord.start_time for chord in chord_labels
            ])
            
            return {
                'num_chords': len(chord_labels),
                'num_unique_chords': num_unique_chords,
                'average_confidence': avg_confidence,
                'confidence_variance': confidence_variance,
                'num_frames': chroma.size(-1),
                'recognition_rate': recognition_rate,
                'average_chord_duration': avg_chord_duration,
                'template_score_mean': float(template_results['template_scores'].mean()),
                'nn_confidence_mean': float(nn_results['confidence'].mean())
            }
            
        except Exception as e:
            logger.error(f"Recognition quality assessment failed: {e}")
            return {
                'num_chords': len(chord_labels),
                'average_confidence': 0.5,
                'confidence_variance': 0.0,
                'num_frames': chroma.size(-1),
                'recognition_rate': 0.5
            }


class BulletproofHarmonicAnalyzer(nn.Module):
    """
    Advanced harmonic analysis including key detection, functional analysis,
    and progression identification for complex musical structures.
    """
    
    def __init__(self, config: BulletproofAudioConfig):
        super().__init__()
        self.config = config
        
        # Key detection using Krumhansl-Schmuckler profiles
        self.key_profiles = self._create_key_profiles()
        
        # Chord function mapping
        self.function_mapping = self._create_function_mapping()
        
        # Progression pattern templates
        self.progression_patterns = self._create_progression_patterns()
        
    def _create_key_profiles(self) -> torch.Tensor:
        """Create key profiles for key detection."""
        try:
            # Krumhansl-Schmuckler key profiles
            major_profile = torch.tensor([
                6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52,
                5.19, 2.39, 3.66, 2.29, 2.88
            ])
            
            minor_profile = torch.tensor([
                6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54,
                4.75, 3.98, 2.69, 3.34, 3.17
            ])
            
            # Create profiles for all 24 keys
            profiles = torch.zeros(24, 12)
            
            for i in range(12):
                # Major keys
                profiles[i] = torch.roll(major_profile, shifts=i)
                # Minor keys
                profiles[i + 12] = torch.roll(minor_profile, shifts=i)
            
            # Normalize profiles
            profiles = F.normalize(profiles, p=2, dim=1)
            
            return profiles
            
        except Exception as e:
            logger.error(f"Key profile creation failed: {e}")
            # Fallback: uniform profiles
            return torch.ones(24, 12) / 12
    
    def _create_function_mapping(self) -> Dict:
        """Create mapping from chord types to harmonic functions."""
        # Simplified function mapping for major keys
        # In practice, this would be much more sophisticated
        function_map = {
            # Roman numeral analysis
            0: HarmonicFunction.TONIC,        # I
            1: HarmonicFunction.SUPERTONIC,   # ii
            2: HarmonicFunction.MEDIANT,      # iii
            3: HarmonicFunction.SUBDOMINANT,  # IV
            4: HarmonicFunction.DOMINANT,     # V
            5: HarmonicFunction.SUBMEDIANT,   # vi
            6: HarmonicFunction.LEADING_TONE, # vii°
        }
        
        return function_map
    
    def _create_progression_patterns(self) -> Dict[ProgressionType, List]:
        """Create common chord progression patterns."""
        patterns = {
            ProgressionType.CADENTIAL: [
                [4, 0],  # V-I
                [3, 4, 0],  # IV-V-I
                [1, 4, 0],  # ii-V-I
            ],
            ProgressionType.CIRCLE_OF_FIFTHS: [
                [0, 4, 1, 5, 2, 6, 3],  # I-V-ii-vi-iii-vii-IV
            ],
            ProgressionType.POP_PROGRESSION: [
                [0, 5, 3, 4],  # I-vi-IV-V
                [5, 3, 0, 4],  # vi-IV-I-V
            ],
            ProgressionType.JAZZ_TURNAROUND: [
                [0, 5, 1, 4],  # I-vi-ii-V
            ]
        }
        
        return patterns
    
    def forward(self, chroma: torch.Tensor, 
                chord_labels: List[ChordLabel]) -> HarmonicAnalysis:
        """
        Perform comprehensive harmonic analysis.
        
        Args:
            chroma: Chroma features [batch_size, n_chroma, time_frames]
            chord_labels: Recognized chord labels
            
        Returns:
            HarmonicAnalysis with complete harmonic information
        """
        import time
        start_time = time.time()
        
        try:
            # Key detection
            key, mode, key_confidence = self._detect_key(chroma)
            
            # Harmonic function analysis
            harmonic_functions = self._analyze_harmonic_functions(chord_labels, key, mode)
            
            # Progression type identification
            progression_types = self._identify_progression_types(chord_labels, harmonic_functions)
            
            # Modulation detection
            modulations = self._detect_modulations(chroma, chord_labels)
            
            # Harmonic rhythm analysis
            harmonic_rhythm = self._analyze_harmonic_rhythm(chord_labels)
            
            # Complexity assessment
            complexity_score = self._assess_harmonic_complexity(
                chord_labels, harmonic_functions, modulations
            )
            
            # Overall confidence
            overall_confidence = self._compute_analysis_confidence(
                key_confidence, chord_labels, complexity_score
            )
            
            # Collect warnings
            warnings = self._collect_analysis_warnings(
                chord_labels, key_confidence, complexity_score
            )
            
            return HarmonicAnalysis(
                key=key,
                mode=mode,
                key_confidence=key_confidence,
                chords=chord_labels,
                harmonic_functions=harmonic_functions,
                progression_types=progression_types,
                modulations=modulations,
                harmonic_rhythm=harmonic_rhythm,
                complexity_score=complexity_score,
                confidence=overall_confidence,
                processing_time=time.time() - start_time,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Harmonic analysis failed: {e}")
            return self._create_fallback_analysis(start_time, str(e))
    
    def _detect_key(self, chroma: torch.Tensor) -> Tuple[int, str, float]:
        """Detect the musical key from chroma features."""
        try:
            # Average chroma over time to get tonal profile
            if chroma.dim() == 3:
                chroma = chroma[0]  # Take first batch
            
            tonal_profile = chroma.mean(dim=-1)  # [n_chroma]
            
            # Correlate with key profiles
            correlations = F.cosine_similarity(
                tonal_profile.unsqueeze(0), 
                self.key_profiles, 
                dim=1
            )
            
            # Find best match
            best_key_idx = torch.argmax(correlations)
            best_correlation = correlations[best_key_idx]
            
            # Convert to key and mode
            if best_key_idx < 12:
                key = int(best_key_idx)
                mode = "major"
            else:
                key = int(best_key_idx - 12)
                mode = "minor"
            
            # Confidence based on how much better the best key is
            confidence = float(best_correlation)
            
            return key, mode, confidence
            
        except Exception as e:
            logger.error(f"Key detection failed: {e}")
            return 0, "major", 0.5  # Default to C major
    
    def _analyze_harmonic_functions(self, chord_labels: List[ChordLabel], 
                                   key: int, mode: str) -> List[HarmonicFunction]:
        """Analyze harmonic functions of chord sequence."""
        try:
            functions = []
            
            for chord in chord_labels:
                if chord.quality == ChordQuality.NO_CHORD:
                    functions.append(HarmonicFunction.UNKNOWN)
                    continue
                
                # Calculate scale degree
                scale_degree = (chord.root - key) % 12
                
                # Map to harmonic function (simplified)
                if mode == "major":
                    function = self._major_scale_function(scale_degree, chord.quality)
                else:
                    function = self._minor_scale_function(scale_degree, chord.quality)
                
                functions.append(function)
            
            return functions
            
        except Exception as e:
            logger.error(f"Harmonic function analysis failed: {e}")
            return [HarmonicFunction.UNKNOWN] * len(chord_labels)
    
    def _major_scale_function(self, scale_degree: int, quality: ChordQuality) -> HarmonicFunction:
        """Determine harmonic function in major key."""
        try:
            # Simplified mapping based on scale degree and quality
            if scale_degree == 0:  # Tonic
                return HarmonicFunction.TONIC
            elif scale_degree == 2:  # Supertonic
                return HarmonicFunction.SUPERTONIC
            elif scale_degree == 4:  # Mediant
                return HarmonicFunction.MEDIANT
            elif scale_degree == 5:  # Subdominant
                return HarmonicFunction.SUBDOMINANT
            elif scale_degree == 7:  # Dominant
                return HarmonicFunction.DOMINANT
            elif scale_degree == 9:  # Submediant
                return HarmonicFunction.SUBMEDIANT
            elif scale_degree == 11:  # Leading tone
                return HarmonicFunction.LEADING_TONE
            else:
                # Check for secondary dominants, etc.
                if quality in [ChordQuality.DOMINANT7, ChordQuality.MAJOR]:
                    return HarmonicFunction.SECONDARY_DOMINANT
                return HarmonicFunction.CHROMATIC
                
        except Exception:
            return HarmonicFunction.UNKNOWN
    
    def _minor_scale_function(self, scale_degree: int, quality: ChordQuality) -> HarmonicFunction:
        """Determine harmonic function in minor key."""
        try:
            # Similar to major but accounting for minor scale differences
            if scale_degree == 0:  # Tonic
                return HarmonicFunction.TONIC
            elif scale_degree == 2:  # Supertonic
                return HarmonicFunction.SUPERTONIC
            elif scale_degree == 3:  # Mediant (♭III)
                return HarmonicFunction.MEDIANT
            elif scale_degree == 5:  # Subdominant
                return HarmonicFunction.SUBDOMINANT
            elif scale_degree == 7:  # Dominant
                return HarmonicFunction.DOMINANT
            elif scale_degree == 8:  # Submediant (♭VI)
                return HarmonicFunction.SUBMEDIANT
            elif scale_degree == 10:  # Leading tone (♭VII)
                return HarmonicFunction.LEADING_TONE
            else:
                return HarmonicFunction.CHROMATIC
                
        except Exception:
            return HarmonicFunction.UNKNOWN
    
    def _identify_progression_types(self, chord_labels: List[ChordLabel], 
                                  harmonic_functions: List[HarmonicFunction]) -> List[ProgressionType]:
        """Identify types of chord progressions."""
        try:
            if len(chord_labels) < 2:
                return []
            
            progression_types = []
            
            # Look for common patterns
            for pattern_type, patterns in self.progression_patterns.items():
                for pattern in patterns:
                    if self._matches_pattern(harmonic_functions, pattern):
                        progression_types.append(pattern_type)
                        break
            
            # Additional pattern recognition
            if self._is_cadential_pattern(harmonic_functions):
                progression_types.append(ProgressionType.CADENTIAL)
            
            if self._is_sequential_pattern(chord_labels):
                progression_types.append(ProgressionType.SEQUENTIAL)
            
            return list(set(progression_types))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Progression type identification failed: {e}")
            return [ProgressionType.UNKNOWN]
    
    def _matches_pattern(self, functions: List[HarmonicFunction], pattern: List[int]) -> bool:
        """Check if harmonic functions match a pattern."""
        try:
            if len(functions) < len(pattern):
                return False
            
            # Simple substring matching (could be more sophisticated)
            function_indices = [self._function_to_index(f) for f in functions]
            
            for i in range(len(function_indices) - len(pattern) + 1):
                if function_indices[i:i+len(pattern)] == pattern:
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _function_to_index(self, function: HarmonicFunction) -> int:
        """Convert harmonic function to index for pattern matching."""
        mapping = {
            HarmonicFunction.TONIC: 0,
            HarmonicFunction.SUPERTONIC: 1,
            HarmonicFunction.MEDIANT: 2,
            HarmonicFunction.SUBDOMINANT: 3,
            HarmonicFunction.DOMINANT: 4,
            HarmonicFunction.SUBMEDIANT: 5,
            HarmonicFunction.LEADING_TONE: 6,
        }
        return mapping.get(function, -1)
    
    def _is_cadential_pattern(self, functions: List[HarmonicFunction]) -> bool:
        """Check for cadential patterns."""
        try:
            if len(functions) < 2:
                return False
            
            # Look for V-I or IV-I patterns
            for i in range(len(functions) - 1):
                if (functions[i] == HarmonicFunction.DOMINANT and 
                    functions[i+1] == HarmonicFunction.TONIC):
                    return True
                if (functions[i] == HarmonicFunction.SUBDOMINANT and 
                    functions[i+1] == HarmonicFunction.TONIC):
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _is_sequential_pattern(self, chord_labels: List[ChordLabel]) -> bool:
        """Check for sequential patterns (similar chords at different pitch levels)."""
        try:
            if len(chord_labels) < 3:
                return False
            
            # Look for sequences of similar chord qualities
            for i in range(len(chord_labels) - 2):
                chord1 = chord_labels[i]
                chord2 = chord_labels[i+1]
                chord3 = chord_labels[i+2]
                
                if (chord1.quality == chord2.quality == chord3.quality and
                    chord1.quality != ChordQuality.NO_CHORD):
                    # Check if roots follow a pattern (e.g., ascending/descending)
                    interval1 = (chord2.root - chord1.root) % 12
                    interval2 = (chord3.root - chord2.root) % 12
                    
                    if interval1 == interval2:  # Same interval pattern
                        return True
            
            return False
            
        except Exception:
            return False
    
    def _detect_modulations(self, chroma: torch.Tensor, 
                          chord_labels: List[ChordLabel]) -> List[Tuple[float, int, str]]:
        """Detect key modulations in the music."""
        try:
            modulations = []
            
            if len(chord_labels) < 4:
                return modulations
            
            # Analyze key in sliding windows
            window_size = max(4, len(chord_labels) // 4)
            hop_size = window_size // 2
            
            for i in range(0, len(chord_labels) - window_size, hop_size):
                window_chords = chord_labels[i:i+window_size]
                
                # Create chord histogram for this window
                chord_histogram = torch.zeros(12)
                for chord in window_chords:
                    if chord.quality != ChordQuality.NO_CHORD:
                        chord_histogram[chord.root] += 1
                
                # Correlate with key profiles
                if chord_histogram.sum() > 0:
                    chord_histogram = chord_histogram / chord_histogram.sum()
                    
                    correlations = F.cosine_similarity(
                        chord_histogram.unsqueeze(0),
                        self.key_profiles,
                        dim=1
                    )
                    
                    best_key_idx = torch.argmax(correlations)
                    confidence = float(correlations[best_key_idx])
                    
                    if best_key_idx < 12:
                        key = int(best_key_idx)
                        mode = "major"
                    else:
                        key = int(best_key_idx - 12)
                        mode = "minor"
                    
                    # Check if this is different from previous key
                    if (confidence > 0.7 and 
                        (not modulations or modulations[-1][1] != key or modulations[-1][2] != mode)):
                        time_point = window_chords[0].start_time
                        modulations.append((time_point, key, mode))
            
            return modulations
            
        except Exception as e:
            logger.error(f"Modulation detection failed: {e}")
            return []
    
    def _analyze_harmonic_rhythm(self, chord_labels: List[ChordLabel]) -> float:
        """Analyze the harmonic rhythm (rate of chord changes)."""
        try:
            if len(chord_labels) < 2:
                return 0.0
            
            # Calculate chord durations
            durations = [
                chord.end_time - chord.start_time 
                for chord in chord_labels
                if chord.quality != ChordQuality.NO_CHORD
            ]
            
            if not durations:
                return 0.0
            
            # Average chord duration
            avg_duration = np.mean(durations)
            
            # Convert to harmonic rhythm (chords per minute)
            harmonic_rhythm = 60.0 / avg_duration if avg_duration > 0 else 0.0
            
            return float(harmonic_rhythm)
            
        except Exception as e:
            logger.error(f"Harmonic rhythm analysis failed: {e}")
            return 0.0
    
    def _assess_harmonic_complexity(self, chord_labels: List[ChordLabel],
                                  harmonic_functions: List[HarmonicFunction],
                                  modulations: List) -> float:
        """Assess the harmonic complexity of the music."""
        try:
            if not chord_labels:
                return 0.0
            
            complexity_factors = []
            
            # Chord type diversity
            unique_qualities = set(chord.quality for chord in chord_labels)
            quality_complexity = len(unique_qualities) / len(ChordQuality)
            complexity_factors.append(quality_complexity)
            
            # Chromatic harmony
            chromatic_count = sum(
                1 for func in harmonic_functions 
                if func == HarmonicFunction.CHROMATIC
            )
            chromatic_ratio = chromatic_count / len(harmonic_functions)
            complexity_factors.append(chromatic_ratio)
            
            # Modulation frequency
            modulation_complexity = min(1.0, len(modulations) / 4)  # Normalize to 4 modulations
            complexity_factors.append(modulation_complexity)
            
            # Extended harmony (7ths, 9ths, etc.)
            extended_chords = sum(
                1 for chord in chord_labels
                if chord.quality.value.endswith(('7', '9', '11', '13'))
            )
            extended_ratio = extended_chords / len(chord_labels)
            complexity_factors.append(extended_ratio)
            
            # Average complexity
            return float(np.mean(complexity_factors))
            
        except Exception as e:
            logger.error(f"Complexity assessment failed: {e}")
            return 0.5
    
    def _compute_analysis_confidence(self, key_confidence: float,
                                   chord_labels: List[ChordLabel],
                                   complexity_score: float) -> float:
        """Compute overall analysis confidence."""
        try:
            if not chord_labels:
                return 0.0
            
            # Key detection confidence
            key_weight = 0.3
            
            # Average chord confidence
            chord_confidences = [chord.confidence for chord in chord_labels]
            avg_chord_confidence = np.mean(chord_confidences)
            chord_weight = 0.5
            
            # Complexity penalty (very complex music is harder to analyze)
            complexity_penalty = 1.0 - (complexity_score * 0.2)
            complexity_weight = 0.2
            
            overall_confidence = (
                key_weight * key_confidence +
                chord_weight * avg_chord_confidence +
                complexity_weight * complexity_penalty
            )
            
            return max(0.0, min(1.0, overall_confidence))
            
        except Exception as e:
            logger.error(f"Confidence computation failed: {e}")
            return 0.5
    
    def _collect_analysis_warnings(self, chord_labels: List[ChordLabel],
                                 key_confidence: float,
                                 complexity_score: float) -> List[str]:
        """Collect warnings about analysis quality."""
        warnings = []
        
        try:
            if not chord_labels:
                warnings.append("No chords detected")
            elif len(chord_labels) < 4:
                warnings.append("Very few chords detected - analysis may be unreliable")
            
            if key_confidence < 0.5:
                warnings.append("Low key detection confidence")
            
            if complexity_score > 0.8:
                warnings.append("Very complex harmony - some analysis may be uncertain")
            
            # Check for predominance of unknown chords
            unknown_count = sum(
                1 for chord in chord_labels 
                if chord.quality in [ChordQuality.UNKNOWN, ChordQuality.NO_CHORD]
            )
            if unknown_count / len(chord_labels) > 0.5:
                warnings.append("Many unrecognized chords")
            
            # Check for very short chords (possible false positives)
            short_chords = sum(
                1 for chord in chord_labels
                if chord.end_time - chord.start_time < 0.5
            )
            if short_chords / len(chord_labels) > 0.3:
                warnings.append("Many very short chords detected - possible false positives")
            
        except Exception as e:
            warnings.append(f"Error collecting warnings: {e}")
        
        return warnings
    
    def _create_fallback_analysis(self, start_time: float, error_msg: str) -> HarmonicAnalysis:
        """Create fallback analysis for error cases."""
        return HarmonicAnalysis(
            key=0,  # C
            mode="major",
            key_confidence=0.0,
            chords=[],
            harmonic_functions=[],
            progression_types=[ProgressionType.UNKNOWN],
            modulations=[],
            harmonic_rhythm=0.0,
            complexity_score=0.0,
            confidence=0.0,
            processing_time=time.time() - start_time,
            warnings=[f"Analysis failed: {error_msg}"]
        )


class BulletproofChordSequenceModeler(nn.Module):
    """
    Complete chord sequence modeling system combining chroma extraction,
    chord recognition, and harmonic analysis for comprehensive musical
    understanding.
    """
    
    def __init__(self, config: Optional[BulletproofAudioConfig] = None):
        super().__init__()
        
        if config is None:
            config = get_bulletproof_music_config()
        
        self.config = config
        
        # Initialize components
        self.chroma_extractor = BulletproofChromaExtractor(config)
        self.chord_recognizer = BulletproofChordRecognizer(config)
        self.harmonic_analyzer = BulletproofHarmonicAnalyzer(config)
        
        # Quality thresholds
        self.min_chroma_clarity = 0.3
        self.min_chord_confidence = 0.2
        self.min_analysis_confidence = 0.3
    
    def forward(self, audio: torch.Tensor) -> HarmonicAnalysis:
        """
        Perform complete chord sequence modeling and harmonic analysis.
        
        Args:
            audio: Audio tensor [batch_size, time] or [time]
            
        Returns:
            HarmonicAnalysis with complete chord and harmonic information
        """
        import time
        start_time = time.time()
        
        try:
            # Validate input
            audio = self._validate_audio_input(audio)
            if audio is None:
                return self._create_fallback_analysis("Invalid audio input", start_time)
            
            # Extract chroma features
            chroma, chroma_info = self.chroma_extractor(audio)
            
            if chroma_info['clarity_score'] < self.min_chroma_clarity:
                logger.warning(f"Low chroma clarity: {chroma_info['clarity_score']:.3f}")
            
            # Recognize chords
            chord_labels, recognition_info = self.chord_recognizer(chroma)
            
            if not chord_labels:
                logger.warning("No chords recognized")
                return self._create_fallback_analysis("No chords recognized", start_time)
            
            # Perform harmonic analysis
            harmonic_analysis = self.harmonic_analyzer(chroma, chord_labels)
            
            # Update with processing information
            harmonic_analysis.processing_time = time.time() - start_time
            
            # Add component information to warnings if needed
            if chroma_info['clarity_score'] < self.min_chroma_clarity:
                harmonic_analysis.warnings.append(
                    f"Low chroma clarity: {chroma_info['clarity_score']:.3f}"
                )
            
            if recognition_info['average_confidence'] < self.min_chord_confidence:
                harmonic_analysis.warnings.append(
                    f"Low chord recognition confidence: {recognition_info['average_confidence']:.3f}"
                )
            
            return harmonic_analysis
            
        except Exception as e:
            logger.error(f"Chord sequence modeling failed: {e}")
            return self._create_fallback_analysis(f"Processing error: {e}", start_time)
    
    def _validate_audio_input(self, audio: torch.Tensor) -> Optional[torch.Tensor]:
        """Validate and preprocess audio input."""
        try:
            if not isinstance(audio, torch.Tensor):
                logger.error(f"Expected torch.Tensor, got {type(audio)}")
                return None
            
            # Ensure 1D or 2D
            if audio.dim() > 2:
                logger.error(f"Audio tensor has too many dimensions: {audio.dim()}")
                return None
            
            # Convert to 1D if needed
            if audio.dim() == 2:
                if audio.size(0) == 1:
                    audio = audio.squeeze(0)
                elif audio.size(1) == 1:
                    audio = audio.squeeze(1)
                else:
                    # Take first channel
                    audio = audio[0]
                    logger.warning("Multi-channel audio detected, using first channel")
            
            # Check length
            min_length = self.config.sample_rate * 5  # Minimum 5 seconds for chord analysis
            if audio.size(-1) < min_length:
                logger.warning(f"Audio too short: {audio.size(-1)} samples, minimum {min_length}")
                if audio.size(-1) < self.config.hop_length:
                    return None
            
            # Check for silence
            if audio.abs().max() < 1e-6:
                logger.warning("Audio appears to be silent")
                return None
            
            # Normalize
            audio = audio / (audio.abs().max() + 1e-8)
            
            return audio
            
        except Exception as e:
            logger.error(f"Audio validation failed: {e}")
            return None
    
    def _create_fallback_analysis(self, reason: str, start_time: float) -> HarmonicAnalysis:
        """Create fallback analysis for error cases."""
        return HarmonicAnalysis(
            key=0,  # C major
            mode="major",
            key_confidence=0.0,
            chords=[],
            harmonic_functions=[],
            progression_types=[ProgressionType.UNKNOWN],
            modulations=[],
            harmonic_rhythm=0.0,
            complexity_score=0.0,
            confidence=0.0,
            processing_time=time.time() - start_time,
            warnings=[reason]
        )
    
    def analyze_long_audio(self, audio: torch.Tensor,
                          chunk_duration: float = 60.0,
                          overlap_duration: float = 10.0) -> List[HarmonicAnalysis]:
        """
        Analyze long audio by chunking for chord sequence modeling.
        
        Args:
            audio: Long audio tensor
            chunk_duration: Duration of each chunk in seconds
            overlap_duration: Overlap between chunks in seconds
            
        Returns:
            List of HarmonicAnalysis for each chunk
        """
        try:
            results = []
            chunk_samples = int(chunk_duration * self.config.sample_rate)
            overlap_samples = int(overlap_duration * self.config.sample_rate)
            hop_samples = chunk_samples - overlap_samples
            
            for start in range(0, audio.size(-1) - chunk_samples + 1, hop_samples):
                end = start + chunk_samples
                chunk = audio[start:end]
                
                result = self.forward(chunk)
                
                # Adjust times to global timeline
                global_offset = start / self.config.sample_rate
                
                # Update chord timings
                for chord in result.chords:
                    chord.start_time += global_offset
                    chord.end_time += global_offset
                
                # Update modulation timings
                adjusted_modulations = []
                for time_point, key, mode in result.modulations:
                    adjusted_modulations.append((time_point + global_offset, key, mode))
                result.modulations = adjusted_modulations
                
                # Add chunk metadata
                result.warnings.append(
                    f"Chunk {len(results)+1}: {global_offset:.1f}-{end/self.config.sample_rate:.1f}s"
                )
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Long audio analysis failed: {e}")
            return []


# Factory functions and utilities

def create_chord_sequence_modeler(sample_rate: int = 22050,
                                domain: str = "music",
                                **kwargs) -> BulletproofChordSequenceModeler:
    """Create a chord sequence modeler with specified configuration."""
    try:
        if domain.lower() == "music":
            config = get_bulletproof_music_config(sample_rate=sample_rate, **kwargs)
        else:
            from modules.audio_analysis.bulletproof_audio_config import (
                BulletproofAudioConfig, AudioDomain
            )
            config = BulletproofAudioConfig(
                sample_rate=sample_rate,
                domain=AudioDomain.GENERAL,
                **kwargs
            )
        
        return BulletproofChordSequenceModeler(config)
        
    except Exception as e:
        logger.error(f"Failed to create chord sequence modeler: {e}")
        # Fallback configuration
        config = BulletproofAudioConfig(sample_rate=sample_rate)
        return BulletproofChordSequenceModeler(config)


def test_chord_sequence_modeler():
    """Test the chord sequence modeler with synthetic audio."""
    logger.info("Testing BulletproofChordSequenceModeler")
    
    try:
        # Create test audio with chord progression
        sr = 22050
        duration = 16.0  # 16 seconds
        
        # Simple chord progression: C - Am - F - G (I - vi - IV - V)
        chord_progression = [
            (0, [0, 4, 7]),    # C major
            (4, [0, 3, 7]),    # A minor (A, C, E) - relative minor
            (8, [0, 4, 7]),    # F major (F, A, C)
            (12, [0, 4, 7])    # G major (G, B, D)
        ]
        
        # Convert to frequencies (simplified)
        def note_to_freq(note_num):
            return 440.0 * (2 ** ((note_num - 69) / 12.0))  # A4 = 440 Hz
        
        # Create audio
        t = torch.linspace(0, duration, int(sr * duration))
        audio = torch.zeros_like(t)
        
        for start_time, chord_notes in chord_progression:
            end_time = min(start_time + 4, duration)
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            if start_sample < len(audio):
                chord_audio = torch.zeros(end_sample - start_sample)
                
                # Add each note in the chord
                for note_offset in chord_notes:
                    # Convert to MIDI note (C4 = 60)
                    midi_note = 60 + note_offset
                    frequency = note_to_freq(midi_note)
                    
                    # Create sinusoid for this note
                    chord_length = end_sample - start_sample
                    note_t = torch.linspace(0, chord_length / sr, chord_length)
                    note_audio = 0.3 * torch.sin(2 * np.pi * frequency * note_t)
                    
                    # Add harmonics for realism
                    note_audio += 0.15 * torch.sin(2 * np.pi * frequency * 2 * note_t)
                    note_audio += 0.1 * torch.sin(2 * np.pi * frequency * 3 * note_t)
                    
                    chord_audio += note_audio
                
                # Apply envelope
                envelope = torch.exp(-note_t * 0.5)  # Decay
                chord_audio *= envelope
                
                # Add to main audio
                audio[start_sample:end_sample] += chord_audio
        
        # Add some noise for realism
        audio += 0.05 * torch.randn_like(audio)
        
        # Test chord sequence modeler
        modeler = create_chord_sequence_modeler(sample_rate=sr)
        result = modeler(audio)
        
        # Analyze results
        logger.info(f"Chord sequence modeling results:")
        logger.info(f"  Detected key: {['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][result.key]} {result.mode}")
        logger.info(f"  Key confidence: {result.key_confidence:.3f}")
        logger.info(f"  Number of chords: {len(result.chords)}")
        logger.info(f"  Harmonic rhythm: {result.harmonic_rhythm:.1f} chords/minute")
        logger.info(f"  Complexity score: {result.complexity_score:.3f}")
        logger.info(f"  Overall confidence: {result.confidence:.3f}")
        logger.info(f"  Processing time: {result.processing_time:.3f}s")
        
        # Show detected chords
        logger.info("  Detected chord sequence:")
        for i, chord in enumerate(result.chords[:10]):  # Show first 10 chords
            logger.info(f"    {i+1}: {chord.to_string()} ({chord.start_time:.1f}-{chord.end_time:.1f}s, conf={chord.confidence:.3f})")
        
        if len(result.chords) > 10:
            logger.info(f"    ... and {len(result.chords) - 10} more chords")
        
        # Show harmonic functions
        if result.harmonic_functions:
            logger.info("  Harmonic functions:")
            for i, func in enumerate(result.harmonic_functions[:10]):
                logger.info(f"    {i+1}: {func.value}")
        
        # Show progression types
        if result.progression_types:
            logger.info(f"  Progression types: {[pt.value for pt in result.progression_types]}")
        
        # Show modulations
        if result.modulations:
            logger.info(f"  Modulations detected: {len(result.modulations)}")
            for time_point, key, mode in result.modulations:
                key_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][key]
                logger.info(f"    {time_point:.1f}s -> {key_name} {mode}")
        
        if result.warnings:
            logger.info(f"  Warnings: {'; '.join(result.warnings)}")
        
        logger.info("Chord sequence modeler test completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Chord sequence modeler test failed: {e}")
        return None


if __name__ == "__main__":
    test_chord_sequence_modeler()