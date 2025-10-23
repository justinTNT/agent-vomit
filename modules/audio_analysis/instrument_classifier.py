import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import Dict, List, Optional, Tuple
import numpy as np


class InstrumentClassifier(nn.Module):
    """
    Multi-scale instrument classification using spectral and temporal features.
    
    Designed for instrument recognition with hierarchical classification:
    - Family level (strings, winds, percussion, etc.)
    - Instrument level (guitar, piano, violin, etc.)
    - Playing technique level (pizzicato, arco, muted, etc.)
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        num_families: int = 8,
        num_instruments: int = 50,
        num_techniques: int = 20,
        hidden_dim: int = 512,
        num_conv_layers: int = 4,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
        **kwargs
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.num_families = num_families
        self.num_instruments = num_instruments
        self.num_techniques = num_techniques
        
        # Audio preprocessing
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=0,
            f_max=sample_rate // 2
        )
        
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=13,
            melkwargs={
                'n_fft': n_fft,
                'hop_length': hop_length,
                'n_mels': n_mels
            }
        )
        
        # Spectral feature extraction
        self.spectral_conv = self._build_conv_layers(
            in_channels=1,
            hidden_dim=hidden_dim,
            num_layers=num_conv_layers,
            dropout=dropout
        )
        
        # Temporal modeling
        self.temporal_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Multi-scale feature fusion
        self.feature_fusion = nn.ModuleDict({
            'mel_proj': nn.Linear(n_mels, hidden_dim),
            'mfcc_proj': nn.Linear(13, hidden_dim),
            'spectral_proj': nn.Linear(hidden_dim, hidden_dim),
            'temporal_proj': nn.Linear(hidden_dim * 2, hidden_dim)  # *2 for bidirectional
        })
        
        # Attention mechanism for temporal pooling
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Hierarchical classification heads
        self.family_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_families)
        )
        
        self.instrument_head = nn.Sequential(
            nn.LayerNorm(hidden_dim + num_families),  # Concat family features
            nn.Linear(hidden_dim + num_families, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_instruments)
        )
        
        self.technique_head = nn.Sequential(
            nn.LayerNorm(hidden_dim + num_families + num_instruments),
            nn.Linear(hidden_dim + num_families + num_instruments, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_techniques)
        )
        
        # Feature statistics for analysis
        self.register_buffer('feature_stats', torch.zeros(4, 2))  # [mean, std] for 4 feature types
        
    def _build_conv_layers(self, in_channels: int, hidden_dim: int, num_layers: int, dropout: float):
        """Build convolutional feature extraction layers."""
        layers = []
        current_channels = in_channels
        
        for i in range(num_layers):
            out_channels = hidden_dim // (2 ** (num_layers - i - 1))
            
            layers.extend([
                nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout)
            ])
            current_channels = out_channels
            
        # Global pooling
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        
        return nn.Sequential(*layers)
        
    def extract_features(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract multi-scale audio features."""
        batch_size = waveform.shape[0]
        
        # Ensure mono audio
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
            
        # Extract mel-spectrogram
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = torchaudio.functional.amplitude_to_DB(mel_spec, multiplier=10.0, amin=1e-8)
        
        # Extract MFCCs
        mfcc = self.mfcc_transform(waveform)
        
        # Spectral features via CNN
        mel_input = mel_spec_db.unsqueeze(1)  # Add channel dim
        spectral_features = self.spectral_conv(mel_input)
        spectral_features = spectral_features.view(batch_size, -1)
        
        # Temporal features via LSTM
        # Use mel-spectrogram as temporal sequence
        mel_sequence = mel_spec_db.transpose(1, 2)  # [batch, time, freq]
        lstm_out, _ = self.temporal_lstm(mel_sequence)
        
        # Attention pooling over time
        temporal_features, _ = self.attention(lstm_out, lstm_out, lstm_out)
        temporal_features = temporal_features.mean(dim=1)  # Pool over time
        
        # Statistical features from raw spectrograms
        mel_stats = torch.cat([
            mel_spec_db.mean(dim=2),  # Spectral centroid-like
            mel_spec_db.std(dim=2)    # Spectral spread-like
        ], dim=1).mean(dim=1)  # Pool over time
        
        mfcc_stats = torch.cat([
            mfcc.mean(dim=2),
            mfcc.std(dim=2)
        ], dim=1).mean(dim=1)
        
        return {
            'mel_features': mel_stats,
            'mfcc_features': mfcc_stats,
            'spectral_features': spectral_features,
            'temporal_features': temporal_features,
            'raw_mel': mel_spec_db,
            'raw_mfcc': mfcc
        }
        
    def forward(
        self, 
        waveform: torch.Tensor,
        return_features: bool = False,
        return_hierarchical: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for instrument classification.
        
        Args:
            waveform: Input audio [batch, samples]
            return_features: Whether to return extracted features
            return_hierarchical: Whether to return all hierarchy levels
            
        Returns:
            Dictionary with classification results and optionally features
        """
        # Extract features
        features = self.extract_features(waveform)
        
        # Project features to common dimension
        mel_proj = self.feature_fusion['mel_proj'](features['mel_features'])
        mfcc_proj = self.feature_fusion['mfcc_proj'](features['mfcc_features'])
        spectral_proj = self.feature_fusion['spectral_proj'](features['spectral_features'])
        temporal_proj = self.feature_fusion['temporal_proj'](features['temporal_features'])
        
        # Combine features
        combined_features = mel_proj + mfcc_proj + spectral_proj + temporal_proj
        combined_features = F.normalize(combined_features, p=2, dim=1)
        
        # Hierarchical classification
        # Level 1: Family classification
        family_logits = self.family_head(combined_features)
        family_probs = F.softmax(family_logits, dim=1)
        
        # Level 2: Instrument classification (conditioned on family)
        instrument_input = torch.cat([combined_features, family_probs], dim=1)
        instrument_logits = self.instrument_head(instrument_input)
        instrument_probs = F.softmax(instrument_logits, dim=1)
        
        # Level 3: Playing technique classification
        technique_input = torch.cat([combined_features, family_probs, instrument_probs], dim=1)
        technique_logits = self.technique_head(technique_input)
        
        result = {
            'family_logits': family_logits,
            'instrument_logits': instrument_logits,
            'technique_logits': technique_logits,
            'combined_features': combined_features
        }
        
        if return_hierarchical:
            result.update({
                'family_probs': family_probs,
                'instrument_probs': instrument_probs,
                'technique_probs': F.softmax(technique_logits, dim=1)
            })
            
        if return_features:
            result['features'] = features
            
        return result
        
    def predict_instrument(
        self, 
        waveform: torch.Tensor,
        family_names: Optional[List[str]] = None,
        instrument_names: Optional[List[str]] = None,
        technique_names: Optional[List[str]] = None,
        confidence_threshold: float = 0.5
    ) -> Dict[str, any]:
        """
        Make human-readable predictions with confidence scores.
        
        Args:
            waveform: Input audio
            family_names: List of family names for interpretation
            instrument_names: List of instrument names
            technique_names: List of technique names
            confidence_threshold: Minimum confidence for prediction
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        with torch.no_grad():
            output = self.forward(waveform, return_hierarchical=True)
            
        # Get top predictions
        family_idx = output['family_probs'].argmax(dim=1)
        instrument_idx = output['instrument_probs'].argmax(dim=1)
        technique_idx = output['technique_probs'].argmax(dim=1)
        
        # Get confidence scores
        family_conf = output['family_probs'].max(dim=1)[0]
        instrument_conf = output['instrument_probs'].max(dim=1)[0]
        technique_conf = output['technique_probs'].max(dim=1)[0]
        
        predictions = []
        for i in range(len(family_idx)):
            pred = {
                'family_idx': family_idx[i].item(),
                'instrument_idx': instrument_idx[i].item(),
                'technique_idx': technique_idx[i].item(),
                'family_confidence': family_conf[i].item(),
                'instrument_confidence': instrument_conf[i].item(),
                'technique_confidence': technique_conf[i].item()
            }
            
            # Add names if provided
            if family_names:
                pred['family'] = family_names[pred['family_idx']]
            if instrument_names:
                pred['instrument'] = instrument_names[pred['instrument_idx']]
            if technique_names:
                pred['technique'] = technique_names[pred['technique_idx']]
                
            # Overall confidence (geometric mean)
            pred['overall_confidence'] = (
                pred['family_confidence'] * 
                pred['instrument_confidence'] * 
                pred['technique_confidence']
            ) ** (1/3)
            
            predictions.append(pred)
            
        return {
            'predictions': predictions,
            'confident': [p['overall_confidence'] > confidence_threshold for p in predictions]
        }
        
    def get_feature_importance(self, waveform: torch.Tensor) -> Dict[str, float]:
        """Analyze feature importance for a given input."""
        with torch.no_grad():
            features = self.extract_features(waveform)
            
            # Compute feature magnitudes
            mel_importance = features['mel_features'].abs().mean().item()
            mfcc_importance = features['mfcc_features'].abs().mean().item()
            spectral_importance = features['spectral_features'].abs().mean().item()
            temporal_importance = features['temporal_features'].abs().mean().item()
            
            total = mel_importance + mfcc_importance + spectral_importance + temporal_importance
            
            return {
                'mel_spectral': mel_importance / total,
                'mfcc': mfcc_importance / total,
                'cnn_spectral': spectral_importance / total,
                'lstm_temporal': temporal_importance / total
            }


# Pre-defined instrument taxonomies
INSTRUMENT_FAMILIES = [
    'strings', 'woodwinds', 'brass', 'percussion', 
    'keyboard', 'electronic', 'vocal', 'other'
]

COMMON_INSTRUMENTS = [
    'guitar', 'piano', 'violin', 'flute', 'trumpet', 'drums',
    'clarinet', 'saxophone', 'cello', 'bass', 'voice', 'organ',
    'harp', 'accordion', 'harmonica', 'banjo', 'mandolin', 'ukulele',
    'oboe', 'bassoon', 'trombone', 'french_horn', 'tuba', 'xylophone',
    'marimba', 'vibraphone', 'timpani', 'cymbals', 'triangle', 'tambourine',
    'synthesizer', 'electric_guitar', 'electric_bass', 'electric_piano',
    'theremin', 'sample_pad', 'drum_machine', 'string_ensemble',
    'choir', 'soprano', 'alto', 'tenor', 'bass_voice', 'whistle',
    'clapping', 'footsteps', 'door_slam', 'glass_break', 'water_drop', 'other'
]

PLAYING_TECHNIQUES = [
    'normal', 'staccato', 'legato', 'pizzicato', 'arco', 'tremolo',
    'vibrato', 'muted', 'palm_muted', 'harmonics', 'glissando', 'trill',
    'flutter_tongue', 'double_stop', 'chord', 'arpeggio', 'sustained',
    'percussive', 'bowed', 'other'
]


# Factory functions
def create_basic_instrument_classifier(**kwargs) -> InstrumentClassifier:
    """Create basic instrument classifier with reasonable defaults."""
    return InstrumentClassifier(
        num_families=len(INSTRUMENT_FAMILIES),
        num_instruments=len(COMMON_INSTRUMENTS),
        num_techniques=len(PLAYING_TECHNIQUES),
        **kwargs
    )

def create_guitar_classifier(**kwargs) -> InstrumentClassifier:
    """Create specialized guitar classifier."""
    guitar_techniques = [
        'clean', 'distorted', 'palm_muted', 'harmonics', 'bend', 
        'slide', 'fingerpicking', 'strumming', 'tapping', 'other'
    ]
    
    return InstrumentClassifier(
        num_families=3,  # acoustic, electric, bass
        num_instruments=10,  # specific guitar types
        num_techniques=len(guitar_techniques),
        **kwargs
    )


# Example usage
if __name__ == "__main__":
    # Create classifier
    classifier = create_basic_instrument_classifier()
    
    # Test with dummy audio
    waveform = torch.randn(2, 22050 * 3)  # 3 seconds
    
    # Forward pass
    output = classifier(waveform, return_features=True)
    
    print(f"Family logits shape: {output['family_logits'].shape}")
    print(f"Instrument logits shape: {output['instrument_logits'].shape}")
    print(f"Technique logits shape: {output['technique_logits'].shape}")
    
    # Make predictions
    predictions = classifier.predict_instrument(
        waveform,
        family_names=INSTRUMENT_FAMILIES,
        instrument_names=COMMON_INSTRUMENTS,
        technique_names=PLAYING_TECHNIQUES
    )
    
    print(f"Predictions: {predictions['predictions'][0]}")
    
    # Feature importance
    importance = classifier.get_feature_importance(waveform[:1])
    print(f"Feature importance: {importance}")