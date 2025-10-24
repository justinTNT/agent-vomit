"""
Multimodal Audio Models

Implements state-of-the-art multimodal models for audio-text and audio-visual learning:
- CLAP: Contrastive Language-Audio Pre-training
- ImageBind Audio: Audio component of ImageBind multimodal model
- AudioCLIP: Audio-image-text contrastive learning
- WavCaps: Audio captioning and retrieval
- MusicCaps: Music description and generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import math
from dataclasses import dataclass
from enum import Enum

from ..audio_analysis.audio_config import AudioModuleConfig, AudioModuleBase


class MultimodalModelType(Enum):
    """Types of multimodal models."""
    CLAP = "clap"
    IMAGEBIND_AUDIO = "imagebind_audio"
    AUDIOCLIP = "audioclip"
    WAVCAPS = "wavcaps"
    MUSICCAPS = "musiccaps"


@dataclass
class MultimodalConfig(AudioModuleConfig):
    """Configuration for multimodal models."""
    model_type: MultimodalModelType = MultimodalModelType.CLAP
    audio_embed_dim: int = 512
    text_embed_dim: int = 512
    shared_embed_dim: int = 512
    audio_encoder_layers: int = 12
    text_encoder_layers: int = 12
    num_heads: int = 8
    temperature: float = 0.07
    audio_length: int = 10  # seconds
    normalize_embeddings: bool = True


class AudioEncoder(nn.Module):
    """
    Audio encoder for multimodal learning.
    
    Converts audio spectrograms to fixed-size embeddings.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        embed_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_cnn_frontend: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.use_cnn_frontend = use_cnn_frontend
        
        if use_cnn_frontend:
            # CNN frontend for local feature extraction
            self.cnn_frontend = nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((8, 8))
            )
            
            # Project CNN features to embedding dimension
            self.feature_projection = nn.Linear(512 * 8 * 8, embed_dim)
        else:
            # Direct linear projection for pre-extracted features
            self.feature_projection = nn.Linear(input_channels, embed_dim)
            
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(1000, embed_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Global pooling and output projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to embedding.
        
        Args:
            x: Audio input - spectrogram [batch, channels, freq, time] or features [batch, seq, dim]
            
        Returns:
            Audio embeddings [batch, embed_dim]
        """
        if self.use_cnn_frontend and len(x.shape) == 4:
            # CNN frontend for spectrogram input
            batch_size = x.shape[0]
            x = self.cnn_frontend(x)  # [batch, 512, 8, 8]
            x = x.view(batch_size, -1)  # [batch, 512*8*8]
            x = self.feature_projection(x)  # [batch, embed_dim]
            x = x.unsqueeze(1)  # [batch, 1, embed_dim] for transformer
        else:
            # Linear projection for feature input
            x = self.feature_projection(x)  # [batch, seq, embed_dim]
            
        # Add positional encoding
        seq_len = x.shape[1]
        pos_enc = self.pos_encoding[:, :seq_len, :].to(x.device)
        x = x + pos_enc
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Global pooling
        if x.shape[1] > 1:
            x = x.transpose(1, 2)  # [batch, embed_dim, seq]
            x = self.global_pool(x).squeeze(-1)  # [batch, embed_dim]
        else:
            x = x.squeeze(1)  # [batch, embed_dim]
            
        # Output projection
        x = self.output_projection(x)
        
        return x


class TextEncoder(nn.Module):
    """
    Text encoder for multimodal learning.
    
    Simple transformer-based encoder for text embeddings.
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        embed_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        max_length: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_length = max_length
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional embedding
        self.pos_embedding = nn.Embedding(max_length, embed_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode text to embedding.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            
        Returns:
            Text embeddings [batch, embed_dim]
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_emb = self.token_embedding(input_ids)
        
        # Positional embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        
        # Combine embeddings
        x = token_emb + pos_emb
        
        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert to transformer format (True = masked)
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None
            
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        
        # Pool to single embedding (mean pooling over non-masked tokens)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(x)
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            x = x.mean(dim=1)
            
        # Output projection
        x = self.output_projection(x)
        
        return x


class CLAP(AudioModuleBase):
    """
    Contrastive Language-Audio Pre-training (CLAP).
    
    Learns joint embeddings of audio and text through contrastive learning.
    """
    
    def __init__(self, config: MultimodalConfig):
        super().__init__(config)
        
        self.config = config
        
        # Audio encoder
        self.audio_encoder = AudioEncoder(
            embed_dim=config.audio_embed_dim,
            num_layers=config.audio_encoder_layers,
            dropout=config.dropout
        )
        
        # Text encoder
        self.text_encoder = TextEncoder(
            embed_dim=config.text_embed_dim,
            num_layers=config.text_encoder_layers,
            dropout=config.dropout
        )
        
        # Projection heads to shared embedding space
        self.audio_projection = nn.Linear(config.audio_embed_dim, config.shared_embed_dim)
        self.text_projection = nn.Linear(config.text_embed_dim, config.shared_embed_dim)
        
        # Temperature parameter for contrastive loss
        self.temperature = nn.Parameter(torch.tensor(config.temperature))
        
        # Use standardized preprocessing
        from ..audio_analysis.audio_preprocessing import StandardAudioPreprocessor
        self.preprocessor = StandardAudioPreprocessor(config)
        
    def encode_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode audio to shared embedding space."""
        # Use standardized preprocessing
        mel_spec = self.preprocessor.extract_mel_spectrogram(waveform)
        
        # Add channel dimension if needed
        if len(mel_spec.shape) == 3:
            mel_spec = mel_spec.unsqueeze(1)
            
        # Encode audio
        audio_features = self.audio_encoder(mel_spec)
        
        # Project to shared space
        audio_embedding = self.audio_projection(audio_features)
        
        # Normalize if specified
        if self.config.normalize_embeddings:
            audio_embedding = F.normalize(audio_embedding, dim=-1)
            
        return audio_embedding
        
    def encode_text(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode text to shared embedding space."""
        # Encode text
        text_features = self.text_encoder(input_ids, attention_mask)
        
        # Project to shared space
        text_embedding = self.text_projection(text_features)
        
        # Normalize if specified
        if self.config.normalize_embeddings:
            text_embedding = F.normalize(text_embedding, dim=-1)
            
        return text_embedding
        
    def compute_contrastive_loss(
        self,
        audio_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute bidirectional contrastive loss."""
        batch_size = audio_embeddings.shape[0]
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(audio_embeddings, text_embeddings.t()) / self.temperature
        
        # Labels (positive pairs are on the diagonal)
        labels = torch.arange(batch_size, device=audio_embeddings.device)
        
        # Audio-to-text loss
        audio_to_text_loss = F.cross_entropy(similarity_matrix, labels)
        
        # Text-to-audio loss
        text_to_audio_loss = F.cross_entropy(similarity_matrix.t(), labels)
        
        # Total loss
        total_loss = (audio_to_text_loss + text_to_audio_loss) / 2
        
        # Accuracy metrics
        with torch.no_grad():
            audio_to_text_acc = (similarity_matrix.argmax(dim=1) == labels).float().mean()
            text_to_audio_acc = (similarity_matrix.t().argmax(dim=1) == labels).float().mean()
            
        return {
            'total_loss': total_loss,
            'audio_to_text_loss': audio_to_text_loss,
            'text_to_audio_loss': text_to_audio_loss,
            'audio_to_text_accuracy': audio_to_text_acc,
            'text_to_audio_accuracy': text_to_audio_acc,
            'similarity_matrix': similarity_matrix
        }
        
    def forward(
        self,
        waveform: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for CLAP training.
        
        Args:
            waveform: Audio waveform [batch, samples]
            input_ids: Text token IDs [batch, seq_len]
            attention_mask: Text attention mask [batch, seq_len]
            
        Returns:
            Dictionary with embeddings and loss
        """
        # Encode modalities
        audio_embeddings = self.encode_audio(waveform)
        text_embeddings = self.encode_text(input_ids, attention_mask)
        
        # Compute contrastive loss
        loss_dict = self.compute_contrastive_loss(audio_embeddings, text_embeddings)
        
        return {
            'audio_embeddings': audio_embeddings,
            'text_embeddings': text_embeddings,
            **loss_dict
        }
        
    def get_text_features(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get text features for retrieval."""
        with torch.no_grad():
            return self.encode_text(input_ids, attention_mask)
            
    def get_audio_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Get audio features for retrieval."""
        with torch.no_grad():
            return self.encode_audio(waveform)


class ImageBindAudio(AudioModuleBase):
    """
    Audio component of ImageBind multimodal model.
    
    Learns embeddings that can be aligned with vision, text, and other modalities.
    """
    
    def __init__(self, config: MultimodalConfig):
        super().__init__(config)
        
        self.config = config
        
        # Multi-scale audio processing
        self.multi_scale_encoder = self._build_multi_scale_encoder()
        
        # Cross-modal attention mechanism
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=config.shared_embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Modality-specific normalization
        self.audio_norm = nn.LayerNorm(config.shared_embed_dim)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(config.shared_embed_dim, config.shared_embed_dim),
            nn.GELU(),
            nn.Linear(config.shared_embed_dim, config.shared_embed_dim)
        )
        
    def _build_multi_scale_encoder(self):
        """Build multi-scale audio encoder."""
        scales = [
            # (n_fft, hop_length, n_mels)
            (1024, 256, 64),    # Fine temporal resolution
            (2048, 512, 128),   # Medium resolution
            (4096, 1024, 256)   # Coarse resolution
        ]
        
        encoders = nn.ModuleDict()
        
        for i, (n_fft, hop_length, n_mels) in enumerate(scales):
            # Mel transform
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.config.sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels
            )
            
            # Audio encoder
            encoder = AudioEncoder(
                input_channels=1,
                embed_dim=self.config.shared_embed_dim,
                num_layers=6,
                num_heads=self.config.num_heads,
                dropout=self.config.dropout
            )
            
            encoders[f'scale_{i}'] = nn.ModuleDict({
                'mel_transform': mel_transform,
                'encoder': encoder
            })
            
        return encoders
        
    def encode_multi_scale(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        """Encode audio at multiple scales."""
        embeddings = []
        
        for scale_name, scale_modules in self.multi_scale_encoder.items():
            mel_transform = scale_modules['mel_transform']
            encoder = scale_modules['encoder']
            
            # Convert to mel-spectrogram
            mel_spec = mel_transform(waveform)
            mel_spec = torch.log(mel_spec + 1e-8)
            
            # Add channel dimension
            if len(mel_spec.shape) == 3:
                mel_spec = mel_spec.unsqueeze(1)
                
            # Encode
            embedding = encoder(mel_spec)
            embeddings.append(embedding)
            
        return embeddings
        
    def fuse_multi_scale_features(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """Fuse multi-scale features using attention."""
        # Stack embeddings
        stacked = torch.stack(embeddings, dim=1)  # [batch, num_scales, embed_dim]
        
        # Self-attention across scales
        fused, _ = self.cross_modal_attention(stacked, stacked, stacked)
        
        # Global pooling
        fused = fused.mean(dim=1)  # [batch, embed_dim]
        
        return fused
        
    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for ImageBind audio.
        
        Args:
            waveform: Audio waveform [batch, samples]
            
        Returns:
            Dictionary with multi-scale embeddings
        """
        # Multi-scale encoding
        scale_embeddings = self.encode_multi_scale(waveform)
        
        # Fuse scales
        fused_embedding = self.fuse_multi_scale_features(scale_embeddings)
        
        # Normalize
        fused_embedding = self.audio_norm(fused_embedding)
        
        # Output projection
        final_embedding = self.output_projection(fused_embedding)
        
        # Normalize for multimodal alignment
        if self.config.normalize_embeddings:
            final_embedding = F.normalize(final_embedding, dim=-1)
            
        return {
            'scale_embeddings': scale_embeddings,
            'fused_embedding': fused_embedding,
            'final_embedding': final_embedding
        }


class AudioCaptioningModel(AudioModuleBase):
    """
    Audio captioning model for generating text descriptions of audio.
    
    Combines audio encoding with text generation.
    """
    
    def __init__(self, config: MultimodalConfig):
        super().__init__(config)
        
        self.config = config
        
        # Audio encoder
        self.audio_encoder = AudioEncoder(
            embed_dim=config.audio_embed_dim,
            num_layers=config.audio_encoder_layers,
            dropout=config.dropout
        )
        
        # Cross-modal fusion
        self.audio_text_fusion = nn.MultiheadAttention(
            embed_dim=config.shared_embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Text decoder
        self.text_decoder = self._build_text_decoder(config)
        
        # Spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )
        
    def _build_text_decoder(self, config: MultimodalConfig):
        """Build text decoder for captioning."""
        vocab_size = 50000  # Placeholder
        
        return nn.ModuleDict({
            'embedding': nn.Embedding(vocab_size, config.text_embed_dim),
            'pos_embedding': nn.Embedding(512, config.text_embed_dim),
            'decoder': nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=config.text_embed_dim,
                    nhead=config.num_heads,
                    dim_feedforward=config.text_embed_dim * 4,
                    dropout=config.dropout,
                    activation='gelu',
                    batch_first=True
                ),
                num_layers=config.text_encoder_layers
            ),
            'output_projection': nn.Linear(config.text_embed_dim, vocab_size)
        })
        
    def encode_audio_for_captioning(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode audio for captioning task."""
        # Convert to mel-spectrogram
        mel_spec = self.mel_transform(waveform)
        mel_spec = torch.log(mel_spec + 1e-8)
        
        if len(mel_spec.shape) == 3:
            mel_spec = mel_spec.unsqueeze(1)
            
        # Encode audio
        audio_features = self.audio_encoder(mel_spec)
        
        # Expand to sequence for cross-attention
        audio_features = audio_features.unsqueeze(1)  # [batch, 1, embed_dim]
        
        return audio_features
        
    def generate_caption(
        self,
        waveform: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0
    ) -> List[List[int]]:
        """Generate captions for audio."""
        batch_size = waveform.shape[0]
        device = waveform.device
        
        # Encode audio
        audio_features = self.encode_audio_for_captioning(waveform)
        
        # Initialize decoder input with start token
        start_token = 1  # Placeholder
        decoder_input = torch.full((batch_size, 1), start_token, device=device)
        
        generated_sequences = []
        
        for b in range(batch_size):
            sequence = [start_token]
            
            for _ in range(max_length):
                # Prepare decoder input
                current_input = torch.tensor([sequence], device=device)
                
                # Add positional embeddings
                pos_ids = torch.arange(len(sequence), device=device).unsqueeze(0)
                token_emb = self.text_decoder['embedding'](current_input)
                pos_emb = self.text_decoder['pos_embedding'](pos_ids)
                decoder_emb = token_emb + pos_emb
                
                # Decode with audio context
                output = self.text_decoder['decoder'](
                    decoder_emb,
                    audio_features[b:b+1]
                )
                
                # Get next token probabilities
                logits = self.text_decoder['output_projection'](output[:, -1:])
                probs = F.softmax(logits / temperature, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs.squeeze(), 1).item()
                
                if next_token == 2:  # End token
                    break
                    
                sequence.append(next_token)
                
            generated_sequences.append(sequence)
            
        return generated_sequences
        
    def forward(
        self,
        waveform: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for audio captioning.
        
        Args:
            waveform: Audio waveform [batch, samples]
            target_ids: Target caption token IDs [batch, seq_len] (for training)
            
        Returns:
            Dictionary with logits and optional loss
        """
        # Encode audio
        audio_features = self.encode_audio_for_captioning(waveform)
        
        if target_ids is not None and self.training:
            # Training mode - compute loss
            batch_size, seq_len = target_ids.shape
            
            # Prepare decoder input (shift target by one)
            decoder_input = target_ids[:, :-1]
            target_output = target_ids[:, 1:]
            
            # Add positional embeddings
            pos_ids = torch.arange(seq_len - 1, device=target_ids.device).unsqueeze(0).expand(batch_size, -1)
            token_emb = self.text_decoder['embedding'](decoder_input)
            pos_emb = self.text_decoder['pos_embedding'](pos_ids)
            decoder_emb = token_emb + pos_emb
            
            # Expand audio features to match sequence length
            audio_expanded = audio_features.expand(-1, seq_len - 1, -1)
            
            # Decode
            output = self.text_decoder['decoder'](decoder_emb, audio_expanded)
            
            # Get logits
            logits = self.text_decoder['output_projection'](output)
            
            # Compute loss
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target_output.reshape(-1),
                ignore_index=0  # Padding token
            )
            
            return {
                'logits': logits,
                'loss': loss,
                'audio_features': audio_features
            }
        else:
            # Inference mode - generate captions
            generated = self.generate_caption(waveform)
            
            return {
                'generated_captions': generated,
                'audio_features': audio_features
            }


# Factory functions
def create_clap(config: Optional[MultimodalConfig] = None) -> CLAP:
    """Create CLAP model."""
    if config is None:
        from ..audio_analysis.audio_config import get_music_config
        base_config = get_music_config()
        config = MultimodalConfig(
            model_type=MultimodalModelType.CLAP,
            sample_rate=base_config.sample_rate,
            n_fft=base_config.n_fft,
            hop_length=base_config.hop_length,
            n_mels=base_config.n_mels,
            normalization_method=base_config.normalization_method
        )
    return CLAP(config)


def create_imagebind_audio(config: Optional[MultimodalConfig] = None) -> ImageBindAudio:
    """Create ImageBind audio component."""
    if config is None:
        from ..audio_analysis.audio_config import get_music_config
        base_config = get_music_config()
        config = MultimodalConfig(
            model_type=MultimodalModelType.IMAGEBIND_AUDIO,
            sample_rate=base_config.sample_rate,
            n_fft=base_config.n_fft,
            hop_length=base_config.hop_length,
            n_mels=base_config.n_mels,
            normalization_method=base_config.normalization_method
        )
    return ImageBindAudio(config)


def create_audio_captioning_model(config: Optional[MultimodalConfig] = None) -> AudioCaptioningModel:
    """Create audio captioning model."""
    if config is None:
        from ..audio_analysis.audio_config import get_music_config
        base_config = get_music_config()
        config = MultimodalConfig(
            model_type=MultimodalModelType.WAVCAPS,
            sample_rate=base_config.sample_rate,
            n_fft=base_config.n_fft,
            hop_length=base_config.hop_length,
            n_mels=base_config.n_mels,
            normalization_method=base_config.normalization_method
        )
    return AudioCaptioningModel(config)


# Example usage
if __name__ == "__main__":
    # Test multimodal models
    sample_rate = 22050
    duration = 5  # seconds
    batch_size = 2
    
    # Generate test data
    waveform = torch.randn(batch_size, sample_rate * duration)
    
    # Dummy text data
    input_ids = torch.randint(1, 1000, (batch_size, 20))  # Random token IDs
    attention_mask = torch.ones_like(input_ids)
    
    print("Testing Multimodal Models:")
    
    # CLAP
    print("\n1. CLAP (Contrastive Language-Audio Pre-training):")
    clap = create_clap()
    clap_result = clap(waveform, input_ids, attention_mask)
    print(f"  Audio embeddings: {clap_result['audio_embeddings'].shape}")
    print(f"  Text embeddings: {clap_result['text_embeddings'].shape}")
    print(f"  Contrastive loss: {clap_result['total_loss']:.4f}")
    print(f"  Audio-to-text accuracy: {clap_result['audio_to_text_accuracy']:.2f}")
    print(f"  Text-to-audio accuracy: {clap_result['text_to_audio_accuracy']:.2f}")
    
    # ImageBind Audio
    print("\n2. ImageBind Audio:")
    imagebind = create_imagebind_audio()
    imagebind_result = imagebind(waveform)
    print(f"  Number of scales: {len(imagebind_result['scale_embeddings'])}")
    print(f"  Scale embeddings shape: {[emb.shape for emb in imagebind_result['scale_embeddings']]}")
    print(f"  Final embedding: {imagebind_result['final_embedding'].shape}")
    print(f"  Final embedding std: {imagebind_result['final_embedding'].std():.4f}")
    
    # Audio Captioning
    print("\n3. Audio Captioning Model:")
    captioner = create_audio_captioning_model()
    caption_result = captioner(waveform)
    print(f"  Generated captions length: {[len(cap) for cap in caption_result['generated_captions']]}")
    print(f"  Audio features: {caption_result['audio_features'].shape}")
    
    # Test retrieval capabilities
    print("\n4. Audio-Text Retrieval:")
    with torch.no_grad():
        # Get features
        audio_features = clap.get_audio_features(waveform)
        text_features = clap.get_text_features(input_ids, attention_mask)
        
        # Compute similarities
        similarities = torch.matmul(audio_features, text_features.t())
        print(f"  Similarity matrix: {similarities.shape}")
        print(f"  Max similarity: {similarities.max():.3f}")
        print(f"  Min similarity: {similarities.min():.3f}")
        print(f"  Diagonal similarities: {similarities.diag()}")
        
    print("\nAll multimodal models created and tested successfully!")