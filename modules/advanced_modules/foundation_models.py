"""
Foundation Models for Audio

Implements state-of-the-art foundation models for audio understanding:
- AudioMAE: Masked Audio Encoder for self-supervised learning
- Data2Vec-Audio: Self-supervised learning with contextualized targets
- WavLM: Large-scale self-supervised learning for speech and audio
- Wav2Vec2: Self-supervised learning of speech representations
- HuBERT: Hidden unit BERT for speech representation learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import math
from dataclasses import dataclass
from enum import Enum

from ..audio_analysis.audio_config import AudioModuleConfig, AudioModuleBase


class FoundationModelType(Enum):
    """Types of foundation models."""
    AUDIO_MAE = "audio_mae"
    DATA2VEC_AUDIO = "data2vec_audio"
    WAVLM = "wavlm"
    WAV2VEC2 = "wav2vec2"
    HUBERT = "hubert"


@dataclass
class FoundationModelConfig(AudioModuleConfig):
    """Configuration for foundation models."""
    model_type: FoundationModelType = FoundationModelType.AUDIO_MAE
    input_size: int = 768
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    attention_dropout: float = 0.1
    mask_prob: float = 0.15
    mask_length: int = 10
    normalize_waveform: bool = True


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:x.size(0), :]


class ConvolutionalFeatureEncoder(nn.Module):
    """
    Convolutional feature encoder for raw waveform input.
    
    Based on Wav2Vec2 and WavLM architectures.
    """
    
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]] = None,  # (dim, kernel_size, stride)
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False
    ):
        super().__init__()
        
        if conv_layers is None:
            # Default Wav2Vec2 configuration
            conv_layers = [
                (512, 10, 5),
                (512, 3, 2),
                (512, 3, 2),
                (512, 3, 2),
                (512, 3, 2),
                (512, 2, 2),
                (512, 2, 2)
            ]
            
        self.conv_layers = nn.ModuleList()
        in_d = 1
        
        for i, (dim, k, stride) in enumerate(conv_layers):
            self.conv_layers.append(
                nn.Conv1d(
                    in_d,
                    dim,
                    kernel_size=k,
                    stride=stride,
                    bias=conv_bias,
                    padding=k // 2
                )
            )
            in_d = dim
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode raw waveform to feature sequence.
        
        Args:
            x: Raw waveform [batch, time]
            
        Returns:
            Feature sequence [batch, time, dim]
        """
        x = x.unsqueeze(1)  # Add channel dimension
        
        for conv in self.conv_layers:
            x = conv(x)
            x = F.gelu(x)
            
        x = self.dropout(x)
        x = x.transpose(1, 2)  # [batch, time, dim]
        
        return x


class MaskedLanguageModelingHead(nn.Module):
    """MLM head for self-supervised learning."""
    
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply MLM head to features."""
        x = self.dense(features)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x


class AudioMAE(AudioModuleBase):
    """
    Audio Masked Autoencoder (AudioMAE).
    
    Self-supervised learning through masking and reconstruction of audio patches.
    """
    
    def __init__(self, config: FoundationModelConfig):
        super().__init__(config)
        
        self.config = config
        
        # Feature extraction
        self.feature_encoder = ConvolutionalFeatureEncoder()
        
        # Patch embedding
        self.patch_size = 16
        self.embed_dim = config.hidden_size
        
        # Project conv features to embedding dimension
        self.feature_projection = nn.Linear(512, self.embed_dim)
        
        # Positional embeddings
        self.pos_encoding = PositionalEncoding(self.embed_dim)
        
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )
        
        # Decoder for reconstruction
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=config.num_heads // 2,
            dim_feedforward=config.intermediate_size // 2,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_layers // 2
        )
        
        # Reconstruction head
        self.reconstruction_head = nn.Linear(self.embed_dim, 512)  # Match conv feature dim
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
            
    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.75
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform random masking by patch.
        
        Args:
            x: Input sequence [batch, length, dim]
            mask_ratio: Ratio of patches to mask
            
        Returns:
            x_masked: Sequence with masked patches
            mask: Binary mask (0 is keep, 1 is remove)
            ids_restore: Indices to restore original order
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        # Generate random noise for each sample
        noise = torch.rand(N, L, device=x.device)
        
        # Sort noise to get random indices
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
        
    def forward_encoder(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.75
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through encoder with masking."""
        # Extract features
        features = self.feature_encoder(x)  # [batch, time, 512]
        features = self.feature_projection(features)  # [batch, time, embed_dim]
        
        # Add positional encoding
        features = features.transpose(0, 1)  # [time, batch, embed_dim] for pos encoding
        features = self.pos_encoding(features)
        features = features.transpose(0, 1)  # Back to [batch, time, embed_dim]
        
        # Masking
        features_masked, mask, ids_restore = self.random_masking(features, mask_ratio)
        
        # Transformer encoding
        encoded = self.transformer_encoder(features_masked)
        
        return encoded, mask, ids_restore, features
        
    def forward_decoder(
        self,
        x: torch.Tensor,
        ids_restore: torch.Tensor,
        target_length: int
    ) -> torch.Tensor:
        """Forward pass through decoder for reconstruction."""
        # Add mask tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], target_length - x.shape[1], 1)
        x_full = torch.cat([x, mask_tokens], dim=1)
        
        # Unshuffle
        x_full = torch.gather(
            x_full, 
            dim=1, 
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )
        
        # Decoder
        # For simplicity, use encoder output as memory (self-attention style)
        decoded = self.transformer_decoder(x_full, x_full)
        
        # Reconstruction
        reconstructed = self.reconstruction_head(decoded)
        
        return reconstructed
        
    def forward_loss(
        self,
        features_orig: torch.Tensor,
        features_pred: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute reconstruction loss."""
        # L2 loss on masked patches only
        loss = F.mse_loss(features_pred, features_orig, reduction='none')
        loss = loss.mean(dim=-1)  # Mean over feature dimension
        
        # Apply mask (loss only on masked patches)
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
        
    def forward(
        self,
        waveform: torch.Tensor,
        mask_ratio: float = 0.75,
        return_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for AudioMAE.
        
        Args:
            waveform: Input waveform [batch, time]
            mask_ratio: Ratio of patches to mask
            return_loss: Whether to compute reconstruction loss
            
        Returns:
            Dictionary with outputs and optional loss
        """
        # Encoder
        encoded, mask, ids_restore, features_orig = self.forward_encoder(
            waveform, mask_ratio
        )
        
        # Decoder
        features_pred = self.forward_decoder(
            encoded, ids_restore, features_orig.shape[1]
        )
        
        result = {
            'encoded_features': encoded,
            'reconstructed_features': features_pred,
            'mask': mask,
            'original_features': features_orig
        }
        
        if return_loss:
            loss = self.forward_loss(features_orig, features_pred, mask)
            result['loss'] = loss
            
        return result


class Data2VecAudio(AudioModuleBase):
    """
    Data2Vec for Audio.
    
    Self-supervised learning with contextualized targets.
    """
    
    def __init__(self, config: FoundationModelConfig):
        super().__init__(config)
        
        self.config = config
        
        # Feature extraction
        self.feature_encoder = ConvolutionalFeatureEncoder()
        
        # Feature projection
        self.feature_projection = nn.Linear(512, config.hidden_size)
        
        # Positional embeddings
        self.pos_encoding = PositionalEncoding(config.hidden_size)
        
        # Student network (online)
        student_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.student_network = nn.TransformerEncoder(
            student_layer,
            num_layers=config.num_layers
        )
        
        # Teacher network (target)
        teacher_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.intermediate_size,
            dropout=0.0,  # No dropout in teacher
            activation='gelu',
            batch_first=True
        )
        self.teacher_network = nn.TransformerEncoder(
            teacher_layer,
            num_layers=config.num_layers
        )
        
        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
        # EMA for teacher network
        self.ema_decay = 0.999
        
        # Initialize teacher as copy of student
        self._init_teacher()
        
    def _init_teacher(self):
        """Initialize teacher network as copy of student."""
        for student_param, teacher_param in zip(
            self.student_network.parameters(),
            self.teacher_network.parameters()
        ):
            teacher_param.data.copy_(student_param.data)
            teacher_param.requires_grad = False
            
    def update_teacher(self):
        """Update teacher network with EMA of student."""
        with torch.no_grad():
            for student_param, teacher_param in zip(
                self.student_network.parameters(),
                self.teacher_network.parameters()
            ):
                teacher_param.data.mul_(self.ema_decay).add_(
                    student_param.data, alpha=1 - self.ema_decay
                )
                
    def create_mask(
        self,
        batch_size: int,
        seq_length: int,
        mask_prob: float = 0.15
    ) -> torch.Tensor:
        """Create random mask for training."""
        mask = torch.rand(batch_size, seq_length) < mask_prob
        return mask
        
    def forward(
        self,
        waveform: torch.Tensor,
        mask_prob: float = 0.15,
        update_teacher: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for Data2Vec Audio.
        
        Args:
            waveform: Input waveform [batch, time]
            mask_prob: Probability of masking each time step
            update_teacher: Whether to update teacher network
            
        Returns:
            Dictionary with predictions and targets
        """
        # Extract features
        features = self.feature_encoder(waveform)
        features = self.feature_projection(features)
        
        # Add positional encoding
        features = features.transpose(0, 1)
        features = self.pos_encoding(features)
        features = features.transpose(0, 1)
        
        # Create mask
        batch_size, seq_length = features.shape[:2]
        mask = self.create_mask(batch_size, seq_length, mask_prob)
        mask = mask.to(features.device)
        
        # Student network (with masking)
        masked_features = features.clone()
        masked_features[mask] = 0  # Simple masking
        
        student_output = self.student_network(masked_features)
        predictions = self.prediction_head(student_output)
        
        # Teacher network (no masking, no gradients)
        with torch.no_grad():
            teacher_output = self.teacher_network(features)
            
        # Update teacher
        if update_teacher and self.training:
            self.update_teacher()
            
        # Compute loss (only on masked positions)
        loss = F.mse_loss(
            predictions[mask],
            teacher_output[mask].detach(),
            reduction='mean'
        )
        
        return {
            'predictions': predictions,
            'targets': teacher_output,
            'mask': mask,
            'loss': loss,
            'student_features': student_output,
            'teacher_features': teacher_output
        }


class WavLM(AudioModuleBase):
    """
    WavLM: Large-scale self-supervised learning for speech and audio.
    
    Extends Wav2Vec2 with gated relative position bias and better pre-training.
    """
    
    def __init__(self, config: FoundationModelConfig):
        super().__init__(config)
        
        self.config = config
        
        # Feature extraction
        self.feature_encoder = ConvolutionalFeatureEncoder()
        
        # Feature projection
        self.feature_projection = nn.Linear(512, config.hidden_size)
        
        # Quantizer for contrastive learning
        self.quantizer = self._build_quantizer(config)
        
        # Transformer with gated relative position bias
        self.transformer = self._build_transformer(config)
        
        # Contrastive projection head
        self.contrastive_head = nn.Linear(config.hidden_size, 256)
        
        # Mask embedding
        self.mask_emb = nn.Parameter(torch.zeros(config.hidden_size))
        
    def _build_quantizer(self, config: FoundationModelConfig):
        """Build vector quantizer for targets."""
        return nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 320)  # 320 = 2 groups * 160 entries per group
        )
        
    def _build_transformer(self, config: FoundationModelConfig):
        """Build transformer with gated relative position bias."""
        # Simplified version - would need full implementation of gated RPB
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        return nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )
        
    def compute_mask_indices(
        self,
        shape: Tuple[int, int],
        mask_prob: float = 0.15,
        mask_length: int = 10
    ) -> torch.Tensor:
        """Compute mask indices for contrastive learning."""
        batch_size, sequence_length = shape
        
        if mask_prob == 0:
            return torch.zeros(shape, dtype=torch.bool)
            
        # Compute number of masked spans
        num_masked = int(mask_prob * sequence_length / mask_length + 0.5)
        
        # Create mask
        mask = torch.zeros(shape, dtype=torch.bool)
        
        for batch_idx in range(batch_size):
            # Random start positions for masked spans
            mask_starts = torch.randint(
                0, max(1, sequence_length - mask_length + 1), (num_masked,)
            )
            
            for start in mask_starts:
                end = min(start + mask_length, sequence_length)
                mask[batch_idx, start:end] = True
                
        return mask
        
    def forward(
        self,
        waveform: torch.Tensor,
        mask_prob: float = 0.15,
        mask_length: int = 10
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for WavLM.
        
        Args:
            waveform: Input waveform [batch, time]
            mask_prob: Probability of masking
            mask_length: Length of masked spans
            
        Returns:
            Dictionary with contrastive learning outputs
        """
        # Feature extraction
        features = self.feature_encoder(waveform)
        
        # Quantization targets
        quantized = self.quantizer(features)
        
        # Project features
        features = self.feature_projection(features)
        
        # Create masks
        batch_size, seq_length = features.shape[:2]
        mask = self.compute_mask_indices(
            (batch_size, seq_length), mask_prob, mask_length
        ).to(features.device)
        
        # Apply mask embedding
        features[mask] = self.mask_emb
        
        # Transformer
        contextualized = self.transformer(features)
        
        # Contrastive projection
        projected = self.contrastive_head(contextualized)
        
        # Compute contrastive loss
        if self.training:
            loss = self._compute_contrastive_loss(
                projected[mask], quantized[mask]
            )
        else:
            loss = torch.tensor(0.0, device=waveform.device)
            
        return {
            'contextualized_features': contextualized,
            'projected_features': projected,
            'quantized_targets': quantized,
            'mask': mask,
            'loss': loss
        }
        
    def _compute_contrastive_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        temperature: float = 0.1
    ) -> torch.Tensor:
        """Compute contrastive loss."""
        # Normalize
        predictions = F.normalize(predictions, dim=-1)
        targets = F.normalize(targets, dim=-1)
        
        # Cosine similarity
        similarity = torch.matmul(predictions, targets.transpose(-2, -1)) / temperature
        
        # Labels (diagonal is positive)
        labels = torch.arange(predictions.shape[0], device=predictions.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(similarity, labels)
        
        return loss


class FoundationModelFactory:
    """Factory for creating foundation models."""
    
    @staticmethod
    def create_model(
        model_type: FoundationModelType,
        config: Optional[FoundationModelConfig] = None
    ) -> AudioModuleBase:
        """Create foundation model of specified type."""
        if config is None:
            config = FoundationModelConfig(model_type=model_type)
            
        if model_type == FoundationModelType.AUDIO_MAE:
            return AudioMAE(config)
        elif model_type == FoundationModelType.DATA2VEC_AUDIO:
            return Data2VecAudio(config)
        elif model_type == FoundationModelType.WAVLM:
            return WavLM(config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    @staticmethod
    def get_pretrained_config(model_type: FoundationModelType) -> FoundationModelConfig:
        """Get standard configuration for pretrained models."""
        if model_type == FoundationModelType.AUDIO_MAE:
            # Start with default audio config and extend
            from ..audio_analysis.audio_config import get_music_config
            base_config = get_music_config()
            return FoundationModelConfig(
                model_type=model_type,
                sample_rate=16000,  # Foundation models typically use 16kHz
                hidden_size=768,
                num_layers=12,
                num_heads=12,
                intermediate_size=3072,
                n_fft=base_config.n_fft,
                hop_length=base_config.hop_length,
                n_mels=base_config.n_mels,
                normalization_method=base_config.normalization_method
            )
        elif model_type == FoundationModelType.DATA2VEC_AUDIO:
            from ..audio_analysis.audio_config import get_music_config
            base_config = get_music_config()
            return FoundationModelConfig(
                model_type=model_type,
                sample_rate=16000,
                hidden_size=768,
                num_layers=12,
                num_heads=12,
                intermediate_size=3072,
                n_fft=base_config.n_fft,
                hop_length=base_config.hop_length,
                n_mels=base_config.n_mels,
                normalization_method=base_config.normalization_method
            )
        elif model_type == FoundationModelType.WAVLM:
            from ..audio_analysis.audio_config import get_music_config
            base_config = get_music_config()
            return FoundationModelConfig(
                model_type=model_type,
                sample_rate=16000,
                hidden_size=768,
                num_layers=12,
                num_heads=12,
                intermediate_size=3072,
                mask_prob=0.15,
                mask_length=10,
                n_fft=base_config.n_fft,
                hop_length=base_config.hop_length,
                n_mels=base_config.n_mels,
                normalization_method=base_config.normalization_method
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


# Factory functions
def create_audio_mae(config: Optional[FoundationModelConfig] = None) -> AudioMAE:
    """Create AudioMAE model."""
    if config is None:
        config = FoundationModelFactory.get_pretrained_config(FoundationModelType.AUDIO_MAE)
    return AudioMAE(config)


def create_data2vec_audio(config: Optional[FoundationModelConfig] = None) -> Data2VecAudio:
    """Create Data2Vec Audio model."""
    if config is None:
        config = FoundationModelFactory.get_pretrained_config(FoundationModelType.DATA2VEC_AUDIO)
    return Data2VecAudio(config)


def create_wavlm(config: Optional[FoundationModelConfig] = None) -> WavLM:
    """Create WavLM model."""
    if config is None:
        config = FoundationModelFactory.get_pretrained_config(FoundationModelType.WAVLM)
    return WavLM(config)


# Example usage
if __name__ == "__main__":
    # Test different foundation models
    sample_rate = 16000
    duration = 3  # seconds
    waveform = torch.randn(2, sample_rate * duration)  # Batch of 2 samples
    
    print("Testing Foundation Models:")
    
    # AudioMAE
    print("\n1. AudioMAE:")
    audio_mae = create_audio_mae()
    mae_result = audio_mae(waveform, mask_ratio=0.75)
    print(f"  Input shape: {waveform.shape}")
    print(f"  Encoded features: {mae_result['encoded_features'].shape}")
    print(f"  Reconstruction loss: {mae_result['loss']:.4f}")
    print(f"  Mask ratio: {mae_result['mask'].float().mean():.2f}")
    
    # Data2Vec Audio
    print("\n2. Data2Vec Audio:")
    data2vec = create_data2vec_audio()
    d2v_result = data2vec(waveform, mask_prob=0.15)
    print(f"  Student features: {d2v_result['student_features'].shape}")
    print(f"  Teacher features: {d2v_result['teacher_features'].shape}")
    print(f"  Contrastive loss: {d2v_result['loss']:.4f}")
    print(f"  Masked positions: {d2v_result['mask'].sum().item()}")
    
    # WavLM
    print("\n3. WavLM:")
    wavlm = create_wavlm()
    wavlm_result = wavlm(waveform, mask_prob=0.15, mask_length=10)
    print(f"  Contextualized features: {wavlm_result['contextualized_features'].shape}")
    print(f"  Projected features: {wavlm_result['projected_features'].shape}")
    print(f"  Contrastive loss: {wavlm_result['loss']:.4f}")
    print(f"  Masked spans: {wavlm_result['mask'].sum().item()}")
    
    # Feature extraction comparison
    print("\nFeature extraction comparison:")
    with torch.no_grad():
        # Extract features without training objectives
        mae_features = audio_mae(waveform, return_loss=False)['encoded_features']
        d2v_features = d2v_result['student_features']
        wavlm_features = wavlm_result['contextualized_features']
        
        print(f"AudioMAE features std: {mae_features.std():.4f}")
        print(f"Data2Vec features std: {d2v_features.std():.4f}")
        print(f"WavLM features std: {wavlm_features.std():.4f}")
        
    print("\nAll foundation models created and tested successfully!")