import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import math
from typing import Dict, Optional, Tuple, List


class PatchEmbedding(nn.Module):
    """
    Convert audio spectrogram to sequence of patches for transformer processing.
    Inspired by Vision Transformer (ViT) applied to audio spectrograms.
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int] = (1024, 128),  # (time_frames, freq_bins)
        patch_size: Tuple[int, int] = (16, 16),
        in_channels: int = 1,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, time, freq]
        B, C, H, W = x.shape
        
        # Ensure input size is compatible
        assert H >= self.patch_size[0] and W >= self.patch_size[1], \
            f"Input size {(H, W)} too small for patch size {self.patch_size}"
        
        # Project to patches: [batch, embed_dim, grid_h, grid_w]
        x = self.proj(x)
        
        # Flatten to sequence: [batch, num_patches, embed_dim]
        x = x.flatten(2).transpose(1, 2)
        
        return x


class AudioSpectrogramTransformer(nn.Module):
    """
    Audio Spectrogram Transformer for audio classification and representation learning.
    
    Based on "AST: Audio Spectrogram Transformer" (Gong et al., 2021)
    Applies transformer architecture to mel-spectrogram patches.
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int] = (1024, 128),
        patch_size: Tuple[int, int] = (16, 16),
        num_classes: int = 527,  # AudioSet classes
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        sample_rate: int = 16000,
        n_mels: int = 128,
        **kwargs
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Mel-spectrogram computation
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=160,
            n_mels=n_mels,
            f_min=0,
            f_max=sample_rate // 2
        )
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=1,
            embed_dim=embed_dim
        )
        
        num_patches = self.patch_embed.num_patches
        
        # Learnable embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            ) for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)
                
    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract mel-spectrogram features from waveform."""
        # waveform: [batch, samples] or [batch, channels, samples]
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)  # Remove channel dim if mono
            
        # Compute mel-spectrogram
        mel_spec = self.mel_transform(waveform)
        
        # Convert to log scale
        mel_spec = torch.log(mel_spec.clamp(min=1e-8))
        
        # Add channel dimension: [batch, 1, time, freq]
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(1)
            
        return mel_spec
        
    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through transformer backbone."""
        # Extract patches
        x = self.patch_embed(x)  # [batch, num_patches, embed_dim]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Store intermediate representations
        features = {}
        
        # Forward through transformer blocks
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in [3, 7, 11]:  # Store intermediate features
                features[f'block_{i}'] = x
                
        # Final normalization
        x = self.norm(x)
        features['final'] = x
        
        return features
        
    def forward(
        self, 
        waveform: Optional[torch.Tensor] = None,
        spectrogram: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for classification.
        
        Args:
            waveform: Raw audio [batch, samples]
            spectrogram: Pre-computed spectrogram [batch, 1, time, freq]
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing logits and optionally features
        """
        if spectrogram is None:
            if waveform is None:
                raise ValueError("Either waveform or spectrogram must be provided")
            spectrogram = self.extract_features(waveform)
            
        # Forward through backbone
        features = self.forward_features(spectrogram)
        
        # Classification using CLS token
        cls_features = features['final'][:, 0]  # [batch, embed_dim]
        logits = self.head(cls_features)
        
        result = {
            'logits': logits,
            'cls_features': cls_features,
            'spectrogram': spectrogram
        }
        
        if return_features:
            result['features'] = features
            
        return result
        
    def get_attention_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract attention maps from all transformer blocks."""
        features = self.forward_features(x)
        attention_maps = []
        
        # Re-run forward pass to collect attention weights
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        
        for block in self.blocks:
            x, attn = block(x, return_attention=True)
            attention_maps.append(attn)
            
        return attention_maps


class TransformerBlock(nn.Module):
    """Standard transformer block with multi-head self-attention and MLP."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        # Self-attention
        x_norm = self.norm1(x)
        if return_attention:
            attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm, need_weights=True)
        else:
            attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
            attn_weights = None
            
        x = x + self.dropout1(attn_out)
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        if return_attention:
            return x, attn_weights
        return x


# Convenience functions for common configurations
def ast_base_384(num_classes: int = 527, **kwargs) -> AudioSpectrogramTransformer:
    """AST-Base model with 384x384 input size."""
    return AudioSpectrogramTransformer(
        img_size=(384, 384),
        patch_size=(16, 16),
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        **kwargs
    )

def ast_small_224(num_classes: int = 527, **kwargs) -> AudioSpectrogramTransformer:
    """AST-Small model with 224x224 input size."""
    return AudioSpectrogramTransformer(
        img_size=(224, 224),
        patch_size=(16, 16),
        num_classes=num_classes,
        embed_dim=384,
        depth=12,
        num_heads=6,
        **kwargs
    )

def ast_tiny_224(num_classes: int = 527, **kwargs) -> AudioSpectrogramTransformer:
    """AST-Tiny model with 224x224 input size."""
    return AudioSpectrogramTransformer(
        img_size=(224, 224),
        patch_size=(16, 16),
        num_classes=num_classes,
        embed_dim=192,
        depth=12,
        num_heads=3,
        **kwargs
    )


# Example usage
if __name__ == "__main__":
    # Create model
    model = ast_base_384(num_classes=10)
    
    # Test with dummy waveform
    batch_size = 2
    waveform = torch.randn(batch_size, 16000 * 10)  # 10 seconds
    
    # Forward pass
    output = model(waveform=waveform, return_features=True)
    
    print(f"Logits shape: {output['logits'].shape}")
    print(f"CLS features shape: {output['cls_features'].shape}")
    print(f"Spectrogram shape: {output['spectrogram'].shape}")
    print(f"Available features: {list(output['features'].keys())}")