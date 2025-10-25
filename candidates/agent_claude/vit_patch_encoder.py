import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


class ViTPatchEncoder(nn.Module):
    def __init__(self,
                 img_size: int = 224,               # Input image size
                 patch_size: int = 16,              # Patch size
                 in_channels: int = 3,              # Number of input channels
                 embed_dim: int = 768,              # Embedding dimension
                 n_heads: int = 12,                 # Number of attention heads
                 n_layers: int = 12,                # Number of transformer layers
                 d_ff: int = 3072,                  # Feed-forward dimension
                 dropout: float = 0.1,              # Dropout rate
                 **kwargs):                         # REQUIRED: Accept extra kwargs
        super().__init__()
        
        # Log unknown parameters
        if kwargs:
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        # Store configuration
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embedding = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            ViTEncoderLayer(
                embed_dim=embed_dim,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
        # Final layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters following ViT paper."""
        # Initialize patch embedding
        nn.init.xavier_uniform_(self.patch_embedding.weight)
        if self.patch_embedding.bias is not None:
            nn.init.constant_(self.patch_embedding.bias, 0)
        
        # Initialize positional embedding
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        
        # Initialize CLS token
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> dict:
        """
        Forward pass of the ViT patch encoder.
        
        Args:
            x: Input images of shape (batch, channels, height, width)
            mask: Optional mask for patches (batch, n_patches)
        
        Returns:
            Dictionary with keys:
                - 'last_hidden_state': All hidden states (batch, n_patches+1, embed_dim)
                - 'pooled_output': CLS token representation (batch, embed_dim)
                - 'patch_embeddings': Initial patch embeddings before transformer
        """
        batch_size = x.size(0)
        
        # Extract patches and embed
        # (batch, channels, height, width) -> (batch, embed_dim, n_patches_h, n_patches_w)
        patches = self.patch_embedding(x)
        
        # Flatten patches
        # (batch, embed_dim, n_patches_h, n_patches_w) -> (batch, n_patches, embed_dim)
        patches = patches.flatten(2).transpose(1, 2)
        
        # Verify number of patches
        seq_len = patches.size(1)
        if seq_len != self.n_patches:
            raise ValueError(
                f"Expected {self.n_patches} patches, got {seq_len}. "
                f"Check image size ({x.size(-2)}x{x.size(-1)}) and patch size ({self.patch_size})"
            )
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)
        
        # Store patch embeddings before adding positional encoding
        patch_embeddings = x.clone()
        
        # Add positional embedding
        x = x + self.pos_embedding[:, :seq_len + 1]
        x = self.dropout(x)
        
        # Handle mask if provided
        if mask is not None:
            # Expand mask to include CLS token (always visible)
            cls_mask = torch.ones(batch_size, 1, dtype=mask.dtype, device=mask.device)
            extended_mask = torch.cat([cls_mask, mask], dim=1)
            
            # Convert to attention mask format if needed
            if extended_mask.dtype == torch.bool:
                attn_mask = extended_mask.float().masked_fill(
                    extended_mask == 0, float('-inf')
                ).masked_fill(extended_mask == 1, 0.0)
            else:
                attn_mask = extended_mask
        else:
            attn_mask = None
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask=attn_mask)
        
        # Final layer normalization
        x = self.layer_norm(x)
        
        # Extract CLS token as pooled output
        pooled_output = x[:, 0]
        
        return {
            'last_hidden_state': x,
            'pooled_output': pooled_output,
            'patch_embeddings': patch_embeddings
        }


class ViTEncoderLayer(nn.Module):
    """Single ViT encoder layer."""
    def __init__(self,
                 embed_dim: int,
                 n_heads: int,
                 d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, embed_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass with pre-norm architecture."""
        # Pre-norm self-attention
        norm_x = self.norm1(x)
        attn_output, _ = self.self_attn(norm_x, norm_x, norm_x, attn_mask=mask)
        x = x + self.dropout(attn_output)
        
        # Pre-norm feed-forward
        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout(ff_output)
        
        return x