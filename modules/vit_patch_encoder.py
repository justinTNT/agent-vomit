import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Linear projection of flattened patches
        self.projection = nn.Linear(in_channels * patch_size * patch_size, embed_dim)
        
    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}x{W}) doesn't match expected size ({self.img_size}x{self.img_size})"
        
        # Extract patches
        # Reshape to (B, C, n_patches_h, patch_size, n_patches_w, patch_size)
        x = x.reshape(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        # Permute to (B, n_patches_h, n_patches_w, patch_size, patch_size, C)
        x = x.permute(0, 2, 4, 3, 5, 1)
        # Flatten patches to (B, n_patches, patch_size * patch_size * C)
        x = x.reshape(B, self.n_patches, -1)
        
        # Linear projection
        x = self.projection(x)
        
        return x


class PositionalEncoding2D(nn.Module):
    def __init__(self, embed_dim, n_patches, temperature=10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_patches = n_patches
        self.grid_size = int(math.sqrt(n_patches))
        
        # Create 2D positional encoding
        pe = torch.zeros(n_patches, embed_dim)
        
        # Create position indices
        y_pos = torch.arange(self.grid_size).unsqueeze(1).repeat(1, self.grid_size).reshape(-1)
        x_pos = torch.arange(self.grid_size).repeat(self.grid_size)
        
        # Create sinusoidal encodings
        dim_t = torch.arange(embed_dim // 4).float()
        dim_t = temperature ** (2 * dim_t / (embed_dim // 2))
        
        # Apply sin/cos for x and y coordinates
        pe[:, 0::4] = torch.sin(x_pos.unsqueeze(1) / dim_t)
        pe[:, 1::4] = torch.cos(x_pos.unsqueeze(1) / dim_t)
        pe[:, 2::4] = torch.sin(y_pos.unsqueeze(1) / dim_t)
        pe[:, 3::4] = torch.cos(y_pos.unsqueeze(1) / dim_t)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: (batch_size, n_patches + 1, embed_dim) if CLS token is present
        # or (batch_size, n_patches, embed_dim) otherwise
        seq_len = x.size(1)
        
        if seq_len == self.n_patches + 1:
            # CLS token is present, don't add PE to it
            x[:, 1:] = x[:, 1:] + self.pe
        else:
            # No CLS token
            x = x + self.pe
            
        return x


class ViTPatchEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, 
                 embed_dim=768, n_heads=12, n_layers=12, d_ff=3072,
                 dropout=0.1, use_cls_token=True, pool_type='cls'):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        self.pool_type = pool_type
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # CLS token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(embed_dim, self.n_patches)
        self.pos_dropout = nn.Dropout(dropout)
        
        # Import transformer blocks
        from modules.transformer_block import TransformerBlock
        
        # Transformer encoder layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # Optional projection head
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def interpolate_pos_encoding(self, x, h, w):
        """Interpolate positional encoding for different image sizes"""
        npatch = x.shape[1] - 1 if self.use_cls_token else x.shape[1]
        N = self.pos_encoding.pe.shape[0]
        
        if npatch == N:
            return self.pos_encoding(x)
        
        # Interpolate position encoding
        dim = self.embed_dim
        pos_embed = self.pos_encoding.pe.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(h // self.patch_size, w // self.patch_size), mode='bicubic', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim).squeeze(0)
        
        if self.use_cls_token:
            x[:, 1:] = x[:, 1:] + pos_embed
        else:
            x = x + pos_embed
            
        return x
    
    def forward(self, x, return_all_tokens=False):
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Add CLS token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)  # (B, 1 + n_patches, embed_dim)
        
        # Add positional encoding
        if H == self.img_size and W == self.img_size:
            x = self.pos_encoding(x)
        else:
            # Handle different image sizes with interpolation
            x = self.interpolate_pos_encoding(x, H, W)
        x = self.pos_dropout(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Final norm
        x = self.final_norm(x)
        
        # Pool features
        if self.pool_type == 'cls' and self.use_cls_token:
            pooled = x[:, 0]
        elif self.pool_type == 'mean':
            if self.use_cls_token:
                pooled = x[:, 1:].mean(dim=1)
            else:
                pooled = x.mean(dim=1)
        elif self.pool_type == 'max':
            if self.use_cls_token:
                pooled, _ = x[:, 1:].max(dim=1)
            else:
                pooled, _ = x.max(dim=1)
        else:
            raise ValueError(f"Unknown pooling type: {self.pool_type}")
        
        # Apply projection head
        pooled = self.projection_head(pooled)
        
        if return_all_tokens:
            return {
                'pooled': pooled,
                'tokens': x,
                'patch_embeddings': x[:, 1:] if self.use_cls_token else x
            }
        else:
            return pooled
    
    def get_attention_maps(self, x):
        """Extract attention maps from all layers"""
        attention_maps = []
        
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.pos_encoding(x)
        x = self.pos_dropout(x)
        
        # We would need to modify TransformerBlock to return attention weights
        # For now, just return the final representation
        for block in self.transformer_blocks:
            x = block(x)
        
        return x