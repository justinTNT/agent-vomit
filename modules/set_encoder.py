import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SetAttention(nn.Module):
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # No positional encoding - sets are permutation invariant
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, set_size, _ = x.shape
        
        # Project and reshape
        Q = self.q_proj(x).view(batch_size, set_size, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, set_size, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, set_size, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # mask shape: (batch, 1, 1, set_size) or (batch, set_size)
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            # Create attention mask
            mask = mask.float()
            mask = (1.0 - mask) * -1e9
            scores = scores + mask
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, set_size, self.d_model
        )
        output = self.out_proj(attn_output)
        
        return output


class SetTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        
        self.self_attention = SetAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out = self.self_attention(x, mask)
        x = self.norm1(x + attn_out)
        
        # FFN with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x


class InducedSetAttentionBlock(nn.Module):
    """ISAB from Set Transformer paper"""
    def __init__(self, d_model, n_heads, n_inducing_points, dropout=0.1):
        super().__init__()
        self.n_inducing_points = n_inducing_points
        
        # Learnable inducing points
        self.inducing_points = nn.Parameter(torch.randn(1, n_inducing_points, d_model))
        
        # Two multi-head attention blocks
        self.mab1 = SetAttention(d_model, n_heads, dropout)
        self.mab2 = SetAttention(d_model, n_heads, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Expand inducing points for batch
        I = self.inducing_points.expand(batch_size, -1, -1)
        
        # MAB(I, X) - inducing points attend to input
        H = self.mab1(torch.cat([I, x], dim=1), mask)[:, :self.n_inducing_points]
        H = self.norm1(I + H)
        
        # MAB(X, H) - input attends to inducing points
        out = self.mab2(torch.cat([x, H], dim=1), mask)[:, :x.size(1)]
        out = self.norm2(x + out)
        
        return out


class PoolingByMultiheadAttention(nn.Module):
    """PMA from Set Transformer paper"""
    def __init__(self, d_model, n_heads, n_seeds=1, dropout=0.1):
        super().__init__()
        self.n_seeds = n_seeds
        
        # Learnable seed vectors
        self.seed_vectors = nn.Parameter(torch.randn(1, n_seeds, d_model))
        
        self.attention = SetAttention(d_model, n_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Expand seed vectors for batch
        seeds = self.seed_vectors.expand(batch_size, -1, -1)
        
        # Seeds attend to set elements
        out = self.attention(torch.cat([seeds, x], dim=1), mask)[:, :self.n_seeds]
        out = self.norm(seeds + out)
        
        # If single seed, squeeze the dimension
        if self.n_seeds == 1:
            out = out.squeeze(1)
            
        return out


class SetEncoder(nn.Module):
    def __init__(self, input_dim, d_model=256, n_heads=8, n_layers=4,
                 d_ff=1024, dropout=0.1, pooling='mean',
                 use_isab=False, n_inducing_points=32):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.pooling = pooling
        self.use_isab = use_isab
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Encoder layers
        self.encoder_blocks = nn.ModuleList()
        for _ in range(n_layers):
            if use_isab:
                # Use induced set attention (more efficient for large sets)
                self.encoder_blocks.append(
                    InducedSetAttentionBlock(d_model, n_heads, n_inducing_points, dropout)
                )
            else:
                # Standard set transformer block
                self.encoder_blocks.append(
                    SetTransformerBlock(d_model, n_heads, d_ff, dropout)
                )
        
        # Pooling layer
        if pooling == 'mean':
            self.pool = lambda x, mask: self._masked_mean(x, mask)
        elif pooling == 'max':
            self.pool = lambda x, mask: self._masked_max(x, mask)
        elif pooling == 'sum':
            self.pool = lambda x, mask: self._masked_sum(x, mask)
        elif pooling == 'attention':
            # Learned attention pooling
            self.pool_attention = nn.Linear(d_model, 1)
        elif pooling == 'pma':
            # Pooling by multihead attention
            self.pool = PoolingByMultiheadAttention(d_model, n_heads, n_seeds=1, dropout=dropout)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)
        
    def _masked_mean(self, x, mask):
        if mask is not None:
            mask = mask.float()
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            x = x * mask
            return x.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return x.mean(dim=1)
    
    def _masked_max(self, x, mask):
        if mask is not None:
            mask = mask.float()
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            x = x.masked_fill(mask == 0, -1e9)
        return x.max(dim=1)[0]
    
    def _masked_sum(self, x, mask):
        if mask is not None:
            mask = mask.float()
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            x = x * mask
        return x.sum(dim=1)
    
    def forward(self, x, mask=None, return_elements=False):
        # x shape: (batch, set_size, input_dim)
        
        # Input projection
        x = self.input_projection(x)
        
        # Apply encoder blocks
        for block in self.encoder_blocks:
            x = block(x, mask)
        
        # Store set elements before pooling
        set_elements = x
        
        # Pooling
        if self.pooling == 'attention':
            # Compute attention scores
            scores = self.pool_attention(x).squeeze(-1)  # (batch, set_size)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            attn_weights = F.softmax(scores, dim=1).unsqueeze(1)  # (batch, 1, set_size)
            pooled = torch.bmm(attn_weights, x).squeeze(1)  # (batch, d_model)
        elif self.pooling == 'pma':
            pooled = self.pool(x, mask)
        else:
            pooled = self.pool(x, mask)
        
        # Output projection
        output = self.output_projection(pooled)
        
        if return_elements:
            return {
                'pooled': output,
                'elements': set_elements,
                'shape': set_elements.shape
            }
        else:
            return output
    
    def forward_with_pairs(self, x, mask=None):
        """Process a set considering pairwise interactions"""
        # Get individual element representations
        x = self.input_projection(x)
        
        # Apply encoder blocks
        for block in self.encoder_blocks:
            x = block(x, mask)
        
        batch_size, set_size, d_model = x.shape
        
        # Compute pairwise features
        # Expand x to (batch, set_size, 1, d_model) and (batch, 1, set_size, d_model)
        x1 = x.unsqueeze(2)
        x2 = x.unsqueeze(1)
        
        # Pairwise differences and products
        diff = x1 - x2  # (batch, set_size, set_size, d_model)
        prod = x1 * x2  # (batch, set_size, set_size, d_model)
        
        # Aggregate pairwise features
        pair_features = torch.cat([diff, prod], dim=-1)  # (batch, set_size, set_size, 2*d_model)
        
        # Pool over pairs for each element
        pair_summary = pair_features.mean(dim=2)  # (batch, set_size, 2*d_model)
        
        # Combine with original features
        combined = torch.cat([x, pair_summary[..., :d_model]], dim=-1)
        
        # Final pooling
        if hasattr(self, 'pool'):
            pooled = self.pool(combined[..., :d_model], mask)
        else:
            pooled = combined[..., :d_model].mean(dim=1)
            
        return self.output_projection(pooled)