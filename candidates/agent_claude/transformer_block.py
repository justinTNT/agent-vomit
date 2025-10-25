import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


class TransformerBlock(nn.Module):
    def __init__(self, 
                 d_model: int = 512,      # Hidden dimension
                 n_heads: int = 8,        # Number of attention heads
                 d_ff: int = 2048,        # Feed-forward dimension
                 dropout: float = 0.1,    # Dropout rate
                 **kwargs):               # REQUIRED: Accept extra kwargs
        super().__init__()
        
        # Log unknown parameters
        if kwargs:
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the transformer block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional mask tensor. Can be boolean (True=keep) or additive
        
        Returns:
            Output tensor of same shape as input
        """
        # Handle mask format
        attn_mask = None
        if mask is not None:
            if mask.dtype == torch.bool:
                # Convert boolean mask to additive mask
                # True = keep, False = mask out
                # MultiheadAttention expects float mask where -inf = masked
                attn_mask = mask.float().masked_fill(mask == 0, float('-inf'))
                attn_mask = attn_mask.masked_fill(mask == 1, 0.0)
            else:
                # Assume it's already an additive mask
                attn_mask = mask
        
        # Multi-head attention with residual connection
        attn_output, _ = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed-forward with residual connection
        ff_output = self.feedforward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x