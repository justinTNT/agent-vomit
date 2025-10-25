import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import math


class SequenceEncoder(nn.Module):
    def __init__(self,
                 vocab_size: int = 50000,           # Vocabulary size
                 d_model: int = 512,                # Model dimension
                 n_heads: int = 8,                  # Number of attention heads
                 n_layers: int = 6,                 # Number of transformer layers
                 d_ff: int = 2048,                  # Feed-forward dimension
                 max_seq_len: int = 5000,           # Maximum sequence length
                 dropout: float = 0.1,              # Dropout rate
                 **kwargs):                         # REQUIRED: Accept extra kwargs
        super().__init__()
        
        # Log unknown parameters
        if kwargs:
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        # Store configuration
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(max_seq_len, d_model)
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
        # Final layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> nn.Parameter:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not trainable)
        self.register_buffer('pos_encoding', pe.unsqueeze(0))
        return pe
    
    def _init_embeddings(self):
        """Initialize embeddings with Xavier uniform."""
        nn.init.xavier_uniform_(self.token_embedding.weight)
    
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor = None) -> dict:
        """
        Forward pass of the sequence encoder.
        
        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            attention_mask: Optional boolean mask (True=keep, False=mask)
        
        Returns:
            Dictionary with keys:
                - 'last_hidden_state': All hidden states (batch, seq_len, d_model)
                - 'pooled_output': Mean pooled representation (batch, d_model)
                - 'all_hidden_states': List of hidden states from each layer
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Add positional encoding
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")
        
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        # Collect hidden states
        all_hidden_states = []
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask=attention_mask)
            all_hidden_states.append(x)
        
        # Final layer norm
        x = self.layer_norm(x)
        
        # Pooled output (mean pooling with masking)
        if attention_mask is not None:
            # Expand mask to match hidden size
            expanded_mask = attention_mask.unsqueeze(-1).expand_as(x)
            sum_embeddings = torch.sum(x * expanded_mask, dim=1)
            sum_mask = torch.clamp(expanded_mask.sum(dim=1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = x.mean(dim=1)
        
        return {
            'last_hidden_state': x,
            'pooled_output': pooled_output,
            'all_hidden_states': all_hidden_states
        }


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer."""
    def __init__(self, 
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass with proper mask handling."""
        # Handle mask format for MultiheadAttention
        attn_mask = None
        if mask is not None:
            if mask.dtype == torch.bool:
                # Convert boolean mask to additive mask
                attn_mask = mask.float().masked_fill(mask == 0, float('-inf'))
                attn_mask = attn_mask.masked_fill(mask == 1, 0.0)
            else:
                attn_mask = mask
        
        # Self-attention with residual
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x