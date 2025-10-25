import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import math


class AttentionDecoder(nn.Module):
    def __init__(self,
                 vocab_size: int = 50000,           # Output vocabulary size
                 d_model: int = 512,                # Model dimension
                 n_heads: int = 8,                  # Number of attention heads
                 n_layers: int = 6,                 # Number of decoder layers
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
        
        # Decoder layers with cross-attention
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer
        self.register_buffer('pos_encoding', pe.unsqueeze(0))
        return pe
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.token_embedding.weight)
        nn.init.xavier_uniform_(self.output_projection.weight)
        if self.output_projection.bias is not None:
            nn.init.constant_(self.output_projection.bias, 0)
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for autoregressive generation."""
        # Create upper triangular matrix (True = mask out)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        # Convert to additive mask format
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self,
                input_ids: torch.Tensor,
                encoder_hidden_states: torch.Tensor = None,
                encoder_attention_mask: torch.Tensor = None,
                decoder_attention_mask: torch.Tensor = None,
                use_causal_mask: bool = True) -> dict:
        """
        Forward pass of the attention decoder.
        
        Args:
            input_ids: Target token IDs of shape (batch, tgt_seq_len)
            encoder_hidden_states: Optional encoder outputs (batch, src_seq_len, d_model)
            encoder_attention_mask: Optional encoder mask (batch, src_seq_len)
            decoder_attention_mask: Optional decoder mask (batch, tgt_seq_len)
            use_causal_mask: Whether to use causal masking for autoregression
        
        Returns:
            Dictionary with keys:
                - 'logits': Output logits (batch, tgt_seq_len, vocab_size)
                - 'hidden_states': Final hidden states
                - 'all_hidden_states': List of hidden states from each layer
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Add positional encoding
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")
        
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        # Create causal mask if needed
        self_attn_mask = None
        if use_causal_mask:
            self_attn_mask = self.create_causal_mask(seq_len, device)
        
        # Combine with provided mask if any
        if decoder_attention_mask is not None:
            if decoder_attention_mask.dtype == torch.bool:
                # Convert boolean to additive mask
                decoder_mask = decoder_attention_mask.float().masked_fill(
                    decoder_attention_mask == 0, float('-inf')
                ).masked_fill(decoder_attention_mask == 1, 0.0)
            else:
                decoder_mask = decoder_attention_mask
            
            if self_attn_mask is not None:
                # Combine masks
                self_attn_mask = self_attn_mask + decoder_mask
            else:
                self_attn_mask = decoder_mask
        
        # Collect hidden states
        all_hidden_states = []
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(
                x,
                encoder_hidden_states=encoder_hidden_states,
                self_attn_mask=self_attn_mask,
                encoder_attn_mask=encoder_attention_mask
            )
            all_hidden_states.append(x)
        
        # Final layer norm
        x = self.layer_norm(x)
        hidden_states = x
        
        # Project to vocabulary
        logits = self.output_projection(x)
        
        return {
            'logits': logits,
            'hidden_states': hidden_states,
            'all_hidden_states': all_hidden_states
        }


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with self-attention and cross-attention."""
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention (if encoder states provided)
        self.cross_attn = nn.MultiheadAttention(
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
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,
                x: torch.Tensor,
                encoder_hidden_states: torch.Tensor = None,
                self_attn_mask: torch.Tensor = None,
                encoder_attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with self-attention and optional cross-attention.
        """
        # Self-attention
        self_attn_output, _ = self.self_attn(x, x, x, attn_mask=self_attn_mask)
        x = x + self.dropout(self_attn_output)
        x = self.norm1(x)
        
        # Cross-attention (if encoder states provided)
        if encoder_hidden_states is not None:
            # Handle encoder attention mask
            cross_attn_mask = None
            if encoder_attn_mask is not None:
                if encoder_attn_mask.dtype == torch.bool:
                    # Convert boolean to additive mask
                    cross_attn_mask = encoder_attn_mask.float().masked_fill(
                        encoder_attn_mask == 0, float('-inf')
                    ).masked_fill(encoder_attn_mask == 1, 0.0)
                else:
                    cross_attn_mask = encoder_attn_mask
            
            cross_attn_output, _ = self.cross_attn(
                x, 
                encoder_hidden_states, 
                encoder_hidden_states,
                attn_mask=cross_attn_mask
            )
            x = x + self.dropout(cross_attn_output)
            x = self.norm2(x)
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm3(x)
        
        return x