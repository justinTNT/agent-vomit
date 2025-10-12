import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class SequenceEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, 
                 d_ff=2048, max_seq_len=5000, dropout=0.1,
                 pooling_strategy='mean', pad_token_id=0):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.pooling_strategy = pooling_strategy
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Scale embeddings
        self.embed_scale = math.sqrt(d_model)
        
        # Import TransformerBlock from our generated module
        from modules.transformer_block import TransformerBlock
        
        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Optional projection head for pooled output
        self.projection = nn.Linear(d_model, d_model)
        
    def create_padding_mask(self, input_ids):
        """Create attention mask for padding tokens"""
        # Shape: (batch_size, seq_len)
        mask = (input_ids != self.pad_token_id).float()
        
        # Expand for attention heads: (batch_size, 1, 1, seq_len)
        mask = mask.unsqueeze(1).unsqueeze(1)
        
        return mask
    
    def pool_sequence(self, sequence_output, attention_mask=None):
        """Pool sequence output based on strategy"""
        if self.pooling_strategy == 'cls':
            # Use first token (CLS token)
            pooled = sequence_output[:, 0]
        elif self.pooling_strategy == 'mean':
            # Mean pooling with attention mask
            if attention_mask is not None:
                # Expand mask to match sequence_output dimensions
                mask_expanded = attention_mask.squeeze(1).squeeze(1).unsqueeze(-1)
                sum_embeddings = (sequence_output * mask_expanded).sum(1)
                sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
                pooled = sum_embeddings / sum_mask
            else:
                pooled = sequence_output.mean(dim=1)
        elif self.pooling_strategy == 'max':
            # Max pooling
            if attention_mask is not None:
                mask_expanded = attention_mask.squeeze(1).squeeze(1).unsqueeze(-1)
                sequence_output = sequence_output.masked_fill(
                    mask_expanded == 0, -1e9
                )
            pooled, _ = sequence_output.max(dim=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
            
        return pooled
    
    def forward(self, input_ids, attention_mask=None):
        # Create padding mask if not provided
        if attention_mask is None:
            attention_mask = self.create_padding_mask(input_ids)
        
        # Embed tokens
        x = self.token_embedding(input_ids) * self.embed_scale
        x = self.positional_encoding(x)
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask=attention_mask)
        
        # Final normalization
        sequence_output = self.final_norm(x)
        
        # Pool sequence
        pooled_output = self.pool_sequence(sequence_output, attention_mask)
        pooled_output = self.projection(pooled_output)
        
        return {
            'sequence_output': sequence_output,
            'pooled_output': pooled_output,
            'attention_mask': attention_mask
        }