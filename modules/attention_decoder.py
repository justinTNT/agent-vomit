import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Import from our generated modules
        from modules.transformer_block import MultiHeadAttention, FeedForwardNetwork
        
        # Self-attention (with causal mask)
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention (attending to encoder output)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, self_attn_mask=None, cross_attn_mask=None):
        # Self-attention with causal mask
        self_attn_out = self.self_attention(x, x, x, mask=self_attn_mask)
        x = self.norm1(x + self.dropout(self_attn_out))
        
        # Cross-attention to encoder output
        cross_attn_out = self.cross_attention(x, encoder_output, encoder_output, mask=cross_attn_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))
        
        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
        
        return x


class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6,
                 d_ff=2048, max_seq_len=5000, dropout=0.1,
                 pad_token_id=0, start_token_id=1, end_token_id=2):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.pad_token_id = pad_token_id
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
        self.max_seq_len = max_seq_len
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.embed_scale = math.sqrt(d_model)
        
        # Import positional encoding from sequence encoder
        from modules.sequence_encoder import PositionalEncoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Stack of decoder blocks
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def create_causal_mask(self, seq_len, device):
        """Create causal mask for self-attention"""
        # Upper triangular matrix of -inf
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        mask = mask.masked_fill(mask == 0, float(0.0))
        
        # Add batch and head dimensions
        mask = mask.unsqueeze(0).unsqueeze(0)
        return mask
    
    def create_padding_mask(self, input_ids):
        """Create attention mask for padding tokens"""
        # Shape: (batch_size, seq_len)
        mask = (input_ids != self.pad_token_id).float()
        
        # Expand for attention heads: (batch_size, 1, 1, seq_len)
        mask = mask.unsqueeze(1).unsqueeze(1)
        
        return mask
    
    def combine_masks(self, causal_mask, padding_mask):
        """Combine causal and padding masks"""
        # Causal mask is already in the right format for attention
        # Just need to expand padding mask if provided
        if padding_mask is not None:
            # Expand padding mask to match causal mask dimensions
            batch_size = padding_mask.shape[0]
            seq_len = causal_mask.shape[-1]
            # Create a mask that's 0 where we should attend, -inf where we shouldn't
            combined = causal_mask.expand(batch_size, -1, -1, -1).clone()
            # Apply padding mask by setting padded positions to -inf
            for i in range(batch_size):
                mask_seq = padding_mask[i, 0, 0, :]
                for j in range(seq_len):
                    if mask_seq[j] == 0:  # This is a padding token
                        combined[i, :, :, j] = float('-inf')  # Can't attend to padding
                        combined[i, :, j, :] = float('-inf')  # Padding can't attend to anything
        else:
            combined = causal_mask
        return combined
    
    def forward(self, input_ids, encoder_output, encoder_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create masks
        causal_mask = self.create_causal_mask(seq_len, device)
        padding_mask = self.create_padding_mask(input_ids)
        self_attn_mask = self.combine_masks(causal_mask, padding_mask)
        
        # Embed tokens
        x = self.token_embedding(input_ids) * self.embed_scale
        x = self.positional_encoding(x)
        
        # Pass through decoder blocks
        for decoder_block in self.decoder_blocks:
            x = decoder_block(
                x, 
                encoder_output,
                self_attn_mask=self_attn_mask,
                cross_attn_mask=encoder_mask
            )
        
        # Project to vocabulary
        logits = self.output_projection(x)
        
        return {
            'logits': logits,
            'hidden_states': x
        }
    
    def generate(self, encoder_output, encoder_mask=None, max_length=50, 
                 temperature=1.0, top_k=50, top_p=0.95):
        """Generate sequences autoregressively"""
        batch_size = encoder_output.shape[0]
        device = encoder_output.device
        
        # Start with start token
        generated = torch.full((batch_size, 1), self.start_token_id, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_length - 1):
            # Get predictions for current sequence
            outputs = self.forward(generated, encoder_output, encoder_mask)
            next_token_logits = outputs['logits'][:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            
            # Update finished sequences
            finished = finished | (next_tokens.squeeze(-1) == self.end_token_id)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_tokens], dim=-1)
            
            # Stop if all sequences are finished
            if finished.all():
                break
        
        return generated