import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SequenceToSequenceModel(nn.Module):
    def __init__(self, encoder, decoder, src_vocab_size=None, tgt_vocab_size=None,
                 share_embeddings=False, tie_embeddings=False):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.share_embeddings = share_embeddings
        self.tie_embeddings = tie_embeddings
        
        # Check if we need to handle embeddings
        self.handle_embeddings = src_vocab_size is not None and tgt_vocab_size is not None
        
        if self.handle_embeddings:
            # Get embedding dimensions from encoder/decoder
            if hasattr(encoder, 'token_embedding'):
                self.src_embed_dim = encoder.token_embedding.embedding_dim
            else:
                self.src_embed_dim = encoder.d_model if hasattr(encoder, 'd_model') else 512
                
            if hasattr(decoder, 'token_embedding'):
                self.tgt_embed_dim = decoder.token_embedding.embedding_dim
            else:
                self.tgt_embed_dim = decoder.d_model if hasattr(decoder, 'd_model') else 512
            
            # Create embeddings if not already in encoder/decoder
            if not hasattr(encoder, 'token_embedding'):
                self.src_embedding = nn.Embedding(src_vocab_size, self.src_embed_dim)
            else:
                self.src_embedding = None
                
            if not hasattr(decoder, 'token_embedding'):
                if share_embeddings and src_vocab_size == tgt_vocab_size:
                    self.tgt_embedding = self.src_embedding or encoder.token_embedding
                else:
                    self.tgt_embedding = nn.Embedding(tgt_vocab_size, self.tgt_embed_dim)
            else:
                self.tgt_embedding = None
                
            # Tie embeddings to output projection if requested
            if tie_embeddings and hasattr(decoder, 'output_projection'):
                if self.tgt_embedding is not None:
                    decoder.output_projection.weight = self.tgt_embedding.weight
                elif hasattr(decoder, 'token_embedding'):
                    decoder.output_projection.weight = decoder.token_embedding.weight
    
    def encode(self, src, src_mask=None):
        """Encode source sequence"""
        # Handle embeddings if needed
        if hasattr(self, 'src_embedding') and self.src_embedding is not None:
            src = self.src_embedding(src)
            
        # Encode
        encoder_output = self.encoder(src, attention_mask=src_mask)
        
        # Extract the relevant output
        if isinstance(encoder_output, dict):
            # Use sequence output for decoder attention
            encoder_states = encoder_output.get('sequence_output', encoder_output.get('tokens', None))
            if encoder_states is None:
                # Fallback to pooled output and expand
                encoder_states = encoder_output.get('pooled', list(encoder_output.values())[0])
                if len(encoder_states.shape) == 2:
                    encoder_states = encoder_states.unsqueeze(1)
        else:
            encoder_states = encoder_output
            
        return encoder_states
    
    def decode(self, tgt, encoder_output, tgt_mask=None, src_mask=None):
        """Decode target sequence given encoder output"""
        # Handle embeddings if needed
        if hasattr(self, 'tgt_embedding') and self.tgt_embedding is not None:
            tgt = self.tgt_embedding(tgt)
            
        # Decode
        decoder_output = self.decoder(
            tgt, 
            encoder_output,
            encoder_mask=src_mask
        )
        
        return decoder_output
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """Full forward pass"""
        # Encode source
        encoder_output = self.encode(src, src_mask)
        
        # Decode target
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
        
        # Extract logits if available
        if isinstance(decoder_output, dict):
            logits = decoder_output.get('logits', None)
        else:
            logits = decoder_output
            
        return {
            'logits': logits,
            'encoder_output': encoder_output,
            'decoder_output': decoder_output
        }
    
    def generate(self, src, src_mask=None, max_length=50, temperature=1.0,
                 top_k=50, top_p=0.95, num_beams=1):
        """Generate target sequence from source"""
        batch_size = src.shape[0]
        device = src.device if isinstance(src, torch.Tensor) else src[0].device
        
        # Encode source
        encoder_output = self.encode(src, src_mask)
        
        if num_beams > 1:
            return self._beam_search(encoder_output, src_mask, max_length, num_beams)
        else:
            return self._greedy_search(encoder_output, src_mask, max_length, temperature, top_k, top_p)
    
    def _greedy_search(self, encoder_output, src_mask, max_length, temperature, top_k, top_p):
        """Greedy/sampling decoding"""
        batch_size = encoder_output.shape[0]
        device = encoder_output.device
        
        # Get start token from decoder
        start_token = getattr(self.decoder, 'start_token_id', 1)
        end_token = getattr(self.decoder, 'end_token_id', 2)
        
        # Initialize with start token
        generated = torch.full((batch_size, 1), start_token, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_length - 1):
            # Decode current sequence
            decoder_output = self.decode(generated, encoder_output, src_mask=src_mask)
            
            # Get logits for next token
            if isinstance(decoder_output, dict):
                logits = decoder_output['logits'][:, -1, :]
            else:
                logits = decoder_output[:, -1, :]
                
            # Apply temperature
            logits = logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')
            
            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            # Sample or take argmax
            if temperature > 0:
                probs = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, 1)
            else:
                next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Update finished sequences
            finished = finished | (next_tokens.squeeze(-1) == end_token)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_tokens], dim=-1)
            
            # Stop if all sequences are finished
            if finished.all():
                break
                
        return generated
    
    def _beam_search(self, encoder_output, src_mask, max_length, num_beams):
        """Simplified beam search - falls back to greedy for now"""
        # For simplicity, just do greedy search with temperature 0
        # Full beam search is complex and error-prone
        return self._greedy_search(encoder_output, src_mask, max_length, 
                                   temperature=0, top_k=0, top_p=1.0)