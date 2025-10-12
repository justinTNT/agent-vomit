import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import deque


class MemoryBank(nn.Module):
    def __init__(self, memory_size=1000, key_dim=128, value_dim=256, 
                 similarity='cosine', update_method='fifo'):
        super().__init__()
        self.memory_size = memory_size
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.similarity = similarity
        self.update_method = update_method
        
        # Initialize memory
        self.register_buffer('keys', torch.zeros(memory_size, key_dim))
        self.register_buffer('values', torch.zeros(memory_size, value_dim))
        self.register_buffer('timestamps', torch.zeros(memory_size))
        self.register_buffer('usage_counts', torch.zeros(memory_size))
        self.register_buffer('is_valid', torch.zeros(memory_size, dtype=torch.bool))
        self.register_buffer('current_size', torch.tensor(0))
        self.register_buffer('write_position', torch.tensor(0))
        self.register_buffer('global_timestamp', torch.tensor(0))
        
        # Key and value projections
        self.key_projection = nn.Linear(key_dim, key_dim)
        self.value_projection = nn.Linear(value_dim, value_dim)
        
    def compute_similarity(self, query, keys):
        """Compute similarity between query and keys"""
        if self.similarity == 'cosine':
            query_norm = F.normalize(query, dim=-1)
            keys_norm = F.normalize(keys, dim=-1)
            if query.dim() == 2:
                # Batch query
                similarity = torch.matmul(query_norm, keys_norm.T)
            else:
                similarity = torch.matmul(keys_norm, query_norm)
        elif self.similarity == 'dot':
            if query.dim() == 2:
                similarity = torch.matmul(query, keys.T)
            else:
                similarity = torch.matmul(keys, query)
        elif self.similarity == 'l2':
            if query.dim() == 2:
                # Batch query
                query_exp = query.unsqueeze(1)  # (batch, 1, key_dim)
                keys_exp = keys.unsqueeze(0)    # (1, memory_size, key_dim)
                similarity = -torch.sum((query_exp - keys_exp) ** 2, dim=-1)
            else:
                similarity = -torch.sum((keys - query.unsqueeze(0)) ** 2, dim=-1)
        else:
            raise ValueError(f"Unknown similarity: {self.similarity}")
            
        return similarity
    
    def retrieve(self, query, k=5, return_similarities=False):
        """Retrieve k most similar memories"""
        # Project query
        query = self.key_projection(query)
        
        # Get valid memories
        valid_mask = self.is_valid[:self.memory_size]
        valid_indices = torch.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            # No valid memories
            batch_size = query.size(0) if query.dim() == 2 else 1
            empty_values = torch.zeros(batch_size, k, self.value_dim, device=query.device)
            empty_scores = torch.zeros(batch_size, k, device=query.device)
            
            if return_similarities:
                return empty_values, empty_scores
            else:
                return empty_values
        
        # Compute similarities only for valid memories
        valid_keys = self.keys[valid_indices]
        similarities = self.compute_similarity(query, valid_keys)
        
        # Get top-k
        k = min(k, len(valid_indices))
        
        if query.dim() == 2:
            # Batch query
            top_scores, top_indices = torch.topk(similarities, k, dim=-1)
            # Map back to original indices
            original_indices = valid_indices[top_indices]
            retrieved_values = self.values[original_indices]
            
            # Update usage counts
            for i in range(query.size(0)):
                self.usage_counts[original_indices[i]] += 1
        else:
            # Single query
            top_scores, top_indices = torch.topk(similarities, k)
            original_indices = valid_indices[top_indices]
            retrieved_values = self.values[original_indices]
            self.usage_counts[original_indices] += 1
        
        # Project values
        retrieved_values = self.value_projection(retrieved_values)
        
        if return_similarities:
            return retrieved_values, top_scores
        else:
            return retrieved_values
    
    def write(self, keys, values):
        """Write new memories"""
        # Project keys and values
        keys = self.key_projection(keys)
        values = self.value_projection(values)
        
        if keys.dim() == 1:
            keys = keys.unsqueeze(0)
            values = values.unsqueeze(0)
            
        batch_size = keys.size(0)
        
        # Debug shapes
        # print(f"Writing keys shape: {keys.shape}, values shape: {values.shape}")
        # print(f"Memory keys shape: {self.keys.shape}, memory values shape: {self.values.shape}")
        
        # Update global timestamp
        self.global_timestamp += 1
        
        for i in range(batch_size):
            if self.update_method == 'fifo':
                # First-in-first-out
                write_idx = int(self.write_position % self.memory_size)
            elif self.update_method == 'lru':
                # Least recently used
                if self.current_size < self.memory_size:
                    write_idx = int(self.current_size)
                else:
                    # Find least recently used
                    write_idx = int(torch.argmin(self.timestamps[:self.memory_size]))
            elif self.update_method == 'lfu':
                # Least frequently used
                if self.current_size < self.memory_size:
                    write_idx = int(self.current_size)
                else:
                    # Find least frequently used
                    write_idx = int(torch.argmin(self.usage_counts[:self.memory_size]))
            else:
                raise ValueError(f"Unknown update method: {self.update_method}")
            
            # Write to memory
            self.keys[write_idx].copy_(keys[i].detach())
            self.values[write_idx].copy_(values[i].detach())
            self.timestamps[write_idx] = self.global_timestamp
            self.usage_counts[write_idx] = 0
            self.is_valid[write_idx] = True
            
            # Update counters
            if self.update_method == 'fifo':
                self.write_position = (self.write_position + 1) % self.memory_size
            
            if self.current_size < self.memory_size:
                self.current_size += 1
    
    def clear(self):
        """Clear all memories"""
        self.is_valid.fill_(False)
        self.current_size.fill_(0)
        self.write_position.fill_(0)
        self.usage_counts.fill_(0)
        self.timestamps.fill_(0)
    
    def forward(self, query, mode='retrieve', k=5, keys=None, values=None):
        if mode == 'retrieve':
            return self.retrieve(query, k)
        elif mode == 'write':
            if keys is None or values is None:
                raise ValueError("Keys and values must be provided for write mode")
            self.write(keys, values)
            return None
        else:
            raise ValueError(f"Unknown mode: {mode}")


class Retriever(nn.Module):
    def __init__(self, encoder, memory_bank=None, memory_size=1000, 
                 key_dim=128, value_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        
        # Get encoder output dimension
        with torch.no_grad():
            # Check if encoder expects token IDs or raw features
            if hasattr(self.encoder, 'token_embedding'):
                dummy_input = torch.randint(0, 1000, (1, 10))  # Token IDs
            else:
                dummy_input = torch.randn(1, 10, 64)  # Raw features
            encoder_out = self.encoder(dummy_input)
            if isinstance(encoder_out, dict):
                encoder_out = encoder_out['pooled'] if 'pooled' in encoder_out else list(encoder_out.values())[0]
            encoder_dim = encoder_out.shape[-1]
        
        # Memory bank
        if memory_bank is None:
            self.memory_bank = MemoryBank(
                memory_size=memory_size,
                key_dim=key_dim,
                value_dim=value_dim
            )
        else:
            self.memory_bank = memory_bank
        
        # Query/key/value projections
        self.query_projection = nn.Linear(encoder_dim, key_dim)
        self.key_projection = nn.Linear(encoder_dim, key_dim)
        self.value_projection = nn.Linear(encoder_dim, value_dim)
        
        # Cross-attention for integrating retrieved memories
        self.cross_attention = nn.MultiheadAttention(
            encoder_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(encoder_dim)
        self.norm2 = nn.LayerNorm(encoder_dim)
        
        # Output projection
        self.output_projection = nn.Linear(encoder_dim + value_dim, encoder_dim)
        
    def encode_and_store(self, inputs, store_in_memory=True):
        """Encode inputs and optionally store in memory"""
        # Encode inputs
        encoded = self.encoder(inputs)
        if isinstance(encoded, dict):
            encoded = encoded['pooled'] if 'pooled' in encoded else list(encoded.values())[0]
        
        if store_in_memory:
            # Ensure batch dimension
            if encoded.dim() == 1:
                encoded = encoded.unsqueeze(0)
            elif encoded.dim() == 3:
                # If we have sequence output, pool it
                encoded = encoded.mean(dim=1)
                
            # Generate keys and values
            keys = self.key_projection(encoded)
            values = self.value_projection(encoded)
            
            # Store in memory
            self.memory_bank.write(keys, values)
        
        return encoded
    
    def retrieve_and_integrate(self, query_input, k=5):
        """Retrieve relevant memories and integrate with query"""
        # Encode query
        query_encoded = self.encoder(query_input)
        if isinstance(query_encoded, dict):
            query_encoded = query_encoded['pooled'] if 'pooled' in query_encoded else list(query_encoded.values())[0]
        
        # Handle sequence output
        if query_encoded.dim() == 3:
            query_encoded = query_encoded.mean(dim=1)
        elif query_encoded.dim() == 1:
            query_encoded = query_encoded.unsqueeze(0)
            
        # Generate query keys
        query_keys = self.query_projection(query_encoded)
        
        # Retrieve from memory
        retrieved_values, similarities = self.memory_bank.retrieve(
            query_keys, k=k, return_similarities=True
        )
        
        # Prepare for cross-attention
        if query_encoded.dim() == 2:
            query_encoded = query_encoded.unsqueeze(1)  # Add sequence dimension
        
        # Cross-attention between query and retrieved memories
        attended, _ = self.cross_attention(
            query_encoded,
            retrieved_values,
            retrieved_values
        )
        
        # Residual connection and norm
        query_encoded = self.norm1(query_encoded + attended)
        
        # Combine query with retrieved information
        if retrieved_values.dim() == 3:
            # Batch processing
            pooled_retrieved = retrieved_values.mean(dim=1)
        else:
            pooled_retrieved = retrieved_values.mean(dim=0)
            
        combined = torch.cat([query_encoded.squeeze(1), pooled_retrieved], dim=-1)
        output = self.output_projection(combined)
        output = self.norm2(output)
        
        return {
            'output': output,
            'retrieved_values': retrieved_values,
            'similarities': similarities,
            'query_encoded': query_encoded.squeeze(1)
        }
    
    def forward(self, inputs, mode='encode_store', k=5):
        if mode == 'encode_store':
            return self.encode_and_store(inputs, store_in_memory=True)
        elif mode == 'encode_only':
            return self.encode_and_store(inputs, store_in_memory=False)
        elif mode == 'retrieve':
            return self.retrieve_and_integrate(inputs, k=k)
        else:
            raise ValueError(f"Unknown mode: {mode}")