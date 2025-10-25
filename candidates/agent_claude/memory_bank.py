import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from typing import Optional, Tuple


class MemoryBank(nn.Module):
    def __init__(self,
                 memory_size: int = 1024,           # Number of memory slots
                 memory_dim: int = 256,             # Dimension of each memory slot
                 n_heads: int = 8,                  # Number of attention heads
                 temperature: float = 1.0,          # Temperature for retrieval
                 similarity_fn: str = 'cosine',     # 'cosine' or 'dot'
                 update_method: str = 'gated',      # 'gated', 'ema', or 'replace'
                 ema_decay: float = 0.999,          # EMA decay rate
                 **kwargs):                         # REQUIRED: Accept extra kwargs
        super().__init__()
        
        # Log unknown parameters
        if kwargs:
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.n_heads = n_heads
        self.temperature = temperature
        self.similarity_fn = similarity_fn
        self.update_method = update_method
        self.ema_decay = ema_decay
        
        # Initialize memory bank
        self.register_buffer('memory', torch.randn(memory_size, memory_dim))
        self.memory = F.normalize(self.memory, dim=1)
        
        # Memory usage tracking
        self.register_buffer('memory_usage', torch.zeros(memory_size))
        self.register_buffer('memory_age', torch.zeros(memory_size))
        
        # Query projection for retrieval
        self.query_projection = nn.Linear(memory_dim, memory_dim)
        
        # Key-value projections for memory
        self.key_projection = nn.Linear(memory_dim, memory_dim)
        self.value_projection = nn.Linear(memory_dim, memory_dim)
        
        # Attention mechanism for retrieval
        self.attention = nn.MultiheadAttention(
            embed_dim=memory_dim,
            num_heads=n_heads,
            batch_first=True
        )
        
        # Gated update mechanism
        if update_method == 'gated':
            self.update_gate = nn.Sequential(
                nn.Linear(memory_dim * 2, memory_dim),
                nn.ReLU(),
                nn.Linear(memory_dim, 1),
                nn.Sigmoid()
            )
        
        # Output projection
        self.output_projection = nn.Linear(memory_dim, memory_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(memory_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, 
                query: torch.Tensor,
                write: bool = False,
                write_data: Optional[torch.Tensor] = None) -> dict:
        """
        Forward pass for memory bank operations.
        
        Args:
            query: Query tensor (batch, query_dim)
            write: Whether to write to memory
            write_data: Data to write (batch, memory_dim), defaults to query
        
        Returns:
            Dictionary with keys:
                - 'retrieved': Retrieved memory content (batch, memory_dim)
                - 'attention_weights': Attention weights over memory (batch, memory_size)
                - 'top_k_indices': Indices of top-k retrieved memories
                - 'write_indices': Indices where data was written (if write=True)
        """
        batch_size = query.size(0)
        device = query.device
        
        # Project query
        query_proj = self.query_projection(query)
        
        # Retrieve from memory
        retrieved, attention_weights, top_k_indices = self._retrieve(query_proj)
        
        # Write to memory if requested
        write_indices = None
        if write:
            if write_data is None:
                write_data = query
            write_indices = self._write(write_data)
        
        # Update memory age
        self.memory_age += 1
        
        # Apply output projection and normalization
        retrieved = self.output_projection(retrieved)
        retrieved = self.layer_norm(retrieved)
        
        return {
            'retrieved': retrieved,
            'attention_weights': attention_weights,
            'top_k_indices': top_k_indices,
            'write_indices': write_indices
        }
    
    def _retrieve(self, query: torch.Tensor, k: int = 10) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve from memory using attention mechanism."""
        batch_size = query.size(0)
        
        # Compute similarities
        if self.similarity_fn == 'cosine':
            # Normalize query and memory
            query_norm = F.normalize(query, dim=-1)
            memory_norm = F.normalize(self.memory, dim=-1)
            similarities = torch.matmul(query_norm, memory_norm.t())
        else:  # dot product
            similarities = torch.matmul(query, self.memory.t())
        
        # Apply temperature scaling
        similarities = similarities / self.temperature
        
        # Get top-k memories
        top_k_values, top_k_indices = torch.topk(similarities, k=min(k, self.memory_size), dim=-1)
        
        # Soft attention weights
        attention_weights = F.softmax(similarities, dim=-1)
        
        # Retrieve using attention
        # Use multi-head attention for richer retrieval
        query_expanded = query.unsqueeze(1)  # (batch, 1, dim)
        memory_expanded = self.memory.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, memory_size, dim)
        
        # Apply key-value projections
        keys = self.key_projection(memory_expanded)
        values = self.value_projection(memory_expanded)
        
        # Attention-based retrieval
        attended, _ = self.attention(query_expanded, keys, values)
        retrieved = attended.squeeze(1)  # (batch, dim)
        
        # Alternative: weighted sum based on attention weights
        # retrieved = torch.matmul(attention_weights, self.memory)
        
        return retrieved, attention_weights, top_k_indices
    
    def _write(self, data: torch.Tensor) -> torch.Tensor:
        """Write data to memory bank."""
        batch_size = data.size(0)
        device = data.device
        
        if self.update_method == 'replace':
            # Find least recently used slots
            _, write_indices = torch.topk(-self.memory_age, k=batch_size)
            
            # Replace memory slots
            for i, idx in enumerate(write_indices):
                idx = int(idx.item())
                self.memory[idx].copy_(data[i].detach())
                self.memory_usage[idx] = 0
                self.memory_age[idx] = 0
        
        elif self.update_method == 'ema':
            # Find most similar memories for EMA update
            data_norm = F.normalize(data, dim=-1)
            memory_norm = F.normalize(self.memory, dim=-1)
            similarities = torch.matmul(data_norm, memory_norm.t())
            
            # Get most similar slots
            _, write_indices = torch.max(similarities, dim=-1)
            
            # EMA update
            for i, idx in enumerate(write_indices):
                idx = int(idx.item())
                self.memory[idx] = self.ema_decay * self.memory[idx] + \
                                  (1 - self.ema_decay) * data[i].detach()
                self.memory[idx] = F.normalize(self.memory[idx], dim=0)
                self.memory_usage[idx] += 1
        
        elif self.update_method == 'gated':
            # Find candidate slots (least used + oldest)
            score = -self.memory_usage - 0.1 * self.memory_age
            _, candidate_indices = torch.topk(score, k=min(batch_size * 2, self.memory_size))
            
            write_indices = []
            for i in range(batch_size):
                # Compare with candidates
                candidates = self.memory[candidate_indices]
                data_expanded = data[i:i+1].expand(candidates.size(0), -1)
                
                # Compute gate values
                concat = torch.cat([candidates, data_expanded], dim=-1)
                gates = self.update_gate(concat).squeeze(-1)
                
                # Select slot with highest gate value
                selected_idx = candidate_indices[torch.argmax(gates)]
                write_indices.append(selected_idx)
                
                # Gated update
                gate = gates[torch.argmax(gates)]
                idx = int(selected_idx.item())
                self.memory[idx] = gate * data[i].detach() + (1 - gate) * self.memory[idx]
                self.memory[idx] = F.normalize(self.memory[idx], dim=0)
                self.memory_usage[idx] += 1
                self.memory_age[idx] = 0
            
            write_indices = torch.tensor(write_indices, device=device)
        
        return write_indices
    
    def clear_memory(self):
        """Clear the memory bank."""
        self.memory.normal_()
        self.memory = F.normalize(self.memory, dim=1)
        self.memory_usage.zero_()
        self.memory_age.zero_()
    
    def get_memory_stats(self) -> dict:
        """Get memory bank statistics."""
        return {
            'total_slots': self.memory_size,
            'used_slots': (self.memory_usage > 0).sum().item(),
            'avg_usage': self.memory_usage.mean().item(),
            'max_usage': self.memory_usage.max().item(),
            'avg_age': self.memory_age.mean().item(),
            'max_age': self.memory_age.max().item()
        }