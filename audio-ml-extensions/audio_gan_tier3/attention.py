"""
Attention module for audio processing and sequence modeling.

This module provides various attention mechanisms optimized for audio applications,
including self-attention, cross-attention, and temporal attention variants
commonly used in audio synthesis and processing models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for audio processing.
    
    Implements scaled dot-product attention with multiple heads, optimized
    for audio sequences. Supports both self-attention and cross-attention.
    
    Args:
        d_model: Model dimension (input/output feature size)
        n_heads: Number of attention heads
        d_k: Key/query dimension per head (default: d_model // n_heads)
        d_v: Value dimension per head (default: d_model // n_heads)
        dropout: Dropout rate for attention weights
        bias: Whether to use bias in linear projections
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        dropout: float = 0.1,
        bias: bool = True,
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else d_model // n_heads
        self.d_v = d_v if d_v is not None else d_model // n_heads
        self.dropout_rate = dropout
        
        assert d_model == n_heads * self.d_k, \
            f"d_model ({d_model}) must equal n_heads ({n_heads}) * d_k ({self.d_k})"
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, n_heads * self.d_k, bias=bias)
        self.w_k = nn.Linear(d_model, n_heads * self.d_k, bias=bias)
        self.w_v = nn.Linear(d_model, n_heads * self.d_v, bias=bias)
        
        # Output projection
        self.w_o = nn.Linear(n_heads * self.d_v, d_model, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor
        self.scale = math.sqrt(self.d_k)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters using Xavier initialization."""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor (batch, seq_len, d_model)
            key: Key tensor (batch, seq_len, d_model). If None, uses query (self-attention)
            value: Value tensor (batch, seq_len, d_model). If None, uses key
            mask: Attention mask (batch, seq_len, seq_len) or (batch, 1, 1, seq_len)
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor (batch, seq_len, d_model) and optionally attention weights
        """
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Handle self-attention case
        if key is None:
            key = query
        if value is None:
            value = key
        
        # Linear projections and reshape for multi-head
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        
        # Apply attention
        output, attention_weights = self._attention(Q, K, V, mask)
        
        # Concatenate heads and apply output projection
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.n_heads * self.d_v
        )
        output = self.w_o(output)
        
        if return_attention:
            return output, attention_weights
        return output
    
    def _attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scaled dot-product attention.
        
        Args:
            Q: Query (batch, n_heads, seq_len, d_k)
            K: Key (batch, n_heads, seq_len, d_k)
            V: Value (batch, n_heads, seq_len, d_v)
            mask: Attention mask
            
        Returns:
            Attention output and weights
        """
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            # Handle different mask shapes
            if mask.dim() == 3:
                # (batch, seq_len, seq_len) -> (batch, 1, seq_len, seq_len)
                mask = mask.unsqueeze(1)
            elif mask.dim() == 2:
                # (batch, seq_len) -> (batch, 1, 1, seq_len)
                mask = mask.unsqueeze(1).unsqueeze(1)
            
            # Apply mask (True = mask out, False = keep)
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(mask, float('-inf'))
            else:
                scores = scores + mask
        
        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for audio sequences.
    
    Designed specifically for processing temporal audio data with
    causal masking and relative positional encoding support.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        max_seq_len: Maximum sequence length for positional encoding
        causal: Whether to apply causal masking
        relative_pos: Whether to use relative positional encoding
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        max_seq_len: int = 5000,
        causal: bool = False,
        relative_pos: bool = True,
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.causal = causal
        self.relative_pos = relative_pos
        
        # Base attention mechanism
        self.attention = MultiHeadAttention(d_model, n_heads, **kwargs)
        
        # Relative positional encoding
        if relative_pos:
            self.rel_pos_emb = nn.Parameter(
                torch.randn(2 * max_seq_len - 1, d_model // n_heads)
            )
        
        # Causal mask (if needed)
        if causal:
            self.register_buffer(
                'causal_mask',
                torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
            )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with temporal-specific attention.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor and optionally attention weights
        """
        batch_size, seq_len, _ = x.shape
        
        # Combine masks
        combined_mask = mask
        
        # Add causal mask if needed
        if self.causal:
            causal_mask = self.causal_mask[:seq_len, :seq_len]
            if combined_mask is None:
                combined_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
            else:
                # Ensure mask dimensions match
                if combined_mask.dim() == 2:
                    combined_mask = combined_mask.unsqueeze(1).expand(-1, seq_len, -1)
                elif combined_mask.dim() == 3:
                    pass  # Already correct shape
                
                # Expand causal mask to match batch size
                causal_expanded = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
                combined_mask = combined_mask | causal_expanded
        
        # Apply attention
        return self.attention(x, mask=combined_mask, return_attention=return_attention)


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for conditioning.
    
    Allows one sequence to attend to another, useful for conditioning
    audio generation on control signals or other modalities.
    
    Args:
        d_model: Model dimension for queries
        d_context: Context dimension for keys/values
        n_heads: Number of attention heads
        dropout: Dropout rate
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        d_model: int,
        d_context: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.d_model = d_model
        self.d_context = d_context
        self.n_heads = n_heads
        
        d_k = d_model // n_heads
        
        # Projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_context, d_model)
        self.w_v = nn.Linear(d_context, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_k)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters."""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Cross-attention forward pass.
        
        Args:
            query: Query sequence (batch, query_len, d_model)
            context: Context sequence (batch, context_len, d_context)
            mask: Optional mask for context (batch, context_len)
            
        Returns:
            Output tensor (batch, query_len, d_model)
        """
        batch_size, query_len = query.size(0), query.size(1)
        context_len = context.size(1)
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, query_len, self.n_heads, -1).transpose(1, 2)
        K = self.w_k(context).view(batch_size, context_len, self.n_heads, -1).transpose(1, 2)
        V = self.w_v(context).view(batch_size, context_len, self.n_heads, -1).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                # (batch, context_len) -> (batch, 1, 1, context_len)
                mask = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Attention weights and output
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.d_model
        )
        
        return self.w_o(output)


class LocalAttention(nn.Module):
    """
    Local attention with limited receptive field.
    
    More efficient than full attention for long sequences,
    focusing attention on local neighborhoods.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        window_size: Size of attention window
        overlap: Overlap between windows
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        window_size: int = 64,
        overlap: int = 16,
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.overlap = overlap
        self.stride = window_size - overlap
        
        # Attention mechanism for windows
        self.attention = MultiHeadAttention(d_model, n_heads, **kwargs)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Local attention forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional mask (batch, seq_len)
            
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        if seq_len <= self.window_size:
            # Sequence is shorter than window, use regular attention
            return self.attention(x, mask=mask)
        
        # Pad sequence if needed
        pad_len = ((seq_len - self.window_size + self.stride - 1) // self.stride) * self.stride + self.window_size - seq_len
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            if mask is not None:
                mask = F.pad(mask, (0, pad_len), value=True)  # Mask padding
        
        # Create overlapping windows
        windows = []
        window_masks = []
        
        for start in range(0, x.size(1) - self.window_size + 1, self.stride):
            end = start + self.window_size
            windows.append(x[:, start:end])
            
            if mask is not None:
                window_masks.append(mask[:, start:end])
            else:
                window_masks.append(None)
        
        # Apply attention to each window
        attended_windows = []
        for window, window_mask in zip(windows, window_masks):
            attended_window = self.attention(window, mask=window_mask)
            attended_windows.append(attended_window)
        
        # Reconstruct sequence with overlapping
        output = torch.zeros_like(x)
        counts = torch.zeros(x.size(1), device=x.device)
        
        for i, window in enumerate(attended_windows):
            start = i * self.stride
            end = start + self.window_size
            
            output[:, start:end] += window
            counts[start:end] += 1
        
        # Average overlapping regions
        output = output / counts.view(1, -1, 1)
        
        # Remove padding
        if pad_len > 0:
            output = output[:, :seq_len]
        
        return output