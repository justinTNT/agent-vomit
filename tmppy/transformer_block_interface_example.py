"""
Example of the recommended interface standard applied to TransformerBlock.

This demonstrates:
1. Comprehensive type annotations with shape information
2. Google-style docstrings with detailed shape specifications
3. Pydantic configuration for validation
4. Optional interface decorator for machine parsing
5. Runtime shape validation helpers
"""

from typing import Dict, List, Optional, Tuple, Union, Literal
from typing_extensions import Annotated, TypeAlias
from dataclasses import dataclass
from pydantic import BaseModel, validator, Field
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# PART 1: Type Aliases and Specifications
# ============================================================================

# Reusable type aliases for common tensor patterns
BatchTensor: TypeAlias = Annotated[torch.Tensor, "shape: (B, ...)"]
SequenceTensor: TypeAlias = Annotated[torch.Tensor, "shape: (B, L, D)"]
AttentionMask: TypeAlias = Annotated[torch.Tensor, "shape: (B, L) or (B, 1, L, L)"]
AttentionWeights: TypeAlias = Annotated[torch.Tensor, "shape: (B, H, L, L)"]


@dataclass
class TensorSpec:
    """Specification for tensor inputs/outputs."""
    shape: Union[Tuple[Union[int, str], ...], str]
    dtype: Optional[torch.dtype] = None
    constraints: Optional[Dict[str, str]] = None
    optional: bool = False
    description: Optional[str] = None


@dataclass
class ModuleInterface:
    """Complete interface specification for a module."""
    inputs: Dict[str, TensorSpec]
    outputs: Dict[str, TensorSpec]
    parameters: Dict[str, str]
    constraints: List[str]
    device_requirements: Optional[str] = None
    memory_estimate: Optional[str] = None


def interface(spec: ModuleInterface):
    """Decorator to attach interface specification to a module."""
    def decorator(cls):
        cls._interface_spec = spec
        return cls
    return decorator


# ============================================================================
# PART 2: Configuration with Validation
# ============================================================================

class TransformerBlockConfig(BaseModel):
    """Configuration for TransformerBlock with built-in validation.
    
    This configuration ensures all parameters are valid before module creation,
    preventing common initialization errors.
    """
    d_model: int = Field(512, description="Model dimension (hidden size)")
    n_heads: int = Field(8, description="Number of attention heads")
    d_ff: int = Field(2048, description="Feed-forward network dimension")
    dropout: float = Field(0.1, ge=0.0, le=1.0, description="Dropout probability")
    layer_norm_eps: float = Field(1e-5, description="Layer normalization epsilon")
    use_bias: bool = Field(True, description="Whether to use bias in linear layers")
    attention_type: Literal["self", "cross", "causal"] = Field("self", description="Type of attention")
    
    @validator('d_model')
    def validate_d_model(cls, v, values):
        if 'n_heads' in values and v % values['n_heads'] != 0:
            raise ValueError(
                f"d_model ({v}) must be divisible by n_heads ({values['n_heads']})"
            )
        if v <= 0:
            raise ValueError("d_model must be positive")
        return v
    
    @validator('n_heads')
    def validate_n_heads(cls, v):
        if v <= 0:
            raise ValueError("n_heads must be positive")
        if v > 32:  # Practical limit
            raise ValueError("n_heads > 32 is impractical and likely an error")
        return v
    
    @validator('d_ff')
    def validate_d_ff(cls, v, values):
        if v <= 0:
            raise ValueError("d_ff must be positive")
        if 'd_model' in values and v < values['d_model']:
            print(f"Warning: d_ff ({v}) is smaller than d_model ({values['d_model']})")
        return v


# ============================================================================
# PART 3: Enhanced Module Implementation
# ============================================================================

@interface(ModuleInterface(
    inputs={
        'x': TensorSpec(
            shape=("B", "L", "D"),
            dtype=torch.float32,
            description="Input sequence tensor"
        ),
        'mask': TensorSpec(
            shape=("B", "L"),
            dtype=torch.bool,
            optional=True,
            description="Attention mask (True = masked/ignored)"
        )
    },
    outputs={
        'output': TensorSpec(
            shape=("B", "L", "D"),
            dtype=torch.float32,
            description="Transformed sequence"
        ),
        'attention_weights': TensorSpec(
            shape=("B", "H", "L", "L"),
            dtype=torch.float32,
            optional=True,
            description="Attention weights from all heads"
        )
    },
    parameters={
        'd_model': 'int > 0, divisible by n_heads',
        'n_heads': 'int in [1, 32]',
        'd_ff': 'int > 0, typically 4*d_model',
        'dropout': 'float in [0, 1]'
    },
    constraints=[
        "D == d_model",
        "H == n_heads",
        "Input and output shapes are identical",
        "Mask shape must match sequence length"
    ],
    memory_estimate="O(B * L^2 * D) for self-attention computation"
))
class TransformerBlock(nn.Module):
    """Self-attention transformer block with residual connections.
    
    Implements a standard transformer encoder block with multi-head self-attention,
    layer normalization, and feedforward network. This implementation follows the
    "Pre-LN" variant where layer normalization is applied before each sub-layer.
    
    The module consists of:
    1. Multi-head self-attention with residual connection
    2. Layer normalization
    3. Position-wise feedforward network with residual connection
    4. Layer normalization
    
    Args:
        config: TransformerBlockConfig instance or dict with configuration.
                Can also pass individual parameters for backward compatibility.
        d_model: Model dimension (if not using config)
        n_heads: Number of attention heads (if not using config)
        d_ff: Feedforward dimension (if not using config)
        dropout: Dropout probability (if not using config)
        
    Shape:
        - Input[x]: (batch_size, seq_len, d_model)
        - Input[mask]: (batch_size, seq_len) optional
        - Output: (batch_size, seq_len, d_model)
        
    Examples:
        Using configuration object:
        >>> config = TransformerBlockConfig(d_model=512, n_heads=8)
        >>> block = TransformerBlock(config)
        >>> x = torch.randn(2, 100, 512)
        >>> output = block(x)
        >>> output.shape
        torch.Size([2, 100, 512])
        
        Using individual parameters:
        >>> block = TransformerBlock(d_model=256, n_heads=4)
        >>> x = torch.randn(2, 50, 256)
        >>> mask = torch.ones(2, 50).bool()  # No masking
        >>> output = block(x, mask=mask)
        
        With attention mask:
        >>> mask = torch.zeros(2, 100).bool()
        >>> mask[:, 50:] = True  # Mask last 50 positions
        >>> output = block(x, mask=mask)
    
    Note:
        - The attention mask follows PyTorch convention: True = masked (ignored)
        - For causal attention, use `attention_type='causal'` in config
        - Memory usage scales quadratically with sequence length due to attention
    """
    
    def __init__(
        self,
        config: Optional[Union[TransformerBlockConfig, Dict]] = None,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        # Handle configuration
        if config is None:
            config = TransformerBlockConfig(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                **kwargs
            )
        elif isinstance(config, dict):
            config = TransformerBlockConfig(**config)
        
        self.config = config
        
        # Store key dimensions for easy access
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads
        
        # Multi-head attention components
        self.self_attention = MultiHeadAttention(
            config.d_model,
            config.n_heads,
            config.dropout,
            config.use_bias
        )
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # Feed-forward components
        self.feed_forward = FeedForwardNetwork(
            config.d_model,
            config.d_ff,
            config.dropout,
            config.use_bias
        )
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(config.dropout)
        
        # Optional: Add input validation
        self._validate_on_forward = True
        
    def forward(
        self,
        x: SequenceTensor,
        mask: Optional[AttentionMask] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through the transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask. Shape (batch_size, seq_len) where
                  True indicates positions to mask (ignore).
            return_attention: If True, return attention weights along with output
            
        Returns:
            If return_attention is False:
                Tensor of shape (batch_size, seq_len, d_model)
            If return_attention is True:
                Dict with keys:
                - 'output': Tensor of shape (batch_size, seq_len, d_model)
                - 'attention_weights': Tensor of shape (batch_size, n_heads, seq_len, seq_len)
        """
        # Optional runtime validation
        if self._validate_on_forward:
            self._validate_input(x, mask)
        
        # Self-attention with residual connection
        if return_attention:
            attn_output, attn_weights = self.self_attention(
                x, mask=mask, return_attention=True
            )
        else:
            attn_output = self.self_attention(x, mask=mask)
            
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        output = self.norm2(x + self.dropout(ff_output))
        
        if return_attention:
            return {
                'output': output,
                'attention_weights': attn_weights
            }
        return output
    
    def _validate_input(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Validate input tensors at runtime."""
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input tensor, got {x.dim()}D")
        
        if x.size(-1) != self.d_model:
            raise ValueError(
                f"Input dimension {x.size(-1)} doesn't match "
                f"d_model {self.d_model}"
            )
        
        if mask is not None:
            if mask.dim() not in [2, 4]:
                raise ValueError(
                    f"Mask must be 2D (B, L) or 4D (B, 1, L, L), "
                    f"got {mask.dim()}D"
                )
            if mask.size(0) != x.size(0):
                raise ValueError("Batch size mismatch between input and mask")
            if mask.dim() == 2 and mask.size(1) != x.size(1):
                raise ValueError("Sequence length mismatch between input and mask")
    
    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"d_ff={self.config.d_ff}, dropout={self.config.dropout}"
        )


# ============================================================================
# PART 4: Supporting Classes (abbreviated for example)
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
        use_bias: Whether to use bias in projections
        
    Shape:
        - Query: (batch_size, seq_len_q, d_model)
        - Key: (batch_size, seq_len_k, d_model)
        - Value: (batch_size, seq_len_k, d_model)
        - Output: (batch_size, seq_len_q, d_model)
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, use_bias: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=use_bias)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for multi-head attention."""
        if key is None:
            key = query
        if value is None:
            value = query
            
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        
        # Project and reshape
        Q = self.q_proj(query).view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(1), float('-inf'))
            else:
                scores = scores + mask
                
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        
        output = self.out_proj(attn_output)
        
        if return_attention:
            return output, attn_weights
        return output


class FeedForwardNetwork(nn.Module):
    """Position-wise feedforward network.
    
    Args:
        d_model: Model dimension
        d_ff: Hidden dimension of feedforward network
        dropout: Dropout probability
        use_bias: Whether to use bias in linear layers
        
    Shape:
        - Input: (batch_size, seq_len, d_model)
        - Output: (batch_size, seq_len, d_model)
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, use_bias: bool = True):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.linear2 = nn.Linear(d_ff, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feedforward network."""
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


# ============================================================================
# PART 5: Usage Examples and Testing
# ============================================================================

if __name__ == "__main__":
    # Example 1: Basic usage with config
    config = TransformerBlockConfig(
        d_model=256,
        n_heads=8,
        d_ff=1024,
        dropout=0.1
    )
    model = TransformerBlock(config)
    
    # Example 2: Validate configuration
    try:
        bad_config = TransformerBlockConfig(
            d_model=256,
            n_heads=7,  # Not a divisor of 256
        )
    except ValueError as e:
        print(f"Config validation caught error: {e}")
    
    # Example 3: Runtime shape validation
    x = torch.randn(2, 50, 256)
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Example 4: With attention weights
    result = model(x, return_attention=True)
    print(f"Output shape: {result['output'].shape}")
    print(f"Attention shape: {result['attention_weights'].shape}")
    
    # Example 5: Access interface specification
    if hasattr(model, '_interface_spec'):
        spec = model._interface_spec
        print(f"\nInterface inputs: {list(spec.inputs.keys())}")
        print(f"Interface outputs: {list(spec.outputs.keys())}")
        print(f"Constraints: {spec.constraints}")