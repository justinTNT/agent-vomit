# Interface Standards for PyTorch Module Description

## Executive Summary

After analyzing various documentation and interface standards, I recommend a **hybrid approach** combining:
1. **Google-style docstrings** for human readability
2. **Comprehensive type annotations** using Python's typing module
3. **Structured metadata decorators** for machine-parseable specifications
4. **Runtime validation** using pydantic models for critical interfaces

This approach balances clarity, machine-readability, and validation capabilities while avoiding excessive verbosity.

## Comparison of Standards

### 1. Documentation Standards

| Standard | Pros | Cons | AI Agent Suitability |
|----------|------|------|---------------------|
| **NumPy Docstrings** | - Detailed sections<br>- Good for scientific docs<br>- Well-structured | - Verbose<br>- Takes excessive vertical space<br>- Redundant with type hints | Medium - Too verbose |
| **Google Docstrings** | - Concise and readable<br>- Clear sections<br>- Good tool support | - Less detailed than NumPy<br>- May need supplementation | High - Good balance |
| **reStructuredText** | - Native to Sphinx<br>- Rich formatting | - More markup required<br>- Less readable in source | Low - Too much markup |

### 2. Type Annotation Standards

| Standard | Pros | Cons | AI Agent Suitability |
|----------|------|------|---------------------|
| **PEP 484 + typing** | - Standard Python<br>- Good IDE support<br>- Runtime available | - Limited for tensor shapes<br>- No validation | High - Standard approach |
| **TorchScript Annotations** | - PyTorch native<br>- JIT compilation support | - Limited expressiveness<br>- PyTorch-specific | Medium - Too limited |
| **Annotated[T, metadata]** | - Extensible with metadata<br>- Standard compliant | - Requires custom parsers<br>- Less tool support | High - Very flexible |

### 3. Interface Description Languages

| Standard | Pros | Cons | AI Agent Suitability |
|----------|------|------|---------------------|
| **ONNX** | - Cross-framework<br>- Well-defined ops | - Export-focused<br>- Not for development | Low - Wrong use case |
| **OpenAPI/Swagger** | - Great for REST APIs<br>- Tool ecosystem | - Not for Python modules<br>- HTTP-centric | Low - Wrong domain |
| **Protocol Buffers** | - Strong typing<br>- Serialization | - Separate schema files<br>- Not Pythonic | Low - Too heavyweight |

## Recommended Approach

### 1. Core Type Annotations with Shape Information

```python
from typing import Dict, List, Optional, Tuple, Union, Literal
from typing_extensions import Annotated, TypeAlias
import torch
from torch import Tensor

# Define reusable type aliases
BatchTensor: TypeAlias = Annotated[Tensor, "shape: (B, ...)"]
SequenceTensor: TypeAlias = Annotated[Tensor, "shape: (B, L, D)"]
ImageTensor: TypeAlias = Annotated[Tensor, "shape: (B, C, H, W)"]
```

### 2. Structured Interface Decorator

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class TensorSpec:
    """Specification for tensor inputs/outputs"""
    shape: Union[Tuple[Union[int, str], ...], str]  # e.g., ("B", "L", 128) or "(B, L, D)"
    dtype: Optional[torch.dtype] = None
    constraints: Optional[Dict[str, Any]] = None  # e.g., {"B": "batch_size", "L": "<=max_seq_len"}

@dataclass
class ModuleInterface:
    """Complete interface specification for a module"""
    inputs: Dict[str, TensorSpec]
    outputs: Dict[str, TensorSpec]
    parameters: Dict[str, Any]
    constraints: Optional[List[str]] = None
    device_requirements: Optional[str] = None
    
def interface(spec: ModuleInterface):
    """Decorator to attach interface specification to a module"""
    def decorator(cls):
        cls._interface_spec = spec
        return cls
    return decorator
```

### 3. Google-style Docstrings with Enhanced Type Information

```python
class ConvEncoder(nn.Module):
    """Hierarchical CNN encoder with intermediate feature extraction.
    
    Processes input images through multiple convolutional blocks, each doubling
    the channel dimensions while halving spatial dimensions. Returns both final
    features and intermediate activations.
    
    Args:
        in_channels: Number of input image channels (e.g., 3 for RGB)
        base_channels: Starting number of channels, doubled at each layer
        num_layers: Number of convolutional blocks (>=1)
        
    Shape:
        - Input: (B, in_channels, H, W) where H, W >= 2^num_layers
        - Output: Dict with:
            - 'features': (B, base_channels * 2^(num_layers-1), H/2^num_layers, W/2^num_layers)
            - 'intermediate': List of (B, C_i, H_i, W_i) for each layer
            - 'pooled': (B, base_channels * 2^(num_layers-1))
            
    Examples:
        >>> encoder = ConvEncoder(in_channels=3, base_channels=64, num_layers=4)
        >>> x = torch.randn(2, 3, 256, 256)
        >>> out = encoder(x)
        >>> out['features'].shape
        torch.Size([2, 512, 16, 16])
    """
```

### 4. Runtime Validation with Pydantic (for critical modules)

```python
from pydantic import BaseModel, validator
import torch

class ConvEncoderConfig(BaseModel):
    in_channels: int = 3
    base_channels: int = 64
    num_layers: int = 4
    
    @validator('num_layers')
    def validate_num_layers(cls, v):
        if v < 1:
            raise ValueError("num_layers must be at least 1")
        return v
    
    @validator('base_channels')
    def validate_base_channels(cls, v):
        if v <= 0 or (v & (v - 1)) != 0:  # Check if power of 2
            raise ValueError("base_channels should be a positive power of 2")
        return v
```

### 5. Comprehensive Example: Recommended Interface Standard

```python
from typing import Dict, List, Optional, Tuple
from typing_extensions import Annotated
import torch
import torch.nn as nn
from pydantic import BaseModel, validator

# 1. Configuration with validation
class TransformerBlockConfig(BaseModel):
    """Configuration for TransformerBlock with validation."""
    hidden_size: int = 512
    num_heads: int = 8
    dropout: float = 0.1
    use_bias: bool = True
    layer_norm_eps: float = 1e-5
    
    @validator('hidden_size')
    def validate_hidden_size(cls, v, values):
        if 'num_heads' in values and v % values['num_heads'] != 0:
            raise ValueError(f"hidden_size ({v}) must be divisible by num_heads ({values['num_heads']})")
        return v

# 2. Module with comprehensive interface
@interface(ModuleInterface(
    inputs={
        'x': TensorSpec(shape=("B", "L", "D"), dtype=torch.float32),
        'mask': TensorSpec(shape=("B", "L"), dtype=torch.bool, constraints={"optional": True})
    },
    outputs={
        'output': TensorSpec(shape=("B", "L", "D"), dtype=torch.float32),
        'attention_weights': TensorSpec(shape=("B", "H", "L", "L"), dtype=torch.float32)
    },
    parameters={
        'hidden_size': 'int > 0, divisible by num_heads',
        'num_heads': 'int > 0',
        'dropout': 'float in [0, 1]'
    },
    constraints=[
        "D == hidden_size",
        "H == num_heads"
    ]
))
class TransformerBlock(nn.Module):
    """Self-attention transformer block with residual connections.
    
    Implements a standard transformer encoder block with multi-head self-attention,
    layer normalization, and feedforward network. Supports both causal and
    bidirectional attention patterns.
    
    Args:
        config: TransformerBlockConfig instance or dict with configuration
        
    Inputs:
        x: Input tensor of shape (batch_size, seq_len, hidden_size)
        mask: Optional attention mask of shape (batch_size, seq_len).
              True values are masked (ignored).
              
    Returns:
        Dict containing:
            - output: Transformed tensor, same shape as input
            - attention_weights: Attention weights from all heads
              
    Examples:
        >>> config = TransformerBlockConfig(hidden_size=512, num_heads=8)
        >>> block = TransformerBlock(config)
        >>> x = torch.randn(2, 100, 512)
        >>> out = block(x)
        >>> out['output'].shape
        torch.Size([2, 100, 512])
        
    Note:
        The attention mask uses the convention where True = masked (ignored),
        matching PyTorch's scaled_dot_product_attention expectations.
    """
    
    def __init__(self, config: Union[TransformerBlockConfig, Dict]):
        super().__init__()
        if isinstance(config, dict):
            config = TransformerBlockConfig(**config)
        self.config = config
        
        # Type annotations for clarity
        self.attention: nn.MultiheadAttention
        self.norm1: nn.LayerNorm
        self.norm2: nn.LayerNorm
        self.ffn: nn.Sequential
        
        # Initialize layers
        self._build_layers()
    
    def forward(
        self, 
        x: Annotated[torch.Tensor, "shape: (B, L, D)"],
        mask: Optional[Annotated[torch.Tensor, "shape: (B, L)"]] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with type-annotated inputs."""
        # Implementation here
        pass
```

## Benefits of This Approach

1. **Clear for AI Agents**: Structured information in multiple forms
2. **Programmatically Validatable**: Pydantic models + decorators enable validation
3. **Comprehensive**: Covers types, shapes, constraints, and examples
4. **Not Overly Verbose**: Google docstrings are concise, metadata is optional
5. **Tool-Friendly**: Works with existing Python tooling
6. **Gradual Adoption**: Can start with just type hints and add more as needed

## Implementation Priority

1. **Level 1 (Minimum)**: Type annotations + Google docstrings with shape info
2. **Level 2 (Recommended)**: Add Pydantic configs for complex modules  
3. **Level 3 (Advanced)**: Add interface decorators for machine parsing
4. **Level 4 (Optional)**: Runtime shape validation for critical paths

This approach provides the best balance for AI agent consumption while maintaining human readability and leveraging existing Python standards.