# Guidelines v2: Explicit Implementation Decisions

## Design Philosophy
Every implementation decision is explicitly documented here. If you need to make a choice, look here first. If it's not here, use your judgment and follow the patterns.

## Module Implementation Specifications

### 1. TransformerBlock
```python
# PARAMETER NAMES (required exactly as shown)
{
    'd_model': int,          # hidden dimension
    'n_heads': int,          # number of attention heads  
    'd_ff': int,             # feedforward dimension
    'dropout': float = 0.1,  # dropout rate
    'activation': str = 'gelu',  # activation function
    'norm_first': bool = True,   # pre-norm vs post-norm
}

# RETURN FORMAT
- Returns: torch.Tensor (same shape as input)
- NOT dict, NOT tuple

# CLASS STRUCTURE  
- Main class: TransformerBlock
- No additional exports required

# FORWARD SIGNATURE
def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    # mask: Optional attention mask (additive, not boolean)
```

### 2. ConvEncoder
```python
# PARAMETER NAMES
{
    'in_channels': int = 3,
    'base_channels': int = 64,  
    'num_layers': int = 4,
    'latent_dim': int = 512,
}

# RETURN FORMAT  
- Returns: Dict[str, Tensor]
  {
      'features': Tensor,  # feature maps before pooling
      'pooled': Tensor,    # global pooled features
      'latent': Tensor,    # final latent representation
  }

# FORWARD SIGNATURE
def forward(self, x: Tensor) -> Dict[str, Tensor]:
    pass
```

### 3. DataVersioner
```python
# PARAMETER NAMES
{
    'storage_path': str = './versions',
    'deduplicate': bool = True,  # content-based dedup
    'compression': bool = False,
}

# CALLABLE INTERFACE
- MUST be callable: version_id = versioner(data, message)
- NOT versioner.create_version()

# REQUIRED METHODS
def __call__(self, data: Tensor, message: str) -> str:
    """Create version, return version_id"""
    
def load(self, version_id: str) -> Tensor:
    """Load version by id"""
    
def list_versions(self) -> List[Dict]:
    """List all versions with metadata"""
```

### 4. StreamJoiner  
```python
# PARAMETER NAMES
{
    'join_type': str = 'inner',  # 'inner', 'left', 'outer'  
    'time_window': float = 1.0,
    'buffer_size': int = 1000,   # REQUIRED parameter
    'primary_stream': Optional[str] = None,
}

# FORWARD SIGNATURE  
# NOTE: timestamp is positional, not kwarg
def forward(self, stream_id: str, data: Tensor, timestamp: float) -> Optional[Dict[str, Tensor]]:
    pass
```

### 5. SnakeActivation
```python  
# REQUIRED EXPORTS
__all__ = ['SnakeActivation', 'SnakeBeta', 'snake']

# PARAMETER NAMES
class SnakeActivation:
    def __init__(self, channels: int, alpha: float = 1.0):
        pass
        
class SnakeBeta:  # Advanced version
    def __init__(self, channels: int, alpha: float = 1.0, beta: float = 1.0):
        pass

# Functional interface
def snake(x: Tensor, alpha: float = 1.0) -> Tensor:
    pass
```

## Global Conventions

### Parameter Naming Dictionary
When you see these concepts, use these EXACT parameter names:
```python
PARAM_NAMES = {
    # Dimensions
    'hidden_dimension': 'd_model',
    'attention_heads': 'n_heads', 
    'feedforward_dim': 'd_ff',
    'input_channels': 'in_channels',
    'output_channels': 'out_channels',
    
    # Images
    'image_size': 'img_size',
    'patch_size': 'patch_size',
    
    # Sequences  
    'sequence_length': 'seq_len',
    'context_length': 'max_len',
    
    # Training
    'batch_size': 'batch_size',  # never 'bs'
    'learning_rate': 'lr',       # abbreviated ok here
    'dropout_rate': 'dropout',
}
```

### Return Format Patterns
```python
RETURN_FORMATS = {
    'encoder': "Dict[str, Tensor] with keys: 'features', 'pooled', 'latent'",
    'decoder': "Tensor (reconstructed output)",
    'transformer': "Tensor (same shape as input)",
    'loss_module': "Dict[str, Tensor] with keys: 'loss', 'components'",
    'data_module': "Depends on operation (see specific module)",
}
```

### Method Naming Conventions  
```python
METHOD_NAMES = {
    'save_data': 'save',      # not store, persist
    'load_data': 'load',      # not get, retrieve  
    'create_item': 'create',  # not make, new
    'remove_item': 'remove',  # not delete, del
    'get_info': 'info',       # not describe, summary
}
```

## Decision Points for New Modules

When implementing a new module, fill out:

```yaml
module_name:
  parameters:
    param_name: type = default  # description
  
  return_type: Tensor | Dict[str, Tensor] | Other
  
  return_format: 
    key1: description
    key2: description
    
  callable: true | false
  
  required_methods:
    - method_name(signature)
    
  required_exports:
    - ClassName
    - function_name
    
  forward_signature: |
    def forward(self, ...):
        pass
```

## Design Principles (When Not Specified Above)

1. **When in doubt, parameterize** - Add parameter rather than hardcode
2. **Follow PyTorch conventions** - nn.Module, forward method, etc.
3. **Type everything** - Use proper type hints
4. **Docstring everything** - At least one line describing purpose

## What Tests Will Validate

Tests will check:
1. Module can be initialized with specified parameters
2. Forward pass produces correct format/shape  
3. Required methods exist and work
4. Module handles edge cases gracefully

Tests will NOT check:
1. Exact internal implementation
2. Specific numeric values (unless critical)
3. Performance characteristics
4. Internal variable names