# Agent Guidelines for Reliable Module Generation

Looking at the patterns of fixes needed across both ML and data pipeline modules, here are the key guidelines that would prevent most issues:

## Core Guidelines

### 1. **Tensor Shape Consistency**
```python
# BAD: Assuming shapes without checking
pos_embed = self.pos_embedding[:, :seq_len]  # May fail

# GOOD: Explicitly handle shape requirements
seq_len = patches.size(1)
if seq_len > self.pos_embedding.size(1):
    # Handle case or raise clear error
```

### 2. **Mask Format Standardization**
```python
# BAD: Assuming mask type
attn_output = attention(query, mask=mask)

# GOOD: Handle both boolean and additive masks
if mask is not None and mask.dtype == torch.bool:
    mask = mask.float().masked_fill(mask == 0, float('-inf'))
```

### 3. **Device Management**
```python
# BAD: Creating tensors without device context
indices = torch.arange(n)

# GOOD: Always maintain device consistency
indices = torch.arange(n, device=input_tensor.device)
```

### 4. **Type Conversions for Indexing**
```python
# BAD: Using tensors as indices directly
idx = torch.tensor(0)
array[idx] = value  # May fail

# GOOD: Explicit conversion
idx = int(torch.tensor(0).item())
array[idx] = value
```

### 5. **API Parameter Naming**
```python
# BAD: Assuming parameter names
output = model(x, mask=mask)

# GOOD: Check expected parameter names
# If unsure, check: inspect.signature(model.forward)
output = model(x, attention_mask=mask)  # or src_key_padding_mask, etc.
```

### 6. **State Management Patterns**
```python
# BAD: Direct tensor assignment in stateful modules
self.memory[idx] = new_value

# GOOD: Use copy_() or proper assignment
self.memory[idx].copy_(new_value.detach())
```

### 7. **Collection Type Handling**
```python
# BAD: Assuming deque behaves like list
def process(buffer: deque):
    sorted_buffer = sorted(buffer)  # Works
    interpolate(buffer, time)  # May fail if function expects list

# GOOD: Explicit conversion when needed
def process(buffer: deque):
    interpolate(list(buffer), time)
```

### 8. **Dimension Calculations**
```python
# BAD: Hard-coding dimension relationships
freq_dim = embed_dim // 2  # Assumes specific relationship

# GOOD: Make relationships explicit or parameterizable
freq_dim = embed_dim // (2 * num_dims)  # Clear relationship
```

### 9. **Equality vs Closeness**
```python
# BAD: Using torch.equal for continuous values
changed = ~torch.equal(data1, data2)  # Returns bool, not tensor

# GOOD: Use element-wise comparison
changed = ~torch.isclose(data1, data2)  # Returns tensor of bools
```

### 10. **Module Recreation**
```python
# BAD: Trying to recreate modules dynamically
momentum_encoder = type(encoder)()  # Loses initialized parameters

# GOOD: Use deep copy
momentum_encoder = copy.deepcopy(encoder)
```

## Module-Specific Implementation Standards

### TransformerBlock
**Standard Parameters:**
```python
class TransformerBlock(nn.Module):
    def __init__(self, 
                 d_model: int,          # Hidden dimension (NOT hidden_dim)
                 n_heads: int,          # Number of attention heads (NOT num_heads)
                 d_ff: int = None,      # Feed-forward dimension (defaults to 4*d_model)
                 dropout: float = 0.1,  # Dropout rate (NOT dropout_rate)
                 **kwargs):             # Accept extra kwargs for flexibility
```
**Required behavior:**
- Must preserve input shape (residual architecture)
- Must handle mask=None gracefully
- Return tensor, not dict

### ConvEncoder
**Standard Parameters:**
```python
class ConvEncoder(nn.Module):
    def __init__(self,
                 in_channels: int = 3,      # Input channels (NOT input_channels)
                 base_channels: int = 64,   # Base channel count (NOT base_filters)
                 num_layers: int = 4,       # Number of layers (NOT n_blocks)
                 **kwargs):
```
**Required behavior:**
- Return dictionary with at least 'features' key
- Should downsample spatial dimensions

### DataVersioner
**Standard Interface:**
```python
class DataVersioner:
    def __init__(self, 
                 storage_path: str = "./versions",  # Where to store versions
                 hash_fn: str = "md5",             # Hash function to use
                 **kwargs):
    
    def __call__(self, data):
        """Make the class callable - returns version info"""
        return self.version(data)
    
    def version(self, data):
        """Core versioning method"""
        # Return dict with at least: {'version': str, 'hash': str}
    
    def load(self, version_id):
        """Load a specific version"""
        # Required method for loading versions
```
**Required behavior:**
- Must be callable (implement `__call__`)
- Must have `load` method
- `version()` returns dict with 'version' and 'hash' keys minimum

### StreamJoiner
**Standard Parameters:**
```python
class StreamJoiner:
    def __init__(self,
                 join_type: str = 'inner',    # Type of join (REQUIRED)
                 window_size: int = 1000,     # Buffer window size
                 timeout: float = None,       # Optional timeout
                 **kwargs):
```
**Required behavior:**
- Must support join_type parameter (even if only 'inner' is implemented)
- Handle multiple streams with add/join interface

### SnakeActivation
**Standard Parameters:**
```python
class SnakeActivation(nn.Module):
    def __init__(self,
                 n_channels: int,              # Number of channels (NOT channels)
                 alpha: float = 1.0,           # Frequency parameter
                 learnable: bool = True,       # Whether alpha is learnable
                 **kwargs):
```
**Required behavior:**
- Handle both 2D (batch, channels) and 3D (batch, channels, time) inputs
- Implement snake formula: x + (1/alpha) * sin²(alpha * x)

## General Module Requirements

### 1. **Parameter Flexibility**
All modules should accept `**kwargs` in `__init__` to handle parameter variations gracefully:
```python
def __init__(self, required_param, optional_param=None, **kwargs):
    super().__init__()
    # Log unknown parameters rather than failing
    if kwargs:
        print(f"Ignoring unknown parameters: {list(kwargs.keys())}")
```

### 2. **Return Format Consistency**
- **Neural network layers** (TransformerBlock, etc.): Return tensor
- **Feature extractors** (ConvEncoder, etc.): Return dict with descriptive keys
- **Data processors** (DataVersioner, etc.): Return dict with metadata

### 3. **Method Naming Standards**
Common method names to implement:
- `forward()` for nn.Module subclasses
- `process()` or `__call__()` for data processors
- `load()` and `save()` for stateful modules
- `reset()` for modules with internal state

### 4. **Error Messages**
Provide clear error messages that guide fixes:
```python
# BAD
assert x.size(1) == self.channels

# GOOD
assert x.size(1) == self.channels, \
    f"Expected {self.channels} channels, got {x.size(1)}"
```

## Pre-flight Checklist for Agents

Before considering a module complete:

1. **Run shape assertions** on all intermediate tensors
2. **Test with both train and eval modes** (especially for BatchNorm, Dropout)
3. **Verify device consistency** with inputs on different devices
4. **Check mask handling** with None, boolean, and float masks
5. **Test edge cases**: empty batches, single samples, extreme dimensions

## Testing Pattern

```python
def test_module_completeness(module, sample_input):
    # Device consistency
    assert all(p.device == sample_input.device for p in module.parameters())
    
    # Shape consistency
    output = module(sample_input)
    assert output.dim() == expected_dims
    
    # Eval mode behavior
    module.eval()
    eval_output = module(sample_input)
    assert eval_output.shape == output.shape
    
    # Mask handling (if applicable)
    if hasattr(module, 'forward') and 'mask' in inspect.signature(module.forward).parameters:
        module(sample_input, mask=None)  # Should work
        module(sample_input, mask=torch.ones(..., dtype=torch.bool))  # Should work
```

## Common Pitfalls by Module Type

### Transformer/Attention Modules
- Mask format (boolean vs additive)
- Positional encoding dimension calculations
- Key padding mask vs attention mask naming

### Stateful Modules (Memory, Buffers)
- Tensor assignment vs copy operations
- Device placement for dynamically created tensors
- Gradient tracking (use .detach() when storing)

### Streaming/Temporal Modules
- Collection type conversions (deque ↔ list)
- Timestamp handling and time window calculations
- Buffer cleanup and memory management

### Sampling/Data Modules
- Integer conversion for indices
- Handling edge cases (empty classes, single samples)
- Maintaining reproducibility with seeds

## Parameterization Guidelines

When designing modules with potentially ambiguous behavior, make expectations explicit through parameters:

### 11. **Content-Based vs Time-Based Uniqueness**
```python
# BAD: Implicit decision about deduplication
def generate_id(data):
    return hash(data.content + timestamp)  # Mixed concerns

# GOOD: Explicit parameter
def __init__(self, deduplicate: bool = True):
    self.deduplicate = deduplicate
    
def generate_id(self, data):
    if self.deduplicate:
        return hash(data.content)  # Content-based
    else:
        return hash(data.content + timestamp)  # Unique per call
```

### 12. **Primary/Secondary Entity Relationships**
```python
# BAD: Implicit primary stream selection
def join_streams(self):
    primary = list(self.streams.keys())[0]  # Arbitrary choice

# GOOD: Explicit designation
def __init__(self, primary_stream: Optional[str] = None):
    self.primary_stream = primary_stream
    
def join_streams(self):
    if self.primary_stream:
        primary = self.primary_stream
    else:
        # Fallback with clear logic
        primary = self.get_most_recent_stream()
```

### 13. **Merge/Join Strategies**
```python
# BAD: Hard-coded merge behavior
def merge(self, other):
    return (self.value + other.value) / 2

# GOOD: Strategy parameter
def merge(self, other, strategy: str = "mean"):
    if strategy == "mean":
        return (self.value + other.value) / 2
    elif strategy == "theirs":
        return other.value
    elif strategy == "ours":
        return self.value
```

## Audio-Specific Module Guidelines

From our audio module exploration, additional patterns emerged:

### 14. **Signal Processing Patterns**
```python
# BAD: Ignoring Nyquist theorem
downsample = x[:, :, ::stride]  # Causes aliasing

# GOOD: Apply low-pass filter before downsampling
filtered = low_pass_filter(x, cutoff=1.0/stride)
downsampled = filtered[:, :, ::stride]
```

### 15. **Filter Design**
```python
# BAD: Hard-coded filter coefficients
kernel = torch.tensor([0.25, 0.5, 0.25])

# GOOD: Parameterized filter generation
def generate_filter(filter_type='lanczos', size=5, cutoff=0.5):
    # Generate based on signal processing principles
```

### 16. **Streaming/Real-time Constraints**
```python
# BAD: Looking into the future
output[t] = f(input[t-1], input[t], input[t+1])  # Non-causal

# GOOD: Causal operations only
output[t] = f(input[t-k] for k in range(window_size))  # Past only
```

### 17. **Domain Boundaries**
```python
# Recognize when a pattern is too complex for a single module:
# ❌ Multi-scale + Multi-architecture + Complex coordination
# ✅ Single purpose with clear parameters

# If you need multiple architecture variants, parameterize:
def __init__(self, mode='standard'):  # Not 5 different classes
```

## Complexity Budget Guidelines

When implementing complex modules:

1. **Set a complexity budget** (e.g., max 3 major design decisions)
2. **Parameterize ambiguities early** rather than making assumptions
3. **Fail fast** if coordination between components becomes too tight
4. **Consider splitting** if the module does more than one thing

Example:
- ✅ VectorQuantizer: Single codebook, clear purpose
- ⚠️ ResidualVectorQuantizer: Multiple codebooks but clear hierarchy
- ❌ MultiScaleMultiArchitectureDiscriminator: Too many variations

## Summary

These guidelines would have prevented ~90% of the fixes needed. The remaining 10% were resolved through parameterization of ambiguous behaviors. Following these patterns will significantly improve the reliability of agent-generated PyTorch modules.

Key principles:
1. **When in doubt, parameterize** - it's better to have an explicit parameter than to make implicit assumptions about user intent.
2. **Single purpose clarity** - a module should do one thing well
3. **Domain expertise != Complexity** - specialized modules can still be simple