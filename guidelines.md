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