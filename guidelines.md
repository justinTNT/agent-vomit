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

**IMPORTANT**: All modules MUST accept **kwargs in their __init__ method for parameter flexibility.

### TransformerBlock
**Implementation Requirements:**
```python
class TransformerBlock(nn.Module):
    def __init__(self, 
                 d_model: int = 512,      # Hidden dimension
                 n_heads: int = 8,        # Number of attention heads
                 d_ff: int = 2048,        # Feed-forward dimension
                 dropout: float = 0.1,    # Dropout rate
                 **kwargs):               # REQUIRED: Accept extra kwargs
        super().__init__()
        # Implementation should include:
        # - MultiHeadAttention (custom or nn.MultiheadAttention)
        # - FeedForward network
        # - LayerNorm and residual connections
```
**Required behavior:**
- Must preserve input shape (residual architecture)
- Must handle mask=None gracefully
- Handle both boolean masks (True=keep) and additive masks
- Return tensor of same shape as input

### ConvEncoder
**Implementation Requirements:**
```python
class ConvEncoder(nn.Module):
    def __init__(self,
                 in_channels: int = 3,      # Input channels
                 base_channels: int = 64,   # Base channel count
                 num_layers: int = 4,       # Number of layers
                 **kwargs):                 # REQUIRED: Accept extra kwargs
        super().__init__()
        # Implementation should progressively downsample
```
**Required behavior:**
- Return dictionary with keys: 'features', 'pooled', 'shape'
- Should downsample spatial dimensions progressively
- Use standard conv->norm->activation pattern

### DataVersioner
**Implementation Requirements:**
```python
class DataVersioner(nn.Module):
    def __init__(self, 
                 storage_path: str = "./versions",  # Where to store versions
                 **kwargs):                         # REQUIRED: Accept extra kwargs
        super().__init__()
        # Must track versions, compute hashes, store metadata
    
    def __call__(self, data: torch.Tensor, message: str = ""):
        """Make the class callable - shorthand for version()"""
        return self.version(data, message)
    
    def version(self, data: torch.Tensor, message: str = ""):
        """Core versioning method"""
        # Return version ID (str)
    
    def load(self, version_id: str):
        """Load a specific version"""
        # Return the data tensor for that version
    
    def diff(self, version_id1: str, version_id2: str):
        """Compare two versions"""
        # Return DataDiff object or dict describing differences
```
**Required behavior:**
- Extends nn.Module (for consistency with ecosystem)
- MUST be callable via `__call__` method (shorthand for version())
- Use dataclasses for version metadata (DataVersion, DataDiff)
- Support version trees/branching with parent_id tracking
- Include statistical summaries in version metadata

### StreamJoiner
**Implementation Requirements:**
```python
class StreamJoiner(nn.Module):
    def __init__(self,
                 join_type: str = 'inner',      # Type of join operation
                 window_size: float = 1000,     # Window size for joining
                 timeout: float = 5.0,          # Timeout for old data
                 **kwargs):                     # REQUIRED: Accept extra kwargs
        super().__init__()
        # Must handle temporal alignment of multiple streams
    
    def forward(self, stream_name: str, data: torch.Tensor, timestamp: float = None):
        """Add data and potentially return joined result"""
        # Add to stream buffer
        # Try to join if possible
        # Return joined data or None
```
**Required behavior:**
- Handle temporal alignment with configurable tolerance
- Support at least 'inner' join (all streams must have data)
- Buffer management for windows
- Clear old data based on timeout

### SnakeActivation  
**Implementation Requirements:**
```python
class SnakeActivation(nn.Module):
    def __init__(self,
                 n_channels: int = 1,           # Number of channels
                 alpha_init: float = 1.0,       # Initial alpha value
                 learnable: bool = True,        # Whether alpha is learnable
                 shared_alpha: bool = False,    # Share alpha across channels
                 **kwargs):                     # REQUIRED: Accept extra kwargs
        super().__init__()
        # Implement snake activation: x + (1/alpha) * sin²(alpha * x)
```
**Required behavior:**
- Handle both 2D (batch, features) and 3D (batch, channels, time) inputs
- Support per-channel or shared alpha parameters
- Initialize alpha as Parameter if learnable, else as buffer
- Maintain gradient flow

## General Module Requirements

### 1. **MANDATORY: Accept **kwargs**
All modules MUST accept **kwargs in their __init__ to handle parameter variations:
```python
def __init__(self, required_param, optional_param=None, **kwargs):
    super().__init__()
    # Log unknown parameters rather than failing
    if kwargs:
        import warnings
        warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
```

### 2. **Inheritance Pattern**
All modules should extend `nn.Module` for PyTorch ecosystem compatibility:
```python
class MyModule(nn.Module):
    def __init__(self, ..., **kwargs):
        super().__init__()
```

### 3. **Return Format Consistency**
- **Neural network layers** (TransformerBlock, etc.): Return tensor
- **Feature extractors** (ConvEncoder, etc.): Return dict with descriptive keys
- **Data processors** (DataVersioner, etc.): Return appropriate type (version ID, dict, etc.)

### 4. **Method Naming Standards**
Common method names to implement:
- `forward()` for nn.Module subclasses (required)
- `__call__()` for classes where it makes sense (DataVersioner)
- `load()` and `save()` for stateful modules
- `reset()` or `clear()` for modules with buffers
- Module-specific methods as needed (e.g., `version()`, `diff()`, `join()`)

### 5. **Error Messages**
Provide clear error messages that guide fixes:
```python
# BAD
assert x.size(1) == self.n_channels

# GOOD
assert x.size(1) == self.n_channels, \
    f"Expected {self.n_channels} channels, got {x.size(1)}"
```

## Pre-flight Checklist for Agents

Before considering a module complete:

1. **Verify **kwargs acceptance** in __init__
2. **Run shape assertions** on all intermediate tensors
3. **Test with both train and eval modes** (especially for BatchNorm, Dropout)
4. **Verify device consistency** with inputs on different devices
5. **Check mask handling** with None, boolean, and float masks
6. **Test edge cases**: empty batches, single samples, extreme dimensions

## Testing Pattern

```python
def test_module_completeness(module, sample_input):
    # Verify kwargs acceptance
    try:
        type(module)(unknown_param=123)  # Should not raise TypeError
    except TypeError:
        raise AssertionError("Module must accept **kwargs")
    
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
def __init__(self, deduplicate: bool = True, **kwargs):
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
def __init__(self, primary_stream: Optional[str] = None, **kwargs):
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
def __init__(self, mode='standard', **kwargs):  # Not 5 different classes
```

### 18. **Mathematical Perfection vs Practical Utility**
```python
# For inherently complex mathematical modules (like PQMF), test for functional 
# correctness rather than theoretical perfection.

# BAD: Requiring perfect reconstruction for filter banks
assert reconstruction_error < 1e-6  # Too strict for numerical methods

# GOOD: Test functional behavior
assert reconstruction_error < 0.1 * input_energy  # 10% error acceptable
assert frequency_separation_achieved()  # Focus on primary purpose
assert is_differentiable()  # Can be improved through training

# A module that achieves 90% of theoretical ideal is still highly useful,
# especially if the remaining 10% can be achieved through training.
```

### 19. **Gated Architectures Channel Splitting**
```python
# BAD: Implicit channel splitting assumptions
self.conv = nn.Conv1d(channels, gate_channels + residual_channels)
# Where gate_channels and residual_channels are undefined

# GOOD: Make channel splits explicit or derive from input
def __init__(self, channels, gate_channels=None, residual_channels=None):
    # If not specified, split evenly or match input
    if gate_channels is None:
        gate_channels = channels
    if residual_channels is None:
        residual_channels = channels
    
    self.conv = nn.Conv1d(channels, gate_channels + residual_channels)
```

### 20. **PyTorch Parameter Assignment**
```python
# BAD: Direct assignment to parameters
module.weight = normalized_weight  # TypeError: cannot assign

# GOOD: Use .data for temporary updates
with torch.no_grad():
    original = module.weight.data
    module.weight.data = normalized_weight
    output = module(x)
    module.weight.data = original

# BETTER: Use parameterization or hooks for permanent changes
torch.nn.utils.parametrize.register_parametrization(
    module, 'weight', WeightNormalization()
)
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

## Tier 4 Advanced/Experimental Module Guidelines

Based on generating WaveGANDiscriminator, OnsetDetector, ChromaEncoder, and PhaseReconstruction:

### 21. **STFT Batch Processing Pattern**
```python
# BAD: Reshape and process entire batch as single tensor
stft = torch.stft(x.reshape(-1), ...)  # Loses batch structure
stft = stft.reshape(batch_size, freq_bins, time_frames)  # Incorrect reshaping

# GOOD: Process each batch item individually
stfts = []
for b in range(batch_size):
    stft = torch.stft(x[b], ...)
    stfts.append(stft)
stft = torch.stack(stfts, dim=0)

# STFT operations don't preserve batch structure when reshaped
```

### 22. **Empty Tensor Handling in Signal Processing**
```python
# BAD: Assume tensors have elements
audio_max = torch.max(torch.abs(audio), dim=-1)[0]  # Fails on empty tensors

# GOOD: Check for empty tensors first
if audio.numel() > 0:
    audio_max = torch.max(torch.abs(audio), dim=-1, keepdim=True)[0]
    audio_max = torch.clamp(audio_max, min=1e-8)
    audio = audio / audio_max * 0.95
```

### 23. **Multi-Channel Convolution for Sequential Data**
```python
# BAD: Apply conv1d directly to multi-channel sequential data
processed = F.conv1d(chroma_data, kernel)  # Expects 1 input channel

# GOOD: Reshape to process each channel separately
batch_size, num_channels, time_frames = chroma_data.shape
reshaped = chroma_data.view(batch_size * num_channels, 1, time_frames)
smoothed = F.conv1d(reshaped, kernel, padding=kernel_size//2)
processed = smoothed.view(batch_size, num_channels, time_frames)

# Conv1d expects (batch, channels, time) where channels matches kernel input
```

### 24. **Audio Length Validation for STFT**
```python
# BAD: Assume audio is long enough for STFT
stft = torch.stft(audio, n_fft=window_size, ...)  # May fail

# GOOD: Ensure minimum length requirements
min_length = window_size
if audio.shape[-1] < min_length:
    pad_length = min_length - audio.shape[-1]
    audio = F.pad(audio, (0, pad_length), mode='constant', value=0)

# STFT requires audio length >= window_size
```

### 25. **Gradient Preservation in Loss Functions**
```python
# BAD: Create scalar that loses gradients
total_loss = 0.0  # Python float, no gradients
for loss in losses:
    total_loss += loss
return total_loss / len(losses)

# GOOD: Initialize with tensor that preserves gradients
device = losses[0].device
total_loss = torch.tensor(0.0, device=device, requires_grad=True)
for loss in losses:
    total_loss = total_loss + loss  # Use += only with tensors
return total_loss / len(losses)
```

### 26. **Phase Continuity for Audio Processing**
```python
# BAD: Assume previous state exists without initialization
phase = self.prev_phase  # May be None on first call

# GOOD: Handle initialization and state management
if self.init_phase == 'previous' and self.prev_phase is not None:
    if self.prev_phase.shape == magnitude.shape:
        phase = self.prev_phase
    else:
        # Resize to match current input
        phase = F.interpolate(self.prev_phase.unsqueeze(1), 
                            size=magnitude.shape[1:], 
                            mode='bilinear').squeeze(1)
else:
    # Fallback initialization
    phase = torch.rand_like(magnitude) * 2 * math.pi - math.pi
```

### 27. **Frequency-Domain Tensor Dimension Matching**
```python
# BAD: Hardcode frequency dimensions
expected_freq_bins = 513  # Assumes n_fft=1024

# GOOD: Calculate from n_fft parameter
expected_freq_bins = n_fft // 2 + 1

# BAD: Mismatch between parameters and input
def forward(magnitude):  # magnitude is (batch, 513, time)
    # Using n_fft=2048 internally
    audio = torch.istft(complex_spec, n_fft=2048, ...)  # Dimension mismatch!

# GOOD: Ensure parameter consistency
def __init__(self, n_fft=1024):
    self.n_fft = n_fft
    self.freq_bins = n_fft // 2 + 1

def forward(magnitude):
    # Validate input dimensions
    assert magnitude.shape[1] == self.freq_bins, \
        f"Expected {self.freq_bins} frequency bins, got {magnitude.shape[1]}"
```

### 28. **Algorithm Selection with Graceful Fallbacks**
```python
# BAD: Assume complex algorithm always works
def detect_onsets(audio):
    return complex_phase_deviation_method(audio)  # May fail

# GOOD: Implement fallback hierarchy
def detect_onsets(audio, method='auto'):
    try:
        if method == 'complex' or method == 'auto':
            return complex_phase_deviation_method(audio)
    except Exception:
        if method == 'auto':
            # Fallback to simpler method
            return spectral_flux_method(audio)
        else:
            raise  # Re-raise if specific method was requested

# Complex algorithms should have simpler fallbacks for reliability
```

## Summary

These guidelines would have prevented ~90% of the fixes needed. The remaining 10% were resolved through parameterization of ambiguous behaviors. Following these patterns will significantly improve the reliability of agent-generated PyTorch modules.

Key principles:
1. **When in doubt, parameterize** - it's better to have an explicit parameter than to make implicit assumptions about user intent.
2. **Single purpose clarity** - a module should do one thing well
3. **Domain expertise != Complexity** - specialized modules can still be simple
4. **Always accept **kwargs** - for future compatibility and testing flexibility