# RAVE Interface Consistency Issues & Fixes

## Critical Issues Found

### 🚨 Parameter Naming Inconsistencies
- **FFT Size**: `fft_size` vs `n_fft` 
  - Fix: Standardize on `n_fft` (librosa/torchaudio convention)
- **Hop Length**: `hop_length` vs `hop_size`
  - Fix: Standardize on `hop_length` (librosa convention)  
- **Channels**: `out_channels` vs `output_channels`
  - Fix: Standardize on `out_channels` (PyTorch convention)

### 🔥 Type Conflicts
- **kernel_size**: `int` vs `Union[int, tuple]`
- **stride**: `int` vs `Union[int, tuple]`
- **channels**: `int` vs `List[int]` (context-dependent)

### 📐 Shape Mismatches
- **Audio inputs**: `(B,1,T)` vs `(B,C,T)` vs `(B,T)` vs `(B,T,C)`
  - Fix: Standardize on `(B, C, T)` for conv1d compatibility
- **Encoder outputs**: tensor vs dict vs tuple
  - Fix: Standardize on dict with "latent" key

## Proposed RAVE Interface Standard

### Audio Processing Modules
```python
def __init__(self, 
    sample_rate: int = 22050,
    n_fft: int = 1024,           # NOT fft_size
    hop_length: int = 256,       # NOT hop_size
    win_length: int = 1024,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
    n_mels: int = 80
):
    # Input: (batch_size, channels, time_steps)
    # Output: (batch_size, freq_bins, time_frames) or dict
```

### Convolution Layers
```python
def __init__(self,
    in_channels: int,
    out_channels: int,           # NOT output_channels
    kernel_size: Union[int, Tuple[int, ...]],
    stride: Union[int, Tuple[int, ...]] = 1,
    padding: Union[int, Tuple[int, ...]] = 0,
    dilation: Union[int, Tuple[int, ...]] = 1,
    groups: int = 1,
    bias: bool = True
):
    # Input: (batch_size, in_channels, *spatial_dims)
    # Output: (batch_size, out_channels, *spatial_dims)
```

### Encoder/Decoder Interface
```python
# Encoder output (standardized)
{
    'latent': Tensor,              # (batch_size, latent_dim)
    'mu': Optional[Tensor],        # For VAE
    'logvar': Optional[Tensor],    # For VAE
    'attention_weights': Optional[Tensor]
}

# Decoder input
def forward(self, latent: Tensor, conditioning: Optional[Tensor] = None):
```

### Quantization Interface
```python
def __init__(self,
    num_quantizers: int,
    codebook_size: int = 1024,
    codebook_dim: int = 256,
    commitment_cost: float = 0.25,
    epsilon: float = 1e-5
):

# Output (standardized)
{
    'quantized': Tensor,
    'indices': Tensor, 
    'loss': Tensor,
    'perplexity': Tensor
}
```

## Migration Plan

### Phase 1: Critical Parameter Fixes (1-2 days)
1. **STFTLoss modules**: Change `fft_size` → `n_fft`, `hop_size` → `hop_length`
2. **Convolution modules**: Fix type hints for `kernel_size`, `stride`, etc.
3. **ConvDecoder**: Change `output_channels` → `out_channels`

### Phase 2: Shape Standardization (2-3 days)  
1. **Audio input shapes**: Ensure all modules expect `(B, C, T)`
2. **Encoder outputs**: Return dict with "latent" key
3. **Decoder inputs**: Accept dict format

### Phase 3: Interface Validation (3-5 days)
1. **Base classes**: Create `BaseEncoder`, `BaseDecoder`, `BaseQuantizer` 
2. **Validation utilities**: Shape checking, parameter compatibility
3. **Migration tests**: Ensure backward compatibility

## High-Priority Fixes for RAVE

The 91% success rate will improve significantly by fixing these interface issues:

1. **Audio parameter naming** - Critical for STFT loss interoperability
2. **Shape standardization** - Essential for audio pipeline flow
3. **Encoder/decoder contracts** - Key for VAE/autoencoder chaining
4. **Type consistency** - Prevents runtime errors

These fixes will likely push success rate to 95%+ and enable robust RAVE implementation.