# Implementation Specifications for All Modules

This document contains every implementation decision extracted from our tests. These are the explicit choices that agents must follow to achieve compatibility.

## Format Guide
- `param_name: type` = required parameter
- `param_name: type = default` = optional parameter with default
- `-> ReturnType` = what the method returns
- `# CRITICAL:` = deviation will cause test failure

---

## ML Components

### 1. transformer_block

```python
# CLASS NAME
TransformerBlock

# INITIALIZATION PARAMETERS
d_model: int                    # Hidden dimension size
n_heads: int                    # Number of attention heads (CRITICAL: not num_heads)
d_ff: int = None               # Feedforward dimension (defaults to 4*d_model)
dropout: float = 0.1           # Dropout rate
activation: str = 'gelu'       # Activation function
norm_first: bool = True        # Pre-norm vs post-norm

# FORWARD METHOD
def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    # x: [batch, seq_len, d_model]
    # mask: Optional attention mask (additive format, not boolean)
    # Returns: Tensor of same shape as input

# RETURN FORMAT
- Type: Tensor (CRITICAL: not dict)
- Shape: Same as input

# BEHAVIORAL REQUIREMENTS
- Must preserve sequence length
- Must preserve hidden dimension
- Must support optional masking
- Must be differentiable
```

### 2. conv_encoder

```python
# CLASS NAME
ConvEncoder

# INITIALIZATION PARAMETERS  
in_channels: int = 3           # Input channels
base_channels: int = 64        # Base channel count
num_layers: int = 4            # Number of conv layers
latent_dim: int = 512         # Output dimension

# FORWARD METHOD
def forward(self, x: Tensor) -> Dict[str, Tensor]:
    # x: [batch, channels, height, width]
    # Returns: Dictionary with specific keys

# RETURN FORMAT (CRITICAL: Must be dict)
{
    'features': Tensor,        # Feature maps before pooling [B, C, H, W]
    'pooled': Tensor,         # After global pooling [B, latent_dim]
    'latent': Tensor          # Final representation [B, latent_dim]
}

# BEHAVIORAL REQUIREMENTS
- Progressive spatial downsampling
- Output dict must have all three keys
- 'pooled' and 'latent' may be identical
```

### 3. sequence_encoder

```python
# CLASS NAME  
SequenceEncoder

# INITIALIZATION PARAMETERS
input_dim: int                 # Input feature dimension
d_model: int                   # Model dimension
n_heads: int                   # Number of attention heads (CRITICAL: not num_heads)
n_layers: int                  # Number of transformer layers
max_len: int = 5000           # Maximum sequence length

# FORWARD METHOD
def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    # x: [batch, seq_len, input_dim]
    # Returns: [batch, seq_len, d_model]

# RETURN FORMAT
- Type: Tensor (encoded sequence)
- Shape: [batch, seq_len, d_model]

# BEHAVIORAL REQUIREMENTS
- Must handle variable length sequences
- Must support optional padding mask
- Positional encoding required
```

### 4. attention_decoder

```python
# CLASS NAME
AttentionDecoder  

# INITIALIZATION PARAMETERS
d_model: int                   # Model dimension
n_heads: int                   # Attention heads (CRITICAL: not num_heads)
num_layers: int               # Number of decoder layers  
vocab_size: int               # Output vocabulary size

# FORWARD METHOD
def forward(self, x: Tensor, memory: Tensor, 
           tgt_mask: Optional[Tensor] = None,
           memory_mask: Optional[Tensor] = None) -> Tensor:
    # x: [batch, tgt_len, d_model] - target sequence
    # memory: [batch, src_len, d_model] - encoder output
    # Returns: [batch, tgt_len, vocab_size]

# RETURN FORMAT
- Type: Tensor (logits)
- Shape: [batch, tgt_len, vocab_size]

# BEHAVIORAL REQUIREMENTS
- Must support cross-attention to memory
- Must support causal masking for tgt
- Output should be logits (pre-softmax)
```

### 5. vit_patch_encoder

```python
# CLASS NAME
ViTPatchEncoder

# INITIALIZATION PARAMETERS
img_size: int = 224           # Input image size (CRITICAL: not image_size)
patch_size: int = 16          # Patch size
in_chans: int = 3            # Input channels (CRITICAL: not in_channels)
embed_dim: int = 768         # Embedding dimension
depth: int = 12              # Number of transformer blocks
num_heads: int = 12          # Number of attention heads

# FORWARD METHOD  
def forward(self, x: Tensor) -> Dict[str, Tensor]:
    # x: [batch, channels, height, width]
    # Returns: Dictionary with features

# RETURN FORMAT
{
    'features': Tensor,       # All patch embeddings [B, N, embed_dim]
    'cls_token': Tensor,     # Class token [B, embed_dim]
}

# BEHAVIORAL REQUIREMENTS
- Must add learnable positional embeddings
- Must include CLS token
- Number of patches = (img_size/patch_size)^2
```

### 6. cross_modal

```python
# CLASS NAME
CrossModalFusion

# INITIALIZATION PARAMETERS
dim_a: int                    # Dimension of modality A
dim_b: int                    # Dimension of modality B  
hidden_dim: int              # Hidden dimension
output_dim: int              # Output dimension
fusion_type: str = 'concat'  # 'concat', 'add', 'multiply', 'attention'

# FORWARD METHOD
def forward(self, feat_a: Tensor, feat_b: Tensor) -> Tensor:
    # feat_a: [batch, seq_a, dim_a]
    # feat_b: [batch, seq_b, dim_b]
    # Returns: [batch, output_dim] or [batch, seq, output_dim]

# BEHAVIORAL REQUIREMENTS
- Must handle different sequence lengths
- Must project to common space before fusion
- Output depends on fusion_type
```

### 7. timeseries_encoder

```python
# CLASS NAME
TimeSeriesEncoder

# INITIALIZATION PARAMETERS
input_dim: int               # Number of input features
hidden_dim: int             # Hidden dimension
n_layers: int = 2           # Number of RNN/LSTM layers
model_type: str = 'lstm'    # 'lstm', 'gru', 'transformer'
bidirectional: bool = True  # Use bidirectional encoding

# FORWARD METHOD
def forward(self, x: Tensor, lengths: Optional[Tensor] = None) -> Dict[str, Tensor]:
    # x: [batch, seq_len, input_dim]
    # lengths: Optional actual lengths for packing
    # Returns: Dictionary

# RETURN FORMAT
{
    'sequence': Tensor,     # Full sequence output [B, L, hidden]
    'last': Tensor,        # Last timestep [B, hidden]
    'pooled': Tensor       # Pooled representation [B, hidden]
}
```

### 8. set_encoder

```python
# CLASS NAME
SetEncoder

# INITIALIZATION PARAMETERS
input_dim: int              # Input feature dimension  
d_model: int               # Model dimension (CRITICAL: not hidden_dim)
n_heads: int = 4          # Number of attention heads
n_layers: int = 2         # Number of self-attention layers
pooling: str = 'mean'     # 'mean', 'max', 'attention'

# FORWARD METHOD
def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    # x: [batch, set_size, input_dim]
    # mask: Optional [batch, set_size] validity mask
    # Returns: [batch, d_model]

# BEHAVIORAL REQUIREMENTS
- Must be permutation invariant
- Must handle variable set sizes with mask
- Output is single vector per set
```

### 9. contrastive

```python
# CLASS NAME
ContrastiveLearner

# INITIALIZATION PARAMETERS
encoder_dim: int            # Input dimension
projection_dim: int = 128  # Projection head output
temperature: float = 0.07  # Temperature parameter

# FORWARD METHOD
def forward(self, x1: Tensor, x2: Tensor) -> Dict[str, Tensor]:
    # x1, x2: [batch, encoder_dim] - paired samples
    # Returns: Loss and embeddings

# RETURN FORMAT
{
    'loss': Tensor,         # Contrastive loss (scalar)
    'embeddings': Tensor,   # Projected embeddings [2B, projection_dim]
    'similarity': Tensor    # Similarity matrix [2B, 2B]
}

# REQUIRED METHODS
def compute_loss(self, embeddings: Tensor) -> Tensor:
    # Compute InfoNCE or similar loss
```

### 10. autoencoder

```python
# CLASS NAME
AutoEncoder

# INITIALIZATION PARAMETERS
input_dim: int             # Input dimension
hidden_dims: List[int]     # Hidden layer dimensions
latent_dim: int           # Latent dimension
activation: str = 'relu'   # Activation function

# FORWARD METHOD
def forward(self, x: Tensor) -> Dict[str, Tensor]:
    # x: Input tensor
    # Returns: Dictionary

# RETURN FORMAT  
{
    'reconstruction': Tensor,  # Reconstructed input
    'latent': Tensor,         # Latent representation
}

# REQUIRED METHODS
def encode(self, x: Tensor) -> Tensor:
    # Returns: latent representation

def decode(self, z: Tensor) -> Tensor:
    # Returns: reconstruction
```

### 11. seq2seq

```python
# CLASS NAME
Seq2SeqModel

# INITIALIZATION PARAMETERS
input_vocab_size: int      # Source vocabulary size
output_vocab_size: int     # Target vocabulary size
d_model: int              # Model dimension
n_heads: int              # Number of attention heads
n_layers: int = 6         # Number of layers
max_len: int = 5000      # Maximum sequence length

# FORWARD METHOD
def forward(self, src: Tensor, tgt: Tensor,
           src_mask: Optional[Tensor] = None,
           tgt_mask: Optional[Tensor] = None) -> Tensor:
    # src: [batch, src_len] - source token ids
    # tgt: [batch, tgt_len] - target token ids  
    # Returns: [batch, tgt_len, output_vocab_size]

# REQUIRED METHODS
def encode(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
    # Returns: encoder output

def decode(self, tgt: Tensor, memory: Tensor, 
          tgt_mask: Optional[Tensor] = None,
          memory_mask: Optional[Tensor] = None) -> Tensor:
    # Returns: decoder output
```

### 12. graph_encoder

```python
# CLASS NAME
GraphEncoder

# INITIALIZATION PARAMETERS
input_dim: int            # Node feature dimension (CRITICAL: not in_features)
hidden_dim: int          # Hidden dimension
output_dim: int          # Output dimension
n_layers: int = 2        # Number of GNN layers
dropout: float = 0.1     # Dropout rate
aggregation: str = 'mean' # 'mean', 'max', 'sum'

# FORWARD METHOD
def forward(self, x: Tensor, edge_index: Tensor, 
           edge_attr: Optional[Tensor] = None,
           batch: Optional[Tensor] = None) -> Dict[str, Tensor]:
    # x: [num_nodes, input_dim] - node features
    # edge_index: [2, num_edges] - edge connectivity
    # batch: Optional batch assignment
    # Returns: Dictionary

# RETURN FORMAT
{
    'node_features': Tensor,  # [num_nodes, output_dim]
    'graph_embedding': Tensor # [batch_size, output_dim] if batch provided
}

# BEHAVIORAL REQUIREMENTS
- Must handle batched graphs
- Must support edge attributes
- Global pooling if batch provided
```

### 13. memory_retriever

```python
# CLASS NAME
MemoryBank

# INITIALIZATION PARAMETERS
memory_size: int          # Number of memory slots
d_model: int             # Memory dimension
n_heads: int = 8         # Attention heads for retrieval
retrieval_method: str = 'attention'  # 'attention', 'cosine', 'learned'

# FORWARD METHOD
def forward(self, query: Tensor, update: bool = False) -> Dict[str, Tensor]:
    # query: [batch, query_dim]
    # update: Whether to update memory with query
    # Returns: Retrieved memory and scores

# RETURN FORMAT
{
    'retrieved': Tensor,     # [batch, d_model]
    'scores': Tensor,       # [batch, memory_size]
    'updated': bool         # Whether memory was updated
}

# REQUIRED METHODS
def update_memory(self, key: Tensor, value: Tensor) -> None:
    # Update memory bank

def clear(self) -> None:
    # Clear all memory
```

### 14. adaptive_computation

```python
# CLASS NAME
AdaptiveComputationTime  # CRITICAL: Must be primary export

# ADDITIONAL REQUIRED EXPORTS
PonderNet                # Secondary implementation
UniversalTransformer     # Third implementation

# AdaptiveComputationTime PARAMETERS
input_dim: int           # Input dimension
hidden_dim: int         # Hidden dimension
max_steps: int = 10     # Maximum pondering steps
threshold: float = 0.99 # Halting threshold

# FORWARD METHOD
def forward(self, x: Tensor) -> Dict[str, Tensor]:
    # x: Input tensor
    # Returns: Dictionary with computation info

# RETURN FORMAT
{
    'output': Tensor,       # Final output
    'halting_prob': Tensor, # Halting probabilities
    'n_steps': Tensor      # Steps taken per sample
}
```

## Data Pipeline Components

### 15. stream_processor

```python
# CLASS NAME
StreamProcessor

# INITIALIZATION PARAMETERS
window_size: int             # Window size
time_based: bool = False     # Count vs time-based windows (CRITICAL: not time_window) 
stride: int = None           # Window stride (defaults to window_size)
aggregation: str = 'mean'   # 'mean', 'sum', 'max', 'min', 'last'

# FORWARD METHOD  
def forward(self, data: Tensor, timestamp: Optional[float] = None) -> Optional[Tensor]:
    # data: Input data point
    # timestamp: Required if time_based=True
    # Returns: Aggregated window or None if not ready

# REQUIRED METHODS
def reset(self) -> None:
    # Clear buffer

def get_buffer(self) -> List[Tuple[Tensor, float]]:
    # Get current buffer contents
```

### 16. data_validator

```python
# CLASS NAME
DataValidator

# REQUIRED ADDITIONAL CLASS
Schema  # CRITICAL: Must also export Schema class

# Schema STRUCTURE
class Schema:
    def __init__(self, rules: Dict[str, Any]):
        self.rules = rules
    
    def validate(self, data: Dict) -> Tuple[bool, List[str]]:
        # Returns: (is_valid, list_of_errors)

# DataValidator PARAMETERS
schema: Schema              # Validation schema
strict: bool = True        # Strict mode
auto_correct: bool = False # Auto-correct minor issues

# FORWARD METHOD
def forward(self, data: Dict[str, Tensor]) -> Dict[str, Any]:
    # Returns: Validation results

# RETURN FORMAT
{
    'valid': bool,          # Overall validity
    'errors': List[str],    # List of errors
    'corrected': Dict,      # Corrected data if auto_correct
    'metadata': Dict        # Validation metadata
}
```

### 17. feature_store

```python
# CLASS NAME
FeatureStore

# REQUIRED ADDITIONAL CLASS
FeatureCompute  # CRITICAL: Must export

# FeatureCompute STRUCTURE  
class FeatureCompute:
    def __init__(self, compute_fn: Callable):
        self.compute_fn = compute_fn
    
    def __call__(self, *args) -> Tensor:
        return self.compute_fn(*args)

# FeatureStore PARAMETERS
cache_size: int = 1000      # LRU cache size
compute_on_miss: bool = True # Auto-compute missing features
ttl: float = None           # Time-to-live in seconds

# FORWARD METHOD
def forward(self, feature_id: str) -> Optional[Tensor]:
    # Retrieve feature by ID

# REQUIRED METHODS  
def register_compute(self, feature_id: str, compute: FeatureCompute) -> None:
    # Register compute function

def store(self, feature_id: str, value: Tensor) -> None:
    # Store computed feature

def invalidate(self, feature_id: str) -> None:
    # Invalidate cached feature
```

### 18. data_versioner

```python
# CLASS NAME
DataVersioner

# INITIALIZATION PARAMETERS
storage_backend: str = 'memory'  # 'memory', 'disk', 'database'
deduplicate: bool = True         # Content-based deduplication (CRITICAL parameter)
compression: bool = False        # Enable compression

# CALLABLE INTERFACE (CRITICAL)
def __call__(self, data: Tensor, message: str) -> str:
    # Create version
    # Returns: version_id (string)

# REQUIRED METHODS
def load(self, version_id: str) -> Tensor:
    # Load version by ID (CRITICAL: method name is 'load' not 'get')

def list_versions(self) -> List[Dict[str, Any]]:
    # List all versions with metadata
    # Returns: [{'id': str, 'message': str, 'timestamp': float, 'size': int}]

def diff(self, v1_id: str, v2_id: str) -> Dict[str, Any]:
    # Compare two versions

# BEHAVIORAL REQUIREMENTS
- Must be callable: version_id = versioner(data, message)
- Deduplication based on content hash when deduplicate=True
- Version IDs should be strings
```

### 19. stream_joiner

```python
# CLASS NAME
StreamJoiner

# INITIALIZATION PARAMETERS  
join_type: str = 'inner'        # 'inner', 'left', 'outer'
time_window: float              # Time window for joins
buffer_size: int                # Max buffer size (CRITICAL: required parameter)
primary_stream: Optional[str] = None  # Primary stream for left joins

# FORWARD METHOD (CRITICAL: timestamp is positional)
def forward(self, stream_id: str, data: Tensor, timestamp: float) -> Optional[Dict[str, Tensor]]:
    # stream_id: Stream identifier
    # data: Data point
    # timestamp: POSITIONAL argument (not kwarg)
    # Returns: Joined data or None

# RETURN FORMAT (when join occurs)
{
    'stream1_name': Tensor,
    'stream2_name': Tensor,
    ...  # All joined streams
    'timestamp': float  # Join timestamp
}

# REQUIRED METHODS
def get_buffer_state(self) -> Dict[str, int]:
    # Get buffer sizes per stream
```

### 20. data_sampler

```python
# CLASS NAME  
DataSampler

# INITIALIZATION PARAMETERS
strategy: str                   # 'uniform', 'weighted', 'stratified', 'importance'
batch_size: int                # Batch size (CRITICAL: not bs or batch)
replacement: bool = True       # Sample with replacement
seed: Optional[int] = None     # Random seed

# FORWARD METHOD
def forward(self, data: Tensor, weights: Optional[Tensor] = None) -> Tensor:
    # data: [N, ...] dataset
    # weights: Optional sampling weights
    # Returns: [batch_size, ...] sampled batch

# REQUIRED METHODS
def set_weights(self, weights: Tensor) -> None:
    # Update sampling weights

def get_statistics(self) -> Dict[str, Any]:
    # Get sampling statistics
```

## Audio Components

### 21. snake_activation

```python
# REQUIRED EXPORTS (CRITICAL: all three must exist)
__all__ = ['SnakeActivation', 'SnakeBeta', 'snake']

# SnakeActivation CLASS
class SnakeActivation(nn.Module):
    def __init__(self, channels: int, alpha: float = 1.0):
        # channels: Number of channels
        # alpha: Frequency parameter

# SnakeBeta CLASS  
class SnakeBeta(nn.Module):
    def __init__(self, channels: int, alpha: float = 1.0, beta: float = 1.0):
        # beta: Additional shaping parameter

# FUNCTION INTERFACE
def snake(x: Tensor, alpha: float = 1.0) -> Tensor:
    # Functional interface
    # Returns: sin^2(x) based activation

# BEHAVIORAL REQUIREMENTS
- Must be smooth and differentiable
- Approximately periodic with parameter alpha
- SnakeBeta adds additional shaping
```

### 22. causal_conv

```python
# REQUIRED EXPORTS (CRITICAL: both must exist)
__all__ = ['CausalConv1d', 'CausalConvTranspose1d']

# CausalConv1d CLASS
class CausalConv1d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int, 
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True):
        pass

# CausalConvTranspose1d CLASS  
class CausalConvTranspose1d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True):
        pass

# BEHAVIORAL REQUIREMENTS
- Must be strictly causal (no future lookahead)
- Output at time t depends only on inputs up to time t
- Handle stride/dilation correctly for causality
```

### 23. stft_loss

```python
# CLASS NAME
MultiScaleSTFTLoss

# INITIALIZATION PARAMETERS
fft_sizes: List[int] = [1024, 2048, 512]  # FFT sizes
hop_sizes: List[int] = [120, 240, 50]     # Hop sizes  
win_lengths: List[int] = [600, 1200, 240] # Window lengths
window: str = 'hann'                       # Window function
epsilon: float = 1e-7                      # Numerical stability

# FORWARD METHOD
def forward(self, pred: Tensor, target: Tensor) -> Dict[str, Tensor]:
    # pred: [batch, time] or [batch, channels, time]
    # target: Same shape as pred
    # Returns: Dictionary of losses

# RETURN FORMAT
{
    'loss': Tensor,              # Combined loss (scalar)
    'spectral_loss': Tensor,     # Spectral convergence  
    'magnitude_loss': Tensor,    # Log magnitude L1
    'phase_loss': Optional[Tensor] # Phase loss if computed
}

# REQUIRED METHODS
def compute_stft_loss(self, pred_stft: Tensor, target_stft: Tensor) -> Dict[str, Tensor]:
    # Compute losses for single STFT scale
```

### 24. antialiased_conv

```python
# REQUIRED EXPORTS
__all__ = ['AntialiasedConv1d', 'AntialiasedConv2d', 'AntialiasedMaxPool1d', 'AntialiasedMaxPool2d']

# AntialiasedConv2d CLASS (primary)
class AntialiasedConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: str = 'same',  # 'same', 'valid', or int
                 filter_type: str = 'kaiser',  # 'kaiser', 'gaussian'
                 beta: float = 12.0):    # Kaiser window parameter
        pass

# BEHAVIORAL REQUIREMENTS
- Apply low-pass filtering before downsampling
- Support multiple filter types
- Preserve spatial/temporal alignment
- Work with different padding modes
```

### 25. residual_vector_quantizer

```python
# REQUIRED EXPORTS (CRITICAL: both classes needed)
__all__ = ['VectorQuantizer', 'ResidualVectorQuantizer', 'ResidualVectorQuantizerWrapper']

# VectorQuantizer CLASS (base)
class VectorQuantizer(nn.Module):
    def __init__(self,
                 codebook_size: int,
                 embedding_dim: int,
                 commitment_cost: float = 0.25,
                 decay: float = 0.99,
                 epsilon: float = 1e-5):
        pass
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # Returns: {'quantized': Tensor, 'indices': Tensor, 'loss': Tensor}

# ResidualVectorQuantizer CLASS  
class ResidualVectorQuantizer(nn.Module):
    def __init__(self,
                 n_quantizers: int,
                 codebook_size: int,
                 embedding_dim: int,
                 commitment_cost: float = 0.25):
        pass
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # Returns: {'quantized': Tensor, 'indices': List[Tensor], 'loss': Tensor}

# BEHAVIORAL REQUIREMENTS
- Hierarchical quantization (coarse to fine)
- Indices shape: [batch, sequence] for each level
- Quantized output reconstructed from all levels
```

---

## Global Conventions Summary

### Parameter Naming
- Always use underscores, never camelCase
- Attention heads: `n_heads` (not `num_heads`)
- Model dimension: `d_model` (not `hidden_dim` for transformers)
- Image size: `img_size` (not `image_size`)
- Input channels: `in_channels` or `in_chans` (see specific module)
- Batch size: `batch_size` (never `bs`)

### Return Formats
- Encoders typically return dictionaries
- Decoders typically return tensors
- Loss modules return dictionaries with 'loss' key
- Data modules vary (see specific module)

### Method Names
- Data loading: `load()` (not `get()` or `retrieve()`)
- Data storage: `save()` or `store()`
- Buffer clearing: `reset()` or `clear()`
- Statistics: `get_statistics()` or `get_stats()`

### Testing Notes
These specifications represent what the tests expect. Following them exactly will result in passing tests. Any deviation in parameter names, return formats, or method signatures will cause test failures.