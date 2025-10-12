# Hierarchical Component Inventory for Agent Generation

## Level 0: Atomic Operations (Framework Primitives)
*These are NOT generated - they're the building blocks we call*

### Tensor Manipulation
- `reshape`, `view`, `transpose`, `permute`
- `squeeze`, `unsqueeze`, `expand`, `repeat`
- `concat`, `stack`, `split`, `chunk`
- `slice`, `index_select`, `gather`, `scatter`

### Mathematical Operations
- `matmul`, `bmm`, `einsum`
- `add`, `sub`, `mul`, `div`, `pow`
- `exp`, `log`, `sqrt`, `abs`, `sign`
- `sin`, `cos`, `tanh`, `sigmoid`

### Reduction Operations
- `sum`, `mean`, `max`, `min`
- `argmax`, `argmin`, `topk`, `sort`
- `cumsum`, `cumprod`, `diff`

### Comparison & Logic
- `eq`, `ne`, `gt`, `lt`, `ge`, `le`
- `logical_and`, `logical_or`, `logical_not`
- `where`, `masked_select`, `masked_fill`

## Level 1: Basic Components (Agent-Generated Primitives)

### Linear Transforms
```python
class LinearTransform:
    """W @ x + b with configurable initialization"""
    - Weight initialization strategies
    - Bias options (with/without)
    - Input/output dimension validation
```

### Activation Functions
```python
class ActivationLibrary:
    """Collection of activation functions"""
    - ReLU, LeakyReLU(slope), ELU(alpha)
    - GELU, SiLU/Swish, Mish
    - Sigmoid, Tanh, Softplus
    - Parameterized: PReLU, APL
```

### Normalization Layers
```python
class Normalizer:
    """(x - stats) / scale with various stats computation"""
    - BatchNorm (batch statistics)
    - LayerNorm (layer statistics)
    - InstanceNorm (instance statistics)
    - GroupNorm (group statistics)
    - RMSNorm (root mean square)
```

### Pooling Operations
```python
class Pooler:
    """Reduce spatial/temporal dimensions"""
    - MaxPool, AvgPool, AdaptivePool
    - GlobalPool (to single value)
    - StochasticPool (probabilistic)
```

### Distance Functions
```python
class DistanceMetric:
    """Measure similarity/dissimilarity"""
    - L1 (Manhattan), L2 (Euclidean)
    - Cosine similarity/distance
    - Dot product similarity
    - Mahalanobis, Minkowski(p)
```

### Sampling Operations
```python
class Sampler:
    """Select elements from distributions"""
    - TopK, TopP (nucleus)
    - Temperature scaling
    - Gumbel-softmax (differentiable)
    - Beam search states
```

### Embedding Tables
```python
class EmbeddingLookup:
    """Learnable lookup tables"""
    - Vocabulary embeddings
    - Positional encodings (learned)
    - Category embeddings
    - Time embeddings (cyclical)
```

### Noise Generators
```python
class NoiseLayer:
    """Add controlled randomness"""
    - Gaussian noise (with variance)
    - Dropout masks (with rate)
    - DropConnect, DropBlock
    - Mixup/CutMix augmentation
```

## Level 2: Compound Components (Assembled from Level 1)

### Attention Mechanism
```python
class AttentionHead:
    """Single attention head: Q,K,V → weighted sum"""
    Composed of:
    - 3x LinearTransform (Q, K, V projections)
    - 1x Scaled dot product (QK^T/√d)
    - 1x Softmax
    - 1x Matrix multiply (attention @ V)
```

### Convolution Block
```python
class ConvBlock:
    """Spatial feature extraction"""
    Composed of:
    - 1x Convolution (sliding window linear)
    - 1x Normalization
    - 1x Activation
    - Optional: Pooling
```

### RNN Cell
```python
class RNNCell:
    """Recurrent state update"""
    Composed of:
    - Multiple LinearTransforms (gates)
    - Activation functions (sigmoid/tanh)
    - Element-wise operations
    - State update logic
```

### Feed-Forward Network
```python
class FFN:
    """Two-layer MLP with activation"""
    Composed of:
    - 1x LinearTransform (expand)
    - 1x Activation
    - 1x LinearTransform (project)
    - Optional: Dropout
```

### Cross-Attention
```python
class CrossAttention:
    """Attend from one sequence to another"""
    Composed of:
    - 1x AttentionHead with separate Q and K,V sources
    - Optional: Multi-head variant
```

### Gated Linear Unit
```python
class GLU:
    """Gated activation: x ⊙ σ(gate)"""
    Composed of:
    - 2x LinearTransform (value, gate)
    - 1x Sigmoid
    - 1x Element-wise multiply
```

### Positional Encoder
```python
class PositionalEncoding:
    """Add position information"""
    Composed of:
    - Sinusoidal encoding generator
    - OR: Learned embedding table
    - Addition to input embeddings
```

### Multi-Head Attention
```python
class MultiHeadAttention:
    """Parallel attention heads"""
    Composed of:
    - Nx AttentionHead
    - 1x Concatenation
    - 1x LinearTransform (output projection)
```

## Level 3: Full Modules (Assembled from Level 2)

### Transformer Block
```python
class TransformerBlock:
    """Standard transformer layer"""
    Composed of:
    - 1x MultiHeadAttention
    - 1x Residual connection
    - 1x LayerNorm
    - 1x FFN
    - 1x Residual connection
    - 1x LayerNorm
```

### Vision Transformer Patch Encoder
```python
class ViTPatchEncoder:
    """Convert image to patch embeddings"""
    Composed of:
    - 1x Patch extraction (reshape)
    - 1x LinearTransform (patch projection)
    - 1x PositionalEncoding
    - Optional: CLS token
```

### Sequence Encoder
```python
class SequenceEncoder:
    """Encode variable-length sequences"""
    Composed of:
    - 1x EmbeddingLookup
    - 1x PositionalEncoding
    - Nx TransformerBlock
    - Optional: Pooling strategy
```

### Convolutional Encoder
```python
class ConvEncoder:
    """Hierarchical feature extraction"""
    Composed of:
    - Multiple ConvBlocks
    - Progressive downsampling
    - Optional: Skip connections
```

### Decoder with Attention
```python
class AttentionDecoder:
    """Generate outputs with cross-attention"""
    Composed of:
    - 1x EmbeddingLookup (output)
    - Nx [SelfAttention, CrossAttention, FFN]
    - 1x Output projection
```

## Composition Algebra Rules

### Basic Composition Patterns
1. **Sequential**: A → B → C
2. **Parallel**: A ⊕ B (concat/add/multiply)
3. **Residual**: A → (Identity + Transform)
4. **Gated**: A → (Value × Gate)

### Type Constraints
- Tensor shapes must match for operations
- Attention requires (seq_len, d_model) format
- Convolutions require (batch, channels, spatial) format

### Common Pipelines
```
Text → Tokenize → Embed → Encode → Transform → Decode → Tokens → Text
Image → Patches → Embed → Transform → Pool → Classify
Table → Encode → Transform → Aggregate → Predict
Audio → Spectrogram → ConvEncode → Transform → Classify
```

## What Makes a Good Agent-Generated Component

1. **Clear Mathematical Definition**: No ambiguity in implementation
2. **Standard I/O Contract**: Predictable tensor shapes
3. **Minimal State**: Preferably stateless or simple state
4. **Well-Documented Pattern**: Agents have seen many examples
5. **Testable**: Can verify correctness with simple tests

## Anti-Patterns to Avoid

1. **Stateful Recursion**: Complex state machines
2. **Dynamic Architecture**: Architecture that changes based on input
3. **Custom Optimization**: Novel training algorithms
4. **Hardware-Specific**: CUDA kernels, TPU ops
5. **Research Novelty**: Anything from papers < 2 years old