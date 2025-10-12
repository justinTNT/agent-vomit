# Component Hierarchy Analysis

## Current Inconsistencies

### Too High Level (Should Be Broken Down)
- **Attention blocks** → Should be: Query/Key/Value projections, Scaled dot product, Softmax
- **LSTM/GRU cells** → Should be: Gates (forget/input/output), Cell state updaters, Hidden state computers
- **PCA** → Should be: Covariance matrix, Eigendecomposition, Projection
- **SMOTE** → Should be: KNN finder, Synthetic sample generator, Class balancer

### Too Low Level (Could Be Combined)
- **Individual activation functions** → Could group as: Activation function library
- **Individual loss functions** → Could group as: Loss function library
- **Various pooling types** → Could be: Configurable pooling layer

### Missing Primitives
- **Matrix operations**: Dot product, outer product, kronecker product
- **Tensor operations**: Reshape, transpose, squeeze, unsqueeze
- **Reduction operations**: Sum, mean, max along axes
- **Broadcasting operations**: Expand, repeat, tile
- **Indexing operations**: Gather, scatter, masking
- **Distance functions**: Euclidean, cosine, Manhattan
- **Sampling operations**: Top-k, nucleus, temperature scaling
- **Gradient operations**: Stop gradient, gradient reversal

## Proposed Hierarchy

### Level 0: Atomic Operations
```
- Tensor ops: reshape, transpose, slice, concat
- Math ops: matmul, add, multiply, exp, log
- Reduction ops: sum, mean, max, argmax
- Comparison ops: greater, less, equal
```

### Level 1: Basic Transforms
```
- Linear transform: W @ x + b
- Activation: f(x) for various f
- Normalization: (x - μ) / σ variants
- Distance: ||x - y|| variants
```

### Level 2: Compound Components
```
- Attention = Query + Key + Value + Softmax + Weighted Sum
- Convolution = Sliding Window + Linear Transform + Activation
- RNN Cell = Gates + State Update + Activation
```

### Level 3: Full Modules
```
- Transformer Block = MultiHeadAttention + FFN + Residual + Norm
- CNN Block = Conv + Pool + Norm + Activation
- Encoder = Embedding + Positional + Transform Blocks
```

## Missing Component Categories

### Probability/Sampling
- **Probability distributions**: Normal, Categorical, Bernoulli
- **Sampling strategies**: Greedy, beam search, ancestral
- **Noise generators**: Gaussian, uniform, dropout masks

### Graph Operations
- **Message passing**: Node→Edge, Edge→Node aggregation
- **Graph pooling**: Global, hierarchical
- **Graph attention**: GAT-style attention

### Optimization Components
- **Gradient modifiers**: Clipping, scaling, accumulation
- **Learning rate schedules**: Step, exponential, cosine
- **Optimizer states**: Momentum, adaptive moments

### Memory/Storage
- **Buffer managers**: Replay buffers, priority queues
- **Cache mechanisms**: Key-value stores, LRU caches
- **Checkpointing**: State serialization/deserialization

### Data Flow Control
- **Routers**: Input→Component mapping
- **Mixers**: Weighted combination of outputs
- **Gates**: Binary/soft selection mechanisms
- **Switches**: Conditional execution paths

## Composition Patterns

### Sequential
```
Input → Encoder → Transform → Decoder → Output
```

### Parallel
```
Input → [Branch1, Branch2, Branch3] → Merge → Output
```

### Residual
```
Input → Transform → Add(Input) → Output
```

### Hierarchical
```
Atoms → Basic Transforms → Compound Components → Full Modules
```

## Recommended Standardization

1. **Define 4 clear levels**: Atomic, Basic, Compound, Module
2. **Each level should compose from the level below**
3. **Atomic operations are framework primitives** (torch.matmul, etc.)
4. **Modules are user-facing components**

This gives us a consistent algebra where everything builds on clear primitives.