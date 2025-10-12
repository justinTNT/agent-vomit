# ML/Agent Components: What Coding Agents Can Actually Build Reliably

## Reality Check Criteria
- Must be buildable in <1000 lines of code
- Must have clear input/output contracts
- Must not require novel research or mathematical breakthroughs
- Must be testable with standard datasets/benchmarks
- Must compose cleanly with other components

## Tier 1: Dead Simple & Highly Reliable (Agents Excel Here)

### Encoders/Decoders
- **Categorical encoders**: One-hot, label, target encoding
- **Text encoders**: Bag-of-words, TF-IDF, n-gram, positional
- **Numerical encoders**: Normalization, standardization, binning
- **Time series encoders**: Lag features, rolling statistics, seasonality
- **Image patch encoders**: Sliding window, grid-based extraction
- **Reality**: These are formulaic transformations, agents implement perfectly

### Transcoders (Type-to-Type Converters)
- **Text↔Token**: Tokenizers, detokenizers (BPE, WordPiece, etc)
- **Image↔Vector**: Flatten/reshape, patch extraction/reconstruction
- **Audio↔Spectrogram**: FFT-based conversion, mel-scale transforms
- **Tabular↔Tensor**: DataFrame to array conversions with proper dtypes
- **Graph↔Matrix**: Adjacency matrix, edge list conversions
- **Reality**: Clear mathematical mappings, agents handle reliably

### Transform Layers
- **Attention blocks**: Single/multi-head self-attention
- **Convolution layers**: 1D/2D/3D with various kernels
- **Recurrent cells**: LSTM, GRU, vanilla RNN
- **Pooling layers**: Max, average, adaptive pooling
- **Normalization layers**: BatchNorm, LayerNorm, GroupNorm
- **Reality**: These are standard implementations, agents know the patterns

### Embedding Generators
- **Word embeddings**: Word2Vec-style, learned lookup tables
- **Positional embeddings**: Sinusoidal, learned, rotary
- **Category embeddings**: Entity embeddings for high-cardinality features
- **Graph embeddings**: Node2Vec-style random walk embeddings
- **Reality**: Well-documented algorithms agents can implement

### Feature Transformers
- **Feature crosses**: Polynomial features, interaction terms
- **Feature projections**: PCA, random projections, autoencoders
- **Feature selections**: Variance threshold, mutual information
- **Feature scalers**: MinMax, Standard, Robust, Quantile
- **Reality**: Scikit-learn style transformers, very reliable

### Sequence Processors
- **Sliding windows**: Fixed/variable length windowing
- **Sequence padders**: Pre/post padding, truncation
- **Sequence samplers**: Fixed-length sampling, importance sampling
- **Sequence augmenters**: Reverse, shuffle, crop, noise injection
- **Reality**: Array manipulation that agents handle well

### Loss Functions & Metrics
- **Classification losses**: Cross-entropy, focal, hinge
- **Regression losses**: MSE, MAE, Huber, quantile
- **Ranking losses**: Triplet, contrastive, margin ranking
- **Custom metrics**: Domain-specific evaluation functions
- **Reality**: Mathematical formulas agents implement correctly

### Data Augmenters
- **Text augmenters**: Synonym swap, random insertion/deletion
- **Tabular augmenters**: SMOTE, random noise, mixup
- **Time series augmenters**: Time warping, magnitude warping
- **Reality**: Standard augmentation recipes work well

### Activation Functions
- **Standard activations**: ReLU, GELU, SiLU, Tanh, Sigmoid
- **Parametric activations**: LeakyReLU, PReLU, ELU
- **Custom activations**: Swish, Mish, composed functions
- **Reality**: One-line mathematical functions

## Tier 2: Moderate Complexity (Agents Can Do With Guidance)

### Model Training Loops
- **Basic trainers**: SGD loops with logging
- **Checkpoint managers**: Save/load model states
- **Early stopping**: Monitor validation metrics
- **Reality**: Need clear specifications but agents can implement

### Data Augmentation
- **Text augmentation**: Synonym replacement, back-translation
- **Tabular augmentation**: SMOTE, noise injection
- **Reality**: Standard techniques work, novel augmentation doesn't

### Evaluation Frameworks
- **A/B test harnesses**: Statistical significance testing
- **Model comparison tools**: Side-by-side evaluation
- **Reality**: Following established patterns works well

### Prompt Engineering Tools
- **Template managers**: Variable substitution, formatting
- **Few-shot selectors**: Similarity-based example selection
- **Reality**: String manipulation + embeddings, agents handle well

## Tier 3: Pushing It (Often Fails or Needs Heavy Iteration)

### Multi-Agent Orchestration
- **Agent routers**: Dispatch to specialized agents
- **Consensus mechanisms**: Voting, aggregation
- **Reality**: Coordination complexity leads to bugs

### Advanced Optimizers
- **Custom learning rate schedules**: Cyclical, warmup
- **Gradient clipping/accumulation**: Memory-efficient training
- **Reality**: Numerical stability issues common

### Model Interpretability
- **SHAP/LIME wrappers**: Feature importance
- **Attention visualizers**: Heatmap generation
- **Reality**: Library integration often brittle

## Tier 4: Don't Even Try (Agents Will Fail)

### Novel Architectures
- **New transformer variants**: Custom attention mechanisms
- **Reality**: Requires research intuition agents lack

### Advanced RL Algorithms
- **PPO/A3C from scratch**: Policy gradient methods
- **Reality**: Too many subtle bugs, hyperparameter sensitivity

### Distributed Training
- **Model parallelism**: Splitting models across GPUs
- **Reality**: Hardware-specific optimizations needed

### Custom CUDA Kernels
- **Optimized operations**: Memory-efficient attention
- **Reality**: Low-level optimization beyond agent capability

## Composition Patterns That Work

### Sequential Pipelines
```
Data Loader -> Preprocessor -> Feature Extractor -> Model -> Evaluator
```
**Reality**: Each component has clear interfaces, agents handle well

### Ensemble Patterns
```
Multiple Models -> Voting/Averaging -> Final Prediction
```
**Reality**: Simple aggregation logic, very reliable

### API Orchestration
```
User Input -> Router -> Specialized Agent -> Response Formatter
```
**Reality**: HTTP/REST patterns agents understand

## Anti-Patterns to Avoid

1. **Stateful Components**: Agents struggle with complex state management
2. **Tight Coupling**: Each component should be independently testable
3. **Novel Math**: Stick to implemented algorithms in libraries
4. **Performance-Critical Code**: Agents write correct but not optimal code
5. **Complex Async/Concurrent**: Race conditions everywhere

## Recommended Focus Areas

1. **Start with Tier 1**: Build confidence and establish patterns
2. **API Integration**: Agents excel at reading docs and implementing clients
3. **Data Pipeline**: Boring but essential, agents do it well
4. **Evaluation Harnesses**: High value, well-defined requirements
5. **Prompt Engineering Tools**: Natural fit for LLM-based agents

## Next Steps

For each component type, we should:
1. Define exact interfaces (input/output types)
2. Create test harnesses with sample data
3. Build reference implementations
4. Measure agent success rate in regenerating them