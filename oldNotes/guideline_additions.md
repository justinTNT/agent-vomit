# Additional Guidelines Based on Test Failures

## Parameter Naming Conventions

When implementing modules, use these EXACT parameter names to match test expectations:

### Transformer/Attention Modules
- `n_heads` (NOT num_heads, n_head, heads)
- `d_model` (NOT dim_model, model_dim, hidden_size) 
- `d_ff` (NOT ff_dim, feedforward_dim, mlp_dim)
- `n_layers` (NOT num_layers, layers)

### Vision Modules
- `img_size` (NOT image_size, input_size)
- `patch_size` (NOT patch_dim)
- `in_chans` (NOT in_channels, input_channels)

### Data Pipeline Modules
- `batch_size` (NOT bs, batch)
- `time_window` (NOT window_size, window)

## Required Imports

ALWAYS include these standard imports at the top of every module:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union, Any
```

## Module Completeness Patterns

### Multi-Component Modules
If the module name suggests multiple components, implement ALL of them:

- **snake_activation**: Must export `SnakeActivation`, `SnakeBeta`, and `snake` function
- **causal_conv**: Must export `CausalConv1d` and `CausalConvTranspose1d` 
- **data_validator**: Must export `DataValidator` and `Schema` classes
- **feature_store**: Must export `FeatureStore` and `FeatureCompute` classes
- **adaptive_computation**: Must export `AdaptiveComputationTime`, `PonderNet`, and `UniversalTransformer`
- **residual_vector_quantizer**: Must export both `VectorQuantizer` and `ResidualVectorQuantizer`

### Return Format Patterns

Check if module should return dictionary vs tensor:

```python
# Encoder modules often return dictionaries:
class ConvEncoder(nn.Module):
    def forward(self, x):
        features = self.encoder(x)
        pooled = self.pool(features)
        return {'features': features, 'pooled': pooled}  # NOT just features
```

## API Consistency Patterns

### Method vs Callable
Some modules expect callable interface, others expect methods:

```python
# DataVersioner: Callable interface expected
class DataVersioner(nn.Module):
    def __call__(self, data, message):
        return self.create_version(data, message)
```

### Timestamp Handling
For streaming modules, timestamp might be:
- A keyword argument: `forward(self, stream_id, data, timestamp=None)`
- A method parameter: `add_data(self, stream_id, data, timestamp)`

## Common Parameter Defaults

Use these specific default values when not otherwise specified:

```python
# Transformer defaults
dropout=0.1 (not 0.0)
activation='gelu' (not 'relu' for transformers)
norm_first=True (for modern transformers)

# Vision defaults  
patch_size=16
num_classes=1000

# Streaming defaults
buffer_size=1000
time_window=1.0
```

## Testing Approach

Before finalizing, ask yourself:
1. Have I exported ALL classes/functions the test might import?
2. Do my parameter names match common PyTorch conventions?
3. Does my return format match what an encoder/decoder would typically return?
4. Have I included all necessary type imports?