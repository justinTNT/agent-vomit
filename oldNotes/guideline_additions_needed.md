# Guideline Additions Needed Based on Failure Analysis

## 1. Module Composition Patterns

Add to guidelines_v3.md:

```python
# When modules require other modules as components:
class ContrastiveLearner(nn.Module):
    def __init__(self, 
                 encoder: nn.Module,          # Pass actual module, not dimensions
                 projection_dim: int = 128,
                 temperature: float = 0.07,
                 **kwargs):
        """
        Args:
            encoder: A torch.nn.Module that maps input -> features
            projection_dim: Dimension of projection head
            temperature: Temperature for InfoNCE loss
        """

class SequenceToSequenceModel(nn.Module):
    def __init__(self,
                 encoder: nn.Module,          # Actual encoder module
                 decoder: nn.Module,          # Actual decoder module
                 **kwargs):
        """Use module composition when the module orchestrates other modules."""
```

## 2. Parameter Structure Specifications

```python
# For multi-modal inputs, use dict:
class CrossModalFusion(nn.Module):
    def __init__(self,
                 modality_dims: Dict[str, int],  # {'visual': 512, 'text': 768}
                 d_hidden: int = 1024,
                 fusion_type: str = 'concat',
                 **kwargs):
```

## 3. Forward Method Signatures

Add explicit forward specifications for each module:

```python
# ContrastiveLearner
def forward(self, features: Tensor) -> Tensor:
    """
    Args:
        features: Shape [batch_size, feature_dim]
    Returns:
        loss: Scalar tensor with InfoNCE loss
    """

# CrossModalFusion  
def forward(self, modality_inputs: Dict[str, Tensor]) -> Tensor:
    """
    Args:
        modality_inputs: {'visual': [B, T1, D1], 'text': [B, T2, D2]}
    Returns:
        fused: [B, T, D_hidden]
    """

# MemoryBank
def store(self, keys: Tensor, values: Tensor) -> None:
    """Store key-value pairs"""
    
def retrieve(self, query: Tensor, k: int = 1) -> Tuple[Tensor, Tensor]:
    """Retrieve k nearest neighbors"""
```

## 4. Exact Parameter Names

Update the parameter name table to include:

| Module | Parameter | Not |
|--------|-----------|-----|
| ResidualVectorQuantizer | dim | ~~embedding_dim~~ |
| ResidualVectorQuantizer | codebook_size | ~~n_embeddings~~ |
| ResidualVectorQuantizer | num_quantizers | ~~n_quantizers~~ |

## 5. Class Naming Conventions

Specify exact class names to avoid confusion:

```
antialiased_conv.py -> class AntialiasedConv  # Not AntiAliasedConv
```

## 6. Tensor Dimensionality

Specify expected input dimensions:

```python
# Time Series: [batch, channels, time]
# Images: [batch, channels, height, width] 
# Sequences: [batch, sequence_length, features]
# Sets: [batch, set_size, features]
```

## Summary

With these guideline additions, we could prevent approximately 60% of the remaining failures. The other 40% are either valid implementation differences or require design decisions beyond what guidelines can specify.