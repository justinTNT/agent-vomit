# Balanced Approach: Essential Specs + Flexible Tests

## Philosophy
- **Specify only what matters** - critical decisions that affect interoperability
- **Test flexibly** - accept common variations that don't affect functionality
- **Document clearly** - but don't over-specify

---

## Essential Implementation Decisions

### 1. Core Parameter Names (Pick One Convention)
```python
# These variations are ALL acceptable - tests will try all:
PARAMETER_VARIATIONS = {
    'attention_heads': ['n_heads', 'num_heads', 'heads'],
    'model_dimension': ['d_model', 'hidden_dim', 'embed_dim'],
    'feedforward': ['d_ff', 'ff_dim', 'mlp_dim'],
    'image_size': ['img_size', 'image_size', 'input_size'],
    'channels': ['in_channels', 'in_chans', 'input_channels'],
}
```

### 2. Return Format Flexibility
```python
# Tests will handle EITHER pattern:

# Pattern A: Dictionary returns (preferred for multi-output)
return {'features': x, 'pooled': pooled}

# Pattern B: Tensor returns (acceptable for single output)
return x

# Pattern C: Tuple returns (acceptable for 2-3 outputs)
return x, pooled
```

### 3. Critical Decisions That MUST Be Specified

#### TransformerBlock Family
- **Mask format**: Additive (not boolean) - this affects correctness
- **Positional encoding**: Required for ViT, SequenceEncoder

#### Data Pipeline
- **Timestamp handling**: 
  - StreamJoiner: needs timestamp (positional or kwarg OK)
  - StreamProcessor: optional timestamp
- **Callable vs Method**:
  - DataVersioner: prefer callable but accept .create_version()

#### Audio Components  
- **Causality**: CausalConv must be strictly causal
- **Multi-scale**: STFT loss needs multiple scales

---

## Relaxed Testing Strategy

### 1. Parameter Name Adaptation
```python
def init_with_variations(ModuleClass, params):
    """Try common parameter variations."""
    # Try original params
    try:
        return ModuleClass(**params)
    except TypeError as e:
        # Try variations
        for canonical, variations in PARAMETER_VARIATIONS.items():
            if canonical in str(e):
                for variant in variations:
                    try:
                        alt_params = params.copy()
                        alt_params[variant] = alt_params.pop(canonical)
                        return ModuleClass(**alt_params)
                    except:
                        continue
        raise e
```

### 2. Return Format Adaptation
```python
def extract_output(output, expected_type='auto'):
    """Handle tensor/dict/tuple returns flexibly."""
    if isinstance(output, dict):
        return output
    elif isinstance(output, tuple):
        # Map tuple to expected dict keys
        if len(output) == 2:
            return {'output': output[0], 'extra': output[1]}
        return {'output': output[0]}
    else:
        # Single tensor
        return {'output': output}
```

### 3. Method Name Flexibility
```python
# Tests try multiple common patterns:
METHOD_VARIATIONS = {
    'load_data': ['load', 'get', 'retrieve', 'fetch'],
    'save_data': ['save', 'store', 'put', 'write'],
    'create': ['create', 'make', 'new', 'build'],
}
```

---

## Simplified Guidelines (What Actually Matters)

### ML Components

#### Transformer-based (TransformerBlock, SequenceEncoder, etc.)
```yaml
critical:
  - Must support attention masking
  - Mask format is additive (negative inf for masked positions)
  - Must preserve sequence length
  
flexible:
  - Parameter names (n_heads vs num_heads)
  - Return dict vs tensor
  - Internal architecture details
```

#### Vision (ViTPatchEncoder, ConvEncoder)
```yaml
critical:
  - Spatial downsampling for ConvEncoder
  - Patch extraction for ViT
  - Include positional embeddings
  
flexible:
  - Exact parameter names
  - Whether to return dict or tuple
  - Pooling strategy
```

### Data Pipeline

#### StreamJoiner
```yaml
critical:
  - Must handle time-based joining
  - Needs timestamp information somehow
  - Buffer management required
  
flexible:
  - Whether timestamp is positional or kwarg
  - Buffer size defaults
  - Return format when no join occurs
```

#### DataVersioner  
```yaml
critical:
  - Must store and retrieve data
  - Version identification (some string ID)
  - Support deduplication option
  
flexible:
  - Callable vs method interface
  - Storage backend details
  - Version ID format
```

### Audio Components

#### CausalConv
```yaml
critical:
  - MUST be strictly causal (no future lookahead)
  - Correct padding for causality
  
flexible:
  - Class naming details
  - Parameter defaults
  - Implementation approach
```

#### Snake Activation
```yaml
critical:
  - Smooth, differentiable activation
  - Roughly periodic behavior
  
flexible:
  - Whether to export functional interface
  - Exact mathematical formulation
  - Parameter initialization
```

---

## Test Adaptation Examples

### Before (Rigid)
```python
def test_transformer_block():
    # Fails if not exactly n_heads
    model = TransformerBlock(d_model=512, n_heads=8)
    output = model(x)
    assert isinstance(output, torch.Tensor)  # Fails if dict
```

### After (Flexible)
```python
def test_transformer_block():
    # Try common parameter variations
    model = init_with_variations(TransformerBlock, {
        'd_model': 512, 
        'n_heads': 8
    })
    
    # Accept multiple return formats
    output = model(x)
    result = extract_output(output)
    
    # Test behavior, not format
    assert result['output'].shape[-1] == 512  # Preserves dim
    assert can_backprop(model, x)  # Differentiable
    
    # Test critical behavior: masking
    mask = create_attention_mask(x)
    masked_out = model(x, mask=mask)
    assert not torch.allclose(output, masked_out)  # Mask has effect
```

---

## Summary: The 80/20 Rule

### Specify (20% - Critical Decisions)
1. Behavioral requirements (causality, masking format)
2. Semantic expectations (what the module should do)
3. Critical interoperability needs

### Be Flexible About (80% - Implementation Details)
1. Parameter naming conventions
2. Return formats (dict/tensor/tuple)
3. Method names
4. Internal architecture
5. Default values

This balanced approach should:
- Reduce specification document by ~70%
- Allow ~60-80% test success with blind implementation
- Maintain clear contracts for critical behaviors
- Make tests resilient to reasonable variations