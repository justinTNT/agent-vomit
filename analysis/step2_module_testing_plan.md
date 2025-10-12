# Step 2: Module Generation Testing Plan

## Modules to Test (Priority Order)

### 1. TransformerBlock
**Why first**: Most common, well-documented, exercises many sub-components
**Tests**:
- Multi-head attention mechanism
- FFN with proper expansion
- Residual connections  
- Layer normalization
- Correct tensor shapes throughout

### 2. ConvEncoder
**Why**: Different paradigm from transformers, tests spatial understanding
**Tests**:
- Progressive downsampling
- Channel expansion
- Pooling strategies
- Skip connections (if ResNet-style)

### 3. SequenceEncoder
**Why**: Full end-to-end encoding pipeline
**Tests**:
- Embedding lookup
- Positional encoding
- Multiple transformer blocks
- Output aggregation

### 4. AttentionDecoder  
**Why**: Tests cross-attention and generation
**Tests**:
- Self-attention
- Cross-attention to encoder
- Causal masking
- Output projection

### 5. ViTPatchEncoder
**Why**: Tests image-specific logic
**Tests**:
- Patch extraction
- Positional embeddings for 2D
- CLS token handling

## Test Methodology

### For Each Module:
1. **Prompt agent to generate module with specific requirements**:
   - Input/output dimensions
   - Number of layers/heads
   - Specific framework (PyTorch/TensorFlow)

2. **Verify implementation**:
   - Runs without errors
   - Correct output shapes
   - Reasonable parameter count
   - No obvious bugs

3. **Test composability**:
   - Can stack multiple modules
   - Works in simple training loop
   - Gradients flow properly

### Success Criteria
- ✅ **Level 1**: Code runs, shapes are correct
- ✅ **Level 2**: Can be trained (gradients flow)
- ✅ **Level 3**: Matches reference implementation behavior

### Budget Per Module
- 2-3 generation attempts max
- If it fails consistently, mark as "needs human help"
- Document failure modes for learning

## Quick Test Script Template
```python
# For each generated module
module = GeneratedModule(config)
x = torch.randn(batch_size, seq_len, d_model)
out = module(x)
assert out.shape == expected_shape
assert not torch.isnan(out).any()

# Gradient check
loss = out.sum()
loss.backward()
assert all(p.grad is not None for p in module.parameters())
```

## Expected Outcomes

### High Confidence (90%+ success)
- TransformerBlock (extremely well documented)
- ConvEncoder (standard CNN patterns)

### Medium Confidence (70% success)
- SequenceEncoder (integration complexity)
- ViTPatchEncoder (2D position encoding edge cases)

### Lower Confidence (50% success)  
- AttentionDecoder (causal masking complexity)

## What We're Really Testing

1. **Can agents reliably generate ~200-500 line modules?**
2. **Do they correctly compose sub-components?**
3. **Do they handle edge cases (padding, masking)?**
4. **Is the generated code maintainable/readable?**

This will tell us if the "agent-generated components" vision is realistic.