# Path to 0% Failures: Guidelines as Config + Generalized Tests

## The Architecture

```
GUIDELINES (Explicit Decisions)          TESTS (Flexible Validators)
┌─────────────────────────────┐         ┌──────────────────────────┐
│ • Parameter names           │         │ • Behavioral validation  │
│ • Return formats            │  ──────>│ • Shape checking        │
│ • Method signatures         │         │ • Property testing      │
│ • Default values            │         │ • Capability detection  │
└─────────────────────────────┘         └──────────────────────────┘
         ▲                                        │
         │                                        │
         │      AGENT IMPLEMENTATION              │
         │      "I follow guidelines"             │
         └────────────────────────────────────────┘
```

## Why This Achieves 0% Failures

### 1. No Ambiguity
```python
# Guidelines say EXACTLY:
'n_heads': int,  # NOT num_heads, NOT heads

# Test validates flexibly:
assert hasattr(module, 'forward')  # Has forward method
assert can_handle_multihead_attention(module)  # Works correctly
```

### 2. Single Source of Truth
```python
# Change needed? Update guidelines only:
# guidelines.yaml
transformer_block:
  parameters:
    n_heads: int  → num_heads: int  # One change

# Tests still work - they test behavior not names
```

### 3. Complete Specification Coverage
```yaml
# Every decision is documented:
stream_joiner:
  forward_signature: |
    # IMPORTANT: timestamp is positional not kwarg!
    def forward(self, stream_id: str, data: Tensor, timestamp: float):

# No more "oh, timestamp should have been positional!"
```

## Example: Zero-Failure Test Suite

### Guidelines Entry
```yaml
transformer_block:
  parameters:
    d_model: int          # required
    n_heads: int          # required  
    d_ff: int            # required
    dropout: float = 0.1  # optional with default
    
  return_format: Tensor  # same shape as input
  
  behaviors:
    - preserves sequence length
    - supports attention masking
    - differentiable
    
  implementation_notes: |
    - Use pre-norm (norm_first=True)
    - Mask is additive not boolean
    - No need to return attention weights
```

### Generalized Test
```python
def test_transformer_block():
    # Load spec from guidelines
    spec = load_guidelines()['transformer_block']
    
    # Initialize with exact param names from guidelines
    module = TransformerBlock(**get_required_params(spec))
    
    # Behavioral tests (flexible)
    assert preserves_shape(module, sample_input)
    assert supports_masking(module)
    assert is_differentiable(module)
    
    # Don't test internals, just behavior
    # ✓ Works with any valid implementation
```

## Migration Plan

### Phase 1: Extract All Decisions (~2 days)
Go through current tests and extract every implicit decision:
- Parameter names
- Return formats  
- Method signatures
- Default values
- Behavioral expectations

### Phase 2: Create Comprehensive Guidelines (~1 day)
Structure all decisions into guidelines:
```yaml
# Complete for all 25 modules
module_name:
  parameters: ...
  return_format: ...
  required_methods: ...
  behaviors: ...
  notes: ...
```

### Phase 3: Generalize Tests (~3 days)
Rewrite tests to:
- Read specs from guidelines
- Test behaviors not implementations
- Handle variations gracefully
- Focus on capabilities

## What "Failure" Means Now

With this approach, failures only occur when:
1. **Agent ignores guidelines** → Their fault, not ambiguity
2. **Behavior is incorrect** → Real bug, not API mismatch
3. **Guidelines incomplete** → Our oversight, easily fixed

## Example Success Metrics

```
Before (Blind):
- 0% pass (wrong param names, return formats, etc.)

After Guidelines + Generalized Tests:
- 95%+ pass (only behavioral bugs remain)
- 100% pass (after fixing true behavioral issues)
```

## Key Insight

This approach separates:
- **Configuration** (guidelines): "Use n_heads not num_heads"  
- **Validation** (tests): "Does it do multi-head attention?"

When these are conflated (tests check both), we get brittleness.
When separated, we get flexibility with precision.

## Next Steps

1. Create guideline template covering all decision points
2. Extract decisions from existing tests  
3. Write generalized test utilities
4. Migrate tests to behavioral validation
5. Test with both existing agents to verify 90%+ improvement