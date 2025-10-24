# Sibling-Aware Module Testing Strategy

## Approach 1: Progressive Generation with Context

Generate modules in a specific order, providing previously generated modules as context:

```python
generation_order = [
    # Start with foundational modules
    'transformer_block',          # Establishes n_heads vs num_heads
    'conv_encoder',              # Establishes encoder return format
    
    # Generate similar modules with context
    'sequence_encoder',          # Will see transformer_block's patterns
    'attention_decoder',         # Will see both above
    'vit_patch_encoder',        # Will see conv_encoder's patterns
    
    # Data pipeline family
    'data_validator',           # Establishes data module patterns  
    'data_versioner',          # Sees validator patterns
    'stream_processor',        # Sees data handling patterns
    
    # Audio family
    'snake_activation',        # Establishes audio patterns
    'causal_conv',            # Sees activation patterns
    'antialiased_conv',       # Sees conv patterns
]
```

## Approach 2: Family-Based Generation

Group modules by family and generate each family together:

### Transformer Family
- Provide all at once: transformer_block, sequence_encoder, attention_decoder
- They'll naturally align on n_heads, d_model conventions

### Data Pipeline Family  
- Provide all at once: data_validator, data_versioner, feature_store, stream_processor
- They'll align on timestamp handling, method patterns

### Audio Family
- Provide all at once: snake_activation, causal_conv, stft_loss
- They'll align on signal processing conventions

## Approach 3: Exemplar-Based Generation

Provide 2-3 completed modules as "style exemplars":

```python
exemplars = {
    'parameter_style': 'transformer_block.py',  # Shows n_heads convention
    'return_style': 'conv_encoder.py',          # Shows dict return pattern  
    'data_style': 'data_validator.py',         # Shows class structure
}
```

## Testing Protocol:

### Option 1: Progressive Context Testing
```bash
# Test 1: Generate all modules blind (current approach)
python test_candidates.py candidates/blind_agent

# Test 2: Generate with 3 exemplar modules
python test_candidates.py candidates/exemplar_agent

# Test 3: Generate with family groupings  
python test_candidates.py candidates/family_agent

# Test 4: Generate progressively with all context
python test_candidates.py candidates/progressive_agent
```

### Option 2: Leave-One-Out Testing
For each module:
1. Provide all OTHER modules as context
2. Generate just that module
3. Test if it matches the family style

```python
for module in all_modules:
    context = all_modules - {module}
    generated = agent.generate(module, context=context)
    success = test(generated)
```

## Expected Success Rate Improvements:

| Approach | Expected Success | Why |
|----------|-----------------|-----|
| Blind (current) | 0% | No style reference |
| 3 Exemplars | 30-40% | Basic patterns established |
| Family Groups | 50-60% | Strong family conventions |
| Progressive | 60-70% | Cumulative pattern learning |
| Leave-One-Out | 80-90% | Maximum context |

## Implementation Considerations:

### What to Share:
- ✅ Sibling module implementations
- ✅ Common base classes/utilities
- ❌ Test files (keep hidden)
- ❌ Results from other modules

### Context Window Management:
- Could provide just module signatures/headers if full code too large
- Or provide 2-3 most relevant siblings based on module type

### Evaluation Metrics:
1. **Style Consistency**: Do all modules use same parameter names?
2. **Pattern Propagation**: Do similar modules follow same patterns?
3. **Family Coherence**: Do related modules work well together?

## Recommended Experiment:

1. Start with Family-Based approach (most realistic)
2. Compare against current blind approach
3. Measure both test success AND style consistency
4. Document which patterns successfully propagate

This would test a more realistic scenario where teams/agents typically have access to existing code style and can maintain consistency.