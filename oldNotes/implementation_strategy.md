# Implementation Strategy: Two-Phase Testing

## Phase 1: Adaptive Tests First (Week 1)
Start here because it:
1. Gives immediate feedback on what's "noise" vs real issues
2. Helps identify which guidelines matter most
3. Creates a more robust test suite overall
4. Benefits ALL testing approaches

### Implementation:
```python
# Add test_utils.py with adaptive helpers
from test_utils import (
    try_init_variants,      # Handle n_heads vs num_heads
    extract_output,         # Handle dict vs tensor returns  
    import_with_fallback,   # Handle missing optional exports
)
```

## Phase 2: Sibling-Aware Harness (Week 2)
Then add context-aware testing:
1. Use adaptive tests to get baseline success rates
2. Test progressive improvement with context
3. Measure which context helps most

### Testing Matrix:

| Approach | Context Provided | Expected Success |
|----------|-----------------|------------------|
| Blind (current) | None | 0% → 40% (with adaptive) |
| Exemplar | 3 example modules | 40% → 60% |
| Family | Related modules | 40% → 70% |
| Progressive | Growing context | 40% → 75% |
| Leave-one-out | All but target | 40% → 85% |

## Guidelines Strategy: **Relax and Refocus**

### Keep (High Value):
- Core principles (device management, shape consistency)
- Common pitfalls (state management, mask handling)
- Parameterization philosophy

### Add (Based on Test Failures):
- Parameter naming conventions (n_heads vs num_heads)
- Module completeness patterns (which classes to export)
- Return format expectations

### Remove (Low Value/Overspecification):
- Specific implementation details
- Overly prescriptive patterns
- Complex domain-specific rules

## Revised Guidelines Structure:

```markdown
# Agent Guidelines v2

## Critical Patterns (MUST follow)
1. Parameter naming conventions [NEW]
2. Module completeness checklist [NEW]  
3. Device consistency
4. Shape management

## Common Patterns (SHOULD follow)
1. Return formats by module type
2. Import conventions
3. Error handling

## Domain Patterns (REFERENCE only)
1. Audio processing tips
2. Streaming considerations
3. Vision conventions
```

## Decision Framework:

### Why Both?
- **Adaptive tests** = Measure true functional equivalence
- **Sibling context** = Test realistic development scenarios
- Together they answer: "What makes a pattern truly reproducible?"

### Why This Order?
1. Adaptive tests first reveals which failures are "real" vs "cosmetic"
2. Then sibling-aware testing shows how much context actually helps
3. Finally, refine guidelines based on what actually matters

## Success Metrics:

### Phase 1 Success:
- [ ] 40-50% pass rate with adaptive tests
- [ ] Clear categorization of "real" vs "style" failures
- [ ] Test suite that works for diverse implementations

### Phase 2 Success:  
- [ ] Demonstrate 70%+ success with family context
- [ ] Identify minimum context for good reproduction
- [ ] Validate which patterns truly propagate

### Final Success:
- [ ] Guidelines that focus on what matters
- [ ] Test suite that validates function over form
- [ ] Proof that "reliable patterns" exist at right abstraction level

## Next Step:
Start with adaptive test utilities that can be reused across all test files?