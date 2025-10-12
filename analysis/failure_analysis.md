# Failure Analysis: DataVersioner and StreamJoiner

## Summary

Both DataVersioner and StreamJoiner represent a specific class of failure where agents struggle with modules requiring:
1. Complex temporal/sequential dependencies
2. Multi-entity state coordination
3. Implicit ordering requirements
4. Reasoning about sequences of operations

## DataVersioner Failures

### Test: `test_merge_strategies`
**Issue**: The test expects that after merging branch1 into main, a subsequent merge from branch2 will average main's current value (2.0) with branch2's value (3.0) to get 2.5. However, the implementation doesn't track what value main currently holds.

**Root cause**: The module doesn't distinguish between "current branch state" and "branch head version". The merge operation updates the branch head but doesn't maintain a clear model of what data the branch represents.

### Test: `test_version_cleanup`
**Issue**: When old versions are cleaned up, the module removes base versions that delta versions depend on, making reconstruction impossible.

**Root cause**: The cleanup logic doesn't understand the dependency graph between full and delta versions. It treats all versions as independent when they're actually linked.

### Test: `test_duplicate_data`
**Issue**: The test expects identical data to return the same version ID, but the implementation includes timestamp in the hash.

**Root cause**: Conflicting requirements between deduplication (content-based addressing) and version tracking (time-based uniqueness).

## StreamJoiner Failures

### Test: `test_asof_join`
**Issue**: ASOF join expects to match historical data from stream2 with new data from stream1, but the implementation requires stream1 to be the "reference stream" with existing data.

**Root cause**: Ambiguity about which stream drives the join and when joining occurs. The concept of "reference stream" isn't clearly established.

### Test: `test_interpolation`
**Issue**: Linear interpolation between time points returns the wrong value, likely due to how buffers are searched or how time windows are calculated.

**Root cause**: The relationship between buffer ordering, time windows, and interpolation points is complex and the implementation makes different assumptions than the test.

## Common Pattern

Both modules fail when they need to:
- Maintain coherent state across multiple operations
- Handle implicit relationships between entities (versions depend on each other, streams have primary/secondary roles)
- Reason about temporal ordering beyond simple timestamps

## Implications

1. **These are not just bugs** - they represent fundamental design ambiguities that the agent couldn't resolve

2. **The complexity is in the semantics**, not the implementation - both modules have reasonable code that makes internally consistent choices

3. **Tests revealed unstated requirements** - the expected behavior wasn't fully specified upfront

## Recommendation

These modules should be marked as **experimental/unreliable** rather than trying to fix them, because:

1. The fixes would require significant redesign, not just bug fixes
2. The agent's initial approach was reasonable given the specifications
3. These represent a genuine boundary of agent capabilities

## Lessons Learned

When asking agents to generate modules, avoid designs that require:
- Multiple coordinated stateful entities
- Implicit temporal relationships
- Operations whose correctness depends on prior operation sequences
- Ambiguous "primary/secondary" relationships

Instead, prefer:
- Single-entity state management
- Explicit relationships
- Operations that are correct regardless of history
- Clear ownership and roles