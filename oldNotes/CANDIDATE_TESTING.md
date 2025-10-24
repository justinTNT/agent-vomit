# Candidate Testing Guide

## Overview
This guide explains how to test candidate implementations of our 25 modules from external agents.

## Directory Structure

```
agent-vomit/
├── modules/              # Our reference implementations
├── tests/                # Test suite (don't share with candidates!)
├── candidates/           # Candidate implementations go here
│   ├── agent_alpha/      # Example candidate directory
│   │   ├── transformer_block.py
│   │   ├── conv_encoder.py
│   │   └── ... (all 25 modules)
│   └── agent_beta/       # Another candidate
├── test_results/         # Test results stored here
├── test_candidates.py    # Test harness
├── reproduction_plan.md  # Give this to candidates
└── guidelines.md         # Give this to candidates
```

## Setup Instructions

1. **Create candidates directory**:
   ```bash
   mkdir -p candidates
   ```

2. **For each candidate agent**:
   - Create a subdirectory: `mkdir candidates/agent_name`
   - Have them place their implementations there
   - Each module should be named exactly as specified (e.g., `transformer_block.py`)

## What to Give Candidates

Provide ONLY these files:
- `reproduction_plan.md` - Module specifications
- `guidelines.md` - Implementation guidelines

Do NOT share:
- Any files from `modules/` directory
- Any files from `tests/` directory
- This testing guide
- The test harness

## Running Tests

### Test all modules from a candidate:
```bash
python test_candidates.py candidates/agent_alpha
```

### Test with verbose output:
```bash
python test_candidates.py candidates/agent_alpha --verbose
```

### Test a specific module:
```bash
python test_candidates.py candidates/agent_alpha --module transformer_block
```

### Available module names for testing:
- ML Components: `transformer_block`, `conv_encoder`, `sequence_encoder`, `attention_decoder`, `vit_patch_encoder`, `cross_modal`, `timeseries_encoder`, `set_encoder`, `contrastive`, `autoencoder`, `seq2seq`, `graph_encoder`, `memory_retriever`, `adaptive_computation`
- Data Pipeline: `stream_processor`, `data_validator`, `feature_store`, `data_versioner`, `stream_joiner`, `data_sampler`
- Audio: `snake_activation`, `causal_conv`, `stft_loss`, `antialiased_conv`, `residual_vector_quantizer`

## Test Results

Results are automatically saved to `test_results/` with timestamps:
- `test_results/agent_alpha_20231012_143022.json`

Each result file contains:
- Module-by-module test results
- Error messages and stack traces
- Summary statistics
- Success rate calculation

## Interpreting Results

### Status Codes:
- ✅ `passed` - All tests passed
- ❌ `failed` - Tests ran but failed
- 📭 `missing` - Module file not found
- 🚨 `error` - Test couldn't run (syntax error, import error, etc.)

### Success Metrics:
- **First-pass success rate**: % of modules that pass without modification
- **Failure patterns**: Common issues across failed modules
- **Ambiguity handling**: Whether candidates parameterized the same ambiguities

## Example Test Session

```bash
# 1. Agent implements modules in candidates/agent_gamma/
# 2. Run full test suite
$ python test_candidates.py candidates/agent_gamma

============================================================
Testing transformer_block
============================================================
Running tests/test_transformer_block.py...
✅ PASSED: transformer_block

============================================================
Testing conv_encoder
============================================================
Running tests/test_conv_encoder.py...
❌ FAILED: conv_encoder

[... continues for all 25 modules ...]

============================================================
TEST SUMMARY
============================================================
Candidate: candidates/agent_gamma
Total modules: 25
✅ Passed: 20
❌ Failed: 3
📭 Missing: 1
🚨 Errors: 1

Success rate: 80.0%

Failed modules:
  - conv_encoder
  - data_versioner
  - stream_joiner
```

## Validation Protocol

1. **Blind Implementation**: Candidates should not see our implementations or tests
2. **No Iteration**: Test results represent first-attempt success
3. **Clean Environment**: Each test runs in isolation
4. **Automatic Rollback**: Original modules are preserved

## Analyzing Patterns

After testing multiple candidates, look for:
1. Which modules consistently pass (truly reliable patterns)
2. Which modules consistently fail (ambiguous specifications)
3. Whether candidates make similar parameterization choices
4. If failure patterns match our original development issues

## Tips for Candidates

If providing feedback after testing:
1. Point them to specific guideline sections for common issues
2. Clarify ambiguous specifications in reproduction_plan.md
3. Note which parameterization choices would help
4. Don't share actual implementation details

This testing framework provides rigorous validation of our claim that these modules are "reliably reproducible" patterns!