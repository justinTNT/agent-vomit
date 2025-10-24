#!/usr/bin/env python3
"""Debug snake activation parameter issue."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
from test_utils import init_with_variations, PARAMETER_VARIATIONS

# Test the original module
print("Testing original snake_activation module:")
from modules.snake_activation import SnakeActivation

# Try direct initialization
try:
    model1 = SnakeActivation(n_channels=64)
    print(f"✅ Direct init with n_channels=64 works")
except Exception as e:
    print(f"❌ Direct init failed: {e}")

# Try with variations
try:
    model2 = init_with_variations(SnakeActivation, {'channels': 64, 'alpha': 1.0}, verbose=True)
    print(f"✅ Flexible init succeeded")
    print(f"   Model has {model2.n_channels} channels")
except Exception as e:
    print(f"❌ Flexible init failed: {e}")

# Now test agent_claude version
print("\nTesting agent_claude snake_activation:")
import shutil
backup = Path('modules/snake_activation.py.backup')
original = Path('modules/snake_activation.py')
candidate = Path('candidates/agent_claude/snake_activation.py')

if original.exists():
    shutil.copy(original, backup)
shutil.copy(candidate, original)

# Reload module
if 'modules.snake_activation' in sys.modules:
    del sys.modules['modules.snake_activation']

from modules.snake_activation import SnakeActivation as ClaudeSnake

# Try with variations
try:
    model3 = init_with_variations(ClaudeSnake, {'channels': 64, 'alpha': 1.0}, verbose=True)
    print(f"✅ agent_claude flexible init succeeded")
    print(f"   Model has {model3.n_channels} channels")
    
    # Test forward pass
    x = torch.randn(2, 64, 100)
    y = model3(x)
    print(f"✅ Forward pass succeeded: {x.shape} -> {y.shape}")
except Exception as e:
    print(f"❌ agent_claude init/forward failed: {e}")
    import traceback
    traceback.print_exc()

# Restore
if backup.exists():
    shutil.copy(backup, original)
    backup.unlink()