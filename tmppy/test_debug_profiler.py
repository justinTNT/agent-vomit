import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'orchestration'))

try:
    from model_profiler import ModelProfiler
    
    profiler = ModelProfiler(warmup_runs=1, profile_runs=2)
    
    # Simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    report = profiler.profile_model(model, torch.randn(4, 10))
    print("Success!")
    
except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()