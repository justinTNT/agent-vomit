#!/usr/bin/env python3
"""Test with corrected module filenames."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

# Map expected names to actual filenames
MODULE_NAME_MAPPING = {
    'stft_loss': 'multi_scale_stft_loss',
    'autoencoder_vae': 'auto_encoder',
    'memory_bank_retriever': 'memory_bank',
    'causal_conv': 'causal_conv1d',
    'sequence_to_sequence': 'sequence_to_sequence_model'
}

# Also check class name mappings
CLASS_NAME_MAPPING = {
    'AutoencoderVAE': 'AutoEncoder',
    'MemoryBankRetriever': 'MemoryBank',
    'CausalConv1d': 'CausalConv1d',  # This one might be correct
    'SequenceToSequenceModel': 'SequenceToSequenceModel'  # This one might be correct
}

def check_actual_modules():
    """Check what modules and classes actually exist."""
    
    print("=== CHECKING ACTUAL MODULE STRUCTURE ===\n")
    
    for candidate in ['agent_claude', 'agent_codex']:
        print(f"\n{candidate.upper()}:")
        print("-" * 40)
        
        candidate_dir = Path(f'candidates/{candidate}')
        
        # Check the "missing" modules
        for expected, actual in MODULE_NAME_MAPPING.items():
            expected_file = candidate_dir / f"{expected}.py"
            actual_file = candidate_dir / f"{actual}.py"
            
            if expected_file.exists():
                print(f"✅ {expected}.py exists")
            elif actual_file.exists():
                print(f"⚠️  {expected}.py → {actual}.py (different name)")
                
                # Check class name
                with open(actual_file, 'r') as f:
                    content = f.read()
                    # Find class definition
                    import re
                    class_matches = re.findall(r'class\s+(\w+)', content)
                    if class_matches:
                        print(f"   Classes found: {', '.join(class_matches)}")
            else:
                print(f"❌ Neither {expected}.py nor {actual}.py exists")

if __name__ == "__main__":
    check_actual_modules()