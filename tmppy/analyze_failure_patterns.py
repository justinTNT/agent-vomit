#!/usr/bin/env python3
"""
Analyze failure patterns from comprehensive test to create systematic parameter fixes
Strategy: Learn from Wave 1/2 success (96.4%) to fix the remaining 119 failing modules
"""

import json
import re
from collections import defaultdict, Counter
from pathlib import Path

def analyze_comprehensive_results():
    """Analyze the comprehensive test results to identify failure patterns"""
    
    # Load results
    with open("/Users/jtnt/Play/agent-vomit/comprehensive_coverage_results.json") as f:
        results = json.load(f)
    
    print("🔍 ANALYZING 119 FAILING MODULES FOR SYSTEMATIC PATTERNS")
    print("=" * 70)
    print(f"Total classes: {results['metadata']['classes_discovered']}")
    print(f"Passed: {results['metadata']['tests_passed']}")
    print(f"Failed: {results['metadata']['classes_discovered'] - results['metadata']['tests_passed']}")
    print()
    
    # Categorize failures
    failure_categories = {
        "missing_required_args": [],
        "unexpected_keyword_args": [],
        "import_failures": [],
        "none_failures": [],
        "other_failures": []
    }
    
    parameter_patterns = defaultdict(list)
    successful_patterns = defaultdict(list)
    
    for result in results["all_results"]:
        class_name = result["class_name"]
        init_error = result["result"].get("init_error")
        
        if result["result"]["module_passed"]:
            # Track successful patterns
            successful_patterns["working_classes"].append(class_name)
        elif init_error:
            # Categorize failures
            if "missing" in init_error and "required positional argument" in init_error:
                # Extract missing arguments
                missing_args = re.findall(r"'([^']+)'", init_error)
                failure_categories["missing_required_args"].append({
                    "class": class_name,
                    "missing_args": missing_args,
                    "error": init_error
                })
                for arg in missing_args:
                    parameter_patterns[arg].append(class_name)
                    
            elif "unexpected keyword argument" in init_error:
                # Extract unexpected arguments
                unexpected_arg = re.search(r"'([^']+)'", init_error)
                if unexpected_arg:
                    arg = unexpected_arg.group(1)
                    failure_categories["unexpected_keyword_args"].append({
                        "class": class_name,
                        "unexpected_arg": arg,
                        "error": init_error
                    })
                    
            elif init_error == "None":
                failure_categories["none_failures"].append({
                    "class": class_name,
                    "error": "None - likely missing test logic"
                })
            else:
                failure_categories["other_failures"].append({
                    "class": class_name,
                    "error": init_error
                })
    
    # Analyze patterns
    print("📊 FAILURE PATTERN ANALYSIS:")
    print("-" * 50)
    for category, failures in failure_categories.items():
        if failures:
            print(f"{category.upper()}: {len(failures)} classes")
            if category == "missing_required_args" and len(failures) <= 10:
                for failure in failures[:5]:  # Show first 5
                    print(f"  - {failure['class']}: needs {failure['missing_args']}")
            elif category == "unexpected_keyword_args" and len(failures) <= 10:
                for failure in failures[:5]:  # Show first 5
                    print(f"  - {failure['class']}: rejected {failure['unexpected_arg']}")
    print()
    
    # Most common missing parameters
    print("🔧 MOST COMMON MISSING PARAMETERS:")
    print("-" * 50)
    param_counts = Counter()
    for param, classes in parameter_patterns.items():
        param_counts[param] = len(classes)
    
    for param, count in param_counts.most_common(15):
        print(f"  {param}: needed by {count} classes")
    print()
    
    return failure_categories, parameter_patterns

def create_parameter_mapping_rules():
    """Create systematic parameter mapping rules based on analysis"""
    
    # Based on successful Wave 1/2 patterns and common requirements
    parameter_rules = {
        # Audio processing patterns
        "sample_rate": 22050,
        "n_fft": 1024,
        "hop_length": 256,
        "hop_size": 256,
        "win_length": 1024,
        "f_min": 50.0,
        "f_max": 8000.0,
        "n_bark_bands": 24,
        
        # Model architecture patterns
        "d_model": 256,
        "n_heads": 8,
        "n_layers": 4,
        "d_ff": 1024,
        "hidden_size": 256,
        "hidden_dim": 256,
        "input_dim": 256,
        "output_dim": 128,
        "embed_dim": 256,
        "embedding_dim": 256,
        "latent_dim": 64,
        
        # Convolution patterns
        "in_channels": 64,
        "out_channels": 128,
        "channels": 256,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "residual_channels": 256,
        "skip_channels": 256,
        
        # Quantization patterns
        "num_quantizers": 4,
        "num_embeddings": 1024,
        "codebook_size": 1024,
        "codebook_dim": 256,
        "commitment_cost": 0.25,
        
        # Sequence patterns
        "vocab_size": 1000,
        "max_seq_len": 100,
        "max_length": 100,
        "seq_len": 100,
        "sequence_length": 100,
        
        # Training patterns
        "dropout": 0.1,
        "learning_rate": 0.001,
        "batch_size": 32,
        
        # Vision patterns
        "img_size": 224,
        "patch_size": 16,
        "n_patches": 196,
        "num_classes": 10,
        
        # Memory/storage patterns
        "memory_size": 1000,
        "cache_size": 1000,
        "buffer_size": 1000,
        "max_size": 1000,
        
        # Positional/indexing
        "pad_token_id": 0,
        "start_token_id": 1,
        "end_token_id": 2,
        "pad_idx": 0,
        
        # Graph/attention patterns
        "n_inducing_points": 32,
        "in_dim": 256,
        "out_dim": 128,
        
        # Audio-specific
        "subbands": 4,
        "filter_size": 64,
        "beta": 9.0,
        "num_scales": 3,
        "input_channels": 1,
        "audiolen": 1024,
        "input_fdim": 128,
        "input_tdim": 100,
        
        # Configuration patterns
        "config": None,  # Will need special handling
        "project_name": "test_project",
        "parameter_space": {},
        
        # Encoder/decoder patterns  
        "encoder": None,  # Will need special handling
        "decoder": None,  # Will need special handling
        
        # Data patterns
        "input_size": 256,
        "output_size": 128,
        "feature_dim": 256,
        "key_dim": 64,
        "value_dim": 128,
        
        # Module composition
        "src_vocab_size": 1000,
        "tgt_vocab_size": 1000,
        "target_vocab_size": 1000,
        "share_embeddings": False,
        "tie_embeddings": False,
        
        # Normalization
        "num_features": 128,
        "normalized_shape": 256,
        "eps": 1e-5,
        
        # Loss functions
        "reduction": "mean",
        "target_real_label": 1.0,
        "target_fake_label": 0.0,
        
        # Advanced patterns
        "layer_type": "gcn",
        "pooling": "mean",
        "fusion_type": "concat",
        "similarity": "cosine",
        "update_method": "momentum",
        "window_type": "tumbling",
        "aggregation": "mean",
        "architecture": "transformer",
        "use_cls_token": True,
        "pool_type": "cls",
        "use_positional": True,
        "use_isab": False,
        "time_based": False,
        "track_distributions": True,
        "evolve_schema": False,
        "use_act": False,
        "act_threshold": 0.99,
        "max_steps": 10,
        "backpressure_threshold": 0.8,
        "window_slide": None,
        "session_timeout": None,
        "window_size": 1000,
        "step_size": 100,
        "transform": "fft"
    }
    
    return parameter_rules

def create_systematic_fixer():
    """Create systematic parameter fixing system"""
    
    print("🛠️  CREATING SYSTEMATIC PARAMETER FIXING SYSTEM")
    print("=" * 70)
    
    parameter_rules = create_parameter_mapping_rules()
    
    print(f"📋 Parameter rules created: {len(parameter_rules)} mappings")
    print("🎯 Strategy: Apply proven successful patterns to all failing modules")
    print()
    
    # Special handling patterns
    special_patterns = {
        "config_based": [
            "config", "project_name", "parameter_space"
        ],
        "encoder_decoder": [
            "encoder", "decoder"
        ],
        "audio_processing": [
            "sample_rate", "n_fft", "hop_length", "f_min", "f_max"
        ],
        "transformer_architecture": [
            "d_model", "n_heads", "n_layers", "d_ff", "dropout"
        ],
        "convolution": [
            "in_channels", "out_channels", "kernel_size", "stride"
        ],
        "quantization": [
            "num_quantizers", "num_embeddings", "codebook_size", "commitment_cost"
        ]
    }
    
    print("🔧 SPECIAL HANDLING PATTERNS:")
    for pattern_type, params in special_patterns.items():
        print(f"  {pattern_type}: {len(params)} parameters")
    print()
    
    return parameter_rules, special_patterns

def generate_fixing_strategy():
    """Generate comprehensive fixing strategy"""
    
    print("📋 SYSTEMATIC FIXING STRATEGY")
    print("=" * 70)
    print("Phase 1: Apply automatic parameter mapping (80% of failures)")
    print("Phase 2: Handle special cases (config, encoder/decoder)")  
    print("Phase 3: Create minimal implementations for complex dependencies")
    print("Phase 4: Verify 90%+ success rate achievement")
    print()
    
    strategy = {
        "phase_1": {
            "description": "Automatic parameter mapping",
            "target": "Fix 80% of parameter mismatch failures",
            "method": "Apply proven parameter rules systematically",
            "expected_improvement": "60-70% success rate"
        },
        "phase_2": {
            "description": "Special case handling", 
            "target": "Fix config and dependency issues",
            "method": "Create minimal config objects and simple encoders/decoders",
            "expected_improvement": "75-85% success rate"
        },
        "phase_3": {
            "description": "Complex dependency resolution",
            "target": "Handle remaining import and dependency failures", 
            "method": "Create mock implementations for missing dependencies",
            "expected_improvement": "85-90% success rate"
        },
        "phase_4": {
            "description": "Final optimization",
            "target": "Achieve 90%+ success rate",
            "method": "Fine-tune remaining edge cases",
            "expected_improvement": "90%+ success rate"
        }
    }
    
    for phase, details in strategy.items():
        print(f"🎯 {phase.upper()}:")
        print(f"  Target: {details['target']}")
        print(f"  Method: {details['method']}")
        print(f"  Expected: {details['expected_improvement']}")
        print()
    
    return strategy

def main():
    """Main analysis and strategy creation"""
    
    print("🚀 SYSTEMATIC PARAMETER FIXING ANALYSIS")
    print("=" * 70)
    print("Goal: Transform 17.9% → 90%+ success rate using proven methods")
    print()
    
    # Analyze failures
    failure_categories, parameter_patterns = analyze_comprehensive_results()
    
    # Create fixing system
    parameter_rules, special_patterns = create_systematic_fixer()
    
    # Generate strategy
    strategy = generate_fixing_strategy()
    
    # Save analysis
    analysis_data = {
        "failure_categories": failure_categories,
        "parameter_patterns": dict(parameter_patterns),
        "parameter_rules": parameter_rules,
        "special_patterns": special_patterns,
        "strategy": strategy,
        "summary": {
            "current_success_rate": 17.9,
            "target_success_rate": 90.0,
            "improvement_needed": 72.1,
            "total_failing_modules": 119,
            "systematic_approach": "Apply Wave 1/2 success patterns (96.4%) to all modules"
        }
    }
    
    with open("systematic_fixing_analysis.json", "w") as f:
        json.dump(analysis_data, f, indent=2, default=str)
    
    print("📁 Analysis saved to: systematic_fixing_analysis.json")
    print()
    print("🎯 READY TO EXECUTE SYSTEMATIC FIXING!")
    print("Expected outcome: 17.9% → 90%+ success rate")
    print("Method: Proven Wave 1/2 success patterns applied systematically")

if __name__ == "__main__":
    main()