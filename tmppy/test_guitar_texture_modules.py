#!/usr/bin/env python3
"""
Test guitar texture modules using the enhanced universal test runner
"""

import json
from typing import Dict, Any
from pathlib import Path
from enhanced_universal_test_runner import (
    EnhancedPythonTestImplementation, 
    EnhancedUniversalTestRunner
)

# Guitar texture module test configurations
GUITAR_TEXTURE_TEST_CONFIG = {
    "schema_version": "1.0",
    "description": "Guitar texture modules test configurations",
    "total_modules": 4,
    "modules": [
        {
            "module_name": "guitar_texture_architecture",
            "class_name": "spherical_interpolation",
            "class_variations": ["spherical_interpolation"],
            "init_params": {},
            "tests": [
                {
                    "name": "boundary_alpha_0",
                    "test_type": "callable",
                    "args": [
                        {"type": "tensor", "shape": [1, 3], "dtype": "float32"},
                        {"type": "tensor", "shape": [1, 3], "dtype": "float32"},
                        0.0
                    ],
                    "description": "Test alpha=0 returns first vector"
                },
                {
                    "name": "boundary_alpha_1", 
                    "test_type": "callable",
                    "args": [
                        {"type": "tensor", "shape": [1, 3], "dtype": "float32"},
                        {"type": "tensor", "shape": [1, 3], "dtype": "float32"},
                        1.0
                    ],
                    "description": "Test alpha=1 returns second vector"
                },
                {
                    "name": "interpolation_midpoint",
                    "test_type": "callable", 
                    "args": [
                        {"type": "tensor", "shape": [1, 3], "dtype": "float32"},
                        {"type": "tensor", "shape": [1, 3], "dtype": "float32"},
                        0.5
                    ],
                    "description": "Test alpha=0.5 interpolation"
                }
            ]
        },
        {
            "module_name": "guitar_texture_architecture",
            "class_name": "GuitarTextureConfig",
            "class_variations": ["GuitarTextureConfig"],
            "init_params": {},
            "tests": [
                {
                    "name": "initialization",
                    "test_type": "callable",
                    "args": [],
                    "description": "Test config initialization"
                }
            ]
        },
        {
            "module_name": "guitar_texture_architecture",
            "class_name": "MultiScaleTextureEncoder",
            "class_variations": ["MultiScaleTextureEncoder"],
            "init_params": {
                "config": {"type": "config_object"}
            },
            "tests": [
                {
                    "name": "forward_pass",
                    "test_type": "forward",
                    "args": [
                        {"type": "tensor", "shape": [2, 1, 256], "dtype": "float32"}
                    ],
                    "description": "Test forward pass with audio input"
                },
                {
                    "name": "multi_scale_output",
                    "test_type": "shape",
                    "input_tensor": {"type": "tensor", "shape": [2, 1, 256], "dtype": "float32"},
                    "expected_shape": "reduce",
                    "description": "Test multi-scale feature extraction"
                }
            ]
        },
        {
            "module_name": "guitar_texture_architecture", 
            "class_name": "HierarchicalTextureQuantizer",
            "class_variations": ["HierarchicalTextureQuantizer"],
            "init_params": {
                "config": {"type": "config_object"}
            },
            "tests": [
                {
                    "name": "quantization",
                    "test_type": "forward",
                    "args": [
                        {"type": "tensor", "shape": [2, 64], "dtype": "float32"}
                    ],
                    "description": "Test texture quantization"
                },
                {
                    "name": "decode_method",
                    "test_type": "method",
                    "method_name": "decode_from_indices",
                    "description": "Test decode method exists"
                }
            ]
        }
    ]
}

class GuitarTextureTestImplementation(EnhancedPythonTestImplementation):
    """Enhanced implementation with guitar texture module support"""
    
    def __init__(self):
        super().__init__()
        
        # Add guitar texture specific fixers
        self.fixers.update({
            'MultiScaleTextureEncoder': self.fix_guitar_texture_encoder,
            'HierarchicalTextureQuantizer': self.fix_guitar_texture_quantizer,
            'GuitarTextureModel': self.fix_guitar_texture_model,
        })
    
    def fix_guitar_texture_encoder(self, init_params: Dict[str, Any]) -> Dict[str, Any]:
        """Fix MultiScaleTextureEncoder initialization"""
        fixed_params = init_params.copy()
        
        # Handle config parameter
        if 'config' in fixed_params:
            if fixed_params['config'].get('type') == 'config_object':
                # Create GuitarTextureConfig instance
                from modules.guitar_texture_architecture import GuitarTextureConfig
                fixed_params['config'] = GuitarTextureConfig()
        
        return fixed_params
    
    def fix_guitar_texture_quantizer(self, init_params: Dict[str, Any]) -> Dict[str, Any]:
        """Fix HierarchicalTextureQuantizer initialization"""
        fixed_params = init_params.copy()
        
        # Handle config parameter
        if 'config' in fixed_params:
            if fixed_params['config'].get('type') == 'config_object':
                from modules.guitar_texture_architecture import GuitarTextureConfig
                fixed_params['config'] = GuitarTextureConfig()
        
        return fixed_params
    
    def fix_guitar_texture_model(self, init_params: Dict[str, Any]) -> Dict[str, Any]:
        """Fix GuitarTextureModel initialization"""
        fixed_params = init_params.copy()
        
        # Handle config parameter
        if 'config' in fixed_params:
            if fixed_params['config'].get('type') == 'config_object':
                from modules.guitar_texture_architecture import GuitarTextureConfig
                fixed_params['config'] = GuitarTextureConfig()
        
        return fixed_params

def test_guitar_texture_modules():
    """Test guitar texture modules specifically"""
    
    print("Testing Guitar Texture Modules with Enhanced Universal Test Runner...")
    print("=" * 80)
    
    # Save test config to temporary file
    config_path = "test_vectors/guitar_texture_test_config.json"
    Path("test_vectors").mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(GUITAR_TEXTURE_TEST_CONFIG, f, indent=2)
    
    # Run tests
    implementation = GuitarTextureTestImplementation()
    runner = EnhancedUniversalTestRunner(implementation)
    
    results = runner.run_from_json_config(config_path)
    runner.print_summary()
    
    # Save results
    results_path = "test_results/guitar_texture_test_results.json"
    Path("test_results").mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        from dataclasses import asdict
        serializable_results = [asdict(r) for r in results]
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n📁 Guitar texture test results saved to: {results_path}")
    
    # Analyze guitar texture specific results
    guitar_passed = sum(1 for r in results if r.passed)
    guitar_total = len(results)
    
    print(f"\n🎸 GUITAR TEXTURE MODULES SUMMARY:")
    print(f"   Passed: {guitar_passed}/{guitar_total} ({100 * guitar_passed / guitar_total:.1f}%)")
    
    # Check critical guitar texture functionality
    critical_tests = ['boundary_alpha_0', 'boundary_alpha_1', 'interpolation_midpoint']
    critical_results = [r for r in results if r.name in critical_tests]
    critical_passed = sum(1 for r in critical_results if r.passed)
    
    if critical_passed == len(critical_results):
        print("   ✅ All critical interpolation tests passed!")
        print("   🎯 Ready for F# spherical interpolation implementation!")
    else:
        print(f"   ⚠️  {critical_passed}/{len(critical_results)} critical tests passed")
    
    return results

if __name__ == "__main__":
    results = test_guitar_texture_modules()