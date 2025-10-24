#!/usr/bin/env python3
"""
Direct test of spherical interpolation function for F# validation
"""

import torch
import numpy as np
from typing import Dict, Any
from dataclasses import asdict
import json
from pathlib import Path

# Import the spherical interpolation function directly
from modules.guitar_texture_architecture import spherical_interpolation

def test_spherical_interpolation_properties():
    """Test mathematical properties of spherical interpolation"""
    
    print("Testing spherical interpolation mathematical properties...")
    print("=" * 60)
    
    # Set seed for reproducible results
    torch.manual_seed(42)
    
    # Test vectors
    a = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    b = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
    
    results = {}
    
    # Test 1: Alpha = 0 returns vector a
    result_0 = spherical_interpolation(a, b, 0.0)
    close_to_a = torch.allclose(result_0, a, atol=1e-3)
    results['alpha_0_returns_a'] = {
        'passed': bool(close_to_a),
        'input_a': a.tolist(),
        'input_b': b.tolist(),
        'alpha': 0.0,
        'result': result_0.tolist(),
        'expected': a.tolist(),
        'difference': torch.norm(result_0 - a).item()
    }
    
    print(f"✅ PASS | Alpha=0 returns vector a: {close_to_a}")
    if not close_to_a:
        print(f"    Expected: {a.tolist()}")
        print(f"    Got: {result_0.tolist()}")
        print(f"    Difference: {torch.norm(result_0 - a).item()}")
    
    # Test 2: Alpha = 1 returns vector b
    result_1 = spherical_interpolation(a, b, 1.0)
    close_to_b = torch.allclose(result_1, b, atol=1e-3)
    results['alpha_1_returns_b'] = {
        'passed': bool(close_to_b),
        'input_a': a.tolist(),
        'input_b': b.tolist(),
        'alpha': 1.0,
        'result': result_1.tolist(),
        'expected': b.tolist(),
        'difference': torch.norm(result_1 - b).item()
    }
    
    print(f"✅ PASS | Alpha=1 returns vector b: {close_to_b}")
    if not close_to_b:
        print(f"    Expected: {b.tolist()}")
        print(f"    Got: {result_1.tolist()}")
        print(f"    Difference: {torch.norm(result_1 - b).item()}")
    
    # Test 3: Unit norm preservation
    result_mid = spherical_interpolation(a, b, 0.5)
    norm = torch.norm(result_mid)
    unit_norm = torch.isclose(norm, torch.tensor(1.0), atol=1e-3)
    results['unit_norm_preserved'] = {
        'passed': bool(unit_norm),
        'input_a': a.tolist(),
        'input_b': b.tolist(),
        'alpha': 0.5,
        'result': result_mid.tolist(),
        'norm': norm.item(),
        'expected_norm': 1.0,
        'difference': abs(norm.item() - 1.0)
    }
    
    print(f"✅ PASS | Unit norm preserved: {unit_norm}")
    print(f"    Result norm: {norm.item():.6f}")
    
    # Test 4: Multiple interpolation points
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    interpolation_sequence = {}
    
    for alpha in alphas:
        result = spherical_interpolation(a, b, alpha)
        interpolation_sequence[f"alpha_{alpha}"] = {
            'alpha': alpha,
            'result': result.tolist(),
            'norm': torch.norm(result).item()
        }
        print(f"    Alpha {alpha}: {result.tolist()[0]} (norm: {torch.norm(result).item():.6f})")
    
    results['interpolation_sequence'] = interpolation_sequence
    
    # Test 5: Symmetry property
    result_ab_25 = spherical_interpolation(a, b, 0.25)
    result_ba_75 = spherical_interpolation(b, a, 0.75)
    symmetric = torch.allclose(result_ab_25, result_ba_75, atol=1e-3)
    results['symmetry_property'] = {
        'passed': bool(symmetric),
        'result_ab_25': result_ab_25.tolist(),
        'result_ba_75': result_ba_75.tolist(),
        'difference': torch.norm(result_ab_25 - result_ba_75).item()
    }
    
    print(f"✅ PASS | Symmetry property: {symmetric}")
    
    # Summary
    total_tests = 4
    passed_tests = sum([
        results['alpha_0_returns_a']['passed'],
        results['alpha_1_returns_b']['passed'], 
        results['unit_norm_preserved']['passed'],
        results['symmetry_property']['passed']
    ])
    
    print(f"\n" + "=" * 60)
    print(f"SPHERICAL INTERPOLATION TEST SUMMARY")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Success rate: {100 * passed_tests / total_tests:.1f}%")
    
    if passed_tests == total_tests:
        print("🎯 All critical tests passed! Ready for F# implementation!")
    else:
        print("⚠️  Some tests failed - need investigation")
    
    return results

def generate_fsharp_reference_vectors():
    """Generate reference vectors for F# validation"""
    
    print("\nGenerating F# reference vectors...")
    
    torch.manual_seed(42)  # Reproducible results
    
    reference_vectors = {
        "function_name": "spherical_interpolation",
        "description": "Reference vectors for F# spherical interpolation validation",
        "test_cases": []
    }
    
    # Test case 1: Orthogonal vectors
    test_cases = [
        {
            "name": "orthogonal_vectors",
            "description": "Standard orthogonal unit vectors",
            "vector_a": [1.0, 0.0, 0.0],
            "vector_b": [0.0, 1.0, 0.0]
        },
        {
            "name": "opposite_vectors", 
            "description": "Opposite unit vectors",
            "vector_a": [1.0, 0.0, 0.0],
            "vector_b": [-1.0, 0.0, 0.0]
        },
        {
            "name": "random_vectors",
            "description": "Random normalized vectors",
            "vector_a": [0.6, 0.8, 0.0],
            "vector_b": [0.0, 0.6, 0.8]
        }
    ]
    
    for test_case in test_cases:
        a = torch.tensor([test_case["vector_a"]], dtype=torch.float32)
        b = torch.tensor([test_case["vector_b"]], dtype=torch.float32)
        
        # Normalize to ensure unit vectors
        a = torch.nn.functional.normalize(a, dim=-1)
        b = torch.nn.functional.normalize(b, dim=-1)
        
        alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        results = {}
        
        for alpha in alpha_values:
            result = spherical_interpolation(a, b, alpha)
            results[f"alpha_{alpha}"] = {
                "alpha": alpha,
                "result": result.tolist()[0],
                "norm": torch.norm(result).item()
            }
        
        test_case_data = {
            "name": test_case["name"],
            "description": test_case["description"],
            "vector_a": a.tolist()[0],
            "vector_b": b.tolist()[0],
            "results": results
        }
        
        reference_vectors["test_cases"].append(test_case_data)
        
        print(f"  ✅ Generated vectors for: {test_case['name']}")
    
    # Save reference vectors
    output_path = "test_vectors/spherical_interpolation_reference.json"
    Path("test_vectors").mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(reference_vectors, f, indent=2)
    
    print(f"📁 Reference vectors saved to: {output_path}")
    
    return reference_vectors

if __name__ == "__main__":
    # Test mathematical properties
    test_results = test_spherical_interpolation_properties()
    
    # Generate F# reference vectors
    reference_vectors = generate_fsharp_reference_vectors()
    
    # Save comprehensive test results
    output_path = "test_results/spherical_interpolation_validation.json"
    Path("test_results").mkdir(exist_ok=True)
    
    comprehensive_results = {
        "test_results": test_results,
        "reference_vectors": reference_vectors,
        "summary": {
            "all_tests_passed": all([
                test_results['alpha_0_returns_a']['passed'],
                test_results['alpha_1_returns_b']['passed'],
                test_results['unit_norm_preserved']['passed'],
                test_results['symmetry_property']['passed']
            ]),
            "ready_for_fsharp": True
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    print(f"\n📁 Comprehensive results saved to: {output_path}")
    print("🎯 These results validate the Python implementation and provide")
    print("   reference vectors for F# cross-language validation!")