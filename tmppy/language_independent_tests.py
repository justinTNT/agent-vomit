#!/usr/bin/env python3
"""
Language-independent test framework for module validation

This framework allows Python and F# implementations to be tested against
the same behavioral specifications and reference vectors.
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class TestResult:
    """Result of a single test"""
    name: str
    passed: bool
    error_message: Optional[str] = None
    metrics: Dict[str, float] = None

class ModuleTestInterface(ABC):
    """Abstract interface that both Python and F# implementations must support"""
    
    @abstractmethod
    def hierarchical_quantization(self, texture_input: np.ndarray) -> Dict[str, np.ndarray]:
        """Test hierarchical texture quantization"""
        pass
    
    @abstractmethod
    def texture_content_disentanglement(self, audio_input: np.ndarray) -> Dict[str, np.ndarray]:
        """Test texture-content separation"""
        pass
    
    @abstractmethod
    def spherical_interpolation(self, vector_a: np.ndarray, vector_b: np.ndarray, alpha: float) -> np.ndarray:
        """Test spherical interpolation"""
        pass
    
    @abstractmethod
    def texture_transfer(self, source_audio: np.ndarray, target_audio: np.ndarray) -> np.ndarray:
        """Test texture transfer between audio samples"""
        pass

class PythonModuleImplementation(ModuleTestInterface):
    """Python implementation wrapper"""
    
    def __init__(self, config_path: str = None):
        from modules.guitar_texture_architecture import create_guitar_texture_model, GuitarTextureConfig
        
        if config_path:
            with open(config_path) as f:
                config_dict = json.load(f)
            config = GuitarTextureConfig(**config_dict["config"])
        else:
            config = GuitarTextureConfig()
        
        self.model = create_guitar_texture_model(config)
        self.model.eval()
    
    def hierarchical_quantization(self, texture_input: np.ndarray) -> Dict[str, np.ndarray]:
        tensor_input = torch.from_numpy(texture_input).float()
        with torch.no_grad():
            output = self.model.quantizer(tensor_input)
        
        return {
            "quantized": output["quantized"].numpy(),
            "level_indices": {
                level: indices.numpy() 
                for level, indices in output["level_indices"].items()
            },
            "commitment_loss": float(output["commitment_loss"])
        }
    
    def texture_content_disentanglement(self, audio_input: np.ndarray) -> Dict[str, np.ndarray]:
        tensor_input = torch.from_numpy(audio_input).float()
        with torch.no_grad():
            output = self.model.vae(tensor_input, sample_texture=False, sample_content=False)
        
        return {
            "texture_mu": output["texture_mu"].numpy(),
            "content_mu": output["content_mu"].numpy(), 
            "reconstruction": output["reconstruction"].numpy()
        }
    
    def spherical_interpolation(self, vector_a: np.ndarray, vector_b: np.ndarray, alpha: float) -> np.ndarray:
        from modules.guitar_texture_architecture import spherical_interpolation
        
        tensor_a = torch.from_numpy(vector_a).float()
        tensor_b = torch.from_numpy(vector_b).float()
        
        with torch.no_grad():
            result = spherical_interpolation(tensor_a, tensor_b, alpha)
        
        return result.numpy()
    
    def texture_transfer(self, source_audio: np.ndarray, target_audio: np.ndarray) -> np.ndarray:
        tensor_source = torch.from_numpy(source_audio).float()
        tensor_target = torch.from_numpy(target_audio).float()
        
        with torch.no_grad():
            result = self.model.vae.texture_transfer(tensor_source, tensor_target)
        
        return result.numpy()

class UniversalTestRunner:
    """Runs tests against any implementation that follows ModuleTestInterface"""
    
    def __init__(self, reference_vectors_path: str):
        with open(reference_vectors_path) as f:
            self.reference = json.load(f)
        self.tolerance = 1e-4  # Numerical tolerance for comparisons
    
    def run_all_tests(self, implementation: ModuleTestInterface) -> List[TestResult]:
        """Run all tests against an implementation"""
        results = []
        
        # Test 1: Hierarchical Quantization
        results.append(self._test_hierarchical_quantization(implementation))
        
        # Test 2: Texture-Content Disentanglement  
        results.append(self._test_texture_content_disentanglement(implementation))
        
        # Test 3: Spherical Interpolation
        results.append(self._test_spherical_interpolation(implementation))
        
        # Test 4: Texture Transfer
        results.append(self._test_texture_transfer(implementation))
        
        return results
    
    def _test_hierarchical_quantization(self, impl: ModuleTestInterface) -> TestResult:
        """Test hierarchical quantization against reference"""
        try:
            test_data = self.reference["tests"]["hierarchical_quantization"]
            input_data = np.array(test_data["input"])
            
            # Run implementation
            result = impl.hierarchical_quantization(input_data)
            
            # Check quantized output shape and values
            expected_quantized = np.array(test_data["outputs"]["quantized"])
            if not self._arrays_close(result["quantized"], expected_quantized):
                return TestResult("hierarchical_quantization", False, 
                                "Quantized output doesn't match reference")
            
            # Check level indices
            for level, expected_indices in test_data["outputs"]["level_indices"].items():
                if level not in result["level_indices"]:
                    return TestResult("hierarchical_quantization", False,
                                    f"Missing level indices for {level}")
                
                expected = np.array(expected_indices)
                actual = result["level_indices"][level]
                if not np.array_equal(actual, expected):
                    return TestResult("hierarchical_quantization", False,
                                    f"Level {level} indices don't match")
            
            return TestResult("hierarchical_quantization", True)
            
        except Exception as e:
            return TestResult("hierarchical_quantization", False, str(e))
    
    def _test_texture_content_disentanglement(self, impl: ModuleTestInterface) -> TestResult:
        """Test texture-content disentanglement"""
        try:
            test_data = self.reference["tests"]["texture_content_disentanglement"]
            input_data = np.array(test_data["input"])
            
            result = impl.texture_content_disentanglement(input_data)
            
            # Check texture_mu
            expected_texture = np.array(test_data["outputs"]["texture_mu"])
            if not self._arrays_close(result["texture_mu"], expected_texture):
                return TestResult("texture_content_disentanglement", False,
                                "Texture mu doesn't match reference")
            
            # Check content_mu  
            expected_content = np.array(test_data["outputs"]["content_mu"])
            if not self._arrays_close(result["content_mu"], expected_content):
                return TestResult("texture_content_disentanglement", False,
                                "Content mu doesn't match reference")
            
            return TestResult("texture_content_disentanglement", True)
            
        except Exception as e:
            return TestResult("texture_content_disentanglement", False, str(e))
    
    def _test_spherical_interpolation(self, impl: ModuleTestInterface) -> TestResult:
        """Test spherical interpolation properties"""
        try:
            test_data = self.reference["tests"]["spherical_interpolation"]
            vector_a = np.array(test_data["inputs"]["vector_a"])
            vector_b = np.array(test_data["inputs"]["vector_b"])
            
            # Test key interpolation points
            for alpha_str, expected in test_data["outputs"].items():
                alpha = float(alpha_str.split("_")[1])
                expected_result = np.array(expected)
                
                actual_result = impl.spherical_interpolation(vector_a, vector_b, alpha)
                
                if not self._arrays_close(actual_result, expected_result):
                    return TestResult("spherical_interpolation", False,
                                    f"Interpolation at alpha={alpha} doesn't match")
            
            # Test boundary conditions
            result_0 = impl.spherical_interpolation(vector_a, vector_b, 0.0)
            result_1 = impl.spherical_interpolation(vector_a, vector_b, 1.0)
            
            if not self._arrays_close(result_0, vector_a, tolerance=1e-3):
                return TestResult("spherical_interpolation", False,
                                "Alpha=0 should return vector_a")
            
            if not self._arrays_close(result_1, vector_b, tolerance=1e-3):
                return TestResult("spherical_interpolation", False,
                                "Alpha=1 should return vector_b")
            
            return TestResult("spherical_interpolation", True)
            
        except Exception as e:
            return TestResult("spherical_interpolation", False, str(e))
    
    def _test_texture_transfer(self, impl: ModuleTestInterface) -> TestResult:
        """Test texture transfer"""
        try:
            test_data = self.reference["tests"]["texture_transfer"]
            source_audio = np.array(test_data["inputs"]["source_audio"])
            target_audio = np.array(test_data["inputs"]["target_audio"])
            
            result = impl.texture_transfer(source_audio, target_audio)
            expected = np.array(test_data["output"])
            
            if not self._arrays_close(result, expected):
                return TestResult("texture_transfer", False,
                                "Texture transfer output doesn't match reference")
            
            return TestResult("texture_transfer", True)
            
        except Exception as e:
            return TestResult("texture_transfer", False, str(e))
    
    def _arrays_close(self, actual: np.ndarray, expected: np.ndarray, tolerance: float = None) -> bool:
        """Check if two arrays are numerically close"""
        if tolerance is None:
            tolerance = self.tolerance
        
        if actual.shape != expected.shape:
            return False
        
        return np.allclose(actual, expected, atol=tolerance, rtol=tolerance)

def run_python_tests(reference_vectors_path: str) -> List[TestResult]:
    """Run tests against Python implementation"""
    implementation = PythonModuleImplementation()
    runner = UniversalTestRunner(reference_vectors_path)
    return runner.run_all_tests(implementation)

def print_test_results(results: List[TestResult]):
    """Print formatted test results"""
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for result in results:
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"{status} | {result.name}")
        if not result.passed and result.error_message:
            print(f"      Error: {result.error_message}")
        passed += result.passed
    
    print("-"*60)
    print(f"SUMMARY: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
    print("="*60)

if __name__ == "__main__":
    # Generate reference vectors first
    print("Generating reference vectors...")
    import sys
    sys.path.append(str(Path(__file__).parent))
    
    from generate_reference_vectors import generate_test_vectors, save_test_vectors
    
    # Create test vectors directory
    Path("test_vectors").mkdir(exist_ok=True)
    
    # Generate and save reference vectors
    vectors = generate_test_vectors()
    reference_path = "test_vectors/guitar_texture_reference.json"
    save_test_vectors(vectors, reference_path)
    
    # Run tests against Python implementation
    print("\nRunning tests against Python implementation...")
    results = run_python_tests(reference_path)
    print_test_results(results)
    
    print(f"\nReference vectors saved to: {reference_path}")
    print("F# implementation can use these same vectors for validation!")