import torch
import pytest
from modules.data_validator import (
    DataValidator, Schema, DType, Shape, Range, 
    NotNaN, NotInf, Custom
)


def test_basic_validation():
    """Test basic tensor validation."""
    schema = Schema(rules={
        "input": [
            DType(torch.float32),
            Shape((10, 20)),
            Range(min_val=0.0, max_val=1.0)
        ]
    })
    
    validator = DataValidator(schema=schema)
    
    # Valid data
    valid_data = {"input": torch.rand(10, 20)}
    is_valid, errors = validator(valid_data)
    assert is_valid
    assert len(errors) == 0
    
    # Invalid dtype
    invalid_dtype = {"input": torch.randint(0, 10, (10, 20))}
    is_valid, errors = validator(invalid_dtype)
    assert not is_valid
    assert any("dtype" in e for e in errors)
    
    # Invalid shape
    invalid_shape = {"input": torch.rand(5, 20)}
    is_valid, errors = validator(invalid_shape)
    assert not is_valid
    assert any("Dimension" in e for e in errors)
    

def test_wildcard_shapes():
    """Test shape validation with wildcards."""
    schema = Schema(rules={
        "batch": [Shape(("*", 128, "*"))]
    })
    
    validator = DataValidator(schema=schema)
    
    # Should accept various batch and sequence dimensions
    valid_shapes = [
        (32, 128, 10),
        (1, 128, 50),
        (64, 128, 100)
    ]
    
    for shape in valid_shapes:
        data = {"batch": torch.rand(shape)}
        is_valid, _ = validator(data)
        assert is_valid
        
    # Should reject wrong hidden dimension
    invalid = {"batch": torch.rand(32, 64, 10)}
    is_valid, _ = validator(invalid)
    assert not is_valid
    

def test_nan_inf_validation():
    """Test NaN and Inf validation."""
    schema = Schema(rules={
        "clean": [NotNaN(), NotInf()]
    })
    
    validator = DataValidator(schema=schema)
    
    # Clean data
    clean = {"clean": torch.rand(10, 10)}
    is_valid, _ = validator(clean)
    assert is_valid
    
    # Data with NaN
    with_nan = {"clean": torch.rand(10, 10)}
    with_nan["clean"][0, 0] = float('nan')
    is_valid, errors = validator(with_nan)
    assert not is_valid
    assert any("NaN" in e for e in errors)
    
    # Data with Inf
    with_inf = {"clean": torch.rand(10, 10)}
    with_inf["clean"][0, 0] = float('inf')
    is_valid, errors = validator(with_inf)
    assert not is_valid
    assert any("infinite" in e for e in errors)
    

def test_custom_validation():
    """Test custom validation functions."""
    def is_normalized(tensor):
        return torch.allclose(tensor.norm(p=2, dim=-1), torch.ones(tensor.shape[0]))
    
    schema = Schema(rules={
        "embeddings": [
            Custom(is_normalized, "Embeddings must be L2 normalized")
        ]
    })
    
    validator = DataValidator(schema=schema)
    
    # Normalized embeddings
    embeddings = torch.randn(10, 128)
    embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
    is_valid, _ = validator({"embeddings": embeddings})
    assert is_valid
    
    # Non-normalized embeddings
    bad_embeddings = torch.randn(10, 128)
    is_valid, errors = validator({"embeddings": bad_embeddings})
    assert not is_valid
    assert any("L2 normalized" in e for e in errors)
    

def test_distribution_tracking():
    """Test distribution shift detection."""
    validator = DataValidator(track_distributions=True)
    
    # Build up statistics with normal data
    for _ in range(20):
        data = {"values": torch.randn(100) * 0.1 + 0.5}
        validator(data)
    
    # Check statistics
    stats = validator.get_statistics()
    assert abs(stats["statistics"]["mean"]["values"] - 0.5) < 0.1
    
    # Send data with distribution shift
    shifted_data = {"values": torch.randn(100) * 0.1 + 5.0}  # Mean shifted to 5.0
    is_valid, errors = validator(shifted_data)
    
    # Should detect the shift
    assert any("mean shift" in e for e in errors)
    

def test_strict_mode():
    """Test strict mode for extra fields."""
    schema = Schema(
        rules={"expected": [DType(torch.float32)]},
        strict=True
    )
    
    validator = DataValidator(schema=schema)
    
    # Extra field should fail in strict mode
    data = {
        "expected": torch.rand(10),
        "unexpected": torch.rand(5)
    }
    is_valid, errors = validator(data)
    assert not is_valid
    assert any("Unexpected field" in e for e in errors)
    
    # Non-strict mode
    schema_loose = Schema(
        rules={"expected": [DType(torch.float32)]},
        strict=False
    )
    validator_loose = DataValidator(schema=schema_loose)
    is_valid, _ = validator_loose(data)
    assert is_valid
    

def test_schema_evolution():
    """Test schema evolution suggestions."""
    validator = DataValidator(evolve_schema=True)
    
    # Start with empty schema
    validator.schema = Schema(rules={})
    
    # Send data that would violate non-existent rules
    data = {"new_field": torch.randn(32, 64).float()}
    validator(data)
    
    # Check suggestions
    suggestions = validator.suggest_schema_updates()
    assert "new_field" in validator.suggested_rules
    assert len(validator.suggested_rules["new_field"]) > 0
    

def test_single_tensor_validation():
    """Test validation of single tensors."""
    validator = DataValidator()
    
    # Add rules dynamically
    validator.add_rule("default", DType(torch.float32))
    validator.add_rule("default", Range(min_val=-1.0, max_val=1.0))
    
    # Validate single tensor
    tensor = torch.rand(10, 10) * 2 - 1  # Range [-1, 1]
    is_valid, _ = validator(tensor)
    assert is_valid
    
    # Out of range
    bad_tensor = torch.rand(10, 10) * 10
    is_valid, errors = validator(bad_tensor)
    assert not is_valid


if __name__ == "__main__":
    test_basic_validation()
    test_wildcard_shapes()
    test_nan_inf_validation()
    test_custom_validation()
    test_distribution_tracking()
    test_strict_mode()
    test_schema_evolution()
    test_single_tensor_validation()
    print("All DataValidator tests passed!")