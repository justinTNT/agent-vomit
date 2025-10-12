import torch
import pytest
import time
from datetime import datetime, timedelta
from modules.feature_store import FeatureStore, FeatureCompute


class MeanFeature(FeatureCompute):
    """Compute mean across dimension."""
    def __init__(self):
        super().__init__("mean", "1.0")
        
    def compute(self, inputs):
        return inputs["data"].mean(dim=-1)


class StdFeature(FeatureCompute):
    """Compute standard deviation."""
    def __init__(self):
        super().__init__("std", "1.0")
        
    def compute(self, inputs):
        return inputs["data"].std(dim=-1)


class NormalizedFeature(FeatureCompute):
    """Normalize using mean and std."""
    def __init__(self):
        super().__init__("normalized", "1.0")
        self.dependencies = ["mean", "std"]
        
    def compute(self, inputs):
        data = inputs["data"]
        mean = inputs["mean"].unsqueeze(-1)
        std = inputs["std"].unsqueeze(-1)
        return (data - mean) / (std + 1e-6)


class EmbeddingFeature(FeatureCompute):
    """Generate embeddings."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__("embeddings", "1.0")
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def compute(self, inputs):
        return self.linear(inputs["data"])


def test_basic_feature_computation():
    """Test basic feature registration and computation."""
    store = FeatureStore()
    
    # Register features
    store.register_feature(MeanFeature())
    store.register_feature(StdFeature())
    
    # Compute features
    data = torch.randn(32, 100)
    results = store.forward(["mean", "std"], {"data": data})
    
    assert "mean" in results
    assert "std" in results
    assert results["mean"].shape == (32,)
    assert results["std"].shape == (32,)
    
    # Verify computation
    expected_mean = data.mean(dim=-1)
    assert torch.allclose(results["mean"], expected_mean)
    

def test_feature_dependencies():
    """Test features with dependencies."""
    store = FeatureStore()
    
    # Register features with dependencies
    store.register_feature(MeanFeature())
    store.register_feature(StdFeature())
    store.register_feature(NormalizedFeature())
    
    # Compute dependent feature
    data = torch.randn(16, 50)
    result = store.get_feature("normalized", {"data": data})
    
    # Check normalization
    assert result.shape == data.shape
    assert torch.abs(result.mean()).item() < 0.1
    assert torch.abs(result.std() - 1.0).mean().item() < 0.1
    

def test_caching():
    """Test feature caching mechanism."""
    store = FeatureStore(cache_size=10)
    
    store.register_feature(MeanFeature())
    
    # First computation
    data = torch.ones(10, 20)
    result1 = store.get_feature("mean", {"data": data})
    
    # Second computation with same data should hit cache
    result2 = store.get_feature("mean", {"data": data})
    
    stats = store.get_statistics()
    assert stats["stats"]["cache_hits"] == 1
    assert stats["stats"]["cache_misses"] == 1
    assert torch.equal(result1, result2)
    

def test_versioning():
    """Test feature versioning."""
    store = FeatureStore(enable_versioning=True)
    
    # Register initial version
    store.register_feature(MeanFeature())
    
    # Register new version
    class MeanFeatureV2(FeatureCompute):
        def __init__(self):
            super().__init__("mean", "2.0")
            
        def compute(self, inputs):
            # Different computation
            return inputs["data"].mean(dim=-1) * 2.0
    
    # This would be done differently in practice, but for testing:
    old_feature = store.features["mean"]
    store.register_feature(MeanFeatureV2())
    
    # Check version history
    assert len(store.versions["mean"]) == 2
    assert store.versions["mean"][0].version == "1.0"
    assert store.versions["mean"][1].version == "2.0"
    

def test_lineage_tracking():
    """Test feature lineage tracking."""
    store = FeatureStore(enable_lineage=True)
    
    # Create feature chain
    store.register_feature(MeanFeature())
    store.register_feature(StdFeature())
    store.register_feature(NormalizedFeature())
    
    # Get lineage
    lineage = store.get_lineage("normalized")
    
    assert lineage["feature"] == "normalized"
    assert set(lineage["direct_dependencies"]) == {"mean", "std"}
    assert set(lineage["all_dependencies"]) == {"mean", "std"}
    

def test_ttl_cache_expiry():
    """Test cache TTL expiration."""
    store = FeatureStore(default_ttl=0.1)  # 100ms TTL
    
    store.register_feature(MeanFeature())
    
    # Compute and cache
    data = torch.randn(5, 10)
    result1 = store.get_feature("mean", {"data": data})
    
    # Immediate access should hit cache
    result2 = store.get_feature("mean", {"data": data})
    assert store.stats["cache_hits"] == 1
    
    # Wait for expiry
    time.sleep(0.15)
    
    # Should recompute
    result3 = store.get_feature("mean", {"data": data})
    assert store.stats["cache_misses"] == 2
    

def test_save_load_features():
    """Test saving and loading features."""
    store = FeatureStore()
    
    store.register_feature(MeanFeature())
    store.register_feature(StdFeature())
    
    # Compute and save
    data = torch.randn(8, 16)
    store.save_features(["mean", "std"], {"data": data}, "/tmp/test_features.pt")
    
    # Load features
    loaded = store.load_features("/tmp/test_features.pt")
    
    assert "mean" in loaded
    assert "std" in loaded
    assert loaded["mean"].shape == (8,)
    

def test_complex_dependency_graph():
    """Test complex feature dependency graphs."""
    store = FeatureStore(enable_lineage=True)
    
    # Create a more complex graph
    class FeatureA(FeatureCompute):
        def __init__(self):
            super().__init__("A", "1.0")
        def compute(self, inputs):
            return inputs["data"] * 2
            
    class FeatureB(FeatureCompute):
        def __init__(self):
            super().__init__("B", "1.0")
            self.dependencies = ["A"]
        def compute(self, inputs):
            return inputs["A"] + 1
            
    class FeatureC(FeatureCompute):
        def __init__(self):
            super().__init__("C", "1.0")
            self.dependencies = ["A"]
        def compute(self, inputs):
            return inputs["A"] * 3
            
    class FeatureD(FeatureCompute):
        def __init__(self):
            super().__init__("D", "1.0")
            self.dependencies = ["B", "C"]
        def compute(self, inputs):
            return inputs["B"] + inputs["C"]
    
    store.register_feature(FeatureA())
    store.register_feature(FeatureB())
    store.register_feature(FeatureC())
    store.register_feature(FeatureD())
    
    # Compute final feature
    data = torch.tensor([1.0, 2.0, 3.0])
    result = store.get_feature("D", {"data": data})
    
    # Verify computation: D = B + C = (A + 1) + (A * 3) = (data*2 + 1) + (data*2 * 3)
    expected = (data * 2 + 1) + (data * 2 * 3)
    assert torch.allclose(result, expected)
    
    # Check dependency graph
    lineage = store.get_lineage("D")
    assert set(lineage["all_dependencies"]) == {"A", "B", "C"}


if __name__ == "__main__":
    test_basic_feature_computation()
    test_feature_dependencies()
    test_caching()
    test_versioning()
    test_lineage_tracking()
    test_ttl_cache_expiry()
    test_save_load_features()
    test_complex_dependency_graph()
    print("All FeatureStore tests passed!")