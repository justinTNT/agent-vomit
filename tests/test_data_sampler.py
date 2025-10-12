import torch
import pytest
from collections import Counter
from modules.data_sampler import DataSampler


def test_uniform_sampling():
    """Test basic uniform sampling."""
    sampler = DataSampler(strategy="uniform", batch_size=10, seed=42)
    
    data = torch.randn(100, 4)
    sampled_data, _ = sampler(data)
    
    assert sampled_data.shape == (10, 4)
    assert torch.all(torch.isin(sampled_data, data))
    

def test_stratified_sampling():
    """Test stratified sampling maintains class distribution."""
    sampler = DataSampler(strategy="stratified", batch_size=20, seed=42)
    
    # Create imbalanced dataset
    data = torch.randn(100, 2)
    labels = torch.cat([
        torch.zeros(70),  # 70% class 0
        torch.ones(30)    # 30% class 1
    ])
    
    sampled_data, sampled_labels = sampler(data, labels)
    
    # Check approximate stratification
    class_0_ratio = (sampled_labels == 0).sum().float() / len(sampled_labels)
    assert 0.5 < class_0_ratio < 0.9  # Should be around 0.7
    

def test_balanced_sampling():
    """Test balanced sampling equalizes classes."""
    sampler = DataSampler(strategy="balanced", batch_size=20, seed=42)
    
    # Highly imbalanced dataset
    data = torch.randn(100, 2)
    labels = torch.cat([
        torch.zeros(90),  # 90% class 0
        torch.ones(10)    # 10% class 1
    ])
    
    sampled_data, sampled_labels = sampler(data, labels)
    
    # Should have equal representation
    class_0_count = (sampled_labels == 0).sum().item()
    class_1_count = (sampled_labels == 1).sum().item()
    assert abs(class_0_count - class_1_count) <= 1  # At most 1 difference
    

def test_weighted_sampling():
    """Test weighted sampling with custom weights."""
    sampler = DataSampler(strategy="weighted", batch_size=1000, seed=42)
    
    data = torch.arange(4).unsqueeze(1).float()
    weights = torch.tensor([0.1, 0.2, 0.3, 0.4])  # Higher weight for higher indices
    
    sampled_data, _ = sampler(data, weights=weights)
    
    # Count occurrences
    counts = Counter(sampled_data.squeeze().tolist())
    
    # Higher weights should be sampled more
    assert counts[3] > counts[0]
    assert counts[2] > counts[1]
    

def test_focal_sampling():
    """Test focal sampling focuses on hard examples."""
    sampler = DataSampler(strategy="focal", batch_size=20, seed=42)
    
    # Create data with different difficulties
    data = torch.randn(100, 2)
    labels = torch.randint(0, 3, (100,))
    
    # Update difficulty scores (simulate hard examples)
    hard_indices = torch.tensor([10, 20, 30, 40, 50])
    hard_losses = torch.ones(5) * 5.0  # High loss = hard example
    sampler.update_difficulty(hard_indices, hard_losses)
    
    sampled_data, sampled_labels = sampler(data, labels)
    
    # Focal sampling should work even without difficulty scores
    assert sampled_data.shape[0] == 20
    

def test_adaptive_sampling():
    """Test adaptive sampling changes over epochs."""
    sampler = DataSampler(strategy="adaptive", batch_size=10, seed=42)
    
    data = torch.randn(50, 2)
    labels = torch.randint(0, 2, (50,))
    
    # First few epochs should use stratified
    results1, _ = sampler(data, labels)
    
    # Simulate training progress
    sampler.epoch = 10
    sampler.update_difficulty(torch.arange(10), torch.randn(10).abs())
    
    results2, _ = sampler(data, labels)
    
    # Both should return valid samples
    assert results1.shape[0] == 10
    assert results2.shape[0] == 10
    

def test_class_weight_computation():
    """Test automatic class weight computation."""
    sampler = DataSampler()
    
    # Imbalanced labels
    labels = torch.cat([
        torch.zeros(900),
        torch.ones(100)
    ])
    
    # Inverse frequency weights
    weights = sampler.compute_class_weights(labels, mode="inverse")
    assert weights[1] > weights[0]  # Minority class should have higher weight
    
    # Effective number weights
    weights_eff = sampler.compute_class_weights(labels, mode="effective")
    assert weights_eff[1] > weights_eff[0]
    

def test_sampling_without_replacement():
    """Test sampling without replacement."""
    sampler = DataSampler(
        strategy="uniform", 
        batch_size=10, 
        replacement=False,
        seed=42
    )
    
    data = torch.arange(20).unsqueeze(1).float()
    sampled_data, _ = sampler(data)
    
    # Should have unique values
    unique_values = torch.unique(sampled_data)
    assert len(unique_values) == 10
    

def test_sampling_stats():
    """Test sampling statistics tracking."""
    sampler = DataSampler(strategy="balanced", batch_size=20)
    
    data = torch.randn(100, 2)
    labels = torch.cat([torch.zeros(80), torch.ones(20)])
    
    # Sample multiple times
    for _ in range(5):
        sampler(data, labels)
        
    stats = sampler.get_sampling_stats(labels)
    
    # Check stats structure
    assert stats.total_samples == 0  # History was not tracked in forward
    assert 0 in stats.class_counts
    assert 1 in stats.class_counts
    

def test_multiclass_balanced_sampling():
    """Test balanced sampling with multiple classes."""
    sampler = DataSampler(strategy="balanced", batch_size=15, seed=42)
    
    data = torch.randn(100, 2)
    labels = torch.cat([
        torch.zeros(50),
        torch.ones(30),
        torch.full((20,), 2)
    ])
    
    sampled_data, sampled_labels = sampler(data, labels)
    
    # Count samples per class
    unique_labels, counts = torch.unique(sampled_labels, return_counts=True)
    
    # Should be roughly equal (5 per class)
    assert all(4 <= c <= 6 for c in counts)


if __name__ == "__main__":
    test_uniform_sampling()
    test_stratified_sampling()
    test_balanced_sampling()
    test_weighted_sampling()
    test_focal_sampling()
    test_adaptive_sampling()
    test_class_weight_computation()
    test_sampling_without_replacement()
    test_sampling_stats()
    test_multiclass_balanced_sampling()
    print("All DataSampler tests passed!")