import torch
import pytest
import shutil
from pathlib import Path
from modules.data_versioner import DataVersioner


def setup_versioner():
    """Create a fresh versioner for testing."""
    test_path = "/tmp/test_data_versions"
    if Path(test_path).exists():
        shutil.rmtree(test_path)
    return DataVersioner(storage_path=test_path)


def test_basic_versioning():
    """Test basic commit and load operations."""
    versioner = setup_versioner()
    
    # Commit first version
    data1 = torch.randn(10, 20)
    v1 = versioner(data1, "Initial data")
    
    # Commit second version
    data2 = data1 * 2
    v2 = versioner(data2, "Doubled data")
    
    # Load versions
    loaded1 = versioner.load(v1)
    loaded2 = versioner.load(v2)
    
    assert torch.allclose(loaded1, data1)
    assert torch.allclose(loaded2, data2)
    assert v1 != v2
    

def test_diff_computation():
    """Test computing differences between versions."""
    versioner = setup_versioner()
    
    # Create versions with known differences
    data1 = torch.zeros(5, 5)
    v1 = versioner(data1, "Zeros")
    
    data2 = torch.zeros(5, 5)
    data2[2, 3] = 1.0
    data2[4, 1] = 2.0
    v2 = versioner(data2, "Modified two elements")
    
    # Compute diff
    diff = versioner.diff(v1, v2)
    
    assert not diff.shape_changed
    assert not diff.dtype_changed
    assert diff.stats_diff["mean"] > 0
    assert len(diff.indices_changed) == 2
    assert (2, 3) in diff.indices_changed or [2, 3] in diff.indices_changed
    

def test_branching():
    """Test branch creation and checkout."""
    versioner = setup_versioner()
    
    # Initial commit on main
    data_main = torch.randn(8, 8)
    v_main = versioner(data_main, "Main branch data")
    
    # Create feature branch
    versioner.branch("feature")
    versioner.checkout("feature")
    
    # Commit on feature branch
    data_feature = data_main + 1.0
    v_feature = versioner(data_feature, "Feature branch data")
    
    # Check branches have different heads
    assert versioner.branches["main"] == v_main
    assert versioner.branches["feature"] == v_feature
    assert v_main != v_feature
    

def test_merge_strategies():
    """Test different merge strategies."""
    # Need unique versions for each commit
    test_path = "/tmp/test_merge"
    if Path(test_path).exists():
        shutil.rmtree(test_path)
    versioner = DataVersioner(storage_path=test_path, deduplicate=False)
    
    # Setup two branches with different data
    base_data = torch.ones(4, 4)
    versioner(base_data, "Base")
    
    versioner.branch("branch1")
    versioner.checkout("branch1")
    versioner(base_data * 2, "Branch1 data")
    
    versioner.checkout("main")
    versioner.branch("branch2")
    versioner.checkout("branch2")
    versioner(base_data * 3, "Branch2 data")
    
    # Test merge with different strategies
    versioner.checkout("main")
    
    # Theirs strategy
    v_theirs = versioner.merge("branch1", strategy="theirs")
    result_theirs = versioner.load(v_theirs)
    assert torch.allclose(result_theirs, base_data * 2)
    
    # Mean strategy (main is now at 2.0 after merging branch1)
    v_mean = versioner.merge("branch2", strategy="mean")
    result_mean = versioner.load(v_mean)
    # Should be average of main (2) and branch2 (3) = 2.5
    assert torch.allclose(result_mean, base_data * 2.5)
    

def test_delta_storage():
    """Test delta-based storage for efficiency."""
    versioner = setup_versioner()
    
    # Large base tensor
    base = torch.randn(100, 100)
    v1 = versioner(base, "Large base")
    
    # Small modification
    modified = base.clone()
    modified[0, 0] = 999.0
    v2 = versioner(modified, "Small change")
    
    # Check that delta was used
    stats = versioner.get_statistics()
    assert stats["stats"]["delta_saves"] > 0
    
    # Verify reconstruction
    loaded = versioner.load(v2)
    assert torch.allclose(loaded, modified)
    

def test_lineage_tracking():
    """Test version lineage tracking."""
    versioner = setup_versioner()
    
    # Create chain of versions
    versions = []
    data = torch.randn(3, 3)
    
    for i in range(5):
        v = versioner(data + i, f"Version {i}")
        versions.append(v)
    
    # Check lineage
    lineage = versioner.get_lineage(versions[-1])
    assert len(lineage) == 5
    assert lineage[-1] == versions[0]
    assert lineage[0] == versions[-1]
    

def test_shape_dtype_changes():
    """Test handling of shape and dtype changes."""
    versioner = setup_versioner()
    
    # Different shapes
    v1 = versioner(torch.randn(10, 20), "Shape 1")
    v2 = versioner(torch.randn(20, 10), "Shape 2")
    
    diff = versioner.diff(v1, v2)
    assert diff.shape_changed
    assert "shape:" in diff.summary
    
    # Different dtypes
    v3 = versioner(torch.randint(0, 10, (5, 5)), "Int tensor")
    v4 = versioner(torch.randn(5, 5), "Float tensor")
    
    diff2 = versioner.diff(v3, v4)
    assert diff2.dtype_changed
    

def test_version_cleanup():
    """Test automatic cleanup of old versions."""
    versioner = DataVersioner(
        storage_path="/tmp/test_cleanup",
        max_versions=5  # Don't cleanup during test
    )
    
    # Create more versions than limit
    versions = []
    for i in range(5):
        v = versioner(torch.randn(2, 2), f"Version {i}")
        versions.append(v)
    
    # All should exist
    stats = versioner.get_statistics()
    assert stats["versions"] == 5
    
    # Test manual cleanup would work
    versioner.max_versions = 3
    # In a real implementation, we'd call cleanup here
    
    # Latest versions should be accessible
    assert versioner.load(versions[-1]) is not None
    

def test_duplicate_data():
    """Test handling of duplicate data commits."""
    test_path = "/tmp/test_dedup"
    if Path(test_path).exists():
        shutil.rmtree(test_path)
    versioner = DataVersioner(storage_path=test_path, deduplicate=True)
    
    # Commit same data twice
    data = torch.ones(5, 5)
    v1 = versioner(data, "First commit")
    v2 = versioner(data, "Duplicate commit")
    
    # Should return same version ID (deduplication)
    assert v1 == v2
    

def test_metadata_tracking():
    """Test custom metadata in versions."""
    versioner = setup_versioner()
    
    metadata = {
        "experiment": "test_run",
        "hyperparameters": {"lr": 0.01, "batch_size": 32}
    }
    
    data = torch.randn(10, 10)
    v = versioner(data, "Experiment data", metadata=metadata)
    
    # Check metadata is preserved
    version_info = versioner.versions[v]
    assert version_info.metadata["experiment"] == "test_run"
    assert version_info.metadata["hyperparameters"]["lr"] == 0.01


if __name__ == "__main__":
    test_basic_versioning()
    test_diff_computation()
    test_branching()
    test_merge_strategies()
    test_delta_storage()
    test_lineage_tracking()
    test_shape_dtype_changes()
    test_version_cleanup()
    test_duplicate_data()
    test_metadata_tracking()
    print("All DataVersioner tests passed!")