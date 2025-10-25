import torch
import torch.nn as nn
import warnings
import hashlib
import pickle
import os
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Any


@dataclass
class DataVersion:
    """Metadata for a data version."""
    version_id: str
    parent_id: Optional[str]
    timestamp: datetime
    message: str
    data_hash: str
    shape: tuple
    dtype: str
    device: str
    statistics: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class DataDiff:
    """Difference between two data versions."""
    version_id1: str
    version_id2: str
    shape_changed: bool
    dtype_changed: bool
    num_changed_elements: int
    total_elements: int
    change_percentage: float
    mean_absolute_diff: float
    max_absolute_diff: float
    statistics_diff: Dict[str, float]


class DataVersioner(nn.Module):
    def __init__(self, 
                 storage_path: str = "./versions",  # Where to store versions
                 **kwargs):                         # REQUIRED: Accept extra kwargs
        super().__init__()
        
        # Log unknown parameters
        if kwargs:
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Version tracking
        self.versions = {}  # version_id -> DataVersion
        self.version_tree = {}  # parent_id -> List[child_ids]
        self.current_version = None
        
        # Load existing versions if any
        self._load_version_metadata()
    
    def __call__(self, data: torch.Tensor, message: str = "") -> str:
        """Make the class callable - shorthand for version()"""
        return self.version(data, message)
    
    def version(self, data: torch.Tensor, message: str = "") -> str:
        """
        Create a new version of the data.
        
        Args:
            data: The tensor to version
            message: Commit message describing the changes
        
        Returns:
            version_id: Unique identifier for this version
        """
        # Compute data hash
        data_hash = self._compute_hash(data)
        
        # Check if this exact data already exists
        for vid, version in self.versions.items():
            if version.data_hash == data_hash:
                self.current_version = vid
                return vid
        
        # Generate version ID
        version_id = self._generate_version_id(data_hash)
        
        # Compute statistics
        statistics = self._compute_statistics(data)
        
        # Create version metadata
        version = DataVersion(
            version_id=version_id,
            parent_id=self.current_version,
            timestamp=datetime.now(),
            message=message,
            data_hash=data_hash,
            shape=tuple(data.shape),
            dtype=str(data.dtype),
            device=str(data.device),
            statistics=statistics,
            metadata={}
        )
        
        # Store data and metadata
        self._store_version(version_id, data, version)
        
        # Update version tracking
        self.versions[version_id] = version
        if version.parent_id:
            if version.parent_id not in self.version_tree:
                self.version_tree[version.parent_id] = []
            self.version_tree[version.parent_id].append(version_id)
        
        self.current_version = version_id
        
        # Save metadata
        self._save_version_metadata()
        
        return version_id
    
    def load(self, version_id: str) -> torch.Tensor:
        """
        Load a specific version of the data.
        
        Args:
            version_id: The version to load
        
        Returns:
            The data tensor for that version
        """
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        # Load data from storage
        data_path = self.storage_path / f"{version_id}.pt"
        if not data_path.exists():
            raise FileNotFoundError(f"Data file for version {version_id} not found")
        
        data = torch.load(data_path, map_location='cpu')
        
        # Move to original device if possible
        version = self.versions[version_id]
        device = version.device
        if device != 'cpu' and torch.cuda.is_available():
            try:
                data = data.to(device)
            except:
                pass
        
        self.current_version = version_id
        return data
    
    def diff(self, version_id1: str, version_id2: str) -> DataDiff:
        """
        Compare two versions and return their differences.
        
        Args:
            version_id1: First version
            version_id2: Second version
        
        Returns:
            DataDiff object describing the differences
        """
        # Load both versions
        data1 = self.load(version_id1)
        data2 = self.load(version_id2)
        
        version1 = self.versions[version_id1]
        version2 = self.versions[version_id2]
        
        # Check basic properties
        shape_changed = version1.shape != version2.shape
        dtype_changed = version1.dtype != version2.dtype
        
        # Compute differences
        if shape_changed or dtype_changed:
            # Can't directly compare - just report metadata differences
            num_changed_elements = -1
            total_elements = max(torch.prod(torch.tensor(version1.shape)).item(),
                               torch.prod(torch.tensor(version2.shape)).item())
            change_percentage = 100.0
            mean_absolute_diff = float('inf')
            max_absolute_diff = float('inf')
        else:
            # Compare element-wise
            diff_mask = ~torch.isclose(data1, data2)
            num_changed_elements = int(diff_mask.sum().item())
            total_elements = int(data1.numel())
            change_percentage = (num_changed_elements / total_elements) * 100
            
            if num_changed_elements > 0:
                absolute_diff = torch.abs(data1 - data2)
                mean_absolute_diff = float(absolute_diff[diff_mask].mean().item())
                max_absolute_diff = float(absolute_diff.max().item())
            else:
                mean_absolute_diff = 0.0
                max_absolute_diff = 0.0
        
        # Compare statistics
        statistics_diff = {}
        for key in set(version1.statistics.keys()) | set(version2.statistics.keys()):
            val1 = version1.statistics.get(key, 0.0)
            val2 = version2.statistics.get(key, 0.0)
            statistics_diff[key] = val2 - val1
        
        return DataDiff(
            version_id1=version_id1,
            version_id2=version_id2,
            shape_changed=shape_changed,
            dtype_changed=dtype_changed,
            num_changed_elements=num_changed_elements,
            total_elements=total_elements,
            change_percentage=change_percentage,
            mean_absolute_diff=mean_absolute_diff,
            max_absolute_diff=max_absolute_diff,
            statistics_diff=statistics_diff
        )
    
    def get_history(self, version_id: Optional[str] = None) -> List[DataVersion]:
        """Get version history from a specific version back to root."""
        if version_id is None:
            version_id = self.current_version
        
        if version_id is None or version_id not in self.versions:
            return []
        
        history = []
        current = version_id
        
        while current is not None:
            version = self.versions[current]
            history.append(version)
            current = version.parent_id
        
        return history
    
    def get_branches(self, version_id: str) -> List[str]:
        """Get all branches (children) from a specific version."""
        return self.version_tree.get(version_id, [])
    
    def _compute_hash(self, data: torch.Tensor) -> str:
        """Compute hash of tensor data."""
        # Convert to bytes for hashing
        data_bytes = data.cpu().numpy().tobytes()
        return hashlib.sha256(data_bytes).hexdigest()[:16]
    
    def _generate_version_id(self, data_hash: str) -> str:
        """Generate unique version ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{data_hash[:8]}"
    
    def _compute_statistics(self, data: torch.Tensor) -> Dict[str, float]:
        """Compute statistical summary of the data."""
        return {
            'mean': float(data.mean().item()),
            'std': float(data.std().item()) if data.numel() > 1 else 0.0,
            'min': float(data.min().item()),
            'max': float(data.max().item()),
            'norm': float(data.norm().item()),
            'sparsity': float((data == 0).float().mean().item()),
        }
    
    def _store_version(self, version_id: str, data: torch.Tensor, version: DataVersion):
        """Store version data and metadata to disk."""
        # Save tensor data
        data_path = self.storage_path / f"{version_id}.pt"
        torch.save(data.cpu(), data_path)
        
        # Save metadata
        meta_path = self.storage_path / f"{version_id}.meta"
        with open(meta_path, 'wb') as f:
            pickle.dump(version, f)
    
    def _save_version_metadata(self):
        """Save version tracking metadata."""
        meta_path = self.storage_path / "versions.meta"
        metadata = {
            'versions': self.versions,
            'version_tree': self.version_tree,
            'current_version': self.current_version
        }
        with open(meta_path, 'wb') as f:
            pickle.dump(metadata, f)
    
    def _load_version_metadata(self):
        """Load existing version metadata if available."""
        meta_path = self.storage_path / "versions.meta"
        if meta_path.exists():
            try:
                with open(meta_path, 'rb') as f:
                    metadata = pickle.load(f)
                self.versions = metadata.get('versions', {})
                self.version_tree = metadata.get('version_tree', {})
                self.current_version = metadata.get('current_version', None)
            except:
                # If loading fails, start fresh
                pass
    
    def forward(self, *args, **kwargs):
        """Forward method for nn.Module compatibility. Calls version()."""
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            return self.version(args[0], args[1] if len(args) > 1 else "")
        raise ValueError("DataVersioner.forward expects a tensor as first argument")