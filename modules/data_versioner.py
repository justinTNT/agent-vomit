import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass
from datetime import datetime
import hashlib
import pickle
from pathlib import Path
import json
from collections import defaultdict


@dataclass
class DataVersion:
    """Metadata for a data version."""
    version_id: str
    parent_id: Optional[str]
    timestamp: datetime
    shape: Tuple[int, ...]
    dtype: str
    stats: Dict[str, float]
    metadata: Dict[str, Any]
    hash: str


@dataclass
class DataDiff:
    """Represents differences between two data versions."""
    version_from: str
    version_to: str
    shape_changed: bool
    dtype_changed: bool
    stats_diff: Dict[str, float]
    indices_changed: Optional[Set[Tuple[int, ...]]]
    summary: str


class DataVersioner(nn.Module):
    """
    Version control for datasets with diffing and branching.
    Tracks data lineage and enables reproducible experiments.
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        chunk_size: int = 1000,
        track_deltas: bool = True,
        compression: bool = True,
        max_versions: Optional[int] = None,
        deduplicate: bool = True  # Content-based addressing
    ):
        super().__init__()
        self.storage_path = Path(storage_path) if storage_path else Path("/tmp/data_versions")
        self.chunk_size = chunk_size
        self.track_deltas = track_deltas
        self.compression = compression
        self.max_versions = max_versions
        self.deduplicate = deduplicate
        
        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Version tracking
        self.versions = {}  # version_id -> DataVersion
        self.branches = {"main": None}  # branch_name -> latest_version_id
        self.current_branch = "main"
        self.deltas = {}  # (from_id, to_id) -> delta_data
        
        # Statistics
        self.stats = {
            "commits": 0,
            "branches": 1,
            "total_size": 0,
            "delta_saves": 0
        }
        
    def forward(self, data: torch.Tensor, 
                message: str = "Data update",
                metadata: Optional[Dict[str, Any]] = None) -> str:
        """Commit a new version of the data."""
        # Generate version ID
        version_id = self._generate_version_id(data)
        
        # Check if data already exists
        if version_id in self.versions:
            return version_id
            
        # Get parent version
        parent_id = self.branches[self.current_branch]
        
        # Create version metadata
        version = DataVersion(
            version_id=version_id,
            parent_id=parent_id,
            timestamp=datetime.now(),
            shape=tuple(data.shape),
            dtype=str(data.dtype),
            stats=self._compute_stats(data),
            metadata=metadata or {},
            hash=self._hash_data(data)
        )
        
        # Save data
        if parent_id and self.track_deltas:
            # Try to save as delta
            parent_data = self.load(parent_id)
            if self._can_save_as_delta(parent_data, data):
                self._save_delta(parent_id, version_id, parent_data, data)
                self.stats["delta_saves"] += 1
            else:
                self._save_full(version_id, data)
        else:
            self._save_full(version_id, data)
            
        # Update tracking
        self.versions[version_id] = version
        self.branches[self.current_branch] = version_id
        self.stats["commits"] += 1
        
        # Cleanup old versions if needed
        if self.max_versions:
            self._cleanup_old_versions()
            
        return version_id
    
    def load(self, version_id: str) -> torch.Tensor:
        """Load a specific version of the data."""
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
            
        # Check if we have the full data
        full_path = self._get_storage_path(version_id, "full")
        if full_path.exists():
            return torch.load(full_path)
            
        # Reconstruct from deltas
        return self._reconstruct_from_deltas(version_id)
    
    def diff(self, version_id1: str, version_id2: str) -> DataDiff:
        """Compute differences between two versions."""
        if version_id1 not in self.versions or version_id2 not in self.versions:
            raise ValueError("Version not found")
            
        v1 = self.versions[version_id1]
        v2 = self.versions[version_id2]
        
        # Basic metadata comparison
        diff = DataDiff(
            version_from=version_id1,
            version_to=version_id2,
            shape_changed=v1.shape != v2.shape,
            dtype_changed=v1.dtype != v2.dtype,
            stats_diff={
                key: v2.stats.get(key, 0) - v1.stats.get(key, 0)
                for key in set(v1.stats) | set(v2.stats)
            },
            indices_changed=None,
            summary=""
        )
        
        # If shapes match, compute detailed diff
        if not diff.shape_changed and not diff.dtype_changed:
            data1 = self.load(version_id1)
            data2 = self.load(version_id2)
            
            # Find changed indices
            changed_mask = ~torch.isclose(data1, data2)
            if changed_mask.any():
                changed_indices = torch.nonzero(changed_mask).tolist()
                diff.indices_changed = set(tuple(idx) for idx in changed_indices[:100])  # Limit to 100
                
        # Generate summary
        changes = []
        if diff.shape_changed:
            changes.append(f"shape: {v1.shape} → {v2.shape}")
        if diff.dtype_changed:
            changes.append(f"dtype: {v1.dtype} → {v2.dtype}")
        if diff.stats_diff:
            major_changes = [(k, v) for k, v in diff.stats_diff.items() if abs(v) > 0.1]
            if major_changes:
                changes.append(f"stats: {len(major_changes)} metrics changed")
                
        diff.summary = "; ".join(changes) if changes else "No significant changes"
        
        return diff
    
    def branch(self, branch_name: str, from_version: Optional[str] = None) -> None:
        """Create a new branch."""
        if branch_name in self.branches:
            raise ValueError(f"Branch {branch_name} already exists")
            
        # Branch from current or specified version
        if from_version:
            if from_version not in self.versions:
                raise ValueError(f"Version {from_version} not found")
            self.branches[branch_name] = from_version
        else:
            self.branches[branch_name] = self.branches[self.current_branch]
            
        self.stats["branches"] += 1
    
    def checkout(self, branch_or_version: str) -> torch.Tensor:
        """Checkout a branch or specific version."""
        if branch_or_version in self.branches:
            # It's a branch
            self.current_branch = branch_or_version
            version_id = self.branches[branch_or_version]
            if version_id:
                return self.load(version_id)
            else:
                raise ValueError(f"Branch {branch_or_version} has no commits")
        elif branch_or_version in self.versions:
            # It's a version
            return self.load(branch_or_version)
        else:
            raise ValueError(f"Unknown branch or version: {branch_or_version}")
    
    def merge(self, from_branch: str, strategy: str = "theirs") -> str:
        """Merge another branch into current branch."""
        if from_branch not in self.branches:
            raise ValueError(f"Branch {from_branch} not found")
            
        from_version = self.branches[from_branch]
        to_version = self.branches[self.current_branch]
        
        if not from_version or not to_version:
            raise ValueError("Cannot merge empty branches")
            
        # Simple merge strategies
        if strategy == "theirs":
            # Take their version
            data = self.load(from_version)
        elif strategy == "ours":
            # Keep our version
            data = self.load(to_version)
        elif strategy == "mean":
            # Average the two (only works for same shape/dtype)
            data1 = self.load(from_version)
            data2 = self.load(to_version)
            if data1.shape != data2.shape or data1.dtype != data2.dtype:
                raise ValueError("Cannot average data with different shapes/dtypes")
            data = (data1 + data2) / 2
        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")
            
        # Commit merge result
        metadata = {
            "merge": True,
            "from_branch": from_branch,
            "to_branch": self.current_branch,
            "strategy": strategy
        }
        
        return self.forward(data, f"Merge {from_branch} into {self.current_branch}", metadata)
    
    def get_lineage(self, version_id: str) -> List[str]:
        """Get the lineage (parent chain) of a version."""
        lineage = []
        current = version_id
        
        while current:
            lineage.append(current)
            version = self.versions.get(current)
            if version:
                current = version.parent_id
            else:
                break
                
        return lineage
    
    def _generate_version_id(self, data: torch.Tensor) -> str:
        """Generate unique version ID."""
        if self.deduplicate:
            # Content-based addressing for deduplication
            content = f"{data.shape}:{data.dtype}:{data.flatten()[:100].sum().item():.6f}:{data.numel()}"
        else:
            # Include timestamp for unique versions
            content = f"{data.shape}:{data.dtype}:{data.sum().item():.6f}:{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]
    
    def _hash_data(self, data: torch.Tensor) -> str:
        """Generate hash of tensor data."""
        # Sample data for large tensors
        if data.numel() > 10000:
            indices = torch.randperm(data.numel())[:1000]
            sample = data.flatten()[indices]
        else:
            sample = data.flatten()
            
        content = f"{data.shape}:{data.dtype}:{sample.sum().item():.6f}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def _compute_stats(self, data: torch.Tensor) -> Dict[str, float]:
        """Compute statistics for data."""
        return {
            "mean": data.float().mean().item(),
            "std": data.float().std().item(),
            "min": data.min().item(),
            "max": data.max().item(),
            "norm": data.float().norm().item()
        }
    
    def _get_storage_path(self, version_id: str, suffix: str) -> Path:
        """Get storage path for version data."""
        return self.storage_path / f"{version_id}.{suffix}.pt"
    
    def _save_full(self, version_id: str, data: torch.Tensor) -> None:
        """Save full data."""
        path = self._get_storage_path(version_id, "full")
        torch.save(data, path)
        self.stats["total_size"] += path.stat().st_size
    
    def _save_delta(self, from_id: str, to_id: str, 
                    from_data: torch.Tensor, to_data: torch.Tensor) -> None:
        """Save delta between versions."""
        # Simple delta: store differences
        delta = to_data - from_data
        
        # Sparse representation for efficiency
        nonzero_mask = delta != 0
        if nonzero_mask.sum() < delta.numel() * 0.1:  # Less than 10% changed
            # Store sparse
            indices = torch.nonzero(nonzero_mask)
            values = delta[nonzero_mask]
            delta_data = {"indices": indices, "values": values, "shape": delta.shape}
        else:
            # Store dense
            delta_data = {"dense": delta}
            
        path = self._get_storage_path(to_id, "delta")
        torch.save(delta_data, path)
        self.deltas[(from_id, to_id)] = delta_data
    
    def _can_save_as_delta(self, from_data: torch.Tensor, to_data: torch.Tensor) -> bool:
        """Check if data can be saved as delta."""
        return (from_data.shape == to_data.shape and 
                from_data.dtype == to_data.dtype)
    
    def _reconstruct_from_deltas(self, version_id: str) -> torch.Tensor:
        """Reconstruct data from delta chain."""
        # Find path from a full version
        lineage = self.get_lineage(version_id)
        
        # Find first full version in lineage
        full_version_idx = None
        for i, vid in enumerate(lineage):
            if self._get_storage_path(vid, "full").exists():
                full_version_idx = i
                break
                
        if full_version_idx is None:
            raise ValueError("No full version found in lineage")
            
        # Load base version
        data = torch.load(self._get_storage_path(lineage[full_version_idx], "full"))
        
        # Apply deltas (if any needed)
        if full_version_idx > 0:
            for i in range(full_version_idx - 1, -1, -1):
                from_id = lineage[i + 1]
                to_id = lineage[i]
                
                delta_path = self._get_storage_path(to_id, "delta")
                if delta_path.exists():
                    delta_data = torch.load(delta_path)
                    
                    if "dense" in delta_data:
                        data = data + delta_data["dense"]
                    else:
                        # Sparse delta
                        indices = delta_data["indices"]
                        values = delta_data["values"]
                        for idx, val in zip(indices, values):
                            data[tuple(idx)] += val
                            
        return data
    
    def _cleanup_old_versions(self) -> None:
        """Remove old versions to maintain size limit."""
        if len(self.versions) <= self.max_versions:
            return
            
        # Get versions sorted by timestamp
        sorted_versions = sorted(
            self.versions.items(),
            key=lambda x: x[1].timestamp
        )
        
        # Keep important versions (branch heads, tagged)
        important = set()
        for branch, version_id in self.branches.items():
            if version_id:
                important.add(version_id)
                
        # Remove oldest non-important versions
        to_remove = []
        for version_id, version in sorted_versions:
            if len(self.versions) - len(to_remove) <= self.max_versions:
                break
            if version_id not in important:
                to_remove.append(version_id)
                
        for version_id in to_remove:
            # Remove files
            for suffix in ["full", "delta"]:
                path = self._get_storage_path(version_id, suffix)
                if path.exists():
                    path.unlink()
                    
            del self.versions[version_id]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get versioning statistics."""
        return {
            "stats": self.stats,
            "versions": len(self.versions),
            "branches": list(self.branches.keys()),
            "current_branch": self.current_branch,
            "storage_size_mb": self.stats["total_size"] / (1024 * 1024)
        }