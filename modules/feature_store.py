import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass
from datetime import datetime
import json
import pickle
from pathlib import Path
import hashlib
from collections import OrderedDict


@dataclass
class FeatureVersion:
    """Metadata for a feature version."""
    version: str
    timestamp: datetime
    shape: Tuple[int, ...]
    dtype: str
    compute_hash: str
    dependencies: List[str]
    metadata: Dict[str, Any]


class FeatureCompute:
    """Base class for feature computation."""
    def __init__(self, name: str, version: str = "1.0"):
        self.name = name
        self.version = version
        self.dependencies = []
        
    def compute(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError
        
    def get_hash(self) -> str:
        """Get hash of compute function for versioning."""
        # Simple hash based on class name and version
        content = f"{self.__class__.__name__}:{self.version}"
        return hashlib.md5(content.encode()).hexdigest()[:8]


class FeatureStore(nn.Module):
    """
    Centralized feature computation and serving with versioning.
    Ensures temporal consistency between training and inference.
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        cache_size: int = 1000,
        enable_versioning: bool = True,
        enable_lineage: bool = True,
        default_ttl: Optional[int] = None  # Time to live in seconds
    ):
        super().__init__()
        self.storage_path = Path(storage_path) if storage_path else None
        self.cache_size = cache_size
        self.enable_versioning = enable_versioning
        self.enable_lineage = enable_lineage
        self.default_ttl = default_ttl
        
        # Feature registry
        self.features = OrderedDict()  # name -> FeatureCompute
        self.versions = {}  # name -> List[FeatureVersion]
        self.current_versions = {}  # name -> version_id
        
        # Cache
        self.cache = OrderedDict()
        self.cache_timestamps = {}
        
        # Lineage tracking
        self.lineage_graph = {}  # feature -> dependencies
        self.compute_graph = {}  # Topological order for computation
        
        # Statistics
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "computes": 0,
            "version_conflicts": 0
        }
        
    def register_feature(self, compute: FeatureCompute) -> None:
        """Register a feature computation."""
        name = compute.name
        self.features[name] = compute
        
        # Create initial version
        if self.enable_versioning:
            version = FeatureVersion(
                version=compute.version,
                timestamp=datetime.now(),
                shape=None,  # Will be set on first compute
                dtype=None,
                compute_hash=compute.get_hash(),
                dependencies=compute.dependencies,
                metadata={}
            )
            
            if name not in self.versions:
                self.versions[name] = []
            self.versions[name].append(version)
            self.current_versions[name] = len(self.versions[name]) - 1
            
        # Update lineage graph
        if self.enable_lineage:
            self.lineage_graph[name] = compute.dependencies
            self._update_compute_graph()
    
    def forward(self, 
                feature_names: List[str],
                inputs: Optional[Dict[str, torch.Tensor]] = None,
                version: Optional[str] = None,
                timestamp: Optional[datetime] = None) -> Dict[str, torch.Tensor]:
        """Compute or retrieve features."""
        if inputs is None:
            inputs = {}
            
        results = {}
        
        for name in feature_names:
            # Check cache first
            cache_key = self._get_cache_key(name, inputs, version)
            if cache_key in self.cache:
                if self._is_cache_valid(cache_key):
                    self.stats["cache_hits"] += 1
                    results[name] = self.cache[cache_key]
                    continue
                    
            # Cache miss - compute feature
            self.stats["cache_misses"] += 1
            
            # Get all dependencies first
            if self.enable_lineage and name in self.lineage_graph:
                dep_names = self._get_all_dependencies(name)
                dep_results = self.forward(dep_names, inputs, version, timestamp)
                inputs.update(dep_results)
                
            # Compute feature
            if name not in self.features:
                raise ValueError(f"Feature {name} not registered")
                
            compute = self.features[name]
            result = compute.compute(inputs)
            self.stats["computes"] += 1
            
            # Update version metadata
            if self.enable_versioning and result is not None:
                self._update_version_metadata(name, result)
                
            # Cache result
            self._cache_result(cache_key, result)
            
            results[name] = result
            
        return results
    
    def get_feature(self, name: str, 
                    inputs: Optional[Dict[str, torch.Tensor]] = None,
                    version: Optional[str] = None) -> torch.Tensor:
        """Get a single feature."""
        results = self.forward([name], inputs, version)
        return results[name]
    
    def _get_cache_key(self, name: str, inputs: Dict[str, torch.Tensor], version: Optional[str]) -> str:
        """Generate cache key for feature."""
        # Include input tensors in key (using shapes and stats)
        input_sig = []
        for k in sorted(inputs.keys()):
            v = inputs[k]
            if isinstance(v, torch.Tensor):
                sig = f"{k}:{v.shape}:{v.dtype}:{v.mean().item():.4f}"
                input_sig.append(sig)
                
        version_str = version or self.features[name].version
        key = f"{name}:v{version_str}:{':'.join(input_sig)}"
        return key
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached value is still valid."""
        if self.default_ttl is None:
            return True
            
        if cache_key not in self.cache_timestamps:
            return False
            
        age = (datetime.now() - self.cache_timestamps[cache_key]).total_seconds()
        return age < self.default_ttl
    
    def _cache_result(self, cache_key: str, result: torch.Tensor) -> None:
        """Cache computation result."""
        # Implement LRU eviction
        if len(self.cache) >= self.cache_size:
            # Remove oldest
            self.cache.popitem(last=False)
            if cache_key in self.cache_timestamps:
                oldest_key = next(iter(self.cache_timestamps))
                del self.cache_timestamps[oldest_key]
                
        self.cache[cache_key] = result.detach() if isinstance(result, torch.Tensor) else result
        self.cache_timestamps[cache_key] = datetime.now()
    
    def _update_version_metadata(self, name: str, result: torch.Tensor) -> None:
        """Update version metadata with actual shape/dtype."""
        if name in self.current_versions:
            idx = self.current_versions[name]
            version = self.versions[name][idx]
            if version.shape is None:
                version.shape = tuple(result.shape)
                version.dtype = str(result.dtype)
    
    def _get_all_dependencies(self, name: str) -> List[str]:
        """Get all transitive dependencies of a feature."""
        visited = set()
        deps = []
        
        def traverse(feat):
            if feat in visited:
                return
            visited.add(feat)
            
            if feat in self.lineage_graph:
                for dep in self.lineage_graph[feat]:
                    traverse(dep)
                    if dep not in deps:
                        deps.append(dep)
                        
        traverse(name)
        return deps
    
    def _update_compute_graph(self) -> None:
        """Update topological ordering of compute graph."""
        # Simple topological sort
        visited = set()
        order = []
        
        def visit(node):
            if node in visited:
                return
            visited.add(node)
            
            if node in self.lineage_graph:
                for dep in self.lineage_graph[node]:
                    visit(dep)
                    
            order.append(node)
            
        for node in self.lineage_graph:
            visit(node)
            
        self.compute_graph = order
    
    def save_features(self, feature_names: List[str], 
                      inputs: Dict[str, torch.Tensor],
                      path: str) -> None:
        """Save computed features to disk."""
        results = self.forward(feature_names, inputs)
        
        save_data = {
            "features": results,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "versions": {
                    name: self.features[name].version 
                    for name in feature_names
                },
                "lineage": {
                    name: self.lineage_graph.get(name, [])
                    for name in feature_names
                }
            }
        }
        
        torch.save(save_data, path)
    
    def load_features(self, path: str) -> Dict[str, torch.Tensor]:
        """Load features from disk."""
        data = torch.load(path)
        
        # Check version compatibility
        if self.enable_versioning:
            metadata = data.get("metadata", {})
            versions = metadata.get("versions", {})
            
            for name, version in versions.items():
                if name in self.features:
                    current_version = self.features[name].version
                    if version != current_version:
                        self.stats["version_conflicts"] += 1
                        print(f"Warning: Version mismatch for {name}: "
                              f"saved={version}, current={current_version}")
                        
        return data["features"]
    
    def get_lineage(self, feature_name: str) -> Dict[str, Any]:
        """Get complete lineage information for a feature."""
        deps = self._get_all_dependencies(feature_name)
        
        lineage = {
            "feature": feature_name,
            "direct_dependencies": self.lineage_graph.get(feature_name, []),
            "all_dependencies": deps,
            "versions": {}
        }
        
        # Add version info
        for feat in [feature_name] + deps:
            if feat in self.versions and feat in self.current_versions:
                idx = self.current_versions[feat]
                version = self.versions[feat][idx]
                lineage["versions"][feat] = {
                    "version": version.version,
                    "hash": version.compute_hash,
                    "timestamp": version.timestamp.isoformat()
                }
                
        return lineage
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get feature store statistics."""
        return {
            "stats": self.stats,
            "cache_size": len(self.cache),
            "registered_features": len(self.features),
            "total_versions": sum(len(v) for v in self.versions.values()),
            "lineage_graph_size": len(self.lineage_graph)
        }
    
    def clear_cache(self) -> None:
        """Clear the feature cache."""
        self.cache.clear()
        self.cache_timestamps.clear()