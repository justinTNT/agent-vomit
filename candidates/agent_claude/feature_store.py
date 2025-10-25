"""
FeatureStore - Centralized feature computation and storage.

This module stores computed features with metadata, supports feature versioning,
and enables feature retrieval by key.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Callable
from collections import OrderedDict
import warnings
from datetime import datetime


class FeatureStore(nn.Module):
    """
    Centralized feature computation and storage with versioning support.
    
    Args:
        max_features: Maximum number of features to store
        max_versions: Maximum versions per feature
        compute_on_store: Whether to compute features on store
        enable_metadata: Whether to track metadata
        device: Device for stored features
        **kwargs: Additional arguments
    """
    
    def __init__(
        self,
        max_features: int = 1000,
        max_versions: int = 10,
        compute_on_store: bool = True,
        enable_metadata: bool = True,
        device: Optional[torch.device] = None,
        **kwargs
    ):
        super().__init__()
        
        if max_features <= 0:
            raise ValueError(f"max_features must be positive, got {max_features}")
        if max_versions <= 0:
            raise ValueError(f"max_versions must be positive, got {max_versions}")
        
        self.max_features = max_features
        self.max_versions = max_versions
        self.compute_on_store = compute_on_store
        self.enable_metadata = enable_metadata
        self.device = device
        
        # Feature storage
        self._features = OrderedDict()
        self._metadata = OrderedDict() if enable_metadata else None
        self._compute_fns = OrderedDict()
        
        # Statistics
        self.register_buffer('total_stores', torch.tensor(0, dtype=torch.long))
        self.register_buffer('total_retrievals', torch.tensor(0, dtype=torch.long))
        self.register_buffer('cache_hits', torch.tensor(0, dtype=torch.long))
        self.register_buffer('cache_misses', torch.tensor(0, dtype=torch.long))
        
        # Handle unused kwargs
        if kwargs:
            warnings.warn(f"Unused arguments: {list(kwargs.keys())}")
    
    def register_compute_fn(self, key: str, compute_fn: Callable[[torch.Tensor], torch.Tensor]):
        """Register a computation function for a feature key."""
        if not callable(compute_fn):
            raise ValueError(f"compute_fn must be callable, got {type(compute_fn)}")
        self._compute_fns[key] = compute_fn
    
    def store(
        self,
        key: str,
        feature: Optional[torch.Tensor] = None,
        input_data: Optional[torch.Tensor] = None,
        version: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Store a feature with optional computation.
        
        Args:
            key: Feature identifier
            feature: Pre-computed feature tensor
            input_data: Input data for computing feature
            version: Specific version number (auto-increment if None)
            metadata: Additional metadata to store
            
        Returns:
            Dictionary containing:
                - stored: Whether storage was successful
                - version: Version number stored
                - computed: Whether feature was computed
        """
        if feature is None and input_data is None:
            raise ValueError("Either feature or input_data must be provided")
        
        # Compute feature if needed
        computed = False
        if feature is None and self.compute_on_store:
            if key not in self._compute_fns:
                raise ValueError(f"No compute function registered for key: {key}")
            feature = self._compute_fns[key](input_data)
            computed = True
        elif feature is None:
            raise ValueError("Feature is None and compute_on_store is False")
        
        if not isinstance(feature, torch.Tensor):
            raise TypeError(f"Feature must be torch.Tensor, got {type(feature)}")
        
        # Move to device if specified
        if self.device is not None:
            feature = feature.to(self.device)
        
        # Initialize key storage if needed
        if key not in self._features:
            self._features[key] = OrderedDict()
            if self.enable_metadata:
                self._metadata[key] = OrderedDict()
        
        # Determine version
        if version is None:
            existing_versions = list(self._features[key].keys())
            version = max(existing_versions, default=-1) + 1
        
        # Store feature
        self._features[key][version] = feature.detach()
        
        # Store metadata
        if self.enable_metadata:
            meta = {
                'timestamp': datetime.now().isoformat(),
                'shape': list(feature.shape),
                'dtype': str(feature.dtype),
                'device': str(feature.device),
                'computed': computed
            }
            if metadata:
                meta.update(metadata)
            self._metadata[key][version] = meta
        
        # Manage versions
        if len(self._features[key]) > self.max_versions:
            oldest_version = min(self._features[key].keys())
            del self._features[key][oldest_version]
            if self.enable_metadata:
                del self._metadata[key][oldest_version]
        
        # Manage total features
        if len(self._features) > self.max_features:
            oldest_key = next(iter(self._features))
            del self._features[oldest_key]
            if self.enable_metadata:
                del self._metadata[oldest_key]
            if oldest_key in self._compute_fns:
                del self._compute_fns[oldest_key]
        
        self.total_stores += 1
        
        return {
            'stored': True,
            'version': version,
            'computed': computed
        }
    
    def retrieve(
        self,
        key: str,
        version: Optional[int] = None,
        input_data: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Retrieve a stored feature or compute it.
        
        Args:
            key: Feature identifier
            version: Specific version (latest if None)
            input_data: Input data for computing if not cached
            
        Returns:
            Dictionary containing:
                - feature: Retrieved or computed feature
                - version: Version number
                - metadata: Associated metadata (if enabled)
                - cache_hit: Whether feature was in cache
        """
        self.total_retrievals += 1
        cache_hit = False
        
        # Try to retrieve from cache
        if key in self._features:
            if version is None:
                # Get latest version
                version = max(self._features[key].keys())
            
            if version in self._features[key]:
                feature = self._features[key][version]
                metadata = self._metadata[key][version] if self.enable_metadata else None
                cache_hit = True
                self.cache_hits += 1
            else:
                raise ValueError(f"Version {version} not found for key {key}")
        else:
            # Compute if possible
            if input_data is not None and key in self._compute_fns:
                feature = self._compute_fns[key](input_data)
                version = 0
                metadata = None
                self.cache_misses += 1
                
                # Optionally store computed feature
                if self.compute_on_store:
                    self.store(key, feature, version=version)
            else:
                raise KeyError(f"Feature key '{key}' not found and cannot compute")
        
        return {
            'feature': feature,
            'version': version,
            'metadata': metadata,
            'cache_hit': cache_hit
        }
    
    def forward(self, x: torch.Tensor, keys: List[str]) -> Dict[str, torch.Tensor]:
        """
        Retrieve or compute multiple features.
        
        Args:
            x: Input tensor for computing features
            keys: List of feature keys to retrieve
            
        Returns:
            Dictionary mapping keys to features
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        
        results = {}
        for key in keys:
            try:
                result = self.retrieve(key, input_data=x)
                results[key] = result['feature']
            except (KeyError, ValueError) as e:
                warnings.warn(f"Could not retrieve feature '{key}': {e}")
                # Compute if possible
                if key in self._compute_fns:
                    results[key] = self._compute_fns[key](x)
                else:
                    results[key] = torch.empty(0, device=x.device)
        
        return results
    
    def list_features(self) -> Dict[str, List[int]]:
        """List all stored features and their versions."""
        return {key: list(versions.keys()) for key, versions in self._features.items()}
    
    def clear(self, key: Optional[str] = None):
        """Clear stored features."""
        if key is None:
            self._features.clear()
            if self.enable_metadata:
                self._metadata.clear()
        else:
            if key in self._features:
                del self._features[key]
            if self.enable_metadata and key in self._metadata:
                del self._metadata[key]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            'total_stores': self.total_stores.item(),
            'total_retrievals': self.total_retrievals.item(),
            'cache_hits': self.cache_hits.item(),
            'cache_misses': self.cache_misses.item(),
            'hit_rate': (self.cache_hits.float() / (self.total_retrievals + 1e-8)).item(),
            'num_features': len(self._features),
            'num_compute_fns': len(self._compute_fns)
        }