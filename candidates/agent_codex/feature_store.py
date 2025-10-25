from __future__ import annotations

import time
from typing import Dict, Mapping, Optional

import torch
from torch import nn

from .utils import warn_on_unused_kwargs


class FeatureStore(nn.Module):
    """In-memory feature store with versioning metadata."""

    def __init__(
        self,
        feature_dim: int = 128,
        allow_overwrite: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        warn_on_unused_kwargs(kwargs)

        self.feature_dim = feature_dim
        self.allow_overwrite = allow_overwrite
        self._store: Dict[str, torch.Tensor] = {}
        self._metadata: Dict[str, Dict[str, object]] = {}

    def write(
        self,
        entity_id: str,
        features: torch.Tensor,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        if features.dim() != 1 or features.size(0) != self.feature_dim:
            raise ValueError(
                f"features must have shape ({self.feature_dim},)"
            )
        if entity_id in self._store and not self.allow_overwrite:
            raise KeyError(f"Entity '{entity_id}' already exists. Enable allow_overwrite to replace.")

        self._store[entity_id] = features.detach().clone()
        meta = dict(metadata) if metadata is not None else {}
        meta.setdefault("updated_at", time.time())
        meta.setdefault("numel", features.numel())
        self._metadata[entity_id] = meta

    def read(self, entity_id: str) -> torch.Tensor:
        if entity_id not in self._store:
            raise KeyError(f"Unknown entity_id '{entity_id}'")
        return self._store[entity_id].clone()

    def delete(self, entity_id: str) -> None:
        if entity_id in self._store:
            del self._store[entity_id]
            self._metadata.pop(entity_id, None)

    def describe(self, entity_id: str) -> Dict[str, object]:
        if entity_id not in self._metadata:
            raise KeyError(f"Unknown entity_id '{entity_id}'")
        return dict(self._metadata[entity_id])

    def list_entities(self) -> Dict[str, Dict[str, object]]:
        return {entity: dict(meta) for entity, meta in self._metadata.items()}

    def clear(self) -> None:
        self._store.clear()
        self._metadata.clear()
