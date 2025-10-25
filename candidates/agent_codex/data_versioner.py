from __future__ import annotations

import json
import time
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .utils import warn_on_unused_kwargs


@dataclass
class DataVersion:
    version_id: str
    parent_id: Optional[str]
    created_at: float
    message: str
    tensor_shape: Tuple[int, ...]
    dtype: str
    tensor_hash: str
    storage_path: str
    stats: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class DataDiff:
    version_id1: str
    version_id2: str
    l2_distance: float
    max_abs_diff: float
    mean_abs_diff: float
    shape_equal: bool
    dtype_equal: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class DataVersioner(nn.Module):
    """Version control for tensors with metadata tracking."""

    METADATA_FILE = "metadata.json"

    def __init__(self, storage_path: str = "./versions", **kwargs) -> None:
        super().__init__()
        warn_on_unused_kwargs(kwargs)

        self.storage_root = Path(storage_path).expanduser().resolve()
        self.storage_root.mkdir(parents=True, exist_ok=True)

        self.metadata_path = self.storage_root / self.METADATA_FILE
        self.versions: Dict[str, DataVersion] = {}
        self._hash_index: Dict[str, str] = {}
        self._latest_version_id: Optional[str] = None
        self._current_parent: Optional[str] = None

        self._load_metadata()

    def __call__(self, data: torch.Tensor, message: str = "") -> str:
        return self.version(data, message)

    def version(self, data: torch.Tensor, message: str = "") -> str:
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(data)!r}")

        parent_id = self._current_parent if self._current_parent else self._latest_version_id

        data_cpu = data.detach().cpu()
        tensor_bytes = data_cpu.contiguous().numpy().tobytes()

        hasher = hashlib.sha256()
        hasher.update(str(data_cpu.dtype).encode("utf-8"))
        hasher.update(str(tuple(data_cpu.shape)).encode("utf-8"))
        hasher.update(tensor_bytes)
        tensor_hash = hasher.hexdigest()

        created_at = time.time()
        version_id = self._build_version_id(created_at, tensor_hash)
        while version_id in self.versions:
            created_at = time.time()
            version_id = self._build_version_id(created_at, tensor_hash)

        stats = self._compute_stats(data_cpu)

        data_path = self.storage_root / f"{version_id}.pt"
        if tensor_hash in self._hash_index:
            # Reuse existing storage to deduplicate identical tensors
            existing_path = Path(self.versions[self._hash_index[tensor_hash]].storage_path)
            if existing_path.exists():
                data_path = existing_path
            else:
                torch.save({"data": data_cpu}, data_path)
        else:
            torch.save({"data": data_cpu}, data_path)

        metadata = DataVersion(
            version_id=version_id,
            parent_id=parent_id,
            created_at=created_at,
            message=message,
            tensor_shape=tuple(data_cpu.shape),
            dtype=str(data_cpu.dtype),
            tensor_hash=tensor_hash,
            storage_path=str(data_path),
            stats=stats,
        )

        self.versions[version_id] = metadata
        self._hash_index[tensor_hash] = version_id
        self._latest_version_id = version_id
        self._current_parent = None
        self._persist_metadata()

        return version_id

    def load(self, version_id: str) -> torch.Tensor:
        metadata = self._get_version(version_id)
        path = Path(metadata.storage_path)
        if not path.exists():
            raise FileNotFoundError(f"Stored tensor not found at {path}")
        payload = torch.load(path, map_location="cpu")
        tensor = payload.get("data")
        if tensor is None:
            raise KeyError(f"Tensor payload missing 'data' key in {path}")
        return tensor

    def diff(self, version_id1: str, version_id2: str) -> DataDiff:
        meta1 = self._get_version(version_id1)
        meta2 = self._get_version(version_id2)

        tensor1 = self.load(version_id1)
        tensor2 = self.load(version_id2)

        same_shape = tensor1.shape == tensor2.shape
        same_dtype = tensor1.dtype == tensor2.dtype

        if same_shape:
            diff_tensor = tensor1.to(torch.float32) - tensor2.to(torch.float32)
            l2_distance = float(diff_tensor.norm().item())
            max_abs_diff = float(diff_tensor.abs().max().item()) if diff_tensor.numel() else 0.0
            mean_abs_diff = float(diff_tensor.abs().mean().item()) if diff_tensor.numel() else 0.0
        else:
            l2_distance = float("inf")
            max_abs_diff = float("inf")
            mean_abs_diff = float("inf")

        return DataDiff(
            version_id1=version_id1,
            version_id2=version_id2,
            l2_distance=l2_distance,
            max_abs_diff=max_abs_diff,
            mean_abs_diff=mean_abs_diff,
            shape_equal=same_shape,
            dtype_equal=same_dtype,
        )

    def set_parent(self, version_id: Optional[str]) -> None:
        if version_id is not None and version_id not in self.versions:
            raise KeyError(f"Unknown version_id '{version_id}'")
        self._current_parent = version_id

    def list_versions(self) -> Dict[str, Dict[str, object]]:
        return {vid: meta.to_dict() for vid, meta in self.versions.items()}

    # Internal helpers -------------------------------------------------

    def _get_version(self, version_id: str) -> DataVersion:
        if version_id not in self.versions:
            raise KeyError(f"Unknown version_id '{version_id}'")
        return self.versions[version_id]

    def _build_version_id(self, created_at: float, tensor_hash: str) -> str:
        timestamp_hex = format(int(created_at * 1e6), "x")
        return f"v{timestamp_hex}-{tensor_hash[:8]}"

    def _compute_stats(self, tensor: torch.Tensor) -> Dict[str, float]:
        if tensor.numel() == 0:
            return {
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "numel": 0.0,
            }
        tensor32 = tensor.to(torch.float32)
        return {
            "mean": float(tensor32.mean().item()),
            "std": float(tensor32.std(unbiased=False).item()),
            "min": float(tensor32.min().item()),
            "max": float(tensor32.max().item()),
            "numel": float(tensor.numel()),
        }

    def _persist_metadata(self) -> None:
        payload = {
            "versions": [meta.to_dict() for meta in self.versions.values()],
            "hash_index": self._hash_index,
            "latest": self._latest_version_id,
        }
        with self.metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _load_metadata(self) -> None:
        if not self.metadata_path.exists():
            return
        with self.metadata_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        versions = payload.get("versions", [])
        self.versions = {}
        for meta in versions:
            data_version = DataVersion(**meta)
            self.versions[data_version.version_id] = data_version
        self._hash_index = payload.get("hash_index", {})
        self._latest_version_id = payload.get("latest")
        if self._latest_version_id not in self.versions:
            self._latest_version_id = None
