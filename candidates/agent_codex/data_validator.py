from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional

import torch
from torch import nn

from .utils import warn_on_unused_kwargs

SchemaType = Mapping[str, Callable[[torch.Tensor], bool]]


class DataValidator(nn.Module):
    """Validates batch dictionaries against a schema and statistics."""

    def __init__(
        self,
        schema: Optional[SchemaType] = None,
        compute_stats: bool = True,
        strict: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        warn_on_unused_kwargs(kwargs)

        self.schema = dict(schema) if schema is not None else {}
        self.compute_stats = compute_stats
        self.strict = strict

    def forward(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, object]:
        if not isinstance(batch, Mapping):
            raise TypeError("batch must be a mapping of tensors")

        results: Dict[str, object] = {"valid": True, "errors": []}
        for key, validator in self.schema.items():
            if key not in batch:
                message = f"Missing required field '{key}'"
                results["errors"].append(message)
                if self.strict:
                    results["valid"] = False
                continue
            tensor = batch[key]
            try:
                valid = bool(validator(tensor))
            except Exception as exc:  # noqa: BLE001
                valid = False
                message = f"Validator for '{key}' raised {exc}"
                results["errors"].append(message)
            if not valid:
                results["errors"].append(f"Validator failed for '{key}'")
                results["valid"] = False

        if self.compute_stats:
            stats: Dict[str, Dict[str, float]] = {}
            for key, tensor in batch.items():
                if not isinstance(tensor, torch.Tensor):
                    continue
                stats[key] = self._statistics(tensor)
            results["stats"] = stats

        if self.strict and results["errors"]:
            results["valid"] = False
        return results

    def _statistics(self, tensor: torch.Tensor) -> Dict[str, float]:
        if tensor.dtype not in (torch.float16, torch.float32, torch.float64):
            return {
                "numel": float(tensor.numel()),
            }
        flat = tensor.float().view(-1)
        return {
            "mean": float(flat.mean().item()) if flat.numel() else float("nan"),
            "std": float(flat.std(unbiased=False).item()) if flat.numel() else float("nan"),
            "min": float(flat.min().item()) if flat.numel() else float("nan"),
            "max": float(flat.max().item()) if flat.numel() else float("nan"),
            "numel": float(flat.numel()),
        }
