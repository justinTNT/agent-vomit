import warnings
from typing import Dict, Iterable


def warn_on_unused_kwargs(kwargs: Dict[str, object]) -> None:
    """Emit a warning when unexpected keyword arguments are provided."""
    if kwargs:
        warnings.warn(
            f"Ignoring unknown parameters: {sorted(kwargs.keys())}",
            RuntimeWarning,
            stacklevel=2,
        )


def ensure_same_device(tensors: Iterable):
    """Ensure all tensors are on the same device as the first tensor."""
    tensors = list(tensors)
    if not tensors:
        return
    reference = tensors[0].device
    for tensor in tensors[1:]:
        if tensor.device != reference:
            raise ValueError(
                f"Expected all tensors on device {reference}, got {tensor.device}"
            )
