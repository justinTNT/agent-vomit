"""
Utility functions for module common operations.
"""
import warnings
from typing import Dict, Any


def warn_on_unused_kwargs(kwargs: Dict[str, Any]) -> None:
    """Emit a warning when unexpected keyword arguments are provided.
    
    Args:
        kwargs: Dictionary of unused keyword arguments
    """
    if kwargs:
        warnings.warn(
            f"Ignoring unknown parameters: {sorted(kwargs.keys())}",
            RuntimeWarning,
            stacklevel=2,
        )