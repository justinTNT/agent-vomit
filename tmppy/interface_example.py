"""Example of comprehensive interface specification for CrossModalFusion."""

from typing import Dict, Optional, Literal, Tuple
from typing_extensions import Annotated
from pydantic import BaseModel, Field, validator
import torch
import torch.nn as nn


# Configuration Model
class CrossModalFusionConfig(BaseModel):
    """Configuration for CrossModalFusion module."""
    
    modality_dims: Dict[str, int] = Field(
        ...,
        description="Dimensions for each modality, e.g. {'visual': 768, 'text': 512}",
        example={'visual': 768, 'text': 512}
    )
    d_hidden: int = Field(
        1024,
        description="Hidden dimension after fusion"
    )
    fusion_type: Literal['concat', 'sum', 'attention'] = Field(
        'concat',
        description="How to combine modalities"
    )
    dropout: float = Field(0.1, ge=0, le=1)
    
    @validator('modality_dims')
    def validate_modalities(cls, v):
        if len(v) < 2:
            raise ValueError("Need at least 2 modalities")
        return v


# Type Aliases for clarity
ModalityTensor = Annotated[torch.Tensor, "Shape[batch, seq_len, dim]"]
ModalityInputs = Dict[str, ModalityTensor]


class CrossModalFusion(nn.Module):
    """Fuses multiple modalities into a unified representation.
    
    This module takes inputs from different modalities (e.g., visual, text)
    and combines them into a single representation using the specified
    fusion strategy.
    
    Args:
        modality_dims: Dictionary mapping modality names to their dimensions
            Example: {'visual': 768, 'text': 512}
        d_hidden: Output hidden dimension (default: 1024)
        fusion_type: How to combine modalities: 'concat', 'sum', 'attention'
        dropout: Dropout rate (default: 0.1)
        **kwargs: Additional arguments (required for compatibility)
    
    Input Shape:
        Dict[str, Tensor] where each tensor has shape [batch, seq_len, dim]
        The dim must match the corresponding modality_dims value
    
    Output Shape:
        Tensor of shape [batch, seq_len, d_hidden]
    
    Examples:
        >>> # Using config
        >>> config = CrossModalFusionConfig(
        ...     modality_dims={'visual': 768, 'text': 512},
        ...     d_hidden=1024
        ... )
        >>> fusion = CrossModalFusion(**config.dict())
        
        >>> # Direct instantiation
        >>> fusion = CrossModalFusion(
        ...     modality_dims={'visual': 768, 'text': 512},
        ...     d_hidden=1024,
        ...     fusion_type='attention'
        ... )
        
        >>> # Forward pass
        >>> inputs = {
        ...     'visual': torch.randn(2, 10, 768),
        ...     'text': torch.randn(2, 15, 512)
        ... }
        >>> output = fusion(inputs)  # Shape: [2, 15, 1024]
    
    Note:
        When fusion_type='concat', output seq_len matches the longest input
        When fusion_type='attention', cross-attention is used between modalities
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        d_hidden: int = 1024,
        fusion_type: Literal['concat', 'sum', 'attention'] = 'concat',
        dropout: float = 0.1,
        **kwargs  # Required for all modules
    ):
        super().__init__()
        
        # Validate config if using Pydantic
        if not isinstance(modality_dims, dict):
            config = CrossModalFusionConfig(
                modality_dims=modality_dims,
                d_hidden=d_hidden,
                fusion_type=fusion_type,
                dropout=dropout
            )
            modality_dims = config.modality_dims
            d_hidden = config.d_hidden
            fusion_type = config.fusion_type
            dropout = config.dropout
        
        self.modality_dims = modality_dims
        self.d_hidden = d_hidden
        self.fusion_type = fusion_type
        
        # Implementation details...
        
    def forward(
        self, 
        inputs: Dict[str, torch.Tensor],
        mask: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Forward pass combining multiple modalities.
        
        Args:
            inputs: Dictionary of modality_name -> tensor[batch, seq_len, dim]
            mask: Optional dictionary of modality_name -> mask[batch, seq_len]
        
        Returns:
            Fused representation tensor[batch, seq_len, d_hidden]
        
        Raises:
            ValueError: If modality names don't match initialization
            RuntimeError: If tensor dimensions don't match modality_dims
        """
        # Validate inputs
        for name, tensor in inputs.items():
            if name not in self.modality_dims:
                raise ValueError(f"Unknown modality: {name}")
            if tensor.size(-1) != self.modality_dims[name]:
                raise RuntimeError(
                    f"Expected dim={self.modality_dims[name]} for {name}, "
                    f"got {tensor.size(-1)}"
                )
        
        # Implementation...
        return torch.randn(2, 15, self.d_hidden)  # Placeholder


# Interface decorator for machine parsing (optional)
from dataclasses import dataclass

@dataclass
class InterfaceSpec:
    """Machine-readable interface specification."""
    module_name: str = "CrossModalFusion"
    filename: str = "cross_modal_fusion.py"
    
    # Required parameters (no defaults)
    required_params: Dict[str, type] = None
    
    # Optional parameters (with defaults)  
    optional_params: Dict[str, Tuple[type, any]] = None
    
    # Forward method specification
    forward_inputs: Dict[str, str] = None
    forward_output: str = "Tensor[batch, seq_len, d_hidden]"
    
    # Constraints
    constraints: Dict[str, str] = None
    
    def __post_init__(self):
        self.required_params = {
            "modality_dims": "Dict[str, int]"
        }
        self.optional_params = {
            "d_hidden": (int, 1024),
            "fusion_type": (str, "concat"),
            "dropout": (float, 0.1)
        }
        self.forward_inputs = {
            "inputs": "Dict[str, Tensor[batch, seq_len, dim]]",
            "mask": "Optional[Dict[str, Tensor[batch, seq_len]]]"
        }
        self.constraints = {
            "modality_count": "len(modality_dims) >= 2",
            "dim_match": "inputs[name].size(-1) == modality_dims[name]"
        }


# Export the specification
INTERFACE = InterfaceSpec()