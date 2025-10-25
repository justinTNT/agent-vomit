"""SetEncoder - Permutation-invariant encoding of sets."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class SetEncoder(nn.Module):
    """Permutation-invariant encoding of sets.
    
    This module implements a permutation-invariant set encoder that can handle
    variable-sized sets. It uses attention-based pooling or DeepSets architecture
    to create both a global set representation and individual element encodings.
    
    Args:
        input_dim: Dimension of input elements
        hidden_dim: Dimension of hidden layers
        output_dim: Dimension of output representation
        num_layers: Number of layers in the element encoder
        pooling_method: Either 'attention' or 'deepsets' (default: 'attention')
        dropout: Dropout probability (default: 0.1)
        **kwargs: Additional keyword arguments
    
    Returns:
        Dict containing:
            - set_representation: Global set representation [batch_size, output_dim]
            - element_encodings: Individual element encodings [batch_size, set_size, hidden_dim]
            - attention_weights: Attention weights (if using attention pooling) [batch_size, set_size]
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        pooling_method: str = 'attention',
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        # Validate inputs
        if pooling_method not in ['attention', 'deepsets']:
            raise ValueError(f"pooling_method must be 'attention' or 'deepsets', got {pooling_method}")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pooling_method = pooling_method
        self.dropout = dropout
        
        # Element encoder (phi network in DeepSets)
        layers = []
        prev_dim = input_dim
        for i in range(num_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.element_encoder = nn.Sequential(*layers)
        
        # Pooling mechanism
        if pooling_method == 'attention':
            # Attention-based pooling
            self.attention_query = nn.Linear(hidden_dim, hidden_dim)
            self.attention_key = nn.Linear(hidden_dim, hidden_dim)
            self.attention_scale = hidden_dim ** 0.5
        
        # Set encoder (rho network in DeepSets)
        self.set_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the set encoder.
        
        Args:
            x: Input tensor of shape [batch_size, set_size, input_dim]
            mask: Optional boolean mask of shape [batch_size, set_size] where True indicates valid elements
        
        Returns:
            Dictionary containing set representation, element encodings, and attention weights (if applicable)
        """
        # Validate inputs
        if x.dim() != 3:
            raise ValueError(f"Expected input tensor to have 3 dimensions, got {x.dim()}")
        
        batch_size, set_size, _ = x.shape
        device = x.device
        
        # Create mask if not provided
        if mask is None:
            mask = torch.ones(batch_size, set_size, dtype=torch.bool, device=device)
        
        # Encode individual elements
        element_encodings = self.element_encoder(x)  # [batch_size, set_size, hidden_dim]
        
        # Apply mask to encodings
        element_encodings = element_encodings * mask.unsqueeze(-1).float()
        
        # Pool elements based on method
        if self.pooling_method == 'attention':
            # Compute attention scores
            queries = self.attention_query(element_encodings)
            keys = self.attention_key(element_encodings)
            
            # Global query (learnable or mean of queries)
            global_query = queries.mean(dim=1, keepdim=True)  # [batch_size, 1, hidden_dim]
            
            # Compute attention scores
            scores = torch.bmm(global_query, keys.transpose(1, 2)) / self.attention_scale  # [batch_size, 1, set_size]
            scores = scores.squeeze(1)  # [batch_size, set_size]
            
            # Mask out invalid positions
            scores = scores.masked_fill(~mask, float('-inf'))
            
            # Compute attention weights
            attention_weights = F.softmax(scores, dim=-1)  # [batch_size, set_size]
            
            # Handle case where all elements are masked
            attention_weights = torch.where(
                mask.any(dim=1, keepdim=True),
                attention_weights,
                torch.zeros_like(attention_weights)
            )
            
            # Weighted sum pooling
            pooled = torch.bmm(attention_weights.unsqueeze(1), element_encodings).squeeze(1)  # [batch_size, hidden_dim]
        
        else:  # deepsets
            # Mean pooling with proper handling of mask
            mask_sum = mask.sum(dim=1, keepdim=True).float()  # [batch_size, 1]
            mask_sum = torch.clamp(mask_sum, min=1.0)  # Avoid division by zero
            
            pooled = element_encodings.sum(dim=1) / mask_sum  # [batch_size, hidden_dim]
            attention_weights = None
        
        # Encode the pooled representation
        set_representation = self.set_encoder(pooled)  # [batch_size, output_dim]
        
        # Prepare output dictionary
        output = {
            'set_representation': set_representation,
            'element_encodings': element_encodings
        }
        
        if attention_weights is not None:
            output['attention_weights'] = attention_weights
        
        return output