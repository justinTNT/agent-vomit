import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


class CrossModalFusion(nn.Module):
    def __init__(self,
                 modal_dims: dict = None,           # Dictionary of modality names to dimensions
                 fusion_dim: int = 512,             # Fusion dimension
                 n_heads: int = 8,                  # Number of attention heads
                 fusion_method: str = 'attention',  # Fusion method: 'attention', 'concat', 'gated'
                 dropout: float = 0.1,              # Dropout rate
                 **kwargs):                         # REQUIRED: Accept extra kwargs
        super().__init__()
        
        # Log unknown parameters
        if kwargs:
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        # Default modal dimensions if not provided
        if modal_dims is None:
            modal_dims = {'visual': 768, 'text': 512, 'audio': 256}
        
        self.modal_dims = modal_dims
        self.modality_names = list(modal_dims.keys())
        self.fusion_dim = fusion_dim
        self.fusion_method = fusion_method
        self.n_modalities = len(modal_dims)
        
        # Projection layers for each modality
        self.projections = nn.ModuleDict({
            modality: nn.Linear(dim, fusion_dim)
            for modality, dim in modal_dims.items()
        })
        
        # Layer normalization for each modality
        self.layer_norms = nn.ModuleDict({
            modality: nn.LayerNorm(fusion_dim)
            for modality in modal_dims.keys()
        })
        
        # Fusion mechanism
        if fusion_method == 'attention':
            # Cross-modal attention
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=fusion_dim,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(fusion_dim)
            
        elif fusion_method == 'gated':
            # Gated fusion mechanism
            self.gate_networks = nn.ModuleDict({
                modality: nn.Sequential(
                    nn.Linear(fusion_dim, fusion_dim),
                    nn.ReLU(),
                    nn.Linear(fusion_dim, 1),
                    nn.Sigmoid()
                ) for modality in modal_dims.keys()
            })
            
        elif fusion_method == 'concat':
            # Simple concatenation with projection
            self.concat_projection = nn.Linear(
                fusion_dim * self.n_modalities, 
                fusion_dim
            )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim)
        )
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(fusion_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, **modality_features) -> dict:
        """
        Forward pass for cross-modal fusion.
        
        Args:
            **modality_features: Dictionary of modality features
                Each value should be a tensor of shape (batch, seq_len, modal_dim)
                or (batch, modal_dim) for single vectors
        
        Returns:
            Dictionary with keys:
                - 'fused': Fused representation (batch, fusion_dim)
                - 'modality_embeddings': Dict of projected modality embeddings
                - 'attention_weights': Attention weights if using attention fusion
        """
        # Check that we have features for at least one modality
        available_modalities = [m for m in self.modality_names if m in modality_features]
        if not available_modalities:
            raise ValueError(f"No features provided. Expected at least one of: {self.modality_names}")
        
        # Project and normalize each modality
        projected_features = {}
        for modality in available_modalities:
            features = modality_features[modality]
            
            # Handle both sequence and single vector inputs
            if features.dim() == 2:
                # Single vector: (batch, dim)
                projected = self.projections[modality](features)
            else:
                # Sequence: (batch, seq_len, dim)
                projected = self.projections[modality](features)
                # Pool sequences to single vector
                projected = projected.mean(dim=1)
            
            projected = self.layer_norms[modality](projected)
            projected_features[modality] = projected
        
        # Perform fusion based on method
        attention_weights = None
        
        if self.fusion_method == 'attention':
            fused = self._attention_fusion(projected_features)
            # Note: We could return attention weights here if we modify _attention_fusion
            
        elif self.fusion_method == 'gated':
            fused = self._gated_fusion(projected_features)
            
        elif self.fusion_method == 'concat':
            fused = self._concat_fusion(projected_features, available_modalities)
        
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        # Output projection
        fused = self.output_projection(fused)
        fused = self.dropout(fused)
        fused = self.final_norm(fused)
        
        result = {
            'fused': fused,
            'modality_embeddings': projected_features
        }
        
        if attention_weights is not None:
            result['attention_weights'] = attention_weights
            
        return result
    
    def _attention_fusion(self, features: dict) -> torch.Tensor:
        """Attention-based fusion of modalities."""
        # Stack all available features
        feature_list = list(features.values())
        
        if len(feature_list) == 1:
            # Single modality, no cross-attention needed
            return feature_list[0]
        
        # Use first modality as query, others as key/value
        query = feature_list[0].unsqueeze(1)  # (batch, 1, fusion_dim)
        
        # Stack other modalities as key/value
        keys_values = torch.stack(feature_list[1:], dim=1)  # (batch, n_modalities-1, fusion_dim)
        
        # Cross-attention
        attended, weights = self.cross_attention(query, keys_values, keys_values)
        attended = attended.squeeze(1)  # Remove sequence dimension
        
        # Combine with residual
        fused = feature_list[0] + self.dropout(attended)
        fused = self.attention_norm(fused)
        
        return fused
    
    def _gated_fusion(self, features: dict) -> torch.Tensor:
        """Gated fusion of modalities."""
        # Compute gates for each modality
        gated_features = []
        gates = {}
        
        for modality, feature in features.items():
            gate = self.gate_networks[modality](feature)
            gates[modality] = gate
            gated_features.append(feature * gate)
        
        # Sum gated features
        fused = torch.stack(gated_features).sum(dim=0)
        
        # Normalize by sum of gates to maintain scale
        total_gates = sum(gates.values())
        fused = fused / (total_gates + 1e-8)
        
        return fused
    
    def _concat_fusion(self, features: dict, available_modalities: list) -> torch.Tensor:
        """Concatenation-based fusion."""
        # Create zero tensors for missing modalities
        batch_size = next(iter(features.values())).size(0)
        device = next(iter(features.values())).device
        
        all_features = []
        for modality in self.modality_names:
            if modality in features:
                all_features.append(features[modality])
            else:
                # Pad with zeros for missing modality
                zeros = torch.zeros(batch_size, self.fusion_dim, device=device)
                all_features.append(zeros)
        
        # Concatenate and project
        concatenated = torch.cat(all_features, dim=-1)
        fused = self.concat_projection(concatenated)
        
        return fused