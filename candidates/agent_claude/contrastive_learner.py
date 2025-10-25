"""ContrastiveLearner - Contrastive learning framework (SimCLR/MoCo style)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import copy


class ContrastiveLearner(nn.Module):
    """Contrastive learning framework supporting both SimCLR and MoCo modes.
    
    This module implements contrastive learning methods for self-supervised representation
    learning. It supports both SimCLR (simple framework) and MoCo (momentum contrast) modes.
    
    Args:
        encoder: Base encoder network
        projection_dim: Dimension of projection head output
        temperature: Temperature parameter for contrastive loss (default: 0.07)
        mode: Either 'simclr' or 'moco' (default: 'simclr')
        momentum: Momentum coefficient for MoCo (default: 0.999)
        queue_size: Queue size for MoCo (default: 65536)
        **kwargs: Additional keyword arguments
    
    Returns:
        Dict containing:
            - loss: Contrastive loss
            - embeddings: Encoder embeddings before projection
            - projections: Projected embeddings
            - logits: Similarity logits
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int,
        temperature: float = 0.07,
        mode: str = 'simclr',
        momentum: float = 0.999,
        queue_size: int = 65536,
        **kwargs
    ):
        super().__init__()
        
        # Validate inputs
        if mode not in ['simclr', 'moco']:
            raise ValueError(f"mode must be 'simclr' or 'moco', got {mode}")
        
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        
        if mode == 'moco' and not (0 <= momentum <= 1):
            raise ValueError(f"momentum must be between 0 and 1, got {momentum}")
        
        self.encoder = encoder
        self.temperature = temperature
        self.mode = mode
        self.momentum = momentum
        self.queue_size = queue_size
        
        # Get encoder output dimension
        # This assumes encoder has an attribute specifying output dimension
        # or we can infer it by running a dummy input
        encoder_dim = self._get_encoder_dim(encoder)
        
        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, projection_dim)
        )
        
        # Mode-specific initialization
        if mode == 'moco':
            # Momentum encoder
            self.momentum_encoder = copy.deepcopy(encoder)
            self.momentum_projection_head = copy.deepcopy(self.projection_head)
            
            # Disable gradients for momentum encoder
            for param in self.momentum_encoder.parameters():
                param.requires_grad = False
            for param in self.momentum_projection_head.parameters():
                param.requires_grad = False
            
            # Queue for negative samples
            self.register_buffer('queue', torch.randn(projection_dim, queue_size))
            self.queue = F.normalize(self.queue, dim=0)
            self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_encoder_dim(self, encoder: nn.Module) -> int:
        """Get output dimension of encoder."""
        # Try common attribute names
        for attr in ['output_dim', 'out_features', 'hidden_dim']:
            if hasattr(encoder, attr):
                return getattr(encoder, attr)
        
        # If no attribute found, try to infer from last linear layer
        for module in reversed(list(encoder.modules())):
            if isinstance(module, nn.Linear):
                return module.out_features
        
        # Default fallback
        raise ValueError("Could not determine encoder output dimension")
    
    def _initialize_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.projection_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    @torch.no_grad()
    def _momentum_update(self):
        """Update momentum encoder using exponential moving average."""
        if self.mode != 'moco':
            return
        
        # Update momentum encoder
        for param, momentum_param in zip(
            self.encoder.parameters(),
            self.momentum_encoder.parameters()
        ):
            momentum_param.data = momentum_param.data * self.momentum + param.data * (1. - self.momentum)
        
        # Update momentum projection head
        for param, momentum_param in zip(
            self.projection_head.parameters(),
            self.momentum_projection_head.parameters()
        ):
            momentum_param.data = momentum_param.data * self.momentum + param.data * (1. - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """Update queue with new keys for MoCo."""
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        
        # Replace oldest entries in queue
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            # Wrap around
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T
        
        # Update pointer
        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the contrastive learner.
        
        Args:
            x1: First view/augmentation of input data
            x2: Second view/augmentation of input data
        
        Returns:
            Dictionary containing loss, embeddings, projections, and logits
        """
        batch_size = x1.shape[0]
        device = x1.device
        
        # Get embeddings from encoder
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        
        # Project embeddings
        p1 = self.projection_head(z1)
        p2 = self.projection_head(z2)
        
        # Normalize projections
        p1 = F.normalize(p1, dim=1)
        p2 = F.normalize(p2, dim=1)
        
        if self.mode == 'simclr':
            # SimCLR: all pairs within batch
            # Concatenate projections
            projections = torch.cat([p1, p2], dim=0)  # [2*batch_size, projection_dim]
            
            # Compute similarity matrix
            sim_matrix = torch.matmul(projections, projections.T) / self.temperature  # [2*batch_size, 2*batch_size]
            
            # Create labels for positive pairs
            labels = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)], dim=0)
            labels = labels.to(device)
            
            # Mask out diagonal (self-similarity)
            mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
            sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))
            
            # Compute loss
            loss = F.cross_entropy(sim_matrix, labels)
            
            logits = sim_matrix
        
        else:  # moco
            # Query: p1, Key: p2 (from momentum encoder)
            with torch.no_grad():
                # Update momentum encoder
                self._momentum_update()
                
                # Get momentum embeddings and projections
                z2_momentum = self.momentum_encoder(x2)
                p2_momentum = self.momentum_projection_head(z2_momentum)
                p2_momentum = F.normalize(p2_momentum, dim=1)
            
            # Positive logits: [batch_size, 1]
            pos_logits = torch.einsum('nc,nc->n', [p1, p2_momentum]).unsqueeze(-1) / self.temperature
            
            # Negative logits: [batch_size, queue_size]
            neg_logits = torch.einsum('nc,ck->nk', [p1, self.queue.clone().detach()]) / self.temperature
            
            # Concatenate positive and negative logits
            logits = torch.cat([pos_logits, neg_logits], dim=1)  # [batch_size, 1 + queue_size]
            
            # Labels: positive is at position 0
            labels = torch.zeros(batch_size, dtype=torch.long, device=device)
            
            # Compute loss
            loss = F.cross_entropy(logits, labels)
            
            # Update queue
            self._dequeue_and_enqueue(p2_momentum)
        
        # Prepare output dictionary
        output = {
            'loss': loss,
            'embeddings': torch.cat([z1, z2], dim=0) if self.mode == 'simclr' else z1,
            'projections': torch.cat([p1, p2], dim=0) if self.mode == 'simclr' else p1,
            'logits': logits
        }
        
        return output