import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
from rave_config_system import RAVEConfig




# TODO: REFACTOR TO CONFIG-FIRST INTERFACE
# New signature: def __init__(self, config: RAVEConfig, decay: float, epsilon: float, distance_metric: str, **kwargs):
# New assignments:
#         self.num_embeddings = config.quantization.codebook_size
        self.embedding_dim = config.quantization.codebook_dim
        self.commitment_cost = config.quantization.commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.distance_metric = distance_metric
class VectorQuantizer(nn.Module):
    """
    Basic vector quantizer for discrete representation learning.
    
    Args:
        num_embeddings: Number of embedding vectors (codebook size)
        embedding_dim: Dimension of each embedding vector
        commitment_cost: Weight for commitment loss
        decay: EMA decay for codebook update (0 = no EMA)
        epsilon: Small constant for numerical stability
        distance_metric: Distance metric for finding nearest code ('euclidean' or 'cosine')
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        distance_metric: str = 'euclidean'
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.distance_metric = distance_metric
        
        # Initialize codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
        # EMA parameters
        if decay > 0:
            self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
            self.register_buffer('ema_embed_avg', self.embedding.weight.data.clone())
            self.register_buffer('ema_initialized', torch.tensor(False))
    
    def forward(
        self, 
        inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize input tensor.
        
        Args:
            inputs: Input tensor [..., embedding_dim]
            
        Returns:
            quantized: Quantized tensor (same shape as input)
            indices: Codebook indices
            commitment_loss: Commitment loss for training
        """
        # Flatten input
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        if self.distance_metric == 'euclidean':
            distances = (
                flat_input.pow(2).sum(dim=1, keepdim=True)
                + self.embedding.weight.pow(2).sum(dim=1)
                - 2 * flat_input @ self.embedding.weight.t()
            )
        else:  # cosine
            flat_input_norm = F.normalize(flat_input, p=2, dim=1)
            embed_norm = F.normalize(self.embedding.weight, p=2, dim=1)
            distances = -1 * (flat_input_norm @ embed_norm.t())
        
        # Find nearest embedding
        indices_flat = torch.argmin(distances, dim=1)
        quantized = self.embedding(indices_flat).view(input_shape)
        
        # Reshape indices to match input shape (minus last dimension)
        indices = indices_flat.view(input_shape[:-1])
        
        # EMA update in training
        if self.training and self.decay > 0:
            self._ema_update(flat_input, indices_flat)
        
        # Commitment loss
        commitment_loss = self.commitment_cost * F.mse_loss(quantized.detach(), inputs)
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, indices, commitment_loss
    
    def _ema_update(self, flat_input: torch.Tensor, indices: torch.Tensor):
        """Update codebook with EMA."""
        # Initialize EMA on first call
        if not self.ema_initialized:
            self.ema_cluster_size.data.copy_(torch.ones_like(self.ema_cluster_size))
            self.ema_embed_avg.data.copy_(self.embedding.weight.data)
            self.ema_initialized.data.copy_(torch.tensor(True))
        
        # Update cluster sizes
        encodings = F.one_hot(indices, self.num_embeddings).float()
        batch_cluster_size = encodings.sum(dim=0)
        self.ema_cluster_size.data.mul_(self.decay).add_(
            batch_cluster_size, alpha=1 - self.decay
        )
        
        # Update embedding averages
        embed_sum = encodings.t() @ flat_input
        self.ema_embed_avg.data.mul_(self.decay).add_(
            embed_sum, alpha=1 - self.decay
        )
        
        # Update embeddings
        n = self.ema_cluster_size.sum()
        cluster_size = (
            (self.ema_cluster_size + self.epsilon) /
            (n + self.num_embeddings * self.epsilon) * n
        )
        embed_normalized = self.ema_embed_avg / cluster_size.unsqueeze(1)
        self.embedding.weight.data.copy_(embed_normalized)



# TODO: REFACTOR TO CONFIG-FIRST INTERFACE
# New signature: def __init__(self, config: RAVEConfig, decay: float, epsilon: float, shared_codebook: bool, quantizer_dropout: float, distance_metric: str, **kwargs):
# New assignments:
#         self.num_quantizers = config.quantization.num_quantizers
        self.num_embeddings = config.quantization.codebook_size
        self.embedding_dim = config.quantization.codebook_dim
        self.commitment_cost = config.quantization.commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.shared_codebook = shared_codebook
        self.quantizer_dropout = quantizer_dropout
        self.distance_metric = distance_metric
class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantizer for hierarchical discrete representation.
    
    Quantizes input hierarchically by encoding residuals at each level.
    Used in models like SoundStream and Encodec for high-quality audio compression.
    
    Args:
        num_quantizers: Number of residual quantizers
        num_embeddings: Codebook size for each quantizer (single value or list)
        embedding_dim: Dimension of embeddings
        commitment_cost: Weight for commitment loss
        decay: EMA decay rate
        epsilon: Small constant for stability
        shared_codebook: If True, all levels share the same codebook
        quantizer_dropout: Dropout rate for quantizers during training
        distance_metric: Distance metric for quantization
    """
    
    def __init__(
        self,
        num_quantizers: int,
        num_embeddings: Union[int, List[int]] = 1024,
        embedding_dim: int = 128,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        shared_codebook: bool = False,
        quantizer_dropout: float = 0.0,
        distance_metric: str = 'euclidean'
    ):
        super().__init__()
        
        self.num_quantizers = num_quantizers
        self.embedding_dim = embedding_dim
        self.shared_codebook = shared_codebook
        self.quantizer_dropout = quantizer_dropout
        
        # Handle num_embeddings as list or single value
        if isinstance(num_embeddings, int):
            num_embeddings = [num_embeddings] * num_quantizers
        else:
            assert len(num_embeddings) == num_quantizers
        self.num_embeddings = num_embeddings
        
        # Create quantizers
        if shared_codebook:
            # Single quantizer shared across all levels
            self.quantizers = nn.ModuleList([
                VectorQuantizer(
                    num_embeddings[0],
                    embedding_dim,
                    commitment_cost,
                    decay,
                    epsilon,
                    distance_metric
                )
                for _ in range(num_quantizers)
            ])
            # Share weights
            for i in range(1, num_quantizers):
                self.quantizers[i].embedding.weight = self.quantizers[0].embedding.weight
        else:
            # Independent quantizers
            self.quantizers = nn.ModuleList([
                VectorQuantizer(
                    n_embed,
                    embedding_dim,
                    commitment_cost,
                    decay,
                    epsilon,
                    distance_metric
                )
                for n_embed in num_embeddings
            ])
    
    def forward(
        self,
        inputs: torch.Tensor,
        num_quantizers: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Hierarchically quantize input tensor.
        
        Args:
            inputs: Input tensor [..., embedding_dim]
            num_quantizers: Number of quantizers to use (for progressive training)
            
        Returns:
            quantized: Quantized reconstruction
            indices: Codebook indices [batch, ..., num_quantizers]
            commitment_loss: Total commitment loss
        """
        if num_quantizers is None:
            num_quantizers = self.num_quantizers
        
        residual = inputs
        quantized = 0
        indices_list = []
        commitment_loss = 0
        
        # Apply quantizer dropout in training
        if self.training and self.quantizer_dropout > 0:
            # Randomly drop some quantizers
            keep_mask = torch.rand(num_quantizers) > self.quantizer_dropout
            # Always keep at least one
            if not keep_mask.any():
                keep_mask[0] = True
        else:
            keep_mask = torch.ones(num_quantizers, dtype=torch.bool)
        
        # Hierarchical quantization
        for i in range(num_quantizers):
            if not keep_mask[i]:
                # Dummy indices for dropped quantizer
                dummy_shape = list(inputs.shape[:-1])
                indices_list.append(torch.zeros(dummy_shape, dtype=torch.long, device=inputs.device))
                continue
            
            # Quantize residual
            quantized_i, indices_i, loss_i = self.quantizers[i](residual)
            
            # Update quantities
            residual = residual - quantized_i
            quantized = quantized + quantized_i
            indices_list.append(indices_i)
            commitment_loss = commitment_loss + loss_i
        
        # Stack indices
        indices = torch.stack(indices_list, dim=-1)
        
        return quantized, indices, commitment_loss
    
    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Encode inputs to discrete codes.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Codebook indices [batch, ..., num_quantizers]
        """
        _, indices, _ = self.forward(inputs)
        return indices
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode from codebook indices.
        
        Args:
            indices: Codebook indices [batch, ..., num_quantizers]
            
        Returns:
            Reconstructed tensor
        """
        quantized = 0
        
        # Sum embeddings from all levels
        for i in range(self.num_quantizers):
            if i < indices.shape[-1]:
                # Get embeddings
                quantized_i = self.quantizers[i].embedding(indices[..., i])
                quantized = quantized + quantized_i
        
        return quantized
    
    def get_codebook_usage(self) -> List[torch.Tensor]:
        """Get histogram of codebook usage if using EMA."""
        usage = []
        for quantizer in self.quantizers:
            if hasattr(quantizer, 'ema_cluster_size'):
                usage.append(quantizer.ema_cluster_size)
            else:
                usage.append(None)
        return usage
    
    def extra_repr(self) -> str:
        """String representation."""
        return (f'num_quantizers={self.num_quantizers}, '
                f'num_embeddings={self.num_embeddings}, '
                f'embedding_dim={self.embedding_dim}, '
                f'shared_codebook={self.shared_codebook}')



# TODO: REFACTOR TO CONFIG-FIRST INTERFACE
# New signature: def __init__(self, config: RAVEConfig, decay: float, epsilon: float, shared_codebook: bool, quantizer_dropout: float, distance_metric: str, **kwargs):
# New assignments:
#         self.num_quantizers = config.quantization.num_quantizers
        self.num_embeddings = config.quantization.codebook_size
        self.embedding_dim = config.quantization.codebook_dim
        self.commitment_cost = config.quantization.commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.shared_codebook = shared_codebook
        self.quantizer_dropout = quantizer_dropout
        self.distance_metric = distance_metric
class ResidualVectorQuantizerWrapper(nn.Module):
    """
    Wrapper to handle different input shapes and add projection layers.
    
    Useful when the input dimension doesn't match the quantizer dimension.
    
    Args:
        input_dim: Input dimension
        rvq_kwargs: Arguments for ResidualVectorQuantizer
    """
    
    def __init__(self, input_dim: int, **rvq_kwargs):
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = rvq_kwargs.get('embedding_dim', 128)
        
        # Projection layers if needed
        if input_dim != self.embedding_dim:
            self.project_in = nn.Linear(input_dim, self.embedding_dim)
            self.project_out = nn.Linear(self.embedding_dim, input_dim)
        else:
            self.project_in = nn.Identity()
            self.project_out = nn.Identity()
        
        # Main quantizer
        self.rvq = ResidualVectorQuantizer(**rvq_kwargs)
    
    def forward(self, inputs: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward with projection."""
        x = self.project_in(inputs)
        quantized, indices, loss = self.rvq(x, **kwargs)
        quantized = self.project_out(quantized)
        return quantized, indices, loss