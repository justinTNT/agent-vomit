
import torch
import torch.nn as nn
import torch.nn.functional as F
from rave_config_system import RAVEConfig

class ConfigFirstVectorQuantizer(nn.Module):
    """Vector Quantizer with config-first interface"""
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__()
        
        self.num_embeddings = config.quantization.codebook_size
        self.embedding_dim = config.quantization.codebook_dim
        self.commitment_cost = config.quantization.commitment_cost
        self.epsilon = config.quantization.epsilon
        
        # Initialize codebook
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
    
    def forward(self, inputs: torch.Tensor) -> dict:
        """Vector quantization forward pass"""
        
        # Flatten input for quantization
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # Get indices of closest embedding vectors
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        # One-hot encoding
        device = inputs.device
        encoding_one_hot = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=device)
        encoding_one_hot.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = torch.matmul(encoding_one_hot, self.embedding.weight)
        quantized = quantized.view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Perplexity
        avg_probs = torch.mean(encoding_one_hot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.epsilon)))
        
        return {
            'quantized': quantized,
            'loss': loss,
            'indices': encoding_indices.view(input_shape[:-1]),
            'perplexity': perplexity
        }

class ConfigFirstResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer with config-first interface"""
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__()
        
        self.num_quantizers = config.quantization.num_quantizers
        self.codebook_size = config.quantization.codebook_size
        self.codebook_dim = config.quantization.codebook_dim
        
        # Create quantizer layers
        self.quantizers = nn.ModuleList([
            ConfigFirstVectorQuantizer(config)
            for _ in range(self.num_quantizers)
        ])
    
    def forward(self, x: torch.Tensor) -> dict:
        """Residual vector quantization"""
        residual = x
        quantized = torch.zeros_like(x)
        indices = []
        total_loss = 0.0
        total_perplexity = 0.0
        
        for quantizer in self.quantizers:
            quant_out = quantizer(residual)
            quantized = quantized + quant_out['quantized']
            residual = residual - quant_out['quantized']
            indices.append(quant_out['indices'])
            total_loss = total_loss + quant_out['loss']
            total_perplexity = total_perplexity + quant_out['perplexity']
        
        return {
            'quantized': quantized,
            'indices': torch.stack(indices, dim=1),
            'loss': total_loss,
            'perplexity': total_perplexity / self.num_quantizers,
            'residual': residual
        }
