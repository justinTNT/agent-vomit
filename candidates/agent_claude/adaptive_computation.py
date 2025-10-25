import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from typing import List, Optional, Tuple


class AdaptiveComputation(nn.Module):
    def __init__(self,
                 input_dim: int = 512,              # Input dimension
                 hidden_dim: int = 512,             # Hidden dimension
                 n_layers: int = 12,                # Maximum number of layers
                 threshold: float = 0.5,            # Exit threshold
                 ponder_cost: float = 0.01,         # Penalty for using more layers
                 exit_method: str = 'threshold',    # 'threshold', 'learned', or 'stochastic'
                 min_layers: int = 3,               # Minimum layers before exit
                 **kwargs):                         # REQUIRED: Accept extra kwargs
        super().__init__()
        
        # Log unknown parameters
        if kwargs:
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.threshold = threshold
        self.ponder_cost = ponder_cost
        self.exit_method = exit_method
        self.min_layers = min_layers
        
        # Processing layers (could be any architecture)
        self.layers = nn.ModuleList([
            TransformLayer(hidden_dim, hidden_dim)
            for _ in range(n_layers)
        ])
        
        # Exit predictors for each layer
        self.exit_predictors = nn.ModuleList([
            ExitPredictor(hidden_dim, method=exit_method)
            for _ in range(n_layers)
        ])
        
        # Optional: Layer-specific output heads
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_layers)
        ])
        
        # Halting unit for ACT (Adaptive Computation Time)
        if exit_method == 'act':
            self.halting_unit = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, 
                force_full_depth: bool = False) -> dict:
        """
        Forward pass with adaptive computation.
        
        Args:
            x: Input tensor (batch, input_dim)
            force_full_depth: Force computation through all layers
        
        Returns:
            Dictionary with keys:
                - 'output': Final output (batch, hidden_dim)
                - 'exit_layer': Layer index where each sample exited
                - 'exit_probabilities': Exit probabilities at each layer
                - 'ponder_cost': Average computational cost
                - 'layer_outputs': Outputs from each layer (if computed)
        """
        batch_size = x.size(0)
        device = x.device
        
        # Initialize tracking
        exited = torch.zeros(batch_size, dtype=torch.bool, device=device)
        exit_layer = torch.zeros(batch_size, dtype=torch.long, device=device)
        accumulated_output = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        layer_outputs = []
        exit_probs = []
        
        # For ACT method
        if self.exit_method == 'act':
            halting_prob = torch.zeros(batch_size, 1, device=device)
            remainder = torch.ones(batch_size, 1, device=device)
            n_updates = torch.zeros(batch_size, 1, device=device)
        
        # Process through layers
        hidden = x
        for i in range(self.n_layers):
            # Apply layer transformation
            hidden = self.layers[i](hidden)
            
            # Get exit probability for this layer
            exit_prob = self.exit_predictors[i](hidden)
            exit_probs.append(exit_prob)
            
            # Get output for this layer
            layer_output = self.output_heads[i](hidden)
            layer_outputs.append(layer_output)
            
            # Check exit conditions
            if not force_full_depth and i >= self.min_layers - 1:
                if self.exit_method == 'threshold':
                    # Simple threshold-based exit
                    should_exit = (exit_prob > self.threshold).squeeze(-1)
                    new_exits = should_exit & ~exited
                    
                    if new_exits.any():
                        accumulated_output[new_exits] = layer_output[new_exits]
                        exit_layer[new_exits] = i
                        exited[new_exits] = True
                    
                    # Check if all samples have exited
                    if exited.all():
                        break
                
                elif self.exit_method == 'learned':
                    # Learned exit with Gumbel-Softmax
                    exit_decision = F.gumbel_softmax(
                        torch.cat([1 - exit_prob, exit_prob], dim=-1),
                        tau=1.0,
                        hard=True
                    )[:, 1]  # Take exit decision
                    
                    new_exits = (exit_decision > 0.5) & ~exited
                    
                    if new_exits.any():
                        accumulated_output[new_exits] = layer_output[new_exits]
                        exit_layer[new_exits] = i
                        exited[new_exits] = True
                    
                    if exited.all():
                        break
                
                elif self.exit_method == 'act':
                    # Adaptive Computation Time mechanism
                    h = self.halting_unit(hidden)
                    
                    # Update halting probability
                    still_running = ~exited.float().unsqueeze(1)
                    new_halting = remainder * h * still_running
                    
                    # Accumulate output
                    accumulated_output += (new_halting * layer_output).squeeze(1)
                    
                    # Update state
                    halting_prob += new_halting
                    remainder = remainder * (1 - h) * still_running
                    n_updates += still_running
                    
                    # Check which samples should halt
                    should_halt = (halting_prob > 1 - 1e-5).squeeze(-1)
                    new_exits = should_halt & ~exited
                    
                    if new_exits.any():
                        # Add remainder
                        accumulated_output[new_exits] += (
                            remainder[new_exits] * layer_output[new_exits]
                        ).squeeze(1)
                        exit_layer[new_exits] = i
                        exited[new_exits] = True
                    
                    if exited.all():
                        break
        
        # Handle any samples that didn't exit
        if not exited.all():
            not_exited = ~exited
            accumulated_output[not_exited] = layer_outputs[-1][not_exited]
            exit_layer[not_exited] = self.n_layers - 1
        
        # Normalize output
        output = self.layer_norm(accumulated_output)
        
        # Compute ponder cost (average number of layers used)
        if self.exit_method == 'act' and 'n_updates' in locals():
            avg_ponder = n_updates.mean()
        else:
            avg_ponder = (exit_layer.float() + 1).mean()
        
        ponder_cost = self.ponder_cost * avg_ponder
        
        # Stack exit probabilities
        exit_probabilities = torch.stack(exit_probs, dim=1)  # (batch, n_layers, 1)
        
        return {
            'output': output,
            'exit_layer': exit_layer,
            'exit_probabilities': exit_probabilities.squeeze(-1),
            'ponder_cost': ponder_cost,
            'layer_outputs': torch.stack(layer_outputs, dim=1) if layer_outputs else None
        }
    
    def compute_adaptive_loss(self, output_dict: dict, targets: torch.Tensor) -> dict:
        """
        Compute loss with adaptive computation penalty.
        
        Args:
            output_dict: Output from forward pass
            targets: Target values
        
        Returns:
            Dictionary with loss components
        """
        # Main task loss (example: MSE)
        task_loss = F.mse_loss(output_dict['output'], targets)
        
        # Ponder penalty
        ponder_penalty = output_dict['ponder_cost']
        
        # Total loss
        total_loss = task_loss + ponder_penalty
        
        return {
            'loss': total_loss,
            'task_loss': task_loss,
            'ponder_penalty': ponder_penalty,
            'avg_exit_layer': output_dict['exit_layer'].float().mean()
        }


class TransformLayer(nn.Module):
    """Single transformation layer."""
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.transform = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim)
        )
        
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x
        x = self.transform(x)
        x = self.dropout(x)
        x = x + residual
        x = self.norm(x)
        return x


class ExitPredictor(nn.Module):
    """Predicts whether to exit at current layer."""
    def __init__(self, input_dim: int, method: str = 'threshold'):
        super().__init__()
        
        self.method = method
        
        if method in ['threshold', 'learned', 'stochastic']:
            # Simple exit predictor
            self.predictor = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, 1),
                nn.Sigmoid()
            )
        elif method == 'act':
            # ACT doesn't use this predictor
            self.predictor = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict exit probability."""
        if self.predictor is not None:
            return self.predictor(x)
        else:
            # For ACT, return dummy value
            return torch.ones(x.size(0), 1, device=x.device) * 0.5