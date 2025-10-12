import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AdaptiveComputationTime(nn.Module):
    """ACT mechanism for dynamic computation depth"""
    def __init__(self, hidden_size, max_steps=10, threshold=0.01, epsilon=0.01):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_steps = max_steps
        self.threshold = threshold
        self.epsilon = epsilon
        
        # Halting unit - produces halting probability
        self.halting_unit = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, compute_fn):
        """
        x: Input tensor (batch_size, seq_len, hidden_size)
        compute_fn: Function that takes state and returns new state
        """
        batch_size, seq_len, hidden_size = x.shape
        device = x.device
        
        # Initialize
        halting_probability = torch.zeros(batch_size, seq_len, 1, device=device)
        remainders = torch.zeros(batch_size, seq_len, 1, device=device)
        n_updates = torch.zeros(batch_size, seq_len, 1, device=device)
        
        # States
        previous_state = torch.zeros_like(x)
        step = 0
        
        # Accumulate weighted states
        accumulated_state = torch.zeros_like(x)
        
        while step < self.max_steps:
            # Compute new state
            state = compute_fn(x + previous_state)
            
            # Compute halting probability for this step
            p = self.halting_unit(state)
            
            # Mask for elements that haven't halted
            still_running = (halting_probability < 1.0 - self.epsilon).float()
            
            # New halting probability
            new_halted = (halting_probability + p * still_running) > self.threshold
            
            # Update halting probability and remainders
            halting_probability = halting_probability + p * still_running
            remainders = remainders + new_halted.float() * (1 - halting_probability)
            halting_probability = halting_probability + new_halted.float() * remainders
            
            # Update n_updates
            n_updates = n_updates + still_running
            
            # Accumulate weighted state
            update_weights = p * still_running + new_halted.float() * remainders
            accumulated_state = accumulated_state + update_weights * state
            
            # Update previous state
            previous_state = state
            
            # Check if all elements have halted
            if (halting_probability >= 1.0 - self.epsilon).all():
                break
                
            step += 1
        
        # Ponder cost (encourage less computation)
        ponder_cost = n_updates.mean()
        
        return {
            'output': accumulated_state,
            'ponder_cost': ponder_cost,
            'n_updates': n_updates,
            'halting_probabilities': halting_probability
        }


class PonderNet(nn.Module):
    """PonderNet - learned halting with regularization"""
    def __init__(self, input_size, hidden_size, output_size, max_steps=10, 
                 lambda_p=0.01, allow_halting=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_steps = max_steps
        self.lambda_p = lambda_p
        self.allow_halting = allow_halting
        
        # Step function (can be any recurrent cell)
        self.cell = nn.LSTMCell(input_size, hidden_size)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Lambda prediction layer (predicts halting probability)
        self.lambda_layer = nn.Linear(hidden_size, 1)
        
    def forward(self, x, return_all_steps=False):
        """
        x: Input tensor (batch_size, input_size)
        """
        batch_size = x.size(0)
        device = x.device
        
        # Initialize hidden and cell states
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)
        
        # Track outputs and probabilities
        outputs = []
        p_values = []
        
        # Probability of not halting until step n
        p_cumulative = torch.ones(batch_size, 1, device=device)
        
        # Final output accumulator
        y_cumulative = torch.zeros(batch_size, self.output_size, device=device)
        
        for n in range(1, self.max_steps + 1):
            # Compute step
            h, c = self.cell(x, (h, c))
            
            # Compute output for this step
            y_n = self.output_layer(h)
            
            # Compute halting probability
            lambda_n = torch.sigmoid(self.lambda_layer(h))
            
            # Probability of halting at exactly step n
            if n < self.max_steps:
                p_n = p_cumulative * lambda_n
            else:
                # Force halt at max steps
                p_n = p_cumulative
            
            # Update cumulative probability
            p_cumulative = p_cumulative * (1 - lambda_n)
            
            # Accumulate weighted output
            y_cumulative = y_cumulative + p_n * y_n
            
            # Store for analysis
            outputs.append(y_n)
            p_values.append(p_n)
            
            # Early stopping if all samples have halted
            if self.allow_halting and (p_cumulative < 1e-5).all():
                break
        
        # Compute regularization loss
        p_g = torch.stack(p_values, dim=1)  # (batch_size, n_steps, 1)
        p_g = p_g.squeeze(-1)  # (batch_size, n_steps)
        
        # KL divergence between p_g and geometric distribution
        n_steps = p_g.size(1)
        n_values = torch.arange(1, n_steps + 1, device=device).float()
        
        # Expected value of n under p_g
        E_n = (p_g * n_values).sum(dim=1)
        
        # Geometric distribution with same mean
        geometric_lambda = 1.0 / E_n
        geometric_probs = geometric_lambda.unsqueeze(1) * (
            (1 - geometric_lambda).unsqueeze(1) ** (n_values - 1)
        )
        
        # KL divergence
        kl_div = (p_g * (torch.log(p_g + 1e-10) - torch.log(geometric_probs + 1e-10))).sum(dim=1)
        regularization_loss = self.lambda_p * kl_div.mean()
        
        if return_all_steps:
            return {
                'output': y_cumulative,
                'all_outputs': torch.stack(outputs, dim=1),
                'halt_probabilities': torch.stack(p_values, dim=1),
                'regularization_loss': regularization_loss,
                'expected_steps': E_n.mean()
            }
        else:
            return y_cumulative, regularization_loss


class UniversalTransformer(nn.Module):
    """Universal Transformer with adaptive computation"""
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, max_steps=6,
                 dropout=0.1, use_act=True, act_threshold=0.01):
        super().__init__()
        self.d_model = d_model
        self.max_steps = max_steps
        self.use_act = use_act
        
        # Import transformer block
        from modules.transformer_block import TransformerBlock
        
        # Single transformer block (reused multiple times)
        self.transformer_block = TransformerBlock(d_model, n_heads, d_ff, dropout)
        
        # Position embeddings for steps
        self.step_embeddings = nn.Embedding(max_steps, d_model)
        
        # ACT mechanism
        if use_act:
            self.act = AdaptiveComputationTime(d_model, max_steps, act_threshold)
        
        # Layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
    def compute_step(self, state, step_idx, mask=None):
        """Single computation step"""
        # Add step embedding
        step_emb = self.step_embeddings(
            torch.full((state.size(0), state.size(1)), step_idx, 
                      device=state.device, dtype=torch.long)
        )
        state_with_step = state + step_emb
        
        # Apply transformer block
        output = self.transformer_block(state_with_step, mask=mask)
        
        return output
    
    def forward(self, x, mask=None, return_steps=False):
        """
        x: Input tensor (batch_size, seq_len, d_model)
        mask: Optional attention mask
        """
        if self.use_act:
            # Adaptive computation
            def compute_fn(state):
                # Find current step based on computation history
                # Simplified - just use step 0 embedding
                return self.compute_step(state, 0, mask)
            
            output_dict = self.act(x, compute_fn)
            output = output_dict['output']
            ponder_cost = output_dict['ponder_cost']
            
            output = self.final_norm(output)
            
            if return_steps:
                return {
                    'output': output,
                    'ponder_cost': ponder_cost,
                    'n_updates': output_dict['n_updates']
                }
            else:
                return output
        else:
            # Fixed computation steps
            state = x
            all_states = []
            
            for step in range(self.max_steps):
                state = self.compute_step(state, step, mask)
                all_states.append(state)
            
            output = self.final_norm(state)
            
            if return_steps:
                return {
                    'output': output,
                    'all_states': torch.stack(all_states, dim=1)
                }
            else:
                return output