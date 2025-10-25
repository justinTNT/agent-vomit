"""SequenceToSequenceModel - Full encoder-decoder architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class SequenceToSequenceModel(nn.Module):
    """Full encoder-decoder architecture for sequence-to-sequence tasks.
    
    This module combines encoder and decoder modules to create a complete
    sequence-to-sequence model. It supports both with and without attention
    mechanisms and handles teacher forcing during training.
    
    Args:
        encoder: Encoder module (must have a forward method returning hidden states)
        decoder: Decoder module (must support step-by-step decoding)
        use_attention: Whether to use attention mechanism (default: True)
        max_length: Maximum sequence length for inference (default: 100)
        sos_token_id: Start-of-sequence token ID (default: 0)
        eos_token_id: End-of-sequence token ID (default: 1)
        **kwargs: Additional keyword arguments
    
    Returns:
        Dict containing:
            - outputs: Decoded output logits [batch_size, target_length, vocab_size]
            - predictions: Predicted token IDs [batch_size, target_length]
            - encoder_hidden: Final encoder hidden states
            - attention_weights: Attention weights (if using attention) [batch_size, target_length, source_length]
            - loss: Cross-entropy loss (if target is provided)
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        use_attention: bool = True,
        max_length: int = 100,
        sos_token_id: int = 0,
        eos_token_id: int = 1,
        **kwargs
    ):
        super().__init__()
        
        # Validate inputs
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
        
        self.encoder = encoder
        self.decoder = decoder
        self.use_attention = use_attention
        self.max_length = max_length
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        
        # Check if decoder supports attention
        if use_attention and not hasattr(decoder, 'attention'):
            raise ValueError("Decoder must have attention mechanism when use_attention=True")
    
    def encode(
        self,
        source: torch.Tensor,
        source_lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Encode source sequence.
        
        Args:
            source: Source sequence tensor [batch_size, source_length] or [batch_size, source_length, features]
            source_lengths: Actual lengths of source sequences [batch_size]
        
        Returns:
            Dictionary with encoder outputs
        """
        # Call encoder
        if hasattr(self.encoder, 'forward') and callable(self.encoder.forward):
            encoder_output = self.encoder(source)
        else:
            raise ValueError("Encoder must have a forward method")
        
        # Handle different encoder output formats
        if isinstance(encoder_output, dict):
            return encoder_output
        elif isinstance(encoder_output, tuple):
            # Assume (hidden_states, cell_states) or (hidden_states,)
            return {
                'hidden_states': encoder_output[0],
                'cell_states': encoder_output[1] if len(encoder_output) > 1 else None,
                'encoder_outputs': encoder_output[0]  # For attention
            }
        else:
            # Single tensor output
            return {
                'hidden_states': encoder_output,
                'encoder_outputs': encoder_output
            }
    
    def decode_step(
        self,
        input_token: torch.Tensor,
        hidden_states: torch.Tensor,
        cell_states: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[torch.Tensor] = None,
        source_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Perform single decoding step.
        
        Args:
            input_token: Input token for current step [batch_size] or [batch_size, 1]
            hidden_states: Hidden states from previous step
            cell_states: Cell states from previous step (for LSTM)
            encoder_outputs: All encoder outputs (for attention)
            source_mask: Source sequence mask
        
        Returns:
            Tuple of (output_logits, new_hidden_states, new_cell_states, attention_weights)
        """
        # Ensure input_token has correct shape
        if input_token.dim() == 1:
            input_token = input_token.unsqueeze(1)  # [batch_size, 1]
        
        # Prepare decoder input
        decoder_input = {
            'input': input_token,
            'hidden': hidden_states
        }
        
        if cell_states is not None:
            decoder_input['cell'] = cell_states
        
        if self.use_attention and encoder_outputs is not None:
            decoder_input['encoder_outputs'] = encoder_outputs
            if source_mask is not None:
                decoder_input['mask'] = source_mask
        
        # Call decoder
        if hasattr(self.decoder, 'forward_step'):
            decoder_output = self.decoder.forward_step(**decoder_input)
        elif hasattr(self.decoder, 'forward'):
            decoder_output = self.decoder(**decoder_input)
        else:
            raise ValueError("Decoder must have either forward_step or forward method")
        
        # Handle different decoder output formats
        if isinstance(decoder_output, dict):
            output_logits = decoder_output.get('logits', decoder_output.get('output'))
            hidden_states = decoder_output.get('hidden', hidden_states)
            cell_states = decoder_output.get('cell', cell_states)
            attention_weights = decoder_output.get('attention_weights', None)
        elif isinstance(decoder_output, tuple):
            # Assume (output, hidden, cell, attention) or similar
            output_logits = decoder_output[0]
            hidden_states = decoder_output[1] if len(decoder_output) > 1 else hidden_states
            cell_states = decoder_output[2] if len(decoder_output) > 2 else cell_states
            attention_weights = decoder_output[3] if len(decoder_output) > 3 else None
        else:
            output_logits = decoder_output
            attention_weights = None
        
        return output_logits, hidden_states, cell_states, attention_weights
    
    def forward(
        self,
        source: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        source_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the sequence-to-sequence model.
        
        Args:
            source: Source sequence tensor
            target: Target sequence tensor (for training)
            source_lengths: Actual lengths of source sequences
            target_lengths: Actual lengths of target sequences
            teacher_forcing_ratio: Probability of using teacher forcing (default: 1.0)
        
        Returns:
            Dictionary containing outputs, predictions, hidden states, attention weights, and loss
        """
        batch_size = source.shape[0]
        device = source.device
        
        # Encode source sequence
        encoder_output = self.encode(source, source_lengths)
        hidden_states = encoder_output['hidden_states']
        cell_states = encoder_output.get('cell_states', None)
        encoder_outputs = encoder_output.get('encoder_outputs', None)
        
        # Create source mask if needed
        if source_lengths is not None and self.use_attention:
            max_source_len = source.shape[1]
            source_mask = torch.arange(max_source_len, device=device).expand(
                batch_size, max_source_len
            ) < source_lengths.unsqueeze(1)
        else:
            source_mask = None
        
        # Determine target length
        if target is not None:
            target_length = target.shape[1]
        else:
            target_length = self.max_length
        
        # Initialize outputs
        outputs = []
        predictions = []
        attention_weights_list = []
        
        # Start with SOS token
        input_token = torch.full((batch_size,), self.sos_token_id, dtype=torch.long, device=device)
        
        # Decode step by step
        for t in range(target_length):
            # Decode one step
            output_logits, hidden_states, cell_states, attention_weights = self.decode_step(
                input_token, hidden_states, cell_states, encoder_outputs, source_mask
            )
            
            outputs.append(output_logits)
            
            # Get predictions
            predicted = output_logits.argmax(dim=-1).squeeze(1)
            predictions.append(predicted)
            
            if attention_weights is not None:
                attention_weights_list.append(attention_weights)
            
            # Determine next input
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # Teacher forcing: use ground truth
                input_token = target[:, t]
            else:
                # Use predicted token
                input_token = predicted
            
            # Stop if all sequences have produced EOS (during inference)
            if target is None and (predicted == self.eos_token_id).all():
                break
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=1)  # [batch_size, target_length, vocab_size]
        predictions = torch.stack(predictions, dim=1)  # [batch_size, target_length]
        
        if attention_weights_list:
            attention_weights = torch.stack(attention_weights_list, dim=1)  # [batch_size, target_length, source_length]
        else:
            attention_weights = None
        
        # Prepare output dictionary
        output_dict = {
            'outputs': outputs,
            'predictions': predictions,
            'encoder_hidden': encoder_output.get('hidden_states'),
        }
        
        if attention_weights is not None:
            output_dict['attention_weights'] = attention_weights
        
        # Compute loss if target is provided
        if target is not None:
            # Flatten outputs and target for loss computation
            outputs_flat = outputs.reshape(-1, outputs.shape[-1])
            target_flat = target.reshape(-1)
            
            # Create mask for padding if target_lengths provided
            if target_lengths is not None:
                target_mask = torch.arange(target_length, device=device).expand(
                    batch_size, target_length
                ) < target_lengths.unsqueeze(1)
                target_mask_flat = target_mask.reshape(-1)
                
                # Compute masked loss
                loss = F.cross_entropy(
                    outputs_flat[target_mask_flat],
                    target_flat[target_mask_flat],
                    ignore_index=-100
                )
            else:
                # Compute loss without masking
                loss = F.cross_entropy(outputs_flat, target_flat, ignore_index=-100)
            
            output_dict['loss'] = loss
        
        return output_dict