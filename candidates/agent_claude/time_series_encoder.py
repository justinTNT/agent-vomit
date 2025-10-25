import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import math


class TimeSeriesEncoder(nn.Module):
    def __init__(self,
                 input_dim: int = 1,                # Input dimension per timestep
                 hidden_dim: int = 256,             # Hidden dimension
                 n_layers: int = 6,                 # Number of layers
                 kernel_size: int = 3,              # Kernel size for TCN/WaveNet
                 architecture: str = 'transformer', # 'tcn', 'wavenet', or 'transformer'
                 n_heads: int = 8,                  # For transformer
                 dropout: float = 0.1,              # Dropout rate
                 max_seq_len: int = 5000,           # Maximum sequence length
                 **kwargs):                         # REQUIRED: Accept extra kwargs
        super().__init__()
        
        # Log unknown parameters
        if kwargs:
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.architecture = architecture
        self.max_seq_len = max_seq_len
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Architecture-specific layers
        if architecture == 'transformer':
            # Positional encoding for transformer
            self.positional_encoding = self._create_positional_encoding(max_seq_len, hidden_dim)
            
            # Transformer encoder layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            
        elif architecture == 'tcn':
            # Temporal Convolutional Network
            self.tcn_layers = nn.ModuleList()
            self.residual_convs = nn.ModuleList()
            
            for i in range(n_layers):
                dilation = 2 ** i
                padding = (kernel_size - 1) * dilation
                
                # Dilated causal convolution
                self.tcn_layers.append(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size,
                             padding=padding, dilation=dilation)
                )
                
                # Residual connection if needed
                if i == 0:
                    self.residual_convs.append(None)
                else:
                    self.residual_convs.append(None)  # Direct connection
            
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            
        elif architecture == 'wavenet':
            # WaveNet-style architecture
            self.wavenet_layers = nn.ModuleList()
            self.skip_connections = nn.ModuleList()
            
            for i in range(n_layers):
                dilation = 2 ** i
                
                # Gated convolution block
                self.wavenet_layers.append(
                    WaveNetBlock(hidden_dim, kernel_size, dilation, dropout)
                )
                
                # Skip connection
                self.skip_connections.append(nn.Conv1d(hidden_dim, hidden_dim, 1))
            
            # Output layers
            self.output_conv1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
            self.output_conv2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
            
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pos_encoding', pe.unsqueeze(0))
        return pe
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> dict:
        """
        Forward pass of the time series encoder.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            mask: Optional boolean mask (batch, seq_len) where True = keep
        
        Returns:
            Dictionary with keys:
                - 'encoded': Encoded sequence (batch, seq_len, hidden_dim)
                - 'pooled': Global pooled representation (batch, hidden_dim)
                - 'last_hidden': Last timestep hidden state (batch, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        if self.architecture == 'transformer':
            # Add positional encoding
            if seq_len > self.max_seq_len:
                raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")
            
            x = x + self.pos_encoding[:, :seq_len, :]
            
            # Create attention mask for transformer
            src_key_padding_mask = None
            if mask is not None:
                # Transformer expects True = ignore, so invert boolean mask
                src_key_padding_mask = ~mask
            
            # Encode
            encoded = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
            
        elif self.architecture == 'tcn':
            # TCN expects (batch, channels, seq_len)
            x = x.transpose(1, 2)
            
            # Apply TCN layers with causal masking
            for i, (conv, res_conv) in enumerate(zip(self.tcn_layers, self.residual_convs)):
                # Causal convolution (truncate future)
                residual = x
                x = conv(x)
                
                # Causal truncation
                dilation = 2 ** i
                truncate = (conv.kernel_size[0] - 1) * dilation
                if truncate > 0:
                    x = x[:, :, :-truncate]
                
                x = self.activation(x)
                x = self.dropout(x)
                
                # Residual connection
                x = x + residual
            
            # Back to (batch, seq_len, channels)
            encoded = x.transpose(1, 2)
            
        elif self.architecture == 'wavenet':
            # WaveNet expects (batch, channels, seq_len)
            x = x.transpose(1, 2)
            
            # Apply WaveNet layers
            skip_sum = 0
            for layer, skip_conv in zip(self.wavenet_layers, self.skip_connections):
                x, skip = layer(x)
                skip_sum = skip_sum + skip_conv(skip)
            
            # Output processing
            x = F.relu(skip_sum)
            x = self.output_conv1(x)
            x = F.relu(x)
            x = self.output_conv2(x)
            
            # Back to (batch, seq_len, channels)
            encoded = x.transpose(1, 2)
        
        # Layer normalization
        encoded = self.layer_norm(encoded)
        
        # Compute pooled and last hidden representations
        if mask is not None:
            # Masked mean pooling
            mask_expanded = mask.unsqueeze(-1).expand_as(encoded)
            sum_encoded = (encoded * mask_expanded).sum(dim=1)
            pooled = sum_encoded / mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
            
            # Get last valid hidden state
            lengths = mask.sum(dim=1)
            last_hidden = encoded[torch.arange(batch_size), lengths - 1]
        else:
            pooled = encoded.mean(dim=1)
            last_hidden = encoded[:, -1]
        
        return {
            'encoded': encoded,
            'pooled': pooled,
            'last_hidden': last_hidden
        }


class WaveNetBlock(nn.Module):
    """WaveNet-style gated convolution block."""
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Dilated causal convolutions for gate and filter
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, 2 * channels, kernel_size,
                             padding=padding, dilation=dilation)
        
        # 1x1 convolution for residual
        self.residual_conv = nn.Conv1d(channels, channels, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass returning (output, skip_connection)."""
        residual = x
        
        # Gated convolution
        x = self.conv(x)
        
        # Causal truncation
        truncate = (self.kernel_size - 1) * self.dilation
        if truncate > 0:
            x = x[:, :, :-truncate]
        
        # Split into filter and gate
        filter, gate = x.chunk(2, dim=1)
        x = torch.tanh(filter) * torch.sigmoid(gate)
        
        # Skip connection
        skip = x
        
        # Residual connection
        x = self.residual_conv(x)
        x = self.dropout(x)
        x = x + residual
        
        return x, skip