import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 dilation=1, causal=True, dropout=0.1):
        super().__init__()
        self.causal = causal
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Calculate padding for causal convolution
        if causal:
            self.padding = (kernel_size - 1) * dilation
        else:
            self.padding = ((kernel_size - 1) * dilation) // 2
        
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=self.padding, dilation=dilation
        )
        
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch, channels, time)
        out = self.conv(x)
        
        # Remove future timesteps for causal convolution
        if self.causal and self.padding > 0:
            out = out[:, :, :-self.padding]
        
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        return out


class WaveNetBlock(nn.Module):
    def __init__(self, residual_channels, skip_channels, kernel_size=2, 
                 dilation=1, dropout=0.1):
        super().__init__()
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        
        # Dilated causal convolution
        self.dilated_conv = nn.Conv1d(
            residual_channels, 2 * residual_channels, kernel_size,
            padding=(kernel_size - 1) * dilation, dilation=dilation
        )
        
        # 1x1 convolutions for residual and skip connections
        self.residual_conv = nn.Conv1d(residual_channels, residual_channels, 1)
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch, residual_channels, time)
        residual = x
        
        # Dilated convolution
        out = self.dilated_conv(x)
        out = out[:, :, :x.size(2)]  # Remove padding for causality
        
        # Gated activation
        gate = torch.sigmoid(out[:, :self.residual_channels])
        filter = torch.tanh(out[:, self.residual_channels:])
        out = gate * filter
        
        # Skip connection
        skip = self.skip_conv(out)
        
        # Residual connection
        out = self.residual_conv(out)
        out = self.dropout(out)
        out = out + residual
        
        return out, skip


class PositionalEncodingTime(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0).transpose(1, 2))
        
    def forward(self, x):
        # x shape: (batch, channels, time)
        return x + self.pe[:, :, :x.size(2)]


class TimeSeriesEncoder(nn.Module):
    def __init__(self, input_dim, d_model=256, n_layers=6, kernel_size=3,
                 architecture='temporal_conv', dropout=0.1, 
                 use_positional=True, pooling='adaptive'):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.architecture = architecture
        self.use_positional = use_positional
        self.pooling = pooling
        
        # Input projection
        self.input_projection = nn.Conv1d(input_dim, d_model, 1)
        
        # Positional encoding
        if use_positional:
            self.pos_encoding = PositionalEncodingTime(d_model)
        
        # Build encoder based on architecture
        if architecture == 'temporal_conv':
            # TCN-style architecture with exponentially increasing dilation
            self.encoder_blocks = nn.ModuleList()
            for i in range(n_layers):
                dilation = 2 ** i
                self.encoder_blocks.append(
                    TemporalConvBlock(d_model, d_model, kernel_size, 
                                      dilation=dilation, causal=True, dropout=dropout)
                )
        
        elif architecture == 'wavenet':
            # WaveNet-style architecture with skip connections
            self.skip_channels = d_model
            self.encoder_blocks = nn.ModuleList()
            for i in range(n_layers):
                dilation = 2 ** (i % 10)  # Reset dilation every 10 layers
                self.encoder_blocks.append(
                    WaveNetBlock(d_model, self.skip_channels, kernel_size, 
                                 dilation=dilation, dropout=dropout)
                )
            # Final skip projection
            self.skip_projection = nn.Conv1d(self.skip_channels, d_model, 1)
            
        elif architecture == 'transformer':
            # Use transformer blocks for time series
            from modules.transformer_block import TransformerBlock
            self.encoder_blocks = nn.ModuleList([
                TransformerBlock(d_model, n_heads=8, d_ff=d_model*4, dropout=dropout)
                for _ in range(n_layers)
            ])
        
        elif architecture == 'conv_transformer':
            # Hybrid: Conv layers followed by transformer
            conv_layers = n_layers // 2
            transformer_layers = n_layers - conv_layers
            
            self.encoder_blocks = nn.ModuleList()
            # Convolutional layers
            for i in range(conv_layers):
                dilation = 2 ** i
                self.encoder_blocks.append(
                    TemporalConvBlock(d_model, d_model, kernel_size, 
                                      dilation=dilation, causal=True, dropout=dropout)
                )
            # Transformer layers
            from modules.transformer_block import TransformerBlock
            for _ in range(transformer_layers):
                self.encoder_blocks.append(
                    TransformerBlock(d_model, n_heads=8, d_ff=d_model*4, dropout=dropout)
                )
        
        # Final normalization (only for transformer)
        if architecture in ['transformer']:
            self.final_norm = nn.LayerNorm(d_model)
        
        # Pooling layers
        if pooling == 'adaptive':
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pooling == 'max':
            self.pool = nn.AdaptiveMaxPool1d(1)
        elif pooling == 'attention':
            self.attention_pool = nn.Linear(d_model, 1)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None, return_sequence=False):
        # x shape: (batch, time, input_dim) or (batch, input_dim, time)
        
        # Ensure correct format (batch, channels, time)
        if x.dim() == 3 and x.size(-1) == self.input_dim:
            x = x.transpose(1, 2)
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        if self.use_positional:
            x = self.pos_encoding(x)
        
        # Apply encoder blocks
        if self.architecture == 'wavenet':
            skip_connections = []
            for block in self.encoder_blocks:
                x, skip = block(x)
                skip_connections.append(skip)
            # Sum skip connections
            x = sum(skip_connections)
            x = F.relu(x)
            x = self.skip_projection(x)
            
        elif self.architecture in ['temporal_conv', 'conv_transformer']:
            for i, block in enumerate(self.encoder_blocks):
                if hasattr(block, 'conv'):  # Conv block
                    x = block(x)
                else:  # Transformer block
                    # Convert to (batch, time, channels) for transformer
                    x = x.transpose(1, 2)
                    x = block(x, mask=mask)
                    x = x.transpose(1, 2)
                    
        elif self.architecture == 'transformer':
            # Convert to (batch, time, channels) for transformer
            x = x.transpose(1, 2)
            for block in self.encoder_blocks:
                x = block(x, mask=mask)
            if hasattr(self, 'final_norm'):
                x = self.final_norm(x)
            x = x.transpose(1, 2)
        
        else:  # temporal_conv
            for block in self.encoder_blocks:
                # Residual connection
                if x.shape == block(x).shape:
                    x = x + block(x)
                else:
                    x = block(x)
        
        # Pool to get single representation
        if self.pooling == 'attention':
            # Attention-based pooling
            x_t = x.transpose(1, 2)  # (batch, time, channels)
            attn_weights = self.attention_pool(x_t).squeeze(-1)  # (batch, time)
            if mask is not None:
                attn_weights = attn_weights.masked_fill(mask.squeeze(1).squeeze(1) == 0, -1e9)
            attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(1)  # (batch, 1, time)
            pooled = torch.bmm(attn_weights, x_t).squeeze(1)  # (batch, channels)
        else:
            pooled = self.pool(x).squeeze(-1)  # (batch, channels)
        
        # Output projection
        output = self.output_projection(pooled)
        
        if return_sequence:
            return {
                'pooled': output,
                'sequence': x,
                'shape': x.shape
            }
        else:
            return output
    
    def get_receptive_field(self):
        """Calculate the receptive field of the encoder"""
        if self.architecture == 'temporal_conv':
            rf = 1
            for i in range(len(self.encoder_blocks)):
                dilation = 2 ** i
                rf += 2 * dilation
            return rf
        elif self.architecture == 'wavenet':
            # WaveNet has exponentially growing receptive field
            return 2 ** len(self.encoder_blocks)
        else:
            return -1  # Transformer has global receptive field