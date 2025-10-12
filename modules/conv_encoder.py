import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 use_pooling=True, use_batchnorm=True, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.use_batchnorm = use_batchnorm
        self.use_pooling = use_pooling
        
        if use_batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)
        if use_pooling:
            self.pool = nn.MaxPool2d(2, 2)
        
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x):
        x = self.conv(x)
        if self.use_batchnorm:
            x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        if self.use_pooling:
            x = self.pool(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual  # Residual connection
        x = F.relu(x)
        return x


class ConvEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_layers=4, 
                 use_residual=True, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_layers = num_layers
        self.use_residual = use_residual
        
        layers = []
        current_channels = in_channels
        
        for i in range(num_layers):
            out_channels = base_channels * (2 ** i)
            
            # Add convolutional block
            layers.append(ConvBlock(
                current_channels, 
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                use_pooling=(i < num_layers - 1),  # No pooling on last layer
                use_batchnorm=True,
                dropout=dropout
            ))
            
            # Add residual blocks if enabled
            if use_residual and i > 0:
                layers.append(ResidualBlock(out_channels, dropout=dropout))
            
            current_channels = out_channels
        
        self.encoder = nn.Sequential(*layers)
        self.output_channels = current_channels
        
        # Global average pooling to get fixed-size output
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        # Encode through conv layers
        features = self.encoder(x)
        
        # Return both spatial features and pooled features
        pooled = self.global_pool(features).flatten(1)
        
        return {
            'features': features,
            'pooled': pooled,
            'shape': features.shape
        }
    
    def get_feature_shape(self, input_shape):
        """Calculate output feature shape given input shape"""
        h, w = input_shape[-2:]
        # Account for pooling layers (each reduces by factor of 2)
        num_pools = self.num_layers - 1
        h_out = h // (2 ** num_pools)
        w_out = w // (2 ** num_pools)
        return (self.output_channels, h_out, w_out)