"""
WaveGANDiscriminator module for time-domain audio discrimination.

This module provides discriminators that operate directly on raw waveforms,
complementing frequency-domain discriminators. Essential for catching temporal
artifacts and phase inconsistencies that spectral discriminators miss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union


class WaveGANDiscriminator(nn.Module):
    """
    1D discriminator for raw waveform discrimination.
    
    Operates directly on time-domain audio to catch temporal artifacts
    that frequency-domain discriminators might miss. Uses 1D convolutions
    with progressively increasing receptive fields.
    
    Args:
        in_channels: Number of input channels (default: 1 for mono)
        base_channels: Base number of channels (default: 64)
        num_layers: Number of discriminator layers (default: 5)
        kernel_sizes: Kernel sizes for each layer (default: [15, 41, 41, 41, 41])
        strides: Strides for each layer (default: [1, 2, 2, 4, 4])
        use_spectral_norm: Whether to use spectral normalization
        activation: Activation function ('leaky_relu', 'relu')
        final_activation: Final layer activation (None, 'sigmoid')
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_layers: int = 5,
        kernel_sizes: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
        use_spectral_norm: bool = True,
        activation: str = 'leaky_relu',
        final_activation: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_layers = num_layers
        self.use_spectral_norm = use_spectral_norm
        
        # Default kernel sizes and strides
        if kernel_sizes is None:
            kernel_sizes = [15, 41, 41, 41, 41]
        if strides is None:
            strides = [1, 2, 2, 4, 4]
        
        # Ensure we have enough kernel sizes and strides
        while len(kernel_sizes) < num_layers:
            kernel_sizes.append(41)
        while len(strides) < num_layers:
            strides.append(4)
        
        self.kernel_sizes = kernel_sizes[:num_layers]
        self.strides = strides[:num_layers]
        
        # Build discriminator layers
        layers = []
        current_channels = in_channels
        
        for i in range(num_layers):
            out_channels = base_channels * (2 ** min(i, 4))  # Cap at 16x base
            kernel_size = self.kernel_sizes[i]
            stride = self.strides[i]
            padding = (kernel_size - 1) // 2
            
            # Create convolution layer
            conv = nn.Conv1d(
                current_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding
            )
            
            # Apply spectral normalization if requested
            if use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            
            layers.append(conv)
            
            # Add activation (except for last layer)
            if i < num_layers - 1:
                if activation == 'leaky_relu':
                    layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif activation == 'relu':
                    layers.append(nn.ReLU(inplace=True))
                else:
                    raise ValueError(f"Unknown activation: {activation}")
            
            current_channels = out_channels
        
        # Final activation
        if final_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif final_activation is not None:
            raise ValueError(f"Unknown final activation: {final_activation}")
        
        self.discriminator = nn.Sequential(*layers)
        
        # Store output channels for feature extraction
        self.output_channels = current_channels
        
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass through discriminator.
        
        Args:
            x: Input waveform (batch, channels, time)
            return_features: Whether to return intermediate features
            
        Returns:
            Discriminator output, optionally with intermediate features
        """
        if return_features:
            features = []
            
            for layer in self.discriminator:
                x = layer(x)
                if isinstance(layer, nn.Conv1d):
                    features.append(x)
            
            return x, features
        else:
            return self.discriminator(x)
    
    def get_receptive_field(self) -> int:
        """Calculate the receptive field of the discriminator."""
        receptive_field = 1
        stride_product = 1
        
        for i in range(self.num_layers):
            kernel_size = self.kernel_sizes[i]
            stride = self.strides[i]
            
            receptive_field += (kernel_size - 1) * stride_product
            stride_product *= stride
        
        return receptive_field


class MultiScaleWaveDiscriminator(nn.Module):
    """
    Multi-scale waveform discriminator operating at different resolutions.
    
    Uses multiple WaveGAN discriminators at different time scales to capture
    both local and global temporal patterns.
    
    Args:
        num_scales: Number of different time scales (default: 3)
        scale_factors: Downsampling factors for each scale (default: [1, 2, 4])
        base_channels: Base channels for each discriminator
        use_spectral_norm: Whether to use spectral normalization
        **kwargs: Additional arguments passed to each discriminator
    """
    
    def __init__(
        self,
        num_scales: int = 3,
        scale_factors: Optional[List[int]] = None,
        base_channels: int = 64,
        use_spectral_norm: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.num_scales = num_scales
        
        if scale_factors is None:
            scale_factors = [1, 2, 4]
        
        # Ensure we have enough scale factors
        while len(scale_factors) < num_scales:
            scale_factors.append(scale_factors[-1] * 2)
        
        self.scale_factors = scale_factors[:num_scales]
        
        # Create discriminators for each scale
        self.discriminators = nn.ModuleList()
        
        for i, scale_factor in enumerate(self.scale_factors):
            # Adjust parameters based on scale
            scale_base_channels = base_channels // (2 ** min(i, 2))
            
            discriminator = WaveGANDiscriminator(
                base_channels=scale_base_channels,
                use_spectral_norm=use_spectral_norm,
                **kwargs
            )
            
            self.discriminators.append(discriminator)
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[List[torch.Tensor]]]]:
        """
        Forward pass through all discriminators.
        
        Args:
            x: Input waveform (batch, channels, time)
            return_features: Whether to return intermediate features
            
        Returns:
            List of discriminator outputs, optionally with features
        """
        outputs = []
        all_features = [] if return_features else None
        
        for i, (discriminator, scale_factor) in enumerate(zip(self.discriminators, self.scale_factors)):
            # Downsample input for this scale
            if scale_factor > 1:
                x_scaled = F.avg_pool1d(x, kernel_size=scale_factor, stride=scale_factor)
            else:
                x_scaled = x
            
            # Apply discriminator
            if return_features:
                output, features = discriminator(x_scaled, return_features=True)
                outputs.append(output)
                all_features.append(features)
            else:
                output = discriminator(x_scaled, return_features=False)
                outputs.append(output)
        
        if return_features:
            return outputs, all_features
        else:
            return outputs
    
    def compute_loss(
        self,
        real_outputs: List[torch.Tensor],
        fake_outputs: List[torch.Tensor],
        loss_type: str = 'hinge'
    ) -> torch.Tensor:
        """
        Compute discriminator loss across all scales.
        
        Args:
            real_outputs: Outputs for real audio
            fake_outputs: Outputs for fake audio
            loss_type: Type of loss ('hinge', 'vanilla', 'lsgan')
            
        Returns:
            Combined discriminator loss
        """
        total_loss = 0.0
        
        for real_out, fake_out in zip(real_outputs, fake_outputs):
            if loss_type == 'hinge':
                real_loss = torch.mean(F.relu(1.0 - real_out))
                fake_loss = torch.mean(F.relu(1.0 + fake_out))
            elif loss_type == 'vanilla':
                real_loss = F.binary_cross_entropy_with_logits(
                    real_out, torch.ones_like(real_out)
                )
                fake_loss = F.binary_cross_entropy_with_logits(
                    fake_out, torch.zeros_like(fake_out)
                )
            elif loss_type == 'lsgan':
                real_loss = F.mse_loss(real_out, torch.ones_like(real_out))
                fake_loss = F.mse_loss(fake_out, torch.zeros_like(fake_out))
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
            
            total_loss += (real_loss + fake_loss)
        
        return total_loss / len(real_outputs)
    
    def compute_generator_loss(
        self,
        fake_outputs: List[torch.Tensor],
        loss_type: str = 'hinge'
    ) -> torch.Tensor:
        """
        Compute generator loss from discriminator outputs.
        
        Args:
            fake_outputs: Discriminator outputs for generated audio
            loss_type: Type of loss ('hinge', 'vanilla', 'lsgan')
            
        Returns:
            Generator loss
        """
        total_loss = 0.0
        
        for fake_out in fake_outputs:
            if loss_type == 'hinge':
                loss = -torch.mean(fake_out)
            elif loss_type == 'vanilla':
                loss = F.binary_cross_entropy_with_logits(
                    fake_out, torch.ones_like(fake_out)
                )
            elif loss_type == 'lsgan':
                loss = F.mse_loss(fake_out, torch.ones_like(fake_out))
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
            
            total_loss += loss
        
        return total_loss / len(fake_outputs)


class FeatureMatchingLoss(nn.Module):
    """
    Feature matching loss for waveform discriminators.
    
    Computes L1 loss between intermediate discriminator features
    to encourage generator to match real data statistics.
    
    Args:
        normalize_features: Whether to normalize features before comparison
        **kwargs: Additional arguments (for compatibility)
    """
    
    def __init__(
        self,
        normalize_features: bool = True,
        **kwargs
    ):
        super().__init__()
        
        # Accept kwargs for flexibility
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.normalize_features = normalize_features
    
    def forward(
        self,
        real_features: List[List[torch.Tensor]],
        fake_features: List[List[torch.Tensor]]
    ) -> torch.Tensor:
        """
        Compute feature matching loss.
        
        Args:
            real_features: Features from real audio [scale][layer]
            fake_features: Features from fake audio [scale][layer]
            
        Returns:
            Feature matching loss
        """
        # Initialize with a tensor that has gradients
        device = real_features[0][0].device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_count = 0
        
        # Iterate over scales
        for real_scale_features, fake_scale_features in zip(real_features, fake_features):
            # Iterate over layers within each scale
            for real_feat, fake_feat in zip(real_scale_features, fake_scale_features):
                if self.normalize_features:
                    # Normalize features to unit norm
                    real_feat = F.normalize(real_feat, p=2, dim=1)
                    fake_feat = F.normalize(fake_feat, p=2, dim=1)
                
                # Compute L1 loss between features
                loss = F.l1_loss(real_feat, fake_feat)
                total_loss = total_loss + loss
                total_count += 1
        
        if total_count > 0:
            return total_loss / total_count
        else:
            return total_loss


class ConditionalWaveDiscriminator(WaveGANDiscriminator):
    """
    Conditional waveform discriminator with conditioning support.
    
    Extends WaveGANDiscriminator to accept conditioning information
    such as speaker embeddings, class labels, or other control signals.
    
    Args:
        conditioning_dim: Dimension of conditioning vector
        conditioning_method: How to apply conditioning ('concat', 'film', 'embed')
        embed_dim: Embedding dimension for categorical conditioning
        **kwargs: Arguments passed to WaveGANDiscriminator
    """
    
    def __init__(
        self,
        conditioning_dim: int,
        conditioning_method: str = 'concat',
        embed_dim: Optional[int] = None,
        **kwargs
    ):
        # Initialize base discriminator
        super().__init__(**kwargs)
        
        self.conditioning_dim = conditioning_dim
        self.conditioning_method = conditioning_method
        
        if conditioning_method == 'concat':
            # Concatenate conditioning to input
            # Modify first layer to accept additional channels
            first_conv = self.discriminator[0]
            new_in_channels = first_conv.in_channels + conditioning_dim
            
            # Create new first layer
            new_first_conv = nn.Conv1d(
                new_in_channels,
                first_conv.out_channels,
                first_conv.kernel_size[0],
                stride=first_conv.stride[0],
                padding=first_conv.padding[0]
            )
            
            # Apply spectral norm if original layer had it
            if self.use_spectral_norm:
                new_first_conv = nn.utils.spectral_norm(new_first_conv)
            
            # Replace first layer
            self.discriminator[0] = new_first_conv
            
        elif conditioning_method == 'film':
            # Feature-wise Linear Modulation
            self.film_layers = nn.ModuleList()
            
            for layer in self.discriminator:
                if isinstance(layer, nn.Conv1d):
                    film = nn.Linear(conditioning_dim, layer.out_channels * 2)  # Scale and bias
                    self.film_layers.append(film)
            
        elif conditioning_method == 'embed':
            # Embedding-based conditioning
            if embed_dim is None:
                embed_dim = conditioning_dim
            
            self.embedding = nn.Linear(conditioning_dim, embed_dim)
            self.embed_dim = embed_dim
            
            # Modify first layer to accept embedded conditioning
            first_conv = self.discriminator[0]
            new_in_channels = first_conv.in_channels + embed_dim
            
            new_first_conv = nn.Conv1d(
                new_in_channels,
                first_conv.out_channels,
                first_conv.kernel_size[0],
                stride=first_conv.stride[0],
                padding=first_conv.padding[0]
            )
            
            # Apply spectral norm if original layer had it
            if self.use_spectral_norm:
                new_first_conv = nn.utils.spectral_norm(new_first_conv)
            
            # Replace first layer
            self.discriminator[0] = new_first_conv
            
        else:
            raise ValueError(f"Unknown conditioning method: {conditioning_method}")
    
    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass with conditioning.
        
        Args:
            x: Input waveform (batch, channels, time)
            conditioning: Conditioning vector (batch, conditioning_dim)
            return_features: Whether to return intermediate features
            
        Returns:
            Discriminator output, optionally with features
        """
        if self.conditioning_method == 'concat':
            # Expand conditioning to match time dimension
            time_steps = x.shape[-1]
            cond_expanded = conditioning.unsqueeze(-1).expand(-1, -1, time_steps)
            
            # Concatenate to input
            x = torch.cat([x, cond_expanded], dim=1)
            
            # Use parent forward method
            return super().forward(x, return_features)
            
        elif self.conditioning_method == 'film':
            # Apply FiLM modulation
            features = [] if return_features else None
            film_idx = 0
            
            for layer in self.discriminator:
                x = layer(x)
                
                # Apply FiLM after conv layers
                if isinstance(layer, nn.Conv1d):
                    film_params = self.film_layers[film_idx](conditioning)
                    scale, bias = torch.chunk(film_params, 2, dim=1)
                    
                    # Apply modulation
                    x = x * scale.unsqueeze(-1) + bias.unsqueeze(-1)
                    
                    if return_features:
                        features.append(x)
                    
                    film_idx += 1
            
            if return_features:
                return x, features
            else:
                return x
            
        elif self.conditioning_method == 'embed':
            # Embedding-based conditioning
            embedded_cond = self.embedding(conditioning)
            time_steps = x.shape[-1]
            cond_expanded = embedded_cond.unsqueeze(-1).expand(-1, -1, time_steps)
            
            # Concatenate to input
            x = torch.cat([x, cond_expanded], dim=1)
            
            # Use parent forward method (which now handles the modified first layer)
            return super().forward(x, return_features)
        
        else:
            raise ValueError(f"Unknown conditioning method: {self.conditioning_method}")