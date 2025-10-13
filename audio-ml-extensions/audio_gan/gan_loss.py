import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union, Callable


class GANLoss(nn.Module):
    """
    Flexible GAN loss module supporting multiple loss types.
    Handles both discriminator and generator losses.
    """
    def __init__(self, 
                 loss_type: str = 'hinge',
                 real_label: float = 1.0,
                 fake_label: float = 0.0,
                 label_smoothing: float = 0.0):
        super().__init__()
        
        self.loss_type = loss_type.lower()
        self.real_label = real_label
        self.fake_label = fake_label
        self.label_smoothing = label_smoothing
        
        # Validate loss type
        valid_losses = ['vanilla', 'lsgan', 'hinge', 'wgan', 'wgan-gp']
        if self.loss_type not in valid_losses:
            raise ValueError(f"Loss type must be one of {valid_losses}, got {self.loss_type}")
            
    def _apply_label_smoothing(self, labels: torch.Tensor, target: float) -> torch.Tensor:
        """Apply label smoothing if configured."""
        if self.label_smoothing > 0 and target == self.real_label:
            return labels * (1 - self.label_smoothing) + self.label_smoothing * 0.5
        return labels
        
    def discriminator_loss(self, 
                         real_pred: torch.Tensor,
                         fake_pred: torch.Tensor,
                         real_data: Optional[torch.Tensor] = None,
                         fake_data: Optional[torch.Tensor] = None,
                         discriminator: Optional[nn.Module] = None) -> torch.Tensor:
        """
        Compute discriminator loss.
        
        Args:
            real_pred: Discriminator output for real data
            fake_pred: Discriminator output for fake data
            real_data: Real input data (needed for gradient penalty)
            fake_data: Fake input data (needed for gradient penalty)
            discriminator: Discriminator model (needed for gradient penalty)
        """
        if self.loss_type == 'vanilla':
            # Standard GAN loss
            real_labels = torch.full_like(real_pred, self.real_label)
            fake_labels = torch.full_like(fake_pred, self.fake_label)
            
            # Apply label smoothing
            real_labels = self._apply_label_smoothing(real_labels, self.real_label)
            
            real_loss = F.binary_cross_entropy_with_logits(real_pred, real_labels)
            fake_loss = F.binary_cross_entropy_with_logits(fake_pred, fake_labels)
            
            return real_loss + fake_loss
            
        elif self.loss_type == 'lsgan':
            # Least squares GAN
            real_loss = torch.mean((real_pred - self.real_label) ** 2)
            fake_loss = torch.mean((fake_pred - self.fake_label) ** 2)
            
            return 0.5 * (real_loss + fake_loss)
            
        elif self.loss_type == 'hinge':
            # Hinge loss
            real_loss = torch.mean(torch.relu(1.0 - real_pred))
            fake_loss = torch.mean(torch.relu(1.0 + fake_pred))
            
            return real_loss + fake_loss
            
        elif self.loss_type == 'wgan':
            # Wasserstein GAN
            real_loss = -torch.mean(real_pred)
            fake_loss = torch.mean(fake_pred)
            
            return real_loss + fake_loss
            
        elif self.loss_type == 'wgan-gp':
            # WGAN with gradient penalty
            real_loss = -torch.mean(real_pred)
            fake_loss = torch.mean(fake_pred)
            base_loss = real_loss + fake_loss
            
            # Compute gradient penalty
            if real_data is not None and fake_data is not None and discriminator is not None:
                gp = self._gradient_penalty(discriminator, real_data, fake_data)
                return base_loss + 10.0 * gp  # Common weight is 10
            else:
                return base_loss
                
    def generator_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute generator loss.
        
        Args:
            fake_pred: Discriminator output for fake/generated data
        """
        if self.loss_type == 'vanilla':
            # Want discriminator to think fake data is real
            real_labels = torch.full_like(fake_pred, self.real_label)
            return F.binary_cross_entropy_with_logits(fake_pred, real_labels)
            
        elif self.loss_type == 'lsgan':
            # Want fake prediction to be close to real label
            return 0.5 * torch.mean((fake_pred - self.real_label) ** 2)
            
        elif self.loss_type == 'hinge':
            # Hinge loss for generator
            return -torch.mean(fake_pred)
            
        elif self.loss_type in ['wgan', 'wgan-gp']:
            # Wasserstein loss for generator
            return -torch.mean(fake_pred)
            
    def _gradient_penalty(self, 
                         discriminator: nn.Module,
                         real_data: torch.Tensor,
                         fake_data: torch.Tensor,
                         lambda_gp: float = 10.0) -> torch.Tensor:
        """
        Compute gradient penalty for WGAN-GP.
        
        Args:
            discriminator: Discriminator model
            real_data: Real data samples
            fake_data: Generated data samples
            lambda_gp: Gradient penalty weight
        """
        batch_size = real_data.size(0)
        device = real_data.device
        
        # Random interpolation weight
        alpha = torch.rand(batch_size, 1, 1, device=device)
        alpha = alpha.expand_as(real_data)
        
        # Interpolate between real and fake data
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        # Get discriminator output for interpolated data
        d_interpolated = discriminator(interpolated)
        
        # Handle multi-scale discriminator output
        if isinstance(d_interpolated, (list, tuple)):
            d_interpolated = d_interpolated[0]  # Use first scale
            if isinstance(d_interpolated, (list, tuple)):
                d_interpolated = d_interpolated[0]  # Handle nested structure
                
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Flatten gradients
        gradients = gradients.view(batch_size, -1)
        
        # Compute penalty
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        
        return gradient_penalty
        
    def forward(self, 
                predictions: Union[torch.Tensor, List[torch.Tensor]],
                is_real: bool,
                is_discriminator: bool = True,
                **kwargs) -> torch.Tensor:
        """
        Unified forward method.
        
        Args:
            predictions: Model predictions (can be list for multi-scale)
            is_real: Whether predictions are for real data
            is_discriminator: Whether computing discriminator or generator loss
            **kwargs: Additional arguments (real_data, fake_data, discriminator for WGAN-GP)
        """
        # Handle multi-scale predictions
        if isinstance(predictions, list):
            losses = []
            for pred in predictions:
                if is_discriminator:
                    # Need both real and fake predictions for discriminator
                    raise ValueError("Use discriminator_loss directly for multi-scale discriminator")
                else:
                    losses.append(self.generator_loss(pred))
            return torch.mean(torch.stack(losses))
            
        # Single prediction
        if is_discriminator:
            raise ValueError("Use discriminator_loss directly for discriminator training")
        else:
            return self.generator_loss(predictions)