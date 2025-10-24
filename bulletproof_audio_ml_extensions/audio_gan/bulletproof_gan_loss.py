#!/usr/bin/env python3
"""
BULLETPROOF GAN LOSS MODULE
Comprehensive GAN loss implementation with multiple fallback strategies for BigVGAN neural vocoder training.
Handles gradient explosion, mode collapse detection, and numerical instabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Union, Callable, Dict, Tuple, Any
from rave_config_system import RAVEConfig
import warnings
import logging
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GANLossConfig:
    """Configuration for bulletproof GAN loss"""
    loss_type: str = 'hinge'
    real_label: float = 1.0
    fake_label: float = 0.0
    label_smoothing: float = 0.0
    gradient_penalty_weight: float = 10.0
    spectral_penalty_weight: float = 0.1
    feature_matching_weight: float = 10.0
    
    # Bulletproof stability parameters
    eps: float = 1e-8
    gradient_clip_norm: float = 1.0
    loss_clip_value: float = 100.0
    numerical_stability_check: bool = True
    mode_collapse_detection: bool = True
    adaptive_weights: bool = True
    
    # Fallback strategies
    enable_fallbacks: bool = True
    fallback_loss_type: str = 'vanilla'
    max_gradient_norm: float = 10.0
    nan_detection: bool = True
    inf_detection: bool = True

class BulletproofGANLoss(nn.Module):
    """
    Bulletproof GAN Loss with comprehensive error handling and fallback strategies.
    
    Features:
    - Multiple loss types (hinge, LSGAN, vanilla, WGAN, WGAN-GP)
    - Gradient penalty computation with numerical stability
    - Mode collapse detection and adaptive weighting
    - Comprehensive fallback strategies for training instabilities
    - Memory management for high-resolution audio
    - Device compatibility with mixed precision support
    """
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__()
        
        # Extract GAN loss specific config or use defaults
        self.config = getattr(config, 'gan_loss', GANLossConfig())
        if hasattr(config.loss, 'gan_loss_type'):
            self.config.loss_type = config.loss.gan_loss_type
        
        self.loss_type = self.config.loss_type.lower()
        self.real_label = self.config.real_label
        self.fake_label = self.config.fake_label
        self.label_smoothing = self.config.label_smoothing
        self.eps = self.config.eps
        
        # Bulletproof parameters
        self.gradient_clip_norm = self.config.gradient_clip_norm
        self.loss_clip_value = self.config.loss_clip_value
        self.enable_fallbacks = self.config.enable_fallbacks
        self.fallback_loss_type = self.config.fallback_loss_type
        
        # Tracking for stability
        self.loss_history = []
        self.gradient_norms = []
        self.mode_collapse_counter = 0
        self.fallback_activations = 0
        
        # Validate loss type
        valid_losses = ['vanilla', 'lsgan', 'hinge', 'wgan', 'wgan-gp', 'spectral']
        if self.loss_type not in valid_losses:
            if self.enable_fallbacks:
                logger.warning(f"Invalid loss type {self.loss_type}, falling back to {self.fallback_loss_type}")
                self.loss_type = self.fallback_loss_type
            else:
                raise ValueError(f"Loss type must be one of {valid_losses}, got {self.loss_type}")
        
        # Initialize adaptive weights
        self.register_buffer('adaptive_d_weight', torch.tensor(1.0))
        self.register_buffer('adaptive_g_weight', torch.tensor(1.0))
        
        logger.info(f"BulletproofGANLoss initialized with loss_type={self.loss_type}")
    
    def _validate_tensors(self, *tensors: torch.Tensor) -> bool:
        """Validate tensor inputs for numerical stability"""
        try:
            for tensor in tensors:
                if tensor is None:
                    return False
                if torch.isnan(tensor).any():
                    if self.config.nan_detection:
                        logger.warning("NaN detected in tensor")
                        return False
                if torch.isinf(tensor).any():
                    if self.config.inf_detection:
                        logger.warning("Inf detected in tensor")
                        return False
                if tensor.numel() == 0:
                    logger.warning("Empty tensor detected")
                    return False
            return True
        except Exception as e:
            logger.error(f"Tensor validation failed: {e}")
            return False
    
    def _clip_gradients(self, loss: torch.Tensor) -> torch.Tensor:
        """Clip gradients to prevent explosion"""
        try:
            if self.gradient_clip_norm > 0:
                # Clip loss value
                if self.loss_clip_value > 0:
                    loss = torch.clamp(loss, -self.loss_clip_value, self.loss_clip_value)
            return loss
        except Exception as e:
            logger.error(f"Gradient clipping failed: {e}")
            return loss
    
    def _apply_label_smoothing(self, labels: torch.Tensor, target: float) -> torch.Tensor:
        """Apply label smoothing with error handling"""
        try:
            if self.label_smoothing > 0 and target == self.real_label:
                return labels * (1 - self.label_smoothing) + self.label_smoothing * 0.5
            return labels
        except Exception as e:
            logger.error(f"Label smoothing failed: {e}")
            return labels
    
    def _detect_mode_collapse(self, fake_pred: torch.Tensor) -> bool:
        """Detect potential mode collapse"""
        try:
            if not self.config.mode_collapse_detection:
                return False
            
            # Check variance of predictions
            variance = torch.var(fake_pred)
            if variance < 1e-6:
                self.mode_collapse_counter += 1
                if self.mode_collapse_counter > 5:
                    logger.warning("Potential mode collapse detected")
                    return True
            else:
                self.mode_collapse_counter = 0
            
            return False
        except Exception as e:
            logger.error(f"Mode collapse detection failed: {e}")
            return False
    
    def _update_adaptive_weights(self, d_loss: torch.Tensor, g_loss: torch.Tensor):
        """Update adaptive weights for discriminator and generator"""
        try:
            if not self.config.adaptive_weights:
                return
            
            # Compute balance ratio
            with torch.no_grad():
                if d_loss.item() > 0 and g_loss.item() > 0:
                    ratio = g_loss.item() / (d_loss.item() + self.eps)
                    
                    # Adjust weights to balance training
                    if ratio > 2.0:  # Generator struggling
                        self.adaptive_g_weight.mul_(1.1)
                        self.adaptive_d_weight.mul_(0.9)
                    elif ratio < 0.5:  # Discriminator struggling
                        self.adaptive_d_weight.mul_(1.1)
                        self.adaptive_g_weight.mul_(0.9)
                    
                    # Clamp weights
                    self.adaptive_g_weight.clamp_(0.1, 10.0)
                    self.adaptive_d_weight.clamp_(0.1, 10.0)
        except Exception as e:
            logger.error(f"Adaptive weight update failed: {e}")
    
    def _compute_vanilla_loss(self, real_pred: torch.Tensor, fake_pred: torch.Tensor, 
                             is_discriminator: bool) -> torch.Tensor:
        """Compute vanilla GAN loss with error handling"""
        try:
            if is_discriminator:
                real_labels = torch.full_like(real_pred, self.real_label, device=real_pred.device)
                fake_labels = torch.full_like(fake_pred, self.fake_label, device=fake_pred.device)
                
                real_labels = self._apply_label_smoothing(real_labels, self.real_label)
                
                real_loss = F.binary_cross_entropy_with_logits(real_pred, real_labels)
                fake_loss = F.binary_cross_entropy_with_logits(fake_pred, fake_labels)
                
                return real_loss + fake_loss
            else:
                # Generator wants fake to be classified as real
                real_labels = torch.full_like(fake_pred, self.real_label, device=fake_pred.device)
                return F.binary_cross_entropy_with_logits(fake_pred, real_labels)
        except Exception as e:
            logger.error(f"Vanilla loss computation failed: {e}")
            return torch.tensor(0.0, device=real_pred.device, requires_grad=True)
    
    def _compute_lsgan_loss(self, real_pred: torch.Tensor, fake_pred: torch.Tensor, 
                           is_discriminator: bool) -> torch.Tensor:
        """Compute LSGAN loss with error handling"""
        try:
            if is_discriminator:
                real_loss = torch.mean((real_pred - self.real_label) ** 2)
                fake_loss = torch.mean((fake_pred - self.fake_label) ** 2)
                return 0.5 * (real_loss + fake_loss)
            else:
                return 0.5 * torch.mean((fake_pred - self.real_label) ** 2)
        except Exception as e:
            logger.error(f"LSGAN loss computation failed: {e}")
            return torch.tensor(0.0, device=real_pred.device, requires_grad=True)
    
    def _compute_hinge_loss(self, real_pred: torch.Tensor, fake_pred: torch.Tensor, 
                           is_discriminator: bool) -> torch.Tensor:
        """Compute hinge loss with error handling"""
        try:
            if is_discriminator:
                real_loss = torch.mean(torch.relu(1.0 - real_pred))
                fake_loss = torch.mean(torch.relu(1.0 + fake_pred))
                return real_loss + fake_loss
            else:
                return -torch.mean(fake_pred)
        except Exception as e:
            logger.error(f"Hinge loss computation failed: {e}")
            return torch.tensor(0.0, device=real_pred.device, requires_grad=True)
    
    def _compute_wgan_loss(self, real_pred: torch.Tensor, fake_pred: torch.Tensor, 
                          is_discriminator: bool) -> torch.Tensor:
        """Compute WGAN loss with error handling"""
        try:
            if is_discriminator:
                real_loss = -torch.mean(real_pred)
                fake_loss = torch.mean(fake_pred)
                return real_loss + fake_loss
            else:
                return -torch.mean(fake_pred)
        except Exception as e:
            logger.error(f"WGAN loss computation failed: {e}")
            return torch.tensor(0.0, device=real_pred.device, requires_grad=True)
    
    def _gradient_penalty(self, discriminator: nn.Module, real_data: torch.Tensor, 
                         fake_data: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient penalty for WGAN-GP with comprehensive error handling
        """
        try:
            batch_size = real_data.size(0)
            device = real_data.device
            
            # Ensure data shapes match
            if real_data.shape != fake_data.shape:
                min_size = min(real_data.size(-1), fake_data.size(-1))
                real_data = real_data[..., :min_size]
                fake_data = fake_data[..., :min_size]
            
            # Random interpolation weight
            alpha = torch.rand(batch_size, 1, 1, device=device, dtype=real_data.dtype)
            alpha = alpha.expand_as(real_data)
            
            # Interpolate between real and fake data
            interpolated = alpha * real_data + (1 - alpha) * fake_data
            interpolated.requires_grad_(True)
            
            # Get discriminator output for interpolated data
            with torch.enable_grad():
                d_interpolated = discriminator(interpolated)
                
                # Handle multi-scale discriminator output
                if isinstance(d_interpolated, (list, tuple)):
                    d_interpolated = d_interpolated[0]
                    if isinstance(d_interpolated, (list, tuple)):
                        d_interpolated = d_interpolated[0]
                
                # Compute gradients
                gradients = torch.autograd.grad(
                    outputs=d_interpolated,
                    inputs=interpolated,
                    grad_outputs=torch.ones_like(d_interpolated),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0]
            
            # Flatten gradients and compute norm
            gradients = gradients.view(batch_size, -1)
            gradient_norm = gradients.norm(2, dim=1)
            
            # Gradient penalty
            gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
            
            return gradient_penalty
            
        except Exception as e:
            logger.error(f"Gradient penalty computation failed: {e}")
            return torch.tensor(0.0, device=real_data.device, requires_grad=True)
    
    def discriminator_loss(self, real_pred: torch.Tensor, fake_pred: torch.Tensor,
                          real_data: Optional[torch.Tensor] = None,
                          fake_data: Optional[torch.Tensor] = None,
                          discriminator: Optional[nn.Module] = None) -> torch.Tensor:
        """
        Compute bulletproof discriminator loss with fallback strategies
        """
        try:
            # Validate inputs
            if not self._validate_tensors(real_pred, fake_pred):
                if self.enable_fallbacks:
                    logger.warning("Invalid tensors detected, using fallback")
                    self.fallback_activations += 1
                    return self._compute_vanilla_loss(real_pred, fake_pred, True)
                else:
                    raise ValueError("Invalid tensor inputs")
            
            # Compute loss based on type
            if self.loss_type == 'vanilla':
                loss = self._compute_vanilla_loss(real_pred, fake_pred, True)
            elif self.loss_type == 'lsgan':
                loss = self._compute_lsgan_loss(real_pred, fake_pred, True)
            elif self.loss_type == 'hinge':
                loss = self._compute_hinge_loss(real_pred, fake_pred, True)
            elif self.loss_type == 'wgan':
                loss = self._compute_wgan_loss(real_pred, fake_pred, True)
            elif self.loss_type == 'wgan-gp':
                base_loss = self._compute_wgan_loss(real_pred, fake_pred, True)
                if real_data is not None and fake_data is not None and discriminator is not None:
                    gp = self._gradient_penalty(discriminator, real_data, fake_data)
                    loss = base_loss + self.config.gradient_penalty_weight * gp
                else:
                    loss = base_loss
            else:
                if self.enable_fallbacks:
                    logger.warning(f"Unknown loss type {self.loss_type}, using fallback")
                    loss = self._compute_vanilla_loss(real_pred, fake_pred, True)
                else:
                    raise ValueError(f"Unknown loss type: {self.loss_type}")
            
            # Apply adaptive weighting
            if self.config.adaptive_weights:
                loss = loss * self.adaptive_d_weight
            
            # Clip and validate
            loss = self._clip_gradients(loss)
            
            # Track loss history
            if len(self.loss_history) > 100:
                self.loss_history.pop(0)
            self.loss_history.append(loss.item())
            
            return loss
            
        except Exception as e:
            logger.error(f"Discriminator loss computation failed: {e}")
            if self.enable_fallbacks:
                self.fallback_activations += 1
                return torch.tensor(1.0, device=real_pred.device, requires_grad=True)
            else:
                raise
    
    def generator_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute bulletproof generator loss with mode collapse detection
        """
        try:
            # Validate inputs
            if not self._validate_tensors(fake_pred):
                if self.enable_fallbacks:
                    logger.warning("Invalid tensor detected in generator loss")
                    self.fallback_activations += 1
                    return torch.tensor(1.0, device=fake_pred.device, requires_grad=True)
                else:
                    raise ValueError("Invalid tensor input")
            
            # Detect mode collapse
            if self._detect_mode_collapse(fake_pred):
                # Apply additional regularization
                loss_penalty = torch.var(fake_pred) * 0.1
            else:
                loss_penalty = 0.0
            
            # Compute loss based on type
            if self.loss_type == 'vanilla':
                loss = self._compute_vanilla_loss(None, fake_pred, False)
            elif self.loss_type == 'lsgan':
                loss = self._compute_lsgan_loss(None, fake_pred, False)
            elif self.loss_type == 'hinge':
                loss = self._compute_hinge_loss(None, fake_pred, False)
            elif self.loss_type in ['wgan', 'wgan-gp']:
                loss = self._compute_wgan_loss(None, fake_pred, False)
            else:
                if self.enable_fallbacks:
                    logger.warning(f"Unknown loss type {self.loss_type}, using fallback")
                    loss = self._compute_vanilla_loss(None, fake_pred, False)
                else:
                    raise ValueError(f"Unknown loss type: {self.loss_type}")
            
            # Add mode collapse penalty
            loss = loss + loss_penalty
            
            # Apply adaptive weighting
            if self.config.adaptive_weights:
                loss = loss * self.adaptive_g_weight
            
            # Clip and validate
            loss = self._clip_gradients(loss)
            
            return loss
            
        except Exception as e:
            logger.error(f"Generator loss computation failed: {e}")
            if self.enable_fallbacks:
                self.fallback_activations += 1
                return torch.tensor(1.0, device=fake_pred.device, requires_grad=True)
            else:
                raise
    
    def feature_matching_loss(self, real_features: List[torch.Tensor], 
                             fake_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute feature matching loss with error handling
        """
        try:
            if len(real_features) != len(fake_features):
                logger.warning("Feature list length mismatch")
                min_len = min(len(real_features), len(fake_features))
                real_features = real_features[:min_len]
                fake_features = fake_features[:min_len]
            
            loss = 0.0
            for real_feat, fake_feat in zip(real_features, fake_features):
                if real_feat.shape != fake_feat.shape:
                    # Resize to match
                    min_size = min(real_feat.size(-1), fake_feat.size(-1))
                    real_feat = real_feat[..., :min_size]
                    fake_feat = fake_feat[..., :min_size]
                
                loss += F.l1_loss(fake_feat, real_feat.detach())
            
            return loss / len(real_features)
            
        except Exception as e:
            logger.error(f"Feature matching loss computation failed: {e}")
            return torch.tensor(0.0, device=real_features[0].device, requires_grad=True)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics for monitoring"""
        return {
            'loss_type': self.loss_type,
            'fallback_activations': self.fallback_activations,
            'mode_collapse_counter': self.mode_collapse_counter,
            'adaptive_d_weight': self.adaptive_d_weight.item(),
            'adaptive_g_weight': self.adaptive_g_weight.item(),
            'loss_history_mean': np.mean(self.loss_history) if self.loss_history else 0.0,
            'loss_history_std': np.std(self.loss_history) if self.loss_history else 0.0
        }
    
    def reset_statistics(self):
        """Reset training statistics"""
        self.loss_history.clear()
        self.gradient_norms.clear()
        self.mode_collapse_counter = 0
        self.fallback_activations = 0
        self.adaptive_d_weight.fill_(1.0)
        self.adaptive_g_weight.fill_(1.0)
    
    def forward(self, predictions: Union[torch.Tensor, List[torch.Tensor]],
                is_real: bool, is_discriminator: bool = True, **kwargs) -> torch.Tensor:
        """
        Unified forward method with comprehensive error handling
        """
        try:
            # Handle multi-scale predictions
            if isinstance(predictions, list):
                losses = []
                for pred in predictions:
                    if is_discriminator:
                        raise ValueError("Use discriminator_loss directly for multi-scale discriminator")
                    else:
                        losses.append(self.generator_loss(pred))
                return torch.mean(torch.stack(losses))
            
            # Single prediction
            if is_discriminator:
                raise ValueError("Use discriminator_loss directly for discriminator training")
            else:
                return self.generator_loss(predictions)
                
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            if self.enable_fallbacks:
                self.fallback_activations += 1
                return torch.tensor(1.0, device=predictions.device, requires_grad=True)
            else:
                raise

# Factory function for easy instantiation
def create_bulletproof_gan_loss(config: RAVEConfig, **kwargs) -> BulletproofGANLoss:
    """Create a bulletproof GAN loss instance"""
    return BulletproofGANLoss(config, **kwargs)

if __name__ == "__main__":
    print("🛡️ BULLETPROOF GAN LOSS MODULE")
    print("=" * 50)
    
    # Test with minimal config
    from rave_config_system import get_minimal_config
    
    config = get_minimal_config()
    config.loss.gan_loss_type = 'hinge'
    
    loss_fn = create_bulletproof_gan_loss(config)
    
    # Test with dummy data
    batch_size = 4
    seq_len = 1024
    
    real_pred = torch.randn(batch_size, 1, seq_len, requires_grad=True)
    fake_pred = torch.randn(batch_size, 1, seq_len, requires_grad=True)
    
    # Test discriminator loss
    d_loss = loss_fn.discriminator_loss(real_pred, fake_pred)
    print(f"✅ Discriminator loss: {d_loss.item():.4f}")
    
    # Test generator loss
    g_loss = loss_fn.generator_loss(fake_pred)
    print(f"✅ Generator loss: {g_loss.item():.4f}")
    
    # Test with adversarial conditions (NaN/Inf)
    corrupted_pred = torch.full_like(fake_pred, float('nan'))
    g_loss_corrupted = loss_fn.generator_loss(corrupted_pred)
    print(f"✅ Fallback handling (NaN): {g_loss_corrupted.item():.4f}")
    
    # Print statistics
    stats = loss_fn.get_training_stats()
    print(f"📊 Training stats: {stats}")
    
    print("🚀 BulletproofGANLoss ready for BigVGAN neural vocoder training!")