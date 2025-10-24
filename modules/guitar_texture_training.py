#!/usr/bin/env python3
"""
Guitar Texture Training Pipeline

Complete training infrastructure for the superior guitar texture architecture.
Includes data loading, training loops, validation, and model checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
import json
import logging
from tqdm import tqdm
import wandb
from collections import defaultdict
import random

from .guitar_texture_architecture import (
    GuitarTextureModel, GuitarTextureConfig, create_guitar_texture_model
)
from .guitar_texture_exploration import GuitarTextureExplorer, TextureAnalyzer


@dataclass
class TrainingConfig:
    """Configuration for training the guitar texture model"""
    
    # Data parameters
    audio_dir: str = "data/guitar_audio"
    mel_dir: str = "data/mel_spectrograms"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Training parameters
    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Optimization
    optimizer: str = "adamw"  # "adamw", "adam", "sgd"
    scheduler: str = "cosine"  # "cosine", "plateau", "step"
    warmup_epochs: int = 5
    
    # Loss weights
    reconstruction_weight: float = 1.0
    texture_kl_weight: float = 0.1
    content_kl_weight: float = 0.9
    quantizer_weight: float = 0.25
    perceptual_weight: float = 0.1
    
    # Training strategies
    use_texture_augmentation: bool = True
    use_adversarial_training: bool = False
    progressive_training: bool = True
    gradient_clipping: float = 1.0
    
    # Validation and logging
    val_every_n_epochs: int = 5
    save_every_n_epochs: int = 10
    log_every_n_steps: int = 100
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints/guitar_texture"
    resume_from_checkpoint: Optional[str] = None
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    mixed_precision: bool = True
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = "guitar-texture-modeling"
    experiment_name: Optional[str] = None


class GuitarTextureDataset(Dataset):
    """Dataset for guitar texture modeling"""
    
    def __init__(self, 
                 audio_files: List[Path],
                 config: GuitarTextureConfig,
                 training_config: TrainingConfig,
                 augment: bool = False):
        
        self.audio_files = audio_files
        self.config = config
        self.training_config = training_config
        self.augment = augment
        
        # Precompute mel-spectrograms for efficiency
        self.mel_cache = {}
        self._precompute_mels()
        
    def _precompute_mels(self):
        """Precompute mel-spectrograms for all audio files"""
        
        print(f"Precomputing mel-spectrograms for {len(self.audio_files)} files...")
        
        for audio_file in tqdm(self.audio_files):
            try:
                # Load audio
                audio, sr = librosa.load(audio_file, sr=self.config.sample_rate)
                
                # Convert to mel-spectrogram
                mel = librosa.feature.melspectrogram(
                    y=audio,
                    sr=sr,
                    n_fft=self.config.n_fft,
                    hop_length=self.config.hop_length,
                    n_mels=self.config.n_mels
                )
                
                # Convert to log scale
                mel_db = librosa.power_to_db(mel, ref=np.max)
                
                # Normalize to [-1, 1]
                mel_normalized = 2 * (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min()) - 1
                
                self.mel_cache[str(audio_file)] = mel_normalized
                
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                continue
    
    def __len__(self):
        return len(self.mel_cache)
    
    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        mel = self.mel_cache[str(audio_file)]
        
        # Apply augmentation if enabled
        if self.augment and self.training_config.use_texture_augmentation:
            mel = self._augment_mel(mel)
        
        # Convert to tensor
        mel_tensor = torch.FloatTensor(mel)
        
        # Ensure consistent length by cropping or padding
        target_length = 128  # frames
        if mel_tensor.shape[1] > target_length:
            # Random crop
            start_idx = random.randint(0, mel_tensor.shape[1] - target_length)
            mel_tensor = mel_tensor[:, start_idx:start_idx + target_length]
        elif mel_tensor.shape[1] < target_length:
            # Pad with zeros
            pad_length = target_length - mel_tensor.shape[1]
            mel_tensor = F.pad(mel_tensor, (0, pad_length))
        
        return {
            'mel': mel_tensor,
            'file_path': str(audio_file)
        }
    
    def _augment_mel(self, mel: np.ndarray) -> np.ndarray:
        """Apply data augmentation to mel-spectrogram"""
        
        augmented = mel.copy()
        
        # Time stretching (simple frame repeat/skip)
        if random.random() < 0.3:
            stretch_factor = random.uniform(0.9, 1.1)
            if stretch_factor > 1.0:
                # Repeat frames
                repeat_indices = np.random.choice(
                    augmented.shape[1], 
                    int(augmented.shape[1] * (stretch_factor - 1.0))
                )
                augmented = np.insert(augmented, repeat_indices, augmented[:, repeat_indices], axis=1)
            else:
                # Skip frames
                keep_ratio = stretch_factor
                keep_indices = np.random.choice(
                    augmented.shape[1],
                    int(augmented.shape[1] * keep_ratio),
                    replace=False
                )
                keep_indices = np.sort(keep_indices)
                augmented = augmented[:, keep_indices]
        
        # Frequency masking
        if random.random() < 0.3:
            freq_mask_size = random.randint(1, 8)
            freq_mask_start = random.randint(0, augmented.shape[0] - freq_mask_size)
            augmented[freq_mask_start:freq_mask_start + freq_mask_size, :] *= 0.1
        
        # Time masking
        if random.random() < 0.3:
            time_mask_size = random.randint(1, 16)
            if augmented.shape[1] > time_mask_size:
                time_mask_start = random.randint(0, augmented.shape[1] - time_mask_size)
                augmented[:, time_mask_start:time_mask_start + time_mask_size] *= 0.1
        
        # Noise injection
        if random.random() < 0.2:
            noise_level = random.uniform(0.01, 0.05)
            noise = np.random.normal(0, noise_level, augmented.shape)
            augmented = augmented + noise
        
        return augmented


def create_data_loaders(training_config: TrainingConfig, 
                       model_config: GuitarTextureConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders"""
    
    # Find all audio files
    audio_dir = Path(training_config.audio_dir)
    audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3"))
    
    if not audio_files:
        raise ValueError(f"No audio files found in {audio_dir}")
    
    # Split data
    random.shuffle(audio_files)
    n_files = len(audio_files)
    n_train = int(n_files * training_config.train_split)
    n_val = int(n_files * training_config.val_split)
    
    train_files = audio_files[:n_train]
    val_files = audio_files[n_train:n_train + n_val]
    test_files = audio_files[n_train + n_val:]
    
    print(f"Data split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    # Create datasets
    train_dataset = GuitarTextureDataset(train_files, model_config, training_config, augment=True)
    val_dataset = GuitarTextureDataset(val_files, model_config, training_config, augment=False)
    test_dataset = GuitarTextureDataset(test_files, model_config, training_config, augment=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


class GuitarTextureTrainer:
    """Complete training pipeline for guitar texture model"""
    
    def __init__(self, 
                 model_config: GuitarTextureConfig,
                 training_config: TrainingConfig):
        
        self.model_config = model_config
        self.training_config = training_config
        
        # Setup device
        self.device = torch.device(training_config.device)
        print(f"Using device: {self.device}")
        
        # Create model
        self.model = create_guitar_texture_model(model_config)
        self.model.to(self.device)
        
        # Setup training components
        self._setup_optimizer()
        self._setup_scheduler()
        
        # Mixed precision training
        if training_config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Logging
        self._setup_logging()
        
        # Checkpoint directory
        self.checkpoint_dir = Path(training_config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Load checkpoint if specified
        if training_config.resume_from_checkpoint:
            self._load_checkpoint(training_config.resume_from_checkpoint)
    
    def _setup_optimizer(self):
        """Setup optimizer"""
        
        if self.training_config.optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay
            )
        elif self.training_config.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay,
                momentum=0.9
            )
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        
        if self.training_config.scheduler.lower() == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.training_config.num_epochs
            )
        elif self.training_config.scheduler.lower() == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10
            )
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
    
    def _setup_logging(self):
        """Setup logging"""
        
        # Setup basic logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Setup wandb if enabled
        if self.training_config.use_wandb:
            experiment_name = (
                self.training_config.experiment_name or 
                f"guitar-texture-{self.training_config.optimizer}-lr{self.training_config.learning_rate}"
            )
            
            wandb.init(
                project=self.training_config.wandb_project,
                name=experiment_name,
                config={
                    **asdict(self.model_config),
                    **asdict(self.training_config)
                }
            )
            wandb.watch(self.model, log='all', log_freq=1000)
    
    def train(self):
        """Main training loop"""
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            self.training_config, self.model_config
        )
        
        print(f"Starting training for {self.training_config.num_epochs} epochs")
        
        for epoch in range(self.current_epoch, self.training_config.num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self._train_epoch(train_loader)
            
            # Validation phase
            if epoch % self.training_config.val_every_n_epochs == 0:
                val_metrics = self._validate_epoch(val_loader)
                
                # Log validation metrics
                self._log_metrics(val_metrics, prefix="val", epoch=epoch)
                
                # Check for best model
                if val_metrics['total_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['total_loss']
                    self._save_checkpoint(f"best_model.pt", is_best=True)
                
                # Update scheduler if plateau-based
                if self.training_config.scheduler.lower() == "plateau":
                    self.scheduler.step(val_metrics['total_loss'])
            
            # Log training metrics
            self._log_metrics(train_metrics, prefix="train", epoch=epoch)
            
            # Update scheduler if not plateau-based
            if self.training_config.scheduler.lower() != "plateau":
                self.scheduler.step()
            
            # Save checkpoint
            if epoch % self.training_config.save_every_n_epochs == 0:
                self._save_checkpoint(f"epoch_{epoch}.pt")
        
        # Final evaluation on test set
        test_metrics = self._validate_epoch(test_loader)
        self._log_metrics(test_metrics, prefix="test", epoch=self.current_epoch)
        
        print("Training completed!")
        
        if self.training_config.use_wandb:
            wandb.finish()
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        
        self.model.train()
        epoch_metrics = defaultdict(list)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            mel = batch['mel'].to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    model_output = self.model(mel.unsqueeze(1))  # Add channel dim
                    loss_dict = self.model.compute_loss(model_output, mel)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss_dict['total_loss']).backward()
                
                # Gradient clipping
                if self.training_config.gradient_clipping > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.training_config.gradient_clipping
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                model_output = self.model(mel.unsqueeze(1))
                loss_dict = self.model.compute_loss(model_output, mel)
                
                loss_dict['total_loss'].backward()
                
                if self.training_config.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config.gradient_clipping
                    )
                
                self.optimizer.step()
            
            # Collect metrics
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    epoch_metrics[key].append(value.item())
                else:
                    epoch_metrics[key].append(value)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss_dict['total_loss'].item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Log step metrics
            if self.global_step % self.training_config.log_every_n_steps == 0:
                self._log_step_metrics(loss_dict, self.global_step)
            
            self.global_step += 1
        
        # Average metrics over epoch
        averaged_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        
        return averaged_metrics
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        
        self.model.eval()
        epoch_metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                mel = batch['mel'].to(self.device, non_blocking=True)
                
                # Forward pass
                model_output = self.model(mel.unsqueeze(1))
                loss_dict = self.model.compute_loss(model_output, mel)
                
                # Collect metrics
                for key, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        epoch_metrics[key].append(value.item())
                    else:
                        epoch_metrics[key].append(value)
        
        # Average metrics
        averaged_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        
        return averaged_metrics
    
    def _log_metrics(self, metrics: Dict[str, float], prefix: str, epoch: int):
        """Log metrics to console and wandb"""
        
        # Console logging
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch} {prefix}: {metric_str}")
        
        # Wandb logging
        if self.training_config.use_wandb:
            wandb_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
            wandb_metrics['epoch'] = epoch
            wandb.log(wandb_metrics)
    
    def _log_step_metrics(self, metrics: Dict[str, torch.Tensor], step: int):
        """Log step metrics to wandb"""
        
        if self.training_config.use_wandb:
            wandb_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    wandb_metrics[f"step/{key}"] = value.item()
                else:
                    wandb_metrics[f"step/{key}"] = value
            
            wandb_metrics['global_step'] = step
            wandb.log(wandb_metrics)
    
    def _save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'model_config': asdict(self.model_config),
            'training_config': asdict(self.training_config)
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            self.logger.info(f"Saved best model to {checkpoint_path}")
        else:
            self.logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {self.current_epoch}")


def train_guitar_texture_model(model_config: Optional[GuitarTextureConfig] = None,
                              training_config: Optional[TrainingConfig] = None):
    """Main training function"""
    
    if model_config is None:
        model_config = GuitarTextureConfig()
    
    if training_config is None:
        training_config = TrainingConfig()
    
    # Create trainer and start training
    trainer = GuitarTextureTrainer(model_config, training_config)
    trainer.train()


if __name__ == "__main__":
    # Example training configuration
    model_config = GuitarTextureConfig(
        sample_rate=22050,
        n_fft=2048,
        hop_length=512,
        n_mels=80,
        encoder_hidden_dim=512,
        texture_dim=128,
        content_dim=256,
        num_quantizer_levels=3,
        codebook_sizes=[64, 256, 1024]
    )
    
    training_config = TrainingConfig(
        audio_dir="data/guitar_audio",
        batch_size=16,
        num_epochs=100,
        learning_rate=1e-4,
        use_wandb=True,
        experiment_name="guitar-texture-v1"
    )
    
    train_guitar_texture_model(model_config, training_config)


# Export main components
__all__ = [
    'GuitarTextureTrainer',
    'GuitarTextureDataset', 
    'TrainingConfig',
    'train_guitar_texture_model',
    'create_data_loaders'
]