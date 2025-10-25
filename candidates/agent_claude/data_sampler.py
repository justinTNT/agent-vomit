"""
DataSampler - Advanced sampling strategies for imbalanced data.

This module supports oversampling, undersampling, SMOTE-like methods,
handles class imbalance, and tracks sampling statistics.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Literal
import warnings
import numpy as np


class DataSampler(nn.Module):
    """
    Advanced sampling strategies for handling imbalanced data.
    
    Args:
        strategy: Sampling strategy ('oversample', 'undersample', 'smote', 'balanced')
        target_ratio: Target class ratio (1.0 for balanced)
        k_neighbors: Number of neighbors for SMOTE
        seed: Random seed for reproducibility
        min_samples_per_class: Minimum samples to keep per class
        **kwargs: Additional arguments
    """
    
    def __init__(
        self,
        strategy: Literal['oversample', 'undersample', 'smote', 'balanced'] = 'balanced',
        target_ratio: float = 1.0,
        k_neighbors: int = 5,
        seed: Optional[int] = None,
        min_samples_per_class: int = 1,
        **kwargs
    ):
        super().__init__()
        
        if strategy not in ['oversample', 'undersample', 'smote', 'balanced']:
            raise ValueError(f"Invalid strategy: {strategy}")
        if target_ratio <= 0:
            raise ValueError(f"target_ratio must be positive, got {target_ratio}")
        if k_neighbors <= 0:
            raise ValueError(f"k_neighbors must be positive, got {k_neighbors}")
        if min_samples_per_class <= 0:
            raise ValueError(f"min_samples_per_class must be positive, got {min_samples_per_class}")
        
        self.strategy = strategy
        self.target_ratio = target_ratio
        self.k_neighbors = k_neighbors
        self.seed = seed
        self.min_samples_per_class = min_samples_per_class
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Statistics tracking
        self.register_buffer('total_samples_processed', torch.tensor(0, dtype=torch.long))
        self.register_buffer('total_samples_generated', torch.tensor(0, dtype=torch.long))
        self.class_counts = {}
        self.sampling_history = []
        
        # Handle unused kwargs
        if kwargs:
            warnings.warn(f"Unused arguments: {list(kwargs.keys())}")
    
    def _compute_class_statistics(
        self, 
        labels: torch.Tensor
    ) -> Tuple[Dict[int, int], int, int]:
        """Compute class counts and identify minority/majority classes."""
        unique_labels, counts = torch.unique(labels, return_counts=True)
        class_counts = {int(label): int(count) for label, count in zip(unique_labels, counts)}
        
        if len(class_counts) < 2:
            raise ValueError("Need at least 2 classes for sampling")
        
        min_class = min(class_counts, key=class_counts.get)
        max_class = max(class_counts, key=class_counts.get)
        
        return class_counts, min_class, max_class
    
    def _oversample(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        class_counts: Dict[int, int],
        min_class: int,
        max_class: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Oversample minority classes."""
        target_count = int(class_counts[max_class] * self.target_ratio)
        
        new_data = [data]
        new_labels = [labels]
        
        for class_label, count in class_counts.items():
            if count < target_count:
                # Find indices of this class
                class_mask = labels == class_label
                class_indices = torch.where(class_mask)[0]
                
                # Compute how many samples to generate
                n_samples = target_count - count
                
                # Randomly sample with replacement
                sampled_indices = class_indices[torch.randint(
                    len(class_indices), (n_samples,), device=labels.device
                )]
                
                new_data.append(data[sampled_indices])
                new_labels.append(torch.full((n_samples,), class_label, device=labels.device))
                
                self.total_samples_generated += n_samples
        
        return torch.cat(new_data), torch.cat(new_labels)
    
    def _undersample(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        class_counts: Dict[int, int],
        min_class: int,
        max_class: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Undersample majority classes."""
        target_count = max(
            int(class_counts[min_class] / self.target_ratio),
            self.min_samples_per_class
        )
        
        keep_indices = []
        
        for class_label, count in class_counts.items():
            class_mask = labels == class_label
            class_indices = torch.where(class_mask)[0]
            
            if count > target_count:
                # Randomly sample without replacement
                perm = torch.randperm(len(class_indices), device=labels.device)
                selected_indices = class_indices[perm[:target_count]]
            else:
                selected_indices = class_indices
            
            keep_indices.append(selected_indices)
        
        keep_indices = torch.cat(keep_indices)
        return data[keep_indices], labels[keep_indices]
    
    def _smote(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        class_counts: Dict[int, int],
        min_class: int,
        max_class: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """SMOTE-like synthetic sample generation."""
        target_count = int(class_counts[max_class] * self.target_ratio)
        
        new_data = [data]
        new_labels = [labels]
        
        for class_label, count in class_counts.items():
            if count < target_count and count >= self.k_neighbors:
                # Find indices of this class
                class_mask = labels == class_label
                class_indices = torch.where(class_mask)[0]
                class_data = data[class_indices]
                
                # Number of synthetic samples to generate
                n_synthetic = target_count - count
                
                # Generate synthetic samples
                synthetic_data = []
                for _ in range(n_synthetic):
                    # Random sample from class
                    idx = torch.randint(len(class_data), (1,)).item()
                    sample = class_data[idx]
                    
                    # Find k nearest neighbors within class
                    distances = torch.cdist(sample.unsqueeze(0), class_data)
                    _, nearest_indices = distances.topk(
                        min(self.k_neighbors + 1, len(class_data)), 
                        largest=False
                    )
                    nearest_indices = nearest_indices[0, 1:]  # Exclude self
                    
                    if len(nearest_indices) > 0:
                        # Random neighbor
                        neighbor_idx = nearest_indices[torch.randint(len(nearest_indices), (1,))].item()
                        neighbor = class_data[neighbor_idx]
                        
                        # Generate synthetic sample
                        alpha = torch.rand(1, device=data.device)
                        synthetic = sample + alpha * (neighbor - sample)
                        synthetic_data.append(synthetic)
                
                if synthetic_data:
                    synthetic_tensor = torch.stack(synthetic_data)
                    new_data.append(synthetic_tensor)
                    new_labels.append(torch.full((len(synthetic_data),), class_label, device=labels.device))
                    self.total_samples_generated += len(synthetic_data)
        
        return torch.cat(new_data), torch.cat(new_labels)
    
    def forward(
        self, 
        data: torch.Tensor, 
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Apply sampling strategy to the data.
        
        Args:
            data: Input data tensor of shape (n_samples, ...)
            labels: Class labels tensor of shape (n_samples,)
            
        Returns:
            Dictionary containing:
                - data: Sampled data
                - labels: Sampled labels
                - original_counts: Original class distribution
                - sampled_counts: New class distribution
                - sampling_ratio: Ratio of sampled to original size
        """
        if not isinstance(data, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise TypeError("data and labels must be torch.Tensors")
        
        if data.size(0) != labels.size(0):
            raise ValueError("data and labels must have same number of samples")
        
        device = data.device
        labels = labels.to(device)
        
        # Compute class statistics
        class_counts, min_class, max_class = self._compute_class_statistics(labels)
        original_size = data.size(0)
        
        # Update tracking
        self.total_samples_processed += original_size
        self.class_counts = class_counts
        
        # Apply sampling strategy
        if self.strategy == 'oversample':
            sampled_data, sampled_labels = self._oversample(
                data, labels, class_counts, min_class, max_class
            )
        elif self.strategy == 'undersample':
            sampled_data, sampled_labels = self._undersample(
                data, labels, class_counts, min_class, max_class
            )
        elif self.strategy == 'smote':
            sampled_data, sampled_labels = self._smote(
                data, labels, class_counts, min_class, max_class
            )
        elif self.strategy == 'balanced':
            # Balanced combines oversampling minority and undersampling majority
            # First oversample to median
            median_count = int(torch.tensor(list(class_counts.values())).median().item())
            temp_ratio = self.target_ratio
            self.target_ratio = 1.0
            
            # Oversample minorities
            temp_data, temp_labels = self._oversample(
                data, labels, class_counts, min_class, max_class
            )
            
            # Recompute stats
            temp_counts, _, _ = self._compute_class_statistics(temp_labels)
            
            # Undersample majorities
            sampled_data, sampled_labels = self._undersample(
                temp_data, temp_labels, temp_counts, min_class, max_class
            )
            
            self.target_ratio = temp_ratio
        
        # Compute new class distribution
        sampled_counts, _, _ = self._compute_class_statistics(sampled_labels)
        
        # Track sampling history
        self.sampling_history.append({
            'original_counts': class_counts,
            'sampled_counts': sampled_counts,
            'strategy': self.strategy,
            'samples_generated': self.total_samples_generated.item()
        })
        
        return {
            'data': sampled_data,
            'labels': sampled_labels,
            'original_counts': class_counts,
            'sampled_counts': sampled_counts,
            'sampling_ratio': torch.tensor(
                sampled_data.size(0) / original_size, 
                device=device
            )
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get sampling statistics."""
        return {
            'total_samples_processed': self.total_samples_processed.item(),
            'total_samples_generated': self.total_samples_generated.item(),
            'current_class_distribution': self.class_counts,
            'strategy': self.strategy,
            'target_ratio': self.target_ratio,
            'sampling_history_length': len(self.sampling_history)
        }