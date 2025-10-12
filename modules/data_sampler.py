import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import numpy as np
from collections import Counter, defaultdict
import random
from dataclasses import dataclass


@dataclass
class SamplingStats:
    """Statistics for sampling operations."""
    total_samples: int
    class_counts: Dict[Any, int]
    sampling_counts: Dict[Any, int]
    effective_ratio: Dict[Any, float]


class DataSampler(nn.Module):
    """
    Advanced sampling strategies for handling imbalanced data and
    creating representative batches.
    """
    
    def __init__(
        self,
        strategy: str = "uniform",  # uniform, stratified, weighted, balanced, focal, adaptive
        batch_size: int = 32,
        replacement: bool = True,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: Optional[int] = None
    ):
        super().__init__()
        self.strategy = strategy
        self.batch_size = batch_size
        self.replacement = replacement
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
        # State for various strategies
        self.class_weights = {}
        self.sample_weights = None
        self.class_indices = defaultdict(list)
        self.sampling_history = defaultdict(int)
        self.epoch = 0
        
        # Adaptive sampling parameters
        self.loss_history = {}
        self.difficulty_scores = {}
        
    def forward(self, 
                data: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from data according to strategy."""
        n_samples = len(data)
        
        if self.strategy == "uniform":
            indices = self._uniform_sample(n_samples)
        elif self.strategy == "stratified":
            indices = self._stratified_sample(n_samples, labels)
        elif self.strategy == "weighted":
            indices = self._weighted_sample(n_samples, weights)
        elif self.strategy == "balanced":
            indices = self._balanced_sample(n_samples, labels)
        elif self.strategy == "focal":
            indices = self._focal_sample(n_samples, labels)
        elif self.strategy == "adaptive":
            indices = self._adaptive_sample(n_samples, labels)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.strategy}")
            
        # Apply sampling
        sampled_data = data[indices]
        sampled_labels = labels[indices] if labels is not None else None
        
        return sampled_data, sampled_labels
    
    def _uniform_sample(self, n_samples: int) -> torch.Tensor:
        """Uniform random sampling."""
        if self.replacement:
            indices = torch.randint(0, n_samples, (self.batch_size,))
        else:
            perm = torch.randperm(n_samples)
            indices = perm[:self.batch_size]
            
        return indices
    
    def _stratified_sample(self, n_samples: int, labels: torch.Tensor) -> torch.Tensor:
        """Stratified sampling to maintain class distribution."""
        if labels is None:
            raise ValueError("Stratified sampling requires labels")
            
        # Count classes
        unique_labels, counts = torch.unique(labels, return_counts=True)
        class_probs = counts.float() / n_samples
        
        # Sample from each class proportionally
        indices = []
        for label, prob in zip(unique_labels, class_probs):
            class_mask = labels == label
            class_indices = torch.where(class_mask)[0]
            n_class_samples = max(1, int(self.batch_size * prob))
            
            if len(class_indices) > 0:
                if self.replacement:
                    sampled = class_indices[torch.randint(0, len(class_indices), (n_class_samples,))]
                else:
                    sampled = class_indices[torch.randperm(len(class_indices))[:n_class_samples]]
                indices.extend(sampled.tolist())
                
        # Shuffle and trim to batch size
        indices = torch.tensor(indices)
        if self.shuffle:
            indices = indices[torch.randperm(len(indices))]
        
        return indices[:self.batch_size]
    
    def _weighted_sample(self, n_samples: int, weights: Optional[torch.Tensor]) -> torch.Tensor:
        """Weighted sampling based on provided weights."""
        if weights is None:
            # Use uniform weights
            weights = torch.ones(n_samples)
            
        # Normalize weights
        weights = weights / weights.sum()
        
        # Sample with weights
        if self.replacement:
            indices = torch.multinomial(weights, self.batch_size, replacement=True)
        else:
            indices = torch.multinomial(weights, 
                                      min(self.batch_size, n_samples), 
                                      replacement=False)
            
        return indices
    
    def _balanced_sample(self, n_samples: int, labels: torch.Tensor) -> torch.Tensor:
        """Balanced sampling to equalize class representation."""
        if labels is None:
            raise ValueError("Balanced sampling requires labels")
            
        unique_labels = torch.unique(labels)
        n_classes = len(unique_labels)
        samples_per_class = self.batch_size // n_classes
        remainder = self.batch_size % n_classes
        
        indices = []
        for i, label in enumerate(unique_labels):
            class_mask = labels == label
            class_indices = torch.where(class_mask)[0]
            
            # Add extra sample for some classes to reach batch_size
            n_samples_this_class = samples_per_class + (1 if i < remainder else 0)
            
            if len(class_indices) > 0:
                if self.replacement or len(class_indices) >= n_samples_this_class:
                    sampled = class_indices[torch.randint(0, len(class_indices), 
                                                         (n_samples_this_class,))]
                else:
                    # Not enough samples, use all available
                    sampled = class_indices
                indices.extend(sampled.tolist())
                
        indices = torch.tensor(indices)
        if self.shuffle:
            indices = indices[torch.randperm(len(indices))]
            
        return indices
    
    def _focal_sample(self, n_samples: int, labels: torch.Tensor) -> torch.Tensor:
        """Focal sampling - focus on hard examples."""
        if labels is None:
            raise ValueError("Focal sampling requires labels")
            
        # Use difficulty scores if available, otherwise use class frequency
        if self.difficulty_scores:
            weights = torch.tensor([
                self.difficulty_scores.get(i, 1.0) 
                for i in range(n_samples)
            ])
        else:
            # Use inverse class frequency as proxy
            unique_labels, counts = torch.unique(labels, return_counts=True)
            class_weights = 1.0 / counts.float()
            weights = torch.zeros(n_samples)
            
            for label, weight in zip(unique_labels, class_weights):
                mask = labels == label
                weights[mask] = weight
                
        # Apply focal weighting (emphasize harder examples)
        focal_gamma = 2.0  # Focusing parameter
        weights = weights ** focal_gamma
        
        return self._weighted_sample(n_samples, weights)
    
    def _adaptive_sample(self, n_samples: int, labels: torch.Tensor) -> torch.Tensor:
        """Adaptive sampling based on training dynamics."""
        # Start with stratified, then adapt based on performance
        if self.epoch < 5:
            return self._stratified_sample(n_samples, labels)
            
        # Use loss history to weight samples
        if self.loss_history:
            weights = torch.tensor([
                self.loss_history.get(i, 1.0)
                for i in range(n_samples)
            ])
            # Smooth weights
            weights = torch.clamp(weights, 0.1, 10.0)
        else:
            weights = torch.ones(n_samples)
            
        return self._weighted_sample(n_samples, weights)
    
    def update_difficulty(self, indices: torch.Tensor, losses: torch.Tensor) -> None:
        """Update difficulty scores for focal/adaptive sampling."""
        for idx, loss in zip(indices, losses):
            idx_int = idx.item()
            # Exponential moving average
            alpha = 0.1
            if idx_int in self.difficulty_scores:
                self.difficulty_scores[idx_int] = (
                    alpha * loss.item() + 
                    (1 - alpha) * self.difficulty_scores[idx_int]
                )
            else:
                self.difficulty_scores[idx_int] = loss.item()
                
            self.loss_history[idx_int] = loss.item()
    
    def set_class_weights(self, weights: Dict[Any, float]) -> None:
        """Set explicit class weights for weighted sampling."""
        self.class_weights = weights
    
    def compute_class_weights(self, labels: torch.Tensor, 
                            mode: str = "inverse") -> Dict[Any, float]:
        """Compute class weights from label distribution."""
        unique_labels, counts = torch.unique(labels, return_counts=True)
        
        if mode == "inverse":
            weights = 1.0 / counts.float()
        elif mode == "inverse_sqrt":
            weights = 1.0 / torch.sqrt(counts.float())
        elif mode == "effective":
            # Effective number of samples
            beta = 0.999
            effective_num = 1.0 - torch.pow(beta, counts.float())
            weights = (1.0 - beta) / effective_num
        else:
            raise ValueError(f"Unknown weight computation mode: {mode}")
            
        # Normalize
        weights = weights / weights.sum() * len(unique_labels)
        
        return {label.item(): weight.item() 
                for label, weight in zip(unique_labels, weights)}
    
    def get_sampling_stats(self, labels: torch.Tensor) -> SamplingStats:
        """Get statistics about sampling distribution."""
        # Count original distribution
        unique_labels, counts = torch.unique(labels, return_counts=True)
        class_counts = {label.item(): count.item() 
                       for label, count in zip(unique_labels, counts)}
        
        # Compute effective sampling ratio
        total = sum(self.sampling_history.values())
        if total > 0:
            effective_ratio = {
                label: self.sampling_history.get(label, 0) / total / (count / len(labels))
                for label, count in class_counts.items()
            }
        else:
            effective_ratio = {label: 1.0 for label in class_counts}
            
        return SamplingStats(
            total_samples=total,
            class_counts=class_counts,
            sampling_counts=dict(self.sampling_history),
            effective_ratio=effective_ratio
        )
    
    def reset_epoch(self) -> None:
        """Reset for new epoch."""
        self.epoch += 1
        self.sampling_history.clear()
    
    def create_batch_sampler(self, dataset_size: int, 
                           labels: Optional[torch.Tensor] = None) -> List[List[int]]:
        """Create list of batches for entire dataset."""
        batches = []
        n_batches = dataset_size // self.batch_size
        
        if not self.drop_last and dataset_size % self.batch_size != 0:
            n_batches += 1
            
        for i in range(n_batches):
            if labels is not None:
                # Create fake labels tensor for this batch
                batch_labels = labels if i == 0 else labels
                indices = self(torch.arange(dataset_size), batch_labels)[1]
            else:
                indices = self._uniform_sample(dataset_size)
                
            batches.append(indices.tolist())
            
        return batches