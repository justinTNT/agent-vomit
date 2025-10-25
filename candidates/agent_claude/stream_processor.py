"""
StreamProcessor - Process streaming data with windowing.

This module handles sliding/tumbling windows, supports different aggregation methods,
and manages buffers for streaming data.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Literal
from collections import deque
import warnings


class StreamProcessor(nn.Module):
    """
    Process streaming data with windowing capabilities.
    
    Args:
        window_size: Size of the window in samples
        window_type: Type of window ('sliding' or 'tumbling')
        stride: Stride for sliding windows (default: 1)
        aggregation: Aggregation method ('mean', 'sum', 'max', 'min', 'std', 'var')
        max_buffer_size: Maximum buffer size to prevent memory overflow
        dim: Dimension along which to apply windowing (default: -1)
        **kwargs: Additional arguments
    """
    
    def __init__(
        self,
        window_size: int,
        window_type: Literal['sliding', 'tumbling'] = 'sliding',
        stride: int = 1,
        aggregation: Literal['mean', 'sum', 'max', 'min', 'std', 'var'] = 'mean',
        max_buffer_size: Optional[int] = None,
        dim: int = -1,
        **kwargs
    ):
        super().__init__()
        
        # Validate inputs
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if window_type not in ['sliding', 'tumbling']:
            raise ValueError(f"window_type must be 'sliding' or 'tumbling', got {window_type}")
        if aggregation not in ['mean', 'sum', 'max', 'min', 'std', 'var']:
            raise ValueError(f"Invalid aggregation method: {aggregation}")
        
        self.window_size = window_size
        self.window_type = window_type
        self.stride = stride if window_type == 'sliding' else window_size
        self.aggregation = aggregation
        self.max_buffer_size = max_buffer_size or window_size * 100
        self.dim = dim
        
        # Buffer for streaming data
        self.buffer = deque(maxlen=self.max_buffer_size)
        
        # Track statistics
        self.register_buffer('total_samples_processed', torch.tensor(0, dtype=torch.long))
        self.register_buffer('total_windows_processed', torch.tensor(0, dtype=torch.long))
        
        # Handle unused kwargs
        if kwargs:
            warnings.warn(f"Unused arguments: {list(kwargs.keys())}")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process streaming data through windowing.
        
        Args:
            x: Input tensor of shape (..., sequence_length, ...)
            
        Returns:
            Dictionary containing:
                - windows: Processed windows tensor
                - aggregated: Aggregated values per window
                - buffer_size: Current buffer size
                - windows_count: Number of windows processed
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        
        device = x.device
        
        # Add new data to buffer
        x_list = x.unbind(dim=self.dim)
        self.buffer.extend(x_list)
        
        # Update total samples
        self.total_samples_processed += len(x_list)
        
        # Extract windows
        windows = []
        aggregated_values = []
        
        buffer_tensor = torch.stack(list(self.buffer)) if self.buffer else torch.empty(0, device=device)
        
        if len(self.buffer) >= self.window_size:
            # Calculate number of windows
            num_windows = (len(self.buffer) - self.window_size) // self.stride + 1
            
            for i in range(num_windows):
                start_idx = i * self.stride
                end_idx = start_idx + self.window_size
                
                if end_idx <= len(self.buffer):
                    window = buffer_tensor[start_idx:end_idx]
                    windows.append(window)
                    
                    # Apply aggregation
                    if self.aggregation == 'mean':
                        agg_value = window.mean(dim=0)
                    elif self.aggregation == 'sum':
                        agg_value = window.sum(dim=0)
                    elif self.aggregation == 'max':
                        agg_value = window.max(dim=0)[0]
                    elif self.aggregation == 'min':
                        agg_value = window.min(dim=0)[0]
                    elif self.aggregation == 'std':
                        agg_value = window.std(dim=0, unbiased=False)
                    elif self.aggregation == 'var':
                        agg_value = window.var(dim=0, unbiased=False)
                    
                    aggregated_values.append(agg_value)
            
            # Update windows processed count
            self.total_windows_processed += num_windows
            
            # Remove processed samples for tumbling windows
            if self.window_type == 'tumbling' and num_windows > 0:
                for _ in range(num_windows * self.window_size):
                    if self.buffer:
                        self.buffer.popleft()
        
        # Stack results
        if windows:
            windows_tensor = torch.stack(windows)
            aggregated_tensor = torch.stack(aggregated_values)
        else:
            # Return empty tensors with correct shape
            windows_tensor = torch.empty(0, self.window_size, *x.shape[1:], device=device)
            aggregated_tensor = torch.empty(0, *x.shape[1:], device=device)
        
        return {
            'windows': windows_tensor,
            'aggregated': aggregated_tensor,
            'buffer_size': torch.tensor(len(self.buffer), device=device),
            'windows_count': torch.tensor(len(windows), device=device)
        }
    
    def reset_buffer(self):
        """Clear the internal buffer."""
        self.buffer.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'total_samples_processed': self.total_samples_processed.item(),
            'total_windows_processed': self.total_windows_processed.item(),
            'current_buffer_size': len(self.buffer),
            'window_size': self.window_size,
            'window_type': self.window_type,
            'stride': self.stride,
            'aggregation': self.aggregation
        }