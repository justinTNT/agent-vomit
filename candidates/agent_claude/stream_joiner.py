import torch
import torch.nn as nn
import warnings
from collections import deque, defaultdict
from typing import Optional, Dict, List, Tuple
import time


class StreamJoiner(nn.Module):
    def __init__(self,
                 join_type: str = 'inner',      # Type of join operation
                 window_size: float = 1000,     # Window size for joining (ms)
                 timeout: float = 5.0,          # Timeout for old data (seconds)
                 tolerance: float = 10.0,       # Time tolerance for matching (ms)
                 interpolate: bool = True,      # Whether to interpolate missing values
                 **kwargs):                     # REQUIRED: Accept extra kwargs
        super().__init__()
        
        # Log unknown parameters
        if kwargs:
            warnings.warn(f"Ignoring unknown parameters: {list(kwargs.keys())}")
        
        self.join_type = join_type
        self.window_size = window_size
        self.timeout = timeout
        self.tolerance = tolerance
        self.interpolate = interpolate
        
        # Stream buffers: stream_name -> deque of (timestamp, data) tuples
        self.stream_buffers = defaultdict(deque)
        
        # Track stream metadata
        self.stream_info = {}  # stream_name -> dict of metadata
        
        # Statistics
        self.join_stats = {
            'successful_joins': 0,
            'failed_joins': 0,
            'timeouts': 0,
            'interpolations': 0
        }
    
    def forward(self, stream_name: str, data: torch.Tensor, 
                timestamp: float = None) -> Optional[Dict[str, torch.Tensor]]:
        """
        Add data to a stream and attempt to join with other streams.
        
        Args:
            stream_name: Name of the stream
            data: Data tensor for this stream
            timestamp: Timestamp in milliseconds (defaults to current time)
        
        Returns:
            Joined data dictionary if join is possible, None otherwise
        """
        if timestamp is None:
            timestamp = time.time() * 1000  # Convert to milliseconds
        
        # Update stream info
        if stream_name not in self.stream_info:
            self.stream_info[stream_name] = {
                'shape': data.shape,
                'dtype': data.dtype,
                'device': data.device,
                'count': 0
            }
        self.stream_info[stream_name]['count'] += 1
        
        # Add to buffer
        self.stream_buffers[stream_name].append((timestamp, data))
        
        # Clean old data
        self._clean_buffers(timestamp)
        
        # Try to join
        result = self._attempt_join(timestamp)
        
        return result
    
    def _clean_buffers(self, current_timestamp: float):
        """Remove old data outside the timeout window."""
        timeout_ms = self.timeout * 1000
        
        for stream_name, buffer in self.stream_buffers.items():
            # Convert deque to list for safe iteration
            buffer_list = list(buffer)
            
            # Remove old entries
            while buffer_list and (current_timestamp - buffer_list[0][0]) > timeout_ms:
                self.stream_buffers[stream_name].popleft()
                buffer_list.pop(0)
                self.join_stats['timeouts'] += 1
    
    def _attempt_join(self, target_timestamp: float) -> Optional[Dict[str, torch.Tensor]]:
        """Attempt to join streams at the target timestamp."""
        # Check if we have data from all required streams
        if self.join_type == 'inner':
            required_streams = set(self.stream_info.keys())
            if not all(len(self.stream_buffers[s]) > 0 for s in required_streams):
                return None
        
        # Find matching data points
        joined_data = {}
        streams_to_join = list(self.stream_info.keys())
        
        for stream_name in streams_to_join:
            buffer = self.stream_buffers[stream_name]
            
            if not buffer:
                if self.join_type == 'inner':
                    self.join_stats['failed_joins'] += 1
                    return None
                else:
                    # For outer joins, skip missing streams
                    continue
            
            # Find closest timestamp within tolerance
            closest_data = self._find_closest_data(
                list(buffer), target_timestamp, self.tolerance
            )
            
            if closest_data is not None:
                joined_data[stream_name] = closest_data
            elif self.interpolate and len(buffer) >= 2:
                # Try interpolation
                interpolated = self._interpolate_data(
                    list(buffer), target_timestamp
                )
                if interpolated is not None:
                    joined_data[stream_name] = interpolated
                    self.join_stats['interpolations'] += 1
            elif self.join_type == 'inner':
                self.join_stats['failed_joins'] += 1
                return None
        
        if joined_data:
            # Remove used data from buffers
            self._consume_joined_data(target_timestamp)
            self.join_stats['successful_joins'] += 1
            
            # Add metadata
            joined_data['_timestamp'] = target_timestamp
            joined_data['_join_type'] = self.join_type
            
            return joined_data
        
        return None
    
    def _find_closest_data(self, buffer: List[Tuple[float, torch.Tensor]], 
                          target_timestamp: float, 
                          tolerance: float) -> Optional[torch.Tensor]:
        """Find data point closest to target timestamp within tolerance."""
        min_diff = float('inf')
        closest_data = None
        
        for timestamp, data in buffer:
            diff = abs(timestamp - target_timestamp)
            if diff <= tolerance and diff < min_diff:
                min_diff = diff
                closest_data = data
        
        return closest_data
    
    def _interpolate_data(self, buffer: List[Tuple[float, torch.Tensor]], 
                         target_timestamp: float) -> Optional[torch.Tensor]:
        """Interpolate data at target timestamp."""
        # Find surrounding points
        before = None
        after = None
        
        for timestamp, data in buffer:
            if timestamp <= target_timestamp:
                before = (timestamp, data)
            elif timestamp > target_timestamp and after is None:
                after = (timestamp, data)
                break
        
        if before is not None and after is not None:
            # Linear interpolation
            t1, data1 = before
            t2, data2 = after
            
            # Interpolation weight
            alpha = (target_timestamp - t1) / (t2 - t1)
            
            # Interpolate
            interpolated = (1 - alpha) * data1 + alpha * data2
            return interpolated
        
        return None
    
    def _consume_joined_data(self, timestamp: float):
        """Remove data points that have been joined."""
        for stream_name, buffer in self.stream_buffers.items():
            # Remove all data points at or before the join timestamp
            while buffer and buffer[0][0] <= timestamp:
                buffer.popleft()
    
    def get_buffer_status(self) -> Dict[str, Dict]:
        """Get current status of all stream buffers."""
        status = {}
        
        for stream_name, buffer in self.stream_buffers.items():
            if buffer:
                timestamps = [t for t, _ in buffer]
                status[stream_name] = {
                    'count': len(buffer),
                    'oldest_timestamp': min(timestamps),
                    'newest_timestamp': max(timestamps),
                    'time_span': max(timestamps) - min(timestamps)
                }
            else:
                status[stream_name] = {
                    'count': 0,
                    'oldest_timestamp': None,
                    'newest_timestamp': None,
                    'time_span': 0
                }
        
        return status
    
    def get_join_statistics(self) -> Dict[str, int]:
        """Get join operation statistics."""
        return self.join_stats.copy()
    
    def reset(self):
        """Reset all buffers and statistics."""
        self.stream_buffers.clear()
        self.stream_info.clear()
        self.join_stats = {
            'successful_joins': 0,
            'failed_joins': 0,
            'timeouts': 0,
            'interpolations': 0
        }
    
    def set_join_type(self, join_type: str):
        """Change join type dynamically."""
        if join_type not in ['inner', 'outer', 'left']:
            raise ValueError(f"Invalid join type: {join_type}")
        self.join_type = join_type