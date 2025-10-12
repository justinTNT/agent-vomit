import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from collections import deque, defaultdict
from datetime import datetime, timedelta
import heapq
import time
from threading import Lock


class StreamJoiner(nn.Module):
    """
    Joins multiple data streams with temporal alignment.
    Handles streams with different frequencies and timing.
    """
    
    def __init__(
        self,
        join_type: str = "inner",  # inner, left, outer, asof
        time_window: float = 1000,  # ms
        interpolation: str = "linear",  # none, linear, nearest, zero
        max_buffer_size: int = 1000,
        tolerance: Optional[float] = None,  # for asof joins
        align_keys: Optional[List[str]] = None,  # keys to align on
        primary_stream: Optional[str] = None  # explicit primary stream for left/asof joins
    ):
        super().__init__()
        self.join_type = join_type
        self.time_window = time_window
        self.interpolation = interpolation
        self.max_buffer_size = max_buffer_size
        self.tolerance = tolerance or time_window / 10
        self.align_keys = align_keys
        self.primary_stream = primary_stream
        
        # Buffers for each stream
        self.buffers = defaultdict(lambda: deque(maxlen=max_buffer_size))
        self.watermarks = {}  # stream -> latest timestamp
        self.lock = Lock()
        
        # Output queue
        self.output_queue = []
        
        # Statistics
        self.stats = {
            "joins": 0,
            "misses": 0,
            "interpolations": 0,
            "dropped": 0
        }
        
    def forward(self, 
                stream_name: str,
                data: torch.Tensor,
                timestamp: Optional[float] = None,
                key: Optional[Any] = None) -> Optional[Dict[str, torch.Tensor]]:
        """Process new data from a stream and attempt join."""
        with self.lock:
            current_time = timestamp if timestamp is not None else time.time()
            
            # Add to buffer
            self.buffers[stream_name].append({
                "data": data,
                "timestamp": current_time,
                "key": key
            })
            
            # Update watermark
            self.watermarks[stream_name] = current_time
            
            # Try to produce joined output
            result = self._try_join(current_time)
            
            # Cleanup old data
            self._cleanup_buffers(current_time)
            
            return result
    
    def _try_join(self, current_time: float) -> Optional[Dict[str, torch.Tensor]]:
        """Attempt to join streams based on join type."""
        if self.join_type == "inner":
            return self._inner_join(current_time)
        elif self.join_type == "left":
            return self._left_join(current_time)
        elif self.join_type == "outer":
            return self._outer_join(current_time)
        elif self.join_type == "asof":
            return self._asof_join(current_time)
        else:
            raise ValueError(f"Unknown join type: {self.join_type}")
    
    def _inner_join(self, current_time: float) -> Optional[Dict[str, torch.Tensor]]:
        """Inner join - output only when all streams have data."""
        # Need at least 2 streams for a join
        if len(self.buffers) < 2:
            self.stats["misses"] += 1
            return None
            
        # Check if all streams have data
        if not all(len(buffer) > 0 for buffer in self.buffers.values()):
            self.stats["misses"] += 1
            return None
            
        # Find matching data within time window
        result = {}
        base_time = None
        
        for stream_name, buffer in self.buffers.items():
            # Find closest match
            best_match = None
            min_time_diff = float('inf')
            
            for item in buffer:
                time_diff = abs(item["timestamp"] - (base_time or current_time))
                if time_diff <= self.time_window / 1000 and time_diff < min_time_diff:
                    best_match = item
                    min_time_diff = time_diff
                    if base_time is None:
                        base_time = item["timestamp"]
                        
            if best_match is None:
                self.stats["misses"] += 1
                return None
                
            result[stream_name] = best_match["data"]
            
        self.stats["joins"] += 1
        return result
    
    def _left_join(self, current_time: float) -> Optional[Dict[str, torch.Tensor]]:
        """Left join - output when first stream has data."""
        # Need at least one stream designated as primary
        if not self.buffers:
            return None
            
        primary_stream = list(self.buffers.keys())[0]
        if not self.buffers[primary_stream]:
            return None
            
        # Get latest from primary stream
        primary_item = self.buffers[primary_stream][-1]
        result = {primary_stream: primary_item["data"]}
        base_time = primary_item["timestamp"]
        
        # Try to match other streams
        for stream_name, buffer in self.buffers.items():
            if stream_name == primary_stream:
                continue
                
            # Find best match or interpolate
            matched_data = self._find_or_interpolate(list(buffer), base_time)
            if matched_data is not None:
                result[stream_name] = matched_data
            elif self.interpolation == "zero":
                # Zero padding for missing data
                result[stream_name] = torch.zeros_like(primary_item["data"])
                
        self.stats["joins"] += 1
        return result
    
    def _outer_join(self, current_time: float) -> Optional[Dict[str, torch.Tensor]]:
        """Outer join - output when any stream has new data."""
        if not any(len(buffer) > 0 for buffer in self.buffers.values()):
            return None
            
        result = {}
        base_time = current_time
        
        # Include data from all streams
        for stream_name, buffer in self.buffers.items():
            if buffer:
                # Use latest available
                result[stream_name] = buffer[-1]["data"]
            else:
                # Could use default/zero values
                pass
                
        if result:
            self.stats["joins"] += 1
            return result
            
        return None
    
    def _asof_join(self, current_time: float) -> Optional[Dict[str, torch.Tensor]]:
        """As-of join - join with most recent data within tolerance."""
        # Similar to pandas merge_asof
        if len(self.buffers) < 2:
            return None
            
        # Use primary stream if set, otherwise the stream with most recent data
        if self.primary_stream and self.primary_stream in self.buffers:
            ref_stream = self.primary_stream
        else:
            # Find stream with most recent update
            ref_stream = max(self.buffers.keys(), 
                           key=lambda s: self.watermarks.get(s, 0))
                           
        if not self.buffers[ref_stream]:
            return None
            
        ref_item = self.buffers[ref_stream][-1]
        result = {ref_stream: ref_item["data"]}
        ref_time = ref_item["timestamp"]
        
        # Find as-of matches for other streams
        for stream_name, buffer in self.buffers.items():
            if stream_name == ref_stream:
                continue
                
            # Find most recent data before ref_time within tolerance
            best_match = None
            for item in reversed(buffer):
                time_diff = ref_time - item["timestamp"]
                if 0 <= time_diff <= self.tolerance / 1000:
                    best_match = item
                    break
                elif time_diff > self.tolerance / 1000:
                    break
                    
            if best_match:
                result[stream_name] = best_match["data"]
                
        if len(result) > 1:  # At least one match found
            self.stats["joins"] += 1
            return result
            
        return None
    
    def _find_or_interpolate(self, buffer: List, target_time: float) -> Optional[torch.Tensor]:
        """Find matching data or interpolate if enabled."""
        if not buffer:
            return None
            
        # Find exact or close match
        best_match = None
        min_diff = float('inf')
        
        for item in buffer:
            diff = abs(item["timestamp"] - target_time)
            if diff < min_diff:
                min_diff = diff
                best_match = item
                
        # Check if within window
        if min_diff <= self.time_window / 1000:
            return best_match["data"]
            
        # Try interpolation
        if self.interpolation == "linear":
            return self._linear_interpolate(buffer, target_time)
        elif self.interpolation == "nearest":
            return best_match["data"] if best_match else None
        
        return None
    
    def _linear_interpolate(self, buffer: List, target_time: float) -> Optional[torch.Tensor]:
        """Linear interpolation between two points."""
        # Find points before and after target time
        before, after = None, None
        
        sorted_items = sorted(buffer, key=lambda x: x["timestamp"])
        for i, item in enumerate(sorted_items):
            if item["timestamp"] <= target_time:
                before = item
            else:
                after = item
                break
                
        if before and after:
            # Interpolate
            t1, t2 = before["timestamp"], after["timestamp"]
            alpha = (target_time - t1) / (t2 - t1)
            
            interpolated = (1 - alpha) * before["data"] + alpha * after["data"]
            self.stats["interpolations"] += 1
            return interpolated
            
        return None
    
    def _cleanup_buffers(self, current_time: float) -> None:
        """Remove old data outside the join window."""
        cutoff_time = current_time - (self.time_window * 2) / 1000
        
        for buffer in self.buffers.values():
            while buffer and buffer[0]["timestamp"] < cutoff_time:
                buffer.popleft()
                self.stats["dropped"] += 1
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """Get current buffer status."""
        with self.lock:
            status = {
                "streams": list(self.buffers.keys()),
                "buffer_sizes": {
                    stream: len(buffer) 
                    for stream, buffer in self.buffers.items()
                },
                "watermarks": dict(self.watermarks),
                "stats": dict(self.stats)
            }
            return status
    
    def reset_buffers(self) -> None:
        """Clear all buffers."""
        with self.lock:
            self.buffers.clear()
            self.watermarks.clear()
            self.output_queue.clear()
    
    def set_primary_stream(self, stream_name: str) -> None:
        """Set primary stream for left joins."""
        # Ensure primary stream is first in buffer order
        if stream_name in self.buffers:
            # Move to front
            buffer = self.buffers[stream_name]
            del self.buffers[stream_name]
            new_buffers = defaultdict(lambda: deque(maxlen=self.max_buffer_size))
            new_buffers[stream_name] = buffer
            new_buffers.update(self.buffers)
            self.buffers = new_buffers