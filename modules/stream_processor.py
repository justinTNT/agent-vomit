import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Any, Tuple
from collections import deque
from threading import Lock
import time


class StreamProcessor(nn.Module):
    """
    Real-time stream processing with windowing and backpressure.
    Supports tumbling, sliding, and session windows with various aggregations.
    """
    
    def __init__(
        self,
        window_type: str = "tumbling",  # tumbling, sliding, session
        window_size: int = 100,  # number of elements or time in ms
        window_slide: Optional[int] = None,  # for sliding windows
        session_timeout: Optional[int] = None,  # for session windows
        aggregation: str = "mean",  # mean, sum, max, min, count, custom
        buffer_size: int = 1000,
        backpressure_threshold: float = 0.8,
        time_based: bool = False
    ):
        super().__init__()
        self.window_type = window_type
        self.window_size = window_size
        self.window_slide = window_slide or window_size
        self.session_timeout = session_timeout or window_size
        self.aggregation = aggregation
        self.buffer_size = buffer_size
        self.backpressure_threshold = backpressure_threshold
        self.time_based = time_based
        
        # Buffers and state
        self.buffer = deque(maxlen=buffer_size)
        self.windows = deque()
        self.current_window = []
        self.window_start_time = None
        self.last_event_time = None
        self.lock = Lock()
        
        # Backpressure state
        self.processing_delay = 0.0
        self.dropped_count = 0
        
        # Aggregation functions
        self.aggregation_funcs = {
            "mean": lambda x: torch.stack(x).mean(dim=0),
            "sum": lambda x: torch.stack(x).sum(dim=0),
            "max": lambda x: torch.stack(x).max(dim=0)[0],
            "min": lambda x: torch.stack(x).min(dim=0)[0],
            "count": lambda x: torch.tensor(len(x))
        }
        
    def forward(self, x: torch.Tensor, timestamp: Optional[float] = None) -> Optional[torch.Tensor]:
        """Process a single element through the stream."""
        with self.lock:
            # Apply backpressure if needed
            if self._should_drop():
                self.dropped_count += 1
                return None
                
            # Add to buffer
            current_time = timestamp if timestamp is not None else time.time()
            self.buffer.append((x, current_time))
            
            # Process windows
            result = self._process_windows(current_time)
            
            return result
    
    def _should_drop(self) -> bool:
        """Determine if we should drop data due to backpressure."""
        buffer_usage = len(self.buffer) / self.buffer_size
        return buffer_usage > self.backpressure_threshold
    
    def _process_windows(self, current_time: float) -> Optional[torch.Tensor]:
        """Process windows based on window type."""
        if self.window_type == "tumbling":
            return self._process_tumbling_window(current_time)
        elif self.window_type == "sliding":
            return self._process_sliding_window(current_time)
        elif self.window_type == "session":
            return self._process_session_window(current_time)
        else:
            raise ValueError(f"Unknown window type: {self.window_type}")
    
    def _process_tumbling_window(self, current_time: float) -> Optional[torch.Tensor]:
        """Process tumbling windows."""
        if self.window_start_time is None:
            self.window_start_time = current_time
        
        # Add to current window first
        self.current_window.append(self.buffer[-1][0])
            
        # Check if window should close
        if self.time_based:
            window_elapsed = (current_time - self.window_start_time) * 1000  # to ms
            should_close = window_elapsed >= self.window_size
        else:
            should_close = len(self.current_window) >= self.window_size
            
        if should_close:
            # Aggregate and return
            result = self._aggregate(self.current_window)
            self.current_window = []
            self.window_start_time = current_time
            return result
        else:
            return None
    
    def _process_sliding_window(self, current_time: float) -> Optional[torch.Tensor]:
        """Process sliding windows."""
        self.current_window.append(self.buffer[-1])
        
        # Remove old elements
        if self.time_based:
            cutoff_time = current_time - (self.window_size / 1000.0)
            self.current_window = [
                (x, t) for x, t in self.current_window 
                if t >= cutoff_time
            ]
        else:
            if len(self.current_window) > self.window_size:
                self.current_window = self.current_window[-self.window_size:]
        
        # Check if we should emit
        if self.time_based:
            should_emit = (self.last_event_time is None or 
                          (current_time - self.last_event_time) * 1000 >= self.window_slide)
        else:
            should_emit = len(self.current_window) >= self.window_slide
            
        if should_emit and self.current_window:
            self.last_event_time = current_time
            values = [x for x, _ in self.current_window]
            return self._aggregate(values)
        
        return None
    
    def _process_session_window(self, current_time: float) -> Optional[torch.Tensor]:
        """Process session windows."""
        # Check for session timeout
        if (self.last_event_time is not None and 
            (current_time - self.last_event_time) * 1000 > self.session_timeout):
            # Session ended, emit if we have data
            if self.current_window:
                result = self._aggregate(self.current_window)
                self.current_window = []
                self.last_event_time = current_time
                return result
        
        # Add to current session
        self.current_window.append(self.buffer[-1][0])
        self.last_event_time = current_time
        return None
    
    def _aggregate(self, values: List[torch.Tensor]) -> torch.Tensor:
        """Apply aggregation function to window values."""
        if not values:
            return None
            
        if isinstance(self.aggregation, str) and self.aggregation in self.aggregation_funcs:
            return self.aggregation_funcs[self.aggregation](values)
        elif callable(self.aggregation):
            # Custom aggregation function
            return self.aggregation(values)
        else:
            raise ValueError(f"Invalid aggregation: {self.aggregation}")
    
    def flush(self) -> Optional[torch.Tensor]:
        """Force emit current window."""
        with self.lock:
            if self.current_window:
                if isinstance(self.current_window[0], tuple):
                    values = [x for x, _ in self.current_window]
                else:
                    values = self.current_window
                result = self._aggregate(values)
                self.current_window = []
                return result
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get stream processing metrics."""
        return {
            "buffer_size": len(self.buffer),
            "buffer_capacity": self.buffer_size,
            "dropped_count": self.dropped_count,
            "current_window_size": len(self.current_window),
            "processing_delay": self.processing_delay
        }