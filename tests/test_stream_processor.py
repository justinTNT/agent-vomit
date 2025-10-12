import torch
import time
import pytest
from modules.stream_processor import StreamProcessor


def test_tumbling_window_count_based():
    """Test tumbling windows with count-based triggering."""
    processor = StreamProcessor(
        window_type="tumbling",
        window_size=3,
        aggregation="sum",
        time_based=False
    )
    
    results = []
    for i in range(10):
        x = torch.tensor([float(i)])
        result = processor(x)
        if result is not None:
            results.append(result.item())
    
    # Should have 3 complete windows: [0,1,2], [3,4,5], [6,7,8]
    assert len(results) == 3
    assert results[0] == 3.0  # 0+1+2
    assert results[1] == 12.0  # 3+4+5
    assert results[2] == 21.0  # 6+7+8
    

def test_sliding_window():
    """Test sliding windows with overlap."""
    processor = StreamProcessor(
        window_type="sliding",
        window_size=4,
        window_slide=2,
        aggregation="mean",
        time_based=False
    )
    
    results = []
    for i in range(8):
        x = torch.tensor([float(i)])
        result = processor(x)
        if result is not None:
            results.append(result.item())
    
    # Windows should overlap
    assert len(results) >= 2
    

def test_session_window():
    """Test session windows with timeouts."""
    processor = StreamProcessor(
        window_type="session",
        session_timeout=100,  # 100ms
        aggregation="count",
        time_based=True
    )
    
    # First session
    for i in range(3):
        processor(torch.tensor([float(i)]))
        time.sleep(0.01)  # 10ms between events
    
    # Gap to trigger session end
    time.sleep(0.15)  # 150ms gap
    
    # Second session
    result = processor(torch.tensor([10.0]))
    
    # Should have emitted the first session
    assert result is not None
    assert result.item() == 3  # 3 events in first session
    

def test_backpressure():
    """Test backpressure mechanism."""
    processor = StreamProcessor(
        buffer_size=10,
        backpressure_threshold=0.8,
        window_type="tumbling",
        window_size=5
    )
    
    # Fill buffer beyond threshold
    dropped = 0
    for i in range(20):
        result = processor(torch.tensor([float(i)]))
        if result is None and i >= 8:  # After 80% full
            dropped += 1
    
    metrics = processor.get_metrics()
    assert metrics["dropped_count"] > 0
    

def test_time_based_window():
    """Test time-based tumbling window."""
    processor = StreamProcessor(
        window_type="tumbling",
        window_size=50,  # 50ms windows
        aggregation="max",
        time_based=True
    )
    
    start_time = time.time()
    results = []
    
    # Send events for 150ms
    while (time.time() - start_time) < 0.15:
        x = torch.tensor([time.time() - start_time])
        result = processor(x)
        if result is not None:
            results.append(result.item())
        time.sleep(0.01)
    
    # Should have at least 2 windows
    assert len(results) >= 2
    

def test_custom_aggregation():
    """Test custom aggregation function."""
    def percentile_90(values):
        stacked = torch.stack(values)
        return torch.quantile(stacked, 0.9, dim=0)
    
    processor = StreamProcessor(
        window_type="tumbling",
        window_size=10,
        aggregation=percentile_90
    )
    
    # First fill a complete window
    for i in range(10):
        result = processor(torch.tensor([float(i)]))
        
    # Result should be returned on the 10th element
    assert result is not None
    assert 7.0 <= result.item() <= 9.0  # 90th percentile of 0-9
    

def test_flush_operation():
    """Test manual flush of windows."""
    processor = StreamProcessor(
        window_type="tumbling",
        window_size=10,
        aggregation="mean"
    )
    
    # Add partial window
    for i in range(5):
        processor(torch.tensor([float(i)]))
    
    # Force flush
    result = processor.flush()
    assert result is not None
    assert result.item() == 2.0  # mean of 0-4
    

def test_multidimensional_data():
    """Test processing multidimensional tensors."""
    processor = StreamProcessor(
        window_type="sliding",
        window_size=3,
        window_slide=1,
        aggregation="mean"
    )
    
    # Process 2D tensors
    for i in range(5):
        x = torch.randn(4, 8)
        result = processor(x)
        if result is not None:
            assert result.shape == (4, 8)


if __name__ == "__main__":
    test_tumbling_window_count_based()
    test_sliding_window()
    test_session_window()
    test_backpressure()
    test_time_based_window()
    test_custom_aggregation()
    test_flush_operation()
    test_multidimensional_data()
    print("All StreamProcessor tests passed!")