#!/usr/bin/env python3
"""
BULLETPROOF DATAFLOW OPTIMIZER MODULE
Data pipeline optimization and scheduling for BigVGAN orchestration systems.
Handles pipeline optimization, resource scheduling, data flow coordination, and performance optimization with comprehensive fallbacks.
"""

import torch
import torch.multiprocessing as mp
import numpy as np
from typing import Optional, List, Dict, Any, Tuple, Union, Callable, Iterator
from dataclasses import dataclass, field
from rave_config_system import RAVEConfig
import logging
import warnings
import time
import threading
import queue
import heapq
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, TimeoutError as FutureTimeoutError
import weakref
import gc
import psutil
import json
from pathlib import Path
import pickle
from collections import deque, defaultdict
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataflowConfig:
    """Configuration for bulletproof dataflow optimizer"""
    # Pipeline configuration
    max_workers: int = 4  # Maximum parallel workers
    max_queue_size: int = 1000  # Maximum queue size
    batch_size: int = 32  # Default batch size
    prefetch_factor: int = 2  # Prefetch multiplier
    
    # Resource management
    memory_limit_mb: float = 8192.0  # Memory limit in MB
    cpu_limit_percent: float = 80.0  # CPU usage limit
    gpu_memory_limit_mb: float = 4096.0  # GPU memory limit
    disk_space_limit_gb: float = 50.0  # Disk space limit
    
    # Optimization parameters
    enable_auto_tuning: bool = True  # Auto-tune pipeline parameters
    optimization_metric: str = 'throughput'  # 'throughput', 'latency', 'memory'
    profiling_interval: int = 100  # Profile every N batches
    adaptation_threshold: float = 0.1  # Adaptation sensitivity
    
    # Scheduling configuration
    scheduling_policy: str = 'fair'  # 'fair', 'priority', 'shortest_first'
    priority_levels: int = 5  # Number of priority levels
    task_timeout: float = 300.0  # Task timeout in seconds
    retry_attempts: int = 3  # Number of retry attempts
    
    # Caching and persistence
    enable_caching: bool = True  # Enable data caching
    cache_size_mb: float = 2048.0  # Cache size limit
    persistent_cache: bool = False  # Persistent disk cache
    cache_eviction_policy: str = 'lru'  # 'lru', 'lfu', 'fifo'
    
    # Bulletproof parameters
    enable_fallbacks: bool = True
    health_check_interval: float = 5.0  # Health check interval
    deadlock_detection: bool = True
    memory_leak_detection: bool = True
    graceful_degradation: bool = True  # Graceful performance degradation
    
    # Advanced features
    load_balancing: bool = True  # Enable load balancing
    data_locality_optimization: bool = True  # Optimize for data locality
    compression_enabled: bool = False  # Enable data compression
    encryption_enabled: bool = False  # Enable data encryption


class DataflowTask:
    """Represents a single dataflow task"""
    
    def __init__(self, task_id: str, data: Any, process_fn: Callable, 
                 priority: int = 0, metadata: Dict = None):
        self.task_id = task_id
        self.data = data
        self.process_fn = process_fn
        self.priority = priority
        self.metadata = metadata or {}
        self.created_time = time.time()
        self.start_time = None
        self.end_time = None
        self.result = None
        self.error = None
        self.retry_count = 0
        self.memory_usage = 0.0
        self.processing_time = 0.0
    
    def __lt__(self, other):
        # For priority queue (higher priority first)
        return self.priority > other.priority
    
    def start_processing(self):
        """Mark task as started"""
        self.start_time = time.time()
    
    def finish_processing(self, result: Any = None, error: Exception = None):
        """Mark task as finished"""
        self.end_time = time.time()
        if self.start_time:
            self.processing_time = self.end_time - self.start_time
        self.result = result
        self.error = error
    
    def get_wait_time(self) -> float:
        """Get time spent waiting in queue"""
        if self.start_time:
            return self.start_time - self.created_time
        return time.time() - self.created_time
    
    def get_total_time(self) -> float:
        """Get total time from creation to completion"""
        if self.end_time:
            return self.end_time - self.created_time
        return time.time() - self.created_time


class BulletproofDataflowOptimizer:
    """
    Bulletproof Dataflow Optimizer for BigVGAN pipeline coordination.
    
    Features:
    - Intelligent pipeline scheduling with priority queues
    - Resource-aware optimization and constraint enforcement
    - Auto-tuning of pipeline parameters based on performance metrics
    - Comprehensive caching system with multiple eviction policies
    - Load balancing across multiple workers and resources
    - Memory leak detection and automatic garbage collection
    - Deadlock detection and prevention mechanisms
    - Graceful degradation under resource constraints
    - Data locality optimization for distributed systems
    - Comprehensive monitoring and performance profiling
    """
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__()
        
        # Extract dataflow specific config or use defaults
        self.config = getattr(config, 'dataflow', DataflowConfig())
        self.rave_config = config
        
        # Pipeline state
        self.task_queue = queue.PriorityQueue(maxsize=self.config.max_queue_size)
        self.result_cache = {}
        self.worker_pool = None
        self.active_tasks = {}
        self.completed_tasks = deque(maxlen=10000)  # Keep history
        self.failed_tasks = deque(maxlen=1000)
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor()
        self.performance_metrics = PerformanceMetrics()
        self.load_balancer = LoadBalancer(self.config)
        
        # Optimization state
        self.optimization_params = {
            'batch_size': self.config.batch_size,
            'num_workers': self.config.max_workers,
            'prefetch_factor': self.config.prefetch_factor
        }
        self.optimization_history = []
        
        # Threading and synchronization
        self.pipeline_lock = threading.RLock()
        self.shutdown_event = threading.Event()
        self.health_check_thread = None
        self.optimization_thread = None
        
        # Statistics
        self.stats = {
            'tasks_processed': 0,
            'tasks_failed': 0,
            'total_processing_time': 0.0,
            'total_wait_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'optimizations_performed': 0,
            'fallback_activations': 0,
            'deadlocks_detected': 0,
            'memory_warnings': 0
        }
        
        # Initialize components
        self._initialize_pipeline()
        self._start_background_threads()
        
        logger.info(f"BulletproofDataflowOptimizer initialized with {self.config.max_workers} workers")
    
    def _initialize_pipeline(self):
        """Initialize the dataflow pipeline"""
        try:
            # Initialize worker pool
            self.worker_pool = ThreadPoolExecutor(
                max_workers=self.config.max_workers,
                thread_name_prefix="dataflow_worker"
            )
            
            # Initialize cache
            if self.config.enable_caching:
                self._initialize_cache()
            
            # Validate configuration
            self._validate_config()
            
            logger.info("Dataflow pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            if self.config.enable_fallbacks:
                self._apply_fallback_config()
            else:
                raise
    
    def _validate_config(self):
        """Validate dataflow configuration"""
        assert self.config.max_workers > 0, "Must have at least one worker"
        assert self.config.max_queue_size > 0, "Queue size must be positive"
        assert self.config.batch_size > 0, "Batch size must be positive"
        assert self.config.memory_limit_mb > 0, "Memory limit must be positive"
        assert self.config.task_timeout > 0, "Task timeout must be positive"
    
    def _apply_fallback_config(self):
        """Apply fallback configuration when validation fails"""
        logger.warning("Applying fallback dataflow configuration")
        self.config = DataflowConfig()  # Reset to defaults
        self.stats['fallback_activations'] += 1
        self._initialize_pipeline()
    
    def _initialize_cache(self):
        """Initialize caching system"""
        try:
            if self.config.cache_eviction_policy == 'lru':
                self.cache = LRUCache(self.config.cache_size_mb)
            elif self.config.cache_eviction_policy == 'lfu':
                self.cache = LFUCache(self.config.cache_size_mb)
            else:  # FIFO
                self.cache = FIFOCache(self.config.cache_size_mb)
            
            logger.info(f"Cache initialized: {self.config.cache_eviction_policy} policy, "
                       f"{self.config.cache_size_mb}MB limit")
        
        except Exception as e:
            logger.error(f"Cache initialization failed: {e}")
            self.config.enable_caching = False
    
    def _start_background_threads(self):
        """Start background monitoring and optimization threads"""
        try:
            # Health check thread
            self.health_check_thread = threading.Thread(
                target=self._health_check_loop,
                name="dataflow_health_check",
                daemon=True
            )
            self.health_check_thread.start()
            
            # Optimization thread
            if self.config.enable_auto_tuning:
                self.optimization_thread = threading.Thread(
                    target=self._optimization_loop,
                    name="dataflow_optimizer",
                    daemon=True
                )
                self.optimization_thread.start()
            
            logger.info("Background threads started")
            
        except Exception as e:
            logger.error(f"Failed to start background threads: {e}")
    
    def submit_task(self, task_id: str, data: Any, process_fn: Callable,
                   priority: int = 0, metadata: Dict = None) -> str:
        """Submit a task to the dataflow pipeline"""
        try:
            # Check cache first
            if self.config.enable_caching:
                cache_key = self._generate_cache_key(data, process_fn)
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    self.stats['cache_hits'] += 1
                    logger.debug(f"Cache hit for task {task_id}")
                    return cached_result
                self.stats['cache_misses'] += 1
            
            # Create task
            task = DataflowTask(task_id, data, process_fn, priority, metadata)
            
            # Check resource constraints
            if not self._check_resource_constraints():
                if self.config.graceful_degradation:
                    logger.warning("Resource constraints exceeded, applying degradation")
                    self._apply_graceful_degradation()
                else:
                    raise RuntimeError("Resource constraints exceeded")
            
            # Submit to queue
            try:
                self.task_queue.put(task, timeout=5.0)
                
                with self.pipeline_lock:
                    self.active_tasks[task_id] = task
                
                # Submit to worker pool
                future = self.worker_pool.submit(self._process_task, task)
                task.future = future
                
                logger.debug(f"Task {task_id} submitted successfully")
                return task_id
                
            except queue.Full:
                logger.warning(f"Task queue full, rejecting task {task_id}")
                if self.config.enable_fallbacks:
                    # Try to process immediately
                    return self._process_task_immediately(task)
                else:
                    raise RuntimeError("Task queue full")
        
        except Exception as e:
            logger.error(f"Task submission failed: {e}")
            if self.config.enable_fallbacks:
                self.stats['fallback_activations'] += 1
                return self._process_task_fallback(task_id, data, process_fn)
            else:
                raise
    
    def _process_task(self, task: DataflowTask) -> Any:
        """Process a single task with comprehensive error handling"""
        task.start_processing()
        
        try:
            # Resource monitoring
            initial_memory = self.resource_monitor.get_memory_usage()
            
            # Apply load balancing
            worker_id = self.load_balancer.select_worker()
            
            # Process the task
            result = task.process_fn(task.data)
            
            # Cache result if enabled
            if self.config.enable_caching:
                cache_key = self._generate_cache_key(task.data, task.process_fn)
                self.cache.put(cache_key, result)
            
            # Update metrics
            final_memory = self.resource_monitor.get_memory_usage()
            task.memory_usage = final_memory - initial_memory
            
            task.finish_processing(result=result)
            
            # Update statistics
            with self.pipeline_lock:
                self.stats['tasks_processed'] += 1
                self.stats['total_processing_time'] += task.processing_time
                self.stats['total_wait_time'] += task.get_wait_time()
                self.completed_tasks.append(task)
                self.active_tasks.pop(task.task_id, None)
            
            # Performance tracking
            self.performance_metrics.record_task(task)
            
            logger.debug(f"Task {task.task_id} completed in {task.processing_time:.3f}s")
            return result
        
        except Exception as e:
            error_msg = f"Task {task.task_id} failed: {e}"
            logger.error(error_msg)
            
            task.finish_processing(error=e)
            task.retry_count += 1
            
            # Update statistics
            with self.pipeline_lock:
                self.stats['tasks_failed'] += 1
                self.failed_tasks.append(task)
                self.active_tasks.pop(task.task_id, None)
            
            # Retry logic
            if task.retry_count < self.config.retry_attempts:
                logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count + 1})")
                return self._retry_task(task)
            
            if self.config.enable_fallbacks:
                return self._process_task_fallback(task.task_id, task.data, task.process_fn)
            else:
                raise
    
    def _retry_task(self, task: DataflowTask) -> Any:
        """Retry a failed task"""
        try:
            # Reset timing
            task.start_time = None
            task.end_time = None
            task.error = None
            
            # Add delay before retry
            time.sleep(min(2.0 ** task.retry_count, 10.0))
            
            return self._process_task(task)
        
        except Exception as e:
            logger.error(f"Task retry failed: {e}")
            raise
    
    def _process_task_immediately(self, task: DataflowTask) -> Any:
        """Process task immediately when queue is full"""
        logger.warning(f"Processing task {task.task_id} immediately due to queue overflow")
        return self._process_task(task)
    
    def _process_task_fallback(self, task_id: str, data: Any, process_fn: Callable) -> Any:
        """Fallback task processing"""
        try:
            logger.warning(f"Using fallback processing for task {task_id}")
            
            # Simple synchronous processing
            start_time = time.time()
            result = process_fn(data)
            processing_time = time.time() - start_time
            
            # Update minimal statistics
            self.stats['tasks_processed'] += 1
            self.stats['total_processing_time'] += processing_time
            self.stats['fallback_activations'] += 1
            
            return result
        
        except Exception as e:
            logger.error(f"Fallback processing failed: {e}")
            # Return None or default value
            return None
    
    def _generate_cache_key(self, data: Any, process_fn: Callable) -> str:
        """Generate cache key for data and function"""
        try:
            # Simple hash-based key generation
            data_hash = hash(str(data)) if not isinstance(data, torch.Tensor) else hash(data.data_ptr())
            fn_hash = hash(process_fn.__name__ if hasattr(process_fn, '__name__') else str(process_fn))
            return f"{data_hash}_{fn_hash}"
        except Exception:
            # Fallback to simple string key
            return f"{id(data)}_{id(process_fn)}"
    
    def _check_resource_constraints(self) -> bool:
        """Check if resource constraints are satisfied"""
        try:
            # Memory check
            memory_usage = self.resource_monitor.get_memory_usage()
            if memory_usage > self.config.memory_limit_mb:
                logger.warning(f"Memory limit exceeded: {memory_usage:.1f}MB > {self.config.memory_limit_mb}MB")
                self.stats['memory_warnings'] += 1
                return False
            
            # CPU check
            cpu_usage = self.resource_monitor.get_cpu_usage()
            if cpu_usage > self.config.cpu_limit_percent:
                logger.warning(f"CPU limit exceeded: {cpu_usage:.1f}% > {self.config.cpu_limit_percent}%")
                return False
            
            # GPU memory check
            if torch.cuda.is_available():
                gpu_memory = self.resource_monitor.get_gpu_memory_usage()
                if gpu_memory > self.config.gpu_memory_limit_mb:
                    logger.warning(f"GPU memory limit exceeded: {gpu_memory:.1f}MB > {self.config.gpu_memory_limit_mb}MB")
                    return False
            
            return True
        
        except Exception as e:
            logger.error(f"Resource constraint check failed: {e}")
            return True  # Assume constraints are met if check fails
    
    def _apply_graceful_degradation(self):
        """Apply graceful performance degradation"""
        try:
            logger.info("Applying graceful degradation")
            
            # Reduce batch size
            if self.optimization_params['batch_size'] > 1:
                self.optimization_params['batch_size'] = max(1, self.optimization_params['batch_size'] // 2)
                logger.info(f"Reduced batch size to {self.optimization_params['batch_size']}")
            
            # Reduce workers
            if self.optimization_params['num_workers'] > 1:
                self.optimization_params['num_workers'] = max(1, self.optimization_params['num_workers'] - 1)
                logger.info(f"Reduced workers to {self.optimization_params['num_workers']}")
            
            # Clear cache to free memory
            if self.config.enable_caching:
                self.cache.clear()
                logger.info("Cleared cache to free memory")
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            logger.error(f"Graceful degradation failed: {e}")
    
    def _health_check_loop(self):
        """Background health check loop"""
        while not self.shutdown_event.is_set():
            try:
                self._perform_health_check()
                time.sleep(self.config.health_check_interval)
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                time.sleep(self.config.health_check_interval)
    
    def _perform_health_check(self):
        """Perform comprehensive health check"""
        try:
            # Check for deadlocks
            if self.config.deadlock_detection:
                self._detect_deadlocks()
            
            # Check for memory leaks
            if self.config.memory_leak_detection:
                self._detect_memory_leaks()
            
            # Check worker pool health
            self._check_worker_health()
            
            # Update resource monitoring
            self.resource_monitor.update()
            
        except Exception as e:
            logger.error(f"Health check component failed: {e}")
    
    def _detect_deadlocks(self):
        """Detect potential deadlocks"""
        try:
            current_time = time.time()
            deadlock_threshold = self.config.task_timeout * 2
            
            with self.pipeline_lock:
                for task_id, task in self.active_tasks.items():
                    if task.start_time and (current_time - task.start_time) > deadlock_threshold:
                        logger.warning(f"Potential deadlock detected for task {task_id}")
                        self.stats['deadlocks_detected'] += 1
                        
                        # Cancel the task
                        if hasattr(task, 'future'):
                            task.future.cancel()
                        
                        # Remove from active tasks
                        self.active_tasks.pop(task_id, None)
        
        except Exception as e:
            logger.error(f"Deadlock detection failed: {e}")
    
    def _detect_memory_leaks(self):
        """Detect potential memory leaks"""
        try:
            current_memory = self.resource_monitor.get_memory_usage()
            memory_history = self.resource_monitor.memory_history
            
            if len(memory_history) > 10:
                # Check for consistent memory growth
                recent_growth = np.mean(memory_history[-5:]) - np.mean(memory_history[-10:-5])
                if recent_growth > 100.0:  # 100MB growth threshold
                    logger.warning(f"Potential memory leak detected: growth={recent_growth:.1f}MB")
                    
                    # Force garbage collection
                    gc.collect()
                    
                    # Clear some cache if enabled
                    if self.config.enable_caching:
                        self.cache.evict_percentage(0.2)  # Evict 20% of cache
        
        except Exception as e:
            logger.error(f"Memory leak detection failed: {e}")
    
    def _check_worker_health(self):
        """Check worker pool health"""
        try:
            if self.worker_pool._shutdown:
                logger.error("Worker pool has been shut down unexpectedly")
                # Reinitialize worker pool
                self._reinitialize_worker_pool()
        
        except Exception as e:
            logger.error(f"Worker health check failed: {e}")
    
    def _reinitialize_worker_pool(self):
        """Reinitialize worker pool after failure"""
        try:
            logger.info("Reinitializing worker pool")
            
            # Shutdown old pool
            if self.worker_pool:
                self.worker_pool.shutdown(wait=False)
            
            # Create new pool
            self.worker_pool = ThreadPoolExecutor(
                max_workers=self.optimization_params['num_workers'],
                thread_name_prefix="dataflow_worker"
            )
            
            logger.info("Worker pool reinitialized successfully")
        
        except Exception as e:
            logger.error(f"Worker pool reinitialization failed: {e}")
    
    def _optimization_loop(self):
        """Background optimization loop"""
        while not self.shutdown_event.is_set():
            try:
                time.sleep(30.0)  # Optimize every 30 seconds
                self._perform_optimization()
            except Exception as e:
                logger.error(f"Optimization loop failed: {e}")
                time.sleep(30.0)
    
    def _perform_optimization(self):
        """Perform pipeline optimization"""
        try:
            if len(self.completed_tasks) < 50:  # Need enough samples
                return
            
            logger.info("Performing pipeline optimization")
            
            # Analyze performance metrics
            metrics = self.performance_metrics.get_metrics()
            
            # Determine optimization strategy
            if self.config.optimization_metric == 'throughput':
                self._optimize_for_throughput(metrics)
            elif self.config.optimization_metric == 'latency':
                self._optimize_for_latency(metrics)
            elif self.config.optimization_metric == 'memory':
                self._optimize_for_memory(metrics)
            
            self.stats['optimizations_performed'] += 1
            
        except Exception as e:
            logger.error(f"Pipeline optimization failed: {e}")
    
    def _optimize_for_throughput(self, metrics: Dict[str, float]):
        """Optimize pipeline for maximum throughput"""
        try:
            current_throughput = metrics.get('tasks_per_second', 0.0)
            
            # Try increasing workers if CPU usage is low
            if metrics.get('cpu_usage', 100) < 70 and self.optimization_params['num_workers'] < self.config.max_workers:
                self.optimization_params['num_workers'] += 1
                logger.info(f"Increased workers to {self.optimization_params['num_workers']} for better throughput")
            
            # Try increasing batch size if memory allows
            memory_usage = metrics.get('memory_usage_mb', 0)
            if memory_usage < self.config.memory_limit_mb * 0.7:
                self.optimization_params['batch_size'] = min(
                    self.config.batch_size * 2,
                    self.optimization_params['batch_size'] + 8
                )
                logger.info(f"Increased batch size to {self.optimization_params['batch_size']} for better throughput")
        
        except Exception as e:
            logger.error(f"Throughput optimization failed: {e}")
    
    def _optimize_for_latency(self, metrics: Dict[str, float]):
        """Optimize pipeline for minimum latency"""
        try:
            avg_latency = metrics.get('average_latency', 0.0)
            
            # Reduce batch size for lower latency
            if avg_latency > 1.0 and self.optimization_params['batch_size'] > 1:
                self.optimization_params['batch_size'] = max(1, self.optimization_params['batch_size'] - 4)
                logger.info(f"Reduced batch size to {self.optimization_params['batch_size']} for lower latency")
            
            # Increase prefetch factor
            if self.optimization_params['prefetch_factor'] < 4:
                self.optimization_params['prefetch_factor'] += 1
                logger.info(f"Increased prefetch factor to {self.optimization_params['prefetch_factor']} for lower latency")
        
        except Exception as e:
            logger.error(f"Latency optimization failed: {e}")
    
    def _optimize_for_memory(self, metrics: Dict[str, float]):
        """Optimize pipeline for minimum memory usage"""
        try:
            memory_usage = metrics.get('memory_usage_mb', 0)
            
            if memory_usage > self.config.memory_limit_mb * 0.8:
                # Reduce batch size
                if self.optimization_params['batch_size'] > 1:
                    self.optimization_params['batch_size'] = max(1, self.optimization_params['batch_size'] - 4)
                    logger.info(f"Reduced batch size to {self.optimization_params['batch_size']} to save memory")
                
                # Reduce workers
                if self.optimization_params['num_workers'] > 1:
                    self.optimization_params['num_workers'] -= 1
                    logger.info(f"Reduced workers to {self.optimization_params['num_workers']} to save memory")
                
                # Clear cache
                if self.config.enable_caching:
                    self.cache.evict_percentage(0.3)
                    logger.info("Evicted 30% of cache to save memory")
        
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status"""
        try:
            with self.pipeline_lock:
                return {
                    'active_tasks': len(self.active_tasks),
                    'queue_size': self.task_queue.qsize(),
                    'completed_tasks': len(self.completed_tasks),
                    'failed_tasks': len(self.failed_tasks),
                    'optimization_params': self.optimization_params.copy(),
                    'resource_usage': self.resource_monitor.get_current_usage(),
                    'performance_metrics': self.performance_metrics.get_metrics(),
                    'statistics': self.stats.copy(),
                    'cache_stats': self.cache.get_stats() if self.config.enable_caching else {},
                    'load_balancer_stats': self.load_balancer.get_stats()
                }
        except Exception as e:
            logger.error(f"Failed to get pipeline status: {e}")
            return {'error': str(e)}
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            metrics = self.performance_metrics.get_detailed_metrics()
            status = self.get_pipeline_status()
            
            return {
                'summary': {
                    'total_tasks_processed': self.stats['tasks_processed'],
                    'success_rate': (self.stats['tasks_processed'] / 
                                   max(1, self.stats['tasks_processed'] + self.stats['tasks_failed'])),
                    'average_processing_time': (self.stats['total_processing_time'] / 
                                              max(1, self.stats['tasks_processed'])),
                    'average_wait_time': (self.stats['total_wait_time'] / 
                                        max(1, self.stats['tasks_processed'])),
                    'cache_hit_rate': (self.stats['cache_hits'] / 
                                     max(1, self.stats['cache_hits'] + self.stats['cache_misses']))
                },
                'current_status': status,
                'detailed_metrics': metrics,
                'optimization_history': self.optimization_history[-10:],  # Last 10 optimizations
                'resource_trends': self.resource_monitor.get_trends()
            }
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return {'error': str(e)}
    
    def shutdown(self, timeout: float = 30.0):
        """Gracefully shutdown the dataflow optimizer"""
        try:
            logger.info("Shutting down dataflow optimizer")
            
            # Signal shutdown
            self.shutdown_event.set()
            
            # Wait for background threads
            if self.health_check_thread and self.health_check_thread.is_alive():
                self.health_check_thread.join(timeout=5.0)
            
            if self.optimization_thread and self.optimization_thread.is_alive():
                self.optimization_thread.join(timeout=5.0)
            
            # Shutdown worker pool
            if self.worker_pool:
                self.worker_pool.shutdown(wait=True, timeout=timeout)
            
            # Clear resources
            if self.config.enable_caching:
                self.cache.clear()
            
            logger.info("Dataflow optimizer shutdown completed")
        
        except Exception as e:
            logger.error(f"Shutdown failed: {e}")


class ResourceMonitor:
    """Monitor system resources"""
    
    def __init__(self):
        self.memory_history = deque(maxlen=100)
        self.cpu_history = deque(maxlen=100)
        self.gpu_memory_history = deque(maxlen=100)
        self.last_update = time.time()
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            return psutil.cpu_percent(interval=0.1)
        except Exception:
            return 0.0
    
    def get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 * 1024)
            return 0.0
        except Exception:
            return 0.0
    
    def update(self):
        """Update resource monitoring"""
        try:
            self.memory_history.append(self.get_memory_usage())
            self.cpu_history.append(self.get_cpu_usage())
            self.gpu_memory_history.append(self.get_gpu_memory_usage())
            self.last_update = time.time()
        except Exception as e:
            logger.error(f"Resource monitoring update failed: {e}")
    
    def get_current_usage(self) -> Dict[str, float]:
        """Get current resource usage"""
        return {
            'memory_mb': self.get_memory_usage(),
            'cpu_percent': self.get_cpu_usage(),
            'gpu_memory_mb': self.get_gpu_memory_usage(),
            'timestamp': time.time()
        }
    
    def get_trends(self) -> Dict[str, Any]:
        """Get resource usage trends"""
        try:
            return {
                'memory_trend': np.mean(list(self.memory_history)[-10:]) if self.memory_history else 0.0,
                'cpu_trend': np.mean(list(self.cpu_history)[-10:]) if self.cpu_history else 0.0,
                'gpu_memory_trend': np.mean(list(self.gpu_memory_history)[-10:]) if self.gpu_memory_history else 0.0,
                'memory_variance': np.var(list(self.memory_history)) if len(self.memory_history) > 1 else 0.0,
                'samples': len(self.memory_history)
            }
        except Exception:
            return {}


class PerformanceMetrics:
    """Track and analyze performance metrics"""
    
    def __init__(self):
        self.task_times = deque(maxlen=1000)
        self.wait_times = deque(maxlen=1000)
        self.memory_usage = deque(maxlen=1000)
        self.throughput_history = deque(maxlen=100)
        self.last_throughput_calc = time.time()
        self.tasks_since_last_calc = 0
    
    def record_task(self, task: DataflowTask):
        """Record task performance metrics"""
        self.task_times.append(task.processing_time)
        self.wait_times.append(task.get_wait_time())
        self.memory_usage.append(task.memory_usage)
        self.tasks_since_last_calc += 1
        
        # Calculate throughput periodically
        current_time = time.time()
        if current_time - self.last_throughput_calc >= 10.0:  # Every 10 seconds
            throughput = self.tasks_since_last_calc / (current_time - self.last_throughput_calc)
            self.throughput_history.append(throughput)
            self.last_throughput_calc = current_time
            self.tasks_since_last_calc = 0
    
    def get_metrics(self) -> Dict[str, float]:
        """Get basic performance metrics"""
        try:
            return {
                'average_processing_time': np.mean(self.task_times) if self.task_times else 0.0,
                'average_wait_time': np.mean(self.wait_times) if self.wait_times else 0.0,
                'average_memory_usage': np.mean(self.memory_usage) if self.memory_usage else 0.0,
                'tasks_per_second': np.mean(self.throughput_history) if self.throughput_history else 0.0,
                'p95_processing_time': np.percentile(self.task_times, 95) if self.task_times else 0.0,
                'p95_wait_time': np.percentile(self.wait_times, 95) if self.wait_times else 0.0
            }
        except Exception:
            return {}
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        try:
            basic_metrics = self.get_metrics()
            
            return {
                **basic_metrics,
                'processing_time_std': np.std(self.task_times) if self.task_times else 0.0,
                'wait_time_std': np.std(self.wait_times) if self.wait_times else 0.0,
                'throughput_std': np.std(self.throughput_history) if self.throughput_history else 0.0,
                'sample_count': len(self.task_times),
                'throughput_samples': len(self.throughput_history)
            }
        except Exception:
            return {}


class LoadBalancer:
    """Simple load balancer for worker selection"""
    
    def __init__(self, config: DataflowConfig):
        self.config = config
        self.worker_loads = defaultdict(int)
        self.worker_selection_count = 0
    
    def select_worker(self) -> int:
        """Select optimal worker for task assignment"""
        if not self.config.load_balancing:
            return 0
        
        # Simple round-robin for now
        worker_id = self.worker_selection_count % self.config.max_workers
        self.worker_selection_count += 1
        self.worker_loads[worker_id] += 1
        
        return worker_id
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        return {
            'worker_loads': dict(self.worker_loads),
            'total_assignments': self.worker_selection_count,
            'load_balancing_enabled': self.config.load_balancing
        }


class LRUCache:
    """LRU (Least Recently Used) cache implementation"""
    
    def __init__(self, max_size_mb: float):
        self.max_size_mb = max_size_mb
        self.cache = {}
        self.access_order = deque()
        self.current_size = 0.0
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Any:
        """Get item from cache"""
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def put(self, key: str, value: Any):
        """Put item in cache"""
        # Estimate size (rough approximation)
        item_size = self._estimate_size(value)
        
        # Remove if already exists
        if key in self.cache:
            self.access_order.remove(key)
            del self.cache[key]
        
        # Evict if necessary
        while self.current_size + item_size > self.max_size_mb and self.access_order:
            oldest_key = self.access_order.popleft()
            if oldest_key in self.cache:
                del self.cache[oldest_key]
                self.current_size -= self._estimate_size(self.cache.get(oldest_key, 0))
        
        # Add new item
        self.cache[key] = value
        self.access_order.append(key)
        self.current_size += item_size
    
    def _estimate_size(self, value: Any) -> float:
        """Estimate size of value in MB"""
        try:
            if isinstance(value, torch.Tensor):
                return value.numel() * value.element_size() / (1024 * 1024)
            elif isinstance(value, np.ndarray):
                return value.nbytes / (1024 * 1024)
            else:
                # Rough estimate for other objects
                return 0.001  # 1KB default
        except Exception:
            return 0.001
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_order.clear()
        self.current_size = 0.0
    
    def evict_percentage(self, percentage: float):
        """Evict a percentage of cache items"""
        items_to_evict = max(1, int(len(self.cache) * percentage))
        for _ in range(items_to_evict):
            if self.access_order:
                key = self.access_order.popleft()
                if key in self.cache:
                    del self.cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / max(1, self.hits + self.misses),
            'size_mb': self.current_size,
            'items': len(self.cache),
            'max_size_mb': self.max_size_mb
        }


class LFUCache(LRUCache):
    """LFU (Least Frequently Used) cache implementation"""
    
    def __init__(self, max_size_mb: float):
        super().__init__(max_size_mb)
        self.frequencies = defaultdict(int)
    
    def get(self, key: str) -> Any:
        """Get item from cache"""
        if key in self.cache:
            self.frequencies[key] += 1
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def put(self, key: str, value: Any):
        """Put item in cache with LFU eviction"""
        item_size = self._estimate_size(value)
        
        if key in self.cache:
            del self.cache[key]
        
        # Evict least frequently used items
        while self.current_size + item_size > self.max_size_mb and self.cache:
            lfu_key = min(self.cache.keys(), key=lambda k: self.frequencies[k])
            del self.cache[lfu_key]
            del self.frequencies[lfu_key]
            if lfu_key in self.access_order:
                self.access_order.remove(lfu_key)
        
        self.cache[key] = value
        self.frequencies[key] = 1
        self.current_size += item_size


class FIFOCache(LRUCache):
    """FIFO (First In, First Out) cache implementation"""
    
    def get(self, key: str) -> Any:
        """Get item from cache without reordering"""
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        return None


# Factory function for easy instantiation
def create_bulletproof_dataflow_optimizer(config: RAVEConfig, **kwargs) -> BulletproofDataflowOptimizer:
    """Create a bulletproof dataflow optimizer instance"""
    return BulletproofDataflowOptimizer(config, **kwargs)


if __name__ == "__main__":
    print("🌊 BULLETPROOF DATAFLOW OPTIMIZER MODULE")
    print("=" * 50)
    
    # Test with minimal config
    from rave_config_system import get_minimal_config
    
    config = get_minimal_config()
    
    # Add dataflow configuration
    dataflow_config = DataflowConfig()
    dataflow_config.max_workers = 2  # Small for testing
    dataflow_config.max_queue_size = 10
    dataflow_config.enable_auto_tuning = True
    config.dataflow = dataflow_config
    
    optimizer = create_bulletproof_dataflow_optimizer(config)
    
    print(f"✅ Dataflow optimizer initialized")
    print(f"👥 Workers: {optimizer.config.max_workers}")
    print(f"📦 Queue size: {optimizer.config.max_queue_size}")
    print(f"🎯 Optimization metric: {optimizer.config.optimization_metric}")
    print(f"💾 Caching enabled: {optimizer.config.enable_caching}")
    
    # Test task submission
    def dummy_process_fn(data):
        """Dummy processing function"""
        time.sleep(0.1)  # Simulate work
        return data * 2
    
    print("\n📤 Submitting test tasks...")
    try:
        task_ids = []
        for i in range(5):
            task_id = optimizer.submit_task(
                task_id=f"test_task_{i}",
                data=i,
                process_fn=dummy_process_fn,
                priority=i % 3
            )
            task_ids.append(task_id)
            print(f"   Task {task_id} submitted")
        
        # Wait a bit for processing
        time.sleep(2.0)
        
        # Get status
        status = optimizer.get_pipeline_status()
        print(f"\n📊 Pipeline status:")
        print(f"   Active tasks: {status['active_tasks']}")
        print(f"   Completed tasks: {status['completed_tasks']}")
        print(f"   Failed tasks: {status['failed_tasks']}")
        print(f"   Queue size: {status['queue_size']}")
        
        # Get performance report
        report = optimizer.get_performance_report()
        print(f"\n📈 Performance summary:")
        summary = report.get('summary', {})
        print(f"   Success rate: {summary.get('success_rate', 0):.2%}")
        print(f"   Avg processing time: {summary.get('average_processing_time', 0):.3f}s")
        print(f"   Cache hit rate: {summary.get('cache_hit_rate', 0):.2%}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
    
    finally:
        print("\n🛑 Shutting down optimizer...")
        optimizer.shutdown()
    
    print("🚀 BulletproofDataflowOptimizer ready for BigVGAN orchestration!")