# core/utils/profiler.py

"""
This module provides a profiler for timing and memory usage of code blocks.

The Profiler class allows you to:
- Start and stop timers for specific code blocks
- Track function execution time and memory usage
- Get flattened metrics with optional execution counts
- Monitor memory usage with adaptive sampling
"""

import time
import os
import threading
import tracemalloc
from typing import Dict, Any, Optional, List, Tuple
from contextlib import contextmanager
import psutil

# Setup basic logging for profiler
# import logging
# logger = logging.getLogger(__name__)

class Profiler:
    def __init__(
            self, 
            reset_on_init: bool = True, 
            use_tracemalloc: bool = False, 
            min_poll_interval: float = 0.001, 
            max_poll_interval: float = 0.1
        ):
        """Initialize a new Profiler instance.
        
        Args:
            reset_on_init: If True, reset timings upon initialization
            use_tracemalloc: If True, use tracemalloc for Python object tracking
            min_poll_interval: Minimum interval (seconds) for memory polling
            max_poll_interval: Maximum interval (seconds) for memory polling
        """
        self.timings = {}
        self._active_timers = set()  # Track currently active timers
        self._process = psutil.Process(os.getpid())  # Get current process for memory tracking
        self._min_poll_interval = min_poll_interval
        self._max_poll_interval = max_poll_interval
        self._timer_flags = {}  # Per-timer stop flags
        self._timer_threads = {}  # Track memory threads by key
        self._lock = threading.Lock()  # Thread synchronization for shared data
        self._use_tracemalloc = use_tracemalloc
        
        if reset_on_init:
            self.reset()
            
        # logger.info(f"Initialized new Profiler instance (tracemalloc: {use_tracemalloc})")

    def _get_timer_dict(self, key: str) -> Dict:
        """Thread-safe access to timer dictionary."""
        keys = key.split(".")
        with self._lock:
            current_level = self.timings
            for k in keys[:-1]:
                if k not in current_level:
                    current_level[k] = {}
                current_level = current_level[k]
            
            if keys[-1] not in current_level:
                current_level[keys[-1]] = {}
                
            return current_level[keys[-1]]

    def _update_timer_value(self, key: str, field: str, value: Any) -> None:
        """Thread-safe update of a timer field."""
        timer_dict = self._get_timer_dict(key)
        with self._lock:
            timer_dict[field] = value

    def start(self, key: str) -> None:
        """Start timing for a specific key.
        
        Args:
            key: Dot-separated string representing the timing hierarchy
        """
        with self._lock:
            if key in self._active_timers:
                error_msg = f"Timer '{key}' is already running"
                # logger.error(error_msg)
                raise ValueError(error_msg)
                
            self._active_timers.add(key)
        
        # Initialize per-timer stop flag
        self._timer_flags[key] = False
        
        # Get current memory metrics
        start_time = time.time()
        start_memory = self._process.memory_info().rss / 1024 / 1024  # Memory in MB
        
        # Get system memory for differential analysis
        system_memory = psutil.virtual_memory()
        system_memory_start = {
            "total": system_memory.total,
            "available": system_memory.available,
            "percent": system_memory.percent
        }
        
        # Start tracemalloc if enabled
        tracemalloc_snapshot = None
        if self._use_tracemalloc:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            tracemalloc_snapshot = tracemalloc.take_snapshot()
        
        # Update timer data
        timer_data = {
            "start": start_time,
            "start_memory": start_memory,
            "peak_memory": start_memory,  # Initialize peak memory
            "system_memory_start": system_memory_start,
            "count": self._get_timer_dict(key).get("count", 0) + 1,
            "memory_samples": []  # Store memory samples for analysis
        }
        
        if tracemalloc_snapshot:
            timer_data["tracemalloc_start"] = tracemalloc_snapshot
            
        # Thread-safe update of timer data
        keys = key.split(".")
        current_level = self.timings
        with self._lock:
            for k in keys[:-1]:
                if k not in current_level:
                    current_level[k] = {}
                current_level = current_level[k]
            current_level[keys[-1]] = timer_data
            
        # logger.debug(f"Started timer for '{key}' at {start_time}")

    def _update_peak_memory(self, key: str, current_memory: float, timestamp: float) -> None:
        """Thread-safe update of peak memory."""
        timer_dict = self._get_timer_dict(key)
        
        with self._lock:
            # Store memory sample with timestamp
            if "memory_samples" in timer_dict:
                timer_dict["memory_samples"].append((timestamp, current_memory))
                
            # Update peak memory if current is higher
            if current_memory > timer_dict.get("peak_memory", 0):
                timer_dict["peak_memory"] = current_memory
                timer_dict["peak_time"] = timestamp

    def _track_memory(self, key: str):
        """Continuously track memory usage with adaptive sampling."""
        last_memory = self._process.memory_info().rss / 1024 / 1024
        current_interval = self._min_poll_interval
        volatility_history = []
        
        try:
            while key in self._active_timers and not self._timer_flags.get(key, True):
                try:
                    # Get current memory and timestamp
                    timestamp = time.time()
                    current_memory = self._process.memory_info().rss / 1024 / 1024
                    
                    # Update peak memory
                    self._update_peak_memory(key, current_memory, timestamp)
                    
                    # Calculate memory volatility for adaptive sampling
                    volatility = abs(current_memory - last_memory)
                    volatility_history.append(volatility)
                    
                    # Adjust sampling interval based on recent volatility
                    if len(volatility_history) > 10:
                        volatility_history.pop(0)
                        avg_volatility = sum(volatility_history) / len(volatility_history)
                        
                        # More volatile = faster sampling, less volatile = slower sampling
                        if avg_volatility > 1.0:  # High volatility
                            current_interval = self._min_poll_interval
                        elif avg_volatility < 0.1:  # Low volatility
                            current_interval = min(current_interval * 1.5, self._max_poll_interval)
                    
                    last_memory = current_memory
                    time.sleep(current_interval)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break  # Stop if process no longer exists or accessible
        except Exception as e:
            # logger.error(f"Memory tracking error for {key}: {str(e)}")
            pass
        finally:
            # Clean up
            if key in self._timer_threads:
                with self._lock:
                    if key in self._timer_threads:
                        del self._timer_threads[key]

    @contextmanager
    def track(self, key: str):
        """Context manager for timing a block of code.
        
        Args:
            key: Dot-separated string representing the timing hierarchy
            
        Example:
            with profiler.track("retrieval.chunking"):
                do_something()
        """
        memory_thread = None
        try:
            self.start(key)
            
            # Start memory tracking in a separate thread
            memory_thread = threading.Thread(target=self._track_memory, args=(key,))
            memory_thread.daemon = True
            with self._lock:
                self._timer_threads[key] = memory_thread
            memory_thread.start()
            
            yield
        finally:
            # Signal thread to stop and clean up
            with self._lock:
                self._timer_flags[key] = True
                
            # Wait for thread to finish
            if memory_thread and memory_thread.is_alive():
                memory_thread.join(timeout=1.0)  # Wait up to 1 second for thread to finish
                
            # Stop the timer
            self.stop(key)

    def track_func(self, key: str):
        """Decorator for timing function execution.
        
        Args:
            key: Dot-separated string representing the timing hierarchy
            
        Example:
            @profiler.track_func("retrieval.embedding")
            def embed_documents():
                pass
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                # logger.debug(f"Executing tracked function '{func.__name__}' with key '{key}'")
                with self.track(key):
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    def reset(self) -> None:
        """Reset all timings and active timers."""
        with self._lock:
            self.timings = {}
            self._active_timers.clear()
            
        # Clean up all timer flags and threads
        for key in list(self._timer_flags.keys()):
            self._timer_flags[key] = True
            
        # Wait for all threads to finish
        for key, thread in list(self._timer_threads.items()):
            if thread and thread.is_alive():
                thread.join(timeout=0.5)
                
        self._timer_flags.clear()
        self._timer_threads.clear()
        
        # logger.info("Profiler reset - all timings cleared")

    def get_metrics(self, include_counts: bool = False) -> Dict[str, Any]:
        """Get flattened metrics with optional execution counts.
        
        Args:
            include_counts: If True, include execution counts in the output
            
        Returns:
            Dictionary mapping timer keys to their metrics
        """
        metrics = {}
        
        def _flatten(d: Dict, prefix: Optional[str] = None, result: Optional[Dict] = None) -> Dict:
            if result is None:
                result = {}
                
            for k, v in d.items():
                new_key = f"{prefix}.{k}" if prefix else k
                
                if isinstance(v, dict):
                    if "start" in v and "end" in v:
                        if include_counts:
                            result[new_key] = {
                                "duration": v["total"],
                                "count": v["count"],
                                "avg_duration": v["total"] / v["count"],
                                "memory": {
                                    "total_mb": v.get("total_memory", 0),
                                    "avg_mb": v.get("total_memory", 0) / v["count"],
                                    "last_mb": v.get("end_memory", 0) - v.get("start_memory", 0)
                                }
                            }
                        else:
                            result[new_key] = {
                                "duration": v["total"],
                                "memory_mb": v.get("total_memory", 0)
                            }
                    else:
                        _flatten(v, new_key, result)
            return result

        metrics = _flatten(self.timings)
        # logger.debug(f"Retrieved metrics: {metrics}")
        return metrics

    def get_active_timers(self) -> set:
        """Get the set of currently active timer keys."""
        return self._active_timers.copy()
