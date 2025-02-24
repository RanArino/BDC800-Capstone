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
        if key in self._active_timers:
            error_msg = f"Timer '{key}' is already running"
            # logger.error(error_msg)
            raise ValueError(error_msg)
            
        self._active_timers.add(key)
        keys = key.split(".")
        current_level = self.timings
        
        for k in keys[:-1]:
            if k not in current_level:
                current_level[k] = {}
            current_level = current_level[k]
        
        start_time = time.time()
        start_memory = self._process.memory_info().rss / 1024 / 1024  # Memory in MB
        current_level[keys[-1]] = {
            "start": start_time,
            "start_memory": start_memory,
            "count": current_level.get(keys[-1], {}).get("count", 0) + 1
        }
        # logger.debug(f"Started timer for '{key}' at {start_time}")

    def stop(self, key: str) -> Optional[Dict[str, float]]:
        """Stop timing for a specific key and return the elapsed time and memory usage.
        
        Args:
            key: Dot-separated string representing the timing hierarchy
            
        Returns:
            Dictionary containing elapsed time and memory metrics, or None if the timer wasn't started
        """
        if key not in self._active_timers:
            error_msg = f"Timer '{key}' was not started"
            # logger.error(error_msg)
            raise ValueError(error_msg)
            
        self._active_timers.remove(key)
        keys = key.split(".")
        current_level = self.timings
        
        for k in keys[:-1]:
            if k not in current_level:
                # logger.warning(f"Timer '{key}' not found in hierarchy")
                return None
            current_level = current_level[k]
            
        if keys[-1] in current_level:
            end_time = time.time()
            end_memory = self._process.memory_info().rss / 1024 / 1024  # Memory in MB
            current_level[keys[-1]]["end"] = end_time
            current_level[keys[-1]]["end_memory"] = end_memory
            
            elapsed = end_time - current_level[keys[-1]]["start"]
            memory_used = end_memory - current_level[keys[-1]]["start_memory"]
            
            # Update accumulated metrics
            current_level[keys[-1]]["total"] = current_level[keys[-1]].get("total", 0) + elapsed
            current_level[keys[-1]]["total_memory"] = current_level[keys[-1]].get("total_memory", 0) + memory_used
            
            metrics = {
                "elapsed": elapsed,
                "memory_used": memory_used,
                "end_memory": end_memory
            }
            # logger.info(f"Timer '{key}' stopped. Metrics: {metrics}")
            return metrics
        return None

    @contextmanager
    def track(self, key: str):
        """Context manager for timing a block of code.
        
        Args:
            key: Dot-separated string representing the timing hierarchy
            
        Example:
            with profiler.track("retrieval.chunking"):
                do_something()
        """
        try:
            self.start(key)
            yield
        finally:
            elapsed = self.stop(key)
            # logger.debug(f"Context manager for '{key}' completed. Time: {elapsed:.4f}s")

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
        self.timings = {}
        self._active_timers.clear()
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
