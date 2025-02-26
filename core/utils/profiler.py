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
        
        # Get existing timer data to preserve accumulated values
        existing_timer = self._get_timer_dict(key)
        
        # Create timer data with preserved accumulated totals
        timer_data = {
            "start": start_time,
            "start_memory": start_memory,
            "peak_memory": start_memory,
            "system_memory_start": system_memory_start,
            "count": existing_timer.get("count", 0) + 1,
            "memory_samples": [],
            # Preserve existing accumulated totals
            "total": existing_timer.get("total", 0),
            "total_memory": existing_timer.get("total_memory", 0)
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

    def get_metrics(self, include_counts: bool = False, include_samples: bool = False) -> Dict[str, Any]:
        """Get flattened metrics with optional execution counts.
        
        Args:
            include_counts: If True, include execution counts in the output
            include_samples: If True, include memory samples in the output
            
        Returns:
            Dictionary mapping timer keys to their metrics.
            Memory metrics show the actual memory overhead (always positive).
        """
        metrics = {}
        
        def _flatten(d: Dict, prefix: Optional[str] = None, result: Optional[Dict] = None) -> Dict:
            if result is None:
                result = {}
                
            for k, v in d.items():
                new_key = f"{prefix}.{k}" if prefix else k
                
                if isinstance(v, dict):
                    if "start" in v and "end" in v:
                        # This is a timer entry
                        metric_data = {
                            "duration": v.get("total", 0),  # Use the accumulated total
                        }
                        
                        # Add memory metrics
                        memory_data = {
                            "total_mb": v.get("total_memory", 0),
                            "peak_mb": v.get("peak_memory", 0) - v.get("start_memory", 0),
                            "adjusted_mb": v.get("adjusted_memory_overhead", 0)
                        }
                        
                        # Add tracemalloc data if available
                        if "tracemalloc_diff" in v:
                            memory_data["tracemalloc_mb"] = v["tracemalloc_diff"]
                            
                        if include_counts:
                            count = v.get("count", 1)
                            metric_data["count"] = count
                            metric_data["avg_duration"] = metric_data["duration"] / count
                            memory_data["avg_mb"] = memory_data["total_mb"] / count
                            
                        if include_samples and "memory_samples" in v:
                            memory_data["samples"] = v["memory_samples"]
                            
                        metric_data["memory"] = memory_data
                        result[new_key] = metric_data
                    else:
                        _flatten(v, new_key, result)
            return result

        with self._lock:
            metrics = _flatten(self.timings)
            
        # logger.debug(f"Retrieved metrics: {metrics}")
        return metrics

    def stop(self, key: str) -> Optional[Dict[str, float]]:
        """Stop timing for a specific key and return the elapsed time and memory usage."""
        with self._lock:
            if key not in self._active_timers:
                error_msg = f"Timer '{key}' was not started"
                raise ValueError(error_msg)
                
            self._active_timers.remove(key)
            
        # Signal memory tracking thread to stop if it exists
        self._timer_flags[key] = True
        
        # Get timer data
        timer_dict = self._get_timer_dict(key)
        
        if not timer_dict:
            return None
            
        # Get end metrics
        end_time = time.time()
        end_memory = self._process.memory_info().rss / 1024 / 1024
        
        # Get system memory for differential analysis
        system_memory_end = psutil.virtual_memory()
        
        # Get tracemalloc snapshot if enabled
        tracemalloc_stats = None
        tracemalloc_diff = 0
        if self._use_tracemalloc and "tracemalloc_start" in timer_dict:
            snapshot_end = tracemalloc.take_snapshot()
            start_snapshot = timer_dict["tracemalloc_start"]
            
            # Get top statistics
            tracemalloc_stats = snapshot_end.compare_to(start_snapshot, 'lineno')
            
            # Calculate total memory difference from tracemalloc
            tracemalloc_diff = sum(stat.size_diff for stat in tracemalloc_stats) / 1024 / 1024  # MB
        
        with self._lock:
            # Update timer data
            timer_dict["end"] = end_time
            timer_dict["end_memory"] = end_memory
            
            # Final peak memory check
            if end_memory > timer_dict.get("peak_memory", 0):
                timer_dict["peak_memory"] = end_memory
                timer_dict["peak_time"] = end_time
            
            # Calculate metrics
            elapsed = end_time - timer_dict["start"]
            memory_overhead = timer_dict["peak_memory"] - timer_dict["start_memory"]
            
            # Calculate differential memory usage
            if "system_memory_start" in timer_dict:
                system_memory_start = timer_dict["system_memory_start"]
                
                # Calculate how much system memory changed during execution
                system_available_diff = (system_memory_start["available"] - system_memory_end.available) / 1024 / 1024  # MB
                
                # Adjust process memory overhead by system changes
                adjusted_memory_overhead = max(0, memory_overhead - max(0, system_available_diff))
                timer_dict["adjusted_memory_overhead"] = adjusted_memory_overhead
            else:
                adjusted_memory_overhead = memory_overhead
                timer_dict["adjusted_memory_overhead"] = adjusted_memory_overhead
            
            # Update tracemalloc data if available
            if tracemalloc_stats:
                timer_dict["tracemalloc_stats"] = tracemalloc_stats[:10]  # Top 10 allocations
                timer_dict["tracemalloc_diff"] = tracemalloc_diff
            
            # Update accumulated metrics
            previous_total = timer_dict.get("total", 0)
            timer_dict["total"] = previous_total + elapsed
            
            # Prepare return metrics
            metrics = {
                "elapsed": elapsed,
                "memory_overhead": memory_overhead,
                "adjusted_memory_overhead": adjusted_memory_overhead,
                "peak_memory": timer_dict["peak_memory"],
                "peak_time_offset": timer_dict.get("peak_time", end_time) - timer_dict["start"]
            }
            
            # Add tracemalloc metrics if available
            if tracemalloc_diff:
                metrics["tracemalloc_diff"] = tracemalloc_diff
                
        return metrics

    def analyze_memory_profile(self, key: str) -> Dict[str, Any]:
        """Analyze memory profile for a specific key.
        
        Args:
            key: Dot-separated string representing the timing hierarchy
            
        Returns:
            Dictionary with memory profile analysis
        """
        timer_dict = self._get_timer_dict(key)
        if not timer_dict or "memory_samples" not in timer_dict:
            return {"error": f"No memory samples found for key '{key}'"}
            
        samples = timer_dict.get("memory_samples", [])
        if not samples:
            return {"error": "No memory samples collected"}
            
        # Extract timestamps and memory values
        timestamps = [s[0] for s in samples]
        memory_values = [s[1] for s in samples]
        
        # Normalize timestamps relative to start
        start_time = timer_dict.get("start", timestamps[0])
        relative_times = [t - start_time for t in timestamps]
        
        # Calculate statistics
        if memory_values:
            min_memory = min(memory_values)
            max_memory = max(memory_values)
            avg_memory = sum(memory_values) / len(memory_values)
            
            # Calculate volatility (standard deviation)
            if len(memory_values) > 1:
                mean = avg_memory
                variance = sum((x - mean) ** 2 for x in memory_values) / len(memory_values)
                std_dev = variance ** 0.5
            else:
                std_dev = 0
                
            # Find growth rate
            if len(memory_values) > 1:
                first_val = memory_values[0]
                last_val = memory_values[-1]
                duration = relative_times[-1]
                if duration > 0:
                    growth_rate = (last_val - first_val) / duration  # MB/s
                else:
                    growth_rate = 0
            else:
                growth_rate = 0
                
            # Identify potential memory leaks
            potential_leak = growth_rate > 0.5 and duration > 1.0  # Heuristic
                
            return {
                "samples_count": len(samples),
                "duration": relative_times[-1] if relative_times else 0,
                "min_memory_mb": min_memory,
                "max_memory_mb": max_memory,
                "avg_memory_mb": avg_memory,
                "std_dev_mb": std_dev,
                "growth_rate_mb_per_sec": growth_rate,
                "potential_memory_leak": potential_leak,
                "peak_memory_mb": timer_dict.get("peak_memory", max_memory),
                "adjusted_overhead_mb": timer_dict.get("adjusted_memory_overhead", max_memory - min_memory),
                "tracemalloc_diff_mb": timer_dict.get("tracemalloc_diff", None)
            }
        
        return {"error": "No valid memory samples found"}
