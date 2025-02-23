# core/utils/profiler.py (Conceptual)

import time
from typing import Dict, Any, Optional
from contextlib import contextmanager

from core.logger.logger import get_logger

logger = get_logger(__name__)

class Profiler:
    def __init__(self):
        self.timings = {}
        self._active_timers = set()  # Track currently active timers
        logger.info("Initialized new Profiler instance")

    def start(self, key: str) -> None:
        """Start timing for a specific key.
        
        Args:
            key: Dot-separated string representing the timing hierarchy
        """
        if key in self._active_timers:
            error_msg = f"Timer '{key}' is already running"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        self._active_timers.add(key)
        keys = key.split(".")
        current_level = self.timings
        
        for k in keys[:-1]:
            if k not in current_level:
                current_level[k] = {}
            current_level = current_level[k]
        
        start_time = time.time()
        current_level[keys[-1]] = {
            "start": start_time,
            "count": current_level.get(keys[-1], {}).get("count", 0) + 1
        }
        logger.debug(f"Started timer for '{key}' at {start_time}")

    def stop(self, key: str) -> Optional[float]:
        """Stop timing for a specific key and return the elapsed time.
        
        Args:
            key: Dot-separated string representing the timing hierarchy
            
        Returns:
            Elapsed time in seconds, or None if the timer wasn't started
        """
        if key not in self._active_timers:
            error_msg = f"Timer '{key}' was not started"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        self._active_timers.remove(key)
        keys = key.split(".")
        current_level = self.timings
        
        for k in keys[:-1]:
            if k not in current_level:
                logger.warning(f"Timer '{key}' not found in hierarchy")
                return None
            current_level = current_level[k]
            
        if keys[-1] in current_level:
            end_time = time.time()
            current_level[keys[-1]]["end"] = end_time
            elapsed = end_time - current_level[keys[-1]]["start"]
            
            # Update accumulated time
            current_level[keys[-1]]["total"] = current_level[keys[-1]].get("total", 0) + elapsed
            
            logger.info(f"Timer '{key}' stopped. Elapsed time: {elapsed:.4f}s")
            return elapsed
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
            logger.debug(f"Context manager for '{key}' completed. Time: {elapsed:.4f}s")

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
                logger.debug(f"Executing tracked function '{func.__name__}' with key '{key}'")
                with self.track(key):
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    def reset(self) -> None:
        """Reset all timings and active timers."""
        self.timings = {}
        self._active_timers.clear()
        logger.info("Profiler reset - all timings cleared")

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
                                "avg_duration": v["total"] / v["count"]
                            }
                        else:
                            result[new_key] = v["total"]
                    else:
                        _flatten(v, new_key, result)
            return result

        metrics = _flatten(self.timings)
        logger.debug(f"Retrieved metrics: {metrics}")
        return metrics

    def get_active_timers(self) -> set:
        """Get the set of currently active timer keys."""
        return self._active_timers.copy()
