"""Performance optimization utilities for MELD experiments.

This module provides utilities for:
- Parallel data loading and preprocessing
- Caching mechanisms
- Performance monitoring
- Resource usage tracking
"""

from __future__ import annotations

import functools
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import numpy as np
from tqdm import tqdm

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

LOGGER = logging.getLogger("performance")

T = TypeVar('T')


class PerformanceMonitor:
    """Monitor performance metrics during experiment execution."""
    
    def __init__(self):
        self.start_time = None
        self.checkpoints: List[Dict[str, Any]] = []
        self.peak_memory = 0.0
        self.peak_gpu_memory = 0.0
    
    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            self.peak_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    def checkpoint(self, name: str, **kwargs):
        """Record a checkpoint."""
        elapsed = time.time() - self.start_time if self.start_time else 0.0
        
        checkpoint = {
            "name": name,
            "elapsed_time": elapsed,
            **kwargs,
        }
        
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            checkpoint["memory_mb"] = memory_mb
            self.peak_memory = max(self.peak_memory, memory_mb)
        
        # GPU memory (if available)
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                checkpoint["gpu_memory_mb"] = gpu_memory_mb
                self.peak_gpu_memory = max(self.peak_gpu_memory, gpu_memory_mb)
        except ImportError:
            pass
        
        self.checkpoints.append(checkpoint)
        return checkpoint
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        total_time = time.time() - self.start_time if self.start_time else 0.0
        return {
            "total_time_seconds": total_time,
            "total_time_minutes": total_time / 60,
            "peak_memory_mb": self.peak_memory,
            "peak_gpu_memory_mb": self.peak_gpu_memory,
            "checkpoints": self.checkpoints,
        }


def parallel_map(
    func: Callable[[T], Any],
    items: List[T],
    n_jobs: int = -1,
    desc: str = "Processing",
    progress: bool = True,
    **kwargs,
) -> List[Any]:
    """Parallel map function with progress bar.
    
    Args:
        func: Function to apply to each item
        items: List of items to process
        n_jobs: Number of parallel jobs (-1 for all CPUs)
        desc: Progress bar description
        progress: Show progress bar
        **kwargs: Additional arguments to pass to func
    
    Returns:
        List of results
    """
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    n_jobs = max(1, min(n_jobs, len(items)))
    
    if n_jobs == 1:
        # Sequential processing
        iterator = items
        if progress:
            iterator = tqdm(iterator, desc=desc, total=len(items))
        return [func(item, **kwargs) for item in iterator]
    
    # Parallel processing
    with mp.Pool(n_jobs) as pool:
        if progress:
            results = list(tqdm(
                pool.imap(functools.partial(func, **kwargs), items),
                desc=desc,
                total=len(items),
            ))
        else:
            results = pool.map(functools.partial(func, **kwargs), items)
    
    return results


# Import safe cache as replacement for DiskCache
from .safe_cache import SafeJSONCache as DiskCache


def cached_disk(cache_dir: str = "./cache", key_func: Optional[Callable] = None):
    """Decorator for disk caching."""
    cache = DiskCache(cache_dir)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                import hashlib
                import pickle
                key_data = pickle.dumps((args, kwargs))
                cache_key = hashlib.md5(key_data).hexdigest()
            
            # Try to get from cache
            cached = cache.get(cache_key)
            if cached is not None:
                LOGGER.debug(f"Cache hit for {func.__name__}")
                return cached
            
            # Compute and cache
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            return result
        
        return wrapper
    return decorator


def batch_process(
    items: List[T],
    batch_size: int,
    func: Callable[[List[T]], List[Any]],
    progress: bool = True,
    desc: str = "Processing",
) -> List[Any]:
    """Process items in batches.
    
    Args:
        items: List of items to process
        batch_size: Batch size
        func: Function that processes a batch and returns list of results
        progress: Show progress bar
        desc: Progress bar description
    
    Returns:
        List of all results
    """
    results = []
    n_batches = (len(items) + batch_size - 1) // batch_size
    
    iterator = range(0, len(items), batch_size)
    if progress:
        iterator = tqdm(iterator, desc=desc, total=n_batches)
    
    for start_idx in iterator:
        batch = items[start_idx:start_idx + batch_size]
        batch_results = func(batch)
        results.extend(batch_results)
    
    return results


def optimize_batch_size(
    func: Callable[[int], float],
    min_batch_size: int = 1,
    max_batch_size: int = 1024,
    target_time_per_batch: float = 1.0,
) -> int:
    """Find optimal batch size for a function.
    
    Args:
        func: Function that takes batch_size and returns time taken
        min_batch_size: Minimum batch size to try
        max_batch_size: Maximum batch size to try
        target_time_per_batch: Target time per batch in seconds
    
    Returns:
        Optimal batch size
    """
    best_batch_size = min_batch_size
    best_efficiency = 0.0
    
    for batch_size in [min_batch_size, max_batch_size // 4, max_batch_size // 2, max_batch_size]:
        try:
            time_taken = func(batch_size)
            efficiency = batch_size / time_taken
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_batch_size = batch_size
        except Exception as e:
            LOGGER.warning(f"Failed to test batch_size {batch_size}: {e}")
            continue
    
    return best_batch_size

