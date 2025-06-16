"""
Performance optimization and caching system for Cognito.
"""

import os
import time
import pickle
import hashlib
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Callable, Union
from dataclasses import dataclass
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import get_config


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    timestamp: float
    ttl: float
    hit_count: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return time.time() > self.timestamp + self.ttl
    
    def touch(self):
        """Update hit count and access time."""
        self.hit_count += 1


class MemoryCache:
    """In-memory cache with TTL and size limits."""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            entry = self.cache.get(key)
            if entry is None:
                self.stats['misses'] += 1
                return None
            
            if entry.is_expired():
                del self.cache[key]
                self.stats['misses'] += 1
                return None
            
            entry.touch()
            self.stats['hits'] += 1
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set value in cache."""
        with self.lock:
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl=ttl or self.default_ttl
            )
    
    def delete(self, key: str):
        """Delete key from cache."""
        with self.lock:
            self.cache.pop(key, None)
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.stats = {'hits': 0, 'misses': 0, 'evictions': 0}
    
    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self.cache:
            return
        
        # Find entry with lowest hit count and oldest timestamp
        lru_key = min(
            self.cache.keys(),
            key=lambda k: (self.cache[k].hit_count, self.cache[k].timestamp)
        )
        del self.cache[lru_key]
        self.stats['evictions'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                **self.stats
            }


class FileCache:
    """Persistent file-based cache."""
    
    def __init__(self, cache_dir: str, max_size_mb: int = 100):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self.lock = threading.Lock()
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from file cache."""
        file_path = self._get_file_path(key)
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'rb') as f:
                entry_data = pickle.load(f)
                entry = CacheEntry(**entry_data)
                
                if entry.is_expired():
                    file_path.unlink(missing_ok=True)
                    return None
                
                return entry.value
        except Exception:
            file_path.unlink(missing_ok=True)
            return None
    
    def set(self, key: str, value: Any, ttl: float = 3600):
        """Set value in file cache."""
        with self.lock:
            self._cleanup_if_needed()
            
            file_path = self._get_file_path(key)
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl=ttl
            )
            
            try:
                with open(file_path, 'wb') as f:
                    pickle.dump(entry.__dict__, f)
            except Exception:
                pass  # Ignore cache write errors
    
    def delete(self, key: str):
        """Delete key from file cache."""
        file_path = self._get_file_path(key)
        file_path.unlink(missing_ok=True)
    
    def clear(self):
        """Clear all cache files."""
        for file_path in self.cache_dir.glob("*.cache"):
            file_path.unlink(missing_ok=True)
    
    def _cleanup_if_needed(self):
        """Clean up expired entries and enforce size limits."""
        cache_files = list(self.cache_dir.glob("*.cache"))
        
        # Remove expired files
        for file_path in cache_files:
            try:
                with open(file_path, 'rb') as f:
                    entry_data = pickle.load(f)
                    entry = CacheEntry(**entry_data)
                    if entry.is_expired():
                        file_path.unlink()
            except Exception:
                file_path.unlink(missing_ok=True)
        
        # Check total size
        total_size = sum(f.stat().st_size for f in cache_files if f.exists())
        max_size = self.max_size_mb * 1024 * 1024
        
        if total_size > max_size:
            # Remove oldest files
            cache_files.sort(key=lambda f: f.stat().st_mtime)
            for file_path in cache_files:
                if not file_path.exists():
                    continue
                file_path.unlink()
                total_size -= file_path.stat().st_size
                if total_size <= max_size * 0.8:  # Leave some headroom
                    break


class CacheManager:
    """Unified cache manager."""
    
    def __init__(self):
        config = get_config()
        self.memory_cache = MemoryCache(max_size=1000, default_ttl=3600)
        self.file_cache = FileCache(
            cache_dir=config.database.cache_dir,
            max_size_mb=100
        )
        self.enabled = config.performance.cache_enabled
    
    def get(self, key: str, use_file_cache: bool = True) -> Optional[Any]:
        """Get value from cache (memory first, then file)."""
        if not self.enabled:
            return None
        
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try file cache
        if use_file_cache:
            value = self.file_cache.get(key)
            if value is not None:
                # Store in memory cache for faster access
                self.memory_cache.set(key, value)
                return value
        
        return None
    
    def set(self, key: str, value: Any, ttl: float = 3600, use_file_cache: bool = True):
        """Set value in cache."""
        if not self.enabled:
            return
        
        self.memory_cache.set(key, value, ttl)
        if use_file_cache:
            self.file_cache.set(key, value, ttl)
    
    def delete(self, key: str):
        """Delete key from all caches."""
        self.memory_cache.delete(key)
        self.file_cache.delete(key)
    
    def clear(self):
        """Clear all caches."""
        self.memory_cache.clear()
        self.file_cache.clear()


class AsyncProcessor:
    """Process tasks asynchronously for better performance."""
    
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_tasks = {}
    
    def submit_analysis(self, code: str, language: str, analysis_id: str) -> str:
        """Submit code analysis task."""
        future = self.executor.submit(self._analyze_code, code, language)
        self.active_tasks[analysis_id] = future
        return analysis_id
    
    def get_result(self, analysis_id: str, timeout: float = 30) -> Optional[Dict[str, Any]]:
        """Get analysis result."""
        future = self.active_tasks.get(analysis_id)
        if not future:
            return None
        
        try:
            result = future.result(timeout=timeout)
            self.active_tasks.pop(analysis_id, None)
            return result
        except Exception as e:
            self.active_tasks.pop(analysis_id, None)
            return {"error": str(e)}
    
    def _analyze_code(self, code: str, language: str) -> Dict[str, Any]:
        """Perform code analysis (placeholder)."""
        from analyzer import analyze_code
        return analyze_code(code, language=language)


def cache_result(ttl: float = 3600, use_file_cache: bool = True, key_func: Optional[Callable] = None):
    """Decorator to cache function results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"
            
            # Try to get from cache
            cache_manager = get_cache_manager()
            result = cache_manager.get(cache_key, use_file_cache)
            if result is not None:
                return result
            
            # Compute result and cache it
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl, use_file_cache)
            return result
        
        return wrapper
    return decorator


def performance_monitor(func):
    """Decorator to monitor function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            execution_time = time.time() - start_time
            if execution_time > 5.0:  # Log slow operations
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Slow operation: {func.__name__} took {execution_time:.2f}s")
    
    return wrapper


class ChunkedProcessor:
    """Process large files in chunks to avoid memory issues."""
    
    def __init__(self, chunk_size: int = 1000000):  # 1MB chunks
        self.chunk_size = chunk_size
    
    def process_large_file(self, file_path: str, processor_func: Callable) -> list:
        """Process large file in chunks."""
        results = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            chunk = ""
            line_count = 0
            
            for line in f:
                chunk += line
                line_count += 1
                
                if len(chunk) >= self.chunk_size or line_count >= 1000:
                    if chunk.strip():
                        result = processor_func(chunk)
                        results.append(result)
                    
                    chunk = ""
                    line_count = 0
            
            # Process remaining chunk
            if chunk.strip():
                result = processor_func(chunk)
                results.append(result)
        
        return results
    
    def process_code_chunks(self, code: str, max_chunk_size: int = 50000) -> list:
        """Split large code into analyzable chunks."""
        if len(code) <= max_chunk_size:
            return [code]
        
        lines = code.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line) + 1  # +1 for newline
            
            if current_size + line_size > max_chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks


# Global instances
_cache_manager: Optional[CacheManager] = None
_async_processor: Optional[AsyncProcessor] = None
_chunked_processor: Optional[ChunkedProcessor] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def get_async_processor() -> AsyncProcessor:
    """Get global async processor."""
    global _async_processor
    if _async_processor is None:
        _async_processor = AsyncProcessor()
    return _async_processor


def get_chunked_processor() -> ChunkedProcessor:
    """Get global chunked processor."""
    global _chunked_processor
    if _chunked_processor is None:
        _chunked_processor = ChunkedProcessor()
    return _chunked_processor