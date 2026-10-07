"""Caching Engine with Redis backend and in-memory fallback.

Provides a unified caching interface with:
- Redis integration for distributed caching
- In-memory LRU cache as fallback when Redis is unavailable
- TTL support, batch operations, counters, and memoization decorator
- Hit/miss statistics for monitoring
"""
import json
import time
import threading
from collections import OrderedDict
from typing import Any, Callable, Optional
from functools import wraps

try:
    import redis
except ImportError:
    redis = None


class CachingEngine:
    """Unified caching engine with Redis backend and in-memory fallback.

    Usage:
        engine = CachingEngine(namespace="myapp")
        engine.set("key", "value", ttl=300)
        value = engine.get("key")

        @engine.cached("expensive_fn", ttl=60)
        def expensive_fn(x):
            return x * 2
    """

    def __init__(
        self,
        namespace: str = "default",
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        use_redis: bool = True,
        max_memory_items: int = 1000,
        serializer: Optional[Callable] = None,
        deserializer: Optional[Callable] = None,
    ):
        self.namespace = namespace
        self._max_memory_items = max_memory_items
        self._memory_cache: OrderedDict = OrderedDict()
        self._memory_ttl: dict = {}
        self._lock = threading.Lock()

        # Statistics
        self._hits = 0
        self._misses = 0

        # Serializer (default: JSON)
        self._serializer = serializer or json.dumps
        self._deserializer = deserializer or json.loads

        # Redis connection
        self._redis: Optional[Any] = None
        self._redis_available = False

        if use_redis and redis is not None:
            try:
                self._redis = redis.Redis(
                    host=host, port=port, db=db, socket_connect_timeout=2, decode_responses=False
                )
                self._redis.ping()
                self._redis_available = True
            except Exception:
                self._redis = None
                self._redis_available = False

    # --- Key Namespacing ---

    def _make_key(self, key: str) -> str:
        """Create a namespaced key."""
        return f"{self.namespace}:{key}"

    def _raw_keys(self, pattern: str) -> list:
        """Get raw keys matching a pattern (for testing/debugging)."""
        if self._redis_available:
            full_pattern = self._make_key(pattern)
            return [k.decode() if isinstance(k, bytes) else k for k in self._redis.keys(full_pattern)]
        return []

    # --- Serialization ---

    def _serialize(self, value: Any) -> str:
        return self._serializer(value)

    def _deserialize(self, value: str) -> Any:
        if value is None:
            return None
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return self._deserializer(value)

    # --- Core Operations ---

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store a value with optional TTL (seconds)."""
        serialized = self._serialize(value)
        namespaced = self._make_key(key)

        if self._redis_available:
            try:
                if ttl is not None:
                    self._redis.set(namespaced, serialized, ex=ttl)
                else:
                    self._redis.set(namespaced, serialized)
                return True
            except Exception:
                self._redis_available = False

        # Fallback to memory
        with self._lock:
            self._memory_cache[namespaced] = serialized
            if ttl is not None:
                self._memory_ttl[namespaced] = time.time() + ttl
            else:
                self._memory_ttl.pop(namespaced, None)
            self._evict_if_needed()
        return True

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value, returning default if not found."""
        namespaced = self._make_key(key)

        if self._redis_available:
            try:
                value = self._redis.get(namespaced)
                if value is not None:
                    self._hits += 1
                    return self._deserialize(value)
                self._misses += 1
                return default
            except Exception:
                self._redis_available = False

        # Fallback to memory
        with self._lock:
            if namespaced in self._memory_cache:
                # Check TTL
                if self._is_memory_expired(namespaced):
                    self._memory_cache.pop(namespaced, None)
                    self._memory_ttl.pop(namespaced, None)
                    self._misses += 1
                    return default
                self._hits += 1
                self._memory_cache.move_to_end(namespaced)
                return self._deserialize(self._memory_cache[namespaced])
            self._misses += 1
            return default

    def delete(self, key: str) -> bool:
        """Delete a key."""
        namespaced = self._make_key(key)

        if self._redis_available:
            try:
                return bool(self._redis.delete(namespaced))
            except Exception:
                self._redis_available = False

        with self._lock:
            existed = namespaced in self._memory_cache
            self._memory_cache.pop(namespaced, None)
            self._memory_ttl.pop(namespaced, None)
            return existed

    def exists(self, key: str) -> bool:
        """Check if a key exists."""
        namespaced = self._make_key(key)

        if self._redis_available:
            try:
                return bool(self._redis.exists(namespaced))
            except Exception:
                self._redis_available = False

        with self._lock:
            if namespaced in self._memory_cache:
                if self._is_memory_expired(namespaced):
                    self._memory_cache.pop(namespaced, None)
                    self._memory_ttl.pop(namespaced, None)
                    return False
                return True
            return False

    def clear(self) -> bool:
        """Clear all keys in this namespace."""
        if self._redis_available:
            try:
                pattern = self._make_key("*")
                keys = self._redis.keys(pattern)
                if keys:
                    self._redis.delete(*keys)
                return True
            except Exception:
                self._redis_available = False

        with self._lock:
            prefix = self._make_key("")
            keys_to_remove = [k for k in self._memory_cache if k.startswith(prefix)]
            for k in keys_to_remove:
                self._memory_cache.pop(k, None)
                self._memory_ttl.pop(k, None)
        return True

    def flush(self) -> bool:
        """Flush the entire Redis database (use with caution)."""
        if self._redis_available:
            try:
                self._redis.flushdb()
                return True
            except Exception:
                self._redis_available = False
        with self._lock:
            self._memory_cache.clear()
            self._memory_ttl.clear()
        return True

    # --- TTL ---

    def ttl(self, key: str) -> int:
        """Get remaining TTL in seconds. -1 if no expiry, -2 if key doesn't exist."""
        namespaced = self._make_key(key)

        if self._redis_available:
            try:
                return self._redis.ttl(namespaced)
            except Exception:
                self._redis_available = False

        with self._lock:
            if namespaced not in self._memory_cache:
                return -2
            if namespaced not in self._memory_ttl:
                return -1
            remaining = self._memory_ttl[namespaced] - time.time()
            if remaining <= 0:
                self._memory_cache.pop(namespaced, None)
                self._memory_ttl.pop(namespaced, None)
                return -2
            return int(remaining)

    # --- get_or_set ---

    def get_or_set(self, key: str, compute: Callable, ttl: Optional[int] = None) -> Any:
        """Get from cache or compute and store on miss."""
        value = self.get(key)
        if value is not None:
            return value
        value = compute()
        if value is not None:
            self.set(key, value, ttl=ttl)
        return value

    # --- Invalidation ---

    def invalidate_prefix(self, prefix: str) -> int:
        """Invalidate all keys matching a prefix. Returns count deleted."""
        pattern = f"{prefix}*"
        return self.invalidate_pattern(pattern)

    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching a glob pattern. Returns count deleted."""
        if self._redis_available:
            try:
                full_pattern = self._make_key(pattern)
                keys = self._redis.keys(full_pattern)
                if keys:
                    self._redis.delete(*keys)
                return len(keys)
            except Exception:
                self._redis_available = False

        with self._lock:
            import fnmatch
            full_pattern = self._make_key(pattern)
            keys_to_remove = [
                k for k in list(self._memory_cache.keys())
                if fnmatch.fnmatch(k, full_pattern)
            ]
            for k in keys_to_remove:
                self._memory_cache.pop(k, None)
                self._memory_ttl.pop(k, None)
            return len(keys_to_remove)

    # --- Statistics ---

    def stats(self) -> dict:
        """Return cache statistics."""
        total = self._hits + self._misses
        hit_ratio = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_ratio": round(hit_ratio, 4),
            "redis_available": self._redis_available,
            "memory_items": len(self._memory_cache),
        }

    def reset_stats(self) -> None:
        """Reset hit/miss counters."""
        self._hits = 0
        self._misses = 0

    # --- Decorator ---

    def cached(self, key_prefix: str, ttl: Optional[int] = None):
        """Decorator to cache function results.

        Usage:
            @engine.cached("my_func", ttl=60)
            def my_func(x, y):
                return expensive_computation(x, y)
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Build cache key from function name + arguments
                cache_key = f"{key_prefix}:{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
                value = self.get(cache_key)
                if value is not None:
                    return value
                value = func(*args, **kwargs)
                if value is not None:
                    self.set(cache_key, value, ttl=ttl)
                return value
            return wrapper
        return decorator

    # --- Counter Operations ---

    def increment(self, key: str, amount: int = 1) -> int:
        """Atomically increment a counter."""
        namespaced = self._make_key(key)

        if self._redis_available:
            try:
                return self._redis.incrby(namespaced, amount)
            except Exception:
                self._redis_available = False

        with self._lock:
            current = 0
            if namespaced in self._memory_cache:
                if not self._is_memory_expired(namespaced):
                    current = int(self._deserialize(self._memory_cache[namespaced]))
            current += amount
            self._memory_cache[namespaced] = self._serialize(current)
            self._memory_cache.move_to_end(namespaced)
            return current

    def decrement(self, key: str, amount: int = 1) -> int:
        """Atomically decrement a counter."""
        return self.increment(key, amount=-amount)

    # --- Batch Operations ---

    def mset(self, mapping: dict, ttl: Optional[int] = None) -> bool:
        """Set multiple key-value pairs at once."""
        serialized = {self._make_key(k): self._serialize(v) for k, v in mapping.items()}

        if self._redis_available:
            try:
                pipe = self._redis.pipeline()
                pipe.mset(serialized)
                if ttl is not None:
                    for k in serialized:
                        pipe.expire(k, ttl)
                pipe.execute()
                return True
            except Exception:
                self._redis_available = False

        with self._lock:
            for k, v in serialized.items():
                self._memory_cache[k] = v
                if ttl is not None:
                    self._memory_ttl[k] = time.time() + ttl
                else:
                    self._memory_ttl.pop(k, None)
            self._evict_if_needed()
        return True

    def mget(self, keys: list) -> list:
        """Get multiple values at once. Missing keys return None."""
        namespaced_keys = [self._make_key(k) for k in keys]

        if self._redis_available:
            try:
                values = self._redis.mget(namespaced_keys)
                result = []
                for v in values:
                    if v is not None:
                        self._hits += 1
                        result.append(self._deserialize(v))
                    else:
                        self._misses += 1
                        result.append(None)
                return result
            except Exception:
                self._redis_available = False

        result = []
        with self._lock:
            for k in namespaced_keys:
                if k in self._memory_cache:
                    if self._is_memory_expired(k):
                        self._memory_cache.pop(k, None)
                        self._memory_ttl.pop(k, None)
                        self._misses += 1
                        result.append(None)
                    else:
                        self._hits += 1
                        self._memory_cache.move_to_end(k)
                        result.append(self._deserialize(self._memory_cache[k]))
                else:
                    self._misses += 1
                    result.append(None)
        return result

    # --- Memory Cache Internals ---

    def _is_memory_expired(self, key: str) -> bool:
        """Check if a memory-cached key has expired."""
        if key not in self._memory_ttl:
            return False
        return time.time() > self._memory_ttl[key]

    def _evict_if_needed(self) -> None:
        """Evict oldest items if memory cache exceeds max size."""
        while len(self._memory_cache) > self._max_memory_items:
            oldest_key, _ = self._memory_cache.popitem(last=False)
            self._memory_ttl.pop(oldest_key, None)

    # --- Context Manager ---

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self) -> None:
        """Close Redis connection."""
        if self._redis is not None:
            try:
                self._redis.close()
            except Exception:
                pass
