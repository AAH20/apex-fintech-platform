"""Tests for the Caching Engine."""
import time
import pytest
from src.caching.engine import CachingEngine


@pytest.fixture
def engine():
    """Create a fresh caching engine with a test namespace."""
    eng = CachingEngine(namespace="test", host="localhost", port=6379, db=9)
    eng.flush()
    yield eng
    eng.flush()


@pytest.fixture
def memory_engine():
    """Create an in-memory-only engine (no Redis)."""
    eng = CachingEngine(namespace="memtest", use_redis=False)
    yield eng


# --- Basic Operations ---

class TestBasicOperations:
    def test_set_and_get(self, engine):
        engine.set("key1", "value1")
        assert engine.get("key1") == "value1"

    def test_get_missing_key_returns_none(self, engine):
        assert engine.get("nonexistent") is None

    def test_get_with_default(self, engine):
        assert engine.get("nonexistent", default="fallback") == "fallback"

    def test_delete(self, engine):
        engine.set("key1", "value1")
        engine.delete("key1")
        assert engine.get("key1") is None

    def test_exists(self, engine):
        engine.set("key1", "value1")
        assert engine.exists("key1") is True
        assert engine.exists("nonexistent") is False

    def test_clear_all(self, engine):
        engine.set("a", 1)
        engine.set("b", 2)
        engine.clear()
        assert engine.get("a") is None
        assert engine.get("b") is None


# --- TTL / Expiration ---

class TestTTL:
    def test_set_with_ttl(self, engine):
        engine.set("key1", "value1", ttl=1)
        assert engine.get("key1") == "value1"
        time.sleep(1.1)
        assert engine.get("key1") is None

    def test_ttl_remaining(self, engine):
        engine.set("key1", "value1", ttl=10)
        remaining = engine.ttl("key1")
        assert 0 < remaining <= 10

    def test_ttl_no_expiration(self, engine):
        engine.set("key1", "value1")
        assert engine.ttl("key1") == -1  # -1 means no expiry in Redis

    def test_overwrite_resets_ttl(self, engine):
        engine.set("key1", "value1", ttl=1)
        engine.set("key1", "value2", ttl=100)
        assert engine.get("key1") == "value2"
        assert engine.ttl("key1") > 1


# --- Serialization ---

class TestSerialization:
    def test_dict_roundtrip(self, engine):
        data = {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}
        engine.set("dict_key", data)
        assert engine.get("dict_key") == data

    def test_list_roundtrip(self, engine):
        data = [1, "two", 3.0, None, True]
        engine.set("list_key", data)
        assert engine.get("list_key") == data

    def test_numeric_roundtrip(self, engine):
        engine.set("int_key", 42)
        engine.set("float_key", 3.14)
        assert engine.get("int_key") == 42
        assert engine.get("float_key") == 3.14


# --- get_or_set (compute-on-miss) ---

class TestGetOrSet:
    def test_get_or_set_computes_on_miss(self, engine):
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return "computed_value"

        result = engine.get_or_set("key1", compute)
        assert result == "computed_value"
        assert call_count == 1

    def test_get_or_set_uses_cache_on_hit(self, engine):
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return "computed_value"

        engine.get_or_set("key1", compute)
        result = engine.get_or_set("key1", compute)
        assert result == "computed_value"
        assert call_count == 1  # compute only called once

    def test_get_or_set_with_ttl(self, engine):
        engine.get_or_set("key1", lambda: "val", ttl=1)
        assert engine.get("key1") == "val"
        time.sleep(1.1)
        assert engine.get("key1") is None


# --- Invalidation ---

class TestInvalidation:
    def test_invalidate_prefix(self, engine):
        engine.set("user:1", "alice")
        engine.set("user:2", "bob")
        engine.set("order:1", "order_data")
        engine.invalidate_prefix("user:")
        assert engine.get("user:1") is None
        assert engine.get("user:2") is None
        assert engine.get("order:1") == "order_data"

    def test_invalidate_pattern(self, engine):
        engine.set("cache:a", 1)
        engine.set("cache:b", 2)
        engine.set("other:c", 3)
        engine.invalidate_pattern("cache:*")
        assert engine.get("cache:a") is None
        assert engine.get("cache:b") is None
        assert engine.get("other:c") == 3


# --- Statistics ---

class TestStatistics:
    def test_hit_count(self, engine):
        engine.set("key1", "val")
        engine.get("key1")
        engine.get("key1")
        stats = engine.stats()
        assert stats["hits"] >= 2

    def test_miss_count(self, engine):
        engine.get("nonexistent")
        engine.get("nonexistent2")
        stats = engine.stats()
        assert stats["misses"] >= 2

    def test_hit_ratio(self, engine):
        engine.set("key1", "val")
        engine.get("key1")  # hit
        engine.get("missing")  # miss
        stats = engine.stats()
        assert stats["hit_ratio"] == pytest.approx(0.5, abs=0.01)


# --- Decorator / Memoization ---

class TestDecorator:
    def test_cached_decorator(self, engine):
        call_count = 0

        @engine.cached("expensive", ttl=60)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        assert expensive_function(5) == 10
        assert expensive_function(5) == 10
        assert call_count == 1

    def test_cached_decorator_different_args(self, engine):
        call_count = 0

        @engine.cached("expensive", ttl=60)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        assert expensive_function(5) == 10
        assert expensive_function(10) == 20
        assert call_count == 2


# --- In-Memory Fallback ---

class TestMemoryEngine:
    def test_memory_set_get(self, memory_engine):
        memory_engine.set("key1", "value1")
        assert memory_engine.get("key1") == "value1"

    def test_memory_ttl(self, memory_engine):
        memory_engine.set("key1", "value1", ttl=1)
        assert memory_engine.get("key1") == "value1"
        time.sleep(1.1)
        assert memory_engine.get("key1") is None

    def test_memory_get_or_set(self, memory_engine):
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return "computed"

        memory_engine.get_or_set("key1", compute)
        memory_engine.get_or_set("key1", compute)
        assert call_count == 1

    def test_memory_lru_eviction(self):
        eng = CachingEngine(namespace="lru", use_redis=False, max_memory_items=3)
        eng.set("a", 1)
        eng.set("b", 2)
        eng.set("c", 3)
        eng.set("d", 4)  # should evict "a"
        assert eng.get("a") is None
        assert eng.get("d") == 4

    def test_memory_delete(self, memory_engine):
        memory_engine.set("key1", "val")
        memory_engine.delete("key1")
        assert memory_engine.get("key1") is None

    def test_memory_clear(self, memory_engine):
        memory_engine.set("a", 1)
        memory_engine.set("b", 2)
        memory_engine.clear()
        assert memory_engine.get("a") is None
        assert memory_engine.get("b") is None


# --- Redis Failure Fallback ---

class TestRedisFailureFallback:
    def test_fallback_to_memory_on_connection_error(self):
        eng = CachingEngine(namespace="fallback", host="localhost", port=6399, db=9)
        # Should not raise even if Redis is unavailable
        eng.set("key1", "value1")
        assert eng.get("key1") == "value1"

    def test_redis_error_graceful_degradation(self):
        eng = CachingEngine(namespace="graceful", host="localhost", port=6399, db=9)
        # get on missing key should return default, not raise
        assert eng.get("missing", default="safe") == "safe"


# --- Counter Operations ---

class TestCounter:
    def test_increment(self, engine):
        engine.delete("counter")
        engine.increment("counter")
        engine.increment("counter")
        engine.increment("counter")
        assert engine.get("counter") == 3

    def test_increment_with_amount(self, engine):
        engine.delete("counter")
        engine.increment("counter", amount=5)
        assert engine.get("counter") == 5

    def test_decrement(self, engine):
        engine.set("counter", 10)
        engine.decrement("counter")
        assert engine.get("counter") == 9


# --- Batch Operations ---

class TestBatchOperations:
    def test_mset(self, engine):
        engine.mset({"a": 1, "b": 2, "c": 3})
        assert engine.get("a") == 1
        assert engine.get("b") == 2
        assert engine.get("c") == 3

    def test_mget(self, engine):
        engine.set("a", 1)
        engine.set("b", 2)
        engine.set("c", 3)
        result = engine.mget(["a", "b", "c"])
        assert result == [1, 2, 3]

    def test_mget_with_missing(self, engine):
        engine.set("a", 1)
        result = engine.mget(["a", "missing"])
        assert result == [1, None]


# --- Key Namespacing ---

class TestNamespacing:
    def test_keys_are_namespaced(self, engine):
        engine.set("mykey", "val")
        # The raw Redis key should include the namespace
        raw_keys = engine._raw_keys("*")
        assert len(raw_keys) > 0
        assert any("test:mykey" in k for k in raw_keys)

    def test_namespace_isolation(self):
        eng1 = CachingEngine(namespace="ns1", host="localhost", port=6379, db=9)
        eng2 = CachingEngine(namespace="ns2", host="localhost", port=6379, db=9)
        eng1.flush()
        eng2.flush()
        eng1.set("key", "value1")
        eng2.set("key", "value2")
        assert eng1.get("key") == "value1"
        assert eng2.get("key") == "value2"
        eng1.flush()
        eng2.flush()
