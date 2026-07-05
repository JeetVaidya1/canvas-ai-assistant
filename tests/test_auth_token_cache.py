"""Bounded token cache in auth.py (LRU eviction + TTL expiry on access)."""
import pytest

import auth


@pytest.fixture(autouse=True)
def _clean_cache():
    auth._cache.clear()
    yield
    auth._cache.clear()


@pytest.mark.unit
def test_cache_hit_within_ttl():
    auth._cache_put("tok", {"id": "u1"}, expires_at=1000.0)
    assert auth._cache_get("tok", now=999.0) == {"id": "u1"}


@pytest.mark.unit
def test_expired_entry_is_removed_on_access():
    auth._cache_put("tok", {"id": "u1"}, expires_at=1000.0)
    assert auth._cache_get("tok", now=1000.0) is None
    assert "tok" not in auth._cache


@pytest.mark.unit
def test_cache_size_is_bounded():
    for i in range(auth._CACHE_MAX_SIZE + 50):
        auth._cache_put(f"tok-{i}", {"id": f"u{i}"}, expires_at=10_000.0)
    assert len(auth._cache) == auth._CACHE_MAX_SIZE
    # Oldest entries were evicted first (FIFO/LRU order).
    assert "tok-0" not in auth._cache
    assert f"tok-{auth._CACHE_MAX_SIZE + 49}" in auth._cache


@pytest.mark.unit
def test_recently_used_entry_survives_eviction():
    for i in range(auth._CACHE_MAX_SIZE):
        auth._cache_put(f"tok-{i}", {"id": f"u{i}"}, expires_at=10_000.0)
    # Touch the oldest entry so it becomes most-recently-used...
    assert auth._cache_get("tok-0", now=0.0) is not None
    # ...then overflow the cache by one: tok-1 (now oldest) should go, not tok-0.
    auth._cache_put("tok-new", {"id": "new"}, expires_at=10_000.0)
    assert "tok-0" in auth._cache
    assert "tok-1" not in auth._cache
