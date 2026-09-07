import time

from src.devices.idempotency import TTLMemoryIdempotencyStore


def test_ttl_store_namespaces_by_device():
    store = TTLMemoryIdempotencyStore(ttl_seconds=60)
    assert store.claim("cam-01", "req-1") is True
    assert store.claim("cam-01", "req-1") is False
    assert store.claim("cam-02", "req-1") is True


def test_ttl_store_expires_requests():
    store = TTLMemoryIdempotencyStore(ttl_seconds=0.01)
    assert store.claim("cam-01", "req-1") is True
    time.sleep(0.03)
    assert store.claim("cam-01", "req-1") is True


def test_ttl_store_is_bounded():
    store = TTLMemoryIdempotencyStore(ttl_seconds=60, max_entries=2)
    assert store.claim("cam-01", "a") is True
    assert store.claim("cam-01", "b") is True
    assert store.claim("cam-01", "c") is True
    assert len(store._entries) == 2
