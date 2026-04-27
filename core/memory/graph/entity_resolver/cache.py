"""
LRU+TTL positive cache, bounded negative cache, and per-key in-flight lock management.

All three concerns live here so EntityResolver stays focused on resolution logic.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from typing import Dict, Optional, Tuple

_DEFAULT_NEGATIVE_CACHE_TTL = 300  # seconds — shorter than positive TTL


class ResolutionCache:
    """Thread-safe (asyncio) cache for entity resolution results.

    Positive entries (resolved entity IDs) use an LRU eviction policy with a
    configurable TTL.  Negative entries (entities that could not be resolved)
    use a separate, smaller structure with a shorter TTL so that the system can
    reattempt resolution after the negative TTL expires without hammering backends
    on every call.

    Per-key in-flight locks prevent duplicate concurrent resolution of the same
    entity.  Locks are cleaned up after use to avoid an unbounded memory leak.
    """

    def __init__(
        self,
        max_size: int = 10_000,
        ttl_seconds: int = 3600,
        negative_ttl_seconds: int = _DEFAULT_NEGATIVE_CACHE_TTL,
        negative_max_size: int = 2_000,
    ) -> None:
        self._max_size = max(int(max_size), 1)
        self._ttl_seconds = max(int(ttl_seconds), 1)
        self._negative_ttl = max(int(negative_ttl_seconds), 1)
        self._negative_max_size = max(int(negative_max_size), 1)

        # Positive cache: key → (entity_id, written_at)
        self._cache: OrderedDict[Tuple[str, str], Tuple[str, float]] = OrderedDict()
        self._cache_lock = asyncio.Lock()

        # Negative cache: key → expiry_timestamp
        self._negative: OrderedDict[Tuple[str, str], float] = OrderedDict()
        self._negative_lock = asyncio.Lock()

        # Per-key in-flight locks for single-flight deduplication.
        self._inflight_guard = asyncio.Lock()
        self._inflight: Dict[Tuple[str, str], asyncio.Lock] = {}

    # ------------------------------------------------------------------
    # Positive cache
    # ------------------------------------------------------------------

    async def get(self, key: Tuple[str, str]) -> Optional[str]:
        """Return the cached entity ID, or ``None`` if absent / expired."""
        now = time.time()
        async with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            entity_id, written_at = entry
            if now - written_at > self._ttl_seconds:
                self._cache.pop(key, None)
                return None
            self._cache.move_to_end(key)
            return entity_id

    async def set(self, key: Tuple[str, str], entity_id: str) -> None:
        """Store a positive resolution result."""
        async with self._cache_lock:
            self._cache[key] = (entity_id, time.time())
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    # ------------------------------------------------------------------
    # Negative cache
    # ------------------------------------------------------------------

    async def is_negative(self, key: Tuple[str, str]) -> bool:
        """Return ``True`` if this key has a valid (non-expired) negative entry."""
        now = time.time()
        async with self._negative_lock:
            expiry = self._negative.get(key)
            if expiry is None:
                return False
            if now >= expiry:
                self._negative.pop(key, None)
                return False
            self._negative.move_to_end(key)
            return True

    async def set_negative(self, key: Tuple[str, str]) -> None:
        """Record that this key could not be resolved (short TTL)."""
        expiry = time.time() + self._negative_ttl
        async with self._negative_lock:
            self._negative[key] = expiry
            self._negative.move_to_end(key)
            while len(self._negative) > self._negative_max_size:
                self._negative.popitem(last=False)

    async def clear_negative(self, key: Tuple[str, str]) -> None:
        """Remove a negative entry, e.g. when an entity is successfully created."""
        async with self._negative_lock:
            self._negative.pop(key, None)

    # ------------------------------------------------------------------
    # Per-key in-flight locks
    # ------------------------------------------------------------------

    async def get_lock(self, key: Tuple[str, str]) -> asyncio.Lock:
        """Return (or create) the per-key lock for single-flight deduplication."""
        async with self._inflight_guard:
            lock = self._inflight.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._inflight[key] = lock
            return lock

    async def release_lock(self, key: Tuple[str, str]) -> None:
        """Remove the per-key lock after use to prevent unbounded growth."""
        async with self._inflight_guard:
            self._inflight.pop(key, None)
