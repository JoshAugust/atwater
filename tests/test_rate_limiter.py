"""
tests/test_rate_limiter.py — Tests for src.resilience.rate_limiter.

No external services required.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from src.resilience.rate_limiter import RateLimiter


# ---------------------------------------------------------------------------
# Init / validation
# ---------------------------------------------------------------------------

class TestInit:
    def test_valid_construction(self):
        rl = RateLimiter(max_calls_per_minute=60, burst=10)
        stats = rl.get_stats()
        assert stats["tokens_available"] == pytest.approx(10.0)

    def test_invalid_rate(self):
        with pytest.raises(ValueError):
            RateLimiter(max_calls_per_minute=0)

    def test_invalid_burst(self):
        with pytest.raises(ValueError):
            RateLimiter(burst=0)

    def test_starts_full(self):
        rl = RateLimiter(max_calls_per_minute=60, burst=5)
        assert rl.get_stats()["tokens_available"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Burst handling
# ---------------------------------------------------------------------------

class TestBurst:
    def test_burst_calls_succeed_immediately(self):
        """All burst calls should complete without sleeping."""
        rl = RateLimiter(max_calls_per_minute=60, burst=5)
        start = time.perf_counter()
        for _ in range(5):
            assert rl.acquire() is True
        elapsed = time.perf_counter() - start
        # 5 burst calls should take < 50ms on any machine
        assert elapsed < 0.05

    def test_tokens_depleted_after_burst(self):
        rl = RateLimiter(max_calls_per_minute=60, burst=3)
        for _ in range(3):
            rl.acquire()
        # After burst, tokens should be near 0 (some refill may occur)
        stats = rl.get_stats()
        assert stats["tokens_available"] < 1.0

    def test_burst_cannot_exceed_capacity(self):
        """The bucket should not overflow above burst capacity."""
        rl = RateLimiter(max_calls_per_minute=6000, burst=3)  # fast refill
        time.sleep(1.0)  # let it refill
        stats = rl.get_stats()
        assert stats["tokens_available"] <= 3.0


# ---------------------------------------------------------------------------
# Rate enforcement
# ---------------------------------------------------------------------------

class TestRateEnforcement:
    def test_rate_limits_calls_per_second(self):
        """With burst=1, rate=60/min → 1/s, 3 calls should take >= 2s."""
        rl = RateLimiter(max_calls_per_minute=60, burst=1)
        start = time.perf_counter()
        for _ in range(3):
            rl.acquire()
        elapsed = time.perf_counter() - start
        # First call is free (burst=1), next two cost ~1s each → ~2s total
        assert elapsed >= 1.8  # allow some slack

    def test_acquire_returns_true(self):
        rl = RateLimiter(max_calls_per_minute=600, burst=10)
        assert rl.acquire() is True

    def test_high_rate_no_throttle(self):
        """Very high rate limit → no meaningful throttling for small bursts."""
        rl = RateLimiter(max_calls_per_minute=10_000, burst=100)
        start = time.perf_counter()
        for _ in range(50):
            rl.acquire()
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5  # should be fast


# ---------------------------------------------------------------------------
# Async acquire
# ---------------------------------------------------------------------------

class TestAsyncAcquire:
    def test_async_acquire_returns_true(self):
        async def run():
            rl = RateLimiter(max_calls_per_minute=600, burst=10)
            result = await rl.async_acquire()
            return result

        assert asyncio.run(run()) is True

    def test_async_burst_calls(self):
        async def run():
            rl = RateLimiter(max_calls_per_minute=600, burst=5)
            results = []
            for _ in range(5):
                results.append(await rl.async_acquire())
            return results

        results = asyncio.run(run())
        assert all(r is True for r in results)

    def test_async_rate_throttles(self):
        """Async acquire should also respect the rate limit."""
        async def run():
            rl = RateLimiter(max_calls_per_minute=60, burst=1)
            start = time.perf_counter()
            for _ in range(2):
                await rl.async_acquire()
            return time.perf_counter() - start

        elapsed = asyncio.run(run())
        assert elapsed >= 0.8


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_keys(self):
        rl = RateLimiter()
        stats = rl.get_stats()
        for key in ("tokens_available", "rate_per_second", "burst",
                    "total_acquired", "total_waited_ms"):
            assert key in stats

    def test_total_acquired_increments(self):
        rl = RateLimiter(max_calls_per_minute=600, burst=10)
        for _ in range(4):
            rl.acquire()
        assert rl.get_stats()["total_acquired"] == 4


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestConcurrency:
    def test_concurrent_burst_no_overflow(self):
        """Multiple threads exhausting the burst bucket simultaneously."""
        rl = RateLimiter(max_calls_per_minute=600, burst=5)
        acquired = []
        lock = threading.Lock()

        def worker():
            rl.acquire()
            with lock:
                acquired.append(1)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(acquired) == 5
