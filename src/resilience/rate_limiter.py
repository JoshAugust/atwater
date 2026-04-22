"""
src.resilience.rate_limiter — Token bucket rate limiter for LLM API calls.

Algorithm: Token bucket with continuous refill.
- Bucket starts at ``burst`` tokens (max capacity).
- Tokens refill at ``max_calls_per_minute / 60`` tokens per second.
- Each call consumes 1 token.
- If no token is available, the caller blocks (sync) or awaits (async) until
  one refills.

Usage (sync)
------------
    from src.resilience.rate_limiter import RateLimiter

    limiter = RateLimiter(max_calls_per_minute=30, burst=5)
    limiter.acquire()        # blocks if throttled
    response = call_llm()

Usage (async)
-------------
    async def handler():
        await limiter.async_acquire()
        response = await async_call_llm()
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token-bucket rate limiter with sync and async acquire methods.

    Parameters
    ----------
    max_calls_per_minute : int
        Sustained call rate allowed per minute.
    burst : int
        Maximum burst capacity (initial token count and bucket ceiling).
    """

    def __init__(
        self,
        max_calls_per_minute: int = 30,
        burst: int = 5,
    ) -> None:
        if max_calls_per_minute <= 0:
            raise ValueError("max_calls_per_minute must be > 0")
        if burst <= 0:
            raise ValueError("burst must be > 0")

        self._rate: float = max_calls_per_minute / 60.0  # tokens per second
        self._burst: int = burst
        self._tokens: float = float(burst)  # start full
        self._last_refill: float = time.monotonic()
        self._lock = threading.Lock()

        self._total_acquired: int = 0
        self._total_waited_ms: float = 0.0

    # ------------------------------------------------------------------
    # Sync API
    # ------------------------------------------------------------------

    def acquire(self) -> bool:
        """
        Consume one token, blocking until one is available.

        Returns
        -------
        bool
            Always True once the token is acquired.
        """
        while True:
            wait_time = self._try_acquire()
            if wait_time == 0.0:
                return True
            # Sleep for the computed wait (but not more than 1s to remain responsive).
            sleep_for = min(wait_time, 1.0)
            logger.debug("RateLimiter: throttling — sleeping %.3fs", sleep_for)
            time.sleep(sleep_for)

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def async_acquire(self) -> bool:
        """
        Async version of acquire(). Yields control while waiting.

        Returns
        -------
        bool
            Always True once the token is acquired.
        """
        while True:
            wait_time = self._try_acquire()
            if wait_time == 0.0:
                return True
            sleep_for = min(wait_time, 1.0)
            logger.debug("RateLimiter: async throttling — sleeping %.3fs", sleep_for)
            await asyncio.sleep(sleep_for)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return current limiter statistics."""
        with self._lock:
            self._refill()
            return {
                "tokens_available": self._tokens,
                "rate_per_second": self._rate,
                "burst": self._burst,
                "total_acquired": self._total_acquired,
                "total_waited_ms": round(self._total_waited_ms, 2),
            }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _refill(self) -> None:
        """Add tokens based on elapsed time since last refill. Must be called under lock."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed * self._rate
        self._tokens = min(float(self._burst), self._tokens + new_tokens)
        self._last_refill = now

    def _try_acquire(self) -> float:
        """
        Attempt to consume a token.

        Returns
        -------
        float
            0.0 if a token was consumed (success).
            Seconds to wait before a token will be available (if > 0).
        """
        with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                self._total_acquired += 1
                return 0.0
            # Compute how long until we have 1 token again.
            wait_seconds = (1.0 - self._tokens) / self._rate
            self._total_waited_ms += wait_seconds * 1000
            return wait_seconds
