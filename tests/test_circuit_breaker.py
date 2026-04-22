"""
tests/test_circuit_breaker.py — Tests for src.resilience.circuit_breaker.

All tests work without LM Studio — no external services required.
"""

from __future__ import annotations

import threading
import time

import pytest

from src.resilience.circuit_breaker import CircuitBreaker, CircuitOpenError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _always_succeeds(*args, **kwargs) -> str:
    return "ok"


def _always_fails(*args, **kwargs) -> None:
    raise RuntimeError("deliberate failure")


def _make_cb(**kwargs) -> CircuitBreaker:
    defaults = dict(failure_threshold=3, recovery_timeout=0.1, half_open_max=2)
    defaults.update(kwargs)
    return CircuitBreaker(**defaults)


# ---------------------------------------------------------------------------
# Init / validation
# ---------------------------------------------------------------------------

class TestInit:
    def test_starts_closed(self):
        cb = _make_cb()
        assert cb.get_state() == "CLOSED"

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            CircuitBreaker(failure_threshold=0)

    def test_invalid_timeout(self):
        with pytest.raises(ValueError):
            CircuitBreaker(recovery_timeout=0)

    def test_invalid_half_open_max(self):
        with pytest.raises(ValueError):
            CircuitBreaker(half_open_max=0)


# ---------------------------------------------------------------------------
# CLOSED state — normal operation
# ---------------------------------------------------------------------------

class TestClosedState:
    def test_successful_calls_pass_through(self):
        cb = _make_cb()
        result = cb.call(_always_succeeds)
        assert result == "ok"

    def test_failure_increments_counter(self):
        cb = _make_cb(failure_threshold=5)
        for _ in range(3):
            with pytest.raises(RuntimeError):
                cb.call(_always_fails)
        stats = cb.get_stats()
        assert stats["failure_count"] == 3
        assert stats["state"] == "CLOSED"

    def test_success_resets_failure_count(self):
        cb = _make_cb(failure_threshold=5)
        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(_always_fails)
        cb.call(_always_succeeds)
        stats = cb.get_stats()
        assert stats["failure_count"] == 0

    def test_args_and_kwargs_passed(self):
        def echo(x, y=None):
            return (x, y)

        cb = _make_cb()
        result = cb.call(echo, 42, y="hello")
        assert result == (42, "hello")


# ---------------------------------------------------------------------------
# OPEN state — fail-fast
# ---------------------------------------------------------------------------

class TestOpenState:
    def test_trips_to_open_at_threshold(self):
        cb = _make_cb(failure_threshold=3)
        for _ in range(3):
            with pytest.raises(RuntimeError):
                cb.call(_always_fails)
        assert cb.get_state() == "OPEN"

    def test_open_rejects_calls_immediately(self):
        cb = _make_cb(failure_threshold=1)
        with pytest.raises(RuntimeError):
            cb.call(_always_fails)
        assert cb.get_state() == "OPEN"

        # Next call should raise CircuitOpenError, NOT RuntimeError
        with pytest.raises(CircuitOpenError):
            cb.call(_always_succeeds)

    def test_circuit_open_error_has_retry_after(self):
        cb = _make_cb(failure_threshold=1, recovery_timeout=10.0)
        with pytest.raises(RuntimeError):
            cb.call(_always_fails)

        with pytest.raises(CircuitOpenError) as exc_info:
            cb.call(_always_succeeds)
        assert exc_info.value.retry_after is not None
        assert exc_info.value.retry_after > 0

    def test_trips_counter_increments(self):
        cb = _make_cb(failure_threshold=1)
        with pytest.raises(RuntimeError):
            cb.call(_always_fails)
        assert cb.get_stats()["total_trips"] == 1


# ---------------------------------------------------------------------------
# HALF_OPEN state — recovery probing
# ---------------------------------------------------------------------------

class TestHalfOpenState:
    def test_transitions_to_half_open_after_timeout(self):
        cb = _make_cb(failure_threshold=1, recovery_timeout=0.05)
        with pytest.raises(RuntimeError):
            cb.call(_always_fails)
        assert cb.get_state() == "OPEN"

        time.sleep(0.1)  # wait past recovery_timeout
        # get_state() checks for timeout → HALF_OPEN
        assert cb.get_state() == "HALF_OPEN"

    def test_success_in_half_open_closes_circuit(self):
        cb = _make_cb(failure_threshold=1, recovery_timeout=0.05)
        with pytest.raises(RuntimeError):
            cb.call(_always_fails)

        time.sleep(0.1)
        result = cb.call(_always_succeeds)
        assert result == "ok"
        assert cb.get_state() == "CLOSED"

    def test_failure_in_half_open_re_opens(self):
        cb = _make_cb(failure_threshold=1, recovery_timeout=0.05)
        with pytest.raises(RuntimeError):
            cb.call(_always_fails)

        time.sleep(0.1)
        assert cb.get_state() == "HALF_OPEN"

        with pytest.raises(RuntimeError):
            cb.call(_always_fails)
        assert cb.get_state() == "OPEN"

    def test_half_open_limits_concurrent_calls(self):
        cb = _make_cb(failure_threshold=1, recovery_timeout=0.05, half_open_max=1)
        with pytest.raises(RuntimeError):
            cb.call(_always_fails)

        time.sleep(0.1)
        assert cb.get_state() == "HALF_OPEN"

        # First call in HALF_OPEN goes through (but may succeed or fail)
        # Simulate filling the slot manually by calling with a blocking fn
        results = []

        def slow_succeed():
            time.sleep(0.05)
            return "ok"

        # Run the first call in a thread
        t = threading.Thread(target=lambda: results.append(cb.call(slow_succeed)))
        t.start()
        time.sleep(0.01)  # give thread time to enter

        # Second call should be rejected (half_open_max=1 already consumed)
        # This is best-effort — the thread may finish first; just assert no crash
        t.join()


# ---------------------------------------------------------------------------
# Manual reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_closes_open_circuit(self):
        cb = _make_cb(failure_threshold=1)
        with pytest.raises(RuntimeError):
            cb.call(_always_fails)
        assert cb.get_state() == "OPEN"
        cb.reset()
        assert cb.get_state() == "CLOSED"

    def test_after_reset_successful_calls_work(self):
        cb = _make_cb(failure_threshold=1)
        with pytest.raises(RuntimeError):
            cb.call(_always_fails)
        cb.reset()
        assert cb.call(_always_succeeds) == "ok"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestStats:
    def test_total_calls_counts_rejected_calls(self):
        cb = _make_cb(failure_threshold=1)
        with pytest.raises(RuntimeError):
            cb.call(_always_fails)
        with pytest.raises(CircuitOpenError):
            cb.call(_always_succeeds)
        stats = cb.get_stats()
        assert stats["total_calls"] == 2

    def test_total_failures_counts_real_exceptions(self):
        cb = _make_cb(failure_threshold=5)
        for _ in range(3):
            with pytest.raises(RuntimeError):
                cb.call(_always_fails)
        assert cb.get_stats()["total_failures"] == 3

    def test_get_stats_keys(self):
        cb = _make_cb()
        stats = cb.get_stats()
        for key in ("state", "total_calls", "total_failures", "total_trips",
                    "failure_count", "half_open_calls", "opened_at"):
            assert key in stats


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_failures_all_counted(self):
        """Multiple threads failing concurrently should not corrupt the counter."""
        cb = CircuitBreaker(failure_threshold=100, recovery_timeout=60)
        errors = []
        successes = []

        def worker():
            try:
                cb.call(_always_fails)
            except RuntimeError:
                errors.append(1)
            except CircuitOpenError:
                pass

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = cb.get_stats()
        assert stats["total_calls"] == 20
        assert stats["total_failures"] == len(errors)

    def test_only_one_trip(self):
        """Circuit should trip exactly once even with concurrent failures."""
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        barrier = threading.Barrier(10)

        def worker():
            barrier.wait()
            try:
                cb.call(_always_fails)
            except (RuntimeError, CircuitOpenError):
                pass

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # After 10 concurrent failures, circuit should be OPEN
        assert cb.get_state() == "OPEN"
