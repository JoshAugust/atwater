"""
src.resilience.circuit_breaker — Thread-safe circuit breaker for LLM calls.

States
------
CLOSED    Normal operation. All calls pass through. Failure counter is tracked.
OPEN      Failure threshold exceeded. All calls fail immediately (no actual call).
HALF_OPEN Recovery probe state. A limited number of test calls are allowed.
          If they succeed → CLOSED. If any fail → OPEN (reset timer).

Usage
-----
    from src.resilience.circuit_breaker import CircuitBreaker, CircuitOpenError

    cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60, half_open_max=2)

    try:
        result = cb.call(my_llm_function, prompt)
    except CircuitOpenError:
        # LLM unavailable — use fallback
        result = fallback()
    except Exception as exc:
        # Actual LLM error (also recorded by circuit breaker)
        ...
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CircuitOpenError(Exception):
    """Raised when a call is attempted while the circuit is OPEN."""

    def __init__(self, retry_after: float | None = None) -> None:
        self.retry_after = retry_after
        if retry_after is not None:
            msg = f"Circuit is OPEN — retry after {retry_after:.1f}s"
        else:
            msg = "Circuit is OPEN — call rejected (fail-fast)"
        super().__init__(msg)


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------

class _State(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """
    Thread-safe circuit breaker.

    Parameters
    ----------
    failure_threshold : int
        Number of consecutive failures before the circuit trips to OPEN.
    recovery_timeout : float
        Seconds to wait in OPEN state before probing (→ HALF_OPEN).
    half_open_max : int
        Maximum concurrent/sequential test calls allowed in HALF_OPEN.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max: int = 2,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be > 0")
        if half_open_max < 1:
            raise ValueError("half_open_max must be >= 1")

        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max = half_open_max

        # Mutable state — always access under self._lock
        self._lock = threading.Lock()
        self._state: _State = _State.CLOSED
        self._failure_count: int = 0
        self._half_open_calls: int = 0  # calls in progress / allowed in HALF_OPEN
        self._last_failure_time: float | None = None
        self._opened_at: float | None = None

        # Statistics (never reset)
        self._total_calls: int = 0
        self._total_failures: int = 0
        self._total_trips: int = 0  # times circuit tripped to OPEN

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Wrap ``fn(*args, **kwargs)`` with circuit-breaker logic.

        Raises
        ------
        CircuitOpenError
            If the circuit is OPEN (fail-fast, no actual call).
        Exception
            Whatever ``fn`` raises (recorded as a failure).
        """
        with self._lock:
            self._total_calls += 1
            state = self._get_state_locked()

            if state == _State.OPEN:
                retry_after = self._seconds_until_half_open()
                raise CircuitOpenError(retry_after=retry_after)

            if state == _State.HALF_OPEN:
                if self._half_open_calls >= self._half_open_max:
                    raise CircuitOpenError(retry_after=0.0)
                self._half_open_calls += 1

        # --- Execute the function (outside the lock for concurrency) ---
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            self._record_failure()
            raise

        self._record_success()
        return result

    def get_state(self) -> str:
        """Return the current state as a string: 'CLOSED', 'OPEN', or 'HALF_OPEN'."""
        with self._lock:
            return self._get_state_locked().value

    def get_stats(self) -> dict[str, Any]:
        """
        Return a snapshot of circuit-breaker statistics.

        Returns
        -------
        dict with keys:
            state          : str — current state
            total_calls    : int — all calls attempted (including rejected)
            total_failures : int — calls that raised exceptions
            total_trips    : int — times circuit tripped to OPEN
            failure_count  : int — current consecutive failure count (CLOSED state)
            half_open_calls: int — calls issued in HALF_OPEN
            opened_at      : float | None — epoch time circuit last opened
        """
        with self._lock:
            return {
                "state": self._get_state_locked().value,
                "total_calls": self._total_calls,
                "total_failures": self._total_failures,
                "total_trips": self._total_trips,
                "failure_count": self._failure_count,
                "half_open_calls": self._half_open_calls,
                "opened_at": self._opened_at,
            }

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED state."""
        with self._lock:
            self._transition_to_closed()

    # ------------------------------------------------------------------
    # Internal — all called under lock unless noted
    # ------------------------------------------------------------------

    def _get_state_locked(self) -> _State:
        """
        Return the current state, automatically transitioning OPEN → HALF_OPEN
        when the recovery timeout has elapsed.

        Must be called while holding self._lock.
        """
        if self._state == _State.OPEN:
            if self._opened_at is not None:
                elapsed = time.monotonic() - self._opened_at
                if elapsed >= self._recovery_timeout:
                    self._transition_to_half_open()
        return self._state

    def _record_failure(self) -> None:
        """Record a call failure. May trip the circuit or re-open it."""
        with self._lock:
            self._total_failures += 1
            self._last_failure_time = time.monotonic()

            if self._state == _State.HALF_OPEN:
                # A failure during probing → back to OPEN, reset timer.
                logger.warning(
                    "CircuitBreaker: failure in HALF_OPEN → re-opening circuit"
                )
                self._transition_to_open()
                return

            # CLOSED state: increment consecutive failure count.
            self._failure_count += 1
            if self._failure_count >= self._failure_threshold:
                logger.warning(
                    "CircuitBreaker: %d consecutive failures → tripping to OPEN",
                    self._failure_count,
                )
                self._transition_to_open()

    def _record_success(self) -> None:
        """Record a successful call. Closes the circuit if in HALF_OPEN."""
        with self._lock:
            if self._state == _State.HALF_OPEN:
                # One success in HALF_OPEN is enough to close the circuit.
                logger.info("CircuitBreaker: success in HALF_OPEN → closing circuit")
                self._transition_to_closed()
            elif self._state == _State.CLOSED:
                # Reset failure counter on any success.
                self._failure_count = 0

    def _transition_to_open(self) -> None:
        """Trip to OPEN state. Must be called under lock."""
        self._state = _State.OPEN
        self._opened_at = time.monotonic()
        self._half_open_calls = 0
        self._total_trips += 1
        logger.error(
            "CircuitBreaker: OPEN (trip #%d). Will probe after %.0fs.",
            self._total_trips,
            self._recovery_timeout,
        )

    def _transition_to_half_open(self) -> None:
        """Move from OPEN to HALF_OPEN for recovery probing. Must be called under lock."""
        self._state = _State.HALF_OPEN
        self._half_open_calls = 0
        logger.info(
            "CircuitBreaker: OPEN → HALF_OPEN (probing recovery, max %d call(s))",
            self._half_open_max,
        )

    def _transition_to_closed(self) -> None:
        """Close the circuit (normal operation). Must be called under lock."""
        self._state = _State.CLOSED
        self._failure_count = 0
        self._half_open_calls = 0
        self._opened_at = None
        logger.info("CircuitBreaker: CLOSED (circuit reset)")

    def _seconds_until_half_open(self) -> float | None:
        """
        Seconds remaining until recovery probe.  None if unknown.
        Must be called under lock.
        """
        if self._opened_at is None:
            return None
        elapsed = time.monotonic() - self._opened_at
        remaining = self._recovery_timeout - elapsed
        return max(0.0, remaining)
