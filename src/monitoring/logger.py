"""
src.monitoring.logger — Structured JSONL event logging for Atwater.

Design goals
------------
- Zero overhead when not writing: file handles are opened lazily on first
  event and only if the log_dir is configured.
- Auto-rotate: one log file per calendar day (UTC), e.g. logs/2026-04-22.jsonl.
- Thread-safe: uses a threading.Lock around every write.
- Clean shutdown: flush() and close() are safe to call multiple times.
- Analysis: read_log(path) is a generator of parsed event dicts.

Event schema (JSONL line)
-------------------------
{
    "ts":           "2026-04-22T16:30:00.123456Z",   # ISO-8601 UTC
    "session_id":   "a1b2c3d4",
    "cycle":        7,                                 # 0 = not in a cycle
    "event_type":   "cycle_start",
    "data":         { ... }
}

Supported event types
---------------------
    cycle_start, cycle_end, agent_call, agent_result,
    knowledge_write, knowledge_promote, optuna_trial,
    state_change, cascade_result, error, checkpoint
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Generator, IO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_EVENT_TYPES: frozenset[str] = frozenset(
    [
        "cycle_start",
        "cycle_end",
        "agent_call",
        "agent_result",
        "knowledge_write",
        "knowledge_promote",
        "optuna_trial",
        "state_change",
        "cascade_result",
        "error",
        "checkpoint",
    ]
)


# ---------------------------------------------------------------------------
# AtwaterLogger
# ---------------------------------------------------------------------------


class AtwaterLogger:
    """
    Structured JSONL event logger for Atwater production runs.

    Parameters
    ----------
    log_dir : str | Path
        Directory where log files are stored.  Created automatically.
        Pass ``None`` to disable disk logging (useful in tests).
    session_id : str | None
        Human-readable session identifier.  Auto-generated (8 hex chars)
        if not provided.

    Examples
    --------
    >>> logger = AtwaterLogger(log_dir="logs/")
    >>> logger.log_event("cycle_start", {"cycle": 1, "params": {}})
    >>> logger.close()
    """

    def __init__(
        self,
        log_dir: str | Path | None = "logs/",
        session_id: str | None = None,
    ) -> None:
        self._log_dir: Path | None = Path(log_dir) if log_dir is not None else None
        self._session_id: str = session_id or uuid.uuid4().hex[:8]

        # Lazy file state — opened on first write
        self._fh: IO[str] | None = None
        self._current_day: str = ""          # YYYY-MM-DD (UTC)
        self._current_cycle: int = 0         # updated via set_cycle()
        self._lock = threading.Lock()
        self._closed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def log_dir(self) -> Path | None:
        return self._log_dir

    def set_cycle(self, cycle_number: int) -> None:
        """Update the current cycle number embedded in every subsequent event."""
        self._current_cycle = cycle_number

    def log_event(
        self,
        event_type: str,
        data: dict[str, Any] | None = None,
        cycle_number: int | None = None,
    ) -> None:
        """
        Write a JSONL event to the current day's log file.

        Parameters
        ----------
        event_type : str
            One of the recognised event types (see VALID_EVENT_TYPES).
            Unknown types are accepted but logged with a warning key.
        data : dict | None
            Arbitrary payload attached to the event.
        cycle_number : int | None
            Override the current cycle number for this event only.
        """
        if self._closed:
            return

        now_utc = datetime.now(timezone.utc)
        record: dict[str, Any] = {
            "ts": now_utc.isoformat().replace("+00:00", "Z"),
            "session_id": self._session_id,
            "cycle": cycle_number if cycle_number is not None else self._current_cycle,
            "event_type": event_type,
            "data": data or {},
        }

        if event_type not in VALID_EVENT_TYPES:
            record["_unknown_event_type"] = True

        line = json.dumps(record, default=str)

        with self._lock:
            if self._log_dir is None:
                return  # disk logging disabled

            # Rotate if the day has changed
            today = now_utc.strftime("%Y-%m-%d")
            if today != self._current_day:
                self._rotate(today)

            try:
                assert self._fh is not None
                self._fh.write(line + "\n")
            except Exception:
                # Best-effort: never crash the caller
                pass

    def flush(self) -> None:
        """Flush buffered log data to disk."""
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.flush()
                except Exception:
                    pass

    def close(self) -> None:
        """Flush and close the log file.  Safe to call multiple times."""
        with self._lock:
            self._closed = True
            if self._fh is not None:
                try:
                    self._fh.flush()
                    self._fh.close()
                except Exception:
                    pass
                finally:
                    self._fh = None

    # ------------------------------------------------------------------
    # Convenience event helpers
    # ------------------------------------------------------------------

    def cycle_start(self, cycle_number: int, params: dict[str, Any]) -> None:
        self.set_cycle(cycle_number)
        self.log_event("cycle_start", {"cycle": cycle_number, "params": params})

    def cycle_end(
        self,
        cycle_number: int,
        score: float | None,
        duration_ms: float,
        knowledge_writes: int = 0,
        errors: dict[str, str] | None = None,
    ) -> None:
        self.log_event(
            "cycle_end",
            {
                "cycle": cycle_number,
                "score": score,
                "duration_ms": duration_ms,
                "knowledge_writes": knowledge_writes,
                "errors": errors or {},
            },
        )

    def agent_call(self, role: str, cycle_number: int | None = None) -> None:
        self.log_event("agent_call", {"role": role}, cycle_number=cycle_number)

    def agent_result(
        self,
        role: str,
        success: bool,
        duration_ms: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cycle_number: int | None = None,
    ) -> None:
        self.log_event(
            "agent_result",
            {
                "role": role,
                "success": success,
                "duration_ms": duration_ms,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
            },
            cycle_number=cycle_number,
        )

    def knowledge_write(self, entry_id: str, tier: str, content_preview: str = "") -> None:
        self.log_event(
            "knowledge_write",
            {"entry_id": entry_id, "tier": tier, "preview": content_preview[:120]},
        )

    def knowledge_promote(self, entry_id: str, from_tier: str, to_tier: str) -> None:
        self.log_event(
            "knowledge_promote",
            {"entry_id": entry_id, "from_tier": from_tier, "to_tier": to_tier},
        )

    def optuna_trial(self, trial_number: int, params: dict[str, Any], score: float | None) -> None:
        self.log_event(
            "optuna_trial",
            {"trial_number": trial_number, "params": params, "score": score},
        )

    def cascade_result(
        self,
        gates_passed: list[str],
        short_circuited: bool,
        total_ms: float,
        gate_scores: dict[str, float] | None = None,
    ) -> None:
        self.log_event(
            "cascade_result",
            {
                "gates_passed": gates_passed,
                "short_circuited": short_circuited,
                "total_ms": total_ms,
                "gate_scores": gate_scores or {},
            },
        )

    def error(self, role: str, message: str, exc_type: str = "") -> None:
        self.log_event(
            "error",
            {"role": role, "message": message, "exc_type": exc_type},
        )

    def checkpoint(self, cycle_number: int, path: str) -> None:
        self.log_event(
            "checkpoint",
            {"cycle": cycle_number, "path": path},
        )

    # ------------------------------------------------------------------
    # Analysis helper
    # ------------------------------------------------------------------

    @staticmethod
    def read_log(path: str | Path) -> Generator[dict[str, Any], None, None]:
        """
        Generator that yields parsed event dicts from a JSONL log file.

        Skips malformed lines silently.

        Parameters
        ----------
        path : str | Path
            Path to a .jsonl log file.

        Yields
        ------
        dict
            Parsed event record.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    @staticmethod
    def read_logs(paths: list[str | Path]) -> Generator[dict[str, Any], None, None]:
        """
        Generator yielding events from multiple JSONL log files, in order.
        """
        for path in paths:
            yield from AtwaterLogger.read_log(path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rotate(self, today: str) -> None:
        """
        Close the current file (if open) and open a new one for ``today``.
        Called with ``self._lock`` already held.
        """
        # Close old handle
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                pass
            self._fh = None

        # Create log dir if needed
        assert self._log_dir is not None
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return

        log_path = self._log_dir / f"{today}.jsonl"
        try:
            self._fh = log_path.open("a", encoding="utf-8", buffering=1)
            self._current_day = today
        except Exception:
            self._fh = None

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "AtwaterLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"AtwaterLogger(session_id={self._session_id!r}, "
            f"log_dir={self._log_dir!r}, closed={self._closed})"
        )
