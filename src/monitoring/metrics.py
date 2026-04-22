"""
src.monitoring.metrics — In-memory metrics collector for Atwater.

Accumulates per-cycle, per-agent, knowledge-base, and cascade statistics
during a run, and exposes a structured summary via get_summary().

All operations are thread-safe (uses a threading.Lock).

Typical usage
-------------
    from src.monitoring.metrics import MetricsCollector

    m = MetricsCollector()
    m.track_cycle(1, score=0.82, params={"style": "bold"}, duration_ms=3200)
    m.track_agent("director", duration_ms=840, tokens_in=1200, tokens_out=300)
    m.track_cascade(gates_passed=["fast", "medium"], short_circuited=False, total_ms=4100)

    summary = m.get_summary()
    print(summary.best_score)    # 0.82
    print(summary.total_cycles)  # 1
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# MetricsSummary dataclass
# ---------------------------------------------------------------------------


@dataclass
class MetricsSummary:
    """
    Snapshot of accumulated metrics for the current session.

    Attributes
    ----------
    total_cycles : int
        Number of cycles tracked.
    best_score : float | None
        Highest score seen so far.
    avg_score : float | None
        Mean score across all cycles with a score.
    score_trend : list[float]
        Last ≤20 cycle scores (oldest first).
    kb_size_by_tier : dict[str, int]
        Entry count per knowledge tier (e.g. {"observation": 12, "rule": 3}).
    cascade_efficiency : dict[str, Any]
        Keys: total_evaluated, short_circuit_rate, avg_time_ms,
              fast_pass_rate, medium_pass_rate, llm_reach_rate.
    total_tokens : dict[str, int]
        {"in": ..., "out": ..., "total": ...}.
    avg_cycle_time_ms : float | None
        Mean wall time per cycle in milliseconds.
    agent_stats : dict[str, dict]
        Per-role breakdown: calls, total_ms, avg_ms, tokens_in, tokens_out.
    current_cycle : int
        The most recently tracked cycle number.
    current_params : dict[str, Any]
        Params used in the most recent cycle.
    current_score : float | None
        Score from the most recent cycle.
    """

    total_cycles: int = 0
    best_score: float | None = None
    avg_score: float | None = None
    score_trend: list[float] = field(default_factory=list)
    kb_size_by_tier: dict[str, int] = field(default_factory=dict)
    cascade_efficiency: dict[str, Any] = field(default_factory=dict)
    total_tokens: dict[str, int] = field(default_factory=lambda: {"in": 0, "out": 0, "total": 0})
    avg_cycle_time_ms: float | None = None
    agent_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    current_cycle: int = 0
    current_params: dict[str, Any] = field(default_factory=dict)
    current_score: float | None = None


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------


class MetricsCollector:
    """
    Thread-safe in-memory metrics accumulator for a single Atwater run.

    Call reset() to start fresh (e.g. between sessions).
    """

    _SCORE_TREND_WINDOW = 20    # keep last N scores in trend
    _SCORE_HISTORY_MAX = 10_000  # full history cap

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all accumulated metrics to zero / empty."""
        with self._lock:
            self._cycle_scores: list[float] = []
            self._cycle_durations_ms: list[float] = []
            self._cycle_params: dict[str, Any] = {}
            self._current_cycle: int = 0
            self._current_score: float | None = None

            # Agent tracking: role → list of (duration_ms, tokens_in, tokens_out)
            self._agent_calls: dict[str, list[tuple[float, int, int]]] = defaultdict(list)

            # Knowledge tracking: tier → count
            self._kb_by_tier: dict[str, int] = defaultdict(int)

            # Cascade tracking
            self._cascade_total = 0
            self._cascade_fast_pass = 0
            self._cascade_medium_pass = 0
            self._cascade_llm_reach = 0
            self._cascade_short_circuited = 0
            self._cascade_total_ms: float = 0.0

    # ------------------------------------------------------------------
    # Cycle tracking
    # ------------------------------------------------------------------

    def track_cycle(
        self,
        cycle_number: int,
        score: float | None,
        params: dict[str, Any],
        duration_ms: float,
    ) -> None:
        """
        Record a completed cycle.

        Parameters
        ----------
        cycle_number : int
            1-based cycle index.
        score : float | None
            Score from the grader (None if grader failed).
        params : dict
            Optuna parameter dict used this cycle.
        duration_ms : float
            Wall time for the cycle in milliseconds.
        """
        with self._lock:
            self._current_cycle = cycle_number
            self._current_score = score
            self._cycle_params = dict(params)
            self._cycle_durations_ms.append(duration_ms)

            if score is not None:
                self._cycle_scores.append(float(score))
                # Cap history
                if len(self._cycle_scores) > self._SCORE_HISTORY_MAX:
                    self._cycle_scores = self._cycle_scores[-self._SCORE_HISTORY_MAX :]

    # ------------------------------------------------------------------
    # Agent tracking
    # ------------------------------------------------------------------

    def track_agent(
        self,
        agent_role: str,
        duration_ms: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> None:
        """
        Record a single agent invocation.

        Parameters
        ----------
        agent_role : str
            The role name ("director", "creator", "grader", …).
        duration_ms : float
            Wall time for this agent call.
        tokens_in : int
            Tokens consumed (prompt).
        tokens_out : int
            Tokens generated (completion).
        """
        with self._lock:
            self._agent_calls[agent_role].append(
                (float(duration_ms), int(tokens_in), int(tokens_out))
            )

    # ------------------------------------------------------------------
    # Knowledge tracking
    # ------------------------------------------------------------------

    def track_knowledge(
        self,
        action: str,
        tier: str,
        entry_id: str = "",
    ) -> None:
        """
        Record a knowledge base action.

        Parameters
        ----------
        action : str
            "write" or "promote" (or any custom string).
        tier : str
            Tier the entry was written/promoted TO (e.g. "observation", "rule").
        entry_id : str
            Optional entry ID for traceability.
        """
        with self._lock:
            if action in ("write", "knowledge_write"):
                self._kb_by_tier[tier] += 1
            elif action in ("promote", "knowledge_promote"):
                self._kb_by_tier[tier] += 1

    # ------------------------------------------------------------------
    # Cascade tracking
    # ------------------------------------------------------------------

    def track_cascade(
        self,
        gates_passed: list[str],
        short_circuited: bool,
        total_ms: float,
    ) -> None:
        """
        Record a verifier cascade evaluation.

        Parameters
        ----------
        gates_passed : list[str]
            Which gates passed (e.g. ["fast", "medium"]).
        short_circuited : bool
            True if the cascade stopped before the LLM gate.
        total_ms : float
            Total wall time for this cascade run.
        """
        with self._lock:
            self._cascade_total += 1
            self._cascade_total_ms += float(total_ms)

            passed_set = set(gates_passed)
            if "fast" in passed_set:
                self._cascade_fast_pass += 1
            if "medium" in passed_set:
                self._cascade_medium_pass += 1
            if "llm" in passed_set:
                self._cascade_llm_reach += 1
            if short_circuited:
                self._cascade_short_circuited += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_summary(self) -> MetricsSummary:
        """
        Return a MetricsSummary snapshot of current metrics.

        Returns
        -------
        MetricsSummary
            Immutable snapshot (safe to pass to dashboard / logger).
        """
        with self._lock:
            scores = self._cycle_scores
            durations = self._cycle_durations_ms

            # Score stats
            best_score = max(scores) if scores else None
            avg_score = sum(scores) / len(scores) if scores else None
            score_trend = list(scores[-self._SCORE_TREND_WINDOW :])

            # Cycle stats
            total_cycles = len(durations)
            avg_cycle_ms = sum(durations) / len(durations) if durations else None

            # Knowledge tier counts
            kb_size_by_tier = dict(self._kb_by_tier)

            # Token totals
            total_in = 0
            total_out = 0
            agent_stats: dict[str, dict[str, Any]] = {}
            for role, calls in self._agent_calls.items():
                call_count = len(calls)
                total_ms_role = sum(c[0] for c in calls)
                role_in = sum(c[1] for c in calls)
                role_out = sum(c[2] for c in calls)
                total_in += role_in
                total_out += role_out
                agent_stats[role] = {
                    "calls": call_count,
                    "total_ms": round(total_ms_role, 1),
                    "avg_ms": round(total_ms_role / call_count, 1) if call_count else 0.0,
                    "tokens_in": role_in,
                    "tokens_out": role_out,
                }

            total_tokens = {
                "in": total_in,
                "out": total_out,
                "total": total_in + total_out,
            }

            # Cascade efficiency
            ct = self._cascade_total
            cascade_efficiency: dict[str, Any] = {
                "total_evaluated": ct,
                "short_circuit_rate": (
                    self._cascade_short_circuited / ct if ct > 0 else 0.0
                ),
                "fast_pass_rate": self._cascade_fast_pass / ct if ct > 0 else 0.0,
                "medium_pass_rate": self._cascade_medium_pass / ct if ct > 0 else 0.0,
                "llm_reach_rate": self._cascade_llm_reach / ct if ct > 0 else 0.0,
                "avg_time_ms": (
                    round(self._cascade_total_ms / ct, 1) if ct > 0 else 0.0
                ),
            }

            return MetricsSummary(
                total_cycles=total_cycles,
                best_score=best_score,
                avg_score=avg_score,
                score_trend=score_trend,
                kb_size_by_tier=kb_size_by_tier,
                cascade_efficiency=cascade_efficiency,
                total_tokens=total_tokens,
                avg_cycle_time_ms=avg_cycle_ms,
                agent_stats=agent_stats,
                current_cycle=self._current_cycle,
                current_params=dict(self._cycle_params),
                current_score=self._current_score,
            )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"MetricsCollector("
                f"cycles={len(self._cycle_durations_ms)}, "
                f"best_score={max(self._cycle_scores, default=None)!r})"
            )
