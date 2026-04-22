"""
src.evaluation.cascade — Verifier cascade orchestrator.

Runs fast → medium → llm gates in sequence, short-circuiting early when a
gate fails.  Saves 60-70% of LLM grading cost by filtering bad outputs before
the expensive LLM judge runs.

Cascade logic:
  1. FastGate  — if fails → score 0, stop (short-circuit at cheapest gate)
  2. MediumGate — if score < threshold → use medium score, stop
  3. LLMGate  — full evaluation → use LLM score as final score

Stats tracking allows measuring cascade efficiency across a run.

Public API:
  VerifierCascade.__init__(fast_gate, medium_gate, llm_gate, thresholds)
  VerifierCascade.evaluate(output, context) -> CascadeResult
  VerifierCascade.stats                    -> dict
  VerifierCascade.reset_stats()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


@dataclass
class CascadeThresholds:
    """
    Per-gate pass thresholds.

    fast_min : float
        FastGate score must be above this to proceed.
        Set to > 0 to enforce; 0.0 = gate must simply pass (any failures = stop).
    medium_min : float
        MediumGate aggregate score must be above this to proceed to LLM gate.
        If below, the cascade stops and medium_score is the final score.
    llm_min : float
        LLM gate overall score below this means the output is low quality.
        (Informational; does not stop the cascade — LLM is the terminal gate.)
    """

    fast_min: float = 0.0      # any failure = stop; 0.0 means check passed/failed
    medium_min: float = 0.40   # medium score below 40% → short-circuit
    llm_min: float = 0.0       # informational


# ---------------------------------------------------------------------------
# VerifierCascade
# ---------------------------------------------------------------------------


class VerifierCascade:
    """
    Three-stage verifier cascade: fast → medium → llm.

    Parameters
    ----------
    fast_gate : FastGate
        Rule-based gate (<10ms).
    medium_gate : MediumGate
        Embedding/perceptual gate (~100ms).
    llm_gate : LLMGate
        Full LLM judge (~2-5s).
    thresholds : CascadeThresholds | None
        Pass/fail thresholds per gate.  Uses defaults if None.
    """

    def __init__(
        self,
        fast_gate: Any,    # FastGate
        medium_gate: Any,  # MediumGate
        llm_gate: Any,     # LLMGate
        thresholds: CascadeThresholds | None = None,
    ) -> None:
        self._fast = fast_gate
        self._medium = medium_gate
        self._llm = llm_gate
        self._thresholds = thresholds or CascadeThresholds()

        # Stats tracking
        self._stats: dict[str, int | float] = {
            "total_evaluated": 0,
            "fast_passed": 0,
            "fast_failed": 0,
            "medium_passed": 0,
            "medium_failed": 0,
            "llm_reached": 0,
            "short_circuited_at_fast": 0,
            "short_circuited_at_medium": 0,
            "total_time_ms": 0.0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        output: Any,  # CreativeOutput
        context: dict[str, Any] | None = None,
    ) -> Any:  # returns CascadeResult
        """
        Run the verifier cascade on a creative output.

        Parameters
        ----------
        output : CreativeOutput
            The creative artefact to evaluate.
        context : dict | None
            Evaluation context.  Recognised keys:
                brand_colors      (list[tuple])    — for FastGate palette check
                brand_embeddings  (np.ndarray)     — for MediumGate style check
                corpus_embeddings (np.ndarray)     — for MediumGate novelty check
                rubric            (dict)            — for LLMGate evaluation
                knowledge_context (list[dict])      — KB entries for LLMGate

        Returns
        -------
        CascadeResult
        """
        from src.evaluation import CascadeResult, GateResult

        t0 = time.perf_counter()
        context = context or {}
        self._stats["total_evaluated"] += 1

        gates_passed: list[str] = []
        gates_failed: list[str] = []
        gate_scores: dict[str, float] = {}
        gate_details: dict[str, Any] = {}

        # ==================================================================
        # Stage 1: Fast gate
        # ==================================================================
        brand_colors = context.get("brand_colors")
        fast_result: Any = self._fast.run_all(output, brand_colors=brand_colors)
        gate_scores["fast"] = fast_result.score
        gate_details["fast"] = fast_result

        if not fast_result.passed or fast_result.score <= self._thresholds.fast_min:
            # Short-circuit: output failed even the cheapest gate
            gates_failed.append("fast")
            self._stats["fast_failed"] += 1
            self._stats["short_circuited_at_fast"] += 1

            total_ms = (time.perf_counter() - t0) * 1000
            self._stats["total_time_ms"] = (
                float(self._stats["total_time_ms"]) + total_ms
            )

            logger.info(
                "[VerifierCascade] SHORT-CIRCUIT at FastGate. "
                "score=0.0 failures=%s time_ms=%.1f",
                fast_result.failures[:3],
                total_ms,
            )

            return CascadeResult(
                final_score=0.0,
                gates_passed=gates_passed,
                gates_failed=gates_failed,
                total_time_ms=total_ms,
                short_circuited=True,
                gate_scores=gate_scores,
                gate_details=gate_details,
            )

        gates_passed.append("fast")
        self._stats["fast_passed"] += 1

        # ==================================================================
        # Stage 2: Medium gate
        # ==================================================================
        brand_context = {
            k: context[k]
            for k in ("brand_embeddings", "corpus_embeddings", "novelty_k", "text_description")
            if k in context
        }
        medium_result: Any = self._medium.run_all(output, brand_context=brand_context)
        gate_scores["medium"] = medium_result.score
        gate_details["medium"] = medium_result

        if not medium_result.passed or medium_result.score < self._thresholds.medium_min:
            # Short-circuit: use medium score as final
            gates_failed.append("medium")
            self._stats["medium_failed"] += 1
            self._stats["short_circuited_at_medium"] += 1

            total_ms = (time.perf_counter() - t0) * 1000
            self._stats["total_time_ms"] = (
                float(self._stats["total_time_ms"]) + total_ms
            )

            logger.info(
                "[VerifierCascade] SHORT-CIRCUIT at MediumGate. "
                "score=%.3f time_ms=%.1f",
                medium_result.score,
                total_ms,
            )

            return CascadeResult(
                final_score=medium_result.score,
                gates_passed=gates_passed,
                gates_failed=gates_failed,
                total_time_ms=total_ms,
                short_circuited=True,
                gate_scores=gate_scores,
                gate_details=gate_details,
            )

        gates_passed.append("medium")
        self._stats["medium_passed"] += 1

        # ==================================================================
        # Stage 3: LLM gate (full evaluation)
        # ==================================================================
        self._stats["llm_reached"] += 1

        rubric = context.get("rubric")
        knowledge_context = context.get("knowledge_context")

        llm_result: Any = self._llm.evaluate(
            output,
            rubric=rubric,
            knowledge_context=knowledge_context,
        )
        gate_scores["llm"] = llm_result.overall_score
        gate_details["llm"] = llm_result

        # LLM gate is terminal — always goes into passed
        gates_passed.append("llm")

        total_ms = (time.perf_counter() - t0) * 1000
        self._stats["total_time_ms"] = float(self._stats["total_time_ms"]) + total_ms

        logger.info(
            "[VerifierCascade] FULL PASS. "
            "final_score=%.3f gates_passed=%s time_ms=%.1f",
            llm_result.overall_score,
            gates_passed,
            total_ms,
        )

        return CascadeResult(
            final_score=llm_result.overall_score,
            gates_passed=gates_passed,
            gates_failed=gates_failed,
            total_time_ms=total_ms,
            short_circuited=False,
            gate_scores=gate_scores,
            gate_details=gate_details,
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        """
        Return cascade efficiency statistics.

        Keys:
            total_evaluated           — total outputs processed
            fast_passed / fast_failed  — FastGate counts
            medium_passed / medium_failed — MediumGate counts
            llm_reached               — outputs that reached the LLM gate
            short_circuited_at_fast   — stopped at fast gate
            short_circuited_at_medium — stopped at medium gate
            short_circuit_rate        — fraction stopped before LLM
            avg_time_ms               — average wall time per evaluation
            total_time_ms             — cumulative wall time
        """
        total = int(self._stats["total_evaluated"])
        sc_fast = int(self._stats["short_circuited_at_fast"])
        sc_medium = int(self._stats["short_circuited_at_medium"])
        total_time = float(self._stats["total_time_ms"])

        return {
            **self._stats,
            "short_circuit_rate": (sc_fast + sc_medium) / total if total > 0 else 0.0,
            "avg_time_ms": total_time / total if total > 0 else 0.0,
        }

    def reset_stats(self) -> None:
        """Reset all cascade stats to zero."""
        for key in self._stats:
            self._stats[key] = 0 if isinstance(self._stats[key], int) else 0.0
        logger.info("[VerifierCascade] Stats reset.")

    def stats_summary(self) -> str:
        """Return a human-readable stats summary string."""
        s = self.stats
        total = s["total_evaluated"]
        if total == 0:
            return "VerifierCascade: no evaluations yet."

        return (
            f"VerifierCascade stats ({total} evaluated): "
            f"fast_pass={s['fast_passed']} fast_fail={s['fast_failed']} | "
            f"medium_pass={s['medium_passed']} medium_fail={s['medium_failed']} | "
            f"llm_reached={s['llm_reached']} | "
            f"short_circuit_rate={s['short_circuit_rate']:.0%} | "
            f"avg_time={s['avg_time_ms']:.1f}ms"
        )
