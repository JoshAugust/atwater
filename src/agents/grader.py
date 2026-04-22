"""
src.agents.grader — Grader Engine.

Responsibility
--------------
Evaluate creator output with structured multi-dimensional scoring and report
the result back to Optuna.  The Grader is the primary driver of knowledge
writes — it records novel findings as observations in the knowledge base.

Scoring dimensions
------------------
- originality         : How fresh/non-repetitive the content is
- brand_alignment     : How well it matches brand guidelines
- technical_quality   : Execution quality (composition, clarity, craft)

Each dimension produces a score (0.0–1.0) and a reasoning string.  The
overall score is a weighted average (configurable).

IMPORTANT: This agent does NOT call an LLM directly.  It builds an evaluation
prompt and returns it in AgentResult.output for the orchestrator to execute.
The orchestrator feeds the LLM response back to ``execute_score_report()``
to produce the final structured result and Optuna report.

State contract
--------------
Reads:  output_path, grading_rubric
Writes: score, structured_analysis
"""

from __future__ import annotations

import logging
from typing import Any

import optuna

from src.agents.base import AgentBase, AgentContext, AgentResult

logger = logging.getLogger(__name__)

# Default dimension weights (must sum to 1.0)
DEFAULT_WEIGHTS: dict[str, float] = {
    "originality": 0.35,
    "brand_alignment": 0.35,
    "technical_quality": 0.30,
}

# Minimum overall score to trigger a knowledge write
KB_WRITE_SCORE_THRESHOLD: float = 0.80


class GraderEngine(AgentBase):
    """
    Grader Engine — scores creator output and reports to Optuna.

    Parameters
    ----------
    study : optuna.Study
        Active Optuna study.  The grader calls study.tell() after scoring.
    dimension_weights : dict[str, float] | None
        Per-dimension weights for overall score calculation.
        Must sum to 1.0.  Defaults to DEFAULT_WEIGHTS.
    kb_write_threshold : float
        Minimum overall score to trigger a knowledge write (default 0.80).
    """

    def __init__(
        self,
        study: optuna.Study,
        dimension_weights: dict[str, float] | None = None,
        kb_write_threshold: float = KB_WRITE_SCORE_THRESHOLD,
    ) -> None:
        self._study = study
        self._weights = dimension_weights or DEFAULT_WEIGHTS
        self._kb_write_threshold = kb_write_threshold

        if abs(sum(self._weights.values()) - 1.0) > 1e-6:
            raise ValueError("dimension_weights must sum to 1.0")

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "GraderEngine"

    @property
    def role(self) -> str:
        return "grader"

    @property
    def readable_state_keys(self) -> list[str]:
        return ["output_path", "grading_rubric"]

    @property
    def writable_state_keys(self) -> list[str]:
        return ["score", "structured_analysis"]

    # ------------------------------------------------------------------
    # Core execute — Phase 1: build evaluation prompt
    # ------------------------------------------------------------------

    def execute(self, context: AgentContext) -> AgentResult:
        """
        Phase 1: Build the structured evaluation prompt for the LLM.

        The orchestrator submits this prompt to the LLM and feeds the
        response to ``execute_score_report()``.

        Returns
        -------
        AgentResult
            output: dict with:
                ``phase``     : "evaluation"
                ``prompt``    : str — evaluation prompt for the LLM
                ``output_path``: str | None
                ``rubric``    : dict | None
        """
        output_path: str | None = context.scoped_state.get("output_path")
        rubric: dict[str, Any] | None = context.scoped_state.get("grading_rubric")
        knowledge_entries = context.knowledge_entries

        prompt = self._build_evaluation_prompt(output_path, rubric, knowledge_entries)

        logger.info(
            "[GraderEngine] Evaluation prompt assembled for output_path=%s", output_path
        )

        return AgentResult(
            output={
                "phase": "evaluation",
                "prompt": prompt,
                "output_path": output_path,
                "rubric": rubric,
            },
            state_updates={},
            knowledge_writes=[],
            score=None,
        )

    # ------------------------------------------------------------------
    # Phase 2: process LLM evaluation response and report to Optuna
    # ------------------------------------------------------------------

    def execute_score_report(
        self,
        context: AgentContext,
        trial: optuna.Trial,
        llm_evaluation: dict[str, Any],
    ) -> AgentResult:
        """
        Phase 2: Parse the LLM evaluation, compute scores, report to Optuna.

        Parameters
        ----------
        context : AgentContext
            Context for this turn (used for knowledge entry retrieval).
        trial : optuna.Trial
            The Optuna trial associated with this evaluation cycle.
            The grader will call study.tell(trial, score) here.
        llm_evaluation : dict[str, Any]
            Parsed LLM evaluation response.  Expected keys per dimension:
                ``score``     : float (0.0–1.0)
                ``reasoning`` : str

        Returns
        -------
        AgentResult
            output: full grading_output dict (see architecture spec)
            score: overall_score (reported to Optuna)
            state_updates: ``score`` and ``structured_analysis``
            knowledge_writes: populated if finding is novel and score >= threshold
        """
        structured = self._build_structured_output(trial, llm_evaluation)
        overall_score: float = structured["overall_score"]

        # Report to Optuna
        self._study.tell(trial, overall_score)
        logger.info(
            "[GraderEngine] Reported score=%.4f for trial #%d to Optuna.",
            overall_score,
            trial.number,
        )

        # Knowledge writes for high-scoring novel findings
        knowledge_writes: list[dict[str, Any]] = []
        if overall_score >= self._kb_write_threshold:
            novel_finding = llm_evaluation.get("novel_finding", "")
            suggest_write = llm_evaluation.get("suggest_knowledge_write", False)

            if novel_finding and suggest_write:
                kw = {
                    "content": novel_finding,
                    "tier": "observation",
                    "confidence": overall_score,
                    "topic_cluster": llm_evaluation.get("topic_cluster", "general"),
                    "metadata": {
                        "trial_id": trial.number,
                        "overall_score": overall_score,
                        "dimensions": structured["dimensions"],
                        "source": "grader",
                    },
                }
                self.validate_knowledge_write(kw)
                knowledge_writes.append(kw)
                logger.info(
                    "[GraderEngine] Novel finding above threshold (%.2f) — writing to KB.",
                    overall_score,
                )

        state_updates = {
            "score": overall_score,
            "structured_analysis": structured,
        }
        self.validate_state_writes(state_updates)

        return AgentResult(
            output=structured,
            state_updates=state_updates,
            knowledge_writes=knowledge_writes,
            score=overall_score,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_evaluation_prompt(
        self,
        output_path: str | None,
        rubric: dict[str, Any] | None,
        knowledge_entries: list[dict[str, Any]],
    ) -> str:
        """Build the evaluation prompt for the LLM."""
        rubric_block = self._format_rubric(rubric)
        kb_block = self._format_knowledge_entries(knowledge_entries)
        path_ref = f"Content to evaluate: {output_path}" if output_path else "Content: (provided inline by orchestrator)"

        return (
            f"You are a professional content evaluator.\n\n"
            f"{path_ref}\n\n"
            "Evaluate the content on these dimensions (score 0.0–1.0 each):\n"
            "  1. originality       — freshness and non-repetitiveness\n"
            "  2. brand_alignment   — consistency with brand guidelines\n"
            "  3. technical_quality — execution quality (composition, clarity, craft)\n\n"
            f"Grading rubric:\n{rubric_block}\n\n"
            f"Relevant knowledge from past experiments:\n{kb_block}\n\n"
            "Output a JSON object with this structure:\n"
            "{\n"
            '  "originality":       {"score": <float>, "reasoning": "<str>"},\n'
            '  "brand_alignment":   {"score": <float>, "reasoning": "<str>"},\n'
            '  "technical_quality": {"score": <float>, "reasoning": "<str>"},\n'
            '  "novel_finding":     "<str or empty>",\n'
            '  "suggest_knowledge_write": <bool>,\n'
            '  "topic_cluster":     "<str>"\n'
            "}"
        )

    def _build_structured_output(
        self,
        trial: optuna.Trial,
        llm_evaluation: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute overall score and build the canonical grading output dict."""
        dimensions: dict[str, dict[str, Any]] = {}
        weighted_sum = 0.0

        for dim, weight in self._weights.items():
            dim_data = llm_evaluation.get(dim, {})
            raw_score = float(dim_data.get("score", 0.0))
            clamped = max(0.0, min(1.0, raw_score))
            weighted_sum += clamped * weight
            dimensions[dim] = {
                "score": clamped,
                "reasoning": dim_data.get("reasoning", ""),
            }

        overall_score = round(weighted_sum, 4)

        return {
            "trial_id": trial.number,
            "overall_score": overall_score,
            "dimensions": dimensions,
            "novel_finding": llm_evaluation.get("novel_finding", ""),
            "suggest_knowledge_write": llm_evaluation.get("suggest_knowledge_write", False),
            "topic_cluster": llm_evaluation.get("topic_cluster", "general"),
        }

    @staticmethod
    def _format_rubric(rubric: dict[str, Any] | None) -> str:
        if not rubric:
            return "  (use general quality standards)"
        lines = []
        for k, v in rubric.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    @staticmethod
    def _format_knowledge_entries(entries: list[dict[str, Any]]) -> str:
        if not entries:
            return "  (no relevant knowledge entries)"
        lines = []
        for i, entry in enumerate(entries, 1):
            tier = entry.get("tier", "?")
            conf = entry.get("confidence", 0.0)
            content = entry.get("content", "")
            lines.append(f"  [{i}] [{tier.upper()} | conf={conf:.2f}] {content}")
        return "\n".join(lines)
