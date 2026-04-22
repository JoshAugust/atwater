"""
src.evaluation.llm_gate — Full LLM judge gate (~2-5s).

Uses LMStudioClient with structured output (GRADER_LLM_SCHEMA) to produce
rich multi-dimensional grading with reasoning and novel findings.

Public API:
  LLMGate.evaluate(output, rubric, knowledge_context) -> GradeResult
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM Gate schema
# ---------------------------------------------------------------------------

# Extended grader schema for the cascade (includes suggestions field).
_LLM_GATE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "originality": {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "reasoning": {"type": "string", "minLength": 10},
            },
            "required": ["score", "reasoning"],
            "additionalProperties": False,
        },
        "brand_alignment": {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "reasoning": {"type": "string", "minLength": 10},
            },
            "required": ["score", "reasoning"],
            "additionalProperties": False,
        },
        "technical_quality": {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "reasoning": {"type": "string", "minLength": 10},
            },
            "required": ["score", "reasoning"],
            "additionalProperties": False,
        },
        "novel_finding": {
            "type": ["string", "null"],
            "description": "Novel finding to write to knowledge base, or null.",
        },
        "suggest_knowledge_write": {
            "type": "boolean",
        },
        "topic_cluster": {
            "type": "string",
            "minLength": 1,
        },
        "suggestions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Actionable improvement suggestions.",
        },
    },
    "required": [
        "originality",
        "brand_alignment",
        "technical_quality",
        "novel_finding",
        "suggest_knowledge_write",
        "topic_cluster",
        "suggestions",
    ],
    "additionalProperties": False,
}

# Default dimension weights for overall score
_DEFAULT_WEIGHTS: dict[str, float] = {
    "originality": 0.35,
    "brand_alignment": 0.35,
    "technical_quality": 0.30,
}


# ---------------------------------------------------------------------------
# LLMGate
# ---------------------------------------------------------------------------


class LLMGate:
    """
    Full LLM judge — the deepest, slowest gate (~2-5s).

    Calls LMStudioClient with structured output using the evaluation schema.
    Returns a GradeResult with per-dimension scores, reasoning, and
    suggestions for improvement.

    Parameters
    ----------
    client : LMStudioClient
        Configured LM Studio HTTP client.
    dimension_weights : dict[str, float] | None
        Weights for overall score calculation.  Must sum to 1.0.
    temperature : float
        LLM temperature.  Lower = more deterministic grading.
    max_tokens : int
        Max tokens for the evaluation response.
    """

    def __init__(
        self,
        client: Any,  # LMStudioClient — typed loosely to avoid tight coupling
        dimension_weights: dict[str, float] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> None:
        self._client = client
        self._weights = dimension_weights or _DEFAULT_WEIGHTS.copy()
        self._temperature = temperature
        self._max_tokens = max_tokens

        if abs(sum(self._weights.values()) - 1.0) > 1e-6:
            raise ValueError("dimension_weights must sum to 1.0")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        output: Any,  # CreativeOutput
        rubric: dict[str, Any] | None = None,
        knowledge_context: list[dict[str, Any]] | None = None,
    ) -> Any:  # returns GradeResult
        """
        Run full LLM evaluation of a creative output.

        Parameters
        ----------
        output : CreativeOutput
            The creative artefact to evaluate.
        rubric : dict | None
            Grading rubric with per-dimension guidance.
        knowledge_context : list[dict] | None
            Relevant knowledge base entries to include as context.

        Returns
        -------
        GradeResult
            Rich evaluation result with per-dimension scores and reasoning.
        """
        from src.evaluation import GradeResult

        t0 = time.perf_counter()

        prompt = self._build_prompt(output, rubric, knowledge_context or [])
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional creative evaluator specialising in "
                    "brand-aligned advertising and visual content. Score outputs "
                    "rigorously and provide actionable feedback."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            raw = self._client.chat_structured(
                messages=messages,
                schema=_LLM_GATE_SCHEMA,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except Exception as exc:
            logger.error("[LLMGate] LLM evaluation failed: %s", exc)
            # Return a zero-score result rather than crashing
            return GradeResult(
                overall_score=0.0,
                dimension_scores={},
                reasoning={},
                novel_finding="",
                suggestions=[f"Evaluation failed: {exc}"],
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("[LLMGate] Evaluation complete. time_ms=%.0f", elapsed_ms)

        return self._parse_result(raw)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        output: Any,
        rubric: dict[str, Any] | None,
        knowledge_context: list[dict[str, Any]],
    ) -> str:
        """Build the structured evaluation prompt."""
        lines: list[str] = []

        # Content reference
        if output.output_path:
            lines.append(f"Content to evaluate: {output.output_path}")
        if output.text_description:
            lines.append(f"Creative description: {output.text_description}")

        lines.append("")
        lines.append("Evaluate this creative output on three dimensions (score 0.0–1.0 each):")
        lines.append("  1. originality       — freshness, non-repetitiveness, creative boldness")
        lines.append("  2. brand_alignment   — consistency with brand guidelines, tone, visual identity")
        lines.append("  3. technical_quality — execution quality (composition, clarity, craft, typography)")

        # Rubric
        if rubric:
            lines.append("")
            lines.append("Grading rubric:")
            for k, v in rubric.items():
                lines.append(f"  {k}: {v}")
        else:
            lines.append("")
            lines.append("Rubric: Apply professional creative industry standards.")

        # Knowledge context
        if knowledge_context:
            lines.append("")
            lines.append("Relevant knowledge from past experiments:")
            for i, entry in enumerate(knowledge_context, 1):
                tier = entry.get("tier", "?").upper()
                conf = entry.get("confidence", 0.0)
                content = entry.get("content", "")
                lines.append(f"  [{i}] [{tier} | conf={conf:.2f}] {content}")

        # Output spec
        lines.append("")
        lines.append(
            "Respond with a JSON object matching the required schema. "
            "Include concrete, actionable suggestions for improvement."
        )

        return "\n".join(lines)

    def _parse_result(self, raw: dict[str, Any]) -> Any:
        """Parse raw LLM output into a GradeResult."""
        from src.evaluation import GradeResult

        dimension_scores: dict[str, float] = {}
        reasoning: dict[str, str] = {}

        for dim in ("originality", "brand_alignment", "technical_quality"):
            dim_data = raw.get(dim, {})
            score = float(dim_data.get("score", 0.0))
            score = max(0.0, min(1.0, score))
            dimension_scores[dim] = score
            reasoning[dim] = dim_data.get("reasoning", "")

        # Weighted overall score
        overall = sum(
            dimension_scores.get(dim, 0.0) * weight
            for dim, weight in self._weights.items()
        )
        overall = round(overall, 4)

        return GradeResult(
            overall_score=overall,
            dimension_scores=dimension_scores,
            reasoning=reasoning,
            novel_finding=raw.get("novel_finding") or "",
            suggestions=raw.get("suggestions") or [],
        )
