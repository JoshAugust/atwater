"""
src.evaluation.process_rewards — Per-step scoring for process reward models.

Instead of scoring only the final creative output (outcome reward), this module
scores at each step in the generation pipeline:
  1. brief     — brief parsing / understanding quality
  2. concept   — creative concept strength
  3. execution — execution fidelity to concept
  4. polish    — final polish / refinement quality

Per-step scoring provides a richer gradient signal to Optuna, enabling faster
convergence than outcome-only scoring.

Reference:
  Lightman et al., "Let's Verify Step by Step" (arXiv:2305.20050, 2023)

Public API:
  ProcessRewardScorer.score_step(step_name, step_output, context) -> StepScore
  ProcessRewardScorer.aggregate(step_scores) -> float
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step configuration
# ---------------------------------------------------------------------------

# Step names and default weights.  Must sum to 1.0.
STEP_NAMES: list[str] = ["brief", "concept", "execution", "polish"]

_DEFAULT_STEP_WEIGHTS: dict[str, float] = {
    "brief":     0.20,
    "concept":   0.30,
    "execution": 0.35,
    "polish":    0.15,
}

# Per-step evaluation criteria (used in LLM prompt).
_STEP_CRITERIA: dict[str, str] = {
    "brief": (
        "Evaluate the quality of the brief interpretation: "
        "Did the agent correctly identify the core objective, target audience, "
        "key message, and constraints? Score on completeness and accuracy."
    ),
    "concept": (
        "Evaluate the creative concept: "
        "Is it original and non-generic? Does it strategically serve the brief? "
        "Is the central idea bold and memorable? Score on creative strength."
    ),
    "execution": (
        "Evaluate execution fidelity: "
        "Does the output faithfully realise the concept? Are visual/copy choices "
        "aligned with the concept's intent? Score on fidelity and craft."
    ),
    "polish": (
        "Evaluate the polish and refinement: "
        "Are details attended to (typography, spacing, colour accuracy)? "
        "Would this output pass professional QA? Score on finish quality."
    ),
}

# Schema for per-step LLM evaluation
_STEP_EVAL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "score": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Step quality score 0.0–1.0.",
        },
        "reasoning": {
            "type": "string",
            "minLength": 10,
            "description": "Explanation for this step score.",
        },
        "issues": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific issues found at this step; empty if none.",
        },
    },
    "required": ["score", "reasoning", "issues"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# ProcessRewardScorer
# ---------------------------------------------------------------------------


class ProcessRewardScorer:
    """
    Scores creative generation at each pipeline step.

    Uses the LMStudioClient to evaluate step output quality, providing a
    finer-grained reward signal than outcome-only grading.

    Parameters
    ----------
    client : LMStudioClient | None
        LM Studio client.  If None, all steps are scored with a heuristic
        fallback (0.5 for any non-empty output, 0.0 for empty).
    step_weights : dict[str, float] | None
        Per-step weights for aggregate scoring.  Must sum to 1.0.
    temperature : float
        LLM sampling temperature for step evaluations.
    """

    def __init__(
        self,
        client: Any | None = None,
        step_weights: dict[str, float] | None = None,
        temperature: float = 0.2,
    ) -> None:
        self._client = client
        self._weights = step_weights or _DEFAULT_STEP_WEIGHTS.copy()
        self._temperature = temperature

        if abs(sum(self._weights.values()) - 1.0) > 1e-6:
            raise ValueError("step_weights must sum to 1.0")

        # Validate step names
        unknown = set(self._weights) - set(STEP_NAMES)
        if unknown:
            raise ValueError(f"Unknown step names in weights: {unknown}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_step(
        self,
        step_name: str,
        step_output: str,
        context: dict[str, Any] | None = None,
    ) -> Any:  # returns StepScore
        """
        Score a single pipeline step.

        Parameters
        ----------
        step_name : str
            One of: "brief", "concept", "execution", "polish".
        step_output : str
            The agent's output for this step (text or path description).
        context : dict | None
            Optional context.  Recognised keys:
                brief          (str) — original creative brief
                previous_steps (dict) — previous step outputs for context

        Returns
        -------
        StepScore
        """
        from src.evaluation import StepScore

        if step_name not in STEP_NAMES:
            raise ValueError(f"Unknown step '{step_name}'. Must be one of: {STEP_NAMES}")

        if not step_output or not step_output.strip():
            logger.warning("[ProcessRewardScorer] Empty output for step '%s'", step_name)
            return StepScore(
                step_name=step_name,
                score=0.0,
                reasoning="Step produced no output.",
                issues=["Empty step output"],
            )

        # Use LLM evaluation if client is available
        if self._client is not None:
            return self._llm_score_step(step_name, step_output, context or {})

        # Fallback: heuristic scoring
        return self._heuristic_score_step(step_name, step_output)

    def aggregate(self, step_scores: list[Any]) -> float:  # list[StepScore]
        """
        Compute weighted aggregate score from a list of step scores.

        Parameters
        ----------
        step_scores : list[StepScore]

        Returns
        -------
        float
            Weighted sum of step scores, clamped to [0.0, 1.0].
        """
        if not step_scores:
            return 0.0

        total = 0.0
        total_weight = 0.0

        for step_score in step_scores:
            weight = self._weights.get(step_score.step_name, 0.0)
            total += step_score.score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return max(0.0, min(1.0, total / total_weight))

    def score_all_steps(
        self,
        step_outputs: dict[str, str],
        context: dict[str, Any] | None = None,
    ) -> tuple[list[Any], float]:
        """
        Score all provided steps and return (step_scores, aggregate_score).

        Parameters
        ----------
        step_outputs : dict[step_name → output_text]
            Mapping of step name to step output text.
        context : dict | None
            Shared context passed to each step scorer.

        Returns
        -------
        (list[StepScore], float)
        """
        step_scores = []
        ctx = context or {}
        accumulated_ctx = dict(ctx)
        previous_steps: dict[str, str] = {}

        for step_name in STEP_NAMES:
            if step_name not in step_outputs:
                continue
            accumulated_ctx["previous_steps"] = previous_steps
            score = self.score_step(step_name, step_outputs[step_name], accumulated_ctx)
            step_scores.append(score)
            previous_steps[step_name] = step_outputs[step_name]

        aggregate = self.aggregate(step_scores)
        return step_scores, aggregate

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _llm_score_step(
        self,
        step_name: str,
        step_output: str,
        context: dict[str, Any],
    ) -> Any:
        """Score a step using LLM evaluation."""
        from src.evaluation import StepScore

        criteria = _STEP_CRITERIA[step_name]
        brief = context.get("brief", "(no brief provided)")
        previous_steps = context.get("previous_steps", {})

        prompt_lines: list[str] = [
            f"You are evaluating the '{step_name}' step of a creative generation pipeline.",
            "",
            f"Step criteria: {criteria}",
            "",
            f"Original brief: {brief}",
        ]

        if previous_steps:
            prompt_lines.append("")
            prompt_lines.append("Previous step outputs:")
            for prev_name, prev_output in previous_steps.items():
                truncated = prev_output[:500] + "..." if len(prev_output) > 500 else prev_output
                prompt_lines.append(f"  [{prev_name}]: {truncated}")

        prompt_lines.extend([
            "",
            f"This step's output ({step_name}):",
            step_output[:1000] + ("..." if len(step_output) > 1000 else ""),
            "",
            "Score this step output on the stated criteria (0.0–1.0). "
            "List any specific issues. Be concise and direct.",
        ])

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precision evaluator for AI-generated creative content. "
                    "Score objectively. Be strict — 0.5 is average, not good."
                ),
            },
            {"role": "user", "content": "\n".join(prompt_lines)},
        ]

        try:
            raw = self._client.chat_structured(
                messages=messages,
                schema=_STEP_EVAL_SCHEMA,
                temperature=self._temperature,
                max_tokens=512,
            )
            score = max(0.0, min(1.0, float(raw.get("score", 0.5))))
            return StepScore(
                step_name=step_name,
                score=score,
                reasoning=raw.get("reasoning", ""),
                issues=raw.get("issues") or [],
            )

        except Exception as exc:
            logger.warning(
                "[ProcessRewardScorer] LLM scoring for step '%s' failed: %s",
                step_name, exc,
            )
            # Fallback to heuristic
            return self._heuristic_score_step(step_name, step_output)

    def _heuristic_score_step(
        self,
        step_name: str,
        step_output: str,
    ) -> Any:
        """
        Simple heuristic scoring when no LLM client is available.

        Provides a rough baseline:
        - Empty output → 0.0
        - Very short output → 0.2
        - Minimal output → 0.4
        - Decent length → 0.5 (neutral baseline)
        """
        from src.evaluation import StepScore

        if not step_output or not step_output.strip():
            return StepScore(
                step_name=step_name,
                score=0.0,
                reasoning="Heuristic: empty output.",
                issues=["Empty step output"],
            )

        word_count = len(step_output.split())
        if word_count < 5:
            score = 0.2
            reasoning = f"Heuristic: very short output ({word_count} words)."
        elif word_count < 20:
            score = 0.4
            reasoning = f"Heuristic: short output ({word_count} words)."
        else:
            score = 0.5
            reasoning = f"Heuristic: output present ({word_count} words). No LLM client for detailed scoring."

        return StepScore(
            step_name=step_name,
            score=score,
            reasoning=reasoning,
            issues=[] if score >= 0.4 else ["Output may be too brief"],
        )
