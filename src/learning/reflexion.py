"""
src.learning.reflexion — Reflexion pattern (verbal reinforcement learning).

Based on: Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement
Learning" (arXiv:2303.11366, NeurIPS 2023).

Pattern
-------
After each creative generation cycle, the ReflexionEngine:
1. Examines the CycleResult (score, params, outputs, errors).
2. Calls an LLM to produce a structured verbal reflection.
3. Stores the reflection as a knowledge-base observation.
4. Maintains a sliding window of the last N reflections.
5. Exposes build_director_context() so the next cycle's Director
   can receive the reflection summary as additional context.

Usage
-----
    engine = ReflexionEngine(knowledge_base=kb, llm_client=client)
    reflection = await engine.generate_reflection(cycle_result, knowledge_context)
    director_context = engine.build_director_context()
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class Reflection:
    """Structured output of a single cycle's verbal reflection.

    Attributes
    ----------
    cycle_number:
        The cycle index this reflection covers.
    what_worked:
        Observations about strategies / parameters that contributed to a
        high score this cycle.
    what_failed:
        Observations about what did not help or actively hurt quality.
    hypotheses:
        Causal hypotheses worth testing in future cycles (e.g. "increasing
        temperature may help when scores plateau").
    next_actions:
        Concrete, actionable suggestions for the next Director decision.
    score:
        Numeric score from the cycle (stored for context).
    raw_llm_response:
        The unstructured LLM text before parsing (useful for debugging).
    knowledge_entry_id:
        ID of the KnowledgeBase entry written for this reflection.
    """

    cycle_number: int
    what_worked: list[str] = field(default_factory=list)
    what_failed: list[str] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    score: float | None = None
    raw_llm_response: str = ""
    knowledge_entry_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_number": self.cycle_number,
            "score": self.score,
            "what_worked": self.what_worked,
            "what_failed": self.what_failed,
            "hypotheses": self.hypotheses,
            "next_actions": self.next_actions,
            "knowledge_entry_id": self.knowledge_entry_id,
        }

    def to_context_string(self) -> str:
        """Compact human-readable form for Director context injection."""
        lines = [f"## Cycle {self.cycle_number} Reflection (score={self.score})"]
        if self.what_worked:
            lines.append("✓ Worked: " + "; ".join(self.what_worked))
        if self.what_failed:
            lines.append("✗ Failed: " + "; ".join(self.what_failed))
        if self.hypotheses:
            lines.append("? Hypotheses: " + "; ".join(self.hypotheses))
        if self.next_actions:
            lines.append("→ Next: " + "; ".join(self.next_actions))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ReflexionEngine:
    """
    Verbal reinforcement learning via structured reflection.

    Parameters
    ----------
    knowledge_base:
        Any object with a ``knowledge_write(content, tier, confidence,
        topic_cluster)`` method.  Pass ``None`` to skip KB storage
        (useful in tests).
    llm_client:
        Any object with a ``chat(messages) -> str`` coroutine *or* regular
        method.  The engine calls ``llm_client.chat(messages)`` synchronously;
        wrap async clients with asyncio.run() or pass a mock.  Pass ``None``
        to use fallback rule-based reflection only.
    window_size:
        Number of recent reflections to keep in the sliding window (default 5).
    model:
        Model name passed through to the LLM client (optional, ignored if
        client handles it internally).
    """

    def __init__(
        self,
        knowledge_base: Any = None,
        llm_client: Any = None,
        window_size: int = 5,
        model: str = "claude-sonnet-4-6",
    ) -> None:
        self._kb = knowledge_base
        self._llm = llm_client
        self._window_size = window_size
        self._model = model
        self._window: deque[Reflection] = deque(maxlen=window_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_reflection(
        self,
        cycle_result: dict[str, Any],
        knowledge_context: str = "",
    ) -> Reflection:
        """
        Produce a structured Reflection for one cycle.

        Parameters
        ----------
        cycle_result:
            Dict with at minimum: ``cycle_number`` (int), ``score`` (float),
            ``params`` (dict), ``outputs`` (list[str]), ``errors`` (list[str]).
        knowledge_context:
            Relevant knowledge snippets retrieved from the KB (optional).

        Returns
        -------
        Reflection
            Structured reflection dataclass, also stored in KB and window.
        """
        cycle_number: int = cycle_result.get("cycle_number", 0)
        score: float | None = cycle_result.get("score")

        messages = self.build_reflection_prompt(
            cycle_result, list(self._window)
        )

        raw_response = ""
        reflection: Reflection

        if self._llm is not None:
            try:
                raw_response = self._call_llm(messages)
                reflection = self._parse_llm_response(
                    raw_response, cycle_number, score
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[ReflexionEngine] LLM call failed (cycle %d): %s — falling back to rule-based.",
                    cycle_number,
                    exc,
                )
                reflection = self._rule_based_reflection(cycle_result)
        else:
            reflection = self._rule_based_reflection(cycle_result)

        reflection.raw_llm_response = raw_response

        # Store in KB
        if self._kb is not None:
            kb_id = self._write_to_kb(reflection, cycle_result)
            reflection.knowledge_entry_id = kb_id

        # Add to sliding window
        self._window.append(reflection)

        logger.info(
            "[ReflexionEngine] Cycle %d reflection stored (score=%.3f, kb_id=%s).",
            cycle_number,
            score or 0.0,
            reflection.knowledge_entry_id,
        )

        return reflection

    def build_reflection_prompt(
        self,
        cycle_result: dict[str, Any],
        previous_reflections: list[Reflection],
    ) -> list[dict[str, str]]:
        """
        Build the messages list for the LLM reflection call.

        Parameters
        ----------
        cycle_result:
            Current cycle data (score, params, outputs, errors).
        previous_reflections:
            Recent reflections to include as context.

        Returns
        -------
        list[dict]
            OpenAI-style messages: [{"role": ..., "content": ...}, ...]
        """
        system_prompt = (
            "You are a self-reflection module for an AI creative optimisation agent "
            "called Atwater. Your job is to produce a concise, structured reflection "
            "after each generation cycle to guide the next cycle's decisions.\n\n"
            "Output ONLY valid JSON with these keys:\n"
            "  what_worked: list[str]   — observations about what helped\n"
            "  what_failed: list[str]   — observations about what hurt or failed\n"
            "  hypotheses:  list[str]   — causal hypotheses to test next\n"
            "  next_actions: list[str]  — concrete Director decisions to try next\n\n"
            "Be specific and actionable. Avoid vague generalities."
        )

        # Summarise previous reflections
        prev_context = ""
        if previous_reflections:
            prev_summaries = "\n".join(
                r.to_context_string() for r in previous_reflections[-3:]
            )
            prev_context = f"\n\nPREVIOUS REFLECTIONS (last {len(previous_reflections)}):\n{prev_summaries}"

        # Summarise cycle result
        cycle_summary = self._format_cycle_result(cycle_result)

        user_content = (
            f"CYCLE RESULT:\n{cycle_summary}"
            f"{prev_context}\n\n"
            "Produce your structured JSON reflection now."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def build_director_context(self) -> str:
        """
        Return a formatted string suitable for injection into the Director's
        next-cycle prompt.  Includes the most recent N reflections from the
        sliding window.
        """
        if not self._window:
            return ""

        sections = ["=== REFLEXION MEMORY ==="]
        for r in self._window:
            sections.append(r.to_context_string())
        sections.append("=== END REFLEXION ===")
        return "\n\n".join(sections)

    def get_window(self) -> list[Reflection]:
        """Return a copy of the current sliding window."""
        return list(self._window)

    def clear_window(self) -> None:
        """Clear the sliding window (useful between sessions)."""
        self._window.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """Call the LLM client. Supports both sync and async clients."""
        import asyncio
        result = self._llm.chat(messages)
        if asyncio.iscoroutine(result):
            result = asyncio.get_event_loop().run_until_complete(result)
        return str(result)

    def _parse_llm_response(
        self,
        raw: str,
        cycle_number: int,
        score: float | None,
    ) -> Reflection:
        """Parse JSON from LLM response into a Reflection dataclass."""
        # Strip markdown fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines if not line.startswith("```")
            ).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON block
            import re
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                raise ValueError(f"Cannot parse JSON from LLM response: {text[:200]}")

        return Reflection(
            cycle_number=cycle_number,
            score=score,
            what_worked=data.get("what_worked", []),
            what_failed=data.get("what_failed", []),
            hypotheses=data.get("hypotheses", []),
            next_actions=data.get("next_actions", []),
        )

    def _rule_based_reflection(
        self,
        cycle_result: dict[str, Any],
    ) -> Reflection:
        """
        Fallback reflection using simple heuristics — no LLM required.
        Generates generic but useful observations from cycle metrics alone.
        """
        cycle_number = cycle_result.get("cycle_number", 0)
        score = cycle_result.get("score")
        params = cycle_result.get("params", {})
        errors = cycle_result.get("errors", [])

        what_worked: list[str] = []
        what_failed: list[str] = []
        hypotheses: list[str] = []
        next_actions: list[str] = []

        if score is not None:
            if score >= 0.8:
                what_worked.append(f"High score ({score:.2f}) achieved this cycle.")
                next_actions.append("Exploit params from this cycle — high performer.")
            elif score >= 0.6:
                what_worked.append(f"Moderate score ({score:.2f}) — room to improve.")
                hypotheses.append("Refining the top param may push score above 0.8.")
                next_actions.append("Try small perturbations around current params.")
            else:
                what_failed.append(f"Low score ({score:.2f}) — current params underperform.")
                hypotheses.append("Current param region may be a local minimum.")
                next_actions.append("Explore a different region of parameter space.")

        if errors:
            what_failed.extend(f"Error: {e}" for e in errors[:3])
            next_actions.append("Investigate and resolve reported errors before next cycle.")

        if params:
            next_actions.append(
                f"Review top params for cycle {cycle_number}: "
                + ", ".join(f"{k}={v}" for k, v in list(params.items())[:5])
            )

        return Reflection(
            cycle_number=cycle_number,
            score=score,
            what_worked=what_worked,
            what_failed=what_failed,
            hypotheses=hypotheses,
            next_actions=next_actions,
        )

    def _write_to_kb(
        self,
        reflection: Reflection,
        cycle_result: dict[str, Any],
    ) -> str | None:
        """Write the reflection as a knowledge base observation. Returns entry ID."""
        content = (
            f"[Cycle {reflection.cycle_number} Reflection | score={reflection.score}] "
            f"Worked: {'; '.join(reflection.what_worked) or 'none'}. "
            f"Failed: {'; '.join(reflection.what_failed) or 'none'}. "
            f"Hypotheses: {'; '.join(reflection.hypotheses) or 'none'}. "
            f"Next: {'; '.join(reflection.next_actions) or 'none'}."
        )
        try:
            entry_id = self._kb.knowledge_write(
                content=content,
                tier="observation",
                confidence=0.6,
                topic_cluster="reflexion",
            )
            return entry_id
        except Exception as exc:  # noqa: BLE001
            logger.warning("[ReflexionEngine] Failed to write to KB: %s", exc)
            return None

    @staticmethod
    def _format_cycle_result(cycle_result: dict[str, Any]) -> str:
        """Format a cycle result dict as a readable string for the LLM."""
        lines = []
        for key in ("cycle_number", "score", "params", "outputs", "errors"):
            val = cycle_result.get(key, "N/A")
            if isinstance(val, dict):
                val = json.dumps(val, indent=2)
            elif isinstance(val, list) and len(val) > 5:
                val = val[:5] + [f"... ({len(val) - 5} more)"]
            lines.append(f"  {key}: {val}")
        return "\n".join(lines)
