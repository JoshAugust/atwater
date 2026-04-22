"""
src.agents.creator — Creator Agent.

Responsibility
--------------
Translate a parameter hypothesis into a content generation prompt, then run
a self-critique against the knowledge base to assess novelty and quality.

IMPORTANT: This agent does NOT call an LLM directly.  It returns a structured
dict in AgentResult.output with a ``prompt`` key.  The orchestrator is
responsible for submitting that prompt to the configured LLM client and feeding
the response back for the self-critique phase.

Two-phase design
----------------
Phase 1 — Generation:
    Build the generation prompt from the hypothesis + relevant knowledge.
    Return it as output for the orchestrator to execute.

Phase 2 — Self-critique (called separately via execute_critique):
    Given the LLM's generated content, evaluate it against KB patterns.
    If the approach was novel AND shows promise → suggest a knowledge write.

State contract
--------------
Reads:  current_hypothesis, last_successful_layout
Writes: output_path, self_critique
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents.base import AgentBase, AgentContext, AgentResult

logger = logging.getLogger(__name__)


class CreatorAgent(AgentBase):
    """
    Creator Agent — builds content generation prompts from hypotheses.

    Parameters
    ----------
    novelty_threshold : float
        Minimum confidence above which a novel finding is worth writing to KB.
        Default 0.70.
    """

    def __init__(self, novelty_threshold: float = 0.70) -> None:
        self._novelty_threshold = novelty_threshold

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "CreatorAgent"

    @property
    def role(self) -> str:
        return "creator"

    @property
    def readable_state_keys(self) -> list[str]:
        return ["current_hypothesis", "last_successful_layout"]

    @property
    def writable_state_keys(self) -> list[str]:
        return ["output_path", "self_critique"]

    # ------------------------------------------------------------------
    # Core execute — Phase 1: build generation prompt
    # ------------------------------------------------------------------

    def execute(self, context: AgentContext) -> AgentResult:
        """
        Phase 1: Build the LLM generation prompt.

        Reads the hypothesis from scoped_state, enriches it with relevant
        knowledge patterns, and returns a structured prompt dict.  The
        orchestrator executes the LLM call and passes the response to
        execute_critique().

        Returns
        -------
        AgentResult
            output: dict with keys:
                ``phase``       : "generation"
                ``prompt``      : str  — the prompt to send to the LLM
                ``hypothesis``  : dict — the parameter combo being tested
                ``kb_context``  : list — knowledge entries provided as context
        """
        hypothesis: dict[str, Any] = context.scoped_state.get("current_hypothesis", {})
        last_layout: str | None = context.scoped_state.get("last_successful_layout")
        knowledge_entries = context.knowledge_entries

        prompt = self._build_generation_prompt(hypothesis, last_layout, knowledge_entries)

        output = {
            "phase": "generation",
            "prompt": prompt,
            "hypothesis": hypothesis,
            "kb_context": knowledge_entries,
        }

        logger.info(
            "[CreatorAgent] Generation prompt assembled for hypothesis: %s",
            hypothesis,
        )

        return AgentResult(
            output=output,
            state_updates={},
            knowledge_writes=[],
            score=None,
        )

    # ------------------------------------------------------------------
    # Phase 2: self-critique (orchestrator calls this after LLM response)
    # ------------------------------------------------------------------

    def execute_critique(
        self,
        context: AgentContext,
        generated_content: str,
        hypothesis: dict[str, Any],
    ) -> AgentResult:
        """
        Phase 2: Evaluate the generated content against the knowledge base.

        Builds a self-critique prompt for the LLM (the orchestrator executes
        it) and decides whether this approach is novel enough to suggest a
        knowledge write.

        Parameters
        ----------
        context : AgentContext
            Same context used in Phase 1 (may have updated knowledge entries).
        generated_content : str
            The raw content produced by the LLM in Phase 1.
        hypothesis : dict[str, Any]
            The parameter combo that was tested.

        Returns
        -------
        AgentResult
            output: dict with keys:
                ``phase``          : "critique"
                ``prompt``         : str  — self-critique prompt for the LLM
                ``generated_content`` : str
            state_updates: sets ``self_critique`` in shared state.
            knowledge_writes: populated if the approach is deemed novel.
        """
        knowledge_entries = context.knowledge_entries
        critique_prompt = self._build_critique_prompt(
            generated_content, hypothesis, knowledge_entries
        )

        is_novel = self._assess_novelty(hypothesis, knowledge_entries)
        knowledge_writes: list[dict[str, Any]] = []

        if is_novel:
            kw = {
                "content": (
                    f"Novel approach tested: {hypothesis}. "
                    f"Generated content preview: {generated_content[:200]}"
                ),
                "tier": "observation",
                "confidence": self._novelty_threshold,
                "topic_cluster": "creator_novelty",
                "metadata": {"hypothesis": hypothesis, "source": "creator_self_critique"},
            }
            self.validate_knowledge_write(kw)
            knowledge_writes.append(kw)
            logger.info("[CreatorAgent] Novel approach detected — suggesting KB write.")

        state_updates = {
            "self_critique": {
                "is_novel": is_novel,
                "critique_prompt": critique_prompt,
                "hypothesis": hypothesis,
            }
        }
        self.validate_state_writes(state_updates)

        return AgentResult(
            output={
                "phase": "critique",
                "prompt": critique_prompt,
                "generated_content": generated_content,
            },
            state_updates=state_updates,
            knowledge_writes=knowledge_writes,
            score=None,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_generation_prompt(
        self,
        hypothesis: dict[str, Any],
        last_layout: str | None,
        knowledge_entries: list[dict[str, Any]],
    ) -> str:
        """Construct the LLM generation prompt from hypothesis + KB context."""
        kb_block = self._format_knowledge_entries(knowledge_entries)
        last_layout_hint = (
            f"\nReference layout from last successful run: {last_layout}"
            if last_layout
            else ""
        )

        return (
            "You are a professional content creator. Generate content for the following "
            "parameter combination:\n\n"
            f"Parameters:\n{self._format_dict(hypothesis)}\n"
            f"{last_layout_hint}\n\n"
            "Relevant knowledge from past experiments:\n"
            f"{kb_block}\n\n"
            "Instructions:\n"
            "1. Produce content that fits the parameter combination exactly.\n"
            "2. Apply relevant patterns from the knowledge base where appropriate.\n"
            "3. If you deviate from established patterns, note why.\n"
            "4. Output format: structured content with a brief rationale."
        )

    def _build_critique_prompt(
        self,
        generated_content: str,
        hypothesis: dict[str, Any],
        knowledge_entries: list[dict[str, Any]],
    ) -> str:
        """Construct the self-critique prompt."""
        kb_block = self._format_knowledge_entries(knowledge_entries)

        return (
            "Review the following generated content against established knowledge patterns.\n\n"
            f"Generated content:\n{generated_content}\n\n"
            f"Parameter combination used:\n{self._format_dict(hypothesis)}\n\n"
            "Established knowledge patterns:\n"
            f"{kb_block}\n\n"
            "Self-critique questions:\n"
            "1. Does this content comply with the established rules and patterns?\n"
            "2. Is there anything novel or unexpected in this approach?\n"
            "3. What is your confidence that this will score well?\n"
            "4. Are there any quality concerns?\n"
            "Respond with a structured critique."
        )

    def _assess_novelty(
        self,
        hypothesis: dict[str, Any],
        knowledge_entries: list[dict[str, Any]],
    ) -> bool:
        """
        Heuristic: an approach is 'novel' if no existing knowledge entry
        references the same parameter combination in its metadata.
        """
        for entry in knowledge_entries:
            existing_params = entry.get("metadata", {}).get("hypothesis", {})
            if existing_params == hypothesis:
                return False  # Already documented
        return True  # No exact match found — counts as novel

    @staticmethod
    def _format_dict(d: dict[str, Any]) -> str:
        return "\n".join(f"  {k}: {v}" for k, v in d.items()) if d else "  (none)"

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
