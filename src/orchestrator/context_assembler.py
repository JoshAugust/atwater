"""
context_assembler.py — Per-turn prompt assembly for Atwater agents.

The ContextAssembler is responsible for deciding WHAT each agent sees on
every turn. It pulls together all context sources, filters by role, and
packs them into a token-budget-aware AgentContext.

Design principles
-----------------
- **Recency > relevance**: the most actionable context goes at the END of the
  prompt, right before the task instruction. Small models anchor on recency.
- **Role scoping**: shared state is pre-filtered. Agents only see what the
  architecture permits.
- **Token budget**: a rough estimate gates aggressive trimming so we never
  blow the model's context window.
- **Lazy loading**: the sentence-transformers model for tool selection is
  loaded on first use only.

AgentContext and AgentResult
-----------------------------
These are the primary data-transfer types between orchestrator and agents.
Defined here because the orchestrator owns the assembly contract; agent
implementations import from here (or from src.orchestrator).
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from typing import Any

import optuna

from src.memory import KnowledgeBase, KnowledgeEntry, SharedState
from src.optimization import get_best_params

# ---------------------------------------------------------------------------
# Type constants
# ---------------------------------------------------------------------------

# Approximate tokens per character for rough budget estimation.
# GPT-family tokenisers average ~4 chars/token; 3.8 is slightly conservative.
_CHARS_PER_TOKEN: float = 3.8

# Default context window budget in tokens. Override per-instance.
DEFAULT_TOKEN_BUDGET: int = 2048

# How many knowledge entries to inject per agent turn.
KNOWLEDGE_TOP_K: int = 3

# Role → readable state keys (mirrors the architecture spec table).
ROLE_READ_KEYS: dict[str, list[str]] = {
    "director": ["current_hypothesis", "historical_success_rates"],
    "creator": ["current_hypothesis", "last_successful_layout"],
    "grader": ["output_path", "grading_rubric"],
    "diversity_guard": ["asset_usage_counts", "deprecation_threshold"],
    "orchestrator": [],  # reads ALL — handled via SharedState.state_read_scoped
}

# Base system prompt injected for every agent. Keep tight.
_BASE_SYSTEM_PROMPT: str = textwrap.dedent("""\
    You are part of the Atwater cognitive agent architecture.
    Follow the READ → DECIDE → WRITE protocol for every turn.
    Be precise. Output valid JSON for tool calls and state writes.
    Never skip a step. If context is missing, request it explicitly.
""").strip()

# Agent-specific instruction blocks (~100-200 tokens each).
_AGENT_INSTRUCTIONS: dict[str, str] = {
    "director": textwrap.dedent("""\
        ROLE: Director Engine
        Goal: Select the next best parameter combination using Optuna.
        - ALWAYS use trial.suggest_*() — never manually pick combos.
        - Read current_hypothesis and historical_success_rates from state.
        - Write proposed_hypothesis to state after deciding.
        - You may fix specific params only when testing a knowledge-base rule.
        Output: {"proposed_hypothesis": {...params...}}
    """).strip(),

    "creator": textwrap.dedent("""\
        ROLE: Creator
        Goal: Execute content generation for the proposed parameter combo.
        - Read current_hypothesis from state.
        - Apply relevant knowledge patterns to your generation approach.
        - After generating, perform a self-critique against knowledge base rules.
        - If your approach is novel AND likely to score well, flag suggest_knowledge_write.
        Output: {"output_path": "...", "self_critique": "...", "suggest_knowledge_write": bool}
    """).strip(),

    "grader": textwrap.dedent("""\
        ROLE: Grader Engine
        Goal: Evaluate quality and produce structured, reasoned scores.
        - Read output_path and grading_rubric from state.
        - Score each dimension with explicit reasoning — no bare numbers.
        - You are the primary driver of knowledge writes.
        - Flag novel_finding if you discover something the knowledge base doesn't cover.
        Output: {"overall_score": float, "dimensions": {...}, "novel_finding": str|null, "suggest_knowledge_write": bool}
    """).strip(),

    "diversity_guard": textwrap.dedent("""\
        ROLE: Diversity Guard
        Goal: Prevent stagnation and detect asset overuse.
        - Read asset_usage_counts and deprecation_threshold from state.
        - If any asset appears in >30% of the last 50 trials: flag for rotation.
        - Every 50 cycles, trigger forced_exploration = true.
        Output: {"asset_status": {...}, "diversity_alerts": [...], "forced_exploration": bool}
    """).strip(),

    "orchestrator": textwrap.dedent("""\
        ROLE: Orchestrator
        Goal: Flow control, context filtering, and protocol enforcement.
        - You have read access to all shared state.
        - Decide what each downstream agent needs to see.
        - Write workflow_state and next_agent to shared state.
        Output: {"workflow_state": "...", "next_agent": "..."}
    """).strip(),

    "consolidator": textwrap.dedent("""\
        ROLE: Consolidator
        Goal: Knowledge compaction and tier management.
        - Pull knowledge entries in the same topic cluster.
        - Cross-reference with Optuna statistics.
        - Merge overlapping entries into higher-tier summaries.
        - Promote observations (statistical backing) → patterns.
        - Promote patterns (200+ trial validation) → rules.
        - Archive stale entries not validated in 200 cycles.
        Output: {"promotions": [...], "merges": [...], "archives": [...]}
    """).strip(),
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class AgentContext:
    """
    Fully assembled context package passed to an agent on each turn.

    Assembling this is the orchestrator's primary job. Every field maps
    to one section of the per-turn prompt structure from ARCHITECTURE.md.

    Attributes:
        role: Agent role name (e.g. "director", "grader").
        system_prompt: The complete system prompt string, ready to send.
        state_snapshot: Role-filtered shared state dict.
        knowledge_entries: Top-K knowledge entries formatted as strings.
        optuna_summary: Best params + recent trial summary from Optuna.
        tool_schemas: Full JSON schemas for the selected tool group.
        task_instruction: The specific instruction for this turn.
        cycle_number: Current production cycle index.
        token_estimate: Rough token count for the full assembled context.
        metadata: Optional extra data for debugging/logging.
    """

    role: str
    system_prompt: str
    state_snapshot: dict[str, Any]
    knowledge_entries: list[str]
    optuna_summary: dict[str, Any]
    tool_schemas: list[dict[str, Any]]
    task_instruction: str
    cycle_number: int
    token_estimate: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_prompt_sections(self) -> list[str]:
        """
        Render context into ordered prompt sections.

        Ordering follows recency > relevance: static/structural content first,
        most actionable content last (right before the task instruction).

        Returns:
            List of formatted string sections. Concatenate with "\\n\\n" to
            build the full prompt.
        """
        sections: list[str] = []

        # 1. Base system prompt (static)
        sections.append(f"## System\n{self.system_prompt}")

        # 2. Agent-specific instructions
        agent_instr = _AGENT_INSTRUCTIONS.get(self.role, "")
        if agent_instr:
            sections.append(f"## Role Instructions\n{agent_instr}")

        # 3. Tool schemas (structural, loads before dynamic context)
        if self.tool_schemas:
            schemas_json = json.dumps(self.tool_schemas, indent=2)
            sections.append(f"## Available Tools\n```json\n{schemas_json}\n```")

        # 4. Optuna context (statistical grounding)
        if self.optuna_summary:
            optuna_text = json.dumps(self.optuna_summary, indent=2)
            sections.append(f"## Optuna Context\n```json\n{optuna_text}\n```")

        # 5. Knowledge entries (recency-biased — near the end)
        if self.knowledge_entries:
            kb_block = "\n".join(
                f"[{i+1}] {entry}" for i, entry in enumerate(self.knowledge_entries)
            )
            sections.append(f"## Relevant Knowledge\n{kb_block}")

        # 6. Scoped shared state (most recent, highest priority — near end)
        if self.state_snapshot:
            state_json = json.dumps(self.state_snapshot, indent=2)
            sections.append(
                f"## Current State (Cycle {self.cycle_number})\n"
                f"```json\n{state_json}\n```"
            )

        # 7. Task instruction (last — the anchor)
        sections.append(f"## Task\n{self.task_instruction}")

        return sections

    def render(self) -> str:
        """Render full prompt as a single string."""
        return "\n\n".join(self.to_prompt_sections())


@dataclass
class AgentResult:
    """
    Output returned by an agent after processing an AgentContext.

    Attributes:
        role: Agent role that produced this result.
        output: Parsed output dict (the agent's JSON response).
        raw_text: Raw text output from the agent (pre-parse).
        knowledge_write_requested: True if the agent flagged a novel finding.
        cycle_number: Cycle this result belongs to.
        success: False if the agent errored or produced unparseable output.
        error: Error description if success is False.
    """

    role: str
    output: dict[str, Any]
    raw_text: str
    knowledge_write_requested: bool
    cycle_number: int
    success: bool = True
    error: str | None = None


# ---------------------------------------------------------------------------
# ContextAssembler
# ---------------------------------------------------------------------------

class ContextAssembler:
    """
    Assembles per-turn AgentContext packages for each agent in the pipeline.

    Responsibilities:
    - Filter shared state by agent role
    - Retrieve semantically relevant knowledge entries (top K)
    - Build Optuna summary (best params + dimension stats)
    - Select and inject tool schemas for the relevant group
    - Enforce token budget by trimming lower-priority sections

    Args:
        shared_state: The live SharedState instance.
        knowledge_base: The live KnowledgeBase instance.
        study: The active Optuna study.
        tool_loader: Optional ToolLoader. If None, a default one is created.
        token_budget: Maximum token budget per assembled context.
    """

    def __init__(
        self,
        shared_state: SharedState,
        knowledge_base: KnowledgeBase,
        study: optuna.Study,
        tool_loader: "ToolLoader | None" = None,  # forward ref
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> None:
        self._state = shared_state
        self._kb = knowledge_base
        self._study = study
        self._token_budget = token_budget

        # Import here to avoid circular — ToolLoader lives in same package.
        if tool_loader is None:
            from .tool_loader import ToolLoader
            self._tool_loader = ToolLoader()
        else:
            self._tool_loader = tool_loader

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assemble_context(
        self,
        agent_role: str,
        task_description: str,
        cycle_number: int,
    ) -> AgentContext:
        """
        Assemble a complete AgentContext for the given agent and turn.

        Steps:
        1. Load scoped shared state for this role.
        2. Retrieve top-K knowledge entries relevant to the task.
        3. Build Optuna context summary.
        4. Select and load tool schemas for this task.
        5. Estimate total tokens; trim if over budget.

        Args:
            agent_role: The agent's role name (e.g. "director", "grader").
            task_description: The task instruction for this turn.
            cycle_number: Current production cycle index.

        Returns:
            A fully assembled AgentContext ready to render and send.
        """
        # Step 1: Role-scoped shared state.
        state_snapshot = self._state.state_read_scoped(agent_role)

        # Step 2: Relevant knowledge entries.
        knowledge_entries = self._fetch_knowledge(task_description, agent_role)

        # Step 3: Optuna summary.
        optuna_summary = self._build_optuna_summary()

        # Step 4: Tool schemas for the most relevant tool group.
        tool_schemas = self._load_tools(task_description)

        # Step 5: Estimate tokens and trim if needed.
        token_estimate = self._estimate_tokens(
            state_snapshot=state_snapshot,
            knowledge_entries=knowledge_entries,
            optuna_summary=optuna_summary,
            tool_schemas=tool_schemas,
            task_description=task_description,
        )

        # Trim knowledge entries if over budget.
        while token_estimate > self._token_budget and len(knowledge_entries) > 1:
            knowledge_entries = knowledge_entries[:-1]
            token_estimate = self._estimate_tokens(
                state_snapshot=state_snapshot,
                knowledge_entries=knowledge_entries,
                optuna_summary=optuna_summary,
                tool_schemas=tool_schemas,
                task_description=task_description,
            )

        # Trim tool schemas if still over budget.
        while token_estimate > self._token_budget and len(tool_schemas) > 1:
            tool_schemas = tool_schemas[:-1]
            token_estimate = self._estimate_tokens(
                state_snapshot=state_snapshot,
                knowledge_entries=knowledge_entries,
                optuna_summary=optuna_summary,
                tool_schemas=tool_schemas,
                task_description=task_description,
            )

        return AgentContext(
            role=agent_role,
            system_prompt=_BASE_SYSTEM_PROMPT,
            state_snapshot=state_snapshot,
            knowledge_entries=knowledge_entries,
            optuna_summary=optuna_summary,
            tool_schemas=tool_schemas,
            task_instruction=task_description,
            cycle_number=cycle_number,
            token_estimate=token_estimate,
            metadata={
                "tool_group_selected": (
                    self._tool_loader.select_tools(task_description, top_k=1)[0]
                    if task_description else "memory"
                ),
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_knowledge(
        self,
        task_description: str,
        agent_role: str,
    ) -> list[str]:
        """
        Retrieve top-K knowledge entries relevant to the task.

        Returns formatted strings (tier + content) rather than raw
        KnowledgeEntry objects so they're prompt-ready.
        """
        if not task_description.strip():
            return []

        try:
            entries: list[KnowledgeEntry] = self._kb.knowledge_read(
                query=task_description,
                max_results=KNOWLEDGE_TOP_K,
            )
        except Exception:
            # Knowledge base may be empty or unavailable — soft failure.
            return []

        formatted: list[str] = []
        for entry in entries:
            tier_label = f"[{entry.tier.upper()}]"
            confidence_label = f"(confidence={entry.confidence:.2f})"
            formatted.append(f"{tier_label} {confidence_label} {entry.content}")

        return formatted

    def _build_optuna_summary(self) -> dict[str, Any]:
        """
        Build a compact Optuna context dict for injection into the prompt.

        Includes best params and a short trial count / recent score summary.
        Returns an empty dict if no trials have completed yet.
        """
        try:
            best = get_best_params(self._study)
        except Exception:
            best = {}

        trials = self._study.trials
        completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]

        summary: dict[str, Any] = {
            "total_trials": len(trials),
            "completed_trials": len(completed),
        }

        if best:
            summary["best_params"] = best
            summary["best_score"] = self._study.best_value

        if completed:
            recent_n = min(10, len(completed))
            recent_scores = [t.value for t in completed[-recent_n:] if t.value is not None]
            if recent_scores:
                summary["recent_mean_score"] = round(
                    sum(recent_scores) / len(recent_scores), 4
                )
                summary["recent_best_score"] = round(max(recent_scores), 4)

        return summary

    def _load_tools(self, task_description: str) -> list[dict[str, Any]]:
        """
        Select the best tool group for this task and return its full schemas.

        Falls back to an empty list if task_description is blank or
        the tool loader raises.
        """
        if not task_description.strip():
            return []

        try:
            groups = self._tool_loader.select_tools(task_description, top_k=1)
            if not groups:
                return []
            return self._tool_loader.get_schemas(groups[0])
        except Exception:
            return []

    def _estimate_tokens(
        self,
        state_snapshot: dict[str, Any],
        knowledge_entries: list[str],
        optuna_summary: dict[str, Any],
        tool_schemas: list[dict[str, Any]],
        task_description: str,
    ) -> int:
        """
        Rough token count estimate for the assembled context.

        Uses character count / _CHARS_PER_TOKEN. Includes fixed overhead
        for system prompt and agent instructions (~400 tokens).
        """
        char_count = 0
        char_count += len(_BASE_SYSTEM_PROMPT)
        char_count += len(json.dumps(state_snapshot))
        char_count += sum(len(e) for e in knowledge_entries)
        char_count += len(json.dumps(optuna_summary))
        char_count += len(json.dumps(tool_schemas))
        char_count += len(task_description)

        estimated = int(char_count / _CHARS_PER_TOKEN)
        # Add fixed overhead for agent instructions + section headers.
        return estimated + 400
