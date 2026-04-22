"""
src.agents.base — Abstract base interface for all Atwater agents.

Every agent in the Atwater architecture inherits from AgentBase.
Agents follow the READ → DECIDE → WRITE protocol:
- Read from context (working memory, scoped state, knowledge, Optuna)
- Decide via their specific logic
- Return AgentResult (output + state updates + knowledge writes + score)

IMPORTANT: Agents never call LLMs directly. They prepare prompts and return
them inside AgentResult.output or as a structured prompt key, for the
orchestrator to execute.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Context & Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AgentContext:
    """
    All inputs available to an agent for a single turn.

    The orchestrator builds this from three sources:
    - Tier 1 (working memory snapshot)
    - Tier 2 (shared state filtered by agent role)
    - Tier 3 (top-k knowledge entries for this task)
    - Optuna context summary for the current study

    Attributes
    ----------
    working_memory : dict[str, Any]
        Snapshot of ephemeral working memory at the start of this turn.
    scoped_state : dict[str, Any]
        Shared state keys filtered to this agent's readable_state_keys.
    knowledge_entries : list[dict[str, Any]]
        Top-k knowledge entries retrieved for the current task.
        Each entry is a dict representation of a KnowledgeEntry.
    optuna_context : dict[str, Any] | None
        Summary of current Optuna study state: best params, recent trial
        summary, parameter importances, etc.  None if no study is active.
    tools : list[dict[str, Any]]
        Tool schemas available to this agent for this turn.
    """

    working_memory: dict[str, Any] = field(default_factory=dict)
    scoped_state: dict[str, Any] = field(default_factory=dict)
    knowledge_entries: list[dict[str, Any]] = field(default_factory=list)
    optuna_context: dict[str, Any] | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentResult:
    """
    Everything an agent produces in a single turn.

    Attributes
    ----------
    output : Any
        Primary output.  For most agents this is a dict (e.g. a structured
        scoring report or a proposed hypothesis).  For creator-type agents it
        is a dict with a ``prompt`` key containing the LLM prompt to execute.
    state_updates : dict[str, Any]
        Key-value pairs to write to shared state after this turn.
        Keys must be within the agent's writable_state_keys.
    knowledge_writes : list[dict[str, Any]]
        Knowledge entries to persist.  Each item must contain at minimum:
        ``content``, ``tier``, ``confidence``, ``topic_cluster``.
    score : float | None
        Numeric score to report to Optuna (0.0–1.0).  Only the Grader
        populates this; all other agents leave it as None.
    """

    output: Any = None
    state_updates: dict[str, Any] = field(default_factory=dict)
    knowledge_writes: list[dict[str, Any]] = field(default_factory=list)
    score: float | None = None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class AgentBase(ABC):
    """
    Abstract base class for all Atwater agents.

    Subclasses must implement:
    - name (property)
    - role (property)
    - readable_state_keys (property)
    - writable_state_keys (property)
    - execute(context) -> AgentResult

    All state reads/writes are validated against the declared key lists at
    runtime to catch permission violations early.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable agent name (e.g. 'DirectorEngine')."""

    @property
    @abstractmethod
    def role(self) -> str:
        """
        Machine role identifier used for state scoping
        (e.g. 'director', 'creator', 'grader').
        """

    # ------------------------------------------------------------------
    # State key declarations
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def readable_state_keys(self) -> list[str]:
        """Shared-state keys this agent is allowed to read."""

    @property
    @abstractmethod
    def writable_state_keys(self) -> list[str]:
        """Shared-state keys this agent is allowed to write."""

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------

    @abstractmethod
    def execute(self, context: AgentContext) -> AgentResult:
        """
        Run the agent for one turn.

        Parameters
        ----------
        context : AgentContext
            Pre-assembled context from the orchestrator.

        Returns
        -------
        AgentResult
            Structured result containing output, state updates, knowledge
            writes, and an optional score.
        """

    # ------------------------------------------------------------------
    # Helpers shared by all agents
    # ------------------------------------------------------------------

    def validate_state_writes(self, updates: dict[str, Any]) -> None:
        """
        Raise ValueError if any key in *updates* is not in writable_state_keys.
        Call this inside execute() before returning state_updates.
        """
        forbidden = [k for k in updates if k not in self.writable_state_keys]
        if forbidden:
            raise ValueError(
                f"[{self.name}] Attempted to write to forbidden state keys: {forbidden}. "
                f"Allowed: {self.writable_state_keys}"
            )

    def validate_knowledge_write(self, entry: dict[str, Any]) -> None:
        """Raise ValueError if a knowledge write dict is missing required fields."""
        required = {"content", "tier", "confidence", "topic_cluster"}
        missing = required - entry.keys()
        if missing:
            raise ValueError(
                f"[{self.name}] Knowledge write missing required fields: {missing}"
            )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} role={self.role!r}>"
