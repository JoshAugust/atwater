"""
src.agents — All agent implementations for the Atwater cognitive architecture.

Agents follow the READ → DECIDE → WRITE protocol and never call LLMs directly.
They prepare prompts and return them in AgentResult for the orchestrator to
execute.

Quick reference
---------------
    from src.agents import (
        AgentBase,
        AgentContext,
        AgentResult,
        DirectorEngine,
        CreatorAgent,
        GraderEngine,
        DiversityGuard,
        ConsolidatorAgent,
    )

    # All agents implement:
    #   agent.execute(context: AgentContext) -> AgentResult

Agent roles and state contracts
--------------------------------
| Agent              | Reads                                        | Writes                          |
|--------------------|----------------------------------------------|---------------------------------|
| DirectorEngine     | current_hypothesis, historical_success_rates | proposed_hypothesis             |
| CreatorAgent       | current_hypothesis, last_successful_layout   | output_path, self_critique      |
| GraderEngine       | output_path, grading_rubric                  | score, structured_analysis      |
| DiversityGuard     | asset_usage_counts, deprecation_threshold    | asset_status                    |
| ConsolidatorAgent  | (reads from Optuna + KB via context)         | (all writes via knowledge_writes)|
"""

from src.agents.base import AgentBase, AgentContext, AgentResult
from src.agents.consolidator_agent import ConsolidatorAgent
from src.agents.creator import CreatorAgent
from src.agents.director import DirectorEngine
from src.agents.diversity_guard import DiversityGuard
from src.agents.grader import GraderEngine

__all__ = [
    # Base interface
    "AgentBase",
    "AgentContext",
    "AgentResult",
    # Concrete agents
    "DirectorEngine",
    "CreatorAgent",
    "GraderEngine",
    "DiversityGuard",
    "ConsolidatorAgent",
]
