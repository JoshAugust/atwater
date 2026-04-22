"""
flow_controller.py — Agent sequencing and production cycle management.

The FlowController decides WHEN agents run and what to do with their results.
It owns the full evolution loop from ARCHITECTURE.md:

    Director → Creator → Grader → DiversityGuard → (Consolidator every N cycles)

Design notes
------------
- FlowController is deterministic about sequencing; agents are pluggable.
- Each agent slot accepts an ``AgentRunner`` callable — a function that takes
  an ``AgentContext`` and returns an ``AgentResult``.
- Stub runners are provided by default so the architecture compiles and runs
  even before real LLM agents are wired in.
- The controller handles Optuna trial lifecycle (ask/tell) on behalf of the
  director and grader roles.
- Knowledge writes triggered by the grader are executed here, not in the agent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import optuna

from src.memory import KnowledgeBase, SharedState
from src.optimization import (
    DEFAULT_SEARCH_SPACE,
    SearchSpace,
    TrialAdapter,
    get_asset_usage,
)

from .context_assembler import AgentContext, AgentResult, ContextAssembler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# An AgentRunner is any callable that processes a context and returns a result.
AgentRunner = Callable[[AgentContext], AgentResult]

# Agents in execution order.
PIPELINE_ROLES: tuple[str, ...] = (
    "director",
    "creator",
    "grader",
    "diversity_guard",
)


# ---------------------------------------------------------------------------
# CycleResult
# ---------------------------------------------------------------------------

@dataclass
class CycleResult:
    """
    Summary of a completed production cycle.

    Attributes:
        cycle_number: The cycle index that was run.
        params_used: Parameter dict suggested by Optuna for this cycle.
        score: Final score reported to Optuna (None if grader failed).
        knowledge_writes: IDs of knowledge entries written this cycle.
        diversity_alerts: Any asset rotation warnings from the diversity guard.
        consolidated: True if the consolidator ran this cycle.
        success: False if any critical agent step failed.
        errors: Map of role → error message for failed steps.
    """

    cycle_number: int
    params_used: dict[str, Any]
    score: float | None
    knowledge_writes: list[str]
    diversity_alerts: list[str]
    consolidated: bool
    success: bool = True
    errors: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Default stub runners
# ---------------------------------------------------------------------------

def _stub_director(ctx: AgentContext) -> AgentResult:
    """
    Stub director: returns a placeholder hypothesis.

    In production, replace with an LLM-backed runner that calls
    trial.suggest_*() via the TrialAdapter and writes to shared state.
    """
    logger.debug("[director-stub] No real agent wired. Returning empty hypothesis.")
    return AgentResult(
        role="director",
        output={"proposed_hypothesis": {}},
        raw_text="[stub]",
        knowledge_write_requested=False,
        cycle_number=ctx.cycle_number,
        success=True,
    )


def _stub_creator(ctx: AgentContext) -> AgentResult:
    """Stub creator: returns a placeholder output path."""
    logger.debug("[creator-stub] No real agent wired. Returning stub output.")
    return AgentResult(
        role="creator",
        output={
            "output_path": f"/tmp/atwater/output_cycle_{ctx.cycle_number}.png",
            "self_critique": "stub",
            "suggest_knowledge_write": False,
        },
        raw_text="[stub]",
        knowledge_write_requested=False,
        cycle_number=ctx.cycle_number,
        success=True,
    )


def _stub_grader(ctx: AgentContext) -> AgentResult:
    """Stub grader: returns a fixed 0.5 score."""
    logger.debug("[grader-stub] No real agent wired. Returning 0.5 score.")
    return AgentResult(
        role="grader",
        output={
            "overall_score": 0.5,
            "dimensions": {},
            "novel_finding": None,
            "suggest_knowledge_write": False,
        },
        raw_text="[stub]",
        knowledge_write_requested=False,
        cycle_number=ctx.cycle_number,
        success=True,
    )


def _stub_diversity_guard(ctx: AgentContext) -> AgentResult:
    """Stub diversity guard: no alerts."""
    logger.debug("[diversity-guard-stub] No real agent wired. No alerts.")
    return AgentResult(
        role="diversity_guard",
        output={
            "asset_status": {},
            "diversity_alerts": [],
            "forced_exploration": False,
        },
        raw_text="[stub]",
        knowledge_write_requested=False,
        cycle_number=ctx.cycle_number,
        success=True,
    )


def _stub_consolidator(ctx: AgentContext) -> AgentResult:
    """Stub consolidator: no-op compaction."""
    logger.debug("[consolidator-stub] No real agent wired. Skipping compaction.")
    return AgentResult(
        role="consolidator",
        output={"promotions": [], "merges": [], "archives": []},
        raw_text="[stub]",
        knowledge_write_requested=False,
        cycle_number=ctx.cycle_number,
        success=True,
    )


DEFAULT_RUNNERS: dict[str, AgentRunner] = {
    "director": _stub_director,
    "creator": _stub_creator,
    "grader": _stub_grader,
    "diversity_guard": _stub_diversity_guard,
    "consolidator": _stub_consolidator,
}


# ---------------------------------------------------------------------------
# FlowController
# ---------------------------------------------------------------------------

class FlowController:
    """
    Orchestrates the full production cycle: Director → Creator → Grader →
    DiversityGuard → (Consolidator every N cycles).

    The controller:
    - Asks Optuna for the next trial before Director runs.
    - Assembles context for each agent via ContextAssembler.
    - Routes results: grader score → Optuna, knowledge writes → KnowledgeBase,
      diversity alerts → shared state.
    - Runs the Consolidator every ``consolidation_interval`` cycles.

    Args:
        shared_state: Live SharedState instance.
        knowledge_base: Live KnowledgeBase instance.
        study: Active Optuna study.
        search_space: SearchSpace definition for trial.suggest_*() calls.
        context_assembler: ContextAssembler instance.
        agent_runners: Map of role → AgentRunner. Stubs used for missing roles.
        consolidation_interval: How many cycles between consolidation passes.
    """

    def __init__(
        self,
        shared_state: SharedState,
        knowledge_base: KnowledgeBase,
        study: optuna.Study,
        search_space: SearchSpace | None = None,
        context_assembler: ContextAssembler | None = None,
        agent_runners: dict[str, AgentRunner] | None = None,
        consolidation_interval: int = 50,
    ) -> None:
        self._state = shared_state
        self._kb = knowledge_base
        self._study = study
        self._search_space = search_space or DEFAULT_SEARCH_SPACE
        self._consolidation_interval = consolidation_interval

        # ContextAssembler: create default if not provided.
        if context_assembler is None:
            self._assembler = ContextAssembler(
                shared_state=shared_state,
                knowledge_base=knowledge_base,
                study=study,
            )
        else:
            self._assembler = context_assembler

        # Merge provided runners over stubs.
        self._runners: dict[str, AgentRunner] = dict(DEFAULT_RUNNERS)
        if agent_runners:
            self._runners.update(agent_runners)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_cycle(self, cycle_number: int) -> CycleResult:
        """
        Execute one full production cycle.

        Sequence:
        1.  Ask Optuna for next trial (open-loop ask).
        2.  Director — selects hypothesis, writes to shared state.
        3.  Creator   — generates output, writes output_path + self_critique.
        4.  Grader    — scores output; score reported to Optuna study.
        5.  DiversityGuard — checks asset health; alerts written to state.
        6.  Consolidator (conditional) — every N cycles.

        Args:
            cycle_number: 1-based cycle index.

        Returns:
            CycleResult summarising what happened this cycle.
        """
        logger.info("=== Cycle %d starting ===", cycle_number)

        knowledge_writes: list[str] = []
        diversity_alerts: list[str] = []
        errors: dict[str, str] = {}
        params_used: dict[str, Any] = {}
        score: float | None = None
        trial: optuna.Trial | None = None
        consolidated = False

        # ------------------------------------------------------------------
        # Open a new Optuna trial for this cycle.
        # ------------------------------------------------------------------
        try:
            trial = self._study.ask()
            adapter = TrialAdapter(trial)
            params_used = adapter.suggest_params(self._search_space)
            logger.debug("Trial #%d params: %s", trial.number, params_used)

            # Make params available to Director via shared state.
            self._state.state_write(
                "optuna_trial_params",
                params_used,
                roles=["director", "creator"],
            )
        except Exception as exc:
            logger.error("Failed to ask Optuna for trial: %s", exc)
            errors["optuna"] = str(exc)

        # ------------------------------------------------------------------
        # STEP 1: Director — propose hypothesis.
        # ------------------------------------------------------------------
        director_result = self._run_agent(
            role="director",
            task_description=(
                f"Cycle {cycle_number}: Select the next parameter combination to test. "
                f"Optuna has suggested: {params_used}. "
                "Write your proposed_hypothesis to shared state."
            ),
            cycle_number=cycle_number,
            errors=errors,
        )

        if director_result and director_result.success:
            proposed = director_result.output.get("proposed_hypothesis", {})
            if proposed:
                self._state.state_write(
                    "current_hypothesis",
                    proposed,
                    roles=["director", "creator"],
                )
        # ------------------------------------------------------------------
        # STEP 2: Creator — generate output.
        # ------------------------------------------------------------------
        creator_result = self._run_agent(
            role="creator",
            task_description=(
                f"Cycle {cycle_number}: Execute content generation for the current hypothesis. "
                "Write output_path and self_critique to shared state."
            ),
            cycle_number=cycle_number,
            errors=errors,
        )

        if creator_result and creator_result.success:
            output_path = creator_result.output.get("output_path", "")
            self_critique = creator_result.output.get("self_critique", "")
            if output_path:
                self._state.state_write(
                    "output_path",
                    output_path,
                    roles=["creator", "grader"],
                )
            if self_critique:
                self._state.state_write(
                    "self_critique",
                    self_critique,
                    roles=["creator", "grader"],
                )

        # ------------------------------------------------------------------
        # STEP 3: Grader — score output, report to Optuna.
        # ------------------------------------------------------------------
        grader_result = self._run_agent(
            role="grader",
            task_description=(
                f"Cycle {cycle_number}: Evaluate the generated output. "
                "Score each rubric dimension with explicit reasoning. "
                "Report overall_score and any novel_finding."
            ),
            cycle_number=cycle_number,
            errors=errors,
        )

        if grader_result and grader_result.success:
            score = grader_result.output.get("overall_score")

            # Close the Optuna trial loop.
            if trial is not None and score is not None:
                try:
                    TrialAdapter.report_score(self._study, trial, float(score))
                    logger.debug("Reported score %.4f for trial #%d", score, trial.number)
                except Exception as exc:
                    logger.error("Failed to report score to Optuna: %s", exc)
                    errors["optuna_tell"] = str(exc)

            # Write structured analysis to shared state.
            self._state.state_write(
                "score",
                score,
                roles=["grader", "orchestrator"],
            )
            self._state.state_write(
                "structured_analysis",
                grader_result.output,
                roles=["grader", "orchestrator"],
            )

            # Handle knowledge write request from grader.
            if grader_result.output.get("suggest_knowledge_write"):
                novel = grader_result.output.get("novel_finding", "")
                if novel:
                    eid = self._write_observation(
                        content=novel,
                        cycle=cycle_number,
                        score=float(score) if score else 0.0,
                    )
                    if eid:
                        knowledge_writes.append(eid)

        # ------------------------------------------------------------------
        # STEP 4: DiversityGuard — check asset health.
        # ------------------------------------------------------------------
        # Inject current asset usage from Optuna into shared state first.
        try:
            usage = get_asset_usage(self._study, last_n=50)
            self._state.state_write(
                "asset_usage_counts",
                usage,
                roles=["diversity_guard"],
            )
        except Exception as exc:
            logger.warning("Could not compute asset usage: %s", exc)

        dg_result = self._run_agent(
            role="diversity_guard",
            task_description=(
                f"Cycle {cycle_number}: Check asset usage distribution. "
                "Flag any assets exceeding 30% usage in the last 50 trials. "
                "Trigger forced_exploration if cycle number is a multiple of 50."
            ),
            cycle_number=cycle_number,
            errors=errors,
        )

        if dg_result and dg_result.success:
            alerts = dg_result.output.get("diversity_alerts", [])
            diversity_alerts.extend(alerts)
            asset_status = dg_result.output.get("asset_status", {})
            self._state.state_write(
                "asset_status",
                asset_status,
                roles=["diversity_guard", "orchestrator"],
            )

            # Write alerts to knowledge base if any.
            for alert in alerts:
                eid = self._write_observation(
                    content=f"Diversity alert: {alert}",
                    cycle=cycle_number,
                    score=0.0,
                    topic_cluster="diversity",
                )
                if eid:
                    knowledge_writes.append(eid)

        # ------------------------------------------------------------------
        # STEP 5 (conditional): Consolidator every N cycles.
        # ------------------------------------------------------------------
        if self.should_consolidate(cycle_number):
            consolidated = self._run_consolidation(cycle_number, errors)

        # ------------------------------------------------------------------
        # Update orchestrator workflow state.
        # ------------------------------------------------------------------
        self._state.state_write(
            "workflow_state",
            {
                "last_cycle": cycle_number,
                "last_score": score,
                "knowledge_writes": len(knowledge_writes),
                "diversity_alerts": len(diversity_alerts),
            },
            roles=["orchestrator"],
        )

        success = len(errors) == 0
        logger.info(
            "Cycle %d complete — score=%.4f, knowledge_writes=%d, alerts=%d, errors=%d",
            cycle_number,
            score or 0.0,
            len(knowledge_writes),
            len(diversity_alerts),
            len(errors),
        )

        return CycleResult(
            cycle_number=cycle_number,
            params_used=params_used,
            score=score,
            knowledge_writes=knowledge_writes,
            diversity_alerts=diversity_alerts,
            consolidated=consolidated,
            success=success,
            errors=errors,
        )

    def should_consolidate(
        self,
        cycle_number: int,
        interval: int | None = None,
    ) -> bool:
        """
        Determine whether the Consolidator should run this cycle.

        Consolidation runs every ``interval`` cycles (default: the
        instance-level ``consolidation_interval``), starting at cycle 1.

        Args:
            cycle_number: The current cycle index.
            interval: Override the instance-level interval for this check.

        Returns:
            True if consolidation should run this cycle.
        """
        effective_interval = interval if interval is not None else self._consolidation_interval
        return cycle_number > 0 and (cycle_number % effective_interval == 0)

    def register_runner(self, role: str, runner: AgentRunner) -> None:
        """
        Register or replace an AgentRunner for a given role.

        Use this to wire in real LLM-backed agents after init.

        Args:
            role: The agent role (e.g. "director", "grader").
            runner: Callable accepting AgentContext, returning AgentResult.
        """
        self._runners[role] = runner
        logger.info("Registered runner for role: %s", role)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_agent(
        self,
        role: str,
        task_description: str,
        cycle_number: int,
        errors: dict[str, str],
    ) -> AgentResult | None:
        """
        Assemble context and run the agent for a given role.

        Errors are logged and collected in ``errors`` dict; the method
        returns None on failure so the caller can decide whether to abort.
        """
        try:
            ctx = self._assembler.assemble_context(
                agent_role=role,
                task_description=task_description,
                cycle_number=cycle_number,
            )
        except Exception as exc:
            msg = f"Context assembly failed for {role}: {exc}"
            logger.error(msg)
            errors[f"{role}_context"] = msg
            return None

        runner = self._runners.get(role)
        if runner is None:
            logger.warning("No runner registered for role %s — skipping.", role)
            return None

        try:
            result = runner(ctx)
            if not result.success:
                logger.warning("Agent %s reported failure: %s", role, result.error)
                errors[role] = result.error or "agent reported failure"
            return result
        except Exception as exc:
            msg = f"Agent {role} raised an exception: {exc}"
            logger.error(msg)
            errors[role] = msg
            return None

    def _run_consolidation(
        self,
        cycle_number: int,
        errors: dict[str, str],
    ) -> bool:
        """
        Run the consolidator agent and return True if it succeeded.
        """
        logger.info("Running consolidation at cycle %d", cycle_number)
        result = self._run_agent(
            role="consolidator",
            task_description=(
                f"Cycle {cycle_number}: Run knowledge compaction. "
                "Merge overlapping entries, promote observations with statistical "
                "backing, archive stale entries. Report promotions, merges, archives."
            ),
            cycle_number=cycle_number,
            errors=errors,
        )
        return result is not None and result.success

    def _write_observation(
        self,
        content: str,
        cycle: int,
        score: float,
        topic_cluster: str = "general",
    ) -> str | None:
        """
        Write an observation-tier knowledge entry.

        Returns the entry ID on success, or None if the write fails.
        """
        try:
            return self._kb.knowledge_write(
                content=content,
                tier="observation",
                confidence=min(max(score, 0.0), 1.0),
                topic_cluster=topic_cluster,
                cycle=cycle,
                optuna_evidence={"cycle": cycle, "score": score},
            )
        except Exception as exc:
            logger.warning("Knowledge write failed: %s", exc)
            return None
