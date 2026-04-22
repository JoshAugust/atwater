"""
src.agents.director — Director Engine.

Responsibility
--------------
Select the next parameter combination for the production cycle.

Strategy
--------
- 80% of the time: delegate fully to Optuna (TPE sampler) via TrialAdapter.
- 20% of the time: override with a knowledge-base hypothesis to actively
  test a qualitative insight that Optuna hasn't explored yet.

State contract
--------------
Reads:  current_hypothesis, historical_success_rates
Writes: proposed_hypothesis

Output (AgentResult.output)
---------------------------
A dict with:
    trial_number   : int   — Optuna trial number
    params         : dict  — suggested parameter combination
    source         : str   — "optuna" | "knowledge_override"
    override_entry : dict  — knowledge entry used (if source == "knowledge_override")
"""

from __future__ import annotations

import logging
import random
from typing import Any

import optuna

from src.optimization import TrialAdapter, get_best_params
from src.agents.base import AgentBase, AgentContext, AgentResult

logger = logging.getLogger(__name__)

# Fraction of cycles where the director overrides Optuna with a KB hypothesis.
KNOWLEDGE_OVERRIDE_RATE: float = 0.20


class DirectorEngine(AgentBase):
    """
    Director Engine — picks the next parameter combination each cycle.

    Parameters
    ----------
    study : optuna.Study
        Active Optuna study.  The director calls study.ask() to create a
        trial and will later expect the grader to call study.tell().
    search_space : dict | None
        Optional custom search space to pass to TrialAdapter.  If None,
        DEFAULT_SEARCH_SPACE from src.optimization is used.
    override_rate : float
        Probability of overriding Optuna with a KB hypothesis (default 0.20).
    """

    def __init__(
        self,
        study: optuna.Study,
        search_space: dict[str, Any] | None = None,
        override_rate: float = KNOWLEDGE_OVERRIDE_RATE,
    ) -> None:
        self._study = study
        self._search_space = search_space
        self._override_rate = override_rate

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "DirectorEngine"

    @property
    def role(self) -> str:
        return "director"

    @property
    def readable_state_keys(self) -> list[str]:
        return ["current_hypothesis", "historical_success_rates"]

    @property
    def writable_state_keys(self) -> list[str]:
        return ["proposed_hypothesis"]

    # ------------------------------------------------------------------
    # Core execute
    # ------------------------------------------------------------------

    def execute(self, context: AgentContext) -> AgentResult:
        """
        Produce the next parameter hypothesis.

        Decision tree:
        1. Roll the override dice.
        2a. If override → find a testable KB hypothesis and inject fixed params.
        2b. Otherwise  → ask Optuna for the next trial via TrialAdapter.
        3. Write proposed_hypothesis to state_updates.
        4. Return structured output.
        """
        knowledge_entries = context.knowledge_entries
        do_override = (
            random.random() < self._override_rate and len(knowledge_entries) > 0
        )

        if do_override:
            result = self._override_from_knowledge(knowledge_entries)
        else:
            result = self._suggest_from_optuna()

        state_updates = {"proposed_hypothesis": result["params"]}
        self.validate_state_writes(state_updates)

        logger.info(
            "[DirectorEngine] Proposed hypothesis (source=%s): %s",
            result["source"],
            result["params"],
        )

        return AgentResult(
            output=result,
            state_updates=state_updates,
            knowledge_writes=[],
            score=None,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _suggest_from_optuna(self) -> dict[str, Any]:
        """Ask Optuna's sampler for the next parameter combination."""
        trial = self._study.ask()
        adapter = TrialAdapter(trial, self._search_space)
        params = adapter.suggest_all()
        return {
            "trial_number": trial.number,
            "params": params,
            "source": "optuna",
            "override_entry": None,
        }

    def _override_from_knowledge(
        self, knowledge_entries: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Select a knowledge-base hypothesis to test.

        Strategy: prefer rule-tier entries (highest confidence / most
        validated), then pattern-tier.  Pick a random one so we don't
        keep testing the same hypothesis.  Extract any parameter hints
        encoded in the entry's metadata and pass them to Optuna as fixed
        params; Optuna still fills the remainder.
        """
        # Rank: rules > patterns > observations
        tier_rank = {"rule": 0, "pattern": 1, "observation": 2}
        sorted_entries = sorted(
            knowledge_entries,
            key=lambda e: (tier_rank.get(e.get("tier", "observation"), 2), -e.get("confidence", 0.0)),
        )

        # Pick from top-3 at random to add variety
        candidates = sorted_entries[:3]
        chosen = random.choice(candidates)

        # Extract parameter hints from metadata if present
        param_hints: dict[str, Any] = chosen.get("metadata", {}).get("params", {})

        # Ask Optuna to fill remaining params (partial fixed params)
        trial = self._study.ask(fixed_distributions=self._build_fixed_distributions(param_hints))
        adapter = TrialAdapter(trial, self._search_space)
        params = adapter.suggest_all()

        # Overwrite with the KB hints so the specific combo is actually tested
        params.update(param_hints)

        return {
            "trial_number": trial.number,
            "params": params,
            "source": "knowledge_override",
            "override_entry": chosen,
        }

    def _build_fixed_distributions(
        self, param_hints: dict[str, Any]
    ) -> dict[str, optuna.distributions.BaseDistribution]:
        """
        Convert raw param hints to Optuna CategoricalDistribution so they
        can be passed to study.ask(fixed_distributions=...).

        Non-categorical hints are silently dropped (they'll be overwritten
        manually after suggestion anyway).
        """
        fixed: dict[str, optuna.distributions.BaseDistribution] = {}
        for key, value in param_hints.items():
            fixed[key] = optuna.distributions.CategoricalDistribution(choices=[value])
        return fixed
