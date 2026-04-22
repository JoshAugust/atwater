"""
src.agents.diversity_guard — Diversity Guard.

Responsibility
--------------
Prevent stagnation and asset overuse by monitoring trial history and
injecting forced exploration when concentration thresholds are breached.

Rules (from architecture spec)
-------------------------------
1. If any single asset appears in >30% of the last 50 trials → flag for rotation.
2. Every 50 cycles → force one fully random trial (RandomSampler for that trial).
3. Track which combos have been tested vs total possible space.
4. Can deprecate overused assets in shared state (``asset_status``).

State contract
--------------
Reads:  asset_usage_counts, deprecation_threshold
Writes: asset_status

Output (AgentResult.output)
---------------------------
A dict with:
    flagged_assets     : list[str]  — assets exceeding concentration threshold
    force_random       : bool       — whether a random trial should be injected
    coverage_ratio     : float      — fraction of search space explored
    recommendations    : list[str]  — human-readable action suggestions
    asset_status_updates : dict     — per-asset status changes
"""

from __future__ import annotations

import logging
from typing import Any

import optuna

from src.optimization import get_asset_usage
from src.agents.base import AgentBase, AgentContext, AgentResult

logger = logging.getLogger(__name__)

# Thresholds
CONCENTRATION_THRESHOLD: float = 0.30   # flag if asset > 30% of recent window
RECENT_WINDOW: int = 50                  # number of recent trials to examine
RANDOM_EXPLORATION_INTERVAL: int = 50   # force random trial every N cycles


class DiversityGuard(AgentBase):
    """
    Diversity Guard — detects and corrects stagnation in the trial distribution.

    Parameters
    ----------
    study : optuna.Study
        Active Optuna study used for reading trial history.
    concentration_threshold : float
        Asset concentration ratio above which an asset is flagged (default 0.30).
    recent_window : int
        Number of recent trials to examine for concentration (default 50).
    random_interval : int
        Force a random exploration trial every N cycles (default 50).
    """

    def __init__(
        self,
        study: optuna.Study,
        concentration_threshold: float = CONCENTRATION_THRESHOLD,
        recent_window: int = RECENT_WINDOW,
        random_interval: int = RANDOM_EXPLORATION_INTERVAL,
    ) -> None:
        self._study = study
        self._concentration_threshold = concentration_threshold
        self._recent_window = recent_window
        self._random_interval = random_interval

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "DiversityGuard"

    @property
    def role(self) -> str:
        return "diversity_guard"

    @property
    def readable_state_keys(self) -> list[str]:
        return ["asset_usage_counts", "deprecation_threshold"]

    @property
    def writable_state_keys(self) -> list[str]:
        return ["asset_status"]

    # ------------------------------------------------------------------
    # Core execute
    # ------------------------------------------------------------------

    def execute(self, context: AgentContext) -> AgentResult:
        """
        Run diversity health check.

        Steps:
        1. Pull recent trial history from Optuna.
        2. Check per-asset concentration in the last ``recent_window`` trials.
        3. Determine if a forced-random cycle is due.
        4. Estimate search-space coverage.
        5. Build asset_status updates for flagged/deprecated assets.
        6. Return recommendations.
        """
        trials = self._study.trials
        total_cycles = len(trials)

        # Asset usage via analytics module
        asset_usage: dict[str, dict[str, Any]] = self._get_asset_usage_summary()

        # Concentration analysis over recent window
        recent_trials = trials[-self._recent_window:] if len(trials) >= self._recent_window else trials
        flagged_assets, concentration_map = self._check_concentration(recent_trials)

        # Force-random decision
        force_random = self._should_force_random(total_cycles)

        # Coverage ratio
        coverage_ratio = self._estimate_coverage(trials, context.scoped_state)

        # Build asset_status updates
        asset_status_updates = self._build_status_updates(
            flagged_assets,
            context.scoped_state.get("deprecation_threshold", 0.50),
            concentration_map,
        )

        # Human-readable recommendations
        recommendations = self._build_recommendations(
            flagged_assets, force_random, coverage_ratio, total_cycles
        )

        state_updates = {"asset_status": asset_status_updates}
        self.validate_state_writes(state_updates)

        logger.info(
            "[DiversityGuard] flagged=%s force_random=%s coverage=%.2f%%",
            flagged_assets,
            force_random,
            coverage_ratio * 100,
        )

        return AgentResult(
            output={
                "flagged_assets": flagged_assets,
                "force_random": force_random,
                "coverage_ratio": coverage_ratio,
                "concentration_map": concentration_map,
                "recommendations": recommendations,
                "asset_status_updates": asset_status_updates,
                "total_cycles": total_cycles,
            },
            state_updates=state_updates,
            knowledge_writes=[],
            score=None,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_asset_usage_summary(self) -> dict[str, dict[str, Any]]:
        """
        Use the analytics module to get per-asset usage.
        Returns an empty dict on failure so the guard never crashes.
        """
        try:
            return get_asset_usage(self._study)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[DiversityGuard] Failed to fetch asset usage: %s", exc)
            return {}

    def _check_concentration(
        self, recent_trials: list[optuna.trial.FrozenTrial]
    ) -> tuple[list[str], dict[str, float]]:
        """
        For each parameter dimension, calculate per-value frequency in recent trials.
        Returns flagged asset names and the full concentration map.
        """
        if not recent_trials:
            return [], {}

        # Collect all param keys
        param_keys: set[str] = set()
        for t in recent_trials:
            param_keys.update(t.params.keys())

        concentration_map: dict[str, float] = {}
        flagged: list[str] = []

        for key in param_keys:
            values = [t.params[key] for t in recent_trials if key in t.params]
            if not values:
                continue
            n = len(values)
            # Count frequencies
            freq: dict[Any, int] = {}
            for v in values:
                freq[v] = freq.get(v, 0) + 1
            for value, count in freq.items():
                ratio = count / n
                asset_id = f"{key}={value}"
                concentration_map[asset_id] = ratio
                if ratio > self._concentration_threshold:
                    flagged.append(asset_id)
                    logger.info(
                        "[DiversityGuard] Asset '%s' appears in %.1f%% of recent %d trials — flagged.",
                        asset_id,
                        ratio * 100,
                        n,
                    )

        return flagged, concentration_map

    def _should_force_random(self, total_cycles: int) -> bool:
        """Return True if we're at a random-exploration milestone."""
        if total_cycles == 0:
            return False
        return total_cycles % self._random_interval == 0

    def _estimate_coverage(
        self,
        trials: list[optuna.trial.FrozenTrial],
        scoped_state: dict[str, Any],
    ) -> float:
        """
        Rough coverage ratio: unique param combos tested / estimated total space.
        Falls back to 0.0 if the search space size cannot be determined.
        """
        if not trials:
            return 0.0

        tested_combos: set[tuple] = set()
        for t in trials:
            combo = tuple(sorted(t.params.items()))
            tested_combos.add(combo)

        # If the state carries a search space size hint, use it
        total_space: int | None = scoped_state.get("search_space_size")
        if total_space and total_space > 0:
            return min(1.0, len(tested_combos) / total_space)

        # Fallback: estimate from cardinalities seen in trial history
        param_values: dict[str, set] = {}
        for t in trials:
            for k, v in t.params.items():
                param_values.setdefault(k, set()).add(v)

        estimated_space = 1
        for values in param_values.values():
            estimated_space *= len(values)

        return min(1.0, len(tested_combos) / max(estimated_space, 1))

    def _build_status_updates(
        self,
        flagged_assets: list[str],
        deprecation_threshold: float,
        concentration_map: dict[str, float],
    ) -> dict[str, str]:
        """
        Build per-asset status strings.
        - "deprecated" if concentration > deprecation_threshold
        - "flagged"    if concentration > concentration_threshold
        - "healthy"    otherwise (only for assets in concentration_map)
        """
        status: dict[str, str] = {}
        for asset_id, ratio in concentration_map.items():
            if ratio > deprecation_threshold:
                status[asset_id] = "deprecated"
            elif asset_id in flagged_assets:
                status[asset_id] = "flagged"
            else:
                status[asset_id] = "healthy"
        return status

    def _build_recommendations(
        self,
        flagged_assets: list[str],
        force_random: bool,
        coverage_ratio: float,
        total_cycles: int,
    ) -> list[str]:
        """Build human-readable action recommendations."""
        recs: list[str] = []

        if flagged_assets:
            recs.append(
                f"Rotate or deprecate overused assets: {', '.join(flagged_assets)}"
            )

        if force_random:
            recs.append(
                f"Cycle {total_cycles} hits the random-exploration interval "
                f"({self._random_interval}). Inject a RandomSampler trial."
            )

        if coverage_ratio < 0.10:
            recs.append(
                f"Search space coverage is very low ({coverage_ratio:.1%}). "
                "Consider broadening exploration before exploiting best combos."
            )
        elif coverage_ratio > 0.80:
            recs.append(
                f"Search space coverage is high ({coverage_ratio:.1%}). "
                "Focusing on refinement is appropriate."
            )

        if not recs:
            recs.append("Diversity health is good. No intervention needed.")

        return recs
