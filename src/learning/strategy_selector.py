"""
src.learning.strategy_selector — Thompson Sampling multi-armed bandit.

Selects among four high-level exploration strategies for each cycle using
Thompson Sampling over Beta-distributed posteriors.

Strategies
----------
exploit     Use Optuna's current best parameters as the starting point.
explore     Inject a fully random trial (wide search space).
hypothesis  Test a hypothesis from the knowledge base.
refine      Tweak the best-known params with small perturbations.

Usage
-----
    selector = StrategySelector(save_path="strategy_state.json")
    strategy = selector.select_strategy()      # "exploit" | "explore" | ...
    # ... run cycle with strategy ...
    selector.update(strategy, reward=0.75)     # reward in [0, 1]
    print(selector.get_stats())
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------

STRATEGIES: list[str] = ["exploit", "explore", "hypothesis", "refine"]

# Initial priors: alpha=1, beta=1 → uniform Beta(1,1). Slightly favour
# exploit and refine on startup by seeding with a small informative prior.
DEFAULT_PRIORS: dict[str, dict[str, float]] = {
    "exploit":    {"alpha": 2.0, "beta": 1.0},   # slight exploit bias at start
    "explore":    {"alpha": 1.0, "beta": 1.0},   # uninformative
    "hypothesis": {"alpha": 1.0, "beta": 1.0},   # uninformative
    "refine":     {"alpha": 1.5, "beta": 1.0},   # mild refinement bias
}


# ---------------------------------------------------------------------------
# Arm state
# ---------------------------------------------------------------------------

@dataclass
class ArmState:
    """Beta distribution parameters for one strategy arm."""
    strategy: str
    alpha: float = 1.0
    beta: float = 1.0
    total_pulls: int = 0
    total_reward: float = 0.0

    @property
    def win_rate(self) -> float:
        """Expected value of the Beta posterior (mean = alpha / (alpha + beta))."""
        return self.alpha / (self.alpha + self.beta)

    def sample(self, rng: np.random.Generator | None = None) -> float:
        """Draw a Thompson sample from Beta(alpha, beta)."""
        if rng is not None:
            return float(rng.beta(self.alpha, self.beta))
        return float(np.random.beta(self.alpha, self.beta))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArmState":
        return cls(**data)


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------

class StrategySelector:
    """
    Multi-armed bandit strategy selector using Thompson Sampling.

    Parameters
    ----------
    strategies:
        List of strategy names (default: STRATEGIES).
    priors:
        Dict of {strategy: {"alpha": float, "beta": float}} initial values.
    save_path:
        Path to persist arm state as JSON.  Pass ``None`` to disable.
    seed:
        Random seed for reproducibility (optional).
    """

    def __init__(
        self,
        strategies: list[str] | None = None,
        priors: dict[str, dict[str, float]] | None = None,
        save_path: str | Path | None = None,
        seed: int | None = None,
    ) -> None:
        self._strategies = strategies or STRATEGIES
        self._priors = priors or DEFAULT_PRIORS
        self._save_path = Path(save_path) if save_path else None
        self._rng = np.random.default_rng(seed)

        self._arms: dict[str, ArmState] = {}

        # Try to load persisted state; fall back to fresh priors
        if self._save_path and self._save_path.exists():
            self._load()
        else:
            self._initialise_arms()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_strategy(self) -> str:
        """
        Thompson sample from all arms and return the strategy with the
        highest draw.

        Returns
        -------
        str
            One of: "exploit", "explore", "hypothesis", "refine".
        """
        samples = {
            name: arm.sample(self._rng)
            for name, arm in self._arms.items()
        }
        chosen = max(samples, key=lambda k: samples[k])
        logger.debug(
            "[StrategySelector] Samples: %s → chosen: %s",
            {k: f"{v:.3f}" for k, v in samples.items()},
            chosen,
        )
        return chosen

    def update(self, strategy: str, reward: float) -> None:
        """
        Update the Beta posterior for the chosen strategy.

        Parameters
        ----------
        strategy:
            The strategy that was used this cycle.
        reward:
            Cycle score in [0, 1].  Values above 0.5 count as a "win"
            with fractional credit proportional to magnitude.
        """
        if strategy not in self._arms:
            logger.warning(
                "[StrategySelector] Unknown strategy '%s' — skipping update.",
                strategy,
            )
            return

        arm = self._arms[strategy]
        arm.total_pulls += 1
        arm.total_reward += reward

        # Bernoulli-like update: reward is the success probability for this
        # pull.  For Thompson Sampling on continuous rewards we use:
        #   alpha += reward
        #   beta  += (1 - reward)
        arm.alpha += reward
        arm.beta += (1.0 - reward)

        logger.debug(
            "[StrategySelector] Updated '%s': alpha=%.2f beta=%.2f (reward=%.3f)",
            strategy, arm.alpha, arm.beta, reward,
        )

        if self._save_path:
            self._save()

    def get_stats(self) -> dict[str, Any]:
        """
        Return a dict of strategy statistics.

        Returns
        -------
        dict
            {strategy: {"alpha", "beta", "win_rate", "total_pulls", "total_reward"}}
        """
        return {
            name: {
                "alpha": arm.alpha,
                "beta": arm.beta,
                "win_rate": arm.win_rate,
                "total_pulls": arm.total_pulls,
                "total_reward": arm.total_reward,
            }
            for name, arm in self._arms.items()
        }

    def best_strategy(self) -> str:
        """Return the strategy with the highest expected win rate (for reporting)."""
        return max(self._arms, key=lambda k: self._arms[k].win_rate)

    def reset(self) -> None:
        """Reset all arms to their initial priors."""
        self._initialise_arms()
        if self._save_path:
            self._save()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """Persist arm state to JSON."""
        data = {name: arm.to_dict() for name, arm in self._arms.items()}
        try:
            self._save_path.write_text(json.dumps(data, indent=2))  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            logger.warning("[StrategySelector] Failed to save state: %s", exc)

    def _load(self) -> None:
        """Load arm state from JSON."""
        try:
            data = json.loads(self._save_path.read_text())  # type: ignore[union-attr]
            self._arms = {
                name: ArmState.from_dict(arm_data)
                for name, arm_data in data.items()
            }
            # Ensure all strategies are represented (handles new strategies added later)
            for strategy in self._strategies:
                if strategy not in self._arms:
                    prior = self._priors.get(strategy, {"alpha": 1.0, "beta": 1.0})
                    self._arms[strategy] = ArmState(
                        strategy=strategy,
                        alpha=prior["alpha"],
                        beta=prior["beta"],
                    )
            logger.info("[StrategySelector] Loaded state from %s.", self._save_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[StrategySelector] Failed to load state (%s) — starting fresh.",
                exc,
            )
            self._initialise_arms()

    def _initialise_arms(self) -> None:
        """Set up arms from priors."""
        self._arms = {}
        for strategy in self._strategies:
            prior = self._priors.get(strategy, {"alpha": 1.0, "beta": 1.0})
            self._arms[strategy] = ArmState(
                strategy=strategy,
                alpha=prior["alpha"],
                beta=prior["beta"],
            )
