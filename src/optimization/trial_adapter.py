"""
trial_adapter.py — Bridge between Optuna trials and the shared agent state.

Provides:
- :class:`SearchSpace` — declarative definition of what parameters exist.
- :class:`TrialAdapter` — wraps an Optuna trial, translates SearchSpace
  definitions into ``trial.suggest_*`` calls, and reports scores back.

Usage example::

    from src.optimization.trial_adapter import TrialAdapter, DEFAULT_SEARCH_SPACE

    trial = study.ask()
    adapter = TrialAdapter(trial)
    params = adapter.suggest_params(DEFAULT_SEARCH_SPACE)

    score = run_pipeline(params)
    adapter.report_score(study, score)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import optuna


# ---------------------------------------------------------------------------
# Search-space definition
# ---------------------------------------------------------------------------

@dataclass
class SearchSpace:
    """
    Declarative description of all parameters Optuna should optimise.

    Attributes:
        categorical: Mapping of parameter name → list of allowed string values.
        continuous: Mapping of parameter name → (low, high) float bounds.
        integer: Mapping of parameter name → (low, high[, step]) int bounds.
            The third element (step) is optional; defaults to 1 when omitted.
    """

    categorical: dict[str, list[str]] = field(default_factory=dict)
    continuous: dict[str, tuple[float, float]] = field(default_factory=dict)
    integer: dict[str, tuple[int, int] | tuple[int, int, int]] = field(
        default_factory=dict
    )


# ---------------------------------------------------------------------------
# Default search space (matches OPTUNA_INTEGRATION.md example)
# ---------------------------------------------------------------------------

DEFAULT_SEARCH_SPACE = SearchSpace(
    categorical={
        "background": ["dark", "gradient", "minimal", "textured", "abstract"],
        "layout": ["hero", "split", "grid", "asymmetric", "stacked"],
        "shot": ["front", "angle", "lifestyle", "closeup", "context"],
        "typography": ["sans-modern", "sans-classic", "serif-editorial", "mono"],
    },
    continuous={
        "bg_opacity": (0.2, 1.0),
        "font_scale": (0.8, 1.5),
        "padding_ratio": (0.02, 0.15),
        "contrast_ratio": (3.0, 10.0),
    },
    integer={
        # (low, high, step)
        "hero_font_size": (24, 72, 4),
    },
)


# ---------------------------------------------------------------------------
# Trial adapter
# ---------------------------------------------------------------------------

class TrialAdapter:
    """
    Wraps an :class:`optuna.Trial` with a cleaner interface for the agent layer.

    The adapter translates a :class:`SearchSpace` into the appropriate
    ``trial.suggest_*`` calls and provides a helper to close the loop by
    reporting a score back to the study.

    Args:
        trial: The Optuna trial object returned by ``study.ask()``.
    """

    def __init__(self, trial: optuna.Trial) -> None:
        self._trial = trial

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def suggest_params(self, search_space: SearchSpace) -> dict[str, Any]:
        """
        Ask Optuna to suggest parameter values for this trial.

        Iterates over all three dimension types in the search space and
        calls the corresponding ``trial.suggest_*`` method for each.

        Args:
            search_space: The :class:`SearchSpace` describing all dims.

        Returns:
            A flat ``{param_name: value}`` dict ready to pass to the
            creative pipeline.
        """
        params: dict[str, Any] = {}

        # Categorical dimensions
        for name, choices in search_space.categorical.items():
            params[name] = self._trial.suggest_categorical(name, choices)

        # Continuous (float) dimensions
        for name, bounds in search_space.continuous.items():
            low, high = bounds
            params[name] = self._trial.suggest_float(name, low, high)

        # Integer dimensions
        for name, bounds in search_space.integer.items():
            if len(bounds) == 3:
                low, high, step = bounds  # type: ignore[misc]
                params[name] = self._trial.suggest_int(name, low, high, step=step)
            else:
                low, high = bounds  # type: ignore[misc]
                params[name] = self._trial.suggest_int(name, low, high)

        return params

    @staticmethod
    def report_score(
        study: optuna.Study,
        trial: optuna.Trial,
        score: float,
    ) -> None:
        """
        Report a trial's final score back to the study.

        Closes the optimisation loop started by ``study.ask()``.  Must be
        called exactly once per trial.

        Args:
            study: The study that owns this trial.
            trial: The trial being completed.
            score: Scalar performance metric (higher is better when the
                study direction is "maximize").
        """
        study.tell(trial, score)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def trial_number(self) -> int:
        """The zero-based index of this trial within the study."""
        return self._trial.number

    @property
    def raw_trial(self) -> optuna.Trial:
        """The underlying Optuna :class:`~optuna.Trial` object."""
        return self._trial
