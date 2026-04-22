"""
analytics.py — Statistical query interface over an Optuna study.

All functions accept an :class:`optuna.Study` and return plain Python
structures or pandas objects so callers can render / log / forward them
without depending on Optuna internals.

Functions
---------
get_importances       — which parameters drive scores most
get_best_params       — parameter dict for the highest-scoring trial
get_dimension_stats   — mean / std / count breakdown for one categorical dim
get_combo_heatmap_data — pivot table of mean score for two dim cross
get_score_trend       — rolling average of scores over time
get_asset_usage       — usage share per value for each categorical dim

Phase 3 additions (multi-objective)
------------------------------------
get_pareto_front      — list of Pareto-optimal trials for multi-obj studies
select_from_pareto    — pick best Pareto trial via TOPSIS-like weighting

All existing functions are compatible with JournalStorage-backed studies
because they rely only on ``study.trials`` / ``study.trials_dataframe()``,
which are storage-agnostic.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
import optuna
import optuna.importance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter importance
# ---------------------------------------------------------------------------

def get_importances(study: optuna.Study) -> dict[str, float]:
    """
    Return the relative importance of each parameter in the study.

    Uses Optuna's built-in fANOVA-based importance estimator.  Higher
    values mean the parameter has more influence on the objective score.

    Works with single-objective studies only.  Returns an empty dict for
    multi-objective studies (fANOVA is undefined over vector objectives).

    Args:
        study: A completed-or-in-progress Optuna study with at least a few
            finished trials.

    Returns:
        ``{param_name: importance_score}`` sorted descending by importance.
        Returns an empty dict when there are too few trials or the study
        is multi-objective.
    """
    if _is_multi_objective(study):
        logger.debug("get_importances: skipping multi-objective study '%s'.", study.study_name)
        return {}
    try:
        raw: dict[str, float] = optuna.importance.get_param_importances(study)
        return dict(sorted(raw.items(), key=lambda kv: kv[1], reverse=True))
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Best parameters
# ---------------------------------------------------------------------------

def get_best_params(study: optuna.Study) -> dict:
    """
    Return the parameter dict for the best trial seen so far.

    For single-objective studies returns ``study.best_params``.
    For multi-objective studies returns the params of the first Pareto
    trial (arbitrary but deterministic).

    Args:
        study: Optuna study.

    Returns:
        ``{param_name: value}`` or an empty dict if no trials completed.
    """
    try:
        if _is_multi_objective(study):
            pareto = get_pareto_front(study)
            return dict(pareto[0].params) if pareto else {}
        return dict(study.best_params)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Per-dimension statistics
# ---------------------------------------------------------------------------

def get_dimension_stats(
    study: optuna.Study,
    dimension: str,
) -> pd.DataFrame:
    """
    Return mean / std / count of scores grouped by each value of *dimension*.

    For multi-objective studies the first objective value is used.

    Args:
        study: Optuna study.
        dimension: The parameter name to group by, e.g. ``"background"``.

    Returns:
        A :class:`pandas.DataFrame` with columns ``mean``, ``std``, ``count``
        and an index of the dimension's distinct values.  Returns an empty
        DataFrame when the dimension is not found or there are no trials.
    """
    df = _get_trials_df(study)
    if df.empty:
        return pd.DataFrame(columns=["mean", "std", "count"])

    score_col = _score_column(df)
    if score_col is None:
        return pd.DataFrame(columns=["mean", "std", "count"])

    col = _resolve_column(df, dimension)
    if col is None:
        return pd.DataFrame(columns=["mean", "std", "count"])

    stats = df.groupby(col)[score_col].agg(["mean", "std", "count"])
    stats.index.name = dimension
    return stats


# ---------------------------------------------------------------------------
# Combination heatmap
# ---------------------------------------------------------------------------

def get_combo_heatmap_data(
    study: optuna.Study,
    dim1: str,
    dim2: str,
) -> pd.DataFrame:
    """
    Return a pivot table of mean scores for all (dim1 × dim2) combinations.

    For multi-objective studies the first objective value is used.

    Args:
        study: Optuna study.
        dim1: Row dimension name (e.g. ``"background"``).
        dim2: Column dimension name (e.g. ``"layout"``).

    Returns:
        A pivot :class:`~pandas.DataFrame`.  Returns an empty DataFrame
        when either dimension is missing or no trials exist.
    """
    df = _get_trials_df(study)
    if df.empty:
        return pd.DataFrame()

    score_col = _score_column(df)
    if score_col is None:
        return pd.DataFrame()

    col1 = _resolve_column(df, dim1)
    col2 = _resolve_column(df, dim2)
    if col1 is None or col2 is None:
        return pd.DataFrame()

    pivot = df.pivot_table(
        values=score_col,
        index=col1,
        columns=col2,
        aggfunc="mean",
    )
    pivot.index.name = dim1
    pivot.columns.name = dim2
    return pivot


# ---------------------------------------------------------------------------
# Score trend
# ---------------------------------------------------------------------------

def get_score_trend(
    study: optuna.Study,
    window: int = 20,
) -> pd.Series:
    """
    Return a rolling-average time series of trial scores.

    For multi-objective studies the first objective value is used.

    Args:
        study: Optuna study.
        window: Number of trials in each rolling window.

    Returns:
        A :class:`pandas.Series` indexed by trial number.  Returns an empty
        Series when there are no trials.
    """
    df = _get_trials_df(study)
    if df.empty:
        return pd.Series(dtype=float)

    score_col = _score_column(df)
    if score_col is None or score_col not in df.columns:
        return pd.Series(dtype=float)

    trend = df[score_col].rolling(window=window, min_periods=1).mean()
    trend.index = df.index
    return trend


# ---------------------------------------------------------------------------
# Asset usage
# ---------------------------------------------------------------------------

def get_asset_usage(
    study: optuna.Study,
    last_n: int = 50,
) -> dict[str, dict[str, float]]:
    """
    Return usage percentages for every categorical dimension.

    Looks at only the most recent *last_n* trials so the numbers reflect
    the current exploitation focus rather than the full historical mix.

    Args:
        study: Optuna study.
        last_n: How many of the most recent trials to examine.

    Returns:
        Nested dict ``{dimension_name: {value: fraction, ...}, ...}``
        where fractions sum to 1.0 for each dimension.  Returns an empty
        dict when there are no trials.
    """
    df = _get_trials_df(study)
    if df.empty:
        return {}

    recent = df.tail(last_n)

    cat_cols = [
        c for c in recent.columns
        if c.startswith("params_") and pd.api.types.is_string_dtype(recent[c])
    ]

    usage: dict[str, dict[str, float]] = {}
    for col in cat_cols:
        dim_name = col[len("params_"):]
        counts = recent[col].value_counts(normalize=True)
        usage[dim_name] = counts.to_dict()

    return usage


# ---------------------------------------------------------------------------
# Multi-objective: Pareto front
# ---------------------------------------------------------------------------

def get_pareto_front(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    """
    Return the list of Pareto-optimal trials for a multi-objective study.

    For single-objective studies returns the single best trial in a list.

    Args:
        study: Optuna study.

    Returns:
        List of :class:`~optuna.trial.FrozenTrial` on the Pareto front,
        or an empty list if no trials have completed.
    """
    try:
        if _is_multi_objective(study):
            return list(study.best_trials)
        # Single-objective fallback
        best = study.best_trial
        return [best] if best else []
    except Exception:
        return []


def select_from_pareto(
    study: optuna.Study,
    weights: dict[str, float] | None = None,
) -> optuna.trial.FrozenTrial | None:
    """
    Choose the best Pareto trial via a TOPSIS-like weighted distance method.

    Each objective is normalised to [0, 1] across the Pareto front, then
    a weighted Euclidean distance to the ideal point (all objectives at
    their personal best) is computed.  The trial closest to the ideal point
    wins.

    If the study is single-objective, the best trial is returned directly.

    Args:
        study: Optuna study (single- or multi-objective).
        weights: Dict mapping objective index (as string, e.g. ``"0"``) or
            a short label to a non-negative weight.  Equal weights are used
            when this is *None* or empty.

            For a 2-objective study:
            ``{"0": 0.7, "1": 0.3}`` weights the first objective at 70 %.

    Returns:
        The selected :class:`~optuna.trial.FrozenTrial`, or *None* if no
        completed trials exist.

    Example::

        best = select_from_pareto(study, weights={"0": 0.6, "1": 0.4})
        print(best.params)
    """
    pareto = get_pareto_front(study)
    if not pareto:
        return None
    if len(pareto) == 1:
        return pareto[0]

    # Collect objective value matrix — shape (n_trials, n_objectives)
    values_matrix = np.array([t.values for t in pareto if t.values is not None], dtype=float)
    if values_matrix.ndim == 1:
        values_matrix = values_matrix.reshape(-1, 1)

    n_obj = values_matrix.shape[1]

    # Build weight vector
    if weights:
        w = np.array([float(weights.get(str(i), 1.0)) for i in range(n_obj)])
    else:
        w = np.ones(n_obj)

    # Normalise to [0, 1]: higher normalised value = better (we're maximising)
    col_min = values_matrix.min(axis=0)
    col_max = values_matrix.max(axis=0)
    col_range = np.where(col_max - col_min == 0, 1.0, col_max - col_min)
    norm = (values_matrix - col_min) / col_range  # 0 = worst, 1 = best

    # Ideal point (1.0 for each objective in normalised space)
    ideal = np.ones(n_obj)

    # Weighted Euclidean distance to ideal
    dist = np.sqrt(np.sum(w * (norm - ideal) ** 2, axis=1))

    best_idx = int(np.argmin(dist))
    return pareto[best_idx]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_multi_objective(study: optuna.Study) -> bool:
    """Return True when the study has multiple optimisation directions."""
    return len(study.directions) > 1


def _get_trials_df(study: optuna.Study) -> pd.DataFrame:
    """
    Return ``study.trials_dataframe()`` or an empty DataFrame on error.

    Compatible with JournalStorage-backed studies.
    """
    try:
        return study.trials_dataframe()
    except Exception as exc:
        logger.debug("trials_dataframe() failed (%s).", exc)
        return pd.DataFrame()


def _score_column(df: pd.DataFrame) -> str | None:
    """
    Find the primary score column in a trials DataFrame.

    Optuna uses ``"value"`` for single-objective and ``"values_0"``,
    ``"values_1"``, … for multi-objective studies.
    """
    if "value" in df.columns:
        return "value"
    for col in df.columns:
        if col.startswith("values_"):
            return col
    return None


def _resolve_column(df: pd.DataFrame, dimension: str) -> str | None:
    """
    Map a bare dimension name to the actual column name in *df*.

    Optuna prefixes parameter columns with ``params_``.  This helper
    accepts either form and returns the exact column name, or *None* if
    neither variant exists.
    """
    if dimension in df.columns:
        return dimension
    prefixed = f"params_{dimension}"
    if prefixed in df.columns:
        return prefixed
    return None
