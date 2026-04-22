"""
analytics.py — Statistical query interface over an Optuna study.

All functions accept an :class:`optuna.Study` and return plain Python
structures or pandas objects so callers can render / log / forward them
without depending on Optuna internals.

Functions
---------
get_importances    — which parameters drive scores most
get_best_params    — parameter dict for the highest-scoring trial
get_dimension_stats — mean / std / count breakdown for one categorical dim
get_combo_heatmap_data — pivot table of mean score for two dim cross
get_score_trend    — rolling average of scores over time
get_asset_usage    — usage share per value for each categorical dim
"""

from __future__ import annotations

import pandas as pd
import optuna
import optuna.importance


# ---------------------------------------------------------------------------
# Parameter importance
# ---------------------------------------------------------------------------

def get_importances(study: optuna.Study) -> dict[str, float]:
    """
    Return the relative importance of each parameter in the study.

    Uses Optuna's built-in fANOVA-based importance estimator.  Higher
    values mean the parameter has more influence on the objective score.

    Args:
        study: A completed-or-in-progress Optuna study with at least a few
            finished trials.

    Returns:
        ``{param_name: importance_score}`` sorted descending by importance.
        Returns an empty dict when there are too few trials to compute.
    """
    try:
        raw: dict[str, float] = optuna.importance.get_param_importances(study)
        return dict(sorted(raw.items(), key=lambda kv: kv[1], reverse=True))
    except Exception:
        # Not enough completed trials yet
        return {}


# ---------------------------------------------------------------------------
# Best parameters
# ---------------------------------------------------------------------------

def get_best_params(study: optuna.Study) -> dict:
    """
    Return the parameter dict for the best trial seen so far.

    Args:
        study: Optuna study.

    Returns:
        ``{param_name: value}`` for the highest-scoring trial, or an
        empty dict if no trials have completed yet.
    """
    try:
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

    Args:
        study: Optuna study.
        dimension: The parameter name to group by, e.g. ``"background"``.
            The function handles both bare names and the ``params_``-prefixed
            column names that Optuna uses internally.

    Returns:
        A :class:`pandas.DataFrame` with columns ``mean``, ``std``, ``count``
        and an index of the dimension's distinct values.  Returns an empty
        DataFrame when the dimension is not found or there are no trials.
    """
    df = study.trials_dataframe()
    if df.empty:
        return pd.DataFrame(columns=["mean", "std", "count"])

    col = _resolve_column(df, dimension)
    if col is None:
        return pd.DataFrame(columns=["mean", "std", "count"])

    stats = df.groupby(col)["value"].agg(["mean", "std", "count"])
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

    Suitable for plotting with ``seaborn.heatmap`` or similar.

    Args:
        study: Optuna study.
        dim1: Row dimension name (e.g. ``"background"``).
        dim2: Column dimension name (e.g. ``"layout"``).

    Returns:
        A pivot :class:`~pandas.DataFrame` where rows = dim1 values,
        columns = dim2 values, cells = mean score.  Returns an empty
        DataFrame when either dimension is missing or no trials exist.
    """
    df = study.trials_dataframe()
    if df.empty:
        return pd.DataFrame()

    col1 = _resolve_column(df, dim1)
    col2 = _resolve_column(df, dim2)
    if col1 is None or col2 is None:
        return pd.DataFrame()

    pivot = df.pivot_table(
        values="value",
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

    Args:
        study: Optuna study.
        window: Number of trials in each rolling window.

    Returns:
        A :class:`pandas.Series` indexed by trial number with the rolling
        mean score.  Returns an empty Series when there are no trials.
    """
    df = study.trials_dataframe()
    if df.empty or "value" not in df.columns:
        return pd.Series(dtype=float)

    trend = df["value"].rolling(window=window, min_periods=1).mean()
    trend.index = df.index  # trial number as index
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
    df = study.trials_dataframe()
    if df.empty:
        return {}

    recent = df.tail(last_n)

    # Identify categorical param columns (prefix "params_").
    # Use pd.api.types.is_string_dtype to handle both legacy object dtype and
    # the newer pandas StringDtype that newer pandas/Optuna combinations produce.
    cat_cols = [
        c for c in recent.columns
        if c.startswith("params_") and pd.api.types.is_string_dtype(recent[c])
    ]

    usage: dict[str, dict[str, float]] = {}
    for col in cat_cols:
        dim_name = col[len("params_"):]  # strip prefix
        counts = recent[col].value_counts(normalize=True)
        usage[dim_name] = counts.to_dict()

    return usage


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
