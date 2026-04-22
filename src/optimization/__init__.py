"""
src.optimization — Optuna-backed optimisation layer for Atwater.

Public surface
--------------
Study lifecycle::

    from src.optimization import create_study, load_study, get_sampler

Trial parameter suggestion::

    from src.optimization import TrialAdapter, SearchSpace, DEFAULT_SEARCH_SPACE

Statistical queries::

    from src.optimization import (
        get_importances,
        get_best_params,
        get_dimension_stats,
        get_combo_heatmap_data,
        get_score_trend,
        get_asset_usage,
    )
"""

from src.optimization.study_manager import (
    create_study,
    load_study,
    get_sampler,
)

from src.optimization.trial_adapter import (
    SearchSpace,
    TrialAdapter,
    DEFAULT_SEARCH_SPACE,
)

from src.optimization.analytics import (
    get_importances,
    get_best_params,
    get_dimension_stats,
    get_combo_heatmap_data,
    get_score_trend,
    get_asset_usage,
)

__all__ = [
    # Study lifecycle
    "create_study",
    "load_study",
    "get_sampler",
    # Trial adapter
    "SearchSpace",
    "TrialAdapter",
    "DEFAULT_SEARCH_SPACE",
    # Analytics
    "get_importances",
    "get_best_params",
    "get_dimension_stats",
    "get_combo_heatmap_data",
    "get_score_trend",
    "get_asset_usage",
]
