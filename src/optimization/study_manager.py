"""
study_manager.py — Optuna study lifecycle management.

Handles creation, loading, and sampler selection for Optuna studies.
Sampler shifts from pure exploration to targeted exploitation as trial
count grows, following the schedule defined in OPTUNA_INTEGRATION.md.
"""

import optuna
from optuna.samplers import RandomSampler, TPESampler
from optuna.storages import RDBStorage


def create_study(
    name: str,
    direction: str = "maximize",
    storage_path: str = "optuna_trials.db",
) -> optuna.Study:
    """
    Create (or resume) a persistent Optuna study.

    The study is stored in a SQLite database so it survives process
    restarts. If a study with the given name already exists in the
    database it is loaded rather than recreated.

    Args:
        name: Unique name for this study.
        direction: Optimisation direction — "maximize" or "minimize".
        storage_path: Path to the SQLite file (relative or absolute).
            The ``sqlite:///`` prefix is added automatically.

    Returns:
        An :class:`optuna.Study` instance ready to use.
    """
    storage_url = f"sqlite:///{storage_path}"

    # Pick sampler based on how many trials already exist (0 on fresh study).
    try:
        existing = optuna.load_study(study_name=name, storage=storage_url)
        total_trials = len(existing.trials)
    except Exception:
        total_trials = 0

    sampler = get_sampler(total_trials)

    study = optuna.create_study(
        study_name=name,
        direction=direction,
        storage=storage_url,
        load_if_exists=True,
        sampler=sampler,
    )
    return study


def load_study(
    name: str,
    storage_path: str = "optuna_trials.db",
) -> optuna.Study:
    """
    Load an existing study from persistent storage.

    Args:
        name: Name of the study to load.
        storage_path: Path to the SQLite file used when the study was created.

    Returns:
        The loaded :class:`optuna.Study`.

    Raises:
        KeyError: If no study with the given name exists in the database.
    """
    storage_url = f"sqlite:///{storage_path}"
    study = optuna.load_study(study_name=name, storage=storage_url)
    return study


def get_sampler(total_trials: int) -> optuna.samplers.BaseSampler:
    """
    Select the appropriate sampler based on the current trial count.

    The schedule:
    - **< 50 trials** — :class:`~optuna.samplers.RandomSampler`:
      pure random exploration so the model sees a broad slice of the
      search space before starting to exploit.
    - **50–199 trials** — standard :class:`~optuna.samplers.TPESampler`:
      Bayesian optimisation with multivariate modelling but not yet
      aggressively exploiting near optima.
    - **≥ 200 trials** — aggressive TPE with ``consider_endpoints=True``:
      focuses tightly around the current best region for fine-grained
      exploitation.

    Args:
        total_trials: Number of completed trials recorded so far.

    Returns:
        An Optuna sampler instance.
    """
    if total_trials < 50:
        return RandomSampler(seed=42)
    elif total_trials < 200:
        return TPESampler(
            seed=42,
            n_startup_trials=0,  # History already covers warm-up
            multivariate=True,   # Model parameter interactions
        )
    else:
        return TPESampler(
            seed=42,
            n_startup_trials=0,
            multivariate=True,
            consider_endpoints=True,  # More precise near optima
        )
