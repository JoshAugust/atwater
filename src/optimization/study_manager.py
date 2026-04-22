"""
study_manager.py — Optuna study lifecycle management.

Handles creation, loading, and sampler selection for Optuna studies.

Phase 3 upgrades:
  - AutoSampler (OptunaHub) replaces hardcoded sampler schedule
  - JournalStorage replaces SQLite RDB (concurrent-safe, append-only)
  - PatientPruner wraps MedianPruner for LLM-expensive evaluations
  - Multi-objective studies via create_multi_objective_study()
  - Warm-start seeding via warm_start_study()

Backward compatibility is preserved — single-objective workflows continue
to work unchanged.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import optuna
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_sampler() -> optuna.samplers.BaseSampler:
    """
    Build an AutoSampler when optunahub is available, fall back to TPE.

    AutoSampler dynamically selects GPSampler / TPESampler based on the
    current trial count and search-space shape, achieving roughly 2× faster
    convergence on mixed integer/categorical spaces.
    """
    try:
        import optunahub  # noqa: PLC0415
        sampler = optunahub.load_module("samplers/auto_sampler").AutoSampler()
        logger.debug("Using AutoSampler (optunahub).")
        return sampler
    except Exception as exc:  # ImportError or network/hub failure
        logger.debug("AutoSampler unavailable (%s); falling back to TPESampler.", exc)
        return TPESampler(seed=42)


def _make_storage(storage_path: str) -> JournalStorage:
    """
    Build a JournalStorage backed by a plain file.

    JournalStorage is append-only and therefore concurrent-safe across
    threads / processes without locking overhead, making it superior to
    SQLite RDB for any workload with >1 parallel worker.

    Args:
        storage_path: Path to the journal log file (created if absent).

    Returns:
        A :class:`~optuna.storages.JournalStorage` instance.
    """
    return JournalStorage(JournalFileBackend(storage_path))


def _make_pruner() -> optuna.pruners.BasePruner:
    """
    Build a PatientPruner wrapping MedianPruner.

    PatientPruner adds a grace period (``patience`` extra steps) before
    pruning, which is important for expensive LLM evaluations where a trial
    may look below-median early but recover later.
    """
    return optuna.pruners.PatientPruner(
        optuna.pruners.MedianPruner(n_startup_trials=5),
        patience=2,
    )


# ---------------------------------------------------------------------------
# Public API — single-objective
# ---------------------------------------------------------------------------

def create_study(
    name: str,
    direction: str = "maximize",
    storage_path: str = "optuna_journal.log",
) -> optuna.Study:
    """
    Create (or resume) a persistent single-objective Optuna study.

    Uses JournalStorage (concurrent-safe) and AutoSampler (adaptive).
    If the study already exists it is resumed transparently.

    Args:
        name: Unique name for this study.
        direction: ``"maximize"`` or ``"minimize"``.
        storage_path: Path to the journal log file.

    Returns:
        An :class:`~optuna.Study` ready to use.
    """
    storage = _make_storage(storage_path)
    sampler = _make_sampler()
    pruner = _make_pruner()

    study = optuna.create_study(
        study_name=name,
        direction=direction,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    logger.info(
        "Study '%s' loaded/created (%d existing trials).",
        name,
        len(study.trials),
    )
    return study


def load_study(
    name: str,
    storage_path: str = "optuna_journal.log",
) -> optuna.Study:
    """
    Load an existing study from a JournalStorage log file.

    Args:
        name: Name of the study to load.
        storage_path: Path to the journal log file used when the study
            was created.

    Returns:
        The loaded :class:`~optuna.Study`.

    Raises:
        KeyError: If no study with the given name exists.
    """
    storage = _make_storage(storage_path)
    study = optuna.load_study(study_name=name, storage=storage)
    return study


# ---------------------------------------------------------------------------
# Public API — multi-objective
# ---------------------------------------------------------------------------

def create_multi_objective_study(
    name: str,
    directions: list[str] | None = None,
    storage_path: str = "optuna_journal.log",
) -> optuna.Study:
    """
    Create (or resume) a multi-objective study using NSGAIISampler.

    Multi-objective studies expose a Pareto front via ``study.best_trials``
    and are ideal for jointly optimising quality + diversity in creative
    generation tasks.

    Args:
        name: Unique name for this study.
        directions: List of ``"maximize"`` / ``"minimize"`` strings, one per
            objective.  Defaults to ``["maximize", "maximize"]`` (quality +
            diversity).
        storage_path: Path to the journal log file.

    Returns:
        An :class:`~optuna.Study` configured for multi-objective optimisation.

    Note:
        Pruners are incompatible with multi-objective studies; none is set.
    """
    if directions is None:
        directions = ["maximize", "maximize"]

    storage = _make_storage(storage_path)
    sampler = optuna.samplers.NSGAIISampler()

    study = optuna.create_study(
        study_name=name,
        directions=directions,
        storage=storage,
        sampler=sampler,
        load_if_exists=True,
    )
    logger.info(
        "Multi-objective study '%s' loaded/created (%d existing trials, directions=%s).",
        name,
        len(study.trials),
        directions,
    )
    return study


# ---------------------------------------------------------------------------
# Warm-start
# ---------------------------------------------------------------------------

def warm_start_study(
    study: optuna.Study,
    known_good_params: list[dict],
) -> None:
    """
    Seed a study with known-good parameter dicts so they run first.

    Uses ``study.enqueue_trial()`` under the hood.  Enqueued trials are
    dequeued in order before Bayesian optimisation takes over, giving the
    sampler a high-quality starting point (transfer learning across runs).

    Args:
        study: The study to seed.
        known_good_params: List of ``{param_name: value}`` dicts.  Each dict
            becomes one enqueued trial.

    Example::

        warm_start_study(study, [
            {"background": "dark", "layout": "hero", "bg_opacity": 0.8},
            {"background": "gradient", "layout": "split", "bg_opacity": 0.6},
        ])
    """
    for params in known_good_params:
        study.enqueue_trial(params)
    logger.info("Enqueued %d warm-start trials for study '%s'.", len(known_good_params), study.study_name)


# ---------------------------------------------------------------------------
# Legacy sampler helper (kept for compatibility / direct use)
# ---------------------------------------------------------------------------

def get_sampler(total_trials: int) -> optuna.samplers.BaseSampler:
    """
    Select a sampler based on total trial count (legacy schedule).

    .. deprecated::
        Prefer :func:`_make_sampler` (AutoSampler) for new code.  This
        function is retained for callers that depend on the deterministic
        schedule introduced in Phase 1.

    Schedule:
        - < 50 trials  → RandomSampler (broad exploration)
        - 50–199       → TPESampler multivariate
        - ≥ 200        → TPESampler aggressive (consider_endpoints=True)
    """
    from optuna.samplers import RandomSampler  # noqa: PLC0415

    if total_trials < 50:
        return RandomSampler(seed=42)
    elif total_trials < 200:
        return TPESampler(seed=42, n_startup_trials=0, multivariate=True)
    else:
        return TPESampler(
            seed=42,
            n_startup_trials=0,
            multivariate=True,
            consider_endpoints=True,
        )


# ---------------------------------------------------------------------------
# Parallel trial runner
# ---------------------------------------------------------------------------

class ParallelTrialRunner:
    """
    Run Optuna trials concurrently using a thread pool.

    Handles the ``study.ask()`` / ``study.tell()`` protocol safely across
    threads.  The trial function receives suggested parameters and must
    return a scalar score (single-objective) or a list of scores
    (multi-objective).

    Args:
        n_parallel: Number of concurrent worker threads.

    Example::

        runner = ParallelTrialRunner(n_parallel=4)
        scores = runner.run_parallel_trials(study, trial_fn=my_objective)
    """

    def __init__(self, n_parallel: int = 4) -> None:
        self.n_parallel = n_parallel

    def run_parallel_trials(
        self,
        study: optuna.Study,
        trial_fn: Callable[[dict], float | list[float]],
        n_trials: int | None = None,
    ) -> list[float | list[float]]:
        """
        Run trials in parallel and collect their scores.

        Args:
            study: The Optuna study to optimise.
            trial_fn: A callable that accepts a ``{param_name: value}``
                dict and returns a score (float) or list of scores
                (multi-objective).
            n_trials: Number of trials to run.  Defaults to
                ``n_parallel * 2`` when not supplied.

        Returns:
            List of scores in the order trials completed (not start order).
        """
        if n_trials is None:
            n_trials = self.n_parallel * 2

        results: list[float | list[float]] = []

        def _run_one(_: int) -> float | list[float]:
            trial = study.ask()
            try:
                params = trial.params if trial.params else {}
                score = trial_fn(params)
                study.tell(trial, score)
                return score
            except Exception as exc:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                logger.warning("Trial %d failed: %s", trial.number, exc)
                raise

        with ThreadPoolExecutor(max_workers=self.n_parallel) as pool:
            futures = [pool.submit(_run_one, i) for i in range(n_trials)]
            for fut in futures:
                try:
                    results.append(fut.result())
                except Exception:
                    pass  # already logged above

        return results
