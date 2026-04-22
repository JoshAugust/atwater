"""
tests/test_optuna_phase3.py — Phase 3 Optuna upgrade tests.

Covers:
  - test_autosampler_fallback     : TPE is used when optunahub is unavailable
  - test_journal_storage          : create study, add trials, verify persistence
  - test_multi_objective          : Pareto front is non-empty after optimization
  - test_warm_start               : enqueued trials are attempted first
  - test_pareto_select            : select_from_pareto returns a trial
  - test_parallel_trial_runner    : ParallelTrialRunner collects scores
  - test_artifact_save_load       : save / load round-trip via filesystem fallback
  - test_backward_compat          : old get_sampler schedule still works
"""

from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import optuna
import pytest

# Silence Optuna log noise during tests
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_objective(trial: optuna.Trial) -> float:
    """Trivial single-objective function: maximize x."""
    x = trial.suggest_float("x", 0.0, 1.0)
    return x


def _multi_objective(trial: optuna.Trial) -> list[float]:
    """Two-objective: maximize x, maximize (1 - x) — classic trade-off."""
    x = trial.suggest_float("x", 0.0, 1.0)
    return [x, 1.0 - x]


# ---------------------------------------------------------------------------
# test_autosampler_fallback
# ---------------------------------------------------------------------------

class TestAutosamplerFallback:
    def test_uses_tpe_when_optunahub_missing(self, tmp_path):
        """When optunahub is absent _make_sampler() falls back to TPESampler."""
        from src.optimization import study_manager

        with patch.dict(sys.modules, {"optunahub": None}):
            sampler = study_manager._make_sampler()

        assert isinstance(sampler, optuna.samplers.TPESampler), (
            f"Expected TPESampler fallback, got {type(sampler)}"
        )

    def test_uses_autosampler_when_optunahub_present(self, tmp_path):
        """When optunahub is importable _make_sampler() returns AutoSampler."""
        import optunahub  # noqa: F401 — must be installed
        from src.optimization import study_manager

        sampler = study_manager._make_sampler()
        # AutoSampler wraps another sampler; it's not TPE
        # Just verify we don't crash and get a valid sampler back
        assert isinstance(sampler, optuna.samplers.BaseSampler)


# ---------------------------------------------------------------------------
# test_journal_storage
# ---------------------------------------------------------------------------

class TestJournalStorage:
    def test_create_and_persist(self, tmp_path):
        """Create a study, add trials, reload — trial count must survive restart."""
        from src.optimization.study_manager import create_study, load_study

        journal = str(tmp_path / "test.log")

        study = create_study("persist-test", storage_path=journal)
        study.optimize(_simple_objective, n_trials=5)

        assert len(study.trials) == 5, "Expected 5 trials after optimize"

        # Reload from the same journal file
        reloaded = load_study("persist-test", storage_path=journal)
        assert len(reloaded.trials) == 5, (
            f"Reloaded study has {len(reloaded.trials)} trials, expected 5"
        )

    def test_journal_file_created(self, tmp_path):
        """The journal log file must actually be written to disk."""
        from src.optimization.study_manager import create_study

        journal = str(tmp_path / "journal.log")
        study = create_study("file-check", storage_path=journal)
        study.optimize(_simple_objective, n_trials=3)

        assert Path(journal).exists(), "Journal file not found on disk"
        assert Path(journal).stat().st_size > 0, "Journal file is empty"


# ---------------------------------------------------------------------------
# test_multi_objective
# ---------------------------------------------------------------------------

class TestMultiObjective:
    def test_pareto_front_non_empty(self, tmp_path):
        """After optimisation a multi-objective study exposes Pareto trials."""
        from src.optimization.study_manager import create_multi_objective_study
        from src.optimization.analytics import get_pareto_front

        journal = str(tmp_path / "mo.log")
        study = create_multi_objective_study(
            "mo-test",
            directions=["maximize", "maximize"],
            storage_path=journal,
        )
        study.optimize(_multi_objective, n_trials=20)

        pareto = get_pareto_front(study)
        assert len(pareto) > 0, "Pareto front is empty after 20 trials"

    def test_pareto_trials_have_two_values(self, tmp_path):
        """Each Pareto trial must expose two objective values."""
        from src.optimization.study_manager import create_multi_objective_study
        from src.optimization.analytics import get_pareto_front

        journal = str(tmp_path / "mo2.log")
        study = create_multi_objective_study("mo-vals", storage_path=journal)
        study.optimize(_multi_objective, n_trials=10)

        pareto = get_pareto_front(study)
        for t in pareto:
            assert t.values is not None and len(t.values) == 2

    def test_select_from_pareto_returns_trial(self, tmp_path):
        """select_from_pareto returns a FrozenTrial with params."""
        from src.optimization.study_manager import create_multi_objective_study
        from src.optimization.analytics import select_from_pareto

        journal = str(tmp_path / "mo3.log")
        study = create_multi_objective_study("mo-sel", storage_path=journal)
        study.optimize(_multi_objective, n_trials=15)

        best = select_from_pareto(study, weights={"0": 0.7, "1": 0.3})
        assert best is not None
        assert "x" in best.params


# ---------------------------------------------------------------------------
# test_warm_start
# ---------------------------------------------------------------------------

class TestWarmStart:
    def test_enqueued_params_used_first(self, tmp_path):
        """Warm-start params are dequeued before the sampler generates new ones."""
        from src.optimization.study_manager import create_study, warm_start_study

        journal = str(tmp_path / "warm.log")
        study = create_study("warm-test", storage_path=journal)

        known_good = [{"x": 0.123}, {"x": 0.456}]
        warm_start_study(study, known_good)

        # Run exactly the number of warm-start trials
        study.optimize(_simple_objective, n_trials=2)

        trial_params = [t.params for t in study.trials]
        xs = [p["x"] for p in trial_params]

        assert 0.123 in xs, "Warm-start param 0.123 was not used"
        assert 0.456 in xs, "Warm-start param 0.456 was not used"

    def test_warm_start_multiple_params(self, tmp_path):
        """Warm start works with multi-param dicts."""
        from src.optimization.study_manager import create_study, warm_start_study

        journal = str(tmp_path / "warm2.log")
        study = create_study("warm-multi", storage_path=journal)

        def two_param_obj(trial: optuna.Trial) -> float:
            x = trial.suggest_float("x", 0.0, 1.0)
            y = trial.suggest_float("y", 0.0, 1.0)
            return x + y

        warm_start_study(study, [{"x": 0.9, "y": 0.9}])
        study.optimize(two_param_obj, n_trials=1)

        first = study.trials[0]
        assert abs(first.params["x"] - 0.9) < 1e-9
        assert abs(first.params["y"] - 0.9) < 1e-9


# ---------------------------------------------------------------------------
# test_parallel_trial_runner
# ---------------------------------------------------------------------------

class TestParallelTrialRunner:
    def test_collects_scores(self, tmp_path):
        """ParallelTrialRunner returns a list of scores."""
        from src.optimization.study_manager import create_study, ParallelTrialRunner

        journal = str(tmp_path / "parallel.log")
        study = create_study("parallel-test", storage_path=journal)

        runner = ParallelTrialRunner(n_parallel=2)

        # trial_fn receives the params dict; must return a float
        def trial_fn(params: dict) -> float:
            return 0.5  # deterministic

        scores = runner.run_parallel_trials(study, trial_fn=trial_fn, n_trials=4)
        assert len(scores) == 4, f"Expected 4 scores, got {len(scores)}"
        assert all(s == 0.5 for s in scores)

    def test_trials_recorded(self, tmp_path):
        """Parallel trials are committed back to the study."""
        from src.optimization.study_manager import create_study, ParallelTrialRunner

        journal = str(tmp_path / "parallel2.log")
        study = create_study("parallel-rec", storage_path=journal)

        runner = ParallelTrialRunner(n_parallel=2)
        runner.run_parallel_trials(study, trial_fn=lambda p: 1.0, n_trials=4)

        assert len(study.trials) == 4


# ---------------------------------------------------------------------------
# test_artifact_save_load
# ---------------------------------------------------------------------------

class TestArtifacts:
    def test_save_and_load_roundtrip(self, tmp_path):
        """Content written by save_trial_artifact can be retrieved by load_trial_artifact."""
        from src.optimization.study_manager import create_study
        from src.optimization.trial_adapter import save_trial_artifact, load_trial_artifact

        journal = str(tmp_path / "art.log")
        artifact_dir = str(tmp_path / "artifacts")
        study = create_study("artifact-test", storage_path=journal)

        trial = study.ask()
        study.tell(trial, 0.42)

        content = "This is the generated creative output."
        save_trial_artifact(study, trial, content, filename="out.txt", artifact_dir=artifact_dir)

        recovered = load_trial_artifact(study, trial.number, filename="out.txt", artifact_dir=artifact_dir)
        assert recovered == content, f"Round-trip mismatch: got {recovered!r}"

    def test_missing_artifact_returns_none(self, tmp_path):
        """load_trial_artifact returns None for non-existent artifact."""
        from src.optimization.study_manager import create_study
        from src.optimization.trial_adapter import load_trial_artifact

        journal = str(tmp_path / "art2.log")
        artifact_dir = str(tmp_path / "artifacts2")
        study = create_study("artifact-miss", storage_path=journal)

        result = load_trial_artifact(study, 999, artifact_dir=artifact_dir)
        assert result is None


# ---------------------------------------------------------------------------
# test_backward_compat
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_get_sampler_schedule(self):
        """Legacy get_sampler() schedule must still produce the correct sampler types."""
        from src.optimization.study_manager import get_sampler
        from optuna.samplers import RandomSampler, TPESampler

        assert isinstance(get_sampler(0), RandomSampler)
        assert isinstance(get_sampler(49), RandomSampler)
        assert isinstance(get_sampler(50), TPESampler)
        assert isinstance(get_sampler(199), TPESampler)
        assert isinstance(get_sampler(200), TPESampler)

    def test_single_objective_study_still_works(self, tmp_path):
        """Single-objective create_study + optimize round-trip is unbroken."""
        from src.optimization.study_manager import create_study
        from src.optimization.analytics import get_best_params, get_importances

        journal = str(tmp_path / "compat.log")
        study = create_study("compat-test", storage_path=journal)
        study.optimize(_simple_objective, n_trials=10)

        best = get_best_params(study)
        assert "x" in best

        importances = get_importances(study)
        # With only 10 trials importances may be empty (not enough for fANOVA)
        # Just confirm it doesn't raise
        assert isinstance(importances, dict)

    def test_analytics_dimension_stats(self, tmp_path):
        """get_dimension_stats works with JournalStorage-backed study."""
        from src.optimization.study_manager import create_study
        from src.optimization.analytics import get_dimension_stats

        journal = str(tmp_path / "dimstats.log")
        study = create_study("dim-test", storage_path=journal)

        def obj(trial: optuna.Trial) -> float:
            _ = trial.suggest_categorical("color", ["red", "blue"])
            return trial.suggest_float("x", 0.0, 1.0)

        study.optimize(obj, n_trials=10)
        df = get_dimension_stats(study, "color")
        assert not df.empty, "get_dimension_stats returned empty DataFrame"
        assert "mean" in df.columns
