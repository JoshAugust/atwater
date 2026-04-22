"""
trial_adapter.py — Bridge between Optuna trials and the shared agent state.

Provides:
- :class:`SearchSpace` — declarative definition of what parameters exist.
- :class:`TrialAdapter` — wraps an Optuna trial, translates SearchSpace
  definitions into ``trial.suggest_*`` calls, and reports scores back.

Phase 3 additions:
- :func:`save_trial_artifact` — persist generated content alongside a trial.
- :func:`load_trial_artifact` — retrieve stored content for a past trial.

Artifact storage uses ``optuna.artifacts.FileSystemArtifactStore`` when
available, with a plain filesystem fallback for maximum compatibility.

Usage example::

    from src.optimization.trial_adapter import TrialAdapter, DEFAULT_SEARCH_SPACE

    trial = study.ask()
    adapter = TrialAdapter(trial)
    params = adapter.suggest_params(DEFAULT_SEARCH_SPACE)

    score = run_pipeline(params)
    adapter.report_score(study, trial, score)

    # Optionally store the generated output alongside the trial
    save_trial_artifact(study, trial, content="<generated text>", filename="output.txt")
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import optuna

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Artifact store (lazy singleton)
# ---------------------------------------------------------------------------

_DEFAULT_ARTIFACT_DIR = "artifacts"


def _get_artifact_store(artifact_dir: str = _DEFAULT_ARTIFACT_DIR):
    """
    Return a FileSystemArtifactStore if optuna.artifacts is available.

    Falls back to None so callers can use their own plain-filesystem path.
    """
    try:
        from optuna.artifacts import FileSystemArtifactStore  # noqa: PLC0415
        store = FileSystemArtifactStore(artifact_dir)
        Path(artifact_dir).mkdir(parents=True, exist_ok=True)
        return store
    except Exception as exc:
        logger.debug("optuna.artifacts unavailable (%s); using filesystem fallback.", exc)
        return None


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
# Artifact helpers
# ---------------------------------------------------------------------------

def save_trial_artifact(
    study: optuna.Study,
    trial: optuna.Trial,
    content: str,
    filename: str = "output.txt",
    artifact_dir: str = _DEFAULT_ARTIFACT_DIR,
) -> str:
    """
    Save *content* as an artifact associated with *trial*.

    Tries the Optuna artifact API first; if unavailable falls back to
    writing ``<artifact_dir>/<study_name>/trial_<number>/<filename>``.

    Args:
        study: The study the trial belongs to.
        trial: The trial whose output you want to store.
        content: Text content to persist.
        filename: Filename for the artifact (default ``"output.txt"``).
        artifact_dir: Root directory for artifact storage.

    Returns:
        The artifact ID (Optuna) or filesystem path (fallback).
    """
    store = _get_artifact_store(artifact_dir)

    if store is not None:
        try:
            from optuna.artifacts import upload_artifact  # noqa: PLC0415
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=os.path.splitext(filename)[1] or ".txt",
                delete=False,
                encoding="utf-8",
            ) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            artifact_id = upload_artifact(
                    study_or_trial=trial, file_path=tmp_path, artifact_store=store
                )
            os.unlink(tmp_path)
            logger.debug(
                "Saved artifact '%s' for trial %d (id=%s).",
                filename,
                trial.number,
                artifact_id,
            )
            return artifact_id
        except Exception as exc:
            logger.warning("optuna artifact upload failed (%s); using filesystem fallback.", exc)

    # Filesystem fallback
    dest_dir = Path(artifact_dir) / study.study_name / f"trial_{trial.number}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename
    dest_path.write_text(content, encoding="utf-8")
    logger.debug("Saved artifact to '%s'.", dest_path)
    return str(dest_path)


def load_trial_artifact(
    study: optuna.Study,
    trial_number: int,
    filename: str = "output.txt",
    artifact_dir: str = _DEFAULT_ARTIFACT_DIR,
) -> str | None:
    """
    Load the artifact stored for *trial_number*.

    Uses the filesystem fallback path (``<artifact_dir>/<study>/<trial>/``).
    Optuna's artifact API does not have a first-class "download by trial"
    query, so we fall back to the structured directory layout written by
    :func:`save_trial_artifact`.

    Args:
        study: The study the trial belongs to.
        trial_number: Zero-based trial index.
        filename: Filename to retrieve (default ``"output.txt"``).
        artifact_dir: Root artifact directory.

    Returns:
        The stored text content, or *None* if not found.
    """
    # First try structured filesystem path (written by save_trial_artifact fallback)
    dest_path = Path(artifact_dir) / study.study_name / f"trial_{trial_number}" / filename
    if dest_path.exists():
        return dest_path.read_text(encoding="utf-8")

    # Check trial user_attrs for artifact_id stored by upload_artifact path
    try:
        trial = study.trials[trial_number]
        artifact_id = trial.user_attrs.get(f"artifact_id:{filename}")
        if artifact_id:
            store = _get_artifact_store(artifact_dir)
            if store is not None:
                from optuna.artifacts import download_artifact  # noqa: PLC0415
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
                    tmp_path = tmp.name
                download_artifact(artifact_id, tmp_path, store)
                content = Path(tmp_path).read_text(encoding="utf-8")
                os.unlink(tmp_path)
                return content
    except Exception as exc:
        logger.debug("Artifact load attempt failed (%s).", exc)

    logger.debug("No artifact found for trial %d / '%s'.", trial_number, filename)
    return None


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
        score: float | list[float],
    ) -> None:
        """
        Report a trial's final score back to the study.

        Closes the optimisation loop started by ``study.ask()``.  Must be
        called exactly once per trial.  Accepts a scalar (single-objective)
        or list of floats (multi-objective).

        Args:
            study: The study that owns this trial.
            trial: The trial being completed.
            score: Scalar or list of performance metrics.
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
