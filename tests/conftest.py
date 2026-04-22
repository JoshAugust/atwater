"""
conftest.py — Shared pytest fixtures for the Atwater test suite.

Fixtures
--------
tmp_db(suffix)      Creates a temporary SQLite path in tmp_path.
sample_knowledge_entries  A list of KnowledgeEntry objects for testing.
sample_optuna_study       A small in-memory Optuna study with 10 completed trials.
mock_embed              Patches out sentence-transformers so tests run without it.
"""

from __future__ import annotations

import math
import random
import sqlite3
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import optuna
import pytest

optuna.logging.set_verbosity(optuna.logging.ERROR)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_random_embedding(dim: int = 384, seed: int | None = None) -> np.ndarray:
    """Return a unit-norm random float32 vector of length *dim*."""
    rng = np.random.default_rng(seed)
    vec = rng.random(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


# ---------------------------------------------------------------------------
# tmp_db fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db(tmp_path: Path):
    """
    Factory fixture that creates a temporary SQLite path in tmp_path.

    Usage::

        def test_something(tmp_db):
            db = tmp_db("state")   # -> tmp_path / "state.db"
    """

    def _factory(name: str) -> Path:
        return tmp_path / f"{name}.db"

    return _factory


# ---------------------------------------------------------------------------
# mock_embed fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=False)
def mock_embed():
    """
    Patch sentence-transformers so tests run without the library installed.

    The mock model returns a deterministic unit-norm random embedding for any
    text.  Tests that call knowledge_write / knowledge_read automatically use
    this when the fixture is active.

    Use::

        def test_kb(mock_embed, tmp_db):
            ...
    """
    import sys

    # Build a minimal mock for SentenceTransformer
    mock_st_module = MagicMock()

    class _MockModel:
        def encode(
            self,
            text: str | list[str],
            convert_to_numpy: bool = True,
            normalize_embeddings: bool = True,
        ) -> np.ndarray:
            seed = abs(hash(text if isinstance(text, str) else str(text))) % (2**31)
            return _make_random_embedding(384, seed=seed)

    mock_st_module.SentenceTransformer.return_value = _MockModel()

    # Patch the module and the internal _get_model / _embed functions
    with (
        patch.dict(
            sys.modules, {"sentence_transformers": mock_st_module}
        ),
        patch(
            "src.memory.knowledge_base._get_model",
            return_value=_MockModel(),
        ),
        patch(
            "src.memory.knowledge_base._embed",
            side_effect=lambda text: _make_random_embedding(
                384, seed=abs(hash(text)) % (2**31)
            ),
        ),
    ):
        # Also reset the module-level singleton so the mock takes effect
        import src.memory.knowledge_base as kb_mod
        original_instance = kb_mod._model_instance
        kb_mod._model_instance = _MockModel()
        yield
        kb_mod._model_instance = original_instance


# ---------------------------------------------------------------------------
# sample_knowledge_entries fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_knowledge_entries():
    """
    Return a list of KnowledgeEntry objects covering all active tiers.

    Each entry is self-contained (no DB required).  Useful for testing the
    consolidation engine, which works on in-memory lists.
    """
    from atwater.src.knowledge.models import KnowledgeEntry

    return [
        KnowledgeEntry(
            content="Sans-serif headlines outperform serif by 23% across 200+ tests",
            tier="rule",
            confidence=0.95,
            created_cycle=1,
            last_validated_cycle=300,
            validation_count=42,
            topic_cluster="typography",
            optuna_evidence={"trial_count": 210, "p_value": 0.02},
        ),
        KnowledgeEntry(
            content="Dark backgrounds work better for tech products",
            tier="pattern",
            confidence=0.75,
            created_cycle=10,
            last_validated_cycle=200,
            validation_count=12,
            topic_cluster="background",
            optuna_evidence={"trial_count": 120, "p_value": 0.08},
        ),
        KnowledgeEntry(
            content="Gradient overlays improve depth perception",
            tier="pattern",
            confidence=0.65,
            created_cycle=20,
            last_validated_cycle=180,
            validation_count=8,
            topic_cluster="background",
            optuna_evidence={"trial_count": 90, "p_value": 0.09},
        ),
        KnowledgeEntry(
            content="Layout C with gradient scored 0.91 on run #847",
            tier="observation",
            confidence=0.80,
            created_cycle=847,
            last_validated_cycle=847,
            validation_count=1,
            topic_cluster="layout",
            optuna_evidence={"trial_count": 1, "p_value": None},
        ),
        KnowledgeEntry(
            content="Hero font size 48px achieved highest CTR in A/B test",
            tier="observation",
            confidence=0.70,
            created_cycle=100,
            last_validated_cycle=100,
            validation_count=2,
            topic_cluster="typography",
            optuna_evidence={"trial_count": 2, "p_value": 0.15},
        ),
    ]


# ---------------------------------------------------------------------------
# sample_optuna_study fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_optuna_study(tmp_db):
    """
    Return a small in-memory Optuna study with 10 completed trials.

    Uses an in-memory SQLite so no disk I/O is required.  The study
    explores the same categorical and continuous dimensions as the
    DEFAULT_SEARCH_SPACE.
    """
    storage_url = "sqlite:///:memory:"

    study = optuna.create_study(
        study_name="test-study",
        direction="maximize",
        storage=storage_url,
        load_if_exists=True,
        sampler=optuna.samplers.RandomSampler(seed=42),
    )

    backgrounds = ["dark", "gradient", "minimal", "textured", "abstract"]
    layouts = ["hero", "split", "grid", "asymmetric", "stacked"]
    shots = ["front", "angle", "lifestyle", "closeup", "context"]

    rng = random.Random(42)

    for trial_idx in range(10):
        trial = study.ask()
        trial.suggest_categorical("background", backgrounds)
        trial.suggest_categorical("layout", layouts)
        trial.suggest_categorical("shot", shots)
        trial.suggest_float("bg_opacity", 0.2, 1.0)
        trial.suggest_int("hero_font_size", 24, 72, step=4)

        score = rng.uniform(0.4, 0.95)
        study.tell(trial, score)

    return study
