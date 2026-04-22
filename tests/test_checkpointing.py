"""
tests/test_checkpointing.py — Tests for src.resilience.checkpointing.

Uses a temporary directory for all file I/O.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.resilience.checkpointing import CheckpointData, CheckpointManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_mgr(tmp_path):
    """A CheckpointManager with a temp directory and short intervals."""
    return CheckpointManager(checkpoint_dir=tmp_path / "ckpts", save_every=5, keep_last=3)


def _sample_state(n: int = 1) -> dict:
    return {"score": 0.5 * n, "cycle": n}


def _sample_kb_stats() -> dict:
    return {"total_entries": 10, "observations": 5}


# ---------------------------------------------------------------------------
# CheckpointData
# ---------------------------------------------------------------------------

class TestCheckpointData:
    def test_round_trip(self):
        cp = CheckpointData(
            cycle_number=42,
            state={"score": 0.99},
            study_name="test-study",
            kb_stats={"entries": 7},
            timestamp=1234567890.0,
        )
        d = cp.to_dict()
        restored = CheckpointData.from_dict(d)
        assert restored.cycle_number == 42
        assert restored.state == {"score": 0.99}
        assert restored.study_name == "test-study"
        assert restored.kb_stats == {"entries": 7}
        assert restored.timestamp == pytest.approx(1234567890.0)

    def test_from_dict_with_missing_fields(self):
        """from_dict should handle missing optional fields gracefully."""
        d = {"cycle_number": 1}
        cp = CheckpointData.from_dict(d)
        assert cp.cycle_number == 1
        assert cp.state == {}
        assert cp.study_name == ""
        assert cp.kb_stats == {}


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

class TestSave:
    def test_save_creates_file(self, tmp_mgr):
        path = tmp_mgr.save_checkpoint(
            cycle_number=5,
            state=_sample_state(),
            study_name="s",
            kb_stats=_sample_kb_stats(),
        )
        assert path is not None
        assert path.exists()

    def test_save_content_is_valid_json(self, tmp_mgr):
        path = tmp_mgr.save_checkpoint(5, _sample_state(), "s", _sample_kb_stats())
        data = json.loads(path.read_text())
        assert data["cycle_number"] == 5

    def test_skips_if_not_due(self, tmp_mgr):
        # save_every=5, cycle 3 should be skipped
        result = tmp_mgr.save_checkpoint(3, _sample_state(), "s", _sample_kb_stats())
        assert result is None

    def test_force_saves_regardless_of_cycle(self, tmp_mgr):
        path = tmp_mgr.save_checkpoint(3, _sample_state(), "s", _sample_kb_stats(), force=True)
        assert path is not None
        assert path.exists()

    def test_creates_dir_if_missing(self, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=tmp_path / "new" / "nested", save_every=1)
        path = mgr.save_checkpoint(1, {}, "s", {})
        assert path is not None and path.exists()

    def test_filename_contains_cycle_number(self, tmp_mgr):
        path = tmp_mgr.save_checkpoint(5, {}, "s", {})
        assert "00000005" in path.name

    def test_timestamp_is_recent(self, tmp_mgr):
        before = time.time()
        path = tmp_mgr.save_checkpoint(5, {}, "s", {})
        after = time.time()
        data = json.loads(path.read_text())
        assert before <= data["timestamp"] <= after


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

class TestLoad:
    def test_load_returns_none_if_no_files(self, tmp_mgr):
        result = tmp_mgr.load_checkpoint()
        assert result is None

    def test_load_returns_most_recent(self, tmp_mgr):
        tmp_mgr.save_checkpoint(5, {"v": 1}, "s", {})
        tmp_mgr.save_checkpoint(10, {"v": 2}, "s", {})

        cp = tmp_mgr.load_checkpoint()
        assert cp is not None
        assert cp.cycle_number == 10

    def test_load_returns_correct_state(self, tmp_mgr):
        tmp_mgr.save_checkpoint(5, {"score": 0.88}, "my-study", {"entries": 99})
        cp = tmp_mgr.load_checkpoint()
        assert cp.state == {"score": 0.88}
        assert cp.study_name == "my-study"
        assert cp.kb_stats == {"entries": 99}

    def test_load_from_explicit_dir(self, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=tmp_path / "cp", save_every=1)
        mgr.save_checkpoint(7, {"x": 7}, "s", {})
        cp = mgr.load_checkpoint(checkpoint_dir=tmp_path / "cp")
        assert cp is not None
        assert cp.cycle_number == 7

    def test_load_returns_none_on_corrupt_file(self, tmp_path):
        cp_dir = tmp_path / "cp"
        cp_dir.mkdir()
        (cp_dir / "checkpoint_cycle_00000005.json").write_text("not json!!!")
        mgr = CheckpointManager(checkpoint_dir=cp_dir, save_every=1)
        result = mgr.load_checkpoint()
        assert result is None


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------

class TestRotation:
    def test_keeps_only_last_n(self, tmp_mgr):
        # keep_last=3, save_every=5 → save at 5, 10, 15, 20, 25
        for cycle in range(5, 30, 5):
            tmp_mgr.save_checkpoint(cycle, {}, "s", {})

        remaining = tmp_mgr.list_checkpoints()
        assert len(remaining) <= 3

    def test_keeps_most_recent_files(self, tmp_mgr):
        for cycle in range(5, 30, 5):
            tmp_mgr.save_checkpoint(cycle, {"c": cycle}, "s", {})

        remaining = tmp_mgr.list_checkpoints()
        # The most recent should be cycle 25
        last_data = json.loads(remaining[-1].read_text())
        assert last_data["cycle_number"] == 25

    def test_exactly_keep_last_files(self, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=tmp_path / "cp", save_every=1, keep_last=2)
        for i in range(1, 6):
            mgr.save_checkpoint(i, {}, "s", {})
        assert len(mgr.list_checkpoints()) == 2


# ---------------------------------------------------------------------------
# Init validation
# ---------------------------------------------------------------------------

class TestInitValidation:
    def test_invalid_save_every(self, tmp_path):
        with pytest.raises(ValueError):
            CheckpointManager(checkpoint_dir=tmp_path, save_every=0)

    def test_invalid_keep_last(self, tmp_path):
        with pytest.raises(ValueError):
            CheckpointManager(checkpoint_dir=tmp_path, keep_last=0)
