"""
src.resilience.checkpointing — Cycle-level checkpoint save/load/rotation.

Usage
-----
    from src.resilience.checkpointing import CheckpointManager, CheckpointData

    mgr = CheckpointManager(checkpoint_dir="checkpoints", save_every=10, keep_last=5)

    # Save (no-op unless cycle_number % save_every == 0)
    mgr.save_checkpoint(
        cycle_number=10,
        state={"last_score": 0.75},
        study_name="my-run",
        kb_stats={"total_entries": 42},
    )

    # Or force-save (e.g. on SIGTERM):
    mgr.save_checkpoint(..., force=True)

    # Load most recent checkpoint on startup:
    cp = mgr.load_checkpoint()
    if cp:
        print(f"Resuming from cycle {cp.cycle_number}")
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CHECKPOINT_PREFIX = "checkpoint_cycle_"
_CHECKPOINT_SUFFIX = ".json"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CheckpointData:
    """
    Data stored in / loaded from a checkpoint file.

    Attributes
    ----------
    cycle_number : int
        The cycle index at which the checkpoint was saved.
    state : dict
        Arbitrary state dict (e.g. last scores, params).
    study_name : str
        Optuna study name associated with this run.
    kb_stats : dict
        Knowledge-base statistics snapshot.
    timestamp : float
        Unix epoch time at save.
    """

    cycle_number: int
    state: dict[str, Any]
    study_name: str
    kb_stats: dict[str, Any]
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CheckpointData":
        return cls(
            cycle_number=int(d["cycle_number"]),
            state=dict(d.get("state", {})),
            study_name=str(d.get("study_name", "")),
            kb_stats=dict(d.get("kb_stats", {})),
            timestamp=float(d.get("timestamp", 0.0)),
        )


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """
    Manages cycle checkpoints: auto-save every N cycles, keep last M files.

    Parameters
    ----------
    checkpoint_dir : str | Path
        Directory to write checkpoint JSON files.  Created if missing.
    save_every : int
        Automatically save on cycles where ``cycle_number % save_every == 0``.
    keep_last : int
        Number of most-recent checkpoint files to retain.  Older files are
        deleted automatically after each save.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path = "checkpoints",
        save_every: int = 10,
        keep_last: int = 5,
    ) -> None:
        if save_every < 1:
            raise ValueError("save_every must be >= 1")
        if keep_last < 1:
            raise ValueError("keep_last must be >= 1")

        self._dir = Path(checkpoint_dir)
        self._save_every = save_every
        self._keep_last = keep_last

        self._dir.mkdir(parents=True, exist_ok=True)
        logger.debug("CheckpointManager: directory=%s, every=%d, keep=%d",
                     self._dir, save_every, keep_last)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        cycle_number: int,
        state: dict[str, Any],
        study_name: str,
        kb_stats: dict[str, Any],
        force: bool = False,
    ) -> Path | None:
        """
        Save a checkpoint if due (or if ``force=True``).

        Parameters
        ----------
        cycle_number : int
            Current cycle index.
        state : dict
            Arbitrary state to persist.
        study_name : str
            Optuna study name.
        kb_stats : dict
            Knowledge-base statistics.
        force : bool
            If True, skip the every-N check and save unconditionally.

        Returns
        -------
        Path | None
            Path to the written file, or None if skipped.
        """
        if not force and not self._is_due(cycle_number):
            return None

        data = CheckpointData(
            cycle_number=cycle_number,
            state=state,
            study_name=study_name,
            kb_stats=kb_stats,
            timestamp=time.time(),
        )

        filename = f"{_CHECKPOINT_PREFIX}{cycle_number:08d}{_CHECKPOINT_SUFFIX}"
        path = self._dir / filename

        try:
            tmp_path = path.with_suffix(".tmp")
            tmp_path.write_text(
                json.dumps(data.to_dict(), indent=2, default=str),
                encoding="utf-8",
            )
            tmp_path.replace(path)  # atomic on same filesystem
            logger.info("CheckpointManager: saved cycle %d → %s", cycle_number, path)
        except OSError as exc:
            logger.error("CheckpointManager: failed to save checkpoint: %s", exc)
            return None

        self._rotate()
        return path

    def load_checkpoint(
        self,
        checkpoint_dir: str | Path | None = None,
    ) -> CheckpointData | None:
        """
        Load the most recent checkpoint from ``checkpoint_dir`` (or the
        manager's own directory if not specified).

        Returns None if no checkpoint files exist or parsing fails.
        """
        directory = Path(checkpoint_dir) if checkpoint_dir else self._dir
        files = self._sorted_checkpoints(directory)
        if not files:
            logger.info("CheckpointManager: no checkpoint files found in %s", directory)
            return None

        latest = files[-1]
        try:
            raw = latest.read_text(encoding="utf-8")
            data = CheckpointData.from_dict(json.loads(raw))
            logger.info(
                "CheckpointManager: loaded checkpoint from cycle %d (%s)",
                data.cycle_number,
                latest.name,
            )
            return data
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.error("CheckpointManager: failed to load %s: %s", latest, exc)
            return None

    def list_checkpoints(self) -> list[Path]:
        """Return all checkpoint files, sorted oldest → newest."""
        return self._sorted_checkpoints(self._dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_due(self, cycle_number: int) -> bool:
        return cycle_number > 0 and (cycle_number % self._save_every == 0)

    def _sorted_checkpoints(self, directory: Path) -> list[Path]:
        """Return checkpoint JSON files sorted by cycle number (ascending)."""
        if not directory.exists():
            return []
        files = sorted(
            directory.glob(f"{_CHECKPOINT_PREFIX}*{_CHECKPOINT_SUFFIX}"),
            key=lambda p: p.name,
        )
        return files

    def _rotate(self) -> None:
        """Delete checkpoint files beyond the keep_last limit."""
        files = self._sorted_checkpoints(self._dir)
        excess = files[: max(0, len(files) - self._keep_last)]
        for old in excess:
            try:
                old.unlink()
                logger.debug("CheckpointManager: deleted old checkpoint %s", old.name)
            except OSError as exc:
                logger.warning("CheckpointManager: could not delete %s: %s", old.name, exc)
