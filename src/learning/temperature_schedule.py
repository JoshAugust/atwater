"""
src.learning.temperature_schedule — Adaptive LLM temperature scheduling.

Implements cosine annealing with an adaptive plateau-detection mode.
Temperature starts high (exploration) and decreases as quality improves.
When scores plateau, the scheduler temporarily bumps temperature to
escape local minima.

Usage
-----
    scheduler = TemperatureScheduler(start=0.9, end=0.3, warmup_cycles=50)
    temp = scheduler.get_temperature(cycle_number=100, recent_scores=[0.7, 0.71, 0.72])
"""

from __future__ import annotations

import logging
import math
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLATEAU_WINDOW: int = 20          # number of recent scores to examine
PLATEAU_STD_THRESHOLD: float = 0.02   # std below this → plateau detected
PLATEAU_BUMP_AMOUNT: float = 0.15     # how much to raise temp on plateau
PLATEAU_BUMP_DECAY: float = 0.85      # how fast the bump decays per cycle

DecayMode = Literal["cosine", "linear", "exponential"]


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class TemperatureScheduler:
    """
    Adaptive temperature scheduler for LLM calls.

    Parameters
    ----------
    start:
        Starting temperature (high — for exploration).
    end:
        Minimum temperature (low — for exploitation).
    warmup_cycles:
        Number of cycles before annealing begins.  During warmup, always
        returns ``start``.
    total_cycles:
        Expected total cycles (used for cosine schedule calculation).
        Defaults to 500 if not specified.
    decay:
        Annealing strategy: "cosine" (default), "linear", or "exponential".
    plateau_window:
        Number of recent scores to use for plateau detection.
    plateau_std_threshold:
        If std of recent scores < this value → plateau detected.
    plateau_bump:
        How much to temporarily raise temperature when a plateau is detected.
    adaptive:
        If True, enables plateau detection and temperature bumping.
    """

    def __init__(
        self,
        start: float = 0.9,
        end: float = 0.3,
        warmup_cycles: int = 50,
        total_cycles: int = 500,
        decay: DecayMode = "cosine",
        plateau_window: int = PLATEAU_WINDOW,
        plateau_std_threshold: float = PLATEAU_STD_THRESHOLD,
        plateau_bump: float = PLATEAU_BUMP_AMOUNT,
        adaptive: bool = True,
    ) -> None:
        if start < end:
            raise ValueError(f"start ({start}) must be >= end ({end})")
        if warmup_cycles < 0:
            raise ValueError("warmup_cycles must be non-negative")

        self.start = start
        self.end = end
        self.warmup_cycles = warmup_cycles
        self.total_cycles = total_cycles
        self.decay = decay
        self.plateau_window = plateau_window
        self.plateau_std_threshold = plateau_std_threshold
        self.plateau_bump = plateau_bump
        self.adaptive = adaptive

        # Bump state
        self._bump_remaining: float = 0.0
        self._plateau_count: int = 0
        self._last_plateau_cycle: int = -1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_temperature(
        self,
        cycle_number: int,
        recent_scores: list[float] | None = None,
    ) -> float:
        """
        Return the temperature for this cycle.

        Parameters
        ----------
        cycle_number:
            Current cycle index (0-based).
        recent_scores:
            Most recent scores from the last N cycles.  Used for plateau
            detection when ``adaptive=True``.

        Returns
        -------
        float
            Temperature in [end, start].
        """
        # Warmup phase: hold at start temperature
        if cycle_number < self.warmup_cycles:
            logger.debug(
                "[TempScheduler] Cycle %d in warmup — temp=%.3f", cycle_number, self.start
            )
            return self.start

        # Compute base annealed temperature
        base_temp = self._anneal(cycle_number)

        # Adaptive plateau bump
        bump = 0.0
        if self.adaptive and recent_scores and len(recent_scores) >= self.plateau_window:
            if self._is_plateau(recent_scores):
                if self._last_plateau_cycle != cycle_number:
                    self._last_plateau_cycle = cycle_number
                    self._plateau_count += 1
                    self._bump_remaining = self.plateau_bump
                    logger.info(
                        "[TempScheduler] Plateau detected at cycle %d (std=%.4f) "
                        "— bumping temperature by %.3f. Plateau #%d.",
                        cycle_number,
                        float(np.std(recent_scores[-self.plateau_window:])),
                        self.plateau_bump,
                        self._plateau_count,
                    )

        if self._bump_remaining > 0:
            bump = self._bump_remaining
            self._bump_remaining *= PLATEAU_BUMP_DECAY
            if self._bump_remaining < 0.001:
                self._bump_remaining = 0.0

        final_temp = min(self.start, base_temp + bump)
        logger.debug(
            "[TempScheduler] Cycle %d: base=%.3f bump=%.3f final=%.3f",
            cycle_number, base_temp, bump, final_temp,
        )
        return round(final_temp, 4)

    def is_plateau(self, recent_scores: list[float]) -> bool:
        """Public helper: return True if scores indicate a plateau."""
        return self._is_plateau(recent_scores)

    @property
    def plateau_count(self) -> int:
        """How many plateau events have been detected so far."""
        return self._plateau_count

    def reset_bump(self) -> None:
        """Manually clear any active temperature bump."""
        self._bump_remaining = 0.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _anneal(self, cycle_number: int) -> float:
        """Compute the annealed temperature for cycle_number (post-warmup)."""
        effective_cycle = cycle_number - self.warmup_cycles
        effective_total = max(self.total_cycles - self.warmup_cycles, 1)
        t = min(effective_cycle / effective_total, 1.0)  # 0 → 1

        if self.decay == "cosine":
            # Cosine annealing: smooth decay from start to end
            temp = self.end + 0.5 * (self.start - self.end) * (1 + math.cos(math.pi * t))
        elif self.decay == "linear":
            temp = self.start - t * (self.start - self.end)
        elif self.decay == "exponential":
            # Exponential decay: start * (end/start)^t
            if self.start > 0 and self.end > 0:
                temp = self.start * (self.end / self.start) ** t
            else:
                temp = max(self.end, self.start * (1 - t))
        else:
            raise ValueError(f"Unknown decay mode: {self.decay!r}")

        return float(np.clip(temp, self.end, self.start))

    def _is_plateau(self, recent_scores: list[float]) -> bool:
        """Return True if the last ``plateau_window`` scores have low std."""
        window = recent_scores[-self.plateau_window:]
        if len(window) < self.plateau_window:
            return False
        std = float(np.std(window))
        return std < self.plateau_std_threshold
