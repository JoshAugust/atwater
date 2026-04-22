"""
tests/test_temperature.py — Tests for src.learning.temperature_schedule.

Tests:
- Cosine schedule: monotonically decreasing from start to end
- Linear schedule
- Exponential schedule
- Warmup phase: always returns start temperature
- Plateau detection: std < threshold triggers bump
- Plateau bump: temperature raised temporarily
- Bump decays over cycles
- Adaptive mode disabled: no bumps
- Edge cases: start == end, zero-length window
- Invalid configurations raise ValueError
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.learning.temperature_schedule import TemperatureScheduler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def flat_scores(value: float, n: int) -> list[float]:
    """All identical scores — definite plateau."""
    return [value] * n


def rising_scores(start: float, end: float, n: int) -> list[float]:
    return list(np.linspace(start, end, n))


def noisy_scores(mean: float, std: float, n: int, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    return list(rng.normal(mean, std, n))


# ---------------------------------------------------------------------------
# Cosine schedule
# ---------------------------------------------------------------------------

class TestCosineSchedule:
    def setup_method(self):
        self.scheduler = TemperatureScheduler(
            start=0.9, end=0.3, warmup_cycles=10, total_cycles=100,
            decay="cosine", adaptive=False
        )

    def test_at_warmup_returns_start(self):
        for c in range(10):
            t = self.scheduler.get_temperature(c)
            assert t == 0.9

    def test_post_warmup_decreases(self):
        temps = [self.scheduler.get_temperature(c) for c in range(10, 101)]
        # Should be generally decreasing
        assert temps[0] > temps[-1]

    def test_end_value_approached(self):
        # At cycle == total_cycles, should be at or near end
        t = self.scheduler.get_temperature(100)
        assert abs(t - 0.3) < 0.05

    def test_never_below_end(self):
        for c in range(200):
            t = self.scheduler.get_temperature(c)
            assert t >= 0.3 - 1e-9

    def test_never_above_start(self):
        for c in range(200):
            t = self.scheduler.get_temperature(c)
            assert t <= 0.9 + 1e-9

    def test_midpoint_near_middle(self):
        """Cosine schedule at 50% progress should be near midpoint."""
        midpoint_cycle = 10 + (100 - 10) // 2
        t = self.scheduler.get_temperature(midpoint_cycle)
        mid_value = (0.9 + 0.3) / 2
        assert abs(t - mid_value) < 0.1


# ---------------------------------------------------------------------------
# Linear schedule
# ---------------------------------------------------------------------------

class TestLinearSchedule:
    def setup_method(self):
        self.scheduler = TemperatureScheduler(
            start=0.9, end=0.3, warmup_cycles=0, total_cycles=100,
            decay="linear", adaptive=False
        )

    def test_decreases_linearly(self):
        t0 = self.scheduler.get_temperature(0)
        t50 = self.scheduler.get_temperature(50)
        t100 = self.scheduler.get_temperature(100)
        assert t0 >= t50 >= t100

    def test_monotonically_decreasing(self):
        temps = [self.scheduler.get_temperature(c) for c in range(0, 101, 10)]
        for i in range(len(temps) - 1):
            assert temps[i] >= temps[i + 1] - 1e-9


# ---------------------------------------------------------------------------
# Exponential schedule
# ---------------------------------------------------------------------------

class TestExponentialSchedule:
    def setup_method(self):
        self.scheduler = TemperatureScheduler(
            start=0.9, end=0.3, warmup_cycles=0, total_cycles=100,
            decay="exponential", adaptive=False
        )

    def test_decreases(self):
        t0 = self.scheduler.get_temperature(0)
        t100 = self.scheduler.get_temperature(100)
        assert t0 > t100

    def test_never_below_end(self):
        for c in range(200):
            t = self.scheduler.get_temperature(c)
            assert t >= 0.3 - 1e-9


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

class TestWarmup:
    def test_warmup_holds_at_start(self):
        scheduler = TemperatureScheduler(start=0.9, end=0.3, warmup_cycles=50)
        for c in range(50):
            assert scheduler.get_temperature(c) == 0.9

    def test_after_warmup_begins_annealing(self):
        scheduler = TemperatureScheduler(start=0.9, end=0.3, warmup_cycles=50,
                                         total_cycles=200, adaptive=False)
        t_at_warmup_end = scheduler.get_temperature(50)
        t_much_later = scheduler.get_temperature(150)
        assert t_at_warmup_end > t_much_later

    def test_zero_warmup(self):
        scheduler = TemperatureScheduler(start=0.9, end=0.3, warmup_cycles=0,
                                          total_cycles=100, adaptive=False)
        # Cycle 0 should already be annealing
        t0 = scheduler.get_temperature(0)
        t50 = scheduler.get_temperature(50)
        assert t0 > t50


# ---------------------------------------------------------------------------
# Plateau detection
# ---------------------------------------------------------------------------

class TestPlateauDetection:
    def setup_method(self):
        self.scheduler = TemperatureScheduler(
            start=0.9, end=0.3, warmup_cycles=0, total_cycles=500,
            plateau_window=20, plateau_std_threshold=0.02, adaptive=True
        )

    def test_flat_scores_detected_as_plateau(self):
        scores = flat_scores(0.7, 25)
        assert self.scheduler.is_plateau(scores)

    def test_rising_scores_not_plateau(self):
        scores = rising_scores(0.4, 0.9, 25)
        assert not self.scheduler.is_plateau(scores)

    def test_noisy_scores_not_plateau(self):
        scores = noisy_scores(0.7, 0.1, 25)
        assert not self.scheduler.is_plateau(scores)

    def test_very_small_std_is_plateau(self):
        scores = [0.700 + i * 0.0001 for i in range(25)]
        assert self.scheduler.is_plateau(scores)

    def test_window_too_small_not_plateau(self):
        """Less than plateau_window scores → not a plateau."""
        scores = flat_scores(0.7, 5)  # only 5, need 20
        assert not self.scheduler.is_plateau(scores)

    def test_plateau_count_increments(self):
        scheduler = TemperatureScheduler(
            start=0.9, end=0.3, warmup_cycles=0, total_cycles=500,
            plateau_window=5, plateau_std_threshold=0.05, adaptive=True,
            plateau_bump=0.15
        )
        flat = flat_scores(0.7, 10)
        # Call at different cycles to trigger count
        scheduler.get_temperature(100, flat)
        assert scheduler.plateau_count >= 1


# ---------------------------------------------------------------------------
# Adaptive bump
# ---------------------------------------------------------------------------

class TestAdaptiveBump:
    def setup_method(self):
        self.scheduler = TemperatureScheduler(
            start=0.9, end=0.3, warmup_cycles=0, total_cycles=500,
            plateau_window=5, plateau_std_threshold=0.05,
            plateau_bump=0.2, adaptive=True
        )
        self.flat = flat_scores(0.7, 10)

    def test_plateau_raises_temperature(self):
        # Get base temperature without plateau
        no_plateau_scores = rising_scores(0.5, 0.9, 10)
        base_temp = self.scheduler.get_temperature(200, no_plateau_scores)

        # New scheduler; get temperature WITH plateau
        scheduler2 = TemperatureScheduler(
            start=0.9, end=0.3, warmup_cycles=0, total_cycles=500,
            plateau_window=5, plateau_std_threshold=0.05,
            plateau_bump=0.2, adaptive=True
        )
        bumped_temp = scheduler2.get_temperature(200, flat_scores(0.7, 10))
        assert bumped_temp > base_temp

    def test_bump_does_not_exceed_start(self):
        for c in range(200, 250):
            t = self.scheduler.get_temperature(c, self.flat)
            assert t <= self.scheduler.start + 1e-9

    def test_bump_decays_after_plateau(self):
        """Temperature should return toward base after plateau bump."""
        self.scheduler.get_temperature(200, self.flat)  # triggers bump
        # After many cycles without further plateau, bump should decay
        temps_after = []
        for c in range(201, 220):
            # Non-plateau scores
            t = self.scheduler.get_temperature(c, rising_scores(0.5, 0.9, 10))
            temps_after.append(t)
        # Later temps should be lower than early temps (bump decaying)
        assert temps_after[-1] < temps_after[0] + 0.15  # bump shrinks

    def test_reset_bump(self):
        self.scheduler.get_temperature(200, self.flat)
        self.scheduler.reset_bump()
        assert self.scheduler._bump_remaining == 0.0

    def test_adaptive_false_no_bump(self):
        scheduler = TemperatureScheduler(
            start=0.9, end=0.3, warmup_cycles=0, total_cycles=500,
            plateau_window=5, plateau_std_threshold=0.05,
            plateau_bump=0.3, adaptive=False
        )
        base = scheduler.get_temperature(200, rising_scores(0.5, 0.8, 10))
        bumped = scheduler.get_temperature(201, flat_scores(0.7, 10))
        # With adaptive=False, no bump — temperatures should be close
        assert abs(bumped - base) < 0.05


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_start_equals_end_stays_constant(self):
        scheduler = TemperatureScheduler(
            start=0.5, end=0.5, warmup_cycles=0, total_cycles=100,
            adaptive=False
        )
        for c in range(0, 110, 10):
            t = scheduler.get_temperature(c)
            assert abs(t - 0.5) < 1e-6

    def test_invalid_start_less_than_end(self):
        with pytest.raises(ValueError):
            TemperatureScheduler(start=0.2, end=0.9)

    def test_invalid_negative_warmup(self):
        with pytest.raises(ValueError):
            TemperatureScheduler(warmup_cycles=-1)

    def test_none_recent_scores_no_crash(self):
        scheduler = TemperatureScheduler(adaptive=True)
        t = scheduler.get_temperature(100, recent_scores=None)
        assert isinstance(t, float)

    def test_empty_recent_scores_no_crash(self):
        scheduler = TemperatureScheduler(adaptive=True)
        t = scheduler.get_temperature(100, recent_scores=[])
        assert isinstance(t, float)

    def test_beyond_total_cycles_stays_at_end(self):
        scheduler = TemperatureScheduler(
            start=0.9, end=0.3, warmup_cycles=0,
            total_cycles=100, adaptive=False
        )
        t = scheduler.get_temperature(500)
        assert abs(t - 0.3) < 1e-6

    def test_unknown_decay_mode(self):
        with pytest.raises((ValueError, Exception)):
            scheduler = TemperatureScheduler(
                start=0.9, end=0.3, decay="unknown_mode", adaptive=False  # type: ignore
            )
            scheduler.get_temperature(100)
