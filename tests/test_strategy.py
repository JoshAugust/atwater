"""
tests/test_strategy.py — Tests for src.learning.strategy_selector.

Tests:
- select_strategy returns a valid strategy
- update modifies alpha/beta correctly
- Thompson sampling convergence (reward signal drives strategy selection)
- get_stats returns correct structure
- best_strategy reflects highest win rate
- Persistence: save/load state to JSON
- Reset clears arm state back to priors
- Unknown strategy in update is handled gracefully
- Seed-based reproducibility
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.learning.strategy_selector import (
    STRATEGIES,
    ArmState,
    StrategySelector,
)


# ---------------------------------------------------------------------------
# ArmState unit tests
# ---------------------------------------------------------------------------

class TestArmState:
    def test_win_rate_formula(self):
        arm = ArmState(strategy="exploit", alpha=3.0, beta=1.0)
        expected = 3.0 / (3.0 + 1.0)
        assert abs(arm.win_rate - expected) < 1e-9

    def test_sample_in_unit_interval(self):
        arm = ArmState(strategy="explore", alpha=2.0, beta=2.0)
        for _ in range(100):
            s = arm.sample()
            assert 0.0 <= s <= 1.0

    def test_sample_with_rng(self):
        rng = np.random.default_rng(42)
        arm = ArmState(strategy="refine", alpha=1.0, beta=1.0)
        s = arm.sample(rng=rng)
        assert 0.0 <= s <= 1.0

    def test_to_dict_roundtrip(self):
        arm = ArmState(strategy="hypothesis", alpha=5.0, beta=3.0,
                       total_pulls=10, total_reward=7.5)
        d = arm.to_dict()
        arm2 = ArmState.from_dict(d)
        assert arm2.strategy == arm.strategy
        assert arm2.alpha == arm.alpha
        assert arm2.beta == arm.beta
        assert arm2.total_pulls == arm.total_pulls
        assert arm2.total_reward == arm.total_reward


# ---------------------------------------------------------------------------
# StrategySelector basic operations
# ---------------------------------------------------------------------------

class TestStrategySelectorBasic:
    def setup_method(self):
        self.selector = StrategySelector(seed=42)

    def test_select_returns_valid_strategy(self):
        for _ in range(50):
            strategy = self.selector.select_strategy()
            assert strategy in STRATEGIES

    def test_update_increases_alpha_on_high_reward(self):
        initial_alpha = self.selector._arms["exploit"].alpha
        self.selector.update("exploit", reward=0.9)
        assert self.selector._arms["exploit"].alpha > initial_alpha

    def test_update_increases_beta_on_low_reward(self):
        initial_beta = self.selector._arms["explore"].beta
        self.selector.update("explore", reward=0.1)
        assert self.selector._arms["explore"].beta > initial_beta

    def test_update_tracks_total_pulls(self):
        self.selector.update("refine", reward=0.7)
        self.selector.update("refine", reward=0.8)
        assert self.selector._arms["refine"].total_pulls == 2

    def test_update_tracks_total_reward(self):
        self.selector.update("hypothesis", reward=0.6)
        self.selector.update("hypothesis", reward=0.4)
        assert abs(self.selector._arms["hypothesis"].total_reward - 1.0) < 1e-9

    def test_update_unknown_strategy_no_crash(self):
        # Should log warning and not raise
        self.selector.update("nonexistent_strategy", reward=0.5)

    def test_alpha_beta_sum_grows_with_updates(self):
        """Each update should increase alpha + beta by 1 (reward + (1-reward) = 1)."""
        arm = self.selector._arms["exploit"]
        initial_sum = arm.alpha + arm.beta
        self.selector.update("exploit", reward=0.75)
        new_sum = arm.alpha + arm.beta
        assert abs((new_sum - initial_sum) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Thompson sampling convergence
# ---------------------------------------------------------------------------

class TestThompsonSamplingConvergence:
    """Test that the bandit learns to prefer strategies with higher rewards."""

    def test_high_reward_strategy_selected_more_often(self):
        """
        After feeding consistently high rewards for "exploit" and low for
        others, "exploit" should be selected significantly more often.
        """
        selector = StrategySelector(seed=99)

        # Train with consistent signals
        for _ in range(100):
            selector.update("exploit", reward=0.95)
        for _ in range(100):
            selector.update("explore", reward=0.1)
            selector.update("hypothesis", reward=0.1)
            selector.update("refine", reward=0.1)

        # Count selections over many draws
        counts = {s: 0 for s in STRATEGIES}
        for _ in range(500):
            chosen = selector.select_strategy()
            counts[chosen] += 1

        # "exploit" should dominate
        assert counts["exploit"] > counts["explore"] * 3
        assert counts["exploit"] > counts["hypothesis"] * 3

    def test_win_rate_reflects_reward_history(self):
        selector = StrategySelector(seed=7)
        for _ in range(50):
            selector.update("refine", reward=0.9)
        for _ in range(50):
            selector.update("explore", reward=0.2)

        stats = selector.get_stats()
        assert stats["refine"]["win_rate"] > stats["explore"]["win_rate"]

    def test_best_strategy_matches_highest_win_rate(self):
        selector = StrategySelector(seed=13)
        for _ in range(30):
            selector.update("hypothesis", reward=0.85)
        for _ in range(30):
            selector.update("exploit", reward=0.4)

        assert selector.best_strategy() == "hypothesis"


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------

class TestGetStats:
    def test_stats_has_all_strategies(self):
        selector = StrategySelector()
        stats = selector.get_stats()
        for s in STRATEGIES:
            assert s in stats

    def test_stats_keys(self):
        selector = StrategySelector()
        stats = selector.get_stats()
        for s, data in stats.items():
            assert "alpha" in data
            assert "beta" in data
            assert "win_rate" in data
            assert "total_pulls" in data
            assert "total_reward" in data

    def test_fresh_selector_zero_pulls(self):
        selector = StrategySelector()
        stats = selector.get_stats()
        for s in STRATEGIES:
            assert stats[s]["total_pulls"] == 0

    def test_win_rate_in_unit_interval(self):
        selector = StrategySelector()
        for _ in range(10):
            selector.update("exploit", reward=0.7)
        stats = selector.get_stats()
        for s, data in stats.items():
            assert 0.0 <= data["win_rate"] <= 1.0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_creates_file(self, tmp_path):
        path = tmp_path / "strategy.json"
        selector = StrategySelector(save_path=path)
        selector.update("exploit", reward=0.8)
        selector.update("explore", reward=0.3)
        # update auto-saves
        assert path.exists()

    def test_save_load_roundtrip(self, tmp_path):
        path = tmp_path / "strategy.json"
        selector = StrategySelector(save_path=path, seed=42)
        for _ in range(5):
            selector.update("exploit", reward=0.9)
        stats_before = selector.get_stats()

        # Load into fresh selector
        selector2 = StrategySelector(save_path=path, seed=42)
        stats_after = selector2.get_stats()

        assert abs(stats_before["exploit"]["alpha"] - stats_after["exploit"]["alpha"]) < 1e-9
        assert stats_before["exploit"]["total_pulls"] == stats_after["exploit"]["total_pulls"]

    def test_load_missing_file_starts_fresh(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        selector = StrategySelector(save_path=path)
        # No crash; should have all strategies
        assert set(selector._arms.keys()) == set(STRATEGIES)

    def test_load_corrupted_file_starts_fresh(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not valid json {{{{")
        selector = StrategySelector(save_path=path)
        # Should recover gracefully
        assert set(selector._arms.keys()) == set(STRATEGIES)

    def test_save_load_preserves_all_arms(self, tmp_path):
        path = tmp_path / "all_arms.json"
        selector = StrategySelector(save_path=path)
        for s in STRATEGIES:
            selector.update(s, reward=0.5)

        selector2 = StrategySelector(save_path=path)
        for s in STRATEGIES:
            assert selector2._arms[s].total_pulls == 1

    def test_json_structure(self, tmp_path):
        path = tmp_path / "check.json"
        selector = StrategySelector(save_path=path)
        selector.update("exploit", reward=0.7)
        data = json.loads(path.read_text())
        assert "exploit" in data
        assert "alpha" in data["exploit"]
        assert "beta" in data["exploit"]


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_pulls(self):
        selector = StrategySelector(seed=0)
        for _ in range(10):
            selector.update("exploit", reward=0.9)
        selector.reset()
        assert selector._arms["exploit"].total_pulls == 0

    def test_reset_restores_priors(self):
        from src.learning.strategy_selector import DEFAULT_PRIORS
        selector = StrategySelector(seed=0)
        for _ in range(20):
            selector.update("exploit", reward=0.1)  # drive alpha down
        selector.reset()
        # alpha should be back near prior
        expected_alpha = DEFAULT_PRIORS["exploit"]["alpha"]
        assert abs(selector._arms["exploit"].alpha - expected_alpha) < 1e-9


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_gives_same_selections(self):
        s1 = StrategySelector(seed=123)
        s2 = StrategySelector(seed=123)
        selections1 = [s1.select_strategy() for _ in range(20)]
        selections2 = [s2.select_strategy() for _ in range(20)]
        assert selections1 == selections2

    def test_different_seed_gives_different_selections(self):
        s1 = StrategySelector(seed=1)
        s2 = StrategySelector(seed=9999)
        selections1 = [s1.select_strategy() for _ in range(20)]
        selections2 = [s2.select_strategy() for _ in range(20)]
        # Very unlikely to be identical with 4 strategies over 20 draws
        assert selections1 != selections2
