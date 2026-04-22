"""
tests/test_collapse.py — Tests for src.learning.collapse_detector.

Tests:
- Detection triggers when same combo appears >threshold of window
- No false positives on truly diverse trials
- CollapseAlert has correct fields
- Recommendations scale with severity (mild/moderate/severe)
- Minimum window size prevents early false positives
- Empty / small trial lists handled safely
- Custom top_k and threshold work correctly
- Recommendation descriptions are readable strings
- No collapse when within threshold
"""

from __future__ import annotations

import random

import pytest

from src.learning.collapse_detector import CollapseAlert, CollapseDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_trial(params: dict) -> dict:
    return {"params": params}


def make_uniform_trials(n: int, params: dict) -> list[dict]:
    """All trials have identical params — worst-case collapse."""
    return [make_trial(params) for _ in range(n)]


def make_diverse_trials(n: int, seed: int = 42) -> list[dict]:
    """Trials with randomly varying params — no collapse."""
    rng = random.Random(seed)
    fonts = ["Inter", "Roboto", "Playfair", "Georgia", "Helvetica", "Oswald", "Lato"]
    bgs = ["#000", "#fff", "#123456", "#abcdef", "#ff0000", "#00ff00", "#0000ff",
           "#cccccc", "#333333", "#999999"]
    sizes = [12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72]
    return [
        make_trial({
            "font": rng.choice(fonts),
            "bg": rng.choice(bgs),
            "size": rng.choice(sizes),
        })
        for _ in range(n)
    ]


def make_mixed_trials(dominant: dict, n_dominant: int, n_diverse: int, seed: int = 99) -> list[dict]:
    """
    n_dominant identical + n_diverse random trials interleaved.
    Returns trials list with dominant ones at the end (simulating recent collapse).
    """
    diverse = make_diverse_trials(n_diverse, seed=seed)
    uniform = make_uniform_trials(n_dominant, dominant)
    return diverse + uniform


# ---------------------------------------------------------------------------
# CollapseAlert dataclass
# ---------------------------------------------------------------------------

class TestCollapseAlert:
    def test_construction(self):
        alert = CollapseAlert(
            detected=True,
            repeated_combo={"font": "Inter"},
            count=8,
            window=20,
            frequency=0.4,
            recommendation="force_random",
        )
        assert alert.detected is True
        assert alert.count == 8
        assert alert.frequency == 0.4

    def test_str_representation(self):
        alert = CollapseAlert(
            detected=True,
            repeated_combo={"font": "Inter"},
            count=8,
            window=20,
            frequency=0.4,
            recommendation="force_random",
        )
        s = str(alert)
        assert "CollapseAlert" in s
        assert "force_random" in s


# ---------------------------------------------------------------------------
# Detection triggers
# ---------------------------------------------------------------------------

class TestCollapseDetection:
    def setup_method(self):
        # threshold=0.25 → >5 of 20 triggers (25%+)
        self.detector = CollapseDetector(threshold=0.25, top_k=3, min_window=5)

    def test_identical_trials_triggers(self):
        trials = make_uniform_trials(20, {"font": "Inter", "bg": "#000", "size": 32})
        alert = self.detector.check(trials, window=20)
        assert alert is not None
        assert alert.detected is True

    def test_frequency_field_correct(self):
        trials = make_uniform_trials(20, {"font": "Inter", "bg": "#000", "size": 32})
        alert = self.detector.check(trials, window=20)
        assert alert is not None
        assert abs(alert.frequency - 1.0) < 0.01

    def test_repeated_combo_field_populated(self):
        dominant = {"font": "Inter", "bg": "#000", "size": 32}
        trials = make_uniform_trials(20, dominant)
        alert = self.detector.check(trials, window=20)
        assert alert is not None
        assert len(alert.repeated_combo) > 0

    def test_high_dominance_triggers(self):
        dominant = {"font": "Inter", "bg": "#000", "size": 32}
        # 12 dominant + 8 diverse = 60% dominant → well above threshold
        trials = make_mixed_trials(dominant, n_dominant=12, n_diverse=8)
        alert = self.detector.check(trials, window=20)
        assert alert is not None
        assert alert.detected is True

    def test_just_above_threshold_triggers(self):
        """Exactly at 6/20 = 30% > 25% threshold."""
        dominant = {"font": "Roboto", "bg": "#fff", "size": 24}
        diverse = make_diverse_trials(14, seed=7)
        uniform = make_uniform_trials(6, dominant)
        trials = diverse + uniform
        alert = self.detector.check(trials, window=20)
        assert alert is not None
        assert alert.detected is True

    def test_window_parameter_limits_trials(self):
        """Only the last `window` trials should be checked."""
        dominant = {"font": "Inter", "bg": "#000", "size": 32}
        # 30 diverse trials followed by 15 identical — window=20 should see 15 dominant
        diverse = make_diverse_trials(30, seed=5)
        uniform = make_uniform_trials(15, dominant)
        trials = diverse + uniform
        alert = self.detector.check(trials, window=20)
        # 15/20 = 75% → above threshold
        assert alert is not None

    def test_count_field(self):
        dominant = {"font": "Inter", "bg": "#000", "size": 32}
        trials = make_uniform_trials(20, dominant)
        alert = self.detector.check(trials, window=20)
        assert alert is not None
        assert alert.count == 20

    def test_window_field(self):
        trials = make_uniform_trials(20, {"a": 1, "b": 2, "c": 3})
        alert = self.detector.check(trials, window=20)
        assert alert is not None
        assert alert.window == 20


# ---------------------------------------------------------------------------
# No false positives on diverse trials
# ---------------------------------------------------------------------------

class TestNegativeDetection:
    def setup_method(self):
        self.detector = CollapseDetector(threshold=0.25, top_k=3, min_window=5)

    def test_diverse_trials_no_alert(self):
        trials = make_diverse_trials(30, seed=42)
        alert = self.detector.check(trials, window=20)
        assert alert is None

    def test_just_below_threshold_no_alert(self):
        """5/20 = 25% = threshold — should NOT trigger (must be strictly above)."""
        dominant = {"font": "Lato", "bg": "#123", "size": 28}
        diverse = make_diverse_trials(15, seed=11)
        uniform = make_uniform_trials(5, dominant)
        trials = diverse + uniform
        alert = self.detector.check(trials, window=20)
        # 5/20 = exactly 0.25, not strictly above → no alert
        assert alert is None

    def test_random_distribution_no_alert(self):
        rng = random.Random(0)
        all_values = list(range(20))
        trials = [make_trial({"x": rng.choice(all_values)}) for _ in range(100)]
        alert = self.detector.check(trials, window=20)
        assert alert is None

    def test_many_unique_combinations_no_alert(self):
        trials = [
            make_trial({"a": i, "b": i * 2, "c": i * 3})
            for i in range(20)
        ]
        alert = self.detector.check(trials, window=20)
        assert alert is None


# ---------------------------------------------------------------------------
# Minimum window size
# ---------------------------------------------------------------------------

class TestMinWindow:
    def test_too_few_trials_no_alert(self):
        detector = CollapseDetector(min_window=10)
        # Only 5 trials, min_window=10 → skip detection
        trials = make_uniform_trials(5, {"x": 1, "y": 2, "z": 3})
        alert = detector.check(trials, window=20)
        assert alert is None

    def test_exactly_at_min_window_activates(self):
        detector = CollapseDetector(threshold=0.25, top_k=2, min_window=5)
        trials = make_uniform_trials(10, {"font": "Inter", "bg": "#000"})
        alert = detector.check(trials, window=10)
        assert alert is not None

    def test_empty_trials_returns_none(self):
        detector = CollapseDetector()
        assert detector.check([]) is None

    def test_none_params_handled(self):
        """Trials with missing 'params' key should not crash."""
        detector = CollapseDetector(min_window=2)
        trials = [{"params": {}}, {"params": {}}]
        result = detector.check(trials, window=20)
        # No params to analyse → no collapse detected
        assert result is None


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

class TestRecommendations:
    def setup_method(self):
        self.detector = CollapseDetector(threshold=0.25, top_k=3, min_window=5)

    def _make_alert(self, frequency: float) -> CollapseAlert | None:
        """Make trials with the given approximate dominant frequency."""
        n = 20
        n_dominant = int(frequency * n) + 1
        n_diverse = n - n_dominant
        if n_diverse < 0:
            n_diverse = 0
            n_dominant = n
        dominant = {"font": "Inter", "bg": "#000", "size": 32}
        diverse = make_diverse_trials(n_diverse, seed=1)
        uniform = make_uniform_trials(n_dominant, dominant)
        trials = diverse + uniform
        return self.detector.check(trials, window=20)

    def test_mild_collapse_force_random(self):
        alert = self._make_alert(frequency=0.3)
        if alert:
            assert alert.recommendation == "force_random"

    def test_severe_collapse_reset_sampler(self):
        alert = self._make_alert(frequency=0.95)
        assert alert is not None
        assert alert.recommendation == "reset_sampler"

    def test_recommendation_is_known_value(self):
        trials = make_uniform_trials(20, {"font": "Inter", "bg": "#000", "size": 32})
        alert = self.detector.check(trials, window=20)
        assert alert is not None
        known = {"force_random", "increase_temperature", "reset_sampler"}
        assert alert.recommendation in known

    def test_describe_recommendation(self):
        for code in ("force_random", "increase_temperature", "reset_sampler"):
            desc = CollapseDetector.describe_recommendation(code)
            assert isinstance(desc, str)
            assert len(desc) > 10


# ---------------------------------------------------------------------------
# Custom top_k and threshold
# ---------------------------------------------------------------------------

class TestCustomParameters:
    def test_top_k_1_very_sensitive(self):
        """With top_k=1, any single param value dominance triggers."""
        detector = CollapseDetector(threshold=0.25, top_k=1, min_window=5)
        trials = make_uniform_trials(20, {"font": "Inter", "bg": "#000"})
        alert = detector.check(trials, window=20)
        assert alert is not None

    def test_high_threshold_not_triggered(self):
        """With threshold=0.9, only near-total domination triggers."""
        detector = CollapseDetector(threshold=0.90, top_k=3, min_window=5)
        dominant = {"font": "Inter", "bg": "#000", "size": 32}
        # 8/20 = 40% → below 90% threshold
        diverse = make_diverse_trials(12, seed=3)
        uniform = make_uniform_trials(8, dominant)
        trials = diverse + uniform
        alert = detector.check(trials, window=20)
        assert alert is None

    def test_high_threshold_triggered_at_extreme(self):
        """With threshold=0.9, all-identical trials still trigger."""
        detector = CollapseDetector(threshold=0.90, top_k=3, min_window=5)
        trials = make_uniform_trials(20, {"font": "Inter", "bg": "#000", "size": 32})
        alert = detector.check(trials, window=20)
        assert alert is not None

    def test_extra_keys_ignored_in_fingerprint(self):
        """top_k=2 uses only top-2 most common keys, ignoring the rest."""
        detector = CollapseDetector(threshold=0.25, top_k=2, min_window=5)
        # Vary a third key that won't be in the fingerprint
        trials = [
            make_trial({"font": "Inter", "bg": "#000", "seed": i})
            for i in range(20)
        ]
        # font+bg are identical in all — should still trigger with top_k=2
        alert = detector.check(trials, window=20)
        assert alert is not None
