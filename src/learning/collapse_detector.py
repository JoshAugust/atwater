"""
src.learning.collapse_detector — Mode collapse detection and prevention.

Detects when the optimisation loop converges to a narrow, repetitive region
of parameter space — the "mode collapse" failure mode.

Detection heuristic
-------------------
If the same top-3 parameter values appear together in more than ``threshold``
of the last ``window`` trials → collapse detected.

Integration point
-----------------
DiversityGuard calls check() each cycle.  If a CollapseAlert is returned,
it follows the recommendation:
  - "force_random"         → inject a RandomSampler trial
  - "increase_temperature" → raise LLM/sampler temperature
  - "reset_sampler"        → reset Optuna sampler to TPE with fresh state

Usage
-----
    detector = CollapseDetector()
    trials = [{"params": {"font": "Inter", "bg": "#fff", "size": 24}}, ...]
    alert = detector.check(trials, window=20)
    if alert:
        print(alert.recommendation)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CollapseAlert:
    """
    Result of a collapse detection check.

    Attributes
    ----------
    detected:
        True if collapse is detected.
    repeated_combo:
        The parameter combination that is dominating recent trials.
    count:
        How many times the combo appeared in the window.
    window:
        The window size used for this detection.
    frequency:
        ``count / window`` — fraction of trials showing the dominant combo.
    recommendation:
        Action suggestion: "force_random", "increase_temperature", or
        "reset_sampler".
    details:
        Additional context string.
    """

    detected: bool
    repeated_combo: dict[str, Any]
    count: int
    window: int
    frequency: float
    recommendation: str
    details: str = ""

    def __str__(self) -> str:
        return (
            f"CollapseAlert(detected={self.detected}, "
            f"freq={self.frequency:.1%}, "
            f"recommendation={self.recommendation!r}, "
            f"combo={self.repeated_combo})"
        )


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class CollapseDetector:
    """
    Mode collapse detector for Atwater's optimisation loop.

    Parameters
    ----------
    threshold:
        Fraction of the window that must show the same top-3 params to
        trigger an alert (default 0.25 → >5 of 20 trials).
    top_k:
        Number of parameter keys to use in the "combo" fingerprint.
        Selects the top-k most-frequent keys across the window (default 3).
    min_window:
        Minimum number of trials required before detection is active
        (avoids false positives at startup).
    """

    def __init__(
        self,
        threshold: float = 0.25,
        top_k: int = 3,
        min_window: int = 10,
    ) -> None:
        self.threshold = threshold
        self.top_k = top_k
        self.min_window = min_window

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        recent_trials: list[dict[str, Any]],
        window: int = 20,
    ) -> CollapseAlert | None:
        """
        Check for mode collapse in recent trials.

        Parameters
        ----------
        recent_trials:
            List of trial dicts.  Each dict must have a ``"params"`` key
            mapping parameter names to values.  Extra keys are ignored.
            The last ``window`` trials are used.
        window:
            How many recent trials to examine.

        Returns
        -------
        CollapseAlert | None
            CollapseAlert if collapse detected, None otherwise.
        """
        if not recent_trials:
            return None

        # Use only the most recent ``window`` trials
        trials = recent_trials[-window:]

        if len(trials) < self.min_window:
            logger.debug(
                "[CollapseDetector] Only %d trials in window (min=%d) — skipping.",
                len(trials),
                self.min_window,
            )
            return None

        # Extract params
        param_sets: list[dict[str, Any]] = [
            t.get("params", {}) for t in trials
        ]

        # Determine the top-k most common parameter *keys* across the window
        key_freq: dict[str, int] = {}
        for params in param_sets:
            for k in params:
                key_freq[k] = key_freq.get(k, 0) + 1

        if not key_freq:
            return None

        top_keys = sorted(key_freq, key=lambda k: key_freq[k], reverse=True)[
            : self.top_k
        ]

        # Build "combo" fingerprints using only top-k keys
        combos: list[tuple] = []
        for params in param_sets:
            combo = tuple(
                (k, params[k]) for k in top_keys if k in params
            )
            combos.append(combo)

        # Count combo frequencies
        combo_counts: dict[tuple, int] = {}
        for combo in combos:
            combo_counts[combo] = combo_counts.get(combo, 0) + 1

        if not combo_counts:
            return None

        # Find the most common combo
        dominant_combo, count = max(combo_counts.items(), key=lambda x: x[1])
        frequency = count / len(trials)

        if frequency <= self.threshold:
            logger.debug(
                "[CollapseDetector] No collapse: dominant combo freq=%.2f (threshold=%.2f).",
                frequency,
                self.threshold,
            )
            return None

        # Collapse detected — choose recommendation based on severity
        repeated_combo_dict = dict(dominant_combo)
        recommendation = self._choose_recommendation(frequency, count, len(trials))

        details = (
            f"Dominant combo {repeated_combo_dict} appeared {count}/{len(trials)} times "
            f"({frequency:.1%}) in last {window} trials. "
            f"Keys examined: {top_keys}."
        )

        logger.warning(
            "[CollapseDetector] Collapse detected! %s → recommend: %s",
            details,
            recommendation,
        )

        return CollapseAlert(
            detected=True,
            repeated_combo=repeated_combo_dict,
            count=count,
            window=len(trials),
            frequency=frequency,
            recommendation=recommendation,
            details=details,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _choose_recommendation(
        self, frequency: float, count: int, window_size: int
    ) -> str:
        """
        Pick the most appropriate recommendation based on collapse severity.

        - Mild (25–40%): force_random — inject one random trial to break pattern
        - Moderate (40–60%): increase_temperature — broaden sampling distribution
        - Severe (>60%): reset_sampler — rebuild sampler from scratch
        """
        if frequency > 0.60:
            return "reset_sampler"
        elif frequency > 0.40:
            return "increase_temperature"
        else:
            return "force_random"

    @staticmethod
    def describe_recommendation(recommendation: str) -> str:
        """Return a human-readable description of a recommendation code."""
        descriptions = {
            "force_random": (
                "Inject a fully random trial using RandomSampler to break "
                "the dominant parameter pattern."
            ),
            "increase_temperature": (
                "Increase LLM/sampler temperature to widen the distribution "
                "of suggested parameter values."
            ),
            "reset_sampler": (
                "Reset the Optuna sampler to a fresh TPE instance to escape "
                "the collapsed region of parameter space."
            ),
        }
        return descriptions.get(
            recommendation,
            f"Unknown recommendation: {recommendation!r}",
        )
