"""
Tests for src.evaluation.cascade (VerifierCascade).

Tests cover:
  - bad output short-circuits at fast gate (score=0)
  - medium-scoring output short-circuits at medium gate
  - good output goes through all gates (no short-circuit)
  - cascade stats tracking
  - stats_summary string
  - reset_stats
  - configurable thresholds

All ML models are mocked — tests run without pyiqa or CLIP installed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.evaluation import (
    CascadeResult,
    CreativeOutput,
    GateResult,
    GradeResult,
)
from src.evaluation.cascade import CascadeThresholds, VerifierCascade


# ---------------------------------------------------------------------------
# Helpers: build mock gates
# ---------------------------------------------------------------------------


def make_fast_gate(passed: bool, score: float, failures: list[str] | None = None) -> MagicMock:
    gate = MagicMock()
    result = GateResult(
        passed=passed,
        score=score,
        failures=failures or [],
        time_ms=0.5,
    )
    gate.run_all.return_value = result
    return gate


def make_medium_gate(passed: bool, score: float, failures: list[str] | None = None) -> MagicMock:
    gate = MagicMock()
    result = GateResult(
        passed=passed,
        score=score,
        failures=failures or [],
        time_ms=50.0,
    )
    gate.run_all.return_value = result
    return gate


def make_llm_gate(overall_score: float) -> MagicMock:
    gate = MagicMock()
    result = GradeResult(
        overall_score=overall_score,
        dimension_scores={"originality": 0.8, "brand_alignment": 0.9, "technical_quality": 0.85},
        reasoning={"originality": "Good.", "brand_alignment": "Strong.", "technical_quality": "Clean."},
        novel_finding="Interesting use of negative space.",
        suggestions=["Consider bolder typography."],
    )
    gate.evaluate.return_value = result
    return gate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def good_output() -> CreativeOutput:
    return CreativeOutput(
        output_path="/tmp/good.png",
        format="PNG",
        width=1080,
        height=1080,
        color_space="RGB",
        typography={"contrast_ratio": 7.0, "min_font_size": 12},
        text_description="A great ad.",
    )


@pytest.fixture
def bad_output() -> CreativeOutput:
    """Output that fails fast gate."""
    return CreativeOutput(
        output_path="/tmp/bad.tiff",
        format="TIFF",
        width=10,
        height=10,
        color_space="CMYK",
        typography={"contrast_ratio": 1.5},
    )


# ---------------------------------------------------------------------------
# Test: short-circuit at fast gate
# ---------------------------------------------------------------------------


class TestCascadeShortCircuitAtFast:
    def test_bad_output_stops_at_fast_gate(self, bad_output):
        """A failing fast gate should return score=0.0 immediately."""
        fast = make_fast_gate(passed=False, score=0.0, failures=["Bad format", "Too small"])
        medium = make_medium_gate(passed=True, score=0.9)
        llm = make_llm_gate(overall_score=0.8)

        cascade = VerifierCascade(fast, medium, llm)
        result = cascade.evaluate(bad_output)

        assert isinstance(result, CascadeResult)
        assert result.final_score == 0.0
        assert result.short_circuited is True
        assert "fast" in result.gates_failed
        assert "medium" not in result.gates_passed
        assert "llm" not in result.gates_passed

    def test_medium_gate_not_called_after_fast_failure(self, bad_output):
        fast = make_fast_gate(passed=False, score=0.0)
        medium = make_medium_gate(passed=True, score=0.9)
        llm = make_llm_gate(overall_score=0.8)

        cascade = VerifierCascade(fast, medium, llm)
        cascade.evaluate(bad_output)

        medium.run_all.assert_not_called()
        llm.evaluate.assert_not_called()

    def test_llm_gate_not_called_after_fast_failure(self, bad_output):
        fast = make_fast_gate(passed=False, score=0.0)
        medium = make_medium_gate(passed=True, score=0.9)
        llm = make_llm_gate(overall_score=0.9)

        cascade = VerifierCascade(fast, medium, llm)
        cascade.evaluate(bad_output)

        llm.evaluate.assert_not_called()

    def test_gate_scores_populated_for_fast_only(self, bad_output):
        fast = make_fast_gate(passed=False, score=0.1)
        medium = make_medium_gate(passed=True, score=0.9)
        llm = make_llm_gate(overall_score=0.8)

        cascade = VerifierCascade(fast, medium, llm)
        result = cascade.evaluate(bad_output)

        assert "fast" in result.gate_scores
        assert "medium" not in result.gate_scores
        assert "llm" not in result.gate_scores

    def test_short_circuit_fast_gate_details(self, bad_output):
        fast = make_fast_gate(passed=False, score=0.0, failures=["Format check failed"])
        medium = make_medium_gate(passed=True, score=0.9)
        llm = make_llm_gate(overall_score=0.8)

        cascade = VerifierCascade(fast, medium, llm)
        result = cascade.evaluate(bad_output)

        assert "fast" in result.gate_details
        assert result.gate_details["fast"].failures == ["Format check failed"]


# ---------------------------------------------------------------------------
# Test: short-circuit at medium gate
# ---------------------------------------------------------------------------


class TestCascadeShortCircuitAtMedium:
    def test_medium_failure_uses_medium_score(self, good_output):
        """Medium gate failure should return medium_score as final score."""
        fast = make_fast_gate(passed=True, score=1.0)
        medium = make_medium_gate(passed=False, score=0.25, failures=["Low aesthetic score"])
        llm = make_llm_gate(overall_score=0.9)

        thresholds = CascadeThresholds(medium_min=0.40)
        cascade = VerifierCascade(fast, medium, llm, thresholds)
        result = cascade.evaluate(good_output)

        assert result.short_circuited is True
        assert result.final_score == pytest.approx(0.25, abs=1e-4)
        assert "fast" in result.gates_passed
        assert "medium" in result.gates_failed
        assert "llm" not in result.gates_passed

    def test_llm_not_called_after_medium_failure(self, good_output):
        fast = make_fast_gate(passed=True, score=1.0)
        medium = make_medium_gate(passed=False, score=0.2)
        llm = make_llm_gate(overall_score=0.9)

        thresholds = CascadeThresholds(medium_min=0.40)
        cascade = VerifierCascade(fast, medium, llm, thresholds)
        cascade.evaluate(good_output)

        llm.evaluate.assert_not_called()

    def test_medium_below_threshold_short_circuits(self, good_output):
        """Medium gate passes (no failures) but score is below threshold → short-circuit."""
        fast = make_fast_gate(passed=True, score=1.0)
        medium = make_medium_gate(passed=True, score=0.35)  # passed=True but score < 0.40
        llm = make_llm_gate(overall_score=0.8)

        thresholds = CascadeThresholds(medium_min=0.40)
        cascade = VerifierCascade(fast, medium, llm, thresholds)
        result = cascade.evaluate(good_output)

        assert result.short_circuited is True
        assert result.final_score == pytest.approx(0.35, abs=1e-4)


# ---------------------------------------------------------------------------
# Test: full pass (no short-circuit)
# ---------------------------------------------------------------------------


class TestCascadeFullPass:
    def test_good_output_reaches_llm_gate(self, good_output):
        fast = make_fast_gate(passed=True, score=1.0)
        medium = make_medium_gate(passed=True, score=0.85)
        llm = make_llm_gate(overall_score=0.87)

        cascade = VerifierCascade(fast, medium, llm)
        result = cascade.evaluate(good_output)

        assert result.short_circuited is False
        assert result.final_score == pytest.approx(0.87, abs=1e-4)

    def test_all_three_gates_in_passed_list(self, good_output):
        fast = make_fast_gate(passed=True, score=1.0)
        medium = make_medium_gate(passed=True, score=0.8)
        llm = make_llm_gate(overall_score=0.9)

        cascade = VerifierCascade(fast, medium, llm)
        result = cascade.evaluate(good_output)

        assert "fast" in result.gates_passed
        assert "medium" in result.gates_passed
        assert "llm" in result.gates_passed
        assert result.gates_failed == []

    def test_all_gate_scores_present_in_full_pass(self, good_output):
        fast = make_fast_gate(passed=True, score=0.95)
        medium = make_medium_gate(passed=True, score=0.80)
        llm = make_llm_gate(overall_score=0.85)

        cascade = VerifierCascade(fast, medium, llm)
        result = cascade.evaluate(good_output)

        assert "fast" in result.gate_scores
        assert "medium" in result.gate_scores
        assert "llm" in result.gate_scores

    def test_llm_result_in_gate_details(self, good_output):
        fast = make_fast_gate(passed=True, score=1.0)
        medium = make_medium_gate(passed=True, score=0.8)
        llm = make_llm_gate(overall_score=0.9)

        cascade = VerifierCascade(fast, medium, llm)
        result = cascade.evaluate(good_output)

        assert isinstance(result.gate_details["llm"], GradeResult)
        assert result.gate_details["llm"].overall_score == pytest.approx(0.9, abs=1e-4)

    def test_llm_gate_called_with_context(self, good_output):
        """LLM gate should receive rubric and knowledge context from context dict."""
        fast = make_fast_gate(passed=True, score=1.0)
        medium = make_medium_gate(passed=True, score=0.8)
        llm = make_llm_gate(overall_score=0.8)

        cascade = VerifierCascade(fast, medium, llm)
        ctx = {
            "rubric": {"originality": "Be bold"},
            "knowledge_context": [{"content": "Dark backgrounds work well"}],
        }
        cascade.evaluate(good_output, context=ctx)

        llm.evaluate.assert_called_once()
        call_kwargs = llm.evaluate.call_args
        assert call_kwargs[1]["rubric"] == ctx["rubric"]
        assert call_kwargs[1]["knowledge_context"] == ctx["knowledge_context"]


# ---------------------------------------------------------------------------
# Test: cascade stats tracking
# ---------------------------------------------------------------------------


class TestCascadeStats:
    def test_stats_start_at_zero(self):
        fast = make_fast_gate(passed=True, score=1.0)
        medium = make_medium_gate(passed=True, score=0.8)
        llm = make_llm_gate(overall_score=0.8)

        cascade = VerifierCascade(fast, medium, llm)
        s = cascade.stats

        assert s["total_evaluated"] == 0
        assert s["fast_passed"] == 0
        assert s["fast_failed"] == 0
        assert s["llm_reached"] == 0

    def test_fast_fail_increments_stats(self, bad_output):
        fast = make_fast_gate(passed=False, score=0.0)
        medium = make_medium_gate(passed=True, score=0.9)
        llm = make_llm_gate(overall_score=0.8)

        cascade = VerifierCascade(fast, medium, llm)
        cascade.evaluate(bad_output)

        s = cascade.stats
        assert s["total_evaluated"] == 1
        assert s["fast_failed"] == 1
        assert s["fast_passed"] == 0
        assert s["short_circuited_at_fast"] == 1
        assert s["llm_reached"] == 0

    def test_medium_fail_increments_stats(self, good_output):
        fast = make_fast_gate(passed=True, score=1.0)
        medium = make_medium_gate(passed=False, score=0.2)
        llm = make_llm_gate(overall_score=0.9)

        thresholds = CascadeThresholds(medium_min=0.40)
        cascade = VerifierCascade(fast, medium, llm, thresholds)
        cascade.evaluate(good_output)

        s = cascade.stats
        assert s["fast_passed"] == 1
        assert s["medium_failed"] == 1
        assert s["short_circuited_at_medium"] == 1
        assert s["llm_reached"] == 0

    def test_full_pass_increments_stats(self, good_output):
        fast = make_fast_gate(passed=True, score=1.0)
        medium = make_medium_gate(passed=True, score=0.8)
        llm = make_llm_gate(overall_score=0.9)

        cascade = VerifierCascade(fast, medium, llm)
        cascade.evaluate(good_output)

        s = cascade.stats
        assert s["fast_passed"] == 1
        assert s["medium_passed"] == 1
        assert s["llm_reached"] == 1
        assert s["short_circuited_at_fast"] == 0
        assert s["short_circuited_at_medium"] == 0

    def test_short_circuit_rate_calculation(self, good_output, bad_output):
        """2 fast failures + 1 medium failure out of 4 total = 75% short-circuit rate."""
        fast_fail = make_fast_gate(passed=False, score=0.0)
        fast_pass = make_fast_gate(passed=True, score=1.0)
        medium_fail = make_medium_gate(passed=False, score=0.2)
        medium_pass = make_medium_gate(passed=True, score=0.8)
        llm = make_llm_gate(overall_score=0.8)

        # We'll simulate multiple evaluations via separate cascades
        cascade = VerifierCascade(fast_pass, medium_pass, llm)

        # 2 fast failures
        for _ in range(2):
            cascade._fast = make_fast_gate(passed=False, score=0.0)
            cascade.evaluate(bad_output)

        # 1 medium failure
        cascade._fast = make_fast_gate(passed=True, score=1.0)
        cascade._medium = make_medium_gate(passed=False, score=0.2)
        thresholds = CascadeThresholds(medium_min=0.40)
        cascade._thresholds = thresholds
        cascade.evaluate(good_output)

        # 1 full pass
        cascade._fast = make_fast_gate(passed=True, score=1.0)
        cascade._medium = make_medium_gate(passed=True, score=0.8)
        cascade._thresholds = CascadeThresholds()
        cascade.evaluate(good_output)

        s = cascade.stats
        assert s["total_evaluated"] == 4
        assert s["short_circuit_rate"] == pytest.approx(0.75, abs=0.01)

    def test_reset_stats(self, good_output):
        fast = make_fast_gate(passed=True, score=1.0)
        medium = make_medium_gate(passed=True, score=0.8)
        llm = make_llm_gate(overall_score=0.9)

        cascade = VerifierCascade(fast, medium, llm)
        cascade.evaluate(good_output)
        cascade.evaluate(good_output)

        assert cascade.stats["total_evaluated"] == 2

        cascade.reset_stats()

        s = cascade.stats
        assert s["total_evaluated"] == 0
        assert s["fast_passed"] == 0
        assert s["llm_reached"] == 0

    def test_stats_summary_string(self, good_output, bad_output):
        fast = make_fast_gate(passed=True, score=1.0)
        medium = make_medium_gate(passed=True, score=0.8)
        llm = make_llm_gate(overall_score=0.9)

        cascade = VerifierCascade(fast, medium, llm)
        cascade.evaluate(good_output)

        summary = cascade.stats_summary()
        assert "VerifierCascade" in summary
        assert "evaluated" in summary.lower() or "1" in summary

    def test_stats_summary_no_evals(self):
        fast = make_fast_gate(passed=True, score=1.0)
        medium = make_medium_gate(passed=True, score=0.8)
        llm = make_llm_gate(overall_score=0.9)

        cascade = VerifierCascade(fast, medium, llm)
        summary = cascade.stats_summary()
        assert "no evaluations" in summary.lower()

    def test_avg_time_ms_tracked(self, good_output):
        fast = make_fast_gate(passed=True, score=1.0)
        medium = make_medium_gate(passed=True, score=0.8)
        llm = make_llm_gate(overall_score=0.9)

        cascade = VerifierCascade(fast, medium, llm)
        cascade.evaluate(good_output)

        s = cascade.stats
        assert s["avg_time_ms"] >= 0.0
        assert s["total_time_ms"] >= 0.0


# ---------------------------------------------------------------------------
# Test: context passthrough
# ---------------------------------------------------------------------------


class TestCascadeContextPassthrough:
    def test_brand_colors_passed_to_fast_gate(self, good_output):
        fast = make_fast_gate(passed=True, score=1.0)
        medium = make_medium_gate(passed=True, score=0.8)
        llm = make_llm_gate(overall_score=0.9)

        cascade = VerifierCascade(fast, medium, llm)
        brand_colors = [(0, 0, 255), (255, 255, 255)]
        cascade.evaluate(good_output, context={"brand_colors": brand_colors})

        call_kwargs = fast.run_all.call_args
        assert call_kwargs[1]["brand_colors"] == brand_colors

    def test_empty_context_is_safe(self, good_output):
        fast = make_fast_gate(passed=True, score=1.0)
        medium = make_medium_gate(passed=True, score=0.8)
        llm = make_llm_gate(overall_score=0.9)

        cascade = VerifierCascade(fast, medium, llm)
        result = cascade.evaluate(good_output, context={})  # no crash

        assert isinstance(result, CascadeResult)

    def test_none_context_is_safe(self, good_output):
        fast = make_fast_gate(passed=True, score=1.0)
        medium = make_medium_gate(passed=True, score=0.8)
        llm = make_llm_gate(overall_score=0.9)

        cascade = VerifierCascade(fast, medium, llm)
        result = cascade.evaluate(good_output, context=None)

        assert isinstance(result, CascadeResult)
