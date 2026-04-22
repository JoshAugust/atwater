"""
Tests for src.evaluation.process_rewards (ProcessRewardScorer).

Tests cover:
  - per-step scoring with mock LLM client
  - heuristic fallback (no LLM client)
  - aggregate weighted scoring
  - score_all_steps convenience method
  - empty/invalid output handling
  - unknown step names
  - LLM failure fallback to heuristic
  - step weights validation

All LLM calls are mocked — tests run without LM Studio or models.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.evaluation import StepScore
from src.evaluation.process_rewards import (
    ProcessRewardScorer,
    STEP_NAMES,
    _DEFAULT_STEP_WEIGHTS,
)


# ---------------------------------------------------------------------------
# Helpers: mock LLM client
# ---------------------------------------------------------------------------


def make_mock_client(score: float = 0.8, reasoning: str = "Good.", issues: list | None = None):
    """Return a mock LMStudioClient that returns a fixed step evaluation."""
    client = MagicMock()
    client.chat_structured.return_value = {
        "score": score,
        "reasoning": reasoning,
        "issues": issues or [],
    }
    return client


def make_failing_client():
    """Return a mock client that always raises an exception."""
    client = MagicMock()
    client.chat_structured.side_effect = Exception("LLM connection failed")
    return client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client():
    return make_mock_client(score=0.75)


@pytest.fixture
def scorer_with_client(mock_client):
    return ProcessRewardScorer(client=mock_client)


@pytest.fixture
def scorer_no_client():
    """Scorer without LLM client — uses heuristic fallback."""
    return ProcessRewardScorer(client=None)


# ---------------------------------------------------------------------------
# Tests: per-step scoring with LLM client
# ---------------------------------------------------------------------------


class TestStepScoringWithLLM:
    def test_score_brief_step(self, scorer_with_client, mock_client):
        result = scorer_with_client.score_step(
            "brief",
            "Target: Gen-Z coffee drinkers. Goal: drive app downloads. Tone: energetic.",
        )
        assert isinstance(result, StepScore)
        assert result.step_name == "brief"
        assert result.score == pytest.approx(0.75, abs=1e-4)

    def test_score_concept_step(self, scorer_with_client):
        result = scorer_with_client.score_step(
            "concept",
            "A minimalist ad showing a single coffee bean transforming into a city skyline.",
        )
        assert result.step_name == "concept"
        assert 0.0 <= result.score <= 1.0

    def test_score_execution_step(self, scorer_with_client):
        result = scorer_with_client.score_step(
            "execution",
            "Dark background, neon green accent, bold sans-serif header, clean CTA.",
        )
        assert result.step_name == "execution"
        assert 0.0 <= result.score <= 1.0

    def test_score_polish_step(self, scorer_with_client):
        result = scorer_with_client.score_step(
            "polish",
            "Typography adjusted for WCAG AA compliance. Spacing tightened.",
        )
        assert result.step_name == "polish"
        assert 0.0 <= result.score <= 1.0

    def test_step_score_has_required_fields(self, scorer_with_client):
        result = scorer_with_client.score_step("brief", "Good brief content here.")
        assert hasattr(result, "step_name")
        assert hasattr(result, "score")
        assert hasattr(result, "reasoning")
        assert hasattr(result, "issues")

    def test_score_clamped_to_0_1(self):
        client = make_mock_client(score=1.5)  # LLM returned > 1.0
        scorer = ProcessRewardScorer(client=client)
        result = scorer.score_step("brief", "Some brief content.")
        assert result.score <= 1.0

    def test_context_brief_included_in_prompt(self, mock_client):
        scorer = ProcessRewardScorer(client=mock_client)
        scorer.score_step(
            "concept",
            "Bold coffee concept.",
            context={"brief": "Sell more coffee via app"},
        )
        call_args = mock_client.chat_structured.call_args
        messages = call_args[1]["messages"]
        # Brief should appear somewhere in the messages
        all_content = " ".join(m["content"] for m in messages)
        assert "Sell more coffee via app" in all_content

    def test_issues_populated_from_llm(self):
        client = make_mock_client(score=0.4, issues=["Concept is too generic", "Missing CTA"])
        scorer = ProcessRewardScorer(client=client)
        result = scorer.score_step("concept", "Coffee is great.")
        assert "Concept is too generic" in result.issues

    def test_llm_schema_passed_to_client(self, mock_client):
        scorer = ProcessRewardScorer(client=mock_client)
        scorer.score_step("brief", "Some content.")
        call_kwargs = mock_client.chat_structured.call_args[1]
        assert "schema" in call_kwargs


# ---------------------------------------------------------------------------
# Tests: heuristic fallback (no LLM client)
# ---------------------------------------------------------------------------


class TestHeuristicScoring:
    def test_empty_output_scores_zero(self, scorer_no_client):
        result = scorer_no_client.score_step("brief", "")
        assert result.score == 0.0
        assert "Empty" in result.issues[0]

    def test_whitespace_only_scores_zero(self, scorer_no_client):
        result = scorer_no_client.score_step("concept", "   \n  ")
        assert result.score == 0.0

    def test_very_short_output_scores_low(self, scorer_no_client):
        result = scorer_no_client.score_step("brief", "ok")
        assert result.score <= 0.3

    def test_short_output_scores_moderate(self, scorer_no_client):
        # 4 words → "very short" tier (0.2); need 5-19 words for 0.4
        result = scorer_no_client.score_step(
            "concept", "Coffee for young people who use apps."  # 8 words → 0.4
        )
        assert 0.3 <= result.score <= 0.5

    def test_decent_output_scores_neutral(self, scorer_no_client):
        result = scorer_no_client.score_step(
            "execution",
            "Bold minimalist design with dark background, white text, and green accent colour "
            "following brand guidelines. Typography at 14pt body, 48pt headline.",
        )
        assert result.score == pytest.approx(0.5, abs=0.01)

    def test_reasoning_present(self, scorer_no_client):
        result = scorer_no_client.score_step("polish", "Final adjustments done.")
        assert len(result.reasoning) > 0

    def test_all_steps_heuristic(self, scorer_no_client):
        for step in STEP_NAMES:
            result = scorer_no_client.score_step(step, "Some output content here for testing.")
            assert isinstance(result, StepScore)
            assert result.step_name == step


# ---------------------------------------------------------------------------
# Tests: LLM failure fallback
# ---------------------------------------------------------------------------


class TestLLMFailureFallback:
    def test_llm_failure_falls_back_to_heuristic(self):
        """When LLM raises, scorer should fall back to heuristic, not crash."""
        client = make_failing_client()
        scorer = ProcessRewardScorer(client=client)
        # 20+ words → heuristic neutral score of 0.5
        result = scorer.score_step(
            "brief",
            "This is a long enough brief with good content for heuristic scoring "
            "and it has more than twenty words total.",
        )
        assert isinstance(result, StepScore)
        assert result.score == pytest.approx(0.5, abs=0.01)  # heuristic neutral

    def test_llm_failure_step_name_preserved(self):
        client = make_failing_client()
        scorer = ProcessRewardScorer(client=client)
        result = scorer.score_step("concept", "Some concept text here for this step.")
        assert result.step_name == "concept"


# ---------------------------------------------------------------------------
# Tests: invalid step names
# ---------------------------------------------------------------------------


class TestInvalidStepName:
    def test_unknown_step_raises(self, scorer_no_client):
        with pytest.raises(ValueError, match="Unknown step"):
            scorer_no_client.score_step("design", "Some output.")

    def test_empty_step_name_raises(self, scorer_no_client):
        with pytest.raises(ValueError):
            scorer_no_client.score_step("", "Some output.")


# ---------------------------------------------------------------------------
# Tests: aggregate scoring
# ---------------------------------------------------------------------------


class TestAggregateScoring:
    def test_aggregate_single_step(self, scorer_no_client):
        scores = [StepScore("brief", 0.8, "Good", [])]
        agg = scorer_no_client.aggregate(scores)
        # brief weight = 0.20; normalised by total weight (0.20) = 0.8
        assert agg == pytest.approx(0.8, abs=0.01)

    def test_aggregate_all_steps(self, scorer_no_client):
        scores = [
            StepScore("brief", 1.0, "", []),
            StepScore("concept", 1.0, "", []),
            StepScore("execution", 1.0, "", []),
            StepScore("polish", 1.0, "", []),
        ]
        agg = scorer_no_client.aggregate(scores)
        assert agg == pytest.approx(1.0, abs=1e-4)

    def test_aggregate_zero_scores(self, scorer_no_client):
        scores = [
            StepScore("brief", 0.0, "", []),
            StepScore("concept", 0.0, "", []),
        ]
        agg = scorer_no_client.aggregate(scores)
        assert agg == pytest.approx(0.0, abs=1e-4)

    def test_aggregate_empty_list(self, scorer_no_client):
        agg = scorer_no_client.aggregate([])
        assert agg == 0.0

    def test_aggregate_weighted(self, scorer_no_client):
        """concept weight 0.30, execution weight 0.35 — execution should matter more."""
        scores = [
            StepScore("concept", 1.0, "", []),     # weight 0.30
            StepScore("execution", 0.0, "", []),   # weight 0.35
        ]
        agg = scorer_no_client.aggregate(scores)
        # 1.0 * 0.30 + 0.0 * 0.35 = 0.30; total weight = 0.65; agg = 0.30/0.65 ≈ 0.46
        assert agg == pytest.approx(0.30 / 0.65, abs=0.01)

    def test_aggregate_clamped_to_0_1(self):
        """Custom weights with inflated scores should still clamp."""
        scorer = ProcessRewardScorer()  # no client
        # Override weights to something that might cause > 1 if not clamped
        scorer._weights = {"brief": 0.5, "concept": 0.5, "execution": 0.0, "polish": 0.0}
        scores = [
            StepScore("brief", 1.0, "", []),
            StepScore("concept", 1.0, "", []),
        ]
        agg = scorer.aggregate(scores)
        assert 0.0 <= agg <= 1.0


# ---------------------------------------------------------------------------
# Tests: score_all_steps
# ---------------------------------------------------------------------------


class TestScoreAllSteps:
    def test_score_all_steps_returns_list_and_aggregate(self):
        client = make_mock_client(score=0.7)
        scorer = ProcessRewardScorer(client=client)

        step_outputs = {
            "brief": "Target audience: Millennials. Goal: brand awareness.",
            "concept": "Coffee meets minimalism — single bean, massive impact.",
            "execution": "Dark BG, white serif font, green accent, 1080x1080.",
            "polish": "Adjusted contrast ratio to 5.1:1 for WCAG AA compliance.",
        }

        step_scores, aggregate = scorer.score_all_steps(step_outputs)

        assert len(step_scores) == 4
        assert all(isinstance(s, StepScore) for s in step_scores)
        assert 0.0 <= aggregate <= 1.0

    def test_partial_steps_only_scores_provided(self):
        client = make_mock_client(score=0.6)
        scorer = ProcessRewardScorer(client=client)

        step_outputs = {"brief": "Good brief.", "concept": "Strong concept."}
        step_scores, aggregate = scorer.score_all_steps(step_outputs)

        assert len(step_scores) == 2
        step_names = [s.step_name for s in step_scores]
        assert "brief" in step_names
        assert "concept" in step_names
        assert "execution" not in step_names

    def test_previous_steps_passed_as_context(self):
        client = make_mock_client(score=0.7)
        scorer = ProcessRewardScorer(client=client)

        step_outputs = {
            "brief": "Great brief.",
            "concept": "Bold concept.",
        }
        scorer.score_all_steps(step_outputs)

        # On the second step (concept), previous_steps should include brief
        calls = client.chat_structured.call_args_list
        assert len(calls) == 2  # one call per step

        # Inspect the second call's messages for brief content
        concept_call_messages = calls[1][1]["messages"]
        all_content = " ".join(m["content"] for m in concept_call_messages)
        assert "Great brief" in all_content  # brief should appear in concept context

    def test_empty_step_outputs_returns_zero(self, scorer_no_client):
        step_scores, aggregate = scorer_no_client.score_all_steps({})
        assert step_scores == []
        assert aggregate == 0.0


# ---------------------------------------------------------------------------
# Tests: configuration validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            ProcessRewardScorer(step_weights={"brief": 0.5, "concept": 0.5, "execution": 0.5, "polish": 0.5})

    def test_unknown_step_in_weights_raises(self):
        with pytest.raises(ValueError, match="Unknown step names"):
            ProcessRewardScorer(step_weights={
                "brief": 0.25, "concept": 0.25, "execution": 0.25, "mystery_step": 0.25
            })

    def test_custom_weights_accepted(self):
        """Custom weights summing to 1.0 with valid step names should work."""
        scorer = ProcessRewardScorer(step_weights={
            "brief": 0.10,
            "concept": 0.40,
            "execution": 0.40,
            "polish": 0.10,
        })
        assert scorer._weights["concept"] == pytest.approx(0.40, abs=1e-6)

    def test_default_weights_sum_to_one(self):
        assert abs(sum(_DEFAULT_STEP_WEIGHTS.values()) - 1.0) < 1e-6

    def test_all_default_steps_in_step_names(self):
        for step in _DEFAULT_STEP_WEIGHTS:
            assert step in STEP_NAMES
