"""
tests/test_reflexion.py — Tests for src.learning.reflexion.

Tests:
- Reflection dataclass construction and serialisation
- Rule-based reflection generation (no LLM required)
- LLM call with mock client (JSON parsing)
- Sliding window behaviour (max N reflections)
- build_reflection_prompt structure
- build_director_context output
- KB storage integration
- LLM failure fallback to rule-based
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.learning.reflexion import Reflection, ReflexionEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_cycle_result(
    cycle_number: int = 1,
    score: float = 0.75,
    params: dict | None = None,
    outputs: list | None = None,
    errors: list | None = None,
) -> dict:
    return {
        "cycle_number": cycle_number,
        "score": score,
        "params": params or {"font": "Inter", "bg": "#000", "size": 32},
        "outputs": outputs or ["image_001.png"],
        "errors": errors or [],
    }


def make_mock_llm(json_response: str) -> MagicMock:
    """Return a mock LLM client whose chat() method returns the given JSON string."""
    mock = MagicMock()
    mock.chat.return_value = json_response
    return mock


# ---------------------------------------------------------------------------
# Reflection dataclass
# ---------------------------------------------------------------------------

class TestReflectionDataclass:
    def test_defaults(self):
        r = Reflection(cycle_number=1)
        assert r.what_worked == []
        assert r.what_failed == []
        assert r.hypotheses == []
        assert r.next_actions == []
        assert r.score is None
        assert r.knowledge_entry_id is None

    def test_to_dict(self):
        r = Reflection(
            cycle_number=3,
            score=0.9,
            what_worked=["good params"],
            what_failed=["bad layout"],
            hypotheses=["try bolder font"],
            next_actions=["increase size"],
        )
        d = r.to_dict()
        assert d["cycle_number"] == 3
        assert d["score"] == 0.9
        assert d["what_worked"] == ["good params"]
        assert d["what_failed"] == ["bad layout"]
        assert d["hypotheses"] == ["try bolder font"]
        assert d["next_actions"] == ["increase size"]

    def test_to_context_string(self):
        r = Reflection(
            cycle_number=5,
            score=0.6,
            what_worked=["dark bg"],
            what_failed=["serif font"],
            hypotheses=["try sans-serif"],
            next_actions=["swap font"],
        )
        ctx = r.to_context_string()
        assert "Cycle 5" in ctx
        assert "dark bg" in ctx
        assert "serif font" in ctx
        assert "try sans-serif" in ctx
        assert "swap font" in ctx

    def test_to_context_string_empty(self):
        r = Reflection(cycle_number=1, score=0.5)
        ctx = r.to_context_string()
        assert "Cycle 1" in ctx
        # No crashes on empty lists
        assert "✓" not in ctx or "Worked" not in ctx


# ---------------------------------------------------------------------------
# ReflexionEngine — rule-based (no LLM)
# ---------------------------------------------------------------------------

class TestReflexionEngineRuleBased:
    def setup_method(self):
        self.engine = ReflexionEngine(knowledge_base=None, llm_client=None, window_size=5)

    def test_generate_reflection_high_score(self):
        result = make_cycle_result(score=0.85)
        reflection = self.engine.generate_reflection(result)
        assert reflection.cycle_number == 1
        assert reflection.score == 0.85
        # High score → worked section populated
        worked_text = " ".join(reflection.what_worked)
        assert "0.85" in worked_text or "High score" in worked_text

    def test_generate_reflection_low_score(self):
        result = make_cycle_result(cycle_number=2, score=0.3)
        reflection = self.engine.generate_reflection(result)
        assert reflection.score == 0.3
        failed_text = " ".join(reflection.what_failed)
        assert "0.3" in failed_text or "Low score" in failed_text or "underperform" in failed_text

    def test_generate_reflection_with_errors(self):
        result = make_cycle_result(
            cycle_number=3,
            score=0.5,
            errors=["timeout during render", "missing asset"]
        )
        reflection = self.engine.generate_reflection(result)
        all_failed = " ".join(reflection.what_failed)
        assert "timeout" in all_failed or "Error" in all_failed

    def test_generate_reflection_moderate_score(self):
        result = make_cycle_result(cycle_number=4, score=0.65)
        reflection = self.engine.generate_reflection(result)
        # Should have some hypotheses/next actions
        assert reflection.hypotheses or reflection.next_actions

    def test_reflection_stored_in_window(self):
        for i in range(3):
            self.engine.generate_reflection(make_cycle_result(cycle_number=i))
        assert len(self.engine.get_window()) == 3

    def test_window_max_size(self):
        """Window should never exceed window_size."""
        engine = ReflexionEngine(window_size=3)
        for i in range(7):
            engine.generate_reflection(make_cycle_result(cycle_number=i))
        assert len(engine.get_window()) == 3

    def test_window_is_sliding(self):
        """Oldest reflections should be evicted first."""
        engine = ReflexionEngine(window_size=3)
        for i in range(5):
            engine.generate_reflection(make_cycle_result(cycle_number=i))
        window = engine.get_window()
        cycle_numbers = [r.cycle_number for r in window]
        assert cycle_numbers == [2, 3, 4]

    def test_clear_window(self):
        engine = ReflexionEngine(window_size=5)
        for i in range(3):
            engine.generate_reflection(make_cycle_result(cycle_number=i))
        engine.clear_window()
        assert len(engine.get_window()) == 0


# ---------------------------------------------------------------------------
# ReflexionEngine — LLM mock
# ---------------------------------------------------------------------------

class TestReflexionEngineLLM:
    VALID_JSON = """{
        "what_worked": ["dark background matched brand"],
        "what_failed": ["hero font too small"],
        "hypotheses": ["larger font size may improve CTR"],
        "next_actions": ["increase hero_font_size to 48px"]
    }"""

    def test_llm_response_parsed(self):
        mock_llm = make_mock_llm(self.VALID_JSON)
        engine = ReflexionEngine(llm_client=mock_llm, window_size=5)
        result = make_cycle_result(score=0.72)
        reflection = engine.generate_reflection(result)
        assert reflection.what_worked == ["dark background matched brand"]
        assert reflection.what_failed == ["hero font too small"]
        assert "larger font size" in reflection.hypotheses[0]
        assert "48px" in reflection.next_actions[0]

    def test_llm_response_with_markdown_fences(self):
        fenced = f"```json\n{self.VALID_JSON}\n```"
        mock_llm = make_mock_llm(fenced)
        engine = ReflexionEngine(llm_client=mock_llm, window_size=5)
        reflection = engine.generate_reflection(make_cycle_result(score=0.8))
        assert reflection.what_worked  # parsed successfully

    def test_llm_failure_falls_back_to_rule_based(self):
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = RuntimeError("API timeout")
        engine = ReflexionEngine(llm_client=mock_llm, window_size=5)
        # Should not raise; falls back to rule-based
        reflection = engine.generate_reflection(make_cycle_result(score=0.75))
        assert isinstance(reflection, Reflection)
        assert reflection.cycle_number == 1

    def test_llm_invalid_json_falls_back(self):
        mock_llm = make_mock_llm("this is not json at all")
        engine = ReflexionEngine(llm_client=mock_llm, window_size=5)
        reflection = engine.generate_reflection(make_cycle_result(score=0.5))
        assert isinstance(reflection, Reflection)

    def test_llm_called_once_per_cycle(self):
        mock_llm = make_mock_llm(self.VALID_JSON)
        engine = ReflexionEngine(llm_client=mock_llm, window_size=5)
        engine.generate_reflection(make_cycle_result(score=0.7))
        assert mock_llm.chat.call_count == 1


# ---------------------------------------------------------------------------
# build_reflection_prompt
# ---------------------------------------------------------------------------

class TestBuildReflectionPrompt:
    def setup_method(self):
        self.engine = ReflexionEngine(window_size=5)

    def test_returns_list_of_dicts(self):
        result = make_cycle_result()
        messages = self.engine.build_reflection_prompt(result, [])
        assert isinstance(messages, list)
        assert all(isinstance(m, dict) for m in messages)

    def test_has_system_and_user_roles(self):
        result = make_cycle_result()
        messages = self.engine.build_reflection_prompt(result, [])
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles

    def test_system_prompt_contains_json_instruction(self):
        result = make_cycle_result()
        messages = self.engine.build_reflection_prompt(result, [])
        system_msg = next(m for m in messages if m["role"] == "system")
        assert "JSON" in system_msg["content"] or "json" in system_msg["content"]

    def test_user_message_contains_cycle_data(self):
        result = make_cycle_result(cycle_number=42, score=0.77)
        messages = self.engine.build_reflection_prompt(result, [])
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "42" in user_msg["content"] or "0.77" in user_msg["content"]

    def test_previous_reflections_included(self):
        result = make_cycle_result()
        prev = [
            Reflection(cycle_number=0, score=0.5, what_worked=["dark bg"])
        ]
        messages = self.engine.build_reflection_prompt(result, prev)
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "dark bg" in user_msg["content"] or "Cycle 0" in user_msg["content"]

    def test_empty_previous_reflections(self):
        """No crash when previous_reflections is empty."""
        result = make_cycle_result()
        messages = self.engine.build_reflection_prompt(result, [])
        assert len(messages) >= 2


# ---------------------------------------------------------------------------
# build_director_context
# ---------------------------------------------------------------------------

class TestBuildDirectorContext:
    def test_empty_window_returns_empty_string(self):
        engine = ReflexionEngine(window_size=5)
        assert engine.build_director_context() == ""

    def test_non_empty_window_returns_string(self):
        engine = ReflexionEngine(window_size=5)
        engine.generate_reflection(make_cycle_result(cycle_number=1, score=0.8))
        ctx = engine.build_director_context()
        assert isinstance(ctx, str)
        assert len(ctx) > 0
        assert "REFLEXION" in ctx

    def test_context_includes_all_window_reflections(self):
        engine = ReflexionEngine(window_size=5)
        for i in range(3):
            engine.generate_reflection(make_cycle_result(cycle_number=i, score=0.5 + i * 0.1))
        ctx = engine.build_director_context()
        assert "Cycle 0" in ctx
        assert "Cycle 1" in ctx
        assert "Cycle 2" in ctx


# ---------------------------------------------------------------------------
# KB storage integration
# ---------------------------------------------------------------------------

class TestKBStorage:
    def test_kb_write_called(self):
        mock_kb = MagicMock()
        mock_kb.knowledge_write.return_value = "entry-123"
        engine = ReflexionEngine(knowledge_base=mock_kb, window_size=5)
        reflection = engine.generate_reflection(make_cycle_result())
        mock_kb.knowledge_write.assert_called_once()
        assert reflection.knowledge_entry_id == "entry-123"

    def test_kb_write_uses_observation_tier(self):
        mock_kb = MagicMock()
        mock_kb.knowledge_write.return_value = "entry-456"
        engine = ReflexionEngine(knowledge_base=mock_kb, window_size=5)
        engine.generate_reflection(make_cycle_result())
        call_kwargs = mock_kb.knowledge_write.call_args.kwargs
        assert call_kwargs.get("tier") == "observation"

    def test_kb_failure_does_not_crash(self):
        mock_kb = MagicMock()
        mock_kb.knowledge_write.side_effect = RuntimeError("DB error")
        engine = ReflexionEngine(knowledge_base=mock_kb, window_size=5)
        reflection = engine.generate_reflection(make_cycle_result())
        assert reflection.knowledge_entry_id is None

    def test_no_kb_skips_storage(self):
        engine = ReflexionEngine(knowledge_base=None, window_size=5)
        reflection = engine.generate_reflection(make_cycle_result())
        assert reflection.knowledge_entry_id is None
