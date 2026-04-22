"""
src.schemas.tests.test_schemas — Schema validation unit tests.

Tests the validate_output() function against all agent schemas with:
- Valid happy-path data (should pass)
- Invalid data (should fail with specific errors)

Run with:  python -m pytest src/schemas/tests/test_schemas.py -v
Or directly: python src/schemas/tests/test_schemas.py
"""

from __future__ import annotations

import sys
import os

# Make sure we can import from src/ regardless of working directory.
_here = os.path.dirname(__file__)
_atwater_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
if _atwater_root not in sys.path:
    sys.path.insert(0, _atwater_root)

from src.schemas.validation import validate_output
from src.schemas.agent_schemas import (
    GRADER_LLM_SCHEMA,
    CREATOR_CRITIQUE_LLM_SCHEMA,
    CONSOLIDATOR_SYNTHESIS_LLM_SCHEMA,
    DIRECTOR_OUTPUT_SCHEMA,
    CREATOR_GENERATION_OUTPUT_SCHEMA,
    CREATOR_CRITIQUE_OUTPUT_SCHEMA,
    GRADER_EVALUATION_OUTPUT_SCHEMA,
    GRADER_SCORE_REPORT_OUTPUT_SCHEMA,
    DIVERSITY_GUARD_OUTPUT_SCHEMA,
    CONSOLIDATOR_OUTPUT_SCHEMA,
    CONSOLIDATOR_SKIPPED_OUTPUT_SCHEMA,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_valid(data, schema, label=""):
    valid, errors = validate_output(data, schema)
    if not valid:
        print(f"  FAIL [{label}]: {errors}")
    else:
        print(f"  PASS [{label}]")
    assert valid, f"[{label}] Expected valid, got errors: {errors}"


def assert_invalid(data, schema, expected_error_fragment: str, label=""):
    valid, errors = validate_output(data, schema)
    if valid:
        print(f"  FAIL [{label}]: expected invalid, but got valid")
        assert False, f"[{label}] Expected schema violation, but data passed."
    joined = " | ".join(errors)
    if expected_error_fragment and expected_error_fragment.lower() not in joined.lower():
        print(f"  FAIL [{label}]: error found but wrong message. Got: {joined!r}")
        assert False, f"[{label}] Expected '{expected_error_fragment}' in errors, got: {joined!r}"
    print(f"  PASS [{label}] (invalid as expected: {errors[0]})")


# ===========================================================================
# GRADER_LLM_SCHEMA — 5 valid, 5 invalid
# ===========================================================================

def test_grader_llm_valid():
    print("\n=== GRADER_LLM_SCHEMA (valid) ===")

    # 1. Full perfect response
    assert_valid({
        "originality": {"score": 0.9, "reasoning": "Fresh concept not seen in past 50 trials."},
        "brand_alignment": {"score": 0.85, "reasoning": "Consistent with established brand voice."},
        "technical_quality": {"score": 0.78, "reasoning": "Good composition and layout."},
        "novel_finding": "High-saturation palettes correlate with originality scores above 0.8.",
        "suggest_knowledge_write": True,
        "topic_cluster": "color_palette",
    }, GRADER_LLM_SCHEMA, "grader_full_perfect")

    # 2. Minimal scores, null novel_finding
    assert_valid({
        "originality": {"score": 0.1, "reasoning": "Very repetitive. Seen this pattern many times."},
        "brand_alignment": {"score": 0.2, "reasoning": "Off-brand color usage."},
        "technical_quality": {"score": 0.15, "reasoning": "Poor resolution and blurry text."},
        "novel_finding": None,
        "suggest_knowledge_write": False,
        "topic_cluster": "general",
    }, GRADER_LLM_SCHEMA, "grader_low_scores_null_finding")

    # 3. Exactly 0.5 on all dimensions
    assert_valid({
        "originality": {"score": 0.5, "reasoning": "Average — neither fresh nor repetitive."},
        "brand_alignment": {"score": 0.5, "reasoning": "Partially aligned with guidelines."},
        "technical_quality": {"score": 0.5, "reasoning": "Acceptable execution overall."},
        "novel_finding": "Some novelty in the layout approach but inconclusive.",
        "suggest_knowledge_write": False,
        "topic_cluster": "layout",
    }, GRADER_LLM_SCHEMA, "grader_midpoint_scores")

    # 4. Integer scores (JSON numbers — both int and float are valid)
    assert_valid({
        "originality": {"score": 1, "reasoning": "Completely unique, never tested before."},
        "brand_alignment": {"score": 0, "reasoning": "Completely off-brand in every dimension."},
        "technical_quality": {"score": 1, "reasoning": "Perfect technical execution."},
        "novel_finding": "Score=1 originality with score=0 alignment is a rare combo.",
        "suggest_knowledge_write": True,
        "topic_cluster": "edge_case",
    }, GRADER_LLM_SCHEMA, "grader_integer_boundary_scores")

    # 5. Different topic cluster values
    assert_valid({
        "originality": {"score": 0.72, "reasoning": "Reasonably fresh with minor repetition."},
        "brand_alignment": {"score": 0.91, "reasoning": "Strong brand alignment across all elements."},
        "technical_quality": {"score": 0.63, "reasoning": "Minor quality issues in background layer."},
        "novel_finding": None,
        "suggest_knowledge_write": False,
        "topic_cluster": "brand_voice_typography",
    }, GRADER_LLM_SCHEMA, "grader_realistic_response")


def test_grader_llm_invalid():
    print("\n=== GRADER_LLM_SCHEMA (invalid) ===")

    # 1. Missing required field (brand_alignment)
    assert_invalid({
        "originality": {"score": 0.8, "reasoning": "Good."},
        "technical_quality": {"score": 0.7, "reasoning": "Fine."},
        "novel_finding": None,
        "suggest_knowledge_write": False,
        "topic_cluster": "general",
    }, GRADER_LLM_SCHEMA, "missing required field", "grader_missing_brand_alignment")

    # 2. Score out of range (above 1.0)
    assert_invalid({
        "originality": {"score": 1.5, "reasoning": "Way too high."},
        "brand_alignment": {"score": 0.5, "reasoning": "OK."},
        "technical_quality": {"score": 0.5, "reasoning": "OK."},
        "novel_finding": None,
        "suggest_knowledge_write": False,
        "topic_cluster": "general",
    }, GRADER_LLM_SCHEMA, "maximum", "grader_score_above_1")

    # 3. Score out of range (below 0.0)
    assert_invalid({
        "originality": {"score": -0.1, "reasoning": "Negative?"},
        "brand_alignment": {"score": 0.5, "reasoning": "OK."},
        "technical_quality": {"score": 0.5, "reasoning": "OK."},
        "novel_finding": None,
        "suggest_knowledge_write": False,
        "topic_cluster": "general",
    }, GRADER_LLM_SCHEMA, "minimum", "grader_score_below_0")

    # 4. suggest_knowledge_write is a string instead of bool
    assert_invalid({
        "originality": {"score": 0.8, "reasoning": "Good originality here."},
        "brand_alignment": {"score": 0.7, "reasoning": "Mostly aligned."},
        "technical_quality": {"score": 0.9, "reasoning": "Excellent."},
        "novel_finding": None,
        "suggest_knowledge_write": "yes",
        "topic_cluster": "general",
    }, GRADER_LLM_SCHEMA, "boolean", "grader_suggest_write_wrong_type")

    # 5. Reasoning too short (minLength: 10)
    assert_invalid({
        "originality": {"score": 0.5, "reasoning": "OK"},
        "brand_alignment": {"score": 0.5, "reasoning": "Consistent with brand guidelines."},
        "technical_quality": {"score": 0.5, "reasoning": "Acceptable execution."},
        "novel_finding": None,
        "suggest_knowledge_write": False,
        "topic_cluster": "general",
    }, GRADER_LLM_SCHEMA, "minLength", "grader_reasoning_too_short")


# ===========================================================================
# CREATOR_CRITIQUE_LLM_SCHEMA — 5 valid, 5 invalid
# ===========================================================================

def test_creator_critique_llm_valid():
    print("\n=== CREATOR_CRITIQUE_LLM_SCHEMA (valid) ===")

    # 1. Novel approach, high confidence
    assert_valid({
        "complies_with_patterns": True,
        "compliance_notes": "Content follows all established rules for this format.",
        "is_novel": True,
        "novelty_description": "First time we see this color combination with this layout.",
        "quality_concerns": [],
        "confidence_score": 0.87,
        "overall_critique": "The content is strong and follows patterns while introducing a novel element. Confident it will score well above average.",
    }, CREATOR_CRITIQUE_LLM_SCHEMA, "creator_critique_novel_confident")

    # 2. Non-novel, has quality concerns
    assert_valid({
        "complies_with_patterns": False,
        "compliance_notes": "Violates rule R3: backgrounds must not use pure black.",
        "is_novel": False,
        "novelty_description": None,
        "quality_concerns": ["Background uses pure black (#000000).", "Text contrast ratio below 4.5:1."],
        "confidence_score": 0.25,
        "overall_critique": "Content has significant compliance issues and is unlikely to score well given established grading rubric.",
    }, CREATOR_CRITIQUE_LLM_SCHEMA, "creator_critique_violations")

    # 3. Mixed compliance, moderate confidence
    assert_valid({
        "complies_with_patterns": True,
        "compliance_notes": "Generally compliant with some minor deviations.",
        "is_novel": False,
        "novelty_description": None,
        "quality_concerns": ["Minor font size inconsistency in footer."],
        "confidence_score": 0.61,
        "overall_critique": "Solid content with minor issues. Should score in the 0.6-0.7 range based on prior similar outputs.",
    }, CREATOR_CRITIQUE_LLM_SCHEMA, "creator_critique_moderate")

    # 4. Exact boundary confidence scores
    assert_valid({
        "complies_with_patterns": True,
        "compliance_notes": "Full compliance verified against all known patterns.",
        "is_novel": True,
        "novelty_description": "Unique aspect ratio never tested before.",
        "quality_concerns": [],
        "confidence_score": 1.0,
        "overall_critique": "Perfect compliance and novel approach. Maximum confidence in outcome.",
    }, CREATOR_CRITIQUE_LLM_SCHEMA, "creator_critique_max_confidence")

    # 5. Multiple quality concerns
    assert_valid({
        "complies_with_patterns": False,
        "compliance_notes": "Multiple rule violations detected.",
        "is_novel": False,
        "novelty_description": None,
        "quality_concerns": [
            "Brand color not used in primary CTA.",
            "Headline font is not approved.",
            "Image resolution below minimum 1200px.",
            "Insufficient whitespace in mobile layout.",
        ],
        "confidence_score": 0.0,
        "overall_critique": "Content fails on multiple critical dimensions. Complete rework recommended.",
    }, CREATOR_CRITIQUE_LLM_SCHEMA, "creator_critique_multiple_failures")


def test_creator_critique_llm_invalid():
    print("\n=== CREATOR_CRITIQUE_LLM_SCHEMA (invalid) ===")

    # 1. Missing overall_critique
    assert_invalid({
        "complies_with_patterns": True,
        "compliance_notes": "All good.",
        "is_novel": False,
        "novelty_description": None,
        "quality_concerns": [],
        "confidence_score": 0.7,
    }, CREATOR_CRITIQUE_LLM_SCHEMA, "missing required", "creator_critique_missing_field")

    # 2. confidence_score out of range
    assert_invalid({
        "complies_with_patterns": True,
        "compliance_notes": "All patterns followed correctly.",
        "is_novel": False,
        "novelty_description": None,
        "quality_concerns": [],
        "confidence_score": 1.5,
        "overall_critique": "Good content overall and well executed.",
    }, CREATOR_CRITIQUE_LLM_SCHEMA, "maximum", "creator_critique_confidence_too_high")

    # 3. quality_concerns is a string, not array
    assert_invalid({
        "complies_with_patterns": False,
        "compliance_notes": "Several issues found.",
        "is_novel": False,
        "novelty_description": None,
        "quality_concerns": "has issues",
        "confidence_score": 0.3,
        "overall_critique": "Content has issues that need to be addressed before publication.",
    }, CREATOR_CRITIQUE_LLM_SCHEMA, "array", "creator_critique_concerns_wrong_type")

    # 4. complies_with_patterns is a string
    assert_invalid({
        "complies_with_patterns": "yes",
        "compliance_notes": "All patterns followed correctly.",
        "is_novel": False,
        "novelty_description": None,
        "quality_concerns": [],
        "confidence_score": 0.7,
        "overall_critique": "Content is solid and meets all requirements.",
    }, CREATOR_CRITIQUE_LLM_SCHEMA, "boolean", "creator_critique_bool_as_string")

    # 5. overall_critique too short
    assert_invalid({
        "complies_with_patterns": True,
        "compliance_notes": "All fine.",
        "is_novel": True,
        "novelty_description": "Novel layout.",
        "quality_concerns": [],
        "confidence_score": 0.8,
        "overall_critique": "Good.",
    }, CREATOR_CRITIQUE_LLM_SCHEMA, "minLength", "creator_critique_summary_too_short")


# ===========================================================================
# DIRECTOR_OUTPUT_SCHEMA — 5 valid, 5 invalid
# ===========================================================================

def test_director_output_valid():
    print("\n=== DIRECTOR_OUTPUT_SCHEMA (valid) ===")

    # 1. Optuna source, no override entry
    assert_valid({
        "trial_number": 42,
        "params": {"aspect_ratio": "16:9", "color_palette": "warm", "font_size": 14},
        "source": "optuna",
        "override_entry": None,
    }, DIRECTOR_OUTPUT_SCHEMA, "director_optuna")

    # 2. Knowledge override with entry
    assert_valid({
        "trial_number": 100,
        "params": {"aspect_ratio": "4:3", "style": "minimalist"},
        "source": "knowledge_override",
        "override_entry": {
            "id": "kb-001",
            "content": "Minimalist style correlates with brand alignment > 0.8",
            "tier": "pattern",
            "confidence": 0.85,
        },
    }, DIRECTOR_OUTPUT_SCHEMA, "director_kb_override")

    # 3. Trial number zero (first trial)
    assert_valid({
        "trial_number": 0,
        "params": {"color": "blue"},
        "source": "optuna",
        "override_entry": None,
    }, DIRECTOR_OUTPUT_SCHEMA, "director_first_trial")

    # 4. Large trial number
    assert_valid({
        "trial_number": 9999,
        "params": {"format": "portrait", "resolution": "4K", "animation": False},
        "source": "optuna",
        "override_entry": None,
    }, DIRECTOR_OUTPUT_SCHEMA, "director_large_trial")

    # 5. Override with complex params
    assert_valid({
        "trial_number": 7,
        "params": {
            "layout": "hero-first",
            "cta_position": "bottom-right",
            "headline_size": 48,
            "animation": True,
        },
        "source": "knowledge_override",
        "override_entry": {"id": "kb-007", "tier": "rule", "confidence": 0.95, "content": "..."},
    }, DIRECTOR_OUTPUT_SCHEMA, "director_complex_override")


def test_director_output_invalid():
    print("\n=== DIRECTOR_OUTPUT_SCHEMA (invalid) ===")

    # 1. Missing params
    assert_invalid({
        "trial_number": 5,
        "source": "optuna",
        "override_entry": None,
    }, DIRECTOR_OUTPUT_SCHEMA, "missing required", "director_missing_params")

    # 2. Invalid source enum
    assert_invalid({
        "trial_number": 5,
        "params": {"color": "red"},
        "source": "random",
        "override_entry": None,
    }, DIRECTOR_OUTPUT_SCHEMA, "not one of", "director_invalid_source")

    # 3. Negative trial number
    assert_invalid({
        "trial_number": -1,
        "params": {"color": "red"},
        "source": "optuna",
        "override_entry": None,
    }, DIRECTOR_OUTPUT_SCHEMA, "minimum", "director_negative_trial")

    # 4. trial_number is a float
    assert_invalid({
        "trial_number": 5.5,
        "params": {"color": "red"},
        "source": "optuna",
        "override_entry": None,
    }, DIRECTOR_OUTPUT_SCHEMA, "integer", "director_float_trial_number")

    # 5. params is a list instead of object
    assert_invalid({
        "trial_number": 5,
        "params": ["color", "red"],
        "source": "optuna",
        "override_entry": None,
    }, DIRECTOR_OUTPUT_SCHEMA, "object", "director_params_wrong_type")


# ===========================================================================
# GRADER_SCORE_REPORT_OUTPUT_SCHEMA — 5 valid, 5 invalid
# ===========================================================================

def test_grader_score_report_valid():
    print("\n=== GRADER_SCORE_REPORT_OUTPUT_SCHEMA (valid) ===")

    # 1. Full valid score report
    assert_valid({
        "trial_id": 12,
        "overall_score": 0.84,
        "dimensions": {
            "originality": {"score": 0.9, "reasoning": "Very original approach."},
            "brand_alignment": {"score": 0.8, "reasoning": "Strong brand presence."},
            "technical_quality": {"score": 0.8, "reasoning": "High technical quality."},
        },
        "novel_finding": "Warm tones consistently outperform cool tones.",
        "suggest_knowledge_write": True,
        "topic_cluster": "color_theory",
    }, GRADER_SCORE_REPORT_OUTPUT_SCHEMA, "score_report_full")

    # 2. Empty novel_finding string (no finding)
    assert_valid({
        "trial_id": 0,
        "overall_score": 0.0,
        "dimensions": {
            "originality": {"score": 0.0, "reasoning": "Completely unoriginal."},
            "brand_alignment": {"score": 0.0, "reasoning": "Off-brand entirely."},
            "technical_quality": {"score": 0.0, "reasoning": "Very poor quality."},
        },
        "novel_finding": "",
        "suggest_knowledge_write": False,
        "topic_cluster": "general",
    }, GRADER_SCORE_REPORT_OUTPUT_SCHEMA, "score_report_zero_scores")

    # 3. Perfect score
    assert_valid({
        "trial_id": 500,
        "overall_score": 1.0,
        "dimensions": {
            "originality": {"score": 1.0, "reasoning": "Never seen before."},
            "brand_alignment": {"score": 1.0, "reasoning": "Perfect brand alignment."},
            "technical_quality": {"score": 1.0, "reasoning": "Flawless execution."},
        },
        "novel_finding": "Perfect score achieved with warm palette + hero layout combo.",
        "suggest_knowledge_write": True,
        "topic_cluster": "layout_color",
    }, GRADER_SCORE_REPORT_OUTPUT_SCHEMA, "score_report_perfect")

    # 4. Rounded 4-decimal score
    assert_valid({
        "trial_id": 77,
        "overall_score": 0.7325,
        "dimensions": {
            "originality": {"score": 0.72, "reasoning": "Mostly original."},
            "brand_alignment": {"score": 0.80, "reasoning": "Good alignment."},
            "technical_quality": {"score": 0.65, "reasoning": "Some technical issues."},
        },
        "novel_finding": "",
        "suggest_knowledge_write": False,
        "topic_cluster": "general",
    }, GRADER_SCORE_REPORT_OUTPUT_SCHEMA, "score_report_decimal")

    # 5. Large trial_id
    assert_valid({
        "trial_id": 10000,
        "overall_score": 0.55,
        "dimensions": {
            "originality": {"score": 0.4, "reasoning": "Somewhat repetitive."},
            "brand_alignment": {"score": 0.7, "reasoning": "Decent alignment."},
            "technical_quality": {"score": 0.6, "reasoning": "Average quality."},
        },
        "novel_finding": "",
        "suggest_knowledge_write": False,
        "topic_cluster": "typography",
    }, GRADER_SCORE_REPORT_OUTPUT_SCHEMA, "score_report_large_trial")


def test_grader_score_report_invalid():
    print("\n=== GRADER_SCORE_REPORT_OUTPUT_SCHEMA (invalid) ===")

    # 1. overall_score above 1
    assert_invalid({
        "trial_id": 5,
        "overall_score": 1.1,
        "dimensions": {
            "originality": {"score": 0.8, "reasoning": "Great."},
            "brand_alignment": {"score": 0.8, "reasoning": "Good."},
            "technical_quality": {"score": 0.8, "reasoning": "Fine."},
        },
        "novel_finding": "",
        "suggest_knowledge_write": False,
        "topic_cluster": "general",
    }, GRADER_SCORE_REPORT_OUTPUT_SCHEMA, "maximum", "score_report_overall_above_1")

    # 2. Missing dimensions
    assert_invalid({
        "trial_id": 5,
        "overall_score": 0.7,
        "novel_finding": "",
        "suggest_knowledge_write": False,
        "topic_cluster": "general",
    }, GRADER_SCORE_REPORT_OUTPUT_SCHEMA, "missing required", "score_report_missing_dimensions")

    # 3. Missing originality in dimensions
    assert_invalid({
        "trial_id": 5,
        "overall_score": 0.7,
        "dimensions": {
            "brand_alignment": {"score": 0.7, "reasoning": "OK."},
            "technical_quality": {"score": 0.7, "reasoning": "Fine."},
        },
        "novel_finding": "",
        "suggest_knowledge_write": False,
        "topic_cluster": "general",
    }, GRADER_SCORE_REPORT_OUTPUT_SCHEMA, "missing required", "score_report_missing_dimension_key")

    # 4. trial_id negative
    assert_invalid({
        "trial_id": -5,
        "overall_score": 0.7,
        "dimensions": {
            "originality": {"score": 0.7, "reasoning": "OK."},
            "brand_alignment": {"score": 0.7, "reasoning": "OK."},
            "technical_quality": {"score": 0.7, "reasoning": "OK."},
        },
        "novel_finding": "",
        "suggest_knowledge_write": False,
        "topic_cluster": "general",
    }, GRADER_SCORE_REPORT_OUTPUT_SCHEMA, "minimum", "score_report_negative_trial_id")

    # 5. suggest_knowledge_write wrong type
    assert_invalid({
        "trial_id": 5,
        "overall_score": 0.7,
        "dimensions": {
            "originality": {"score": 0.7, "reasoning": "OK."},
            "brand_alignment": {"score": 0.7, "reasoning": "OK."},
            "technical_quality": {"score": 0.7, "reasoning": "OK."},
        },
        "novel_finding": "",
        "suggest_knowledge_write": 1,
        "topic_cluster": "general",
    }, GRADER_SCORE_REPORT_OUTPUT_SCHEMA, "boolean", "score_report_suggest_int")


# ===========================================================================
# DIVERSITY_GUARD_OUTPUT_SCHEMA — 5 valid, 5 invalid
# ===========================================================================

def test_diversity_guard_valid():
    print("\n=== DIVERSITY_GUARD_OUTPUT_SCHEMA (valid) ===")

    # 1. No flags, healthy state
    assert_valid({
        "flagged_assets": [],
        "force_random": False,
        "coverage_ratio": 0.23,
        "concentration_map": {"aspect_ratio=16:9": 0.22, "color=warm": 0.18},
        "recommendations": ["Diversity health is good. No intervention needed."],
        "asset_status_updates": {"aspect_ratio=16:9": "healthy", "color=warm": "healthy"},
        "total_cycles": 47,
    }, DIVERSITY_GUARD_OUTPUT_SCHEMA, "guard_healthy")

    # 2. Flagged assets + force random
    assert_valid({
        "flagged_assets": ["color=warm", "layout=hero"],
        "force_random": True,
        "coverage_ratio": 0.08,
        "concentration_map": {"color=warm": 0.45, "layout=hero": 0.38, "color=cool": 0.17},
        "recommendations": [
            "Rotate or deprecate overused assets: color=warm, layout=hero",
            "Cycle 50 hits the random-exploration interval (50). Inject a RandomSampler trial.",
            "Search space coverage is very low (8.0%). Consider broadening exploration.",
        ],
        "asset_status_updates": {"color=warm": "flagged", "layout=hero": "flagged", "color=cool": "healthy"},
        "total_cycles": 50,
    }, DIVERSITY_GUARD_OUTPUT_SCHEMA, "guard_flagged_force_random")

    # 3. Deprecated asset
    assert_valid({
        "flagged_assets": ["format=square"],
        "force_random": False,
        "coverage_ratio": 0.55,
        "concentration_map": {"format=square": 0.72, "format=portrait": 0.28},
        "recommendations": ["Rotate or deprecate overused assets: format=square"],
        "asset_status_updates": {"format=square": "deprecated", "format=portrait": "healthy"},
        "total_cycles": 120,
    }, DIVERSITY_GUARD_OUTPUT_SCHEMA, "guard_deprecated")

    # 4. Full coverage
    assert_valid({
        "flagged_assets": [],
        "force_random": False,
        "coverage_ratio": 1.0,
        "concentration_map": {},
        "recommendations": ["Search space coverage is high (100.0%). Focusing on refinement is appropriate."],
        "asset_status_updates": {},
        "total_cycles": 512,
    }, DIVERSITY_GUARD_OUTPUT_SCHEMA, "guard_full_coverage")

    # 5. Zero total cycles (start of run)
    assert_valid({
        "flagged_assets": [],
        "force_random": False,
        "coverage_ratio": 0.0,
        "concentration_map": {},
        "recommendations": ["Diversity health is good. No intervention needed."],
        "asset_status_updates": {},
        "total_cycles": 0,
    }, DIVERSITY_GUARD_OUTPUT_SCHEMA, "guard_zero_cycles")


def test_diversity_guard_invalid():
    print("\n=== DIVERSITY_GUARD_OUTPUT_SCHEMA (invalid) ===")

    # 1. coverage_ratio above 1.0
    assert_invalid({
        "flagged_assets": [],
        "force_random": False,
        "coverage_ratio": 1.5,
        "concentration_map": {},
        "recommendations": ["All good."],
        "asset_status_updates": {},
        "total_cycles": 10,
    }, DIVERSITY_GUARD_OUTPUT_SCHEMA, "maximum", "guard_coverage_above_1")

    # 2. force_random is string
    assert_invalid({
        "flagged_assets": [],
        "force_random": "no",
        "coverage_ratio": 0.5,
        "concentration_map": {},
        "recommendations": ["All good."],
        "asset_status_updates": {},
        "total_cycles": 10,
    }, DIVERSITY_GUARD_OUTPUT_SCHEMA, "boolean", "guard_force_random_wrong_type")

    # 3. Missing recommendations
    assert_invalid({
        "flagged_assets": [],
        "force_random": False,
        "coverage_ratio": 0.5,
        "concentration_map": {},
        "asset_status_updates": {},
        "total_cycles": 10,
    }, DIVERSITY_GUARD_OUTPUT_SCHEMA, "missing required", "guard_missing_recommendations")

    # 4. flagged_assets contains non-string
    assert_invalid({
        "flagged_assets": [123, "color=warm"],
        "force_random": False,
        "coverage_ratio": 0.3,
        "concentration_map": {},
        "recommendations": ["Rotate assets."],
        "asset_status_updates": {},
        "total_cycles": 25,
    }, DIVERSITY_GUARD_OUTPUT_SCHEMA, "string", "guard_flagged_wrong_item_type")

    # 5. negative total_cycles
    assert_invalid({
        "flagged_assets": [],
        "force_random": False,
        "coverage_ratio": 0.0,
        "concentration_map": {},
        "recommendations": ["All good."],
        "asset_status_updates": {},
        "total_cycles": -1,
    }, DIVERSITY_GUARD_OUTPUT_SCHEMA, "minimum", "guard_negative_cycles")


# ===========================================================================
# CONSOLIDATOR schemas — valid and invalid
# ===========================================================================

def test_consolidator_schemas():
    print("\n=== CONSOLIDATOR schemas ===")

    # Valid skipped
    assert_valid({
        "skipped": True,
        "reason": "not_due",
        "current_cycle": 37,
    }, CONSOLIDATOR_SKIPPED_OUTPUT_SCHEMA, "consolidator_skipped")

    # Invalid skipped — wrong reason
    assert_invalid({
        "skipped": True,
        "reason": "error",
        "current_cycle": 37,
    }, CONSOLIDATOR_SKIPPED_OUTPUT_SCHEMA, "not one of", "consolidator_skipped_bad_reason")

    # Valid full run
    assert_valid({
        "skipped": False,
        "current_cycle": 100,
        "importance_entries_written": 5,
        "changelog": "2 promoted, 1 merged, 0 archived",
        "best_params": {"color": "warm", "layout": "hero"},
        "importances": {"color": 0.45, "layout": 0.33, "font": 0.22},
        "synthesis_prompt": "You are a knowledge synthesis expert. Below are statistical findings...",
    }, CONSOLIDATOR_OUTPUT_SCHEMA, "consolidator_full_run")

    # Valid full run — null changelog (no consolidation performed)
    assert_valid({
        "skipped": False,
        "current_cycle": 50,
        "importance_entries_written": 0,
        "changelog": None,
        "best_params": {},
        "importances": {},
        "synthesis_prompt": "No importances available to synthesize at this time.",
    }, CONSOLIDATOR_OUTPUT_SCHEMA, "consolidator_null_changelog")

    # Invalid — missing importances
    assert_invalid({
        "skipped": False,
        "current_cycle": 50,
        "importance_entries_written": 3,
        "changelog": None,
        "best_params": {},
        "synthesis_prompt": "Some synthesis prompt text here.",
    }, CONSOLIDATOR_OUTPUT_SCHEMA, "missing required", "consolidator_missing_importances")


# ===========================================================================
# CONSOLIDATOR_SYNTHESIS_LLM_SCHEMA — 5 valid, 5 invalid
# ===========================================================================

def test_consolidator_synthesis_llm_valid():
    print("\n=== CONSOLIDATOR_SYNTHESIS_LLM_SCHEMA (valid) ===")

    # 1. Full response
    assert_valid({
        "insights": [
            {
                "parameter": "color_palette",
                "importance_rank": 1,
                "insight": "Color palette is the most impactful parameter, driving originality scores above 0.8 when warm tones are used.",
                "suggested_tier_elevation": "pattern",
            },
            {
                "parameter": "layout",
                "importance_rank": 2,
                "insight": "Hero-first layouts consistently outperform grid layouts in brand alignment.",
                "suggested_tier_elevation": None,
            },
        ],
        "surprising_findings": ["Low font_size importance despite it being a key brand element."],
        "patterns_to_elevate": ["Warm palette → high originality"],
        "overall_summary": "Color palette dominates trial outcomes. Hero layout is secondary driver. Font parameters have minimal impact and may be worth removing from the search space to speed up exploration.",
    }, CONSOLIDATOR_SYNTHESIS_LLM_SCHEMA, "synthesis_full")

    # 2. Empty insights (no importances available)
    assert_valid({
        "insights": [],
        "surprising_findings": [],
        "patterns_to_elevate": [],
        "overall_summary": "Insufficient trial data for meaningful synthesis. More cycles needed to derive reliable patterns.",
    }, CONSOLIDATOR_SYNTHESIS_LLM_SCHEMA, "synthesis_empty")

    # 3. Single high-importance insight
    assert_valid({
        "insights": [
            {
                "parameter": "resolution",
                "importance_rank": 1,
                "insight": "Resolution is the dominant factor, with 4K outputs scoring significantly higher on technical quality.",
                "suggested_tier_elevation": "rule",
            }
        ],
        "surprising_findings": ["Brand alignment appears insensitive to resolution — unexpected."],
        "patterns_to_elevate": [],
        "overall_summary": "Resolution emerges as the single most important variable. Consider fixing it at 4K for all future trials to reduce search space.",
    }, CONSOLIDATOR_SYNTHESIS_LLM_SCHEMA, "synthesis_single_insight")

    # 4. Insight with elevation to 'observation'
    assert_valid({
        "insights": [
            {
                "parameter": "animation",
                "importance_rank": 3,
                "insight": "Animation shows moderate importance but is inconsistent — high variance suggests interaction with other params.",
                "suggested_tier_elevation": "observation",
            }
        ],
        "surprising_findings": [],
        "patterns_to_elevate": ["Static assets outperform animated ones in brand alignment checks."],
        "overall_summary": "Animation is not yet well understood. More targeted exploration recommended before drawing conclusions.",
    }, CONSOLIDATOR_SYNTHESIS_LLM_SCHEMA, "synthesis_observation_elevation")

    # 5. Multiple surprising findings
    assert_valid({
        "insights": [
            {
                "parameter": "font_family",
                "importance_rank": 1,
                "insight": "Font family has unexpectedly high importance for brand alignment scores.",
                "suggested_tier_elevation": "pattern",
            }
        ],
        "surprising_findings": [
            "Font family outranks color palette — contradicts prior assumptions.",
            "Image resolution has near-zero importance despite hypothesis.",
        ],
        "patterns_to_elevate": ["Sans-serif fonts → higher brand alignment"],
        "overall_summary": "Font family emerges as the primary driver, overturning the color-first assumption. Redesign hypothesis prioritization accordingly.",
    }, CONSOLIDATOR_SYNTHESIS_LLM_SCHEMA, "synthesis_surprising")


def test_consolidator_synthesis_llm_invalid():
    print("\n=== CONSOLIDATOR_SYNTHESIS_LLM_SCHEMA (invalid) ===")

    # 1. Missing overall_summary
    assert_invalid({
        "insights": [],
        "surprising_findings": [],
        "patterns_to_elevate": [],
    }, CONSOLIDATOR_SYNTHESIS_LLM_SCHEMA, "missing required", "synthesis_missing_summary")

    # 2. overall_summary too short
    assert_invalid({
        "insights": [],
        "surprising_findings": [],
        "patterns_to_elevate": [],
        "overall_summary": "Short.",
    }, CONSOLIDATOR_SYNTHESIS_LLM_SCHEMA, "minLength", "synthesis_summary_too_short")

    # 3. Invalid suggested_tier_elevation
    assert_invalid({
        "insights": [
            {
                "parameter": "color",
                "importance_rank": 1,
                "insight": "Color is very important for the overall quality of outputs.",
                "suggested_tier_elevation": "tier3",
            }
        ],
        "surprising_findings": [],
        "patterns_to_elevate": [],
        "overall_summary": "Color dominates outcomes. Recommend elevating color-related patterns.",
    }, CONSOLIDATOR_SYNTHESIS_LLM_SCHEMA, "not one of", "synthesis_bad_tier_elevation")

    # 4. importance_rank below minimum (0)
    assert_invalid({
        "insights": [
            {
                "parameter": "layout",
                "importance_rank": 0,
                "insight": "Layout has minimal impact on final scores.",
                "suggested_tier_elevation": None,
            }
        ],
        "surprising_findings": [],
        "patterns_to_elevate": [],
        "overall_summary": "Layout appears unimportant. Consider removing from search space.",
    }, CONSOLIDATOR_SYNTHESIS_LLM_SCHEMA, "minimum", "synthesis_rank_zero")

    # 5. insights is a dict, not array
    assert_invalid({
        "insights": {"color": "important"},
        "surprising_findings": [],
        "patterns_to_elevate": [],
        "overall_summary": "Color is the most important parameter by a significant margin.",
    }, CONSOLIDATOR_SYNTHESIS_LLM_SCHEMA, "array", "synthesis_insights_wrong_type")


# ===========================================================================
# CREATOR phase output schemas
# ===========================================================================

def test_creator_output_schemas():
    print("\n=== CREATOR phase output schemas ===")

    # Valid generation phase
    assert_valid({
        "phase": "generation",
        "prompt": "You are a professional content creator. Generate content for...",
        "hypothesis": {"color": "warm", "layout": "hero", "font_size": 16},
        "kb_context": [
            {"id": "kb-001", "content": "Warm tones perform well.", "tier": "pattern", "confidence": 0.85}
        ],
    }, CREATOR_GENERATION_OUTPUT_SCHEMA, "creator_gen_valid")

    # Invalid — wrong phase enum
    assert_invalid({
        "phase": "critique",  # wrong for generation schema
        "prompt": "Generate content...",
        "hypothesis": {},
        "kb_context": [],
    }, CREATOR_GENERATION_OUTPUT_SCHEMA, "not one of", "creator_gen_wrong_phase")

    # Valid critique phase
    assert_valid({
        "phase": "critique",
        "prompt": "Review the following generated content against established knowledge patterns...",
        "generated_content": "A warm-toned hero layout with 16px body font and clear CTA placement.",
    }, CREATOR_CRITIQUE_OUTPUT_SCHEMA, "creator_critique_valid")

    # Invalid — missing generated_content
    assert_invalid({
        "phase": "critique",
        "prompt": "Review the content...",
    }, CREATOR_CRITIQUE_OUTPUT_SCHEMA, "missing required", "creator_critique_missing_content")


# ===========================================================================
# GRADER evaluation phase schema
# ===========================================================================

def test_grader_evaluation_output_schema():
    print("\n=== GRADER_EVALUATION_OUTPUT_SCHEMA ===")

    # Valid with all fields
    assert_valid({
        "phase": "evaluation",
        "prompt": "You are a professional content evaluator. Content to evaluate: /outputs/trial_42.png",
        "output_path": "/outputs/trial_42.png",
        "rubric": {"originality": "Must differ from last 10 trials.", "brand_alignment": "Follow brand guide v2."},
    }, GRADER_EVALUATION_OUTPUT_SCHEMA, "grader_eval_full")

    # Valid with null output_path and rubric
    assert_valid({
        "phase": "evaluation",
        "prompt": "You are a professional content evaluator. Content: (provided inline by orchestrator)",
        "output_path": None,
        "rubric": None,
    }, GRADER_EVALUATION_OUTPUT_SCHEMA, "grader_eval_null_fields")

    # Invalid — wrong phase
    assert_invalid({
        "phase": "scoring",
        "prompt": "Evaluate this content...",
        "output_path": None,
        "rubric": None,
    }, GRADER_EVALUATION_OUTPUT_SCHEMA, "not one of", "grader_eval_wrong_phase")


# ===========================================================================
# Run all tests
# ===========================================================================

def run_all():
    tests = [
        test_grader_llm_valid,
        test_grader_llm_invalid,
        test_creator_critique_llm_valid,
        test_creator_critique_llm_invalid,
        test_director_output_valid,
        test_director_output_invalid,
        test_grader_score_report_valid,
        test_grader_score_report_invalid,
        test_diversity_guard_valid,
        test_diversity_guard_invalid,
        test_consolidator_schemas,
        test_consolidator_synthesis_llm_valid,
        test_consolidator_synthesis_llm_invalid,
        test_creator_output_schemas,
        test_grader_evaluation_output_schema,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"\n  ❌ {test_fn.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n  💥 {test_fn.__name__} ERROR: {type(e).__name__}: {e}")
            failed += 1

    total = passed + failed
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} test functions passed ({failed} failed)")
    if failed == 0:
        print("✅ All tests passed.")
    else:
        print("❌ Some tests failed — check output above.")
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
