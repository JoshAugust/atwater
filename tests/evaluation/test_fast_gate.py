"""
Tests for src.evaluation.fast_gate.

Tests cover:
  - known-good outputs pass all checks
  - known-bad outputs fail appropriate checks
  - format validation (allowed/disallowed formats, dimensions, color spaces)
  - typography validation (contrast ratio, font size bounds)
  - color palette compliance
  - composition validation (safe zones, rule of thirds)
  - run_all aggregation
  - RulesConfig customisation
  - wcag_contrast_ratio utility
"""

from __future__ import annotations

import pytest

from src.evaluation import CreativeOutput
from src.evaluation.fast_gate import (
    FastGate,
    RulesConfig,
    wcag_contrast_ratio,
    _rgb_distance,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gate() -> FastGate:
    return FastGate()


@pytest.fixture
def strict_gate() -> FastGate:
    """Gate with strict thresholds for testing failures."""
    return FastGate(RulesConfig(
        min_contrast_ratio=4.5,
        min_font_size=10,
        max_font_size=100,
        palette_tolerance=30.0,
        min_palette_compliance=0.8,
        require_safe_zone=True,
        thirds_tolerance=0.05,
        min_balance_score=0.6,
    ))


@pytest.fixture
def good_output() -> CreativeOutput:
    """A well-formed creative output that should pass all checks."""
    return CreativeOutput(
        output_path="/tmp/test.png",
        format="PNG",
        width=1080,
        height=1080,
        color_space="RGB",
        typography={
            "contrast_ratio": 7.5,
            "min_font_size": 12,
            "max_font_size": 72,
        },
        color_palette=[
            (0, 51, 102),   # brand dark blue
            (255, 255, 255), # white
            (0, 102, 204),  # brand light blue
        ],
        composition={
            "safe_zone_clear": True,
            "focal_point": (0.33, 0.33),  # on rule-of-thirds intersection
            "balance_score": 0.85,
        },
        text_description="A clean professional ad for the brand.",
    )


@pytest.fixture
def bad_output() -> CreativeOutput:
    """A poorly-formed output that should fail multiple checks."""
    return CreativeOutput(
        output_path="/tmp/test_bad.tiff",
        format="TIFF",           # not in allowed list
        width=50,                 # too small
        height=50,
        color_space="CMYK",       # not allowed for digital
        typography={
            "contrast_ratio": 2.1,  # below WCAG AA
            "min_font_size": 4,     # too small
            "max_font_size": 250,   # too large
        },
        color_palette=[
            (255, 0, 0),   # red — not on brand
            (0, 255, 0),   # green — not on brand
        ],
        composition={
            "safe_zone_clear": False,  # violates safe zone
            "focal_point": (0.5, 0.5),  # centred — not on thirds
            "balance_score": 0.2,       # poor balance
        },
    )


# ---------------------------------------------------------------------------
# Tests: check_format
# ---------------------------------------------------------------------------


class TestCheckFormat:
    def test_good_format_passes(self, gate, good_output):
        passed, failures = gate.check_format(good_output)
        assert passed is True
        assert failures == []

    def test_bad_format_fails(self, gate, bad_output):
        passed, failures = gate.check_format(bad_output)
        assert passed is False
        assert any("TIFF" in f for f in failures)

    def test_undersized_image_fails(self, gate):
        output = CreativeOutput(format="PNG", width=50, height=50, color_space="RGB")
        passed, failures = gate.check_format(output)
        assert passed is False
        assert any("Width" in f or "Height" in f for f in failures)

    def test_oversized_image_fails(self):
        cfg = RulesConfig(max_width=2000, max_height=2000)
        gate = FastGate(cfg)
        output = CreativeOutput(format="PNG", width=3000, height=3000, color_space="RGB")
        passed, failures = gate.check_format(output)
        assert passed is False
        assert any("exceeds maximum" in f for f in failures)

    def test_disallowed_color_space_fails(self, gate, bad_output):
        passed, failures = gate.check_format(bad_output)
        assert any("CMYK" in f for f in failures)

    def test_missing_format_skips_gracefully(self, gate):
        """None format should not crash or fail — just skip."""
        output = CreativeOutput(format=None, width=1080, height=1080, color_space="RGB")
        passed, failures = gate.check_format(output)
        assert passed is True  # width/height/color_space are fine

    def test_missing_dimensions_skips_gracefully(self, gate):
        output = CreativeOutput(format="PNG", width=None, height=None, color_space="RGB")
        passed, failures = gate.check_format(output)
        assert passed is True

    def test_all_fields_none_passes(self, gate):
        """Empty output has nothing to fail on."""
        output = CreativeOutput()
        passed, failures = gate.check_format(output)
        assert passed is True
        assert failures == []

    def test_jpeg_allowed(self, gate):
        output = CreativeOutput(format="JPEG", width=800, height=600, color_space="RGB")
        passed, failures = gate.check_format(output)
        assert passed is True

    def test_case_insensitive_format(self, gate):
        output = CreativeOutput(format="png", width=800, height=600, color_space="RGB")
        passed, failures = gate.check_format(output)
        assert passed is True


# ---------------------------------------------------------------------------
# Tests: check_typography
# ---------------------------------------------------------------------------


class TestCheckTypography:
    def test_good_typography_passes(self, gate, good_output):
        passed, failures = gate.check_typography(good_output)
        assert passed is True
        assert failures == []

    def test_low_contrast_fails(self, gate):
        output = CreativeOutput(typography={"contrast_ratio": 2.0})
        passed, failures = gate.check_typography(output)
        assert passed is False
        assert any("contrast" in f.lower() for f in failures)

    def test_exact_wcag_aa_passes(self, gate):
        output = CreativeOutput(typography={"contrast_ratio": 4.5})
        passed, failures = gate.check_typography(output)
        assert passed is True

    def test_font_too_small_fails(self, gate):
        output = CreativeOutput(typography={"min_font_size": 4})
        passed, failures = gate.check_typography(output)
        assert passed is False
        assert any("font size" in f.lower() for f in failures)

    def test_font_too_large_fails(self, strict_gate):
        output = CreativeOutput(typography={"max_font_size": 200})  # > strict 100
        passed, failures = strict_gate.check_typography(output)
        assert passed is False
        assert any("exceeds allowed" in f.lower() for f in failures)

    def test_none_typography_passes(self, gate):
        output = CreativeOutput(typography=None)
        passed, failures = gate.check_typography(output)
        assert passed is True
        assert failures == []

    def test_partial_typography_skips_missing_keys(self, gate):
        """Only keys present in the dict should be checked."""
        output = CreativeOutput(typography={"contrast_ratio": 7.0})  # only ratio
        passed, failures = gate.check_typography(output)
        assert passed is True  # no font size to check → skip


# ---------------------------------------------------------------------------
# Tests: check_color_palette
# ---------------------------------------------------------------------------


class TestCheckColorPalette:
    BRAND_COLORS = [
        (0, 51, 102),    # dark blue
        (255, 255, 255), # white
    ]

    def test_on_brand_palette_passes(self, gate, good_output):
        passed, failures = gate.check_color_palette(good_output, self.BRAND_COLORS)
        assert passed is True

    def test_off_brand_palette_fails(self, strict_gate, bad_output):
        # bad_output has pure red and green — far from brand colors
        passed, failures = strict_gate.check_color_palette(bad_output, self.BRAND_COLORS)
        assert passed is False
        assert any("compliance" in f.lower() for f in failures)

    def test_no_brand_colors_skips(self, gate, good_output):
        passed, failures = gate.check_color_palette(good_output, brand_colors=None)
        assert passed is True
        assert failures == []

    def test_no_palette_skips(self, gate):
        output = CreativeOutput(color_palette=None)
        passed, failures = gate.check_color_palette(output, self.BRAND_COLORS)
        assert passed is True

    def test_empty_palette_skips(self, gate):
        output = CreativeOutput(color_palette=[])
        passed, failures = gate.check_color_palette(output, self.BRAND_COLORS)
        assert passed is True

    def test_partial_compliance(self):
        """50% on-brand should pass default (50% threshold) but fail strict 80%."""
        brand = [(0, 0, 255)]  # blue
        palette = [(0, 0, 255), (255, 0, 0)]  # half on-brand

        gate_default = FastGate()  # default min_palette_compliance=0.5
        gate_strict = FastGate(RulesConfig(min_palette_compliance=0.8))

        output = CreativeOutput(color_palette=palette)
        assert gate_default.check_color_palette(output, brand)[0] is True
        assert gate_strict.check_color_palette(output, brand)[0] is False


# ---------------------------------------------------------------------------
# Tests: check_composition
# ---------------------------------------------------------------------------


class TestCheckComposition:
    def test_good_composition_passes(self, gate, good_output):
        passed, failures = gate.check_composition(good_output)
        assert passed is True

    def test_safe_zone_violation_fails_when_required(self, strict_gate):
        output = CreativeOutput(composition={
            "safe_zone_clear": False,
            "focal_point": (0.33, 0.33),
            "balance_score": 0.8,
        })
        passed, failures = strict_gate.check_composition(output)
        assert passed is False
        assert any("safe zone" in f.lower() for f in failures)

    def test_safe_zone_not_required_by_default(self, gate):
        output = CreativeOutput(composition={"safe_zone_clear": False})
        passed, failures = gate.check_composition(output)
        assert passed is True  # require_safe_zone=False by default

    def test_off_thirds_focal_point_fails_strict(self, strict_gate):
        # (0.5, 0.5) is far from any thirds line with tolerance 0.05
        output = CreativeOutput(composition={
            "focal_point": (0.5, 0.5),
            "safe_zone_clear": True,
            "balance_score": 0.8,
        })
        passed, failures = strict_gate.check_composition(output)
        assert passed is False
        assert any("thirds" in f.lower() for f in failures)

    def test_thirds_aligned_passes(self, strict_gate):
        # Exactly on 1/3 line
        output = CreativeOutput(composition={
            "focal_point": (0.333, 0.5),  # x on thirds line
            "safe_zone_clear": True,
            "balance_score": 0.8,
        })
        passed, failures = strict_gate.check_composition(output)
        assert passed is True

    def test_poor_balance_fails_strict(self, strict_gate):
        output = CreativeOutput(composition={
            "focal_point": (0.33, 0.33),
            "safe_zone_clear": True,
            "balance_score": 0.3,  # below strict min of 0.6
        })
        passed, failures = strict_gate.check_composition(output)
        assert passed is False
        assert any("balance" in f.lower() for f in failures)

    def test_none_composition_skips(self, gate):
        output = CreativeOutput(composition=None)
        passed, failures = gate.check_composition(output)
        assert passed is True


# ---------------------------------------------------------------------------
# Tests: run_all (aggregate)
# ---------------------------------------------------------------------------


class TestRunAll:
    def test_good_output_passes_run_all(self, gate, good_output):
        result = gate.run_all(good_output)
        assert result.passed is True
        assert result.score > 0.5
        assert result.failures == []
        assert result.time_ms >= 0

    def test_bad_output_fails_run_all(self, strict_gate, bad_output):
        brand_colors = [(0, 0, 0), (255, 255, 255)]
        result = strict_gate.run_all(bad_output, brand_colors=brand_colors)
        assert result.passed is False
        assert len(result.failures) > 0
        assert result.score < 1.0

    def test_result_has_required_fields(self, gate, good_output):
        result = gate.run_all(good_output)
        assert hasattr(result, "passed")
        assert hasattr(result, "score")
        assert hasattr(result, "failures")
        assert hasattr(result, "time_ms")

    def test_score_is_normalised(self, gate, bad_output):
        result = gate.run_all(bad_output)
        assert 0.0 <= result.score <= 1.0

    def test_empty_output_passes_with_score_1(self, gate):
        """Empty output has nothing to fail on — all checks skip."""
        result = gate.run_all(CreativeOutput())
        assert result.passed is True
        assert result.score == 1.0


# ---------------------------------------------------------------------------
# Tests: WCAG contrast ratio utility
# ---------------------------------------------------------------------------


class TestWcagContrastRatio:
    def test_black_on_white_is_21(self):
        ratio = wcag_contrast_ratio((0, 0, 0), (255, 255, 255))
        assert abs(ratio - 21.0) < 0.1

    def test_white_on_white_is_1(self):
        ratio = wcag_contrast_ratio((255, 255, 255), (255, 255, 255))
        assert abs(ratio - 1.0) < 0.01

    def test_symmetric(self):
        a, b = (100, 150, 200), (255, 255, 255)
        assert abs(wcag_contrast_ratio(a, b) - wcag_contrast_ratio(b, a)) < 0.01

    def test_wcag_aa_compliant_pair(self):
        # Dark navy on white — should be >> 4.5
        ratio = wcag_contrast_ratio((0, 51, 102), (255, 255, 255))
        assert ratio >= 4.5

    def test_wcag_non_compliant_pair(self):
        # Light grey on white — should be < 4.5
        ratio = wcag_contrast_ratio((200, 200, 200), (255, 255, 255))
        assert ratio < 4.5


# ---------------------------------------------------------------------------
# Tests: RGB distance utility
# ---------------------------------------------------------------------------


class TestRgbDistance:
    def test_same_color_is_zero(self):
        assert _rgb_distance((100, 100, 100), (100, 100, 100)) == 0.0

    def test_black_white_distance(self):
        dist = _rgb_distance((0, 0, 0), (255, 255, 255))
        # Euclidean: sqrt(3 * 255^2) ≈ 441.67
        assert abs(dist - 441.67) < 1.0

    def test_symmetric(self):
        a, b = (100, 50, 200), (30, 80, 110)
        assert abs(_rgb_distance(a, b) - _rgb_distance(b, a)) < 1e-6
