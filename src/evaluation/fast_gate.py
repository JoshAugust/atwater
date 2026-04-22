"""
src.evaluation.fast_gate — Rule-based quality gate (<10ms).

ZERO external dependencies.  Pure Python only.  No ML models.

All checks operate on a CreativeOutput's pre-computed metadata fields.
Gates that cannot run due to missing data emit a warning and pass (never block).

Public API:
  FastGate.check_format(output)       -> tuple[bool, list[str]]
  FastGate.check_typography(output)   -> tuple[bool, list[str]]
  FastGate.check_color_palette(output, brand_colors) -> tuple[bool, list[str]]
  FastGate.check_composition(output)  -> tuple[bool, list[str]]
  FastGate.run_all(output)            -> GateResult
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Avoid circular import — import types from parent at runtime below.


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RulesConfig:
    """
    Configurable thresholds for all FastGate checks.

    Format checks
    -------------
    allowed_formats : list[str]
        Permitted file format strings.  Case-insensitive comparison.
    allowed_color_spaces : list[str]
        Permitted color spaces.
    min_width, min_height : int
        Minimum asset dimensions in pixels.
    max_width, max_height : int
        Maximum asset dimensions in pixels (0 = no limit).

    Typography checks
    -----------------
    min_contrast_ratio : float
        WCAG 2.1 minimum contrast ratio.  AA = 4.5 for normal text.
    min_font_size : int
        Minimum body font size in points.
    max_font_size : int
        Maximum heading font size in points.

    Color palette checks
    --------------------
    palette_tolerance : float
        Maximum Euclidean distance in RGB space for a colour to be
        considered "on-brand".  Range: 0–441 (max possible distance).
    min_palette_compliance : float
        Fraction of dominant colours that must be on-brand (0–1).

    Composition checks
    ------------------
    require_safe_zone : bool
        Whether safe-zone compliance is required.
    thirds_tolerance : float
        How close (fraction of image) a focal point must be to a
        rule-of-thirds gridline to count as compliant.
    min_balance_score : float
        Minimum left/right, top/bottom balance score (0–1).
    """

    # Format
    allowed_formats: list[str] = field(default_factory=lambda: [
        "PNG", "JPEG", "JPG", "WEBP", "SVG", "PDF",
    ])
    allowed_color_spaces: list[str] = field(default_factory=lambda: [
        "RGB", "RGBA", "L", "LA",
    ])
    min_width: int = 100
    min_height: int = 100
    max_width: int = 0   # 0 = no limit
    max_height: int = 0  # 0 = no limit

    # Typography
    min_contrast_ratio: float = 4.5    # WCAG 2.1 AA
    min_font_size: int = 8             # pts
    max_font_size: int = 200           # pts

    # Color palette
    palette_tolerance: float = 60.0    # Euclidean RGB distance
    min_palette_compliance: float = 0.5  # at least 50% of colours on-brand

    # Composition
    require_safe_zone: bool = False
    thirds_tolerance: float = 0.1     # 10% of image dimension
    min_balance_score: float = 0.0    # 0 = disabled by default


# ---------------------------------------------------------------------------
# FastGate
# ---------------------------------------------------------------------------


class FastGate:
    """
    Rule-based quality gate — runs in <10ms with zero external dependencies.

    Parameters
    ----------
    config : RulesConfig | None
        Threshold configuration.  Uses sensible defaults if None.
    """

    def __init__(self, config: RulesConfig | None = None) -> None:
        self._config = config or RulesConfig()

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_format(self, output: Any) -> tuple[bool, list[str]]:
        """
        Validate asset format, dimensions, and color space.

        Parameters
        ----------
        output : CreativeOutput
            The creative artefact to check.

        Returns
        -------
        (passed, failures)
            passed  — True if all checks pass or data is missing (warn only).
            failures — list of human-readable failure descriptions.
        """
        failures: list[str] = []
        cfg = self._config

        # --- File format ---
        if output.format is not None:
            fmt = output.format.upper()
            allowed = [f.upper() for f in cfg.allowed_formats]
            if fmt not in allowed:
                failures.append(
                    f"Format '{output.format}' not in allowed formats {cfg.allowed_formats}"
                )
        else:
            logger.debug("check_format: output.format is None — skipping format check")

        # --- Dimensions ---
        if output.width is not None and output.height is not None:
            w, h = output.width, output.height

            if w < cfg.min_width:
                failures.append(f"Width {w}px below minimum {cfg.min_width}px")
            if h < cfg.min_height:
                failures.append(f"Height {h}px below minimum {cfg.min_height}px")
            if cfg.max_width > 0 and w > cfg.max_width:
                failures.append(f"Width {w}px exceeds maximum {cfg.max_width}px")
            if cfg.max_height > 0 and h > cfg.max_height:
                failures.append(f"Height {h}px exceeds maximum {cfg.max_height}px")
        else:
            logger.debug("check_format: dimensions missing — skipping dimension check")

        # --- Color space ---
        if output.color_space is not None:
            cs = output.color_space.upper()
            allowed_cs = [a.upper() for a in cfg.allowed_color_spaces]
            if cs not in allowed_cs:
                failures.append(
                    f"Color space '{output.color_space}' not in allowed "
                    f"spaces {cfg.allowed_color_spaces}"
                )
        else:
            logger.debug("check_format: color_space is None — skipping color space check")

        return len(failures) == 0, failures

    def check_typography(self, output: Any) -> tuple[bool, list[str]]:
        """
        Validate typography: contrast ratio (WCAG AA) and font size bounds.

        Requires output.typography dict with any of:
            contrast_ratio (float)
            min_font_size  (int)
            max_font_size  (int)

        Returns
        -------
        (passed, failures)
        """
        failures: list[str] = []
        cfg = self._config

        if output.typography is None:
            logger.debug("check_typography: typography metadata missing — skipping")
            return True, []

        typo = output.typography

        # --- Contrast ratio (WCAG 2.1 AA) ---
        if "contrast_ratio" in typo:
            ratio = float(typo["contrast_ratio"])
            if ratio < cfg.min_contrast_ratio:
                failures.append(
                    f"Contrast ratio {ratio:.2f}:1 below WCAG AA minimum "
                    f"{cfg.min_contrast_ratio}:1"
                )

        # --- Font size lower bound ---
        if "min_font_size" in typo:
            min_size = int(typo["min_font_size"])
            if min_size < cfg.min_font_size:
                failures.append(
                    f"Minimum font size {min_size}pt below allowed minimum {cfg.min_font_size}pt"
                )

        # --- Font size upper bound ---
        if "max_font_size" in typo:
            max_size = int(typo["max_font_size"])
            if max_size > cfg.max_font_size:
                failures.append(
                    f"Maximum font size {max_size}pt exceeds allowed maximum "
                    f"{cfg.max_font_size}pt"
                )

        return len(failures) == 0, failures

    def check_color_palette(
        self,
        output: Any,
        brand_colors: list[tuple[int, int, int]] | None = None,
    ) -> tuple[bool, list[str]]:
        """
        Validate color palette compliance against brand colours.

        Parameters
        ----------
        output : CreativeOutput
        brand_colors : list of (R, G, B) tuples | None
            Brand colour palette.  If None or empty, check is skipped.

        Returns
        -------
        (passed, failures)
        """
        failures: list[str] = []
        cfg = self._config

        if not brand_colors:
            logger.debug("check_color_palette: no brand colors provided — skipping")
            return True, []

        if output.color_palette is None or len(output.color_palette) == 0:
            logger.debug("check_color_palette: output.color_palette is None — skipping")
            return True, []

        palette = output.color_palette
        compliant_count = 0

        for color in palette:
            # Find minimum distance to any brand colour
            min_dist = min(
                _rgb_distance(color, bc) for bc in brand_colors
            )
            if min_dist <= cfg.palette_tolerance:
                compliant_count += 1

        compliance_ratio = compliant_count / len(palette)

        if compliance_ratio < cfg.min_palette_compliance:
            failures.append(
                f"Color palette compliance {compliance_ratio:.0%} below minimum "
                f"{cfg.min_palette_compliance:.0%}. "
                f"{compliant_count}/{len(palette)} dominant colours are on-brand "
                f"(tolerance={cfg.palette_tolerance:.0f} RGB distance)."
            )

        return len(failures) == 0, failures

    def check_composition(self, output: Any) -> tuple[bool, list[str]]:
        """
        Validate composition: safe zones and rule-of-thirds focal point.

        Requires output.composition dict with any of:
            safe_zone_clear (bool)
            focal_point     (tuple[float, float])  — (x, y) relative 0-1
            balance_score   (float 0-1)

        Returns
        -------
        (passed, failures)
        """
        failures: list[str] = []
        cfg = self._config

        if output.composition is None:
            logger.debug("check_composition: composition metadata missing — skipping")
            return True, []

        comp = output.composition

        # --- Safe zone ---
        if cfg.require_safe_zone and "safe_zone_clear" in comp:
            if not comp["safe_zone_clear"]:
                failures.append(
                    "Content violates safe zone — text or logos too close to edge"
                )

        # --- Rule of thirds ---
        if "focal_point" in comp:
            fx, fy = comp["focal_point"]
            tol = cfg.thirds_tolerance
            thirds_x = [1/3, 2/3]
            thirds_y = [1/3, 2/3]

            on_thirds_x = any(abs(fx - t) <= tol for t in thirds_x)
            on_thirds_y = any(abs(fy - t) <= tol for t in thirds_y)

            if not (on_thirds_x or on_thirds_y):
                failures.append(
                    f"Focal point ({fx:.2f}, {fy:.2f}) does not align with rule-of-thirds "
                    f"gridlines (tolerance ±{tol:.0%})"
                )

        # --- Balance score ---
        if cfg.min_balance_score > 0 and "balance_score" in comp:
            bal = float(comp["balance_score"])
            if bal < cfg.min_balance_score:
                failures.append(
                    f"Balance score {bal:.2f} below minimum {cfg.min_balance_score:.2f}"
                )

        return len(failures) == 0, failures

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def run_all(
        self,
        output: Any,
        brand_colors: list[tuple[int, int, int]] | None = None,
    ) -> Any:  # returns GateResult — typed at runtime to avoid circular import
        """
        Run all fast checks and return an aggregate GateResult.

        Parameters
        ----------
        output : CreativeOutput
        brand_colors : list of (R, G, B) | None
            Brand colours for palette compliance check.

        Returns
        -------
        GateResult
        """
        from src.evaluation import GateResult  # lazy import

        t0 = time.perf_counter()

        all_failures: list[str] = []
        check_results: list[tuple[bool, list[str]]] = []

        # Run all four checks
        check_results.append(self.check_format(output))
        check_results.append(self.check_typography(output))
        check_results.append(self.check_color_palette(output, brand_colors))
        check_results.append(self.check_composition(output))

        for passed, failures in check_results:
            all_failures.extend(failures)

        passed = len(all_failures) == 0

        # Score: fraction of checks that passed, weighted by failures
        n_checks = len(check_results)
        n_failed_checks = sum(1 for p, _ in check_results if not p)
        score = (n_checks - n_failed_checks) / n_checks if n_checks > 0 else 1.0

        elapsed_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "[FastGate] passed=%s score=%.3f failures=%d time_ms=%.2f",
            passed, score, len(all_failures), elapsed_ms,
        )

        return GateResult(
            passed=passed,
            score=score,
            failures=all_failures,
            time_ms=elapsed_ms,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rgb_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    """Euclidean distance between two RGB colours (0–441 range)."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def wcag_contrast_ratio(
    foreground: tuple[int, int, int],
    background: tuple[int, int, int],
) -> float:
    """
    Compute WCAG 2.1 contrast ratio between two sRGB colours.

    Returns a ratio ≥ 1.0 (higher is more contrast).
    WCAG AA requires ≥ 4.5 for normal text, ≥ 3.0 for large text.
    """
    lum_fg = _relative_luminance(foreground)
    lum_bg = _relative_luminance(background)
    lighter = max(lum_fg, lum_bg)
    darker = min(lum_fg, lum_bg)
    return (lighter + 0.05) / (darker + 0.05)


def _relative_luminance(rgb: tuple[int, int, int]) -> float:
    """WCAG 2.1 relative luminance formula."""
    def linearise(c: int) -> float:
        s = c / 255.0
        return s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * linearise(r) + 0.7152 * linearise(g) + 0.0722 * linearise(b)
