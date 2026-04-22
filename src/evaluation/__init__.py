"""
src.evaluation — Grading Pipeline: Verifier Cascade.

The evaluation module implements a three-stage cascade that short-circuits
bad outputs early, saving 60-70% of full LLM grading cost.

Cascade order:
  1. FastGate      (<10ms)   — pure-Python rule-based checks
  2. MediumGate    (~100ms)  — embedding/perceptual model checks
  3. LLMGate       (~2-5s)   — full LLM judge

Supporting module:
  ProcessRewardScorer — per-step scoring for better Optuna gradient signal

Public types:
  CreativeOutput   — the creative artefact passed to all gates
  GateResult       — single-gate result (passed, score, failures, time_ms)
  CascadeResult    — full cascade result from VerifierCascade.evaluate()
  GradeResult      — rich result from LLMGate.evaluate()
  StepScore        — per-step score from ProcessRewardScorer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------


@dataclass
class CreativeOutput:
    """
    Represents a single creative artefact to be evaluated.

    All fields are optional so callers can provide whatever metadata they
    have.  Gates gracefully skip checks for missing data.

    Parameters
    ----------
    output_path : str | None
        Filesystem path to the image/asset.
    format : str | None
        File format string: "PNG", "JPEG", "SVG", "PDF", etc.
    width : int | None
        Asset width in pixels.
    height : int | None
        Asset height in pixels.
    color_space : str | None
        Color space: "RGB", "RGBA", "CMYK", "L" (greyscale), etc.
    typography : dict | None
        Pre-computed typography metadata.  Recognised keys:
            contrast_ratio (float): foreground/background WCAG ratio
            min_font_size  (int):   smallest detected font size in pts
            max_font_size  (int):   largest detected font size in pts
            has_text       (bool):  whether the output contains text
    color_palette : list[tuple[int, int, int]] | None
        Dominant RGB colour tuples extracted from the asset.
    composition : dict | None
        Pre-computed composition metadata.  Recognised keys:
            safe_zone_clear (bool): True if text/logos are inside safe zones
            focal_point     (tuple[float, float]): relative (x, y) of main subject
            balance_score   (float 0-1): left/right & top/bottom balance
    text_description : str | None
        Natural language description of the creative (for CLIP alignment).
    metadata : dict
        Free-form extra metadata (campaign type, trial ID, etc.).
    """

    output_path: str | None = None
    format: str | None = None
    width: int | None = None
    height: int | None = None
    color_space: str | None = None
    typography: dict[str, Any] | None = None
    color_palette: list[tuple[int, int, int]] | None = None
    composition: dict[str, Any] | None = None
    text_description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GateResult:
    """Result from a single gate (fast, medium, or aggregate)."""

    passed: bool
    score: float          # 0.0–1.0
    failures: list[str]   # human-readable failure reasons
    time_ms: float        # wall time for this gate


@dataclass
class GradeResult:
    """Rich result from LLMGate.evaluate()."""

    overall_score: float
    dimension_scores: dict[str, float]        # dimension → 0.0–1.0
    reasoning: dict[str, str]                 # dimension → reasoning text
    novel_finding: str                         # empty string if nothing notable
    suggestions: list[str]                     # improvement suggestions


@dataclass
class CascadeResult:
    """Full result from VerifierCascade.evaluate()."""

    final_score: float
    gates_passed: list[str]      # gate names that passed ("fast", "medium", "llm")
    gates_failed: list[str]      # gate names that failed
    total_time_ms: float
    short_circuited: bool        # True if pipeline stopped before LLM gate
    gate_scores: dict[str, float]             # gate name → score
    gate_details: dict[str, GateResult | GradeResult]  # gate name → full result


@dataclass
class StepScore:
    """Per-step score from ProcessRewardScorer."""

    step_name: str    # "brief" | "concept" | "execution" | "polish"
    score: float      # 0.0–1.0
    reasoning: str
    issues: list[str]


# ---------------------------------------------------------------------------
# Public re-exports (populated after submodule imports)
# ---------------------------------------------------------------------------

__all__ = [
    "CreativeOutput",
    "GateResult",
    "GradeResult",
    "CascadeResult",
    "StepScore",
    "FastGate",
    "MediumGate",
    "LLMGate",
    "VerifierCascade",
    "ProcessRewardScorer",
]

# Lazy imports to avoid circular deps and heavy model loads at import time.
# Users should import directly from submodules for type-checker support.
from src.evaluation.fast_gate import FastGate  # noqa: E402
from src.evaluation.medium_gate import MediumGate  # noqa: E402
from src.evaluation.llm_gate import LLMGate  # noqa: E402
from src.evaluation.cascade import VerifierCascade  # noqa: E402
from src.evaluation.process_rewards import ProcessRewardScorer  # noqa: E402
