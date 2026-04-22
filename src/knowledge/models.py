"""
Knowledge models for the Atwater cognitive agent architecture.

Defines the core data structures for the tiered knowledge base:
- KnowledgeEntry: the atomic unit of stored knowledge
- Tier constants: hierarchy of epistemic confidence
- PromotionCriteria: thresholds for tier advancement
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------

KnowledgeTier = Literal["rule", "pattern", "observation", "archived"]

# Ordered from most to least authoritative (useful for sorting/comparisons)
TIER_RANK: dict[KnowledgeTier, int] = {
    "rule": 3,
    "pattern": 2,
    "observation": 1,
    "archived": 0,
}


# ---------------------------------------------------------------------------
# Core data model
# ---------------------------------------------------------------------------

@dataclass
class KnowledgeEntry:
    """
    A single unit of knowledge in the Atwater knowledge base.

    Entries move up through tiers (observation → pattern → rule) as they
    accumulate validation evidence and statistical support.  They decay and
    can be demoted or archived when they stop being validated.

    Attributes
    ----------
    id:
        Unique identifier (UUID4 string by default).
    content:
        Human-readable description of the knowledge.
    tier:
        Epistemic confidence level.  One of "observation", "pattern",
        "rule", or "archived".
    confidence:
        Floating-point score in [0.0, 1.0].  Decays over time when the
        entry is not validated.
    created_cycle:
        The optimisation cycle in which this entry was first created.
    last_validated_cycle:
        The most recent cycle in which this entry was confirmed by new
        evidence.  Used to drive confidence decay.
    validation_count:
        Total number of times this entry has been independently validated.
    contradicted_by:
        IDs of other KnowledgeEntry objects that conflict with this one.
    optuna_evidence:
        Raw statistical evidence from Optuna (p-values, trial counts, effect
        sizes, etc.).  ``None`` if no Optuna data has been attached yet.
    topic_cluster:
        Label of the HDBSCAN topic cluster this entry belongs to, e.g.
        ``"cluster_4"`` or ``"unclustered"``.
    lineage:
        IDs of entries that were merged/archived to produce this entry.
        Empty list for original observations.
    """

    content: str
    tier: KnowledgeTier = "observation"
    confidence: float = 0.5
    created_cycle: int = 0
    last_validated_cycle: int = 0
    validation_count: int = 0
    contradicted_by: list[str] = field(default_factory=list)
    optuna_evidence: dict | None = None
    topic_cluster: str = "unclustered"
    lineage: list[str] = field(default_factory=list)
    embedding: bytes | None = field(default=None, repr=False)

    # id last so callers can omit it and get an auto-generated UUID
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def embedding_array(self):
        """Decode the stored embedding bytes back to a float32 numpy array."""
        if self.embedding is None:
            return None
        import numpy as np  # lazy import — avoid hard dep in pure-model module
        return np.frombuffer(self.embedding, dtype=np.float32)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def tier_rank(self) -> int:
        """Numeric rank of the current tier (higher = more authoritative)."""
        return TIER_RANK[self.tier]

    @property
    def is_active(self) -> bool:
        """True for any tier that participates in active retrieval."""
        return self.tier != "archived"

    @property
    def weight(self) -> float:
        """Composite retrieval weight: confidence × validation_count."""
        return self.confidence * self.validation_count

    def __repr__(self) -> str:
        short_id = self.id[:8]
        short_content = self.content[:60] + ("…" if len(self.content) > 60 else "")
        return (
            f"KnowledgeEntry(id={short_id!r}, tier={self.tier!r}, "
            f"confidence={self.confidence:.2f}, content={short_content!r})"
        )


# ---------------------------------------------------------------------------
# Promotion criteria
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PromotionCriteria:
    """
    Thresholds that must all be satisfied before a tier promotion is granted.

    Two built-in criteria are provided as class-level constants:

    - ``PromotionCriteria.OBSERVATION_TO_PATTERN``
    - ``PromotionCriteria.PATTERN_TO_RULE``

    Parameters
    ----------
    min_validations:
        Minimum ``validation_count`` required.
    max_p_value:
        Maximum Optuna p-value (significance threshold).  The entry must
        have ``optuna_evidence["p_value"] <= max_p_value``.
    min_trials:
        Minimum number of Optuna trials that produced the evidence.
        Set to 0 when not applicable.
    description:
        Human-readable label for log/changelog output.
    """

    min_validations: int
    max_p_value: float
    min_trials: int
    description: str

    def satisfied_by(self, entry: KnowledgeEntry) -> bool:
        """
        Return True if *entry* meets all numeric thresholds.

        If ``optuna_evidence`` is absent the p-value and trial checks
        automatically fail (the entry cannot be promoted without statistical
        backing).
        """
        if entry.validation_count < self.min_validations:
            return False

        evidence = entry.optuna_evidence or {}
        p_value: float | None = evidence.get("p_value")
        trial_count: int = evidence.get("trial_count", 0)

        if p_value is None:
            return False
        if p_value > self.max_p_value:
            return False
        if trial_count < self.min_trials:
            return False

        return True


# Canonical promotion thresholds (mirrors the table in KNOWLEDGE_SCALING.md)
PromotionCriteria.OBSERVATION_TO_PATTERN = PromotionCriteria(  # type: ignore[attr-defined]
    min_validations=5,
    max_p_value=0.1,
    min_trials=0,
    description="Observation → Pattern (5+ validations, p < 0.1)",
)

PromotionCriteria.PATTERN_TO_RULE = PromotionCriteria(  # type: ignore[attr-defined]
    min_validations=20,
    max_p_value=0.05,
    min_trials=200,
    description="Pattern → Rule (20+ validations, p < 0.05, 200+ trials)",
)
