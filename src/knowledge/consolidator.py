"""
Knowledge consolidation engine for the Atwater cognitive agent architecture.

Implements the full consolidation pipeline described in KNOWLEDGE_SCALING.md:

1. Cluster active entries by topic (delegated to TopicClusterer).
2. Merge each cluster into a single authoritative entry.
3. Promote entries that have accrued sufficient validation / statistical evidence.
4. Apply confidence decay to stale entries.
5. Auto-demote entries whose confidence has fallen below tier thresholds.
6. Produce a human-readable changelog of every action taken.

Typical usage
-------------
::

    engine = ConsolidationEngine()
    changelog = engine.run_consolidation(knowledge_base, current_cycle=150)
    print(changelog)
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .models import (
    KnowledgeEntry,
    KnowledgeTier,
    PromotionCriteria,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal action records (used to build the changelog)
# ---------------------------------------------------------------------------

@dataclass
class _Action:
    kind: str          # "merge", "promote", "demote", "archive", "decay"
    entry_id: str
    detail: str        # human-readable description


# ---------------------------------------------------------------------------
# ConsolidationEngine
# ---------------------------------------------------------------------------

class ConsolidationEngine:
    """
    Orchestrates the full knowledge-base consolidation lifecycle.

    Parameters
    ----------
    consolidation_interval:
        How many cycles between full consolidation passes.  The caller is
        responsible for checking this; ``run_consolidation`` always runs
        unconditionally.
    decay_grace_cycles:
        Number of cycles of inactivity before confidence decay begins.
        Default 100, per KNOWLEDGE_SCALING.md.
    decay_rate:
        Fraction of confidence *retained* per 100 additional idle cycles
        beyond the grace period.  Default 0.9 (≈ −10 % per 100 cycles).
    pattern_demote_threshold:
        Pattern entries whose decayed confidence falls below this value are
        demoted to *observation*.  Default 0.3.
    observation_archive_threshold:
        Observation entries whose decayed confidence falls below this value
        are archived.  Default 0.1.
    """

    def __init__(
        self,
        consolidation_interval: int = 50,
        decay_grace_cycles: int = 100,
        decay_rate: float = 0.9,
        pattern_demote_threshold: float = 0.3,
        observation_archive_threshold: float = 0.1,
    ) -> None:
        self.consolidation_interval = consolidation_interval
        self.decay_grace_cycles = decay_grace_cycles
        self.decay_rate = decay_rate
        self.pattern_demote_threshold = pattern_demote_threshold
        self.observation_archive_threshold = observation_archive_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_consolidation(
        self,
        knowledge_base: list[KnowledgeEntry],
        current_cycle: int,
        embeddings: np.ndarray | None = None,
    ) -> str:
        """
        Execute a full consolidation pass over *knowledge_base*.

        Steps
        -----
        1. Apply confidence decay to every active entry.
        2. Auto-demote entries whose decayed confidence is too low.
        3. If *embeddings* are supplied, cluster entries and merge each cluster.
        4. Attempt tier promotions on all resulting active entries.
        5. Return a human-readable changelog.

        Parameters
        ----------
        knowledge_base:
            The full list of KnowledgeEntry objects (mutated in-place).
        current_cycle:
            The current optimisation cycle number.
        embeddings:
            Optional pre-computed embedding matrix aligned with
            *knowledge_base*.  When provided, topic clustering and
            cluster merging are performed.  When ``None``, only decay,
            demotion, and promotion passes run.

        Returns
        -------
        str
            Changelog summarising every action taken.
        """
        actions: list[_Action] = []

        active = [e for e in knowledge_base if e.is_active]

        # Step 1 — Confidence decay
        for entry in active:
            old_conf = entry.confidence
            new_conf = self.decay_confidence(entry, current_cycle)
            if not math.isclose(old_conf, new_conf, rel_tol=1e-6):
                entry.confidence = new_conf
                actions.append(_Action(
                    kind="decay",
                    entry_id=entry.id,
                    detail=(
                        f"confidence {old_conf:.3f} → {new_conf:.3f} "
                        f"(idle {current_cycle - entry.last_validated_cycle} cycles)"
                    ),
                ))

        # Step 2 — Auto-demotion
        for entry in list(active):  # list() because we may mutate tier
            result = self.auto_demote(entry, current_cycle)
            if result is not None:
                old_tier, new_tier = result
                actions.append(_Action(
                    kind="demote" if new_tier != "archived" else "archive",
                    entry_id=entry.id,
                    detail=(
                        f"{old_tier} → {new_tier} "
                        f"(confidence={entry.confidence:.3f})"
                    ),
                ))

        # Step 3 — Cluster + merge (only when embeddings are available)
        if embeddings is not None:
            active_after_demote = [e for e in knowledge_base if e.is_active]
            active_indices = [i for i, e in enumerate(knowledge_base) if e.is_active]
            active_embeddings = embeddings[active_indices]

            from .clustering import TopicClusterer
            clusterer = TopicClusterer()
            clusters = clusterer.cluster_entries(active_after_demote, active_embeddings)

            for cluster_key, cluster_entries in clusters.items():
                if cluster_key == "unclustered" or len(cluster_entries) < 2:
                    continue  # nothing to merge

                merged = self.consolidate_cluster(cluster_entries)
                knowledge_base.append(merged)

                original_ids = [e.id for e in cluster_entries]
                actions.append(_Action(
                    kind="merge",
                    entry_id=merged.id,
                    detail=(
                        f"merged {len(cluster_entries)} entries from {cluster_key!r} "
                        f"→ new entry {merged.id[:8]} "
                        f"(confidence={merged.confidence:.3f}, tier={merged.tier})"
                    ),
                ))
                for eid in original_ids:
                    actions.append(_Action(
                        kind="archive",
                        entry_id=eid,
                        detail=f"archived as source of merged entry {merged.id[:8]}",
                    ))

        # Step 4 — Tier promotions
        for entry in [e for e in knowledge_base if e.is_active]:
            promoted = self._attempt_promotion(entry)
            if promoted:
                old_tier, new_tier = promoted
                actions.append(_Action(
                    kind="promote",
                    entry_id=entry.id,
                    detail=f"{old_tier} → {new_tier}",
                ))

        return self.generate_changelog(actions)  # type: ignore[arg-type]

    def consolidate_cluster(
        self,
        entries: list[KnowledgeEntry],
    ) -> KnowledgeEntry:
        """
        Merge a list of semantically similar entries into one authoritative entry.

        The highest-ranked entry (by ``confidence × validation_count``) is used
        as the base.  Supporting entries contribute to a weighted-average
        confidence and cumulative validation count.  All source entries are
        archived.

        Parameters
        ----------
        entries:
            Two or more entries belonging to the same topic cluster.

        Returns
        -------
        KnowledgeEntry
            A new entry at the appropriate tier, with lineage tracking.
        """
        if not entries:
            raise ValueError("consolidate_cluster requires at least one entry.")

        ranked = sorted(entries, key=lambda e: e.weight, reverse=True)
        base = ranked[0]

        merged_content = self._synthesize_content([e.content for e in ranked])
        merged_confidence = self._weighted_confidence(ranked)
        total_validations = sum(e.validation_count for e in ranked)
        merged_tier = self._determine_promotion_tier(ranked)
        merged_optuna = self._merge_optuna_evidence([e.optuna_evidence for e in ranked])
        earliest_cycle = min(e.created_cycle for e in entries)
        latest_validated = max(e.last_validated_cycle for e in entries)

        merged = KnowledgeEntry(
            content=merged_content,
            tier=merged_tier,
            confidence=merged_confidence,
            created_cycle=earliest_cycle,
            last_validated_cycle=latest_validated,
            validation_count=total_validations,
            contradicted_by=[],
            optuna_evidence=merged_optuna,
            topic_cluster=base.topic_cluster,
            lineage=[e.id for e in entries],
        )

        # Archive source entries
        for entry in entries:
            entry.tier = "archived"

        logger.debug(
            "consolidate_cluster: merged %d entries → %s (tier=%s, conf=%.3f)",
            len(entries),
            merged.id[:8],
            merged.tier,
            merged.confidence,
        )
        return merged

    def resolve_contradictions(
        self,
        entry_a: KnowledgeEntry,
        entry_b: KnowledgeEntry,
        optuna_stats: dict[str, Any] | None = None,
    ) -> KnowledgeEntry:
        """
        Determine which of two contradicting entries should be kept, or create
        a conditional rule that captures both conditions.

        Resolution strategy
        -------------------
        1. If one entry has significantly stronger Optuna evidence (lower p-value
           *and* higher trial count), the weaker entry is archived.
        2. If both have similar statistical weight, a new *conditional* entry is
           created that captures both findings (e.g. "X holds when Y, otherwise Z").
           Both originals are archived.
        3. If neither has Optuna evidence, the entry with the higher composite
           weight (confidence × validation_count) wins.

        Parameters
        ----------
        entry_a, entry_b:
            The two contradicting entries.
        optuna_stats:
            Optional external Optuna summary that supersedes the per-entry
            ``optuna_evidence`` fields.

        Returns
        -------
        KnowledgeEntry
            The surviving entry (possibly a newly created conditional rule).
        """
        stats_a = optuna_stats or entry_a.optuna_evidence or {}
        stats_b = optuna_stats or entry_b.optuna_evidence or {}

        p_a: float | None = stats_a.get("p_value")
        p_b: float | None = stats_b.get("p_value")
        trials_a: int = stats_a.get("trial_count", 0)
        trials_b: int = stats_b.get("trial_count", 0)

        def _archive(loser: KnowledgeEntry, winner: KnowledgeEntry) -> KnowledgeEntry:
            loser.tier = "archived"
            winner.contradicted_by = [
                c for c in winner.contradicted_by if c != loser.id
            ]
            logger.info(
                "resolve_contradictions: archived %s in favour of %s",
                loser.id[:8],
                winner.id[:8],
            )
            return winner

        # Both have p-values → compare statistically
        if p_a is not None and p_b is not None:
            significance_gap = abs(p_a - p_b)
            if significance_gap > 0.05:
                # Clear statistical winner
                if p_a < p_b:
                    return _archive(entry_b, entry_a)
                return _archive(entry_a, entry_b)

            # Similar p-values but one has far more trials
            if abs(trials_a - trials_b) > 50:
                if trials_a > trials_b:
                    return _archive(entry_b, entry_a)
                return _archive(entry_a, entry_b)

            # Roughly equal evidence → create a conditional rule
            return self._create_conditional_entry(entry_a, entry_b)

        # No Optuna evidence → fall back to composite weight
        if entry_a.weight >= entry_b.weight:
            return _archive(entry_b, entry_a)
        return _archive(entry_a, entry_b)

    def decay_confidence(
        self,
        entry: KnowledgeEntry,
        current_cycle: int,
    ) -> float:
        """
        Compute the decayed confidence for *entry* without mutating it.

        No decay is applied for the first ``decay_grace_cycles`` of inactivity.
        Beyond that, confidence is multiplied by ``decay_rate`` per 100 idle
        cycles.  Rules are never decayed.

        Parameters
        ----------
        entry:
            The entry to evaluate.
        current_cycle:
            The current cycle number.

        Returns
        -------
        float
            The new confidence value (clamped to [0.0, 1.0]).
        """
        if entry.tier == "rule":
            return entry.confidence  # Rules are never auto-decayed

        idle = current_cycle - entry.last_validated_cycle
        if idle < self.decay_grace_cycles:
            return entry.confidence

        extra_idle = idle - self.decay_grace_cycles
        decay_factor = self.decay_rate ** (extra_idle / 100.0)
        new_conf = entry.confidence * decay_factor
        return max(0.0, min(1.0, new_conf))

    def auto_demote(
        self,
        entry: KnowledgeEntry,
        current_cycle: int,
    ) -> tuple[KnowledgeTier, KnowledgeTier] | None:
        """
        Demote *entry* if its confidence has fallen below tier thresholds.

        Rules
        -----
        - ``pattern`` with confidence < ``pattern_demote_threshold`` (0.3) →
          demoted to ``observation``.
        - ``observation`` with confidence < ``observation_archive_threshold``
          (0.1) → archived.
        - ``rule`` entries are never auto-demoted.
        - ``archived`` entries are skipped.

        Parameters
        ----------
        entry:
            Entry to evaluate (mutated in-place if demoted).
        current_cycle:
            Used to compute the current decayed confidence before testing.

        Returns
        -------
        tuple[old_tier, new_tier] | None
            The tier transition that occurred, or ``None`` if no demotion
            was needed.
        """
        if entry.tier in ("rule", "archived"):
            return None

        current_conf = self.decay_confidence(entry, current_cycle)

        if entry.tier == "pattern" and current_conf < self.pattern_demote_threshold:
            old_tier = entry.tier
            entry.tier = "observation"
            entry.confidence = current_conf
            logger.info(
                "auto_demote: %s demoted pattern → observation (conf=%.3f)",
                entry.id[:8],
                current_conf,
            )
            return (old_tier, "observation")

        if entry.tier == "observation" and current_conf < self.observation_archive_threshold:
            old_tier = entry.tier
            entry.tier = "archived"
            entry.confidence = current_conf
            logger.info(
                "auto_demote: %s archived (conf=%.3f)", entry.id[:8], current_conf
            )
            return (old_tier, "archived")

        return None

    @staticmethod
    def generate_changelog(actions: list[_Action]) -> str:
        """
        Render a human-readable summary of consolidation actions.

        Parameters
        ----------
        actions:
            Ordered list of ``_Action`` records produced during a
            consolidation pass.

        Returns
        -------
        str
            Multi-line changelog string.
        """
        if not actions:
            return "No changes during this consolidation pass."

        counts: dict[str, int] = {}
        lines: list[str] = ["=== Knowledge Consolidation Changelog ==="]

        for action in actions:
            counts[action.kind] = counts.get(action.kind, 0) + 1
            short_id = action.entry_id[:8]
            lines.append(f"  [{action.kind.upper():8s}] {short_id} — {action.detail}")

        summary_parts = [f"{v} {k}" for k, v in sorted(counts.items())]
        lines.insert(1, "Summary: " + ", ".join(summary_parts))
        lines.insert(2, "-" * 40)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _attempt_promotion(
        self, entry: KnowledgeEntry
    ) -> tuple[KnowledgeTier, KnowledgeTier] | None:
        """Try to promote *entry* to the next tier.  Returns transition or None."""
        if entry.tier == "observation":
            criteria = PromotionCriteria.OBSERVATION_TO_PATTERN  # type: ignore[attr-defined]
            if criteria.satisfied_by(entry):
                old = entry.tier
                entry.tier = "pattern"
                logger.info(
                    "Promoted %s: observation → pattern (val=%d)",
                    entry.id[:8],
                    entry.validation_count,
                )
                return (old, "pattern")

        elif entry.tier == "pattern":
            criteria = PromotionCriteria.PATTERN_TO_RULE  # type: ignore[attr-defined]
            if criteria.satisfied_by(entry):
                old = entry.tier
                entry.tier = "rule"
                logger.info(
                    "Promoted %s: pattern → rule (val=%d)",
                    entry.id[:8],
                    entry.validation_count,
                )
                return (old, "rule")

        return None

    @staticmethod
    def _synthesize_content(contents: list[str]) -> str:
        """
        Produce a merged content string from multiple entries.

        The current implementation uses the highest-ranked entry's content as
        the authoritative text and appends a note about the number of
        supporting observations.  A future version can integrate an LLM
        summarisation step here.
        """
        primary = contents[0]
        n_supporting = len(contents) - 1
        if n_supporting == 0:
            return primary
        suffix = f" [consolidated from {n_supporting} supporting observation(s)]"
        return primary + suffix

    @staticmethod
    def _weighted_confidence(
        entries: list[KnowledgeEntry],
    ) -> float:
        """
        Compute a weighted average confidence across *entries*.

        Each entry's contribution is proportional to its ``validation_count``.
        Falls back to a simple mean when all validation counts are zero.
        """
        total_weight = sum(e.validation_count for e in entries)
        if total_weight == 0:
            return sum(e.confidence for e in entries) / len(entries)
        return sum(e.confidence * e.validation_count for e in entries) / total_weight

    @staticmethod
    def _determine_promotion_tier(
        entries: list[KnowledgeEntry],
    ) -> KnowledgeTier:
        """
        Choose the tier for the merged entry: the highest tier among all sources,
        capped at *pattern* (a merge of observations cannot become a rule directly).
        """
        from .models import TIER_RANK
        best = max(entries, key=lambda e: TIER_RANK[e.tier])
        # Merging can produce at most a pattern; rules are formed through
        # individual promotion via PromotionCriteria.
        if best.tier == "rule":
            return "pattern"
        return best.tier  # "pattern" or "observation"

    @staticmethod
    def _merge_optuna_evidence(
        evidences: list[dict | None],
    ) -> dict | None:
        """
        Merge a list of per-entry Optuna evidence dicts into one summary.

        Strategy: take the minimum p-value (most significant) and the total
        trial count.  Returns ``None`` if no entries have evidence.
        """
        valid = [e for e in evidences if e is not None]
        if not valid:
            return None

        min_p = min((e.get("p_value", 1.0) for e in valid), default=None)
        total_trials = sum(e.get("trial_count", 0) for e in valid)

        merged: dict[str, Any] = {"trial_count": total_trials}
        if min_p is not None:
            merged["p_value"] = min_p

        # Preserve any extra keys from the highest-trial evidence dict
        best_evidence = max(valid, key=lambda e: e.get("trial_count", 0))
        for k, v in best_evidence.items():
            if k not in merged:
                merged[k] = v

        return merged

    def _create_conditional_entry(
        self,
        entry_a: KnowledgeEntry,
        entry_b: KnowledgeEntry,
    ) -> KnowledgeEntry:
        """
        Create a new conditional *pattern* entry that encodes both
        contradicting findings.  Both originals are archived.
        """
        conditional_content = (
            f"[CONDITIONAL] When context matches '{entry_a.content[:80]}' "
            f"OR '{entry_b.content[:80]}' — findings differ; review required."
        )
        merged_conf = (entry_a.confidence + entry_b.confidence) / 2.0
        merged_val = entry_a.validation_count + entry_b.validation_count
        merged_optuna = self._merge_optuna_evidence(
            [entry_a.optuna_evidence, entry_b.optuna_evidence]
        )

        conditional = KnowledgeEntry(
            content=conditional_content,
            tier="pattern",
            confidence=merged_conf,
            created_cycle=min(entry_a.created_cycle, entry_b.created_cycle),
            last_validated_cycle=max(
                entry_a.last_validated_cycle, entry_b.last_validated_cycle
            ),
            validation_count=merged_val,
            optuna_evidence=merged_optuna,
            topic_cluster=entry_a.topic_cluster,
            lineage=[entry_a.id, entry_b.id],
        )

        entry_a.tier = "archived"
        entry_b.tier = "archived"

        logger.info(
            "resolve_contradictions: created conditional entry %s from %s + %s",
            conditional.id[:8],
            entry_a.id[:8],
            entry_b.id[:8],
        )
        return conditional
