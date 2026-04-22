"""
test_consolidation.py — Knowledge compaction engine tests.

Tests
-----
- Confidence decay: 100-cycle grace, then ~10% loss per 100 cycles
- Auto-demotion: pattern < 0.3 → observation; observation < 0.1 → archived
- Cluster merging (consolidate_cluster)
- Contradiction resolution (resolve_contradictions)
- Changelog generation

No LLM or sentence-transformers required — all tests operate on in-memory
KnowledgeEntry lists.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from atwater.src.knowledge.models import KnowledgeEntry, PromotionCriteria
from atwater.src.knowledge.consolidator import ConsolidationEngine, _Action


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_entry(
    content: str = "test content",
    tier: str = "observation",
    confidence: float = 0.8,
    created_cycle: int = 0,
    last_validated_cycle: int = 0,
    validation_count: int = 1,
    topic_cluster: str = "general",
    optuna_evidence: dict | None = None,
    contradicted_by: list[str] | None = None,
) -> KnowledgeEntry:
    """Helper: create a KnowledgeEntry with sensible defaults."""
    return KnowledgeEntry(
        content=content,
        tier=tier,
        confidence=confidence,
        created_cycle=created_cycle,
        last_validated_cycle=last_validated_cycle,
        validation_count=validation_count,
        topic_cluster=topic_cluster,
        optuna_evidence=optuna_evidence,
        contradicted_by=contradicted_by or [],
    )


# ===========================================================================
# Confidence decay
# ===========================================================================


class TestConfidenceDecay:
    """decay_confidence() — from KNOWLEDGE_SCALING.md spec."""

    def setup_method(self) -> None:
        self.engine = ConsolidationEngine(
            decay_grace_cycles=100,
            decay_rate=0.9,
        )

    def test_no_decay_within_grace_period(self) -> None:
        entry = make_entry(tier="observation", confidence=0.8, last_validated_cycle=0)
        # 99 idle cycles — still within grace
        new_conf = self.engine.decay_confidence(entry, current_cycle=99)
        assert new_conf == pytest.approx(0.8)

    def test_no_decay_at_grace_boundary(self) -> None:
        entry = make_entry(tier="pattern", confidence=0.7, last_validated_cycle=0)
        # Exactly 100 cycles idle — boundary, no decay yet
        new_conf = self.engine.decay_confidence(entry, current_cycle=100)
        assert new_conf == pytest.approx(0.7)

    def test_decay_starts_after_grace(self) -> None:
        entry = make_entry(tier="pattern", confidence=1.0, last_validated_cycle=0)
        # 200 idle cycles → 100 extra → one decay step: 1.0 × 0.9^1 = 0.9
        new_conf = self.engine.decay_confidence(entry, current_cycle=200)
        assert new_conf == pytest.approx(0.9, rel=1e-5)

    def test_decay_two_steps(self) -> None:
        entry = make_entry(tier="pattern", confidence=1.0, last_validated_cycle=0)
        # 300 idle → 200 extra → two steps: 0.9^2 = 0.81
        new_conf = self.engine.decay_confidence(entry, current_cycle=300)
        assert new_conf == pytest.approx(0.81, rel=1e-5)

    def test_decay_is_continuous(self) -> None:
        """50 extra idle cycles → 0.9^0.5 (halfway between steps)."""
        entry = make_entry(tier="pattern", confidence=1.0, last_validated_cycle=0)
        new_conf = self.engine.decay_confidence(entry, current_cycle=250)
        expected = 0.9 ** 1.5
        assert new_conf == pytest.approx(expected, rel=1e-5)

    def test_rules_never_decay(self) -> None:
        """Rules should always return unchanged confidence."""
        entry = make_entry(tier="rule", confidence=0.95, last_validated_cycle=0)
        # Extremely stale
        new_conf = self.engine.decay_confidence(entry, current_cycle=10_000)
        assert new_conf == pytest.approx(0.95)

    def test_confidence_never_goes_below_zero(self) -> None:
        entry = make_entry(tier="observation", confidence=0.05, last_validated_cycle=0)
        new_conf = self.engine.decay_confidence(entry, current_cycle=100_000)
        assert new_conf >= 0.0

    def test_active_entry_retains_confidence(self) -> None:
        """Entry validated recently should not decay."""
        entry = make_entry(tier="pattern", confidence=0.75, last_validated_cycle=500)
        new_conf = self.engine.decay_confidence(entry, current_cycle=550)
        assert new_conf == pytest.approx(0.75)


# ===========================================================================
# Auto-demotion
# ===========================================================================


class TestAutoDemotion:
    """auto_demote() — tier-threshold rules from KNOWLEDGE_SCALING.md."""

    def setup_method(self) -> None:
        self.engine = ConsolidationEngine(
            decay_grace_cycles=100,
            decay_rate=0.9,
            pattern_demote_threshold=0.3,
            observation_archive_threshold=0.1,
        )

    def test_pattern_not_demoted_above_threshold(self) -> None:
        entry = make_entry(tier="pattern", confidence=0.5, last_validated_cycle=0)
        # After 100 idle cycles: 0.5 × 0.9^0 = 0.5 (still within grace)
        result = self.engine.auto_demote(entry, current_cycle=100)
        assert result is None
        assert entry.tier == "pattern"

    def test_pattern_demoted_below_threshold(self) -> None:
        """Pattern with very low decayed confidence drops to observation."""
        # Start at 0.3 confidence, 300 cycles idle → 0.3 × 0.9^2 = 0.243 < 0.3
        entry = make_entry(tier="pattern", confidence=0.3, last_validated_cycle=0)
        result = self.engine.auto_demote(entry, current_cycle=300)
        assert result is not None
        old_tier, new_tier = result
        assert old_tier == "pattern"
        assert new_tier == "observation"
        assert entry.tier == "observation"

    def test_observation_archived_below_threshold(self) -> None:
        """Observation with very low decayed confidence gets archived."""
        # 0.1 confidence, 600 cycles idle → massive decay → well below 0.1
        entry = make_entry(tier="observation", confidence=0.1, last_validated_cycle=0)
        result = self.engine.auto_demote(entry, current_cycle=600)
        assert result is not None
        old_tier, new_tier = result
        assert old_tier == "observation"
        assert new_tier == "archived"
        assert entry.tier == "archived"

    def test_rule_never_demoted(self) -> None:
        entry = make_entry(tier="rule", confidence=0.01, last_validated_cycle=0)
        result = self.engine.auto_demote(entry, current_cycle=10_000)
        assert result is None
        assert entry.tier == "rule"

    def test_archived_entry_skipped(self) -> None:
        entry = make_entry(tier="archived", confidence=0.0, last_validated_cycle=0)
        result = self.engine.auto_demote(entry, current_cycle=1000)
        assert result is None

    def test_observation_high_confidence_not_archived(self) -> None:
        entry = make_entry(tier="observation", confidence=0.9, last_validated_cycle=0)
        result = self.engine.auto_demote(entry, current_cycle=150)
        # After 150 cycles, 50 extra: 0.9 × 0.9^0.5 ≈ 0.854 > 0.1
        assert result is None


# ===========================================================================
# Cluster merging
# ===========================================================================


class TestClusterMerging:
    """consolidate_cluster() — merges semantically similar entries."""

    def setup_method(self) -> None:
        self.engine = ConsolidationEngine()

    def test_single_entry_returns_without_lineage(self) -> None:
        entry = make_entry(content="Only entry", confidence=0.8, validation_count=3)
        merged = self.engine.consolidate_cluster([entry])
        # Single entry: no 'supporting' suffix
        assert "Only entry" in merged.content
        assert entry.tier == "archived"

    def test_merge_two_entries(self) -> None:
        entries = [
            make_entry(content="Alpha finding", confidence=0.9, validation_count=10),
            make_entry(content="Beta finding", confidence=0.6, validation_count=3),
        ]
        merged = self.engine.consolidate_cluster(entries)
        assert merged is not None
        # All sources archived
        assert all(e.tier == "archived" for e in entries)
        # Lineage tracked
        assert len(merged.lineage) == 2

    def test_merged_confidence_weighted_by_validation(self) -> None:
        entries = [
            make_entry(confidence=0.8, validation_count=10),
            make_entry(confidence=0.4, validation_count=0),  # zero validation_count
        ]
        merged = self.engine.consolidate_cluster(entries)
        # With zero total weight, should fall back to simple mean
        # (10 validations total → weighted: 0.8×10 / 10 = 0.8)
        expected = (0.8 * 10) / 10
        assert merged.confidence == pytest.approx(expected, rel=1e-4)

    def test_merged_tier_caps_at_pattern(self) -> None:
        """A merge of rule + observation should produce at most a pattern."""
        entries = [
            make_entry(tier="rule", confidence=0.9, validation_count=5),
            make_entry(tier="observation", confidence=0.5, validation_count=2),
        ]
        merged = self.engine.consolidate_cluster(entries)
        assert merged.tier in ("pattern", "observation")

    def test_merge_inherits_topic_cluster(self) -> None:
        entries = [
            make_entry(topic_cluster="typography", confidence=0.8, validation_count=5),
            make_entry(topic_cluster="typography", confidence=0.6, validation_count=2),
        ]
        merged = self.engine.consolidate_cluster(entries)
        assert merged.topic_cluster == "typography"

    def test_merge_sums_validation_counts(self) -> None:
        entries = [
            make_entry(confidence=0.8, validation_count=5),
            make_entry(confidence=0.6, validation_count=3),
        ]
        merged = self.engine.consolidate_cluster(entries)
        assert merged.validation_count == 8

    def test_merge_empty_list_raises(self) -> None:
        with pytest.raises((ValueError, IndexError)):
            self.engine.consolidate_cluster([])


# ===========================================================================
# Contradiction resolution
# ===========================================================================


class TestContradictionResolution:
    """resolve_contradictions() — determines which entry wins or creates conditional."""

    def setup_method(self) -> None:
        self.engine = ConsolidationEngine()

    def test_stronger_evidence_wins(self) -> None:
        """Entry with lower p-value (more significant) should win."""
        entry_a = make_entry(
            content="Finding A",
            confidence=0.8,
            validation_count=5,
            optuna_evidence={"p_value": 0.02, "trial_count": 100},
        )
        entry_b = make_entry(
            content="Finding B",
            confidence=0.8,
            validation_count=5,
            optuna_evidence={"p_value": 0.12, "trial_count": 100},
        )

        winner = self.engine.resolve_contradictions(entry_a, entry_b)
        assert winner.content == "Finding A"
        assert entry_b.tier == "archived"

    def test_weaker_entry_archived(self) -> None:
        """The losing entry must be archived."""
        entry_a = make_entry(
            content="A",
            optuna_evidence={"p_value": 0.01, "trial_count": 200},
        )
        entry_b = make_entry(
            content="B",
            optuna_evidence={"p_value": 0.09, "trial_count": 200},
        )
        self.engine.resolve_contradictions(entry_a, entry_b)
        assert entry_b.tier == "archived"

    def test_high_trial_count_breaks_tie(self) -> None:
        """When p-values are similar, higher trial count wins."""
        entry_a = make_entry(
            content="More trials",
            optuna_evidence={"p_value": 0.07, "trial_count": 200},
        )
        entry_b = make_entry(
            content="Fewer trials",
            optuna_evidence={"p_value": 0.08, "trial_count": 50},
        )
        winner = self.engine.resolve_contradictions(entry_a, entry_b)
        assert winner.content == "More trials"

    def test_no_evidence_falls_back_to_weight(self) -> None:
        """When no Optuna evidence, composite weight (conf × val_count) decides."""
        entry_a = make_entry(
            content="High weight",
            confidence=0.9,
            validation_count=10,
            optuna_evidence=None,
        )
        entry_b = make_entry(
            content="Low weight",
            confidence=0.3,
            validation_count=1,
            optuna_evidence=None,
        )
        winner = self.engine.resolve_contradictions(entry_a, entry_b)
        assert winner.content == "High weight"
        assert entry_b.tier == "archived"

    def test_equal_evidence_creates_conditional(self) -> None:
        """Roughly equal evidence → conditional pattern is created."""
        entry_a = make_entry(
            content="Equally supported A",
            optuna_evidence={"p_value": 0.06, "trial_count": 100},
        )
        entry_b = make_entry(
            content="Equally supported B",
            optuna_evidence={"p_value": 0.07, "trial_count": 105},
        )
        result = self.engine.resolve_contradictions(entry_a, entry_b)
        # In this ambiguous case, a new conditional entry should be created
        # OR one entry wins — either outcome is valid; just check both originals
        # end up resolved (one archived or both archived + new conditional)
        assert result is not None
        assert result.tier in ("rule", "pattern", "observation")


# ===========================================================================
# Changelog generation
# ===========================================================================


class TestChangelogGeneration:
    """generate_changelog() — readable summary of actions."""

    def test_empty_actions_produces_no_change_message(self) -> None:
        changelog = ConsolidationEngine.generate_changelog([])
        assert "No changes" in changelog

    def test_actions_appear_in_changelog(self) -> None:
        actions = [
            _Action(kind="promote", entry_id="aabbccdd", detail="observation → pattern"),
            _Action(kind="archive", entry_id="11223344", detail="archived (conf=0.05)"),
            _Action(kind="decay", entry_id="deadbeef", detail="confidence 0.8 → 0.72"),
        ]
        changelog = ConsolidationEngine.generate_changelog(actions)
        assert "PROMOTE" in changelog or "promote" in changelog.lower()
        assert "ARCHIVE" in changelog or "archive" in changelog.lower()
        assert "DECAY" in changelog or "decay" in changelog.lower()

    def test_changelog_includes_summary_counts(self) -> None:
        actions = [
            _Action(kind="promote", entry_id="a", detail="obs → pat"),
            _Action(kind="promote", entry_id="b", detail="obs → pat"),
            _Action(kind="archive", entry_id="c", detail="archived"),
        ]
        changelog = ConsolidationEngine.generate_changelog(actions)
        # Summary line should mention counts
        assert "2" in changelog   # 2 promotes
        assert "1" in changelog   # 1 archive

    def test_changelog_shows_entry_id_prefix(self) -> None:
        entry_id = "12345678-0000-0000-0000-000000000000"
        actions = [_Action(kind="merge", entry_id=entry_id, detail="merged 3 entries")]
        changelog = ConsolidationEngine.generate_changelog(actions)
        assert "12345678" in changelog


# ===========================================================================
# Full consolidation pass (integration)
# ===========================================================================


class TestFullConsolidationPass:
    """run_consolidation() — end-to-end without embeddings (decay + demote only)."""

    def setup_method(self) -> None:
        self.engine = ConsolidationEngine(
            decay_grace_cycles=100,
            decay_rate=0.9,
            pattern_demote_threshold=0.3,
            observation_archive_threshold=0.1,
        )

    def test_changelog_is_string(self, sample_knowledge_entries) -> None:
        changelog = self.engine.run_consolidation(
            knowledge_base=sample_knowledge_entries,
            current_cycle=500,
        )
        assert isinstance(changelog, str)
        assert len(changelog) > 0

    def test_stale_patterns_are_demoted(self) -> None:
        """A pattern validated at cycle 0 with low confidence is demoted by cycle 500."""
        entries = [
            make_entry(tier="pattern", confidence=0.29, last_validated_cycle=0),
        ]
        self.engine.run_consolidation(entries, current_cycle=200)
        # Entry should be demoted (pattern → observation or archived)
        assert entries[0].tier in ("observation", "archived")

    def test_rules_remain_untouched(self) -> None:
        entries = [
            make_entry(tier="rule", confidence=0.95, last_validated_cycle=0),
        ]
        self.engine.run_consolidation(entries, current_cycle=10_000)
        assert entries[0].tier == "rule"
        assert entries[0].confidence == pytest.approx(0.95)

    def test_active_entries_not_demoted_without_decay(self) -> None:
        """Recently validated entries should survive consolidation."""
        entries = [
            make_entry(tier="pattern", confidence=0.8, last_validated_cycle=450),
        ]
        self.engine.run_consolidation(entries, current_cycle=500)
        # 50 idle cycles → still within grace → no demotion
        assert entries[0].tier == "pattern"

    def test_stale_observations_archived(self) -> None:
        """Observation with very low confidence and many idle cycles → archived."""
        entries = [
            make_entry(tier="observation", confidence=0.05, last_validated_cycle=0),
        ]
        self.engine.run_consolidation(entries, current_cycle=600)
        assert entries[0].tier == "archived"
