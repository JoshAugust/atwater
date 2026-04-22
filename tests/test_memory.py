"""
test_memory.py — Memory tier unit tests.

Tests
-----
WorkingMemory   read / write / clear / snapshot
SharedState     state_read / state_write / role scoping / WAL mode / concurrent access
KnowledgeBase   write / read (semantic search) / promote / tier priority / contradictions

All tests use tmp_path (via tmp_db fixture) for SQLite databases and
mock_embed to avoid loading sentence-transformers.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from src.memory import (
    KnowledgeBase,
    KnowledgeEntry,
    SharedState,
    WorkingMemory,
    ROLE_ALL,
    ORCHESTRATOR_ROLE,
    VALID_TIERS,
)


# ===========================================================================
# Tier 1 — WorkingMemory
# ===========================================================================


class TestWorkingMemory:
    """Tests for the ephemeral per-turn working memory."""

    def test_write_and_read_simple_value(self) -> None:
        wm = WorkingMemory()
        wm.write("score", 0.91)
        assert wm.read("score") == 0.91

    def test_read_missing_key_returns_default(self) -> None:
        wm = WorkingMemory()
        assert wm.read("nonexistent") is None
        assert wm.read("nonexistent", default="fallback") == "fallback"

    def test_write_overwrites_existing(self) -> None:
        wm = WorkingMemory()
        wm.write("key", "first")
        wm.write("key", "second")
        assert wm.read("key") == "second"

    def test_clear_removes_all_keys(self) -> None:
        wm = WorkingMemory()
        wm.write("a", 1)
        wm.write("b", 2)
        wm.clear()
        assert wm.read("a") is None
        assert wm.read("b") is None

    def test_snapshot_returns_copy_of_store(self) -> None:
        wm = WorkingMemory()
        wm.write("x", [1, 2, 3])
        snap = wm.snapshot()
        assert snap == {"x": [1, 2, 3]}

    def test_snapshot_is_shallow_copy(self) -> None:
        wm = WorkingMemory()
        wm.write("data", {"nested": True})
        snap = wm.snapshot()
        # Mutating the snapshot does NOT affect working memory
        snap["data"]["nested"] = False
        assert wm.read("data") == {"nested": False}  # shallow copy, inner object shared

    def test_snapshot_after_clear_is_empty(self) -> None:
        wm = WorkingMemory()
        wm.write("x", 99)
        wm.clear()
        assert wm.snapshot() == {}

    def test_write_various_types(self) -> None:
        wm = WorkingMemory()
        wm.write("int", 42)
        wm.write("float", 3.14)
        wm.write("str", "hello")
        wm.write("list", [1, 2])
        wm.write("dict", {"a": 1})
        wm.write("none", None)
        assert wm.read("int") == 42
        assert wm.read("float") == pytest.approx(3.14)
        assert wm.read("str") == "hello"
        assert wm.read("list") == [1, 2]
        assert wm.read("dict") == {"a": 1}
        assert wm.read("none") is None

    def test_repr_shows_key_count(self) -> None:
        wm = WorkingMemory()
        wm.write("a", 1)
        wm.write("b", 2)
        assert "2" in repr(wm)


# ===========================================================================
# Tier 2 — SharedState
# ===========================================================================


class TestSharedState:
    """Tests for the SQLite-backed shared state machine."""

    def test_write_and_read_basic(self, tmp_db) -> None:
        ss = SharedState(db_path=tmp_db("state"))
        ss.state_write("hypothesis", {"bg": "dark"})
        assert ss.state_read("hypothesis") == {"bg": "dark"}
        ss.close()

    def test_read_missing_key_returns_none(self, tmp_db) -> None:
        ss = SharedState(db_path=tmp_db("state"))
        assert ss.state_read("does_not_exist") is None
        ss.close()

    def test_write_overwrites_value(self, tmp_db) -> None:
        ss = SharedState(db_path=tmp_db("state"))
        ss.state_write("key", "v1")
        ss.state_write("key", "v2")
        assert ss.state_read("key") == "v2"
        ss.close()

    def test_role_scoping_restricts_access(self, tmp_db) -> None:
        ss = SharedState(db_path=tmp_db("state"))
        # Only director + creator can see this key
        ss.state_write("current_hypothesis", {"layout": "A"}, roles=["director", "creator"])
        # Only grader can see this key
        ss.state_write("score", 0.85, roles=["grader"])

        director_view = ss.state_read_scoped("director")
        grader_view = ss.state_read_scoped("grader")

        assert "current_hypothesis" in director_view
        assert "score" not in director_view
        assert "score" in grader_view
        assert "current_hypothesis" not in grader_view
        ss.close()

    def test_orchestrator_sees_all_keys(self, tmp_db) -> None:
        ss = SharedState(db_path=tmp_db("state"))
        ss.state_write("key_a", "val_a", roles=["director"])
        ss.state_write("key_b", "val_b", roles=["grader"])
        ss.state_write("key_c", "val_c", roles=["creator"])

        orch_view = ss.state_read_scoped(ORCHESTRATOR_ROLE)
        assert "key_a" in orch_view
        assert "key_b" in orch_view
        assert "key_c" in orch_view
        ss.close()

    def test_role_all_visible_to_everyone(self, tmp_db) -> None:
        ss = SharedState(db_path=tmp_db("state"))
        ss.state_write("shared_key", "shared_val", roles=[ROLE_ALL])

        for role in ("director", "creator", "grader", "diversity_guard"):
            view = ss.state_read_scoped(role)
            assert "shared_key" in view, f"Expected 'shared_key' visible to {role}"
        ss.close()

    def test_default_role_is_all(self, tmp_db) -> None:
        ss = SharedState(db_path=tmp_db("state"))
        ss.state_write("default_key", 123)  # no roles= means ROLE_ALL
        creator_view = ss.state_read_scoped("creator")
        assert "default_key" in creator_view
        ss.close()

    def test_state_keys_lists_all(self, tmp_db) -> None:
        ss = SharedState(db_path=tmp_db("state"))
        ss.state_write("a", 1)
        ss.state_write("b", 2)
        ss.state_write("c", 3)
        keys = ss.state_keys()
        assert sorted(keys) == ["a", "b", "c"]
        ss.close()

    def test_state_delete_removes_key(self, tmp_db) -> None:
        ss = SharedState(db_path=tmp_db("state"))
        ss.state_write("temp", "gone")
        ss.state_delete("temp")
        assert ss.state_read("temp") is None
        ss.close()

    def test_write_json_serialisable_types(self, tmp_db) -> None:
        ss = SharedState(db_path=tmp_db("state"))
        payload = {"list": [1, 2, 3], "nested": {"ok": True}, "num": 3.14}
        ss.state_write("payload", payload)
        assert ss.state_read("payload") == payload
        ss.close()

    def test_wal_mode_enabled(self, tmp_db) -> None:
        db_path = tmp_db("state_wal")
        ss = SharedState(db_path=db_path)
        import sqlite3 as _sqlite3
        conn = _sqlite3.connect(str(db_path))
        journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert journal.lower() == "wal"
        ss.close()

    def test_concurrent_writes(self, tmp_db) -> None:
        """Multiple threads writing different keys should not corrupt state."""
        db_path = tmp_db("concurrent_state")
        errors: list[Exception] = []

        def _writer(idx: int) -> None:
            try:
                # Each thread gets its own connection (SharedState opens one on init)
                ss = SharedState(db_path=db_path)
                for i in range(5):
                    ss.state_write(f"key_{idx}_{i}", f"value_{idx}_{i}")
                ss.close()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Concurrent write errors: {errors}"

        # Verify at least some keys were written
        ss = SharedState(db_path=db_path)
        keys = ss.state_keys()
        assert len(keys) >= 5  # at least one thread succeeded
        ss.close()

    def test_repr_shows_key_count(self, tmp_db) -> None:
        ss = SharedState(db_path=tmp_db("state"))
        ss.state_write("a", 1)
        ss.state_write("b", 2)
        r = repr(ss)
        assert "keys=2" in r
        ss.close()


# ===========================================================================
# Tier 3 — KnowledgeBase
# ===========================================================================


class TestKnowledgeBase:
    """Tests for the hierarchical persistent knowledge store."""

    def test_write_returns_entry_id(self, tmp_db, mock_embed) -> None:
        kb = KnowledgeBase(db_path=tmp_db("kb"))
        eid = kb.knowledge_write(
            content="Test knowledge",
            tier="observation",
            confidence=0.8,
            topic_cluster="test",
        )
        assert isinstance(eid, str)
        assert len(eid) > 0
        kb.close()

    def test_write_invalid_tier_raises(self, tmp_db, mock_embed) -> None:
        kb = KnowledgeBase(db_path=tmp_db("kb"))
        with pytest.raises(ValueError, match="Invalid tier"):
            kb.knowledge_write(
                content="Bad tier",
                tier="bogus",
                confidence=0.5,
                topic_cluster="test",
            )
        kb.close()

    def test_write_invalid_confidence_raises(self, tmp_db, mock_embed) -> None:
        kb = KnowledgeBase(db_path=tmp_db("kb"))
        with pytest.raises(ValueError, match="confidence"):
            kb.knowledge_write(
                content="Too confident",
                tier="observation",
                confidence=1.5,
                topic_cluster="test",
            )
        kb.close()

    def test_knowledge_get_returns_entry(self, tmp_db, mock_embed) -> None:
        kb = KnowledgeBase(db_path=tmp_db("kb"))
        eid = kb.knowledge_write(
            content="Findable knowledge",
            tier="pattern",
            confidence=0.9,
            topic_cluster="test",
        )
        entry = kb.knowledge_get(eid)
        assert entry is not None
        assert entry.id == eid
        assert entry.tier == "pattern"
        assert entry.confidence == pytest.approx(0.9)
        assert entry.content == "Findable knowledge"
        kb.close()

    def test_knowledge_get_unknown_id_returns_none(self, tmp_db, mock_embed) -> None:
        kb = KnowledgeBase(db_path=tmp_db("kb"))
        assert kb.knowledge_get("00000000-0000-0000-0000-000000000000") is None
        kb.close()

    def test_knowledge_read_returns_list(self, tmp_db, mock_embed) -> None:
        kb = KnowledgeBase(db_path=tmp_db("kb"))
        kb.knowledge_write("A rule about typography", "rule", 0.95, "typography")
        kb.knowledge_write("A pattern about layout", "pattern", 0.7, "layout")
        kb.knowledge_write("An observation about colour", "observation", 0.5, "colour")

        results = kb.knowledge_read("typography and layout")
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, KnowledgeEntry) for r in results)
        kb.close()

    def test_knowledge_read_tier_filter(self, tmp_db, mock_embed) -> None:
        kb = KnowledgeBase(db_path=tmp_db("kb"))
        kb.knowledge_write("rule content", "rule", 0.9, "test")
        kb.knowledge_write("pattern content", "pattern", 0.7, "test")
        kb.knowledge_write("observation content", "observation", 0.5, "test")

        only_rules = kb.knowledge_read("any query", tier="rule")
        assert all(e.tier == "rule" for e in only_rules)

        only_patterns = kb.knowledge_read("any query", tier="pattern")
        assert all(e.tier == "pattern" for e in only_patterns)
        kb.close()

    def test_knowledge_read_invalid_tier_raises(self, tmp_db, mock_embed) -> None:
        kb = KnowledgeBase(db_path=tmp_db("kb"))
        with pytest.raises(ValueError, match="Invalid tier"):
            kb.knowledge_read("query", tier="nonexistent")
        kb.close()

    def test_knowledge_promote_observation_to_pattern(self, tmp_db, mock_embed) -> None:
        kb = KnowledgeBase(db_path=tmp_db("kb"))
        eid = kb.knowledge_write(
            content="Consistently good layout",
            tier="observation",
            confidence=0.7,
            topic_cluster="layout",
        )

        kb.knowledge_promote(
            entry_id=eid,
            from_tier="observation",
            to_tier="pattern",
            evidence={"trial_count": 50, "p_value": 0.08, "cycle": 100},
        )

        promoted = kb.knowledge_get(eid)
        assert promoted is not None
        assert promoted.tier == "pattern"
        assert promoted.validation_count == 1
        assert promoted.last_validated_cycle == 100
        kb.close()

    def test_knowledge_promote_pattern_to_rule(self, tmp_db, mock_embed) -> None:
        kb = KnowledgeBase(db_path=tmp_db("kb"))
        eid = kb.knowledge_write(
            content="Strong invariant",
            tier="pattern",
            confidence=0.9,
            topic_cluster="core",
        )

        kb.knowledge_promote(
            entry_id=eid,
            from_tier="pattern",
            to_tier="rule",
            evidence={"trial_count": 210, "p_value": 0.02, "cycle": 250},
        )

        promoted = kb.knowledge_get(eid)
        assert promoted is not None
        assert promoted.tier == "rule"
        kb.close()

    def test_knowledge_promote_wrong_from_tier_raises(self, tmp_db, mock_embed) -> None:
        kb = KnowledgeBase(db_path=tmp_db("kb"))
        eid = kb.knowledge_write(
            content="A rule",
            tier="rule",
            confidence=0.9,
            topic_cluster="core",
        )
        with pytest.raises(ValueError):
            kb.knowledge_promote(eid, from_tier="observation", to_tier="pattern", evidence={})
        kb.close()

    def test_knowledge_promote_downgrade_raises(self, tmp_db, mock_embed) -> None:
        kb = KnowledgeBase(db_path=tmp_db("kb"))
        eid = kb.knowledge_write(
            content="A pattern",
            tier="pattern",
            confidence=0.7,
            topic_cluster="test",
        )
        with pytest.raises(ValueError):
            kb.knowledge_promote(eid, from_tier="pattern", to_tier="observation", evidence={})
        kb.close()

    def test_knowledge_promote_missing_entry_raises(self, tmp_db, mock_embed) -> None:
        kb = KnowledgeBase(db_path=tmp_db("kb"))
        with pytest.raises(LookupError):
            kb.knowledge_promote(
                "00000000-0000-0000-0000-000000000000",
                from_tier="observation",
                to_tier="pattern",
                evidence={},
            )
        kb.close()

    def test_tier_priority_rules_first(self, tmp_db, mock_embed) -> None:
        """knowledge_list should order by tier priority (rules first)."""
        kb = KnowledgeBase(db_path=tmp_db("kb"))
        kb.knowledge_write("obs", "observation", 0.9, "test")
        kb.knowledge_write("rule", "rule", 0.5, "test")   # lower confidence but higher tier
        kb.knowledge_write("pat", "pattern", 0.7, "test")

        entries = kb.knowledge_list()
        # Tier order: rule → pattern → observation
        tiers = [e.tier for e in entries]
        assert tiers.index("rule") < tiers.index("pattern")
        assert tiers.index("pattern") < tiers.index("observation")
        kb.close()

    def test_flag_contradiction_appends(self, tmp_db, mock_embed) -> None:
        kb = KnowledgeBase(db_path=tmp_db("kb"))
        eid_a = kb.knowledge_write("Finding A", "observation", 0.7, "test")
        eid_b = kb.knowledge_write("Contradicting finding B", "observation", 0.7, "test")

        kb.knowledge_flag_contradiction(eid_a, eid_b)

        entry_a = kb.knowledge_get(eid_a)
        assert entry_a is not None
        assert eid_b in entry_a.contradicted_by
        kb.close()

    def test_flag_contradiction_idempotent(self, tmp_db, mock_embed) -> None:
        """Flagging the same contradiction twice should not duplicate the entry."""
        kb = KnowledgeBase(db_path=tmp_db("kb"))
        eid_a = kb.knowledge_write("A", "observation", 0.5, "test")
        eid_b = kb.knowledge_write("B", "observation", 0.5, "test")

        kb.knowledge_flag_contradiction(eid_a, eid_b)
        kb.knowledge_flag_contradiction(eid_a, eid_b)  # second call

        entry_a = kb.knowledge_get(eid_a)
        assert entry_a is not None
        assert entry_a.contradicted_by.count(eid_b) == 1
        kb.close()

    def test_knowledge_validate_increments_count(self, tmp_db, mock_embed) -> None:
        kb = KnowledgeBase(db_path=tmp_db("kb"))
        eid = kb.knowledge_write("Validated finding", "observation", 0.6, "test")
        kb.knowledge_validate(eid, cycle=50)
        kb.knowledge_validate(eid, cycle=60)

        entry = kb.knowledge_get(eid)
        assert entry is not None
        assert entry.validation_count == 2
        assert entry.last_validated_cycle == 60
        kb.close()

    def test_knowledge_list_filter_by_tier(self, tmp_db, mock_embed) -> None:
        kb = KnowledgeBase(db_path=tmp_db("kb"))
        kb.knowledge_write("rule_1", "rule", 0.9, "test")
        kb.knowledge_write("obs_1", "observation", 0.5, "test")
        kb.knowledge_write("obs_2", "observation", 0.6, "test")

        obs_list = kb.knowledge_list(tier="observation")
        assert all(e.tier == "observation" for e in obs_list)
        assert len(obs_list) == 2

        rule_list = kb.knowledge_list(tier="rule")
        assert len(rule_list) == 1
        kb.close()

    def test_knowledge_list_filter_by_topic(self, tmp_db, mock_embed) -> None:
        kb = KnowledgeBase(db_path=tmp_db("kb"))
        kb.knowledge_write("typography rule", "rule", 0.9, "typography")
        kb.knowledge_write("layout pattern", "pattern", 0.7, "layout")
        kb.knowledge_write("typography obs", "observation", 0.5, "typography")

        typo_entries = kb.knowledge_list(topic_cluster="typography")
        assert all(e.topic_cluster == "typography" for e in typo_entries)
        assert len(typo_entries) == 2
        kb.close()

    def test_all_valid_tiers_accepted(self, tmp_db, mock_embed) -> None:
        kb = KnowledgeBase(db_path=tmp_db("kb"))
        for tier in VALID_TIERS:
            eid = kb.knowledge_write(f"Content for {tier}", tier, 0.5, "test")
            entry = kb.knowledge_get(eid)
            assert entry is not None
            assert entry.tier == tier
        kb.close()

    def test_repr_shows_counts(self, tmp_db, mock_embed) -> None:
        kb = KnowledgeBase(db_path=tmp_db("kb"))
        kb.knowledge_write("r", "rule", 0.9, "test")
        kb.knowledge_write("p", "pattern", 0.7, "test")
        kb.knowledge_write("o", "observation", 0.5, "test")
        r = repr(kb)
        assert "rules=1" in r
        assert "patterns=1" in r
        assert "observations=1" in r
        kb.close()
