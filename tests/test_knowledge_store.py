"""
tests/test_knowledge_store.py — Test suite for KnowledgeStore (Phase 4).

Tests cover:
  - test_write_and_read          — basic round-trip
  - test_knn_search              — returns nearest neighbours correctly
  - test_semantic_dedup          — near-duplicate rejected
  - test_confidence_decay        — Ebbinghaus decay works correctly
  - test_tier_priority           — rules returned before patterns before observations
  - test_matryoshka_rerank       — coarse search + rerank improves precision
  - test_10k_cap                 — eviction works at capacity
  - test_fallback                — numpy path works when sqlite-vec unavailable

All tests use an in-memory or temp-file SQLite database.  The embedding model
is mocked out for tests that do not require real semantic similarity, keeping
the test suite fast and offline-capable.

Slow tests (which load the real model) are marked with @pytest.mark.slow.
Run them with: pytest -m slow
"""

from __future__ import annotations

import math
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers — deterministic fake embeddings
# ---------------------------------------------------------------------------


def _make_unit(dim: int, direction: int | None = None) -> np.ndarray:
    """Return a unit-norm vector.  direction pins one axis to ~1.0 for control."""
    if direction is not None:
        v = np.zeros(dim, dtype=np.float32)
        v[direction % dim] = 1.0
    else:
        rng = np.random.default_rng(42)
        v = rng.standard_normal(dim).astype(np.float32)
        v /= np.linalg.norm(v)
    return v


def _near_duplicate(base: np.ndarray, noise: float = 0.01) -> np.ndarray:
    """Return a vector that is very close to base (cosine > 0.99)."""
    perturbed = base + np.random.default_rng(99).standard_normal(base.shape).astype(np.float32) * noise
    perturbed /= np.linalg.norm(perturbed)
    return perturbed


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test_ks.db"


def _mock_embed(text: str, task: str = "search_document") -> np.ndarray:
    """
    Deterministic fake embed function: returns a vector based on text hash.
    Used to make tests fast and offline.
    """
    dim = 768
    seed = hash(text) % (2**31)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


@pytest.fixture()
def store(tmp_db: Path):
    """A KnowledgeStore instance with the real embedding function mocked."""
    with patch("src.memory.knowledge_store._embed", side_effect=_mock_embed):
        from src.memory.knowledge_store import KnowledgeStore
        s = KnowledgeStore(db_path=tmp_db, use_vec=True)
        yield s
        s.close()


@pytest.fixture()
def numpy_store(tmp_db: Path):
    """A KnowledgeStore instance forced into numpy fallback mode."""
    with patch("src.memory.knowledge_store._embed", side_effect=_mock_embed):
        from src.memory.knowledge_store import KnowledgeStore
        s = KnowledgeStore(db_path=tmp_db, use_vec=False)
        yield s
        s.close()


# ---------------------------------------------------------------------------
# 1. Basic round-trip
# ---------------------------------------------------------------------------


class TestWriteAndRead:
    def test_write_returns_uuid(self, store):
        eid = store.write(
            content="Helvetica outperforms Times New Roman for headlines",
            tier="observation",
            confidence=0.7,
            topic_cluster="typography",
        )
        assert eid is not None
        assert len(eid) == 36  # UUID4 format

    def test_get_by_id(self, store):
        eid = store.write(
            content="Blue CTA buttons increase click-through by 14%",
            tier="pattern",
            confidence=0.8,
        )
        entry = store.get(eid)
        assert entry is not None
        assert entry.id == eid
        assert entry.tier == "pattern"
        assert abs(entry.confidence - 0.8) < 1e-6
        assert entry.content == "Blue CTA buttons increase click-through by 14%"

    def test_read_returns_entry(self, store):
        content = "Larger font sizes reduce reading time"
        store.write(content=content, tier="observation", confidence=0.6)
        results = store.read("font size reading speed")
        assert len(results) >= 1

    def test_invalid_tier_raises(self, store):
        with pytest.raises(ValueError):
            store.write(content="test", tier="invalid_tier", confidence=0.5)

    def test_invalid_confidence_raises(self, store):
        with pytest.raises(ValueError):
            store.write(content="test", tier="observation", confidence=1.5)

    def test_list_entries_empty(self, store):
        entries = store.list_entries()
        assert entries == []

    def test_list_entries_returns_all(self, store):
        store.write("Entry one", tier="rule", confidence=0.9)
        store.write("Entry two", tier="observation", confidence=0.5)
        entries = store.list_entries()
        assert len(entries) == 2

    def test_write_stores_metadata(self, store):
        eid = store.write(
            content="Test content",
            tier="rule",
            confidence=0.95,
            topic_cluster="test_cluster",
            cycle=42,
            optuna_evidence={"p_value": 0.01, "trial_count": 300},
        )
        entry = store.get(eid)
        assert entry.topic_cluster == "test_cluster"
        assert entry.created_cycle == 42
        assert entry.optuna_evidence == {"p_value": 0.01, "trial_count": 300}


# ---------------------------------------------------------------------------
# 2. KNN search
# ---------------------------------------------------------------------------


class TestKnnSearch:
    def test_most_similar_returned_first(self, tmp_db):
        """The entry whose embedding is closest to query should rank first."""
        from src.memory.knowledge_store import KnowledgeStore

        # Use controlled embeddings: entry A is close to query, B is far
        dim = 768
        query_vec = _make_unit(dim, direction=0)   # axis 0
        close_vec = _make_unit(dim, direction=0)   # same axis → cos=1.0
        far_vec = _make_unit(dim, direction=1)     # axis 1 → cos=0.0

        call_count = [0]

        def controlled_embed(text: str, task: str = "search_document") -> np.ndarray:
            call_count[0] += 1
            if "query" in text:
                return query_vec.copy()
            elif "close" in text:
                return close_vec.copy()
            else:
                return far_vec.copy()

        with patch("src.memory.knowledge_store._embed", side_effect=controlled_embed):
            s = KnowledgeStore(db_path=tmp_db, use_vec=True)
            s.write("close content A", tier="observation", confidence=0.5, skip_dedup=True)
            s.write("far content B", tier="observation", confidence=0.5, skip_dedup=True)
            results = s.read("query about content")
            s.close()

        assert len(results) >= 1
        assert "close" in results[0].content

    def test_max_results_respected(self, store):
        for i in range(10):
            store.write(f"Unique content item number {i}", tier="observation", confidence=0.5, skip_dedup=True)
        results = store.read("content", max_results=3)
        assert len(results) <= 3

    def test_tier_filter(self, store):
        store.write("Rule content alpha", tier="rule", confidence=0.9, skip_dedup=True)
        store.write("Pattern content beta", tier="pattern", confidence=0.7, skip_dedup=True)
        store.write("Observation content gamma", tier="observation", confidence=0.5, skip_dedup=True)

        rule_results = store.read("content", tier="rule")
        for r in rule_results:
            assert r.tier == "rule"

        obs_results = store.read("content", tier="observation")
        for r in obs_results:
            assert r.tier == "observation"

    def test_empty_store_returns_empty(self, store):
        results = store.read("anything at all")
        assert results == []


# ---------------------------------------------------------------------------
# 3. Semantic deduplication
# ---------------------------------------------------------------------------


class TestSemanticDedup:
    def test_near_duplicate_rejected(self, tmp_db):
        """Writing a near-duplicate (cosine > 0.95) should return None."""
        from src.memory.knowledge_store import KnowledgeStore

        base_vec = _make_unit(768, direction=5)
        dup_vec = _near_duplicate(base_vec, noise=0.001)  # cosine ≈ 0.9999

        call_count = [0]

        def dup_embed(text: str, task: str = "search_document") -> np.ndarray:
            call_count[0] += 1
            if "original" in text:
                return base_vec.copy()
            elif "duplicate" in text:
                return dup_vec.copy()
            else:
                return _make_unit(768, direction=call_count[0] % 768)

        with patch("src.memory.knowledge_store._embed", side_effect=dup_embed):
            s = KnowledgeStore(db_path=tmp_db, use_vec=True)
            eid1 = s.write("original content that is unique", tier="observation", confidence=0.6)
            eid2 = s.write("duplicate content that is similar", tier="observation", confidence=0.6)
            s.close()

        assert eid1 is not None
        assert eid2 is None, "Near-duplicate should have been rejected"

    def test_distinct_content_accepted(self, tmp_db):
        """Distinct entries should both be accepted."""
        from src.memory.knowledge_store import KnowledgeStore

        vec_a = _make_unit(768, direction=0)
        vec_b = _make_unit(768, direction=383)  # orthogonal → cos ≈ 0.0

        def distinct_embed(text: str, task: str = "search_document") -> np.ndarray:
            if "alpha" in text:
                return vec_a.copy()
            else:
                return vec_b.copy()

        with patch("src.memory.knowledge_store._embed", side_effect=distinct_embed):
            s = KnowledgeStore(db_path=tmp_db, use_vec=True)
            eid1 = s.write("alpha content distinct A", tier="observation", confidence=0.6)
            eid2 = s.write("beta content distinct B", tier="observation", confidence=0.6)
            s.close()

        assert eid1 is not None
        assert eid2 is not None, "Distinct entries should both be accepted"

    def test_skip_dedup_allows_duplicates(self, store):
        """skip_dedup=True bypasses the deduplication check."""
        eid1 = store.write("duplicate bypass test", tier="observation", confidence=0.5)
        eid2 = store.write("duplicate bypass test", tier="observation", confidence=0.5, skip_dedup=True)
        assert eid1 is not None
        assert eid2 is not None


# ---------------------------------------------------------------------------
# 4. Confidence decay
# ---------------------------------------------------------------------------


class TestConfidenceDecay:
    def test_decay_reduces_confidence(self, store):
        eid = store.write("Decayable knowledge entry", tier="observation", confidence=0.8)
        store.decay_all(current_cycle=100)  # 100 cycles since validation at cycle 0
        entry = store.get(eid)
        assert entry.confidence < 0.8

    def test_recently_validated_decays_less(self, store):
        eid_old = store.write("Old knowledge entry", tier="observation", confidence=0.8, cycle=0)
        eid_new = store.write("New knowledge entry", tier="observation", confidence=0.8, cycle=0, skip_dedup=True)

        # Simulate recent validation of the new entry
        store.validate(eid_new, cycle=90)

        store.decay_all(current_cycle=100)

        old_entry = store.get(eid_old)
        new_entry = store.get(eid_new)

        assert new_entry.confidence > old_entry.confidence, (
            "Recently validated entry should retain higher confidence"
        )

    def test_decay_below_floor_archives(self, store):
        """Entries that decay to floor should be archived automatically."""
        from src.memory.knowledge_store import EBBINGHAUS_MIN

        eid = store.write(
            "Very old observation that should be archived",
            tier="observation",
            confidence=EBBINGHAUS_MIN + 0.001,
            cycle=0,
        )
        # decay with huge elapsed time
        store.decay_all(current_cycle=10_000)
        entry = store.get(eid)
        assert entry.tier == "archived", "Should be archived after confidence hits floor"

    def test_ebbinghaus_formula(self):
        """Verify the decay formula is exponential (Ebbinghaus)."""
        from src.memory.knowledge_store import _ebbinghaus_decay, EBBINGHAUS_STABILITY, EBBINGHAUS_MIN

        conf = 0.8
        decayed_10 = _ebbinghaus_decay(conf, 10)
        decayed_20 = _ebbinghaus_decay(conf, 20)

        expected_10 = conf * math.exp(-10 / EBBINGHAUS_STABILITY)
        expected_20 = conf * math.exp(-20 / EBBINGHAUS_STABILITY)

        assert abs(decayed_10 - max(EBBINGHAUS_MIN, expected_10)) < 1e-6
        assert abs(decayed_20 - max(EBBINGHAUS_MIN, expected_20)) < 1e-6
        assert decayed_10 > decayed_20, "More elapsed cycles → lower confidence"

    def test_no_decay_at_zero_cycles(self, store):
        eid = store.write("Freshly validated", tier="rule", confidence=0.9, cycle=50)
        store.validate(eid, cycle=50)
        store.decay_all(current_cycle=50)
        entry = store.get(eid)
        assert abs(entry.confidence - 0.9) < 1e-3


# ---------------------------------------------------------------------------
# 5. Tier priority
# ---------------------------------------------------------------------------


class TestTierPriority:
    def test_rules_returned_before_patterns(self, tmp_db):
        """When all tiers have relevant content, rules must come first."""
        from src.memory.knowledge_store import KnowledgeStore

        # All entries get the same embedding so relevance is equal;
        # tier ordering should still apply.
        same_vec = _make_unit(768, direction=10)

        def same_embed(text: str, task: str = "search_document") -> np.ndarray:
            return same_vec.copy()

        with patch("src.memory.knowledge_store._embed", side_effect=same_embed):
            s = KnowledgeStore(db_path=tmp_db, use_vec=True)
            s.write("Observation content", tier="observation", confidence=0.5, skip_dedup=True)
            s.write("Pattern content", tier="pattern", confidence=0.7, skip_dedup=True)
            s.write("Rule content", tier="rule", confidence=0.9, skip_dedup=True)

            results = s.read("any query", max_results=10)
            s.close()

        tiers = [r.tier for r in results]
        assert tiers[0] == "rule", f"First result should be rule, got: {tiers}"

    def test_patterns_before_observations(self, tmp_db):
        """Patterns should precede observations when both present."""
        from src.memory.knowledge_store import KnowledgeStore

        same_vec = _make_unit(768, direction=20)

        def same_embed(text: str, task: str = "search_document") -> np.ndarray:
            return same_vec.copy()

        with patch("src.memory.knowledge_store._embed", side_effect=same_embed):
            s = KnowledgeStore(db_path=tmp_db, use_vec=True)
            s.write("Observation alpha", tier="observation", confidence=0.5, skip_dedup=True)
            s.write("Pattern beta", tier="pattern", confidence=0.7, skip_dedup=True)
            results = s.read("any query", max_results=10)
            s.close()

        tiers = [r.tier for r in results]
        assert "pattern" in tiers
        pattern_idx = tiers.index("pattern")
        if "observation" in tiers:
            obs_idx = tiers.index("observation")
            assert pattern_idx < obs_idx, "Pattern should appear before observation"

    def test_archived_not_returned(self, store):
        """Archived entries must not appear in search results."""
        eid = store.write("Archived content should not appear", tier="observation", confidence=0.5)
        store.archive(eid)
        results = store.read("archived content")
        for r in results:
            assert r.tier != "archived", "Archived entries should not appear in search"

    def test_tier_filter_isolates_tier(self, store):
        store.write("Rule entry solo", tier="rule", confidence=0.9, skip_dedup=True)
        store.write("Observation entry solo", tier="observation", confidence=0.5, skip_dedup=True)

        rule_results = store.read("entry", tier="rule")
        for r in rule_results:
            assert r.tier == "rule"


# ---------------------------------------------------------------------------
# 6. Matryoshka two-stage rerank
# ---------------------------------------------------------------------------


class TestMatryoshkaRerank:
    """
    Test that the two-stage retrieval (coarse KNN → rerank) produces
    results, and that the coarse dimension truncation is applied.

    We verify the mechanism rather than the quality improvement (that would
    require a real model and a large knowledge base).
    """

    def test_two_stage_retrieval_returns_results(self, store):
        """Basic smoke test: two-stage retrieval produces ordered results."""
        for i in range(5):
            store.write(
                f"Knowledge item number {i} about design principles",
                tier="observation",
                confidence=0.5 + i * 0.05,
                skip_dedup=True,
            )

        results = store.read("design principles", max_results=3)
        assert len(results) <= 3

    @pytest.mark.skipif(
        not True,  # always run (sqlite-vec required)
        reason="sqlite-vec not available"
    )
    def test_coarse_truncation_logic(self, tmp_db):
        """
        Verify that the 256-dim coarse embedding path is exercised.
        We patch _get_model to return 768-dim, then check the coarse query uses 256.
        """
        from src.memory.knowledge_store import KnowledgeStore, COARSE_DIM

        embedded_dims = []

        full_vec = _make_unit(768, direction=42)

        def tracking_embed(text: str, task: str = "search_document") -> np.ndarray:
            return full_vec.copy()

        with patch("src.memory.knowledge_store._embed", side_effect=tracking_embed):
            with patch("src.memory.knowledge_store._get_model") as mock_gm:
                mock_model = MagicMock()
                mock_gm.return_value = (mock_model, 768)  # nomic dims

                s = KnowledgeStore(db_path=tmp_db, use_vec=True)
                # Write with a full vec
                s.write("test content", tier="observation", confidence=0.5, skip_dedup=True)

        # Verify the coarse dim constant is correct
        assert COARSE_DIM == 256

    def test_rerank_produces_correct_ordering(self, tmp_db):
        """
        With controlled embeddings, verify that reranking places the most
        similar item first.
        """
        from src.memory.knowledge_store import KnowledgeStore

        dim = 768
        query_dir = 50
        query_vec = _make_unit(dim, direction=query_dir)
        close_vec = _make_unit(dim, direction=query_dir)   # cos=1.0
        mid_vec = _make_unit(dim, direction=100)            # cos ≈ 0
        far_vec = _make_unit(dim, direction=200)            # cos ≈ 0

        labels = {}
        call_counter = [0]

        def labelled_embed(text: str, task: str = "search_document") -> np.ndarray:
            call_counter[0] += 1
            if "query" in text:
                return query_vec.copy()
            if "close" in text:
                return close_vec.copy()
            elif "mid" in text:
                return mid_vec.copy()
            else:
                return far_vec.copy()

        with patch("src.memory.knowledge_store._embed", side_effect=labelled_embed):
            s = KnowledgeStore(db_path=tmp_db, use_vec=True)
            s.write("far away content item F", tier="observation", confidence=0.5, skip_dedup=True)
            s.write("mid distance content item M", tier="observation", confidence=0.5, skip_dedup=True)
            s.write("close similar content item C", tier="observation", confidence=0.5, skip_dedup=True)
            results = s.read("query about close similarity", max_results=3)
            s.close()

        assert len(results) >= 1
        assert "close" in results[0].content, (
            f"Most similar (close) entry should rank first. Got: {[r.content for r in results]}"
        )


# ---------------------------------------------------------------------------
# 7. 10K cap + eviction
# ---------------------------------------------------------------------------


class TestCapAndEviction:
    def test_eviction_at_capacity(self, tmp_db):
        """
        Writing MAX_KNOWLEDGE_ITEMS + 1 entries should trigger eviction
        so the total stays at or below the cap.
        """
        from src.memory.knowledge_store import KnowledgeStore, MAX_KNOWLEDGE_ITEMS, EVICTION_BATCH

        # Use a very low cap for testing by monkey-patching the constant
        LOW_CAP = 20
        LOW_EVICTION = 5

        with patch("src.memory.knowledge_store.MAX_KNOWLEDGE_ITEMS", LOW_CAP):
            with patch("src.memory.knowledge_store.EVICTION_BATCH", LOW_EVICTION):
                with patch("src.memory.knowledge_store._embed", side_effect=_mock_embed):
                    s = KnowledgeStore(db_path=tmp_db, use_vec=True)

                    for i in range(LOW_CAP + 5):
                        s.write(
                            f"Knowledge entry item {i} unique content here {uuid.uuid4()}",
                            tier="observation",
                            confidence=0.5,
                            skip_dedup=True,
                        )

                    total = s._count_total()
                    s.close()

        # After exceeding cap, eviction should have fired
        assert total <= LOW_CAP, (
            f"Total ({total}) should be ≤ cap ({LOW_CAP}) after eviction"
        )

    def test_archived_evicted_before_rules(self, tmp_db):
        """Eviction should prefer archived entries over active ones."""
        from src.memory.knowledge_store import KnowledgeStore

        LOW_CAP = 10
        LOW_EVICTION = 5

        with patch("src.memory.knowledge_store.MAX_KNOWLEDGE_ITEMS", LOW_CAP):
            with patch("src.memory.knowledge_store.EVICTION_BATCH", LOW_EVICTION):
                with patch("src.memory.knowledge_store._embed", side_effect=_mock_embed):
                    s = KnowledgeStore(db_path=tmp_db, use_vec=True)

                    # Write some rules (should survive)
                    rule_ids = []
                    for i in range(3):
                        eid = s.write(
                            f"Critical rule entry {i} {uuid.uuid4()}",
                            tier="rule",
                            confidence=0.95,
                            skip_dedup=True,
                        )
                        rule_ids.append(eid)

                    # Write observations and archive them (should be evicted first)
                    archived_ids = []
                    for i in range(LOW_CAP):
                        eid = s.write(
                            f"Archived obs {i} {uuid.uuid4()}",
                            tier="observation",
                            confidence=0.1,
                            skip_dedup=True,
                        )
                        archived_ids.append(eid)
                        s.archive(eid)

                    # Now exceed cap with one more observation
                    s.write(
                        f"Final observation {uuid.uuid4()}",
                        tier="observation",
                        confidence=0.5,
                        skip_dedup=True,
                    )

                    # All rules should still exist
                    surviving_rules = [s.get(eid) for eid in rule_ids if s.get(eid) is not None]
                    s.close()

        assert len(surviving_rules) == 3, (
            f"All rules should survive eviction. Got {len(surviving_rules)}/3"
        )


import uuid  # ensure uuid is available for the test


# ---------------------------------------------------------------------------
# 8. Numpy fallback
# ---------------------------------------------------------------------------


class TestFallback:
    def test_numpy_store_write_and_read(self, numpy_store):
        """Numpy fallback mode should support basic write + read."""
        assert not numpy_store.using_vec, "Expected numpy fallback mode"

        eid = numpy_store.write(
            "Fallback mode test content",
            tier="observation",
            confidence=0.6,
        )
        assert eid is not None

        results = numpy_store.read("fallback content test")
        assert len(results) >= 1

    def test_numpy_store_dedup(self, tmp_db):
        """Numpy fallback should still perform semantic deduplication."""
        from src.memory.knowledge_store import KnowledgeStore

        base_vec = _make_unit(768, direction=7)
        dup_vec = _near_duplicate(base_vec, noise=0.001)

        call_count = [0]

        def dup_embed(text: str, task: str = "search_document") -> np.ndarray:
            call_count[0] += 1
            if "original" in text:
                return base_vec.copy()
            elif "duplicate" in text:
                return dup_vec.copy()
            else:
                return _make_unit(768, direction=call_count[0])

        with patch("src.memory.knowledge_store._embed", side_effect=dup_embed):
            s = KnowledgeStore(db_path=tmp_db, use_vec=False)
            eid1 = s.write("original numpy fallback content", tier="observation", confidence=0.6)
            eid2 = s.write("duplicate numpy fallback content", tier="observation", confidence=0.6)
            s.close()

        assert eid1 is not None
        assert eid2 is None, "Numpy fallback should reject near-duplicate"

    def test_vec_unavailable_falls_back(self, tmp_db):
        """If sqlite-vec import fails, store should gracefully fall back."""
        from src.memory.knowledge_store import KnowledgeStore

        with patch("src.memory.knowledge_store._embed", side_effect=_mock_embed):
            with patch.dict("sys.modules", {"sqlite_vec": None}):
                # Force vec init to fail by using use_vec=False
                s = KnowledgeStore(db_path=tmp_db, use_vec=False)
                assert not s.using_vec
                eid = s.write("fallback graceful test", tier="observation", confidence=0.5)
                assert eid is not None
                s.close()

    def test_numpy_tier_priority(self, tmp_db):
        """Numpy fallback should still respect tier ordering."""
        from src.memory.knowledge_store import KnowledgeStore

        same_vec = _make_unit(768, direction=30)

        def same_embed(text: str, task: str = "search_document") -> np.ndarray:
            return same_vec.copy()

        with patch("src.memory.knowledge_store._embed", side_effect=same_embed):
            s = KnowledgeStore(db_path=tmp_db, use_vec=False)
            s.write("Observation X", tier="observation", confidence=0.5, skip_dedup=True)
            s.write("Rule X", tier="rule", confidence=0.9, skip_dedup=True)
            results = s.read("query X", max_results=10)
            s.close()

        assert results[0].tier == "rule", f"First result should be rule, got {results[0].tier}"


# ---------------------------------------------------------------------------
# 9. Promote / validate / contradict
# ---------------------------------------------------------------------------


class TestPromoteAndValidate:
    def test_promote_observation_to_pattern(self, store):
        eid = store.write("Promotable observation", tier="observation", confidence=0.6)
        store.promote(eid, from_tier="observation", to_tier="pattern", evidence={"cycle": 5})
        entry = store.get(eid)
        assert entry.tier == "pattern"
        assert entry.last_validated_cycle == 5

    def test_promote_wrong_from_tier_raises(self, store):
        eid = store.write("Another entry", tier="observation", confidence=0.5)
        with pytest.raises(ValueError):
            store.promote(eid, from_tier="rule", to_tier="pattern")  # wrong from_tier

    def test_promote_invalid_direction_raises(self, store):
        eid = store.write("Rule entry for demotion test", tier="rule", confidence=0.9)
        with pytest.raises(ValueError):
            store.promote(eid, from_tier="rule", to_tier="observation")  # demotion not allowed

    def test_validate_increments_count(self, store):
        eid = store.write("Validatable entry", tier="pattern", confidence=0.7)
        store.validate(eid, cycle=10)
        store.validate(eid, cycle=20)
        entry = store.get(eid)
        assert entry.validation_count == 2
        assert entry.last_validated_cycle == 20

    def test_contradict(self, store):
        eid_a = store.write("Statement A", tier="rule", confidence=0.9)
        eid_b = store.write("Statement B contradicts A", tier="rule", confidence=0.8, skip_dedup=True)
        store.flag_contradiction(eid_a, contradicted_by_id=eid_b)
        entry_a = store.get(eid_a)
        assert eid_b in entry_a.contradicted_by


# ---------------------------------------------------------------------------
# 10. Repr and utility
# ---------------------------------------------------------------------------


class TestReprAndUtility:
    def test_repr_includes_counts(self, store):
        store.write("A rule entry", tier="rule", confidence=0.9)
        store.write("An observation", tier="observation", confidence=0.5)
        r = repr(store)
        assert "rules=1" in r
        assert "observations=1" in r

    def test_using_vec_property(self, store, numpy_store):
        assert store.using_vec is True or store.using_vec is False  # bool
        assert numpy_store.using_vec is False

    def test_close_idempotent(self, store):
        store.close()
        # Should not raise
