"""
tests/test_knowledge_graph.py — Tests for src.knowledge.graph.

Tests:
- add_entry / remove_entry / has_entry
- add_relationship with all valid types
- Invalid relationship type raises ValueError
- get_related: depth traversal, type filter
- get_contradictions: incoming and outgoing
- get_lineage: derived_from chain
- get_importance: PageRank scores
- get_top_entries: returns sorted list
- Persistence: save/load roundtrip
- export_dot: valid DOT format string
- stats() returns expected structure
- Stub nodes auto-created for orphan edges
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest

from src.knowledge.graph import KnowledgeGraph, VALID_RELATIONSHIP_TYPES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_graph() -> KnowledgeGraph:
    """A small pre-built graph for reuse."""
    kg = KnowledgeGraph()
    kg.add_entry("e1", "Sans-serif beats serif by 23%", tier="rule")
    kg.add_entry("e2", "Serif fonts improve readability", tier="observation")
    kg.add_entry("e3", "Dark backgrounds improve CTR", tier="pattern")
    kg.add_entry("e4", "Light backgrounds perform better in print", tier="observation")
    kg.add_relationship("e1", "e2", "contradicts")
    kg.add_relationship("e3", "e4", "contradicts")
    kg.add_relationship("e1", "e3", "supports")
    return kg


@pytest.fixture
def lineage_graph() -> KnowledgeGraph:
    """A graph with derived_from relationships for lineage testing."""
    kg = KnowledgeGraph()
    kg.add_entry("root", "Original observation", tier="observation")
    kg.add_entry("mid", "Derived pattern", tier="pattern")
    kg.add_entry("leaf", "Leaf rule", tier="rule")
    kg.add_relationship("mid", "root", "derived_from")
    kg.add_relationship("leaf", "mid", "derived_from")
    return kg


# ---------------------------------------------------------------------------
# add_entry
# ---------------------------------------------------------------------------

class TestAddEntry:
    def test_basic_add(self):
        kg = KnowledgeGraph()
        kg.add_entry("e1", "Test content", tier="observation")
        assert kg.has_entry("e1")

    def test_node_attributes_stored(self):
        kg = KnowledgeGraph()
        kg.add_entry("e1", "Content here", tier="rule", metadata={"confidence": 0.95})
        data = kg.get_entry_data("e1")
        assert data["content"] == "Content here"
        assert data["tier"] == "rule"
        assert data["confidence"] == 0.95

    def test_update_existing_node(self):
        kg = KnowledgeGraph()
        kg.add_entry("e1", "Original", tier="observation")
        kg.add_entry("e1", "Updated", tier="pattern")
        data = kg.get_entry_data("e1")
        assert data["content"] == "Updated"
        assert data["tier"] == "pattern"

    def test_has_entry_false_for_missing(self):
        kg = KnowledgeGraph()
        assert not kg.has_entry("nonexistent")

    def test_get_entry_data_returns_none_for_missing(self):
        kg = KnowledgeGraph()
        assert kg.get_entry_data("missing") is None


# ---------------------------------------------------------------------------
# remove_entry
# ---------------------------------------------------------------------------

class TestRemoveEntry:
    def test_remove_existing(self):
        kg = KnowledgeGraph()
        kg.add_entry("e1", "Content", tier="observation")
        kg.remove_entry("e1")
        assert not kg.has_entry("e1")

    def test_remove_also_removes_edges(self, simple_graph):
        simple_graph.remove_entry("e1")
        # e2 should still exist; edges from/to e1 should be gone
        assert not simple_graph.has_entry("e1")
        assert simple_graph.has_entry("e2")

    def test_remove_missing_no_crash(self):
        kg = KnowledgeGraph()
        kg.remove_entry("does_not_exist")  # should not raise


# ---------------------------------------------------------------------------
# add_relationship
# ---------------------------------------------------------------------------

class TestAddRelationship:
    def test_all_valid_types(self):
        kg = KnowledgeGraph()
        kg.add_entry("a", "A", tier="observation")
        kg.add_entry("b", "B", tier="observation")
        for rel_type in VALID_RELATIONSHIP_TYPES:
            # Use a fresh pair per type
            kg.add_entry(f"x_{rel_type}", "X", tier="observation")
            kg.add_entry(f"y_{rel_type}", "Y", tier="observation")
            kg.add_relationship(f"x_{rel_type}", f"y_{rel_type}", rel_type)

    def test_invalid_type_raises(self):
        kg = KnowledgeGraph()
        kg.add_entry("a", "A", tier="observation")
        kg.add_entry("b", "B", tier="observation")
        with pytest.raises(ValueError):
            kg.add_relationship("a", "b", "unknown_type")  # type: ignore

    def test_edge_stored_with_rel_type(self):
        kg = KnowledgeGraph()
        kg.add_entry("a", "A", tier="observation")
        kg.add_entry("b", "B", tier="observation")
        kg.add_relationship("a", "b", "supports")
        edge_data = kg._G.get_edge_data("a", "b")
        assert edge_data["rel_type"] == "supports"

    def test_stub_nodes_auto_created(self):
        kg = KnowledgeGraph()
        # Neither node exists
        kg.add_relationship("x", "y", "supports")
        assert kg.has_entry("x")
        assert kg.has_entry("y")

    def test_weight_stored(self):
        kg = KnowledgeGraph()
        kg.add_entry("a", "A", tier="observation")
        kg.add_entry("b", "B", tier="observation")
        kg.add_relationship("a", "b", "supports", weight=2.5)
        edge_data = kg._G.get_edge_data("a", "b")
        assert edge_data["weight"] == 2.5


# ---------------------------------------------------------------------------
# get_related
# ---------------------------------------------------------------------------

class TestGetRelated:
    def test_direct_neighbours(self, simple_graph):
        # e1 has edges to e2 and e3
        related = simple_graph.get_related("e1", depth=1)
        ids = [r["entry_id"] for r in related]
        assert "e2" in ids
        assert "e3" in ids

    def test_depth_two_reaches_further(self, simple_graph):
        # e1 → e3 → e4 (depth=2 should reach e4)
        related = simple_graph.get_related("e1", depth=2)
        ids = [r["entry_id"] for r in related]
        assert "e4" in ids

    def test_depth_one_does_not_reach_far(self, simple_graph):
        related = simple_graph.get_related("e1", depth=1)
        ids = [r["entry_id"] for r in related]
        # e4 is 2 hops away
        assert "e4" not in ids

    def test_rel_type_filter(self, simple_graph):
        # Only "supports" edges from e1 — should get e3 but not e2 (contradicts)
        related = simple_graph.get_related("e1", rel_type="supports", depth=1)
        ids = [r["entry_id"] for r in related]
        assert "e3" in ids
        assert "e2" not in ids

    def test_unknown_entry_returns_empty(self):
        kg = KnowledgeGraph()
        assert kg.get_related("nonexistent") == []

    def test_isolated_node_returns_empty(self):
        kg = KnowledgeGraph()
        kg.add_entry("solo", "Alone", tier="observation")
        related = kg.get_related("solo")
        assert related == []

    def test_result_has_distance_field(self, simple_graph):
        related = simple_graph.get_related("e1", depth=2)
        for r in related:
            assert "distance" in r
            assert isinstance(r["distance"], int)


# ---------------------------------------------------------------------------
# get_contradictions
# ---------------------------------------------------------------------------

class TestGetContradictions:
    def test_outgoing_contradiction(self, simple_graph):
        # e1 contradicts e2
        contras = simple_graph.get_contradictions("e1")
        ids = [c["entry_id"] for c in contras]
        assert "e2" in ids

    def test_incoming_contradiction(self, simple_graph):
        # e1 contradicts e2 → e2 is contradicted by e1
        contras = simple_graph.get_contradictions("e2")
        ids = [c["entry_id"] for c in contras]
        assert "e1" in ids

    def test_direction_field(self, simple_graph):
        contras = simple_graph.get_contradictions("e2")
        directions = {c["entry_id"]: c["direction"] for c in contras}
        assert directions["e1"] == "incoming"

    def test_no_contradictions(self):
        kg = KnowledgeGraph()
        kg.add_entry("a", "A", tier="observation")
        kg.add_entry("b", "B", tier="observation")
        kg.add_relationship("a", "b", "supports")
        assert kg.get_contradictions("a") == []

    def test_unknown_entry_returns_empty(self):
        kg = KnowledgeGraph()
        assert kg.get_contradictions("ghost") == []


# ---------------------------------------------------------------------------
# get_lineage
# ---------------------------------------------------------------------------

class TestGetLineage:
    def test_direct_parent(self, lineage_graph):
        lineage = lineage_graph.get_lineage("mid")
        ids = [e["entry_id"] for e in lineage]
        assert "root" in ids

    def test_transitive_ancestor(self, lineage_graph):
        lineage = lineage_graph.get_lineage("leaf")
        ids = [e["entry_id"] for e in lineage]
        assert "mid" in ids
        assert "root" in ids

    def test_root_has_no_lineage(self, lineage_graph):
        lineage = lineage_graph.get_lineage("root")
        assert lineage == []

    def test_unknown_entry_returns_empty(self):
        kg = KnowledgeGraph()
        assert kg.get_lineage("ghost") == []

    def test_lineage_does_not_include_self(self, lineage_graph):
        lineage = lineage_graph.get_lineage("leaf")
        ids = [e["entry_id"] for e in lineage]
        assert "leaf" not in ids


# ---------------------------------------------------------------------------
# get_importance (PageRank)
# ---------------------------------------------------------------------------

class TestGetImportance:
    def test_returns_float(self, simple_graph):
        score = simple_graph.get_importance("e1")
        assert isinstance(score, float)

    def test_in_unit_interval(self, simple_graph):
        for node in ["e1", "e2", "e3", "e4"]:
            score = simple_graph.get_importance(node)
            assert 0.0 <= score <= 1.0

    def test_missing_entry_returns_zero(self):
        kg = KnowledgeGraph()
        assert kg.get_importance("ghost") == 0.0

    def test_highly_referenced_node_has_higher_score(self):
        """A node pointed to by many others should have higher PageRank."""
        kg = KnowledgeGraph()
        kg.add_entry("hub", "Hub node", tier="pattern")
        for i in range(5):
            kg.add_entry(f"spoke_{i}", f"Spoke {i}", tier="observation")
            kg.add_relationship(f"spoke_{i}", "hub", "supports")
        hub_score = kg.get_importance("hub")
        spoke_score = kg.get_importance("spoke_0")
        assert hub_score > spoke_score

    def test_get_top_entries_sorted(self, simple_graph):
        top = simple_graph.get_top_entries(k=3)
        assert len(top) <= 3
        scores = [score for _, score in top]
        assert scores == sorted(scores, reverse=True)

    def test_empty_graph_importance(self):
        kg = KnowledgeGraph()
        assert kg.get_importance("any") == 0.0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_creates_file(self, tmp_path, simple_graph):
        path = tmp_path / "graph.json"
        simple_graph.save(path)
        assert path.exists()

    def test_save_is_valid_json(self, tmp_path, simple_graph):
        path = tmp_path / "graph.json"
        simple_graph.save(path)
        data = json.loads(path.read_text())
        assert "nodes" in data
        # NetworkX 3.x uses "edges"; older versions use "links"
        assert "edges" in data or "links" in data

    def test_load_restores_nodes(self, tmp_path, simple_graph):
        path = tmp_path / "graph.json"
        simple_graph.save(path)
        kg2 = KnowledgeGraph()
        kg2.load(path)
        assert kg2.has_entry("e1")
        assert kg2.has_entry("e2")
        assert kg2.has_entry("e3")
        assert kg2.has_entry("e4")

    def test_load_restores_edges(self, tmp_path, simple_graph):
        path = tmp_path / "graph.json"
        simple_graph.save(path)
        kg2 = KnowledgeGraph()
        kg2.load(path)
        edge_data = kg2._G.get_edge_data("e1", "e2")
        assert edge_data is not None
        assert edge_data["rel_type"] == "contradicts"

    def test_auto_load_on_init(self, tmp_path, simple_graph):
        path = tmp_path / "graph.json"
        simple_graph.save(path)
        kg3 = KnowledgeGraph(path=path)
        assert kg3.has_entry("e1")

    def test_no_path_raises_on_save(self):
        kg = KnowledgeGraph()
        kg.add_entry("e1", "test", tier="observation")
        with pytest.raises(ValueError):
            kg.save()

    def test_roundtrip_preserves_node_content(self, tmp_path):
        kg = KnowledgeGraph()
        kg.add_entry("x", "Important insight", tier="rule",
                     metadata={"confidence": 0.9})
        path = tmp_path / "g.json"
        kg.save(path)
        kg2 = KnowledgeGraph()
        kg2.load(path)
        data = kg2.get_entry_data("x")
        assert data["content"] == "Important insight"
        assert data["tier"] == "rule"


# ---------------------------------------------------------------------------
# export_dot
# ---------------------------------------------------------------------------

class TestExportDot:
    def test_returns_string(self, simple_graph):
        dot = simple_graph.export_dot()
        assert isinstance(dot, str)

    def test_starts_with_digraph(self, simple_graph):
        dot = simple_graph.export_dot()
        assert dot.strip().startswith("digraph")

    def test_contains_node_ids(self, simple_graph):
        dot = simple_graph.export_dot()
        assert '"e1"' in dot
        assert '"e2"' in dot

    def test_contains_edge_labels(self, simple_graph):
        dot = simple_graph.export_dot()
        assert "contradicts" in dot

    def test_empty_graph_valid_dot(self):
        kg = KnowledgeGraph()
        dot = kg.export_dot()
        assert "digraph" in dot
        assert "}" in dot


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_structure(self, simple_graph):
        s = simple_graph.stats()
        assert "num_nodes" in s
        assert "num_edges" in s
        assert "density" in s
        assert "rel_type_counts" in s

    def test_stats_node_count(self, simple_graph):
        s = simple_graph.stats()
        assert s["num_nodes"] == 4

    def test_stats_edge_type_counts(self, simple_graph):
        s = simple_graph.stats()
        counts = s["rel_type_counts"]
        assert counts.get("contradicts", 0) >= 2
        assert counts.get("supports", 0) >= 1
