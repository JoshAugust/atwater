"""
src.knowledge.graph — Lightweight typed knowledge graph.

Wraps NetworkX DiGraph to add:
- Typed edge relationships: "supports", "contradicts", "derived_from", "supersedes"
- PageRank-based importance scoring
- Graph traversal: related, lineage, contradictions
- JSON persistence (NetworkX node_link format)
- Graphviz DOT export for visualisation

Usage
-----
    from src.knowledge.graph import KnowledgeGraph

    kg = KnowledgeGraph()
    kg.add_entry("e1", "Serif fonts improve readability", "observation")
    kg.add_entry("e2", "Sans-serif headlines outperform serif by 23%", "pattern")
    kg.add_relationship("e2", "e1", "contradicts")
    kg.add_relationship("e2", "e1", "derived_from")

    print(kg.get_importance("e2"))        # PageRank score
    print(kg.get_contradictions("e1"))    # entries contradicting e1
    print(kg.get_lineage("e2"))           # ancestors of e2

    kg.save("knowledge_graph.json")
    kg.load("knowledge_graph.json")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

import networkx as nx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

RelationshipType = Literal["supports", "contradicts", "derived_from", "supersedes"]

VALID_RELATIONSHIP_TYPES: frozenset[str] = frozenset(
    ["supports", "contradicts", "derived_from", "supersedes"]
)


# ---------------------------------------------------------------------------
# KnowledgeGraph
# ---------------------------------------------------------------------------

class KnowledgeGraph:
    """
    Typed directed knowledge graph for Atwater.

    Nodes represent knowledge entries (by their KB ID).
    Edges carry a ``rel_type`` attribute indicating the semantic relationship.

    Parameters
    ----------
    path:
        Optional path to persist the graph.  If provided, the graph is
        loaded on init if the file exists, and saved automatically on each
        write operation when ``auto_save=True``.
    auto_save:
        If True, automatically save to ``path`` on every write.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        auto_save: bool = False,
    ) -> None:
        self._G: nx.DiGraph = nx.DiGraph()
        self._path = Path(path) if path else None
        self._auto_save = auto_save
        self._pagerank_cache: dict[str, float] | None = None
        self._pagerank_dirty: bool = True

        if self._path and self._path.exists():
            self.load(self._path)

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_entry(
        self,
        entry_id: str,
        content: str,
        tier: str = "observation",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add or update a knowledge entry node.

        Parameters
        ----------
        entry_id:
            Unique ID (matches KnowledgeBase entry id).
        content:
            Human-readable content of the entry.
        tier:
            Knowledge tier ("observation", "pattern", "rule", "archived").
        metadata:
            Optional additional attributes to store on the node.
        """
        attrs: dict[str, Any] = {
            "content": content,
            "tier": tier,
        }
        if metadata:
            attrs.update(metadata)

        self._G.add_node(entry_id, **attrs)
        self._pagerank_dirty = True

        if self._auto_save and self._path:
            self.save(self._path)

        logger.debug("[KnowledgeGraph] Added node %s (tier=%s).", entry_id, tier)

    def remove_entry(self, entry_id: str) -> None:
        """Remove a node and all its edges from the graph."""
        if entry_id in self._G:
            self._G.remove_node(entry_id)
            self._pagerank_dirty = True
            if self._auto_save and self._path:
                self.save(self._path)

    def has_entry(self, entry_id: str) -> bool:
        """Return True if the node exists in the graph."""
        return entry_id in self._G

    def get_entry_data(self, entry_id: str) -> dict[str, Any] | None:
        """Return all stored attributes for a node, or None if not found."""
        if entry_id not in self._G:
            return None
        return dict(self._G.nodes[entry_id])

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_relationship(
        self,
        from_id: str,
        to_id: str,
        rel_type: RelationshipType,
        weight: float = 1.0,
    ) -> None:
        """
        Add a typed directed edge from ``from_id`` to ``to_id``.

        If either node does not exist as a full entry, a stub node is created
        so edges are never orphaned.

        Parameters
        ----------
        from_id:
            Source node ID.
        to_id:
            Target node ID.
        rel_type:
            Semantic relationship type.
        weight:
            Edge weight for PageRank computation (default 1.0).
        """
        if rel_type not in VALID_RELATIONSHIP_TYPES:
            raise ValueError(
                f"Invalid rel_type {rel_type!r}. "
                f"Must be one of: {sorted(VALID_RELATIONSHIP_TYPES)}"
            )

        # Ensure nodes exist
        for nid in (from_id, to_id):
            if nid not in self._G:
                self._G.add_node(nid, content="", tier="observation")

        self._G.add_edge(from_id, to_id, rel_type=rel_type, weight=weight)
        self._pagerank_dirty = True

        if self._auto_save and self._path:
            self.save(self._path)

        logger.debug(
            "[KnowledgeGraph] Edge %s -[%s]-> %s", from_id, rel_type, to_id
        )

    def remove_relationship(
        self, from_id: str, to_id: str, rel_type: str | None = None
    ) -> None:
        """
        Remove an edge (or all edges between two nodes if rel_type is None).
        """
        if not self._G.has_edge(from_id, to_id):
            return
        if rel_type is None:
            self._G.remove_edge(from_id, to_id)
        else:
            edge_data = self._G.get_edge_data(from_id, to_id, {})
            if edge_data.get("rel_type") == rel_type:
                self._G.remove_edge(from_id, to_id)
        self._pagerank_dirty = True

    # ------------------------------------------------------------------
    # Traversal queries
    # ------------------------------------------------------------------

    def get_related(
        self,
        entry_id: str,
        rel_type: str | None = None,
        depth: int = 2,
    ) -> list[dict[str, Any]]:
        """
        Get entries within ``depth`` hops of ``entry_id``.

        Parameters
        ----------
        entry_id:
            Starting node.
        rel_type:
            If provided, only follow edges with this relationship type.
        depth:
            Maximum number of hops.

        Returns
        -------
        list[dict]
            Each dict has: "entry_id", "content", "tier", "distance",
            and any other node attributes.
        """
        if entry_id not in self._G:
            return []

        if rel_type is None:
            subgraph = nx.ego_graph(self._G, entry_id, radius=depth, undirected=True)
            neighbors = set(subgraph.nodes) - {entry_id}
        else:
            # Filter graph to only edges of the requested type
            filtered = nx.DiGraph(
                (u, v, d)
                for u, v, d in self._G.edges(data=True)
                if d.get("rel_type") == rel_type
            )
            if entry_id not in filtered:
                return []
            subgraph = nx.ego_graph(filtered, entry_id, radius=depth, undirected=True)
            neighbors = set(subgraph.nodes) - {entry_id}

        results = []
        for nid in neighbors:
            data = dict(self._G.nodes.get(nid, {}))
            data["entry_id"] = nid
            # Shortest path distance
            try:
                dist = nx.shortest_path_length(
                    self._G.to_undirected(), entry_id, nid
                )
            except nx.NetworkXNoPath:
                dist = -1
            data["distance"] = dist
            results.append(data)

        return sorted(results, key=lambda x: x["distance"])

    def get_contradictions(self, entry_id: str) -> list[dict[str, Any]]:
        """
        Return all entries that directly contradict ``entry_id``.
        Looks at both incoming and outgoing "contradicts" edges.
        """
        if entry_id not in self._G:
            return []

        contradicting: list[dict[str, Any]] = []

        # Outgoing contradictions (entry_id contradicts X)
        for _, target, data in self._G.out_edges(entry_id, data=True):
            if data.get("rel_type") == "contradicts":
                node_data = dict(self._G.nodes.get(target, {}))
                node_data["entry_id"] = target
                node_data["direction"] = "outgoing"
                contradicting.append(node_data)

        # Incoming contradictions (X contradicts entry_id)
        for source, _, data in self._G.in_edges(entry_id, data=True):
            if data.get("rel_type") == "contradicts":
                node_data = dict(self._G.nodes.get(source, {}))
                node_data["entry_id"] = source
                node_data["direction"] = "incoming"
                contradicting.append(node_data)

        return contradicting

    def get_lineage(self, entry_id: str) -> list[dict[str, Any]]:
        """
        Return the lineage (ancestor chain) of ``entry_id``.

        Follows "derived_from" edges forward — i.e. the entries that
        ``entry_id`` was derived from, their ancestors, etc.

        Convention: add_relationship(child, parent, "derived_from") means
        "child was derived from parent", so the edge goes child → parent.
        get_lineage follows these outgoing edges to find ancestors.

        Returns list ordered from direct parents to most distant ancestors.
        """
        if entry_id not in self._G:
            return []

        ancestors: list[dict[str, Any]] = []
        visited: set[str] = {entry_id}
        queue: list[str] = [entry_id]

        while queue:
            current = queue.pop(0)
            # Follow outgoing "derived_from" edges: current → parent
            for _, target, data in self._G.out_edges(current, data=True):
                if data.get("rel_type") == "derived_from" and target not in visited:
                    visited.add(target)
                    node_data = dict(self._G.nodes.get(target, {}))
                    node_data["entry_id"] = target
                    ancestors.append(node_data)
                    queue.append(target)

        return ancestors

    def get_superseded_by(self, entry_id: str) -> list[dict[str, Any]]:
        """Return entries that supersede ``entry_id``."""
        if entry_id not in self._G:
            return []
        results = []
        for source, _, data in self._G.in_edges(entry_id, data=True):
            if data.get("rel_type") == "supersedes":
                node_data = dict(self._G.nodes.get(source, {}))
                node_data["entry_id"] = source
                results.append(node_data)
        return results

    # ------------------------------------------------------------------
    # Importance
    # ------------------------------------------------------------------

    def get_importance(self, entry_id: str) -> float:
        """
        Return the PageRank score for ``entry_id``.

        Higher score = more central / more referenced node.
        Returns 0.0 if the entry is not found.
        """
        pagerank = self._get_pagerank()
        return pagerank.get(entry_id, 0.0)

    def get_top_entries(self, k: int = 10) -> list[tuple[str, float]]:
        """Return top-k entries by PageRank score."""
        pagerank = self._get_pagerank()
        sorted_items = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:k]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path | None = None) -> None:
        """
        Persist the graph to a JSON file using NetworkX node_link format.

        Parameters
        ----------
        path:
            File path.  Falls back to the instance's configured path.
        """
        target = Path(path) if path else self._path
        if target is None:
            raise ValueError("No path configured for save.")

        target.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self._G)
        target.write_text(json.dumps(data, indent=2))
        logger.info("[KnowledgeGraph] Saved %d nodes, %d edges to %s.",
                    self._G.number_of_nodes(), self._G.number_of_edges(), target)

    def load(self, path: str | Path | None = None) -> None:
        """
        Load the graph from a JSON file (NetworkX node_link format).

        Parameters
        ----------
        path:
            File path.  Falls back to the instance's configured path.
        """
        target = Path(path) if path else self._path
        if target is None:
            raise ValueError("No path configured for load.")

        data = json.loads(target.read_text())
        # NetworkX 3.4+ changed the default key from "links" to "edges".
        # Pass edges_key explicitly to handle both old and new formats.
        edges_key = "edges" if "edges" in data else "links"
        self._G = nx.node_link_graph(data, directed=True, edges=edges_key)
        self._pagerank_dirty = True
        logger.info("[KnowledgeGraph] Loaded %d nodes, %d edges from %s.",
                    self._G.number_of_nodes(), self._G.number_of_edges(), target)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def export_dot(self) -> str:
        """
        Export the graph in Graphviz DOT format.

        Returns
        -------
        str
            DOT language string suitable for rendering with graphviz.

        Example
        -------
            dot_str = kg.export_dot()
            Path("knowledge_graph.dot").write_text(dot_str)
            # Then: dot -Tpng knowledge_graph.dot -o knowledge_graph.png
        """
        lines = ["digraph KnowledgeGraph {"]
        lines.append('    rankdir=LR;')
        lines.append('    node [shape=box, style=filled, fontname="Helvetica"];')

        # Tier colours
        tier_colours = {
            "rule": "#2ecc71",
            "pattern": "#3498db",
            "observation": "#f39c12",
            "archived": "#95a5a6",
        }

        # Nodes
        for node_id, attrs in self._G.nodes(data=True):
            tier = attrs.get("tier", "observation")
            colour = tier_colours.get(tier, "#ecf0f1")
            content = attrs.get("content", "")[:50].replace('"', '\\"')
            label = f"{node_id}\\n{content}" if content else node_id
            lines.append(f'    "{node_id}" [label="{label}", fillcolor="{colour}"];')

        # Edge style per relationship type
        edge_styles = {
            "supports":     'color="#2ecc71"',
            "contradicts":  'color="#e74c3c", style=dashed',
            "derived_from": 'color="#3498db"',
            "supersedes":   'color="#9b59b6", style=bold',
        }

        # Edges
        for u, v, data in self._G.edges(data=True):
            rel = data.get("rel_type", "unknown")
            style = edge_styles.get(rel, "")
            label = rel
            lines.append(
                f'    "{u}" -> "{v}" [label="{label}", {style}];'
            )

        lines.append("}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Stats / inspection
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return graph statistics."""
        return {
            "num_nodes": self._G.number_of_nodes(),
            "num_edges": self._G.number_of_edges(),
            "is_dag": nx.is_directed_acyclic_graph(self._G),
            "density": nx.density(self._G),
            "rel_type_counts": self._edge_type_counts(),
        }

    def _edge_type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, _, data in self._G.edges(data=True):
            rel = data.get("rel_type", "unknown")
            counts[rel] = counts.get(rel, 0) + 1
        return counts

    def _get_pagerank(self) -> dict[str, float]:
        """Return cached PageRank, recomputing if dirty."""
        if self._pagerank_dirty or self._pagerank_cache is None:
            if self._G.number_of_nodes() == 0:
                self._pagerank_cache = {}
            else:
                try:
                    self._pagerank_cache = nx.pagerank(
                        self._G, weight="weight", alpha=0.85
                    )
                except nx.PowerIterationFailedConvergence:
                    logger.warning(
                        "[KnowledgeGraph] PageRank did not converge — using uniform scores."
                    )
                    n = self._G.number_of_nodes()
                    self._pagerank_cache = {
                        node: 1.0 / n for node in self._G.nodes
                    }
            self._pagerank_dirty = False
        return self._pagerank_cache  # type: ignore[return-value]
