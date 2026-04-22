"""
Topic clustering for the Atwater knowledge base.

Uses HDBSCAN (via scikit-learn) to automatically group KnowledgeEntry
objects by semantic similarity, without requiring a pre-specified number
of clusters.

Typical usage
-------------
::

    clusterer = TopicClusterer(min_cluster_size=3)
    embeddings = np.array([embed(e.content) for e in entries])
    clusters = clusterer.cluster_entries(entries, embeddings)
    # {"cluster_0": [...], "cluster_1": [...], "unclustered": [...]}
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from sklearn.cluster import HDBSCAN

if TYPE_CHECKING:
    from .models import KnowledgeEntry

logger = logging.getLogger(__name__)

# Label assigned by HDBSCAN to noise / unclustered points
_NOISE_LABEL = -1
_UNCLUSTERED_KEY = "unclustered"


class TopicClusterer:
    """
    Semantic topic clusterer backed by HDBSCAN.

    Parameters
    ----------
    min_cluster_size:
        Minimum number of entries required to form a cluster.  Points that
        cannot join any cluster of this size are labelled *unclustered*.
        Default is 3, matching the KNOWLEDGE_SCALING.md recommendation.
    min_samples:
        HDBSCAN ``min_samples`` parameter.  Controls how conservative the
        algorithm is — higher values produce fewer, denser clusters.  When
        ``None`` defaults to ``min_cluster_size``.
    cluster_selection_epsilon:
        HDBSCAN epsilon value for merging micro-clusters.  A small positive
        value (e.g. 0.05) can reduce fragmentation.
    metric:
        Distance metric passed to HDBSCAN.  ``"euclidean"`` works well for
        normalised embedding vectors; use ``"cosine"`` for raw embeddings.
    """

    def __init__(
        self,
        min_cluster_size: int = 3,
        min_samples: int | None = None,
        cluster_selection_epsilon: float = 0.0,
        metric: str = "euclidean",
    ) -> None:
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples if min_samples is not None else min_cluster_size
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.metric = metric

        # Kept after fitting so callers can inspect labels directly
        self._last_labels: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cluster_entries(
        self,
        entries: list[KnowledgeEntry],
        embeddings: np.ndarray,
    ) -> dict[str, list[KnowledgeEntry]]:
        """
        Cluster *entries* by their pre-computed embedding vectors.

        Parameters
        ----------
        entries:
            Knowledge entries to cluster.  Must be in the same order as
            *embeddings*.
        embeddings:
            2-D float array of shape ``(len(entries), embedding_dim)``.
            Each row is the embedding for the corresponding entry.

        Returns
        -------
        dict[str, list[KnowledgeEntry]]
            Mapping from cluster key (e.g. ``"cluster_0"``) to the list of
            entries assigned to that cluster.  Noise points are grouped
            under ``"unclustered"``.

        Raises
        ------
        ValueError
            If *entries* and *embeddings* have different lengths, or if
            *embeddings* is not 2-D.
        """
        self._validate_inputs(entries, embeddings)

        if len(entries) == 0:
            logger.debug("TopicClusterer.cluster_entries: no entries to cluster.")
            return {}

        if len(entries) < self.min_cluster_size:
            # Not enough entries to form even one cluster — all go unclustered
            logger.debug(
                "TopicClusterer: %d entries < min_cluster_size=%d; "
                "all placed in 'unclustered'.",
                len(entries),
                self.min_cluster_size,
            )
            return {_UNCLUSTERED_KEY: list(entries)}

        labels = self._fit(embeddings)
        clusters = self._build_cluster_dict(entries, labels)
        self._apply_labels_to_entries(entries, labels)

        n_clusters = len([k for k in clusters if k != _UNCLUSTERED_KEY])
        n_noise = len(clusters.get(_UNCLUSTERED_KEY, []))
        logger.info(
            "TopicClusterer: %d entries → %d clusters, %d unclustered.",
            len(entries),
            n_clusters,
            n_noise,
        )
        return clusters

    def recluster(
        self,
        entries: list[KnowledgeEntry],
        embeddings: np.ndarray,
    ) -> dict[str, list[KnowledgeEntry]]:
        """
        Re-run clustering from scratch, updating ``entry.topic_cluster``
        for every entry.

        This is intended for periodic updates (e.g. every 100 cycles) so
        that newly discovered entries are folded into the existing topic
        structure — or cause existing clusters to merge/split.

        Parameters
        ----------
        entries:
            All *active* entries in the knowledge base (archived entries
            are typically excluded to save time).
        embeddings:
            Corresponding embedding matrix.

        Returns
        -------
        dict[str, list[KnowledgeEntry]]
            Fresh cluster assignments.
        """
        logger.info(
            "TopicClusterer.recluster: re-clustering %d entries.", len(entries)
        )
        return self.cluster_entries(entries, embeddings)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit(self, embeddings: np.ndarray) -> np.ndarray:
        """Run HDBSCAN and return integer cluster labels."""
        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric=self.metric,
        )
        labels: np.ndarray = clusterer.fit_predict(embeddings)
        self._last_labels = labels
        return labels

    @staticmethod
    def _build_cluster_dict(
        entries: list[KnowledgeEntry],
        labels: np.ndarray,
    ) -> dict[str, list[KnowledgeEntry]]:
        """
        Map HDBSCAN integer labels → named cluster buckets.

        Noise points (label == -1) go into ``"unclustered"``.
        All other points go into ``"cluster_<label>"``.
        """
        clusters: dict[str, list[KnowledgeEntry]] = {}
        for entry, label in zip(entries, labels):
            key = _UNCLUSTERED_KEY if label == _NOISE_LABEL else f"cluster_{label}"
            clusters.setdefault(key, []).append(entry)
        return clusters

    @staticmethod
    def _apply_labels_to_entries(
        entries: list[KnowledgeEntry],
        labels: np.ndarray,
    ) -> None:
        """Write the cluster key back onto each entry's ``topic_cluster`` field."""
        for entry, label in zip(entries, labels):
            entry.topic_cluster = (
                _UNCLUSTERED_KEY if label == _NOISE_LABEL else f"cluster_{label}"
            )

    @staticmethod
    def _validate_inputs(
        entries: list[KnowledgeEntry],
        embeddings: np.ndarray,
    ) -> None:
        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be a 2-D array, got shape {embeddings.shape}."
            )
        if len(entries) != embeddings.shape[0]:
            raise ValueError(
                f"entries ({len(entries)}) and embeddings ({embeddings.shape[0]}) "
                "must have the same length."
            )
