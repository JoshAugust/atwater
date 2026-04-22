"""
atwater.src.knowledge — Knowledge base layer for the Atwater cognitive agent.

Public API
----------

Models
~~~~~~
- ``KnowledgeEntry``    — atomic unit of stored knowledge
- ``KnowledgeTier``     — type alias for tier literals
- ``TIER_RANK``         — numeric rank mapping for tier comparison
- ``PromotionCriteria`` — thresholds for tier advancement

Consolidation
~~~~~~~~~~~~~
- ``ConsolidationEngine`` — full consolidation pipeline (decay, merge, promote)

Clustering
~~~~~~~~~~
- ``TopicClusterer`` — HDBSCAN-backed semantic topic grouping

Typical workflow
----------------
::

    from atwater.src.knowledge import (
        KnowledgeEntry,
        ConsolidationEngine,
        TopicClusterer,
    )

    # Build / load your knowledge base
    kb: list[KnowledgeEntry] = [...]

    # Compute embeddings externally (any embedding model)
    embeddings = np.array([embed(e.content) for e in kb])

    # Cluster (optional — updates entry.topic_cluster in-place)
    clusterer = TopicClusterer(min_cluster_size=3)
    clusters = clusterer.cluster_entries(kb, embeddings)

    # Run a full consolidation pass
    engine = ConsolidationEngine()
    changelog = engine.run_consolidation(kb, current_cycle=150, embeddings=embeddings)
    print(changelog)
"""

from .clustering import TopicClusterer
from .consolidator import ConsolidationEngine
from .models import (
    TIER_RANK,
    KnowledgeEntry,
    KnowledgeTier,
    PromotionCriteria,
)

__all__ = [
    # Models
    "KnowledgeEntry",
    "KnowledgeTier",
    "TIER_RANK",
    "PromotionCriteria",
    # Consolidation
    "ConsolidationEngine",
    # Clustering
    "TopicClusterer",
]
