"""
memory — Three-tier memory system for the Atwater cognitive architecture.

Tiers
-----
Tier 1 — WorkingMemory     Ephemeral per-turn state (dict-based, cleared each turn).
Tier 2 — SharedState       SQLite-backed state machine with role-based scoping.
Tier 3 — KnowledgeBase     Hierarchical persistent store with semantic search.

Quick start
-----------
    from memory import WorkingMemory, SharedState, KnowledgeBase, KnowledgeEntry

    # Tier 1
    wm = WorkingMemory()
    wm.write("score", 0.91)
    wm.snapshot()   # -> {"score": 0.91}
    wm.clear()

    # Tier 2
    ss = SharedState()                                              # default: state.db
    ss.state_write("current_hypothesis", {"bg": "dark"}, roles=["director"])
    ss.state_read("current_hypothesis")                             # -> {"bg": "dark"}
    ss.state_read_scoped("director")                                # -> filtered dict

    # Tier 3
    kb = KnowledgeBase()                                            # default: knowledge.db
    eid = kb.knowledge_write("Sans-serif beats serif by 23%",
                             tier="rule", confidence=0.95,
                             topic_cluster="typography")
    kb.knowledge_read("best headline font")                         # -> [KnowledgeEntry, ...]
    kb.knowledge_promote(eid, from_tier="pattern", to_tier="rule",
                         evidence={"trial_count": 210})
"""

from .knowledge_base import KnowledgeBase, KnowledgeEntry, VALID_TIERS
from .shared_state import SharedState, ROLE_ALL, ORCHESTRATOR_ROLE
from .working import WorkingMemory

__all__ = [
    # Tier 1
    "WorkingMemory",
    # Tier 2
    "SharedState",
    "ROLE_ALL",
    "ORCHESTRATOR_ROLE",
    # Tier 3
    "KnowledgeBase",
    "KnowledgeEntry",
    "VALID_TIERS",
]
