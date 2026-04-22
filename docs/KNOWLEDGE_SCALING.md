# Knowledge Scaling: Sustainability at 1,000+ Cycles

How to keep a knowledge base effective as it grows from tens to thousands of entries.

---

## The Problem

A flat knowledge base with vector search works for ~100-500 entries. Past that:
- Semantically similar entries compete during retrieval
- Model gets conflicting signals from old vs new findings
- Retrieval quality degrades logarithmically with size

---

## Solution 1: Knowledge Consolidation

After every N cycles (default 50), run a dedicated consolidation pass:

1. Pull all entries tagged with the same topic cluster
2. Merge into a single authoritative entry (highest-confidence findings)
3. Archive raw originals, promote merged version
4. Track lineage: "derived from cycles 47, 82, 91, 103"

```python
def consolidate_cluster(cluster_entries: list[KnowledgeEntry]) -> KnowledgeEntry:
    """Merge multiple observations into one pattern or rule."""
    # Sort by confidence * validation_count
    ranked = sorted(
        cluster_entries,
        key=lambda e: e.confidence * e.validation_count,
        reverse=True
    )
    
    # Use the highest-ranked as base, incorporate supporting evidence
    merged = KnowledgeEntry(
        content=synthesize_content([e.content for e in ranked]),
        tier=determine_promotion_tier(ranked),
        confidence=weighted_confidence(ranked),
        validation_count=sum(e.validation_count for e in ranked),
        lineage=[e.id for e in cluster_entries],
    )
    
    # Archive originals
    for entry in cluster_entries:
        entry.tier = "archived"
    
    return merged
```

Without consolidation, by cycle 500 you'll have 40 entries saying slightly different things about the same pattern. Agents won't know which to trust.

---

## Solution 2: Hierarchical Retrieval

### Three-Tier Query Strategy

```python
def retrieve_knowledge(query: str, max_results: int = 5) -> list[KnowledgeEntry]:
    results = []
    
    # 1. Check Rules first (highest authority)
    rules = semantic_search(query, tier="rule", limit=3)
    results.extend(rules)
    
    # 2. If Rules fully answer, stop here
    if sufficient_coverage(results, query):
        return results
    
    # 3. Check Patterns
    patterns = semantic_search(query, tier="pattern", limit=3)
    results.extend(patterns)
    
    if sufficient_coverage(results, query):
        return results[:max_results]
    
    # 4. Only fall through to Observations if needed
    observations = semantic_search(query, tier="observation", limit=3)
    results.extend(observations)
    
    return results[:max_results]
```

### Tier Promotion Criteria

| Promotion | Required Evidence |
|-----------|------------------|
| Observation → Pattern | Validated 5+ times AND Optuna shows consistent effect (p < 0.1) |
| Pattern → Rule | Validated 20+ times AND Optuna shows strong effect (p < 0.05) across 200+ trials |
| Any → Archived | Not validated in 200 cycles AND no Optuna evidence of ongoing relevance |

---

## Solution 3: Confidence Decay

```python
def decay_confidence(entry: KnowledgeEntry, current_cycle: int) -> float:
    """Entries that haven't been validated recently lose confidence."""
    cycles_since_validation = current_cycle - entry.last_validated_cycle
    
    # No decay for first 100 cycles
    if cycles_since_validation < 100:
        return entry.confidence
    
    # Gentle decay: loses ~10% confidence per 100 cycles of inactivity
    decay_factor = 0.9 ** ((cycles_since_validation - 100) / 100)
    return entry.confidence * decay_factor
```

### Automatic Tier Demotion

- Pattern with decayed confidence < 0.3 → demoted to Observation
- Observation with decayed confidence < 0.1 → archived
- Rules are never auto-demoted (require explicit override)

---

## Solution 4: Topic Clustering

Don't rely on agents to manually tag topics. Auto-cluster:

```python
from sklearn.cluster import HDBSCAN

def cluster_knowledge(entries: list[KnowledgeEntry]) -> dict[str, list]:
    embeddings = embed_all([e.content for e in entries])
    
    clusterer = HDBSCAN(min_cluster_size=3)
    labels = clusterer.fit_predict(embeddings)
    
    clusters = {}
    for entry, label in zip(entries, labels):
        cluster_key = f"cluster_{label}" if label >= 0 else "unclustered"
        clusters.setdefault(cluster_key, []).append(entry)
    
    return clusters
```

Re-cluster periodically (every 100 cycles). This catches when two separate topics have converged and should be merged.

---

## Scale Targets

After implementing all four solutions:

| Cycle Count | Expected Active KB Size | Why |
|-------------|------------------------|-----|
| 100 | ~80 entries | Mostly observations, few patterns |
| 500 | ~60 entries | Consolidation has merged clusters, patterns promoted |
| 1,000 | ~50 entries | Rules established, observations aggressively archived |
| 2,000 | ~40-60 entries | Stable core of rules + patterns, rotating observations |

The knowledge base should **plateau**, not grow linearly. If it's growing linearly past cycle 500, consolidation isn't aggressive enough.

---

## Stress Test Protocol

To validate the system works at scale:

1. Generate 1,000 synthetic knowledge entries (simulate cycle 1,000)
2. Run consolidation
3. Measure retrieval precision: "does the agent get the right knowledge for a given query?"
4. Compare: flat search vs hierarchical retrieval vs graph-walk
5. Key metric: **does cycle 2,000 produce measurably better output than cycle 200?**

If yes → system learns. If plateau → tune exploration/exploitation balance.
