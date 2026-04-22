"""
knowledge_base.py — Hierarchical persistent knowledge store.

Three-tier knowledge system with semantic search.

Tiers
-----
- rules        Hard constraints and proven invariants (permanent until overturned).
- patterns     Reliable heuristics (re-validated every 100 cycles).
- observations Single-cycle findings (auto-archived after 50 cycles unless promoted).

Retrieval is priority-ordered: rules → patterns → observations.
If rules answer the query sufficiently (score ≥ RULES_STOP_THRESHOLD), pattern
and observation tiers are skipped.

Embeddings are lazy-loaded: the sentence-transformers model is only imported
and loaded on first use, keeping startup fast when the knowledge base is not
needed.

Schema
------
knowledge (
    id                   TEXT PRIMARY KEY,
    content              TEXT NOT NULL,
    tier                 TEXT NOT NULL,
    confidence           REAL NOT NULL,
    created_cycle        INTEGER NOT NULL,
    last_validated_cycle INTEGER NOT NULL,
    validation_count     INTEGER NOT NULL DEFAULT 0,
    contradicted_by      TEXT NOT NULL DEFAULT '[]',   -- JSON list of entry IDs
    optuna_evidence      TEXT,                          -- JSON dict or NULL
    topic_cluster        TEXT NOT NULL,
    embedding            BLOB                           -- numpy float32 bytes
)

Usage:
    kb = KnowledgeBase()
    eid = kb.knowledge_write(
        content="Sans-serif headlines outperform serif by 23% across 200+ tests",
        tier="rule",
        confidence=0.95,
        topic_cluster="typography",
    )
    results = kb.knowledge_read("which headline font performs best?")
    kb.knowledge_promote(eid, from_tier="pattern", to_tier="rule",
                         evidence={"trial_count": 210, "mean_score": 0.88})
"""

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_TIERS: tuple[str, ...] = ("rule", "pattern", "observation")
TIER_ORDER: dict[str, int] = {"rule": 0, "pattern": 1, "observation": 2}

# Cosine similarity threshold at which rule results are "good enough" and
# lower tiers are skipped.
RULES_STOP_THRESHOLD: float = 0.75

# Sentence-transformers model name for embeddings.
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class KnowledgeEntry:
    """A single entry in the knowledge base."""

    id: str
    content: str
    tier: str  # "rule" | "pattern" | "observation"
    confidence: float  # 0.0 – 1.0
    created_cycle: int
    last_validated_cycle: int
    validation_count: int
    contradicted_by: list[str]      # list of entry IDs that contradict this one
    optuna_evidence: dict[str, Any] | None
    topic_cluster: str
    embedding: bytes | None = field(repr=False)  # raw float32 bytes (numpy)

    def embedding_array(self) -> np.ndarray | None:
        """Decode the stored embedding bytes back to a float32 numpy array."""
        if self.embedding is None:
            return None
        return np.frombuffer(self.embedding, dtype=np.float32)


# ---------------------------------------------------------------------------
# Lazy model loader
# ---------------------------------------------------------------------------

_model_instance = None  # module-level singleton


def _get_model():
    """Return the sentence-transformer model, loading it on first call."""
    global _model_instance
    if _model_instance is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for KnowledgeBase. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        _model_instance = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model_instance


def _embed(text: str) -> np.ndarray:
    """Embed a single text string, returning a float32 numpy array."""
    model = _get_model()
    vec = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype(np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two unit-norm vectors.

    Because sentence-transformers returns normalised embeddings by default
    (normalize_embeddings=True), this is just the dot product.
    """
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# KnowledgeBase
# ---------------------------------------------------------------------------


class KnowledgeBase:
    """
    Hierarchical persistent knowledge store backed by SQLite.

    Embeddings are computed lazily; instantiating KnowledgeBase does NOT
    load sentence-transformers. The model loads only on the first
    knowledge_read() or knowledge_write() call.
    """

    def __init__(self, db_path: str | Path = "knowledge.db") -> None:
        """
        Initialise the knowledge base.

        Args:
            db_path: Path to the SQLite database. Defaults to ``knowledge.db``
                     in the current working directory. Parent directories are
                     created automatically.
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = self._connect()
        self._bootstrap()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            timeout=5.0,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _bootstrap(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS knowledge (
                    id                   TEXT PRIMARY KEY,
                    content              TEXT    NOT NULL,
                    tier                 TEXT    NOT NULL,
                    confidence           REAL    NOT NULL,
                    created_cycle        INTEGER NOT NULL,
                    last_validated_cycle INTEGER NOT NULL,
                    validation_count     INTEGER NOT NULL DEFAULT 0,
                    contradicted_by      TEXT    NOT NULL DEFAULT '[]',
                    optuna_evidence      TEXT,
                    topic_cluster        TEXT    NOT NULL,
                    embedding            BLOB,
                    created_at           REAL    NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_knowledge_tier
                    ON knowledge(tier);
                CREATE INDEX IF NOT EXISTS idx_knowledge_topic
                    ON knowledge(topic_cluster);
                """
            )

    def _row_to_entry(self, row: sqlite3.Row) -> KnowledgeEntry:
        return KnowledgeEntry(
            id=row["id"],
            content=row["content"],
            tier=row["tier"],
            confidence=row["confidence"],
            created_cycle=row["created_cycle"],
            last_validated_cycle=row["last_validated_cycle"],
            validation_count=row["validation_count"],
            contradicted_by=json.loads(row["contradicted_by"]),
            optuna_evidence=json.loads(row["optuna_evidence"])
            if row["optuna_evidence"]
            else None,
            topic_cluster=row["topic_cluster"],
            embedding=row["embedding"],
        )

    def _fetch_tier(self, tier: str) -> list[KnowledgeEntry]:
        """Fetch all entries for a given tier (with embeddings)."""
        rows = self._conn.execute(
            "SELECT * FROM knowledge WHERE tier = ? ORDER BY confidence DESC",
            (tier,),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def _rank_entries(
        self,
        query_vec: np.ndarray,
        entries: list[KnowledgeEntry],
        max_results: int,
    ) -> list[tuple[float, KnowledgeEntry]]:
        """
        Rank entries by cosine similarity to query_vec.

        Entries without an embedding are scored as 0.0 and ranked last.

        Returns:
            List of (score, entry) sorted descending by score, capped at max_results.
        """
        scored: list[tuple[float, KnowledgeEntry]] = []
        for entry in entries:
            emb = entry.embedding_array()
            if emb is None:
                score = 0.0
            else:
                score = _cosine_similarity(query_vec, emb)
            scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:max_results]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def knowledge_write(
        self,
        content: str,
        tier: str,
        confidence: float,
        topic_cluster: str,
        cycle: int = 0,
        optuna_evidence: dict[str, Any] | None = None,
    ) -> str:
        """
        Write a new knowledge entry.

        Computes and stores a semantic embedding for the content.

        Args:
            content: The knowledge text.
            tier: One of ``"rule"``, ``"pattern"``, ``"observation"``.
            confidence: Confidence score in [0.0, 1.0].
            topic_cluster: Cluster label for grouping (e.g. ``"typography"``).
            cycle: The current production cycle number.
            optuna_evidence: Optional dict of statistical evidence from Optuna.

        Returns:
            The new entry's UUID string.

        Raises:
            ValueError: If tier is not one of the valid values.
        """
        if tier not in VALID_TIERS:
            raise ValueError(
                f"Invalid tier {tier!r}. Must be one of {VALID_TIERS}."
            )
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {confidence}.")

        entry_id = str(uuid.uuid4())
        embedding_vec = _embed(content)
        embedding_bytes = embedding_vec.tobytes()

        with self._conn:
            self._conn.execute(
                """
                INSERT INTO knowledge (
                    id, content, tier, confidence,
                    created_cycle, last_validated_cycle, validation_count,
                    contradicted_by, optuna_evidence, topic_cluster,
                    embedding, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 0, '[]', ?, ?, ?, ?)
                """,
                (
                    entry_id,
                    content,
                    tier,
                    confidence,
                    cycle,
                    cycle,
                    json.dumps(optuna_evidence) if optuna_evidence else None,
                    topic_cluster,
                    embedding_bytes,
                    time.time(),
                ),
            )

        return entry_id

    def knowledge_read(
        self,
        query: str,
        tier: str | None = None,
        max_results: int = 5,
    ) -> list[KnowledgeEntry]:
        """
        Semantic search over the knowledge base.

        Retrieval is priority-ordered: rules → patterns → observations.
        If searching all tiers and rule results exceed RULES_STOP_THRESHOLD,
        lower tiers are skipped (stop-early behaviour).

        Args:
            query: Natural language query to embed and match against.
            tier: If set, restrict results to this tier only
                  (``"rule"``, ``"pattern"``, or ``"observation"``).
            max_results: Maximum number of entries to return.

        Returns:
            List of KnowledgeEntry ordered by relevance (highest first).

        Raises:
            ValueError: If an invalid tier name is provided.
        """
        if tier is not None and tier not in VALID_TIERS:
            raise ValueError(
                f"Invalid tier {tier!r}. Must be one of {VALID_TIERS}."
            )

        query_vec = _embed(query)

        if tier is not None:
            # Single-tier restricted search.
            entries = self._fetch_tier(tier)
            ranked = self._rank_entries(query_vec, entries, max_results)
            return [e for _, e in ranked]

        # Multi-tier priority search: rules → patterns → observations.
        results: list[KnowledgeEntry] = []
        remaining = max_results

        for search_tier in ("rule", "pattern", "observation"):
            if remaining <= 0:
                break
            entries = self._fetch_tier(search_tier)
            if not entries:
                continue
            ranked = self._rank_entries(query_vec, entries, remaining)
            tier_results = [(score, entry) for score, entry in ranked]

            results.extend(e for _, e in tier_results)
            remaining -= len(tier_results)

            # Stop-early: if any rule result is highly relevant, skip lower tiers.
            if search_tier == "rule" and tier_results:
                best_score = tier_results[0][0]
                if best_score >= RULES_STOP_THRESHOLD:
                    break

        return results

    def knowledge_promote(
        self,
        entry_id: str,
        from_tier: str,
        to_tier: str,
        evidence: dict[str, Any],
    ) -> None:
        """
        Promote a knowledge entry from one tier to a higher one.

        Updates the tier, appends the promotion evidence to optuna_evidence,
        increments validation_count, and records the current time as
        last_validated_cycle (using the evidence dict's ``cycle`` key if
        present, else 0).

        Args:
            entry_id: UUID of the entry to promote.
            from_tier: The entry's current tier (validated for safety).
            to_tier: The target tier (must be higher priority than from_tier).
            evidence: Dict of promotion evidence (e.g. Optuna statistics).

        Raises:
            ValueError: If tier ordering is wrong or from_tier doesn't match DB.
            LookupError: If entry_id is not found.
        """
        if from_tier not in VALID_TIERS or to_tier not in VALID_TIERS:
            raise ValueError(
                f"Invalid tier. Must be one of {VALID_TIERS}."
            )
        if TIER_ORDER[to_tier] >= TIER_ORDER[from_tier]:
            raise ValueError(
                f"to_tier {to_tier!r} must have higher priority than "
                f"from_tier {from_tier!r}. Priority order: rule > pattern > observation."
            )

        row = self._conn.execute(
            "SELECT * FROM knowledge WHERE id = ?", (entry_id,)
        ).fetchone()

        if row is None:
            raise LookupError(f"No knowledge entry with id={entry_id!r}.")

        if row["tier"] != from_tier:
            raise ValueError(
                f"Entry {entry_id!r} is in tier {row['tier']!r}, "
                f"not {from_tier!r}."
            )

        # Merge promotion evidence with existing optuna_evidence.
        existing_evidence: dict[str, Any] = (
            json.loads(row["optuna_evidence"]) if row["optuna_evidence"] else {}
        )
        existing_evidence.update(evidence)

        cycle = evidence.get("cycle", 0)
        new_validation_count = row["validation_count"] + 1

        with self._conn:
            self._conn.execute(
                """
                UPDATE knowledge
                SET tier = ?,
                    optuna_evidence = ?,
                    last_validated_cycle = ?,
                    validation_count = ?
                WHERE id = ?
                """,
                (
                    to_tier,
                    json.dumps(existing_evidence),
                    cycle,
                    new_validation_count,
                    entry_id,
                ),
            )

    def knowledge_flag_contradiction(
        self, entry_id: str, contradicted_by_id: str
    ) -> None:
        """
        Flag that entry_id is contradicted by contradicted_by_id.

        Appends contradicted_by_id to the entry's contradicted_by list
        without overwriting existing flags.

        Args:
            entry_id: The entry being flagged.
            contradicted_by_id: The entry that contradicts it.

        Raises:
            LookupError: If entry_id is not found.
        """
        row = self._conn.execute(
            "SELECT contradicted_by FROM knowledge WHERE id = ?", (entry_id,)
        ).fetchone()

        if row is None:
            raise LookupError(f"No knowledge entry with id={entry_id!r}.")

        existing: list[str] = json.loads(row["contradicted_by"])
        if contradicted_by_id not in existing:
            existing.append(contradicted_by_id)

        with self._conn:
            self._conn.execute(
                "UPDATE knowledge SET contradicted_by = ? WHERE id = ?",
                (json.dumps(existing), entry_id),
            )

    def knowledge_validate(self, entry_id: str, cycle: int) -> None:
        """
        Record a validation event for an entry (increments count, updates cycle).

        Args:
            entry_id: The entry being validated.
            cycle: The current production cycle.

        Raises:
            LookupError: If entry_id is not found.
        """
        with self._conn:
            result = self._conn.execute(
                """
                UPDATE knowledge
                SET validation_count = validation_count + 1,
                    last_validated_cycle = ?
                WHERE id = ?
                """,
                (cycle, entry_id),
            )
        if result.rowcount == 0:
            raise LookupError(f"No knowledge entry with id={entry_id!r}.")

    def knowledge_get(self, entry_id: str) -> KnowledgeEntry | None:
        """
        Fetch a single entry by ID.

        Args:
            entry_id: The UUID of the entry.

        Returns:
            KnowledgeEntry if found, else None.
        """
        row = self._conn.execute(
            "SELECT * FROM knowledge WHERE id = ?", (entry_id,)
        ).fetchone()
        return self._row_to_entry(row) if row else None

    def knowledge_list(
        self,
        tier: str | None = None,
        topic_cluster: str | None = None,
    ) -> list[KnowledgeEntry]:
        """
        List all entries, optionally filtered by tier and/or topic_cluster.

        Args:
            tier: If provided, restrict to this tier.
            topic_cluster: If provided, restrict to this cluster.

        Returns:
            List of KnowledgeEntry ordered by tier priority then confidence.
        """
        query = "SELECT * FROM knowledge WHERE 1=1"
        params: list[Any] = []

        if tier is not None:
            query += " AND tier = ?"
            params.append(tier)
        if topic_cluster is not None:
            query += " AND topic_cluster = ?"
            params.append(topic_cluster)

        query += " ORDER BY confidence DESC"

        rows = self._conn.execute(query, params).fetchall()
        entries = [self._row_to_entry(r) for r in rows]

        # Sort by tier priority, then confidence descending.
        entries.sort(key=lambda e: (TIER_ORDER[e.tier], -e.confidence))
        return entries

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    def __repr__(self) -> str:
        counts = {
            tier: self._conn.execute(
                "SELECT COUNT(*) FROM knowledge WHERE tier = ?", (tier,)
            ).fetchone()[0]
            for tier in VALID_TIERS
        }
        return (
            f"KnowledgeBase(db={self._db_path!r}, "
            f"rules={counts['rule']}, patterns={counts['pattern']}, "
            f"observations={counts['observation']})"
        )
