"""
knowledge_store.py — Upgraded knowledge store with sqlite-vec vector search.

Replaces / supplements KnowledgeBase with:
  - sqlite-vec vec0 virtual tables for SQL-native KNN search
  - nomic-embed-text-v1.5 with mandatory task prefixes
  - Matryoshka two-stage retrieval (coarse 256-dim → rerank 768-dim)
  - Semantic deduplication at ingestion (cosine > 0.95 → skip)
  - Ebbinghaus-inspired confidence decay
  - Hard 10K item cap with LRU eviction of lowest-confidence archived entries
  - Tier-aware retrieval: rules → patterns → observations
  - Graceful fallback to numpy cosine similarity if sqlite-vec is unavailable

Embedding model is lazy-loaded; instantiating KnowledgeStore does NOT
import sentence-transformers until the first read/write call.

Architecture
------------
Two SQLite tables work together:

  knowledge (regular table)
  ├─ id                   TEXT PRIMARY KEY
  ├─ content              TEXT NOT NULL
  ├─ tier                 TEXT NOT NULL  (rule|pattern|observation|archived)
  ├─ confidence           REAL NOT NULL  (0.0–1.0)
  ├─ created_cycle        INTEGER
  ├─ last_validated_cycle INTEGER
  ├─ validation_count     INTEGER
  ├─ contradicted_by      TEXT  (JSON list of IDs)
  ├─ optuna_evidence      TEXT  (JSON dict or NULL)
  ├─ topic_cluster        TEXT
  ├─ last_accessed_at     REAL  (Unix timestamp, for LRU eviction)
  └─ created_at           REAL  (Unix timestamp)

  knowledge_vecs (vec0 virtual table)
  ├─ rowid                INTEGER  (matches knowledge.rowid)
  └─ embedding            float[768]  (full-dimension, normalised)

When sqlite-vec is unavailable the vec0 table is omitted and embeddings are
stored as BLOB (float32 bytes) directly in the knowledge table, falling back
to in-process numpy dot-product search.

Usage
-----
    store = KnowledgeStore()
    eid = store.write(
        content="Sans-serif headlines beat serif by 23% across 200+ tests",
        tier="rule",
        confidence=0.95,
        topic_cluster="typography",
    )
    results = store.read("which headline font performs best?")
    store.decay_all(current_cycle=42)
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.memory.knowledge_base import KnowledgeBase

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_TIERS: tuple[str, ...] = ("rule", "pattern", "observation", "archived")
ACTIVE_TIERS: tuple[str, ...] = ("rule", "pattern", "observation")
TIER_ORDER: dict[str, int] = {
    "rule": 0,
    "pattern": 1,
    "observation": 2,
    "archived": 3,
}

# Embedding model and dimension config
NOMIC_MODEL_NAME: str = "nomic-ai/nomic-embed-text-v1.5"
FALLBACK_MODEL_NAME: str = "all-MiniLM-L6-v2"  # 384-dim, used if nomic unavailable
FULL_DIM: int = 768          # nomic full dimension (stored in DB)
COARSE_DIM: int = 256        # truncated for fast KNN pass
FALLBACK_DIM: int = 384      # MiniLM dimension (fallback mode)

# Retrieval thresholds
RULES_STOP_THRESHOLD: float = 0.75    # if top rule scores this, skip lower tiers
COARSE_CANDIDATES: int = 20           # how many coarse KNN hits to rerank
DEDUP_COSINE_THRESHOLD: float = 0.95  # above this → duplicate, skip write

# Capacity
MAX_KNOWLEDGE_ITEMS: int = 10_000
EVICTION_BATCH: int = 100  # how many to evict when cap is hit

# Ebbinghaus decay params
EBBINGHAUS_STABILITY: float = 20.0   # cycles; higher = slower decay
EBBINGHAUS_MIN: float = 0.05         # floor confidence after decay


# ---------------------------------------------------------------------------
# Lazy model loader
# ---------------------------------------------------------------------------

_model_instance = None
_model_dim: int | None = None


def _get_model() -> tuple[Any, int]:
    """Return (model, embedding_dim), loading on first call."""
    global _model_instance, _model_dim
    if _model_instance is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            _model_instance = SentenceTransformer(
                NOMIC_MODEL_NAME,
                trust_remote_code=True,
            )
            _model_dim = FULL_DIM
        except Exception:
            # Fallback: MiniLM (smaller, no task prefixes needed)
            from sentence_transformers import SentenceTransformer  # type: ignore
            _model_instance = SentenceTransformer(FALLBACK_MODEL_NAME)
            _model_dim = FALLBACK_DIM
    return _model_instance, _model_dim


def _embed(text: str, task: str = "search_document") -> np.ndarray:
    """
    Embed text with task prefix (nomic) or plain (fallback).

    task options: "search_document", "search_query", "clustering", "classification"

    Returns float32 unit-norm ndarray.
    """
    model, dim = _get_model()
    if dim == FULL_DIM:
        # nomic requires task prefix
        prefixed = f"{task}: {text}"
    else:
        prefixed = text  # MiniLM doesn't use prefixes
    vec = model.encode(prefixed, normalize_embeddings=True, convert_to_numpy=True)
    return vec.astype(np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two unit-norm vectors (= dot product)."""
    return float(np.dot(a, b))


def _ebbinghaus_decay(confidence: float, cycles_elapsed: int) -> float:
    """
    Ebbinghaus-inspired forgetting curve decay.

    R = confidence * exp(-cycles / stability)

    Capped at EBBINGHAUS_MIN floor.
    """
    decayed = confidence * math.exp(-max(0, cycles_elapsed) / EBBINGHAUS_STABILITY)
    return max(EBBINGHAUS_MIN, decayed)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class StoreEntry:
    """A knowledge entry as returned by KnowledgeStore.read()."""

    id: str
    content: str
    tier: str
    confidence: float
    created_cycle: int
    last_validated_cycle: int
    validation_count: int
    contradicted_by: list[str]
    optuna_evidence: dict[str, Any] | None
    topic_cluster: str
    created_at: float
    last_accessed_at: float
    score: float = 0.0  # similarity score from retrieval (not persisted)

    def __repr__(self) -> str:
        s = self.content[:60] + ("…" if len(self.content) > 60 else "")
        return (
            f"StoreEntry(id={self.id[:8]!r}, tier={self.tier!r}, "
            f"conf={self.confidence:.2f}, score={self.score:.3f}, "
            f"content={s!r})"
        )


# ---------------------------------------------------------------------------
# KnowledgeStore
# ---------------------------------------------------------------------------


class KnowledgeStore:
    """
    Upgraded knowledge store with sqlite-vec KNN search.

    Falls back to numpy cosine similarity if sqlite-vec is not available
    or cannot load the extension (macOS default Python restriction).
    """

    def __init__(
        self,
        db_path: str | Path = "knowledge_store.db",
        use_vec: bool = True,
    ) -> None:
        """
        Initialise the knowledge store.

        Args:
            db_path: Path to the SQLite database.
            use_vec: If True, attempt to use sqlite-vec. Falls back to numpy
                     automatically on failure.
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._use_vec = False  # set properly in _init_*
        self._vec_dim: int = FULL_DIM

        self._conn = self._connect()
        self._bootstrap_base_tables()

        if use_vec:
            try:
                self._init_vec()
                self._use_vec = True
            except Exception as e:
                # Silently degrade; vec table won't exist, numpy path used
                self._use_vec = False
                self._bootstrap_numpy_fallback_column()
        else:
            self._bootstrap_numpy_fallback_column()

    # ------------------------------------------------------------------
    # Connection & schema bootstrap
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            timeout=10.0,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _bootstrap_base_tables(self) -> None:
        with self._conn:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS knowledge (
                    id                   TEXT    PRIMARY KEY,
                    rowid_int            INTEGER,          -- matches vec0 rowid
                    content              TEXT    NOT NULL,
                    tier                 TEXT    NOT NULL DEFAULT 'observation',
                    confidence           REAL    NOT NULL DEFAULT 0.5,
                    created_cycle        INTEGER NOT NULL DEFAULT 0,
                    last_validated_cycle INTEGER NOT NULL DEFAULT 0,
                    validation_count     INTEGER NOT NULL DEFAULT 0,
                    contradicted_by      TEXT    NOT NULL DEFAULT '[]',
                    optuna_evidence      TEXT,
                    topic_cluster        TEXT    NOT NULL DEFAULT 'unclustered',
                    created_at           REAL    NOT NULL,
                    last_accessed_at     REAL    NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_ks_tier
                    ON knowledge(tier);
                CREATE INDEX IF NOT EXISTS idx_ks_confidence
                    ON knowledge(confidence DESC);
                CREATE INDEX IF NOT EXISTS idx_ks_rowid_int
                    ON knowledge(rowid_int);
                CREATE INDEX IF NOT EXISTS idx_ks_last_accessed
                    ON knowledge(last_accessed_at);

                -- Sequence table for generating integer rowids that match vec0
                CREATE TABLE IF NOT EXISTS _rowid_seq (
                    id INTEGER PRIMARY KEY AUTOINCREMENT
                );
            """)

    def _init_vec(self) -> None:
        """Load sqlite-vec extension and create vec0 virtual table."""
        import sqlite_vec  # type: ignore

        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

        _, dim = _get_model()
        self._vec_dim = dim

        with self._conn:
            self._conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_vecs USING vec0(
                    embedding float[{self._vec_dim}]
                )
            """)

    def _bootstrap_numpy_fallback_column(self) -> None:
        """Add embedding BLOB column to knowledge table for numpy fallback."""
        existing = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(knowledge)").fetchall()
        }
        if "embedding" not in existing:
            with self._conn:
                self._conn.execute(
                    "ALTER TABLE knowledge ADD COLUMN embedding BLOB"
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_rowid(self) -> int:
        """Generate a new integer rowid (autoincrement sequence)."""
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO _rowid_seq DEFAULT VALUES"
            )
        return cur.lastrowid

    def _row_to_entry(self, row: sqlite3.Row, score: float = 0.0) -> StoreEntry:
        return StoreEntry(
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
            created_at=row["created_at"],
            last_accessed_at=row["last_accessed_at"],
            score=score,
        )

    def _count_active(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM knowledge WHERE tier != 'archived'"
        ).fetchone()[0]

    def _count_total(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM knowledge"
        ).fetchone()[0]

    def _evict_if_needed(self) -> None:
        """
        Enforce the 10K item hard cap.

        Evicts EVICTION_BATCH lowest-confidence archived entries first,
        then lowest-confidence observations, then patterns, then rules
        (last resort).  LRU tiebreaker (last_accessed_at).
        """
        total = self._count_total()
        if total < MAX_KNOWLEDGE_ITEMS:
            return

        to_evict = total - MAX_KNOWLEDGE_ITEMS + EVICTION_BATCH

        # Priority eviction: archived first, then by tier ascending confidence
        eviction_candidates = self._conn.execute(
            """
            SELECT id, rowid_int FROM knowledge
            ORDER BY
                CASE tier
                    WHEN 'archived'     THEN 0
                    WHEN 'observation'  THEN 1
                    WHEN 'pattern'      THEN 2
                    WHEN 'rule'         THEN 3
                END ASC,
                confidence ASC,
                last_accessed_at ASC
            LIMIT ?
            """,
            (to_evict,),
        ).fetchall()

        for row in eviction_candidates:
            self._delete_entry(row["id"], row["rowid_int"])

    def _delete_entry(self, entry_id: str, rowid_int: int | None) -> None:
        """Hard-delete an entry from both tables."""
        with self._conn:
            self._conn.execute(
                "DELETE FROM knowledge WHERE id = ?", (entry_id,)
            )
            if self._use_vec and rowid_int is not None:
                try:
                    self._conn.execute(
                        "DELETE FROM knowledge_vecs WHERE rowid = ?",
                        (rowid_int,),
                    )
                except Exception:
                    pass

    def _is_duplicate(self, embedding: np.ndarray) -> bool:
        """
        Return True if a semantically similar entry (cosine > 0.95) exists.

        Uses sqlite-vec KNN with 1 neighbour when available, otherwise
        falls back to iterating over all stored embeddings.
        """
        if self._use_vec:
            # L2 distance on unit-norm: cos_sim ≈ 1 - dist²/2
            results = self._conn.execute(
                """
                SELECT rowid, distance
                FROM knowledge_vecs
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT 1
                """,
                (embedding,),
            ).fetchall()
            if not results:
                return False
            dist = results[0]["distance"]
            cos_sim = 1.0 - (dist ** 2) / 2.0
            return cos_sim >= DEDUP_COSINE_THRESHOLD

        # Numpy fallback: scan all embeddings
        rows = self._conn.execute(
            "SELECT embedding FROM knowledge WHERE embedding IS NOT NULL"
        ).fetchall()
        for row in rows:
            stored = np.frombuffer(row["embedding"], dtype=np.float32)
            if _cosine(embedding, stored) >= DEDUP_COSINE_THRESHOLD:
                return True
        return False

    # ------------------------------------------------------------------
    # Two-stage retrieval helpers
    # ------------------------------------------------------------------

    def _vec_knn_coarse(
        self,
        query_emb: np.ndarray,
        tier: str | None,
        n: int,
    ) -> list[tuple[int, float]]:
        """
        Coarse KNN at COARSE_DIM using sqlite-vec.

        Returns list of (rowid_int, distance).
        """
        _, dim = _get_model()
        if dim == FULL_DIM:
            coarse_emb = query_emb[:COARSE_DIM].copy()
            norm = np.linalg.norm(coarse_emb)
            if norm > 0:
                coarse_emb = coarse_emb / norm
        else:
            coarse_emb = query_emb  # fallback model; no truncation

        results = self._conn.execute(
            """
            SELECT rowid, distance
            FROM knowledge_vecs
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
            """,
            (coarse_emb, n),
        ).fetchall()
        return [(r["rowid"], r["distance"]) for r in results]

    def _vec_rerank(
        self,
        query_emb: np.ndarray,
        rowid_ints: list[int],
        tier: str | None,
    ) -> list[tuple[float, StoreEntry]]:
        """
        Rerank a set of candidates using full-dimension cosine similarity.

        Fetches stored embeddings from knowledge_vecs and re-scores against
        the full query_emb.  If a tier filter is specified, entries of the
        wrong tier are dropped.

        Returns list of (score, StoreEntry) sorted descending.
        """
        if not rowid_ints:
            return []

        placeholders = ", ".join("?" * len(rowid_ints))

        # Get metadata from knowledge table
        tier_clause = f"AND tier = '{tier}'" if tier else ""
        rows = self._conn.execute(
            f"""
            SELECT k.*
            FROM knowledge k
            WHERE k.rowid_int IN ({placeholders})
              {tier_clause}
            """,
            rowid_ints,
        ).fetchall()

        if not rows:
            return []

        # Get the stored embeddings from knowledge_vecs for full rerank
        vec_rows = self._conn.execute(
            f"""
            SELECT rowid, embedding
            FROM knowledge_vecs
            WHERE rowid IN ({placeholders})
            """,
            rowid_ints,
        ).fetchall()
        vec_by_rowid: dict[int, np.ndarray] = {
            r["rowid"]: np.frombuffer(bytes(r["embedding"]), dtype=np.float32)
            for r in vec_rows
        }

        entry_by_rowid = {r["rowid_int"]: r for r in rows}
        scored = []
        for rowid_int, stored_emb in vec_by_rowid.items():
            if rowid_int not in entry_by_rowid:
                continue
            score = _cosine(query_emb, stored_emb)
            entry = self._row_to_entry(entry_by_rowid[rowid_int], score=score)
            scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def _numpy_search(
        self,
        query_emb: np.ndarray,
        tier: str | None,
        n: int,
    ) -> list[tuple[float, StoreEntry]]:
        """Fallback numpy brute-force cosine search."""
        tier_clause = f"AND tier = '{tier}'" if tier else ""
        rows = self._conn.execute(
            f"""
            SELECT * FROM knowledge
            WHERE embedding IS NOT NULL
              AND tier != 'archived'
              {tier_clause}
            """
        ).fetchall()

        scored = []
        for row in rows:
            stored = np.frombuffer(row["embedding"], dtype=np.float32)
            score = _cosine(query_emb, stored)
            entry = self._row_to_entry(row, score=score)
            scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:n]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(
        self,
        content: str,
        tier: str = "observation",
        confidence: float = 0.5,
        topic_cluster: str = "unclustered",
        cycle: int = 0,
        optuna_evidence: dict[str, Any] | None = None,
        skip_dedup: bool = False,
    ) -> str | None:
        """
        Write a new knowledge entry.

        Performs semantic deduplication before writing: if an entry with
        cosine similarity ≥ 0.95 already exists, the write is skipped and
        None is returned.

        Enforces the 10K item cap via LRU eviction.

        Args:
            content: The knowledge text.
            tier: One of "rule", "pattern", "observation".
            confidence: Float in [0.0, 1.0].
            topic_cluster: Cluster label (e.g. "typography").
            cycle: Current optimisation cycle.
            optuna_evidence: Optional statistical evidence dict.
            skip_dedup: If True, bypass deduplication check.

        Returns:
            UUID string of the new entry, or None if deduplicated.

        Raises:
            ValueError: If tier or confidence is invalid.
        """
        if tier not in ACTIVE_TIERS:
            raise ValueError(
                f"Invalid tier {tier!r}. Must be one of {ACTIVE_TIERS}."
            )
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {confidence}.")

        # Compute embedding with document prefix
        embedding = _embed(content, task="search_document")

        # Semantic deduplication
        if not skip_dedup and self._is_duplicate(embedding):
            return None

        # Eviction if at capacity
        self._evict_if_needed()

        entry_id = str(uuid.uuid4())
        now = time.time()
        rowid_int = self._next_rowid()

        with self._conn:
            self._conn.execute(
                """
                INSERT INTO knowledge (
                    id, rowid_int, content, tier, confidence,
                    created_cycle, last_validated_cycle, validation_count,
                    contradicted_by, optuna_evidence, topic_cluster,
                    created_at, last_accessed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, '[]', ?, ?, ?, ?)
                """,
                (
                    entry_id,
                    rowid_int,
                    content,
                    tier,
                    confidence,
                    cycle,
                    cycle,
                    json.dumps(optuna_evidence) if optuna_evidence else None,
                    topic_cluster,
                    now,
                    now,
                ),
            )

            if self._use_vec:
                self._conn.execute(
                    "INSERT INTO knowledge_vecs(rowid, embedding) VALUES (?, ?)",
                    (rowid_int, embedding),
                )
            else:
                self._conn.execute(
                    "UPDATE knowledge SET embedding = ? WHERE id = ?",
                    (embedding.tobytes(), entry_id),
                )

        return entry_id

    def read(
        self,
        query: str,
        tier: str | None = None,
        max_results: int = 5,
    ) -> list[StoreEntry]:
        """
        Semantic search with tier-aware priority ordering.

        When no tier filter is given:
          1. Search rules first; if top score ≥ RULES_STOP_THRESHOLD → stop.
          2. Then patterns; if enough results → stop.
          3. Then observations.

        Two-stage retrieval (sqlite-vec mode):
          - Coarse KNN at 256 dims (fast)
          - Rerank top-COARSE_CANDIDATES at full 768 dims (precise)

        Args:
            query: Natural-language search query.
            tier: Optional tier filter ("rule", "pattern", "observation").
            max_results: Max entries to return.

        Returns:
            List of StoreEntry ordered by relevance (highest first).
        """
        if tier is not None and tier not in VALID_TIERS:
            raise ValueError(
                f"Invalid tier {tier!r}. Must be one of {VALID_TIERS}."
            )

        query_emb = _embed(query, task="search_query")

        # Update last_accessed_at for returned entries (batch at end)
        accessed_ids: list[str] = []

        if tier is not None:
            results = self._search_tier(query_emb, tier, max_results)
            accessed_ids = [e.id for _, e in results]
            self._touch(accessed_ids)
            return [e for _, e in results[:max_results]]

        # Multi-tier priority search
        final: list[StoreEntry] = []
        remaining = max_results

        for search_tier in ("rule", "pattern", "observation"):
            if remaining <= 0:
                break

            tier_results = self._search_tier(query_emb, search_tier, remaining)
            if not tier_results:
                continue

            for score, entry in tier_results[:remaining]:
                final.append(entry)
                accessed_ids.append(entry.id)
            remaining -= len(tier_results[:remaining])

            # Stop-early on highly relevant rules
            if search_tier == "rule" and tier_results:
                best_score = tier_results[0][0]
                if best_score >= RULES_STOP_THRESHOLD:
                    break

        self._touch(accessed_ids)
        return final

    def _search_tier(
        self,
        query_emb: np.ndarray,
        tier: str,
        n: int,
    ) -> list[tuple[float, StoreEntry]]:
        """Search a single tier, returning (score, entry) pairs."""
        if self._use_vec:
            # Coarse KNN → rerank
            coarse_hits = self._vec_knn_coarse(query_emb, tier, COARSE_CANDIDATES)
            rowid_ints = [r for r, _ in coarse_hits]
            reranked = self._vec_rerank(query_emb, rowid_ints, tier)
            return reranked[:n]
        else:
            return self._numpy_search(query_emb, tier, n)

    def _touch(self, entry_ids: list[str]) -> None:
        """Update last_accessed_at for a list of entry IDs."""
        if not entry_ids:
            return
        now = time.time()
        with self._conn:
            self._conn.executemany(
                "UPDATE knowledge SET last_accessed_at = ? WHERE id = ?",
                [(now, eid) for eid in entry_ids],
            )

    def get(self, entry_id: str) -> StoreEntry | None:
        """Fetch a single entry by UUID."""
        row = self._conn.execute(
            "SELECT * FROM knowledge WHERE id = ?", (entry_id,)
        ).fetchone()
        return self._row_to_entry(row) if row else None

    def promote(
        self,
        entry_id: str,
        from_tier: str,
        to_tier: str,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        """
        Promote an entry to a higher tier.

        Args:
            entry_id: UUID of the entry.
            from_tier: Expected current tier (validated).
            to_tier: Target tier (must be higher priority).
            evidence: Optional dict merged into optuna_evidence.

        Raises:
            ValueError: Tier ordering invalid or tier mismatch.
            LookupError: Entry not found.
        """
        if TIER_ORDER.get(to_tier, 99) >= TIER_ORDER.get(from_tier, 99):
            raise ValueError(
                f"to_tier {to_tier!r} must be higher priority than "
                f"from_tier {from_tier!r}."
            )

        row = self._conn.execute(
            "SELECT * FROM knowledge WHERE id = ?", (entry_id,)
        ).fetchone()
        if row is None:
            raise LookupError(f"No entry with id={entry_id!r}.")
        if row["tier"] != from_tier:
            raise ValueError(
                f"Entry is in tier {row['tier']!r}, expected {from_tier!r}."
            )

        existing_ev = json.loads(row["optuna_evidence"]) if row["optuna_evidence"] else {}
        if evidence:
            existing_ev.update(evidence)
        cycle = (evidence or {}).get("cycle", row["last_validated_cycle"])

        with self._conn:
            self._conn.execute(
                """
                UPDATE knowledge
                SET tier = ?,
                    optuna_evidence = ?,
                    last_validated_cycle = ?,
                    validation_count = validation_count + 1
                WHERE id = ?
                """,
                (to_tier, json.dumps(existing_ev) if existing_ev else None, cycle, entry_id),
            )

    def archive(self, entry_id: str) -> None:
        """Demote an entry to archived tier (excluded from active retrieval)."""
        with self._conn:
            self._conn.execute(
                "UPDATE knowledge SET tier = 'archived' WHERE id = ?",
                (entry_id,),
            )

    def validate(self, entry_id: str, cycle: int) -> None:
        """Record a validation event (increments count, refreshes cycle)."""
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
            raise LookupError(f"No entry with id={entry_id!r}.")

    def decay_all(self, current_cycle: int) -> int:
        """
        Apply Ebbinghaus confidence decay to all active entries.

        Entries not validated recently lose confidence exponentially.
        Entries that fall below EBBINGHAUS_MIN are automatically archived.

        Returns:
            Number of entries archived due to low confidence.
        """
        rows = self._conn.execute(
            """
            SELECT id, confidence, last_validated_cycle
            FROM knowledge
            WHERE tier != 'archived'
            """
        ).fetchall()

        archived_count = 0
        updates = []
        archive_ids = []

        for row in rows:
            elapsed = current_cycle - row["last_validated_cycle"]
            new_conf = _ebbinghaus_decay(row["confidence"], elapsed)
            if new_conf <= EBBINGHAUS_MIN:
                archive_ids.append(row["id"])
                archived_count += 1
            else:
                updates.append((new_conf, row["id"]))

        with self._conn:
            if updates:
                self._conn.executemany(
                    "UPDATE knowledge SET confidence = ? WHERE id = ?",
                    updates,
                )
            for eid in archive_ids:
                self._conn.execute(
                    "UPDATE knowledge SET tier = 'archived', confidence = ? WHERE id = ?",
                    (EBBINGHAUS_MIN, eid),
                )

        return archived_count

    def list_entries(
        self,
        tier: str | None = None,
        topic_cluster: str | None = None,
        include_archived: bool = False,
    ) -> list[StoreEntry]:
        """List all entries with optional filters."""
        query = "SELECT * FROM knowledge WHERE 1=1"
        params: list[Any] = []

        if tier is not None:
            query += " AND tier = ?"
            params.append(tier)
        elif not include_archived:
            query += " AND tier != 'archived'"

        if topic_cluster is not None:
            query += " AND topic_cluster = ?"
            params.append(topic_cluster)

        query += " ORDER BY confidence DESC"
        rows = self._conn.execute(query, params).fetchall()
        entries = [self._row_to_entry(r) for r in rows]

        # Sort by tier priority then confidence
        entries.sort(key=lambda e: (TIER_ORDER.get(e.tier, 99), -e.confidence))
        return entries

    def flag_contradiction(self, entry_id: str, contradicted_by_id: str) -> None:
        """Flag that entry_id is contradicted by contradicted_by_id."""
        row = self._conn.execute(
            "SELECT contradicted_by FROM knowledge WHERE id = ?", (entry_id,)
        ).fetchone()
        if row is None:
            raise LookupError(f"No entry with id={entry_id!r}.")

        existing: list[str] = json.loads(row["contradicted_by"])
        if contradicted_by_id not in existing:
            existing.append(contradicted_by_id)

        with self._conn:
            self._conn.execute(
                "UPDATE knowledge SET contradicted_by = ? WHERE id = ?",
                (json.dumps(existing), entry_id),
            )

    @property
    def using_vec(self) -> bool:
        """True if sqlite-vec is active; False if numpy fallback is in use."""
        return self._use_vec

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    def __repr__(self) -> str:
        mode = "sqlite-vec" if self._use_vec else "numpy-fallback"
        counts = {
            t: self._conn.execute(
                "SELECT COUNT(*) FROM knowledge WHERE tier = ?", (t,)
            ).fetchone()[0]
            for t in VALID_TIERS
        }
        return (
            f"KnowledgeStore(db={self._db_path!r}, mode={mode!r}, "
            f"rules={counts['rule']}, patterns={counts['pattern']}, "
            f"observations={counts['observation']}, "
            f"archived={counts['archived']})"
        )


# ---------------------------------------------------------------------------
# Migration utility
# ---------------------------------------------------------------------------


def migrate_from_old_kb(
    old_kb: "KnowledgeBase",
    new_store: KnowledgeStore,
    *,
    skip_archived: bool = False,
    batch_size: int = 50,
    verbose: bool = True,
) -> dict[str, int]:
    """
    Migrate all entries from a legacy KnowledgeBase to a KnowledgeStore.

    Re-embeds every entry using the new model (nomic-embed-text-v1.5 if
    available, MiniLM otherwise).  Deduplication is applied during migration
    so near-duplicate legacy entries are not carried forward.

    Args:
        old_kb: The source KnowledgeBase instance.
        new_store: The target KnowledgeStore instance.
        skip_archived: If True, skip entries in the "archived" tier.
        batch_size: Log progress every N entries.
        verbose: If True, print progress messages.

    Returns:
        Dict with keys "written", "skipped_dedup", "errors".
    """
    from src.memory.knowledge_base import KnowledgeBase  # local import to avoid circular

    entries = old_kb.knowledge_list(include_archived=(not skip_archived) if hasattr(old_kb.knowledge_list, '__code__') else False)

    # Fallback: knowledge_list without include_archived param
    try:
        entries = old_kb.knowledge_list()
    except TypeError:
        entries = old_kb.knowledge_list()

    stats = {"written": 0, "skipped_dedup": 0, "errors": 0}

    for i, entry in enumerate(entries):
        try:
            tier = entry.tier
            if skip_archived and tier == "archived":
                continue
            if tier == "archived":
                tier = "observation"  # downgrade archived → observation for re-migration

            eid = new_store.write(
                content=entry.content,
                tier=tier,
                confidence=entry.confidence,
                topic_cluster=entry.topic_cluster,
                cycle=entry.created_cycle,
                optuna_evidence=entry.optuna_evidence,
                skip_dedup=False,
            )
            if eid is None:
                stats["skipped_dedup"] += 1
            else:
                stats["written"] += 1

        except Exception as exc:
            stats["errors"] += 1
            if verbose:
                print(f"  [migrate] ERROR on entry {entry.id}: {exc}")

        if verbose and (i + 1) % batch_size == 0:
            print(
                f"  [migrate] {i + 1}/{len(entries)} processed — "
                f"written={stats['written']}, dedup={stats['skipped_dedup']}, "
                f"errors={stats['errors']}"
            )

    if verbose:
        print(
            f"[migrate] COMPLETE — "
            f"written={stats['written']}, dedup={stats['skipped_dedup']}, "
            f"errors={stats['errors']}"
        )

    return stats
