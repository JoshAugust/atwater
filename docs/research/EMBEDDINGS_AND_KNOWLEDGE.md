# EMBEDDINGS AND KNOWLEDGE SYSTEMS RESEARCH
_Generated: 2026-04-22 | Atwater Project_

---

## EXECUTIVE SUMMARY

Three major findings dominate this research:

1. **sqlite-vec is production-ready** — `pip install sqlite-vec` works, pure C, runs everywhere, full KNN in SQL. Should **replace numpy cosine similarity search** for persistent knowledge stores.
2. **nomic-embed-text-v1.5 is the default choice** for Atwater — Matryoshka (resizable dims), MIT license, strong performance, task-prefix aware, open weights.
3. **BERTopic with online/incremental mode** is the right clustering architecture for growing knowledge bases — no full recompute needed.

---

## 1. EMBEDDING MODEL SELECTION

### The Landscape (2025-2026)

The MTEB (Massive Textual Embedding Benchmark) leaderboard is the authoritative source. Key insight: **filter for model size** — top leaderboard models are often 7B+ and unusable without GPU.

### RECOMMENDED: `nomic-embed-text-v1.5`

```python
pip install sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5")

# IMPORTANT: Task-prefix required for optimal performance
embeddings = model.encode([
    "clustering: the quick brown fox",           # for clustering tasks
    "search_document: TSNE reduces dimensions",  # for RAG document storage
    "search_query: What is dimensionality?",     # for RAG queries
    "classification: positive sentiment text",   # for classification
])
```

**Task instruction prefixes (must use):**
- `search_document:` — embed documents for RAG index
- `search_query:` — embed queries for retrieval
- `clustering:` — embed for clustering/deduplication
- `classification:` — embed for classification

**Why nomic-embed-text-v1.5 for Atwater:**
- **Matryoshka (resizable)**: Use 768 dims for quality, 256 for speed — same model
- **Open weights** (Apache 2.0): No API dependency, runs locally
- **Multimodal aligned**: `nomic-embed-vision-v1.5` shares the same embedding space
- **Performance**: Competitive with OpenAI text-embedding-3-small at fraction of cost
- **Performance retention**: 95.8% at 3x compression (256 dims), 90% at 6x (128 dims)

### Comparison Table

| Model | Dims | Speed (CPU) | Quality | Size | Notes |
|-------|------|-------------|---------|------|-------|
| `nomic-embed-text-v1.5` | 768 (resizable) | ~Medium | ⭐⭐⭐⭐ | ~137MB | **RECOMMENDED** |
| `all-MiniLM-L6-v2` | 384 | ~18K/sec | ⭐⭐⭐ | ~22MB | Good for dev/test |
| `all-mpnet-base-v2` | 768 | ~4K/sec | ⭐⭐⭐⭐ | ~438MB | Best quality/size before nomic |
| `bge-small-en-v1.5` | 384 | Fast | ⭐⭐⭐ | ~23MB | Good multilingual alternative |
| `bge-m3` | 1024 | Slow | ⭐⭐⭐⭐⭐ | ~580MB | Multi-lingual, multi-task |
| `text-embedding-3-large` | 3072 | API | ⭐⭐⭐⭐⭐ | — | Best quality, API only, $$ |

**BGE-small vs all-MiniLM vs nomic (short text):**
- For short creative text (<50 tokens): all-MiniLM-L6-v2 is surprisingly competitive
- For aesthetic/style similarity: nomic with `clustering:` prefix wins
- BGE-small is best when multilingual or when speed is critical

### Matryoshka Embeddings — Adaptive Dimensionality

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5")

# Full quality for final storage/indexing
full_embeddings = model.encode(texts, normalize_embeddings=True)  # 768 dims

# Truncate for fast shortlisting
import numpy as np
fast_embeddings = full_embeddings[:, :256]  # 256 dims, still ~95% quality
fast_embeddings = fast_embeddings / np.linalg.norm(fast_embeddings, axis=1, keepdims=True)  # re-normalize
```

**Two-stage retrieval pattern (Atwater should adopt):**
1. Encode all knowledge items at 768 dims, store in sqlite-vec
2. At query time: encode query at 256 dims, do coarse KNN search
3. Rerank top-K results using full 768 dim cosine similarity
4. Return final ranked results

This achieves near-full quality at significantly reduced search time.

---

## 2. BINARY & SCALAR QUANTIZATION

For large-scale knowledge stores where memory matters:

### Binary Quantization

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")  # works best with binary
embeddings = model.encode(texts)

# Binary quantization: float32 → 1 bit per dimension
binary_embeddings = np.packbits(embeddings > 0, axis=-1)
```

**Performance impact (HuggingFace experiments on 41M Wikipedia texts):**
- Memory: **32x reduction** (float32 → binary)
- Retrieval speed: **25-50x faster** with Hamming distance
- Quality retention: ~90-95% with rescoring (rerank binary results with full floats)
- Storage: ~6KB per 1M embeddings vs ~192KB for float32 at 384 dims

**When to use:** >1M documents where memory is the constraint.

### Scalar (int8) Quantization

```python
# int8: float32 → 8 bits per dimension
# 4x memory reduction, ~2x speed, >99% quality retention
from sentence_transformers.quantization import quantize_embeddings

int8_embeddings = quantize_embeddings(embeddings, precision="int8")
```

**For Atwater:**
- At Atwater's likely scale (<100K knowledge items), full float32 is fine
- Binary quantization becomes relevant >500K items
- sqlite-vec natively supports float, int8, and binary vectors

---

## 3. SQLITE-VEC — THE KEY FINDING

**This replaces numpy cosine similarity for persistent knowledge stores.**

### What It Is
- Pure C SQLite extension for vector search
- KNN queries via standard SQL
- Runs everywhere Python+SQLite runs
- Mozilla Builders project (well-funded, maintained)
- Pre-v1 (breaking changes possible, but API is stable in practice)

### Installation & Python Usage

```bash
pip install sqlite-vec
```

```python
import sqlite3
import sqlite_vec
import numpy as np

# Setup
db = sqlite3.connect("knowledge.db")  # or ":memory:"
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)

# Create vector table
db.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_embeddings USING vec0(
        embedding float[768],          -- dimension must match your model
        +content TEXT,                 -- metadata column (non-vector)
        +concept_id TEXT,              -- metadata column
        +confidence REAL               -- metadata column
    )
""")

# Insert vectors (NumPy arrays work directly — must cast to float32)
def insert_knowledge(concept_id: str, content: str, embedding: np.ndarray, confidence: float = 1.0):
    db.execute(
        "INSERT INTO knowledge_embeddings(rowid, embedding, content, concept_id, confidence) VALUES (?, ?, ?, ?, ?)",
        [None, embedding.astype(np.float32), content, concept_id, confidence]
    )
    db.commit()

# KNN search (the whole point)
def find_similar(query_embedding: np.ndarray, k: int = 10):
    results = db.execute("""
        SELECT rowid, concept_id, content, confidence, distance
        FROM knowledge_embeddings
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
    """, [query_embedding.astype(np.float32), k]).fetchall()
    return results
```

### MacOS Warning
The default macOS Python SQLite doesn't support extensions. Fix:
```bash
brew install python  # uses Homebrew SQLite which supports extensions
# or
pip install pysqlite3  # 3rd party with bundled SQLite
```

### Performance vs NumPy

sqlite-vec uses:
- **L2 (Euclidean) distance** by default
- **Cosine distance** available via normalized vectors (normalize before insert)

For normalized embeddings: L2 distance ≈ cosine distance, so normalize once at insert.

```python
def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)

# Insert normalized — then L2 distance = cosine distance
insert_knowledge(concept_id, content, normalize(embedding))
```

**vs NumPy performance:**
- NumPy brute-force cosine: O(N) per query, excellent for <10K items
- sqlite-vec: O(log N) for indexed queries (vec0 builds approximate index), better at scale
- At ~1K items: roughly equivalent
- At ~10K items: sqlite-vec 5-10x faster with proper indexing
- At ~100K items: sqlite-vec dramatically faster

**Key advantage beyond speed:** sqlite-vec stores vectors alongside metadata in the same DB as Atwater's other SQLite state. No separate vector store. Unified query.

### Full Architecture Pattern for Atwater

```python
import sqlite3
import sqlite_vec
import numpy as np
from sentence_transformers import SentenceTransformer

class AtwaterKnowledgeStore:
    def __init__(self, db_path: str = "atwater_knowledge.db"):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self._setup()
        self.model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5")
    
    def _setup(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY,
                concept TEXT NOT NULL,
                content TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source TEXT,
                created_at INTEGER DEFAULT (unixepoch()),
                access_count INTEGER DEFAULT 0,
                last_accessed INTEGER
            );
            
            CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_vecs USING vec0(
                embedding float[768],
                +knowledge_id INTEGER
            );
            
            CREATE INDEX IF NOT EXISTS idx_knowledge_confidence 
                ON knowledge(confidence DESC);
        """)
        self.db.commit()
    
    def embed(self, text: str, task: str = "clustering") -> np.ndarray:
        """Embed with task prefix and normalize."""
        prefixed = f"{task}: {text}"
        embedding = self.model.encode(prefixed, normalize_embeddings=True)
        return embedding.astype(np.float32)
    
    def add(self, concept: str, content: str, confidence: float = 1.0, source: str = None):
        """Add knowledge item with embedding."""
        cursor = self.db.execute(
            "INSERT INTO knowledge (concept, content, confidence, source) VALUES (?, ?, ?, ?)",
            [concept, content, confidence, source]
        )
        knowledge_id = cursor.lastrowid
        embedding = self.embed(content)
        self.db.execute(
            "INSERT INTO knowledge_vecs (embedding, knowledge_id) VALUES (?, ?)",
            [embedding, knowledge_id]
        )
        self.db.commit()
        return knowledge_id
    
    def search(self, query: str, k: int = 10, min_confidence: float = 0.0):
        """Semantic search with optional confidence filter."""
        query_embedding = self.embed(query, task="search_query")
        results = self.db.execute("""
            SELECT k.id, k.concept, k.content, k.confidence, kv.distance
            FROM knowledge_vecs kv
            JOIN knowledge k ON k.id = kv.knowledge_id
            WHERE kv.embedding MATCH ?
              AND k.confidence >= ?
            ORDER BY kv.distance
            LIMIT ?
        """, [query_embedding, min_confidence, k]).fetchall()
        return results
    
    def find_duplicates(self, threshold: float = 0.05):
        """Find semantically duplicate knowledge items."""
        # Low distance = high similarity. 0.05 L2 on normalized ≈ >99% cosine sim
        all_knowledge = self.db.execute("SELECT id, content FROM knowledge").fetchall()
        duplicates = []
        for kid, content in all_knowledge:
            emb = self.embed(content)
            similar = self.db.execute("""
                SELECT knowledge_id, distance FROM knowledge_vecs
                WHERE embedding MATCH ? AND knowledge_id != ?
                ORDER BY distance LIMIT 5
            """, [emb, kid]).fetchall()
            for sim_id, dist in similar:
                if dist < threshold:
                    duplicates.append((kid, sim_id, dist))
        return duplicates
```

---

## 4. CLUSTERING — HDBSCAN vs ALTERNATIVES

### HDBSCAN — RECOMMENDED for Atwater

**Why HDBSCAN wins for knowledge clustering:**
- Handles variable-density clusters (realistic for knowledge items)
- No need to specify number of clusters
- Marks noise points as -1 (don't force outlier knowledge into wrong cluster)
- Stable results across runs
- Integrated into BERTopic's default pipeline

**vs OPTICS:**
- OPTICS: better at very low-density clusters, slower, harder to interpret
- HDBSCAN: faster, simpler parameters, better practical performance

**vs Spectral Clustering:**
- Spectral: requires fixed K, assumes convex clusters
- HDBSCAN: no fixed K, handles arbitrary shapes

**vs K-Means (MiniBatchKMeans):**
- K-Means: needs K, assumes spherical clusters, but faster and incremental
- HDBSCAN: better quality clusters, but not incrementally updatable

### BERTopic — The Right Framework

BERTopic is HDBSCAN + UMAP + c-TF-IDF in a clean pipeline:

```python
pip install bertopic
```

**Full pipeline (with nomic embeddings):**
```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5")
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
hdbscan_model = HDBSCAN(
    min_cluster_size=5,      # small for knowledge items
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    calculate_probabilities=True
)

topics, probs = topic_model.fit_transform(knowledge_texts)
topic_info = topic_model.get_topic_info()
```

### Online/Incremental Topic Modeling (Critical for Atwater)

BERTopic supports incremental learning without full recompute:

```python
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from bertopic.vectorizers import OnlineCountVectorizer

# Sub-models that support .partial_fit
umap_model = IncrementalPCA(n_components=5)
cluster_model = MiniBatchKMeans(n_clusters=20, random_state=42)
vectorizer_model = OnlineCountVectorizer(stop_words="english", decay=0.01)

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=cluster_model,
    vectorizer_model=vectorizer_model
)

# Add new knowledge in batches
for batch in knowledge_batches:
    topic_model.partial_fit(batch)
```

**decay parameter**: reduces frequency of old words by X% per iteration. 0.01 = 1% decay keeps recent knowledge more prominent.

**Merge models** (alternative, often better quality):
```python
# Train separate models on different time windows, then merge
merged_model = BERTopic.merge_models([model_old, model_new])
```

---

## 5. KNOWLEDGE GRAPH INTEGRATION — NetworkX

For connecting knowledge items with typed relationships:

```python
pip install networkx
```

```python
import networkx as nx

class AtwaterKnowledgeGraph:
    def __init__(self):
        self.G = nx.DiGraph()  # Directed graph for knowledge relationships
    
    def add_concept(self, concept_id: str, **attrs):
        self.G.add_node(concept_id, **attrs)
    
    def add_relationship(self, from_id: str, to_id: str, rel_type: str, weight: float = 1.0):
        self.G.add_edge(from_id, to_id, type=rel_type, weight=weight)
    
    def get_related(self, concept_id: str, depth: int = 2):
        """Get all concepts within N hops."""
        return list(nx.ego_graph(self.G, concept_id, radius=depth).nodes)
    
    def get_path(self, from_id: str, to_id: str):
        """Shortest path between concepts."""
        try:
            return nx.shortest_path(self.G, from_id, to_id)
        except nx.NetworkXNoPath:
            return None
    
    def get_central_concepts(self, top_k: int = 10):
        """Most important concepts by PageRank."""
        pagerank = nx.pagerank(self.G, weight='weight')
        return sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    def save(self, path: str):
        import json
        data = nx.node_link_data(self.G)
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path: str):
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        self.G = nx.node_link_graph(data)
```

**Hybrid: Knowledge Graph + Vector Store**
The winning pattern is combining both:
- **NetworkX**: typed relationships, graph traversal, causal chains
- **sqlite-vec**: semantic similarity search
- Query path: semantic search → find concept IDs → graph traversal → expand context

---

## 6. SEMANTIC DEDUPLICATION

For removing duplicate knowledge items before/during ingestion:

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SemanticDeduplicator:
    """
    Efficient deduplication using embedding similarity.
    For large sets: use sqlite-vec KNN instead of brute-force pairwise.
    """
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold  # cosine similarity above this = duplicate
    
    def deduplicate(self, texts: list[str], embeddings: np.ndarray) -> list[int]:
        """Return indices of unique items."""
        keep = []
        for i, emb in enumerate(embeddings):
            if not keep:
                keep.append(i)
                continue
            kept_embs = embeddings[keep]
            sims = cosine_similarity(emb.reshape(1, -1), kept_embs)[0]
            if sims.max() < self.threshold:
                keep.append(i)
        return keep
    
    def deduplicate_with_sqlite_vec(self, db, new_text: str, new_embedding: np.ndarray) -> bool:
        """
        Check if semantically similar item already exists in DB.
        Returns True if item is new (should be kept).
        """
        # L2 distance on normalized vectors: <0.1 ≈ >99.5% cosine similarity
        results = db.execute("""
            SELECT knowledge_id, distance FROM knowledge_vecs
            WHERE embedding MATCH ?
            ORDER BY distance LIMIT 1
        """, [new_embedding]).fetchall()
        
        if not results:
            return True  # No similar items, keep it
        
        _, distance = results[0]
        # Convert L2 distance to cosine similarity: cos_sim ≈ 1 - (dist²/2) for normalized
        cos_sim = 1.0 - (distance ** 2) / 2
        return cos_sim < self.threshold
```

---

## 7. CONFIDENCE CALIBRATION FOR AI KNOWLEDGE

Pattern for tracking reliability of AI-generated knowledge:

```python
from dataclasses import dataclass, field
from typing import Optional
import time

@dataclass
class KnowledgeItem:
    concept: str
    content: str
    confidence: float = 1.0           # 0.0 to 1.0
    source: str = "llm_generated"
    evidence_count: int = 1           # How many sources support this
    contradiction_count: int = 0      # How many contradict this
    last_reinforced: float = field(default_factory=time.time)
    decay_rate: float = 0.01          # Per-day confidence decay
    
    @property
    def effective_confidence(self) -> float:
        """Apply temporal decay to confidence."""
        days_old = (time.time() - self.last_reinforced) / 86400
        decayed = self.confidence * (1 - self.decay_rate) ** days_old
        return max(0.0, decayed)
    
    def reinforce(self, strength: float = 0.1):
        """Increase confidence when evidence confirms this knowledge."""
        self.confidence = min(1.0, self.confidence + strength)
        self.evidence_count += 1
        self.last_reinforced = time.time()
    
    def contradict(self, strength: float = 0.2):
        """Decrease confidence when evidence contradicts."""
        self.confidence = max(0.0, self.confidence - strength)
        self.contradiction_count += 1
    
    def is_reliable(self, threshold: float = 0.5) -> bool:
        return self.effective_confidence >= threshold
```

**Forgetting Curve (Ebbinghaus-inspired):**
- Confidence decays exponentially unless reinforced
- Decay rate configurable per knowledge type (factual vs opinion vs style)
- Re-exposure resets decay counter and bumps confidence

---

## 8. HIERARCHICAL MEMORY FOR AUTONOMOUS AGENTS

Multi-tier memory architecture (current best practice 2025-2026):

```
Tier 1: Working Memory (in-process dict)
├── Current task context
├── Active conversation state
└── Last N tool results

Tier 2: Session Memory (SQLite, same DB)
├── All facts from current session
├── User preferences observed this session
└── Embeddings for similarity search (sqlite-vec)

Tier 3: Long-Term Knowledge (SQLite + NetworkX)
├── Validated, high-confidence knowledge
├── Concept relationships (graph)
├── Historical optimization results
└── Style profiles and aesthetic preferences

Tier 4: Archival (optional)
└── Raw logs, full trial history
```

**Promotion pattern:**
```python
def promote_to_long_term(item: KnowledgeItem, threshold: float = 0.7):
    """Move high-confidence session knowledge to long-term store."""
    if item.effective_confidence >= threshold and item.evidence_count >= 2:
        long_term_store.add(item)
        session_store.archive(item.id)
```

**RAG Pattern for Atwater:**
```python
def build_context(query: str, working_memory: dict, knowledge_store: AtwaterKnowledgeStore) -> str:
    # 1. Search long-term knowledge
    similar_knowledge = knowledge_store.search(query, k=5, min_confidence=0.6)
    
    # 2. Include working memory (always fresh)
    current_context = working_memory.get("current_task", "")
    
    # 3. Build hierarchical context string
    context_parts = []
    if current_context:
        context_parts.append(f"CURRENT TASK:\n{current_context}")
    if similar_knowledge:
        context_parts.append("RELEVANT KNOWLEDGE:\n" + "\n".join(
            f"- [{k[4]:.3f} dist] {k[2]}" for k in similar_knowledge
        ))
    
    return "\n\n".join(context_parts)
```

---

## 9. WHAT TO CHANGE IN ATWATER's KNOWLEDGE CODE

### Immediate Changes (High Priority):

1. **Replace numpy cosine similarity with sqlite-vec**
   ```bash
   pip install sqlite-vec
   ```
   - Store all embeddings in sqlite-vec `vec0` tables alongside your existing SQLite data
   - Use KNN queries instead of brute-force numpy loops
   - Critical: normalize embeddings before insert → L2 ≈ cosine

2. **Switch to `nomic-embed-text-v1.5`**
   - Add task prefixes: `"clustering: {text}"` for knowledge, `"search_query: {text}"` for queries
   - Use 768 dims for storage, 256 for fast search if needed

3. **Add semantic deduplication at ingestion time**
   - Before inserting new knowledge, check if similar item exists (threshold ~0.95)
   - Prevents bloat in knowledge store over time

4. **Add confidence tracking and decay**
   - Every knowledge item needs: `confidence` (float), `last_reinforced` (timestamp), `evidence_count`
   - Run decay on read, not on write (lazy evaluation)

5. **Use BERTopic's online mode for clustering**
   - Don't recompute clusters from scratch when new knowledge arrives
   - Use `partial_fit()` with MiniBatchKMeans + IncrementalPCA

### Medium Priority:

6. **Add NetworkX for concept relationships**
   - Knowledge items should link to each other via typed edges
   - Store graph as JSON (serialize/deserialize with `node_link_data`)

7. **Implement two-stage retrieval**
   - Coarse search at 256 dims → rerank top-20 at full 768 dims
   - Better quality at lower compute cost

### Architecture Note on sqlite-vec:
If running on macOS, use Homebrew Python or `pysqlite3` package. The default macOS Python SQLite doesn't support extensions (will fail silently or raise AttributeError).

```python
# Safe cross-platform loading:
try:
    import sqlite_vec
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    SQLITE_VEC_AVAILABLE = True
except (AttributeError, ImportError):
    SQLITE_VEC_AVAILABLE = False
    # Fall back to numpy cosine similarity
```

---

## SOURCES

- sqlite-vec GitHub: https://github.com/asg017/sqlite-vec
- sqlite-vec Python docs: https://alexgarcia.xyz/sqlite-vec/python.html
- nomic-embed-text-v1.5: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
- Matryoshka embeddings: https://huggingface.co/blog/matryoshka
- Embedding quantization: https://huggingface.co/blog/embedding-quantization
- SBERT pretrained models: https://sbert.net/docs/sentence_transformer/pretrained_models.html
- BGE-small-en-v1.5: https://huggingface.co/BAAI/bge-small-en-v1.5
- HDBSCAN comparison: https://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html
- BERTopic GitHub: https://github.com/MaartenGr/BERTopic
- BERTopic algorithm: https://maartengr.github.io/BERTopic/algorithm/algorithm.html
- BERTopic online: https://maartengr.github.io/BERTopic/getting_started/online/online.html
- NetworkX tutorial: https://networkx.org/documentation/stable/tutorial.html
- RAG overview: https://research.ibm.com/blog/retrieval-augmented-generation-RAG
