# SYNTHESIS.md — Phase 1 Research Master Synthesis

> **Date:** 2026-04-22  
> **Status:** COMPLETE  
> **Input:** 7 research documents (SMALL_MODELS, OPENFANG_DEEP_DIVE, OPTUNA_ADVANCED, EMBEDDINGS_AND_KNOWLEDGE, PRODUCTION_PATTERNS, CREATIVE_EVALUATION, TOOLING)  
> **Purpose:** The definitive "what we know now" document that should drive all Phase 2+ decisions.

---

## 1. Top 20 Findings That Should Change the Implementation

Ranked by practical impact on Atwater's success. Each finding includes the source document and the specific change required.

### Tier A: Architecture-Breaking (Must Fix Before Building)

**#1. The HAND.toml is fundamentally wrong and will silently fail**  
*Source: OPENFANG_DEEP_DIVE*  
The current `config/openfang_hand.toml` has no `id` field (required), no `[agent]` section (required), uses invented fields (`[hand.entrypoint]`, `[hand.schedule]`, `[hand.env]`, `[hand.dependencies]`) that are silently ignored by Serde, and defines dashboard metrics with wrong field names. **Nothing in this file works.** The corrected HAND.toml is in OPENFANG_DEEP_DIVE §11.  
**Impact: 10/10** — Without this fix, Atwater cannot integrate with OpenFang at all.

**#2. OpenFang Hands are chat agents, not process managers**  
*Source: OPENFANG_DEEP_DIVE*  
The architecture assumes OpenFang will execute `python -m atwater.main`. That's not how Hands work. A Hand is a chat-loop LLM agent that calls tools (including `shell_exec`). Atwater's Python code must be invoked via `shell_exec` by the Hand's agent, or run standalone with OpenFang as a monitoring/dashboard layer only.  
**Impact: 10/10** — Fundamental misunderstanding of the integration model.

**#3. Constrained generation is non-negotiable for small models**  
*Source: SMALL_MODELS*  
Small models (4B-8B) cannot reliably produce valid JSON without grammar enforcement. Every LM Studio API call must include `response_format: { type: "json_schema", json_schema: {...} }`. Without this, the Grader's structured output will fail ~15-30% of the time, corrupting Optuna trials.  
**Impact: 9/10** — Silent data corruption in the optimisation loop.

**#4. Switch from TPE to AutoSampler immediately**  
*Source: OPTUNA_ADVANCED*  
OptunaHub's AutoSampler automatically selects GPSampler (better sample efficiency) or TPESampler based on problem characteristics. Hardcoding TPE leaves significant performance on the table, especially for the first 200-300 trials where GPSampler excels.  
**Impact: 8/10** — Free optimisation improvement, ~2x faster convergence in early trials.

### Tier B: Major Architecture Improvements

**#5. Replace numpy cosine similarity with sqlite-vec**  
*Source: EMBEDDINGS_AND_KNOWLEDGE*  
sqlite-vec is production-ready (`pip install sqlite-vec`), runs pure C, supports KNN in SQL, and stores vectors alongside metadata in the same SQLite database as Atwater's state. This eliminates the need for a separate vector store and gives O(log N) search instead of O(N) brute force.  
**Impact: 8/10** — Unified data layer, dramatically better knowledge retrieval at scale.

**#6. Qwen3's thinking/non-thinking mode is a free upgrade**  
*Source: SMALL_MODELS*  
Qwen3 models support `enable_thinking=True` for reasoning-heavy tasks (Director routing) and `enable_thinking=False` for fast structured output (Grader scoring). This gives us better Director decisions AND faster Grader throughput from the same model family.  
**Impact: 8/10** — Architecture-level capability we didn't know existed.

**#7. Use JournalStorage instead of SQLite RDBStorage for concurrent Optuna workers**  
*Source: OPTUNA_ADVANCED*  
SQLite with Optuna's RDBStorage suffers write contention with >1 concurrent worker. JournalStorage with file backend is append-only (no write locks). If Atwater ever runs parallel trials (which LM Studio's parallel batching enables), this is mandatory.  
**Impact: 7/10** — Prerequisite for parallel trial execution.

**#8. LM Studio supports parallel requests with continuous batching**  
*Source: TOOLING*  
LM Studio v0.4.0+ supports up to N simultaneous requests per model (default 4 slots). Combined with JournalStorage (#7), Atwater could run 4 Optuna trials concurrently against the same model. This is potentially a 3-4x throughput improvement.  
**Impact: 7/10** — Major throughput win for the evolution loop.

**#9. Multi-objective optimisation for creative tasks**  
*Source: OPTUNA_ADVANCED, CREATIVE_EVALUATION*  
Use `NSGAIISampler` with `directions=["maximize", "maximize"]` to simultaneously optimise quality AND diversity. Currently we'd have to manually balance these; Optuna can find the Pareto front automatically.  
**Impact: 7/10** — Better creative output diversity without sacrificing quality.

**#10. Dashboard metrics use memory_store, not SQL queries**  
*Source: OPENFANG_DEEP_DIVE*  
The OpenFang dashboard reads key-value pairs from `memory_store`, not SQL. Atwater's Python layer must write summary values (cycle count, best score, KB size) to OpenFang's memory_store via the agent. SQLite/Optuna DB contents cannot be directly surfaced.  
**Impact: 7/10** — Without this, the dashboard shows nothing.

### Tier C: Significant Improvements

**#11. nomic-embed-text-v1.5 with task prefixes for knowledge embeddings**  
*Source: EMBEDDINGS_AND_KNOWLEDGE*  
Matryoshka embeddings (resizable 768→256 dims), MIT license, task-prefix aware (`search_document:`, `search_query:`, `clustering:`). Two-stage retrieval: coarse search at 256 dims → rerank at 768 dims.  
**Impact: 6/10** — Better knowledge retrieval quality and flexibility.

**#12. Optuna artifacts module for storing generated outputs per trial**  
*Source: OPTUNA_ADVANCED*  
Store actual creative outputs alongside trial scores using `upload_artifact()`. Enables reviewing what was generated for any trial via Optuna Dashboard.  
**Impact: 6/10** — Critical for debugging and understanding what the optimiser is learning.

**#13. Process Reward Models over Outcome Reward**  
*Source: PRODUCTION_PATTERNS*  
Score creative generation step-by-step (brief → concept → execution → polish) rather than just the final output. PRM signal trains the optimiser faster. Map to Atwater as: Director quality → Creator quality → Grader consistency → overall.  
**Impact: 6/10** — Better gradient signal for Optuna.

**#14. Reflexion/verbal reinforcement learning pattern**  
*Source: PRODUCTION_PATTERNS*  
After each cycle, write a structured reflection that seeds the next cycle's prompt. No weight updates needed. Maps directly to Atwater's knowledge observation tier.  
**Impact: 6/10** — Achieves learning without fine-tuning.

**#15. Verifier cascade (fast→medium→slow)**  
*Source: PRODUCTION_PATTERNS, CREATIVE_EVALUATION*  
Run rule-based checks first (colour compliance, format, typography — <10ms), then embedding-based checks (CLIP style consistency — ~100ms), then LLM judge (~2-5s). Only escalate passing outputs. Saves 60-70% grading cost.  
**Impact: 6/10** — Major efficiency gain in the grading pipeline.

### Tier D: Good-to-Have

**#16. Textual (Textualize) for TUI monitoring dashboard**  
*Source: TOOLING*  
Modern Python TUI framework with CSS styling, widget library, async support, and dual terminal/web rendering. Perfect for a real-time agent monitoring dashboard showing trial status, agent activity, token usage.  
**Impact: 5/10** — Developer experience improvement.

**#17. AgentOps for agent-level tracing (self-hosted, MIT)**  
*Source: TOOLING*  
Decorator-based (`@session`, `@agent`, `@operation`), session replay, auto-LLM tracking. Self-hostable. Maps cleanly to Atwater's agent lifecycle.  
**Impact: 5/10** — Observability without platform lock-in.

**#18. Speculative decoding for Grader (2-2.5x speedup)**  
*Source: SMALL_MODELS*  
Qwen3-0.5B draft + Qwen3-8B target gives 2-2.5x speedup on structured JSON output. High acceptance rate due to grammar constraint.  
**Impact: 5/10** — Free latency improvement for the most-called agent.

**#19. Confidence decay and knowledge lifecycle management**  
*Source: EMBEDDINGS_AND_KNOWLEDGE*  
Knowledge items should decay over time unless reinforced. Ebbinghaus-inspired forgetting curve prevents stale knowledge from polluting decisions.  
**Impact: 5/10** — Prevents knowledge rot in long-running systems.

**#20. BERTopic online/incremental mode for knowledge clustering**  
*Source: EMBEDDINGS_AND_KNOWLEDGE*  
Use `partial_fit()` with MiniBatchKMeans instead of recomputing all clusters when new knowledge arrives. Critical for scaling knowledge base past ~1000 items.  
**Impact: 4/10** — Scaling concern, not immediate.

---

## 2. Revised Phase 2-9 Recommendations

### Phase 2: Core Agent Loop (REVISED — was "Core Loop + OpenFang Integration")

**ADD:**
- JSON schema enforcement on every LM Studio call (Finding #3)
- Qwen3 thinking/non-thinking mode per agent role (Finding #6)
- Corrected HAND.toml using real OpenFang schema (Finding #1)
- Decision: Run Atwater standalone OR as OpenFang Hand — not both. Recommend standalone first, OpenFang integration as Phase 8.

**CUT:**
- OpenFang integration from Phase 2 entirely. The integration model is more complex than assumed (Finding #2). Get the core loop working standalone first.

**REPRIORITIZE:**
- Structured output validation moves from "nice to have" to **critical path**. Without it, everything downstream is corrupt.

### Phase 3: Optuna Integration (REVISED)

**ADD:**
- AutoSampler instead of hardcoded TPE (Finding #4)
- JournalStorage for concurrent-safe trials (Finding #7)
- Artifact storage for trial outputs (Finding #12)
- `enqueue_trial()` for warm-starting from previous runs
- PatientPruner wrapping MedianPruner for expensive LLM evaluations

**CUT:**
- Custom sampler development (premature — AutoSampler handles this)

**REPRIORITIZE:**
- Multi-objective (quality + diversity) moves to Phase 3, not Phase 6+ (Finding #9)

### Phase 4: Knowledge Base (REVISED)

**ADD:**
- sqlite-vec for vector storage (Finding #5)
- nomic-embed-text-v1.5 with task prefixes (Finding #11)
- Semantic deduplication at ingestion (threshold ~0.95 cosine)
- Confidence tracking with temporal decay (Finding #19)

**CUT:**
- Any custom vector store implementation — sqlite-vec does it all
- FAISS dependency — sqlite-vec is simpler and sufficient for our scale

**REPRIORITIZE:**
- NetworkX knowledge graph moves to Phase 6 (not critical for MVP)
- BERTopic clustering moves to Phase 7 (premature optimisation)

### Phase 5: Grading Pipeline (REVISED)

**ADD:**
- Verifier cascade: rules → embeddings → LLM judge (Finding #15)
- Process reward scoring at each generation step (Finding #13)
- Multi-probe CLIPScore for brand alignment
- pyiqa metrics (NIMA, BRISQUE, MUSIQ) as fast quality gates
- Rubric calibration with anchor examples

**CUT:**
- FID metrics (unreliable at small batch sizes — use KID or LPIPS instead)

### Phase 6: Diversity & Exploration (REVISED)

**ADD:**
- Multi-objective Pareto front via NSGAIISampler (Finding #9)
- Embedding-based novelty scoring (k-NN distance from corpus)
- Thompson Sampling for strategy selection (multi-armed bandit)
- Temperature scheduling (start high, anneal as quality improves)

### Phase 7: Consolidation & Learning (REVISED)

**ADD:**
- Reflexion pattern: structured reflection notes per cycle (Finding #14)
- BERTopic incremental clustering for knowledge organisation
- Contradiction resolution with statistical evidence weighting

### Phase 8: OpenFang Integration (NEW — was part of Phase 2)

**ADD:**
- Corrected HAND.toml (from OPENFANG_DEEP_DIVE §11)
- Memory_store bridge: Python writes summary metrics → OpenFang dashboard reads
- shell_exec integration: OpenFang agent triggers `python -m atwater.main --cycles 1`
- Schedule management via schedule_create tool (NOT HAND.toml config)
- Test on OpenFang v0.6.0+ (schedule_create was broken before v0.5.10)

### Phase 9: Monitoring & Observability (REVISED)

**ADD:**
- Optuna Dashboard for trial visualisation (Finding #12)
- Textual TUI for real-time agent monitoring (Finding #16)
- AgentOps decorators for agent tracing (Finding #17)
- Structured JSONL trace logs per session

**CUT:**
- LangSmith integration (platform lock-in, unnecessary)
- MLflow (too heavy for our use case)

---

## 3. Architecture Changes — Specific Modifications to ARCHITECTURE.md

### Change 1: Add Constrained Output to All Agent Definitions

Every agent definition must include an `output_schema` field with a JSON Schema. The orchestrator must pass this schema to every LM Studio API call via `response_format`.

```python
# Add to each agent config
AGENT_CONFIGS = {
    "director": {
        "model": "qwen3-8b-q8",
        "thinking_mode": True,   # NEW: Qwen3 thinking mode
        "temperature": 0.1,
        "output_schema": DIRECTOR_SCHEMA,
    },
    "grader": {
        "model": "qwen3-4b-q8",
        "thinking_mode": False,  # NEW: fast mode for grading
        "temperature": 0.0,
        "output_schema": GRADER_SCHEMA,
    },
    # ...
}
```

### Change 2: Replace Statistical Layer Configuration

**Before (ARCHITECTURE.md):**
```python
sampler=optuna.samplers.TPESampler(seed=42)
storage="sqlite:///trials.db"
```

**After:**
```python
import optunahub
from optuna.storages import JournalStorage, JournalFileBackend

sampler = optunahub.load_module("samplers/auto_sampler").AutoSampler()
storage = JournalStorage(JournalFileBackend("./studies/atwater.log"))
pruner = optuna.pruners.PatientPruner(
    optuna.pruners.MedianPruner(n_startup_trials=5), patience=2
)
```

### Change 3: Add Verifier Cascade to Grading Pipeline

Insert between Creator output and full LLM Grader:

```
Creator output → Fast Gate (rules, <10ms) → Medium Gate (CLIP/embedding, ~100ms) → LLM Grader (~2-5s)
```

Only outputs passing each gate proceed to the next. Failed at Fast Gate = immediate score 0, skip expensive evaluation.

### Change 4: Replace Knowledge Retrieval Implementation

**Before:** Implied numpy cosine similarity  
**After:** sqlite-vec with nomic-embed-text-v1.5

```python
# knowledge_read() implementation changes to:
def knowledge_read(query, tier=None, k=5, min_confidence=0.5):
    query_emb = embed(f"search_query: {query}")
    results = db.execute("""
        SELECT k.*, v.distance FROM knowledge_vecs v
        JOIN knowledge k ON k.id = v.knowledge_id
        WHERE v.embedding MATCH ? AND k.confidence >= ?
        ORDER BY v.distance LIMIT ?
    """, [query_emb, min_confidence, k]).fetchall()
    if tier:
        results = [r for r in results if r.tier == tier]
    return results
```

### Change 5: Add Multi-Objective Support

Extend the evolution loop to support multi-objective optimisation:

```python
# Single-objective (default, backward-compatible)
study = optuna.create_study(direction="maximize", ...)

# Multi-objective (quality + diversity)
study = optuna.create_study(
    directions=["maximize", "maximize"],
    sampler=optuna.samplers.NSGAIISampler(),
    ...
)
# Grader returns tuple: (quality_score, diversity_score)
```

### Change 6: Decouple OpenFang Integration

Move OpenFang from core architecture to an optional integration layer:

```
Core Atwater (standalone Python):
  main.py → evolution loop → Optuna → knowledge base → agents → LM Studio

Optional Integration Layer:
  OpenFang Hand → shell_exec("python -m atwater.main --cycles 1")
  OpenFang Dashboard ← memory_store bridge ← Python summary metrics
```

---

## 4. New Features Worth Building (Ranked by Effort vs Impact)

| Rank | Feature | Effort | Impact | Phase |
|------|---------|--------|--------|-------|
| 1 | JSON schema enforcement wrapper | 2h | Critical | 2 |
| 2 | AutoSampler + JournalStorage setup | 1h | High | 3 |
| 3 | sqlite-vec knowledge store | 4h | High | 4 |
| 4 | Verifier cascade (fast→medium→slow grading) | 6h | High | 5 |
| 5 | Qwen3 thinking mode toggle per agent | 2h | High | 2 |
| 6 | Optuna artifact storage for trial outputs | 2h | Medium | 3 |
| 7 | Warm-start with enqueue_trial | 1h | Medium | 3 |
| 8 | Multi-objective (quality + diversity) | 4h | Medium | 6 |
| 9 | Reflexion notes per cycle | 3h | Medium | 7 |
| 10 | Textual TUI dashboard | 8h | Medium | 9 |
| 11 | AgentOps tracing integration | 4h | Medium | 9 |
| 12 | Confidence decay on knowledge items | 3h | Low-Med | 4 |
| 13 | Corrected OpenFang HAND.toml + bridge | 6h | Low* | 8 |
| 14 | Speculative decoding for Grader | 2h | Low-Med | 9 |
| 15 | BERTopic incremental clustering | 8h | Low | 7 |

*OpenFang integration is low impact because Atwater should work standalone first.

---

## 5. Things We Got Wrong

### Wrong: OpenFang is a process manager that will run our Python code
**Reality:** OpenFang Hands are chat-based LLM agents. The Hand's agent can call `shell_exec` to run Python, but it's mediated by an LLM. Atwater's evolution loop should run as standalone Python, with OpenFang as an optional monitoring/orchestration layer.

### Wrong: Dashboard metrics can show SQL queries against Optuna/SQLite DBs
**Reality:** Dashboard metrics read from `memory_store` key-value pairs only. No SQL, no direct DB access. The Python layer must bridge summary data.

### Wrong: HAND.toml supports `[hand.entrypoint]`, `[hand.schedule]`, `[hand.env]`
**Reality:** None of these fields exist. Tools is a flat string array. Scheduling is done by the agent at runtime via `schedule_create` tool. Environment configuration is via OpenFang's `config.toml`.

### Wrong: TPE is the right default sampler for Optuna
**Reality:** AutoSampler (OptunaHub) dynamically selects the best algorithm. GPSampler outperforms TPE for the first 200-300 trials on mixed integer/float problems — exactly Atwater's use case.

### Wrong: SQLite RDBStorage is fine for concurrent Optuna workers
**Reality:** Write contention at >1 worker. JournalStorage with file backend is append-only and avoids this entirely.

### Wrong: Raw model output is reliable enough to parse as JSON
**Reality:** Small models (4B-8B) produce invalid JSON 15-30% of the time without constrained generation. Every API call needs `response_format` with a JSON schema.

### Wrong: Knowledge retrieval via numpy cosine similarity is sufficient
**Reality:** sqlite-vec provides persistent, indexed, SQL-queryable vector search with O(log N) complexity. Numpy is fine for <1K items but doesn't persist and doesn't scale.

---

## 6. Things We Got Right

### Right: Three-tier memory architecture (Working → Shared State → Knowledge Base)
**Validated by:** PRODUCTION_PATTERNS (hierarchical memory), EMBEDDINGS_AND_KNOWLEDGE (multi-tier memory for autonomous agents). The tier structure maps cleanly to the research best practices. Atwater's architecture is ahead of most agent frameworks here.

### Right: Knowledge tiers (Rules → Patterns → Observations) with promotion
**Validated by:** EMBEDDINGS_AND_KNOWLEDGE (confidence calibration), PRODUCTION_PATTERNS (trajectory database). The promotion mechanism with statistical evidence is a sound design.

### Right: Diversity Guard as a first-class agent
**Validated by:** PRODUCTION_PATTERNS (mode collapse prevention), CREATIVE_EVALUATION (novelty detection). Research strongly supports dedicated diversity monitoring. Most agent systems lack this.

### Right: Consolidator agent for knowledge compaction
**Validated by:** EMBEDDINGS_AND_KNOWLEDGE (semantic deduplication), PRODUCTION_PATTERNS (meta-learning). The consolidation concept is validated; specifics can be refined.

### Right: Optuna for statistical experiment tracking (separate from qualitative knowledge)
**Validated by:** OPTUNA_ADVANCED (rich feature set validates the choice). Optuna is the right tool — we just need to use it better (AutoSampler, JournalStorage, artifacts, multi-objective).

### Right: Context scoping per agent (orchestrator filters what each agent sees)
**Validated by:** PRODUCTION_PATTERNS (AutoAct division-of-labour), SMALL_MODELS (system prompt <500 tokens). Small models especially benefit from minimal, focused context.

### Right: Grader with structured output format
**Validated by:** CREATIVE_EVALUATION (rubric-based scoring), PRODUCTION_PATTERNS (process reward models). The structured grading format is correct — it just needs constrained generation to be reliable.

### Right: Evolution loop architecture (Director → Creator → Grader → Diversity Guard)
**Validated by:** PRODUCTION_PATTERNS (Reflexion loop, multi-agent debate). The flow is sound. Research suggests adding step-level scoring (process reward) and verbal reflection for even better learning.

---

## 7. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|-----------|
| 1 | **LM Studio JSON schema enforcement is incomplete** — structured output may not be grammar-constrained, just "hopeful" | Medium | Critical | Test empirically with Qwen3-4B. Fallback: use llama.cpp server directly with GBNF grammar, or Outlines library. |
| 2 | **Qwen3 thinking mode not supported via OpenAI-compatible API** — may require native LM Studio SDK | Medium | High | Test via `enable_thinking` parameter. Fallback: use prompt-level "think step by step" instructions. |
| 3 | **sqlite-vec on macOS requires Homebrew Python** — default macOS Python SQLite doesn't support extensions | High | Medium | Use `brew install python` or `pip install pysqlite3`. Document in setup guide. Already known issue. |
| 4 | **AutoSampler from OptunaHub has external dependency** — may not be available offline | Low | Medium | Fallback to `TPESampler(seed=42)`. Wrap in try/except. |
| 5 | **Parallel LM Studio requests require enough VRAM** — 4 concurrent requests to 8B model needs ~36GB unified memory | Medium | High | Start with n_parallel=1, test scaling. On 16GB Mac: use 4B models or n_parallel=2. |
| 6 | **OpenFang v0.6.0 schedule_create may still have edge cases** — was broken until v0.5.10 | Medium | Medium | Test scheduling thoroughly. Fallback: use system cron or Python's own scheduling (APScheduler). |
| 7 | **Multi-objective optimisation produces Pareto front, not single best** — requires choosing from candidates | Low | Medium | Use TOPSIS or weighted sum to select from Pareto front for deployment. Keep Pareto exploration for discovery. |
| 8 | **Knowledge base grows unbounded without aggressive pruning** — cosine similarity search degrades | Medium | Medium | Implement confidence decay + periodic consolidation. Set hard cap at 10K items with LRU eviction. |
| 9 | **nomic-embed-text-v1.5 may not capture creative/aesthetic similarity well** — trained on general text | Medium | Low | Test with creative text pairs. Fallback: use CLIP embeddings for visual/aesthetic content, nomic for text knowledge only. |
| 10 | **Speculative decoding adds complexity for marginal gain** — draft model loading, cache management | Low | Low | Defer to Phase 9. The 2x speedup is nice but not critical. |
| 11 | **AgentOps decorator overhead on hot path** — tracing every LLM call adds latency | Low | Low | Profile first. Disable in production hot path, enable for debugging runs only. |
| 12 | **BERTopic incremental mode requires MiniBatchKMeans** — loses HDBSCAN's density-based benefits | Medium | Low | Use HDBSCAN for periodic full recomputation (every 500 items), incremental for between recomputes. |

---

## Appendix A: Recommended Dependency List

```
# Core
optuna>=4.8.0
optunahub
sqlite-vec
sentence-transformers
numpy

# Knowledge
nomic-embed-text-v1.5  # via sentence-transformers
networkx  # Phase 6+
bertopic  # Phase 7+

# Evaluation (Phase 5)
pyiqa
clip  # openai/CLIP
scikit-learn
scipy

# Monitoring (Phase 9)
textual
agentops
rich

# LLM Client
openai  # for OpenAI-compatible LM Studio API

# Testing
pytest
pytest-asyncio
pytest-mock
pytest-timeout
pytest-benchmark

# Development
lmstudio  # optional, for native SDK features
```

## Appendix B: Model Selection Quick Reference

| Agent | Primary Model | Quant | Thinking | Temperature |
|-------|--------------|-------|----------|-------------|
| Director | Qwen3-8B | Q8_0 | ON | 0.1 |
| Creator | Qwen3-8B | Q5_K_M | OFF | 0.7 |
| Grader | Qwen3-4B | Q8_0 | OFF | 0.0 |
| Diversity Guard | Qwen3-4B | Q4_K_M | OFF | 0.8 |
| Orchestrator | Qwen3-8B | Q8_0 | ON | 0.3 |
| Consolidator | Qwen3-8B | Q8_0 | ON | 0.3 |

## Appendix C: Critical First Steps (Before Writing Any Code)

1. **Test LM Studio JSON schema enforcement** with Qwen3-4B — does `response_format` actually constrain output?
2. **Test Qwen3 thinking mode** via OpenAI-compatible API — does `enable_thinking` pass through?
3. **Install sqlite-vec** and verify it loads on the target macOS machine
4. **Install OptunaHub AutoSampler** and verify offline fallback works
5. **Create corrected HAND.toml** from OPENFANG_DEEP_DIVE §11 — test with `openfang hand install`
6. **Decide standalone vs OpenFang-first** — recommendation: standalone first, OpenFang Phase 8

---

*End of synthesis. This document should be the single source of truth for Phase 2 planning.*
