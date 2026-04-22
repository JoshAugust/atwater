# NIGHTWATCH PLAN — 14-Hour Deep Build (REVISED)

## Status: PHASE 2 — BUILDING
Research complete: 2026-04-22 15:22 PDT
Build started: 2026-04-22 15:30 PDT

## Key Research Findings Driving Changes
1. HAND.toml is fundamentally broken (4 fatal errors, 5 invented sections)
2. OpenFang Hands are chat agents, not process managers — decouple to Phase 8
3. JSON schema enforcement is mandatory for small models (15-30% failure without)
4. AutoSampler replaces hardcoded TPE (2x faster convergence)
5. sqlite-vec replaces numpy cosine similarity (production-ready, pip install)
6. JournalStorage replaces SQLite RDB for Optuna (concurrent-safe)
7. Qwen3 thinking/non-thinking mode is a free quality/speed toggle
8. Verifier cascade saves 60-70% grading cost
9. pyiqa/CLIP/LAION for local creative evaluation — no cloud needed
10. LM Studio supports 4 concurrent requests — parallel trials possible

---

## Phase 2 — Integration Surgery + Critical Fixes (Hours 1-3)
*Make existing code work, fix architecture-breaking issues*

### 2A: Import Chain & Interface Resolution
- [ ] 2.1 Run full import chain, catalog broken imports
- [ ] 2.2 Resolve duplicate AgentContext (base.py vs context_assembler.py)
- [ ] 2.3 Resolve KnowledgeEntry duplication (memory/ vs knowledge/)
- [ ] 2.4 Audit all __init__.py exports
- [ ] 2.5 Add src/__init__.py for package resolution
- [ ] 2.6 Create pyproject.toml for proper packaging
- [ ] 2.7 Verify all dataclass defaults are immutable
- [ ] 2.8 Run mypy, fix type errors
- [ ] 2.9 Run ruff, fix lint errors
- [ ] 2.10 Fix analytics.py pandas dtype bug

### 2B: JSON Schema Enforcement (CRITICAL — Finding #3)
- [ ] 2.11 Define JSON schemas for Director output
- [ ] 2.12 Define JSON schemas for Creator output  
- [ ] 2.13 Define JSON schemas for Grader output (structured scores)
- [ ] 2.14 Define JSON schemas for Diversity Guard output
- [ ] 2.15 Define JSON schemas for Consolidator output
- [ ] 2.16 Add response_format parameter to LMStudioClient.chat_structured()
- [ ] 2.17 Add schema validation wrapper with retry on parse failure
- [ ] 2.18 Create src/schemas/ directory with all agent output schemas
- [ ] 2.19 Test: invalid JSON from model → retry with schema → valid output
- [ ] 2.20 Test: schema validation catches missing required fields

### 2C: Qwen3 Thinking Mode (Finding #6)
- [ ] 2.21 Add thinking_mode parameter to LMStudioClient
- [ ] 2.22 Add per-agent model config (model, quant, thinking, temperature)
- [ ] 2.23 Update Director to use thinking=True
- [ ] 2.24 Update Grader to use thinking=False
- [ ] 2.25 Update config/settings.py with per-agent model configs
- [ ] 2.26 Document model selection rationale in config

### 2D: Wire It Together
- [ ] 2.27 Wire LMStudioClient into FlowController agent runners
- [ ] 2.28 Run main.py --cycles 1 with mock LLM — fix everything
- [ ] 2.29 Run main.py --cycles 5 — verify multi-cycle stability
- [ ] 2.30 Run main.py --cycles 50 — verify consolidation triggers
- [ ] 2.31 Profile memory usage across 50 cycles
- [ ] 2.32 Profile SQLite query performance
- [ ] 2.33 Verify WAL mode is enabled (PRAGMA query)
- [ ] 2.34 Test concurrent SharedState access (threading)
- [ ] 2.35 Verify Optuna study persistence (kill + restart)

## Phase 3 — Optuna Upgrade (Hours 3-4.5)
*AutoSampler, JournalStorage, artifacts, multi-objective*

- [ ] 3.1 Install and configure OptunaHub AutoSampler
- [ ] 3.2 Add try/except fallback to TPE if AutoSampler unavailable
- [ ] 3.3 Replace SQLite RDBStorage with JournalStorage
- [ ] 3.4 Add JournalFileBackend configuration
- [ ] 3.5 Add Optuna artifact storage for trial outputs
- [ ] 3.6 Implement upload_artifact() in grader flow
- [ ] 3.7 Add enqueue_trial() for warm-starting studies
- [ ] 3.8 Add PatientPruner wrapping MedianPruner
- [ ] 3.9 Add multi-objective study support (NSGAIISampler)
- [ ] 3.10 Add directions=["maximize","maximize"] for quality+diversity
- [ ] 3.11 Implement Pareto front selection (TOPSIS or weighted sum)
- [ ] 3.12 Update analytics.py for multi-objective studies
- [ ] 3.13 Add parallel trial support (LM Studio batching)
- [ ] 3.14 Test: 4 concurrent trials via LM Studio
- [ ] 3.15 Test: AutoSampler outperforms TPE on first 200 trials
- [ ] 3.16 Test: JournalStorage with 2 concurrent writers
- [ ] 3.17 Test: artifact retrieval for specific trial
- [ ] 3.18 Test: warm-start resumes optimization correctly
- [ ] 3.19 Test: pruner kills unpromising trials early
- [ ] 3.20 Update requirements.txt with optunahub

## Phase 4 — Knowledge Base Upgrade (Hours 4.5-6.5)
*sqlite-vec, nomic embeddings, semantic dedup*

- [ ] 4.1 Install sqlite-vec, verify it loads on macOS
- [ ] 4.2 Document Homebrew Python requirement if needed
- [ ] 4.3 Rewrite knowledge_base.py to use sqlite-vec for vector storage
- [ ] 4.4 Create vec0 virtual table for embeddings
- [ ] 4.5 Implement SQL-based KNN search replacing numpy
- [ ] 4.6 Add nomic-embed-text-v1.5 support with task prefixes
- [ ] 4.7 Implement search_document: prefix for writes
- [ ] 4.8 Implement search_query: prefix for reads
- [ ] 4.9 Implement clustering: prefix for consolidation
- [ ] 4.10 Add Matryoshka dimensionality (768→256 for coarse, 768 for fine)
- [ ] 4.11 Implement two-stage retrieval (coarse→rerank)
- [ ] 4.12 Add semantic deduplication at ingestion (0.95 cosine threshold)
- [ ] 4.13 Add confidence decay with Ebbinghaus curve
- [ ] 4.14 Add hard cap at 10K knowledge items with LRU eviction
- [ ] 4.15 Update consolidator.py for sqlite-vec backend
- [ ] 4.16 Add knowledge migration tool (old format → sqlite-vec)
- [ ] 4.17 Test: write + read cycle with sqlite-vec
- [ ] 4.18 Test: KNN returns correct nearest neighbors
- [ ] 4.19 Test: semantic dedup catches near-duplicates
- [ ] 4.20 Test: confidence decay over 200 cycles
- [ ] 4.21 Test: 1000-entry KB query latency <100ms
- [ ] 4.22 Test: Matryoshka dim reduction preserves retrieval quality
- [ ] 4.23 Benchmark: sqlite-vec vs numpy at 100/1K/10K entries
- [ ] 4.24 Test: migration from old format preserves data
- [ ] 4.25 Update requirements.txt

## Phase 5 — Grading Pipeline Upgrade (Hours 6.5-8.5)
*Verifier cascade, pyiqa, CLIP, process rewards*

- [ ] 5.1 Create src/evaluation/__init__.py
- [ ] 5.2 Build src/evaluation/fast_gate.py — rule-based checks (<10ms)
- [ ] 5.3 Implement format validation (dimensions, file type, color space)
- [ ] 5.4 Implement typography rules (min contrast, font size bounds)
- [ ] 5.5 Implement color palette compliance check
- [ ] 5.6 Build src/evaluation/medium_gate.py — embedding/model checks (~100ms)
- [ ] 5.7 Integrate pyiqa for NIMA aesthetic scoring
- [ ] 5.8 Integrate pyiqa for BRISQUE quality scoring
- [ ] 5.9 Add CLIP text-image alignment scoring
- [ ] 5.10 Add style consistency via CLIP embedding distance
- [ ] 5.11 Build src/evaluation/llm_gate.py — full LLM judge (~2-5s)
- [ ] 5.12 Wire verifier cascade: fast→medium→LLM
- [ ] 5.13 Add cascade short-circuit (fail fast = score 0, skip expensive eval)
- [ ] 5.14 Add process reward scoring (per-step: brief→concept→execution→polish)
- [ ] 5.15 Add multi-probe CLIPScore for brand alignment
- [ ] 5.16 Build rubric calibration system with anchor examples
- [ ] 5.17 Add LAION aesthetic predictor integration
- [ ] 5.18 Add novelty scoring via k-NN CLIP distance from corpus
- [ ] 5.19 Test: fast gate rejects obviously bad output
- [ ] 5.20 Test: medium gate catches subtle quality issues
- [ ] 5.21 Test: cascade saves 60%+ evaluations vs full LLM grading
- [ ] 5.22 Test: process rewards provide better signal than outcome-only
- [ ] 5.23 Benchmark: full cascade latency vs LLM-only latency
- [ ] 5.24 Test: rubric calibration produces consistent scores
- [ ] 5.25 Update requirements.txt with pyiqa, clip

## Phase 6 — Advanced Exploration & Learning (Hours 8.5-10)
*Multi-objective, Reflexion, knowledge graph, Thompson sampling*

- [ ] 6.1 Implement Reflexion pattern — structured reflection per cycle
- [ ] 6.2 Build reflection template (what worked, what failed, what to try)
- [ ] 6.3 Wire reflection output into next cycle's Director context
- [ ] 6.4 Add reflection storage to knowledge base (observation tier)
- [ ] 6.5 Implement Thompson Sampling for strategy selection
- [ ] 6.6 Define strategy space (exploitation, exploration, hypothesis-testing)
- [ ] 6.7 Update bandit arms based on cycle outcomes
- [ ] 6.8 Add temperature scheduling (start high 0.9, anneal to 0.3)
- [ ] 6.9 Implement embedding-based novelty scoring
- [ ] 6.10 Add novelty bonus to Optuna objective
- [ ] 6.11 Build src/knowledge/graph.py with NetworkX
- [ ] 6.12 Add typed edges: supports, contradicts, derived_from, supersedes
- [ ] 6.13 Add PageRank for knowledge importance ranking
- [ ] 6.14 Wire graph queries into context assembler
- [ ] 6.15 Add mode collapse detection (same combo 5+ times in 20 trials)
- [ ] 6.16 Add automatic reset when mode collapse detected
- [ ] 6.17 Implement curriculum learning (simple combos first, complex later)
- [ ] 6.18 Add incremental clustering (BERTopic partial_fit)
- [ ] 6.19 Test: Reflexion improves scores over 50 cycles
- [ ] 6.20 Test: Thompson Sampling converges to best strategy
- [ ] 6.21 Test: knowledge graph edges form correctly
- [ ] 6.22 Test: mode collapse detection triggers correctly
- [ ] 6.23 Test: curriculum learning produces monotonic improvement
- [ ] 6.24 Test: incremental clustering matches batch results
- [ ] 6.25 Benchmark: with vs without Reflexion over 200 cycles

## Phase 7 — Robustness & Error Handling (Hours 10-11)
*Everything that breaks at 3 AM*

- [ ] 7.1 LM Studio down at startup — graceful error + clear message
- [ ] 7.2 LM Studio dies mid-cycle — retry, skip, checkpoint
- [ ] 7.3 LM Studio returns garbage JSON — retry with stricter schema
- [ ] 7.4 SQLite corruption detection + rebuild from journals
- [ ] 7.5 SQLite disk full — detect + graceful shutdown
- [ ] 7.6 Embedding model too large — fallback to TF-IDF
- [ ] 7.7 Build TF-IDF fallback in knowledge_base.py
- [ ] 7.8 All assets deprecated — auto-reset deprecations
- [ ] 7.9 Grader score outside 0-1 — clamp + warn
- [ ] 7.10 Director override >50% — warn Optuna isn't steering
- [ ] 7.11 Consolidation too aggressive — minimum 5 rules threshold
- [ ] 7.12 SIGTERM handling — checkpoint + clean exit
- [ ] 7.13 SIGHUP handling — reload config without restart
- [ ] 7.14 Memory leak detection — RSS tracking over 100 cycles
- [ ] 7.15 Add --dry-run flag (full cycle, no DB writes)
- [ ] 7.16 Add --validate flag (check configs, connections, models)
- [ ] 7.17 Build tools/health_check.py (verify everything)
- [ ] 7.18 Add rate limiting for LLM calls (configurable)
- [ ] 7.19 Add circuit breaker (5 consecutive failures → pause + alert)
- [ ] 7.20 Cycle checkpointing (save state every N cycles)
- [ ] 7.21 Add --resume flag to continue from checkpoint
- [ ] 7.22 Test: each failure mode + recovery
- [ ] 7.23 Test: checkpoint + resume produces same results
- [ ] 7.24 Test: circuit breaker triggers at threshold
- [ ] 7.25 Test: rate limiter caps requests correctly

## Phase 8 — Monitoring & Observability (Hours 11-12.5)
*Dashboards, tracing, reporting*

- [ ] 8.1 Build src/monitoring/__init__.py
- [ ] 8.2 Build src/monitoring/logger.py — structured JSONL logging
- [ ] 8.3 Log every state read/write with agent + timestamp
- [ ] 8.4 Log every knowledge create/promote/archive with reasoning
- [ ] 8.5 Log every Optuna trial with params + score
- [ ] 8.6 Log every LLM call with tokens + latency + model
- [ ] 8.7 Build src/monitoring/dashboard.py — Textual TUI
- [ ] 8.8 Dashboard: current cycle, best score, KB size
- [ ] 8.9 Dashboard: score trend sparkline
- [ ] 8.10 Dashboard: per-agent execution time
- [ ] 8.11 Dashboard: knowledge health (entries per tier)
- [ ] 8.12 Dashboard: parameter importance bars
- [ ] 8.13 Dashboard: diversity alerts
- [ ] 8.14 Dashboard: token usage counter
- [ ] 8.15 Add --dashboard flag to main.py
- [ ] 8.16 Build tools/analyze_run.py — post-hoc analysis
- [ ] 8.17 Generate score improvement curves
- [ ] 8.18 Generate knowledge growth/compaction curves
- [ ] 8.19 Generate parameter convergence plots
- [ ] 8.20 Build tools/export_report.py — markdown report
- [ ] 8.21 Add AgentOps decorators for tracing
- [ ] 8.22 Test: dashboard renders without errors
- [ ] 8.23 Test: JSONL logs are parseable
- [ ] 8.24 Test: analyze_run produces valid charts
- [ ] 8.25 Test: report generation from empty/small/large runs

## Phase 9 — Scale Testing & Documentation (Hours 12.5-14)
*Prove it works, document everything*

### 9A: Scale Stress Tests
- [ ] 9.1 500-cycle simulation with synthetic LLM responses
- [ ] 9.2 1000-cycle simulation — measure KB plateau
- [ ] 9.3 2000-cycle simulation — full stress test
- [ ] 9.4 Measure active KB entries at 500/1000/1500/2000
- [ ] 9.5 Measure retrieval precision at each checkpoint
- [ ] 9.6 Measure Optuna score trend (improve then plateau, not degrade)
- [ ] 9.7 Measure consolidation time per pass (<5s target)
- [ ] 9.8 Measure SQLite DB sizes at 2000 cycles
- [ ] 9.9 Measure cycle time degradation (cycle 1 vs cycle 2000)
- [ ] 9.10 Test: cycle 2000 objectively better than cycle 200
- [ ] 9.11 Identify scaling ceiling
- [ ] 9.12 Test KB recovery — delete 50%, verify rebuild
- [ ] 9.13 Test 10K Optuna trials — analytics still <1s
- [ ] 9.14 Generate visual scale report
- [ ] 9.15 Write docs/SCALE_REPORT.md

### 9B: Model Benchmarking
- [ ] 9.16 Build tools/benchmark_models.py
- [ ] 9.17 Benchmark JSON reliability per model (100 attempts)
- [ ] 9.18 Benchmark tool call accuracy per model
- [ ] 9.19 Benchmark creative quality consistency
- [ ] 9.20 Benchmark multi-turn coherence
- [ ] 9.21 Benchmark response latency per model
- [ ] 9.22 Build comparison matrix
- [ ] 9.23 Write docs/MODEL_GUIDE.md
- [ ] 9.24 Create config/model_profiles/ for each model
- [ ] 9.25 Test mixed model deployment (different per role)

### 9C: Documentation
- [ ] 9.26 Write QUICKSTART.md — 5-minute setup
- [ ] 9.27 Write CONTRIBUTING.md — how to extend
- [ ] 9.28 Write docs/AGENT_DEVELOPMENT.md
- [ ] 9.29 Write docs/TUNING_GUIDE.md
- [ ] 9.30 Write docs/TROUBLESHOOTING.md
- [ ] 9.31 Add architecture diagram to README
- [ ] 9.32 Add data flow diagram
- [ ] 9.33 Clean all TODO/FIXME comments
- [ ] 9.34 Final ruff + mypy pass
- [ ] 9.35 Final commit + push
- [ ] 9.36 Write HANDOFF.md for Cheezfish

---

## Total: 211 tasks across 9 phases
## Estimated: 14 hours with parallel agent execution
## Decision points: After each phase, evaluate if plan needs revision
