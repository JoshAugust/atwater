# Build Plan: Atwater Reference Implementation

## Overview
Build a complete, runnable reference implementation of the cognitive agent architecture.
Target: OpenFang 0.6 compatible, Python, lean dependencies, LM Studio for inference.

## Tasks

### Wave 1 — Core Memory System (parallel)
- [ ] T1: Three-tier memory system (working/shared/knowledge) — Sonnet — 20 min
  - `src/memory/working.py` — ephemeral per-turn state
  - `src/memory/shared_state.py` — SQLite-backed state machine with WAL
  - `src/memory/knowledge_base.py` — hierarchical storage with semantic search
  - `src/memory/__init__.py` — clean exports
  
- [ ] T2: Optuna optimization layer — Sonnet — 20 min
  - `src/optimization/study_manager.py` — study creation, search space definition
  - `src/optimization/trial_adapter.py` — bridge between Optuna trials and shared state
  - `src/optimization/analytics.py` — statistical queries, importance, heatmaps
  - `src/optimization/__init__.py`

- [ ] T3: Knowledge consolidation engine — Sonnet — 25 min
  - `src/knowledge/consolidator.py` — merge, promote, archive, confidence decay
  - `src/knowledge/clustering.py` — HDBSCAN topic clustering
  - `src/knowledge/models.py` — KnowledgeEntry dataclass, tier definitions
  - `src/knowledge/__init__.py`

### Wave 2 — Agent Definitions (parallel, after Wave 1)
- [ ] T4: Agent implementations — Sonnet — 30 min
  - `src/agents/director.py` — hypothesis selection via Optuna
  - `src/agents/creator.py` — content generation with self-critique
  - `src/agents/grader.py` — structured evaluation + knowledge writes
  - `src/agents/diversity_guard.py` — stagnation prevention
  - `src/agents/consolidator_agent.py` — periodic knowledge compaction
  - `src/agents/base.py` — shared agent interface
  - `src/agents/__init__.py`

- [ ] T5: Context assembler / Orchestrator — Sonnet — 25 min
  - `src/orchestrator/context_assembler.py` — per-turn prompt assembly with scoping
  - `src/orchestrator/flow_controller.py` — agent sequencing, decision tree
  - `src/orchestrator/tool_loader.py` — lazy loading, semantic tool selection
  - `src/orchestrator/__init__.py`

### Wave 3 — Integration + Config (after Wave 2)
- [ ] T6: OpenFang + LM Studio integration — Sonnet — 20 min
  - `src/llm/client.py` — LM Studio API client (OpenAI-compatible)
  - `src/llm/prompts.py` — prompt templates per agent role
  - `config/openfang.toml` — OpenFang hand configuration
  - `config/lmstudio.json` — model + endpoint config

- [ ] T7: Main runner + stress tests — Sonnet — 25 min
  - `src/main.py` — full cycle runner
  - `tests/test_memory.py` — memory tier tests
  - `tests/test_consolidation.py` — knowledge compaction tests
  - `tests/test_scale.py` — synthetic 1000-entry stress test
  - `requirements.txt` — lean dependencies

## Parallel Groups
- **Wave 1** (parallel): T1, T2, T3
- **Wave 2** (parallel, after Wave 1): T4, T5
- **Wave 3** (sequential, after Wave 2): T6, T7

## Constraints
- Python 3.11+, minimal dependencies
- No LangChain, no heavyweight frameworks
- Optuna + SQLite + sentence-transformers + scikit-learn (HDBSCAN)
- LM Studio via OpenAI-compatible API
- All code must be importable and runnable standalone
