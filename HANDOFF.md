# HANDOFF.md — For Cheezfish

> Read this first. Everything else is reference material.

---

## What Atwater Is

Atwater is a **self-improving creative production system**. It runs a loop of agents — Director, Creator, Grader, Diversity Guard, Consolidator — that collaboratively generate creative assets (initially ad creatives for Caffy Studio), score them, and learn from results. Optuna handles the statistical optimisation; a hierarchical knowledge base captures qualitative insights.

Think of it as "Bayesian optimisation meets a team of AI agents" — the agents bring creative reasoning, Optuna brings mathematical rigour, and the knowledge base is institutional memory.

---

## What's Solid and Production-Ready

### ✅ Core Architecture
- **Three-tier memory system** (Working → SharedState → KnowledgeBase) — fully implemented, tested, SQLite-backed with WAL mode
- **SharedState** with role-based scoping — agents only see state keys they're allowed to read
- **KnowledgeBase** with semantic search, tier-aware retrieval (Rules → Patterns → Observations), confidence tracking, and Ebbinghaus decay
- **KnowledgeStore** (upgraded) with sqlite-vec KNN search, Matryoshka two-stage retrieval, semantic deduplication, and 10K item cap with LRU eviction

### ✅ Optuna Integration
- **JournalStorage** (concurrent-safe, append-only) replaces SQLite RDB
- **AutoSampler** (OptunaHub) auto-selects GPSampler/TPESampler based on trial count
- **PatientPruner** wrapping MedianPruner for expensive LLM evaluations
- **Multi-objective studies** via NSGAIISampler (quality + diversity Pareto front)
- **Warm-start** with `enqueue_trial()` for transfer learning across runs
- **Trial artifacts** — store generated outputs alongside trial scores
- **ParallelTrialRunner** — thread-pool-based concurrent trial execution

### ✅ FlowController & Context Assembly
- Full evolution loop: Director → Creator → Grader → DiversityGuard → Consolidator
- **ContextAssembler** builds token-budget-aware prompts with role scoping, semantic knowledge retrieval, Optuna context injection, and tool schema loading
- Stub runners for all agent roles — the architecture compiles and runs end-to-end
- Knowledge writes triggered by grader results flow automatically

### ✅ Agent Framework
- `AgentBase` ABC with READ → DECIDE → WRITE protocol
- State key validation (read/write permissions per role)
- Agent implementations for all roles: Director, Creator, Grader, DiversityGuard, Consolidator
- JSON schema definitions for structured output per agent

### ✅ Evaluation Pipeline
- **Three-stage verifier cascade**: Fast Gate (rules, <10ms) → Medium Gate (embedding, ~100ms) → LLM Gate (~2-5s)
- **Process reward scoring** — step-by-step evaluation, not just final output
- Cascade short-circuits: failed at Fast Gate = immediate rejection, saves expensive LLM calls

### ✅ Learning System
- **Reflexion** — structured reflection notes per cycle that seed next cycle's prompt
- **Strategy selector** — multi-armed bandit for exploration/exploitation balance
- **Temperature schedule** — annealing from exploration to exploitation over time
- **Collapse detector** — monitors for mode collapse (score plateau + parameter convergence)

### ✅ Resilience
- **Circuit breaker** — protects against cascading LLM failures (open/half-open/closed states)
- **Checkpointing** — auto-save every N cycles, resume from checkpoint, emergency save on SIGTERM
- **Graceful shutdown** — SIGINT/SIGTERM handlers with clean DB close
- **Health checker** — validates LM Studio connection, DB paths, sqlite-vec availability
- **Rate limiter** — token bucket for API calls
- **Fallback chains** — degraded operation when subsystems fail

### ✅ CLI
- `python src/main.py --cycles N --verbose --study-name X`
- `--validate` mode for health checks
- `--dry-run` mode (full cycles, no DB writes)
- `--resume` from last checkpoint
- Configurable via JSON (`config/settings.json`) + env vars (`ATWATER_*`)

---

## What Needs Your Input

### 🔧 Search Space Definition
The default search space in `src/optimization/trial_adapter.py` is generic:
```python
categorical: background, layout, shot, typography
continuous: bg_opacity, font_scale, padding_ratio, contrast_ratio
integer: hero_font_size
```
**You need to define the actual Caffy Studio dimensions** — what backgrounds you have, what layouts exist, what product shots are available, what brand fonts/colors are in play. This is the most impactful thing you can do first.

### 🔧 Brand Assets
The system optimises over a search space — but the search space needs to map to real assets. You need:
- A directory of background images/styles
- A set of layout templates
- Product shot variants
- Typography presets matching brand guidelines

### 🔧 Grading Rubrics
The grader scores along dimensions (originality, brand_alignment, technical_quality). **You define what "good" means for Caffy Studio.** The rubric lives in knowledge base entries at the "rule" tier. Seed these manually before the first real run:
```python
kb.knowledge_write(
    "Brand alignment requires: dark navy or charcoal backgrounds, "
    "Inter/Satoshi typography, minimum 4.5:1 contrast ratio",
    tier="rule", confidence=1.0, topic_cluster="brand"
)
```

### 🔧 Evaluation Criteria for Fast Gate
`src/evaluation/fast_gate.py` has rule-based checks. Add your hard constraints:
- Required colour palette
- Minimum text contrast
- Forbidden layout combinations
- Required brand elements

---

## What to Try First

1. **Install and validate:**
   ```bash
   cd atwater
   pip install -e .
   python src/main.py --validate
   ```

2. **Run 10 cycles with stubs** (no LM Studio needed):
   ```bash
   python src/main.py --cycles 10 --verbose
   ```
   This proves the full loop works. Scores will be flat (stub grader returns 0.5) but the Optuna study, knowledge writes, checkpointing, and consolidation all fire.

3. **Start LM Studio with Qwen3-8B** (Q8_0 recommended):
   ```
   http://localhost:1234/v1
   ```
   Wire real agents into the FlowController by replacing stub runners.

4. **Seed your search space and rubrics**, then:
   ```bash
   python src/main.py --cycles 50 --verbose --study-name caffy-v1
   ```

5. **Check Optuna Dashboard:**
   ```bash
   pip install optuna-dashboard
   optuna-dashboard optuna_journal.log
   ```

---

## Known Limitations

| Limitation | Impact | Workaround |
|-----------|--------|-----------|
| **Agents use stubs by default** | No real LLM calls until you wire in LM Studio client | Stubs let you test the full architecture without a GPU |
| **sqlite-vec may not load on macOS default Python** | Falls back to numpy brute-force (slower at scale) | Use `brew install python` or `pip install pysqlite3` |
| **Single-threaded by default** | One trial at a time | `ParallelTrialRunner` exists but needs LM Studio parallel batching |
| **No image generation** | Creator produces text prompts, not actual images | Wire in Stable Diffusion/DALL-E/ComfyUI as a tool |
| **Knowledge base semantic search requires model download** | First run downloads nomic-embed-text-v1.5 (~250MB) | Falls back to all-MiniLM-L6-v2 (90MB) if nomic fails |
| **OpenFang integration deferred** | No dashboard, no scheduling via OpenFang | Run standalone; OpenFang is Phase 8 |

---

## What Would Improve with Fine-Tuning

1. **Grader consistency** — A fine-tuned 4B grader model on your rubric would be faster and more consistent than prompt-engineering a general model
2. **Director strategy** — After 500+ trials with a baseline model, you'll have enough data to fine-tune a Director that naturally suggests productive parameter combinations
3. **Creator style alignment** — Fine-tuning on approved brand assets would dramatically improve first-pass quality
4. **JSON reliability** — Even with constrained generation, fine-tuning on your exact output schemas improves parsing success rate from ~85% to ~98%

**Don't fine-tune yet.** Run 500+ cycles with the base system first. The data you collect IS the fine-tuning dataset.

---

## OpenFang Integration Status

**Status: Deferred to Phase 8.**

The original HAND.toml was fundamentally broken (see `docs/research/SYNTHESIS.md` Finding #1 and #2). Key issues:
- Missing required `id` field
- No `[agent]` section
- Used invented config fields that Serde silently ignores
- Misunderstood the execution model (Hands are chat agents, not process managers)

A **corrected HAND.toml** is available at `config/openfang_hand.toml` with the right schema. The integration model is:
- OpenFang Hand calls `shell_exec("python -m atwater.main --cycles 1")` per cycle
- Dashboard metrics read from OpenFang's `memory_store` (not direct DB access)
- Schedule management via `schedule_create` tool at runtime (not HAND.toml config)

**Recommendation:** Get the standalone system working well first. OpenFang adds monitoring/scheduling value but is not needed for core optimisation.

---

## Project Structure Quick Reference

```
atwater/
├── src/
│   ├── main.py                  # CLI entry point, cycle runner
│   ├── agents/                  # Agent definitions (Director, Creator, Grader, etc.)
│   │   ├── base.py              # AgentBase ABC + AgentContext/AgentResult
│   │   ├── director.py          # Hypothesis selection
│   │   ├── creator.py           # Content generation
│   │   ├── grader.py            # Structured scoring
│   │   ├── diversity_guard.py   # Stagnation prevention
│   │   └── consolidator_agent.py # Knowledge compaction
│   ├── memory/                  # Three-tier memory system
│   │   ├── shared_state.py      # SQLite shared state with role scoping
│   │   ├── knowledge_base.py    # Hierarchical KB with semantic search
│   │   ├── knowledge_store.py   # Upgraded KB with sqlite-vec + nomic embeddings
│   │   └── working.py           # Ephemeral working memory
│   ├── optimization/            # Optuna integration
│   │   ├── study_manager.py     # Study lifecycle (AutoSampler, JournalStorage)
│   │   ├── trial_adapter.py     # SearchSpace + TrialAdapter + artifacts
│   │   └── analytics.py         # Statistical queries (importances, trends)
│   ├── orchestrator/            # Flow control
│   │   ├── flow_controller.py   # Full cycle sequencing
│   │   └── context_assembler.py # Per-turn prompt assembly
│   ├── evaluation/              # Grading pipeline
│   │   ├── fast_gate.py         # Rule-based checks (<10ms)
│   │   ├── medium_gate.py       # Embedding-based checks (~100ms)
│   │   ├── llm_gate.py          # Full LLM grading (~2-5s)
│   │   ├── cascade.py           # Three-stage verifier cascade
│   │   └── process_rewards.py   # Step-by-step scoring
│   ├── learning/                # Adaptive learning
│   │   ├── reflexion.py         # Structured reflection per cycle
│   │   ├── strategy_selector.py # Multi-armed bandit
│   │   ├── temperature_schedule.py # Exploration annealing
│   │   └── collapse_detector.py # Mode collapse detection
│   ├── resilience/              # Production hardening
│   │   ├── circuit_breaker.py   # Cascading failure protection
│   │   ├── checkpointing.py     # Auto-save + resume
│   │   ├── graceful_shutdown.py # Signal handling
│   │   ├── health_check.py      # System validation
│   │   ├── rate_limiter.py      # API rate limiting
│   │   └── fallbacks.py         # Degraded operation
│   ├── schemas/                 # JSON schemas for structured output
│   ├── llm/                     # LM Studio client + prompt templates
│   └── config/                  # Agent-level config
├── config/
│   ├── settings.py              # Central Settings dataclass + loader
│   └── openfang_hand.toml       # Corrected OpenFang HAND definition
├── tests/                       # Comprehensive test suite
├── docs/                        # Architecture, research, guides
└── tools/                       # Benchmarking utilities
```

---

*Last updated: 2026-04-22. Questions? The code is the source of truth.*
