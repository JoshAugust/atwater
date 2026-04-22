# Atwater

**A self-improving creative production system powered by multi-agent collaboration and Bayesian optimisation.**

Atwater runs a loop of specialised AI agents — Director, Creator, Grader, Diversity Guard, and Consolidator — that collaboratively explore a creative search space, score outputs against rubrics, and learn from results. Optuna handles the statistical optimisation; a hierarchical knowledge base captures qualitative insights that numbers can't express.

Built for the `caffy_studio_cognitive` framework. Runs on local hardware with LM Studio.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        EVOLUTION LOOP                           │
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────────┐  │
│  │ Director  │──▶│ Creator  │──▶│  Grader  │──▶│ Diversity  │  │
│  │          │   │          │   │          │   │   Guard     │  │
│  │ suggest  │   │ generate │   │  score   │   │  monitor   │  │
│  │ params   │   │ content  │   │  output  │   │  staleness │  │
│  └────┬─────┘   └──────────┘   └────┬─────┘   └────────────┘  │
│       │                              │                          │
│       │         ┌────────────────────┘                          │
│       │         │                                               │
│  ┌────▼─────────▼──┐              ┌─────────────┐              │
│  │   Optuna Study  │              │Consolidator │ (every N)    │
│  │  (JournalStorage│              │ merge/promote│              │
│  │   AutoSampler)  │              │ archive/decay│              │
│  └────────┬────────┘              └──────┬──────┘              │
│           │                              │                      │
│  ┌────────▼──────────────────────────────▼──────┐              │
│  │            Knowledge Base (SQLite)            │              │
│  │  Rules ▸ Patterns ▸ Observations ▸ Archived  │              │
│  │  sqlite-vec KNN │ nomic embeddings │ decay   │              │
│  └──────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
                    ┌─────────────┐
                    │  LM Studio  │ (local models)
                    │ Qwen3-8B/4B │
                    └──────┬──────┘
                           │ OpenAI-compatible API
                           ▼
┌──────────────────────────────────────────────────┐
│              FlowController                       │
│  ┌─────────────────────────────────────────────┐ │
│  │         ContextAssembler                     │ │
│  │  role scoping │ knowledge retrieval │ tools  │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  SharedState ◄──── Agent Results ────► Knowledge  │
│  (SQLite WAL)      (per cycle)        (sqlite-vec)│
│                                                   │
│  Optuna Study ◄── scores ── Grader               │
│  (JournalStorage)                                 │
│                                                   │
│  Checkpoints ◄── auto-save every N cycles         │
└──────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Install
pip install -e .

# Validate (checks LM Studio, DBs, sqlite-vec)
python src/main.py --validate

# Run 10 cycles with stub agents (no LLM needed)
python src/main.py --cycles 10 --verbose

# Run 50 cycles with LM Studio
python src/main.py --cycles 50 --verbose --study-name my-first-run
```

See **[QUICKSTART.md](QUICKSTART.md)** for the full 5-minute guide.

---

## Features

### Core
- **Multi-agent evolution loop** — Director → Creator → Grader → Diversity Guard → Consolidator
- **Bayesian optimisation** via Optuna with AutoSampler, JournalStorage, PatientPruner
- **Three-tier memory** — ephemeral working memory, SQLite shared state with role scoping, persistent knowledge base
- **Hierarchical knowledge** — Rules → Patterns → Observations with confidence tracking and Ebbinghaus decay
- **sqlite-vec KNN search** — semantic retrieval with nomic-embed-text-v1.5 and Matryoshka two-stage retrieval

### Evaluation
- **Three-stage verifier cascade** — fast rules (<10ms) → embedding checks (~100ms) → LLM judge (~2-5s)
- **Process reward scoring** — step-by-step evaluation, not just final output
- **Multi-objective optimisation** — quality + diversity Pareto front via NSGAIISampler

### Learning
- **Reflexion** — structured reflection notes per cycle
- **Strategy selection** — multi-armed bandit for exploration/exploitation
- **Temperature scheduling** — annealing from exploration to exploitation
- **Collapse detection** — monitors for score plateau + parameter convergence

### Production Resilience
- **Circuit breaker** — cascading failure protection with configurable thresholds
- **Checkpointing** — auto-save, resume from checkpoint, emergency save on SIGTERM
- **Graceful shutdown** — signal handling with clean DB close
- **Health checks** — validates LM Studio, DBs, extensions
- **Rate limiting** — token bucket for API calls

---

## Module Overview

```
atwater/
├── src/
│   ├── main.py                    # CLI entry point + cycle runner
│   ├── agents/                    # Agent definitions
│   │   ├── base.py                #   AgentBase ABC + AgentContext/AgentResult
│   │   ├── director.py            #   Hypothesis selection via Optuna
│   │   ├── creator.py             #   Content generation
│   │   ├── grader.py              #   Structured scoring
│   │   ├── diversity_guard.py     #   Stagnation prevention
│   │   └── consolidator_agent.py  #   Knowledge compaction
│   ├── memory/                    # Three-tier memory system
│   │   ├── shared_state.py        #   SQLite state with role scoping
│   │   ├── knowledge_base.py      #   Hierarchical KB + semantic search
│   │   ├── knowledge_store.py     #   Upgraded KB with sqlite-vec + nomic
│   │   └── working.py             #   Ephemeral working memory
│   ├── optimization/              # Optuna integration
│   │   ├── study_manager.py       #   Study lifecycle (AutoSampler, Journal)
│   │   ├── trial_adapter.py       #   SearchSpace + TrialAdapter + artifacts
│   │   └── analytics.py           #   Importances, trends, heatmaps
│   ├── orchestrator/              # Flow control
│   │   ├── flow_controller.py     #   Full cycle sequencing
│   │   └── context_assembler.py   #   Per-turn prompt assembly
│   ├── evaluation/                # Grading pipeline
│   │   ├── cascade.py             #   Three-stage verifier
│   │   ├── fast_gate.py           #   Rule-based checks
│   │   ├── medium_gate.py         #   Embedding-based checks
│   │   ├── llm_gate.py            #   Full LLM grading
│   │   └── process_rewards.py     #   Step-by-step scoring
│   ├── learning/                  # Adaptive learning
│   │   ├── reflexion.py           #   Structured reflection
│   │   ├── strategy_selector.py   #   Multi-armed bandit
│   │   ├── temperature_schedule.py#   Exploration annealing
│   │   └── collapse_detector.py   #   Mode collapse detection
│   ├── resilience/                # Production hardening
│   │   ├── circuit_breaker.py     #   Failure protection
│   │   ├── checkpointing.py       #   Auto-save + resume
│   │   ├── graceful_shutdown.py   #   Signal handling
│   │   ├── health_check.py        #   System validation
│   │   ├── rate_limiter.py        #   API rate limiting
│   │   └── fallbacks.py           #   Degraded operation
│   ├── schemas/                   # JSON schemas for agent output
│   ├── llm/                       # LM Studio client + prompts
│   └── config/                    # Per-agent model config
├── config/
│   ├── settings.py                # Central Settings + env var loader
│   └── openfang_hand.toml         # OpenFang HAND definition (corrected)
├── tests/                         # Unit + scale tests
├── tools/                         # Benchmarking utilities
└── docs/                          # Architecture, research, guides
```

---

## Documentation

| Document | What It Covers |
|----------|---------------|
| **[QUICKSTART.md](QUICKSTART.md)** | 5-minute setup and first run |
| **[HANDOFF.md](HANDOFF.md)** | What's ready, what's not, what needs input |
| **[CONTRIBUTING.md](CONTRIBUTING.md)** | How to add agents, dimensions, tools, tests |
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | Full cognitive architecture specification |
| **[docs/AGENT_DEVELOPMENT.md](docs/AGENT_DEVELOPMENT.md)** | Build custom agents from scratch |
| **[docs/TUNING_GUIDE.md](docs/TUNING_GUIDE.md)** | Configure for your domain |
| **[docs/MODEL_GUIDE.md](docs/MODEL_GUIDE.md)** | Model selection and quantization |
| **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** | Common issues and fixes |
| **[docs/research/SYNTHESIS.md](docs/research/SYNTHESIS.md)** | Research synthesis (20 key findings) |

---

## Stack

- **Python 3.11+** — core runtime
- **Optuna** — Bayesian optimisation, trial management, analytics
- **SQLite** — shared state (WAL mode), knowledge base
- **sqlite-vec** — vector KNN search (optional, falls back to numpy)
- **sentence-transformers** — nomic-embed-text-v1.5 for semantic search
- **LM Studio** — local model inference (OpenAI-compatible API)

---

## License

Private. Contact the maintainers for access.
