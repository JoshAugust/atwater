# Atwater

Reference implementation of a cognitive agent architecture with statistical optimization.

Built for the `caffy_studio_cognitive` framework running on OpenFang 0.6 + LM Studio.

## Architecture

- **Three-tier memory model**: Ephemeral working memory → Shared state machine → Persistent knowledge base
- **Statistical backbone**: Optuna for combinatorial experiment tracking across N-dimensional asset spaces
- **Knowledge sustainability**: Hierarchical consolidation (Rules → Patterns → Observations) with confidence decay
- **Agent roles**: Director, Creator, Grader, Diversity Guard, Orchestrator, Consolidator

## Stack

- Python (lean, minimal dependencies)
- Optuna (experiment tracking + Bayesian optimization)
- SQLite (shared state, WAL mode for concurrency)
- OpenFang 0.6 (agent orchestration)
- LM Studio (local model inference)

## Structure

```
atwater/
├── README.md
├── docs/                    # Research & architecture docs
│   ├── ARCHITECTURE.md      # Full cognitive architecture spec
│   ├── CONTEXT_TIPS.md      # Practical tips for small model agents
│   ├── KNOWLEDGE_SCALING.md # Knowledge sustainability at scale
│   ├── OPTUNA_INTEGRATION.md# Statistical experiment tracking
│   └── BUILD_PLAN_16H.md   # 16-hour implementation plan
├── src/                     # Reference implementation
│   ├── memory/              # Three-tier memory system
│   ├── agents/              # Agent definitions + prompts
│   ├── optimization/        # Optuna integration layer
│   ├── knowledge/           # Consolidation + hierarchical storage
│   └── orchestrator/        # Flow control + context assembly
├── config/                  # OpenFang + LM Studio configs
└── tests/                   # Scale stress tests
```
