# Quickstart — Atwater in 5 Minutes

---

## Prerequisites

| Requirement | Version | Notes |
|------------|---------|-------|
| Python | 3.11+ | 3.12 recommended. Use `brew install python` on macOS for sqlite-vec support. |
| LM Studio | 0.4.0+ | For local model inference. [Download](https://lmstudio.ai/) |
| RAM | 16GB+ | 8B models need ~10GB VRAM/unified memory |

### Recommended Models (download in LM Studio)

| Model | Use Case | Quantization |
|-------|----------|-------------|
| **Qwen3-8B** | Director, Creator, Orchestrator, Consolidator | Q8_0 (best) or Q5_K_M |
| **Qwen3-4B** | Grader, Diversity Guard | Q8_0 |

Load at least one model in LM Studio before your first real run. The stub runners work without any model loaded.

---

## Install

```bash
git clone <your-repo-url> atwater
cd atwater

# Option A: editable install (recommended for development)
pip install -e .

# Option B: requirements only
pip install -r requirements.txt
```

### Verify sqlite-vec (optional but recommended)

```bash
python -c "import sqlite_vec; print('sqlite-vec OK')"
```

If this fails, Atwater falls back to numpy cosine similarity (slower at scale, still works). Fix with:

```bash
# macOS: use Homebrew Python (system Python restricts extensions)
brew install python
/opt/homebrew/bin/python3 -m pip install sqlite-vec

# Linux
pip install sqlite-vec
```

---

## Configure

Copy the default settings:

```bash
mkdir -p config
cat > config/settings.json << 'EOF'
{
  "lm_studio_url": "http://localhost:1234/v1",
  "model_name": null,
  "state_db_path": "state.db",
  "knowledge_db_path": "knowledge.db",
  "optuna_db_path": "optuna_journal.log",
  "consolidation_interval": 50,
  "study_name": "atwater_production",
  "log_level": "INFO"
}
EOF
```

Or use environment variables (override JSON):

```bash
export ATWATER_LM_STUDIO_URL="http://localhost:1234/v1"
export ATWATER_MODEL_NAME="qwen3-8b-q8_0"
```

---

## First Run

### Validate the installation

```bash
python src/main.py --validate
```

This checks:
- LM Studio connectivity
- Database paths are writable
- sqlite-vec extension loads
- Required Python packages installed

### Run 10 cycles (stub agents, no LLM needed)

```bash
python src/main.py --cycles 10 --verbose
```

You'll see output like:

```
Atwater — starting 10 cycle(s)  [study='atwater-default']
  state_db     : state.db
  knowledge_db : knowledge.db
  optuna_db    : optuna_journal.log

  Running 10 cycle(s) (start=1)...

  [████░░░░░░░░░░░░░░░░]    1/10  score=0.5000  kb_writes= 0  elapsed=  0.1s
  [████████░░░░░░░░░░░░]    2/10  score=0.5000  kb_writes= 0  elapsed=  0.2s
  ...
```

Scores are flat at 0.5 because stub runners return fixed values. This proves the full architecture works: Optuna trials, shared state, knowledge writes, checkpointing, and consolidation all fire correctly.

### Run with LM Studio (real agents)

1. Start LM Studio, load Qwen3-8B
2. Ensure the server is running at `http://localhost:1234/v1`
3. Run:

```bash
python src/main.py --cycles 50 --verbose --study-name my-first-run
```

---

## What to Expect on First Run

- **Cycles 1-20:** Scores are noisy. Optuna is exploring the search space randomly (AutoSampler uses broad exploration early).
- **Cycles 20-100:** Scores start trending upward. TPE kicks in, exploiting what worked.
- **Cycle 50:** First consolidation pass runs — knowledge entries get merged, promoted, or archived.
- **Cycles 100+:** Scores plateau near the best achievable for your search space. Knowledge base stabilises.

### Files created

| File | Purpose |
|------|---------|
| `state.db` | Shared agent state (SQLite WAL) |
| `knowledge.db` | Knowledge base with semantic embeddings |
| `optuna_journal.log` | Optuna trial history (append-only) |
| `checkpoints/` | Auto-saved cycle snapshots |

---

## Key CLI Options

```bash
python src/main.py --help

# Common usage patterns:
python src/main.py --cycles 100 --verbose              # standard run
python src/main.py --cycles 50 --dry-run               # test without DB writes
python src/main.py --resume --cycles 100                # resume from checkpoint
python src/main.py --validate                           # health check only
python src/main.py --config config/production.json      # custom config file
python src/main.py --study-name experiment-v2           # named study
python src/main.py --consolidation-interval 25          # consolidate more often
python src/main.py --checkpoint-every 5                 # checkpoint more often
```

---

## Troubleshooting Common Issues

### "Failed to import memory module"
Missing dependency. Run `pip install -e .` or `pip install -r requirements.txt`.

### "SharedState init failed"
SQLite database path is not writable. Check permissions on the directory.

### "Optuna study init failed"
JournalStorage log file cannot be created. Check the `optuna_db_path` setting.

### LM Studio connection refused
1. Is LM Studio running? Check the app.
2. Is a model loaded? The API returns 404 if no model is active.
3. Is the port correct? Default is 1234. Check your `lm_studio_url` setting.

### "sqlite-vec extension not loaded"
macOS default Python restricts C extensions. Use Homebrew Python or set `use_vec=False` in KnowledgeStore constructor (automatic fallback).

### Score never improves
- With stub agents: expected. Stubs return fixed 0.5.
- With real agents: check that the grader's JSON output is parsing correctly. Enable `--verbose` and look for JSON parse errors in stderr.

---

## Next Steps

- **[CONTRIBUTING.md](CONTRIBUTING.md)** — How to add agents, search dimensions, and tools
- **[docs/AGENT_DEVELOPMENT.md](docs/AGENT_DEVELOPMENT.md)** — Build custom agents from scratch
- **[docs/TUNING_GUIDE.md](docs/TUNING_GUIDE.md)** — Tune the system for your domain
- **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** — Full troubleshooting guide
- **[HANDOFF.md](HANDOFF.md)** — What's ready, what's not, what needs your input
