# Troubleshooting

Common issues and how to fix them.

---

## LM Studio Connection Issues

### Symptom: "Connection refused" or timeout on startup

**Check 1: Is LM Studio running?**
```bash
curl http://localhost:1234/v1/models
```
Should return a JSON list of loaded models. If it fails, start LM Studio.

**Check 2: Is a model loaded?**
LM Studio's API returns 404 if no model is active. Load a model in the GUI or via API:
```bash
curl http://localhost:1234/v1/models
# Should show at least one model
```

**Check 3: Is the URL correct?**
Default is `http://localhost:1234/v1`. Check your config:
```bash
cat config/settings.json | grep lm_studio_url
echo $ATWATER_LM_STUDIO_URL
```

**Check 4: Is another process using port 1234?**
```bash
lsof -i :1234
```

### Symptom: Slow responses (>30s per cycle)

- Check LM Studio's "Performance" tab for GPU utilisation
- Reduce `n_parallel` if running multiple concurrent requests
- Use a smaller model (Qwen3-4B instead of 8B) for the Grader role
- Enable speculative decoding: load Qwen3-0.5B as draft model alongside Qwen3-8B

---

## sqlite-vec Extension Not Loading (macOS)

### Symptom: `KnowledgeStore` falls back to numpy mode

```
KnowledgeStore(db=..., mode='numpy-fallback', ...)
```

**Root cause:** macOS system Python restricts `enable_load_extension()` in its bundled SQLite.

### Fix options

**Option A: Use Homebrew Python (recommended)**
```bash
brew install python
/opt/homebrew/bin/python3 -m pip install sqlite-vec
/opt/homebrew/bin/python3 -c "import sqlite_vec; print('OK')"
```

**Option B: Install pysqlite3**
```bash
pip install pysqlite3-binary
```
Then in your code, monkey-patch before importing sqlite3:
```python
import pysqlite3
import sys
sys.modules["sqlite3"] = pysqlite3
```

**Option C: Accept the fallback**
numpy cosine similarity works fine for < 1,000 knowledge entries. It's O(N) instead of O(log N), so it degrades above 5K entries. For small-to-medium runs, this is fine.

### Verification
```bash
python -c "
import sqlite3
conn = sqlite3.connect(':memory:')
conn.enable_load_extension(True)
import sqlite_vec
sqlite_vec.load(conn)
print('sqlite-vec loaded successfully')
"
```

---

## JSON Parse Failures (Schema Enforcement)

### Symptom: Grader output fails to parse, trial gets score=None

**Root cause:** Small models (4B-8B) produce invalid JSON 15-30% of the time without constrained generation.

### Fix: Enable response_format

Every LM Studio API call should include:
```python
response = client.chat.completions.create(
    model="qwen3-8b",
    messages=[...],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "grader_output",
            "schema": GRADER_SCHEMA,
        },
    },
)
```

### Verify LM Studio supports it

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b",
    "messages": [{"role": "user", "content": "Score this: test"}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "test",
        "schema": {
          "type": "object",
          "properties": {"score": {"type": "number"}},
          "required": ["score"]
        }
      }
    }
  }'
```

If LM Studio doesn't enforce the schema (returns free-form text), check:
- LM Studio version ≥ 0.4.0
- The model supports structured output (Qwen3, Llama 3, Gemma 2 do; some older models don't)

### Fallback: Retry with regex extraction

If constrained generation isn't available, the LLM client should:
1. Try parsing the full response as JSON
2. Try extracting JSON from markdown code blocks (` ```json ... ``` `)
3. Try regex extraction of `{...}` blocks
4. Fail the trial and let the circuit breaker handle it

---

## Knowledge Base Growing Too Fast

### Symptom: >1000 active entries after 200 cycles

**Root cause:** Every cycle with a novel finding writes an observation. If the grader is too liberal with `suggest_knowledge_write`, the KB bloats.

### Fixes

**1. Increase deduplication threshold sensitivity**

In `src/memory/knowledge_store.py`:
```python
DEDUP_COSINE_THRESHOLD = 0.90  # was 0.95; catches more near-duplicates
```

**2. Run consolidation more often**

```bash
python src/main.py --consolidation-interval 25  # was 50
```

**3. Lower the knowledge write threshold in the Grader**

Only write observations when the score is genuinely interesting (top/bottom 10%):
```python
if score > study.best_value * 0.95 or score < 0.2:
    suggest_knowledge_write = True
```

**4. Apply decay more aggressively**

Reduce `EBBINGHAUS_STABILITY` in `knowledge_store.py`:
```python
EBBINGHAUS_STABILITY = 10.0  # was 20.0; entries decay faster
```

**5. Check the 10K hard cap**

The KnowledgeStore automatically evicts when hitting 10K items (archived first, then lowest-confidence). If you're hitting the cap, consolidation isn't keeping up.

---

## Mode Collapse Detected

### Symptom: CollapseDetector fires alerts; scores plateau; same parameters repeat

**Root cause:** Optuna found a local optimum and is stuck exploiting the same region of the search space.

### Fixes

**1. Forced exploration** — the Diversity Guard triggers this every 50 cycles by default, injecting a RandomSampler trial. Increase frequency:
```bash
# In flow_controller.py or via config
diversity_guard_exploration_interval = 25
```

**2. Reheat the temperature schedule**

Temporarily increase temperature to force broader exploration:
```python
schedule = TemperatureSchedule(reheat_interval=100, reheat_temp=0.7)
```

**3. Add new search dimensions**

If all existing dimensions are well-explored, add new ones. Optuna will explore the expanded space.

**4. Reset the study (nuclear option)**

```python
# Start a new study but warm-start with the best params from the old one
old_best = study.best_params
new_study = create_study(name="fresh-start")
warm_start_study(new_study, [old_best])
```

---

## Circuit Breaker Tripping

### Symptom: "Circuit breaker OPEN at cycle N" errors

**Root cause:** 5+ consecutive cycle failures (configurable threshold). The circuit breaker opens to prevent cascading failures.

### Diagnosis

```python
stats = circuit_breaker.get_stats()
print(stats)
# {'state': 'OPEN', 'total_trips': 3, 'total_failures': 15, 'total_calls': 200}
```

### Common triggers

| Trigger | Fix |
|--------|-----|
| LM Studio crashed/OOM | Restart LM Studio; reduce model size or batch size |
| Rate limiting | Increase rate limiter token bucket; reduce request frequency |
| JSON parse failures | Enable constrained generation (see above) |
| Database locked | Check for another Atwater process; increase SQLite busy_timeout |

### Adjusting sensitivity

```python
circuit_breaker = CircuitBreaker(
    failure_threshold=10,    # was 5; more tolerant
    recovery_timeout=30,     # was 60; try recovery sooner
    half_open_max=3,         # was 2; more recovery attempts
)
```

---

## Checkpoint Corruption

### Symptom: `--resume` fails or loads stale state

**Check checkpoint integrity:**
```bash
ls -la checkpoints/
# Look for the most recent .json file
python -c "
import json
with open('checkpoints/latest.json') as f:
    print(json.dumps(json.load(f), indent=2))
"
```

### Recovery options

**1. Skip corrupted checkpoint**
```bash
# Start fresh (ignores checkpoints)
python src/main.py --cycles 100  # no --resume
```

**2. Use an older checkpoint**
The `CheckpointManager` keeps the last 5 checkpoints (configurable via `keep_last`). Manually copy an older one:
```bash
ls checkpoints/  # find an older .json
cp checkpoints/checkpoint_cycle_90.json checkpoints/latest.json
python src/main.py --resume --cycles 100
```

**3. Rebuild from Optuna journal**
The Optuna journal log (`optuna_journal.log`) is the source of truth for trial history. Even if checkpoints and state DBs are lost, the full trial history survives:
```python
study = load_study(name="my-study", storage_path="optuna_journal.log")
print(f"Trials: {len(study.trials)}")
print(f"Best: {study.best_params}")
```

The knowledge base and shared state need to be rebuilt from scratch, but Optuna's statistical memory is preserved.
