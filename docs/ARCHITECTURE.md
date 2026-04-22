# Cognitive Architecture Specification v2.0

Evolved from the caffy_studio_cognitive v1.0 spec. Adds statistical grounding, knowledge sustainability, and context-aware orchestration.

---

## Core Loop

All agent actions follow: **READ → DECIDE → WRITE**. Never skip a step.

---

## Memory Tiers

### Tier 1: Working Memory (Ephemeral)
- Holds variables, immediate inputs, current turn state
- Agents read at start of turn, write temporary results
- **Nothing important lives here** — cleared between turns

### Tier 2: Shared State (State Machine)
- Authoritative transient record of the current production run
- SQLite-backed with WAL mode for concurrent access
- Tools: `state_read(key)`, `state_write(key, value)`

**Context scoping rule:** Each agent only sees state keys tagged for its role. The orchestrator pre-filters.

| Agent | Reads | Writes |
|-------|-------|--------|
| director | current_hypothesis, historical_success_rates | proposed_hypothesis |
| creator | current_hypothesis, last_successful_layout | output_path, self_critique |
| grader | output_path, grading_rubric | score, structured_analysis |
| diversity_guard | asset_usage_counts, deprecation_threshold | asset_status |
| orchestrator | ALL | workflow_state, next_agent |

### Tier 3: Knowledge Base (Persistent, Hierarchical)

Three sub-tiers with different lifecycles:

| Level | What | Example | Lifespan |
|-------|------|---------|----------|
| **Rules** | Hard constraints, proven invariants | "Sans-serif headlines outperform serif by 23% across 200+ tests" | Permanent until overturned |
| **Patterns** | Reliable heuristics | "Dark backgrounds work better for tech products" | Re-validated every 100 cycles |
| **Observations** | Single-cycle findings | "Layout C with gradient scored 0.91 on run #847" | Auto-archived after 50 cycles unless promoted |

Tools:
- `knowledge_read(query, tier=None)` — semantic search, optionally filtered by tier
- `knowledge_write(content, tier, confidence, metadata)`
- `knowledge_promote(entry_id, from_tier, to_tier, evidence)`

**Retrieval priority:** Rules → Patterns → Observations (stop early if Rules answer the query).

---

## Statistical Layer: Optuna Integration

### Purpose
Quantitative experiment tracking lives OUTSIDE the knowledge base. Optuna handles combo performance; the knowledge base handles qualitative insights.

### Search Space Definition

```python
import optuna

def create_study(study_name="production_run"):
    return optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage="sqlite:///trials.db",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42)
    )

def suggest_params(trial):
    return {
        "background": trial.suggest_categorical("background", BACKGROUNDS),
        "layout": trial.suggest_categorical("layout", LAYOUTS),
        "product_shot": trial.suggest_categorical("shot", SHOTS),
        # Continuous parameters
        "opacity": trial.suggest_float("opacity", 0.3, 1.0),
        "font_size": trial.suggest_int("font_size", 14, 48),
        # Add dimensions as needed
    }
```

### Agent Integration

| Agent | Optuna Role |
|-------|-------------|
| director | Calls `trial.suggest_*()` to pick next combination |
| creator | Executes the combo, produces output |
| grader | Returns score via `study.tell()` or objective return |
| diversity_guard | Reads `study.trials_dataframe()`, checks asset concentration |
| consolidator | Reads `get_param_importances()`, writes interpretations to knowledge base |

### Statistical Queries (Free from Optuna)

```python
# Best combo
study.best_params

# Parameter importance
optuna.importance.get_param_importances(study)

# Per-dimension analysis
df = study.trials_dataframe()
df.groupby("params_background")["value"].mean()
df.groupby(["params_background", "params_layout"])["value"].agg(["mean", "std", "count"])
```

---

## Context Assembly (Orchestrator's Key Job)

The orchestrator is a **context assembler**, not just a router.

### Per-Turn Prompt Assembly

```
1. Base system prompt (static, ~200 tokens)
2. Agent-specific instructions (~100-200 tokens)
3. Scoped shared state (filtered by role, ~50-100 tokens)
4. Relevant knowledge entries (top 3 by semantic similarity, ~150-300 tokens)
5. Optuna context (best params, recent trial summary, ~100 tokens)
6. Tool schemas (only relevant group, ~200-400 tokens)
7. Current task instruction (~50-100 tokens)

Total: ~850-1500 tokens per agent call
```

### Tool Loading

Two-tier: catalog first, full schemas on demand.

```python
def assemble_tools(agent_role, task_description):
    # Semantic match: which tool group does this task need?
    relevant_group = select_tools(task_description, top_k=1)[0]
    return TOOL_SCHEMAS[relevant_group]
```

---

## Agent Definitions

### 1. Director Engine
- **Goal:** Select next best parameter combination
- **Input:** Optuna study state + knowledge rules + shared state constraints
- **Output:** Proposed hypothesis written to shared state
- **Key rule:** Never manually pick combos. Always use `trial.suggest_*()`. The director can override by requesting specific fixed params only when testing a knowledge-base hypothesis.

### 2. Creator
- **Goal:** Execute content generation for the proposed combo
- **Input:** Hypothesis from shared state + relevant knowledge patterns
- **Output:** Generated content + self-critique
- **Key rule:** After generating, run self-critique against knowledge base. If approach was novel AND scored well, suggest a knowledge write.

### 3. Grader Engine
- **Goal:** Evaluate quality, produce structured scores
- **Input:** Creator output + grading rubric from knowledge base
- **Output:** Structured analysis (metric, score, reasoning)
- **Key rule:** Every score must include reasoning. This agent is the primary driver of knowledge writes.

```python
grading_output = {
    "trial_id": trial.number,
    "overall_score": 0.85,
    "dimensions": {
        "originality": {"score": 0.9, "reasoning": "..."},
        "brand_alignment": {"score": 0.8, "reasoning": "..."},
        "technical_quality": {"score": 0.85, "reasoning": "..."}
    },
    "novel_finding": "Gradient overlays on dark backgrounds create depth without losing readability",
    "suggest_knowledge_write": True
}
```

### 4. Diversity Guard
- **Goal:** Prevent stagnation and asset overuse
- **Input:** Optuna trial history + asset usage counts from shared state
- **Output:** Deprecation flags, forced exploration triggers
- **Key rules:**
  - If any single asset appears in >30% of last 50 trials → flag for rotation
  - Every 50 cycles, force one fully random trial (inject `RandomSampler` for one trial)
  - Track which combos have been tested vs total possible space

### 5. Orchestrator
- **Goal:** Flow control, context assembly, protocol enforcement
- **Reads:** Entire shared state
- **Manages:** Agent sequencing, tool loading, context filtering
- **Key rule:** This agent decides what each other agent sees. It is the context bottleneck by design.

### 6. Consolidator (NEW)
- **Goal:** Knowledge compaction and tier management
- **Runs:** Every N cycles (configurable, default 50)
- **Process:**
  1. Pull all knowledge entries in the same topic cluster
  2. Cross-reference with Optuna statistics
  3. Merge overlapping entries into higher-tier summaries
  4. Promote observations with statistical backing to patterns
  5. Promote patterns validated across 200+ trials to rules
  6. Archive stale entries (not validated in 200 cycles)
- **Output:** Changelog of promotions, merges, and archives

---

## Confidence & Contradiction

Every knowledge entry tracks:

```python
@dataclass
class KnowledgeEntry:
    id: str
    content: str
    tier: str  # "rule" | "pattern" | "observation"
    confidence: float  # 0.0 - 1.0
    created_cycle: int
    last_validated_cycle: int
    validation_count: int
    contradicted_by: list[str]  # entry IDs
    optuna_evidence: dict  # statistical backing
    topic_cluster: str
```

### Contradiction Resolution
When grader writes a finding that contradicts an existing entry:
1. Don't overwrite — flag both entries
2. On next consolidation pass, resolve:
   - If new finding has stronger Optuna evidence → narrow or replace old entry
   - If old entry has more validations → keep old, archive new as anomaly
   - If ambiguous → create a conditional rule ("X works, but only when Y")

---

## I/O Permissions

| Action | Required Trigger | Protocol |
|--------|-----------------|----------|
| File Writing | Successful grader evaluation OR director final decision | Format per knowledge base template |
| External Data Read | State insufficient OR knowledge gap identified | Must specify why + which memory key to update |
| State Modification | Any counter/status/hypothesis change | Atomic: Read → Calculate → Write |
| Knowledge Write | Novel finding OR grader score ≥ threshold | Include tier, confidence, topic cluster |
| Optuna Trial | Director proposes new hypothesis | Via study.ask() / study.tell() flow |

---

## Evolution Loop

```
Cycle N:
  1. Orchestrator assembles context for Director
  2. Director suggests params via Optuna
  3. Orchestrator assembles context for Creator
  4. Creator generates output
  5. Orchestrator assembles context for Grader
  6. Grader scores + reports to Optuna
  7. If novel finding → knowledge write (observation tier)
  8. Diversity Guard checks asset health
  9. Every 50 cycles → Consolidator runs compaction
  10. System evolves: better combos found, knowledge refined
```

The system learns because:
- **Optuna** steers toward statistically better combos
- **Knowledge base** captures qualitative insights that statistics can't express
- **Consolidation** prevents knowledge rot and keeps the base manageable
- **Diversity guard** prevents local optima traps
