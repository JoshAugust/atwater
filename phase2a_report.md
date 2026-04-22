# Phase 2A Report — Integration Surgery

**Status: COMPLETE**
All import chains pass. `python3 src/main.py --cycles 1 --verbose` completes a full cycle successfully.

---

## Import Errors Fixed

### 1. `src/knowledge/__init__.py`
**Problem:** Used `from atwater.src.knowledge.X import ...` — absolute paths with the wrong package prefix. `atwater` is not a registered package on sys.path; the root package is `src`.  
**Fix:** Changed to relative imports (`from .X import ...`):
```python
# Before:
from atwater.src.knowledge.clustering import TopicClusterer
from atwater.src.knowledge.consolidator import ConsolidationEngine
from atwater.src.knowledge.models import (...)

# After:
from .clustering import TopicClusterer
from .consolidator import ConsolidationEngine
from .models import (...)
```

### 2. `src/knowledge/consolidator.py`
**Problem:** Three `from atwater.src.knowledge.X` imports — two at module level, one deferred inside a method.  
**Fix:** Changed all three to relative imports:
- Line 32: `from atwater.src.knowledge.models import ...` → `from .models import ...`
- Line 170 (deferred): `from atwater.src.knowledge.clustering import TopicClusterer` → `from .clustering import TopicClusterer`
- Line 548 (deferred): `from atwater.src.knowledge.models import TIER_RANK` → `from .models import TIER_RANK`

### 3. `src/knowledge/clustering.py`
**Problem:** TYPE_CHECKING block used `from atwater.src.knowledge.models import KnowledgeEntry`.  
**Fix:** `from .models import KnowledgeEntry`
(This was a TYPE_CHECKING-only import, so it didn't cause runtime errors, but fixed for consistency.)

### 4. `src/agents/consolidator_agent.py`
**Problem:** Had a complex sys.path manipulation hack (~20 lines) to add the parent of `atwater/` to sys.path, then a `try/except` importing via `atwater.src.knowledge.*`. The hack was self-defeating since `atwater` was never a top-level package.  
**Fix:** Removed the entire sys.path hack and try/except stub. Replaced with clean direct imports:
```python
from src.knowledge.consolidator import ConsolidationEngine
from src.knowledge.models import KnowledgeEntry, KnowledgeTier
```

### 5. `src/__init__.py` — Missing
**Problem:** `src/` had no `__init__.py`, so it wasn't recognised as a Python package by import machinery.  
**Fix:** Created `src/__init__.py` with a minimal docstring comment.

### 6. `src/main.py` — sys.path not set when running as script
**Problem:** `python3 src/main.py` adds `src/` to sys.path but not the project root, so `from src.memory import ...` failed with `ModuleNotFoundError: No module named 'src'`.  
**Fix:** Added a sys.path bootstrap at the top of `main.py`:
```python
_project_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
```

---

## Files Modified

| File | Change |
|------|--------|
| `src/__init__.py` | **Created** (new) |
| `src/knowledge/__init__.py` | 3 absolute → relative imports |
| `src/knowledge/consolidator.py` | 3 absolute → relative imports |
| `src/knowledge/clustering.py` | TYPE_CHECKING import fixed |
| `src/agents/consolidator_agent.py` | Replaced sys.path hack with clean imports |
| `src/main.py` | Added project-root sys.path bootstrap |
| `pyproject.toml` | **Created** (new) |

---

## Package Structure Created

### `pyproject.toml`
```toml
[project]
name = "atwater"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "optuna>=3.6", "optunahub>=0.4", "sentence-transformers>=2.7",
    "scikit-learn>=1.4", "numpy>=1.26", "pandas>=2.2"
]
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

---

## Dataclass Mutable Default Audit

All dataclasses checked via AST analysis. **No mutable defaults found without `field(default_factory=...)`**. All list/dict fields correctly use `field(default_factory=list)` or `field(default_factory=dict)`.

---

## Interface Conflicts (Design Issues — Not Fixed)

These are architectural mismatches that cannot be resolved without partial rewrites. Documented for Phase 2B.

### A. Duplicate `AgentContext` / `AgentResult` Classes

Two separate class definitions exist with the same names but incompatible field structures:

**`src/agents/base.py`** (used by real agent implementations):
```python
@dataclass class AgentContext:
    working_memory: dict
    scoped_state: dict
    knowledge_entries: list[dict]  # structured dicts
    optuna_context: dict | None
    tools: list[dict]

@dataclass class AgentResult:
    output: Any
    state_updates: dict
    knowledge_writes: list[dict]
    score: float | None
```

**`src/orchestrator/context_assembler.py`** (used by FlowController stubs and ContextAssembler):
```python
@dataclass class AgentContext:
    role: str
    system_prompt: str
    state_snapshot: str       # serialised string, not dict
    knowledge_entries: list[str]  # formatted strings, not dicts
    cycle_number: int
    # ...+ more fields

@dataclass class AgentResult:
    role: str
    output: dict
    raw_text: str
    success: bool
    error: str | None
    cycle_number: int
```

**Why this can't be trivially resolved:** The `FlowController` stubs and `ContextAssembler.assemble_context()` use the orchestrator's version (with `cycle_number`, `state_snapshot`, `success`). The real agents (`DirectorEngine.execute()`, `CreatorAgent.execute()`, etc.) use the base version (with `scoped_state`, `knowledge_entries` as dicts). Merging them without rewriting both the orchestrator and all agents would produce a bloated class with conflicting semantics.

**Recommendation for Phase 2B:** Rename orchestrator's classes to `PromptContext` / `RunnerResult` to make the distinction explicit, or implement an adapter layer.

### B. Duplicate `KnowledgeEntry` Class

Two `KnowledgeEntry` dataclasses exist:

- **`src/memory/knowledge_base.py`**: SQLite-backed, has `embedding: bytes | None` field, used by `KnowledgeBase` for persistent storage.
- **`src/knowledge/models.py`**: Pure dataclass, has `lineage: list[str]` field, used by `ConsolidationEngine` and `TopicClusterer`.

They share most fields but differ in `embedding` (memory) vs `lineage` (knowledge). This is a real split: the memory layer deals with raw storage + embeddings, the knowledge layer with logical relationships.

**Recommendation for Phase 2B:** Decide if `KnowledgeBase` should store and return `src.knowledge.models.KnowledgeEntry` (adding `embedding` as an optional field there), or if the split is intentional and both classes should be explicit about their scope.

### C. `TrialAdapter` API Mismatch

`DirectorEngine` (in `src/agents/director.py`) calls:
```python
adapter = TrialAdapter(trial, self._search_space)  # passes 2 args
params = adapter.suggest_all()                       # method doesn't exist
```

But `TrialAdapter.__init__` only accepts `(self, trial: optuna.Trial)` and has `suggest_params(search_space)` not `suggest_all()`.

This is a runtime bug but doesn't affect imports. It will only surface when `DirectorEngine` is wired as a real runner (not the current stubs).

**Recommendation for Phase 2B:** Either update `TrialAdapter` to accept an optional `search_space` in `__init__` and add `suggest_all()` as an alias for `suggest_params(self._search_space)`, OR update `DirectorEngine` to call `adapter.suggest_params(search_space)`.

---

## `python3 src/main.py --cycles 1 --verbose` Run Summary

**Exit code: 0** — Full cycle completed successfully.

**Non-Python errors observed (expected, not Python bugs):**
- `AutoSampler unavailable (cmaes, scipy, torch missing)` → fell back to TPESampler. Normal, optional deps.
- `No modules.json found for all-MiniLM-L6-v2` + `We couldn't connect to 'https://huggingface.co'` → sentence-transformers model not cached, HuggingFace unreachable from this machine. `ToolLoader` gracefully fell back to the `memory` tool group. **Not a Python error.**
- `ToolLoader.tool_loader` ran 4 times (once per agent role per cycle) — each call creates a new SentenceTransformer instance. This is a performance issue (model should be shared singleton) but not a bug.

**Cycle result:** 1 cycle, score=0.5000 (stub grader), 0 KB writes, 0 diversity alerts.

---

## Verification

```
$ python3 -c "
import sys; sys.path.insert(0, '.')
from src.memory import WorkingMemory, SharedState, KnowledgeBase
from src.optimization import create_study, TrialAdapter, get_importances
from src.knowledge import KnowledgeEntry, ConsolidationEngine, TopicClusterer
from src.agents import DirectorEngine, CreatorAgent, GraderEngine, DiversityGuard
from src.orchestrator import ContextAssembler, FlowController, ToolLoader
from src.llm import LMStudioClient
print('All imports OK')
"
All imports OK
```
