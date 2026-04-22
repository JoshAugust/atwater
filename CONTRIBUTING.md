# Contributing to Atwater

---

## Adding a New Agent Role

Every agent follows the READ → DECIDE → WRITE protocol. Here's how to add one.

### 1. Define the agent class

Create `src/agents/my_agent.py`:

```python
from src.agents.base import AgentBase, AgentContext, AgentResult


class BrandConsultant(AgentBase):
    @property
    def name(self) -> str:
        return "BrandConsultant"

    @property
    def role(self) -> str:
        return "brand_consultant"

    @property
    def readable_state_keys(self) -> list[str]:
        return ["current_hypothesis", "output_path", "score"]

    @property
    def writable_state_keys(self) -> list[str]:
        return ["brand_feedback"]

    def execute(self, context: AgentContext) -> AgentResult:
        # READ: pull what you need from context
        hypothesis = context.scoped_state.get("current_hypothesis", {})
        knowledge = context.knowledge_entries

        # DECIDE: your agent logic here
        feedback = self._evaluate_brand_alignment(hypothesis, knowledge)

        # WRITE: return structured result
        return AgentResult(
            output=feedback,
            state_updates={"brand_feedback": feedback},
            knowledge_writes=[],
            score=None,  # only Grader sets score
        )
```

### 2. Add a JSON schema

Create the schema in `src/schemas/agent_schemas.py`:

```python
BRAND_CONSULTANT_SCHEMA = {
    "type": "object",
    "properties": {
        "brand_alignment_score": {"type": "number", "minimum": 0, "maximum": 1},
        "issues": {"type": "array", "items": {"type": "string"}},
        "suggestions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["brand_alignment_score", "issues", "suggestions"],
}
```

### 3. Add role scoping

In `src/orchestrator/context_assembler.py`, add your role to `ROLE_READ_KEYS`:

```python
ROLE_READ_KEYS["brand_consultant"] = ["current_hypothesis", "output_path", "score"]
```

And add agent instructions in `_AGENT_INSTRUCTIONS`:

```python
_AGENT_INSTRUCTIONS["brand_consultant"] = """
    ROLE: Brand Consultant
    Goal: Evaluate brand alignment of the current creative output.
    ...
"""
```

### 4. Create a stub runner

In `src/orchestrator/flow_controller.py`, add a stub:

```python
def _stub_brand_consultant(ctx: AgentContext) -> AgentResult:
    return AgentResult(
        role="brand_consultant",
        output={"brand_alignment_score": 0.8, "issues": [], "suggestions": []},
        raw_text="[stub]",
        knowledge_write_requested=False,
        cycle_number=ctx.cycle_number,
        success=True,
    )
```

### 5. Wire into FlowController

Either add to the pipeline sequence or call conditionally in `run_cycle()`.

### 6. Write tests

Add tests in `tests/test_brand_consultant.py` covering:
- Context scoping (agent only sees allowed state keys)
- Output schema validation
- State write permission enforcement
- Edge cases (empty context, missing state)

---

## Adding New Search Dimensions to Optuna

### 1. Update the SearchSpace

In `src/optimization/trial_adapter.py`, modify `DEFAULT_SEARCH_SPACE`:

```python
DEFAULT_SEARCH_SPACE = SearchSpace(
    categorical={
        ...,
        "color_scheme": ["warm", "cool", "monochrome", "complementary"],
    },
    continuous={
        ...,
        "saturation": (0.0, 1.0),
    },
    integer={
        ...,
        "grid_columns": (1, 6),
    },
)
```

### 2. No study migration needed

Optuna handles new dimensions gracefully — existing trials just won't have the new params. AutoSampler explores the new dimensions on the next trial.

### 3. Update the Creator agent

The Creator needs to know how to use the new parameter. Update its prompt template and execution logic.

---

## Adding New Tool Groups to the Orchestrator

### 1. Define tool schemas

Create a new tool group in `src/orchestrator/tool_loader.py`:

```python
TOOL_GROUPS["image_generation"] = [
    {
        "name": "generate_image",
        "description": "Generate an image from a text prompt",
        "parameters": { ... },
    },
]
```

### 2. Add semantic matching keywords

The tool loader uses keyword matching to select relevant tool groups. Add entries that describe when your tool group is useful.

---

## Adding New Evaluation Gates

### 1. Create the gate module

Create `src/evaluation/my_gate.py` implementing a function that takes creative output and returns a pass/fail result with a score:

```python
def check(output: dict) -> tuple[bool, float, str]:
    """Returns (passed, score, reasoning)."""
    ...
```

### 2. Wire into the cascade

In `src/evaluation/cascade.py`, add your gate to the evaluation chain. Gates run in order: fast → medium → slow. Insert yours at the appropriate latency tier.

---

## Adding New Learning Modules

Learning modules hook into the post-cycle phase. Options:

1. **Reflexion notes** — Add to `src/learning/reflexion.py` to capture new types of reflections
2. **Strategy selector** — Add new strategy arms to `src/learning/strategy_selector.py`
3. **Custom detectors** — Create a new detector in `src/learning/` and call it from `FlowController.run_cycle()`

---

## Code Style and Conventions

### Python
- Python 3.11+ (use `from __future__ import annotations` in every file)
- Type hints everywhere — `mypy --strict` should pass
- Docstrings: Google style, with Args/Returns/Raises sections
- Module-level `logger = logging.getLogger(__name__)`
- Dataclasses over plain dicts for structured data
- No business logic in `__init__.py` — only re-exports

### Naming
- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private: prefix with `_`

### Imports
- Standard library first, third-party second, project third
- Avoid `from X import *`
- Use lazy imports for heavy dependencies (sentence_transformers, torch)

### Architecture Rules
- **Agents never call LLMs directly** — they return prompts; the orchestrator executes
- **FlowController owns sequencing** — agents don't know about each other
- **Knowledge writes go through FlowController** — agents request, controller executes
- **State scoping is enforced** — agents only see their allowed keys

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Fast tests only (skip scale tests)
pytest tests/ -v -m "not slow"

# Scale tests only
pytest tests/ -v -m slow

# Specific test file
pytest tests/test_memory.py -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

---

## PR Guidelines

1. **One concern per PR** — don't mix a new feature with a refactor
2. **Tests required** — no PR without tests for new code
3. **Stubs first** — if adding a new agent, include the stub runner so the system still compiles
4. **Update docs** — if you change the architecture, update ARCHITECTURE.md
5. **Run the full test suite** before submitting
6. **Keep knowledge base backward-compatible** — don't break existing entries
