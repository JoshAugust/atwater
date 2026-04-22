# Agent Development Guide

How to build custom agents for the Atwater architecture.

---

## Agent Base Class

Every agent inherits from `src.agents.base.AgentBase`:

```python
class AgentBase(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...          # Human-readable name
    
    @property
    @abstractmethod
    def role(self) -> str: ...          # Machine identifier for state scoping
    
    @property
    @abstractmethod
    def readable_state_keys(self) -> list[str]: ...   # What state this agent can read
    
    @property
    @abstractmethod
    def writable_state_keys(self) -> list[str]: ...   # What state this agent can write
    
    @abstractmethod
    def execute(self, context: AgentContext) -> AgentResult: ...  # The core logic
```

The base class also provides:
- `validate_state_writes(updates)` — raises `ValueError` if writing to forbidden keys
- `validate_knowledge_write(entry)` — checks required fields exist

---

## AgentContext — What the Agent Sees

The orchestrator builds this before each agent turn:

```python
@dataclass
class AgentContext:
    working_memory: dict[str, Any]          # Tier 1: ephemeral per-turn data
    scoped_state: dict[str, Any]            # Tier 2: role-filtered shared state
    knowledge_entries: list[dict[str, Any]] # Tier 3: top-K relevant knowledge
    optuna_context: dict[str, Any] | None   # Study state: best params, trial count
    tools: list[dict[str, Any]]             # Available tool schemas for this turn
```

**Agents never access the raw database.** Everything comes pre-filtered through `AgentContext`. This is enforced at the architecture level — the `ContextAssembler` filters by role before the agent ever sees data.

### What each field contains

- **working_memory**: Scratch space for the current turn. Cleared between turns. Use for intermediate calculations.
- **scoped_state**: Only keys listed in your `readable_state_keys`. Example: the Director sees `current_hypothesis` and `historical_success_rates` but NOT `output_path` or `score`.
- **knowledge_entries**: Semantic search results from the knowledge base, returned as dicts:
  ```python
  {"tier": "rule", "confidence": 0.95, "content": "Sans-serif headlines outperform serif..."}
  ```
- **optuna_context**: Statistical grounding:
  ```python
  {"total_trials": 150, "completed_trials": 142, "best_params": {...}, "best_score": 0.91}
  ```
- **tools**: JSON schemas for tools available this turn. The tool loader selects the relevant group based on the task description.

---

## AgentResult — What the Agent Returns

```python
@dataclass
class AgentResult:
    output: Any                              # Primary output (usually a dict)
    state_updates: dict[str, Any]            # Keys to write to shared state
    knowledge_writes: list[dict[str, Any]]   # Knowledge entries to persist
    score: float | None                      # Only Grader populates this
```

**Rules:**
- `state_updates` keys must be in your `writable_state_keys` — validated at runtime
- `knowledge_writes` entries must have: `content`, `tier`, `confidence`, `topic_cluster`
- `score` is only set by the Grader. All other agents return `None`.
- `output` is the primary data. For most agents, it's a dict matching their JSON schema.

---

## How Context Scoping Works

The architecture spec defines which state keys each role can read/write:

| Agent | Reads | Writes |
|-------|-------|--------|
| director | current_hypothesis, historical_success_rates | proposed_hypothesis |
| creator | current_hypothesis, last_successful_layout | output_path, self_critique |
| grader | output_path, grading_rubric | score, structured_analysis |
| diversity_guard | asset_usage_counts, deprecation_threshold | asset_status |
| orchestrator | ALL | workflow_state, next_agent |

This is enforced at two levels:
1. **ContextAssembler** (`src/orchestrator/context_assembler.py`) pre-filters `SharedState` by role before building `AgentContext`
2. **AgentBase.validate_state_writes()** rejects writes to keys not in `writable_state_keys`

If an agent tries to read a key it shouldn't see, it simply won't be in `scoped_state`. If it tries to write a forbidden key, a `ValueError` is raised.

---

## How to Add JSON Schema for Your Agent

Every agent should have a structured output schema so LM Studio can use constrained generation. Define it in `src/schemas/agent_schemas.py`:

```python
MY_AGENT_SCHEMA = {
    "type": "object",
    "properties": {
        "field_name": {"type": "string"},
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["field_name", "score"],
    "additionalProperties": False,
}
```

This schema is passed to LM Studio via `response_format: { type: "json_schema", json_schema: {...} }` to enforce valid JSON output. Without this, small models (4B-8B) produce invalid JSON 15-30% of the time.

Register the schema in `src/schemas/agent_schemas.py`'s `SCHEMAS` dict:

```python
SCHEMAS = {
    ...,
    "my_agent": MY_AGENT_SCHEMA,
}
```

---

## How to Wire into FlowController

The `FlowController` accepts agent runners via its constructor or `register_runner()`:

```python
# At construction time
flow = FlowController(
    shared_state=ss,
    knowledge_base=kb,
    study=study,
    agent_runners={"brand_consultant": my_runner_function},
)

# Or after construction
flow.register_runner("brand_consultant", my_runner_function)
```

An `AgentRunner` is any callable with signature:
```python
def my_runner(ctx: AgentContext) -> AgentResult:
    ...
```

Note: `AgentContext` in the flow controller is actually `OrchestratorContext` (from `context_assembler.py`), not `AgentBase.AgentContext`. They have different fields. The orchestrator context includes `system_prompt`, `tool_schemas`, `optuna_summary`, etc. The agent base context has `working_memory`, `scoped_state`, etc.

To add your agent to the pipeline sequence, modify `FlowController.run_cycle()`. You can either:
1. Add to `PIPELINE_ROLES` tuple for sequential execution
2. Call conditionally (like the Consolidator runs every N cycles)

---

## Example: Building a "Brand Consultant" Agent

Let's build a complete agent from scratch.

### Step 1: Agent class

```python
# src/agents/brand_consultant.py

from __future__ import annotations
from src.agents.base import AgentBase, AgentContext, AgentResult
from typing import Any


class BrandConsultant(AgentBase):
    """
    Evaluates creative output for brand alignment.
    
    Reads the current hypothesis and grader score, cross-references with
    brand rules in the knowledge base, and produces actionable feedback.
    """

    @property
    def name(self) -> str:
        return "BrandConsultant"

    @property
    def role(self) -> str:
        return "brand_consultant"

    @property
    def readable_state_keys(self) -> list[str]:
        return ["current_hypothesis", "output_path", "score", "structured_analysis"]

    @property
    def writable_state_keys(self) -> list[str]:
        return ["brand_feedback", "brand_violations"]

    def execute(self, context: AgentContext) -> AgentResult:
        # READ
        hypothesis = context.scoped_state.get("current_hypothesis", {})
        score = context.scoped_state.get("score")
        brand_rules = [
            e for e in context.knowledge_entries
            if e.get("topic_cluster") == "brand"
        ]

        # DECIDE
        violations = self._check_violations(hypothesis, brand_rules)
        alignment_score = self._compute_alignment(hypothesis, brand_rules)
        suggestions = self._generate_suggestions(violations)

        feedback = {
            "brand_alignment_score": alignment_score,
            "violations": violations,
            "suggestions": suggestions,
            "reviewed_params": list(hypothesis.keys()),
        }

        # WRITE
        state_updates = {
            "brand_feedback": feedback,
            "brand_violations": violations,
        }
        self.validate_state_writes(state_updates)

        knowledge_writes = []
        if violations and score and score > 0.8:
            # High-scoring creative with brand violations = noteworthy
            knowledge_writes.append({
                "content": (
                    f"High score ({score:.2f}) despite brand violations: "
                    f"{', '.join(violations)}. Params: {hypothesis}"
                ),
                "tier": "observation",
                "confidence": 0.6,
                "topic_cluster": "brand",
            })

        return AgentResult(
            output=feedback,
            state_updates=state_updates,
            knowledge_writes=knowledge_writes,
            score=None,
        )

    def _check_violations(
        self, hypothesis: dict, rules: list[dict]
    ) -> list[str]:
        # Your brand-specific logic here
        return []

    def _compute_alignment(
        self, hypothesis: dict, rules: list[dict]
    ) -> float:
        # Score from 0-1 based on rule matching
        return 0.8

    def _generate_suggestions(self, violations: list[str]) -> list[str]:
        return [f"Fix: {v}" for v in violations]
```

### Step 2: JSON schema

```python
# In src/schemas/agent_schemas.py

BRAND_CONSULTANT_SCHEMA = {
    "type": "object",
    "properties": {
        "brand_alignment_score": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "violations": {
            "type": "array",
            "items": {"type": "string"},
        },
        "suggestions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "reviewed_params": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["brand_alignment_score", "violations", "suggestions"],
    "additionalProperties": False,
}
```

### Step 3: Context scoping

Add to `ROLE_READ_KEYS` in `context_assembler.py`:

```python
ROLE_READ_KEYS["brand_consultant"] = [
    "current_hypothesis", "output_path", "score", "structured_analysis"
]
```

### Step 4: Stub runner

```python
# In flow_controller.py

def _stub_brand_consultant(ctx):
    return AgentResult(
        role="brand_consultant",
        output={
            "brand_alignment_score": 0.85,
            "violations": [],
            "suggestions": [],
            "reviewed_params": [],
        },
        raw_text="[stub]",
        knowledge_write_requested=False,
        cycle_number=ctx.cycle_number,
        success=True,
    )
```

### Step 5: Wire in

In `FlowController.run_cycle()`, add after the Grader step:

```python
# STEP 3.5: BrandConsultant — check brand alignment
bc_result = self._run_agent(
    role="brand_consultant",
    task_description=f"Cycle {cycle_number}: Review brand alignment...",
    cycle_number=cycle_number,
    errors=errors,
)
```

### Step 6: Test

```python
# tests/test_brand_consultant.py

def test_brand_consultant_scoping():
    agent = BrandConsultant()
    assert "output_path" in agent.readable_state_keys
    assert "proposed_hypothesis" not in agent.writable_state_keys

def test_brand_consultant_validates_writes():
    agent = BrandConsultant()
    with pytest.raises(ValueError):
        agent.validate_state_writes({"forbidden_key": "value"})
```

---

## Tips

- **Keep agents stateless** — all state flows through `AgentContext` and `AgentResult`. No instance variables that persist between turns.
- **Fail gracefully** — if a required state key is missing, return a degraded result rather than crashing. The FlowController catches exceptions but silent degradation is better.
- **Be specific in knowledge writes** — include the exact parameters and scores. Vague observations like "this worked well" are useless for the Consolidator.
- **Respect the token budget** — your agent's output schema should be compact. The ContextAssembler trims knowledge entries and tool schemas to fit the budget, but output parsing failures waste entire cycles.
