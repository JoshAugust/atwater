# Context Tips for Small Model Agents (4-8B)

Practical heuristics for building reliable agents on local models.

---

## Lazy Loading Toolsets

The trick is a two-tier prompt structure:

1. **Top level** gets a *tool catalog* (just names + one-line descriptions, no schemas)
2. Model picks which tool group it needs
3. Inject the full schema for **only that group** on the next turn

Think of it like dynamic imports — you don't load the whole toolbox, you load the drawer.

**Sweet spot:** 3-5 tool schemas max per turn before accuracy drops on 4-8B models.

### Implementation

```python
TOOL_CATALOG = {
    "memory": "Read/write to shared state and knowledge base",
    "creative": "Generate content, layouts, compositions",
    "analysis": "Score, evaluate, compare outputs",
    "research": "Search external data, fetch references"
}

TOOL_SCHEMAS = {
    "memory": [...full schemas...],
    "creative": [...full schemas...],
    # etc.
}

def build_prompt(task, selected_group=None):
    if selected_group:
        return base_prompt + TOOL_SCHEMAS[selected_group]
    else:
        return base_prompt + catalog_summary(TOOL_CATALOG)
```

---

## Context Scoping Heuristics

### Recency > Relevance
Small models are bad at fishing important facts out of long context. Put the important stuff at the **end**, right before the instruction.

### Structured Preamble, Conversational Tail
System prompt is rigid/structured, but the last 2-3 turns should read naturally. Small models anchor hard on format cues.

### Summarise Before Reinserting
For tool results, don't feed raw JSON back. Have a template that extracts the 2-3 fields that matter:

```python
def summarise_tool_result(raw_result, relevant_keys):
    """Extract only what matters before putting back in context."""
    return {k: raw_result[k] for k in relevant_keys if k in raw_result}
```

---

## Embedding Models ↔ LLMs

They don't need to be from the same family. What matters:

- **Chunking strategy must match retrieval query style**
- If your LLM generates verbose queries → use an embedding model trained on longer passages (nomic-embed, BGE-large)
- If queries are terse keywords → smaller models like all-MiniLM are fine

### Semantic Tool Selection
Embed your tool descriptions and do semantic tool selection before prompt assembly. This is how you scale past 10-15 tools on a small model:

```python
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Pre-compute tool embeddings
tool_embeddings = {
    name: embedder.encode(desc)
    for name, desc in TOOL_CATALOG.items()
}

def select_tools(query, top_k=3):
    query_emb = embedder.encode(query)
    scores = {
        name: cosine_sim(query_emb, emb)
        for name, emb in tool_embeddings.items()
    }
    return sorted(scores, key=scores.get, reverse=True)[:top_k]
```

---

## Fine Tuning Priorities

### Highest ROI: LoRA on tool-call format
- 200-500 examples of your exact tool schema + correct calls
- Train for 2-3 epochs
- Dramatically more reliable than prompt engineering alone

### What to fine tune
- **Grader engine** — structured JSON output is the perfect LoRA candidate
- **Tool call formatting** — match your exact schema

### What NOT to fine tune
- **Orchestrator** — prompt engineering is sufficient for routing
- **General knowledge** — base instruction tuning is good enough

### DPO with failed runs
Use your actual failed runs as negative examples in Direct Preference Optimization:
- Pair: (correct tool call, failed tool call) for the same input
- This teaches the model to avoid specific failure modes

### Model-specific notes
- Qwen 3 uses `<tool_call>` XML-ish format
- Gemma prefers JSON function calling
- **Match your prompt template to the model's native format** — way fewer parsing failures

---

## Multi-Step Reliability

Already validated: 4-12 steps with loops work on 4-8B models. Key patterns:

- **Context compaction** every 8-10 steps (summarise state, archive history)
- **Structured state** over conversational memory (JSON state object > conversation history)
- **Checkpoint after each step** — if the model loses track, restart from last checkpoint, not from scratch
