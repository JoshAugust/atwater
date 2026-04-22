# Model Selection Guide

Which models to use for each agent role, and how to configure them.

---

## Per-Agent Role Recommendations

| Agent | Primary Model | Quant | Thinking Mode | Temperature | Why |
|-------|--------------|-------|---------------|-------------|-----|
| **Director** | Qwen3-8B | Q8_0 | ON | 0.1 | Needs reasoning to interpret Optuna stats and knowledge base rules. Low temp = deterministic decisions. |
| **Creator** | Qwen3-8B | Q5_K_M | OFF | 0.7 | Creative generation benefits from higher temperature. No thinking needed — just execute the hypothesis. |
| **Grader** | Qwen3-4B | Q8_0 | OFF | 0.0 | Structured scoring must be consistent and fast. Zero temp = deterministic. Smaller model is fine for evaluation. |
| **Diversity Guard** | Qwen3-4B | Q4_K_M | OFF | 0.8 | Simple rule-checking. Higher temp to occasionally flag edge cases. Smallest viable model. |
| **Orchestrator** | Qwen3-8B | Q8_0 | ON | 0.3 | Needs reasoning for context assembly decisions. Low-medium temp for reliability. |
| **Consolidator** | Qwen3-8B | Q8_0 | ON | 0.3 | Knowledge synthesis requires reasoning. Moderate temp for balanced consolidation. |

### Configuration

In `config/settings.json`:
```json
{
  "agent_configs": {
    "director": {
      "model_name": "qwen3-8b-q8_0",
      "thinking_mode": true,
      "temperature": 0.1
    },
    "grader": {
      "model_name": "qwen3-4b-q8_0",
      "thinking_mode": false,
      "temperature": 0.0
    },
    "creator": {
      "model_name": "qwen3-8b-q5_k_m",
      "thinking_mode": false,
      "temperature": 0.7
    }
  }
}
```

---

## Model Comparison: Qwen3-8B vs Gemma 4 vs Llama 4

Based on research findings for Atwater's use case (structured output, reasoning, creative generation).

### Qwen3-8B (Recommended)

| Strength | Detail |
|----------|--------|
| **Thinking/non-thinking mode** | Toggle `enable_thinking` per call. Reasoning ON for Director/Orchestrator, OFF for fast Grader/Creator calls. This is a unique capability. |
| **Structured output** | Excellent JSON compliance with `response_format`. Best-in-class for the 8B tier. |
| **Multilingual** | Strong if your brand content spans languages |
| **Tool calling** | Native support; reliable function/tool use patterns |

| Weakness | Detail |
|----------|--------|
| Context window | 32K default (sufficient for Atwater's ~2K prompt budget) |
| Creative writing | Slightly more formulaic than Llama 4 at higher temperatures |

### Gemma 4 (12B)

| Strength | Detail |
|----------|--------|
| **Instruction following** | Extremely precise prompt adherence |
| **Safety alignment** | Well-calibrated refusals (won't hallucinate harmful content) |
| **Reasoning** | Strong on multi-step tasks |

| Weakness | Detail |
|----------|--------|
| **Size** | 12B — larger VRAM footprint, slower inference |
| **Structured output** | Good but slightly behind Qwen3 for strict JSON schema compliance |
| **No thinking toggle** | Always one mode; no per-call control |

### Llama 4 (Scout 17B / Maverick 8B)

| Strength | Detail |
|----------|--------|
| **Creative generation** | Best creative quality at the 8B tier |
| **Long context** | 128K context window (overkill for Atwater but nice to have) |
| **Community** | Largest ecosystem, most fine-tuning datasets available |

| Weakness | Detail |
|----------|--------|
| **Structured output** | Inconsistent JSON compliance without grammar enforcement |
| **Tool calling** | Less reliable than Qwen3 for complex tool schemas |
| **Scout size** | 17B active params in MoE — needs more VRAM than the name suggests |

### Verdict

**Qwen3-8B is the default recommendation** because:
1. The thinking mode toggle is architecturally valuable for Atwater (reasoning ON/OFF per agent)
2. Best structured output reliability in the 8B class
3. Good balance of size/quality/speed

Consider **Gemma 4** if you need stronger safety alignment. Consider **Llama 4 Maverick** if creative quality is your top priority and you can tolerate more JSON parse failures.

---

## Quantization Impact on Structured Output

Quantization reduces model size and speeds up inference, but can degrade structured output reliability.

| Quantization | Size (8B model) | Structured Output Reliability | Recommended For |
|-------------|-----------------|------------------------------|----------------|
| **Q8_0** | ~8.5 GB | 95-98% | Grader, Director (accuracy matters) |
| **Q6_K** | ~6.5 GB | 93-96% | Good general-purpose compromise |
| **Q5_K_M** | ~5.5 GB | 90-94% | Creator (creative quality > parse reliability) |
| **Q4_K_M** | ~4.5 GB | 85-90% | Diversity Guard (simple outputs, speed matters) |
| **Q3_K_M** | ~3.5 GB | 70-80% | Not recommended for any Atwater role |
| **Q2_K** | ~3.0 GB | 50-65% | Never use for structured output |

**Rule of thumb:** Use Q8_0 for any agent that produces structured JSON. Drop to Q5_K_M or Q4_K_M only for agents where occasional parse failures are acceptable (Creator with retry logic).

### VRAM planning

For Apple Silicon (unified memory):
```
1 × Qwen3-8B Q8_0  = ~8.5 GB
1 × Qwen3-4B Q8_0  = ~4.5 GB
Overhead             = ~2 GB
Total                ≈ 15 GB  (fits in 16GB, tight)
                     ≈ 15 GB  (comfortable in 32GB)
```

For parallel inference (4 concurrent requests):
```
1 × Qwen3-8B Q8_0 with 4 slots = ~8.5 GB + ~4 GB KV cache = ~12.5 GB
```

---

## Thinking Mode: When to Enable

Qwen3's thinking mode (`enable_thinking=True`) adds a chain-of-thought reasoning step before generating the response. The thinking tokens are separate from the output.

### Enable thinking for:

| Agent | Why |
|-------|-----|
| **Director** | Needs to reason about Optuna stats, knowledge rules, and hypothesis selection |
| **Orchestrator** | Context assembly decisions benefit from step-by-step reasoning |
| **Consolidator** | Knowledge synthesis requires comparing and merging multiple entries |

### Disable thinking for:

| Agent | Why |
|-------|-----|
| **Grader** | Scoring should be fast and deterministic. Thinking adds latency. |
| **Creator** | Creative generation is about fluency, not deliberation. Thinking makes output more formulaic. |
| **Diversity Guard** | Simple rule checks. Thinking is wasted computation. |

### How to toggle

Via LM Studio's OpenAI-compatible API:
```python
response = client.chat.completions.create(
    model="qwen3-8b",
    messages=[...],
    extra_body={"enable_thinking": True},  # or False
)
```

If `enable_thinking` doesn't pass through the API, use prompt-level control:
```
Think step by step before responding. Show your reasoning in <think>...</think> tags, 
then provide the final answer in JSON.
```

---

## Speculative Decoding Setup

Speculative decoding uses a small draft model to predict tokens, then the main model verifies in parallel. This gives 2-2.5× speedup for structured output.

### Setup in LM Studio

1. Load the draft model: Qwen3-0.5B (Q8_0, ~500MB)
2. Load the target model: Qwen3-8B (Q8_0, ~8.5GB)
3. In LM Studio settings, enable speculative decoding and point to the draft model

### When it helps

- **Grader** (structured JSON output): High acceptance rate because grammar constrains both models to the same tokens → 2-2.5× speedup
- **Any agent with response_format**: The JSON schema constrains the output space, making draft predictions more accurate

### When it doesn't help

- **Creator with high temperature**: Speculative tokens diverge frequently → many rejections → minimal speedup
- **Thinking mode**: The thinking step is free-form text with low predictability → marginal benefit

### VRAM cost

Draft model (Qwen3-0.5B Q8_0) adds ~500MB — negligible. The speedup is essentially free if you have the VRAM headroom.
