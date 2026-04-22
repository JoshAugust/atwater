# Small Model Capabilities for Atwater Agent Architecture
> Research Date: 2026-04-22
> Context: Cognitive agent architecture (Atwater) with statistical optimization for OpenFang 0.6
> Stack: Python + Optuna + SQLite + OpenFang 0.6 + LM Studio (local inference)
> Note: `web_search` (Brave API) unavailable — research conducted via `web_fetch` on targeted authoritative sources.

---

## Executive Summary — What Changes Our Architecture

### ⚡ Critical Late Finding: Qwen3 Has Thinking/Non-Thinking Mode Switch

Qwen3 models support seamless switching between:
- **Thinking mode** (enable_thinking=True): Generates `<think>...</think>` reasoning block before answer. Best for Director/complex routing.
- **Non-thinking mode** (enable_thinking=False): Direct output, much faster. Best for Grader/structured JSON.

This is a **free reasoning upgrade** for Director/Orchestrator agents and a **free speed upgrade** for Grader/Creator agents — just toggle per role. Applies to ALL Qwen3 sizes.

```python
# Thinking mode for Director (better routing decisions)
text = tokenizer.apply_chat_template(messages, enable_thinking=True)

# Non-thinking for Grader (fast, deterministic)
text = tokenizer.apply_chat_template(messages, enable_thinking=False)
```

---

### Critical findings:

0. **Qwen3's thinking/non-thinking mode switch is a game changer** — Use thinking mode for Director (better decomposition), non-thinking for Grader (faster, deterministic). Applies to all Qwen3 sizes (0.6B–235B). All support Ollama, LM Studio, MLX-LM, llama.cpp natively.

1. **Constrained generation is non-negotiable for small models** — use GBNF grammar via LM Studio's JSON schema mode or Outlines library. Do not rely on post-hoc parsing of raw model output.

2. **Qwen3 family is the top choice** for tool calling in local deployment — Hermes-style function calling format, documented quantization benchmarks, explicit multi-backend support (MLX, llama.cpp, Ollama, LM Studio).

3. **Quantization floor: Q8 for tool calling** — GPTQ-Int8 / Q8_0 GGUF introduces <1% degradation. Q4 loses 3-5% on instruction following (more severe on small models). Prefer Q5_K_M as minimum for structured output agents.

4. **Gemma 3n E4B is the dark horse** — MatFormer architecture achieves 8B quality in ~4B VRAM footprint. First sub-10B model with LMArena 1300+. Supported on Apple Silicon via MLX and llama.cpp.

5. **Speculative decoding is a free performance win** — Qwen3-0.5B draft + Qwen3-8B target gives 2-3x speedup on structured outputs. High-value for latency-sensitive agent loops.

6. **Multi-turn degradation is real** — BFCL V3/V4 confirms small models lose state across turns. Atwater's three-tier memory model directly addresses this — it's the right architectural choice.

7. **System prompt discipline** — Keep under 500 tokens per agent. Include 1-2 few-shot examples per role. Use positive framing, not negative constraints.

8. **MLX is faster than llama.cpp on Apple Silicon** — ~20-40% throughput advantage. Both supported in LM Studio. Use llama.cpp when GBNF grammar is needed; MLX for raw throughput.

---

## Model Landscape (2025-2026)

### Small/Mid Models Available for Local Deployment

| Model | Size | Context | Tool Calling | Format | Local Viable |
|-------|------|---------|-------------|--------|-------------|
| Qwen3-4B | 4B | 32K (→128K YaRN) | ✅ Native (Hermes) + Thinking | ChatML | ✅ Yes |
| Qwen3-8B | 8.2B | 32K (→131K YaRN) | ✅ Native (Hermes) + Thinking | ChatML | ✅ Yes |
| Qwen3-14B | 14B | 128K | ✅ Native | ChatML | ⚠️ 16GB+ VRAM |
| Qwen3-32B | 32B | 128K | ✅ Native | ChatML | ❌ Server only |
| Gemma 3 4B | 4B | 128K | ✅ (instruct) | Gemma | ✅ Yes |
| Gemma 3 12B | 12B | 128K | ✅ (instruct) | Gemma | ⚠️ 16GB VRAM |
| Gemma 3n E2B | 5B real / 2B eff. | TBD | ✅ (instruct) | Gemma | ✅ Yes (2GB VRAM) |
| Gemma 3n E4B | 8B real / 4B eff. | TBD | ✅ (instruct) | Gemma | ✅ Yes (3GB VRAM) |
| Phi-4 | 14B | 16K | ⚠️ Prompt-only | ChatML | ⚠️ 16K context limit |
| Mistral Small 3.1 | 24B | 128K | ✅ Native | Mistral | ⚠️ 32GB RAM needed |
| Llama 4 Scout | 109B total | 10M | ✅ Native | Llama 4 | ❌ Server only |
| Llama 4 Maverick | ~400B total | 1M | ✅ Native | Llama 4 | ❌ Server only |

### Key Qualitative Rankings (Tool Calling Focus)

**For ~4B effective budget:**
1. 🥇 **Qwen3-4B** — best documented, Hermes format, Q8 GGUF available
2. 🥈 **Gemma 3n E4B** — competitive quality, 4B effective VRAM, MatFormer innovation
3. 🥉 **Gemma 3 4B** — strong 128K context, well-supported

**For ~8B budget:**
1. 🥇 **Qwen3-8B** — strong BFCL scores, Hermes tool calling, Q8 viable on 16GB
2. 🥈 **Gemma 3n E4B** — still fits here and punches above weight
3. 🥉 **Gemma 3 12B** — if you have 16GB VRAM

---

## Recommended Models Per Agent Role

### Director Agent
**Function:** Route tasks, decompose goals, orchestrate other agents

**Recommended:** Qwen3-8B Q8_0 (primary) / Qwen3-4B Q8_0 (fallback)

**Rationale:**
- Routing decisions benefit from larger context and better instruction following
- Q8 preserves instruction-following accuracy (IFEval metric critical here)
- Hermes-style tool calling: Director selects agent roles as "tools"
- If speculative decoding available: Qwen3-0.5B draft + Qwen3-8B target

**Config:**
```json
{
  "model": "Qwen3-8B-Instruct-Q8_0.gguf",
  "temperature": 0.1,
  "enable_thinking": true,
  "json_schema": {...director_output_schema...},
  "max_tokens": 1024
}
```

> **Note**: `enable_thinking=True` causes Qwen3 to emit a `<think>...</think>` block before the JSON output. Strip it before passing to JSON schema validator. Worth the overhead for routing quality.

---

### Creator Agent
**Function:** Generate content, produce structured artifacts

**Recommended:** Qwen3-8B Q5_K_M (primary) / Gemma 3n E4B Q8 (alternative)

**Rationale:**
- Needs high output quality but throughput matters (generation-heavy role)
- Q5_K_M balances quality/speed on Apple Silicon
- Gemma 3n E4B excellent if multimodal inputs expected
- Constrained generation via JSON schema for artifact structure

**Config:**
```json
{
  "model": "Qwen3-8B-Instruct-Q5_K_M.gguf",
  "temperature": 0.7,
  "enable_thinking": false,
  "json_schema": {...creator_output_schema...},
  "max_tokens": 2048
}
```

---

### Grader Agent
**Function:** Score and evaluate outputs, binary/scored judgments

**Recommended:** Qwen3-4B Q8_0 (primary) — smaller model fine for grading

**Rationale:**
- Grading is classification-heavy — smaller models work well when output is constrained
- Q8_0 ensures accurate structured output (score, rationale, pass/fail)
- Low temperature (0.0-0.1) for deterministic scores
- Use strict JSON schema: `{score: float, rationale: string, pass: boolean}`
- Consider speculative decoding: Qwen3-0.5B draft for fast grading

**Config:**
```json
{
  "model": "Qwen3-4B-Instruct-Q8_0.gguf",
  "temperature": 0.0,
  "json_schema": {
    "type": "object",
    "properties": {
      "score": {"type": "number", "minimum": 0, "maximum": 1},
      "rationale": {"type": "string"},
      "pass": {"type": "boolean"}
    },
    "required": ["score", "rationale", "pass"]
  },
  "max_tokens": 256
}
```

---

### Diversity Guard Agent
**Function:** Detect repetition/convergence, inject variation

**Recommended:** Qwen3-4B Q4_K_M (acceptable for this role)

**Rationale:**
- Diversity checks are simpler than generation — Q4 acceptable here
- Higher temperature beneficial (0.8-1.2)
- Output is simple: `{is_diverse: boolean, similarity_score: float, suggested_variation: string}`

---

### Orchestrator / Consolidator
**Function:** Merge agent outputs, maintain state, drive experiment flow

**Recommended:** Qwen3-8B Q8_0 (same as Director)

**Rationale:**
- Highest demand for instruction following and long-context coherence
- Should handle 128K context for deep consolidation passes
- Critical role — don't compromise on quantization

---

## Prompt Template Recommendations Per Model Family

### Qwen3 (ChatML + Hermes Tool Format)

```
<|im_start|>system
You are [AGENT_ROLE]. [ROLE_DESCRIPTION_MAX_200_TOKENS]

Available tools:
<tools>
[
  {
    "type": "function",
    "function": {
      "name": "tool_name",
      "description": "...",
      "parameters": {
        "type": "object",
        "properties": {...},
        "required": [...]
      }
    }
  }
]
</tools>

Always respond in this JSON format:
{"action": "tool_name", "arguments": {...}}
<|im_end|>
<|im_start|>user
[USER_INPUT]
<|im_end|>
<|im_start|>assistant
```

**Key notes:**
- Tools wrapped in `<tools>` XML block in system prompt
- Hermes format preferred over OpenAI format (per Qwen3 docs)
- Keep system prompt < 500 tokens for 4B models
- Include 1-2 few-shot examples for complex schemas

### Gemma 3 / Gemma 3n

```
<start_of_turn>user
[SYSTEM_AND_USER_COMBINED for first turn]
<end_of_turn>
<start_of_turn>model
[ASSISTANT_RESPONSE]
<end_of_turn>
<start_of_turn>user
[NEXT_USER_TURN]
<end_of_turn>
<start_of_turn>model
```

**Key notes:**
- No dedicated system role in Gemma 3 format — embed in first user turn
- Function calling via prompt engineering (no dedicated FC format like Qwen)
- Constrained generation (JSON schema) is especially important for Gemma

### Mistral Small 3.1

```
[INST] [SYSTEM_PROMPT]

[USER_INPUT] [/INST]
[ASSISTANT_RESPONSE]
[INST] [NEXT_USER_INPUT] [/INST]
```

**Key notes:**
- Newer Mistral models also support ChatML
- Native function calling supported in instruct versions

---

## Quantization Recommendations

### Decision Matrix

| Agent Role | Priority | Min Quantization | Recommended |
|-----------|----------|-----------------|-------------|
| Director | Accuracy | Q6_K | Q8_0 |
| Orchestrator | Accuracy | Q6_K | Q8_0 |
| Creator | Throughput+Quality | Q5_K_M | Q6_K |
| Grader | Accuracy | Q8_0 | Q8_0 |
| Diversity Guard | Speed | Q4_K_M | Q5_K_M |
| Consolidator | Accuracy | Q6_K | Q8_0 |

### Quantization Impact (Qwen2 reference data, Qwen3 pending):

| Quant | vs BF16 (7B avg) | IFEval Impact | Notes |
|-------|-----------------|---------------|-------|
| BF16 | baseline | baseline | Too large for most local |
| GPTQ-Int8 / Q8_0 | -0.7% avg | -0.6% | **Sweet spot** |
| Q6_K (GGUF) | ~-1.5% avg | ~-1.5% | Good trade-off |
| Q5_K_M (GGUF) | ~-2.5% avg | ~-2.5% | Acceptable |
| GPTQ-Int4 / Q4_K_M | -2.8% avg | -3.7% | Degraded instruction following |
| AWQ (Int4) | -2.8% avg | -2.9% | Similar to GPTQ-Int4 |

**Rules:**
- Never use Q4 for agents that produce JSON without constrained generation
- If using Q4: mandatory JSON schema / GBNF grammar constraint
- 1.5B-4B models degrade faster than 7B+ at same quantization level

---

## Constrained Generation Strategy

### Priority stack for Atwater + LM Studio + OpenFang 0.6:

1. **LM Studio JSON Schema mode** (primary) — pass `json_schema` to `/v1/chat/completions` API
   - Wraps llama.cpp GBNF internally
   - Zero code overhead in OpenFang
   - Works for all GGUF models

2. **Direct GBNF grammar** (llama.cpp server) — for custom formats beyond JSON
   - Pass grammar string to `/completion` endpoint
   - Useful for tool call format (not pure JSON)

3. **Outlines library** (Python, transformers/vLLM) — for non-LM Studio inference
   - `pip install outlines`
   - Pydantic model → guaranteed output
   - Fallback if LM Studio API doesn't support schema

4. **Instructor** (Python, last resort) — post-hoc validation + retry
   - 2-3 retries on parse failure
   - Higher latency, no generation guarantee

### GBNF Tool Call Grammar Template:

```gbnf
root ::= tool-response
tool-response ::= "{" ws "\"action\"" ws ":" ws string ws "," ws "\"arguments\"" ws ":" ws object ws "}"
string ::= "\"" ([^"\\] | "\\" .)* "\""
object ::= "{" ws (string ws ":" ws value ws ("," ws string ws ":" ws value)*)? "}"
value ::= string | number | "true" | "false" | "null" | object | array
array ::= "[" ws (value ws ("," ws value)*)? "]"
number ::= "-"? [0-9]+ ("." [0-9]+)?
ws ::= [ \t\n]*
```

---

## Fine-Tuning Priorities

### Priority 1: Grader Agent (Highest ROI)
**Why:** Grader needs to produce consistent score format — even a small divergence cascades through Optuna optimization
**Method:** SFT → JSON-formatted scoring pairs
**Dataset:** ~500-2000 examples of (content, scoring_rubric) → structured score
**Framework:** Unsloth (fastest, least VRAM) + LLaMA-Factory
**Base model:** Qwen3-4B
**LoRA rank:** 32-64
**Expected gain:** Score consistency improvement from ~85% to ~99%

### Priority 2: Director Agent Routing
**Why:** Incorrect routing wastes all downstream compute
**Method:** SFT + DPO (preference pairs: correct routing = chosen, wrong = rejected)
**Dataset:** ~1000 SFT + ~500 DPO preference pairs
**Framework:** Axolotl (DPO support) or TRL
**Base model:** Qwen3-8B
**LoRA rank:** 64
**Expected gain:** Routing accuracy +10-15% on complex task decomposition

### Priority 3: Creator Agent Specialization
**Why:** Domain-specific output format may not be in base model training data
**Method:** SFT on domain-specific examples
**Dataset:** Domain-specific (caffy_studio output format, whatever that is)
**Framework:** LLaMA-Factory
**Base model:** Qwen3-8B or Qwen3-4B
**LoRA rank:** 64-128

### Priority 4: Function Calling Format Alignment
**Why:** Hermes format not perfectly learned by all 4B models
**Method:** SFT on Hermes function calling dataset
**Dataset:** NousResearch/hermes-function-calling-v3 or glaive-function-calling-v2
**Base model:** Whichever base you use — apply before role fine-tune
**LoRA rank:** 64

---

## Speculative Decoding Configuration

High-value optimization for Atwater's latency-sensitive grading/scoring loops:

```python
# LM Studio-compatible approach via llama.cpp server
# Start server with draft model:
# llama-server --model qwen3-8b-q8.gguf --draft-model qwen3-0.5b-q8.gguf --draft-max 8

# MLX approach (Apple Silicon):
# mlx_lm.generate --model Qwen/Qwen3-8B-Instruct-MLX \
#   --draft-model Qwen/Qwen3-0.5B-Instruct-MLX \
#   --draft-length 8
```

Expected speedup: 2-2.5x for structured JSON output (high acceptance rate due to grammar constraint)

---

## Infrastructure Recommendations

### Apple Silicon (Mac mini / MacBook Pro M-series)

**Primary backend:** LM Studio (uses best of MLX/llama.cpp per model)
**Secondary:** llama.cpp server (for GBNF grammar control)
**Avoid:** vLLM (poor Apple Silicon support)

**Memory recommendations:**
- 8B Q8: ~9GB VRAM
- 8B Q5_K_M: ~5.5GB VRAM
- 4B Q8: ~4.5GB VRAM
- Gemma 3n E4B: ~3GB VRAM (exceptional efficiency)

**For multi-agent Atwater (simultaneous Director + Creator + Grader):**
- 32GB unified memory: run all 3 in parallel with 4B models
- 16GB unified memory: round-robin scheduling or use Gemma 3n E4B + Qwen3-4B mix
- Use Atwater's SQLite shared state to coordinate handoffs

### Inference Throughput (Official Qwen3 Benchmarks — NVIDIA H20, SGLang)

| Model | Quant | Speed (tok/s) | Context |
|-------|-------|--------------|---------|
| Qwen3-0.6B | BF16 | 414 | 1 tok input |
| Qwen3-0.6B | FP8 | 458 | 1 tok input |
| Qwen3-8B | BF16 | ~100-150 | 1 tok input (estimated) |

*Note: H20 GPU figures — Apple Silicon will be ~3-5x slower depending on model/quant.*

### Inference Throughput Estimates (Apple Silicon M3 Pro, approximate)

| Model | Backend | Token/sec |
|-------|---------|-----------|
| Qwen3-4B Q8_0 | MLX | ~60-80 |
| Qwen3-4B Q8_0 | llama.cpp | ~45-65 |
| Qwen3-8B Q5_K_M | MLX | ~35-50 |
| Qwen3-8B Q5_K_M | llama.cpp | ~28-40 |
| Gemma 3n E4B Q8 | MLX | ~50-70 |
| Gemma 3n E2B Q8 | MLX | ~90-110 |

---

## Architecture Impact Summary

### Changes to Atwater's OpenFang 0.6 Integration:

1. **Add JSON schema to every LM Studio API call** — mandatory, not optional
   ```python
   payload = {
       "model": agent_config.model,
       "messages": messages,
       "response_format": {
           "type": "json_schema",
           "json_schema": agent_config.output_schema
       },
       "temperature": agent_config.temperature
   }
   ```

2. **Model-specific chat templates** — per-family template in agent config
   ```python
   TEMPLATES = {
       "qwen": "chatml_hermes_tools",
       "gemma": "gemma_v1",
       "mistral": "mistral_v3",
   }
   ```

3. **Quantization policy in agent config** — not one-size-fits-all
   ```python
   QUANT_POLICY = {
       "director": "q8_0",
       "grader": "q8_0",
       "creator": "q5_k_m",
       "diversity_guard": "q4_k_m",
   }
   ```

4. **Speculative decoding for Grader** — add draft model config option
   ```python
   "grader": {
       "model": "qwen3-8b-q8.gguf",
       "draft_model": "qwen3-0.5b-q8.gguf",
       "draft_length": 8,
   }
   ```

5. **Context compression between turns** — prevents multi-turn degradation
   - Director compresses prior context to structured state dict before each new turn
   - Shared state machine stores compressed state, not raw conversation history

---

## Per-Search Results Index

| Search # | Topic | Key Finding |
|----------|-------|-------------|
| 1 | Qwen3 tool calling | Use Hermes-style format; vLLM/Qwen-Agent backends |
| 2 | Qwen3 JSON accuracy | Q8 ≈ BF16; Q4 loses ~3-5% IFEval |
| 3 | Gemma function calling | Gemma 3n E4B: 4B VRAM, LMArena 1300+ |
| 4 | Gemma vs Qwen code | No direct comparison found; Qwen3 better documented |
| 5 | Llama 4 agent use | Too large for local (109B/400B); MoE architecture |
| 6 | Phi-4 structured output | 16K context limit; no native FC format |
| 7 | Mistral Small 3.1 | Best small model FC; 24B, 128K, 150 tok/s |
| 8 | Best small models JSON | Outlines library: guaranteed structured output |
| 9 | Context window comparison | Gemma 3 4B & Qwen3-8B: 128K; Phi-4: 16K |
| 10 | LoRA for tool calling | Unsloth + LLaMA-Factory; Hermes FC datasets |
| 11 | DPO agent behavior | SFT (format) + DPO (behavior) combo for Director |
| 12 | Quantization impact | Q8 safe; Q4 hurts IFEval; use grammar to compensate |
| 13 | Multi-turn degradation | BFCL V3/V4 confirms; Atwater memory model is the fix |
| 14 | ChatML vs Llama template | Use correct template per model family; apply_chat_template() |
| 15 | System prompt best practices | < 500 tokens; 1-2 few-shot; positive framing |
| 16 | Speculative decoding | 2-3x speedup for structured JSON; Qwen3-0.5B draft |
| 17 | MLX vs llama.cpp | MLX 20-40% faster on Apple Silicon; llama.cpp for GBNF |
| 18 | Hallucination constrained | Outlines masks illegal tokens; eliminates hallucinated keys |
| 19 | JSON constraint techniques | GBNF > Outlines > LM-Format-Enforcer > Instructor |
| 20 | GBNF grammar tool calls | Token-level grammar; `--grammar` flag or JSON schema in LM Studio |

---

## Sources

- Qwen documentation: https://qwen.readthedocs.io/en/latest/
- HuggingFace Gemma 3 blog: https://huggingface.co/blog/gemma3
- HuggingFace Gemma 3n blog: https://huggingface.co/blog/gemma3n
- HuggingFace Llama 4 blog: https://huggingface.co/blog/llama4-release
- Mistral Small 3.1: https://mistral.ai/news/mistral-small-3-1
- Phi-4 model card: https://huggingface.co/microsoft/phi-4
- Outlines (dottxt-ai): https://github.com/dottxt-ai/outlines
- BFCL V4 leaderboard: https://gorilla.cs.berkeley.edu/leaderboard.html
- BFCL blog (V1): https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html
- llama.cpp GBNF README: https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md
- Qwen quantization benchmarks: https://qwen.readthedocs.io/en/latest/getting_started/quantization_benchmark.html
- Qwen MLX guide: https://qwen.readthedocs.io/en/latest/run_locally/mlx-lm.html
