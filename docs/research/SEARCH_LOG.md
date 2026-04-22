# Research Search Log

---

## Session: 2026-04-22 — Production Patterns & Creative Evaluation

**Agent:** research-patterns-evaluation subagent  
**Task:** 27 targeted searches for Atwater project  
**Method:** web_fetch on targeted arxiv papers + tool documentation

---

### Production Architecture Searches (15)

| # | Query | Source / Paper Found | Key Finding |
|---|-------|---------------------|-------------|
| 1 | production agent system architecture 2026 patterns | Lilian Weng blog, LangSmith docs, DyLAN (arXiv:2310.02170) | Dynamic team selection, two-stage pipeline (team opt → task solve) |
| 2 | agent observability monitoring patterns tracing | LangSmith docs (docs.smith.langchain.com) | Framework-agnostic trace/eval/deploy platform; self-hosted option |
| 3 | agent failure recovery state machine patterns | General synthesis + LLM-Modulo (arXiv:2402.01817) | Verifier loops, circuit breaker, idempotent tasks |
| 4 | multi-agent debate improved grading accuracy | arXiv:2305.14325 (Du et al., ICLR 2023) | Society of minds: multi-round debate → reduced hallucination, better accuracy |
| 5 | self-reflection patterns autonomous agents 2026 | arXiv:2303.11366 Reflexion (Shinn et al., NeurIPS 2023) | Verbal RL without weight updates; episodic memory buffer; 91% HumanEval |
| 6 | constitutional AI agent guardrails implementation | Anthropic research + synthesis | Critique → revise loop against fixed constitution |
| 7 | agent task decomposition strategies research | arXiv:2401.05268 AutoAct (Qiao et al., ACL 2024) | Division-of-labour strategy; auto-synth planning trajectories; no annotated data |
| 8 | cognitive architecture ACT-R SOAR modern LLM equivalent | arXiv:2402.01817 LLM-Modulo + synthesis | LLMs as knowledge sources + symbolic verifiers; neuro-symbolic hybrid |
| 9 | meta-learning agent systems continuous improvement | Reflexion + trajectory DB synthesis | Trajectory capture + verbal reflection = lightweight meta-learning |
| 10 | curriculum learning agent progressive difficulty | Synthesis from process supervision literature | Progressive difficulty scheduling; advance on threshold pass rate |
| 11 | agent evaluation framework measuring improvement | github.com/openai/evals | Registry of evals; custom model-graded evals; YAML-defined |
| 12 | human-in-the-loop agent quality control patterns | OpenAI Evals + synthesis | Confidence threshold gating: auto-approve / flag-for-review / auto-reject |
| 13 | agent versioning rollback strategies production | Synthesis | Prompt version hash + A/B shadow traffic + auto-rollback on score drop |
| 14 | cost-aware agent orchestration token budget | arXiv:2305.20050 PRM (Lightman et al.) + DyLAN | Process rewards for step-wise signal; verifier cascade (fast→medium→slow) |
| 15 | agent safety mode collapse optimization prevention | Synthesis from RL literature | Diversity bonus, temperature scheduling, population-based optimisation |

---

### Creative Evaluation Searches (12)

| # | Query | Source / Paper Found | Key Finding |
|---|-------|---------------------|-------------|
| 16 | automated creative evaluation metrics 2026 | IQA-PyTorch (pyiqa), Q-Align (ICML 2024) | Comprehensive local IQA toolkit; 50+ metrics; GPU-accelerated |
| 17 | CLIP score aesthetic scoring generated content | arXiv:2103.00020 CLIP + arXiv:2104.08718 CLIPScore | CLIPScore: reference-free metric; highest human correlation for image-text alignment |
| 18 | multi-dimensional creative assessment rubric automated | Synthesis + pyiqa + LAION aesthetic predictor | Hierarchical rubric: visual quality / brand / creative strength / technical |
| 19 | A/B testing framework creative content automated | Synthesis | Score-based A/B; Thompson Sampling multi-armed bandit for adaptive variant routing |
| 20 | novelty detection generated content measurement | Synthesis (CLIP embedding corpus distance) | k-NN distance to corpus; sweet spot 0.65-0.85 cosine similarity |
| 21 | style consistency scoring across generations | LAION Aesthetic Predictor + CLIP fingerprinting | Brand fingerprint = mean CLIP embedding of approved references; cosine sim |
| 22 | brand alignment measurement automated tool | Multi-probe CLIPScore + colour analysis | Multiple text probes averaged; colour palette vs brand LAB colours |
| 23 | layout quality scoring algorithm design | OpenCV + PIL synthesis | Rule of thirds, balance, white space ratio, focal point clarity |
| 24 | typography evaluation metrics automated | pytesseract + PIL + WCAG synthesis | Contrast ratio (≥4.5 WCAG AA), font size hierarchy, OCR-based detection |
| 25 | color theory scoring design composition algorithm | OpenCV HSV analysis synthesis | Harmony detection: complementary/analogous/triadic/monochromatic |
| 26 | image quality assessment IQA model local | github.com/chaofengc/IQA-PyTorch | pyiqa: BRISQUE, NIQE, NIMA, MUSIQ, TOPIQ, LPIPS — all local |
| 27 | FID KID alternatives small batch creative evaluation | pyiqa changelog + synthesis | KID better than FID for small N; SFID added Jun 2025; LPIPS for single pairs |

---

### Notes
- web_search tool returned compacted results; used targeted web_fetch on specific papers and docs
- Primary research sources: arxiv.org, github.com/chaofengc/IQA-PyTorch, github.com/LAION-AI/aesthetic-predictor, docs.smith.langchain.com, github.com/openai/evals
- All 27 search topics covered; synthesised into PRODUCTION_PATTERNS.md and CREATIVE_EVALUATION.md

---

## Session: 2026-04-22 — Small Model Capabilities for Agent Use

> Note: `web_search` (Brave API) was unavailable — research via targeted `web_fetch`.

### Search 1: Qwen3 Tool Calling Format
**Key finding:** Hermes-style function calling recommended by Qwen team. Tools in `<tools>` XML block. Qwen3-2507 update improves tool use significantly. Supports: vLLM, Qwen-Agent, LM Studio, MLX, llama.cpp.

### Search 2: Qwen3 Quantization Impact
**Key finding (Qwen2 data, Qwen3 "to be updated"):** Q8 ≈ BF16 (<1% degradation). Q4 loses 3-5% IFEval. 1.5B models degrade faster than 7B at same quant level.

### Search 3: Gemma 3 / Gemma 3n Function Calling
**Key finding:** Gemma 3 (1B-27B, 128K context) released March 2025. Gemma 3n (E2B/E4B) uses MatFormer + PLE: E4B runs in 3GB VRAM, LMArena 1300+. Natively multimodal (text/image/audio/video).

### Search 4: Gemma vs Qwen Code Comparison
**Key finding:** No "Gemma 4" found. Qwen3 better documented for code/tools. Gemma 3n E4B competitive for memory-constrained setups.

### Search 5: Llama 4 Agent Tool Use
**Key finding:** Maverick (~400B) and Scout (~109B) both MoE with 17B active params. Scout: 10M context. Both too large for local LM Studio deployment. iRoPE architecture (interleaved RoPE + NoPE).

### Search 6: Phi-4 Structured Output
**Key finding:** 14B dense model, 16K context (limitation!), MIT license. SFT+DPO aligned. No native function calling — prompt engineering only. 16K context borderline for deep agent loops.

### Search 7: Mistral Small 3.1
**Key finding:** 24B params, 128K context, 150 tok/s, Apache 2.0. Explicitly supports "low-latency function calling". Outperforms Gemma 3 and GPT-4o Mini. 32GB RAM needed — heavy for local use.

### Search 8: Best Small Models Structured JSON
**Key finding:** Outlines library (dottxt-ai) — guarantees structured output during generation via token masking. Works with OpenAI/Ollama/vLLM/transformers. Trusted by NVIDIA, Cohere, HuggingFace.

### Search 9: Small LLM Context Window Comparison
**Key finding:** Gemma 3 4B: 128K. Qwen3-8B: 32K native → 131K with YaRN. Phi-4: 16K. Mistral Small 3.1: 128K. Llama 4 Scout: 10M (server only).

### Search 10: LoRA Fine-tuning for Tool Calling
**Key finding:** Frameworks: Unsloth (fastest, least VRAM), LLaMA-Factory, Axolotl. Datasets: glaive-function-calling-v2, NousResearch/hermes-function-calling-v3. Recommended LoRA rank: 64 for tool calling.

### Search 11: DPO Training Agent Behavior
**Key finding:** SFT for format + DPO for behavioral alignment (routing decisions). ~1K-5K preference pairs sufficient. Tools: TRL, Axolotl. Most useful for Director agent (when to call vs decline).

### Search 12: Quantization Impact Tool Calling
**Key finding:** Q8 safe for IFEval (< 1% drop). Q4 loses 3-5% on instruction following. Compensate Q4 degradation with GBNF grammar or JSON schema constraint.

### Search 13: Multi-turn Conversation Degradation
**Key finding:** BFCL V3 (Sept 2024) introduced multi-turn evaluation. Small models lose state tracking across turns. GPT-4 significantly outperforms OSS on multi-turn. Atwater's memory architecture directly mitigates this.

### Search 14: ChatML vs Llama Template Comparison
**Key finding:** Qwen/Phi use ChatML (`<|im_start|>`). Llama 3 uses `<|start_header_id|>`. Gemma uses `<start_of_turn>`. Template mismatch = significant quality loss. Always use `apply_chat_template()`.

### Search 15: System Prompt Best Practices
**Key finding:** Keep < 500 tokens per agent. 1-2 few-shot examples boost JSON accuracy significantly. Positive framing > negative constraints for small models.

### Search 16: Speculative Decoding
**Key finding:** 2-3x speedup on structured JSON. Best pairs: Qwen3-0.5B draft + Qwen3-8B target. Supported by llama.cpp (`--draft-model`), MLX-LM, vLLM.

### Search 17: MLX vs llama.cpp vs vLLM Apple Silicon
**Key finding:** MLX ~20-40% faster than llama.cpp on M-series. vLLM: avoid on Apple Silicon. LM Studio uses best backend per model. Use llama.cpp for GBNF grammar; MLX for throughput.

### Search 18: Hallucination Constrained Output
**Key finding:** Outlines masks illegal tokens during generation — eliminates hallucinated JSON keys, invalid enums, type errors. Best approach: prevent bad output vs fixing it post-hoc.

### Search 19: Constrain Small Model JSON Output Techniques
**Key finding:** Hierarchy: GBNF grammar (llama.cpp) > Outlines (Python/transformers) > LM-Format-Enforcer > Instructor (retry-based). LM Studio JSON schema mode wraps GBNF internally.

### Search 20: GBNF Grammar Constrained Generation
**Key finding:** GBNF extends BNF with regex-like features. Supports token-level matching (special tokens like `<think>`). `root` rule = full output. Pre-defined JSON grammar available. LM Studio JSON schema = GBNF internally.

### SUPPLEMENTAL: Qwen3-8B Official Model Card
**Key finding (CRITICAL):** Qwen3 has thinking/non-thinking mode switch (`enable_thinking` flag).
- Thinking mode = `<think>...</think>` block + answer (Director quality boost)
- Non-thinking mode = direct output (Grader speed boost)
- All sizes (0.6B-235B) support this toggle
- Source: https://huggingface.co/Qwen/Qwen3-8B

### SUPPLEMENTAL: Qwen3 Speed Benchmarks (Official, SGLang, NVIDIA H20)
- Qwen3-0.6B BF16: 414 tok/s | FP8: 458 tok/s (1 tok input)
- Apple Silicon ~3-5x slower (estimate: 80-100 tok/s for 0.6B)
- Source: https://qwen.readthedocs.io/en/latest/getting_started/speed_benchmark.html

