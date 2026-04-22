# Production Agent Architecture Patterns

_Research compiled: 2026-04-22 | Atwater project_

---

## Overview

This document synthesises key patterns from production agent system research relevant to Atwater — a cognitive agent architecture with statistical optimisation. Focus is on patterns that are practical, locally deployable, and battle-tested in production.

---

## 1. Top Patterns We Should Implement

### 1.1 Reflexion / Verbal Reinforcement Learning
**Paper:** Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning" (arXiv:2303.11366, NeurIPS 2023)

- Agents verbally reflect on task feedback (scalar scores or free-form text), storing reflections in an **episodic memory buffer**
- Next episode starts with previous reflections as context — no weight updates required
- Achieves 91% pass@1 on HumanEval, beating GPT-4 at 80%
- **Atwater fit:** Directly maps to our per-session reflection loop. After each creative generation cycle, the agent writes a structured reflection note that seeds the next cycle's prompt.

```
Pattern:
  execute(task) → observe(feedback) → reflect(linguistic) → store(memory) → retry
```

### 1.2 Multi-Agent Debate / Society of Minds
**Paper:** Du et al., "Improving Factuality and Reasoning through Multiagent Debate" (arXiv:2305.14325)

- Multiple LLM instances independently produce answers, then debate over multiple rounds
- Convergence improves factual accuracy and reduces hallucinations
- Works with **black-box models** — no fine-tuning needed
- **Atwater fit:** Use for grading/evaluation. Spawn 3 evaluator agents with different rubric emphases; debate to consensus score. More reliable than single-judge scoring.

```
Pattern:
  [Agent A, Agent B, Agent C] → propose() → debate(N_rounds) → converge(answer)
```

### 1.3 Dynamic Agent Team Selection (DyLAN)
**Paper:** "A Dynamic LLM-Powered Agent Network for Task-Oriented Agent Collaboration" (arXiv:2310.02170, COLM 2024)

- Two-stage: (1) **Team Optimisation** via Agent Importance Score (unsupervised), (2) **Task Solving** with dynamically selected team
- Improves accuracy by up to 25% on MMLU subject domains vs fixed teams
- **Atwater fit:** Rather than always running all agents, select the optimal subset per task type. Reduces cost while maintaining quality.

### 1.4 Process Reward Models (PRMs) over Outcome Reward
**Paper:** Lightman et al., "Let's Verify Step by Step" (arXiv:2305.20050, OpenAI)

- PRMs provide feedback at each reasoning **step** rather than final output only
- Process supervision significantly outperforms outcome supervision on complex tasks (78% solve rate on MATH subset)
- PRM800K dataset: 800K step-level human feedback labels
- **Atwater fit:** Score creative generation step-by-step (brief → concept → execution → polish) rather than just grading the final output. Better gradient signal for improvement.

### 1.5 LLM-Modulo Framework (Hybrid Neuro-Symbolic)
**Paper:** Kambhampati et al., "LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks" (arXiv:2402.01817, ICML 2024)

- LLMs as knowledge sources + external verifiers/validators in tight bidirectional loop
- LLM generates candidates → symbolic verifier checks constraints → feedback loop
- **Atwater fit:** Use rule-based creative constraint checkers (brand colours, layout rules, typography specs) as the "symbolic verifier" layer. LLM generates; rules gate quality.

### 1.6 AutoAct: Division-of-Labour Agent Specialisation
**Paper:** Qiao et al., "AutoAct: Automatic Agent Learning for QA via Self-Planning" (arXiv:2401.05268, ACL 2024)

- Automatically differentiates sub-agents by role based on task information and synthesised trajectories
- No large-scale annotated data needed — bootstraps from limited seed data + tool library
- **Atwater fit:** Specialise distinct sub-agents (brief parser, concept generator, visual composer, quality scorer) with role-specific prompts synthesised from task analysis.

---

## 2. Failure Recovery Strategies

### 2.1 State Machine with Checkpointing

```
States: INIT → PLANNING → EXECUTING → VERIFYING → DONE | FAILED | RETRY

Transitions:
  PLANNING fails → retry with simplified decomposition (max 3x)
  EXECUTING fails → checkpoint last good state → resume from checkpoint
  VERIFYING fails (score < threshold) → route to RETRY with reflection note
  FAILED (3x retries) → escalate to human-in-the-loop
```

**Key rule:** Never retry naively. Each retry must include the failure reason in context — otherwise you're just running the same broken input again.

### 2.2 Exponential Backoff with Jitter
For tool calls and LLM API calls:
```python
delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
```
Prevents thundering herd on rate limits.

### 2.3 Circuit Breaker Pattern
- Track error rate over a sliding window (e.g., 5 failures in 60s)
- Open circuit → fast-fail with graceful degradation (use cached output or simpler model)
- Half-open after cooldown period → probe with single request before resuming

### 2.4 Graceful Degradation Tiers
```
Tier 1: Full pipeline (all agents, full compute)
Tier 2: Single-agent with reduced tool calls
Tier 3: Template fallback (pre-computed outputs)
Tier 4: Human escalation
```

Automatically step down on sustained failures. Never expose raw errors to end users.

### 2.5 Idempotent Tasks with Deduplication Keys
Each task should carry a `task_id` (hash of inputs). Before executing, check if result already exists. On retry, skip completed steps.

---

## 3. Observability Approaches

### 3.1 Structured Trace Format
Every agent action should emit a structured trace event:

```json
{
  "trace_id": "uuid",
  "span_id": "uuid",
  "parent_span_id": "uuid|null",
  "agent_name": "brief_parser",
  "action": "parse_brief",
  "input_tokens": 450,
  "output_tokens": 120,
  "latency_ms": 1250,
  "model": "claude-sonnet-4-6",
  "score": 0.87,
  "status": "success|failure|retry",
  "metadata": {}
}
```

Implement as a **local append-only JSONL log** per session. No cloud required.

### 3.2 Key Metrics to Track (Local Dashboard)

| Metric | Why |
|--------|-----|
| Per-agent success rate | Identify weak links |
| Token usage per pipeline run | Cost control |
| Score distribution per task type | Quality trends |
| Retry rate | Signals brittleness |
| End-to-end latency p50/p95 | UX quality |
| Reflection improvement delta | Does self-reflection actually help? |

### 3.3 LangSmith-style Local Tracing (Self-Hosted)
LangSmith (docs.smith.langchain.com) supports:
- Trace requests with full step visibility
- Evaluate outputs with custom scorers
- Compare model versions
- **Self-hosted option available** for privacy-sensitive workloads

For Atwater: Implement a lightweight `Tracer` class that writes JSONL locally and optionally pushes to a self-hosted LangSmith instance.

### 3.4 Prompt Version Tracking
Every prompt used in production should have:
- Version hash (SHA of prompt content)
- Commit reference
- A/B test tag (if running variants)
- Score rollup per version

This is the minimum viable prompt observability loop.

---

## 4. Multi-Agent Grading Techniques

### 4.1 Debate-Based Consensus Scoring
Based on Du et al. (arXiv:2305.14325):

```
1. Each evaluator agent scores independently (no visibility into others)
2. Scores + reasoning shared across evaluators
3. Evaluators revise scores in light of others' reasoning (1-2 rounds)
4. Final score = weighted average of final round scores
```

Use 3 evaluators minimum. Weight by evaluator specialisation (e.g., brand evaluator gets 2x weight on brand dimension).

### 4.2 Criteria-Separated Rubric Scoring
Don't ask one agent to score everything. Assign each dimension to a specialist:

| Evaluator | Criteria |
|-----------|----------|
| `visual_evaluator` | Layout, composition, visual hierarchy |
| `brand_evaluator` | Colour compliance, brand voice, consistency |
| `creative_evaluator` | Novelty, memorability, concept strength |
| `technical_evaluator` | Typography, contrast ratios, accessibility |

Aggregate with a configurable weight matrix per campaign type.

### 4.3 Calibrated Scoring via Examples
Each evaluator prompt should include 2-3 calibration examples with known scores ("here is a score-8 ad and why"). Without calibration, scores are inconsistent across runs.

### 4.4 Process Reward vs Outcome Reward
- **Outcome reward:** Score the final creative (easy, but coarse signal)
- **Process reward:** Score at each generation step — brief quality, concept strength, execution fidelity, refinement progress
- PRM signal trains the optimiser faster; use for Optuna's objective function

### 4.5 Verifier Cascade
```
Fast verifier (rule-based, <10ms): brand colour check, size check, format check
Medium verifier (CLIP/embedding, ~100ms): style consistency, brand alignment score
Slow verifier (LLM judge, ~2-5s): holistic quality, concept coherence, narrative
```

Run fast first; only escalate passing outputs to slow verifiers. Saves 60-70% of grading cost.

---

## 5. Safety Patterns

### 5.1 Constitutional AI / Self-Critique
**Anthropic's Constitutional AI approach:**
- Agent generates output
- Critique stage: agent identifies violations against a fixed constitution (brand rules, content policy, legal constraints)
- Revision stage: agent rewrites to fix violations
- Iterate until no violations remain (max N iterations)

For Atwater, the "constitution" = brand guidelines + legal constraints + quality floor thresholds.

### 5.2 Mode Collapse Prevention
Mode collapse = agent converges to a narrow range of outputs, losing diversity. Critical risk in optimisation loops.

**Prevention strategies:**
1. **Diversity bonus** in scoring function: penalise outputs too similar to recent N outputs (measured by CLIP cosine similarity or text embedding distance)
2. **Temperature scheduling:** start high (0.9-1.2) to explore, anneal toward 0.6-0.7 as quality improves
3. **Prompt mutation:** periodically inject novel stylistic constraints to break convergence
4. **Population-based approach:** maintain K candidate solutions simultaneously, not just top-1

### 5.3 Guardrails on Tool Calls
Before any agent makes an external call (API, file write, browser action):
- Validate input schema
- Check against allowlist of permitted actions
- Rate-limit per-agent, not just globally
- Log full input/output of every tool call

### 5.4 Output Filtering Layer
Never pass raw LLM output directly to users or downstream systems:
```
LLM output → format validator → content filter → schema enforcer → downstream
```

### 5.5 Human-in-the-Loop Escalation Gates
Define confidence thresholds explicitly:

```python
if score >= 0.85:
    auto_approve()
elif score >= 0.65:
    flag_for_review()  # human sees it before ship
else:
    auto_reject()  # regenerate
```

Track human decisions as training signal for improving evaluator calibration.

---

## 6. Meta-Learning & Continuous Improvement

### 6.1 Trajectory Database
Store every generation attempt with:
- Input (brief, constraints)
- Agent actions taken
- Output produced
- Human/auto score
- Was it accepted?

Use this database to improve prompts, identify failure patterns, and retrain evaluators.

### 6.2 Curriculum Learning
**Progressive difficulty scheduling:**
1. Start with simple, well-defined briefs (clear brand, single message)
2. Gradually introduce complexity (multi-audience, ambiguous briefs, novel formats)
3. Track performance at each difficulty level
4. Only advance when success rate > threshold at current level

This prevents early optimisation failures from poisoning the agent's trajectory history.

### 6.3 Prompt Version A/B Testing
For every significant prompt change:
- Route 10% of traffic to new version
- Collect scores for 100+ samples
- Statistical significance test before full rollout
- Auto-rollback if new version scores drop > 5% below baseline

---

## References

| Paper | ArXiv | Year |
|-------|-------|------|
| Reflexion: Language Agents with Verbal RL | 2303.11366 | 2023 |
| Improving Factuality via Multiagent Debate | 2305.14325 | 2023 |
| DyLAN: Dynamic LLM-Powered Agent Network | 2310.02170 | 2024 |
| Let's Verify Step by Step (PRM) | 2305.20050 | 2023 |
| LLMs Can't Plan, LLM-Modulo Frameworks | 2402.01817 | 2024 |
| AutoAct: Agent Learning via Self-Planning | 2401.05268 | 2024 |
| Training Verifiers to Solve Math Problems (GSM8K) | 2110.14168 | 2021 |
| Augmented Language Models Survey | 2302.07842 | 2023 |
| LangSmith observability platform | docs.smith.langchain.com | 2024 |
| OpenAI Evals framework | github.com/openai/evals | 2023 |
