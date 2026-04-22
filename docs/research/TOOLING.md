# TOOLING.md — Developer Tooling Research

> Research agent: tooling-synthesis | Date: 2026-04-22

---

## 1. LM Studio Capabilities (Critical for Atwater)

### Current State (v0.4.0+)

LM Studio has evolved significantly and now provides a mature local inference platform:

**API Compatibility:**
- **OpenAI-compatible endpoints**: `/v1/chat/completions`, `/v1/responses`, `/v1/completions`, `/v1/embeddings`, `/v1/models`
- **Anthropic-compatible endpoints**: Claude-style Messages API against local models
- **Native Python SDK** (`pip install lmstudio`): Convenience, scoped-resource, and async APIs
- **Native TypeScript SDK** (`@lmstudio/sdk`)
- **Stateful REST API** (`/v1/chat`): Conversation state maintained server-side via `previous_response_id`

**Key Features for Atwater:**
- **Parallel requests with continuous batching**: Up to N simultaneous requests per model (default 4 slots, unified KV cache). This is crucial — Atwater can run multiple agent evaluations against the same model concurrently.
- **`/v1/responses` endpoint**: Supports streaming, reasoning (with `effort` param), prior response state, and **Remote MCP tools** (opt-in). Models can call external tools via MCP servers.
- **`llmster` daemon**: Headless deployment without GUI. Install via `curl -fsSL https://lmstudio.ai/install.sh | bash`. Perfect for CI/CD and server deployments.
- **Permission keys**: Fine-grained access control for the local server.
- **CLI (`lms`)**: `lms daemon up`, `lms get <model>`, `lms server start`, `lms chat`, `lms log stream`

**Structured Output:**
- Chat completions support standard OpenAI parameters: `temperature`, `top_p`, `top_k`, `max_tokens`, `stop`, `seed`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `repeat_penalty`
- The `/v1/responses` endpoint supports tool use patterns (MCP integration)
- **JSON mode / structured output via response_format is supported** through the OpenAI-compatible interface (same as OpenAI's `response_format: { type: "json_object" }`)

**Python SDK Details (v1.5.0+):**
- Three API styles: convenience (sync one-liner), scoped resource (context managers), async (structured concurrency)
- Configurable sync timeout (default 60s, adjustable via `lmstudio.set_sync_api_timeout()`)
- Agent support: define functions as tools for autonomous agent flows running entirely locally

**What This Means for Atwater:**
1. We can use `openai.OpenAI(base_url="http://localhost:1234/v1")` — zero code changes between local and cloud
2. Parallel batching means Optuna trials can evaluate concurrently (major speedup)
3. MCP tool integration opens door for agents that call external tools locally
4. The `llmster` daemon means we can script model loading/unloading programmatically
5. Stateful conversations reduce token overhead for multi-turn agent sessions

### Gaps / Risks
- **No explicit structured output documentation found** for JSON Schema enforcement (grammar-constrained decoding). LM Studio may rely on model compliance rather than guaranteed output format.
- MLX engine doesn't yet support parallel requests (llama.cpp only)
- Some advanced features (like response_format with json_schema) may have partial support — needs runtime testing

---

## 2. Debugging & Tracing Tools

### Tier 1: Best Fit for Atwater

#### Pydantic Logfire
- **What**: Observability platform from the Pydantic team, built on OpenTelemetry
- **Why it matters**: Python-centric insights, event-loop telemetry, profiling, SQL-queryable traces
- **Key features**: `logfire.span()` for manual tracing, auto-instrumentation for FastAPI/httpx/etc., rich Python object display
- **Fit for Atwater**: Excellent for tracing agent decision chains. OpenTelemetry foundation means we can export to any OTel backend. SDKs for Python, TypeScript, Rust.
- **Caveat**: Platform (UI/backend) is closed source. Self-hosting requires enterprise license. SDK is open source though — can export to Jaeger/Grafana.
- **Recommendation**: Use the SDK for instrumentation, export to a local OTel collector (Jaeger) for full local-first operation.

#### AgentOps
- **What**: Python SDK for AI agent monitoring, cost tracking, benchmarking
- **Why it matters**: Purpose-built for agent systems with session replay, step-by-step execution graphs
- **Key features**: `@session`, `@agent`, `@operation`, `@workflow` decorators; auto-LLM call tracking; supports async/generators; self-hostable (MIT license, open source app)
- **Fit for Atwater**: The decorator-based approach maps perfectly to Atwater's agent lifecycle. Session replays would give incredible visibility into optimization loops.
- **Recommendation**: Strong candidate for agent-level tracing. Can self-host entirely.

#### LangSmith
- **What**: Framework-agnostic platform for building, debugging, deploying AI agents
- **Key features**: Observability, evaluation, prompt engineering, deployment, CLI
- **Fit for Atwater**: Overkill for our needs — we're not using LangChain. The evaluation features are interesting but we're building our own with Optuna.
- **Recommendation**: Skip. We'd be adopting a platform when we need a library.

### Tier 2: Worth Watching

#### MLflow
- **What**: Open source AI engineering platform (60M+ monthly downloads)
- **Key features**: Tracing (OpenTelemetry-based), evaluation (50+ built-in metrics), prompt management, AI Gateway for cost/routing
- **Fit for Atwater**: The experiment tracking is mature but heavy. `mlflow.openai.autolog()` is nice but we want more control.
- **Recommendation**: Consider for experiment tracking if our custom Optuna integration proves insufficient. The `uvx mlflow server` one-liner is appealing.

### Recommendation
**Primary**: AgentOps for agent-level observability (self-hosted, decorator-based, MIT) + Python's built-in `logging` with structured output  
**Secondary**: Pydantic Logfire SDK → local Jaeger for deep tracing when debugging specific issues  
**Skip**: LangSmith (platform lock-in), MLflow (too heavy for our use case)

---

## 3. Local Experiment Tracking Alternatives

### For Optuna Optimization Runs

#### Optuna Dashboard (optuna-dashboard)
- **What**: Real-time web dashboard for Optuna studies
- **Install**: `pip install optuna-dashboard` → `optuna-dashboard sqlite:///db.sqlite3`
- **Features**: Live optimization history, hyperparameter importance, parallel coordinate plots, contour plots, Pareto fronts, EDF plots, rank plots, timeline visualization
- **Integrations**: JupyterLab extension, VS Code extension, browser-only version (SQLite3 Wasm + Rust — no Python needed!)
- **Docker**: `ghcr.io/optuna/optuna-dashboard` with SQLite3/MySQL/PostgreSQL support
- **Recommendation**: **Must-have**. This is our primary visualization for Optuna trials. Zero additional code needed — just point at the SQLite storage.

#### Optuna Built-in Visualization (`optuna.visualization`)
- Uses Plotly for interactive charts: `plot_contour`, `plot_edf`, `plot_hypervolume_history`, `plot_intermediate_values`, `plot_optimization_history`, `plot_parallel_coordinate`, `plot_param_importances`, `plot_pareto_front`, `plot_rank`, `plot_slice`, `plot_timeline`
- Requires `scikit-learn` for param importance
- **Recommendation**: Use for programmatic chart generation (export to reports), use Dashboard for interactive exploration.

#### Aim
- **What**: Open-source, self-hosted ML experiment tracker designed for 10K+ training runs
- **Key features**: Beautiful comparison UI, Python expression querying, system resource tracking, real-time alerting
- **Fit for Atwater**: The query-by-Python-expression feature is powerful. Could track Optuna trials as "runs" with hyperparameters as tracked params.
- **Recommendation**: Consider as a supplementary dashboard if Optuna Dashboard is insufficient for cross-study comparison.

#### MLflow (Local Mode)
- `uvx mlflow server` → runs entirely local on port 5000
- Tracks experiments, parameters, metrics, artifacts
- **Recommendation**: Heavier than needed. Optuna Dashboard + Aim covers our use case better.

### Recommendation
**Primary**: Optuna Dashboard (it's literally built for this)  
**Secondary**: Optuna's built-in `optuna.visualization` for programmatic exports  
**Tertiary**: Aim if we need cross-experiment comparison beyond what Optuna provides

---

## 4. TUI Dashboard Approaches

### Textual (by Textualize)
- **What**: The lean application framework for Python. Build sophisticated UIs with a simple Python API. Run in terminal AND web browser.
- **Key features**: CSS-based styling, de-coupled components, advanced testing framework, predefined themes, widget library (buttons, trees, data tables, inputs, text areas), async under the hood (but doesn't force it)
- **Cross-platform**: macOS, Linux, Windows
- **Why it fits Atwater**: We could build a real-time agent monitoring TUI that shows:
  - Current Optuna trial status (params, objective value)
  - Agent conversation flow (thinking → acting → observing)
  - Token usage / cost tracking
  - Model performance metrics (tokens/sec, latency)
  - Live log stream
- **The web browser mode** is particularly interesting — same codebase renders in terminal or browser
- **Recommendation**: **Best choice for Atwater's TUI**. Modern API, actively maintained, testing support, dual terminal/web rendering.

### Rich (by Textualize)
- Lower-level library that Textual is built on
- Great for formatted console output (tables, progress bars, syntax highlighting)
- Not a full application framework
- **Recommendation**: Use Rich within Textual for formatting, but build the TUI with Textual.

### Alternative: Plain curses / blessed
- Too low-level for what we need
- **Recommendation**: Skip in favor of Textual

### TUI Architecture for Atwater
```
┌─────────────────────────────────────────────────┐
│  Atwater Agent Monitor                    [q]uit│
├───────────────┬─────────────────────────────────┤
│ Optuna Study  │  Agent Activity Log             │
│ ────────────  │  ─────────────────              │
│ Trial #42     │  [15:03:01] Think: analyzing... │
│ temp=0.7      │  [15:03:03] Act: calling tool   │
│ top_p=0.9     │  [15:03:04] Observe: got result │
│ score=0.847   │  [15:03:05] Think: evaluating   │
│               │                                 │
│ Best: 0.892   │                                 │
│ Trials: 42/100│                                 │
├───────────────┼─────────────────────────────────┤
│ Performance   │  Token Budget                   │
│ 45.2 tok/s    │  ████████░░ 80% used            │
│ TTFT: 120ms   │  1,234 / 1,500 tokens           │
└───────────────┴─────────────────────────────────┘
```

---

## 5. Testing Strategies for Agent Systems

### Core Challenge
Agent systems are non-deterministic, multi-step, and depend on external services (LLMs). Traditional unit testing breaks down.

### Recommended Testing Pyramid for Atwater

#### Layer 1: Unit Tests (Fast, Deterministic)
- **Mock the LLM**: Replace LM Studio calls with canned responses
- **Test decision logic**: Given this observation, does the agent choose the right action?
- **Test prompt construction**: Does the prompt builder produce correct prompts for given state?
- **Test Optuna integration**: Given trial params, are they correctly applied to the agent config?
- **Test structured output parsing**: Given valid/invalid JSON, does the parser handle it correctly?
- **Tools**: `pytest` + `pytest-asyncio` + `unittest.mock` / `pytest-mock`

#### Layer 2: Integration Tests (Slower, Semi-Deterministic)
- **Test against real LM Studio** with a small model (e.g., Qwen3-4B) and `seed` parameter for reproducibility
- **Test full agent loops**: prompt → LLM → parse → decide → act (with mocked tools)
- **Test Optuna trial lifecycle**: create study → suggest params → run agent → report result
- **Snapshot testing**: Record LLM responses, replay in future tests to detect prompt regressions
- **Tools**: `pytest` + `pytest-recording` (VCR-style) + `seed` for reproducible LLM output

#### Layer 3: Evaluation Tests (Slow, Statistical)
- **Run N trials and assert statistical properties**: "mean score > 0.7", "p95 latency < 5s"
- **A/B comparison tests**: "new prompt config is not worse than baseline (p < 0.05)"
- **Regression detection**: Track metric distributions across commits
- **Tools**: `pytest` + `scipy.stats` for hypothesis testing + custom fixtures

#### Layer 4: End-to-End Smoke Tests
- **Full pipeline**: Load model → configure agent → run Optuna study (3-5 trials) → verify results stored
- **Run sparingly**: CI nightly or pre-release only
- **Tools**: `pytest` with long timeout markers

### Specific Patterns

**Fixture for LM Studio connection:**
```python
@pytest.fixture
def lm_client():
    """Provides an OpenAI client pointed at local LM Studio."""
    return OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

@pytest.fixture
def mock_lm_client():
    """Provides a mock that returns canned responses."""
    client = MagicMock()
    client.chat.completions.create.return_value = CANNED_RESPONSE
    return client
```

**Parametrized prompt testing:**
```python
@pytest.mark.parametrize("persona,expected_tone", [
    ("analytical", "formal"),
    ("creative", "playful"),
])
def test_prompt_tone(persona, expected_tone, mock_lm_client):
    agent = Agent(persona=persona, client=mock_lm_client)
    prompt = agent.build_prompt("test task")
    assert expected_tone_marker in prompt
```

**Statistical evaluation test:**
```python
@pytest.mark.slow
def test_optimization_improves_baseline(lm_client):
    """Verify that 10 Optuna trials beat the default config."""
    baseline = run_agent_with_defaults(lm_client, n=5)
    optimized = run_optuna_study(lm_client, n_trials=10)
    assert optimized.best_value > baseline.mean_score
```

**VCR-style response recording:**
```python
@pytest.fixture
def recorded_responses(tmp_path):
    """Record/replay LLM responses for deterministic testing."""
    cassette_path = tmp_path / "cassettes"
    # Use a custom VCR that intercepts OpenAI client calls
    return LLMCassette(cassette_path, record_mode="once")
```

### Recommended pytest Plugins
| Plugin | Purpose |
|--------|---------|
| `pytest-asyncio` | Async test support (for async LM Studio SDK) |
| `pytest-mock` | Simplified mocking |
| `pytest-timeout` | Prevent hung tests from blocking CI |
| `pytest-xdist` | Parallel test execution |
| `pytest-benchmark` | Performance regression testing |
| `pytest-randomly` | Randomize test order to catch hidden state |
| `pytest-cov` | Coverage reporting |

---

## 6. Python Profiling for IO-Bound Agent Workloads

### The Challenge
Agent workloads are overwhelmingly IO-bound (waiting for LLM responses). Traditional CPU profilers miss the picture entirely.

### Recommended Tools

**For async IO profiling:**
- `asyncio` debug mode: `PYTHONASYNCIODEBUG=1` — warns on slow callbacks
- `py-spy`: Sampling profiler that works with async code, zero overhead
- Logfire's event-loop telemetry: Shows where the event loop is blocked

**For token throughput:**
- Custom instrumentation: Track `tokens_per_second`, `time_to_first_token`, `total_latency` per LLM call
- LM Studio provides these stats in `/v1/responses` output

**For memory profiling:**
- `tracemalloc`: Built-in Python memory tracking
- `memray`: Modern memory profiler for Python (async-aware)

**Recommendation**: Don't over-invest here. The bottleneck is always the LLM. Focus instrumentation on:
1. Time waiting for LLM responses vs. time in application logic
2. Token throughput per trial
3. Memory usage of conversation history buffers

---

## Summary: Tooling Stack for Atwater

| Category | Primary Choice | Why |
|----------|---------------|-----|
| LLM Runtime | LM Studio (llmster) | OpenAI-compatible, parallel batching, headless daemon, MCP tools |
| Agent Tracing | AgentOps (self-hosted) | Decorator-based, session replay, MIT license |
| Experiment Viz | Optuna Dashboard | Purpose-built, real-time, zero-config with SQLite |
| TUI Monitor | Textual | Modern Python TUI framework, dual terminal/web |
| Testing | pytest + layered strategy | Unit → Integration → Statistical → E2E |
| Profiling | py-spy + custom LLM metrics | Focus on IO wait time, not CPU |
| Deep Debugging | Logfire SDK → Jaeger | OpenTelemetry-based, when needed |

### Key Insight
The tooling ecosystem has matured significantly. We should **not** build custom versions of things that exist. Specifically:
- Don't build a custom dashboard — use Optuna Dashboard + Textual TUI
- Don't build custom tracing — use AgentOps decorators
- Don't build custom experiment tracking — Optuna's SQLite storage + Dashboard is the answer
- **Do** build: the agent loop, the optimization logic, the prompt engineering — that's the novel part
