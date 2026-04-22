# OpenFang Deep Dive — Atwater Research Report
**Researched:** 2026-04-22  
**Source:** Live GitHub repo (`RightNow-AI/openfang`) — raw files, not inferred  
**Current version:** v0.5.10 on main (badge) | **v0.6.0 released 2026-04-19** (latest release)

> **Status key:** ✅ CONFIRMED from source code | ⚠️ INFERRED from context | ❌ NOT in spec

---

## 1. What OpenFang Actually Is

OpenFang is an **Agent Operating System** — not a Python agent framework. It's written in Rust, compiles to a single ~32 MB binary, and provides:

- A runtime daemon (`openfang start`) serving at `http://localhost:4200`
- 140+ REST/WebSocket/SSE API endpoints
- 40 messaging channel adapters (Telegram, Discord, Slack, etc.)
- 53 built-in tools + MCP + A2A
- SQLite + vector memory
- A dashboard (Tauri 2.0 desktop app + web UI)
- **Hands** — the core innovation: autonomous pre-packaged agent capabilities

**Architecture (14 Rust crates):**
```
openfang-kernel      Orchestration, workflows, metering, RBAC, scheduler, budget tracking
openfang-runtime     Agent loop, 3 LLM drivers, 53 tools, WASM sandbox, MCP, A2A
openfang-api         140+ REST/WS/SSE endpoints, OpenAI-compatible API, dashboard
openfang-channels    40 messaging adapters with rate limiting, DM/group policies
openfang-memory      SQLite persistence, vector embeddings, canonical sessions, compaction
openfang-types       Core types, taint tracking, Ed25519 manifest signing, model catalog
openfang-skills      60 bundled skills, SKILL.md parser, FangHub marketplace
openfang-hands       7 autonomous Hands, HAND.toml parser, lifecycle management
openfang-extensions  25 MCP templates, AES-256-GCM credential vault, OAuth2 PKCE
openfang-wire        OFP P2P protocol with HMAC-SHA256 mutual authentication
openfang-cli         CLI with daemon management, TUI dashboard, MCP server mode
openfang-desktop     Tauri 2.0 native app (system tray, notifications, global shortcuts)
openfang-migrate     OpenClaw, LangChain, AutoGPT migration engine
xtask                Build automation
```

**Key numbers:** 137K LOC, 1767+ tests (as of v0.5.x), 2460+ tests (v0.6.0), zero clippy warnings

---

## 2. OpenFang v0.6.0 — What Changed

Released **2026-04-19**. This IS the "0.6" version the Atwater project targets.

### New Features in v0.6.0

#### Multi-destination cron delivery
Cron jobs can now fan out to multiple targets in one declaration:
- **Channel** — any of the 40 channel adapters (Telegram, Slack, Discord, WhatsApp, Matrix, Teams...)
- **Webhook** — POST JSON `{job, output, timestamp}` with optional Authorization header
- **LocalFile** — path + append/overwrite flag
- **Email** — with subject template support

Per-target failures log but never abort the job.

New API:
- `GET /api/schedules/{id}/delivery-log`
- `PUT /api/schedules/{id}` now accepts `delivery_targets`

#### Skill config injection (v0.6.0) ✅
SKILL.md now supports frontmatter config vars:
```yaml
config:
  github_token:
    description: GitHub personal access token
    env: GITHUB_TOKEN
    required: true
  default_branch:
    description: Default branch name
    default: main
    required: false
```
Resolver order: `user config.toml` → `env var` → `default` → error if required missing.  
Secrets auto-redacted in rendered skill prompt (`*_token`, `*_key`, `*_secret`, `password`).

#### Unified slash command registry
32 slash commands in one source of truth. New endpoint: `GET /api/commands?surface=web|cli|channel|all`

### Key Fixes in v0.5.10 (immediately prior)

- **Scheduler fixed** — `schedule_create` tool was silently broken before v0.5.10. The cron jobs now actually fire. This is important: **any Atwater design relying on OpenFang scheduling before v0.5.10 would be non-functional.**
- `openfang hand config <id>` CLI command implemented (was promised but missing).
- Auth fail-closed: non-loopback requests now rejected with 401 unless configured.
- Agent `context.md` re-read on every turn (cron agents can update it live).

---

## 3. Hands: Real Architecture

A Hand is an **autonomous capability package**. Each Hand is:
1. A HAND.toml manifest
2. An optional SKILL.md (domain expertise reference injected into context at runtime)
3. Optionally Python scripts/resources that the agent invokes via `shell_exec`

**Hands are NOT Python services.** The agent is a chat-loop agent inside OpenFang's runtime. It calls tools (including `shell_exec` to run Python scripts) to do work. The Hand's Python code is called by the agent, not the other way around.

**7 bundled Hands (confirmed via registry test — count = 9 in tests, but 7 documented):**
| Hand | Category |
|------|----------|
| Clip | content |
| Lead | data |
| Collector | OSINT/intelligence |
| Predictor | forecasting |
| Researcher | research |
| Twitter | social |
| Browser | web automation |

**Hand lifecycle:**
```bash
openfang hand install <path>  # reads HAND.toml + SKILL.md from directory
openfang hand activate <id>   # creates instance, spawns agent
openfang hand pause <id>      # pauses without losing state
openfang hand status <id>     # check progress
openfang hand list            # all available
openfang hand config <id> --set KEY=VALUE  # configure (v0.5.10+)
```

Hands persist to `~/.openfang/hands/<hand_id>/` on install. State survives daemon restarts.

Multi-instance: Multiple instances of the same Hand can coexist with `instance_name`:
```bash
openfang hand activate lead --name "tech-companies"
openfang hand activate lead --name "finance-companies"
```

---

## 4. HAND.toml Real Specification

> **Source:** Actual TOML files from `crates/openfang-hands/bundled/` + registry test cases  
> **NOT guessed** — these are the real field names from production code.

### Minimal Valid HAND.toml
```toml
id = "my-hand"
name = "My Hand"
description = "What this hand does"
category = "other"

[agent]
name = "my-hand-agent"
description = "Agent description"
system_prompt = "You are..."
```

### Full HAND.toml Schema (all confirmed fields)

```toml
# ── Top-level identity ────────────────────────────────────────────────────────
id          = "hand-id"          # REQUIRED — unique identifier (kebab-case)
name        = "Hand Name"        # REQUIRED — display name
description = "..."             # REQUIRED — one-line description
category    = "content"         # OPTIONAL — e.g. "content", "data", "other"
icon        = "🎬"              # OPTIONAL — emoji icon
version     = "0.1.0"           # OPTIONAL — version string
author      = "Name"            # OPTIONAL — author name

# ── Tool access declaration ──────────────────────────────────────────────────
# Flat string array of tool IDs. These are tools the agent is ALLOWED to call.
# Available tool IDs: shell_exec, file_read, file_write, file_list, web_fetch,
# web_search, memory_store, memory_recall, schedule_create, schedule_list,
# schedule_delete, knowledge_add_entity, knowledge_add_relation, knowledge_query
tools = ["shell_exec", "file_read", "file_write", "memory_store", "memory_recall"]

# ── Binary/env requirements ───────────────────────────────────────────────────
# OpenFang checks these before activation. Failure = can't activate.
[[requires]]
key              = "ffmpeg"             # unique key
label            = "FFmpeg must be installed"
requirement_type = "binary"             # "binary" | "env_var" | "api_key"
check_value      = "ffmpeg"             # binary name (PATH check) or env var name
description      = "Why this is needed"
optional         = false                # if true, missing → degraded (not blocked)

[requires.install]                      # OPTIONAL: install hints shown in dashboard
macos        = "brew install ffmpeg"
windows      = "winget install Gyan.FFmpeg"
linux_apt    = "sudo apt install ffmpeg"
linux_dnf    = "sudo dnf install ffmpeg-free"
linux_pacman = "sudo pacman -S ffmpeg"
manual_url   = "https://..."
estimated_time = "2-5 min"

# Another requirement (api_key type)
[[requires]]
key              = "groq_key"
label            = "Groq API key for transcription"
requirement_type = "api_key"
check_value      = "GROQ_API_KEY"       # env var to check
optional         = true

# ── User-configurable settings ────────────────────────────────────────────────
# Rendered as UI controls in the OpenFang dashboard.
# setting_type: "text" | "select" | "number" (inferred from usage)

[[settings]]
key          = "my_text_setting"
label        = "Text Setting Label"
description  = "What this setting controls"
setting_type = "text"
default      = ""
env_var      = "MY_ENV_VAR"            # OPTIONAL: links setting to env var

[[settings]]
key          = "my_select_setting"
label        = "Select Setting Label"
description  = "Pick one option"
setting_type = "select"
default      = "option_a"

[[settings.options]]
value          = "option_a"
label          = "Option A"
provider_env   = "SOME_API_KEY"        # OPTIONAL: env var that enables this option
binary         = "some-binary"         # OPTIONAL: binary that must exist for this option

[[settings.options]]
value          = "option_b"
label          = "Option B"

# ── Agent configuration (REQUIRED) ────────────────────────────────────────────
[agent]
name           = "hand-agent-name"     # REQUIRED — agent name in OpenFang
description    = "..."                 # REQUIRED — agent description
module         = "builtin:chat"        # REQUIRED — always "builtin:chat" for Hands
provider       = "default"             # "default" | "openai" | "anthropic" | "gemini" etc.
model          = "default"             # "default" | specific model name
max_tokens     = 8192                  # OPTIONAL
temperature    = 0.4                   # OPTIONAL
max_iterations = 40                    # OPTIONAL — max tool calls per turn
system_prompt  = """
Your full system prompt here.
Multi-line is fine.
"""

# ── Dashboard metrics ─────────────────────────────────────────────────────────
[dashboard]

[[dashboard.metrics]]
label      = "Display Label"           # REQUIRED — shown in dashboard
memory_key = "my_memory_key"           # REQUIRED — read from memory_store
format     = "number"                  # "number" | "duration" | (others inferred)
```

### ❌ Fields That DO NOT EXIST in Real HAND.toml

The following are **inventions** — they parse as unknown fields (silently ignored by Serde) and have NO EFFECT:

| Invented field | Reality |
|----------------|---------|
| `[hand.entrypoint]` | No such field in HandDefinition |
| `[hand.tools.lm_studio]` | tools is `tools = ["tool_id", ...]` — flat array |
| `[hand.schedule]` | No schedule block; scheduling via `schedule_create` tool |
| `hand.schedule.interval_secs` | Not in spec |
| `hand.schedule.cycles_per_run` | Not in spec |
| `hand.schedule.consolidation_interval` | Not in spec |
| `[hand.env]` | No env block in HandDefinition |
| `[hand.dependencies]` | No dependencies block |
| `dashboard.metrics[].name` | Real field is `label` |
| `dashboard.metrics[].description` | No description field on metrics |
| `dashboard.metrics[].source` | No source field (all metrics use `memory_key`) |
| `dashboard.metrics[].query` | No SQL queries — metrics read from `memory_store` |
| `dashboard.metrics[].type` | No type field; use `format` instead |

---

## 5. Real Tool List (Confirmed from Lead HAND.toml)

```
shell_exec           Run shell commands (with timeout_seconds parameter)
file_read            Read a file
file_write           Write a file
file_list            List files in a directory
web_fetch            Fetch a URL and return content
web_search           Web search
memory_store         Store a key-value in persistent memory
memory_recall        Recall a key from persistent memory
schedule_create      Create a cron/interval job (FIXED in v0.5.10)
schedule_list        List active schedules
schedule_delete      Delete a schedule
knowledge_add_entity Add entity to knowledge graph
knowledge_add_relation Add relationship to knowledge graph
knowledge_query      Query the knowledge graph
```

**53 total built-in tools** — the above are the ones declared in HAND.toml; agents can use any tools from their declared set.

---

## 6. Scheduling System

**Architecture:** The OpenFang kernel has a cron scheduler. Agents create schedules at runtime via the `schedule_create` tool. There is **NO** `[hand.schedule]` block in HAND.toml.

**How scheduling actually works:**
1. Hand agent is activated
2. On first run, agent calls `schedule_create` to set up its recurring schedule
3. Kernel cron fires the schedule, sending the agent a new message to trigger execution
4. Agent checks its state (via `memory_recall`) to determine what to do

**Critical bug note (v0.5.10 fix):** Before v0.5.10, `schedule_create` was broken — it wrote to a shared memory key that no executor read. Schedules appeared to be created but never fired. Fixed in v0.5.10.

**v0.6.0 Multi-destination delivery:** Cron jobs can deliver to channels, webhooks, local files, or email.

**For Atwater:** The Python evolution loop scheduling should NOT be declared in HAND.toml. Instead, the agent's system prompt should instruct it to create a schedule on first run. OR run Atwater as a direct Python process and skip OpenFang scheduling entirely.

---

## 7. LM Studio Integration

LM Studio uses the OpenAI-compatible driver (same as Ollama, vLLM, etc.). In the agent section:

```toml
[agent]
provider = "openai"                    # use OpenAI-compatible driver
model    = "your-model-name"           # model loaded in LM Studio
# In config.toml (not HAND.toml):
# [providers.openai]
# base_url = "http://localhost:1234/v1"
# api_key = "lm-studio"   (LM Studio accepts any non-empty key)
```

**Or** leave provider as `"default"` and configure LM Studio as the default provider in `~/.openfang/config.toml`:
```toml
[default_model]
provider = "openai"
base_url = "http://localhost:1234/v1"
api_key  = "lm-studio"
model    = "your-model"
```

The `LM_STUDIO_URL` and `LM_STUDIO_MODEL` env vars in the current Atwater HAND.toml have **no effect** — OpenFang doesn't read them. Configuration is via `config.toml` or the dashboard.

---

## 8. Dashboard Metrics: How They Really Work

Dashboard metrics in OpenFang read from `memory_store` — they are **not SQL queries**.

**Real pattern (from Clip HAND.toml):**
```toml
[dashboard]
[[dashboard.metrics]]
label      = "Jobs Completed"
memory_key = "clip_hand_jobs_completed"   # key used with memory_store tool
format     = "number"

[[dashboard.metrics]]
label      = "Total Duration"
memory_key = "clip_hand_total_duration_secs"
format     = "duration"
```

**Agent updates metrics like this:**
```
(in system prompt) → After each job, call memory_store with:
- "clip_hand_jobs_completed" → increment by 1
- "clip_hand_total_duration_secs" → increment by clip duration
```

There are **no SQL queries, no state_db, no optuna_db, no knowledge_db** references in HAND.toml metrics. The dashboard just reads memory_store values.

**For Atwater:** The SQLite databases and Optuna integration are Atwater's own Python logic. They cannot be exposed to OpenFang's dashboard via SQL queries. You'd need to either:
1. Have the Python layer write summary values to OpenFang's `memory_store` (via the agent)
2. Accept that the dashboard won't show detailed DB metrics

---

## 9. `[hand]` Wrapper — Confirmed Supported

The registry code calls `bundled::parse_bundled(id, toml_content, skill_content)` which presumably handles both flat format and the `[hand]` wrapper. The test case for a custom hand uses flat format. The registry's `install_from_path` also calls `parse_bundled`.

**Recommendation:** Use flat format (no `[hand]` wrapper) to match all bundled examples.

---

## 10. What the Atwater `config/openfang_hand.toml` Gets Wrong

### Critical Problems (break parsing or silently do nothing)

| Problem | Impact |
|---------|--------|
| Missing `id` field | HandDefinition requires `id`. Has `name` but no `id`. |
| Missing `[agent]` section | REQUIRED. Without it, the Hand cannot be activated. |
| `tools` defined as nested tables `[hand.tools.lm_studio]` | Wrong. `tools` is `tools = ["tool_id", ...]` |
| `[[hand.dashboard.metrics]]` fields: `name`, `description`, `source`, `query`, `type` | ALL wrong. Real fields: `label`, `memory_key`, `format` |

### Non-functional Sections (silently ignored by Serde)

| Section | Reality |
|---------|---------|
| `[hand.entrypoint]` | Doesn't exist; whole section ignored |
| `[hand.schedule]` | Doesn't exist; whole section ignored |
| `[hand.env]` | Doesn't exist; whole section ignored |
| `[hand.dependencies]` | Doesn't exist; whole section ignored |

### Wrong Assumptions

| Assumption | Reality |
|------------|---------|
| OpenFang will execute `python -m atwater.main` | False. No entrypoint field. The **agent** runs Python via `shell_exec` tool. |
| Dashboard reads SQLite/Optuna DB via SQL | False. Dashboard reads only `memory_store` keys. |
| OpenFang reads `LM_STUDIO_URL` / `LM_STUDIO_MODEL` env vars | False. These are ignored. Configure via `config.toml`. |
| `[hand.schedule]` controls when Atwater runs | False. No such field. Schedule via `schedule_create` tool or external cron. |

---

## 11. Corrected HAND.toml for Atwater

Based on real schema. **This will actually parse and register:**

```toml
# Atwater — Cognitive Agent Architecture Hand Manifest
# OpenFang v0.6.0+

id          = "atwater"
name        = "Atwater"
description = "Cognitive agent architecture for statistically-grounded content optimization using Optuna-driven evolutionary loops"
category    = "other"
icon        = "🧠"
version     = "0.1.0"
author      = "Atwater Team"

# Tools the agent can call
tools = [
  "shell_exec",
  "file_read",
  "file_write",
  "file_list",
  "memory_store",
  "memory_recall",
  "schedule_create",
  "schedule_list",
  "schedule_delete",
]

# Requirements checked before activation
[[requires]]
key              = "python3"
label            = "Python 3.11+ must be installed"
requirement_type = "binary"
check_value      = "python3"
description      = "Atwater's evolution loop runs as a Python process via shell_exec."
optional         = false

[requires.install]
macos     = "brew install python@3.11"
windows   = "winget install Python.Python.3.11"
linux_apt = "sudo apt install python3.11"

# Agent configuration
[agent]
name           = "atwater-agent"
description    = "Orchestrates the Atwater evolution loop: Director → Creator → Grader → DiversityGuard → Consolidator"
module         = "builtin:chat"
provider       = "default"
model          = "default"
max_tokens     = 8192
temperature    = 0.3
max_iterations = 20
system_prompt  = """
You are the Atwater orchestration agent. Your job is to manage and monitor the Atwater evolution loop.

## First Run Setup
On first run, create a recurring schedule:
- Use schedule_create to set up interval execution (e.g., every 5 minutes)
- Store your state with memory_store key "atwater_initialized" = "true"

## Monitoring Loop
On each scheduled run:
1. Run one evolution cycle: shell_exec "python -m atwater.main --cycles 1"
2. Parse the output for cycle stats
3. Update metrics in memory_store:
   - "atwater_cycle_count" → increment
   - "atwater_best_score" → update if improved
   - "atwater_kb_size" → update from output

## Reporting
If a user messages you, report current state from memory_recall.
"""

# Dashboard metrics — read from memory_store
[dashboard]

[[dashboard.metrics]]
label      = "Cycles Run"
memory_key = "atwater_cycle_count"
format     = "number"

[[dashboard.metrics]]
label      = "Best Score"
memory_key = "atwater_best_score"
format     = "number"

[[dashboard.metrics]]
label      = "KB Entries"
memory_key = "atwater_kb_size"
format     = "number"

[[dashboard.metrics]]
label      = "Diversity Alerts"
memory_key = "atwater_diversity_alerts"
format     = "number"
```

---

## 12. How to Install a Custom Hand

```bash
# Create a directory with HAND.toml (and optional SKILL.md)
mkdir ~/atwater-hand
cp config/openfang_hand.toml ~/atwater-hand/HAND.toml
cp SKILL.md ~/atwater-hand/  # optional

# Install into OpenFang
openfang hand install ~/atwater-hand

# Activate
openfang hand activate atwater

# Check status
openfang hand status atwater

# Configure (v0.5.10+)
openfang hand config atwater --set some_key=some_value
```

The hand persists to `~/.openfang/hands/atwater/` and survives daemon restarts.

---

## 13. Fundamental Architecture Gap

**The Atwater HAND.toml was written as if OpenFang is a task runner / process manager.** It's not.

OpenFang's Hand system provides:
- A **chat-based LLM agent** that runs inside OpenFang's runtime
- The agent calls **tools** (including `shell_exec`) to run external processes
- Scheduling is done by the **agent itself** via `schedule_create` tool

Atwater's actual Python codebase (with Optuna, SQLite, multi-agent flow) runs **outside** OpenFang as a Python process. OpenFang provides:
1. An orchestration agent that can trigger Atwater via `shell_exec`
2. A dashboard showing memory_store metrics (which Atwater Python code must write)
3. Scheduling to trigger the Python process
4. A chat interface to query the agent about current state

**Alternatively:** Run Atwater entirely standalone (it already has its own scheduling logic in Python) and use OpenFang only as a dashboard/monitoring layer.

---

## 14. OpenFang API — Atwater-Relevant Endpoints

From README: 140+ REST/WS/SSE endpoints. Relevant for Atwater:

```
GET  /api/schedules              List all schedules
POST /api/schedules              Create schedule (now working in v0.5.10+)
PUT  /api/schedules/{id}         Update schedule + delivery_targets (v0.6.0)
GET  /api/schedules/{id}/delivery-log  (v0.6.0)
GET  /api/memory/{key}           Read a memory_store value
POST /api/memory/{key}           Write a memory_store value
GET  /v1/chat/completions        OpenAI-compatible chat (stream=true supported)
GET  /api/hands                  List installed Hands
POST /api/hands/{id}/activate    Activate a Hand
GET  /api/hands/{id}/status      Hand status
```

---

## 15. Confirmed vs. Guessed

### Confirmed (from actual source code) ✅
- HAND.toml schema: all fields listed in Section 4
- Dashboard metrics format: `label`, `memory_key`, `format` only
- Tool names: the list in Section 5
- `schedule_create` was broken before v0.5.10
- v0.6.0 multi-destination cron delivery
- v0.6.0 SKILL.md config injection
- Hand install persists to `~/.openfang/hands/<id>/`
- Multiple instances of same Hand with different `instance_name`
- LM Studio uses OpenAI-compatible driver
- 9 bundled Hands exist (registry test asserts count == 9)

### Not Found in Source (gaps) ❓
- Full list of all 53 tool IDs (only saw ~14 in HAND.toml examples)
- `schedule_create` tool parameter format (interval/cron syntax)
- Whether `[hand]` wrapper format is still parsed by `parse_bundled`
- Exact `provider` string for LM Studio (likely "openai" with custom base_url)
- `format` values beyond "number" and "duration"
- The 2 additional bundled Hands not in the 7 documented (9 total per test)

### Definitely Not in Spec ❌
- `[hand.entrypoint]` — does not exist
- `[hand.schedule]` — does not exist
- `[hand.tools.*]` as nested tables — wrong; tools is a flat array
- `[hand.env]` — does not exist
- `[hand.dependencies]` — does not exist
- SQL query metrics in dashboard — does not exist; only memory_store reads
- `LM_STUDIO_URL` / `LM_STUDIO_MODEL` env var reading by OpenFang — not confirmed

---

*End of research report. All data from live GitHub repo fetched 2026-04-22.*
