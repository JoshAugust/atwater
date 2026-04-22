"""
src.config.agent_configs — Per-agent model configuration for the Atwater system.

Each agent role has a recommended model, quantization, thinking mode,
temperature, and token budget. These defaults can be overridden via
config/settings.json under the "agent_configs" key.

Default model assignments (Phase 2C):
┌─────────────────┬────────────┬──────────┬──────────┬─────────────┐
│ Agent           │ Model      │ Quant    │ Thinking │ Temperature │
├─────────────────┼────────────┼──────────┼──────────┼─────────────┤
│ director        │ Qwen3-8B   │ Q8_0     │ ON       │ 0.1         │
│ creator         │ Qwen3-8B   │ Q5_K_M   │ OFF      │ 0.7         │
│ grader          │ Qwen3-4B   │ Q8_0     │ OFF      │ 0.0         │
│ diversity_guard │ Qwen3-4B   │ Q4_K_M   │ OFF      │ 0.8         │
│ orchestrator    │ Qwen3-8B   │ Q8_0     │ ON       │ 0.3         │
│ consolidator    │ Qwen3-8B   │ Q8_0     │ ON       │ 0.3         │
└─────────────────┴────────────┴──────────┴──────────┴─────────────┘

Usage
-----
    from src.config.agent_configs import get_agent_config, AgentModelConfig

    cfg = get_agent_config("director")
    response = llm_client.chat(
        messages,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        model=cfg.model_name,
        thinking_mode=cfg.thinking_mode,
    )

Overriding via settings.json
----------------------------
Add an "agent_configs" dict in config/settings.json. Each key is a role name;
each value is a partial dict — only fields you want to override are needed:

    {
      "agent_configs": {
        "creator": {
          "temperature": 0.5,
          "thinking_mode": true
        },
        "grader": {
          "model_name": "qwen3-4b-q8_0",
          "max_tokens": 512
        }
      }
    }

The model_name should match the exact ID string that LM Studio reports for the
loaded model (check GET /v1/models). If model_name is None the LLM client falls
back to its globally configured model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AgentModelConfig:
    """
    Model configuration for a single agent role.

    Attributes
    ----------
    model_name : str | None
        LM Studio model identifier for this agent's calls.
        If None, the LMStudioClient falls back to its own default.
    quantization : str
        Informational label for the recommended quantization (e.g. "Q8_0").
        Not sent to the API — used for documentation and logging.
    thinking_mode : bool
        When True, enable_thinking=True is added to the LM Studio request,
        activating Qwen3 extended chain-of-thought reasoning.
    temperature : float
        Sampling temperature for this agent's LLM calls.
    max_tokens : int
        Maximum tokens to generate per call.
    """

    model_name: str | None = None
    quantization: str = "Q8_0"
    thinking_mode: bool = False
    temperature: float = 0.7
    max_tokens: int = 2048

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "quantization": self.quantization,
            "thinking_mode": self.thinking_mode,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentModelConfig":
        """Create from a (possibly partial) dict, falling back to defaults."""
        defaults = cls()
        return cls(
            model_name=data.get("model_name", defaults.model_name),
            quantization=data.get("quantization", defaults.quantization),
            thinking_mode=bool(data.get("thinking_mode", defaults.thinking_mode)),
            temperature=float(data.get("temperature", defaults.temperature)),
            max_tokens=int(data.get("max_tokens", defaults.max_tokens)),
        )


# ---------------------------------------------------------------------------
# Default configuration table
# ---------------------------------------------------------------------------
# model_name=None means "use whatever is globally configured in settings".
# To target a specific model, set model_name to the exact LM Studio model ID.

_DEFAULT_AGENT_CONFIGS: dict[str, AgentModelConfig] = {
    "director": AgentModelConfig(
        model_name=None,        # Qwen3-8B Q8_0 — load this in LM Studio
        quantization="Q8_0",
        thinking_mode=True,     # Director benefits from reasoning
        temperature=0.1,
        max_tokens=2048,
    ),
    "creator": AgentModelConfig(
        model_name=None,        # Qwen3-8B Q5_K_M
        quantization="Q5_K_M",
        thinking_mode=False,    # Fast creative generation, no COT overhead
        temperature=0.7,
        max_tokens=2048,
    ),
    "grader": AgentModelConfig(
        model_name=None,        # Qwen3-4B Q8_0
        quantization="Q8_0",
        thinking_mode=False,    # Structured scoring — deterministic, fast
        temperature=0.0,
        max_tokens=1024,
    ),
    "diversity_guard": AgentModelConfig(
        model_name=None,        # Qwen3-4B Q4_K_M
        quantization="Q4_K_M",
        thinking_mode=False,    # Lightweight similarity check
        temperature=0.8,
        max_tokens=512,
    ),
    "orchestrator": AgentModelConfig(
        model_name=None,        # Qwen3-8B Q8_0
        quantization="Q8_0",
        thinking_mode=True,     # Orchestrator plans across agents — COT helps
        temperature=0.3,
        max_tokens=2048,
    ),
    "consolidator": AgentModelConfig(
        model_name=None,        # Qwen3-8B Q8_0
        quantization="Q8_0",
        thinking_mode=True,     # Synthesis task — COT improves coherence
        temperature=0.3,
        max_tokens=4096,        # Consolidator writes longer summaries
    ),
}

# Populated by apply_overrides(); consumers should call get_agent_config().
_ACTIVE_CONFIGS: dict[str, AgentModelConfig] = dict(_DEFAULT_AGENT_CONFIGS)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def get_agent_config(role: str) -> AgentModelConfig:
    """
    Return the AgentModelConfig for a given agent role.

    Falls back to a generic default if the role is not in the table.
    The generic default has thinking_mode=False and temperature=0.7 to
    match the original LMStudioClient defaults, ensuring backward compat.

    Parameters
    ----------
    role : str
        Agent role string (e.g. "director", "grader").

    Returns
    -------
    AgentModelConfig
    """
    cfg = _ACTIVE_CONFIGS.get(role)
    if cfg is None:
        logger.warning(
            "No AgentModelConfig found for role %r — using generic defaults.", role
        )
        return AgentModelConfig()
    return cfg


def apply_overrides(overrides: dict[str, dict[str, Any]]) -> None:
    """
    Merge per-role overrides from settings.json into the active config table.

    This is called by config.settings.load_settings() after reading the JSON
    file, so settings.json overrides win over the compile-time defaults above.

    Parameters
    ----------
    overrides : dict[str, dict[str, Any]]
        Mapping of role → partial AgentModelConfig dict.
        Unknown roles are added as new entries.
        Unknown keys within a role dict are ignored.
    """
    for role, raw in overrides.items():
        if not isinstance(raw, dict):
            logger.warning("Ignoring invalid agent_configs entry for %r: %r", role, raw)
            continue
        base = _DEFAULT_AGENT_CONFIGS.get(role, AgentModelConfig())
        # Merge: start from default, overlay only provided keys.
        merged_dict = base.to_dict()
        merged_dict.update({k: v for k, v in raw.items() if k in merged_dict})
        _ACTIVE_CONFIGS[role] = AgentModelConfig.from_dict(merged_dict)
        logger.debug("agent_configs: applied overrides for role %r → %s", role, _ACTIVE_CONFIGS[role])


def list_configs() -> dict[str, AgentModelConfig]:
    """Return a copy of the current active config table (for debugging/logging)."""
    return dict(_ACTIVE_CONFIGS)
