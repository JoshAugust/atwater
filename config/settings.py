"""
config.settings — Central configuration for the Atwater cognitive architecture.

Usage
-----
    from config.settings import load_settings

    settings = load_settings()                          # uses default path
    settings = load_settings("config/settings.json")   # explicit path
    settings = load_settings(env_override=True)         # env vars win over JSON

Environment variable overrides (all optional):
    ATWATER_LM_STUDIO_URL        → lm_studio_url
    ATWATER_MODEL_NAME           → model_name
    ATWATER_STATE_DB_PATH        → state_db_path
    ATWATER_KNOWLEDGE_DB_PATH    → knowledge_db_path
    ATWATER_OPTUNA_DB_PATH       → optuna_db_path
    ATWATER_CONSOLIDATION_INTERVAL → consolidation_interval
    ATWATER_DIVERSITY_THRESHOLD  → diversity_threshold
    ATWATER_MAX_KNOWLEDGE_RESULTS → max_knowledge_results
    ATWATER_EXPLORATION_RATIO    → exploration_ratio
    ATWATER_TOKEN_BUDGET         → token_budget
    ATWATER_LOG_LEVEL            → log_level
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = "config/settings.json"


@dataclass
class Settings:
    """
    All configurable values for the Atwater system.

    Attributes
    ----------
    lm_studio_url : str
        Base URL for the LM Studio OpenAI-compatible API.
    model_name : str | None
        Model identifier. If None, the first model returned by /v1/models is used.
    state_db_path : str
        Path to the SQLite database for shared agent state.
    knowledge_db_path : str
        Path to the SQLite database for the knowledge base.
    optuna_db_path : str
        Optuna storage URL (SQLite), e.g. "sqlite:///data/trials.db".
    consolidation_interval : int
        Number of cycles between consolidator runs. Default: 50.
    diversity_threshold : float
        Fraction of last-N trials above which an asset is flagged for rotation.
        Default: 0.30 (30%).
    max_knowledge_results : int
        Maximum number of knowledge entries to inject per agent context.
        Default: 5.
    exploration_ratio : float
        Fraction of cycles that use a RandomSampler instead of TPE for
        forced exploration. Default: 0.20 (20%).
    token_budget : int
        Soft cap on total tokens per agent prompt (used for context trimming).
        Default: 2048.
    log_level : str
        Python logging level string. Default: "INFO".
    study_name : str
        Optuna study name. Default: "atwater_production".
    optuna_seed : int
        Random seed for the Optuna TPE sampler. Default: 42.
    diversity_window : int
        Number of recent trials to consider when computing asset concentration.
        Default: 50.
    knowledge_min_confidence : float
        Minimum confidence required to include a knowledge entry in a prompt.
        Default: 0.30.
    """

    # LLM
    lm_studio_url: str = "http://localhost:1234/v1"
    model_name: str | None = None

    # Databases
    state_db_path: str = "data/state.db"
    knowledge_db_path: str = "data/knowledge.db"
    optuna_db_path: str = "sqlite:///data/trials.db"

    # Cycle control
    consolidation_interval: int = 50
    diversity_threshold: float = 0.30
    max_knowledge_results: int = 5
    exploration_ratio: float = 0.20
    token_budget: int = 2048

    # Logging
    log_level: str = "INFO"

    # Optuna
    study_name: str = "atwater_production"
    optuna_seed: int = 42
    diversity_window: int = 50

    # Knowledge base
    knowledge_min_confidence: float = 0.30

    def to_dict(self) -> dict[str, Any]:
        """Serialise settings to a plain dict (JSON-safe)."""
        return asdict(self)

    def save(self, config_path: str = _DEFAULT_CONFIG_PATH) -> None:
        """
        Persist current settings to a JSON file.

        Creates parent directories if they don't exist.

        Parameters
        ----------
        config_path : str
            File path to write. Existing file is overwritten.
        """
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        logger.info("Settings saved to %s", path)

    def configure_logging(self) -> None:
        """Apply log_level to the root logger."""
        level = getattr(logging, self.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """
        Return a list of validation error messages (empty = valid).
        Call after loading from JSON or env to catch misconfiguration early.
        """
        errors: list[str] = []
        if not self.lm_studio_url.startswith(("http://", "https://")):
            errors.append(f"lm_studio_url must start with http:// or https://: {self.lm_studio_url!r}")
        if not (0.0 < self.diversity_threshold <= 1.0):
            errors.append(f"diversity_threshold must be in (0, 1]: {self.diversity_threshold}")
        if not (0.0 <= self.exploration_ratio <= 1.0):
            errors.append(f"exploration_ratio must be in [0, 1]: {self.exploration_ratio}")
        if self.max_knowledge_results < 1:
            errors.append(f"max_knowledge_results must be >= 1: {self.max_knowledge_results}")
        if self.consolidation_interval < 1:
            errors.append(f"consolidation_interval must be >= 1: {self.consolidation_interval}")
        if self.token_budget < 256:
            errors.append(f"token_budget seems too low (< 256): {self.token_budget}")
        if not (0.0 <= self.knowledge_min_confidence <= 1.0):
            errors.append(
                f"knowledge_min_confidence must be in [0, 1]: {self.knowledge_min_confidence}"
            )
        return errors


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_settings(
    config_path: str = _DEFAULT_CONFIG_PATH,
    env_override: bool = True,
) -> Settings:
    """
    Load Settings from a JSON file, with optional environment variable overrides.

    Resolution order (highest priority first):
    1. Environment variables (if env_override=True)
    2. JSON file at config_path (if the file exists)
    3. Dataclass defaults

    Parameters
    ----------
    config_path : str
        Path to the JSON settings file.
    env_override : bool
        If True, ATWATER_* environment variables override JSON values.

    Returns
    -------
    Settings
        Populated and validated settings instance.

    Raises
    ------
    ValueError
        If the settings fail validation. The error message lists all issues.
    """
    data: dict[str, Any] = {}

    # --- Load from JSON ---
    path = Path(config_path)
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            logger.debug("Loaded settings from %s", path)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse %s: %s — using defaults.", path, exc)
    else:
        logger.debug("Settings file not found at %s — using defaults.", path)

    # --- Apply environment variable overrides ---
    if env_override:
        _apply_env_overrides(data)

    # --- Build Settings from merged dict ---
    settings = _build_settings(data)

    # --- Validate ---
    errors = settings.validate()
    if errors:
        raise ValueError(
            f"Settings validation failed ({len(errors)} error(s)):\n"
            + "\n".join(f"  • {e}" for e in errors)
        )

    return settings


def _apply_env_overrides(data: dict[str, Any]) -> None:
    """Mutate *data* in-place with ATWATER_* env var values."""
    str_vars: dict[str, str] = {
        "ATWATER_LM_STUDIO_URL": "lm_studio_url",
        "ATWATER_MODEL_NAME": "model_name",
        "ATWATER_STATE_DB_PATH": "state_db_path",
        "ATWATER_KNOWLEDGE_DB_PATH": "knowledge_db_path",
        "ATWATER_OPTUNA_DB_PATH": "optuna_db_path",
        "ATWATER_LOG_LEVEL": "log_level",
        "ATWATER_STUDY_NAME": "study_name",
        # Also support the shorter form from HAND.toml
        "LM_STUDIO_URL": "lm_studio_url",
        "LM_STUDIO_MODEL": "model_name",
    }
    int_vars: dict[str, str] = {
        "ATWATER_CONSOLIDATION_INTERVAL": "consolidation_interval",
        "ATWATER_MAX_KNOWLEDGE_RESULTS": "max_knowledge_results",
        "ATWATER_TOKEN_BUDGET": "token_budget",
        "ATWATER_OPTUNA_SEED": "optuna_seed",
        "ATWATER_DIVERSITY_WINDOW": "diversity_window",
    }
    float_vars: dict[str, str] = {
        "ATWATER_DIVERSITY_THRESHOLD": "diversity_threshold",
        "ATWATER_EXPLORATION_RATIO": "exploration_ratio",
        "ATWATER_KNOWLEDGE_MIN_CONFIDENCE": "knowledge_min_confidence",
    }

    for env_key, field_name in str_vars.items():
        val = os.environ.get(env_key)
        if val is not None:
            data[field_name] = val

    for env_key, field_name in int_vars.items():
        val = os.environ.get(env_key)
        if val is not None:
            try:
                data[field_name] = int(val)
            except ValueError:
                logger.warning("Ignoring invalid int env var %s=%r", env_key, val)

    for env_key, field_name in float_vars.items():
        val = os.environ.get(env_key)
        if val is not None:
            try:
                data[field_name] = float(val)
            except ValueError:
                logger.warning("Ignoring invalid float env var %s=%r", env_key, val)


def _build_settings(data: dict[str, Any]) -> Settings:
    """Construct a Settings instance from a (possibly partial) dict."""
    defaults = Settings()
    defaults_dict = asdict(defaults)
    # Only pass keys that are valid Settings fields.
    valid_keys = set(defaults_dict.keys())
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return Settings(**{**defaults_dict, **filtered})
