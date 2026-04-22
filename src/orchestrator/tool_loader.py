"""
tool_loader.py — Lazy tool loading with semantic group selection.

Implements the two-tier pattern from CONTEXT_TIPS.md:

    Tier 1: Tool catalog — names + one-line descriptions, no schemas.
            Shown to the orchestrator / injected as context for selection.
    Tier 2: Full schemas — loaded on demand for the selected group only.

This keeps the per-turn prompt tight. Rather than dumping 15 tool schemas
into every prompt (which tanks accuracy on 4-8B models), we:

  1. Semantically match the task description against tool group descriptions.
  2. Inject only the schemas for the matched group(s).

The sentence-transformers model is lazy-loaded on first use and attempts to
share the module-level singleton with KnowledgeBase to avoid loading the
model twice. If the knowledge base singleton isn't available, ToolLoader
loads its own instance.

Tool groups
-----------
- ``memory``   — Read/write shared state and knowledge base.
- ``creative`` — Generate content, layouts, compositions.
- ``analysis`` — Score, evaluate, compare outputs.
- ``research`` — Search external data, fetch references.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentence-transformers model name (must match knowledge_base.py to share).
# ---------------------------------------------------------------------------
_EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
_model_instance = None  # module-level singleton


def _get_model():
    """
    Return the sentence-transformer model, loading it on first call.

    Attempts to reuse the KnowledgeBase module's singleton first so both
    components share the same loaded model object.
    """
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    # Try to share with KnowledgeBase's singleton.
    try:
        from src.memory.knowledge_base import _get_model as kb_get_model
        _model_instance = kb_get_model()
        logger.debug("ToolLoader reusing KnowledgeBase embedding model.")
        return _model_instance
    except Exception:
        pass  # Fall through to loading our own.

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for ToolLoader. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    _model_instance = SentenceTransformer(_EMBEDDING_MODEL_NAME)
    logger.debug("ToolLoader loaded embedding model: %s", _EMBEDDING_MODEL_NAME)
    return _model_instance


def _embed(text: str) -> np.ndarray:
    """Embed a text string, returning a normalised float32 numpy array."""
    model = _get_model()
    vec = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype(np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two unit-norm vectors (= dot product)."""
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Tool catalog — one-line descriptions for semantic selection.
# ---------------------------------------------------------------------------

TOOL_CATALOG: dict[str, str] = {
    "memory": "Read and write shared state, knowledge base entries, and working memory",
    "creative": "Generate creative content, layouts, compositions, and visual outputs",
    "analysis": "Score outputs, evaluate quality, run rubric-based grading, compare options",
    "research": "Search external data sources, fetch references, retrieve market information",
}

# ---------------------------------------------------------------------------
# Full tool schemas — loaded only for the selected group.
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: dict[str, list[dict[str, Any]]] = {
    "memory": [
        {
            "name": "state_read",
            "description": "Read a value from shared state by key.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The state key to read.",
                    },
                },
                "required": ["key"],
            },
        },
        {
            "name": "state_write",
            "description": "Atomically write a value to shared state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The state key to write.",
                    },
                    "value": {
                        "description": "Any JSON-serialisable value.",
                    },
                    "roles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Agent roles that may read this key.",
                    },
                },
                "required": ["key", "value"],
            },
        },
        {
            "name": "knowledge_read",
            "description": "Semantic search over the knowledge base. Returns top-K entries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query.",
                    },
                    "tier": {
                        "type": "string",
                        "enum": ["rule", "pattern", "observation"],
                        "description": "Optional: restrict to this tier.",
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 5,
                        "description": "Maximum number of results to return.",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "knowledge_write",
            "description": "Write a new knowledge entry to the persistent knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The knowledge text to store.",
                    },
                    "tier": {
                        "type": "string",
                        "enum": ["rule", "pattern", "observation"],
                        "description": "Knowledge tier.",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Confidence score in [0.0, 1.0].",
                    },
                    "topic_cluster": {
                        "type": "string",
                        "description": "Topic cluster label for grouping.",
                    },
                    "optuna_evidence": {
                        "type": "object",
                        "description": "Optional statistical evidence from Optuna.",
                    },
                },
                "required": ["content", "tier", "confidence", "topic_cluster"],
            },
        },
        {
            "name": "knowledge_promote",
            "description": "Promote a knowledge entry from one tier to a higher one.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entry_id": {
                        "type": "string",
                        "description": "UUID of the entry to promote.",
                    },
                    "from_tier": {
                        "type": "string",
                        "enum": ["observation", "pattern"],
                        "description": "Current tier of the entry.",
                    },
                    "to_tier": {
                        "type": "string",
                        "enum": ["pattern", "rule"],
                        "description": "Target tier (must be higher priority).",
                    },
                    "evidence": {
                        "type": "object",
                        "description": "Promotion evidence dict (e.g. trial_count, mean_score).",
                    },
                },
                "required": ["entry_id", "from_tier", "to_tier", "evidence"],
            },
        },
    ],

    "creative": [
        {
            "name": "generate_layout",
            "description": "Generate a content layout from a parameter set.",
            "parameters": {
                "type": "object",
                "properties": {
                    "params": {
                        "type": "object",
                        "description": "Parameter dict (background, layout, shot, typography, etc.).",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "File path to write the generated output.",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["png", "jpg", "svg", "html"],
                        "default": "png",
                        "description": "Output format.",
                    },
                },
                "required": ["params", "output_path"],
            },
        },
        {
            "name": "apply_typography",
            "description": "Apply typography settings to a layout element.",
            "parameters": {
                "type": "object",
                "properties": {
                    "layout_path": {
                        "type": "string",
                        "description": "Path to the layout file to modify.",
                    },
                    "font_family": {
                        "type": "string",
                        "description": "Font family name.",
                    },
                    "font_size": {
                        "type": "integer",
                        "description": "Font size in points.",
                    },
                    "contrast_ratio": {
                        "type": "number",
                        "description": "Minimum contrast ratio (WCAG).",
                    },
                },
                "required": ["layout_path"],
            },
        },
        {
            "name": "compose_background",
            "description": "Compose a background element with optional overlay.",
            "parameters": {
                "type": "object",
                "properties": {
                    "background_type": {
                        "type": "string",
                        "enum": ["dark", "gradient", "minimal", "textured", "abstract"],
                    },
                    "opacity": {
                        "type": "number",
                        "minimum": 0.2,
                        "maximum": 1.0,
                        "description": "Background opacity.",
                    },
                    "output_path": {
                        "type": "string",
                    },
                },
                "required": ["background_type", "output_path"],
            },
        },
    ],

    "analysis": [
        {
            "name": "score_output",
            "description": "Score a generated output against a multi-dimension rubric.",
            "parameters": {
                "type": "object",
                "properties": {
                    "output_path": {
                        "type": "string",
                        "description": "Path to the generated asset to evaluate.",
                    },
                    "rubric": {
                        "type": "object",
                        "description": "Grading rubric dict mapping dimension → weight.",
                    },
                    "dimensions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Dimensions to evaluate (e.g. originality, brand_alignment).",
                    },
                },
                "required": ["output_path"],
            },
        },
        {
            "name": "compare_outputs",
            "description": "Compare two outputs side-by-side on specified metrics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path_a": {"type": "string", "description": "Path to first output."},
                    "path_b": {"type": "string", "description": "Path to second output."},
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Metrics to compare (e.g. contrast, legibility).",
                    },
                },
                "required": ["path_a", "path_b"],
            },
        },
        {
            "name": "get_param_importances",
            "description": "Query Optuna for parameter importance rankings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "study_name": {
                        "type": "string",
                        "description": "Name of the Optuna study to query.",
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 5,
                        "description": "Return top K most important parameters.",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "get_dimension_stats",
            "description": "Get mean/std/count breakdown for a categorical parameter dimension.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dimension": {
                        "type": "string",
                        "description": "Parameter name to analyse (e.g. 'background', 'layout').",
                    },
                },
                "required": ["dimension"],
            },
        },
    ],

    "research": [
        {
            "name": "search_web",
            "description": "Search the web for external reference data or trend information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string.",
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 5,
                        "description": "Maximum number of results to return.",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "fetch_reference",
            "description": "Fetch and extract content from a URL for reference analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to fetch.",
                    },
                    "extract_mode": {
                        "type": "string",
                        "enum": ["text", "markdown", "json"],
                        "default": "text",
                    },
                },
                "required": ["url"],
            },
        },
        {
            "name": "get_market_benchmarks",
            "description": "Retrieve industry performance benchmarks for a given category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Product or content category.",
                    },
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Benchmark metrics of interest.",
                    },
                },
                "required": ["category"],
            },
        },
    ],
}


# ---------------------------------------------------------------------------
# ToolLoader
# ---------------------------------------------------------------------------

class ToolLoader:
    """
    Lazy tool loading with semantic group selection.

    Maintains a two-tier structure:
    - Catalog (light): one-line descriptions for all tool groups.
    - Schemas (heavy): full JSON schemas, loaded only for the selected group.

    The sentence-transformers model is lazy-loaded on first ``select_tools()``
    call. Tool group embeddings are computed once and cached.

    Args:
        catalog: Override the default TOOL_CATALOG dict.
        schemas: Override the default TOOL_SCHEMAS dict.
    """

    def __init__(
        self,
        catalog: dict[str, str] | None = None,
        schemas: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self._catalog: dict[str, str] = catalog or TOOL_CATALOG
        self._schemas: dict[str, list[dict[str, Any]]] = schemas or TOOL_SCHEMAS

        # Lazy-loaded: catalog description embeddings.
        self._catalog_embeddings: dict[str, np.ndarray] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def catalog_summary(self) -> str:
        """
        Return a compact string listing all tool groups with descriptions.

        Suitable for injecting into a prompt to let a model pick which group
        it needs before committing to loading full schemas.
        """
        lines = ["Available tool groups:"]
        for group, desc in self._catalog.items():
            lines.append(f"  - {group}: {desc}")
        return "\n".join(lines)

    def select_tools(
        self,
        task_description: str,
        top_k: int = 1,
    ) -> list[str]:
        """
        Semantically select the best-matching tool group(s) for a task.

        Embeds ``task_description`` and ranks tool group catalog descriptions
        by cosine similarity. Returns the names of the top-K groups.

        Args:
            task_description: The agent's current task text.
            top_k: Number of tool groups to return.

        Returns:
            List of tool group names, sorted by relevance descending.
            Falls back to ["memory"] if embedding fails.
        """
        if not task_description.strip():
            return list(self._catalog.keys())[:top_k]

        try:
            catalog_embs = self._get_catalog_embeddings()
            query_emb = _embed(task_description)

            scores: dict[str, float] = {
                group: _cosine_similarity(query_emb, emb)
                for group, emb in catalog_embs.items()
            }

            ranked = sorted(scores, key=lambda g: scores[g], reverse=True)
            logger.debug(
                "Tool selection scores for task=%r: %s",
                task_description[:60],
                {g: round(scores[g], 3) for g in ranked},
            )
            return ranked[:top_k]

        except Exception as exc:
            logger.warning("Tool semantic selection failed: %s — falling back to 'memory'", exc)
            return ["memory"]

    def get_schemas(self, group_name: str) -> list[dict[str, Any]]:
        """
        Return the full tool schemas for a given group.

        Args:
            group_name: One of the keys in TOOL_CATALOG / the instance catalog.

        Returns:
            List of tool schema dicts. Empty list if the group is unknown.
        """
        schemas = self._schemas.get(group_name, [])
        if not schemas:
            logger.warning("No schemas found for tool group %r.", group_name)
        return schemas

    def get_catalog(self) -> dict[str, str]:
        """Return the tool catalog (names → one-line descriptions)."""
        return dict(self._catalog)

    def available_groups(self) -> list[str]:
        """Return all tool group names."""
        return list(self._catalog.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_catalog_embeddings(self) -> dict[str, np.ndarray]:
        """
        Compute and cache catalog description embeddings.

        Embeddings are computed once per ToolLoader instance on first call.
        """
        if self._catalog_embeddings is not None:
            return self._catalog_embeddings

        logger.debug("Computing tool catalog embeddings...")
        self._catalog_embeddings = {
            group: _embed(desc)
            for group, desc in self._catalog.items()
        }
        return self._catalog_embeddings
