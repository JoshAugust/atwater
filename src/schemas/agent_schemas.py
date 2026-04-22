"""
src.schemas.agent_schemas — JSON Schema definitions for all Atwater agents.

Two schema families:

1. *_LLM_SCHEMA  — passed as `response_format.json_schema.schema` to the LM Studio API.
   These enforce what the *model* must output before the system even sees it.

2. *_OUTPUT_SCHEMA — used internally to validate AgentResult.output dicts
   after the agent runs (catches bugs in agent logic, not just LLM failures).

Design rules:
- All required fields are listed in "required".
- Numeric scores are constrained to [0.0, 1.0] with minimum/maximum.
- String enums are used wherever the domain is closed.
- additionalProperties: False is set on LLM schemas to prevent hallucinated fields.
- "null" alternatives are expressed as {"type": ["string", "null"]} for LM Studio compat.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Reusable sub-schemas
# ---------------------------------------------------------------------------

# A single grading dimension as the LLM should produce it.
_DIMENSION_LLM_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "score": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Dimension score from 0.0 (worst) to 1.0 (best).",
        },
        "reasoning": {
            "type": "string",
            "minLength": 10,
            "description": "Explanation for this dimension score.",
        },
    },
    "required": ["score", "reasoning"],
    "additionalProperties": False,
}

# A single grading dimension as stored in the structured output dict.
_DIMENSION_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reasoning": {"type": "string"},
    },
    "required": ["score", "reasoning"],
}


# ---------------------------------------------------------------------------
# GRADER — LLM response schema
# ---------------------------------------------------------------------------
# This is what the LLM must return when GraderEngine.execute() fires an
# evaluation prompt.  The orchestrator parses this and passes it to
# execute_score_report().

GRADER_LLM_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "originality": _DIMENSION_LLM_SCHEMA,
        "brand_alignment": _DIMENSION_LLM_SCHEMA,
        "technical_quality": _DIMENSION_LLM_SCHEMA,
        "novel_finding": {
            "type": ["string", "null"],
            "description": (
                "A concise description of a novel finding worth writing to the "
                "knowledge base. Set to null if nothing notable."
            ),
        },
        "suggest_knowledge_write": {
            "type": "boolean",
            "description": "True if novel_finding should be persisted to the knowledge base.",
        },
        "topic_cluster": {
            "type": "string",
            "minLength": 1,
            "description": "Knowledge cluster tag, e.g. 'brand_voice', 'layout', 'general'.",
        },
    },
    "required": [
        "originality",
        "brand_alignment",
        "technical_quality",
        "novel_finding",
        "suggest_knowledge_write",
        "topic_cluster",
    ],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# CREATOR self-critique — LLM response schema
# ---------------------------------------------------------------------------
# The orchestrator runs a self-critique prompt (built by CreatorAgent.execute_critique)
# through the LLM.  This schema constrains that response.

CREATOR_CRITIQUE_LLM_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "complies_with_patterns": {
            "type": "boolean",
            "description": "True if the content follows established KB rules and patterns.",
        },
        "compliance_notes": {
            "type": "string",
            "description": "Explanation of any pattern violations or compliance observations.",
        },
        "is_novel": {
            "type": "boolean",
            "description": "True if this approach is meaningfully different from prior KB entries.",
        },
        "novelty_description": {
            "type": ["string", "null"],
            "description": "Brief description of what is novel. Null if not novel.",
        },
        "quality_concerns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of quality concerns; empty list if none.",
        },
        "confidence_score": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence that this content will score well (0.0–1.0).",
        },
        "overall_critique": {
            "type": "string",
            "minLength": 20,
            "description": "One-paragraph overall critique of the generated content.",
        },
    },
    "required": [
        "complies_with_patterns",
        "compliance_notes",
        "is_novel",
        "novelty_description",
        "quality_concerns",
        "confidence_score",
        "overall_critique",
    ],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# CONSOLIDATOR synthesis — LLM response schema
# ---------------------------------------------------------------------------
# The consolidator builds a synthesis prompt and the orchestrator optionally
# runs it through the LLM.  This constrains that response.

_INSIGHT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "parameter": {
            "type": "string",
            "description": "The parameter this insight refers to.",
        },
        "importance_rank": {
            "type": "integer",
            "minimum": 1,
            "description": "Rank of this parameter by importance (1 = highest).",
        },
        "insight": {
            "type": "string",
            "minLength": 20,
            "description": "Qualitative one-sentence insight about this parameter.",
        },
        "suggested_tier_elevation": {
            "type": ["string", "null"],
            "enum": ["observation", "pattern", "rule", None],
            "description": "Suggest promoting this insight to a higher KB tier, or null.",
        },
    },
    "required": ["parameter", "importance_rank", "insight", "suggested_tier_elevation"],
    "additionalProperties": False,
}

CONSOLIDATOR_SYNTHESIS_LLM_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "insights": {
            "type": "array",
            "items": _INSIGHT_SCHEMA,
            "minItems": 0,
            "description": "List of qualitative insights, one per notable parameter.",
        },
        "surprising_findings": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of surprising or counter-intuitive findings.",
        },
        "patterns_to_elevate": {
            "type": "array",
            "items": {"type": "string"},
            "description": "KB entries or patterns that should be promoted to higher tiers.",
        },
        "overall_summary": {
            "type": "string",
            "minLength": 30,
            "description": "2-3 sentence summary of the consolidation cycle's key takeaways.",
        },
    },
    "required": [
        "insights",
        "surprising_findings",
        "patterns_to_elevate",
        "overall_summary",
    ],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# DIRECTOR — AgentResult.output schema
# ---------------------------------------------------------------------------

DIRECTOR_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "trial_number": {
            "type": "integer",
            "minimum": 0,
            "description": "Optuna trial index.",
        },
        "params": {
            "type": "object",
            "description": "The suggested parameter combination (key→value).",
        },
        "source": {
            "type": "string",
            "enum": ["optuna", "knowledge_override"],
            "description": "Whether Optuna or a KB entry drove this hypothesis.",
        },
        "override_entry": {
            "type": ["object", "null"],
            "description": "The KB entry used for override; null if source=optuna.",
        },
    },
    "required": ["trial_number", "params", "source", "override_entry"],
}

# ---------------------------------------------------------------------------
# CREATOR — AgentResult.output schemas (two phases)
# ---------------------------------------------------------------------------

CREATOR_GENERATION_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "phase": {
            "type": "string",
            "enum": ["generation"],
        },
        "prompt": {
            "type": "string",
            "minLength": 10,
        },
        "hypothesis": {
            "type": "object",
        },
        "kb_context": {
            "type": "array",
            "items": {"type": "object"},
        },
    },
    "required": ["phase", "prompt", "hypothesis", "kb_context"],
}

CREATOR_CRITIQUE_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "phase": {
            "type": "string",
            "enum": ["critique"],
        },
        "prompt": {
            "type": "string",
            "minLength": 10,
        },
        "generated_content": {
            "type": "string",
        },
    },
    "required": ["phase", "prompt", "generated_content"],
}

# ---------------------------------------------------------------------------
# GRADER — AgentResult.output schemas (two phases)
# ---------------------------------------------------------------------------

GRADER_EVALUATION_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "phase": {
            "type": "string",
            "enum": ["evaluation"],
        },
        "prompt": {
            "type": "string",
            "minLength": 10,
        },
        "output_path": {
            "type": ["string", "null"],
        },
        "rubric": {
            "type": ["object", "null"],
        },
    },
    "required": ["phase", "prompt", "output_path", "rubric"],
}

GRADER_SCORE_REPORT_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "trial_id": {
            "type": "integer",
            "minimum": 0,
        },
        "overall_score": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "dimensions": {
            "type": "object",
            "properties": {
                "originality": _DIMENSION_OUTPUT_SCHEMA,
                "brand_alignment": _DIMENSION_OUTPUT_SCHEMA,
                "technical_quality": _DIMENSION_OUTPUT_SCHEMA,
            },
            "required": ["originality", "brand_alignment", "technical_quality"],
        },
        "novel_finding": {
            "type": "string",
            "description": "Empty string if nothing notable.",
        },
        "suggest_knowledge_write": {
            "type": "boolean",
        },
        "topic_cluster": {
            "type": "string",
        },
    },
    "required": [
        "trial_id",
        "overall_score",
        "dimensions",
        "novel_finding",
        "suggest_knowledge_write",
        "topic_cluster",
    ],
}

# ---------------------------------------------------------------------------
# DIVERSITY GUARD — AgentResult.output schema
# ---------------------------------------------------------------------------

DIVERSITY_GUARD_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "flagged_assets": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Asset identifiers exceeding concentration threshold.",
        },
        "force_random": {
            "type": "boolean",
            "description": "Whether to inject a random trial this cycle.",
        },
        "coverage_ratio": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Fraction of parameter search space explored.",
        },
        "concentration_map": {
            "type": "object",
            "description": "Maps asset_id → concentration ratio (float 0-1).",
        },
        "recommendations": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "description": "Human-readable action recommendations.",
        },
        "asset_status_updates": {
            "type": "object",
            "description": "Maps asset_id → status string (healthy/flagged/deprecated).",
        },
        "total_cycles": {
            "type": "integer",
            "minimum": 0,
        },
    },
    "required": [
        "flagged_assets",
        "force_random",
        "coverage_ratio",
        "concentration_map",
        "recommendations",
        "asset_status_updates",
        "total_cycles",
    ],
}

# ---------------------------------------------------------------------------
# CONSOLIDATOR — AgentResult.output schemas (skipped vs. ran)
# ---------------------------------------------------------------------------

CONSOLIDATOR_SKIPPED_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "skipped": {
            "type": "boolean",
            "enum": [True],
        },
        "reason": {
            "type": "string",
            "enum": ["not_due"],
        },
        "current_cycle": {
            "type": "integer",
            "minimum": 0,
        },
    },
    "required": ["skipped", "reason", "current_cycle"],
}

CONSOLIDATOR_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "skipped": {
            "type": "boolean",
            "enum": [False],
        },
        "current_cycle": {
            "type": "integer",
            "minimum": 0,
        },
        "importance_entries_written": {
            "type": "integer",
            "minimum": 0,
        },
        "changelog": {
            "type": ["string", "null"],
        },
        "best_params": {
            "type": "object",
        },
        "importances": {
            "type": "object",
            "description": "Maps param_name → importance float.",
        },
        "synthesis_prompt": {
            "type": "string",
        },
    },
    "required": [
        "skipped",
        "current_cycle",
        "importance_entries_written",
        "changelog",
        "best_params",
        "importances",
        "synthesis_prompt",
    ],
}

# ---------------------------------------------------------------------------
# Schema registry — maps role/phase → LLM schema for lookup by client code
# ---------------------------------------------------------------------------

LLM_SCHEMA_REGISTRY: dict[str, dict] = {
    "grader_evaluation": GRADER_LLM_SCHEMA,
    "creator_critique": CREATOR_CRITIQUE_LLM_SCHEMA,
    "consolidator_synthesis": CONSOLIDATOR_SYNTHESIS_LLM_SCHEMA,
}

# Convenience: maps to the response_format wrapper expected by LM Studio.
def build_response_format(schema: dict, name: str = "agent_output") -> dict:
    """
    Wrap a JSON Schema dict into the response_format payload for LM Studio.

    Usage
    -----
        response_format = build_response_format(GRADER_LLM_SCHEMA, "grader_evaluation")
        result = client.chat_structured(messages, response_format=response_format)

    Parameters
    ----------
    schema : dict
        A JSON Schema dict (as defined in this module).
    name : str
        Identifier for the schema; shown in LM Studio debug logs.

    Returns
    -------
    dict
        ``{"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}``
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": schema,
            "strict": True,
        },
    }
