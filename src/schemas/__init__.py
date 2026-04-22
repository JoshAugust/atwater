"""
src.schemas — JSON Schema definitions and validation for all Atwater agents.

This package ensures every LLM response conforms to a strict contract,
preventing the 15-30% invalid-JSON failure rate seen with small models.

Usage
-----
    from src.schemas import GRADER_LLM_SCHEMA, DIRECTOR_SCHEMA
    from src.schemas.validation import validate_output

    valid, errors = validate_output(llm_response, GRADER_LLM_SCHEMA)
    if not valid:
        raise ValueError(f"Schema violations: {errors}")

Schema categories
-----------------
*_LLM_SCHEMA  — schemas for what the LLM must output (used in response_format)
*_OUTPUT_SCHEMA — schemas for the full AgentResult.output dict (internal validation)
"""

from src.schemas.agent_schemas import (
    # LLM response schemas (passed as response_format to the API)
    GRADER_LLM_SCHEMA,
    CREATOR_CRITIQUE_LLM_SCHEMA,
    CONSOLIDATOR_SYNTHESIS_LLM_SCHEMA,

    # Full agent output schemas (AgentResult.output validation)
    DIRECTOR_OUTPUT_SCHEMA,
    CREATOR_GENERATION_OUTPUT_SCHEMA,
    CREATOR_CRITIQUE_OUTPUT_SCHEMA,
    GRADER_EVALUATION_OUTPUT_SCHEMA,
    GRADER_SCORE_REPORT_OUTPUT_SCHEMA,
    DIVERSITY_GUARD_OUTPUT_SCHEMA,
    CONSOLIDATOR_OUTPUT_SCHEMA,
    CONSOLIDATOR_SKIPPED_OUTPUT_SCHEMA,

    # Schema registry — maps agent name → LLM schema for convenience
    LLM_SCHEMA_REGISTRY,
)

from src.schemas.validation import validate_output, SchemaValidationError

__all__ = [
    # LLM schemas
    "GRADER_LLM_SCHEMA",
    "CREATOR_CRITIQUE_LLM_SCHEMA",
    "CONSOLIDATOR_SYNTHESIS_LLM_SCHEMA",
    # Agent output schemas
    "DIRECTOR_OUTPUT_SCHEMA",
    "CREATOR_GENERATION_OUTPUT_SCHEMA",
    "CREATOR_CRITIQUE_OUTPUT_SCHEMA",
    "GRADER_EVALUATION_OUTPUT_SCHEMA",
    "GRADER_SCORE_REPORT_OUTPUT_SCHEMA",
    "DIVERSITY_GUARD_OUTPUT_SCHEMA",
    "CONSOLIDATOR_OUTPUT_SCHEMA",
    "CONSOLIDATOR_SKIPPED_OUTPUT_SCHEMA",
    # Registry
    "LLM_SCHEMA_REGISTRY",
    # Validation
    "validate_output",
    "SchemaValidationError",
]
