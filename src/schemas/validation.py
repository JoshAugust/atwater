"""
src.schemas.validation — Zero-dependency JSON Schema validator for Atwater.

Implements a subset of JSON Schema draft-07 sufficient to validate all
agent output schemas defined in agent_schemas.py.

Supported keywords
------------------
- type (string, number, integer, boolean, object, array, null + arrays of types)
- properties
- required
- additionalProperties (boolean)
- minimum / maximum (for number/integer)
- minLength / maxLength (for string)
- minItems / maxItems (for array)
- enum (exact value matching)
- items (for array — validates each element against the items sub-schema)
- oneOf / anyOf (validates that exactly one / at least one sub-schema matches)

NOT implemented (rarely needed for our schemas):
- $ref, $defs, if/then/else, format, pattern, unevaluatedProperties

Usage
-----
    from src.schemas.validation import validate_output, SchemaValidationError

    valid, errors = validate_output(data, GRADER_LLM_SCHEMA)
    if not valid:
        raise SchemaValidationError(errors)
"""

from __future__ import annotations

from typing import Any


class SchemaValidationError(Exception):
    """Raised when validate_output returns errors and the caller wants to raise."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(f"Schema validation failed ({len(errors)} error(s)): " + "; ".join(errors))


# ---------------------------------------------------------------------------
# Type coercion map — Python types that correspond to JSON Schema type names.
# ---------------------------------------------------------------------------

_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "number": (int, float),  # JSON number includes both; we allow Python int too
    "integer": int,
    "boolean": bool,
    "object": dict,
    "array": list,
    "null": type(None),
}

# Number types for numeric keyword checks
_NUMBER_TYPES = (int, float)


def validate_output(data: Any, schema: dict) -> tuple[bool, list[str]]:
    """
    Validate ``data`` against ``schema``.

    Parameters
    ----------
    data : Any
        The value to validate.
    schema : dict
        A JSON Schema dict (as defined in agent_schemas.py).

    Returns
    -------
    (valid, errors) : tuple[bool, list[str]]
        valid — True if no violations found.
        errors — List of human-readable error strings (empty when valid).
    """
    errors: list[str] = []
    _validate_value(data, schema, path="$", errors=errors)
    return (len(errors) == 0), errors


def _validate_value(
    value: Any,
    schema: dict,
    path: str,
    errors: list[str],
) -> None:
    """Recursively validate ``value`` against ``schema``, appending to ``errors``."""

    # --- type ---
    if "type" in schema:
        _check_type(value, schema["type"], path, errors)

    # --- enum ---
    if "enum" in schema:
        _check_enum(value, schema["enum"], path, errors)

    # --- numeric constraints ---
    if isinstance(value, _NUMBER_TYPES) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(
                f"{path}: value {value} is below minimum {schema['minimum']}"
            )
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(
                f"{path}: value {value} exceeds maximum {schema['maximum']}"
            )

    # --- string constraints ---
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(
                f"{path}: string length {len(value)} is below minLength {schema['minLength']}"
            )
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(
                f"{path}: string length {len(value)} exceeds maxLength {schema['maxLength']}"
            )

    # --- object ---
    if isinstance(value, dict):
        _validate_object(value, schema, path, errors)

    # --- array ---
    if isinstance(value, list):
        _validate_array(value, schema, path, errors)

    # --- oneOf ---
    if "oneOf" in schema:
        _check_one_of(value, schema["oneOf"], path, errors)

    # --- anyOf ---
    if "anyOf" in schema:
        _check_any_of(value, schema["anyOf"], path, errors)


def _check_type(value: Any, type_spec: Any, path: str, errors: list[str]) -> None:
    """Validate that ``value`` matches the type(s) named in ``type_spec``."""
    if isinstance(type_spec, list):
        allowed_types = type_spec
    else:
        allowed_types = [type_spec]

    # Build Python type(s) to check against.
    python_types: list[type] = []
    for t in allowed_types:
        if t == "null":
            python_types.append(type(None))
        elif t in _TYPE_MAP:
            mapped = _TYPE_MAP[t]
            if isinstance(mapped, tuple):
                python_types.extend(mapped)
            else:
                python_types.append(mapped)
        # Unknown type names are silently ignored (forward compat).

    if not python_types:
        return  # Nothing to check

    # Special: JSON Schema "integer" must not allow float, but "number" allows both.
    if "integer" in allowed_types and "number" not in allowed_types:
        # Strict integer check — float values like 1.0 are NOT integers in JSON Schema.
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(
                f"{path}: expected integer, got {type(value).__name__} ({value!r})"
            )
        return

    # Special: boolean must NOT match as int (Python bool is subclass of int).
    if "boolean" in allowed_types and "integer" not in allowed_types and "number" not in allowed_types:
        if not isinstance(value, bool):
            errors.append(
                f"{path}: expected boolean, got {type(value).__name__} ({value!r})"
            )
        return

    # General check — but exclude bool from matching number/integer accidentally.
    if isinstance(value, bool) and "boolean" not in allowed_types:
        errors.append(
            f"{path}: expected {type_spec}, got bool ({value!r})"
        )
        return

    if not isinstance(value, tuple(python_types)):
        errors.append(
            f"{path}: expected type {type_spec!r}, got {type(value).__name__} ({value!r})"
        )


def _check_enum(value: Any, allowed: list[Any], path: str, errors: list[str]) -> None:
    """Validate that ``value`` is one of the allowed enum values (None included)."""
    if value not in allowed:
        errors.append(
            f"{path}: value {value!r} is not one of {allowed}"
        )


def _validate_object(value: dict, schema: dict, path: str, errors: list[str]) -> None:
    """Validate object properties, required fields, and additionalProperties."""

    properties: dict[str, dict] = schema.get("properties", {})
    required: list[str] = schema.get("required", [])
    additional_props: bool | dict | None = schema.get("additionalProperties", None)

    # Check required fields
    for field in required:
        if field not in value:
            errors.append(f"{path}: missing required field '{field}'")

    # Validate known properties
    for field, sub_schema in properties.items():
        if field in value:
            _validate_value(value[field], sub_schema, path=f"{path}.{field}", errors=errors)

    # additionalProperties: False — flag unknown keys
    if additional_props is False:
        known_keys = set(properties.keys())
        for key in value:
            if key not in known_keys:
                errors.append(f"{path}: unexpected additional property '{key}'")


def _validate_array(value: list, schema: dict, path: str, errors: list[str]) -> None:
    """Validate array length constraints and per-item schemas."""

    min_items: int | None = schema.get("minItems")
    max_items: int | None = schema.get("maxItems")
    items_schema: dict | None = schema.get("items")

    if min_items is not None and len(value) < min_items:
        errors.append(
            f"{path}: array length {len(value)} is below minItems {min_items}"
        )
    if max_items is not None and len(value) > max_items:
        errors.append(
            f"{path}: array length {len(value)} exceeds maxItems {max_items}"
        )

    if items_schema is not None:
        for i, item in enumerate(value):
            _validate_value(item, items_schema, path=f"{path}[{i}]", errors=errors)


def _check_one_of(value: Any, sub_schemas: list[dict], path: str, errors: list[str]) -> None:
    """Validate exactly one sub-schema matches (oneOf)."""
    matching = 0
    for sub in sub_schemas:
        sub_errors: list[str] = []
        _validate_value(value, sub, path=path, errors=sub_errors)
        if not sub_errors:
            matching += 1
    if matching != 1:
        errors.append(
            f"{path}: oneOf requires exactly 1 matching schema, found {matching} for value {value!r}"
        )


def _check_any_of(value: Any, sub_schemas: list[dict], path: str, errors: list[str]) -> None:
    """Validate at least one sub-schema matches (anyOf)."""
    for sub in sub_schemas:
        sub_errors: list[str] = []
        _validate_value(value, sub, path=path, errors=sub_errors)
        if not sub_errors:
            return  # At least one matches — pass
    errors.append(
        f"{path}: anyOf requires at least 1 matching schema, none matched for value {value!r}"
    )
