"""
src.llm.client — LM Studio API client for the Atwater cognitive architecture.

Uses urllib.request (stdlib) to keep dependencies lean.

Key design decisions:
- Model auto-detection: if model=None at init, /v1/models is queried and the
  first loaded model is used. Cached after first call.
- Retry: 3 attempts with exponential backoff (1s, 2s, 4s).
- chat() → plain text completion (for generative roles).
- chat_structured() → JSON-mode completion with schema enforcement.

Schema enforcement (Phase 2B)
------------------------------
Small models (4-8B) produce invalid JSON 15-30% of the time without
constrained generation. chat_structured() now:
  1. Passes response_format containing a JSON Schema to the API.
  2. Validates the parsed JSON against the schema after parsing.
  3. Retries up to _SCHEMA_RETRY_ATTEMPTS times with an escalating prompt
     if parsing or schema validation fails.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY = 1.0  # seconds; doubles each attempt

# Extra retry attempts dedicated to schema validation failures.
_SCHEMA_RETRY_ATTEMPTS = 3


class LMStudioClientError(Exception):
    """Raised when LM Studio returns an unexpected response."""


class SchemaValidationError(LMStudioClientError):
    """Raised when the model's response fails schema validation after all retries."""

    def __init__(self, errors: list[str], raw_response: str) -> None:
        self.errors = errors
        self.raw_response = raw_response
        super().__init__(
            f"Response failed schema validation ({len(errors)} error(s)): "
            + "; ".join(errors[:5])
        )


class LMStudioClient:
    """
    Thin HTTP client for LM Studio's OpenAI-compatible API.

    Parameters
    ----------
    base_url : str
        Base URL of the LM Studio server, e.g. ``http://localhost:1234/v1``.
    model : str | None
        Model identifier string. If None, the first available model returned
        by GET /v1/models is used and cached for the lifetime of this client.
    timeout : int
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: str | None = None,
        timeout: int = 60,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._resolved_model: str | None = None  # set on first use if model=None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """
        Send a chat completion request and return the assistant's reply as text.

        Parameters
        ----------
        messages : list[dict]
            OpenAI-format message list, e.g.
            [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        temperature : float
            Sampling temperature (0.0 = deterministic).
        max_tokens : int
            Maximum tokens to generate.

        Returns
        -------
        str
            The assistant's reply text.

        Raises
        ------
        LMStudioClientError
            On non-recoverable HTTP errors or malformed responses.
        """
        payload = self._build_payload(messages, temperature, max_tokens)
        response = self._post_with_retry("/chat/completions", payload)
        return self._extract_text(response)

    def chat_structured(
        self,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None = None,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """
        Send a structured (JSON-mode) chat completion and return a validated dict.

        Schema enforcement flow
        -----------------------
        1. Build the response_format payload (wrapping ``schema`` if provided).
        2. POST to the API with the schema in the payload so LM Studio uses
           constrained generation.
        3. Parse the JSON response.
        4. Validate against ``schema`` using the internal validator.
        5. On parse or validation failure: retry up to _SCHEMA_RETRY_ATTEMPTS
           times, prepending a stricter instruction to the messages.

        Parameters
        ----------
        messages : list[dict]
            OpenAI-format message list.
        response_format : dict | None
            Full OpenAI response_format spec.  If provided, used as-is.
            Mutually exclusive with ``schema``.
        schema : dict | None
            A JSON Schema dict.  The method wraps it into the correct
            response_format payload automatically.  Use this for convenience.
        temperature : float
            Lower temperature recommended for reliable JSON output.
        max_tokens : int
            Maximum tokens to generate.

        Returns
        -------
        dict
            Parsed and schema-validated JSON from the model's reply.

        Raises
        ------
        LMStudioClientError
            On HTTP errors or if the model returns non-JSON content after retries.
        SchemaValidationError
            If all retries are exhausted and the response still fails schema
            validation.
        """
        if schema is not None and response_format is not None:
            raise ValueError(
                "Provide either 'schema' or 'response_format', not both."
            )

        # Build the response_format payload.
        resolved_format: dict[str, Any] | None = None
        if schema is not None:
            resolved_format = self._build_json_schema_format(schema)
        elif response_format is not None:
            resolved_format = response_format
        else:
            # Fallback: plain JSON object mode (no schema enforcement).
            resolved_format = {"type": "json_object"}

        # Attempt structured call with schema-retry loop.
        last_parse_error: Exception | None = None
        last_validation_errors: list[str] = []
        last_raw: str = ""
        active_messages = list(messages)

        for attempt in range(_SCHEMA_RETRY_ATTEMPTS):
            if attempt > 0:
                # Escalate with a stricter prepended instruction.
                schema_text = json.dumps(schema or {}, indent=2)
                strict_instruction: dict[str, str] = {
                    "role": "user",
                    "content": (
                        f"IMPORTANT: You MUST respond with valid JSON that exactly matches "
                        f"this schema. No extra text, no markdown fences, no explanations — "
                        f"only the JSON object.\n\nRequired schema:\n{schema_text}"
                    ),
                }
                # Insert at the front (after any system message) so it's prominent.
                insert_pos = 1 if active_messages and active_messages[0]["role"] == "system" else 0
                active_messages = (
                    active_messages[:insert_pos]
                    + [strict_instruction]
                    + active_messages[insert_pos:]
                )
                logger.warning(
                    "chat_structured: schema retry %d/%d (prev errors: %s)",
                    attempt + 1,
                    _SCHEMA_RETRY_ATTEMPTS,
                    last_validation_errors[:3] or str(last_parse_error),
                )

            payload = self._build_payload(active_messages, temperature, max_tokens)
            if resolved_format:
                payload["response_format"] = resolved_format

            try:
                response = self._post_with_retry("/chat/completions", payload)
            except LMStudioClientError:
                raise  # HTTP errors are not schema errors; re-raise immediately.

            last_raw = self._extract_text(response)

            # --- Parse JSON ---
            try:
                parsed = json.loads(last_raw)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "chat_structured attempt %d: JSON parse failed — %s | raw=%s",
                    attempt + 1,
                    exc,
                    last_raw[:200],
                )
                last_parse_error = exc
                continue  # Retry

            # --- Schema validation ---
            if schema is not None:
                valid, validation_errors = self._validate_against_schema(parsed, schema)
                if not valid:
                    last_validation_errors = validation_errors
                    logger.warning(
                        "chat_structured attempt %d: schema validation failed — %s",
                        attempt + 1,
                        validation_errors[:3],
                    )
                    continue  # Retry

            # Success
            logger.debug(
                "chat_structured: success on attempt %d/%d",
                attempt + 1,
                _SCHEMA_RETRY_ATTEMPTS,
            )
            return parsed

        # All retries exhausted.
        if last_parse_error is not None:
            raise LMStudioClientError(
                f"Model returned non-JSON output after {_SCHEMA_RETRY_ATTEMPTS} attempts: "
                f"{last_raw[:300]}"
            ) from last_parse_error

        raise SchemaValidationError(
            errors=last_validation_errors,
            raw_response=last_raw,
        )

    def list_models(self) -> list[str]:
        """
        Return a list of model IDs currently loaded in LM Studio.

        Returns
        -------
        list[str]
            Loaded model IDs, or an empty list if none are loaded.
        """
        url = f"{self._base_url}/models"
        raw = self._get(url)
        data = json.loads(raw)
        return [m["id"] for m in data.get("data", [])]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_against_schema(
        data: Any, schema: dict[str, Any]
    ) -> tuple[bool, list[str]]:
        """
        Validate ``data`` against ``schema`` using the internal validator.

        Deliberately lazy-imports to avoid circular imports at module load time,
        and to remain functional even if the schemas package is missing (falls
        back to no-op validation with a warning).
        """
        try:
            from src.schemas.validation import validate_output  # type: ignore[import]
            return validate_output(data, schema)
        except ImportError:
            logger.warning(
                "src.schemas.validation not found — schema validation skipped. "
                "Install the schemas package to enable enforcement."
            )
            return True, []

    @staticmethod
    def _build_json_schema_format(schema: dict[str, Any], name: str = "agent_output") -> dict[str, Any]:
        """
        Wrap a JSON Schema dict into the LM Studio response_format payload.

        Format expected by LM Studio / OpenAI structured outputs:
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "<identifier>",
                    "schema": { ... },
                    "strict": true
                }
            }
        """
        return {
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "schema": schema,
                "strict": True,
            },
        }

    def _resolve_model(self) -> str:
        """Return the configured model, auto-detecting if needed."""
        if self._model:
            return self._model
        if self._resolved_model:
            return self._resolved_model
        # Auto-detect: use first loaded model.
        models = self.list_models()
        if not models:
            raise LMStudioClientError(
                "No models are loaded in LM Studio. "
                "Please load a model before running Atwater."
            )
        self._resolved_model = models[0]
        logger.info("LMStudioClient: auto-detected model %r", self._resolved_model)
        return self._resolved_model

    def _build_payload(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        return {
            "model": self._resolve_model(),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    def _post_with_retry(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST to ``path`` with retry logic. Returns parsed JSON response."""
        url = f"{self._base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        last_exc: Exception | None = None

        for attempt in range(_RETRY_ATTEMPTS):
            try:
                req = urllib.request.Request(
                    url,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    raw = resp.read().decode("utf-8")
                    return json.loads(raw)
            except urllib.error.HTTPError as exc:
                status = exc.code
                body_text = exc.read().decode("utf-8", errors="replace")[:300]
                logger.warning(
                    "LM Studio HTTP %d on attempt %d/%d: %s",
                    status, attempt + 1, _RETRY_ATTEMPTS, body_text,
                )
                last_exc = exc
                # 4xx errors are not retryable (except 429).
                if status != 429 and 400 <= status < 500:
                    raise LMStudioClientError(
                        f"LM Studio returned HTTP {status}: {body_text}"
                    ) from exc
            except urllib.error.URLError as exc:
                logger.warning(
                    "LM Studio connection error on attempt %d/%d: %s",
                    attempt + 1, _RETRY_ATTEMPTS, exc,
                )
                last_exc = exc
            except json.JSONDecodeError as exc:
                raise LMStudioClientError("LM Studio returned non-JSON response.") from exc

            if attempt < _RETRY_ATTEMPTS - 1:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.debug("Retrying in %.1fs…", delay)
                time.sleep(delay)

        raise LMStudioClientError(
            f"LM Studio request failed after {_RETRY_ATTEMPTS} attempts."
        ) from last_exc

    def _get(self, url: str) -> str:
        """Simple GET request, no retry needed (used for model listing)."""
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return resp.read().decode("utf-8")
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            raise LMStudioClientError(f"Failed to GET {url}: {exc}") from exc

    @staticmethod
    def _extract_text(response: dict[str, Any]) -> str:
        """Pull the assistant message text out of a chat completion response."""
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LMStudioClientError(
                f"Unexpected response shape: {json.dumps(response)[:300]}"
            ) from exc
