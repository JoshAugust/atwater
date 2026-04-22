"""
src.llm.client — LM Studio API client for the Atwater cognitive architecture.

Uses urllib.request (stdlib) to keep dependencies lean.

Key design decisions:
- Model auto-detection: if model=None at init, /v1/models is queried and the
  first loaded model is used. Cached after first call.
- Retry: 3 attempts with exponential backoff (1s, 2s, 4s).
- chat() → plain text completion (for generative roles).
- chat_structured() → JSON-mode completion (for grader scoring, etc.).
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


class LMStudioClientError(Exception):
    """Raised when LM Studio returns an unexpected response."""


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
        response_format: dict[str, Any],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """
        Send a structured (JSON-mode) chat completion and return parsed dict.

        Parameters
        ----------
        messages : list[dict]
            OpenAI-format message list.
        response_format : dict
            OpenAI response_format spec, e.g. ``{"type": "json_object"}``.
        temperature : float
            Lower temperature recommended for reliable JSON output.
        max_tokens : int
            Maximum tokens to generate.

        Returns
        -------
        dict
            Parsed JSON from the model's reply.

        Raises
        ------
        LMStudioClientError
            On HTTP errors or if the model returns non-JSON content.
        """
        payload = self._build_payload(messages, temperature, max_tokens)
        payload["response_format"] = response_format
        response = self._post_with_retry("/chat/completions", payload)
        raw_text = self._extract_text(response)
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise LMStudioClientError(
                f"Model returned non-JSON output: {raw_text[:200]}"
            ) from exc

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
