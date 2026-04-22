"""
src.resilience.fallbacks — Graceful degradation when core systems are unavailable.

Classes
-------
TFIDFFallback
    Knowledge retrieval using TF-IDF (sklearn) when sentence-transformers
    cannot be loaded.  Implements the same fit/search interface so it can be
    dropped in transparently.

MockLLMFallback
    Returns deterministic, schema-valid defaults for each agent role when
    LM Studio is down.  Downstream code must not crash on these outputs.

EmbeddingFallback
    Drop-in replacement for transformer-based embeddings using TF-IDF sparse
    vectors. Compatible with the KnowledgeStore interface.
"""

from __future__ import annotations

import logging
import random
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TFIDFFallback
# ---------------------------------------------------------------------------

class TFIDFFallback:
    """
    TF-IDF-based knowledge retrieval fallback.

    Mirrors the interface of the sentence-transformers-backed search so it can
    be swapped in transparently when the embedding model is unavailable.

    Usage
    -----
        fb = TFIDFFallback()
        fb.fit(["document one", "document two", ...])
        results = fb.search("query text", k=5)
        # → [(doc_id, score), ...]
    """

    def __init__(self) -> None:
        self._vectorizer: Any = None
        self._matrix: Any = None
        self._doc_ids: list[str] = []
        self._fitted = False

    def fit(self, documents: list[str], doc_ids: list[str] | None = None) -> "TFIDFFallback":
        """
        Fit the TF-IDF vectorizer on a corpus.

        Parameters
        ----------
        documents : list[str]
            Text documents to index.
        doc_ids : list[str] | None
            Optional IDs for each document.  If None, integer indices are used.

        Returns
        -------
        self (for chaining)
        """
        if not documents:
            logger.warning("TFIDFFallback.fit(): empty document list — search will return nothing")
            self._fitted = True
            self._doc_ids = []
            return self

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for TFIDFFallback. "
                "Install it with: pip install scikit-learn"
            ) from exc

        self._vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            max_features=10_000,
            stop_words="english",
        )
        self._matrix = self._vectorizer.fit_transform(documents)
        self._doc_ids = doc_ids if doc_ids is not None else [str(i) for i in range(len(documents))]
        self._fitted = True
        logger.info(
            "TFIDFFallback: fitted on %d documents (%d features)",
            len(documents),
            self._matrix.shape[1],
        )
        return self

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """
        Search for the top-k most similar documents.

        Parameters
        ----------
        query : str
            The query string.
        k : int
            Number of results to return.

        Returns
        -------
        list of (doc_id, score) tuples, sorted highest score first.
        """
        if not self._fitted or self._matrix is None:
            logger.warning("TFIDFFallback.search(): not fitted yet — returning empty results")
            return []
        if not query.strip():
            return []

        try:
            import numpy as np  # type: ignore[import]

            query_vec = self._vectorizer.transform([query])
            # Cosine similarity via dot product (TF-IDF rows are L2-normalised by default)
            scores = (self._matrix @ query_vec.T).toarray().flatten()
            top_indices = np.argsort(scores)[::-1][:k]

            results = [
                (self._doc_ids[i], float(scores[i]))
                for i in top_indices
                if scores[i] > 0.0
            ]
            logger.debug("TFIDFFallback.search(): %d results for query=%r", len(results), query[:50])
            return results
        except Exception as exc:
            logger.error("TFIDFFallback.search() failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# MockLLMFallback
# ---------------------------------------------------------------------------

class MockLLMFallback:
    """
    Returns schema-valid default outputs for each agent role when LM Studio
    is unavailable.

    The outputs are carefully chosen to:
    - Pass downstream schema validation.
    - Not crash any consumer code.
    - Be obviously flagged as fallback data (reasoning fields say "LLM unavailable").

    Usage
    -----
        fb = MockLLMFallback(search_space={"lr": (1e-5, 1e-2), ...})

        director_out = fb.director_output(cycle_number=1)
        grader_out   = fb.grader_output(cycle_number=1)
        creator_out  = fb.creator_output(cycle_number=1)
        guard_out    = fb.diversity_guard_output(cycle_number=1)
    """

    def __init__(
        self,
        search_space: dict[str, tuple[float, float]] | None = None,
        seed: int | None = None,
    ) -> None:
        self._search_space = search_space or {
            "learning_rate": (1e-5, 1e-2),
            "batch_size": (8, 128),
            "weight_decay": (0.0, 0.1),
            "dropout": (0.0, 0.5),
        }
        self._rng = random.Random(seed)

    def director_output(self, cycle_number: int = 0) -> dict[str, Any]:
        """
        Random parameter combination from the search space.
        Suitable as a Director agent output.
        """
        params = {}
        for name, (lo, hi) in self._search_space.items():
            params[name] = round(self._rng.uniform(lo, hi), 6)

        return {
            "proposed_hypothesis": {
                "params": params,
                "reasoning": "LLM unavailable — using random fallback parameters.",
                "cycle_number": cycle_number,
                "confidence": 0.0,
                "strategy": "random_fallback",
            }
        }

    def grader_output(self, cycle_number: int = 0) -> dict[str, Any]:
        """
        Neutral 0.5 score with an explicit fallback note.
        Suitable as a Grader agent output.
        """
        return {
            "overall_score": 0.5,
            "dimensions": {
                "quality": 0.5,
                "creativity": 0.5,
                "coherence": 0.5,
            },
            "novel_finding": None,
            "suggest_knowledge_write": False,
            "reasoning": (
                f"LLM unavailable at cycle {cycle_number}. "
                "Neutral score (0.5) assigned as fallback. "
                "No knowledge write suggested."
            ),
            "fallback": True,
        }

    def creator_output(self, cycle_number: int = 0) -> dict[str, Any]:
        """
        Stub output path and critique for Creator agent fallback.
        """
        return {
            "output_path": f"/tmp/atwater/fallback_cycle_{cycle_number}.txt",
            "self_critique": (
                "LLM unavailable — no actual content generated. "
                "This is a placeholder output produced by MockLLMFallback."
            ),
            "suggest_knowledge_write": False,
            "fallback": True,
        }

    def diversity_guard_output(self, cycle_number: int = 0) -> dict[str, Any]:
        """
        No-alert diversity guard output for fallback mode.
        """
        return {
            "asset_status": {},
            "diversity_alerts": [],
            "forced_exploration": False,
            "reasoning": "LLM unavailable — diversity check skipped.",
            "fallback": True,
        }

    def consolidator_output(self, cycle_number: int = 0) -> dict[str, Any]:
        """No-op consolidator output for fallback mode."""
        return {
            "promotions": [],
            "merges": [],
            "archives": [],
            "reasoning": "LLM unavailable — consolidation skipped.",
            "fallback": True,
        }

    def output_for_role(self, role: str, cycle_number: int = 0) -> dict[str, Any]:
        """
        Dispatch to the appropriate fallback method by role name.

        Raises KeyError if the role is unknown.
        """
        dispatch: dict[str, Any] = {
            "director": self.director_output,
            "creator": self.creator_output,
            "grader": self.grader_output,
            "diversity_guard": self.diversity_guard_output,
            "consolidator": self.consolidator_output,
        }
        if role not in dispatch:
            raise KeyError(f"MockLLMFallback: unknown role {role!r}")
        return dispatch[role](cycle_number=cycle_number)


# ---------------------------------------------------------------------------
# EmbeddingFallback
# ---------------------------------------------------------------------------

class EmbeddingFallback:
    """
    Drop-in replacement for transformer-based embeddings using TF-IDF vectors.

    This class provides a compatible interface with the project's KnowledgeStore
    embedding calls, but internally uses TF-IDF sparse vectors, which are much
    lighter-weight than sentence-transformer models.

    Interface contract
    ------------------
    embed(text: str) -> list[float]
        Return a dense vector representation of ``text``.

    embed_batch(texts: list[str]) -> list[list[float]]
        Return a list of dense vectors.

    similarity(vec_a: list[float], vec_b: list[float]) -> float
        Return cosine similarity between two vectors.
    """

    def __init__(self, max_features: int = 5_000) -> None:
        self._max_features = max_features
        self._vectorizer: Any = None
        self._fitted = False
        self._vocab_size: int = 0

        logger.info(
            "EmbeddingFallback: initialised with max_features=%d "
            "(sentence-transformers replacement)",
            max_features,
        )

    def fit(self, documents: list[str]) -> "EmbeddingFallback":
        """
        Fit the internal TF-IDF model on a corpus.
        Must be called before embed() or embed_batch().
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("scikit-learn is required for EmbeddingFallback.") from exc

        self._vectorizer = TfidfVectorizer(
            max_features=self._max_features,
            sublinear_tf=True,
        )
        if documents:
            self._vectorizer.fit(documents)
            self._vocab_size = len(self._vectorizer.vocabulary_)
            self._fitted = True
            logger.info("EmbeddingFallback: fitted on %d docs, vocab=%d", len(documents), self._vocab_size)
        else:
            logger.warning("EmbeddingFallback.fit(): no documents provided")
        return self

    def embed(self, text: str) -> list[float]:
        """
        Return a dense float vector for ``text``.

        Always returns exactly ``max_features`` floats, zero-padded if the
        fitted vocabulary is smaller than max_features.

        If the model has not been fitted, returns a zero vector of length
        max_features (safe default).
        """
        if not self._fitted or self._vectorizer is None:
            return [0.0] * self._max_features
        try:
            vec = self._vectorizer.transform([text])
            raw = vec.toarray().flatten().tolist()
            # Pad to max_features if vocabulary is smaller
            if len(raw) < self._max_features:
                raw += [0.0] * (self._max_features - len(raw))
            return raw[: self._max_features]
        except Exception as exc:
            logger.error("EmbeddingFallback.embed() failed: %s", exc)
            return [0.0] * self._max_features

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Return dense vectors for a batch of texts.
        Each vector is exactly ``max_features`` floats, zero-padded if needed.
        """
        if not self._fitted or self._vectorizer is None:
            return [[0.0] * self._max_features for _ in texts]
        try:
            matrix = self._vectorizer.transform(texts)
            rows = matrix.toarray().tolist()
            result = []
            for row in rows:
                if len(row) < self._max_features:
                    row = row + [0.0] * (self._max_features - len(row))
                result.append(row[: self._max_features])
            return result
        except Exception as exc:
            logger.error("EmbeddingFallback.embed_batch() failed: %s", exc)
            return [[0.0] * self._max_features for _ in texts]

    @staticmethod
    def similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """
        Cosine similarity between two dense vectors.
        Returns 0.0 on zero-length vectors.
        """
        if not vec_a or not vec_b:
            return 0.0
        try:
            import math
            dot = sum(a * b for a, b in zip(vec_a, vec_b))
            norm_a = math.sqrt(sum(a * a for a in vec_a))
            norm_b = math.sqrt(sum(b * b for b in vec_b))
            if norm_a == 0.0 or norm_b == 0.0:
                return 0.0
            return dot / (norm_a * norm_b)
        except Exception as exc:
            logger.error("EmbeddingFallback.similarity() failed: %s", exc)
            return 0.0
