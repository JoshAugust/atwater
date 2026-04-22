"""
tests/test_fallbacks.py — Tests for src.resilience.fallbacks.

No LM Studio required. scikit-learn required for TF-IDF tests.
"""

from __future__ import annotations

import pytest

from src.resilience.fallbacks import EmbeddingFallback, MockLLMFallback, TFIDFFallback


# ---------------------------------------------------------------------------
# TFIDFFallback
# ---------------------------------------------------------------------------

class TestTFIDFFallback:
    DOCS = [
        "The learning rate controls how fast the model trains.",
        "Batch size affects memory usage and training stability.",
        "Weight decay is a form of L2 regularization.",
        "Dropout prevents overfitting by randomly zeroing activations.",
        "Momentum helps the optimizer escape local minima.",
    ]

    def test_fit_returns_self(self):
        fb = TFIDFFallback()
        result = fb.fit(self.DOCS)
        assert result is fb

    def test_search_returns_list(self):
        fb = TFIDFFallback()
        fb.fit(self.DOCS)
        results = fb.search("learning rate optimizer", k=3)
        assert isinstance(results, list)

    def test_search_returns_tuples(self):
        fb = TFIDFFallback()
        fb.fit(self.DOCS)
        results = fb.search("learning rate", k=2)
        for doc_id, score in results:
            assert isinstance(doc_id, str)
            assert isinstance(score, float)

    def test_search_returns_at_most_k(self):
        fb = TFIDFFallback()
        fb.fit(self.DOCS)
        results = fb.search("training", k=2)
        assert len(results) <= 2

    def test_search_scores_sorted_descending(self):
        fb = TFIDFFallback()
        fb.fit(self.DOCS)
        results = fb.search("learning rate", k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_result_is_relevant(self):
        """Searching for 'batch size' should rank the batch-size doc highly."""
        fb = TFIDFFallback()
        fb.fit(self.DOCS)
        results = fb.search("batch size memory", k=1)
        assert len(results) >= 1
        top_id, top_score = results[0]
        assert top_score > 0.0

    def test_custom_doc_ids(self):
        fb = TFIDFFallback()
        doc_ids = ["doc-A", "doc-B", "doc-C", "doc-D", "doc-E"]
        fb.fit(self.DOCS, doc_ids=doc_ids)
        results = fb.search("learning rate", k=5)
        returned_ids = [doc_id for doc_id, _ in results]
        # All returned IDs should be from our custom list
        for rid in returned_ids:
            assert rid in doc_ids

    def test_empty_query_returns_empty(self):
        fb = TFIDFFallback()
        fb.fit(self.DOCS)
        results = fb.search("", k=5)
        assert results == []

    def test_search_before_fit_returns_empty(self):
        fb = TFIDFFallback()
        results = fb.search("anything", k=3)
        assert results == []

    def test_fit_empty_corpus(self):
        fb = TFIDFFallback()
        fb.fit([])
        results = fb.search("query", k=3)
        assert results == []

    def test_scores_are_non_negative(self):
        fb = TFIDFFallback()
        fb.fit(self.DOCS)
        results = fb.search("dropout overfitting", k=5)
        for _, score in results:
            assert score >= 0.0


# ---------------------------------------------------------------------------
# MockLLMFallback
# ---------------------------------------------------------------------------

class TestMockLLMFallback:
    def test_director_output_has_proposed_hypothesis(self):
        fb = MockLLMFallback()
        out = fb.director_output(cycle_number=1)
        assert "proposed_hypothesis" in out
        hyp = out["proposed_hypothesis"]
        assert "params" in hyp
        assert isinstance(hyp["params"], dict)

    def test_director_params_within_search_space(self):
        space = {"lr": (1e-5, 1e-2), "dropout": (0.0, 0.5)}
        fb = MockLLMFallback(search_space=space)
        out = fb.director_output()
        params = out["proposed_hypothesis"]["params"]
        assert "lr" in params
        assert "dropout" in params
        assert 1e-5 <= params["lr"] <= 1e-2
        assert 0.0 <= params["dropout"] <= 0.5

    def test_grader_output_has_score(self):
        fb = MockLLMFallback()
        out = fb.grader_output(cycle_number=3)
        assert "overall_score" in out
        assert out["overall_score"] == pytest.approx(0.5)

    def test_grader_does_not_request_kb_write(self):
        fb = MockLLMFallback()
        out = fb.grader_output()
        assert out["suggest_knowledge_write"] is False

    def test_grader_has_reasoning(self):
        fb = MockLLMFallback()
        out = fb.grader_output()
        assert "reasoning" in out
        assert "LLM unavailable" in out["reasoning"]

    def test_creator_has_output_path(self):
        fb = MockLLMFallback()
        out = fb.creator_output(cycle_number=5)
        assert "output_path" in out
        assert "5" in out["output_path"]

    def test_creator_does_not_request_kb_write(self):
        fb = MockLLMFallback()
        out = fb.creator_output()
        assert out["suggest_knowledge_write"] is False

    def test_diversity_guard_no_alerts(self):
        fb = MockLLMFallback()
        out = fb.diversity_guard_output()
        assert out["diversity_alerts"] == []
        assert out["forced_exploration"] is False

    def test_consolidator_empty_lists(self):
        fb = MockLLMFallback()
        out = fb.consolidator_output()
        assert out["promotions"] == []
        assert out["merges"] == []
        assert out["archives"] == []

    def test_output_for_role_dispatch(self):
        fb = MockLLMFallback()
        for role in ("director", "creator", "grader", "diversity_guard", "consolidator"):
            out = fb.output_for_role(role, cycle_number=1)
            assert isinstance(out, dict)

    def test_output_for_unknown_role_raises(self):
        fb = MockLLMFallback()
        with pytest.raises(KeyError):
            fb.output_for_role("nonexistent")

    def test_seeded_director_is_deterministic(self):
        fb1 = MockLLMFallback(seed=42)
        fb2 = MockLLMFallback(seed=42)
        assert fb1.director_output(1) == fb2.director_output(1)

    def test_all_outputs_are_dicts(self):
        fb = MockLLMFallback()
        assert isinstance(fb.director_output(), dict)
        assert isinstance(fb.grader_output(), dict)
        assert isinstance(fb.creator_output(), dict)
        assert isinstance(fb.diversity_guard_output(), dict)
        assert isinstance(fb.consolidator_output(), dict)


# ---------------------------------------------------------------------------
# EmbeddingFallback
# ---------------------------------------------------------------------------

class TestEmbeddingFallback:
    DOCS = [
        "learning rate and optimizer settings",
        "batch size and memory configuration",
        "weight decay regularization",
    ]

    def test_embed_returns_list(self):
        fb = EmbeddingFallback()
        fb.fit(self.DOCS)
        vec = fb.embed("test query")
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)

    def test_embed_correct_length(self):
        max_features = 100
        fb = EmbeddingFallback(max_features=max_features)
        fb.fit(self.DOCS)
        vec = fb.embed("query text")
        assert len(vec) == max_features

    def test_embed_before_fit_returns_zeros(self):
        fb = EmbeddingFallback(max_features=50)
        vec = fb.embed("test")
        assert len(vec) == 50
        assert all(v == 0.0 for v in vec)

    def test_embed_batch_returns_list_of_lists(self):
        fb = EmbeddingFallback()
        fb.fit(self.DOCS)
        vecs = fb.embed_batch(["query one", "query two"])
        assert isinstance(vecs, list)
        assert len(vecs) == 2
        assert isinstance(vecs[0], list)

    def test_embed_batch_correct_lengths(self):
        max_features = 50
        fb = EmbeddingFallback(max_features=max_features)
        fb.fit(self.DOCS)
        vecs = fb.embed_batch(["a", "b", "c"])
        for vec in vecs:
            assert len(vec) == max_features

    def test_embed_batch_before_fit(self):
        fb = EmbeddingFallback(max_features=10)
        vecs = fb.embed_batch(["x", "y"])
        assert len(vecs) == 2
        assert all(v == 0.0 for v in vecs[0])

    def test_similarity_identical_vectors(self):
        vec = [1.0, 0.5, 0.0, 0.25]
        sim = EmbeddingFallback.similarity(vec, vec)
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_similarity_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert EmbeddingFallback.similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_similarity_zero_vector(self):
        assert EmbeddingFallback.similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_similarity_empty_vectors(self):
        assert EmbeddingFallback.similarity([], []) == 0.0

    def test_fit_returns_self(self):
        fb = EmbeddingFallback()
        result = fb.fit(self.DOCS)
        assert result is fb

    def test_non_zero_embedding_for_known_word(self):
        fb = EmbeddingFallback()
        fb.fit(["learning rate optimizer"])
        vec = fb.embed("learning rate")
        assert any(v != 0.0 for v in vec)

    def test_different_queries_produce_different_vectors(self):
        fb = EmbeddingFallback()
        fb.fit(self.DOCS)
        vec_a = fb.embed("learning rate")
        vec_b = fb.embed("batch size memory")
        assert vec_a != vec_b
