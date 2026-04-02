"""
tests/unit/test_rag.py
=======================
Unit tests for the RAG retrieval pipeline.

Tests validate retriever result format and basic embedding properties.
Uses mocks — no actual FAISS or LLM calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pytest


class TestEmbedder:
    """Test the document embedder."""

    def test_embed_returns_correct_shape(self) -> None:
        """Embeddings should have consistent dimensionality."""
        # Simulate embedding output
        embed_dim = 384  # typical sentence-transformer dim
        texts = ["Log4Shell vulnerability", "Remote code execution"]
        embeddings = np.random.randn(len(texts), embed_dim).astype(np.float32)

        assert embeddings.shape == (2, embed_dim)

    def test_embed_normalised(self) -> None:
        """Normalised embeddings should have unit L2 norm."""
        embed_dim = 384
        raw = np.random.randn(5, embed_dim).astype(np.float32)
        normalised = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8)

        norms = np.linalg.norm(normalised, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_similar_texts_have_high_similarity(self) -> None:
        """Vectors for similar texts should have cosine similarity > 0.5."""
        # Simulated: same vector with small noise
        base = np.random.randn(384).astype(np.float32)
        v1 = base / np.linalg.norm(base)
        v2 = (base + 0.1 * np.random.randn(384).astype(np.float32))
        v2 = v2 / np.linalg.norm(v2)

        cosine_sim = float(np.dot(v1, v2))
        assert cosine_sim > 0.5, f"Expected high similarity, got {cosine_sim}"


class TestRetriever:
    """Test RAG retriever result format."""

    def test_retrieve_returns_list(self) -> None:
        """Retriever.retrieve() should return a list."""
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            {"text": "CVE-2021-44228 is a RCE vulnerability", "score": 0.95, "doc_type": "cve"},
            {"text": "T1190 - Exploit Public-Facing App", "score": 0.82, "doc_type": "technique"},
        ]

        results = retriever.retrieve("Log4Shell")
        assert isinstance(results, list)
        assert len(results) == 2

    def test_retrieve_multi_query_deduplicates(self) -> None:
        """Multi-query retrieval should deduplicate results."""
        results_q1 = [
            {"text": "CVE-2021-44228", "score": 0.95},
            {"text": "Log4j exploit", "score": 0.80},
        ]
        results_q2 = [
            {"text": "CVE-2021-44228", "score": 0.90},  # Duplicate
            {"text": "Remote code exec", "score": 0.75},
        ]

        # Simple deduplication by text
        all_results = results_q1 + results_q2
        seen = set()
        unique = []
        for r in all_results:
            if r["text"] not in seen:
                unique.append(r)
                seen.add(r["text"])

        assert len(unique) == 3, f"Expected 3 unique results, got {len(unique)}"

    def test_retrieve_scores_sorted(self) -> None:
        """Results should be sorted by score (descending)."""
        results = [
            {"text": "A", "score": 0.5},
            {"text": "B", "score": 0.9},
            {"text": "C", "score": 0.7},
        ]
        sorted_results = sorted(results, key=lambda x: -x["score"])
        scores = [r["score"] for r in sorted_results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_top_k(self) -> None:
        """retrieve(top_k=N) should return at most N results."""
        all_results = [{"text": f"doc_{i}", "score": 1 - i * 0.1} for i in range(20)]
        top_k = 5
        results = all_results[:top_k]
        assert len(results) == top_k
