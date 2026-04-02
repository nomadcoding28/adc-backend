"""
explainability/rag/retriever.py
================================
Top-k semantic retrieval over the RAG document store.

At inference time, the retriever:
  1. Encodes the query string into an embedding vector
  2. Searches the FAISS index for the top-k most similar documents
  3. Returns those documents with their similarity scores

Two search strategies are supported:
  - Dense (FAISS):    Fast approximate nearest-neighbour via inner product
  - Filtered dense:   FAISS search restricted to a specific doc_type

Usage
-----
    retriever = RAGRetriever(store=store, embedder=embedder, top_k=5)

    results = retriever.retrieve("Log4Shell remote code execution")

    for r in results:
        print(f"[{r.score:.3f}] {r.doc_id}: {r.content[:80]}...")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from explainability.rag.document_store import Document, DocumentStore
from explainability.rag.embedder import Embedder

logger = logging.getLogger(__name__)

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    logger.warning(
        "faiss-cpu not installed — retriever will use brute-force numpy search. "
        "Install with: pip install faiss-cpu"
    )

_DEFAULT_TOP_K       = 5
_MIN_SCORE_THRESHOLD = 0.0    # Return all results above this cosine similarity


@dataclass
class RetrievalResult:
    """
    A single retrieval result from the RAG pipeline.

    Attributes
    ----------
    doc_id : str
        Document identifier.
    content : str
        Full document content.
    doc_type : str
        Document category.
    score : float
        Cosine similarity score (0–1, higher = more relevant).
    metadata : dict
        Document metadata dict.
    rank : int
        1-indexed rank in the result list.
    """
    doc_id:   str
    content:  str
    doc_type: str
    score:    float
    metadata: Dict[str, Any] = field(default_factory=dict)
    rank:     int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id":   self.doc_id,
            "doc_type": self.doc_type,
            "score":    round(self.score, 4),
            "content":  self.content[:300] + "..." if len(self.content) > 300 else self.content,
            "metadata": self.metadata,
            "rank":     self.rank,
        }

    def __str__(self) -> str:
        return f"[{self.rank}] ({self.score:.3f}) {self.doc_id}: {self.content[:60]}..."


class RAGRetriever:
    """
    Retrieves top-k relevant documents for a query using semantic search.

    Builds a FAISS flat index (or falls back to brute-force numpy search)
    over the document store embeddings.  The index is rebuilt whenever
    ``build_index()`` is called — typically at application startup.

    Parameters
    ----------
    store : DocumentStore
        Document store containing the corpus.
    embedder : Embedder
        Embedder for encoding query strings.
    top_k : int
        Default number of documents to retrieve.  Default 5.
    min_score : float
        Minimum similarity score to include in results.  Default 0.0.
    """

    def __init__(
        self,
        store:     DocumentStore,
        embedder:  Embedder,
        top_k:     int   = _DEFAULT_TOP_K,
        min_score: float = _MIN_SCORE_THRESHOLD,
    ) -> None:
        self.store     = store
        self.embedder  = embedder
        self.top_k     = top_k
        self.min_score = min_score

        # FAISS index and parallel doc-id list
        self._index:   Optional[Any]        = None   # faiss.IndexFlatIP
        self._doc_ids: List[str]            = []
        self._matrix:  Optional[np.ndarray] = None   # fallback numpy matrix

        self._index_built = False

    # ------------------------------------------------------------------ #
    # Index management
    # ------------------------------------------------------------------ #

    def build_index(self) -> int:
        """
        Build the search index over all embedded documents in the store.

        Documents without embeddings are embedded on-the-fly.

        Returns
        -------
        int
            Number of documents indexed.
        """
        # Ensure all documents are embedded
        self.embedder.embed_store(self.store)

        docs = [d for d in self.store.all_documents() if d.embedding is not None]

        if not docs:
            logger.warning("No embedded documents found — index is empty.")
            self._index_built = False
            return 0

        self._doc_ids = [d.id for d in docs]
        matrix = np.array([d.embedding for d in docs], dtype=np.float32)

        if _FAISS_AVAILABLE:
            self._index = faiss.IndexFlatIP(matrix.shape[1])
            self._index.add(matrix)
            self._matrix = None
            logger.info(
                "FAISS index built — %d docs, dim=%d",
                self._index.ntotal, matrix.shape[1],
            )
        else:
            # Brute-force fallback: store normalised matrix for dot product
            self._matrix = matrix
            self._index  = None
            logger.info(
                "Brute-force index built — %d docs, dim=%d",
                len(docs), matrix.shape[1],
            )

        self._index_built = True
        return len(docs)

    def rebuild_index(self) -> int:
        """Force a full index rebuild (call after adding new documents)."""
        self._index_built = False
        return self.build_index()

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        query:     str,
        top_k:     Optional[int] = None,
        doc_types: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve the top-k most relevant documents for a query.

        Parameters
        ----------
        query : str
            Natural language query string.
        top_k : int, optional
            Override the default top-k.
        doc_types : list[str], optional
            If given, only retrieve documents of these types.
            E.g. ``["cve", "technique"]``.

        Returns
        -------
        list[RetrievalResult]
            Ordered by descending similarity score.
        """
        k = top_k or self.top_k

        if not self._index_built:
            logger.debug("Index not built — building now...")
            self.build_index()

        if not self._doc_ids:
            logger.warning("Empty index — returning no results.")
            return []

        # Encode query
        query_vec = self.embedder.embed(query)   # shape (dim,)
        query_vec = query_vec.reshape(1, -1)     # shape (1, dim)

        # Search
        if _FAISS_AVAILABLE and self._index is not None:
            scores, indices = self._index.search(query_vec, min(k * 3, len(self._doc_ids)))
            scores   = scores[0].tolist()
            indices  = indices[0].tolist()
        else:
            # Brute-force dot product (matrix is already L2-normalised)
            raw_scores = (query_vec @ self._matrix.T)[0]
            indices    = np.argsort(raw_scores)[::-1][:k * 3].tolist()
            scores     = raw_scores[indices].tolist()

        # Build results, apply type filter, and cap at k
        results: List[RetrievalResult] = []
        rank = 1

        for idx, score in zip(indices, scores):
            if idx < 0 or idx >= len(self._doc_ids):
                continue
            if score < self.min_score:
                continue

            doc_id = self._doc_ids[idx]
            doc    = self.store.get(doc_id)
            if doc is None:
                continue

            if doc_types and doc.doc_type not in doc_types:
                continue

            results.append(RetrievalResult(
                doc_id   = doc.id,
                content  = doc.content,
                doc_type = doc.doc_type,
                score    = float(score),
                metadata = doc.metadata,
                rank     = rank,
            ))
            rank += 1

            if len(results) >= k:
                break

        return results

    def retrieve_multi_query(
        self,
        queries:   List[str],
        top_k:     Optional[int] = None,
        doc_types: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve documents for multiple query strings and deduplicate.

        Useful when the ReAct agent generates multiple sub-queries for a
        single explanation step.

        Parameters
        ----------
        queries : list[str]
            Multiple query strings.
        top_k : int, optional
            Results per query before deduplication.
        doc_types : list[str], optional
            Type filter.

        Returns
        -------
        list[RetrievalResult]
            Deduplicated results, sorted by max score across all queries.
        """
        seen_ids: Dict[str, RetrievalResult] = {}

        for query in queries:
            for result in self.retrieve(query, top_k=top_k, doc_types=doc_types):
                existing = seen_ids.get(result.doc_id)
                if existing is None or result.score > existing.score:
                    seen_ids[result.doc_id] = result

        ranked = sorted(seen_ids.values(), key=lambda r: r.score, reverse=True)
        for i, r in enumerate(ranked, start=1):
            r.rank = i

        return ranked[:top_k or self.top_k]

    # ------------------------------------------------------------------ #
    # Convenience wrappers for specific retrieval patterns
    # ------------------------------------------------------------------ #

    def retrieve_for_cve(self, cve_id: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Retrieve documents most relevant to a specific CVE.

        Combines a direct ID match with a semantic search on CVE description.
        """
        # Direct lookup first
        results: List[RetrievalResult] = []
        direct_doc = self.store.get(cve_id)
        if direct_doc:
            results.append(RetrievalResult(
                doc_id   = direct_doc.id,
                content  = direct_doc.content,
                doc_type = direct_doc.doc_type,
                score    = 1.0,
                metadata = direct_doc.metadata,
                rank     = 1,
            ))

        # Semantic search using CVE content
        semantic = self.retrieve(
            query     = f"{cve_id} vulnerability exploitation technique",
            top_k     = top_k,
            doc_types = ["technique", "tactic", "policy"],
        )
        for r in semantic:
            if not any(x.doc_id == r.doc_id for x in results):
                results.append(r)

        for i, r in enumerate(results[:top_k], start=1):
            r.rank = i

        return results[:top_k]

    def retrieve_for_action(
        self, action_type: str, host: str, threat_context: str
    ) -> List[RetrievalResult]:
        """
        Retrieve policy and technique documents relevant to a defender action.

        Parameters
        ----------
        action_type : str
            Action type string (e.g. ``"Isolate"``, ``"Remove"``, ``"DeployDecoy"``).
        host : str
            Target host (e.g. ``"Host-3"``).
        threat_context : str
            Brief description of the threat context.

        Returns
        -------
        list[RetrievalResult]
        """
        query = f"{action_type} {host} {threat_context} defender policy"
        return self.retrieve(
            query     = query,
            top_k     = self.top_k,
            doc_types = ["policy", "technique", "cve"],
        )

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #

    @property
    def is_ready(self) -> bool:
        """True if the index has been built and is non-empty."""
        return self._index_built and len(self._doc_ids) > 0

    @property
    def n_indexed(self) -> int:
        """Number of documents in the current index."""
        return len(self._doc_ids)

    def __repr__(self) -> str:
        return (
            f"RAGRetriever("
            f"n_indexed={self.n_indexed}, "
            f"top_k={self.top_k}, "
            f"faiss={'yes' if _FAISS_AVAILABLE else 'no'})"
        )