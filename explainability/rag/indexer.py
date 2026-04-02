"""
explainability/rag/indexer.py
==============================
Build, persist, and refresh the FAISS index used by the RAG retriever.

``FAISSIndexer`` orchestrates the full index lifecycle:
  1. Load source documents (from KG cache, policy files)
  2. Embed all documents using sentence-BERT
  3. Build FAISS IndexFlatIP (inner product = cosine similarity for L2-normalised vectors)
  4. Save index + document store to disk
  5. Reload on subsequent application starts (fast path)

Usage
-----
    indexer = FAISSIndexer(
        store    = DocumentStore(),
        embedder = Embedder(),
        config   = {
            "index_path":      "data/embeddings/rag_index.faiss",
            "store_path":      "data/embeddings/rag_store.json",
            "nvd_cache":       "data/kg_cache/nvd_cves.json",
            "attck_cache":     "data/kg_cache/attck_parsed.json",
            "max_cves":        500,
            "min_cvss":        6.0,
        },
    )

    # First run — builds and saves
    n = indexer.build()

    # Subsequent runs — loads from disk (fast)
    n = indexer.load_or_build()
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from explainability.rag.document_store import DocumentStore
from explainability.rag.embedder import Embedder
from explainability.rag.retriever import RAGRetriever

logger = logging.getLogger(__name__)

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


class FAISSIndexer:
    """
    Manages the lifecycle of the FAISS RAG index.

    Parameters
    ----------
    store : DocumentStore
        Document store to populate and index.
    embedder : Embedder
        Embedder for converting document text to vectors.
    config : dict
        Index configuration.  Keys:
            index_path   : path for the FAISS binary index file
            store_path   : path for the document store JSON
            nvd_cache    : path to nvd_cves.json
            attck_cache  : path to attck_parsed.json
            max_cves     : max CVEs to load
            min_cvss     : min CVSS to load
    """

    def __init__(
        self,
        store:    DocumentStore,
        embedder: Embedder,
        config:   Optional[Dict[str, Any]] = None,
    ) -> None:
        self.store    = store
        self.embedder = embedder
        self.config   = config or {}

        self._index_path = Path(
            self.config.get("index_path", "data/embeddings/rag_index.faiss")
        )
        self._store_path = Path(
            self.config.get("store_path", "data/embeddings/rag_store.json")
        )
        self._meta_path  = self._index_path.with_suffix(".meta.json")

    # ------------------------------------------------------------------ #
    # Primary interface
    # ------------------------------------------------------------------ #

    def load_or_build(self, force_rebuild: bool = False) -> int:
        """
        Load the index from disk if it exists and is fresh; otherwise build.

        Parameters
        ----------
        force_rebuild : bool
            If True, always rebuild even if a cached index exists.

        Returns
        -------
        int
            Number of documents indexed.
        """
        if not force_rebuild and self._index_exists():
            try:
                return self.load()
            except Exception as exc:
                logger.warning("Failed to load existing index: %s — rebuilding.", exc)

        return self.build()

    def build(self) -> int:
        """
        Build the full RAG index from source documents.

        Steps:
            1. Clear the current store
            2. Load CVE documents from NVD cache
            3. Load ATT&CK technique/tactic documents
            4. Load ACD policy documents
            5. Embed all documents
            6. Build FAISS index
            7. Save index + store to disk

        Returns
        -------
        int
            Total number of documents indexed.
        """
        start = time.monotonic()
        logger.info("=== Building RAG index ===")

        self.store.clear()

        # Load document sources
        n_cves = self.store.load_from_kg_cache(
            nvd_cache_path = self.config.get("nvd_cache", "data/kg_cache/nvd_cves.json"),
            max_cves       = self.config.get("max_cves", 500),
            min_cvss       = self.config.get("min_cvss", 6.0),
        )
        n_attck = self.store.load_attck_techniques(
            attck_cache_path = self.config.get(
                "attck_cache", "data/kg_cache/attck_parsed.json"
            )
        )
        n_policy = self.store.load_acd_policies()

        total_docs = n_cves + n_attck + n_policy
        logger.info(
            "Documents loaded — CVEs: %d, ATT&CK: %d, Policies: %d, Total: %d",
            n_cves, n_attck, n_policy, total_docs,
        )

        # Embed all documents
        logger.info("Embedding %d documents...", total_docs)
        n_embedded = self.embedder.embed_store(self.store)

        # Build FAISS index
        n_indexed = self._build_faiss_index()

        # Save to disk
        self._save()

        elapsed = time.monotonic() - start
        logger.info(
            "=== RAG index built === %d docs indexed in %.1fs",
            n_indexed, elapsed,
        )
        return n_indexed

    def load(self) -> int:
        """
        Load a pre-built FAISS index from disk.

        Returns
        -------
        int
            Number of documents loaded into the index.

        Raises
        ------
        FileNotFoundError
            If the index files do not exist.
        """
        if not self._index_exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self._index_path}. "
                "Run indexer.build() first."
            )

        # Load document store
        self.store.load(str(self._store_path))

        # Re-embed any documents missing embeddings
        self.embedder.embed_store(self.store)

        # Re-build FAISS index from loaded embeddings
        n_indexed = self._build_faiss_index()

        # Optionally load raw FAISS binary (faster for large indexes)
        if _FAISS_AVAILABLE and self._index_path.exists():
            try:
                import faiss as _faiss
                raw_index = _faiss.read_index(str(self._index_path))
                logger.debug(
                    "Loaded raw FAISS index (%d vectors).", raw_index.ntotal
                )
            except Exception as exc:
                logger.warning("Could not load raw FAISS binary: %s", exc)

        logger.info("RAG index loaded — %d documents.", n_indexed)
        return n_indexed

    def build_retriever(
        self, top_k: int = 5, min_score: float = 0.0
    ) -> RAGRetriever:
        """
        Build and return a ready-to-use ``RAGRetriever`` backed by this index.

        The retriever shares the same store and embedder — no data is copied.

        Parameters
        ----------
        top_k : int
            Default top-k for the retriever.
        min_score : float
            Minimum similarity score threshold.

        Returns
        -------
        RAGRetriever
        """
        retriever = RAGRetriever(
            store     = self.store,
            embedder  = self.embedder,
            top_k     = top_k,
            min_score = min_score,
        )
        retriever.build_index()
        return retriever

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _build_faiss_index(self) -> int:
        """
        Build the FAISS IndexFlatIP from document embeddings.

        Returns number of vectors indexed.
        """
        docs = [d for d in self.store.all_documents() if d.embedding is not None]

        if not docs:
            logger.warning("No embedded documents to index.")
            return 0

        matrix = np.array([d.embedding for d in docs], dtype=np.float32)

        if _FAISS_AVAILABLE:
            index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            # Save the binary FAISS index
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                faiss.write_index(index, str(self._index_path))
                logger.debug("FAISS index saved to %s", self._index_path)
            except Exception as exc:
                logger.warning("Could not save FAISS index: %s", exc)

        return len(docs)

    def _save(self) -> None:
        """Persist document store and metadata to disk."""
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store.save(str(self._store_path))

        meta = {
            "built_at":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_docs":      self.store.n_docs,
            "store_stats": self.store.stats(),
            "embedder":    self.embedder.model_name,
            "faiss":       _FAISS_AVAILABLE,
        }
        self._meta_path.write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        logger.info("Index metadata saved to %s", self._meta_path)

    def _index_exists(self) -> bool:
        """True if both store and index files exist."""
        return self._store_path.exists()

    def get_meta(self) -> Optional[Dict[str, Any]]:
        """Return the index metadata dict, or None if not built yet."""
        if not self._meta_path.exists():
            return None
        return json.loads(self._meta_path.read_text(encoding="utf-8"))

    def __repr__(self) -> str:
        return (
            f"FAISSIndexer("
            f"index_path={self._index_path!r}, "
            f"n_docs={self.store.n_docs})"
        )