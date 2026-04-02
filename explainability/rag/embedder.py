"""
explainability/rag/embedder.py
===============================
Text → embedding vector using sentence-BERT.

Shared embedding utility used by both the RAG pipeline (for document
indexing and query encoding) and the BERT mapper in the knowledge pipeline.

The embedder is designed to be stateless and reusable — one instance
handles both the offline document embedding pass (at index build time)
and the online query embedding (at inference time).

Usage
-----
    embedder = Embedder(model_name="all-MiniLM-L6-v2", device="cpu")

    # Single text
    vec = embedder.embed("Apache Log4j2 JNDI injection")
    # vec.shape → (384,)

    # Batch
    vecs = embedder.embed_batch(["text1", "text2", "text3"])
    # vecs.shape → (3, 384)

    # Embed all documents in a store (in-place — sets doc.embedding)
    embedder.embed_store(document_store)
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
    logger.warning(
        "sentence-transformers not installed — "
        "Embedder will use deterministic random fallback vectors."
    )

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_EMBEDDING_DIM = 384     # MiniLM output dimension
_BATCH_SIZE    = 64


class Embedder:
    """
    Wraps a sentence-transformers model for text encoding.

    All outputs are L2-normalised float32 numpy arrays, suitable for
    cosine similarity computation via dot product.

    Parameters
    ----------
    model_name : str
        Sentence-transformers model identifier.
        Default: ``"all-MiniLM-L6-v2"`` (fast, 80 MB, 384-dim).
    device : str
        ``"cpu"``, ``"cuda"``, or ``"mps"``.  Default ``"cpu"``.
    batch_size : int
        Encoding batch size.  Larger = faster on GPU.
    normalize : bool
        If True (default), L2-normalise all output vectors.
    """

    def __init__(
        self,
        model_name: str  = _DEFAULT_MODEL,
        device:     str  = "cpu",
        batch_size: int  = _BATCH_SIZE,
        normalize:  bool = True,
    ) -> None:
        self.model_name = model_name
        self.device     = device
        self.batch_size = batch_size
        self.normalize  = normalize

        self._model: Optional[Any] = None
        self._dim:   int           = _EMBEDDING_DIM

        if _ST_AVAILABLE:
            try:
                logger.info(
                    "Loading sentence-transformer: %s on %s", model_name, device
                )
                self._model = SentenceTransformer(model_name, device=device)
                # Probe actual embedding dimension
                probe = self._model.encode(["probe"], convert_to_numpy=True)
                self._dim = probe.shape[1]
                logger.info("Embedder ready — dim=%d", self._dim)
            except Exception as exc:
                logger.error("Failed to load embedder model: %s", exc)
                self._model = None

    # ------------------------------------------------------------------ #
    # Primary interface
    # ------------------------------------------------------------------ #

    @property
    def dim(self) -> int:
        """Output embedding dimension."""
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        """
        Encode a single string into an embedding vector.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        np.ndarray
            Shape (dim,), dtype float32.
        """
        result = self.embed_batch([text])
        return result[0]

    def embed_batch(
        self,
        texts:              List[str],
        show_progress_bar:  bool = False,
    ) -> np.ndarray:
        """
        Encode a list of strings into a matrix of embedding vectors.

        Parameters
        ----------
        texts : list[str]
            Input texts.
        show_progress_bar : bool
            Show tqdm progress bar during encoding (useful for large batches).

        Returns
        -------
        np.ndarray
            Shape (len(texts), dim), dtype float32.
            Rows are L2-normalised if ``self.normalize`` is True.
        """
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)

        if self._model is None or not _ST_AVAILABLE:
            return self._fallback_embed(texts)

        try:
            vecs = self._model.encode(
                texts,
                batch_size           = self.batch_size,
                show_progress_bar    = show_progress_bar and len(texts) > 100,
                convert_to_numpy     = True,
                normalize_embeddings = False,  # we normalise manually below
            ).astype(np.float32)
        except Exception as exc:
            logger.warning("Embedding failed: %s — using fallback.", exc)
            return self._fallback_embed(texts)

        if self.normalize:
            vecs = self._l2_normalise(vecs)

        return vecs

    def embed_store(self, store: Any) -> int:
        """
        Embed all documents in a ``DocumentStore`` in-place.

        Sets ``doc.embedding`` on each document that doesn't already
        have one.  Skips documents with existing embeddings.

        Parameters
        ----------
        store : DocumentStore
            The document store to embed.

        Returns
        -------
        int
            Number of documents embedded.
        """
        from explainability.rag.document_store import Document

        docs_to_embed = [
            doc for doc in store.all_documents()
            if doc.embedding is None
        ]

        if not docs_to_embed:
            logger.debug("All documents already have embeddings — skipping.")
            return 0

        logger.info("Embedding %d documents...", len(docs_to_embed))
        texts = [doc.content for doc in docs_to_embed]
        vecs  = self.embed_batch(texts, show_progress_bar=len(texts) > 50)

        for doc, vec in zip(docs_to_embed, vecs):
            doc.embedding = vec.tolist()

        logger.info("Embedded %d documents (dim=%d).", len(docs_to_embed), self._dim)
        return len(docs_to_embed)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _fallback_embed(self, texts: List[str]) -> np.ndarray:
        """
        Deterministic random embedding fallback when model is unavailable.

        Uses the hash of each text as a seed so the same text always maps
        to the same random vector — consistent enough for testing.
        """
        vecs = []
        for text in texts:
            seed = abs(hash(text[:200])) % (2 ** 32)
            rng  = np.random.default_rng(seed)
            vec  = rng.standard_normal(self._dim).astype(np.float32)
            vecs.append(vec)

        result = np.array(vecs, dtype=np.float32)
        if self.normalize:
            result = self._l2_normalise(result)
        return result

    @staticmethod
    def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
        """L2-normalise each row of a matrix."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return matrix / norms

    def __repr__(self) -> str:
        return (
            f"Embedder("
            f"model={self.model_name!r}, "
            f"dim={self._dim}, "
            f"device={self.device!r}, "
            f"available={self._model is not None})"
        )