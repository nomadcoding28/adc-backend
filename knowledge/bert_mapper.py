"""
knowledge/bert_mapper.py
========================
Maps CVE descriptions to MITRE ATT&CK techniques using sentence-BERT
semantic similarity — the core intelligence of the Knowledge Graph pipeline.

Mathematical approach
---------------------
Given:
    - CVE description text  d_i
    - ATT&CK technique text t_j  (name + description)

Compute:
    e_i = BERT_encode(d_i)   ∈ R^768
    f_j = BERT_encode(t_j)   ∈ R^768

    similarity(i, j) = cosine(e_i, f_j)
                     = (e_i · f_j) / (|e_i| |f_j|)

Map CVE i to technique j* = argmax_j similarity(i, j)

Accept mapping only if similarity ≥ threshold (default 0.65).

Model used
----------
    all-MiniLM-L6-v2   (default — fast, 80MB, 768-dim)
    all-mpnet-base-v2  (higher accuracy, 420MB, 768-dim)

Both are from the sentence-transformers library.

Usage
-----
    from knowledge import BERTMapper, CVEFetcher, MITREParser

    fetcher = CVEFetcher()
    parser  = MITREParser()
    parser.load_from_file("data/kg_cache/enterprise-attack.json")

    mapper = BERTMapper(model_name="all-MiniLM-L6-v2")
    mapper.build_technique_index(parser.get_techniques())

    cves = fetcher.load_from_cache()
    results = mapper.map_cves(cves, top_k=3, threshold=0.65)

    for r in results[:5]:
        print(r.cve_id, "→", r.technique_ids, "scores:", r.scores)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from knowledge.nvd_fetcher import CVERecord
from knowledge.mitre_parser import TechniqueRecord

logger = logging.getLogger(__name__)

# Try importing sentence_transformers — optional dependency
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
    logger.warning(
        "sentence-transformers not installed. "
        "Install with: pip install sentence-transformers. "
        "BERT mapping will use random fallback embeddings."
    )

_DEFAULT_MODEL     = "all-MiniLM-L6-v2"
_DEFAULT_THRESHOLD = 0.65
_BATCH_SIZE        = 64


# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class MappingResult:
    """
    Result of mapping a single CVE to one or more ATT&CK techniques.

    Attributes
    ----------
    cve_id : str
        CVE identifier.
    cve_description : str
        The description that was encoded.
    technique_ids : list[str]
        Top-k matched technique IDs, ordered by descending similarity.
    technique_names : list[str]
        Corresponding technique names.
    scores : list[float]
        Cosine similarity scores for each matched technique.
    accepted : list[bool]
        Whether each match exceeds the acceptance threshold.
    """
    cve_id:           str
    cve_description:  str
    technique_ids:    List[str]  = field(default_factory=list)
    technique_names:  List[str]  = field(default_factory=list)
    scores:           List[float] = field(default_factory=list)
    accepted:         List[bool]  = field(default_factory=list)

    @property
    def best_technique_id(self) -> Optional[str]:
        """The top-ranked technique ID, or None if no accepted match."""
        for tid, ok in zip(self.technique_ids, self.accepted):
            if ok:
                return tid
        return None

    @property
    def best_score(self) -> float:
        """Similarity score of the top-ranked technique."""
        return self.scores[0] if self.scores else 0.0

    @property
    def accepted_techniques(self) -> List[Tuple[str, float]]:
        """List of (technique_id, score) tuples for accepted mappings only."""
        return [
            (tid, score)
            for tid, score, ok in zip(self.technique_ids, self.scores, self.accepted)
            if ok
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cve_id":           self.cve_id,
            "technique_ids":    self.technique_ids,
            "technique_names":  self.technique_names,
            "scores":           [round(s, 4) for s in self.scores],
            "accepted":         self.accepted,
            "best_technique":   self.best_technique_id,
            "best_score":       round(self.best_score, 4),
        }


# ── Mapper ───────────────────────────────────────────────────────────────────

class BERTMapper:
    """
    Semantic similarity mapper: CVE description → ATT&CK technique.

    Uses sentence-BERT to encode both CVE descriptions and ATT&CK
    technique texts into a shared embedding space, then performs
    nearest-neighbour search via cosine similarity.

    Parameters
    ----------
    model_name : str
        Sentence-transformers model name.  Default: ``"all-MiniLM-L6-v2"``.
    threshold : float
        Minimum cosine similarity to accept a mapping.  Default 0.65.
    device : str
        ``"cpu"``, ``"cuda"``, or ``"mps"``.  Default ``"cpu"``.
    batch_size : int
        Encoding batch size.  Larger = faster on GPU.  Default 64.
    """

    def __init__(
        self,
        model_name: str   = _DEFAULT_MODEL,
        threshold:  float = _DEFAULT_THRESHOLD,
        device:     str   = "cpu",
        batch_size: int   = _BATCH_SIZE,
    ) -> None:
        self.model_name = model_name
        self.threshold  = threshold
        self.device     = device
        self.batch_size = batch_size

        # Loaded after build_technique_index()
        self._technique_ids:    List[str]        = []
        self._technique_names:  List[str]        = []
        self._technique_matrix: Optional[np.ndarray] = None   # (N, D)

        # Load model
        self._model: Optional[Any] = None
        if _ST_AVAILABLE:
            logger.info("Loading BERT model: %s on %s", model_name, device)
            try:
                self._model = SentenceTransformer(model_name, device=device)
                logger.info("BERT model loaded successfully.")
            except Exception as exc:
                logger.error("Failed to load BERT model: %s", exc)
                self._model = None

    # ------------------------------------------------------------------ #
    # Index building
    # ------------------------------------------------------------------ #

    def build_technique_index(
        self,
        techniques: List[TechniqueRecord],
    ) -> "BERTMapper":
        """
        Encode all ATT&CK techniques and store as a matrix for fast lookup.

        Parameters
        ----------
        techniques : list[TechniqueRecord]
            All technique records from ``MITREParser.get_techniques()``.

        Returns
        -------
        BERTMapper
            Returns self for method chaining.
        """
        if not techniques:
            raise ValueError("techniques list is empty.")

        self._technique_ids   = [t.technique_id for t in techniques]
        self._technique_names = [t.name for t in techniques]
        texts = [t.full_text for t in techniques]

        logger.info(
            "Building technique index — %d techniques, model=%s",
            len(techniques), self.model_name,
        )

        embeddings = self._encode(texts)
        # L2-normalise for cosine similarity via dot product
        self._technique_matrix = self._l2_normalise(embeddings)

        logger.info(
            "Technique index built — matrix shape: %s",
            self._technique_matrix.shape,
        )
        return self

    def save_index(
        self, path: str = "data/embeddings/technique_index.npz"
    ) -> Path:
        """
        Save the technique embedding matrix to disk.

        Avoids re-encoding all techniques on subsequent runs (encoding
        193 techniques takes ~30s on CPU).

        Parameters
        ----------
        path : str
            Path for the .npz file.

        Returns
        -------
        Path
        """
        if self._technique_matrix is None:
            raise RuntimeError("Index not built yet. Call build_technique_index() first.")

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            str(p),
            matrix       = self._technique_matrix,
            technique_ids   = np.array(self._technique_ids),
            technique_names = np.array(self._technique_names),
        )
        logger.info("Technique index saved to: %s", p)
        return p

    def load_index(
        self, path: str = "data/embeddings/technique_index.npz"
    ) -> "BERTMapper":
        """
        Load a pre-built technique embedding matrix from disk.

        Parameters
        ----------
        path : str
            Path to a .npz file written by ``save_index()``.

        Returns
        -------
        BERTMapper
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Technique index not found: {p}. "
                "Run build_technique_index() and save_index() first."
            )

        data = np.load(str(p), allow_pickle=True)
        self._technique_matrix = data["matrix"]
        self._technique_ids    = list(data["technique_ids"])
        self._technique_names  = list(data["technique_names"])

        logger.info(
            "Technique index loaded — %d techniques, dim=%d",
            len(self._technique_ids),
            self._technique_matrix.shape[1],
        )
        return self

    # ------------------------------------------------------------------ #
    # Mapping
    # ------------------------------------------------------------------ #

    def map_cve(
        self,
        cve: CVERecord,
        top_k: int = 3,
    ) -> MappingResult:
        """
        Map a single CVE to its top-k most similar ATT&CK techniques.

        Parameters
        ----------
        cve : CVERecord
            CVE with description to encode.
        top_k : int
            Number of top matches to return.

        Returns
        -------
        MappingResult
        """
        if self._technique_matrix is None:
            raise RuntimeError(
                "Technique index not built. Call build_technique_index() first."
            )

        # Encode CVE description
        emb = self._encode([cve.description])
        emb = self._l2_normalise(emb)   # shape (1, D)

        # Cosine similarity via dot product (both sides normalised)
        scores = (emb @ self._technique_matrix.T)[0]   # shape (N,)

        # Top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores  = scores[top_indices].tolist()

        tech_ids   = [self._technique_ids[i]   for i in top_indices]
        tech_names = [self._technique_names[i] for i in top_indices]
        accepted   = [s >= self.threshold for s in top_scores]

        return MappingResult(
            cve_id          = cve.cve_id,
            cve_description = cve.description,
            technique_ids   = tech_ids,
            technique_names = tech_names,
            scores          = top_scores,
            accepted        = accepted,
        )

    def map_cves(
        self,
        cves:      List[CVERecord],
        top_k:     int = 3,
        threshold: Optional[float] = None,
    ) -> List[MappingResult]:
        """
        Map a list of CVEs to ATT&CK techniques in batch.

        Processes CVEs in batches for efficiency — encodes all descriptions
        at once rather than one at a time.

        Parameters
        ----------
        cves : list[CVERecord]
            CVE records to map.
        top_k : int
            Top-k techniques per CVE.
        threshold : float, optional
            Override the instance threshold for this call.

        Returns
        -------
        list[MappingResult]
            One result per input CVE, in the same order.
        """
        if self._technique_matrix is None:
            raise RuntimeError(
                "Technique index not built. Call build_technique_index() first."
            )

        effective_threshold = threshold if threshold is not None else self.threshold

        logger.info("Mapping %d CVEs to ATT&CK techniques (top_k=%d)...", len(cves), top_k)

        # Batch-encode all CVE descriptions
        descriptions = [c.description for c in cves]
        all_embs     = self._encode(descriptions)
        all_embs     = self._l2_normalise(all_embs)   # (M, D)

        # Similarity matrix: (M CVEs) × (N techniques)
        sim_matrix = all_embs @ self._technique_matrix.T   # (M, N)

        results: List[MappingResult] = []
        for i, cve in enumerate(cves):
            scores = sim_matrix[i]
            top_indices = np.argsort(scores)[::-1][:top_k]
            top_scores  = scores[top_indices].tolist()

            tech_ids   = [self._technique_ids[j]   for j in top_indices]
            tech_names = [self._technique_names[j] for j in top_indices]
            accepted   = [s >= effective_threshold for s in top_scores]

            results.append(MappingResult(
                cve_id          = cve.cve_id,
                cve_description = cve.description,
                technique_ids   = tech_ids,
                technique_names = tech_names,
                scores          = top_scores,
                accepted        = accepted,
            ))

        accepted_count = sum(1 for r in results if r.best_technique_id is not None)
        logger.info(
            "Mapping complete — %d/%d CVEs have at least one accepted mapping "
            "(threshold=%.2f)",
            accepted_count, len(cves), effective_threshold,
        )
        return results

    def map_text(
        self,
        text:  str,
        top_k: int = 5,
    ) -> List[Tuple[str, str, float]]:
        """
        Map an arbitrary text string to ATT&CK techniques.

        Used by the RAG retriever to find techniques relevant to a
        free-text query (e.g. from the LLM explanation pipeline).

        Parameters
        ----------
        text : str
            Free-text description of an attack or behaviour.
        top_k : int
            Number of results to return.

        Returns
        -------
        list of (technique_id, technique_name, score)
        """
        if self._technique_matrix is None:
            return []

        emb    = self._encode([text])
        emb    = self._l2_normalise(emb)
        scores = (emb @ self._technique_matrix.T)[0]

        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            (self._technique_ids[i], self._technique_names[i], float(scores[i]))
            for i in top_idx
        ]

    # ------------------------------------------------------------------ #
    # Encoding helpers
    # ------------------------------------------------------------------ #

    def _encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into embedding vectors.

        Falls back to random unit vectors if the model is unavailable
        (useful for testing without sentence-transformers installed).
        """
        if self._model is None or not _ST_AVAILABLE:
            # Reproducible random fallback (seeded by text hash)
            dim = 384   # MiniLM embedding size
            vecs = []
            for t in texts:
                rng = np.random.default_rng(abs(hash(t)) % (2**32))
                vecs.append(rng.standard_normal(dim).astype(np.float32))
            return np.array(vecs)

        return self._model.encode(
            texts,
            batch_size        = self.batch_size,
            show_progress_bar = len(texts) > 100,
            convert_to_numpy  = True,
            normalize_embeddings = False,   # we normalise manually
        )

    @staticmethod
    def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
        """L2-normalise each row of a 2D matrix."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)   # avoid division by zero
        return matrix / norms

    def __repr__(self) -> str:
        index_size = (
            len(self._technique_ids) if self._technique_matrix is not None else 0
        )
        return (
            f"BERTMapper("
            f"model={self.model_name!r}, "
            f"threshold={self.threshold}, "
            f"index_size={index_size})"
        )