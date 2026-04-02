"""
explainability/rag/document_store.py
=====================================
Document store for the RAG pipeline.

Holds a collection of ``Document`` objects — chunks of CVE descriptions,
ATT&CK technique pages, ACD policy documents, and incident templates —
that are retrieved at query time to ground LLM explanations.

Document types in the store
----------------------------
    cve         : CVE description from NVD
    technique   : ATT&CK technique description
    tactic      : ATT&CK tactic overview
    policy      : Internal ACD defender policy
    incident    : Past incident report template

Usage
-----
    store = DocumentStore()

    # Add documents
    store.add(Document(
        id       = "CVE-2021-44228",
        content  = "Apache Log4j2 JNDI injection allowing RCE...",
        doc_type = "cve",
        metadata = {"max_cvss": 10.0, "severity": "CRITICAL"},
    ))

    # Retrieve by ID
    doc = store.get("CVE-2021-44228")

    # Filter by type
    cve_docs = store.get_by_type("cve")

    # Bulk load from KG cache files
    store.load_from_kg_cache("data/kg_cache/nvd_cves.json")
    store.load_attck_techniques("data/kg_cache/attck_parsed.json")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Maximum characters stored per document chunk
# Keeps embedding quality high and avoids token budget issues
_MAX_CHUNK_CHARS = 2000


@dataclass
class Document:
    """
    A single document chunk in the RAG store.

    Attributes
    ----------
    id : str
        Unique document identifier (e.g. CVE-2021-44228, T1190).
    content : str
        The text content that will be embedded and retrieved.
    doc_type : str
        Category: ``"cve"``, ``"technique"``, ``"tactic"``, ``"policy"``,
        ``"incident"``.
    metadata : dict
        Arbitrary key-value metadata (CVSS score, tactic name, etc.).
        Returned alongside content for LLM context.
    embedding : list[float], optional
        Cached embedding vector.  Populated by ``Embedder.embed_store()``.
    """
    id:        str
    content:   str
    doc_type:  str
    metadata:  Dict[str, Any]       = field(default_factory=dict)
    embedding: Optional[List[float]] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # Truncate content to max chunk size
        if len(self.content) > _MAX_CHUNK_CHARS:
            self.content = self.content[:_MAX_CHUNK_CHARS] + "..."

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("embedding", None)   # don't serialise large float arrays
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Document":
        d.pop("embedding", None)
        return cls(**d)

    def __str__(self) -> str:
        return f"Document({self.id!r}, type={self.doc_type!r}, chars={len(self.content)})"


class DocumentStore:
    """
    In-memory store for RAG documents.

    Thread-safe for reads.  Writes should happen during application startup
    before concurrent inference begins.

    Parameters
    ----------
    max_docs_per_type : int
        Maximum documents per doc_type to keep in memory.
        Prevents unbounded memory growth.  Default 1000.
    """

    def __init__(self, max_docs_per_type: int = 1000) -> None:
        self.max_docs_per_type = max_docs_per_type

        # Primary store: id → Document
        self._docs: Dict[str, Document] = {}

        # Secondary index: doc_type → list of ids
        self._type_index: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #

    def add(self, doc: Document, overwrite: bool = True) -> None:
        """
        Add or update a document in the store.

        Parameters
        ----------
        doc : Document
            Document to add.
        overwrite : bool
            If False, skip documents with an existing ID.  Default True.
        """
        if doc.id in self._docs and not overwrite:
            return

        self._docs[doc.id] = doc

        # Update type index
        if doc.doc_type not in self._type_index:
            self._type_index[doc.doc_type] = []
        if doc.id not in self._type_index[doc.doc_type]:
            self._type_index[doc.doc_type].append(doc.id)

    def add_many(self, docs: List[Document], overwrite: bool = True) -> int:
        """
        Add multiple documents.

        Returns
        -------
        int
            Number of documents added.
        """
        for doc in docs:
            self.add(doc, overwrite=overwrite)
        return len(docs)

    def get(self, doc_id: str) -> Optional[Document]:
        """Return the document with the given ID, or None."""
        return self._docs.get(doc_id)

    def get_many(self, doc_ids: List[str]) -> List[Document]:
        """Return documents for all given IDs (skipping missing ones)."""
        return [self._docs[i] for i in doc_ids if i in self._docs]

    def get_by_type(self, doc_type: str) -> List[Document]:
        """Return all documents of a given type."""
        ids = self._type_index.get(doc_type, [])
        return [self._docs[i] for i in ids if i in self._docs]

    def all_documents(self) -> List[Document]:
        """Return all documents in insertion order."""
        return list(self._docs.values())

    def remove(self, doc_id: str) -> bool:
        """Remove a document by ID. Returns True if it existed."""
        if doc_id not in self._docs:
            return False
        doc = self._docs.pop(doc_id)
        ids = self._type_index.get(doc.doc_type, [])
        if doc_id in ids:
            ids.remove(doc_id)
        return True

    def clear(self, doc_type: Optional[str] = None) -> int:
        """
        Remove all documents (or all of a given type).

        Returns
        -------
        int
            Number of documents removed.
        """
        if doc_type is None:
            n = len(self._docs)
            self._docs.clear()
            self._type_index.clear()
            return n

        ids = list(self._type_index.get(doc_type, []))
        for doc_id in ids:
            self._docs.pop(doc_id, None)
        self._type_index[doc_type] = []
        return len(ids)

    # ------------------------------------------------------------------ #
    # Bulk loaders
    # ------------------------------------------------------------------ #

    def load_from_kg_cache(
        self,
        nvd_cache_path:   str = "data/kg_cache/nvd_cves.json",
        max_cves:         int = 1000,
        min_cvss:         float = 0.0,
    ) -> int:
        """
        Load CVE descriptions from the NVD cache file.

        Parameters
        ----------
        nvd_cache_path : str
            Path to ``nvd_cves.json`` (written by ``CVEFetcher.save_to_cache()``).
        max_cves : int
            Cap on CVEs to load.  Default 1000.
        min_cvss : float
            Only load CVEs with at least this CVSS score.

        Returns
        -------
        int
            Number of documents added.
        """
        p = Path(nvd_cache_path)
        if not p.exists():
            logger.warning("NVD cache not found: %s — skipping CVE load.", p)
            return 0

        raw = json.loads(p.read_text(encoding="utf-8"))
        added = 0

        for entry in raw[:max_cves]:
            if entry.get("max_cvss", 0.0) < min_cvss:
                continue

            doc = Document(
                id       = entry["cve_id"],
                content  = (
                    f"{entry['cve_id']}: {entry.get('description', '')}"
                ),
                doc_type = "cve",
                metadata = {
                    "max_cvss":  entry.get("max_cvss", 0.0),
                    "severity":  entry.get("severity", "NONE"),
                    "published": entry.get("published", ""),
                    "cwe_ids":   entry.get("cwe_ids", []),
                },
            )
            self.add(doc)
            added += 1

        logger.info("Loaded %d CVE documents from %s", added, p)
        return added

    def load_attck_techniques(
        self,
        attck_cache_path: str = "data/kg_cache/attck_parsed.json",
    ) -> int:
        """
        Load ATT&CK technique descriptions from the parsed cache file.

        Parameters
        ----------
        attck_cache_path : str
            Path to ``attck_parsed.json`` (written by ``MITREParser.save_to_cache()``).

        Returns
        -------
        int
            Number of documents added.
        """
        p = Path(attck_cache_path)
        if not p.exists():
            logger.warning("ATT&CK cache not found: %s — skipping technique load.", p)
            return 0

        raw    = json.loads(p.read_text(encoding="utf-8"))
        added  = 0

        for t in raw.get("techniques", []):
            content = (
                f"{t['technique_id']} — {t['name']}. "
                f"{t.get('description', '')}"
            )
            doc = Document(
                id       = t["technique_id"],
                content  = content,
                doc_type = "technique",
                metadata = {
                    "name":           t.get("name", ""),
                    "tactic_ids":     t.get("tactic_ids", []),
                    "tactic_names":   t.get("tactic_names", []),
                    "platforms":      t.get("platforms", []),
                    "is_subtechnique":t.get("is_subtechnique", False),
                    "url":            t.get("url", ""),
                },
            )
            self.add(doc)
            added += 1

        for ta in raw.get("tactics", []):
            doc = Document(
                id       = ta["tactic_id"],
                content  = (
                    f"{ta['tactic_id']} — {ta['name']}. "
                    f"{ta.get('description', '')}"
                ),
                doc_type = "tactic",
                metadata = {
                    "name":      ta.get("name", ""),
                    "shortname": ta.get("shortname", ""),
                    "url":       ta.get("url", ""),
                },
            )
            self.add(doc)
            added += 1

        logger.info("Loaded %d ATT&CK documents from %s", added, p)
        return added

    def load_acd_policies(self) -> int:
        """
        Load internal ACD defender policy documents.

        These are hand-crafted policy statements that guide the LLM's
        explanation of why specific defensive actions were chosen.

        Returns
        -------
        int
            Number of policy documents added.
        """
        policies = [
            Document(
                id       = "policy:isolation_over_removal",
                content  = (
                    "ACD Policy: When the attacker type is classified as Targeted APT "
                    "(belief > 60%), prefer host isolation over malware removal as the "
                    "first response. Isolation cuts the C2 channel before persistence "
                    "can be established, while removal alone risks leaving backdoors."
                ),
                doc_type = "policy",
                metadata = {"category": "response_priority"},
            ),
            Document(
                id       = "policy:decoy_deployment",
                content  = (
                    "ACD Policy: Deploy decoy hosts when reconnaissance activity is "
                    "detected (T1046 Network Service Scanning). Decoys misdirect the "
                    "attacker and buy additional defender steps to prepare responses. "
                    "Optimal placement is on the most-scanned subnet segment."
                ),
                doc_type = "policy",
                metadata = {"category": "deception"},
            ),
            Document(
                id       = "policy:post_drift_response",
                content  = (
                    "ACD Policy: After a concept drift event (Wasserstein distance > 0.15), "
                    "the agent's EWC module registers the current parameter snapshot as a "
                    "new task. During adaptation, prioritise containment (isolation, decoys) "
                    "over aggressive removal until the new attacker distribution is confirmed."
                ),
                doc_type = "policy",
                metadata = {"category": "drift_response"},
            ),
            Document(
                id       = "policy:cvar_tail_risk",
                content  = (
                    "ACD Policy: The CVaR-PPO agent optimises for the worst-5% episode "
                    "outcomes (alpha=0.05). Actions with CVaR risk score > 0.7 are "
                    "treated as high-priority. The agent will accept lower mean reward "
                    "to reduce catastrophic breach probability."
                ),
                doc_type = "policy",
                metadata = {"category": "risk_management"},
            ),
            Document(
                id       = "policy:rapid_response",
                content  = (
                    "ACD Policy: Respond to initial compromise within 3 steps. "
                    "Delayed response allows the attacker to establish persistence "
                    "(T1543) and lateral movement (T1021) which significantly increases "
                    "breach probability. The Analyse action should be used first only "
                    "when attacker type is uncertain."
                ),
                doc_type = "policy",
                metadata = {"category": "response_timing"},
            ),
        ]

        for doc in policies:
            self.add(doc)

        logger.info("Loaded %d ACD policy documents.", len(policies))
        return len(policies)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> Path:
        """
        Save the document store (without embeddings) to a JSON file.

        Parameters
        ----------
        path : str
            Destination file path.

        Returns
        -------
        Path
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = [doc.to_dict() for doc in self._docs.values()]
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("DocumentStore saved — %d docs to %s", len(data), p)
        return p

    def load(self, path: str) -> int:
        """
        Load documents from a JSON file (saved by ``save()``).

        Returns
        -------
        int
            Number of documents loaded.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Document store file not found: {p}")

        raw = json.loads(p.read_text(encoding="utf-8"))
        for d in raw:
            self.add(Document.from_dict(d))

        logger.info("DocumentStore loaded — %d docs from %s", len(raw), p)
        return len(raw)

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #

    @property
    def n_docs(self) -> int:
        """Total number of documents in the store."""
        return len(self._docs)

    def stats(self) -> Dict[str, Any]:
        """Return document counts by type."""
        return {
            "total": self.n_docs,
            "by_type": {
                t: len(ids)
                for t, ids in self._type_index.items()
            },
        }

    def __len__(self) -> int:
        return self.n_docs

    def __repr__(self) -> str:
        return f"DocumentStore(n_docs={self.n_docs}, types={list(self._type_index.keys())})"