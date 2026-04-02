"""
knowledge/mitre_parser.py
==========================
Parses the MITRE ATT&CK framework STIX 2.1 bundle into clean
``TechniqueRecord`` and ``TacticRecord`` dataclasses.

ATT&CK STIX bundle source
--------------------------
    Enterprise ATT&CK:
    https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json

The STIX bundle is a large JSON file (~20 MB) containing every technique,
tactic, group, software, and relationship in ATT&CK.  This module only
extracts the elements needed for the ACD knowledge graph:
    - Tactics          (14 enterprise tactics, e.g. TA0001 Initial Access)
    - Techniques       (193+ enterprise techniques, e.g. T1190 Exploit Public-Facing App)
    - Sub-techniques   (e.g. T1059.001 PowerShell)
    - Tactic memberships (which tactics each technique belongs to)

Usage
-----
    parser = MITREParser()

    # Load from local file (preferred — fast)
    parser.load_from_file("data/kg_cache/enterprise-attack.json")

    # Or download directly from MITRE CTI GitHub
    parser.download_and_load()

    tactics    = parser.get_tactics()
    techniques = parser.get_techniques(include_subtechniques=True)
    mapping    = parser.get_technique_tactic_mapping()

    # Save parsed records to cache
    parser.save_to_cache("data/kg_cache/attck_parsed.json")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# MITRE CTI raw STIX bundle URL
_ATTCK_BUNDLE_URL = (
    "https://raw.githubusercontent.com/mitre/cti/master/"
    "enterprise-attack/enterprise-attack.json"
)

# STIX type identifiers
_TYPE_ATTACK_PATTERN = "attack-pattern"    # technique
_TYPE_X_MITRE_TACTIC = "x-mitre-tactic"   # tactic
_TYPE_RELATIONSHIP    = "relationship"
_TYPE_DEPRECATION_MARKER = "revoked-by"


# ── Data models ─────────────────────────────────────────────────────────────

@dataclass
class TacticRecord:
    """
    A MITRE ATT&CK tactic (the 'why' of an attack).

    Attributes
    ----------
    tactic_id : str
        ATT&CK tactic ID, e.g. ``"TA0001"``.
    name : str
        Tactic name, e.g. ``"Initial Access"``.
    shortname : str
        Tactic shortname as used in technique kill-chain phase references,
        e.g. ``"initial-access"``.
    description : str
        Full tactic description from ATT&CK.
    url : str
        ATT&CK web URL for this tactic.
    """
    tactic_id:   str
    name:        str
    shortname:   str
    description: str
    url:         str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TacticRecord":
        return cls(**d)

    def __str__(self) -> str:
        return f"{self.tactic_id}: {self.name}"


@dataclass
class TechniqueRecord:
    """
    A MITRE ATT&CK technique or sub-technique (the 'how' of an attack).

    Attributes
    ----------
    technique_id : str
        ATT&CK technique ID, e.g. ``"T1190"`` or ``"T1059.001"``.
    name : str
        Technique name, e.g. ``"Exploit Public-Facing Application"``.
    description : str
        Full technique description.
    tactic_ids : list[str]
        IDs of tactics this technique belongs to.
    tactic_names : list[str]
        Human-readable tactic names (parallel to tactic_ids).
    is_subtechnique : bool
        True if this is a sub-technique (e.g. T1059.001).
    parent_id : str, optional
        Parent technique ID for sub-techniques.
    platforms : list[str]
        Operating systems/platforms this technique applies to.
    data_sources : list[str]
        ATT&CK data sources that can detect this technique.
    detection : str
        Detection guidance text.
    url : str
        ATT&CK web URL for this technique.
    stix_id : str
        Internal STIX object ID (used to resolve relationships).
    """
    technique_id:    str
    name:            str
    description:     str
    tactic_ids:      List[str]  = field(default_factory=list)
    tactic_names:    List[str]  = field(default_factory=list)
    is_subtechnique: bool       = False
    parent_id:       Optional[str] = None
    platforms:       List[str]  = field(default_factory=list)
    data_sources:    List[str]  = field(default_factory=list)
    detection:       str        = ""
    url:             str        = ""
    stix_id:         str        = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TechniqueRecord":
        return cls(**d)

    @property
    def full_text(self) -> str:
        """Combined name + description for BERT embedding."""
        return f"{self.name}. {self.description}"

    def __str__(self) -> str:
        return f"{self.technique_id}: {self.name}"


# ── Parser ───────────────────────────────────────────────────────────────────

class MITREParser:
    """
    Parses the MITRE ATT&CK STIX 2.1 enterprise bundle.

    Maintains parsed records in memory after loading.  The parsed records
    are then passed to ``KGBuilder`` for ingestion into Neo4j.

    Parameters
    ----------
    include_deprecated : bool
        If True, include deprecated/revoked techniques.
        Default False — deprecated techniques add noise.
    include_subtechniques : bool
        If True, include sub-techniques (e.g. T1059.001).
        Default True.
    """

    def __init__(
        self,
        include_deprecated:    bool = False,
        include_subtechniques: bool = True,
    ) -> None:
        self.include_deprecated    = include_deprecated
        self.include_subtechniques = include_subtechniques

        # Parsed records (populated after load)
        self._tactics:    Dict[str, TacticRecord]    = {}
        self._techniques: Dict[str, TechniqueRecord] = {}

        # Internal: STIX ID → ATT&CK ID mapping
        self._stix_to_tactic:    Dict[str, str] = {}
        self._stix_to_technique: Dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def load_from_file(self, path: str) -> "MITREParser":
        """
        Parse the ATT&CK STIX bundle from a local JSON file.

        Parameters
        ----------
        path : str
            Path to the enterprise-attack.json file.

        Returns
        -------
        MITREParser
            Returns self for method chaining.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"ATT&CK bundle not found: {p}. "
                f"Run scripts/download_attck.py to download it."
            )

        logger.info("Loading ATT&CK STIX bundle from: %s", p)
        bundle = json.loads(p.read_text(encoding="utf-8"))
        self._parse_bundle(bundle)
        return self

    def download_and_load(
        self,
        url:      str = _ATTCK_BUNDLE_URL,
        save_to:  Optional[str] = "data/kg_cache/enterprise-attack.json",
    ) -> "MITREParser":
        """
        Download the ATT&CK bundle from MITRE CTI GitHub and parse it.

        Parameters
        ----------
        url : str
            URL of the STIX bundle.
        save_to : str, optional
            If provided, saves the downloaded bundle to this path for
            caching.  Pass None to skip saving.

        Returns
        -------
        MITREParser
        """
        import requests

        logger.info("Downloading ATT&CK STIX bundle from: %s", url)
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        bundle = resp.json()

        if save_to:
            p = Path(save_to)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
            logger.info("ATT&CK bundle saved to: %s", p)

        self._parse_bundle(bundle)
        return self

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    def get_tactics(self) -> List[TacticRecord]:
        """Return all parsed ATT&CK tactics."""
        return list(self._tactics.values())

    def get_techniques(
        self, include_subtechniques: Optional[bool] = None
    ) -> List[TechniqueRecord]:
        """
        Return parsed ATT&CK techniques.

        Parameters
        ----------
        include_subtechniques : bool, optional
            Override the instance setting for this call.

        Returns
        -------
        list[TechniqueRecord]
        """
        include_sub = (
            include_subtechniques
            if include_subtechniques is not None
            else self.include_subtechniques
        )

        return [
            t for t in self._techniques.values()
            if include_sub or not t.is_subtechnique
        ]

    def get_technique(self, technique_id: str) -> Optional[TechniqueRecord]:
        """Return a single technique by ID (e.g. ``"T1190"``)."""
        return self._techniques.get(technique_id)

    def get_tactic(self, tactic_id: str) -> Optional[TacticRecord]:
        """Return a single tactic by ID (e.g. ``"TA0001"``)."""
        return self._tactics.get(tactic_id)

    def get_technique_tactic_mapping(self) -> Dict[str, List[str]]:
        """
        Return a dict mapping technique ID → list of tactic IDs.

        Example
        -------
            {
                "T1190": ["TA0001"],
                "T1059": ["TA0002"],
                ...
            }
        """
        return {
            tid: list(t.tactic_ids)
            for tid, t in self._techniques.items()
        }

    def get_techniques_for_tactic(self, tactic_id: str) -> List[TechniqueRecord]:
        """Return all techniques that belong to a given tactic."""
        return [
            t for t in self._techniques.values()
            if tactic_id in t.tactic_ids
        ]

    def search_techniques(self, query: str) -> List[TechniqueRecord]:
        """
        Simple keyword search across technique name + description.

        Parameters
        ----------
        query : str
            Search string (case-insensitive).

        Returns
        -------
        list[TechniqueRecord]
            Techniques where name or description contains the query.
        """
        q = query.lower()
        return [
            t for t in self._techniques.values()
            if q in t.name.lower() or q in t.description.lower()
        ]

    @property
    def n_tactics(self) -> int:
        """Number of parsed tactics."""
        return len(self._tactics)

    @property
    def n_techniques(self) -> int:
        """Number of parsed techniques (including sub-techniques if enabled)."""
        return len(self._techniques)

    # ------------------------------------------------------------------ #
    # Cache I/O
    # ------------------------------------------------------------------ #

    def save_to_cache(
        self, path: str = "data/kg_cache/attck_parsed.json"
    ) -> Path:
        """
        Save parsed records to a JSON cache file.

        Faster to load on subsequent runs than re-parsing the full bundle.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "tactics":    [t.to_dict() for t in self._tactics.values()],
            "techniques": [t.to_dict() for t in self._techniques.values()],
        }
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info(
            "Saved %d tactics + %d techniques to %s",
            self.n_tactics, self.n_techniques, p,
        )
        return p

    def load_from_cache(
        self, path: str = "data/kg_cache/attck_parsed.json"
    ) -> "MITREParser":
        """
        Load pre-parsed records from a JSON cache file.

        Parameters
        ----------
        path : str
            Path written by ``save_to_cache``.

        Returns
        -------
        MITREParser
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"ATT&CK cache not found: {p}. "
                f"Run the parser first to generate it."
            )

        data = json.loads(p.read_text(encoding="utf-8"))
        self._tactics    = {t["tactic_id"]:    TacticRecord.from_dict(t)    for t in data["tactics"]}
        self._techniques = {t["technique_id"]: TechniqueRecord.from_dict(t) for t in data["techniques"]}

        logger.info(
            "Loaded %d tactics + %d techniques from %s",
            self.n_tactics, self.n_techniques, p,
        )
        return self

    # ------------------------------------------------------------------ #
    # Internal STIX parsing
    # ------------------------------------------------------------------ #

    def _parse_bundle(self, bundle: Dict[str, Any]) -> None:
        """
        Parse the raw STIX 2.1 bundle dict into internal data structures.

        Processing order:
            1. Extract tactics (x-mitre-tactic objects)
            2. Extract techniques (attack-pattern objects)
            3. Resolve tactic memberships via kill-chain phase references
        """
        objects = bundle.get("objects", [])

        # Revoked STIX IDs (skip these)
        revoked: Set[str] = set()
        if not self.include_deprecated:
            for obj in objects:
                if obj.get("revoked", False) or obj.get("x_mitre_deprecated", False):
                    revoked.add(obj.get("id", ""))

        # ── Pass 1: Extract tactics ────────────────────────────────────
        for obj in objects:
            if obj.get("type") != _TYPE_X_MITRE_TACTIC:
                continue
            if obj.get("id", "") in revoked:
                continue
            self._parse_tactic(obj)

        logger.info("Parsed %d tactics", self.n_tactics)

        # ── Pass 2: Extract techniques ─────────────────────────────────
        for obj in objects:
            if obj.get("type") != _TYPE_ATTACK_PATTERN:
                continue
            if obj.get("id", "") in revoked:
                continue

            is_sub = obj.get("x_mitre_is_subtechnique", False)
            if is_sub and not self.include_subtechniques:
                continue

            self._parse_technique(obj)

        logger.info(
            "Parsed %d techniques (%s sub-techniques)",
            self.n_techniques,
            "including" if self.include_subtechniques else "excluding",
        )

    def _parse_tactic(self, obj: Dict[str, Any]) -> None:
        """Parse a single x-mitre-tactic STIX object."""
        external = obj.get("external_references", [])
        tactic_id = next(
            (r["external_id"] for r in external if r.get("source_name") == "mitre-attack"),
            "",
        )
        url = next(
            (r.get("url", "") for r in external if r.get("source_name") == "mitre-attack"),
            "",
        )
        shortname = obj.get("x_mitre_shortname", "")
        name      = obj.get("name", "")
        desc      = self._clean_text(obj.get("description", ""))

        if not tactic_id:
            return

        record = TacticRecord(
            tactic_id   = tactic_id,
            name        = name,
            shortname   = shortname,
            description = desc,
            url         = url,
        )
        self._tactics[tactic_id] = record
        # Map shortname → tactic_id for kill-chain phase resolution
        self._stix_to_tactic[shortname] = tactic_id

    def _parse_technique(self, obj: Dict[str, Any]) -> None:
        """Parse a single attack-pattern STIX object (technique or sub-technique)."""
        external = obj.get("external_references", [])
        technique_id = next(
            (r["external_id"] for r in external if r.get("source_name") == "mitre-attack"),
            "",
        )
        url = next(
            (r.get("url", "") for r in external if r.get("source_name") == "mitre-attack"),
            "",
        )

        if not technique_id:
            return

        # Determine tactic memberships via kill_chain_phases
        tactic_ids:   List[str] = []
        tactic_names: List[str] = []
        for phase in obj.get("kill_chain_phases", []):
            if phase.get("kill_chain_name") == "mitre-attack":
                shortname = phase.get("phase_name", "")
                tactic_id = self._stix_to_tactic.get(shortname, "")
                if tactic_id:
                    tactic_ids.append(tactic_id)
                    rec = self._tactics.get(tactic_id)
                    tactic_names.append(rec.name if rec else shortname)

        # Parent technique for sub-techniques (T1059.001 → T1059)
        is_sub    = obj.get("x_mitre_is_subtechnique", False)
        parent_id = technique_id.rsplit(".", 1)[0] if is_sub and "." in technique_id else None

        record = TechniqueRecord(
            technique_id    = technique_id,
            name            = obj.get("name", ""),
            description     = self._clean_text(obj.get("description", "")),
            tactic_ids      = tactic_ids,
            tactic_names    = tactic_names,
            is_subtechnique = is_sub,
            parent_id       = parent_id,
            platforms       = obj.get("x_mitre_platforms", []),
            data_sources    = obj.get("x_mitre_data_sources", []),
            detection       = self._clean_text(obj.get("x_mitre_detection", "")),
            url             = url,
            stix_id         = obj.get("id", ""),
        )
        self._techniques[technique_id] = record

        # Keep STIX ID → ATT&CK ID mapping for relationship resolution
        if record.stix_id:
            self._stix_to_technique[record.stix_id] = technique_id

    @staticmethod
    def _clean_text(text: str) -> str:
        """Remove ATT&CK STIX markdown artifacts from description text."""
        if not text:
            return ""
        # Remove citation markers like (Citation: Source Name)
        import re
        text = re.sub(r"\(Citation:[^)]+\)", "", text)
        # Normalise whitespace
        text = " ".join(text.split())
        return text.strip()

    def __repr__(self) -> str:
        return (
            f"MITREParser("
            f"tactics={self.n_tactics}, "
            f"techniques={self.n_techniques}, "
            f"subtechniques={'included' if self.include_subtechniques else 'excluded'})"
        )