"""
knowledge/kg_builder.py
========================
Orchestrates the complete Knowledge Graph build pipeline.

Pipeline stages
---------------
    Stage 1 — Fetch CVEs        : NVD REST API → CVERecord list
    Stage 2 — Parse ATT&CK      : STIX bundle → Technique/Tactic records
    Stage 3 — Map CVE→Technique : sentence-BERT cosine similarity
    Stage 4 — Write to Neo4j    : Batch MERGE statements via Neo4jClient
    Stage 5 — Add host nodes    : CybORG network hosts linked to CVEs

The KG schema in Neo4j
-----------------------
    Nodes
    ──────
        (:CVE           {id, description, max_cvss, severity, published})
        (:Technique     {technique_id, name, description, platforms})
        (:Tactic        {tactic_id, name, shortname, description})
        (:Host          {name, scenario, status})

    Relationships
    ──────────────
        (:CVE)       -[:MAPS_TO {score}]->    (:Technique)
        (:Technique) -[:BELONGS_TO]->         (:Tactic)
        (:CVE)       -[:EXPLOITS]->           (:Host)
        (:Host)      -[:USES]->               (:Technique)

Usage
-----
    from knowledge import KGBuilder, Neo4jClient

    client  = Neo4jClient.from_env()
    builder = KGBuilder(
        neo4j_client = client,
        config       = {
            "nvd": {"api_key": "...", "max_cves": 1000},
            "bert": {"model": "all-MiniLM-L6-v2", "threshold": 0.65},
            "attck": {"bundle_path": "data/kg_cache/enterprise-attack.json"},
        },
    )

    # Full rebuild (takes ~5–10 minutes first time)
    stats = builder.build_full()
    print(stats)

    # Incremental update (only new CVEs since last build)
    stats = builder.update_recent(days=7)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from knowledge.neo4j_client import Neo4jClient
from knowledge.nvd_fetcher import CVEFetcher, CVERecord
from knowledge.mitre_parser import MITREParser, TechniqueRecord, TacticRecord
from knowledge.bert_mapper import BERTMapper, MappingResult
from knowledge.cache import DiskCache

logger = logging.getLogger(__name__)

# Host → CVE relevance mapping
# Maps CybORG host names to CVE keywords likely to affect them
_HOST_CVE_KEYWORDS: Dict[str, List[str]] = {
    "User0":        ["browser", "client", "desktop", "phishing", "macro"],
    "User1":        ["browser", "client", "desktop", "phishing"],
    "User2":        ["browser", "client", "remote"],
    "User3":        ["smb", "lateral", "windows", "credential"],
    "User4":        ["smb", "lateral", "windows"],
    "Enterprise0":  ["apache", "tomcat", "web", "server", "injection", "rce"],
    "Op_Server0":   ["critical", "rce", "privilege", "escalation", "ssh"],
}


class KGBuilder:
    """
    Builds and maintains the ACD cybersecurity knowledge graph in Neo4j.

    Coordinates all four pipeline stages: fetch, parse, map, write.

    Parameters
    ----------
    neo4j_client : Neo4jClient
        Connected Neo4j client instance.
    config : dict
        Build configuration.  Keys:
            nvd.api_key         : NVD API key (or set NVD_API_KEY env var)
            nvd.max_cves        : Max CVEs to fetch.  Default 1000.
            nvd.min_cvss        : Minimum CVSS score.  Default 6.0.
            bert.model          : sentence-BERT model name.
            bert.threshold      : Mapping acceptance threshold.  Default 0.65.
            bert.top_k          : Top-k techniques per CVE.  Default 3.
            attck.bundle_path   : Local path to enterprise-attack.json.
            cache_dir           : Directory for intermediate cache files.
    """

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        config:       Optional[Dict[str, Any]] = None,
    ) -> None:
        self.client = neo4j_client
        self.config = config or {}

        # Sub-configs
        self._nvd_cfg  = self.config.get("nvd", {})
        self._bert_cfg = self.config.get("bert", {})
        self._attck_cfg = self.config.get("attck", {})

        cache_dir = self.config.get("cache_dir", "data/kg_cache")
        self._cache = DiskCache(cache_dir=cache_dir)

        # Build stats
        self._last_build_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Full build
    # ------------------------------------------------------------------ #

    def build_full(
        self,
        drop_existing: bool = False,
        scenarios:     Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the complete KG build pipeline from scratch.

        Parameters
        ----------
        drop_existing : bool
            If True, deletes all existing nodes before rebuilding.
            Default False — uses MERGE to update in place.
        scenarios : list[str], optional
            CybORG scenario names to add host nodes for.
            Default: ``["scenario2"]``.

        Returns
        -------
        dict
            Build statistics: n_cves, n_techniques, n_tactics,
            n_mappings, n_hosts, elapsed_s.
        """
        start = time.monotonic()
        scenarios = scenarios or ["scenario2"]

        logger.info("=== KG Full Build Starting ===")

        # ── Setup ─────────────────────────────────────────────────────
        self.client.create_constraints()

        if drop_existing:
            logger.warning("Dropping all existing KG data...")
            self.client.drop_all()

        # ── Stage 1: Fetch CVEs ────────────────────────────────────────
        cves = self._fetch_cves()

        # ── Stage 2: Parse ATT&CK ─────────────────────────────────────
        parser = self._load_attck()
        techniques = parser.get_techniques(include_subtechniques=True)
        tactics    = parser.get_tactics()

        # ── Stage 3: Map CVE → Technique ──────────────────────────────
        mapper   = self._build_mapper(techniques)
        mappings = mapper.map_cves(
            cves,
            top_k     = self._bert_cfg.get("top_k", 3),
            threshold = self._bert_cfg.get("threshold", 0.65),
        )

        # Write mappings back to CVE records
        for cve, mapping in zip(cves, mappings):
            cve.mapped_techniques = [
                tid for tid, ok in zip(mapping.technique_ids, mapping.accepted) if ok
            ]

        # ── Stage 4: Write to Neo4j ────────────────────────────────────
        n_tactics    = self._write_tactics(tactics)
        n_techniques = self._write_techniques(techniques)
        n_cves       = self._write_cves(cves)
        n_mappings   = self._write_mappings(mappings)
        n_tech_tactic = self._write_technique_tactic_edges(techniques)

        # ── Stage 5: Host nodes ────────────────────────────────────────
        n_hosts = self._write_hosts(scenarios, cves)

        elapsed = time.monotonic() - start

        stats = {
            "n_cves":              n_cves,
            "n_techniques":        n_techniques,
            "n_tactics":           n_tactics,
            "n_cve_mappings":      n_mappings,
            "n_tech_tactic_edges": n_tech_tactic,
            "n_hosts":             n_hosts,
            "elapsed_s":           round(elapsed, 1),
        }
        self._last_build_stats = stats

        logger.info(
            "=== KG Build Complete === "
            "CVEs=%d, Techniques=%d, Tactics=%d, Mappings=%d, Hosts=%d — %.1fs",
            n_cves, n_techniques, n_tactics, n_mappings, n_hosts, elapsed,
        )
        return stats

    def update_recent(self, days: int = 7) -> Dict[str, Any]:
        """
        Incremental update: fetch new CVEs from the last N days and add them.

        Does not re-parse ATT&CK (assumed to be up-to-date).
        Uses cached technique index for fast BERT mapping.

        Parameters
        ----------
        days : int
            How many days back to check for new CVEs.

        Returns
        -------
        dict
            Update statistics.
        """
        logger.info("=== KG Incremental Update (last %d days) ===", days)
        start = time.monotonic()

        fetcher = CVEFetcher(api_key=self._nvd_cfg.get("api_key"))
        new_cves = fetcher.fetch_recent(
            days        = days,
            max_results = self._nvd_cfg.get("max_recent", 200),
        )

        if not new_cves:
            logger.info("No new CVEs found in the last %d days.", days)
            return {"n_new_cves": 0, "elapsed_s": 0.0}

        # Load existing technique index
        mapper = BERTMapper(
            model_name = self._bert_cfg.get("model", "all-MiniLM-L6-v2"),
            threshold  = self._bert_cfg.get("threshold", 0.65),
        )
        index_path = Path(self.config.get("cache_dir", "data/kg_cache")) / "technique_index.npz"
        if index_path.exists():
            mapper.load_index(str(index_path))
        else:
            # Rebuild index from cached ATT&CK data
            parser     = self._load_attck()
            techniques = parser.get_techniques()
            mapper.build_technique_index(techniques)

        mappings = mapper.map_cves(new_cves, top_k=3)
        for cve, mapping in zip(new_cves, mappings):
            cve.mapped_techniques = [
                tid for tid, ok in zip(mapping.technique_ids, mapping.accepted) if ok
            ]

        n_cves    = self._write_cves(new_cves)
        n_mappings = self._write_mappings(mappings)

        elapsed = time.monotonic() - start
        return {
            "n_new_cves":     n_cves,
            "n_new_mappings": n_mappings,
            "elapsed_s":      round(elapsed, 1),
        }

    # ------------------------------------------------------------------ #
    # Stage implementations
    # ------------------------------------------------------------------ #

    def _fetch_cves(self) -> List[CVERecord]:
        """Stage 1: Fetch CVEs from NVD or load from cache."""
        cache_path = self._cache.path("nvd_cves.json")

        if self._cache.exists("nvd_cves.json"):
            logger.info("Loading CVEs from cache: %s", cache_path)
            return CVEFetcher.load_from_cache(str(cache_path))

        logger.info("Fetching CVEs from NVD API...")
        fetcher = CVEFetcher(api_key=self._nvd_cfg.get("api_key"))
        cves    = fetcher.fetch_by_severity(
            min_cvss    = self._nvd_cfg.get("min_cvss", 6.0),
            max_results = self._nvd_cfg.get("max_cves", 1000),
        )
        CVEFetcher.save_to_cache(cves, str(cache_path))
        return cves

    def _load_attck(self) -> MITREParser:
        """Stage 2: Load ATT&CK STIX bundle."""
        bundle_path = self._attck_cfg.get(
            "bundle_path", "data/kg_cache/enterprise-attack.json"
        )

        parser = MITREParser(include_subtechniques=True)

        parsed_cache = self._cache.path("attck_parsed.json")
        if self._cache.exists("attck_parsed.json"):
            logger.info("Loading ATT&CK from parsed cache: %s", parsed_cache)
            return parser.load_from_cache(str(parsed_cache))

        if Path(bundle_path).exists():
            parser.load_from_file(bundle_path)
        else:
            logger.info("ATT&CK bundle not found locally — downloading...")
            parser.download_and_load(save_to=bundle_path)

        parser.save_to_cache(str(parsed_cache))
        return parser

    def _build_mapper(self, techniques: List[TechniqueRecord]) -> BERTMapper:
        """Stage 3a: Build or load BERT technique index."""
        mapper = BERTMapper(
            model_name = self._bert_cfg.get("model", "all-MiniLM-L6-v2"),
            threshold  = self._bert_cfg.get("threshold", 0.65),
            device     = self._bert_cfg.get("device", "cpu"),
        )

        index_path = self._cache.path("technique_index.npz")
        if self._cache.exists("technique_index.npz"):
            logger.info("Loading BERT technique index from: %s", index_path)
            mapper.load_index(str(index_path))
        else:
            logger.info("Building BERT technique index...")
            mapper.build_technique_index(techniques)
            mapper.save_index(str(index_path))

        return mapper

    # ------------------------------------------------------------------ #
    # Neo4j write methods
    # ------------------------------------------------------------------ #

    def _write_tactics(self, tactics: List[TacticRecord]) -> int:
        """Write tactic nodes to Neo4j."""
        rows = [
            {
                "tactic_id":   t.tactic_id,
                "name":        t.name,
                "shortname":   t.shortname,
                "description": t.description[:500],
                "url":         t.url,
            }
            for t in tactics
        ]
        self.client.execute_batch(
            "UNWIND $rows AS row "
            "MERGE (t:Tactic {tactic_id: row.tactic_id}) "
            "SET t.name = row.name, t.shortname = row.shortname, "
            "    t.description = row.description, t.url = row.url",
            rows=rows,
        )
        logger.info("Wrote %d tactic nodes.", len(rows))
        return len(rows)

    def _write_techniques(self, techniques: List[TechniqueRecord]) -> int:
        """Write technique nodes to Neo4j."""
        rows = [
            {
                "technique_id":    t.technique_id,
                "name":            t.name,
                "description":     t.description[:1000],
                "is_subtechnique": t.is_subtechnique,
                "parent_id":       t.parent_id or "",
                "platforms":       ", ".join(t.platforms),
                "detection":       t.detection[:500],
                "url":             t.url,
            }
            for t in techniques
        ]
        self.client.execute_batch(
            "UNWIND $rows AS row "
            "MERGE (t:Technique {technique_id: row.technique_id}) "
            "SET t.name = row.name, t.description = row.description, "
            "    t.is_subtechnique = row.is_subtechnique, "
            "    t.parent_id = row.parent_id, t.platforms = row.platforms, "
            "    t.detection = row.detection, t.url = row.url",
            rows=rows,
        )
        logger.info("Wrote %d technique nodes.", len(rows))
        return len(rows)

    def _write_cves(self, cves: List[CVERecord]) -> int:
        """Write CVE nodes to Neo4j."""
        rows = [
            {
                "id":          c.cve_id,
                "description": c.description[:2000],
                "published":   c.published,
                "modified":    c.modified,
                "max_cvss":    c.max_cvss,
                "severity":    c.severity,
                "cwe_ids":     ", ".join(c.cwe_ids[:5]),
            }
            for c in cves
        ]
        self.client.execute_batch(
            "UNWIND $rows AS row "
            "MERGE (c:CVE {id: row.id}) "
            "SET c.description = row.description, c.published = row.published, "
            "    c.modified = row.modified, c.max_cvss = row.max_cvss, "
            "    c.severity = row.severity, c.cwe_ids = row.cwe_ids",
            rows=rows,
        )
        logger.info("Wrote %d CVE nodes.", len(rows))
        return len(rows)

    def _write_mappings(self, mappings: List[MappingResult]) -> int:
        """Write CVE → Technique MAPS_TO relationships."""
        rows = []
        for m in mappings:
            for tid, score, accepted in zip(m.technique_ids, m.scores, m.accepted):
                if accepted:
                    rows.append({
                        "cve_id":       m.cve_id,
                        "technique_id": tid,
                        "score":        round(score, 4),
                    })

        if not rows:
            return 0

        self.client.execute_batch(
            "UNWIND $rows AS row "
            "MATCH (c:CVE {id: row.cve_id}) "
            "MATCH (t:Technique {technique_id: row.technique_id}) "
            "MERGE (c)-[r:MAPS_TO]->(t) "
            "SET r.score = row.score",
            rows=rows,
        )
        logger.info("Wrote %d CVE→Technique MAPS_TO edges.", len(rows))
        return len(rows)

    def _write_technique_tactic_edges(
        self, techniques: List[TechniqueRecord]
    ) -> int:
        """Write Technique → Tactic BELONGS_TO relationships."""
        rows = []
        for t in techniques:
            for tactic_id in t.tactic_ids:
                rows.append({
                    "technique_id": t.technique_id,
                    "tactic_id":    tactic_id,
                })

        if not rows:
            return 0

        self.client.execute_batch(
            "UNWIND $rows AS row "
            "MATCH (t:Technique {technique_id: row.technique_id}) "
            "MATCH (ta:Tactic {tactic_id: row.tactic_id}) "
            "MERGE (t)-[:BELONGS_TO]->(ta)",
            rows=rows,
        )
        logger.info("Wrote %d Technique→Tactic BELONGS_TO edges.", len(rows))
        return len(rows)

    def _write_hosts(
        self,
        scenarios: List[str],
        cves: List[CVERecord],
    ) -> int:
        """
        Write CybORG Host nodes and link them to relevant CVEs.

        Uses keyword matching (``_HOST_CVE_KEYWORDS``) to create
        EXPLOITS edges from CVEs to hosts.
        """
        from envs.scenario_loader import ScenarioLoader
        loader = ScenarioLoader()
        host_rows = []

        for scenario in scenarios:
            try:
                meta = loader.get_metadata(scenario)
                for host in meta["hosts"]:
                    host_rows.append({
                        "name":     host,
                        "scenario": scenario,
                        "status":   "clean",
                    })
            except Exception as exc:
                logger.warning("Could not get hosts for scenario %r: %s", scenario, exc)

        if host_rows:
            self.client.execute_batch(
                "UNWIND $rows AS row "
                "MERGE (h:Host {name: row.name}) "
                "SET h.scenario = row.scenario, h.status = row.status",
                rows=host_rows,
            )

        # Create CVE → Host EXPLOITS edges via keyword matching
        exploit_rows = []
        for cve in cves:
            desc_lower = cve.description.lower()
            for host, keywords in _HOST_CVE_KEYWORDS.items():
                if any(kw in desc_lower for kw in keywords):
                    exploit_rows.append({
                        "cve_id":    cve.cve_id,
                        "host_name": host,
                    })

        if exploit_rows:
            self.client.execute_batch(
                "UNWIND $rows AS row "
                "MATCH (c:CVE {id: row.cve_id}) "
                "MATCH (h:Host {name: row.host_name}) "
                "MERGE (c)-[:EXPLOITS]->(h)",
                rows=exploit_rows,
            )

        n_hosts = len(host_rows)
        logger.info("Wrote %d host nodes, %d EXPLOITS edges.", n_hosts, len(exploit_rows))
        return n_hosts

    def __repr__(self) -> str:
        return (
            f"KGBuilder("
            f"neo4j={self.client!r}, "
            f"last_build={self._last_build_stats.get('n_cves', 0)} CVEs)"
        )