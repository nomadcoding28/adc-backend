"""
knowledge/nvd_fetcher.py
========================
Fetches CVE data from the NIST National Vulnerability Database (NVD)
REST API v2 and converts it into clean ``CVERecord`` dataclasses.

NVD API v2 documentation:
    https://nvd.nist.gov/developers/vulnerabilities

Rate limits
-----------
    Without API key : 5 requests / 30 seconds
    With API key    : 50 requests / 30 seconds

Set ``NVD_API_KEY`` environment variable to use the higher rate limit.
Get a free key at: https://nvd.nist.gov/developers/request-an-api-key

Usage
-----
    fetcher = CVEFetcher(api_key=os.getenv("NVD_API_KEY"))

    # Fetch a single CVE
    record = fetcher.fetch_cve("CVE-2021-44228")

    # Fetch all CVEs matching a keyword
    records = fetcher.search(keyword="Log4j", max_results=100)

    # Fetch CVEs with CVSS score >= 7.0 published in 2021
    records = fetcher.fetch_by_severity(
        min_cvss=7.0,
        pub_start="2021-01-01",
        pub_end="2021-12-31",
    )

    # Fetch CVEs relevant to specific CWEs
    records = fetcher.fetch_by_cwe(["CWE-79", "CWE-89"], max_results=500)

    # Save to disk (for KG build pipeline)
    fetcher.save_to_cache(records, path="data/kg_cache/nvd_cves.json")
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ── NVD API constants ───────────────────────────────────────────────────────
_NVD_BASE_URL    = "https://services.nvd.nist.gov/rest/json/cves/2.0"
_PAGE_SIZE       = 2000      # Max results per NVD API page
_RATE_LIMIT_DELAY_NO_KEY  = 6.0   # seconds between requests (no key)
_RATE_LIMIT_DELAY_WITH_KEY = 0.6  # seconds between requests (with key)
_REQUEST_TIMEOUT = 30.0            # seconds per HTTP request


# ── Data model ──────────────────────────────────────────────────────────────

@dataclass
class CVSSScore:
    """CVSS score details for a single CVSS version."""
    version:         str
    vector_string:   str
    base_score:      float
    severity:        str          # "NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"
    exploitability:  Optional[float] = None
    impact:          Optional[float] = None


@dataclass
class CVERecord:
    """
    Structured representation of a single CVE entry from NVD.

    This is the canonical data structure passed between the fetcher,
    BERT mapper, and KG builder.

    Attributes
    ----------
    cve_id : str
        CVE identifier, e.g. ``"CVE-2021-44228"``.
    description : str
        Full English description from NVD.
    published : str
        ISO 8601 publication date string.
    modified : str
        ISO 8601 last-modified date string.
    cvss_scores : list[CVSSScore]
        CVSS scores (one per CVSS version available).
    max_cvss : float
        Highest base score across all CVSS versions.
    severity : str
        Severity label for the highest CVSS score.
    cwe_ids : list[str]
        Associated CWE weakness identifiers.
    cpe_uris : list[str]
        Affected CPE URIs (product/platform strings).
    references : list[str]
        Reference URLs from the NVD entry.
    mapped_techniques : list[str]
        ATT&CK technique IDs mapped by BERTMapper (populated later).
    """
    cve_id:             str
    description:        str
    published:          str
    modified:           str
    cvss_scores:        List[CVSSScore]   = field(default_factory=list)
    max_cvss:           float             = 0.0
    severity:           str               = "NONE"
    cwe_ids:            List[str]         = field(default_factory=list)
    cpe_uris:           List[str]         = field(default_factory=list)
    references:         List[str]         = field(default_factory=list)
    mapped_techniques:  List[str]         = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serialisable dict."""
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CVERecord":
        """Reconstruct from a dict (as stored in cache JSON)."""
        cvss_raw = data.pop("cvss_scores", [])
        scores = [CVSSScore(**s) for s in cvss_raw]
        return cls(cvss_scores=scores, **data)

    def __str__(self) -> str:
        return f"{self.cve_id} (CVSS {self.max_cvss} {self.severity})"


# ── Fetcher ─────────────────────────────────────────────────────────────────

class CVEFetcher:
    """
    Fetches CVE data from the NVD REST API v2.

    Parameters
    ----------
    api_key : str, optional
        NVD API key.  Without a key, requests are rate-limited to 5/30s.
    request_timeout : float
        Per-request timeout in seconds.
    max_retries : int
        HTTP retry count on transient failures (5xx, connection errors).
    """

    def __init__(
        self,
        api_key:         Optional[str] = None,
        request_timeout: float = _REQUEST_TIMEOUT,
        max_retries:     int = 3,
    ) -> None:
        self.api_key         = api_key or os.getenv("NVD_API_KEY")
        self.request_timeout = request_timeout
        self._delay          = (
            _RATE_LIMIT_DELAY_WITH_KEY if self.api_key
            else _RATE_LIMIT_DELAY_NO_KEY
        )

        # Build a requests session with automatic retry on 5xx
        self._session = requests.Session()
        retry = Retry(
            total             = max_retries,
            backoff_factor    = 1.5,
            status_forcelist  = [429, 500, 502, 503, 504],
            allowed_methods   = ["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        self._session.mount("https://", adapter)

        if self.api_key:
            self._session.headers.update({"apiKey": self.api_key})

        logger.info(
            "CVEFetcher initialised — api_key=%s, rate_delay=%.1fs",
            "set" if self.api_key else "not set",
            self._delay,
        )

    # ------------------------------------------------------------------ #
    # Public fetch methods
    # ------------------------------------------------------------------ #

    def fetch_cve(self, cve_id: str) -> Optional[CVERecord]:
        """
        Fetch a single CVE by ID.

        Parameters
        ----------
        cve_id : str
            CVE identifier, e.g. ``"CVE-2021-44228"``.

        Returns
        -------
        CVERecord or None
            None if the CVE is not found.
        """
        params = {"cveId": cve_id}
        data   = self._get(_NVD_BASE_URL, params)

        if not data or not data.get("vulnerabilities"):
            logger.warning("CVE not found: %s", cve_id)
            return None

        return self._parse_vulnerability(data["vulnerabilities"][0])

    def search(
        self,
        keyword:     str,
        max_results: int = 200,
    ) -> List[CVERecord]:
        """
        Search CVEs by keyword (full-text search on description).

        Parameters
        ----------
        keyword : str
            Search term (e.g. ``"Log4j"``, ``"remote code execution"``).
        max_results : int
            Maximum number of records to return.

        Returns
        -------
        list[CVERecord]
        """
        return self._paginate(
            params      = {"keywordSearch": keyword, "keywordExactMatch": False},
            max_results = max_results,
        )

    def fetch_by_severity(
        self,
        min_cvss:  float = 7.0,
        pub_start: Optional[str] = None,
        pub_end:   Optional[str] = None,
        max_results: int = 1000,
    ) -> List[CVERecord]:
        """
        Fetch CVEs filtered by minimum CVSS score and publication date range.

        Parameters
        ----------
        min_cvss : float
            Minimum CVSS base score (0.0–10.0).
        pub_start : str, optional
            ISO 8601 start date, e.g. ``"2021-01-01T00:00:00.000"``.
        pub_end : str, optional
            ISO 8601 end date.
        max_results : int
            Cap on total records returned.

        Returns
        -------
        list[CVERecord]
        """
        params: Dict[str, Any] = {
            "cvssV3Severity": self._cvss_to_severity_label(min_cvss),
        }
        if pub_start:
            params["pubStartDate"] = pub_start
        if pub_end:
            params["pubEndDate"] = pub_end

        return self._paginate(params=params, max_results=max_results)

    def fetch_by_cwe(
        self,
        cwe_ids:     List[str],
        max_results: int = 500,
    ) -> List[CVERecord]:
        """
        Fetch CVEs associated with specific CWE weakness categories.

        NVD API supports only one CWE per request, so this method
        makes one request per CWE and deduplicates by CVE ID.

        Parameters
        ----------
        cwe_ids : list[str]
            List of CWE identifiers, e.g. ``["CWE-79", "CWE-89"]``.
        max_results : int
            Total cap across all CWEs.

        Returns
        -------
        list[CVERecord]
        """
        seen_ids: set = set()
        records:  List[CVERecord] = []

        for cwe_id in cwe_ids:
            if len(records) >= max_results:
                break

            batch = self._paginate(
                params      = {"cweId": cwe_id},
                max_results = min(200, max_results - len(records)),
            )

            for rec in batch:
                if rec.cve_id not in seen_ids:
                    seen_ids.add(rec.cve_id)
                    records.append(rec)

        logger.info(
            "Fetched %d unique CVEs for CWEs: %s", len(records), cwe_ids
        )
        return records

    def fetch_recent(
        self,
        days:        int = 30,
        max_results: int = 500,
    ) -> List[CVERecord]:
        """
        Fetch CVEs published in the last ``days`` days.

        Parameters
        ----------
        days : int
            How many days back to search.  Default 30.
        max_results : int
            Cap on records returned.

        Returns
        -------
        list[CVERecord]
        """
        from datetime import datetime, timedelta, timezone

        end   = datetime.now(timezone.utc)
        start = end - timedelta(days=days)

        fmt = "%Y-%m-%dT%H:%M:%S.000"
        return self.fetch_by_severity(
            min_cvss    = 0.0,
            pub_start   = start.strftime(fmt),
            pub_end     = end.strftime(fmt),
            max_results = max_results,
        )

    # ------------------------------------------------------------------ #
    # Cache I/O
    # ------------------------------------------------------------------ #

    @staticmethod
    def save_to_cache(
        records: List[CVERecord],
        path:    str = "data/kg_cache/nvd_cves.json",
    ) -> Path:
        """
        Persist a list of CVERecords to a JSON cache file.

        Parameters
        ----------
        records : list[CVERecord]
        path : str
            Destination file path.

        Returns
        -------
        Path
            Resolved path of the written file.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        data = [r.to_dict() for r in records]
        p.write_text(json.dumps(data, indent=2))

        logger.info("Saved %d CVE records to %s", len(records), p)
        return p

    @staticmethod
    def load_from_cache(
        path: str = "data/kg_cache/nvd_cves.json",
    ) -> List[CVERecord]:
        """
        Load CVERecords from a JSON cache file.

        Parameters
        ----------
        path : str
            Path to the cache file written by ``save_to_cache``.

        Returns
        -------
        list[CVERecord]
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"CVE cache not found: {p}. "
                f"Run scripts/download_nvd.py to generate it."
            )

        raw = json.loads(p.read_text())
        records = [CVERecord.from_dict(d) for d in raw]
        logger.info("Loaded %d CVE records from %s", len(records), p)
        return records

    # ------------------------------------------------------------------ #
    # Internal HTTP helpers
    # ------------------------------------------------------------------ #

    def _paginate(
        self,
        params:      Dict[str, Any],
        max_results: int,
    ) -> List[CVERecord]:
        """
        Iterate NVD API pages until ``max_results`` records are collected.
        """
        records: List[CVERecord] = []
        start_idx = 0

        while len(records) < max_results:
            page_params = {
                **params,
                "startIndex":   start_idx,
                "resultsPerPage": min(_PAGE_SIZE, max_results - len(records)),
            }

            data = self._get(_NVD_BASE_URL, page_params)
            if not data:
                break

            vulns = data.get("vulnerabilities", [])
            if not vulns:
                break

            for v in vulns:
                try:
                    rec = self._parse_vulnerability(v)
                    records.append(rec)
                except Exception as exc:
                    logger.debug("Failed to parse CVE entry: %s", exc)

            total_available = data.get("totalResults", 0)
            start_idx += len(vulns)

            logger.debug(
                "Fetched page — start=%d, got=%d, total=%d, collected=%d",
                start_idx - len(vulns), len(vulns), total_available, len(records),
            )

            if start_idx >= total_available:
                break

            # Respect NVD rate limits
            time.sleep(self._delay)

        logger.info("Paginator complete — %d records collected", len(records))
        return records[:max_results]

    def _get(
        self, url: str, params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Execute a single GET request with rate-limit delay."""
        try:
            resp = self._session.get(
                url,
                params  = params,
                timeout = self.request_timeout,
            )
            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 403:
                logger.error("NVD API key rejected or rate limit exceeded: %s", exc)
            else:
                logger.error("NVD HTTP error: %s", exc)
            return None

        except requests.exceptions.RequestException as exc:
            logger.error("NVD request failed: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    # Parsing helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_vulnerability(raw: Dict[str, Any]) -> CVERecord:
        """Convert a raw NVD v2 vulnerability entry to a CVERecord."""
        cve_data = raw.get("cve", raw)
        cve_id   = cve_data.get("id", "UNKNOWN")

        # Description (prefer English)
        descriptions = cve_data.get("descriptions", [])
        description  = next(
            (d["value"] for d in descriptions if d.get("lang") == "en"),
            descriptions[0]["value"] if descriptions else "",
        )

        # Dates
        published = cve_data.get("published", "")
        modified  = cve_data.get("lastModified", "")

        # CVSS scores
        cvss_scores: List[CVSSScore] = []
        metrics = cve_data.get("metrics", {})

        for version_key, version_label in [
            ("cvssMetricV31", "3.1"),
            ("cvssMetricV30", "3.0"),
            ("cvssMetricV2",  "2.0"),
        ]:
            for m in metrics.get(version_key, []):
                cvss_data = m.get("cvssData", {})
                cvss_scores.append(CVSSScore(
                    version        = version_label,
                    vector_string  = cvss_data.get("vectorString", ""),
                    base_score     = float(cvss_data.get("baseScore", 0.0)),
                    severity       = m.get("baseSeverity", cvss_data.get("baseSeverity", "NONE")),
                    exploitability = m.get("exploitabilityScore"),
                    impact         = m.get("impactScore"),
                ))

        max_cvss = max((s.base_score for s in cvss_scores), default=0.0)
        severity = next(
            (s.severity for s in cvss_scores if s.base_score == max_cvss),
            "NONE",
        )

        # CWEs
        weaknesses = cve_data.get("weaknesses", [])
        cwe_ids: List[str] = []
        for w in weaknesses:
            for d in w.get("description", []):
                if d.get("lang") == "en" and d.get("value", "").startswith("CWE-"):
                    cwe_ids.append(d["value"])

        # CPE URIs (affected configurations)
        cpe_uris: List[str] = []
        for config in cve_data.get("configurations", []):
            for node in config.get("nodes", []):
                for cpe_match in node.get("cpeMatch", []):
                    uri = cpe_match.get("criteria", "")
                    if uri:
                        cpe_uris.append(uri)

        # Reference URLs
        references = [
            r.get("url", "")
            for r in cve_data.get("references", [])
            if r.get("url")
        ]

        return CVERecord(
            cve_id      = cve_id,
            description = description,
            published   = published,
            modified    = modified,
            cvss_scores = cvss_scores,
            max_cvss    = max_cvss,
            severity    = severity,
            cwe_ids     = list(set(cwe_ids)),
            cpe_uris    = cpe_uris[:50],   # cap to avoid bloat
            references  = references[:20],
        )

    @staticmethod
    def _cvss_to_severity_label(min_cvss: float) -> str:
        """Map a minimum CVSS score to the NVD severity filter string."""
        if min_cvss >= 9.0:
            return "CRITICAL"
        elif min_cvss >= 7.0:
            return "HIGH"
        elif min_cvss >= 4.0:
            return "MEDIUM"
        return "LOW"

    def __repr__(self) -> str:
        return (
            f"CVEFetcher("
            f"api_key={'set' if self.api_key else 'not set'}, "
            f"rate_delay={self._delay}s)"
        )