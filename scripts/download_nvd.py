"""
scripts/download_nvd.py
========================
Download CVE data from the NIST National Vulnerability Database (NVD)
REST API v2 and save to the local KG cache.

API reference: https://nvd.nist.gov/developers/vulnerabilities

Rate limits
-----------
    Without API key : 5 requests per 30 seconds
    With API key    : 50 requests per 30 seconds

Get a free API key at: https://nvd.nist.gov/developers/request-an-api-key
Set it via: export NVD_API_KEY=your-key-here
Or in .env: NVD_API_KEY=your-key-here

Usage
-----
    # Download up to 1000 CVEs with CVSS ≥ 7.0 (high + critical)
    python scripts/download_nvd.py

    # Download a specific number with custom severity filter
    python scripts/download_nvd.py --max-cves 2000 --min-cvss 6.0

    # Force re-download even if cache exists
    python scripts/download_nvd.py --force

    # Download CVEs modified in the last 7 days (incremental update)
    python scripts/download_nvd.py --days 7

Output
------
    data/kg_cache/nvd_cves.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.structlog_config import configure_logging
from utils.timer import Timer

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────
NVD_API_BASE       = "https://services.nvd.nist.gov/rest/json/cves/2.0"
DEFAULT_OUTPUT     = Path("data/kg_cache/nvd_cves.json")
DEFAULT_MAX_CVES   = 1000
DEFAULT_MIN_CVSS   = 7.0
PAGE_SIZE          = 2000   # NVD max per page
RETRY_LIMIT        = 5
RETRY_DELAY_S      = 6.0    # NVD asks for 6s between requests without key
RETRY_DELAY_KEY_S  = 0.6    # with API key


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download CVE data from NVD API v2 to local cache."
    )
    p.add_argument("--output",    default=str(DEFAULT_OUTPUT),  help="Output JSON file path")
    p.add_argument("--max-cves",  type=int,   default=DEFAULT_MAX_CVES, help="Maximum CVEs to fetch")
    p.add_argument("--min-cvss",  type=float, default=DEFAULT_MIN_CVSS, help="Minimum CVSS v3 base score")
    p.add_argument("--days",      type=int,   default=None, help="Only fetch CVEs modified in last N days")
    p.add_argument("--api-key",   default=None, help="NVD API key (overrides NVD_API_KEY env var)")
    p.add_argument("--force",     action="store_true", help="Force re-download even if cache exists")
    p.add_argument("--verbose",   action="store_true")
    return p.parse_args()


def _build_request_params(
    start_index: int,
    min_cvss:    float,
    days:        Optional[int],
) -> Dict[str, Any]:
    """Build the NVD API v2 query parameters dict."""
    params: Dict[str, Any] = {
        "startIndex":   start_index,
        "resultsPerPage": PAGE_SIZE,
    }

    if min_cvss > 0:
        params["cvssV3Severity"] = _cvss_to_severity(min_cvss)

    if days is not None:
        now  = datetime.now(timezone.utc)
        then = now - timedelta(days=days)
        params["lastModStartDate"] = then.strftime("%Y-%m-%dT%H:%M:%S.000")
        params["lastModEndDate"]   = now.strftime("%Y-%m-%dT%H:%M:%S.000")

    return params


def _cvss_to_severity(min_score: float) -> str:
    """Convert minimum CVSS score to NVD severity string."""
    if min_score >= 9.0:  return "CRITICAL"
    if min_score >= 7.0:  return "HIGH"
    if min_score >= 4.0:  return "MEDIUM"
    return "LOW"


def _parse_cve_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Parse a single NVD CVE 2.0 API item into our internal format.

    Parameters
    ----------
    item : dict
        Raw item from the NVD API ``vulnerabilities`` array.

    Returns
    -------
    dict or None
        Parsed CVE dict, or None if the item is rejected or rejected.
    """
    cve_data = item.get("cve", {})
    cve_id   = cve_data.get("id", "")

    if not cve_id:
        return None

    # Description (prefer English)
    descriptions = cve_data.get("descriptions", [])
    description  = ""
    for d in descriptions:
        if d.get("lang") == "en":
            description = d.get("value", "")
            break

    # CVSS v3 scores
    max_cvss  = 0.0
    severity  = "NONE"
    metrics   = cve_data.get("metrics", {})

    for key in ("cvssMetricV31", "cvssMetricV30"):
        entries = metrics.get(key, [])
        for entry in entries:
            data = entry.get("cvssData", {})
            score = data.get("baseScore", 0.0)
            if score > max_cvss:
                max_cvss = score
                severity = data.get("baseSeverity", "NONE")

    # Fallback to CVSSv2 if no v3
    if max_cvss == 0.0:
        for entry in metrics.get("cvssMetricV2", []):
            data  = entry.get("cvssData", {})
            score = data.get("baseScore", 0.0)
            if score > max_cvss:
                max_cvss = score
                severity = entry.get("baseSeverity", "NONE")

    # CWE IDs
    cwe_ids: List[str] = []
    for weakness in cve_data.get("weaknesses", []):
        for wd in weakness.get("description", []):
            val = wd.get("value", "")
            if val.startswith("CWE-"):
                cwe_ids.append(val)

    # Published / modified dates
    published = cve_data.get("published", "")
    modified  = cve_data.get("lastModified", "")

    # CPE / affected products
    affected_products: List[str] = []
    for cfg in cve_data.get("configurations", []):
        for node in cfg.get("nodes", []):
            for cpe_match in node.get("cpeMatch", []):
                if cpe_match.get("vulnerable"):
                    affected_products.append(cpe_match.get("criteria", ""))

    return {
        "cve_id":              cve_id,
        "description":         description[:2000],    # cap length
        "max_cvss":            round(max_cvss, 1),
        "severity":            severity,
        "cwe_ids":             cwe_ids,
        "published":           published[:10],        # YYYY-MM-DD
        "modified":            modified[:10],
        "affected_products":   affected_products[:10], # cap list
    }


def fetch_cves(
    max_cves:  int,
    min_cvss:  float,
    days:      Optional[int],
    api_key:   Optional[str],
) -> List[Dict[str, Any]]:
    """
    Fetch CVEs from the NVD API v2.

    Parameters
    ----------
    max_cves : int
    min_cvss : float
    days : int or None
    api_key : str or None

    Returns
    -------
    list[dict]
        Parsed CVE records.
    """
    try:
        import requests
    except ImportError:
        raise ImportError("requests not installed. pip install requests")

    headers = {"User-Agent": "ACD-Framework/1.0 (MSRIT Research Project)"}
    if api_key:
        headers["apiKey"] = api_key

    delay    = RETRY_DELAY_KEY_S if api_key else RETRY_DELAY_S
    results  = []
    offset   = 0

    while len(results) < max_cves:
        params = _build_request_params(offset, min_cvss, days)

        for attempt in range(1, RETRY_LIMIT + 1):
            try:
                logger.info(
                    "Fetching NVD page — offset=%d, attempt=%d/%d",
                    offset, attempt, RETRY_LIMIT,
                )
                resp = requests.get(
                    NVD_API_BASE,
                    params  = params,
                    headers = headers,
                    timeout = 30,
                )
                resp.raise_for_status()
                data = resp.json()
                break

            except requests.exceptions.HTTPError as exc:
                if exc.response.status_code == 404:
                    logger.info("NVD returned 404 — no more pages.")
                    return results
                if exc.response.status_code == 429:
                    wait = delay * (2 ** attempt)
                    logger.warning("Rate limited — waiting %.1fs", wait)
                    time.sleep(wait)
                else:
                    logger.error("NVD HTTP error %s", exc)
                    time.sleep(delay * attempt)

            except Exception as exc:
                logger.warning(
                    "NVD request failed (attempt %d): %s", attempt, exc
                )
                time.sleep(delay * attempt)

        else:
            logger.error("All retries exhausted — stopping.")
            break

        vulnerabilities = data.get("vulnerabilities", [])
        total_results   = data.get("totalResults", 0)

        if not vulnerabilities:
            logger.info("No more CVEs returned — total fetched: %d", len(results))
            break

        for item in vulnerabilities:
            parsed = _parse_cve_item(item)
            if parsed and parsed["max_cvss"] >= min_cvss:
                results.append(parsed)
                if len(results) >= max_cves:
                    break

        logger.info(
            "Progress: %d/%d CVEs fetched (API total: %d)",
            len(results), max_cves, total_results,
        )

        offset += len(vulnerabilities)
        if offset >= total_results:
            break

        # Respect NVD rate limits
        time.sleep(delay)

    return results


def save_cves(cves: List[Dict[str, Any]], output_path: Path) -> None:
    """Save CVE list to JSON cache file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort by CVSS descending for easier browsing
    cves_sorted = sorted(cves, key=lambda c: -c.get("max_cvss", 0.0))

    output_path.write_text(
        json.dumps(cves_sorted, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        "Saved %d CVEs to %s (%.1f KB)",
        len(cves_sorted),
        output_path,
        output_path.stat().st_size / 1024,
    )


def print_summary(cves: List[Dict[str, Any]]) -> None:
    """Print a summary of downloaded CVEs to the console."""
    from collections import Counter
    sev_counts  = Counter(c["severity"] for c in cves)
    top_cwe     = Counter(cwe for c in cves for cwe in c.get("cwe_ids", []))
    cvss_values = [c["max_cvss"] for c in cves if c["max_cvss"] > 0]

    print("\n" + "="*60)
    print("NVD DOWNLOAD SUMMARY")
    print("="*60)
    print(f"  Total CVEs:         {len(cves)}")
    print(f"  By severity:")
    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"]:
        count = sev_counts.get(sev, 0)
        if count:
            bar = "█" * min(30, count // max(1, len(cves) // 30))
            print(f"    {sev:<10} {count:>5}  {bar}")
    if cvss_values:
        import statistics
        print(f"  CVSS mean:          {statistics.mean(cvss_values):.2f}")
        print(f"  CVSS max:           {max(cvss_values):.1f}")
    if top_cwe:
        print(f"  Top CWEs:")
        for cwe, count in top_cwe.most_common(5):
            print(f"    {cwe:<15} {count}")
    print("="*60 + "\n")


def main() -> None:
    args = parse_args()
    configure_logging(level="DEBUG" if args.verbose else "INFO")

    output = Path(args.output)

    # Check if cache already exists and --force not set
    if output.exists() and not args.force:
        try:
            existing = json.loads(output.read_text())
            if len(existing) >= args.max_cves // 2:
                logger.info(
                    "Cache exists with %d CVEs at %s. "
                    "Use --force to re-download.",
                    len(existing), output,
                )
                print_summary(existing)
                return
        except Exception:
            pass

    # Resolve API key
    api_key = args.api_key or os.getenv("NVD_API_KEY")
    if api_key:
        logger.info("Using NVD API key — rate limit: 50 req/30s")
    else:
        logger.info(
            "No NVD API key — rate limit: 5 req/30s. "
            "Set NVD_API_KEY env var for faster downloads."
        )

    logger.info(
        "Downloading up to %d CVEs with CVSS ≥ %.1f%s",
        args.max_cves,
        args.min_cvss,
        f" (last {args.days} days)" if args.days else "",
    )

    with Timer("NVD download") as t:
        cves = fetch_cves(
            max_cves = args.max_cves,
            min_cvss = args.min_cvss,
            days     = args.days,
            api_key  = api_key,
        )

    logger.info("Downloaded %d CVEs in %.1fs", len(cves), t.elapsed_s)
    save_cves(cves, output)
    print_summary(cves)


if __name__ == "__main__":
    main()