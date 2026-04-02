"""
scripts/download_attck.py
==========================
Download the MITRE ATT&CK Enterprise STIX 2.1 bundle from MITRE's
GitHub releases page and parse it into our local KG cache format.

Source
------
    https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json

The file is ~30 MB and is parsed into two lists:
    - techniques  : ATT&CK technique/sub-technique objects
    - tactics     : ATT&CK tactic (kill-chain phase) objects

Usage
-----
    # Download and parse (saves to data/kg_cache/attck_parsed.json)
    python scripts/download_attck.py

    # Use existing local STIX bundle (skip download)
    python scripts/download_attck.py --local data/kg_cache/enterprise-attack.json

    # Force re-download even if cache exists
    python scripts/download_attck.py --force

Output
------
    data/kg_cache/attck_parsed.json
    data/kg_cache/enterprise-attack.json  (raw STIX bundle)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.structlog_config import configure_logging
from utils.timer import Timer

logger = logging.getLogger(__name__)

# ── Source URL ───────────────────────────────────────────────────────────────
ATTCK_URL   = (
    "https://raw.githubusercontent.com/mitre/cti/master/"
    "enterprise-attack/enterprise-attack.json"
)
RAW_OUTPUT  = Path("data/kg_cache/enterprise-attack.json")
PARSED_OUT  = Path("data/kg_cache/attck_parsed.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download and parse MITRE ATT&CK Enterprise STIX bundle."
    )
    p.add_argument("--output",    default=str(PARSED_OUT),  help="Parsed output JSON path")
    p.add_argument("--raw-output",default=str(RAW_OUTPUT),  help="Raw STIX bundle save path")
    p.add_argument("--local",     default=None, help="Use existing local STIX bundle (skip download)")
    p.add_argument("--force",     action="store_true", help="Re-download even if cache exists")
    p.add_argument("--verbose",   action="store_true")
    return p.parse_args()


# ── Download ─────────────────────────────────────────────────────────────────

def download_bundle(save_path: Path) -> Dict[str, Any]:
    """
    Download the ATT&CK STIX bundle and save locally.

    Parameters
    ----------
    save_path : Path
        Where to save the raw STIX JSON.

    Returns
    -------
    dict
        Parsed STIX bundle.
    """
    try:
        import requests
    except ImportError:
        raise ImportError("requests not installed. pip install requests")

    logger.info("Downloading ATT&CK STIX bundle from MITRE GitHub...")
    logger.info("URL: %s", ATTCK_URL)

    resp = requests.get(ATTCK_URL, timeout=120, stream=True)
    resp.raise_for_status()

    content_length = int(resp.headers.get("content-length", 0))
    if content_length:
        logger.info("Bundle size: %.1f MB", content_length / (1024 ** 2))

    # Stream the response to file
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("wb") as f:
        downloaded = 0
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            downloaded += len(chunk)
            if content_length:
                pct = downloaded / content_length * 100
                print(f"\r  Downloading... {pct:.0f}%", end="", flush=True)

    print()  # newline after progress
    logger.info("Saved raw STIX bundle to %s (%.1f MB)", save_path, save_path.stat().st_size / 1024**2)

    return json.loads(save_path.read_text(encoding="utf-8"))


# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_bundle(bundle: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
    """
    Parse a STIX 2.1 ATT&CK bundle into techniques and tactics.

    Parameters
    ----------
    bundle : dict
        Raw STIX bundle dict.

    Returns
    -------
    tuple(techniques, tactics)
    """
    objects = bundle.get("objects", [])
    logger.info("Parsing %d STIX objects...", len(objects))

    # Build tactic id → tactic info lookup from x-mitre-tactic objects
    tactic_lookup: Dict[str, Dict] = {}
    for obj in objects:
        if obj.get("type") == "x-mitre-tactic":
            ext   = obj.get("x_mitre_shortname", "")
            t_id  = _get_external_id(obj, "mitre-attack")
            tactic_lookup[ext] = {
                "tactic_id":  t_id or ext,
                "name":       obj.get("name", ""),
                "shortname":  ext,
                "description":obj.get("description", "")[:500],
                "url":        _get_attck_url(obj),
            }

    techniques: List[Dict] = []
    tactics:    List[Dict] = list(tactic_lookup.values())

    for obj in objects:
        if obj.get("type") != "attack-pattern":
            continue

        # Skip deprecated/revoked techniques
        if obj.get("x_mitre_deprecated") or obj.get("revoked"):
            continue

        tech_id = _get_external_id(obj, "mitre-attack")
        if not tech_id:
            continue

        # Kill-chain phases map to ATT&CK tactics
        kill_chain = obj.get("kill_chain_phases", [])
        tactic_ids   = []
        tactic_names = []
        for phase in kill_chain:
            if phase.get("kill_chain_name") != "mitre-attack":
                continue
            shortname = phase.get("phase_name", "")
            if shortname in tactic_lookup:
                tac = tactic_lookup[shortname]
                tactic_ids.append(tac["tactic_id"])
                tactic_names.append(tac["name"])

        # Platforms
        platforms = obj.get("x_mitre_platforms", [])

        # Data sources
        data_sources = obj.get("x_mitre_data_sources", [])

        techniques.append({
            "technique_id":   tech_id,
            "name":           obj.get("name", ""),
            "description":    obj.get("description", "")[:1000],
            "tactic_ids":     tactic_ids,
            "tactic_names":   tactic_names,
            "platforms":      ", ".join(platforms),
            "data_sources":   data_sources[:5],
            "is_subtechnique":bool(obj.get("x_mitre_is_subtechnique", False)),
            "url":            _get_attck_url(obj),
            "detection":      obj.get("x_mitre_detection", "")[:500],
            "permissions":    obj.get("x_mitre_permissions_required", []),
        })

    logger.info(
        "Parsed %d techniques (%d sub-techniques) and %d tactics.",
        len([t for t in techniques if not t["is_subtechnique"]]),
        len([t for t in techniques if t["is_subtechnique"]]),
        len(tactics),
    )
    return techniques, tactics


def _get_external_id(obj: Dict, source_name: str) -> Optional[str]:
    """Extract an external ID (e.g. T1190) from a STIX object."""
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == source_name:
            return ref.get("external_id")
    return None


def _get_attck_url(obj: Dict) -> str:
    """Extract the ATT&CK URL from a STIX object's external references."""
    for ref in obj.get("external_references", []):
        url = ref.get("url", "")
        if "attack.mitre.org" in url:
            return url
    return ""


def save_parsed(
    techniques: List[Dict],
    tactics:    List[Dict],
    output:     Path,
) -> None:
    """Save parsed techniques and tactics to JSON cache."""
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {"techniques": techniques, "tactics": tactics}
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "Saved parsed ATT&CK data to %s (%.1f KB)",
        output, output.stat().st_size / 1024,
    )


def print_summary(techniques: List[Dict], tactics: List[Dict]) -> None:
    """Print a summary to console."""
    from collections import Counter
    tactic_dist = Counter(
        tname
        for t in techniques
        for tname in t.get("tactic_names", [])
    )

    print("\n" + "="*60)
    print("ATT&CK DOWNLOAD SUMMARY")
    print("="*60)
    print(f"  Tactics:              {len(tactics)}")
    print(f"  Techniques (parent):  {sum(1 for t in techniques if not t['is_subtechnique'])}")
    print(f"  Sub-techniques:       {sum(1 for t in techniques if t['is_subtechnique'])}")
    print(f"  Total technique objs: {len(techniques)}")
    print(f"\n  By tactic (top 10):")
    for tname, count in tactic_dist.most_common(10):
        bar = "█" * min(20, count // 2)
        print(f"    {tname:<30} {count:>3}  {bar}")
    print("="*60 + "\n")


def main() -> None:
    args = parse_args()
    configure_logging(level="DEBUG" if args.verbose else "INFO")

    output     = Path(args.output)
    raw_output = Path(args.raw_output)

    # Check cache
    if output.exists() and not args.force:
        try:
            existing = json.loads(output.read_text())
            n_tech   = len(existing.get("techniques", []))
            n_tac    = len(existing.get("tactics",    []))
            if n_tech > 100:
                logger.info(
                    "ATT&CK cache exists (%d techniques, %d tactics) at %s. "
                    "Use --force to re-download.",
                    n_tech, n_tac, output,
                )
                print_summary(existing["techniques"], existing["tactics"])
                return
        except Exception:
            pass

    with Timer("ATT&CK download + parse") as t:
        # Load from local file or download
        if args.local:
            local = Path(args.local)
            if not local.exists():
                logger.error("Local STIX file not found: %s", local)
                sys.exit(1)
            logger.info("Loading local STIX bundle from %s", local)
            bundle = json.loads(local.read_text(encoding="utf-8"))
        else:
            bundle = download_bundle(raw_output)

        techniques, tactics = parse_bundle(bundle)
        save_parsed(techniques, tactics, output)

    logger.info(
        "ATT&CK download complete — %d objects in %.1fs",
        len(techniques) + len(tactics), t.elapsed_s,
    )
    print_summary(techniques, tactics)


if __name__ == "__main__":
    main()