"""
knowledge/cache.py
==================
Disk-based cache for the knowledge pipeline.

Prevents re-fetching CVE data from the NVD API and re-parsing the ATT&CK
STIX bundle on every KG rebuild.  Uses plain JSON for portability (no
binary serialisation dependencies).

Cache directory layout
----------------------
    data/kg_cache/
    ├── nvd_cves.json              Raw CVE records (from NVD API)
    ├── attck_parsed.json          Pre-parsed Tactics + Techniques
    ├── technique_index.npz        BERT embedding matrix
    ├── bert_mappings.json         CVE → Technique mapping results
    └── build_manifest.json        Last build timestamp + stats

Usage
-----
    cache = DiskCache(cache_dir="data/kg_cache")

    # Check if a key exists and is fresh
    if cache.is_fresh("nvd_cves.json", max_age_hours=24):
        records = CVEFetcher.load_from_cache(str(cache.path("nvd_cves.json")))
    else:
        records = fetcher.fetch_by_severity(...)
        CVEFetcher.save_to_cache(records, str(cache.path("nvd_cves.json")))
        cache.touch("nvd_cves.json")

    # Store arbitrary JSON-serialisable data
    cache.set("build_manifest.json", {"built_at": "2024-01-01", "n_cves": 847})
    manifest = cache.get("build_manifest.json")
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = "data/kg_cache"
_MANIFEST_KEY      = "build_manifest.json"


class DiskCache:
    """
    Simple disk-based key-value cache for the knowledge pipeline.

    Keys are file names (relative to ``cache_dir``).
    Values are JSON-serialisable Python objects.

    Parameters
    ----------
    cache_dir : str or Path
        Directory to store cached files.  Created if it does not exist.
    """

    def __init__(self, cache_dir: str = _DEFAULT_CACHE_DIR) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("DiskCache initialised at: %s", self.cache_dir)

    # ------------------------------------------------------------------ #
    # Primary interface
    # ------------------------------------------------------------------ #

    def path(self, key: str) -> Path:
        """
        Return the full filesystem path for a cache key.

        Parameters
        ----------
        key : str
            Cache key (file name, e.g. ``"nvd_cves.json"``).

        Returns
        -------
        Path
        """
        return self.cache_dir / key

    def exists(self, key: str) -> bool:
        """
        Return True if the cache file for this key exists.

        Parameters
        ----------
        key : str
            Cache key.

        Returns
        -------
        bool
        """
        return self.path(key).exists()

    def is_fresh(
        self,
        key:          str,
        max_age_hours: float = 24.0,
    ) -> bool:
        """
        Return True if the cache file exists and was modified within
        the last ``max_age_hours`` hours.

        Parameters
        ----------
        key : str
            Cache key.
        max_age_hours : float
            Maximum acceptable file age in hours.

        Returns
        -------
        bool
        """
        p = self.path(key)
        if not p.exists():
            return False

        age_s   = time.time() - p.stat().st_mtime
        age_hrs = age_s / 3600.0
        fresh   = age_hrs <= max_age_hours

        if not fresh:
            logger.debug(
                "Cache key %r is stale (%.1fh old, max=%.1fh).",
                key, age_hrs, max_age_hours,
            )

        return fresh

    def get(self, key: str) -> Optional[Any]:
        """
        Load a JSON value from the cache.

        Parameters
        ----------
        key : str
            Cache key.

        Returns
        -------
        Parsed Python object, or None if the key does not exist.
        """
        p = self.path(key)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            logger.debug("Cache hit: %r", key)
            return data
        except json.JSONDecodeError as exc:
            logger.warning("Cache file %r is corrupt: %s — ignoring.", key, exc)
            return None

    def set(self, key: str, value: Any) -> Path:
        """
        Persist a JSON-serialisable value to the cache.

        Parameters
        ----------
        key : str
            Cache key.
        value : Any
            JSON-serialisable Python object.

        Returns
        -------
        Path
            Resolved path of the written file.
        """
        p = self.path(key)
        p.write_text(
            json.dumps(value, indent=2, default=str),
            encoding="utf-8",
        )
        logger.debug("Cache write: %r (%d bytes)", key, p.stat().st_size)
        return p

    def delete(self, key: str) -> bool:
        """
        Delete a cache file.

        Parameters
        ----------
        key : str
            Cache key.

        Returns
        -------
        bool
            True if the file existed and was deleted.
        """
        p = self.path(key)
        if p.exists():
            p.unlink()
            logger.debug("Cache deleted: %r", key)
            return True
        return False

    def touch(self, key: str) -> None:
        """
        Update the modification timestamp of a cache file.

        Used to mark a file as freshly validated without rewriting it.
        """
        p = self.path(key)
        if p.exists():
            os.utime(p, None)

    def clear(self, pattern: str = "*.json") -> int:
        """
        Delete all cache files matching a glob pattern.

        Parameters
        ----------
        pattern : str
            Glob pattern.  Default ``"*.json"`` (all JSON caches).
            Use ``"*"`` to clear everything including .npz files.

        Returns
        -------
        int
            Number of files deleted.
        """
        deleted = 0
        for p in self.cache_dir.glob(pattern):
            if p.is_file():
                p.unlink()
                deleted += 1
        logger.info("Cache cleared — %d files deleted (pattern=%r)", deleted, pattern)
        return deleted

    def list_keys(self) -> List[str]:
        """
        Return a sorted list of all cache keys (file names).

        Returns
        -------
        list[str]
        """
        return sorted(p.name for p in self.cache_dir.iterdir() if p.is_file())

    # ------------------------------------------------------------------ #
    # Build manifest
    # ------------------------------------------------------------------ #

    def save_manifest(self, stats: Dict[str, Any]) -> None:
        """
        Save KG build statistics to the manifest file.

        The manifest records when the KG was last built and what it contains.
        Used by the API ``/kg/stats`` route and the freshness feature extractor.

        Parameters
        ----------
        stats : dict
            Build statistics (from ``KGBuilder.build_full()``).
        """
        manifest = {
            "built_at":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "built_at_unix": time.time(),
            **stats,
        }
        self.set(_MANIFEST_KEY, manifest)
        logger.info("Build manifest saved: %s", manifest)

    def load_manifest(self) -> Optional[Dict[str, Any]]:
        """
        Load the KG build manifest.

        Returns
        -------
        dict or None
            Manifest dict if it exists, else None.
        """
        return self.get(_MANIFEST_KEY)

    def get_build_age_hours(self) -> Optional[float]:
        """
        Return the age of the last KG build in hours.

        Returns
        -------
        float or None
            Hours since last build, or None if no manifest exists.
        """
        manifest = self.load_manifest()
        if manifest is None:
            return None
        built_at = manifest.get("built_at_unix")
        if built_at is None:
            return None
        return (time.time() - float(built_at)) / 3600.0

    # ------------------------------------------------------------------ #
    # Storage info
    # ------------------------------------------------------------------ #

    def get_storage_info(self) -> Dict[str, Any]:
        """
        Return storage usage statistics for the cache directory.

        Returns
        -------
        dict
            Keys: ``n_files``, ``total_size_mb``, ``files`` (list of
            ``{name, size_kb, modified}``).
        """
        files = []
        total_bytes = 0

        for p in sorted(self.cache_dir.iterdir()):
            if p.is_file():
                size = p.stat().st_size
                total_bytes += size
                files.append({
                    "name":       p.name,
                    "size_kb":    round(size / 1024, 1),
                    "modified":   time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ",
                        time.gmtime(p.stat().st_mtime)
                    ),
                })

        return {
            "cache_dir":      str(self.cache_dir),
            "n_files":        len(files),
            "total_size_mb":  round(total_bytes / (1024 * 1024), 2),
            "files":          files,
        }

    def __repr__(self) -> str:
        keys = self.list_keys()
        return (
            f"DiskCache("
            f"dir={self.cache_dir!r}, "
            f"n_files={len(keys)})"
        )