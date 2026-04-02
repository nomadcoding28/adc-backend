"""
cache/keys.py
=============
All Redis cache key templates — no magic strings anywhere else.

Usage
-----
    from cache.keys import CacheKeys

    key = CacheKeys.network_topology()        # "acd:network:topology"
    key = CacheKeys.cve(cve_id="CVE-2021-44228")  # "acd:kg:cve:CVE-2021-44228"
    key = CacheKeys.agent_metrics()           # "acd:agent:metrics"
"""

from __future__ import annotations


class CacheKeys:
    """Static cache key templates with TTL recommendations."""

    # Prefix
    _P = "acd"

    # TTL constants (seconds)
    TTL_REALTIME   = 5      # Fast-changing: network state, drift score
    TTL_SHORT      = 30     # Semi-realtime: agent metrics
    TTL_MEDIUM     = 300    # 5 min: KG stats, belief state
    TTL_LONG       = 3600   # 1 hour: CVE details, technique details
    TTL_PERMANENT  = 86400  # 24 hours: KG graph, built indices

    # ── Network ───────────────────────────────────────────────────────────
    @classmethod
    def network_topology(cls) -> str:
        return f"{cls._P}:network:topology"

    @classmethod
    def host_state(cls, host_name: str) -> str:
        return f"{cls._P}:network:host:{host_name}"

    # ── Agent ─────────────────────────────────────────────────────────────
    @classmethod
    def agent_metrics(cls) -> str:
        return f"{cls._P}:agent:metrics"

    @classmethod
    def cvar_metrics(cls) -> str:
        return f"{cls._P}:agent:cvar"

    @classmethod
    def training_status(cls) -> str:
        return f"{cls._P}:agent:training_status"

    # ── Drift ─────────────────────────────────────────────────────────────
    @classmethod
    def drift_score(cls) -> str:
        return f"{cls._P}:drift:score"

    @classmethod
    def drift_history(cls) -> str:
        return f"{cls._P}:drift:history"

    # ── KG ────────────────────────────────────────────────────────────────
    @classmethod
    def kg_stats(cls) -> str:
        return f"{cls._P}:kg:stats"

    @classmethod
    def kg_graph(cls) -> str:
        return f"{cls._P}:kg:graph"

    @classmethod
    def cve(cls, cve_id: str) -> str:
        return f"{cls._P}:kg:cve:{cve_id}"

    @classmethod
    def technique(cls, technique_id: str) -> str:
        return f"{cls._P}:kg:tech:{technique_id}"

    @classmethod
    def attack_chain(cls, cve_id: str) -> str:
        return f"{cls._P}:kg:chain:{cve_id}"

    # ── Game ──────────────────────────────────────────────────────────────
    @classmethod
    def belief_state(cls) -> str:
        return f"{cls._P}:game:belief"

    @classmethod
    def game_state(cls) -> str:
        return f"{cls._P}:game:state"

    # ── Evaluation ────────────────────────────────────────────────────────
    @classmethod
    def latest_eval(cls) -> str:
        return f"{cls._P}:eval:latest"

    # ── Rate limiting ─────────────────────────────────────────────────────
    @classmethod
    def rate_limit(cls, ip: str, endpoint: str) -> str:
        return f"{cls._P}:ratelimit:{endpoint}:{ip}"