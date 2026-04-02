"""
knowledge/kg_features.py
========================
Extracts 16-dimensional feature vectors from the knowledge graph for
augmenting the RL agent's observation space.

Why augment with KG features?
------------------------------
The raw CybORG observation tells the agent *what* is happening (which hosts
are compromised, which connections are active).  The KG tells it *why* and
*what comes next* — which CVEs are being exploited, how severe they are,
and what ATT&CK techniques the attacker is likely using.

This structured threat intelligence lets the agent make better decisions
without needing to rediscover attack patterns from scratch during training.

Feature vector layout (16 dimensions)
--------------------------------------
  Dims  0–2  : CVE severity features
                  [0] max CVSS score of active CVEs (normalised 0–10 → 0–1)
                  [1] fraction of active CVEs that are CRITICAL (≥9.0)
                  [2] mean CVSS of active CVEs (normalised)

  Dims  3–6  : ATT&CK tactic distribution (one-hot-ish)
                  [3] Initial Access    tactic active (0/1)
                  [4] Execution        tactic active (0/1)
                  [5] Persistence      tactic active (0/1)
                  [6] Lateral Movement tactic active (0/1)

  Dims  7–9  : Kill-chain stage estimate
                  [7]  Early stage  (Initial Access / Recon)
                  [8]  Mid stage    (Execution / Persistence / Privesc)
                  [9]  Late stage   (Lateral / Exfil / Impact)

  Dims 10–12 : Host-level KG signals
                  [10] fraction of compromised hosts linked to known CVEs
                  [11] max CVE score on any currently compromised host
                  [12] number of attack techniques detected (normalised)

  Dims 13–15 : Graph connectivity
                  [13] CVE→Technique edge density (normalised)
                  [14] attacker reach estimate (fraction of reachable hosts with CVEs)
                  [15] KG freshness flag (1.0 if KG built <24h ago, else 0.5)

Total: 16 float32 dimensions

Usage
-----
    from knowledge import KGFeatureExtractor, Neo4jClient

    client    = Neo4jClient.from_env()
    extractor = KGFeatureExtractor(client)

    # Get features for a given network state
    obs_decoded = obs_processor.decode(obs_vec)
    features    = extractor.extract(obs_decoded)
    # features.shape → (16,)

    # Concatenate to agent's observation
    augmented_obs = np.concatenate([obs_vec, features])   # (54+16,) = (70,)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from knowledge.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

# Tactic IDs for the 4 features in dims 3–6
_INITIAL_ACCESS_ID    = "TA0001"
_EXECUTION_ID         = "TA0002"
_PERSISTENCE_ID       = "TA0003"
_LATERAL_MOVEMENT_ID  = "TA0008"

# Early / mid / late tactic sets for kill-chain stage estimation
_EARLY_TACTICS  = {"TA0001", "TA0043"}        # Initial Access, Reconnaissance
_MID_TACTICS    = {"TA0002", "TA0003", "TA0004", "TA0005"}  # Exec, Persist, Privesc, DefEvasion
_LATE_TACTICS   = {"TA0008", "TA0009", "TA0010"}  # Lateral, Collection, Exfil

_MAX_TECHNIQUES_NORM = 20.0   # Normalise technique count by this value
_MAX_CVSS            = 10.0   # Max CVSS score (normalise to [0, 1])
_KG_FRESHNESS_TTL_S  = 86_400 # 24 hours


class KGFeatureExtractor:
    """
    Extracts KG-derived feature vectors to augment the agent's observation.

    Queries the Neo4j knowledge graph based on the current decoded network
    state (which hosts are compromised, which are reachable) and returns a
    compact float32 feature vector.

    Results are cached for ``cache_ttl_s`` seconds to avoid flooding Neo4j
    with queries during training.

    Parameters
    ----------
    client : Neo4jClient
        Connected Neo4j client.
    cache_ttl_s : float
        Feature cache TTL in seconds.  Default 5.0 (refresh every 5 steps).
    n_features : int
        Feature vector length.  Default 16.
    """

    def __init__(
        self,
        client:      Neo4jClient,
        cache_ttl_s: float = 5.0,
        n_features:  int   = 16,
    ) -> None:
        self.client      = client
        self.cache_ttl_s = cache_ttl_s
        self.n_features  = n_features

        # Feature cache
        self._cached_features:     Optional[np.ndarray] = None
        self._cache_timestamp:     float = 0.0
        self._last_state_key:      str   = ""

        # KG build timestamp (for freshness feature)
        self._kg_build_time: float = time.time()

    # ------------------------------------------------------------------ #
    # Primary interface
    # ------------------------------------------------------------------ #

    def extract(self, obs_decoded: Dict[str, Any]) -> np.ndarray:
        """
        Extract a 16-dim KG feature vector from the decoded observation.

        Parameters
        ----------
        obs_decoded : dict
            Output of ``ObservationProcessor.decode()``.  Expected keys:
            ``"hosts"`` (dict of host states) and ``"action_feedback"``.

        Returns
        -------
        np.ndarray
            Shape (16,), dtype float32, values in [0, 1].
        """
        state_key = self._make_state_key(obs_decoded)
        now       = time.monotonic()

        # Return cached result if state hasn't changed and cache is fresh
        if (
            state_key == self._last_state_key
            and (now - self._cache_timestamp) < self.cache_ttl_s
            and self._cached_features is not None
        ):
            return self._cached_features

        features = self._compute_features(obs_decoded)

        self._cached_features  = features
        self._cache_timestamp  = now
        self._last_state_key   = state_key

        return features

    def extract_batch(
        self, obs_decoded_list: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Extract features for a batch of decoded observations.

        Parameters
        ----------
        obs_decoded_list : list[dict]
            List of decoded observation dicts.

        Returns
        -------
        np.ndarray
            Shape (batch, 16), dtype float32.
        """
        return np.stack([self.extract(obs) for obs in obs_decoded_list])

    def null_features(self) -> np.ndarray:
        """
        Return a zero feature vector (used when Neo4j is unavailable).

        Returns
        -------
        np.ndarray
            Shape (16,), all zeros.
        """
        return np.zeros(self.n_features, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Feature computation
    # ------------------------------------------------------------------ #

    def _compute_features(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Query Neo4j and compute the full 16-dim feature vector.

        Falls back to zero vector if Neo4j is unavailable.
        """
        vec = np.zeros(self.n_features, dtype=np.float32)

        hosts_state = obs.get("hosts", {})
        compromised = [h for h, s in hosts_state.items() if s.get("compromised")]
        reachable   = [h for h, s in hosts_state.items() if s.get("reachable")]

        # If no hosts are compromised, return zeros (no KG signal needed)
        if not compromised:
            return vec

        try:
            # ── Query KG for active CVE information ───────────────────
            cve_data       = self._query_host_cves(compromised)
            technique_data = self._query_host_techniques(compromised)

            # ── Dims 0–2: CVE severity ─────────────────────────────────
            if cve_data:
                scores   = [c["max_cvss"] for c in cve_data if c.get("max_cvss")]
                critical = [s for s in scores if s >= 9.0]

                vec[0] = min(max(scores) / _MAX_CVSS, 1.0) if scores else 0.0
                vec[1] = len(critical) / max(len(scores), 1)
                vec[2] = min(sum(scores) / len(scores) / _MAX_CVSS, 1.0) if scores else 0.0

            # ── Dims 3–6: ATT&CK tactic flags ─────────────────────────
            active_tactics: set = set()
            for t in technique_data:
                for tactic_id in t.get("tactic_ids", []):
                    active_tactics.add(tactic_id)

            vec[3] = 1.0 if _INITIAL_ACCESS_ID   in active_tactics else 0.0
            vec[4] = 1.0 if _EXECUTION_ID         in active_tactics else 0.0
            vec[5] = 1.0 if _PERSISTENCE_ID       in active_tactics else 0.0
            vec[6] = 1.0 if _LATERAL_MOVEMENT_ID  in active_tactics else 0.0

            # ── Dims 7–9: Kill-chain stage ─────────────────────────────
            early = len(active_tactics & _EARLY_TACTICS)
            mid   = len(active_tactics & _MID_TACTICS)
            late  = len(active_tactics & _LATE_TACTICS)
            total = max(early + mid + late, 1)

            vec[7] = early / total
            vec[8] = mid   / total
            vec[9] = late  / total

            # ── Dims 10–12: Host-level KG signals ─────────────────────
            hosts_with_cves = len(set(c["host"] for c in cve_data if "host" in c))
            vec[10] = hosts_with_cves / max(len(compromised), 1)

            host_scores = {}
            for c in cve_data:
                h = c.get("host", "")
                s = c.get("max_cvss", 0.0)
                if h not in host_scores or s > host_scores[h]:
                    host_scores[h] = s

            max_compromised_cvss = max(
                (host_scores.get(h, 0.0) for h in compromised), default=0.0
            )
            vec[11] = min(max_compromised_cvss / _MAX_CVSS, 1.0)
            vec[12] = min(len(technique_data) / _MAX_TECHNIQUES_NORM, 1.0)

            # ── Dims 13–15: Graph connectivity ─────────────────────────
            n_mappings = self._count_cve_technique_edges()
            n_cves     = max(len(cve_data), 1)
            vec[13]    = min(n_mappings / (n_cves * 3), 1.0)  # avg edges per CVE / 3

            reachable_with_cves = self._count_reachable_with_cves(reachable)
            vec[14] = reachable_with_cves / max(len(reachable), 1)

            # KG freshness: 1.0 if recently built, 0.5 if older
            kg_age = time.time() - self._kg_build_time
            vec[15] = 1.0 if kg_age < _KG_FRESHNESS_TTL_S else 0.5

        except Exception as exc:
            logger.warning("KG feature extraction failed: %s — using zeros.", exc)
            return np.zeros(self.n_features, dtype=np.float32)

        return np.clip(vec, 0.0, 1.0)

    # ------------------------------------------------------------------ #
    # Neo4j queries
    # ------------------------------------------------------------------ #

    def _query_host_cves(
        self, compromised_hosts: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Query CVEs linked to the currently compromised hosts.

        Returns list of dicts with keys: host, cve_id, max_cvss, severity.
        """
        result = self.client.execute_query(
            "MATCH (c:CVE)-[:EXPLOITS]->(h:Host) "
            "WHERE h.name IN $hosts "
            "RETURN h.name AS host, c.id AS cve_id, "
            "       c.max_cvss AS max_cvss, c.severity AS severity "
            "ORDER BY c.max_cvss DESC LIMIT 50",
            hosts=compromised_hosts,
        )
        return result

    def _query_host_techniques(
        self, compromised_hosts: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Query ATT&CK techniques linked to compromised hosts via CVE mappings.

        Returns list of dicts with keys: technique_id, name, tactic_ids.
        """
        result = self.client.execute_query(
            "MATCH (c:CVE)-[:EXPLOITS]->(h:Host) "
            "WHERE h.name IN $hosts "
            "MATCH (c)-[:MAPS_TO]->(t:Technique) "
            "OPTIONAL MATCH (t)-[:BELONGS_TO]->(ta:Tactic) "
            "RETURN t.technique_id AS technique_id, t.name AS name, "
            "       collect(ta.tactic_id) AS tactic_ids",
            hosts=compromised_hosts,
        )
        return result

    def _count_cve_technique_edges(self) -> int:
        """Count total CVE→Technique MAPS_TO edges in the graph."""
        result = self.client.execute_query(
            "MATCH ()-[r:MAPS_TO]->() RETURN count(r) AS n"
        )
        return result[0]["n"] if result else 0

    def _count_reachable_with_cves(self, reachable_hosts: List[str]) -> int:
        """Count how many reachable hosts have at least one linked CVE."""
        if not reachable_hosts:
            return 0
        result = self.client.execute_query(
            "MATCH (c:CVE)-[:EXPLOITS]->(h:Host) "
            "WHERE h.name IN $hosts "
            "RETURN count(DISTINCT h) AS n",
            hosts=reachable_hosts,
        )
        return result[0]["n"] if result else 0

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_state_key(obs_decoded: Dict[str, Any]) -> str:
        """
        Build a compact string key representing the current network state.

        Used for cache invalidation — if the state hasn't changed,
        we skip re-querying Neo4j.
        """
        hosts = obs_decoded.get("hosts", {})
        parts = []
        for host, state in sorted(hosts.items()):
            parts.append(
                f"{host}:{'C' if state.get('compromised') else '_'}"
                f"{'R' if state.get('reachable') else '_'}"
            )
        return "|".join(parts)

    def mark_kg_rebuilt(self) -> None:
        """
        Update the internal KG build timestamp.

        Call this after a full KG rebuild so the freshness feature
        correctly reflects when the graph was last updated.
        """
        self._kg_build_time = time.time()
        logger.debug("KG build timestamp updated.")

    def get_feature_names(self) -> List[str]:
        """Return human-readable names for each feature dimension."""
        return [
            "max_cvss_norm",
            "critical_cve_fraction",
            "mean_cvss_norm",
            "tactic_initial_access",
            "tactic_execution",
            "tactic_persistence",
            "tactic_lateral_movement",
            "kill_chain_early",
            "kill_chain_mid",
            "kill_chain_late",
            "compromised_hosts_with_cve",
            "max_cvss_compromised",
            "n_techniques_norm",
            "cve_technique_density",
            "reachable_with_cve",
            "kg_freshness",
        ]

    def __repr__(self) -> str:
        return (
            f"KGFeatureExtractor("
            f"n_features={self.n_features}, "
            f"cache_ttl={self.cache_ttl_s}s)"
        )