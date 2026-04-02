"""
envs/observation_space.py
==========================
Converts raw CybORG observation dictionaries into flat, normalised
float32 tensors suitable for neural network input.

CybORG returns observations as nested dicts keyed by host name, each
containing sub-dicts describing processes, connections, files, and user
sessions.  This module flattens, encodes, and normalises that structure
into a fixed-length vector.

Observation vector layout (54 dimensions)
------------------------------------------
  Dims  0–13  : Host status block  (7 hosts × 2 features each)
                  [0] compromised flag  (0 or 1)
                  [1] decoy flag        (0 or 1)
  Dims 14–27  : Process / service block (7 hosts × 2 features)
                  [0] malicious process detected (0 or 1)
                  [1] number of active connections (normalised 0–1)
  Dims 28–34  : Network connectivity block (7 hosts × 1 feature)
                  [0] reachable from attacker (0 or 1)
  Dims 35–41  : User session block (7 hosts × 1 feature)
                  [0] privileged session active (0 or 1)
  Dims 42–47  : Action feedback block (6 features)
                  [0] last action success (0 or 1)
                  [1] last action type (one of 9, normalised)
                  [2] attacker's last known host (normalised host index)
                  [3] steps elapsed (normalised 0–1 over max_steps)
                  [4] hosts compromised count (normalised)
                  [5] decoys deployed count (normalised)
  Dims 48–53  : Reserved / padding (zeros — future expansion)

Total: 54 float32 dimensions

Usage
-----
    processor = ObservationProcessor(n_hosts=7, max_steps=100)
    obs_vec   = processor.process(raw_cyborg_obs, step=42, last_action_result=True)
    obs_vec   = processor.normalize(obs_vec)
    gym_space = processor.observation_space   # gymnasium.spaces.Box
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

OBS_DIM = 54          # Total observation vector length (fixed)
N_HOSTS = 7           # Maximum hosts in any supported scenario
MAX_CONNECTIONS = 20  # Used to normalise connection counts

# CybORG action type strings → integer index for encoding
_ACTION_TYPE_INDEX: Dict[str, int] = {
    "Monitor":         0,
    "Analyse":         1,
    "Remove":          2,
    "Restore":         3,
    "DecoyApache":     4,
    "DecoyTomcat":     5,
    "DecoyVsftpd":     6,
    "DeployDecoy":     7,
    "BlockTraffic":    8,
}
N_ACTION_TYPES = len(_ACTION_TYPE_INDEX)

# Host names as they appear in CybORG Scenario2 (order matters)
SCENARIO2_HOSTS: List[str] = [
    "User0", "User1", "User2", "User3", "User4",
    "Enterprise0", "Op_Server0",
]


class ObservationProcessor:
    """
    Converts raw CybORG observation dicts into fixed-length float32 vectors.

    Thread-safe (stateless processing) — one instance can be shared across
    multiple vectorised environment workers.

    Parameters
    ----------
    n_hosts : int
        Number of hosts in the current scenario.  Must be ≤ N_HOSTS (7).
    max_steps : int
        Episode horizon used to normalise the elapsed-steps feature.
    host_names : list[str], optional
        Ordered list of host names as they appear in CybORG obs dicts.
        Defaults to ``SCENARIO2_HOSTS``.
    obs_dim : int
        Total observation vector length.  Default 54.
    """

    def __init__(
        self,
        n_hosts:    int = N_HOSTS,
        max_steps:  int = 100,
        host_names: Optional[List[str]] = None,
        obs_dim:    int = OBS_DIM,
    ) -> None:
        if n_hosts > N_HOSTS:
            raise ValueError(
                f"n_hosts={n_hosts} exceeds maximum supported ({N_HOSTS})."
            )

        self.n_hosts    = n_hosts
        self.max_steps  = max_steps
        self.host_names = host_names or SCENARIO2_HOSTS[:n_hosts]
        self.obs_dim    = obs_dim

        # Pre-compute the gym observation space (reused by the wrapper)
        self._obs_space = gym.spaces.Box(
            low   = 0.0,
            high  = 1.0,
            shape = (self.obs_dim,),
            dtype = np.float32,
        )

        # Track the previous raw observation for delta-feature computation
        self._prev_raw: Optional[Dict[str, Any]] = None

        logger.debug(
            "ObservationProcessor: n_hosts=%d, obs_dim=%d, max_steps=%d",
            n_hosts, obs_dim, max_steps,
        )

    # ------------------------------------------------------------------ #
    # Primary interface
    # ------------------------------------------------------------------ #

    @property
    def observation_space(self) -> gym.spaces.Box:
        """The Gym Box space corresponding to the processed observation."""
        return self._obs_space

    def process(
        self,
        raw_obs:           Dict[str, Any],
        step:              int = 0,
        last_action_success: bool = True,
        last_action_type:  str = "Monitor",
        attacker_host_idx: int = 0,
    ) -> np.ndarray:
        """
        Convert a raw CybORG observation dict to a float32 vector.

        Parameters
        ----------
        raw_obs : dict
            The observation dict returned by ``CybORG.step()``.
        step : int
            Current episode step (0-indexed).
        last_action_success : bool
            Whether the last Blue action succeeded.
        last_action_type : str
            String name of the last action (e.g. ``"Remove"``).
        attacker_host_idx : int
            Index of the host the attacker is currently believed to be on.

        Returns
        -------
        np.ndarray
            Shape (54,), dtype float32, values in [0, 1].
        """
        vec = np.zeros(self.obs_dim, dtype=np.float32)

        # ── Host status block (dims 0–13) ──────────────────────────────
        compromised_count = 0
        decoy_count = 0

        for i, hostname in enumerate(self.host_names):
            host_obs = self._extract_host_obs(raw_obs, hostname)
            base = i * 2

            compromised = float(host_obs.get("compromised", False))
            is_decoy    = float(host_obs.get("is_decoy", False))

            vec[base]     = compromised
            vec[base + 1] = is_decoy

            compromised_count += int(compromised)
            decoy_count       += int(is_decoy)

        # ── Process / service block (dims 14–27) ───────────────────────
        for i, hostname in enumerate(self.host_names):
            host_obs = self._extract_host_obs(raw_obs, hostname)
            base = 14 + i * 2

            malicious  = float(host_obs.get("malicious_process", False))
            n_conns    = min(host_obs.get("connections", 0), MAX_CONNECTIONS)
            norm_conns = n_conns / MAX_CONNECTIONS

            vec[base]     = malicious
            vec[base + 1] = norm_conns

        # ── Network reachability block (dims 28–34) ────────────────────
        for i, hostname in enumerate(self.host_names):
            host_obs = self._extract_host_obs(raw_obs, hostname)
            vec[28 + i] = float(host_obs.get("reachable", False))

        # ── User session block (dims 35–41) ────────────────────────────
        for i, hostname in enumerate(self.host_names):
            host_obs = self._extract_host_obs(raw_obs, hostname)
            vec[35 + i] = float(host_obs.get("privileged_session", False))

        # ── Action feedback block (dims 42–47) ─────────────────────────
        vec[42] = float(last_action_success)
        vec[43] = _ACTION_TYPE_INDEX.get(last_action_type, 0) / N_ACTION_TYPES
        vec[44] = min(attacker_host_idx, self.n_hosts - 1) / max(self.n_hosts - 1, 1)
        vec[45] = min(step, self.max_steps) / self.max_steps
        vec[46] = compromised_count / self.n_hosts
        vec[47] = decoy_count / self.n_hosts

        # Dims 48–53 stay zero (reserved)

        # Clamp to valid range (defensive — should already be in [0, 1])
        np.clip(vec, 0.0, 1.0, out=vec)

        self._prev_raw = raw_obs
        return vec

    def reset(self) -> None:
        """Reset internal state between episodes."""
        self._prev_raw = None

    # ------------------------------------------------------------------ #
    # Parsing helpers
    # ------------------------------------------------------------------ #

    def _extract_host_obs(
        self, raw_obs: Dict[str, Any], hostname: str
    ) -> Dict[str, Any]:
        """
        Pull host-specific features from the raw CybORG obs dict.

        CybORG's observation format changed between versions.  This method
        handles both the dict-of-dicts format and the flat key format, and
        returns a normalised sub-dict with well-known keys.

        Parameters
        ----------
        raw_obs : dict
            Full raw observation from CybORG.
        hostname : str
            Host to extract features for.

        Returns
        -------
        dict
            Standardised host feature dict with keys:
            ``compromised``, ``is_decoy``, ``malicious_process``,
            ``connections``, ``reachable``, ``privileged_session``.
        """
        result: Dict[str, Any] = {
            "compromised":        False,
            "is_decoy":           False,
            "malicious_process":  False,
            "connections":        0,
            "reachable":          False,
            "privileged_session": False,
        }

        if raw_obs is None or not isinstance(raw_obs, dict):
            return result

        # ── Format A: nested dict keyed by hostname ────────────────────
        if hostname in raw_obs:
            host_data = raw_obs[hostname]
            if isinstance(host_data, dict):
                result["compromised"]        = bool(host_data.get("Compromised", False))
                result["is_decoy"]           = bool(host_data.get("IsDecoy", False))
                result["malicious_process"]  = bool(host_data.get("MaliciousProcess", False))
                result["connections"]        = int(host_data.get("ActiveConnections", 0))
                result["reachable"]          = bool(host_data.get("Reachable", False))
                result["privileged_session"] = bool(host_data.get("PrivilegedSession", False))
            return result

        # ── Format B: flat keys with hostname prefix ───────────────────
        prefix = hostname + "_"
        for key, value in raw_obs.items():
            if key.startswith(prefix):
                field = key[len(prefix):].lower()
                if field == "compromised":
                    result["compromised"] = bool(value)
                elif field in ("isdecoy", "is_decoy"):
                    result["is_decoy"] = bool(value)
                elif field in ("maliciousprocess", "malicious_process"):
                    result["malicious_process"] = bool(value)
                elif field in ("activeconnections", "connections"):
                    result["connections"] = int(value)
                elif field == "reachable":
                    result["reachable"] = bool(value)
                elif field in ("privilegedsession", "privileged_session"):
                    result["privileged_session"] = bool(value)

        return result

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #

    def decode(self, obs_vec: np.ndarray) -> Dict[str, Any]:
        """
        Decode a processed observation vector back into a human-readable dict.

        Used by the explainability pipeline and the API ``/network`` route
        to present host states to the dashboard.

        Parameters
        ----------
        obs_vec : np.ndarray
            Processed observation vector, shape (54,).

        Returns
        -------
        dict
            Nested dict: ``{"hosts": {...}, "action_feedback": {...}}``.
        """
        if obs_vec.shape[0] != self.obs_dim:
            raise ValueError(
                f"Expected obs_dim={self.obs_dim}, got {obs_vec.shape[0]}"
            )

        hosts = {}
        for i, hostname in enumerate(self.host_names):
            hosts[hostname] = {
                "compromised":        bool(obs_vec[i * 2] > 0.5),
                "is_decoy":           bool(obs_vec[i * 2 + 1] > 0.5),
                "malicious_process":  bool(obs_vec[14 + i * 2] > 0.5),
                "active_connections": round(obs_vec[14 + i * 2 + 1] * MAX_CONNECTIONS),
                "reachable":          bool(obs_vec[28 + i] > 0.5),
                "privileged_session": bool(obs_vec[35 + i] > 0.5),
            }

        # Recover action type string from normalised index
        action_type_idx = round(obs_vec[43] * N_ACTION_TYPES)
        action_type_str = next(
            (k for k, v in _ACTION_TYPE_INDEX.items() if v == action_type_idx),
            "Unknown",
        )

        action_feedback = {
            "last_action_success": bool(obs_vec[42] > 0.5),
            "last_action_type":    action_type_str,
            "attacker_host_idx":   round(obs_vec[44] * max(self.n_hosts - 1, 1)),
            "step_fraction":       round(float(obs_vec[45]), 3),
            "compromised_ratio":   round(float(obs_vec[46]), 3),
            "decoy_ratio":         round(float(obs_vec[47]), 3),
        }

        return {"hosts": hosts, "action_feedback": action_feedback}

    def __repr__(self) -> str:
        return (
            f"ObservationProcessor("
            f"n_hosts={self.n_hosts}, "
            f"obs_dim={self.obs_dim}, "
            f"max_steps={self.max_steps})"
        )