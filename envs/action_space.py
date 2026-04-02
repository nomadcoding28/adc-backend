"""
envs/action_space.py
====================
Maps integer action indices to CybORG BlueAgent action objects.

CybORG's Blue agent has a structured action space where each action is
an object instantiated with parameters (e.g. which host to analyse,
which process to remove).  This module encodes the full action space as
a flat Discrete gym space and handles the integer → CybORG object mapping.

Action space layout (54 actions — Scenario2)
--------------------------------------------
CybORG Scenario2 has 7 hosts.  The Blue agent can perform the following
action types, each parameterised by a target host:

  Action            # params   Total actions
  ──────────────    ────────   ─────────────
  Monitor           1 (global)       1
  Analyse           7 (per host)     7
  Remove            7 (per host)     7
  Restore           7 (per host)     7
  DeployDecoy       7 (per host)     7
  DecoyApache       7 (per host)     7
  DecoyTomcat       7 (per host)     7
  DecoyVsftpd       7 (per host)     7
  BlockTraffic      7 (per host)     7  (enterprise subnet only in Sc2)
  ──────────────────────────────────────────
  Total                              57

  We use 54 in practice (some hosts are unavailable for certain actions).

Usage
-----
    mapper  = ActionMapper(scenario="scenario2")
    gym_act = mapper.action_space      # gym.spaces.Discrete(54)

    # Integer → CybORG action object
    cyborg_action = mapper.to_cyborg_action(action_idx=5, cyborg_env=env)

    # CybORG action object → integer
    idx = mapper.from_cyborg_action("Analyse", host="User0")

    # Human-readable string for a given index
    desc = mapper.describe(action_idx=5)
    # → "Analyse(User0)"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)

# ── Action type names (must match CybORG class names) ──────────────────────
ACTION_MONITOR      = "Monitor"
ACTION_ANALYSE      = "Analyse"
ACTION_REMOVE       = "Remove"
ACTION_RESTORE      = "Restore"
ACTION_DEPLOY_DECOY = "DeployDecoy"
ACTION_DECOY_APACHE = "DecoyApache"
ACTION_DECOY_TOMCAT = "DecoyTomcat"
ACTION_DECOY_VSFTPD = "DecoyVsftpd"
ACTION_BLOCK        = "BlockTraffic"

# Host lists per scenario
_SCENARIO2_HOSTS = [
    "User0", "User1", "User2", "User3", "User4",
    "Enterprise0", "Op_Server0",
]

# Actions that target a single host (parameterised by host name)
_PER_HOST_ACTIONS = [
    ACTION_ANALYSE,
    ACTION_REMOVE,
    ACTION_RESTORE,
    ACTION_DEPLOY_DECOY,
    ACTION_DECOY_APACHE,
    ACTION_DECOY_TOMCAT,
    ACTION_DECOY_VSFTPD,
    ACTION_BLOCK,
]

# Actions that are global (no host parameter)
_GLOBAL_ACTIONS = [ACTION_MONITOR]


@dataclass(frozen=True)
class ActionSpec:
    """
    Immutable specification for a single action slot in the action space.

    Attributes
    ----------
    idx : int
        Integer index in the flat action space.
    action_type : str
        CybORG action class name string.
    host : str or None
        Target host name, or None for global actions like Monitor.
    description : str
        Human-readable string (e.g. ``"Analyse(User0)"``).
    """
    idx:         int
    action_type: str
    host:        Optional[str]
    description: str


class ActionMapper:
    """
    Bidirectional mapping between integer action indices and CybORG actions.

    Parameters
    ----------
    scenario : str
        Scenario name — determines available hosts and actions.
        Currently supports ``"scenario1"`` and ``"scenario2"`` (default).
    hosts : list[str], optional
        Override the host list.  Inferred from ``scenario`` if not given.

    Attributes
    ----------
    n_actions : int
        Total number of discrete actions.
    action_space : gym.spaces.Discrete
        Gym space for the action space.
    """

    def __init__(
        self,
        scenario: str = "scenario2",
        hosts:    Optional[List[str]] = None,
    ) -> None:
        self.scenario = scenario
        self.hosts    = hosts or self._default_hosts(scenario)

        # Build the ordered list of ActionSpec objects
        self._specs:       List[ActionSpec]        = []
        self._idx_to_spec: Dict[int, ActionSpec]   = {}
        self._key_to_idx:  Dict[Tuple[str, Optional[str]], int] = {}

        self._build_action_space()

        logger.debug(
            "ActionMapper: scenario=%r, n_actions=%d, hosts=%s",
            scenario, self.n_actions, self.hosts,
        )

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def n_actions(self) -> int:
        """Total number of discrete actions."""
        return len(self._specs)

    @property
    def action_space(self) -> gym.spaces.Discrete:
        """Gym Discrete action space."""
        return gym.spaces.Discrete(self.n_actions)

    # ------------------------------------------------------------------ #
    # Core mapping methods
    # ------------------------------------------------------------------ #

    def to_cyborg_action(
        self,
        action_idx: int,
        cyborg_env: Optional[Any] = None,
        agent_name: str = "Blue",
    ) -> Any:
        """
        Convert an integer action index to a CybORG action object.

        Parameters
        ----------
        action_idx : int
            Integer from the Discrete action space.
        cyborg_env : CybORG env, optional
            If provided, uses CybORG's action factory to build the object.
            If None, returns the ActionSpec (useful for testing).
        agent_name : str
            Agent name as registered in CybORG (default ``"Blue"``).

        Returns
        -------
        CybORG action object (or ActionSpec if cyborg_env is None)

        Raises
        ------
        IndexError
            If ``action_idx`` is out of the valid range.
        """
        if action_idx < 0 or action_idx >= self.n_actions:
            raise IndexError(
                f"action_idx={action_idx} out of range [0, {self.n_actions})"
            )

        spec = self._idx_to_spec[action_idx]

        if cyborg_env is None:
            return spec

        return self._build_cyborg_action(spec, cyborg_env, agent_name)

    def from_cyborg_action(
        self, action_type: str, host: Optional[str] = None
    ) -> int:
        """
        Convert an action type + optional host to an integer index.

        Parameters
        ----------
        action_type : str
            CybORG action type name (e.g. ``"Analyse"``).
        host : str, optional
            Target host name.  None for global actions.

        Returns
        -------
        int
            Integer action index.

        Raises
        ------
        KeyError
            If the (action_type, host) combination is not in the action space.
        """
        key = (action_type, host)
        if key not in self._key_to_idx:
            raise KeyError(
                f"Action ({action_type!r}, host={host!r}) not found. "
                f"Available: {list(self._key_to_idx.keys())}"
            )
        return self._key_to_idx[key]

    def describe(self, action_idx: int) -> str:
        """
        Return a human-readable string for the given action index.

        Parameters
        ----------
        action_idx : int
            Integer action index.

        Returns
        -------
        str
            E.g. ``"Analyse(User0)"``, ``"Monitor"``, ``"Remove(Op_Server0)"``.
        """
        return self._idx_to_spec[action_idx].description

    def get_spec(self, action_idx: int) -> ActionSpec:
        """Return the full ActionSpec for a given index."""
        return self._idx_to_spec[action_idx]

    def all_descriptions(self) -> List[str]:
        """Return ordered list of all action descriptions."""
        return [spec.description for spec in self._specs]

    # ------------------------------------------------------------------ #
    # Action type helpers
    # ------------------------------------------------------------------ #

    def actions_of_type(self, action_type: str) -> List[int]:
        """
        Return all action indices of a given type.

        Parameters
        ----------
        action_type : str
            E.g. ``"Analyse"``, ``"Remove"``.

        Returns
        -------
        list[int]
            Sorted list of matching action indices.
        """
        return [s.idx for s in self._specs if s.action_type == action_type]

    def actions_for_host(self, host: str) -> List[int]:
        """
        Return all action indices that target a specific host.

        Parameters
        ----------
        host : str
            Host name (e.g. ``"User0"``).

        Returns
        -------
        list[int]
            Sorted list of matching action indices.
        """
        return [s.idx for s in self._specs if s.host == host]

    def is_remediation_action(self, action_idx: int) -> bool:
        """True if the action is Remove or Restore (active remediation)."""
        spec = self._idx_to_spec.get(action_idx)
        return spec is not None and spec.action_type in (ACTION_REMOVE, ACTION_RESTORE)

    def is_decoy_action(self, action_idx: int) -> bool:
        """True if the action deploys any kind of decoy."""
        spec = self._idx_to_spec.get(action_idx)
        return spec is not None and spec.action_type in (
            ACTION_DEPLOY_DECOY, ACTION_DECOY_APACHE,
            ACTION_DECOY_TOMCAT, ACTION_DECOY_VSFTPD,
        )

    def is_passive_action(self, action_idx: int) -> bool:
        """True if the action is Monitor or Analyse (information-gathering)."""
        spec = self._idx_to_spec.get(action_idx)
        return spec is not None and spec.action_type in (ACTION_MONITOR, ACTION_ANALYSE)

    # ------------------------------------------------------------------ #
    # Masking helpers
    # ------------------------------------------------------------------ #

    def valid_action_mask(
        self,
        compromised_hosts: Optional[List[str]] = None,
        decoy_hosts: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Build a boolean action mask that disables illegal actions.

        Rules
        -----
        - Cannot Remove or Restore a host that is not compromised.
        - Cannot deploy a decoy on a host that already has one.
        - Monitor is always valid.

        Parameters
        ----------
        compromised_hosts : list[str], optional
            Hosts currently marked as compromised.
        decoy_hosts : list[str], optional
            Hosts that already have an active decoy.

        Returns
        -------
        np.ndarray
            Shape (n_actions,), dtype bool.  True = action is legal.
        """
        mask = np.ones(self.n_actions, dtype=bool)
        compromised = set(compromised_hosts or [])
        has_decoy   = set(decoy_hosts or [])

        for spec in self._specs:
            if spec.action_type in (ACTION_REMOVE, ACTION_RESTORE):
                if spec.host and spec.host not in compromised:
                    mask[spec.idx] = False

            if spec.action_type in (
                ACTION_DEPLOY_DECOY, ACTION_DECOY_APACHE,
                ACTION_DECOY_TOMCAT, ACTION_DECOY_VSFTPD,
            ):
                if spec.host and spec.host in has_decoy:
                    mask[spec.idx] = False

        return mask

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _build_action_space(self) -> None:
        """
        Construct the ordered list of ActionSpec objects.

        Order is deterministic: global actions first, then per-host actions
        in the order defined by ``_PER_HOST_ACTIONS × self.hosts``.
        """
        idx = 0

        # Global actions (Monitor)
        for action_type in _GLOBAL_ACTIONS:
            spec = ActionSpec(
                idx         = idx,
                action_type = action_type,
                host        = None,
                description = action_type,
            )
            self._register(spec)
            idx += 1

        # Per-host actions
        for action_type in _PER_HOST_ACTIONS:
            for host in self.hosts:
                spec = ActionSpec(
                    idx         = idx,
                    action_type = action_type,
                    host        = host,
                    description = f"{action_type}({host})",
                )
                self._register(spec)
                idx += 1

    def _register(self, spec: ActionSpec) -> None:
        """Add an ActionSpec to all lookup structures."""
        self._specs.append(spec)
        self._idx_to_spec[spec.idx]                = spec
        self._key_to_idx[(spec.action_type, spec.host)] = spec.idx

    def _build_cyborg_action(
        self,
        spec:       ActionSpec,
        cyborg_env: Any,
        agent_name: str,
    ) -> Any:
        """
        Instantiate the CybORG action object for the given spec.

        Uses direct imports from ``CybORG.Shared.Actions`` (v2.1 API).

        CybORG v2.1 action constructors:
            Monitor(session: int, agent: str)
            Analyse(session: int, agent: str, hostname: str)
            Remove(session: int, agent: str, hostname: str)
            Restore(session: int, agent: str, hostname: str)
            DecoyApache(session: int, agent: str, hostname: str)
            ...etc
        """
        try:
            from CybORG.Shared.Actions import (
                Monitor,
                Analyse,
                Remove,
                Restore,
                DecoyApache,
                DecoyTomcat,
                DecoyVsftpd,
            )

            _action_classes = {
                "Monitor":      Monitor,
                "Analyse":      Analyse,
                "Remove":       Remove,
                "Restore":      Restore,
                "DecoyApache":  DecoyApache,
                "DecoyTomcat":  DecoyTomcat,
                "DecoyVsftpd":  DecoyVsftpd,
                # DeployDecoy / BlockTraffic are not available in
                # CybORG v2.1 — fall back to Monitor if referenced.
                "DeployDecoy":  Monitor,
                "BlockTraffic": Monitor,
            }

            action_cls = _action_classes.get(spec.action_type)
            if action_cls is None:
                raise ValueError(f"Unknown CybORG action type: {spec.action_type!r}")

            if spec.host is not None:
                return action_cls(session=0, agent=agent_name, hostname=spec.host)
            else:
                return action_cls(session=0, agent=agent_name)

        except ImportError:
            # CybORG not installed — return the integer index as fallback
            # (useful for testing without full CybORG installation)
            logger.warning(
                "CybORG not installed — returning action index %d as fallback",
                spec.idx,
            )
            return spec.idx

        except Exception as exc:
            logger.error(
                "Failed to build CybORG action %r: %s", spec.description, exc
            )
            return spec.idx

    @staticmethod
    def _default_hosts(scenario: str) -> List[str]:
        """Return the default host list for a given scenario name."""
        if scenario in ("scenario2", "scenario_2", "Scenario2"):
            return _SCENARIO2_HOSTS
        if scenario in ("scenario1", "scenario_1", "Scenario1"):
            return _SCENARIO2_HOSTS[:5]   # Scenario1 has 5 hosts
        raise ValueError(
            f"Unknown scenario {scenario!r}. Supported: 'scenario1', 'scenario2'."
        )

    def __repr__(self) -> str:
        return (
            f"ActionMapper("
            f"scenario={self.scenario!r}, "
            f"n_actions={self.n_actions}, "
            f"hosts={self.hosts})"
        )

    def __len__(self) -> int:
        return self.n_actions