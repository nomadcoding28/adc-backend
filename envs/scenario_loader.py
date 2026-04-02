"""
envs/scenario_loader.py
========================
Load, configure, and manage CybORG scenario files.

CybORG scenarios are YAML files that define the network topology, host
configurations, agent types, and initial conditions.  This module provides
a clean interface for loading both built-in scenarios (Scenario1, Scenario2)
and custom scenario files.

Scenarios supported
-------------------
  scenario1   : 5-host enterprise network (simpler, faster training)
  scenario2   : 7-host enterprise network (CAGE Challenge 2 standard)
  custom      : Any user-supplied YAML scenario file

Usage
-----
    loader = ScenarioLoader()

    # Load built-in scenario
    cyborg_env = loader.load("scenario2")

    # Load custom scenario from file
    cyborg_env = loader.load_from_file("/path/to/my_scenario.yaml")

    # Get scenario metadata (hosts, subnets, etc.)
    meta = loader.get_metadata("scenario2")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Built-in scenario definitions ──────────────────────────────────────────

_SCENARIO_METADATA: Dict[str, Dict[str, Any]] = {
    "scenario1": {
        "name":        "Scenario1",
        "n_hosts":     5,
        "max_steps":   30,
        "subnets":     ["User", "Enterprise"],
        "hosts":       ["User0", "User1", "User2", "Enterprise0", "Op_Server0"],
        "description": "5-host network — CAGE Challenge 1 baseline",
        "difficulty":  "easy",
    },
    "scenario2": {
        "name":        "Scenario2",
        "n_hosts":     7,
        "max_steps":   100,
        "subnets":     ["User", "Enterprise", "Op"],
        "hosts":       [
            "User0", "User1", "User2", "User3", "User4",
            "Enterprise0", "Op_Server0",
        ],
        "description": "7-host network — CAGE Challenge 2 standard",
        "difficulty":  "medium",
    },
}


class ScenarioLoader:
    """
    Loads and configures CybORG simulation environments.

    Handles the differences between CybORG 2.x and 3.x APIs, and provides
    a consistent interface regardless of which version is installed.

    Parameters
    ----------
    scenario_dir : str or Path, optional
        Directory to search for custom scenario YAML files.
        Defaults to ``data/scenarios/`` relative to the project root.
    """

    def __init__(
        self, scenario_dir: Optional[str] = None
    ) -> None:
        self.scenario_dir = Path(scenario_dir or "data/scenarios")

    # ------------------------------------------------------------------ #
    # Primary load methods
    # ------------------------------------------------------------------ #

    def load(
        self,
        scenario: str = "scenario2",
        red_agent:   str = "B_lineAgent",
        max_steps:   Optional[int] = None,
        seed:        Optional[int] = None,
    ) -> Any:
        """
        Load a built-in CybORG scenario and return the environment object.

        Parameters
        ----------
        scenario : str
            Scenario name — ``"scenario1"`` or ``"scenario2"``.
        red_agent : str
            Red agent type: ``"B_lineAgent"`` (default), ``"RedMeanderAgent"``,
            or ``"SleepAgent"`` (passive — useful for debugging).
        max_steps : int, optional
            Override the scenario's default episode length.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        CybORG environment object

        Raises
        ------
        ValueError
            If the scenario name is not recognised.
        ImportError
            If CybORG is not installed.
        """
        if scenario not in _SCENARIO_METADATA:
            raise ValueError(
                f"Unknown scenario {scenario!r}. "
                f"Available: {list(_SCENARIO_METADATA.keys())}"
            )

        meta = _SCENARIO_METADATA[scenario]
        effective_max_steps = max_steps or meta["max_steps"]

        logger.info(
            "Loading CybORG %r — red_agent=%r, max_steps=%d",
            scenario, red_agent, effective_max_steps,
        )

        return self._instantiate_cyborg(
            scenario_name = meta["name"],
            red_agent     = red_agent,
            max_steps     = effective_max_steps,
            seed          = seed,
        )

    def load_from_file(
        self,
        path:      str,
        red_agent: str = "B_lineAgent",
        max_steps: int = 100,
        seed:      Optional[int] = None,
    ) -> Any:
        """
        Load a custom CybORG scenario from a YAML file.

        Parameters
        ----------
        path : str
            Absolute or relative path to the YAML scenario file.
        red_agent : str
            Red agent type string.
        max_steps : int
            Episode horizon for the custom scenario.
        seed : int, optional
            Random seed.

        Returns
        -------
        CybORG environment object
        """
        path_obj = Path(path)
        if not path_obj.exists():
            # Try relative to scenario_dir
            path_obj = self.scenario_dir / path
        if not path_obj.exists():
            raise FileNotFoundError(
                f"Scenario file not found: {path!r}. "
                f"Searched in scenario_dir={self.scenario_dir}"
            )

        logger.info("Loading custom scenario from: %s", path_obj)

        return self._instantiate_cyborg(
            scenario_path = str(path_obj),
            red_agent     = red_agent,
            max_steps     = max_steps,
            seed          = seed,
        )

    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #

    def get_metadata(self, scenario: str) -> Dict[str, Any]:
        """
        Return metadata for a named scenario.

        Parameters
        ----------
        scenario : str
            Scenario name.

        Returns
        -------
        dict
            Keys: ``name``, ``n_hosts``, ``max_steps``, ``subnets``,
            ``hosts``, ``description``, ``difficulty``.
        """
        if scenario not in _SCENARIO_METADATA:
            raise ValueError(
                f"Unknown scenario {scenario!r}. "
                f"Available: {list(_SCENARIO_METADATA.keys())}"
            )
        return dict(_SCENARIO_METADATA[scenario])

    def list_scenarios(self) -> List[str]:
        """Return list of available built-in scenario names."""
        available = list(_SCENARIO_METADATA.keys())

        # Also include any YAML files found in scenario_dir
        if self.scenario_dir.exists():
            for f in self.scenario_dir.glob("*.yaml"):
                available.append(f.stem)

        return sorted(set(available))

    def list_red_agents(self) -> List[str]:
        """Return the list of Red agent type strings available in CybORG."""
        return [
            "B_lineAgent",        # Directed attack: User → Enterprise → Op
            "RedMeanderAgent",    # Random walk attacker
            "SleepAgent",         # Passive — does nothing (debugging)
        ]

    # ------------------------------------------------------------------ #
    # CybORG instantiation (handles 2.x / 3.x API differences)
    # ------------------------------------------------------------------ #

    def _instantiate_cyborg(
        self,
        scenario_name: Optional[str] = None,
        scenario_path: Optional[str] = None,
        red_agent:     str = "B_lineAgent",
        max_steps:     int = 100,
        seed:          Optional[int] = None,
    ) -> Any:
        """
        Internal: create a CybORG environment instance.

        Tries CybORG v2.1 API first, then falls back to MockCybORG
        if CybORG is not installed (for unit testing / CI).
        """
        try:
            return self._load_cyborg_v2(
                scenario_name, scenario_path, red_agent, max_steps, seed
            )
        except ImportError as exc:
            logger.warning("CybORG import failed: %s", exc)

        logger.warning(
            "CybORG not installed — returning MockCybORG for testing."
        )
        return MockCybORG(
            scenario  = scenario_name or "scenario2",
            max_steps = max_steps,
            seed      = seed,
        )

    @staticmethod
    def _load_cyborg_v2(
        scenario_name: Optional[str],
        scenario_path: Optional[str],
        red_agent:     str,
        max_steps:     int,
        seed:          Optional[int],
    ) -> Any:
        """
        Load using CybORG v2.1 API (CAGE Challenge 2).

        CybORG v2.1 constructor signature:
            CybORG(scenario_file: str, environment="sim", env_config=None, agents=None)

        Note: ``max_steps`` is not a CybORG parameter — the wrapper manages
        episode length externally.
        """
        from CybORG.CybORG import CybORG
        from CybORG.Agents import B_lineAgent, RedMeanderAgent, SleepAgent

        _red_agent_map = {
            "B_lineAgent":      B_lineAgent,
            "RedMeanderAgent":  RedMeanderAgent,
            "SleepAgent":       SleepAgent,
        }
        red_cls = _red_agent_map.get(red_agent, B_lineAgent)

        # Resolve the scenario YAML file path
        if scenario_path:
            sc_file = scenario_path
        elif scenario_name:
            sc_file = ScenarioLoader._find_scenario_yaml(scenario_name)
        else:
            raise ValueError("Either scenario_name or scenario_path is required.")

        logger.info("CybORG v2.1 scenario file: %s", sc_file)

        # CybORG v2.1: first positional arg is the scenario YAML path
        env = CybORG(
            scenario_file=sc_file,
            agents={"Red": red_cls},
        )

        if seed is not None:
            env.set_seed(seed)

        return env

    @staticmethod
    def _find_scenario_yaml(scenario_name: str) -> str:
        """
        Resolve the absolute path to a built-in CybORG scenario YAML file.

        Uses the actual CybORG module file location (not importlib.util.find_spec,
        which may return a stub/namespace package path for dev installs).

        Fallback order:
            1. Resolve from CybORG.CybORG module __file__ → parent / Shared/Scenarios/
            2. Resolve from importlib.util.find_spec search locations
            3. Check well-known Windows dev install paths

        Raises
        ------
        FileNotFoundError
            If the scenario YAML cannot be found.
        """
        yaml_name = f"{scenario_name}.yaml"
        candidates: list[Path] = []

        # Strategy 1: From the actual CybORG.CybORG module __file__
        try:
            from CybORG.CybORG import CybORG as _CybORGClass
            module_file = getattr(_CybORGClass, "__module__", None)
            import sys
            mod = sys.modules.get("CybORG.CybORG")
            if mod and hasattr(mod, "__file__") and mod.__file__:
                cyborg_dir = Path(mod.__file__).resolve().parent
                candidates.append(cyborg_dir / "Shared" / "Scenarios" / yaml_name)
        except ImportError:
            pass

        # Strategy 2: importlib.util.find_spec
        try:
            import importlib.util
            spec = importlib.util.find_spec("CybORG")
            if spec and spec.submodule_search_locations:
                for loc in spec.submodule_search_locations:
                    candidates.append(
                        Path(loc) / "Shared" / "Scenarios" / yaml_name
                    )
        except Exception:
            pass

        # Strategy 3: Well-known dev paths (Windows)
        candidates.extend([
            Path(r"C:\work\CybORG-2.1\CybORG\Shared\Scenarios") / yaml_name,
            Path(os.path.expanduser("~")) / "CybORG" / "Shared" / "Scenarios" / yaml_name,
        ])

        for candidate in candidates:
            if candidate.exists():
                logger.info("Found scenario YAML: %s", candidate)
                return str(candidate)

        searched = [str(c) for c in candidates]
        raise FileNotFoundError(
            f"Could not find {yaml_name}. "
            f"Searched: {searched}"
        )

    def __repr__(self) -> str:
        return f"ScenarioLoader(scenario_dir={self.scenario_dir!r})"


# ══════════════════════════════════════════════════════════════════════════════
# MockCybORG — lightweight stand-in for unit testing without CybORG installed
# ══════════════════════════════════════════════════════════════════════════════

class MockCybORG:
    """
    Minimal CybORG stand-in for unit tests and CI environments where CybORG
    is not installed.

    Produces random observations and rewards with the correct structure.

    Parameters
    ----------
    scenario : str
        Scenario name — determines n_hosts and max_steps.
    max_steps : int
        Episode horizon.
    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        scenario:  str = "scenario2",
        max_steps: int = 100,
        seed:      Optional[int] = None,
    ) -> None:
        meta           = _SCENARIO_METADATA.get(scenario, _SCENARIO_METADATA["scenario2"])
        self.n_hosts   = meta["n_hosts"]
        self.max_steps = max_steps
        self._step     = 0
        self._rng      = __import__("numpy").random.default_rng(seed)
        self.hosts     = meta["hosts"]
        self._scenario = scenario

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        self._step = 0
        if seed is not None:
            self._rng = __import__("numpy").random.default_rng(seed)
        return self._make_obs()

    def step(self, action: Any) -> tuple:
        self._step += 1
        obs     = self._make_obs()
        reward  = float(self._rng.uniform(-1.0, 1.0))
        done    = self._step >= self.max_steps
        info: Dict[str, Any] = {
            "action_success": bool(self._rng.random() > 0.2),
            "step":           self._step,
        }
        return obs, reward, done, False, info

    def _make_obs(self) -> Dict[str, Any]:
        """Generate a random but structurally valid CybORG obs dict."""
        obs: Dict[str, Any] = {}
        for host in self.hosts:
            obs[host] = {
                "Compromised":        bool(self._rng.random() < 0.15),
                "IsDecoy":            bool(self._rng.random() < 0.1),
                "MaliciousProcess":   bool(self._rng.random() < 0.1),
                "ActiveConnections":  int(self._rng.integers(0, 5)),
                "Reachable":          bool(self._rng.random() > 0.3),
                "PrivilegedSession":  bool(self._rng.random() < 0.1),
            }
        return obs

    def __repr__(self) -> str:
        return f"MockCybORG(scenario={self._scenario!r}, max_steps={self.max_steps})"