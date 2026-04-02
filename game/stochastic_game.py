"""
game/stochastic_game.py
========================
Two-player stochastic game model for the ACD Framework.

Defines the formal game structure G = (S, A_blue, A_red, T, R, γ) and
tracks the evolving game state across an episode.

State space
-----------
The game state s ∈ S encodes the full network configuration:
    - Per-host status (clean, compromised, isolated, decoy)
    - Attacker's current believed position
    - Kill-chain stage estimate
    - Steps elapsed

State is represented as a compact integer tuple for hashing / lookup in
the transition and value tables, and as a rich ``GameState`` dataclass for
human-readable access and API serialisation.

Transition function
-------------------
T(s' | s, a_blue, a_red) is factored per host:

    For each host h:
        P(h_compromised' | h_compromised, a_blue_h, a_red_h)

The full state transition is the product of per-host transitions,
conditioned on which actions both players took.

Reward function
---------------
R(s, a_blue, a_red) = Σ_h w_h · (1 - compromised_h') - step_cost

where w_h is the value weight of host h (Op_Server0 has highest weight).

Usage
-----
    game = StochasticGame(config)
    game.reset()

    for step in episode:
        state       = game.get_state()
        blue_action = agent.predict(state.to_obs())
        red_action  = game.sample_red_action(state, attacker_type)
        next_state, reward, done = game.step(blue_action, red_action)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Host value weights (higher = more critical to protect) ─────────────────
HOST_VALUES: Dict[str, float] = {
    "User0":       1.0,
    "User1":       1.0,
    "User2":       1.0,
    "User3":       1.5,
    "User4":       1.5,
    "Enterprise0": 2.5,
    "Op_Server0":  4.0,    # Most critical — attacker's primary target
}

ALL_HOSTS = list(HOST_VALUES.keys())

# ── Host status enum ────────────────────────────────────────────────────────
class HostStatus(IntEnum):
    CLEAN      = 0
    COMPROMISED= 1
    ISOLATED   = 2
    DECOY      = 3
    RESTORED   = 4


# ── Kill-chain stage enum ────────────────────────────────────────────────────
class KillChainStage(IntEnum):
    RECONNAISSANCE  = 0
    INITIAL_ACCESS  = 1
    EXECUTION       = 2
    PERSISTENCE     = 3
    LATERAL_MOVE    = 4
    COLLECTION      = 5
    EXFILTRATION    = 6


@dataclass
class GameState:
    """
    Rich representation of the current game state.

    Attributes
    ----------
    step : int
        Current game step (0-indexed).
    host_statuses : dict[str, HostStatus]
        Current status of each host.
    attacker_position : str
        Host the attacker is currently operating on.
    kill_chain_stage : KillChainStage
        Estimated kill-chain stage of the current attack.
    blue_score : float
        Defender's cumulative reward this episode.
    red_score : float
        Attacker's cumulative reward this episode.
    n_compromised : int
        Number of currently compromised hosts.
    n_isolated : int
        Number of currently isolated hosts.
    n_decoys : int
        Number of active decoy hosts.
    is_terminal : bool
        True if the episode has ended (breach or max steps).
    """
    step:               int
    host_statuses:      Dict[str, HostStatus]
    attacker_position:  str
    kill_chain_stage:   KillChainStage
    blue_score:         float               = 0.0
    red_score:          float               = 0.0
    n_compromised:      int                 = 0
    n_isolated:         int                 = 0
    n_decoys:           int                 = 0
    is_terminal:        bool                = False

    def __post_init__(self) -> None:
        self.n_compromised = sum(
            1 for s in self.host_statuses.values()
            if s == HostStatus.COMPROMISED
        )
        self.n_isolated = sum(
            1 for s in self.host_statuses.values()
            if s == HostStatus.ISOLATED
        )
        self.n_decoys = sum(
            1 for s in self.host_statuses.values()
            if s == HostStatus.DECOY
        )

    def to_obs_vector(self) -> np.ndarray:
        """
        Convert to a compact float32 observation vector.

        Layout (n_hosts * 5 status bits + 3 global features):
            Per host: one-hot HostStatus (5 dims)
            Global: [attacker_pos_idx/n_hosts, kc_stage/6, step/max_steps]
        """
        n = len(ALL_HOSTS)
        vec = np.zeros(n * 5 + 3, dtype=np.float32)
        for i, host in enumerate(ALL_HOSTS):
            status = self.host_statuses.get(host, HostStatus.CLEAN)
            vec[i * 5 + int(status)] = 1.0

        att_idx = ALL_HOSTS.index(self.attacker_position) if self.attacker_position in ALL_HOSTS else 0
        vec[n * 5]     = att_idx / max(n - 1, 1)
        vec[n * 5 + 1] = int(self.kill_chain_stage) / 6.0
        vec[n * 5 + 2] = self.step / 100.0
        return vec

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step":              self.step,
            "host_statuses":     {h: s.name for h, s in self.host_statuses.items()},
            "attacker_position": self.attacker_position,
            "kill_chain_stage":  self.kill_chain_stage.name,
            "blue_score":        round(self.blue_score, 4),
            "red_score":         round(self.red_score, 4),
            "n_compromised":     self.n_compromised,
            "n_isolated":        self.n_isolated,
            "n_decoys":          self.n_decoys,
            "is_terminal":       self.is_terminal,
        }

    @property
    def defender_winning(self) -> bool:
        """True if defender currently has more score than attacker."""
        return self.blue_score > self.red_score

    def __hash__(self) -> int:
        return hash((
            self.step,
            tuple(sorted((h, int(s)) for h, s in self.host_statuses.items())),
            self.attacker_position,
            int(self.kill_chain_stage),
        ))


@dataclass
class GameStep:
    """
    Record of a single game step (transition).

    Used for building the transition history and training data.
    """
    step:           int
    prev_state:     GameState
    next_state:     GameState
    blue_action:    int
    red_action:     int
    reward:         float
    done:           bool
    info:           Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step":        self.step,
            "blue_action": self.blue_action,
            "red_action":  self.red_action,
            "reward":      round(self.reward, 4),
            "done":        self.done,
            "state":       self.next_state.to_dict(),
        }


class StochasticGame:
    """
    Two-player stochastic game model.

    Maintains the game state, applies player actions, computes rewards,
    and checks terminal conditions.

    The game is zero-sum: defender reward = -attacker reward (roughly).
    Practically, we use a shaped reward for the defender to accelerate
    learning, while the attacker pursues a fixed strategy.

    Parameters
    ----------
    config : dict
        Game configuration.  Keys:
            max_steps       : Episode horizon.  Default 100.
            gamma           : Discount factor.  Default 0.99.
            step_cost       : Defender step cost.  Default 0.01.
            breach_penalty  : Large negative reward for full breach.  Default 50.
            host_values     : Dict override for HOST_VALUES.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}

        self.max_steps      = cfg.get("max_steps",      100)
        self.gamma          = cfg.get("gamma",          0.99)
        self.step_cost      = cfg.get("step_cost",      0.01)
        self.breach_penalty = cfg.get("breach_penalty", 50.0)
        self._host_values   = {**HOST_VALUES, **cfg.get("host_values", {})}

        # Episode state
        self._state:    Optional[GameState] = None
        self._history:  List[GameStep]      = []
        self._rng       = np.random.default_rng(cfg.get("seed"))

        # Transition probability parameters
        self._p_compromise_base = cfg.get("p_compromise_base", 0.3)
        self._p_spread          = cfg.get("p_spread", 0.15)
        self._p_remove_success  = cfg.get("p_remove_success", 0.85)

    # ------------------------------------------------------------------ #
    # Episode management
    # ------------------------------------------------------------------ #

    def reset(self, seed: Optional[int] = None) -> GameState:
        """
        Reset the game to the initial state.

        The attacker starts at a random User host (initial access phase).
        All hosts start clean.

        Parameters
        ----------
        seed : int, optional
            Random seed for this episode.

        Returns
        -------
        GameState
            Initial game state.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._history.clear()

        # Attacker starts at a random User host
        user_hosts         = [h for h in ALL_HOSTS if h.startswith("User")]
        attacker_start     = self._rng.choice(user_hosts)

        host_statuses = {h: HostStatus.CLEAN for h in ALL_HOSTS}
        # Attacker has already achieved initial access on starting host
        host_statuses[attacker_start] = HostStatus.COMPROMISED

        self._state = GameState(
            step              = 0,
            host_statuses     = host_statuses,
            attacker_position = attacker_start,
            kill_chain_stage  = KillChainStage.INITIAL_ACCESS,
        )

        logger.debug(
            "Game reset — attacker starts at %s", attacker_start
        )
        return self._state

    def step(
        self,
        blue_action: int,
        red_action:  int,
    ) -> Tuple[GameState, float, bool]:
        """
        Advance the game by one step.

        Both players act simultaneously.  The transition function is
        applied to compute the next state, then the reward is calculated.

        Parameters
        ----------
        blue_action : int
            Defender's action index.
        red_action : int
            Attacker's action index (from AttackerModel).

        Returns
        -------
        next_state : GameState
        reward : float
            Shaped reward for the defender.
        done : bool
            True if the episode has ended.
        """
        if self._state is None:
            raise RuntimeError("Game not initialised. Call reset() first.")

        prev_state = self._state

        # Apply both actions to get next host statuses
        next_statuses = self._apply_actions(
            current_statuses  = dict(prev_state.host_statuses),
            blue_action       = blue_action,
            red_action        = red_action,
            attacker_position = prev_state.attacker_position,
        )

        # Update attacker position (lateral movement)
        next_attacker_pos, next_kc_stage = self._update_attacker_position(
            statuses     = next_statuses,
            current_pos  = prev_state.attacker_position,
            current_stage= prev_state.kill_chain_stage,
        )

        # Compute reward
        reward = self._compute_reward(
            prev_statuses = prev_state.host_statuses,
            next_statuses = next_statuses,
            blue_action   = blue_action,
        )

        # Check terminal conditions
        next_step = prev_state.step + 1
        done      = self._is_terminal(next_statuses, next_step)

        if done:
            # Large penalty if Op_Server0 gets compromised (full breach)
            if next_statuses.get("Op_Server0") == HostStatus.COMPROMISED:
                reward -= self.breach_penalty

        next_state = GameState(
            step              = next_step,
            host_statuses     = next_statuses,
            attacker_position = next_attacker_pos,
            kill_chain_stage  = next_kc_stage,
            blue_score        = prev_state.blue_score + reward,
            red_score         = prev_state.red_score  - reward,
            is_terminal       = done,
        )

        game_step = GameStep(
            step        = next_step,
            prev_state  = prev_state,
            next_state  = next_state,
            blue_action = blue_action,
            red_action  = red_action,
            reward      = reward,
            done        = done,
            info        = {
                "attacker_position": next_attacker_pos,
                "kill_chain_stage":  next_kc_stage.name,
                "n_compromised":     next_state.n_compromised,
            },
        )

        self._history.append(game_step)
        self._state = next_state

        return next_state, reward, done

    # ------------------------------------------------------------------ #
    # Transition logic
    # ------------------------------------------------------------------ #

    def _apply_actions(
        self,
        current_statuses:  Dict[str, HostStatus],
        blue_action:       int,
        red_action:        int,
        attacker_position: str,
    ) -> Dict[str, HostStatus]:
        """
        Apply both player actions and return the resulting host statuses.

        Blue actions: Remove (0-6), Restore (7-13), Isolate (14-20),
                     DeployDecoy (21-27), Monitor (28+)
        Red actions:  0=exploit_current, 1=spread, 2=persist, 3=exfiltrate
        """
        statuses = dict(current_statuses)

        # ── Apply red action first ─────────────────────────────────────
        statuses = self._apply_red_action(
            statuses, red_action, attacker_position
        )

        # ── Apply blue action ──────────────────────────────────────────
        statuses = self._apply_blue_action(statuses, blue_action)

        return statuses

    def _apply_red_action(
        self,
        statuses:          Dict[str, HostStatus],
        red_action:        int,
        attacker_position: str,
    ) -> Dict[str, HostStatus]:
        """
        Apply the attacker's action.

        Red actions:
            0 : Exploit deeper on current host (increase compromise)
            1 : Spread to adjacent clean host
            2 : Establish persistence (mark as harder to remove)
            3 : Exfiltrate (no state change — just score impact)
        """
        s = dict(statuses)

        if red_action == 0:
            # Reinforce compromise on current host
            if s.get(attacker_position) == HostStatus.CLEAN:
                s[attacker_position] = HostStatus.COMPROMISED

        elif red_action == 1:
            # Try to spread to an adjacent clean host
            clean_hosts = [
                h for h, st in s.items()
                if st == HostStatus.CLEAN and h != attacker_position
            ]
            if clean_hosts:
                target = self._rng.choice(clean_hosts)
                if self._rng.random() < self._p_spread:
                    s[target] = HostStatus.COMPROMISED

        elif red_action == 2:
            # Persistence: mark current host (harder to remove — tracked externally)
            # No state change here — modelled in reward shaping
            pass

        elif red_action == 3:
            # Exfiltration attempt — no state change
            pass

        return s

    def _apply_blue_action(
        self,
        statuses:    Dict[str, HostStatus],
        blue_action: int,
    ) -> Dict[str, HostStatus]:
        """
        Apply the defender's action.

        Action index ranges (7 hosts each):
            0–6   : Remove malware from host i
            7–13  : Restore host i to clean
            14–20 : Isolate host i
            21–27 : Deploy decoy on host i
            28+   : Monitor (no state change)
        """
        s = dict(statuses)

        if blue_action < 7:
            # Remove malware
            host = ALL_HOSTS[blue_action]
            if s.get(host) == HostStatus.COMPROMISED:
                if self._rng.random() < self._p_remove_success:
                    s[host] = HostStatus.CLEAN

        elif blue_action < 14:
            # Restore host (guaranteed success)
            host = ALL_HOSTS[blue_action - 7]
            s[host] = HostStatus.RESTORED

        elif blue_action < 21:
            # Isolate host
            host = ALL_HOSTS[blue_action - 14]
            s[host] = HostStatus.ISOLATED

        elif blue_action < 28:
            # Deploy decoy
            host = ALL_HOSTS[blue_action - 21]
            if s.get(host) == HostStatus.CLEAN:
                s[host] = HostStatus.DECOY

        # 28+: Monitor — no state change

        return s

    def _update_attacker_position(
        self,
        statuses:      Dict[str, HostStatus],
        current_pos:   str,
        current_stage: KillChainStage,
    ) -> Tuple[str, KillChainStage]:
        """
        Update the attacker's position and kill-chain stage.

        Attacker prioritises: Compromised hosts > Clean > Isolated/Decoy.
        Stage advances when attacker successfully moves deeper.
        """
        # If current position is isolated or decoy, attacker must move
        current_status = statuses.get(current_pos, HostStatus.CLEAN)

        if current_status in (HostStatus.ISOLATED, HostStatus.DECOY):
            # Attacker trapped or misdirected — move to nearest compromised
            compromised = [
                h for h, s in statuses.items()
                if s == HostStatus.COMPROMISED and h != current_pos
            ]
            if compromised:
                new_pos = compromised[0]   # Pick first (deterministic)
            else:
                # No compromised host — attacker stays put (denied)
                new_pos = current_pos
        else:
            new_pos = current_pos

        # Advance kill-chain stage based on compromised host count
        n_comp = sum(1 for s in statuses.values() if s == HostStatus.COMPROMISED)

        if n_comp == 0:
            new_stage = KillChainStage.RECONNAISSANCE
        elif n_comp <= 1:
            new_stage = KillChainStage.INITIAL_ACCESS
        elif n_comp <= 2:
            new_stage = max(current_stage, KillChainStage.EXECUTION)
        elif n_comp <= 3:
            new_stage = max(current_stage, KillChainStage.PERSISTENCE)
        elif n_comp <= 4:
            new_stage = max(current_stage, KillChainStage.LATERAL_MOVE)
        elif n_comp <= 5:
            new_stage = max(current_stage, KillChainStage.COLLECTION)
        else:
            new_stage = KillChainStage.EXFILTRATION

        return new_pos, new_stage

    def _compute_reward(
        self,
        prev_statuses: Dict[str, HostStatus],
        next_statuses: Dict[str, HostStatus],
        blue_action:   int,
    ) -> float:
        """
        Compute the defender's reward for this transition.

        Components:
            + host_value for each clean host (survival reward)
            - host_value * 2 for each newly compromised host
            + host_value * 1.5 for each host cleaned this step
            - step_cost (constant)
        """
        reward = 0.0

        for host, value in self._host_values.items():
            prev = prev_statuses.get(host, HostStatus.CLEAN)
            curr = next_statuses.get(host, HostStatus.CLEAN)

            # Survival reward for clean / isolated / restored hosts
            if curr in (HostStatus.CLEAN, HostStatus.ISOLATED, HostStatus.RESTORED):
                reward += value * 0.1

            # Penalty for newly compromised host
            if prev != HostStatus.COMPROMISED and curr == HostStatus.COMPROMISED:
                reward -= value * 2.0

            # Bonus for cleaning a compromised host
            if prev == HostStatus.COMPROMISED and curr in (
                HostStatus.CLEAN, HostStatus.RESTORED
            ):
                reward += value * 1.5

        # Step cost
        reward -= self.step_cost

        return round(reward, 4)

    def _is_terminal(
        self,
        statuses: Dict[str, HostStatus],
        step:     int,
    ) -> bool:
        """
        Check episode termination conditions.

        Terminates if:
            1. Max steps reached
            2. Op_Server0 is compromised (full network breach)
            3. All hosts are clean for 3+ consecutive steps (defender win)
        """
        if step >= self.max_steps:
            return True

        if statuses.get("Op_Server0") == HostStatus.COMPROMISED:
            return True

        return False

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> Optional[GameState]:
        """Current game state."""
        return self._state

    @property
    def history(self) -> List[GameStep]:
        """Full episode history of game steps."""
        return list(self._history)

    @property
    def episode_length(self) -> int:
        """Number of steps taken in the current episode."""
        return len(self._history)

    def get_value_map(self) -> Dict[str, float]:
        """Return the host value weights used for reward computation."""
        return dict(self._host_values)

    def get_transition_summary(self) -> Dict[str, Any]:
        """Return a compact summary of the current episode for the API."""
        if self._state is None:
            return {}
        return {
            "episode_length":  self.episode_length,
            "current_state":   self._state.to_dict(),
            "blue_score":      round(self._state.blue_score, 4),
            "red_score":       round(self._state.red_score, 4),
            "defender_winning": self._state.defender_winning,
        }

    def __repr__(self) -> str:
        step = self._state.step if self._state else 0
        return (
            f"StochasticGame("
            f"max_steps={self.max_steps}, "
            f"current_step={step})"
        )