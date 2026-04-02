"""
game/attacker_model.py
=======================
Three attacker type strategies for the ACD stochastic game.

The defender does not know which attacker type it faces.  The three types
cover the realistic spectrum from opportunistic to sophisticated:

    Random       : Uniform random action selection — opportunistic scanner
    TargetedAPT  : Directed attack path User→Enterprise→Op_Server0
    Adaptive     : Observes defender behaviour and switches tactics

Each type is modelled as a policy π_red(a | s) — a probability distribution
over red actions given the current game state.

Red action space (4 actions)
-----------------------------
    0 : Exploit current host (reinforce compromise)
    1 : Spread laterally (target adjacent clean host)
    2 : Establish persistence (harder to remove)
    3 : Exfiltrate data (if Op_Server0 reached)

Usage
-----
    model = AttackerModel()
    action = model.sample_action(
        attacker_type  = AttackerType.TARGETED_APT,
        state          = game_state,
        belief_history = [],
    )
    print(model.get_action_probabilities(AttackerType.TARGETED_APT, state))
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from game.stochastic_game import GameState, HostStatus, KillChainStage, ALL_HOSTS

logger = logging.getLogger(__name__)

# Red action indices
RED_EXPLOIT     = 0
RED_SPREAD      = 1
RED_PERSIST     = 2
RED_EXFILTRATE  = 3
N_RED_ACTIONS   = 4

# Attack path for Targeted APT (ordered target list)
_APT_PATH = [
    "User0", "User1", "User2", "User3", "User4",
    "Enterprise0", "Op_Server0",
]


class AttackerType(str, Enum):
    """
    Enumeration of the three modelled attacker types.

    Attributes
    ----------
    RANDOM       : Uniform random attacker — low sophistication
    TARGETED_APT : Advanced Persistent Threat — follows a directed attack path
    ADAPTIVE     : Adaptive attacker — reacts to defender behaviour
    """
    RANDOM       = "Random"
    TARGETED_APT = "TargetedAPT"
    ADAPTIVE     = "Adaptive"

    @classmethod
    def all_types(cls) -> List["AttackerType"]:
        return [cls.RANDOM, cls.TARGETED_APT, cls.ADAPTIVE]


class AttackerStrategy:
    """
    Policy representation for a single attacker type.

    Stores the action probability vector π(a | s) for a given state.

    Attributes
    ----------
    attacker_type : AttackerType
    action_probs : np.ndarray
        Shape (N_RED_ACTIONS,) — probability distribution over actions.
    reasoning : str
        Human-readable reasoning behind the distribution.
    """
    __slots__ = ("attacker_type", "action_probs", "reasoning")

    def __init__(
        self,
        attacker_type: AttackerType,
        action_probs:  np.ndarray,
        reasoning:     str = "",
    ) -> None:
        self.attacker_type = attacker_type
        self.action_probs  = action_probs
        self.reasoning     = reasoning

    def sample(self, rng: np.random.Generator) -> int:
        """Sample an action from the distribution."""
        return int(rng.choice(N_RED_ACTIONS, p=self.action_probs))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attacker_type":  self.attacker_type.value,
            "action_probs":   {
                "exploit":     round(float(self.action_probs[RED_EXPLOIT]),    4),
                "spread":      round(float(self.action_probs[RED_SPREAD]),     4),
                "persist":     round(float(self.action_probs[RED_PERSIST]),    4),
                "exfiltrate":  round(float(self.action_probs[RED_EXFILTRATE]), 4),
            },
            "reasoning": self.reasoning,
        }

    def __repr__(self) -> str:
        dominant = ["exploit","spread","persist","exfiltrate"][
            int(np.argmax(self.action_probs))
        ]
        return (
            f"AttackerStrategy({self.attacker_type.value}, "
            f"dominant={dominant}, p={self.action_probs.max():.2f})"
        )


class AttackerModel:
    """
    Encodes the three attacker type policies.

    Each attacker type has a conditional policy π_θ(a | s) that maps
    game states to action probability distributions.

    Parameters
    ----------
    config : dict, optional
        Configuration overrides.  Keys:
            apt_spread_prob   : P(spread) for APT when no adjacent compromised.
            random_temp       : Softmax temperature for Random attacker.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._apt_spread_prob = cfg.get("apt_spread_prob", 0.7)
        self._rng = np.random.default_rng(cfg.get("seed"))

        # Behaviour history tracking for adaptive attacker
        self._defender_action_counts: Dict[int, int] = {}

    # ------------------------------------------------------------------ #
    # Primary interface
    # ------------------------------------------------------------------ #

    def get_strategy(
        self,
        attacker_type:   AttackerType,
        state:           GameState,
        belief_history:  Optional[List[Dict[str, Any]]] = None,
    ) -> AttackerStrategy:
        """
        Compute the action probability distribution for the given attacker type
        and current game state.

        Parameters
        ----------
        attacker_type : AttackerType
        state : GameState
        belief_history : list[dict], optional
            History of defender belief states (used by Adaptive attacker).

        Returns
        -------
        AttackerStrategy
        """
        if attacker_type == AttackerType.RANDOM:
            return self._random_strategy(state)

        elif attacker_type == AttackerType.TARGETED_APT:
            return self._apt_strategy(state)

        elif attacker_type == AttackerType.ADAPTIVE:
            return self._adaptive_strategy(state, belief_history or [])

        raise ValueError(f"Unknown attacker type: {attacker_type!r}")

    def sample_action(
        self,
        attacker_type:  AttackerType,
        state:          GameState,
        belief_history: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """
        Sample a single red action from the attacker's policy.

        Parameters
        ----------
        attacker_type : AttackerType
        state : GameState
        belief_history : list[dict], optional

        Returns
        -------
        int
            Red action index (0–3).
        """
        strategy = self.get_strategy(attacker_type, state, belief_history)
        return strategy.sample(self._rng)

    def get_action_probabilities(
        self,
        attacker_type: AttackerType,
        state:         GameState,
    ) -> Dict[str, float]:
        """
        Return the action probability dict for dashboard display.

        Returns
        -------
        dict
            Keys: ``exploit``, ``spread``, ``persist``, ``exfiltrate``.
        """
        strategy = self.get_strategy(attacker_type, state)
        return strategy.to_dict()["action_probs"]

    def record_defender_action(self, blue_action: int) -> None:
        """
        Record a defender action for the adaptive attacker's counter-strategy.

        Called by the StochasticGame after each step.
        """
        self._defender_action_counts[blue_action] = (
            self._defender_action_counts.get(blue_action, 0) + 1
        )

    # ------------------------------------------------------------------ #
    # Strategy implementations
    # ------------------------------------------------------------------ #

    def _random_strategy(self, state: GameState) -> AttackerStrategy:
        """
        Random attacker: approximately uniform with slight bias toward spread.

        Reasoning: opportunistic attacker — doesn't have a specific target,
        just looking for any foothold to establish.
        """
        probs = np.array([0.30, 0.35, 0.20, 0.15], dtype=np.float32)

        # If already on Op_Server0, strongly prefer exfiltration
        if state.attacker_position == "Op_Server0":
            probs = np.array([0.10, 0.10, 0.20, 0.60], dtype=np.float32)

        return AttackerStrategy(
            attacker_type = AttackerType.RANDOM,
            action_probs  = probs,
            reasoning     = (
                "Random attacker: uniform spread with slight lateral bias. "
                "No specific target — opportunistic exploitation."
            ),
        )

    def _apt_strategy(self, state: GameState) -> AttackerStrategy:
        """
        Targeted APT: follows the directed attack path User→Enterprise→Op_Server0.

        Reasoning: APT knows the network topology and has a specific target
        (Op_Server0).  It moves methodically through the kill chain.
        """
        current_pos = state.attacker_position
        stage       = state.kill_chain_stage
        statuses    = state.host_statuses

        # Find next target on the APT path
        try:
            current_idx = _APT_PATH.index(current_pos)
        except ValueError:
            current_idx = 0

        next_target_idx  = min(current_idx + 1, len(_APT_PATH) - 1)
        next_target      = _APT_PATH[next_target_idx]
        next_target_status = statuses.get(next_target, HostStatus.CLEAN)

        reasoning = f"APT stage={stage.name}, target={next_target}"

        # ── Decision logic ─────────────────────────────────────────────
        if current_pos == "Op_Server0":
            # At target — exfiltrate
            probs = np.array([0.05, 0.05, 0.15, 0.75], dtype=np.float32)
            reasoning += " — at target, exfiltrating"

        elif next_target_status == HostStatus.ISOLATED:
            # Next hop is isolated — wait (persist) and look for alternative
            probs = np.array([0.10, 0.50, 0.30, 0.10], dtype=np.float32)
            reasoning += f" — {next_target} isolated, seeking alternate path"

        elif next_target_status == HostStatus.DECOY:
            # Next hop is a decoy — APT detects it and pivots
            # (APT is sophisticated enough to detect decoys with some probability)
            if self._rng.random() < 0.4:
                # Detected decoy — persist and wait
                probs = np.array([0.10, 0.20, 0.60, 0.10], dtype=np.float32)
                reasoning += f" — {next_target} identified as decoy, persisting"
            else:
                # Fooled by decoy — moves toward it
                probs = np.array([0.20, self._apt_spread_prob, 0.05, 0.05], dtype=np.float32)
                probs /= probs.sum()
                reasoning += f" — moving toward {next_target} (decoy not detected)"

        elif stage in (KillChainStage.INITIAL_ACCESS, KillChainStage.EXECUTION):
            # Early stage — exploit current host then spread
            probs = np.array([0.35, self._apt_spread_prob, 0.10, 0.05], dtype=np.float32)
            probs /= probs.sum()
            reasoning += " — early stage, exploiting then spreading"

        elif stage == KillChainStage.PERSISTENCE:
            # Mid stage — prioritise persistence then lateral movement
            probs = np.array([0.15, 0.40, 0.40, 0.05], dtype=np.float32)
            reasoning += " — establishing persistence before moving"

        else:
            # Late stage — rapid lateral movement toward Op_Server0
            probs = np.array([0.10, 0.65, 0.15, 0.10], dtype=np.float32)
            reasoning += f" — lateral movement toward {next_target}"

        return AttackerStrategy(
            attacker_type = AttackerType.TARGETED_APT,
            action_probs  = probs.astype(np.float32),
            reasoning     = reasoning,
        )

    def _adaptive_strategy(
        self,
        state:          GameState,
        belief_history: List[Dict[str, Any]],
    ) -> AttackerStrategy:
        """
        Adaptive attacker: counter-strategy based on observed defender behaviour.

        Analyses the defender's most frequent actions and selects the
        strategy that is hardest to counter:
            - If defender frequently removes malware → use persistence
            - If defender frequently deploys decoys → use reconnaissance
            - If defender frequently isolates → spread before isolation

        Reasoning: sophisticated attacker that observes defender TTPs and adapts.
        """
        # Count defender action types
        remove_count  = sum(
            v for k, v in self._defender_action_counts.items() if k < 7
        )
        isolate_count = sum(
            v for k, v in self._defender_action_counts.items() if 14 <= k < 21
        )
        decoy_count   = sum(
            v for k, v in self._defender_action_counts.items() if 21 <= k < 28
        )
        total         = max(remove_count + isolate_count + decoy_count, 1)

        remove_rate  = remove_count  / total
        isolate_rate = isolate_count / total
        decoy_rate   = decoy_count   / total

        reasoning = (
            f"Adaptive: remove_rate={remove_rate:.2f}, "
            f"isolate_rate={isolate_rate:.2f}, "
            f"decoy_rate={decoy_rate:.2f}"
        )

        # Counter-strategy selection
        if isolate_rate > 0.3:
            # Defender uses isolation a lot → spread quickly before isolation
            probs = np.array([0.15, 0.60, 0.15, 0.10], dtype=np.float32)
            reasoning += " → counter: rapid spread before isolation"

        elif decoy_rate > 0.25:
            # Defender uses many decoys → exploit current host (avoid unknown hosts)
            probs = np.array([0.55, 0.15, 0.20, 0.10], dtype=np.float32)
            reasoning += " → counter: deep exploitation on known-safe host"

        elif remove_rate > 0.3:
            # Defender removes malware a lot → establish persistence first
            probs = np.array([0.10, 0.25, 0.55, 0.10], dtype=np.float32)
            reasoning += " → counter: persistence before spreading"

        else:
            # No strong defender pattern → balanced APT-like strategy
            apt = self._apt_strategy(state)
            return AttackerStrategy(
                attacker_type = AttackerType.ADAPTIVE,
                action_probs  = apt.action_probs,
                reasoning     = f"Adaptive (no pattern detected): {apt.reasoning}",
            )

        return AttackerStrategy(
            attacker_type = AttackerType.ADAPTIVE,
            action_probs  = probs.astype(np.float32),
            reasoning     = reasoning,
        )

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #

    def get_all_strategies(
        self, state: GameState
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return strategy dicts for all three attacker types given a state.

        Used by the dashboard Game Model panel.
        """
        return {
            at.value: self.get_strategy(at, state).to_dict()
            for at in AttackerType.all_types()
        }

    def likelihood_of_observation(
        self,
        observation:   Dict[str, Any],
        attacker_type: AttackerType,
        state:         GameState,
    ) -> float:
        """
        P(observation | attacker_type, state) — observation likelihood.

        Used by the Bayesian belief updater.

        Parameters
        ----------
        observation : dict
            Observed attacker behaviour (action taken, hosts affected).
        attacker_type : AttackerType
        state : GameState

        Returns
        -------
        float
            Likelihood value in (0, 1].
        """
        strategy = self.get_strategy(attacker_type, state)
        observed_action = observation.get("red_action", -1)

        if observed_action < 0 or observed_action >= N_RED_ACTIONS:
            # Unknown action — uniform likelihood
            return 1.0 / N_RED_ACTIONS

        return float(strategy.action_probs[observed_action]) + 1e-8

    def reset(self) -> None:
        """Reset the defender action history (call at episode start)."""
        self._defender_action_counts.clear()

    def __repr__(self) -> str:
        return (
            f"AttackerModel("
            f"types={[t.value for t in AttackerType.all_types()]}, "
            f"defender_actions_tracked={sum(self._defender_action_counts.values())})"
        )