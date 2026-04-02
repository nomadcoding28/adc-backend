"""
game/
=====
Game-theoretic attacker modelling — Novelty 4 of the ACD Framework.

Models the interaction between the Blue (defender) agent and the Red
(attacker) agent as a two-player stochastic game.  The defender maintains
a Bayesian belief over the attacker's type and uses a Nash equilibrium
solver to find its best-response strategy.

Mathematical framework
----------------------
The interaction is modelled as a finite two-player zero-sum stochastic game:

    G = (S, A_blue, A_red, T, R, γ)

where:
    S             : state space (network host configurations)
    A_blue        : Blue agent action set (54 defender actions)
    A_red         : Red agent action set (attack actions per attacker type)
    T(s'|s,a,b)   : state transition function
    R(s,a,b)      : reward (positive for defender, negative for attacker)
    γ             : discount factor

The attacker type θ ∈ {Random, TargetedAPT, Adaptive} is unknown to the
defender.  The Bayesian belief updater maintains:

    P(θ | history) ∝ P(obs | θ) · P(θ)

and the Nash solver finds the minimax strategy:

    π*_blue = argmax_π min_σ V(π, σ)

Public API
----------
    from game import StochasticGame, AttackerModel, BeliefUpdater
    from game import NashSolver, GameMetrics
"""

from game.stochastic_game import StochasticGame, GameState, GameStep
from game.attacker_model import AttackerModel, AttackerType, AttackerStrategy
from game.belief_updater import BeliefUpdater, BeliefState
from game.nash_solver import NashSolver, NashEquilibrium
from game.game_metrics import GameMetrics, GameMetricsSnapshot

__all__ = [
    "StochasticGame",
    "GameState",
    "GameStep",
    "AttackerModel",
    "AttackerType",
    "AttackerStrategy",
    "BeliefUpdater",
    "BeliefState",
    "NashSolver",
    "NashEquilibrium",
    "GameMetrics",
    "GameMetricsSnapshot",
]