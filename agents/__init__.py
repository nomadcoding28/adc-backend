"""
agents/
=======
Reinforcement learning agents for the ACD Framework.

Public API
----------
    from agents import ACDPPOAgent, ContinualLearner, EWC, AgentRegistry

Module layout
-------------
    base_agent.py          Abstract base — all agents implement this interface
    cvar_ppo.py            CVaR-PPO implementation (Novelty 3)
    cvar_optimizer.py      CVaR evaluation and ablation utilities
    ewc.py                 Elastic Weight Consolidation (Novelty 1)
    continual_learner.py   Drift → EWC → Adapt orchestration pipeline
    adversarial_trainer.py Min-max adversarial training loop (Novelty 2)
    perturbation.py        FGSM, PGD, noise, reward-poisoning generators
    registry.py            Agent factory — build any agent from config name
"""

from agents.base_agent import BaseAgent
from agents.cvar_ppo import CVaRPPO, ACDPPOAgent
from agents.cvar_optimizer import CVaROptimizer
from agents.ewc import ElasticWeightConsolidation, ExperienceBuffer
from agents.continual_learner import ContinualLearner, ContinualLearningCallback
from agents.adversarial_trainer import AdversarialTrainer
from agents.perturbation import (
    FGSMPerturbation,
    PGDPerturbation,
    GaussianNoisePerturbation,
    RewardPoisoner,
    ObservationDelayPerturbation,
)
from agents.registry import AgentRegistry

__all__ = [
    # Base
    "BaseAgent",
    # CVaR-PPO
    "CVaRPPO",
    "ACDPPOAgent",
    "CVaROptimizer",
    # EWC / Continual Learning
    "ElasticWeightConsolidation",
    "ExperienceBuffer",
    "ContinualLearner",
    "ContinualLearningCallback",
    # Adversarial
    "AdversarialTrainer",
    "FGSMPerturbation",
    "PGDPerturbation",
    "GaussianNoisePerturbation",
    "RewardPoisoner",
    "ObservationDelayPerturbation",
    # Registry
    "AgentRegistry",
]