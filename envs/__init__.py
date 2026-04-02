"""
envs/
=====
CybORG environment wrappers for the ACD Framework.

Every module in this package wraps or extends the CybORG cyber operations
simulation to make it compatible with Stable-Baselines3 and our custom
training pipeline.

Public API
----------
    from envs import make_env, make_vec_env
    from envs import CybORGWrapper
    from envs import ObservationProcessor, ActionMapper
    from envs import RewardShaper
    from envs import DriftInjector

Module layout
-------------
    cyborg_wrapper.py      Main Gym-compatible wrapper (primary interface)
    observation_space.py   Raw CybORG obs dict → flat 54-dim float32 tensor
    action_space.py        Integer action → CybORG BlueAgent action object
    reward_shaper.py       Custom reward signal with shaped components
    scenario_loader.py     Load / configure CybORG scenarios
    drift_injector.py      Synthetically inject concept drift for testing
    multi_env.py           Vectorised environment (SubprocVecEnv wrapper)
    env_factory.py         Build any env configuration from a config dict

Design notes
------------
- All wrappers assume CybORG ≥ 2.1 (CAGE Challenge 2 compatible).
- The observation space is fixed at 54 float32 dimensions regardless of
  scenario; ``ObservationProcessor`` handles the normalisation and padding.
- The action space is Discrete(54) — one integer per possible Blue action.
- ``CybORGWrapper`` is the single object that training scripts interact with;
  all other modules are implementation details consumed by the wrapper.
"""

from envs.cyborg_wrapper import CybORGWrapper
from envs.observation_space import ObservationProcessor
from envs.action_space import ActionMapper
from envs.reward_shaper import RewardShaper
from envs.scenario_loader import ScenarioLoader
from envs.drift_injector import DriftInjector
from envs.multi_env import make_vec_env
from envs.env_factory import make_env, EnvFactory

__all__ = [
    "CybORGWrapper",
    "ObservationProcessor",
    "ActionMapper",
    "RewardShaper",
    "ScenarioLoader",
    "DriftInjector",
    "make_vec_env",
    "make_env",
    "EnvFactory",
]