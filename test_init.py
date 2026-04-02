import sys
sys.path.append('.')
from envs.env_factory import make_env
from agents.registry import AgentRegistry

config = {
    "agent_type": "cvar_ppo",
    "total_timesteps": 10000,
    "checkpoint_dir": "data/checkpoints"
}

try:
    env = make_env(config, n_envs=1)
    agent = AgentRegistry.build(env, config)
    print("Success!")
except Exception as e:
    import traceback
    with open("test_out_u8.txt", "w", encoding="utf-8") as f:
        traceback.print_exc(file=f)
