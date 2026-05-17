from gymnasium import register
from dyn_rl_benchmarks.envs.drawbridge import DrawbridgeEnv

register(
    id="Drawbridge-v1",
    entry_point="dyn_rl_benchmarks.envs:DrawbridgeEnv",
    max_episode_steps=DrawbridgeEnv.max_episode_length,
)

register(id="Tennis2D-v1", entry_point="dyn_rl_benchmarks.envs:Tennis2DEnv")

register(
    id="Tennis2DDenseReward-v1",
    entry_point="dyn_rl_benchmarks.envs:Tennis2DDenseRewardEnv",
)

register(id="Platforms-v1", entry_point="dyn_rl_benchmarks.envs:PlatformsEnv")

register(id="PlatformsTime-v1", entry_point="dyn_rl_benchmarks.envs:PlatformsTimeEnv")
