import ray
import gymnasium as gym
from ray import tune
from ray.rllib.algorithms import ppo, sac
from ray.rllib.agents.ppo import PPOTrainer
from envs.hunter import Hunter, Hunter_config
from ray.tune import register_env
#ray.shutdown()
ray.init(ignore_reinit_error=True)
config = {
    "gamma": 0.9,
    "lr": 1e-2,
    "num_workers":1,
    "disable_env_checking": True,
}
def env_creator(Hunter_config):
    return Hunter(config=Hunter_config)

register_env("hunter", env_creator)

#print(isinstance(Hunter(Hunter_config), gym.Env))
trainer = PPOTrainer(env=Hunter, config=config)

