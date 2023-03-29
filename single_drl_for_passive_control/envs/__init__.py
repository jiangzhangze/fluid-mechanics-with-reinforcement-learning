from ray.tune.registry import register_env
from envs.hunter import Hunter


def env_creator(config):
    return Hunter(config)

register_env("Hunter", env_creator)
