from ray.rllib.utils.pre_checks.env import check_env
from envs.hunter import Hunter, Hunter_config
from envs.base_env import BaseEnv
env = BaseEnv()

check_env(env=env)