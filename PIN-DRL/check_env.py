from ray.rllib.utils.pre_checks.env import check_env
from envs.passive_flow_control import PassiveControl
from envs.base_env import BaseEnv
env = PassiveControl()

check_env(env=env)