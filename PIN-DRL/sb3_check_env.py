from stable_baselines3.common.env_checker import check_env
from envs.active_gym import ActiveControl
env = ActiveControl()
check_env(env)
