from ray.rllib.algorithms.ppo import PPOConfig, PPO

import random
from envs.active_flow_control import ActiveControl
from ray import tune
from ray.tune.registry import register_env
from ray.tune import CLIReporter


def env_creator(config):
    return ActiveControl(config=None)


register_env("ActiveControl", env_creator)

config = PPOConfig()
config.environment(env="ActiveControl")
config.framework(framework="torch")
config.debugging(log_level="ERROR")
config.rollouts(
    num_rollout_workers=36,
    num_envs_per_worker=1
)
config.training(
    gamma=0.9,
    lr=1e-5,
)

reporter = CLIReporter()
reporter.add_metric_column("active_control_results/policy_reward_mean")
experiment_results = tune.run(
    run_or_experiment=config.algo_class,
    config=config.to_dict(),
    stop={
        "timesteps_total": 10000,
        "episode_reward_mean": 0
    },
    progress_reporter=reporter,
    local_dir="active_control_results",
    checkpoint_freq=10,
    checkpoint_at_end=True,
    verbose=3,
    metric="episode_reward_mean",
    mode="max",
)

best_trial = experiment_results.get_best_trial()
print("Best trial", best_trial)
best_checkpoint = experiment_results.get_best_checkpoint(trial=best_trial, metric="episode_reward_mean", mode="max")
print(f"Best checkpoint from training: {best_checkpoint}")