import ray
from ray.tune.registry import register_env
from ray.tune import CLIReporter
from envs.hunter import Hunter, Hunter_config
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo import PPOConfig
from ray import tune
from ray.tune.logger import pretty_print

ray.shutdown()
ray.init(ignore_reinit_error=True)

hunter_config = PPOConfig()
def env_creator(config):
    return Hunter(config=Hunter_config)

register_env("Hunter", env_creator)
hunter_config.environment(env="Hunter", env_config=Hunter_config)
hunter_config.framework(framework="torch")
hunter_config.debugging(log_level="ERROR")
hunter_config.rollouts(
    num_rollout_workers=12,
    num_envs_per_worker=2
)
hunter_config.training(
    lr=tune.grid_search([5e-5, 2e-4]),
    train_batch_size=tune.grid_search([128, 256])
)
reporter = CLIReporter()
experiment_results = tune.run(
    run_or_experiment=hunter_config.algo_class,
    config=hunter_config.to_dict(),
    stop={
        "timesteps_total": 10000,
        "episode_reward_mean": 400
    },
    progress_reporter=reporter,
    local_dir="hunter_results",
    checkpoint_freq=10,
    checkpoint_at_end=True,
    verbose=3,
    metric="episode_reward_mean",
    mode="max"
)

best_trial = experiment_results.get_best_trial()
print("best trial:", best_trial)
best_checkpoint = experiment_results.get_best_checkpoint(
    trial=best_trial,
    metric="episode_reward_mean",
    mode="max"
)

print(f"Best checkpoint from training:{best_checkpoint}")
