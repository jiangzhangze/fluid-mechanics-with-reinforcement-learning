import ray
from ray.tune.registry import register_env
from ray.tune import CLIReporter
from envs.hunter import Hunter, Hunter_config
from envs.passive_flow_control import PassiveControl
from ray.rllib.agents import ppo
from ray.rllib.algorithms.sac.sac import SACConfig
from ray import tune
from ray.tune.logger import pretty_print

ray.shutdown()
ray.init(ignore_reinit_error=True)

sac_config = SACConfig()


def env_creator(config):
    return PassiveControl(config=None)


register_env("PassiveControl", env_creator)
sac_config.environment(env="PassiveControl")
sac_config.framework(framework="torch")
sac_config.debugging(log_level="ERROR")
sac_config.rollouts(
    num_rollout_workers=36,
    num_envs_per_worker=1
)
sac_config.training(
    gamma=tune.grid_search([1, 0.999, 0.99, 0.9]),
    lr=tune.grid_search([5e-5, 2e-4, 1e-2]),
    train_batch_size=tune.grid_search([128, 256, 512, 1024])
)
reporter = CLIReporter()
experiment_results = tune.run(
    run_or_experiment=sac_config.algo_class,
    config=sac_config.to_dict(),
    stop={
        "episode_reward_mean": 1
    },
    progress_reporter=reporter,
    local_dir="passive_control_results",
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
