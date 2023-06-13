import ray
from ray.tune.registry import register_env
from ray.tune import CLIReporter
from envs.passive_flow_control import PassiveControl
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
from ray.tune.logger import pretty_print

ray.shutdown()
ray.init(ignore_reinit_error=True)

ppo_config = PPOConfig()


def env_creator(config):
    return PassiveControl(config=None)


register_env("PassiveControl", env_creator)
ppo_config.environment(env="PassiveControl")
ppo_config.framework(framework="torch")
ppo_config.debugging(log_level="ERROR")
ppo_config.rollouts(
    num_rollout_workers=36,
    num_envs_per_worker=1
)
ppo_config.training(
    lr=tune.grid_search([5e-5, 2e-4, 1e-2]),
    train_batch_size=tune.grid_search([128, 256, 512, 1024])
)

hyperparam_muactions = {

}
reporter = CLIReporter()
experiment_results = tune.run(
    run_or_experiment=ppo_config.algo_class,
    config=ppo_config.to_dict(),
    stop={
        "timesteps_total": 1,
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
