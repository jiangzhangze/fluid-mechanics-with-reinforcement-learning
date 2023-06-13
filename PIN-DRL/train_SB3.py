import sys
from envs.active_gym import ActiveControl
from envs.passive_gym import PassiveControl
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, BaseCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
import stable_baselines3.common.running_mean_std
import os

'''
#TODO：
1. save best model; √
2.save checkpoint; √
3.multi-environment;
4.tune hyparameter. 
'''

models_dir = "models/PPO_ACTIVE"
model_path = f"{models_dir}/1000.zip"
logdir = "easy_logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

if __name__ == "__main__":
    env = make_vec_env(ActiveControl, n_envs=20, vec_env_cls=SubprocVecEnv)
    env = VecNormalize(env)
    env.reset()

    model = PPO("MlpPolicy", env, verbose=2, tensorboard_log=logdir, learning_rate=3e-4, n_steps=64, batch_size=16)
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path="./logs/checkpoints",
        name_prefix="PPO_model",
    )
    eval_callback = EvalCallback(env, best_model_save_path="./easy_logs/best_model",
                                 log_path="./easy_logs/evaluations", eval_freq=20,
                                 deterministic=True, render=False)


    class MyCallback(BaseCallback):
        def __init__(self, verbose=2):
            super(MyCallback, self).__init__(verbose)

        def _on_step(self) -> bool:
            current_step = self.model.num_timesteps
            print("progress:{}/{}".format(current_step, self.model._total_timesteps))

            return True


    class TensorboardCallback(BaseCallback):
        def __init__(self, verbose=2):
            super(TensorboardCallback, self).__init__(verbose)

        def _on_step(self) -> bool:
            drags = self.locals["infos"][0]["drags"]
            lifts = self.locals["infos"][0]["lifts"]
            self.logger.record("drag", drags)
            self.logger.record("lift", lifts)
            return True


    print_callback = MyCallback()
    callback_list = CallbackList([print_callback, eval_callback, checkpoint_callback])
    env.reset()
    model.learn(total_timesteps=30000, reset_num_timesteps=False, tb_log_name="PPO_ACTIVE", callback=callback_list)
