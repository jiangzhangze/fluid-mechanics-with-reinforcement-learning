import gym
from stable_baselines3 import PPO
from envs.active_gym import ActiveControl
import os

env = ActiveControl()
env.reset()

model = PPO.load(model_path, env=env)
episodes = 1

for episode in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action, someInfo = model.predict(obs)
        obs, reward, done, info = env.step(action)

env.close()eward, done, info = env.step(action)

env.close()