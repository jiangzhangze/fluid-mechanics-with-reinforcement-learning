import gym
from stable_baselines3 import DQN
import os

env = gym.make("CartPole-v1")
env.reset()

models_dir = "models/DQN"
model_path = f"{models_dir}/290000.zip"
logdir = "logs"
model = DQN.load(model_path, env=env)
episodes = 10

for episode in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action, someInfo = model.predict(obs)
        obs, reward, done, info = env.step(action)

env.close()