from envs.passive_flow_control import PassiveControl
from envs.active_flow_control import ActiveControl
import numpy as np
from tqdm import tqdm

env = ActiveControl()

num_episodes = 1
total_step = 0
episode_rewards = []

'''for ep in tqdm(range(num_episodes)):
    env.reset()
    done = False
    episode_reward = 0
    while True:
        action = env.action_space.sample()
        obs, reward, done, truncated, _ = env.step(action)
        episode_reward += reward
        print("action is")
        if done:
            episode_rewards.append(episode_reward)
            break'''
env.reset()
rewards = []
for i in range(5):
    done = False
    action = env.action_space.sample()
    obs, reward, done, truncated, _ = env.step(action)
    rewards.append(reward)
env.reset()
print(rewards)
