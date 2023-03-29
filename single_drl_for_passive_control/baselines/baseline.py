from envs.hunter import Hunter
import numpy as np
from tqdm import tqdm

env = Hunter()

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
for i in range(10):
    env.reset()
    done = False
    action = env.action_space.sample()
    obs, reward, done, truncated, _ = env.step(action)
    print("*************************************")
    print("action is", action)
    print("reward is", reward)
    print("*************************************")
