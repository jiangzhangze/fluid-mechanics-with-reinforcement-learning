import ray
from envs.active_gym import ActiveControl
env = ActiveControl()
action_space = env.action_space
actions = []
for i in range(1):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(reward)

