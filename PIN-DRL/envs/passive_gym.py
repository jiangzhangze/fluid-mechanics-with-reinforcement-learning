
import gym
import numpy as np
from fluid_mechanics.simulation import simulation
from fluid_mechanics.area import *
import random

Hunter_config = {
    "l": 0.01,
    "c_x": 0.2,
    "c_y": 0.2,
    "o_x": 0.3,
    "o_y": 0.2,
    "r": 0.05,
    "r2": 0,
    "x_min": 0.4,
    "y_min": 0.05,
    "length": 0.3,
    "num": 100,
    "action_space": gym.spaces.Box(low=np.array([0, 0]), high=np.array([2.2, 0.41]), dtype=np.float16),
    "observation_space": gym.spaces.Box(low=-1, high=1, shape=([1000 * 2, ]), dtype=np.float16)

}


class PassiveControl(gym.Env):
    def __init__(self, config=None):
        super(PassiveControl, self).__init__()
        self.l = 0.01
        self.c_x = 0.3
        self.c_y = 0.2
        self.o_x = 0.4
        self.o_y = 0.2
        self.r2 = 0.02
        self.r = 0.05
        self.x_min = 0.55
        self.y_min = 0.05
        self.length = 0.3
        self.num = 100
        self.points = probes(x_min=self.x_min, y_min=self.y_min, length=self.length, num=self.num)

        # drl environment parameters
        self.T = 50
        self.current_step = 0
        self.episode = 1
        # the action will be a coordinate
        self.action_space = gym.spaces.Box(low=np.array([0.05, 0.05]),
                                           high=np.array([0.5, 0.30]),
                                           dtype=np.float16)
        self.problem = self.build_problem(o_x=self.o_x, o_y=self.o_y, r2=0)
        # we take the value of the velocity probes as the observation.
        self.observation_space = gym.spaces.Box(low=-1, high=1,
                                                shape=([self.num * 2 * self.problem.num_steps + 2, ]),
                                                dtype=np.float16)

    def build_problem(self, o_x, o_y, r2=0.01):
        problem = simulation(
            c_x=self.c_x,
            c_y=self.c_y,
            o_x=o_x,
            o_y=o_y,
            r2=r2,
            r=self.r
        )
        mesh, ft = problem.generate_mesh()
        self.probes_u, probes_p, drags, lifts = problem.compute(mesh=mesh, ft=ft, points=self.points)
        print("init drag is", drags)
        return problem

    def get_obs(self, action):
        obs = np.append(self.problem.u_probes_t, action)
        return obs

    def do_actions(self, action):
        o_x = action[0]
        o_y = action[1]
        problem = self.build_problem(o_x=o_x, o_y=o_y)
        self.problem = problem
        return self.problem

    def step(self, action):
        self.current_step += 1
        if self.current_step > self.T:
            self.current_step = 0
        self.do_actions(action)
        self.mesh, self.ft = self.problem.generate_mesh()
        probes_u, probes_p, drags, lifts = self.problem.compute(mesh=self.mesh, ft=self.ft, points=self.points)
        obs = self.get_obs(action)
        reward = 1 + 10 * (self.init_drags - drags)
        done = (self.current_step == self.T) or (reward > 1)
        print("-------------------------------------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------------------------------------")
        print("obs is", obs)
        print("reward the action: (%d, %d) is %d" % (action[0], action[1], reward))
        print("-------------------------------------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------------------------------------")
        info = {"lifts": lifts}
        return obs, reward, done, info

    def reset(self):
        self.current_step = 0
        self.problem = self.build_problem(o_x=self.o_x, o_y=self.o_y, r2=0)
        self.init_drags = self.problem.drags
        obs = self.get_obs(action=[0.3, 0.2])
        info = {}
        return obs, info

    def render(self):
        pass
