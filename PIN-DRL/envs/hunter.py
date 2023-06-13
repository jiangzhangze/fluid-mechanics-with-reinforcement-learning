import gymnasium as gym
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


class Hunter(gym.Env):
    def __init__(self, config=None):
        super(Hunter, self).__init__()
        self.l = 0.01
        self.c_x = 0.2
        self.c_y = 0.2
        self.o_x = 0.3
        self.o_y = 0.2
        self.r2 = 0
        self.cylinder_coordinates = [(self.c_x, self.c_y),
                                     (self.c_x - self.l, self.c_y),
                                     (self.c_x + self.l, self.c_y),
                                     (self.c_x, self.c_y - self.l),
                                     (self.c_x, self.c_y + self.l)]
        self.cylinder_x = random.sample(self.cylinder_coordinates, 1)[0][0]
        self.cylinder_y = random.sample(self.cylinder_coordinates, 1)[0][1]
        self.r = 0.05
        self.x_min = 0.4
        self.y_min = 0.05
        self.length = 0.3
        self.num = 100
        self.points = probes(x_min=self.x_min, y_min=self.y_min, length=self.length, num=self.num)

        # drl environment parameters
        self.T = 50
        self.current_step = 0
        self.episode = 1
        # the action will be a coordinate
        self.action_space = gym.spaces.Box(low=np.array([0, 0]),
                                           high=np.array([2.2, 0.41]),
                                           dtype=np.float16)
        self.problem = self.build_problem(cylinder_x=self.cylinder_x, cylinder_y=self.cylinder_y)
        # we take the value of the velocity probes as the observation.
        self.observation_space = gym.spaces.Box(low=-1, high=1,
                                                shape=([self.num * 2 * self.problem.num_steps, ]),
                                                dtype=np.float16)

    def build_problem(self, cylinder_x, cylinder_y):
        problem = simulation(
            c_x=cylinder_x,
            c_y=cylinder_y,
            o_x=self.o_x,
            o_y=self.o_y,
            r2=self.r2,
            r=self.r
        )
        mesh, ft = problem.generate_mesh()
        probes_u, probes_p = problem.compute(mesh=mesh, ft=ft, points=self.points)
        np_probes_u = np.array(probes_u)
        print('init obs =', np_probes_u[0:10])

        return problem

    def get_obs(self):
        obs = self.problem.u_field_probes
        return obs

    def step(self, action):
        self.current_step += 1
        if self.current_step > self.T:
            self.current_step = 0
        obs = self.get_obs()
        target_x = action[0]
        target_y = action[1]
        self.distance_x = (target_x - self.cylinder_x) ** 2
        self.distance_y = (target_y - self.cylinder_y) ** 2
        reward = 8 - 10 * self.distance_x - 10 * self.distance_y
        done = (self.current_step == self.T) or (self.distance_x + self.distance_y == 0)
        info = {}
        print()
        '''mesh, ft = self.problem.generate_mesh()
        probes_u, probes_p = self.problem.compute(mesh=mesh, ft=ft, points=self.points)'''
        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        self.cylinder_x = random.sample(self.cylinder_coordinates, 1)[0][0]
        self.cylinder_y = random.sample(self.cylinder_coordinates, 1)[0][1]
        self.current_step = 0
        self.problem = self.build_problem(cylinder_x=self.cylinder_x, cylinder_y=self.cylinder_y)
        obs = self.get_obs()
        info = {}
        return obs, info

    def render(self):
        pass
