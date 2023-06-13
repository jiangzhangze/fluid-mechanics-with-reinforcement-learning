from fluid_mechanics.area import *
import gymnasium as gym
from math import *
import numpy as np
from fluid_mechanics.jet_simulation import simulation
from fluid_mechanics.area import *


class ActiveControl(gym.Env):
    def __init__(self, config=None):
        super(ActiveControl, self).__init__()
        self.l = 0.01
        self.c_x = 0.3
        self.c_y = 0.2
        self.o_x = 0.4
        self.o_y = 0.2
        self.r2 = 0
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
        self.action_space = gym.spaces.Box(low=np.array([-1, -1]),
                                           high=np.array([1, 1]),
                                           dtype=np.float16)
        r = 0.05
        x1 = r * cos(pi / 2) + 0.3
        y1 = r * sin(pi / 2) + 0.2
        x2 = r * cos(3 / 2 * pi) + 0.3
        y2 = r * sin(3 * pi / 2) + 0.2
        jet_positions = [(x1, y1, 0), (x2, y2, 0)]
        self.jet_coordinates = positions(jet_positions)
        self.init_jet = {
            "position": self.jet_coordinates,
            "jet_x": 0,
            "jet_y": 0
        }
        self.problem = self.build_problem(self.init_jet)
        # we take the value of the velocity probes as the observation.
        self.observation_space = gym.spaces.Box(low=-1, high=1,
                                                shape=([self.num * 2 * self.problem.num_steps + 2, ]),
                                                dtype=np.float16)

    def build_problem(self, jet):
        problem = simulation(
            c_x=self.c_x,
            c_y=self.c_y,
            o_x=self.o_x,
            o_y=self.o_y,
            r2=self.r2,
            r=self.r
        )
        mesh, ft = problem.generate_mesh()
        probes_u, probes_p, drags, lifts = problem.compute(mesh=mesh, ft=ft, points=self.points, jet=jet)
        return problem

    def get_obs(self, action):
        obs = self.problem.u_field_probes
        obs = np.append(obs, action)
        return obs

    def do_actions(self, action):
        jet_x = action[0]
        jet_y = action[1]
        jet = {
            "position": self.jet_coordinates,
            "jet_x": jet_x,
            "jet_y": jet_y
        }
        problem = self.build_problem(jet)
        self.problem = problem
        return self.problem

    def step(self, action):
        jet_x = action[0]
        jet_y = action[1]
        jet = {
            "position": self.jet_coordinates,
            "jet_x": jet_x,
            "jet_y": jet_y
        }
        self.current_step += 1
        if self.current_step > self.T:
            self.current_step = 0
        obs = self.get_obs(action)
        self.do_actions(action)
        self.mesh, self.ft = self.problem.generate_mesh()
        probes_u, probes_p, drags, lifts = self.problem.compute(mesh=self.mesh, ft=self.ft, points=self.points, jet=jet)
        reward = 10 - 10 * (lifts + drags)
        print("-------------------------------------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------------------------------------")
        print("obs is", obs)
        print("reward the action: (%d, %d) is %d" % (action[0], action[1], reward))
        print("-------------------------------------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------------------------------------")
        done = (self.current_step == self.T)
        info = {}
        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.problem = self.build_problem(self.init_jet)
        self.init_drags = self.problem.drags
        obs = self.get_obs(action=[0, 0])
        info = {}
        return obs, info

    def render(self):
        pass
