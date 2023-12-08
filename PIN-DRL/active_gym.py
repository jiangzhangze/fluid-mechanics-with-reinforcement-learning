import gym
from math import *
import numpy as np
from fluid_mechanics.jet_simulation import simulation
from fluid_mechanics.area import *


class ActiveControl(gym.Env):
    metadata = {"remder.modes": ["human"]}
    def __init__(self):
        super().__init__()
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
        self.points1 = probes(x_min=self.x_min, y_min=self.y_min, length=self.length, num=self.num) # 100
        self.points2 = circle_probes(0.05, 4)# 4 wall
        self.points3 = circle_probes(0.09, 48)
        self.points12 = circle_probes(0.07, 12)
        self.points13 = circle_probes(0.07, 5)
        self.points14 = add_points(self.points2, self.points13)#9 wall
        self.points4 = add_points(self.points2, self.points12)# 16 wall
        self.points5 = add_points(self.points4, self.points3)# 64 wall
        self.points6 = probes(x_min=self.x_min, y_min=self.y_min, length=self.length, num=4)#4 street
        self.points7 = probes(x_min=self.x_min, y_min=self.y_min, length=self.length, num=16)#16 street
        self.points8 = probes(x_min=self.x_min, y_min=self.y_min, length=self.length, num=64)# 64 street
        self.points9 = add_points(self.points1, self.points5)#164
        self.points10 = add_points(self.points4, self.points7)
        self.points11 = probes(x_min=self.x_min, y_min=self.y_min, length=self.length, num=9)#9 street

        self.points = self.points11
        self.episode = 0

        # drl environment parameters
        self.T = 50
        self.current_step = 0
        self.episode = 1
        # the action will be a coordinate
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0]),
                                           high=np.array([1.0, 1.0]),
                                           dtype=np.float32)
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
        #self.problem = self.build_problem(self.init_jet)
        # we take the value of the velocity probes as the observation.
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0,
                                                shape=([4 * 2 * 9 + 2, ]),
                                                dtype=np.float64)
        self.problem = self.build_problem()
        self.problem.update_jet_bc(jet=self.init_jet)
        self.problem.compute(mesh=self.problem.mesh, ft=self.problem.ft, points=self.points)

    def build_problem(self):
        problem = simulation(
            c_x=self.c_x,
            c_y=self.c_y,
            o_x=self.o_x,
            o_y=self.o_y,
            r2=self.r2,
            r=self.r
        )
        return problem

    def get_obs(self, action):
        obs = self.problem.u_probes_t
        obs = np.append(obs, action)
        #print(obs)
        return obs

    def step(self, action):
        jet_x = 0.5 * action[0]
        jet_y = 0.5 * action[1]
        jet = {
            "position": self.jet_coordinates,
            "jet_x": jet_x,
            "jet_y": jet_y
        }
        self.current_step += 1
        if self.current_step > self.T:
            self.current_step = 0
        obs = self.get_obs(action)
        u_probes_t, p_probes_t, drags, lifts = self.problem.excute(jet=jet)
        reward = 3.41 - (lifts + drags)
        done = (self.current_step == self.T)
        info = {"drags": drags,
                "lifts": lifts}
        return obs, reward, done, info

    def reset(self, *, seed: [int] = None,
        return_info: bool = False,
        options: [dict] = None,):
        self.episode += 1
        self.current_step = 0
        self.problem = self.build_problem()
        self.problem.update_jet_bc(jet=self.init_jet)
        self.problem.compute(mesh=self.problem.mesh, ft=self.problem.ft, points=self.points)
        self.problem.excute(jet=self.init_jet)
        self.init_drags = self.problem.drags
        obs = self.get_obs(action=[0, 0])
        info = {}
        return obs

    def render(self, mode="human"):
        pass

    def close(self):
        pass