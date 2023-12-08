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
        self.points1 = probes(x_min=self.x_min, y_min=self.y_min, length=self.length, num=100)#100
        self.points2 = circle_probes(0.05, 4)#4
        self.points3 = circle_probes(0.07, 20)#20
        self.points4 = circle_probes(0.09, 40)#40
        self.points5 = add_points(self.points3, self.points4)#60
        self.points6 = rectangular(low=[0.2, 0.05], high=[0.5, 0.15], num=400) #400
        self.points7 = np.concatenate([self.points2, self.points3], axis=1)#24
        self.points8 = add_points(self.points5, self.points4)#64
        self.points9 = add_points(self.points1, self.points2)#104
        self.points10 = add_points(self.points6, self.points5)#164
        self.points11 = probes(x_min=self.x_min, y_min=self.y_min, length=self.length, num=64)# street 64
        self.points12 = probes(x_min=self.x_min, y_min=self.y_min, length=self.length, num=4)# street 4
        self.points = self.points11

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
                                                shape=([4 * 2 * 64 + 2, ]),
                                                dtype=np.float64)
        self.problem = self.build_problem(self.init_jet)

    def build_problem(self, jet):
        problem = simulation(
            c_x=self.c_x,
            c_y=self.c_y,
            o_x=self.o_x,
            o_y=self.o_y,
            r2=self.r2,
            r=self.r
        )
        problem.update_jet_bc(jet)
        probes_u, probes_p, drags, lifts = problem.compute(mesh=problem.mesh, ft=problem.ft, points=self.points)
        return problem

    def get_obs(self, action):
        obs = self.problem.u_probes_t
        obs = np.append(obs, action)
        #print(obs)
        return obs

    def step(self, action):
        jet_x = 5 * action[0]
        jet_y = 5 * action[1]
        jet = {
            "position": self.jet_coordinates,
            "jet_x": jet_x,
            "jet_y": jet_y
        }
        self.current_step += 1
        if self.current_step > self.T:
            self.current_step = 0
        obs = self.get_obs(action)
        self.problem.update_jet_bc(jet=jet)
        u_probes_t, p_probes_t, drags, lifts = self.problem.compute(mesh=self.problem.mesh, ft=self.problem.ft, points=self.points)
        reward = 2.56 - (lifts + drags)
        done = (self.current_step == self.T)
        #print("lift is", lifts)
        #print("drag is", drags)
        #print("reward the action: (%f, %f) is %f" % (action[0], action[1], reward))
        info = {"drags": drags,
                "lifts": lifts}
        return obs, reward, done, info

    def reset(self, *, seed: [int] = None,
        return_info: bool = False,
        options: [dict] = None,):
        self.episode += 1
        self.current_step = 0
        self.problem = self.build_problem(self.init_jet)
        self.init_drags = self.problem.drags
        obs = self.get_obs(action=[0, 0])
        info = {}
        return obs

    def render(self, mode="human"):
        pass

    def close(self):
        pass
