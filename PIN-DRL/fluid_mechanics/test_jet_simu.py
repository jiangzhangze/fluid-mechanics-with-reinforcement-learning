# from jet_simulation import simulation
from simulation_visual import simulation
from area import *
from math import sin, cos, pi
import pandas as pd

r = 0.05
x1 = r * cos(pi / 2) + 0.3
y1 = r * sin(pi / 2) + 0.2
x2 = r * cos(3 / 2 * pi) + 0.3
y2 = r * sin(3 * pi / 2) + 0.2
jet_positions = [(x1, y1, 0), (x2, y2, 0)]
jet_coordinates = positions(jet_positions)
jet = {
    "jet_x": -0.7907469,
    "jet_y": 0.554599,
    "position": jet_coordinates
}
points = probes(x_min=0.4, y_min=0.05, length=0.1, num=10)
jet_x = jet.get("jet_x")
jet_y = jet.get("jet_y")
simu = simulation(c_x=0.3, c_y=0.2, o_x=0.3, o_y=0.2, r=0.05, r2=0.02)
mesh, ft = simu.generate_mesh()
t_u, t_p, p_diff, C_D, C_L, num_velocity_dofs, num_pressure_dofs = simu.compute(mesh=mesh, ft=ft, points=points,
                                                                                jet=jet, save='True')

dataframe = pd.DataFrame({'t_u': t_u, 't_p': t_p, 'p_diff': p_diff, 'C_D': C_D, 'C_L': C_L, 'num_velocity_dofs': num_velocity_dofs, 'num_pressure_dofs':num_pressure_dofs})
dataframe.to_csv(f"visual_{jet_x}_{jet_y}.csv")
