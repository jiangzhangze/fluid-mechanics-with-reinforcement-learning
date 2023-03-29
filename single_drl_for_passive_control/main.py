from fluid_mechanics.simulation import simulation
import numpy as np
x = np.linspace(0.4, 2, 100)
y = np.linspace(0 + 0.001, 0.41 - 0.001, 100)
points = np.zeros((3, 100))
points[0] = x
points[1] = y

simulation = simulation(c_x=0.2 , c_y=0.2, r=0.05, o_x=0.3, o_y=0.2, r2=0, obstacle_type='cylinder', save='true',
                            visualize='false')

