from simulation import simulation
from area import *

points = probes(x_min=0.4, y_min=0.05, length=0.1, num=10)
simu = simulation(c_x=0.2, c_y=0.2, o_x=0.387, o_y=0.216, r=0.05, r2=0.02, visualize='true')
mesh, ft = simu.generate_mesh()
#u, p, drag, lift = simu.compute(mesh=mesh, ft=ft, points=points, save='True')

#print(simu.u_probes_t.shape)