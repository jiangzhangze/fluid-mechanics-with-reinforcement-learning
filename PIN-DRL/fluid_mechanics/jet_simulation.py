import gmsh
import numpy as np
import sys
from mpi4py import MPI

from petsc4py import PETSc
from math import *
from dolfinx import geometry
from dolfinx.fem import (Constant, Function, FunctionSpace,
                         assemble_scalar, dirichletbc, form, locate_dofs_topological, set_bc)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, create_matrix, set_bc)
from dolfinx.geometry import BoundingBoxTree, compute_collisions, compute_colliding_cells
from dolfinx.io import (XDMFFile, distribute_entity_data, gmshio)
from utils.encoder import normalization
from ufl import (FacetNormal, FiniteElement, Identity, Measure, TestFunction, TrialFunction, VectorElement,
                 as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym)


class simulation:
    def __init__(self, c_x, c_y, o_x, o_y, r, r2,
                 obstacle_type='cylinder',
                 save_mesh='false',
                 save='false',
                 visualize='false'):

        self.L = 2.2
        self.H = 0.41
        self.c_x = c_x
        self.c_y = c_y
        self.r = r
        self.gdim = 2
        self.model_rank = 0
        self.mesh_comm = MPI.COMM_WORLD
        self.save = save
        self.o_x = o_x
        self.o_y = o_y
        self.r2 = r2
        self.obstacle_type = obstacle_type
        self.save_mesh = save_mesh
        self.visualize = visualize
        self.t = 0
        self.T = 7
        self.dt = 1 / 200
        self.num_steps = 100
        self.mesh, self.ft = self.generate_mesh()
        v_cg2 = VectorElement("CG", self.mesh.ufl_cell(), 2)
        s_cg1 = FiniteElement("CG", self.mesh.ufl_cell(), 1)
        self.V = FunctionSpace(self.mesh, v_cg2)
        self.Q = FunctionSpace(self.mesh, s_cg1)
        self.fdim = self.mesh.topology.dim - 1
        self.jet_x = 0
        self.jet_y = 0

    def generate_mesh(self):
        # L*H:computational domain's size
        # (c_x,c_y):the coordinates of big cylinder
        # r:radius of the big cylinder
        # r2:radius of the small cylinder
        gmsh.initialize()
        gmsh.clear()
        gmsh.option.setNumber('General.Verbosity', 1)
        #gmsh.
        rectangle = gmsh.model.occ.addRectangle(0, 0, 0, self.L, self.H, tag=1)
        obstacle1 = gmsh.model.occ.addDisk(self.c_x, self.c_y, 0, self.r, self.r)
        pre_fluid = gmsh.model.occ.cut([(self.gdim, rectangle)], [(self.gdim, obstacle1)])
        if self.r2 == 0:
            fluid = pre_fluid
        else:
            if self.obstacle_type == 'cylinder':
                obstacle2 = gmsh.model.occ.addDisk(self.o_x, self.o_y, 0, self.r2, self.r2)
            elif self.obstacle_type == 'rectangular':
                obstacle2 = gmsh.model.occ.addRectangle(self.o_x, self.o_y, 0, self.r2, self.r2)
            fluid = gmsh.model.occ.cut([(self.gdim, rectangle)], [(self.gdim, obstacle2)])

        gmsh.model.occ.synchronize()
        fluid_marker = 1

        volumes = gmsh.model.getEntities(dim=self.gdim)
        assert (len(volumes) == 1)
        gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
        gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")

        self.inlet_marker, self.outlet_marker, self.wall_marker, self.obstacle_marker = 2, 3, 4, 5
        inflow, outflow, walls, obstacle = [], [], [], []

        boundaries = gmsh.model.getBoundary(volumes, oriented=False)
        for boundary in boundaries:
            center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
            if np.allclose(center_of_mass, [0, self.H / 2, 0]):
                inflow.append(boundary[1])
            elif np.allclose(center_of_mass, [self.L, self.H / 2, 0]):
                outflow.append(boundary[1])
            elif np.allclose(center_of_mass, [self.L / 2, self.H, 0]) or np.allclose(center_of_mass, [self.L / 2, 0, 0]):
                walls.append(boundary[1])
            else:
                obstacle.append(boundary[1])
        gmsh.model.addPhysicalGroup(1, walls, self.wall_marker)
        gmsh.model.setPhysicalName(1, self.wall_marker, "Walls")
        gmsh.model.addPhysicalGroup(1, inflow, self.inlet_marker)
        gmsh.model.setPhysicalName(1, self.inlet_marker, "Inlet")
        gmsh.model.addPhysicalGroup(1, outflow, self.outlet_marker)
        gmsh.model.setPhysicalName(1, self.outlet_marker, "Outlet")
        gmsh.model.addPhysicalGroup(1, obstacle, self.obstacle_marker)
        gmsh.model.setPhysicalName(1, self.obstacle_marker, "Obstacle")

        res_min = self.r / 3
        distance_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", obstacle)
        threshold_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.25 * self.H)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", self.r)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * self.H)
        min_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
        gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        gmsh.model.mesh.generate(self.gdim)
        gmsh.model.mesh.setOrder(2)
        gmsh.model.mesh.optimize("Netgen")
        self.mesh, _, ft = gmshio.model_to_mesh(gmsh.model, self.mesh_comm, self.model_rank, gdim=self.gdim)
        ft.name = "facet markers"

        if self.visualize == 'true':
            if '-nopopup' not in sys.argv:
                gmsh.fltk.run()
        if self.save_mesh == 'true':
            gmsh.write("cylinders.msh")

        return self.mesh, ft

    def update_jet_bc(self, jet=None):

        if jet == None:
            jet = {
                "jet_x": self.jet_x,
                "jet_y": self.jet_y,
                "position": jet_coordinates
            }
        jet_points = []
        bb_tree = geometry.BoundingBoxTree(self.mesh, self.mesh.topology.dim)
        jet_position = jet.get("position")
        jet_x = jet.get("jet_x")
        jet_y = jet.get("jet_y")
        jet_cell_candidates = geometry.compute_collisions(bb_tree, jet_position.T)
        jet_colliding_cells = geometry.compute_colliding_cells(self.mesh, jet_cell_candidates, jet_position.T)
        self.jet_cells = []
        for i, jet_point in enumerate(jet_position.T):
            if len(jet_colliding_cells.links(i)) > 0:
                jet_points.append(jet_point)
                self.jet_cells.append(jet_colliding_cells.links(i)[0])
        class JetVelocity():
            def __init__(self, t, jetx, jety):
                self.t_jet = t
                self.jetx = jetx
                self.jety = jety

            def __call__(self, x):
                values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
                values[1] = self.jety
                index = np.where(x[1, :] < 0.2)
                values[1, index] = -self.jety
                values[0] = self.jetx
                return values

        self.jet_x = self.jet_x + 0.1*(jet_x-self.jet_x)
        self.jet_y = self.jet_y + 0.1*(jet_y-self.jet_y)
        jet_velocity = JetVelocity(self.t, self.jet_x, self.jet_y)
        u_jet = Function(self.V)
        u_jet.interpolate(jet_velocity)
        self.bcu_jets = dirichletbc(u_jet, locate_dofs_topological(self.V, self.fdim + 1, self.jet_cells))

    def compute(self, mesh,  ft, points):
        global u_probes_t, p_probes_t
        self.mesh = mesh
        bb_tree = geometry.BoundingBoxTree(self.mesh, self.mesh.topology.dim)
        self.cells = []
        self.points_on_pro = []
        cell_candidates = geometry.compute_collisions(bb_tree, points.T)
        colliding_cells = geometry.compute_colliding_cells(self.mesh, cell_candidates, points.T)
        for i, point in enumerate(points.T):
            if len(colliding_cells.links(i)) > 0:
                self.points_on_pro.append(point)
                self.cells.append(colliding_cells.links(i)[0])

        # Physical and discretization parameters
        # Following the DGF-2 benchmark, we define our problem specific parameters
         # int(T/dt)
        self.k = Constant(self.mesh, PETSc.ScalarType(self.dt))
        self.mu = Constant(self.mesh, PETSc.ScalarType(0.001))  # Dynamic viscosity
        self.rho = Constant(self.mesh, PETSc.ScalarType(1))  # Density

        # ```{admonition} Reduced end-time of problem
        # In the current demo, we have reduced the run time to one second to make it easier to illustrate the concepts of the benchmark. By increasing the end-time `T` to 8, the runtime in a notebook is approximately 25 minutes. If you convert the notebook to a python file and use `mpirun`, you can reduce the runtime of the problem.
        # ```
        #
        # ## Boundary conditions
        # As we have created the mesh and relevant mesh tags, we can now specify the function spaces `V` and `Q` along with the boundary conditions. As the `ft` contains markers for facets, we use this class to find the facets for the inlet and walls.

        # +

        # Define boundary conditions
        class InletVelocity():
            def __init__(self, t):
                self.t = t

            def __call__(self, x):
                values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
                values[0] = 4 * 1.5 * x[1] * (0.41 - x[1]) / (0.41 ** 2)
                return values

        # Inlet

        self.inlet_velocity = InletVelocity(self.t)
        self.u_inlet = Function(self.V)
        self.u_inlet.interpolate(self.inlet_velocity)

        # Walls
        u_nonslip = np.array((0,) * self.mesh.geometry.dim, dtype=PETSc.ScalarType)
        # Obstacle
        self.bcu_inflow = dirichletbc(self.u_inlet,
                                       locate_dofs_topological(self.V, self.fdim, ft.find(self.inlet_marker)))
        self.bcu_walls = dirichletbc(u_nonslip, locate_dofs_topological(self.V, self.fdim, ft.find(self.wall_marker)),
                                       self.V)
        self.bcu_obstacle = dirichletbc(u_nonslip,
                                       locate_dofs_topological(self.V, self.fdim, ft.find(self.obstacle_marker)),
                                       self.V)
        self.bcu = [self.bcu_inflow, self.bcu_obstacle, self.bcu_walls, self.bcu_jets]
        # Outlet
        bcp_outlet = dirichletbc(PETSc.ScalarType(0), locate_dofs_topological(self.Q, self.fdim, ft.find(self.outlet_marker)),
                                 self.Q)
        self.bcp = [bcp_outlet]
        # -
        #
        # We start by defining all the variables used in the variational formulations.

        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)
        self.u_ = Function(self.V)
        self.u_.name = "u"
        self.u_s = Function(self.V)
        self.u_n = Function(self.V)
        self.u_n1 = Function(self.V)
        self.p = TrialFunction(self.Q)
        self.q = TestFunction(self.Q)
        self.p_ = Function(self.Q)
        self.p_.name = "p"
        self.phi = Function(self.Q)

        # Next, we define the variational formulation for the first step, where we have integrated the diffusion term, as well as the pressure term by parts.

        self.f = Constant(self.mesh, PETSc.ScalarType((0, 0)))
        self.F1 = self.rho / self.k * dot(self.u - self.u_n, self.v) * dx
        self.F1 += inner(dot(1.5 * self.u_n - 0.5 * self.u_n1, 0.5 * nabla_grad(self.u + self.u_n)), self.v) * dx
        self.F1 += 0.5 * self.mu * inner(grad(self.u + self.u_n), grad(self.v)) * dx - dot(self.p_, div(self.v)) * dx
        self.F1 += dot(self.f, self.v) * dx
        self.a1 = form(lhs(self.F1))
        self.L1 = form(rhs(self.F1))
        self.A1 = create_matrix(self.a1)
        self.b1 = create_vector(self.L1)

        # Next we define the second step

        self.a2 = form(dot(grad(self.p), grad(self.q)) * dx)
        self.L2 = form(-self.rho / self.k * dot(div(self.u_s), self.q) * dx)
        self.A2 = assemble_matrix(self.a2, bcs=self.bcp)
        self.A2.assemble()
        self.b2 = create_vector(self.L2)

        # We finally create the last step

        self.a3 = form(self.rho * dot(self.u, self.v) * dx)
        self.L3 = form(self.rho * dot(self.u_s, self.v) * dx - self.k * dot(nabla_grad(self.phi), self.v) * dx)
        self.A3 = assemble_matrix(self.a3)
        self.A3.assemble()
        self.b3 = create_vector(self.L3)

        # As in the previous tutorials, we use PETSc as a linear algebra backend.

        # +
        # Solver for step 1
        self.solver1 = PETSc.KSP().create(self.mesh.comm)
        self.solver1.setOperators(self.A1)
        self.solver1.setType(PETSc.KSP.Type.BCGS)
        pc1 = self.solver1.getPC()
        pc1.setType(PETSc.PC.Type.JACOBI)

        # Solver for step 2
        self.solver2 = PETSc.KSP().create(self.mesh.comm)
        self.solver2.setOperators(self.A2)
        self.solver2.setType(PETSc.KSP.Type.MINRES)
        pc2 = self.solver2.getPC()
        pc2.setType(PETSc.PC.Type.HYPRE)
        pc2.setHYPREType("boomeramg")

        # Solver for step 3
        self.solver3 = PETSc.KSP().create(self.mesh.comm)
        self.solver3.setOperators(self.A3)
        self.solver3.setType(PETSc.KSP.Type.CG)
        pc3 = self.solver3.getPC()
        pc3.setType(PETSc.PC.Type.SOR)
        # -

        # ## Verification of the implementation compute known physical quantities
        # As a further verification of our implementation, we compute the drag and lift coefficients over the obstacle, defined as
        #
        # $$
        #     C_{\text{D}}(u,p,t,\partial\Omega_S) = \frac{2}{\rho L U_{mean}^2}\int_{\partial\Omega_S}\rho \nu n \cdot \nabla u_{t_S}(t)n_y -p(t)n_x~\mathrm{d} s,
        # $$
        # $$
        #     C_{\text{L}}(u,p,t,\partial\Omega_S) = -\frac{2}{\rho L U_{mean}^2}\int_{\partial\Omega_S}\rho \nu n \cdot \nabla u_{t_S}(t)n_x + p(t)n_y~\mathrm{d} s,
        # $$
        #
        # where $u_{t_S}$ is the tangential velocity component at the interface of the obstacle $\partial\Omega_S$, defined as $u_{t_S}=u\cdot (n_y,-n_x)$, $U_{mean}=1$ the average inflow velocity, and $L$ the length of the channel. We use `UFL` to create the relevant integrals, and assemble them at each time step.

        n = -FacetNormal(self.mesh)  # Normal pointing out of obstacle
        dObs = Measure("ds", domain=self.mesh, subdomain_data=ft, subdomain_id=self.obstacle_marker)
        u_t = inner(as_vector((n[1], -n[0])), self.u_)
        self.drag = form(2 / 0.1 * (self.mu / self.rho * inner(grad(u_t), n) * n[1] - self.p_ * n[0]) * dObs)
        self.lift = form(-2 / 0.1 * (self.mu / self.rho * inner(grad(u_t), n) * n[0] + self.p_ * n[1]) * dObs)
        if self.mesh.comm.rank == 0:
            self.C_D = np.zeros(self.num_steps, dtype=PETSc.ScalarType)
            self.C_L = np.zeros(self.num_steps, dtype=PETSc.ScalarType)
            self.t_u = np.zeros(self.num_steps, dtype=np.float64)
            self.t_p = np.zeros(self.num_steps, dtype=np.float64)

        # We will also evaluate the pressure at two points, on in front of the obstacle, $(0.15, 0.2)$, and one behind the obstacle, $(0.25, 0.2)$. To do this, we have to find which cell is containing each of the points, so that we can create a linear combination of the local basis functions and coefficients.

        tree = BoundingBoxTree(self.mesh, self.mesh.geometry.dim)

        # ## Solving the time-dependent problem
        # ```{admonition} Stability of the Navier-Stokes equation
        # Note that the current splitting scheme has to fullfil the a [Courant–Friedrichs–Lewy condition](https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition). This limits the spatial discretization with respect to the inlet velocity and temporal discretization.
        # Other temporal discretization schemes such as the second order backward difference discretization or Crank-Nicholson discretization with Adams-Bashforth linearization are better behaved than our simple backward difference scheme.
        # ```
        #
        # As in the previous example, we create output files for the velocity and pressure and solve the time-dependent problem. As we are solving a time dependent problem with many time steps, we use the `tqdm`-package to visualize the progress. This package can be install with `pip3`.

        # + tags=[]
    def excute(self, jet):
        u_probes = []
        p_probes = []
        drag_coeffs = []
        lift_coeffs = []
        for i in range(self.num_steps):
            # progress.update(1)
            # Update current time step
            self.t += self.dt
            # Update inlet velocity
            self.inlet_velocity.t = self.t
            self.update_jet_bc(jet=jet)
            self.bcu = [self.bcu_inflow, self.bcu_obstacle, self.bcu_walls, self.bcu_jets]
            self.u_inlet.interpolate(self.inlet_velocity)
            # print(self.t)
            # print((self.jet_x, self.jet_x))
            # Step 1: Tentative velocity step
            self.A1.zeroEntries()
            assemble_matrix(self.A1, self.a1, bcs=self.bcu)
            self.A1.assemble()
            with self.b1.localForm() as loc:
                loc.set(0)
            assemble_vector(self.b1, self.L1)
            apply_lifting(self.b1, [self.a1], [self.bcu])
            self.b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(self.b1, self.bcu)
            self.solver1.solve(self.b1, self.u_s.vector)
            self.u_s.x.scatter_forward()

            # Step 2: Pressure corrrection step
            with self.b2.localForm() as loc:
                loc.set(0)
            assemble_vector(self.b2, self.L2)
            apply_lifting(self.b2, [self.a2], [self.bcp])
            self.b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(self.b2, self.bcp)
            self.solver2.solve(self.b2, self.phi.vector)
            self.phi.x.scatter_forward()

            self.p_.vector.axpy(1, self.phi.vector)
            self.p_.x.scatter_forward()

            # Step 3: Velocity correction step
            with self.b3.localForm() as loc:
                loc.set(0)
            assemble_vector(self.b3, self.L3)
            self.b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            self.solver3.solve(self.b3, self.u_.vector)
            self.u_.x.scatter_forward()

            # write the data of probes
            if i % 25 == 0:
                u_values = self.u_.eval(self.points_on_pro, self.cells)
                p_values = self.p_.eval(self.points_on_pro, self.cells)
                u_probes.append(u_values)
                p_probes.append(p_values)
                with self.u_.vector.localForm() as loc_, self.u_n.vector.localForm() as loc_n, self.u_n1.vector.localForm() as loc_n1:
                    loc_n.copy(loc_n1)
                    loc_.copy(loc_n)
            drag_coeff = self.mesh.comm.gather(assemble_scalar(self.drag), root=0)
            lift_coeff = self.mesh.comm.gather(assemble_scalar(self.lift), root=0)
            drag_coeffs.append(drag_coeff)
            lift_coeffs.append(lift_coeff)
            if self.mesh.comm.rank == 0:
                self.t_u[i] = self.t
                self.t_p[i] = self.t - self.dt / 2
                self.C_D[i] = sum(drag_coeff)
                self.C_L[i] = sum(lift_coeff)
        u_probes_t = np.array(u_probes).reshape(-1)
        p_probes_t = np.array(p_probes).reshape(
            -1
        )
        self.u_field_probes = np.array(u_probes_t)
        self.u_field_probes = np.nan_to_num(self.u_field_probes, nan=0)
        self.p_field_probes = np.array(p_probes_t)
        u_probes_t = np.array(u_probes).reshape(2, -1)
        self.u_probes_t = np.array(normalization(u_probes_t)).reshape(1, -1)
        p_probes_t = np.array(p_probes).reshape(-1)
        C_D = self.C_D[-20:]
        C_L = self.C_L[-20:]
        # print(C_D)
        self.drags = sum(C_D) / len(C_D)
        self.lifts = sum(map(abs, C_L)) / len(C_L)
        gmsh.clear()
        return u_probes_t, p_probes_t, self.drags, self.lifts




if __name__ == '__main__':
    from area import *
    from math import sin, cos, pi

    r = 0.05
    x1 = r * cos(pi / 2) + 0.3
    y1 = r * sin(pi / 2) + 0.2
    x2 = r * cos(3 / 2 * pi) + 0.3
    y2 = r * sin(3 * pi / 2) + 0.2
    jet_positions = [(x1, y1, 0), (x2, y2, 0)]
    jet_coordinates = positions(jet_positions)
    jet = {
        "jet_x": -1,
        "jet_y": -1,
        "position": jet_coordinates
    }
    points = rectangular(low=[0.2, 0.05], high=[0.4, 0.15], num=400)
    jet_x = jet.get("jet_x")
    jet_y = jet.get("jet_y")
    simu = simulation(c_x=0.3, c_y=0.2, o_x=0.3, o_y=0.2, r=0.05, r2=0)
    for i in range(128):
        if simu.t < 0.05:
            simu.update_jet_bc()
            u, p, drags, lifts = simu.compute(mesh=simu.mesh, ft=simu.ft, points=points)
        else:
            simu.update_jet_bc(jet=jet)
            u, p, drags, lifts = simu.compute(mesh=simu.mesh, ft=simu.ft, points=points)
