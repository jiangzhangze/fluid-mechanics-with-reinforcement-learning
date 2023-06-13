import gmsh
import numpy as np
import sys
from mpi4py import MPI
from pathlib import Path
from petsc4py import PETSc
from utils.encoder import normalization
from tqdm import tqdm, autonotebook
from dolfinx import geometry
from dolfinx.fem import (Constant, Function, FunctionSpace,
                         assemble_scalar, dirichletbc, form, locate_dofs_topological, set_bc)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, create_matrix, set_bc)
from dolfinx.io import (XDMFFile, distribute_entity_data, gmshio)

from ufl import (FacetNormal, FiniteElement, Identity, Measure, TestFunction, TrialFunction, VectorElement,
                 as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym)

class simulation:
    def __init__(self, c_x, c_y, o_x, o_y, r, r2, obstacle_type='cylinder', save_mesh='false', save='false', visualize='false'):

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

    def generate_mesh(self):
        # L*H:computational domain's size
        # (c_x,c_y):the coordinates of big cylinder
        # r:radius of the big cylinder
        # r2:radius of the small cylinder
        gmsh.initialize()

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
        mesh, _, ft = gmshio.model_to_mesh(gmsh.model, self.mesh_comm, self.model_rank, gdim=self.gdim)
        ft.name = "facet markers"

        if self.visualize == 'true':
            if '-nopopup' not in sys.argv:
                gmsh.fltk.run()
        if self.save_mesh == 'true':
            gmsh.write("cylinders.msh")

        return mesh, ft

    def compute(self, mesh,  ft, points, save='False'):
        global u_probes_t, p_probes_t
        bb_tree = geometry.BoundingBoxTree(mesh, mesh.topology.dim)
        cells = []
        points_on_pro = []
        cell_candidates = geometry.compute_collisions(bb_tree, points.T)
        colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, points.T)
        for i, point in enumerate(points.T):
            if len(colliding_cells.links(i)) > 0:
                points_on_pro.append(point)
                cells.append(colliding_cells.links(i)[0])

        # Physical and discretization parameters
        # Following the DGF-2 benchmark, we define our problem specific parameters

        t = 0
        T = 10  # Final time
        dt = 1 / 1600  # Time step size
        #num_steps = int(T / dt)
        self.num_steps = int(T / dt)
        k = Constant(mesh, PETSc.ScalarType(dt))
        mu = Constant(mesh, PETSc.ScalarType(0.001))  # Dynamic viscosity
        rho = Constant(mesh, PETSc.ScalarType(1))  # Density
        cache_dir = f"{str(Path.cwd())}/.cache"
        jit_options = {"cffi_extra_compile_args": ["-O3", "-march=native"],
                       "cache_dir": cache_dir,
                       "cffi_libraries": ["m"]}

        # ```{admonition} Reduced end-time of problem
        # In the current demo, we have reduced the run time to one second to make it easier to illustrate the concepts of the benchmark. By increasing the end-time `T` to 8, the runtime in a notebook is approximately 25 minutes. If you convert the notebook to a python file and use `mpirun`, you can reduce the runtime of the problem.
        # ```
        #
        # ## Boundary conditions
        # As we have created the mesh and relevant mesh tags, we can now specify the function spaces `V` and `Q` along with the boundary conditions. As the `ft` contains markers for facets, we use this class to find the facets for the inlet and walls.

        # +
        v_cg2 = VectorElement("CG", mesh.ufl_cell(), 2)
        s_cg1 = FiniteElement("CG", mesh.ufl_cell(), 1)
        V = FunctionSpace(mesh, v_cg2)
        Q = FunctionSpace(mesh, s_cg1)

        fdim = mesh.topology.dim - 1

        # Define boundary conditions
        class InletVelocity():
            def __init__(self, t):
                self.t = t

            def __call__(self, x):
                values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
                values[0] = 4 * 1.5 * np.sin(self.t * np.pi / 8) * x[1] * (0.41 - x[1]) / (0.41 ** 2)
                return values

        # Inlet
        u_inlet = Function(V)
        inlet_velocity = InletVelocity(t)
        u_inlet.interpolate(inlet_velocity)
        bcu_inflow = dirichletbc(u_inlet, locate_dofs_topological(V, fdim, ft.find(self.inlet_marker)))
        # Walls
        u_nonslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
        bcu_walls = dirichletbc(u_nonslip, locate_dofs_topological(V, fdim, ft.find(self.wall_marker)), V)
        # Obstacle
        bcu_obstacle = dirichletbc(u_nonslip, locate_dofs_topological(V, fdim, ft.find(self.obstacle_marker)), V)
        bcu = [bcu_inflow, bcu_obstacle, bcu_walls]
        # Outlet
        bcp_outlet = dirichletbc(PETSc.ScalarType(0), locate_dofs_topological(Q, fdim, ft.find(self.outlet_marker)), Q)
        bcp = [bcp_outlet]
        # -
        #
        # We start by defining all the variables used in the variational formulations.

        u = TrialFunction(V)
        v = TestFunction(V)
        u_ = Function(V)
        u_.name = "u"
        u_s = Function(V)
        u_n = Function(V)
        u_n1 = Function(V)
        p = TrialFunction(Q)
        q = TestFunction(Q)
        p_ = Function(Q)
        p_.name = "p"
        phi = Function(Q)

        # Next, we define the variational formulation for the first step, where we have integrated the diffusion term, as well as the pressure term by parts.

        f = Constant(mesh, PETSc.ScalarType((0, 0)))
        F1 = rho / k * dot(u - u_n, v) * dx
        F1 += inner(dot(1.5 * u_n - 0.5 * u_n1, 0.5 * nabla_grad(u + u_n)), v) * dx
        F1 += 0.5 * mu * inner(grad(u + u_n), grad(v)) * dx - dot(p_, div(v)) * dx
        F1 += dot(f, v) * dx
        a1 = form(lhs(F1), jit_options=jit_options)
        L1 = form(rhs(F1), jit_options=jit_options)
        A1 = create_matrix(a1)
        b1 = create_vector(L1)

        # Next we define the second step

        a2 = form(dot(grad(p), grad(q)) * dx, jit_options=jit_options)
        L2 = form(-rho / k * dot(div(u_s), q) * dx, jit_options=jit_options)
        A2 = assemble_matrix(a2, bcs=bcp)
        A2.assemble()
        b2 = create_vector(L2)

        # We finally create the last step

        a3 = form(rho * dot(u, v) * dx, jit_options=jit_options)
        L3 = form(rho * dot(u_s, v) * dx - k * dot(nabla_grad(phi), v) * dx, jit_options=jit_options)
        A3 = assemble_matrix(a3)
        A3.assemble()
        b3 = create_vector(L3)

        # As in the previous tutorials, we use PETSc as a linear algebra backend.

        # +
        # Solver for step 1
        solver1 = PETSc.KSP().create(mesh.comm)
        solver1.setOperators(A1)
        solver1.setType(PETSc.KSP.Type.BCGS)
        pc1 = solver1.getPC()
        pc1.setType(PETSc.PC.Type.JACOBI)

        # Solver for step 2
        solver2 = PETSc.KSP().create(mesh.comm)
        solver2.setOperators(A2)
        solver2.setType(PETSc.KSP.Type.MINRES)
        pc2 = solver2.getPC()
        pc2.setType(PETSc.PC.Type.HYPRE)
        pc2.setHYPREType("boomeramg")

        # Solver for step 3
        solver3 = PETSc.KSP().create(mesh.comm)
        solver3.setOperators(A3)
        solver3.setType(PETSc.KSP.Type.CG)
        pc3 = solver3.getPC()
        pc3.setType(PETSc.PC.Type.SOR)

        n = -FacetNormal(mesh)  # Normal pointing out of obstacle
        dObs = Measure("ds", domain=mesh, subdomain_data=ft, subdomain_id=self.obstacle_marker)
        u_t = inner(as_vector((n[1], -n[0])), u_)
        drag = form(2 / 0.1 * (mu / rho * inner(grad(u_t), n) * n[1] - p_ * n[0]) * dObs)
        lift = form(-2 / 0.1 * (mu / rho * inner(grad(u_t), n) * n[0] + p_ * n[1]) * dObs)
        C_D = np.zeros(self.num_steps, dtype=PETSc.ScalarType)
        C_L = np.zeros(self.num_steps, dtype=PETSc.ScalarType)
        t_u = np.zeros(self.num_steps, dtype=np.float64)
        t_p = np.zeros(self.num_steps, dtype=np.float64)

        # + tags=[]
        if save =='True':
            xdmf = XDMFFile(mesh.comm, "flow_around_cylinder.xdmf", "w")
            xdmf.write_mesh(mesh)

        u_probes = []
        p_probes = []
        #progress = autonotebook.tqdm(desc="Solving PDE", total=self.num_steps)
        drag_coeffs = []
        lift_coeffs = []
        for i in range(self.num_steps):
            #progress.update(1)
            # Update current time step
            t += dt
            # Update inlet velocity
            inlet_velocity.t = t
            u_inlet.interpolate(inlet_velocity)

            # Step 1: Tentative velocity step
            A1.zeroEntries()
            assemble_matrix(A1, a1, bcs=bcu)
            A1.assemble()
            with b1.localForm() as loc:
                loc.set(0)
            assemble_vector(b1, L1)
            apply_lifting(b1, [a1], [bcu])
            b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b1, bcu)
            solver1.solve(b1, u_s.vector)
            u_s.x.scatter_forward()

            # Step 2: Pressure corrrection step
            with b2.localForm() as loc:
                loc.set(0)
            assemble_vector(b2, L2)
            apply_lifting(b2, [a2], [bcp])
            b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b2, bcp)
            solver2.solve(b2, phi.vector)
            phi.x.scatter_forward()

            p_.vector.axpy(1, phi.vector)
            p_.x.scatter_forward()

            # Step 3: Velocity correction step
            with b3.localForm() as loc:
                loc.set(0)
            assemble_vector(b3, L3)
            b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            solver3.solve(b3, u_.vector)
            u_.x.scatter_forward()

            # write the data of probes
            u_values = u_.eval(points_on_pro, cells)
            p_values = p_.eval(points_on_pro, cells)
            u_probes.append(u_values)
            p_probes.append(p_values)

            # u_probes_t = np.array(u_probes).transpose([0, 2, 1])

            # p_probes_t = np.array(p_probes).transpose([0, 2, 1])
            # Write solutions to file
            if save == 'True' and (i % 50 == 0 or i == self.num_steps-1):
                xdmf.write_function(u_, t)
                xdmf.write_function(p_, t)

                # Update variable with solution form this time step
            with u_.vector.localForm() as loc_, u_n.vector.localForm() as loc_n, u_n1.vector.localForm() as loc_n1:
                loc_n.copy(loc_n1)
                loc_.copy(loc_n)

                # Compute physical quantities
                # For this to work in paralell, we gather contributions from all processors
                # to processor zero and sum the contributions.
            drag_coeff = assemble_scalar(drag)
            lift_coeff = assemble_scalar(lift)
            drag_coeffs.append(drag_coeff)
            lift_coeffs.append(lift_coeff)
            '''if mesh.comm.rank == 0:
                t_u[i] = t
                t_p[i] = t - dt / 2
                C_D[i] = sum(drag_coeff)
                C_L[i] = sum(lift_coeff)'''
        u_probes_t = np.array(u_probes).reshape(2, -1)
        self.u_probes_t = np.array(normalization(u_probes_t)).reshape(1, -1)
        p_probes_t = np.array(p_probes).reshape(-1)
        self.drags = sum(drag_coeffs)/len(drag_coeffs)
        lift_coeffs = map(abs, lift_coeffs)
        lifts = sum(lift_coeffs)/len([lift_coeffs])
        if save == 'True':
            xdmf.close()
        gmsh.clear()
        return self.u_probes_t, p_probes_t, self.drags, lifts


