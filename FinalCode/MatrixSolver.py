from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from xii import *
from block import *
from scipy.sparse.linalg import *
import vtk 
from block.iterative import LGMRES
import pyvista as pv
import copy
import os
import VerticalAverage as va 
from abc import ABC, abstractmethod

class MatrixSolver(ABC):

    def __init__(self, mesh, V, n0, c_0, dt, T, save_interval, times, doses, path_sol, f,dz,bc_type):
        self.mesh = mesh
        self.V = V
        self.n0 = n0
        self.c_k = c_0
        self.dt = dt
        self.T = T
        self.save_interval = save_interval
        self.times = times
        self.doses = doses
        self.path_sol = path_sol
        self.Dxc = 4.32*1e3
        self.gamma = 4*self.Dxc
        self.K_m = 0.005
        self.Dxn = 2.4*1e-3
        self.Dsn = 1.2*1e-4
        self.p_csc = 0.12
        self.p_dc = 0.48
        self.K_csc = 0.05
        self.K_dc = 0.3 
        self.g_csc = 0.1
        self.g_dc = 0.2
        self.g_tdc = 0.1
        self.s_csc = 0
        self.s_dc = 0.55
        self.epsilon = 0.05
        self.epsilon_k = 0.01
        self.V_plus = 1.92*1e-2
        self.V_minus = 0.48*1e-2
        self.csi_plus = 0.1
        self.csi_minus = 0.1
        self.c_H = 0.3
        self.c_N = 0.0125
        self.c_R = 0.1
        self.d_tdc = 0.024
        self.d_n = 2.4
        self.OER = 3
        self.alpha_min = 0.007
        self.delta_alpha = 0.143
        self.beta_min = 0.002
        self.delta_beta = 0.018
        self.k=5
        self.dz = dz
        self.f = f
        self.bc_type = bc_type

    def set_parameters(self, dict_params):
        for key, value in dict_params.items():
            self.key = value

    def solve(self):

        self.boundaries()
        n_steps = int(self.T/self.dt)
    
        N = Function(self.V)
        C = Function(self.V)
        n = TrialFunction(self.V)
        v = TestFunction(self.V)
        
        c = TrialFunction(self.V)
        w = TestFunction(self.V)
        phi = Function(self.V)
        nh0 = Function(self.V)
        nh1 = Function(self.V)
        
        bcs_c = self.oxygen_bc(self.bc_type)

        mass = []
        csc_mass = []
        dc_mass = []
        tdc_mass = []

        t=0
        d=0
        tol = 1.0E-3    
        maxiter = 30  
        Ns = int(1/self.dz)

        cfile = XDMFFile(f"{self.path_sol}/c.xdmf")
        cfile.parameters['rewrite_function_mesh'] = False
        phi_file = XDMFFile(f"{self.path_sol}/phi.xdmf")
        phi_file.parameters['rewrite_function_mesh'] = False

        n0_array = [Function(self.V) for _ in range(Ns+1)]
        sum = np.array([0.0]*self.V.dim())
        for s in range(Ns+1):
            self.n0.s = s*self.dz
            temp = Function(self.V)
            temp.assign(self.n0)
            n0_array[s].vector()[:] = temp.vector()[:]

        phi = self.vertical_average(n0_array,Ns)
        
        while(t < n_steps):
            print('time=%g: ' %(t*self.dt))

            a_c = self.Dxc * inner(grad(c),grad(w))*dx + self.gamma/(self.c_k + self.K_m)*phi*c*w*dx
            L_c = self.f*w*dx
            
            iter = 0  
            eps= 1.0
            while eps > tol and iter < maxiter:
                iter += 1
                solve(a_c==L_c,C,bcs_c)
                difference = C.vector() - self.c_k.vector()
                eps = np.linalg.norm(difference) / np.linalg.norm(self.c_k.vector())
                print('iter=%d: norm=%g' % (iter, eps))
                self.c_k.assign(C) 
        
            matrix_list = []
            # first row 
            s = 0

            P = Expression('(p_csc*pow(c,4)/(pow(K_csc,4)+pow(c,4))*exp(-pow((s-s_csc)/g_csc,2)) \
                    + p_dc*pow(c,4)/(pow(K_dc,4)+pow(c,4))*exp(-pow((s-s_dc)/g_dc,2)))*(1-phi)',
                    p_csc=self.p_csc,p_dc=self.p_dc,K_csc=self.K_csc,K_dc=self.K_dc,g_csc=self.g_csc,g_dc=self.g_dc,
                    s_csc=self.s_csc,s_dc=self.s_dc,phi=phi,s=s*self.dz,c=C,degree=1)
            K = Expression('d_tdc * exp(-((1-s)/g_tdc)) + d_n * (0.5+0.5*tanh(pow(epsilon_k,-1)*(c_N-c)))',
                    d_tdc=self.d_tdc,d_n=self.d_n,g_tdc=self.g_tdc,epsilon_k=self.epsilon_k,c_N=self.c_N,c=C,s=s*self.dz,degree=1)
            F = Expression("P - K", degree=1, P=P, K=K)
            vs = Expression('V_plus*tanh(s/csi_plus)*tanh((1-s)/csi_plus)*(0.5+0.5*tanh((c-c_H)*pow(epsilon,-1))) \
                        - V_minus*tanh(s/csi_minus)*tanh(pow(1-s,2)/csi_minus)*(0.5+0.5*tanh((c_H-c)*pow(epsilon,-1)))',
                        V_plus=self.V_plus,V_minus=self.V_minus,csi_minus=self.csi_minus,csi_plus=self.csi_plus,c_H=self.c_H,
                        epsilon=self.epsilon,c=C,s=(s+0.5)*self.dz,degree=1)

            a00 = (1/self.dt)*n*v*dx + self.Dxn*inner(grad(n),grad(v))*dx + self.Dsn/(self.dz*self.dz)*n*v*dx - n*v*F*dx + 0.5/self.dz*vs*n*v*dx
            a01 = -self.Dsn/(self.dz*self.dz)*n*v*dx + 0.5/(self.dz)*vs*n*v*dx
            A0 = assemble(a00)
            A0r = assemble(a01)
            row1 = [0] * (Ns+1)
            row1[s] = A0
            row1[s+1] = A0r
            matrix_list.append(row1)

            # middle rows
            for s in range(1,Ns):
                P = Expression('(p_csc*pow(c,4)/(pow(K_csc,4)+pow(c,4))*exp(-pow((s-s_csc)/g_csc,2)) \
                    + p_dc*pow(c,4)/(pow(K_dc,4)+pow(c,4))*exp(-pow((s-s_dc)/g_dc,2)))*(1-phi)',
                    p_csc=self.p_csc,p_dc=self.p_dc,K_csc=self.K_csc,K_dc=self.K_dc,g_csc=self.g_csc,g_dc=self.g_dc,
                    s_csc=self.s_csc,s_dc=self.s_dc,phi=phi,s=s*self.dz,c=C,degree=1)
                K = Expression('d_tdc * exp(-((1-s)/g_tdc)) + d_n * (0.5+0.5*tanh(pow(epsilon_k,-1)*(c_N-c)))',
                    d_tdc=self.d_tdc,d_n=self.d_n,g_tdc=self.g_tdc,epsilon_k=self.epsilon_k,c_N=self.c_N,c=C,s=s*self.dz,degree=1)
                vs_plus = Expression('V_plus*tanh(s/csi_plus)*tanh((1-s)/csi_plus)*(0.5+0.5*tanh((c-c_H)*pow(epsilon,-1))) \
                        - V_minus*tanh(s/csi_minus)*tanh(pow(1-s,2)/csi_minus)*(0.5+0.5*tanh((c_H-c)*pow(epsilon,-1)))',
                        V_plus=self.V_plus,V_minus=self.V_minus,csi_minus=self.csi_minus,csi_plus=self.csi_plus,c_H=self.c_H,
                        epsilon=self.epsilon,c=C,s=(s+0.5)*self.dz,degree=1)
                vs_minus = Expression('V_plus*tanh(s/csi_plus)*tanh((1-s)/csi_plus)*(0.5+0.5*tanh((c-c_H)*pow(epsilon,-1))) \
                        - V_minus*tanh(s/csi_minus)*tanh(pow(1-s,2)/csi_minus)*(0.5+0.5*tanh((c_H-c)*pow(epsilon,-1)))',
                        V_plus=self.V_plus,V_minus=self.V_minus,csi_minus=self.csi_minus,csi_plus=self.csi_plus,c_H=self.c_H,
                        epsilon=self.epsilon,c=C,s=(s-0.5)*self.dz,degree=1)
                F = Expression("P - K", degree=1, P=P, K=K)
                a = (1/self.dt)*n*v*dx + self.Dxn*inner(grad(n),grad(v))*dx + 2*self.Dsn/(self.dz*self.dz)*n*v*dx - n*v*F*dx + 0.5/self.dz*vs_plus*n*v*dx - 0.5/self.dz*vs_minus*n*v*dx
                ar = -self.Dsn/(self.dz*self.dz)*n*v*dx + 0.5/(self.dz)*vs_plus*n*v*dx
                al = -self.Dsn/(self.dz*self.dz)*n*v*dx - 0.5/(self.dz)*vs_minus*n*v*dx

                Ac = assemble(a)
                Ar = assemble(ar)
                Al = assemble(al)

                new_row = [0] * (Ns+1)
                new_row[s] = Ac
                new_row[s-1] = Al
                new_row[s+1] = Ar
                matrix_list.append(new_row)

            s = Ns
            
            P = Expression('(p_csc*pow(c,4)/(pow(K_csc,4)+pow(c,4))*exp(-pow((s-s_csc)/g_csc,2)) \
                    + p_dc*pow(c,4)/(pow(K_dc,4)+pow(c,4))*exp(-pow((s-s_dc)/g_dc,2)))*(1-phi)',
                    p_csc=self.p_csc,p_dc=self.p_dc,K_csc=self.K_csc,K_dc=self.K_dc,g_csc=self.g_csc,g_dc=self.g_dc,
                    s_csc=self.s_csc,s_dc=self.s_dc,phi=phi,s=s*self.dz,c=C,degree=1)
            K = Expression('d_tdc * exp(-((1-s)/g_tdc)) + d_n * (0.5+0.5*tanh(pow(epsilon_k,-1)*(c_N-c)))',
                    d_tdc=self.d_tdc,d_n=self.d_n,g_tdc=self.g_tdc,epsilon_k=self.epsilon_k,c_N=self.c_N,c=C,s=s*self.dz,degree=1)
            F = Expression("P - K", degree=1, P=P, K=K)
            vs = Expression('V_plus*tanh(s/csi_plus)*tanh((1-s)/csi_plus)*(0.5+0.5*tanh((c-c_H)*pow(epsilon,-1))) \
                        - V_minus*tanh(s/csi_minus)*tanh(pow(1-s,2)/csi_minus)*(0.5+0.5*tanh((c_H-c)*pow(epsilon,-1)))',
                        V_plus=self.V_plus,V_minus=self.V_minus,csi_minus=self.csi_minus,csi_plus=self.csi_plus,c_H=self.c_H,
                        epsilon=self.epsilon,c=C,s=(s-0.5)*self.dz,degree=1)

            aNN = (1/self.dt)*n*v*dx + self.Dxn*inner(grad(n),grad(v))*dx + self.Dsn/(self.dz*self.dz)*n*v*dx - n*v*F*dx - 0.5/self.dz*vs*n*v*dx
            
            aNl = -self.Dsn/(self.dz*self.dz)*n*v*dx - 0.5/(self.dz)*vs*n*v*dx

            AN = assemble(aNN)
            ANl = assemble(aNl)
            lastrow = [0] * (Ns+1)
            lastrow[s] = AN
            lastrow[s-1] = ANl
            matrix_list.append(lastrow)
            
            # block matrix and vector assembly
            A = block_mat(matrix_list)
            b_list = []
            for s in range(Ns+1):
                L = (1/self.dt)*n0_array[s]*v*dx
                b_curr = assemble(L)
                b_list.append(b_curr)
            b = block_vec(b_list)
                   
            # solve linear system        
            AAinv = LGMRES(A, tolerance=1E-6, maxiter=500,show=1)
            x = AAinv * b

            # update n 
            for i in range(Ns+1):
                n0_array[i].vector()[:] = x[i]

            # update phi and compute mass 
            sum = 0
            for i in range(Ns):
                sum += self.dz*(x[i]+x[i+1])/2
            phi = Function(self.V)
            phi.vector()[:] = sum
            phi= interpolate(phi,self.V)
            mass.append(assemble(phi*dx))

            # save solution
            if t % self.save_interval == 0:
                cfile.write_checkpoint(C,"c",t,XDMFFile.Encoding.HDF5, True)
                phi_file.write_checkpoint(phi,"phi",t,XDMFFile.Encoding.HDF5, True)
                self.save_solution(x, t, Ns)
              
            # compute tumor composition
            csc = 0
            dc = 0
            tdc = 0
            i = 0
            while i*self.dz <= 0.3:
                csc += self.dz*(x[i]+x[i+1])/2
                i+=1
            csc_function = Function(self.V)
            csc_function.vector()[:] = csc
            csc_mass.append(assemble(csc_function*dx)/mass[-1])
            while i*self.dz <= 0.8:
                dc += self.dz*(x[i]+x[i+1])/2
                i+=1
            dc_function = Function(self.V)
            dc_function.vector()[:] = dc
            dc_mass.append(assemble(dc_function*dx)/mass[-1])
            while i < Ns:
                tdc += self.dz*(x[i]+x[i+1])/2
                i+=1
            tdc_function = Function(self.V)
            tdc_function.vector()[:] = tdc
            tdc_mass.append(assemble(tdc_function*dx)/mass[-1])

            d=0
            t+=1

        np.save(self.path_sol + "/" + "mass.npy", mass)
        np.save(self.path_sol + "/" + "csc_mass.npy", csc_mass)
        np.save(self.path_sol + "/" + "dc_mass.npy", dc_mass)
        np.save(self.path_sol + "/" + "tdc_mass.npy", tdc_mass)


class MatrixSolver2D(MatrixSolver):

    def __init__(self, mesh, V, n0, c_0, dt, T, save_interval, times, doses, path_sol, f,dz,bc_type):
        super().__init__(mesh, V, n0, c_0, dt, T, save_interval, times, doses, path_sol, f,dz,bc_type)

    def boundaries(self):
    
        boundary_markers = MeshFunction('size_t', self.mesh,1)

        class BoundaryX0(SubDomain):
            tol = 1E-14
            def inside(self, x, on_boundary):
                return on_boundary and near(x[0], -1, 1E-14)

        self.bx0 = BoundaryX0()
        self.bx0.mark(boundary_markers, 3)

        class BoundaryX1(SubDomain):
            tol = 1E-14
            def inside(self, x, on_boundary):
                return on_boundary and near(x[0], 1, 1E-14)

        self.bx1 = BoundaryX1()
        self.bx1.mark(boundary_markers, 1)

        class BoundaryY1(SubDomain):
            tol = 1E-14
            def inside(self, x, on_boundary):
                return on_boundary and near(x[1], 1, 1E-14)

        self.by1 = BoundaryY1()
        self.by1.mark(boundary_markers, 2)

        class BoundaryY0(SubDomain):
            tol = 1E-14
            def inside(self, x, on_boundary):
                return on_boundary and near(x[1], -1, 1E-14)

        self.by0 = BoundaryY0()
        self.by0.mark(boundary_markers, 0)

        self.ds = Measure("ds",domain=self.mesh, subdomain_data=boundary_markers)

    def vertical_average(self,n0_array,Ns):
        phi = va.VerticalAverage2D(self.n0, 20, degree=2)
        phi_h = interpolate(phi, self.V)
        return phi_h
    
    def oxygen_bc(self,bc_type):
        if bc_type == 'Dirichlet_0':
            bc_x1 = DirichletBC(self.V,Constant(0.0),self.bx1)
            bc_x0 = DirichletBC(self.V,Constant(0.0),self.bx0)
            bc_y0 = DirichletBC(self.V,Constant(0.0),self.by0)
            bc_y1 = DirichletBC(self.V,Constant(0.0),self.by1)
            bcs_c = [bc_x1, bc_x0, bc_y0, bc_y1]
        elif bc_type == 'Dirichlet_1':
            bcs_c = [DirichletBC(self.V, 1, self.bx1)]
        return bcs_c
    
    def save_solution(self, sol, t, Ns):
        vec = []
        f2D = Function(self.V)
        for i in range(Ns+1):
            f2D.vector()[:] = sol[i]
            vec.extend(f2D.compute_vertex_values())
        
        # mesh3D = UnitCubeMesh(20,20,Ns)
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # sol = ax.scatter(mesh3D.coordinates()[:,0],mesh3D.coordinates()[:,1],mesh3D.coordinates()[:,2],c=vec)
        # plt.colorbar(sol)
        # plt.show()

        m = pv.read('boxmesh.vtk')        
        temp_path = os.path.join(self.path_sol, 'solution')
        if not os.path.exists(temp_path):
            os.mkdir(temp_path)

        sol_points = np.array(vec)

        # riempire sol points
        m1 = copy.deepcopy(m)
        m1['solution'] = sol_points
        m1.save(os.path.join(temp_path, f'sol_{t}.vtk'))
            
class MatrixSolver3D(MatrixSolver):

    def __init__(self, mesh, V, n0, c_0, dt, T, save_interval, times, doses, path_sol, f,dz,bc_type):
        super().__init__(mesh, V, n0, c_0, dt, T, save_interval, times, doses, path_sol, f,dz,bc_type)

    def boundaries(self):
        boundary_markers = MeshFunction('size_t', self.mesh,2)

        class BoundaryX0(SubDomain):
            tol = 1E-14
            def inside(self, x, on_boundary):
                return on_boundary and near(x[0], 0, 1E-14)

        self.bx0 = BoundaryX0()
        self.bx0.mark(boundary_markers, 3)

        class BoundaryX1(SubDomain):
            tol = 1E-14
            def inside(self, x, on_boundary):
                return on_boundary and near(x[0], 1, 1E-14)

        self.bx1 = BoundaryX1()
        self.bx1.mark(boundary_markers, 1)

        class BoundaryY1(SubDomain):
            tol = 1E-14
            def inside(self, x, on_boundary):
                return on_boundary and near(x[1], 1, 1E-14)

        self.by1 = BoundaryY1()
        self.by1.mark(boundary_markers, 2)

        class BoundaryY0(SubDomain):
            tol = 1E-14
            def inside(self, x, on_boundary):
                return on_boundary and near(x[1], 0, 1E-14)

        self.by0 = BoundaryY0()
        self.by0.mark(boundary_markers, 0)

        class BoundaryZ1(SubDomain):
            tol = 1E-14
            def inside(self, x, on_boundary):
                return on_boundary and near(x[2], 0.5, 1E-14)

        self.by1 = BoundaryZ1()
        self.by1.mark(boundary_markers, 4)

        class BoundaryZ0(SubDomain):
            tol = 1E-14
            def inside(self, x, on_boundary):
                return on_boundary and near(x[2], 0, 1E-14)

        self.by0 = BoundaryZ0()
        self.by0.mark(boundary_markers, 5)

        self.ds = Measure("ds",domain=self.mesh, subdomain_data=boundary_markers)

    def vertical_average(self,n0_array,Ns):
        sum = 0
        phi = Function(self.V)
        for s in range(Ns+1):
            sum += n0_array[s].vector()[:]
        sum = sum/Ns
        phi.vector()[:] = sum
        phi = interpolate(phi,self.V)
        return phi
    
    def save_solution(self, sol, t, Ns):
        
        slices = [0, int(Ns/2), Ns]
        for s in slices:
            f3D = Function(self.V)
            f3D.vector()[:] = sol[s]

            m = pv.read('mesh3D.vtk')        
            temp_path = os.path.join(self.path_sol, f's_{s}')
            if not os.path.exists(temp_path):
                os.mkdir(temp_path)

            sol_points = np.array(f3D.compute_vertex_values())

            # riempire sol points
            m1 = copy.deepcopy(m)
            m1['n'] = sol_points
            m1.save(os.path.join(temp_path, f'sol_{t}.vtk'))

    def oxygen_bc(self,bc_type):
        bcs_c = [DirichletBC(self.V, 1, self.bx1)]
        return bcs_c