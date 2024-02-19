from dolfin import *
from numpy.polynomial.legendre import leggauss
import numpy as np
from ufl import tanh
import matplotlib.pyplot as plt
from scipy import sparse
from xii import *
from block import *
from scipy.sparse.linalg import *

class VerticalAverage(UserExpression):
    def __init__(self, f, quad_degree, **kwargs):
        super().__init__(**kwargs)
        self.f = f
        self.points, self.weights = leggauss(quad_degree)
        self.x = 0.5*(self.points + 1)
        assert f.ufl_shape == ()
        
    def eval(self, values, x):
        values[0] = 0.5*sum(wq*self.f(x[0], xq) for xq, wq in zip(self.x, self.weights))

    def value_shape(self):
        return ()


class Solver3D:

    def __init__(self, mesh, V, n0, c_0, dt, T, save_interval, times, doses, path_sol):
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

    def set_parameters(self, dict_params):
        for key, value in dict_params.items():
            self.key = value

    def boundaries(self):
    
        boundary_markers = MeshFunction('size_t', self.mesh,1)
        # subdomains_markers = MeshFunction('size_t', self.mesh,2)

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

        # class BoundaryY1(SubDomain):
        #     tol = 1E-14
        #     def inside(self, x, on_boundary):
        #         return on_boundary and near(x[1], 1, 1E-14)

        # self.by1 = BoundaryY1()
        # self.by1.mark(boundary_markers, 2)

        # class BoundaryY0(SubDomain):
        #     tol = 1E-14
        #     def inside(self, x, on_boundary):
        #         return on_boundary and near(x[1], -1, 1E-14)

        # self.by0 = BoundaryY0()
        # self.by0.mark(boundary_markers, 0)

        self.ds = Measure("ds",domain=self.mesh, subdomain_data=boundary_markers)

        # return self.bx1,self.bx0,self.by0,self.by1,self.bs0,self.bs1,self.ds,self.dx

    
    def solve(self):

        self.boundaries()
        n_steps = int(self.T/self.dt)

        nfile = XDMFFile(self.path_sol + "/" + "n.xdmf")
        cfile = XDMFFile(self.path_sol + "/" + "c.xdmf")
        phifile = XDMFFile(self.path_sol + "/" + "phi.xdmf")
    
        #V = FunctionSpace(mesh,"P",2)
        N = Function(self.V)
        C = Function(self.V)
        # n = TrialFunction(self.V)
        v = TestFunction(self.V)
        
        c = TrialFunction(self.V)
        w = TestFunction(self.V)
        phi = Function(self.V)
        nh0 = Function(self.V)
        nh1 = Function(self.V)

        

        f = Constant(0.0)
        # L_c = f*w*dx 

        # Vc = Expression('1e8*exp(-x[0]*x[0]/(sigma_v*sigma_v) - x[1]*x[1]/(sigma_v*sigma_v))',sigma_v = 1e-3,degree=2)
        
        bc_x1 = DirichletBC(self.V,Constant(1.0),self.bx1)
        # bc_x0 = DirichletBC(self.V,Constant(0.0),self.bx0)
        # bc_y0 = DirichletBC(self.V,Constant(0.0),self.by0)
        # bc_y1 = DirichletBC(self.V,Constant(0.0),self.by1)
        bcs_c = [bc_x1]

        mass = []
        n_vect = []
        c_vect = []
        csc_mass = []
        dc_mass = []
        tdc_mass = []
        phi_vect = []

        t=0
        i=0
        d=0
        tol = 1.0E-3    
        maxiter = 30  

        nfile = XDMFFile(self.path_sol + "/" + "n.xdmf")
        cfile = XDMFFile(self.path_sol + "/" + "c.xdmf")
        nfile.parameters['rewrite_function_mesh'] = False
        cfile.parameters['rewrite_function_mesh'] = False

        phi = VerticalAverage(self.n0, quad_degree=20, degree=2)
        phi_h = interpolate(phi, self.V)

        dz = 0.01
        Ns = int(1/dz)

        n0_array = [Function(self.V) for _ in range(Ns-1)]
        for s in range(Ns-1):
            self.n0.s = (s+1)*dz
            temp = Function(self.V)
            temp.assign(self.n0)
            n0_array[s].vector()[:] = temp.vector()[:]
            # n0_array[s] = interpolate(n0_array[s],self.V)
            # plot(n0_array[s])
            # plt.show()

        x_vect = []
        mesh2D = UnitSquareMesh(20,Ns)
        V2D = FunctionSpace(mesh2D,"P",1)
        n0_2D = Function(V2D)

        nmesh = 21

        while(t < n_steps):
            print('time=%g: ' %(t*self.dt))

            a_c = self.Dxc * inner(grad(c),grad(w))*dx + self.gamma/(self.c_k + self.K_m)*phi_h*c*w*dx
            L_c = f*w*dx
            
            iter = 0  
            eps= 1.0
            while eps > tol and iter < maxiter:
                iter += 1
                solve(a_c==L_c,C,bcs_c)
                difference = C.vector() - self.c_k.vector()
                eps = np.linalg.norm(difference) / np.linalg.norm(self.c_k.vector())
                print('iter=%d: norm=%g' % (iter, eps))
                self.c_k.assign(C) 

            # ch = interpolate(C,self.V)
            # plot(ch)
            # plt.title('oxygen')
            # plt.show()

            # if t % self.save_interval == 0:
            # c_vect.append(C.vector().get_local().copy())
            cfile.write_checkpoint(C,"c",t,XDMFFile.Encoding.HDF5, True)
            
            

            # n0_array = [Function(self.V) for _ in range(s_steps)]

            s = 1

            P = Expression('(p_csc*pow(c,4)/(pow(K_csc,4)+pow(c,4))*exp(-pow((s-s_csc)/g_csc,2)) \
                    + p_dc*pow(c,4)/(pow(K_dc,4)+pow(c,4))*exp(-pow((s-s_dc)/g_dc,2)))*(1-phi)',
                    p_csc=self.p_csc,p_dc=self.p_dc,K_csc=self.K_csc,K_dc=self.K_dc,g_csc=self.g_csc,g_dc=self.g_dc,
                    s_csc=self.s_csc,s_dc=self.s_dc,phi=phi_h,s=s*dz,c=C,degree=1)
            K = Expression('d_tdc * exp(-((1-s)/g_tdc)) + d_n * (0.5+0.5*tanh(pow(epsilon_k,-1)*(c_N-c)))',
                    d_tdc=self.d_tdc,d_n=self.d_n,g_tdc=self.g_tdc,epsilon_k=self.epsilon_k,c_N=self.c_N,c=C,s=s*dz,degree=1)
            vs = Expression('V_plus*tanh(s/csi_plus)*tanh((1-s)/csi_plus)*(0.5+0.5*tanh((c-c_H)*pow(epsilon,-1))) \
                        - V_minus*tanh(s/csi_minus)*tanh(pow(1-s,2)/csi_minus)*(0.5+0.5*tanh((c_H-c)*pow(epsilon,-1)))',
                        V_plus=self.V_plus,V_minus=self.V_minus,csi_minus=self.csi_minus,csi_plus=self.csi_plus,c_H=self.c_H,
                        epsilon=self.epsilon,c=C,s=(s+1)*dz,degree=1)

            a1 = Expression('(c > c_R) ? 1 : 1/OER',c_R=self.c_R,OER=self.OER,c=C,degree=1)
            a2 = Expression('alpha_min + delta_alpha*tanh(k*s)',alpha_min=self.alpha_min,delta_alpha=self.delta_alpha,k=self.k,s=s*dz,degree=2)
            a3 = Expression('1+P/Pmax', P=P,Pmax=self.p_dc,degree=1)
            a = Expression('a1*a2*a3',a1=a1,a2=a2,a3=a3,degree=1)
            #a = interpolate(a,V)
            b1 = Expression('(c > c_R) ? 1 : 1/(OER*OER)',c_R=self.c_R,OER=self.OER,c=C,degree=1)
            b2 = Expression('beta_min + delta_beta*tanh(k*s)',beta_min=self.beta_min,delta_beta=self.delta_beta,k=self.k,s=s*dz,degree=2)
            b = Expression('b1*b2*b3',b1=b1,b2=b2,b3=a3,degree=1)
            if i < len(self.times) and self.t*self.dt == self.times[i]: 
                    print('dose')
                    d=self.doses[i]
                    print(d)
                    i+=1

            S_rt = Expression('-a*d - b*d*d', a=a, b=b, d=d,degree=1)
            F = Expression("P - K", degree=1, P=P, K=K)

            # n = [TrialFunction(self.V) for _ in range(Ns-1)]
            n = TrialFunction(self.V)
            
            matrix_list = []
            # A = BlockMatrix(Ns-1,Ns-1)

            a00 = (1/self.dt)*n*v*dx + self.Dxn*inner(grad(n),grad(v))*dx + self.Dsn/(2*dz*dz)*n*v*dx - n*v*F*dx
            # vs = Expression('V_plus*tanh(s/csi_plus)*tanh((1-s)/csi_plus)*(0.5+0.5*tanh((c-c_H)*pow(epsilon,-1))) \
            #             - V_minus*tanh(s/csi_minus)*tanh(pow(1-s,2)/csi_minus)*(0.5+0.5*tanh((c_H-c)*pow(epsilon,-1)))',
            #             V_plus=self.V_plus,V_minus=self.V_minus,csi_minus=self.csi_minus,csi_plus=self.csi_plus,c_H=self.c_H,
            #             epsilon=self.epsilon,c=C,s=s*dz,degree=2)
            a01 = -self.Dsn/(2*dz*dz)*n*v*dx + 1/(2*dz)*vs*n*v*dx

            A0 = assemble(a00)
            A0r = assemble(a01)
            row1 = [0] * (Ns-1)
            row1[0] = A0
            row1[1] = A0r
            
        
            matrix_list.append(row1)

            for s in range(1,Ns-2):
                P = Expression('(p_csc*pow(c,4)/(pow(K_csc,4)+pow(c,4))*exp(-pow((s-s_csc)/g_csc,2)) \
                    + p_dc*pow(c,4)/(pow(K_dc,4)+pow(c,4))*exp(-pow((s-s_dc)/g_dc,2)))*(1-phi)',
                    p_csc=self.p_csc,p_dc=self.p_dc,K_csc=self.K_csc,K_dc=self.K_dc,g_csc=self.g_csc,g_dc=self.g_dc,
                    s_csc=self.s_csc,s_dc=self.s_dc,phi=phi_h,s=(s+1)*dz,c=C,degree=1)
                K = Expression('d_tdc * exp(-((1-s)/g_tdc)) + d_n * (0.5+0.5*tanh(pow(epsilon_k,-1)*(c_N-c)))',
                    d_tdc=self.d_tdc,d_n=self.d_n,g_tdc=self.g_tdc,epsilon_k=self.epsilon_k,c_N=self.c_N,c=C,s=(s+1)*dz,degree=2)
                vs = Expression('V_plus*tanh(s/csi_plus)*tanh((1-s)/csi_plus)*(0.5+0.5*tanh((c-c_H)*pow(epsilon,-1))) \
                        - V_minus*tanh(s/csi_minus)*tanh(pow(1-s,2)/csi_minus)*(0.5+0.5*tanh((c_H-c)*pow(epsilon,-1)))',
                        V_plus=self.V_plus,V_minus=self.V_minus,csi_minus=self.csi_minus,csi_plus=self.csi_plus,c_H=self.c_H,
                        epsilon=self.epsilon,c=C,s=(s+2)*dz,degree=1)
                F = Expression("P - K", degree=2, P=P, K=K)
                a = (1/self.dt)*n*v*dx + self.Dxn*inner(grad(n),grad(v))*dx + 2*self.Dsn/(2*dz*dz)*n*v*dx - n*v*F*dx
                ar = -self.Dsn/(2*dz*dz)*n*v*dx + 1/(2*dz)*vs*n*v*dx
                vs = Expression('V_plus*tanh(s/csi_plus)*tanh((1-s)/csi_plus)*(0.5+0.5*tanh((c-c_H)*pow(epsilon,-1))) \
                        - V_minus*tanh(s/csi_minus)*tanh(pow(1-s,2)/csi_minus)*(0.5+0.5*tanh((c_H-c)*pow(epsilon,-1)))',
                        V_plus=self.V_plus,V_minus=self.V_minus,csi_minus=self.csi_minus,csi_plus=self.csi_plus,c_H=self.c_H,
                        epsilon=self.epsilon,c=C,s=(s)*dz,degree=1)
                al = -self.Dsn/(2*dz*dz)*n*v*dx - 1/(2*dz)*vs*n*v*dx
                Ac = assemble(a)
                Ar = assemble(ar)
                Al = assemble(al)

                new_row = [0] * (Ns-1)
                new_row[s] = Ac
                new_row[s-1] = Al
                new_row[s+1] = Ar
                matrix_list.append(new_row)

            s = Ns-2

            P = Expression('(p_csc*pow(c,4)/(pow(K_csc,4)+pow(c,4))*exp(-pow((s-s_csc)/g_csc,2)) \
                    + p_dc*pow(c,4)/(pow(K_dc,4)+pow(c,4))*exp(-pow((s-s_dc)/g_dc,2)))*(1-phi)',
                    p_csc=self.p_csc,p_dc=self.p_dc,K_csc=self.K_csc,K_dc=self.K_dc,g_csc=self.g_csc,g_dc=self.g_dc,
                    s_csc=self.s_csc,s_dc=self.s_dc,phi=phi_h,s=(s+1)*dz,c=C,degree=1)
            K = Expression('d_tdc * exp(-((1-s)/g_tdc)) + d_n * (0.5+0.5*tanh(pow(epsilon_k,-1)*(c_N-c)))',
                    d_tdc=self.d_tdc,d_n=self.d_n,g_tdc=self.g_tdc,epsilon_k=self.epsilon_k,c_N=self.c_N,c=C,s=(s+1)*dz,degree=2)
            F = Expression("P - K", degree=1, P=P, K=K)

            aNN = (1/self.dt)*n*v*dx + self.Dxn*inner(grad(n),grad(v))*dx + self.Dsn/(2*dz*dz)*n*v*dx - n*v*F*dx
            vs = Expression('V_plus*tanh(s/csi_plus)*tanh((1-s)/csi_plus)*(0.5+0.5*tanh((c-c_H)*pow(epsilon,-1))) \
                        - V_minus*tanh(s/csi_minus)*tanh(pow(1-s,2)/csi_minus)*(0.5+0.5*tanh((c_H-c)*pow(epsilon,-1)))',
                        V_plus=self.V_plus,V_minus=self.V_minus,csi_minus=self.csi_minus,csi_plus=self.csi_plus,c_H=self.c_H,
                        epsilon=self.epsilon,c=C,s=(s)*dz,degree=1)
            aNl = -self.Dsn/(2*dz*dz)*n*v*dx - 1/(2*dz)*vs*n*v*dx

            AN = assemble(aNN)
            ANl = assemble(aNl)
            lastrow = [0] * (Ns-1)
            lastrow[-1] = AN
            lastrow[-2] = ANl
            matrix_list.append(lastrow)
            
            A = block_mat(matrix_list)
            
        
            b_list = []
            for s in range(Ns-1):
                L = (1/self.dt)*n0_array[s]*v*dx
                b_curr = assemble(L)
                b_list.append(b_curr)
                # print(b_curr.size())
            b = block_vec(b_list)
            # print(b)

            # A = A.block_collapse()
            AA = ii_convert(A)
            AA = AA.mat()
            # print(AA.getValuesCSR())
            # print(AA.getValuesCSR()[::-1])
            A_CSR=sparse.csr_matrix(AA.getValuesCSR()[::-1],shape=AA.size)
            print(A_CSR.shape)
            B = ii_convert(b)
            B_CSR=sparse.csr_matrix(B)
            B_CSR=B_CSR.transpose()
                
            # print('a',AA.size)
            # print(A_CSR.toarray().shape)
            # print(B_CSR.toarray().shape)

            
            # _,s,_=svds(A_CSR,k=50)
            # cond=max(s)/min(s)
            # print(cond)

            x,exit=gmres(A_CSR,B_CSR.toarray())

            sum = 0
            # for i in range(Ns-2):
            #     sum += dz*(x[nmesh*(i+1):nmesh*(i+2)] + x[nmesh*i:nmesh*(i+1)])/2
            # phi = Function(self.V)
            # phi.vector()[:] = sum
            # phi_h = interpolate(phi,self.V)
            # plot(phi_h)
            # plt.show()
        
            for s in range(Ns-1):
                n0_array[s].vector()[:] = x[nmesh*s:nmesh*(s+1)]

            # xnew = np.pad(x, (21, 21)) 
            xnew = np.concatenate([x[0:nmesh],x,x[-nmesh:]])

            for i in range(Ns):
                sum += dz*(xnew[nmesh*(i+1):nmesh*(i+2)] + xnew[nmesh*i:nmesh*(i+1)])/2
            phi = Function(self.V)
            phi.vector()[:] = sum
            phi_h = interpolate(phi,self.V)
            # plot(phi_h)
            # plt.show()


            # for s in range(0,Ns+1): 
            #     xnew[nmesh*s:nmesh*(s+1)] = np.flip(xnew[nmesh*s:nmesh*(s+1)])
            xnew = np.array([xnew[i] for i in dof_to_vertex_map(V2D)])

            n0_2D.vector()[:] = xnew
            n0_2D = interpolate(n0_2D,V2D)
            plot(n0_2D)
            plt.colorbar(plot(n0_2D))
            plt.show()
            # plt.scatter(mesh2D.coordinates()[:,0],mesh2D.coordinates()[:,1],c=xnew)
            # plt.colorbar()
            # plt.show()

            subdomains_markers = MeshFunction('size_t', mesh2D,2)
            class Omega_CSC(SubDomain):
                def inside(self, x, on_boundary):
                    return x[1] <= 0.3 + 1E-14

            class Omega_DC(SubDomain):
                def inside(self, x, on_boundary): 
                    return x[1] > 0.3 - 1E-14 and x[1] <= 0.8 + 1E-14
                
            class Omega_TDC(SubDomain):
                def inside(self, x, on_boundary): 
                    return x[1] > 0.8 - 1E-14
                
            subdomain_CSC = Omega_CSC()
            subdomain_DC = Omega_DC()
            subdomain_TDC = Omega_TDC()
            subdomain_CSC.mark(subdomains_markers, 0)
            subdomain_DC.mark(subdomains_markers, 1)
            subdomain_TDC.mark(subdomains_markers, 2)
            self.dx = Measure("dx",domain=mesh2D, subdomain_data=subdomains_markers)
            mass.append(assemble(phi_h*dx))
            csc_mass.append(assemble(n0_2D*self.dx(0))/mass[-1])
            dc_mass.append(assemble(n0_2D*self.dx(1))/mass[-1])
            tdc_mass.append(assemble(n0_2D*self.dx(2))/mass[-1])

            nfile.write_checkpoint(n0_2D,"n",t,XDMFFile.Encoding.HDF5, True)

            x_vect.append(x)

            d=0
            t+=1
        # np.save(self.path_sol + "/" + "n.npy", x_vect)
        np.save(self.path_sol + "/" + "mass.npy", mass)
        np.save(self.path_sol + "/" + "csc_mass.npy", csc_mass)
        np.save(self.path_sol + "/" + "dc_mass.npy", dc_mass)
        np.save(self.path_sol + "/" + "tdc_mass.npy", tdc_mass)

        return mass, x_vect
    