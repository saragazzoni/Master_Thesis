from dolfin import *
from numpy.polynomial.legendre import leggauss
import numpy as np
from ufl import tanh
import math

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
    

class Solver1D:

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
        self.K_m = 0.01
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
            setattr(self, key, value)

    def boundaries(self):
        
        boundary_markers = MeshFunction('size_t', self.mesh,1)
        subdomains_markers = MeshFunction('size_t', self.mesh,2)

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

        self.ds = Measure("ds",domain=self.mesh, subdomain_data=boundary_markers)
        self.dx = Measure("dx",domain=self.mesh, subdomain_data=subdomains_markers)

        return self.bx1,self.bx0,self.by0,self.by1,self.ds,self.dx

    def solve(self):

        self.boundaries()
        n_steps = int(self.T/self.dt)
        N = Function(self.V)
        C = Function(self.V)
        n = TrialFunction(self.V)
        v = TestFunction(self.V)
        c = TrialFunction(self.V)
        w = TestFunction(self.V)

        f = Constant(0.0)
        #L_c = f*w*dx
        # bc_c1 = DirichletBC(V,Constant(0.5),bx0)
        bc_c = DirichletBC(self.V,Constant(1.0),self.bx1)
        bcs_c = [bc_c]

        mass = []
        n_vect = []
        c_vect = []
        csc_mass = []
        dc_mass = []
        tdc_mass = []

        nfile = XDMFFile(self.path_sol + "/" + "n.xdmf")
        cfile = XDMFFile(self.path_sol + "/" + "c.xdmf")
        nfile.parameters['rewrite_function_mesh'] = False
        cfile.parameters['rewrite_function_mesh'] = False

        t=0
        i=0
        d=0

        # Vc = Expression('exp(-(x[0]-x0)*(x[0]-x0)/(sigma_v*sigma_v))',sigma_v = 0.01, x0=0.5, degree=2)
        # bc_c1 = DirichletBC(self.V,Vc,self.bx1)
        # bc_c0 = DirichletBC(self.V,Vc,self.bx0)
        # bcs_c = [bc_c1]

        while(t<n_steps):
            print('time=%g: ' %(t*self.dt))

            # update phi
            phi = VerticalAverage(self.n0, quad_degree=20, degree=2)
            phi_h = interpolate(phi, self.V)
            a_c = self.Dxc * inner(grad(c)[0],grad(w)[0])*dx + self.gamma/(self.c_k + self.K_m)*phi_h*c*w*dx
            L_c = f*w*dx #- self.gamma*(0.5 + 0.5*tanh((self.c_k-self.c_N)/0.05))*phi_h*w*dx
            
            # fixed point -> solve c
            eps= 1.0
            tol = 1.0E-3
            iter = 0        
            maxiter = 30    
            while eps > tol and iter < maxiter:
                iter += 1
                solve(a_c==L_c,C,bcs_c)
                difference = C.vector() - self.c_k.vector()
                eps = np.linalg.norm(difference) / np.linalg.norm(self.c_k.vector())
                print('iter=%d: norm=%g' % (iter, eps))
                # print('cmin:', C.vector().min())
                self.c_k.assign(C) 

            if t % self.save_interval == 0:
                n_vect.append(self.n0.vector().get_local().copy())
                c_vect.append(C.vector().get_local().copy())
                #phi_vect.append(phi_h)

                nfile.write_checkpoint(self.n0,"n",t,XDMFFile.Encoding.HDF5, True)
                cfile.write_checkpoint(C,"c",t,XDMFFile.Encoding.HDF5, True)

            mass.append(assemble(phi_h*dx))
            csc_mass.append(assemble(self.n0*self.dx(0))/mass[-1])
            dc_mass.append(assemble(self.n0*self.dx(1))/mass[-1])
            tdc_mass.append(assemble(self.n0*self.dx(2))/mass[-1])
        
            # solve n
            P = Expression('(p_csc*pow(c,4)/(pow(K_csc,4)+pow(c,4))*exp(-pow((x[1]-s_csc)/g_csc,2)) \
                    + p_dc*pow(c,4)/(pow(K_dc,4)+pow(c,4))*exp(-pow((x[1]-s_dc)/g_dc,2)))*(1-phi)',
                    p_csc=self.p_csc,p_dc=self.p_dc,K_csc=self.K_csc,K_dc=self.K_dc,g_csc=self.g_csc,g_dc=self.g_dc,
                    s_csc=self.s_csc,s_dc=self.s_dc,phi=phi_h,c=C,degree=2)
            K = Expression('d_tdc * exp(-((1-x[1])/g_tdc)) + d_n * (0.5+0.5*tanh(pow(epsilon_k,-1)*(c_N-c)))',
                    d_tdc=self.d_tdc,d_n=self.d_n,g_tdc=self.g_tdc,epsilon_k=self.epsilon_k,c_N=self.c_N,c=C,degree=2)
            
            vs = Expression('V_plus*tanh(x[1]/csi_plus)*tanh((1-x[1])/csi_plus)*(0.5+0.5*tanh((c-c_H)*pow(epsilon,-1))) \
                        - V_minus*tanh(x[1]/csi_minus)*tanh(pow(1-x[1],2)/csi_minus)*(0.5+0.5*tanh((c_H-c)*pow(epsilon,-1)))',
                        V_plus=self.V_plus,V_minus=self.V_minus,csi_minus=self.csi_minus,csi_plus=self.csi_plus,c_H=self.c_H,
                        epsilon=self.epsilon,c=C,degree=2)
            vs = interpolate(vs,self.V)

            a1 = Expression('(c > c_R) ? 1 : 1/OER',c_R=self.c_R,OER=self.OER,c=C,degree=2)
            a2 = Expression('alpha_min + delta_alpha*tanh(k*x[1])',alpha_min=self.alpha_min,delta_alpha=self.delta_alpha,k=self.k,degree=2)
            a3 = Expression('1+P/Pmax', P=P,Pmax=self.p_dc,degree=2)
            a = Expression('a1*a2*a3',a1=a1,a2=a2,a3=a3,degree=2)
            #a = interpolate(a,V)
            b1 = Expression('(c > c_R) ? 1 : 1/(OER*OER)',c_R=self.c_R,OER=self.OER,c=C,degree=2)
            b2 = Expression('beta_min + delta_beta*tanh(k*x[1])',beta_min=self.beta_min,delta_beta=self.delta_beta,k=self.k,degree=2)
            b = Expression('b1*b2*b3',b1=b1,b2=b2,b3=a3,degree=2)
            #b = interpolate(b,V)

            if i < len(self.times) and t*self.dt == self.times[i]: #math.floor(t*dt) == times[i]:  
                print('dose')
                d=self.doses[i]
                print(d)
                i+=1
                    
            S_rt = Expression('-a*d - b*d*d', a=a, b=b, d=d,degree=2)
            F = Expression("P - K", degree=2, P=P, K=K)

            a_n = n*v*dx + self.dt*self.Dxn*inner(grad(n)[0],grad(v)[0])*dx + self.dt*self.Dsn*inner(grad(n)[1],grad(v)[1])*dx \
                  + self.dt*grad(vs*n)[1]*v*dx - self.dt*n*v*F*dx - n*v*S_rt*dx  + self.dt*n*v*vs*self.ds(0) - self.dt*n*v*vs*self.ds(2) 
            L_n = self.n0*v*dx

            solve(a_n==L_n,N)
            self.n0.assign(N)

            d=0
            t+=1
        
        np.save(self.path_sol + "/" + "mass.npy", mass)
        np.save(self.path_sol + "/" + "csc_mass.npy", csc_mass)
        np.save(self.path_sol + "/" + "dc_mass.npy", dc_mass)
        np.save(self.path_sol + "/" + "tdc_mass.npy", tdc_mass)

        return n_vect,c_vect,mass,csc_mass,dc_mass,tdc_mass
    