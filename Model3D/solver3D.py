from dolfin import *
from numpy.polynomial.legendre import leggauss
import numpy as np
from ufl import tanh
import matplotlib.pyplot as plt

class VerticalAverage(UserExpression):
    def __init__(self, f, quad_degree, **kwargs):
        super().__init__(**kwargs)
        self.f = f
        self.points, self.weights = leggauss(quad_degree)
        self.x = 0.5*(self.points + 1)
        assert f.ufl_shape == ()
        
    def eval(self, values, x):
        values[0] = 0.5*sum(wq*self.f(x[0],x[1], xq) for xq, wq in zip(self.x, self.weights))

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
        n = TrialFunction(self.V)
        v = TestFunction(self.V)
        c = TrialFunction(self.V)
        w = TestFunction(self.V)
        phi = Function(self.V)
        nh0 = Function(self.V)
        nh1 = Function(self.V)

        dz = 0.05
        s_steps = int(1/dz)
        n0_array = [Function(self.V) for _ in range(s_steps)]

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

        nfile.parameters['rewrite_function_mesh'] = False
        cfile.parameters['rewrite_function_mesh'] = False

        for s in range(s_steps):
            temp = Function(self.V)
            self.n0.s = s*dz
            temp.assign(self.n0)
            n0_array[s] = temp
            temp = interpolate(temp,self.V)
            phi.assign(phi + temp)
        phi.assign(phi/s_steps)
        phi = interpolate(phi,self.V)

        while(t < n_steps):
            print('time=%g: ' %(t*self.dt))

            a_c = self.Dxc * inner(grad(c),grad(w))*dx + self.gamma/(self.c_k + self.K_m)*phi*c*w*dx
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

            if t % self.save_interval == 0:
                # n_vect.append(n0_array.vector().get_local().copy())
                c_vect.append(C.vector().get_local().copy())
                # nfile.write_checkpoint(self.n0,"n",t,XDMFFile.Encoding.HDF5, True)
                cfile.write_checkpoint(C,"c",t,XDMFFile.Encoding.HDF5, True)
            # c_kh = interpolate(self.c_k,self.V)
            # plot(c_kh)
            # plt.show()
            
            mass.append(assemble(phi*dx))

            P = Expression('(p_csc*pow(c,4)/(pow(K_csc,4)+pow(c,4))*exp(-pow((s-s_csc)/g_csc,2)) \
                        + p_dc*pow(c,4)/(pow(K_dc,4)+pow(c,4))*exp(-pow((s-s_dc)/g_dc,2)))*(1-phi)',
                        p_csc=self.p_csc,p_dc=self.p_dc,K_csc=self.K_csc,K_dc=self.K_dc,g_csc=self.g_csc,g_dc=self.g_dc,
                        s_csc=self.s_csc,s_dc=self.s_dc,phi=phi,s=dz,c=C,degree=2)
            K = Expression('d_tdc * exp(-((1-s)/g_tdc)) + d_n * (0.5+0.5*tanh(pow(epsilon_k,-1)*(c_N-c)))',
                    d_tdc=self.d_tdc,d_n=self.d_n,g_tdc=self.g_tdc,epsilon_k=self.epsilon_k,c_N=self.c_N,c=C,s=dz,degree=2)
            vs = Expression('V_plus*tanh(s/csi_plus)*tanh((1-s)/csi_plus)*(0.5+0.5*tanh((c-c_H)*pow(epsilon,-1))) \
                        - V_minus*tanh(s/csi_minus)*tanh(pow(1-s,2)/csi_minus)*(0.5+0.5*tanh((c_H-c)*pow(epsilon,-1)))',
                        V_plus=self.V_plus,V_minus=self.V_minus,csi_minus=self.csi_minus,csi_plus=self.csi_plus,c_H=self.c_H,
                        epsilon=self.epsilon,c=C,s=dz,degree=2)
            vs0 = Expression('V_plus*tanh(s/csi_plus)*tanh((1-s)/csi_plus)*(0.5+0.5*tanh((c-c_H)*pow(epsilon,-1))) \
                        - V_minus*tanh(s/csi_minus)*tanh(pow(1-s,2)/csi_minus)*(0.5+0.5*tanh((c_H-c)*pow(epsilon,-1)))',
                        V_plus=self.V_plus,V_minus=self.V_minus,csi_minus=self.csi_minus,csi_plus=self.csi_plus,c_H=self.c_H,
                        epsilon=self.epsilon,c=C,s=0,degree=2)
            # vs = interpolate(vs,self.V)

            a1 = Expression('(c > c_R) ? 1 : 1/OER',c_R=self.c_R,OER=self.OER,c=C,degree=2)
            a2 = Expression('alpha_min + delta_alpha*tanh(k*s)',alpha_min=self.alpha_min,delta_alpha=self.delta_alpha,k=self.k,s=dz,degree=2)
            a3 = Expression('1+P/Pmax', P=P,Pmax=self.p_dc,degree=2)
            a = Expression('a1*a2*a3',a1=a1,a2=a2,a3=a3,degree=2)
            #a = interpolate(a,V)
            b1 = Expression('(c > c_R) ? 1 : 1/(OER*OER)',c_R=self.c_R,OER=self.OER,c=C,degree=2)
            b2 = Expression('beta_min + delta_beta*tanh(k*s)',beta_min=self.beta_min,delta_beta=self.delta_beta,k=self.k,s=dz,degree=2)
            b = Expression('b1*b2*b3',b1=b1,b2=b2,b3=a3,degree=2)
            if i < len(self.times) and self.t*self.dt == self.times[i]: 
                    print('dose')
                    d=self.doses[i]
                    print(d)
                    i+=1
                        
            S_rt = Expression('-a*d - b*d*d', a=a, b=b, d=d,degree=2)
            F = Expression("P - K", degree=2, P=P, K=K)

            a_n = n*v*dx + self.dt*self.Dxn*inner(grad(n)[0],grad(v)[0])*dx  \
                - 2*(1/(dz*dz))*self.dt*self.Dsn*n*v*dx + (1/dz)*self.dt*vs*n*v*dx - self.dt*n*v*F*dx - n*v*S_rt*dx
            L_n = n0_array[1]*v*dx + (1/dz)*self.dt*vs0*nh0*v*dx - 2*(1/(dz*dz))*self.dt*self.Dsn*nh0*v*dx

            nh1.assign(nh0)
            nh0.assign(N) 
            temp2 = Function(self.V)
            temp2.assign(N)
            phi.assign(temp2)
            n0_array[1] = temp2

            # print('s=%g: ' %((s+1)*dz))
            # temp2 = interpolate(temp2,self.V)
            # plot(temp2)
            # plt.ylim([0,0.01])
            # plt.show()

            s=1
        
            while(s < s_steps-2):     
                # P.s = s+1
                # K.s = s+1
                # vs.s = s+1
                # vs0.s = s
                # a3.P = P
                # a2.s = s+1
                # b2.s = s+1
                # a.a2 = a2
                # a.a3 = a3
                # b.b2 = b2
                # S_rt.a = a
                # S_rt.b = b
                # F.P = P
                # F.K = K

                P = Expression('(p_csc*pow(c,4)/(pow(K_csc,4)+pow(c,4))*exp(-pow((s-s_csc)/g_csc,2)) \
                        + p_dc*pow(c,4)/(pow(K_dc,4)+pow(c,4))*exp(-pow((s-s_dc)/g_dc,2)))*(1-phi)',
                        p_csc=self.p_csc,p_dc=self.p_dc,K_csc=self.K_csc,K_dc=self.K_dc,g_csc=self.g_csc,g_dc=self.g_dc,
                        s_csc=self.s_csc,s_dc=self.s_dc,phi=phi,s=(s+1)*dz,c=C,degree=2)
                K = Expression('d_tdc * exp(-((1-s)/g_tdc)) + d_n * (0.5+0.5*tanh(pow(epsilon_k,-1)*(c_N-c)))',
                        d_tdc=self.d_tdc,d_n=self.d_n,g_tdc=self.g_tdc,epsilon_k=self.epsilon_k,c_N=self.c_N,c=C,s=(s+1)*dz,degree=2)
                vs = Expression('V_plus*tanh(s/csi_plus)*tanh((1-s)/csi_plus)*(0.5+0.5*tanh((c-c_H)*pow(epsilon,-1))) \
                            - V_minus*tanh(s/csi_minus)*tanh(pow(1-s,2)/csi_minus)*(0.5+0.5*tanh((c_H-c)*pow(epsilon,-1)))',
                            V_plus=self.V_plus,V_minus=self.V_minus,csi_minus=self.csi_minus,csi_plus=self.csi_plus,c_H=self.c_H,
                            epsilon=self.epsilon,c=C,s=(s+1)*dz,degree=2)
                vs0 = Expression('V_plus*tanh(s/csi_plus)*tanh((1-s)/csi_plus)*(0.5+0.5*tanh((c-c_H)*pow(epsilon,-1))) \
                            - V_minus*tanh(s/csi_minus)*tanh(pow(1-s,2)/csi_minus)*(0.5+0.5*tanh((c_H-c)*pow(epsilon,-1)))',
                            V_plus=self.V_plus,V_minus=self.V_minus,csi_minus=self.csi_minus,csi_plus=self.csi_plus,c_H=self.c_H,
                            epsilon=self.epsilon,c=C,s=s*dz,degree=2)
                # vs = interpolate(vs,self.V)

                a1 = Expression('(c > c_R) ? 1 : 1/OER',c_R=self.c_R,OER=self.OER,c=C,degree=2)
                a2 = Expression('alpha_min + delta_alpha*tanh(k*s)',alpha_min=self.alpha_min,delta_alpha=self.delta_alpha,k=self.k,s=(s+1)*dz,degree=2)
                a3 = Expression('1+P/Pmax', P=P,Pmax=self.p_dc,degree=2)
                a = Expression('a1*a2*a3',a1=a1,a2=a2,a3=a3,degree=2)
                #a = interpolate(a,V)
                b1 = Expression('(c > c_R) ? 1 : 1/(OER*OER)',c_R=self.c_R,OER=self.OER,c=C,degree=2)
                b2 = Expression('beta_min + delta_beta*tanh(k*s)',beta_min=self.beta_min,delta_beta=self.delta_beta,k=self.k,s=(s+1)*dz,degree=2)
                b = Expression('b1*b2*b3',b1=b1,b2=b2,b3=a3,degree=2)            
                S_rt = Expression('-a*d - b*d*d', a=a, b=b, d=d,degree=2)
                F = Expression("P - K", degree=2, P=P, K=K)
                
                # vs = interpolate(vs,self.V)
                # plot(vs)
                # plt.show()

                a_n = n*v*dx + self.dt*self.Dxn*inner(grad(n)[0],grad(v)[0])*dx  \
                    - (1/(dz*dz))*self.dt*self.Dsn*n*v*dx + (1/dz)*self.dt*vs*n*v*dx - self.dt*n*v*F*dx - n*v*S_rt*dx
                L_n = n0_array[s+1]*v*dx + (1/dz)*self.dt*vs0*nh0*v*dx - 2*(1/(dz*dz))*self.dt*self.Dsn*nh0*v*dx + (1/(dz*dz))*self.dt*self.Dsn*nh1*v*dx

                solve(a_n==L_n,N)

                nh1.assign(nh0)
                nh0.assign(N) 
                temp3 = Function(self.V)
                temp3.assign(N)
                phi.assign(temp3 + phi)
                n0_array[s+1] = temp3

                # print('s=%g: ' %((s+1)*dz))
                # temp3 = interpolate(temp3,self.V)
                # print(temp3(0.5))
                # plot(temp3)
                # plt.ylim([0,0.01])
                # plt.show()

                s+=1
            
            # last step
            
            P = Expression('(p_csc*pow(c,4)/(pow(K_csc,4)+pow(c,4))*exp(-pow((s-s_csc)/g_csc,2)) \
                        + p_dc*pow(c,4)/(pow(K_dc,4)+pow(c,4))*exp(-pow((s-s_dc)/g_dc,2)))*(1-phi)',
                        p_csc=self.p_csc,p_dc=self.p_dc,K_csc=self.K_csc,K_dc=self.K_dc,g_csc=self.g_csc,g_dc=self.g_dc,
                        s_csc=self.s_csc,s_dc=self.s_dc,phi=phi,s=(s+1)*dz,c=C,degree=2)
            K = Expression('d_tdc * exp(-((1-s)/g_tdc)) + d_n * (0.5+0.5*tanh(pow(epsilon_k,-1)*(c_N-c)))',
                    d_tdc=self.d_tdc,d_n=self.d_n,g_tdc=self.g_tdc,epsilon_k=self.epsilon_k,c_N=self.c_N,c=C,s=(s+1)*dz,degree=2)
            vs = Expression('V_plus*tanh(s/csi_plus)*tanh((1-s)/csi_plus)*(0.5+0.5*tanh((c-c_H)*pow(epsilon,-1))) \
                        - V_minus*tanh(s/csi_minus)*tanh(pow(1-s,2)/csi_minus)*(0.5+0.5*tanh((c_H-c)*pow(epsilon,-1)))',
                        V_plus=self.V_plus,V_minus=self.V_minus,csi_minus=self.csi_minus,csi_plus=self.csi_plus,c_H=self.c_H,
                        epsilon=self.epsilon,c=C,s=(s+1)*dz,degree=2)
            vs0 = Expression('V_plus*tanh(s/csi_plus)*tanh((1-s)/csi_plus)*(0.5+0.5*tanh((c-c_H)*pow(epsilon,-1))) \
                        - V_minus*tanh(s/csi_minus)*tanh(pow(1-s,2)/csi_minus)*(0.5+0.5*tanh((c_H-c)*pow(epsilon,-1)))',
                        V_plus=self.V_plus,V_minus=self.V_minus,csi_minus=self.csi_minus,csi_plus=self.csi_plus,c_H=self.c_H,
                        epsilon=self.epsilon,c=C,s=s*dz,degree=2)
            # vs = interpolate(vs,self.V)

            a1 = Expression('(c > c_R) ? 1 : 1/OER',c_R=self.c_R,OER=self.OER,c=C,degree=2)
            a2 = Expression('alpha_min + delta_alpha*tanh(k*s)',alpha_min=self.alpha_min,delta_alpha=self.delta_alpha,k=self.k,s=(s+1)*dz,degree=2)
            a3 = Expression('1+P/Pmax', P=P,Pmax=self.p_dc,degree=2)
            a = Expression('a1*a2*a3',a1=a1,a2=a2,a3=a3,degree=2)
            #a = interpolate(a,V)
            b1 = Expression('(c > c_R) ? 1 : 1/(OER*OER)',c_R=self.c_R,OER=self.OER,c=C,degree=2)
            b2 = Expression('beta_min + delta_beta*tanh(k*s)',beta_min=self.beta_min,delta_beta=self.delta_beta,k=self.k,s=(s+1)*dz,degree=2)
            b = Expression('b1*b2*b3',b1=b1,b2=b2,b3=a3,degree=2)            
            S_rt = Expression('-a*d - b*d*d', a=a, b=b, d=d,degree=2)
            F = Expression("P - K", degree=2, P=P, K=K)
            
            
            a_n = n*v*dx + self.dt*self.Dxn*inner(grad(n)[0],grad(v)[0])*dx  \
                    + 2*(1/(dz*dz))*self.dt*self.Dsn*n*v*dx + self.dt*vs*n*v*dx - self.dt*n*v*F*dx - n*v*S_rt*dx
            L_n = n0_array[s+1]*v*dx + (1/dz)*self.dt*vs0*nh0*v*dx + 2*(1/(dz*dz))*self.dt*self.Dsn*nh1*v*dx
            solve(a_n==L_n,N)

            nh1.assign(nh0)
            nh0.assign(N) 
            temp3 = Function(self.V)
            temp3.assign(N)
            phi.assign(temp3 + phi)
            n0_array[s+1] = temp3
            phi.assign(phi/s_steps)

            # print('s=%g: ' %((s+1)*dz))
            # temp3 = interpolate(temp3,self.V)
            # plot(temp3)
            # plt.ylim([0,0.01])
            # plt.show()

            # phi = interpolate(phi,self.V)
            # plot(phi)
            # # plt.ylim([0,0.01])
            # plt.show()


            d=0
            t+=1
        
        np.save(self.path_sol + "/" + "mass.npy", mass)

        return mass