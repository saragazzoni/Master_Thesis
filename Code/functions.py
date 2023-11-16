from dolfin import *
from numpy.polynomial.legendre import leggauss
import numpy as np
from ufl import tanh

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
    
def boundaries(mesh):
    
    boundary_markers = MeshFunction('size_t', mesh,1)

    class BoundaryX0(SubDomain):
        tol = 1E-14
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 0, 1E-14)

    bx0 = BoundaryX0()
    bx0.mark(boundary_markers, 3)

    class BoundaryX1(SubDomain):
        tol = 1E-14
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 1, 1E-14)

    bx1 = BoundaryX1()
    bx1.mark(boundary_markers, 1)

    class BoundaryY1(SubDomain):
        tol = 1E-14
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 1, 1E-14)

    by1 = BoundaryY1()
    by1.mark(boundary_markers, 2)

    class BoundaryY0(SubDomain):
        tol = 1E-14
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 0, 1E-14)

    by0 = BoundaryY0()
    by0.mark(boundary_markers, 0)

    ds = Measure("ds",domain=mesh, subdomain_data=boundary_markers)

    return bx1,bx0,by0,by1,ds
        

def solver(mesh,m0,dt,T,save_interval):

    bx1,bx0,by0,by1,ds = boundaries(mesh)

    # parameters setting 
    s0 = 0.5
    sigma = sqrt(0.02)
    Dxc = 4.32*1e3
    gamma = 4*Dxc
    K_m = 0.0125
    Dxn = 2.4*1e-3
    Dsn = 1.2*1e-4
    p_csc = 0.12
    p_dc = 0.48
    K_csc = 0.05
    K_dc = 0.3 
    g_csc = 0.1
    g_dc = 0.2
    g_tdc = 0.1
    s_csc = 0
    s_dc = 0.55
    epsilon = 0.05
    epsilon_k = 0.01
    V_plus = 1.92*1e-2
    V_minus = 0.48*1e-2
    csi_plus = 0.1
    csi_minus = 0.1
    c_H = 0.3
    c_N = 0.0125
    c_R = 0.1
    d_tdc = 0.024
    d_n = 2.4
    n_steps = int(T/dt)

    V = FunctionSpace(mesh,"P",2)
    N = Function(V)
    C = Function(V)
    n = TrialFunction(V)
    v = TestFunction(V)
    c = TrialFunction(V)
    w = TestFunction(V)

    # initial cell distribution
    n0 = Expression("m0/(pow(2*pi,0.5)*sigma)*exp(-pow(x[1]-s0,2)/(2*sigma*sigma))",m0 = m0,s0 = s0,sigma=sigma,degree=2)
    n0 = interpolate(n0,V)

    c_k = interpolate(Constant(1.0), V)
    f = Constant(0.0)
    L_c = f*w*dx
    bc_c = DirichletBC(V,Constant(1.0),bx1)
    bcs_c = [bc_c]

    mass = []
    n_vect = [n0.vector().get_local().copy()]
    c_vect = [c_k.vector().get_local().copy()]
    phi_vect = []


    t=1
    while(t<n_steps):
        print('time=%g: ' % t)

        # update phi
        phi = VerticalAverage(n0, quad_degree=20, degree=2)
        phi_h = interpolate(phi, V)
        a_c = Dxc * inner(grad(c)[0],grad(w)[0])*dx + gamma/(c_k + K_m)*phi_h*c*w*dx
        
        # fixed point -> solve c
        eps= 1.0
        tol = 1.0E-3    
        iter = 0        
        maxiter = 30    
        while eps > tol and iter < maxiter:
            iter += 1
            solve(a_c==L_c,C,bcs_c)
            difference = C.vector() - c_k.vector()
            eps = np.linalg.norm(difference) / np.linalg.norm(c_k.vector())
            print('iter=%d: norm=%g' % (iter, eps))
            c_k.assign(C) 
    
        # solve n
        P = Expression('(p_csc*pow(c,4)/(pow(K_csc,4)+pow(c,4))*exp(-pow((x[1]-s_csc)/g_csc,2)) \
                + p_dc*pow(c,4)/(pow(K_dc,4)+pow(c,4))*exp(-pow((x[1]-s_dc)/g_dc,2)))*(1-phi)',
                p_csc=p_csc,p_dc=p_dc,K_csc=K_csc,K_dc=K_dc,g_csc=g_csc,g_dc=g_dc,s_csc=s_csc,s_dc=s_dc,phi=phi_h,c=C,degree=2)
        K = Expression('d_tdc * exp(-((1-x[1])/g_tdc)) + d_n * (0.5+0.5*tanh(pow(epsilon_k,-1)*(c_N-c)))',
                d_tdc=d_tdc,d_n=d_n,g_tdc=g_tdc,epsilon_k=epsilon_k,c_N=c_N,c=C,degree=2)
        F = Expression("P - K", degree=2, P=P, K=K)
        vs = Expression('V_plus*tanh(x[1]/csi_plus)*tanh((1-x[1])/csi_plus)*(0.5+0.5*tanh((c-c_H)*pow(epsilon,-1))) \
                    - V_minus*tanh(x[1]/csi_minus)*tanh(pow(1-x[1],2)/csi_minus)*(0.5+0.5*tanh((c_H-c)*pow(epsilon,-1)))',
                    V_plus=V_plus,V_minus=V_minus,csi_minus=csi_minus,csi_plus=csi_plus,c_H=c_H,epsilon=epsilon,c=C,degree=2)
        vs = interpolate(vs,V)
        a_n = n*v*dx + dt*Dxn*inner(grad(n)[0],grad(v)[0])*dx + dt*Dsn*inner(grad(n)[1],grad(v)[1])*dx + dt*grad(vs*n)[1]*v*dx \
            - dt*n*v*F*dx + dt*n*v*vs*ds(0) - dt*n*v*vs*ds(2)  #+ dt*vs.dx(1)*n*v*dx \
        L_n = n0*v*dx

        solve(a_n==L_n,N)
        n0.assign(N)

        if t % save_interval == 0:
            n_vect.append(N.vector().get_local().copy())
            c_vect.append(C.vector().get_local().copy())
            #phi_vect.append(phi_h)
        mass.append(assemble(phi_h*dx))

        t+=1

    return n_vect,c_vect,mass
    
def plot(vect,dt,save_interval):

    for i,x in enumerate(vect):
        sol = plot(x)
        plt.colorbar(sol)
        plt.title('Time t=%f' %(i*save_interval*dt))
        plt.xlabel('x')
        plt.ylabel('s')
        plt.show()