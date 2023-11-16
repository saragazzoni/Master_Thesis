from dolfin import *

from numpy.polynomial.legendre import leggauss

# (s, t) grid
# NOTE: (0, 10) x (-1, 1) just to make Legendra quadrature work
mesh = RectangleMesh(Point(0, -1), Point(10, 1), 16, 64)
plot(mesh)

V = FunctionSpace(mesh, 'CG', 1)
n = interpolate(Expression('sin(pi*(x[0] - x[1]))', degree=1), V)


class VerticalAverage(UserExpression):
    def __init__(self, f, quad_degree, **kwargs):
        super().__init__(**kwargs)
        self.f = f
        self.points, self.weights = leggauss(quad_degree)
        assert f.ufl_shape == ()
        
    def eval(self, values, x):
        values[0] = sum(wq*self.f(x[0], xq) for xq, wq in zip(self.points, self.weights))

    def value_shape(self):
        return ()

vn = VerticalAverage(n, quad_degree=20, degree=1)
vn_h = interpolate(vn, V)

File('n.pvd') << n
File('vn_h.pvd') << vn_h