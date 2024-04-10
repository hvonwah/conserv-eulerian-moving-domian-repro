from ngsolve import Parameter, CF, x, y, sqrt, sin, cos, pi
from netgen.geom2d import SplineGeometry

# Convergence study parameters
t_end = 1
h0, dt0 = 0.4, t_end / 2

# Mesh
background_domain = SplineGeometry()
background_domain.AddRectangle([-1.5, -1.5], [2.5, 1.5], bc=1)
ngmesh = background_domain.GenerateMesh(maxh=h0, quad_dominated=False)

# Convection field, initial condition and right-hand side
nu = 2e-1

t = Parameter(0.0)
rho = CF((1 - y**2) * t)
_r = sqrt((x - rho)**2 + y**2)
_r0 = 1
w = CF((rho.Diff(t), 0))
div_w = 0
velmax = 1

u0 = CF(0)
u_ex = CF(cos(pi * _r / _r0) * sin(pi / 2 * t))

rhs = (u_ex.Diff(t)
       - nu * (u_ex.Diff(x).Diff(x) + u_ex.Diff(y).Diff(y))
       + w[0] * u_ex.Diff(x) + w[1] * u_ex.Diff(y))


# Level set
def levelset_func(t):
    rho_t = CF((1 - y**2) * t)
    return sqrt((x - rho_t)**2 + y**2) - _r0
