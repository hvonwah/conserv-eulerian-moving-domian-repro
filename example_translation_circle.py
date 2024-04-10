from ngsolve import Parameter, CF, x, y, sqrt, sin, cos, pi
from netgen.geom2d import SplineGeometry

# Convergence study parameters
t_end = 0.2
h0, dt0 = 0.4, t_end / 2

# Mesh
background_domain = SplineGeometry()
background_domain.AddRectangle([-0.7, -0.7], [0.9, 0.7], bc=1)
ngmesh = background_domain.GenerateMesh(maxh=h0, quad_dominated=False)

# Example Data
nu = 1e-0

t = Parameter(0.0)
rho = CF(1 / pi * sin(2 * pi * t))
_r0 = 0.5
_r1 = pi / (2 * _r0)
w = CF((rho.Diff(t), 0))
div_w = 0
velmax = 2

u0 = CF(cos(_r1 * sqrt(x**2 + y**2))**2)
u_ex = CF(cos(_r1 * sqrt((x - rho)**2 + y**2))**2)
rhs = (u_ex.Diff(t)
       - nu * (u_ex.Diff(x).Diff(x) + u_ex.Diff(y).Diff(y))
       + w[0] * u_ex.Diff(x) + w[1] * u_ex.Diff(y))


# Level set
def levelset_func(t):
    rho_t = CF(1 / pi * sin(2 * pi * t))
    return sqrt((x - rho_t)**2 + y**2) - _r0
