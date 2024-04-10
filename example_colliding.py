from ngsolve import Parameter, CF, x, y, IfPos, sqrt
from netgen.geom2d import SplineGeometry

# Convergence study parameters
t_end = 1.5
h0, dt0 = 0.07, t_end / 80

# Mesh
background_domain = SplineGeometry()
background_domain.AddRectangle([-0.6, -1.35], [0.6, 1.35], bc=1)
ngmesh = background_domain.GenerateMesh(maxh=h0, quad_dominated=False)

# Example Data
nu = 1e-1

t = Parameter(0.0)
w = IfPos(y, -1, 1) * IfPos(t - t_end / 2, 1, -1) * CF((0, -1))
div_w = 0
velmax = 1

u0 = IfPos(y, 1, -1)
u_ex = CF(0)
rhs = CF(0)


# Level set
def levelset_func(t):
    l1 = sqrt(x**2 + (y - t + 3 / 4)**2)
    l2 = sqrt(x**2 + (y + t - 3 / 4)**2)
    return IfPos(l1 - l2, l2, l1) - 0.5
