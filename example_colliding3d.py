from ngsolve import Parameter, CF, x, y, z, IfPos, sqrt
from netgen.occ import OCCGeometry, Box, Pnt

# Convergence study parameters
t_end = 1.5
h0, dt0 = 0.07, t_end / 80

# Mesh
ngmesh = OCCGeometry(Box(Pnt(-0.6, -0.6, -1.35), Pnt(0.6, 0.6, 1.35))).GenerateMesh(maxh=h0)

# Example Data
nu = 1e-1

t = Parameter(0.0)
w = IfPos(z, -1, 1) * IfPos(t - t_end / 2, 1, -1) * CF((0, 0, -1))
div_w = 0
velmax = 1

u0 = IfPos(z, 1, -1)
u_ex = CF(0)
rhs = CF(0)


# Level set
def levelset_func(t):
    l1 = sqrt(x**2 + y**2 + (z - t + 3 / 4)**2)
    l2 = sqrt(x**2 + y**2 + (z + t - 3 / 4)**2)
    return IfPos(l1 - l2, l2, l1) - 0.5
