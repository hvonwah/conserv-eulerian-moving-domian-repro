# ------------------------------ LOAD LIBRARIES -------------------------------
from netgen.geom2d import SplineGeometry
from ngsolve import *
from eulerian_moving_domain import *
from math import log2
import pickle
import argparse

ngsglobals.msg_level = 2
SetNumThreads(4)

# --------------------------- COMMAND-LINE OPTIONS ---------------------------- 
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-Lx', type=int, default=6, help='Number of mesh refinements')
parser.add_argument('-Lt', type=int, default=7, help='Number of time step refinements')
parser.add_argument('-bdf', type=int, choices=[1, 2], default=1, help='Use BDF1 or BDF2 time-stepping')
parser.add_argument('-ex', '--example', type=str, choices=['travel_cir', 'kite', 'collide', 'collide3d'], default='kite', help='Chose example to run')
parser.add_argument('-vtk', type=int, choices=[0, 1], default='0', help='Export VTK in every time step')
options = vars(parser.parse_args())
print(options)


# --------------------------------- OPTIONS -----------------------------------
k = 1                                 # Finite element space order
Lx = options['Lx']                    # Number of mesh refinements
Lt = options['Lt']                    # Number of time step refinements

filename_errs = 'results/moving_domian_convection_diffusion'
filename_errs += f'_{options["example"]}_bdf{options["bdf"]}'
vtk_flag = bool(options['vtk'])
basename_vtk = f'vtk/{options["example"]}_'

if options['bdf'] == 1:
    solver = solve_moving_domain_convection_difusion_bdf1
elif options['bdf'] == 2:
    solver = solve_moving_domain_convection_difusion_bdf2
else:
    raise ImportError(f"Ups... I don't know the BDF{options['bdf']} formula")

if options['example'] == 'travel_cir':
    from example_translation_circle import *
elif options['example'] == 'kite':
    from example_kite import *
elif options['example'] == 'collide':
    from example_colliding import *
elif options['example'] == 'collide3d':
    from example_colliding3d import *
else:
    raise ImportError("Ups... I don't know what example to load")

filename_errs += f'h{h0}dt{dt0}nu{nu}.dat'

# ----------------------------------- MAIN ------------------------------------
try:
    err = pickle.load(open(filename_errs, "rb"))
    print("loaded the following error dictionary:\n", err)
except OSError:
    err = {}

for lx in range(Lx):
    if lx > 0:
        ngmesh.Refine()

    mesh = Mesh(ngmesh)
    h_max = h0 / 2**lx

    for lt in range(Lt):
        if (lx, lt) in err.keys():
            continue

        print(f" lx = {lx}/{Lx - 1}, lt = {lt}/{Lt - 1}")
        vtk_name = f'{basename_vtk}lx{lx}lt{lt}'
        if vtk_flag is False:
            vtk_name = None

        dt = dt0 / 2**lt
        t.Set(0.0)

        with TaskManager():
            l2, h1, mass = solver(
                mesh=mesh, k=k, h_max=h_max, dt=dt, t_end=t_end, t=t,
                lset=levelset_func, w=w, velmax=velmax, nu=nu, rhs=rhs, u0=u0,
                u_ex=u_ex, c_gamma=1, vtk_name=vtk_name)

        err[(lx, lt)] = {"l2": l2, "h1": h1, "mass": mass}

        with open(filename_errs, "wb") as fid:
            pickle.dump(err, fid)


# ------------------------------ POST PROCESSING ------------------------------
for lx in range(Lx):
    for lt in range(Lt):
        dt = dt0 / 2**lt
        err[(lx, lt)]["l2l2"] = sqrt(dt * sum([e**2 for e in err[(lx, lt)]["l2"]]))
        err[(lx, lt)]["l2h1"] = sqrt(dt * sum([e**2 for e in err[(lx, lt)]["h1"]]))

for _n in ["l2l2", 'l2h1']:
    nxx = (Lx - 1) // 2
    ii = [(Lx - 1 - 2 * i, Lt - 1 - i) for i in range(nxx)]
    print(f"{_n}:\nLt\\Lx   | ", end='')
    [print(f'{lx}\t\t\t', end='') for lx in range(Lx)]
    print("| eoc_t eoc_txx\n", end='')
    print('----------' + '------------' * Lx + '-------------')
    for lt in range(Lt):
        print(f'{lt}\t\t| ', end='')
        for lx in range(Lx):
            print(f'{err[(lx, lt)][_n]:4.2e}\t', end='')
        if lt == 0:
            print('| ----', end='')
        else:
            rate = log2(err[(lx, lt - 1)][_n]) - log2(err[(lx, lt)][_n])
            print(f"| {rate:4.2f}", end='')
        if lt < Lt - nxx:
            print('  ----')
        else:
            lx, lt = ii[Lt - lt - 1]
            rate = log2(err[(lx - 2, lt - 1)][_n]) - log2(err[(lx, lt)][_n])
            print(f"  {rate:4.2f}")

    print('----------' + '------------' * Lx + '-------------')
    print('eoc_x\t| ----\t\t', end='')
    for lx in range(1, Lx):
        lt = Lt - 1
        rate = log2(err[(lx - 1, lt)][_n]) - log2(err[(lx, lt)][_n])
        print(f"{rate:4.2f}\t\t", end='')
    it, ix = max(0, Lx - Lt), max(0, Lt - Lx)
    print('\neoc_xt\t| ' + '----\t\t' * (it + 1), end='')
    for lx in range(1 + max(ix, it), max(Lx, Lt)):
        rate = log2(err[(lx - 1 - ix, lx - 1 - it)][_n]) - log2(err[(lx - ix, lx - it)][_n])
        print(f"{rate:4.2f}\t\t", end='')
    ntt = (Lt - 1) // 2
    ii = [(Lx - 1 - i, Lt - 1 - 2 * i) for i in range(ntt)]
    print('\neoc_xtt\t| ' + '----\t\t' * (Lx - len(ii)), end='')
    for lx, lt in ii:
        rate = log2(err[(lx - 1, lt - 2)][_n]) - log2(err[(lx, lt)][_n])
        print(f"{rate:4.2f}\t\t", end="")
    print('')
