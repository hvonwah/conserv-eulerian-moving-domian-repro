from xfem import *
from ngsolve import *
from xfem.lsetcurv import *

__all__ = ['solve_moving_domain_convection_difusion_bdf1',
           'solve_moving_domain_convection_difusion_bdf2',
           ]


def solve_moving_domain_convection_difusion_bdf1(mesh, k, h_max, dt, t_end, t, lset, w, velmax, nu, rhs, u0, u_ex, c_gamma=1, inverse="pardiso", vtk_name=None):
    # FE Space
    V = H1(mesh, order=k, dgjumps=True)
    gfu = GridFunction(V)

    # Higher order discrete level set approximation
    V1 = H1(mesh, order=1)
    lsetp1, lsetp1_last = GridFunction(V1), GridFunction(V1)

    # Cut-Info classes for element marking
    ci_main = CutInfo(mesh)
    ci_inner = CutInfo(mesh)
    ci_outer = CutInfo(mesh)

    # Element and facet markers
    els_hasneg = ci_main.GetElementsOfType(HASNEG)
    # els_if = ci_main.GetElementsOfType(IF)
    els_out = ci_outer.GetElementsOfType(HASNEG)
    els_in = ci_inner.GetElementsOfType(NEG)

    els_ring = BitArray(mesh.ne)
    facets_ring = BitArray(mesh.nedge)
    els_hasneg_last, els_test = BitArray(mesh.ne), BitArray(mesh.ne)

    delta = 1 * dt * velmax
    K_tilde = int(ceil(delta / h_max))
    gamma_s = c_gamma * K_tilde

    dx = dCut(levelset=lsetp1, domain_type=NEG, definedonelements=els_hasneg)
    # ds = dCut(levelset=lsetp1, domain_type=IF, definedonelements=els_if)
    dx_last = dCut(levelset=lsetp1_last, domain_type=NEG,
                   definedonelements=els_hasneg_last)
    dw = dFacetPatch(definedonelements=facets_ring)

    u, v = V.TnT()
    h = specialcf.mesh_size
    # n_lset = Normalize(Grad(lsetp1))

    # Bilinear and linear forms
    a_form = (1 / dt) * u * v
    a_form += nu * InnerProduct(Grad(u), Grad(v))
    a_form += - u * InnerProduct(w, Grad(v))
    # a_form_bnd = 10 * InnerProduct(n_lset, Grad(u)) * InnerProduct(n_lset, Grad(v))
    a_gp = gamma_s * (1 / h**2) * (u - u.Other()) * (v - v.Other())

    a = RestrictedBilinearForm(V, element_restriction=els_out,
                               facet_restriction=facets_ring,
                               check_unused=False)
    a += a_form.Compile() * dx
    # a += a_form_bnd.Compile() * ds
    a += a_gp.Compile() * dw

    f_form = rhs * v
    f_form_last = (1 / dt) * gfu * v

    f = LinearForm(V)
    f += f_form.Compile() * dx
    f += f_form_last.Compile() * dx_last

    # Error computation
    errors_L2, errors_H1 = [], []
    dx_2k = dx.order(2 * k)
    u_ex = u_ex.Compile()
    if mesh.dim == 2:
        grad_u_ex = CF((u_ex.Diff(x), u_ex.Diff(y))).Compile()
    else:
        grad_u_ex = CF((u_ex.Diff(x), u_ex.Diff(y), u_ex.Diff(z))).Compile()

    def comp_errs():
        l2 = sqrt(Integrate((gfu - u_ex)**2 * dx_2k, mesh))
        h1 = sqrt(Integrate((Grad(gfu) - grad_u_ex)**2 * dx_2k, mesh))
        errors_L2.append(l2), errors_H1.append(h1)
        return l2

    # Track mass conservation
    mass, errors_mass = [], []

    def comp_mass_change():
        m = Integrate(gfu * dx, mesh)
        f = Integrate(rhs * dx, mesh)
        errors_mass.append(m - mass[0] - dt * f)
        mass.append(m)
        return None

    # Time stepping loop
    t.Set(0.0)
    InterpolateToP1(lset(t) - delta, lsetp1)
    ci_outer.Update(lsetp1)

    gfu.Set(u0)
    InterpolateToP1(lset(t), lsetp1)
    ci_main.Update(lsetp1)
    mass.append(Integrate(gfu * dx, mesh))

    # Visualisation
    Draw(IfPos(lsetp1, float('NaN'), gfu), mesh, "gfu")
    Draw(IfPos(lsetp1, float('NaN'), u_ex), mesh, "u_ex")
    Draw(IfPos(lsetp1, float('NaN'), gfu - u_ex), mesh, "l2err")

    vtk_flag = False
    if vtk_name is not None and isinstance(vtk_name, str):
        cf_els_out = BitArrayCF(els_out)
        vtk = VTKOutput(ma=mesh, coefs=[gfu, lsetp1, cf_els_out],
                        names=['gfu', 'lset', 'els_out'], filename=vtk_name,
                        floatsize='single', order=1)
        vtk_flag = True
        vtk.Do(time=0.0)

    for it in range(1, int(t_end / dt + 0.5) + 1):
        lsetp1_last.vec.data = lsetp1.vec
        els_hasneg_last[:] = els_hasneg
        t.Set(it * dt)

        # Update Element markers: Extension facets.
        InterpolateToP1(lset(t) - delta, lsetp1)
        ci_outer.Update(lsetp1)
        InterpolateToP1(lset(t) + delta, lsetp1)
        ci_inner.Update(lsetp1)

        els_ring[:] = els_out & ~els_in
        facets_ring[:] = GetFacetsWithNeighborTypes(mesh, a=els_out, b=els_ring)
        active_dofs = GetDofsOfElements(V, els_out)

        # Update Element markers: Domain elements
        InterpolateToP1(lset(t), lsetp1)
        ci_main.Update(lsetp1)

        # Check element extension for mass conservation
        els_test[:] = ~els_out & els_hasneg_last
        assert sum(els_test) == 0, 'Some conservation elements are not active'

        # Update Linear System
        a.Assemble(reallocate=True)
        f.Assemble()
        inv = a.mat.Inverse(active_dofs, inverse=inverse)
        # Solve Problem
        gfu.vec.data = inv * f.vec

        # Compute Errors
        l2err = comp_errs()
        comp_mass_change()

        # Output
        str_out = f"  dt={dt:8.6f}, t={t.Get():8.6f}, err(L2)={l2err:4.2e}, "
        str_out += f"err(mass)={errors_mass[-1]: 4.2e}, "
        str_out += f"active_els={sum(els_out)}, K = {K_tilde}"
        print(str_out)
        Redraw(blocking=True)
        if vtk_flag:
            vtk.Do(time=t.Get())

    return errors_L2, errors_H1, errors_mass


def solve_moving_domain_convection_difusion_bdf2(mesh, k, h_max, dt, t_end, t, lset, w, velmax, nu, rhs, u0, u_ex, c_gamma=1, inverse="pardiso", vtk_name=None):
    # FE Space
    V = H1(mesh, order=k, dgjumps=True)
    gfu, gfu_last = GridFunction(V), GridFunction(V)

    # Higher order discrete level set approximation
    V1 = H1(mesh, order=1)
    lsetp1 = GridFunction(V1)
    lsetp1_last, lsetp1_last2 = GridFunction(V1), GridFunction(V1)

    # Cut-Info classes for element marking
    ci_main = CutInfo(mesh)
    ci_inner = CutInfo(mesh)
    ci_outer = CutInfo(mesh)

    # Element and facet markers
    els_hasneg = ci_main.GetElementsOfType(HASNEG)
    # els_if = ci_main.GetElementsOfType(IF)
    els_out = ci_outer.GetElementsOfType(HASNEG)
    els_in = ci_inner.GetElementsOfType(NEG)

    els_ring = BitArray(mesh.ne)
    facets_ring = BitArray(mesh.nedge)
    els_hasneg_last, els_hasneg_last2 = BitArray(mesh.ne), BitArray(mesh.ne)
    els_test = BitArray(mesh.ne)

    delta = 2 * dt * velmax
    K_tilde = int(ceil(delta / h_max))
    gamma_s = c_gamma * K_tilde

    dx = dCut(levelset=lsetp1, domain_type=NEG, definedonelements=els_hasneg)
    # ds = dCut(levelset=lsetp1, domain_type=IF, definedonelements=els_if)
    dx_last = dCut(levelset=lsetp1_last, domain_type=NEG,
                   definedonelements=els_hasneg_last)
    dx_last2 = dCut(levelset=lsetp1_last2, domain_type=NEG,
                    definedonelements=els_hasneg_last2)
    dw = dFacetPatch(definedonelements=facets_ring)

    u, v = V.TnT()
    h = specialcf.mesh_size
    # n_lset = Normalize(Grad(lsetp1))

    # Bilinear and linear forms
    a_form = nu * InnerProduct(Grad(u), Grad(v)) - u * InnerProduct(w, Grad(v))
    # a_form_bnd = 10 * InnerProduct(n_lset, Grad(u)) * InnerProduct(n_lset, Grad(v))
    a_gp = gamma_s * (1 / h**2) * (u - u.Other()) * (v - v.Other())

    a1 = RestrictedBilinearForm(V, element_restriction=els_out,
                                facet_restriction=facets_ring,
                                check_unused=False)
    a1 += ((1 / dt) * u * v + a_form).Compile() * dx
    # a1 += a_form_bnd.Compile() * ds
    a1 += a_gp.Compile() * dw

    a2 = RestrictedBilinearForm(V, element_restriction=els_out,
                                facet_restriction=facets_ring,
                                check_unused=False)
    a2 += ((3 / 2 / dt) * u * v + a_form).Compile() * dx
    # a2 += a_form_bnd.Compile() * ds
    a2 += a_gp.Compile() * dw

    f_form = rhs * v
    f1_form_last = (1 / dt) * gfu * v
    f2_form_last = (2 / dt) * gfu * v
    f2_form_last2 = -(1 / 2 / dt) * gfu_last * v

    f1 = LinearForm(V)
    f1 += f_form.Compile() * dx
    f1 += f1_form_last.Compile() * dx_last

    f2 = LinearForm(V)
    f2 += f_form.Compile() * dx
    f2 += f2_form_last.Compile() * dx_last
    f2 += f2_form_last2.Compile() * dx_last2

    # Error computation
    errors_L2, errors_H1 = [], []
    dx_2k = dx.order(2 * k)
    u_ex = u_ex.Compile()
    if mesh.dim == 2:
        grad_u_ex = CF((u_ex.Diff(x), u_ex.Diff(y))).Compile()
    else:
        grad_u_ex = CF((u_ex.Diff(x), u_ex.Diff(y), u_ex.Diff(z))).Compile()

    def comp_errs():
        l2 = sqrt(Integrate((gfu - u_ex)**2 * dx_2k, mesh))
        h1 = sqrt(Integrate((Grad(gfu) - grad_u_ex)**2 * dx_2k, mesh))
        errors_L2.append(l2), errors_H1.append(h1)
        return l2

    # Track mass conservation
    mass, errors_mass = [], []

    def comp_mass_change(it):
        m = Integrate(gfu * dx, mesh)
        f = Integrate(rhs * dx, mesh)
        if it == 1:
            errors_mass.append(m - mass[0] - dt * f)
        else:
            errors_mass.append(1.5 * m - 2 * mass[-1] + 0.5 * mass[-2] - dt * f)
        mass.append(m)
        return None

    # Visualisation
    Draw(IfPos(lsetp1, float('NaN'), gfu), mesh, "gfu")
    Draw(IfPos(lsetp1, float('NaN'), u_ex), mesh, "u_ex")
    Draw(IfPos(lsetp1, float('NaN'), gfu - u_ex), mesh, "l2err")

    vtk_flag = False
    if vtk_name is not None and isinstance(vtk_name, str):
        cf_els_out = BitArrayCF(els_out)
        vtk = VTKOutput(ma=mesh, coefs=[gfu, lsetp1, cf_els_out],
                        names=['gfu', 'lset', 'els_out'], filename=vtk_name,
                        floatsize='single', order=1)
        vtk_flag = True

    # Time-stepping
    def do_step(a, f):
        # Update Element markers: Extension facets.
        InterpolateToP1(lset(t) - delta, lsetp1)
        ci_outer.Update(lsetp1)
        InterpolateToP1(lset(t) + delta, lsetp1)
        ci_inner.Update(lsetp1)

        els_ring[:] = els_out & ~els_in
        facets_ring[:] = GetFacetsWithNeighborTypes(mesh, a=els_out, b=els_ring)
        active_dofs = GetDofsOfElements(V, els_out)

        # Update Element markers: Domain elements
        InterpolateToP1(lset(t), lsetp1)
        ci_main.Update(lsetp1)

        # Check element extension for mass conservation
        els_test[:] = ~els_out & els_hasneg_last & els_hasneg_last2
        assert sum(els_test) == 0, 'Some conservation elements are not active'

        # Update Linear System
        a.Assemble(reallocate=True)
        f.Assemble()
        inv = a.mat.Inverse(active_dofs, inverse=inverse)
        # Solve Problem
        gfu_last.vec.data = gfu.vec
        gfu.vec.data = inv * f.vec

    # Initial condition
    t.Set(0.0)
    InterpolateToP1(lset(t) - delta, lsetp1)
    ci_outer.Update(lsetp1)

    gfu.Set(u0)
    els_hasneg_last.Set()
    InterpolateToP1(lset(t), lsetp1)
    ci_main.Update(lsetp1)
    mass.append(Integrate(gfu * dx, mesh))

    if vtk_flag:
        vtk.Do(time=0.0)

    # Main loop
    for it in range(1, int(t_end / dt + 0.5) + 1):
        lsetp1_last2.vec.data = lsetp1_last.vec
        lsetp1_last.vec.data = lsetp1.vec
        els_hasneg_last2[:] = els_hasneg_last
        els_hasneg_last[:] = els_hasneg

        t.Set(it * dt)
        if it == 1:
            do_step(a1, f1)
        else:
            do_step(a2, f2)

        # Compute Errors
        l2err = comp_errs()
        comp_mass_change(it)

        # Output
        str_out = f"  dt={dt:8.6f}, t={t.Get():8.6f}, err(L2)={l2err:4.2e}, "
        str_out += f"err(mass)={errors_mass[-1]: 4.2e}, "
        str_out += f"active_els={sum(els_out)}, K = {K_tilde}"
        print(str_out)

        Redraw(blocking=True)
        if vtk_flag:
            vtk.Do(time=t.Get())

    return errors_L2, errors_H1, errors_mass
