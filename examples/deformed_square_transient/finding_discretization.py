from netgen.geom2d import unit_square
from ngsolve import *
import numpy as np

# Deformed transient Darcy setup
Sstor = 1.0
Kinv = 1.0
p_init = 0.0

# Match rom_deformed_transient.py defaults
T_default = 1.0
source_amp = 0.0
source_sigma = 0.12
source_center = (0.7, 0.4)


def amplitude(t, T=T_default):
    return 1.0 - np.exp(-4.0 * float(t) / float(T))


def source_base():
    cx, cy = source_center
    s = source_sigma
    return float(source_amp) * exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s ** 2))


def source_cf_time(t, T=T_default):
    return amplitude(t, T=T) * source_base()


def p_boundary_time(t, T=T_default):
    return amplitude(t, T=T) * (x * x - y * y)


def apply_top_right_deformation(mesh, order, mu):
    mu1, mu2 = float(mu[0]), float(mu[1])
    deform = GridFunction(VectorH1(mesh, order=int(order)))
    deform.Set(CoefficientFunction((x * y * mu1, x * y * mu2)))
    mesh.SetDeformation(deform)
    return deform


def build_solver(mesh, order, dt, inverse='umfpack'):
    V = HDiv(mesh, order=int(order))
    Q = L2(mesh, order=int(order) - 1)
    Y = FESpace([V, Q])

    (u, p) = Y.TrialFunction()
    (v, q) = Y.TestFunction()

    gfu = GridFunction(Y)
    uh, ph = gfu.components

    p_old = GridFunction(Q)
    p_old.Set(CoefficientFunction(float(p_init)))

    a = BilinearForm(Y, symmetric=False)
    a += (
        Kinv * InnerProduct(u, v)
        - p * div(v)
        + q * div(u)
        + (float(Sstor) / float(dt)) * p * q
    ) * dx
    a.Assemble()

    invA = a.mat.Inverse(Y.FreeDofs(), inverse=inverse)
    return V, Q, Y, invA, gfu, uh, ph, p_old


def solve_history(mesh, order, dt, T, mu, inverse='umfpack'):
    ratio = T / dt
    nsteps = int(np.rint(ratio))
    tol = 1e-10 * max(1.0, abs(ratio))
    if abs(ratio - nsteps) > tol:
        raise ValueError(f'T/dt must be integer (within floating tolerance); got T={T}, dt={dt}, ratio={ratio}')

    apply_top_right_deformation(mesh, order, mu)
    try:
        V, Q, Y, invA, gfu, uh, ph, p_old = build_solver(mesh, order, dt, inverse=inverse)
        (_, _), (_, qY) = Y.TrialFunction(), Y.TestFunction()

        Nu, Np = V.ndof, Q.ndof
        Uhist = np.empty((nsteps + 1, Nu), dtype=np.float64)
        Phist = np.empty((nsteps + 1, Np), dtype=np.float64)

        Uhist[0, :] = 0.0
        Phist[0, :] = p_old.vec.FV().NumPy()

        nrm = specialcf.normal(mesh.dim)

        for n in range(1, nsteps + 1):
            t = n * dt

            L = LinearForm(Y)
            L += (
                source_cf_time(t, T=T) * qY
                + (float(Sstor) / float(dt)) * p_old * qY
            ) * dx
            L += (-p_boundary_time(t, T=T) * InnerProduct(Y.TestFunction()[0].Trace(), nrm)) * ds
            L.Assemble()

            gfu.vec.data = invA * L.vec
            p_old.vec.data = ph.vec

            Uhist[n, :] = uh.vec.FV().NumPy()
            Phist[n, :] = ph.vec.FV().NumPy()

        return V, Q, Uhist, Phist
    finally:
        mesh.UnsetDeformation()


def gf_from_vec(space, vec):
    g = GridFunction(space)
    g.vec.FV().NumPy()[:] = vec
    return g


def sampled_l2_pressure_error_and_ref_norm(mesh_num, p_num, mesh_ref, p_ref, npts=81, eps=1e-10):
    xs = np.linspace(eps, 1.0 - eps, int(npts))
    ys = np.linspace(eps, 1.0 - eps, int(npts))

    e2_p = 0.0
    r2_p = 0.0
    cnt = 0

    for xv in xs:
        for yv in ys:
            mip_num = mesh_num(xv, yv)
            mip_ref = mesh_ref(xv, yv)
            if mip_num is None or mip_ref is None:
                continue

            pv_num = float(p_num(mip_num))
            pv_ref = float(p_ref(mip_ref))
            dp = pv_num - pv_ref

            e2_p += dp * dp
            r2_p += pv_ref * pv_ref
            cnt += 1

    if cnt == 0:
        raise RuntimeError('No sampling points inside both meshes')

    abs_p_err = np.sqrt(e2_p / cnt)
    ref_p_norm = np.sqrt(r2_p / cnt)
    return abs_p_err, ref_p_norm


def pressure_max_in_time_relative_error(mesh_num, Q_num, Phist_num, mesh_ref, Q_ref, Phist_ref, stride_ref=1, npts=81):
    n_num = int(Phist_num.shape[0])
    n_ref = int(Phist_ref.shape[0])
    expected_ref = (n_num - 1) * int(stride_ref) + 1
    if expected_ref != n_ref:
        raise ValueError(f'Incompatible history sizes for pressure error: n_num={n_num}, n_ref={n_ref}, stride_ref={stride_ref}')

    max_abs_err = 0.0
    max_ref_norm = 0.0

    for n in range(n_num):
        p_num = gf_from_vec(Q_num, Phist_num[n, :])
        p_ref = gf_from_vec(Q_ref, Phist_ref[n * stride_ref, :])
        abs_err_n, ref_norm_n = sampled_l2_pressure_error_and_ref_norm(mesh_num, p_num, mesh_ref, p_ref, npts=npts)
        max_abs_err = max(max_abs_err, abs_err_n)
        max_ref_norm = max(max_ref_norm, ref_norm_n)

    rel_p_max = max_abs_err / max_ref_norm if max_ref_norm > 1e-30 else max_abs_err
    return max_abs_err, rel_p_max


def sampled_velocity_l2_hdiv_error_and_ref_norm(mesh_num, u_num, mesh_ref, u_ref, npts=81, eps=1e-10):
    xs = np.linspace(eps, 1.0 - eps, int(npts))
    ys = np.linspace(eps, 1.0 - eps, int(npts))

    div_u_num = div(u_num)
    div_u_ref = div(u_ref)

    e2_l2 = 0.0
    r2_l2 = 0.0
    e2_hdiv = 0.0
    r2_hdiv = 0.0
    cnt = 0

    for xv in xs:
        for yv in ys:
            mip_num = mesh_num(xv, yv)
            mip_ref = mesh_ref(xv, yv)
            if mip_num is None or mip_ref is None:
                continue

            uv_num = np.array(u_num(mip_num), dtype=np.float64)
            uv_ref = np.array(u_ref(mip_ref), dtype=np.float64)
            du = uv_num - uv_ref

            dnum = float(np.asarray(div_u_num(mip_num), dtype=np.float64).reshape(-1)[0])
            dref = float(np.asarray(div_u_ref(mip_ref), dtype=np.float64).reshape(-1)[0])
            dd = dnum - dref

            l2_err_pt = float(np.dot(du, du))
            l2_ref_pt = float(np.dot(uv_ref, uv_ref))

            e2_l2 += l2_err_pt
            r2_l2 += l2_ref_pt
            e2_hdiv += l2_err_pt + dd * dd
            r2_hdiv += l2_ref_pt + dref * dref
            cnt += 1

    if cnt == 0:
        raise RuntimeError('No sampling points inside both meshes')

    abs_u_l2_err = np.sqrt(e2_l2 / cnt)
    ref_u_l2_norm = np.sqrt(r2_l2 / cnt)
    abs_u_hdiv_err = np.sqrt(e2_hdiv / cnt)
    ref_u_hdiv_norm = np.sqrt(r2_hdiv / cnt)

    return abs_u_l2_err, ref_u_l2_norm, abs_u_hdiv_err, ref_u_hdiv_norm


def sampled_divergence_error_and_ref_norm(mesh_num, u_num, mesh_ref, u_ref, npts=81, eps=1e-10):
    xs = np.linspace(eps, 1.0 - eps, int(npts))
    ys = np.linspace(eps, 1.0 - eps, int(npts))

    div_u_num = div(u_num)
    div_u_ref = div(u_ref)

    e2_div = 0.0
    r2_div = 0.0
    cnt = 0

    for xv in xs:
        for yv in ys:
            mip_num = mesh_num(xv, yv)
            mip_ref = mesh_ref(xv, yv)
            if mip_num is None or mip_ref is None:
                continue

            dnum = float(np.asarray(div_u_num(mip_num), dtype=np.float64).reshape(-1)[0])
            dref = float(np.asarray(div_u_ref(mip_ref), dtype=np.float64).reshape(-1)[0])
            dd = dnum - dref

            e2_div += dd * dd
            r2_div += dref * dref
            cnt += 1

    if cnt == 0:
        raise RuntimeError('No sampling points inside both meshes')

    abs_div_err = np.sqrt(e2_div / cnt)
    ref_div_norm = np.sqrt(r2_div / cnt)
    return abs_div_err, ref_div_norm


def velocity_max_in_time_relative_errors(mesh_num, V_num, Uhist_num, mesh_ref, V_ref, Uhist_ref, stride_ref=1, npts=81):
    n_num = int(Uhist_num.shape[0])
    n_ref = int(Uhist_ref.shape[0])
    expected_ref = (n_num - 1) * int(stride_ref) + 1
    if expected_ref != n_ref:
        raise ValueError(f'Incompatible velocity history sizes for error: n_num={n_num}, n_ref={n_ref}, stride_ref={stride_ref}')

    max_abs_l2_err = 0.0
    max_ref_l2_norm = 0.0
    max_abs_hdiv_err = 0.0
    max_ref_hdiv_norm = 0.0

    for n in range(n_num):
        u_num = gf_from_vec(V_num, Uhist_num[n, :])
        u_ref = gf_from_vec(V_ref, Uhist_ref[n * stride_ref, :])
        abs_l2_n, ref_l2_n, abs_hdiv_n, ref_hdiv_n = sampled_velocity_l2_hdiv_error_and_ref_norm(mesh_num, u_num, mesh_ref, u_ref, npts=npts)

        max_abs_l2_err = max(max_abs_l2_err, abs_l2_n)
        max_ref_l2_norm = max(max_ref_l2_norm, ref_l2_n)
        max_abs_hdiv_err = max(max_abs_hdiv_err, abs_hdiv_n)
        max_ref_hdiv_norm = max(max_ref_hdiv_norm, ref_hdiv_n)

    rel_u_l2 = max_abs_l2_err / max_ref_l2_norm if max_ref_l2_norm > 1e-30 else max_abs_l2_err
    rel_u_hdiv = max_abs_hdiv_err / max_ref_hdiv_norm if max_ref_hdiv_norm > 1e-30 else max_abs_hdiv_err

    return max_abs_l2_err, rel_u_l2, max_abs_hdiv_err, rel_u_hdiv


def divergence_max_in_time_errors(mesh_num, V_num, Uhist_num, mesh_ref, V_ref, Uhist_ref, stride_ref=1, npts=81):
    n_num = int(Uhist_num.shape[0])
    n_ref = int(Uhist_ref.shape[0])
    expected_ref = (n_num - 1) * int(stride_ref) + 1
    if expected_ref != n_ref:
        raise ValueError(f'Incompatible velocity history sizes for divergence error: n_num={n_num}, n_ref={n_ref}, stride_ref={stride_ref}')

    max_abs_div_err = 0.0
    max_ref_div_norm = 0.0

    for n in range(n_num):
        u_num = gf_from_vec(V_num, Uhist_num[n, :])
        u_ref = gf_from_vec(V_ref, Uhist_ref[n * stride_ref, :])
        abs_div_n, ref_div_n = sampled_divergence_error_and_ref_norm(mesh_num, u_num, mesh_ref, u_ref, npts=npts)
        max_abs_div_err = max(max_abs_div_err, abs_div_n)
        max_ref_div_norm = max(max_ref_div_norm, ref_div_n)

    rel_div = max_abs_div_err / max_ref_div_norm if max_ref_div_norm > 1e-30 else max_abs_div_err
    return max_abs_div_err, rel_div


def run_space_time_study(
    mu=(0.3, 0.3),
    T=1.0,
    order=2,
    maxh_list=(0.20, 0.10, 0.05),
    dt_list=(0.20, 0.10, 0.05, 0.025),
    maxh_ref=0.02,
    dt_ref=0.0125,
    sample_npts=81,
    inverse='umfpack',
):
    maxh_list = [float(h) for h in maxh_list]
    dt_list = [float(dt) for dt in dt_list]

    mesh_ref = Mesh(unit_square.GenerateMesh(maxh=float(maxh_ref)))
    Vref, Qref, Uref_hist, Pref_hist = solve_history(mesh_ref, order, dt_ref, T, mu, inverse=inverse)

    print('=== Deformed transient discretization study ===')
    print(f'mu={mu}, T={T}, order={order}, maxh_ref={maxh_ref}, dt_ref={dt_ref}')
    print(f'sampling grid: {sample_npts} x {sample_npts}\n')

    print('=== Time study (fixed space = maxh_ref) ===')
    mesh_time = Mesh(unit_square.GenerateMesh(maxh=float(maxh_ref)))
    _, _, Utime_ref, Ptime_ref = solve_history(mesh_time, order, dt_ref, T, mu, inverse=inverse)

    time_rows = []
    for dt in dt_list:
        ratio = dt / dt_ref
        rint = int(np.rint(ratio))
        tol = 1e-10 * max(1.0, abs(ratio))
        if abs(ratio - rint) > tol:
            raise ValueError(f'Each dt must be integer multiple of dt_ref; got dt={dt}, dt_ref={dt_ref}, ratio={ratio}')

        Vt, Qt, Ut_hist, Pt_hist = solve_history(mesh_time, order, dt, T, mu, inverse=inverse)

        abs_p, rel_p = pressure_max_in_time_relative_error(mesh_time, Qt, Pt_hist, mesh_time, Qt, Ptime_ref, stride_ref=rint, npts=sample_npts)
        abs_u_l2, rel_u_l2, abs_u_hdiv, rel_u_hdiv = velocity_max_in_time_relative_errors(mesh_time, Vt, Ut_hist, mesh_time, Vt, Utime_ref, stride_ref=rint, npts=sample_npts)
        abs_div, rel_div = divergence_max_in_time_errors(mesh_time, Vt, Ut_hist, mesh_time, Vt, Utime_ref, stride_ref=rint, npts=sample_npts)

        time_rows.append((dt, abs_p, rel_p, abs_u_l2, rel_u_l2, abs_u_hdiv, rel_u_hdiv, abs_div, rel_div))
        print(f'dt={dt:g}: rel_p_max_t={rel_p:.3e}, rel_u_l2_max_t={rel_u_l2:.3e}, rel_u_hdiv_max_t={rel_u_hdiv:.3e}, abs_div_max_t={abs_div:.3e}')

    print('\n=== Space study (fixed time step = dt_ref) ===')
    space_rows = []
    for h in maxh_list:
        mesh_h = Mesh(unit_square.GenerateMesh(maxh=float(h)))
        Vh, Qh, Uh_hist, Ph_hist = solve_history(mesh_h, order, dt_ref, T, mu, inverse=inverse)

        abs_p, rel_p = pressure_max_in_time_relative_error(mesh_h, Qh, Ph_hist, mesh_ref, Qref, Pref_hist, stride_ref=1, npts=sample_npts)
        abs_u_l2, rel_u_l2, abs_u_hdiv, rel_u_hdiv = velocity_max_in_time_relative_errors(mesh_h, Vh, Uh_hist, mesh_ref, Vref, Uref_hist, stride_ref=1, npts=sample_npts)
        abs_div, rel_div = divergence_max_in_time_errors(mesh_h, Vh, Uh_hist, mesh_ref, Vref, Uref_hist, stride_ref=1, npts=sample_npts)

        space_rows.append((h, abs_p, rel_p, abs_u_l2, rel_u_l2, abs_u_hdiv, rel_u_hdiv, abs_div, rel_div))
        print(f'maxh={h:g}: rel_p_max_t={rel_p:.3e}, rel_u_l2_max_t={rel_u_l2:.3e}, rel_u_hdiv_max_t={rel_u_hdiv:.3e}, abs_div_max_t={abs_div:.3e}')

    print('\n=== Joint (space,time) study ===')
    grid = {}
    for h in maxh_list:
        mesh_h = Mesh(unit_square.GenerateMesh(maxh=float(h)))
        for dt in dt_list:
            ratio = dt / dt_ref
            rint = int(np.rint(ratio))
            tol = 1e-10 * max(1.0, abs(ratio))
            if abs(ratio - rint) > tol:
                raise ValueError(f'Each dt must be integer multiple of dt_ref; got dt={dt}, dt_ref={dt_ref}, ratio={ratio}')

            Vh, Qh, Uh_hist, Ph_hist = solve_history(mesh_h, order, dt, T, mu, inverse=inverse)

            abs_p, rel_p = pressure_max_in_time_relative_error(mesh_h, Qh, Ph_hist, mesh_ref, Qref, Pref_hist, stride_ref=rint, npts=sample_npts)
            abs_u_l2, rel_u_l2, abs_u_hdiv, rel_u_hdiv = velocity_max_in_time_relative_errors(mesh_h, Vh, Uh_hist, mesh_ref, Vref, Uref_hist, stride_ref=rint, npts=sample_npts)
            abs_div, rel_div = divergence_max_in_time_errors(mesh_h, Vh, Uh_hist, mesh_ref, Vref, Uref_hist, stride_ref=rint, npts=sample_npts)

            grid[(float(h), float(dt))] = {
                'abs_p': abs_p,
                'rel_p': rel_p,
                'abs_u_l2': abs_u_l2,
                'rel_u_l2': rel_u_l2,
                'abs_u_hdiv': abs_u_hdiv,
                'rel_u_hdiv': rel_u_hdiv,
                'abs_div': abs_div,
                'rel_div': rel_div,
            }
            print(f'(maxh={h:g}, dt={dt:g}): rel_p_max_t={rel_p:.3e}, rel_u_l2_max_t={rel_u_l2:.3e}, rel_u_hdiv_max_t={rel_u_hdiv:.3e}, abs_div_max_t={abs_div:.3e}')

    return {
        'time_rows': time_rows,
        'space_rows': space_rows,
        'grid': grid,
    }


# Requested parameter
results = run_space_time_study(
    mu=(0.0, 0.0),
    T=1.0,
    order=2,
    maxh_list=(0.08,),
    dt_list=(0.00125/2,),
    maxh_ref=0.02,
    dt_ref=0.00125/4,
    sample_npts=81,
    inverse='pardiso',
)
