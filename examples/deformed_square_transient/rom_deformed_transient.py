from ngsolve import *
from ngsolve.webgui import Draw
from netgen.geom2d import unit_square
import json
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.lines import Line2D
from numpy.polynomial.legendre import legvander
from scipy.linalg import qr, cho_factor, cho_solve, lu_factor, lu_solve
from pymor.algorithms.pod import pod
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.numpy import NumpyMatrixOperator
plt.rcParams.update({
    "font.family": "serif",
    # Put a guaranteed Matplotlib serif first to avoid sans fallback.
    "font.serif": ["DejaVu Serif", "CMU Serif", "Computer Modern Roman", "Latin Modern Roman"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
})

# ============================================================
# Parameter domain
# ============================================================
class ParameterDomain:
    def __init__(self, mu1_range, mu2_range):
        self.mu1_min, self.mu1_max = mu1_range
        self.mu2_min, self.mu2_max = mu2_range

    def normalize(self, mu):
        mu1, mu2 = mu
        m1 = 2 * (mu1 - self.mu1_min) / (self.mu1_max - self.mu1_min) - 1
        m2 = 2 * (mu2 - self.mu2_min) / (self.mu2_max - self.mu2_min) - 1
        return m1, m2


# ============================================================
# Geometry deformation
# ============================================================
class DeformationModel:
    def evaluate(self, mu):
        raise NotImplementedError


class TopRightDeformation(DeformationModel):
    def evaluate(self, mu):
        mu1, mu2 = mu
        return CoefficientFunction((x * y * mu1, x * y * mu2))


class GeometryHandler:
    def __init__(self, mesh, order, deformation_model):
        self.mesh = mesh
        self.deformation_model = deformation_model
        self.deform_gf = GridFunction(VectorH1(mesh, order=order))

    def apply(self, mu):
        self.deform_gf.Set(self.deformation_model.evaluate(mu))
        self.mesh.SetDeformation(self.deform_gf)

    def clear(self):
        self.mesh.UnsetDeformation()


# ============================================================
# Transient Darcy model
#
#   K^{-1} u + grad(p) = 0
#   S p_t + div(u) = f
# ============================================================
class TransientDarcyModel:
    def __init__(
        self,
        K=1.0,
        Sstor=1.0,
        T=1.0,
        source_amp=10.0,
        source_sigma=0.12,
        source_center=(0.7, 0.4),
        p_init=0.0,
    ):
        self.K = float(K)
        self.Kinv = 1.0 / self.K
        self.Sstor = float(Sstor)
        self.T = float(T)
        self.source_amp = float(source_amp)
        self.source_sigma = float(source_sigma)
        self.source_center = tuple(source_center)
        self.p_init = float(p_init)

    def amplitude(self, t):
        return 1.0 - np.exp(-4.0 * float(t) / self.T)

    def source_base(self):
        cx, cy = self.source_center
        s = self.source_sigma
        return self.source_amp * exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s ** 2))

    def source(self, t):
        return self.amplitude(t) * self.source_base()

    def p_boundary(self, t):
        a = self.amplitude(t)
        return a * (x * x - y * y)

    def p_initial(self):
        return CoefficientFunction(self.p_init)


# ============================================================
# Sampling
# ============================================================
def sample_parameters(method, N, domain: ParameterDomain, seed=0):
    mu1_min, mu1_max = domain.mu1_min, domain.mu1_max
    mu2_min, mu2_max = domain.mu2_min, domain.mu2_max

    if method == "grid":
        mu1 = np.linspace(mu1_min, mu1_max, N)
        mu2 = np.linspace(mu2_min, mu2_max, N)
        mu = np.array([(a, b) for a in mu1 for b in mu2], dtype=float)

        w = np.ones((N, N))
        w[0, 1:-1] = w[-1, 1:-1] = w[1:-1, 0] = w[1:-1, -1] = 0.5
        w[0, 0] = w[0, -1] = w[-1, 0] = w[-1, -1] = 0.25
        return mu, np.diag(w.flatten())

    if method == "random":
        rng = np.random.default_rng(seed)
        mu1 = rng.uniform(mu1_min, mu1_max, N)
        mu2 = rng.uniform(mu2_min, mu2_max, N)
        return np.column_stack((mu1, mu2)), np.eye(N)

    raise ValueError(method)


def sample_parameters_gauss_lobatto(n_1d, domain: ParameterDomain):
    """
    Tensor-product Gauss-Lobatto points on the parameter rectangle.
    Returns (mu_points, W_diag) where W_diag is a diagonal weight matrix.
    """
    n = int(n_1d)
    if n < 2:
        raise ValueError(f"Gauss-Lobatto requires n_1d >= 2, got {n}")

    # Nodes in [-1, 1]: endpoints +/-1 plus roots of d/dx P_{n-1}(x)
    Pnm1 = np.polynomial.legendre.Legendre.basis(n - 1)
    dP = Pnm1.deriv()
    interior = np.asarray(dP.roots(), dtype=float) if n > 2 else np.array([], dtype=float)
    xhat = np.concatenate(([-1.0], interior, [1.0]))

    # 1D Gauss-Lobatto weights on [-1, 1]
    w1d = np.empty(n, dtype=float)
    w_end = 2.0 / (n * (n - 1))
    w1d[0] = w_end
    w1d[-1] = w_end
    if n > 2:
        vals = Pnm1(interior)
        w1d[1:-1] = 2.0 / (n * (n - 1) * vals * vals)

    a1, b1 = float(domain.mu1_min), float(domain.mu1_max)
    a2, b2 = float(domain.mu2_min), float(domain.mu2_max)
    x1 = 0.5 * (b1 - a1) * xhat + 0.5 * (a1 + b1)
    x2 = 0.5 * (b2 - a2) * xhat + 0.5 * (a2 + b2)
    w1_phys = 0.5 * (b1 - a1) * w1d
    w2_phys = 0.5 * (b2 - a2) * w1d

    mu = []
    w = []
    for i in range(n):
        for j in range(n):
            mu.append((float(x1[i]), float(x2[j])))
            w.append(float(w1_phys[i] * w2_phys[j]))
    return np.asarray(mu, dtype=float), np.diag(np.asarray(w, dtype=float))


# ============================================================
# Legendre basis
# ============================================================
def legendre_basis(mu, P, domain):
    m1, m2 = domain.normalize(mu)
    m1 = np.atleast_1d(m1)
    m2 = np.atleast_1d(m2)

    Vx = legvander(m1, P)
    Vy = legvander(m2, P)
    B = (Vx[:, :, None] * Vy[:, None, :]).reshape(len(m1), -1)
    return B[0] if B.shape[0] == 1 else B


# ============================================================
# POD helpers
# ============================================================
def build_pod_basis_from_snapshots(snapshots, product_mat):
    """
    Accept snapshots in either orientation:
      - (nsnaps, ndof)  [row-wise]
      - (ndof, nsnaps)  [column-wise]
    and build a pyMOR vector array robustly across pyMOR versions.
    """
    snapshots = np.asarray(snapshots, dtype=float)
    if snapshots.ndim != 2:
        raise ValueError(f"Expected 2D snapshot matrix, got shape {snapshots.shape}")

    ndof_from_product = int(product_mat.shape[0])
    if snapshots.shape[1] == ndof_from_product:
        snaps_rowwise = snapshots
    elif snapshots.shape[0] == ndof_from_product:
        snaps_rowwise = snapshots.T
    else:
        raise ValueError(
            f"Snapshot shape {snapshots.shape} incompatible with product matrix shape {product_mat.shape}"
        )

    nsnaps, ndof = snaps_rowwise.shape
    space = NumpyVectorSpace(ndof)

    # pyMOR API conventions differ by version. Try row-wise first, then column-wise.
    try:
        S = space.make_array(snaps_rowwise)
    except AssertionError:
        S = space.make_array(snaps_rowwise.T)
    product = NumpyMatrixOperator(product_mat)
    RB, svals = pod(S, rtol=1e-16, product=product)

    V = RB.to_numpy()
    if V.shape[0] != ndof:
        V = V.T

    return V, svals


def choose_rank_from_svals(svals, tau):
    svals = np.asarray(svals, dtype=float)
    if len(svals) == 0:
        return 0
    if len(svals) == 1:
        return 1
    s2 = svals ** 2
    total = np.sum(s2)
    if total <= 0:
        return 1
    tail = np.sqrt(np.cumsum(s2[::-1])[::-1] / total)[1:]
    idx = np.where(tail < tau)[0]
    return int(idx[0] + 1) if len(idx) else len(svals)


def metric_orthonormalize(V, M, tol=1e-12):
    if V.size == 0:
        return V
    G = V.T @ M @ V
    G = 0.5 * (G + G.T)
    evals, evecs = np.linalg.eigh(G)
    keep = evals > tol * max(1.0, np.max(np.abs(evals)))
    if not np.any(keep):
        return np.zeros((V.shape[0], 0))
    L = evecs[:, keep] @ np.diag(1.0 / np.sqrt(evals[keep]))
    return V @ L


# ============================================================
# Robust weighted L2 fit
# ============================================================
def fit_affine_matrix_family(P, mu_train, mats, W, domain, tol=1e-12):
    """
    mats shape: (m, n1, n2)
    returns coeffs shape: (Q, n1, n2)
    """
    B = np.array([legendre_basis(mu, P, domain) for mu in mu_train], dtype=float)
    m, Q = B.shape

    w = np.sqrt(np.diag(W))
    Bw = B * w[:, None]

    mats_flat = mats.reshape(m, -1) * w[:, None]

    Qw, R, piv = qr(Bw, mode="economic", pivoting=True)

    diagR = np.abs(np.diag(R))
    thresh = tol * np.max(diagR) if len(diagR) else tol
    rank = np.sum(diagR > thresh)

    Qr = Qw[:, :rank]
    Rr = R[:rank, :rank]
    pivr = piv[:rank]

    C_r = np.linalg.solve(Rr, Qr.T @ mats_flat)

    C = np.zeros((Q, mats_flat.shape[1]))
    C[pivr, :] = C_r

    n1, n2 = mats.shape[1], mats.shape[2]
    coeffs = C.T.reshape(n1, n2, Q).transpose(2, 0, 1)
    return coeffs


def fit_affine_vector_family(P, mu_train, vecs, W, domain, tol=1e-12):
    """
    vecs shape: (m, n)
    returns coeffs shape: (Q, n)
    """
    B = np.array([legendre_basis(mu, P, domain) for mu in mu_train], dtype=float)
    m, Q = B.shape

    w = np.sqrt(np.diag(W))
    Bw = B * w[:, None]

    vecs_w = vecs * w[:, None]

    Qw, R, piv = qr(Bw, mode="economic", pivoting=True)

    diagR = np.abs(np.diag(R))
    thresh = tol * np.max(diagR) if len(diagR) else tol
    rank = np.sum(diagR > thresh)

    Qr = Qw[:, :rank]
    Rr = R[:rank, :rank]
    pivr = piv[:rank]

    C_r = np.linalg.solve(Rr, Qr.T @ vecs_w)

    C = np.zeros((Q, vecs_w.shape[1]))
    C[pivr, :] = C_r
    return C


def eval_affine_matrix(mu, P, coeffs, domain):
    theta = np.array(legendre_basis(mu, P, domain))
    return np.tensordot(theta, coeffs, axes=(0, 0))


def eval_affine_vector(mu, P, coeffs, domain):
    theta = np.array(legendre_basis(mu, P, domain))
    return theta @ coeffs


# ============================================================
# Time cut helpers
# ============================================================
def cut_history_indices(times, t_cut):
    """
    history includes t=0 snapshot.
    return indices j such that times[j] >= t_cut
    """
    times = np.asarray(times, dtype=float)
    return np.where(times >= float(t_cut))[0]


def cut_snapshot_rows(times, S, t_cut):
    """
    S shape = (nsnaps, ndof) if stored row-wise
    """
    idx = cut_history_indices(times, t_cut)
    return times[idx], S[idx, :], idx


# ============================================================
# Transient mixed Darcy FOM solver
# ============================================================
class TransientMixedDarcyFOMSolver:
    def __init__(self, mesh, order, darcy_model: TransientDarcyModel):
        self.mesh = mesh
        self.order = order
        self.model = darcy_model

    def assemble_reference_operators(self, dt):
        V = HDiv(self.mesh, order=self.order)
        Q = L2(self.mesh, order=self.order - 1)
        Y = FESpace([V, Q])

        uV, vV = V.TrialFunction(), V.TestFunction()
        pQ, qQ = Q.TrialFunction(), Q.TestFunction()

        n = specialcf.normal(self.mesh.dim)
        Kinv_cf = CoefficientFunction(self.model.Kinv)
        S_cf = CoefficientFunction(self.model.Sstor)

        Auu = BilinearForm(V, symmetric=True)
        Auu += Kinv_cf * InnerProduct(uV, vV) * dx
        Auu.Assemble()

        B = BilinearForm(trialspace=V, testspace=Q)
        B += qQ * div(uV) * dx
        B.Assemble()

        Mp = BilinearForm(Q, symmetric=True)
        Mp += pQ * qQ * dx
        Mp.Assemble()

        f_base = self.model.source_base()
        lf_q = LinearForm(Q)
        lf_q += f_base * qQ * dx
        lf_q.Assemble()

        p_bdry_base = x * x - y * y
        lf_bd = LinearForm(V)
        lf_bd += (-p_bdry_base * InnerProduct(vV.Trace(), n)) * ds
        lf_bd.Assemble()

        u, p = Y.TrialFunction()
        v, q = Y.TestFunction()

        A_BE = BilinearForm(Y, symmetric=False)
        A_BE += (
            Kinv_cf * InnerProduct(u, v)
            - p * div(v)
            + q * div(u)
            + (S_cf / dt) * p * q
        ) * dx
        A_BE.Assemble()

        return {
            "V": V,
            "Q": Q,
            "Y": Y,
            "Nu": V.ndof,
            "Np": Q.ndof,
            "Auu": np.array(Auu.mat.ToDense(), dtype=float),
            "B": np.array(B.mat.ToDense(), dtype=float),
            "Mp": np.array(Mp.mat.ToDense(), dtype=float),
            "f_q_base": np.array(lf_q.vec.FV().NumPy(), dtype=float),
            "f_u_bdry_base": np.array(lf_bd.vec.FV().NumPy(), dtype=float),
            "A_BE": np.array(A_BE.mat.ToDense(), dtype=float),
        }

    def solve_time_history(self, dt=0.02, T=None, store_every=1, draw=False, sleep=0.0):
        if T is None:
            T = self.model.T

        ref = self.assemble_reference_operators(dt)
        V = ref["V"]
        Q = ref["Q"]
        Y = ref["Y"]
        Nu = ref["Nu"]
        Np = ref["Np"]

        Kinv_cf = CoefficientFunction(self.model.Kinv)
        S_cf = CoefficientFunction(self.model.Sstor)

        (u, p) = Y.TrialFunction()
        (v, q) = Y.TestFunction()

        gfu = GridFunction(Y)
        uh, ph = gfu.components

        p_old = GridFunction(Q)
        p_old.Set(self.model.p_initial())

        A_BE = BilinearForm(Y, symmetric=False)
        A_BE += (
            Kinv_cf * InnerProduct(u, v)
            - p * div(v)
            + q * div(u)
            + (S_cf / dt) * p * q
        ) * dx
        A_BE.Assemble()

        inv = A_BE.mat.Inverse(Y.FreeDofs(), inverse="pardiso")
        L = LinearForm(Y)
        L.Assemble()
        rhs_np = L.vec.FV().NumPy()
        rhs_tmp = np.zeros_like(rhs_np)
        f_u_base = ref["f_u_bdry_base"]
        f_q_base = ref["f_q_base"]
        Mp_dense = ref["Mp"]
        s_over_dt = self.model.Sstor / dt

        nsteps = int(round(T / dt))
        history = [{
            "t": 0.0,
            "u": np.zeros(Nu, dtype=float),
            "p": p_old.vec.FV().NumPy().copy(),
        }]
        p_old_np = history[0]["p"].copy()

        if draw:
            view_u = Draw(uh, self.mesh, "u_transient")
            view_p = Draw(ph, self.mesh, "p_transient")

        for nstep in range(1, nsteps + 1):
            t = nstep * dt

            alpha = self.model.amplitude(t)
            rhs_tmp[:Nu] = alpha * f_u_base
            rhs_tmp[Nu:] = alpha * f_q_base + s_over_dt * (Mp_dense @ p_old_np)
            rhs_np[:] = rhs_tmp

            gfu.vec.data = inv * L.vec
            p_old.vec.data = ph.vec
            p_old_np = ph.vec.FV().NumPy().copy()

            if (nstep % store_every) == 0:
                history.append({
                    "t": float(t),
                    "u": uh.vec.FV().NumPy().copy(),
                    "p": ph.vec.FV().NumPy().copy(),
                })

            if draw:
                view_u.Redraw()
                view_p.Redraw()
                if sleep > 0:
                    time.sleep(sleep)

        return {
            "history": history,
            "gfu": gfu,
            **ref,
            "dt": dt,
            "T": T,
        }


# ============================================================
# Experiment wrapper
# ============================================================
class TransientDarcyExperiment:
    def __init__(self, mesh, order, darcy_model, domain, deformation_model):
        self.mesh = mesh
        self.order = order
        self.domain = domain

        self.geom = GeometryHandler(mesh, order, deformation_model)
        self.solver = TransientMixedDarcyFOMSolver(mesh, order, darcy_model)
        self._fom_cache = {}

    def _fom_cache_key(self, mu, dt, T, store_every):
        mu_t = tuple(float(v) for v in mu)
        T_eff = self.solver.model.T if T is None else float(T)
        return (mu_t, float(dt), float(T_eff), int(store_every))

    def solve_full(self, mu, dt=0.02, T=None, store_every=1, draw=False, sleep=0.0):
        self.geom.apply(mu)
        out = self.solver.solve_time_history(
            dt=dt,
            T=T,
            store_every=store_every,
            draw=draw,
            sleep=sleep,
        )
        self.geom.clear()
        return out

    def solve_full_cached(self, mu, dt=0.02, T=None, store_every=1):
        key = self._fom_cache_key(mu, dt, T, store_every)
        if key not in self._fom_cache:
            self._fom_cache[key] = self.solve_full(
                key[0], dt=key[1], T=key[2], store_every=key[3], draw=False
            )
        return self._fom_cache[key]


# ============================================================
# Snapshot helpers
# ============================================================
def history_to_snapshot_matrices(history, Nu, Np):
    nsnaps = len(history)
    Su = np.empty((Nu, nsnaps), dtype=float)
    Sp = np.empty((Np, nsnaps), dtype=float)
    times = np.empty(nsnaps, dtype=float)

    for j, snap in enumerate(history):
        times[j] = snap["t"]
        Su[:, j] = snap["u"]
        Sp[:, j] = snap["p"]

    return times, Su, Sp


def replay_history(mesh, Y, history, sleep=0.05):
    gfu = GridFunction(Y)
    uh, ph = gfu.components

    view_u = Draw(uh, mesh, "u_replay")
    view_p = Draw(ph, mesh, "p_replay")

    for snap in history:
        uh.vec.FV().NumPy()[:] = snap["u"]
        ph.vec.FV().NumPy()[:] = snap["p"]
        view_u.Redraw()
        view_p.Redraw()
        print(f"t = {snap['t']:.3f}")
        time.sleep(sleep)


# ============================================================
# Transient ROM with supremizers
# ============================================================
class TransientMixedDarcyL2FitROM_Blocked:
    def __init__(self, experiment: TransientDarcyExperiment):
        self.experiment = experiment
        self.domain = experiment.domain

        self.Nu = None
        self.Np = None

        self.Auu_ref = None
        self.B_ref = None
        self.Mp_ref = None

        self.Vu_pod = None
        self.Vp = None
        self.S_sup = None
        self.Vu_stab = None

        self.r_u_pod = None
        self.r_p = None
        self.r_u_stab = None
        self.r_tot = None

        self.P = None
        self.Vrb = None

        self.Auu_coeffs = None
        self.B_coeffs = None
        self.Mp_coeffs = None
        self.fq_coeffs = None
        self.gu_coeffs = None

        self.dt = None
        self.T = None
        self.nsteps = None
        self.times = None
        self.t_cut = 0.0
        self.cut_indices = None
        self.times_cut = None

        # Offline reuse caches
        self._offline_cache_key = None
        self._offline_data = None
        self._pod_full = None
        self._tau_cache = {}

    # --------------------------------------------------------
    # Collect training data
    # --------------------------------------------------------
    def collect_training_data(self, mu_train, dt=0.02, T=1.0, store_every=1, t_cut=0.0):
        Uu_list = []
        Up_list = []

        Auu_list = []
        B_list = []
        Mp_list = []
        fq_list = []
        gu_list = []

        for i, mu in enumerate(mu_train):
            out = self.experiment.solve_full_cached(mu, dt=dt, T=T, store_every=store_every)

            if self.Nu is None:
                self.Nu = out["Nu"]
                self.Np = out["Np"]
                self.Auu_ref = out["Auu"]
                self.B_ref = out["B"]
                self.Mp_ref = out["Mp"]

                self.dt = dt
                self.T = T

            times, Su, Sp = history_to_snapshot_matrices(out["history"], out["Nu"], out["Np"])

            idx = cut_history_indices(times, t_cut)
            if len(idx) == 0:
                raise ValueError(f"No snapshots left after t_cut={t_cut}")

            self.times = times
            self.cut_indices = idx
            self.times_cut = times[idx]
            self.nsteps = len(self.times_cut) - 1

            # row-wise snapshots for POD: (nsnaps, ndof)
            Uu_list.append(Su[:, idx].T)
            Up_list.append(Sp[:, idx].T)

            Auu_list.append(out["Auu"])
            B_list.append(out["B"])
            Mp_list.append(out["Mp"])
            fq_list.append(out["f_q_base"])
            gu_list.append(out["f_u_bdry_base"])

        Uu = np.vstack(Uu_list)
        Up = np.vstack(Up_list)

        return {
            "Uu": Uu,
            "Up": Up,
            "Auu": np.array(Auu_list, dtype=float),
            "B": np.array(B_list, dtype=float),
            "Mp": np.array(Mp_list, dtype=float),
            "fq": np.array(fq_list, dtype=float),
            "gu": np.array(gu_list, dtype=float),
        }

    # --------------------------------------------------------
    # POD bases
    # --------------------------------------------------------
    def build_pod_bases(self, Uu, Up, tau_u=1e-8, tau_p=1e-8):
        Vu_full, svals_u = build_pod_basis_from_snapshots(Uu, product_mat=self.Auu_ref)
        Vp_full, svals_p = build_pod_basis_from_snapshots(Up, product_mat=self.Mp_ref)

        self.r_u_pod = choose_rank_from_svals(svals_u, tau_u)
        self.r_p = choose_rank_from_svals(svals_p, tau_p)

        self.Vu_pod = Vu_full[:, :self.r_u_pod]
        self.Vp = Vp_full[:, :self.r_p]

        return {
            "svals_u": svals_u,
            "svals_p": svals_p,
            "r_u_pod": self.r_u_pod,
            "r_p": self.r_p,
        }

    # --------------------------------------------------------
    # Supremizers
    # --------------------------------------------------------
    def build_supremizers(self):
        rhs = self.B_ref.T @ self.Vp
        self.S_sup = np.linalg.solve(self.Auu_ref, rhs)

        Vaug = np.column_stack([self.Vu_pod, self.S_sup])
        self.Vu_stab = metric_orthonormalize(Vaug, self.Auu_ref)

        self.r_u_stab = self.Vu_stab.shape[1]
        self.r_tot = self.r_u_stab + self.r_p

        self.Vrb = np.block([
            [self.Vu_stab, np.zeros((self.Nu, self.r_p))],
            [np.zeros((self.Np, self.r_u_stab)), self.Vp]
        ])

    # --------------------------------------------------------
    # Fit reduced blocks separately
    # --------------------------------------------------------
    def fit(
        self,
        mu_train,
        W_train,
        dt=0.02,
        T=1.0,
        store_every=1,
        P=4,
        tau_u=1e-8,
        tau_p=1e-8,
        t_cut=0.0,
    ):
        self.P = int(P)
        self.t_cut = float(t_cut)
        mu_train_arr = np.asarray(mu_train, dtype=float)
        offline_key = (
            tuple(tuple(float(v) for v in mu) for mu in mu_train_arr),
            float(dt),
            float(T),
            int(store_every),
            float(t_cut),
        )
        if self._offline_cache_key != offline_key:
            data = self.collect_training_data(
                mu_train_arr,
                dt=dt,
                T=T,
                store_every=store_every,
                t_cut=t_cut,
            )
            self._offline_cache_key = offline_key
            self._offline_data = data
            self._pod_full = None
            self._tau_cache = {}
        else:
            data = self._offline_data

        if self._pod_full is None:
            Vu_full, svals_u = build_pod_basis_from_snapshots(data["Uu"], product_mat=self.Auu_ref)
            Vp_full, svals_p = build_pod_basis_from_snapshots(data["Up"], product_mat=self.Mp_ref)
            self._pod_full = {
                "Vu_full": Vu_full,
                "Vp_full": Vp_full,
                "svals_u": svals_u,
                "svals_p": svals_p,
            }

        tau_key = (float(tau_u), float(tau_p))
        if tau_key not in self._tau_cache:
            svals_u = self._pod_full["svals_u"]
            svals_p = self._pod_full["svals_p"]
            Vu_full = self._pod_full["Vu_full"]
            Vp_full = self._pod_full["Vp_full"]

            r_u_pod = choose_rank_from_svals(svals_u, tau_key[0])
            r_p = choose_rank_from_svals(svals_p, tau_key[1])
            Vu_pod = Vu_full[:, :r_u_pod]
            Vp = Vp_full[:, :r_p]

            rhs = self.B_ref.T @ Vp
            S_sup = np.linalg.solve(self.Auu_ref, rhs)
            Vaug = np.column_stack([Vu_pod, S_sup])
            Vu_stab = metric_orthonormalize(Vaug, self.Auu_ref)
            r_u_stab = Vu_stab.shape[1]
            r_tot = r_u_stab + r_p
            Vrb = np.block([
                [Vu_stab, np.zeros((self.Nu, r_p))],
                [np.zeros((self.Np, r_u_stab)), Vp]
            ])

            m = len(mu_train_arr)
            Auu_r = np.empty((m, r_u_stab, r_u_stab), dtype=float)
            B_r   = np.empty((m, r_p, r_u_stab), dtype=float)
            Mp_r  = np.empty((m, r_p, r_p), dtype=float)
            fq_r  = np.empty((m, r_p), dtype=float)
            gu_r  = np.empty((m, r_u_stab), dtype=float)
            for i in range(m):
                Auu_r[i] = Vu_stab.T @ data["Auu"][i] @ Vu_stab
                B_r[i]   = Vp.T      @ data["B"][i]   @ Vu_stab
                Mp_r[i]  = Vp.T      @ data["Mp"][i]  @ Vp
                fq_r[i]  = Vp.T      @ data["fq"][i]
                gu_r[i]  = Vu_stab.T @ data["gu"][i]

            self._tau_cache[tau_key] = {
                "svals_u": svals_u,
                "svals_p": svals_p,
                "r_u_pod": r_u_pod,
                "r_p": r_p,
                "r_u_stab": r_u_stab,
                "r_tot": r_tot,
                "Vu_pod": Vu_pod,
                "Vp": Vp,
                "S_sup": S_sup,
                "Vu_stab": Vu_stab,
                "Vrb": Vrb,
                "Auu_r": Auu_r,
                "B_r": B_r,
                "Mp_r": Mp_r,
                "fq_r": fq_r,
                "gu_r": gu_r,
            }

        tc = self._tau_cache[tau_key]
        self.svals_u = tc["svals_u"]
        self.svals_p = tc["svals_p"]
        self.r_u_pod = tc["r_u_pod"]
        self.r_p = tc["r_p"]
        self.r_u_stab = tc["r_u_stab"]
        self.r_tot = tc["r_tot"]
        self.Vu_pod = tc["Vu_pod"]
        self.Vp = tc["Vp"]
        self.S_sup = tc["S_sup"]
        self.Vu_stab = tc["Vu_stab"]
        self.Vrb = tc["Vrb"]

        self.Auu_coeffs = fit_affine_matrix_family(self.P, mu_train_arr, tc["Auu_r"], W_train, self.domain)
        self.B_coeffs   = fit_affine_matrix_family(self.P, mu_train_arr, tc["B_r"],   W_train, self.domain)
        self.Mp_coeffs  = fit_affine_matrix_family(self.P, mu_train_arr, tc["Mp_r"],  W_train, self.domain)
        self.fq_coeffs  = fit_affine_vector_family(self.P, mu_train_arr, tc["fq_r"],  W_train, self.domain)
        self.gu_coeffs  = fit_affine_vector_family(self.P, mu_train_arr, tc["gu_r"],  W_train, self.domain)

        return {
            "P": self.P,
            "r_u_pod": self.r_u_pod,
            "r_p": self.r_p,
            "r_u_stab": self.r_u_stab,
            "r_tot": self.r_tot,
            "svals_u": self.svals_u,
            "svals_p": self.svals_p,
            "t_cut": self.t_cut,
        }

    # --------------------------------------------------------
    # Assemble online reduced blocks
    # --------------------------------------------------------
    def assemble_online_blocks(self, mu):
        Auu_r = eval_affine_matrix(mu, self.P, self.Auu_coeffs, self.domain)
        B_r   = eval_affine_matrix(mu, self.P, self.B_coeffs,   self.domain)
        Mp_r  = eval_affine_matrix(mu, self.P, self.Mp_coeffs,  self.domain)
        fq_r  = eval_affine_vector(mu, self.P, self.fq_coeffs,  self.domain)
        gu_r  = eval_affine_vector(mu, self.P, self.gu_coeffs,  self.domain)

        return Auu_r, B_r, Mp_r, fq_r, gu_r

    # --------------------------------------------------------
    # Online rollout
    # --------------------------------------------------------
    def solve_online(self, mu, p0_value=0.0, reconstruct_full=True):
        if self.cut_indices is None:
            raise RuntimeError("Call fit(...) before solve_online(...).")

        Auu_r, B_r, Mp_r, fq_r, gu_r = self.assemble_online_blocks(mu)

        r_u = self.r_u_stab
        r_p = self.r_p

        A_BE = np.zeros((r_u + r_p, r_u + r_p), dtype=float)
        A_BE[:r_u, :r_u] = Auu_r
        A_BE[:r_u, r_u:] = -B_r.T
        A_BE[r_u:, :r_u] = B_r
        A_BE[r_u:, r_u:] = (self.experiment.solver.model.Sstor / self.dt) * Mp_r

        Mp_fac = cho_factor(Mp_r, lower=True, check_finite=False)

        p0_r = np.zeros(r_p, dtype=float)
        if abs(p0_value) > 0:
            one_coeff = np.ones(self.Np, dtype=float) * p0_value
            rhs0 = self.Vp.T @ (self.Mp_ref @ one_coeff)
            p0_r = cho_solve(Mp_fac, rhs0, check_finite=False)

        p_old_r = p0_r.copy()

        A_BE_lu, A_BE_piv = lu_factor(A_BE, check_finite=False)

        # solve full reduced timeline first, then cut
        nsteps_full = len(self.times) - 1
        z_hist_full = None
        if reconstruct_full:
            z_hist_full = np.zeros((nsteps_full + 1, r_u + r_p), dtype=float)
            z_hist_full[0, r_u:] = p0_r

        for nstep in range(1, nsteps_full + 1):
            t = nstep * self.dt
            alpha = self.experiment.solver.model.amplitude(t)

            rhs = np.zeros(r_u + r_p, dtype=float)
            rhs[:r_u] = alpha * gu_r
            rhs[r_u:] = alpha * fq_r + (self.experiment.solver.model.Sstor / self.dt) * (Mp_r @ p_old_r)

            z = lu_solve((A_BE_lu, A_BE_piv), rhs, check_finite=False)
            if reconstruct_full:
                z_hist_full[nstep, :] = z
            p_old_r = z[r_u:]

        u_hist_cut = []
        p_hist_cut = []
        z_hist_cut = []

        if reconstruct_full:
            for j in self.cut_indices:
                z = z_hist_full[j]
                z_hist_cut.append(z.copy())
                u_hist_cut.append((self.Vu_stab @ z[:r_u]).copy())
                p_hist_cut.append((self.Vp @ z[r_u:]).copy())

        return {
            "z_hist": np.array(z_hist_cut),
            "u_hist": u_hist_cut,
            "p_hist": p_hist_cut,
            "times": self.times_cut.copy(),
            "A_BE": A_BE,
            "cut_indices": self.cut_indices.copy(),
        }


# ============================================================
# Error helpers
# ============================================================
def euclidean_rel_error(a, b, eps=1e-14):
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), eps)


def transient_rom_vs_fom_errors(rom_out, fom_out, t_cut=0.0, eps=1e-14):
    times_f, Su_f, Sp_f = history_to_snapshot_matrices(
        fom_out["history"], fom_out["Nu"], fom_out["Np"]
    )

    idx = cut_history_indices(times_f, t_cut)
    times_c = times_f[idx]
    Su_c = Su_f[:, idx]
    Sp_c = Sp_f[:, idx]

    u_hist_r = rom_out["u_hist"]
    p_hist_r = rom_out["p_hist"]

    nt = len(times_c)
    if len(u_hist_r) != nt:
        raise ValueError(
            f"ROM/FOM length mismatch after cut: len(u_hist_r)={len(u_hist_r)}, nt={nt}"
        )

    fom_u_norms = np.array([np.linalg.norm(Su_c[:, j]) for j in range(nt)], dtype=float)
    fom_p_norms = np.array([np.linalg.norm(Sp_c[:, j]) for j in range(nt)], dtype=float)

    max_u_ref = max(np.max(fom_u_norms), eps)
    max_p_ref = max(np.max(fom_p_norms), eps)

    abs_u = np.array(
        [np.linalg.norm(u_hist_r[j] - Su_c[:, j]) for j in range(nt)],
        dtype=float
    )
    abs_p = np.array(
        [np.linalg.norm(p_hist_r[j] - Sp_c[:, j]) for j in range(nt)],
        dtype=float
    )

    rel_u_timewise = abs_u / np.maximum(fom_u_norms, eps)
    rel_p_timewise = abs_p / np.maximum(fom_p_norms, eps)

    scaled_u = abs_u / max_u_ref
    scaled_p = abs_p / max_p_ref

    return {
        "times": times_c,
        "t_cut": float(t_cut),

        "abs_u": abs_u,
        "abs_p": abs_p,
        "max_abs_u": float(np.max(abs_u)),
        "max_abs_p": float(np.max(abs_p)),
        "mean_abs_u": float(np.mean(abs_u)),
        "mean_abs_p": float(np.mean(abs_p)),

        "rel_u_timewise": rel_u_timewise,
        "rel_p_timewise": rel_p_timewise,
        "max_rel_u_timewise": float(np.max(rel_u_timewise)),
        "max_rel_p_timewise": float(np.max(rel_p_timewise)),
        "mean_rel_u_timewise": float(np.mean(rel_u_timewise)),
        "mean_rel_p_timewise": float(np.mean(rel_p_timewise)),

        "scaled_u": scaled_u,
        "scaled_p": scaled_p,
        "max_scaled_u": float(np.max(scaled_u)),
        "max_scaled_p": float(np.max(scaled_p)),
        "mean_scaled_u": float(np.mean(scaled_u)),
        "mean_scaled_p": float(np.mean(scaled_p)),

        "max_u_ref": float(max_u_ref),
        "max_p_ref": float(max_p_ref),
    }


def discarded_energy_curve(svals):
    svals = np.asarray(svals, dtype=float)
    if len(svals) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    s2 = svals**2
    total = np.sum(s2)
    if total <= 0:
        return np.arange(1, len(svals)+1, dtype=int), np.zeros_like(svals)
    disc = np.sqrt(np.cumsum(s2[::-1])[::-1] / total)
    return np.arange(1, len(svals)+1, dtype=int), disc

def plot_discarded_energy(results, filename="discarded_energy.pdf"):
    ru, disc_u = discarded_energy_curve(results["svals_u"])
    rp, disc_p = discarded_energy_curve(results["svals_p"])
    curves = [np.asarray(disc_u, dtype=float), np.asarray(disc_p, dtype=float)]
    positive_vals = np.concatenate([c[c > 0.0] for c in curves if c.size > 0])
    if positive_vals.size:
        # Use a floor tied to available data so the log-axis doesn't extend
        # far below the visible curves.
        y_floor = float(np.min(positive_vals))
    else:
        y_floor = 1e-12
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ru, disc_u, linestyle="-", label="Velocity")
    ax.plot(rp, disc_p, linestyle="-", label="Pressure")
    ax.set_xlabel("Number of modes $r$")
    ax.set_ylabel("Relative discarded energy")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend()
    y_max = max(float(np.max(disc_u)) if len(disc_u) else y_floor, float(np.max(disc_p)) if len(disc_p) else y_floor)
    ax.set_yscale("log")
    if y_max <= y_floor:
        y_max = y_floor * 10.0
    x_max = max(int(np.max(ru)) if len(ru) else 1, int(np.max(rp)) if len(rp) else 1)
    ax.set_xlim(1, x_max)
    ax.set_ylim(y_floor, y_max)
    fig.tight_layout(); fig.savefig(filename, dpi=300, bbox_inches="tight"); plt.close(fig)

def _tau_to_latex(tau):
    tau = float(tau)
    if tau == 0.0:
        return "0"
    exp = int(np.round(np.log10(abs(tau))))
    if not np.isclose(abs(tau), 10.0**exp, rtol=1e-12, atol=0.0):
        return f"{tau:.2e}"
    sign = "-" if tau < 0 else ""
    return rf"{sign}10^{{{exp}}}"

def compute_transient_rom_solution_errors(experiment, mu_train, W_train, P, tau, mu_test, dt, T, store_every, t_cut, timing_mus=None, timing_repeats=0, fom_eval_outs=None, rom=None):
    ref_out = experiment.solve_full_cached(tuple(mu_train[0]), dt=dt, T=T, store_every=store_every)
    Mu = np.asarray(ref_out["Auu"], dtype=float)
    Mp = np.asarray(ref_out["Mp"], dtype=float)
    Mhdiv = build_hdiv_product(Mu, Mp, np.asarray(ref_out["B"], dtype=float))

    if rom is None:
        rom = TransientMixedDarcyL2FitROM_Blocked(experiment)
    rom_data = rom.fit(
        mu_train=mu_train, W_train=W_train, dt=dt, T=T, store_every=store_every,
        P=int(P), tau_u=float(tau), tau_p=float(tau), t_cut=float(t_cut),
    )

    errs_u_l2 = []
    errs_u_hdiv = []
    errs_p_l2 = []
    for mu in mu_test:
        mu_t = tuple(mu)
        rom_out = rom.solve_online(mu_t, p0_value=0.0)
        if fom_eval_outs is not None and mu_t in fom_eval_outs:
            fom_out = fom_eval_outs[mu_t]
        else:
            fom_out = experiment.solve_full_cached(mu_t, dt=dt, T=T, store_every=store_every)
        em = transient_rom_vs_fom_errors_metric(rom_out, fom_out, Mu, Mp, Mhdiv, t_cut=t_cut)
        errs_u_l2.append(em["u_l2_mean"])
        errs_u_hdiv.append(em["u_hdiv_mean"])
        errs_p_l2.append(em["p_l2_mean"])

    timing = None
    if timing_mus is not None and timing_repeats > 0 and len(timing_mus) > 0:
        rom.solve_online(tuple(timing_mus[0]), p0_value=0.0, reconstruct_full=False)
        rom_times = []
        for _ in range(timing_repeats):
            t0 = time.perf_counter()
            for mu in timing_mus:
                rom.solve_online(tuple(mu), p0_value=0.0, reconstruct_full=False)
            rom_times.append(time.perf_counter() - t0)
        rom_total_mean = float(np.mean(rom_times))
        timing = {
            "rom_times": np.array(rom_times, dtype=float),
            "rom_total_mean": rom_total_mean,
            "rom_time_per_solve": rom_total_mean / len(timing_mus),
        }

    return {
        "u_l2_mean": float(np.mean(errs_u_l2)),
        "u_l2_max": float(np.max(errs_u_l2)),
        "u_hdiv_mean": float(np.mean(errs_u_hdiv)),
        "u_hdiv_max": float(np.max(errs_u_hdiv)),
        "p_l2_mean": float(np.mean(errs_p_l2)),
        "p_l2_max": float(np.max(errs_p_l2)),
    }, {
        "r_tot": int(rom_data["r_tot"]),
        "r_p": int(rom_data["r_p"]),
        "r_u_stab": int(rom_data["r_u_stab"]),
        "svals_u": np.array(rom_data["svals_u"], dtype=float),
        "svals_p": np.array(rom_data["svals_p"], dtype=float),
    }, timing

def run_convergence_study_transient(
    experiment,
    domain,
    P_values,
    tau_values,
    mu_train,
    W_train,
    n_test_random=40,
    random_seed=1234,
    dt=0.1,
    T=1.0,
    store_every=1,
    t_cut=0.0,
    measure_speed=False,
    speed_repeats=3,
    operator_sampling_method="gauss_lobatto",
    operator_n_1d=8,
):
    mu_test, _ = sample_parameters(method="random", N=n_test_random, domain=domain, seed=random_seed)
    if operator_sampling_method == "gauss_lobatto":
        mu_operator_eval, _ = sample_parameters_gauss_lobatto(operator_n_1d, domain=domain)
    elif operator_sampling_method == "random":
        mu_operator_eval, _ = sample_parameters(method="random", N=n_test_random, domain=domain, seed=random_seed + 1)
    else:
        raise ValueError(f"Unknown operator_sampling_method={operator_sampling_method!r}")

    timing_mus = [tuple(mu) for mu in mu_test]
    mu_train_tuples = [tuple(mu) for mu in np.asarray(mu_train, dtype=float)]

    train_blocks = _collect_affine_training_blocks(experiment, mu_train_tuples, dt, T, store_every)
    shared_rom = TransientMixedDarcyL2FitROM_Blocked(experiment)
    fom_eval_outs = {}
    for mu in timing_mus:
        fom_eval_outs[mu] = experiment.solve_full_cached(mu, dt=dt, T=T, store_every=store_every)

    results = {
        "P_values": np.array(P_values, dtype=int),
        "tau_values": list(tau_values),
        "l2fit_operator_int": [],
        "l2fit_operator_max": [],
        "l2fit_u_l2_int": [],
        "l2fit_u_l2_max": [],
        "l2fit_u_hdiv_int": [],
        "l2fit_u_hdiv_max": [],
        "l2fit_p_l2_int": [],
        "l2fit_p_l2_max": [],
        "rom_errors": {
            tau: {
                "u_l2_mean": [], "u_l2_max": [],
                "u_hdiv_mean": [], "u_hdiv_max": [],
                "p_l2_mean": [], "p_l2_max": [],
            } for tau in tau_values
        },
        "rom_modes": {},
        "svals_u": None,
        "svals_p": None,
    }

    if measure_speed:
        speed = {
            "n_test": len(timing_mus),
            "n_repeats": int(speed_repeats),
            "mu_test": np.array(mu_test, dtype=float),
            "rom_total_by_tau": {tau: [] for tau in tau_values},
            "rom_per_solve_by_tau": {tau: [] for tau in tau_values},
            "speedup_by_tau": {tau: [] for tau in tau_values},
        }
        experiment.solve_full(tuple(timing_mus[0]), dt=dt, T=T, store_every=store_every, draw=False)
        fom_times = []
        for _ in range(speed_repeats):
            t0 = time.perf_counter()
            for mu in timing_mus:
                experiment.solve_full(tuple(mu), dt=dt, T=T, store_every=store_every, draw=False)
            fom_times.append(time.perf_counter() - t0)
        speed["fom_times"] = np.array(fom_times, dtype=float)
        speed["fom_total_mean"] = float(np.mean(fom_times))
        speed["fom_time_per_solve"] = speed["fom_total_mean"] / max(1, len(timing_mus))
        results["speed"] = speed

    for P in P_values:
        fiterrs = compute_transient_l2fit_operator_and_solution_errors(
            experiment, domain, mu_train, W_train, int(P), mu_operator_eval, dt, T, store_every,
            precomputed_train=train_blocks, fom_eval_outs=fom_eval_outs,
        )
        results["l2fit_operator_int"].append(fiterrs["operator_int"])
        results["l2fit_operator_max"].append(fiterrs["operator_max"])
        results["l2fit_u_l2_int"].append(fiterrs["u_l2_int"])
        results["l2fit_u_l2_max"].append(fiterrs["u_l2_max"])
        results["l2fit_u_hdiv_int"].append(fiterrs["u_hdiv_int"])
        results["l2fit_u_hdiv_max"].append(fiterrs["u_hdiv_max"])
        results["l2fit_p_l2_int"].append(fiterrs["p_l2_int"])
        results["l2fit_p_l2_max"].append(fiterrs["p_l2_max"])

        for tau in tau_values:
            rom_errs, rom_data, rom_timing = compute_transient_rom_solution_errors(
                experiment=experiment, mu_train=mu_train, W_train=W_train, P=P, tau=tau,
                mu_test=mu_test, dt=dt, T=T, store_every=store_every, t_cut=t_cut,
                timing_mus=timing_mus if measure_speed else None,
                timing_repeats=speed_repeats if measure_speed else 0,
                fom_eval_outs=fom_eval_outs,
                rom=shared_rom,
            )
            for key in rom_errs:
                results["rom_errors"][tau][key].append(rom_errs[key])
            if tau not in results["rom_modes"]:
                results["rom_modes"][tau] = rom_data["r_tot"]
                if "rom_r_components" not in results:
                    results["rom_r_components"] = {}
                if "r_p" in rom_data and "r_u_stab" in rom_data:
                    r_p_val = int(rom_data["r_p"])
                    r_u_stab_val = int(rom_data["r_u_stab"])
                    r_trip = r_triplet_from_stabilized_dims(r_u_stab_val, r_p_val)
                else:
                    # Backward-compatible fallback for older cached metadata
                    r_trip = (np.nan, np.nan, np.nan)
                results["rom_r_components"][tau] = {
                    "r_p": float(r_trip[0]),
                    "r_u": float(r_trip[1]),
                    "r_s": float(r_trip[2]),
                }
            if results["svals_u"] is None:
                results["svals_u"] = np.array(rom_data["svals_u"], dtype=float)
                results["svals_p"] = np.array(rom_data["svals_p"], dtype=float)
            if measure_speed and rom_timing is not None:
                results["speed"]["rom_total_by_tau"][tau].append(rom_timing["rom_total_mean"])
                results["speed"]["rom_per_solve_by_tau"][tau].append(rom_timing["rom_time_per_solve"])
                results["speed"]["speedup_by_tau"][tau].append(results["speed"]["fom_total_mean"] / rom_timing["rom_total_mean"])

    for key in [
        "l2fit_operator_int", "l2fit_operator_max",
        "l2fit_u_l2_int", "l2fit_u_l2_max",
        "l2fit_u_hdiv_int", "l2fit_u_hdiv_max",
        "l2fit_p_l2_int", "l2fit_p_l2_max",
    ]:
        results[key] = np.asarray(results[key], dtype=float)

    for tau in tau_values:
        for key in results["rom_errors"][tau]:
            results["rom_errors"][tau][key] = np.asarray(results["rom_errors"][tau][key], dtype=float)

    if measure_speed:
        for tau in tau_values:
            results["speed"]["rom_total_by_tau"][tau] = np.asarray(results["speed"]["rom_total_by_tau"][tau], dtype=float)
            results["speed"]["rom_per_solve_by_tau"][tau] = np.asarray(results["speed"]["rom_per_solve_by_tau"][tau], dtype=float)
            results["speed"]["speedup_by_tau"][tau] = np.asarray(results["speed"]["speedup_by_tau"][tau], dtype=float)

    return results

def format_r_value(r):
    return f"{float(r):.2f}".rstrip("0").rstrip(".")

def format_r_triplet(vals):
    if vals is None:
        return "N/A"
    r_p, r_u, r_s = vals
    return f"[{format_r_value(r_p)},{format_r_value(r_u)},{format_r_value(r_s)}]"

def r_triplet_from_stabilized_dims(r_u_stab, r_p):
    r_p = float(r_p)
    r_s = r_p
    r_u = float(r_u_stab) - r_s
    return (r_p, r_u, r_s)

def _json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_results_bundle_json(path, results=None, speed_results=None, extra=None):
    payload = {
        "results": _json_ready(results),
        "speed_results": _json_ready(speed_results),
        "extra": _json_ready(extra) if extra is not None else {},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_results_bundle_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def speed_results_from_convergence(results):
    if "speed" not in results:
        raise ValueError("Convergence results do not contain speed data. Run with measure_speed=True.")
    tau_values = np.asarray(results["tau_values"], dtype=float)
    P_values = np.asarray(results["P_values"], dtype=int)
    return {
        "P_values": P_values,
        "tau_values": tau_values,
        "speedup_by_tau": {float(tau): np.asarray(results["speed"]["speedup_by_tau"][tau], dtype=float) for tau in tau_values},
        "rom_modes_by_tau": {float(tau): np.full(len(P_values), int(results["rom_modes"][tau]), dtype=int) for tau in tau_values},
        "rom_total_by_tau": {float(tau): np.asarray(results["speed"]["rom_total_by_tau"][tau], dtype=float) for tau in tau_values},
        "rom_per_solve_by_tau": {float(tau): np.asarray(results["speed"]["rom_per_solve_by_tau"][tau], dtype=float) for tau in tau_values},
        "n_test": int(results["speed"]["n_test"]),
        "n_repeats": int(results["speed"]["n_repeats"]),
        "mu_test": np.asarray(results["speed"]["mu_test"], dtype=float),
        "fom_times": np.asarray(results["speed"]["fom_times"], dtype=float),
        "fom_total_mean": float(results["speed"]["fom_total_mean"]),
        "fom_time_per_solve": float(results["speed"]["fom_time_per_solve"]),
    }

def plot_speedup_vs_polynomial_degree(speed_results, filename="rom_speedup_vs_polynomial_degree.pdf"):
    P_values = np.asarray(speed_results["P_values"], dtype=int)
    tau_values = np.asarray(speed_results["tau_values"], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    for tau in tau_values:
        tau_key = float(tau)
        r_tot = int(speed_results["rom_modes_by_tau"][tau_key][0])
        lbl = rf"$r={r_tot}$"
        ax.plot(P_values, speed_results["speedup_by_tau"][tau_key], marker="o", label=lbl)
    ax.set_xticks(P_values)
    ax.set_xlabel("Polynomial degree $P$")
    ax.set_ylabel("Speed-up (FOM / ROM)")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend()
    fig.tight_layout(); fig.savefig(filename, dpi=300, bbox_inches="tight"); plt.close(fig)

def plot_rom_velocity_errors(results, filename="rom_velocity_errors.pdf"):
    P_values = results["P_values"]
    tau_values = results["tau_values"]
    fig, ax = plt.subplots(figsize=(8, 5))
    tau_handles = []
    meta_handles = [
        Line2D([0], [0], color="black", marker="x", linestyle="None", markersize=7, label=r"$L^2$"),
        Line2D([0], [0], color="black", marker="s", markerfacecolor="none", linestyle="None", markersize=6, label=r"$H(\mathrm{div})$"),
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.8, label="Mean"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.8, label="Max"),
    ]
    for tau in tau_values:
        rom_errs_map = results["rom_errors"]
        rom_modes_map = results["rom_modes"]
        errs = rom_errs_map.get(tau, rom_errs_map.get(str(tau)))
        r_tot = rom_modes_map.get(tau, rom_modes_map.get(str(tau)))
        r_lbl = str(r_tot)
        comp_src = None
        if "rom_r_components" in results:
            comp_map = results["rom_r_components"]
            comp_src = comp_map.get(tau, comp_map.get(str(tau), None))
        if comp_src is not None:
            r_lbl = format_r_triplet((comp_src.get("r_p", np.nan), comp_src.get("r_u", np.nan), comp_src.get("r_s", np.nan)))
        else:
            # Backward-compatible reconstruction when old JSON lacks stored components.
            if "svals_p" in results:
                r_p = choose_rank_from_svals(np.asarray(results["svals_p"], dtype=float), float(tau))
                r_u_stab = float(r_tot) - float(r_p)
                r_lbl = format_r_triplet(r_triplet_from_stabilized_dims(r_u_stab, float(r_p)))

        h1, = ax.plot(P_values, errs["u_l2_mean"], marker="x", linestyle="-", alpha=0.55, zorder=2)
        ax.plot(P_values, errs["u_l2_max"], marker="x", linestyle="--", color=h1.get_color(), alpha=0.55, zorder=2)

        ax.plot(P_values, errs["u_hdiv_mean"], marker="s", linestyle="-", color=h1.get_color(), alpha=0.95, markerfacecolor="none", markeredgewidth=1.4, zorder=3)
        ax.plot(P_values, errs["u_hdiv_max"], marker="s", linestyle="--", color=h1.get_color(), alpha=0.95, markerfacecolor="none", markeredgewidth=1.4, zorder=3)

        tau_handles.append(Line2D([0], [0], color=h1.get_color(), linestyle="-", linewidth=1.8, label=rf"$\tau={_tau_to_latex(tau)}$ $(r={r_lbl})$"))

    ax.set_yscale("log")
    ax.set_xlabel("Polynomial degree $P$")
    ax.set_ylabel(r"Relative error")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    leg1 = ax.legend(handles=tau_handles, loc="upper right", title="")
    ax.add_artist(leg1)
    ax.legend(handles=meta_handles, loc="lower left", title="")
    fig.tight_layout(); fig.savefig(filename, dpi=300, bbox_inches="tight"); plt.close(fig)

def plot_rom_pressure_errors(results, filename="rom_pressure_errors.pdf"):
    P_values = results["P_values"]
    tau_values = results["tau_values"]
    fig, ax = plt.subplots(figsize=(8, 5))
    quantity_handles = []
    style_handles = [
        Line2D([0], [0], color="black", marker="o", linestyle="-", linewidth=1.8, markerfacecolor="black", markeredgecolor="black", label="Mean"),
        Line2D([0], [0], color="black", marker="o", linestyle="--", linewidth=1.8, markerfacecolor="none", markeredgecolor="black", markeredgewidth=1.4, label="Max"),
    ]
    for tau in tau_values:
        rom_errs_map = results["rom_errors"]
        rom_modes_map = results["rom_modes"]
        errs = rom_errs_map.get(tau, rom_errs_map.get(str(tau)))
        r_tot = rom_modes_map.get(tau, rom_modes_map.get(str(tau)))
        r_lbl = str(r_tot)
        comp_src = None
        if "rom_r_components" in results:
            comp_map = results["rom_r_components"]
            comp_src = comp_map.get(tau, comp_map.get(str(tau), None))
        if comp_src is not None:
            r_lbl = format_r_triplet((comp_src.get("r_p", np.nan), comp_src.get("r_u", np.nan), comp_src.get("r_s", np.nan)))
        else:
            # Backward-compatible reconstruction when old JSON lacks stored components.
            if "svals_p" in results:
                r_p = choose_rank_from_svals(np.asarray(results["svals_p"], dtype=float), float(tau))
                r_u_stab = float(r_tot) - float(r_p)
                r_lbl = format_r_triplet(r_triplet_from_stabilized_dims(r_u_stab, float(r_p)))
        h, = ax.plot(P_values, errs["p_l2_mean"], marker="o", linestyle="-")
        ax.plot(P_values, errs["p_l2_max"], marker="o", linestyle="--", color=h.get_color(), markerfacecolor="none", markeredgewidth=1.4)
        quantity_handles.append(Line2D([0], [0], color=h.get_color(), linestyle="-", linewidth=1.8, label=rf"$\tau={_tau_to_latex(tau)}$ $(r={r_lbl})$"))
    ax.set_yscale("log")
    ax.set_xlabel("Polynomial degree $P$")
    ax.set_ylabel(r"Relative $L^2$ error")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    leg1 = ax.legend(handles=quantity_handles, loc="upper right", title="")
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, loc="lower left", title="")
    fig.tight_layout(); fig.savefig(filename, dpi=300, bbox_inches="tight"); plt.close(fig)

# ============================================================
# Main
# ============================================================

def build_hdiv_product(Mu, Mp, B):
    return Mu + B.T @ Mp @ B


def squared_metric_norm(x, M):
    x = np.asarray(x, dtype=float)
    return float(x.T @ (M @ x))


def relative_metric_error(x_approx, x_true, M, eps=1e-14):
    e = np.asarray(x_approx, dtype=float) - np.asarray(x_true, dtype=float)
    num = np.sqrt(max(squared_metric_norm(e, M), 0.0))
    den = np.sqrt(max(squared_metric_norm(np.asarray(x_true, dtype=float), M), eps))
    return num / max(den, eps)


def transient_rom_vs_fom_errors_metric(rom_out, fom_out, Mu, Mp, Mhdiv, t_cut=0.0, eps=1e-14):
    times_f, Su_f, Sp_f = history_to_snapshot_matrices(fom_out["history"], fom_out["Nu"], fom_out["Np"])
    idx = cut_history_indices(times_f, t_cut)
    Su_c = Su_f[:, idx]
    Sp_c = Sp_f[:, idx]
    u_hist_r = rom_out["u_hist"]
    p_hist_r = rom_out["p_hist"]

    nt = Su_c.shape[1]
    if len(u_hist_r) != nt:
        raise ValueError(f"ROM/FOM length mismatch: {len(u_hist_r)} vs {nt}")

    eu_l2 = np.empty(nt, dtype=float)
    eu_hdiv = np.empty(nt, dtype=float)
    ep_l2 = np.empty(nt, dtype=float)

    for j in range(nt):
        u_t = Su_c[:, j]
        p_t = Sp_c[:, j]
        eu_l2[j] = relative_metric_error(u_hist_r[j], u_t, Mu, eps=eps)
        eu_hdiv[j] = relative_metric_error(u_hist_r[j], u_t, Mhdiv, eps=eps)
        ep_l2[j] = relative_metric_error(p_hist_r[j], p_t, Mp, eps=eps)

    return {
        "u_l2_mean": float(np.mean(eu_l2)),
        "u_l2_max": float(np.max(eu_l2)),
        "u_hdiv_mean": float(np.mean(eu_hdiv)),
        "u_hdiv_max": float(np.max(eu_hdiv)),
        "p_l2_mean": float(np.mean(ep_l2)),
        "p_l2_max": float(np.max(ep_l2)),
    }


def _collect_affine_training_blocks(experiment, mu_train, dt, T, store_every):
    Auu_list = []
    B_list = []
    Mp_list = []
    fq_list = []
    gu_list = []
    for mu in mu_train:
        out = experiment.solve_full_cached(tuple(mu), dt=dt, T=T, store_every=store_every)
        Auu_list.append(out["Auu"])
        B_list.append(out["B"])
        Mp_list.append(out["Mp"])
        fq_list.append(out["f_q_base"])
        gu_list.append(out["f_u_bdry_base"])
    return {
        "Auu": np.array(Auu_list, dtype=float),
        "B": np.array(B_list, dtype=float),
        "Mp": np.array(Mp_list, dtype=float),
        "fq": np.array(fq_list, dtype=float),
        "gu": np.array(gu_list, dtype=float),
    }


def compute_transient_l2fit_operator_and_solution_errors(experiment, domain, mu_train, W_train, P, mu_eval, dt, T, store_every, precomputed_train=None, fom_eval_outs=None):
    train = precomputed_train if precomputed_train is not None else _collect_affine_training_blocks(experiment, mu_train, dt, T, store_every)
    C_Auu = fit_affine_matrix_family(P, mu_train, train["Auu"], W_train, domain)
    C_B = fit_affine_matrix_family(P, mu_train, train["B"], W_train, domain)
    C_Mp = fit_affine_matrix_family(P, mu_train, train["Mp"], W_train, domain)
    C_fq = fit_affine_vector_family(P, mu_train, train["fq"], W_train, domain)
    C_gu = fit_affine_vector_family(P, mu_train, train["gu"], W_train, domain)

    relA = []
    u_l2_vals = []
    u_hdiv_vals = []
    p_l2_vals = []

    for mu in mu_eval:
        mu_t = tuple(mu)
        if fom_eval_outs is not None and mu_t in fom_eval_outs:
            out = fom_eval_outs[mu_t]
        else:
            out = experiment.solve_full_cached(mu_t, dt=dt, T=T, store_every=store_every)

        Auu_hat = eval_affine_matrix(mu_t, P, C_Auu, domain)
        B_hat = eval_affine_matrix(mu_t, P, C_B, domain)
        Mp_hat = eval_affine_matrix(mu_t, P, C_Mp, domain)
        fq_hat = eval_affine_vector(mu_t, P, C_fq, domain)
        gu_hat = eval_affine_vector(mu_t, P, C_gu, domain)

        numA = (
            np.linalg.norm(Auu_hat - out["Auu"], ord="fro")**2
            + np.linalg.norm(B_hat - out["B"], ord="fro")**2
            + np.linalg.norm(Mp_hat - out["Mp"], ord="fro")**2
            + np.linalg.norm(fq_hat - out["f_q_base"])**2
            + np.linalg.norm(gu_hat - out["f_u_bdry_base"])**2
        )
        denA = (
            np.linalg.norm(out["Auu"], ord="fro")**2
            + np.linalg.norm(out["B"], ord="fro")**2
            + np.linalg.norm(out["Mp"], ord="fro")**2
            + np.linalg.norm(out["f_q_base"])**2
            + np.linalg.norm(out["f_u_bdry_base"])**2
        )
        relA.append(np.sqrt(numA / max(denA, 1e-30)))

        Mu = np.asarray(out["Auu"], dtype=float)
        Mp = np.asarray(out["Mp"], dtype=float)
        B = np.asarray(out["B"], dtype=float)
        Mhdiv = build_hdiv_product(Mu, Mp, B)

        fq_true = np.asarray(out["f_q_base"], dtype=float)
        gu_true = np.asarray(out["f_u_bdry_base"], dtype=float)

        A_BE_true = np.block([
            [out["Auu"], -out["B"].T],
            [out["B"], (experiment.solver.model.Sstor / dt) * out["Mp"]],
        ])
        A_BE_fit = np.block([
            [Auu_hat, -B_hat.T],
            [B_hat, (experiment.solver.model.Sstor / dt) * Mp_hat],
        ])

        p_old_true = np.asarray(out["history"][0]["p"], dtype=float)
        p_old_fit = p_old_true.copy()

        for n in range(1, len(out["history"])):
            t = float(out["history"][n]["t"])
            alpha = experiment.solver.model.amplitude(t)

            rhs_true = np.zeros(out["Nu"] + out["Np"], dtype=float)
            rhs_true[:out["Nu"]] = alpha * gu_true
            rhs_true[out["Nu"]:] = alpha * fq_true + (experiment.solver.model.Sstor / dt) * (out["Mp"] @ p_old_true)
            z_true = np.linalg.solve(A_BE_true, rhs_true)
            u_true = z_true[:out["Nu"]]
            p_true = z_true[out["Nu"]:]
            p_old_true = p_true

            rhs_fit = np.zeros(out["Nu"] + out["Np"], dtype=float)
            rhs_fit[:out["Nu"]] = alpha * gu_hat
            rhs_fit[out["Nu"]:] = alpha * fq_hat + (experiment.solver.model.Sstor / dt) * (Mp_hat @ p_old_fit)
            z_fit = np.linalg.solve(A_BE_fit, rhs_fit)
            u_fit = z_fit[:out["Nu"]]
            p_fit = z_fit[out["Nu"]:]
            p_old_fit = p_fit

            u_l2_vals.append(relative_metric_error(u_fit, u_true, Mu))
            u_hdiv_vals.append(relative_metric_error(u_fit, u_true, Mhdiv))
            p_l2_vals.append(relative_metric_error(p_fit, p_true, Mp))

    relA = np.asarray(relA, dtype=float)
    u_l2_vals = np.asarray(u_l2_vals, dtype=float)
    u_hdiv_vals = np.asarray(u_hdiv_vals, dtype=float)
    p_l2_vals = np.asarray(p_l2_vals, dtype=float)

    return {
        "operator_int": float(np.mean(relA)),
        "operator_max": float(np.max(relA)),
        "u_l2_int": float(np.mean(u_l2_vals)),
        "u_l2_max": float(np.max(u_l2_vals)),
        "u_hdiv_int": float(np.mean(u_hdiv_vals)),
        "u_hdiv_max": float(np.max(u_hdiv_vals)),
        "p_l2_int": float(np.mean(p_l2_vals)),
        "p_l2_max": float(np.max(p_l2_vals)),
    }

def plot_l2_fit_operator_errors(results, filename="l2_fit_errors.pdf"):
    P_values = np.asarray(results["P_values"], dtype=int)

    fig, ax = plt.subplots(figsize=(8, 5))

    h_op, = ax.plot(P_values, results["l2fit_operator_int"], marker="o", linestyle="-", label=r"$\|e_{\mathbf{A}}\|_F$")
    ax.plot(P_values, results["l2fit_operator_max"], marker="o", linestyle="--", color=h_op.get_color(), label="_nolegend_")

    h_p, = ax.plot(P_values, results["l2fit_p_l2_int"], marker="o", markersize=4.2, linestyle="-", zorder=2, label=r"$\|e_p\|_{L^2}$")
    ax.plot(P_values, results["l2fit_p_l2_max"], marker="o", linestyle="--", mfc="none", mec=h_p.get_color(), mew=1.3, color=h_p.get_color(), label="_nolegend_")

    h_ul2, = ax.plot(P_values, results["l2fit_u_l2_int"], marker="x", linestyle="-", label=r"$\|e_u\|_{L^2}$")
    ax.plot(P_values, results["l2fit_u_l2_max"], marker="x", linestyle="--", color=h_ul2.get_color(), label="_nolegend_")

    h_uhd, = ax.plot(P_values, results["l2fit_u_hdiv_int"], marker="s", linestyle="-", markerfacecolor="none", markeredgewidth=1.3, label=r"$\|e_u\|_{H(\mathrm{div})}$")
    ax.plot(P_values, results["l2fit_u_hdiv_max"], marker="s", linestyle="--", markerfacecolor="none", markeredgewidth=1.3, color=h_uhd.get_color(), label="_nolegend_")

    all_series = [
        np.asarray(results["l2fit_operator_int"], dtype=float),
        np.asarray(results["l2fit_operator_max"], dtype=float),
        np.asarray(results["l2fit_u_l2_int"], dtype=float),
        np.asarray(results["l2fit_u_l2_max"], dtype=float),
        np.asarray(results["l2fit_u_hdiv_int"], dtype=float),
        np.asarray(results["l2fit_u_hdiv_max"], dtype=float),
        np.asarray(results["l2fit_p_l2_int"], dtype=float),
        np.asarray(results["l2fit_p_l2_max"], dtype=float),
    ]
    y_max = max(float(np.max(arr)) for arr in all_series)
    y_min = min(float(np.min(arr[arr > 0])) for arr in all_series if np.any(arr > 0))

    ax.set_yscale("log")
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(P_values)
    ax.set_xlabel("Polynomial degree $P$")
    ax.set_ylabel("Relative error")
    ax.grid(True, which="both", ls="--", alpha=0.5)

    quantity_handles = [h_op, h_p, h_ul2, h_uhd]
    style_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.8, label="Integrated"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.8, label="Max"),
    ]
    leg1 = ax.legend(handles=quantity_handles, loc="upper right", title="")
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, loc="lower left", title="")

    fig.tight_layout(); fig.savefig(filename, dpi=300, bbox_inches="tight"); plt.close(fig)



# ============================================================
# Chosen-parameter comparison plots (FOM vs ROM for two taus)
# ============================================================
def _mesh_vertex_triangulation(mesh, deformation_gf=None):
    vertices = sorted(list(mesh.vertices), key=lambda v: v.nr)
    vx_ref = np.array([float(v.point[0]) for v in vertices], dtype=float)
    vy_ref = np.array([float(v.point[1]) for v in vertices], dtype=float)

    if deformation_gf is None:
        vx = vx_ref.copy()
        vy = vy_ref.copy()
    else:
        vx = np.empty_like(vx_ref)
        vy = np.empty_like(vy_ref)
        for i, (xv, yv) in enumerate(zip(vx_ref, vy_ref)):
            dxy = deformation_gf(mesh(xv, yv))
            vx[i] = xv + float(dxy[0])
            vy[i] = yv + float(dxy[1])

    triangles = []
    for el in mesh.Elements(VOL):
        ids = [v.nr for v in el.vertices]
        if len(ids) == 3:
            triangles.append(ids)
        elif len(ids) == 4:
            triangles.append([ids[0], ids[1], ids[2]])
            triangles.append([ids[0], ids[2], ids[3]])

    return vx, vy, np.array(triangles, dtype=int), vx_ref, vy_ref


def _sample_pressure_and_velocity_vertices(mesh, order, p_vec, u_vec, vx_ref, vy_ref):
    Q = L2(mesh, order=order - 1)
    V = HDiv(mesh, order=order)
    gp = GridFunction(Q)
    gu = GridFunction(V)
    gp.vec.FV().NumPy()[:] = np.asarray(p_vec, dtype=float)
    gu.vec.FV().NumPy()[:] = np.asarray(u_vec, dtype=float)

    p_vals = np.full(len(vx_ref), np.nan, dtype=float)
    ux_vals = np.full(len(vx_ref), np.nan, dtype=float)
    uy_vals = np.full(len(vx_ref), np.nan, dtype=float)
    umag_vals = np.full(len(vx_ref), np.nan, dtype=float)
    for i, (xv, yv) in enumerate(zip(vx_ref, vy_ref)):
        try:
            mip = mesh(float(xv), float(yv))
            p = gp(mip)
            u = gu(mip)
            ux = float(u[0])
            uy = float(u[1])
            p_vals[i] = float(p)
            ux_vals[i] = ux
            uy_vals[i] = uy
            umag_vals[i] = float(np.sqrt(ux * ux + uy * uy))
        except Exception:
            pass
    return p_vals, ux_vals, uy_vals, umag_vals

def _nearest_time_indices(times, times_plot):
    t = np.asarray(times, dtype=float)
    idx = []
    used = []
    for tr in times_plot:
        k = int(np.argmin(np.abs(t - float(tr))))
        idx.append(k)
        used.append(float(t[k]))
    return idx, used


def plot_transient_two_tau_field_comparison(
    experiment,
    mu_train,
    W_train,
    mu_plot,
    times_plot,
    tau_values=(1e-1, 1e-2),
    P=6,
    dt=0.1,
    T=1.0,
    store_every=1,
    t_cut=0.0,
    outdir="ROM_deformed_transient_plots",
):
    if len(tau_values) != 2:
        raise ValueError("tau_values must have length 2")

    tau_a, tau_b = float(tau_values[0]), float(tau_values[1])
    os.makedirs(outdir, exist_ok=True)

    fom = experiment.solve_full(tuple(mu_plot), dt=dt, T=T, store_every=store_every, draw=False)
    t_fom, Su_f, Sp_f = history_to_snapshot_matrices(fom["history"], fom["Nu"], fom["Np"])

    roms = {}
    for tau in (tau_a, tau_b):
        rom = TransientMixedDarcyL2FitROM_Blocked(experiment)
        rom_data = rom.fit(
            mu_train=mu_train,
            W_train=W_train,
            dt=dt,
            T=T,
            store_every=store_every,
            P=int(P),
            tau_u=tau,
            tau_p=tau,
            t_cut=t_cut,
        )
        roms[tau] = {"out": rom.solve_online(tuple(mu_plot), p0_value=0.0), "r_tot": int(rom_data["r_tot"]), "r_triplet": r_triplet_from_stabilized_dims(rom_data["r_u_stab"], rom_data["r_p"])}

    idx, times_used = _nearest_time_indices(t_fom, times_plot)

    r_a = format_r_triplet(roms[tau_a].get("r_triplet", None))
    r_b = format_r_triplet(roms[tau_b].get("r_triplet", None))
    labels = [
        "FOM",
        f"ROM (r={r_a})",
        f"ROM (r={r_b})",
        f"Diff (r={r_a})",
        f"Diff (r={r_b})",
    ]

    experiment.geom.apply(tuple(mu_plot))
    try:
        vx, vy, triangles, vx_ref, vy_ref = _mesh_vertex_triangulation(experiment.mesh, deformation_gf=experiment.geom.deform_gf)
        triang = mtri.Triangulation(vx, vy, triangles=triangles)

        for quantity in ("pressure", "velocity"):
            nrows, ncols = 5, len(idx)
            fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.8 * nrows), squeeze=False)
            fig.subplots_adjust(left=0.07, right=0.87, bottom=0.06, top=0.93, wspace=0.05, hspace=0.10)

            panel_data = []
            for col, k in enumerate(idx):
                p_f = Sp_f[:, k]
                u_f = Su_f[:, k]

                k_ra = min(k, len(roms[tau_a]["out"]["p_hist"]) - 1)
                k_rb = min(k, len(roms[tau_b]["out"]["p_hist"]) - 1)

                p_a = np.asarray(roms[tau_a]["out"]["p_hist"][k_ra], dtype=float)
                u_a = np.asarray(roms[tau_a]["out"]["u_hist"][k_ra], dtype=float)
                p_b = np.asarray(roms[tau_b]["out"]["p_hist"][k_rb], dtype=float)
                u_b = np.asarray(roms[tau_b]["out"]["u_hist"][k_rb], dtype=float)

                pF, uxF, uyF, mF = _sample_pressure_and_velocity_vertices(experiment.mesh, experiment.order, p_f, u_f, vx_ref, vy_ref)
                pA, uxA, uyA, mA = _sample_pressure_and_velocity_vertices(experiment.mesh, experiment.order, p_a, u_a, vx_ref, vy_ref)
                pB, uxB, uyB, mB = _sample_pressure_and_velocity_vertices(experiment.mesh, experiment.order, p_b, u_b, vx_ref, vy_ref)

                if quantity == "pressure":
                    rows = [
                        {"z": pF},
                        {"z": pA},
                        {"z": pB},
                        {"z": pA - pF},
                        {"z": pB - pF},
                    ]
                else:
                    rows = [
                        {"ux": uxF, "uy": uyF, "mag": mF},
                        {"ux": uxA, "uy": uyA, "mag": mA},
                        {"ux": uxB, "uy": uyB, "mag": mB},
                        {"ux": uxA - uxF, "uy": uyA - uyF, "mag": np.sqrt((uxA - uxF)**2 + (uyA - uyF)**2)},
                        {"ux": uxB - uxF, "uy": uyB - uyF, "mag": np.sqrt((uxB - uxF)**2 + (uyB - uyF)**2)},
                    ]
                panel_data.append(rows)

            if quantity == "pressure":
                main_vals = [panel_data[c][r]["z"] for c in range(ncols) for r in (0,1,2)]
                diff_vals = [panel_data[c][r]["z"] for c in range(ncols) for r in (3,4)]

                def _bounds(arrs):
                    vals = [a[np.isfinite(a)] for a in arrs if np.any(np.isfinite(a))]
                    if not vals:
                        return 0.0, 1.0
                    vv = np.concatenate(vals)
                    return float(np.min(vv)), float(np.max(vv))

                vmin_main, vmax_main = _bounds(main_vals)
                dmin, dmax = _bounds(diff_vals)
                dabs = max(abs(dmin), abs(dmax), 1e-14)

                row_mappables = [None] * nrows
                for r in range(nrows):
                    for c in range(ncols):
                        ax = axes[r, c]
                        z = panel_data[c][r]["z"]
                        if r <= 2:
                            m = ax.tripcolor(triang, z, shading='gouraud', cmap='viridis', vmin=vmin_main, vmax=vmax_main)
                        else:
                            m = ax.tripcolor(triang, z, shading='gouraud', cmap='RdBu', vmin=-dabs, vmax=dabs)
                        if row_mappables[r] is None:
                            row_mappables[r] = m

                        # No border around panels
                        for sp in ax.spines.values():
                            sp.set_visible(False)
                        ax.set_aspect('equal', adjustable='box')
                        if r == 0:
                            ax.set_title(f"t={times_used[c]:.2f}")
                        if c == 0:
                            ax.set_ylabel(labels[r])
                        ax.set_xticks([])
                        ax.set_yticks([])

                # One colorbar per row with fixed shared x-position (strict alignment)
                row_boxes = []
                for r in range(nrows):
                    bb_l = axes[r, 0].get_position()
                    bb_r = axes[r, -1].get_position()
                    row_boxes.append((bb_l.y0, bb_l.y1, bb_r.x1))
                cbar_x = max(b[2] for b in row_boxes) + 0.008
                cbar_w = 0.012
                for r in range(nrows):
                    y0, y1, _ = row_boxes[r]
                    cax = fig.add_axes([cbar_x, y0, cbar_w, y1 - y0])
                    cb = fig.colorbar(row_mappables[r], cax=cax)
                    cb.ax.tick_params(labelsize=14)

            else:
                # velocity quiver (length and color by magnitude)
                main_mags = [panel_data[c][r]["mag"] for c in range(ncols) for r in (0,1,2)]
                diff_mags = [panel_data[c][r]["mag"] for c in range(ncols) for r in (3,4)]

                def _maxmag(arrs):
                    vals = [a[np.isfinite(a)] for a in arrs if np.any(np.isfinite(a))]
                    if not vals:
                        return 1.0
                    return max(float(np.max(v)) for v in vals)

                vmax_main = max(_maxmag(main_mags), 1e-14)
                vmax_diff = max(_maxmag(diff_mags), 1e-14)

                from matplotlib.colors import Normalize
                from matplotlib.cm import ScalarMappable
                norm_main = Normalize(vmin=0.0, vmax=vmax_main)
                norm_diff = Normalize(vmin=0.0, vmax=vmax_diff)

                # Match rom_deformedv3.ipynb sampling density
                step = max(1, len(vx) // 120)
                ids = np.arange(0, len(vx), step)

                # Match rom_deformedv3.ipynb arrow scaling
                span = max(float(np.ptp(vx)), float(np.ptp(vy)), 1e-14)
                target_arrow = 0.18 * span
                scale_main = max(vmax_main / target_arrow, 1e-14)
                scale_diff = max(vmax_diff / target_arrow, 1e-14)

                row_mappables = [None] * nrows
                for r in range(nrows):
                    for c in range(ncols):
                        ax = axes[r, c]
                        ux = panel_data[c][r]["ux"]
                        uy = panel_data[c][r]["uy"]
                        mag = panel_data[c][r]["mag"]

                        valid = np.isfinite(ux) & np.isfinite(uy) & np.isfinite(mag)
                        ids_v = ids[valid[ids]]

                        if len(ids_v) > 0:
                            if r <= 2:
                                q = ax.quiver(
                                    vx[ids_v], vy[ids_v], ux[ids_v], uy[ids_v], mag[ids_v],
                                    cmap='viridis', norm=norm_main,
                                    angles='xy', scale_units='xy', scale=scale_main,
                                    pivot='mid', width=0.0025, headwidth=3.2, headlength=4.2,
                                )
                                if row_mappables[r] is None:
                                    row_mappables[r] = ScalarMappable(norm=norm_main, cmap='viridis')
                            else:
                                q = ax.quiver(
                                    vx[ids_v], vy[ids_v], ux[ids_v], uy[ids_v], mag[ids_v],
                                    cmap='RdBu', norm=norm_diff,
                                    angles='xy', scale_units='xy', scale=scale_diff,
                                    pivot='mid', width=0.0030, headwidth=3.5, headlength=4.5, minlength=0.0,
                                )
                                if row_mappables[r] is None:
                                    row_mappables[r] = ScalarMappable(norm=norm_diff, cmap='RdBu')
                        else:
                            if row_mappables[r] is None:
                                row_mappables[r] = ScalarMappable(norm=norm_main if r <= 2 else norm_diff, cmap='viridis' if r <= 2 else 'RdBu')

                        # No border around panels
                        for sp in ax.spines.values():
                            sp.set_visible(False)
                        ax.set_aspect('equal', adjustable='box')
                        if r == 0:
                            ax.set_title(f"t={times_used[c]:.2f}")
                        if c == 0:
                            ax.set_ylabel(labels[r])
                        ax.set_xticks([])
                        ax.set_yticks([])

                row_boxes = []
                for r in range(nrows):
                    bb_l = axes[r, 0].get_position()
                    bb_r = axes[r, -1].get_position()
                    row_boxes.append((bb_l.y0, bb_l.y1, bb_r.x1))
                cbar_x = max(b[2] for b in row_boxes) + 0.008
                cbar_w = 0.012
                for r in range(nrows):
                    y0, y1, _ = row_boxes[r]
                    cax = fig.add_axes([cbar_x, y0, cbar_w, y1 - y0])
                    cb = fig.colorbar(row_mappables[r], cax=cax)
                    cb.ax.tick_params(labelsize=14)

            # fig.suptitle(f"{quantity.capitalize()} comparison at mu=({mu_plot[0]:.2f},{mu_plot[1]:.2f}), P={int(P)}")
            # Keep fixed layout: tight_layout would move panels after manual colorbar placement
            out = os.path.join(outdir, f"transient_{quantity}_two_tau_comparison_P{int(P)}.pdf")
            fig.savefig(out, dpi=300, bbox_inches='tight')
            plt.close(fig)
    finally:
        experiment.geom.clear()


# Run convergence and save JSON bundle
mesh = Mesh(unit_square.GenerateMesh(maxh=0.16))
domain = ParameterDomain(mu1_range=(-0.3, 0.3), mu2_range=(-0.3, 0.3))
deformation = TopRightDeformation()
darcy_model = TransientDarcyModel(
    K=1.0,
    Sstor=1.0,
    T=1.0,
    source_amp=0.0,
    source_sigma=0.12,
    source_center=(0.7, 0.4),
    p_init=0.0,
)
experiment = TransientDarcyExperiment(
    mesh=mesh,
    order=2,
    darcy_model=darcy_model,
    domain=domain,
    deformation_model=deformation,
)

mu_train, W_train = sample_parameters(method="grid", N=16, domain=domain)

# Keep this modest by default; increase if desired.
P_values = list(range(10))
tau_values = [1e-1, 1e-2,1e-3,1e-4]
run_speed_test = True

dt = 0.005
T = 1.0

results = run_convergence_study_transient(
    experiment=experiment,
    domain=domain,
    P_values=P_values,
    tau_values=tau_values,
    mu_train=mu_train,
    W_train=W_train,
    n_test_random=256,
    random_seed=0,
    dt=dt,
    T=T,
    store_every=1,
    t_cut=0.0,
    measure_speed=run_speed_test,
    speed_repeats=10,
    operator_sampling_method="gauss_lobatto",
    operator_n_1d=16,
)

speed_results = speed_results_from_convergence(results) if run_speed_test else None

outdir_json = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
results_json_path = os.path.join(outdir_json, "rom_deformed_transient_results.json")
save_results_bundle_json(
    results_json_path,
    results=results,
    speed_results=speed_results,
    extra={
        "P_values": P_values,
        "tau_values": tau_values,
        "run_speed_test": run_speed_test,
        "dt": dt,
        "T": T,
    },
)
print("Saved results JSON:", results_json_path)



def _restore_numeric_keys(obj):
    if isinstance(obj, list):
        return [_restore_numeric_keys(v) for v in obj]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            nk = k
            if isinstance(k, str):
                try:
                    nk = float(k)
                except ValueError:
                    nk = k
            out[nk] = _restore_numeric_keys(v)
        return out
    return obj
# Load JSON bundle and plot
json_path = results_json_path
bundle = load_results_bundle_json(json_path)

results = _restore_numeric_keys(bundle.get("results", None))
speed_results = _restore_numeric_keys(bundle.get("speed_results", None))
extra = _restore_numeric_keys(bundle.get("extra", {}))

if results is None:
    raise ValueError(f"No 'results' found in {json_path}")

outdir = ""
outdir = "ROM_deformed_transient_plots"
if outdir:
    os.makedirs(outdir, exist_ok=True)

def outpath(name):
    return os.path.join(outdir, name) if outdir else name

plot_rom_velocity_errors(results, outpath("rom_velocity_errors.pdf"))
plot_rom_pressure_errors(results, outpath("rom_pressure_errors.pdf"))
plot_discarded_energy(results, outpath("discarded_energy.pdf"))
plot_l2_fit_operator_errors(results, outpath("l2_fit_errors.pdf"))

if speed_results is not None:
    plot_speedup_vs_polynomial_degree(speed_results, outpath("rom_speedup_vs_polynomial_degree.pdf"))

print("Plotted from JSON:", json_path)



# Field comparison plots (FOM vs ROM) for two taus, saved with other plots
# plot_transient_two_tau_field_comparison(
#     experiment=experiment,
#     mu_train=mu_train,
#     W_train=W_train,
#     mu_plot=(0.3, 0.3),
#     times_plot=[0.1, 0.5, 1.0],
#     tau_values=(1e-1, 1e-2),
#     P=3,
#     dt=dt,
#     T=T,
#     store_every=1,
#     t_cut=0.0,
#     outdir=outdir,
# )
print('Saved transient two-tau field comparison plots in:', outdir)






