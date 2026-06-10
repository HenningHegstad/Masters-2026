#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import time
import math
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Any, Optional, Sequence

import numpy as np
from numpy.polynomial.legendre import Legendre
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from netgen.geom2d import unit_square
from ngsolve import *
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg import lu_factor, lu_solve
import copy
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
    "legend.fontsize": 11,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
})
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
# ============================================================
# Logging
# ============================================================
T0 = time.time()


def log(msg: str):
    dt = time.time() - T0
    hh = int(dt // 3600)
    mm = int((dt % 3600) // 60)
    ss = int(dt % 60)
    print(f"[{hh:02d}:{mm:02d}:{ss:02d}] {msg}", flush=True)


# ============================================================
# Config
# ============================================================
@dataclass
class Config:
    # FE / mesh
    maxh: float = 0.08
    order: int = 2
    Gamma_N: str = "right|top|bottom"

    # time
    dt: float = 0.02#0.00125
    T_snap: float = 1.0
    T_test: float = 1.0
    error_sample_start_time: float = 0.0
    error_integration_order: int = 12
    Sstor: float = 1.0
    p_init: float = 0.0
    stationary_mode: bool = True

    # permeability
    k0: float = 1.0
    delta: float = 10.0
    w: float = 0.06

    # forcing
    f_amp: float = 10.0
    f_mu_x: float = 0.7
    f_mu_y: float = 0.4
    f_sigma: float = 0.1

    # parameter domain
    a_mu: float = 0.2
    b_mu: float = 0.8
    Nsplit: int = 1

    # training/test
    ntrain_1d: int = 48
    n_test: int = 256
    rng_seed: int = 0
    muu_ridge: float = 0
    # ROM options
    P_list: Tuple[int, ...] = (9,)#(0,1,2,3,4,5,6,7,8,9)
    tau_list: Tuple[float, ...] = (1e-2,1e-3)#(1e-2,1e-3,1e-4)

    # snapshot downsampling
    store_every: int = 1

    # norm used for final leaf POD / ROM construction
    #   "L2"       -> Euclidean POD for velocity
    #   "Kinv_ref" -> weighted velocity POD with leaf-local reference K^{-1}
    norm_mode: str = "Kinv_ref"


    # internal POD mode used by final leaf build
    # set automatically from norm_mode in main()
    pod_mode: str = "snap_weights"

    # parallel
    nproc_regions: int = 4
    parallel_mu_chunks_min_size: int = 64
    costa_parallel: bool = False

    # prints
    print_every_mu: int = 1024
    print_every_step: int = 1024

    # ramp
    use_ramp: bool = True
    t_ramp_end: float = 0.25
    ramp_target: float = 0.6321205588

    # output
    out_dir: str = "out"
    tag: str = "test"

    # direct full-order L2-fit error study for M_uu
    compute_muu_l2fit_errors: bool = False#True
    debug_use_exact_muu_in_direct_fit_solver: bool = False


    # number of Gauss-Legendre points per parameter direction
    # total number of integration points = l2fit_quad_n_1d^2
    l2fit_quad_n_1d: int = 16

    # threshold for detecting the sparse pattern of M_uu
    l2fit_pattern_tol: float = 0.0

        # DNN-CoSTA options
    # costa_use_dnn: bool = True
    # costa_hidden_width: int = 128
    # costa_hidden_depth: int = 2
    # costa_activation: str = "tanh"   # "tanh", "relu", "gelu"
    # costa_batch_size: int = 1024
    # costa_epochs: int = 500
    # costa_lr: float = 2e-4
    # costa_weight_decay: float = 0#1e-6
    # costa_val_fraction: float = 0.1
    # costa_early_stop_patience: int = 80
    # costa_min_epochs: int = 100
    # costa_use_mu_features: bool = False
    # costa_use_time_feature: bool = False
    # costa_device: str = "cpu"

    # DNN-CoSTA options (retuned for smaller dt / more time steps)
    costa_use_dnn: bool = True
    costa_mode: str = "ridge"  # "dnn" or "ridge"
    costa_hidden_width: int = 64
    costa_hidden_depth: int = 1
    costa_activation: str = "relu"      # faster than tanh
    costa_batch_size: int = 2048
    costa_epochs: int = 120
    costa_lr: float = 1e-3
    costa_weight_decay: float = 0.0
    costa_val_fraction: float = 0.05
    costa_early_stop_patience: int = 12
    costa_min_epochs: int = 20
    costa_use_mu_features: bool = False
    costa_use_time_feature: bool = False
    costa_device: str = "cpu"
    costa_ridge: float = 100


    # plotting defaults (used when corresponding CLI args are omitted)
    plot_P: Optional[int] = None
    plot_tau: Optional[float] = None
    plot_tau_pair: Optional[Tuple[float, float]] = None
    plot_mu_x: float = 0.3
    plot_mu_y: float = 0.7
    plot_times: Tuple[float, ...] = (0.1, 0.5, 1.0)

# ============================================================
# Direct full-order L2-fit of M_uu and error evaluation
# ============================================================
def regression_entries_rect(mu_train, Y, Pdeg, ax, bx, ay, by, ridge=0.0):
    """
    Least-squares fit for many scalar outputs at once.

    Parameters
    ----------
    mu_train : list[(mu1,mu2)]
    Y        : array of shape (Ns, nout)
               Each column is one scalar quantity to fit over parameter space.
    """
    Ns, nout = Y.shape
    nfeat = (Pdeg + 1) * (Pdeg + 1)

    B = np.array(
        [legendre_basis_rect(mu, Pdeg, ax, bx, ay, by) for mu in mu_train],
        dtype=np.float64
    )
    Y = np.asarray(Y, dtype=np.float64)

    if ridge > 0.0:
        sqrtlam = np.sqrt(ridge)
        B = np.vstack([B, sqrtlam * np.eye(nfeat)])
        Y = np.vstack([Y, np.zeros((nfeat, nout), dtype=np.float64)])

    C, *_ = np.linalg.lstsq(B, Y, rcond=None)
    # shape: (nfeat, nout)
    return C.T.copy()  # (nout, nfeat)


def build_leaf_full_muu_l2fit(cfg: Config, leaf: PseudoLeaf, mu_train: List[Tuple[float, float]]):
    """
    Build an L2-fit of the full-order M_uu block on one leaf, but only on its sparse pattern.
    This avoids storing a full dense matrix for every training sample.
    """
    if len(mu_train) == 0:
        return (leaf.leaf_id, {})

    mesh = Mesh(unit_square.GenerateMesh(maxh=cfg.maxh))
    V = HDiv(mesh, order=cfg.order, dirichlet=cfg.Gamma_N)
    Q = L2(mesh, order=cfg.order - 1)
    f_base = forcing_base_cf(cfg)

    ax, bx, ay, by = leaf.ax, leaf.bx, leaf.ay, leaf.by
    Ns = len(mu_train)

    # detect sparsity pattern from first sample
    M0_mat, _, _, _ = assemble_sparse_blocks(mu_train[0], V, Q, mesh, f_base, cfg.k0, cfg.delta, cfg.w)
    M0 = np.array(M0_mat.ToDense(), dtype=np.float64)

    rows, cols = np.where(np.abs(M0) > float(cfg.l2fit_pattern_tol))
    if rows.size == 0:
        raise RuntimeError(f"No nonzero pattern detected for leaf {leaf.leaf_id}")

    nnz = rows.size
    Y = np.empty((Ns, nnz), dtype=np.float64)
    Y[0, :] = M0[rows, cols]

    for i, mu in enumerate(mu_train[1:], start=1):
        M_mat, _, _, _ = assemble_sparse_blocks(mu, V, Q, mesh, f_base, cfg.k0, cfg.delta, cfg.w)
        Md = np.array(M_mat.ToDense(), dtype=np.float64)
        Y[i, :] = Md[rows, cols]

        if cfg.print_every_mu and ((i + 1) % max(1, cfg.print_every_mu) == 0):
            print(f"      leaf {leaf.leaf_id}: full M_uu fit sample {i+1}/{Ns}", flush=True)

    P_cap_leaf = max_admissible_poly_degree(Ns)
    fit_map = {}

    for Pdeg_req in cfg.P_list:
        Pdeg_eff = min(int(Pdeg_req), P_cap_leaf)

        coeffs = regression_entries_rect(
            mu_train=mu_train,
            Y=Y,
            Pdeg=Pdeg_eff,
            ax=ax, bx=bx, ay=ay, by=by,
            ridge=cfg.muu_ridge,
        )

        fit_map[int(Pdeg_req)] = {
            "leaf_id": leaf.leaf_id,
            "shape": tuple(M0.shape),
            "rows": rows.copy(),
            "cols": cols.copy(),
            "coeffs": coeffs,   # shape (nnz, nfeat)
            "Pdeg": int(Pdeg_eff),
            "ax": float(ax),
            "bx": float(bx),
            "ay": float(ay),
            "by": float(by),
            "nnz_pattern": int(nnz),
            "n_mu_train_leaf": int(Ns),
        }

    return (leaf.leaf_id, fit_map)


def build_all_leaf_full_muu_l2fits(
    cfg: Config,
    final_leaves: Dict[str, PseudoLeaf],
    mu_buckets: Dict[str, List[Tuple[float, float]]],
):
    """
    Optional offline stage:
    build full-order L2-fits of M_uu for every final leaf and every requested P.
    """
    out = {int(Pdeg_req): {} for Pdeg_req in cfg.P_list}

    for leaf_id, leaf in final_leaves.items():
        mu_train_leaf = mu_buckets[leaf_id]
        if len(mu_train_leaf) == 0:
            continue

        log(f"Building direct full-order M_uu L2-fit on leaf {leaf_id} with {len(mu_train_leaf)} samples ...")
        leaf_id_out, fit_map = build_leaf_full_muu_l2fit(cfg, leaf, mu_train_leaf)

        for Pdeg_req, fit_obj in fit_map.items():
            out[int(Pdeg_req)][leaf_id_out] = fit_obj

    return out


def eval_sparse_matrix_from_fit_rect(fit_obj, mu, symmetrize=True):
    theta = legendre_basis_rect(
        mu, fit_obj["Pdeg"],
        fit_obj["ax"], fit_obj["bx"], fit_obj["ay"], fit_obj["by"]
    )

    vals = fit_obj["coeffs"] @ theta
    M = np.zeros(fit_obj["shape"], dtype=np.float64)
    M[fit_obj["rows"], fit_obj["cols"]] = vals

    if symmetrize:
        M = 0.5 * (M + M.T)

    return M


def build_parameter_quadrature_square(cfg: Config):
    """
    Tensor-product Gauss-Legendre quadrature on [a_mu, b_mu]^2.
    """
    n1d = int(cfg.l2fit_quad_n_1d)
    if n1d < 1:
        raise ValueError(f"l2fit_quad_n_1d must be >= 1, got {n1d}")

    x1d, w1d = np.polynomial.legendre.leggauss(n1d)

    a = float(cfg.a_mu)
    b = float(cfg.b_mu)

    x_phys = 0.5 * (b - a) * x1d + 0.5 * (a + b)
    w_phys = 0.5 * (b - a) * w1d

    quad = []
    for i in range(n1d):
        for j in range(n1d):
            mu = (float(x_phys[i]), float(x_phys[j]))
            w = float(w_phys[i] * w_phys[j])
            quad.append((mu, w))

    return quad


def classify_mu_in_partition_payload(mu, partition_payload: dict, cfg: Config) -> str:
    return classify_mu_in_rect_partition_dict(mu, partition_payload["leaves"])

def compute_history_solution_error_sums(
    hu_fit: np.ndarray,
    hp_fit: np.ndarray,
    hu_fom: np.ndarray,
    hp_fom: np.ndarray,
    eval_ctx: dict,
    cfg: Config,
    mu,
    quad_weight: float,
):
    """
    Accumulate squared-in-time, squared-in-parameter error/solution norms
    for one parameter sample, and also track max-in-time relative errors
    for this mu.

    Returns numerator/denominator sums for:
      - u L2 relative error
      - p L2 relative error
      - u Hdiv relative error
      - u K^{-1}-Hdiv relative error

    Also returns:
      - max_t relative errors for this mu
      - max_t absolute errors for this mu
    """
    mesh = eval_ctx["mesh"]
    V = eval_ctx["V"]
    Q = eval_ctx["Q"]

    Kinv_cf = 1.0 / permeability_cf(mu, cfg)

    sums = {
        "u_l2_num": 0.0,
        "u_l2_den": 0.0,
        "p_l2_num": 0.0,
        "p_l2_den": 0.0,
        "u_hdiv_num": 0.0,
        "u_hdiv_den": 0.0,
        "u_khdiv_num": 0.0,
        "u_khdiv_den": 0.0,

        # max-in-time relative errors for this mu
        "u_l2_rel_maxt_mu": 0.0,
        "p_l2_rel_maxt_mu": 0.0,
        "u_hdiv_rel_maxt_mu": 0.0,
        "u_khdiv_rel_maxt_mu": 0.0,

        # max-in-time absolute errors for this mu
        "u_l2_abs_maxt_mu": 0.0,
        "p_l2_abs_maxt_mu": 0.0,
        "u_hdiv_abs_maxt_mu": 0.0,
        "u_khdiv_abs_maxt_mu": 0.0,
    }

    nsteps = hu_fom.shape[0]
    eps = 1e-14

    t_start = max(0.0, float(getattr(cfg, "error_sample_start_time", 0.0)))

    for n in range(nsteps):
        t_n = (n + 1) * float(cfg.dt)
        if t_n < t_start:
            continue

        u_fit = hu_fit[n, :]
        p_fit = hp_fit[n, :]

        u_fom = hu_fom[n, :]
        p_fom = hp_fom[n, :]

        int_order = int(cfg.error_integration_order)

        eu_l2 = abs_L2_error_vector(u_fit, u_fom, V, mesh, order=int_order)
        nu_l2 = L2_norm_vector_on_mesh(u_fom, V, mesh, order=int_order)

        ep_l2 = abs_L2_error_scalar(p_fit, p_fom, Q, mesh, order=int_order)
        np_l2 = L2_norm_scalar_on_mesh(p_fom, Q, mesh, order=int_order)

        eu_hdiv = abs_Hdiv_error_vector(u_fit, u_fom, V, mesh, order=int_order)
        nu_hdiv = Hdiv_norm_vector_on_mesh(u_fom, V, mesh, order=int_order)

        eu_khdiv = abs_KHdiv_error_vector(u_fit, u_fom, V, mesh, Kinv_cf, order=int_order)
        nu_khdiv = KHdiv_norm_vector_on_mesh(u_fom, V, mesh, Kinv_cf, order=int_order)

        ru_l2 = eu_l2 / max(nu_l2, eps)
        rp_l2 = ep_l2 / max(np_l2, eps)
        ru_hdiv = eu_hdiv / max(nu_hdiv, eps)
        ru_khdiv = eu_khdiv / max(nu_khdiv, eps)

        sums["u_l2_num"] += quad_weight * eu_l2**2
        sums["u_l2_den"] += quad_weight * nu_l2**2

        sums["p_l2_num"] += quad_weight * ep_l2**2
        sums["p_l2_den"] += quad_weight * np_l2**2

        sums["u_hdiv_num"] += quad_weight * eu_hdiv**2
        sums["u_hdiv_den"] += quad_weight * nu_hdiv**2

        sums["u_khdiv_num"] += quad_weight * eu_khdiv**2
        sums["u_khdiv_den"] += quad_weight * nu_khdiv**2

        sums["u_l2_rel_maxt_mu"] = max(sums["u_l2_rel_maxt_mu"], float(ru_l2))
        sums["p_l2_rel_maxt_mu"] = max(sums["p_l2_rel_maxt_mu"], float(rp_l2))
        sums["u_hdiv_rel_maxt_mu"] = max(sums["u_hdiv_rel_maxt_mu"], float(ru_hdiv))
        sums["u_khdiv_rel_maxt_mu"] = max(sums["u_khdiv_rel_maxt_mu"], float(ru_khdiv))

        sums["u_l2_abs_maxt_mu"] = max(sums["u_l2_abs_maxt_mu"], float(eu_l2))
        sums["p_l2_abs_maxt_mu"] = max(sums["p_l2_abs_maxt_mu"], float(ep_l2))
        sums["u_hdiv_abs_maxt_mu"] = max(sums["u_hdiv_abs_maxt_mu"], float(eu_hdiv))
        sums["u_khdiv_abs_maxt_mu"] = max(sums["u_khdiv_abs_maxt_mu"], float(eu_khdiv))

    return sums


def build_full_fit_eval_context(cfg: Config):
    mesh = Mesh(unit_square.GenerateMesh(maxh=cfg.maxh))
    V = HDiv(mesh, order=cfg.order, dirichlet=cfg.Gamma_N)
    Q = L2(mesh, order=cfg.order - 1)
    Y = FESpace([V, Q])

    f_base = forcing_base_cf(cfg)

    mu_mid = (0.5 * (cfg.a_mu + cfg.b_mu), 0.5 * (cfg.a_mu + cfg.b_mu))
    _, B_mat, Mp_mat, fq_vec = assemble_sparse_blocks(
        mu_mid, V, Q, mesh, f_base, cfg.k0, cfg.delta, cfg.w
    )

    B_dense = np.array(B_mat.ToDense(), dtype=np.float64)
    Mp_dense = np.array(Mp_mat.ToDense(), dtype=np.float64)
    fq_np = fq_vec.FV().NumPy().copy().astype(np.float64)

    g1 = GridFunction(Q)
    g1.Set(CoefficientFunction(1.0))
    p1_full = g1.vec.FV().NumPy().copy().astype(np.float64)

    return {
        "mesh": mesh,
        "V": V,
        "Q": Q,
        "Y": Y,
        "f_base": f_base,
        "B_dense": B_dense,
        "Mp_dense": Mp_dense,
        "fq_np": fq_np,
        "p1_full": p1_full,
        "Nu": int(V.ndof),
        "Np": int(Q.ndof),
    }

def solve_fom_history_with_fitted_muu_ngsolve(eval_ctx: dict, cfg: Config, Muu_fit_dense: np.ndarray):
    """
    Solve the full mixed system in time using a fitted full-order M_uu block.

    This version builds the full dense block matrix, but restricts the solve
    to Y.FreeDofs(), so it matches the algebraic treatment of the FOM much
    better than solving on all dofs.
    """
    mesh = eval_ctx["mesh"]
    V = eval_ctx["V"]
    Q = eval_ctx["Q"]
    Y = eval_ctx["Y"]
    f_base = eval_ctx["f_base"]

    Nu = V.ndof
    Np = Q.ndof
    Ny = Nu + Np

    (uY, pY) = Y.TrialFunction()
    (vY, qY) = Y.TestFunction()

    S_cf = CoefficientFunction(float(cfg.Sstor))

    # Assemble the parameter-independent part of the mixed operator
    a_rest = BilinearForm(Y, symmetric=True)
    a_rest += (-pY * div(vY) + qY * div(uY) + (S_cf / cfg.dt) * pY * qY) * dx
    a_rest.Assemble()

    A_rest_dense = np.array(a_rest.mat.ToDense(), dtype=np.float64)

    # Insert fitted velocity block into the full mixed matrix
    A_dense = A_rest_dense.copy()
    A_dense[:Nu, :Nu] += Muu_fit_dense

    # Free dofs of the product space
    freedofs_ba = Y.FreeDofs()
    try:
        freedofs = np.array(freedofs_ba.NumPy(), dtype=bool)
    except Exception:
        freedofs = np.array([bool(freedofs_ba[i]) for i in range(Ny)], dtype=bool)

    if freedofs.shape[0] != Ny:
        raise RuntimeError(f"Unexpected FreeDofs size: got {freedofs.shape[0]}, expected {Ny}")

    free_idx = np.where(freedofs)[0]
    if free_idx.size == 0:
        raise RuntimeError("No free dofs found in Y.FreeDofs()")

    A_ff = A_dense[np.ix_(free_idx, free_idx)]
    A_ff_lu = lu_factor(A_ff, check_finite=False)

    p_old = GridFunction(Q)
    p_old.Set(CoefficientFunction(float(cfg.p_init)))

    gfu = GridFunction(Y)
    uh, ph = gfu.components

    nsteps = int(np.round(cfg.T_test / cfg.dt))

    hu = np.empty((nsteps, Nu), dtype=np.float32)
    hp = np.empty((nsteps, Np), dtype=np.float32)

    for nstep in range(1, nsteps + 1):
        t = nstep * cfg.dt
        f_cf_t = ramp_value(t, cfg) * f_base

        L = LinearForm(Y)
        L += (f_cf_t * qY + (S_cf / cfg.dt) * p_old * qY) * dx
        L.Assemble()

        rhs_full = L.vec.FV().NumPy().copy().astype(np.float64, copy=False)
        rhs_f = rhs_full[free_idx]

        sol_f = lu_solve(A_ff_lu, rhs_f, check_finite=False)

        sol_full = np.zeros(Ny, dtype=np.float64)
        sol_full[free_idx] = sol_f

        gfu.vec.FV().NumPy()[:] = sol_full
        p_old.vec.data = ph.vec

        hu[nstep - 1, :] = uh.vec.FV().NumPy().astype(np.float32, copy=False)
        hp[nstep - 1, :] = ph.vec.FV().NumPy().astype(np.float32, copy=False)

    return hu, hp

def compute_direct_muu_l2fit_errors(
    cfg: Config,
    partition_payload: dict,
    full_muu_fit_by_P: Dict[int, Dict[str, dict]],
):
    """
    Compute:
      1) relative Frobenius fit error for the fitted M_uu block,
         normalized by the Frobenius norm of the full mixed operator
      2) relative solution errors induced by solving the full system with fitted M_uu

    Improvements over the original version:
      - exact FOM histories are cached once per quadrature point
      - evaluation is parallelized over requested P values
    """
    import multiprocessing as mp

    quad = build_parameter_quadrature_square(cfg)

    # Build one eval context in the main process only for the FOM cache
    eval_ctx_main = build_full_fit_eval_context(cfg)

    log(f"Building quadrature FOM cache on {len(quad)} parameter points ...")
    t_cache0 = time.time()
    fom_cache = build_quad_fom_cache(eval_ctx_main, cfg, quad)
    log(f"Built quadrature FOM cache in {time.time() - t_cache0:.2f}s")

    out = {}

    P_list_int = [int(Pdeg_req) for Pdeg_req in cfg.P_list]

    # Serial path for single P
    if len(P_list_int) == 1:
        Pkey = P_list_int[0]
        out[str(Pkey)] = compute_direct_muu_l2fit_errors_one_P(
            Pkey=Pkey,
            cfg=cfg,
            partition_payload=partition_payload,
            full_muu_fit_by_P_local=full_muu_fit_by_P,
            eval_ctx=eval_ctx_main,
            quad=quad,
            fom_cache=fom_cache,
        )
        return out

    # Parallel over P
    cfg_dict = asdict(cfg)
    worker_args = [
        (
            cfg_dict,
            int(Pkey),
            partition_payload,
            full_muu_fit_by_P[int(Pkey)],
            quad,
            fom_cache,
        )
        for Pkey in P_list_int
    ]

    nproc = min(len(worker_args), max(1, int(cfg.nproc_regions)))
    ctx = mp.get_context("spawn")

    with ctx.Pool(processes=nproc) as pool:
        for Pstr, res in pool.imap_unordered(worker_direct_muu_l2fit_one_P, worker_args, chunksize=1):
            out[Pstr] = res

    return out
# ============================================================
# Partition nodes
# ============================================================

@dataclass
class PseudoLeaf:
    leaf_id: str
    ax: float
    bx: float
    ay: float
    by: float
    depth: int

    def as_dict(self):
        return asdict(self)
@dataclass
class RectLeaf:
    leaf_id: str
    ax: float
    bx: float
    ay: float
    by: float
    depth: int
    mu_train: List[Tuple[float, float]]

    def as_dict(self):
        return {
            "leaf_id": self.leaf_id,
            "ax": float(self.ax),
            "bx": float(self.bx),
            "ay": float(self.ay),
            "by": float(self.by),
            "depth": int(self.depth),
            "mu_train": [[float(a), float(b)] for a, b in self.mu_train],
        }
def build_uniform_rect_partition(cfg: Config, mu_train_all: List[Tuple[float, float]]):
    N = int(cfg.Nsplit)
    if N < 1:
        raise ValueError(f"Nsplit must be >= 1, got {N}")

    xs = np.linspace(cfg.a_mu, cfg.b_mu, N + 1)
    ys = np.linspace(cfg.a_mu, cfg.b_mu, N + 1)

    buckets = {(i, j): [] for i in range(N) for j in range(N)}

    for mu in mu_train_all:
        mx, my = float(mu[0]), float(mu[1])

        ix = min(N - 1, max(0, int(np.searchsorted(xs, mx, side="right") - 1)))
        iy = min(N - 1, max(0, int(np.searchsorted(ys, my, side="right") - 1)))

        buckets[(ix, iy)].append((mx, my))

    leaves = {}
    for i in range(N):
        for j in range(N):
            leaf_id = f"R{i}_{j}"
            leaves[leaf_id] = RectLeaf(
                leaf_id=leaf_id,
                ax=float(xs[i]),
                bx=float(xs[i + 1]),
                ay=float(ys[j]),
                by=float(ys[j + 1]),
                depth=0,
                mu_train=buckets[(i, j)],
            )

    return {"mode": "rect", "leaves": leaves}
def make_partition_payload_rect(rect_obj):
    return {
        "mode": "rect",
        "leaves": {leaf_id: leaf.as_dict() for leaf_id, leaf in rect_obj["leaves"].items()},
        "leaf_ids": list(rect_obj["leaves"].keys()),
        "n_leaves": int(len(rect_obj["leaves"])),
    }

def classify_mu_in_rect_partition_dict(mu, leaves_dict: Dict[str, dict]) -> str:
    mx, my = float(mu[0]), float(mu[1])

    for leaf_id, leaf in leaves_dict.items():
        ax = float(leaf["ax"])
        bx = float(leaf["bx"])
        ay = float(leaf["ay"])
        by = float(leaf["by"])

        in_x = (ax <= mx < bx) or np.isclose(mx, bx)
        in_y = (ay <= my < by) or np.isclose(my, by)

        if in_x and in_y:
            return leaf_id

    raise RuntimeError(f"Could not classify mu={mu} in rectangular partition")
# ============================================================
# Small helpers
# ============================================================
def chunk_list(seq, nchunks):
    nchunks = max(1, int(nchunks))
    n = len(seq)
    if n == 0:
        return []
    out = []
    start = 0
    for k in range(nchunks):
        stop = start + (n + k) // nchunks
        if stop > start:
            out.append(seq[start:stop])
        start = stop
    return out


def safe_inverse(mat, freedofs):
    for invtype in ["pardiso", "sparsecholesky", "umfpack"]:
        try:
            return mat.Inverse(freedofs, inverse=invtype)
        except Exception:
            pass
    return mat.Inverse(freedofs)



def max_admissible_poly_degree(n_points: int) -> int:
    """
    Largest P such that (P+1)^2 <= n_points.
    Returns at least 0.
    """
    if n_points <= 0:
        return 0
    return max(0, int(math.floor(math.sqrt(n_points))) - 1)


def cap_poly_degree(P_requested: int, n_points: int) -> int:
    return min(int(P_requested), max_admissible_poly_degree(n_points))





# ============================================================
# Ramp helpers
# ============================================================
def ramp_tau(cfg: Config) -> float:
    if (not cfg.use_ramp) or cfg.t_ramp_end <= 0:
        return 0.0
    target = min(float(cfg.ramp_target), 1.0 - 1e-15)
    return -float(cfg.t_ramp_end) / float(np.log(1.0 - target))


def ramp_value(t: float, cfg: Config) -> float:
    if not cfg.use_ramp:
        return 1.0
    t = float(t)
    if t <= 0.0:
        return 0.0
    tau = ramp_tau(cfg)
    if tau <= 0.0:
        return 1.0
    return 1.0 - float(np.exp(-t / tau))


def ramp_value_model(t: float, model: dict) -> float:
    if not model.get("use_ramp", False):
        return 1.0
    t = float(t)
    if t <= 0.0:
        return 0.0
    t_end = float(model.get("t_ramp_end", 0.0))
    target = min(float(model.get("ramp_target", 0.99)), 1.0 - 1e-15)
    if t_end <= 0.0:
        return 1.0
    tau = -t_end / np.log(1.0 - target)
    return 1.0 - np.exp(-t / tau)


# ============================================================
# PDE / assembly helpers
# ============================================================
def forcing_base_cf(cfg: Config):
    return cfg.f_amp * exp(-((x - cfg.f_mu_x) ** 2 + (y - cfg.f_mu_y) ** 2) / (2 * cfg.f_sigma ** 2))


def permeability_cf(mu, cfg: Config):
    mu_x, mu_y = float(mu[0]), float(mu[1])
    return cfg.k0 + cfg.delta * exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * cfg.w ** 2))


def assemble_sparse_blocks(mu, V, Q, mesh, f_base, k0, delta, w):
    mu_x, mu_y = mu
    K = k0 + delta * exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * w ** 2))
    Kinv = 1.0 / K

    uV, vV = V.TrialFunction(), V.TestFunction()
    pQ, qQ = Q.TrialFunction(), Q.TestFunction()

    bfM = BilinearForm(V, symmetric=True)
    bfM += (Kinv * InnerProduct(uV, vV)) * dx
    bfM.Assemble()

    bfB = BilinearForm(trialspace=V, testspace=Q)
    bfB += (qQ * div(uV)) * dx
    bfB.Assemble()

    bfMp = BilinearForm(Q, symmetric=True)
    bfMp += (pQ * qQ) * dx
    bfMp.Assemble()

    lf = LinearForm(Q)
    lf += (f_base * qQ) * dx
    lf.Assemble()

    return bfM.mat, bfB.mat, bfMp.mat, lf.vec


# ============================================================
# Regression helpers
# ============================================================
def scale_mu_rect(mu, ax, bx, ay, by):
    return (2.0 * (mu[0] - ax) / (bx - ax) - 1.0, 2.0 * (mu[1] - ay) / (by - ay) - 1.0)


def legendre_basis_rect(mu, P, ax, bx, ay, by):
    mu1, mu2 = scale_mu_rect(mu, ax, bx, ay, by)
    Lx = [Legendre.basis(i)(mu1) for i in range(P + 1)]
    Ly = [Legendre.basis(j)(mu2) for j in range(P + 1)]
    return np.array([lx * ly for lx in Lx for ly in Ly], dtype=float)


def regression_matrix_entries_rect(mu_train, Mats, Pdeg, ax, bx, ay, by, ridge=0.0):
    Ns, m, _ = Mats.shape
    nfeat = (Pdeg + 1) * (Pdeg + 1)

    B = np.array([legendre_basis_rect(mu, Pdeg, ax, bx, ay, by) for mu in mu_train], dtype=np.float64)
    Y = Mats.reshape(Ns, -1).astype(np.float64, copy=False)

    if ridge > 0:
        sqrtlam = np.sqrt(ridge)
        B = np.vstack([B, sqrtlam * np.eye(nfeat)])
        Y = np.vstack([Y, np.zeros((nfeat, Y.shape[1]))])

    C, *_ = np.linalg.lstsq(B, Y, rcond=None)
    return C.T.reshape(m, m, -1)


def eval_matrix_from_coeffs_rect(coeffs, mu, Pdeg, ax, bx, ay, by, symmetrize=True):
    theta = legendre_basis_rect(mu, Pdeg, ax, bx, ay, by)
    m = coeffs.shape[0]
    M = (coeffs.reshape(m * m, -1) @ theta).reshape(m, m)
    if symmetrize:
        M = 0.5 * (M + M.T)
    return M


def build_quad_fom_cache(eval_ctx: dict, cfg: Config, quad):
    """
    Cache exact FOM histories once for each quadrature point.

    Returns
    -------
    cache : dict
        cache[tuple(mu)] = (hu_fom, hp_fom)
    """
    cache = {}

    for iq, (mu, _wq) in enumerate(quad, start=1):
        hu_fom, hp_fom = solve_fom_history_for_test_full(
            eval_ctx["mesh"],
            eval_ctx["V"],
            eval_ctx["Q"],
            eval_ctx["Y"],
            eval_ctx["f_base"],
            cfg,
            mu,
        )
        cache[tuple(mu)] = (hu_fom, hp_fom)

        if cfg.print_every_mu and ((iq % max(1, cfg.print_every_mu)) == 0 or iq == len(quad)):
            print(f"      FOM quadrature cache: point {iq}/{len(quad)}", flush=True)

    return cache


def compute_direct_muu_l2fit_errors_one_P(
    Pkey: int,
    cfg: Config,
    partition_payload: dict,
    full_muu_fit_by_P_local: Dict[int, Dict[str, dict]],
    eval_ctx: dict,
    quad,
    fom_cache: Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]],
):
    """
    Evaluate direct full-order M_uu L2-fit errors for a single requested P.

    Report both:
      1) global RMS-type relative errors (existing behavior)
      2) maxima:
         - max operator-fit relative error over quadrature points
         - max solution relative/absolute errors over time and quadrature points
    """
    log(f"Direct full-order M_uu L2-fit error evaluation for P={Pkey} on {len(quad)} quadrature points ...")

    op_num = 0.0
    op_den = 0.0

    # maximum operator relative fit error over parameter quadrature
    op_rel_max = 0.0

    sol_acc = {
        "u_l2_num": 0.0, "u_l2_den": 0.0,
        "p_l2_num": 0.0, "p_l2_den": 0.0,
        "u_hdiv_num": 0.0, "u_hdiv_den": 0.0,
        "u_khdiv_num": 0.0, "u_khdiv_den": 0.0,
    }

    # max over all mu and time
    sol_max = {
        "u_l2_rel_maxt_allmu": 0.0,
        "p_l2_rel_maxt_allmu": 0.0,
        "u_hdiv_rel_maxt_allmu": 0.0,
        "u_khdiv_rel_maxt_allmu": 0.0,

        "u_l2_abs_maxt_allmu": 0.0,
        "p_l2_abs_maxt_allmu": 0.0,
        "u_hdiv_abs_maxt_allmu": 0.0,
        "u_khdiv_abs_maxt_allmu": 0.0,
    }

    Nu = eval_ctx["Nu"]
    Np = eval_ctx["Np"]
    B_dense = eval_ctx["B_dense"]
    Mp_dense = eval_ctx["Mp_dense"]

    for iq, (mu, wq) in enumerate(quad, start=1):
        leaf_id = classify_mu_in_partition_payload(mu, partition_payload, cfg)
        fit_obj = full_muu_fit_by_P_local[Pkey][leaf_id]

        Muu_exact_mat, _, _, _ = assemble_sparse_blocks(
            mu,
            eval_ctx["V"],
            eval_ctx["Q"],
            eval_ctx["mesh"],
            eval_ctx["f_base"],
            cfg.k0,
            cfg.delta,
            cfg.w,
        )
        Muu_exact = np.array(Muu_exact_mat.ToDense(), dtype=np.float64)

        if getattr(cfg, "debug_use_exact_muu_in_direct_fit_solver", False):
            Muu_fit = Muu_exact.copy()
        else:
            Muu_fit = eval_sparse_matrix_from_fit_rect(fit_obj, mu, symmetrize=True)

        # Numerator: only the fitted-block error
        D = Muu_fit - Muu_exact
        D_fro_sq = float(np.sum(D * D))
        op_num += wq * D_fro_sq

        # Denominator: Frobenius norm of the full mixed operator
        A_exact = np.zeros((Nu + Np, Nu + Np), dtype=np.float64)
        A_exact[:Nu, :Nu] = Muu_exact
        A_exact[:Nu, Nu:] = -B_dense.T
        A_exact[Nu:, :Nu] = B_dense
        A_exact[Nu:, Nu:] = (cfg.Sstor / cfg.dt) * Mp_dense

        A_fro_sq = float(np.sum(A_exact * A_exact))
        op_den += wq * A_fro_sq

        op_rel_mu = np.sqrt(D_fro_sq / max(A_fro_sq, 1e-30))
        op_rel_max = max(op_rel_max, float(op_rel_mu))

        # Reuse cached exact FOM history
        hu_fom, hp_fom = fom_cache[tuple(mu)]

        hu_fit, hp_fit = solve_fom_history_with_fitted_muu_ngsolve(
            eval_ctx,
            cfg,
            Muu_fit,
        )

        part = compute_history_solution_error_sums(
            hu_fit,
            hp_fit,
            hu_fom,
            hp_fom,
            eval_ctx,
            cfg,
            mu,
            wq,
        )

        for k in ("u_l2_num", "u_l2_den", "p_l2_num", "p_l2_den",
                  "u_hdiv_num", "u_hdiv_den", "u_khdiv_num", "u_khdiv_den"):
            sol_acc[k] += part[k]

        sol_max["u_l2_rel_maxt_allmu"] = max(sol_max["u_l2_rel_maxt_allmu"], part["u_l2_rel_maxt_mu"])
        sol_max["p_l2_rel_maxt_allmu"] = max(sol_max["p_l2_rel_maxt_allmu"], part["p_l2_rel_maxt_mu"])
        sol_max["u_hdiv_rel_maxt_allmu"] = max(sol_max["u_hdiv_rel_maxt_allmu"], part["u_hdiv_rel_maxt_mu"])
        sol_max["u_khdiv_rel_maxt_allmu"] = max(sol_max["u_khdiv_rel_maxt_allmu"], part["u_khdiv_rel_maxt_mu"])

        sol_max["u_l2_abs_maxt_allmu"] = max(sol_max["u_l2_abs_maxt_allmu"], part["u_l2_abs_maxt_mu"])
        sol_max["p_l2_abs_maxt_allmu"] = max(sol_max["p_l2_abs_maxt_allmu"], part["p_l2_abs_maxt_mu"])
        sol_max["u_hdiv_abs_maxt_allmu"] = max(sol_max["u_hdiv_abs_maxt_allmu"], part["u_hdiv_abs_maxt_mu"])
        sol_max["u_khdiv_abs_maxt_allmu"] = max(sol_max["u_khdiv_abs_maxt_allmu"], part["u_khdiv_abs_maxt_mu"])

        if cfg.print_every_mu and ((iq % max(1, cfg.print_every_mu)) == 0 or iq == len(quad)):
            print(f"      direct L2-fit eval P={Pkey}: quad point {iq}/{len(quad)}", flush=True)

    return {
        "quad_n_1d": int(cfg.l2fit_quad_n_1d),
        "n_quad_points": int(len(quad)),

        # global RMS-type relative errors (existing outputs)
        "muu_over_fullA_fro_rel": float(np.sqrt(op_num / max(op_den, 1e-30))),
        "u_l2_rel": float(np.sqrt(sol_acc["u_l2_num"] / max(sol_acc["u_l2_den"], 1e-30))),
        "p_l2_rel": float(np.sqrt(sol_acc["p_l2_num"] / max(sol_acc["p_l2_den"], 1e-30))),
        "u_hdiv_rel": float(np.sqrt(sol_acc["u_hdiv_num"] / max(sol_acc["u_hdiv_den"], 1e-30))),
        "u_khdiv_rel": float(np.sqrt(sol_acc["u_khdiv_num"] / max(sol_acc["u_khdiv_den"], 1e-30))),

        # new maxima
        "muu_over_fullA_fro_rel_maxmu": float(op_rel_max),

        "u_l2_rel_maxt_maxmu": float(sol_max["u_l2_rel_maxt_allmu"]),
        "p_l2_rel_maxt_maxmu": float(sol_max["p_l2_rel_maxt_allmu"]),
        "u_hdiv_rel_maxt_maxmu": float(sol_max["u_hdiv_rel_maxt_allmu"]),
        "u_khdiv_rel_maxt_maxmu": float(sol_max["u_khdiv_rel_maxt_allmu"]),

        "u_l2_abs_maxt_maxmu": float(sol_max["u_l2_abs_maxt_allmu"]),
        "p_l2_abs_maxt_maxmu": float(sol_max["p_l2_abs_maxt_allmu"]),
        "u_hdiv_abs_maxt_maxmu": float(sol_max["u_hdiv_abs_maxt_allmu"]),
        "u_khdiv_abs_maxt_maxmu": float(sol_max["u_khdiv_abs_maxt_allmu"]),
    }

def worker_direct_muu_l2fit_one_P(args):
    """
    Worker for one polynomial degree P.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    cfg_dict, Pkey, partition_payload, fit_map_P, quad, fom_cache = args
    cfg = Config(**cfg_dict)

    eval_ctx = build_full_fit_eval_context(cfg)

    # reconstruct the expected shape:
    # {Pkey: {leaf_id: fit_obj}}
    full_muu_fit_by_P_local = {int(Pkey): fit_map_P}

    res = compute_direct_muu_l2fit_errors_one_P(
        Pkey=int(Pkey),
        cfg=cfg,
        partition_payload=partition_payload,
        full_muu_fit_by_P_local=full_muu_fit_by_P_local,
        eval_ctx=eval_ctx,
        quad=quad,
        fom_cache=fom_cache,
    )

    return str(int(Pkey)), res

# ============================================================
# POD helpers
# ============================================================
def orthonormalize_columns(A, tol=1e-12):
    Qm, Rm = np.linalg.qr(A)
    d = np.abs(np.diag(Rm))
    keep = np.where(d > tol)[0]
    return Qm[:, keep]


def pick_r_from_svals(svals, tau):
    svals = np.asarray(svals, dtype=float)
    n = len(svals)
    if n <= 1:
        return max(1, n)
    s2 = svals ** 2
    tot = float(np.sum(s2))
    if tot <= 0:
        return 1
    tail = np.sqrt(np.cumsum(s2[::-1])[::-1] / tot)[1:]
    idx = np.where(tail < float(tau))[0]
    return (int(idx[0]) + 1) if len(idx) else int(n)


def weighted_pod_svd_data_dense(S, M_dense, jitter=1e-12):
    N = M_dense.shape[0]
    M = 0.5 * (M_dense + M_dense.T)
    eps = float(jitter) * float(np.trace(M)) / max(N, 1)
    L = np.linalg.cholesky(M + eps * np.eye(N))
    Sw = L @ S
    Uhat, s, _ = np.linalg.svd(Sw, full_matrices=False)
    return L, Uhat, s


def weighted_pod_basis_from_svd(L, Uhat, s, tau):
    r = pick_r_from_svals(s, tau)
    V = np.linalg.solve(L, Uhat[:, :r])
    return V, r


def l2_pod_basis_by_tau(S: np.ndarray, tau_list: Tuple[float, ...]) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
    S64 = np.asarray(S, dtype=np.float64, order="F")
    U, s, _ = np.linalg.svd(S64, full_matrices=False)
    out = {}
    for tau in tau_list:
        r = pick_r_from_svals(s, float(tau))
        out[float(tau)] = (U[:, :r].copy(), s.copy())
    return out


def matvec_ngsolve(mat: BaseMatrix, x_np: np.ndarray, xvec=None, yvec=None) -> np.ndarray:
    if xvec is None:
        xvec = mat.CreateColVector()
    if yvec is None:
        yvec = mat.CreateRowVector()
    xvec.FV().NumPy()[:] = x_np
    mat.Mult(xvec, yvec)
    return yvec.FV().NumPy().copy()


def weighted_pod_operator_basis(
    S: np.ndarray,
    Mmat: BaseMatrix,
    tau_list: Tuple[float, ...],
    max_modes: int = None,
    oversample: int = 10,
    niter: int = 2,
    tol: float = 1e-8,
    maxit: int = 200,
    eps_eig: float = 1e-14,
):
    from scipy.sparse.linalg import LinearOperator, lobpcg

    S64 = np.asarray(S, dtype=np.float64, order="F")
    _, ns = S64.shape

    xvec = Mmat.CreateColVector()
    yvec = Mmat.CreateRowVector()

    def M_apply_block(Y):
        out = np.empty_like(Y)
        for j in range(Y.shape[1]):
            out[:, j] = matvec_ngsolve(Mmat, Y[:, j], xvec=xvec, yvec=yvec)
        return out

    def C_mv(V):
        Y = S64 @ V
        MY = M_apply_block(Y)
        return S64.T @ MY

    C = LinearOperator(
        (ns, ns),
        matvec=lambda v: C_mv(v.reshape(ns, 1)).ravel(),
        matmat=C_mv,
        dtype=np.float64
    )

    tau_min = float(min(tau_list))
    k_guess = min(ns, 200)
    if max_modes is not None:
        k_guess = min(k_guess, int(max_modes))

    X = np.random.default_rng(0).standard_normal((ns, k_guess))
    for _ in range(max(0, int(niter))):
        X = C_mv(X)
        X, _ = np.linalg.qr(X)

    vals, vecs = lobpcg(C, X, tol=tol, maxiter=maxit, largest=True)

    idx = np.argsort(vals)[::-1]
    lam = np.maximum(vals[idx], 0.0)
    W = vecs[:, idx]

    lam_max = float(lam[0]) if lam.size else 0.0
    keep = lam > (eps_eig * max(lam_max, 1.0))
    lam = lam[keep]
    W = W[:, keep]

    if lam.size == 0:
        svals = np.array([0.0], dtype=float)
        V_fallback = orthonormalize_columns(S64[:, :1].copy())
        return {float(tau): (V_fallback, svals) for tau in tau_list}

    svals = np.sqrt(lam)

    r_max = pick_r_from_svals(svals, tau_min)
    r_max = min(r_max, W.shape[1])

    A = W[:, :r_max] / svals[:r_max].reshape(1, -1)
    Vmax = S64 @ A

    MV = M_apply_block(Vmax)
    Gram = 0.5 * (Vmax.T @ MV + (Vmax.T @ MV).T)
    R = np.linalg.cholesky(Gram + 1e-14 * np.eye(r_max))
    Vmax = Vmax @ np.linalg.inv(R)

    out = {}
    for tau in tau_list:
        r = pick_r_from_svals(svals, float(tau))
        r = int(min(r, Vmax.shape[1]))
        out[float(tau)] = (Vmax[:, :r].copy(), svals.copy())
    return out


# ============================================================
# Reduction helpers
# ============================================================
def reduce_square_mat(mat, U: np.ndarray):
    _, r = U.shape
    R = np.empty((r, r), dtype=float)
    x = mat.CreateColVector()
    y = mat.CreateRowVector()
    for j in range(r):
        x.FV().NumPy()[:] = U[:, j]
        mat.Mult(x, y)
        R[:, j] = U.T @ y.FV().NumPy()
    return R


def project_vec(V: np.ndarray, v_np: np.ndarray):
    return V.T @ v_np


def build_Br_by_forms(mesh, V, Q, Vp: np.ndarray, Vu: np.ndarray):
    r_p = Vp.shape[1]
    r_u = Vu.shape[1]
    Br = np.empty((r_p, r_u), dtype=float)

    q = Q.TestFunction()
    uh = GridFunction(V)

    for j in range(r_u):
        uh.vec.FV().NumPy()[:] = Vu[:, j]
        lf = LinearForm(Q)
        lf += (div(uh) * q) * dx
        lf.Assemble()
        Br[:, j] = Vp.T @ lf.vec.FV().NumPy()

    return Br


# ============================================================
# FOM solvers
# ============================================================
def solve_fom_store_coarse_snapshots(
    mesh, V, Q, Y, f_base,
    mu_x, mu_y, cfg: Config,
    Su_out: np.ndarray, Sp_out: np.ndarray,
    col0: int
) -> int:
    (uY, pY) = Y.TrialFunction()
    (vY, qY) = Y.TestFunction()

    K = cfg.k0 + cfg.delta * exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * cfg.w ** 2))
    Kinv = 1.0 / K
    S_cf = CoefficientFunction(float(cfg.Sstor))

    p_old = GridFunction(Q)
    p_old.Set(CoefficientFunction(float(cfg.p_init)))

    gfu = GridFunction(Y)
    uh, ph = gfu.components

    a = BilinearForm(Y, symmetric=True)
    a += (Kinv * InnerProduct(uY, vY)
          - pY * div(vY)
          + qY * div(uY)
          + (S_cf / cfg.dt) * pY * qY) * dx
    a.Assemble()
    inv = a.mat.Inverse(Y.FreeDofs(), inverse="pardiso")

    nsteps = int(np.round(cfg.T_snap / cfg.dt))
    col = col0

    for nstep in range(1, nsteps + 1):
        t = nstep * cfg.dt
        f_cf_t = ramp_value(t, cfg) * f_base

        L = LinearForm(Y)
        L += (f_cf_t * qY + (S_cf / cfg.dt) * p_old * qY) * dx
        L.Assemble()

        gfu.vec.data = inv * L.vec
        p_old.vec.data = ph.vec

        if (nstep % cfg.store_every) == 0:
            Su_out[:, col] = uh.vec.FV().NumPy().astype(np.float32, copy=False)
            Sp_out[:, col] = ph.vec.FV().NumPy().astype(np.float32, copy=False)
            col += 1

        if cfg.print_every_step and (nstep % cfg.print_every_step == 0):
            print(f"      step {nstep}/{nsteps}", flush=True)

    return col


def solve_fom_history_for_test_full(mesh, V, Q, Y, f_base, cfg: Config, mu):
    mu_x, mu_y = float(mu[0]), float(mu[1])

    (uY, pY) = Y.TrialFunction()
    (vY, qY) = Y.TestFunction()

    K = cfg.k0 + cfg.delta * exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * cfg.w ** 2))
    Kinv = 1.0 / K
    S_cf = CoefficientFunction(float(cfg.Sstor))

    p_old = GridFunction(Q)
    p_old.Set(CoefficientFunction(float(cfg.p_init)))

    gfu = GridFunction(Y)
    uh, ph = gfu.components

    a = BilinearForm(Y, symmetric=True)
    a += (Kinv * InnerProduct(uY, vY)
          - pY * div(vY)
          + qY * div(uY)
          + (S_cf / cfg.dt) * pY * qY) * dx
    a.Assemble()
    inv = a.mat.Inverse(Y.FreeDofs(), inverse="pardiso")

    nsteps = int(np.round(cfg.T_test / cfg.dt))

    Nu, Np = V.ndof, Q.ndof
    hu = np.empty((nsteps, Nu), dtype=np.float32)
    hp = np.empty((nsteps, Np), dtype=np.float32)

    for nstep in range(1, nsteps + 1):
        t = nstep * cfg.dt
        f_cf_t = ramp_value(t, cfg) * f_base

        L = LinearForm(Y)
        L += (f_cf_t * qY + (S_cf / cfg.dt) * p_old * qY) * dx
        L.Assemble()

        gfu.vec.data = inv * L.vec
        p_old.vec.data = ph.vec

        hu[nstep - 1, :] = uh.vec.FV().NumPy().astype(np.float32, copy=False)
        hp[nstep - 1, :] = ph.vec.FV().NumPy().astype(np.float32, copy=False)

    return hu, hp


# ============================================================
# Global snapshot cache
# ============================================================
def worker_snapshot_chunk(args):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    cfg_dict, mu_chunk = args
    cfg = Config(**cfg_dict)

    mesh = Mesh(unit_square.GenerateMesh(maxh=cfg.maxh))
    V = HDiv(mesh, order=cfg.order, dirichlet=cfg.Gamma_N)
    Q = L2(mesh, order=cfg.order - 1)
    Y = FESpace([V, Q])

    f_base = forcing_base_cf(cfg)

    Nu, Np = V.ndof, Q.ndof
    nsteps_fine = int(round(cfg.T_snap / cfg.dt))
    se = int(cfg.store_every)
    if nsteps_fine % se != 0:
        raise RuntimeError(f"nsteps_fine={nsteps_fine} not divisible by store_every={se}")
    keep_steps = nsteps_fine // se

    Su = np.empty((Nu, len(mu_chunk) * keep_steps), dtype=np.float32)
    Sp = np.empty((Np, len(mu_chunk) * keep_steps), dtype=np.float32)

    col = 0
    for i, mu in enumerate(mu_chunk, start=1):
        print(f"[worker {os.getpid()}] snapshot chunk {i}/{len(mu_chunk)} mu={mu}", flush=True)
        col = solve_fom_store_coarse_snapshots(
            mesh, V, Q, Y, f_base,
            mu_x=float(mu[0]), mu_y=float(mu[1]),
            cfg=cfg,
            Su_out=Su, Sp_out=Sp,
            col0=col
        )

    return {
        "mu_chunk": [tuple(mu) for mu in mu_chunk],
        "Su": Su,
        "Sp": Sp,
    }


def build_global_snapshot_cache(cfg: Config, mu_train_all: List[Tuple[float, float]]):
    import multiprocessing as mp

    if len(mu_train_all) == 0:
        raise ValueError("mu_train_all is empty")

    nproc = min(cfg.nproc_regions, len(mu_train_all))
    mu_chunks = chunk_list(mu_train_all, nproc)
    cfg_dict = asdict(cfg)

    ctx = mp.get_context("spawn") if hasattr(mp, "get_context") else mp
    worker_args = [(cfg_dict, chunk) for chunk in mu_chunks if len(chunk) > 0]

    out_parts = []
    with ctx.Pool(processes=len(worker_args)) as pool:
        for part in pool.imap_unordered(worker_snapshot_chunk, worker_args, chunksize=1):
            out_parts.append(part)

    order = {tuple(mu): i for i, mu in enumerate(mu_train_all)}
    out_parts.sort(key=lambda part: order[tuple(part["mu_chunk"][0])])

    mu_ordered = []
    for part in out_parts:
        mu_ordered.extend([tuple(mu) for mu in part["mu_chunk"]])

    if mu_ordered != list(mu_train_all):
        raise RuntimeError("Global snapshot cache assembly changed mu ordering")

    Su_all = np.concatenate([part["Su"] for part in out_parts], axis=1)
    Sp_all = np.concatenate([part["Sp"] for part in out_parts], axis=1)

    return {
        "mu_list": list(mu_train_all),
        "mu_to_index": {tuple(mu): i for i, mu in enumerate(mu_train_all)},
        "Su_all": Su_all,
        "Sp_all": Sp_all,
    }


def extract_mu_snapshot_block(cache, mu, keep_steps):
    idx = cache["mu_to_index"][tuple(mu)]
    c0 = idx * keep_steps
    c1 = (idx + 1) * keep_steps
    Su = cache["Su_all"][:, c0:c1]
    Sp = cache["Sp_all"][:, c0:c1]
    return Su, Sp


def extract_leaf_snapshot_matrix_from_cache(cache: dict, mu_list_leaf: List[Tuple[float, float]], keep_steps: int):
    mu_to_index = cache["mu_to_index"]
    Su_all = cache["Su_all"]
    Sp_all = cache["Sp_all"]

    col_blocks = []
    for mu in mu_list_leaf:
        idx = mu_to_index[tuple(mu)]
        c0 = idx * keep_steps
        c1 = (idx + 1) * keep_steps
        col_blocks.append((c0, c1))

    total_cols = len(mu_list_leaf) * keep_steps
    Nu = Su_all.shape[0]
    Np = Sp_all.shape[0]

    Su = np.empty((Nu, total_cols), dtype=Su_all.dtype)
    Sp = np.empty((Np, total_cols), dtype=Sp_all.dtype)

    dest = 0
    for c0, c1 in col_blocks:
        width = c1 - c0
        Su[:, dest:dest + width] = Su_all[:, c0:c1]
        Sp[:, dest:dest + width] = Sp_all[:, c0:c1]
        dest += width

    return Su, Sp

# ============================================================
# POD-only metadata from cache
# ============================================================
def build_leaf_pod_only_from_cache(
    cfg: Config,
    leaf_id: str,
    depth: int,
    mu_train: List[Tuple[float, float]],
    cache: dict,
    bounds: Optional[Tuple[float, float, float, float]] = None,
):
    mesh = Mesh(unit_square.GenerateMesh(maxh=cfg.maxh))
    V = HDiv(mesh, order=cfg.order, dirichlet=cfg.Gamma_N)
    Q = L2(mesh, order=cfg.order - 1)

    f_base = forcing_base_cf(cfg)
    mu_mid = (0.5 * (cfg.a_mu + cfg.b_mu), 0.5 * (cfg.a_mu + cfg.b_mu))
    _, _, Mp_mat, _ = assemble_sparse_blocks(mu_mid, V, Q, mesh, f_base, cfg.k0, cfg.delta, cfg.w)

    if len(mu_train) == 0:
        return None

    if bounds is None:
        raise ValueError("bounds must be provided in the rectangular-only code path")
    ax, bx, ay, by = bounds

    mus = np.asarray(mu_train, dtype=float)
    mu_ref = tuple(np.mean(mus, axis=0))
    # mu_ref = (0.5 * (ax + bx), 0.5 * (ay + by))
    Muu_ref_mat, _, _, _ = assemble_sparse_blocks(mu_ref, V, Q, mesh, f_base, cfg.k0, cfg.delta, cfg.w)

    nsteps_fine = int(round(cfg.T_snap / cfg.dt))
    se = int(cfg.store_every)
    if nsteps_fine % se != 0:
        raise RuntimeError(f"nsteps_fine={nsteps_fine} not divisible by store_every={se}")
    keep_steps = nsteps_fine // se

    Su, Sp = extract_leaf_snapshot_matrix_from_cache(cache, mu_train, keep_steps)

    tmp_p = weighted_pod_operator_basis(
        Sp, Mp_mat, cfg.tau_list,
        oversample=10, niter=1, tol=1e-7, maxit=150,
        max_modes=250,
    )
    s_p = next(iter(tmp_p.values()))[1].copy()
    rp_by_tau = {str(float(tau)): int(pick_r_from_svals(s_p, float(tau))) for tau in cfg.tau_list}

    pod_mode = str(cfg.pod_mode).strip()
    if pod_mode == "dense_weight":
        Muu_ref_dense = np.array(Muu_ref_mat.ToDense())
        Su64 = np.asarray(Su, dtype=np.float64, order="F")
        L_u, Uhat_u, s_u = weighted_pod_svd_data_dense(Su64, Muu_ref_dense)
        ru_raw_by_tau = {str(float(tau)): int(pick_r_from_svals(s_u, float(tau))) for tau in cfg.tau_list}
    elif pod_mode == "snap_weights":
        tmp_u = weighted_pod_operator_basis(
            Su, Muu_ref_mat, cfg.tau_list,
            oversample=10, niter=1, tol=1e-7, maxit=150,
            max_modes=250,
        )
        s_u = next(iter(tmp_u.values()))[1].copy()
        ru_raw_by_tau = {str(float(tau)): int(pick_r_from_svals(s_u, float(tau))) for tau in cfg.tau_list}
    elif pod_mode == "L2":
        tmp_u = l2_pod_basis_by_tau(Su, cfg.tau_list)
        s_u = next(iter(tmp_u.values()))[1].copy()
        ru_raw_by_tau = {str(float(tau)): int(pick_r_from_svals(s_u, float(tau))) for tau in cfg.tau_list}
    else:
        raise ValueError(f"Unknown pod_mode={cfg.pod_mode!r}")

    pod_meta = {
        "leaf_id": leaf_id,
        "depth": int(depth),
        "bounds": {"ax": ax, "bx": bx, "ay": ay, "by": by},
        "mu_ref": [float(mu_ref[0]), float(mu_ref[1])],
        "pod_mode_u": str(cfg.pod_mode),
        "u_svals_full": np.asarray(s_u, dtype=float).tolist(),
        "p_svals_full": np.asarray(s_p, dtype=float).tolist(),
        "u_raw_r_by_tau": ru_raw_by_tau,
        "p_r_by_tau": rp_by_tau,
        "n_mu_train": int(len(mu_train)),
    }
    return (leaf_id, pod_meta)


# ============================================================
# Supremizers
# ============================================================
def compute_supremizers_for_pressure_basis(Vspace, Qspace, V_p_basis, mu_ref, cfg: Config, inverse_type="pardiso"):
    mu_x, mu_y = mu_ref
    K_ref = cfg.k0 + cfg.delta * exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * cfg.w ** 2))
    Kinv_ref = 1.0 / K_ref

    uV, vV = Vspace.TrialFunction(), Vspace.TestFunction()
    aV = BilinearForm(Vspace, symmetric=True)
    aV += (Kinv_ref * InnerProduct(uV, vV)) * dx
    aV.Assemble()

    try:
        invA = aV.mat.Inverse(Vspace.FreeDofs(), inverse=inverse_type)
    except Exception:
        invA = safe_inverse(aV.mat, Vspace.FreeDofs())

    r_p = V_p_basis.shape[1]
    Nu = Vspace.ndof
    S_sup = np.empty((Nu, r_p), dtype=float)

    for i in range(r_p):
        psi = GridFunction(Qspace)
        psi.vec.FV().NumPy()[:] = V_p_basis[:, i]

        Ls = LinearForm(Vspace)
        Ls += (-psi) * div(vV) * dx
        Ls.Assemble()

        s = GridFunction(Vspace)
        s.vec.data = invA * Ls.vec
        S_sup[:, i] = s.vec.FV().NumPy()

    return S_sup


# ============================================================
# ROM / CoSTA
# ============================================================
# ============================================================
# DNN CoSTA helpers
# ============================================================
def get_activation(name: str):
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation {name!r}")


class CoSTAMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_width: int, hidden_depth: int, activation: str):
        super().__init__()
        layers = []
        act = get_activation(activation)

        d = in_dim
        for _ in range(int(hidden_depth)):
            layers.append(nn.Linear(d, int(hidden_width)))
            layers.append(copy.deepcopy(act))
            d = int(hidden_width)

        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def make_costa_feature_vector(
    z_tilde_n: np.ndarray,
    mu,
    tprev_norm: float,
    model: dict,
    cfg: Optional[Config] = None,
):
    use_mu = bool(model.get("costa_use_mu_features", True))
    use_time = bool(model.get("costa_use_time_feature", True))

    if cfg is not None:
        use_mu = bool(cfg.costa_use_mu_features)
        use_time = bool(cfg.costa_use_time_feature)

    feats = [np.asarray(z_tilde_n, dtype=np.float64)]

    if use_mu:
        Pdeg = int(model["Pdeg"])
        ax, bx, ay, by = model["ax"], model["bx"], model["ay"], model["by"]
        theta_mu = legendre_basis_rect(mu, Pdeg, ax, bx, ay, by)
        feats.append(np.asarray(theta_mu, dtype=np.float64))

    if use_time:
        feats.append(np.array([float(tprev_norm)], dtype=np.float64))

    return np.concatenate(feats, axis=0)

def build_reduced_rhs(model, p_old_r: np.ndarray, t: float):
    dt = float(model["dt"])
    S = float(model["Sstor"])
    r_u = int(model["r_u"])
    r_p = int(model["r_p"])

    Mpr = model["Mpr"]
    frq = model["fr_q"]

    alpha = ramp_value_model(t, model)

    rhs = np.zeros(r_u + r_p, dtype=np.float64)
    rhs[r_u:] = alpha * frq + (S / dt) * (Mpr @ p_old_r)
    return rhs


def fit_standardizer(X: np.ndarray, eps: float = 1e-12):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std = np.where(std < eps, 1.0, std)
    return mean, std


def apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (X - mean) / std

def build_costa_training_data_from_Zref(model, mu_train, Zref_list, cfg: Config):
    nsteps = int(model["nsteps_train"])

    X_list = []
    Y_list = []

    for i, mu in enumerate(mu_train):
        mu_loc = (float(mu[0]), float(mu[1]))
        Zref = Zref_list[i]

        Ar = build_reduced_matrix_Ar(model, mu_loc, symmetrize=True)
        Ar_lu = lu_factor(Ar, check_finite=False)

        for n in range(1, nsteps + 1):
            z_prev_ref = Zref[n - 1, :]
            z_ref_n = Zref[n, :]

            t = n * float(model["dt"])
            rhs = build_reduced_rhs(model, z_prev_ref[model["r_u"]:], t)

            z_tilde_n = lu_solve(Ar_lu, rhs, check_finite=False)
            sigma_n = Ar @ z_ref_n - rhs

            tprev_norm = (n - 1) / float(nsteps)
            x_n = make_costa_feature_vector(
                z_tilde_n=z_tilde_n,
                mu=mu_loc,
                tprev_norm=tprev_norm,
                model=model,
                cfg=cfg,
            )

            X_list.append(x_n)
            Y_list.append(np.asarray(sigma_n, dtype=np.float64))

    X = np.asarray(X_list, dtype=np.float64)
    Y = np.asarray(Y_list, dtype=np.float64)
    return X, Y

def train_costa_dnn_from_Zref(model, mu_train, cfg: Config, Zref_list: List[np.ndarray]):
    """
    Train a CoSTA DNN with:
      - MSE loss
      - Adam
      - validation monitoring
      - early stopping
      - normalization

    Important multiprocessing note:
    We store the trained state_dict as plain NumPy arrays, not torch tensors,
    so the model object can be sent safely through multiprocessing pipes.
    """
    X, Y = build_costa_training_data_from_Zref(model, mu_train, Zref_list, cfg)

    x_mean, x_std = fit_standardizer(X)
    y_mean, y_std = fit_standardizer(Y)

    Xn = apply_standardizer(X, x_mean, x_std)
    Yn = apply_standardizer(Y, y_mean, y_std)

    X_t = torch.tensor(Xn, dtype=torch.float32)
    Y_t = torch.tensor(Yn, dtype=torch.float32)

    dataset = TensorDataset(X_t, Y_t)

    n_total = len(dataset)
    n_val = max(1, int(round(cfg.costa_val_fraction * n_total)))
    n_val = min(n_val, n_total - 1) if n_total > 1 else 0
    n_train = n_total - n_val

    if n_val > 0:
        train_ds, val_ds = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(cfg.rng_seed),
        )
    else:
        train_ds = dataset
        val_ds = None

    train_loader = DataLoader(
        train_ds,
        batch_size=min(int(cfg.costa_batch_size), max(1, len(train_ds))),
        shuffle=True,
        drop_last=False,
    )

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=min(int(cfg.costa_batch_size), max(1, len(val_ds))),
            shuffle=False,
            drop_last=False,
        )

    in_dim = X.shape[1]
    out_dim = Y.shape[1]

    device = torch.device(cfg.costa_device)
    net = CoSTAMLP(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_width=cfg.costa_hidden_width,
        hidden_depth=cfg.costa_hidden_depth,
        activation=cfg.costa_activation,
    ).to(device)

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=float(cfg.costa_lr),
        weight_decay=float(cfg.costa_weight_decay),
    )
    loss_fn = nn.MSELoss()

    best_state = None
    best_val = np.inf
    bad_epochs = 0

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(1, int(cfg.costa_epochs) + 1):
        net.train()
        train_loss_sum = 0.0
        train_count = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = net(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            bsz = xb.shape[0]
            train_loss_sum += float(loss.item()) * bsz
            train_count += bsz

        train_loss = train_loss_sum / max(train_count, 1)
        history["train_loss"].append(float(train_loss))

        if val_loader is not None:
            net.eval()
            val_loss_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    pred = net(xb)
                    loss = loss_fn(pred, yb)
                    bsz = xb.shape[0]
                    val_loss_sum += float(loss.item()) * bsz
                    val_count += bsz

            val_loss = val_loss_sum / max(val_count, 1)
        else:
            val_loss = train_loss

        history["val_loss"].append(float(val_loss))

        if val_loss < best_val:
            best_val = float(val_loss)
            best_state = copy.deepcopy(net.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if (
            epoch >= int(cfg.costa_min_epochs)
            and bad_epochs >= int(cfg.costa_early_stop_patience)
        ):
            break

    if best_state is None:
        best_state = copy.deepcopy(net.state_dict())

    # ---------------------------------------------------------
    # Convert state_dict to plain NumPy arrays for safe transport
    # through multiprocessing.
    # ---------------------------------------------------------
    best_state_numpy = {
        key: val.detach().cpu().numpy().copy()
        for key, val in best_state.items()
    }

    model_c = dict(model)
    model_c["costa_mode"] = "dnn_predictor_corrector"
    model_c["costa_use_dnn"] = True
    model_c["costa_net_state_dict"] = best_state_numpy
    model_c["costa_net_in_dim"] = int(in_dim)
    model_c["costa_net_out_dim"] = int(out_dim)
    model_c["costa_hidden_width"] = int(cfg.costa_hidden_width)
    model_c["costa_hidden_depth"] = int(cfg.costa_hidden_depth)
    model_c["costa_activation"] = str(cfg.costa_activation)
    model_c["costa_x_mean"] = np.asarray(x_mean, dtype=np.float64)
    model_c["costa_x_std"] = np.asarray(x_std, dtype=np.float64)
    model_c["costa_y_mean"] = np.asarray(y_mean, dtype=np.float64)
    model_c["costa_y_std"] = np.asarray(y_std, dtype=np.float64)
    model_c["costa_train_history"] = history

    return model_c


def train_costa_ridge_from_Zref(model, mu_train, cfg: Config, Zref_list: List[np.ndarray]):
    """
    Train a linear ridge CoSTA corrector:
      sigma_hat = y_mean + y_std * ([x_norm, 1] @ W)
    """
    X, Y = build_costa_training_data_from_Zref(model, mu_train, Zref_list, cfg)

    x_mean, x_std = fit_standardizer(X)
    y_mean, y_std = fit_standardizer(Y)

    Xn = apply_standardizer(X, x_mean, x_std)
    Yn = apply_standardizer(Y, y_mean, y_std)

    n_samples, in_dim = Xn.shape
    out_dim = Yn.shape[1]

    Xa = np.hstack([Xn, np.ones((n_samples, 1), dtype=np.float64)])
    lam = float(cfg.costa_ridge)
    reg = lam * np.eye(in_dim + 1, dtype=np.float64)
    reg[-1, -1] = 0.0  # do not regularize the bias term

    lhs = Xa.T @ Xa + reg
    rhs = Xa.T @ Yn
    W = np.linalg.solve(lhs, rhs)  # shape: (in_dim + 1, out_dim)

    Yn_hat = Xa @ W
    train_mse = float(np.mean((Yn_hat - Yn) ** 2))

    model_c = dict(model)
    model_c["costa_mode"] = "ridge_predictor_corrector"
    model_c["costa_use_dnn"] = False
    model_c["costa_net_in_dim"] = int(in_dim)
    model_c["costa_net_out_dim"] = int(out_dim)
    model_c["costa_x_mean"] = np.asarray(x_mean, dtype=np.float64)
    model_c["costa_x_std"] = np.asarray(x_std, dtype=np.float64)
    model_c["costa_y_mean"] = np.asarray(y_mean, dtype=np.float64)
    model_c["costa_y_std"] = np.asarray(y_std, dtype=np.float64)
    model_c["costa_ridge_lambda"] = lam
    model_c["costa_ridge_weights"] = np.asarray(W, dtype=np.float64)
    model_c["costa_train_history"] = {"train_mse_norm": train_mse}

    return model_c
def worker_build_one_tauP(args):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    cfg_dict, tau, Pdeg_req, leaf_dict, mu_train, tau_payload = args
    cfg = Config(**cfg_dict)
    leaf = PseudoLeaf(**leaf_dict)

    ax, bx, ay, by = leaf.ax, leaf.bx, leaf.ay, leaf.by
    Ns = len(mu_train)

    P_cap_leaf = max_admissible_poly_degree(Ns)
    Pdeg_eff = min(int(Pdeg_req), P_cap_leaf)

    Muu_coeffs = regression_matrix_entries_rect(
        mu_train=mu_train,
        Mats=tau_payload["Muu_rs"],
        Pdeg=Pdeg_eff,
        ax=ax, bx=bx, ay=ay, by=by,
        ridge=cfg.muu_ridge,
    )

    model = {
        "dt": float(cfg.dt),
        "Sstor": float(cfg.Sstor),
        "ax": float(ax), "bx": float(bx), "ay": float(ay), "by": float(by),
        "Pdeg": int(Pdeg_eff),
        "V_u_stab": tau_payload["Vu_stab"],
        "V_p": tau_payload["Vp"],
        "r_u": int(tau_payload["r_u"]),
        "r_p": int(tau_payload["r_p"]),
        "Br": tau_payload["Br"],
        "Mpr": tau_payload["Mpr"],
        "fr_q": tau_payload["fr_q"],
        "Muu_coeffs": Muu_coeffs,
        "nsteps_train": int(tau_payload["nsteps_fine"]),
        "store_every": int(tau_payload["store_every"]),
        "p0_r": tau_payload["p0_r"],
        "use_ramp": bool(cfg.use_ramp),
        "t_ramp_end": float(cfg.t_ramp_end),
        "ramp_target": float(cfg.ramp_target),
        "n_mu_train_leaf": int(Ns),
        "mu_ref": tau_payload["mu_ref"],
    }

    model_c = costa_fit_shared_lowram_from_Zref(
        model=model,
        mu_train=mu_train,
        cfg=cfg,
        Zref_list=tau_payload["Zref_list"],
    )

    return {
        "comb": (float(tau), int(Pdeg_req)),
        "model": model,
        "model_c": model_c,
    }

def build_costa_net_from_model(model_c: dict, device: str = "cpu"):
    net = CoSTAMLP(
        in_dim=int(model_c["costa_net_in_dim"]),
        out_dim=int(model_c["costa_net_out_dim"]),
        hidden_width=int(model_c["costa_hidden_width"]),
        hidden_depth=int(model_c["costa_hidden_depth"]),
        activation=str(model_c["costa_activation"]),
    )

    raw_state = model_c["costa_net_state_dict"]

    # Rebuild a proper torch state_dict from NumPy arrays
    state_dict = {
        key: torch.tensor(val, dtype=torch.float32)
        for key, val in raw_state.items()
    }

    net.load_state_dict(state_dict)
    net.to(torch.device(device))
    net.eval()
    return net

def build_reduced_matrix_Ar(model, mu, symmetrize=True):
    dt, S = model["dt"], model["Sstor"]
    Br, Mpr = model["Br"], model["Mpr"]
    r_u, r_p = model["r_u"], model["r_p"]

    Muu_r = eval_matrix_from_coeffs_rect(
        model["Muu_coeffs"], mu, model["Pdeg"],
        model["ax"], model["bx"], model["ay"], model["by"],
        symmetrize=symmetrize
    )

    Ar = np.zeros((r_u + r_p, r_u + r_p), dtype=float)
    Ar[:r_u, :r_u] = Muu_r
    Ar[:r_u, r_u:] = -Br.T
    Ar[r_u:, :r_u] = Br
    Ar[r_u:, r_u:] = (S / dt) * Mpr
    return Ar


def rom_predictor_rollout(model, mu, nsteps, p0=0.0):
    dt, S = model["dt"], model["Sstor"]
    r_u, r_p = model["r_u"], model["r_p"]
    r = r_u + r_p
    Mpr, frq = model["Mpr"], model["fr_q"]

    Ar = build_reduced_matrix_Ar(model, mu, symmetrize=True)
    Ar_lu = lu_factor(Ar, check_finite=False)

    Z = np.zeros((nsteps + 1, r), dtype=float)

    p0_r = model.get("p0_r", None)
    if p0_r is None:
        p_old_r = np.zeros(r_p, dtype=float)
        p_old_r[:] = float(p0)
    else:
        p_old_r = np.array(p0_r, dtype=float, copy=True)

    Z[0, r_u:] = p_old_r

    for n in range(1, nsteps + 1):
        t = n * dt
        alpha = ramp_value_model(t, model)

        rhs = np.zeros(r, dtype=float)
        rhs[r_u:] = alpha * frq + (S / dt) * (Mpr @ p_old_r)

        z = lu_solve(Ar_lu, rhs, check_finite=False)
        p_old_r = z[r_u:]
        Z[n, :] = z

    return Z


def build_Zref_fine_from_coarse_snapshots(
    Su_coarse: np.ndarray,
    Sp_coarse: np.ndarray,
    mu_idx: int,
    keep_steps: int,
    nsteps_fine: int,
    store_every: int,
    Vu_stab: np.ndarray,
    Vp: np.ndarray,
    project_p_cached,
    p0_r: np.ndarray,
):
    r_u = Vu_stab.shape[1]
    r_p = Vp.shape[1]
    r = r_u + r_p

    Zc = np.zeros((keep_steps + 1, r), dtype=float)
    Zc[0, r_u:] = p0_r

    col0 = mu_idx * keep_steps
    for k in range(1, keep_steps + 1):
        u_full = Su_coarse[:, col0 + (k - 1)].astype(np.float64, copy=False)
        p_full = Sp_coarse[:, col0 + (k - 1)].astype(np.float64, copy=False)
        Zc[k, :r_u] = Vu_stab.T @ u_full
        Zc[k, r_u:] = project_p_cached(p_full)

    Zf = np.zeros((nsteps_fine + 1, r), dtype=float)
    Zf[0, :] = Zc[0, :]

    se = int(store_every)
    for n in range(1, nsteps_fine + 1):
        k_right = int(np.ceil(n / se))
        k_right = min(max(k_right, 1), keep_steps)
        k_left = k_right - 1

        n_left = k_left * se
        n_right = k_right * se
        alpha = 1.0 if n_right == n_left else (n - n_left) / float(n_right - n_left)

        Zf[n, :] = (1.0 - alpha) * Zc[k_left, :] + alpha * Zc[k_right, :]

    return Zf


def get_cached_costa_net(model_c: dict, device: str = "cpu"):
    cache_key = f"_costa_net_cached_{device}"
    if cache_key not in model_c:
        model_c[cache_key] = build_costa_net_from_model(model_c, device=device)
    return model_c[cache_key]
def online_rollout_transient_ROM_CoSTA_shared(model_c, mu, nsteps, p0=0.0):
    """
    Predictor-corrector CoSTA rollout in ROM form.
    """
    r_u = int(model_c["r_u"])
    r_p = int(model_c["r_p"])
    r = r_u + r_p

    costa_mode = str(model_c.get("costa_mode", "dnn_predictor_corrector")).strip().lower()
    use_dnn = costa_mode.startswith("dnn")
    net = get_cached_costa_net(model_c, device="cpu") if use_dnn else None
    W_ridge = np.asarray(model_c["costa_ridge_weights"], dtype=np.float64) if not use_dnn else None

    x_mean = np.asarray(model_c["costa_x_mean"], dtype=np.float64)
    x_std = np.asarray(model_c["costa_x_std"], dtype=np.float64)
    y_mean = np.asarray(model_c["costa_y_mean"], dtype=np.float64)
    y_std = np.asarray(model_c["costa_y_std"], dtype=np.float64)

    Ar = build_reduced_matrix_Ar(model_c, mu, symmetrize=True)
    Ar_lu = lu_factor(Ar, check_finite=False)

    Zc = np.zeros((nsteps + 1, r), dtype=np.float64)

    p0_r = model_c.get("p0_r", None)
    if p0_r is None:
        z_prev = np.zeros(r, dtype=np.float64)
        z_prev[r_u:] = float(p0)
    else:
        z_prev = np.zeros(r, dtype=np.float64)
        z_prev[r_u:] = np.array(p0_r, dtype=np.float64, copy=True)

    Zc[0, :] = z_prev

    for n in range(1, nsteps + 1):
        t = n * float(model_c["dt"])
        rhs = build_reduced_rhs(model_c, z_prev[r_u:], t)

        z_tilde_n = lu_solve(Ar_lu, rhs, check_finite=False)

        tprev_norm = (n - 1) / float(nsteps)
        x_feat = make_costa_feature_vector(
            z_tilde_n=z_tilde_n,
            mu=mu,
            tprev_norm=tprev_norm,
            model=model_c,
        )

        x_norm = (x_feat - x_mean) / x_std
        if use_dnn:
            x_t = torch.tensor(x_norm[None, :], dtype=torch.float32)
            with torch.no_grad():
                y_hat_norm = net(x_t).cpu().numpy()[0]
        else:
            x_aug = np.concatenate([x_norm, np.array([1.0], dtype=np.float64)])
            y_hat_norm = x_aug @ W_ridge

        sigma_hat = y_mean + y_std * y_hat_norm
        z_corr = lu_solve(Ar_lu, rhs + sigma_hat, check_finite=False)

        Zc[n, :] = z_corr
        z_prev = z_corr

    return Zc

# ============================================================
# Norms / errors
# ============================================================
def L2_norm_scalar_on_mesh(p_vec, Qspace, mesh, order=5):
    g = GridFunction(Qspace)
    g.vec.FV().NumPy()[:] = p_vec
    return float(sqrt(Integrate(g * g, mesh, order=int(order))))


def L2_norm_vector_on_mesh(u_vec, Vspace, mesh, order=5):
    g = GridFunction(Vspace)
    g.vec.FV().NumPy()[:] = u_vec
    return float(sqrt(Integrate(InnerProduct(g, g), mesh, order=int(order))))


def abs_L2_error_scalar(p_rb, p_fom, Qspace, mesh, order=5):
    g1 = GridFunction(Qspace)
    g1.vec.FV().NumPy()[:] = p_rb
    g2 = GridFunction(Qspace)
    g2.vec.FV().NumPy()[:] = p_fom
    diff = g2 - g1
    return float(sqrt(Integrate(diff * diff, mesh, order=int(order))))


def abs_L2_error_vector(u_rb, u_fom, Vspace, mesh, order=5):
    g1 = GridFunction(Vspace)
    g1.vec.FV().NumPy()[:] = u_rb
    g2 = GridFunction(Vspace)
    g2.vec.FV().NumPy()[:] = u_fom
    diff = g2 - g1
    return float(sqrt(Integrate(InnerProduct(diff, diff), mesh, order=int(order))))


def rel_L2_error_scalar(p_rb, p_fom, Qspace, mesh, eps=1e-14, order=5):
    num = abs_L2_error_scalar(p_rb, p_fom, Qspace, mesh, order=order)
    den = L2_norm_scalar_on_mesh(p_fom, Qspace, mesh, order=order)
    return num / max(den, eps)


def rel_L2_error_vector(u_rb, u_fom, Vspace, mesh, eps=1e-14, order=5):
    num = abs_L2_error_vector(u_rb, u_fom, Vspace, mesh, order=order)
    den = L2_norm_vector_on_mesh(u_fom, Vspace, mesh, order=order)
    return num / max(den, eps)


def Hdiv_norm_vector_on_mesh(u_vec, Vspace, mesh, order=5):
    g = GridFunction(Vspace)
    g.vec.FV().NumPy()[:] = u_vec
    val = Integrate(InnerProduct(g, g) + div(g) * div(g), mesh, order=int(order))
    return float(sqrt(val))


def abs_Hdiv_error_vector(u_rb, u_fom, Vspace, mesh, order=5):
    g1 = GridFunction(Vspace)
    g1.vec.FV().NumPy()[:] = u_rb
    g2 = GridFunction(Vspace)
    g2.vec.FV().NumPy()[:] = u_fom
    diff_val = g2 - g1
    diff_div = div(g2) - div(g1)
    val = Integrate(InnerProduct(diff_val, diff_val) + diff_div * diff_div, mesh, order=int(order))
    return float(sqrt(val))


def rel_Hdiv_error_vector(u_rb, u_fom, Vspace, mesh, eps=1e-14, order=5):
    num = abs_Hdiv_error_vector(u_rb, u_fom, Vspace, mesh, order=order)
    den = Hdiv_norm_vector_on_mesh(u_fom, Vspace, mesh, order=order)
    return num / max(den, eps)


def KHdiv_norm_vector_on_mesh(u_vec, Vspace, mesh, Kinv_cf, order=5):
    g = GridFunction(Vspace)
    g.vec.FV().NumPy()[:] = u_vec
    val = Integrate(Kinv_cf * InnerProduct(g, g) + div(g) * div(g), mesh, order=int(order))
    return float(sqrt(val))


def abs_KHdiv_error_vector(u_rb, u_fom, Vspace, mesh, Kinv_cf, order=5):
    g1 = GridFunction(Vspace)
    g1.vec.FV().NumPy()[:] = u_rb
    g2 = GridFunction(Vspace)
    g2.vec.FV().NumPy()[:] = u_fom
    diff_val = g2 - g1
    diff_div = div(g2) - div(g1)
    val = Integrate(Kinv_cf * InnerProduct(diff_val, diff_val) + diff_div * diff_div, mesh, order=int(order))
    return float(sqrt(val))


def rel_KHdiv_error_vector(u_rb, u_fom, Vspace, mesh, Kinv_cf, eps=1e-14, order=5):
    num = abs_KHdiv_error_vector(u_rb, u_fom, Vspace, mesh, Kinv_cf, order=order)
    den = KHdiv_norm_vector_on_mesh(u_fom, Vspace, mesh, Kinv_cf, order=order)
    return num / max(den, eps)

# ============================================================
# Full evaluation
# ============================================================
def online_time_metrics_both_full(mesh, V, Q, Y, cfg: Config, f_base, model, model_c, mu, fom_cache_entry=None):
    mu = (float(mu[0]), float(mu[1]))

    if fom_cache_entry is None:
        hu, hp = solve_fom_history_for_test_full(mesh, V, Q, Y, f_base, cfg, mu)
    else:
        hu = fom_cache_entry["hu"]
        hp = fom_cache_entry["hp"]

    nsteps = hu.shape[0]

    Zb = rom_predictor_rollout(model, mu, nsteps=nsteps, p0=cfg.p_init)
    Zc = online_rollout_transient_ROM_CoSTA_shared(model_c, mu, nsteps=nsteps, p0=cfg.p_init)

    r_u = model["r_u"]

    K_cf = permeability_cf(mu, cfg)
    Kinv_cf = 1.0 / K_cf
    int_order = int(cfg.error_integration_order)

    metrics = {}

    def alloc(prefixes, kinds):
        for pfx in prefixes:
            for kind in kinds:
                metrics[f"{pfx}_{kind}"] = np.empty(nsteps, dtype=float)

    alloc(
        prefixes=[
            "baseline_u_l2", "costa_u_l2",
            "baseline_p_l2", "costa_p_l2",
            "baseline_u_hdiv", "costa_u_hdiv",
            "baseline_u_khdiv", "costa_u_khdiv",
        ],
        kinds=["abs", "rel", "scaled"]
    )

    u_l2_ref = np.empty(nsteps, dtype=float)
    p_l2_ref = np.empty(nsteps, dtype=float)
    u_hdiv_ref = np.empty(nsteps, dtype=float)
    u_khdiv_ref = np.empty(nsteps, dtype=float)

    max_u_l2_ref = 0.0
    max_p_l2_ref = 0.0
    max_u_hdiv_ref = 0.0
    max_u_khdiv_ref = 0.0
    for n in range(nsteps):
        n_u_l2 = L2_norm_vector_on_mesh(hu[n, :], V, mesh, order=int_order)
        n_p_l2 = L2_norm_scalar_on_mesh(hp[n, :], Q, mesh, order=int_order)
        n_u_hdiv = Hdiv_norm_vector_on_mesh(hu[n, :], V, mesh, order=int_order)
        n_u_khdiv = KHdiv_norm_vector_on_mesh(hu[n, :], V, mesh, Kinv_cf, order=int_order)

        u_l2_ref[n] = n_u_l2
        p_l2_ref[n] = n_p_l2
        u_hdiv_ref[n] = n_u_hdiv
        u_khdiv_ref[n] = n_u_khdiv

        max_u_l2_ref = max(max_u_l2_ref, n_u_l2)
        max_p_l2_ref = max(max_p_l2_ref, n_p_l2)
        max_u_hdiv_ref = max(max_u_hdiv_ref, n_u_hdiv)
        max_u_khdiv_ref = max(max_u_khdiv_ref, n_u_khdiv)

    eps = 1e-14
    max_u_l2_ref = max(max_u_l2_ref, eps)
    max_p_l2_ref = max(max_p_l2_ref, eps)
    max_u_hdiv_ref = max(max_u_hdiv_ref, eps)
    max_u_khdiv_ref = max(max_u_khdiv_ref, eps)

    for n in range(nsteps):
        u_fom = hu[n, :]
        p_fom = hp[n, :]

        zb = Zb[n + 1, :]
        ub = model["V_u_stab"] @ zb[:r_u]
        pb = model["V_p"] @ zb[r_u:]

        zc = Zc[n + 1, :]
        uc = model_c["V_u_stab"] @ zc[:r_u]
        pc = model_c["V_p"] @ zc[r_u:]

        bu_l2_abs = abs_L2_error_vector(ub, u_fom, V, mesh, order=int_order)
        cu_l2_abs = abs_L2_error_vector(uc, u_fom, V, mesh, order=int_order)
        bp_l2_abs = abs_L2_error_scalar(pb, p_fom, Q, mesh, order=int_order)
        cp_l2_abs = abs_L2_error_scalar(pc, p_fom, Q, mesh, order=int_order)

        bu_hdiv_abs = abs_Hdiv_error_vector(ub, u_fom, V, mesh, order=int_order)
        cu_hdiv_abs = abs_Hdiv_error_vector(uc, u_fom, V, mesh, order=int_order)

        bu_khdiv_abs = abs_KHdiv_error_vector(ub, u_fom, V, mesh, Kinv_cf, order=int_order)
        cu_khdiv_abs = abs_KHdiv_error_vector(uc, u_fom, V, mesh, Kinv_cf, order=int_order)

        metrics["baseline_u_l2_abs"][n] = bu_l2_abs
        metrics["costa_u_l2_abs"][n] = cu_l2_abs
        metrics["baseline_p_l2_abs"][n] = bp_l2_abs
        metrics["costa_p_l2_abs"][n] = cp_l2_abs

        metrics["baseline_u_hdiv_abs"][n] = bu_hdiv_abs
        metrics["costa_u_hdiv_abs"][n] = cu_hdiv_abs
        metrics["baseline_u_khdiv_abs"][n] = bu_khdiv_abs
        metrics["costa_u_khdiv_abs"][n] = cu_khdiv_abs

        den_u_l2 = max(u_l2_ref[n], eps)
        den_p_l2 = max(p_l2_ref[n], eps)
        den_u_hdiv = max(u_hdiv_ref[n], eps)
        den_u_khdiv = max(u_khdiv_ref[n], eps)

        metrics["baseline_u_l2_rel"][n] = bu_l2_abs / den_u_l2
        metrics["costa_u_l2_rel"][n] = cu_l2_abs / den_u_l2
        metrics["baseline_p_l2_rel"][n] = bp_l2_abs / den_p_l2
        metrics["costa_p_l2_rel"][n] = cp_l2_abs / den_p_l2

        metrics["baseline_u_hdiv_rel"][n] = bu_hdiv_abs / den_u_hdiv
        metrics["costa_u_hdiv_rel"][n] = cu_hdiv_abs / den_u_hdiv

        metrics["baseline_u_khdiv_rel"][n] = bu_khdiv_abs / den_u_khdiv
        metrics["costa_u_khdiv_rel"][n] = cu_khdiv_abs / den_u_khdiv

        metrics["baseline_u_l2_scaled"][n] = bu_l2_abs / max_u_l2_ref
        metrics["costa_u_l2_scaled"][n] = cu_l2_abs / max_u_l2_ref
        metrics["baseline_p_l2_scaled"][n] = bp_l2_abs / max_p_l2_ref
        metrics["costa_p_l2_scaled"][n] = cp_l2_abs / max_p_l2_ref

        metrics["baseline_u_hdiv_scaled"][n] = bu_hdiv_abs / max_u_hdiv_ref
        metrics["costa_u_hdiv_scaled"][n] = cu_hdiv_abs / max_u_hdiv_ref

        metrics["baseline_u_khdiv_scaled"][n] = bu_khdiv_abs / max_u_khdiv_ref
        metrics["costa_u_khdiv_scaled"][n] = cu_khdiv_abs / max_u_khdiv_ref

    out = {}
    for name, arr in metrics.items():
        out[f"{name}_maxt"] = float(np.max(arr))
        out[f"{name}_l2t"] = float(np.sqrt(np.mean(arr ** 2)))
    return out
def costa_fit_shared_lowram_from_Zref(model, mu_train, cfg: Config, Zref_list: List[np.ndarray]):
    """
    DNN-based CoSTA fit.
    Keeps the ROM setting, but replaces the linear ridge correction
    by a predictor-to-correction neural map.
    """
    model = dict(model)
    model["costa_use_mu_features"] = bool(cfg.costa_use_mu_features)
    model["costa_use_time_feature"] = bool(cfg.costa_use_time_feature)

    costa_mode = str(getattr(cfg, "costa_mode", "dnn")).strip().lower()
    if costa_mode == "dnn":
        model_c = train_costa_dnn_from_Zref(
            model=model,
            mu_train=mu_train,
            cfg=cfg,
            Zref_list=Zref_list,
        )
    elif costa_mode == "ridge":
        model_c = train_costa_ridge_from_Zref(
            model=model,
            mu_train=mu_train,
            cfg=cfg,
            Zref_list=Zref_list,
        )
    else:
        raise ValueError(f"Unknown costa_mode={costa_mode!r}. Expected 'dnn' or 'ridge'.")
    return model_c
# ============================================================
# Full leaf builder
# ============================================================
def build_leaf_full_core(
    cfg: Config,
    leaf: PseudoLeaf,
    mu_train: List[Tuple[float, float]],
    Su: np.ndarray,
    Sp: np.ndarray,
):
    mesh = Mesh(unit_square.GenerateMesh(maxh=cfg.maxh))
    V = HDiv(mesh, order=cfg.order, dirichlet=cfg.Gamma_N)
    Q = L2(mesh, order=cfg.order - 1)
    Y = FESpace([V, Q])

    g1 = GridFunction(Q)
    g1.Set(CoefficientFunction(1.0))
    p1_full = g1.vec.FV().NumPy().copy()

    f_base = forcing_base_cf(cfg)

    mu_mid = (0.5 * (cfg.a_mu + cfg.b_mu), 0.5 * (cfg.a_mu + cfg.b_mu))
    _, _, Mp_mat, fq_vec = assemble_sparse_blocks(
        mu_mid, V, Q, mesh, f_base, cfg.k0, cfg.delta, cfg.w
    )
    fq_np = fq_vec.FV().NumPy().copy()

    if len(mu_train) == 0:
        return None

    ax, bx, ay, by = leaf.ax, leaf.bx, leaf.ay, leaf.by
    mus = np.asarray(mu_train, dtype=float)
    mu_ref = tuple(np.mean(mus, axis=0))
    Muu_ref_mat, _, _, _ = assemble_sparse_blocks(
        mu_ref, V, Q, mesh, f_base, cfg.k0, cfg.delta, cfg.w
    )

    nsteps_fine = int(round(cfg.T_snap / cfg.dt))
    se = int(cfg.store_every)

    if nsteps_fine % se != 0:
        raise RuntimeError(f"nsteps_fine={nsteps_fine} not divisible by store_every={se}")

    keep_steps = nsteps_fine // se
    Ns = len(mu_train)
    P_cap_leaf = max_admissible_poly_degree(Ns)

    tau_min = float(min(cfg.tau_list))

    tmp_p = weighted_pod_operator_basis(
        Sp, Mp_mat, cfg.tau_list,
        oversample=10, niter=1, tol=1e-7, maxit=150,
        max_modes=250,
    )

    Vp_by_tau = {float(tau): Vtau for tau, (Vtau, _sfull) in tmp_p.items()}
    s_p = next(iter(tmp_p.values()))[1].copy()

    rp_by_tau = {
        str(float(tau)): int(pick_r_from_svals(s_p, float(tau)))
        for tau in cfg.tau_list
    }
    rp_max = max(rp_by_tau.values())
    Vp_max = Vp_by_tau[tau_min]

    S_sup_all = compute_supremizers_for_pressure_basis(
        V, Q, Vp_max[:, :rp_max], mu_ref, cfg
    )

    pod_mode = str(cfg.pod_mode).strip()

    if pod_mode == "dense_weight":
        Muu_ref_dense = np.array(Muu_ref_mat.ToDense())
        Su64 = np.asarray(Su, dtype=np.float64, order="F")

        L_u, Uhat_u, s_u = weighted_pod_svd_data_dense(Su64, Muu_ref_dense)

        Vu_by_tau = {}
        for tau in cfg.tau_list:
            Vu, _ = weighted_pod_basis_from_svd(L_u, Uhat_u, s_u, float(tau))
            Vu_by_tau[float(tau)] = Vu

    elif pod_mode == "snap_weights":
        tmp_u = weighted_pod_operator_basis(
            Su, Muu_ref_mat, cfg.tau_list,
            oversample=10, niter=1, tol=1e-7, maxit=150,
            max_modes=250,
        )
        Vu_by_tau = {float(tau): Vtau for tau, (Vtau, _sfull) in tmp_u.items()}
        s_u = next(iter(tmp_u.values()))[1].copy()

    elif pod_mode == "L2":
        tmp_u = l2_pod_basis_by_tau(Su, cfg.tau_list)
        Vu_by_tau = {float(tau): Vtau for tau, (Vtau, _sfull) in tmp_u.items()}
        s_u = next(iter(tmp_u.values()))[1].copy()

    else:
        raise ValueError(f"Unknown pod_mode={cfg.pod_mode!r}")

    ru_raw_by_tau = {
        str(float(tau)): int(pick_r_from_svals(s_u, float(tau)))
        for tau in cfg.tau_list
    }

    models_map = {}
    models_c_map = {}

    # ------------------------------------------------------------
    # Precompute tau-specific payloads once
    # ------------------------------------------------------------
    tau_payloads = {}

    for tau in cfg.tau_list:
        tau = float(tau)

        Vu = Vu_by_tau[tau]
        Vp = Vp_by_tau[tau]

        rp = int(Vp.shape[1])

        Vu_stab = orthonormalize_columns(
            np.column_stack([Vu, S_sup_all[:, :rp]])
        )

        r_u = int(Vu_stab.shape[1])
        r_p = int(rp)

        Br = build_Br_by_forms(mesh, V, Q, Vp, Vu_stab)
        Mpr = reduce_square_mat(Mp_mat, Vp)
        fr_q = project_vec(Vp, fq_np)

        Mpr_fac = cho_factor(Mpr, lower=True, check_finite=False)

        xv = Mp_mat.CreateColVector()
        yv = Mp_mat.CreateRowVector()

        def project_p_cached(p_full_np, Vp_local=Vp, Mpr_fac_local=Mpr_fac, xv_local=xv, yv_local=yv):
            p64 = np.asarray(p_full_np, dtype=np.float64)
            xv_local.FV().NumPy()[:] = p64
            Mp_mat.Mult(xv_local, yv_local)
            rhs = Vp_local.T @ yv_local.FV().NumPy()
            return cho_solve(Mpr_fac_local, rhs, check_finite=False)

        p0_r = project_p_cached(float(cfg.p_init) * p1_full)

        Zref_list = []
        for mu_idx in range(Ns):
            Zref = build_Zref_fine_from_coarse_snapshots(
                Su_coarse=Su,
                Sp_coarse=Sp,
                mu_idx=mu_idx,
                keep_steps=keep_steps,
                nsteps_fine=nsteps_fine,
                store_every=se,
                Vu_stab=Vu_stab,
                Vp=Vp,
                project_p_cached=project_p_cached,
                p0_r=p0_r,
            )
            Zref_list.append(Zref)

        Muu_rs = np.empty((Ns, r_u, r_u))
        for i, mu in enumerate(mu_train):
            Muu_i, _, _, _ = assemble_sparse_blocks(
                mu, V, Q, mesh, f_base, cfg.k0, cfg.delta, cfg.w
            )
            Muu_rs[i, :, :] = reduce_square_mat(Muu_i, Vu_stab)

        tau_payloads[tau] = {
            "Vu_stab": Vu_stab,
            "Vp": Vp,
            "r_u": r_u,
            "r_p": r_p,
            "Br": Br,
            "Mpr": Mpr,
            "fr_q": fr_q,
            "p0_r": p0_r,
            "Zref_list": Zref_list,
            "Muu_rs": Muu_rs,
            "nsteps_fine": nsteps_fine,
            "store_every": se,
            "mu_ref": [float(mu_ref[0]), float(mu_ref[1])],
        }

    # ------------------------------------------------------------
    # Build all (tau, P) combinations
    # Parallel only when cfg.costa_parallel == True
    # This is safe because in the multi-leaf case the outer layer
    # already parallelizes across leaves.
    # ------------------------------------------------------------
    tauP_tasks = []
    cfg_dict_local = asdict(cfg)
    leaf_dict_local = leaf.as_dict()

    for tau in cfg.tau_list:
        tau = float(tau)
        tau_payload = tau_payloads[tau]
        for Pdeg_req in cfg.P_list:
            tauP_tasks.append(
                (cfg_dict_local, tau, int(Pdeg_req), leaf_dict_local, mu_train, tau_payload)
            )

    if cfg.costa_parallel and len(tauP_tasks) > 1:
        import multiprocessing as mp
        nproc_tauP = min(len(tauP_tasks), max(1, cfg.nproc_regions))
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=nproc_tauP) as pool:
            tauP_results = list(
                pool.imap_unordered(worker_build_one_tauP, tauP_tasks, chunksize=1)
            )
    else:
        tauP_results = [worker_build_one_tauP(task) for task in tauP_tasks]

    for item in tauP_results:
        comb = item["comb"]
        models_map[comb] = item["model"]
        models_c_map[comb] = item["model_c"]

    pod_meta = {
        "leaf_id": leaf.leaf_id,
        "depth": int(leaf.depth),
        "bounds": {"ax": ax, "bx": bx, "ay": ay, "by": by},
        "pod_mode_u": str(cfg.pod_mode),
        "u_svals_full": np.asarray(s_u).tolist(),
        "p_svals_full": np.asarray(s_p).tolist(),
        "u_raw_r_by_tau": ru_raw_by_tau,
        "p_r_by_tau": rp_by_tau,
        "n_mu_train": int(Ns),
        "P_cap_leaf": int(P_cap_leaf),
    }

    return (leaf.leaf_id, models_map, models_c_map, pod_meta)
def worker_build_leaf_full(args):

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    cfg_dict, leaf_dict, mu_train, Su_leaf, Sp_leaf = args

    cfg = Config(**cfg_dict)
    leaf = PseudoLeaf(**leaf_dict)

    return build_leaf_full_core(
        cfg=cfg,
        leaf=leaf,
        mu_train=mu_train,
        Su=Su_leaf,
        Sp=Sp_leaf,
    )
# ============================================================
# Parallel evaluation globals
# ============================================================
_EVAL_CFG = None
_EVAL_PARTITION_MODE = None
_EVAL_PARTITION_ROOT_ID = None
_EVAL_PARTITION_NODES = None
_EVAL_LEAF_MODELS = None
_EVAL_LEAF_MODELS_C = None
_EVAL_FOM_CACHE = None
_EVAL_MESH = None
_EVAL_V = None
_EVAL_Q = None
_EVAL_Y = None
_EVAL_FBASE = None

def _eval_worker_init(cfg_dict, partition_payload, leaf_models, leaf_models_c, fom_cache):
    global _EVAL_CFG, _EVAL_PARTITION_MODE, _EVAL_PARTITION_ROOT_ID, _EVAL_PARTITION_NODES
    global _EVAL_LEAF_MODELS, _EVAL_LEAF_MODELS_C, _EVAL_FOM_CACHE
    global _EVAL_MESH, _EVAL_V, _EVAL_Q, _EVAL_Y, _EVAL_FBASE

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    _EVAL_CFG = Config(**cfg_dict)
    _EVAL_PARTITION_MODE = "rect"
    _EVAL_PARTITION_ROOT_ID = None
    _EVAL_PARTITION_NODES = partition_payload["leaves"]

    _EVAL_LEAF_MODELS = leaf_models
    _EVAL_LEAF_MODELS_C = leaf_models_c
    _EVAL_FOM_CACHE = fom_cache

    _EVAL_MESH = Mesh(unit_square.GenerateMesh(maxh=_EVAL_CFG.maxh))
    _EVAL_V = HDiv(_EVAL_MESH, order=_EVAL_CFG.order, dirichlet=_EVAL_CFG.Gamma_N)
    _EVAL_Q = L2(_EVAL_MESH, order=_EVAL_CFG.order - 1)
    _EVAL_Y = FESpace([_EVAL_V, _EVAL_Q])

    _EVAL_FBASE = forcing_base_cf(_EVAL_CFG)
def _eval_one_mu(mu):
    global _EVAL_CFG, _EVAL_PARTITION_NODES
    global _EVAL_LEAF_MODELS, _EVAL_LEAF_MODELS_C, _EVAL_FOM_CACHE
    global _EVAL_MESH, _EVAL_V, _EVAL_Q, _EVAL_Y, _EVAL_FBASE

    mu = (float(mu[0]), float(mu[1]))
    leaf_id = classify_mu_in_rect_partition_dict(mu, _EVAL_PARTITION_NODES)

    model = _EVAL_LEAF_MODELS[leaf_id]
    model_c = _EVAL_LEAF_MODELS_C[leaf_id]

    fom_cache_entry = None
    if _EVAL_FOM_CACHE is not None:
        fom_cache_entry = _EVAL_FOM_CACHE[mu]

    # Use precomputed FOM history/timing when available.
    if fom_cache_entry is not None:
        hu = fom_cache_entry["hu"]
        hp = fom_cache_entry["hp"]
        fom_time = float(fom_cache_entry.get("fom_time", np.nan))
        if not np.isfinite(fom_time):
            fom_time = 0.0
    else:
        t_fom_start = time.time()
        hu, hp = solve_fom_history_for_test_full(_EVAL_MESH, _EVAL_V, _EVAL_Q, _EVAL_Y, _EVAL_FBASE, _EVAL_CFG, mu)
        t_fom_end = time.time()
        fom_time = t_fom_end - t_fom_start

    # Measure ROM (baseline) time
    t_rom_start = time.time()
    Zb = rom_predictor_rollout(model, mu, nsteps=hu.shape[0], p0=_EVAL_CFG.p_init)
    t_rom_end = time.time()
    rom_time = t_rom_end - t_rom_start

    # Measure ROM with CoSTA time
    t_costa_start = time.time()
    Zc = online_rollout_transient_ROM_CoSTA_shared(model_c, mu, nsteps=hu.shape[0], p0=_EVAL_CFG.p_init)
    t_costa_end = time.time()
    costa_time = t_costa_end - t_costa_start

    # Compute speed-ups
    rom_speedup = fom_time / rom_time if rom_time > 0 else float('inf')
    costa_speedup = fom_time / costa_time if costa_time > 0 else float('inf')

    # Compute metrics
    out = online_time_metrics_both_full(
        _EVAL_MESH, _EVAL_V, _EVAL_Q, _EVAL_Y,
        _EVAL_CFG, _EVAL_FBASE, model, model_c, mu,
        fom_cache_entry={"hu": hu, "hp": hp},
    )

    out["r_u"] = int(model["r_u"])
    out["r_p"] = int(model["r_p"])
    out["r_tot"] = int(model["r_u"] + model["r_p"])
    out["fom_time"] = fom_time
    out["rom_time"] = rom_time
    out["costa_time"] = costa_time
    out["rom_speedup"] = rom_speedup
    out["costa_speedup"] = costa_speedup

    return {"mu": [mu[0], mu[1]], "leaf_id": leaf_id, **out}

def parallel_eval_for_comb(
    cfg: Config,
    partition_payload: dict,
    mu_test: List[Tuple[float, float]],
    leaf_models: Dict[str, Dict[str, Any]],
    leaf_models_c: Dict[str, Dict[str, Any]],
    fom_test_cache: Dict[Tuple[float, float], Dict[str, np.ndarray]],
    nproc_eval: int = None,
    chunksize: int = 1,
):
    import multiprocessing as mp

    if nproc_eval is None:
        nproc_eval = max(1, min(len(mu_test), os.cpu_count() or 1))

    cfg_dict = asdict(cfg)

    if nproc_eval <= 1:
        _eval_worker_init(cfg_dict, partition_payload, leaf_models, leaf_models_c, fom_test_cache)
        return [_eval_one_mu(mu) for mu in mu_test]

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=nproc_eval,
        initializer=_eval_worker_init,
        initargs=(cfg_dict, partition_payload, leaf_models, leaf_models_c, fom_test_cache),
    ) as pool:
        out_list = list(pool.imap(_eval_one_mu, mu_test, chunksize=chunksize))

    return out_list
# ============================================================
# FOM test cache
# ============================================================
def worker_fom_test_chunk(args):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    cfg_dict, mu_chunk = args
    cfg = Config(**cfg_dict)

    mesh = Mesh(unit_square.GenerateMesh(maxh=cfg.maxh))
    V = HDiv(mesh, order=cfg.order, dirichlet=cfg.Gamma_N)
    Q = L2(mesh, order=cfg.order - 1)
    Y = FESpace([V, Q])
    f_base = forcing_base_cf(cfg)

    out = []
    for i, mu in enumerate(mu_chunk, start=1):
        print(f"[worker {os.getpid()}] FOM test cache {i}/{len(mu_chunk)} mu={mu}", flush=True)
        t0 = time.time()
        hu, hp = solve_fom_history_for_test_full(mesh, V, Q, Y, f_base, cfg, mu)
        fom_time = time.time() - t0
        out.append({
            "mu": (float(mu[0]), float(mu[1])),
            "hu": hu,
            "hp": hp,
            "fom_time": float(fom_time),
        })

    return out


def build_fom_test_cache(cfg: Config, mu_test: List[Tuple[float, float]]):
    if len(mu_test) == 0:
        return {}

    mesh = Mesh(unit_square.GenerateMesh(maxh=cfg.maxh))
    V = HDiv(mesh, order=cfg.order, dirichlet=cfg.Gamma_N)
    Q = L2(mesh, order=cfg.order - 1)
    Y = FESpace([V, Q])
    f_base = forcing_base_cf(cfg)

    cache = {}
    for i, mu in enumerate(mu_test, start=1):
        print(f"[main] FOM test cache {i}/{len(mu_test)} mu={mu}", flush=True)
        t0 = time.time()
        hu, hp = solve_fom_history_for_test_full(mesh, V, Q, Y, f_base, cfg, mu)
        fom_time = time.time() - t0
        cache[(float(mu[0]), float(mu[1]))] = {
            "hu": hu,
            "hp": hp,
            "fom_time": float(fom_time),
        }

    if len(cache) != len(mu_test):
        raise RuntimeError("FOM test cache build failed: some mu_test entries are missing")

    return cache

# ============================================================
# Summary helpers
# ============================================================
METRIC_SPECS = [
    ("u_l2_abs", "u L2 abs"),
    ("u_l2_rel", "u L2 rel"),
    ("u_l2_scaled", "u L2 scaled"),
    ("p_l2_abs", "p L2 abs"),
    ("p_l2_rel", "p L2 rel"),
    ("p_l2_scaled", "p L2 scaled"),
    ("u_hdiv_abs", "u Hdiv abs"),
    ("u_hdiv_rel", "u Hdiv rel"),
    ("u_hdiv_scaled", "u Hdiv scaled"),
    ("u_khdiv_abs", "u K^-1-Hdiv abs"),
    ("u_khdiv_rel", "u K^-1-Hdiv rel"),
    ("u_khdiv_scaled", "u K^-1-Hdiv scaled"),
]


def aggregate_metric_over_mu(per_mu: List[dict], prefix: str):
    maxt_vals = np.array([d[f"{prefix}_maxt"] for d in per_mu], dtype=float)
    l2t_vals = np.array([d[f"{prefix}_l2t"] for d in per_mu], dtype=float)
    return {
        "maxt_mean": float(np.mean(maxt_vals)),
        "maxt_max": float(np.max(maxt_vals)),
        "l2t_mean": float(np.mean(l2t_vals)),
        "l2t_max": float(np.max(l2t_vals)),
    }


def summarize_comb_metrics(per_mu: List[dict]):
    summary = {}
    for prefix, _label in METRIC_SPECS:
        summary[f"baseline_{prefix}"] = aggregate_metric_over_mu(per_mu, f"baseline_{prefix}")
        summary[f"costa_{prefix}"] = aggregate_metric_over_mu(per_mu, f"costa_{prefix}")

    # Aggregate online-throughput speed-up metrics
    #   throughput speed-up = (sum FOM online time) / (sum ROM online time)
    # We intentionally avoid averaging per-mu speed-up ratios here.
    fom_times = np.array([d["fom_time"] for d in per_mu], dtype=float)
    rom_times = np.array([d["rom_time"] for d in per_mu], dtype=float)
    costa_times = np.array([d["costa_time"] for d in per_mu], dtype=float)
    fom_time_sum = float(np.sum(fom_times))
    rom_time_sum = float(np.sum(rom_times))
    costa_time_sum = float(np.sum(costa_times))

    summary["fom_time_avg"] = float(np.mean(fom_times))
    summary["rom_time_avg"] = float(np.mean(rom_times))
    summary["costa_time_avg"] = float(np.mean(costa_times))
    summary["fom_time_sum"] = fom_time_sum
    summary["rom_time_sum"] = rom_time_sum
    summary["costa_time_sum"] = costa_time_sum
    summary["rom_speedup_throughput"] = (fom_time_sum / rom_time_sum) if rom_time_sum > 0 else float("inf")
    summary["costa_speedup_throughput"] = (fom_time_sum / costa_time_sum) if costa_time_sum > 0 else float("inf")

    return summary

def append_comb_results(results_tau: dict, summary: dict):
    for prefix, _label in METRIC_SPECS:
        b = summary[f"baseline_{prefix}"]
        c = summary[f"costa_{prefix}"]
        for stat_name, stat_val in b.items():
            results_tau.setdefault(f"baseline_{prefix}_{stat_name}", []).append(float(stat_val))
        for stat_name, stat_val in c.items():
            results_tau.setdefault(f"costa_{prefix}_{stat_name}", []).append(float(stat_val))


def _fmt4(x):
    return f"{x:.3e}"


def summarize_ranks_for_comb(leaf_models: Dict[str, Dict[str, Any]]):
    if not leaf_models:
        return {
            "r_u_avg": 0.0, "r_u_med": 0.0, "r_u_max": 0,
            "r_p_avg": 0.0, "r_p_med": 0.0, "r_p_max": 0,
            "r_tot_avg": 0.0, "r_tot_med": 0.0, "r_tot_max": 0,
        }
    ru = np.array([m["r_u"] for m in leaf_models.values()], dtype=float)
    rp = np.array([m["r_p"] for m in leaf_models.values()], dtype=float)
    rt = ru + rp
    return {
        "r_u_avg": float(np.mean(ru)), "r_u_med": float(np.median(ru)), "r_u_max": int(np.max(ru)),
        "r_p_avg": float(np.mean(rp)), "r_p_med": float(np.median(rp)), "r_p_max": int(np.max(rp)),
        "r_tot_avg": float(np.mean(rt)), "r_tot_med": float(np.median(rt)), "r_tot_max": int(np.max(rt)),
    }


def log_comb_summary(tau: float, Pdeg_req: int, rank_stats: dict, summary: dict):
    log(
        f"[eval done] tau={tau:g} P_req={Pdeg_req}   "
        f"r_u avg/med/max={rank_stats['r_u_avg']:.1f}/{rank_stats['r_u_med']:.1f}/{rank_stats['r_u_max']}   "
        f"r_p avg/med/max={rank_stats['r_p_avg']:.1f}/{rank_stats['r_p_med']:.1f}/{rank_stats['r_p_max']}   "
        f"r_tot avg/med/max={rank_stats['r_tot_avg']:.1f}/{rank_stats['r_tot_med']:.1f}/{rank_stats['r_tot_max']}"
    )
    log(
        f"  Speed-up (online throughput): ROM={summary['rom_speedup_throughput']:.2f}x, "
        f"CoSTA={summary['costa_speedup_throughput']:.2f}x"
    )
    groups = [
        ("u L2", "u_l2_abs", "u_l2_rel", "u_l2_scaled"),
        ("p L2", "p_l2_abs", "p_l2_rel", "p_l2_scaled"),
        ("u Hdiv", "u_hdiv_abs", "u_hdiv_rel", "u_hdiv_scaled"),
        ("u K^-1-Hdiv", "u_khdiv_abs", "u_khdiv_rel", "u_khdiv_scaled"),
    ]

    for title, k_abs, k_rel, k_scl in groups:
        b_abs = summary[f"baseline_{k_abs}"]
        c_abs = summary[f"costa_{k_abs}"]
        b_rel = summary[f"baseline_{k_rel}"]
        c_rel = summary[f"costa_{k_rel}"]
        b_scl = summary[f"baseline_{k_scl}"]
        c_scl = summary[f"costa_{k_scl}"]

        log(
            f"  {title:<13} abs    maxt(mean/max) {_fmt4(b_abs['maxt_mean'])}/{_fmt4(b_abs['maxt_max'])}"
            f" -> {_fmt4(c_abs['maxt_mean'])}/{_fmt4(c_abs['maxt_max'])}   "
            f"l2t(mean/max) {_fmt4(b_abs['l2t_mean'])}/{_fmt4(b_abs['l2t_max'])}"
            f" -> {_fmt4(c_abs['l2t_mean'])}/{_fmt4(c_abs['l2t_max'])}"
        )
        log(
            f"  {'':13} rel    maxt(mean/max) {_fmt4(b_rel['maxt_mean'])}/{_fmt4(b_rel['maxt_max'])}"
            f" -> {_fmt4(c_rel['maxt_mean'])}/{_fmt4(c_rel['maxt_max'])}   "
            f"l2t(mean/max) {_fmt4(b_rel['l2t_mean'])}/{_fmt4(b_rel['l2t_max'])}"
            f" -> {_fmt4(c_rel['l2t_mean'])}/{_fmt4(c_rel['l2t_max'])}"
        )
        log(
            f"  {'':13} scaled maxt(mean/max) {_fmt4(b_scl['maxt_mean'])}/{_fmt4(b_scl['maxt_max'])}"
            f" -> {_fmt4(c_scl['maxt_mean'])}/{_fmt4(c_scl['maxt_max'])}   "
            f"l2t(mean/max) {_fmt4(b_scl['l2t_mean'])}/{_fmt4(b_scl['l2t_max'])}"
            f" -> {_fmt4(c_scl['l2t_mean'])}/{_fmt4(c_scl['l2t_max'])}"
        )
def build_rom_ranks_payload(models_by_leaf: Dict[Tuple[float, int], Dict[str, Dict[str, Any]]]):
    rom_ranks = {}

    for (tau, Pdeg_req), leaf_models in models_by_leaf.items():
        tau_key = str(float(tau))
        P_key = str(int(Pdeg_req))

        rom_ranks.setdefault(tau_key, {})
        rom_ranks[tau_key].setdefault(P_key, {})

        for leaf_id, model in leaf_models.items():
            r_u = int(model["r_u"])
            r_p = int(model["r_p"])
            rom_ranks[tau_key][P_key][leaf_id] = {
                "r_u": r_u,
                "r_p": r_p,
                "r_tot": int(r_u + r_p),
            }

    return rom_ranks


def select_tau_for_plot(tau_req: Optional[float], tau_list: Sequence[float]) -> float:
    taus = [float(t) for t in tau_list]
    if len(taus) == 0:
        raise RuntimeError("tau_list is empty; cannot build plot")
    if tau_req is None:
        return float(taus[0])
    for t in taus:
        if np.isclose(float(t), float(tau_req), rtol=0.0, atol=1e-14):
            return float(t)
    raise ValueError(
        f"Requested plot_tau={tau_req} not found in tau_list={taus}"
    )


def select_two_taus_for_plot(
    tau_pair_req: Optional[Sequence[float]],
    tau_list: Sequence[float],
) -> Tuple[float, float]:
    taus = [float(t) for t in tau_list]
    if len(taus) < 2:
        raise RuntimeError("Need at least two tau values in tau_list for stationary two-tau plot")

    if tau_pair_req is None:
        return float(taus[0]), float(taus[1])

    if len(tau_pair_req) != 2:
        raise ValueError("plot_tau_pair must contain exactly two tau values")

    out = []
    for req in tau_pair_req:
        found = None
        for t in taus:
            if np.isclose(float(t), float(req), rtol=0.0, atol=1e-14):
                found = float(t)
                break
        if found is None:
            raise ValueError(f"Requested tau={req} not found in tau_list={taus}")
        out.append(found)

    if np.isclose(out[0], out[1], rtol=0.0, atol=1e-14):
        raise ValueError(f"plot_tau_pair requires two distinct values, got {out}")

    return float(out[0]), float(out[1])


def select_P_for_plot(P_req: Optional[int], P_list: Sequence[int]) -> int:
    Ps = [int(p) for p in P_list]
    if len(Ps) == 0:
        raise RuntimeError("P_list is empty; cannot build plot")
    if P_req is None:
        return int(Ps[0])
    if int(P_req) not in Ps:
        raise ValueError(
            f"Requested plot_P={P_req} not found in P_list={Ps}"
        )
    return int(P_req)


def time_to_step_index(t: float, dt: float, nsteps: int) -> Tuple[int, float]:
    n = int(round(float(t) / float(dt)))
    n = max(1, min(int(nsteps), n))
    t_used = n * float(dt)
    return n, t_used


def sample_scalar_gridfunction(mesh, gf, nx: int = 160, ny: int = 160) -> np.ndarray:
    vals = np.zeros((ny, nx), dtype=np.float64)
    eps = 1e-10
    xs = np.linspace(eps, 1.0 - eps, nx)
    ys = np.linspace(eps, 1.0 - eps, ny)
    for j, yv in enumerate(ys):
        for i, xv in enumerate(xs):
            vals[j, i] = float(gf(mesh(float(xv), float(yv))))
    return vals


def sample_velocity_magnitude_gridfunction(mesh, gf, nx: int = 160, ny: int = 160) -> np.ndarray:
    vals = np.zeros((ny, nx), dtype=np.float64)
    eps = 1e-10
    xs = np.linspace(eps, 1.0 - eps, nx)
    ys = np.linspace(eps, 1.0 - eps, ny)
    for j, yv in enumerate(ys):
        for i, xv in enumerate(xs):
            v = np.asarray(gf(mesh(float(xv), float(yv))), dtype=np.float64).reshape(-1)
            vals[j, i] = float(np.linalg.norm(v))
    return vals


def sample_velocity_vector_gridfunction(mesh, gf, nx: int = 16, ny: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    u = np.zeros((ny, nx), dtype=np.float64)
    v = np.zeros((ny, nx), dtype=np.float64)
    eps = 1e-10
    xs = np.linspace(eps, 1.0 - eps, nx)
    ys = np.linspace(eps, 1.0 - eps, ny)
    for j, yv in enumerate(ys):
        for i, xv in enumerate(xs):
            uv = np.asarray(gf(mesh(float(xv), float(yv))), dtype=np.float64).reshape(-1)
            u[j, i] = float(uv[0])
            v[j, i] = float(uv[1]) if uv.size > 1 else 0.0
    return u, v


def _draw_pressure_panel_grid(
    pdf: PdfPages,
    timed_fields: List[dict],
    times_used: List[float],
    tau_plot: float,
    P_plot: int,
    mu_plot: Tuple[float, float],
    fom_dofs: int,
    rom_r_tot: int,
    costa_r_tot: int,
):
    row_keys = ["fom", "rom", "rom_diff", "costa", "costa_diff"]
    row_titles = [
        f"FOM ({int(fom_dofs)} DoFs)",
        f"ROM ($r={int(rom_r_tot)}$)",
        "ROM $-$ FOM",
        f"CoSTA-ROM ($r={int(costa_r_tot)}$)",
        "CoSTA-ROM $-$ FOM",
    ]
    nrows = len(row_keys)
    ncols = len(times_used)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.8 * nrows), squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.90, top=0.90, bottom=0.05, wspace=0.01, hspace=0.12)

    def _add_row_colorbar(mappable, ax_anchor):
        pos = ax_anchor.get_position()
        cax = fig.add_axes([
            pos.x1 + 0.012,
            pos.y0 + 0.03 * pos.height,
            0.015,
            0.94 * pos.height,
        ])
        cb = fig.colorbar(mappable, cax=cax)
        cb.ax.tick_params(labelsize=14)

    # Shared contour levels across all pressure solution panels (FOM/ROM/CoSTA, all times).
    sol_keys = ("fom", "rom", "costa")
    sol_global_min = min(float(np.min(tf[k])) for tf in timed_fields for k in sol_keys)
    sol_global_max = max(float(np.max(tf[k])) for tf in timed_fields for k in sol_keys)
    if abs(sol_global_max - sol_global_min) > 1e-14:
        solution_contour_levels = np.linspace(sol_global_min, sol_global_max, 12)
    else:
        solution_contour_levels = None

    # One shared color scale for all pressure solution rows (FOM/ROM/CoSTA), across all times.
    sol_vmin = sol_global_min
    sol_vmax = sol_global_max
    if sol_vmax <= sol_vmin:
        sol_vmax = sol_vmin + 1e-14

    for irow, key in enumerate(row_keys):
        mats = [tf[key] for tf in timed_fields]
        is_diff = key in ("rom_diff", "costa_diff")
        if is_diff:
            # Shared signed pressure-difference scale across both diff rows and all times.
            vmax = max(
                max(float(np.max(np.abs(tf["rom_diff"]))) for tf in timed_fields),
                max(float(np.max(np.abs(tf["costa_diff"]))) for tf in timed_fields),
            )
            if vmax <= 0.0:
                vmax = 1e-14
            vmin = -vmax
            cmap = "RdBu"
        else:
            vmin = sol_vmin
            vmax = sol_vmax
            cmap = "viridis"

        mappable = None
        for icol, t_used in enumerate(times_used):
            ax = axes[irow][icol]
            mat = mats[icol]
            im = ax.imshow(
                mat,
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
            )
            if not is_diff:
                ny, nx = mat.shape
                xs = np.linspace(0.0, 1.0, nx)
                ys = np.linspace(0.0, 1.0, ny)
                X, Y = np.meshgrid(xs, ys)
                if solution_contour_levels is not None:
                    ax.contour(
                        X,
                        Y,
                        mat,
                        levels=solution_contour_levels,
                        colors="k",
                        linewidths=0.5,
                        alpha=0.6,
                    )

            if irow == 0:
                ax.set_title(f"t={t_used:.3f}")
            if icol == 0:
                ax.set_ylabel(row_titles[irow])
            ax.set_xticks([])
            ax.set_yticks([])
            mappable = im

        _add_row_colorbar(mappable, axes[irow][ncols - 1])

    pdf.savefig(fig, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _draw_velocity_panel_grid(
    pdf: PdfPages,
    timed_fields: List[dict],
    times_used: List[float],
    tau_plot: float,
    P_plot: int,
    mu_plot: Tuple[float, float],
    fom_dofs: int,
    rom_r_tot: int,
    costa_r_tot: int,
):
    row_keys = ["fom", "rom", "rom_diff", "costa", "costa_diff"]
    row_titles = [
        f"FOM ({int(fom_dofs)} DoFs)",
        f"ROM ($r={int(rom_r_tot)}$)",
        "ROM $-$ FOM",
        f"CoSTA-ROM ($r={int(costa_r_tot)}$)",
        "CoSTA-ROM $-$ FOM",
    ]
    nrows = len(row_keys)
    ncols = len(times_used)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.8 * nrows), squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.90, top=0.90, bottom=0.05, wspace=0.01, hspace=0.12)

    def _add_row_colorbar(mappable, ax_anchor):
        pos = ax_anchor.get_position()
        cax = fig.add_axes([
            pos.x1 + 0.012,
            pos.y0 + 0.03 * pos.height,
            0.015,
            0.94 * pos.height,
        ])
        cb = fig.colorbar(mappable, cax=cax)
        cb.ax.tick_params(labelsize=14)

    # One shared color scale for all velocity solution rows (FOM/ROM/CoSTA), across all times.
    sol_keys = ("fom", "rom", "costa")
    sol_vmin = min(float(np.min(tf[k]["color"])) for tf in timed_fields for k in sol_keys)
    sol_vmax = max(float(np.max(tf[k]["color"])) for tf in timed_fields for k in sol_keys)
    if sol_vmax <= sol_vmin:
        sol_vmax = sol_vmin + 1e-14

    # One shared color scale for both velocity difference rows, across all times.
    diff_keys = ("rom_diff", "costa_diff")
    diff_abs_max = 0.0
    diff_vec_max = 0.0
    for key in diff_keys:
        for tf in timed_fields:
            cmat = tf[key]["color"]
            diff_abs_max = max(diff_abs_max, float(np.max(np.abs(cmat))))
            um = tf[key]["u"]
            vm = tf[key]["v"]
            diff_vec_max = max(diff_vec_max, float(np.max(np.sqrt(um**2 + vm**2))))
    if diff_abs_max <= 0.0:
        diff_abs_max = 1e-14
    if diff_vec_max <= 0.0:
        diff_vec_max = 1e-14

    # Match the idea used in your paper plotting code: target a visible max arrow length.
    domain_span = 1.0
    target_arrow = 0.18 * domain_span
    diff_scale_shared = max(diff_vec_max / target_arrow, 1e-14)

    for irow, key in enumerate(row_keys):
        colors = [tf[key]["color"] for tf in timed_fields]
        is_diff = key in diff_keys
        if is_diff:
            vmin = -diff_abs_max
            vmax = diff_abs_max
            cmap = "RdBu"
        else:
            vmin = sol_vmin
            vmax = sol_vmax
            cmap = "viridis"

        mappable = None
        for icol, t_used in enumerate(times_used):
            ax = axes[irow][icol]
            u = timed_fields[icol][key]["u"]
            v = timed_fields[icol][key]["v"]
            c = timed_fields[icol][key]["color"]
            quiver_scale = diff_scale_shared if is_diff else 7.5
            quiver_width = 0.0054 if is_diff else 0.0038
            ny, nx = u.shape
            xs = np.linspace(0.0, 1.0, nx)
            ys = np.linspace(0.0, 1.0, ny)
            X, Y = np.meshgrid(xs, ys)
            q = ax.quiver(
                X, Y, u, v, c,
                cmap=cmap,
                clim=(vmin, vmax),
                angles="xy",
                scale_units="xy",
                scale=quiver_scale,
                pivot="mid",
                width=quiver_width,
                headwidth=3.8 if is_diff else 3.1,
                headlength=5.2 if is_diff else 4.0,
            )
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_aspect("equal")
            if irow == 0:
                ax.set_title(f"t={t_used:.3f}")
            if icol == 0:
                ax.set_ylabel(row_titles[irow])
            ax.set_xticks([])
            ax.set_yticks([])
            mappable = q

        _add_row_colorbar(mappable, axes[irow][ncols - 1])

    pdf.savefig(fig, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _draw_stationary_pressure_two_tau(
    out_pdf_pressure_path: str,
    p_fom_grid: np.ndarray,
    rows: List[dict],
    P_plot: int,
    fom_dofs: int,
):
    field_min = min(
        float(np.min(p_fom_grid)),
        float(np.min(rows[0]["p_rom_grid"])),
        float(np.min(rows[1]["p_rom_grid"])),
        float(np.min(rows[0]["p_costa_grid"])),
        float(np.min(rows[1]["p_costa_grid"])),
    )
    field_max = max(
        float(np.max(p_fom_grid)),
        float(np.max(rows[0]["p_rom_grid"])),
        float(np.max(rows[1]["p_rom_grid"])),
        float(np.max(rows[0]["p_costa_grid"])),
        float(np.max(rows[1]["p_costa_grid"])),
    )
    diff_lim_rom = max(
        float(np.max(np.abs(rows[0]["p_diff_grid"]))),
        float(np.max(np.abs(rows[1]["p_diff_grid"]))),
    )
    diff_lim_costa = max(
        float(np.max(np.abs(rows[0]["p_costa_diff_grid"]))),
        float(np.max(np.abs(rows[1]["p_costa_diff_grid"]))),
    )
    diff_lim = max(diff_lim_rom, diff_lim_costa)
    if diff_lim <= 0.0:
        diff_lim = 1e-14

    fig = plt.figure(figsize=(7.6, 15.0))
    gs = fig.add_gridspec(
        5,
        2,
        left=0.08,
        right=0.86,
        bottom=0.06,
        top=0.975,
        hspace=0.14,
        wspace=0.5,
    )
    title_fs = 11

    def _set_bottom_title(ax, text):
        ax.set_title("")
        ax.text(0.5, -0.06, text, transform=ax.transAxes, ha="center", va="top", fontsize=title_fs)

    ax_fem = fig.add_subplot(gs[0, 0])
    ax_rom_1 = fig.add_subplot(gs[1, 0])
    ax_rom_2 = fig.add_subplot(gs[1, 1])
    ax_diff_1 = fig.add_subplot(gs[2, 0])
    ax_diff_2 = fig.add_subplot(gs[2, 1])
    ax_costa_1 = fig.add_subplot(gs[3, 0])
    ax_costa_2 = fig.add_subplot(gs[3, 1])
    ax_cdiff_1 = fig.add_subplot(gs[4, 0])
    ax_cdiff_2 = fig.add_subplot(gs[4, 1])

    im_fom = ax_fem.imshow(
        p_fom_grid, origin="lower", extent=(0.0, 1.0, 0.0, 1.0),
        cmap="viridis", vmin=field_min, vmax=field_max, aspect="equal",
    )
    im_rom_1 = ax_rom_1.imshow(
        rows[0]["p_rom_grid"], origin="lower", extent=(0.0, 1.0, 0.0, 1.0),
        cmap="viridis", vmin=field_min, vmax=field_max, aspect="equal",
    )
    ax_rom_2.imshow(
        rows[1]["p_rom_grid"], origin="lower", extent=(0.0, 1.0, 0.0, 1.0),
        cmap="viridis", vmin=field_min, vmax=field_max, aspect="equal",
    )

    im_diff = ax_diff_1.imshow(
        rows[0]["p_diff_grid"], origin="lower", extent=(0.0, 1.0, 0.0, 1.0),
        cmap="RdBu", vmin=-diff_lim, vmax=diff_lim, aspect="equal",
    )
    ax_diff_2.imshow(
        rows[1]["p_diff_grid"], origin="lower", extent=(0.0, 1.0, 0.0, 1.0),
        cmap="RdBu", vmin=-diff_lim, vmax=diff_lim, aspect="equal",
    )

    im_costa = ax_costa_1.imshow(
        rows[0]["p_costa_grid"], origin="lower", extent=(0.0, 1.0, 0.0, 1.0),
        cmap="viridis", vmin=field_min, vmax=field_max, aspect="equal",
    )
    ax_costa_2.imshow(
        rows[1]["p_costa_grid"], origin="lower", extent=(0.0, 1.0, 0.0, 1.0),
        cmap="viridis", vmin=field_min, vmax=field_max, aspect="equal",
    )

    im_cdiff = ax_cdiff_1.imshow(
        rows[0]["p_costa_diff_grid"], origin="lower", extent=(0.0, 1.0, 0.0, 1.0),
        cmap="RdBu", vmin=-diff_lim, vmax=diff_lim, aspect="equal",
    )
    ax_cdiff_2.imshow(
        rows[1]["p_costa_diff_grid"], origin="lower", extent=(0.0, 1.0, 0.0, 1.0),
        cmap="RdBu", vmin=-diff_lim, vmax=diff_lim, aspect="equal",
    )

    # Keep stationary pressure styling close to the transient version:
    # shared contour levels on solution panels only (not on differences).
    p_sol_arrays = [
        p_fom_grid,
        rows[0]["p_rom_grid"], rows[1]["p_rom_grid"],
        rows[0]["p_costa_grid"], rows[1]["p_costa_grid"],
    ]
    p_sol_min = min(float(np.min(a)) for a in p_sol_arrays)
    p_sol_max = max(float(np.max(a)) for a in p_sol_arrays)
    contour_levels = None
    if abs(p_sol_max - p_sol_min) > 1e-14:
        contour_levels = np.linspace(p_sol_min, p_sol_max, 12)
    if contour_levels is not None:
        ny, nx = p_fom_grid.shape
        xs = np.linspace(0.0, 1.0, nx)
        ys = np.linspace(0.0, 1.0, ny)
        X, Y = np.meshgrid(xs, ys)
        for ax_c, mat_c in [
            (ax_fem, p_fom_grid),
            (ax_rom_1, rows[0]["p_rom_grid"]),
            (ax_rom_2, rows[1]["p_rom_grid"]),
            (ax_costa_1, rows[0]["p_costa_grid"]),
            (ax_costa_2, rows[1]["p_costa_grid"]),
        ]:
            ax_c.contour(
                X,
                Y,
                mat_c,
                levels=contour_levels,
                colors="k",
                linewidths=0.5,
                alpha=0.6,
            )

    _set_bottom_title(ax_fem, f"FOM solution ({int(fom_dofs)} DoFs)")
    _set_bottom_title(
        ax_rom_1,
        rf"ROM solution, $r={rows[0]['r_tot']}, P={int(P_plot)}$",
    )
    _set_bottom_title(
        ax_rom_2,
        rf"ROM solution, $r={rows[1]['r_tot']}, P={int(P_plot)}$",
    )
    _set_bottom_title(
        ax_diff_1,
        rf"ROM $-$ FOM difference, $r={rows[0]['r_tot']}, P={int(P_plot)}$",
    )
    _set_bottom_title(
        ax_diff_2,
        rf"ROM $-$ FOM difference, $r={rows[1]['r_tot']}, P={int(P_plot)}$",
    )
    _set_bottom_title(
        ax_costa_1,
        rf"CoSTA-ROM solution, $r={rows[0]['r_tot_costa']}, P={int(P_plot)}$",
    )
    _set_bottom_title(
        ax_costa_2,
        rf"CoSTA-ROM solution, $r={rows[1]['r_tot_costa']}, P={int(P_plot)}$",
    )
    _set_bottom_title(
        ax_cdiff_1,
        rf"CoSTA-ROM $-$ FOM difference, $r={rows[0]['r_tot_costa']}, P={int(P_plot)}$",
    )
    _set_bottom_title(
        ax_cdiff_2,
        rf"CoSTA-ROM $-$ FOM difference, $r={rows[1]['r_tot_costa']}, P={int(P_plot)}$",
    )

    for ax in [ax_fem, ax_rom_1, ax_rom_2, ax_diff_1, ax_diff_2, ax_costa_1, ax_costa_2, ax_cdiff_1, ax_cdiff_2]:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    p1 = ax_rom_1.get_position()
    p2 = ax_rom_2.get_position()
    p0 = ax_fem.get_position()
    x_center = 0.5 * (p1.x0 + p2.x1)
    ax_fem.set_position([x_center - 0.5 * p1.width, p0.y0, p1.width, p0.height])

    def _add_side_cbar(mappable, ax_anchor):
        pos = ax_anchor.get_position()
        cax = fig.add_axes([
            pos.x1 + 0.012,
            pos.y0 + 0.03 * pos.height,
            0.015,
            0.94 * pos.height,
        ])
        cb = fig.colorbar(mappable, cax=cax)
        cb.ax.tick_params(labelsize=12)

    _add_side_cbar(im_fom, ax_fem)
    _add_side_cbar(im_rom_1, ax_rom_2)
    _add_side_cbar(im_diff, ax_diff_2)
    _add_side_cbar(im_costa, ax_costa_2)
    _add_side_cbar(im_cdiff, ax_cdiff_2)

    os.makedirs(os.path.dirname(out_pdf_pressure_path) or ".", exist_ok=True)
    with PdfPages(out_pdf_pressure_path) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _draw_stationary_velocity_two_tau(
    out_pdf_velocity_path: str,
    rows: List[dict],
    P_plot: int,
    fom_dofs: int,
):
    field_min = min(
        float(np.min(rows[0]["u_fom_mag"])),
        float(np.min(rows[0]["u_rom_mag"])),
        float(np.min(rows[1]["u_rom_mag"])),
        float(np.min(rows[0]["u_costa_mag"])),
        float(np.min(rows[1]["u_costa_mag"])),
    )
    field_max = max(
        float(np.max(rows[0]["u_fom_mag"])),
        float(np.max(rows[0]["u_rom_mag"])),
        float(np.max(rows[1]["u_rom_mag"])),
        float(np.max(rows[0]["u_costa_mag"])),
        float(np.max(rows[1]["u_costa_mag"])),
    )
    # Match transient behavior:
    # color by signed speed difference, but scale arrows by vector-error magnitude.
    rom_signed_0 = rows[0]["u_rom_mag"] - rows[0]["u_fom_mag"]
    rom_signed_1 = rows[1]["u_rom_mag"] - rows[1]["u_fom_mag"]
    costa_signed_0 = rows[0]["u_costa_mag"] - rows[0]["u_fom_mag"]
    costa_signed_1 = rows[1]["u_costa_mag"] - rows[1]["u_fom_mag"]
    diff_abs_max = max(
        float(np.max(np.abs(rom_signed_0))),
        float(np.max(np.abs(rom_signed_1))),
        float(np.max(np.abs(costa_signed_0))),
        float(np.max(np.abs(costa_signed_1))),
    )
    if diff_abs_max <= 0.0:
        diff_abs_max = 1e-14

    diff_vec_max = max(
        float(np.max(rows[0]["u_diff_mag"])),
        float(np.max(rows[1]["u_diff_mag"])),
        float(np.max(rows[0]["u_costa_diff_mag"])),
        float(np.max(rows[1]["u_costa_diff_mag"])),
    )
    if diff_vec_max <= 0.0:
        diff_vec_max = 1e-14
    domain_span = 1.0
    target_arrow = 0.18 * domain_span
    diff_scale_shared = max(diff_vec_max / target_arrow, 1e-14)

    fig = plt.figure(figsize=(7.6, 15.0))
    gs = fig.add_gridspec(
        5,
        2,
        left=0.08,
        right=0.86,
        bottom=0.06,
        top=0.975,
        hspace=0.14,
        wspace=0.5,
    )
    title_fs = 11

    def _set_bottom_title(ax, text):
        ax.set_title("")
        ax.text(0.5, -0.06, text, transform=ax.transAxes, ha="center", va="top", fontsize=title_fs)

    ax_fem = fig.add_subplot(gs[0, 0])
    ax_rom_1 = fig.add_subplot(gs[1, 0])
    ax_rom_2 = fig.add_subplot(gs[1, 1])
    ax_diff_1 = fig.add_subplot(gs[2, 0])
    ax_diff_2 = fig.add_subplot(gs[2, 1])
    ax_costa_1 = fig.add_subplot(gs[3, 0])
    ax_costa_2 = fig.add_subplot(gs[3, 1])
    ax_cdiff_1 = fig.add_subplot(gs[4, 0])
    ax_cdiff_2 = fig.add_subplot(gs[4, 1])

    ny, nx = rows[0]["u_fom_u"].shape
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(xs, ys)

    def _vel_quiver(ax, ux, uy, mag):
        q = ax.quiver(
            X,
            Y,
            ux,
            uy,
            mag,
            cmap="viridis",
            angles="xy",
            scale_units="xy",
            scale=7.5,
            pivot="mid",
            width=0.0038,
            headwidth=3.1,
            headlength=4.0,
        )
        q.set_clim(field_min, field_max)
        return q

    def _err_quiver(ax, du, dv, color_signed, emag):
        q = ax.quiver(
            X,
            Y,
            du,
            dv,
            color_signed,
            cmap="RdBu",
            angles="xy",
            scale_units="xy",
            scale=diff_scale_shared,
            pivot="mid",
            width=0.0054,
            headwidth=3.8,
            headlength=5.2,
        )
        q.set_clim(-diff_abs_max, diff_abs_max)
        return q

    q_fem = _vel_quiver(ax_fem, rows[0]["u_fom_u"], rows[0]["u_fom_v"], rows[0]["u_fom_mag"])
    q_rom = _vel_quiver(ax_rom_1, rows[0]["u_rom_u"], rows[0]["u_rom_v"], rows[0]["u_rom_mag"])
    _vel_quiver(ax_rom_2, rows[1]["u_rom_u"], rows[1]["u_rom_v"], rows[1]["u_rom_mag"])

    q_diff = _err_quiver(
        ax_diff_1,
        rows[0]["u_diff_u"],
        rows[0]["u_diff_v"],
        rom_signed_0,
        rows[0]["u_diff_mag"],
    )
    _err_quiver(
        ax_diff_2,
        rows[1]["u_diff_u"],
        rows[1]["u_diff_v"],
        rom_signed_1,
        rows[1]["u_diff_mag"],
    )
    q_costa = _vel_quiver(ax_costa_1, rows[0]["u_costa_u"], rows[0]["u_costa_v"], rows[0]["u_costa_mag"])
    _vel_quiver(ax_costa_2, rows[1]["u_costa_u"], rows[1]["u_costa_v"], rows[1]["u_costa_mag"])

    q_cdiff = _err_quiver(
        ax_cdiff_1,
        rows[0]["u_costa_diff_u"],
        rows[0]["u_costa_diff_v"],
        costa_signed_0,
        rows[0]["u_costa_diff_mag"],
    )
    _err_quiver(
        ax_cdiff_2,
        rows[1]["u_costa_diff_u"],
        rows[1]["u_costa_diff_v"],
        costa_signed_1,
        rows[1]["u_costa_diff_mag"],
    )

    _set_bottom_title(ax_fem, f"FOM solution ({int(fom_dofs)} DoFs)")
    _set_bottom_title(
        ax_rom_1,
        rf"ROM solution, $r={rows[0]['r_tot']}, P={int(P_plot)}$",
    )
    _set_bottom_title(
        ax_rom_2,
        rf"ROM solution, $r={rows[1]['r_tot']}, P={int(P_plot)}$",
    )
    _set_bottom_title(
        ax_diff_1,
        rf"ROM $-$ FOM difference, $r={rows[0]['r_tot']}, P={int(P_plot)}$",
    )
    _set_bottom_title(
        ax_diff_2,
        rf"ROM $-$ FOM difference, $r={rows[1]['r_tot']}, P={int(P_plot)}$",
    )
    _set_bottom_title(
        ax_costa_1,
        rf"CoSTA-ROM solution, $r={rows[0]['r_tot_costa']}, P={int(P_plot)}$",
    )
    _set_bottom_title(
        ax_costa_2,
        rf"CoSTA-ROM solution, $r={rows[1]['r_tot_costa']}, P={int(P_plot)}$",
    )
    _set_bottom_title(
        ax_cdiff_1,
        rf"CoSTA-ROM $-$ FOM difference, $r={rows[0]['r_tot_costa']}, P={int(P_plot)}$",
    )
    _set_bottom_title(
        ax_cdiff_2,
        rf"CoSTA-ROM $-$ FOM difference, $r={rows[1]['r_tot_costa']}, P={int(P_plot)}$",
    )

    for ax in [ax_fem, ax_rom_1, ax_rom_2, ax_diff_1, ax_diff_2, ax_costa_1, ax_costa_2, ax_cdiff_1, ax_cdiff_2]:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    p1 = ax_rom_1.get_position()
    p2 = ax_rom_2.get_position()
    p0 = ax_fem.get_position()
    x_center = 0.5 * (p1.x0 + p2.x1)
    ax_fem.set_position([x_center - 0.5 * p1.width, p0.y0, p1.width, p0.height])

    def _add_side_cbar(mappable, ax_anchor):
        pos = ax_anchor.get_position()
        cax = fig.add_axes([
            pos.x1 + 0.012,
            pos.y0 + 0.03 * pos.height,
            0.015,
            0.94 * pos.height,
        ])
        cb = fig.colorbar(mappable, cax=cax)
        cb.ax.tick_params(labelsize=12)

    _add_side_cbar(q_fem, ax_fem)
    _add_side_cbar(q_rom, ax_rom_2)
    _add_side_cbar(q_diff, ax_diff_2)
    _add_side_cbar(q_costa, ax_costa_2)
    _add_side_cbar(q_cdiff, ax_cdiff_2)

    os.makedirs(os.path.dirname(out_pdf_velocity_path) or ".", exist_ok=True)
    with PdfPages(out_pdf_velocity_path) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_stationary_two_tau_solution_comparison_pdf(
    cfg: Config,
    partition_payload: dict,
    models_by_leaf: Dict[Tuple[float, int], Dict[str, Dict[str, Any]]],
    models_c_by_leaf: Dict[Tuple[float, int], Dict[str, Dict[str, Any]]],
    P_plot: int,
    tau_pair: Tuple[float, float],
    mu_plot: Tuple[float, float],
    out_pdf_pressure_path: str,
    out_pdf_velocity_path: str,
):
    tau_a = float(tau_pair[0])
    tau_b = float(tau_pair[1])
    comb_a = (tau_a, int(P_plot))
    comb_b = (tau_b, int(P_plot))

    if comb_a not in models_by_leaf:
        raise RuntimeError(f"Combination (tau={tau_a}, P={P_plot}) not present in models_by_leaf")
    if comb_b not in models_by_leaf:
        raise RuntimeError(f"Combination (tau={tau_b}, P={P_plot}) not present in models_by_leaf")
    if comb_a not in models_c_by_leaf:
        raise RuntimeError(f"Combination (tau={tau_a}, P={P_plot}) not present in models_c_by_leaf")
    if comb_b not in models_c_by_leaf:
        raise RuntimeError(f"Combination (tau={tau_b}, P={P_plot}) not present in models_c_by_leaf")

    leaf_id = classify_mu_in_partition_payload(mu_plot, partition_payload, cfg)
    if leaf_id not in models_by_leaf[comb_a]:
        raise RuntimeError(f"Leaf {leaf_id} missing baseline model for (tau={tau_a}, P={P_plot})")
    if leaf_id not in models_by_leaf[comb_b]:
        raise RuntimeError(f"Leaf {leaf_id} missing baseline model for (tau={tau_b}, P={P_plot})")
    if leaf_id not in models_c_by_leaf[comb_a]:
        raise RuntimeError(f"Leaf {leaf_id} missing CoSTA model for (tau={tau_a}, P={P_plot})")
    if leaf_id not in models_c_by_leaf[comb_b]:
        raise RuntimeError(f"Leaf {leaf_id} missing CoSTA model for (tau={tau_b}, P={P_plot})")

    model_a = models_by_leaf[comb_a][leaf_id]
    model_b = models_by_leaf[comb_b][leaf_id]
    model_c_a = models_c_by_leaf[comb_a][leaf_id]
    model_c_b = models_c_by_leaf[comb_b][leaf_id]

    mesh = Mesh(unit_square.GenerateMesh(maxh=cfg.maxh))
    V = HDiv(mesh, order=cfg.order, dirichlet=cfg.Gamma_N)
    Q = L2(mesh, order=cfg.order - 1)
    Y = FESpace([V, Q])
    f_base = forcing_base_cf(cfg)
    fom_dofs = int(V.ndof + Q.ndof)

    hu_fom, hp_fom = solve_fom_history_for_test_full(mesh, V, Q, Y, f_base, cfg, mu_plot)
    nsteps = hu_fom.shape[0]
    i_fom = nsteps - 1
    i_rom = nsteps

    u_fom = hu_fom[i_fom, :].astype(np.float64, copy=False)
    p_fom = hp_fom[i_fom, :].astype(np.float64, copy=False)

    gfu = GridFunction(V)
    gfp = GridFunction(Q)

    gfp.vec.FV().NumPy()[:] = p_fom
    p_fom_grid = sample_scalar_gridfunction(mesh, gfp)

    gfu.vec.FV().NumPy()[:] = u_fom
    u_fom_u, u_fom_v = sample_velocity_vector_gridfunction(mesh, gfu)
    u_fom_mag = np.sqrt(u_fom_u**2 + u_fom_v**2)

    rows = []
    for tau_use, model, model_c in [
        (tau_a, model_a, model_c_a),
        (tau_b, model_b, model_c_b),
    ]:
        Zb = rom_predictor_rollout(model, mu_plot, nsteps=nsteps, p0=cfg.p_init)
        Zc = online_rollout_transient_ROM_CoSTA_shared(model_c, mu_plot, nsteps=nsteps, p0=cfg.p_init)
        r_u = int(model["r_u"])
        r_uc = int(model_c["r_u"])
        ub = (model["V_u_stab"] @ Zb[i_rom, :r_u]).astype(np.float64, copy=False)
        pb = (model["V_p"] @ Zb[i_rom, r_u:]).astype(np.float64, copy=False)
        uc = (model_c["V_u_stab"] @ Zc[i_rom, :r_uc]).astype(np.float64, copy=False)
        pc = (model_c["V_p"] @ Zc[i_rom, r_uc:]).astype(np.float64, copy=False)

        gfp.vec.FV().NumPy()[:] = pb
        p_rom_grid = sample_scalar_gridfunction(mesh, gfp)
        gfp.vec.FV().NumPy()[:] = (pb - p_fom)
        p_diff_grid = sample_scalar_gridfunction(mesh, gfp)
        gfp.vec.FV().NumPy()[:] = pc
        p_costa_grid = sample_scalar_gridfunction(mesh, gfp)
        gfp.vec.FV().NumPy()[:] = (pc - p_fom)
        p_costa_diff_grid = sample_scalar_gridfunction(mesh, gfp)

        gfu.vec.FV().NumPy()[:] = ub
        u_rom_u, u_rom_v = sample_velocity_vector_gridfunction(mesh, gfu)
        u_rom_mag = np.sqrt(u_rom_u**2 + u_rom_v**2)
        u_diff_u = u_rom_u - u_fom_u
        u_diff_v = u_rom_v - u_fom_v
        u_diff_mag = np.sqrt(u_diff_u**2 + u_diff_v**2)
        gfu.vec.FV().NumPy()[:] = uc
        u_costa_u, u_costa_v = sample_velocity_vector_gridfunction(mesh, gfu)
        u_costa_mag = np.sqrt(u_costa_u**2 + u_costa_v**2)
        u_costa_diff_u = u_costa_u - u_fom_u
        u_costa_diff_v = u_costa_v - u_fom_v
        u_costa_diff_mag = np.sqrt(u_costa_diff_u**2 + u_costa_diff_v**2)

        rows.append({
            "tau": float(tau_use),
            "r_tot": int(model["r_u"] + model["r_p"]),
            "r_tot_costa": int(model_c["r_u"] + model_c["r_p"]),
            "p_rom_grid": p_rom_grid,
            "p_diff_grid": p_diff_grid,
            "p_costa_grid": p_costa_grid,
            "p_costa_diff_grid": p_costa_diff_grid,
            "u_fom_u": u_fom_u,
            "u_fom_v": u_fom_v,
            "u_fom_mag": u_fom_mag,
            "u_rom_u": u_rom_u,
            "u_rom_v": u_rom_v,
            "u_rom_mag": u_rom_mag,
            "u_diff_u": u_diff_u,
            "u_diff_v": u_diff_v,
            "u_diff_mag": u_diff_mag,
            "u_costa_u": u_costa_u,
            "u_costa_v": u_costa_v,
            "u_costa_mag": u_costa_mag,
            "u_costa_diff_u": u_costa_diff_u,
            "u_costa_diff_v": u_costa_diff_v,
            "u_costa_diff_mag": u_costa_diff_mag,
        })

    _draw_stationary_pressure_two_tau(
        out_pdf_pressure_path=out_pdf_pressure_path,
        p_fom_grid=p_fom_grid,
        rows=rows,
        P_plot=P_plot,
        fom_dofs=fom_dofs,
    )
    _draw_stationary_velocity_two_tau(
        out_pdf_velocity_path=out_pdf_velocity_path,
        rows=rows,
        P_plot=P_plot,
        fom_dofs=fom_dofs,
    )


def build_solution_comparison_pdf(
    cfg: Config,
    partition_payload: dict,
    models_by_leaf: Dict[Tuple[float, int], Dict[str, Dict[str, Any]]],
    models_c_by_leaf: Dict[Tuple[float, int], Dict[str, Dict[str, Any]]],
    P_plot: int,
    tau_plot: float,
    mu_plot: Tuple[float, float],
    times_plot: Sequence[float],
    out_pdf_pressure_path: str,
    out_pdf_velocity_path: str,
):
    comb = (float(tau_plot), int(P_plot))
    if comb not in models_by_leaf:
        raise RuntimeError(f"Combination (tau={tau_plot}, P={P_plot}) not present in models_by_leaf")
    if comb not in models_c_by_leaf:
        raise RuntimeError(f"Combination (tau={tau_plot}, P={P_plot}) not present in models_c_by_leaf")

    leaf_id = classify_mu_in_partition_payload(mu_plot, partition_payload, cfg)
    if leaf_id not in models_by_leaf[comb]:
        raise RuntimeError(f"Leaf {leaf_id} missing baseline model for (tau={tau_plot}, P={P_plot})")
    if leaf_id not in models_c_by_leaf[comb]:
        raise RuntimeError(f"Leaf {leaf_id} missing CoSTA model for (tau={tau_plot}, P={P_plot})")

    model = models_by_leaf[comb][leaf_id]
    model_c = models_c_by_leaf[comb][leaf_id]

    mesh = Mesh(unit_square.GenerateMesh(maxh=cfg.maxh))
    V = HDiv(mesh, order=cfg.order, dirichlet=cfg.Gamma_N)
    Q = L2(mesh, order=cfg.order - 1)
    Y = FESpace([V, Q])
    f_base = forcing_base_cf(cfg)
    fom_dofs = int(V.ndof + Q.ndof)

    hu_fom, hp_fom = solve_fom_history_for_test_full(mesh, V, Q, Y, f_base, cfg, mu_plot)
    nsteps = hu_fom.shape[0]
    Zb = rom_predictor_rollout(model, mu_plot, nsteps=nsteps, p0=cfg.p_init)
    Zc = online_rollout_transient_ROM_CoSTA_shared(model_c, mu_plot, nsteps=nsteps, p0=cfg.p_init)

    r_u = int(model["r_u"])

    gfu = GridFunction(V)
    gfp = GridFunction(Q)
    timed_pressure_fields = []
    timed_velocity_fields = []
    times_used = []

    for t_req in times_plot:
        nstep, t_used = time_to_step_index(float(t_req), cfg.dt, nsteps)
        i_fom = nstep - 1
        i_rom = nstep
        times_used.append(t_used)

        u_fom = hu_fom[i_fom, :].astype(np.float64, copy=False)
        p_fom = hp_fom[i_fom, :].astype(np.float64, copy=False)

        ub = (model["V_u_stab"] @ Zb[i_rom, :r_u]).astype(np.float64, copy=False)
        pb = (model["V_p"] @ Zb[i_rom, r_u:]).astype(np.float64, copy=False)
        uc = (model_c["V_u_stab"] @ Zc[i_rom, :r_u]).astype(np.float64, copy=False)
        pc = (model_c["V_p"] @ Zc[i_rom, r_u:]).astype(np.float64, copy=False)

        gfp.vec.FV().NumPy()[:] = p_fom
        p_fom_grid = sample_scalar_gridfunction(mesh, gfp)
        gfp.vec.FV().NumPy()[:] = pb
        p_rom_grid = sample_scalar_gridfunction(mesh, gfp)
        gfp.vec.FV().NumPy()[:] = (pb - p_fom)
        p_rom_diff_grid = sample_scalar_gridfunction(mesh, gfp)
        gfp.vec.FV().NumPy()[:] = pc
        p_costa_grid = sample_scalar_gridfunction(mesh, gfp)
        gfp.vec.FV().NumPy()[:] = (pc - p_fom)
        p_costa_diff_grid = sample_scalar_gridfunction(mesh, gfp)

        timed_pressure_fields.append({
            "fom": p_fom_grid,
            "rom": p_rom_grid,
            "rom_diff": p_rom_diff_grid,
            "costa": p_costa_grid,
            "costa_diff": p_costa_diff_grid,
        })

        gfu.vec.FV().NumPy()[:] = u_fom
        u_fom_u, u_fom_v = sample_velocity_vector_gridfunction(mesh, gfu)
        u_fom_mag = np.sqrt(u_fom_u**2 + u_fom_v**2)

        gfu.vec.FV().NumPy()[:] = ub
        u_rom_u, u_rom_v = sample_velocity_vector_gridfunction(mesh, gfu)
        u_rom_mag = np.sqrt(u_rom_u**2 + u_rom_v**2)

        gfu.vec.FV().NumPy()[:] = uc
        u_costa_u, u_costa_v = sample_velocity_vector_gridfunction(mesh, gfu)
        u_costa_mag = np.sqrt(u_costa_u**2 + u_costa_v**2)

        u_rom_diff_u = u_rom_u - u_fom_u
        u_rom_diff_v = u_rom_v - u_fom_v
        u_rom_diff_mag = np.sqrt(u_rom_diff_u**2 + u_rom_diff_v**2)
        u_rom_signed_speed_diff = u_rom_mag - u_fom_mag

        u_costa_diff_u = u_costa_u - u_fom_u
        u_costa_diff_v = u_costa_v - u_fom_v
        u_costa_diff_mag = np.sqrt(u_costa_diff_u**2 + u_costa_diff_v**2)
        u_costa_signed_speed_diff = u_costa_mag - u_fom_mag

        timed_velocity_fields.append({
            "fom": {"u": u_fom_u, "v": u_fom_v, "mag": u_fom_mag, "color": u_fom_mag},
            "rom": {"u": u_rom_u, "v": u_rom_v, "mag": u_rom_mag, "color": u_rom_mag},
            "rom_diff": {
                "u": u_rom_diff_u,
                "v": u_rom_diff_v,
                "mag": u_rom_diff_mag,
                "color": u_rom_signed_speed_diff,
            },
            "costa": {"u": u_costa_u, "v": u_costa_v, "mag": u_costa_mag, "color": u_costa_mag},
            "costa_diff": {
                "u": u_costa_diff_u,
                "v": u_costa_diff_v,
                "mag": u_costa_diff_mag,
                "color": u_costa_signed_speed_diff,
            },
        })

    os.makedirs(os.path.dirname(out_pdf_pressure_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_pdf_velocity_path) or ".", exist_ok=True)

    with PdfPages(out_pdf_pressure_path) as pdf:
        _draw_pressure_panel_grid(
            pdf=pdf,
            timed_fields=timed_pressure_fields,
            times_used=times_used,
            tau_plot=tau_plot,
            P_plot=P_plot,
            mu_plot=mu_plot,
            fom_dofs=fom_dofs,
            rom_r_tot=int(model["r_u"] + model["r_p"]),
            costa_r_tot=int(model_c["r_u"] + model_c["r_p"]),
        )

    with PdfPages(out_pdf_velocity_path) as pdf:
        _draw_velocity_panel_grid(
            pdf=pdf,
            timed_fields=timed_velocity_fields,
            times_used=times_used,
            tau_plot=tau_plot,
            P_plot=P_plot,
            mu_plot=mu_plot,
            fom_dofs=fom_dofs,
            rom_r_tot=int(model["r_u"] + model["r_p"]),
            costa_r_tot=int(model_c["r_u"] + model_c["r_p"]),
        )

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--nproc", type=int, default=None)
    parser.add_argument("--maxh", type=float, default=None)
    parser.add_argument("--store_every", type=int, default=None)
    parser.add_argument("--error_sample_start_time", type=float, default=None)
    parser.add_argument("--error_integration_order", type=int, default=None)
    parser.add_argument("--Nsplit", type=int, default=None)
    parser.add_argument("--muu_ridge", type=float, default=None)
    parser.add_argument("--costa_mode", type=str, default=None, choices=["dnn", "ridge"])
    parser.add_argument("--costa_ridge", type=float, default=None)

    parser.add_argument("--norm_mode", type=str, default=None, choices=["L2", "Kinv_ref"])
    parser.add_argument("--debug_use_exact_muu_in_direct_fit_solver", action="store_true")

    parser.add_argument("--compute_muu_l2fit_errors", dest="compute_muu_l2fit_errors", action="store_true")
    parser.add_argument("--skip_muu_l2fit_errors", dest="compute_muu_l2fit_errors", action="store_false")
    parser.set_defaults(compute_muu_l2fit_errors=Config.compute_muu_l2fit_errors)
    parser.add_argument("--l2fit_quad_n_1d", type=int, default=None)
    parser.add_argument("--l2fit_pattern_tol", type=float, default=None)
    parser.add_argument("--stationary", action="store_true")
    parser.add_argument("--plot_P", type=int, default=None)
    parser.add_argument("--plot_tau", type=float, default=None)
    parser.add_argument("--plot_tau_pair", type=float, nargs=2, default=None)
    parser.add_argument("--plot_mu_x", type=float, default=None)
    parser.add_argument("--plot_mu_y", type=float, default=None)
    parser.add_argument("--plot_times", type=float, nargs="+", default=None)
    parser.add_argument("--plot_pdf", type=str, default=None)
    parser.add_argument("--skip_solution_pdf", action="store_true")

    args = parser.parse_args()

    cfg = Config()
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.tag is not None:
        cfg.tag = args.tag
    if args.nproc is not None:
        cfg.nproc_regions = int(args.nproc)
    if args.maxh is not None:
        cfg.maxh = float(args.maxh)
    if args.store_every is not None:
        cfg.store_every = int(args.store_every)
    if args.error_sample_start_time is not None:
        cfg.error_sample_start_time = float(args.error_sample_start_time)
    if args.error_integration_order is not None:
        cfg.error_integration_order = int(args.error_integration_order)
    if args.Nsplit is not None:
        cfg.Nsplit = int(args.Nsplit)
    if args.muu_ridge is not None:
        cfg.muu_ridge = float(args.muu_ridge)
    if args.costa_mode is not None:
        cfg.costa_mode = str(args.costa_mode)
    if args.costa_ridge is not None:
        cfg.costa_ridge = float(args.costa_ridge)

    # Keep legacy flag consistent with the new mode switch.
    cfg.costa_use_dnn = (str(cfg.costa_mode).strip().lower() == "dnn")


    if args.norm_mode is not None:
        cfg.norm_mode = str(args.norm_mode)

    cfg.compute_muu_l2fit_errors = bool(args.compute_muu_l2fit_errors)
    if args.l2fit_quad_n_1d is not None:
        cfg.l2fit_quad_n_1d = int(args.l2fit_quad_n_1d)
    if args.l2fit_pattern_tol is not None:
        cfg.l2fit_pattern_tol = float(args.l2fit_pattern_tol)
    if args.stationary:
        cfg.stationary_mode = True

    if cfg.norm_mode == "L2":
        cfg.pod_mode = "L2"
    elif cfg.norm_mode == "Kinv_ref":
        cfg.pod_mode = "snap_weights"
    else:
        raise ValueError(f"Unknown norm_mode={cfg.norm_mode!r}")

    if int(cfg.error_integration_order) < 1:
        raise ValueError(f"error_integration_order must be >= 1, got {cfg.error_integration_order}")

    if args.debug_use_exact_muu_in_direct_fit_solver:
        cfg.debug_use_exact_muu_in_direct_fit_solver = True

    if cfg.stationary_mode:
        # Stationary-like run inside the transient pipeline:
        # zero storage term, no forcing ramp (time-independent forcing),
        # and one effective time step for snapshot/test horizons.
        cfg.Sstor = 0.0
        cfg.use_ramp = False
        cfg.costa_use_time_feature = False
        cfg.store_every = 1
        cfg.T_snap = float(cfg.dt)
        cfg.T_test = float(cfg.dt)

    plot_P_req = args.plot_P if args.plot_P is not None else cfg.plot_P
    plot_tau_req = args.plot_tau if args.plot_tau is not None else cfg.plot_tau
    plot_tau_pair_req = args.plot_tau_pair if args.plot_tau_pair is not None else cfg.plot_tau_pair
    plot_mu_x = args.plot_mu_x if args.plot_mu_x is not None else cfg.plot_mu_x
    plot_mu_y = args.plot_mu_y if args.plot_mu_y is not None else cfg.plot_mu_y
    plot_times_req = args.plot_times if args.plot_times is not None else cfg.plot_times

    plot_P = select_P_for_plot(plot_P_req, cfg.P_list)
    plot_tau = select_tau_for_plot(plot_tau_req, cfg.tau_list)
    plot_tau_pair = None
    if not args.skip_solution_pdf:
        plot_tau_pair = select_two_taus_for_plot(plot_tau_pair_req, cfg.tau_list)
    plot_mu = (float(plot_mu_x), float(plot_mu_y))
    plot_times = [float(t) for t in plot_times_req]
    if len(plot_times) == 0:
        raise ValueError("plot_times must contain at least one time value")

    if args.plot_pdf is None:
        if cfg.stationary_mode and not args.skip_solution_pdf:
            tau_tag_a = f"{plot_tau_pair[0]:.3e}".replace("+", "").replace("-", "m").replace(".", "p")
            tau_tag_b = f"{plot_tau_pair[1]:.3e}".replace("+", "").replace("-", "m").replace(".", "p")
            plot_pdf_base = os.path.join(
                cfg.out_dir,
                # f"solutions_{cfg.tag}_stationary_P{plot_P}_tau{tau_tag_a}_tau{tau_tag_b}",
                f"rom_costa_solution_compare_stationary",
            )
        else:
            tau_tag = f"{plot_tau:.3e}".replace("+", "").replace("-", "m").replace(".", "p")
            plot_pdf_base = os.path.join(cfg.out_dir, f"solutions_{cfg.tag}_P{plot_P}_tau{tau_tag}")
    else:
        plot_pdf_base = os.path.splitext(str(args.plot_pdf))[0]

    plot_pdf_pressure_path = f"{plot_pdf_base}_pressure.pdf"
    plot_pdf_velocity_path = f"{plot_pdf_base}_velocity.pdf"

    os.makedirs(cfg.out_dir, exist_ok=True)

    log(
        f"Starting run tag={cfg.tag}, maxh={cfg.maxh}, nproc_regions={cfg.nproc_regions}, "
        f"store_every={cfg.store_every}, pod_mode={cfg.pod_mode}, "
        f"costa_mode={cfg.costa_mode}, costa_ridge={cfg.costa_ridge}, "
        f"error_sample_start_time={cfg.error_sample_start_time}, "
        f"error_integration_order={cfg.error_integration_order}, "
        f"compute_muu_l2fit_errors={cfg.compute_muu_l2fit_errors}, "
        f"l2fit_quad_n_1d={cfg.l2fit_quad_n_1d}, "
        f"stationary_mode={cfg.stationary_mode}, "
        f"plot_P={plot_P}, plot_tau={plot_tau:g}, plot_tau_pair={plot_tau_pair}, "
        f"plot_mu={plot_mu}, plot_times={plot_times}, "
        f"skip_solution_pdf={args.skip_solution_pdf}"
    )

    xg = np.linspace(cfg.a_mu, cfg.b_mu, cfg.ntrain_1d)
    yg = np.linspace(cfg.a_mu, cfg.b_mu, cfg.ntrain_1d)
    mu_train_all = [(float(xv), float(yv)) for xv in xg for yv in yg]

    rng = np.random.default_rng(cfg.rng_seed)
    mu_test = list(zip(
        rng.uniform(cfg.a_mu, cfg.b_mu, cfg.n_test),
        rng.uniform(cfg.a_mu, cfg.b_mu, cfg.n_test)
    ))
    mu_test = [(float(a), float(b)) for a, b in mu_test]
    log(f"mu_test: {mu_test}")

    t_build0 = time.time()

    log("Building global POD snapshot cache ...")
    t_cache0 = time.time()
    pod_cache = build_global_snapshot_cache(cfg, mu_train_all)
    log(f"Built global POD snapshot cache in {time.time() - t_cache0:.2f}s")

    nsteps_fine = int(round(cfg.T_snap / cfg.dt))
    se = int(cfg.store_every)
    if nsteps_fine % se != 0:
        raise RuntimeError(f"nsteps_fine={nsteps_fine} not divisible by store_every={se}")
    keep_steps = nsteps_fine // se

    log(f"Building uniform rectangular partition with Nsplit={cfg.Nsplit} ...")
    rect_obj = build_uniform_rect_partition(cfg, mu_train_all)
    partition_payload = make_partition_payload_rect(rect_obj)

    log(f"Final rectangular leaves: {len(rect_obj['leaves'])}")
    final_pod_spectra_by_leaf: Dict[str, Dict[str, Any]] = {}
    final_leaves: Dict[str, PseudoLeaf] = {}
    mu_buckets: Dict[str, List[Tuple[float, float]]] = {}
    for leaf_id, leaf in rect_obj["leaves"].items():
        if len(leaf.mu_train) == 0:
            continue

        final_leaves[leaf_id] = PseudoLeaf(
            leaf_id=leaf.leaf_id,
            ax=float(leaf.ax),
            bx=float(leaf.bx),
            ay=float(leaf.ay),
            by=float(leaf.by),
            depth=int(leaf.depth),
        )
        mu_buckets[leaf_id] = list(leaf.mu_train)

        out = build_leaf_pod_only_from_cache(
            cfg=cfg,
            leaf_id=leaf_id,
            depth=leaf.depth,
            mu_train=leaf.mu_train,
            cache=pod_cache,
            bounds=(leaf.ax, leaf.bx, leaf.ay, leaf.by),
        )
        if out is not None:
            lid, pod_meta = out
            final_pod_spectra_by_leaf[lid] = pod_meta
    # --------------------------------------------------------
    # Full build on final leaves
    # --------------------------------------------------------
    import multiprocessing as mp
    import gc
    import copy

    log("Building full models only for final leaves ...")

    leaf_items = list(final_leaves.items())
    full_out = []

    if len(final_leaves) == 1:
        # Single leaf:
        # build in main process, and allow inner parallelization over (tau, P)
        cfg_single = copy.deepcopy(cfg)
        cfg_single.costa_parallel = True
        log("Single leaf detected -> parallelizing over (tau, P) inside leaf")

        leaf_id, leaf = leaf_items[0]
        mu_train_leaf = mu_buckets[leaf_id]
        Su_leaf, Sp_leaf = extract_leaf_snapshot_matrix_from_cache(
            pod_cache, mu_train_leaf, keep_steps
        )

        full_out = [build_leaf_full_core(
            cfg=cfg_single,
            leaf=leaf,
            mu_train=mu_train_leaf,
            Su=Su_leaf,
            Sp=Sp_leaf,
        )]

    else:
        # Multiple leaves:
        # parallelize across leaves only, forbid inner multiprocessing
        cfg_multi = copy.deepcopy(cfg)
        cfg_multi.costa_parallel = False
        log(f"{len(final_leaves)} leaves detected -> parallelizing across leaves only")

        cfg_dict_multi = asdict(cfg_multi)
        worker_args = []

        for leaf_id, leaf in leaf_items:
            mu_train_leaf = mu_buckets[leaf_id]
            Su_leaf, Sp_leaf = extract_leaf_snapshot_matrix_from_cache(
                pod_cache,
                mu_train_leaf,
                keep_steps
            )

            worker_args.append((
                cfg_dict_multi,
                leaf.as_dict(),
                mu_train_leaf,
                Su_leaf,
                Sp_leaf
            ))

        nproc_leaf = min(cfg.nproc_regions, len(worker_args))
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=nproc_leaf) as pool:
            full_out = list(
                pool.imap_unordered(worker_build_leaf_full, worker_args, chunksize=1)
            )

    full_out = [out for out in full_out if out is not None]

    models_by_leaf: Dict[Tuple[float, int], Dict[str, Dict[str, Any]]] = {}
    models_c_by_leaf: Dict[Tuple[float, int], Dict[str, Dict[str, Any]]] = {}

    for tau in cfg.tau_list:
        for Pdeg_req in cfg.P_list:
            models_by_leaf[(float(tau), int(Pdeg_req))] = {}
            models_c_by_leaf[(float(tau), int(Pdeg_req))] = {}

    for leaf_id, models_map, models_c_map, pod_meta in full_out:
        for comb, model in models_map.items():
            models_by_leaf[comb][leaf_id] = model
        for comb, model_c in models_c_map.items():
            models_c_by_leaf[comb][leaf_id] = model_c
        final_pod_spectra_by_leaf[leaf_id] = pod_meta

    if not args.skip_solution_pdf:
        if cfg.stationary_mode:
            log(
                f"Building stationary two-tau solution comparison PDF for P={plot_P}, "
                f"taus=({plot_tau_pair[0]:g}, {plot_tau_pair[1]:g}), "
                f"mu=({plot_mu[0]:.6f}, {plot_mu[1]:.6f}) ..."
            )
        else:
            log(
                f"Building solution comparison PDF for P={plot_P}, tau={plot_tau:g}, "
                f"mu=({plot_mu[0]:.6f}, {plot_mu[1]:.6f}), times={plot_times} ..."
            )
        t_plot0 = time.time()
        if cfg.stationary_mode:
            build_stationary_two_tau_solution_comparison_pdf(
                cfg=cfg,
                partition_payload=partition_payload,
                models_by_leaf=models_by_leaf,
                models_c_by_leaf=models_c_by_leaf,
                P_plot=plot_P,
                tau_pair=plot_tau_pair,
                mu_plot=plot_mu,
                out_pdf_pressure_path=plot_pdf_pressure_path,
                out_pdf_velocity_path=plot_pdf_velocity_path,
            )
        else:
            build_solution_comparison_pdf(
                cfg=cfg,
                partition_payload=partition_payload,
                models_by_leaf=models_by_leaf,
                models_c_by_leaf=models_c_by_leaf,
                P_plot=plot_P,
                tau_plot=plot_tau,
                mu_plot=plot_mu,
                times_plot=plot_times,
                out_pdf_pressure_path=plot_pdf_pressure_path,
                out_pdf_velocity_path=plot_pdf_velocity_path,
            )
        log(
            f"Wrote pressure PDF to {plot_pdf_pressure_path} and velocity PDF to "
            f"{plot_pdf_velocity_path} in {time.time() - t_plot0:.2f}s"
        )

    
    log(f"Leaf model build done in {time.time() - t_build0:.2f}s")
    rom_ranks = build_rom_ranks_payload(models_by_leaf)

    direct_muu_l2fit_errors = None
    if cfg.compute_muu_l2fit_errors:
        log("Building direct full-order L2-fits of M_uu on final leaves ...")
        t_fit0 = time.time()

        full_muu_fit_by_P = build_all_leaf_full_muu_l2fits(
            cfg=cfg,
            final_leaves=final_leaves,
            mu_buckets=mu_buckets,
        )

        log(f"Built direct full-order M_uu L2-fits in {time.time() - t_fit0:.2f}s")

        log("Evaluating direct Frobenius M_uu fit errors and fitted-solution errors ...")
        t_eval_fit0 = time.time()

        direct_muu_l2fit_errors = compute_direct_muu_l2fit_errors(
            cfg=cfg,
            partition_payload=partition_payload,
            full_muu_fit_by_P=full_muu_fit_by_P,
        )

        log(f"Direct M_uu L2-fit error evaluation done in {time.time() - t_eval_fit0:.2f}s")

        for Pdeg_req in cfg.P_list:
            res = direct_muu_l2fit_errors[str(int(Pdeg_req))]
            log(
                f"[direct M_uu L2-fit] P={int(Pdeg_req)}   "
                f"fro={res['muu_over_fullA_fro_rel']:.3e}   "
                f"u_L2={res['u_l2_rel']:.3e}   "
                f"p_L2={res['p_l2_rel']:.3e}   "
                f"u_Hdiv={res['u_hdiv_rel']:.3e}   "
                f"u_KHdiv={res['u_khdiv_rel']:.3e}"
            )

    del pod_cache
    del full_out
    gc.collect()

    log("Freed training snapshot cache before evaluation")
    log("Starting evaluation on mu_test (full fine grid) [PARALLEL] ...")

    results = {
        str(tau): {
            "P": [],
            "r_u": [],
            "r_p": [],
            "r_tot": [],
            "r_u_avg": [],
            "r_u_med": [],
            "r_u_max": [],
            "r_p_avg": [],
            "r_p_med": [],
            "r_p_max": [],
            "r_tot_avg": [],
            "r_tot_med": [],
            "r_tot_max": [],
            "rom_speedup_throughput": [],
            "costa_speedup_throughput": [],
            "per_mu": []
        } for tau in cfg.tau_list
    }

    log("Building FOM test cache for mu_test ...")
    t_fom_cache0 = time.time()
    fom_test_cache = build_fom_test_cache(cfg, mu_test)
    log(f"Built FOM test cache in {time.time() - t_fom_cache0:.2f}s")

    for tau in cfg.tau_list:
        tau = float(tau)
        for Pdeg_req in cfg.P_list:
            Pdeg_req = int(Pdeg_req)
            comb = (tau, Pdeg_req)

            leaf_models = models_by_leaf[comb]
            leaf_models_c = models_c_by_leaf[comb]

            rank_stats = summarize_ranks_for_comb(leaf_models)

            r_u_med = int(round(rank_stats["r_u_med"]))
            r_p_med = int(round(rank_stats["r_p_med"]))
            r_tot_med = int(round(rank_stats["r_tot_med"]))

            log(f"[eval serial] tau={tau:g} P_req={Pdeg_req} over {len(mu_test)} mu_test ...")

            per_mu = parallel_eval_for_comb(
                cfg=cfg,
                partition_payload=partition_payload,
                mu_test=mu_test,
                leaf_models=leaf_models,
                leaf_models_c=leaf_models_c,
                fom_test_cache=fom_test_cache,
                nproc_eval=1,
                chunksize=4,
            )

            summary = summarize_comb_metrics(per_mu)

            tau_key = str(tau)
            results[tau_key]["P"].append(Pdeg_req)
            results[tau_key]["r_u"].append(r_u_med)
            results[tau_key]["r_p"].append(r_p_med)
            results[tau_key]["r_tot"].append(r_tot_med)

            results[tau_key]["r_u_avg"].append(float(rank_stats["r_u_avg"]))
            results[tau_key]["r_u_med"].append(float(rank_stats["r_u_med"]))
            results[tau_key]["r_u_max"].append(int(rank_stats["r_u_max"]))

            results[tau_key]["r_p_avg"].append(float(rank_stats["r_p_avg"]))
            results[tau_key]["r_p_med"].append(float(rank_stats["r_p_med"]))
            results[tau_key]["r_p_max"].append(int(rank_stats["r_p_max"]))

            results[tau_key]["r_tot_avg"].append(float(rank_stats["r_tot_avg"]))
            results[tau_key]["r_tot_med"].append(float(rank_stats["r_tot_med"]))
            results[tau_key]["r_tot_max"].append(int(rank_stats["r_tot_max"]))

            results[tau_key]["per_mu"].append({"P": Pdeg_req, "tau": tau, "data": per_mu})
            results[tau_key]["rom_speedup_throughput"].append(summary["rom_speedup_throughput"])
            results[tau_key]["costa_speedup_throughput"].append(summary["costa_speedup_throughput"])
            append_comb_results(results[tau_key], summary)
            log_comb_summary(tau, Pdeg_req, rank_stats, summary)

    payload = {
        "meta": {
            "tag": cfg.tag,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": (
                "Supports both adaptive anchor-tree and equispaced rectangular parameter partitions. "
                "The pod_spectra entry stores raw POD ranks; rom_ranks stores the actual stabilized reduced dimensions."
            ),
        },
        "config": asdict(cfg),
        "mu_test": mu_test,
        "partition": partition_payload,
        "pod_spectra": {
            "pod_mode_u": cfg.pod_mode,
            "regions": final_pod_spectra_by_leaf,
        },
        "rom_ranks": rom_ranks,
        "results": results,
        "direct_muu_l2fit_errors": direct_muu_l2fit_errors,
    }

    out_path = os.path.join(cfg.out_dir, f"results_{cfg.tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    log(f"Done. Wrote {out_path}")


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    main()
