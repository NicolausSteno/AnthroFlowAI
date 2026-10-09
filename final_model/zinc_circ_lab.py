#!/usr/bin/env python3
"""
zinc_circ_lab.py — lab module for WP-2c/2d/2e
=============================================

Shared machinery for the three Chapter 2 diagnostics that run on the WP-2a
deliverable `analysis/wp2a_A_of_t.npz`:

  * **WP-2c** dynamic (frozen-time eigenvalue) sensitivity
        `dlambda_k/dp_n = q_k^T (dA/dp_n) v_k / (q_k^T v_k)`   # Ch2 Eq. eigenvalue_sensitivity
  * **WP-2d** circularity indicators, frozen-time vs non-autonomous
        `tau^fr_i(t)   = 1^T [-A_FD(t)]^-1 e_i`                # Ch2 Eq. frozen_time_circularity
        `tau_FD,i(t0)  = int_0^inf 1^T Phi(t0+u,t0) e_i du`    # Ch2 Eq. nonautonomous_lifetime
        `upsilon_i(t0) = int_0^inf c_use(t0+u)^T Phi(t0+u,t0) e_i du`
                                                              # Ch2 Eq. nonautonomous_use_count
  * **WP-2e** fundamental-matrix sensitivities
        `dtau_i/dalpha_l  = 1^T N_T (dA_T/dalpha_l) N_T e_i`   # Ch2 Eq. tau_sensitivity
        `dUpsilon_{m,i}/dalpha_l`                              # Ch2 Eq. flow_count_sensitivity

Pattern
-------
Follows `zinc_A_lab.py` (CLAUDE.md rule 1): `zinc_colloc_v5.py` is imported,
never edited, and `check()` delegates to `zinc_A_lab.check()` so the resolved
drivers, `input_dim` and digests are printed by every driver script.  No
patches are needed — like WP-2b, all three packages are post-processing of the
WP-2a arrays and require no refit.

Three mapping decisions, stated once here because all three packages inherit
them
-------------------------------------------------------------------------
**1. `A_FD(t) = A(t)`.**  Chapter 2 defines `A_FD` as the sub-generator
obtained by *removing landfill from the transient state space while retaining
landfilling and dissipative flows as absorbing exits*.  The UDE has no
landfill, tailings or slag state at all (spec §1: `s6, s7, s8` absent or
lumped), and WP-2a established `1^T A(t) = -l(t)^T` with `l` assembled from
the six named loss flows — refinery, waelz, first-use, end-use, dissipative
and end-of-life.  End-of-life loss `(1 - tau_olds)/mu_k` *is* landfilling, and
it already leaves the 6-state.  So the estimated `A(t)` is exactly Chapter 2's
`A_FD`, not an approximation to it, and the transient space needs no
truncation.  The corollary is that the UDE has **no analogue of the
dissipation-only boundary** `A_T = A`: because landfill is not carried as a
state, the "time to dissipation" reading of the lifetime is not available.
Only the final-disposal reading is.  `-A(t)` is a non-singular M-matrix
wherever WP-2a's certificate `max_j sum_i a_ij <= -eta < 0` holds, so
`N(t) = (-A(t))^-1 >= 0` entrywise; this is asserted, not assumed.

**2. `alpha_8 e_3^T` becomes a rate *functional* `c_use(t)^T`.**  Chapter 2's
entry into use is the single flow `f_8 = alpha_8 s_3`, so the use-count
integrand is `alpha_8 e_3^T`.  The UDE aggregates Chapter 2's `s_2` and `s_3`
into one Refined stock (spec §1) and old scrap re-enters manufacturing
directly, so entry into use is fed by *two* stocks:

    rate into use = inflow_share * (alpha_refc S_ref + alpha_dr S_scrap)
                  = c_use(t)^T S,
    c_use(t)_1 = sum_k A[2+k, 1](t),   c_use(t)_5 = sum_k A[2+k, 5](t).

Every Chapter 2 formula of the form `alpha_m e_k^T (...)` is therefore
implemented as `c_m(t)^T (...)`, which reduces to Chapter 2's expression
exactly when the flow has a single upstream stock.  `phi_lm` in
Eq. flow_count_sensitivity generalises correspondingly to `dc_m/dp_l`.

**3. The continuum between year nodes.**  WP-2a delivers coefficients at the
40 integer year nodes.  `Phi(t, t0)` needs `A(t)` everywhere, so a rule is
required: coefficients (not `A` entries) are interpolated **linearly between
nodes**, then `A` is assembled.  This reproduces WP-2a's `A` exactly at the
nodes, and keeps the Metzler property and `l >= 0` by construction, which
interpolating `A` entries directly would not guarantee under extrapolation.

CLI
---
    python zinc_circ_lab.py --check
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import scipy.linalg as sla

import zinc_A_lab as Alab
from zinc_A_lab import ODE_STATE_NAMES, N_ODE          # noqa: F401

N_COH = 3

# ---------------------------------------------------------------------------
# parameter vector
# ---------------------------------------------------------------------------
# The `p_n` of Ch2 Eq. eigenvalue_sensitivity / Eq. totalsens, for the UDE.
# Chapter 2's list is "transfer rates alpha_{9,z}, dissipative rates
# alpha_{18,z}, serial-compartment rates kappa_z, or constant product shares
# w_z".  The UDE analogues are: the four alphas, the four binary taus, the
# three manufacturing/use simplices (the w_z analogue) and the fixed cohort
# lifetimes mu_k (the 1/kappa_z analogue).  mu is not learned, but its
# sensitivity is exactly the question WP-10 would act on, so it is carried.
PARAM_NAMES = [
    "alpha_cc", "alpha_refc", "alpha_win", "alpha_dr",
    "tau_ref", "tau_waelz", "tau_olds", "tau_diss",
    "frac_fu_new", "frac_fu_loss", "frac_fu_out",
    "frac_eu_new", "frac_eu_loss", "frac_eu_into",
    "f_cohort_10yr", "f_cohort_20yr", "f_cohort_44yr",
    "mu_10yr", "mu_20yr", "mu_44yr",
]
N_PARAM = len(PARAM_NAMES)
ALPHA_SLICE = slice(0, 4)

# Chapter 2 labels, for the cross-reference table the thesis needs.
CH2_LABEL = {
    "alpha_cc": "alpha_1", "alpha_refc": "alpha_2/alpha_8",
    "alpha_win": "alpha_14", "alpha_dr": "alpha_13",
    "tau_olds": "alpha_9 share", "mu_44yr": "1/kappa_3",
}

# The three simplices, as (name, index tuple).  A coefficient inside a simplex
# cannot move alone: Ch2 handles this with the directional derivative
# `dlambda/dtheta = dlambda/dw_r - dlambda/dw_q` (Appendix, constant product
# shares).  `SIMPLEX_DIRECTIONS` enumerates the unordered pairs; the reported
# direction "r<-q" means "move mass from q to r".
SIMPLEXES = {
    "frac_fu": (8, 9, 10),
    "frac_eu": (11, 12, 13),
    "f_cohort": (14, 15, 16),
}


def simplex_directions():
    out = []
    for name, idx in SIMPLEXES.items():
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                out.append((name, idx[a], idx[b],
                            f"{PARAM_NAMES[idx[a]]} <- {PARAM_NAMES[idx[b]]}"))
    return out


def pack_params(d, mu):
    """Stack the WP-2a coefficient arrays into `P` of shape (..., N_PARAM).

    `d` is the loaded `wp2a_A_of_t.npz` (or any mapping with the same keys);
    `mu` is `MU_COHORTS`, broadcast to every node because it is a parameter of
    the assembly even though it is not learned.
    """
    alphas = np.asarray(d["alphas"], float)          # (..., 4)
    taus = np.asarray(d["taus"], float)              # (..., 4)
    fu = np.asarray(d["frac_fu"], float)             # (..., 3)
    eu = np.asarray(d["frac_eu"], float)             # (..., 3)
    fc = np.asarray(d["f_cohort"], float)            # (..., 3)
    mu_b = np.broadcast_to(np.asarray(mu, float), alphas.shape[:-1] + (3,))
    return np.concatenate([alphas, taus, fu, eu, fc, mu_b], axis=-1)


def assemble_from_params(P):
    """`A` of shape (..., 6, 6) from a packed parameter array (..., N_PARAM).

    A dtype-generic transcription of `zinc_A_lab.assemble_A` (which is itself
    read off `zinc_colloc_v5.compute_flows_from_nn`), so that complex-step
    differentiation can be run through it.  Verified against
    `zinc_A_lab.assemble_A` at every node by `verify_assembly`.

    # Ch2 Eq. system, Eq. a1
    """
    P = np.asarray(P)
    shape = P.shape[:-1]
    a_cc, a_refc, a_win, a_dr = (P[..., i] for i in range(4))
    t_ref, t_wae, t_old, t_dis = (P[..., i] for i in range(4, 8))
    fu_new, _fu_loss, fu_out = (P[..., i] for i in range(8, 11))
    eu_new, _eu_loss, eu_into = (P[..., i] for i in range(11, 14))
    fc = [P[..., 14 + k] for k in range(N_COH)]
    mu = [P[..., 17 + k] for k in range(N_COH)]

    g = [1.0 / (m + 1e-12) for m in mu]          # the code's guarded reciprocal
    inflow_share = (1.0 - t_dis) * (eu_into * fu_out)
    newscrap_share = fu_new + eu_new * fu_out

    A = np.zeros(shape + (N_ODE, N_ODE), dtype=P.dtype)
    A[..., 0, 0] = -a_cc
    A[..., 1, 0] = (1.0 - t_ref) * a_cc
    A[..., 1, 1] = -a_refc
    A[..., 1, 5] = t_wae * a_win
    for k in range(N_COH):
        A[..., 2 + k, 1] = fc[k] * inflow_share * a_refc
        A[..., 2 + k, 5] = fc[k] * inflow_share * a_dr
        A[..., 2 + k, 2 + k] = -g[k]
        A[..., 5, 2 + k] = t_old * g[k]
    A[..., 5, 1] = newscrap_share * a_refc
    A[..., 5, 5] = newscrap_share * a_dr - (a_win + a_dr)
    return A


def dA_dparams(P, h=1e-30):
    """`dA/dp_n` of shape (..., N_PARAM, 6, 6), by complex step.

    `assemble_from_params` is a polynomial in `P` with no conjugation, so the
    complex step `Im f(p + i h)/h` is exact to machine precision with no
    subtractive cancellation — unlike a finite difference, which would have to
    trade off truncation against round-off on entries spanning 1e-2 to 1e+1.
    Cross-checked against a central difference in `verify_derivatives`.

    # Ch2 Eq. totalsens: da_ij/dp_n, the part that "depends on the model
    # formulation and can be readily calculated".
    """
    P = np.asarray(P, float)
    out = np.empty(P.shape[:-1] + (N_PARAM, N_ODE, N_ODE))
    for n in range(N_PARAM):
        Z = P.astype(complex)
        Z[..., n] += 1j * h
        out[..., n, :, :] = assemble_from_params(Z).imag / h
    return out


# ---------------------------------------------------------------------------
# flow-rate functionals  (the `alpha_m e_k^T` of Ch2, generalised)
# ---------------------------------------------------------------------------
def flow_rows(A):
    """Row functionals `c_m^T` such that flow `m` = `c_m^T S`, shape (..., 6).

    Each is a linear functional of `A` itself, so `dc_m/dp = ` the same
    functional applied to `dA/dp` — which is what makes the `phi_lm` term of
    Ch2 Eq. flow_count_sensitivity computable without a second transcription
    of the algebra.

    Returned keys and their Chapter 2 counterparts:
      use_entry            f_8  = alpha_8 s_3      (entry into use)
      oldscrap_metallurgy  f_14 = alpha_14 s_5     (old scrap -> metallurgy)
      oldscrap_manufacture f_13 = alpha_13 s_5     (old scrap -> manufacturing)
      eol_collection       f_9  = alpha_9 s_4      (end-of-life collection)
    """
    A = np.asarray(A)
    shape = A.shape[:-2]
    z = lambda: np.zeros(shape + (N_ODE,), dtype=A.dtype)      # noqa: E731

    use = z()
    use[..., 1] = A[..., 2:5, 1].sum(-1)
    use[..., 5] = A[..., 2:5, 5].sum(-1)

    wae = z()
    wae[..., 5] = A[..., 1, 5]

    dr = z()
    dr[..., 5] = A[..., 2:5, 5].sum(-1)

    eol = z()
    eol[..., 2:5] = A[..., 5, 2:5]

    return dict(use_entry=use, oldscrap_metallurgy=wae,
                oldscrap_manufacture=dr, eol_collection=eol)


FLOW_NAMES = ["use_entry", "oldscrap_metallurgy", "oldscrap_manufacture",
              "eol_collection"]
FLOW_CH2 = {"use_entry": "f_8 (upsilon)", "oldscrap_metallurgy": "f_14",
            "oldscrap_manufacture": "f_13", "eol_collection": "f_9"}


# ---------------------------------------------------------------------------
# fundamental matrix
# ---------------------------------------------------------------------------
def spectral_abscissa(A):
    """`s(A) = max_k Re lambda_k`.  Negative iff the transient system is
    asymptotically stable, which is the condition under which
    `tau_i = 1^T (-A)^-1 e_i` is a finite expected absorption time."""
    return np.linalg.eigvals(np.asarray(A, float)).real.max(axis=-1)


def fundamental(A, check=True, on_singular="raise", tol=1e-9):
    """`N = (-A)^-1`, the continuous-time Green matrix.   # Ch2 Eq. N_T

    For a non-singular M-matrix `-A` (Metzler `A` with strictly negative
    column sums, which WP-2a certified) `N >= 0` entrywise and
    `tau = 1^T N e_i` is a genuine expected absorption time.  Both properties
    are asserted rather than assumed, because a sign flip in `N` would
    silently produce negative "lifetimes".

    `on_singular="nan"` returns NaN for any node whose spectral abscissa has
    reached 0 instead of raising.  That case is not hypothetical: under the
    `trend` extrapolation rule of `coeff_path`, continuing the estimated
    trend long enough clips `tau_waelz` and `tau_ref` at their feasible
    boundaries, the corresponding stock loses its only absorbing exit, and the
    technological lifetime genuinely diverges.  That is a result about the
    extrapolation, not a numerical failure, so it is flagged and reported
    rather than suppressed.
    """
    A = np.asarray(A, float)
    sing = spectral_abscissa(A) >= -tol
    if sing.any():
        if on_singular == "raise":
            raise AssertionError(
                f"-A is singular / not asymptotically stable at "
                f"{int(sing.sum())} node(s): max Re lambda = "
                f"{spectral_abscissa(A).max():.3e}")
        A = np.where(sing[..., None, None], -np.eye(A.shape[-1]), A)
    N = np.linalg.inv(-A)
    if check:
        good = N[~sing] if sing.any() else N
        if good.size and good.min() < -1e-9:
            raise AssertionError(f"N = (-A)^-1 has negative entries "
                                 f"(min {good.min():.3e}); -A is not an "
                                 f"M-matrix")
        off = ~np.eye(N.shape[-1], dtype=bool)
        if A[..., off].min() < -1e-12:
            raise AssertionError("A is not Metzler")
    if sing.any():
        N = np.where(sing[..., None, None], np.nan, N)
    return N


def frozen_indicators(A, on_singular="raise"):
    """Frozen-time circularity indicators.     # Ch2 Eq. frozen_time_circularity

    Returns `tau_fr` (..., 6) — expected time to final disposal or dissipation
    for a unit starting in each stock — and one `(..., 6)` array per flow in
    `flow_rows`, the expected cumulative number of passages through that flow
    before absorption (`upsilon` for `use_entry`).
    """
    N = fundamental(A, on_singular=on_singular)
    ones = np.ones(N.shape[:-2] + (N_ODE,))
    tau = np.einsum("...i,...ij->...j", ones, N)
    rows = flow_rows(A)
    counts = {k: np.einsum("...i,...ij->...j", c, N) for k, c in rows.items()}
    return tau, counts, N


def A_pre(A):
    """Transient matrix with the in-use stocks removed from the transient
    space, entry into use retained as an absorbing exit — Chapter 2's
    `A_pre` (Eq. prob_use_appendix), on the retained states
    `[S_conc, S_ref, S_scrap]`.

    Used for `P_i^use`, the probability that a unit reaches use at least once
    before final disposal, and hence for the conditional use count
    `upsilon_1 / P_1^use` (Ch2 Eq. conditional_use_appendix).
    """
    keep = np.array([0, 1, 5])
    return np.asarray(A, float)[..., keep[:, None], keep[None, :]], keep


def prob_reaches_use(A):
    """`P_i^use = c_use^T (-A_pre)^-1 e_i` on the retained states, lifted back
    to a length-6 vector with NaN in the removed in-use slots (a unit that
    starts in use has already reached use, so the quantity is not defined by
    this construction)."""
    Ap, keep = A_pre(A)
    Np = np.linalg.inv(-Ap)
    c = flow_rows(A)["use_entry"][..., keep]
    p = np.einsum("...i,...ij->...j", c, Np)
    out = np.full(np.asarray(A).shape[:-2] + (N_ODE,), np.nan)
    out[..., keep] = p
    return out


# ---------------------------------------------------------------------------
# eigen-sensitivity                       # Ch2 Eq. eigenvalue_sensitivity
# ---------------------------------------------------------------------------
def eig_sensitivity(A, dA):
    """`dlambda_k/dp_n = q_k^T (dA/dp_n) v_k / (q_k^T v_k)`.

    `A`  (..., 6, 6); `dA` (..., n_p, 6, 6).  Returns
    `lam` (..., 6), `dlam` (..., 6, n_p), `cond` (..., 6) = `1/|q_k.v_k|` at
    unit norms, and `relgap` (..., 6) — the guard.  Valid for simple and
    semisimple eigenvalues only, hence the guard is returned alongside rather
    than left to the caller to remember.
    """
    A = np.asarray(A, float)
    dA = np.asarray(dA, float)
    flat = A.reshape(-1, N_ODE, N_ODE)
    dflat = dA.reshape(flat.shape[0], -1, N_ODE, N_ODE)
    n_p = dflat.shape[1]
    lam = np.zeros((flat.shape[0], N_ODE), complex)
    dlam = np.zeros((flat.shape[0], N_ODE, n_p), complex)
    cond = np.zeros((flat.shape[0], N_ODE))
    for m in range(flat.shape[0]):
        w, ql, vr = sla.eig(flat[m], left=True, right=True)
        vr = vr / np.linalg.norm(vr, axis=0)
        ql = ql / np.linalg.norm(ql, axis=0)
        den = np.einsum("ik,ik->k", ql.conj(), vr)
        cond[m] = 1.0 / np.abs(den)
        ql = ql / den.conj()                       # q_k . v_k = 1
        lam[m] = w
        # q_k^H (dA/dp) v_k, for every parameter at once
        dlam[m] = np.einsum("ik,pij,jk->kp", ql.conj(), dflat[m], vr)
    d = np.abs(lam[:, :, None] - lam[:, None, :])
    d[:, np.arange(N_ODE), np.arange(N_ODE)] = np.inf
    relgap = d.min(-1) / np.maximum(np.abs(lam), 1e-300)
    shape = A.shape[:-2]
    return (lam.reshape(shape + (N_ODE,)),
            dlam.reshape(shape + (N_ODE, n_p)),
            cond.reshape(shape + (N_ODE,)),
            relgap.reshape(shape + (N_ODE,)))


# ---------------------------------------------------------------------------
# fundamental-matrix sensitivities        # Ch2 Eq. tau_sensitivity,
#                                         #     Eq. flow_count_sensitivity
# ---------------------------------------------------------------------------
def indicator_sensitivity(A, dA):
    """`dtau_i/dp_n` and `dUpsilon_{m,i}/dp_n` at frozen coefficients.

    From `dN = N (dA) N` (Ch2 Eq. fundamental_matrix_derivative):

        dtau_i/dp        = 1^T N (dA/dp) N e_i
        dUpsilon_{m,i}/dp = (dc_m/dp)^T N e_i + c_m^T N (dA/dp) N e_i

    The first term of the second line is Chapter 2's `phi_lm e_k^T N_T e_i`,
    generalised from "flow m has rate parameter alpha_m" to "flow m has rate
    functional c_m(p)" — see the module docstring, mapping decision 2.

    Returns `dtau` (..., n_p, 6) and `{flow: (..., n_p, 6)}`.
    """
    A = np.asarray(A, float)
    dA = np.asarray(dA, float)
    N = fundamental(A)
    ones = np.ones(N.shape[:-2] + (N_ODE,))
    # M[p] = N (dA/dp) N
    M = np.einsum("...ij,...pjk,...kl->...pil", N, dA, N)
    dtau = np.einsum("...i,...pij->...pj", ones, M)

    rows = flow_rows(A)
    drows = flow_rows(dA)          # linear in A, so this is dc_m/dp
    dcounts = {}
    for k in FLOW_NAMES:
        term1 = np.einsum("...pi,...ij->...pj", drows[k], N)
        term2 = np.einsum("...i,...pij->...pj", rows[k], M)
        dcounts[k] = term1 + term2
    return dtau, dcounts, N


# ---------------------------------------------------------------------------
# the continuum: interpolation and extrapolation of the coefficient path
# ---------------------------------------------------------------------------
def _clip_to_feasible(P):
    """Project a parameter array back onto the feasible set: alphas >= 0,
    binary taus in [0, 1], each simplex non-negative and renormalised, mu
    untouched.  Applied after extrapolation so that the extrapolated `A`
    stays Metzler with `l >= 0` — i.e. so the extrapolated system still
    conserves mass and cannot create it."""
    P = np.array(P, float, copy=True)
    P[..., 0:4] = np.clip(P[..., 0:4], 0.0, None)
    P[..., 4:8] = np.clip(P[..., 4:8], 0.0, 1.0)
    for idx in SIMPLEXES.values():
        blk = np.clip(P[..., list(idx)], 0.0, None)
        s = blk.sum(-1, keepdims=True)
        P[..., list(idx)] = np.where(s > 0, blk / np.maximum(s, 1e-300),
                                     1.0 / len(idx))
    return P


def coeff_path(P_nodes, years, t, rule="hold", trend_window=10, horizon=20.0):
    """`P(t)` for arbitrary `t`, given node values `P_nodes` (..., T, n_p).

    Inside `[years[0], years[-1]]`: linear interpolation between nodes, so the
    path reproduces WP-2a exactly at the nodes.  Beyond `years[-1]`, one of
    the extrapolation rules below; `Phi` needs coefficients over the whole
    absorption horizon, and the spec requires the rule to be stated and its
    influence tested.

      `hold`     freeze at the 2019 node.
      `mean{W}`  freeze at the mean of the last W nodes — same "no further
                 change" assumption as `hold` but insensitive to endpoint
                 noise in the fitted coefficients.
      `trend`    least-squares linear trend over the last `trend_window`
                 nodes, continued for `horizon` years, then frozen; clipped
                 to the feasible set at every step.  WP-2a found every
                 learned entry of A rising by 1.78-3.32x over the window, so
                 this is the rule under which the past 40 years' direction of
                 travel is assumed to persist.

    Below `years[0]` the first node is held; this only matters if a caller
    asks for it, which none of WP-2c/d/e does.
    """
    P_nodes = np.asarray(P_nodes, float)
    years = np.asarray(years, float)
    t = np.atleast_1d(np.asarray(t, float))
    T = years.size
    y0, y1 = years[0], years[-1]

    # --- interior: linear interpolation ------------------------------------
    tc = np.clip(t, y0, y1)
    j = np.clip(np.searchsorted(years, tc, side="right") - 1, 0, T - 2)
    w = (tc - years[j]) / (years[j + 1] - years[j])
    lo = np.take(P_nodes, j, axis=-2)
    hi = np.take(P_nodes, j + 1, axis=-2)
    P = lo + (hi - lo) * w[..., None]

    if rule == "hold":
        end = P_nodes[..., -1, :]
        P = np.where((t > y1)[..., None], end[..., None, :], P)
        t_switch = y1
    elif rule.startswith("mean"):
        W = int(rule[4:])
        end = P_nodes[..., -W:, :].mean(axis=-2)
        end = _clip_to_feasible(end)
        P = np.where((t > y1)[..., None], end[..., None, :], P)
        t_switch = y1
    elif rule.startswith("trend"):
        H = float(rule[5:]) if len(rule) > 5 else horizon
        W = int(trend_window)
        x = years[-W:] - y1
        Xd = np.stack([np.ones_like(x), x], 1)
        Y = P_nodes[..., -W:, :]
        coef = np.linalg.lstsq(Xd, np.moveaxis(Y, -2, 0).reshape(W, -1),
                               rcond=None)[0]
        slope = coef[1].reshape(Y.shape[:-2] + (Y.shape[-1],))
        # anchored at the 2019 node, so the path is continuous there and the
        # rule contributes only a direction of travel, not a level jump
        dt = np.clip(t - y1, 0.0, H)
        ex = P_nodes[..., -1, None, :] + slope[..., None, :] * dt[..., None]
        ex = _clip_to_feasible(ex)
        P = np.where((t > y1)[..., None], ex, P)
        t_switch = y1 + H
    else:
        raise ValueError(f"unknown extrapolation rule {rule!r}")
    return P, float(t_switch)


RULES = ["hold", "mean5", "trend20", "trend50"]


# ---------------------------------------------------------------------------
# the transition operator Phi(t0+u, t0) and its integrals
# ---------------------------------------------------------------------------
def _step_operators(A_mid, h):
    """For piecewise-constant `A` on a step of width `h`, return both
    `expm(A h)` and `int_0^h expm(A s) ds` from a single augmented
    exponential

        expm([[A, I], [0, 0]] h) = [[e^{Ah}, int_0^h e^{As} ds], [0, I]].

    Evaluating `A` at the step midpoint makes the scheme the exponential
    midpoint rule, second order in `h` and exact whenever `A` is constant —
    which is precisely the regime beyond `t_switch`, so the tail below carries
    no discretisation error at all.
    """
    A_mid = np.asarray(A_mid, float)
    shape = A_mid.shape[:-2]
    M = np.zeros(shape + (2 * N_ODE, 2 * N_ODE))
    M[..., :N_ODE, :N_ODE] = A_mid * h
    idx = np.arange(N_ODE)
    M[..., idx, N_ODE + idx] = h
    E = sla.expm(M)
    return E[..., :N_ODE, :N_ODE], E[..., :N_ODE, N_ODE:]


def nonautonomous_indicators(P_nodes, years, rule="hold", n_sub=64,
                             trend_window=10, horizon=20.0, t0_idx=None):
    """Chapter 2's non-autonomous lifetime and flow counts, at every year node.

        tau_FD,i(t0)  = int_0^inf 1^T Phi(t0+u, t0) e_i du      # Ch2 Eq. nonautonomous_lifetime
        Upsilon_m,i(t0) = int_0^inf c_m(t0+u)^T Phi(t0+u,t0) e_i du
                                                                # Ch2 Eq. nonautonomous_use_count

    `Phi` is generated by the fitted `A_FD(t)` (= `A(t)`, mapping decision 1).
    The integral is split at `t_switch`, beyond which the coefficient path is
    constant by construction of every rule in `coeff_path`:

        int_0^inf = int_0^{U} (quadrature)  +  c_const^T N_const Phi(U)

    so the infinite tail is **closed in closed form** rather than truncated.
    The only error is the O(h^2) exponential-midpoint error on the finite part,
    which `run_wp2d.py` measures against a stiff reference integrator and by
    step halving.

    Returns a dict of `(n_seed, n_t0, 6)` arrays plus diagnostics.  Vectorised
    over seeds and over `t0` simultaneously: the substep propagators are built
    once on a common fine grid and then accumulated forward, so the cost is one
    batched `expm` per substep rather than one integration per (seed, t0).
    """
    P_nodes = np.asarray(P_nodes, float)
    years = np.asarray(years, float)
    S = P_nodes.shape[0]
    t0s = np.arange(years.size) if t0_idx is None else np.asarray(t0_idx, int)
    n0 = t0s.size

    h = 1.0 / n_sub
    t_end = coeff_path(P_nodes, years, np.array([years[-1]]), rule=rule,
                       trend_window=trend_window, horizon=horizon)[1]
    n_steps = int(round((t_end - years[0]) * n_sub))
    edges = years[0] + np.arange(n_steps + 1) * h
    mids = 0.5 * (edges[:-1] + edges[1:])

    P_mid, _ = coeff_path(P_nodes, years, mids, rule=rule,
                          trend_window=trend_window, horizon=horizon)
    A_mid = assemble_from_params(P_mid)                 # (S, n_steps, 6, 6)
    c_mid = flow_rows(A_mid)                            # each (S, n_steps, 6)

    # constant tail
    P_end, _ = coeff_path(P_nodes, years, np.array([t_end + 1.0]), rule=rule,
                          trend_window=trend_window, horizon=horizon)
    A_end = assemble_from_params(P_end)[:, 0]           # (S, 6, 6)
    s_end = spectral_abscissa(A_end)
    end_stable = s_end < -1e-9
    N_end = fundamental(A_end, on_singular="nan")
    c_end = {k: v[:, 0] for k, v in flow_rows(A_end[:, None]).items()}

    # substep propagators, batched over seeds to keep peak memory bounded
    Pr = np.empty((S, n_steps, N_ODE, N_ODE))
    Qi = np.empty((S, n_steps, N_ODE, N_ODE))
    for s in range(S):
        Pr[s], Qi[s] = _step_operators(A_mid[s], h)

    # forward accumulation, all t0 at once
    start = np.searchsorted(edges, years[t0s] - 1e-9)   # step index of each t0
    # Phi starts at the identity and is only propagated once its own t0 is
    # reached, so a t0 sitting exactly at `t_switch` (t0 = 2019 under `hold`)
    # correctly returns the pure closed-form tail, i.e. the frozen indicator.
    Phi = np.broadcast_to(np.eye(N_ODE), (S, n0, N_ODE, N_ODE)).copy()
    J = np.zeros((S, n0, N_ODE))                        # int 1^T Phi
    K = {k: np.zeros((S, n0, N_ODE)) for k in FLOW_NAMES}
    live = np.zeros(n0, bool)
    ones = np.ones(N_ODE)
    for j in range(n_steps):
        live |= start == j
        if not live.any():
            continue
        Pj, Qj = Pr[:, j], Qi[:, j]
        Phil = Phi[:, live]
        # accumulate the integrals over this step BEFORE propagating:
        # int_{u_j}^{u_j+h} X(t)^T Phi(t) dt = X_mid^T Q_j Phi(u_j)
        QPhi = np.einsum("sij,snjk->snik", Qj, Phil)
        J[:, live] += np.einsum("i,snij->snj", ones, QPhi)
        for k in FLOW_NAMES:
            K[k][:, live] += np.einsum("si,snij->snj", c_mid[k][:, j], QPhi)
        Phi[:, live] = np.einsum("sij,snjk->snik", Pj, Phil)

    # closed-form tail from t_switch onward
    tail_tau = np.einsum("i,sij,snjk->snk", ones, N_end, Phi)
    tau = J + tail_tau
    counts = {}
    for k in FLOW_NAMES:
        counts[k] = K[k] + np.einsum("si,sij,snjk->snk", c_end[k], N_end, Phi)

    return dict(tau=tau, counts=counts, years_t0=years[t0s],
                Phi_at_switch=Phi, t_switch=t_end, n_steps=n_steps,
                A_end=A_end, spectral_abscissa_end=s_end,
                end_stable=end_stable,
                tail_share_tau=tail_tau / np.maximum(tau, 1e-300))


# ---------------------------------------------------------------------------
# statistics (project convention: HL intervals, not significance tests)
# ---------------------------------------------------------------------------
def hodges_lehmann(v, conf=0.95):
    """HL location estimate + distribution-free CI.  Same implementation as
    `run_wp1b.py` / `run_wp1d.py` / `run_wp2b.py`."""
    from scipy import stats
    v = np.asarray([x for x in np.asarray(v, float) if np.isfinite(x)])
    n = v.size
    if n == 0:
        return dict(hl=np.nan, lo=np.nan, hi=np.nan, n=0)
    w = (v[:, None] + v[None, :]) / 2.0
    walsh = np.sort(w[np.triu_indices(n)])
    est = float(np.median(walsh))
    if n < 6:
        return dict(hl=est, lo=np.nan, hi=np.nan, n=n)
    N = walsh.size
    z = stats.norm.ppf(1.0 - (1.0 - conf) / 2.0)
    sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    k = max(int(np.floor(N / 2.0 - z * sd)), 0)
    return dict(hl=est, lo=float(walsh[k]),
                hi=float(walsh[min(N - 1 - k, N - 1)]), n=n)


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------
def verify_assembly(d, A_ref):
    """`assemble_from_params` against `zinc_A_lab.assemble_A` — i.e. against
    the transcription WP-2a already verified to machine precision against
    `zinc_colloc_v5.make_rhs` itself."""
    P = pack_params(d, np.asarray(d["mu_cohorts"], float))
    A = assemble_from_params(P)
    return float(np.max(np.abs(A - np.asarray(A_ref, float))))


def verify_derivatives(P, rel=1e-5):
    """Complex-step `dA/dp` against a central difference, and the frozen
    eigen-sensitivity against a finite difference of the eigenvalue itself.

    The second check is the one that matters: it tests
    Ch2 Eq. eigenvalue_sensitivity as a statement about `lambda(A(p))`, not
    just the linear algebra used to evaluate it.  Errors are normalised by the
    largest sensitivity of the same mode over all parameters, not by the
    sensitivity being tested: a mode with `|lambda| ~ 5 /yr` and
    `dlambda/dp ~ 1e-5` cannot be resolved to its own relative precision by any
    finite difference, and normalising per-entry would report double-precision
    round-off in `eig` as a formula error.
    """
    P = np.asarray(P, float)
    dA = dA_dparams(P)
    err_dA = 0.0
    for n in range(N_PARAM):
        step = rel * np.maximum(np.abs(P[..., n]), 1e-3)
        Pp, Pm = P.copy(), P.copy()
        Pp[..., n] += step
        Pm[..., n] -= step
        fd = (assemble_from_params(Pp) - assemble_from_params(Pm)) / (
            2 * step[..., None, None])
        scale = np.maximum(np.abs(dA[..., n, :, :]).max(), 1e-12)
        err_dA = max(err_dA, float(np.max(np.abs(fd - dA[..., n, :, :])) / scale))

    A = assemble_from_params(P)
    lam, dlam, cond, relgap = eig_sensitivity(A, dA)
    order = np.argsort(-lam.real, axis=-1)
    an = np.take_along_axis(dlam.real, order[..., None], axis=-2)
    scale = np.maximum(np.abs(an).max(axis=-1), 1e-12)      # per (node, mode)
    err_lam = 0.0
    for n in range(N_PARAM):
        step = rel * np.maximum(np.abs(P[..., n]), 1e-3)
        Pp, Pm = P.copy(), P.copy()
        Pp[..., n] += step
        Pm[..., n] -= step
        lp = np.sort(np.linalg.eigvals(assemble_from_params(Pp)).real,
                     axis=-1)[..., ::-1]
        lm = np.sort(np.linalg.eigvals(assemble_from_params(Pm)).real,
                     axis=-1)[..., ::-1]
        fd = (lp - lm) / (2 * step[..., None])
        err_lam = max(err_lam,
                      float(np.max(np.abs(fd - an[..., n]) / scale)))
    return dict(max_rel_err_dA=err_dA, max_rel_err_dlambda=err_lam)


def verify_indicator_derivatives(P, rel=1e-5):
    """`dtau/dp` and `dUpsilon/dp` against central differences of the
    indicators themselves (Ch2 Eq. tau_sensitivity, Eq. flow_count_sensitivity).

    Normalised, as in `verify_derivatives`, by the largest sensitivity of the
    same indicator over all parameters.
    """
    P = np.asarray(P, float)
    A = assemble_from_params(P)
    dA = dA_dparams(P)
    dtau, dcounts, _ = indicator_sensitivity(A, dA)
    sc_t = np.maximum(np.abs(dtau).max(axis=-2), 1e-12)
    sc_c = {k: np.maximum(np.abs(v).max(axis=-2), 1e-12)
            for k, v in dcounts.items()}
    err_t, err_c = 0.0, 0.0
    for n in range(N_PARAM):
        step = rel * np.maximum(np.abs(P[..., n]), 1e-3)
        Pp, Pm = P.copy(), P.copy()
        Pp[..., n] += step
        Pm[..., n] -= step
        tp, cp, _ = frozen_indicators(assemble_from_params(Pp))
        tm, cm, _ = frozen_indicators(assemble_from_params(Pm))
        fd = (tp - tm) / (2 * step[..., None])
        err_t = max(err_t, float(np.max(np.abs(fd - dtau[..., n, :]) / sc_t)))
        for k in FLOW_NAMES:
            fdc = (cp[k] - cm[k]) / (2 * step[..., None])
            err_c = max(err_c, float(np.max(
                np.abs(fdc - dcounts[k][..., n, :]) / sc_c[k])))
    return dict(max_rel_err_dtau=err_t, max_rel_err_dUpsilon=err_c)


# ===========================================================================
# WP-2g additions: frozen equilibrium, saturation, transient amplification
# ===========================================================================
# WP-2a established `use_stock_input=False` and measured coefficient
# state-dependence at exactly 0.0, so `dS/dt = A(t) S + b(t)` holds
# IDENTICALLY and the system is linear in the state.  Everything below is
# therefore a closed form for the fitted model, not a linearisation of it.

def equilibrium(A, b, on_singular="raise"):
    """Frozen-time equilibrium `S*(t) = -A(t)^-1 b(t) = N(t) b(t)`.

    # Ch2 Eq. system   (dS/dt = A S + b, so dS/dt = 0 <=> S = -A^-1 b)
    # Ch2 Eq. vecb     (b = [E(t), 0, ..., 0])

    WP-2d verified `-A` is a non-singular M-matrix at all 1 400 nodes, so `N`
    exists, is unique and is entrywise non-negative; with `b >= 0` that makes
    `S* >= 0` without any further assumption.  `A` (..., 6, 6), `b` (..., 6).
    """
    N = fundamental(A, on_singular=on_singular)
    return np.einsum("...ij,...j->...i", N, np.asarray(b, float)), N


def equilibrium_sensitivity(A, dA, b, db=None):
    """`dS*/dp = N (dA/dp) N b - N (db/dp)`, shape (..., n_p, 6).

    Differentiating `S* = N b` with `dN = N (dA) N`
    (# Ch2 Eq. fundamental_matrix_derivative, the same identity WP-2e uses)
    gives `dS*/dp = (dN) b + N db = N (dA) N b + N db`; the sign convention
    here follows the spec's statement of the formula, in which `dA` enters
    through `N = (-A)^-1` so that `d(-A)^-1 = N (dA) N`.  Validated against
    central differences of `S*(p +- h)` by `verify_equilibrium_derivatives`.

    `db` is `dS*/dp` through the forcing; `b = [cp, 0, ...]` and none of the
    20 parameters of `PARAM_NAMES` enters `cp`, so it is zero here and is
    carried only so the routine stays correct if a driver-dependent `b` is
    ever differentiated.
    """
    A = np.asarray(A, float)
    dA = np.asarray(dA, float)
    b = np.asarray(b, float)
    N = fundamental(A)
    Nb = np.einsum("...ij,...j->...i", N, b)
    out = np.einsum("...ij,...pjk,...k->...pi", N, dA, Nb)
    if db is not None:
        out = out + np.einsum("...ij,...pj->...pi", N, np.asarray(db, float))
    return out, N


def numerical_abscissa(A):
    """`omega(A) = lambda_max((A + A^T)/2)`.

    The initial growth rate of `||exp(A u)||_2`: `d/du ||exp(Au)x||` at
    `u = 0` is bounded by `omega(A)`, and `omega(A) > s(A)` exactly when `A`
    is non-normal.  A stable system (`s(A) < 0`) with `omega(A) > 0` grows
    before it decays.
    """
    A = np.asarray(A, float)
    return np.linalg.eigvalsh(0.5 * (A + np.swapaxes(A, -1, -2)))[..., -1]


def frozen_amplification(A, u):
    """`sup_u ||exp(A u)||_2` for FROZEN `A`, plus the maximiser.

    The Kreiss matrix theorem bounds the AUTONOMOUS semigroup `exp(Au)`, not
    the non-autonomous `Phi(t+u,t)`, so this is the quantity the Kreiss
    constant may legitimately be checked against.  Reporting both also
    separates how much of the transient amplification is a property of the
    frozen matrix and how much the time variation adds.

    Every eigenvalue is real and simple here (WP-2b: `max |Im lambda| = 0.0`
    exactly, min relative gap 0.0079), so `exp(Au) = V diag(exp(lam u)) V^-1`
    is exact and vectorises over `u` — no repeated `expm`.
    """
    A = np.asarray(A, float)
    u = np.asarray(u, float).ravel()
    lam, V = np.linalg.eig(A)
    lam, V = lam.real, V.real
    Vi = np.linalg.inv(V)
    E = np.exp(lam[..., None, :] * u[:, None])               # (..., n_u, 6)
    M = np.einsum("...ik,...uk,...kj->...uij", V, E, Vi)
    n2 = np.linalg.svd(M, compute_uv=False)[..., 0]
    k = np.argmax(n2, axis=-1)
    return (np.take_along_axis(n2, k[..., None], axis=-1)[..., 0],
            u[k], n2)


def kreiss_constant(A, n_re=48, n_im=48, span=6.0):
    """`K(A) = sup_{Re z > 0} Re(z) ||(zI - A)^-1||_2`, by grid search.

    The Kreiss matrix theorem gives `K(A) <= sup_u ||exp(Au)||_2 <= e n K(A)`
    for an `n x n` matrix, so `K` is a rigorous LOWER bound on the transient
    amplification and `e n K` an upper one.  Reported alongside the directly
    computed supremum as an independent check that the amplification is real
    and not a quadrature artefact.

    The sup is attained at finite `z`; the grid is log-spaced in `Re z` over
    `[eps, span * max|lambda|]` and linear in `Im z` over `[0, span *
    max|lambda|]`, using conjugate symmetry (`A` real) to halve the work.
    `K >= 1` always, since `Re(z) ||(zI-A)^-1|| -> 1` as `z -> +inf` along the
    real axis.
    """
    A = np.asarray(A, float)
    flat = A.reshape(-1, A.shape[-1], A.shape[-1])
    n = A.shape[-1]
    out = np.empty(flat.shape[0])
    I = np.eye(n)
    for m in range(flat.shape[0]):
        scale = max(float(np.abs(np.linalg.eigvals(flat[m])).max()), 1e-12)
        re = np.geomspace(1e-4 * scale, span * scale, n_re)
        im = np.linspace(0.0, span * scale, n_im)
        Z = re[:, None] + 1j * im[None, :]
        R = np.linalg.inv(Z[..., None, None] * I - flat[m])
        s = np.linalg.svd(R, compute_uv=False)[..., 0]
        out[m] = float(np.max(Z.real * s))
    return out.reshape(A.shape[:-2])


def frozen_rollout(A, b, S0, u):
    """`S(t0 + u)` for FROZEN `A`, `b`: the closed form
    `S(u) = S* + exp(A u) (S0 - S*)`, evaluated on the grid `u`.

    Returns `S` of shape (..., n_u, 6) and the modal decomposition of the gap:
    with `A = V diag(lambda) V^-1` (all eigenvalues real and simple here —
    WP-2b: `max |Im lambda| = 0.0` exactly, min relative gap 0.0079),

        S(u) - S* = sum_k c_k exp(lambda_k u) v_k,    c = V^-1 (S0 - S*),

    so `contrib[..., u, k, i] = c_k exp(lambda_k u) v_ki` says which mode
    carries the residual gap at each horizon.

    A frozen counterfactual, NOT a forecast.
    """
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    S0 = np.asarray(S0, float)
    u = np.asarray(u, float).ravel()
    Sstar, _ = equilibrium(A, b)
    gap0 = S0 - Sstar
    lam, V = np.linalg.eig(A)
    lam = lam.real
    c = np.linalg.solve(V.real, gap0[..., None])[..., 0]
    E = np.exp(lam[..., None, :] * u[:, None])               # (..., n_u, 6)
    contrib = E[..., None] * (c[..., None, :, None] * np.swapaxes(
        V.real, -1, -2)[..., None, :, :])                    # (...,n_u,6,6)
    S = Sstar[..., None, :] + contrib.sum(axis=-2)
    return S, Sstar, contrib, lam, V.real


def time_to_fraction(S, Sstar, u, fracs=(0.90, 0.95, 0.99)):
    """First `u` at which `S(u)` reaches `frac * S*`, per stock.

    Approach is from below wherever `S0 < S*`; where it is from above the
    crossing is defined symmetrically as `S(u) <= (2 - frac) * S*`.  Returns
    NaN where the level is not reached inside the grid, with the caller
    expected to report that rather than to extend the horizon silently.
    """
    S = np.asarray(S, float)
    Sstar = np.asarray(Sstar, float)
    u = np.asarray(u, float).ravel()
    below = S[..., 0, :] < Sstar
    out = np.full(S.shape[:-2] + (len(fracs), S.shape[-1]), np.nan)
    for j, f in enumerate(fracs):
        tgt = np.where(below, f * Sstar, (2.0 - f) * Sstar)
        hit = np.where(below[..., None, :], S >= tgt[..., None, :],
                       S <= tgt[..., None, :])
        any_hit = hit.any(axis=-2)
        idx = np.argmax(hit, axis=-2)
        out[..., j, :] = np.where(any_hit, u[idx], np.nan)
    return out


def amplification_scan(P_nodes, years, t0_idx, u_max=60.0, n_sub=16,
                       rule="hold", trend_window=10, horizon=20.0,
                       keep_full=()):
    """`||Phi(t0+u, t0)||` along `u`, for the NON-autonomous system.
                                             # Ch2 Eq. product_transition_matrices

    Same exponential-midpoint propagator as `nonautonomous_indicators` (which
    `run_wp2d.py` validated against a stiff reference integrator to 0.0006%
    and by step halving), and the same structural trick: every `t0` shares one
    fine grid of substep propagators, so the expensive `expm` is paid once per
    (seed, substep) rather than once per (seed, t0, substep).  What differs
    from WP-2d is that the SHAPE of `||Phi||` in `u` is the quantity of
    interest here rather than its integral, so the norm is recorded at every
    substep and all `t0` are accumulated simultaneously.

    Two norms are tracked, and the pair is the point:

      `||.||_2`  can exceed 1 — `A(t)` is strongly non-normal (Metzler, and
                 near block-triangular: WP-2c notes row 0 has no off-diagonal);
      `||.||_1`  is the largest surviving mass fraction from any single
                 starting stock, and mass balance (`1^T A = -l^T`, `l >= 0`)
                 forces it to be non-increasing.  Checked, not assumed.

    `Phi` itself is not retained for every `(seed, t0, u)` — 0.4 GB at the
    default grid — only the norm traces, the maximiser and `Phi` there.
    `keep_full` names `t0` indices whose full trajectory is also returned.

    Returns a dict; see `run_wp2g.py` for the fields used.
    """
    P_nodes = np.asarray(P_nodes, float)
    years = np.asarray(years, float)
    S = P_nodes.shape[0]
    t0s = np.asarray(t0_idx, int)
    n0 = t0s.size
    h = 1.0 / n_sub
    n_u = int(round(u_max * n_sub))

    # one fine grid covering every (t0, u) pair that will be asked for
    t_lo = float(years[t0s].min())
    t_hi = float(years[t0s].max()) + u_max
    n_steps = int(round((t_hi - t_lo) * n_sub))
    edges = t_lo + np.arange(n_steps + 1) * h
    mids = 0.5 * (edges[:-1] + edges[1:])
    P_mid, _ = coeff_path(P_nodes, years, mids, rule=rule,
                          trend_window=trend_window, horizon=horizon)
    A_mid = assemble_from_params(P_mid)                      # (S, n_steps, 6,6)

    start = np.rint((years[t0s] - t_lo) * n_sub).astype(int)
    u = np.arange(n_u + 1) * h
    norms = np.full((S, n0, n_u + 1), np.nan)
    norms1 = np.full((S, n0, n_u + 1), np.nan)
    col_sup = np.zeros((S, n0, N_ODE))
    col_u = np.zeros((S, n0, N_ODE))
    Phi_at = np.broadcast_to(np.eye(N_ODE), (S, n0, N_ODE, N_ODE)).copy()
    best = np.ones((S, n0))
    k_at = np.zeros((S, n0), int)
    norms[:, :, 0] = 1.0
    norms1[:, :, 0] = 1.0
    col_sup[:] = 1.0
    full = {int(t): [np.broadcast_to(np.eye(N_ODE), (S, N_ODE, N_ODE)).copy()]
            for t in keep_full}

    cur = np.broadcast_to(np.eye(N_ODE), (S, n0, N_ODE, N_ODE)).copy()
    live = np.zeros(n0, bool)
    done = np.zeros(n0, bool)
    for j in range(n_steps):
        live |= (start == j)
        act = live & ~done
        if not act.any():
            continue
        Pj, _ = _step_operators(A_mid[:, j], h)               # (S, 6, 6)
        cur[:, act] = np.einsum("sij,snjk->snik", Pj, cur[:, act])
        idx = j - start + 1                                   # u index per t0
        sel = np.where(act)[0]
        sub = cur[:, sel]
        nj = np.linalg.svd(sub, compute_uv=False)[..., 0]
        n1 = np.abs(sub).sum(-2).max(-1)
        cn = np.linalg.norm(sub, axis=-2)
        for m_i, m in enumerate(sel):
            iu = idx[m]
            norms[:, m, iu] = nj[:, m_i]
            norms1[:, m, iu] = n1[:, m_i]
            upd = cn[:, m_i] > col_sup[:, m]
            col_u[:, m] = np.where(upd, u[iu], col_u[:, m])
            col_sup[:, m] = np.where(upd, cn[:, m_i], col_sup[:, m])
            better = nj[:, m_i] > best[:, m]
            best[:, m] = np.where(better, nj[:, m_i], best[:, m])
            k_at[:, m] = np.where(better, iu, k_at[:, m])
            Phi_at[:, m] = np.where(better[:, None, None], sub[:, m_i],
                                    Phi_at[:, m])
            ti = int(t0s[m])
            if ti in full:
                full[ti].append(sub[:, m_i].copy())
            if iu >= n_u:
                done[m] = True

    for k in list(full):
        full[k] = np.stack(full[k], axis=1)                   # (S, n_u+1, 6,6)

    sup = np.take_along_axis(norms, k_at[..., None], axis=-1)[..., 0]
    # Which stock pair carries the amplification.  The largest ENTRY of Phi is
    # the wrong readout — it is always the slow in-use diagonal, which is a
    # statement about decay, not growth.  The 2-norm is attained on the
    # leading singular pair, so `v_1` is the initial state that gets amplified
    # and `u_1` is the state it lands in.
    U, _sv, Vt = np.linalg.svd(Phi_at)
    from_idx = np.argmax(np.abs(Vt[..., 0, :]), axis=-1)
    to_idx = np.argmax(np.abs(U[..., :, 0]), axis=-1)
    return dict(norms=norms, norms1=norms1, u=u, sup=sup, u_at=u[k_at],
                sup1=np.nanmax(norms1, axis=-1), col_sup=col_sup, col_u=col_u,
                to_idx=to_idx, from_idx=from_idx,
                sing_in=Vt[..., 0, :], sing_out=U[..., :, 0],
                Phi_at=Phi_at, full=full, n_sub=n_sub, u_max=u_max)


# ---------------------------------------------------------------------------
# WP-2g verification
# ---------------------------------------------------------------------------
def verify_equilibrium(A, b, u_max=4000.0, rtol=1e-11, atol=1e-8):
    """`S* = -A^-1 b` against an INDEPENDENT numerical reference.

    The frozen system `dS/du = A S + b` is integrated from `S = 0` with the
    same stiff reference integrator WP-2d used (`solve_ivp`, LSODA, rtol
    1e-11) and the terminal state compared against the linear solve.  Nothing
    is shared between the two routes — one is `numpy.linalg.inv`, the other a
    variable-order multistep integrator — so agreement is a real check rather
    than a restatement.  (Using `frozen_rollout` here instead would be
    vacuous: at `u_max` the transient is `exp(-80)` and the closed form
    returns `S*` bit-exactly by construction.)
    """
    from scipy.integrate import solve_ivp

    A = np.asarray(A, float).reshape(-1, N_ODE, N_ODE)
    b = np.asarray(b, float).reshape(-1, N_ODE)
    Sstar, _ = equilibrium(A, b)
    worst_abs = worst_rel = 0.0
    for m in range(A.shape[0]):
        sol = solve_ivp(lambda _t, y, M=A[m], c=b[m]: M @ y + c,
                        (0.0, u_max), np.zeros(N_ODE), method="LSODA",
                        rtol=rtol, atol=atol, dense_output=False)
        err = np.abs(sol.y[:, -1] - Sstar[m])
        worst_abs = max(worst_abs, float(err.max()))
        worst_rel = max(worst_rel,
                        float((err / max(np.abs(Sstar[m]).max(), 1e-300)).max()))
    return dict(max_abs=worst_abs, max_rel=worst_rel, n_nodes=int(A.shape[0]),
                u_max=float(u_max))


def verify_equilibrium_derivatives(P, b, rel=1e-5):
    """`dS*/dp` against central differences of `S*(p +- h)`, as WP-2c did for
    `dlambda/dp`.  Normalised per node by the largest sensitivity of that node
    over all parameters, for the same reason WP-2c gives: a parameter with a
    1e-12 effect cannot be resolved to its own relative precision."""
    P = np.asarray(P, float)
    b = np.asarray(b, float)
    A = assemble_from_params(P)
    dA = dA_dparams(P)
    dS, _ = equilibrium_sensitivity(A, dA, b)
    # The UNCONSTRAINED partial is what `equilibrium_sensitivity` returns, so
    # the finite difference must not renormalise the simplices — projecting
    # back onto the feasible set would measure a different direction
    # entirely.  (Ch2's constrained directions are the differences of two
    # unconstrained partials, `simplex_directions`, and are exact once these
    # are.)
    scale = np.maximum(np.abs(dS).max(axis=(-2, -1))[..., None], 1e-12)
    worst = 0.0
    for n in range(N_PARAM):
        step = rel * np.maximum(np.abs(P[..., n]), 1e-3)
        Pp, Pm = P.copy(), P.copy()
        Pp[..., n] += step
        Pm[..., n] -= step
        Sp, _ = equilibrium(assemble_from_params(Pp), b)
        Sm, _ = equilibrium(assemble_from_params(Pm), b)
        fd = (Sp - Sm) / (2 * step[..., None])
        worst = max(worst, float(np.max(np.abs(fd - dS[..., n, :]) / scale)))
    return worst


# ===========================================================================
# WP-2f additions: structural stability and distance to bifurcation
# ===========================================================================
# Gandolfo (1992) §2.  For characteristic root `mu_j` and parameter `theta_i`
# with (asymptotic) standard error `sigma_i`, the BIFURCATION VALUE of
# `theta_i` is `theta_i + dtheta_i` with
#
#     dtheta_i = -mu_j / (dmu_j/dtheta_i)                     # the displacement
#                                                             # that drives mu_j to 0
#
# and the system is structurally UNSTABLE with respect to `theta_i` when that
# value lies INSIDE the confidence interval:
#
#     |mu_j / (dmu_j/dtheta_i)| < z_{p/2} sigma_i             (form 1, "distance")
#     |dmu_j/dtheta_i|          > |mu_j| / (z_{p/2} sigma_i)  (form 2, "derivative")
#     |psi_ji|                  > theta_i / (z_{p/2} sigma_i) (form 3, "elasticity")
#
# with `psi_ji = (dmu_j/dtheta_i) theta_i / mu_j`.  The right-hand side of
# form 3 is the t-statistic divided by `z`, so `|psi_ji| > 1` is NECESSARY
# whenever `theta_i` is significantly non-zero.  All three are implemented and
# `gandolfo` checks they agree.
#
# `sigma` is the delicate part and is a PLUGGABLE INPUT here.  The default
# prototype is the across-seed spread, which WP-1c flag 1 establishes is a
# LOWER BOUND on the variance of `alpha(t)` — it carries initialisation and
# optimisation path only, on the same data and the same split.  The criterion
# is therefore ANTI-CONSERVATIVE under it: it will UNDER-declare structural
# instability.  WP-4c's Laplace marginals and WP-8h's combined product drop in
# through the same argument without any other change.

Z_95 = 1.959963984540054            # z_{0.025}

# Feasible set of the estimated cycle, parameter by parameter, in the same
# order as PARAM_NAMES.  `None` = unbounded on that side.  Rates are
# non-negative but not capped (alpha_cc already reaches 6.7 /yr by 2019 —
# the concentrate stock turns over several times a year); the binary taus are
# transfer coefficients in [0, 1]; simplex components are shares in [0, 1]
# (the sum-to-one constraint is handled by Ch2's directional derivative, not
# by a box); the cohort lifetimes are strictly positive.
FEASIBLE_LO = np.array([0.0] * 4 + [0.0] * 4 + [0.0] * 9 + [1e-9] * 3)
FEASIBLE_HI = np.array([np.inf] * 4 + [1.0] * 4 + [1.0] * 9 + [np.inf] * 3)


def seed_sigma(P, ddof=1):
    """Prototype `sigma_i(t)`: the across-seed standard deviation of each
    parameter at each year node.  `P` is (n_seed, n_year, n_p).

    A LOWER bound on the true parameter uncertainty (WP-1c flag 1) — same
    data, same split, so it carries only initialisation and optimisation
    path.  Label it as a prototype wherever it is used.
    """
    return np.std(np.asarray(P, float), axis=0, ddof=ddof)


def gandolfo(mu, dmu, theta, sigma, z=Z_95, tol=1e-9):
    """Gandolfo's structural-stability criterion, all three forms.

    `mu`    (...,)            the characteristic root
    `dmu`   (..., n_p)        `dmu/dtheta_i`
    `theta` (..., n_p)        the parameter values
    `sigma` (..., n_p)        their standard errors (any provider)

    Returns a dict with `dtheta` (the displacement to the bifurcation value),
    `bif_value`, `dist_sigma` = `|dtheta|/sigma`, `psi` (the elasticity), the
    per-form verdicts and `forms_agree`.

    `dmu -> 0` gives `dtheta -> inf`: the root is insensitive to that
    parameter and no perturbation of it can ever reach a bifurcation.  That
    is returned as `+inf` rather than as a numerical blow-up, and it is not
    hypothetical — WP-2c found `tau_ref`, `frac_fu_loss` and `frac_eu_loss`
    spectrally invisible, with `dmu/dtheta = 0` at every node and every mode.
    """
    mu = np.asarray(mu, float)[..., None]
    dmu = np.asarray(dmu, float)
    theta = np.asarray(theta, float)
    sigma = np.asarray(sigma, float)

    with np.errstate(divide="ignore", invalid="ignore"):
        dtheta = np.where(np.abs(dmu) > 0, -mu / dmu, np.inf * np.sign(mu + 1e-300))
        dtheta = np.where(np.abs(dmu) > 0, dtheta, np.inf)
        dist = np.abs(dtheta) / sigma
        v1 = dist < z                                          # form 1
        v2 = np.abs(dmu) > np.abs(mu) / (z * sigma)            # form 2
        psi = dmu * theta / mu                                 # form 3
        # Gandolfo writes |theta_i| / (z sigma_i) — the absolute value matters
        # for a parameter that can be negative.  Every parameter of this cycle
        # is non-negative, so this is currently a no-op; `verify_gandolfo_paper`
        # records that it is the paper's form and not a transcription of it.
        rhs3 = np.abs(theta) / (z * sigma)
        v3 = np.abs(psi) > rhs3
    ok = np.isfinite(dist) & np.isfinite(psi) & (sigma > 0)
    agree = bool(np.all((v1 == v2)[ok]) and np.all((v1 == v3)[ok]))
    return dict(dtheta=dtheta, bif_value=theta + dtheta, dist_sigma=dist,
                psi=psi, t_stat=theta / sigma, rhs_form3=rhs3,
                unstable_form1=v1, unstable_form2=v2, unstable_form3=v3,
                forms_agree=agree, z=z)


def verify_gandolfo_paper(z=Z_95):
    """Reproduce the worked example of Gandolfo (1992) p. 45 / Gandolfo &
    Padoan (1990) Appendix 3, as an external check on `gandolfo`.

    The paper's numbers, verbatim: the 95% confidence interval for `alpha_8`
    (the adjustment speed of exports in the Italian continuous time model) is
    (0.5907, 1.1473); the bifurcation value of `alpha_8` with respect to the
    first real characteristic root `mu_1` is 1.1400, which "falls within the
    confidence interval"; the elasticity of `mu_1` with respect to `alpha_8`
    is 3.22, "which is greater than 3.12 (the value of the righthand side of
    (iv))".

    Everything else is recovered from those four numbers: with a symmetric
    normal interval, `theta = (lo + hi)/2` and `sigma = (hi - lo)/(2 z)`; the
    bifurcation value fixes `dtheta = 1.1400 - theta` and hence
    `dmu/dtheta = -mu/dtheta`.  `mu` itself never enters any of the three
    verdicts (it cancels), so it is set to -1 and the result is invariant to
    that choice — checked here rather than asserted.

    The paper reports its own quantities to three or four significant figures,
    so agreement is expected at the 1e-3 level and not better; the returned
    dict carries the discrepancies so the findings note can quote them.
    """
    ci_lo, ci_hi, bif = 0.5907, 1.1473, 1.1400
    theta = 0.5 * (ci_lo + ci_hi)
    sigma = (ci_hi - ci_lo) / (2.0 * z)
    dtheta = bif - theta
    out = {}
    for mu in (-1.0, -0.37, -12.5):                 # invariance to mu
        dmu = -mu / dtheta
        g = gandolfo(np.array([mu]), np.array([[dmu]]),
                     np.array([[theta]]), np.array([[sigma]]), z=z)
        out[mu] = dict(bif_value=float(g["bif_value"][0, 0]),
                       dist_sigma=float(g["dist_sigma"][0, 0]),
                       psi=float(abs(g["psi"][0, 0])),
                       rhs3=float(g["rhs_form3"][0, 0]),
                       unstable=bool(g["unstable_form1"][0, 0]),
                       forms_agree=bool(g["forms_agree"]))
    ref = out[-1.0]
    spread = max(abs(out[m]["dist_sigma"] - ref["dist_sigma"]) for m in out)
    return dict(
        theta=theta, sigma=sigma, dtheta=dtheta,
        bif_value=ref["bif_value"], paper_bif_value=bif,
        abs_err_bif=abs(ref["bif_value"] - bif),
        elasticity=ref["psi"], paper_elasticity=3.22,
        rel_err_elasticity=abs(ref["psi"] - 3.22) / 3.22,
        rhs_form3=ref["rhs3"], paper_rhs_form3=3.12,
        rel_err_rhs=abs(ref["rhs3"] - 3.12) / 3.12,
        verdict_unstable=ref["unstable"], paper_verdict_unstable=True,
        verdict_matches=bool(ref["unstable"] is True),
        forms_agree=ref["forms_agree"],
        invariance_to_mu=spread, z=z)


# Gandolfo (1992) p. 45: "our criterion is not meant to be applied
# mechanically: for example, if one considers 99% or higher confidence
# intervals, one will include a greater number of partial derivatives."  The
# criterion is monotone in z, so reporting the verdict at several z is the
# author's own recommended practice rather than an embellishment.
Z_LEVELS = {"90%": 1.6448536269514722,
            "95%": 1.959963984540054,
            "99%": 2.5758293035489004,
            "99.9%": 3.2905267314919255}


def z_required(dtheta, sigma):
    """The confidence level, in units of `z`, at which a parameter would first
    be declared structurally unstable: `z* = |dtheta| / sigma`.

    Reporting `z*` collapses the whole z-sensitivity into one number per
    parameter and makes Gandolfo's caveat quantitative — a parameter with
    `z* = 2.1` is a different object from one with `z* = 1000`.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(np.asarray(dtheta, float)) / np.asarray(sigma, float)


def root_sigma(dmu, cov):
    """Delta-method standard error of a characteristic root.

    `dmu`  (..., n_p)          `dmu_j/dp_n`
    `cov`  (..., n_p, n_p)     covariance of the parameter vector

    Gandolfo & Padoan (1990) report "the estimates of the characteristic roots
    together with their asymptotic standard errors" and conclude stability
    because the one positive root "is not significantly different from zero at
    the 5% level".  That is a DIFFERENT test from the bifurcation-distance
    criterion: it asks whether the root itself is resolved, not how far the
    parameters would have to move.  It needs the full covariance, not the
    marginals — the manufacturing shares are exactly anti-correlated by their
    simplex, and treating them as independent inflates the root's variance.

    # Ch2 Eq. eigenvalue_sensitivity supplies dmu; the covariance comes from
    # WP-4c's Laplace approximation.
    """
    dmu = np.asarray(dmu, float)
    cov = np.asarray(cov, float)
    var = np.einsum("...a,...ab,...b->...", dmu, cov, dmu)
    return np.sqrt(np.maximum(var, 0.0))


def feasible(values, lo=None, hi=None):
    """Elementwise membership of the feasible parameter set, last axis
    aligned to `PARAM_NAMES`.  NaN and +-inf are infeasible by definition:
    an unreachable bifurcation value is exactly what Part 2 is looking for."""
    v = np.asarray(values, float)
    lo = FEASIBLE_LO if lo is None else np.asarray(lo, float)
    hi = FEASIBLE_HI if hi is None else np.asarray(hi, float)
    with np.errstate(invalid="ignore"):
        return np.isfinite(v) & (v >= lo) & (v <= hi)


def first_crossing(path, target, t):
    """First `t` at which `path(t)` reaches `target`, in whichever direction
    it starts from.  `path` (..., n_t), `target` (...,).  NaN if never — the
    caller reports that with its seed list rather than extending the horizon.
    """
    path = np.asarray(path, float)
    target = np.asarray(target, float)[..., None]
    t = np.asarray(t, float)
    up = path[..., :1] < target
    hit = np.where(up, path >= target, path <= target)
    any_hit = hit.any(-1) & np.isfinite(target[..., 0])
    return np.where(any_hit, t[np.argmax(hit, -1)], np.nan)


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------
def check(verbose=True):
    """`zinc_A_lab.check()` (drivers, input_dim, digests, state layout) plus
    the WP-2c/d/e parameter layout and mapping decisions."""
    info = Alab.check(verbose=verbose)
    info["param_names"] = list(PARAM_NAMES)
    info["n_param"] = N_PARAM
    info["flow_functionals"] = {k: FLOW_CH2[k] for k in FLOW_NAMES}
    info["extrapolation_rules"] = list(RULES)
    if verbose:
        print("zinc_circ_lab --check  (WP-2c/2d/2e additions)")
        print("=" * 74)
        print(f"  parameters p_n       : {N_PARAM}")
        for grp, sl in (("alpha", slice(0, 4)), ("tau", slice(4, 8)),
                        ("frac_fu", slice(8, 11)), ("frac_eu", slice(11, 14)),
                        ("f_cohort", slice(14, 17)), ("mu (fixed)",
                                                      slice(17, 20))):
            print(f"    {grp:<12s} {PARAM_NAMES[sl]}")
        print(f"  simplex directions   : {len(simplex_directions())} "
              f"constrained pairs (Ch2: dl/dw_r - dl/dw_q)")
        print(f"  A_FD(t)              : = A(t) — no landfill state in the "
              f"UDE, so the 6-state IS the transient space")
        print(f"  flow functionals c_m : "
              + ", ".join(f"{k} ({FLOW_CH2[k]})" for k in FLOW_NAMES))
        print(f"  extrapolation rules  : {RULES}")
        print(f"  coefficient continuum: linear between year nodes, in "
              f"coefficient space (exact at nodes, keeps A Metzler)")
        print("=" * 74)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description="WP-2c/d/e shared lab module")
    ap.add_argument("--check", action="store_true")
    ap.parse_args(argv)
    check(verbose=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
