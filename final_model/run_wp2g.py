#!/usr/bin/env python3
"""
run_wp2g.py — WP-2g: frozen equilibrium, saturation, transient amplification
============================================================================

WP-2a established `use_stock_input=False` and measured coefficient
state-dependence at exactly 0.0, so                    # Ch2 Eq. system

    dS/dt = A(t) S + b(t),        b(t) = [cp(t), 0, 0, 0, 0, 0]   # Ch2 Eq. vecb

holds IDENTICALLY and the fitted system is linear in the state.  The
frozen-time equilibrium is therefore closed-form,

    S*(t) = -A(t)^-1 b(t) = N(t) b(t),

and WP-2d verified `-A` is a non-singular M-matrix at all 1 400 nodes, so
`S*` exists, is unique and is non-negative.  **Nothing here is a
linearisation.**

  Part 1  the saturation ratio `S(t)/S*(t)`, per stock, 1980-2019.  The
          In-Use aggregate is the headline — it is the industrial-ecology
          question.
  Part 2  does it saturate at all?  The dominant modal timescale (WP-2b:
          47.7 -> 50.6 yr) is a RELAXATION rate, not a saturation time: it
          assumes the target stands still.  `|dlog S*/dt|` against
          `1/T_dom(t)` tests that — the levels-space analogue of WP-2b's
          adiabaticity index.
  Part 3  time to saturation, as a FROZEN COUNTERFACTUAL, NOT A FORECAST.
  Part 4  `dS*/dp = N (dA/dp) N b - N db/dp` — Chapter 2's comparative
          statics evaluated empirically, in kt per unit parameter.
                                        # Ch2 Eq. fundamental_matrix_derivative
  Part 5  transient amplification `sup_u ||Phi(t+u,t)||_2`: the part of
          resilience the spectral abscissa cannot see.
  Part 6  what is NOT available — see `wp2g_findings.md`.

    python run_wp2g.py --check
    python run_wp2g.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

import zinc_circ_lab as L

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "analysis")
A_NPZ = os.path.join(OUT_DIR, "wp2a_A_of_t.npz")

STATE = list(L.ODE_STATE_NAMES)
OBS4 = ["Concentrate", "Refined", "In-Use", "Scrap"]
IU = slice(2, 5)
COL = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00", "#56B4E9"]
GREY = "#555555"

# Part 3 horizon.  Long enough that the 99% crossing of the 44-yr in-use
# cohort (the slowest object in the system) lands well inside it; anything
# not reached is reported as NaN, never extrapolated.
U_GRID = np.arange(0.0, 1000.0 + 1e-9, 0.25)
FRACS = (0.90, 0.95, 0.99)
B_RULES = ["hold", "mean5", "trend20"]

# Part 5 grid.  `u_max = 60` covers the amplification peak (a few years) with
# three orders of magnitude of margin; `n_sub` is checked by halving.
U_MAX = 60.0
N_SUB = 16


def hl(v):
    return L.hodges_lehmann(np.asarray(v, float))


def load(path=A_NPZ):
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found — run run_wp2a.py first")
    d = np.load(path, allow_pickle=True)
    return d, hashlib.md5(open(path, "rb").read()).hexdigest()


def to_obs4(X):
    """Collapse the 6-state to the four observable stocks (in-use cohorts
    summed), which is the space the dataset reports in."""
    X = np.asarray(X, float)
    return np.stack([X[..., 0], X[..., 1], X[..., IU].sum(-1), X[..., 5]], -1)


# ===========================================================================
# Part 1 + 2
# ===========================================================================
def part12(A, b, S_obs, S_pred, years):
    Sstar, N = L.equilibrium(A, b)
    S4 = to_obs4(Sstar)                                   # (seed, year, 4)
    lam = np.linalg.eigvals(A).real                       # all real (WP-2b)
    lam_dom = lam.max(-1)
    T_dom = 1.0 / np.abs(lam_dom)
    # levels-space analogue of WP-2b's adiabaticity index eps_k = |dlam/dt|/lam^2
    dlog = np.gradient(np.log(np.maximum(S4, 1e-300)), years, axis=1)
    ratio = np.abs(dlog) * T_dom[..., None]

    rows = []
    for j, nm in enumerate(OBS4):
        for i, y in enumerate(years):
            star = S4[:, i, j]
            obs = S_obs[i, j]
            pred = S_pred[:, i, j]
            rr = ratio[:, i, j]
            h = hl(obs / star)
            rows.append(dict(
                stock=nm, year=float(y),
                Sstar_median=float(np.median(star)),
                Sstar_q1=float(np.percentile(star, 25)),
                Sstar_q3=float(np.percentile(star, 75)),
                S_obs=float(obs),
                S_pred_median=float(np.median(pred)),
                gap_obs_kt_median=float(np.median(obs - star)),
                gap_pred_kt_median=float(np.median(pred - star)),
                sat_ratio_obs_median=float(np.median(obs / star)),
                sat_ratio_obs_hl=h["hl"], sat_ratio_obs_lo=h["lo"],
                sat_ratio_obs_hi=h["hi"],
                sat_ratio_pred_median=float(np.median(pred / star)),
                dlogSstar_dt_median=float(np.median(dlog[:, i, j])),
                inv_T_dom_median=float(np.median(1.0 / T_dom[:, i])),
                target_vs_relax_ratio_median=float(np.median(rr)),
                target_vs_relax_ratio_q1=float(np.percentile(rr, 25)),
                target_vs_relax_ratio_q3=float(np.percentile(rr, 75))))
    return pd.DataFrame(rows), Sstar, S4, N, T_dom, ratio


# ===========================================================================
# Part 3
# ===========================================================================
def extrapolate_cp(cp, years, rule):
    """The constant `cp` each of WP-2d's rules settles at beyond 2019.

    `coeff_path` extrapolates the COEFFICIENT vector; `b` is not part of it,
    so the same three conventions are applied here directly and named the
    same way.  `trend20` is the level the 10-node trend reaches at the end of
    its 20-year horizon, i.e. the constant the rule is frozen at — which is
    what a frozen counterfactual needs.
    """
    cp = np.asarray(cp, float)                            # (seed, year)
    if rule == "hold":
        return cp[:, -1]
    if rule.startswith("mean"):
        return cp[:, -int(rule[4:]):].mean(1)
    if rule.startswith("trend"):
        H = float(rule[5:])
        W = 10
        x = years[-W:] - years[-1]
        X = np.stack([np.ones_like(x), x], 1)
        slope = np.linalg.lstsq(X, cp[:, -W:].T, rcond=None)[0][1]
        return np.maximum(cp[:, -1] + slope * H, 0.0)
    raise ValueError(rule)


def part3(A, b, cp, S_obs, S_ode_pred, years, seeds):
    """Frozen counterfactual from the observed 2019 state.

    The observed 2019 in-use total is lifted into the three cohorts with the
    model's own 2019 cohort shares — the only split available, and the same
    device WP-2a used (there with a RANDOM split, precisely to show the
    algebra does not depend on it).
    """
    A19 = A[:, -1]
    w = S_ode_pred[:, -1, IU]
    w = w / w.sum(-1, keepdims=True)
    n = A.shape[0]
    S0 = np.stack([np.full(n, S_obs[-1, 0]), np.full(n, S_obs[-1, 1]),
                   w[:, 0] * S_obs[-1, 2], w[:, 1] * S_obs[-1, 2],
                   w[:, 2] * S_obs[-1, 2], np.full(n, S_obs[-1, 3])], -1)

    rows, modal_rows = [], []
    store = {}
    for rule in B_RULES:
        cp_end = extrapolate_cp(cp, years, rule)
        b19 = np.zeros_like(b[:, -1])
        b19[:, 0] = cp_end
        S, Sstar, contrib, lam, V = L.frozen_rollout(A19, b19, S0, U_GRID)
        tt = L.time_to_fraction(S, Sstar, U_GRID, FRACS)
        # the observable-4 view, with the in-use aggregate as the headline
        S4 = to_obs4(S)
        S4star = to_obs4(Sstar)
        tt4 = L.time_to_fraction(S4, S4star, U_GRID, FRACS)
        store[rule] = dict(S=S, Sstar=Sstar, tt=tt, S4=S4, S4star=S4star,
                           tt4=tt4, cp_end=cp_end, lam=lam, contrib=contrib)
        for j, nm in enumerate(STATE):
            for i, f in enumerate(FRACS):
                v = tt[:, i, j]
                rows.append(dict(space="ode6", stock=nm, rule=rule,
                                 fraction=f, n_reached=int(np.isfinite(v).sum()),
                                 median_yr=float(np.nanmedian(v)),
                                 q1_yr=float(np.nanpercentile(v, 25)),
                                 q3_yr=float(np.nanpercentile(v, 75)),
                                 Sstar_median=float(np.median(Sstar[:, j])),
                                 S0_median=float(np.median(S0[:, j]))))
        for j, nm in enumerate(OBS4):
            for i, f in enumerate(FRACS):
                v = tt4[:, i, j]
                rows.append(dict(space="obs4", stock=nm, rule=rule,
                                 fraction=f, n_reached=int(np.isfinite(v).sum()),
                                 median_yr=float(np.nanmedian(v)),
                                 q1_yr=float(np.nanpercentile(v, 25)),
                                 q3_yr=float(np.nanpercentile(v, 75)),
                                 Sstar_median=float(np.median(S4star[:, j])),
                                 S0_median=float(np.median(to_obs4(S0)[:, j]))))
        # Modal decomposition of the approach, in-use aggregate.
        # `np.linalg.eig` returns modes in no particular order, so each is
        # labelled by the compartment its right eigenvector loads on — WP-2b
        # established that assignment is unambiguous here (min
        # self-participation 0.892, Hungarian and greedy agree at 100% of
        # nodes).  Labelled per seed, then aggregated by label.
        if rule == "hold":
            gap = contrib[..., IU].sum(-1)               # (seed, n_u, mode)
            tot = np.abs(gap).sum(-1)
            lab = np.argmax(np.abs(V), axis=-2)          # (seed, mode) -> state
            for uu in (0.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0):
                i = int(np.argmin(np.abs(U_GRID - uu)))
                share = np.abs(gap[:, i]) / np.maximum(tot[:, i, None], 1e-300)
                for st in range(len(STATE)):
                    sel = lab == st                      # (seed, mode)
                    if not sel.any():
                        continue
                    sh = np.where(sel, share, 0.0).sum(-1)
                    lm = np.where(sel, lam, np.nan)
                    modal_rows.append(dict(
                        u_yr=float(uu), mode=STATE[st],
                        lam_median=float(np.nanmedian(lm)),
                        timescale_yr=float(np.nanmedian(1.0 / np.abs(lm))),
                        share_of_inuse_gap_median=float(np.median(sh)),
                        share_q1=float(np.percentile(sh, 25)),
                        share_q3=float(np.percentile(sh, 75))))
    return pd.DataFrame(rows), pd.DataFrame(modal_rows), store, S0


# ===========================================================================
# Part 4
# ===========================================================================
def part4(A, b, P, years):
    """`dS*/dp` in kt per unit parameter, and as elasticities."""
    dA = L.dA_dparams(P)
    dS, N = L.equilibrium_sensitivity(A, dA, b)           # (seed, yr, n_p, 6)
    Sstar = np.einsum("...ij,...j->...i", N, b)
    dS4 = to_obs4(dS)                                    # cohorts summed
    S4 = to_obs4(Sstar)
    with np.errstate(divide="ignore", invalid="ignore"):
        el = dS4 * P[..., :, None] / np.maximum(S4[..., None, :], 1e-300)

    rows = []
    for n_, pname in enumerate(L.PARAM_NAMES):
        for j, nm in enumerate(OBS4):
            for i in (0, len(years) - 1):
                v = dS4[:, i, n_, j]
                e = el[:, i, n_, j]
                h, he = hl(v), hl(e)
                rows.append(dict(
                    param=pname, ch2_label=L.CH2_LABEL.get(pname, ""),
                    stock=nm, year=float(years[i]),
                    dSstar_kt_per_unit_median=float(np.median(v)),
                    dSstar_kt_per_unit_hl=h["hl"], lo=h["lo"], hi=h["hi"],
                    elasticity_median=float(np.median(e)),
                    elasticity_hl=he["hl"], elasticity_lo=he["lo"],
                    elasticity_hi=he["hi"],
                    Sstar_median=float(np.median(S4[:, i, j]))))
    # Ch2's constrained simplex directions, in kt
    simp_rows = []
    for name, ia, ib, label in L.simplex_directions():
        for j, nm in enumerate(OBS4):
            for i in (0, len(years) - 1):
                v = (dS4[:, i, ia, j] - dS4[:, i, ib, j]) * 0.01   # per +1 pp
                h = hl(v)
                simp_rows.append(dict(simplex=name, direction=label, stock=nm,
                                      year=float(years[i]),
                                      d_kt_per_pp_hl=h["hl"], lo=h["lo"],
                                      hi=h["hi"]))
    return pd.DataFrame(rows), pd.DataFrame(simp_rows), dS4, el, dA


# ===========================================================================
# Part 5
# ===========================================================================
# Frozen-amplification `u` grid: fine where the peak is (a few years),
# coarse in the tail where the norm is monotonically decaying.
U_FROZEN = np.concatenate([np.arange(0.0, 20.0, 0.02),
                           np.arange(20.0, 120.0 + 1e-9, 0.5)])


def part5(A, P, years, seeds, keep=(0, 39)):
    amp = L.amplification_scan(P, years, np.arange(len(years)),
                               u_max=U_MAX, n_sub=N_SUB, rule="hold",
                               keep_full=keep)
    omega = L.numerical_abscissa(A)
    sabs = L.spectral_abscissa(A)
    kre = L.kreiss_constant(A)
    # The Kreiss theorem bounds the AUTONOMOUS semigroup, so it is checked
    # against the frozen amplification; the non-autonomous `sup ||Phi||` is a
    # different object and is reported alongside rather than bounded by it.
    fsup, fu, _ = L.frozen_amplification(A, U_FROZEN)
    rows = []
    for i, y in enumerate(years):
        s = amp["sup"][:, i]
        h = hl(s)
        rows.append(dict(
            year=float(y),
            sup_norm2_median=float(np.median(s)),
            sup_norm2_q1=float(np.percentile(s, 25)),
            sup_norm2_q3=float(np.percentile(s, 75)),
            sup_norm2_hl=h["hl"], lo=h["lo"], hi=h["hi"],
            u_at_median_yr=float(np.median(amp["u_at"][:, i])),
            u_at_q1_yr=float(np.percentile(amp["u_at"][:, i], 25)),
            u_at_q3_yr=float(np.percentile(amp["u_at"][:, i], 75)),
            sup_norm1_median=float(np.median(amp["sup1"][:, i])),
            frozen_sup_norm2_median=float(np.median(fsup[:, i])),
            frozen_sup_norm2_q1=float(np.percentile(fsup[:, i], 25)),
            frozen_sup_norm2_q3=float(np.percentile(fsup[:, i], 75)),
            frozen_u_at_median_yr=float(np.median(fu[:, i])),
            numerical_abscissa_median=float(np.median(omega[:, i])),
            spectral_abscissa_median=float(np.median(sabs[:, i])),
            kreiss_median=float(np.median(kre[:, i])),
            kreiss_q1=float(np.percentile(kre[:, i], 25)),
            kreiss_q3=float(np.percentile(kre[:, i], 75)),
            n_seeds_amplifying=int(np.sum(s > 1.0 + 1e-9)),
            max_col_sup_norm2_median=float(
                np.median(amp["col_sup"][:, i].max(-1))),
            top_input_stock=STATE[int(np.bincount(
                amp["from_idx"][:, i], minlength=6).argmax())],
            top_output_stock=STATE[int(np.bincount(
                amp["to_idx"][:, i], minlength=6).argmax())],
            mean_abs_sing_in=json.dumps(
                [round(float(x), 4) for x in
                 np.median(np.abs(amp["sing_in"][:, i]), axis=0)]),
            mean_abs_sing_out=json.dumps(
                [round(float(x), 4) for x in
                 np.median(np.abs(amp["sing_out"][:, i]), axis=0)])))
    return pd.DataFrame(rows), amp, omega, sabs, kre, fsup, fu


# ===========================================================================
# figures
# ===========================================================================
def fig_saturation(sat, t3, store, years, mask_test, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = float(years[np.asarray(mask_test, bool)][0])
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))

    ax = axes[0, 0]
    for j, nm in enumerate(OBS4):
        s = sat[sat.stock == nm]
        ax.plot(s.year, s.sat_ratio_obs_median, color=COL[j], lw=1.9, label=nm)
        ax.fill_between(s.year, s.sat_ratio_obs_lo, s.sat_ratio_obs_hi,
                        color=COL[j], alpha=0.15, lw=0)
    ax.axhline(1.0, color="k", lw=0.9, ls="--")
    ax.axvline(t0, color=GREY, lw=0.9, ls=":")
    ax.set_ylabel("$S(t)\\,/\\,S^*(t)$")
    ax.set_xlabel("year")
    ax.set_title("(a) saturation ratio — observed stock against the "
                 "frozen-time equilibrium\n     $S^*(t)=-A(t)^{-1}b(t)$; "
                 "band = 95% HL CI across seeds", loc="left", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, ncol=2)

    ax = axes[0, 1]
    s = sat[sat.stock == "In-Use"]
    ax.fill_between(s.year, s.Sstar_q1 / 1e3, s.Sstar_q3 / 1e3,
                    color=COL[2], alpha=0.2, lw=0)
    ax.plot(s.year, s.Sstar_median / 1e3, color=COL[2], lw=2.0,
            label="$S^*_{\\rm in\\text{-}use}(t)$  (frozen equilibrium)")
    ax.plot(s.year, s.S_obs / 1e3, color="k", lw=1.8, label="observed")
    ax.plot(s.year, s.S_pred_median / 1e3, color=GREY, lw=1.3, ls="--",
            label="model free-run")
    ax.axvline(t0, color=GREY, lw=0.9, ls=":")
    ax.set_ylabel("in-use zinc stock  (Mt)")
    ax.set_xlabel("year")
    ax.set_title("(b) the in-use stock and the target it is chasing",
                 loc="left", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False)

    ax = axes[1, 0]
    for j, nm in enumerate(OBS4):
        s = sat[sat.stock == nm]
        ax.plot(s.year, s.target_vs_relax_ratio_median, color=COL[j], lw=1.8,
                label=nm)
        ax.fill_between(s.year, s.target_vs_relax_ratio_q1,
                        s.target_vs_relax_ratio_q3, color=COL[j],
                        alpha=0.13, lw=0)
    ax.axhline(1.0, color="k", lw=1.1, ls="--")
    ax.axvline(t0, color=GREY, lw=0.9, ls=":")
    ax.set_yscale("log")
    ax.set_ylabel("$|d\\log S^*/dt|\\cdot T_{dom}$")
    ax.set_xlabel("year")
    ax.set_title("(c) does it saturate at all?  above 1 = the equilibrium "
                 "moves\n     faster than the system relaxes, so the gap "
                 "never closes", loc="left", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, ncol=2)

    ax = axes[1, 1]
    u = U_GRID
    for r, rule in enumerate(B_RULES):
        iu = store[rule]["S4"][:, :, 2] / 1e3
        lo, med, hi = np.percentile(iu, [25, 50, 75], axis=0)
        m = u <= 250
        ax.fill_between(u[m], lo[m], hi[m], color=COL[r], alpha=0.15, lw=0)
        ax.plot(u[m], med[m], color=COL[r], lw=1.8,
                label=f"$b$ rule: {rule}")
        ax.axhline(np.median(store[rule]["S4star"][:, 2]) / 1e3,
                   color=COL[r], lw=0.9, ls=":")
    ax.set_xlabel("years after 2019")
    ax.set_ylabel("in-use zinc stock  (Mt)")
    ax.set_title("(d) FROZEN COUNTERFACTUAL, NOT A FORECAST — coefficients "
                 "and $b$\n     frozen at 2019, from the observed 2019 state",
                 loc="left", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False)

    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(lw=0.4, alpha=0.3)
    fig.suptitle("WP-2g  frozen equilibrium and saturation of the estimated "
                 "zinc cycle  (Ch2 Eq. system, Eq. vecb)\n"
                 "anchor_v4, 35 seeds; dotted vertical = start of held-out "
                 "window", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=300)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


def fig_amplification(amp_df, amp, years, mask_test, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = float(years[np.asarray(mask_test, bool)][0])
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))

    ax = axes[0]
    for n, ti in enumerate(sorted(amp["full"])):
        i = int(ti)
        nn = amp["norms"][:, i]
        lo, med, hi = np.percentile(nn, [25, 50, 75], axis=0)
        m = amp["u"] <= 20
        ax.fill_between(amp["u"][m], lo[m], hi[m], color=COL[n], alpha=0.18, lw=0)
        ax.plot(amp["u"][m], med[m], color=COL[n], lw=1.9,
                label=f"$t_0$ = {int(years[i])}")
        n1 = np.median(amp["norms1"][:, i], axis=0)
        ax.plot(amp["u"][m], n1[m], color=COL[n], lw=1.0, ls="--")
    ax.axhline(1.0, color="k", lw=0.9, ls=":")
    ax.set_xlabel("$u$  (years)")
    ax.set_ylabel("$\\|\\Phi(t_0+u,t_0)\\|$")
    ax.set_title("(a) solid $\\|\\cdot\\|_2$, dashed $\\|\\cdot\\|_1$\n"
                 "     mass balance caps the 1-norm at 1",
                 loc="left", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False)

    ax = axes[1]
    ax.fill_between(amp_df.year, amp_df.sup_norm2_q1, amp_df.sup_norm2_q3,
                    color=COL[0], alpha=0.18, lw=0)
    ax.plot(amp_df.year, amp_df.sup_norm2_median, color=COL[0], lw=1.9,
            label="$\\sup_u\\|\\Phi\\|_2$")
    ax.fill_between(amp_df.year, amp_df.kreiss_q1, amp_df.kreiss_q3,
                    color=COL[2], alpha=0.18, lw=0)
    ax.plot(amp_df.year, amp_df.frozen_sup_norm2_median, color=COL[1], lw=1.5,
            ls="-.", label="$\\sup_u\\|e^{Au}\\|_2$  (frozen)")
    ax.plot(amp_df.year, amp_df.kreiss_median, color=COL[2], lw=1.6, ls="--",
            label="Kreiss constant (lower bound)")
    ax.plot(amp_df.year, amp_df.sup_norm1_median, color=GREY, lw=1.2, ls=":",
            label="$\\sup_u\\|\\Phi\\|_1$")
    ax.axhline(1.0, color="k", lw=0.9)
    ax.axvline(t0, color=GREY, lw=0.9, ls=":")
    ax.set_xlabel("start year $t_0$")
    ax.set_title("(b) maximum transient amplification", loc="left", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False)

    ax = axes[2]
    ax.plot(amp_df.year, amp_df.numerical_abscissa_median, color=COL[3], lw=1.9,
            label="$\\omega(A)=\\lambda_{max}((A+A^T)/2)$")
    ax.plot(amp_df.year, amp_df.spectral_abscissa_median, color=COL[1], lw=1.9,
            label="$s(A)=\\max\\,\\mathrm{Re}\\,\\lambda$")
    ax.axhline(0.0, color="k", lw=0.9)
    ax.axvline(t0, color=GREY, lw=0.9, ls=":")
    ax.set_yscale("symlog", linthresh=1e-2)
    ax.set_xlabel("year")
    ax.set_ylabel("/yr")
    ax.set_title("(c) non-normality: initial growth rate against\n"
                 "     the asymptotic one", loc="left", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(lw=0.4, alpha=0.3)
    fig.suptitle("WP-2g Part 5  transient amplification — the part of "
                 "resilience the spectral abscissa cannot see  "
                 "(Ch2 Eq. product_transition_matrices)\n"
                 "anchor_v4, 35 seeds (median, IQR); every eigenvalue is real "
                 "and negative at every node (WP-2b)",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(path, dpi=300)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


def fig_sensitivity(sens, years, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    y_end = float(years[-1])
    for n, (nm, lab) in enumerate([("In-Use", "in-use stock"),
                                   ("Scrap", "scrap stock")]):
        ax = axes[n]
        s = sens[(sens.stock == nm) & (sens.year == y_end)].copy()
        s = s.reindex(s.dSstar_kt_per_unit_hl.abs()
                      .sort_values(ascending=False).index)[:10][::-1]
        y = np.arange(len(s))
        ax.barh(y, s.dSstar_kt_per_unit_hl / 1e3, color=COL[0], height=0.6)
        for i, (_, r) in enumerate(s.iterrows()):
            ax.plot([r.lo / 1e3, r.hi / 1e3], [i, i], color="k", lw=1.1)
        ax.set_yticks(y)
        ax.set_yticklabels(s.param, fontsize=7.5)
        ax.axvline(0, color="k", lw=0.9)
        ax.set_xlabel("$\\partial S^*/\\partial p$   (Mt per unit parameter)")
        ax.set_title(f"({'ab'[n]}) equilibrium {lab}, {int(y_end)}",
                     loc="left", fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(lw=0.4, alpha=0.3, axis="x")
    fig.suptitle("WP-2g Part 4  comparative statics of the frozen "
                 "equilibrium in physical units  "
                 "($\\partial S^*/\\partial p = N(\\partial A/\\partial p)Nb$, "
                 "Ch2 Eq. fundamental_matrix_derivative)\n"
                 "anchor_v4, 35 seeds; bars = Hodges–Lehmann with 95% CI",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path, dpi=300)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


# ===========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args(argv)

    info = L.check(verbose=True)
    if args.check:
        return 0

    d, md5 = load()
    A = np.asarray(d["A"], float)
    b = np.asarray(d["b"], float)
    cp = np.asarray(d["cp"], float)
    years = np.asarray(d["years"], float)
    seeds = np.asarray(d["seeds"], int)
    S_obs = np.asarray(d["S_obs"], float)
    S_pred = np.asarray(d["S_pred"], float)
    S_ode_pred = np.asarray(d["S_ode_pred"], float)
    mu = np.asarray(d["mu_cohorts"], float)
    P = L.pack_params(d, mu)
    os.makedirs(args.out, exist_ok=True)
    print(f"\nsource wp2a_A_of_t.npz md5 {md5[:8]}…   "
          f"{A.shape[0]} seeds x {A.shape[1]} years")

    validation = []

    # ---- Part 1 + 2 --------------------------------------------------------
    sat, Sstar, S4star, N, T_dom, ratio = part12(A, b, S_obs, S_pred, years)
    sat.to_csv(os.path.join(args.out, "wp2g_saturation.csv"), index=False)

    v = L.verify_equilibrium(A[:, [0, -1]].reshape(-1, 6, 6),
                             b[:, [0, -1]].reshape(-1, 6))
    validation.append(dict(
        part=1, check="S* = -A^-1 b against LSODA integration to steady state "
                      "(rtol 1e-11), 1980 and 2019, 35 seeds",
        result=v["max_rel"], exact_answer=0.0, units="relative"))
    validation.append(dict(
        part=1, check="S* >= 0 entrywise (-A an M-matrix, b >= 0)",
        result=float(Sstar.min()), exact_answer=0.0, units="kt, >= 0"))

    print("\n" + "=" * 74)
    print("Part 1 — the saturation ratio")
    print("=" * 74)
    print(f"  S* vs LSODA reference: max relative error {v['max_rel']:.2e}  "
          f"(exact 0)")
    print(f"  min S* over all nodes: {Sstar.min():.1f} kt  (must be >= 0)")
    print(f"\n  {'stock':<14s}{'S*(2019) kt':>16s}{'obs 2019':>12s}"
          f"{'S/S* 1980':>12s}{'S/S* 2019':>12s}")
    for nm in OBS4:
        s0 = sat[(sat.stock == nm) & (sat.year == years[0])].iloc[0]
        s1 = sat[(sat.stock == nm) & (sat.year == years[-1])].iloc[0]
        print(f"  {nm:<14s}{s1.Sstar_median:>16.1f}{s1.S_obs:>12.1f}"
              f"{s0.sat_ratio_obs_median:>12.3f}{s1.sat_ratio_obs_median:>12.3f}")
    iu19 = sat[(sat.stock == "In-Use") & (sat.year == years[-1])].iloc[0]
    print(f"\n  HEADLINE: S*_iu(2019) = {iu19.Sstar_median:,.0f} kt "
          f"[{iu19.Sstar_q1:,.0f}, {iu19.Sstar_q3:,.0f}] against an observed "
          f"in-use stock of {iu19.S_obs:,.0f} kt")
    print(f"            saturation ratio {iu19.sat_ratio_obs_hl:.3f} "
          f"[{iu19.sat_ratio_obs_lo:.3f}, {iu19.sat_ratio_obs_hi:.3f}]")

    print("\n" + "=" * 74)
    print("Part 2 — does it saturate at all?")
    print("=" * 74)
    print(f"  1/T_dom: {np.median(1/T_dom[:, 0]):.4f} /yr (1980) -> "
          f"{np.median(1/T_dom[:, -1]):.4f} /yr (2019)   "
          f"[T_dom {np.median(T_dom[:, 0]):.1f} -> {np.median(T_dom[:, -1]):.1f} yr]")
    print(f"  {'stock':<14s}{'ratio 1980':>13s}{'ratio 2019':>13s}"
          f"{'window mean':>13s}   > 1 means the target outruns the system")
    for j, nm in enumerate(OBS4):
        s = sat[sat.stock == nm]
        print(f"  {nm:<14s}{s.iloc[0].target_vs_relax_ratio_median:>13.2f}"
              f"{s.iloc[-1].target_vs_relax_ratio_median:>13.2f}"
              f"{np.median(ratio[:, :, j].mean(1)):>13.2f}")
    n_above = int(np.sum(ratio > 1.0))
    print(f"  nodes with ratio > 1: {n_above} of {ratio.size} "
          f"({100*n_above/ratio.size:.1f}%)")

    # ---- Part 3 ------------------------------------------------------------
    t3, modal, store, S0 = part3(A, b, cp, S_obs, S_ode_pred, years, seeds)
    t3.to_csv(os.path.join(args.out, "wp2g_time_to_saturation.csv"), index=False)
    modal.to_csv(os.path.join(args.out, "wp2g_modal_approach.csv"), index=False)
    print("\n" + "=" * 74)
    print("Part 3 — time to saturation  (FROZEN COUNTERFACTUAL, NOT A FORECAST)")
    print("=" * 74)
    for rule in B_RULES:
        print(f"  b rule `{rule}`: cp frozen at "
              f"{np.median(store[rule]['cp_end']):,.0f} kt/yr, "
              f"S*_iu = {np.median(store[rule]['S4star'][:, 2]):,.0f} kt")
    print(f"\n  {'stock':<14s}{'rule':<10s}" +
          "".join(f"{int(f*100)}%".rjust(11) for f in FRACS) + "   (years)")
    for nm in ["In-Use", "Scrap", "Refined", "Concentrate"]:
        for rule in B_RULES:
            s = t3[(t3.space == "obs4") & (t3.stock == nm) & (t3.rule == rule)]
            line = f"  {nm:<14s}{rule:<10s}"
            for f in FRACS:
                r = s[s.fraction == f].iloc[0]
                line += (f"{r.median_yr:>11.1f}" if r.n_reached
                         else f"{'n/r':>11s}")
            print(line)
    print("\n  per ODE state, rule `hold` (years to 95%):")
    for nm in STATE:
        r = t3[(t3.space == "ode6") & (t3.stock == nm) & (t3.rule == "hold")
               & (t3.fraction == 0.95)].iloc[0]
        print(f"    {nm:<14s} {r.median_yr:8.1f}  "
              f"[{r.q1_yr:.1f}, {r.q3_yr:.1f}]   "
              f"S* {r.Sstar_median:>12,.0f} kt   S0 {r.S0_median:>12,.0f} kt")

    # ---- Part 4 ------------------------------------------------------------
    sens, simp, dS4, el, dA = part4(A, b, P, years)
    sens.to_csv(os.path.join(args.out, "wp2g_equilibrium_sensitivity.csv"),
                index=False)
    simp.to_csv(os.path.join(args.out, "wp2g_simplex_directions.csv"),
                index=False)
    err = L.verify_equilibrium_derivatives(P[:, [0, -1]], b[:, [0, -1]])
    validation.append(dict(
        part=4, check="dS*/dp against central differences of S*(p +- h), "
                      "1980 and 2019, 35 seeds",
        result=err, exact_answer=0.0, units="relative"))

    # The old-scrap scaling identity.  The scrap COLUMN of A is homogeneous of
    # degree 1 in (alpha_win, alpha_dr) jointly — every entry carries exactly
    # one factor, including the diagonal `ns*alpha_dr - (alpha_win+alpha_dr)`.
    # Scaling that column by `lambda` divides S*_scrap by `lambda` and leaves
    # every other equilibrium untouched, so by Euler's theorem the two
    # elasticities are exactly equal and opposite everywhere except on Scrap,
    # where they sum to -1.  Checked both ways: on the closed-form
    # elasticities, and by directly rescaling the coefficients.
    i_win, i_dr = L.PARAM_NAMES.index("alpha_win"), L.PARAM_NAMES.index("alpha_dr")
    el_sum = el[..., i_win, :] + el[..., i_dr, :]        # (seed, year, 4)
    want = np.array([0.0, 0.0, 0.0, -1.0])
    P_up = P.copy()
    P_up[..., [i_win, i_dr]] *= 1.5
    S_up = to_obs4(L.equilibrium(L.assemble_from_params(P_up), b)[0])
    ratio = S_up / to_obs4(Sstar)
    want_ratio = np.array([1.0, 1.0, 1.0, 1.0 / 1.5])
    validation += [
        dict(part=4, check="old-scrap scaling identity: elasticities of "
                           "alpha_win + alpha_dr sum to (0,0,0,-1) over the "
                           "four stocks",
             result=float(np.max(np.abs(el_sum - want))), exact_answer=0.0,
             units="absolute"),
        dict(part=4, check="same, by direct rescaling: (alpha_win, alpha_dr) "
                           "x1.5 leaves S* unchanged except S*_scrap /1.5",
             result=float(np.max(np.abs(ratio - want_ratio))),
             exact_answer=0.0, units="relative")]
    print(f"  old-scrap scaling identity (elasticities)      : "
          f"{np.max(np.abs(el_sum - want)):.2e}  (exact 0)")
    print(f"  old-scrap scaling identity (direct x1.5 rescale): "
          f"{np.max(np.abs(ratio - want_ratio)):.2e}  (exact 0)")
    print( "  -> the equilibrium depends on the alpha_13/alpha_14 SPLIT, not "
           "on their level:")
    print( "     scaling both changes only how fast scrap turns over, not "
           "where the material goes.")
    print("\n" + "=" * 74)
    print("Part 4 — equilibrium sensitivity in physical units")
    print("=" * 74)
    print(f"  dS*/dp vs central differences: {err:.2e}  (exact 0)")
    s = sens[(sens.stock == "In-Use") & (sens.year == years[-1])].copy()
    s = s.reindex(s.dSstar_kt_per_unit_hl.abs()
                  .sort_values(ascending=False).index)[:8]
    print(f"\n  top levers on the equilibrium IN-USE stock, {int(years[-1])}:")
    for _, r in s.iterrows():
        print(f"    {r.param:<16s} {r.dSstar_kt_per_unit_hl:>14,.0f} kt/unit "
              f"[{r.lo:>12,.0f},{r.hi:>12,.0f}]   "
              f"elasticity {r.elasticity_hl:+.3f}")
    r = sens[(sens.param == "tau_olds") & (sens.stock == "In-Use")
             & (sens.year == years[-1])].iloc[0]
    print(f"\n  quotable: +1 percentage point of end-of-life collection "
          f"(`tau_olds`) is worth\n            "
          f"{r.dSstar_kt_per_unit_hl*0.01:,.0f} kt "
          f"[{r.lo*0.01:,.0f}, {r.hi*0.01:,.0f}] of equilibrium in-use zinc.")

    # ---- Part 5 ------------------------------------------------------------
    amp_df, amp, omega, sabs, kre, fsup, fu = part5(A, P, years, seeds)
    amp_df.to_csv(os.path.join(args.out, "wp2g_amplification.csv"), index=False)
    validation.append(dict(
        part=5, check="mass balance caps ||Phi||_1 at 1 (1^T A = -l^T, l >= 0)",
        result=float(amp["sup1"].max() - 1.0), exact_answer=0.0,
        units="excess over 1"))
    validation.append(dict(
        part=5, check="Kreiss lower bound K <= sup_u ||exp(Au)||_2 (frozen; "
                      "the Kreiss theorem bounds the autonomous semigroup)",
        result=float(np.max(kre - fsup)), exact_answer=0.0,
        units="violation, <= 0"))
    validation.append(dict(
        part=5, check="Kreiss upper bound sup_u ||exp(Au)||_2 <= e n K",
        result=float(np.max(fsup - np.e * 6 * kre)), exact_answer=0.0,
        units="violation, <= 0"))
    # step-halving check on the propagator
    amp_h = L.amplification_scan(P[:5], years, [0, len(years) - 1],
                                 u_max=U_MAX, n_sub=2 * N_SUB, rule="hold")
    conv = float(np.max(np.abs(amp_h["sup"] - amp["sup"][:5][:, [0, -1]])
                        / amp["sup"][:5][:, [0, -1]]))
    validation.append(dict(
        part=5, check=f"step halving {N_SUB} -> {2*N_SUB} substeps/yr on "
                      f"sup||Phi||_2", result=conv, exact_answer=0.0,
        units="relative"))
    print("\n" + "=" * 74)
    print("Part 5 — transient amplification")
    print("=" * 74)
    print(f"  ||Phi||_1 never exceeds 1 (mass balance): max excess "
          f"{amp['sup1'].max()-1.0:.2e}")
    print(f"  Kreiss bounds on the frozen semigroup: "
          f"max (K - sup||e^(Au)||_2) = {np.max(kre - fsup):+.4f}  and  "
          f"max (sup - e*n*K) = {np.max(fsup - np.e*6*kre):+.3f}  "
          f"(both must be <= 0)")
    print(f"  step halving {N_SUB} -> {2*N_SUB}/yr: {conv:.2e} relative")
    for i, y in [(0, years[0]), (len(years)//2, years[len(years)//2]),
                 (len(years)-1, years[-1])]:
        r = amp_df[amp_df.year == float(y)].iloc[0]
        print(f"  t0 = {int(y)}: sup||Phi||_2 = {r.sup_norm2_median:.3f} "
              f"[{r.sup_norm2_q1:.3f},{r.sup_norm2_q3:.3f}] at u = "
              f"{r.u_at_median_yr:.2f} yr;  K = {r.kreiss_median:.3f};  "
              f"omega(A) = {r.numerical_abscissa_median:+.3f} /yr vs "
              f"s(A) = {r.spectral_abscissa_median:+.4f} /yr;  "
              f"frozen sup = {r.frozen_sup_norm2_median:.3f}")
    print(f"  seeds amplifying (sup > 1) at every start year: "
          f"{int(amp_df.n_seeds_amplifying.min())}–"
          f"{int(amp_df.n_seeds_amplifying.max())} of {A.shape[0]}")
    print(f"  max over start years of sup_u ||Phi e_j||_2, any single stock j: "
          f"{np.max(amp['col_sup']):.6f}  (<= 1: no SINGLE compartment's "
          f"content is amplified)")
    print("  median |v_1| (amplified input direction), 2019: "
          + ", ".join(f"{STATE[k]} {x:.2f}" for k, x in
                      enumerate(np.median(np.abs(amp["sing_in"][:, -1]), 0))))
    print("  median |u_1| (output direction),          2019: "
          + ", ".join(f"{STATE[k]} {x:.2f}" for k, x in
                      enumerate(np.median(np.abs(amp["sing_out"][:, -1]), 0))))

    # ---- persist -----------------------------------------------------------
    pd.DataFrame(validation).to_csv(
        os.path.join(args.out, "wp2g_validation.csv"), index=False)
    eq_rows = []
    for s_i, sd in enumerate(seeds):
        for i, y in enumerate(years):
            row = dict(seed=int(sd), year=float(y))
            for k, nm in enumerate(STATE):
                row[f"Sstar_{nm}"] = float(Sstar[s_i, i, k])
            for k, nm in enumerate(OBS4):
                row[f"Sstar_{nm}_obs4"] = float(S4star[s_i, i, k])
                row[f"sat_ratio_{nm}"] = float(S_obs[i, k] / S4star[s_i, i, k])
            row["T_dom_yr"] = float(T_dom[s_i, i])
            row["numerical_abscissa"] = float(omega[s_i, i])
            row["spectral_abscissa"] = float(sabs[s_i, i])
            row["kreiss"] = float(kre[s_i, i])
            row["sup_norm2_Phi"] = float(amp["sup"][s_i, i])
            eq_rows.append(row)
    pd.DataFrame(eq_rows).to_csv(
        os.path.join(args.out, "wp2g_equilibrium.csv"), index=False)

    np.savez_compressed(
        os.path.join(args.out, "wp2g_equilibrium.npz"),
        years=years, seeds=seeds, source_md5=np.array(md5),
        state_names=np.array(STATE, object),
        obs4_names=np.array(OBS4, object),
        param_names=np.array(L.PARAM_NAMES, object),
        Sstar=Sstar, Sstar_obs4=S4star, N=N, S_obs=S_obs, S_pred=S_pred,
        T_dom=T_dom, target_vs_relax_ratio=ratio,
        dSstar_dparams_obs4=dS4, elasticity_obs4=el,
        numerical_abscissa=omega, spectral_abscissa=sabs, kreiss=kre,
        amp_sup=amp["sup"], amp_u_at=amp["u_at"], amp_norms=amp["norms"],
        amp_norms1=amp["norms1"], amp_col_sup=amp["col_sup"],
        amp_sing_in=amp["sing_in"], amp_sing_out=amp["sing_out"],
        frozen_amp_sup=fsup, frozen_amp_u_at=fu, u_frozen=U_FROZEN,
        u_grid=U_GRID, b_rules=np.array(B_RULES, object),
        frozen_S0=S0,
        **{f"frozen_S4_{r}": store[r]["S4"] for r in B_RULES},
        **{f"frozen_S4star_{r}": store[r]["S4star"] for r in B_RULES})

    fig_saturation(sat, t3, store, years, d["mask_test"],
                   os.path.join(args.out, "wp2g_saturation.png"))
    fig_amplification(amp_df, amp, years, d["mask_test"],
                      os.path.join(args.out, "wp2g_amplification.png"))
    fig_sensitivity(sens, years, os.path.join(args.out, "wp2g_sensitivity.png"))
    with open(os.path.join(args.out, "wp2g_check.json"), "w") as fh:
        json.dump({k: v for k, v in info.items() if k != "digests"}, fh,
                  indent=2, default=str)
    print(f"\nwrote wp2g_* to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
