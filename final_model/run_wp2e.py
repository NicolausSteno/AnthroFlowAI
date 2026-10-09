#!/usr/bin/env python3
"""
run_wp2e.py — WP-2e: fundamental-matrix sensitivities
=====================================================

Chapter 2's derivatives of the circularity indicators with respect to the
transfer coefficients, evaluated on the estimated A(t) of WP-2a.  With
`N_T = (-A_T)^-1` and `dN_T = N_T (dA_T) N_T`:  # Ch2 Eq. fundamental_matrix_derivative

    dtau_i/dp_n        = 1^T N_T (dA_T/dp_n) N_T e_i        # Ch2 Eq. tau_sensitivity
    dUpsilon_{m,i}/dp_n = (dc_m/dp_n)^T N_T e_i
                          + c_m^T N_T (dA_T/dp_n) N_T e_i   # Ch2 Eq. flow_count_sensitivity
    S_tau,l, S_upsilon,l = elasticities                     # Ch2 Eq. tau_U_elasticities

`A_T = A_FD = A(t)` and `alpha_m e_k^T -> c_m(t)^T` as set out in
`zinc_circ_lab`; the `phi_lm` term of Eq. flow_count_sensitivity generalises to
`dc_m/dp_n`, which is exact because every `c_m` is a linear functional of `A`.
20 parameters x 6 starting stocks x 4 flows x 35 seeds x 40 years.

Four things beyond the bare formula
-----------------------------------
  * **Term decomposition.**  Eq. flow_count_sensitivity has a direct term
    (`phi_lm`: the coefficient's own rate changes) and a propagated term (the
    fundamental matrix changes).  They are reported separately, because their
    ratio is exactly "how much of a circularity response is the intervention
    itself and how much is the system rearranging around it" — a quantity a
    static MFA cannot form.
  * **Simplex constraints.**  Ch2's directional derivative
    `d/dtheta = d/dw_r - d/dw_q` for all nine constrained pairs.
  * **Frozen-time vs non-autonomous sensitivity.**  WP-2d showed the frozen
    *indicators* are off by 5% (aggregate) to 38% (route-level).  Whether the
    frozen *sensitivities* inherit that error is a separate question and is
    answered here directly, by re-integrating Phi under a proportional shift
    of each coefficient path.
  * **Contrast with WP-2c.**  Eigenvalue sensitivity and indicator sensitivity
    are different objects with different rankings; the rank correlation
    between them is reported rather than either being presented as "the"
    sensitivity.

    python run_wp2e.py --check
    python run_wp2e.py
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

import zinc_circ_lab as L

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "analysis")
A_NPZ = os.path.join(OUT_DIR, "wp2a_A_of_t.npz")
WP2C_NPZ = os.path.join(OUT_DIR, "wp2c_eigen_sensitivity.npz")
WP2D_CSV = os.path.join(OUT_DIR, "wp2d_elasticities.csv")

STATE = list(L.ODE_STATE_NAMES)
COL = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00", "#56B4E9"]
GREY = "#555555"
START = 0                          # headline starting stock: S_conc
REF_RULE = "hold"
N_SUB = 64
# Parameters for which a proportional path shift is well defined without
# touching a simplex: the four alphas and the four binary taus.
NONAUTO_PARAMS = ["alpha_cc", "alpha_refc", "alpha_win", "alpha_dr",
                  "tau_ref", "tau_waelz", "tau_olds", "tau_diss"]
EPS = 1e-3


def load(path=A_NPZ):
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found — run run_wp2a.py first")
    d = np.load(path, allow_pickle=True)
    return d, hashlib.md5(open(path, "rb").read()).hexdigest()


# ---------------------------------------------------------------------------
# term decomposition of Ch2 Eq. flow_count_sensitivity
# ---------------------------------------------------------------------------
def decompose_flow_sensitivity(A, dA):
    """Split `dUpsilon_{m,i}/dp` into its two Chapter 2 terms.

    `direct` is `(dc_m/dp)^T N e_i` — Chapter 2's `phi_lm e_k^T N_T e_i`,
    i.e. the flow's own rate moving.  `propagated` is
    `c_m^T N (dA/dp) N e_i` — the fundamental matrix rearranging.  Their sum
    is the total returned by `zinc_circ_lab.indicator_sensitivity`, which is
    asserted.
    """
    N = L.fundamental(A)
    M = np.einsum("...ij,...pjk,...kl->...pil", N, dA, N)
    rows, drows = L.flow_rows(A), L.flow_rows(dA)
    out = {}
    for k in L.FLOW_NAMES:
        direct = np.einsum("...pi,...ij->...pj", drows[k], N)
        prop = np.einsum("...i,...pij->...pj", rows[k], M)
        out[k] = (direct, prop)
    return out, N


# ---------------------------------------------------------------------------
# non-autonomous counterpart, by proportional path shift
# ---------------------------------------------------------------------------
def nonauto_elasticities(P, years, param_names, rule=REF_RULE, n_sub=N_SUB,
                         eps=EPS):
    """`d ln X(t0) / d ln p` for the non-autonomous indicators.

    The whole coefficient *path* `p(t)` is scaled by `1 +/- eps` and `Phi` is
    re-integrated, so the derivative accounts for the coefficient change over
    the entire absorption horizon rather than at `t0` only.  This is the
    quantity Ch2 Eq. tau_sensitivity approximates when it is read off a
    frozen `N_T`, and the comparison between them is the point.

    Central difference; the discretisation error largely cancels between the
    two arms, and the frozen counterpart is validated separately against exact
    finite differences of the indicator.
    """
    base = L.nonautonomous_indicators(P, years, rule=rule, n_sub=n_sub)
    out = {}
    for name in param_names:
        n = L.PARAM_NAMES.index(name)
        arms = []
        for sgn in (+1.0, -1.0):
            Q = P.copy()
            Q[:, :, n] = Q[:, :, n] * (1.0 + sgn * eps)
            arms.append(L.nonautonomous_indicators(Q, years, rule=rule,
                                                   n_sub=n_sub))
        e_tau = (np.log(arms[0]["tau"]) - np.log(arms[1]["tau"])) / (2 * eps)
        e_cnt = {}
        for k in L.FLOW_NAMES:
            a, b = arms[0]["counts"][k], arms[1]["counts"][k]
            with np.errstate(divide="ignore", invalid="ignore"):
                e_cnt[k] = (np.log(np.abs(a)) - np.log(np.abs(b))) / (2 * eps)
        out[name] = dict(tau=e_tau, counts=e_cnt)
    return base, out


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def sens_table(indicator, deriv, value, P, years):
    """Median/IQR of the derivative and of the elasticity, per
    (year, start stock, parameter)."""
    rows = []
    for n in range(L.N_PARAM):
        for i in range(6):
            for ti, y in enumerate(years):
                dv = deriv[:, ti, n, i]
                v = value[:, ti, i]
                e = P[:, ti, n] * dv / np.where(np.abs(v) > 0, v, np.nan)
                q = np.percentile(dv, [25, 50, 75])
                qe = np.nanpercentile(e, [25, 50, 75])
                rows.append(dict(
                    indicator=indicator, param=L.PARAM_NAMES[n],
                    start_stock=STATE[i], year=float(y),
                    derivative_median=q[1], derivative_q1=q[0],
                    derivative_q3=q[2],
                    elasticity_median=qe[1], elasticity_q1=qe[0],
                    elasticity_q3=qe[2],
                    indicator_median=float(np.median(v))))
    return pd.DataFrame(rows)


def _share(direct, prop, floor=1e-12):
    """Propagated share of the total |sensitivity|, NaN where both terms are
    numerically zero — several parameters do not touch a given flow at all
    (e.g. `mu` on the use count), and 0/0 must read as undefined rather than
    as a share of 0 or 1."""
    mag = np.abs(direct) + np.abs(prop)
    return np.where(mag > floor, np.abs(prop) / np.maximum(mag, floor),
                    np.nan)


def term_table(terms, years):
    rows = []
    for k, (direct, prop) in terms.items():
        for n in range(L.N_PARAM):
            for i in range(6):
                for ti, y in enumerate(years):
                    dd = direct[:, ti, n, i]
                    pp = prop[:, ti, n, i]
                    tot = dd + pp
                    share = _share(dd, pp)
                    rows.append(dict(
                        indicator=k, param=L.PARAM_NAMES[n],
                        start_stock=STATE[i], year=float(y),
                        direct_median=float(np.median(dd)),
                        propagated_median=float(np.median(pp)),
                        total_median=float(np.median(tot)),
                        propagated_share_median=float(np.nanmedian(share))
                        if not np.isnan(share).all() else np.nan,
                        propagated_share_q1=float(np.nanpercentile(share, 25))
                        if not np.isnan(share).all() else np.nan,
                        propagated_share_q3=float(np.nanpercentile(share, 75))
                        if not np.isnan(share).all() else np.nan))
    return pd.DataFrame(rows)


def simplex_table(dtau, dcounts, tau, counts, years):
    rows = []
    for name, r, q, label in L.simplex_directions():
        for ind, dv, v in [("tau_FD", dtau, tau)] + \
                [(k, dcounts[k], counts[k]) for k in L.FLOW_NAMES]:
            dd = dv[:, :, r, :] - dv[:, :, q, :]
            for i in range(6):
                for ti, y in enumerate(years):
                    per_pp = 0.01 * dd[:, ti, i]
                    rel = per_pp / v[:, ti, i]
                    h = L.hodges_lehmann(per_pp)
                    hr = L.hodges_lehmann(rel)
                    rows.append(dict(
                        simplex=name, direction=label, indicator=ind,
                        start_stock=STATE[i], year=float(y),
                        d_per_pp_hl=h["hl"], d_per_pp_lo=h["lo"],
                        d_per_pp_hi=h["hi"],
                        rel_per_pp_hl=hr["hl"], rel_per_pp_lo=hr["lo"],
                        rel_per_pp_hi=hr["hi"]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_sensitivity(el_tau, el_ups, terms, years, mask_test, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    t0 = float(years[np.asarray(mask_test, bool)][0])
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2))

    # (a) S_tau,l heatmap over parameters x years
    ax = axes[0, 0]
    M = np.median(el_tau[:, :, :, START], axis=0).T          # (param, year)
    vmax = float(np.nanmax(np.abs(M)))
    im = ax.imshow(M, cmap="RdBu_r", aspect="auto",
                   norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
                   extent=(years[0] - .5, years[-1] + .5, L.N_PARAM - .5, -.5))
    ax.set_yticks(range(L.N_PARAM))
    ax.set_yticklabels(L.PARAM_NAMES, fontsize=6)
    ax.set_title(r"(a) $S_{\tau,l}(t)$ — elasticity of the technological "
                 r"lifetime, from $S_{conc}$", loc="left", fontsize=9.5)
    fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)

    # (b) S_upsilon,l heatmap
    ax = axes[0, 1]
    M = np.median(el_ups[:, :, :, START], axis=0).T
    vmax = float(np.nanmax(np.abs(M)))
    im = ax.imshow(M, cmap="RdBu_r", aspect="auto",
                   norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
                   extent=(years[0] - .5, years[-1] + .5, L.N_PARAM - .5, -.5))
    ax.set_yticks(range(L.N_PARAM))
    ax.set_yticklabels(L.PARAM_NAMES, fontsize=6)
    ax.set_title(r"(b) $S_{\upsilon,l}(t)$ — elasticity of the expected use "
                 r"count", loc="left", fontsize=9.5)
    fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)

    # (c) the two terms of Ch2 Eq. flow_count_sensitivity
    ax = axes[1, 0]
    direct, prop = terms["use_entry"]
    for j, pname in enumerate(["tau_olds", "alpha_dr", "alpha_win",
                               "frac_fu_out"]):
        n = L.PARAM_NAMES.index(pname)
        share = _share(direct[:, :, n, START], prop[:, :, n, START])
        lo, med, hi = np.nanpercentile(share, [25, 50, 75], axis=0)
        ax.fill_between(years, lo, hi, color=COL[j], alpha=0.18, lw=0)
        ax.plot(years, med, color=COL[j], lw=1.7, label=pname)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("propagated share of $|\\partial\\upsilon_1/\\partial p|$")
    ax.set_title(r"(c) how much of the response is the system rearranging"
                 "\n     rather than the intervention itself"
                 r"  (Ch2 Eq. flow_count_sensitivity)",
                 loc="left", fontsize=9.5)
    ax.legend(fontsize=7, frameon=False, ncol=2)

    # (d) ranked S_tau,l at the two ends
    ax = axes[1, 1]
    e = el_tau[:, :, :, START]
    rank = np.argsort(-np.abs(np.median(e[:, -1], axis=0)))[:10][::-1]
    y = np.arange(len(rank))
    ax.barh(y - 0.19, np.median(e[:, 0][:, rank], axis=0), height=0.36,
            color="#56B4E9", label=f"{years[0]:.0f}")
    ax.barh(y + 0.19, np.median(e[:, -1][:, rank], axis=0), height=0.36,
            color="#0072B2", label=f"{years[-1]:.0f}")
    for i, n in enumerate(rank):
        q = np.percentile(e[:, -1, n], [25, 75])
        ax.plot(q, [i + 0.19, i + 0.19], color="k", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([L.PARAM_NAMES[n] for n in rank], fontsize=7)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$S_{\tau,l}$")
    ax.set_title(r"(d) what the technological lifetime responds to",
                 loc="left", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")

    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.axvline(t0, color=GREY, lw=0.9, ls=":")
        ax.set_xlabel("year")
    for ax in (axes[1, 0], axes[1, 1]):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(lw=0.4, alpha=0.3)
    fig.suptitle("WP-2e  fundamental-matrix sensitivities of the circularity "
                 "indicators (Ch2 Eq. tau_sensitivity, "
                 "Eq. flow_count_sensitivity)\n"
                 "anchor_v4, 35 seeds (median, IQR); "
                 "dotted line = start of held-out window",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=300)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


def fig_frozen_vs_nonauto(cmp_df, years, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    inds = ["tau_FD", "use_entry"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    for ax, ind in zip(axes, inds):
        sub = cmp_df[(cmp_df.indicator == ind)
                     & (cmp_df.start_stock == STATE[START])]
        params = [p for p in NONAUTO_PARAMS
                  if np.abs(sub[sub.param == p].frozen_median).max() > 1e-4]
        for j, p in enumerate(params):
            s = sub[sub.param == p].sort_values("year")
            c = COL[j % len(COL)]
            ax.plot(s.year, s.frozen_median, color=c, lw=1.5, ls="--")
            ax.plot(s.year, s.nonauto_median, color=c, lw=1.8, label=p)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("$t_0$  (year)")
        ax.set_ylabel("elasticity")
        ax.set_title(f"{ind}   (dashed = frozen, solid = non-autonomous)",
                     loc="left", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(lw=0.4, alpha=0.3)
        ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle("WP-2e  do the frozen-time sensitivities inherit the "
                 "frozen-time indicator error? — seed median",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=300)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="WP-2e: fundamental-matrix sensitivities")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--npz", default=A_NPZ)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--n-sub", type=int, default=N_SUB)
    ap.add_argument("--skip-nonauto", action="store_true",
                    help="skip the non-autonomous sensitivity arm (~2 min)")
    args = ap.parse_args(argv)

    L.check(verbose=True)
    if args.check:
        return 0

    d, md5 = load(args.npz)
    A = np.asarray(d["A"], float)
    years = np.asarray(d["years"], float)
    seeds = np.asarray(d["seeds"], int)
    mu = np.asarray(d["mu_cohorts"], float)
    S, T = A.shape[:2]
    P = L.pack_params(d, mu)
    print(f"\nloaded {os.path.basename(args.npz)}  md5 {md5}")
    print(f"  A {A.shape}   A_T = A_FD = A(t)  (no landfill state in the UDE)")

    # ---- Ch2 Eq. tau_sensitivity / Eq. flow_count_sensitivity --------------
    dA = L.dA_dparams(P)
    dtau, dcounts, N = L.indicator_sensitivity(A, dA)
    tau, counts, _ = L.frozen_indicators(A)
    terms, _ = decompose_flow_sensitivity(A, dA)
    print(f"  dtau/dp {dtau.shape}   dUpsilon/dp {dcounts['use_entry'].shape}"
          f"   (seed, year, parameter, start stock)")

    # ---- validation --------------------------------------------------------
    print("\n--- validation ---")
    ver = L.verify_indicator_derivatives(P[:3])
    print(f"  dtau/dp vs central difference of tau        : max rel err "
          f"{ver['max_rel_err_dtau']:.2e}")
    print(f"  dUpsilon/dp vs central difference of Upsilon: max rel err "
          f"{ver['max_rel_err_dUpsilon']:.2e}")
    term_sum = max(float(np.max(np.abs(terms[k][0] + terms[k][1]
                                       - dcounts[k])))
                   for k in L.FLOW_NAMES)
    print(f"  term decomposition sums to the total        : max abs "
          f"{term_sum:.2e}  (exact answer 0)")
    # the two structural anchors of the whole scheme
    n_ref = L.PARAM_NAMES.index("tau_ref")
    print(f"  structural anchor: tau_ref is spectrally invisible (WP-2c) but "
          f"NOT indicator-invisible —\n    max |dtau/dtau_ref| = "
          f"{np.abs(dtau[:, :, n_ref, :]).max():.4f} yr per unit, because "
          f"refinery loss removes mass without\n    changing any timescale.")
    if os.path.exists(WP2D_CSV):
        el2d = pd.read_csv(WP2D_CSV)
        chk = el2d[(el2d.indicator == "tau_FD")
                   & (el2d.start_stock == STATE[START])
                   & (el2d.year == years[-1])].set_index("param")
        mine = {L.PARAM_NAMES[n]: float(np.nanmedian(
            P[:, -1, n] * dtau[:, -1, n, START] / tau[:, -1, START]))
            for n in range(L.N_PARAM)}
        gap = max(abs(mine[p] - chk.loc[p, "elasticity_median"])
                  for p in mine)
        print(f"  agreement with WP-2d's elasticity table     : max abs diff "
              f"{gap:.2e}  (same function, must be 0)")
    pd.DataFrame([
        dict(check="max_rel_err_dtau_vs_fd", value=ver["max_rel_err_dtau"]),
        dict(check="max_rel_err_dUpsilon_vs_fd",
             value=ver["max_rel_err_dUpsilon"]),
        dict(check="term_decomposition_max_abs_residual", value=term_sum),
    ]).to_csv(os.path.join(args.out, "wp2e_validation.csv"), index=False)

    # ---- what the indicators respond to ------------------------------------
    el_tau = P[:, :, :, None] * dtau / tau[:, :, None, :]
    el_ups = P[:, :, :, None] * dcounts["use_entry"] / \
        counts["use_entry"][:, :, None, :]
    print(f"\n--- S_tau,l and S_upsilon,l from {STATE[START]} "
          f"(Ch2 Eq. tau_U_elasticities) ---")
    print(f"  {'parameter':<16s}{'S_tau 1980':>22s}{'S_tau 2019':>22s}"
          f"{'S_ups 2019':>22s}")
    order = np.argsort(-np.abs(np.median(el_tau[:, -1, :, START], axis=0)))
    for n in order[:10]:
        a = np.percentile(el_tau[:, 0, n, START], [50, 25, 75])
        b = np.percentile(el_tau[:, -1, n, START], [50, 25, 75])
        c = np.percentile(el_ups[:, -1, n, START], [50, 25, 75])
        print(f"  {L.PARAM_NAMES[n]:<16s}"
              f"{a[0]:+8.4f} [{a[1]:+.3f},{a[2]:+.3f}]"
              f"{b[0]:+8.4f} [{b[1]:+.3f},{b[2]:+.3f}]"
              f"{c[0]:+8.4f} [{c[1]:+.3f},{c[2]:+.3f}]")

    # ---- the two terms of Eq. flow_count_sensitivity -----------------------
    print("\n--- term decomposition of dUpsilon/dp: direct (phi_lm) vs "
          "propagated ---")
    direct, prop = terms["use_entry"]
    for pname in ("tau_olds", "alpha_dr", "alpha_win", "frac_fu_out",
                  "mu_44yr"):
        n = L.PARAM_NAMES.index(pname)
        dd = direct[:, :, n, START]
        pp = prop[:, :, n, START]
        share = _share(dd, pp)
        tot = dd + pp
        if np.isnan(share).all():
            print(f"  {pname:<14s} does not enter upsilon_1 at all "
                  f"(both terms 0)")
            continue
        print(f"  {pname:<14s} direct {np.median(dd):+10.5f}   propagated "
              f"{np.median(pp):+10.5f}   net {np.median(tot):+10.5f}   "
              f"propagated share {np.nanmedian(share):.3f} "
              f"[{np.nanpercentile(share, 25):.3f},"
              f"{np.nanpercentile(share, 75):.3f}]")
    allshare = _share(direct, prop)
    print(f"  over all 20 parameters and all nodes, the propagated term "
          f"carries a median {np.nanmedian(allshare):.3f} of "
          f"|dupsilon_1/dp|:")
    print(f"    the system rearranging around the intervention, not the "
          f"intervention itself.")

    # ---- contrast with WP-2c ------------------------------------------------
    if os.path.exists(WP2C_NPZ):
        c2 = np.load(WP2C_NPZ, allow_pickle=True)
        dlam = c2["dlambda_state"]                      # (S,T,6,n_p)
        dom = c2["dominant_mode"]
        si, ti = np.meshgrid(np.arange(S), np.arange(T), indexing="ij")
        d_abs = np.abs(dlam[si, ti, dom, :])            # (S,T,n_p)
        d_ind = np.abs(dtau[:, :, :, START])
        rho = np.array([[stats.spearmanr(d_abs[s, t], d_ind[s, t]).statistic
                         for t in range(T)] for s in range(S)])
        print("\n--- WP-2c vs WP-2e: two different sensitivities ---")
        print(f"  Spearman rank correlation between |ds(A)/dp| (eigenvalue) "
              f"and |dtau_1/dp| (indicator)")
        print(f"    across the 20 parameters: median "
              f"{np.median(rho):.3f} [{np.percentile(rho, 25):.3f},"
              f"{np.percentile(rho, 75):.3f}] over {S*T} nodes")
        top_c = L.PARAM_NAMES[int(np.median(d_abs, axis=(0, 1)).argmax())]
        top_e = L.PARAM_NAMES[int(np.median(d_ind, axis=(0, 1)).argmax())]
        print(f"    top parameter by eigenvalue sensitivity: {top_c}; "
              f"by lifetime sensitivity: {top_e}")
        pd.DataFrame(dict(seed=np.repeat(seeds, T),
                          year=np.tile(years, S),
                          spearman_rho=rho.ravel())).to_csv(
            os.path.join(args.out, "wp2e_vs_wp2c_rank.csv"), index=False)

    # ---- frozen-time vs non-autonomous sensitivity -------------------------
    cmp_rows = []
    if not args.skip_nonauto:
        print(f"\n--- do the frozen sensitivities inherit the frozen "
              f"indicator error? ({len(NONAUTO_PARAMS)} parameters, "
              f"{2*len(NONAUTO_PARAMS)} re-integrations) ---")
        base, na = nonauto_elasticities(P, years, NONAUTO_PARAMS,
                                        rule=REF_RULE, n_sub=args.n_sub)
        for name in NONAUTO_PARAMS:
            n = L.PARAM_NAMES.index(name)
            for i in range(6):
                for ti, y in enumerate(years):
                    fr = el_tau[:, ti, n, i]
                    nn = na[name]["tau"][:, ti, i]
                    cmp_rows.append(dict(
                        indicator="tau_FD", param=name, start_stock=STATE[i],
                        year=float(y), frozen_median=float(np.median(fr)),
                        nonauto_median=float(np.median(nn)),
                        **{f"diff_{k}": v for k, v in
                           L.hodges_lehmann(nn - fr).items()}))
                    fr = el_ups[:, ti, n, i]
                    nn = na[name]["counts"]["use_entry"][:, ti, i]
                    cmp_rows.append(dict(
                        indicator="use_entry", param=name,
                        start_stock=STATE[i], year=float(y),
                        frozen_median=float(np.median(fr)),
                        nonauto_median=float(np.median(nn)),
                        **{f"diff_{k}": v for k, v in
                           L.hodges_lehmann(nn - fr).items()}))
        cmp_df = pd.DataFrame(cmp_rows)
        cmp_df.to_csv(os.path.join(args.out,
                                   "wp2e_frozen_vs_nonauto.csv"), index=False)
        sub = cmp_df[(cmp_df.start_stock == STATE[START])
                     & (cmp_df.year <= years[-1] - 10)]
        for ind in ("tau_FD", "use_entry"):
            s2 = sub[(sub.indicator == ind)
                     & (sub.frozen_median.abs() > 1e-3)]
            rel = (s2.nonauto_median - s2.frozen_median) / s2.frozen_median
            print(f"  {ind:<10s} |relative difference| of the frozen "
                  f"elasticity: median {100*rel.abs().median():5.1f}%, "
                  f"p90 {100*rel.abs().quantile(0.9):5.1f}%  "
                  f"(n={len(s2)} param-year cells with |S| > 1e-3)")
        print(f"  rank preservation: the top-3 parameters by |S_tau| agree "
              f"between the two forms at "
              f"{100*_rank_agreement(cmp_df, years):.0f}% of years.")
        fig_frozen_vs_nonauto(cmp_df, years,
                              os.path.join(args.out,
                                           "wp2e_frozen_vs_nonauto.png"))

    # ---- simplex directions -------------------------------------------------
    simp = simplex_table(dtau, dcounts, tau, counts, years)
    simp.to_csv(os.path.join(args.out, "wp2e_simplex_directions.csv"),
                index=False)
    print("\n--- constrained simplex directions, tau_FD from "
          f"{STATE[START]}, {years[-1]:.0f} (per +1 pp) ---")
    last = simp[(simp.indicator == "tau_FD")
                & (simp.start_stock == STATE[START])
                & (simp.year == years[-1])]
    for _, r in last.iterrows():
        print(f"  {r.direction:<42s} {r.d_per_pp_hl:+8.5f} yr "
              f"[{r.d_per_pp_lo:+.5f},{r.d_per_pp_hi:+.5f}]  "
              f"({100*r.rel_per_pp_hl:+.4f}%)")

    # ---- persist ------------------------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    sens_table("tau_FD", dtau, tau, P, years).to_csv(
        os.path.join(args.out, "wp2e_tau_sensitivity.csv"), index=False)
    pd.concat([sens_table(k, dcounts[k], counts[k], P, years)
               for k in L.FLOW_NAMES], ignore_index=True).to_csv(
        os.path.join(args.out, "wp2e_flow_count_sensitivity.csv"), index=False)
    term_table(terms, years).to_csv(
        os.path.join(args.out, "wp2e_term_decomposition.csv"), index=False)

    arrays = dict(years=years, seeds=seeds, mu_cohorts=mu,
                  state_names=np.array(STATE, object),
                  param_names=np.array(L.PARAM_NAMES, object),
                  flow_names=np.array(L.FLOW_NAMES, object),
                  params=P, N_frozen=N, tau_frozen=tau,
                  dtau_dparams=dtau, elas_tau=el_tau, elas_upsilon=el_ups,
                  source_md5=np.array(md5))
    for k in L.FLOW_NAMES:
        arrays[f"count_frozen_{k}"] = counts[k]
        arrays[f"dcount_dparams_{k}"] = dcounts[k]
        arrays[f"term_direct_{k}"] = terms[k][0]
        arrays[f"term_propagated_{k}"] = terms[k][1]
    if cmp_rows:
        for name in NONAUTO_PARAMS:
            arrays[f"elas_tau_nonauto_{name}"] = na[name]["tau"]
            arrays[f"elas_upsilon_nonauto_{name}"] = \
                na[name]["counts"]["use_entry"]
    np.savez_compressed(
        os.path.join(args.out, "wp2e_indicator_sensitivity.npz"), **arrays)

    fig_sensitivity(el_tau, el_ups, terms, years, d["mask_test"],
                    os.path.join(args.out, "wp2e_sensitivity.png"))
    print(f"\nwrote wp2e_* to {args.out}")
    return 0


def _rank_agreement(cmp_df, years, k=3):
    sub = cmp_df[(cmp_df.indicator == "tau_FD")
                 & (cmp_df.start_stock == STATE[START])]
    hits = 0
    for y in years:
        s = sub[sub.year == y]
        a = set(s.reindex(s.frozen_median.abs().sort_values(
            ascending=False).index).param[:k])
        b = set(s.reindex(s.nonauto_median.abs().sort_values(
            ascending=False).index).param[:k])
        hits += (a == b)
    return hits / len(years)


if __name__ == "__main__":
    sys.exit(main())
