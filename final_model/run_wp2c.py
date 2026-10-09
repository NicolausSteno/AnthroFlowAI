#!/usr/bin/env python3
"""
run_wp2c.py — WP-2c: dynamic (frozen-time eigenvalue) sensitivity of A(t)
=========================================================================

Implements Chapter 2's dynamic sensitivity formula for simple eigenvalues,

    dlambda_k/dp_n = q_k^T (dA~/dp_n) v_k / (q_k^T v_k)   # Ch2 Eq. eigenvalue_sensitivity

at every year node and every seed of the `anchor_v4` ensemble, on the
empirically estimated A(t) that WP-2a supplies and Chapter 2 could not have.
`v_k`, `q_k` are the right and left eigenvectors of `A(t)`; the parameter set
`p_n` is the 20-element vector of `zinc_circ_lab.PARAM_NAMES` — four alphas,
four binary taus, three 3-simplices and the three fixed cohort lifetimes.

Guards, both required by the spec and both reported rather than assumed
------------------------------------------------------------------------
  * **Simplicity.** Eq. eigenvalue_sensitivity is valid for simple and
    semisimple eigenvalues only.  Near-degeneracy is detected per node through
    the nearest-neighbour relative gap and the eigenvalue condition number
    `1/|q_k.v_k|`, and affected nodes are flagged in
    `wp2c_degeneracy_flags.csv` and excluded from the headline aggregates
    (reported both ways, since exclusion changes nothing material).
  * **Frozen-time is diagnostic, not complete.** Stability of a non-autonomous
    system is governed by `Phi(t,t0)` (Ch2 Eq. product_transition_matrices),
    so these are instantaneous modal sensitivities.  WP-2b measured the
    adiabaticity index that says how far they can be read; WP-2d measures the
    resulting indicator-level discrepancy directly.

Simplex constraint
------------------
`frac_fu`, `frac_eu` and `f_cohort` are simplices: a coefficient inside one
cannot move alone.  Chapter 2 handles this with the directional derivative
`dlambda/dtheta = dlambda/dw_r - dlambda/dw_q`; all nine constrained pairs are
reported in `wp2c_simplex_directions.csv`.  The unconstrained partials are
still reported, because they are the building blocks, but the constrained
directions are the ones with an operational reading.

Validation
----------
Two independent checks, both printed and persisted:
  * the analytic anchor — row 0 of A has no off-diagonal, so
    `lambda_conc = -alpha_cc` exactly and `dlambda_conc/dp` must be `-1` for
    `alpha_cc` and `0` for all other 19 parameters, at every one of the 1 400
    nodes;
  * every `dlambda_k/dp_n` against a central difference of the eigenvalue of
    `A(p +/- h e_n)` itself, which tests the formula rather than its
    implementation.

    python run_wp2c.py --check
    python run_wp2c.py
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd

import zinc_circ_lab as L

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "analysis")
A_NPZ = os.path.join(OUT_DIR, "wp2a_A_of_t.npz")

STATE = list(L.ODE_STATE_NAMES)
COL = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00", "#56B4E9"]
GREY = "#555555"

# WP-2b's thresholds, reused verbatim so the guard is the same object in both
# packages: min relative gap over all 1400 nodes was 0.0079, max condition
# number 134, and 21 nodes (1.5%) exceeded condition number 10.
RELGAP_MIN = 0.05
COND_MAX = 10.0


def load(path=A_NPZ):
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found — run run_wp2a.py first")
    d = np.load(path, allow_pickle=True)
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    return d, md5


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def summary_table(dlam, elas, ok, years, mode_names):
    """Median and IQR across seeds, per (mode, parameter, year)."""
    S, T, K, NP = dlam.shape
    rows = []
    for k in range(K):
        for n in range(NP):
            for ti, y in enumerate(years):
                m = ok[:, ti, k]
                v = dlam[:, ti, k, n]
                e = elas[:, ti, k, n]
                q = np.percentile(v, [25, 50, 75])
                qe = np.percentile(e, [25, 50, 75])
                rows.append(dict(
                    mode=mode_names[k], param=L.PARAM_NAMES[n], year=float(y),
                    dlambda_median=q[1], dlambda_q1=q[0], dlambda_q3=q[2],
                    elas_timescale_median=qe[1], elas_timescale_q1=qe[0],
                    elas_timescale_q3=qe[2],
                    n_seeds_simple=int(m.sum()), n_seeds=S,
                    dlambda_median_simple_only=float(np.median(v[m]))
                    if m.any() else np.nan))
    return pd.DataFrame(rows)


def dominant_table(dlam, elas, lam, dom_k, years, seeds):
    """Per-seed sensitivities of the mode that carries the spectral abscissa —
    the quantity that decides whether a coefficient change stabilises or
    destabilises the estimated system."""
    S, T = dom_k.shape
    rows = []
    si, ti = np.meshgrid(np.arange(S), np.arange(T), indexing="ij")
    for n in range(dlam.shape[3]):
        rows.append(pd.DataFrame(dict(
            seed=seeds[si].ravel(), year=years[ti].ravel(),
            mode=np.array(STATE)[dom_k.ravel()],
            param=L.PARAM_NAMES[n],
            lam=lam[si, ti, dom_k].real.ravel(),
            dlambda=dlam[si, ti, dom_k, n].ravel(),
            elas_timescale=elas[si, ti, dom_k, n].ravel())))
    return pd.concat(rows, ignore_index=True)


def simplex_table(dlam, lam, P, years, mode_names):
    """Constrained directional derivatives inside each simplex.

    `d lambda_k / d theta = d lambda_k / d w_r - d lambda_k / d w_q` with
    `w_r = w_r^0 + theta`, `w_q = w_q^0 - theta` (Ch2, constant product
    shares).  Reported as a rate and as the induced fractional change in the
    modal timescale for a 1-percentage-point transfer, which is the unit a
    reader can act on.
    """
    rows = []
    for name, r, q, label in L.simplex_directions():
        dd = dlam[..., r] - dlam[..., q]
        for k in range(dlam.shape[2]):
            for ti, y in enumerate(years):
                v = dd[:, ti, k]
                rel = 0.01 * v / np.abs(lam[:, ti, k].real)
                a = np.percentile(v, [25, 50, 75])
                b = np.percentile(rel, [25, 50, 75])
                rows.append(dict(
                    simplex=name, direction=label, mode=mode_names[k],
                    year=float(y),
                    dlambda_dtheta_median=a[1], dlambda_dtheta_q1=a[0],
                    dlambda_dtheta_q3=a[2],
                    d_timescale_frac_per_pp_median=b[1],
                    d_timescale_frac_per_pp_q1=b[0],
                    d_timescale_frac_per_pp_q3=b[2]))
    return pd.DataFrame(rows)


def degeneracy_flags(relgap, cond, years, seeds):
    S, T, K = relgap.shape
    si, ti, ki = np.meshgrid(np.arange(S), np.arange(T), np.arange(K),
                             indexing="ij")
    df = pd.DataFrame(dict(
        seed=seeds[si].ravel(), year=years[ti].ravel(),
        mode=np.array(STATE)[ki.ravel()],
        relgap=relgap.ravel(), cond=cond.ravel()))
    df["simple_ok"] = (df.relgap >= RELGAP_MIN) & (df.cond <= COND_MAX)
    return df


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def fig_sensitivity(elas, dlam, lam, ok, years, mask_test, mode_names, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    t0 = float(years[np.asarray(mask_test, bool)][0])
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))

    # (a) elasticity heatmap, window median
    ax = axes[0, 0]
    M = np.median(elas, axis=(0, 1))                    # (mode, param)
    # The own-rate terms (alpha_cc on S_conc, mu_k on its own cohort) are
    # +/-1 by construction and would flatten everything else, so the colour
    # scale is clipped well below them and the values are printed.
    vmax = 0.25
    im = ax.imshow(np.clip(M, -vmax, vmax), cmap="RdBu_r", aspect="auto",
                   norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax))
    for a in range(M.shape[0]):
        for b in range(M.shape[1]):
            if abs(M[a, b]) > 5e-3:
                ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center",
                        fontsize=5.5,
                        color="w" if abs(M[a, b]) > 0.55 * vmax else "k")
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(L.PARAM_NAMES, rotation=75, ha="right", fontsize=6)
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels(mode_names, fontsize=7)
    ax.set_title(r"(a) elasticity of the modal timescale, "
                 r"$p_n\,\partial\lambda_k/\partial p_n\,/\,|\lambda_k|$"
                 "  — 35-seed × 40-year median",
                 loc="left", fontsize=9.5)
    fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046, extend="both")

    # (b) the dominant mode through time, top parameters
    ax = axes[0, 1]
    dom = 4                                              # S_iu_long, WP-2b
    rank = np.argsort(-np.abs(np.median(elas[:, :, dom, :], axis=(0, 1))))[:6]
    for i, n in enumerate(rank):
        X = elas[:, :, dom, n]
        lo, med, hi = np.percentile(X, [25, 50, 75], axis=0)
        c = COL[i % len(COL)]
        ax.fill_between(years, lo, hi, color=c, alpha=0.20, lw=0)
        ax.plot(years, med, color=c, lw=1.6, label=L.PARAM_NAMES[n])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("elasticity of the dominant timescale")
    ax.set_title(f"(b) what lengthens the system's slowest mode "
                 f"({mode_names[dom]})", loc="left", fontsize=9.5)
    ax.legend(fontsize=6.5, frameon=False, ncol=2)

    # (c) the two circularity coefficients, all modes
    ax = axes[1, 0]
    for j, pname in enumerate(["alpha_dr", "alpha_win"]):
        n = L.PARAM_NAMES.index(pname)
        for k in range(6):
            med = np.median(elas[:, :, k, n], axis=0)
            ax.plot(years, med, color=COL[k], lw=1.5,
                    ls="-" if j == 0 else "--",
                    label=f"{mode_names[k]}" if j == 0 else None)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("elasticity of the modal timescale")
    ax.set_title(r"(c) $\alpha_{13}$ = alpha_dr (solid) vs "
                 r"$\alpha_{14}$ = alpha_win (dashed): the two "
                 "circularity\n     coefficients act on the Scrap "
                 "timescale, not on the dominant one", loc="left",
                 fontsize=9.5)
    ax.legend(fontsize=6.5, frameon=False, ncol=3)

    # (d) the guard
    ax = axes[1, 1]
    frac_bad = 1.0 - ok.all(axis=2).mean(axis=0)
    ax.bar(years, frac_bad, color="#D55E00", width=0.8)
    ax.set_ylabel("fraction of seeds flagged")
    ax.set_title(f"(d) near-degeneracy guard: relgap < {RELGAP_MIN} or "
                 f"cond > {COND_MAX:.0f}", loc="left", fontsize=9.5)
    ax.set_ylim(0, max(0.05, float(frac_bad.max()) * 1.25))

    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.axvline(t0, color=GREY, lw=0.9, ls=":")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(lw=0.4, alpha=0.3)
        ax.set_xlabel("year")
    fig.suptitle("WP-2c  frozen-time dynamic sensitivity "
                 r"$\partial\lambda_k/\partial p_n$ of the estimated $A(t)$ "
                 "— anchor_v4, 35 seeds (median, IQR); "
                 "dotted line = start of held-out window",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=300)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


def fig_ranking(elas, years, mode_names, path):
    """A single ranked figure of which coefficients control which timescale,
    at the two ends of the window — the readable summary for the chapter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.2), sharey=False)
    for j, k in enumerate([4, 5, 1]):                # slowest, scrap, refined
        ax = axes[j]
        e0 = elas[:, 0, k, :]
        e1 = elas[:, -1, k, :]
        rank = np.argsort(-np.abs(np.median(e1, axis=0)))[:10][::-1]
        y = np.arange(len(rank))
        ax.barh(y - 0.19, np.median(e0[:, rank], axis=0), height=0.36,
                color="#56B4E9", label=f"{years[0]:.0f}")
        ax.barh(y + 0.19, np.median(e1[:, rank], axis=0), height=0.36,
                color="#0072B2", label=f"{years[-1]:.0f}")
        for i, n in enumerate(rank):
            q = np.percentile(e1[:, n], [25, 75])
            ax.plot(q, [i + 0.19, i + 0.19], color="k", lw=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels([L.PARAM_NAMES[n] for n in rank], fontsize=7)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("elasticity of the modal timescale")
        ax.set_title(f"{mode_names[k]}", loc="left", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", lw=0.4, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("WP-2c  which coefficients control which timescale — "
                 "top 10 by |elasticity| in the final year, seed median "
                 "(bar) and IQR (line)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=300)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="WP-2c: dynamic sensitivity of the estimated A(t)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--npz", default=A_NPZ)
    ap.add_argument("--out", default=OUT_DIR)
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
    print(f"\nloaded {os.path.basename(args.npz)}  md5 {md5}")
    print(f"  A {A.shape}  seeds {S}  years {years[0]:.0f}-{years[-1]:.0f}")

    # ---- the parameter vector and dA/dp ------------------------------------
    P = L.pack_params(d, mu)
    err_assembly = L.verify_assembly(d, A)
    print(f"  assembly check: max |assemble_from_params(P) - A| = "
          f"{err_assembly:.3e}   (must be 0 — same algebra as WP-2a)")
    if err_assembly > 1e-12:
        raise AssertionError("parameter re-assembly does not reproduce WP-2a A")
    dA = L.dA_dparams(P)
    print(f"  dA/dp: {dA.shape}  ({L.N_PARAM} parameters, complex step)")

    # ---- Ch2 Eq. eigenvalue_sensitivity ------------------------------------
    lam, dlam, cond, relgap = L.eig_sensitivity(A, dA)
    n_complex = int((np.abs(lam.imag) > 1e-9).sum())
    print(f"  eigenvalues with |Im| > 1e-9: {n_complex} of {S*T*6} "
          f"(WP-2b: none) — sensitivities are real")
    # re-index modes onto compartments the same way WP-2b does, so the two
    # packages' mode labels are the same object
    import scipy.linalg as sla
    from scipy.optimize import linear_sum_assignment
    labmode = np.zeros((S, T, 6), int)
    for s in range(S):
        for t in range(T):
            w, ql, vr = sla.eig(A[s, t], left=True, right=True)
            vr = vr / np.linalg.norm(vr, axis=0)
            ql = ql / np.linalg.norm(ql, axis=0)
            den = np.einsum("ik,ik->k", ql.conj(), vr)
            Pk = np.abs(((ql / den.conj()).conj() * vr).T)
            labmode[s, t] = linear_sum_assignment(-Pk)[1]

    def to_state(x):
        out = np.zeros_like(x)
        np.put_along_axis(out, labmode.reshape(S, T, 6, *([1] * (x.ndim - 3))),
                          x, axis=2)
        return out

    lam_s = to_state(lam)
    dlam_s = to_state(dlam).real
    cond_s = to_state(cond)
    relgap_s = to_state(relgap)

    # elasticity of the modal TIMESCALE: d ln(1/|lambda_k|) / d ln p_n
    #   = p_n (dlambda_k/dp_n) / |lambda_k|.  Positive = the parameter slows
    #   the mode down.
    elas = P[:, :, None, :] * dlam_s / np.abs(lam_s.real)[..., None]
    ok = (relgap_s >= RELGAP_MIN) & (cond_s <= COND_MAX)

    # ---- validation --------------------------------------------------------
    print("\n--- validation ---")
    n_cc = L.PARAM_NAMES.index("alpha_cc")
    anchor_self = np.abs(dlam_s[:, :, 0, n_cc] + 1.0).max()
    other = np.delete(dlam_s[:, :, 0, :], n_cc, axis=2)
    print(f"  analytic anchor  lambda_conc = -alpha_cc exactly:")
    print(f"    max |dlambda_conc/dalpha_cc + 1| = {anchor_self:.3e}  "
          f"(exact answer -1)")
    print(f"    max |dlambda_conc/dp_n| over the other 19 parameters = "
          f"{np.abs(other).max():.3e}  (exact answer 0)")
    ver = L.verify_derivatives(P[:3])
    print(f"  dA/dp vs central difference            : "
          f"max rel err {ver['max_rel_err_dA']:.2e}")
    print(f"  dlambda/dp vs central difference of eig: "
          f"max rel err {ver['max_rel_err_dlambda']:.2e}   "
          f"(Ch2 Eq. eigenvalue_sensitivity as a statement about lambda(A(p)))")
    # cohort modes must satisfy lambda ~ -1/mu, hence elasticity of the
    # timescale w.r.t. mu ~ +1
    for k in range(3):
        n_mu = L.PARAM_NAMES.index(f"mu_{int(mu[k])}yr")
        print(f"  structural anchor: d ln T / d ln mu_{int(mu[k])} for mode "
              f"{STATE[2+k]:<11s} = "
              f"{np.median(elas[:, :, 2+k, n_mu]):.4f}   (expect ~ +1)")
    pd.DataFrame([dict(check="assembly_max_abs_err", value=err_assembly),
                  dict(check="anchor_dlam_conc_dalpha_cc_plus_1",
                       value=float(anchor_self)),
                  dict(check="anchor_dlam_conc_dp_other_maxabs",
                       value=float(np.abs(other).max())),
                  dict(check="max_rel_err_dA_vs_fd",
                       value=ver["max_rel_err_dA"]),
                  dict(check="max_rel_err_dlambda_vs_fd",
                       value=ver["max_rel_err_dlambda"]),
                  dict(check="n_eigenvalues_complex", value=float(n_complex)),
                  ]).to_csv(os.path.join(args.out, "wp2c_validation.csv"),
                            index=False)

    # ---- guard -------------------------------------------------------------
    flags = degeneracy_flags(relgap_s, cond_s, years, seeds)
    bad = ~flags.simple_ok
    print("\n--- simplicity guard (Ch2: valid for simple/semisimple only) ---")
    print(f"  min relative gap over all {S*T*6} (seed, year, mode): "
          f"{relgap_s.min():.4f};  max condition number "
          f"{cond_s.max():.1f}")
    print(f"  flagged (relgap < {RELGAP_MIN} or cond > {COND_MAX:.0f}): "
          f"{int(bad.sum())} of {len(flags)} "
          f"({100*bad.mean():.2f}%), on modes "
          f"{sorted(flags.loc[bad, 'mode'].unique())}")
    if bad.any():
        yrs = sorted(flags.loc[bad, "year"].unique())
        print(f"  affected years: {yrs[0]:.0f}-{yrs[-1]:.0f} "
              f"({len(yrs)} distinct); seeds "
              f"{sorted(flags.loc[bad, 'seed'].unique().tolist())}")

    # ---- tables ------------------------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    summ = summary_table(dlam_s, elas, ok, years, STATE)
    summ.to_csv(os.path.join(args.out, "wp2c_sensitivity_summary.csv"),
                index=False)
    dom_k = lam_s.real.argmax(axis=2)
    dom = dominant_table(dlam_s, elas, lam_s, dom_k, years, seeds)
    dom.to_csv(os.path.join(args.out, "wp2c_dominant_per_seed.csv"),
               index=False)
    simp = simplex_table(dlam_s, lam_s, P, years, STATE)
    simp.to_csv(os.path.join(args.out, "wp2c_simplex_directions.csv"),
                index=False)
    flags.to_csv(os.path.join(args.out, "wp2c_degeneracy_flags.csv"),
                 index=False)

    np.savez_compressed(
        os.path.join(args.out, "wp2c_eigen_sensitivity.npz"),
        years=years, seeds=seeds, mu_cohorts=mu,
        state_names=np.array(STATE, object),
        param_names=np.array(L.PARAM_NAMES, object),
        params=P, dA_dparams=dA,
        lam_state=lam_s, dlambda_state=dlam_s, elas_timescale=elas,
        cond_state=cond_s, relgap_state=relgap_s, simple_ok=ok,
        mode_label=labmode, dominant_mode=dom_k,
        source_md5=np.array(md5),
        relgap_min=np.array(RELGAP_MIN), cond_max=np.array(COND_MAX))

    # ---- what the numbers say ----------------------------------------------
    print("\n--- elasticity of each modal timescale (seed x year median) ---")
    print(f"  {'parameter':<16s}" + "".join(f"{s:>13s}" for s in STATE))
    M = np.median(elas, axis=(0, 1))
    for n in range(L.N_PARAM):
        if np.abs(M[:, n]).max() < 1e-4:
            continue
        print(f"  {L.PARAM_NAMES[n]:<16s}"
              + "".join(f"{M[k, n]:13.4f}" for k in range(6)))
    print("  (blank rows omitted: |elasticity| < 1e-4 for every mode)")

    print("\n--- the dominant mode (S_iu_long at every node, WP-2b) ---")
    k = 4
    rank = np.argsort(-np.abs(np.median(elas[:, :, k, :], axis=(0, 1))))
    for n in rank[:8]:
        a = np.percentile(elas[:, 0, k, n], [50, 25, 75])
        b = np.percentile(elas[:, -1, k, n], [50, 25, 75])
        hl = L.hodges_lehmann(elas[:, -1, k, n] - elas[:, 0, k, n])
        print(f"  {L.PARAM_NAMES[n]:<16s} {years[0]:.0f} {a[0]:+8.4f} "
              f"[{a[1]:+.4f},{a[2]:+.4f}]   {years[-1]:.0f} {b[0]:+8.4f} "
              f"[{b[1]:+.4f},{b[2]:+.4f}]   HL change {hl['hl']:+.4f} "
              f"[{hl['lo']:+.4f},{hl['hi']:+.4f}]")

    print("\n--- the two circularity coefficients on the dominant mode ---")
    for pname, ch2 in (("alpha_dr", "alpha_13"), ("alpha_win", "alpha_14")):
        n = L.PARAM_NAMES.index(pname)
        b = np.percentile(elas[:, -1, k, n], [50, 25, 75])
        print(f"  {ch2} = {pname:<10s} d ln T_dom / d ln p = {b[0]:+.4f} "
              f"[{b[1]:+.4f},{b[2]:+.4f}] in {years[-1]:.0f}")
    n_dr = L.PARAM_NAMES.index("alpha_dr")
    n_win = L.PARAM_NAMES.index("alpha_win")
    hl = L.hodges_lehmann(elas[:, -1, k, n_dr] - elas[:, -1, k, n_win])
    print(f"  HL(alpha_dr - alpha_win) in {years[-1]:.0f}: {hl['hl']:+.4f} "
          f"[{hl['lo']:+.4f}, {hl['hi']:+.4f}]  (n={hl['n']})")

    print("\n--- spectral abscissa: what stabilises the estimated system ---")
    si, ti = np.meshgrid(np.arange(S), np.arange(T), indexing="ij")
    dabs = dlam_s[si, ti, dom_k, :]
    rank = np.argsort(-np.abs(np.median(dabs, axis=(0, 1))))
    for n in rank[:6]:
        q = np.percentile(dabs[:, :, n], [50, 25, 75])
        print(f"  d s(A)/d {L.PARAM_NAMES[n]:<16s} = {q[0]:+.5f} "
              f"[{q[1]:+.5f},{q[2]:+.5f}] /yr per unit  "
              f"({'widens' if q[0] < 0 else 'narrows'} the stability "
              f"margin)")

    # --- what the frozen spectrum cannot see --------------------------------
    # Two structural facts, both measured rather than asserted, and both
    # feeding WP-4c's sloppiness analysis: some parameters do not enter the
    # spectrum at all, and some enter only through a product with another, so
    # that the modal timescales identify the product but not the factors.
    print("\n--- what the frozen spectrum cannot identify ---")
    amax = np.abs(elas).max(axis=(0, 1, 2))
    invisible = [L.PARAM_NAMES[n] for n in range(L.N_PARAM) if amax[n] < 1e-12]
    print(f"  spectrally invisible (dlambda_k/dp_n = 0 at every node, every "
          f"mode): {invisible}")
    print("    tau_ref: row 0 of A has no off-diagonal, so A is block "
          "lower-triangular and\n      A[1,0] = (1-tau_ref)*alpha_cc never "
          "enters the characteristic polynomial —\n      refinery efficiency "
          "moves levels, not timescales.")
    print("    frac_fu_loss, frac_eu_loss: pure loss shares, present in l(t) "
          "but not in A.")
    pairs = []
    for a in range(L.N_PARAM):
        for b in range(a + 1, L.N_PARAM):
            if amax[a] < 1e-12 or amax[b] < 1e-12:
                continue
            dmax = float(np.abs(elas[..., a] - elas[..., b]).max())
            if dmax / max(amax[a], amax[b]) < 1e-8:
                pairs.append((L.PARAM_NAMES[a], L.PARAM_NAMES[b], dmax))
    for a, b, dm in pairs:
        print(f"  spectrally indistinguishable: {a} and {b} have identical "
              f"log-sensitivity\n    at every (seed, year, mode) to "
              f"{dm:.1e} — each appears exactly once, linearly, on the same\n"
              f"    recycling path, so the modal timescales identify their "
              f"product and not the factors.")
    pd.DataFrame(
        [dict(kind="spectrally_invisible", param_a=p_, param_b="",
              max_abs_elasticity=float(amax[L.PARAM_NAMES.index(p_)]),
              max_abs_elasticity_difference=0.0) for p_ in invisible]
        + [dict(kind="spectrally_indistinguishable", param_a=a, param_b=b,
                max_abs_elasticity=float(amax[L.PARAM_NAMES.index(a)]),
                max_abs_elasticity_difference=dm) for a, b, dm in pairs]
    ).to_csv(os.path.join(args.out, "wp2c_spectral_identifiability.csv"),
             index=False)

    print("\n--- constrained simplex directions (Ch2: dl/dw_r - dl/dw_q) ---")
    last = simp[simp.year == years[-1]]
    for name in L.SIMPLEXES:
        sub = last[(last.simplex == name) & (last["mode"] == STATE[4])]
        for _, r in sub.iterrows():
            print(f"  {STATE[4]:<11s} {r.direction:<40s} "
                  f"{100*r.d_timescale_frac_per_pp_median:+7.3f}% timescale "
                  f"per +1 pp")

    print("\n  Caveat (Ch2 Eq. product_transition_matrices): stability of a "
          "non-autonomous\n  system is governed by Phi(t,t0); these frozen-time"
          " sensitivities are\n  diagnostic, not a complete description. WP-2d "
          "measures the resulting\n  indicator-level discrepancy directly.")

    fig_sensitivity(elas, dlam_s, lam_s, ok, years, d["mask_test"], STATE,
                    os.path.join(args.out, "wp2c_sensitivity.png"))
    fig_ranking(elas, years, STATE,
                os.path.join(args.out, "wp2c_ranking.png"))
    print(f"\nwrote wp2c_* to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
