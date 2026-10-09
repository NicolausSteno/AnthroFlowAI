#!/usr/bin/env python3
"""
run_wp2f.py — WP-2f: structural stability and distance to bifurcation
=====================================================================

Gandolfo (1992) §2 asks, of an estimated dynamic system, not "is it stable?"
but "how far is it from not being stable, in units of the estimation error?".
For characteristic root `mu_j` and parameter `theta_i` with standard error
`sigma_i`, the bifurcation value is `theta_i + dtheta_i`,
`dtheta_i = -mu_j / (dmu_j/dtheta_i)`, and the system is structurally
UNSTABLE with respect to `theta_i` when that value lies INSIDE the confidence
interval.  Three equivalent forms, all implemented in `zinc_circ_lab.gandolfo`
and checked against one another:

    |mu_j / (dmu_j/dtheta_i)| < z sigma_i
    |dmu_j/dtheta_i|          > |mu_j| / (z sigma_i)
    |psi_ji|                  > theta_i / (z sigma_i)      psi = elasticity

The RHS of the third is the t-statistic divided by `z`, so `|psi| > 1` is
NECESSARY whenever `theta_i` is significantly non-zero.

  Part 1  the z-table: `dtheta_i` for the dominant mode and for `s(A)`, the
          distance `|dtheta_i|/sigma_i`, and the verdict.
  Part 2  FEASIBILITY, which is the stronger statement.  WP-2a's column
          identity `1^T A = -l^T` means `s(A)` can reach zero only if the
          loss rate does — i.e. only if the loop closes completely.  So each
          bifurcation value is tested against the FEASIBLE parameter set, not
          only against the CI.  Infeasible ones are STRUCTURALLY UNREACHABLE.
  Part 3  Gandolfo's `dtheta` recast as `dt`: the year at which each
          coefficient reaches its bifurcation value by CONTINUATION of the
          estimated trend, under WP-2d's four rules.
  Part 4  the oscillation boundary of WP-2b, recast in the same units.  A
          node->focus transition, NOT a Hopf bifurcation and NOT a change of
          stability.

  Part 5  the standard error of the characteristic ROOT, which is Gandolfo &
          Padoan's own stability test and a different question from Parts 1-4:
          not "how far would the parameters have to move" but "is the root
          resolved at all".  Needs the full parameter covariance, which WP-4c
          supplies; the simplex shares are exactly anti-correlated and
          marginals alone give the wrong answer.

**`sigma` is a pluggable input, and which provider is used changes the
reading.**  The default is the across-seed spread, which WP-1c flag 1
establishes is a LOWER BOUND on the parameter variance (initialisation and
optimisation path only, same data, same split), so the criterion is
ANTI-CONSERVATIVE under it and UNDER-declares structural instability.
Gandolfo's `sigma_i` is explicitly "the asymptotic standard error of the
parameter", which is what WP-4c's Laplace marginal is and what the seed spread
is not; `--sigma laplace` and `--sigma combined` substitute them from
`analysis/wp4c_sigma.npz`.

`Gandolfo, 1992.pdf` and `Gandolfo & Padoan, 1990.pdf` are now BOTH in the
repository (they were not when this package was first written).  The
implementation has been checked against Gandolfo (1992) §2 and against the
worked example on p. 45 / Appendix 3 of the 1990 paper, which
`zinc_circ_lab.verify_gandolfo_paper` reproduces to the paper's own reporting
precision; see `wp2f_validation.csv`.

    python run_wp2f.py --check
    python run_wp2f.py
    python run_wp2f.py --sigma laplace   --tag laplace
    python run_wp2f.py --sigma combined  --tag combined
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
WP2C_NPZ = os.path.join(OUT_DIR, "wp2c_eigen_sensitivity.npz")
WP2C_SIMPLEX = os.path.join(OUT_DIR, "wp2c_simplex_directions.csv")
WP2B_OSC = os.path.join(OUT_DIR, "wp2b_oscillation_scan.csv")

STATE = list(L.ODE_STATE_NAMES)
COL = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00", "#56B4E9"]
GREY = "#555555"

# WP-2c's near-degeneracy guard, carried through unchanged (42 of 8 400
# mode-nodes; zero of them dominant).
RELGAP_MIN = 0.05
COND_MAX = 10.0

# Part 3: the future grid.  200 years at half-year resolution — long enough
# that anything not reached inside it is genuinely not reached under the rule,
# which is the answer the spec asks to be reported rather than extrapolated.
FUTURE_H = 200.0
FUTURE_DT = 0.5
ETA_WP2A = 0.0115                     # /yr, WP-2a's uniform column certificate

# sigma below this (relative to |theta|) counts as "no across-seed spread":
# the parameter is pinned to data or fixed by construction, so the seed
# ensemble carries no information about its uncertainty at all.
SIGMA_ZERO_REL = 1e-12


def hl(v):
    return L.hodges_lehmann(np.asarray(v, float))


def load():
    for p in (A_NPZ, WP2C_NPZ):
        if not os.path.exists(p):
            raise SystemExit(f"{p} not found — run run_wp2a.py / run_wp2c.py first")
    dA = np.load(A_NPZ, allow_pickle=True)
    dC = np.load(WP2C_NPZ, allow_pickle=True)
    md5A = hashlib.md5(open(A_NPZ, "rb").read()).hexdigest()
    md5C = hashlib.md5(open(WP2C_NPZ, "rb").read()).hexdigest()
    return dA, dC, md5A, md5C


WP4C_SIGMA = os.path.join(OUT_DIR, "wp4c_sigma.npz")


def sigma_seed_provider(P):
    """Default `sigma` provider: the across-seed spread, broadcast back over
    seeds.  Any (n_seed, n_year, n_p) array with the same shape can replace
    it — that is the whole point of keeping it a separate function."""
    s = L.seed_sigma(P)                                  # (n_year, n_p)
    return np.broadcast_to(s, P.shape).copy(), "seed-ensemble spread (PROTOTYPE)"


def sigma_wp4c_provider(P, which, path=WP4C_SIGMA):
    """`sigma` from WP-4c: the Laplace marginal, or the WP-8h combination.

    Gandolfo's criterion is defined on "the asymptotic standard error of the
    parameter".  The Laplace marginal IS that object for this estimator; the
    seed spread is not, and the substitution is the point of having kept the
    provider pluggable.  Both are reported so the reader can see how much of
    the verdict depends on which uncertainty is being asked about.
    """
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found — run `python run_wp4c.py` first")
    d = np.load(path, allow_pickle=True)
    key = {"laplace": "sigma_laplace", "combined": "sigma_combined"}[which]
    sig = np.asarray(d[key], float)
    if [str(x) for x in d["param_names"]] != list(L.PARAM_NAMES):
        raise SystemExit("wp4c_sigma.npz param_names do not match "
                         "zinc_circ_lab.PARAM_NAMES — refusing to align by "
                         "position (CLAUDE.md rule 2).")
    if sig.shape != P.shape:
        raise SystemExit(f"sigma shape {sig.shape} != params {P.shape}")
    label = {"laplace": "WP-4c Laplace marginal at lambda/lambda_1 = %g "
                        "(asymptotic standard error — Gandolfo's own object)"
                        % float(d["lam_ref"]),
             "combined": "WP-8h product: Laplace marginal and across-seed "
                         "spread in quadrature"}[which]
    return sig, label


def load_wp4c_cov(path=WP4C_SIGMA, key="cov_laplace"):
    """Parameter covariance from WP-4c, for Part 5.  None if unavailable."""
    if not os.path.exists(path):
        return None, None
    d = np.load(path, allow_pickle=True)
    if key not in d.files:
        return None, None
    return np.asarray(d[key], float), str(d["note"])


def zero_sigma_mask(sigma, P):
    """Parameters with no across-seed spread at all — pinned to data or fixed
    by construction.  They have no seed `sigma`, so Gandolfo's criterion is
    undefined for them and Part 1 must exclude them (Part 2 still applies)."""
    scale = np.maximum(np.abs(P).max(axis=(0, 1)), 1.0)
    return sigma.max(axis=(0, 1)) < SIGMA_ZERO_REL * scale


# ===========================================================================
# Part 1
# ===========================================================================
def part1(P, lam, dlam, dominant, sigma, sigma_label, flag_bad, years, seeds):
    """The z-table, for the dominant mode and for `s(A)`."""
    n_s, n_y, n_p = P.shape
    dom_lam = np.take_along_axis(lam, dominant[..., None], axis=-1)[..., 0]
    dom_dl = np.take_along_axis(
        dlam, dominant[..., None, None].repeat(n_p, -1), axis=-2)[..., 0, :]
    sabs_idx = np.argmax(lam, axis=-1)                   # s(A) = max Re lambda
    s_lam = np.take_along_axis(lam, sabs_idx[..., None], axis=-1)[..., 0]
    s_dl = np.take_along_axis(
        dlam, sabs_idx[..., None, None].repeat(n_p, -1), axis=-2)[..., 0, :]
    same = bool(np.all(sabs_idx == dominant))

    out = {}
    for tag, mu, dmu in (("dominant", dom_lam, dom_dl), ("s(A)", s_lam, s_dl)):
        out[tag] = L.gandolfo(mu, dmu, P, sigma)
        out[tag]["mu"] = mu
        out[tag]["dmu"] = dmu
    zero = zero_sigma_mask(sigma, P)

    rows = []
    for tag in ("dominant", "s(A)"):
        g = out[tag]
        for i_p, pname in enumerate(L.PARAM_NAMES):
            for i_y in (0, n_y - 1):
                d = g["dtheta"][:, i_y, i_p]
                bv = g["bif_value"][:, i_y, i_p]
                ds = g["dist_sigma"][:, i_y, i_p]
                fin = np.isfinite(d)
                rows.append(dict(
                    root=tag, param=pname,
                    ch2_label=L.CH2_LABEL.get(pname, ""),
                    year=float(years[i_y]),
                    has_seed_sigma=bool(not zero[i_p]),
                    theta_median=float(np.median(P[:, i_y, i_p])),
                    sigma_median=float(np.median(sigma[:, i_y, i_p])),
                    mu_median=float(np.median(g["mu"][:, i_y])),
                    dmu_dtheta_median=float(np.median(g["dmu"][:, i_y, i_p])),
                    psi_median=float(np.median(g["psi"][:, i_y, i_p])),
                    t_stat_median=float(np.median(g["t_stat"][:, i_y, i_p]))
                    if not zero[i_p] else np.inf,
                    dtheta_median=float(np.median(d[fin])) if fin.any() else np.inf,
                    bif_value_median=float(np.median(bv[fin]))
                    if fin.any() else np.inf,
                    dist_sigma_median=float(np.median(ds[fin]))
                    if fin.any() else np.inf,
                    dist_sigma_min=float(np.min(ds[fin])) if fin.any() else np.inf,
                    # Gandolfo p.45: "our criterion is not meant to be applied
                    # mechanically ... if one considers 99% or higher
                    # confidence intervals, one will include a greater number
                    # of partial derivatives".  `z_star` is the confidence
                    # abscissa at which this parameter would FIRST be declared
                    # unstable, which collapses that whole sensitivity into one
                    # number: z* = |dtheta|/sigma.
                    z_star_median=float(np.median(
                        L.z_required(d[fin], sigma[fin, i_y, i_p])))
                    if fin.any() and not zero[i_p] else np.inf,
                    z_star_min=float(np.min(
                        L.z_required(d[fin], sigma[fin, i_y, i_p])))
                    if fin.any() and not zero[i_p] else np.inf,
                    **{f"n_seeds_unstable_at_{lbl}":
                       (int(np.sum(np.abs(d) < zz * sigma[:, i_y, i_p]))
                        if not zero[i_p] else 0)
                       for lbl, zz in L.Z_LEVELS.items()},
                    n_seeds_unstable=int(np.sum(g["unstable_form1"][:, i_y, i_p]))
                    if not zero[i_p] else 0,
                    verdict=("no seed sigma — see Part 2" if zero[i_p]
                             else ("STRUCTURALLY UNSTABLE"
                                   if np.any(g["unstable_form1"][:, i_y, i_p])
                                   else "structurally stable")),
                    n_flagged_nodes=int(flag_bad[tag][:, i_y].sum())))
    return pd.DataFrame(rows), out, zero, same


# ===========================================================================
# Part 2
# ===========================================================================
def bif_residual(P_year, bif_year, i_p, feas_mask):
    """`s(A)` re-evaluated with `theta_i` set to its bifurcation value.

    Gandolfo's `dtheta` is a FIRST-ORDER device: it linearises `mu_j` in
    `theta_i`.  Where `mu_j(theta_i)` is genuinely nonlinear the "bifurcation
    value" is not where the root actually reaches zero, and a verdict of
    REACHABLE can be an artefact of the linearisation rather than a statement
    about the system.  This is not hypothetical — it is exactly what happens
    for the cohort lifetimes, where `lambda ~ -1/mu` and the root approaches
    zero only as `mu -> inf`.  So every FEASIBLE bifurcation value is
    substituted back and the residual reported next to the verdict.

    Outside the feasible set the reassembled `A` is not Metzler (negative
    rates, `tau > 1`) and its spectral abscissa is not a statement about any
    cycle, so it is returned as NaN rather than as a number inviting a
    reading it cannot support.
    """
    P2 = np.array(P_year, float, copy=True)
    P2[:, i_p] = bif_year
    ok = np.isfinite(bif_year) & np.asarray(feas_mask, bool)
    out = np.full(P2.shape[0], np.nan)
    if ok.any():
        out[ok] = L.spectral_abscissa(L.assemble_from_params(P2[ok]))
    return out


def part2(P, g, zero, years, seeds, simplex_csv=WP2C_SIMPLEX):
    """Feasibility of every bifurcation value, plus the simplex directions."""
    rows = []
    for tag in ("dominant", "s(A)"):
        bv = g[tag]["bif_value"]                          # (s, y, p)
        feas = L.feasible(bv)
        for i_p, pname in enumerate(L.PARAM_NAMES):
            for i_y in (0, len(years) - 1):
                f = feas[:, i_y, i_p]
                b = bv[:, i_y, i_p]
                fin = np.isfinite(b)
                res = bif_residual(P[:, i_y], b, i_p, f)
                mu0 = g[tag]["mu"][:, i_y]
                rows.append(dict(
                    root=tag, param=pname, year=float(years[i_y]),
                    has_seed_sigma=bool(not zero[i_p]),
                    theta_median=float(np.median(P[:, i_y, i_p])),
                    feasible_lo=float(L.FEASIBLE_LO[i_p]),
                    feasible_hi=float(L.FEASIBLE_HI[i_p]),
                    bif_value_median=float(np.median(b[fin]))
                    if fin.any() else np.inf,
                    n_seeds_feasible=int(f.sum()),
                    n_seeds=int(f.size),
                    s_at_bif_value_median=float(np.nanmedian(res))
                    if np.isfinite(res).any() else np.nan,
                    s_at_estimate_median=float(np.median(mu0)),
                    linearisation_residual=float(
                        np.nanmedian(np.abs(res)) / np.abs(np.median(mu0)))
                    if np.isfinite(res).any() else np.nan,
                    verdict=("REACHABLE (feasible bifurcation value)" if f.all()
                             else ("structurally unreachable" if not f.any()
                                   else "mixed across seeds"))))
    feas_df = pd.DataFrame(rows)

    # Ch2's constrained simplex directions: dlambda/dtheta = dl/dw_r - dl/dw_q
    simp_rows = []
    for name, ia, ib, label in L.simplex_directions():
        for tag in ("dominant", "s(A)"):
            dmu = g[tag]["dmu"][..., ia] - g[tag]["dmu"][..., ib]
            mu = g[tag]["mu"]
            with np.errstate(divide="ignore", invalid="ignore"):
                dth = np.where(np.abs(dmu) > 0, -mu / dmu, np.inf)
            # the direction moves mass from q to r, so the constrained
            # coordinate is theta = w_r with w_q = (w_r + w_q) - theta
            th = P[..., ia]
            cap = P[..., ia] + P[..., ib]                  # w_r + w_q, the cap
            bv = th + dth
            ok = np.isfinite(bv) & (bv >= 0.0) & (bv <= cap)
            for i_y in (0, len(years) - 1):
                d = dth[:, i_y]
                fin = np.isfinite(d)
                simp_rows.append(dict(
                    root=tag, simplex=name, direction=label,
                    year=float(years[i_y]),
                    dtheta_median=float(np.median(d[fin])) if fin.any() else np.inf,
                    theta_median=float(np.median(th[:, i_y])),
                    cap_median=float(np.median(cap[:, i_y])),
                    n_seeds_feasible=int(ok[:, i_y].sum()),
                    verdict=("REACHABLE" if ok[:, i_y].all()
                             else ("structurally unreachable"
                                   if not ok[:, i_y].any() else "mixed"))))
    return feas_df, pd.DataFrame(simp_rows)


# ===========================================================================
# Part 3
# ===========================================================================
def part3(P, g, years, seeds, rules=L.RULES):
    """Gandolfo's `dtheta` recast as `dt`.

    The bifurcation reachable by PERTURBATION is likely unreachable (Part 2);
    the one reachable by CONTINUATION of the estimated trend is a different
    question.  Each coefficient path is extrapolated forward under WP-2d's
    four rules and asked when it reaches the bifurcation value computed at the
    last node.

    Two readings are reported and they are not the same thing:

      `param_crossing_yr`  when `theta_i(t)` reaches its own 2019 bifurcation
                           value, holding the other coefficients on the same
                           extrapolated path;
      `system_bif_yr`      when `s(A(t))` along the extrapolated path actually
                           reaches zero — the bifurcation of the MOVING
                           system, which is what WP-2d saw break under
                           `trend50`.

    `coeff_path` clips the extrapolation to the feasible set at every step
    (that is what keeps the extrapolated `A` Metzler), so a bifurcation value
    outside the feasible set can never be crossed under any rule — which is
    Part 2's conclusion arriving again by a different route, and is reported
    as such rather than as a missing number.
    """
    t = years[-1] + np.arange(0.0, FUTURE_H + 1e-9, FUTURE_DT)
    bif = g["dominant"]["bif_value"][:, -1, :]            # (s, p)
    rows, paths = [], {}
    for rule in rules:
        Pt, t_switch = L.coeff_path(P, years, t, rule=rule)
        At = L.assemble_from_params(Pt)
        s_of_t = L.spectral_abscissa(At)                  # (s, n_t)
        cross = L.first_crossing(np.moveaxis(Pt, -1, 1), bif, t)   # (s, p)
        sysbif = L.first_crossing(s_of_t, np.zeros(P.shape[0]) - 1e-12, t)
        paths[rule] = dict(t=t, P=Pt, s=s_of_t, t_switch=t_switch)
        for i_p, pname in enumerate(L.PARAM_NAMES):
            v = cross[:, i_p]
            fin = np.isfinite(v)
            rows.append(dict(
                rule=rule, param=pname, quantity="param_crossing",
                t_switch=float(t_switch),
                bif_value_median=float(np.median(bif[np.isfinite(bif[:, i_p]),
                                                     i_p]))
                if np.isfinite(bif[:, i_p]).any() else np.inf,
                theta_2019_median=float(np.median(P[:, -1, i_p])),
                n_reached=int(fin.sum()), n_seeds=int(v.size),
                year_median=float(np.median(v[fin])) if fin.any() else np.nan,
                year_q1=float(np.percentile(v[fin], 25)) if fin.any() else np.nan,
                year_q3=float(np.percentile(v[fin], 75)) if fin.any() else np.nan,
                seeds_not_reaching=json.dumps(
                    [int(s) for s in np.asarray(seeds)[~fin]])))
        fin = np.isfinite(sysbif)
        rows.append(dict(
            rule=rule, param="s(A) = 0", quantity="system_bifurcation",
            t_switch=float(t_switch), bif_value_median=0.0,
            theta_2019_median=float(np.median(L.spectral_abscissa(
                L.assemble_from_params(P[:, -1])))),
            n_reached=int(fin.sum()), n_seeds=int(sysbif.size),
            year_median=float(np.median(sysbif[fin])) if fin.any() else np.nan,
            year_q1=float(np.percentile(sysbif[fin], 25)) if fin.any() else np.nan,
            year_q3=float(np.percentile(sysbif[fin], 75)) if fin.any() else np.nan,
            seeds_not_reaching=json.dumps(
                [int(s) for s in np.asarray(seeds)[~fin]])))
    return pd.DataFrame(rows), paths


# ===========================================================================
# Part 4
# ===========================================================================
def part4(A, P, sigma, years, seeds, osc_csv=WP2B_OSC):
    """The oscillation boundary of WP-2b, in Part 1-3 units.

    WP-2b scaled each structurally non-zero entry of `A` over 1e-6..1e+6 with
    the same column's diagonal absorbing the change (so `l_j` is unchanged),
    and found that only the three end-of-life routes `A[5,k] = tau_olds/mu_k`
    reach a complex pair at all, needing x158-631 at best while mass balance
    caps them at x5.2 (`A[5,k] <= -A[k,k]`, i.e. `tau_olds <= 1`).

    Two readings, because the scan is defined on ENTRIES and policy acts on
    PARAMETERS, and they are not interchangeable: scaling `A[5,k]` alone with
    the diagonal compensating is not achievable by moving `tau_olds`, which
    would move `l` too.  Both are infeasible, which is the finding.

    **This is a node->focus transition, NOT a Hopf bifurcation.**  Every
    eigenvalue stays in the open left half-plane throughout; what changes is
    whether the approach to equilibrium is monotone or oscillatory.  Stability
    is not at issue anywhere in this part.
    """
    osc = pd.read_csv(osc_csv)
    osc = osc[osc.reaches_complex.astype(bool)]
    i_tau = L.PARAM_NAMES.index("tau_olds")
    sig_tau = sigma[:, :, i_tau]
    rows = []
    for _, r in osc.iterrows():
        i, j = int(r.row), int(r.col)
        ent = A[:, :, i, j]                               # (s, y)
        sig_ent = np.std(ent, axis=0, ddof=1)             # across-seed, per year
        for f_tag, f in (("best case (crit_up_min)", float(r.crit_up_min)),
                         ("typical (crit_up_median)", float(r.crit_up_median))):
            for i_y in (0, len(years) - 1):
                d_ent = ent[:, i_y] * (f - 1.0)
                d_tau = P[:, i_y, i_tau] * (f - 1.0)
                rows.append(dict(
                    entry=f"A[{i},{j}]", to_stock=r.to_stock,
                    from_stock=r.from_stock, case=f_tag, multiplier=f,
                    year=float(years[i_y]),
                    entry_median=float(np.median(ent[:, i_y])),
                    entry_sigma=float(sig_ent[i_y]),
                    d_entry_median=float(np.median(d_ent)),
                    dist_sigma_entry=float(np.median(d_ent) / sig_ent[i_y]),
                    tau_olds_median=float(np.median(P[:, i_y, i_tau])),
                    tau_olds_sigma=float(np.median(sig_tau[:, i_y])),
                    d_tau_olds_median=float(np.median(d_tau)),
                    dist_sigma_tau_olds=float(
                        np.median(d_tau) / np.median(sig_tau[:, i_y])),
                    tau_olds_bif_value=float(np.median(
                        P[:, i_y, i_tau] * f)),
                    max_feasible_multiplier=float(r.max_feasible_multiplier),
                    shortfall=f / float(r.max_feasible_multiplier),
                    feasible=bool(np.median(P[:, i_y, i_tau] * f) <= 1.0),
                    transition="node -> focus (NOT a Hopf bifurcation; "
                               "no change of stability)"))
    return pd.DataFrame(rows)


# ===========================================================================
# figure
# ===========================================================================
# ===========================================================================
# Part 5 — the standard error of the root itself (Gandolfo & Padoan 1990)
# ===========================================================================
def part5(P, lam, dlam, dominant, covs, years, seeds, flag_bad):
    """Gandolfo & Padoan report the characteristic roots "together with their
    asymptotic standard errors" and conclude that the Italian model is stable
    because its one positive root "is not significantly different from zero at
    the 5% level".  That is a different test from Parts 1-4 and it is the one
    an econometrician would run first.

    `sigma(mu_j) = sqrt( (dmu_j/dp)^T Cov(p) (dmu_j/dp) )` — the delta method
    through Ch2 Eq. `eigenvalue_sensitivity`.  It needs the FULL covariance:
    `frac_fu_new` and `frac_fu_out` are exactly anti-correlated by their
    simplex, so treating the marginals as independent inflates the variance of
    any root that moves along that direction.  Both are computed and the gap
    between them is reported, because it is the whole reason the covariance
    was carried through WP-4c.
    """
    n_s, n_y, n_p = P.shape
    dom_lam = np.take_along_axis(lam, dominant[..., None], axis=-1)[..., 0]
    dom_dl = np.take_along_axis(
        dlam, dominant[..., None, None].repeat(n_p, -1), axis=-2)[..., 0, :]
    sabs_idx = np.argmax(lam, axis=-1)
    s_lam = np.take_along_axis(lam, sabs_idx[..., None], axis=-1)[..., 0]
    s_dl = np.take_along_axis(
        dlam, sabs_idx[..., None, None].repeat(n_p, -1), axis=-2)[..., 0, :]

    rows = []
    for (cov_label, cov), (tag, mu, dmu) in [
            (c, r) for c in covs.items()
            for r in (("dominant", dom_lam, dom_dl), ("s(A)", s_lam, s_dl))]:
        if cov.ndim == 3:                       # (y, p, p) -> broadcast seeds
            cov = np.broadcast_to(cov[None], (n_s,) + cov.shape)
        sig_full = L.root_sigma(dmu, cov)                       # (s, y)
        diag = cov * np.eye(n_p)[None, None, :, :]
        sig_diag = L.root_sigma(dmu, diag)
        for i_y in range(n_y):
            m = mu[:, i_y]
            sf, sd_ = sig_full[:, i_y], sig_diag[:, i_y]
            with np.errstate(divide="ignore", invalid="ignore"):
                tstat = np.abs(m) / sf
            hlr = L.hodges_lehmann(tstat[np.isfinite(tstat)])
            rows.append(dict(
                root=tag, cov_source=cov_label, year=float(years[i_y]),
                mu_median=float(np.median(m)),
                sigma_mu_median=float(np.median(sf)),
                sigma_mu_q1=float(np.percentile(sf, 25)),
                sigma_mu_q3=float(np.percentile(sf, 75)),
                sigma_mu_diagonal_only_median=float(np.median(sd_)),
                covariance_effect=float(np.median(sf) / max(np.median(sd_), 1e-300)),
                t_stat_median=float(np.median(tstat)),
                t_stat_hl=hlr["hl"], t_stat_hl_lo=hlr["lo"],
                t_stat_hl_hi=hlr["hi"],
                n_seeds_root_not_significant=int(np.sum(tstat < L.Z_95)),
                n_seeds=int(n_s),
                verdict=("root NOT significantly different from zero "
                         "at 5% — stability not established"
                         if np.median(tstat) < L.Z_95
                         else "root significantly negative at 5% — "
                              "asymptotic stability established"),
                n_flagged_nodes=int(flag_bad[tag][:, i_y].sum())))
    return pd.DataFrame(rows)


def fig_wp2f(z_tab, feas, t3, p4, paths, years, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2))
    y_end = float(years[-1])

    # (a) distance to bifurcation in sigma, dominant mode
    ax = axes[0, 0]
    s = z_tab[(z_tab.root == "dominant") & (z_tab.year == y_end)
              & z_tab.has_seed_sigma].copy()
    s = s[np.isfinite(s.dist_sigma_median)]
    s = s.sort_values("dist_sigma_median")
    y = np.arange(len(s))
    ax.barh(y, s.dist_sigma_median, color=COL[0], height=0.6)
    for i, (_, r) in enumerate(s.iterrows()):
        ax.plot([r.dist_sigma_min, r.dist_sigma_median], [i, i], color="k", lw=1.0)
    ax.axvline(L.Z_95, color=COL[2], lw=1.6, ls="--",
               label=f"$z_{{0.025}}$ = {L.Z_95:.2f}  (Gandolfo's threshold)")
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(s.param, fontsize=7.5)
    ax.set_xlabel("$|d\\theta_i|\\,/\\,\\sigma_i$   (prototype $\\sigma$)")
    ax.set_title(f"(a) distance to the bifurcation value, dominant mode, "
                 f"{int(y_end)}\n     left of the dashed line = "
                 f"structurally unstable", loc="left", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")

    # (b) feasibility of the bifurcation value
    ax = axes[0, 1]
    s = feas[(feas.root == "dominant") & (feas.year == y_end)].copy()
    s = s[np.isfinite(s.bif_value_median)]
    s = s.sort_values("param")
    y = np.arange(len(s))
    for i, (_, r) in enumerate(s.iterrows()):
        lo, hi = r.feasible_lo, min(r.feasible_hi, 3.0)
        ax.plot([lo, hi], [i, i], color=GREY, lw=5, solid_capstyle="butt",
                alpha=0.35)
        ax.plot(r.theta_median, i, "o", color=COL[1], ms=5)
        raw = r.bif_value_median
        bv = np.clip(raw, -0.6, 3.4)
        # tri-state: a value feasible at 2 of 35 seeds is not "reachable"
        c = (COL[2] if r.n_seeds_feasible == 0
             else (COL[4] if r.n_seeds_feasible < r.n_seeds else COL[0]))
        ax.plot(bv, i, "X", color=c, ms=7)
        # An off-scale bifurcation value must not read as if it sat at the
        # axis limit: annotate the clipped ones with their actual value.
        if not np.isclose(bv, raw):
            ax.annotate(("+inf" if not np.isfinite(raw) else f"{raw:,.0f}"),
                        (bv, i), textcoords="offset points",
                        xytext=(-4 if bv > 0 else 8, 0), fontsize=6,
                        color=c, va="center",
                        ha="right" if bv > 0 else "left")
    ax.set_yticks(y)
    ax.set_yticklabels(s.param, fontsize=7.5)
    ax.set_xlim(-0.7, 3.5)
    ax.set_xlabel("parameter value")
    ax.set_title("(b) is the bifurcation value even feasible?\n"
                 "     grey = feasible set · ● estimate · ✕ bifurcation value "
                 "(red = unreachable at every seed, amber = at some)",
                 loc="left", fontsize=9.5)

    # (c) spectral abscissa along the extrapolated paths
    ax = axes[1, 0]
    for r, rule in enumerate(L.RULES):
        p = paths[rule]
        lo, med, hi = np.percentile(p["s"], [25, 50, 75], axis=0)
        ax.fill_between(p["t"], lo, hi, color=COL[r], alpha=0.14, lw=0)
        ax.plot(p["t"], med, color=COL[r], lw=1.7, label=rule)
    ax.axhline(0.0, color="k", lw=1.2)
    ax.axhline(-ETA_WP2A, color=GREY, lw=1.0, ls="--",
               label=f"WP-2a certificate $-\\eta$ = {-ETA_WP2A:.4f}/yr")
    ax.set_xlabel("year")
    ax.set_ylabel("$s(A(t))$   (/yr)")
    ax.set_title("(c) Part 3 — does continuation of the estimated trend "
                 "reach\n     the stability boundary?", loc="left", fontsize=9.5)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="lower right")

    # (d) the oscillation boundary, in sigma
    ax = axes[1, 1]
    s = p4[(p4.year == y_end)].copy()
    lbl, vals, cols = [], [], []
    for case, c in (("best case (crit_up_min)", COL[0]),
                    ("typical (crit_up_median)", COL[4])):
        sub = s[s.case == case]
        for _, r in sub.iterrows():
            lbl.append(f"{r.entry} {r.case.split()[0]}")
            vals.append(r.dist_sigma_entry)
            cols.append(c)
    y = np.arange(len(vals))
    ax.barh(y, vals, color=cols, height=0.62)
    ax.axvline(L.Z_95, color=COL[2], lw=1.6, ls="--", label="$z_{0.025}$")
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(lbl, fontsize=7)
    ax.set_xlabel("distance to the real→complex boundary  ($\\sigma$ of the "
                  "entry)")
    ax.set_title("(d) Part 4 — the oscillation boundary\n"
                 "     node→focus, NOT a Hopf bifurcation, NOT a change of "
                 "stability", loc="left", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False)

    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(lw=0.4, alpha=0.3)
    fig.suptitle("WP-2f  structural stability of the estimated zinc cycle "
                 "(Gandolfo 1992 §2) — distance to bifurcation in estimation "
                 "error and in feasibility\n"
                 "anchor_v4, 35 seeds; $\\sigma$ = across-seed spread, a "
                 "PROTOTYPE and a lower bound, so the criterion "
                 "under-declares instability", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=300)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


# ===========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--sigma", default="seed",
                    choices=("seed", "laplace", "combined"),
                    help="which standard-error provider Gandolfo's criterion "
                         "uses (default: the across-seed prototype)")
    ap.add_argument("--tag", default="",
                    help="suffix for every output file, so a re-run under a "
                         "different sigma does not overwrite the published one")
    args = ap.parse_args(argv)
    TAG = ("_" + args.tag) if args.tag else ""
    out = lambda stem, ext: os.path.join(args.out, f"{stem}{TAG}.{ext}")

    info = L.check(verbose=True)
    if args.check:
        print("zinc_circ_lab --check  (WP-2f additions)")
        print("=" * 74)
        print(f"  Gandolfo threshold z : {L.Z_95:.6f}  (two-sided 95%)")
        print(f"  sigma provider       : across-seed spread — a PROTOTYPE and "
              f"a LOWER bound (WP-1c flag 1);")
        print( "                         the criterion is therefore "
               "ANTI-CONSERVATIVE and under-declares instability")
        print(f"  feasible set         : "
              + ", ".join(f"{n} in [{lo:g},{hi:g}]" for n, lo, hi in
                          list(zip(L.PARAM_NAMES, L.FEASIBLE_LO,
                                   L.FEASIBLE_HI))[:8]) + ", …")
        gp = L.verify_gandolfo_paper()
        print(f"  Gandolfo, 1992.pdf   : PRESENT.  §2 read and the worked "
              f"example on p.45 reproduced:")
        print(f"                         theta={gp['theta']:.4f} "
              f"sigma={gp['sigma']:.5f}  bif={gp['bif_value']:.4f} "
              f"(paper 1.1400)")
        print(f"                         elasticity {gp['elasticity']:.4f} "
              f"(paper 3.22, rel {gp['rel_err_elasticity']:.1e})  "
              f"RHS(iv) {gp['rhs_form3']:.4f} (paper 3.12)")
        print(f"                         verdict reproduced: "
              f"{gp['verdict_matches']}   forms agree: {gp['forms_agree']}")
        print(f"  z levels reported    : {L.Z_LEVELS}")
        print(f"  sigma providers      : seed | laplace | combined "
              f"(--sigma; laplace/combined read {os.path.basename(WP4C_SIGMA)})")
        print("=" * 74)
        return 0

    dA, dC, md5A, md5C = load()
    A = np.asarray(dA["A"], float)
    years = np.asarray(dA["years"], float)
    seeds = np.asarray(dA["seeds"], int)
    P = np.asarray(dC["params"], float)
    lam = np.asarray(dC["lam_state"]).real
    dlam = np.asarray(dC["dlambda_state"], float)
    dominant = np.asarray(dC["dominant_mode"], int)
    relgap = np.asarray(dC["relgap_state"], float)
    cond = np.asarray(dC["cond_state"], float)
    os.makedirs(args.out, exist_ok=True)
    print(f"\nsources: wp2a_A_of_t.npz md5 {md5A[:8]}…  "
          f"wp2c_eigen_sensitivity.npz md5 {md5C[:8]}…")
    assert float(np.max(np.abs(np.asarray(dC["params"], float)
                               - L.pack_params(dA, dA["mu_cohorts"])))) == 0.0

    if args.sigma == "seed":
        sigma, sigma_label = sigma_seed_provider(P)
    else:
        sigma, sigma_label = sigma_wp4c_provider(P, args.sigma)
    covs = {}
    for lbl, key in (("laplace", "cov_laplace"),
                     ("rubin_total", "cov_rubin_total")):
        c, _n = load_wp4c_cov(key=key)
        if c is not None:
            covs[lbl] = c
    bad = (relgap < RELGAP_MIN) | (cond > COND_MAX)      # WP-2c's guard
    sabs_idx = np.argmax(lam, axis=-1)
    flag_bad = {
        "dominant": np.take_along_axis(bad, dominant[..., None], -1)[..., 0],
        "s(A)": np.take_along_axis(bad, sabs_idx[..., None], -1)[..., 0]}

    validation = []
    # --- the criterion against the paper it comes from ---------------------
    # Gandolfo (1992) p.45 / Gandolfo & Padoan (1990) App. 3 work one example
    # end to end.  Reproducing it is the only external check available on this
    # implementation, and it is the first thing to run now that both PDFs are
    # in the repository.
    gp = L.verify_gandolfo_paper()
    validation.append(dict(
        part=0, check="Gandolfo (1992) p.45 worked example — verdict "
                      "(structurally unstable) reproduced",
        result=float(not gp["verdict_matches"]), exact_answer=0.0,
        note="external check against the source paper"))
    validation.append(dict(
        part=0, check="Gandolfo (1992) p.45 worked example — elasticity "
                      f"{gp['elasticity']:.4f} vs the paper's 3.22 (relative)",
        result=gp["rel_err_elasticity"], exact_answer=0.0,
        note="the paper reports 3 s.f., so agreement is expected at 1e-3"))
    validation.append(dict(
        part=0, check="Gandolfo (1992) p.45 worked example — RHS of form (iv) "
                      f"{gp['rhs_form3']:.4f} vs the paper's 3.12 (relative)",
        result=gp["rel_err_rhs"], exact_answer=0.0,
        note="the paper reports 3 s.f."))
    validation.append(dict(
        part=0, check="criterion is invariant to the value of mu (it cancels "
                      "in all three forms)",
        result=gp["invariance_to_mu"], exact_answer=0.0,
        note="numerical check"))
    # --- derivative chain, as WP-2c did -----------------------------------
    vd = L.verify_derivatives(P[:3, [0, -1]])
    validation.append(dict(
        part=0, check="dA/dp (complex step) vs central difference",
        result=vd["max_rel_err_dA"], exact_answer=0.0,
        note="numerical check"))
    validation.append(dict(
        part=0, check="dlambda_k/dp_n vs central difference of eig(A(p+-h)) "
                      "— WP-2c reached 8.5e-08",
        result=vd["max_rel_err_dlambda"], exact_answer=0.0,
        note="numerical check"))
    print(f"  dA/dp vs central difference        : {vd['max_rel_err_dA']:.2e}")
    print(f"  dlambda/dp vs central difference   : "
          f"{vd['max_rel_err_dlambda']:.2e}   (WP-2c: 8.5e-08)")

    # ---- Part 1 ------------------------------------------------------------
    z_tab, g, zero, dom_is_sabs = part1(P, lam, dlam, dominant, sigma,
                                        sigma_label, flag_bad, years, seeds)
    for tag in ("dominant", "s(A)"):
        validation.append(dict(
            part=1, check=f"Gandolfo's three forms agree ({tag} root)",
            result=float(not g[tag]["forms_agree"]), exact_answer=0.0,
            note="numerical check"))
    validation.append(dict(
        part=1, check="dominant mode is the spectral abscissa at every node "
                      "(WP-2b: S_iu_long at 1400/1400)",
        result=float(not dom_is_sabs), exact_answer=0.0,
        note="numerical check"))
    z_tab.to_csv(out("wp2f_z_table", "csv"), index=False)

    print("\n" + "=" * 74)
    print("Part 1 — the z-table  (Gandolfo 1992 §2)")
    print("=" * 74)
    print(f"  sigma provider: {sigma_label}")
    print(f"  three forms agree: dominant {g['dominant']['forms_agree']}, "
          f"s(A) {g['s(A)']['forms_agree']}")
    print(f"  dominant mode == argmax Re lambda at every node: {dom_is_sabs}")
    excl = [n for n, z in zip(L.PARAM_NAMES, zero) if z]
    print(f"  excluded — no across-seed sigma ({len(excl)}): {excl}")
    print(f"\n  {'parameter':<16s}{'theta':>10s}{'sigma':>10s}"
          f"{'dtheta':>12s}{'bif value':>12s}{'|dtheta|/sigma':>16s}"
          f"  verdict   ({int(years[-1])}, dominant mode)")
    s = z_tab[(z_tab.root == "dominant") & (z_tab.year == float(years[-1]))
              & z_tab.has_seed_sigma]
    for _, r in s.iterrows():
        ds = ("inf" if not np.isfinite(r.dist_sigma_median)
              else f"{r.dist_sigma_median:,.1f}")
        dt_ = ("inf" if not np.isfinite(r.dtheta_median)
               else f"{r.dtheta_median:+.4f}")
        bv = ("inf" if not np.isfinite(r.bif_value_median)
              else f"{r.bif_value_median:+.4f}")
        print(f"  {r.param:<16s}{r.theta_median:>10.4f}{r.sigma_median:>10.4f}"
              f"{dt_:>12s}{bv:>12s}{ds:>16s}  {r.verdict}")
    n_uns = int((z_tab[z_tab.has_seed_sigma].n_seeds_unstable > 0).sum())
    print(f"\n  parameters declared structurally unstable at any seed: {n_uns}")
    print(f"  near-degeneracy nodes carried from WP-2c "
          f"(relgap < {RELGAP_MIN} or cond > {COND_MAX}): "
          f"{int(bad.sum())} of {bad.size} mode-nodes, "
          f"{int(flag_bad['dominant'].sum())} of them dominant")

    # ---- Part 2 ------------------------------------------------------------
    feas, simp = part2(P, g, zero, years, seeds)
    feas.to_csv(out("wp2f_feasibility", "csv"), index=False)
    _fs = feas[(feas.root == "dominant") & (feas.n_seeds_feasible > 0)]
    validation.append(dict(
        part=2, check="s(A) re-evaluated at every FEASIBLE bifurcation value "
                      "reaches 0 (tests the first-order device itself)",
        result=float(_fs.linearisation_residual.max(skipna=True)),
        exact_answer=0.0,
        note="NOT a code check — a measurement of Gandolfo's first-order "
             "device itself.  A large value means the linearised bifurcation "
             "value is not where the root actually reaches zero, which is a "
             "reportable property of the system (mu_44yr: lambda ~ -1/mu, so "
             "the true bifurcation is at mu -> infinity)."))
    simp.to_csv(out("wp2f_simplex_feasibility", "csv"),
                index=False)
    ell = np.asarray(dA["loss_rate"], float)
    print("\n" + "=" * 74)
    print("Part 2 — feasibility, which is the stronger statement")
    print("=" * 74)
    print(f"  WP-2a column identity 1^T A = -l^T holds, and min_j l_j > 0 at "
          f"every node (min {ell.min():.5f} /yr),")
    print(f"  so s(A) = 0 requires the loop to close COMPLETELY.  "
          f"WP-2a's certificate: eta = {ETA_WP2A} /yr.")
    s = feas[(feas.root == "dominant") & (feas.year == float(years[-1]))]
    n_reach = int((s.n_seeds_feasible > 0).sum())
    print(f"\n  bifurcation values inside the feasible set, {int(years[-1])}, "
          f"dominant mode: {n_reach} of {len(s)} parameters")
    for _, r in s.iterrows():
        bv = ("inf" if not np.isfinite(r.bif_value_median)
              else f"{r.bif_value_median:+.4f}")
        print(f"    {r.param:<16s} theta {r.theta_median:8.4f}   "
              f"feasible [{r.feasible_lo:g}, {r.feasible_hi:g}]   "
              f"bif {bv:>12s}   {r.verdict:<40s}  "
              f"({r.n_seeds_feasible}/{r.n_seeds})  "
              + ("" if not np.isfinite(r.s_at_bif_value_median)
                 else f"s(A) there {r.s_at_bif_value_median:+.5f} "
                      f"({100*r.linearisation_residual:5.1f}% of s(A) remains)"))
    bad_lin = s[(s.n_seeds_feasible > 0)
                & (s.linearisation_residual.fillna(0) > 0.05)]
    if len(bad_lin):
        print(f"\n  !! first-order artefact: for "
              f"{', '.join(bad_lin.param)} the root has NOT reached zero at "
              f"the linearised bifurcation value.")
        print( "     Gandolfo's dtheta linearises mu_j in theta_i; for the "
               "cohort lifetimes lambda ~ -1/mu, so the")
        print( "     true bifurcation is at mu -> infinity and the REACHABLE "
               "verdict is an artefact of the device, not")
        print( "     a property of the cycle.  Reported rather than "
               "suppressed.")
    print(f"\n  constrained simplex directions (Ch2 dl/dw_r - dl/dw_q), "
          f"{int(years[-1])}, dominant mode:")
    for _, r in simp[(simp.root == "dominant")
                     & (simp.year == float(years[-1]))].iterrows():
        dt_ = ("inf" if not np.isfinite(r.dtheta_median)
               else f"{r.dtheta_median:+.4f}")
        print(f"    {r.direction:<40s} dtheta {dt_:>10s}  "
              f"cap {r.cap_median:.4f}   {r.verdict}")

    # ---- Part 3 ------------------------------------------------------------
    t3, paths = part3(P, g, years, seeds)
    t3.to_csv(out("wp2f_bifurcation_years", "csv"), index=False)
    print("\n" + "=" * 74)
    print("Part 3 — Gandolfo's dtheta recast as dt")
    print("=" * 74)
    for rule in L.RULES:
        sub = t3[t3.rule == rule]
        sysr = sub[sub.quantity == "system_bifurcation"].iloc[0]
        reached = sub[(sub.quantity == "param_crossing") & (sub.n_reached > 0)]
        print(f"  rule `{rule}` (frozen beyond {sysr.t_switch:.0f}):")
        print(f"    s(A(t)) reaches 0: {sysr.n_reached}/{sysr.n_seeds} seeds"
              + (f", median {sysr.year_median:.0f} "
                 f"[{sysr.year_q1:.0f}, {sysr.year_q3:.0f}]"
                 if sysr.n_reached else "  — NEVER, within "
                 f"{int(FUTURE_H)} yr"))
        if len(reached):
            for _, r in reached.iterrows():
                print(f"    {r.param:<16s} reaches its bifurcation value at "
                      f"{r.year_median:.0f} "
                      f"[{r.year_q1:.0f},{r.year_q3:.0f}]  "
                      f"({r.n_reached}/{r.n_seeds} seeds)")
        else:
            print( "    no parameter reaches its bifurcation value under this "
                   "rule, at any seed")
        s_end = paths[rule]["s"][:, -1]
        print(f"    s(A) at {years[-1] + FUTURE_H:.0f}: median "
              f"{np.median(s_end):+.5f} /yr "
              f"[{np.percentile(s_end, 25):+.5f}, "
              f"{np.percentile(s_end, 75):+.5f}]")

    # ---- Part 4 ------------------------------------------------------------
    p4 = part4(A, P, sigma, years, seeds)
    p4.to_csv(out("wp2f_oscillation_boundary", "csv"),
              index=False)
    print("\n" + "=" * 74)
    print("Part 4 — the oscillation boundary  "
          "(node -> focus, NOT a Hopf bifurcation)")
    print("=" * 74)
    for _, r in p4[(p4.year == float(years[-1]))
                   & (p4.case == "best case (crit_up_min)")].iterrows():
        print(f"  {r.entry} ({r.from_stock} -> {r.to_stock}): needs "
              f"x{r.multiplier:,.0f}  =  {r.dist_sigma_entry:,.0f} sigma of "
              f"the entry, {r.dist_sigma_tau_olds:,.0f} sigma of tau_olds;")
        print(f"      tau_olds would have to reach "
              f"{r.tau_olds_bif_value:,.1f} (feasible <= 1) — "
              f"{'FEASIBLE' if r.feasible else 'structurally unreachable'}, "
              f"short of the mass-balance cap by "
              f"x{r.shortfall:,.0f}")
    print("  Every eigenvalue remains in the open left half-plane throughout: "
          "this is a change in the")
    print("  APPROACH to equilibrium (monotone -> oscillatory), not a change "
          "of stability.")

    # ---- Part 5 ------------------------------------------------------------
    p5 = None
    print("\n" + "=" * 74)
    print("Part 5 — the standard error of the root itself  "
          "(Gandolfo & Padoan 1990, Tables 4-5)")
    print("=" * 74)
    if not covs:
        print("  SKIPPED — analysis/wp4c_sigma.npz carries no `cov_laplace`.")
        print("  Run `python run_wp4c.py` first; Part 5 needs the FULL "
              "parameter covariance, not the marginals.")
        validation.append(dict(
            part=5, check="root standard error computed", result=1.0,
            exact_answer=0.0, note="SKIPPED — WP-4c covariance not available"))
    else:
        p5 = part5(P, lam, dlam, dominant, covs, years, seeds, flag_bad)
        p5.to_csv(out("wp2f_root_significance", "csv"), index=False)
        for cl in covs:
            for yy in (years[0], years[-1]):
                r = p5[(p5.root == "dominant") & (p5.cov_source == cl)
                       & (p5.year == float(yy))].iloc[0]
                print(f"  dominant {int(yy)}  cov={cl:<12s} "
                      f"mu = {r.mu_median:+.5f} /yr, "
                      f"sigma(mu) = {r.sigma_mu_median:.5f}, "
                      f"|mu|/sigma = {r.t_stat_median:,.1f}")
                print(f"                 {r.verdict}")
        r = p5[p5.root == "dominant"].copy()
        print(f"  covariance vs marginals-only sigma(mu): median ratio "
              f"{r.covariance_effect.median():.3f} "
              f"(1.0 would mean the correlations do not matter)")
        validation.append(dict(
            part=5, check="seed-years at which the dominant root is NOT "
                          "significantly different from zero at 5%, any "
                          "covariance source",
            result=float(p5[p5.root == "dominant"]
                         .n_seeds_root_not_significant.sum()),
            exact_answer=0.0,
            note="Gandolfo & Padoan's own stability test; 0 means the root is "
                 "resolved as negative at every seed and year"))

    # ---- persist -----------------------------------------------------------
    pd.DataFrame(validation).to_csv(
        out("wp2f_validation", "csv"), index=False)
    np.savez_compressed(
        out("wp2f_structural_stability", "npz"),
        years=years, seeds=seeds, params=P, sigma=sigma,
        sigma_label=np.array(sigma_label, object),
        param_names=np.array(L.PARAM_NAMES, object),
        state_names=np.array(STATE, object),
        z=np.array(L.Z_95), feasible_lo=L.FEASIBLE_LO, feasible_hi=L.FEASIBLE_HI,
        sigma_provider=np.array(args.sigma, object),
        z_levels=np.array(json.dumps(L.Z_LEVELS), object),
        gandolfo_paper_check=np.array(json.dumps(gp), object),
        zero_sigma=zero, degeneracy_flag=bad,
        source_md5_wp2a=np.array(md5A), source_md5_wp2c=np.array(md5C),
        **{f"{k}_{tag}": g[tag][k]
           for tag in ("dominant", "s(A)")
           for k in ("dtheta", "bif_value", "dist_sigma", "psi", "t_stat",
                     "unstable_form1", "unstable_form2", "unstable_form3",
                     "mu", "dmu")},
        **{f"path_s_{r}": paths[r]["s"] for r in L.RULES},
        **{f"path_P_{r}": paths[r]["P"] for r in L.RULES},
        future_t=paths[L.RULES[0]]["t"])

    fig_wp2f(z_tab, feas, t3, p4, paths, years,
             out("wp2f_structural_stability", "png"))
    with open(out("wp2f_check", "json"), "w") as fh:
        json.dump({k: v for k, v in info.items() if k != "digests"}, fh,
                  indent=2, default=str)
    print(f"\nwrote wp2f_* to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
