#!/usr/bin/env python3
"""
run_wp8h.py — WP-8h: the coefficient uncertainty product
========================================================

The spec calls this "the artifact the field can actually reuse and is likely
the paper's most citable output".  It combines the two uncertainty sources
that WP-4c and WP-1c measure separately into credible bands on every learned
coefficient, and exports them with units and provenance in a format another
MFA modeller can load without reading any of this code.

Combination — Rubin's rules
---------------------------
Thirty-five seeds give thirty-five estimates `g_s(t)`, each with its own
Laplace covariance `Sigma_s(t)` from WP-4c.  These are not repeated
measurements of the same thing and they are not independent replicates; they
are one estimator run from thirty-five initialisations on identical data.
Rubin (1987) is the right pooling rule for exactly that structure:

    gbar   = mean_s g_s                                    (pooled estimate)
    Wbar   = mean_s Sigma_s                                (within)
    B      = (1/(S-1)) sum_s (g_s - gbar)(g_s - gbar)^T    (between)
    T      = Wbar + (1 + 1/S) B                            (total)
    nu     = (S-1) (1 + Wbar/((1+1/S) B))^2                (dof)

`Wbar` is the sampling uncertainty of the estimator at fixed initialisation.
`B` is the initialisation-and-optimisation-path spread, which WP-1c flag 1
establishes is a LOWER bound on the total parameter variance because every
seed saw the same data and the same split.  Neither is the whole answer;
reporting the decomposition is more useful than reporting either alone, so
`wp8h_uncertainty_decomposition.csv` gives the share each contributes.

Bands
-----
Symmetric normal bands would put negative values inside the interval for a
small rate and values above 1 inside the interval for a share.  Bands are
therefore built on the scale the model itself parameterises the quantity on
and mapped back:

    rates  (alpha_*)              log-normal:   gbar * exp(+- z sd/gbar)
    shares (tau_*, frac_*, f_*)   logit-normal: sigmoid(logit(gbar) +- z sd/(g(1-g)))

The symmetric band is exported alongside so the transformation can be undone.

What the bands do NOT cover
---------------------------
Five coefficients are pinned to the Rostek data (`tau_ref`, `tau_waelz`,
`tau_diss`, `frac_fu_loss`, `frac_eu_loss`) and three are fixed by
construction (`mu_10yr/20yr/44yr`).  For these the estimator has no
uncertainty at all, and the export says `estimated = 0` rather than
`sd = 0` without comment: the quantity is uncertain, this estimator simply
does not measure that uncertainty.  Nor do the bands cover model-structure
error — the fixed-exponential cohorts, the aggregation of Chapter 2's s2 and
s3, the absence of s6-s8 — or the observation-construction bias in
`alpha_obs` that WP-1a flags and WP-3 would quantify.

    python run_wp8h.py --check
    python run_wp8h.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

import zinc_circ_lab as C
import zinc_fisher_lab as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "analysis")
SEED_DIR = os.path.join(OUT_DIR, "wp4c")
SIGMA_NPZ = os.path.join(OUT_DIR, "wp4c_sigma.npz")
A_NPZ = os.path.join(OUT_DIR, "wp2a_A_of_t.npz")
NOTES = os.path.join(OUT_DIR, "notes")

COL = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00", "#56B4E9"]
GREY = "#555555"

Z68, Z95 = 0.9944578832097535, 1.959963984540054

# Units and one-line descriptions, carried into both exports.  A reuse
# artifact without units is a table of numbers, not data.
UNITS = {}
LONGNAME = {}
for _n in C.PARAM_NAMES:
    if _n.startswith("alpha"):
        UNITS[_n], LONGNAME[_n] = "1/yr", "transfer rate out of the parent stock"
    elif _n.startswith("mu"):
        UNITS[_n], LONGNAME[_n] = "yr", "mean in-use residence time (fixed)"
    else:
        UNITS[_n], LONGNAME[_n] = "1", "dimensionless transfer share"
LONGNAME.update({
    "alpha_cc": "concentrate consumption rate (Ch2 alpha_1)",
    "alpha_refc": "refined-metal consumption rate (Ch2 alpha_2/alpha_8)",
    "alpha_win": "old scrap to metallurgy, Waelz route (Ch2 alpha_14)",
    "alpha_dr": "old scrap to manufacturing, direct reuse (Ch2 alpha_13)",
    "tau_ref": "refining loss share (pinned to data)",
    "tau_waelz": "Waelz recovery share (pinned to data)",
    "tau_olds": "end-of-life collection share (Ch2 alpha_9 share)",
    "tau_diss": "dissipative loss share (pinned to data)",
    "frac_fu_new": "first-use: new scrap share",
    "frac_fu_loss": "first-use: loss share (pinned to data)",
    "frac_fu_out": "first-use: share proceeding to end use",
    "frac_eu_new": "end use: new scrap share",
    "frac_eu_loss": "end use: loss share (pinned to data)",
    "frac_eu_into": "end use: share entering the in-use stock",
    "f_cohort_10yr": "in-use cohort share, mean lifetime 10 yr",
    "f_cohort_20yr": "in-use cohort share, mean lifetime 20 yr",
    "f_cohort_44yr": "in-use cohort share, mean lifetime 44 yr",
    "mu_10yr": "short cohort mean lifetime (fixed, Rostek SI Table S4)",
    "mu_20yr": "medium cohort mean lifetime (fixed)",
    "mu_44yr": "long cohort mean lifetime (fixed)",
})

RATE = [n for n in C.PARAM_NAMES if n.startswith("alpha")]
SHARE = [n for n in C.PARAM_NAMES
         if not n.startswith("alpha") and not n.startswith("mu")]


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ===========================================================================
# pooling
# ===========================================================================
def rubin(coef, cov):
    """Rubin's rules over the seed ensemble.

    `coef` (S, T, C), `cov` (S, T, C, C).  Returns the pooled estimate, the
    within / between / total covariances and the per-component degrees of
    freedom.
    """
    S = coef.shape[0]
    gbar = coef.mean(0)
    dev = coef - gbar[None]
    B = np.einsum("stc,std->tcd", dev, dev) / max(S - 1, 1)
    Wbar = cov.mean(0)
    T = Wbar + (1.0 + 1.0 / S) * B
    w = np.einsum("tcc->tc", Wbar)
    b = np.einsum("tcc->tc", B) * (1.0 + 1.0 / S)
    with np.errstate(divide="ignore", invalid="ignore"):
        nu = (S - 1) * (1.0 + w / np.where(b > 0, b, np.nan)) ** 2
    nu = np.where(np.isfinite(nu), nu, float(S - 1))
    return gbar, Wbar, B, T, np.clip(nu, 1.0, 1e6), w, b


def bands(gbar, sd, names, nu):
    """Credible bands on the scale the model parameterises each quantity on."""
    from scipy import stats
    t68 = stats.t.ppf(0.5 + 0.68 / 2.0, nu)
    t95 = stats.t.ppf(0.975, nu)
    out = {}
    for j, n in enumerate(names):
        g, s = gbar[:, j], sd[:, j]
        sym = dict(lo68=g - t68[:, j] * s, hi68=g + t68[:, j] * s,
                   lo95=g - t95[:, j] * s, hi95=g + t95[:, j] * s)
        if n in RATE:
            r = np.where(g > 0, s / np.maximum(g, 1e-300), 0.0)
            tr = dict(lo68=g * np.exp(-t68[:, j] * r),
                      hi68=g * np.exp(+t68[:, j] * r),
                      lo95=g * np.exp(-t95[:, j] * r),
                      hi95=g * np.exp(+t95[:, j] * r), scale="log-normal")
        elif n in SHARE:
            p = np.clip(g, 1e-9, 1 - 1e-9)
            r = s / (p * (1 - p))
            lg = np.log(p / (1 - p))
            sig = lambda x: 1.0 / (1.0 + np.exp(-x))
            tr = dict(lo68=sig(lg - t68[:, j] * r), hi68=sig(lg + t68[:, j] * r),
                      lo95=sig(lg - t95[:, j] * r), hi95=sig(lg + t95[:, j] * r),
                      scale="logit-normal")
        else:
            tr = dict(lo68=g, hi68=g, lo95=g, hi95=g, scale="fixed")
        out[n] = dict(sym=sym, tr=tr)
    return out


# ===========================================================================
# transfer matrix
# ===========================================================================
def A_with_bands(gbar_full, T_full, years):
    """`A(t)` and the delta-method standard error of every entry.

    `dA/dp` comes from `zinc_circ_lab.dA_dparams` (complex step, exact for a
    polynomial assembly), so
    `Var(A_ij) = (dA_ij/dp)^T Cov(p) (dA_ij/dp)` — with the FULL covariance,
    which matters here more than anywhere: `A[1,0] = (1 - tau_ref) alpha_cc`
    mixes two parameters, and the three cohort rows mix five.

    # Ch2 Eq. system, Eq. a1
    """
    A = C.assemble_from_params(gbar_full)                  # (T, 6, 6)
    dA = C.dA_dparams(gbar_full)                           # (T, P, 6, 6)
    var = np.einsum("tpij,tpq,tqij->tij", dA, T_full, dA)
    return A, np.sqrt(np.maximum(var, 0.0))


# ===========================================================================
# Chapter 2 indicators with estimation uncertainty (addendum to WP-2d/2e)
# ===========================================================================
WP2E_NPZ = os.path.join(OUT_DIR, "wp2e_indicator_sensitivity.npz")


def indicator_uncertainty(cov_seed, seeds, years):
    """Frozen-time circularity indicators with a proper standard error.

    WP-2d called the circularity indicators the highest-value result of its
    package and reported them with the across-seed spread alone, because that
    was the only uncertainty available at the time.  WP-2e already computed
    every derivative needed to do better — `dtau_i/dp_n` and
    `dUpsilon_{m,i}/dp_n` for all 20 parameters, from
    `dN = N (dA) N` (Ch2 Eq. `fundamental_matrix_derivative`) — so with WP-4c's
    covariance the delta method finishes the job with no new integration:

        Var(tau_i) = (dtau_i/dp)^T Cov(p) (dtau_i/dp)

    Pooled across seeds by the same Rubin rules as the coefficients.

    # Ch2 Eq. frozen_time_circularity, Eq. old_scrap_count_appendix

    Restricted to the FROZEN-TIME form.  WP-2e stored non-autonomous
    elasticities for the eight rate and tau parameters only, not for the nine
    simplex shares or the three cohort lifetimes, so the same propagation
    cannot be closed for `tau_FD` and `upsilon` without recomputing those
    twelve derivatives — flagged, not approximated.
    """
    if not os.path.exists(WP2E_NPZ):
        return None
    e = np.load(WP2E_NPZ, allow_pickle=True)
    if not np.array_equal(np.asarray(e["seeds"]), np.asarray(seeds)):
        raise SystemExit("wp2e seeds do not match wp4c seeds")
    if [str(x) for x in e["param_names"]] != list(C.PARAM_NAMES):
        raise SystemExit("wp2e param_names do not match PARAM_NAMES")
    S = len(seeds)
    state = list(C.ODE_STATE_NAMES)

    targets = [("tau_frozen", "dtau_dparams", "yr",
                "frozen-time residence time of a unit entering this state")]
    for fl in [str(x) for x in e["flow_names"]]:
        targets.append((f"count_frozen_{fl}", f"dcount_dparams_{fl}", "1",
                        f"frozen-time expected number of passages through "
                        f"`{fl}`"))

    rows = []
    for val_key, der_key, unit, long in targets:
        V = np.asarray(e[val_key], float)                    # (S, T, 6)
        D = np.asarray(e[der_key], float)                    # (S, T, 20, 6)
        # per-seed delta-method variance
        var_s = np.einsum("stpi,stpq,stqi->sti", D, cov_seed, D)
        gbar = V.mean(0)
        B = V.var(0, ddof=1)
        Wbar = var_s.mean(0)
        Tot = Wbar + (1.0 + 1.0 / S) * B
        sd = np.sqrt(np.maximum(Tot, 0.0))
        for t, y in enumerate(years):
            for i, st in enumerate(state):
                rows.append(dict(
                    year=int(y), indicator=val_key, state=st, unit=unit,
                    long_name=long, value=float(gbar[t, i]),
                    sd_within=float(np.sqrt(max(Wbar[t, i], 0.0))),
                    sd_between=float(np.sqrt(max(B[t, i], 0.0))),
                    sd_total=float(sd[t, i]),
                    lo95=float(gbar[t, i] - Z95 * sd[t, i]),
                    hi95=float(gbar[t, i] + Z95 * sd[t, i]),
                    # Residence times and passage counts are non-negative, so
                    # a symmetric band can run below zero on the least
                    # determined of them; the log-normal band is the one to
                    # quote and the symmetric one is kept for reference.
                    lo95_log=float(gbar[t, i] * np.exp(
                        -Z95 * sd[t, i] / max(abs(gbar[t, i]), 1e-300))),
                    hi95_log=float(gbar[t, i] * np.exp(
                        +Z95 * sd[t, i] / max(abs(gbar[t, i]), 1e-300))),
                    rel_sd=float(sd[t, i] / max(abs(gbar[t, i]), 1e-300))))
    return pd.DataFrame(rows)


def nonauto_indicator_uncertainty(coef, years, seeds, *, n_draw=48,
                                  rule="hold", n_sub=64, verbose=True):
    """Chapter 2's non-autonomous indicators with a proper standard error.

    `tau_FD,i(t0)` and `upsilon_i(t0)` (Ch2 Eqs. `nonautonomous_lifetime`,
    `nonautonomous_use_count`) integrate the coefficient path over the whole
    absorption horizon, so their uncertainty depends on how the coefficient
    error at one year relates to the error at another — which the per-year
    marginals cannot say.  `zinc_fisher_lab.sample_coef_paths` draws whole
    paths from each seed's Laplace posterior in weight space, where the
    cross-year structure is automatic, and the indicator is then evaluated on
    each draw.  Nothing is assumed about the correlation.

    Pooling follows the law of total variance, which is Rubin's rule with the
    within term estimated by Monte Carlo instead of by the delta method:
    within = mean over seeds of the across-draw variance; between = across-seed
    variance of the unperturbed indicator.

    The frozen-time indicator is evaluated on the same draws, so the
    frozen-vs-non-autonomous **discrepancy** — WP-2d's headline — gets a
    standard error too, computed on paired draws rather than from two
    independent error bars.
    """
    import jax.numpy as jnp
    import zinc_cf_lab as cflab

    cfg = F.load_anchor_config()
    fit = cflab.init_fit(cfg=cfg, seed=0)
    S, T, NC = coef.shape
    NP = len(C.PARAM_NAMES)
    MU = np.array([10.0, 20.0, 44.0])
    state = list(C.ODE_STATE_NAMES)

    def pad(g):
        out = np.zeros(g.shape[:-1] + (NP,))
        out[..., :NC] = g
        out[..., 17:] = MU
        return out

    P0 = pad(coef)                                    # (S, T, 20)
    base_na = C.nonautonomous_indicators(P0, years, rule=rule, n_sub=n_sub)
    fr_tau0, fr_cnt0, _N0 = C.frozen_indicators(C.assemble_from_params(P0))

    keys = ["tau"] + list(base_na["counts"])
    var_w = {k: np.zeros((T, 6)) for k in keys}
    var_w_fr = {k: np.zeros((T, 6)) for k in keys}
    var_w_dis = {k: np.zeros((T, 6)) for k in keys}
    checks = []
    for i, sd_ in enumerate(seeds):
        theta, _ = __import__("jax").flatten_util.ravel_pytree(
            cflab.load_params(os.path.join(F.WEIGHTS_DIR_DEFAULT,
                                           f"A_seed{sd_}.npz")))
        dg, info = F.sample_coef_paths(fit, cfg, jnp.asarray(theta),
                                       n_draw=n_draw, rng_seed=1000 + int(sd_))
        chk_row = dict(seed=int(sd_), **{k: v for k, v in info.items()
                                         if k != "coef"})
        Pd_raw = pad(coef[i][None] + dg)              # (K, T, 20)
        # A posterior draw can land outside the feasible set — a negative rate,
        # a share above 1 — and there `-A` is not an M-matrix, the fundamental
        # matrix has negative entries and the indicator is not defined.  Draws
        # are projected back (`zinc_circ_lab._clip_to_feasible`, the same
        # projection WP-2d applies to extrapolated paths) and the share of
        # draws that needed it is reported: it is a direct readout of how much
        # of the Laplace posterior the physics rules out, and a Gaussian
        # posterior on a constrained parameter is exactly where that happens.
        Pd = C._clip_to_feasible(Pd_raw)
        clipped = float(np.mean(np.any(
            np.abs(Pd - Pd_raw) > 1e-12, axis=-1)))
        na = C.nonautonomous_indicators(Pd, years, rule=rule, n_sub=n_sub)
        fr_tau, fr_cnt, _Nd = C.frozen_indicators(
            C.assemble_from_params(Pd))
        for k in keys:
            a = na["tau"] if k == "tau" else na["counts"][k]
            b = fr_tau if k == "tau" else fr_cnt[k]
            var_w[k] += a.var(0, ddof=1) / S
            var_w_fr[k] += b.var(0, ddof=1) / S
            var_w_dis[k] += (a - b).var(0, ddof=1) / S
        chk_row["frac_draw_years_clipped"] = clipped
        # How correlated is the coefficient error across years INSIDE one
        # seed's posterior?  This is the quantity a coherent-path-shift
        # approximation would set to 1, and the reason the indicator variance
        # is sampled rather than assumed.
        for jj, nm in ((2, "alpha_win"), (3, "alpha_dr"), (0, "alpha_cc")):
            a, b = dg[:, 0, jj], dg[:, -1, jj]
            m, mm = dg[:, 10, jj], dg[:, 30, jj]
            chk_row[f"post_corr_1980_2019_{nm}"] = (
                float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0
                else np.nan)
            chk_row[f"post_corr_1990_2010_{nm}"] = (
                float(np.corrcoef(m, mm)[0, 1]) if m.std() > 0 and mm.std() > 0
                else np.nan)
        checks.append(chk_row)
        if verbose:
            print(f"    seed {sd_:2d}  marginal check "
                  f"{info['check_marginals']:.3f}  clipped "
                  f"{100*clipped:.1f}% of draw-years", flush=True)
        __import__("jax").clear_caches()

    rows = []
    for k in keys:
        an = base_na["tau"] if k == "tau" else base_na["counts"][k]
        af = fr_tau0 if k == "tau" else fr_cnt0[k]
        unit = "yr" if k == "tau" else "1"
        for label, val, vw in (("non-autonomous", an, var_w[k]),
                               ("frozen", af, var_w_fr[k]),
                               ("discrepancy (nonauto - frozen)", an - af,
                                var_w_dis[k])):
            gbar = val.mean(0)
            B = val.var(0, ddof=1)
            tot = vw + (1.0 + 1.0 / S) * B
            sd = np.sqrt(np.maximum(tot, 0.0))
            for t, y in enumerate(years):
                for j, st in enumerate(state):
                    g = float(gbar[t, j])
                    rows.append(dict(
                        year=int(y), indicator=k, form=label, state=st,
                        unit=unit, value=g,
                        sd_within=float(np.sqrt(max(vw[t, j], 0.0))),
                        sd_between=float(np.sqrt(max(B[t, j], 0.0))),
                        sd_total=float(sd[t, j]),
                        lo95=g - Z95 * float(sd[t, j]),
                        hi95=g + Z95 * float(sd[t, j]),
                        rel_sd=float(sd[t, j] / max(abs(g), 1e-300)),
                        n_draw=int(n_draw), rule=rule))
    return pd.DataFrame(rows), pd.DataFrame(checks)


# ===========================================================================
# exports
# ===========================================================================
def _ascii(v):
    """NetCDF-3 attributes are ASCII.  Em dashes and the like are transposed
    rather than dropped, so the provenance string stays readable."""
    return (str(v).replace("\u2014", " - ").replace("\u2013", "-")
            .replace("\u2018", "'").replace("\u2019", "'")
            .replace("\u201c", '"').replace("\u201d", '"')
            .encode("ascii", "replace").decode("ascii"))


def write_netcdf(path, years, names, data2d, attrs, var_attrs):
    """NetCDF-3 classic through `scipy.io.netcdf_file`.

    Classic rather than NetCDF-4 on purpose: it needs no HDF5 and no
    `netCDF4`/`xarray` install, and every tool that reads netCDF reads it.
    The coefficient axis is written as a character matrix, which is how
    NetCDF-3 carries strings.
    """
    from scipy.io import netcdf_file

    nlen = max(len(n) for n in names)
    f = netcdf_file(path, "w")
    f.createDimension("year", len(years))
    f.createDimension("coefficient", len(names))
    f.createDimension("namelen", nlen)
    v = f.createVariable("year", "i", ("year",))
    v[:] = np.asarray(years, np.int32)
    v.units = "year"
    v.long_name = "calendar year (point-in-time for stocks, see the note)"
    nv = f.createVariable("coefficient_name", "c", ("coefficient", "namelen"))
    for i, n in enumerate(names):
        nv[i, :] = np.array(list(n.ljust(nlen)), "c")
    for key, arr in data2d.items():
        vv = f.createVariable(key, "d", ("year", "coefficient"))
        vv[:] = np.asarray(arr, np.float64)
        for a, val in var_attrs.get(key, {}).items():
            setattr(vv, a, val)
    uv = f.createVariable("units", "c", ("coefficient", "namelen"))
    for i, n in enumerate(names):
        u = UNITS[n].ljust(nlen)[:nlen]
        uv[i, :] = np.array(list(u), "c")
    for k, val in attrs.items():
        setattr(f, k, _ascii(val))
    f.close()
    return path


def write_netcdf_matrix(path, years, state_names, A, sdA, attrs):
    from scipy.io import netcdf_file

    nlen = max(len(n) for n in state_names)
    f = netcdf_file(path, "w")
    f.createDimension("year", len(years))
    f.createDimension("to_state", A.shape[1])
    f.createDimension("from_state", A.shape[2])
    f.createDimension("namelen", nlen)
    v = f.createVariable("year", "i", ("year",)); v[:] = np.asarray(years, np.int32)
    v.units = "year"
    nv = f.createVariable("state_name", "c", ("to_state", "namelen"))
    for i, n in enumerate(state_names):
        nv[i, :] = np.array(list(n.ljust(nlen)), "c")
    for key, arr, ln in (("A", A, "transfer matrix dS/dt = A(t) S + b(t)"),
                         ("A_sd", sdA, "delta-method standard error of A")):
        vv = f.createVariable(key, "d", ("year", "to_state", "from_state"))
        vv[:] = np.asarray(arr, np.float64)
        vv.units = "1/yr"
        vv.long_name = ln
    for k, val in attrs.items():
        setattr(f, k, _ascii(val))
    f.close()
    return path


# ===========================================================================
# main
# ===========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--seed-dir", default=SEED_DIR)
    ap.add_argument("--nonauto", action="store_true",
                    help="also propagate uncertainty into Chapter 2's "
                         "NON-AUTONOMOUS indicators by sampling whole "
                         "coefficient paths from each seed's Laplace "
                         "posterior (~1 min per seed)")
    ap.add_argument("--nonauto-draws", type=int, default=48)
    args = ap.parse_args(argv)

    if args.check:
        F.check(verbose=True)
        print("run_wp8h --check")
        print("=" * 74)
        print(f"  seed dumps           : {args.seed_dir}")
        print(f"  pooling              : Rubin (1987) — T = Wbar + (1+1/S) B")
        print(f"  band scales          : rates log-normal, shares "
              f"logit-normal, fixed quantities point")
        print(f"  units                : "
              + ", ".join(f"{n}[{UNITS[n]}]" for n in C.PARAM_NAMES[:6]) + ", …")
        print(f"  netCDF               : NetCDF-3 classic via "
              f"scipy.io.netcdf_file (no HDF5 dependency)")
        print(f"  NOT covered by the bands: model-structure error, the pinned "
              f"coefficients' own data uncertainty,")
        print( "                            and the alpha_obs construction "
               "bias (WP-1a secondary hypothesis, WP-3).")
        print("=" * 74)
        return 0

    if not os.path.exists(SIGMA_NPZ):
        raise SystemExit(f"{SIGMA_NPZ} not found — run `python run_wp4c.py`")
    paths = sorted((p for p in os.listdir(args.seed_dir)
                    if p.startswith("wp4c_seed")),
                   key=lambda p: int(p[len("wp4c_seed"):-4]))
    ds = [np.load(os.path.join(args.seed_dir, p), allow_pickle=True)
          for p in paths]
    seeds = np.array([int(d["seed"]) for d in ds])
    years = np.asarray(ds[0]["years"], float).ravel()
    names17 = [str(x) for x in ds[0]["coef_names"]]
    lam_rel = np.asarray(ds[0]["lam_rel"], float)
    il = int(np.argmin(np.abs(lam_rel - F.LAMBDA_REF)))

    coef = np.stack([np.asarray(d["coef"]) for d in ds])           # (S,T,17)
    cov = np.stack([np.asarray(d["coef_cov"])[il] for d in ds])    # (S,T,17,17)
    est = np.stack([np.asarray(d["estimated"]) for d in ds]).any(0)  # (T,17)

    gbar, Wbar, B, Tot, nu, wdiag, bdiag = rubin(coef, cov)
    sd = np.sqrt(np.maximum(np.einsum("tcc->tc", Tot), 0.0))
    bd = bands(gbar, sd, names17, nu)

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(NOTES, exist_ok=True)

    # ---- long CSV --------------------------------------------------------
    rows = []
    for t, y in enumerate(years):
        for j, n in enumerate(names17):
            tr, sym = bd[n]["tr"], bd[n]["sym"]
            rows.append(dict(
                year=int(y), coefficient=n, unit=UNITS[n],
                long_name=LONGNAME[n],
                # Not "" and not "n/a": an empty cell reads back as NaN, and
                # so does "n/a" — it is in pandas' default NA list.  A reuse
                # artifact should not need a footnote to say that a missing
                # label means "no Chapter 2 analogue".
                ch2_label=C.CH2_LABEL.get(n, "not_mapped"),
                estimated=int(bool(est[t, j])),
                value=float(gbar[t, j]),
                sd_total=float(sd[t, j]),
                sd_within=float(np.sqrt(max(wdiag[t, j], 0.0))),
                sd_between=float(np.sqrt(max(bdiag[t, j], 0.0))),
                dof=float(nu[t, j]),
                lo68=float(tr["lo68"][t]), hi68=float(tr["hi68"][t]),
                lo95=float(tr["lo95"][t]), hi95=float(tr["hi95"][t]),
                band_scale=tr["scale"],
                lo95_symmetric=float(sym["lo95"][t]),
                hi95_symmetric=float(sym["hi95"][t]),
                n_seeds=int(len(seeds))))
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out, "wp8h_coefficients.csv")
    df.to_csv(csv_path, index=False)

    # ---- uncertainty decomposition ---------------------------------------
    dec = []
    for j, n in enumerate(names17):
        if not est[:, j].any():
            dec.append(dict(coefficient=n, estimated=0, within_share=np.nan,
                            between_share=np.nan, sd_total_median=0.0,
                            rel_sd_total_median=np.nan,
                            note="pinned to data — this estimator carries no "
                                 "uncertainty for it"))
            continue
        w, b = wdiag[:, j], bdiag[:, j]
        tot = np.maximum(w + b, 1e-300)
        dec.append(dict(
            coefficient=n, estimated=1,
            within_share=float(np.median(w / tot)),
            between_share=float(np.median(b / tot)),
            sd_total_median=float(np.median(sd[:, j])),
            rel_sd_total_median=float(np.median(
                sd[:, j] / np.maximum(np.abs(gbar[:, j]), 1e-300))),
            dof_median=float(np.median(nu[:, j])),
            note=("Laplace-dominated" if np.median(w / tot) > 0.5
                  else "seed-spread-dominated")))
    dec = pd.DataFrame(dec)
    dec.to_csv(os.path.join(args.out, "wp8h_uncertainty_decomposition.csv"),
               index=False)

    # ---- transfer matrix -------------------------------------------------
    P = len(C.PARAM_NAMES)
    gfull = np.zeros((len(years), P))
    gfull[:, :len(names17)] = gbar
    gfull[:, 17:] = np.asarray(
        np.load(A_NPZ, allow_pickle=True)["mu_cohorts"], float)[None, :] \
        if os.path.exists(A_NPZ) else np.array([10.0, 20.0, 44.0])[None, :]
    Tfull = np.zeros((len(years), P, P))
    Tfull[:, :len(names17), :len(names17)] = Tot
    A, sdA = A_with_bands(gfull, Tfull, years)

    state = list(C.ODE_STATE_NAMES)
    arows = []
    for t, y in enumerate(years):
        for i in range(A.shape[1]):
            for k in range(A.shape[2]):
                if abs(A[t, i, k]) == 0.0 and sdA[t, i, k] == 0.0:
                    continue
                arows.append(dict(year=int(y), to_state=state[i],
                                  from_state=state[k], unit="1/yr",
                                  value=float(A[t, i, k]),
                                  sd=float(sdA[t, i, k]),
                                  lo95=float(A[t, i, k] - Z95 * sdA[t, i, k]),
                                  hi95=float(A[t, i, k] + Z95 * sdA[t, i, k])))
    pd.DataFrame(arows).to_csv(
        os.path.join(args.out, "wp8h_transfer_matrix.csv"), index=False)

    # ---- provenance ------------------------------------------------------
    prov = dict(
        title="Time-varying transfer coefficients of the global anthropogenic "
              "zinc cycle, 1980-2019, with credible bands",
        summary="Coefficients of a universal differential equation model "
                "(dS/dt = A(t) S + b(t)) fitted to the Rostek et al. (2022) "
                "zinc MFA. Uncertainty pools a 35-seed ensemble with the "
                "per-seed Laplace approximation by Rubin's rules.",
        source="zinc_colloc_v5.py (anchor_v4 configuration), 35 seeds; "
               "WP-4c Fisher information; WP-8h pooling",
        model_md5=_md5(os.path.join(HERE, "zinc_colloc_v5.py")),
        data_md5=_md5(os.path.join(HERE, "zinc_dataset.xlsx")),
        data_source="Rostek, Pauliuk et al. (2022), global zinc MFA, "
                    "N = 40 annual observations 1980-2019",
        n_seeds=int(len(seeds)),
        laplace_lambda_rel=float(F.LAMBDA_REF),
        pooling="Rubin (1987): T = Wbar + (1 + 1/S) B",
        band_convention="rates log-normal, shares logit-normal; symmetric "
                        "bands also provided in the CSV",
        not_covered="model-structure error (fixed-exponential in-use cohorts; "
                    "Ch2 s2/s3 aggregated; s6-s8 absent); the pinned "
                    "coefficients' own data uncertainty; the alpha_obs "
                    "construction bias (integral over interpolated point)",
        pinned_coefficients="tau_ref, tau_waelz, tau_diss, frac_fu_loss, "
                            "frac_eu_loss (data); mu_10yr, mu_20yr, mu_44yr "
                            "(fixed by construction)",
        held_out_window="2007-2019 (pre-registered; never used for model "
                        "selection)",
        history=f"created {_dt.date.today().isoformat()} by run_wp8h.py",
        conventions="CF-like attribute names, but not CF-compliant",
        license="not specified — contact the author before redistribution",
    )
    with open(os.path.join(args.out, "wp8h_provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2)

    data2d = dict(value=gbar, sd_total=sd,
                  sd_within=np.sqrt(np.maximum(wdiag, 0.0)),
                  sd_between=np.sqrt(np.maximum(bdiag, 0.0)),
                  dof=nu, estimated=est.astype(float),
                  lo68=np.stack([bd[n]["tr"]["lo68"] for n in names17], 1),
                  hi68=np.stack([bd[n]["tr"]["hi68"] for n in names17], 1),
                  lo95=np.stack([bd[n]["tr"]["lo95"] for n in names17], 1),
                  hi95=np.stack([bd[n]["tr"]["hi95"] for n in names17], 1))
    var_attrs = {k: dict(long_name=k) for k in data2d}
    var_attrs["value"] = dict(long_name="pooled coefficient estimate")
    var_attrs["sd_total"] = dict(long_name="Rubin total standard deviation")
    var_attrs["estimated"] = dict(
        long_name="1 if this estimator carries uncertainty for the "
                  "coefficient, 0 if it is pinned to data or fixed")
    nc = write_netcdf(os.path.join(args.out, "wp8h_coefficients.nc"),
                      years, names17, data2d, prov, var_attrs)
    ncA = write_netcdf_matrix(
        os.path.join(args.out, "wp8h_transfer_matrix.nc"),
        years, state, A, sdA, prov)

    # ---- read-back check -------------------------------------------------
    from scipy.io import netcdf_file
    g2 = netcdf_file(nc, "r")
    rb = float(np.max(np.abs(np.asarray(g2.variables["value"][:]) - gbar)))
    rbnames = ["".join(c.decode() for c in row).strip()
               for row in np.asarray(g2.variables["coefficient_name"][:])]
    g2.close()
    val = pd.DataFrame([
        dict(check="netCDF read-back of `value` vs the in-memory array",
             result=rb, exact_answer=0.0),
        dict(check="netCDF coefficient names round-trip",
             result=float(rbnames != names17), exact_answer=0.0),
        dict(check="pooled estimate equals the seed mean",
             result=float(np.max(np.abs(gbar - coef.mean(0)))),
             exact_answer=0.0),
        dict(check="total covariance is positive semidefinite at every year",
             result=float(np.min([np.linalg.eigvalsh(Tot[t]).min()
                                  for t in range(len(years))])),
             exact_answer=0.0),
    ])
    if os.path.exists(A_NPZ):
        A_seedmean = np.asarray(np.load(A_NPZ, allow_pickle=True)["A"],
                                float).mean(0)
        gap = np.abs(A - A_seedmean)
        scale = np.maximum(np.abs(A_seedmean), 1e-12)
        rel = np.where(np.abs(A_seedmean) > 1e-9, gap / scale, 0.0)
        i, j, k = np.unravel_index(np.argmax(rel), rel.shape)
        val = pd.concat([val, pd.DataFrame([
            # A is a PRODUCT of coefficients ((1-tau_ref) alpha_cc, and the
            # three cohort rows mix five), so E[A(p)] != A(E[p]) by Jensen.
            # The gap is the second-order term, not an error; it is reported
            # with its size because a reuser needs to know which of the two
            # objects the export is.
            dict(check="Jensen gap: A(pooled coefficients) vs mean_s A_s "
                       "(max abs, 1/yr) — expected non-zero, A is a product",
                 result=float(gap.max()), exact_answer=np.nan),
            dict(check=f"Jensen gap, max RELATIVE, at "
                       f"A[{state[j]}<-{state[k]}] in {int(years[i])}",
                 result=float(rel[i, j, k]), exact_answer=np.nan),
        ])], ignore_index=True)
    val.to_csv(os.path.join(args.out, "wp8h_validation.csv"), index=False)

    # ---- Chapter 2 indicators (addendum to WP-2d/2e) ---------------------
    C_lap = np.stack([np.asarray(d["coef_cov"])[il] for d in ds])
    cov_full = np.zeros((len(seeds), len(years), P, P))
    cov_full[:, :, :len(names17), :len(names17)] = C_lap
    ind = indicator_uncertainty(cov_full, seeds, years)
    if ind is not None:
        ind.to_csv(os.path.join(args.out, "wp8h_indicator_uncertainty.csv"),
                   index=False)
        print("\n  Chapter 2 frozen-time indicators at 2019 "
              "(delta method through WP-4c's covariance):")
        sel = ind[(ind.year == int(years[-1]))
                  & ind.indicator.isin(["tau_frozen",
                                        "count_frozen_oldscrap_metallurgy",
                                        "count_frozen_oldscrap_manufacture"])]
        for _, r in sel[sel.state.isin(["S_conc", "S_ref", "S_scrap"])].iterrows():
            print(f"    {r.indicator:34s} {r.state:11s} "
                  f"{r.value:9.4f} +- {r.sd_total:.4f} "
                  f"({100*r.rel_sd:5.1f}%)  [{r.lo95:.4f}, {r.hi95:.4f}]")

    # Across-seed correlation of the coefficient deviations between the first
    # and last year — cheap, always written, and the testable half of the
    # question the non-autonomous block has to answer.
    devc = coef - coef.mean(0, keepdims=True)
    pd.DataFrame([
        dict(coefficient=n,
             corr_1980_2019=(float(np.corrcoef(devc[:, 0, j],
                                               devc[:, -1, j])[0, 1])
                             if devc[:, 0, j].std() > 0
                             and devc[:, -1, j].std() > 0 else np.nan),
             corr_1990_2010=(float(np.corrcoef(devc[:, 10, j],
                                               devc[:, 30, j])[0, 1])
                             if devc[:, 10, j].std() > 0
                             and devc[:, 30, j].std() > 0 else np.nan))
        for j, n in enumerate(names17)
    ]).to_csv(os.path.join(args.out, "wp8h_coherence_check.csv"), index=False)

    if args.nonauto:
        print(f"\n  non-autonomous indicators by posterior path sampling "
              f"({args.nonauto_draws} draws x {len(seeds)} seeds, ~1 min per "
              f"seed) …", flush=True)
        na, nachk = nonauto_indicator_uncertainty(
            coef, years, seeds, n_draw=args.nonauto_draws)
        na.to_csv(os.path.join(args.out,
                               "wp8h_indicator_uncertainty_nonauto.csv"),
                  index=False)
        nachk.to_csv(os.path.join(args.out, "wp8h_nonauto_sampler_check.csv"),
                     index=False)
        sel = na[(na.year == int(years[-1])) & (na.state == "S_conc")]
        for _, r in sel.iterrows():
            print(f"    {r.indicator:22s} {r['form']:32s} "
                  f"{r.value:9.4f} +- {r.sd_total:.4f} ({100*r.rel_sd:6.1f}%)")
        print(f"    sampler check (max relative gap between the Monte Carlo "
              f"and analytic marginals): "
              f"{nachk.check_marginals.max():.3f}")

    # ---- figure ----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = ["alpha_cc", "alpha_refc", "alpha_win", "alpha_dr",
            "tau_olds", "frac_fu_new", "frac_eu_new", "f_cohort_44yr"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.4))
    for ax, n in zip(axes.ravel(), show):
        j = names17.index(n)
        tr = bd[n]["tr"]
        ax.fill_between(years, tr["lo95"], tr["hi95"], color=COL[0], alpha=0.18,
                        lw=0, label="95%")
        ax.fill_between(years, tr["lo68"], tr["hi68"], color=COL[0], alpha=0.35,
                        lw=0, label="68%")
        for s in range(coef.shape[0]):
            ax.plot(years, coef[s, :, j], color=GREY, lw=0.4, alpha=0.30)
        ax.plot(years, gbar[:, j], color="k", lw=1.8)
        ax.set_title(f"{n}  [{UNITS[n]}]", fontsize=9)
        ax.grid(alpha=0.25)
        if n in RATE:
            ax.set_yscale("log")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("WP-8h — pooled coefficient trajectories with credible bands "
                 "(grey: individual seeds)", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out, f"wp8h_coefficients.{ext}"), dpi=200)
    plt.close(fig)

    # ---- console ---------------------------------------------------------
    print("=" * 74)
    print("WP-8h — coefficient uncertainty product")
    print("=" * 74)
    print(f"  seeds {len(seeds)}   years {int(years[0])}-{int(years[-1])}   "
          f"coefficients {len(names17)} (+3 fixed mu)")
    print(f"\n  {'coefficient':16s}{'2019 value':>12s}{'sd_total':>11s}"
          f"{'rel':>8s}{'within%':>9s}{'between%':>10s}  band")
    for j, n in enumerate(names17):
        if not est[:, j].any():
            print(f"  {n:16s}{gbar[-1, j]:12.4f}{'—':>11s}{'—':>8s}"
                  f"{'—':>9s}{'—':>10s}  pinned to data")
            continue
        w, b = wdiag[-1, j], bdiag[-1, j]
        tot = max(w + b, 1e-300)
        print(f"  {n:16s}{gbar[-1, j]:12.4f}{sd[-1, j]:11.4f}"
              f"{sd[-1, j]/max(abs(gbar[-1, j]),1e-30)*100:7.1f}%"
              f"{100*w/tot:8.1f}%{100*b/tot:9.1f}%  "
              f"[{bd[n]['tr']['lo95'][-1]:.4f}, {bd[n]['tr']['hi95'][-1]:.4f}]")
    print("\n  validation:")
    for _, r in val.iterrows():
        print(f"    {r['check']:<62s} {r['result']:.3e}")
    print(f"\nwrote wp8h_coefficients.{{csv,nc}}, wp8h_transfer_matrix.{{csv,nc}},")
    print(f"      wp8h_uncertainty_decomposition.csv, wp8h_provenance.json, "
          f"wp8h_validation.csv, figures  ->  {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
