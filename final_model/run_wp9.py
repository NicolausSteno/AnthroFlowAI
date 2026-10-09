#!/usr/bin/env python3
"""
run_wp9.py — WP-9: Chapter 1 hypothesis tests
=============================================

Chapter 1 (`ch1_macro_materials.tex`) derives the comparative dynamics of a
capital--material growth model and of metal intensity of use, and calibrates
it to the same Rostek et al. (2022) zinc data the UDE is fitted to.  It has
never been confronted with an estimated cycle.  WP-9 does the confrontation.

The hypotheses are pre-registered in `zinc_ch1_lab.HYPOTHESES`, fixed in
source before any number here was computed.  Confirmations and contradictions
are reported with equal prominence (spec, WP-9).

    H1  the transfer coefficients are technology, invariant to economics
    H2  faster growth dilutes the stock-based recovery loop
    H3  the recovery loop collapses into one scalar multiplier
    H4  in-use and discard stocks respond with opposite sign
    H5  general-equilibrium amplification is negligible for one metal
    H6  energy rations throughput proportionally, so it shifts no composition
    H7  metal intensity of use is constant on the sustainable path
    C1  CONTROL — Ch1 leaves the fabrication-loss sign ambiguous
    C2  CONTROL — urban mining is worthless without dilution (not testable)

No refits.  Two inputs: the `anchor_v4` weight dumps in `analysis/wp2a/`, and
one `jax.jacfwd` per seed over 13 driver level shifts and 13 driver
growth-rate rotations, cached to `analysis/wp9/ch1_seed{N}.npz`.  H1 also
reads WP-8a's attribution table and WP-8d's permutation ranking, so the
coefficient-level invariance test and the Chapter 1-parameter-level test are
both reported and can disagree.

    python run_wp9.py --check
    python run_wp9.py --all
    python run_wp9.py --sens --seeds 0,1,2      # just refresh the cache
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import zinc_ch1_lab as CH1
import zinc_circ_lab as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "analysis")
SEED_DIR = os.path.join(OUT_DIR, "wp9")

# Which ensemble the estimators below read.  `anchor_v4` by default — every
# WP-9 number in `wp9_findings.md` was produced with these values untouched.
# `run_wp9_cp.py` repoints them at the `learn_cp: true` arm so that arm is
# scored by *this* code rather than by a second copy of it (WP-9 close-out).
WEIGHTS_DIR = CH1.WEIGHTS_DIR_DEFAULT
CFG = None                      # None -> `CH1.load_anchor_config()`


def set_arm(seed_dir, weights_dir, cfg=None):
    """Point the sensitivity cache and the weight source at another arm."""
    global SEED_DIR, WEIGHTS_DIR, CFG
    SEED_DIR, WEIGHTS_DIR, CFG = seed_dir, weights_dir, cfg
    return dict(seed_dir=SEED_DIR, weights_dir=WEIGHTS_DIR)

COL = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00", "#56B4E9"]
GREY = "#555555"

# Ch1's own zinc calibration, for the reference lines.
CH1_CALIB = {"lam": CH1.CH1_ZINC["lam"], "sig": CH1.CH1_ZINC["sig"],
             "kap": CH1.CH1_ZINC["kap"], "rho": CH1.CH1_ZINC["rho"],
             "chi": CH1.CH1_ZINC_STEADY["chi_star"]}
CH1_PARAM_LABEL = {"lam": r"$\lambda$  end-of-life rate",
                   "sig": r"$\sigma$  fabrication loss",
                   "kap": r"$\kappa$  durable-use share",
                   "rho": r"$\rho$  recycling coefficient",
                   "chi": r"$\chi$  circularity index"}


def hl(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (np.nan, np.nan, np.nan)
    d = C.hodges_lehmann(v)
    return (float(d["hl"]), float(d["lo"]), float(d["hi"]))


def med_iqr(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (np.nan, np.nan, np.nan)
    return (float(np.median(v)), float(np.percentile(v, 25)),
            float(np.percentile(v, 75)))


def survives(lo, hi):
    return bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))


def trend_slope_weights(t):
    """OLS-on-time weights `w` such that `slope = sum(w * y)`."""
    t = np.asarray(t, float)
    tc = t - t.mean()
    return tc / np.sum(tc ** 2)


def spearman(a, b):
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    return float(stats.spearmanr(a[ok], b[ok]).statistic)


# ===========================================================================
# sensitivity cache
# ===========================================================================

def sens_path(seed):
    return os.path.join(SEED_DIR, f"ch1_seed{seed}.npz")


def compute_sens(seeds, *, force=False, verbose=True):
    """One `jax.jacfwd` per seed, cached.  ~25 s per seed on this machine."""
    os.makedirs(SEED_DIR, exist_ok=True)
    todo = [s for s in seeds if force or not os.path.exists(sens_path(s))]
    if not todo:
        return
    import jax
    import zinc_colloc_v5 as v5

    CH1.install(v5)
    cfg = CFG or CH1.load_anchor_config()
    fit, ctx = CH1.build_context(cfg, WEIGHTS_DIR)
    for k, s in enumerate(todo):
        t0 = time.time()
        params = CH1.load_params(s, WEIGHTS_DIR)
        out = CH1.sensitivity(fit, params, ctx)
        np.savez_compressed(
            sens_path(s),
            years=out["years"], drivers=np.array(out["drivers"], dtype=object),
            labels=np.array(out["labels"], dtype=object),
            n_drivers=out["n_drivers"], growth_unit=out["growth_unit"],
            t_mid=out["t_mid"], sd=out["sd"],
            S=out["S"], F=out["F"], dS=out["dS"], dF=out["dF"],
            gap_S=out["gap_S"], gap_F=out["gap_F"], gap_exog=out["gap_exog"])
        jax.clear_caches()                       # CLAUDE.md rule 6
        if verbose:
            print(f"  seed {s:2d}  {time.time()-t0:6.1f}s  "
                  f"({k+1}/{len(todo)})  RSS {CH1._rss_mb():.0f} MB")


def load_sens(seed):
    d = np.load(sens_path(seed), allow_pickle=True)
    return dict(years=d["years"], drivers=list(d["drivers"]),
                labels=list(d["labels"]), n_drivers=int(d["n_drivers"]),
                growth_unit=float(d["growth_unit"]), sd=d["sd"],
                S=d["S"], F=d["F"], dS=d["dS"], dF=d["dF"],
                gap_S=float(d["gap_S"]), gap_F=float(d["gap_F"]),
                gap_exog=float(d["gap_exog"]))


def seed_table(seeds):
    """Per seed: the Ch1 parameters, their driver Jacobian, and the derived
    scalars every hypothesis reads."""
    rows = []
    for s in seeds:
        z = load_sens(s)
        par = CH1.ch1_params_from_traj(z["S"], z["F"])
        jac = CH1.ch1_params_jacobian(z["S"], z["F"], z["dS"], z["dF"])
        dec = CH1.chi_decomposition(par, jac)
        rows.append(dict(seed=s, z=z, par=par, jac=jac, dec=dec))
    return rows


# ===========================================================================
# H1 — invariance
# ===========================================================================

def run_h1(tab, seeds):
    """Two levels: the UDE coefficient slots (WP-8a / WP-8d) and Chapter 1's
    own parameters (this package's Jacobian)."""
    drivers = tab[0]["z"]["drivers"]
    nD = tab[0]["z"]["n_drivers"]

    # --- level (b): Ch1's parameters -------------------------------------
    rows = []
    for pk, block in (("level", 0), ("growth", nD)):
        for j, dn in enumerate(drivers):
            for key in ("rho", "lam", "sig", "kap", "chi"):
                v = []
                for r in tab:
                    base = float(np.mean(r["par"][key]))
                    v.append(float(np.mean(r["jac"][key][:, block + j])) / base)
                e, lo, hi = hl(v)
                m, q1, q3 = med_iqr(v)
                rows.append(dict(
                    level="ch1_parameter", perturbation=pk, driver=dn,
                    quantity=key,
                    ude_status=("learned" if key in ("rho", "lam", "chi")
                                else "pinned"),
                    median=m, q1=q1, q3=q3, hl=e, hl_lo=lo, hl_hi=hi,
                    survives=survives(lo, hi)))
    ch1_par = pd.DataFrame(rows)

    # --- level (a): UDE coefficient slots, from WP-8a and WP-8d ----------
    a_path = os.path.join(OUT_DIR, "wp8a_attribution.csv")
    coef_rows = []
    if os.path.exists(a_path):
        a = pd.read_csv(a_path)
        a = a[a["order"].astype(str) == "sum"]
        learned = ["alpha_cc", "alpha_refc", "alpha_win", "alpha_dr",
                   "tau_olds", "frac_fu_new", "frac_eu_new",
                   "f_cohort_10yr", "f_cohort_20yr", "f_cohort_44yr"]
        for meth, g in a.groupby("method"):
            gl = g[g["coefficient"].isin(learned)]
            coef_rows.append(dict(
                level="ude_coefficient", source="wp8a", method=meth,
                n_cells=int(len(gl)),
                share_surviving=float(gl["survives"].mean()),
                nominal_rate=0.05))
    d_path = os.path.join(OUT_DIR, "wp8d_permutation.csv")
    if os.path.exists(d_path):
        d = pd.read_csv(d_path)
        if "family" in d.columns and "survives" in d.columns:
            for fam, g in d.groupby("family"):
                coef_rows.append(dict(
                    level="ude_coefficient", source="wp8d",
                    method=f"permutation::{fam}", n_cells=int(len(g)),
                    share_surviving=float(g["survives"].mean()),
                    nominal_rate=0.05))
    coef = pd.DataFrame(coef_rows)
    return ch1_par, coef


# ===========================================================================
# H2 — growth dilution
# ===========================================================================

def run_h2(tab, seeds):
    """`dchi / dg_throughput`, comparable with Ch1's own `dchi/dg = -4.87`.

    A `+1 pp/yr` growth rotation of driver `j` is not a `+1 pp/yr` change in
    the growth rate of aggregate throughput, so the raw response has to be
    expressed per unit of induced throughput growth.  Dividing driver by
    driver is a bad estimator here: `anchor_v4` runs with `learn_cp: false`,
    so the scale of the cycle is anchored to the observed ILZSG concentrate
    series and a single driver moves throughput growth by only ~1e-4/yr per
    +1 pp/yr — several of the thirteen denominators sit at or across zero and
    the ratio explodes.  Chapter 1's prediction is one scalar, so it is
    estimated as one: an origin regression of `dchi/dtheta_j` on
    `dg_throughput/dtheta_j` across the growth directions, per seed.  The
    per-driver responses are still reported, descriptively, in the same table.
    """
    drivers = tab[0]["z"]["drivers"]
    nD = tab[0]["z"]["n_drivers"]
    years = tab[0]["z"]["years"][1:]
    w = trend_slope_weights(years)
    ch1_dchi_dg = CH1.ch1_chi_derivs(CH1.CH1_ZINC)["numeric"]["g"]
    block_idx = [j for j, dn in enumerate(drivers)
                 if dn in CH1.ACTIVITY_DRIVERS or dn == CH1.POPULATION_DRIVER]

    # --- per-driver descriptive table ------------------------------------
    rows, per_seed = [], []
    for j, dn in enumerate(drivers):
        raw, dg = [], []
        for r in tab:
            col = nD + j                                  # growth block
            d_chi = float(np.mean(r["jac"]["chi"][:, col]))
            d_g = float(np.sum(w * r["jac"]["throughput"][:, col]
                               / r["par"]["throughput"]))
            raw.append(d_chi)
            dg.append(d_g)
            per_seed.append(dict(driver=dn, seed=r["seed"],
                                 dchi_per_pp=d_chi, dgthr_per_pp=d_g))
        er, lor, hir = hl(raw)
        m, q1, q3 = med_iqr(raw)
        mg, _, _ = med_iqr(dg)
        rows.append(dict(
            driver=dn,
            ch1_block=("activity (g)" if dn in CH1.ACTIVITY_DRIVERS else
                       "population (n)" if dn == CH1.POPULATION_DRIVER else
                       "energy" if dn == CH1.ENERGY_DRIVER else "other"),
            dchi_per_pp_median=m, dchi_per_pp_q1=q1, dchi_per_pp_q3=q3,
            dchi_per_pp_hl=er, dchi_per_pp_lo=lor, dchi_per_pp_hi=hir,
            dgthr_per_pp_median=mg,
            sign_negative=bool(er < 0), survives=survives(lor, hir)))
    per_driver = pd.DataFrame(rows)

    # --- the pooled estimator, which is the H2 headline -------------------
    # Three channels, because Ch1's -4.87 is a *fixed-coefficient* statement:
    # the total response, the dilution term alone (the only one Ch1's closure
    # admits) and the coefficient adaptation it rules out.  Comparing the
    # total against Ch1's number would be comparing two different quantities.
    pooled = []
    for label, idx in (("all 13 drivers", list(range(nD))),
                       ("Ch1 activity + population block", block_idx)):
        for channel in ("total", "dilution only (Ch1's channel)",
                        "coefficient adaptation (excluded by Ch1)"):
            slopes, r2s = [], []
            for r in tab:
                x = np.array([float(np.sum(w * r["jac"]["throughput"][:, nD + j]
                                           / r["par"]["throughput"]))
                              for j in idx])
                tot = np.array([float(np.mean(r["jac"]["chi"][:, nD + j]))
                                for j in idx])
                dil = np.array([float(np.mean(r["dec"]["dilution"][:, nD + j]))
                                for j in idx])
                y = (tot if channel == "total"
                     else dil if channel.startswith("dilution")
                     else tot - dil)
                b, r2 = CH1.ols_through_origin(x, y)
                slopes.append(b)
                r2s.append(r2)
            e, lo, hi = hl(slopes)
            m, q1, q3 = med_iqr(slopes)
            mr, _, _ = med_iqr(r2s)
            pooled.append(dict(
                subset=label, channel=channel, n_directions=len(idx),
                dchi_dg_median=m, dchi_dg_q1=q1, dchi_dg_q3=q3,
                dchi_dg_hl=e, dchi_dg_lo=lo, dchi_dg_hi=hi, r2_median=mr,
                ch1_predicted=ch1_dchi_dg,
                ratio_to_ch1=e / ch1_dchi_dg if ch1_dchi_dg else np.nan,
                sign_negative=bool(e < 0), survives=survives(lo, hi),
                ch1_in_interval=bool(np.isfinite(lo)
                                     and lo <= ch1_dchi_dg <= hi),
                hit=bool(survives(lo, hi) and e < 0)))
    return per_driver, pd.DataFrame(pooled), pd.DataFrame(per_seed), ch1_dchi_dg


def run_decomposition(tab):
    """How much of `dchi/ddriver` is the dilution channel Chapter 1 allows,
    and how much is the coefficient adaptation it rules out."""
    drivers = tab[0]["z"]["drivers"]
    nD = tab[0]["z"]["n_drivers"]
    rows = []
    for pk, block in (("level", 0), ("growth", nD)):
        for j, dn in enumerate(drivers):
            shares = {k: [] for k in ("drho", "dsigma", "dlambda", "dilution")}
            recon = []
            for r in tab:
                t = {k: float(np.mean(r["dec"][k][:, block + j]))
                     for k in shares}
                tot = sum(abs(x) for x in t.values())
                for k in shares:
                    shares[k].append(abs(t[k]) / tot if tot > 0 else np.nan)
                direct = float(np.mean(r["jac"]["chi"][:, block + j]))
                formula = float(np.mean(r["dec"]["ch1_formula_total"][:, block + j]))
                recon.append(abs(formula - direct)
                             / max(abs(direct), 1e-12))
            row = dict(perturbation=pk, driver=dn)
            for k in shares:
                m, q1, q3 = med_iqr(shares[k])
                row[f"share_{k}"] = m
                row[f"share_{k}_q1"] = q1
                row[f"share_{k}_q3"] = q3
            row["ch1_admissible_share"] = row["share_dilution"]
            row["ch1_excluded_share"] = 1.0 - row["share_dilution"]
            row["formula_vs_direct_rel_gap"] = med_iqr(recon)[0]
            rows.append(row)
    return pd.DataFrame(rows)


# ===========================================================================
# H3 — the scalar multiplier
# ===========================================================================

def run_h3(tab):
    """Across drivers, Ch1 Eq. `ext_master_elasticities` says
    `dln s/ddriver = M * dchi/ddriver` with one `M` for every driver."""
    nD = tab[0]["z"]["n_drivers"]
    mult = CH1.ch1_multiplier(CH1.CH1_ZINC)
    # Chapter 1's multiplier evaluated at the circularity the UDE actually
    # measures, not at Ch1's steady-state chi* — the fair comparison, since
    # 1/(1-chi) is steeply increasing and the sample sits below chi*.
    chi_meas = float(np.median([np.mean(r["par"]["chi"]) for r in tab]))
    mult_meas = mult["ge_amplification"] / (1.0 - chi_meas)
    years = tab[0]["z"]["years"][1:]
    late = years >= 2000.0
    rows, per_seed = [], []
    for pk, block in (("level", 0), ("growth", nD)):
        slopes, r2s = [], []
        for r in tab:
            x = np.array([np.mean(r["jac"]["chi"][:, block + j])
                          for j in range(nD)])
            y = np.array([np.mean(r["jac"]["throughput"][:, block + j]
                                  / r["par"]["throughput"])
                          for j in range(nD)])
            b, r2 = CH1.ols_through_origin(x, y)
            slopes.append(b)
            r2s.append(r2)
            # robustness: drop the highest-leverage driver, and restrict to
            # 2000-2019.  A one-SD *additive* level shift is a very large
            # relative move on a series that grew by orders of magnitude
            # (China Total Manufacturing Output), so the early years sit far
            # outside the range where the linearisation is meaningful.
            drop = int(np.argmax(np.abs(x)))
            keep = np.array([j for j in range(nD) if j != drop])
            b_d, r2_d = CH1.ols_through_origin(x[keep], y[keep])
            xl = np.array([np.mean(r["jac"]["chi"][late, block + j])
                           for j in range(nD)])
            yl = np.array([np.mean(r["jac"]["throughput"][late, block + j]
                                   / r["par"]["throughput"][late])
                           for j in range(nD)])
            b_l, r2_l = CH1.ols_through_origin(xl, yl)
            per_seed.append(dict(perturbation=pk, seed=r["seed"],
                                 slope=b, r2=r2,
                                 slope_drop_max_leverage=b_d, r2_drop=r2_d,
                                 dropped_driver=tab[0]["z"]["drivers"][drop],
                                 slope_2000_2019=b_l, r2_2000_2019=r2_l))
        e, lo, hi = hl(slopes)
        m, q1, q3 = med_iqr(slopes)
        mr, qr1, qr3 = med_iqr(r2s)
        rows.append(dict(
            perturbation=pk, slope_median=m, slope_q1=q1, slope_q3=q3,
            slope_hl=e, slope_lo=lo, slope_hi=hi,
            r2_median=mr, r2_q1=qr1, r2_q3=qr3,
            ch1_slope_full=mult["total"],
            ch1_slope_mass_balance=mult["mass_balance"],
            ch1_ge_amplification=mult["ge_amplification"],
            chi_measured=chi_meas,
            ch1_slope_at_measured_chi=mult_meas,
            ch1_slope_in_interval=bool(np.isfinite(lo)
                                       and lo <= mult["total"] <= hi),
            ch1_slope_at_measured_chi_in_interval=bool(
                np.isfinite(lo) and lo <= mult_meas <= hi)))
    ps = pd.DataFrame(per_seed)
    for r_ in rows:
        g = ps[ps["perturbation"] == r_["perturbation"]]
        r_["slope_drop_max_leverage_median"] = float(
            g["slope_drop_max_leverage"].median())
        r_["r2_drop_max_leverage_median"] = float(g["r2_drop"].median())
        r_["slope_2000_2019_median"] = float(g["slope_2000_2019"].median())
        r_["r2_2000_2019_median"] = float(g["r2_2000_2019"].median())
        r_["dropped_driver"] = g["dropped_driver"].mode().iloc[0]
    return pd.DataFrame(rows), ps


# ===========================================================================
# H4 — in-use versus discard
# ===========================================================================

def run_h4(tab):
    nD = tab[0]["z"]["n_drivers"]
    el = CH1.ch1_elasticities(CH1.CH1_ZINC)["rho"]
    rows, per_seed = [], []
    for pk, block in (("level", 0), ("growth", nD)):
        rho_s, gap_u, gap_d = [], [], []
        for r in tab:
            du = np.array([np.mean(r["jac"]["inuse"][:, block + j]
                                   / r["par"]["inuse"]) for j in range(nD)])
            dd = np.array([np.mean(r["jac"]["discard"][:, block + j]
                                   / r["par"]["discard"]) for j in range(nD)])
            ds = np.array([np.mean(r["jac"]["throughput"][:, block + j]
                                   / r["par"]["throughput"]) for j in range(nD)])
            drho = np.array([np.mean(r["jac"]["rho"][:, block + j])
                             for j in range(nD)])
            rbar = float(np.mean(r["par"]["rho"]))
            sp = spearman(du, dd)
            rho_s.append(sp)
            # Ch1: dln u - dln s = 0 ; dln d - dln s = -drho/(1-rho)
            gap_u.append(float(np.mean(np.abs(du - ds))
                               / max(np.mean(np.abs(ds)), 1e-12)))
            pred_d = -drho / (1.0 - rbar)
            gap_d.append(float(np.mean(np.abs((dd - ds) - pred_d))
                               / max(np.mean(np.abs(pred_d)), 1e-12)))
            per_seed.append(dict(perturbation=pk, seed=r["seed"],
                                 spearman_u_vs_d=sp))
        e, lo, hi = hl(rho_s)
        m, q1, q3 = med_iqr(rho_s)
        rows.append(dict(
            perturbation=pk, spearman_median=m, spearman_q1=q1,
            spearman_q3=q3, spearman_hl=e, spearman_lo=lo, spearman_hi=hi,
            ch1_predicted_sign="negative",
            hit=bool(survives(lo, hi) and e < 0),
            ch1_dlnu_drho=el["dlnu"], ch1_dlnd_drho=el["dlnd"],
            ch1_dlns_drho=el["dlns"],
            rel_gap_u_vs_s=med_iqr(gap_u)[0],
            rel_gap_d_vs_prediction=med_iqr(gap_d)[0]))
    return pd.DataFrame(rows), pd.DataFrame(per_seed)


# ===========================================================================
# H5 — GE amplification, and the chi gap
# ===========================================================================

def run_h5(tab):
    rows = []
    g = CH1.CH1_PRODUCTION_GRID
    for al in g["alpha"]:
        for nu in g["nu"]:
            for th in g["vartheta"]:
                prod = dict(alpha=al, nu=nu, vartheta=th)
                m = CH1.ch1_multiplier(CH1.CH1_ZINC, prod)
                rows.append(dict(
                    alpha=al, nu=nu, vartheta=th,
                    central=bool(al == CH1.CH1_PRODUCTION["alpha"]
                                 and nu == CH1.CH1_PRODUCTION["nu"]
                                 and th == CH1.CH1_PRODUCTION["vartheta"]),
                    chi=m["chi"], mass_balance=m["mass_balance"],
                    ge_amplification=m["ge_amplification"],
                    total=m["total"],
                    ge_share_pct=100.0 * (m["ge_amplification"] - 1.0)
                    / m["ge_amplification"]))
    grid = pd.DataFrame(rows)

    # Where the level gap between Ch1's chi* and the measured chi comes from.
    # Ch1: chi = rho*(sigma + lambda*u/s).  The steady-state u/s is
    # (1-sigma)*kappa/(n+lambda); the observed one is not there yet.
    rows = []
    for r in tab:
        p = r["par"]
        ratio = float(np.mean(p["inuse"] / p["throughput"]))
        pm = dict(CH1.CH1_ZINC)
        pm.update(lam=float(np.mean(p["lam"])), sig=float(np.mean(p["sig"])),
                  kap=float(np.mean(p["kap"])), rho=float(np.mean(p["rho"])))
        rows.append(dict(
            seed=r["seed"], chi_measured=float(np.mean(p["chi"])),
            ratio_measured=ratio, ratio_ch1_steady=CH1.ch1_BU(pm),
            chi_at_measured_ratio=float(np.mean(p["rho"]))
            * (float(np.mean(p["sig"])) + float(np.mean(p["lam"])) * ratio),
            chi_at_ch1_steady_ratio=CH1.ch1_chi(pm),
            saturation=ratio / CH1.ch1_BU(pm)))
    gap = pd.DataFrame(rows)
    return grid, gap


# ===========================================================================
# H6 — energy neutrality
# ===========================================================================

def run_h6(tab):
    drivers = tab[0]["z"]["drivers"]
    nD = tab[0]["z"]["n_drivers"]
    j = drivers.index(CH1.ENERGY_DRIVER)
    rows = []
    for pk, block in (("level", 0), ("growth", nD)):
        for key in ("chi", "rho"):
            v = [float(np.mean(r["jac"][key][:, block + j])) for r in tab]
            rel = [x / float(np.mean(r["par"][key]))
                   for x, r in zip(v, tab)]
            e, lo, hi = hl(rel)
            m, q1, q3 = med_iqr(rel)
            rows.append(dict(
                perturbation=pk, quantity=key, driver=CH1.ENERGY_DRIVER,
                rel_response_median=m, q1=q1, q3=q3,
                hl=e, hl_lo=lo, hl_hi=hi,
                ch1_maintained="zero (proportional rationing, Eq. Phi_energy)",
                ch1_alternative="positive (cost-minimising sourcing, the "
                                "footnote to Eq. Phi_energy)",
                rejects_zero=survives(lo, hi),
                favours=("alternative" if survives(lo, hi) and e > 0 else
                         "neither — sign is negative" if survives(lo, hi)
                         else "maintained (cannot reject zero)")))
    return pd.DataFrame(rows)


# ===========================================================================
# H7 — metal intensity of use
# ===========================================================================

def run_h7(tab, ctx_drivers):
    """Elasticity of the in-use stock to GDP, and the trend in `I_U = u/y`."""
    t_gdp, gdp_full = ctx_drivers["gdp"]
    years = tab[0]["z"]["years"]
    gdp = np.interp(years, t_gdp, gdp_full)
    x = np.log(gdp)
    rows = []
    for r in tab:
        u = r["z"]["S"][:, 2]
        y = np.log(u)
        A = np.c_[x, np.ones_like(x)]
        b, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ b
        r2 = 1.0 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)
        iu = u / gdp
        w = trend_slope_weights(years)
        rows.append(dict(seed=r["seed"], elasticity=float(b[0]), r2=float(r2),
                         iu_log_trend_per_yr=float(np.sum(w * np.log(iu))),
                         iu_first=float(iu[0]), iu_last=float(iu[-1])))
    per_seed = pd.DataFrame(rows)
    e, lo, hi = hl(per_seed["elasticity"])
    m, q1, q3 = med_iqr(per_seed["elasticity"])
    te, tlo, thi = hl(per_seed["iu_log_trend_per_yr"])
    verdict = ("intensity-sustainable (elasticity indistinguishable from 1)"
               if lo <= 1.0 <= hi else
               "relative dematerialisation (elasticity < 1, Ch1 vartheta < 1)"
               if hi < 1.0 else
               "intensifying (elasticity > 1)")
    summ = pd.DataFrame([dict(
        quantity="elasticity of in-use stock to GDP",
        median=m, q1=q1, q3=q3, hl=e, hl_lo=lo, hl_hi=hi,
        ch1_predicted=1.0, contains_one=bool(lo <= 1.0 <= hi),
        iu_trend_hl=te, iu_trend_lo=tlo, iu_trend_hi=thi,
        verdict=verdict)])
    return summ, per_seed, dict(years=years, gdp=gdp)


# ===========================================================================
# figures
# ===========================================================================

def fig_parameters(tab, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    years = tab[0]["z"]["years"][1:]
    keys = ["lam", "sig", "kap", "rho", "chi"]
    fig, axes = plt.subplots(1, 5, figsize=(16.5, 3.3))
    for ax, k, c in zip(axes, keys, COL):
        M = np.array([r["par"][k] for r in tab])
        med = np.median(M, axis=0)
        ax.fill_between(years, np.percentile(M, 25, axis=0),
                        np.percentile(M, 75, axis=0), color=c, alpha=0.25,
                        lw=0)
        ax.plot(years, med, color=c, lw=2.0)
        ax.axhline(CH1_CALIB[k], color=GREY, ls="--", lw=1.4)
        lo, hi = ax.get_ylim()
        above = (CH1_CALIB[k] - lo) / (hi - lo) > 0.8   # keep clear of the title
        ax.annotate(f"Ch1 {CH1_CALIB[k]:.3g}", (years[1], CH1_CALIB[k]),
                    textcoords="offset points",
                    xytext=(2, -11 if above else 4),
                    fontsize=8, color=GREY)
        ax.set_title(CH1_PARAM_LABEL[k]
                     + ("\n(pinned to data in anchor_v4)" if k == "kap"
                        else ""), fontsize=10)
        ax.set_xlabel("year")
        ax.margins(x=0.02)
    fig.supylabel("UDE ensemble, median and IQR (35 seeds)", fontsize=9)
    fig.suptitle("WP-9  Chapter 1's structural parameters, measured on the "
                 "estimated cycle", fontsize=11)
    fig.tight_layout(rect=(0.012, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


def fig_dilution(tab, h2d, h2p, decomp, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.2),
                                   gridspec_kw=dict(width_ratios=[1.0, 1.15]))

    d = h2d.sort_values("dchi_per_pp_hl")
    ypos = np.arange(len(d))
    cmap = {"activity (g)": COL[0], "population (n)": COL[2],
            "energy": COL[4], "other": GREY}
    cols = [cmap[b] for b in d["ch1_block"]]
    ax1.hlines(ypos, d["dchi_per_pp_lo"], d["dchi_per_pp_hi"], color=cols,
               lw=2.2)
    ax1.scatter(d["dchi_per_pp_hl"], ypos, color=cols, s=34, zorder=3)
    ax1.axvline(0.0, color="k", lw=0.9)
    ax1.set_yticks(ypos)
    ax1.set_yticklabels([s if len(s) < 34 else s[:31] + "..."
                         for s in d["driver"]], fontsize=8)
    ax1.set_xlabel(r"$\partial\chi/\partial g_j$   per +1 pp/yr on driver $j$"
                   "\n(HL estimate, 95% interval)")
    sub = h2p[h2p["subset"] == "all 13 drivers"]
    dil = sub[sub["channel"].str.startswith("dilution")].iloc[0]
    tot = sub[sub["channel"] == "total"].iloc[0]
    ax1.set_title("H2  growth dilution of circularity\n"
                  r"pooled $\partial\chi/\partial g_{\mathrm{thr}}$:  "
                  f"dilution {dil['dchi_dg_hl']:+.2f} "
                  f"[{dil['dchi_dg_lo']:+.2f}, {dil['dchi_dg_hi']:+.2f}],  "
                  f"net {tot['dchi_dg_hl']:+.2f} "
                  f"[{tot['dchi_dg_lo']:+.2f}, {tot['dchi_dg_hi']:+.2f}]\n"
                  f"Ch1 predicts {dil['ch1_predicted']:+.2f}", fontsize=9)
    for lab, c in cmap.items():
        ax1.plot([], [], color=c, lw=2.2, label=lab)
    ax1.legend(fontsize=8, loc="lower right", frameon=False)

    g = decomp[decomp["perturbation"] == "growth"].set_index("driver")
    g = g.loc[list(d["driver"])]
    keys = ["dilution", "drho", "dlambda", "dsigma"]
    labs = [r"dilution  $\rho\lambda\,d(u/s)$   — Ch1's channel",
            r"$(\sigma+\lambda u/s)\,d\rho$", r"$\rho(u/s)\,d\lambda$",
            r"$\rho\,d\sigma$"]
    V = np.array([g[f"share_{k}"].to_numpy() for k in keys])
    V = V / V.sum(axis=0, keepdims=True)     # medians do not sum to 1 by
                                             # themselves; renormalised here
    left = np.zeros(len(g))
    for v, lab, c in zip(V, labs, [COL[1], COL[2], COL[3], COL[4]]):
        ax2.barh(ypos, v, left=left, color=c, label=lab, height=0.72)
        left = left + v
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([])
    ax2.set_xlim(0, 1)
    ax2.set_xlabel(r"share of $|\partial\chi/\partial\theta|$"
                   "   (per-term medians, renormalised)")
    ax2.set_title("decomposition of the circularity response\n"
                  "(growth perturbation; Ch1 admits only the first term)",
                  fontsize=10)
    ax2.legend(fontsize=8, loc="upper center", frameon=False,
               bbox_to_anchor=(0.5, -0.13), ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


def fig_multiplier(tab, h3, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nD = tab[0]["z"]["n_drivers"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=False)
    for ax, (pk, block) in zip(axes, (("level", 0), ("growth", nD))):
        xs, ys = [], []
        for r in tab:
            x = np.array([np.mean(r["jac"]["chi"][:, block + j])
                          for j in range(nD)])
            y = np.array([np.mean(r["jac"]["throughput"][:, block + j]
                                  / r["par"]["throughput"])
                          for j in range(nD)])
            xs.append(x)
            ys.append(y)
        X, Y = np.concatenate(xs), np.concatenate(ys)
        ax.scatter(X, Y, s=10, color=COL[0], alpha=0.35, lw=0)
        row = h3[h3["perturbation"] == pk].iloc[0]
        lim = np.array([X.min(), X.max()])
        ax.plot(lim, row["ch1_slope_full"] * lim, color=GREY, ls="--", lw=1.8,
                label=f"Ch1  slope {row['ch1_slope_full']:.2f}")
        ax.plot(lim, row["ch1_slope_at_measured_chi"] * lim, color=GREY,
                ls=":", lw=1.8,
                label=(f"Ch1 at the measured $\\chi$="
                       f"{row['chi_measured']:.2f}  slope "
                       f"{row['ch1_slope_at_measured_chi']:.2f}"))
        ax.plot(lim, row["slope_median"] * lim, color=COL[2], lw=1.8,
                label=(f"measured  slope {row['slope_median']:.2f}"
                       f"  $R^2$ {row['r2_median']:.3f}"))
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel(r"$\partial\chi/\partial\theta$")
        ax.set_ylabel(r"$\partial\ln s/\partial\theta$")
        ax.set_title(f"{pk} perturbation", fontsize=10)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("WP-9  H3  does the recovery loop collapse into one scalar "
                 "multiplier?   (13 drivers x 35 seeds)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


def fig_intensity(tab, h7, h7_seed, aux, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    years, gdp = aux["years"], aux["gdp"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    x = np.log(gdp)
    for r in tab:
        ax1.plot(x, np.log(r["z"]["S"][:, 2]), color=COL[0], alpha=0.12, lw=1.0)
    med = np.median(np.array([np.log(r["z"]["S"][:, 2]) for r in tab]), axis=0)
    ax1.plot(x, med, color=COL[0], lw=2.2, label="UDE ensemble median")
    b = float(h7["median"].iloc[0])
    ref = med[0] + 1.0 * (x - x[0])
    ax1.plot(x, ref, color=GREY, ls="--", lw=1.8,
             label=r"Ch1 intensity path, slope $=1$")
    ax1.set_xlabel("ln GDP")
    ax1.set_ylabel("ln in-use zinc stock")
    ax1.set_title(f"H7  measured elasticity {b:.2f} "
                  f"[{h7['hl_lo'].iloc[0]:.2f}, {h7['hl_hi'].iloc[0]:.2f}]",
                  fontsize=10)
    ax1.legend(fontsize=8, frameon=False)

    IU = np.array([r["z"]["S"][:, 2] / gdp for r in tab])
    IU = IU / IU[:, :1]
    ax2.fill_between(years, np.percentile(IU, 25, axis=0),
                     np.percentile(IU, 75, axis=0), color=COL[1], alpha=0.25,
                     lw=0)
    ax2.plot(years, np.median(IU, axis=0), color=COL[1], lw=2.2)
    ax2.axhline(1.0, color=GREY, ls="--", lw=1.4)
    ax2.set_xlabel("year")
    ax2.set_ylabel(r"$\mathcal{I}_U(t)/\mathcal{I}_U(1980)$")
    ax2.set_title("metal intensity of use, indexed to 1980\n"
                  r"(Ch1 Eq. oc_IU_def; a flat line is $\mathcal{I}_U$ constant)",
                  fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


def fig_extensions(tab, h9_mult, h10s, h11, path):
    """The two appendix extensions: what substitutability is worth on the
    measured cycle, and where the recovery technology sits on its own curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    years = tab[0]["z"]["years"][1:]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.8))

    # --- pi: the realised circularity multiplier ------------------------
    CHI = np.array([r["par"]["chi"] for r in tab])
    chi_med = np.median(CHI, axis=0)
    pis = np.linspace(0.0, 1.0, 101)
    chi_bar = float(np.median(CHI.mean(axis=1)))
    ax1.plot(pis, CH1.ch1_X_multiplier(chi_bar, pis), color=COL[0], lw=2.2)
    ax1.fill_between(pis,
                     CH1.ch1_X_multiplier(np.percentile(CHI.mean(axis=1), 25),
                                          pis),
                     CH1.ch1_X_multiplier(np.percentile(CHI.mean(axis=1), 75),
                                          pis),
                     color=COL[0], alpha=0.2, lw=0)
    ax1.plot(pis, CH1.ch1_X_multiplier(float(chi_med[0]), pis), color=GREY,
             ls=":", lw=1.4, label="at the 1981 $\\chi$")
    ax1.plot(pis, CH1.ch1_X_multiplier(float(chi_med[-1]), pis), color=GREY,
             ls="--", lw=1.4, label="at the 2019 $\\chi$")
    ax1.axhline(1.0, color="k", lw=0.7)
    ax1.set_xlabel(r"substitutability $\pi$")
    ax1.set_ylabel(r"$\mathcal{X}(\chi,\pi)=1+\pi\chi/(1-\chi)$")
    ax1.set_title("H9  the circularity dividend is\nquality-weighted "
                  "(Ch1 App. Eq. pi_Xcal_def)", fontsize=10)
    ax1.legend(fontsize=8, frameon=False, loc="upper left")

    # --- pi identifiability ---------------------------------------------
    r11 = h11.iloc[0]
    lo_pi = CH1.ch1_slope_with_pi(chi_bar, 0.0)
    hi_pi = CH1.ch1_slope_with_pi(chi_bar, 1.0)
    ax2.axhspan(lo_pi, hi_pi, color=COL[2], alpha=0.55,
                label=r"everything $\pi\in[0,1]$ can move")
    nD = tab[0]["z"]["n_drivers"]
    ms = []
    for r in tab:
        x = np.array([np.mean(r["jac"]["chi"][:, j]) for j in range(nD)])
        y = np.array([np.mean(r["jac"]["throughput"][:, j]
                              / r["par"]["throughput"]) for j in range(nD)])
        ms.append(CH1.ols_through_origin(x, y)[0])
    ms = np.array(ms)
    ax2.axhspan(np.percentile(ms, 25), np.percentile(ms, 75), color=COL[0],
                alpha=0.25, label="measured slope, seed IQR")
    ax2.axhline(np.median(ms), color=COL[0], lw=2.0)
    ax2.set_xticks([])
    ax2.set_ylabel(r"$\partial\ln s/\partial\chi$")
    ax2.set_title("H9  $\\pi$ is not identifiable:\nits whole range is "
                  "narrower than the ensemble", fontsize=10)
    ax2.legend(fontsize=8, frameon=False, loc="lower right")

    # --- rho: where the recovery technology sits ------------------------
    RHO = np.array([r["par"]["rho"] for r in tab])
    COLL = np.array([CH1.collectable_flow(r["par"]) for r in tab])
    m = np.median(COLL, axis=0) / np.median(COLL, axis=0)[0]
    ax3.fill_between(m, np.percentile(RHO, 25, axis=0),
                     np.percentile(RHO, 75, axis=0), color=COL[1], alpha=0.25,
                     lw=0)
    ax3.plot(m, np.median(RHO, axis=0), color=COL[1], lw=2.2,
             label="measured $\\rho$")
    r10 = h10s.iloc[0]
    for c, lab in ((r10["ceiling_hill"], "ceiling (Hill fit)"),
                   (r10["ceiling_exponential"], "ceiling (exponential fit)")):
        if np.isfinite(c):
            ax3.axhline(c, color=GREY, ls="--", lw=1.3)
            ax3.annotate(f"{lab}  {c:.2f}", (m[0], c),
                         textcoords="offset points", xytext=(2, 3),
                         fontsize=8, color=GREY)
    ax3.set_xlabel("collectable flow, indexed to 1981\n"
                   r"($\lambda u+\sigma s$, Ch1 App. Eq. rho_s_rho)")
    ax3.set_ylabel(r"recovery rate $\rho$")
    ax3.set_title(f"H10  entirely on the saturated branch\n"
                  f"min $m/m_{{\\mathrm{{infl}}}}$ = "
                  f"{r10['min_m_over_m_infl']:.2f} > 1  "
                  f"({r10['proposition'].split('—')[0].strip()})", fontsize=10)
    ax3.legend(fontsize=8, frameon=False, loc="lower right")

    fig.suptitle("WP-9  the two Chapter 1 appendix extensions "
                 "(app_ch1_trans_dyn.tex)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


# ===========================================================================
# main
# ===========================================================================

# ===========================================================================
# H8 — balanced growth
# ===========================================================================

def run_h8(tab, ctx_drivers):
    """Ch1's BGP restriction `g_s = g_u = g_d = vartheta*g_k` in two forms: a
    common elasticity to GDP, and — the GDP-free version Ch1 itself uses for
    copper in App. Sec. `j6_numerical` — trendless stock-flow ratios."""
    t_gdp, gdp_full = ctx_drivers["gdp"]
    years = tab[0]["z"]["years"][1:]
    gdp = np.interp(years, t_gdp, gdp_full)
    x = np.log(gdp)
    A = np.c_[x, np.ones_like(x)]
    w = trend_slope_weights(years)

    keys = ("throughput", "inuse", "discard")
    el = {k: [] for k in keys}
    ratios = {"u_over_s": [], "d_over_s": []}
    trends = {"u_over_s": [], "d_over_s": []}
    for r in tab:
        for k in keys:
            b, *_ = np.linalg.lstsq(A, np.log(r["par"][k]), rcond=None)
            el[k].append(float(b[0]))
        for nm, num in (("u_over_s", "inuse"), ("d_over_s", "discard")):
            v = r["par"][num] / r["par"]["throughput"]
            ratios[nm].append(float(np.mean(v)))
            trends[nm].append(float(np.sum(w * np.log(v))))

    rows = []
    for k in keys:
        e, lo, hi = hl(el[k])
        m, q1, q3 = med_iqr(el[k])
        rows.append(dict(quantity=f"elasticity of {k} to GDP", median=m,
                         q1=q1, q3=q3, hl=e, hl_lo=lo, hl_hi=hi,
                         ch1_predicted="a common vartheta"))
    for a, b_ in (("throughput", "inuse"), ("discard", "throughput")):
        d = np.array(el[a]) - np.array(el[b_])
        e, lo, hi = hl(d)
        m, q1, q3 = med_iqr(d)
        rows.append(dict(quantity=f"elasticity gap: {a} - {b_}", median=m,
                         q1=q1, q3=q3, hl=e, hl_lo=lo, hl_hi=hi,
                         ch1_predicted=0.0, equal=bool(not survives(lo, hi))))
    for nm, ch1_ref in (("u_over_s", CH1.ch1_BU(CH1.CH1_ZINC)),
                        ("d_over_s", CH1.ch1_BD(CH1.CH1_ZINC))):
        e, lo, hi = hl(ratios[nm])
        m, q1, q3 = med_iqr(ratios[nm])
        rows.append(dict(quantity=f"{nm} (years of throughput)", median=m,
                         q1=q1, q3=q3, hl=e, hl_lo=lo, hl_hi=hi,
                         ch1_predicted=ch1_ref))
        e, lo, hi = hl(trends[nm])
        m, q1, q3 = med_iqr(trends[nm])
        rows.append(dict(quantity=f"trend in ln({nm}), per yr", median=m,
                         q1=q1, q3=q3, hl=e, hl_lo=lo, hl_hi=hi,
                         ch1_predicted=0.0, equal=bool(not survives(lo, hi))))
    # The discard stock's 1980 level cannot be reconstructed from in-sample
    # flows, and every `d` statistic inherits that choice.  Reported rather
    # than buried: `s` and `u` are anchor-free, so the BGP rejection rests on
    # them; the `d` numbers are conditional on Ch1's own `d/u` ratio.
    for anch, lab in ((0.0, "0 (no pre-1980 discard stock)"),
                      (CH1.DISCARD_ANCHOR_RATIO, "Ch1 d/u = 0.759 (used)"),
                      (2 * CH1.DISCARD_ANCHOR_RATIO, "2x Ch1")):
        el_d, tr_d = [], []
        for r in tab:
            par = CH1.ch1_params_from_traj(r["z"]["S"], r["z"]["F"],
                                           discard_anchor_ratio=anch)
            b, *_ = np.linalg.lstsq(A, np.log(par["discard"]), rcond=None)
            el_d.append(float(b[0]))
            tr_d.append(float(np.sum(w * np.log(par["discard"]
                                                / par["throughput"]))))
        rows.append(dict(quantity=f"[anchor sensitivity] elasticity of "
                                  f"discard to GDP, anchor = {lab}",
                         median=float(np.median(el_d)),
                         hl=hl(el_d)[0], hl_lo=hl(el_d)[1], hl_hi=hl(el_d)[2],
                         ch1_predicted="a common vartheta"))
        rows.append(dict(quantity=f"[anchor sensitivity] trend in "
                                  f"ln(d_over_s), anchor = {lab}",
                         median=float(np.median(tr_d)),
                         hl=hl(tr_d)[0], hl_lo=hl(tr_d)[1], hl_hi=hl(tr_d)[2],
                         ch1_predicted=0.0))
    return pd.DataFrame(rows)


# ===========================================================================
# H9 — substitutability between virgin and recycled material
# ===========================================================================

def run_h9(tab, h3):
    """The appendix's `pi` is a valuation parameter: Ch1 App. Sec.
    `pi_extension` states that the tonnage balance and the stock equations are
    untouched by it.  A mass-flow model therefore cannot see it directly, and
    the question is quantitative — how much of an observable does `pi` move,
    against how precisely that observable is measured?"""
    chi = float(np.median([np.mean(r["par"]["chi"]) for r in tab]))
    row = h3[h3["perturbation"] == "level"].iloc[0]
    lo_pi = CH1.ch1_slope_with_pi(chi, 0.0)
    hi_pi = CH1.ch1_slope_with_pi(chi, 1.0)
    span = abs(hi_pi - lo_pi)
    ci = abs(row["slope_hi"] - row["slope_lo"])

    ident = pd.DataFrame([dict(
        chi_measured=chi,
        h3_slope_at_pi0=lo_pi, h3_slope_at_pi1=hi_pi,
        pi_span_of_prediction=span,
        measured_slope=row["slope_median"],
        measured_ci_width=ci,
        ci_over_span=ci / span if span else np.nan,
        identified=bool(span > ci),
        verdict=("pi identifiable from the throughput response"
                 if span > ci else
                 "pi NOT identifiable: its whole effect is smaller than the "
                 "ensemble's own interval"))])

    # the realised circularity multiplier, evaluated on the measured chi
    per_year = np.array([r["par"]["chi"] for r in tab])
    grid = []
    for pi in (0.0, 0.25, 0.50, 0.75, 1.0):
        X = CH1.ch1_X_multiplier(per_year, pi)
        Xm = np.median(X, axis=0)
        grid.append(dict(pi=pi,
                         X_at_mean_chi=float(CH1.ch1_X_multiplier(chi, pi)),
                         service_dividend_pct=float(
                             100 * (CH1.ch1_X_multiplier(chi, pi) - 1.0)),
                         X_1981=float(Xm[0]), X_2019=float(Xm[-1]),
                         overstatement_factor_vs_pi1=(1.0 / pi if pi > 0
                                                      else np.inf)))
    mult = pd.DataFrame(grid)

    # a physically measurable proxy for the quality margin: which of the two
    # old-scrap re-entry routes the secondary supply comes back through.
    import zinc_colloc_v5 as v5
    idr = v5.FLOW_NAMES.index("direct_reuse_recycling")
    iwz = v5.FLOW_NAMES.index("waelz_recycling")
    shares = np.array([r["z"]["F"][:, idr]
                       / (r["z"]["F"][:, idr] + r["z"]["F"][:, iwz])
                       for r in tab])
    years = tab[0]["z"]["years"][1:]
    w = trend_slope_weights(years)
    tr = [float(np.sum(w * np.log(sh))) for sh in shares]
    e, lo, hi = hl(tr)
    route = pd.DataFrame([dict(
        quantity="direct-reuse share of secondary supply",
        share_1981=float(np.median(shares[:, 0])),
        share_2019=float(np.median(shares[:, -1])),
        mean=float(np.median(shares.mean(axis=1))),
        log_trend_per_yr=e, trend_lo=lo, trend_hi=hi,
        note="the two old-scrap re-entry routes differ physically in quality "
             "(direct re-use bypasses remelting, Waelz does not), so their "
             "mix is the closest observable analogue of Ch1's pi; assigning "
             "pi values to them would be assumption, not measurement")])
    return ident, mult, route


# ===========================================================================
# H10 — recovery capacity, and which branch of the Hill technology
# ===========================================================================

def run_h10(tab):
    """Ch1 App. Prop. 1 (`h <= 1`) and Prop. 2 (`h > 1`) are mutually
    exclusive, and the appendix is explicit that what decides the regime is not
    `h` alone but where the system sits relative to the inflection scale
    `m_infl` (Eq. `rho_inflection`) — its own copper example has `h = 2` yet
    sits "well onto the saturated branch" and so behaves like the fixed-recovery
    model.  Three independent readings are taken."""
    from scipy.optimize import curve_fit

    curv, H, RMAX, FRAC, MRATIO, R2, CEIL = [], [], [], [], [], [], []
    for r in tab:
        par = r["par"]
        years = r["z"]["years"][1:]
        coll = CH1.collectable_flow(par)

        # (a) local curvature of rho in the log collectable flow
        lx = np.log(coll)
        lx = (lx - lx.mean()) / lx.std()
        b, *_ = np.linalg.lstsq(np.c_[np.ones_like(lx), lx, lx ** 2],
                                par["rho"], rcond=None)
        curv.append(float(b[2]))

        # (b) the Hill technology itself, with the collectable flow as the
        #     stated observable proxy for the recovery bundle.  The scale
        #     normalisation is absorbed by `mbar`, so `h` and the branch
        #     location are invariant to it.
        m = coll / coll[0]
        try:
            p, _ = curve_fit(lambda mm, rmax, mbar, h: CH1.ch1_hill(mm, rmax,
                                                                    mbar, h),
                             m, par["rho"], p0=[0.6, 1.0, 1.0], maxfev=60000,
                             bounds=([0.2, 0.05, 0.05], [0.99, 50.0, 6.0]))
            rmax, mbar, h = p
            pred = CH1.ch1_hill(m, rmax, mbar, h)
            R2.append(1.0 - np.sum((par["rho"] - pred) ** 2)
                      / np.sum((par["rho"] - par["rho"].mean()) ** 2))
            minfl = CH1.ch1_inflection(mbar, h)
            H.append(float(h))
            RMAX.append(float(rmax))
            FRAC.append(float(np.mean(m > minfl)) if minfl > 0 else 1.0)
            MRATIO.append(float(m.min() / minfl) if minfl > 0 else np.inf)
        except Exception:
            pass

        # (c) an assumption-light ceiling: exponential approach in time
        try:
            p, _ = curve_fit(
                lambda t, rinf, r0, kp: rinf - (rinf - r0)
                * np.exp(-kp * (t - years[0])),
                years, par["rho"], p0=[0.5, 0.32, 0.05], maxfev=20000,
                bounds=([0.2, 0.05, 0.001], [1.0, 0.6, 1.0]))
            CEIL.append(float(p[0]))
        except Exception:
            pass

    def block(name, v, extra=None):
        e, lo, hi = hl(v)
        m, q1, q3 = med_iqr(v)
        d = dict(quantity=name, median=m, q1=q1, q3=q3, hl=e, hl_lo=lo,
                 hl_hi=hi)
        d.update(extra or {})
        return d

    cop = CH1.CH1_COPPER_RHO_EXAMPLE
    rows = [
        block("quadratic coefficient of rho on ln(collectable flow)", curv,
              dict(ch1_reading="negative = concave = Prop. 1 branch (h <= 1)")),
        block("fitted Hill exponent h", H,
              dict(ch1_reading="h > 1 admits the trap geometry, but only "
                               "below the inflection scale",
                   ch1_copper_example=cop["h"])),
        block("fitted recovery ceiling rho_max (Hill)", RMAX,
              dict(ch1_copper_example=cop["rho_max"])),
        block("fitted recovery ceiling (exponential approach in time)", CEIL,
              dict(ch1_copper_example=cop["rho_max"])),
        block("share of observed years on the saturated branch", FRAC,
              dict(ch1_reading="1.0 = never in the increasing-returns region")),
        block("min(m)/m_infl over the sample", MRATIO,
              dict(ch1_reading="> 1 = entirely above the inflection scale",
                   ch1_copper_example=cop["m_star"] / cop["m_infl"])),
    ]
    df = pd.DataFrame(rows)
    df.loc[df["quantity"] == "fitted Hill exponent h", "hill_fit_r2_median"] = (
        float(np.median(R2)) if R2 else np.nan)

    rho_last = float(np.median([r["par"]["rho"][-1] for r in tab]))
    summary = pd.DataFrame([dict(
        rho_2019=rho_last,
        hill_exponent_h=float(np.median(H)) if H else np.nan,
        frac_years_saturated=float(np.median(FRAC)) if FRAC else np.nan,
        min_m_over_m_infl=float(np.median(MRATIO)) if MRATIO else np.nan,
        ceiling_hill=float(np.median(RMAX)) if RMAX else np.nan,
        ceiling_exponential=float(np.median(CEIL)) if CEIL else np.nan,
        ratio_to_ceiling_hill=(rho_last / np.median(RMAX)) if RMAX else np.nan,
        ratio_to_ceiling_exponential=(rho_last / np.median(CEIL))
        if CEIL else np.nan,
        ch1_copper_ratio_to_ceiling=cop["ratio_to_ceiling"],
        hill_fit_r2_median=float(np.median(R2)) if R2 else np.nan,
        branch="saturated (concave)" if np.median(FRAC) >= 1.0
        else "partly increasing-returns",
        proposition="App. Prop. 1 — no trap, no multiplicity, no Hopf route"
        if np.median(FRAC) >= 1.0 else "App. Prop. 2 geometry is live")])
    return df, summary


# ===========================================================================
# H11 — the short-run material-balance derivative
# ===========================================================================

def run_h11(tab):
    """Ch1 App. Eq. `rho_s_rho` against the measured pooled response.  The
    prediction carries no free parameters once the UDE's own measured
    `(lambda, sigma, rho, u, s)` are substituted."""
    nD = tab[0]["z"]["n_drivers"]
    slopes, r2s, preds = [], [], []
    for r in tab:
        x = np.array([np.mean(r["jac"]["rho"][:, j]) for j in range(2 * nD)])
        y = np.array([np.mean(r["jac"]["throughput"][:, j]
                              / r["par"]["throughput"])
                      for j in range(2 * nD)])
        b, r2 = CH1.ols_through_origin(x, y)
        slopes.append(b)
        r2s.append(r2)
        preds.append(float(np.mean(CH1.ch1_s_rho(r["par"]))))
    e, lo, hi = hl(slopes)
    m, q1, q3 = med_iqr(slopes)
    pe, plo, phi = hl(preds)
    long_run = CH1.ch1_elasticities(CH1.CH1_ZINC)["rho"]["dlns"]
    return pd.DataFrame([dict(
        quantity="dln s / drho, pooled over 26 perturbation directions",
        median=m, q1=q1, q3=q3, hl=e, hl_lo=lo, hl_hi=hi,
        r2_median=float(np.median(r2s)),
        ch1_short_run=pe, ch1_short_run_lo=plo, ch1_short_run_hi=phi,
        ch1_long_run=long_run,
        short_run_in_interval=bool(lo <= pe <= hi),
        long_run_in_interval=bool(lo <= long_run <= hi),
        between=bool(min(pe, long_run) <= e <= max(pe, long_run)),
        rel_error_vs_short_run=float(abs(e - pe) / abs(pe)))])


def run_decomposition_pooled(tab):
    """The pooled slope of each term of the `chi` decomposition separately —
    which term flips the sign of the net growth response.  Reported because the
    aggregate `dilution / adaptation` split of H2 does not say which of the
    three adaptation terms carries it."""
    nD = tab[0]["z"]["n_drivers"]
    years = tab[0]["z"]["years"][1:]
    w = trend_slope_weights(years)
    rows = []
    for key in ("dilution", "drho", "dlambda", "dsigma"):
        v = []
        for r in tab:
            x = np.array([float(np.sum(w * r["jac"]["throughput"][:, nD + j]
                                       / r["par"]["throughput"]))
                          for j in range(nD)])
            y = np.array([float(np.mean(r["dec"][key][:, nD + j]))
                          for j in range(nD)])
            v.append(CH1.ols_through_origin(x, y)[0])
        e, lo, hi = hl(v)
        m, q1, q3 = med_iqr(v)
        rows.append(dict(term=key, median=m, q1=q1, q3=q3, hl=e, hl_lo=lo,
                         hl_hi=hi, survives=survives(lo, hi)))
    return pd.DataFrame(rows)


def run_validation(tab):
    """Everything WP-9 asserts that can be checked against something already
    on disk, or against the solver."""
    rows = [dict(check="forward integrator vs the fit's own, stocks",
                 quantity="max relative gap",
                 value=max(r["z"]["gap_S"] for r in tab), tolerance=0.0,
                 passes=bool(max(r["z"]["gap_S"] for r in tab) == 0.0)),
            dict(check="forward integrator vs the fit's own, flow integrals",
                 quantity="max relative gap",
                 value=max(r["z"]["gap_F"] for r in tab), tolerance=0.0,
                 passes=bool(max(r["z"]["gap_F"] for r in tab) == 0.0)),
            dict(check="JAX preprocess_exog vs the core's NumPy version",
                 quantity="max absolute gap",
                 value=max(r["z"]["gap_exog"] for r in tab), tolerance=1e-12,
                 passes=bool(max(r["z"]["gap_exog"] for r in tab) < 1e-12))]

    # the level block against WP-8c, which used the identical perturbation
    p8c = os.path.join(OUT_DIR, "wp8c_forward_sens.npz")
    if os.path.exists(p8c):
        d = np.load(p8c, allow_pickle=True)
        seeds8c = {int(s): i for i, s in enumerate(d["seeds"])}
        nD = tab[0]["z"]["n_drivers"]
        worst = 0.0
        n_cmp = 0
        for r in tab:
            i = seeds8c.get(int(r["seed"]))
            if i is None:
                continue
            a = d["dS"][i]                       # (T, 4, 13)
            b = r["z"]["dS"][:, :, :nD]
            sc = max(float(np.max(np.abs(a))), 1e-12)
            worst = max(worst, float(np.max(np.abs(a - b)) / sc))
            n_cmp += 1
        rows.append(dict(
            check=f"level-shift dS vs WP-8c ({n_cmp} shared seeds)",
            quantity="max relative gap", value=worst, tolerance=1e-6,
            passes=bool(worst < 1e-6)))

    # Ch1's equations reproduce Ch1's own tables
    info = CH1.check(verbose=False)
    rows.append(dict(check="Ch1 equations vs Ch1 Tables tab:calibration / "
                           "tab:steady_state_stability",
                     quantity="max absolute gap",
                     value=info["ch1_repro_max_gap"], tolerance=0.1,
                     passes=bool(info["ch1_repro_max_gap"] < 0.1)))
    rows.append(dict(check="closed-form chi_z vs central difference",
                     quantity="max absolute gap",
                     value=info["ch1_chi_deriv_closed_vs_numeric"],
                     tolerance=1e-5,
                     passes=bool(info["ch1_chi_deriv_closed_vs_numeric"] < 1e-5)))
    rows.append(dict(check="ch1_params_jacobian chain rule vs a central "
                           "difference of the solver (best step)",
                     quantity="max relative gap",
                     value=info["jacobian_vs_finite_difference"],
                     tolerance=0.05,
                     passes=bool(info["jacobian_vs_finite_difference"] < 0.05)))
    return pd.DataFrame(rows)


def build_summary(h1_par, h1_coef, h2d, h2p, h2_dchi, h3, h4, h5_grid, h6, h7,
                  decomp, h8=None, h9_id=None, h10s=None, h11=None):
    from scipy import stats

    rows = []
    reg = {h["id"]: h for h in CH1.HYPOTHESES}

    # H1 — binomial against the 5% nominal rate, not an eyeballed threshold
    act = h1_par[(h1_par["level"] == "ch1_parameter")
                 & (h1_par["quantity"].isin(["rho", "lam"]))]
    k, n = int(act["survives"].sum()), int(len(act))
    share = k / n if n else np.nan
    pval = float(stats.binomtest(k, n, 0.05, alternative="greater").pvalue)
    coef_share = np.nan
    if len(h1_coef) and "method" in h1_coef.columns:
        sel = h1_coef[h1_coef["method"] == "ig"]
        if len(sel):
            coef_share = float(sel["share_surviving"].iloc[0])
    rows.append(dict(
        id="H1", hypothesis=reg["H1"]["name"],
        ch1_prediction="0 responses; 0.05 of cells at the nominal rate",
        measured=f"{share:.2f} of Ch1-parameter cells ({k}/{n}, binomial "
                 f"p={pval:.1e} against 0.05); {coef_share:.2f} of UDE "
                 f"coefficient cells (WP-8a, integrated gradients)",
        outcome="CONTRADICTED" if pval < 0.01 else "held"))
    # H2 — the pooled estimator is the headline, by channel
    act = h2d[h2d["ch1_block"].isin(["activity (g)", "population (n)"])]
    n_act = int(len(act))
    n_neg = int((act["dchi_per_pp_hl"] < 0).sum())
    n_act_hit = int(((act["dchi_per_pp_hl"] < 0) & act["survives"]).sum())
    sub = h2p[h2p["subset"] == "all 13 drivers"]
    tot = sub[sub["channel"] == "total"].iloc[0]
    dil = sub[sub["channel"].str.startswith("dilution")].iloc[0]
    adp = sub[sub["channel"].str.startswith("coefficient")].iloc[0]
    rows.append(dict(
        id="H2", hypothesis=reg["H2"]["name"],
        ch1_prediction=f"dchi/dg = {h2_dchi:.2f} < 0 (a fixed-coefficient "
                       f"statement)",
        measured=f"per-driver dchi/dg_j negative for {n_neg}/{n_act} "
                 f"activity+population drivers ({n_act_hit} surviving the "
                 f"ensemble); pooled dilution channel {dil['dchi_dg_hl']:+.2f} "
                 f"[{dil['dchi_dg_lo']:+.2f}, {dil['dchi_dg_hi']:+.2f}] "
                 f"({100*dil['ratio_to_ch1']:.0f}% of Ch1); coefficient "
                 f"adaptation {adp['dchi_dg_hl']:+.2f} "
                 f"[{adp['dchi_dg_lo']:+.2f}, {adp['dchi_dg_hi']:+.2f}]; "
                 f"net {tot['dchi_dg_hl']:+.2f} [{tot['dchi_dg_lo']:+.2f}, "
                 f"{tot['dchi_dg_hi']:+.2f}]",
        outcome=("SIGN CONFIRMED, magnitude an order of magnitude smaller; "
                 "the net is not identified here (learn_cp: false anchors "
                 "throughput, so dg_throughput/dtheta is near zero and of "
                 "mixed sign)"
                 if dil["hit"] and n_neg >= 0.6 * n_act else
                 "CONFIRMED" if dil["hit"] and tot["hit"] else
                 "CONTRADICTED" if not dil["sign_negative"] else
                 "INCONCLUSIVE")))
    # H3
    r = h3[h3["perturbation"] == "level"].iloc[0]
    rows.append(dict(
        id="H3", hypothesis=reg["H3"]["name"],
        ch1_prediction=f"slope {r['ch1_slope_full']:.2f} at Ch1's chi*, "
                       f"{r['ch1_slope_at_measured_chi']:.2f} at the measured "
                       f"chi; R^2 = 1",
        measured=f"slope {r['slope_median']:.2f} "
                 f"[{r['slope_lo']:.2f}, {r['slope_hi']:.2f}], "
                 f"R^2 {r['r2_median']:.3f}",
        outcome=("CONFIRMED" if (r["ch1_slope_at_measured_chi_in_interval"]
                                 and r["r2_median"] > 0.9)
                 else "CONFIRMED in structure, not in level"
                 if r["r2_median"] > 0.9 else
                 "PARTIAL" if r["r2_median"] > 0.5 else "CONTRADICTED")))
    # H4
    r = h4[h4["perturbation"] == "level"].iloc[0]
    rows.append(dict(
        id="H4", hypothesis=reg["H4"]["name"],
        ch1_prediction="negative rank correlation between the in-use and "
                       "discard responses",
        measured=f"Spearman {r['spearman_hl']:+.2f} "
                 f"[{r['spearman_lo']:+.2f}, {r['spearman_hi']:+.2f}]",
        outcome="CONFIRMED" if r["hit"] else "CONTRADICTED"))
    # H5
    c = h5_grid[h5_grid["central"]].iloc[0]
    lo = h5_grid["ge_amplification"].min()
    hi = h5_grid["ge_amplification"].max()
    rows.append(dict(
        id="H5", hypothesis=reg["H5"]["name"],
        ch1_prediction="amplification within a few % of 1",
        measured=f"{c['ge_amplification']:.4f} at the central case "
                 f"(range {lo:.3f}-{hi:.3f} over the grid)",
        outcome="CONFIRMED (analytic)"))
    # H6
    r = h6[(h6["perturbation"] == "level") & (h6["quantity"] == "chi")].iloc[0]
    rows.append(dict(
        id="H6", hypothesis=reg["H6"]["name"],
        ch1_prediction="dchi/d(energy) = 0",
        measured=f"{r['rel_response_median']:+.4f} "
                 f"[{r['hl_lo']:+.4f}, {r['hl_hi']:+.4f}] per one-SD level; "
                 f"favours {r['favours']}",
        outcome=("CONTRADICTED — favours Ch1's own footnoted alternative"
                 if r["favours"] == "alternative" else
                 "held (cannot reject zero)" if not r["rejects_zero"]
                 else "CONTRADICTED — wrong sign for both readings")))
    # H7
    r = h7.iloc[0]
    rows.append(dict(
        id="H7", hypothesis=reg["H7"]["name"],
        ch1_prediction="elasticity = 1",
        measured=f"{r['median']:.2f} [{r['hl_lo']:.2f}, {r['hl_hi']:.2f}]",
        outcome=("CONFIRMED" if r["contains_one"] else f"REFINED — {r['verdict']}")))
    # H8 — balanced growth
    if h8 is not None:
        gap = h8[h8["quantity"].str.startswith("elasticity gap")]
        tru = h8[h8["quantity"] == "trend in ln(u_over_s), per yr"].iloc[0]
        trd = h8[h8["quantity"] == "trend in ln(d_over_s), per yr"].iloc[0]
        els = {q.split()[-3]: v for q, v in
               zip(h8["quantity"], h8["hl"]) if q.startswith("elasticity of")}
        gap = gap[~gap["quantity"].str.startswith("[anchor")]
        n_equal = int(gap["equal"].sum()) if "equal" in gap.columns else 0
        rows.append(dict(
            id="H8", hypothesis=reg["H8"]["name"],
            ch1_prediction="one elasticity shared by s, u and d; trendless "
                           "u/s and d/s",
            measured=f"elasticities to GDP: throughput "
                     f"{els.get('throughput', float('nan')):.2f}, in-use "
                     f"{els.get('inuse', float('nan')):.2f}, discard "
                     f"{els.get('discard', float('nan')):.2f}; u/s drifts "
                     f"{100*tru['hl']:+.2f}%/yr, d/s {100*trd['hl']:+.2f}%/yr",
            outcome=("CONFIRMED" if n_equal == len(gap) else
                     "CONTRADICTED — the cycle is transitional, not balanced "
                     "(the throughput/in-use gap is anchor-free; the discard "
                     "numbers are conditional on the 1980 discard anchor)")))
    # H9 — substitutability
    if h9_id is not None:
        r = h9_id.iloc[0]
        rows.append(dict(
            id="H9", hypothesis=reg["H9"]["name"],
            ch1_prediction="pi is a valuation parameter and cannot be "
                           "identified from a tonnage model",
            measured=f"the whole pi in [0,1] moves the H3 slope by "
                     f"{r['pi_span_of_prediction']:.4f} "
                     f"({r['h3_slope_at_pi0']:.4f} to "
                     f"{r['h3_slope_at_pi1']:.4f}), against a measured "
                     f"interval {r['measured_ci_width']:.4f} wide — "
                     f"{r['ci_over_span']:.1f}x larger",
            outcome=("CONFIRMED — pi is not identifiable from the material "
                     "cycle" if not r["identified"] else
                     "pi is identifiable, contrary to the construction")))
    # H10 — recovery capacity branch
    if h10s is not None:
        r = h10s.iloc[0]
        rows.append(dict(
            id="H10", hypothesis=reg["H10"]["name"],
            ch1_prediction="Prop. 1 (h <= 1, concave) or Prop. 2 (h > 1 below "
                           "the inflection, trap geometry)",
            measured=f"fitted h = {r['hill_exponent_h']:.2f} (> 1), but "
                     f"{100*r['frac_years_saturated']:.0f}% of observed years "
                     f"sit on the saturated branch, min m/m_infl = "
                     f"{r['min_m_over_m_infl']:.2f}; rho 2019 "
                     f"{r['rho_2019']:.3f} against a ceiling of "
                     f"{r['ceiling_hill']:.2f}-{r['ceiling_exponential']:.2f}",
            outcome=("App. Prop. 1 regime — h > 1 is fitted, but the cycle "
                     "never leaves the saturated branch, which Ch1 says "
                     "behaves like the fixed-recovery model; the Skiba "
                     "threshold, circularity trap and Hopf route of App. "
                     "Prop. 2 are not live over 1981-2019"
                     if r["frac_years_saturated"] >= 1.0 else
                     "App. Prop. 2 geometry is live — part of the sample sits "
                     "in the increasing-returns region")))
    # H11 — the short-run derivative
    if h11 is not None:
        r = h11.iloc[0]
        rows.append(dict(
            id="H11", hypothesis=reg["H11"]["name"],
            ch1_prediction=f"dln s/drho = {r['ch1_short_run']:.3f} "
                           f"(App. Eq. rho_s_rho, no free parameters); "
                           f"long-run {r['ch1_long_run']:.3f}",
            measured=f"{r['median']:.3f} [{r['hl_lo']:.3f}, {r['hl_hi']:.3f}], "
                     f"R^2 {r['r2_median']:.3f}; "
                     f"{100*r['rel_error_vs_short_run']:.1f}% from the "
                     f"short-run closed form",
            outcome=("CONFIRMED" if r["short_run_in_interval"]
                     and not r["long_run_in_interval"] else
                     "CONFIRMED but does not separate the two"
                     if r["short_run_in_interval"] else
                     "CONTRADICTED")))
    # controls
    rows.append(dict(
        id="C1", hypothesis=reg["C1"]["name"],
        ch1_prediction="no sign prediction",
        measured="sigma maps to frac_fu_loss / frac_eu_loss, both pinned in "
                 "anchor_v4; the invariance test on them measures the pinning",
        outcome="not testable"))
    rows.append(dict(
        id="C2", hypothesis=reg["C2"]["name"],
        ch1_prediction="chi_gamma -> 0 as n -> 0",
        measured="gamma = 0 in Ch1's zinc baseline; the UDE carries no "
                 "landfill stock",
        outcome="not testable"))
    return pd.DataFrame(rows)


def write_registration(path):
    reg = []
    for h in CH1.HYPOTHESES:
        reg.append(dict(id=h["id"], name=h["name"], ch1_reference=h["ch1_ref"],
                        predicted=h["predicted"] or "(no prediction)",
                        statistic=h["statistic"],
                        ch1_value=h["ch1_value"],
                        stated_alternative=h["alternative"] or ""))
    pd.DataFrame(reg).to_csv(path, index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description="WP-9 Chapter 1 hypothesis tests")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sens", action="store_true",
                    help="(re)compute the per-seed sensitivity cache only")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seeds", default=None)
    a = ap.parse_args(argv)

    if a.check:
        CH1.check()
        return 0

    seeds = ([int(s) for s in a.seeds.split(",") if s.strip()] if a.seeds
             else CH1.seed_list())
    os.makedirs(SEED_DIR, exist_ok=True)

    print(f"[wp9] sensitivities for {len(seeds)} seeds "
          f"(cache {SEED_DIR})")
    t0 = time.time()
    compute_sens(seeds, force=a.force)
    print(f"[wp9] cache ready in {time.time()-t0:.0f}s")
    if a.sens and not a.all:
        return 0

    tab = seed_table(seeds)
    gapS = max(r["z"]["gap_S"] for r in tab)
    gapF = max(r["z"]["gap_F"] for r in tab)
    print(f"[wp9] forward integrator vs the fit's own: max rel gap "
          f"stocks {gapS:.2e}  flows {gapF:.2e}  ({len(tab)} seeds)")

    # Ch1 parameter recovery table
    rows = []
    for k in ("lam", "sig", "kap", "rho", "chi"):
        v = [float(np.mean(r["par"][k])) for r in tab]
        e, lo, hi = hl(v)
        m, q1, q3 = med_iqr(v)
        first = [float(r["par"][k][0]) for r in tab]
        last = [float(r["par"][k][-1]) for r in tab]
        rows.append(dict(parameter=k,
                         ch1_symbol=CH1.CH1_TO_UDE[k]["ch1"]
                         if k in CH1.CH1_TO_UDE else "chi — circularity index",
                         ude_median=m, ude_q1=q1, ude_q3=q3,
                         ude_hl=e, ude_lo=lo, ude_hi=hi,
                         ude_1981=float(np.median(first)),
                         ude_2019=float(np.median(last)),
                         ch1_calibrated=CH1_CALIB[k],
                         rel_gap_pct=100.0 * (m - CH1_CALIB[k]) / CH1_CALIB[k],
                         ch1_in_iqr=bool(q1 <= CH1_CALIB[k] <= q3)))
    params_df = pd.DataFrame(rows)
    params_df.to_csv(os.path.join(OUT_DIR, "wp9_ch1_parameters.csv"),
                     index=False)
    print("\n[wp9] Chapter 1's parameters, measured on the estimated cycle")
    print(params_df[["parameter", "ude_median", "ude_q1", "ude_q3",
                     "ch1_calibrated", "rel_gap_pct", "ch1_in_iqr"]]
          .to_string(index=False, float_format=lambda x: f"{x:8.4f}"))

    write_registration(os.path.join(OUT_DIR, "wp9_registration.csv"))

    h1_par, h1_coef = run_h1(tab, seeds)
    h1_par.to_csv(os.path.join(OUT_DIR, "wp9_invariance.csv"), index=False)
    if len(h1_coef):
        h1_coef.to_csv(os.path.join(OUT_DIR, "wp9_invariance_coefficients.csv"),
                       index=False)

    h2d, h2p, h2_seed, h2_dchi = run_h2(tab, seeds)
    h2d.to_csv(os.path.join(OUT_DIR, "wp9_dilution.csv"), index=False)
    h2p.to_csv(os.path.join(OUT_DIR, "wp9_dilution_pooled.csv"), index=False)
    h2_seed.to_csv(os.path.join(OUT_DIR, "wp9_dilution_per_seed.csv"),
                   index=False)

    decomp = run_decomposition(tab)
    decomp.to_csv(os.path.join(OUT_DIR, "wp9_chi_decomposition.csv"),
                  index=False)

    h3, h3_seed = run_h3(tab)
    h3.to_csv(os.path.join(OUT_DIR, "wp9_multiplier.csv"), index=False)
    h3_seed.to_csv(os.path.join(OUT_DIR, "wp9_multiplier_per_seed.csv"),
                   index=False)

    h4, h4_seed = run_h4(tab)
    h4.to_csv(os.path.join(OUT_DIR, "wp9_stock_asymmetry.csv"), index=False)

    h5_grid, h5_gap = run_h5(tab)
    h5_grid.to_csv(os.path.join(OUT_DIR, "wp9_ge_amplification.csv"),
                   index=False)
    h5_gap.to_csv(os.path.join(OUT_DIR, "wp9_chi_level_gap.csv"), index=False)

    h6 = run_h6(tab)
    h6.to_csv(os.path.join(OUT_DIR, "wp9_energy.csv"), index=False)

    cfg = CH1.load_anchor_config()
    fit, ctx = CH1.build_context(cfg)
    h7, h7_seed, aux = run_h7(tab, dict(gdp=CH1.driver_series(ctx,
                                                             CH1.GDP_DRIVER)))
    h7.to_csv(os.path.join(OUT_DIR, "wp9_intensity_of_use.csv"), index=False)
    h7_seed.to_csv(os.path.join(OUT_DIR, "wp9_intensity_of_use_per_seed.csv"),
                   index=False)

    val = run_validation(tab)
    val.to_csv(os.path.join(OUT_DIR, "wp9_validation.csv"), index=False)
    print("\n[wp9] validation")
    print(val.to_string(index=False))

    h8 = run_h8(tab, dict(gdp=CH1.driver_series(ctx, CH1.GDP_DRIVER)))
    h8.to_csv(os.path.join(OUT_DIR, "wp9_balanced_growth.csv"), index=False)

    h9_id, h9_mult, h9_route = run_h9(tab, h3)
    h9_id.to_csv(os.path.join(OUT_DIR, "wp9_substitutability.csv"), index=False)
    h9_mult.to_csv(os.path.join(OUT_DIR, "wp9_circularity_multiplier.csv"),
                   index=False)
    h9_route.to_csv(os.path.join(OUT_DIR, "wp9_secondary_route_mix.csv"),
                    index=False)

    h10, h10s = run_h10(tab)
    h10.to_csv(os.path.join(OUT_DIR, "wp9_recovery_capacity.csv"), index=False)
    h10s.to_csv(os.path.join(OUT_DIR, "wp9_recovery_branch.csv"), index=False)

    h11 = run_h11(tab)
    h11.to_csv(os.path.join(OUT_DIR, "wp9_s_rho.csv"), index=False)

    decomp_pooled = run_decomposition_pooled(tab)
    decomp_pooled.to_csv(os.path.join(OUT_DIR,
                                      "wp9_chi_decomposition_pooled.csv"),
                         index=False)

    summ = build_summary(h1_par, h1_coef, h2d, h2p, h2_dchi, h3, h4, h5_grid,
                         h6, h7, decomp, h8, h9_id, h10s, h11)
    summ.to_csv(os.path.join(OUT_DIR, "wp9_hypothesis_table.csv"), index=False)
    print("\n[wp9] hypothesis table")
    with pd.option_context("display.max_colwidth", 62, "display.width", 200):
        print(summ.to_string(index=False))

    fig_parameters(tab, os.path.join(OUT_DIR, "wp9_ch1_parameters"))
    fig_dilution(tab, h2d, h2p, decomp,
                 os.path.join(OUT_DIR, "wp9_dilution"))
    fig_multiplier(tab, h3, os.path.join(OUT_DIR, "wp9_multiplier"))
    fig_intensity(tab, h7, h7_seed, aux,
                  os.path.join(OUT_DIR, "wp9_intensity_of_use"))
    fig_extensions(tab, h9_mult, h10s, h11,
                   os.path.join(OUT_DIR, "wp9_extensions"))

    info = CH1.check(verbose=False)
    info.update(n_seeds=len(seeds), max_gap_S=gapS, max_gap_F=gapF,
                outcomes=dict(zip(summ["id"], summ["outcome"])))
    with open(os.path.join(OUT_DIR, "wp9_check.json"), "w") as fh:
        json.dump({k: v for k, v in info.items() if k != "digests"}, fh,
                  indent=2, default=str)
    print(f"\n[wp9] wrote analysis/wp9_*.{{csv,png,pdf}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
