#!/usr/bin/env python3
"""
run_wp7.py — WP-7: counterfactuals
==================================

Aggregates the per-seed counterfactual re-integrations written by
`zinc_cf_lab.py` into the three results the spec asks for:

  1. **Price path.**  Hold the real zinc price on its 1990s trend through
     2008–2019 — divergence in In-Use stock and in secondary supply.
  2. **Policy intervention.**  Step `tau_olds` up 10 pp from 2005 — the
     trajectory, and the **time-to-effect** in secondary supply, which is not
     imposed anywhere: it emerges from mass balance (the scrap pool has to
     fill before it can be drawn down) and from the cohort structure (the
     extra metal that reaches the use phase only returns after 10 / 20 / 44
     years).
  3. **Shock removal.**  Zero the 2008 driver shock and decompose the observed
     downstream change into a driver-explained part and a residual.

Everything is reported with seed-ensemble bands (median ± IQR over 35 seeds),
with Hodges–Lehmann point estimates and distribution-free CIs on the headline
effects.

Three conventions that the numbers depend on
--------------------------------------------
**Secondary supply** is `waelz_recycling + direct_reuse_recycling` — the mass
re-entering production from the scrap pool through Chapter 2's two old-scrap
routes, α₁₄ and α₁₃.  Both are period integrals over (Y−1, Y] taken from the
integrator's flow accumulators, never point-sampled (spec §1).
`old_scrap_recovery` is reported separately: it flows *into* the scrap pool,
so counting it as supply would double-count.

**Effects are differences against a matched reference**, not against the data.
Driver arms are differenced against `factual`, which reproduces the stored
`anchor_v4` rollout bit-for-bit; the `tau_olds` arms are differenced against
`factual_tau_ref`, an identical no-intervention run through the same
intervened closure.  `factual − factual_tau_ref` is not zero — the two agree
bitwise at the RHS but the adaptive solver takes different steps — and its
size is carried through this script as the **numerical noise floor**, so every
effect can be read against the smallest one that means anything.

**The shock decomposition is done on changes from 2007**, not on levels.  The
free-run carries ~27% stock relRMSE accumulated over 27 years (spec §1), which
would swamp a level-space residual and has nothing to do with 2008.  Since the
intervention starts in 2008, factual and counterfactual coincide through 2007,
so with `Δx(Y) = x(Y) − x(2007)`,

    Δobs − Δcf  =  (Δobs − Δfac)  +  (fac − cf)(Y)
                    residual         driver-explained

exactly, and the driver-explained term is the counterfactual effect itself.

    python run_wp7.py --check
    python run_wp7.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DUMP_DIR = os.path.join(HERE, "analysis", "wp7")
OUT_DIR = os.path.join(HERE, "analysis")

STOCK_NAMES = ["Concentrate", "Refined", "In-Use", "Scrap"]
SECONDARY = ("waelz_recycling", "direct_reuse_recycling")

# Flows carried through the reporting tables.  `secondary_supply` is derived.
FLOW_VARS = ["secondary_supply", "waelz_recycling", "direct_reuse_recycling",
             "old_scrap_recovery", "end_of_life", "inuse_inflow",
             "primary_refining", "refined_consumption"]

BASE_YEAR = 2007.0          # last year before every intervention window
SHOCK_YEARS = (2008.0, 2010.0)

# Okabe–Ito, colourblind-safe.
COL = {"factual": "#000000", "price_trend90s": "#0072B2",
       "price_trend90s_anchored": "#56B4E9", "price_flat90s": "#009E73",
       "tau_olds_step10_2005": "#D55E00", "tau_olds_step05_2005": "#E69F00",
       "shock2008_bridge": "#CC79A7", "shock2008_bridge3": "#8B5A9E"}
GREY = "#555555"


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def med_iqr(v):
    v = np.asarray([x for x in np.asarray(v, float) if np.isfinite(x)])
    if v.size == 0:
        return dict(median=np.nan, q1=np.nan, q3=np.nan, n=0)
    q1, q3 = np.percentile(v, [25, 75])
    return dict(median=float(np.median(v)), q1=float(q1), q3=float(q3),
                n=int(v.size))


def _walsh(d):
    w = (d[:, None] + d[None, :]) / 2.0
    return np.sort(w[np.triu_indices_from(w)])


def hl_location(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(_walsh(x))) if x.size else float("nan")


def hl_ci(x, conf=0.95):
    """Distribution-free CI for the HL location estimate (Walsh-average order
    statistics with the normal approximation to the signed-rank null)."""
    from math import sqrt

    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return float("nan"), float("nan")
    w = _walsh(x)
    z = {0.90: 1.6448536, 0.95: 1.9599640, 0.99: 2.5758293}.get(conf, 1.9599640)
    k = max(int(np.floor(w.size / 2.0
                         - z * sqrt(n * (n + 1) * (2 * n + 1) / 24.0))), 0)
    return float(w[k]), float(w[w.size - 1 - k])


# ---------------------------------------------------------------------------
# loading and variable extraction
# ---------------------------------------------------------------------------
def load_dumps(dump_dir=DUMP_DIR):
    paths = sorted(glob.glob(os.path.join(dump_dir, "cf_seed*.npz")),
                   key=lambda p: int(os.path.basename(p)[7:-4]))
    if not paths:
        raise SystemExit(f"no dumps in {dump_dir} — run zinc_cf_lab.py first")
    return [(int(os.path.basename(p)[7:-4]), np.load(p, allow_pickle=True))
            for p in paths]


def _key(arm):
    return arm.replace("::", "__")


def arm_table(d):
    """`{arm: spec}` with the reference each arm is differenced against."""
    return json.loads(str(d["arm_specs"]))


def series(d, arm, var):
    """One (variable, arm) time series with its year axis.

    Stocks are point-in-time on `years`; flows are period integrals on
    `years[1:]`, row *i* being the interval (years[i], years[i+1]].  The two
    are never mixed on one axis.
    """
    years = np.asarray(d["years"], float).ravel()
    k = _key(arm)
    if var in STOCK_NAMES:
        return years, np.asarray(d[f"S__{k}"], float)[:, STOCK_NAMES.index(var)]
    fn = [str(x) for x in d["flow_names"]]
    F = np.asarray(d[f"F__{k}"], float)
    if var == "secondary_supply":
        y = sum(F[:, fn.index(f)] for f in SECONDARY)
    elif var in fn:
        y = F[:, fn.index(var)]
    else:
        raise KeyError(f"unknown variable {var!r}")
    return years[1:], y


def observed(d, var):
    """The reported series for the same variable, on the same axis."""
    years = np.asarray(d["years"], float).ravel()
    if var in STOCK_NAMES:
        return years, np.asarray(d["stocks_obs"], float)[:, STOCK_NAMES.index(var)]
    fn = [str(x) for x in d["flow_names"]]
    idx = np.asarray(d["flow_obs_to_pred_idx"], int)
    obs_names = [fn[i] for i in idx]
    O = np.asarray(d["flows_obs"], float)
    if var == "secondary_supply":
        if not all(f in obs_names for f in SECONDARY):
            return years[1:], np.full(len(years) - 1, np.nan)
        y = sum(O[:, obs_names.index(f)] for f in SECONDARY)
    elif var in obs_names:
        y = O[:, obs_names.index(var)]
    else:
        return years[1:], np.full(len(years) - 1, np.nan)
    return years[1:], y


ALL_VARS = STOCK_NAMES + FLOW_VARS


# ---------------------------------------------------------------------------
# effects
# ---------------------------------------------------------------------------
def effects_by_year(dumps, arms):
    """Per (arm, variable, year): counterfactual minus its matched reference,
    in kt and in per cent, summarised across seeds."""
    rows = []
    for arm, spec in arms.items():
        ref = spec.get("ref")
        if ref is None:
            continue
        for var in ALL_VARS:
            diffs, pcts, refs = [], [], []
            for _seed, d in dumps:
                t, y = series(d, arm, var)
                _t, y0 = series(d, ref, var)
                diffs.append(y - y0)
                pcts.append(100.0 * (y - y0) / np.where(np.abs(y0) > 1e-12,
                                                        y0, np.nan))
                refs.append(y0)
            D = np.stack(diffs); P = np.stack(pcts); R = np.stack(refs)
            for j, yr in enumerate(t):
                sd = med_iqr(D[:, j]); sp = med_iqr(P[:, j])
                rows.append(dict(
                    arm=arm, ref=ref, variable=var, year=float(yr),
                    ref_median=float(np.median(R[:, j])),
                    diff_kt_median=sd["median"], diff_kt_q1=sd["q1"],
                    diff_kt_q3=sd["q3"],
                    diff_pct_median=sp["median"], diff_pct_q1=sp["q1"],
                    diff_pct_q3=sp["q3"], n_seeds=sd["n"]))
    return pd.DataFrame(rows)


def variable_noise_floor(dumps, ref_a="factual", ref_b="factual_tau_ref"):
    """Per-variable numerical noise floor.

    `factual` and `factual_tau_ref` are the same model with the same weights on
    the same data and agree bitwise at the RHS; they differ only in the step
    sequence the adaptive solver chooses (see `zinc_cf_lab.make_cf_machinery`).
    Their separation is therefore the resolution limit of every counterfactual
    in this package, and it has to be read per variable and in the same units
    as the effect it is being compared against — a floor quoted as a single
    max-over-stocks kt number would be dominated by whichever stock is largest.
    """
    out = {}
    for var in ALL_VARS:
        kt, pct = [], []
        for _seed, d in dumps:
            _t, a = series(d, ref_a, var)
            _t, b = series(d, ref_b, var)
            kt.append(float(np.max(np.abs(a - b))))
            pct.append(float(np.max(np.abs(100.0 * (a - b)
                                           / np.where(np.abs(a) > 1e-12,
                                                      a, np.nan)))))
        out[var] = dict(kt=float(np.median(kt)), pct=float(np.median(pct)),
                        kt_max=float(np.max(kt)), pct_max=float(np.max(pct)))
    return out


def summary_table(dumps, arms, noise, window=(2008.0, 2019.0)):
    """Headline effect per (arm, variable): terminal, mean over the window,
    and — for flows — the cumulative effect in kt.

    The cumulative flow effect is the physically meaningful one: a flow is
    already a period integral, so summing it over the window totals mass.
    """
    rows = []
    vnf = variable_noise_floor(dumps)
    for arm, spec in arms.items():
        ref = spec.get("ref")
        if ref is None:
            continue
        for var in ALL_VARS:
            term, mean_pct, cum, term_pct = [], [], [], []
            for _seed, d in dumps:
                t, y = series(d, arm, var)
                _t, y0 = series(d, ref, var)
                m = (t >= window[0]) & (t <= window[1])
                dd = y - y0
                pp = 100.0 * dd / np.where(np.abs(y0) > 1e-12, y0, np.nan)
                term.append(dd[-1]); term_pct.append(pp[-1])
                mean_pct.append(float(np.nanmean(pp[m])))
                cum.append(float(np.sum(dd[m])) if var not in STOCK_NAMES
                           else np.nan)
            st = med_iqr(term); sp = med_iqr(term_pct)
            sm = med_iqr(mean_pct); sc = med_iqr(cum)
            lo, hi = hl_ci(np.asarray(term_pct, float))
            rows.append(dict(
                arm=arm, ref=ref, variable=var, label=spec.get("label", ""),
                window=f"{window[0]:.0f}–{window[1]:.0f}",
                terminal_year=float(t[-1]),
                terminal_kt_median=st["median"], terminal_kt_q1=st["q1"],
                terminal_kt_q3=st["q3"],
                terminal_pct_median=sp["median"], terminal_pct_q1=sp["q1"],
                terminal_pct_q3=sp["q3"],
                terminal_pct_HL=hl_location(np.asarray(term_pct, float)),
                terminal_pct_HL_lo=lo, terminal_pct_HL_hi=hi,
                mean_pct_over_window_median=sm["median"],
                cumulative_kt_median=sc["median"], cumulative_kt_q1=sc["q1"],
                cumulative_kt_q3=sc["q3"],
                noise_floor_kt=vnf[var]["kt"], noise_floor_pct=vnf[var]["pct"],
                abs_terminal_over_noise_floor=(
                    abs(sp["median"]) / vnf[var]["pct"]
                    if vnf[var]["pct"] > 0 else np.inf),
                clears_noise_floor=bool(abs(sp["median"])
                                        > vnf[var]["pct_max"]),
                n_seeds=st["n"]))
    return pd.DataFrame(rows)


def noise_floor_table(dumps):
    rows = []
    for seed, d in dumps:
        chk = json.loads(str(d["check_json"]))
        rows.append({k: v for k, v in chk.items() if k != "path"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2 — time to effect
# ---------------------------------------------------------------------------
def time_to_effect(dumps, arms, arm="tau_olds_step10_2005",
                   variables=("secondary_supply", "old_scrap_recovery",
                              "Scrap", "In-Use"),
                   fractions=(0.25, 0.5, 0.9)):
    """Years from the intervention until the response reaches a fraction of
    its terminal (2019) value.

    Terminal, not asymptotic: the record stops 14 years after the 2005 step,
    which is short against the 44-year in-use cohort, so for `In-Use` these
    are fractions of an unconverged response and are labelled as such by
    `terminal_is_asymptote=False`.  `t_x` is the first year at which the
    response has reached the fraction and stays at or above it thereafter, so
    a single early overshoot cannot produce a spuriously short lag.
    """
    spec = arms[arm]
    t0 = float(spec.get("t0", np.nan))
    ref = spec["ref"]
    rows = []
    for var in variables:
        for seed, d in dumps:
            t, y = series(d, arm, var)
            _t, y0 = series(d, ref, var)
            r = y - y0
            # A flow labelled Y is the integral over (Y−1, Y], so the step at
            # t0 leaves the row labelled t0 essentially untouched (it overlaps
            # the intervention on a set of measure zero) and the first
            # intervened interval is the row at t0 + 1.  Stocks are
            # point-in-time and the row at t0 is the intervention instant.
            m = (t > t0) if var not in STOCK_NAMES else (t >= t0)
            tt, rr = t[m], r[m]
            if rr.size == 0 or not np.isfinite(rr[-1]) or rr[-1] == 0.0:
                continue
            frac = rr / rr[-1]
            rec = dict(seed=seed, arm=arm, variable=var,
                       intervention_year=t0, kind=("stock" if var in STOCK_NAMES
                                                   else "flow"),
                       first_response_year=float(tt[0]),
                       terminal_year=float(tt[-1]),
                       terminal_response_kt=float(rr[-1]),
                       first_year_response_kt=float(rr[0]),
                       first_year_frac=float(frac[0]),
                       terminal_is_asymptote=False)
            for f in fractions:
                ok = frac >= f
                # first index from which the threshold holds for good
                hit = np.nan
                if ok.any():
                    idx = np.where(ok)[0]
                    for i in idx:
                        if ok[i:].all():
                            hit = float(tt[i] - t0)
                            break
                rec[f"years_to_{int(100*f)}pct"] = hit
            rows.append(rec)
    return pd.DataFrame(rows)


def summarise_tte(tte):
    rows = []
    for (arm, var), g in tte.groupby(["arm", "variable"], sort=False):
        rec = dict(arm=arm, variable=var, n_seeds=int(g.seed.nunique()),
                   intervention_year=float(g.intervention_year.iloc[0]))
        for c in ["terminal_response_kt", "first_year_response_kt",
                  "first_year_frac", "years_to_25pct", "years_to_50pct",
                  "years_to_90pct"]:
            st = med_iqr(g[c].values)
            rec[f"{c}_median"] = st["median"]
            rec[f"{c}_q1"] = st["q1"]
            rec[f"{c}_q3"] = st["q3"]
        rows.append(rec)
    return pd.DataFrame(rows)


def linearity_check(dumps, arms, big="tau_olds_step10_2005",
                    small="tau_olds_step05_2005"):
    """Ratio of the 10 pp response to the 5 pp response.

    Exactly 2 would mean the system responds linearly in the intervention
    size over this range; departures locate the nonlinearity (α·S with S
    itself moving, plus the manufacturing simplex).
    """
    rows = []
    for var in ("secondary_supply", "old_scrap_recovery", "Scrap", "In-Use"):
        ratios = []
        for _seed, d in dumps:
            t, yb = series(d, big, var)
            _t, ys = series(d, small, var)
            _t, y0 = series(d, arms[big]["ref"], var)
            rb, rs = yb[-1] - y0[-1], ys[-1] - y0[-1]
            if abs(rs) > 1e-9:
                ratios.append(rb / rs)
        st = med_iqr(ratios)
        rows.append(dict(variable=var, ratio_10pp_over_5pp_median=st["median"],
                         q1=st["q1"], q3=st["q3"], n_seeds=st["n"],
                         linear_reference=2.0))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3 — shock decomposition
# ---------------------------------------------------------------------------
def _at(t, y, year):
    i = int(np.argmin(np.abs(np.asarray(t, float) - year)))
    return float(y[i])


def shock_decomposition(dumps, arms, arm="shock2008_bridge",
                        base_year=BASE_YEAR):
    """Observed change since `base_year`, split into driver-explained and
    residual (see module docstring for the identity)."""
    ref = arms[arm]["ref"]
    rows = []
    for var in ALL_VARS:
        for _seed, d in dumps:
            t, y_cf = series(d, arm, var)
            _t, y_fa = series(d, ref, var)
            to, y_ob = observed(d, var)
            if not np.isfinite(y_ob).any():
                continue
            b_fa, b_cf = _at(t, y_fa, base_year), _at(t, y_cf, base_year)
            b_ob = _at(to, y_ob, base_year)
            for j, yr in enumerate(t):
                if yr <= base_year:
                    continue
                io = int(np.argmin(np.abs(to - yr)))
                d_ob = y_ob[io] - b_ob
                d_fa = y_fa[j] - b_fa
                d_cf = y_cf[j] - b_cf
                rows.append(dict(
                    seed=int(_seed), arm=arm, variable=var, year=float(yr),
                    observed_change=d_ob, factual_change=d_fa,
                    counterfactual_change=d_cf,
                    driver_explained=y_fa[j] - y_cf[j],
                    residual=d_ob - d_fa,
                    total_vs_counterfactual=d_ob - d_cf))
    return pd.DataFrame(rows)


def summarise_shock(dec, window=SHOCK_YEARS):
    """Per (variable, year) bands, plus a cumulative share over the shock
    window.  The share is only reported where the denominator is large enough
    to make a ratio meaningful."""
    rows = []
    for (var, yr), g in dec.groupby(["variable", "year"], sort=False):
        rec = dict(variable=var, year=float(yr), n_seeds=int(g.seed.nunique()),
                   observed_change=float(g.observed_change.median()))
        for c in ("factual_change", "counterfactual_change",
                  "driver_explained", "residual", "total_vs_counterfactual"):
            st = med_iqr(g[c].values)
            rec[f"{c}_median"] = st["median"]
            rec[f"{c}_q1"] = st["q1"]
            rec[f"{c}_q3"] = st["q3"]
        rows.append(rec)
    by_year = pd.DataFrame(rows)

    shares = []
    w = dec[(dec.year >= window[0]) & (dec.year <= window[1])]
    for var, g in w.groupby("variable", sort=False):
        is_stock = var in STOCK_NAMES
        if is_stock:
            # A stock is a level: summing its change-since-2007 over the years
            # of the window totals nothing physical, so it is read at the end
            # of the window instead.
            gg = g[g.year == window[1]]
            agg = gg.groupby("seed")[["driver_explained", "residual",
                                      "total_vs_counterfactual",
                                      "observed_change"]].last()
            unit, basis = "kt", f"level at {window[1]:.0f}"
        else:
            # A flow is already a period integral, so summing it over the
            # window totals mass.
            agg = g.groupby("seed")[["driver_explained", "residual",
                                     "total_vs_counterfactual",
                                     "observed_change"]].sum()
            unit, basis = "kt", f"cumulated {window[0]:.0f}–{window[1]:.0f}"
        den = agg["total_vs_counterfactual"].values
        de = agg["driver_explained"].values
        res = agg["residual"].values
        # A signed share is meaningless when the denominator is near zero, and
        # for several downstream variables it is: the observed change and the
        # model's counterfactual change nearly cancel.  Gate it on the
        # denominator being large against the two parts it is made of, and
        # report the unsigned attribution index — which is well defined
        # whatever the denominator does — alongside.
        ok = np.abs(den) > 0.5 * (np.abs(de) + np.abs(res))
        sh = np.where(ok, de / np.where(ok, den, np.nan), np.nan)
        idx = np.abs(de) / np.maximum(np.abs(de) + np.abs(res), 1e-12)
        st, si = med_iqr(sh), med_iqr(idx)
        shares.append(dict(
            variable=var, basis=basis, unit=unit,
            window=f"{window[0]:.0f}–{window[1]:.0f}",
            observed_kt=float(np.median(agg["observed_change"])),
            driver_explained_kt=float(np.median(de)),
            residual_kt=float(np.median(res)),
            total_vs_counterfactual_kt=float(np.median(den)),
            driver_explained_share_median=st["median"],
            driver_explained_share_q1=st["q1"],
            driver_explained_share_q3=st["q3"],
            n_seeds_with_usable_denominator=int(np.isfinite(sh).sum()),
            attribution_index_median=si["median"],
            attribution_index_q1=si["q1"],
            attribution_index_q3=si["q3"],
            # Denominator is a DATA quantity, so it cannot go near zero the
            # way `total_vs_counterfactual` can: the fraction of the observed
            # move that removing the shock would have undone.
            driver_explained_over_observed=(
                float(np.median(de)) / float(np.median(agg["observed_change"]))
                if abs(float(np.median(agg["observed_change"]))) > 1e-9
                else np.nan)))
    return by_year, pd.DataFrame(shares)


def per_driver_attribution(dumps, arms, joint="shock2008_bridge",
                           variables=("secondary_supply", "In-Use", "Refined",
                                      "inuse_inflow")):
    """Effect of bridging one driver at a time, against the joint arm.

    `sum of singles` versus `joint` measures how much of the response is
    interaction rather than the sum of separable driver effects — a property
    a per-flow regression cannot report at all.
    """
    singles = [a for a in arms if a.startswith("shock2008_only::")]
    rows = []
    for var in variables:
        joint_eff, single_eff = [], {a: [] for a in singles}
        for _seed, d in dumps:
            t, y0 = series(d, arms[joint]["ref"], var)
            _t, yj = series(d, joint, var)
            joint_eff.append(yj - y0)
            for a in singles:
                _t, ya = series(d, a, var)
                single_eff[a].append(ya - y0)
        J = np.stack(joint_eff)
        for a in singles:
            A = np.stack(single_eff[a])
            for tag, yr in (("2010", 2010.0), ("2019", float(t[-1]))):
                j = int(np.argmin(np.abs(t - yr)))
                st = med_iqr(A[:, j])
                rows.append(dict(
                    variable=var, driver=a.split("::", 1)[1], year=float(t[j]),
                    effect_kt_median=st["median"], effect_kt_q1=st["q1"],
                    effect_kt_q3=st["q3"],
                    joint_effect_kt_median=float(np.median(J[:, j])),
                    share_of_joint=(float(np.median(A[:, j]))
                                    / float(np.median(J[:, j]))
                                    if abs(np.median(J[:, j])) > 1e-9 else np.nan)))
        # Σ singles and the interaction.  Both are formed PER SEED and then
        # medianed, so `<joint − Σ singles>` really is the difference of the
        # two rows above it — the column of per-driver medians is not additive
        # (each is a median of a different, partly cancelling distribution),
        # which is itself the readout that per-driver attribution here is not
        # seed-stable.
        S = np.sum(np.stack([np.stack(single_eff[a]) for a in singles]), axis=0)
        for label, M in (("<Σ singles>", S), ("<joint − Σ singles>", J - S)):
            for yr in (2010.0, float(t[-1])):
                j = int(np.argmin(np.abs(t - yr)))
                st = med_iqr(M[:, j])
                rows.append(dict(
                    variable=var, driver=label, year=float(t[j]),
                    effect_kt_median=st["median"], effect_kt_q1=st["q1"],
                    effect_kt_q3=st["q3"],
                    joint_effect_kt_median=float(np.median(J[:, j])),
                    share_of_joint=(st["median"] / float(np.median(J[:, j]))
                                    if abs(np.median(J[:, j])) > 1e-9 else np.nan)))
    return pd.DataFrame(rows)


def coefficient_response(dumps, arms, years_of_interest=(2010.0, 2019.0)):
    """Which learned coefficients actually move under each arm.

    This is the mechanism table: a counterfactual on the drivers can only
    reach the stocks through α(t) and the learned τ, so reading the
    coefficient response says *through which route* each result runs — and
    therefore which WP-1a / WP-8g caveat attaches to it.

    It doubles as a check.  `use_stock_input: false` makes the MLP a function
    of time and drivers alone, so an arm that does not touch the drivers must
    leave α exactly unchanged; the `tau_olds` arms must show 0.0 in every α
    column and a step in `tau_olds` alone.
    """
    import zinc_colloc_v5 as v5

    a_names = list(v5.ALPHA_NAMES)
    t_names = list(v5.TAU_BINARY_NAMES)
    rows = []
    for arm, spec in arms.items():
        ref = spec.get("ref")
        if ref is None:
            continue
        years = np.asarray(dumps[0][1]["years"], float).ravel()
        for block, names in (("alphas", a_names), ("taus", t_names)):
            A = np.stack([np.asarray(d[f"{block}__{_key(arm)}"], float)
                          for _s, d in dumps])
            R = np.stack([np.asarray(d[f"{block}__{_key(ref)}"], float)
                          for _s, d in dumps])
            for k, nm in enumerate(names):
                for yr in years_of_interest:
                    j = int(np.argmin(np.abs(years - yr)))
                    pct = 100.0 * (A[:, j, k] - R[:, j, k]) / np.where(
                        np.abs(R[:, j, k]) > 1e-12, R[:, j, k], np.nan)
                    st = med_iqr(pct)
                    sa = med_iqr(A[:, j, k] - R[:, j, k])
                    rows.append(dict(
                        arm=arm, coefficient=nm, year=float(years[j]),
                        ref_median=float(np.median(R[:, j, k])),
                        abs_change_median=sa["median"],
                        pct_change_median=st["median"],
                        pct_change_q1=st["q1"], pct_change_q3=st["q3"],
                        max_abs_change_any_seed=float(
                            np.max(np.abs(A[:, j, k] - R[:, j, k])))))
    return pd.DataFrame(rows)


def price_elasticity(dumps, arms, paths, window=(2008.0, 2019.0)):
    """Implied elasticity of each response to the counterfactual price move.

    Mean per-cent change in the response over the intervention window divided
    by the mean per-cent change in the price that produced it — a single,
    checkable number per arm, and the form in which an economist will want to
    read a price counterfactual.  It is an arc elasticity over a large,
    non-marginal move, not a local derivative, and the three price arms differ
    enough in size to show whether it is stable across them.
    """
    fac = paths[(paths.arm == "factual") & (paths.driver == "Zinc real price")]
    fac = fac.set_index("year")["value"]
    rows = []
    for arm in [a for a in arms if a.startswith("price_")]:
        cf = paths[(paths.arm == arm) & (paths.driver == "Zinc real price")]
        cf = cf.set_index("year")["value"]
        yrs = [y for y in fac.index if window[0] <= y <= window[1]]
        dp = float(np.mean(100.0 * (cf[yrs].values - fac[yrs].values)
                           / fac[yrs].values))
        for var in ("secondary_supply", "In-Use", "Scrap", "Refined"):
            vals = []
            for _seed, d in dumps:
                t, y = series(d, arm, var)
                _t, y0 = series(d, arms[arm]["ref"], var)
                m = (t >= window[0]) & (t <= window[1])
                vals.append(float(np.nanmean(
                    100.0 * (y[m] - y0[m]) / np.where(np.abs(y0[m]) > 1e-12,
                                                      y0[m], np.nan))))
            st = med_iqr(vals)
            rows.append(dict(
                arm=arm, variable=var,
                window=f"{window[0]:.0f}–{window[1]:.0f}",
                mean_price_change_pct=dp,
                mean_response_pct_median=st["median"],
                mean_response_pct_q1=st["q1"], mean_response_pct_q3=st["q3"],
                arc_elasticity_median=(st["median"] / dp if abs(dp) > 1e-9
                                       else np.nan),
                # dividing by a negative price change reverses the order of
                # the seed quantiles, so re-sort rather than mislabel them
                arc_elasticity_q1=(min(st["q1"] / dp, st["q3"] / dp)
                                   if abs(dp) > 1e-9 else np.nan),
                arc_elasticity_q3=(max(st["q1"] / dp, st["q3"] / dp)
                                   if abs(dp) > 1e-9 else np.nan)))
    return pd.DataFrame(rows)


def driver_paths(dumps, arms):
    """The counterfactual driver series themselves, for the figures."""
    _seed, d = dumps[0]
    t = np.asarray(d["driver_times_full"], float)
    names = [str(x) for x in d["driver_names"]]
    rows = []
    for arm in arms:
        X = np.asarray(d[f"drivers__{_key(arm)}"], float)
        for j, nm in enumerate(names):
            if not np.allclose(X[:, j],
                               np.asarray(d["drivers__factual"], float)[:, j]) \
                    or arm == "factual":
                for i, yr in enumerate(t):
                    rows.append(dict(arm=arm, driver=nm, year=float(yr),
                                     value=float(X[i, j])))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _band(ax, t, eff, colour, label, key="diff_pct"):
    ax.plot(t, eff[f"{key}_median"], color=colour, lw=1.9, label=label)
    ax.fill_between(t, eff[f"{key}_q1"], eff[f"{key}_q3"],
                    color=colour, alpha=0.20, lw=0)


def fig_price(eff, paths, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = ["price_trend90s", "price_trend90s_anchored", "price_flat90s"]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2))

    ax = axes[0]
    f = paths[(paths.arm == "factual") & (paths.driver == "Zinc real price")]
    ax.plot(f.year, f.value, color="k", lw=1.8, label="observed")
    for a in arms:
        g = paths[(paths.arm == a) & (paths.driver == "Zinc real price")]
        ax.plot(g.year, g.value, color=COL[a], lw=1.6, ls="--", label=a)
    ax.axvspan(2008, 2019, color=GREY, alpha=0.08, lw=0)
    ax.set_ylabel("real zinc price (constant USD/t)")
    ax.set_xlabel("year")
    ax.set_title("(a) the intervention", loc="left", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False)

    for ax, var, ttl in ((axes[1], "In-Use", "(b) In-Use stock"),
                         (axes[2], "secondary_supply", "(c) secondary supply")):
        for a in arms:
            e = eff[(eff.arm == a) & (eff.variable == var)].sort_values("year")
            _band(ax, e.year.values, e, COL[a], a)
        ax.axhline(0, color="k", lw=0.9)
        ax.axvspan(2008, 2019, color=GREY, alpha=0.08, lw=0)
        ax.set_ylabel("counterfactual − factual (%)")
        ax.set_xlabel("year")
        ax.set_title(ttl, loc="left", fontsize=10)
        ax.legend(fontsize=7.5, frameon=False)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(lw=0.4, alpha=0.3)
    fig.suptitle("WP-7.1  zinc price held on its 1990s path, 2008–2019 — "
                 "35-seed median with IQR band",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=300); fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


def fig_policy(dumps, eff, tte_sum, path, arm="tau_olds_step10_2005"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    years = np.asarray(dumps[0][1]["years"], float).ravel()
    T_ref = np.stack([np.asarray(d[f"taus__{_key('factual_tau_ref')}"],
                                 float)[:, 2] for _s, d in dumps])
    T_cf = np.stack([np.asarray(d[f"taus__{_key(arm)}"], float)[:, 2]
                     for _s, d in dumps])

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2))

    ax = axes[0]
    for M, c, lab in ((T_ref, "k", "fitted τ_olds"),
                      (T_cf, COL[arm], "+10 pp from 2005")):
        lo, md, hi = np.percentile(M, [25, 50, 75], axis=0)
        ax.plot(years, md, color=c, lw=1.8, label=lab)
        ax.fill_between(years, lo, hi, color=c, alpha=0.18, lw=0)
    ax.axvline(2005, color=GREY, ls=":", lw=1.1)
    ax.set_ylabel("τ_olds  (old-scrap collection rate)")
    ax.set_xlabel("year")
    ax.set_title("(a) the intervention", loc="left", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    for var, c in (("secondary_supply", COL[arm]),
                   ("old_scrap_recovery", COL["tau_olds_step05_2005"]),
                   ("In-Use", "#0072B2")):
        e = eff[(eff.arm == arm) & (eff.variable == var)].sort_values("year")
        _band(ax, e.year.values, e, c, var)
    ax.axhline(0, color="k", lw=0.9)
    ax.axvline(2005, color=GREY, ls=":", lw=1.1)
    ax.set_ylabel("counterfactual − factual (%)")
    ax.set_xlabel("year")
    ax.set_title("(b) response", loc="left", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[2]
    for var, c in (("secondary_supply", COL[arm]), ("In-Use", "#0072B2")):
        e = eff[(eff.arm == arm) & (eff.variable == var)].sort_values("year")
        m = e.year.values >= 2005
        t = e.year.values[m]
        r = e.diff_kt_median.values[m]
        if abs(r[-1]) < 1e-12:
            continue
        ax.plot(t - 2005, r / r[-1], color=c, lw=1.9, marker="o", ms=3,
                label=var)
        s = tte_sum[(tte_sum.arm == arm) & (tte_sum.variable == var)]
        if len(s):
            for frac, key in ((0.5, "years_to_50pct_median"),
                              (0.9, "years_to_90pct_median")):
                v = float(s.iloc[0][key])
                if np.isfinite(v):
                    ax.plot([v], [frac], marker="v", ms=7, color=c, zorder=4)
    for frac in (0.5, 0.9):
        ax.axhline(frac, color=GREY, lw=0.7, ls="--")
    ax.set_ylabel("response / response at 2019")
    ax.set_xlabel("years since the 2005 intervention")
    ax.set_title("(c) time-to-effect (▼ = seed-median crossing)",
                 loc="left", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(lw=0.4, alpha=0.3)
    fig.suptitle("WP-7.2  τ_olds stepped up 10 pp from 2005 — the lag is not "
                 "imposed; it comes from mass balance and the cohort structure",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=300); fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


def fig_shock(by_year, attrib, paths, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2))

    ax = axes[0]
    show = ["Zinc real price", "US Industrial Production Total Index",
            "World Stock Market Capitalisation (% of GDP)"]
    for i, dn in enumerate(show):
        f = paths[(paths.arm == "factual") & (paths.driver == dn)]
        g = paths[(paths.arm == "shock2008_bridge") & (paths.driver == dn)]
        if not len(f) or not len(g):
            continue
        base = float(f[f.year == 2007].value.iloc[0])
        c = ["#0072B2", "#D55E00", "#009E73"][i % 3]
        ax.plot(f.year, 100 * f.value / base, color=c, lw=1.7,
                label=dn[:28])
        ax.plot(g.year, 100 * g.value / base, color=c, lw=1.4, ls="--")
    ax.set_xlim(2003, 2014)
    ax.axvspan(2008, 2009, color=GREY, alpha=0.10, lw=0)
    ax.set_ylabel("index, 2007 = 100")
    ax.set_xlabel("year")
    ax.set_title("(a) observed (—) vs bridged (--)", loc="left", fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    g = by_year[by_year.variable == "secondary_supply"].sort_values("year")
    ax.plot(g.year, g.observed_change, color="k", lw=1.9, marker="o", ms=3,
            label="observed change since 2007")
    ax.plot(g.year, g.factual_change_median, color="#0072B2", lw=1.7,
            label="model, factual drivers")
    ax.fill_between(g.year, g.factual_change_q1, g.factual_change_q3,
                    color="#0072B2", alpha=0.18, lw=0)
    ax.plot(g.year, g.counterfactual_change_median, color=COL["shock2008_bridge"],
            lw=1.7, ls="--", label="model, 2008 shock removed")
    ax.fill_between(g.year, g.counterfactual_change_q1,
                    g.counterfactual_change_q3,
                    color=COL["shock2008_bridge"], alpha=0.18, lw=0)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.axvspan(2008, 2010, color=GREY, alpha=0.10, lw=0)
    ax.set_ylabel("secondary supply, change since 2007 (kt/yr)")
    ax.set_xlabel("year")
    ax.set_title("(b) driver-explained vs residual", loc="left", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False)

    ax = axes[2]
    a = attrib[(attrib.variable == "secondary_supply") & (attrib.year == 2010.0)]
    a = a.sort_values("effect_kt_median")
    y = np.arange(len(a))
    ax.barh(y, a.effect_kt_median.values,
            xerr=[a.effect_kt_median.values - a.effect_kt_q1.values,
                  a.effect_kt_q3.values - a.effect_kt_median.values],
            color=[GREY if "joint" in s else COL["shock2008_bridge"]
                   for s in a.driver],
            error_kw=dict(lw=0.8), height=0.7)
    ax.axvline(0, color="k", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([s[:32] for s in a.driver], fontsize=7)
    ax.set_xlabel("effect on secondary supply at 2010 (kt/yr)")
    ax.set_title("(c) one driver at a time", loc="left", fontsize=10)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(lw=0.4, alpha=0.3)
    fig.suptitle("WP-7.3  the 2008 driver shock removed by bridging 2008–2009 "
                 "— decomposition of the change since 2007",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=300); fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
def check():
    import zinc_cf_lab as lab
    return lab.check(verbose=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="WP-7 counterfactual aggregation")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dumps", default=DUMP_DIR)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args(argv)

    check()
    if args.check:
        return 0

    dumps = load_dumps(args.dumps)
    arms = arm_table(dumps[0][1])
    print(f"\nloaded {len(dumps)} seed dumps from {args.dumps}; "
          f"{len(arms)} arms")

    noise = noise_floor_table(dumps)
    bad = noise[noise.max_abs_dS_factual_vs_stored > 0]
    print(f"factual arm vs stored anchor_v4: max |ΔS| = "
          f"{noise.max_abs_dS_factual_vs_stored.max():.3e} over {len(noise)} "
          f"seeds ({len(bad)} non-identical); wrapper identity gap = "
          f"{max(noise.max_abs_wrapper_coef_identity.max(), noise.max_abs_wrapper_rhs_identity.max()):.1e}")
    nf = med_iqr(noise.solver_noise_floor_maxabs_kt.values)
    print(f"solver noise floor (|factual − factual_tau_ref|): median "
          f"{nf['median']:.2e} kt (IQR {nf['q1']:.2e}–{nf['q3']:.2e}), "
          f"max {noise.solver_noise_floor_maxabs_kt.max():.2e} kt")
    vnf = variable_noise_floor(dumps)
    print("per-variable noise floor (median | max across seeds, %): "
          + ",  ".join(f"{v} {vnf[v]['pct']:.3f}|{vnf[v]['pct_max']:.3f}"
                       for v in ("In-Use", "Scrap", "Refined",
                                 "secondary_supply")))

    eff = effects_by_year(dumps, arms)
    summ = summary_table(dumps, arms, noise)
    tte = time_to_effect(dumps, arms)
    tte_sum = summarise_tte(tte)
    lin = linearity_check(dumps, arms)
    dec = shock_decomposition(dumps, arms)
    by_year, shares = summarise_shock(dec)
    attrib = per_driver_attribution(dumps, arms)
    paths = driver_paths(dumps, arms)
    elas = price_elasticity(dumps, arms, paths)
    coef = coefficient_response(dumps, arms)

    os.makedirs(args.out, exist_ok=True)
    eff.to_csv(os.path.join(args.out, "wp7_effects_by_year.csv"), index=False)
    summ.to_csv(os.path.join(args.out, "wp7_summary.csv"), index=False)
    tte.to_csv(os.path.join(args.out, "wp7_time_to_effect_per_seed.csv"), index=False)
    tte_sum.to_csv(os.path.join(args.out, "wp7_time_to_effect.csv"), index=False)
    lin.to_csv(os.path.join(args.out, "wp7_linearity.csv"), index=False)
    by_year.to_csv(os.path.join(args.out, "wp7_shock_decomposition.csv"), index=False)
    shares.to_csv(os.path.join(args.out, "wp7_shock_shares.csv"), index=False)
    attrib.to_csv(os.path.join(args.out, "wp7_shock_per_driver.csv"), index=False)
    noise.to_csv(os.path.join(args.out, "wp7_noise_floor.csv"), index=False)
    paths.to_csv(os.path.join(args.out, "wp7_driver_paths.csv"), index=False)
    elas.to_csv(os.path.join(args.out, "wp7_price_elasticity.csv"), index=False)
    coef.to_csv(os.path.join(args.out, "wp7_coefficient_response.csv"), index=False)

    fig_price(eff, paths, os.path.join(args.out, "wp7_price.png"))
    fig_policy(dumps, eff, tte_sum, os.path.join(args.out, "wp7_policy.png"))
    fig_shock(by_year, attrib, paths, os.path.join(args.out, "wp7_shock.png"))

    pd.set_option("display.width", 210, "display.max_columns", 40)
    key = ["In-Use", "Scrap", "secondary_supply"]
    print("\n--- 1. price path: effect at 2019 (median [IQR], HL 95% CI on %) ---")
    s = summ[(summ.arm.str.startswith("price_")) & (summ.variable.isin(key))]
    print(s[["arm", "variable", "terminal_kt_median", "terminal_pct_median",
             "terminal_pct_HL_lo", "terminal_pct_HL_hi",
             "cumulative_kt_median", "noise_floor_pct",
             "abs_terminal_over_noise_floor", "clears_noise_floor"]]
          .round(3).to_string(index=False))

    print("\n    implied arc elasticity over 2008–2019 "
          "(mean %% response / mean %% price change):")
    print(elas[elas.variable.isin(["secondary_supply", "In-Use"])]
          [["arm", "variable", "mean_price_change_pct",
            "mean_response_pct_median", "arc_elasticity_median",
            "arc_elasticity_q1", "arc_elasticity_q3"]]
          .round(3).to_string(index=False))

    print("\n    which coefficients move (2019; the route each result runs "
          "through):")
    cr = coef[(coef.year == 2019.0)
              & coef.arm.isin(["price_trend90s", "tau_olds_step10_2005",
                               "shock2008_bridge"])]
    print(cr.pivot_table(index="coefficient", columns="arm",
                         values="pct_change_median")
          .round(3).to_string())
    zero = coef[coef.arm.str.startswith("tau_olds")
                & coef.coefficient.str.startswith("alpha")]
    print(f"    check — α response to the τ_olds arms (must be exactly 0 with "
          f"use_stock_input=false): max |Δ| = "
          f"{zero.max_abs_change_any_seed.max():.3e}")

    print("\n--- 2. policy: time-to-effect of τ_olds +10 pp from 2005 ---")
    print(tte_sum[["variable", "terminal_response_kt_median",
                   "first_year_response_kt_median",
                   "first_year_frac_median", "years_to_25pct_median",
                   "years_to_50pct_median", "years_to_90pct_median"]]
          .round(3).to_string(index=False))
    print("\n    linearity (10 pp response / 5 pp response; 2.0 = linear):")
    print(lin.round(3).to_string(index=False))

    print("\n--- 3. shock removal: effect at 2019 (does anything persist?) ---")
    sh = summ[(summ.arm.str.startswith("shock2008_bridge"))
              & (summ.variable.isin(key + ["Refined"]))]
    print(sh[["arm", "variable", "terminal_kt_median", "terminal_pct_median",
              "noise_floor_pct", "clears_noise_floor"]]
          .round(4).to_string(index=False))
    print("\n--- 3. shock removal: cumulative over 2008–2010 ---")
    print(shares[shares.variable.isin(key + ["inuse_inflow", "Refined"])]
          [["variable", "basis", "observed_kt", "driver_explained_kt",
            "residual_kt", "total_vs_counterfactual_kt",
            "driver_explained_over_observed", "attribution_index_median",
            "driver_explained_share_median",
            "n_seeds_with_usable_denominator"]].round(3).to_string(index=False))
    print("\n    per-driver effect on secondary supply at 2010 (kt/yr):")
    a = attrib[(attrib.variable == "secondary_supply") & (attrib.year == 2010.0)]
    print(a.reindex(a.effect_kt_median.abs().sort_values(ascending=False).index)
          [["driver", "effect_kt_median", "effect_kt_q1", "effect_kt_q3",
            "joint_effect_kt_median", "share_of_joint"]]
          .round(3).to_string(index=False))

    print(f"\nwrote {args.out}/wp7_{{summary,effects_by_year,time_to_effect,"
          f"linearity,shock_decomposition,shock_shares,shock_per_driver,"
          f"noise_floor,driver_paths,price_elasticity,"
          f"coefficient_response}}.csv and "
          f"wp7_{{price,policy,shock}}.{{png,pdf}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
