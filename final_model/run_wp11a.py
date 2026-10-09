#!/usr/bin/env python3
"""
run_wp11a.py — WP-11a: frequency–value curves
=============================================

The spec's question, in its own words: *"On the synthetic twin, generate a
truth containing realistic sub-annual structure and sample at 1, 2, 4 and 12
observations per year. Fit each."*  And then, separately, *"Measure both halves
of the question"* — forecast value and understanding value — under a design
that keeps frequency and sample size from being confounded.

`zinc_freq_lab.py` builds the arms and states what is and is not run (the
spec's 80-year span cannot be: the driver record ends in 2019).  This driver
fits them, scores them, and emits the curve WP-11b's pre-registered gate
consumes.

The scoring grid, and why there is one
--------------------------------------
An arm observed monthly predicts alpha at 469 nodes and an arm observed every
tenth year at 40 (its record is masked, not shortened — WP-6b's operator).  A
relRMSE taken on each arm's own nodes is therefore not the same statistic
twice, and a curve built from such numbers would move with the node count.
Every arm is scored on **the annual grid**, 1980-2019: point quantities
(alpha, tau, stocks) are interpolated onto it from the arm's own nodes, and
flow window-integrals are *summed* onto it, which is exact for every delta that
divides a year.  That makes the curve comparable across arms and denominated
in the same currency as WP-6b's real-data curve.

Three references, three questions
---------------------------------
Each fit is scored against three things, and conflating them is the mistake
this package exists to avoid:

  vs **truth**      understanding value.  Only a twin has this.
  vs **annual obs** the currency WP-6b's real curve is in, and therefore the
                    one the WP-11b gate needs — a like-for-like comparison
                    needs the same denominator on both sides.
  vs **arm obs**    what the estimator was actually shown; the gap between
                    this and the truth column is target degradation, which
                    WP-6b and WP-11b both had to separate out and so does this.

Parts
-----
  1  fit      every arm x seed                          (the expensive part)
  2  score    the two halves, on the annual grid
  3  spectrum the WP-4c identifiability spectrum recomputed at each frequency
  4  curves   `wp11a_frequency_value.csv` (the gate handoff) + the span curve
  5  figures

    python run_wp11a.py --check
    python run_wp11a.py --fit --seeds 0,1,2,3,4,5,6,7
    python run_wp11a.py --score --curves --figures
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import zinc_alpha_lab as lab
import zinc_freq_lab as F
import zinc_synth_lab as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "analysis")
SYNTH_DIR = os.path.join(OUT_DIR, "synth")
FIT_DIR = os.path.join(SYNTH_DIR, "fits")

COL = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00", "#56B4E9"]
GREY = "#555555"

ALPHA_NAMES = F.ALPHA_NAMES
TAU_SUP_NAMES = F.TAU_SUP_NAMES
STOCK_NAMES = F.STOCK_NAMES


# ===========================================================================
# metrics
# ===========================================================================
def rel_rmse_pct(pred, obs, mask=None):
    """relRMSE in percent; denominator `mean(|obs|)` (CLAUDE.md conventions)."""
    p, o = np.asarray(pred, float), np.asarray(obs, float)
    m = np.isfinite(p) & np.isfinite(o)
    if mask is not None:
        m &= np.asarray(mask, bool)
    if not m.any():
        return np.nan
    den = float(np.mean(np.abs(o[m])))
    if den < 1e-12:
        return np.nan
    return 100.0 * float(np.sqrt(np.mean((p[m] - o[m]) ** 2))) / den


def hl(v):
    """Hodges–Lehmann location and its distribution-free CI (project rule)."""
    v = np.asarray([x for x in np.ravel(v) if np.isfinite(x)], float)
    n = v.size
    if n == 0:
        return np.nan, np.nan, np.nan
    w = np.array([(v[i] + v[j]) / 2.0 for i in range(n) for j in range(i, n)])
    w.sort()
    est = float(np.median(w))
    if n < 4:
        return est, np.nan, np.nan
    k = max(int(np.floor(0.025 * w.size)), 0)
    return est, float(w[k]), float(w[w.size - 1 - k])


# ===========================================================================
# the annual scoring grid
# ===========================================================================
ANNUAL = np.arange(1980.0, 2020.0)

# The shortest arm in the design (`span` = 10 yr) is fitted to 1980-1987, so
# 1987 is the latest year every arm in the package has both been fitted on and
# can be scored over.  Scoring each arm on *its own* fitted span answers "how
# well did this record let you recover the coefficients", which is the natural
# question, but it makes the span and matched-N arms incomparable: a 7-year
# window sits entirely before every turning point in `zinc_synth_lab.TRUTH`
# (1997-2008) and is a much easier target than a 28-year one.  The `common`
# region removes that confound at the cost of judging the long arms on a
# fraction of what they saw.  Both are reported; neither alone is the answer.
COMMON_SCORE_END = 1987.0


def to_annual_point(arm_years, values, target):
    """Point-in-time quantity from the arm's nodes onto `target` years.

    Linear interpolation, and `np.nan` outside the arm's span so a short-span
    arm is never credited with a prediction it did not make.
    """
    x = np.asarray(arm_years, float).ravel()
    V = np.atleast_2d(np.asarray(values, float).T).T
    out = np.full((target.size, V.shape[1]), np.nan)
    inside = (target >= x[0] - 1e-9) & (target <= x[-1] + 1e-9)
    for j in range(V.shape[1]):
        col = V[:, j]
        ok = np.isfinite(col)
        if ok.sum() < 2:
            continue
        out[inside, j] = np.interp(target[inside], x[ok], col[ok])
    return out


def to_annual_integral(arm_years, F_int, target):
    """Window integrals summed onto annual intervals.

    `F_int` row i is the integral over `(arm_years[i], arm_years[i+1]]`.  The
    annual value for the interval ending at `target[k]` is the sum of every
    arm window inside it, which is exact whenever the arm's width divides a
    year and whenever it is a multiple of one — the only two cases the design
    contains.  Intervals not exactly covered are left `nan` rather than
    apportioned, so nothing is invented at an arm's ragged tail.
    """
    x = np.asarray(arm_years, float).ravel()
    A = np.asarray(F_int, float)
    c = np.vstack([np.zeros((1, A.shape[1])), np.cumsum(A, axis=0)])   # (T, NF)
    out = np.full((target.size - 1, A.shape[1]), np.nan)
    for k in range(1, target.size):
        lo, hi = target[k - 1], target[k]
        if lo < x[0] - 1e-9 or hi > x[-1] + 1e-9:
            continue
        i0 = int(np.argmin(np.abs(x - lo)))
        i1 = int(np.argmin(np.abs(x - hi)))
        if abs(x[i0] - lo) > 1e-6 or abs(x[i1] - hi) > 1e-6:
            continue
        out[k - 1] = c[i1] - c[i0]
    return out


def operator_selftest(fit_dir=FIT_DIR, out_dir=OUT_DIR):
    """The scoring grid must be exact on the truth before it is used on a fit.

    Both reductions have an identity available: aggregating the twin's own
    sub-annual flow integrals onto annual intervals must reproduce the annual
    twin's flow integrals exactly (they are differences of the same
    accumulator), and interpolating the truth's point coefficients from a fine
    grid onto the annual nodes must reproduce the annual truth exactly (the
    annual nodes are a subset of the fine ones).  Anything short of 0 here
    would mean the curve is measuring the scorer.
    """
    rows = []
    for tag in [r["tag"] for r in F.design()]:
        ps = _fit_paths(tag, fit_dir)
        if not ps:
            continue
        d = np.load(ps[0], allow_pickle=True)
        yrs = np.asarray(d["years"], float)
        agg = to_annual_integral(yrs, d["F_true"], ANNUAL)
        ref = np.asarray(d["F_true_annual"], float)
        m = np.isfinite(agg) & np.isfinite(ref) & (np.abs(ref) > 1e-9)
        e_f = float(np.max(np.abs(agg[m] - ref[m]) / np.abs(ref[m]))) if m.any() else np.nan
        pt = to_annual_point(yrs, d["alpha_true"], ANNUAL)
        rt = np.asarray(d["alpha_true_annual"], float)
        m2 = np.isfinite(pt) & np.isfinite(rt) & (np.abs(rt) > 1e-12)
        e_p = float(np.max(np.abs(pt[m2] - rt[m2]) / np.abs(rt[m2]))) if m2.any() else np.nan
        rows.append(dict(tag=tag, n_obs=int(yrs.size),
                         flow_integral_max_rel_err=e_f,
                         point_interp_max_rel_err=e_p,
                         ok=bool((not np.isfinite(e_f) or e_f < 1e-9)
                                 and (not np.isfinite(e_p) or e_p < 1e-9))))
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(os.path.join(out_dir, "wp11a_operator_check.csv"), index=False)
    return df


# ===========================================================================
# part 1 — fitting
# ===========================================================================
def part1_fit(tags, seeds, out_dir=FIT_DIR, cfg=None, skip_existing=True,
              spectrum_seeds=(0,), verbose=False):
    cfg = cfg or lab.load_anchor_config()
    rows = []
    for tag in tags:
        for sd in seeds:
            t0 = time.time()
            try:
                p = F.fit_arm(sd, tag, out_dir=out_dir, cfg=cfg,
                              verbose=verbose, skip_existing=skip_existing,
                              spectrum=(sd in spectrum_seeds))
                rows.append(dict(tag=tag, seed=sd, ok=True,
                                 wall_s=time.time() - t0, path=p))
            except Exception as e:            # a dead arm must not kill a sweep
                import traceback
                traceback.print_exc()
                rows.append(dict(tag=tag, seed=sd, ok=False,
                                 wall_s=time.time() - t0,
                                 path=f"{type(e).__name__}: {e}"))
    return pd.DataFrame(rows)


# ===========================================================================
# part 2 — scoring
# ===========================================================================
def _fit_paths(tag, fit_dir=FIT_DIR):
    return sorted(glob.glob(os.path.join(fit_dir, f"{tag}_seed*.npz")))


def score_one(path):
    """Every score for one fit, on the annual grid.  Returns a list of rows."""
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta_json"]))
    yrs = np.asarray(d["years"], float)
    ya = np.asarray(d["years_annual"], float)

    # regions, resolved on the *year* axis so every arm means the same thing
    val_end = float(meta["val_end_year"])
    reg = {"fitted": ya <= val_end + 1e-9,
           "test": ya > val_end + 1e-9,
           "common": ya <= COMMON_SCORE_END + 1e-9}
    # an arm that stops early has no prediction past its own span
    span_hi = float(yrs[-1])
    for k in reg:
        reg[k] = reg[k] & (ya <= span_hi + 1e-9)
    regF = {k: v[1:] for k, v in reg.items()}          # flow intervals

    a_pred = to_annual_point(yrs, d["alpha_pred_B_at_obs"], ya)
    t_pred = to_annual_point(yrs, d["tau_pred_B_at_obs"], ya)
    s_pred = to_annual_point(yrs, d["S_pred_B"], ya)
    f_pred = to_annual_integral(yrs, d["F_pred_B"], ya)

    a_true = np.asarray(d["alpha_true_annual"], float)
    t_true = np.asarray(d["tau_true_annual"], float)
    s_true = np.asarray(d["S_true_annual"], float)
    f_true = np.asarray(d["F_true_annual"], float)
    a_ann = np.asarray(d["alpha_obs_annual"], float)
    # the arm's own alpha target, lifted to the annual grid for the
    # target-degradation column (nan where the arm has no reading)
    a_arm = to_annual_point(yrs, d["alpha_obs_arm"], ya)

    rows = []
    common = dict(tag=meta["tag"], kind=meta["kind"],
                  twin=str(meta.get("twin", "base")),
                  delta=float(meta["delta"]), span=float(meta["span"]),
                  n_obs=int(meta.get("n_obs", yrs.size)),
                  seed=int(os.path.basename(path).split("seed")[-1].split(".")[0]))
    for split, m in reg.items():
        for k, name in enumerate(ALPHA_NAMES):
            rows.append(dict(common, family="alpha", channel=name, split=split,
                             pred_vs_true=rel_rmse_pct(a_pred[:, k], a_true[:, k], m),
                             pred_vs_annual=rel_rmse_pct(a_pred[:, k], a_ann[:, k], m),
                             pred_vs_arm=rel_rmse_pct(a_pred[:, k], a_arm[:, k], m),
                             target_vs_true=rel_rmse_pct(a_arm[:, k], a_true[:, k], m),
                             annual_target_vs_true=rel_rmse_pct(a_ann[:, k], a_true[:, k], m)))
        for k, name in enumerate(TAU_SUP_NAMES):
            rows.append(dict(common, family="tau", channel=name, split=split,
                             pred_vs_true=rel_rmse_pct(t_pred[:, k], t_true[:, k], m),
                             pred_vs_annual=np.nan, pred_vs_arm=np.nan,
                             target_vs_true=np.nan, annual_target_vs_true=np.nan))
        for k, name in enumerate(STOCK_NAMES):
            rows.append(dict(common, family="stock", channel=name, split=split,
                             pred_vs_true=rel_rmse_pct(s_pred[:, k], s_true[:, k], m),
                             pred_vs_annual=np.nan, pred_vs_arm=np.nan,
                             target_vs_true=np.nan, annual_target_vs_true=np.nan))
        mf = regF[split]
        fnames = [str(x) for x in d["flow_names_pred"]] if "flow_names_pred" in d.files \
            else [f"flow{i}" for i in range(f_true.shape[1])]
        for k, name in enumerate(fnames):
            if not np.isfinite(f_true[mf, k]).any() or np.mean(np.abs(f_true[mf, k])) < 1e-9:
                continue
            rows.append(dict(common, family="flow", channel=name, split=split,
                             pred_vs_true=rel_rmse_pct(f_pred[:, k], f_true[:, k], mf),
                             pred_vs_annual=np.nan, pred_vs_arm=np.nan,
                             target_vs_true=np.nan, annual_target_vs_true=np.nan))
    return rows


def part2_score(tags, fit_dir=FIT_DIR, out_dir=OUT_DIR):
    rows = []
    for tag in tags:
        for p in _fit_paths(tag, fit_dir):
            rows.extend(score_one(p))
    per_seed = pd.DataFrame(rows)
    if per_seed.empty:
        print("[score] no fits found", flush=True)
        return per_seed, per_seed
    keys = ["tag", "kind", "twin", "delta", "span", "n_obs", "family",
            "channel", "split"]
    g = per_seed.groupby(keys, as_index=False)
    summ = g.agg(n_seeds=("seed", "nunique"),
                 pred_vs_true=("pred_vs_true", "median"),
                 pred_vs_true_q1=("pred_vs_true", lambda v: v.quantile(0.25)),
                 pred_vs_true_q3=("pred_vs_true", lambda v: v.quantile(0.75)),
                 pred_vs_annual=("pred_vs_annual", "median"),
                 pred_vs_arm=("pred_vs_arm", "median"),
                 target_vs_true=("target_vs_true", "median"),
                 annual_target_vs_true=("annual_target_vs_true", "median"))
    per_seed.to_csv(os.path.join(out_dir, "wp11a_per_seed.csv"), index=False)
    summ.to_csv(os.path.join(out_dir, "wp11a_summary.csv"), index=False)
    return per_seed, summ


# ===========================================================================
# part 2b — how much of the curve is the noise draw?
# ===========================================================================
def part2b_noise_replicates(n_rep=32, out_dir=OUT_DIR, cfg=None):
    """Bound the one confound the seed ensemble cannot average over.

    Each arm is a *different random sample*: a record observed monthly and one
    observed annually draw independent lognormal errors, and there is no
    construction that makes them the same draw, because iid observation noise
    has no continuous-time limit that survives re-aggregation.  The eight
    seeds vary the network initialisation, not the record, so nothing in the
    fitted curve averages this out.

    What *can* be measured without a single fit is how much the noise draw
    moves the **target** — `alpha_obs` against the analytic truth — at each
    width.  That is the input the estimator sees, so its spread across draws
    is a lower bound on the arm-to-arm noise contribution to the curve, and it
    is reported next to the between-seed spread of the fitted curve rather
    than left as a caveat with no number attached.
    """
    ctx = S.driver_context(cfg)
    sigma = S.calibrate_noise()
    dense = S.dense_solve(ctx, F.TWIN_ARM)
    rows = []
    for dl in F.DELTAS_FREQ:
        clean = S.sample(dense, dl, ctx)
        tw = S.truth_coefficients(ctx, clean["years"], F.TWIN_ARM,
                                  S_path=clean["stocks_clean"])
        a_true = np.asarray(tw["alphas"], float)
        for rep in range(int(n_rep)):
            sm = S.sample(dense, dl, ctx, noise_scale=F.NOISE_SCALE,
                          rng_seed=1000 + rep, sigma=sigma)
            for k, ch in enumerate(ALPHA_NAMES):
                rows.append(dict(delta=float(dl), rep=rep, channel=ch,
                                 target_vs_true=rel_rmse_pct(
                                     sm["alpha_obs"][:, k], a_true[:, k])))
    per_rep = pd.DataFrame(rows)
    summ = per_rep.groupby(["delta", "channel"], as_index=False).agg(
        n_rep=("target_vs_true", "size"),
        median=("target_vs_true", "median"),
        q1=("target_vs_true", lambda v: v.quantile(0.25)),
        q3=("target_vs_true", lambda v: v.quantile(0.75)))
    summ["iqr"] = summ.q3 - summ.q1
    per_rep.to_csv(os.path.join(out_dir, "wp11a_noise_replicates.csv"),
                   index=False)
    summ.to_csv(os.path.join(out_dir, "wp11a_noise_spread.csv"), index=False)
    return per_rep, summ


# ===========================================================================
# part 3 — the identifiability spectrum at each frequency
# ===========================================================================
def part3_spectrum(tags, fit_dir=FIT_DIR, out_dir=OUT_DIR):
    """Collect whatever `zinc_freq_lab` dumped alongside the fits."""
    rows = []
    for tag in tags:
        arm = F.arm_of(tag)
        for p in sorted(glob.glob(os.path.join(fit_dir, f"{tag}_seed*_spec.npz"))):
            d = np.load(p, allow_pickle=True)
            sd = int(os.path.basename(p).split("seed")[-1].split("_")[0])
            eig = np.asarray(d["eig"], float)
            eig = eig[eig > 0]
            row = dict(tag=tag, kind=arm["kind"], delta=arm["delta"],
                       span=arm["span"], n_obs=arm["n_obs"], seed=sd,
                       n_res=int(d["n_res"]), n_par=int(d["n_par"]),
                       rank=int(d["rank"]), p_eff=float(d["p_eff"]),
                       kappa2=float(d["kappa2"]),
                       eig_max=float(eig[0]) if eig.size else np.nan,
                       eig_min_pos=float(eig[-1]) if eig.size else np.nan,
                       log10_condition=float(np.log10(eig[0] / eig[-1]))
                       if eig.size > 1 else np.nan)
            for k, name in enumerate(ALPHA_NAMES):
                sd = float(np.asarray(d["sd_alpha"], float)[k])
                row[f"sd_{name}"] = sd
                # In percent of the channel's own geometric-mean level, so the
                # curvature is on the same scale as the realised relRMSE and
                # the two can be read against each other.  `A` is the truth's
                # own constant (zinc_synth_lab.TRUTH), not a fitted quantity.
                row[f"sd_pct_{name}"] = 100.0 * sd / float(
                    S.TRUTH["alpha"][name]["A"])
            rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(os.path.join(out_dir, "wp11a_spectrum.csv"), index=False)
    return df


# ===========================================================================
# part 4 — the curves
# ===========================================================================
GATE_KINDS = ("freq", "coarse")


def part4_curves(per_seed, out_dir=OUT_DIR):
    """The gate handoff and the span curve, kept in separate files.

    `wp11b.load_refit_curve` groups by (channel, delta) and takes the median,
    so the frequency file must contain **only full-span arms** — a span arm at
    delta = 1 would otherwise be pooled into the delta = 1 point and quietly
    move it.  The span curve goes to its own file, keyed by span.
    """
    if per_seed.empty:
        return None, None
    a = per_seed[(per_seed.family == "alpha") & (per_seed.split == "fitted")]

    freq = a[a.kind.isin(GATE_KINDS)].copy()
    freq = freq.rename(columns={"pred_vs_annual": "alpha_relRMSE_pct",
                                "pred_vs_true": "alpha_relRMSE_true_pct",
                                "pred_vs_arm": "alpha_relRMSE_arm_pct",
                                "target_vs_true": "target_relRMSE_true_pct"})
    keep = ["tag", "kind", "delta", "n_obs", "seed", "channel",
            "alpha_relRMSE_pct", "alpha_relRMSE_true_pct",
            "alpha_relRMSE_arm_pct", "target_relRMSE_true_pct"]
    freq = freq[keep].sort_values(["delta", "channel", "seed"])
    freq.to_csv(os.path.join(out_dir, "wp11a_frequency_value.csv"), index=False)

    span = per_seed[(per_seed.family == "alpha")
                    & per_seed.split.isin(("fitted", "common"))]
    span = span[span.kind.isin(("span", "matchedN"))
                | (span.kind == "freq")].copy()
    span = span.rename(columns={"pred_vs_annual": "alpha_relRMSE_pct",
                                "pred_vs_true": "alpha_relRMSE_true_pct"})
    span = span[["tag", "kind", "delta", "span", "n_obs", "seed", "split",
                 "channel", "alpha_relRMSE_pct", "alpha_relRMSE_true_pct"]]
    span = span.sort_values(["split", "span", "delta", "channel", "seed"])
    span.to_csv(os.path.join(out_dir, "wp11a_span_value.csv"), index=False)
    return freq, span


def part4b_paired(per_seed, out_dir=OUT_DIR):
    """Paired-by-seed shifts against the annual full-span reference arm.

    Everything in this package is a comparison *against annual*, and a median
    of medians hides that the arms share seeds.  Each arm's alpha recovery is
    differenced seed by seed against `wp11a_freq_1y` and summarised with a
    Hodges-Lehmann shift and its CI, which is the project's convention.
    """
    if per_seed.empty:
        return None
    ref_tag = F.arm_tag("freq", 1.0, F.FULL_SPAN)
    a = per_seed[(per_seed.family == "alpha") & (per_seed.split == "fitted")]
    ref = a[a.tag == ref_tag].set_index(["seed", "channel"])["pred_vs_true"]
    rows = []
    for (tag, ch), sub in a.groupby(["tag", "channel"]):
        if tag == ref_tag:
            continue
        arm = F.arm_of(tag)
        dv, n_better = [], 0
        for _i, r in sub.iterrows():
            key = (int(r.seed), ch)
            if key not in ref.index:
                continue
            delta_v = float(r.pred_vs_true) - float(ref.loc[key])
            if np.isfinite(delta_v):
                dv.append(delta_v)
                n_better += int(delta_v < 0)
        if not dv:
            continue
        est, lo, hi = hl(dv)
        rows.append(dict(tag=tag, kind=arm["kind"], delta=arm["delta"],
                         span=arm["span"], n_obs=arm["n_obs"], channel=ch,
                         n_pairs=len(dv), hl_shift_pp=est, hl_lo=lo, hl_hi=hi,
                         n_better=n_better,
                         frac_better=n_better / float(len(dv))))
    df = pd.DataFrame(rows).sort_values(["kind", "delta", "span", "channel"])
    df.to_csv(os.path.join(out_dir, "wp11a_paired_vs_annual.csv"), index=False)
    return df


# ===========================================================================
# part 4c — the falsifiable prediction WP-11b left for this package
# ===========================================================================
def part4c_prediction_check(per_seed, out_dir=OUT_DIR):
    """Test WP-11b's Fisher-arm prediction against the realised improvement.

    `COMPUTE_STATUS.md` records it as "one prediction for WP-11a, from the
    Fisher arm and falsifiable by it": monthly observation should take the
    marginal SD of the alpha channels to **0.27-0.38** of its annual value,
    close to the observation-counting law `sqrt(Delta) = 0.289`.  A marginal
    SD is not a relRMSE, and the two coincide only if the fitted error is
    dominated by the noise-driven variance the curvature describes rather than
    by bias, optimiser stopping point or target construction -- so the test is
    one-sided in interpretation: a realised ratio near `sqrt(Delta)`
    corroborates the local Gaussian picture, a ratio far above it says
    something other than observation count is binding.  Both readings are
    written out; nothing is reselected on the outcome (CLAUDE.md rule 3).
    """
    if per_seed.empty:
        return None
    a = per_seed[(per_seed.family == "alpha") & (per_seed.split == "fitted")
                 & (per_seed.kind.isin(("freq", "season")))]
    rows = []
    for (twin, ch), sub in a.groupby(["twin", "channel"]):
        ref = sub[np.isclose(sub.delta, 1.0)].pred_vs_true.median()
        for dl, s2 in sub.groupby("delta"):
            if not np.isfinite(ref) or ref <= 0:
                continue
            got = float(s2.pred_vs_true.median())
            counting = float(np.sqrt(dl))
            band = (0.27, 0.38) if np.isclose(dl, 1.0 / 12.0) else \
                   (0.45, 0.57) if np.isclose(dl, 0.25) else (np.nan, np.nan)
            rows.append(dict(twin=twin, channel=ch, delta=float(dl),
                             alpha_relRMSE_pct=got,
                             ratio_to_annual=got / float(ref),
                             counting_law_sqrt_delta=counting,
                             wp11b_predicted_lo=band[0],
                             wp11b_predicted_hi=band[1],
                             inside_wp11b_band=bool(
                                 np.isfinite(band[0])
                                 and band[0] <= got / float(ref) <= band[1])))
    df = pd.DataFrame(rows).sort_values(["twin", "delta", "channel"])
    df.to_csv(os.path.join(out_dir, "wp11a_prediction_check.csv"), index=False)
    return df


# ===========================================================================
# part 5 — figures
# ===========================================================================
def figures(per_seed, spec, out_dir=OUT_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(13.4, 4.3))

    a = per_seed[(per_seed.family == "alpha") & (per_seed.split == "fitted")]

    # (a) understanding value vs observation width
    a0 = ax[0]
    sub = a[a.kind.isin(GATE_KINDS)]
    for i, ch in enumerate(ALPHA_NAMES):
        s = sub[sub.channel == ch].groupby("delta").pred_vs_true
        med, q1, q3 = s.median(), s.quantile(.25), s.quantile(.75)
        a0.fill_between(med.index, q1, q3, color=COL[i], alpha=0.15, lw=0)
        a0.plot(med.index, med.values, "o-", color=COL[i], label=ch, ms=4)
    a0.axvline(1.0, color=GREY, ls=":", lw=1)
    a0.text(1.02, a0.get_ylim()[1], " annual", color=GREY, fontsize=7,
            va="top", ha="left")
    a0.set_xscale("log"); a0.set_yscale("log")
    a0.set_xlabel("observation width $\\Delta$ (yr)")
    a0.set_ylabel("$\\alpha$ recovery vs truth, relRMSE %")
    a0.set_title("(a) understanding value", fontsize=10, loc="left")
    a0.legend(fontsize=7, frameon=False)

    # (b) forecast value on the held-out window
    a1 = ax[1]
    for fam, mk in (("stock", "o-"), ("flow", "s--")):
        f = per_seed[(per_seed.family == fam) & (per_seed.split == "test")
                     & per_seed.kind.isin(GATE_KINDS)]
        if f.empty:
            continue
        s = f.groupby("delta").pred_vs_true
        med, q1, q3 = s.median(), s.quantile(.25), s.quantile(.75)
        c = COL[0] if fam == "stock" else COL[2]
        a1.fill_between(med.index, q1, q3, color=c, alpha=0.15, lw=0)
        a1.plot(med.index, med.values, mk, color=c, label=fam, ms=4)
    a1.axvline(1.0, color=GREY, ls=":", lw=1)
    a1.set_xscale("log")
    a1.set_xlabel("observation width $\\Delta$ (yr)")
    a1.set_ylabel("test-window relRMSE vs truth, %")
    a1.set_title("(b) forecast value", fontsize=10, loc="left")
    a1.legend(fontsize=7, frameon=False)

    # (c) resolution against sample size, on one N axis
    a2 = ax[2]
    for kind, mk, lab_ in (("freq", "o-", "$\\Delta$ varies, span 39 yr"),
                           ("span", "s--", "span varies, $\\Delta$ = 1 yr"),
                           ("matchedN", "D:", "matched $N$, $\\Delta$ < 1")):
        f = a[a.kind == kind]
        if kind == "span":
            f = pd.concat([f, a[(a.kind == "freq") & (a.delta == 1.0)]])
        if f.empty:
            continue
        s = f.groupby("n_obs").pred_vs_true.median()
        c = {"freq": COL[0], "span": COL[2], "matchedN": COL[3]}[kind]
        a2.plot(s.index, s.values, mk, color=c, label=lab_, ms=4)
    a2.set_xscale("log"); a2.set_yscale("log")
    a2.set_xlabel("number of observations $N$")
    a2.set_ylabel("$\\alpha$ recovery vs truth, relRMSE %")
    a2.set_title("(c) resolution vs record length", fontsize=10, loc="left")
    a2.legend(fontsize=7, frameon=False)

    for x in ax:
        x.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"wp11a_frequency_value.{ext}"),
                    dpi=180, bbox_inches="tight")
    plt.close(fig)
    return os.path.join(out_dir, "wp11a_frequency_value.png")


# ===========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--spectrum", action="store_true")
    ap.add_argument("--noise-replicates", type=int, default=0,
                    help="draws for the target-side noise-spread bound")
    ap.add_argument("--curves", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--arms", default="all")
    ap.add_argument("--spectrum-seeds", default="0")
    ap.add_argument("--fit-dir", default=FIT_DIR)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--refit", action="store_true",
                    help="do not skip arms already on disk")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    all_tags = [r["tag"] for r in F.design()]
    tags = all_tags if args.arms == "all" else [
        t for t in args.arms.split(",") if t.strip()]
    unknown = [t for t in tags if t not in all_tags]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; known: {all_tags}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if args.check:
        F.check(verbose=True)
        F.noop_check()
        ck = operator_selftest(fit_dir=args.fit_dir, out_dir=args.out)
        if ck is not None and not ck.empty:
            print("\n  scoring-grid self-test (exact on the truth):")
            print(ck.to_string(index=False))
        return 0

    os.makedirs(args.out, exist_ok=True)
    if args.fit:
        spec_seeds = {int(s) for s in args.spectrum_seeds.split(",") if s.strip()}
        st = part1_fit(tags, seeds, out_dir=args.fit_dir,
                       skip_existing=not args.refit,
                       spectrum_seeds=spec_seeds, verbose=args.verbose)
        st.to_csv(os.path.join(args.out, "wp11a_fit_status.csv"), index=False)
        print(st.to_string(index=False), flush=True)

    per_seed = summ = None
    if args.score or args.curves or args.figures:
        per_seed, summ = part2_score(tags, fit_dir=args.fit_dir, out_dir=args.out)
    if args.noise_replicates:
        _pr, ns = part2b_noise_replicates(args.noise_replicates, out_dir=args.out)
        print("\n[target degradation across noise draws, median (IQR)]")
        print(ns.pivot(index="delta", columns="channel",
                       values="median").round(2).to_string(), flush=True)
        print(ns.pivot(index="delta", columns="channel",
                       values="iqr").round(2).to_string(), flush=True)
    if args.spectrum:
        sp = part3_spectrum(tags, fit_dir=args.fit_dir, out_dir=args.out)
        if sp is not None and not sp.empty:
            print(sp.to_string(index=False), flush=True)
    if args.curves and per_seed is not None and not per_seed.empty:
        freq, span = part4_curves(per_seed, out_dir=args.out)
        paired = part4b_paired(per_seed, out_dir=args.out)
        pred = part4c_prediction_check(per_seed, out_dir=args.out)
        if pred is not None and not pred.empty:
            print("\n[WP-11b's prediction vs the realised ratio to annual]")
            print(pred[pred.delta < 1.0].to_string(index=False), flush=True)
        if freq is not None:
            print("\n[frequency curve, median over seeds]")
            print(freq.groupby(["delta", "channel"]).alpha_relRMSE_true_pct
                  .median().unstack().round(2).to_string(), flush=True)
        if span is not None:
            for sp_ in ("fitted", "common"):
                sub = span[span.split == sp_]
                if sub.empty:
                    continue
                print(f"\n[span/matched-N curve, {sp_} window, "
                      f"median over seeds]")
                print(sub.groupby(["kind", "delta", "n_obs", "channel"])
                      .alpha_relRMSE_true_pct.median().unstack()
                      .round(2).to_string(), flush=True)
    if args.figures and per_seed is not None and not per_seed.empty:
        spec = None
        p = figures(per_seed, spec, out_dir=args.out)
        print(f"[figures] -> {p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
