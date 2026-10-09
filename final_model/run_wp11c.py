#!/usr/bin/env python3
"""
run_wp11c.py — WP-11c: value of information and observation ranking
===================================================================

Turns "the community should collect better data" into a prioritised list with
numbers attached.  Consumes the per-seed dumps written by `zinc_voi_lab.py`
(`analysis/wp11c/wp11c_seed*.npz`) and produces the deliverable the spec asks
for:

  Part 1  the ranking.  For every candidate observation — series x frequency —
          the percentage reduction in the marginal standard deviation of every
          coefficient, and the D-optimality gain `log det FIM_new -
          log det FIM_current`.  Median +/- IQR across the seed ensemble,
          Hodges-Lehmann point estimate and distribution-free CI on the
          headline effects.
  Part 2  frequency versus sample size.  The apparent value of monthly
          reporting is decomposed into the part that is finer RESOLUTION and
          the part that is simply MORE INDEPENDENT OBSERVATIONS, by holding
          the annual-equivalent precision of the series fixed.  The spec makes
          this distinction a design requirement of 11a because it changes the
          recommendation from "report monthly" to "keep reporting"; 11c can
          make it without any refits.
  Part 3  the free lunch.  The published anchor sets `stageB_w_F = 0`, so the
          flow integrals are not fitted directly at all.  Every annual flow
          candidate is therefore a series that is ALREADY REPORTED in the
          Rostek dataset and already on disk, and its VOI is available at zero
          collection cost.
  Part 4  validation, including the comparison against WP-11a that the spec
          calls for.

**The WP-11a comparison, stated first.**  The spec asks that the VOI
predictions be checked against WP-11a, "where the actual improvement from added
observations is known".  WP-11a was run on 2026-08-26 (160 fits on the
synthetic twin), so that comparison IS made here, by
`validate_against_wp11a()`.  Its three legs and their estimand mismatches are
documented on that function.  The outcome, in one line: the VOI's predicted
monthly/annual SD ratio matches 11a's realised alpha-recovery ratio to within
5-31% on four of five coefficients (median |error| 15.7% on the fitted window,
12.2% on the common one), with one badly wrong channel per scoring window and
not the same one — `alpha_cc` at +221% on the fitted window, `alpha_dr` at
-71% on the common; the ranking over coefficients is exact on the fitted
window once `alpha_cc` is set aside, and Spearman 0.4 with it; and 11c's
"only ~5% of the value of monthly reporting is within-year resolution" does NOT
survive 11a's matched-N arms.  Per the spec's own instruction the VOI is
therefore reported as **indicative rather than quantitative** — with the
refinement that it is the LEVEL on one channel and the resolution/sample-size
decomposition that fail, not the ordering.

Checked alongside it, and independent of 11a: the algebra against an exact
answer, the ranking against WP-3's independently measured per-series noise
ablation, and the first-order response of the marginals to a change in the
observation design against WP-4c's two synthetic arms, which are refits under
two different noise designs.

**One assumption the numbers rest on.**  `Sigma` is diagonal, so a candidate
is treated as an independent measurement.  That is right for a reporting
stream that does not exist yet — every sub-annual arm — and optimistic for
the annual arm of a series the objective already sees through the stock block
or through the `alpha_obs = F_obs/S_obs` target, where the same reported
number would be counted twice with independent errors.  Part 3's numbers are
therefore an upper bound; the sub-annual arms are not affected.

    python run_wp11c.py --check
    python run_wp11c.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

import zinc_voi_lab as V

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "analysis")
SEED_DIR = os.path.join(OUT_DIR, "wp11c")
NOTES_DIR = os.path.join(OUT_DIR, "notes")

# Okabe-Ito, colourblind-safe.
COL = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00", "#56B4E9",
       "#F0E442"]
GREY = "#555555"

ALPHA_CHANNELS = ["alpha_cc", "alpha_refc", "alpha_dr", "alpha_win"]
COHORT = ["f_cohort_10yr", "f_cohort_20yr", "f_cohort_44yr"]
HEADLINE_COEFS = ALPHA_CHANNELS + ["tau_olds"] + COHORT

# The FIM is built on the Stage B trainval window; `test` is the pre-registered
# held-out span.  2007 is the boundary year and belongs to the fitting window,
# so the two are disjoint by construction.
WINDOWS = {"trainval": lambda y: y <= 2007.0, "test": lambda y: y > 2007.0}

FREQ_ORDER = ["annual", "quarterly", "monthly"]

# Candidates that are bundles or derived duplicates, excluded from the
# "best single series" ranking so that a bundle cannot win a race it is not in.
BUNDLES = ("all_stocks", "all_flows")


# ---------------------------------------------------------------------------
# statistics — project convention (CLAUDE.md): median +/- IQR, HL intervals
# ---------------------------------------------------------------------------
def med_iqr(v):
    v = np.asarray([x for x in np.asarray(v, float) if np.isfinite(x)])
    if v.size == 0:
        return dict(median=np.nan, q1=np.nan, q3=np.nan, n=0)
    q1, q3 = np.percentile(v, [25, 75])
    return dict(median=float(np.median(v)), q1=float(q1), q3=float(q3),
                n=int(v.size))


def hodges_lehmann(v, conf=0.95):
    """HL location estimate + distribution-free CI.  Same implementation as
    `run_wp1b.py` / `run_wp2b.py` / `zinc_circ_lab.py`."""
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


def spearman(a, b):
    from scipy import stats

    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    return float(stats.spearmanr(a[ok], b[ok]).statistic)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_seeds(seed_dir=SEED_DIR):
    paths = sorted(glob.glob(os.path.join(seed_dir, "wp11c_seed*.npz")),
                   key=lambda p: int(os.path.basename(p)
                                     .split("seed")[1].split(".")[0]))
    if not paths:
        raise SystemExit(f"no per-seed dumps in {seed_dir} — "
                         f"run `python zinc_voi_lab.py --all-seeds` first")
    ds = [np.load(p, allow_pickle=True) for p in paths]
    seeds = [int(d["seed"]) for d in ds]
    return ds, seeds


def _key_frame(d):
    """The candidate index as a frame, one row per stored candidate."""
    return pd.DataFrame(dict(
        k=np.arange(len(d["cand_name"])),
        candidate=[str(x) for x in d["cand_name"]],
        freq=[str(x) for x in d["cand_freq"]],
        sigma_mode=[str(x) for x in d["cand_sigma_mode"]],
        sigma_mult=np.asarray(d["cand_sigma_mult"], float),
        m=np.asarray(d["cand_m"], int),
        logdet_gain=np.asarray(d["logdet_gain"], float)))


# ---------------------------------------------------------------------------
# Part 1 — the ranking
# ---------------------------------------------------------------------------
def part1(ds, seeds):
    """Per-seed reduction in marginal SD, then the seed-ensemble summary."""
    coef_names = [str(x) for x in ds[0]["coef_names"]]
    rows = []
    for d, s in zip(ds, seeds):
        years = np.asarray(d["years"], float).ravel()
        sb, sn = d["sd_base"], d["sd_new"]           # (T,17), (K,T,17)
        coef = np.asarray(d["coef"], float)
        est = np.asarray(d["estimated"], bool)
        key = _key_frame(d)
        rel_base = sb / np.maximum(np.abs(coef), 1e-300)
        ratio = sn / np.maximum(sb[None], 1e-300)    # (K,T,17)
        for wname, sel in WINDOWS.items():
            msk = sel(years)
            base_w = np.where(est[msk], rel_base[msk], np.nan)
            with np.errstate(invalid="ignore"):
                base_med = np.nanmedian(base_w, axis=0)          # (17,)
                r_med = np.nanmedian(
                    np.where(est[None, msk], ratio[:, msk], np.nan), axis=1)
            for r in key.itertuples():
                for c, cn in enumerate(coef_names):
                    if not np.isfinite(base_med[c]):
                        continue
                    rows.append(dict(
                        seed=s, candidate=r.candidate, freq=r.freq,
                        sigma_mode=r.sigma_mode, sigma_mult=r.sigma_mult,
                        m=r.m, logdet_gain=r.logdet_gain, window=wname,
                        coef=cn, rel_sd_base_pct=100.0 * base_med[c],
                        rel_sd_new_pct=100.0 * base_med[c] * r_med[r.k, c],
                        reduction_pct=100.0 * (1.0 - r_med[r.k, c])))
    per_seed = pd.DataFrame(rows)

    grp = per_seed.groupby(["candidate", "freq", "sigma_mode", "sigma_mult",
                            "window", "coef"], sort=False)
    out = []
    for k, g in grp:
        mi = med_iqr(g.reduction_pct)
        hl = hodges_lehmann(g.reduction_pct.to_numpy())
        bb = med_iqr(g.rel_sd_base_pct)
        nn = med_iqr(g.rel_sd_new_pct)
        ld = med_iqr(g.logdet_gain)
        out.append(dict(
            candidate=k[0], freq=k[1], sigma_mode=k[2], sigma_mult=k[3],
            window=k[4], coef=k[5], n_seeds=mi["n"], m=int(g.m.iloc[0]),
            rel_sd_base_pct=bb["median"], rel_sd_new_pct=nn["median"],
            reduction_pct=mi["median"], reduction_q1=mi["q1"],
            reduction_q3=mi["q3"], reduction_hl=hl["hl"],
            reduction_hl_lo=hl["lo"], reduction_hl_hi=hl["hi"],
            logdet_gain=ld["median"], logdet_q1=ld["q1"], logdet_q3=ld["q3"]))
    return per_seed, pd.DataFrame(out)


def headline(rank):
    """The deliverable-statement table: per coefficient, the best single
    reported series at each frequency, in the `per_obs` convention."""
    sub = rank[(rank.sigma_mode == "per_obs") & (rank.sigma_mult == 1.0)
               & (rank.window == "trainval")
               & (~rank.candidate.isin(BUNDLES))]
    rows = []
    for coef in HEADLINE_COEFS:
        s = sub[sub.coef == coef]
        for freq in FREQ_ORDER:
            f = s[s.freq == freq].sort_values("reduction_pct", ascending=False)
            if f.empty:
                continue
            top = f.iloc[0]
            rows.append(dict(coef=coef, freq=freq, best_series=top.candidate,
                             rel_sd_base_pct=top.rel_sd_base_pct,
                             rel_sd_new_pct=top.rel_sd_new_pct,
                             reduction_pct=top.reduction_pct,
                             reduction_hl_lo=top.reduction_hl_lo,
                             reduction_hl_hi=top.reduction_hl_hi,
                             runner_up=f.iloc[1].candidate if len(f) > 1 else "",
                             runner_up_pct=(f.iloc[1].reduction_pct
                                            if len(f) > 1 else np.nan)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part 2 — resolution versus sample size
# ---------------------------------------------------------------------------
def part2(rank):
    """Split the monthly/quarterly gain into resolution and sample size.

    `fixed_total` holds a year's worth of sub-annual reports to exactly the
    information content of one annual report, so whatever it still buys is
    RESOLUTION.  The remainder of the `per_obs` gain is the extra independent
    observations.  Both are expressed as reductions in marginal SD, and the
    split is reported as the resolution share of the total gain.
    """
    base = rank[(rank.sigma_mult == 1.0) & (rank.window == "trainval")]
    ann = base[(base.freq == "annual") & (base.sigma_mode == "per_obs")]
    ann = ann.set_index(["candidate", "coef"]).reduction_pct
    rows = []
    for freq in ("quarterly", "monthly"):
        per = base[(base.freq == freq) & (base.sigma_mode == "per_obs")]
        fix = base[(base.freq == freq) & (base.sigma_mode == "fixed_total")]
        fix = fix.set_index(["candidate", "coef"]).reduction_pct
        for r in per.itertuples():
            key = (r.candidate, r.coef)
            if key not in ann.index or key not in fix.index:
                continue
            a, f, p = float(ann[key]), float(fix[key]), float(r.reduction_pct)
            # Work in variance ratios so the shares compose: (1-red/100)^2 is
            # the variance ratio against the current posterior.
            va, vf, vp = [(1.0 - x / 100.0) ** 2 for x in (a, f, p)]
            tot = np.log(va / vp) if vp > 0 else np.nan     # total info gain
            res = np.log(va / vf) if vf > 0 else np.nan     # resolution only
            rows.append(dict(
                candidate=r.candidate, coef=r.coef, freq=freq,
                reduction_annual_pct=a, reduction_fixed_total_pct=f,
                reduction_per_obs_pct=p,
                resolution_share=(res / tot) if (np.isfinite(tot) and tot > 0)
                else np.nan,
                info_gain_total=tot, info_gain_resolution=res))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part 3 — the flows that are already reported and carry zero weight
# ---------------------------------------------------------------------------
def part3(rank, xlsx_check=True):
    """Annual flow candidates against `stageB_w_F = 0`.

    The published anchor gives the flow-divergence term zero weight, so no flow
    integral is fitted directly; the flows reach the objective only through the
    `alpha_obs = F_obs/S_obs` targets.  Every annual flow candidate is
    therefore a series already present in `flows_obs` and already on disk, and
    its VOI is a statement about the ESTIMATOR, not about data collection.
    """
    flows = [c["name"] for c in V.CANDIDATES
             if c["kind"] in ("flow", "flow_sum")]
    sub = rank[(rank.candidate.isin(flows)) & (rank.freq == "annual")
               & (rank.sigma_mode == "per_obs") & (rank.sigma_mult == 1.0)
               & (rank.window == "trainval") & (rank.coef.isin(HEADLINE_COEFS))]
    piv = sub.pivot(index="candidate", columns="coef",
                    values="reduction_pct").reindex(columns=HEADLINE_COEFS)

    cover = {}
    if xlsx_check:
        d = np.load(os.path.join(HERE, "anchor_v4", "pred_seed0.npz"),
                    allow_pickle=True)
        fn = [str(x) for x in d["flow_names"]]
        yr = np.asarray(d["years_flow"], float)
        tv = yr <= 2007.0
        for c in V.CANDIDATES:
            if c["kind"] == "flow" and c["member"] in fn:
                col = d["flows_obs"][tv, fn.index(c["member"])]
                cover[c["name"]] = int(np.isfinite(col).sum())
            elif c["kind"] == "flow_sum":
                ok = [m in fn for m in c["member"]]
                if all(ok):
                    cols = np.stack([d["flows_obs"][tv, fn.index(m)]
                                     for m in c["member"]], 1)
                    cover[c["name"]] = int(np.isfinite(cols).all(1).sum())
    piv = piv.assign(n_years_already_observed=[cover.get(i, 0)
                                               for i in piv.index])
    return piv.reset_index()


# ---------------------------------------------------------------------------
# Part 4 — validation
# ---------------------------------------------------------------------------
def validate_algebra(ds, seeds, wp4c_dir=None):
    """Checks with an exact answer: the baseline against WP-4c's stored
    marginals, the monotonicity of the update, and the nesting of the
    frequency arms."""
    wp4c_dir = wp4c_dir or os.path.join(OUT_DIR, "wp4c")
    rows = []
    for d, s in zip(ds, seeds):
        p = os.path.join(wp4c_dir, f"wp4c_seed{s}.npz")
        e_base = np.nan
        if os.path.exists(p):
            r = np.load(p, allow_pickle=True)
            k = int(np.argmin(np.abs(r["lam_rel"] - float(d["lam_rel"]))))
            ref = r["sd_by_lambda"][k]
            e_base = float(np.max(np.abs(d["sd_base"] - ref)
                                  / np.maximum(np.abs(ref), 1e-300)))
        gap = float(np.max(d["sd_new"] - d["sd_base"][None]))
        key = _key_frame(d)
        # nesting: for an integral series the coarse observation is an exact
        # sum of the fine ones, so information must be monotone in frequency
        # in BOTH sigma conventions.
        worst_nest = 0.0
        for (c, mo, mu), g in key.groupby(["candidate", "sigma_mode",
                                           "sigma_mult"]):
            kind = {x["name"]: x["kind"] for x in V.CANDIDATES}[c]
            if kind not in ("flow", "flow_sum", "composition"):
                continue
            idx = {r.freq: r.k for r in g.itertuples()}
            for a, b in (("annual", "quarterly"), ("quarterly", "monthly")):
                if a in idx and b in idx:
                    worst_nest = max(worst_nest, float(
                        np.max(d["sd_new"][idx[b]] - d["sd_new"][idx[a]])))
        rows.append(dict(seed=s, baseline_vs_wp4c_rel=e_base,
                         max_sd_increase=gap, max_nesting_violation=worst_nest,
                         logdet_min=float(np.min(d["logdet_gain"]))))
    return pd.DataFrame(rows)


def validate_wp3(rank, path=None):
    """The VOI ranking against WP-3's per-series noise ablation.

    WP-3 measured, on the twin, which single reported series carries each alpha
    channel's TARGET error; WP-11c computes which single reported series would
    most tighten the same channel's ESTIMATE.  They are different questions
    about the same object, and COMPUTE_STATUS asks 11c to reproduce or
    contradict the WP-3 answer.  Agreement is reported per channel rather than
    as one score, because a near-tie in both analyses is a different situation
    from a genuine contradiction.
    """
    path = path or os.path.join(OUT_DIR, "wp3_alpha_target_noise.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    w3 = pd.read_csv(path)
    w3 = w3[(w3.arm == "base") & (np.isclose(w3.delta, 1.0))]
    piv = w3.pivot(index="channel", columns="noise_group",
                   values="relRMSE_median_pct")
    # WP-3's own mapping of channel -> the series its target error travels
    # through (wp3_findings.md).  Kept explicit rather than re-derived.
    W3_SERIES = {"alpha_cc": "concentrate_stock",
                 "alpha_refc": "refined_stock",
                 "alpha_win": "waelz_input",
                 "alpha_dr": "direct_reuse_recycling"}
    sub = rank[(rank.sigma_mode == "per_obs") & (rank.sigma_mult == 1.0)
               & (rank.window == "trainval")
               & (~rank.candidate.isin(BUNDLES))]
    rows = []
    for ch, w3s in W3_SERIES.items():
        base = float(piv.loc[ch, "none"]) if ch in piv.index else np.nan
        st = float(piv.loc[ch, "parent_stock_only"]) if ch in piv.index else np.nan
        fl = float(piv.loc[ch, "parent_flow_only"]) if ch in piv.index else np.nan
        for freq in FREQ_ORDER:
            f = sub[(sub.coef == ch) & (sub.freq == freq)].sort_values(
                "reduction_pct", ascending=False)
            if f.empty:
                continue
            top, second = f.iloc[0], (f.iloc[1] if len(f) > 1 else None)
            rows.append(dict(
                channel=ch, freq=freq,
                wp3_dominant_series=w3s,
                wp3_target_err_none_pct=base,
                wp3_target_err_parent_stock_pct=st,
                wp3_target_err_parent_flow_pct=fl,
                voi_top_series=top.candidate,
                voi_top_reduction_pct=top.reduction_pct,
                voi_second_series=(second.candidate if second is not None else ""),
                voi_second_reduction_pct=(second.reduction_pct
                                          if second is not None else np.nan),
                agree=bool(top.candidate == w3s),
                voi_wp3_series_reduction_pct=float(
                    f[f.candidate == w3s].reduction_pct.iloc[0])
                if (f.candidate == w3s).any() else np.nan,
                wp3_series_voi_rank=int(
                    (f.candidate.tolist().index(w3s) + 1)
                    if w3s in f.candidate.tolist() else -1),
                voi_top3=" > ".join(
                    f"{r.candidate} {r.reduction_pct:.0f}%"
                    for r in f.head(3).itertuples())))
    return pd.DataFrame(rows)


def validate_design_response(synth_dir=None):
    """Does a first-order change in the observation design predict the
    realised change in the marginals?

    This is the closest thing to WP-11a that exists without refits.  WP-4c's
    synthetic arm fitted the SAME twin twice under two different observation
    noise designs — `base_d1y_clean` and `base_d1y_noisy` — and reported the
    marginals from both.  Under the local Gaussian approximation the whole
    effect of inflating observation noise is the `kappa` rescaling, so the
    predicted ratio of marginal SDs is `kappa_noisy / kappa_clean`, computed
    from the fits' own residuals and using nothing from the noisy arm's
    Jacobian.  The measured ratio comes from the refits.  The gap between them
    is exactly the non-locality this package is exposed to.
    """
    synth_dir = synth_dir or os.path.join(OUT_DIR, "wp4c_synth")
    arms = ("base_d1y_clean", "base_d1y_noisy")
    if not all(os.path.isdir(os.path.join(synth_dir, a)) for a in arms):
        return pd.DataFrame()
    got = {}
    for a in arms:
        fs = sorted(glob.glob(os.path.join(synth_dir, a, "*.npz")))
        kap, sds, names = [], [], None
        for f in fs:
            d = np.load(f, allow_pickle=True)
            kap.append(float(d["kappa2"]) ** 0.5)
            k = int(np.argmin(np.abs(d["lam_rel"] - V.LAMBDA_REF)))
            years = np.asarray(d["years"], float).ravel()
            m = years <= 2007.0
            rel = d["sd_by_lambda"][k] / np.maximum(np.abs(d["coef"]), 1e-300)
            with np.errstate(invalid="ignore"):
                sds.append(np.nanmedian(np.where(
                    np.asarray(d["estimated"], bool)[m], rel[m], np.nan), 0))
            names = [str(x) for x in d["coef_names"]]
        got[a] = dict(kappa=float(np.median(kap)),
                      sd=np.nanmedian(np.stack(sds), 0), names=names)
    pred = got[arms[1]]["kappa"] / got[arms[0]]["kappa"]
    rows = []
    for i, n in enumerate(got[arms[0]]["names"]):
        c, w = got[arms[0]]["sd"][i], got[arms[1]]["sd"][i]
        if not (np.isfinite(c) and np.isfinite(w) and c > 0):
            continue
        rows.append(dict(coefficient=n, sd_clean_pct=100 * c,
                         sd_noisy_pct=100 * w, ratio_measured=w / c,
                         ratio_predicted=pred,
                         rel_error_pct=100.0 * ((w / c) - pred) / pred))
    out = pd.DataFrame(rows)
    out.attrs["kappa_clean"] = got[arms[0]]["kappa"]
    out.attrs["kappa_noisy"] = got[arms[1]]["kappa"]
    return out


# The WP-11a arms this package can be scored against.  WP-11a densifies the
# WHOLE record on the synthetic twin at Delta in {1, 1/2, 1/4, 1/12}, so the
# comparable VOI object is a BUNDLE observed at the same frequency, not a
# single series: no arm of 11a adds one series at a time.
WP11A_FREQ_DELTA = {"annual": 1.0, "quarterly": 0.25, "monthly": 1.0 / 12.0}
WP11A_COEFS = ["alpha_cc", "alpha_refc", "alpha_win", "alpha_dr", "tau_olds"]


def _wp11a_realised(summary, split, coefs=WP11A_COEFS):
    """Realised recovery error on WP-11a's `freq` arm, as a ratio to annual."""
    g = summary[(summary.kind == "freq") & (summary.twin == "base")
                & (summary.split == split) & (summary.channel.isin(coefs))]
    piv = g.pivot_table(index="delta", columns="channel", values="pred_vs_true")
    n = g.pivot_table(index="delta", values="n_obs", aggfunc="first")["n_obs"]
    ann = piv.loc[1.0]
    return piv, piv.divide(ann, axis=1), n


def _wp11a_spectrum_ratio(spectrum, coefs=WP11A_COEFS):
    """WP-11a's own local-Gaussian side: the refitted marginal SD per Delta.

    This is the same machinery WP-11c runs (WP-4c's Gauss-Newton curvature at
    the optimum), evaluated on 11a's twin at each observation frequency, so it
    is the honest upper bound on what the VOI picture predicts when the WHOLE
    record is re-observed rather than one bundle added to it.
    """
    s = spectrum[spectrum.kind == "freq"].sort_values("delta")
    out = {}
    for c in coefs:
        col = f"sd_pct_{c}"
        if col not in s.columns:
            continue
        v = s.set_index("delta")[col]
        out[c] = v / v.loc[1.0]
    return pd.DataFrame(out)


def validate_against_wp11a(rank, out_dir=OUT_DIR):
    """The validation the spec asks for, and the one this package could not
    make until WP-11a existed.

    *"Check its predictions against 11a, where the actual improvement from
    added observations is known.  If the VOI ranking does not predict the
    realised improvements, report the discrepancy and present VOI as
    indicative rather than quantitative."*

    Three things are compared, and the estimands are NOT identical — say so
    rather than smoothing over it:

      Leg 1  LEVEL.  WP-11c predicts the ratio of marginal SDs when a bundle
             is observed at frequency `f` instead of annually.  WP-11a
             measures the ratio of realised alpha-recovery relRMSE when the
             whole record is observed at Delta instead of annually.  Both are
             "monthly against annual, same coefficient", and both are ratios,
             so they are directly comparable in the only sense that matters
             for the recommendation.
      Leg 2  RANKING.  Across coefficients, does the VOI order of predicted
             gain reproduce the realised order?  This is the object the spec
             names.  11a cannot vary series one at a time, so the ranking that
             can be tested is the ranking OVER COEFFICIENTS at fixed
             observation design, not the ranking over series.  Stated as a
             limitation, not worked around.
      Leg 3  THE DECOMPOSITION.  WP-11c's headline claim is that ~5% of the
             value of monthly reporting is within-year RESOLUTION and the rest
             is extra independent observations.  WP-11a's matched-N arms hold
             the observation count fixed and vary Delta, which is the closest
             refitted test of that claim that exists.  The two estimands
             differ — 11c holds total PRECISION fixed at a fixed span, 11a
             holds COUNT fixed and shortens the span — and the difference is
             reported with the result.

    Returns `(per_coef, resolution, summary_rows)`; empty frames if WP-11a's
    outputs are not on disk.
    """
    fs = os.path.join(out_dir, "wp11a_summary.csv")
    fp = os.path.join(out_dir, "wp11a_spectrum.csv")
    if not os.path.exists(fs):
        return pd.DataFrame(), pd.DataFrame(), []
    summary = pd.read_csv(fs)
    spectrum = pd.read_csv(fp) if os.path.exists(fp) else pd.DataFrame()

    # ---- the VOI side: bundles, per_obs, reference sigma, fitting window ---
    r = rank[(rank.sigma_mode == "per_obs") & (rank.sigma_mult == 1.0)
             & (rank.window == "trainval") & (rank.coef.isin(WP11A_COEFS))]
    pred = {}
    for b in BUNDLES:
        p = r[r.candidate == b].pivot_table(
            index="freq", columns="coef", values="rel_sd_new_pct")
        if p.empty or "annual" not in p.index:
            continue
        pred[b] = p.divide(p.loc["annual"], axis=1)

    specr = (_wp11a_spectrum_ratio(spectrum) if not spectrum.empty
             else pd.DataFrame())

    rows = []
    for split in ("fitted", "common", "test"):
        lvl, ratio, nobs = _wp11a_realised(summary, split)
        if ratio.empty:
            continue
        for freq, delta in WP11A_FREQ_DELTA.items():
            if delta == 1.0:
                continue
            hit = [d for d in ratio.index if abs(d - delta) < 1e-9]
            if not hit:
                continue
            d = hit[0]
            for c in WP11A_COEFS:
                if c not in ratio.columns:
                    continue
                row = dict(coef=c, freq=freq, delta=d,
                           split=split, n_obs=int(nobs.loc[d]),
                           realised_relrmse_pct=float(lvl.loc[d, c]),
                           realised_annual_pct=float(lvl.loc[1.0, c]),
                           realised_ratio=float(ratio.loc[d, c]))
                for b, p in pred.items():
                    v = float(p.loc[freq, c]) if c in p.columns else np.nan
                    row[f"voi_ratio_{b}"] = v
                    row[f"rel_error_{b}_pct"] = (
                        100.0 * (row["realised_ratio"] - v) / v
                        if np.isfinite(v) and v > 0 else np.nan)
                sr = (float(specr.loc[d, c])
                      if (not specr.empty and c in specr.columns
                          and d in specr.index) else np.nan)
                row["wp11a_spectrum_ratio"] = sr
                row["rel_error_spectrum_pct"] = (
                    100.0 * (row["realised_ratio"] - sr) / sr
                    if np.isfinite(sr) and sr > 0 else np.nan)
                rows.append(row)
    per_coef = pd.DataFrame(rows)

    # ---- Leg 3: the resolution-vs-sample-size decomposition ---------------
    res_rows = []
    fdec = os.path.join(out_dir, "wp11c_frequency_decomposition.csv")
    share = pd.read_csv(fdec) if os.path.exists(fdec) else pd.DataFrame()
    ft = rank[(rank.sigma_mode == "fixed_total") & (rank.sigma_mult == 1.0)
              & (rank.window == "trainval") & (rank.coef.isin(WP11A_COEFS))]
    for split in ("fitted", "common"):
        lvl, _, _ = _wp11a_realised(summary, split)
        m = summary[(summary.kind == "matchedN") & (summary.twin == "base")
                    & (summary.split == split)
                    & (summary.channel.isin(WP11A_COEFS))]
        if lvl.empty or m.empty:
            continue
        fq = summary[(summary.kind == "freq") & (summary.twin == "base")
                     & (summary.split == split)
                     & (summary.channel.isin(WP11A_COEFS))]
        for tag, gm in m.groupby("tag"):
            n = int(gm.n_obs.iloc[0])
            # the `freq` arm at the same observation count, coarser Delta
            cand = fq.assign(dn=(fq.n_obs - n).abs()).sort_values("dn")
            if cand.empty:
                continue
            ntag = cand.tag.iloc[0]
            gf = fq[fq.tag == ntag]
            for c in WP11A_COEFS:
                a = gm[gm.channel == c]
                b = gf[gf.channel == c]
                if a.empty or b.empty:
                    continue
                fine = float(a.pred_vs_true.iloc[0])
                coarse = float(b.pred_vs_true.iloc[0])
                pf = ft[(ft.coef == c) & (ft.candidate == "all_flows")]
                pv = pf.pivot_table(index="freq", values="rel_sd_new_pct")
                voi_ft = (float(pv.loc["monthly", "rel_sd_new_pct"]
                                / pv.loc["annual", "rel_sd_new_pct"])
                          if {"monthly", "annual"} <= set(pv.index)
                          else np.nan)
                sh = share[(share.candidate == "all_flows")
                           & (share.coef == c) & (share.freq == "monthly")]
                res_rows.append(dict(
                    split=split, coef=c,
                    fine_tag=tag, fine_delta=float(gm.delta.iloc[0]),
                    fine_span=float(gm.span.iloc[0]), n_obs=n,
                    coarse_tag=ntag, coarse_delta=float(gf.delta.iloc[0]),
                    coarse_span=float(gf.span.iloc[0]),
                    coarse_n_obs=int(gf.n_obs.iloc[0]),
                    fine_relrmse_pct=fine, coarse_relrmse_pct=coarse,
                    realised_ratio_fine_over_coarse=fine / coarse,
                    voi_fixed_total_ratio=voi_ft,
                    voi_resolution_share=(float(sh.resolution_share.iloc[0])
                                          if not sh.empty else np.nan)))
    resolution = pd.DataFrame(res_rows)

    # ---- the numbers that go into the validation table --------------------
    out = []
    if not per_coef.empty:
        for split in ("fitted", "common"):
            g = per_coef[(per_coef.split == split)
                         & (per_coef.freq == "monthly")]
            if g.empty:
                continue
            e = g.rel_error_all_flows_pct.abs()
            out.append(dict(
                check=f"WP-11a level test, monthly, {split} window: median "
                      f"|error| of the VOI ratio (%)",
                value=float(np.nanmedian(e)), exact="0"))
            out.append(dict(
                check=f"WP-11a level test, monthly, {split} window: worst "
                      f"|error| (%)",
                value=float(np.nanmax(e)), exact="0"))
            out.append(dict(
                check=f"WP-11a ranking test, monthly, {split} window: "
                      f"Spearman over {len(g)} coefficients",
                value=spearman(g.voi_ratio_all_flows.to_numpy(),
                               g.realised_ratio.to_numpy()), exact="1"))
            gx = g[g.coef != "alpha_cc"]
            out.append(dict(
                check=f"WP-11a ranking test, monthly, {split} window: "
                      f"Spearman excluding alpha_cc",
                value=spearman(gx.voi_ratio_all_flows.to_numpy(),
                               gx.realised_ratio.to_numpy()), exact="1"))
        g = per_coef[(per_coef.split == "fitted")
                     & (per_coef.freq == "monthly")]
        if not g.empty and g.rel_error_spectrum_pct.notna().any():
            out.append(dict(
                check="WP-11a level test, monthly, fitted window: median "
                      "|error| of 11a's OWN refitted spectrum (%)",
                value=float(np.nanmedian(g.rel_error_spectrum_pct.abs())),
                exact="0"))
    if not resolution.empty:
        for split in ("common", "fitted"):
            g = resolution[resolution.split == split]
            if g.empty:
                continue
            out.append(dict(
                check=f"WP-11a resolution test, {split} window: realised gain "
                      f"at matched N from finer Delta (%), median",
                value=float(100.0 * (1.0 - np.nanmedian(
                    g.realised_ratio_fine_over_coarse))), exact="—"))
            out.append(dict(
                check=f"WP-11a resolution test, {split} window: pairs of "
                      f"{len(g)} where finer Delta wins at matched N",
                value=float((g.realised_ratio_fine_over_coarse < 1.0).sum()),
                exact="—"))
        out.append(dict(
            check="WP-11a resolution test: VOI-predicted gain at fixed total "
                  "precision (%), median",
            value=float(100.0 * (1.0 - np.nanmedian(
                resolution.voi_fixed_total_ratio))), exact="—"))
    return per_coef, resolution, out


def validate_stability(per_seed):
    """Is the RANKING a property of the data, or of the seed?

    WP-4c made the same distinction for the identifiability ordering (the level
    moves with the ridge, the order does not).  Here the ordering of candidates
    within each alpha channel is correlated, seed by seed, against the ordering
    of the pooled median.
    """
    sub = per_seed[(per_seed.sigma_mode == "per_obs")
                   & (per_seed.sigma_mult == 1.0)
                   & (per_seed.window == "trainval")
                   & (per_seed.freq == "monthly")
                   & (~per_seed.candidate.isin(BUNDLES))]
    rows = []
    for coef, g in sub.groupby("coef"):
        ref = g.groupby("candidate").reduction_pct.median()
        for s, gs in g.groupby("seed"):
            v = gs.set_index("candidate").reduction_pct.reindex(ref.index)
            rows.append(dict(coef=coef, seed=int(s),
                             spearman_vs_pooled=spearman(v.to_numpy(),
                                                         ref.to_numpy())))
    return pd.DataFrame(rows)


def validate_ridge(seed_dir=SEED_DIR, pattern="wp11c_lam*"):
    """Is the RANKING a property of the data, or of the regulariser?

    WP-4c had to answer the same question about its identifiability ordering,
    because with a spectrum spanning eleven decades there is no scale at which
    "the" marginal uncertainty is defined without a prior.  Its answer was that
    the LEVEL moves with the ridge and the ORDER does not.  The same test is
    run here on the candidate ordering: the whole VOI calculation is repeated
    at `lambda/lambda_1` two decades either side of the reference on a subset
    of seeds, and the ordering is correlated against the reference ordering.
    """
    alt = sorted(glob.glob(os.path.join(os.path.dirname(seed_dir), pattern)))
    if not alt:
        return pd.DataFrame()

    def order(d):
        key = _key_frame(d)
        years = np.asarray(d["years"], float).ravel()
        m = years <= 2007.0
        est = np.asarray(d["estimated"], bool)
        with np.errstate(invalid="ignore"):
            ratio = np.nanmedian(np.where(
                est[None, m], d["sd_new"][:, m] / np.maximum(d["sd_base"][m], 1e-300),
                np.nan), axis=1)
        key = key.assign(**{c: 100.0 * (1.0 - ratio[:, i])
                            for i, c in enumerate(str(x) for x
                                                  in d["coef_names"])})
        sel = ((key.sigma_mode == "per_obs") & (key.sigma_mult == 1.0)
               & (key.freq == "monthly")
               & (~key.candidate.isin(BUNDLES)))
        return key[sel].set_index("candidate")

    rows = []
    for a in alt:
        lam_tag = os.path.basename(a).split("lam")[-1]
        for p in sorted(glob.glob(os.path.join(a, "wp11c_seed*.npz"))):
            s = int(os.path.basename(p).split("seed")[1].split(".")[0])
            ref_p = os.path.join(seed_dir, f"wp11c_seed{s}.npz")
            if not os.path.exists(ref_p):
                continue
            oa = order(np.load(p, allow_pickle=True))
            ob = order(np.load(ref_p, allow_pickle=True))
            for c in HEADLINE_COEFS:
                if c not in oa.columns:
                    continue
                v = oa[c].reindex(ob.index)
                rows.append(dict(lam_rel=lam_tag, seed=s, coef=c,
                                 spearman_vs_reference=spearman(
                                     v.to_numpy(), ob[c].to_numpy()),
                                 level_ratio=float(np.nanmedian(
                                     v.to_numpy() / np.maximum(
                                         ob[c].to_numpy(), 1e-12)))))
    return pd.DataFrame(rows)


def validate_identity(rank):
    """`primary_refining` against `concentrate_consumption`.

    `primary_refining = (1 - tau_ref) * concentrate_consumption` pointwise, and
    `tau_ref` is PINNED to data, so in log space the two differ by a term with
    no parameter dependence at all.  Over a period INTEGRAL the cancellation is
    not quite exact — the integral of a product is not the product of integrals
    unless `tau_ref` is constant across the window — so the two VOI columns
    must agree to the size of `tau_ref`'s within-window variation and not
    further.  A gap of order 1e-3 pp is the expected answer; a large one would
    mean the aggregation or the noise model is wrong.
    """
    a = rank[(rank.candidate == "primary_refining")].set_index(
        ["freq", "sigma_mode", "sigma_mult", "window", "coef"]).reduction_pct
    b = rank[(rank.candidate == "concentrate_consumption")].set_index(
        ["freq", "sigma_mode", "sigma_mult", "window", "coef"]).reduction_pct
    j = a.align(b, join="inner")
    return float(np.max(np.abs(j[0] - j[1]))) if len(j[0]) else float("nan")


# ---------------------------------------------------------------------------
# Part 5 — the deliverable statements
# ---------------------------------------------------------------------------
# Exactly the candidates the spec names, in the sentence form the spec asks
# for, so the paragraph a reporting body reads is generated from the table
# rather than transcribed from it.
SPEC_CANDIDATES = [
    ("refined_stock", "monthly", "Monthly reporting of refined metal stock"),
    ("refined_production", "monthly", "Monthly reporting of refined production"),
    ("concentrate_production", "monthly",
     "Monthly reporting of concentrate production"),
    ("scrap_collection", "annual",
     "Annual reporting of old-scrap collection tonnage"),
    ("eol_composition", "annual",
     "Annual end-of-life product-level composition"),
    ("refined_stock", "quarterly", "Quarterly reporting of refined metal stock"),
    ("refined_production", "quarterly",
     "Quarterly reporting of refined production"),
    ("concentrate_production", "quarterly",
     "Quarterly reporting of concentrate production"),
    ("scrap_collection", "quarterly",
     "Quarterly reporting of old-scrap collection tonnage"),
    ("eol_composition", "quarterly",
     "Quarterly end-of-life product-level composition"),
]


def statements(rank, coefs=None):
    """The spec's named candidates, rendered as sentences."""
    coefs = coefs or (ALPHA_CHANNELS + ["tau_olds", "f_cohort_20yr"])
    sub = rank[(rank.sigma_mode == "per_obs") & (rank.sigma_mult == 1.0)
               & (rank.window == "trainval")]
    lines, rows = [], []
    for cand, freq, phrase in SPEC_CANDIDATES:
        g = sub[(sub.candidate == cand) & (sub.freq == freq)
                & (sub.coef.isin(coefs))].set_index("coef").reindex(coefs)
        if g.reduction_pct.isna().all():
            continue
        parts = []
        for c in coefs:
            r = g.loc[c]
            if not np.isfinite(r.reduction_pct):
                continue
            parts.append(f"{c} by {r.reduction_pct:.0f}%"
                         if r.reduction_pct >= 1.0
                         else f"{c} by under 1%")
            rows.append(dict(statement=phrase, candidate=cand, freq=freq,
                             coef=c, reduction_pct=r.reduction_pct,
                             rel_sd_base_pct=r.rel_sd_base_pct,
                             rel_sd_new_pct=r.rel_sd_new_pct,
                             hl_lo=r.reduction_hl_lo, hl_hi=r.reduction_hl_hi))
        lines.append(f"- {phrase} reduces uncertainty in "
                     + "; ".join(parts) + ".")
    return lines, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def figure_wp11a(a_coef, a_res, out_dir):
    """The validation the spec asks for, as a picture.

    Left: what the VOI predicted against what WP-11a's refits delivered, one
    point per coefficient per scoring window, with the 1:1 line.  Right: the
    resolution test — WP-11c predicts almost no gain from finer observation at
    fixed measurement effort, WP-11a's matched-N refits deliver a large one.
    """
    if a_coef.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

    ax = axes[0]
    g = a_coef[a_coef.freq == "monthly"]
    marks = {"fitted": "o", "common": "s", "test": "^"}
    for j, c in enumerate(WP11A_COEFS):
        for split, mk in marks.items():
            r = g[(g.coef == c) & (g.split == split)]
            if r.empty:
                continue
            ax.scatter(r.voi_ratio_all_flows, r.realised_ratio, s=46, marker=mk,
                       color=COL[j % len(COL)], alpha=0.9,
                       label=c if split == "fitted" else None)
    sp = a_coef[(a_coef.freq == "monthly") & (a_coef.split == "fitted")]
    for j, c in enumerate(WP11A_COEFS):
        r = sp[sp.coef == c]
        if r.empty or not np.isfinite(r.wp11a_spectrum_ratio.iloc[0]):
            continue
        ax.scatter(r.wp11a_spectrum_ratio, r.realised_ratio, s=46, marker="x",
                   color=COL[j % len(COL)], alpha=0.9)
    lim = [0.0, max(1.75, float(np.nanmax(g.realised_ratio)) * 1.05)]
    ax.plot(lim, lim, color=GREY, lw=1.0, ls="--")
    ax.axhline(1.0, color=GREY, lw=0.7, alpha=0.5)
    ax.set_xlim(0, 0.95)
    ax.set_ylim(lim)
    ax.set_xlabel("predicted monthly/annual marginal SD ratio\n"
                  "(filled = WP-11c `all_flows`; "
                  "cross = WP-11a's own refitted spectrum)")
    ax.set_ylabel("realised monthly/annual α-recovery ratio (WP-11a refits)")
    ax.set_title("(a) the VOI against the refits\n"
                 "circle = fitted window, square = common, triangle = test")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    if not a_res.empty:
        r = a_res[a_res.split == "common"]
        y = np.arange(len(r))
        ax.barh(y + 0.19, 100.0 * (1.0 - r.realised_ratio_fine_over_coarse),
                height=0.36, color=COL[2], label="realised (WP-11a refits)")
        ax.barh(y - 0.19, 100.0 * (1.0 - r.voi_fixed_total_ratio),
                height=0.36, color=COL[0],
                label="WP-11c prediction (fixed total precision)")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{t.coef} · N={t.n_obs}" for t in r.itertuples()],
                           fontsize=8)
        ax.axvline(0.0, color=GREY, lw=0.8)
        ax.set_xlabel("gain from finer Δ at matched observation count (%)")
        ax.set_title("(b) the resolution-versus-sample-size decomposition\n"
                     "does not survive (common scoring window)")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.25, axis="x")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"wp11c_wp11a_validation.{ext}"),
                    dpi=200)
    plt.close(fig)


def figures(rank, freq_dec, w3, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = rank[(rank.sigma_mode == "per_obs") & (rank.sigma_mult == 1.0)
               & (rank.window == "trainval") & (~rank.candidate.isin(BUNDLES))]
    show = ALPHA_CHANNELS + ["tau_olds", "f_cohort_20yr"]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))

    # ---- (a) the ranking, monthly ---------------------------------------
    ax = axes[0, 0]
    mon = sub[sub.freq == "monthly"]
    piv = mon.pivot(index="candidate", columns="coef",
                    values="reduction_pct").reindex(columns=show)
    piv = piv.loc[piv.max(1).sort_values().index]
    ypos = np.arange(len(piv))
    h = 0.8 / len(show)
    for j, c in enumerate(show):
        ax.barh(ypos + j * h - 0.4 + h / 2, piv[c].to_numpy(), height=h,
                color=COL[j % len(COL)], label=c)
    ax.set_yticks(ypos)
    ax.set_yticklabels(piv.index, fontsize=7.5)
    ax.set_xlabel("reduction in marginal SD (%), median over 1980–2007 and seeds")
    ax.set_title("(a) value of one added series, monthly reporting")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(alpha=0.25, axis="x")

    # ---- (b) frequency curves, per_obs vs fixed_total --------------------
    ax = axes[0, 1]
    best = {c: sub[(sub.coef == c) & (sub.freq == "monthly")]
            .sort_values("reduction_pct", ascending=False).candidate.iloc[0]
            for c in ALPHA_CHANNELS}
    x = np.arange(len(FREQ_ORDER))
    for j, c in enumerate(ALPHA_CHANNELS):
        for mode, ls, mk in (("per_obs", "-", "o"), ("fixed_total", "--", "s")):
            g = rank[(rank.candidate == best[c]) & (rank.coef == c)
                     & (rank.sigma_mode == mode) & (rank.sigma_mult == 1.0)
                     & (rank.window == "trainval")]
            g = g.set_index("freq").reindex(FREQ_ORDER)
            ax.plot(x, g.reduction_pct.to_numpy(), ls=ls, marker=mk, ms=4,
                    color=COL[j % len(COL)], lw=1.5,
                    label=f"{c} · {best[c]}" if mode == "per_obs" else None)
    ax.set_xticks(x)
    ax.set_xticklabels(FREQ_ORDER)
    ax.set_ylabel("reduction in marginal SD (%)")
    ax.set_title("(b) frequency: solid = more observations + resolution,\n"
                 "dashed = resolution only (annual-equivalent precision fixed)")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.25)

    # ---- (c) D-optimality against the coefficient criterion --------------
    ax = axes[1, 0]
    for j, c in enumerate(("alpha_win", "f_cohort_20yr")):
        g = sub[sub.coef == c]
        ax.scatter(g.logdet_gain, g.reduction_pct, s=22,
                   color=COL[j % len(COL)], alpha=0.85, label=c)
        for r in g.itertuples():
            if r.reduction_pct > 25:
                ax.annotate(f"{r.candidate}·{r.freq[:3]}",
                            (r.logdet_gain, r.reduction_pct), fontsize=6,
                            xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel(r"D-optimality gain, $\log\det \mathrm{FIM}_{new}"
                  r" - \log\det \mathrm{FIM}_{cur}$")
    ax.set_ylabel("reduction in marginal SD (%)")
    ax.set_title("(c) the two criteria do not rank the same way\n"
                 "(labelled where the coefficient criterion exceeds 25%)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)

    # ---- (d) the cohort split, and the sigma it assumes ------------------
    ax = axes[1, 1]
    for j, c in enumerate(COHORT):
        for mult, ls in ((1.0, "-"), (2.0, "--"), (5.0, ":")):
            g = rank[(rank.candidate == "eol_composition") & (rank.coef == c)
                     & (rank.sigma_mode == "per_obs")
                     & (np.isclose(rank.sigma_mult, mult))
                     & (rank.window == "trainval")]
            g = g.set_index("freq").reindex(FREQ_ORDER)
            ax.plot(np.arange(3), g.reduction_pct.to_numpy(), ls=ls,
                    marker="o", ms=4, color=COL[j % len(COL)], lw=1.4,
                    label=f"{c}" if mult == 1.0 else None)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(FREQ_ORDER)
    ax.set_ylabel("reduction in marginal SD (%)")
    ax.set_title("(d) EoL product composition on the in-use cohort split\n"
                 r"(solid $\sigma$, dashed $2\sigma$, dotted $5\sigma$)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"wp11c_voi_ranking.{ext}"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--seed-dir", default=SEED_DIR)
    args = ap.parse_args(argv)

    if args.check:
        V.check(verbose=True)
        print("run_wp11c --check")
        print("=" * 74)
        print(f"  windows              : trainval = years <= 2007 "
              f"(the FIM window), test = years > 2007")
        print(f"  headline coefficients: {HEADLINE_COEFS}")
        print(f"  bundles excluded from the single-series ranking: {BUNDLES}")
        have = os.path.exists(os.path.join(args.out, "wp11a_summary.csv"))
        print(f"  VALIDATION AGAINST WP-11a: "
              f"{'RUN' if have else 'NOT POSSIBLE — 11a outputs missing'}"
              f" (needs analysis/wp11a_summary.csv,")
        print( "                         wp11a_spectrum.csv).  Three legs: "
               "the LEVEL of the predicted")
        print( "                         monthly/annual SD ratio, the RANKING "
               "over coefficients, and")
        print( "                         the resolution-vs-sample-size "
               "decomposition against 11a's")
        print( "                         matched-N arms.  The VOI is reported "
               "as INDICATIVE, per the")
        print( "                         spec's own instruction.")
        print( "  checks independent of 11a: the algebra against an exact "
               "answer; the ranking")
        print( "                         against WP-3's per-series noise "
               "ablation; the first-order")
        print( "                         design response against WP-4c's two "
               "synthetic arms.")
        print("=" * 74)
        return 0

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(NOTES_DIR, exist_ok=True)
    ds, seeds = load_seeds(args.seed_dir)
    print(f"loaded {len(seeds)} seeds from {args.seed_dir}")

    # ---- candidate registry ---------------------------------------------
    sigma = V.load_sigma()
    reg = []
    for c in V.CANDIDATES:
        mem = c.get("member", "")
        mem = ", ".join(mem) if isinstance(mem, tuple) else str(mem)
        if c["kind"] == "stock":
            s = sigma["stock"][c["member"]]
        elif c["kind"] == "flow":
            s = sigma["flow"][c["member"]]
        elif c["kind"] == "composition":
            s = sigma["flow"]["end_of_life"] * np.sqrt(2.0)
        else:
            s = np.nan
        op = dict(stock="point-in-time", flow="period integral",
                  flow_sum="period integral of a sum",
                  composition="log-ratios of period integrals",
                  bundle="joint")[c["kind"]]
        reg.append(dict(candidate=c["name"], kind=c["kind"], operator=op,
                        members=mem, sigma_rel=s))
    pd.DataFrame(reg).to_csv(
        os.path.join(args.out, "wp11c_candidates.csv"), index=False)

    # ---- Part 1 ----------------------------------------------------------
    per_seed, rank = part1(ds, seeds)
    per_seed.to_csv(os.path.join(args.out, "wp11c_voi_per_seed.csv"),
                    index=False)
    rank.to_csv(os.path.join(args.out, "wp11c_voi_ranking.csv"), index=False)
    head = headline(rank)
    head.to_csv(os.path.join(args.out, "wp11c_voi_headline.csv"), index=False)

    dopt = (rank[(rank.sigma_mode == "per_obs") & (rank.sigma_mult == 1.0)
                 & (rank.window == "trainval") & (rank.coef == "alpha_win")]
            [["candidate", "freq", "m", "logdet_gain", "logdet_q1",
              "logdet_q3"]].sort_values("logdet_gain", ascending=False))
    dopt.to_csv(os.path.join(args.out, "wp11c_dopt.csv"), index=False)

    print("\n  Part 1 — best single reported series per coefficient "
          "(monthly, per-observation sigma):")
    for r in head[head.freq == "monthly"].itertuples():
        print(f"    {r.coef:15s} {r.best_series:24s} "
              f"{r.rel_sd_base_pct:6.2f}% -> {r.rel_sd_new_pct:6.2f}%  "
              f"({r.reduction_pct:5.1f}% reduction, "
              f"HL 95% [{r.reduction_hl_lo:.1f}, {r.reduction_hl_hi:.1f}])"
              f"   runner-up {r.runner_up} ({r.runner_up_pct:.1f}%)")

    # ---- Part 2 ----------------------------------------------------------
    fdec = part2(rank)
    fdec.to_csv(os.path.join(args.out, "wp11c_frequency_decomposition.csv"),
                index=False)
    m = fdec[(fdec.freq == "monthly") & (fdec.coef.isin(ALPHA_CHANNELS))]
    print(f"\n  Part 2 — of the information gain from monthly reporting, the "
          f"share that is RESOLUTION\n"
          f"           rather than extra observations: median "
          f"{np.nanmedian(m.resolution_share)*100:.0f}% "
          f"(IQR {np.nanpercentile(m.resolution_share,25)*100:.0f}"
          f"–{np.nanpercentile(m.resolution_share,75)*100:.0f}%) "
          f"over alpha channels x series")
    for c in ALPHA_CHANNELS:
        b = head[(head.coef == c) & (head.freq == "monthly")]
        if b.empty:
            continue
        ser = b.best_series.iloc[0]
        g = fdec[(fdec.coef == c) & (fdec.candidate == ser)
                 & (fdec.freq == "monthly")]
        if g.empty:
            continue
        g = g.iloc[0]
        print(f"           {c:12s} {ser:24s} annual "
              f"{g.reduction_annual_pct:5.1f}%  -> monthly at fixed annual "
              f"precision {g.reduction_fixed_total_pct:5.1f}%  -> monthly "
              f"{g.reduction_per_obs_pct:5.1f}%   "
              f"(resolution share {g.resolution_share*100:.0f}%)")

    # ---- Part 3 ----------------------------------------------------------
    free = part3(rank)
    free.to_csv(os.path.join(args.out, "wp11c_already_reported.csv"),
                index=False)
    print("\n  Part 3 — annual flow series already in `flows_obs` and already "
          "on disk (stageB_w_F = 0):")
    for r in free.sort_values("alpha_win", ascending=False).itertuples():
        print(f"    {r.candidate:24s} obs {r.n_years_already_observed:2d}/27 yr"
              f"   cc {getattr(r, 'alpha_cc'):5.1f}%  "
              f"refc {getattr(r, 'alpha_refc'):5.1f}%  "
              f"dr {getattr(r, 'alpha_dr'):5.1f}%  "
              f"win {getattr(r, 'alpha_win'):5.1f}%  "
              f"tau_olds {getattr(r, 'tau_olds'):5.1f}%")

    # ---- Part 4 ----------------------------------------------------------
    alg = validate_algebra(ds, seeds)
    w3 = validate_wp3(rank)
    dr = validate_design_response()
    stab = validate_stability(per_seed)
    ridge = validate_ridge(args.seed_dir)
    ident = validate_identity(rank)
    a_coef, a_res, a_rows = validate_against_wp11a(rank, args.out)

    vrows = [
        dict(check="baseline marginal SD vs wp4c sd_by_lambda (max rel, "
                   "worst seed)", value=float(alg.baseline_vs_wp4c_rel.max()),
             exact="0"),
        dict(check="max increase in marginal SD from adding data (kt units)",
             value=float(alg.max_sd_increase.max()), exact="<= 0"),
        dict(check="max frequency-nesting violation, integral series",
             value=float(alg.max_nesting_violation.max()), exact="<= 0"),
        dict(check="min D-optimality gain over all candidates",
             value=float(alg.logdet_min.min()), exact=">= 0"),
        dict(check="primary_refining vs concentrate_consumption "
                   "(pinned tau_ref near-identity), max abs pp",
             value=ident, exact="~1e-3, see validate_identity"),
        dict(check="ranking Spearman vs pooled, median over seeds and coefs",
             value=float(stab.spearman_vs_pooled.median()), exact="1"),
    ]
    if not dr.empty:
        vrows += [
            dict(check="design-response test: predicted SD ratio "
                       "kappa_noisy/kappa_clean",
                 value=float(dr.ratio_predicted.iloc[0]), exact="—"),
            dict(check="design-response test: median |error| of that "
                       "prediction (%)",
                 value=float(np.median(np.abs(dr.rel_error_pct))), exact="0"),
            dict(check="design-response test: worst |error| (%)",
                 value=float(np.max(np.abs(dr.rel_error_pct))), exact="0"),
        ]
    if not ridge.empty:
        vrows.append(dict(check="ranking Spearman vs the reference ridge, "
                                "lambda/lambda_1 from 1e-6 to 1e-2, worst",
                          value=float(ridge.spearman_vs_reference.min()),
                          exact="1"))
    vrows += a_rows
    if not w3.empty:
        mon = w3[w3.freq == "monthly"]
        vrows.append(dict(check="alpha channels where the VOI top series "
                                "matches WP-3's noise ablation (of 4)",
                          value=float(mon.agree.sum()), exact="4"))
    pd.DataFrame(vrows).to_csv(
        os.path.join(args.out, "wp11c_validation.csv"), index=False)
    alg.to_csv(os.path.join(args.out, "wp11c_validation_per_seed.csv"),
               index=False)
    if not w3.empty:
        w3.to_csv(os.path.join(args.out, "wp11c_wp3_agreement.csv"),
                  index=False)
    if not dr.empty:
        dr.to_csv(os.path.join(args.out, "wp11c_design_response.csv"),
                  index=False)
    if not a_coef.empty:
        a_coef.to_csv(os.path.join(args.out, "wp11c_wp11a_validation.csv"),
                      index=False)
    if not a_res.empty:
        a_res.to_csv(os.path.join(args.out, "wp11c_wp11a_resolution.csv"),
                     index=False)
    stab.to_csv(os.path.join(args.out, "wp11c_ranking_stability.csv"),
                index=False)
    if not ridge.empty:
        ridge.to_csv(os.path.join(args.out, "wp11c_ridge_stability.csv"),
                     index=False)

    print("\n  Part 4 — validation:")
    for r in vrows:
        print(f"    {r['check']:66s} {r['value']:12.4g}  (exact {r['exact']})")
    if a_coef.empty:
        print("    WP-11a comparison                                          "
              "         NOT RUN — WP-11a outputs not on disk")
    else:
        print("\n    WP-11a comparison — the VOI's monthly/annual SD ratio "
              "against 11a's realised")
        print("    alpha recovery ratio on the same twin (`all_flows` bundle, "
              "per_obs, fitted window):")
        g = a_coef[(a_coef.split == "fitted") & (a_coef.freq == "monthly")]
        for r in g.itertuples():
            print(f"      {r.coef:11s} VOI {r.voi_ratio_all_flows:5.3f}   "
                  f"11a spectrum {r.wp11a_spectrum_ratio:5.3f}   "
                  f"realised {r.realised_ratio:5.3f}   "
                  f"error {r.rel_error_all_flows_pct:+7.1f}%")
        if not a_res.empty:
            print("\n    WP-11a resolution test — finer Delta at MATCHED N "
                  "(11a refits) against the")
            print("    VOI's fixed-total-precision prediction:")
            for r in a_res.itertuples():
                print(f"      [{r.split:6s}] {r.coef:11s} N={r.n_obs:4d}  "
                      f"Delta {r.fine_delta:6.3f}/{r.fine_span:.0f}yr "
                      f"{r.fine_relrmse_pct:6.2f}%  vs  "
                      f"{r.coarse_delta:6.3f}/{r.coarse_span:.0f}yr "
                      f"{r.coarse_relrmse_pct:6.2f}%   ratio "
                      f"{r.realised_ratio_fine_over_coarse:5.3f}   "
                      f"VOI predicts {r.voi_fixed_total_ratio:5.3f}")

    if not w3.empty:
        print("\n    VOI top series against WP-3's per-series noise ablation "
              "(monthly):")
        for r in w3[w3.freq == "monthly"].itertuples():
            mark = "agree" if r.agree else (
                f"DIFFER — WP-3's series is rank {r.wp3_series_voi_rank} at "
                f"{r.voi_wp3_series_reduction_pct:.1f}%")
            print(f"      {r.channel:12s} WP-3 {r.wp3_dominant_series:24s} "
                  f"VOI {r.voi_top_series:24s} {mark}")
            print(f"                   VOI top 3: {r.voi_top3}")

    # ---- Part 5 ----------------------------------------------------------
    lines, stmt = statements(rank)
    stmt.to_csv(os.path.join(args.out, "wp11c_statements.csv"), index=False)
    txt = ["# WP-11c — the deliverable statements",
           "",
           "Reduction in the marginal standard deviation of each coefficient "
           "if the named", "series were also observed over 1980–2007, per "
           "observation carrying WP-3's",
           "calibrated relative error.  Median over the fitting window and "
           "over " + str(len(seeds)) + " seeds.", ""] + lines
    with open(os.path.join(args.out, "wp11c_statements.md"), "w") as fh:
        fh.write("\n".join(txt) + "\n")
    print("\n  Part 5 — deliverable statements:")
    for ln in lines:
        print("   " + ln)

    figures(rank, fdec, w3, args.out)
    figure_wp11a(a_coef, a_res, args.out)
    print(f"\nwrote {args.out}/wp11c_voi_ranking.{{csv,png,pdf}} "
          f"+ 8 companion tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
