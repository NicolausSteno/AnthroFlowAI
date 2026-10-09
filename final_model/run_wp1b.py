#!/usr/bin/env python3
"""
run_wp1b.py — WP-1b: compensation test on the α·S parameterisation
==================================================================

Spec: "Per seed and channel, correlate per-year α error against per-year error
in the source stock. Systematic anti-correlation confirms compensating error —
a reportable identifiability result about α·S parameterisations."

No new fits.  Consumes the 35-seed α dumps written for WP-1a
(`analysis/wp1a/alpha_seed*.npz`) plus the stored `anchor_v4/pred_seed*.npz`
flow rollouts.

What this test isolates, precisely
----------------------------------
The spec motivates the test by "α is evaluated at *predicted* stocks while
`alpha_obs` was built from *observed* stocks".  That mechanism is **not**
operative in `anchor_v4`: the config sets `use_stock_input: false`, so
`_features` zeroes the state slots (`zinc_colloc_v5.py:503`) and α is a
function of (t, exogenous drivers) only — verified in WP-1a to 0.0 across all
35 seeds.  α therefore cannot read S_pred and adjust.

The test is still live, and cleaner for it.  Any anti-correlation that remains
comes from the *training objective*, not from the network's inputs: Stage B
penalises stock and flow divergence, and the flow is the product α·S, so a
seed whose S_pred runs high on a channel's parent stock is rewarded for
pulling that channel's α low.  That is a statement about the α·S
parameterisation itself rather than about the feature set, which is what the
spec wants to report.

Three readouts, in increasing strength
--------------------------------------
1.  **Correlation** (the spec's test): r(e_α, e_S) per seed and channel, in
    levels and in first differences, Pearson and Spearman.
2.  **Log-residual decomposition**: with a = log α_pred − log α_obs and
    b = log S̄_pred − log S̄_obs (S̄ = the trapezoid exposure `alpha_obs` is
    actually built on), the log flow-rate residual is a + b.  Reports the
    compensation fraction C = −2·Cov(a,b) / (Var(a) + Var(b)) — C = 1 is
    perfect cancellation, C = 0 independence, C < 0 amplification — and the
    variance-reduction ratio SD(a+b) / sqrt(Var(a)+Var(b)).
3.  **Realised benefit**: α relRMSE against the relRMSE of the flow that
    channel drives.  If the product is estimated far better than the factor,
    compensation is not merely correlational — it is doing measurable work.

Controls
--------
* First differences, to rule out shared trend.
* **Seed-mismatched pairing**: seed i's e_α against seed j's e_S (i ≠ j).
  Genuine within-fit compensation must be stronger in matched pairs than in
  mismatched ones; if the two are equal, the anti-correlation is common
  structure in the data, not compensation.
* **Stage A**, reported alongside Stage B.  Stage A's α is fitted pointwise
  with no trajectory term, so its α cannot have been adjusted to offset a
  stock error — while the `S_pred_A` it is compared against is still a
  free-run ODE rollout.  Stage A is therefore the no-compensation-by-
  construction arm, and whatever anti-correlation it shows is the mechanical
  baseline that Stage B must beat for the compensation reading to hold.

    python run_wp1b.py --check
    python run_wp1b.py
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DUMP_DIR = os.path.join(HERE, "analysis", "wp1a")
ANCHOR_DIR = os.path.join(HERE, "anchor_v4")
OUT_DIR = os.path.join(HERE, "analysis")

CHANNELS = ["alpha_cc", "alpha_refc", "alpha_win", "alpha_dr"]
# zinc_colloc_v5.ALPHA_PARENT_STOCK_IDX = (0, 1, 3, 3)
PARENT_IDX = {"alpha_cc": 0, "alpha_refc": 1, "alpha_win": 3, "alpha_dr": 3}
STOCK_NAMES = ["Concentrate", "Refined", "In-Use", "Scrap"]
# The flow each α channel drives, i.e. the numerator `_build_empirical_alphas`
# divides by the parent-stock exposure (zinc_colloc_v5.py:962-968).
CHANNEL_FLOW = {"alpha_cc": "concentrate_consumption",
                "alpha_refc": "refined_consumption",
                "alpha_win": "waelz_input",
                "alpha_dr": "direct_reuse_recycling"}

CH_COLOURS = {"alpha_cc": "#0072B2", "alpha_refc": "#009E73",
              "alpha_win": "#D55E00", "alpha_dr": "#CC79A7"}
GREY = "#555555"


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def rel_rmse_pct(pred, obs, eps=1e-12):
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    m = np.isfinite(pred) & np.isfinite(obs)
    if not m.any():
        return float("nan")
    den = float(np.mean(np.abs(obs[m])))
    if den < eps:
        return float("nan")
    return 100.0 * float(np.sqrt(np.mean((pred[m] - obs[m]) ** 2))) / den


def _clean_pair(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def pearson(x, y):
    x, y = _clean_pair(x, y)
    if x.size < 3 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    x, y = _clean_pair(x, y)
    if x.size < 3:
        return float("nan")
    r = stats.spearmanr(x, y).statistic
    return float(r) if np.isfinite(r) else float("nan")


def ols_slope(x, y):
    x, y = _clean_pair(x, y)
    if x.size < 3 or np.var(x) < 1e-15:
        return float("nan")
    return float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))


def hodges_lehmann(v, conf=0.95):
    """HL location estimate of a one-sample distribution + distribution-free CI.

    Point estimate is the median of the Walsh averages (x_i + x_j)/2, i ≤ j.
    The CI endpoints are order statistics of those Walsh averages, indexed by
    the Wilcoxon signed-rank critical value (normal approximation with
    continuity correction, which is accurate at n = 35).  Project convention
    is HL intervals rather than significance tests.
    """
    v = np.asarray([x for x in np.asarray(v, float) if np.isfinite(x)])
    n = v.size
    if n == 0:
        return dict(hl=np.nan, lo=np.nan, hi=np.nan, n=0)
    w = (v[:, None] + v[None, :]) / 2.0
    walsh = np.sort(w[np.triu_indices(n)])
    est = float(np.median(walsh))
    if n < 6:
        return dict(hl=est, lo=np.nan, hi=np.nan, n=n)
    N = walsh.size                                   # n(n+1)/2
    z = stats.norm.ppf(1.0 - (1.0 - conf) / 2.0)
    mu = N / 2.0
    sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    k = int(np.floor(mu - z * sd))                   # 0-based lower index
    k = max(k, 0)
    lo = float(walsh[k])
    hi = float(walsh[min(N - 1 - k, N - 1)])
    return dict(hl=est, lo=lo, hi=hi, n=n)


def med_iqr(v):
    v = np.asarray([x for x in np.asarray(v, float) if np.isfinite(x)])
    if v.size == 0:
        return dict(median=np.nan, q1=np.nan, q3=np.nan, iqr=np.nan, n=0)
    q1, q3 = np.percentile(v, [25, 75])
    return dict(median=float(np.median(v)), q1=float(q1), q3=float(q3),
                iqr=float(q3 - q1), n=int(v.size))


def compensation_fraction(a, b):
    """C = −2·Cov(a,b) / (Var(a) + Var(b)).

    a + b is the log flow-rate residual.  C = 1 ⇔ Var(a+b) = 0 (perfect
    cancellation); C = 0 ⇔ the two residuals are uncorrelated; C < 0 ⇔ they
    reinforce.  Also returns SD(a+b) relative to the no-covariance case.
    """
    a, b = _clean_pair(a, b)
    if a.size < 3:
        return float("nan"), float("nan")
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    cov = float(np.cov(a, b, ddof=1)[0, 1])
    denom = va + vb
    if denom < 1e-30:
        return float("nan"), float("nan")
    C = -2.0 * cov / denom
    ratio = float(np.sqrt(max(va + vb + 2.0 * cov, 0.0) / denom))
    return float(C), ratio


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def _safe_log(x):
    x = np.asarray(x, float)
    return np.where(x > 0, np.log(np.where(x > 0, x, 1.0)), np.nan)


def load_seeds(dump_dir=DUMP_DIR, anchor_dir=ANCHOR_DIR):
    """Per seed, assemble the residual series WP-1b needs.

    Returns a list of dicts.  All arrays are year-aligned to `years` (T = 40);
    exposure-based quantities carry NaN in row 0, matching `alpha_obs`.
    """
    paths = sorted(glob.glob(os.path.join(dump_dir, "alpha_seed*.npz")),
                   key=lambda p: int(os.path.basename(p)[10:-4]))
    if not paths:
        raise SystemExit(f"no WP-1a dumps in {dump_dir} — run zinc_alpha_lab.py first")
    out = []
    for p in paths:
        seed = int(os.path.basename(p)[10:-4])
        d = np.load(p, allow_pickle=True)
        years = np.asarray(d["years"], float).ravel()
        rec = dict(seed=seed, years=years,
                   mask=dict(train=np.asarray(d["mask_train"], bool),
                             val=np.asarray(d["mask_val"], bool),
                             test=np.asarray(d["mask_test"], bool),
                             all=np.ones(years.size, bool)),
                   alpha_obs=np.asarray(d["alpha_obs"], float),
                   S_obs=np.asarray(d["S_obs"], float))
        for stage in ("A", "B"):
            rec[f"alpha_{stage}"] = np.asarray(d[f"alpha_pred_{stage}_at_obs"], float)
            rec[f"S_{stage}"] = np.asarray(d[f"S_pred_{stage}"], float)
        # stored flow rollout (Stage B only — run_anchor.py persists "B")
        fp = os.path.join(anchor_dir, f"pred_seed{seed}.npz")
        if os.path.exists(fp):
            f = np.load(fp, allow_pickle=True)
            names = [str(x) for x in f["flow_names"]]
            rec["flow_names"] = names
            rec["years_flow"] = np.asarray(f["years_flow"], float).ravel()
            rec["F_pred_B"] = np.asarray(f["F_pred_B"], float)
            rec["flows_obs"] = np.asarray(f["flows_obs"], float)
        out.append(rec)
    return out


def exposure(S, k):
    """Trapezoid parent-stock exposure, the denominator `alpha_obs` is built on
    (`_build_empirical_alphas`, zinc_colloc_v5.py:866).  Row 0 is NaN."""
    S = np.asarray(S, float)
    out = np.full(S.shape[0], np.nan)
    out[1:] = 0.5 * (S[:-1, k] + S[1:, k])
    return out


# ---------------------------------------------------------------------------
# the test
# ---------------------------------------------------------------------------
def residuals(rec, ch, stage):
    """Per-year residual series for one seed and channel."""
    k = CHANNELS.index(ch)
    p = PARENT_IDX[ch]
    e_alpha = rec[f"alpha_{stage}"][:, k] - rec["alpha_obs"][:, k]
    e_stock = rec[f"S_{stage}"][:, p] - rec["S_obs"][:, p]
    exp_pred, exp_obs = exposure(rec[f"S_{stage}"], p), exposure(rec["S_obs"], p)
    e_exposure = exp_pred - exp_obs
    a = _safe_log(rec[f"alpha_{stage}"][:, k]) - _safe_log(rec["alpha_obs"][:, k])
    b = _safe_log(exp_pred) - _safe_log(exp_obs)
    return dict(e_alpha=e_alpha, e_stock=e_stock, e_exposure=e_exposure,
                log_a=a, log_b=b)


def per_seed_table(seeds):
    rows = []
    for rec in seeds:
        for stage in ("A", "B"):
            for ch in CHANNELS:
                r = residuals(rec, ch, stage)
                for split, m in rec["mask"].items():
                    ea, es = r["e_alpha"][m], r["e_stock"][m]
                    ex = r["e_exposure"][m]
                    a, b = r["log_a"][m], r["log_b"][m]
                    C, ratio = compensation_fraction(a, b)
                    # first differences, to rule out shared trend
                    dea, des = np.diff(r["e_alpha"])[m[1:]], np.diff(r["e_stock"])[m[1:]]
                    rows.append(dict(
                        seed=rec["seed"], channel=ch, stage=stage, split=split,
                        parent_stock=STOCK_NAMES[PARENT_IDX[ch]],
                        n_years=int(np.isfinite(ea * es).sum()),
                        r_pearson_levels=pearson(ea, es),
                        r_spearman_levels=spearman(ea, es),
                        r_pearson_exposure=pearson(ea, ex),
                        r_pearson_diffs=pearson(dea, des),
                        r_spearman_diffs=spearman(dea, des),
                        slope_alpha_on_stock=ols_slope(es, ea),
                        r_pearson_log=pearson(a, b),
                        compensation_fraction=C,
                        sd_ratio_sum_vs_indep=ratio,
                        sd_log_a=float(np.nanstd(a, ddof=1)),
                        sd_log_b=float(np.nanstd(b, ddof=1)),
                        sd_log_sum=float(np.nanstd(a + b, ddof=1)),
                        # signed direction: is the compensation a bias pair?
                        median_e_alpha=float(np.nanmedian(ea)),
                        median_e_stock=float(np.nanmedian(es)),
                        median_log_a=float(np.nanmedian(a)),
                        median_log_b=float(np.nanmedian(b))))
    return pd.DataFrame(rows)


def mismatch_control(seeds, stage="B", split="all"):
    """Seed-mismatched pairing: seed i's e_α against seed j's e_S, i ≠ j.

    If matched-pair anti-correlation is no stronger than mismatched, the effect
    is shared structure in the data rather than within-fit compensation.
    """
    rows = []
    for ch in CHANNELS:
        R = [residuals(rec, ch, stage) for rec in seeds]
        m = seeds[0]["mask"][split]
        matched = [pearson(r["e_alpha"][m], r["e_stock"][m]) for r in R]
        mismatched = [pearson(R[i]["e_alpha"][m], R[j]["e_stock"][m])
                      for i in range(len(R)) for j in range(len(R)) if i != j]
        mt, mm = med_iqr(matched), med_iqr(mismatched)
        hl_m, hl_x = hodges_lehmann(matched), hodges_lehmann(mismatched)
        rows.append(dict(channel=ch, stage=stage, split=split,
                         matched_median=mt["median"], matched_iqr=mt["iqr"],
                         matched_hl=hl_m["hl"], matched_hl_lo=hl_m["lo"],
                         matched_hl_hi=hl_m["hi"], n_matched=mt["n"],
                         mismatched_median=mm["median"], mismatched_iqr=mm["iqr"],
                         mismatched_hl=hl_x["hl"], n_mismatched=mm["n"],
                         matched_minus_mismatched=mt["median"] - mm["median"]))
    return pd.DataFrame(rows)


def factor_vs_product(seeds):
    """α relRMSE against the relRMSE of the flow that channel drives.

    α is scored on the test-window year mask; the flow is scored on the same
    years of the interval-aligned grid (`years_flow` row i closes
    `years_all[i+1]`), so the two cover the same calendar span.
    """
    rows = []
    for rec in seeds:
        if "F_pred_B" not in rec:
            continue
        m_test = rec["mask"]["test"]
        yrs_test = rec["years"][m_test]
        mf = np.isin(rec["years_flow"], yrs_test)
        for k, ch in enumerate(CHANNELS):
            fl = CHANNEL_FLOW[ch]
            if fl not in rec["flow_names"]:
                continue
            j = rec["flow_names"].index(fl)
            a_rel = rel_rmse_pct(rec["alpha_B"][m_test, k],
                                 rec["alpha_obs"][m_test, k])
            f_rel = rel_rmse_pct(rec["F_pred_B"][mf, j], rec["flows_obs"][mf, j])
            p = PARENT_IDX[ch]
            s_rel = rel_rmse_pct(rec["S_B"][m_test, p], rec["S_obs"][m_test, p])
            rows.append(dict(seed=rec["seed"], channel=ch, flow=fl,
                             parent_stock=STOCK_NAMES[p],
                             alpha_relRMSE=a_rel, stock_relRMSE=s_rel,
                             flow_relRMSE=f_rel,
                             ratio_flow_over_alpha=f_rel / a_rel
                             if a_rel and np.isfinite(a_rel) else np.nan))
    return pd.DataFrame(rows)


def summarise(per_seed, fvp, stage="B", split="all"):
    sub = per_seed[(per_seed.stage == stage) & (per_seed.split == split)]
    rows = []
    for ch in CHANNELS:
        s = sub[sub.channel == ch]
        rec = dict(channel=ch, stage=stage, split=split,
                   parent_stock=STOCK_NAMES[PARENT_IDX[ch]],
                   n_seeds=int(s.seed.nunique()))
        for col in ("r_pearson_levels", "r_spearman_levels", "r_pearson_diffs",
                    "r_pearson_log", "compensation_fraction",
                    "sd_ratio_sum_vs_indep", "slope_alpha_on_stock",
                    "median_log_a", "median_log_b"):
            st = med_iqr(s[col].values)
            hl = hodges_lehmann(s[col].values)
            rec[f"{col}_median"] = st["median"]
            rec[f"{col}_iqr"] = st["iqr"]
            rec[f"{col}_hl"] = hl["hl"]
            rec[f"{col}_hl_lo"] = hl["lo"]
            rec[f"{col}_hl_hi"] = hl["hi"]
        f = fvp[fvp.channel == ch]
        if len(f):
            for col in ("alpha_relRMSE", "stock_relRMSE", "flow_relRMSE",
                        "ratio_flow_over_alpha"):
                rec[f"{col}_median"] = float(np.nanmedian(f[col].values))
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _strip(ax, x, vals, colour, rng, width=0.16):
    vals = np.asarray([v for v in np.asarray(vals, float) if np.isfinite(v)])
    if vals.size == 0:
        return
    ax.scatter(x + rng.uniform(-width, width, vals.size), vals, s=18,
               color=colour, alpha=0.55, edgecolors="none", zorder=2)
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    ax.plot([x - 0.30, x + 0.30], [med, med], color=colour, lw=2.4, zorder=3)
    ax.plot([x, x], [q1, q3], color=colour, lw=1.2, zorder=3)


def fig_compensation(per_seed, mism, summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    sub = per_seed[(per_seed.stage == "B") & (per_seed.split == "all")]
    subA = per_seed[(per_seed.stage == "A") & (per_seed.split == "all")]

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.4))

    # (a) the spec's test: per-seed r(e_alpha, e_stock), Stage B vs Stage A
    ax = axes[0]
    for i, ch in enumerate(CHANNELS):
        _strip(ax, i - 0.18, sub[sub.channel == ch]["r_pearson_levels"].values,
               CH_COLOURS[ch], rng, width=0.10)
        _strip(ax, i + 0.18, subA[subA.channel == ch]["r_pearson_levels"].values,
               GREY, rng, width=0.10)
        row = mism[mism.channel == ch]
        if len(row):
            ax.plot([i - 0.34, i - 0.02], [float(row.iloc[0]["mismatched_median"])] * 2,
                    color="k", ls=":", lw=1.6, zorder=4)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.plot([], [], color=GREY, lw=2.4, label="Stage A α (no trajectory loss)")
    ax.plot([], [], color="k", ls=":", lw=1.6, label="seed-mismatched control")
    ax.set_ylabel("r(e$_α$, e$_S$) per seed, full window")
    ax.set_title("(a) compensation correlation", loc="left", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")

    # (b) log-residual decomposition: compensation fraction
    ax = axes[1]
    for i, ch in enumerate(CHANNELS):
        _strip(ax, i, sub[sub.channel == ch]["compensation_fraction"].values,
               CH_COLOURS[ch], rng)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhline(1.0, color=GREY, ls="--", lw=1.2, label="perfect cancellation")
    ax.set_ylabel("C = −2·Cov(a,b) / (Var a + Var b)")
    ax.set_title("(b) log-residual compensation", loc="left", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")

    # (c) factor vs product: is the product better estimated than the factor?
    ax = axes[2]
    s = summary.set_index("channel")
    for i, ch in enumerate(CHANNELS):
        r = s.loc[ch]
        ax.plot([i - 0.22, i + 0.22],
                [r["alpha_relRMSE_median"], r["flow_relRMSE_median"]],
                color=CH_COLOURS[ch], lw=1.4, zorder=2)
        ax.scatter([i - 0.22], [r["alpha_relRMSE_median"]], s=52,
                   color=CH_COLOURS[ch], marker="o", zorder=3)
        ax.scatter([i + 0.22], [r["flow_relRMSE_median"]], s=64,
                   color=CH_COLOURS[ch], marker="v", zorder=3)
    ax.scatter([], [], s=52, color=GREY, marker="o", label="α relRMSE (the factor)")
    ax.scatter([], [], s=64, color=GREY, marker="v", label="flow relRMSE (the product)")
    ax.set_yscale("log")
    ax.set_ylabel("test-window relRMSE (%), median over seeds")
    ax.set_title("(c) factor versus product", loc="left", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")

    for ax in axes:
        ax.set_xticks(range(len(CHANNELS)))
        ax.set_xticklabels(CHANNELS, rotation=15)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", lw=0.4, alpha=0.3)
        ax.set_xlim(-0.6, len(CHANNELS) - 0.4)
    n = int(sub.seed.nunique())
    fig.suptitle(f"WP-1b  compensation in the α·S parameterisation — "
                 f"anchor_v4, {n} seeds", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=300)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


def fig_scatter(seeds, path):
    """Pooled per-year residual scatter, one panel per channel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.7))
    for ax, ch in zip(axes, CHANNELS):
        xs, ys = [], []
        for rec in seeds:
            r = residuals(rec, ch, "B")
            m = rec["mask"]["all"]
            xs.append(r["e_stock"][m]); ys.append(r["e_alpha"][m])
        x = np.concatenate(xs); y = np.concatenate(ys)
        ax.scatter(x, y, s=6, color=CH_COLOURS[ch], alpha=0.25, edgecolors="none")
        xc, yc = _clean_pair(x, y)
        if xc.size > 3:
            sl = ols_slope(xc, yc)
            ic = float(np.mean(yc) - sl * np.mean(xc))
            xx = np.linspace(np.percentile(xc, 1), np.percentile(xc, 99), 20)
            ax.plot(xx, sl * xx + ic, color="k", lw=1.4)
            ax.text(0.97, 0.94, f"r = {pearson(xc, yc):+.2f}", transform=ax.transAxes,
                    fontsize=9, va="top", ha="right")
        ax.axhline(0.0, color=GREY, lw=0.7)
        ax.axvline(0.0, color=GREY, lw=0.7)
        ax.set_title(f"{ch}  (÷ {STOCK_NAMES[PARENT_IDX[ch]]})", loc="left",
                     fontsize=9.5, color=CH_COLOURS[ch])
        ax.set_xlabel(f"e$_S$ = S_pred − S_obs (kt)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(lw=0.4, alpha=0.3)
    axes[0].set_ylabel("e$_α$ = α_pred − α_obs (1/yr)")
    fig.suptitle("WP-1b  per-year α error against parent-stock error, "
                 "all seeds pooled, Stage B, 1980–2019",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=300)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="WP-1b compensation test")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dumps", default=DUMP_DIR)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args(argv)

    import zinc_alpha_lab as lab
    lab.check(verbose=True)
    if args.check:
        return 0

    seeds = load_seeds(args.dumps)
    print(f"\nloaded {len(seeds)} seeds; "
          f"{sum('F_pred_B' in r for r in seeds)} with stored flow rollouts")

    # State-independence of α is what makes this test about the objective
    # rather than the feature set (WP-1a flag 1) — re-assert it here rather
    # than trusting the config, since the whole interpretation turns on it.
    d0 = np.load(os.path.join(args.dumps, f"alpha_seed{seeds[0]['seed']}.npz"),
                 allow_pickle=True)
    gap = max(float(np.nanmax(np.abs(d0[f"alpha_pred_{st}_at_obs"]
                                     - d0[f"alpha_pred_{st}_at_pred"])))
              for st in ("A", "B"))
    print(f"α state-independence check: max |α(at S_obs) − α(at S_pred)| = {gap:.3e} "
          f"(use_stock_input=false) → this test probes the training objective, "
          f"not the input layout")

    per_seed = per_seed_table(seeds)
    fvp = factor_vs_product(seeds)
    mism = mismatch_control(seeds)
    summary = summarise(per_seed, fvp)

    os.makedirs(args.out, exist_ok=True)
    per_seed.to_csv(os.path.join(args.out, "wp1b_compensation_per_seed.csv"), index=False)
    summary.to_csv(os.path.join(args.out, "wp1b_compensation.csv"), index=False)
    mism.to_csv(os.path.join(args.out, "wp1b_mismatch_control.csv"), index=False)
    fvp.to_csv(os.path.join(args.out, "wp1b_factor_vs_product.csv"), index=False)

    fig_compensation(per_seed, mism, summary,
                     os.path.join(args.out, "wp1b_compensation.png"))
    fig_scatter(seeds, os.path.join(args.out, "wp1b_residual_scatter.png"))

    pd.set_option("display.width", 200, "display.max_columns", 60)
    print("\n--- (1) the spec's test: r(e_α, e_S), Stage B, full window ---")
    cols = ["channel", "parent_stock", "n_seeds",
            "r_pearson_levels_median", "r_pearson_levels_iqr",
            "r_pearson_levels_hl", "r_pearson_levels_hl_lo", "r_pearson_levels_hl_hi",
            "r_spearman_levels_median", "r_pearson_diffs_median"]
    print(summary[cols].round(3).to_string(index=False))

    print("\n--- (2) log-residual decomposition (a = Δlog α, b = Δlog S̄) ---")
    cols = ["channel", "r_pearson_log_median", "compensation_fraction_median",
            "compensation_fraction_hl", "compensation_fraction_hl_lo",
            "compensation_fraction_hl_hi", "sd_ratio_sum_vs_indep_median",
            "median_log_a_median", "median_log_b_median"]
    print(summary[cols].round(3).to_string(index=False))

    print("\n--- (3) factor versus product, test window, median over seeds ---")
    cols = ["channel", "parent_stock", "alpha_relRMSE_median",
            "stock_relRMSE_median", "flow_relRMSE_median",
            "ratio_flow_over_alpha_median"]
    print(summary[cols].round(3).to_string(index=False))

    print("\n--- in-sample vs out-of-sample (Stage B): where the anti-correlation lives ---")
    sb = per_seed[per_seed.stage == "B"]
    piv = (sb[sb.split.isin(["train", "val", "test"])]
           .pivot_table(index="channel", columns="split",
                        values=["r_pearson_levels", "compensation_fraction"],
                        aggfunc="median")
           .reindex(CHANNELS))
    print(piv.round(3).to_string())

    print("\n--- controls: matched vs seed-mismatched pairing (Stage B, full window) ---")
    cols = ["channel", "matched_median", "matched_hl", "matched_hl_lo",
            "matched_hl_hi", "mismatched_median", "matched_minus_mismatched",
            "n_matched", "n_mismatched"]
    print(mism[cols].round(3).to_string(index=False))

    print("\n--- Stage A control (α fitted with no trajectory loss), "
          "r(e_α, e_S) median ---")
    sa = summarise(per_seed, fvp, stage="A")
    print(sa[["channel", "r_pearson_levels_median", "r_pearson_levels_iqr",
              "compensation_fraction_median"]].round(3).to_string(index=False))

    print(f"\nwrote {args.out}/wp1b_compensation.{{csv,png,pdf}}, "
          f"wp1b_compensation_per_seed.csv, wp1b_mismatch_control.csv, "
          f"wp1b_factor_vs_product.csv, wp1b_residual_scatter.{{png,pdf}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
