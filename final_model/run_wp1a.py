#!/usr/bin/env python3
"""
run_wp1a.py — WP-1a: per-channel α decomposition
================================================

Breaks the anchor_v4 family-mean α relRMSE (64.25% median, spec §1) into its
four channels — `alpha_cc`, `alpha_refc`, `alpha_win`, `alpha_dr` — median and
IQR across the 35-seed ensemble.

Consumes the per-seed dumps written by `zinc_alpha_lab.py`, which refits the
anchor_v4 config seed for seed (α is not persisted by `run_anchor.py`;
SCHEMA §1, §3).  Reproduction is bit-exact against the stored `pred_seed*.npz`
rollouts, verified per seed here.

Reported per channel, per the spec:
  * level-space relRMSE% — the decomposition of the headline number;
  * log-space RMSE in nats and in units of `stats["alpha_log_std"]`, the
    Stage A/B loss denominator, since a good log fit can read badly in levels;
  * year-on-year volatility of `alpha_obs`, plus a persistence benchmark on
    the target (predict last year's α), to say whether close tracking would
    amount to chasing reconstruction noise.

Rollouts.  `FitResult.diagnose` evaluates α at OBSERVED stocks for every
`kind`, so the freerun and testrun α numbers are identical by construction —
verified across all 35 seeds rather than assumed.  The rollout-dependent
quantity is α along the trajectory (α at predicted stocks), reported
separately as `at_pred`.

    python run_wp1a.py --check
    python run_wp1a.py
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
DUMP_DIR = os.path.join(HERE, "analysis", "wp1a")
ANCHOR_DIR = os.path.join(HERE, "anchor_v4")
GAM_DIR = os.path.join(HERE, "anchor_gam")
OUT_DIR = os.path.join(HERE, "analysis")

# Okabe–Ito, colourblind-safe.  Ordered to match ALPHA_NAMES.
CH_COLOURS = {"alpha_cc": "#0072B2", "alpha_refc": "#009E73",
              "alpha_win": "#D55E00", "alpha_dr": "#CC79A7"}
GREY = "#555555"


# ---------------------------------------------------------------------------
# metrics — mirror zinc_colloc_v5._rel_rmse_pct / _logmae exactly
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


def log_rmse(pred, obs, eps=1e-12):
    """RMSE of log α_pred − log α_obs, in nats.  Same residual the loss uses."""
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    m = np.isfinite(pred) & np.isfinite(obs) & (pred > 0) & (obs > 0)
    if not m.any():
        return float("nan")
    r = np.log(np.maximum(pred[m], eps)) - np.log(np.maximum(obs[m], eps))
    return float(np.sqrt(np.mean(r ** 2)))


def mape_pct(pred, obs, eps=1e-9):
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    m = np.isfinite(pred) & np.isfinite(obs) & (np.abs(obs) > eps)
    if not m.any():
        return float("nan")
    return 100.0 * float(np.mean(np.abs((pred[m] - obs[m]) / obs[m])))


def med_iqr(v):
    v = np.asarray([x for x in np.asarray(v, float) if np.isfinite(x)])
    if v.size == 0:
        return dict(median=np.nan, q1=np.nan, q3=np.nan, iqr=np.nan, n=0)
    q1, q3 = np.percentile(v, [25, 75])
    return dict(median=float(np.median(v)), q1=float(q1), q3=float(q3),
                iqr=float(q3 - q1), n=int(v.size))


def hl_shift(a, b):
    """Hodges–Lehmann estimate of the paired shift a−b (project convention:
    HL point estimate rather than a significance test)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    if d.size == 0:
        return float("nan")
    walsh = (d[:, None] + d[None, :]) / 2.0
    return float(np.median(walsh[np.triu_indices_from(walsh)]))


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_dumps(dump_dir=DUMP_DIR):
    paths = sorted(glob.glob(os.path.join(dump_dir, "alpha_seed*.npz")),
                   key=lambda p: int(os.path.basename(p)[10:-4]))
    if not paths:
        raise SystemExit(f"no dumps in {dump_dir} — run zinc_alpha_lab.py first")
    out = []
    for p in paths:
        seed = int(os.path.basename(p)[10:-4])
        out.append((seed, np.load(p, allow_pickle=True)))
    return out


def verify_reproduction(dumps):
    """Check each refit reproduces the stored anchor_v4 rollout and its
    family-mean α number.  A silent divergence here would invalidate the
    decomposition, so it is checked rather than assumed."""
    per_seed = pd.read_csv(os.path.join(ANCHOR_DIR, "per_seed.csv"))
    rows = []
    for seed, d in dumps:
        stored_path = os.path.join(ANCHOR_DIR, f"pred_seed{seed}.npz")
        s_diff = np.nan
        if os.path.exists(stored_path):
            st = np.load(stored_path, allow_pickle=True)
            s_diff = float(np.nanmax(np.abs(d["S_pred_B"] - st["S_pred_B"])))
        mt = d["mask_test"]
        fam = float(np.mean([rel_rmse_pct(d["alpha_pred_B_at_obs"][mt, k],
                                          d["alpha_obs"][mt, k]) for k in range(4)]))
        sub = per_seed[(per_seed.seed == seed) & (per_seed.rollout == "freerun")]
        stored_fam = float(sub.iloc[0]["alpha_relRMSE"]) if len(sub) else np.nan
        fr = per_seed[(per_seed.seed == seed) & (per_seed.rollout == "freerun")]
        tr = per_seed[(per_seed.seed == seed) & (per_seed.rollout == "testrun")]
        rollout_gap = (float(fr.iloc[0]["alpha_relRMSE"] - tr.iloc[0]["alpha_relRMSE"])
                       if len(fr) and len(tr) else np.nan)
        rows.append(dict(seed=seed, S_pred_B_maxabsdiff=s_diff,
                         alpha_family_refit=fam, alpha_family_stored=stored_fam,
                         alpha_family_absdiff=abs(fam - stored_fam),
                         alpha_freerun_minus_testrun=rollout_gap))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# the decomposition
# ---------------------------------------------------------------------------
def per_seed_table(dumps):
    """Long-form: one row per (seed, channel, split, evaluation point)."""
    rows = []
    for seed, d in dumps:
        names = [str(x) for x in d["alpha_names"]]
        a_obs = d["alpha_obs"]
        a_std = d["alpha_log_std"]
        masks = dict(train=d["mask_train"], val=d["mask_val"], test=d["mask_test"],
                     all=np.ones(len(d["years"]), bool))
        for split, m in masks.items():
            for k, nm in enumerate(names):
                for stage in ("A", "B"):
                    for where in ("at_obs", "at_pred"):
                        key = f"alpha_pred_{stage}_{where}"
                        if key not in d.files:
                            continue
                        pr, ob = d[key][m, k], a_obs[m, k]
                        lr = log_rmse(pr, ob)
                        rows.append(dict(
                            seed=seed, channel=nm, split=split, stage=stage,
                            eval_at=where,
                            relRMSE=rel_rmse_pct(pr, ob),
                            MAPE=mape_pct(pr, ob),
                            logRMSE_nats=lr,
                            logRMSE_over_alpha_log_std=lr / float(a_std[k]),
                            alpha_log_std=float(a_std[k])))
    return pd.DataFrame(rows)


def target_volatility(dumps):
    """Year-on-year volatility of `alpha_obs`, and naive benchmarks on it.

    Two persistence variants, because they answer different questions:
      * `persistence_boundary` carries the last trainval-year α forward across
        the whole test window — the same construction `run_anchor.py`'s
        `persistence_benchmark()` uses for stocks, so it is the project's
        like-for-like naive comparator;
      * `persistence_rolling` predicts α_obs[t−1] for each test year t.  It
        uses the target inside the test window, which the model never sees,
        so it is not a fair forecast comparator — it is a read on how much
        year-on-year movement is there to track at all.
    Also `logRMSE_const_geomean`: predicting the train-window geometric mean,
    the constant-α null in log space.

    The target is identical across seeds (it is data, not a fit), so this is
    computed once from the first dump and cross-checked against the rest.
    """
    seed0, d0 = dumps[0]
    names = [str(x) for x in d0["alpha_names"]]
    a_obs = np.asarray(d0["alpha_obs"], float)
    for _, d in dumps[1:]:
        if not np.allclose(np.nan_to_num(d["alpha_obs"], nan=0.0),
                           np.nan_to_num(a_obs, nan=0.0)):
            raise AssertionError("alpha_obs differs between seeds — not a fixed target")
    years = np.asarray(d0["years"], float).ravel()
    mt = np.asarray(d0["mask_test"], bool)
    m_train = np.asarray(d0["mask_train"], bool)
    rows = []
    for k, nm in enumerate(names):
        y = a_obs[:, k]
        dlog = np.diff(np.log(np.where(y > 0, y, np.nan)))
        dlog_test = dlog[mt[1:]]
        idx = np.where(mt)[0]
        pers_roll = rel_rmse_pct(y[idx - 1], y[idx])
        pers_bnd = rel_rmse_pct(np.full(idx.size, y[idx[0] - 1]), y[idx])
        gm = float(np.exp(np.nanmean(np.log(np.where(y[m_train] > 0,
                                                     y[m_train], np.nan)))))
        rows.append(dict(
            channel=nm,
            alpha_log_std=float(d0["alpha_log_std"][k]),
            alpha_obs_mean_test=float(np.nanmean(y[mt])),
            alpha_obs_geomean_train=gm,
            yoy_dlog_rms_all=float(np.sqrt(np.nanmean(dlog ** 2))),
            yoy_dlog_rms_test=float(np.sqrt(np.nanmean(dlog_test ** 2))),
            yoy_dlog_median_abs_test=float(np.nanmedian(np.abs(dlog_test))),
            persistence_boundary_relRMSE_test=pers_bnd,
            persistence_rolling_relRMSE_test=pers_roll,
            const_geomean_relRMSE_test=rel_rmse_pct(np.full(idx.size, gm), y[idx]),
            logRMSE_const_geomean_test=log_rmse(np.full(idx.size, gm), y[idx])))
    return pd.DataFrame(rows), years, a_obs


def per_year_error(dumps, drop_tail=2):
    """Where inside the test window the α error actually sits.

    The empirical target's parent stock for the two scrap channels
    (`S_scrap`) falls off a cliff in the last two reported years, so the
    ratio target `F_int / S_parent` spikes there.  This tabulates each test
    year's share of the channel's test-window MSE, and recomputes relRMSE
    with the last `drop_tail` years dropped.

    The trimmed number is a ROBUSTNESS READOUT ONLY.  The published metric
    stays the full pre-registered window (spec §2 constraint 3); nothing is
    selected on it.
    """
    _, d0 = dumps[0]
    years = np.asarray(d0["years"], float).ravel()
    mt = np.asarray(d0["mask_test"], bool)
    names = [str(x) for x in d0["alpha_names"]]
    S_obs = np.asarray(d0["S_obs"], float)
    parent = {"alpha_cc": 0, "alpha_refc": 1, "alpha_win": 3, "alpha_dr": 3}

    P = np.stack([d["alpha_pred_B_at_obs"] for _, d in dumps])     # (S, T, 4)
    a_obs = np.asarray(d0["alpha_obs"], float)
    idx = np.where(mt)[0]
    keep = idx[:-drop_tail] if drop_tail else idx

    rows, trim = [], []
    for k, nm in enumerate(names):
        sq = np.nanmedian((P[:, idx, k] - a_obs[idx, k]) ** 2, axis=0)   # (n_test,)
        tot = float(np.nansum(sq))
        for j, t in enumerate(idx):
            rows.append(dict(channel=nm, year=float(years[t]),
                             median_abs_err=float(np.nanmedian(
                                 np.abs(P[:, t, k] - a_obs[t, k]))),
                             alpha_obs=float(a_obs[t, k]),
                             parent_stock_obs=float(S_obs[t, parent[nm]]),
                             mse_share=float(sq[j] / tot) if tot > 0 else np.nan))
        full = [rel_rmse_pct(P[s, idx, k], a_obs[idx, k]) for s in range(P.shape[0])]
        trimmed = [rel_rmse_pct(P[s, keep, k], a_obs[keep, k]) for s in range(P.shape[0])]
        trim.append(dict(
            channel=nm,
            relRMSE_median_full=float(np.median(full)),
            relRMSE_median_trimmed=float(np.median(trimmed)),
            trimmed_drop_years=f"{years[idx[-drop_tail]]:.0f}–{years[idx[-1]]:.0f}"
                               if drop_tail else "",
            mse_share_dropped_years=float(np.nansum(
                np.nanmedian((P[:, idx[-drop_tail:], k]
                              - a_obs[idx[-drop_tail:], k]) ** 2, axis=0))
                / max(float(np.nansum(np.nanmedian(
                    (P[:, idx, k] - a_obs[idx, k]) ** 2, axis=0))), 1e-30))
                if drop_tail else np.nan))
    return pd.DataFrame(rows), pd.DataFrame(trim)


def covariation_flag(dumps):
    """Indicative-only diagnostic for the secondary hypothesis.

    `alpha_obs = F_int / (trapezoid exposure in S_parent)` divides an annual
    integral by an interpolated stock, so it is a biased estimator of the
    time-averaged α by roughly the WITHIN-year F–S covariance.  That term is
    not observable at annual resolution.  What is observable is the sign and
    strength of F–S covariation BETWEEN years, reported here as a pointer
    only; the bias itself is quantified on the synthetic twin in WP-3, where
    the true time-averaged α is known.
    """
    _, d0 = dumps[0]
    names = [str(x) for x in d0["alpha_names"]]
    a_obs = np.asarray(d0["alpha_obs"], float)
    S_obs = np.asarray(d0["S_obs"], float)
    # ALPHA_PARENT_STOCK_IDX in zinc_colloc_v5: cc<-Concentrate, refc<-Refined,
    # win<-Scrap, dr<-Scrap  (spec §1 "Learned coefficients" table).
    parent = {"alpha_cc": 0, "alpha_refc": 1, "alpha_win": 3, "alpha_dr": 3}
    rows = []
    for k, nm in enumerate(names):
        p = parent[nm]
        Sp = 0.5 * (S_obs[:-1, p] + S_obs[1:, p])            # exposure, (T-1,)
        F = a_obs[1:, k] * Sp                                # implied F_int
        m = np.isfinite(F) & np.isfinite(Sp) & (F > 0) & (Sp > 0)
        dF = np.diff(np.log(F[m])); dS = np.diff(np.log(Sp[m]))
        rows.append(dict(channel=nm, parent_stock=["Concentrate", "Refined",
                                                   "In-Use", "Scrap"][p],
                         corr_dlogF_dlogS=float(np.corrcoef(dF, dS)[0, 1])))
    return pd.DataFrame(rows)


def summarise(per_seed, vol, trim=None, split="test", stage="B", eval_at="at_obs"):
    sub = per_seed[(per_seed.split == split) & (per_seed.stage == stage)
                   & (per_seed.eval_at == eval_at)]
    rows = []
    for nm in ["alpha_cc", "alpha_refc", "alpha_win", "alpha_dr"]:
        s = sub[sub.channel == nm]
        rec = dict(channel=nm, split=split, stage=stage, eval_at=eval_at,
                   n_seeds=int(s.seed.nunique()))
        for metric in ("relRMSE", "MAPE", "logRMSE_nats",
                       "logRMSE_over_alpha_log_std"):
            st = med_iqr(s[metric].values)
            rec[f"{metric}_median"] = st["median"]
            rec[f"{metric}_q1"] = st["q1"]
            rec[f"{metric}_q3"] = st["q3"]
            rec[f"{metric}_iqr"] = st["iqr"]
        v = vol[vol.channel == nm].iloc[0]
        for c in ("alpha_log_std", "yoy_dlog_rms_test",
                  "persistence_boundary_relRMSE_test",
                  "persistence_rolling_relRMSE_test",
                  "const_geomean_relRMSE_test", "logRMSE_const_geomean_test"):
            rec[c] = float(v[c])
        rec["relRMSE_median_over_persistence_boundary"] = (
            rec["relRMSE_median"] / float(v["persistence_boundary_relRMSE_test"]))
        rec["logRMSE_median_over_const_geomean"] = (
            rec["logRMSE_nats_median"] / float(v["logRMSE_const_geomean_test"]))
        if trim is not None:
            t = trim[trim.channel == nm].iloc[0]
            rec["relRMSE_median_trimmed"] = float(t["relRMSE_median_trimmed"])
            rec["mse_share_dropped_years"] = float(t["mse_share_dropped_years"])
            rec["trimmed_drop_years"] = str(t["trimmed_drop_years"])
        rows.append(rec)
    # family mean, computed per seed then summarised (not a mean of medians)
    fam = (sub.groupby("seed")["relRMSE"].mean().values)
    st = med_iqr(fam)
    rows.append(dict(channel="<alpha-family-mean>", split=split, stage=stage,
                     eval_at=eval_at, n_seeds=st["n"],
                     relRMSE_median=st["median"], relRMSE_q1=st["q1"],
                     relRMSE_q3=st["q3"], relRMSE_iqr=st["iqr"]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _strip(ax, xs, vals, colour, rng, width=0.16):
    jit = rng.uniform(-width, width, size=len(vals))
    ax.scatter(xs + jit, vals, s=18, color=colour, alpha=0.55,
               edgecolors="none", zorder=2)
    med = np.nanmedian(vals)
    q1, q3 = np.nanpercentile(vals, [25, 75])
    ax.plot([xs - 0.30, xs + 0.30], [med, med], color=colour, lw=2.4, zorder=3)
    ax.plot([xs, xs], [q1, q3], color=colour, lw=1.2, zorder=3)
    return med


def fig_channels(per_seed, summary, vol, gam_alpha, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chans = ["alpha_cc", "alpha_refc", "alpha_win", "alpha_dr"]
    sub = per_seed[(per_seed.split == "test") & (per_seed.stage == "B")
                   & (per_seed.eval_at == "at_obs")]
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.4))

    # (a) level-space relRMSE
    ax = axes[0]
    for i, nm in enumerate(chans):
        _strip(ax, i, sub[sub.channel == nm]["relRMSE"].values, CH_COLOURS[nm], rng)
    fam_med = float(summary[summary.channel == "<alpha-family-mean>"]
                    ["relRMSE_median"].iloc[0])
    ax.axhline(fam_med, color=GREY, ls="--", lw=1.2,
               label=f"UDE family mean ({fam_med:.1f}%)")
    if np.isfinite(gam_alpha):
        ax.axhline(gam_alpha, color="k", ls=":", lw=1.2,
                   label=f"GAM family mean ({gam_alpha:.1f}%)")
    ax.set_ylabel("α relRMSE, test window (%)")
    ax.set_title("(a) level space", loc="left", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")

    # (b) log-space RMSE, with the loss denominator and target volatility
    ax = axes[1]
    for i, nm in enumerate(chans):
        _strip(ax, i, sub[sub.channel == nm]["logRMSE_nats"].values,
               CH_COLOURS[nm], rng)
        v = vol[vol.channel == nm].iloc[0]
        ax.plot([i - 0.30, i + 0.30], [v["alpha_log_std"]] * 2, color=GREY,
                lw=1.4, ls="--", zorder=4)
        ax.plot([i - 0.30, i + 0.30], [v["yoy_dlog_rms_test"]] * 2, color="k",
                lw=1.4, ls=":", zorder=4)
    ax.plot([], [], color=GREY, ls="--", lw=1.4, label="alpha_log_std (loss scale)")
    ax.plot([], [], color="k", ls=":", lw=1.4, label="target YoY volatility (RMS Δlog α)")
    ax.set_ylabel("RMSE of log α_pred − log α_obs (nats)")
    ax.set_title("(b) log space", loc="left", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")

    # (c) model vs a persistence forecast of the target itself
    ax = axes[2]
    for i, nm in enumerate(chans):
        _strip(ax, i, sub[sub.channel == nm]["relRMSE"].values, CH_COLOURS[nm], rng)
        v = vol[vol.channel == nm].iloc[0]
        ax.plot([i - 0.30, i + 0.30],
                [v["persistence_boundary_relRMSE_test"]] * 2,
                color="k", lw=1.6, ls="-", zorder=4)
        ax.plot([i - 0.30, i + 0.30],
                [v["persistence_rolling_relRMSE_test"]] * 2,
                color=GREY, lw=1.4, ls="--", zorder=4)
    ax.plot([], [], color="k", lw=1.6, label="persistence, boundary α carried forward")
    ax.plot([], [], color=GREY, lw=1.4, ls="--",
            label="persistence, rolling α_obs[t−1] (uses the target)")
    ax.set_ylabel("α relRMSE, test window (%)")
    ax.set_title("(c) versus naive benchmarks on α_obs", loc="left", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")

    for ax in axes:
        ax.set_xticks(range(len(chans)))
        ax.set_xticklabels(chans, rotation=15)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", lw=0.4, alpha=0.3)
        ax.set_xlim(-0.6, len(chans) - 0.4)
    n_seeds = int(sub.seed.nunique())
    fig.suptitle(f"WP-1a  per-channel α decomposition — anchor_v4, "
                 f"{n_seeds} seeds, test window 2007–2019",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=300)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


def fig_error_by_year(peryear, path):
    """Where the test-window error sits in time, and the target's denominator."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chans = ["alpha_cc", "alpha_refc", "alpha_win", "alpha_dr"]
    years = np.unique(peryear.year.values)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9),
                             gridspec_kw=dict(width_ratios=[1.5, 1.0]))

    ax = axes[0]
    w = 0.2
    for i, nm in enumerate(chans):
        s = peryear[peryear.channel == nm].sort_values("year")
        ax.bar(s.year.values + (i - 1.5) * w, 100 * s.mse_share.values, width=w,
               color=CH_COLOURS[nm], label=nm)
    ax.set_ylabel("share of test-window α MSE (%)")
    ax.set_xlabel("year")
    ax.set_title("(a) when the α error happens", loc="left", fontsize=10)
    ax.legend(fontsize=8, frameon=False, ncol=2)

    ax = axes[1]
    s = peryear[peryear.channel == "alpha_win"].sort_values("year")
    ax.plot(s.year.values, s.parent_stock_obs.values, color="k", lw=1.6,
            marker="o", ms=3)
    ax.set_ylabel("observed Scrap stock (kt)")
    ax.set_xlabel("year")
    ax.set_title("(b) the shared denominator of alpha_win / alpha_dr",
                 loc="left", fontsize=10)
    ax.set_ylim(0, None)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", lw=0.4, alpha=0.3)
    fig.suptitle("WP-1a  α error concentration in time, and the reported "
                 "S_scrap series that divides into both scrap channels",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=300)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


def fig_trajectories(dumps, years, a_obs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chans = ["alpha_cc", "alpha_refc", "alpha_win", "alpha_dr"]
    P = np.stack([d["alpha_pred_B_at_obs"] for _, d in dumps])   # (S, T, 4)
    mt = np.asarray(dumps[0][1]["mask_test"], bool)
    t0 = float(years[mt][0])

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.4), sharex=True)
    for i, (ax, nm) in enumerate(zip(axes.ravel(), chans)):
        lo, med, hi = np.percentile(P[:, :, i], [25, 50, 75], axis=0)
        ax.fill_between(years, lo, hi, color=CH_COLOURS[nm], alpha=0.25, lw=0)
        ax.plot(years, med, color=CH_COLOURS[nm], lw=1.8, label="α_pred (median, IQR)")
        ax.plot(years, a_obs[:, i], color="k", lw=1.2, ls="--", marker="o",
                ms=2.6, label="α_obs")
        ax.axvline(t0, color=GREY, lw=0.9, ls=":")
        ax.set_title(nm, loc="left", fontsize=10, color=CH_COLOURS[nm])
        ax.set_yscale("log")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(lw=0.4, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("year")
    fig.suptitle("WP-1a  learned α(t) at observed stocks vs empirical target "
                 "(dotted line = start of held-out window)",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=300)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
def check():
    import zinc_alpha_lab as lab
    return lab.check(verbose=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="WP-1a per-channel α decomposition")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dumps", default=DUMP_DIR)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args(argv)

    check()
    if args.check:
        return 0

    dumps = load_dumps(args.dumps)
    print(f"\nloaded {len(dumps)} seed dumps from {args.dumps}")

    repro = verify_reproduction(dumps)
    bad = repro[repro.S_pred_B_maxabsdiff > 0]
    print(f"reproduction vs stored anchor_v4: max |ΔS_pred_B| = "
          f"{np.nanmax(repro.S_pred_B_maxabsdiff):.3e} over {len(repro)} seeds "
          f"({len(bad)} non-identical); "
          f"max |Δ family-mean α relRMSE| = "
          f"{np.nanmax(repro.alpha_family_absdiff):.3e}")
    gap = np.nanmax(np.abs(repro.alpha_freerun_minus_testrun))
    print(f"stored freerun−testrun α relRMSE, max |gap| across seeds: {gap:.3e} "
          f"→ α is rollout-independent (evaluated at observed stocks)")

    per_seed = per_seed_table(dumps)
    vol, years, a_obs = target_volatility(dumps)
    cov = covariation_flag(dumps)
    peryear, trim = per_year_error(dumps)
    summary = summarise(per_seed, vol, trim)

    # Stage A vs Stage B is WP-1d's business; the paired shift is recorded
    # here only because the dumps make it free, and is not interpreted.
    ab = []
    for nm in ["alpha_cc", "alpha_refc", "alpha_win", "alpha_dr"]:
        s = per_seed[(per_seed.split == "test") & (per_seed.eval_at == "at_obs")
                     & (per_seed.channel == nm)]
        a = s[s.stage == "A"].sort_values("seed")["relRMSE"].values
        b = s[s.stage == "B"].sort_values("seed")["relRMSE"].values
        ab.append(dict(channel=nm, HL_shift_B_minus_A_relRMSE=hl_shift(b, a)))
    ab = pd.DataFrame(ab)

    os.makedirs(args.out, exist_ok=True)
    summary.to_csv(os.path.join(args.out, "wp1a_alpha_channels.csv"), index=False)
    per_seed.to_csv(os.path.join(args.out, "wp1a_alpha_per_seed.csv"), index=False)
    vol.merge(cov, on="channel").merge(ab, on="channel", how="left").to_csv(
        os.path.join(args.out, "wp1a_alpha_target_volatility.csv"), index=False)
    repro.to_csv(os.path.join(args.out, "wp1a_reproduction_check.csv"), index=False)
    peryear.to_csv(os.path.join(args.out, "wp1a_alpha_error_by_year.csv"), index=False)

    gam_alpha = float("nan")
    gam_csv = os.path.join(GAM_DIR, "per_seed.csv")
    if os.path.exists(gam_csv):
        g = pd.read_csv(gam_csv)
        g = g[g.rollout == "freerun"]["alpha_relRMSE"].dropna()
        if len(g):
            gam_alpha = float(np.median(g))

    fig_channels(per_seed, summary, vol, gam_alpha,
                 os.path.join(args.out, "wp1a_alpha_channels.png"))
    fig_trajectories(dumps, years, a_obs,
                     os.path.join(args.out, "wp1a_alpha_trajectories.png"))
    fig_error_by_year(peryear,
                      os.path.join(args.out, "wp1a_alpha_error_by_year.png"))

    pd.set_option("display.width", 160, "display.max_columns", 40)
    print("\n--- per-channel α relRMSE%, test window, Stage B @ observed stocks ---")
    cols = ["channel", "n_seeds", "relRMSE_median", "relRMSE_iqr",
            "logRMSE_nats_median", "alpha_log_std",
            "logRMSE_over_alpha_log_std_median", "logRMSE_const_geomean_test",
            "logRMSE_median_over_const_geomean",
            "yoy_dlog_rms_test", "persistence_boundary_relRMSE_test",
            "persistence_rolling_relRMSE_test",
            "relRMSE_median_over_persistence_boundary"]
    print(summary[[c for c in cols if c in summary]].round(3).to_string(index=False))
    print(f"\nGAM family-mean α relRMSE (deterministic reference line): {gam_alpha:.2f}%")
    print("\n--- error concentration in time (robustness readout, NOT a metric change) ---")
    print(trim.round(3).to_string(index=False))
    print("\n--- between-year F–S covariation (indicative only; WP-3 quantifies) ---")
    print(cov.round(3).to_string(index=False))
    print(f"\nwrote {args.out}/wp1a_alpha_channels.{{csv,png,pdf}}, "
          f"wp1a_alpha_per_seed.csv, wp1a_alpha_target_volatility.csv, "
          f"wp1a_alpha_error_by_year.{{csv,png,pdf}}, "
          f"wp1a_reproduction_check.csv, wp1a_alpha_trajectories.{{png,pdf}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
