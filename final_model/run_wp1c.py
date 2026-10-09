#!/usr/bin/env python3
"""
run_wp1c.py — WP-1c: variance decomposition on α(t)
===================================================

Spec: "Per channel and year: between-seed SD of `α_pred` versus the gap between
ensemble-mean `α_pred` and `alpha_obs`.  Bias-dominated and variance-dominated
require opposite write-ups."

No new fits.  Consumes the 35-seed α dumps written for WP-1a
(`analysis/wp1a/alpha_seed*.npz`), which reproduce the stored `anchor_v4`
rollouts bit-exactly (verified there).

The decomposition
-----------------
For channel k and year t, with S seeds, α_pred[s,t,k]:

    E_s[(α_pred − α_obs)²]  =  bias(t,k)²  +  σ(t,k)²

with bias = E_s[α_pred] − α_obs and σ² the between-seed variance.  The
plug-in estimator of bias² is biased upward by σ²/S at finite S, so the
unbiased pair is used throughout:

    σ̂²    = Var_s(α_pred, ddof=1)
    biaŝ² = (mean_s α_pred − α_obs)² − σ̂²/S

These two sum exactly to the realised mean-over-seeds squared error, so the
decomposition is an identity on the ensemble, not an approximation.

Two spaces, because they answer different questions
---------------------------------------------------
* **level** — the space the headline relRMSE is reported in (spec §2), so the
  shares here decompose the published number;
* **log** — the space the loss actually optimises (`log α_pred − log α_obs`
  scaled by `stats["alpha_log_std"]`).  WP-1a showed the level number is a
  harsh readout of a decent log fit, so the log decomposition is reported
  alongside rather than instead.

What the shares mean operationally
----------------------------------
The bias term is what survives averaging: an S-member ensemble mean has error
`sqrt(b² + σ²/S)`, so `sqrt(b²)` is the S → ∞ asymptote and

    sqrt(b² / (b² + σ²))

is the fraction of a typical seed's error that no amount of seed-averaging
removes.  Both the asymptote (`relRMSE_ensemble`) and the error of the 35-seed
ensemble actually in hand (`relRMSE_ensemble_at_S`) are reported, since where
the bias is small the two differ noticeably.  A variance-dominated channel is
an ensembling problem — cheap to fix, and the published single-seed number
overstates what the model knows.  A bias-dominated channel is not: every seed
is wrong the same way, and the fault
is in the model, the target, or the data, never in the optimiser's luck.

Reported per channel and split: bias², σ², the bias share with a seed-
bootstrap interval, the ensemble and expected-single-seed relRMSE, and the
between-seed coefficient of variation.  Per channel and year: bias(t), σ(t),
and the year's share of the channel's total MSE.

Robustness readout (not a metric change).  WP-1a established that 2018–2019
carry 94–98% of the two scrap channels' test MSE through a reconstructed
`S_scrap` denominator that collapses 4,591 → 500 kt.  The decomposition is
therefore also reported with those two years dropped, purely to say whether
the bias/variance verdict is a property of the channel or of the tail.  The
published metric stays the full pre-registered window (spec §2 constraint 3);
nothing is selected on the trimmed number.

    python run_wp1c.py --check
    python run_wp1c.py
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DUMP_DIR = os.path.join(HERE, "analysis", "wp1a")
OUT_DIR = os.path.join(HERE, "analysis")

CHANNELS = ["alpha_cc", "alpha_refc", "alpha_win", "alpha_dr"]
# zinc_colloc_v5.ALPHA_PARENT_STOCK_IDX = (0, 1, 3, 3)
PARENT = {"alpha_cc": "Concentrate", "alpha_refc": "Refined",
          "alpha_win": "Scrap", "alpha_dr": "Scrap"}

CH_COLOURS = {"alpha_cc": "#0072B2", "alpha_refc": "#009E73",
              "alpha_win": "#D55E00", "alpha_dr": "#CC79A7"}
BIAS_C, VAR_C, GREY = "#B0511E", "#56B4E9", "#555555"

N_BOOT = 2000
TRIM_TAIL = 2          # years dropped in the robustness readout only


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_dumps(dump_dir=DUMP_DIR):
    paths = sorted(glob.glob(os.path.join(dump_dir, "alpha_seed*.npz")),
                   key=lambda p: int(os.path.basename(p)[10:-4]))
    if not paths:
        raise SystemExit(f"no dumps in {dump_dir} — run zinc_alpha_lab.py first")
    return [(int(os.path.basename(p)[10:-4]), np.load(p, allow_pickle=True))
            for p in paths]


def stack(dumps, stage="B", where="at_obs"):
    """(S, T, 4) predictions, plus the seed-invariant target and axes."""
    key = f"alpha_pred_{stage}_{where}"
    P = np.stack([np.asarray(d[key], float) for _, d in dumps])
    _, d0 = dumps[0]
    return P, np.asarray(d0["alpha_obs"], float), np.asarray(d0["years"], float).ravel()


def splits(dumps):
    _, d0 = dumps[0]
    T = len(np.asarray(d0["years"]).ravel())
    return dict(train=np.asarray(d0["mask_train"], bool),
                val=np.asarray(d0["mask_val"], bool),
                test=np.asarray(d0["mask_test"], bool),
                all=np.ones(T, bool))


def rollout_invariance(dumps):
    """WP-1a flag 1: `use_stock_input: false`, so α never sees the state and
    α-at-observed-stocks equals α-along-the-rollout identically.  Re-verified
    rather than assumed, because if it ever stopped holding the ensemble here
    would be decomposing a different quantity per rollout."""
    g = 0.0
    for _, d in dumps:
        for stage in ("A", "B"):
            a, b = f"alpha_pred_{stage}_at_obs", f"alpha_pred_{stage}_at_pred"
            if a in d.files and b in d.files:
                g = max(g, float(np.nanmax(np.abs(d[a] - d[b]))))
    return g


# ---------------------------------------------------------------------------
# the decomposition
# ---------------------------------------------------------------------------
def _transform(P, obs, space):
    """Level space as-is; log space is the residual the loss is built on.

    Non-positive entries would be undefined in log space; they are mapped to
    NaN and dropped by the finite mask rather than clipped, so a bad value is
    excluded from the decomposition instead of silently biasing it.
    """
    if space == "level":
        return P, obs
    lp = np.where(P > 0, np.log(np.where(P > 0, P, 1.0)), np.nan)
    lo = np.where(obs > 0, np.log(np.where(obs > 0, obs, 1.0)), np.nan)
    return lp, lo


def per_year(P, obs, k, space):
    """bias(t), σ(t) and the unbiased bias²(t) for one channel, all years."""
    X, O = _transform(P, obs, space)
    x, o = X[:, :, k], O[:, k]                       # (S, T), (T,)
    S = x.shape[0]
    ok = np.isfinite(o) & np.isfinite(x).all(axis=0)
    mean = np.where(ok, x.mean(axis=0), np.nan)
    var = np.where(ok, x.var(axis=0, ddof=1), np.nan)
    bias = mean - np.where(ok, o, np.nan)
    b2 = bias ** 2 - var / S                         # unbiased at finite S
    return dict(ok=ok, mean=mean, obs=np.where(ok, o, np.nan), bias=bias,
                sd=np.sqrt(var), var=var, b2=b2, S=S)


def aggregate(py, mask, obs_raw=None, space="level"):
    """Average the per-year terms over a split and turn them into relRMSEs.

    `b2` is averaged before being clipped at zero: a per-year negative is a
    finite-seed artefact of the unbiased estimator (bias ≈ 0 there) and
    dropping the sign early would reintroduce the upward bias the correction
    removes.  The clip is applied once, to the mean.
    """
    m = mask & py["ok"]
    n = int(m.sum())
    if n == 0:
        return dict(n_years=0)
    b2 = float(np.mean(py["b2"][m]))
    var = float(np.mean(py["var"][m]))
    mse = max(b2, 0.0) + var
    share = max(b2, 0.0) / mse if mse > 0 else np.nan
    out = dict(n_years=n, bias2=b2, var=var, expected_mse=b2 + var,
               bias_share=share,
               rms_bias=float(np.sqrt(max(b2, 0.0))),
               rms_sd=float(np.sqrt(var)),
               mean_signed_bias=float(np.mean(py["bias"][m])),
               const_offset_share=float(np.mean(py["bias"][m]) ** 2
                                        / np.mean(py["bias"][m] ** 2))
               if np.mean(py["bias"][m] ** 2) > 0 else np.nan,
               # A ratio to the mean is only meaningful where the mean is a
               # positive magnitude, i.e. in level space; a log-space "CV"
               # would divide by a quantity that crosses zero.  The log-space
               # dispersion is `rms_sd`, already in nats.
               cv_between_seeds=(float(np.mean(py["sd"][m] / np.abs(py["mean"][m])))
                                 if space == "level" else np.nan))
    if obs_raw is not None:                          # level space only
        den = float(np.mean(np.abs(obs_raw[m])))
        S = int(py["S"])
        # `relRMSE_ensemble` is the S → ∞ asymptote, sqrt(bias²).  The ensemble
        # actually in hand has S members, so its own error carries a residual
        # σ²/S — reported separately rather than conflated with the asymptote.
        out["relRMSE_ensemble"] = 100.0 * out["rms_bias"] / den
        out["relRMSE_ensemble_at_S"] = 100.0 * float(np.sqrt(max(b2, 0.0) + var / S)) / den
        out["relRMSE_expected_single"] = 100.0 * float(np.sqrt(max(b2, 0.0) + var)) / den
        out["ensembling_gain_pct"] = 100.0 * (1.0 - out["relRMSE_ensemble_at_S"]
                                              / out["relRMSE_expected_single"])
        out["ensembling_gain_asymptotic_pct"] = 100.0 * (
            1.0 - out["relRMSE_ensemble"] / out["relRMSE_expected_single"])
    return out


def bootstrap_share(P, obs, k, mask, space, n_boot=N_BOOT, rng_seed=0):
    """Percentile interval on the bias share from resampling seeds.

    The ensemble is the only source of randomness available without refitting,
    so this is uncertainty in the share *given these 35 fits* — it is not a
    parameter posterior.  Reported as an interval, per project convention.
    """
    rng = np.random.default_rng(rng_seed)
    S = P.shape[0]
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, S, S)
        py = per_year(P[idx], obs, k, space)
        a = aggregate(py, mask, space=space)
        vals[i] = a.get("bias_share", np.nan)
    v = vals[np.isfinite(vals)]
    if v.size == 0:
        return np.nan, np.nan
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def summary_table(dumps, stages=("B", "A"), spaces=("level", "log"),
                  do_boot=True):
    masks = splits(dumps)
    rows = []
    for stage in stages:
        P, obs, _ = stack(dumps, stage=stage)
        for space in spaces:
            for k, nm in enumerate(CHANNELS):
                py = per_year(P, obs, k, space)
                for split, m in masks.items():
                    a = aggregate(py, m, space=space,
                                  obs_raw=obs[:, k] if space == "level" else None)
                    if not a.get("n_years"):
                        continue
                    lo = hi = np.nan
                    if do_boot and stage == "B":
                        lo, hi = bootstrap_share(P, obs, k, m, space)
                    rows.append(dict(channel=nm, parent_stock=PARENT[nm],
                                     split=split, stage=stage, space=space,
                                     n_seeds=P.shape[0],
                                     bias_share_lo=lo, bias_share_hi=hi, **a))
    return pd.DataFrame(rows)


def by_year_table(dumps, stage="B"):
    P, obs, years = stack(dumps, stage=stage)
    masks = splits(dumps)
    lab = np.array(["train"] * len(years), dtype=object)
    lab[masks["val"]] = "val"
    lab[masks["test"]] = "test"
    rows = []
    for space in ("level", "log"):
        for k, nm in enumerate(CHANNELS):
            py = per_year(P, obs, k, space)
            tot = float(np.nansum(np.where(masks["test"] & py["ok"],
                                           np.maximum(py["b2"], 0) + py["var"], 0.0)))
            for t in range(len(years)):
                if not py["ok"][t]:
                    continue
                b2 = float(max(py["b2"][t], 0.0))
                v = float(py["var"][t])
                rows.append(dict(
                    channel=nm, space=space, stage=stage, year=float(years[t]),
                    split=str(lab[t]),
                    alpha_obs=float(py["obs"][t]),
                    ensemble_mean=float(py["mean"][t]),
                    bias=float(py["bias"][t]),
                    between_seed_sd=float(py["sd"][t]),
                    bias2_unbiased=float(py["b2"][t]),
                    bias_share=b2 / (b2 + v) if (b2 + v) > 0 else np.nan,
                    test_mse_share=((b2 + v) / tot
                                    if (masks["test"][t] and tot > 0) else np.nan)))
    return pd.DataFrame(rows)


def trimmed_table(dumps, stage="B", drop=TRIM_TAIL):
    """Same decomposition with the last `drop` test years removed.

    ROBUSTNESS READOUT ONLY (WP-1a's convention).  Says whether a channel's
    bias/variance verdict is a property of the channel or of the two years in
    which the reconstructed `S_scrap` denominator collapses.
    """
    P, obs, years = stack(dumps, stage=stage)
    mt = splits(dumps)["test"]
    idx = np.where(mt)[0]
    keep = np.zeros_like(mt)
    keep[idx[:-drop] if drop else idx] = True
    rows = []
    for space in ("level", "log"):
        for k, nm in enumerate(CHANNELS):
            py = per_year(P, obs, k, space)
            full = aggregate(py, mt, space=space,
                             obs_raw=obs[:, k] if space == "level" else None)
            trim = aggregate(py, keep, space=space,
                             obs_raw=obs[:, k] if space == "level" else None)
            rows.append(dict(
                channel=nm, space=space, stage=stage,
                dropped_years=f"{years[idx[-drop]]:.0f}–{years[idx[-1]]:.0f}" if drop else "",
                bias_share_full=full["bias_share"], bias_share_trimmed=trim["bias_share"],
                relRMSE_ensemble_full=full.get("relRMSE_ensemble", np.nan),
                relRMSE_ensemble_trimmed=trim.get("relRMSE_ensemble", np.nan),
                relRMSE_ensemble_at_S_full=full.get("relRMSE_ensemble_at_S", np.nan),
                relRMSE_ensemble_at_S_trimmed=trim.get("relRMSE_ensemble_at_S", np.nan),
                relRMSE_expected_full=full.get("relRMSE_expected_single", np.nan),
                relRMSE_expected_trimmed=trim.get("relRMSE_expected_single", np.nan)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_decomposition(summary, trimmed, alpha_log_std, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    lvl = summary[(summary.stage == "B") & (summary.space == "level")]
    log = summary[(summary.stage == "B") & (summary.space == "log")]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.4))

    # (a) where the error lives, per split
    ax = axes[0]
    order, xs, ticks, centres = ["train", "val", "test"], [], [], []
    x = 0.0
    for nm in CHANNELS:
        centres.append((nm, x + 1.0))
        for split in order:
            r = lvl[(lvl.channel == nm) & (lvl.split == split)]
            if not len(r):
                x += 1.0
                continue
            r = r.iloc[0]
            tot = max(r.bias2, 0.0) + r["var"]
            ax.bar(x, 100 * max(r.bias2, 0.0) / tot, color=BIAS_C, width=0.78)
            ax.bar(x, 100 * r["var"] / tot, bottom=100 * max(r.bias2, 0.0) / tot,
                   color=VAR_C, width=0.78)
            if split == "test":
                ax.errorbar(x, 100 * r.bias_share,
                            yerr=[[100 * (r.bias_share - r.bias_share_lo)],
                                  [100 * (r.bias_share_hi - r.bias_share)]],
                            color="k", lw=1.1, capsize=2.5, zorder=5)
            xs.append(x)
            ticks.append({"train": "tr", "val": "v", "test": "te"}[split])
            x += 1.0
        x += 0.7
    ax.set_xticks(xs)
    ax.set_xticklabels(ticks, fontsize=8)
    for nm, cx in centres:
        ax.text(cx, -0.11, nm, ha="center", fontsize=8.5, color=CH_COLOURS[nm],
                transform=ax.get_xaxis_transform())
    ax.set_ylim(0, 100)
    ax.set_ylabel("share of expected squared error (%)")
    ax.legend(handles=[Patch(color=BIAS_C, label="bias²  (survives ensembling)"),
                       Patch(color=VAR_C, label="between-seed variance  (averages away)"),
                       Line2D([], [], color="k", lw=1.1,
                              label="bias share, 95% seed bootstrap")],
              fontsize=7.3, loc="lower left", framealpha=0.92, facecolor="white",
              edgecolor="none")
    ax.set_title("(a) level space, by split (train / val / test)", loc="left",
                 fontsize=10)

    # (b) what ensembling actually buys, test window
    ax = axes[1]
    w = 0.36
    for i, nm in enumerate(CHANNELS):
        r = lvl[(lvl.channel == nm) & (lvl.split == "test")].iloc[0]
        ax.bar(i - w / 2, r.relRMSE_expected_single, width=w, color=CH_COLOURS[nm],
               alpha=0.45, edgecolor="none")
        ax.bar(i + w / 2, r.relRMSE_ensemble_at_S, width=w, color=CH_COLOURS[nm])
        t = trimmed[(trimmed.channel == nm) & (trimmed.space == "level")].iloc[0]
        ax.plot([i + w / 2 - 0.15, i + w / 2 + 0.15],
                [t.relRMSE_ensemble_at_S_trimmed] * 2, color="k", lw=1.5, zorder=4)
    n_seeds = int(lvl.n_seeds.iloc[0])
    ax.set_ylabel("α relRMSE, test window (%)")
    ax.set_title("(b) irreducible under seed-averaging", loc="left", fontsize=10)
    ax.legend(handles=[Patch(color=GREY, alpha=0.45, label="expected single seed"),
                       Patch(color=GREY, label=f"{n_seeds}-seed ensemble mean"),
                       Line2D([], [], color="k", lw=1.5,
                              label=f"ensemble, last {TRIM_TAIL} yr dropped")],
              fontsize=7.3, frameon=False, loc="upper left")

    # (c) log space — the residual the loss is built on.  Grouped, not stacked:
    # RMS components add in quadrature, so stacking them would imply a sum
    # that does not hold.
    ax = axes[2]
    w = 0.34
    for i, nm in enumerate(CHANNELS):
        r = log[(log.channel == nm) & (log.split == "test")].iloc[0]
        ax.bar(i - w / 2, r.rms_bias, width=w, color=BIAS_C)
        ax.bar(i + w / 2, r.rms_sd, width=w, color=VAR_C)
        ax.plot([i - 0.42, i + 0.42], [alpha_log_std[i]] * 2, color=GREY,
                lw=1.4, ls="--", zorder=4)
    ax.set_ylabel("log-space residual components (nats)")
    ax.set_title("(c) log space, test window", loc="left", fontsize=10)
    ax.legend(handles=[Patch(color=BIAS_C, label="|bias| of the ensemble mean"),
                       Patch(color=VAR_C, label="between-seed SD"),
                       Line2D([], [], color=GREY, lw=1.4, ls="--",
                              label="alpha_log_std (loss scale)")],
              fontsize=7.3, frameon=False, loc="upper left")

    for ax in axes[1:]:
        ax.set_xticks(range(len(CHANNELS)))
        ax.set_xticklabels(CHANNELS, rotation=15, fontsize=8.5)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", lw=0.4, alpha=0.3)
    fig.suptitle(f"WP-1c  bias–variance decomposition of α(t) across the "
                 f"{n_seeds}-seed anchor_v4 ensemble",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=300)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


def fig_by_year(byyear, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = byyear[(byyear.space == "level") & (byyear.stage == "B")]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.6), sharex=True)
    t0 = float(sub[sub.split == "test"].year.min())
    for ax, nm in zip(axes.ravel(), CHANNELS):
        s = sub[sub.channel == nm].sort_values("year")
        yr = s.year.values
        ax.fill_between(yr, s.ensemble_mean - s.between_seed_sd,
                        s.ensemble_mean + s.between_seed_sd,
                        color=CH_COLOURS[nm], alpha=0.30, lw=0,
                        label="ensemble mean ± between-seed SD")
        ax.plot(yr, s.ensemble_mean.values, color=CH_COLOURS[nm], lw=1.8)
        ax.plot(yr, s.alpha_obs.values, color="k", lw=1.2, ls="--", marker="o",
                ms=2.6, label="α_obs")
        ax.axvline(t0, color=GREY, lw=0.9, ls=":")
        ax.set_yscale("log")
        ax.set_title(f"{nm}  (parent {PARENT[nm]})", loc="left", fontsize=9.5,
                     color=CH_COLOURS[nm])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(lw=0.4, alpha=0.3)

        tw = ax.twinx()                      # bias share, test years only
        st = s[s.split == "test"]
        tw.bar(st.year.values, st.bias_share.values, width=0.62, color=GREY,
               alpha=0.28)
        # A twin axes is drawn above its parent whatever the artists' zorder,
        # so the parent is lifted and its patch made transparent to keep the
        # bars behind the trajectories.
        ax.set_zorder(tw.get_zorder() + 1)
        ax.patch.set_visible(False)
        tw.set_ylim(0, 1)
        tw.set_yticks([0, 0.5, 1.0])
        tw.tick_params(labelsize=7)
        tw.spines[["top"]].set_visible(False)
        if nm == CHANNELS[1] or nm == CHANNELS[3]:
            tw.set_ylabel("bias share", fontsize=8)
    axes[0, 0].legend(fontsize=7.6, loc="upper left", framealpha=0.9,
                      facecolor="white", edgecolor="none")
    for ax in axes[-1]:
        ax.set_xlabel("year")
    fig.suptitle("WP-1c  ensemble mean ± between-seed SD against the target; "
                 "grey bars = per-year bias share of MSE (test window)",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=300)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="WP-1c variance decomposition on α(t)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dumps", default=DUMP_DIR)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--no-bootstrap", action="store_true")
    args = ap.parse_args(argv)

    import zinc_alpha_lab as lab
    lab.check(verbose=True)
    if args.check:
        return 0

    dumps = load_dumps(args.dumps)
    print(f"\nloaded {len(dumps)} seed dumps from {args.dumps}")
    g = rollout_invariance(dumps)
    print(f"max |α(at S_obs) − α(at S_pred)| across seeds and stages: {g:.3e} "
          f"→ one α per seed, not one per rollout (WP-1a flag 1)")

    summary = summary_table(dumps, do_boot=not args.no_bootstrap)
    byyear = by_year_table(dumps)
    trimmed = trimmed_table(dumps)

    os.makedirs(args.out, exist_ok=True)
    summary.to_csv(os.path.join(args.out, "wp1c_variance_decomposition.csv"),
                   index=False)
    byyear.to_csv(os.path.join(args.out, "wp1c_by_year.csv"), index=False)
    trimmed.to_csv(os.path.join(args.out, "wp1c_trimmed.csv"), index=False)

    alpha_log_std = np.asarray(dumps[0][1]["alpha_log_std"], float)
    fig_decomposition(summary, trimmed, alpha_log_std,
                      os.path.join(args.out, "wp1c_variance_decomposition.png"))
    fig_by_year(byyear, os.path.join(args.out, "wp1c_by_year.png"))

    pd.set_option("display.width", 200, "display.max_columns", 40)
    for space in ("level", "log"):
        s = summary[(summary.stage == "B") & (summary.space == space)
                    & (summary.split == "test")]
        cols = ["channel", "parent_stock", "bias2", "var", "bias_share",
                "bias_share_lo", "bias_share_hi", "cv_between_seeds",
                "mean_signed_bias", "const_offset_share"]
        if space == "level":
            cols += ["relRMSE_expected_single", "relRMSE_ensemble_at_S",
                     "relRMSE_ensemble", "ensembling_gain_pct"]
        print(f"\n--- {space} space, test window, Stage B ---")
        print(s[[c for c in cols if c in s]].round(4).to_string(index=False))

    print("\n--- bias share by split (level, Stage B) ---")
    p = (summary[(summary.stage == "B") & (summary.space == "level")]
         .pivot(index="channel", columns="split", values="bias_share")
         .reindex(CHANNELS)[["train", "val", "test"]])
    print(p.round(3).to_string())

    print(f"\n--- robustness readout: last {TRIM_TAIL} test years dropped "
          f"(NOT a metric change) ---")
    t = trimmed[trimmed.space == "level"]
    print(t[["channel", "dropped_years", "bias_share_full", "bias_share_trimmed",
             "relRMSE_ensemble_at_S_full", "relRMSE_ensemble_at_S_trimmed",
             "relRMSE_expected_trimmed"]].round(3).to_string(index=False))

    print(f"\nwrote {args.out}/wp1c_variance_decomposition.{{csv,png,pdf}}, "
          f"wp1c_by_year.{{csv,png,pdf}}, wp1c_trimmed.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
