#!/usr/bin/env python3
"""
run_wp3.py — WP-3: the synthetic twin, its observation operators, and the
alpha-target bias
=========================================================================

WP-3 is infrastructure, so most of its value is in `zinc_synth_lab.py` and the
datasets under `analysis/synth/`.  This driver produces the three things the
spec asks WP-3 to *report*:

  Part 1  the observation operators, verified.  Mass conservation, exact
          cross-frequency aggregation, the cp rate identity, pinned-tau
          recovery, and the boxcar's harmonic nulls measured against the
          closed-form |sinc(pi f Delta)| — the last is WP-4a's benchmark and
          is computed here because the `season` arm is what validates the
          generator.

  Part 2  **the alpha-target bias.**  The spec's named extra deliverable and
          the direct test of WP-1a's secondary hypothesis: `alpha_obs` is an
          integral over an interpolated point stock, so it is a biased
          estimator of the time-averaged alpha.  On the twin the truth is
          known, so the bias splits exactly into a within-window covariance
          term and a trapezoid quadrature term, per channel and per window
          width.

  Part 3  acceptance.  The real pipeline, unmodified, fitted to the noiseless
          annually-sampled twin.  Pre-registered tolerance, stated below and
          fixed before the fits were run:

            (A)  median over seeds of per-channel relRMSE(alpha_pred,
                 alpha_true) on the trainval window <= 15%, all four channels;
            (B)  reported alongside and not part of the gate: the split of
                 that error into the target's own bias relRMSE(alpha_obs,
                 alpha_true) and the estimator's departure from its target
                 relRMSE(alpha_pred, alpha_obs).

          If (A) fails the spec says to report it rather than proceed, and
          Part 2 is what says why.

    python run_wp3.py --check
    python run_wp3.py --operators --bias
    python run_wp3.py --accept --seeds 0,1,2,3,4,5,6,7
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import zinc_synth_lab as S
import zinc_circ_lab as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "analysis")
SYNTH_DIR = os.path.join(OUT_DIR, "synth")
FIT_DIR = os.path.join(SYNTH_DIR, "fits")

COL = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00", "#56B4E9"]
GREY = "#555555"

ALPHA_NAMES = S.ALPHA_NAMES
TAU_SUP_NAMES = S.TAU_SUP_NAMES

# Pre-registered acceptance tolerance (see module docstring).
ACCEPT_TOL_PCT = 15.0
ACCEPT_ARM, ACCEPT_DELTA = "base", 1.0
TRAINVAL_END = 2007.0


def hl(v):
    """`(estimate, lo, hi)`; `zinc_circ_lab.hodges_lehmann` returns a dict and
    leaves the interval NaN below six observations."""
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if v.size == 0:
        return (np.nan, np.nan, np.nan)
    d = C.hodges_lehmann(v)
    return (float(d["hl"]), float(d["lo"]), float(d["hi"]))


def rel_rmse_pct(pred, obs, mask=None):
    p, o = np.asarray(pred, float), np.asarray(obs, float)
    m = np.isfinite(p) & np.isfinite(o)
    if mask is not None:
        m &= np.asarray(mask, bool)
    if not m.any():
        return np.nan
    return 100.0 * np.sqrt(np.mean((p[m] - o[m]) ** 2)) / max(np.mean(np.abs(o[m])), 1e-12)


# ===========================================================================
# Part 1 — operators
# ===========================================================================
def sinc_gain(f, delta):
    """|sinc(pi f Delta)| — the boxcar (period-integral) transfer function.

    A period integral of width Delta applied to exp(2 pi i f t) returns
    Delta * sinc(pi f Delta) * exp(...) with sinc(x) = sin(x)/x, so the gain
    of the *mean* rate is |sin(pi f Delta) / (pi f Delta)|, which is exactly
    zero at f = k/Delta.  Annual reporting therefore annihilates 1/yr, 2/yr,
    3/yr rather than aliasing them.
    """
    x = np.pi * np.asarray(f, float) * float(delta)
    return np.abs(np.where(np.abs(x) < 1e-12, 1.0, np.sin(x) / np.where(x == 0, 1, x)))


def part1_operators(out_dir=OUT_DIR):
    """Verification table plus the measured-vs-closed-form boxcar gain."""
    man = json.load(open(os.path.join(SYNTH_DIR, "manifest.json")))
    rows = []
    for arm, rep in man["arms"].items():
        for k, v in rep["verify"].items():
            rows.append(dict(arm=arm, check=k, value=float(v)))
        for k, name in enumerate(["Concentrate", "Refined", "In-Use", "Scrap"]):
            rows.append(dict(arm=arm, check=f"realism_relRMSE_{name}",
                             value=float(rep["realism"]["rel_rmse_pct"][k])))
    ver = pd.DataFrame(rows)
    ver.to_csv(os.path.join(out_dir, "wp3_operator_checks.csv"), index=False)

    # ---- boxcar gain, measured on the twin's own grid --------------------
    # Phase-safe by construction: each component is isolated analytically, its
    # window mean is computed by Simpson on the same 96-node/yr grid the twin
    # was integrated on, and the amplitude is recovered as rms * sqrt(2).
    # Projecting the *sum* onto one component's phase is not safe — at
    # Delta = 1 the window centres fall where sin(2 pi f (tc - t0)) is
    # identically zero for every integer f, so the projection is rank
    # deficient and reports a spurious zero rather than the boxcar's.
    d = np.load(os.path.join(SYNTH_DIR, "season_dense.npz"), allow_pickle=True)
    t = d["t"]
    n_sub = int(json.loads(str(d["meta_json"]))["n_sub"])
    h = 1.0 / n_sub
    t0 = float(S.TRUTH["season_t0"])
    grows = []
    for (chan, f, amp, ph) in S.TRUTH["season"]:
        comp = amp * np.sin(2 * np.pi * f * (t - t0) + ph)
        for delta in (0.25, 1.0, 2.0, 5.0):
            step = int(round(delta * n_sub))
            if step % 2:
                continue
            n_w = (t.size - 1) // step
            idx = np.arange(n_w + 1) * step
            w = np.array([S._simpson(comp[idx[i]:idx[i + 1] + 1][None, :], h)[0]
                          / delta for i in range(n_w)])
            grows.append(dict(
                channel=chan, freq_per_yr=f, amplitude_nats=amp, delta=delta,
                n_windows=int(n_w),
                theory_sinc=float(sinc_gain(f, delta)),
                measured_gain_rms=float(np.sqrt(2.0) * np.sqrt(np.mean(w ** 2)) / amp),
                measured_gain_peak=float(np.max(np.abs(w)) / amp),
                harmonic=bool(abs(f * delta - round(f * delta)) < 1e-9)))
    gain = pd.DataFrame(grows)
    gain.to_csv(os.path.join(out_dir, "wp3_boxcar_gain.csv"), index=False)
    return ver, gain


# ===========================================================================
# Part 2 — the alpha-target bias
# ===========================================================================
def part2_bias(arms=("base", "season", "state"),
               deltas=(0.25, 1.0, 2.0, 5.0, 10.0), out_dir=OUT_DIR):
    """Decompose `alpha_obs` against the analytic truth, per channel and Delta."""
    rows, per_window = [], []
    for arm in arms:
        for dl in deltas:
            tag = f"{arm}_d{S._dtag(dl)}_clean"
            p = os.path.join(SYNTH_DIR, f"{tag}.npz")
            if not os.path.exists(p):
                continue
            tr = S.load_truth(p)
            aw, au, ao = (tr["alphas_window_weighted"], tr["alphas_window_unweighted"],
                          tr["alphas_window_obs"])
            ap = tr["alphas_point"][1:]      # alpha_true at the REPORTING node
            corr, cov = tr["corr_FS"], tr["cov_term"]
            iS, tS = tr["int_S"], tr["trapz_S"]
            yrs = tr["window_years"]
            for k, name in enumerate(ALPHA_NAMES):
                tot = ao[:, k] / np.maximum(au[:, k], 1e-12) - 1.0
                wgt = aw[:, k] / np.maximum(au[:, k], 1e-12) - 1.0
                qud = iS[:, k] / np.maximum(tS[:, k], 1e-12) - 1.0
                cov_pred = cov[:, k] / np.maximum(au[:, k], 1e-12)
                rows.append(dict(
                    arm=arm, delta=float(dl), channel=name, n_windows=int(yrs.size),
                    bias_total_pct=100.0 * float(np.median(tot)),
                    bias_weighting_pct=100.0 * float(np.median(wgt)),
                    bias_quadrature_pct=100.0 * float(np.median(qud)),
                    cov_term_pred_pct=100.0 * float(np.median(cov_pred)),
                    bias_alignment_pct=100.0 * float(np.median(
                        au[:, k] / np.maximum(ap[:, k], 1e-12) - 1.0)),
                    relRMSE_obs_vs_point_pct=rel_rmse_pct(ao[:, k], ap[:, k]),
                    bias_total_max_pct=100.0 * float(np.max(np.abs(tot))),
                    relRMSE_obs_vs_true_pct=rel_rmse_pct(ao[:, k], au[:, k]),
                    relRMSE_obs_vs_weighted_pct=rel_rmse_pct(ao[:, k], aw[:, k]),
                    corr_FS_median=float(np.nanmedian(corr[:, k])),
                ))
                for i in range(yrs.size):
                    per_window.append(dict(
                        arm=arm, delta=float(dl), channel=name, year=float(yrs[i]),
                        alpha_true_unweighted=float(au[i, k]),
                        alpha_true_weighted=float(aw[i, k]),
                        alpha_obs=float(ao[i, k]),
                        bias_total_pct=100.0 * float(tot[i]),
                        bias_weighting_pct=100.0 * float(wgt[i]),
                        bias_quadrature_pct=100.0 * float(qud[i]),
                        corr_FS=float(corr[i, k])))
    df = pd.DataFrame(rows)
    dw = pd.DataFrame(per_window)
    df.to_csv(os.path.join(out_dir, "wp3_alpha_target_bias.csv"), index=False)
    dw.to_csv(os.path.join(out_dir, "wp3_alpha_target_bias_by_window.csv"), index=False)

    # identity check: the weighting term must equal Cov_w(a,S)/mean_w(S)/a_unw
    ok = df.assign(gap=(df.bias_weighting_pct - df.cov_term_pred_pct).abs())
    ok[["arm", "delta", "channel", "bias_weighting_pct", "cov_term_pred_pct", "gap"]] \
        .to_csv(os.path.join(out_dir, "wp3_covariance_identity.csv"), index=False)
    return df, dw


# ===========================================================================
# Part 2b — how big would the sub-annual structure have to be?
# ===========================================================================
SWEEP_FREQ = 1.35          # non-harmonic, so the boxcar attenuates rather than
SWEEP_AMPS = (0.0, 0.05, 0.10, 0.20, 0.40, 0.80)   # annihilates it
SWEEP_LO, SWEEP_HI = 5.0, 95.0


def part2b_amplitude_sweep(out_dir=OUT_DIR, delta=1.0):
    """Bias against sub-annual amplitude, on **all four** channels at once.

    The `season` arm puts its oscillation on `alpha_refc` only, so it cannot
    say whether the covariance term could explain the two old-scrap channels
    — the ones WP-1a finds carry the error.  This sweep answers that directly:
    it drives every channel at the same non-harmonic frequency and traces the
    target bias as a function of the log-amplitude, so the question "how much
    within-year structure would it take for target construction to account for
    the observed error?" gets a number instead of an argument.

    Reported alongside the amplitude is the implied peak-to-trough swing of
    the coefficient, `exp(2a) - 1`, because that is the quantity a reviewer
    can judge the plausibility of.
    """
    import zinc_colloc_v5 as v5

    ctx = S.driver_context()
    rows = []
    for amp in SWEEP_AMPS:
        season = [(n, SWEEP_FREQ, float(amp), 0.0) for n in ALPHA_NAMES]
        dense = S.dense_solve(ctx, "base", season=season)
        bias = S.alpha_target_bias(ctx, dense, delta, arm_name="base", season=season)
        for k, name in enumerate(ALPHA_NAMES):
            tot = bias["a_obs"][:, k] / np.maximum(bias["a_unweighted"][:, k], 1e-12) - 1.0
            wgt = bias["a_weighted"][:, k] / np.maximum(bias["a_unweighted"][:, k], 1e-12) - 1.0
            rows.append(dict(
                amplitude_nats=float(amp), channel=name, delta=float(delta),
                peak_to_trough_pct=100.0 * (np.exp(2 * amp) - 1.0),
                bias_total_median_pct=100.0 * float(np.median(tot)),
                bias_total_p95_pct=100.0 * float(np.percentile(np.abs(tot), SWEEP_HI)),
                bias_weighting_median_pct=100.0 * float(np.median(wgt)),
                relRMSE_obs_vs_true_pct=rel_rmse_pct(bias["a_obs"][:, k],
                                                     bias["a_unweighted"][:, k]),
                corr_FS_median=float(np.nanmedian(bias["corr_FS"][:, k]))))
        import jax
        jax.clear_caches()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "wp3_bias_amplitude_sweep.csv"), index=False)
    return df


# ===========================================================================
# Part 2c — observation noise in the alpha target, and where it comes from
# ===========================================================================
# The alpha parents, in `v5.FLOW_NAMES` order, and the stock each divides by.
ALPHA_PARENT_FLOW = ["concentrate_consumption", "refined_consumption",
                     "waelz_input", "direct_reuse_recycling"]
ALPHA_PARENT_STOCK = [0, 1, 3, 3]
NOISE_GROUPS = ("none", "stocks", "flows", "all",
                "parent_flow_only", "parent_stock_only")


def _sigma_group(sigma, group, k=None):
    """A copy of the calibrated sigma with only one group left switched on."""
    import zinc_colloc_v5 as v5
    out = dict(stock=np.zeros(4), flow=np.zeros(len(sigma["flow"])),
               flow_names=sigma["flow_names"], n_seeds=sigma["n_seeds"])
    if group in ("stocks", "all"):
        out["stock"] = np.asarray(sigma["stock"], float).copy()
    if group in ("flows", "all"):
        out["flow"] = np.asarray(sigma["flow"], float).copy()
    if group == "parent_flow_only" and k is not None:
        j = sigma["flow_names"].index(ALPHA_PARENT_FLOW[k])
        out["flow"][j] = sigma["flow"][j]
    if group == "parent_stock_only" and k is not None:
        out["stock"][ALPHA_PARENT_STOCK[k]] = sigma["stock"][ALPHA_PARENT_STOCK[k]]
    return out


def part2c_noise(arms=("base", "season"), deltas=(0.25, 1.0, 2.0, 5.0),
                 n_draw=32, out_dir=OUT_DIR):
    """How much of the alpha target's error is observation noise, not the
    operator — and which observed series it enters through.

    Part 2 measured the *construction* bias with noiseless observations.  This
    measures the other half: the same `F_integral / trapezoid(S)` ratio built
    from stocks and flow integrals carrying the lognormal noise calibrated to
    the real fit's Stage B residuals.  Because those residuals contain model
    error as well as observation error, every number here is an **upper bound**
    on the noise contribution; that is the conservative direction.

    `n_draw` independent noise draws per setting, so the reported error is a
    distribution rather than one realisation of one RNG seed.  The per-group
    ablation switches noise on for one observed series at a time, which is
    what identifies *which* reported series a modeller would have to measure
    better to sharpen a given alpha channel.
    """
    ctx = S.driver_context()
    sigma = S.calibrate_noise()
    rows = []
    for arm in arms:
        dense = S.dense_solve(ctx, arm)
        for dl in deltas:
            bias = S.alpha_target_bias(ctx, dense, dl, arm_name=arm)
            au = bias["a_unweighted"]
            clean = S.sample(dense, dl, ctx, noise_scale=0.0)
            for group in NOISE_GROUPS:
                for k, name in enumerate(ALPHA_NAMES):
                    errs = []
                    for r in range(1 if group == "none" else n_draw):
                        sg = _sigma_group(sigma, group, k)
                        sm = (clean if group == "none" else
                              S.sample(dense, dl, ctx, noise_scale=1.0,
                                       rng_seed=1000 * k + r, sigma=sg))
                        errs.append(rel_rmse_pct(sm["alpha_obs"][1:, k], au[:, k]))
                    e = np.asarray(errs, float)
                    rows.append(dict(
                        arm=arm, delta=float(dl), noise_group=group,
                        channel=name, n_draw=int(e.size),
                        relRMSE_median_pct=float(np.median(e)),
                        relRMSE_q05_pct=float(np.percentile(e, 5)),
                        relRMSE_q95_pct=float(np.percentile(e, 95))))
        import jax
        jax.clear_caches()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "wp3_alpha_target_noise.csv"), index=False)

    # the quadrature decomposition: total^2 ~= construction^2 + noise^2
    piv = df[df.noise_group.isin(("none", "all"))].pivot_table(
        index=["arm", "delta", "channel"], columns="noise_group",
        values="relRMSE_median_pct").reset_index()
    piv["noise_only_implied_pct"] = np.sqrt(
        np.maximum(piv["all"] ** 2 - piv["none"] ** 2, 0.0))
    piv.rename(columns={"none": "construction_pct", "all": "total_pct"},
               inplace=True)
    piv.to_csv(os.path.join(out_dir, "wp3_alpha_target_decomposition.csv"),
               index=False)
    return df, piv


# ===========================================================================
# Part 3 — acceptance
# ===========================================================================
def fit_one(seed, tag, cfg, out_dir=FIT_DIR, verbose=False):
    """One full `anchor_v4` fit against an armed synthetic dataset."""
    import zinc_colloc_v5 as v5
    import zinc_A_lab as Alab

    os.makedirs(out_dir, exist_ok=True)
    ds = S.load_dataset(os.path.join(SYNTH_DIR, f"{tag}.npz"))
    S.arm(ds)
    t0 = time.time()
    fit = v5.run(f"wp3_{tag}_s{seed}", **dict(cfg, seed=int(seed), verbose=verbose))
    wall = time.time() - t0
    path = os.path.join(out_dir, f"{tag}_seed{seed}.npz")
    Alab.dump_A(fit, path, stage="B")
    S.disarm()
    import jax
    jax.clear_caches()
    return path, wall


def part3_accept(tags, seeds, out_dir=OUT_DIR, fit_dir=FIT_DIR):
    """Score every completed acceptance fit against the analytic truth."""
    rows, paths_rows = [], []
    for tag in tags:
        tp = os.path.join(SYNTH_DIR, f"{tag}.npz")
        if not os.path.exists(tp):
            continue
        tr = S.load_truth(tp)
        ds = S.load_dataset(tp)
        yrs = ds["years"]
        tv = yrs <= TRAINVAL_END
        a_true = tr["alphas_point"]                  # (T, 4) point at year end
        a_win = np.vstack([np.full((1, 4), np.nan),
                           tr["alphas_window_unweighted"]])   # window mean
        a_obs = ds["alpha_obs"]
        t_true = tr["tau_sup_point"]
        for seed in seeds:
            p = os.path.join(fit_dir, f"{tag}_seed{seed}.npz")
            if not os.path.exists(p):
                continue
            d = np.load(p, allow_pickle=True)
            a_pred = d["alphas"]                     # at observed stocks, year nodes
            t_pred = np.concatenate(
                [d["taus"], d["frac_fu"][:, :2], d["frac_eu"][:, :2]], axis=1)
            for k, name in enumerate(ALPHA_NAMES):
                for split, m in (("trainval", tv), ("test", ~tv)):
                    rows.append(dict(
                        tag=tag, seed=int(seed), channel=name, split=split,
                        pred_vs_true=rel_rmse_pct(a_pred[:, k], a_true[:, k], m),
                        obs_vs_true=rel_rmse_pct(a_obs[:, k], a_true[:, k], m),
                        pred_vs_obs=rel_rmse_pct(a_pred[:, k], a_obs[:, k], m),
                        pred_vs_window=rel_rmse_pct(a_pred[:, k], a_win[:, k], m),
                        obs_vs_window=rel_rmse_pct(a_obs[:, k], a_win[:, k], m),
                        log_bias_pred=float(np.nanmedian(
                            np.log(np.maximum(a_pred[m, k], 1e-12))
                            - np.log(np.maximum(a_true[m, k], 1e-12)))),
                        log_bias_obs=float(np.nanmedian(
                            np.log(np.maximum(a_obs[m, k], 1e-12))
                            - np.log(np.maximum(a_true[m, k], 1e-12)))),
                    ))
            for j, name in enumerate(TAU_SUP_NAMES):
                rows.append(dict(
                    tag=tag, seed=int(seed), channel=name, split="trainval",
                    pred_vs_true=rel_rmse_pct(t_pred[:, j], t_true[:, j], tv),
                    obs_vs_true=rel_rmse_pct(ds["tau_sup_obs"][:, j], t_true[:, j], tv),
                    pred_vs_obs=rel_rmse_pct(t_pred[:, j], ds["tau_sup_obs"][:, j], tv),
                    pred_vs_window=np.nan, obs_vs_window=np.nan,
                    log_bias_pred=np.nan, log_bias_obs=np.nan))
            paths_rows.append(dict(tag=tag, seed=int(seed), path=p))
    per_seed = pd.DataFrame(rows)
    if per_seed.empty:
        return per_seed, per_seed
    g = per_seed.groupby(["tag", "channel", "split"])
    summ = g.agg(n_seeds=("seed", "nunique"),
                 pred_vs_true=("pred_vs_true", "median"),
                 obs_vs_true=("obs_vs_true", "median"),
                 pred_vs_obs=("pred_vs_obs", "median"),
                 pred_vs_window=("pred_vs_window", "median"),
                 obs_vs_window=("obs_vs_window", "median")).reset_index()
    lo, hi = [], []
    for (tag, ch, sp), sub in g:
        v = sub.pred_vs_true.to_numpy()
        est, l, h = hl(v)
        lo.append(l); hi.append(h)
    summ["pred_vs_true_hl_lo"], summ["pred_vs_true_hl_hi"] = lo, hi
    summ["passes_tolerance"] = np.where(
        summ.split.eq("trainval") & summ.channel.isin(ALPHA_NAMES),
        summ.pred_vs_true <= ACCEPT_TOL_PCT, np.nan)
    per_seed.to_csv(os.path.join(out_dir, "wp3_acceptance_per_seed.csv"), index=False)
    summ.to_csv(os.path.join(out_dir, "wp3_acceptance.csv"), index=False)
    return per_seed, summ


# ===========================================================================
# figures
# ===========================================================================
def figures(bias, by_window, gain, summ, sweep=None, out_dir=OUT_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- fig 1: the alpha-target bias -----------------------------------
    b = bias[bias.arm == "base"]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.1))
    for k, name in enumerate(ALPHA_NAMES):
        s = b[b.channel == name].sort_values("delta")
        ax[0].plot(s.delta, s.bias_total_pct, "o-", color=COL[k], label=name, lw=1.6)
        ax[1].plot(s.delta, s.bias_weighting_pct, "o-", color=COL[k], lw=1.6)
        ax[1].plot(s.delta, s.bias_quadrature_pct, "s--", color=COL[k], lw=1.2,
                   alpha=0.7)
    ax[0].axhline(0, color=GREY, lw=0.8)
    ax[0].set_xscale("log"); ax[0].set_xlabel(r"window width $\Delta$ (yr)")
    ax[0].set_ylabel(r"median bias of $\alpha_{\mathrm{obs}}$ (%)")
    ax[0].set_title(r"total: $\alpha_{\mathrm{obs}}$ vs $\frac{1}{\Delta}\int\alpha\,dt$")
    ax[0].legend(frameon=False, fontsize=8); ax[0].grid(alpha=0.25)
    ax[1].axhline(0, color=GREY, lw=0.8)
    ax[1].set_xscale("log"); ax[1].set_xlabel(r"window width $\Delta$ (yr)")
    ax[1].set_ylabel("component of the bias (%)")
    ax[1].set_title("solid: exposure weighting   dashed: trapezoid quadrature")
    ax[1].grid(alpha=0.25)

    w = by_window[(by_window.arm == "base") & (by_window.delta == 1.0)]
    for k, name in enumerate(ALPHA_NAMES):
        s = w[w.channel == name]
        ax[2].scatter(s.corr_FS, s.bias_total_pct, s=16, color=COL[k], label=name,
                      alpha=0.8)
    ax[2].axhline(0, color=GREY, lw=0.8)
    ax[2].set_xlabel(r"within-window $\mathrm{corr}(F, S_{\mathrm{parent}})$")
    ax[2].set_ylabel("bias of $\\alpha_{\\mathrm{obs}}$ (%)")
    ax[2].set_title(r"$\Delta = 1$ yr, per window")
    ax[2].grid(alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"wp3_alpha_target_bias.{ext}"), dpi=200)
    plt.close(fig)

    # ---- fig 2: the boxcar, theory against measurement -------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    ff = np.linspace(0.01, 4.0, 2000)
    for i, dl in enumerate((1.0, 0.25)):
        ax[0].plot(ff, sinc_gain(ff, dl), color=COL[i], lw=1.5,
                   label=rf"$|\mathrm{{sinc}}(\pi f \Delta)|$, $\Delta={dl}$ yr")
    g1 = gain[gain.delta == 1.0]
    nh = g1[~g1.harmonic]
    ax[0].scatter(nh.freq_per_yr, nh.measured_gain_rms, s=48, marker="D",
                  facecolor="none", edgecolor="k", zorder=5,
                  label=r"measured on the twin, $\Delta=1$ yr")
    FLOOR = 1e-17
    hm = g1[g1.harmonic]
    ax[0].scatter(hm.freq_per_yr, np.full(len(hm), FLOOR * 3), s=60, marker="v",
                  color="k", zorder=5,
                  label="harmonics: measured at machine zero")
    ax[0].set_yscale("log"); ax[0].set_ylim(FLOOR, 2)
    ax[0].set_xlabel("frequency (1/yr)"); ax[0].set_ylabel("gain")
    ax[0].set_title("period integration is a boxcar with exact nulls at $f=k/\\Delta$")
    ax[0].legend(frameon=False, fontsize=8, loc="lower left"); ax[0].grid(alpha=0.25)

    x = np.arange(len(g1))
    ax[1].bar(x - 0.2, np.maximum(g1.theory_sinc, FLOOR), 0.38, color=COL[0],
              label="closed form $|\\mathrm{sinc}(\\pi f \\Delta)|$")
    ax[1].bar(x + 0.2, np.maximum(g1.measured_gain_rms, FLOOR), 0.38, color=COL[2],
              label="measured on the twin")
    ax[1].set_yscale("log"); ax[1].set_ylim(FLOOR, 2)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([f"{f:g}/yr" + ("\n(harmonic)" if h else "")
                           for f, h in zip(g1.freq_per_yr, g1.harmonic)],
                          fontsize=8)
    ax[1].set_ylabel("gain at $\\Delta = 1$ yr"); ax[1].grid(alpha=0.25)
    ax[1].legend(frameon=False, fontsize=8, loc="lower left")
    for xi, (_, r) in zip(x, g1.iterrows()):
        if r.harmonic:
            ax[1].annotate("annihilated", (xi, 1e-14), ha="center", fontsize=8,
                           rotation=90, va="bottom", color=GREY)
    ax[1].set_title("harmonics are annihilated, not aliased")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"wp3_boxcar_gain.{ext}"), dpi=200)
    plt.close(fig)

    # ---- fig 3: acceptance ----------------------------------------------
    if summ is not None and not summ.empty:
        a = summ[summ.channel.isin(ALPHA_NAMES) & summ.split.eq("trainval")]
        tags = sorted(a.tag.unique())
        fig, ax = plt.subplots(1, len(tags), figsize=(5.6 * len(tags), 4.0),
                               squeeze=False)
        for ti, tag in enumerate(tags):
            s = a[a.tag == tag].set_index("channel").loc[ALPHA_NAMES]
            x = np.arange(4)
            ax[0][ti].bar(x - 0.26, s.pred_vs_true, 0.25, color=COL[0],
                          label=r"$\alpha_{\mathrm{pred}}$ vs $\alpha_{\mathrm{true}}$")
            ax[0][ti].bar(x, s.obs_vs_true, 0.25, color=COL[2],
                          label=r"$\alpha_{\mathrm{obs}}$ vs $\alpha_{\mathrm{true}}$")
            ax[0][ti].bar(x + 0.26, s.pred_vs_obs, 0.25, color=COL[1],
                          label=r"$\alpha_{\mathrm{pred}}$ vs $\alpha_{\mathrm{obs}}$")
            ax[0][ti].axhline(ACCEPT_TOL_PCT, color=GREY, ls="--", lw=1.2,
                              label=f"tolerance {ACCEPT_TOL_PCT:g}%")
            ax[0][ti].set_xticks(x); ax[0][ti].set_xticklabels(ALPHA_NAMES, rotation=20)
            ax[0][ti].set_ylabel("relRMSE (%), trainval")
            ax[0][ti].set_title(tag)
            ax[0][ti].grid(alpha=0.25, axis="y")
            if ti == 0:
                ax[0][ti].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out_dir, f"wp3_acceptance.{ext}"), dpi=200)
        plt.close(fig)

    # ---- fig 3b: how much sub-annual structure would it take? ------------
    if sweep is not None and not sweep.empty:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
        for k, name in enumerate(ALPHA_NAMES):
            s0 = sweep[sweep.channel == name].sort_values("amplitude_nats")
            ax[0].plot(s0.peak_to_trough_pct, s0.relRMSE_obs_vs_true_pct, "o-",
                       color=COL[k], lw=1.6, label=name)
            ax[1].plot(s0.peak_to_trough_pct, s0.corr_FS_median, "o-",
                       color=COL[k], lw=1.6)
        ax[0].set_xlabel("implied peak-to-trough swing of $\\alpha$ within a year (%)")
        ax[0].set_ylabel(r"relRMSE($\alpha_{\mathrm{obs}}$, $\alpha_{\mathrm{true}}$) (%)")
        ax[0].set_title(f"target-construction error at $\\Delta=1$ yr, "
                        f"$f = {SWEEP_FREQ:g}$/yr")
        ax[0].legend(frameon=False, fontsize=8); ax[0].grid(alpha=0.25)
        ax[1].set_xlabel("implied peak-to-trough swing of $\\alpha$ within a year (%)")
        ax[1].set_ylabel(r"within-window corr($F$, $S_{\mathrm{parent}}$)")
        ax[1].grid(alpha=0.25)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out_dir, f"wp3_bias_amplitude_sweep.{ext}"), dpi=200)
        plt.close(fig)

    # ---- fig 4: the twin itself ------------------------------------------
    fig, ax = plt.subplots(2, 4, figsize=(15, 6.4))
    d = np.load(os.path.join(SYNTH_DIR, "base_dense.npz"), allow_pickle=True)
    dsea = np.load(os.path.join(SYNTH_DIR, "season_dense.npz"), allow_pickle=True)
    ctx_real = np.load(os.path.join(SYNTH_DIR, "base_d1y_clean.npz"), allow_pickle=True)
    for k, name in enumerate(["Concentrate", "Refined", "In-Use", "Scrap"]):
        ax[0][k].plot(d["t"], d["S4"][:, k], color=COL[0], lw=1.4, label="twin")
        ax[0][k].plot(ctx_real["years"], ctx_real["stocks_obs"][:, k], "o",
                      ms=3, color=COL[2], label="annual sample")
        ax[0][k].set_title(name); ax[0][k].grid(alpha=0.25)
        if k == 0:
            ax[0][k].legend(frameon=False, fontsize=8)
        ax[0][k].set_ylabel("kt")
    for k, name in enumerate(ALPHA_NAMES):
        ax[1][k].plot(d["t"], d["alphas"][:, k], color=COL[0], lw=1.3, label="base")
        if name == "alpha_refc":
            ax[1][k].plot(dsea["t"], dsea["alphas"][:, k], color=COL[4], lw=0.7,
                          alpha=0.75, label="season")
        tr = S.load_truth(os.path.join(SYNTH_DIR, "base_d1y_clean.npz"))
        ax[1][k].plot(tr["window_years"], tr["alphas_window_obs"][:, k], "s", ms=3,
                      color=COL[2], label=r"$\alpha_{\mathrm{obs}}$")
        ax[1][k].set_title(name); ax[1][k].grid(alpha=0.25)
        ax[1][k].set_xlabel("year"); ax[1][k].set_ylabel("1/yr")
        if k == 1:
            ax[1][k].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"wp3_twin.{ext}"), dpi=200)
    plt.close(fig)


# ===========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="WP-3 driver")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--operators", action="store_true")
    ap.add_argument("--bias", action="store_true")
    ap.add_argument("--fit", action="store_true", help="run the acceptance fits")
    ap.add_argument("--accept", action="store_true", help="score existing fits")
    ap.add_argument("--tags", default="base_d1y_clean,base_d1y_noisy")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args(argv)

    import zinc_colloc_v5 as v5
    S.integrity_check()
    S.install(v5)
    if args.check:
        S.check()
        return 0

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if args.fit:
        cfg = S.load_anchor_config()
        for tag in tags:
            for seed in seeds:
                p = os.path.join(FIT_DIR, f"{tag}_seed{seed}.npz")
                if os.path.exists(p):
                    print(f"[fit] {tag} seed {seed} exists, skipping", flush=True)
                    continue
                _, wall = fit_one(seed, tag, cfg)
                print(f"[fit] {tag} seed {seed}  {wall:6.1f}s  "
                      f"rss {S._rss_mb():.0f} MB", flush=True)

    ver = gain = bias = by_window = summ = sweep = noise = None
    if args.operators or args.figures:
        ver, gain = part1_operators()
        print("\n[Part 1] operator checks (max over arms):")
        print(ver.groupby("check").value.max().to_string())
        print("\n[Part 1] boxcar gain at Delta = 1 yr:")
        print(gain[gain.delta == 1.0].to_string(index=False))
    if args.bias or args.figures:
        bias, by_window = part2_bias()
        print("\n[Part 2] alpha-target bias, base arm (median %, per window):")
        print(bias[bias.arm == "base"].pivot_table(
            index="channel", columns="delta", values="bias_total_pct").round(2).to_string())
        print("\n[Part 2] at Delta = 1 yr, decomposition:")
        print(bias[(bias.arm == "base") & (bias.delta == 1.0)][
            ["channel", "bias_total_pct", "bias_weighting_pct",
             "bias_quadrature_pct", "bias_alignment_pct", "corr_FS_median",
             "relRMSE_obs_vs_true_pct", "relRMSE_obs_vs_point_pct"]]
            .round(3).to_string(index=False))
    if args.sweep or args.figures:
        p_sweep = os.path.join(OUT_DIR, "wp3_bias_amplitude_sweep.csv")
        if args.sweep or not os.path.exists(p_sweep):
            sweep = part2b_amplitude_sweep()
        else:
            sweep = pd.read_csv(p_sweep)
        print("\n[Part 2b] target-construction error vs sub-annual amplitude:")
        print(sweep.pivot_table(index="peak_to_trough_pct", columns="channel",
                                values="relRMSE_obs_vs_true_pct").round(2).to_string())
    if args.noise or args.figures:
        p_noise = os.path.join(OUT_DIR, "wp3_alpha_target_noise.csv")
        if args.noise or not os.path.exists(p_noise):
            noise, dec = part2c_noise()
        else:
            noise = pd.read_csv(p_noise)
            dec = pd.read_csv(os.path.join(
                OUT_DIR, "wp3_alpha_target_decomposition.csv"))
        print("\n[Part 2c] alpha-target error, construction vs observation "
              "noise (Delta = 1 yr):")
        print(dec[(dec.arm == "base") & (dec.delta == 1.0)][
            ["channel", "construction_pct", "total_pct",
             "noise_only_implied_pct"]].round(2).to_string(index=False))
        print("\n[Part 2c] which observed series the noise enters through "
              "(base, Delta = 1 yr):")
        print(noise[(noise.arm == "base") & (noise.delta == 1.0)].pivot_table(
            index="channel", columns="noise_group",
            values="relRMSE_median_pct").round(2).to_string())
    if args.accept or args.figures:
        _, summ = part3_accept(tags, seeds)
        if summ is not None and not summ.empty:
            print("\n[Part 3] acceptance, trainval, alpha channels:")
            print(summ[summ.channel.isin(ALPHA_NAMES) & summ.split.eq("trainval")][
                ["tag", "channel", "n_seeds", "pred_vs_true", "obs_vs_true",
                 "pred_vs_obs", "pred_vs_window", "obs_vs_window",
                 "passes_tolerance"]].round(2).to_string(index=False))
        else:
            print("\n[Part 3] no acceptance fits on disk yet (run --fit).")
    if args.figures:
        figures(bias, by_window, gain, summ, sweep)
        print("\n[figures] written to analysis/wp3_*.{png,pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
