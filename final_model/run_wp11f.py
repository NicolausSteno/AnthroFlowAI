#!/usr/bin/env python3
"""
run_wp11f.py — WP-11f: attenuation bias magnitude on the real estimates
=======================================================================

The question: for a true sub-annual component of amplitude `A` at frequency
`f`, how much bias does annual observation induce in the estimated alpha?
The spec asks for a bias surface over (amplitude, frequency) per channel with
the two mechanisms separated, and then for `A` to be bounded so the surface
becomes a statement about the published estimates.

The two mechanisms, and a third that turns out to matter
--------------------------------------------------------
1. **Attenuation.**  Period integration is a boxcar, transfer
   `|sinc(pi f Delta)|`, exact nulls at `f = k/Delta`.  For a *log*-additive
   oscillation this is not only lost amplitude: the window mean of
   `A exp(a sin)` is `A I_0(a)`, so an oscillating coefficient is reported at
   a level biased **upward** by `log I_0(a) ~ a^2/4` regardless of frequency.
   That is a bias, not a variance, and it does not average out over years.
2. **Target construction.**  `alpha_obs = F_int / trapz(S)` is an integral
   over an interpolated point stock.  WP-3 established the exact
   decomposition into a within-window `Cov_w(alpha, S)/mean_w(S)` term and a
   trapezoid quadrature term; both are re-measured here as functions of
   `(f, a)` rather than at one operating point.
3. **Placement.**  `_build_empirical_alphas` stores the window average at the
   *closing* year and the model is supervised at that node, so what the
   estimate is biased *relative to* is the instantaneous truth at an integer
   year.  For a component at exactly `k/yr` the node phase is the same every
   year, so the whole thing collapses to a constant log offset — the worst
   case, and the one annual data cannot detect.

The bound on `A`, and why it is one-sided
-----------------------------------------
An annual record does constrain sub-annual amplitude at *non-harmonic*
frequencies: a component at `f` reaches the annual series attenuated by
`|sinc(pi f)|` and folded to `|f - round(f)|`, so the high-frequency content
that is actually present in the annual `alpha_obs` is an upper bound on
`A |sinc(pi f)|`.  At `f = k/yr` that bound is **vacuous**, because the gain
is exactly zero: annual data carry no information whatever about the
amplitude of an annual-harmonic oscillation, while that is precisely the
component that produces a pure level bias in alpha.  This is the sharpest
form of the argument for sub-annual collection and is Part 5's output.

    python run_wp11f.py --check
    python run_wp11f.py --closed-form --surface
    python run_wp11f.py --transfer --bound --statement --figures
    python run_wp11f.py --all
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

import zinc_market_lab as M
import zinc_synth_lab as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "analysis")
SYNTH_DIR = os.path.join(OUT_DIR, "synth")
FIT_DIR = os.path.join(SYNTH_DIR, "fits")
WEIGHTS_DIR = os.path.join(OUT_DIR, "wp2a")

COL = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00", "#56B4E9"]
GREY = "#555555"

ALPHA_NAMES = S.ALPHA_NAMES
DELTA = 1.0
TRAINVAL_END = 2007.0

# Minimum spread of target log-bias across the available fits before a
# target-to-estimate transfer slope is regarded as identified.
TRANSFER_MIN_RANGE = 0.05      # nats, i.e. a 5% spread in the target's bias

# The surface grid.  Amplitudes in nats of log-alpha; the peak-to-trough swing
# `exp(2a) - 1` is carried alongside because that is the quantity whose
# plausibility a reviewer can judge.
FREQS = M.BIAS_FREQS
AMPS = (0.05, 0.10, 0.20, 0.40)
PHASES = M.BIAS_PHASES


def hl(v):
    """Hodges-Lehmann estimate and 95% interval; NaN interval below six."""
    import zinc_circ_lab as C
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    d = C.hodges_lehmann(v)
    return (float(d.get("estimate", np.median(v))),
            float(d.get("lo", np.nan)), float(d.get("hi", np.nan)))


# ===========================================================================
# Part 1 — closed forms
# ===========================================================================
def part1_closed_form(out_dir=OUT_DIR):
    """The two frequency-domain facts, tabulated and verified numerically.

    `boxcar_gain` is what survives period integration; `jensen_log_pct` is the
    level bias an oscillation of amplitude `a` puts into a window-averaged
    coefficient, independent of frequency; `phase_swing_pct` is the extra
    year-to-year dispersion `+-a` that the placement convention adds at
    non-harmonic frequencies and *collapses into a constant* at harmonic ones.
    """
    rows = []
    for f in FREQS:
        g = float(M.sinc_gain(f, DELTA))
        fa = float(M.alias_freq(f, DELTA))
        harm = abs(f * DELTA - round(f * DELTA)) < 1e-9
        for a in AMPS:
            rows.append(dict(
                f_per_yr=float(f), amplitude_nats=float(a),
                peak_to_trough_pct=100.0 * (np.exp(2 * a) - 1.0),
                harmonic_of_year=bool(harm),
                boxcar_gain=g,
                alias_f_per_yr=fa,
                surviving_amp_nats=a * g,
                jensen_log_pct=100.0 * float(M.jensen_log_bias(a)),
                phase_swing_pct=100.0 * a,
                # what annual data can say about `a`: nothing at a harmonic
                identifiable_from_annual=bool(g > 1e-6)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "wp11f_closed_form.csv"), index=False)
    return df


# ===========================================================================
# Part 2 — the measured bias surface
# ===========================================================================
def part2_surface(ctx, freqs=FREQS, amps=AMPS, phases=PHASES, out_dir=OUT_DIR,
                  verbose=True):
    """`M.bias_cell` over the whole grid, decomposed and phase-enveloped.

    All four channels are driven in one solve (as `run_wp3.part2b_amplitude_
    sweep` does); the shared-trajectory contamination that buys is bounded in
    `zinc_market_lab.check()` and reported in the findings note rather than
    silently absorbed.

    Everything is in **log units**, so the three mechanisms add exactly:
    `total = attenuation + covariance + quadrature`.  Percentages are
    `100 (exp(x) - 1)` so they read as multiplicative errors on alpha.
    """
    rows = []
    n = len(freqs) * len(amps) * len(phases)
    i = 0
    t0 = time.time()
    for f in freqs:
        for a in amps:
            for ph in phases:
                cells, _ = M.bias_cell(ctx, freq=f, amp=a, phase=ph,
                                       delta=DELTA)
                i += 1
                for name, c in cells.items():
                    yrs = c["years"]
                    tv = yrs <= TRAINVAL_END
                    row = dict(channel=name, f_per_yr=float(f),
                               amplitude_nats=float(a), phase_rad=float(ph),
                               peak_to_trough_pct=100.0 * (np.exp(2 * a) - 1.0),
                               harmonic_of_year=bool(
                                   abs(f * DELTA - round(f * DELTA)) < 1e-9),
                               boxcar_gain=float(M.sinc_gain(f, DELTA)))
                    for k in ("attenuation", "covariance", "quadrature", "total"):
                        v = np.asarray(c[k], float)[tv]
                        row[f"{k}_median_pct"] = 100.0 * (np.exp(np.median(v)) - 1.0)
                        row[f"{k}_rms_pct"] = 100.0 * (
                            np.exp(np.sqrt(np.mean(v ** 2))) - 1.0)
                    rows.append(row)
                if verbose:
                    print(f"  [{i:3d}/{n}] f={f:5.2f} a={a:4.2f} ph={ph:4.2f}  "
                          f"{time.time() - t0:6.0f}s", flush=True)
                import jax
                jax.clear_caches()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "wp11f_surface_per_phase.csv"), index=False)

    # phase envelope: the median over phases and the worst phase, per cell
    g = df.groupby(["channel", "f_per_yr", "amplitude_nats"])
    env = g.agg(n_phase=("phase_rad", "nunique"),
                peak_to_trough_pct=("peak_to_trough_pct", "first"),
                harmonic_of_year=("harmonic_of_year", "first"),
                boxcar_gain=("boxcar_gain", "first"),
                attenuation_pct=("attenuation_median_pct", "median"),
                covariance_pct=("covariance_median_pct", "median"),
                quadrature_pct=("quadrature_median_pct", "median"),
                total_pct=("total_median_pct", "median"),
                total_pct_lo=("total_median_pct", "min"),
                total_pct_hi=("total_median_pct", "max"),
                total_rms_pct=("total_rms_pct", "median")).reset_index()
    env["total_abs_worst_pct"] = env[["total_pct_lo", "total_pct_hi"]].abs().max(axis=1)
    env["attenuation_share"] = (env.attenuation_pct.abs()
                                / env[["attenuation_pct", "covariance_pct",
                                       "quadrature_pct"]].abs().sum(axis=1).replace(0, np.nan))
    env.to_csv(os.path.join(out_dir, "wp11f_attenuation_bias.csv"), index=False)
    return df, env


# ===========================================================================
# Part 3 — from a biased target to a biased estimate
# ===========================================================================
def part3_transfer(out_dir=OUT_DIR, fit_dir=FIT_DIR):
    """How much of a log bias in `alpha_obs` reaches the fitted `alpha_hat`?

    The surface in Part 2 is a bias in the *target*.  What the paper has to
    state is a bias in the *estimate*.  The two are linked by however hard
    the objective pulls `alpha_hat` onto `alpha_obs`, which is measurable on
    every twin fit that exists: per channel and seed, the median log gap of
    the target from the truth at the supervised node, and the median log gap
    of the fit from the same truth.  A slope of 1 means the bias passes
    straight through.

    Uses whatever fits are on disk — WP-3's `base` acceptance arm, where the
    target bias is small, and WP-11e's market arms, where it is not.
    """
    rows = []
    tags = []
    for p in sorted(glob.glob(os.path.join(SYNTH_DIR, "*_d1y_clean.npz"))):
        tags.append(os.path.basename(p)[:-4])
    for tag in tags:
        tp = os.path.join(SYNTH_DIR, f"{tag}.npz")
        tr = S.load_truth(tp)
        ds = S.load_dataset(tp)
        yrs = np.asarray(ds["years"], float)
        tv = yrs <= TRAINVAL_END
        a_true = np.asarray(tr["alphas_point"], float)
        a_obs = np.asarray(ds["alpha_obs"], float)
        for p in sorted(glob.glob(os.path.join(fit_dir, f"{tag}_seed*.npz"))):
            d = np.load(p, allow_pickle=True)
            if "alphas" not in d.files:
                continue
            a_hat = np.asarray(d["alphas"], float)
            for k, name in enumerate(ALPHA_NAMES):
                lo = np.log(np.maximum(a_obs[:, k], 1e-30)) - np.log(np.maximum(a_true[:, k], 1e-30))
                lh = np.log(np.maximum(a_hat[:, k], 1e-30)) - np.log(np.maximum(a_true[:, k], 1e-30))
                m = tv & np.isfinite(lo) & np.isfinite(lh)
                rows.append(dict(
                    tag=tag, seed=int(os.path.basename(p).split("seed")[1][:-4]),
                    channel=name,
                    log_bias_obs=float(np.median(lo[m])),
                    log_bias_fit=float(np.median(lh[m])),
                    n=int(m.sum())))
    df = pd.DataFrame(rows)
    if df.empty:
        return df, df
    df.to_csv(os.path.join(out_dir, "wp11f_transfer_per_seed.csv"), index=False)
    out = []
    for ch, sub in df.groupby("channel"):
        x = sub.log_bias_obs.to_numpy(); y = sub.log_bias_fit.to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        # A slope is only meaningful if the fits on disk actually span a range
        # of target bias.  They do not: every arm generated so far (`base`,
        # `state`, `mkt_ref`) has a near-unbiased target, so the regression
        # would be fitted on noise.  Below this range the slope is refused and
        # the prior value of 1 is used instead — the objective supervises
        # `log alpha` on `log alpha_obs`, so a constant log offset in the
        # target passes straight through to any estimate that fits its target.
        identified = bool(m.sum() >= 6 and np.ptp(x[m]) >= TRANSFER_MIN_RANGE)
        if identified:
            A = np.column_stack([np.ones(m.sum()), x[m]])
            beta, *_ = np.linalg.lstsq(A, y[m], rcond=None)
            r = y[m] - A @ beta
            r2 = 1.0 - r.var() / max(y[m].var(), 1e-30)
            slope, inter = float(beta[1]), float(beta[0])
        else:
            slope = inter = r2 = np.nan
        est, lo, hi = hl(y[m] - x[m])
        out.append(dict(channel=ch, n_fits=int(m.sum()),
                        transfer_identified=identified,
                        slope=slope, intercept=inter, r2=r2,
                        obs_bias_range_nats=float(np.ptp(x[m])) if m.sum() else np.nan,
                        excess_fit_over_obs_hl=est, excess_lo=lo, excess_hi=hi,
                        tags=",".join(sorted(sub.tag.unique()))))
    summ = pd.DataFrame(out)
    summ.to_csv(os.path.join(out_dir, "wp11f_transfer.csv"), index=False)
    return df, summ


# ===========================================================================
# Part 4 — bounding A from the real record
# ===========================================================================
def _smooth_resid(y, k=5):
    """Residual of a centred moving average of width `k` — a crude high-pass
    that assumes nothing about the model.  `k=5` keeps everything with a
    period under about 5 yr, which is where an alias of a sub-annual
    component lands."""
    y = np.asarray(y, float)
    n = y.size
    out = np.full(n, np.nan)
    h = k // 2
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1)
        w = y[lo:hi]
        w = w[np.isfinite(w)]
        out[i] = y[i] - w.mean() if w.size else np.nan
    return out


def part4_bound(cfg, out_dir=OUT_DIR, weights_dir=WEIGHTS_DIR):
    """Upper bound on the true sub-annual amplitude from the annual record.

    Two independent high-frequency budgets on the real `log alpha_obs`:

      `resid_vs_fit`    against the published 35-seed `alpha_hat`.  Tight, but
                        the network can absorb a low-frequency alias, so it
                        under-states what a sub-annual component could be.
      `resid_vs_smooth` against a 5-yr centred moving average.  Model-free.

    Either way the bound is `A <= sqrt(2) rms / |sinc(pi f)|` and it diverges
    at every annual harmonic, which is the point of the exercise.
    """
    import zinc_colloc_v5 as v5

    S.disarm()
    loader = S._ORIG_LOADER or v5.load_zinc_data
    raw = loader(cfg["xlsx_path"], extra_exog_cols=cfg.get("extra_exog_cols"))
    yrs = np.asarray(raw["years"], float).ravel()
    a_obs = np.asarray(raw["alpha_obs"], float)
    tv = yrs <= TRAINVAL_END

    A_hat = []
    for p in sorted(glob.glob(os.path.join(weights_dir, "A_seed*.npz"))):
        d = np.load(p, allow_pickle=True)
        if "alphas" in d.files:
            A_hat.append(np.asarray(d["alphas"], float))
    A_hat = np.asarray(A_hat) if A_hat else None

    rows = []
    for k, name in enumerate(ALPHA_NAMES):
        lo = np.log(np.maximum(a_obs[:, k], 1e-30))
        m = tv & np.isfinite(lo)
        rms_smooth = float(np.nanstd(_smooth_resid(np.where(m, lo, np.nan))[m]))
        if A_hat is not None:
            r = [float(np.nanstd((lo - np.log(np.maximum(A_hat[s][:, k], 1e-30)))[m]))
                 for s in range(A_hat.shape[0])]
            rms_fit = float(np.median(r))
            n_seeds = int(A_hat.shape[0])
        else:
            rms_fit, n_seeds = np.nan, 0
        for f in FREQS:
            g = float(M.sinc_gain(f, DELTA))
            rows.append(dict(
                channel=name, f_per_yr=float(f), boxcar_gain=g,
                n_seeds=n_seeds,
                rms_log_resid_vs_fit=rms_fit,
                rms_log_resid_vs_smooth=rms_smooth,
                A_bound_vs_fit_nats=(np.sqrt(2.0) * rms_fit / g) if g > 1e-9 else np.inf,
                A_bound_vs_smooth_nats=(np.sqrt(2.0) * rms_smooth / g) if g > 1e-9 else np.inf,
                identifiable_from_annual=bool(g > 1e-6)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "wp11f_amplitude_bound.csv"), index=False)
    return df


# ===========================================================================
# Part 5 — the deliverable statement
# ===========================================================================
def part5_statement(env, bound, transfer, out_dir=OUT_DIR):
    """The spec's named sentence, per channel, with its own caveat attached.

    For each channel: take the amplitude the annual record still permits at
    the most informative non-harmonic frequency on the grid, read the bias off
    the surface, multiply by the measured target-to-estimate transfer, and
    report the phase envelope.  Then repeat at the annual harmonic, where the
    permitted amplitude is unbounded and only the surface has anything to say.
    """
    rows = []
    tr = transfer.set_index("channel") if (transfer is not None
                                           and not transfer.empty) else None
    for name in ALPHA_NAMES:
        b = bound[(bound.channel == name) & bound.identifiable_from_annual]
        if b.empty:
            continue
        # the tightest bound the annual record gives, over non-harmonic f
        i = b.A_bound_vs_smooth_nats.idxmin()
        f_tight = float(b.loc[i, "f_per_yr"])
        a_tight = float(b.loc[i, "A_bound_vs_smooth_nats"])
        ident = (tr is not None and name in tr.index
                 and bool(tr.loc[name, "transfer_identified"]))
        slope = float(tr.loc[name, "slope"]) if ident else 1.0
        for label, f_at, a_at in (("bounded_non_harmonic", f_tight, a_tight),
                                  ("annual_harmonic_unbounded", 1.0, a_tight)):
            sub = env[(env.channel == name) & (np.abs(env.f_per_yr - f_at) < 1e-9)]
            if sub.empty:
                continue
            # the largest grid amplitude the record still permits — never one
            # above the bound, so the reported bias is not extrapolated past
            # what the data allow
            a_grid = np.sort(sub.amplitude_nats.unique())
            ok = a_grid[a_grid <= a_at]
            a_use = float(ok[-1]) if ok.size else float(a_grid[0])
            r = sub[np.abs(sub.amplitude_nats - a_use) < 1e-12].iloc[0]
            rows.append(dict(
                channel=name, case=label, f_per_yr=f_at,
                amplitude_used_nats=float(r.amplitude_nats),
                amplitude_permitted_nats=a_at,
                peak_to_trough_pct=float(r.peak_to_trough_pct),
                bias_target_pct=float(r.total_pct),
                bias_target_worst_pct=float(r.total_abs_worst_pct),
                transfer_slope=slope,
                transfer_identified=bool(ident),
                bias_estimate_pct=float(r.total_pct) * slope,
                bias_estimate_worst_pct=float(r.total_abs_worst_pct) * slope,
                # the deliverable range: the phase of a real sub-annual
                # component is unknown, so the honest statement is an
                # interval over it, not a point
                bias_estimate_lo_pct=min(abs(float(r.total_pct_lo)),
                                         abs(float(r.total_pct_hi))) * abs(slope),
                bias_estimate_hi_pct=float(r.total_abs_worst_pct) * abs(slope),
                attenuation_pct=float(r.attenuation_pct),
                covariance_pct=float(r.covariance_pct),
                quadrature_pct=float(r.quadrature_pct)))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.to_csv(os.path.join(out_dir, "wp11f_statements.csv"), index=False)

    lines = ["# WP-11f — the attenuation-bias statement, per channel", "",
             "Generated by `run_wp11f.py`.  Percentages are multiplicative "
             "errors on alpha; the worst-phase column is the envelope over "
             f"{len(PHASES)} phases of the injected oscillation.", ""]
    for _, r in df.iterrows():
        if r.case == "bounded_non_harmonic":
            lines.append(
                f"- **{r.channel}** — the annual record permits a sub-annual "
                f"swing of at most {100*(np.exp(2*r.amplitude_permitted_nats)-1):.0f}% "
                f"peak-to-trough at {r.f_per_yr:.2f}/yr.  At the largest grid "
                f"amplitude within that bound ({r.peak_to_trough_pct:.0f}% swing) the "
                f"annual target is biased by "
                f"{r.bias_estimate_lo_pct:.1f}-{r.bias_estimate_hi_pct:.1f}% "
                f"depending on the unknown phase of the oscillation"
                + ("" if r.transfer_identified else
                   " (target-to-estimate transfer taken as 1; not identified "
                   "from the fits on disk)") + ".")
        else:
            lines.append(
                f"- **{r.channel}** — at exactly 1/yr the annual record "
                f"constrains the amplitude **not at all** (boxcar gain 0). "
                f"A {r.peak_to_trough_pct:.0f}% within-year swing at that "
                f"frequency would bias the estimate by "
                f"{r.bias_estimate_lo_pct:.1f}-{r.bias_estimate_hi_pct:.1f}%, "
                f"undetectably.")
    with open(os.path.join(out_dir, "wp11f_statements.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return df


# ===========================================================================
# figures
# ===========================================================================
def figures(env, bound, out_dir=OUT_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11.0, 8.0))

    # (a) total bias against frequency, one line per amplitude, alpha_win
    a0 = ax[0, 0]
    sub = env[env.channel == "alpha_win"]
    for i, a in enumerate(sorted(sub.amplitude_nats.unique())):
        s = sub[sub.amplitude_nats == a].sort_values("f_per_yr")
        a0.plot(s.f_per_yr, s.total_pct, "o-", color=COL[i % len(COL)], lw=1.2,
                ms=3, label=f"a = {a:.2f} nats "
                            f"({100*(np.exp(2*a)-1):.0f}% swing)")
        a0.fill_between(s.f_per_yr, s.total_pct_lo, s.total_pct_hi,
                        color=COL[i % len(COL)], alpha=0.12, lw=0)
    for h in (1.0, 2.0, 3.0):
        a0.axvline(h, color=GREY, ls=":", lw=0.8)
    a0.axhline(0.0, color=GREY, lw=0.6)
    a0.set_xlabel("frequency [1/yr]")
    a0.set_ylabel(r"bias in annual $\alpha$ target [%]")
    a0.set_title(r"(a) $\alpha_{\rm win}$: bias surface, phase envelope shaded")
    a0.legend(fontsize=7, frameon=False)

    # (b) mechanism decomposition at the largest amplitude
    a1 = ax[0, 1]
    amax = max(env.amplitude_nats.unique())
    w = 0.2
    xs = np.arange(len(ALPHA_NAMES))
    for i, (k, lab) in enumerate((("attenuation_pct", "attenuation + Jensen"),
                                  ("covariance_pct", r"$Cov_w(\alpha,S)$"),
                                  ("quadrature_pct", "trapezoid quadrature"))):
        v = [env[(env.channel == c) & (env.amplitude_nats == amax)
                 & (np.abs(env.f_per_yr - 1.35) < 1e-9)][k].median()
             for c in ALPHA_NAMES]
        a1.bar(xs + (i - 1) * w, v, width=w, color=COL[i], label=lab)
    a1.axhline(0.0, color=GREY, lw=0.6)
    a1.set_xticks(xs); a1.set_xticklabels(ALPHA_NAMES, rotation=20, fontsize=7)
    a1.set_ylabel("bias [%]")
    a1.set_title(f"(b) mechanisms at f = 1.35/yr, a = {amax:.2f}")
    a1.legend(fontsize=7, frameon=False)

    # (c) the boxcar and the bound it implies
    a2 = ax[1, 0]
    f = np.linspace(0.05, 5.0, 2001)
    a2.semilogy(f, M.sinc_gain(f, 1.0), color=COL[0], lw=1.2)
    a2.set_xlabel("frequency [1/yr]")
    a2.set_ylabel(r"$|\mathrm{sinc}(\pi f \Delta)|$")
    a2.set_ylim(1e-4, 2.0)
    for h in (1.0, 2.0, 3.0, 4.0):
        a2.axvline(h, color=GREY, ls=":", lw=0.8)
    a2.set_title("(c) annual data see nothing at the harmonics")

    # (d) permitted amplitude per channel
    a3 = ax[1, 1]
    for i, c in enumerate(ALPHA_NAMES):
        s = bound[(bound.channel == c) & bound.identifiable_from_annual].sort_values("f_per_yr")
        a3.semilogy(s.f_per_yr, 100.0 * (np.exp(2 * s.A_bound_vs_smooth_nats) - 1.0),
                    "o-", color=COL[i], lw=1.0, ms=3, label=c)
    a3.axhline(100.0, color=GREY, ls=":", lw=0.9)
    a3.text(a3.get_xlim()[0], 115.0, "above here the bound is vacuous",
            fontsize=7, color=GREY)
    a3.set_ylim(8.0, 2.0e3)
    a3.set_xlabel("frequency [1/yr]")
    a3.set_ylabel("permitted peak-to-trough swing [%]")
    a3.set_title("(d) what the annual record still allows")
    a3.legend(fontsize=7, frameon=False, loc="lower right")

    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"wp11f_attenuation_bias.{e}"), dpi=200)
    plt.close(fig)


# ===========================================================================
# CLI
# ===========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--closed-form", action="store_true")
    ap.add_argument("--surface", action="store_true")
    ap.add_argument("--transfer", action="store_true")
    ap.add_argument("--bound", action="store_true")
    ap.add_argument("--statement", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args(argv)

    import zinc_colloc_v5 as v5
    M.install(v5)
    cfg = M.load_anchor_config()
    ctx = S.driver_context(cfg)
    n_ex = len(ctx["exog_cols"])
    orders = len(cfg.get("exog_feature_orders", (0, 1)))
    print(f"[wp11f] drivers: {list(S.DRIVER_ALIASES.values())}")
    print(f"[wp11f] exog columns = {n_ex}, feature orders = {orders}, "
          f"input_dim = {1 + 4 + n_ex * orders}  "
          f"(CLAUDE.md rule 2 says 23 — SCHEMA §9 flag 1, open)")

    if a.check:
        M.check(verbose=True)
        return 0

    env = bound = tsum = None
    if a.closed_form or a.all:
        cf = part1_closed_form()
        print(cf.head(12).to_string(index=False))
    if a.surface or a.all:
        _, env = part2_surface(ctx)
    if a.transfer or a.all:
        _, tsum = part3_transfer()
        if tsum is not None and not tsum.empty:
            print(tsum.to_string(index=False))
    if a.bound or a.all:
        bound = part4_bound(cfg)
        print(bound[bound.identifiable_from_annual][
            ["channel", "f_per_yr", "A_bound_vs_smooth_nats"]].to_string(index=False))
    if a.statement or a.figures or a.all:
        env = env if env is not None else pd.read_csv(
            os.path.join(OUT_DIR, "wp11f_attenuation_bias.csv"))
        bound = bound if bound is not None else pd.read_csv(
            os.path.join(OUT_DIR, "wp11f_amplitude_bound.csv"))
        tp = os.path.join(OUT_DIR, "wp11f_transfer.csv")
        tsum = tsum if tsum is not None else (
            pd.read_csv(tp) if os.path.exists(tp) else None)
    if a.statement or a.all:
        st = part5_statement(env, bound, tsum)
        if not st.empty:
            print(st.to_string(index=False))
    if a.figures or a.all:
        figures(env, bound)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
