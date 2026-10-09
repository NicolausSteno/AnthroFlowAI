#!/usr/bin/env python3
"""
run_wp11e.py — WP-11e: sub-annual dynamics with market relevance (Ch. 3 link)
============================================================================

The spec asks two questions of a Chapter-3-style lead-time oscillation
embedded in the synthetic truth:

  Q1  does annual sampling destroy it entirely, or does it survive in
      aliased form?
  Q2  can an annually-fitted model reproduce the short-run dynamics that
      matter for markets and supply chains, or only the long-run stock
      trajectory?

Before either can be answered, one thing the spec assumes has to be checked
rather than assumed: **that the Ch. 3 mechanism produces a sub-annual
oscillation in the first place.**  Gambaro et al. (2025) Eqs. (18)-(21) are
solved here at their own published parameters and swept over their own
published lead-time range, and they do not.  Part 1 reports that; Parts 2-4
then answer Q1 and Q2 on two arms that carry identical market volatility and
differ only in frequency — the published reference case, and the same
equations in the fast-clearing regime that is the only way they oscillate
inside the year.

Parts
-----
  1  the cobweb, characterised.  Period and amplitude against the mining
     lead time, over the paper's range and below it.
  2  Q1 — survival.  Per Fourier component of each arm: the closed-form
     boxcar gain into a period-integrated flow, the measured gain on the
     twin, the point-sampling alias destination for a stock, and the share
     of the market signal that survives at all.
  3  the datasets and the fits (annual observation, `anchor_v4` unmodified).
  4  Q2 — short-run reproduction.  Each annually-fitted model is evaluated
     at monthly resolution — it is a continuous-time model, so it can be
     asked — and its alpha path is compared with the truth in-band and
     out-of-band.  The published real fit is measured the same way, which
     bounds what any annually-driven model of this class can represent.

    python run_wp11e.py --check
    python run_wp11e.py --cobweb --survival
    python run_wp11e.py --generate
    python run_wp11e.py --fits --seeds 0,1,2,3,4,5,6,7
    python run_wp11e.py --shortrun --figures
"""

from __future__ import annotations

import argparse
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
ANCHOR_DIR = os.path.join(HERE, "anchor_v4")

COL = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00", "#56B4E9"]
GREY = "#555555"

ALPHA_NAMES = S.ALPHA_NAMES
ARMS = tuple(M.MARKET_ARMS)
DELTA = 1.0                       # the observation width the arms are fitted at
N_SUB_EVAL = 24                   # nodes/yr for the continuous evaluation in Q2
TRAINVAL_END = 2007.0
# Components below this share of the arm's oscillation amplitude are noise of
# the Fourier bridge and are not reported per-component.
COMP_MIN_SHARE = 0.02


def _tag(arm):
    return f"{arm}_d1y_clean"


def rel_rmse_pct(pred, obs, mask=None):
    pred = np.asarray(pred, float); obs = np.asarray(obs, float)
    m = np.isfinite(pred) & np.isfinite(obs)
    if mask is not None:
        m &= np.asarray(mask, bool)
    if not np.any(m):
        return np.nan
    den = np.mean(np.abs(obs[m]))
    return 100.0 * np.sqrt(np.mean((pred[m] - obs[m]) ** 2)) / max(den, 1e-12)


# ===========================================================================
# Part 1 — the cobweb, characterised
# ===========================================================================
def part1_cobweb(ctx, out_dir=OUT_DIR):
    """Period and amplitude of the Ch. 3 oscillation against the lead time.

    The paper's own sweep (Sec. 5.1) runs `tau_E` from 2 to 32 yr; three
    shorter values are added below it, labelled as extrapolation, precisely
    to test whether shortening the lead time can push the cycle inside the
    year.  Reported alongside is the number of complete cycles the 1980-2019
    record contains, because a period estimated from two crossings is
    quantised and should not be quoted to more than a bin.
    """
    rows = []
    for te in tuple(M.CH3_TAU_SHORT) + tuple(M.CH3_TAU_PUBLISHED):
        sol = M.cobweb_solve(ctx, tau_E=te, tau_R=min(M.CH3["tau_R"], te))
        dp = M.dominant_period(sol)
        rows.append(dict(
            tau_E_yr=float(te), tau_R_yr=float(min(M.CH3["tau_R"], te)),
            c_per_yr=float(M.CH3["c"]),
            published_range=bool(te in M.CH3_TAU_PUBLISHED),
            period_zc_yr=dp["period_yr"], period_fft_yr=dp["period_fft_yr"],
            f_zc_per_yr=1.0 / dp["period_yr"] if np.isfinite(dp["period_yr"]) else 0.0,
            cycles_in_record=dp["cycles_in_record"],
            rms_log_price=dp["rms_log"],
            peak_to_trough_price_pct=100.0 * (sol["P"].max() / max(sol["P"].min(), 1e-12) - 1.0),
            subannual=bool(dp["period_yr"] < 1.0)))
    # the fast-clearing regime: what it takes for the same equations to
    # oscillate inside the year
    for c in (1.6, 5.0, 16.0, 50.0):
        for te in (0.25, 0.5):
            try:
                sol = M.cobweb_solve(ctx, tau_E=te, tau_R=te, params=dict(c=c))
            except FloatingPointError:
                continue
            dp = M.dominant_period(sol)
            rows.append(dict(
                tau_E_yr=float(te), tau_R_yr=float(te), c_per_yr=float(c),
                published_range=False,
                period_zc_yr=dp["period_yr"], period_fft_yr=dp["period_fft_yr"],
                f_zc_per_yr=1.0 / dp["period_yr"] if np.isfinite(dp["period_yr"]) else 0.0,
                cycles_in_record=dp["cycles_in_record"],
                rms_log_price=dp["rms_log"],
                peak_to_trough_price_pct=100.0 * (sol["P"].max() / max(sol["P"].min(), 1e-12) - 1.0),
                subannual=bool(dp["period_yr"] < 1.0)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "wp11e_cobweb.csv"), index=False)
    return df


# ===========================================================================
# Part 2 — Q1: does annual observation destroy the market signal?
# ===========================================================================
def part2_survival(ctx, arms=ARMS, deltas=(1.0,), out_dir=OUT_DIR):
    """Per-component survival of the market oscillation under each operator.

    Two operators, two fates.  A period-integrated flow sees the boxcar
    `|sinc(pi f Delta)|`, which is **exactly zero** at `f = k/Delta` — the
    component is annihilated, not aliased.  A point-sampled stock sees unit
    gain and folding: the same component reappears at `|f - k/Delta|`, and at
    `f = k/Delta` it folds to zero frequency, i.e. it becomes a *level shift*
    in the sampled series rather than disappearing from it.

    Measured as well as derived: the twin is solved with the arm's season
    injected, the flow accumulators are differenced at width `Delta`, and the
    amplitude at each component frequency is recovered by least squares
    (`project_amplitude`, phase-safe — none of these frequencies is a
    harmonic of the record).
    """
    import zinc_colloc_v5 as v5

    rows = []
    for arm in arms:
        a = M.arm_solution(ctx, arm)
        dense = S.dense_solve(ctx, "base", season=a["season"])
        fc = a["fourier"]
        amax = float(np.max(fc["a"])) if fc["a"].size else 0.0
        keep = np.where(fc["a"] >= COMP_MIN_SHARE * amax)[0]
        for delta in deltas:
            idx = S._window_indices(dense, delta)
            F_int = dense["C"][idx[1:]] - dense["C"][idx[:-1]]
            t_mid = 0.5 * (dense["t"][idx[1:]] + dense["t"][idx[:-1]])
            S_pt = dense["S4"][idx]
            t_pt = dense["t"][idx]
            for i in keep:
                f = float(fc["f"][i])
                a_inj = float(fc["a"][i] * a["amplitude_scale"])
                gain = float(M.sinc_gain(f, delta))
                fa = float(M.alias_freq(f, delta))
                row = dict(
                    arm=arm, published=a["published"], delta=float(delta),
                    f_per_yr=f, period_yr=1.0 / f if f > 0 else np.inf,
                    amp_logP_nats=a_inj,
                    boxcar_gain=gain,
                    alias_f_per_yr=fa,
                    alias_period_yr=1.0 / fa if fa > 1e-9 else np.inf,
                    above_nyquist=bool(f > 0.5 / delta),
                    harmonic_of_window=bool(abs(f * delta - round(f * delta)) < 1e-6))
                rows.append(row)
        # arm-level summary of how much of the signal is above Nyquist
        f_all, a_all = fc["f"], fc["a"] * a["amplitude_scale"]
        p = a_all ** 2
        for delta in deltas:
            hi = f_all > 0.5 / delta
            g2 = M.sinc_gain(f_all, delta) ** 2
            rows.append(dict(
                arm=arm, published=a["published"], delta=float(delta),
                f_per_yr=np.nan, period_yr=a["period"]["period_yr"],
                amp_logP_nats=float(np.sqrt(p.sum())),
                boxcar_gain=float(np.sqrt((p * g2).sum() / max(p.sum(), 1e-30))),
                alias_f_per_yr=np.nan, alias_period_yr=np.nan,
                above_nyquist=np.nan, harmonic_of_window=np.nan,
                summary_power_share_above_nyquist=float(p[hi].sum() / max(p.sum(), 1e-30)),
                # of the power that *is* above Nyquist, how much a period
                # integral lets through, and where it ends up
                summary_boxcar_gain_above_nyquist=float(np.sqrt(
                    (p[hi] * g2[hi]).sum() / max(p[hi].sum(), 1e-30))),
                summary_alias_band_lo=float(M.alias_freq(f_all[hi], delta).min())
                if hi.any() else np.nan,
                summary_alias_band_hi=float(M.alias_freq(f_all[hi], delta).max())
                if hi.any() else np.nan))
        import jax
        jax.clear_caches()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "wp11e_survival.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# Part 2b — the operators, verified on the twin by controlled single tones
# ---------------------------------------------------------------------------
# A per-component measurement inside a broadband arm is not usable: dozens of
# components fold onto the same low-frequency bins, so a regression at `f_alias`
# picks up whatever genuinely lives there as well as the alias, and the
# realised "gain" comes out anywhere between 1 and 39.  That is reported as a
# flag rather than as a result.  The clean measurement drives the twin with a
# *single* tone and differences the trajectory against a tone-free baseline,
# which isolates the tone's own contribution exactly.
def part2b_single_tone(ctx, freqs=(0.35, 0.95, 1.0, 1.21, 1.35, 2.0, 2.7),
                       amp=0.20, phase=0.0, delta=1.0, out_dir=OUT_DIR):
    """Realised gain of each operator at a single injected frequency."""
    import zinc_colloc_v5 as v5

    chans = [c for c in M.MARKET_ELASTICITY]
    base = S.dense_solve(ctx, "base")
    t0 = float(S.TRUTH["season_t0"])
    idx = S._window_indices(base, delta)
    rows = []
    for f in freqs:
        season = [(c, float(f), float(amp * M.MARKET_ELASTICITY[c] / 0.35), phase)
                  for c in chans]
        dense = S.dense_solve(ctx, "base", season=season)
        fa = float(M.alias_freq(f, delta))
        g = float(M.sinc_gain(f, delta))
        for fl in ("waelz_input", "direct_reuse_recycling"):
            j = v5.FLOW_NAMES.index(fl)
            rate = lambda D: np.log(np.maximum(
                np.diff(D["C"][:, j]) / np.diff(D["t"]), 1e-12))
            t_d = 0.5 * (dense["t"][1:] + dense["t"][:-1])
            a_dense = M.project_amplitude(t_d, rate(dense) - rate(base), f, t0=t0)["amp"]
            Fi = dense["C"][idx[1:]] - dense["C"][idx[:-1]]
            Fb = base["C"][idx[1:]] - base["C"][idx[:-1]]
            t_m = 0.5 * (dense["t"][idx[1:]] + dense["t"][idx[:-1]])
            d_ann = (np.log(np.maximum(Fi[:, j], 1e-12))
                     - np.log(np.maximum(Fb[:, j], 1e-12)))
            dc = fa < 1e-9
            a_ann = (np.nan if dc else
                     M.project_amplitude(t_m, d_ann, fa, t0=t0)["amp"])
            rows.append(dict(f_per_yr=float(f), operator="period integral",
                             series=fl, alias_f_per_yr=fa,
                             closed_form_gain=g,
                             dense_amp_nats=a_dense, annual_amp_nats=a_ann,
                             realised_gain=(np.nan if dc else
                                            a_ann / max(a_dense, 1e-12)),
                             # a component folded to zero frequency is not an
                             # oscillation in the sampled series but a level
                             # shift, so it is reported as one.  For a period
                             # integral the expected shift is the Jensen gap
                             # log I_0(a), which survives the null.
                             level_shift_nats=float(np.mean(d_ann)),
                             jensen_expected_nats=float(
                                 M.jensen_log_bias(amp)) if dc else np.nan))
        for st, nm in ((1, "Refined"), (3, "Scrap")):
            ld = np.log(np.maximum(dense["S4"][:, st], 1e-12)) \
                 - np.log(np.maximum(base["S4"][:, st], 1e-12))
            a_dense = M.project_amplitude(dense["t"], ld, f, t0=t0)["amp"]
            dc = fa < 1e-9
            a_ann = (np.nan if dc else
                     M.project_amplitude(dense["t"][idx], ld[idx], fa, t0=t0)["amp"])
            rows.append(dict(f_per_yr=float(f), operator="point sample",
                             series=nm, alias_f_per_yr=fa,
                             closed_form_gain=1.0,
                             dense_amp_nats=a_dense, annual_amp_nats=a_ann,
                             realised_gain=(np.nan if dc else
                                            a_ann / max(a_dense, 1e-12)),
                             level_shift_nats=float(np.mean(ld[idx])),
                             jensen_expected_nats=np.nan))
        import jax
        jax.clear_caches()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "wp11e_single_tone.csv"), index=False)
    return df


# ===========================================================================
# Part 3 — datasets and fits
# ===========================================================================
def part3_generate(ctx, arms=ARMS, out_dir=SYNTH_DIR, verbose=True):
    """One `load_zinc_data`-shaped dataset per arm at annual observation,
    with the analytic truth stored alongside, exactly as WP-3 does."""
    os.makedirs(out_dir, exist_ok=True)
    report = {}
    for arm in arms:
        a = M.arm_solution(ctx, arm)
        if a["fourier"]["rel_l2"] > M.FOURIER_TOL:
            raise RuntimeError(
                f"{arm}: Fourier bridge rel L2 {a['fourier']['rel_l2']:.3f} "
                f"exceeds {M.FOURIER_TOL}; raise n_keep before generating")
        dense = S.dense_solve(ctx, "base", season=a["season"])
        samp = S.sample(dense, DELTA, ctx, noise_scale=0.0)
        ds = S.build_dataset(ctx, samp, DELTA)
        bias = S.alpha_target_bias(ctx, dense, DELTA, arm_name="base",
                                   season=a["season"])
        tc_pt = S.truth_coefficients(ctx, samp["years"], "base",
                                     S_path=samp["stocks_clean"],
                                     season=a["season"])
        truth = dict(alphas_point=tc_pt["alphas"],
                     tau_sup_point=tc_pt["tau_sup"],
                     f_cohort_point=tc_pt["f_cohort"],
                     alphas_window_unweighted=bias["a_unweighted"],
                     alphas_window_weighted=bias["a_weighted"],
                     alphas_window_obs=bias["a_obs"],
                     window_years=bias["years"],
                     alpha_names=np.array(ALPHA_NAMES, dtype=object))
        ver = S.verify(ctx, dense, {DELTA: samp})
        # `zinc_synth_lab.verify` re-solves for its convergence check through
        # `dense_solve(ctx, dense["arm"])` and cannot pass a `season` that was
        # supplied explicitly rather than implied by the arm name, so its two
        # solver keys would be comparing a market twin against a plain one.
        # The core is read-only and so is that module's WP-3 contract, so the
        # check is redone here with the season carried through and the two
        # keys overwritten.
        ref = S.dense_solve(ctx, "base", season=a["season"], n_sub=2 * S.N_SUB)
        i2 = np.searchsorted(ref["t"], dense["t"])
        ver["solver_stock_rel"] = float(np.max(
            np.abs(ref["S4"][i2] - dense["S4"])
            / np.maximum(np.abs(dense["S4"]), 1.0)))
        ver["solver_flow_rel"] = float(np.max(
            np.abs(ref["C"][i2] - dense["C"])
            / np.maximum(np.abs(dense["C"]), 1.0)))
        ver["fourier_rel_l2"] = float(a["fourier"]["rel_l2"])
        meta = dict(arm=arm, base_arm="base", delta=DELTA, noise_scale=0.0,
                    market=dict(tau_E=a["sol"]["tau_E"], tau_R=a["sol"]["tau_R"],
                                params=a["sol"]["params"], ic=a["sol"]["ic"],
                                published=a["published"],
                                period_yr=a["period"]["period_yr"],
                                rms_logP=a["rms_logP"],
                                amplitude_scale=a["amplitude_scale"],
                                elasticity=M.MARKET_ELASTICITY,
                                n_fourier=len(a["fourier"]["f"]),
                                fourier_rel_l2=a["fourier"]["rel_l2"]),
                    verify=ver)
        path = S.save_dataset(os.path.join(out_dir, f"{_tag(arm)}.npz"), ds,
                              truth=truth, meta=meta)
        np.savez_compressed(os.path.join(out_dir, f"{arm}_dense.npz"),
                            t=dense["t"], S4=dense["S4"], C=dense["C"],
                            S_cohorts=dense["S_cohorts"],
                            season=np.asarray(a["season"], dtype=object),
                            osc_t=a["sol"]["t"], osc=a["sol"]["osc"],
                            P=a["sol"]["P"], E=a["sol"]["E"], R=a["sol"]["R"],
                            meta_json=np.asarray(json.dumps(meta["market"]),
                                                 dtype=object))
        report[arm] = dict(path=path, **{k: float(v) for k, v in ver.items()})
        if verbose:
            print(f"[{arm}] {path}")
            for k, v in ver.items():
                print(f"    {k:28s} {v:.3e}")
        import jax
        jax.clear_caches()
    with open(os.path.join(out_dir, "wp11e_manifest.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    return report


def fit_one(seed, tag, cfg, out_dir=FIT_DIR, verbose=False):
    """One full `anchor_v4` fit against an armed market twin.  Mirrors
    `run_wp3.fit_one` exactly — the pipeline is unmodified."""
    import zinc_colloc_v5 as v5
    import zinc_A_lab as Alab

    os.makedirs(out_dir, exist_ok=True)
    ds = S.load_dataset(os.path.join(SYNTH_DIR, f"{tag}.npz"))
    S.arm(ds)
    t0 = time.time()
    fit = v5.run(f"wp11e_{tag}_s{seed}", **dict(cfg, seed=int(seed), verbose=verbose))
    wall = time.time() - t0
    path = os.path.join(out_dir, f"{tag}_seed{seed}.npz")
    Alab.dump_A(fit, path, stage="B")
    S.disarm()
    import jax
    jax.clear_caches()
    return path, wall


def part3_fits(arms, seeds, cfg, out_dir=OUT_DIR, fit_dir=FIT_DIR, verbose=True):
    rows = []
    log = os.path.join(out_dir, "wp11e_fits.jsonl")
    for arm in arms:
        tag = _tag(arm)
        for seed in seeds:
            path = os.path.join(fit_dir, f"{tag}_seed{seed}.npz")
            if os.path.exists(path):
                if verbose:
                    print(f"[skip] {path}")
                continue
            p, wall = fit_one(seed, tag, cfg)
            rows.append(dict(arm=arm, tag=tag, seed=int(seed), wall_s=wall,
                             path=p, rss_mb=S._rss_mb()))
            with open(log, "a") as fh:
                fh.write(json.dumps(rows[-1]) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            if verbose:
                print(f"[fit] {tag} seed {seed}  {wall:.0f}s  "
                      f"RSS {rows[-1]['rss_mb']:.0f} MB")
    return pd.DataFrame(rows)


# ===========================================================================
# Part 4 — Q2: can an annually-fitted model reproduce the short run?
# ===========================================================================
def _log_paths(a_hat, a_true):
    return (np.log(np.maximum(a_hat, 1e-30)),
            np.log(np.maximum(a_true, 1e-30)))


def part4_shortrun(ctx, arms, seeds, cfg, out_dir=OUT_DIR, fit_dir=FIT_DIR):
    """Evaluate each annually-fitted model between its own observation nodes.

    The UDE is continuous in `t`, so `alpha_hat(t)` exists at any resolution.
    What it *cannot* contain is genuine sub-annual information, because the
    only time-varying inputs it has are the drivers, delivered by `exog_fn`
    as a piecewise-linear interpolation of annual values.  Any power it shows
    above the annual Nyquist is therefore the spectrum of that interpolation
    and not a recovered market cycle.  Both halves are measured: the
    in-band amplitude at the market frequency, which the model *could*
    recover, and the out-of-band share, which it cannot.
    """
    rows = []
    for arm in arms:
        tag = _tag(arm)
        tp = os.path.join(SYNTH_DIR, f"{tag}.npz")
        if not os.path.exists(tp):
            continue
        a = M.arm_solution(ctx, arm)
        f_mkt = 1.0 / a["period"]["period_yr"]
        ds = S.load_dataset(tp)
        yrs = np.asarray(ds["years"], float)
        t_fine = np.linspace(yrs[0], yrs[-1],
                             int(round((yrs[-1] - yrs[0]) * N_SUB_EVAL)) + 1)
        # truth on the fine grid, at the twin's own stock path
        dense = S.dense_solve(ctx, "base", season=a["season"])
        S_fine = np.column_stack([np.interp(t_fine, dense["t"], dense["S4"][:, j])
                                  for j in range(4)])
        tc = S.truth_coefficients(ctx, t_fine, "base", S_path=S_fine,
                                  season=a["season"])
        a_true = tc["alphas"]
        tv = t_fine <= TRAINVAL_END

        S.arm(ds)
        import zinc_cf_lab as cflab
        fit = cflab.init_fit(cfg=cfg, seed=0)
        for seed in seeds:
            p = os.path.join(fit_dir, f"{tag}_seed{seed}.npz")
            if not os.path.exists(p):
                continue
            params = cflab.load_params(p)
            cp = M.coeff_path(fit, params, t_fine, S_path=S_fine)
            a_hat = cp["alphas"]
            for k, name in enumerate(ALPHA_NAMES):
                lh, lt = _log_paths(a_hat[:, k], a_true[:, k])
                pr_h = M.project_amplitude(t_fine, lh, f_mkt,
                                           t0=float(S.TRUTH["season_t0"]))
                pr_t = M.project_amplitude(t_fine, lt, f_mkt,
                                           t0=float(S.TRUTH["season_t0"]))
                bp_h = M.band_power(t_fine, lh, f_cut=0.5)
                bp_t = M.band_power(t_fine, lt, f_cut=0.5)
                # The sharpest link between Q1 and Q2: a market cycle above
                # the annual Nyquist reaches the record folded to `f_alias`,
                # so an annually-fitted model has every reason to reproduce a
                # cycle *there* which the truth does not contain.  Measured
                # rather than asserted.
                f_al = float(M.alias_freq(f_mkt, 1.0))
                al_h = M.project_amplitude(t_fine, lh, f_al,
                                           t0=float(S.TRUTH["season_t0"]))
                al_t = M.project_amplitude(t_fine, lt, f_al,
                                           t0=float(S.TRUTH["season_t0"]))
                rows.append(dict(
                    arm=arm, published=a["published"], seed=int(seed),
                    channel=name, f_market_per_yr=f_mkt,
                    market_above_nyquist=bool(f_mkt > 0.5),
                    amp_true_nats=pr_t["amp"], amp_fit_nats=pr_h["amp"],
                    amp_ratio=pr_h["amp"] / max(pr_t["amp"], 1e-12),
                    phase_gap_rad=float(np.angle(np.exp(
                        1j * (pr_h["phase"] - pr_t["phase"])))),
                    f_alias_per_yr=f_al,
                    amp_true_at_alias_nats=al_t["amp"],
                    amp_fit_at_alias_nats=al_h["amp"],
                    alias_amp_ratio=al_h["amp"] / max(al_t["amp"], 1e-12),
                    hi_share_true=bp_t["share_above"],
                    hi_share_fit=bp_h["share_above"],
                    rms_log_true=bp_t["rms"], rms_log_fit=bp_h["rms"],
                    relRMSE_fine_pct=rel_rmse_pct(a_hat[:, k], a_true[:, k], tv),
                    relRMSE_annual_pct=rel_rmse_pct(
                        np.interp(yrs, t_fine, a_hat[:, k]),
                        np.interp(yrs, t_fine, a_true[:, k]),
                        yrs <= TRAINVAL_END)))
        S.disarm()
        import jax
        jax.clear_caches()
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(os.path.join(out_dir, "wp11e_shortrun_per_seed.csv"), index=False)
        g = df.groupby(["arm", "channel"])
        summ = g.agg(n_seeds=("seed", "nunique"),
                     f_market_per_yr=("f_market_per_yr", "first"),
                     amp_true_nats=("amp_true_nats", "median"),
                     amp_fit_nats=("amp_fit_nats", "median"),
                     amp_ratio=("amp_ratio", "median"),
                     f_alias_per_yr=("f_alias_per_yr", "first"),
                     amp_true_at_alias_nats=("amp_true_at_alias_nats", "median"),
                     amp_fit_at_alias_nats=("amp_fit_at_alias_nats", "median"),
                     alias_amp_ratio=("alias_amp_ratio", "median"),
                     hi_share_true=("hi_share_true", "median"),
                     hi_share_fit=("hi_share_fit", "median"),
                     relRMSE_fine_pct=("relRMSE_fine_pct", "median"),
                     relRMSE_annual_pct=("relRMSE_annual_pct", "median")).reset_index()
        summ.to_csv(os.path.join(out_dir, "wp11e_market_dynamics.csv"), index=False)
        return df, summ
    return df, df


def part4b_published_bound(cfg, out_dir=OUT_DIR, n_seeds=35):
    """The same out-of-band measurement on the **published real fit**.

    No twin involved: this is what fraction of the fitted `alpha_hat(t)`'s
    variance sits above the annual Nyquist on the real 35-seed ensemble.
    With `use_stock_input: false` the only time-varying input is `exog_fn`'s
    piecewise-linear interpolation of annual drivers, so whatever is found
    here is an upper bound on the sub-annual structure any model of this
    class can express — and it is the shape of the interpolation, not a
    market cycle.
    """
    import zinc_cf_lab as cflab

    S.disarm()
    fit = cflab.init_fit(cfg=cfg, seed=0)
    yrs = np.asarray(fit.data_all["years"], float).ravel()
    t_fine = np.linspace(yrs[0], yrs[-1],
                         int(round((yrs[-1] - yrs[0]) * N_SUB_EVAL)) + 1)
    rows = []
    for seed in range(n_seeds):
        p = os.path.join(M.WEIGHTS_DIR_DEFAULT, f"A_seed{seed}.npz")
        if not os.path.exists(p):
            continue
        params = cflab.load_params(p)
        cp = M.coeff_path(fit, params, t_fine)
        for k, name in enumerate(ALPHA_NAMES):
            l = np.log(np.maximum(cp["alphas"][:, k], 1e-30))
            bp = M.band_power(t_fine, l, f_cut=0.5)
            rows.append(dict(seed=int(seed), channel=name,
                             hi_share=bp["share_above"], rms_log=bp["rms"]))
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(os.path.join(out_dir, "wp11e_published_band.csv"), index=False)
    return df


# ===========================================================================
# figures
# ===========================================================================
def figures(ctx, cob, surv, summ, out_dir=OUT_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11.0, 8.0))

    # (a) the cobweb price paths
    a0 = ax[0, 0]
    for i, arm in enumerate(ARMS):
        a = M.arm_solution(ctx, arm)
        y = a["sol"]["osc"] * a["amplitude_scale"]
        a0.plot(a["sol"]["t"], y, color=COL[i], lw=1.0,
                label=f"{arm}  ({a['period']['period_yr']:.2f} yr cycle)")
    a0.axhline(0.0, color=GREY, lw=0.6)
    a0.set_xlabel("year"); a0.set_ylabel("log price oscillation [nats]")
    a0.set_title("(a) Ch. 3 cobweb, rescaled to common volatility")
    a0.legend(fontsize=7, frameon=False)

    # (b) period against lead time
    a1 = ax[0, 1]
    pub = cob[cob.published_range & (cob.c_per_yr == M.CH3["c"])]
    ext = cob[(~cob.published_range) & (cob.c_per_yr == M.CH3["c"])]
    a1.loglog(pub.tau_E_yr, pub.period_zc_yr, "o-", color=COL[0],
              label="Ch. 3 published range")
    a1.loglog(ext.tau_E_yr, ext.period_zc_yr, "s--", color=COL[2],
              label="shorter lead times (ours)")
    fast = cob[cob.c_per_yr > M.CH3["c"]]
    if not fast.empty:
        a1.loglog(fast.tau_E_yr, fast.period_zc_yr, "^", color=COL[1],
                  label="fast-clearing regime")
    a1.axhline(1.0, color=GREY, ls=":", lw=1.0)
    a1.text(a1.get_xlim()[0] * 1.1, 1.05, "one year", fontsize=7, color=GREY)
    a1.set_xlabel(r"mining lead time $\tau_E$ [yr]")
    a1.set_ylabel("oscillation period [yr]")
    a1.set_title("(b) the delay does not buy sub-annual cycles")
    a1.legend(fontsize=7, frameon=False)

    # (c) the two operators
    a2 = ax[1, 0]
    f = np.linspace(0.0, 5.0, 2001)
    a2.plot(f, M.sinc_gain(f, 1.0), color=COL[0], lw=1.2,
            label=r"flow: period integral, $|\mathrm{sinc}(\pi f\Delta)|$")
    a2.plot(f, np.ones_like(f), color=COL[2], lw=1.2, ls="--",
            label="stock: point sample, unit gain")
    a2.plot(f, M.alias_freq(f, 1.0), color=COL[1], lw=1.0, ls=":",
            label="alias destination [1/yr]")
    # Mark each arm at its *cycle* frequency (1 / the zero-crossing period),
    # not at its largest Fourier component: both arms carry a slow envelope
    # whose amplitude dominates the spectrum without being the cycle.
    for arm, c in zip(ARMS, (COL[3], COL[4])):
        sub = surv[(surv.arm == arm) & surv.f_per_yr.isna()]
        if sub.empty:
            continue
        f_cyc = 1.0 / float(sub.period_yr.iloc[0])
        a2.axvline(f_cyc, color=c, lw=1.2, alpha=0.85)
        a2.text(f_cyc + 0.06, 1.02, f"{arm} ({f_cyc:.2f}/yr)", fontsize=7,
                color=c, rotation=90, va="bottom")
    a2.set_ylim(-0.02, 1.55)
    a2.set_xlabel("frequency [1/yr]"); a2.set_ylabel("gain / alias frequency")
    a2.set_title(r"(c) annual observation, $\Delta=1$ yr")
    a2.legend(fontsize=7, frameon=False)

    # (d) recovered amplitude against truth
    a3 = ax[1, 1]
    if summ is not None and not summ.empty:
        w = 0.35
        xs = np.arange(len(ALPHA_NAMES))
        for i, arm in enumerate(ARMS):
            sub = summ[summ.arm == arm].set_index("channel").reindex(ALPHA_NAMES)
            a3.bar(xs + (i - 0.5) * w, sub.amp_ratio.to_numpy(), width=w,
                   color=COL[i], label=arm)
        a3.axhline(1.0, color=GREY, lw=0.8, ls="--")
        a3.set_xticks(xs); a3.set_xticklabels(ALPHA_NAMES, rotation=20, fontsize=7)
        a3.set_ylabel(r"recovered / true amplitude at $f_{\rm market}$")
        a3.set_title("(d) what the annual fit recovers")
        a3.legend(fontsize=7, frameon=False)
    else:
        a3.text(0.5, 0.5, "fits not run", ha="center", va="center",
                transform=a3.transAxes, color=GREY)
        a3.set_axis_off()

    fig.tight_layout()
    for ext_ in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"wp11e_market_dynamics.{ext_}"),
                    dpi=200)
    plt.close(fig)


# ===========================================================================
# CLI
# ===========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--cobweb", action="store_true")
    ap.add_argument("--survival", action="store_true")
    ap.add_argument("--single-tone", action="store_true")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--fits", action="store_true")
    ap.add_argument("--shortrun", action="store_true")
    ap.add_argument("--published-band", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    a = ap.parse_args(argv)

    import zinc_colloc_v5 as v5
    M.install(v5)
    cfg = M.load_anchor_config()
    ctx = S.driver_context(cfg)

    print(f"[wp11e] drivers resolved by name: "
          f"{ {k: v for k, v in S.DRIVER_ALIASES.items()} }")
    n_ex = len(ctx["exog_cols"])
    orders = len(cfg.get("exog_feature_orders", (0, 1)))
    print(f"[wp11e] exog columns = {n_ex}, feature orders = {orders}, "
          f"input_dim = {1 + 4 + n_ex * orders}  "
          f"(CLAUDE.md rule 2 says 23 — SCHEMA §9 flag 1, open)")

    if a.check:
        M.check(verbose=True)
        return 0

    arms = tuple(x for x in a.arms.split(",") if x)
    seeds = [int(x) for x in a.seeds.split(",") if x != ""]
    os.makedirs(OUT_DIR, exist_ok=True)

    cob = surv = summ = None
    if a.cobweb or a.all:
        cob = part1_cobweb(ctx)
        print(cob[["tau_E_yr", "c_per_yr", "period_zc_yr", "rms_log_price",
                   "subannual"]].to_string(index=False))
    if a.survival or a.all:
        surv = part2_survival(ctx, arms)
        print(surv[surv.f_per_yr.notna()][
            ["arm", "f_per_yr", "amp_logP_nats", "boxcar_gain",
             "alias_f_per_yr"]].to_string(index=False))
    if a.single_tone or a.all:
        st = part2b_single_tone(ctx)
        print(st.to_string(index=False))
    if a.generate or a.all:
        part3_generate(ctx, arms)
    if a.fits or a.all:
        part3_fits(arms, seeds, cfg)
    if a.shortrun or a.all:
        _, summ = part4_shortrun(ctx, arms, seeds, cfg)
        if summ is not None and not summ.empty:
            print(summ.to_string(index=False))
    if a.published_band or a.all:
        pb = part4b_published_bound(cfg)
        if not pb.empty:
            print(pb.groupby("channel").hi_share.median().to_string())
    if a.figures or a.all:
        cob = cob if cob is not None else pd.read_csv(
            os.path.join(OUT_DIR, "wp11e_cobweb.csv"))
        surv = surv if surv is not None else pd.read_csv(
            os.path.join(OUT_DIR, "wp11e_survival.csv"))
        mp = os.path.join(OUT_DIR, "wp11e_market_dynamics.csv")
        summ = summ if summ is not None else (
            pd.read_csv(mp) if os.path.exists(mp) else None)
        figures(ctx, cob, surv, summ)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
