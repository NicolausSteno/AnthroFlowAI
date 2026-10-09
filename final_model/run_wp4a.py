#!/usr/bin/env python3
"""
run_wp4a.py — WP-4a: attenuation and aliasing
=============================================

The question, restated precisely.  Flows are observed as **period integrals**,
which is a boxcar of width `Delta` with transfer function `|sinc(pi f Delta)|`
and exact nulls at `f = k/Delta`.  Stocks are observed **point-in-time**, which
is impulse sampling and has no such nulls.  The spec's two regimes are

  harmonic     1/yr, 2/yr   -- at an exact null of the annual boxcar
  non-harmonic 1.35/yr, 2.70/yr -- attenuated by |sinc|, then aliased

and the spec's hypothesis is that the harmonic regime is *irrecoverable in
principle, by any estimator*.

Parts
-----
1. the closed-form bound, the nulls, and where each component aliases to
2. what actually survives: the observation vector of each regime against the
   no-season control, decomposed into a resolvable oscillation and a level
   shift.  **This is where the spec's hypothesis needs amending** -- see the
   findings note.
3. the detectability surface: trace amplitude against the WP-3 calibrated
   noise, i.e. the recovery threshold in (frequency, amplitude, Delta), with
   no fits and no estimator
4. the architectural floor: how much power the model's *inputs* carry at each
   component frequency, and what a fit on season-free data reports there --
   the null distribution every recovery claim is measured against
5. the fits: 3 regimes x 3 window widths x 8 seeds
6. scoring: recovered amplitude and phase against the bound, and the damage
   the unrecoverable structure does to the coefficients that are recoverable
7. figures

Budget note.  The spec's compute table pre-registered "≥ 4 frequencies x
2 regimes x 8 seeds = 64" fits.  This runs 3 window widths (annual, quarterly,
monthly -- the three the spec names in the 4a text) x 3 regimes x 8 seeds = 72,
trading the fourth width for a **no-season control at every width**.  Without
that control none of the damage in Part 6 can be attributed to the sub-annual
structure rather than to the window width itself.  Eight of the 72 already
exist: `zinc_alias_lab.check()` verifies the `flat` twin at `Delta = 1` is
bit-identical to WP-3's `base_d1y_clean`, so that cell reuses WP-3's fits.

Usage
-----
    python run_wp4a.py --check
    python run_wp4a.py --parts 1,2,3,4        # no fits, minutes
    python run_wp4a.py --fit --workers 3      # the sweep
    python run_wp4a.py --parts 6,7            # scoring and figures
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import zinc_alias_lab as L                                        # noqa: E402
import zinc_synth_lab as S                                        # noqa: E402
import zinc_cf_lab as cflab                                       # noqa: E402
import zinc_circ_lab as C                                         # noqa: E402
from zinc_alpha_lab import integrity_check, load_anchor_config     # noqa: E402

OUT_DIR = os.path.join(HERE, "analysis")
SYNTH_DIR = L.OUT_DIR_DEFAULT
FIT_DIR = L.FIT_DIR_DEFAULT
WP3_FIT_DIR = L.WP3_FIT_DIR

ALPHA_NAMES = L.ALPHA_NAMES
TAU_SUP_NAMES = L.TAU_SUP_NAMES
TRAINVAL_END = 2006.0
N_SUB_EVAL = 96                    # dense grid for every projection
RSS_LIMIT_MB = 12000.0

REGIMES = ("flat", "harm", "nonharm")
DELTAS = L.DELTAS
SEEDS = L.SEEDS

# The frequency grid Part 1 and the figures are drawn on.
F_GRID = np.linspace(0.0, 6.0, 1201)[1:]
# Amplitudes for the detectability surface, in nats of log alpha.
AMP_GRID = np.array([0.02, 0.04, 0.08, 0.12, 0.20, 0.40, 0.80])


def _rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024.0 ** 2) if sys.platform == "darwin" else r / 1024.0


def hl(v):
    """`(estimate, lo, hi)` Hodges-Lehmann; interval is NaN below six points."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
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


def dense_grid(years):
    y = np.asarray(years, float)
    n = int(round((y[-1] - y[0]) * N_SUB_EVAL))
    return y[0] + np.arange(n + 1, dtype=float) / float(N_SUB_EVAL)


def fit_tag(regime, delta):
    return L.tag_for(regime, delta)


def fit_path(regime, delta, seed, arch="pub"):
    """Where a fit's weight dump lives.  The (flat, Delta=1, pub) cell is
    WP-3's, reused because the datasets are bit-identical (`check()`)."""
    if arch == "pub" and regime == "flat" and abs(delta - 1.0) < 1e-12:
        return os.path.join(WP3_FIT_DIR, f"base_d1y_clean_seed{seed}.npz")
    suffix = "" if arch == "pub" else f"_{arch}"
    return os.path.join(FIT_DIR, f"{fit_tag(regime, delta)}{suffix}_seed{seed}.npz")


# ===========================================================================
# Part 1 — the closed-form bound
# ===========================================================================
def part1_bound(out_dir=OUT_DIR):
    rows = []
    for dl in DELTAS:
        g = L.sinc_gain(F_GRID, dl)
        for f, gv in zip(F_GRID, g):
            rows.append(dict(delta=float(dl), dtag=L.dtag(dl), freq_per_yr=float(f),
                             sinc_gain=float(gv),
                             alias_freq_per_yr=float(L.alias_freq(f, dl)),
                             nyquist_per_yr=0.5 / float(dl)))
    curve = pd.DataFrame(rows)
    curve.to_csv(os.path.join(out_dir, "wp4a_sinc_bound.csv"), index=False)

    comp = []
    for reg in ("harm", "nonharm"):
        for (chan, f, a, ph) in L.REGIMES[reg]:
            for dl in DELTAS:
                g = float(L.sinc_gain(f, dl))
                comp.append(dict(
                    regime=reg, channel=chan, freq_per_yr=float(f),
                    amp_nats=float(a), phase=float(ph),
                    delta=float(dl), dtag=L.dtag(dl), sinc_gain=g,
                    is_null=bool(g < 1e-12),
                    nyquist_per_yr=0.5 / float(dl),
                    above_nyquist=bool(f > 0.5 / float(dl)),
                    alias_freq_per_yr=float(L.alias_freq(f, dl)),
                    trace_nats=float(a) * g,
                    n_windows=int(round(39.0 / float(dl)))))
    comp = pd.DataFrame(comp)
    comp.to_csv(os.path.join(out_dir, "wp4a_components.csv"), index=False)
    return curve, comp


# ===========================================================================
# Part 2 — what actually survives
# ===========================================================================
def _sliding_window_mean(t, y, delta):
    """Mean of `y` over a width-`Delta` window *at every dense offset*.

    The integration operator on its own, with the sampling deliberately left
    out.  Exact on the fine grid via a cumulative trapezoid, so the returned
    series is the boxcar-filtered signal evaluated at the window midpoints and
    nothing else has been done to it.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    h = float(t[1] - t[0])
    step = int(round(float(delta) / h))
    if abs(step * h - float(delta)) > 1e-9:
        raise ValueError(f"delta={delta} is not a whole number of grid steps")
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * h)])
    lo = np.arange(0, t.size - step)
    wm = (cum[lo + step] - cum[lo]) / float(delta)
    tm = 0.5 * (t[lo] + t[lo + step])
    return tm, wm


def _boxcar_measured(dense_path, comps, delta):
    """Measured gain of the **integration** operator, against `|sinc|`.

    Phase-safe by construction: the window mean is taken at every dense offset
    rather than on the non-overlapping observation grid, so this isolates the
    boxcar from the decimation that follows it.  Mixing the two is what makes a
    naive measurement report a gain of 16 where the closed form says zero --
    the annual observation grid cannot even represent a 1/yr regressor, so the
    projection is singular rather than small.  Sampling is Part 2b's job.
    """
    d = np.load(dense_path, allow_pickle=True)
    t, A = np.asarray(d["t"], float), np.asarray(d["alphas"], float)
    y = np.log(A[:, ALPHA_NAMES.index(L.SEASON_CHANNEL)])
    tm, wm = _sliding_window_mean(t, y, delta)
    out = []
    for (chan, f, a, ph) in comps:
        p_dense = L.project(t, y, f)
        p_win = L.project(tm, wm, f)
        out.append(dict(freq_per_yr=float(f), amp_nats=float(a),
                        dense_amp=p_dense["amp"], window_amp=p_win["amp"],
                        measured_gain=p_win["amp"] / max(p_dense["amp"], 1e-15),
                        theory_sinc=float(L.sinc_gain(f, delta)),
                        gain_abs_err=abs(p_win["amp"] / max(p_dense["amp"], 1e-15)
                                         - float(L.sinc_gain(f, delta))),
                        phase_err=float(np.abs(np.angle(np.exp(
                            1j * (p_win["phase"] - p_dense["phase"]))))),
                        n_offsets=int(tm.size)))
    return out


def _sampling_fate(dense_path, comps, delta):
    """Where the surviving trace lands once the window means are decimated.

    Three outcomes, and the paper needs them named apart:

      * `alias_freq == 0`      the component folds onto **DC**.  What reaches
        the record is a level shift, not a cycle: unidentifiable as structure
        and actively misattributed to the trend.
      * `alias_freq == nyquist` the fold lands exactly on the Nyquist rate,
        where only one quadrature survives and the phase is unidentifiable.
      * otherwise              the component appears at `alias_freq`, at the
        amplitude `|sinc|` left it, masquerading as slower structure.
    """
    d = np.load(dense_path, allow_pickle=True)
    t, A = np.asarray(d["t"], float), np.asarray(d["alphas"], float)
    y = np.log(A[:, ALPHA_NAMES.index(L.SEASON_CHANNEL)])
    tm, wm = _sliding_window_mean(t, y, delta)
    step = int(round(float(delta) / float(t[1] - t[0])))
    ts, ys = tm[::step], wm[::step]                 # the observation grid
    nyq = 0.5 / float(delta)
    out = []
    for (chan, f, a, ph) in comps:
        fa = float(L.alias_freq(f, delta))
        fate = ("dc" if fa < 1e-12 else
                "nyquist" if abs(fa - nyq) < 1e-12 else "aliased")
        rec = dict(freq_per_yr=float(f), amp_nats=float(a), delta=float(delta),
                   sinc_gain=float(L.sinc_gain(f, delta)),
                   alias_freq_per_yr=fa, nyquist_per_yr=nyq, fate=fate,
                   n_obs=int(ts.size))
        if fate == "aliased":
            knot = max(L.TREND_KNOT_YR, 4.0 * float(delta))
            rec["alias_amp"] = L.project(ts, ys, fa, knot_yr=knot)["amp"]
        else:
            rec["alias_amp"] = np.nan
        # the DC content the component contributes, whatever its fate
        rec["level_shift"] = float(np.mean(ys))
        out.append(rec)
    return out


def part2_survives(out_dir=OUT_DIR):
    """Operator gain on the truth, and the observation vector against control."""
    gain_rows, obs_rows, fate_rows = [], [], []
    for reg in ("harm", "nonharm"):
        dp = os.path.join(SYNTH_DIR, f"{reg}_dense.npz")
        flat_dp = os.path.join(SYNTH_DIR, "flat_dense.npz")
        for dl in DELTAS:
            for r in _boxcar_measured(dp, L.REGIMES[reg], dl):
                gain_rows.append(dict(regime=reg, delta=float(dl),
                                      dtag=L.dtag(dl), **r))
            base = {r["freq_per_yr"]: r["level_shift"]
                    for r in _sampling_fate(flat_dp, L.REGIMES[reg], dl)}
            for r in _sampling_fate(dp, L.REGIMES[reg], dl):
                r["level_shift_vs_flat"] = r["level_shift"] - base[r["freq_per_yr"]]
                fate_rows.append(dict(regime=reg, dtag=L.dtag(dl), **r))

    k = ALPHA_NAMES.index(L.SEASON_CHANNEL)
    for reg in ("harm", "nonharm"):
        for dl in DELTAS:
            A = S.load_dataset(os.path.join(SYNTH_DIR, f"{L.tag_for(reg, dl)}.npz"))
            B = S.load_dataset(os.path.join(SYNTH_DIR, f"{L.tag_for('flat', dl)}.npz"))
            rec = dict(regime=reg, delta=float(dl), dtag=L.dtag(dl),
                       n_obs=int(A["years"].size))
            for key in ("stocks_obs", "flows_obs", "alpha_obs", "tau_sup_obs"):
                x, y = np.asarray(A[key], float), np.asarray(B[key], float)
                m = np.isfinite(x) & np.isfinite(y) & (np.abs(y) > 1e-12)
                rec[f"max_rel_{key}"] = float(np.max(np.abs(x[m] - y[m]) / np.abs(y[m])))
            # the alpha target the season rides on, in log space
            x, y = A["alpha_obs"][:, k], B["alpha_obs"][:, k]
            m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            lg = np.log(x[m]) - np.log(y[m])
            rec["alpha_refc_log_level"] = float(np.median(lg))
            rec["alpha_refc_log_sd"] = float(np.std(lg))
            # how much of that spread is a resolvable oscillation?
            yrs = np.asarray(A["years"], float)[m]
            osc = 0.0
            for (_c, f, a, ph) in L.REGIMES[reg]:
                nyq = 0.5 / float(dl)
                if f < nyq:
                    p = L.project(yrs, lg, f, knot_yr=max(L.TREND_KNOT_YR, 4 * dl))
                    osc = float(np.hypot(osc, p["amp"]))
                    rec[f"osc_amp_{f:g}"] = p["amp"]
                    rec[f"osc_phase_err_{f:g}"] = float(np.abs(np.angle(
                        np.exp(1j * (p["phase"] - ph)))))
                else:
                    rec[f"osc_amp_{f:g}"] = np.nan
                    rec[f"osc_phase_err_{f:g}"] = np.nan
            rec["osc_amp_total"] = osc if osc > 0 else np.nan
            rec["truth_amp_total"] = float(np.sqrt(sum(
                a ** 2 for (_c, _f, a, _p) in L.REGIMES[reg])))
            rec["rectification_naive"] = float(sum(
                L.rectification(a) for (_c, _f, a, _p) in L.REGIMES[reg]))
            # Is any of it above the noise the real record carries?  The
            # coefficient path is annihilated exactly (Part 2b); what is left
            # here is the season's effect on the *observation vector*, which is
            # not zero because stocks are point-sampled rather than integrated
            # and because the alpha target is a ratio of one to the other.  The
            # question that decides the spec's hypothesis is whether that
            # residue clears the WP-3 calibrated observation noise.
            sig_a = L.alpha_target_sigma()
            sig_S = float(np.asarray(S.calibrate_noise()["stock"], float)[1])
            rec["sigma_alpha_log"] = sig_a
            rec["sigma_stock_log"] = sig_S
            rec["level_over_sigma"] = abs(rec["alpha_refc_log_level"]) / sig_a
            rec["stock_over_sigma"] = rec["max_rel_stocks_obs"] / sig_S
            rec["osc_over_sigma"] = (osc / sig_a) if osc > 0 else np.nan
            obs_rows.append(rec)

    gain = pd.DataFrame(gain_rows)
    obs = pd.DataFrame(obs_rows)
    fate = pd.DataFrame(fate_rows)
    gain.to_csv(os.path.join(out_dir, "wp4a_boxcar_gain.csv"), index=False)
    obs.to_csv(os.path.join(out_dir, "wp4a_annihilation.csv"), index=False)
    fate.to_csv(os.path.join(out_dir, "wp4a_sampling_fate.csv"), index=False)
    return gain, obs, fate


# ===========================================================================
# Part 3 — the detectability surface
# ===========================================================================
def part3_detectability(out_dir=OUT_DIR):
    freqs = sorted({float(f) for reg in ("harm", "nonharm")
                    for (_c, f, _a, _p) in L.REGIMES[reg]}
                   | {0.5, 0.75, 1.25, 1.5, 3.0, 4.0, 5.0})
    rows = L.detectability(freqs, DELTAS, AMP_GRID)
    df = pd.DataFrame(rows)
    df["dtag"] = df.delta.map(L.dtag)
    df.to_csv(os.path.join(out_dir, "wp4a_detectability.csv"), index=False)
    return df


# ===========================================================================
# Part 4 — the architectural floor
# ===========================================================================
def part4_capacity(out_dir=OUT_DIR, seeds=SEEDS):
    """Input-feature power at each component frequency, and the null.

    Under `anchor_v4` both `use_time_input` and `use_stock_input` are false, so
    alpha is a deterministic function of the exog feature block alone, and
    `v5.exog_fn` makes that block a piecewise-linear interpolant of an
    *annual* driver grid at every `Delta`.  Whatever power it carries at a
    sub-annual frequency is an interpolation-kink artefact at a fixed phase,
    not sub-annual information -- so the floor is measured, not assumed, and
    the null column is what a fit on season-free data reports at the same
    frequencies.  Every recovery claim in Part 6 is read against this.
    """
    rows = []
    freqs = sorted({float(f) for reg in ("harm", "nonharm")
                    for (_c, f, _a, _p) in L.REGIMES[reg]})
    for dl in DELTAS:
        tag = L.tag_for("flat", dl)
        fit, ds = L.fit_context(tag)
        t = dense_grid(ds["years"])
        for f in freqs:
            fp = L.feature_power(fit, f, t_grid=t)
            rec = dict(delta=float(dl), dtag=L.dtag(dl), freq_per_yr=f,
                       feat_rel_max=float(np.max(fp["rel"])),
                       feat_rel_median=float(np.median(fp["rel"])),
                       n_features=int(fp["amp"].size))
            amps = []
            for sd in seeds:
                p = fit_path("flat", dl, sd)
                if not os.path.exists(p):
                    continue
                params = cflab.load_params(p)
                A = L.dense_alpha(fit, params, t)
                amps.append(L.project(
                    t, np.log(A[:, ALPHA_NAMES.index(L.SEASON_CHANNEL)]), f)["amp"])
            rec["null_amp_median"] = float(np.median(amps)) if amps else np.nan
            rec["null_amp_q3"] = float(np.percentile(amps, 75)) if amps else np.nan
            rec["null_amp_max"] = float(np.max(amps)) if amps else np.nan
            rec["n_null_seeds"] = len(amps)
            rows.append(rec)
        S.disarm()
        import jax
        jax.clear_caches()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "wp4a_capacity.csv"), index=False)
    return df


# ===========================================================================
# Part 5 — the fits
# ===========================================================================
def scaled_config(delta, cfg=None, arch="pub"):
    """`anchor_v4`, with the Stage B curriculum rescaled to the node grid.

    `stageB_curriculum` widths are in **nodes**, so a 22-node window is 22
    years at `Delta = 1` and 22 months at `Delta = 1/12`.  Scaling by
    `1/Delta` holds the physical window fixed, which is the only way the arms
    are comparable; the timing runs in `analysis/wp4a_timing*.log` used the
    same convention.
    """
    cfg = dict(cfg or load_anchor_config())
    scale = int(round(1.0 / float(delta)))
    cfg["stageB_curriculum"] = [[int(s), int(w * scale)]
                                for (s, w) in cfg["stageB_curriculum"]]
    if arch == "tin":
        cfg["use_time_input"] = True
    return cfg


def fit_one(regime, delta, seed, *, arch="pub", out_dir=FIT_DIR, verbose=False):
    """One fit against an armed WP-4a dataset."""
    import zinc_colloc_v5 as v5
    import zinc_A_lab as Alab

    os.makedirs(out_dir, exist_ok=True)
    tag = fit_tag(regime, delta)
    ds = S.load_dataset(os.path.join(SYNTH_DIR, f"{tag}.npz"))
    L.install(v5)
    S.arm(ds)
    cfg = scaled_config(delta, arch=arch)
    t0 = time.time()
    fit = v5.run(f"wp4a_{tag}_{arch}_s{seed}",
                 **dict(cfg, seed=int(seed), verbose=verbose))
    wall = time.time() - t0
    path = fit_path(regime, delta, seed, arch=arch)
    Alab.dump_A(fit, path, stage="B")
    S.disarm()
    import jax
    jax.clear_caches()
    return path, wall


def plan(regimes=REGIMES, deltas=DELTAS, seeds=SEEDS, arch="pub",
         with_time_input=True):
    """Every (regime, delta, seed, arch) cell the package needs."""
    cells = [(r, float(d), int(s), arch)
             for r in regimes for d in deltas for s in seeds]
    if with_time_input:
        cells += [("nonharm", 1.0 / 12.0, int(s), "tin") for s in seeds]
    return cells


def part5_fit(cells=None, *, workers=1, out_dir=FIT_DIR, jsonl=None,
              verbose=False):
    """Run every missing cell in-process, with the rule-6 hygiene."""
    cells = cells or plan()
    os.makedirs(out_dir, exist_ok=True)
    jsonl = jsonl or os.path.join(out_dir, "wp4a_fits.jsonl")
    todo = [c for c in cells if not os.path.exists(fit_path(*c[:3], arch=c[3]))]
    print(f"[wp4a] {len(cells)} cells, {len(cells)-len(todo)} on disk, "
          f"{len(todo)} to run", flush=True)
    done = []
    for i, (reg, dl, sd, arch) in enumerate(todo, 1):
        rss = _rss_mb()
        if rss > RSS_LIMIT_MB:
            print(f"[wp4a] RSS watchdog {rss:.0f} MB > {RSS_LIMIT_MB:.0f} MB; "
                  f"stopping after {i-1} fits", flush=True)
            break
        p, wall = fit_one(reg, dl, sd, arch=arch, out_dir=out_dir, verbose=verbose)
        rec = dict(regime=reg, delta=dl, dtag=L.dtag(dl), seed=sd, arch=arch,
                   wall_s=round(wall, 1), rss_mb=round(_rss_mb()),
                   path=os.path.basename(p), t=time.time())
        done.append(rec)
        with open(jsonl, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        print(f"[wp4a] {i}/{len(todo)} {reg} {L.dtag(dl)} {arch} seed{sd} "
              f"{wall:.0f}s rss {rec['rss_mb']} MB", flush=True)
    return done


# ===========================================================================
# Part 6 — scoring
# ===========================================================================
def _truth_dense_amp(regime, f):
    d = np.load(os.path.join(SYNTH_DIR, f"{regime}_dense.npz"), allow_pickle=True)
    t = np.asarray(d["t"], float)
    y = np.log(np.asarray(d["alphas"], float)[:, ALPHA_NAMES.index(L.SEASON_CHANNEL)])
    return L.project(t, y, f)


def part6_score(out_dir=OUT_DIR, seeds=SEEDS, arches=("pub", "tin")):
    """Recovered amplitude/phase against the bound, and the collateral damage."""
    rec_rows, dam_rows = [], []
    null = {}
    cap_p = os.path.join(out_dir, "wp4a_capacity.csv")
    cap = pd.read_csv(cap_p) if os.path.exists(cap_p) else None

    for dl in DELTAS:
        for reg in REGIMES:
            for arch in arches:
                tag = L.tag_for(reg, dl)
                cells = [s for s in seeds
                         if os.path.exists(fit_path(reg, dl, s, arch=arch))]
                if not cells:
                    continue
                fit, ds = L.fit_context(tag, cfg=scaled_config(dl, arch=arch))
                t = dense_grid(ds["years"])
                yrs = np.asarray(ds["years"], float)
                tv = yrs <= TRAINVAL_END
                tr = S.load_truth(os.path.join(SYNTH_DIR, f"{tag}.npz"))
                a_true = tr["alphas_point"]
                t_true = tr["tau_sup_point"]
                for sd in cells:
                    params = cflab.load_params(fit_path(reg, dl, sd, arch=arch))
                    A = L.dense_alpha(fit, params, t)
                    yk = np.log(A[:, ALPHA_NAMES.index(L.SEASON_CHANNEL)])
                    # --- recovery, only meaningful where a component exists
                    for (chan, f, a, ph) in L.REGIMES[reg]:
                        p = L.project(t, yk, f)
                        td = _truth_dense_amp(reg, f)
                        g = float(L.sinc_gain(f, dl))
                        nl = np.nan
                        if cap is not None:
                            sub = cap[(cap.delta.sub(dl).abs() < 1e-9)
                                      & (cap.freq_per_yr.sub(f).abs() < 1e-9)]
                            if len(sub):
                                nl = float(sub.null_amp_q3.iloc[0])
                        rec_rows.append(dict(
                            regime=reg, delta=float(dl), dtag=L.dtag(dl),
                            arch=arch, seed=int(sd), channel=chan,
                            freq_per_yr=float(f), amp_true=float(a),
                            amp_true_measured=td["amp"],
                            amp_rec=p["amp"], gain_rec=p["amp"] / max(td["amp"], 1e-15),
                            phase_err=float(np.abs(np.angle(
                                np.exp(1j * (p["phase"] - ph))))),
                            sinc_gain=g, is_null=bool(g < 1e-12),
                            above_nyquist=bool(f > 0.5 / float(dl)),
                            null_amp_q3=nl,
                            snr_over_null=p["amp"] / nl if nl and nl > 0 else np.nan))
                    # --- damage: everything the season is not
                    d = np.load(fit_path(reg, dl, sd, arch=arch), allow_pickle=True)
                    a_pred = d["alphas"]
                    t_pred = np.concatenate(
                        [d["taus"], d["frac_fu"][:, :2], d["frac_eu"][:, :2]], axis=1)
                    for k, name in enumerate(ALPHA_NAMES):
                        for split, m in (("trainval", tv), ("test", ~tv)):
                            dam_rows.append(dict(
                                regime=reg, delta=float(dl), dtag=L.dtag(dl),
                                arch=arch, seed=int(sd), family="alpha",
                                channel=name, split=split,
                                rel_rmse=rel_rmse_pct(a_pred[:, k], a_true[:, k], m)))
                    for j, name in enumerate(TAU_SUP_NAMES):
                        dam_rows.append(dict(
                            regime=reg, delta=float(dl), dtag=L.dtag(dl),
                            arch=arch, seed=int(sd), family="tau", channel=name,
                            split="trainval",
                            rel_rmse=rel_rmse_pct(t_pred[:, j], t_true[:, j], tv)))
                    for k, name in enumerate(["Concentrate", "Refined", "In-Use", "Scrap"]):
                        dam_rows.append(dict(
                            regime=reg, delta=float(dl), dtag=L.dtag(dl),
                            arch=arch, seed=int(sd), family="stock", channel=name,
                            split="trainval",
                            rel_rmse=rel_rmse_pct(d["S_pred"][:, k],
                                                  tr["stocks_clean"][:, k], tv)))
                S.disarm()
                import jax
                jax.clear_caches()

    rec = pd.DataFrame(rec_rows)
    dam = pd.DataFrame(dam_rows)
    if not rec.empty:
        rec.to_csv(os.path.join(out_dir, "wp4a_recovery_per_seed.csv"), index=False)
        g = rec.groupby(["regime", "dtag", "arch", "freq_per_yr"])
        summ = g.agg(delta=("delta", "first"), n_seeds=("seed", "nunique"),
                     amp_true=("amp_true", "first"),
                     sinc_gain=("sinc_gain", "first"),
                     is_null=("is_null", "first"),
                     above_nyquist=("above_nyquist", "first"),
                     null_amp_q3=("null_amp_q3", "first"),
                     amp_rec=("amp_rec", "median"),
                     gain_rec=("gain_rec", "median"),
                     phase_err=("phase_err", "median")).reset_index()
        lo, hi = [], []
        for _key, sub in g:
            _e, l, h = hl(sub.gain_rec.to_numpy())
            lo.append(l); hi.append(h)
        summ["gain_hl_lo"], summ["gain_hl_hi"] = lo, hi
        summ["detected"] = summ.amp_rec > summ.null_amp_q3
        summ.to_csv(os.path.join(out_dir, "wp4a_recovery.csv"), index=False)
    else:
        summ = rec

    if not dam.empty:
        dam.to_csv(os.path.join(out_dir, "wp4a_damage_per_seed.csv"), index=False)
        # paired against the no-season control at the same Delta, arch and seed
        base = dam[dam.regime.eq("flat")].set_index(
            ["dtag", "arch", "seed", "family", "channel", "split"]).rel_rmse
        rows = []
        for reg in ("harm", "nonharm"):
            sub = dam[dam.regime.eq(reg)]
            for key, grp in sub.groupby(["dtag", "arch", "family", "channel", "split"]):
                dtg, arch, fam, ch, sp = key
                d = []
                for _i, r in grp.iterrows():
                    k = (dtg, arch, r.seed, fam, ch, sp)
                    if k in base.index:
                        d.append(r.rel_rmse - float(base.loc[k]))
                est, l, h = hl(np.asarray(d))
                if not d:
                    # No matched control at this (dtag, arch): the time-input
                    # arm was run on `nonharm` only, so it has no `flat`
                    # counterpart to pair against.  Its evidence is the
                    # recovery table, not this one; an unpaired row here would
                    # read as a comparison that was never made.
                    continue
                rows.append(dict(regime=reg, dtag=dtg, arch=arch, family=fam,
                                 channel=ch, split=sp, n_pairs=len(d),
                                 delta_relrmse_hl=est, hl_lo=l, hl_hi=h,
                                 n_worse=int(np.sum(np.asarray(d) > 0))))
        pd.DataFrame(rows).to_csv(
            os.path.join(out_dir, "wp4a_damage.csv"), index=False)
    return rec, summ, dam


# ===========================================================================
# Part 7 — figures
# ===========================================================================
def part7_figures(out_dir=OUT_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = pd.read_csv(os.path.join(out_dir, "wp4a_sinc_bound.csv"))
    comp = pd.read_csv(os.path.join(out_dir, "wp4a_components.csv"))
    det = pd.read_csv(os.path.join(out_dir, "wp4a_detectability.csv"))
    rp = os.path.join(out_dir, "wp4a_recovery.csv")
    dp = os.path.join(out_dir, "wp4a_damage.csv")
    rec = pd.read_csv(rp) if os.path.exists(rp) else None
    dam = pd.read_csv(dp) if os.path.exists(dp) else None

    order = ["1y", "1o4y", "1o12y"]
    label = {"1y": "annual", "1o4y": "quarterly", "1o12y": "monthly"}
    col = {"1y": "#c1443c", "1o4y": "#d98c34", "1o12y": "#2e6fa7"}
    fig, axes = plt.subplots(1, 4, figsize=(20.5, 4.7))

    # --- 1. the closed-form bound -----------------------------------------
    ax = axes[0]
    for dl, sub in curve.groupby("delta"):
        dt = L.dtag(dl)
        ax.plot(sub.freq_per_yr, sub.sinc_gain, lw=1.7, color=col[dt],
                label=f"{label[dt]} ($\\Delta$={dl:g} yr)")
    for f in sorted(comp.freq_per_yr.unique()):
        ax.axvline(f, color="0.8", lw=0.7, ls=":", zorder=0)
        ax.text(f, 1.04, f"{f:g}", ha="center", fontsize=7.5, color="0.35")
    ax.set_xlabel("frequency (1/yr)")
    ax.set_ylabel(r"$|\mathrm{sinc}(\pi f \Delta)|$")
    ax.set_title("1. Period integration attenuates,\n"
                 r"with exact nulls at $f=k/\Delta$", fontsize=10.5)
    ax.set_ylim(0, 1.13)
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    # --- 2. the recovery threshold in amplitude ---------------------------
    ax = axes[1]
    sub = det[det.amp_nats.sub(0.12).abs() < 1e-9]
    for dt in order:
        s = sub[sub.dtag.eq(dt)].sort_values("freq_per_yr")
        amin = s.amp_min_nats.replace(np.inf, np.nan)
        ax.semilogy(s.freq_per_yr, amin, "o-", ms=4.5, lw=1.5, color=col[dt],
                    label=label[dt])
        gone = s[~np.isfinite(s.amp_min_nats)]
        ax.plot(gone.freq_per_yr, np.full(len(gone), 3.0), "x", ms=9, mew=2,
                color=col[dt], clip_on=False)
    ax.axhline(0.12, color="k", lw=1.0, ls="--")
    ax.text(0.55, 0.13, "the amplitude tested here", fontsize=7.5)
    ax.set_ylim(1e-2, 4.0)
    ax.set_xlabel("frequency (1/yr)")
    ax.set_ylabel("smallest detectable amplitude (nats)")
    ax.set_title("2. Recovery threshold\n($\\times$ = annihilated, no amplitude suffices)",
                 fontsize=10.5)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    # --- 3. what the estimator actually recovers --------------------------
    ax = axes[2]
    if rec is not None and len(rec):
        pub = rec[rec.arch.eq("pub")]
        xs = {d: i for i, d in enumerate(order)}
        for (reg, f), g in pub.groupby(["regime", "freq_per_yr"]):
            g = g.assign(x=g.dtag.map(xs)).sort_values("x")
            ax.plot(g.x, g.sinc_gain, ":", lw=1.1, color="0.6", zorder=1)
            ax.errorbar(g.x, g.gain_rec,
                        yerr=[g.gain_rec - g.gain_hl_lo, g.gain_hl_hi - g.gain_rec],
                        fmt="o-" if reg == "harm" else "s-", ms=5, lw=1.4,
                        capsize=2.5, label=f"{reg} {f:g}/yr", zorder=3)
        tin = rec[rec.arch.eq("tin")]
        if len(tin):
            ax.plot(tin.dtag.map(xs), tin.gain_rec, "k*", ms=11, zorder=4,
                    label="time-input control")
        ax.set_xticks(list(xs.values()))
        ax.set_xticklabels([label[d] for d in order])
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("recovered / true amplitude")
        ax.set_title("3. The estimator recovers none of it\n"
                     "(grey dotted: the $|\\mathrm{sinc}|$ bound)", fontsize=10.5)
        ax.legend(fontsize=7.5, frameon=False, ncol=2)
    else:
        ax.set_axis_off()

    # --- 4. what it costs the coefficient that carries it -----------------
    ax = axes[3]
    if dam is not None and len(dam):
        d = dam[dam.family.eq("alpha") & dam.channel.eq("alpha_refc")
                & dam.split.eq("trainval") & dam.arch.eq("pub")]
        xs = {dd: i for i, dd in enumerate(order)}
        for reg, g in d.groupby("regime"):
            g = g.assign(x=g.dtag.map(xs)).sort_values("x")
            ax.errorbar(g.x, g.delta_relrmse_hl,
                        yerr=[g.delta_relrmse_hl - g.hl_lo,
                              g.hl_hi - g.delta_relrmse_hl],
                        fmt="o-" if reg == "harm" else "s-", ms=6, lw=1.6,
                        capsize=3, label=reg)
        ax.axhline(10.2, color="0.4", lw=1.0, ls="--")
        ax.text(0.02, 10.5, "RMS of the oscillation the model cannot represent",
                fontsize=7.5, color="0.35")
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_xticks(list(xs.values()))
        ax.set_xticklabels([label[dd] for dd in order])
        ax.set_ylabel(r"$\alpha_{refc}$ relRMSE penalty (pp)")
        ax.set_title("4. Finer observation makes it worse,\nnot better", fontsize=10.5)
        ax.legend(fontsize=8, frameon=False)
    else:
        ax.set_axis_off()

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"wp4a_bound.{ext}"), dpi=170)
    plt.close(fig)
    return True


# ===========================================================================
def check(verbose=True):
    info = L.check(verbose=verbose)
    cells = plan()
    have = sum(os.path.exists(fit_path(*c[:3], arch=c[3])) for c in cells)
    if verbose:
        print(f"  fit plan             : {len(cells)} cells "
              f"({len(REGIMES)} regimes x {len(DELTAS)} widths x {len(SEEDS)} "
              f"seeds, + {len(SEEDS)} time-input control)")
        print(f"  on disk              : {have}  |  to run: {len(cells)-have}")
        for dl in DELTAS:
            c = scaled_config(dl)
            print(f"    Delta={dl:7.4f} [{L.dtag(dl):6s}] curriculum "
                  f"{c['stageB_curriculum']}")
        print("=" * 74)
    info["n_cells"], info["n_on_disk"] = len(cells), have
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description="WP-4a attenuation and aliasing")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--parts", default="")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--no-time-input", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args(argv)

    integrity_check()
    if args.check or (not args.parts and not args.fit and not args.all):
        check()
        if not (args.parts or args.fit or args.all):
            return 0

    parts = ([int(x) for x in args.parts.split(",") if x.strip()]
             if args.parts else ([1, 2, 3, 4, 5, 6, 7] if args.all else []))

    if 1 in parts:
        curve, comp = part1_bound()
        print(f"[part1] bound curve {len(curve)} rows, components {len(comp)} rows")
    if 2 in parts:
        gain, obs, fate = part2_survives()
        print(f"[part2] boxcar gain {len(gain)} rows, annihilation {len(obs)} "
              f"rows, sampling fate {len(fate)} rows")
    if 3 in parts:
        det = part3_detectability()
        print(f"[part3] detectability {len(det)} rows")
    if 4 in parts:
        cap = part4_capacity()
        print(f"[part4] capacity {len(cap)} rows")
    if 5 in parts or args.fit:
        cells = plan(with_time_input=not args.no_time_input)
        if args.n_shards > 1:
            cells = cells[args.shard::args.n_shards]
        part5_fit(cells)
    if 6 in parts:
        rec, summ, dam = part6_score()
        print(f"[part6] recovery {len(rec)} rows, damage {len(dam)} rows")
    if 7 in parts:
        part7_figures()
        print("[part7] figures written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
