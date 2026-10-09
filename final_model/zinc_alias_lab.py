#!/usr/bin/env python3
"""
zinc_alias_lab.py — lab module for WP-4a: attenuation and aliasing
==================================================================

WP-4a asks what annual reporting does to sub-annual structure, and how close
an estimator can get to the limit that imposes.  The observation operator for
flows is a **period integral**, i.e. a boxcar of width `Delta`, whose transfer
function is `|sinc(pi f Delta)|` with *exact nulls* at `f = k/Delta`.  So the
question is not the naive Nyquist one:

  * components at exactly `1/yr, 2/yr, ...` are **annihilated** by annual
    integration, not aliased.  No estimator can recover them, because the
    observation vector does not depend on them.
  * components at other frequencies are **attenuated** by `|sinc|` and only
    then aliased by the sampling.  They are recoverable in principle; whether
    they are recoverable in practice is an SNR question.

Pattern (CLAUDE.md rule 1)
--------------------------
`zinc_colloc_v5.py` is imported and never edited.  This module owns no patch
of its own: the twin reaches the training pipeline through
`zinc_synth_lab`'s `load_zinc_data` dispatcher, and `install()` simply
forwards to it, so `PATCHES` is that module's list and nothing else is
rebound.  The MD5s of the core and of the `anchor_v4` data copy are pinned
through `zinc_alpha_lab.integrity_check`.

The two regimes (pre-registered, frozen before any fit)
-------------------------------------------------------
Both are the two halves of WP-3's `season` arm, kept verbatim — same channel
(`alpha_refc`), same amplitudes, same phases — so WP-4a is continuous with the
boxcar verification already reported in `analysis/wp3_boxcar_gain.csv`:

    harm     alpha_refc  1.00/yr  a=0.12 nats  phase 0
                         2.00/yr  a=0.08 nats  phase pi/2
    nonharm  alpha_refc  1.35/yr  a=0.12 nats  phase pi/3
                         2.70/yr  a=0.08 nats  phase 5pi/4
    flat     (no sub-annual component; the control)

`flat` is WP-3's `base` arm regenerated under this module's code path;
`check()` verifies the two datasets are bit-identical at `Delta = 1`, which is
what licenses reusing WP-3's eight `base_d1y_clean` fits as this package's
annual control.

Amplitudes are in **nats of log alpha**, because the truth enters as
`alpha = A exp(trend + drivers + a sin(2 pi f (t - t0) + phi))`.  That has a
consequence WP-4a has to measure rather than assume: `E[exp(a sin)] = I_0(a)`,
so a zero-mean oscillation in log space carries a **non-zero mean effect** of
`log I_0(a) ~ a^2/4` on the level.  Annual integration annihilates the
oscillation; it does not annihilate its rectification.  See `rectification()`.

What this module provides
-------------------------
* `generate()`     — the three regimes at each Delta, through `zinc_synth_lab`'s
                     own operators (no reimplementation).
* `sinc_gain()`, `alias_freq()`, `nulls()` — the closed-form benchmark.
* `project()`      — least-squares amplitude/phase of a component at frequency
                     `f`, against a spline trend basis too coarse to alias into
                     it.  Applied identically to truth and to estimate.
* `dense_alpha()`  — `alpha_k(t)` on an arbitrary grid from a stored Stage-B
                     weight dump, evaluated the way the integrator evaluates it
                     (features linearly interpolated by `v5.exog_fn`).
* `feature_power()` — power of the model's *input* features at a frequency.
                     Under `anchor_v4` (`use_time_input: false`,
                     `use_stock_input: false`) every input is an interpolant of
                     an annual driver, so this is the architectural floor on
                     what alpha can represent, and it is independent of Delta.
* `detectability()` — the observable trace `A |sinc(pi f Delta)|` against the
                     WP-3 calibrated noise, i.e. the recovery threshold as a
                     function of frequency and amplitude, with no fits.

Run `python zinc_alias_lab.py --check` first: it prints the resolved driver
names and `input_dim` (CLAUDE.md conventions) and the no-op guard.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import zinc_synth_lab as S                                       # noqa: E402
import zinc_cf_lab as cflab                                      # noqa: E402
from zinc_alpha_lab import integrity_check, load_anchor_config    # noqa: F401,E402

OUT_DIR_DEFAULT = os.path.join(HERE, "analysis", "synth_wp4a")
FIT_DIR_DEFAULT = os.path.join(HERE, "analysis", "synth_wp4a", "fits")
WP3_SYNTH_DIR = os.path.join(HERE, "analysis", "synth")
WP3_FIT_DIR = os.path.join(HERE, "analysis", "synth", "fits")

ALPHA_NAMES = S.ALPHA_NAMES
TAU_SUP_NAMES = S.TAU_SUP_NAMES
SEASON_CHANNEL = "alpha_refc"
SEASON_T0 = float(S.TRUTH["season_t0"])

# ---------------------------------------------------------------------------
# pre-registered design.  Frozen before any fit; do not edit after Part 5 runs.
# ---------------------------------------------------------------------------
REGIMES: dict[str, list[tuple]] = {
    "flat":    [],
    "harm":    [(SEASON_CHANNEL, 1.00, 0.12, 0.0),
                (SEASON_CHANNEL, 2.00, 0.08, np.pi / 2)],
    "nonharm": [(SEASON_CHANNEL, 1.35, 0.12, np.pi / 3),
                (SEASON_CHANNEL, 2.70, 0.08, 5 * np.pi / 4)],
}

# annual / quarterly / monthly -- the three the spec names.  n_sub = 96 is
# divisible by 1, 4 and 12, so each is an exact re-aggregation of one dense
# solve rather than a separate integration (`_window_indices` enforces it).
DELTAS: tuple[float, ...] = (1.0, 0.25, 1.0 / 12.0)

SEEDS = tuple(range(8))

# Trend basis for `project`: cubic B-spline, knots every TREND_KNOT_YR years.
# A cubic spline on 2-year knots cannot represent anything above ~0.25/yr, so
# it removes the tanh trend and the driver term without absorbing any part of
# a component at 1/yr or above.  Verified in `check()`.
TREND_KNOT_YR = 2.0

PATCHES: list[str] = S.PATCHES


# ---------------------------------------------------------------------------
# install / check
# ---------------------------------------------------------------------------
def install(v5mod=None):
    """No patch of this module's own; forward to the twin dispatcher."""
    return S.install(v5mod)


def dtag(delta):
    """Filename tag for a window width.  `1y`, `0p25y`, `1o12y`."""
    d = float(delta)
    if abs(d - round(d)) < 1e-12:
        return f"{int(round(d))}y"
    inv = 1.0 / d
    if abs(inv - round(inv)) < 1e-9:
        return f"1o{int(round(inv))}y"
    return f"{d:.4f}".rstrip("0").rstrip(".").replace(".", "p") + "y"


def tag_for(regime, delta, noisy=False):
    return f"{regime}_d{dtag(delta)}_{'noisy' if noisy else 'clean'}"


# ---------------------------------------------------------------------------
# the closed-form benchmark
# ---------------------------------------------------------------------------
def sinc_gain(f, delta):
    """`|sinc(pi f Delta)|` — the boxcar (period-integral) transfer function.

    A period integral of width `Delta` applied to `exp(2 pi i f t)` returns
    `Delta * sinc(pi f Delta) * exp(...)`, so the gain of the window-*mean*
    rate is `|sin(pi f Delta) / (pi f Delta)|`, exactly zero at `f = k/Delta`.
    Identical to `run_wp3.sinc_gain`; repeated here so this module stands
    alone, and cross-checked against it in `check()`.
    """
    x = np.pi * np.asarray(f, float) * float(delta)
    return np.abs(np.where(np.abs(x) < 1e-12, 1.0,
                           np.sin(x) / np.where(x == 0.0, 1.0, x)))


def alias_freq(f, delta):
    """Where a component at `f` lands after sampling at spacing `Delta`.

    Sampling folds the axis about multiples of the Nyquist rate `1/(2 Delta)`;
    the returned frequency is in `[0, 1/(2 Delta)]`.  Meaningful only where
    `sinc_gain` is non-zero: at an exact null nothing is left to alias, which
    is the whole point of the harmonic regime.
    """
    f = np.asarray(f, float)
    fs = 1.0 / float(delta)
    r = np.mod(f, fs)
    return np.where(r > fs / 2.0, fs - r, r)


def nulls(delta, f_max=6.0):
    """Frequencies annihilated by a width-`Delta` period integral, up to `f_max`."""
    k = np.arange(1, int(np.floor(f_max * float(delta))) + 1)
    return k / float(delta)


def rectification(a):
    """Mean log-level shift of `exp(a sin(.))`: `log I_0(a)`, ~ `a^2/4`.

    The component the boxcar annihilates is the *oscillation*.  Its rectified
    mean survives, because the truth is exponential in log-alpha, and it is
    indistinguishable from a change in the trend.  This is what makes the
    harmonic regime worse than merely unrecoverable.
    """
    from numpy import i0
    a = np.asarray(a, float)
    return np.log(i0(a))


# ---------------------------------------------------------------------------
# component projection
# ---------------------------------------------------------------------------
def _trend_basis(t, knot_yr=TREND_KNOT_YR):
    """Cubic B-spline design matrix on `knot_yr`-spaced knots."""
    from scipy.interpolate import BSpline
    t = np.asarray(t, float).ravel()
    lo, hi = float(t[0]), float(t[-1])
    n_int = max(int(round((hi - lo) / float(knot_yr))), 1)
    inner = lo + (hi - lo) * np.arange(1, n_int) / n_int
    knots = np.concatenate([np.full(4, lo), inner, np.full(4, hi)])
    return np.asarray(BSpline.design_matrix(
        np.clip(t, lo, hi), knots, 3, extrapolate=False).todense())


def project(t, y, f, *, t0=SEASON_T0, knot_yr=TREND_KNOT_YR, return_fit=False):
    """Least-squares amplitude and phase of a component at frequency `f`.

    Model: `y(t) = spline_trend(t) + c cos(w) + s sin(w)`, `w = 2 pi f (t-t0)`.
    Returns `dict(amp, phase, c, s)` with `amp = hypot(c, s)` in the units of
    `y` (nats, when `y` is `log alpha`) and `phase` the phase of the *sine*
    convention the truth is written in, so a perfect recovery of a component
    entered as `a sin(w + phi)` returns `amp = a`, `phase = phi`.

    The spline trend is deliberately too coarse to represent anything near
    `f` (`check()` verifies the leakage), so this separates the sub-annual
    component from the tanh trend and the driver term without knowing either.
    """
    t = np.asarray(t, float).ravel()
    y = np.asarray(y, float).ravel()
    m = np.isfinite(y)
    w = 2.0 * np.pi * float(f) * (t - float(t0))
    X = np.column_stack([_trend_basis(t, knot_yr), np.cos(w), np.sin(w)])
    beta, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
    c, s = float(beta[-2]), float(beta[-1])
    out = dict(amp=float(np.hypot(c, s)), phase=float(np.arctan2(c, s)),
               c=c, s=s)
    if return_fit:
        out["fitted"] = X @ beta
        out["trend"] = X[:, :-2] @ beta[:-2]
    return out


def component_table(t, y, comps, **kw):
    """`project` over a regime's component list; returns a list of dicts."""
    rows = []
    for (chan, f, a, ph) in comps:
        p = project(t, y, f, **kw)
        rows.append(dict(channel=chan, freq_per_yr=float(f), amp_true=float(a),
                         phase_true=float(ph), amp=p["amp"], phase=p["phase"]))
    return rows


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
def _verify_solver(ctx, dense, comps, n_sub_ref=192):
    """Solver-convergence check at the *right* truth.

    `zinc_synth_lab.verify` re-solves with `dense_solve(ctx, dense["arm"])` and
    passes no `season`, so for an arm named `season` its reference solve falls
    back to `TRUTH["season"]` — all four of WP-3's components — rather than the
    regime under test.  That is correct for WP-3, which only ever ran the full
    `season` list, and silently wrong for any module that drives the component
    list itself.  Flagged in the findings note; worked around here rather than
    edited, because the twin generator is shared with WP-3/4b/5/11e.
    """
    arm_name = "season" if comps else "base"
    ref = S.dense_solve(ctx, arm_name, n_sub=n_sub_ref, season=comps)
    i2 = np.searchsorted(ref["t"], dense["t"])
    return dict(
        solver_stock_rel=float(np.max(
            np.abs(ref["S4"][i2] - dense["S4"])
            / np.maximum(np.abs(dense["S4"]), 1.0))),
        solver_flow_rel=float(np.max(
            np.abs(ref["C"][i2] - dense["C"])
            / np.maximum(np.abs(dense["C"]), 1.0))))


def generate(regimes=tuple(REGIMES), deltas=DELTAS, *, out_dir=OUT_DIR_DEFAULT,
             cfg=None, noise_scale=0.0, rng_seed=0, verbose=True):
    """Every (regime, Delta) dataset, with its ground truth alongside.

    Mirrors `zinc_synth_lab.generate` but drives the season component list
    explicitly instead of taking it from `TRUTH['season']`, which is the only
    thing WP-4a needs that WP-3's generator does not expose.  Every operator,
    solve and truth evaluation is `zinc_synth_lab`'s own.
    """
    os.makedirs(out_dir, exist_ok=True)
    ctx = S.driver_context(cfg)
    sigma = S.calibrate_noise()
    report = {"regimes": {}, "deltas": [float(d) for d in deltas],
              "n_sub": S.N_SUB, "trend_knot_yr": TREND_KNOT_YR,
              "season_t0": SEASON_T0,
              "components": {k: [[c[0], float(c[1]), float(c[2]), float(c[3])]
                                 for c in v] for k, v in REGIMES.items()},
              "sigma": {"stock": sigma["stock"].tolist(),
                        "flow": sigma["flow"].tolist(),
                        "flow_names": sigma["flow_names"],
                        "n_seeds": sigma["n_seeds"]}}

    for reg in regimes:
        comps = REGIMES[reg]
        arm_name = "season" if comps else "base"
        t0 = time.time()
        dense = S.dense_solve(ctx, arm_name, season=comps)
        samples = {float(dl): S.sample(dense, dl, ctx, noise_scale=0.0)
                   for dl in deltas}
        v = S.verify(ctx, dense, samples)
        v.update(_verify_solver(ctx, dense, comps))
        r = S.realism(ctx, dense)
        tc = S.truth_coefficients(ctx, dense["t"], arm_name, S_path=dense["S4"],
                                  season=comps)
        np.savez_compressed(
            os.path.join(out_dir, f"{reg}_dense.npz"),
            t=dense["t"], S4=dense["S4"], S_cohorts=dense["S_cohorts"],
            C=dense["C"], alphas=tc["alphas"], tau_sup=tc["tau_sup"],
            f_cohort=tc["f_cohort"], cp=tc["cp"],
            alpha_names=np.asarray(ALPHA_NAMES, dtype=object),
            tau_sup_names=np.asarray(TAU_SUP_NAMES, dtype=object),
            meta_json=np.asarray(json.dumps(dict(
                regime=reg, arm=arm_name, n_sub=S.N_SUB, verify=v,
                components=[[c[0], float(c[1]), float(c[2]), float(c[3])]
                            for c in comps])), dtype=object))
        if verbose:
            print(f"[{reg}] dense solve {time.time()-t0:5.1f}s  "
                  f"mass {v['mass_conservation_rel']:.2e}  "
                  f"solver S {v['solver_stock_rel']:.2e}", flush=True)

        rep = {"verify": v, "realism": {k: np.asarray(x).tolist()
                                        for k, x in r.items()}, "datasets": {}}
        for dl in deltas:
            for noisy in ((False, True) if noise_scale > 0 else (False,)):
                ns = float(noise_scale) if noisy else 0.0
                s = (samples[float(dl)] if not noisy else
                     S.sample(dense, dl, ctx, noise_scale=ns,
                              rng_seed=rng_seed, sigma=sigma))
                ds = S.build_dataset(ctx, s, dl)
                tw = S.truth_coefficients(ctx, s["years"], arm_name,
                                          S_path=s["stocks_clean"], season=comps)
                bias = S.alpha_target_bias(ctx, dense, dl, arm_name=arm_name,
                                           season=comps)
                truth = dict(
                    alphas_point=tw["alphas"], tau_sup_point=tw["tau_sup"],
                    f_cohort_point=tw["f_cohort"],
                    alphas_window_unweighted=bias["a_unweighted"],
                    alphas_window_weighted=bias["a_weighted"],
                    alphas_window_obs=bias["a_obs"],
                    stocks_clean=s["stocks_clean"], F_int_clean=s["F_int_clean"],
                    window_years=bias["years"],
                    alpha_names=np.asarray(ALPHA_NAMES, dtype=object),
                    tau_sup_names=np.asarray(TAU_SUP_NAMES, dtype=object))
                tag = tag_for(reg, dl, noisy)
                p = S.save_dataset(
                    os.path.join(out_dir, f"{tag}.npz"), ds, truth=truth,
                    meta=dict(regime=reg, arm=arm_name, delta=float(dl),
                              noise_scale=ns, rng_seed=int(rng_seed),
                              n_sub=S.N_SUB, T=int(ds["years"].size),
                              components=[[c[0], float(c[1]), float(c[2]),
                                           float(c[3])] for c in comps],
                              season_t0=SEASON_T0))
                rep["datasets"][tag] = dict(path=os.path.basename(p),
                                            delta=float(dl), noise=ns,
                                            T=int(ds["years"].size))
        report["regimes"][reg] = rep
        if verbose:
            print(f"[{reg}] wrote {len(rep['datasets'])} datasets  "
                  f"rss {S._rss_mb():.0f} MB", flush=True)
        import jax
        jax.clear_caches()

    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    return report


# ---------------------------------------------------------------------------
# the estimator side: alpha(t) from a stored weight dump
# ---------------------------------------------------------------------------
def fit_context(tag, *, cfg=None, synth_dir=OUT_DIR_DEFAULT, seed=0):
    """Arm dataset `tag` and build the zero-step fit its weights belong to.

    Returns `(fit, data_all)`.  Unlike `zinc_interp_lab.ude_context` this does
    not rebuild the exog features — `dense_alpha` reads the fit's own
    `data_all["exog_values"]`, which is by construction the matrix the weights
    were trained against, so there is nothing to cross-check.
    """
    import zinc_colloc_v5 as v5
    install(v5)
    cfg = dict(cfg or load_anchor_config())
    ds = S.load_dataset(os.path.join(synth_dir, f"{tag}.npz"))
    S.arm(ds)
    fit = cflab.init_fit(cfg, seed=seed)
    return fit, ds


def dense_alpha(fit, params, t_grid, *, stocks=None):
    """`alpha_k(t)` on `t_grid` from frozen weights, as the integrator sees it.

    `v5.exog_fn` interpolates the *feature* matrix linearly between observation
    nodes, which is exactly what the RHS does at intermediate solver times.  So
    this is the model's own alpha path, not a resampling of a node-level one,
    and its spectrum is the spectrum of what the fitted model actually asserts
    about the world between reporting dates.

    Under `anchor_v4` alpha does not read the state (`use_stock_input: false`),
    so `stocks` is immaterial; it is carried for the `use_time_input` control
    arm and defaults to the observed stocks interpolated onto `t_grid`.
    """
    import jax
    import jax.numpy as jnp
    import zinc_colloc_v5 as v5

    data = dict(fit.data_all)
    t_grid = np.asarray(t_grid, float).ravel()
    years = np.asarray(fit.data_all["years"], float).ravel()
    if stocks is None:
        S_obs = np.asarray(fit.data_all["stocks_obs"], float)
        stocks = np.column_stack([np.interp(t_grid, years, S_obs[:, k])
                                  for k in range(S_obs.shape[1])])
    stocks = np.asarray(stocks, float)

    def one(t, S4):
        ex = v5.exog_fn(t, data["exog_times"], data["exog_values"])
        _cp, alphas, _tau, _fc, _raw = fit.nn_eval_sup(params, t, S4, ex, data)
        return alphas

    return np.asarray(jax.vmap(one)(jnp.asarray(t_grid), jnp.asarray(stocks)))


def feature_power(fit, f, *, t_grid=None, t0=SEASON_T0):
    """Per-feature projected amplitude at frequency `f`, in feature units.

    The architectural floor.  `alpha` is a deterministic function of the input
    vector; if no input carries power at `f`, no weight setting can make
    `alpha` carry power at `f`.  Under `anchor_v4` the only live inputs are the
    exog features, and `v5.exog_fn` makes them piecewise-linear interpolants of
    an *annual* driver grid whatever `Delta` is, so this is expected to be at
    the interpolation-kink level and independent of the observation frequency.
    """
    import zinc_colloc_v5 as v5
    import jax.numpy as jnp

    data = fit.data_all
    years = np.asarray(data["years"], float).ravel()
    if t_grid is None:
        t_grid = np.linspace(years[0], years[-1], int((years[-1] - years[0]) * 96) + 1)
    t_grid = np.asarray(t_grid, float).ravel()
    X = np.asarray(jnp.stack([v5.exog_fn(float(t), data["exog_times"],
                                         data["exog_values"]) for t in t_grid]))
    amps = np.array([project(t_grid, X[:, j], f, t0=t0)["amp"]
                     for j in range(X.shape[1])])
    scale = np.maximum(X.std(axis=0), 1e-12)
    return dict(amp=amps, rel=amps / scale, X=X, t=t_grid)


# ---------------------------------------------------------------------------
# detectability: the recovery threshold with no fits
# ---------------------------------------------------------------------------
def detectability(freqs, deltas, amps, *, sigma_log=None, snr=1.0):
    """Rows of `A |sinc(pi f Delta)|` against the observation noise floor.

    `sigma_log` is the log-scale noise on the channel the component rides on.
    It defaults to the WP-3 calibrated total for the `alpha_refc` *target*
    (its parent stock plus the flow it is built from, in quadrature), because
    that is the series in which a would-be detector would look for the
    oscillation.  `A_min` is the amplitude at which the surviving trace equals
    `snr` noise SDs — the recovery threshold as a function of frequency and
    window width, and it is estimator-independent.
    """
    if sigma_log is None:
        sigma_log = alpha_target_sigma()
    rows = []
    for dl in deltas:
        for f in freqs:
            g = float(sinc_gain(f, dl))
            for a in amps:
                trace = float(a) * g
                rows.append(dict(
                    delta=float(dl), freq_per_yr=float(f), amp_nats=float(a),
                    sinc_gain=g, is_null=bool(g < 1e-12),
                    alias_freq_per_yr=float(alias_freq(f, dl)),
                    nyquist_per_yr=0.5 / float(dl),
                    trace_nats=trace, sigma_log=float(sigma_log),
                    snr=trace / float(sigma_log),
                    amp_min_nats=(float(snr) * float(sigma_log) / g
                                  if g > 1e-12 else np.inf),
                    rectification_nats=float(rectification(a))))
    return rows


def alpha_target_sigma(channel=SEASON_CHANNEL):
    """Log-scale noise on an alpha target, from the WP-3 calibration.

    `alpha_obs = F_int / trapz(S)`, so its log noise is the parent flow's and
    the parent stock's in quadrature.  Channels and their parents follow
    `zinc_synth_lab.sample`'s construction: `alpha_refc` is
    `refined_consumption / S_ref`.
    """
    import zinc_colloc_v5 as v5
    sig = S.calibrate_noise()
    parents = {"alpha_cc": (0, "concentrate_consumption"),
               "alpha_refc": (1, "refined_consumption"),
               "alpha_win": (3, "waelz_input"),
               "alpha_dr": (3, "direct_reuse_recycling")}
    k, fname = parents[channel]
    s_stock = float(np.asarray(sig["stock"], float)[k])
    names = list(sig["flow_names"])
    s_flow = float(np.asarray(sig["flow"], float)[names.index(fname)]) \
        if fname in names else 0.0
    return float(np.hypot(s_stock, s_flow))


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------
def check(verbose=True):
    """Resolve drivers and `input_dim`, and prove every precondition WP-4a
    stands on, before any dataset is written or any fit is run."""
    import zinc_colloc_v5 as v5

    info = S.check(verbose=False)
    info["patches"] = list(PATCHES)

    # 1. the closed form agrees with WP-3's independent implementation
    import run_wp3
    ff = np.array([0.3, 1.0, 1.35, 2.0, 2.7, 4.0])
    gaps = [float(np.max(np.abs(sinc_gain(ff, dl) - run_wp3.sinc_gain(ff, dl))))
            for dl in DELTAS]
    info["sinc_vs_wp3_max_abs"] = float(max(gaps))

    # 2. the nulls are where they are claimed to be
    info["nulls"] = {dtag(dl): nulls(dl).tolist() for dl in DELTAS}
    info["component_gain"] = {
        reg: {f"{f:g}/yr": {dtag(dl): float(sinc_gain(f, dl)) for dl in DELTAS}
              for (_c, f, _a, _p) in comps}
        for reg, comps in REGIMES.items() if comps}

    # 3. the trend basis cannot represent any component -- if it could, the
    #    projection would absorb the oscillation into the trend and report a
    #    spurious zero.  Leakage = the amplitude a pure component loses to the
    #    spline, measured on the dense grid the projections use.
    t = np.arange(1980.0, 2019.0 + 1e-9, 1.0 / 96.0)
    leak = {}
    for (_c, f, a, ph) in REGIMES["harm"] + REGIMES["nonharm"]:
        y = a * np.sin(2.0 * np.pi * f * (t - SEASON_T0) + ph)
        p = project(t, y, f)
        leak[f"{f:g}/yr"] = dict(amp=p["amp"], amp_err=abs(p["amp"] - a) / a,
                                 phase_err=float(np.abs(np.angle(
                                     np.exp(1j * (p["phase"] - ph))))))
    info["projection_leakage"] = leak
    info["projection_max_amp_err"] = float(max(v["amp_err"] for v in leak.values()))

    # 4. the flat regime must reproduce WP-3's `base` arm bit-for-bit at
    #    Delta = 1, which is what licenses reusing WP-3's eight fits.
    gap = None
    a_p = os.path.join(OUT_DIR_DEFAULT, tag_for("flat", 1.0))
    b_p = os.path.join(WP3_SYNTH_DIR, "base_d1y_clean.npz")
    if os.path.exists(a_p + ".npz") and os.path.exists(b_p):
        A, B = S.load_dataset(a_p + ".npz"), S.load_dataset(b_p)
        gap = 0.0
        for k, va in A.items():
            if isinstance(va, np.ndarray) and va.dtype.kind in "fiu":
                gap = max(gap, float(np.nanmax(np.abs(
                    np.asarray(B[k], float) - np.asarray(va, float)))))
    info["flat_vs_wp3_base_max_abs"] = gap

    # 5. rectification: the level shift the boxcar does *not* remove
    info["rectification_nats"] = {f"a={a:g}": float(rectification(a))
                                  for a in (0.08, 0.12)}
    info["alpha_target_sigma_log"] = alpha_target_sigma()

    if verbose:
        print("=" * 74)
        print("zinc_alias_lab --check   (WP-4a: attenuation and aliasing)")
        print("=" * 74)
        for label, (got, ok) in info["digests"].items():
            print(f"  {label:20s} md5 {got}  {'OK' if ok else 'MISMATCH'}")
        print(f"  patches applied      : {info['patches']}")
        print(f"  dispatcher identity  : max |delta| = "
              f"{info['dispatcher_identity_max_abs']:.3e}  "
              f"({'inert when disarmed' if info['dispatcher_identity_max_abs'] == 0.0 else 'NOT INERT'})")
        print(f"\n  resolved drivers (canonical order, {info['n_universe']}):")
        for i, c in enumerate(info["exog_cols"]):
            print(f"    [{i:2d}] {c}")
        print(f"\n  exog_feature_orders  : {info['orders']}")
        print(f"  input_dim            : 1 (t) + 4 (S) + "
              f"{info['n_exog_features']} = {info['input_dim']}")
        if info["input_dim"] != 23:
            print(f"  !! input_dim is {info['input_dim']}, not the 23 asserted "
                  f"by CLAUDE.md rule 2 (SCHEMA §9 flag 1, unresolved).")
        print(f"\n  regimes              :")
        for reg, comps in REGIMES.items():
            if not comps:
                print(f"    {reg:8s} (control, no sub-annual component)")
                continue
            print(f"    {reg:8s} " + ", ".join(
                f"{f:g}/yr a={a:g} phi={p:.3f}" for (_c, f, a, p) in comps))
        print(f"  window widths (yr)   : " +
              ", ".join(f"{d:g} [{dtag(d)}]" for d in DELTAS))
        print(f"  seeds                : {list(SEEDS)}")
        print(f"\n  boxcar nulls (f = k/Delta, /yr):")
        for dl in DELTAS:
            print(f"    Delta={dl:7.4f} [{dtag(dl):6s}] -> "
                  f"{np.round(nulls(dl), 3).tolist()}")
        print(f"\n  closed-form gain |sinc(pi f Delta)| at each component:")
        print(f"    {'component':>12s}  " +
              "  ".join(f"{dtag(d):>8s}" for d in DELTAS))
        for reg, comps in REGIMES.items():
            for (_c, f, _a, _p) in comps:
                print(f"    {f:9g}/yr  " + "  ".join(
                    f"{sinc_gain(f, d):8.5f}" for d in DELTAS))
        print(f"\n  sinc vs run_wp3      : max |delta| = "
              f"{info['sinc_vs_wp3_max_abs']:.3e}  (must be 0)")
        print(f"  projection leakage   : max amplitude error = "
              f"{info['projection_max_amp_err']:.3e}  "
              f"(cubic spline, {TREND_KNOT_YR:g}-yr knots)")
        print(f"  flat vs WP-3 base    : "
              + ("not generated yet" if gap is None
                 else f"max |delta| = {gap:.3e}  "
                      f"({'bit-identical' if gap == 0.0 else 'DIFFERS'})"))
        print(f"  rectification log I0 : " + ", ".join(
            f"{k} -> {v:.5f} nats" for k, v in info["rectification_nats"].items()))
        print(f"  alpha_refc target sd : {info['alpha_target_sigma_log']:.4f} "
              f"nats (WP-3 calibration, stock + flow in quadrature)")
        print("=" * 74)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description="WP-4a attenuation and aliasing")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--regimes", default=",".join(REGIMES))
    ap.add_argument("--deltas", default=",".join(repr(d) for d in DELTAS))
    ap.add_argument("--out", default=OUT_DIR_DEFAULT)
    args = ap.parse_args(argv)

    if args.check or not args.generate:
        check()
        if not args.generate:
            return 0
    import zinc_colloc_v5 as v5
    integrity_check()
    install(v5)
    generate([r.strip() for r in args.regimes.split(",") if r.strip()],
             [float(eval(d)) for d in args.deltas.split(",") if d.strip()],
             out_dir=args.out)
    check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
