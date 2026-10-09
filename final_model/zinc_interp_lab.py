#!/usr/bin/env python3
"""
zinc_interp_lab.py — lab module for WP-5: the interpretability head-to-head
==========================================================================

WP-5 converts "our model is more interpretable" from assertion into
measurement.  The spec fixes the design and this module supplies the
machinery:

    1. Quantity          alpha_k(t)  and  d alpha_k / d driver
    2. Ground truth      from WP-3 (`zinc_synth_lab.TRUTH`) — analytic
    3. Two estimators    the UDE, which outputs alpha directly;
                         a per-flow AR/ARX ensemble, whose alpha is the
                         derived ratio  alpha_hat = F_hat / exposure(S_obs)
    4. Score             bias, variance, RMSE against alpha_true, and the
                         sign/magnitude accuracy of the driver response

Both estimators are legitimate readings of the same quantity, and they are
handed the *same* dataset: the WP-3 twin at Delta = 1 yr, N = 40, with the
noise calibrated from the real fit's Stage B residuals.  That is what makes
the comparison like-for-like rather than rhetorical.

Pattern (CLAUDE.md rule 1)
--------------------------
`zinc_colloc_v5.py` is imported and never edited.  This module installs no
patch of its own: the one patch WP-5 needs — the arm-aware `load_zinc_data`
dispatcher — already exists in `zinc_synth_lab`, and re-implementing it here
would give the twin two entry points that could drift apart.  `install()`
therefore delegates, and `PATCHES` reports what `zinc_synth_lab` installed.

The ground truth of the driver response, and why it is exact
------------------------------------------------------------
The twin's alpha is

    alpha_k(t) = A_k exp( m_k (tanh((t - t*_k)/w_k) - c_k)
                          + sum_j beta_{k,j} z_j(t) )

with `z_j` the log1p'd driver level, z-scored over 1980-2019.  So

    d log alpha_k / d z_j  ==  beta_{k,j}      exactly, and constant in t.

`TRUTH_BETA` reads those betas out of `zinc_synth_lab.TRUTH` **by driver
name** through `DRIVER_ALIASES` (CLAUDE.md rule 2), never by position, and
`truth_response()` re-derives the same matrix numerically from the truth
closure as a check.  Eight of the 52 (channel, driver) entries are non-zero,
so the ground truth is *sparse*: an estimator can be scored not only on the
sign and size of the responses that exist but on how many it invents.

One perturbation, applied identically to all three
--------------------------------------------------
The response is measured by a **permanent** shift of one driver's entire
observed history by `delta` in the truth's own z-coordinate,

    X'_j = expm1( log1p(X_j) + delta * sigma_j ),   sigma_j = SD(log1p(X_j))

which shifts `z_j(t)` by exactly `delta` at every t and leaves every
first-difference feature untouched.  The response is
`mean_t [ log alpha_hat(perturbed) - log alpha_hat(base) ] / delta` on the
trainval window.  For the truth this returns `beta_{k,j}` for any `delta`;
for the two estimators it is whatever they learned.  Because the same
perturbation of the same observable series drives all three, the three
matrices are directly comparable.

The AR estimator is given every advantage
-----------------------------------------
The comparator is WP-1f's ensemble — one independent ARX per flow, lag order
and driver subset by AIC — reused rather than rewritten (`run_wp1f.fit_arx_one`
is imported).  Two deliberately generous choices:

*   **Accuracy** is scored on the AR's *one-step-ahead fitted values* inside
    the estimation window (observed lags, not simulated ones), which is the
    most accurate in-sample path it can produce, and on WP-1f's dynamic
    forecast over the pre-registered test window.
*   **The driver response** is instead read off a *dynamic simulation from
    the start of the record*, so a permanent driver shift propagates through
    the lag polynomial and the AR is credited with its full long-run
    multiplier rather than a one-step partial derivative.  `ar_long_run()`
    reports the closed-form `gamma / (1 - sum phi)` alongside as a check.

Using the more favourable construction for each purpose is intentional: any
gap that survives is not an artefact of how the comparator was set up.

`alpha_hat = F_hat / exposure` uses `_build_empirical_alphas`' own trapezoid
exposure (v5:864), so the AR modeller's alpha is built exactly the way the
MFA modeller's `alpha_obs` is — only the numerator changes.  That is what
makes the denominator-noise hypothesis testable: `ar_alpha_path` accepts a
`stocks` override, so the same AR flows can be divided by the noisy observed
stock or by the twin's clean one and the difference attributed.

CLI
---
    python zinc_interp_lab.py --check
    python zinc_interp_lab.py --replicates 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import zinc_synth_lab as S
import zinc_cf_lab as cflab
from zinc_alpha_lab import integrity_check, load_anchor_config      # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
SYNTH_DIR = os.path.join(HERE, "analysis", "synth")
FIT_DIR = os.path.join(SYNTH_DIR, "fits")
OUT_DIR = os.path.join(HERE, "analysis")

ALPHA_NAMES = S.ALPHA_NAMES                      # cc, refc, win, dr
TRAINVAL_END = 2007.0                            # pre-registered (WP-3, run_wp1f)

# The four alpha numerator flows, by name.  `_build_empirical_alphas` divides
# each by the trapezoid exposure of its parent stock; the parent indices are
# the core's own `ALPHA_PARENT_STOCK_IDX` and are read from it, not restated.
ALPHA_FLOW = {
    "alpha_cc":   "concentrate_consumption",
    "alpha_refc": "refined_consumption",
    "alpha_win":  "waelz_input",
    "alpha_dr":   "direct_reuse_recycling",
}

# Replicate draws of the twin.  WP-3's `base_d1y_noisy` is rng_seed 0, so the
# replicates start at 1 and are disjoint from it.
REPLICATE_SEEDS = tuple(range(1, 9))
REPLICATE_TAG = "base_d1y_rep{:02d}"

# Perturbation size for the driver response, in SD of log1p(driver).  1.0
# matches WP-8b/8c's convention so the magnitudes are comparable across
# packages; `RESP_DELTA_CHECK` re-runs the UDE at a quarter of it to show the
# response is not an artefact of a large step.
RESP_DELTA = 1.0
RESP_DELTA_CHECK = 0.25

PATCHES: list[str] = []


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------
def install(v5mod=None):
    """Delegate to `zinc_synth_lab.install` — WP-5 adds no patch of its own."""
    global PATCHES
    PATCHES = list(S.install(v5mod))
    return PATCHES


# ---------------------------------------------------------------------------
# the ground truth of the driver response
# ---------------------------------------------------------------------------
def truth_beta(cols):
    """`(4, n_drivers)` analytic `d log alpha_k / d z_j`, resolved by name.

    `cols` is the driver ordering the model actually resolved
    (`load_zinc_data`'s `exog_cols`).  Every short name in `TRUTH` is mapped
    through `zinc_synth_lab.DRIVER_ALIASES` to a full column name and looked
    up in `cols`; a name that does not resolve is an error, never a silent
    zero.
    """
    cols = list(cols)
    B = np.zeros((len(ALPHA_NAMES), len(cols)))
    for k, ch in enumerate(ALPHA_NAMES):
        for short, b in S.TRUTH["alpha"][ch]["beta"].items():
            name = S.DRIVER_ALIASES[short]
            if name not in cols:
                raise KeyError(f"truth driver {short!r} -> {name!r} not in exog_cols")
            B[k, cols.index(name)] = float(b)
    return B


def driver_sigma(ctx):
    """`sigma_j = SD(log1p(X_j))` over 1980-2019 — the perturbation unit.

    Same expression `driver_context` z-scores with, so shifting `log1p(X_j)`
    by `delta * sigma_j` shifts the truth's `z_j` by exactly `delta`.
    """
    X = np.asarray(ctx["exog_values"], float)
    return np.array([np.std(np.log1p(np.maximum(X[:, j], 0.0)))
                     for j in range(X.shape[1])])


def perturb(X, j, delta, sigma):
    """Permanent `+delta * sigma_j` shift of driver `j` in log1p space."""
    w = np.zeros(X.shape[1]); w[int(j)] = 1.0
    return perturb_many(X, w, delta, sigma)


def perturb_many(X, w, delta, sigma):
    """Permanent shift of every driver `j` by `delta * w_j * sigma_j`.

    The single-driver perturbation is the case `w = e_j`.  A general `w` is
    what the *directional* response needs: the truth's own beta vector, scaled
    to unit norm, is a direction in driver space along which the coefficient
    genuinely moves, and unlike a single-driver partial derivative it is
    identifiable in the presence of collinear drivers.
    """
    X = np.asarray(X, float)
    w = np.asarray(w, float)
    shift = float(delta) * w * np.asarray(sigma, float)
    return np.expm1(np.log1p(np.maximum(X, 0.0)) + shift[None, :])


def truth_response(ctx, years, delta=RESP_DELTA, arm="base"):
    """The truth's response matrix, re-derived numerically from the closure.

    Returns `(4, n_drivers)`.  Must reproduce `truth_beta` to round-off; the
    check is what guarantees the perturbation is expressed in the truth's own
    coordinates and not in some rescaled version of them.
    """
    cols = list(ctx["exog_cols"])
    base = S.truth_coefficients(ctx, years, arm)["alphas"]
    R = np.zeros((len(ALPHA_NAMES), len(cols)))
    for j, short in _short_by_index(cols).items():
        c2 = dict(ctx)
        z2 = {k: v.copy() for k, v in ctx["z"].items()}
        z2[short] = z2[short] + float(delta)
        c2["z"] = z2
        a2 = S.truth_coefficients(c2, years, arm)["alphas"]
        R[:, j] = (np.log(a2) - np.log(base)).mean(0) / float(delta)
    return R


def _short_by_index(cols):
    """`{column index: TRUTH short name}` for the drivers the truth uses."""
    cols = list(cols)
    return {cols.index(name): short for short, name in S.DRIVER_ALIASES.items()
            if name in cols}


# ---------------------------------------------------------------------------
# the UDE estimator
# ---------------------------------------------------------------------------
def ude_context(tag, cfg=None, synth_dir=SYNTH_DIR):
    """Arm dataset `tag` and build the zero-step fit its weights belong to.

    Returns `(fit, uctx)` where `uctx` carries the full-history driver matrix,
    the resolved column names and the train cut.  The exog rebuild is checked
    bit-for-bit against the fit's own feature matrix before anything is
    perturbed — the same precondition `zinc_cf_lab.build_context` enforces for
    WP-7/8, and the reason a driver perturbation here means what it says.
    """
    import zinc_colloc_v5 as v5

    install(v5)
    cfg = dict(cfg or load_anchor_config())
    ds = S.load_dataset(os.path.join(synth_dir, f"{tag}.npz"))
    S.arm(ds)
    fit = cflab.init_fit(cfg)

    years = np.asarray(fit.data_all["years"], float).ravel()
    t_full = np.asarray(ds["exog_times_full"], float)
    X_full = np.asarray(ds["exog_values_full"], float)
    cols = list(ds["exog_cols"])
    cut, n_tv = cflab.split_cut(len(years), cfg.get("trainval_frac", 0.7),
                               cfg.get("val_frac", 0.2))

    exog_ref = np.asarray(fit.data_all["exog_values"], float)
    exog_rb = cflab.rebuild_exog(cfg, years, t_full, X_full, cut)
    gap = float(np.max(np.abs(exog_rb - exog_ref)))
    if gap != 0.0:
        raise RuntimeError(
            f"exog rebuild for {tag} differs from the fit's own features by "
            f"{gap:.3e}; the driver perturbation would not be measuring the "
            f"model the weights belong to")

    uctx = dict(tag=tag, cfg=cfg, ds=ds, years=years, t_full=t_full,
                X_full=X_full, cols=cols, cut=int(cut), n_trainval=int(n_tv),
                exog_base=exog_rb, gap_exog=gap)
    return fit, uctx


def ude_alphas(fit, params, exog, stocks=None):
    """`alpha_k(t)` at the year nodes from a frozen parameter set.

    Evaluated at the *observed* stocks, which is what `zinc_A_lab.dump_A`
    persists and therefore what WP-3 scored.  Under `anchor_v4`
    (`use_stock_input: false`) alpha does not read the state at all, so the
    choice is immaterial and `check()` verifies that it is.
    """
    import jax
    import jax.numpy as jnp
    import zinc_colloc_v5 as v5

    data = dict(fit.data_all)
    data["exog_values"] = jnp.asarray(exog)
    years = jnp.asarray(np.asarray(fit.data_all["years"], float).ravel())
    if stocks is None:
        stocks = np.asarray(fit.data_all["stocks_obs"], float)

    def one(t, S4):
        ex = v5.exog_fn(t, data["exog_times"], data["exog_values"])
        _cp, alphas, _tau, _fc, _raw = fit.nn_eval_sup(params, t, S4, ex, data)
        return alphas

    return np.asarray(jax.vmap(one)(years, jnp.asarray(stocks)))


def ude_response(fit, params, uctx, delta=RESP_DELTA, mask=None):
    """`(4, n_drivers)` UDE `d log alpha_k / d z_j` under the shared shift."""
    n = len(uctx["cols"])
    R = np.zeros((len(ALPHA_NAMES), n))
    for j in range(n):
        w = np.zeros(n); w[j] = 1.0
        R[:, j] = ude_response_dir(fit, params, uctx, w, delta=delta, mask=mask)
    return R


def ude_response_dir(fit, params, uctx, w, delta=RESP_DELTA, mask=None):
    """`(4,)` UDE response to a permanent shift along the direction `w`."""
    cfg, years = uctx["cfg"], uctx["years"]
    m = np.ones(years.size, bool) if mask is None else np.asarray(mask, bool)
    a0 = ude_alphas(fit, params, uctx["exog_base"])
    Xp = perturb_many(uctx["X_full"], w, delta, uctx["sigma"])
    exog = cflab.rebuild_exog(cfg, years, uctx["t_full"], Xp, uctx["cut"])
    ap = ude_alphas(fit, params, exog)
    return (np.log(ap[m]) - np.log(a0[m])).mean(0) / float(delta)


# ---------------------------------------------------------------------------
# the AR/ARX estimator
# ---------------------------------------------------------------------------
def _series_from_dataset(ds):
    """The observation arrays the AR ensemble consumes, from a twin dataset.

    Flow row *i* closes the interval `(years[i], years[i+1]]`, matching
    `load_zinc_data`'s convention and `run_wp1f`'s reading of it.
    """
    years = np.asarray(ds["years"], float).ravel()
    return dict(years=years, years_flow=years[1:],
                stocks_obs=np.asarray(ds["stocks_obs"], float),
                flows_obs=np.asarray(ds["flows_obs"], float),
                flow_names=[str(x) for x in ds["flow_obs_names"]],
                exog_values=np.asarray(ds["exog_values"], float),
                exog_cols=[str(x) for x in ds["exog_cols"]])


def _fit_arx_fixed_subset(y, Z, n_train, sub, max_lag=3):
    """AIC over lag order only, with the driver subset held fixed.

    The `oracle` specification: the AR is *told* which drivers the truth uses,
    so its response cannot be blamed on subset selection.  Reuses WP-1f's own
    OLS and AIC so the two specifications differ in exactly one thing.
    """
    from run_wp1f import _ols, _aic

    best = None
    for p in range(0, max_lag + 1):
        rows = np.arange(p, n_train)
        if rows.size < 8:
            continue
        cols = [np.ones(rows.size)] + [y[rows - i] for i in range(1, p + 1)] \
            + [Z[rows, j] for j in sub]
        X = np.column_stack(cols)
        k = X.shape[1]
        if rows.size - k < 3:
            continue
        try:
            beta, rss = _ols(X, y[rows])
        except np.linalg.LinAlgError:
            continue
        a = _aic(rss, rows.size, k)
        if best is None or a < best["aic"]:
            best = dict(aic=a, p=p, sub=tuple(sub), beta=beta,
                        sigma2=rss / max(rows.size - k, 1), n_eff=rows.size)
    if best is None:
        m = float(np.mean(y[:n_train]))
        best = dict(aic=np.inf, p=0, sub=(), beta=np.array([m]),
                    sigma2=float(np.var(y[:n_train])), n_eff=n_train)
    return best


def ar_fit(ds, *, max_lag=3, max_drivers=3, log=True, trainval_end=TRAINVAL_END,
           oracle=False):
    """Fit one independent ARX per alpha-numerator flow, WP-1f's specification.

    Returns a dict carrying, per channel, the AIC-selected order and driver
    subset, the coefficients, the residual variance, and the **frozen**
    training-window standardisation of the driver matrix.  Freezing it is what
    makes a later driver perturbation a perturbation of the *input* rather
    than of the model: re-standardising a shifted series against its own mean
    would divide the shift straight back out.

    `oracle=True` replaces the AIC subset search with the truth's own two
    drivers for each channel — the most generous specification available to
    the comparator, and the one that separates *selection* error from
    *estimation* error in the driver response.
    """
    from run_wp1f import fit_arx_one

    sr = _series_from_dataset(ds)
    yrs_f = sr["years_flow"]
    is_test = yrs_f > float(trainval_end)
    n_train = int(np.argmax(is_test)) if is_test.any() else len(yrs_f)
    n_total = len(yrs_f)

    Xex = sr["exog_values"][1:, :]                    # closing endpoint of each row
    mu, sd = Xex[:n_train].mean(0), Xex[:n_train].std(0)
    sd = np.where(sd > 0, sd, 1.0)

    Z = (Xex - mu) / sd
    B_true = truth_beta(sr["exog_cols"])
    I = {n: i for i, n in enumerate(sr["flow_names"])}
    models = {}
    for k, (ch, fname) in enumerate((c, ALPHA_FLOW[c]) for c in ALPHA_NAMES):
        raw = sr["flows_obs"][:, I[fname]]
        y = np.log(np.maximum(raw, 1e-12)) if log else raw.copy()
        if oracle:
            sub = tuple(int(j) for j in np.flatnonzero(B_true[k]))
            f = _fit_arx_fixed_subset(y, Z, n_train, sub, max_lag=max_lag)
        else:
            f = fit_arx_one(y, Z, n_train, max_lag=max_lag,
                            max_drivers=max_drivers, all_drivers=False)
        models[ch] = dict(f, flow=fname, y=y, raw=raw, log=bool(log))
    return dict(models=models, mu=mu, sd=sd, Xex=Xex, n_train=n_train,
                n_total=n_total, series=sr, log=bool(log), oracle=bool(oracle),
                is_test=is_test, trainval_end=float(trainval_end))


def _design_row(f, path, Z, t):
    p, sub, = f["p"], f["sub"]
    return np.array([1.0] + [path[t - i] for i in range(1, p + 1)]
                    + [Z[t, j] for j in sub])


def ar_fitted_path(f, Z, n_train, n_total):
    """One-step-ahead fitted values in-sample; dynamic forecast out.

    Rows before the lag order carry no fitted value and are returned as NaN —
    an AR(p) has nothing to say about its own first p observations, and
    padding them with the data would credit the estimator with the answer.
    """
    y, p = f["y"], f["p"]
    out = np.full(n_total, np.nan)
    for t in range(p, n_train):                        # observed lags
        out[t] = float(np.dot(f["beta"], _design_row(f, y, Z, t)))
    path = np.array(y, float, copy=True)
    path[:n_train] = y[:n_train]
    for t in range(n_train, n_total):                  # own lags
        path[t] = float(np.dot(f["beta"], _design_row(f, path, Z, t)))
        out[t] = path[t]
    return out


def ar_simulate(f, Z, n_total, start=None):
    """Dynamic simulation from `start` (default: the lag order) to the end.

    Used only for the driver response, where a permanent shift has to be
    allowed to propagate through the lag polynomial.  The unperturbed
    simulation is the baseline it is differenced against, so the AR's own
    simulation error cancels and what is left is its response.
    """
    y, p = f["y"], f["p"]
    start = p if start is None else int(start)
    path = np.array(y, float, copy=True)
    for t in range(start, n_total):
        path[t] = float(np.dot(f["beta"], _design_row(f, path, Z, t)))
    return path


def ar_long_run(f):
    """Closed-form long-run multiplier `gamma_j / (1 - sum phi)` per driver.

    `None` when the AR polynomial is not stable, which is the only case where
    the simulated response is not the long-run one.
    """
    p, sub, beta = f["p"], f["sub"], f["beta"]
    phi = beta[1:1 + p]
    denom = 1.0 - float(np.sum(phi))
    if abs(denom) < 1e-8 or denom < 0:
        return None
    return {int(j): float(g) / denom for j, g in zip(sub, beta[1 + p:])}


def ar_long_run_response(arf, uctx, delta=RESP_DELTA, mask=None):
    """`(4, n_drivers)` closed-form cross-check on `ar_response`.

    `gamma_j / (1 - sum phi)` is the response to a unit move of the AR's *own*
    standardised regressor.  The shared perturbation is a unit move in the
    truth's log1p z-coordinate, which is a different size in raw-standardised
    units and varies over the window, so the multiplier is scaled by the mean
    realised `Delta Z_j`.  Agreement with the simulated response to within
    that approximation is the check that the simulation is not mis-driven;
    `NaN` marks an unstable lag polynomial, where no long-run response exists.
    """
    sigma, cols = uctx["sigma"], uctx["cols"]
    Xann = arf["series"]["exog_values"]
    m = (np.ones(arf["n_total"], bool) if mask is None
         else np.asarray(mask, bool)[1:])
    dZ = np.array([
        float(np.mean(((perturb(Xann, j, delta, sigma)[1:, j] - Xann[1:, j])
                       / arf["sd"][j])[m]))
        for j in range(len(cols))])
    R = np.full((len(ALPHA_NAMES), len(cols)), np.nan)
    for k, ch in enumerate(ALPHA_NAMES):
        lr = ar_long_run(arf["models"][ch])
        if lr is None:
            continue
        R[k, :] = 0.0
        for j, g in lr.items():
            R[k, j] = g * dZ[j] / float(delta)
    return R


def _exposure(years, stocks):
    """`_build_empirical_alphas`' trapezoid exposure, per alpha channel.

    Shape `(T-1, 4)`; row *i* is the exposure over `(years[i], years[i+1]]`.
    """
    import zinc_colloc_v5 as v5
    dt = years[1:] - years[:-1]
    Sp = np.array([stocks[:, k] for k in v5.ALPHA_PARENT_STOCK_IDX]).T
    return 0.5 * (Sp[:-1] + Sp[1:]) * dt[:, None]


def ar_alpha_path(arf, *, stocks=None, smear=True, flows=None):
    """`alpha_hat = F_hat / exposure`, shape `(T, 4)` with row 0 NaN.

    `stocks` overrides the denominator (the twin's clean stocks, for the
    denominator-noise decomposition); `flows` overrides the numerator with an
    already-computed log-flow matrix `(T-1, 4)` (the perturbed simulations).
    """
    sr = arf["series"]
    years, T = sr["years"], sr["years"].size
    stocks = sr["stocks_obs"] if stocks is None else np.asarray(stocks, float)
    exposure = _exposure(years, stocks)

    out = np.full((T, len(ALPHA_NAMES)), np.nan)
    for k, ch in enumerate(ALPHA_NAMES):
        f = arf["models"][ch]
        if flows is not None:
            path = np.asarray(flows, float)[:, k]
        else:
            path = ar_fitted_path(f, (arf["Xex"] - arf["mu"]) / arf["sd"],
                                  arf["n_train"], arf["n_total"])
        if f["log"]:
            lvl = np.exp(path) * (np.exp(0.5 * f["sigma2"]) if smear else 1.0)
        else:
            lvl = path
        with np.errstate(divide="ignore", invalid="ignore"):
            out[1:, k] = np.where(exposure[:, k] > 1e-12,
                                  lvl / np.maximum(exposure[:, k], 1e-12), np.nan)
    return out


def ar_alpha_logsd(arf):
    """The AR's *own* reported uncertainty on `log alpha_hat`, `(T, 4)`.

    For the log-linear ARX the fitted value at row `t` is `x_t' beta_hat`, so
    the coefficient uncertainty is `sqrt(x_t' Cov(beta_hat) x_t)` with
    `Cov = sigma2 (X'X)^-1` from the training design.  The denominator
    contributes nothing: in the AR's own worldview `S_obs` is data, not an
    estimate, which is precisely the asymmetry WP-5 is testing.

    In-sample rows only — out of sample the lag recursion makes the
    propagated variance a different (and much larger) object, and reporting
    the in-sample number is the generous choice.
    """
    sr = arf["series"]
    T = sr["years"].size
    Z = (arf["Xex"] - arf["mu"]) / arf["sd"]
    out = np.full((T, len(ALPHA_NAMES)), np.nan)
    for k, ch in enumerate(ALPHA_NAMES):
        f = arf["models"][ch]
        if not f["log"]:
            continue
        p, sub, y = f["p"], f["sub"], f["y"]
        rows = np.arange(p, arf["n_train"])
        X = np.column_stack([np.ones(rows.size)]
                            + [y[rows - i] for i in range(1, p + 1)]
                            + [Z[rows, j] for j in sub])
        try:
            XtXi = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            continue
        for t in rows:
            x = _design_row(f, y, Z, t)
            out[t + 1, k] = np.sqrt(max(f["sigma2"] * float(x @ XtXi @ x), 0.0))
    return out


def ar_response(arf, uctx, delta=RESP_DELTA, mask=None):
    """`(4, n_drivers)` AR `d log alpha_hat / d z_j` under the shared shift.

    The denominator does not move — the AR ensemble models flows, and the
    stock it divides by is data.  So this is the response of the numerator,
    which is exactly the point: an estimator with no mechanism linking the two
    can only respond through one of them.
    """
    n = len(uctx["cols"])
    R = np.zeros((len(ALPHA_NAMES), n))
    for j in range(n):
        w = np.zeros(n); w[j] = 1.0
        R[:, j] = ar_response_dir(arf, uctx, w, delta=delta, mask=mask)
    return R


def ar_response_dir(arf, uctx, w, delta=RESP_DELTA, mask=None):
    """`(4,)` AR response to a permanent shift along the direction `w`."""
    years = arf["series"]["years"]
    m = np.ones(years.size - 1, bool) if mask is None else np.asarray(mask, bool)[1:]
    Z0 = (arf["Xex"] - arf["mu"]) / arf["sd"]
    Xp = perturb_many(arf["series"]["exog_values"], w, delta, uctx["sigma"])[1:, :]
    Zp = (Xp - arf["mu"]) / arf["sd"]
    out = np.zeros(len(ALPHA_NAMES))
    for k, ch in enumerate(ALPHA_NAMES):
        f = arf["models"][ch]
        b = ar_simulate(f, Z0, arf["n_total"])
        s = ar_simulate(f, Zp, arf["n_total"])
        d = (s - b) if f["log"] else (np.log(np.maximum(s, 1e-12))
                                      - np.log(np.maximum(b, 1e-12)))
        out[k] = float(np.mean(d[m])) / float(delta)
    return out


# ---------------------------------------------------------------------------
# replicate draws of the twin
# ---------------------------------------------------------------------------
def make_replicates(seeds=REPLICATE_SEEDS, *, arm="base", delta=1.0,
                    out_dir=SYNTH_DIR, cfg=None, verbose=True):
    """Independent noise draws of the WP-3 twin at Delta = 1 yr.

    Only the noise draw changes: the dense solve, the observation operators
    and the ground truth are WP-3's, re-aggregated rather than re-integrated,
    so every replicate carries the identical `alpha_true`.  That is what makes
    the spread across replicates an estimator variance and not a truth
    variance.

    A replicate is skipped if it already exists, so this is safe to re-run.
    """
    ctx = S.driver_context(cfg)
    sigma = S.calibrate_noise()
    written, dense = [], None
    for rs in seeds:
        tag = REPLICATE_TAG.format(int(rs))
        path = os.path.join(out_dir, f"{tag}.npz")
        if os.path.exists(path):
            if verbose:
                print(f"[rep] {tag} exists, skipping")
            written.append(path)
            continue
        if dense is None:
            dense = S.dense_solve(ctx, arm)
        s = S.sample(dense, delta, ctx, noise_scale=1.0, rng_seed=int(rs),
                     sigma=sigma)
        ds = S.build_dataset(ctx, s, delta)
        tw = S.truth_coefficients(ctx, s["years"], arm, S_path=s["stocks_clean"])
        bias = S.alpha_target_bias(ctx, dense, delta, arm_name=arm)
        truth = dict(alphas_point=tw["alphas"], tau_sup_point=tw["tau_sup"],
                     f_cohort_point=tw["f_cohort"],
                     alphas_window_unweighted=bias["a_unweighted"],
                     alphas_window_weighted=bias["a_weighted"],
                     alphas_window_obs=bias["a_obs"],
                     stocks_clean=s["stocks_clean"],
                     F_int_clean=s["F_int_clean"],
                     window_years=bias["years"],
                     alpha_names=np.asarray(ALPHA_NAMES, dtype=object))
        written.append(S.save_dataset(
            path, ds, truth=truth,
            meta=dict(arm=arm, delta=float(delta), noise_scale=1.0,
                      rng_seed=int(rs), n_sub=S.N_SUB, T=int(ds["years"].size),
                      replicate_of="base_d1y_noisy", package="WP-5")))
        if verbose:
            print(f"[rep] wrote {tag}")
    if dense is not None:
        import jax
        jax.clear_caches()
    return written


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------
def check(verbose=True, tag="base_d1y_noisy"):
    """Resolved drivers, `input_dim`, and the three preconditions WP-5 rests on.

    1. the exog rebuild is bit-identical to the fit's own features
    2. the UDE alpha reconstructed from a dumped weight set reproduces the
       dump's own `alphas` exactly
    3. the numerically measured truth response equals the analytic `beta`
    """
    import zinc_colloc_v5 as v5

    integrity_check()
    install(v5)
    cfg = load_anchor_config()

    fit, uctx = ude_context(tag, cfg)
    ctx = S.driver_context(cfg)
    uctx["sigma"] = driver_sigma(ctx)
    cols = uctx["cols"]
    input_dim = int(np.asarray(fit.data_all["exog_values"]).shape[1]) + 1 + 4

    B = truth_beta(cols)
    R = truth_response(ctx, uctx["years"])
    beta_gap = float(np.max(np.abs(B - R)))

    wpath = os.path.join(FIT_DIR, f"{tag}_seed0.npz")
    alpha_gap = float("nan")
    state_gap = float("nan")
    if os.path.exists(wpath):
        params = cflab.load_params(wpath)
        a = ude_alphas(fit, params, uctx["exog_base"])
        a_ref = np.load(wpath, allow_pickle=True)["alphas"]
        alpha_gap = float(np.max(np.abs(a - a_ref)))
        rng = np.random.default_rng(0)
        S_alt = np.asarray(fit.data_all["stocks_obs"], float) * np.exp(
            rng.normal(0, 0.25, (uctx["years"].size, 4)))
        state_gap = float(np.max(np.abs(
            ude_alphas(fit, params, uctx["exog_base"], stocks=S_alt) - a)))

    reps = sorted(os.path.basename(p) for p in
                  [os.path.join(SYNTH_DIR, REPLICATE_TAG.format(r) + ".npz")
                   for r in REPLICATE_SEEDS] if os.path.exists(p))
    fits = sorted(f for f in os.listdir(FIT_DIR)) if os.path.isdir(FIT_DIR) else []

    info = dict(
        tag=tag, patches=list(PATCHES), n_drivers=len(cols), drivers=list(cols),
        input_dim=input_dim, exog_feature_orders=list(cfg.get("exog_feature_orders", (0,))),
        use_stock_input=bool(cfg.get("use_stock_input", False)),
        train_cut=uctx["cut"], n_trainval=uctx["n_trainval"],
        trainval_end=float(uctx["years"][uctx["n_trainval"] - 1]),
        gap_exog_rebuild=uctx["gap_exog"], gap_alpha_reconstruction=alpha_gap,
        gap_truth_beta_vs_numeric=beta_gap,
        alpha_state_dependence=state_gap,
        truth_beta_nonzero=int(np.count_nonzero(B)),
        truth_beta_entries=int(B.size),
        alpha_flows={k: v for k, v in ALPHA_FLOW.items()},
        alpha_parent_stock_idx=list(v5.ALPHA_PARENT_STOCK_IDX),
        n_replicates_on_disk=len(reps), n_fit_dumps=len(fits),
    )
    if verbose:
        print("zinc_interp_lab --check  (WP-5)")
        print("=" * 74)
        print(f"  patches              : {PATCHES}")
        print(f"  armed dataset        : {tag}")
        print(f"  drivers ({len(cols)}):")
        for j, c in enumerate(cols):
            sh = _short_by_index(cols).get(j)
            print(f"      [{j:2d}] {c}" + (f"   <- TRUTH z[{sh}]" if sh else ""))
        print(f"  exog_feature_orders  : {info['exog_feature_orders']}")
        print(f"  input_dim            : {input_dim}   "
              f"(= 1 t + 4 S + {len(cols)} drivers x "
              f"{len(info['exog_feature_orders'])} orders)")
        print(f"  use_stock_input      : {info['use_stock_input']}")
        print(f"  train cut / trainval : {uctx['cut']} / {uctx['n_trainval']} "
              f"(trainval ends {info['trainval_end']:.0f})")
        print(f"  alpha channels       : " + ", ".join(
            f"{c} = {ALPHA_FLOW[c]} / S[{v5.ALPHA_PARENT_STOCK_IDX[i]}]"
            for i, c in enumerate(ALPHA_NAMES)))
        print("  preconditions:")
        print(f"      exog rebuild gap            : {uctx['gap_exog']:.3e}  "
              f"(must be 0)")
        print(f"      alpha reconstruction gap    : {alpha_gap:.3e}  (must be 0)")
        print(f"      truth beta vs numeric gap   : {beta_gap:.3e}")
        print(f"      alpha state dependence      : {state_gap:.3e}  "
              f"(0 under use_stock_input=false)")
        print(f"  truth response sparsity : {info['truth_beta_nonzero']} of "
              f"{info['truth_beta_entries']} entries non-zero")
        print(f"  replicates on disk      : {info['n_replicates_on_disk']} of "
              f"{len(REPLICATE_SEEDS)}")
        print(f"  fit dumps on disk       : {info['n_fit_dumps']}")
    S.disarm()
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description="WP-5 lab module")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--tag", default="base_d1y_noisy")
    ap.add_argument("--replicates", type=int, default=0,
                    help="generate N independent noise draws of the twin")
    args = ap.parse_args(argv)

    import zinc_colloc_v5 as v5
    integrity_check()
    install(v5)
    if args.replicates:
        make_replicates(tuple(range(1, args.replicates + 1)))
    if args.check or not args.replicates:
        info = check(tag=args.tag)
        print("\n" + json.dumps(
            {k: v for k, v in info.items() if k not in ("drivers", "alpha_flows")},
            indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
