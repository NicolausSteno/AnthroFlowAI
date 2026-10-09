#!/usr/bin/env python3
"""
zinc_xai_lab.py — lab module for the WP-8 explainability suite
==============================================================

The spec's organising question for WP-8: *what does a physics-constrained
neural model let you learn about a resource system that a regression or
autoregressive model does not?*  Every sub-package has to produce a number or
a figure, not a claim.  This module supplies the instruments; `run_wp8.py`
runs them and writes the tables.

  8a  input-gradient attribution      `attribution_seed`
  8b  temporal attribution            `influence_surface`
  8c  forward-mode ODE sensitivities  `forward_sensitivity`
  8d  block-permutation importance    `permutation_seed`
  8e  Morris / Sobol screening        `morris`, `sobol`
  8f  regime-break recovery           `breaks`

Pattern (CLAUDE.md rule 1)
--------------------------
`zinc_colloc_v5.py` is imported and never edited; `zinc_alpha_lab.integrity_check`
pins its MD5 and the MD5 of the `anchor_v4` data copy.  `PATCHES` is empty:
every instrument here is *composed* from the core's public factories
(`make_nn_eval`, `make_integrator`, `make_rhs`, `exog_fn`, `preprocess_exog`)
and from `zinc_cf_lab`'s already-verified driver rebuild, which is checked
bit-for-bit against the fit's own feature matrix before use.  Weights are
reloaded from the `zinc_A_lab` dumps in `analysis/wp2a/` — no refits anywhere
in WP-8.

Two facts about `anchor_v4` that shape the whole package
-------------------------------------------------------
1. **`use_stock_input: false`.**  The first five entries of `_norm_inputs`
   (v5:477) are identically zero, so the `d alpha / d S` block the spec wants
   reported "separately and prominently" is **exactly zero, by construction,
   not by measurement**.  `attribution_seed` computes it anyway and reports
   the exact zero, because a saliency map that is silently structurally
   dead is the kind of thing Adebayo et al. (2018) exists to catch.  The
   mechanistic state-dependence the spec wants is still measurable — it just
   lives in the *propagated* response (8c) rather than in the network's local
   map (8a), and the 8a-vs-8c contrast is precisely where it shows up.
2. **Thirteen drivers at two orders, not nine at three.**  The real input
   layout is `1 + 4 + 13 * 2 = 31`, not the `1 + 4 + 3 * N` of spec §1
   (SCHEMA §9 flag 1, unresolved).  `layout_index` resolves (driver, order)
   pairs against the *actual* preprocessing rather than the documented one,
   and every table carries the order it came from.  Order 1 is the first
   difference, so a driver that matters only at order 1 is one whose *rate of
   change* moves the coefficient — the substantive reading the spec asks for.

Gradients, in which coordinates
-------------------------------
`nn_eval_sup` is a function of `(t, S, exog_t)` and `_norm_inputs` maps those
affinely onto the network's input `x`:

    x = [ (t - t_mean)/t_std * use_t,  S_feat * use_S,  (exog - mu)/sigma ]

so `d f / d x_exog = (d f / d exog) * sigma` exactly, with no finite
differencing.  `jax.jacrev` is applied to the core's own closure and the chain
rule to the affine map is applied analytically.  Attributions are reported in
**normalised-input coordinates** (`sigma`-multiplied), which is what makes
them comparable across drivers of wildly different units, and additionally in
log-coefficient space (`d log alpha / d x`), which is a unit-free elasticity
against a one-SD driver move and is the form the paper should quote.

Three variants, per the spec: vanilla gradient, gradient x input, and
integrated gradients with the training-period mean input as baseline and 50
steps.  Guided backprop and deconvnet are **not** implemented — they fail the
Adebayo sanity checks by being largely independent of the model parameters,
and the spec says so.  The IG completeness identity
`sum_i IG_i = f(x) - f(baseline)` is checked numerically and reported.

Temporal attribution (8b)
-------------------------
The MLP gradient is instantaneous; the *system* has memory, through stock
accumulation and the in-use cohorts.  So the influence surface is built by
perturbing the raw driver history with a narrow Gaussian bump at year `s`,
rebuilding the exogenous features through `preprocess_exog` exactly as the fit
did (`zinc_cf_lab.rebuild_exog`, verified bit-identical to the fit's own
features), re-integrating, and differencing against the factual rollout.
Because the bump is applied to the *raw* driver before differencing, both the
level and the first-difference features move, which is the honest version:
you cannot perturb a series' level in one year without perturbing its
difference in two.

`memory_length` reports how far a shock stays detectable above a noise floor
that is measured, not assumed — `zinc_cf_lab` found that two bitwise-identical
closures separate by up to 14 kt over a 39-year free-run because adaptive
stepping is not associative, so the floor is the factual-minus-factual
difference under a zero-amplitude bump.

Forward-mode ODE sensitivities (8c)
-----------------------------------
`train_model` leaves diffrax's default `RecursiveCheckpointAdjoint` in place
(v5:2265; CLAUDE.md rule 4 describes `zinc_baseline`, not the UDE — WP-4c
flag 1), and that is a `custom_vjp`, so forward-mode AD through the fit's own
integrator is impossible.  A matched integrator with `adjoint=ForwardMode()`
is therefore composed from the core's pieces and **verified against the fit's
own trajectory** before any derivative is taken; `jax.jacfwd` over a
13-vector of per-driver perturbation amplitudes then integrates
`d S / d theta` alongside the state, which is what the spec asks for, for the
drivers only and not for the 2 250 live weights.

Sobol and Morris (8e)
---------------------
Over the interpretable structural coefficients of
`zinc_circ_lab.PARAM_NAMES[:17]`, not over network weights — specifically over
the **nine free** ones (`FREE_PARAMS`).  Five of the seventeen are pinned to
data by `anchor_v4` and have exactly zero across-seed spread, and three are
determined by simplex closure; drawing a simplex's components independently
would put sample mass outside the simplex and make every index meaningless, so
they are closed rather than drawn.  The joint distribution is stated explicitly
(`STRUCT_DIST`) and the sensitivity of the result to that statement is
reported over `k in {1, 2, 4}`, because it is a judgement call.

Breaks (8f)
-----------
Bai-Perron by dynamic programming on a mean-shift model of the log-coefficient
series, `k` selected by BIC, plus a CUSUM cross-check.  The event list is
pre-registered in `EVENTS` and is fixed in source before any break date is
looked at.

CLI
---
    python zinc_xai_lab.py --check
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

import numpy as np

import zinc_alpha_lab as alab
from zinc_alpha_lab import integrity_check, load_anchor_config      # noqa: F401
import zinc_cf_lab as cflab

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR_DEFAULT = os.path.join(HERE, "analysis", "wp8")
WEIGHTS_DIR_DEFAULT = os.path.join(HERE, "analysis", "wp2a")

# Coefficient vector used by every attribution table, in the order
# `nn_eval_sup` emits it: 4 alphas, then the 8 supervised tau-likes, then the
# 3 cohort shares.  Same names as `zinc_fisher_lab.COEF_NAMES` where they
# overlap so the two packages' tables join without reindexing.
ALPHA_NAMES = ["alpha_cc", "alpha_refc", "alpha_win", "alpha_dr"]
TAU_SUP_NAMES = ["tau_ref", "tau_waelz", "tau_olds", "tau_diss",
                 "frac_fu_new", "frac_fu_loss", "frac_eu_new", "frac_eu_loss"]
COHORT_NAMES = ["f_cohort_10yr", "f_cohort_20yr", "f_cohort_44yr"]
COEF_NAMES = ALPHA_NAMES + TAU_SUP_NAMES + COHORT_NAMES
N_COEF = len(COEF_NAMES)
ALPHA_SLICE = slice(0, 4)

# Which coefficient slots the network actually controls under `anchor_v4`.
# The pinned ones are interpolations of data and their gradient w.r.t. the
# drivers is exactly zero; reporting them as "unimportant drivers" would be a
# category error, so they are masked and named.
def learned_mask(fit):
    lay = fit.layout
    m = np.zeros(N_COEF, dtype=bool)
    m[0:4] = True                                            # alphas
    for j, pinned in enumerate(lay["pin_taus_tuple"]):       # tau_ref..tau_diss
        m[4 + j] = not pinned
    m[8] = not lay["pin_fu_new"]
    m[9] = not lay["pin_fu_loss"]
    m[10] = not lay["pin_eu_new"]
    m[11] = not lay["pin_eu_loss"]
    m[12:15] = True                                          # cohort shares
    return m


# 8a integrated-gradients settings (spec: baseline = training-period mean
# input, 50 steps)
IG_STEPS = 50

# 8b influence-surface settings
BUMP_WIDTH_YR = 1.0
BUMP_AMPLITUDE_SD = 0.25          # fraction of the driver's own SD

# 8d block permutation (spec: circular block bootstrap, block ~5 yr, 200 reps)
BLOCK_LEN_YR = 5
N_PERM = 200

# 8f pre-registered events.  Fixed in source before any break date was
# computed; misses are reported (spec).
EVENTS = [
    ("Asian financial crisis",            1997, 1998),
    ("China commodity boom",              2003, 2008),
    ("Global financial crisis",           2008, 2009),
    ("Commodity slump",                   2015, 2016),
    ("China scrap import restrictions",   2018, 2019),
]
BREAK_MAX_K = 4
BREAK_MIN_SEG = 4

# 8e joint distribution over the interpretable structural coefficients.
# Stated explicitly because it is a judgement call and the spec says reviewers
# will ask.  Ranges are +-2 across-seed SD around the ensemble median at each
# year, truncated to the feasible set of `zinc_circ_lab`.
STRUCT_DIST = dict(kind="independent_uniform_pm_k_sd", k=2.0,
                   source="analysis/wp2a, 35 anchor_v4 seeds, final year",
                   free_dims=9, closed_by_simplex=3, pinned_to_data=5,
                   note="independence across the nine free dims is a stated "
                        "judgement; WP-2c measured the seeds' coefficients as "
                        "correlated, so the ranking is quoted and the level is "
                        "not")
MORRIS_LEVELS, MORRIS_TRAJ = 8, 64
SOBOL_BASE_N = 512

PATCHES: list[str] = []


def install(v5mod=None):
    """Apply import-time patches to the core module.  None needed for WP-8."""
    if v5mod is None:
        import zinc_colloc_v5 as v5mod          # noqa: F401
    return PATCHES


def jnp_(x):
    import jax.numpy as jnp
    return jnp.asarray(x)


def _rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024.0 ** 2) if sys.platform == "darwin" else r / 1024.0


# ---------------------------------------------------------------------------
# input-vector layout
# ---------------------------------------------------------------------------
def layout_index(fit, cfg):
    """The *actual* input-vector layout, resolved from the fit, not the spec.

        index 0            : t          (dead: `use_time_input: false`)
        indices 1..4       : S_conc, S_ref, S_inuse, S_scrap
                             (dead: `use_stock_input: false`)
        indices 5 + o*n_u + j : driver j at order o, for o in `orders`

    Returns a dict with the driver names in canonical order, the order tuple,
    the (n_features,) arrays `driver_of` / `order_of` mapping every exogenous
    feature back to its (driver, order), and the exog normaliser `sigma`.
    """
    import zinc_colloc_v5 as v5
    data_np = v5.load_zinc_data(cfg["xlsx_path"],
                                extra_exog_cols=cfg.get("extra_exog_cols"))
    cols = list(data_np["exog_cols"])
    orders = tuple(sorted(set(int(o) for o in cfg.get("exog_feature_orders", (0,)))))
    n_u = len(cols)
    n_feat = n_u * len(orders)
    driver_of = np.tile(np.arange(n_u), len(orders))
    order_of = np.repeat(np.asarray(orders), n_u)
    stats = fit.data_all["stats"]
    sigma = np.asarray(stats["exog_std"], float)
    mu = np.asarray(stats["exog_mean"], float)
    assert sigma.size == n_feat, (sigma.size, n_feat)
    return dict(drivers=cols, n_universe=n_u, orders=orders, n_features=n_feat,
                driver_of=driver_of, order_of=order_of, sigma=sigma, mu=mu,
                input_dim=1 + 4 + n_feat,
                use_time_input=float(np.asarray(stats["use_time_input"])),
                use_stock_input=float(np.asarray(stats["use_stock_input"])))


def coef_vector_fn(fit):
    """`g(params, t, S, ex, data) -> (N_COEF,)` from the core's own closure.

    Concatenates exactly what `nn_eval_sup` returns, in `COEF_NAMES` order.
    Nothing is reimplemented: the pin-aware composition, the exponentials and
    the simplex closures are all the core's.
    """
    import jax.numpy as jnp

    def g(params, t, S, ex, data):
        cp, alphas, tau_sup, f_cohort, _raw = fit.nn_eval_sup(params, t, S, ex, data)
        return jnp.concatenate([alphas, tau_sup, f_cohort])
    return g


# ---------------------------------------------------------------------------
# 8a — input-gradient attribution
# ---------------------------------------------------------------------------
def attribution_seed(fit, params, lay, *, data=None, ig_steps=IG_STEPS,
                     baseline_years=None):
    """Vanilla gradient, gradient x input and integrated gradients, per year.

    Returned in normalised-input coordinates (`d/dx`, i.e. per one-SD driver
    move) and in log-coefficient space, plus the `d/dS` block, which the spec
    wants reported prominently and which is exactly zero here.

    Shapes, with `T` years and `F = n_universe * n_orders` exogenous features:
        grad        (T, N_COEF, F)     d coef / d x
        grad_log    (T, N_COEF, F)     d log coef / d x
        gxi         (T, N_COEF, F)     grad * (x - 0)      [x is z-scored]
        ig          (T, N_COEF, F)     integrated gradients, baseline = mean x
        dS          (T, N_COEF, 4)     d coef / d S        (structurally 0)
        completeness(T, N_COEF)        |sum(ig) - (f(x) - f(base))| / |f(x)|
    """
    import jax
    import jax.numpy as jnp
    import zinc_colloc_v5 as v5

    d = fit.data_all if data is None else data
    years = np.asarray(d["years"], float).ravel()
    S_obs = np.asarray(d["stocks_obs"], float)
    g = coef_vector_fn(fit)
    sigma = jnp.asarray(lay["sigma"])

    ex_all = np.stack([np.asarray(v5.exog_fn(jnp.asarray(t),
                                             d["exog_times"], d["exog_values"]))
                       for t in years])                          # (T, F) raw
    if baseline_years is None:
        # training core, the core's own arithmetic (cf_lab.split_cut)
        cut, _ = cflab.split_cut(years.size)
        baseline_years = years[:cut]
    m_base = np.isin(years, baseline_years)
    ex_base = ex_all[m_base].mean(axis=0)
    S_base = S_obs[m_base].mean(axis=0)

    @jax.jit
    def _grads(t, S, ex):
        val = g(params, t, S, ex, d)
        J_ex = jax.jacrev(g, argnums=3)(params, t, S, ex, d)      # (N_COEF, F)
        J_S = jax.jacrev(g, argnums=2)(params, t, S, ex, d)       # (N_COEF, 4)
        return val, J_ex * sigma[None, :], J_S

    @jax.jit
    def _ig(t, S, ex):
        """Integrated gradients along the straight path in raw exog space.

        The state and the pinned values are held at the evaluation point: the
        baseline is a *driver* baseline, which is what the spec asks for, and
        moving `t` as well would change which year's pinned tau is read and
        make the path uninterpretable.
        """
        s = (jnp.arange(ig_steps) + 0.5) / ig_steps
        def one(a):
            exa = ex_base + a * (ex - ex_base)
            return jax.jacrev(g, argnums=3)(params, t, S, exa, d)
        J = jax.vmap(one)(s).mean(axis=0)                         # (N_COEF, F)
        # IG is invariant under the affine rescaling x = (ex - mu)/sigma:
        # J_x * (x - x_base) = (J_ex sigma)(ex - ex_base)/sigma, so the raw
        # form below is already in normalised-input coordinates.
        ig = J * (ex - ex_base)[None, :]
        return ig, g(params, t, S, ex, d), g(params, t, S, ex_base, d)

    T = years.size
    val = np.zeros((T, N_COEF))
    grad = np.zeros((T, N_COEF, lay["n_features"]))
    dS = np.zeros((T, N_COEF, 4))
    ig = np.zeros_like(grad)
    comp = np.zeros((T, N_COEF))
    for i, t in enumerate(years):
        tj, Sj, exj = jnp.asarray(t), jnp.asarray(S_obs[i]), jnp.asarray(ex_all[i])
        v, Je, Js = _grads(tj, Sj, exj)
        val[i], grad[i], dS[i] = np.asarray(v), np.asarray(Je), np.asarray(Js)
        ig_raw, f1, f0 = _ig(tj, Sj, exj)
        ig[i] = np.asarray(ig_raw)
        comp[i] = np.abs(np.asarray(ig_raw).sum(axis=1) - np.asarray(f1 - f0)) \
            / np.maximum(np.abs(np.asarray(f1)), 1e-12)

    x_all = (ex_all - lay["mu"][None, :]) / np.maximum(lay["sigma"][None, :], 1e-30)
    gxi = grad * x_all[:, None, :]
    grad_log = grad / np.maximum(np.abs(val)[:, :, None], 1e-12)
    return dict(years=years, value=val, grad=grad, grad_log=grad_log, gxi=gxi,
                ig=ig, dS=dS, completeness=comp, x=x_all, ex=ex_all,
                ex_baseline=ex_base, S_baseline=S_base,
                coef_names=np.asarray(COEF_NAMES, dtype=object),
                driver_names=np.asarray(lay["drivers"], dtype=object),
                driver_of=lay["driver_of"], order_of=lay["order_of"])


def per_driver(A, lay):
    """Sum an attribution tensor `(..., F)` over orders -> `(..., n_universe)`."""
    A = np.asarray(A)
    out = np.zeros(A.shape[:-1] + (lay["n_universe"],))
    for j in range(lay["n_universe"]):
        out[..., j] = A[..., lay["driver_of"] == j].sum(axis=-1)
    return out


def per_order(A, lay):
    """Split an attribution tensor `(..., F)` into `(..., n_orders, n_universe)`."""
    A = np.asarray(A)
    orders = lay["orders"]
    out = np.zeros(A.shape[:-1] + (len(orders), lay["n_universe"]))
    for oi, o in enumerate(orders):
        sel = lay["order_of"] == o
        out[..., oi, :] = A[..., sel]
    return out


# ---------------------------------------------------------------------------
# shared: fits, weights, machinery
# ---------------------------------------------------------------------------
def build_context(cfg=None, weights_dir=WEIGHTS_DIR_DEFAULT):
    """`(fit, ctx)` with `ctx = (cfg, t_full, X_full, cols, cut, weights_dir)`.

    Delegates to `zinc_cf_lab.build_context`, which already verifies that the
    `preprocess_exog` rebuild reproduces the fit's own feature matrix
    bit-for-bit — the precondition for every driver perturbation in 8b and 8d.
    """
    return cflab.build_context(cfg, weights_dir)


def load_params(seed, weights_dir=WEIGHTS_DIR_DEFAULT):
    return cflab.load_params(os.path.join(weights_dir, f"A_seed{seed}.npz"))


def machinery(fit):
    """`(integrate, coeffs)` on the fit's own closure.

    `integrate(params, data)` is the free-run augmented integrator started
    from the observed 1980 stocks; `coeffs(params, data, S)` returns the full
    15-vector of `COEF_NAMES` at every year node, evaluated at the stocks it
    is handed.  Both are jitted once per fit and cached, so 8b's 18 000 solves
    and 8d's 91 000 pay one compile.
    """
    import jax
    import jax.numpy as jnp
    import zinc_colloc_v5 as v5

    cache = getattr(machinery, "_cache", None)
    if cache is None:
        cache = machinery._cache = {}
    if id(fit) in cache:
        return cache[id(fit)]

    integrate_aug = v5.make_integrator("diffrax", fit.nn_eval)

    @jax.jit
    def integrate(params, data):
        return integrate_aug(params, data, data["stocks_obs"][0], data["years"])

    @jax.jit
    def coeffs(params, data, S):
        def one(t, S4):
            ex = v5.exog_fn(t, data["exog_times"], data["exog_values"])
            cp, alphas, tau_sup, f_cohort, _raw = fit.nn_eval_sup(
                params, t, S4, ex, data)
            return jnp.concatenate([alphas, tau_sup, f_cohort])
        return jax.vmap(one)(jnp.asarray(data["years"]), S)

    cache[id(fit)] = (integrate, coeffs)
    return cache[id(fit)]


def data_with_drivers(fit, cfg, X_full, t_full, cut):
    """A copy of `fit.data_all` whose exog features come from `X_full`."""
    import jax.numpy as jnp
    years = np.asarray(fit.data_all["years"], float).ravel()
    exog = cflab.rebuild_exog(cfg, years, t_full, X_full, cut)
    data = dict(fit.data_all)
    data["exog_values"] = jnp.asarray(exog)
    return data, exog


# ---------------------------------------------------------------------------
# 8b — temporal attribution (influence surfaces)
# ---------------------------------------------------------------------------
def gaussian_bump(t_full, s, width=BUMP_WIDTH_YR):
    """Unit-height Gaussian centred at `s`, SD `width`/2 so the full width at
    half maximum is ~`width` years (the spec's "width ~1 yr")."""
    sd = float(width) / 2.0
    return np.exp(-0.5 * ((np.asarray(t_full, float) - float(s)) / sd) ** 2)


def influence_surface(fit, params, ctx, driver, *, amplitude_sd=BUMP_AMPLITUDE_SD,
                      width=BUMP_WIDTH_YR, mach=None, centres=None):
    """`I_d(s, t)`: the response at year `t` to a bump in driver `d` at year `s`.

    Perturbs the **raw** driver history, rebuilds the features through
    `preprocess_exog`, re-integrates, and differences against the factual
    rollout.  Both the level and the difference feature move, which is
    unavoidable and correct: a series cannot have its level changed in one
    year without its first difference changing in two.

    Returns per-centre responses in the four observable stocks and in the 15
    coefficients, plus the zero-amplitude control that measures the adaptive
    integrator's own noise floor.
    """
    cfg, t_full, X_full, cols, cut, _wd = ctx
    j = list(cols).index(driver)
    years = np.asarray(fit.data_all["years"], float).ravel()
    if centres is None:
        centres = years
    integrate, coeffs = mach or machinery(fit)

    amp = float(amplitude_sd) * float(np.std(X_full[:, j]))
    data0, _ = data_with_drivers(fit, cfg, X_full, t_full, cut)
    S0, F0, _ = integrate(params, data0)
    C0 = np.asarray(coeffs(params, data0, jnp_(data0["stocks_obs"])))

    dS = np.zeros((len(centres), years.size, 4))
    dC = np.zeros((len(centres), years.size, N_COEF))
    dF = np.zeros((len(centres), years.size - 1, 4))
    for i, s in enumerate(centres):
        X = np.array(X_full, float, copy=True)
        X[:, j] = X[:, j] + amp * gaussian_bump(t_full, s, width)
        data, _ = data_with_drivers(fit, cfg, X, t_full, cut)
        S, F, _ = integrate(params, data)
        dS[i] = np.asarray(S) - np.asarray(S0)
        dC[i] = np.asarray(coeffs(params, data, jnp_(data["stocks_obs"]))) - C0
        dF[i] = np.asarray(F)[:, [1, 2, 3, 4]] - np.asarray(F0)[:, [1, 2, 3, 4]]

    # noise floor: the same pipeline at zero amplitude
    Xz = np.array(X_full, float, copy=True)
    dataz, _ = data_with_drivers(fit, cfg, Xz, t_full, cut)
    Sz, _Fz, _ = integrate(params, dataz)
    floor_S = np.max(np.abs(np.asarray(Sz) - np.asarray(S0)), axis=0)

    return dict(driver=driver, driver_index=int(j), centres=np.asarray(centres, float),
                years=years, dS=dS, dC=dC, dF=dF, amplitude=amp,
                amplitude_sd=float(amplitude_sd), width=float(width),
                floor_S=floor_S, S_factual=np.asarray(S0), C_factual=C0)


def memory_length(surf, *, rel_peak=0.10, rel_stock=1e-4):
    """How long a driver shock stays visible, per centre year and per stock.

    Two definitions, both reported, because they answer different questions and
    a single number would hide the choice:

    `mem_rel_peak`  the last lag at which the response still exceeds
                    `rel_peak` of its own peak for that centre — a decay
                    length, independent of any threshold on absolute size.
    `mem_rel_stock` the last lag at which the response exceeds `rel_stock` of
                    the observed stock — an absolute detectability length.

    A floor measured from the integrator's own reproducibility is *not* used:
    re-running the identical closure on identical data here is bit-identical
    (`floor_S` is exactly 0), unlike `zinc_cf_lab`'s case where two different
    closures diverge by up to 14 kt.  Reporting 0 as a detection threshold
    would make every response "detectable" for the whole horizon, so the two
    relative definitions above are used instead and the zero is reported as
    what it is.

    Returns `(mem_rel_peak, mem_rel_stock, peak_lag)`, each `(n_centres, 4)`.
    """
    dS, years, centres = surf["dS"], surf["years"], surf["centres"]
    S_fact = np.abs(surf["S_factual"])
    out = [np.full((centres.size, 4), np.nan) for _ in range(3)]
    for i, s in enumerate(centres):
        fwd = years >= s
        lags = years[fwd] - s
        for k in range(4):
            r = np.abs(dS[i, fwd, k])
            pk = r.max()
            if pk <= 0:
                continue
            out[2][i, k] = float(lags[int(np.argmax(r))])
            d1 = r > rel_peak * pk
            if d1.any():
                out[0][i, k] = float(lags[np.where(d1)[0][-1]])
            d2 = r > rel_stock * np.maximum(S_fact[fwd, k], 1.0)
            if d2.any():
                out[1][i, k] = float(lags[np.where(d2)[0][-1]])
    return out


# ---------------------------------------------------------------------------
# 8c — forward-mode ODE sensitivities
# ---------------------------------------------------------------------------
def make_forward_integrator(fit, *, rtol=1e-5, atol=1e-7, max_steps=20000):
    """The fit's integrator, re-composed with `adjoint=ForwardMode()`.

    `make_integrator` leaves diffrax's default `RecursiveCheckpointAdjoint`
    in place, which is a `custom_vjp` and therefore has no JVP rule, so
    `jax.jacfwd` through the fitted integrator raises.  The core is read-only,
    so the forward-mode integrator is *composed* from `make_rhs`,
    `_make_rhs_data`, `_make_Y0` and `ode_to_4obs` — the same recipe
    `zinc_fisher_lab.make_tight_integrator` uses — at the fit's own
    tolerances, so it is the same trajectory and not a tighter one.
    `forward_sensitivity` verifies that before differentiating.
    """
    import jax.numpy as jnp
    import diffrax as dfx
    import zinc_colloc_v5 as v5

    rhs = v5.make_rhs(fit.nn_eval)

    def integrate(params, data, S0, years):
        years = jnp.asarray(years)
        rhs_data = v5._make_rhs_data(data)

        def f(t, y, args):
            p, dd = args
            return rhs(y, t, p, dd)

        sol = dfx.diffeqsolve(
            dfx.ODETerm(f), dfx.Tsit5(), t0=years[0], t1=years[-1],
            dt0=jnp.asarray(0.1, dtype=years.dtype), y0=v5._make_Y0(S0),
            args=(params, rhs_data), saveat=dfx.SaveAt(ts=years),
            stepsize_controller=dfx.PIDController(rtol=rtol, atol=atol),
            max_steps=max_steps, adjoint=dfx.ForwardMode())
        Y = sol.ys
        S_ode = Y[:, :v5.N_ODE_STOCKS]
        C = Y[:, v5.N_ODE_STOCKS:]
        return v5.ode_to_4obs(S_ode), C[1:] - C[:-1], S_ode[:, 2:2 + v5.N_COHORTS]

    return integrate


def forward_sensitivity(fit, params, ctx, *, drivers=None, eps_sd=1.0):
    """`d S(t) / d theta_j` and `d F(t) / d theta_j` for the exogenous drivers.

    `theta_j` shifts driver `j`'s whole raw history by `theta_j` within-sample
    SDs, additively, so `theta` is dimensionless and one unit is the same size
    of move as 8b's bump amplitude convention.  `jax.jacfwd`
    over the 13-vector integrates the sensitivity system alongside the state,
    which is what the spec asks for; it is done for the drivers only, never
    for the 2 250 live weights.

    The trajectory produced by the `ForwardMode` integrator is checked against
    the fit's own `integrate_aug` first, and the max relative gap is returned
    so a reader can see it is the same solve.
    """
    import jax
    import jax.numpy as jnp

    cfg, t_full, X_full, cols, cut, _wd = ctx
    if drivers is None:
        drivers = list(cols)
    jidx = np.array([list(cols).index(dn) for dn in drivers], int)
    sd = np.array([np.std(X_full[:, j]) for j in jidx], float)
    years = np.asarray(fit.data_all["years"], float).ravel()

    fwd = make_forward_integrator(fit)
    data0 = dict(fit.data_all)

    # same-trajectory check against the fit's own integrator
    S_ref, F_ref, _ = fit.integrate_aug(params, data0, data0["stocks_obs"][0],
                                        data0["years"])
    S_fwd, F_fwd, _ = fwd(params, data0, data0["stocks_obs"][0], data0["years"])
    gap_S = float(np.max(np.abs(np.asarray(S_fwd) - np.asarray(S_ref))
                         / np.maximum(np.abs(np.asarray(S_ref)), 1.0)))
    gap_F = float(np.max(np.abs(np.asarray(F_fwd) - np.asarray(F_ref))
                         / np.maximum(np.abs(np.asarray(F_ref)), 1.0)))

    X_j = jnp.asarray(X_full)
    t_full_j = jnp.asarray(t_full)
    years_j = jnp.asarray(years)
    orders = tuple(sorted(set(int(o) for o in cfg.get("exog_feature_orders", (0,)))))
    do_log1p = bool(cfg.get("exog_log1p", True))
    diff_pad = cfg.get("exog_diff_pad", "edge")
    tgt = np.clip(np.searchsorted(t_full, years), 0, max(t_full.size - 1, 0))
    tgt_j = jnp.asarray(tgt)

    def _preprocess(X):
        """`preprocess_exog` in JAX, for the `anchor_v4` settings only.

        `exog_detrend` is false and `exog_pca_components` is None in
        `anchor_v4`, so the transform is log1p followed by the concatenated
        difference orders with edge padding — reproduced here because the
        core's version is NumPy and cannot be differentiated through.  The
        reproduction is verified bit-for-bit against `preprocess_exog` in
        `forward_sensitivity`'s return value (`gap_exog`).
        """
        Xs = jnp.log1p(jnp.maximum(X, 0.0)) if do_log1p else X
        feats = []
        for o in orders:
            if o == 0:
                Xo = Xs
            else:
                D = jnp.diff(Xs, n=o, axis=0)
                pad = (jnp.repeat(D[:1], o, axis=0) if diff_pad == "edge"
                       else jnp.zeros((o, Xs.shape[1]), Xs.dtype))
                Xo = jnp.concatenate([pad, D], axis=0)
            feats.append(Xo[tgt_j])
        return jnp.concatenate(feats, axis=1) if len(feats) > 1 else feats[0]

    gap_exog = float(np.max(np.abs(np.asarray(_preprocess(X_j))
                                   - np.asarray(fit.data_all["exog_values"]))))

    jidx_j, sd_j = jnp.asarray(jidx), jnp.asarray(sd)

    def traj(theta):
        # theta_j = 1 shifts driver j's entire raw history up by one
        # within-sample SD.  Additive, so it matches 8b's bump convention
        # (amplitude = a fraction of the series SD) and the two packages'
        # magnitudes are directly comparable.
        X = X_j.at[:, jidx_j].add((theta * sd_j)[None, :])
        data = dict(data0)
        data["exog_values"] = _preprocess(X)
        S, F, _ = fwd(params, data, data0["stocks_obs"][0], years_j)
        return jnp.concatenate([S.reshape(-1), F.reshape(-1)])

    theta0 = jnp.zeros(len(jidx))
    J = np.asarray(jax.jacfwd(traj)(theta0))
    nS = years.size * 4
    dS = J[:nS].reshape(years.size, 4, len(jidx))
    dF = J[nS:].reshape(years.size - 1, -1, len(jidx))
    return dict(years=years, drivers=list(drivers), driver_index=jidx,
                dS=dS, dF=dF, sd=sd, gap_S=gap_S, gap_F=gap_F,
                gap_exog=gap_exog, S=np.asarray(S_ref), F=np.asarray(F_ref))


# ---------------------------------------------------------------------------
# 8d — block-permutation importance
# ---------------------------------------------------------------------------
def circular_block_permute(x, block_len, rng):
    """Circular block bootstrap of a 1-D series (Politis-Romano).

    An i.i.d. shuffle destroys the autocorrelation that makes a time series a
    time series, and the resulting "importance" measures nothing; blocks of
    ~5 yr keep the local structure and only move it in time, which is what
    the spec asks for.
    """
    x = np.asarray(x, float)
    n = x.size
    nb = int(np.ceil(n / block_len))
    starts = rng.integers(0, n, size=nb)
    out = np.concatenate([np.take(x, np.arange(s, s + block_len), mode="wrap")
                          for s in starts])
    return out[:n]


def permutation_seed(fit, params, ctx, *, n_perm=N_PERM, block_len=BLOCK_LEN_YR,
                     rng_seed=0, drivers=None, mach=None):
    """Degradation in stock, flow and coefficient fit when a driver is
    block-permuted, with no refit.

    Reported against the factual metrics on the same seed, and against the
    null distribution the 200 repeats trace out, so "important" means
    "outside its own null" rather than "bigger than the others".
    """
    cfg, t_full, X_full, cols, cut, _wd = ctx
    if drivers is None:
        drivers = list(cols)
    integrate, coeffs = mach or machinery(fit)
    d = fit.data_all
    years = np.asarray(d["years"], float).ravel()
    S_obs = np.asarray(d["stocks_obs"], float)
    F_obs = np.asarray(d["flows_obs"], float)
    obs_idx = np.asarray(fit.flow_obs_to_pred_idx, int)
    a_obs = np.asarray(d["alpha_obs"], float)

    data0, _ = data_with_drivers(fit, cfg, X_full, t_full, cut)
    S0, F0, _ = integrate(params, data0)
    C0 = np.asarray(coeffs(params, data0, jnp_(S_obs)))

    def metrics(S, F, C):
        S, F = np.asarray(S), np.asarray(F)
        return dict(
            stock=_rel_rmse(S, S_obs),
            flow=_rel_rmse(F[:, obs_idx], F_obs),
            alpha=_rel_rmse(C[:, ALPHA_SLICE], a_obs))

    base = metrics(S0, F0, C0)
    rng = np.random.default_rng(int(rng_seed))
    rows = []
    for dn in drivers:
        j = list(cols).index(dn)
        vals = {k: np.zeros(n_perm) for k in ("stock", "flow", "alpha")}
        drift = np.zeros(n_perm)
        for r in range(n_perm):
            X = np.array(X_full, float, copy=True)
            X[:, j] = circular_block_permute(X_full[:, j], block_len, rng)
            data, _ = data_with_drivers(fit, cfg, X, t_full, cut)
            S, F, _ = integrate(params, data)
            C = np.asarray(coeffs(params, data, jnp_(S_obs)))
            m = metrics(S, F, C)
            for k in vals:
                vals[k][r] = m[k]
            drift[r] = float(np.max(np.abs(C[:, ALPHA_SLICE] - C0[:, ALPHA_SLICE])
                                    / np.maximum(np.abs(C0[:, ALPHA_SLICE]), 1e-12)))
        rows.append(dict(driver=dn, driver_index=int(j),
                         **{f"base_{k}": base[k] for k in base},
                         **{f"perm_{k}_median": float(np.median(vals[k])) for k in vals},
                         **{f"perm_{k}_q05": float(np.percentile(vals[k], 5)) for k in vals},
                         **{f"perm_{k}_q95": float(np.percentile(vals[k], 95)) for k in vals},
                         alpha_max_rel_shift=float(np.median(drift))))
    return dict(rows=rows, base=base, n_perm=int(n_perm), block_len=int(block_len))


def _rel_rmse(pred, obs):
    p, o = np.asarray(pred, float), np.asarray(obs, float)
    m = np.isfinite(p) & np.isfinite(o)
    if not m.any():
        return np.nan
    return 100.0 * np.sqrt(np.mean((p[m] - o[m]) ** 2)) / max(np.mean(np.abs(o[m])), 1e-12)


# ---------------------------------------------------------------------------
# 8e — global sensitivity over the interpretable structural parameters
# ---------------------------------------------------------------------------
# The nine free structural degrees of freedom.  The other eight entries of
# `zinc_circ_lab.PARAM_NAMES[:17]` are either pinned to data by `anchor_v4`
# (tau_ref, tau_waelz, tau_diss, frac_fu_loss, frac_eu_loss — zero across-seed
# spread, so they are constants here) or determined by closure
# (frac_fu_out, frac_eu_into, f_cohort_44yr).  Sampling a simplex's components
# independently would put mass outside the simplex and make every index
# meaningless, so the residuals are closed rather than drawn.
FREE_PARAMS = ["alpha_cc", "alpha_refc", "alpha_win", "alpha_dr", "tau_olds",
               "frac_fu_new", "frac_eu_new", "f_cohort_10yr", "f_cohort_20yr"]


def struct_bounds(weights_dir=WEIGHTS_DIR_DEFAULT, year_index=None, k=2.0):
    """A joint distribution over the free structural coefficients, stated.

    Independent uniforms on `median +- k * SD` across the 35 `anchor_v4` seeds
    at one year, intersected with `zinc_circ_lab`'s feasible box, over the nine
    free dimensions of `FREE_PARAMS`; the simplex residuals are closed and the
    data-pinned coefficients are held at their (seed-invariant) values.

    Independence across the nine is the judgement call, and it is the wrong
    one in detail: WP-2c measured the seeds' coefficients as correlated, so
    the marginals here are honest while the joint is not.  `run_wp8.py`
    therefore reports how far the indices move when the box is rescaled by
    `k in {1, 2, 4}`, which is the sensitivity-to-the-specification the spec
    asks for; the *ranking*, not the level, is what the note quotes.
    """
    import zinc_circ_lab as C
    import glob as _glob

    P = []
    for p in sorted(_glob.glob(os.path.join(weights_dir, "A_seed*.npz"))):
        d = np.load(p, allow_pickle=True)
        yi = (d["years"].size - 1) if year_index is None else int(year_index)
        P.append(C.pack_params(dict(
            alphas=d["alphas"][yi], taus=d["taus"][yi],
            frac_fu=d["frac_fu"][yi], frac_eu=d["frac_eu"][yi],
            f_cohort=d["f_cohort"][yi]), d["mu_cohorts"]))
    P = np.asarray(P)
    med, sd = np.median(P, axis=0), P.std(axis=0, ddof=1)
    lo = np.maximum(med - k * sd, C.FEASIBLE_LO)
    hi = np.minimum(med + k * sd, C.FEASIBLE_HI)
    names = list(C.PARAM_NAMES[:17])
    free_idx = np.array([names.index(n) for n in FREE_PARAMS], int)
    return dict(names=names, free_names=list(FREE_PARAMS), free_idx=free_idx,
                lo=lo[:17][free_idx], hi=np.maximum(hi[:17][free_idx],
                                                    lo[:17][free_idx] + 1e-9),
                base=med[:17], sd=sd[:17][free_idx], mu=med[17:],
                n_seeds=P.shape[0], k_sd=float(k),
                year_index=(-1 if year_index is None else int(year_index)))


def expand(x_free, bounds):
    """The full 17-vector from the nine free coordinates, simplexes closed."""
    P = np.asarray(bounds["base"], float).copy()
    P[bounds["free_idx"]] = np.asarray(x_free, float)
    nm = bounds["names"]
    P[nm.index("frac_fu_out")] = max(
        1.0 - P[nm.index("frac_fu_new")] - P[nm.index("frac_fu_loss")], 1e-9)
    P[nm.index("frac_eu_into")] = max(
        1.0 - P[nm.index("frac_eu_new")] - P[nm.index("frac_eu_loss")], 1e-9)
    c10, c20 = P[nm.index("f_cohort_10yr")], P[nm.index("f_cohort_20yr")]
    tot = c10 + c20
    if tot >= 1.0 - 1e-6:                      # renormalise onto the simplex
        c10, c20 = c10 / (tot + 1e-6), c20 / (tot + 1e-6)
        P[nm.index("f_cohort_10yr")], P[nm.index("f_cohort_20yr")] = c10, c20
    P[nm.index("f_cohort_44yr")] = max(1.0 - c10 - c20, 1e-9)
    return P


def struct_outputs(P17, mu):
    """The Chapter 2 indicators as a function of the 17 structural parameters.

    Outputs, all frozen-time and all from `zinc_circ_lab`, so WP-8e measures
    the same objects WP-2b / WP-2d report:
        spectral_abscissa, dominant_timescale, tau_from_conc,
        upsilon_from_conc, prob_reaches_use, tau_from_scrap
    """
    import zinc_circ_lab as C

    P = np.concatenate([np.asarray(P17, float), np.asarray(mu, float)])
    P = C._clip_to_feasible(P)
    A = C.assemble_from_params(P)
    try:
        # `frozen_indicators` hard-asserts the M-matrix property, which a draw
        # from the box can violate; `fundamental(check=False)` is the same
        # object without the assertion and `flow_rows` supplies the
        # `use_entry` row, so the mathematics stays the core lab's.
        N = C.fundamental(A, check=False, on_singular="nan")
        tau = np.ones(N.shape[-1]) @ N
        ups = C.flow_rows(A)["use_entry"] @ N
        ev = np.linalg.eigvals(A)
        abscissa = float(np.max(ev.real))
        try:
            pr = float(np.asarray(C.prob_reaches_use(A), float)[0])
        except Exception:
            pr = np.nan
        return np.array([abscissa, 1.0 / max(abs(abscissa), 1e-12),
                         float(tau[0]), float(ups[0]), pr, float(tau[-1])])
    except Exception:
        return np.full(6, np.nan)


STRUCT_OUT_NAMES = ["spectral_abscissa", "dominant_timescale", "tau_from_conc",
                    "upsilon_from_conc", "prob_reaches_use", "tau_from_scrap"]


def morris(bounds, *, n_traj=MORRIS_TRAJ, levels=MORRIS_LEVELS, rng_seed=0):
    """Morris elementary effects over the free coordinates: `mu*`, `mu`, `sigma`."""
    lo, hi, mu = bounds["lo"], bounds["hi"], bounds["mu"]
    p = lo.size
    delta = levels / (2.0 * (levels - 1))
    rng = np.random.default_rng(rng_seed)
    EE = np.full((n_traj, p, len(STRUCT_OUT_NAMES)), np.nan)
    for r in range(n_traj):
        x = rng.integers(0, levels // 2, size=p) / (levels - 1.0)
        order = rng.permutation(p)
        y_prev = struct_outputs(expand(lo + x * (hi - lo), bounds), mu)
        for j in order:
            xn = x.copy()
            xn[j] = x[j] + delta if x[j] + delta <= 1.0 else x[j] - delta
            y = struct_outputs(expand(lo + xn * (hi - lo), bounds), mu)
            EE[r, j] = (y - y_prev) / ((xn[j] - x[j]) * max(hi[j] - lo[j], 1e-12))
            x, y_prev = xn, y
    return dict(mu_star=np.nanmean(np.abs(EE), axis=0),
                mu=np.nanmean(EE, axis=0), sigma=np.nanstd(EE, axis=0),
                names=bounds["free_names"], out_names=list(STRUCT_OUT_NAMES),
                n_traj=int(n_traj), levels=int(levels))


def sobol(bounds, *, n_base=SOBOL_BASE_N, rng_seed=0):
    """Saltelli first-order and total Sobol indices over the free coordinates."""
    lo, hi, mu = bounds["lo"], bounds["hi"], bounds["mu"]
    p, m = lo.size, len(STRUCT_OUT_NAMES)
    rng = np.random.default_rng(rng_seed)
    A = lo + rng.random((n_base, p)) * (hi - lo)
    B = lo + rng.random((n_base, p)) * (hi - lo)

    def ev(M):
        return np.stack([struct_outputs(expand(M[i], bounds), mu)
                         for i in range(M.shape[0])])

    YA, YB = ev(A), ev(B)
    S1, ST = np.zeros((p, m)), np.zeros((p, m))
    varY = np.nanvar(np.concatenate([YA, YB]), axis=0, ddof=1)
    for j in range(p):
        AB = A.copy(); AB[:, j] = B[:, j]
        YAB = ev(AB)
        S1[j] = np.nanmean(YB * (YAB - YA), axis=0) / np.maximum(varY, 1e-30)
        ST[j] = 0.5 * np.nanmean((YA - YAB) ** 2, axis=0) / np.maximum(varY, 1e-30)
    return dict(S1=S1, ST=ST, names=bounds["free_names"],
                out_names=list(STRUCT_OUT_NAMES), n_base=int(n_base), var=varY)


# ---------------------------------------------------------------------------
# 8f — regime-break recovery
# ---------------------------------------------------------------------------
def _segment_costs(y, Z, min_seg):
    """`cost[i, j]` = residual sum of squares of `y` on `Z` over `[i, j]`."""
    n = y.size
    cost = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i + min_seg - 1, n):
            Zs, ys = Z[i:j + 1], y[i:j + 1]
            if Zs.shape[0] <= Zs.shape[1]:
                continue
            beta, *_ = np.linalg.lstsq(Zs, ys, rcond=None)
            r = ys - Zs @ beta
            cost[i, j] = float(r @ r)
    return cost


def bai_perron(y, *, Z=None, max_k=BREAK_MAX_K, min_seg=BREAK_MIN_SEG):
    """Least-squares multiple-break estimation with a per-segment regression.

    Exact dynamic programming over segment residual sums of squares (Bai &
    Perron 1998, 2003), `k` chosen by BIC.  `Z` is the within-segment design
    matrix; `None` means a column of ones, i.e. the pure mean-shift model.
    Passing `Z = [1, t]` gives the segmented-trend model, which is the right
    object for a trending coefficient path: a "regime break" in an alpha
    trajectory is a change in its growth rate, not a jump in its level, and a
    mean-shift model applied to a trending series finds spurious breaks
    wherever the trend has travelled far enough.

    Returns the selected break indices (last observation of each segment
    except the final one), the BIC path and the SSR path.
    """
    y = np.asarray(y, float).ravel()
    n = y.size
    Z = np.ones((n, 1)) if Z is None else np.asarray(Z, float).reshape(n, -1)
    q = Z.shape[1]
    cost = _segment_costs(y, Z, min_seg)

    best = {0: ([], float(cost[0, n - 1]))}
    tabs = {0: {j: (float(cost[0, j]), []) for j in range(min_seg - 1, n)
                if np.isfinite(cost[0, j])}}
    for k in range(1, max_k + 1):
        tab = {}
        for j in range(min_seg * (k + 1) - 1, n):
            cands = [(tabs[k - 1][b][0] + cost[b + 1, j], b) for b in tabs[k - 1]
                     if b + min_seg <= j and np.isfinite(cost[b + 1, j])]
            if not cands:
                continue
            v, b = min(cands)
            tab[j] = (v, tabs[k - 1][b][1] + [b])
        if (n - 1) not in tab:
            break
        tabs[k] = tab
        best[k] = (tab[n - 1][1], float(tab[n - 1][0]))
    bic = {}
    for k, (brk, ssr) in best.items():
        pnum = q * (k + 1) + k              # k+1 segment fits + k break dates
        bic[k] = n * np.log(max(ssr, 1e-30) / n) + pnum * np.log(n)
    k_hat = min(bic, key=bic.get)
    return dict(k=int(k_hat), breaks=list(best[k_hat][0]),
                bic={int(k): float(v) for k, v in bic.items()},
                ssr={int(k): float(v) for k, (_b, v) in best.items()},
                n_params_per_segment=int(q))


def cusum(y):
    """OLS-CUSUM of a mean model; returns the standardised path and its max.

    The 5% critical value for the sup of the standardised OLS-CUSUM is 1.358
    (Brown, Durbin & Evans 1975), and the argmax is the CUSUM break estimate.
    """
    y = np.asarray(y, float).ravel()
    n = y.size
    r = y - y.mean()
    s = np.sqrt(max(np.sum(r ** 2) / max(n - 1, 1), 1e-30))
    path = np.cumsum(r) / (s * np.sqrt(n))
    return dict(path=path, sup=float(np.max(np.abs(path))),
                argmax=int(np.argmax(np.abs(path))), crit05=1.358)


def breaks_for_series(years, y, *, max_k=BREAK_MAX_K, min_seg=5):
    """Break dates for one coefficient series, under two nested models.

    `trend`  segmented linear trend on `log y` against `[1, t]`.  The primary
             readout: a break is a change in the coefficient's growth rate.
             Break index `b` is the last observation of a segment, so the
             break year reported is `years[b]`, the last year of the old
             regime; the transition is `(years[b], years[b+1]]`.
    `diff`   mean shift on `diff(log y)`.  A stricter test of the same idea
             that needs no trend nuisance parameters, reported alongside
             because a package that only shows the model that found something
             is not reporting.

    CUSUM on the differenced series is the third, model-free cross-check.
    """
    y = np.asarray(y, float).ravel()
    t = np.asarray(years, float).ravel()
    ly = np.log(np.maximum(y, 1e-12))
    Z = np.column_stack([np.ones_like(t), t - t.mean()])
    tr = bai_perron(ly, Z=Z, max_k=max_k, min_seg=min_seg)
    dl = np.diff(ly)
    df = bai_perron(dl, max_k=max_k, min_seg=BREAK_MIN_SEG)
    cs = cusum(dl)
    return dict(k=tr["k"],
                break_years=[float(t[b]) for b in tr["breaks"]],
                bic=tr["bic"], ssr=tr["ssr"],
                k_diff=df["k"],
                break_years_diff=[float(t[b + 1]) for b in df["breaks"]],
                cusum_sup=cs["sup"],
                cusum_year=float(t[cs["argmax"] + 1]),
                cusum_signif=bool(cs["sup"] > cs["crit05"]))


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------
def check(verbose=True, weights_dir=WEIGHTS_DIR_DEFAULT):
    """Resolve drivers and the input layout, and verify every instrument's
    precondition, before any work is done."""
    import zinc_colloc_v5 as v5

    info = alab.check(verbose=False)
    install(v5)
    cfg = load_anchor_config()
    fit, ctx = build_context(cfg, weights_dir)
    lay = layout_index(fit, cfg)
    params = load_params(0, weights_dir)

    # the d/dS block, computed rather than asserted
    att = attribution_seed(fit, params, lay, ig_steps=8)
    dS_max = float(np.max(np.abs(att["dS"])))
    comp = float(np.nanmax(att["completeness"]))

    n_w = len([f for f in os.listdir(weights_dir)
               if f.startswith("A_seed") and f.endswith(".npz")])
    info.update(
        drivers=lay["drivers"], n_universe=lay["n_universe"],
        orders=lay["orders"], n_features=lay["n_features"],
        input_dim=lay["input_dim"], n_weight_dumps=int(n_w),
        dalpha_dS_max_abs=dS_max, ig_completeness_max=comp,
        learned_slots=[COEF_NAMES[i] for i, b in enumerate(learned_mask(fit)) if b],
        pinned_slots=[COEF_NAMES[i] for i, b in enumerate(learned_mask(fit)) if not b],
        events=[e[0] for e in EVENTS], n_perm=N_PERM, block_len=BLOCK_LEN_YR,
        patches=list(PATCHES))

    if verbose:
        print("=" * 74)
        print("zinc_xai_lab --check   (WP-8 explainability suite)")
        print("=" * 74)
        for label, (got, ok) in info["digests"].items():
            print(f"  {label:20s} md5 {got}  {'OK' if ok else 'MISMATCH'}")
        print(f"  patches applied      : {PATCHES or 'none (WP-8 needs none)'}")
        print(f"  weights              : {n_w} dumps in {weights_dir}")
        print(f"\n  resolved drivers (canonical order, {lay['n_universe']}):")
        for i, c in enumerate(lay["drivers"]):
            print(f"    [{i:2d}] {c}")
        print(f"\n  exog_feature_orders  : {lay['orders']}   "
              f"(order 1 = first difference)")
        print(f"  input_dim            : 1 (t) + 4 (S) + {lay['n_features']} "
              f"= {lay['input_dim']}")
        if lay["input_dim"] != 23:
            print(f"  !! input_dim is {lay['input_dim']}, not the 23 asserted by "
                  f"CLAUDE.md rule 2 (SCHEMA §9 flag 1, unresolved).")
            print(f"     spec §1's layout assumes 3 orders and "
                  f"CANON_DIM = 1+4+3N; the fit has {len(lay['orders'])}.")
        print(f"  use_time_input       : {lay['use_time_input']:.0f}   "
              f"use_stock_input : {lay['use_stock_input']:.0f}")
        print(f"\n  8a  max |d coef / d S| over all years, seed 0 : {dS_max:.3e}")
        if dS_max == 0.0:
            print("      -> exactly zero.  `use_stock_input: false` makes the "
                  "state block structurally dead;")
            print("         the state-dependence the spec wants is in 8c's "
                  "propagated response, not in 8a.")
        print(f"  8a  max IG completeness error (8-step probe)  : {comp:.3e}")
        print(f"\n  learned coefficient slots ({sum(learned_mask(fit))}): "
              f"{info['learned_slots']}")
        print(f"  pinned  coefficient slots ({N_COEF - sum(learned_mask(fit))}): "
              f"{info['pinned_slots']}")
        print(f"\n  8d  block permutation : {N_PERM} reps, block {BLOCK_LEN_YR} yr")
        print(f"  8f  pre-registered events ({len(EVENTS)}):")
        for name, y0, y1 in EVENTS:
            print(f"      {y0}-{y1}  {name}")
        print("=" * 74)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description="WP-8 explainability lab")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--weights", default=WEIGHTS_DIR_DEFAULT)
    args = ap.parse_args(argv)
    import zinc_colloc_v5 as v5
    integrity_check()
    install(v5)
    check(weights_dir=args.weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
