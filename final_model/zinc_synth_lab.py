#!/usr/bin/env python3
"""
zinc_synth_lab.py — lab module for WP-3: the synthetic twin generator
====================================================================

WP-3 builds the shared infrastructure that WP-4 (identifiability and
aliasing), WP-5 (interpretability head-to-head) and parts of WP-8 stand on: a
system whose transfer coefficients are *known analytic functions*, integrated
through the real `zinc_colloc_v5` right-hand side, and then observed through
the same operators the real dataset is built with.

Why a twin at all
-----------------
Every question of the form "how much of the 64% α relRMSE is estimator error
and how much is target construction?" is unanswerable on the real data,
because `alpha_obs` is the only α anyone has.  On the twin the true
coefficient path is known in closed form, so the two can be separated by
subtraction.  That is this module's whole purpose, and the α-target bias
decomposition in `alpha_target_bias` is the deliverable the spec singles out.

Pattern (CLAUDE.md rule 1)
--------------------------
`zinc_colloc_v5.py` is imported and never edited; its MD5 and the MD5 of the
`anchor_v4` data copy are pinned through `zinc_alpha_lab.integrity_check`.
Unlike WP-1/2/4c this package *does* need one import-time patch, because the
whole point is to feed the real training pipeline something other than the
real spreadsheet:

    PATCHES = ["zinc_colloc_v5.load_zinc_data -> arm-aware dispatcher"]

`install()` rebinds `v5.load_zinc_data` to a dispatcher that returns the real
loader's output **unless a twin has been armed** with `arm(dataset)`.  With no
twin armed the patched module is bit-identical to the unpatched one, which
`check()` verifies by comparing every array of the real load through the
dispatcher against the original function.  `disarm()` restores plain
behaviour.  Nothing inside the core is otherwise touched: the truth
coefficients enter through a closure with the *same signature* as
`make_nn_eval`'s output, so `make_rhs`, `_make_Y0`, `_make_rhs_data` and
`ode_to_4obs` are the core's own and the twin is integrated by the real RHS
rather than by a reimplementation of it.

The truth
---------
Learned coefficients are analytic in `(t, drivers(t), S(t))`.  Levels are
anchored on the `anchor_v4` 35-seed ensemble medians (`analysis/wp2a`) so the
twin is a plausible zinc cycle rather than an arbitrary ODE; the *shapes* are
this module's own and are fully documented in `TRUTH`.  Each channel carries a
`tanh` trend whose inflection is a **pre-registered turning point**:

    alpha_cc    A=3.767  t*=2003 w=9  m=+0.45   + 0.18 z[TC]     - 0.12 z[Energy]
    alpha_refc  A=8.198  t*=1997 w=7  m=+0.50   + 0.22 z[GDP]    - 0.15 z[ZnPrice]
    alpha_win   A=0.142  t*=1999 w=8  m=+0.55   + 0.25 z[ScrapPPI] - 0.10 z[Energy]
    alpha_dr    A=0.433  t*=2008 w=6  m=+0.32   + 0.20 z[ScrapPPI] + 0.10 z[ChinaVA]

Constants were chosen once, for plausibility against the `anchor_v4` ensemble
median coefficient paths, and then frozen; `realism()` reports how far the
resulting trajectory sits from the real stocks rather than being used to tune
it further.  Nothing in the twin is fitted to the real alpha target, which is
the whole point — the target is the object under test.

`z[.]` is the log1p'd driver level, z-scored over 1980–2019, linearly
interpolated in continuous time — the same interpolation `exog_fn` applies.
The trend term is mean-centred over the window so `A_k` *is* the geometric
mean of the channel.  τ_olds and the two manufacturing split degrees of
freedom are logistic in the same style; `f_cohort` is a softmax of two drifting
logits.  See `TRUTH` for every constant.

Representability, deliberately
------------------------------
`anchor_v4` runs with `use_stock_input: false`, so its α cannot depend on the
state at all.  The default arm (`base`) therefore has **no state term**: it
lies inside the anchor architecture's representable set, which is the
precondition for the spec's acceptance test to mean anything.  State
dependence and sub-annual structure are separate arms:

    base    trend + drivers.                       representable.  acceptance arm.
    season  base + four sub-annual components on alpha_refc, two of them at
            exactly 1/yr and 2/yr (annihilated by annual integration) and two
            at 1.35/yr and 2.70/yr (attenuated by |sinc| then aliased).
            Not representable from annual data — that is WP-4a's subject.
    state   base + a capacity gate exp(-gamma (S_ref/S_ref_bar - 1)) on
            alpha_refc.  Not representable under `use_stock_input: false` —
            this is the WP-4b misspecification arm.

Pinned coefficients are not estimands
-------------------------------------
`anchor_v4` pins cp, tau_ref, tau_waelz, tau_diss, frac_fu_loss and
frac_eu_loss to the data, and the RHS reads them back through `_interp_cp` /
`_interp_tau` as **piecewise constants on (Y-1, Y]**.  The twin therefore
takes them from the real dataset unchanged and keeps them piecewise-constant
annual.  Two consequences, both wanted: the pins a refit sees are exactly the
pins the truth used (no spurious misspecification), and the twin's
concentrate input is the real ILZSG series, so its trajectory is anchored to
the real system's forcing.  `frac_fu_new` and `frac_eu_new` are generated as
`sigma(t) * (1 - loss_pinned(t))`, which is exactly the `loss_pinned` branch
of `_simplex_value` (v5:741), so the twin sits inside the model's simplex
parameterisation by construction rather than by luck.

Observation operators (spec §1 — "not optional here")
-----------------------------------------------------
One tight dense solve produces the continuous truth on a grid of
`N_SUB = 96` points per year, saving the **cumulative** flow accumulators.
Because 96 is divisible by 4 and 12, every observation window this module
supports (annual, quarterly, monthly, 2/5/10-yearly) has endpoints on that
grid, so every frequency is an *exact re-aggregation of one trajectory*
rather than a separate solve.  Then, per window `(t_{i-1}, t_i]`:

  stocks        point-in-time    `S_fine[idx_i]`
  flows         period integral  `C_fine[idx_i] - C_fine[idx_{i-1}]`  (exact)
  cp            period *rate*    integral / Delta                      (see below)
  tau_sup       exposure-weighted ratio of two flow integrals
  alpha_obs     `v5._build_empirical_alphas` on the sampled arrays

**A convention asymmetry that matters at Delta != 1 and that the core does not
document.**  `flows_obs` is compared against `F_int` from the accumulator, so
it must be a window *integral*.  `cp_obs` is read by `_interp_cp` as the value
of `cp(t)` on the window, so it must be a window-mean *rate* if the identity
`int cp dt = cp_obs * Delta` is to hold.  At Delta = 1 the two coincide
numerically and nothing in the codebase distinguishes them; at any other
Delta they differ by a factor of Delta.  This module stores cp as a rate and
flows as integrals, and `verify_operators` checks the cp identity to solver
tolerance at every Delta.  Any frequency experiment that misses this measures
the factor Delta and calls it information loss.

The alpha target is rebuilt, not rescaled
-----------------------------------------
`alpha_obs` is `F_integral / (trapezoid exposure)` (v5:864).  Its bias against
the truth splits exactly into two terms, both computable here because
`alpha_true` is known:

    F_int / trapz(S)  =  [ int alpha S dt / int S dt ]   exposure-weighted mean
                         x [ int S dt / trapz(S) ]        quadrature error
    and the estimand a modeller reads it as is the *unweighted* mean
    (1/Delta) int alpha dt, which differs from the exposure-weighted mean by
    Cov_w(alpha, S) / mean_w(S) over the window.

`alpha_target_bias` reports all three pieces per channel and per Delta, with
the within-window F-S correlation alongside, which is the WP-1a secondary
hypothesis made numerical.

Noise
-----
`noise_scale=1.0` applies lognormal multiplicative noise to sampled stocks and
flow integrals with per-series sigma equal to the median across the 35
`anchor_v4` seeds of the Stage B log-residual SD on the trainval window
(`calibrate_noise`).  Those residuals conflate model error with observation
error, so they are an *upper bound* on the latter; that is the conservative
direction and it is what "calibrated to the residual distribution of the real
fit" can mean without a second, independent dataset.  tau observations inherit
noise through the flow integrals they are ratios of, which is how a real MFA
compilation propagates it; pinned tau slots stay exact because they are inputs.
`noise_scale=0.0` is the acceptance arm.

CLI
---
    python zinc_synth_lab.py --check
    python zinc_synth_lab.py --generate --arms base,season,state
    python zinc_synth_lab.py --generate --arms base --deltas 1.0 --noise 0.0
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import resource
import sys
import time

import numpy as np

import zinc_alpha_lab as alab
from zinc_alpha_lab import integrity_check, load_anchor_config      # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR_DEFAULT = os.path.join(HERE, "analysis", "synth")
WEIGHTS_DIR_DEFAULT = os.path.join(HERE, "analysis", "wp2a")
ANCHOR_DIR = os.path.join(HERE, "anchor_v4")

# Dense grid: 96 sub-steps per year.  96 = 2^5 * 3, so annual, quarterly and
# monthly window endpoints all land exactly on grid nodes and every frequency
# is an exact re-aggregation of the same solve.
N_SUB = 96

# Window widths in years.  Sub-annual first, then annual, then the coarsenings
# WP-6b/WP-11a want.  1/12 is representable on the grid (96/12 = 8).
DELTAS_DEFAULT = (0.25, 1.0, 2.0, 5.0, 10.0)

TIGHT_RTOL, TIGHT_ATOL, TIGHT_MAXSTEPS = 1e-10, 1e-12, 4_000_000

# Drivers the truth uses, resolved by explicit name (CLAUDE.md rule 2 — never
# by sheet position).  Keys are the short names used in `TRUTH`.
DRIVER_ALIASES = {
    "ZnPrice":  "Zinc real price",
    "Energy":   "Energy index",
    "GDP":      "GDP",
    "ScrapPPI": "Producer Price Index of Non-ferrous Scrap",
    "TC":       "Real Treatment Charges",
    "ChinaVA":  "China All Industry Value Added (Constant 2015 USD)",
}

ALPHA_NAMES = ["alpha_cc", "alpha_refc", "alpha_win", "alpha_dr"]
TAU_SUP_NAMES = ["tau_ref", "tau_waelz", "tau_olds", "tau_diss",
                 "frac_fu_new", "frac_fu_loss", "frac_eu_new", "frac_eu_loss"]

# ---------------------------------------------------------------------------
# The truth.  Every constant that defines the twin lives here and nowhere else.
# ---------------------------------------------------------------------------
TRUTH = dict(
    # ---- alpha: A_k * exp( m_k*(tanh((t-tstar)/w) - centre) + sum_j beta_j z_j ) ----
    # A_k are the anchor_v4 35-seed ensemble geometric means (analysis/wp2a).
    # tstar are the pre-registered turning points; they are what a break test
    # on the twin has to recover, and they are fixed before any test is run.
    alpha=dict(
        alpha_cc=dict(A=3.767, tstar=2003.0, w=9.0, m=+0.45,
                      beta={"TC": +0.18, "Energy": -0.12}),
        alpha_refc=dict(A=8.198, tstar=1997.0, w=7.0, m=+0.50,
                        beta={"GDP": +0.22, "ZnPrice": -0.15}),
        alpha_win=dict(A=0.1417, tstar=1999.0, w=8.0, m=+0.55,
                       beta={"ScrapPPI": +0.25, "Energy": -0.10}),
        alpha_dr=dict(A=0.4332, tstar=2008.0, w=6.0, m=+0.32,
                      beta={"ScrapPPI": +0.20, "ChinaVA": +0.10}),
    ),
    # ---- tau_olds: sigmoid(logit(p0) + m*(tanh((t-tstar)/w) - centre) + beta.z) ----
    tau_olds=dict(p0=0.180, tstar=2002.0, w=8.0, m=+0.35,
                  beta={"ScrapPPI": +0.12}),
    # ---- manufacturing splits: the free sigma of the `loss_pinned` branch ----
    sigma_fu=dict(p0=0.0950, tstar=2005.0, w=10.0, m=+0.10,
                  beta={"GDP": +0.05}),
    sigma_eu=dict(p0=0.0560, tstar=1996.0, w=9.0, m=+0.12,
                  beta={"ZnPrice": -0.06}),
    # ---- f_cohort: softmax([l0, l1, 0]) with drifting logits ----
    # Calibrated to the ensemble median split, (0.104, 0.309, 0.576) in 1980
    # drifting to (0.133, 0.323, 0.498) in 2019: a slow shift of new zinc out
    # of the 44-yr construction cohort into the 10- and 20-yr ones.
    f_cohort=dict(l0=-1.517, l1=-0.528, d0=+0.196, d1=+0.095,
                  tstar=2000.0, w=10.0),
    # ---- arm: season.  (channel, frequency [1/yr], amplitude [nats], phase) ----
    # 1.0 and 2.0 /yr are harmonics of the annual window and are annihilated
    # by period integration (exact nulls of sinc(pi f Delta) at f = k/Delta).
    # 1.35 and 2.70 /yr are not, and survive attenuated by |sinc|.
    season=[("alpha_refc", 1.00, 0.12, 0.0),
            ("alpha_refc", 2.00, 0.08, np.pi / 2),
            ("alpha_refc", 1.35, 0.12, np.pi / 3),
            ("alpha_refc", 2.70, 0.08, 5 * np.pi / 4)],
    season_t0=1980.0,
    # ---- arm: state.  capacity gate on alpha_refc ----
    state=dict(channel="alpha_refc", gamma=0.35),
)

ARMS = ("base", "season", "state")

PATCHES: list[str] = []


# ---------------------------------------------------------------------------
# install / arm
# ---------------------------------------------------------------------------
_ARMED: dict | None = None
_ORIG_LOADER = None


def install(v5mod=None):
    """Rebind `v5.load_zinc_data` to the arm-aware dispatcher.

    Idempotent.  With no twin armed the dispatcher forwards to the original
    function, so an installed-but-disarmed module is behaviourally identical
    to an uninstalled one (`check()` verifies this array by array).
    """
    global _ORIG_LOADER
    if v5mod is None:
        import zinc_colloc_v5 as v5mod
    if _ORIG_LOADER is not None:
        return PATCHES
    _ORIG_LOADER = v5mod.load_zinc_data

    def _dispatch(xlsx_path, extra_exog_cols=None, extra_sheet="extra"):
        if _ARMED is None:
            return _ORIG_LOADER(xlsx_path, extra_exog_cols=extra_exog_cols,
                                extra_sheet=extra_sheet)
        return {k: (v.copy() if isinstance(v, np.ndarray) else
                    (list(v) if isinstance(v, list) else v))
                for k, v in _ARMED.items()}

    v5mod.load_zinc_data = _dispatch
    PATCHES.append("zinc_colloc_v5.load_zinc_data -> synth arm-aware dispatcher")
    return PATCHES


def arm(dataset):
    """Arm a synthetic dataset so the next `train_model` consumes it.

    `dataset` is either a `load_zinc_data`-shaped dict (as returned by
    `build_dataset`) or a path to one of this module's `.npz` dumps.
    """
    global _ARMED
    if isinstance(dataset, str):
        dataset = load_dataset(dataset)
    need = ("years", "stocks_obs", "cp_obs", "alpha_obs", "tau_sup_obs",
            "flows_obs", "flow_obs_names", "flow_obs_to_pred_idx",
            "exog_times", "exog_values", "exog_times_full",
            "exog_values_full", "exog_cols")
    missing = [k for k in need if k not in dataset]
    if missing:
        raise KeyError(f"armed dataset is missing {missing}")
    _ARMED = dataset
    return _ARMED


def disarm():
    global _ARMED
    _ARMED = None


def _rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024.0 ** 2) if sys.platform == "darwin" else r / 1024.0


# ---------------------------------------------------------------------------
# driver context
# ---------------------------------------------------------------------------
def driver_context(cfg=None):
    """Real annual grid, real pinned series, and the z-scored driver levels.

    Returns a dict of NumPy arrays.  Drivers are `log1p`'d and z-scored over
    the whole 1980-2019 window — *not* over the training core, because these
    are the truth's own coordinates and must not depend on a split the truth
    knows nothing about.  Resolution is by explicit name through
    `DRIVER_ALIASES` (CLAUDE.md rule 2).
    """
    import zinc_colloc_v5 as v5

    cfg = dict(cfg or load_anchor_config())
    loader = _ORIG_LOADER or v5.load_zinc_data
    raw = loader(cfg["xlsx_path"], extra_exog_cols=cfg.get("extra_exog_cols"))

    cols = list(raw["exog_cols"])
    missing = {k: n for k, n in DRIVER_ALIASES.items() if n not in cols}
    if missing:
        raise KeyError(f"drivers not resolvable by name: {missing}")

    years = np.asarray(raw["years"], float).ravel()
    X = np.asarray(raw["exog_values"], float)
    z = {}
    for short, name in DRIVER_ALIASES.items():
        col = np.log1p(np.maximum(X[:, cols.index(name)], 0.0))
        z[short] = (col - col.mean()) / max(col.std(), 1e-9)

    return dict(
        years=years, exog_cols=cols,
        z=z, z_names={k: DRIVER_ALIASES[k] for k in z},
        cp_obs=np.asarray(raw["cp_obs"], float),
        tau_sup_obs=np.asarray(raw["tau_sup_obs"], float),
        stocks_obs=np.asarray(raw["stocks_obs"], float),
        exog_values=X,
        exog_times_full=np.asarray(raw["exog_times_full"], float),
        exog_values_full=np.asarray(raw["exog_values_full"], float),
        flow_obs_names=list(raw["flow_obs_names"]),
        flow_obs_to_pred_idx=np.asarray(raw["flow_obs_to_pred_idx"], np.int32),
        flows_obs_real=np.asarray(raw["flows_obs"], float),
    )


def _centre(tstar, w, years):
    """Window mean of tanh((t - tstar)/w), so `A_k` is the geometric mean."""
    return float(np.mean(np.tanh((years - tstar) / w)))


# ---------------------------------------------------------------------------
# the truth closure
# ---------------------------------------------------------------------------
def make_truth_nn_eval(ctx, arm_name="base", *, season=None, state_gamma=None):
    """A closure with `make_nn_eval`'s exact output contract, evaluating the
    analytic truth instead of an MLP.

    Signature `f(params, t, S, exog_t, data)` -> dict with keys
    `cp, alphas, taus, frac_fu, frac_eu, f_cohort, raw_norm`.  `params` and
    `exog_t` are ignored: the truth closes over its own driver arrays so it is
    independent of `preprocess_exog`'s feature construction, which changes
    with the observation frequency.  Pinned quantities are read from `data`
    through the core's own `_interp_cp` / `_interp_tau`, so a twin integrated
    here and a refit reading the same `data` see identical pins.
    """
    import jax
    import jax.numpy as jnp
    import zinc_colloc_v5 as v5

    if arm_name not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm_name!r}")

    yrs = jnp.asarray(ctx["years"], float)
    Z = {k: jnp.asarray(v, float) for k, v in ctx["z"].items()}
    S_ref_bar = float(np.mean(ctx["stocks_obs"][:, 1]))

    if season is None:
        season = TRUTH["season"] if arm_name == "season" else []
    if state_gamma is None:
        state_gamma = TRUTH["state"]["gamma"] if arm_name == "state" else 0.0
    t0_season = float(TRUTH["season_t0"])

    def zval(short, t):
        return jnp.interp(t, yrs, Z[short])

    def _trend(spec, t):
        c = _centre(spec["tstar"], spec["w"], ctx["years"])
        return spec["m"] * (jnp.tanh((t - spec["tstar"]) / spec["w"]) - c)

    def _drivers(spec, t):
        out = 0.0
        for short, b in spec["beta"].items():
            out = out + b * zval(short, t)
        return out

    def _season_for(name, t):
        s = 0.0
        for (chan, f, a, ph) in season:
            if chan == name:
                s = s + a * jnp.sin(2.0 * jnp.pi * f * (t - t0_season) + ph)
        return s

    def _logistic(spec, t):
        p0 = spec["p0"]
        l0 = float(np.log(p0 / (1.0 - p0)))
        return jax.nn.sigmoid(l0 + _trend(spec, t) + _drivers(spec, t))

    A_specs = TRUTH["alpha"]
    coh = TRUTH["f_cohort"]
    c_coh = _centre(coh["tstar"], coh["w"], ctx["years"])

    def truth_eval(params, t, S, exog_t, data):
        S4 = jnp.maximum(jnp.asarray(S), 0.0)

        alphas = []
        for name in ALPHA_NAMES:
            sp = A_specs[name]
            g = _trend(sp, t) + _drivers(sp, t) + _season_for(name, t)
            if state_gamma and name == TRUTH["state"]["channel"]:
                # Capacity gate: the rate out of S_ref falls as the refined
                # inventory builds.  Smooth, bounded, and invisible to an
                # architecture with `use_stock_input: false`.
                g = g - state_gamma * (S4[1] / S_ref_bar - 1.0)
            alphas.append(sp["A"] * jnp.exp(g))
        alphas = jnp.stack(alphas)

        # binary taus: three pinned, tau_olds learned
        taus = jnp.stack([
            v5._interp_tau(t, data, 0),                     # tau_ref    pinned
            v5._interp_tau(t, data, 1),                     # tau_waelz  pinned
            _logistic(TRUTH["tau_olds"], t),                # tau_olds   learned
            v5._interp_tau(t, data, 3),                     # tau_diss   pinned
        ])

        # manufacturing simplexes, `loss_pinned` branch of v5:741
        fu_loss = jnp.clip(v5._interp_tau(t, data, 5), 0.0, 1.0 - 1e-6)
        s_fu = _logistic(TRUTH["sigma_fu"], t)
        avail_fu = jnp.maximum(1.0 - fu_loss, 1e-6)
        frac_fu = jnp.stack([s_fu * avail_fu, fu_loss, (1.0 - s_fu) * avail_fu])

        eu_loss = jnp.clip(v5._interp_tau(t, data, 7), 0.0, 1.0 - 1e-6)
        s_eu = _logistic(TRUTH["sigma_eu"], t)
        avail_eu = jnp.maximum(1.0 - eu_loss, 1e-6)
        frac_eu = jnp.stack([s_eu * avail_eu, eu_loss, (1.0 - s_eu) * avail_eu])

        d = jnp.tanh((t - coh["tstar"]) / coh["w"]) - c_coh
        logits = jnp.stack([coh["l0"] + coh["d0"] * d,
                            coh["l1"] + coh["d1"] * d,
                            jnp.zeros_like(t)])
        f_cohort = jax.nn.softmax(logits)

        return dict(cp=v5._interp_cp(t, data), alphas=alphas, taus=taus,
                    frac_fu=frac_fu, frac_eu=frac_eu, f_cohort=f_cohort,
                    raw_norm=jnp.asarray(0.0))

    return truth_eval


def truth_coefficients(ctx, t_grid, arm_name="base", *, S_path=None,
                       season=None, state_gamma=None):
    """The analytic truth evaluated on `t_grid`, as NumPy arrays.

    `S_path` (T, 4) is required only for the `state` arm, which reads S_ref.
    Returns a dict with `alphas` (T, 4), `tau_sup` (T, 8), `f_cohort` (T, 3)
    and `cp` (T,).
    """
    import jax
    import jax.numpy as jnp

    ev = make_truth_nn_eval(ctx, arm_name, season=season, state_gamma=state_gamma)
    data = rhs_data_for(ctx)
    t_grid = np.asarray(t_grid, float).ravel()
    if S_path is None:
        S_path = np.zeros((t_grid.size, 4))
    S_path = np.asarray(S_path, float)

    @jax.jit
    def one(t, S):
        o = ev(None, t, S, None, data)
        tau_sup = jnp.concatenate([o["taus"],
                                   jnp.array([o["frac_fu"][0], o["frac_fu"][1],
                                              o["frac_eu"][0], o["frac_eu"][1]])])
        return o["alphas"], tau_sup, o["f_cohort"], o["cp"]

    A, Tsup, FC, CP = jax.vmap(one)(jnp.asarray(t_grid), jnp.asarray(S_path))
    return dict(alphas=np.asarray(A), tau_sup=np.asarray(Tsup),
                f_cohort=np.asarray(FC), cp=np.asarray(CP))


def rhs_data_for(ctx):
    """The minimal `data` dict the truth RHS reads: the real annual grid and
    the real pinned series.  Mirrors `v5._make_rhs_data`'s key set."""
    import jax.numpy as jnp
    return dict(years=jnp.asarray(ctx["years"]),
                cp_obs=jnp.asarray(ctx["cp_obs"]),
                tau_sup_obs=jnp.asarray(ctx["tau_sup_obs"]),
                exog_times=jnp.asarray(ctx["years"]),
                exog_values=jnp.asarray(ctx["exog_values"]),
                stats={})


# ---------------------------------------------------------------------------
# dense solve
# ---------------------------------------------------------------------------
def fine_grid(years, n_sub=N_SUB):
    """Uniform grid with `n_sub` nodes per year, endpoints included."""
    years = np.asarray(years, float).ravel()
    n = int(round((years[-1] - years[0]) * n_sub))
    return years[0] + np.arange(n + 1, dtype=float) / float(n_sub)


def dense_solve(ctx, arm_name="base", *, n_sub=N_SUB, S0=None,
                season=None, state_gamma=None,
                rtol=TIGHT_RTOL, atol=TIGHT_ATOL, max_steps=TIGHT_MAXSTEPS):
    """Integrate the real `zinc_colloc_v5` RHS under the analytic truth.

    Composed from the core's public pieces exactly as `zinc_A_lab.rhs_dSdt`
    and `zinc_fisher_lab.make_tight_integrator` do — `make_integrator` hard-
    codes `rtol=1e-5, atol=1e-7` (v5:1351) and the core is read-only, so the
    tolerance a *truth* generator needs is reached by re-composing rather than
    by editing.  Returns the fine grid, point-in-time stocks, and the
    **cumulative** flow accumulators, from which every observation window is
    an exact difference.
    """
    import jax.numpy as jnp
    import diffrax as dfx
    import zinc_colloc_v5 as v5

    ev = make_truth_nn_eval(ctx, arm_name, season=season, state_gamma=state_gamma)
    rhs = v5.make_rhs(ev)
    data = rhs_data_for(ctx)
    ts = fine_grid(ctx["years"], n_sub)
    if S0 is None:
        S0 = ctx["stocks_obs"][0]

    def f(t, y, args):
        return rhs(y, t, None, args)

    sol = dfx.diffeqsolve(
        dfx.ODETerm(f), dfx.Tsit5(),
        t0=float(ts[0]), t1=float(ts[-1]), dt0=1.0 / (4.0 * n_sub),
        y0=v5._make_Y0(jnp.asarray(S0, float)), args=data,
        saveat=dfx.SaveAt(ts=jnp.asarray(ts)),
        stepsize_controller=dfx.PIDController(rtol=rtol, atol=atol),
        max_steps=max_steps)

    Y = np.asarray(sol.ys)
    S_ode = Y[:, :v5.N_ODE_STOCKS]
    S4 = np.asarray(v5.ode_to_4obs(jnp.asarray(S_ode)))
    C = Y[:, v5.N_ODE_STOCKS:]                      # cumulative flow integrals
    return dict(t=ts, S_ode=S_ode, S4=S4, C=C,
                S_cohorts=S_ode[:, 2:2 + v5.N_COHORTS],
                arm=arm_name, n_sub=int(n_sub))


# ---------------------------------------------------------------------------
# observation operators
# ---------------------------------------------------------------------------
def _window_indices(dense, delta):
    """Fine-grid indices of the observation nodes for window width `delta`."""
    n_sub = dense["n_sub"]
    step = delta * n_sub
    if abs(step - round(step)) > 1e-9:
        raise ValueError(
            f"delta={delta} is not representable on an n_sub={n_sub} grid "
            f"(delta*n_sub = {step}).  Raise n_sub or pick another width.")
    step = int(round(step))
    n_nodes = (dense["t"].size - 1) // step
    return np.arange(n_nodes + 1, dtype=int) * step


# Flow-integral index triples that reconstruct each tau_sup slot as an
# exposure-weighted window ratio: (numerator, denominator terms +, terms -).
# Indices are into `v5.FLOW_NAMES`.
_TAU_RATIO = {
    0: ([7], [1], []),                 # tau_ref     = refinery_losses / cc
    1: ([8], [3], []),                 # tau_waelz   = waelz_recycling / w_in
    2: ([17], [5], []),                # tau_olds    = old_scrap_recovery / eol
    3: ([11], [10], []),               # tau_diss    = dissipative / into_use
    4: ([13], [2, 4], []),             # frac_fu_new = fu_new_scrap / (refc+dr)
    5: ([14], [2, 4], []),             # frac_fu_loss
    6: ([15], [2, 4], [13, 14]),       # frac_eu_new = eu_new / fu_out
    7: ([16], [2, 4], [13, 14]),       # frac_eu_loss
}


def sample(dense, delta, ctx, *, noise_scale=0.0, rng_seed=0, sigma=None,
           pinned_tau=(0, 1, 3, 5, 7)):
    """Apply the observation operators at window width `delta`.

    stocks  point-in-time at the node       flows  period integral over the window
    cp      window-mean **rate**            tau    exposure-weighted flow ratio

    See the module docstring on the cp rate/integral asymmetry.  Noise is
    lognormal and multiplicative, applied to sampled stocks and to flow
    integrals; the learned tau slots and the alpha targets inherit it through
    the ratios they are built from, as they do in a real MFA compilation.

    `pinned_tau` and cp are **model inputs, not estimands** — `anchor_v4` pins
    slots (tau_ref, tau_waelz, tau_diss, frac_fu_loss, frac_eu_loss) and reads
    them straight out of `data`.  Perturbing them would make the refit's
    forcing differ from the truth's, which is a misspecification experiment
    (WP-4b) and not observation noise, so they are taken from the *noiseless*
    ratios.  At delta = 1 those reproduce the real dataset's annual values
    exactly, because `_interp_tau` makes the factorisation
    `int tau F dt = tau_obs int F dt` exact for a piecewise-constant tau.
    """
    import zinc_colloc_v5 as v5

    idx = _window_indices(dense, delta)
    t_nodes = dense["t"][idx]
    T = t_nodes.size

    S_clean = dense["S4"][idx]                                       # (T, 4)
    F_clean = dense["C"][idx[1:]] - dense["C"][idx[:-1]]             # (T-1, NF)
    S_pt, F_int = S_clean.copy(), np.maximum(F_clean, 0.0).copy()

    if noise_scale and noise_scale > 0.0:
        if sigma is None:
            sigma = calibrate_noise()
        rng = np.random.default_rng(int(rng_seed))
        sS = np.asarray(sigma["stock"], float) * float(noise_scale)
        S_pt = S_pt * np.exp(rng.normal(0.0, 1.0, S_pt.shape) * sS[None, :])
        sF = np.zeros(v5.N_FLOWS)
        for name, s in zip(sigma["flow_names"], sigma["flow"]):
            sF[v5.FLOW_NAMES.index(name)] = s * float(noise_scale)
        F_int = F_int * np.exp(rng.normal(0.0, 1.0, F_int.shape) * sF[None, :])

    def _tau_from(F):
        out = np.full((T, v5.N_TAU_SUP), np.nan)
        for j, (num, pos, neg) in _TAU_RATIO.items():
            n = F[:, num].sum(axis=1)
            d = F[:, pos].sum(axis=1) - (F[:, neg].sum(axis=1) if neg else 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                out[1:, j] = np.where(d > 1e-12, n / np.maximum(d, 1e-12), np.nan)
        out[0] = out[1]
        return out

    tau = _tau_from(F_int)
    tau_clean = _tau_from(np.maximum(F_clean, 0.0))
    for j in pinned_tau:
        tau[:, j] = tau_clean[:, j]

    # cp as a window-mean rate: the RHS reads it as the value of cp(t), not as
    # an integral, so `int cp dt = cp_obs * delta` only holds in rate units.
    # cp is a pinned input, so it is taken noiseless for the same reason the
    # pinned taus are.
    cp_rate = np.empty(T)
    cp_rate[1:] = np.maximum(F_clean[:, 0], 0.0) / delta
    cp_rate[0] = cp_rate[1]          # never read by `_interval_pick` (clip >= 1)

    # observable flow table in `load_zinc_data`'s column order and convention:
    # row i is the interval (t_{i-1}, t_i], row 0 unused (only `[1:]` is read).
    pred_idx = np.asarray(ctx["flow_obs_to_pred_idx"], int)
    F_full = np.zeros((T, pred_idx.size))
    F_full[1:] = F_int[:, pred_idx]
    F_full[1:, list(pred_idx).index(0)] = np.maximum(F_clean[:, 0], 0.0)

    F_for_alpha = np.zeros((T, 4))
    F_for_alpha[1:] = F_int[:, [1, 2, 3, 4]]     # cc, refc, w_in, dr
    alpha_obs = v5._build_empirical_alphas(t_nodes, S_pt, F_for_alpha)

    return dict(years=t_nodes, idx=idx, delta=float(delta),
                stocks_obs=S_pt, flows_obs=np.clip(F_full[1:], 0.0, np.inf),
                cp_obs=cp_rate, tau_sup_obs=tau, alpha_obs=alpha_obs,
                F_int=F_int, F_int_clean=np.maximum(F_clean, 0.0),
                stocks_clean=S_clean)


# ---------------------------------------------------------------------------
# noise calibration
# ---------------------------------------------------------------------------
def calibrate_noise(anchor_dir=ANCHOR_DIR, train_end=2007.0):
    """Per-series lognormal sigma from the `anchor_v4` Stage B residuals.

    Median across the 35 seeds of the SD of `log(pred) - log(obs)` on the
    trainval window.  These residuals contain model error as well as
    observation error, so using them as an observation-noise scale is
    conservative — the twin is noisier than the real data can be shown to be,
    not quieter.  `concentrate_production` is forced to zero: cp is pinned, so
    its "residual" (2e-4) measures solver noise, and cp is an input to the
    twin rather than an observation of it.
    """
    S_sd, F_sd = [], []
    paths = sorted(glob.glob(os.path.join(anchor_dir, "pred_seed*.npz")))
    if not paths:
        raise FileNotFoundError(f"no pred_seed*.npz under {anchor_dir}")
    for p in paths:
        d = np.load(p, allow_pickle=True)
        yr, yf = d["years_all"], d["years_flow"]
        So, Sp = d["stocks_obs"], d["S_pred_B"]
        r = np.log(np.maximum(Sp, 1e-9)) - np.log(np.maximum(So, 1e-9))
        S_sd.append(np.nanstd(r[yr <= train_end], axis=0))
        Fo, Fp = d["flows_obs"], d["F_pred_B"]
        m = (Fo > 1e-6) & (Fp > 1e-9)
        rf = np.where(m, np.log(np.maximum(Fp, 1e-12)) - np.log(np.maximum(Fo, 1e-12)),
                      np.nan)
        F_sd.append(np.nanstd(rf[yf <= train_end], axis=0))
    flow_names = [str(x) for x in np.load(paths[0], allow_pickle=True)["flow_names"]]
    fs = np.median(np.asarray(F_sd), axis=0)
    fs[flow_names.index("concentrate_production")] = 0.0
    return dict(stock=np.median(np.asarray(S_sd), axis=0),
                flow=fs, flow_names=flow_names, n_seeds=len(paths))


# ---------------------------------------------------------------------------
# dataset assembly
# ---------------------------------------------------------------------------
def _exog_on(ctx, years, delta):
    """Drivers on the sample grid, plus a pre-t0 history at the same spacing.

    `preprocess_exog` builds its difference features on whatever grid it is
    handed, so at `delta != 1` the order-1 feature is a per-*window* change
    rather than a per-year change.  That is the honest translation — a
    modeller with quarterly data differences quarters — but it means the
    feature block is not comparable across frequencies without rescaling, and
    any cross-frequency claim has to say which it used.  At `delta == 1` the
    real full history is passed through unchanged so the feature block is
    bit-identical to the real pipeline's.
    """
    Xt, Xv = ctx["exog_times_full"], ctx["exog_values_full"]
    cols = np.column_stack([np.interp(years, ctx["years"], ctx["exog_values"][:, j])
                            for j in range(ctx["exog_values"].shape[1])])
    if abs(delta - 1.0) < 1e-12:
        return cols, Xt.copy(), Xv.copy()
    n_back = int(np.floor((years[0] - Xt[0]) / delta))
    t_full = np.concatenate([years[0] - np.arange(n_back, 0, -1) * delta, years])
    v_full = np.column_stack([np.interp(t_full, Xt, Xv[:, j])
                              for j in range(Xv.shape[1])])
    return cols, t_full, v_full


def build_dataset(ctx, samp, delta):
    """A `load_zinc_data`-shaped dict from a sampled twin, ready for `arm()`."""
    years = samp["years"]
    exog, t_full, v_full = _exog_on(ctx, years, delta)
    return dict(
        years=years.copy(),
        stocks_obs=samp["stocks_obs"].copy(),
        cp_obs=samp["cp_obs"].copy(),
        alpha_obs=samp["alpha_obs"].copy(),
        tau_sup_obs=samp["tau_sup_obs"].copy(),
        flows_obs=samp["flows_obs"].copy(),
        flow_obs_names=list(ctx["flow_obs_names"]),
        flow_obs_to_pred_idx=np.asarray(ctx["flow_obs_to_pred_idx"], np.int32),
        exog_times=years.copy(),
        exog_values=exog,
        exog_times_full=t_full,
        exog_values_full=v_full,
        exog_cols=list(ctx["exog_cols"]),
    )


# ---------------------------------------------------------------------------
# the alpha-target bias decomposition  (WP-3's named extra deliverable)
# ---------------------------------------------------------------------------
def _simpson(y, h):
    """Composite Simpson over the last axis; `y` must have an odd node count."""
    n = y.shape[-1] - 1
    if n % 2:
        raise ValueError("Simpson needs an even number of intervals")
    w = np.ones(n + 1)
    w[1:-1:2], w[2:-1:2] = 4.0, 2.0
    return (h / 3.0) * np.tensordot(y, w, axes=([-1], [0]))


def alpha_target_bias(ctx, dense, delta, *, arm_name="base", season=None,
                      state_gamma=None):
    """Decompose `alpha_obs`'s error against the analytic truth, per window.

        alpha_obs = F_int / trapz(S)
                  = [ int a S dt / int S dt ]  x  [ int S dt / trapz(S) ]
                    \\___ exposure-weighted __/     \\___ quadrature ____/

    and the quantity a modeller *reads* `alpha_obs` as is the unweighted
    window mean `(1/Delta) int a dt`.  The gap between the exposure-weighted
    and the unweighted mean is exactly `Cov_w(a, S) / mean_w(S)` over the
    window — the within-window covariance term of WP-1a's secondary
    hypothesis.  All three are returned, per channel and per window, with the
    within-window correlation of F against its parent stock.
    """
    import zinc_colloc_v5 as v5

    idx = _window_indices(dense, delta)
    step = int(idx[1] - idx[0])
    if step % 2:
        raise ValueError(f"delta={delta} gives an odd sub-step count ({step}); "
                         f"Simpson needs an even one")
    h = 1.0 / dense["n_sub"]
    t = dense["t"]

    tc = truth_coefficients(ctx, t, arm_name, S_path=dense["S4"],
                            season=season, state_gamma=state_gamma)
    a_fine = tc["alphas"]                                   # (Nf, 4)
    parent = np.stack([dense["S4"][:, k] for k in v5.ALPHA_PARENT_STOCK_IDX], axis=1)
    F_fine = a_fine * np.maximum(parent, 0.0)

    nW = idx.size - 1
    out = dict(a_unweighted=np.zeros((nW, 4)), a_weighted=np.zeros((nW, 4)),
               a_obs=np.zeros((nW, 4)), int_S=np.zeros((nW, 4)),
               trapz_S=np.zeros((nW, 4)), corr_FS=np.zeros((nW, 4)),
               cov_term=np.zeros((nW, 4)))
    for i in range(nW):
        sl = slice(idx[i], idx[i + 1] + 1)
        aw, Sw, Fw = a_fine[sl], parent[sl], F_fine[sl]
        iS = _simpson(Sw.T, h)
        iF = _simpson(Fw.T, h)
        ia = _simpson(aw.T, h)
        out["int_S"][i] = iS
        out["a_unweighted"][i] = ia / delta
        out["a_weighted"][i] = iF / np.maximum(iS, 1e-12)
        # trapezoid exposure exactly as `_build_empirical_alphas` forms it
        out["trapz_S"][i] = 0.5 * (Sw[0] + Sw[-1]) * delta
        out["a_obs"][i] = iF / np.maximum(out["trapz_S"][i], 1e-12)
        for k in range(4):
            sd = Sw[:, k].std(); fd = Fw[:, k].std()
            out["corr_FS"][i, k] = (np.corrcoef(Fw[:, k], Sw[:, k])[0, 1]
                                    if sd > 1e-12 and fd > 1e-12 else np.nan)
            # Cov_w(a, S) / mean_w(S) with the same Simpson weights
            ma = ia[k] / delta
            mS = iS[k] / delta
            cov = _simpson(((aw[:, k] - ma) * (Sw[:, k] - mS))[None, :], h)[0] / delta
            out["cov_term"][i, k] = cov / max(mS, 1e-12)
    out["years"] = dense["t"][idx[1:]]
    out["delta"] = float(delta)
    return out


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------
def verify(ctx, dense, samples, *, n_sub_ref=192):
    """Checks that must pass before any twin is used for a result.

    1. mass conservation on the twin, per stock, from the accumulators
    2. cross-frequency consistency: sub-annual integrals aggregate to annual
    3. the cp identity `int cp dt = cp_obs * Delta` in rate units
    4. pinned-tau recovery at Delta = 1 against the real dataset's own values
    5. solver convergence: a second dense solve at twice the grid density
    """
    import zinc_colloc_v5 as v5

    res = {}
    C, S4 = dense["C"], dense["S4"]
    tot = C[-1] - C[0]
    F = {n: tot[i] for i, n in enumerate(v5.FLOW_NAMES)}
    dS = S4[-1] - S4[0]
    bal = np.array([
        F["concentrate_production"] - F["concentrate_consumption"],
        F["primary_refining"] + F["waelz_recycling"] - F["refined_consumption"],
        F["inuse_inflow"] - F["end_of_life"],
        F["old_scrap_recovery"] + F["first_use_new_scrap"] + F["end_use_new_scrap"]
        - F["waelz_input"] - F["direct_reuse_recycling"]])
    scale = np.maximum(np.abs(dS), 1.0)
    res["mass_conservation_rel"] = float(np.max(np.abs(bal - dS) / scale))

    if 1.0 in samples and 0.25 in samples:
        a, q = samples[1.0], samples[0.25]
        agg = q["F_int_clean"].reshape(-1, 4, q["F_int_clean"].shape[1]).sum(axis=1)
        d = np.abs(agg - a["F_int_clean"]) / np.maximum(np.abs(a["F_int_clean"]), 1.0)
        res["quarterly_to_annual_rel"] = float(np.max(d))

    cp_err = []
    for dl, s in samples.items():
        lhs = s["F_int_clean"][:, 0]
        cp_err.append(np.max(np.abs(lhs - s["cp_obs"][1:] * dl)
                             / np.maximum(np.abs(lhs), 1.0)))
    res["cp_rate_identity_rel"] = float(np.max(cp_err))

    if 1.0 in samples:
        got = samples[1.0]["tau_sup_obs"]
        want = ctx["tau_sup_obs"]
        n = min(got.shape[0], want.shape[0])
        res["pinned_tau_vs_real_abs"] = float(np.nanmax(
            np.abs(got[1:n, [0, 1, 3, 5, 7]] - want[1:n, [0, 1, 3, 5, 7]])))

    ref = dense_solve(ctx, dense["arm"], n_sub=n_sub_ref)
    i2 = np.searchsorted(ref["t"], dense["t"])
    res["solver_stock_rel"] = float(np.max(
        np.abs(ref["S4"][i2] - S4) / np.maximum(np.abs(S4), 1.0)))
    res["solver_flow_rel"] = float(np.max(
        np.abs((ref["C"][i2] - C)) / np.maximum(np.abs(C), 1.0)))
    return res


def realism(ctx, dense):
    """How far the twin's trajectory sits from the real one, per stock.

    The twin is not meant to reproduce the data — it is meant to be a
    plausible zinc cycle whose coefficients are known.  This is the number
    that says whether the analytic constants in `TRUTH` produced something
    physical, and it is reported rather than tuned against.
    """
    idx = _window_indices(dense, 1.0)
    S = dense["S4"][idx]
    R = ctx["stocks_obs"]
    n = min(S.shape[0], R.shape[0])
    rel = 100.0 * np.sqrt(np.mean(((S[:n] - R[:n]) / np.maximum(np.abs(R[:n]), 1e-9)) ** 2,
                                  axis=0))
    return dict(rel_rmse_pct=rel, min_stock=S.min(axis=0),
                ratio_range=np.stack([(S[:n] / np.maximum(R[:n], 1e-9)).min(axis=0),
                                      (S[:n] / np.maximum(R[:n], 1e-9)).max(axis=0)]))


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------
def save_dataset(path, ds, *, truth=None, meta=None):
    arrays = {k: (np.asarray(v, dtype=object) if k in ("flow_obs_names", "exog_cols")
                  else np.asarray(v))
              for k, v in ds.items()}
    if truth:
        arrays.update({f"truth_{k}": np.asarray(v) for k, v in truth.items()})
    arrays["meta_json"] = np.asarray(json.dumps(meta or {}), dtype=object)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def load_dataset(path):
    d = np.load(path, allow_pickle=True)
    keys = ("years", "stocks_obs", "cp_obs", "alpha_obs", "tau_sup_obs",
            "flows_obs", "exog_times", "exog_values", "exog_times_full",
            "exog_values_full")
    out = {k: np.asarray(d[k], float) for k in keys}
    out["flow_obs_names"] = [str(x) for x in d["flow_obs_names"]]
    out["exog_cols"] = [str(x) for x in d["exog_cols"]]
    out["flow_obs_to_pred_idx"] = np.asarray(d["flow_obs_to_pred_idx"], np.int32)
    return out


def load_truth(path):
    d = np.load(path, allow_pickle=True)
    out = {k[len("truth_"):]: np.asarray(d[k]) for k in d.files
           if k.startswith("truth_")}
    out["meta"] = json.loads(str(d["meta_json"]))
    return out


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
def generate(arms=ARMS, deltas=DELTAS_DEFAULT, *, out_dir=OUT_DIR_DEFAULT,
             noise_scale=1.0, rng_seed=0, cfg=None, verbose=True,
             also_noiseless=True):
    """Generate every (arm, delta) dataset with its ground truth alongside."""
    os.makedirs(out_dir, exist_ok=True)
    ctx = driver_context(cfg)
    sigma = calibrate_noise()
    report = {"arms": {}, "sigma": {"stock": sigma["stock"].tolist(),
                                    "flow": sigma["flow"].tolist(),
                                    "flow_names": sigma["flow_names"],
                                    "n_seeds": sigma["n_seeds"]},
              "n_sub": N_SUB, "deltas": list(deltas), "truth": _json_truth()}

    for arm_name in arms:
        t0 = time.time()
        dense = dense_solve(ctx, arm_name)
        samples = {float(dl): sample(dense, dl, ctx, noise_scale=0.0)
                   for dl in deltas}
        v = verify(ctx, dense, samples)
        r = realism(ctx, dense)
        if verbose:
            print(f"[{arm_name}] dense solve {time.time()-t0:5.1f}s  "
                  f"mass {v['mass_conservation_rel']:.2e}  "
                  f"solver S {v['solver_stock_rel']:.2e}  "
                  f"stock relRMSE vs real {np.round(r['rel_rmse_pct'],1)}")

        # continuous truth on the fine grid, saved once per arm
        tc = truth_coefficients(ctx, dense["t"], arm_name, S_path=dense["S4"])
        np.savez_compressed(
            os.path.join(out_dir, f"{arm_name}_dense.npz"),
            t=dense["t"], S4=dense["S4"], S_cohorts=dense["S_cohorts"],
            C=dense["C"], alphas=tc["alphas"], tau_sup=tc["tau_sup"],
            f_cohort=tc["f_cohort"], cp=tc["cp"],
            alpha_names=np.asarray(ALPHA_NAMES, dtype=object),
            tau_sup_names=np.asarray(TAU_SUP_NAMES, dtype=object),
            meta_json=np.asarray(json.dumps(
                dict(arm=arm_name, n_sub=N_SUB, verify=v,
                     realism={k: np.asarray(x).tolist() for k, x in r.items()})),
                dtype=object))

        arm_rep = {"verify": v,
                   "realism": {k: np.asarray(x).tolist() for k, x in r.items()},
                   "datasets": {}}
        for dl in deltas:
            for noisy in ((True, False) if (noise_scale > 0 and also_noiseless)
                          else (noise_scale > 0,)):
                ns = noise_scale if noisy else 0.0
                tag = f"{arm_name}_d{_dtag(dl)}_{'noisy' if noisy else 'clean'}"
                s = (samples[float(dl)] if not noisy else
                     sample(dense, dl, ctx, noise_scale=ns, rng_seed=rng_seed,
                            sigma=sigma))
                ds = build_dataset(ctx, s, dl)
                tw = truth_coefficients(ctx, s["years"], arm_name,
                                        S_path=s["stocks_clean"])
                bias = alpha_target_bias(ctx, dense, dl, arm_name=arm_name)
                truth = dict(
                    alphas_point=tw["alphas"], tau_sup_point=tw["tau_sup"],
                    f_cohort_point=tw["f_cohort"],
                    alphas_window_unweighted=bias["a_unweighted"],
                    alphas_window_weighted=bias["a_weighted"],
                    alphas_window_obs=bias["a_obs"],
                    corr_FS=bias["corr_FS"], cov_term=bias["cov_term"],
                    int_S=bias["int_S"], trapz_S=bias["trapz_S"],
                    stocks_clean=s["stocks_clean"], F_int_clean=s["F_int_clean"],
                    window_years=bias["years"],
                    alpha_names=np.asarray(ALPHA_NAMES, dtype=object),
                    tau_sup_names=np.asarray(TAU_SUP_NAMES, dtype=object))
                p = save_dataset(
                    os.path.join(out_dir, f"{tag}.npz"), ds, truth=truth,
                    meta=dict(arm=arm_name, delta=float(dl), noise_scale=float(ns),
                              rng_seed=int(rng_seed), n_sub=N_SUB, T=int(ds["years"].size),
                              truth_spec=_json_truth()))
                arm_rep["datasets"][tag] = dict(path=os.path.basename(p),
                                                delta=float(dl), noise=float(ns),
                                                T=int(ds["years"].size))
        report["arms"][arm_name] = arm_rep
        if verbose:
            print(f"[{arm_name}] wrote {len(arm_rep['datasets'])} datasets  "
                  f"rss {_rss_mb():.0f} MB")
        import jax
        jax.clear_caches()

    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    return report


def _dtag(dl):
    return (f"{int(round(dl))}y" if abs(dl - round(dl)) < 1e-9
            else f"{dl:.3f}".rstrip("0").rstrip(".").replace(".", "p") + "y")


def _json_truth():
    return json.loads(json.dumps(TRUTH, default=lambda o: (
        list(o) if isinstance(o, (tuple, np.ndarray)) else float(o))))


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------
def check(verbose=True):
    """Resolve drivers and `input_dim`, and prove the patch is inert when
    disarmed, before any work is done."""
    import zinc_colloc_v5 as v5

    info = alab.check(verbose=False)
    install(v5)
    cfg = load_anchor_config()

    # the dispatcher must be the identity when no twin is armed
    disarm()
    a = _ORIG_LOADER(cfg["xlsx_path"], extra_exog_cols=cfg.get("extra_exog_cols"))
    b = v5.load_zinc_data(cfg["xlsx_path"], extra_exog_cols=cfg.get("extra_exog_cols"))
    dmax = 0.0
    for k, va in a.items():
        vb = b[k]
        if isinstance(va, np.ndarray) and va.dtype.kind in "fiu":
            dmax = max(dmax, float(np.nanmax(np.abs(np.asarray(vb, float)
                                                    - np.asarray(va, float)))))
        else:
            assert list(va) == list(vb), k
    info["dispatcher_identity_max_abs"] = dmax

    ctx = driver_context(cfg)
    info["truth_drivers"] = {k: DRIVER_ALIASES[k] for k in DRIVER_ALIASES}
    info["arms"] = list(ARMS)
    info["n_sub"] = N_SUB
    info["deltas"] = list(DELTAS_DEFAULT)
    sig = calibrate_noise()
    info["sigma_stock"] = sig["stock"].tolist()

    if verbose:
        print("=" * 74)
        print("zinc_synth_lab --check")
        print("=" * 74)
        for label, (got, ok) in info["digests"].items():
            print(f"  {label:20s} md5 {got}  {'OK' if ok else 'MISMATCH'}")
        print(f"  patches applied      : {PATCHES}")
        print(f"  dispatcher identity  : max |delta| = {dmax:.3e}  "
              f"({'inert when disarmed' if dmax == 0.0 else 'NOT INERT'})")
        print(f"\n  resolved drivers (canonical order, {info['n_universe']}):")
        for i, c in enumerate(info["exog_cols"]):
            mark = ""
            for short, name in DRIVER_ALIASES.items():
                if name == c:
                    mark = f"   <- truth uses this as z[{short}]"
            print(f"    [{i:2d}] {c}{mark}")
        print(f"\n  exog_feature_orders  : {info['orders']}")
        print(f"  input_dim            : 1 (t) + 4 (S) + "
              f"{info['n_exog_features']} = {info['input_dim']}")
        if info["input_dim"] != 23:
            print(f"  !! input_dim is {info['input_dim']}, not the 23 asserted by "
                  f"CLAUDE.md rule 2 (SCHEMA §9 flag 1, unresolved).")
        print(f"\n  arms                 : {list(ARMS)}")
        print(f"  window widths (yr)   : {list(DELTAS_DEFAULT)}")
        print(f"  dense grid           : {N_SUB} nodes/yr, "
              f"{fine_grid(ctx['years']).size} nodes total")
        print(f"  turning points       : " + ", ".join(
            f"{k}@{v['tstar']:.0f}" for k, v in TRUTH["alpha"].items()))
        print(f"  sub-annual (season)  : " + ", ".join(
            f"{f:g}/yr a={a:g}" for (_, f, a, _) in TRUTH["season"]))
        print(f"  noise sigma (stock)  : "
              f"{np.round(sig['stock'], 4).tolist()}  from {sig['n_seeds']} seeds")
        print("=" * 74)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description="WP-3 synthetic twin generator")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--deltas", default=",".join(str(d) for d in DELTAS_DEFAULT))
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--out", default=OUT_DIR_DEFAULT)
    args = ap.parse_args(argv)

    if args.check or not args.generate:
        check()
        if not args.generate:
            return 0
    import zinc_colloc_v5 as v5
    integrity_check()
    install(v5)
    generate([a.strip() for a in args.arms.split(",") if a.strip()],
             [float(d) for d in args.deltas.split(",") if d.strip()],
             out_dir=args.out, noise_scale=args.noise, rng_seed=args.rng_seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
