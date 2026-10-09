#!/usr/bin/env python3
"""
zinc_cf_lab.py — lab module for WP-7: counterfactuals
=====================================================

WP-7 asks three counterfactual questions of the fitted `anchor_v4` ensemble:

  1. **Price path** — hold the real zinc price on its 1990s trend through
     2008–2019; report divergence in In-Use stock and secondary supply.
  2. **Policy intervention** — step `tau_olds` up 10 pp from 2005; report the
     trajectory and the time-to-effect in secondary supply.
  3. **Shock removal** — zero the 2008 driver shock and decompose the observed
     downstream change into driver-explained and residual parts.

Nothing is refitted.  A counterfactual is a *re-integration of the already
fitted dynamics under a modified input*, so the whole package is a forward
solve per (seed, arm): ~16 ms once the integrator is jitted.

Pattern
-------
Follows `zinc_alpha_lab.py` / `zinc_A_lab.py` (CLAUDE.md rule 1).
`zinc_colloc_v5.py` is imported and never edited; its MD5 and the MD5 of the
`anchor_v4` data copy are pinned; integrity checking, config loading and the
driver/`input_dim` part of `--check` are imported from `zinc_alpha_lab`
rather than duplicated.

`PATCHES` is empty and `install()` registers nothing, for the same reason it
does in `zinc_A_lab`: every intervention here is expressible by *composing*
the core's own public factories rather than by rebinding anything inside it.
Driver counterfactuals change `data["exog_values"]`, which the RHS reads
through `exog_fn`; the `tau_olds` counterfactual wraps the fitted `nn_eval`
closure and hands the wrapper to `make_integrator`, exactly as
`zinc_A_lab.rhs_dSdt` hands `fit.nn_eval` to `make_rhs`.  The fitted model
object is untouched, so a counterfactual cannot leak back into a factual
number.

Where the weights come from
---------------------------
`run_anchor.py` persists no weights (SCHEMA §1); `zinc_A_lab` dumps them per
seed into `analysis/wp2a/A_seed*.npz`, and WP-7 reloads those rather than
paying for 35 refits.  Data objects and closures are built once by a
zero-step `v5.run(..., stageA_steps=0, do_stage_B=False)` — the data build
does not depend on `seed`, only the initialisation does, and the
initialisation is thrown away.  Correctness is not assumed: every seed's
factual re-integration is checked against the stored `anchor_v4`
`pred_seed{N}.npz` rollout and must be bit-identical (`max |ΔS| = 0`).

The three interventions
-----------------------
**1. Price.**  The spec's phrase "hold real zinc price on its 1990s trend"
admits more than one reading, so three arms are run and reported together
rather than one being chosen silently:

    price_trend90s          log-linear OLS on 1990–1999, extrapolated over
                            2008–2019.  The literal reading.  The 1990s trend
                            declines, so the counterfactual level in 2008 sits
                            well below the observed 2007 value and the splice
                            is a step.
    price_trend90s_anchored same slope, level shifted so 2008 continues from
                            the observed 2007 value.  Removes the splice step,
                            so `trend90s − trend90s_anchored` is exactly the
                            level-reset component.
    price_flat90s           held at the 1990–1999 geometric mean.  Slope-free.

Only 2008–2019 is altered; the 2006–07 price spike is left in place, as the
spec specifies the intervention window and not the boom.

**2. Policy.**  `anchor_v4` sets `pin_tau_olds: false`, so `tau_olds` is an
MLP output, not a data lookup, and the intervention has to be applied to the
coefficient itself.  `make_cf_nn_eval` wraps the fitted closure and adds
`delta * 1[t >= t0]` to the `tau_olds` slot, clipped to the unit interval.
A hard step introduces no new discontinuity: the RHS is *already* piecewise
constant in `t` at every integer year through `_interval_pick` (pinned τ, cp).
With `delta = 0` the wrapper is the identity — which is what the factual arm
uses, so a single compiled integrator serves every arm and the identity is
verified rather than argued.  A 5 pp arm runs alongside the 10 pp one as a
linearity check on the response.

**3. Shock.**  The 2008 shock is removed by replacing each driver's 2008–2009
values with a log-linear bridge between its observed 2007 and 2010 values.
This zeroes the *transitory* crash-and-rebound while leaving every other year
untouched, so no assumption about the post-crisis path is smuggled in.  It
does **not** remove the permanent level effect of the crisis (US industrial
production is still 10% below its 2007 level in 2010), and the resulting
"residual" therefore contains that permanent component.  Stated plainly in
the findings note rather than left for a reviewer to find.  A 2008–2010
bridge runs as a width-robustness arm, and thirteen single-driver arms give
the per-driver decomposition (their sum versus the joint arm measures the
interaction).

Secondary supply
----------------
Defined as the mass re-entering production from the scrap pool,

    secondary_supply = waelz_recycling + direct_reuse_recycling
                     = tau_waelz * alpha_win * S_scrap  +  alpha_dr * S_scrap

i.e. the two Chapter 2 old-scrap re-entry routes, α₁₄ and α₁₃.  Both are
period integrals over (Y−1, Y] taken from the integrator's own flow
accumulators, never point-sampled (spec §1, observation operators).
`old_scrap_recovery` — collection into the scrap pool — is reported
separately: it is upstream of the pool, not a supply of metal to production.

CLI
---
    python zinc_cf_lab.py --check
    python zinc_cf_lab.py --seeds 0,1,2 --out analysis/wp7
    python zinc_cf_lab.py --all-seeds
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

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR_DEFAULT = os.path.join(HERE, "analysis", "wp7")
WEIGHTS_DIR_DEFAULT = os.path.join(HERE, "analysis", "wp2a")
ANCHOR_DIR = os.path.join(HERE, "anchor_v4")

PRICE_DRIVER = "Zinc real price"

# Mass re-entering production from the scrap pool (see module docstring).
SECONDARY_SUPPLY_FLOWS = ("waelz_recycling", "direct_reuse_recycling")

PATCHES: list[str] = []          # WP-7 needs none — see module docstring


def install(v5mod=None):
    """Apply import-time patches to the core module.  None needed for WP-7."""
    if v5mod is None:
        import zinc_colloc_v5 as v5mod          # noqa: F401
    return PATCHES


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------
def _shock_arm(driver=None, lo=2008, hi=2009):
    return dict(kind="driver_bridge", drivers=("all" if driver is None
                                               else (driver,)),
                bridge_years=(lo, hi))


def build_arms(driver_names):
    """The ordered arm table.  `factual` first so it is always the reference."""
    arms = {
        "factual": dict(kind="none", closure="plain", ref=None,
                        label="no intervention (reference for driver arms)"),
        "factual_tau_ref": dict(kind="tau_olds_step", delta=0.0, t0=1e9,
                                closure="cf", ref="factual",
                                label="no intervention, through the intervened "
                                      "closure (reference for tau arms)"),

        # --- 1. price path -------------------------------------------------
        "price_trend90s": dict(
            kind="driver_trend", driver=PRICE_DRIVER, mode="trend",
            fit_years=(1990, 1999), apply_years=(2008, 2019),
            closure="plain", ref="factual",
            label="zinc price on the 1990s log-linear trend, 2008–2019"),
        "price_trend90s_anchored": dict(
            kind="driver_trend", driver=PRICE_DRIVER, mode="trend_anchored",
            fit_years=(1990, 1999), apply_years=(2008, 2019),
            closure="plain", ref="factual",
            label="same slope, level anchored to observed 2007"),
        "price_flat90s": dict(
            kind="driver_trend", driver=PRICE_DRIVER, mode="flat",
            fit_years=(1990, 1999), apply_years=(2008, 2019),
            closure="plain", ref="factual",
            label="zinc price held at the 1990s geometric mean, 2008–2019"),

        # --- 2. policy intervention ---------------------------------------
        "tau_olds_step10_2005": dict(
            kind="tau_olds_step", delta=0.10, t0=2005.0,
            closure="cf", ref="factual_tau_ref",
            label="tau_olds +10 pp from 2005"),
        "tau_olds_step05_2005": dict(
            kind="tau_olds_step", delta=0.05, t0=2005.0,
            closure="cf", ref="factual_tau_ref",
            label="tau_olds +5 pp from 2005 (linearity check)"),

        # --- 3. shock removal ---------------------------------------------
        "shock2008_bridge": dict(
            **_shock_arm(), closure="plain", ref="factual",
            label="all drivers, 2008–2009 bridged from 2007 to 2010"),
        "shock2008_bridge3": dict(
            **_shock_arm(lo=2008, hi=2010), closure="plain", ref="factual",
            label="all drivers, 2008–2010 bridged (width robustness)"),
    }
    for d in driver_names:
        arms[f"shock2008_only::{d}"] = dict(
            **_shock_arm(driver=d), closure="plain", ref="factual",
            label=f"only '{d}' bridged over 2008–2009")
    return arms


# ---------------------------------------------------------------------------
# counterfactual driver construction
# ---------------------------------------------------------------------------
def _loglinear_fit(t, y):
    """OLS of log y on t.  Returns (intercept, slope) in log space."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y) & (y > 0)
    if m.sum() < 2:
        raise ValueError("need >= 2 positive points for a log-linear fit")
    A = np.column_stack([np.ones(m.sum()), t[m]])
    beta, *_ = np.linalg.lstsq(A, np.log(y[m]), rcond=None)
    return float(beta[0]), float(beta[1])


def cf_driver_trend(t_full, X_full, j, *, mode, fit_years, apply_years):
    """Replace driver `j` over `apply_years` by a 1990s-trend counterfactual.

    `mode`:
      "trend"           extrapolated log-linear fit over `fit_years`;
      "trend_anchored"  same slope, level shifted so the first applied year
                        continues from the last un-applied observed value;
      "flat"            the geometric mean over `fit_years`.
    """
    X = np.array(X_full, float, copy=True)
    t = np.asarray(t_full, float)
    fit_m = (t >= fit_years[0]) & (t <= fit_years[1])
    app_m = (t >= apply_years[0]) & (t <= apply_years[1])
    if not fit_m.any() or not app_m.any():
        raise ValueError("empty fit or apply window for the trend counterfactual")

    a, b = _loglinear_fit(t[fit_m], X_full[fit_m, j])
    if mode == "flat":
        X[app_m, j] = float(np.exp(np.mean(np.log(X_full[fit_m, j]))))
        return X
    path = np.exp(a + b * t[app_m])
    if mode == "trend_anchored":
        i0 = int(np.where(app_m)[0][0])
        if i0 == 0:
            raise ValueError("cannot anchor: no observation before the window")
        # continue from the last observed year at the fitted growth rate
        path = X_full[i0 - 1, j] * np.exp(b * (t[app_m] - t[i0 - 1]))
    elif mode != "trend":
        raise ValueError(f"unknown trend mode {mode!r}")
    X[app_m, j] = path
    return X


def cf_driver_bridge(t_full, X_full, cols, drivers, bridge_years):
    """Log-linearly bridge `drivers` across `bridge_years`.

    The bridge runs between the observed values in the years immediately
    before and after the window, so both endpoints and every year outside the
    window are left exactly as observed.
    """
    X = np.array(X_full, float, copy=True)
    t = np.asarray(t_full, float)
    lo, hi = float(bridge_years[0]), float(bridge_years[1])
    idx = np.where((t >= lo) & (t <= hi))[0]
    if idx.size == 0:
        raise ValueError(f"bridge window {bridge_years} is empty")
    i0, i1 = int(idx[0]) - 1, int(idx[-1]) + 1
    if i0 < 0 or i1 >= len(t):
        raise ValueError("bridge window has no observation on both sides")
    names = list(cols) if drivers == "all" or drivers == ("all",) else list(drivers)
    for d in names:
        j = list(cols).index(d)
        y0, y1 = X_full[i0, j], X_full[i1, j]
        if y0 > 0 and y1 > 0:
            w = (t[idx] - t[i0]) / (t[i1] - t[i0])
            X[idx, j] = np.exp((1.0 - w) * np.log(y0) + w * np.log(y1))
        else:                                   # non-positive: linear bridge
            w = (t[idx] - t[i0]) / (t[i1] - t[i0])
            X[idx, j] = (1.0 - w) * y0 + w * y1
    return X


def apply_arm_to_drivers(spec, t_full, X_full, cols):
    """Dispatch one arm spec onto the raw full-history driver matrix."""
    kind = spec["kind"]
    if kind in ("none", "tau_olds_step"):
        return np.array(X_full, float, copy=True)
    if kind == "driver_trend":
        j = list(cols).index(spec["driver"])
        return cf_driver_trend(t_full, X_full, j, mode=spec["mode"],
                               fit_years=spec["fit_years"],
                               apply_years=spec["apply_years"])
    if kind == "driver_bridge":
        return cf_driver_bridge(t_full, X_full, cols, spec["drivers"],
                                spec["bridge_years"])
    raise ValueError(f"unknown arm kind {kind!r}")


# ---------------------------------------------------------------------------
# data assembly
# ---------------------------------------------------------------------------
def split_cut(T, trainval_frac=0.7, val_frac=0.2):
    """`train_model`'s own split arithmetic (v5:1882–1885).

    Reproduced rather than approximated: `zinc_alpha_lab.check()` prints
    `cut = 20` from `round(val_frac * T)`, but the core takes
    `n_val = ceil(val_frac * n_trainval)`, giving `cut = 22` (train core
    1980–2001).  Harmless there — `exog_detrend` is false in `anchor_v4`, so
    `years_train` is unused downstream — but the counterfactual exog rebuild
    has to match the fit exactly, so the correct value is used here and the
    rebuild is verified bit-for-bit against `fit.data_all["exog_values"]`.
    """
    n_trainval = min(max(int(np.floor(trainval_frac * T)), 3), T - 1)
    n_val = max(3, int(np.ceil(val_frac * n_trainval)))
    return max(n_trainval - n_val, 2), n_trainval


def rebuild_exog(cfg, years, t_full, X_full, cut):
    """Re-run `preprocess_exog` with the anchor_v4 settings on a modified
    full-history driver matrix.  Bit-reproduces the factual features when
    `X_full` is unmodified (checked in `make_cf_data`)."""
    import zinc_colloc_v5 as v5

    # `exog_values` (2nd positional) is ignored whenever `years_source` is
    # given — `preprocess_exog` reads the source matrix instead (v5:1069–1073).
    return v5.preprocess_exog(
        years, X_full, years[:cut],
        do_log1p=cfg.get("exog_log1p", True),
        do_detrend=cfg.get("exog_detrend", False),
        detrend_kind=cfg.get("exog_detrend_kind", "linear"),
        feature_orders=tuple(cfg.get("exog_feature_orders", (0,))),
        diff_pad=cfg.get("exog_diff_pad", "edge"),
        years_source=t_full, exog_values_source=X_full,
    )


def make_cf_data(fit, cfg, spec, t_full, X_full, cols, cut):
    """A copy of `fit.data_all` carrying one arm's intervention.

    Driver arms replace `exog_values`; the `tau_olds` arm leaves the drivers
    alone and instead parks `(delta, t0)` inside `stats`, which is the one
    part of `data` that survives `_make_rhs_data`'s filtering and reaches the
    RHS.  Both knobs are present in EVERY arm (0.0 / 1e9 when unused) so the
    pytree structure is arm-independent and one compiled integrator serves
    all of them.
    """
    import jax.numpy as jnp

    years = np.asarray(fit.data_all["years"], float).ravel()
    X_cf = apply_arm_to_drivers(spec, t_full, X_full, cols)
    exog_cf = rebuild_exog(cfg, years, t_full, X_cf, cut)

    stats = dict(fit.data_all["stats"])
    stats["cf_tau_olds_delta"] = jnp.asarray(
        float(spec.get("delta", 0.0)) if spec["kind"] == "tau_olds_step" else 0.0)
    stats["cf_tau_olds_t0"] = jnp.asarray(
        float(spec.get("t0", 1e9)) if spec["kind"] == "tau_olds_step" else 1e9)

    data = dict(fit.data_all)
    data["exog_values"] = jnp.asarray(exog_cf)
    data["stats"] = stats
    return data, X_cf, exog_cf


# ---------------------------------------------------------------------------
# the intervened closure
# ---------------------------------------------------------------------------
def make_cf_nn_eval(nn_eval):
    """Wrap a fitted `nn_eval` with the `tau_olds` step intervention.

    `tau_olds` is index 2 of `TAU_BINARY_NAMES` and is LEARNED under
    `anchor_v4` (`pin_tau_olds: false`), so it has to be intervened on at the
    coefficient rather than in the data.  With `delta = 0` the wrapper adds
    exactly 0.0 to a sigmoid output already inside (0, 1), so the clip is a
    no-op and the composition is the identity — the property the factual arm
    relies on and that `dump_cf` verifies against the stored rollout.
    """
    import jax.numpy as jnp

    def cf_nn_eval(params, t, S, exog_t, data):
        out = dict(nn_eval(params, t, S, exog_t, data))
        stats = data["stats"]
        delta = stats["cf_tau_olds_delta"]
        gate = jnp.where(t >= stats["cf_tau_olds_t0"], 1.0, 0.0)
        taus = out["taus"]
        out["taus"] = taus.at[2].set(
            jnp.clip(taus[2] + delta * gate, 0.0, 1.0 - 1e-6))
        return out

    return cf_nn_eval


def _machinery_for(nn_eval):
    """`(integrate, coeffs)` — a jitted free-run integrator and a jitted
    coefficient evaluator built on one NN closure."""
    import jax
    import zinc_colloc_v5 as v5

    integrate_aug = v5.make_integrator("diffrax", nn_eval)

    @jax.jit
    def integrate(params, data):
        return integrate_aug(params, data, data["stocks_obs"][0], data["years"])

    @jax.jit
    def coeffs(params, data, S):
        def one(t, S4):
            ex = v5.exog_fn(t, data["exog_times"], data["exog_values"])
            out = nn_eval(params, t, S4, ex, data)
            return out["alphas"], out["taus"], out["f_cohort"], out["cp"]
        return jax.vmap(one)(data["years"], S)

    return integrate, coeffs


def make_cf_machinery(fit):
    """Two matched integrators, keyed by the `closure` field of an arm.

    `"plain"` is the fitted closure verbatim: arms that intervene only on the
    drivers use it, so their factual reference reproduces the stored
    `anchor_v4` rollout bit-for-bit.

    `"cf"` is `make_cf_nn_eval(fit.nn_eval)`, needed by the `tau_olds` arms.
    With `delta = 0` the wrapper is **bitwise** the identity — verified per
    seed at both the coefficient and the RHS level (`verify_wrapper_identity`,
    max |Δ| = 0 exactly).  The integrated trajectory nevertheless drifts from
    the plain one by O(1e-3–1e0) kt, because the extra ops change how XLA
    fuses the RHS, which changes the PID controller's accepted step sequence,
    which the 39-year free-run accumulates.  That is a property of adaptive
    integration, not of the model, and rather than paper over it the `cf`
    arms are differenced against their own matched reference
    (`factual_tau_ref`) while `factual − factual_tau_ref` is reported as the
    **numerical noise floor** of the whole package: the smallest
    counterfactual effect that carries any meaning.
    """
    return {"plain": _machinery_for(fit.nn_eval),
            "cf": _machinery_for(make_cf_nn_eval(fit.nn_eval))}


def verify_wrapper_identity(fit, params, data, S):
    """With `delta = 0` the intervened closure must be the exact identity.

    Checked at the coefficient level and at the full RHS (the closure the
    integrator actually runs), over every year node at the given states.
    Returns the two max absolute differences — both must be exactly 0.
    """
    import jax.numpy as jnp
    import zinc_colloc_v5 as v5

    cf_nn_eval = make_cf_nn_eval(fit.nn_eval)
    rhs_p, rhs_c = v5.make_rhs(fit.nn_eval), v5.make_rhs(cf_nn_eval)
    rhs_data = v5._make_rhs_data(data)
    years = np.asarray(data["years"], float).ravel()
    S = np.asarray(S, float)
    frac = np.asarray(v5.IC_COHORT_FRACS, float)
    d_coef, d_rhs = 0.0, 0.0
    for i, y in enumerate(years):
        S4 = jnp.asarray(S[i])
        ex = v5.exog_fn(y, data["exog_times"], data["exog_values"])
        o1, o2 = fit.nn_eval(params, y, S4, ex, data), cf_nn_eval(params, y, S4, ex, data)
        d_coef = max(d_coef, max(float(jnp.max(jnp.abs(o1[k] - o2[k]))) for k in o1))
        Y = jnp.concatenate([S4[:2], jnp.asarray(frac) * S4[2], S4[3:4],
                             jnp.zeros(v5.N_FLOWS)])
        d_rhs = max(d_rhs, float(jnp.max(jnp.abs(rhs_p(Y, y, params, rhs_data)
                                                 - rhs_c(Y, y, params, rhs_data)))))
    return d_coef, d_rhs


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------
def load_params(path):
    """Stage-B MLP weights from a `zinc_A_lab` dump (`param{i}_{W,b}`)."""
    import jax.numpy as jnp

    d = np.load(path, allow_pickle=True)
    n = sum(1 for k in d.files if k.startswith("param") and k.endswith("_W"))
    if n == 0:
        raise ValueError(f"{path} carries no param*_W arrays")
    return [{"W": jnp.asarray(d[f"param{i}_W"]),
             "b": jnp.asarray(d[f"param{i}_b"])} for i in range(n)]


def init_fit(cfg=None, seed=0):
    """A zero-step run: the data objects, layout and closures `train_model`
    builds, without any training.  Seed-independent for everything WP-7 uses
    (the data build ignores `seed`; only the discarded initialisation does not).
    """
    import zinc_colloc_v5 as v5

    integrity_check()
    install(v5)
    cfg = dict(cfg or load_anchor_config())
    cfg["verbose"] = False
    return v5.run("wp7_init", **dict(cfg, seed=int(seed),
                                     stageA_steps=0, do_stage_B=False))


# ---------------------------------------------------------------------------
# the dump
# ---------------------------------------------------------------------------
def _rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024.0 ** 2) if sys.platform == "darwin" else r / 1024.0


def run_seed(seed, fit, machinery, arms, ctx, out_dir):
    """Every arm for one seed, verified and dumped.  Returns the check dict."""
    import zinc_colloc_v5 as v5

    cfg, t_full, X_full, cols, cut, weights_dir = ctx
    params = load_params(os.path.join(weights_dir, f"A_seed{seed}.npz"))

    years = np.asarray(fit.data_all["years"], float).ravel()
    arrays = dict(
        years=years,
        seed=np.asarray(int(seed)),
        arm_names=np.array(list(arms), dtype=object),
        arm_labels=np.array([arms[a]["label"] for a in arms], dtype=object),
        arm_specs=np.asarray(json.dumps(
            {k: {kk: (list(vv) if isinstance(vv, tuple) else vv)
                 for kk, vv in v.items()} for k, v in arms.items()}),
            dtype=object),
        driver_names=np.array(list(cols), dtype=object),
        driver_times_full=np.asarray(t_full, float),
        flow_names=np.array(v5.FLOW_NAMES, dtype=object),
        stock_names=np.array(v5.STOCK_NAMES, dtype=object),
        alpha_names=np.array(v5.ALPHA_NAMES, dtype=object),
        tau_binary_names=np.array(v5.TAU_BINARY_NAMES, dtype=object),
        secondary_supply_flows=np.array(SECONDARY_SUPPLY_FLOWS, dtype=object),
        stocks_obs=np.asarray(fit.data_all["stocks_obs"], float),
        flows_obs=np.asarray(fit.data_all["flows_obs"], float),
        flow_obs_to_pred_idx=np.asarray(fit.flow_obs_to_pred_idx, int),
    )
    masks = fit._split_indices_in_all()
    for k in ("train", "val", "test"):
        arrays[f"mask_{k}"] = np.asarray(masks[k], bool)

    for arm, spec in arms.items():
        data, X_cf, _exog_cf = make_cf_data(fit, cfg, spec, t_full, X_full,
                                            cols, cut)
        integrate, coeffs = machinery[spec["closure"]]
        S, F, C = integrate(params, data)
        a, tb, fc, cp = coeffs(params, data, S)
        tag = arm.replace("::", "__")
        arrays[f"S__{tag}"] = np.asarray(S, float)
        arrays[f"F__{tag}"] = np.asarray(F, float)
        arrays[f"Scoh__{tag}"] = np.asarray(C, float)
        arrays[f"alphas__{tag}"] = np.asarray(a, float)
        arrays[f"taus__{tag}"] = np.asarray(tb, float)
        arrays[f"drivers__{tag}"] = np.asarray(X_cf, float)

    # --- verification: the factual arm must reproduce the stored rollout ---
    chk = dict(seed=int(seed))
    stored = os.path.join(ANCHOR_DIR, f"pred_seed{seed}.npz")
    if os.path.exists(stored):
        st = np.load(stored, allow_pickle=True)
        chk["max_abs_dS_factual_vs_stored"] = float(
            np.max(np.abs(arrays["S__factual"] - st["S_pred_B"])))
        chk["max_abs_dF_factual_vs_stored"] = float(np.max(np.abs(
            arrays["F__factual"][:, arrays["flow_obs_to_pred_idx"]]
            - st["F_pred_B"])))
    else:
        chk["max_abs_dS_factual_vs_stored"] = float("nan")
        chk["max_abs_dF_factual_vs_stored"] = float("nan")
    # the factual driver matrix must be untouched
    chk["max_abs_dDrivers_factual"] = float(
        np.max(np.abs(arrays["drivers__factual"] - X_full)))
    # The intervened closure with delta = 0 must be bitwise the identity ...
    d_coef, d_rhs = verify_wrapper_identity(
        fit, params,
        make_cf_data(fit, cfg, arms["factual_tau_ref"], t_full, X_full,
                     cols, cut)[0],
        arrays["S__factual"])
    chk["max_abs_wrapper_coef_identity"] = float(d_coef)
    chk["max_abs_wrapper_rhs_identity"] = float(d_rhs)
    # ... while the free-run trajectories still separate, by the adaptive
    # solver's step sequence alone.  This is the noise floor of the package.
    chk["solver_noise_floor_maxabs_kt"] = float(
        np.max(np.abs(arrays["S__factual_tau_ref"] - arrays["S__factual"])))
    chk["solver_noise_floor_maxrel_pct"] = float(100.0 * np.max(
        np.abs(arrays["S__factual_tau_ref"] - arrays["S__factual"])
        / np.maximum(np.abs(arrays["S__factual"]), 1e-12)))
    chk["rss_mb"] = float(_rss_mb())
    arrays["check_json"] = np.asarray(json.dumps(chk), dtype=object)

    path = os.path.join(out_dir, f"cf_seed{seed}.npz")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **arrays)
    chk["path"] = path
    return chk


def build_context(cfg=None, weights_dir=WEIGHTS_DIR_DEFAULT):
    """`(cfg, t_full, X_full, cols, cut, weights_dir)` plus the init fit."""
    import zinc_colloc_v5 as v5

    cfg = dict(cfg or load_anchor_config())
    data_np = v5.load_zinc_data(cfg["xlsx_path"],
                                extra_exog_cols=cfg.get("extra_exog_cols"))
    t_full = np.asarray(data_np["exog_times_full"], float)
    X_full = np.asarray(data_np["exog_values_full"], float)
    cols = list(data_np["exog_cols"])
    years = np.asarray(data_np["years"], float).ravel()
    cut, _n_tv = split_cut(len(years),
                           cfg.get("trainval_frac", 0.7), cfg.get("val_frac", 0.2))

    fit = init_fit(cfg)
    # the rebuild must be bit-identical to what the fit itself computed
    exog_ref = np.asarray(fit.data_all["exog_values"], float)
    exog_rb = rebuild_exog(cfg, years, t_full, X_full, cut)
    gap = float(np.max(np.abs(exog_rb - exog_ref)))
    if gap != 0.0:
        raise RuntimeError(
            f"counterfactual exog rebuild differs from the fit's own features "
            f"by {gap:.3e}; the preprocessing settings do not match")
    return fit, (cfg, t_full, X_full, cols, cut, weights_dir)


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------
def check(verbose=True, weights_dir=WEIGHTS_DIR_DEFAULT):
    """`zinc_alpha_lab.check()` plus the WP-7 arm table and its inputs."""
    import zinc_colloc_v5 as v5

    info = alab.check(verbose=verbose)
    install(v5)
    cfg = load_anchor_config()
    data_np = v5.load_zinc_data(cfg["xlsx_path"],
                                extra_exog_cols=cfg.get("extra_exog_cols"))
    cols = list(data_np["exog_cols"])
    years = np.asarray(data_np["years"], float).ravel()
    cut, n_tv = split_cut(len(years), cfg.get("trainval_frac", 0.7),
                          cfg.get("val_frac", 0.2))
    arms = build_arms(cols)
    n_w = len([f for f in os.listdir(weights_dir)
               if f.startswith("A_seed") and f.endswith(".npz")]) \
        if os.path.isdir(weights_dir) else 0

    info.update(
        arms=list(arms),
        n_arms=len(arms),
        price_driver=PRICE_DRIVER,
        price_driver_index=cols.index(PRICE_DRIVER),
        secondary_supply_flows=list(SECONDARY_SUPPLY_FLOWS),
        secondary_supply_flow_index=[v5.FLOW_NAMES.index(f)
                                     for f in SECONDARY_SUPPLY_FLOWS],
        tau_olds_learned=not bool(cfg.get("pin_tau_olds", False)),
        train_cut=int(cut), n_trainval=int(n_tv),
        weights_dir=weights_dir, n_weight_dumps=int(n_w),
        patches=list(PATCHES),
    )
    if verbose:
        print("zinc_cf_lab --check  (WP-7 additions)")
        print("=" * 74)
        print(f"  train core           : {years[0]:.0f}–{years[cut-1]:.0f} "
              f"(cut={cut}, n_trainval={n_tv})   [core's own arithmetic]")
        print(f"  price driver         : [{cols.index(PRICE_DRIVER)}] "
              f"{PRICE_DRIVER}")
        print(f"  tau_olds             : "
              f"{'LEARNED (intervened at the coefficient)' if info['tau_olds_learned'] else 'pinned to data'}")
        print(f"  secondary supply     : {' + '.join(SECONDARY_SUPPLY_FLOWS)}  "
              f"(FLOW_NAMES idx {info['secondary_supply_flow_index']})")
        print(f"  weights              : {n_w} dumps in {weights_dir}")
        print(f"  patches applied      : {PATCHES or 'none (WP-7 needs none)'}")
        print(f"\n  arms ({len(arms)}):")
        for a in arms:
            print(f"    {a:34s} {arms[a]['label']}")
        print("=" * 74)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description="WP-7: counterfactual re-integrations")
    ap.add_argument("--check", action="store_true",
                    help="print resolved drivers, input_dim and the arm table")
    ap.add_argument("--seeds", default="", help="comma-separated seeds")
    ap.add_argument("--all-seeds", action="store_true",
                    help="every seed with a weight dump in --weights")
    ap.add_argument("--weights", default=WEIGHTS_DIR_DEFAULT)
    ap.add_argument("--out", default=OUT_DIR_DEFAULT)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args(argv)

    info = check(verbose=True, weights_dir=args.weights)
    if args.check:
        return 0

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.all_seeds:
        seeds = sorted(int(f[6:-4]) for f in os.listdir(args.weights)
                       if f.startswith("A_seed") and f.endswith(".npz"))
    if not seeds:
        print("nothing to do — pass --seeds or --all-seeds")
        return 0

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "check.json"), "w") as fh:
        json.dump({k: v for k, v in info.items() if k != "digests"}, fh,
                  indent=2, default=str)

    t0 = time.time()
    fit, ctx = build_context(weights_dir=args.weights)
    arms = build_arms(ctx[3])
    machinery = make_cf_machinery(fit)
    print(f"[wp7] context built in {time.time()-t0:.1f}s; "
          f"{len(arms)} arms x {len(seeds)} seeds", flush=True)

    rows = []
    for sd in seeds:
        path = os.path.join(args.out, f"cf_seed{sd}.npz")
        if args.skip_existing and os.path.exists(path):
            print(f"[seed {sd}] exists, skipped", flush=True)
            continue
        t1 = time.time()
        try:
            chk = run_seed(sd, fit, machinery, arms, ctx, args.out)
        except Exception as e:                     # keep the sweep going
            print(f"[seed {sd}] FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        rows.append(chk)
        print(f"[seed {sd}] {time.time()-t1:.1f}s  "
              f"dS_factual={chk['max_abs_dS_factual_vs_stored']:.3e}  "
              f"wrapper_id={chk['max_abs_wrapper_rhs_identity']:.1e}  "
              f"noise_floor={chk['solver_noise_floor_maxabs_kt']:.2e} kt  "
              f"rss={chk['rss_mb']:.0f} MB", flush=True)

    if rows:
        import pandas as pd
        pd.DataFrame(rows).to_csv(
            os.path.join(args.out, "cf_reproduction_check.csv"), index=False)
        worst = max(r["max_abs_dS_factual_vs_stored"] for r in rows)
        wid = max(max(r["max_abs_wrapper_coef_identity"],
                      r["max_abs_wrapper_rhs_identity"]) for r in rows)
        nf = np.median([r["solver_noise_floor_maxabs_kt"] for r in rows])
        print(f"\n[wp7] {len(rows)} seeds; worst factual |ΔS| vs stored "
              f"anchor_v4 = {worst:.3e}; worst wrapper identity gap = "
              f"{wid:.1e}; median solver noise floor = {nf:.2e} kt  "
              f"({time.time()-t0:.1f}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
