#!/usr/bin/env python3
"""
zinc_coarse_lab.py — lab module for WP-6b (temporal coarsening)
===============================================================

WP-6b asks what happens to coefficient identifiability when the record is
reported every 2nd / 5th / 10th year instead of annually.  This module
supplies the observation operator; `run_wp6b.py` drives it.

Pattern
-------
Follows `zinc_synth_lab.py` (CLAUDE.md rule 1): `zinc_colloc_v5.py` is
imported and never edited, its MD5 is pinned, and the **single** patch is a
dispatcher rebound onto `v5.load_zinc_data` that post-processes the loader's
own output.  Nothing in the loss, the integrator or the optimiser is
touched, so a Delta = 1 arm is bit-identical to `anchor_v4` by construction
and `--noop` verifies it rather than assuming it.  Integrity check and
config loader are reused from `zinc_alpha_lab`.

The observation operator
------------------------
A record reported every `Delta` years retains the annual-grid indices
`{0, Delta, 2*Delta, ...}` (anchored at 1980) and nothing else.  What that
does to a series depends on what kind of quantity the series *is*, and the
three cases are kept apart deliberately:

1.  **Point-in-time observables** — the four stocks.  The analyst has two
    readings `Delta` years apart and interpolates between them, which is what
    every MFA does.  Implemented as linear interpolation through the retained
    readings.  The series stays finite everywhere, which it must: it supplies
    the ODE initial condition (`zinc_colloc_v5.py:1608`) and the training-window
    normalisation constants `S_mean`/`S_std`/`S_target_std`/`S_ref`
    (`:1960-2007`), none of which are NaN-aware.

2.  **Interval quantities** — the 18 flow columns, `cp_obs`, and any *pinned*
    tau, all of which the core reads as an integral or an interval-effective
    coefficient over `(Y-1, Y]`.  The coarse record reports the `Delta`-year
    total; the reconstruction that preserves that total exactly is the flat
    window mean, which is also the only reconstruction consistent with the
    core's own piecewise-constant `_interval_pick` convention (`:638`).
    Units stay annual, so the Stage-B flow loss and the pinned forcings
    remain well posed.

3.  **Supervision-only targets** — `alpha_obs` and the *learned* taus.  These
    are what the coarsening is actually about, and they are genuinely
    **masked**: NaN outside the coarse grid.  Both losses mask on
    `jnp.isfinite` already (`:1464`, `:1473`), and both loss denominators
    (`alpha_log_std`, `tau_sup_std`) are `nanstd` (`:1973`, `:1988`), so no
    patch is needed and CLAUDE.md rule 5 is satisfied by the core's own
    `_per_feature_masked_mse`.

The pinned/learned split in case 2 vs 3 is exactly `learned_tau_mask`
(`:1441`): a tau column is read by the ODE iff it is pinned, and enters the
loss iff it is learned, never both.  Pinned columns therefore must stay
finite and learned columns are free to be masked.

`alpha_obs` under coarsening
----------------------------
alpha is not an observation, it is constructed from one.  Under a coarse
record the construction still holds — annual flow integrals sum to a window
integral without approximation — but it is available once per window, not
once per year:

    alpha_k[c] = ( sum_{i=p}^{c-1} F_k[i] )
                 / ( sum_{i=p}^{c-1} 1/2 (S_par[i] + S_par[i+1]) dt_i )

for consecutive retained indices `p < c`, NaN elsewhere.  At Delta = 1 this
reduces term by term to `_build_empirical_alphas` (`:856`), which is what
makes the no-op guard bit-exact.  The grid alpha lives on is the grid of its
*parent flow*: a coarsened parent stock changes alpha's denominator but not
how often alpha can be formed.

This is the same coarse target WP-11b constructed for its observation-side
surrogate (`wp11b_observation.csv`), so the two packages measure the same
object — 11b measured how far the target moves, 6b measures what refitting
against the moved target costs.

What is *not* coarsened
-----------------------
The exogenous drivers.  WP-6b is about the reporting frequency of the
endogenous record; driver ablation is WP-6c.  `exog_values`,
`exog_values_full` and `exog_cols` pass through untouched, so the driver
universe and `input_dim` are identical in every arm and seed pairing across
arms stays valid (CLAUDE.md rule 2).

CLI
---
    python zinc_coarse_lab.py --check
    python zinc_coarse_lab.py --noop                    # Delta=1 array identity
    python zinc_coarse_lab.py --reach                   # which series reach the loss
    python zinc_coarse_lab.py --seeds 0 --delta 5 --series all --out analysis/wp6b
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import time

import numpy as np

import zinc_alpha_lab as lab

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "analysis", "wp6b")

# Reporting intervals, in years.  Delta = 1 is the published annual record and
# is carried as an arm so the curve has its own internal control rather than
# borrowing anchor_v4's number.
DELTAS = (1, 2, 5, 10)

STOCK_NAMES = ("Concentrate", "Refined", "In-Use", "Scrap")

# The four alpha channels, as (parent flow column name, parent stock index).
# Mirrors ALPHA_NAMES / ALPHA_PARENT_STOCK_IDX and the F_for_alpha stack in
# `load_zinc_data` (:952-969).  Resolved by explicit name, never by position
# in the sheet (CLAUDE.md rule 2).
ALPHA_PARENTS = (
    ("alpha_cc",    "concentrate_consumption", 0),
    ("alpha_refc",  "refined_consumption",     1),
    ("alpha_win",   "waelz_input",             3),
    ("alpha_dr",    "direct_reuse_recycling",  3),
)

PATCHES: list[str] = []

_ORIG_LOADER = None
_ARMED: dict | None = None          # {"delta": int, "series": tuple[str, ...]}


# ---------------------------------------------------------------------------
# series universe
# ---------------------------------------------------------------------------
def series_universe(data=None, cfg=None):
    """Canonical, fixed ordering of every endogenous series in the record.

    4 stocks + 18 flows + 8 tau-supervision columns = 30.  The order is
    stocks, then the flow columns in `flow_obs_names` order, then
    `TAU_SUP_NAMES` order — all of which are fixed by the core — so an arm
    index means the same thing in every run.  Never re-order this.
    """
    import zinc_colloc_v5 as v5
    if data is None:
        data = raw_data(cfg)
    names = [f"stock:{n}" for n in STOCK_NAMES]
    names += [f"flow:{n}" for n in list(data["flow_obs_names"])]
    names += [f"tau:{n}" for n in list(v5.TAU_SUP_NAMES)]
    return tuple(names)


def raw_data(cfg=None):
    """The unpatched loader's output — the annual reference record."""
    import zinc_colloc_v5 as v5
    cfg = dict(cfg or lab.load_anchor_config())
    loader = _ORIG_LOADER or v5.load_zinc_data
    return loader(cfg["xlsx_path"], extra_exog_cols=cfg.get("extra_exog_cols"))


def _resolve(series, universe):
    """Expand `series` to a concrete tuple of universe members."""
    if series is None:
        return ()
    if isinstance(series, str):
        if series == "all":
            return tuple(universe)
        if series == "stocks":
            return tuple(n for n in universe if n.startswith("stock:"))
        if series == "flows":
            return tuple(n for n in universe if n.startswith("flow:"))
        if series == "taus":
            return tuple(n for n in universe if n.startswith("tau:"))
        series = [series]
    out = []
    for s in series:
        if s not in universe:
            raise KeyError(f"unknown series {s!r}; universe has {len(universe)} "
                           f"members, e.g. {universe[:3]}")
        out.append(s)
    return tuple(out)


# ---------------------------------------------------------------------------
# the observation operator
# ---------------------------------------------------------------------------
def keep_indices(T, delta):
    """Retained annual-grid indices for a record reported every `delta` years."""
    return np.arange(0, T, int(delta), dtype=int)


def _window_bounds(keep, n_rows):
    """(p, c) index pairs for consecutive retained readings, plus the trailing
    partial window if the grid does not divide the record.

    Flow row i closes year i+1, so window (p, c] owns flow rows p .. c-1.
    A partial tail (e.g. Delta = 10 on 40 years leaves nothing; Delta = 3
    would) is kept as its own shorter window rather than discarded, so no
    observation is silently dropped.
    """
    edges = list(keep)
    if edges[-1] < n_rows:
        edges.append(n_rows)
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def _flat_window(series, keep, T):
    """Interval quantity -> flat window mean, preserving each window total.

    `series` is indexed so that row i is the integral over (years[i], years[i+1]]
    for flows (length T-1) or over (years[i-1], years[i]] for cp/tau (length T,
    row 0 outside the integration domain and left untouched).
    """
    out = np.array(series, dtype=float, copy=True)
    n = out.shape[0]
    off = 0 if n == T - 1 else 1          # cp/tau are year-indexed, flows are not
    for p, c in _window_bounds(keep, T - 1):
        rows = np.arange(p, c) + off
        rows = rows[rows < n]
        if rows.size == 0:
            continue
        vals = out[rows]
        if not np.any(np.isfinite(vals)):
            continue
        out[rows] = np.nanmean(vals)
    return out


def _interp_series(series, keep):
    """Point-in-time quantity -> linear interpolation through retained readings."""
    y = np.asarray(series, dtype=float)
    idx = np.arange(y.shape[0], dtype=float)
    k = np.asarray(keep, dtype=int)
    good = k[np.isfinite(y[k])]
    if good.size < 2:
        return y.copy()
    return np.interp(idx, good.astype(float), y[good])


def _mask_off_grid(series, keep):
    """Supervision-only quantity -> NaN outside the coarse grid."""
    out = np.array(series, dtype=float, copy=True)
    m = np.zeros(out.shape[0], dtype=bool)
    m[np.asarray(keep, dtype=int)] = True
    out[~m] = np.nan
    return out


def _rebuild_alpha(years, S_obs, flows_obs, flow_names, alpha_grids):
    """Empirical alpha on each channel's own coarse grid.

    `alpha_grids[k]` is the retained index array for channel k's parent flow.
    At a single-year grid this reproduces `_build_empirical_alphas` term by
    term, which `--noop` checks to 0.0.
    """
    import zinc_colloc_v5 as v5

    T = len(years)
    dt = years[1:] - years[:-1]
    out = np.full((T, v5.N_ALPHAS), np.nan, dtype=float)
    cols = {n: i for i, n in enumerate(flow_names)}

    for k, (_name, parent_flow, parent_stock) in enumerate(ALPHA_PARENTS):
        F = np.clip(flows_obs[:, cols[parent_flow]], 0.0, np.inf)     # (T-1,)
        Sp = S_obs[:, parent_stock]                                   # (T,)
        expo_annual = 0.5 * (Sp[:-1] + Sp[1:]) * dt                   # (T-1,)
        for p, c in _window_bounds(alpha_grids[k], T - 1):
            num = np.sum(F[p:c])
            den = np.sum(expo_annual[p:c])
            if np.isfinite(num) and np.isfinite(den) and den > 1e-12:
                out[c, k] = num / max(den, 1e-12)
    return out


def coarsen(data, delta, series, cfg=None, verbose=False):
    """Apply the WP-6b observation operator to a `load_zinc_data` dict.

    `series` names which members of `series_universe()` are degraded; every
    other series keeps its annual resolution.  Returns a new dict; `data` is
    not modified.  Exogenous arrays pass through by reference-copy untouched.
    """
    import zinc_colloc_v5 as v5

    delta = int(delta)
    if delta < 1:
        raise ValueError(f"delta must be >= 1, got {delta}")

    out = {k: (v.copy() if isinstance(v, np.ndarray) else
               (list(v) if isinstance(v, list) else v))
           for k, v in data.items()}
    universe = series_universe(data)
    targets = set(_resolve(series, universe))
    if delta == 1 or not targets:
        return out

    years = np.asarray(out["years"], float)
    T = len(years)
    keep = keep_indices(T, delta)
    flow_names = list(out["flow_obs_names"])
    tau_names = list(v5.TAU_SUP_NAMES)
    learned = _learned_tau_mask(cfg)

    # --- 1) point-in-time: stocks -----------------------------------------
    S = np.asarray(out["stocks_obs"], float)
    for j, nm in enumerate(STOCK_NAMES):
        if f"stock:{nm}" in targets:
            S[:, j] = _interp_series(S[:, j], keep)
    out["stocks_obs"] = S

    # --- 2) interval quantities: flows, cp, pinned taus -------------------
    F = np.asarray(out["flows_obs"], float)
    for j, nm in enumerate(flow_names):
        if f"flow:{nm}" in targets:
            F[:, j] = _flat_window(F[:, j], keep, T)
    out["flows_obs"] = F

    # cp_obs is the `concentrate_production` column read as a pinned forcing
    # (`_interp_cp`, :681).  It is the same physical series, so it is degraded
    # with it and never separately.
    if "flow:concentrate_production" in targets:
        out["cp_obs"] = _flat_window(np.asarray(out["cp_obs"], float), keep, T)

    # --- 3) taus: pinned reconstruct, learned mask ------------------------
    tau = np.asarray(out["tau_sup_obs"], float)
    for j, nm in enumerate(tau_names):
        if f"tau:{nm}" not in targets:
            continue
        if learned[j]:
            tau[:, j] = _mask_off_grid(tau[:, j], keep)
        else:
            tau[:, j] = _flat_window(tau[:, j], keep, T)
    out["tau_sup_obs"] = tau

    # --- 4) alpha: rebuilt on each channel's parent-flow grid -------------
    annual = np.arange(T, dtype=int)
    grids = [keep if f"flow:{pf}" in targets else annual
             for (_n, pf, _s) in ALPHA_PARENTS]
    out["alpha_obs"] = _rebuild_alpha(years, out["stocks_obs"], out["flows_obs"],
                                      flow_names, grids)

    if verbose:
        n_a = np.isfinite(out["alpha_obs"]).sum(0)
        print(f"[coarsen] delta={delta} series={len(targets)} "
              f"alpha rows/channel={n_a.tolist()}")
    return out


def _learned_tau_mask(cfg=None):
    """`learned_tau_mask` for a config, without building a layout (:1441)."""
    import zinc_colloc_v5 as v5
    cfg = dict(v5.DEFAULT_CONFIG, **dict(cfg or lab.load_anchor_config()))
    return np.array([
        not cfg["pin_tau_ref"], not cfg["pin_tau_waelz"],
        not cfg["pin_tau_olds"], not cfg["pin_tau_diss"],
        not cfg["pin_frac_fu_new"], not cfg["pin_frac_fu_loss"],
        not cfg["pin_frac_eu_new"], not cfg["pin_frac_eu_loss"],
    ], dtype=bool)


# ---------------------------------------------------------------------------
# install / arm
# ---------------------------------------------------------------------------
def install(v5mod=None):
    """Rebind `v5.load_zinc_data` to the coarsening dispatcher.  Idempotent.

    With no arm set the dispatcher forwards to the original loader, so an
    installed-but-disarmed module is behaviourally identical to an
    uninstalled one (`--noop` verifies this array by array).
    """
    global _ORIG_LOADER
    if v5mod is None:
        import zinc_colloc_v5 as v5mod
    if _ORIG_LOADER is not None:
        return PATCHES
    _ORIG_LOADER = v5mod.load_zinc_data

    def _dispatch(xlsx_path, extra_exog_cols=None, extra_sheet="extra"):
        raw = _ORIG_LOADER(xlsx_path, extra_exog_cols=extra_exog_cols,
                           extra_sheet=extra_sheet)
        if _ARMED is None:
            return raw
        return coarsen(raw, _ARMED["delta"], _ARMED["series"],
                       cfg=_ARMED.get("cfg"))

    v5mod.load_zinc_data = _dispatch
    PATCHES.append("zinc_colloc_v5.load_zinc_data -> WP-6b coarsening dispatcher")
    return PATCHES


def arm(delta, series, cfg=None):
    """Arm a coarsening so the next `train_model` consumes the degraded record."""
    global _ARMED
    universe = series_universe(cfg=cfg)
    _ARMED = dict(delta=int(delta), series=_resolve(series, universe), cfg=cfg)
    return _ARMED


def disarm():
    global _ARMED
    _ARMED = None


def arm_tag(delta, series):
    """Filename-safe, round-trippable tag for one arm."""
    if isinstance(series, str) and series in ("all", "stocks", "flows", "taus"):
        s = series
    else:
        s = "+".join(series) if not isinstance(series, str) else series
        s = s.replace(":", "-").replace(" ", "_")
    return f"{s}_d{int(delta)}"


def _rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024.0 ** 2) if sys.platform == "darwin" else r / 1024.0


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------
def noop_check(cfg=None, verbose=True):
    """Delta = 1 must return the annual record array-for-array, bit-exactly.

    Also checks the disarmed dispatcher, and that a full `all`-series arm at
    Delta = 1 leaves `alpha_obs` — which is *rebuilt*, not passed through —
    identical to the core's own `_build_empirical_alphas`.  That last one is
    the guard that matters: it is the only place this module re-implements
    core arithmetic.
    """
    import zinc_colloc_v5 as v5
    install(v5)
    cfg = dict(cfg or lab.load_anchor_config())
    ref = raw_data(cfg)

    rows = []
    for tag, armed in (("disarmed", None),
                       ("d1_all", dict(delta=1, series="all")),
                       ("d1_stocks", dict(delta=1, series="stocks"))):
        if armed is None:
            disarm()
        else:
            arm(armed["delta"], armed["series"], cfg=cfg)
        got = v5.load_zinc_data(cfg["xlsx_path"],
                                extra_exog_cols=cfg.get("extra_exog_cols"))
        for k, v in ref.items():
            if not isinstance(v, np.ndarray) or v.dtype == object:
                continue
            g = np.asarray(got[k], float)
            r = np.asarray(v, float)
            same_nan = np.array_equal(np.isnan(g), np.isnan(r))
            d = np.nanmax(np.abs(g - r)) if r.size else 0.0
            rows.append(dict(arm=tag, array=k, max_abs_diff=float(np.nan_to_num(d)),
                             nan_pattern_same=bool(same_nan)))
    disarm()

    # The rebuild path, exercised explicitly on the annual grid.
    annual = [np.arange(len(ref["years"]), dtype=int)] * 4
    rebuilt = _rebuild_alpha(np.asarray(ref["years"], float),
                             np.asarray(ref["stocks_obs"], float),
                             np.asarray(ref["flows_obs"], float),
                             list(ref["flow_obs_names"]), annual)
    a = np.asarray(ref["alpha_obs"], float)
    rows.append(dict(arm="rebuild_annual", array="alpha_obs",
                     max_abs_diff=float(np.nanmax(np.abs(rebuilt - a))),
                     nan_pattern_same=bool(np.array_equal(np.isnan(rebuilt),
                                                          np.isnan(a)))))
    ok = all(r["max_abs_diff"] == 0.0 and r["nan_pattern_same"] for r in rows)
    if verbose:
        bad = [r for r in rows if not (r["max_abs_diff"] == 0.0
                                       and r["nan_pattern_same"])]
        print(f"[noop] {len(rows)} array comparisons, "
              f"{'ALL BIT-EXACT' if ok else f'{len(bad)} MISMATCH'}")
        for r in bad:
            print(f"    {r['arm']:14s} {r['array']:16s} "
                  f"max|d|={r['max_abs_diff']:.3e} nan_same={r['nan_pattern_same']}")
    return rows, ok


def reach_audit(delta=10, cfg=None, verbose=True):
    """Which series can reach the anchor_v4 objective at all, and how far.

    For every member of the universe, coarsen it alone at `delta` and record
    (i) which observation arrays move, and (ii) whether the Stage A and
    Stage B losses and their gradients move, evaluated at the *same*
    warm-started initial parameters.  Needs no fit: a series that cannot move
    the loss at the initialisation cannot move it anywhere, because it has no
    path into the objective at all.

    This matters because `anchor_v4` sets `stageB_w_F = 0.0` — the flow-
    divergence term is off — so most flow columns enter only through the four
    alpha numerators, and the rest are inert by construction.  Running the
    full 30-series grid without checking this would spend most of its fits
    re-deriving `anchor_v4`.
    """
    import jax
    import jax.numpy as jnp
    import zinc_colloc_v5 as v5

    lab.integrity_check()
    install(v5)
    cfg = dict(v5.DEFAULT_CONFIG, **dict(cfg or lab.load_anchor_config()))
    cfg["verbose"] = False
    universe = series_universe(cfg=cfg)

    disarm()
    fit0 = v5.run("wp6b_reach", **dict(cfg, seed=0, stageA_steps=0,
                                       do_stage_B=False))
    params = fit0.params_A
    stageA_loss, _ = v5.make_stageA_loss(fit0.nn_eval_sup, fit0.layout)
    stageB_loss = v5.make_stageB_loss(
        fit0.integrate_aug, fit0.nn_eval_sup, fit0.layout,
        fit0.flow_obs_to_pred_idx,
        stock_loss_kind=cfg["stock_loss_kind"])

    kwA = dict(w_alpha=cfg["stageA_w_alpha"], w_tau=cfg["stageA_w_tau"],
               w_cp=cfg["stageA_w_cp"],
               w_smooth_alpha=cfg["stageA_w_smooth_alpha"],
               w_smooth_tau=cfg["stageA_w_smooth_tau"],
               w_cohort_prior=cfg["stageA_w_cohort_prior"],
               w_raw_reg=cfg["stageA_w_raw_reg"])
    kwB = dict(w_alpha=cfg["stageB_w_alpha"], w_tau=cfg["stageB_w_tau"],
               w_cp=cfg["stageB_w_cp"], w_S=cfg["stageB_w_S"],
               w_F=cfg["stageB_w_F"],
               w_smooth_alpha=cfg["stageB_w_smooth_alpha"],
               w_smooth_tau=cfg["stageB_w_smooth_tau"],
               w_cohort_prior=cfg["stageB_w_cohort_prior"],
               w_raw_reg=cfg["stageB_w_raw_reg"])
    win = int(cfg["stageB_curriculum"][-1][1])

    def _probe(data_j):
        la, ga = jax.value_and_grad(lambda p: stageA_loss(p, data_j, **kwA))(params)
        lb, gb = jax.value_and_grad(
            lambda p: stageB_loss(p, data_j, 0, win, **kwB)[0])(params)
        gn = lambda g: float(jnp.sqrt(sum(jnp.sum(x ** 2)
                                          for x in jax.tree_util.tree_leaves(g))))
        return float(la), gn(ga), float(lb), gn(gb)

    def _rebuild_data(raw):
        """Re-run `train_model`'s data assembly on a coarsened record."""
        arm_cfg = dict(cfg, seed=0, stageA_steps=0, do_stage_B=False)
        f = v5.run("wp6b_probe", **arm_cfg)
        return f

    ref_raw = raw_data(cfg)
    base = _probe(fit0.data_all)
    rows = []
    for nm in universe:
        arm(delta, [nm], cfg=cfg)
        cd = coarsen(ref_raw, delta, [nm], cfg=cfg)
        moved = sorted(k for k, v in cd.items()
                       if isinstance(v, np.ndarray) and v.dtype != object
                       and not (np.array_equal(np.isnan(np.asarray(v, float)),
                                               np.isnan(np.asarray(ref_raw[k], float)))
                                and np.allclose(np.asarray(v, float),
                                                np.asarray(ref_raw[k], float),
                                                equal_nan=True, rtol=0, atol=0)))
        f = _rebuild_data(None)
        got = _probe(f.data_all)
        del f
        jax.clear_caches()
        rows.append(dict(
            series=nm, delta=int(delta), arrays_moved=";".join(moved),
            dA=abs(got[0] - base[0]), dgA=abs(got[1] - base[1]),
            dB=abs(got[2] - base[2]), dgB=abs(got[3] - base[3]),
            reaches_loss=bool(abs(got[0] - base[0]) > 0 or abs(got[2] - base[2]) > 0),
        ))
        if verbose:
            r = rows[-1]
            print(f"  {nm:44s} dLA={r['dA']:.3e} dLB={r['dB']:.3e} "
                  f"{'REACHES' if r['reaches_loss'] else 'inert'}", flush=True)
    disarm()
    del fit0
    jax.clear_caches()
    return rows, base


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------
def dump_fit(fit, path, ref):
    """Persist what WP-6b scores: NN alpha/tau at observed stocks, the
    rollouts, and — separately — the *annual* reference targets.

    The distinction is the whole point.  A coarse arm is fitted against a
    degraded target but must be **scored against the annual record**,
    otherwise the curve confounds how much worse the estimator got with how
    much worse the target got.  WP-11b measured the second on its own; this
    file carries both so the two can be reported apart.
    """
    import zinc_colloc_v5 as v5

    masks = fit._split_indices_in_all()
    prB = fit.predictions("B")
    stats = fit.data_all["stats"]
    arrays = dict(
        years=np.asarray(fit.data_all["years"], float).ravel(),
        alpha_names=np.array(v5.ALPHA_NAMES, dtype=object),
        tau_names=np.array(v5.TAU_SUP_NAMES, dtype=object),
        stock_names=np.array(STOCK_NAMES, dtype=object),
        flow_names=np.array(list(ref["flow_obs_names"]), dtype=object),
        flow_obs_to_pred_idx=np.asarray(fit.flow_obs_to_pred_idx, int),
        # the record this arm was fitted against
        alpha_obs_arm=np.asarray(fit.data_all["alpha_obs"], float),
        tau_obs_arm=np.asarray(fit.data_all["tau_sup_obs"], float),
        S_obs_arm=np.asarray(fit.data_all["stocks_obs"], float),
        F_obs_arm=np.asarray(fit.data_all["flows_obs"], float),
        # the annual record, which is what everything is scored against
        alpha_obs_annual=np.asarray(ref["alpha_obs"], float),
        tau_obs_annual=np.asarray(ref["tau_sup_obs"], float),
        S_obs_annual=np.asarray(ref["stocks_obs"], float),
        F_obs_annual=np.asarray(ref["flows_obs"], float),
        # predictions
        alpha_pred_B_at_obs=np.asarray(prB["alphas_at_obs"], float),
        tau_pred_B_at_obs=np.asarray(prB["taus_at_obs"], float),
        S_pred_B=np.asarray(prB["S_pred"], float),
        F_pred_B=np.asarray(prB["F_int"], float),
        alpha_log_std=np.asarray(stats["alpha_log_std"], float),
        mask_train=np.asarray(masks["train"], bool),
        mask_val=np.asarray(masks["val"], bool),
        mask_test=np.asarray(masks["test"], bool),
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def fit_arm(seed, delta, series, out_dir=OUT_DIR, cfg=None, verbose=False,
            skip_existing=False):
    """Fit one seed under one coarsening arm and dump it."""
    import jax
    import zinc_colloc_v5 as v5

    lab.integrity_check()
    install(v5)
    cfg = dict(cfg or lab.load_anchor_config())
    cfg["verbose"] = bool(verbose)
    tag = arm_tag(delta, series)
    path = os.path.join(out_dir, f"{tag}_seed{seed}.npz")
    if skip_existing and os.path.exists(path):
        print(f"[{tag} seed {seed}] exists, skipped", flush=True)
        return path

    ref = raw_data(cfg)
    arm(delta, series, cfg=cfg)
    t0 = time.time()
    try:
        fit = v5.run(f"wp6b_{tag}_s{seed}", **dict(cfg, seed=seed))
        dump_fit(fit, path, ref)
    finally:
        disarm()
    dt = time.time() - t0

    del fit
    jax.clear_caches()                       # CLAUDE.md rule 6
    print(f"[{tag} seed {seed}] {dt/60:.1f} min  rss={_rss_mb():.0f} MB -> {path}",
          flush=True)
    return path


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------
def check(verbose=True):
    """Resolved drivers, input_dim, series universe, patches — before any work."""
    import zinc_colloc_v5 as v5
    install(v5)
    disarm()
    info = lab.check(verbose=verbose)
    data = raw_data()
    universe = series_universe(data)
    learned = _learned_tau_mask()
    info = dict(info, series_universe=list(universe),
                n_series=len(universe), deltas=list(DELTAS),
                learned_tau=[n for n, l in zip(v5.TAU_SUP_NAMES, learned) if l],
                pinned_tau=[n for n, l in zip(v5.TAU_SUP_NAMES, learned) if not l],
                patches=list(PATCHES))
    if verbose:
        print("=" * 74)
        print("zinc_coarse_lab --check")
        print("=" * 74)
        print(f"  patches applied      : {PATCHES}")
        print(f"  reporting intervals  : {list(DELTAS)} yr")
        print(f"  series universe      : {len(universe)} "
              f"(4 stocks + {len(data['flow_obs_names'])} flows + "
              f"{len(v5.TAU_SUP_NAMES)} taus)")
        for i, n in enumerate(universe):
            print(f"    [{i:2d}] {n}")
        print(f"\n  learned tau (masked when coarsened) : {info['learned_tau']}")
        print(f"  pinned  tau (reconstructed)         : {info['pinned_tau']}")
        for d in DELTAS:
            k = keep_indices(len(data["years"]), d)
            print(f"  delta={d:2d}: {len(k):2d} retained readings, "
                  f"{len(_window_bounds(k, len(data['years']) - 1))} alpha windows")
        print("=" * 74)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--noop", action="store_true",
                    help="verify delta=1 is bit-exactly the annual record")
    ap.add_argument("--reach", action="store_true",
                    help="per-series loss-reach audit (no fits)")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--delta", type=int, default=1)
    ap.add_argument("--series", default="all")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    info = check(verbose=True)
    if args.check:
        return 0
    if args.noop:
        _rows, ok = noop_check()
        return 0 if ok else 1
    if args.reach:
        rows, _base = reach_audit()
        import pandas as pd
        os.makedirs(args.out, exist_ok=True)
        pd.DataFrame(rows).to_csv(os.path.join(args.out, "reach.csv"), index=False)
        return 0
    if not args.seeds:
        return 0

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "check.json"), "w") as fh:
        json.dump({k: v for k, v in info.items() if k != "digests"}, fh,
                  indent=2, default=str)
    series = args.series if args.series in ("all", "stocks", "flows", "taus") \
        else [s for s in args.series.split(",") if s.strip()]
    cfg = lab.load_anchor_config()
    for sd in [int(s) for s in args.seeds.split(",") if s.strip()]:
        try:
            fit_arm(sd, args.delta, series, out_dir=args.out, cfg=cfg,
                    verbose=args.verbose, skip_existing=args.skip_existing)
        except Exception as e:
            print(f"[delta={args.delta} seed {sd}] FAILED: "
                  f"{type(e).__name__}: {e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
