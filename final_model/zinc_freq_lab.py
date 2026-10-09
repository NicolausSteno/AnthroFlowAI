#!/usr/bin/env python3
"""
zinc_freq_lab.py — lab module for WP-11a (frequency-value curves)
=================================================================

WP-11a asks the direct form of the paper's data-collection question: does
observing the cycle *more often* buy anything, and is whatever it buys coming
from resolution or merely from having more numbers?  The spec's design answers
that by refusing to confound the two — one arm varies the observation width at
a fixed span, a second varies the span at a fixed width — and this module
supplies the datasets, the split, the curriculum and the scoring grid that
make the arms comparable.  `run_wp11a.py` drives it.

Pattern (CLAUDE.md rule 1)
--------------------------
`zinc_colloc_v5.py` is imported and never edited.  Two rebindings are
involved.  The first is `zinc_synth_lab.install()`'s arm-aware
`load_zinc_data` dispatcher, which is reused rather than duplicated, and
`zinc_coarse_lab.coarsen` is called as a *pure function* on an already built
dataset dict rather than through its own installer, so the two labs never
contend for the same symbol.  The second is **this module's own, and it is the
only one**: `install_exog()` wraps `preprocess_exog` so the order-o exogenous
block can be divided by `delta**o`, which is what the `freqdx` control arm
needs (see "The order-1 confound" below).  It is **disarmed by default and
armed only for a `freqdx` arm** — disarmed it returns the original array bit
for bit, and armed at delta = 1 it is likewise the identity, both asserted by
`--noop` rather than claimed.  Every fit in this package outside the three
`freqdx` arms therefore runs against an unpatched core.  Everything else
WP-11a changes about a run — how many observation rows there are, where the
trainval boundary sits, how long a Stage B window is — is either a property of
the armed dataset or a plain `train_model` keyword.

The four arms
-------------
All are the `base` (representable, no sub-annual structure, no state term)
twin under WP-3's calibrated observation noise, so nothing here is testing
misspecification; the object under test is the observation design alone.

    freq     span 1980-2019, delta in {1/12, 1/4, 1/2, 1}      12/4/2/1 per yr
             Spec arm 1 — resolution at fixed record length.  The record is
             re-sampled from the same dense solve, so a finer arm is a strict
             refinement of a coarser one and no arm sees a different truth.

    coarse   span 1980-2019, delta in {2, 5, 10}
             The overlap region WP-11b's gate is evaluated on.  Built with
             **WP-6b's own operator** (`zinc_coarse_lab.coarsen`) applied to
             the twin's annual record, not with the re-sampler: the gate
             compares a synthetic curve against a real one, and it is only a
             comparison of *data* if the operator on both sides is the same
             function.  Stocks are interpolated, interval quantities are flat
             window means, learned targets are masked — see that module.

    span     delta = 1, span in {10, 20, 39} yr
             Spec arm 2 — record length at fixed resolution.  Truncated from
             1980 forward, so every span shares an initial condition and a
             driver history and the shorter arms are prefixes of the longer.

    matchedN delta in {1/4, 1/12} on spans chosen to hold N fixed
             Not in the spec, and it is here because the spec's arm 2 as
             written cannot be run: it asks for an 80-year record and the
             driver record is 1975-2019.  See "What is not run" below.  Two
             settings put a fine-resolution short record at the same N as a
             coarse-resolution long one, which is the contrast arm 2's 80-year
             point was for, obtained without fabricating drivers.

What is not run, and why
------------------------
**The spec's 80-year span.**  `exog_times_full` spans 1975-2019 and the pinned
series (`cp_obs`, the four pinned tau slots) span 1980-2019.  An 80-year twin
would need forty fabricated years of world GDP, zinc price, treatment charges
and concentrate production, and every coefficient in `zinc_synth_lab.TRUTH` is
a tanh whose inflection lies in 1997-2008, so the fabricated half would sit
entirely in the saturated tail and carry systematically less information per
year than the observed half.  The N-vs-resolution contrast that point was for
is made instead by the `matchedN` arm, which holds N fixed and varies delta
using only real drivers.  Reported as a limitation, not worked around.

The split, pinned in years
--------------------------
Cross-frequency comparison is meaningless if the arms do not hold out the same
*period*.  `train_model`'s default `trainval_frac=0.7` splits on row count, so
at delta = 1/12 it would put the boundary in a different year than at
delta = 1.  `split_for` instead resolves the boundary from the year axis and
hands `train_model` explicit `split_indices`, and at (delta = 1, full span) it
reproduces the default's indices exactly — `check()` asserts that rather than
trusting it.  The fractions themselves (train to 1980+21/39 of the span, val
to 1980+27/39) are `anchor_v4`'s own, read off the annual case.

The curriculum, matched in years
--------------------------------
`stageB_curriculum` windows are row counts, so the published `[[800, 8],
[800, 16], [400, 22]]` means 8/16/22-year windows only on an annual grid.  At
delta = 1/12 the same numbers would be 8/16/22-*month* windows and the arm
would be measuring the curriculum, not the frequency.  `curriculum_for`
rescales the window lengths by 1/delta so the physical horizon is fixed, and
leaves the step counts alone so the optimiser budget is fixed too.  The core
clamps a window to the trainval length itself (v5:2309), which is what the
short `span` arms need and is left to it.

Known confound, measured rather than assumed
--------------------------------------------
`preprocess_exog` builds the order-1 feature by differencing consecutive rows
of whatever grid it is handed, so at delta < 1 it is a per-*window* change and
its magnitude falls with delta; `exog_std` clamps at 1.0 (v5:1965) so nothing
rescales it back.  The order-1 block therefore shrinks as the record gets
finer.  `feature_scale_report` measures the per-block SD at every delta so the
size of the effect is on the record: 0.1035 at delta = 1 against 0.0084 at
1/12, a twelvefold artefact, small next to the order-0 block (~0.30) at every
frequency but not nothing.

**It is corrected and tested rather than only flagged.**  The `freqdx` arms
divide the order-o block by `delta**o`, turning the row difference into a
difference QUOTIENT and restoring the annual-equivalent scale (0.1008 at
delta = 1/12 against the annual 0.1035, a residual of 2.6%).  That needs the
one patch this module owns, because the correction cannot be expressed as a
`train_model` keyword.  The arms share the `freq` record file, so they isolate
the feature scale and nothing else.

CLI
---
    python zinc_freq_lab.py --check
    python zinc_freq_lab.py --generate
    python zinc_freq_lab.py --noop
    python zinc_freq_lab.py --fit freq_d1y --seeds 0,1
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

import numpy as np

import zinc_alpha_lab as lab
import zinc_synth_lab as S
import zinc_coarse_lab as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "analysis")
SYNTH_DIR = os.path.join(OUT_DIR, "synth")
FIT_DIR = os.path.join(SYNTH_DIR, "fits")
MANIFEST = os.path.join(SYNTH_DIR, "wp11a_manifest.json")

# This module rebinds nothing.  Kept as a named empty list so `--check`'s
# report has the same shape as every other lab module's.
PATCHES: list[str] = []

ALPHA_NAMES = S.ALPHA_NAMES
TAU_SUP_NAMES = S.TAU_SUP_NAMES
STOCK_NAMES = ["Concentrate", "Refined", "In-Use", "Scrap"]

TWIN_ARM = "base"
NOISE_SCALE = 1.0
NOISE_RNG_SEED = 0

# ---- the design ------------------------------------------------------------
# Spec arm 1: 1, 2, 4 and 12 observations per year over the full record.
DELTAS_FREQ = (1.0 / 12.0, 0.25, 0.5, 1.0)
# The overlap the WP-11b gate is evaluated on, built with WP-6b's operator.
DELTAS_COARSE = (2.0, 5.0, 10.0)
# Spec arm 2: record length at annual frequency.  1980-2019 is 39 years of
# flow intervals (WP-3 flag 2), so the full span is 39, not 40.
FULL_SPAN = 39.0
SPANS = (10.0, 20.0, FULL_SPAN)
# Matched-N diagonal: (delta, span) chosen so N matches a `freq` setting.
#   (1/4, 10 yr)  -> N = 41  against  (1, 39 yr) -> N = 40
#   (1/12, 13 yr) -> N = 157 against  (1/4, 39 yr) -> N = 157
MATCHED_N = ((0.25, 10.0), (1.0 / 12.0, 13.0))

# The spec's own words are "generate a truth *containing realistic sub-annual
# structure* and sample at 1, 2, 4 and 12 observations per year", so the
# frequency widths are run a second time on WP-3's `season` twin, whose
# alpha_refc carries four sub-annual components -- two at exact harmonics of
# the annual window (annihilated by period integration) and two off-harmonic
# (attenuated by |sinc| and aliased).  It is a *second* curve rather than the
# main one for a reason worth stating up front: `anchor_v4` sets
# `use_time_input: false` and `use_stock_input: false` (SCHEMA flag 2), so the
# only inputs the network has are annually-interpolated drivers and **no
# architecture in this family can emit a sub-annual oscillation at any
# observation frequency**.  On the season twin, therefore, finer observation
# cannot buy representation of the seasonal term; what it can buy is a target
# less corrupted by it, and separating those two is the point of running both
# twins rather than one.
SEASON_DELTAS = DELTAS_FREQ

# A control on the one hyperparameter that `curriculum_for` does not reach.
# `zinc_colloc_v5._smoothness_pen` (v5:1391) is `mean((x[i+1] - x[i])**2)` over
# *rows*, with no reference to the spacing, so a weight tuned on an annual grid
# penalises a per-month change on a monthly one.  Holding the penalty on
# `d log alpha / dt` fixed instead of the penalty on the per-row step means
# scaling the weight by `1/delta**2`, since `(x[i+1]-x[i])**2 ~ (dx/dt)**2
# delta**2`.  The `freqsm` arms are the `freq` arms refitted with exactly that
# rescaling and **the identical record** -- they share the dataset file, so the
# pair is a clean control on the smoothing and on nothing else.
SMOOTH_KEYS = ("stageA_w_smooth_alpha", "stageA_w_smooth_tau",
               "stageB_w_smooth_alpha", "stageB_w_smooth_tau")
FREQSM_DELTAS = (1.0 / 12.0, 0.25, 0.5)      # delta = 1 is the arm itself

# A second control, on the other grid-dependent quantity `curriculum_for` does
# not reach.  Stage B samples window *start indices* uniformly on
# [0, T_trainval - window], so the final trainval row is covered by only those
# starts that reach it -- 14.3% of them at delta = 1, 4.5% at 1/4 and 1.6% at
# 1/12.  Rescaling the window in years keeps the horizon fixed but makes the
# *coverage* of the window's own edge fall with delta, and the alpha error at
# the trainval boundary is exactly where the frequency arm looks worst.  The
# `freqfw` arms set the final curriculum stage to the full trainval span, so
# every row is covered by every start at every width, and are otherwise
# identical to `freq` -- same record file, same everything else.
FREQFW_DELTAS = (1.0 / 12.0, 1.0)

# The third control, and the one flag 1 of the findings note left open.
# `preprocess_exog` builds its order-o features as `np.diff(X, n=o, axis=0)`
# over ROWS, with no reference to the row spacing, so an order-1 feature is
# `dx/dt * delta` and its scale falls linearly with delta -- measured at
# 0.1035 (annual) against 0.0084 (monthly) by `feature_scale_report`.
# `exog_std` would normally absorb that, but it clamps at 1.0 (v5:1965) and
# both values are far below the clamp, so nothing restores it: the network
# sees the same rate-of-change information at 1/12 the amplitude.  The
# `freqdx` arms divide the order-o block by `delta**o`, turning the difference
# into a difference QUOTIENT and putting the block on an annual-equivalent
# footing at every width.  At delta = 1 the correction is exactly the identity,
# which `noop_check` asserts rather than assumes.  Same record file as `freq`,
# so this is a control on the feature scale and on nothing else.
FREQDX_DELTAS = (1.0 / 12.0, 0.25, 0.5)      # delta = 1 is the arm itself

# `anchor_v4`'s own split, read off the annual case and expressed on the year
# axis so every arm holds out the same period.  T = 40, trainval_frac = 0.7 and
# val_frac = 0.2 give train_end = 21, val_end = 27 -> these two fractions.
TRAIN_END_FRAC = 21.0 / 39.0
VAL_END_FRAC = 27.0 / 39.0

# Stage B curriculum expressed in years rather than rows.
CURRICULUM_YEARS = ((800, 8.0), (800, 16.0), (400, 22.0))

# The grid every arm is scored on, so curves at different delta are comparable.
COMPARE_STEP = 1.0


# ---------------------------------------------------------------------------
# the one patch this module owns (CLAUDE.md rule 1: import-time, in the lab)
# ---------------------------------------------------------------------------
_ORIG_PREPROCESS = None
_DX_DELTA: float | None = None          # None => the wrapper is the identity


def _orders_of(feature_orders):
    fo = (feature_orders if isinstance(feature_orders, (list, tuple, np.ndarray))
          else [feature_orders])
    return tuple(sorted(set(int(o) for o in fo)))


def install_exog(v5mod=None):
    """Rebind `v5.preprocess_exog` so the order-o block can be put on a
    delta-invariant footing.

    Idempotent, and **disarmed by default**: with `_DX_DELTA` unset the
    wrapper forwards the original array unchanged, so an installed-but-
    disarmed module is behaviourally identical to an uninstalled one.  This is
    the same contract `zinc_synth_lab.install` keeps for the loader, and
    `noop_check` verifies it array-by-array rather than asserting it.
    """
    global _ORIG_PREPROCESS
    if v5mod is None:
        import zinc_colloc_v5 as v5mod
    if _ORIG_PREPROCESS is not None:
        return PATCHES
    _ORIG_PREPROCESS = v5mod.preprocess_exog

    def _dispatch(years_all, exog_values, years_train, **kw):
        X = _ORIG_PREPROCESS(years_all, exog_values, years_train, **kw)
        if _DX_DELTA is None or float(_DX_DELTA) == 1.0:
            return X
        orders = _orders_of(kw.get("feature_orders", (0,)))
        if len(orders) == 0 or X.shape[1] % len(orders) != 0:
            return X
        n0 = X.shape[1] // len(orders)
        X = np.array(X, dtype=float, copy=True)
        for i, o in enumerate(orders):
            if o == 0:
                continue
            X[:, i * n0:(i + 1) * n0] /= float(_DX_DELTA) ** o
        return X

    v5mod.preprocess_exog = _dispatch
    PATCHES.append("zinc_colloc_v5.preprocess_exog -> delta-invariant "
                   "order-o exogenous block (disarmed by default)")
    return PATCHES


def arm_exog(delta):
    """Arm the order-o rescaling at width `delta` for the next fit."""
    global _DX_DELTA
    _DX_DELTA = float(delta)
    return _DX_DELTA


def disarm_exog():
    global _DX_DELTA
    _DX_DELTA = None


def _exog_delta_for(arm):
    """The rescaling width an arm asks for; `None` for every arm but `freqdx`.

    Kept as one function so there is a single place where an arm can turn the
    patch on, and so `--check` can print it for every arm in the design.
    """
    return float(arm["delta"]) if arm.get("kind") == "freqdx" else None


def _rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024.0 ** 2) if sys.platform == "darwin" else r / 1024.0


# ---------------------------------------------------------------------------
# arm naming
# ---------------------------------------------------------------------------
def arm_tag(kind, delta, span):
    """Stable tag.  `_dtag` is `zinc_synth_lab`'s so the two agree at delta=1."""
    d = S._dtag(float(delta))
    if kind in ("freq", "coarse", "season", "freqsm", "freqfw", "freqdx"):
        return f"wp11a_{kind}_{d}"
    return f"wp11a_{kind}_{d}_s{int(round(span))}y"


def design():
    """Every (twin, kind, delta, span) this package fits, in a fixed order."""
    rows = []
    for dl in DELTAS_FREQ:
        rows.append(dict(kind="freq", twin="base", delta=float(dl),
                         span=FULL_SPAN))
    for dl in DELTAS_COARSE:
        rows.append(dict(kind="coarse", twin="base", delta=float(dl),
                         span=FULL_SPAN))
    for sp in SPANS:
        if abs(sp - FULL_SPAN) < 1e-9:
            continue                      # identical to freq delta = 1
        rows.append(dict(kind="span", twin="base", delta=1.0, span=float(sp)))
    for dl, sp in MATCHED_N:
        rows.append(dict(kind="matchedN", twin="base", delta=float(dl),
                         span=float(sp)))
    for dl in SEASON_DELTAS:
        rows.append(dict(kind="season", twin="season", delta=float(dl),
                         span=FULL_SPAN))
    for dl in FREQSM_DELTAS:
        rows.append(dict(kind="freqsm", twin="base", delta=float(dl),
                         span=FULL_SPAN))
    for dl in FREQFW_DELTAS:
        rows.append(dict(kind="freqfw", twin="base", delta=float(dl),
                         span=FULL_SPAN))
    for dl in FREQDX_DELTAS:
        rows.append(dict(kind="freqdx", twin="base", delta=float(dl),
                         span=FULL_SPAN))
    for r in rows:
        r["tag"] = arm_tag(r["kind"], r["delta"], r["span"])
        r["n_obs"] = int(round(r["span"] / r["delta"])) + 1
    return rows


def arm_of(tag):
    for r in design():
        if r["tag"] == tag:
            return r
    raise KeyError(f"unknown arm tag {tag!r}; known: "
                   f"{[r['tag'] for r in design()]}")


# ---------------------------------------------------------------------------
# split and curriculum
# ---------------------------------------------------------------------------
def split_for(years):
    """`split_indices` pinned on the year axis, not the row count.

    Returns the dict `train_model` takes plus the boundary years, so a caller
    can assert two arms hold out the same period.  At the annual full span
    this reproduces `trainval_frac=0.7, val_frac=0.2` exactly.
    """
    years = np.asarray(years, float).ravel()
    t0, span = float(years[0]), float(years[-1] - years[0])
    y_tr = t0 + TRAIN_END_FRAC * span
    y_val = t0 + VAL_END_FRAC * span
    i_tr = int(np.argmin(np.abs(years - y_tr)))
    i_val = int(np.argmin(np.abs(years - y_val)))
    i_tr = max(1, min(i_tr, years.size - 3))
    i_val = max(i_tr + 1, min(i_val, years.size - 2))
    return dict(split_indices=dict(train_end=i_tr, val_end=i_val,
                                   test_start=i_val + 1),
                train_end_year=float(years[i_tr]),
                val_end_year=float(years[i_val]),
                test_start_year=float(years[i_val + 1]))


def curriculum_for(delta):
    """Stage B windows rescaled so the physical horizon is delta-independent."""
    return [[int(n), max(2, int(round(w / float(delta))))]
            for (n, w) in CURRICULUM_YEARS]


def config_for(arm, cfg=None):
    """The `train_model` config for one arm: anchor_v4 plus split and windows."""
    cfg = dict(cfg or lab.load_anchor_config())
    ds = S.load_dataset(dataset_path(arm["tag"]))
    sp = split_for(ds["years"])
    cfg["split_indices"] = sp["split_indices"]
    cfg["stageB_curriculum"] = curriculum_for(
        1.0 if arm["kind"] == "coarse" else arm["delta"])
    if arm["kind"] == "freqfw":
        n_tv = int(sp["split_indices"]["val_end"]) + 1
        cfg["stageB_curriculum"] = cfg["stageB_curriculum"][:-1] + [
            [int(cfg["stageB_curriculum"][-1][0]), n_tv]]
    if arm["kind"] == "freqsm":
        f = 1.0 / float(arm["delta"]) ** 2
        for k in SMOOTH_KEYS:
            if k in cfg:
                cfg[k] = float(cfg[k]) * f
    cfg.pop("trainval_frac", None)
    cfg.pop("val_frac", None)
    return cfg, sp


# ---------------------------------------------------------------------------
# dataset construction
# ---------------------------------------------------------------------------
def dataset_path(tag, out_dir=SYNTH_DIR):
    """Where an arm's record lives.

    `freqsm`, `freqfw` and `freqdx` have no record of their own: each is the
    `freq` arm of the same width refitted under one changed setting, and
    pointing both at the same file is what makes them controls rather than
    second experiments.
    """
    for k in ("freqsm", "freqfw", "freqdx"):
        if tag.startswith(f"wp11a_{k}_"):
            tag = tag.replace(f"wp11a_{k}_", "wp11a_freq_")
    return os.path.join(out_dir, f"{tag}.npz")


def _truncate(ds, n):
    """First `n` observation rows of a `load_zinc_data`-shaped dict.

    Flow rows are interval quantities and there is one fewer of them
    (`sample` stores row i as the interval ending at `years[i]`, dropping the
    unused row 0), so the flow table keeps `n-1` rows.  `exog_*` are rebuilt by
    the caller, which is why they are not touched here.
    """
    out = dict(ds)
    for k in ("years", "stocks_obs", "cp_obs", "alpha_obs", "tau_sup_obs"):
        out[k] = np.asarray(ds[k])[:n].copy()
    out["flows_obs"] = np.asarray(ds["flows_obs"])[:max(n - 1, 0)].copy()
    return out


def build_arm(arm, ctx, dense_by_twin, sigma, *, out_dir=SYNTH_DIR,
              verbose=True):
    """Generate and persist one arm's dataset, truth and metadata."""
    kind, delta, span = arm["kind"], arm["delta"], arm["span"]
    twin = arm.get("twin", TWIN_ARM)
    dense = dense_by_twin[twin]
    tag = arm["tag"]

    if kind == "coarse":
        # WP-6b's operator, applied to the twin's *annual* record.  Built here
        # rather than re-sampled so the gate compares data and not operators.
        base = arm_of(arm_tag("freq", 1.0, FULL_SPAN))
        ann = S.load_dataset(dataset_path(base["tag"], out_dir))
        ann_truth = S.load_truth(dataset_path(base["tag"], out_dir))
        ds = C.coarsen(dict(ann), int(round(delta)), "all",
                       cfg=lab.load_anchor_config(), verbose=False)
        truth = {k: v for k, v in ann_truth.items() if k != "meta"}
        n_obs = int(np.asarray(ds["years"]).size)
        extra = dict(operator="zinc_coarse_lab.coarsen (WP-6b)",
                     annual_source=base["tag"],
                     n_retained=int(arm["n_obs"]))
    else:
        samp = S.sample(dense, delta, ctx, noise_scale=NOISE_SCALE,
                        rng_seed=NOISE_RNG_SEED, sigma=sigma)
        ds = S.build_dataset(ctx, samp, delta)
        tw = S.truth_coefficients(ctx, samp["years"], twin,
                                  S_path=samp["stocks_clean"])
        bias = S.alpha_target_bias(ctx, dense, delta, arm_name=twin)
        truth = dict(
            alphas_point=tw["alphas"], tau_sup_point=tw["tau_sup"],
            f_cohort_point=tw["f_cohort"],
            alphas_window_unweighted=bias["a_unweighted"],
            alphas_window_weighted=bias["a_weighted"],
            alphas_window_obs=bias["a_obs"],
            corr_FS=bias["corr_FS"], cov_term=bias["cov_term"],
            stocks_clean=samp["stocks_clean"],
            F_int_clean=samp["F_int_clean"],
            window_years=bias["years"],
            alpha_names=np.asarray(ALPHA_NAMES, dtype=object),
            tau_sup_names=np.asarray(TAU_SUP_NAMES, dtype=object))

        n_obs = int(round(span / delta)) + 1
        if n_obs < np.asarray(ds["years"]).size:
            ds = _truncate(ds, n_obs)
            exog, t_full, v_full = S._exog_on(ctx, ds["years"], delta)
            ds["exog_values"] = exog
            ds["exog_times"] = ds["years"].copy()
            ds["exog_times_full"] = t_full
            ds["exog_values_full"] = v_full
            for k in ("alphas_point", "tau_sup_point", "f_cohort_point",
                      "stocks_clean"):
                truth[k] = np.asarray(truth[k])[:n_obs]
            for k in ("alphas_window_unweighted", "alphas_window_weighted",
                      "alphas_window_obs", "corr_FS", "cov_term",
                      "window_years", "F_int_clean"):
                truth[k] = np.asarray(truth[k])[:max(n_obs - 1, 0)]
        extra = dict(operator="zinc_synth_lab.sample (re-aggregation)")

    sp = split_for(ds["years"])
    meta = dict(arm=twin, twin=twin, kind=kind, delta=float(delta),
                span=float(span),
                tag=tag, noise_scale=NOISE_SCALE, rng_seed=NOISE_RNG_SEED,
                n_sub=S.N_SUB, T=n_obs,
                years=(float(ds["years"][0]), float(ds["years"][-1])),
                curriculum=curriculum_for(1.0 if kind == "coarse" else delta),
                **{k: v for k, v in sp.items() if k != "split_indices"},
                split_indices=sp["split_indices"], **extra)
    p = S.save_dataset(dataset_path(tag, out_dir), ds, truth=truth, meta=meta)
    if verbose:
        print(f"[{tag}] T={n_obs:4d}  delta={delta:.4f}  span={span:g}  "
              f"train->{sp['train_end_year']:.2f} val->{sp['val_end_year']:.2f}"
              f"  -> {os.path.basename(p)}", flush=True)
    return p, meta


def generate(out_dir=SYNTH_DIR, verbose=True, cfg=None, tags=None):
    """Every arm, in dependency order (`coarse` reads the annual `freq` arm).

    `tags` restricts generation to a subset, which matters more than it looks:
    rewriting a dataset that a running fit is reading would corrupt it, so a
    later addition to the design is generated on its own and the manifest is
    *merged* rather than overwritten.
    """
    os.makedirs(out_dir, exist_ok=True)
    ctx = S.driver_context(cfg)
    sigma = S.calibrate_noise()
    rows = design()
    if tags is not None:
        want = set(tags)
        rows = [r for r in rows if r["tag"] in want]
        missing = want - {r["tag"] for r in rows}
        if missing:
            raise KeyError(f"unknown arm tags {sorted(missing)}")
    dense_by_twin = {}
    for tw in sorted({r.get("twin", TWIN_ARM) for r in rows}):
        t0 = time.time()
        dense_by_twin[tw] = S.dense_solve(ctx, tw)
        if verbose:
            d = dense_by_twin[tw]
            print(f"[dense/{tw}] {time.time()-t0:.1f}s  {d['t'].size} nodes  "
                  f"{d['t'][0]:.0f}-{d['t'][-1]:.0f}", flush=True)
    rows = [r for r in rows                              # share `freq`'s file
            if r["kind"] not in ("freqsm", "freqfw")]
    rows.sort(key=lambda r: (r["kind"] == "coarse",))    # coarse last
    report = {"arms": {}, "n_sub": S.N_SUB, "twin_arms": sorted(dense_by_twin),
              "noise_scale": NOISE_SCALE, "rng_seed": NOISE_RNG_SEED,
              "feature_scale": None}
    for r in rows:
        _p, meta = build_arm(r, ctx, dense_by_twin, sigma, out_dir=out_dir,
                             verbose=verbose)
        report["arms"][r["tag"]] = meta
    report["feature_scale"] = feature_scale_report(out_dir=out_dir, cfg=cfg)
    if os.path.exists(MANIFEST):
        try:
            with open(MANIFEST) as fh:
                prev = json.load(fh)
            merged = dict(prev.get("arms", {}))
            merged.update(report["arms"])
            report["arms"] = merged
        except (ValueError, OSError):
            pass
    with open(MANIFEST, "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    if verbose:
        print(f"[generate] {len(rows)} arms -> {MANIFEST}  "
              f"rss {_rss_mb():.0f} MB", flush=True)
    return report


# ---------------------------------------------------------------------------
# the measured confound
# ---------------------------------------------------------------------------
def feature_scale_report(out_dir=SYNTH_DIR, cfg=None):
    """Per-order SD of the exogenous feature block, per arm.

    The order-1 features are per-window differences, so their scale falls with
    delta and `exog_std`'s clamp at 1.0 does not restore it.  Measured here so
    the findings note can quote the size of the effect instead of asserting it
    is small.

    Each arm is also reported under the `freqdx` correction (`order{o}_sd_dx`,
    the block divided by `delta**o`), so the size of the fix is visible before
    any fit is run and the `freqdx` arms can be read as a control rather than
    taken on trust.
    """
    import zinc_colloc_v5 as v5
    cfg = dict(cfg or lab.load_anchor_config())
    orders = tuple(sorted(set(int(o) for o in cfg.get("exog_feature_orders", (0,)))))
    out = {}
    for r in design():
        p = dataset_path(r["tag"], out_dir)
        if not os.path.exists(p):
            continue
        ds = S.load_dataset(p)
        sp = split_for(ds["years"])
        cut = sp["split_indices"]["train_end"] + 1
        X = v5.preprocess_exog(
            ds["years"], ds["exog_values"], ds["years"][:cut],
            do_log1p=cfg.get("exog_log1p", False),
            do_detrend=cfg.get("exog_detrend", False),
            feature_orders=orders,
            years_source=ds["exog_times_full"],
            exog_values_source=ds["exog_values_full"])
        n0 = X.shape[1] // len(orders)
        blocks = {}
        for i, o in enumerate(orders):
            B = X[:cut, i * n0:(i + 1) * n0]
            blocks[f"order{o}_sd"] = float(np.mean(np.std(B, axis=0)))
            blocks[f"order{o}_sd_after_clamp"] = float(np.mean(
                np.std(B, axis=0) / np.maximum(np.std(B, axis=0), 1.0)))
            # the ROW spacing, not the arm's nominal delta: the coarse arms
            # keep the annual grid and mask their targets, so their rows are
            # one year apart and their order-1 block is already annual.
            h = float(np.median(np.diff(np.asarray(ds["years"], float))))
            blocks[f"order{o}_sd_dx"] = float(np.mean(np.std(
                B / h ** o, axis=0)))
        out[r["tag"]] = dict(delta=r["delta"], span=r["span"],
                             n_features=int(X.shape[1]), **blocks)
    return out


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------
def dump_fit(fit, path, arm, truth, annual):
    """Persist what WP-11a scores.

    Three references are carried side by side because the package asks three
    different questions of the same fit: the **truth** (understanding value,
    available only on a twin), the arm's **own record** (what the estimator was
    actually shown), and the **annual** record (the currency WP-6b's real curve
    is denominated in, and therefore the one the WP-11b gate needs).
    """
    import zinc_colloc_v5 as v5

    masks = fit._split_indices_in_all()
    prB = fit.predictions("B")
    years = np.asarray(fit.data_all["years"], float).ravel()
    arrays = dict(
        years=years,
        alpha_names=np.array(v5.ALPHA_NAMES, dtype=object),
        tau_names=np.array(v5.TAU_SUP_NAMES, dtype=object),
        stock_names=np.array(STOCK_NAMES, dtype=object),
        flow_names=np.array(list(annual["flow_obs_names"]), dtype=object),
        # `F_int` and the truth accumulator are both in the model's internal
        # flow order, which is longer than the observable table -- carried by
        # name so the scorer never has to align them by position.
        flow_names_pred=np.array(list(v5.FLOW_NAMES), dtype=object),
        # what the estimator was shown
        alpha_obs_arm=np.asarray(fit.data_all["alpha_obs"], float),
        tau_obs_arm=np.asarray(fit.data_all["tau_sup_obs"], float),
        S_obs_arm=np.asarray(fit.data_all["stocks_obs"], float),
        F_obs_arm=np.asarray(fit.data_all["flows_obs"], float),
        # the analytic truth, on the arm's own nodes
        alpha_true=np.asarray(truth["alphas_point"], float),
        tau_true=np.asarray(truth["tau_sup_point"], float),
        S_true=np.asarray(truth["stocks_clean"], float),
        F_true=np.asarray(truth["F_int_clean"], float),
        # the annual record and the annual truth, for the cross-arm grid
        years_annual=np.asarray(annual["years"], float),
        alpha_obs_annual=np.asarray(annual["alpha_obs"], float),
        alpha_true_annual=np.asarray(annual["alphas_point"], float),
        tau_true_annual=np.asarray(annual["tau_sup_point"], float),
        S_true_annual=np.asarray(annual["stocks_clean"], float),
        F_true_annual=np.asarray(annual["F_int_clean"], float),
        # predictions
        alpha_pred_B_at_obs=np.asarray(prB["alphas_at_obs"], float),
        # the eight supervised tau slots in `TAU_SUP_NAMES` order: the four
        # binary taus, then the two free manufacturing-simplex components of
        # each of frac_fu / frac_eu (v5's third component is 1 - the others).
        tau_pred_B_at_obs=np.concatenate(
            [np.asarray(prB["taus_at_obs"], float),
             np.asarray(prB["frac_fu_at_obs"], float)[:, :2],
             np.asarray(prB["frac_eu_at_obs"], float)[:, :2]], axis=1),
        S_pred_B=np.asarray(prB["S_pred"], float),
        F_pred_B=np.asarray(prB["F_int"], float),
        mask_train=np.asarray(masks["train"], bool),
        mask_val=np.asarray(masks["val"], bool),
        mask_test=np.asarray(masks["test"], bool),
        meta_json=np.asarray(json.dumps(
            dict(arm, path=os.path.basename(path))), dtype=object),
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def _annual_reference(twin=TWIN_ARM, out_dir=SYNTH_DIR):
    """The delta = 1 full-span dataset and its truth, on the comparison grid.

    Per twin: a `season` arm has to be scored against the season truth and the
    season annual record, or the curve would be measuring the gap between two
    different systems.
    """
    p = dataset_path(arm_tag("freq" if twin == TWIN_ARM else "season",
                             1.0, FULL_SPAN), out_dir)
    ds = S.load_dataset(p)
    tr = S.load_truth(p)
    return dict(years=ds["years"], alpha_obs=ds["alpha_obs"],
                flow_obs_names=ds["flow_obs_names"],
                alphas_point=tr["alphas_point"],
                tau_sup_point=tr["tau_sup_point"],
                stocks_clean=tr["stocks_clean"],
                F_int_clean=tr["F_int_clean"])


def dump_spectrum(fit, cfg, path):
    """WP-4c's identifiability spectrum, recomputed on one arm's own fit.

    The spec asks for "the identifiability spectrum from WP-4c recomputed at
    each frequency", and the only honest way to get it is from the fit that
    frequency produced -- so it is computed here, inside the fitting process,
    from the live `fit` object rather than reconstructed later from weights.
    `zinc_fisher_lab` supplies every piece; nothing is re-implemented.

    Cost is `n_residuals` reverse-mode passes through the ODE, so it scales
    with the observation count and is run for a subset of seeds only.
    """
    import jax
    import jax.flatten_util
    import zinc_fisher_lab as FL

    params = fit._params_for("B")
    theta, _un = jax.flatten_util.ravel_pytree(params)
    J, Jp, G, meta = FL.jacobians(fit, cfg, theta)
    dead = FL.structural_zero_mask(fit)
    sp = FL.spectrum(J, Jp, dead=dead)
    lam_ref = FL.LAMBDA_REF * float(sp["eig"][0])
    p_eff = FL.p_effective(sp, lam_ref)
    kap2, rss, dof = FL.kappa_squared(meta["r_data"], sp["n_res"], p_eff)
    marg = FL.laplace_marginals(sp, G, kap2, lambdas=[lam_ref], dead=dead)
    sd = np.asarray(marg["sd_by_lambda"][lam_ref], float)     # (T, 17)
    # One number per alpha channel: the median over the fitted rows, which is
    # the same reduction WP-11b's Fisher arm uses.
    n_tv = int(np.asarray(fit.data_trainval["years"]).shape[0])
    sd_alpha = np.nanmedian(sd[:n_tv, list(FL.ALPHA_IDX)], axis=0)
    np.savez_compressed(
        path, eig=sp["eig"], sv=sp["sv"], rank=np.asarray(sp["rank"]),
        n_res=np.asarray(sp["n_res"]), n_par=np.asarray(sp["n_par"]),
        n_dead=np.asarray(int(dead.sum())), p_eff=np.asarray(p_eff),
        kappa2=np.asarray(kap2), rss=np.asarray(rss), dof=np.asarray(dof),
        lam_ref=np.asarray(lam_ref), sd_alpha=np.asarray(sd_alpha),
        sd_all=sd, coef_names=np.array(FL.COEF_NAMES, dtype=object))
    del J, Jp, G
    return path


def fit_arm(seed, tag, out_dir=FIT_DIR, cfg=None, verbose=False,
            skip_existing=False, spectrum=False):
    """One full fit of one arm.  Arms the twin, fits, dumps, disarms."""
    import jax
    import zinc_colloc_v5 as v5

    lab.integrity_check()
    S.install(v5)
    install_exog(v5)
    arm = arm_of(tag)
    path = os.path.join(out_dir, f"{tag}_seed{seed}.npz")
    if skip_existing and os.path.exists(path):
        print(f"[{tag} seed {seed}] exists, skipped", flush=True)
        return path

    cfg, sp = config_for(arm, cfg)
    cfg["verbose"] = bool(verbose)
    dp = dataset_path(tag)
    ds = S.load_dataset(dp)
    truth = S.load_truth(dp)
    annual = _annual_reference(arm.get("twin", TWIN_ARM))

    S.arm(ds)
    dx = _exog_delta_for(arm)
    if dx is not None:
        arm_exog(dx)
    t0 = time.time()
    try:
        fit = v5.run(f"wp11a_{tag}_s{seed}", **dict(cfg, seed=int(seed)))
        dump_fit(fit, path, dict(arm, **{k: v for k, v in sp.items()
                                         if k != "split_indices"}),
                 truth, annual)
        if spectrum:
            ts = time.time()
            dump_spectrum(fit, cfg, path.replace(".npz", "_spec.npz"))
            print(f"[{tag} seed {seed}] spectrum {time.time()-ts:.0f}s",
                  flush=True)
    finally:
        S.disarm()
        disarm_exog()
    dt = time.time() - t0
    del fit
    jax.clear_caches()                       # CLAUDE.md rule 6
    print(f"[{tag} seed {seed}] {dt/60:.1f} min  rss={_rss_mb():.0f} MB "
          f"-> {os.path.basename(path)}", flush=True)
    return path


# ---------------------------------------------------------------------------
# --check / --noop
# ---------------------------------------------------------------------------
def noop_check(verbose=True):
    """Two identities that must hold before any fit is trusted.

    1. **The split reduces to the published one.**  `split_for` on the annual
       full-span grid must return exactly the indices `trainval_frac=0.7,
       val_frac=0.2` produce, so the delta = 1 arm is the published design and
       every other arm is that design transported to another grid.
    2. **The re-sampler is a refinement.**  The delta = 1 record must be
       recoverable from the delta = 1/2 one: stocks at even nodes must agree
       to 0, and consecutive pairs of flow integrals must sum to the annual
       integral.  This is the property that makes "more observations of the
       same system" the right description of the frequency arm.
    3. **The exogenous patch is disarmed by default and inert at delta = 1.**
       An installed-but-disarmed `preprocess_exog` must return the original
       array bit for bit, or the 160 fits this package already holds are not
       comparable with anything fitted after the patch landed; and armed at
       delta = 1 it must also be the identity, since dividing by `1.0**o` is
       what makes `freqdx` a control against the annual arm rather than a
       fourth experiment.
    """
    rows = []
    yr = np.arange(1980.0, 2020.0)
    sp = split_for(yr)
    want = dict(train_end=21, val_end=27, test_start=28)
    ok_split = sp["split_indices"] == want
    rows.append(dict(check="split_reduces_to_published", ok=bool(ok_split),
                     got=json.dumps(sp["split_indices"]),
                     want=json.dumps(want), value=np.nan))

    p1 = dataset_path(arm_tag("freq", 1.0, FULL_SPAN))
    ph = dataset_path(arm_tag("freq", 0.5, FULL_SPAN))
    if os.path.exists(p1) and os.path.exists(ph):
        d1, dh = S.load_dataset(p1), S.load_dataset(ph)
        t1 = S.load_truth(p1)
        th = S.load_truth(ph)
        dS = float(np.max(np.abs(t1["stocks_clean"]
                                 - th["stocks_clean"][::2])))
        Fh = np.asarray(th["F_int_clean"])
        dF = float(np.max(np.abs(np.asarray(t1["F_int_clean"])
                                 - (Fh[0::2] + Fh[1::2]))))
        dy = float(np.max(np.abs(d1["years"] - dh["years"][::2])))
        rows += [dict(check="refinement_stocks_clean", ok=bool(dS < 1e-8),
                      got="", want="0", value=dS),
                 dict(check="refinement_flow_integrals", ok=bool(dF < 1e-6),
                      got="", want="0", value=dF),
                 dict(check="refinement_year_nodes", ok=bool(dy < 1e-9),
                      got="", want="0", value=dy)]

    # 3. the delta = 1 arm *is* WP-3's annual twin.  Same operator, same noise
    #    seed, same sigma -- so it must be the same file, byte for byte, and
    #    if it is not then this package and WP-3 are not looking at the same
    #    system and nothing downstream is comparable.
    p3 = os.path.join(SYNTH_DIR, "base_d1y_noisy.npz")
    if os.path.exists(p1) and os.path.exists(p3):
        a, b = S.load_dataset(p3), S.load_dataset(p1)
        worst, where = 0.0, ""
        for k in ("years", "stocks_obs", "cp_obs", "alpha_obs", "tau_sup_obs",
                  "flows_obs", "exog_values", "exog_values_full"):
            x, y = np.asarray(a[k], float), np.asarray(b[k], float)
            v = (np.inf if x.shape != y.shape
                 else float(np.nanmax(np.abs(x - y))))
            if v > worst:
                worst, where = v, k
        rows.append(dict(check="annual_arm_is_wp3_twin", ok=bool(worst == 0.0),
                         got=where, want="0", value=worst))

    pc = dataset_path(arm_tag("coarse", 2.0, FULL_SPAN))
    if os.path.exists(p1) and os.path.exists(pc):
        d1, dc = S.load_dataset(p1), S.load_dataset(pc)
        dT = int(dc["years"].size) - int(d1["years"].size)
        rows.append(dict(check="coarse_keeps_annual_grid", ok=(dT == 0),
                         got=str(dc["years"].size), want=str(d1["years"].size),
                         value=float(dT)))
        finite = np.isfinite(dc["alpha_obs"]).sum(axis=0)
        rows.append(dict(check="coarse_masks_alpha_targets",
                         ok=bool(np.all(finite < d1["years"].size)),
                         got=",".join(str(int(x)) for x in finite),
                         want=f"<{d1['years'].size}", value=float(finite.max())))

    # 5. the exogenous patch: disarmed, and armed at delta = 1, are both the
    #    identity -- checked against the unpatched function on a real record.
    if os.path.exists(p1):
        import zinc_colloc_v5 as v5
        install_exog(v5)
        cfgn = lab.load_anchor_config()
        d1 = S.load_dataset(p1)
        spn = split_for(d1["years"])
        cutn = spn["split_indices"]["train_end"] + 1
        kw = dict(do_log1p=cfgn.get("exog_log1p", False),
                  do_detrend=cfgn.get("exog_detrend", False),
                  feature_orders=_orders_of(
                      cfgn.get("exog_feature_orders", (0,))),
                  years_source=d1["exog_times_full"],
                  exog_values_source=d1["exog_values_full"])
        base = _ORIG_PREPROCESS(d1["years"], d1["exog_values"],
                                d1["years"][:cutn], **kw)
        disarm_exog()
        off = v5.preprocess_exog(d1["years"], d1["exog_values"],
                                 d1["years"][:cutn], **kw)
        arm_exog(1.0)
        one = v5.preprocess_exog(d1["years"], d1["exog_values"],
                                 d1["years"][:cutn], **kw)
        disarm_exog()
        rows += [dict(check="exog_patch_disarmed_is_identity",
                      ok=bool(np.array_equal(base, off)), got="", want="0",
                      value=float(np.max(np.abs(base - off)))),
                 dict(check="exog_patch_at_delta1_is_identity",
                      ok=bool(np.array_equal(base, one)), got="", want="0",
                      value=float(np.max(np.abs(base - one))))]
        # and armed at delta < 1 it restores the annual-equivalent scale: the
        # monthly order-1 block, divided by delta, must carry the SD the
        # annual block does.  The drivers are annually interpolated, so the
        # difference quotient of a piecewise-linear interpolant is the annual
        # slope and the two SDs agree to interpolation error, not exactly.
        pm = dataset_path(arm_tag("freq", 1.0 / 12.0, FULL_SPAN))
        orders = _orders_of(cfgn.get("exog_feature_orders", (0,)))
        if os.path.exists(pm) and 1 in orders:
            dm = S.load_dataset(pm)
            spm = split_for(dm["years"])
            cutm = spm["split_indices"]["train_end"] + 1
            arm_exog(1.0 / 12.0)
            Xm = v5.preprocess_exog(
                dm["years"], dm["exog_values"], dm["years"][:cutm],
                do_log1p=kw["do_log1p"], do_detrend=kw["do_detrend"],
                feature_orders=orders,
                years_source=dm["exog_times_full"],
                exog_values_source=dm["exog_values_full"])
            disarm_exog()
            n0 = base.shape[1] // len(orders)
            i1 = orders.index(1)
            sd_a = float(np.mean(np.std(
                base[:cutn, i1 * n0:(i1 + 1) * n0], axis=0)))
            sd_m = float(np.mean(np.std(
                Xm[:cutm, i1 * n0:(i1 + 1) * n0], axis=0)))
            ratio = sd_m / sd_a if sd_a > 0 else np.inf
            rows.append(dict(check="exog_patch_restores_annual_scale",
                             ok=bool(0.8 < ratio < 1.25),
                             got=f"{sd_m:.4f} vs {sd_a:.4f}",
                             want="ratio in [0.8, 1.25]", value=ratio))

    ok = all(r["ok"] for r in rows)
    if verbose:
        print("=" * 74)
        print("zinc_freq_lab --noop")
        print("=" * 74)
        for r in rows:
            flag = "OK  " if r["ok"] else "FAIL"
            v = "" if not np.isfinite(r["value"]) else f"  {r['value']:.3e}"
            print(f"  [{flag}] {r['check']:32s}{v}"
                  + (f"   got={r['got']} want={r['want']}"
                     if not r["ok"] or r["got"] else ""))
        print("=" * 74)
    return rows, ok


def check(verbose=True, cfg=None):
    """Resolved drivers, input_dim, the design, the split, the curriculum."""
    import zinc_colloc_v5 as v5
    S.install(v5)
    S.disarm()
    install_exog(v5)
    disarm_exog()
    info = lab.check(verbose=verbose)
    rows = design()
    info = dict(info, patches=list(PATCHES),
                synth_patches=list(S.PATCHES),
                design=rows, n_arms=len(rows))
    if verbose:
        print("=" * 74)
        print("zinc_freq_lab --check")
        print("=" * 74)
        print(f"  patches applied (this module) : {PATCHES}")
        print(f"  ... armed for                 : the {len(FREQDX_DELTAS)} "
              f"`freqdx` arms only; disarmed it is the identity "
              f"(`--noop` asserts it)")
        print(f"  patches reused (synth lab)    : {S.PATCHES}")
        print(f"  twin arm / noise              : {TWIN_ARM} / "
              f"scale {NOISE_SCALE} seed {NOISE_RNG_SEED}")
        print(f"  comparison grid               : {COMPARE_STEP:g} yr, "
              f"1980-2019")
        print("\n  design:")
        print(f"    {'tag':30s} {'twin':7s} {'kind':9s} {'delta':>7s} "
              f"{'span':>6s} {'N':>5s}  {'curriculum (rows)':>22s}  split")
        for r in rows:
            p = dataset_path(r["tag"])
            sp = (split_for(S.load_dataset(p)["years"])
                  if os.path.exists(p) else None)
            cur = curriculum_for(1.0 if r["kind"] == "coarse" else r["delta"])
            spl = ("--" if sp is None else
                   f"{sp['train_end_year']:.2f}/{sp['val_end_year']:.2f}")
            print(f"    {r['tag']:30s} {r.get('twin', TWIN_ARM):7s} "
                  f"{r['kind']:9s} {r['delta']:7.4f} "
                  f"{r['span']:6.0f} {r['n_obs']:5d}  "
                  f"{str([w for _s, w in cur]):>22s}  {spl}"
                  + ("" if os.path.exists(p) else "   [dataset missing]"))
        fs = feature_scale_report(cfg=cfg)
        if fs:
            print("\n  exogenous feature-block SD (train rows), per arm:")
            for tag, b in fs.items():
                keys = [k for k in b if k.endswith("_sd")]
                dxk = [k for k in b if k.endswith("_sd_dx")
                       and not k.startswith("order0")]
                print(f"    {tag:30s} " + "  ".join(
                    f"{k[:-3]}={b[k]:.4f}" for k in keys)
                    + ("   " + "  ".join(f"{k[:-6]}_dx={b[k]:.4f}"
                                         for k in dxk) if dxk else ""))
            print("    (order1 falls with delta — the documented confound; "
                  "exog_std clamps at 1.0 so nothing restores it.")
            print("     order1_sd_dx is the same block under the `freqdx` "
                  "correction, which restores it.)")
        print("=" * 74)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--arms", default="all",
                    help="restrict --generate to a comma-separated tag list")
    ap.add_argument("--noop", action="store_true")
    ap.add_argument("--fit", default="", help="arm tag to fit")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--out", default=FIT_DIR)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    if args.check:
        check(verbose=True)
        return 0
    if args.generate:
        generate(verbose=True,
                 tags=None if args.arms == "all"
                 else [t for t in args.arms.split(",") if t.strip()])
        _rows, ok = noop_check()
        return 0 if ok else 1
    if args.noop:
        _rows, ok = noop_check()
        return 0 if ok else 1
    if args.fit:
        cfg = lab.load_anchor_config()
        for sd in [int(s) for s in args.seeds.split(",") if s.strip()]:
            fit_arm(sd, args.fit, out_dir=args.out, cfg=cfg,
                    verbose=args.verbose, skip_existing=args.skip_existing)
        return 0
    check(verbose=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
