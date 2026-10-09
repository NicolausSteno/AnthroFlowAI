#!/usr/bin/env python3
"""
zinc_scale_lab.py — lab module for WP-11d (scaling, and the crossover N*)
=========================================================================

WP-11d asks whether the claim that this model class "suits an era of abundant
data" is true, by putting four estimators on the *same* records at a range of
sample sizes and asking where their learning curves cross.  The spec names the
four: the UDE, the GAM, a per-flow AR/ARX ensemble, and a constant-TC MFA.
This module supplies the three that do not exist yet as fitted objects, and
makes all four land in one file format so a single scorer reads them.

`run_wp11d.py` drives it.

Nothing here generates data.  Every record is a WP-11a arm, already on disk,
already fitted by the UDE — which is the point: *the same data*, in the spec's
words, is a constraint on this package, and the cheapest way to honour it is
to re-use the files rather than rebuild them.

Pattern (CLAUDE.md rule 1)
--------------------------
`zinc_colloc_v5.py` is imported and never edited, and neither is
`zinc_baseline.py`.  One rebinding is installed here and it exists for a
mechanical reason: `zinc_baseline` does `from zinc_colloc_v5 import
load_zinc_data` at import time (zinc_baseline.py:186), so it holds its own
reference and `zinc_synth_lab.install()`'s rebinding of the *core's* name does
not reach it.  `install()` therefore points `zinc_baseline.load_zinc_data` at
the very same dispatcher object `zinc_synth_lab` installed on the core, so the
two modules cannot drift apart: there is one dispatcher, one `arm()`, one
`disarm()`.  `--noop` verifies that with nothing armed the rebound name
returns the real record array-for-array.

The four model classes
----------------------
    ude       WP-11a's own fits, read from disk.  8 seeds per arm.  No fit is
              run by this module -- re-fitting them would produce a *different*
              estimator (SCHEMA/COMPUTE_STATUS flag: this model is not
              reproducible across thread counts) and the whole package rests
              on the four classes seeing identical records.

    gam       `zinc_baseline.run_baseline` under `anchor_gam`'s own config,
              with WP-11a's `split_indices` substituted for the row-fraction
              split.  Penalised B-spline / ridge Stage A, then the same Stage B
              through the same ODE.

    constTC   the same estimator family with `regressor_kinds="const"` and
              **no Stage B**: every learned coefficient is fixed at its
              training-window mean on the link scale, which for the `exp` link
              is the geometric mean of `alpha_obs`.  That is exactly what a
              conventional MFA does -- one time-invariant `A`, read off the
              data as a ratio, integrated forward -- and it reaches it through
              the core's own integrator rather than a re-implementation, so
              the only difference from the GAM arm is the regressor.
              `constTC_B` is the same thing allowed a Stage B refinement; it is
              reported as a generous variant, not as the MFA baseline, because
              an MFA modeller does not have an adjoint.

    arx       WP-1f's per-flow ensemble (`run_wp1f.fit_ensemble`), one
              independent ARX per observed flow, order and driver subset by
              AIC, log space with the lognormal smearing correction, given the
              actual driver values over the test window.  Re-used, not
              rewritten.

Giving the AR ensemble a stock path, an alpha and a tau
-------------------------------------------------------
The AR predicts eighteen flows and nothing else, so three of the four scored
families have to be *derived* from those flows before it can appear on the same
axes as the other three.  Each derivation is the AR modeller's own natural
construction and each is deliberately the generous one:

*   **stocks** — accumulated through WP-1f's `BALANCE` map from the observed
    stock at the trainval boundary, `S_k(i+1) = S_k(i) + (inflows - outflows)`.
    This is the implied stock path WP-1f already reports; anchoring it at the
    observed boundary value rather than at 1980 hands the AR the true level at
    the start of the forecast and asks it only to get the *changes* right.

*   **alpha** — `alpha_hat = F_hat / exposure`, with `exposure` the trapezoid
    of the **observed** parent stock (`zinc_interp_lab._exposure`, itself the
    core's `_build_empirical_alphas` denominator).  Dividing by the observed
    rather than the implied stock is the generous choice and it is WP-5's, so
    the two packages' AR alphas are the same object.

*   **tau** — the eight supervised slots are exact ratios of the flows the AR
    predicts (`compute_flows_from_nn`, v5:1162-1177), so they are read off by
    those identities.  **This is not free of bias and the module measures the
    bias rather than asserting it away**: a ratio of two window integrals is a
    flow-weighted window average, not the point value `tau_sup_point` that the
    scorer compares against, and `tau_identity_check` reports the gap on the
    observed flows themselves.  On the annual base twin it is exactly zero for
    the three learned slots and 3.6-26% for the pinned ones.  The same
    aggregation bias sits in the UDE's own alpha target (WP-3), so this is a
    property of annual reporting shared by every estimator here, but the tau
    column for `arx` should be read with it in mind.

Placement follows the core's own convention (v5:832): a quantity built from
the interval `(t_i, t_{i+1}]` is stored at the **closing endpoint** `t_{i+1}`,
and row 0 is NaN.  That is where `alpha_obs` lives, so the AR's derived
coefficients are aligned exactly like the target the UDE was supervised on.

What varies, and what does not
------------------------------
The arms are WP-11a's, restricted to the `base` twin and to the kinds that
move N:

    resolution   span 39 yr, delta in {10, 5, 2, 1, 1/2, 1/4, 1/12}
                 N = 5, 9, 21, 40, 79, 157, 469
    length       delta = 1 yr, span in {10, 20, 39} yr
                 N = 11, 21, 40
    matchedN     (1/4, 10 yr) and (1/12, 13 yr): N = 41 and 157 reached by
                 resolution instead of by length.  Not a curve -- a control on
                 whether N alone predicts the error, which is the assumption a
                 learning curve makes.

The `coarse` arms are masked, not shortened (WP-6b's operator keeps 40 rows and
blanks the retained-reading gaps), so `n_obs` is the *effective* independent
reading count, `span/delta + 1`, and that -- not the row count -- is the x-axis.
It is the same number WP-11a's own CSVs carry, so the two packages' N agree.

One estimator, one config, one split
------------------------------------
Every baseline arm is given `zinc_freq_lab.split_for`'s `split_indices`, the
same object the UDE arm was fitted under, so all four classes hold out the same
*period* at every N.  `zinc_baseline` supports `split_indices` (zinc_baseline
.py:1799) even though it is absent from `BASELINE_DEFAULT_CONFIG`; `check()`
asserts the substitution reproduces the row-fraction split at the annual full
span rather than trusting it.

`n_cv_splits` is the one estimator setting that cannot be held fixed: the GAM's
Stage A cross-validates over the training rows and the shortest arms have
fewer training rows than folds.  It is clamped to the rows available and the
clamp is *recorded per arm* in the fit-status table, because an arm whose
estimator was weakened is not evidence about sample size.

CLI
---
    python zinc_scale_lab.py --check
    python zinc_scale_lab.py --noop
    python zinc_scale_lab.py --tau-identity
    python zinc_scale_lab.py --fit gam --tags wp11a_freq_1y
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
import zinc_freq_lab as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "analysis", "wp11d")
FIT_DIR = os.path.join(OUT_DIR, "fits")
SYNTH_DIR = F.SYNTH_DIR
UDE_FIT_DIR = F.FIT_DIR
ANCHOR_GAM_CFG = os.path.join(HERE, "anchor_gam", "config_used.json")

PATCHES: list[str] = []
_ORIG_BASELINE_LOADER = None

ALPHA_NAMES = F.ALPHA_NAMES
TAU_SUP_NAMES = F.TAU_SUP_NAMES
STOCK_NAMES = F.STOCK_NAMES

# The four classes the spec names, plus the one generous variant.  `ude` is
# read from WP-11a's dumps; the rest are fitted here.
MODELS = ("ude", "gam", "arx", "constTC")
FITTED_MODELS = ("gam", "constTC", "constTC_B")

# WP-1f's primary specification, by name, so the two packages' AR is one AR.
ARX_SPEC = "ARX-aic"

# Which WP-11a kinds carry N variation, and which curve each belongs to.
KIND_FAMILY = {"freq": "resolution", "coarse": "resolution",
               "span": "length", "matchedN": "matchedN"}

# The GAM's Stage A cross-validates; below this many training rows it cannot.
CV_SPLITS_DEFAULT = 5
CV_MIN_ROWS_PER_FOLD = 3


def _rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024.0 ** 2) if sys.platform == "darwin" else r / 1024.0


# ---------------------------------------------------------------------------
# the design
# ---------------------------------------------------------------------------
def arms():
    """The WP-11a arms this package scores, in a fixed order.

    `n_obs` is WP-11a's own effective reading count (`span/delta + 1`), not the
    row count -- the `coarse` arms keep 40 rows and mask the gaps.
    """
    rows = []
    for r in F.design():
        if r["twin"] != F.TWIN_ARM:
            continue
        fam = KIND_FAMILY.get(r["kind"])
        if fam is None:                       # freqsm / freqfw are controls
            continue
        rows.append(dict(r, family=fam))
    # `freq` at delta = 1 is the shared point of both curves: it is the
    # 39-year annual record, which is simultaneously the finest `length` point
    # and the 1/yr `resolution` point.  Carried once, marked as both.
    for r in rows:
        if r["kind"] == "freq" and abs(r["delta"] - 1.0) < 1e-9:
            r["family"] = "resolution+length"
    rows.sort(key=lambda r: (r["family"], r["n_obs"]))
    return rows


def arm_of(tag):
    for r in arms():
        if r["tag"] == tag:
            return r
    raise KeyError(f"{tag!r} is not a WP-11d arm; known: "
                   f"{[r['tag'] for r in arms()]}")


def tags(family=None):
    return [r["tag"] for r in arms()
            if family is None or family in r["family"].split("+")]


def fit_path(tag, model, seed=0, out_dir=FIT_DIR):
    return os.path.join(out_dir, f"{tag}__{model}_seed{seed}.npz")


def ude_fit_paths(tag, fit_dir=UDE_FIT_DIR):
    import glob
    return sorted(glob.glob(os.path.join(fit_dir, f"{tag}_seed*.npz")))


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------
def install(v5mod=None):
    """Point `zinc_baseline`'s own `load_zinc_data` at the synth dispatcher.

    `zinc_synth_lab.install()` rebinds the name on the *core* module; the
    baseline captured the original at import time and would otherwise keep
    loading the real xlsx while the twin is armed.  Rebinding it to the same
    dispatcher object -- not to a second one -- is what keeps `arm()` and
    `disarm()` meaning one thing.  Idempotent.
    """
    global _ORIG_BASELINE_LOADER
    if v5mod is None:
        import zinc_colloc_v5 as v5mod
    import zinc_baseline as zb

    S.install(v5mod)                          # installs the dispatcher on v5
    if _ORIG_BASELINE_LOADER is not None:
        return PATCHES
    _ORIG_BASELINE_LOADER = zb.load_zinc_data
    zb.load_zinc_data = v5mod.load_zinc_data
    PATCHES.append("zinc_baseline.load_zinc_data -> zinc_synth_lab dispatcher")
    return PATCHES


def arm(dataset):
    return S.arm(dataset)


def disarm():
    return S.disarm()


# ---------------------------------------------------------------------------
# configs
# ---------------------------------------------------------------------------
def _load_gam_config():
    """`anchor_gam`'s config, with the path pinned the way SCHEMA §7 asks."""
    with open(ANCHOR_GAM_CFG) as fh:
        cfg = json.load(fh)
    cfg["xlsx_path"] = os.path.join(HERE, os.path.basename(cfg["xlsx_path"]))
    return cfg


def cv_splits_for(n_train):
    """Folds the training window can actually support, and the clamp flag."""
    k = min(CV_SPLITS_DEFAULT, max(2, int(n_train // CV_MIN_ROWS_PER_FOLD)))
    return int(k), bool(k != CV_SPLITS_DEFAULT)


def config_for(arm_row, model, cfg=None):
    """`run_baseline` config for one (arm, model).  Returns (cfg, split, info).

    The split is WP-11a's, resolved on the year axis, so every model class at
    every N holds out the same period.  `stageB_curriculum` has no analogue
    here: the baseline's Stage B is a plain full-trainval refinement, which is
    one fewer grid-dependent quantity than the UDE arm carries and is noted in
    the findings rather than corrected.
    """
    cfg = dict(cfg or _load_gam_config())
    ds = S.load_dataset(F.dataset_path(arm_row["tag"]))
    sp = F.split_for(ds["years"])
    cfg["split_indices"] = sp["split_indices"]
    cfg.pop("trainval_frac", None)
    cfg.pop("val_frac", None)

    n_train = int(sp["split_indices"]["train_end"]) + 1
    k, clamped = cv_splits_for(n_train)
    cfg["n_cv_splits"] = k

    if model.startswith("constTC"):
        cfg["regressor_kinds"] = "const"
        cfg["gam_smooth_features"] = {}
        cfg["do_stage_B"] = model.endswith("_B")
    elif model == "gam":
        pass                                  # anchor_gam's own settings
    else:
        raise ValueError(f"{model!r} is not fitted by this module "
                         f"(expected one of {FITTED_MODELS})")
    cfg["verbose"] = False
    info = dict(n_train_rows=n_train, n_cv_splits=k, cv_clamped=clamped,
                do_stage_B=bool(cfg.get("do_stage_B", False)),
                regressor_kinds=cfg.get("regressor_kinds"))
    return cfg, sp, info


# ---------------------------------------------------------------------------
# fitting the two estimator-family baselines
# ---------------------------------------------------------------------------
def fit_baseline(tag, model, seed=0, out_dir=FIT_DIR, cfg=None,
                 skip_existing=False, verbose=False):
    """One `zinc_baseline` fit of one arm, dumped in WP-11a's own format.

    `zinc_freq_lab.dump_fit` is called unchanged: `BaselineFitResult` subclasses
    `FitResult` (zinc_baseline.py:1537), so `predictions`, `data_all` and
    `_split_indices_in_all` are the inherited ones and the dump is byte-for-byte
    the same schema the UDE arms wrote.  That is what lets one scorer read all
    four classes.
    """
    import jax
    import zinc_colloc_v5 as v5
    import zinc_baseline as zb

    lab.integrity_check()
    install(v5)
    row = arm_of(tag)
    path = fit_path(tag, model, seed, out_dir)

    cfg, sp, info = config_for(row, model, cfg)
    base = dict(info, tag=tag, model=model, seed=int(seed), n_obs=row["n_obs"],
                delta=row["delta"], span=row["span"], family=row["family"])
    if skip_existing and os.path.exists(path):
        print(f"[{tag} {model}] exists, skipped", flush=True)
        return path, dict(base, status_="skipped")
    cfg["verbose"] = bool(verbose)
    dp = F.dataset_path(tag)
    ds = S.load_dataset(dp)
    truth = S.load_truth(dp)
    annual = F._annual_reference(row.get("twin", F.TWIN_ARM))

    arm(ds)
    t0 = time.time()
    status = dict(info)
    try:
        fit = zb.run_baseline(f"wp11d_{tag}_{model}", **dict(cfg, seed=int(seed)))
        meta = dict(row, model=model,
                    **{k: v for k, v in sp.items() if k != "split_indices"})
        F.dump_fit(fit, path, meta, truth, annual)
        status.update(status_="ok")
        del fit
    except Exception as exc:                  # recorded, never swallowed
        status.update(status_="failed", error=f"{type(exc).__name__}: {exc}")
        path = None
    finally:
        disarm()
    dt = time.time() - t0
    jax.clear_caches()                        # CLAUDE.md rule 6
    status.update(base, wall_s=dt, rss_mb=_rss_mb())
    print(f"[{tag} {model}] {dt/60:.1f} min  rss={_rss_mb():.0f} MB  "
          f"{status['status_']}", flush=True)
    return path, status


# ---------------------------------------------------------------------------
# the per-flow AR/ARX ensemble, and the three families derived from it
# ---------------------------------------------------------------------------
def _flow_index(names):
    return {str(n): i for i, n in enumerate(names)}


def derive_tau(Fmat, names):
    """The eight supervised tau slots from a flow table, `(T-1, 8)`.

    Exactly the identities `compute_flows_from_nn` inverts (v5:1162-1177).
    A ratio of two window integrals, so it is a flow-weighted window average
    of the underlying point coefficient -- see `tau_identity_check`.
    """
    I = _flow_index(names)
    g = lambda n: np.asarray(Fmat[:, I[n]], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        cc = g("concentrate_consumption")
        tau_ref = g("refinery_losses") / cc
        tau_waelz = g("waelz_recycling") / g("waelz_input")
        tau_olds = g("old_scrap_recovery") / g("end_of_life")
        tau_diss = g("dissipative_use") / g("total_products_into_use")
        parent = g("refined_consumption") + g("direct_reuse_recycling")
        fu_new = g("first_use_new_scrap") / parent
        fu_loss = g("first_use_losses") / parent
        fu_out = parent - g("first_use_new_scrap") - g("first_use_losses")
        eu_new = g("end_use_new_scrap") / fu_out
        eu_loss = g("end_use_losses") / fu_out
    out = np.column_stack([tau_ref, tau_waelz, tau_olds, tau_diss,
                           fu_new, fu_loss, eu_new, eu_loss])
    return np.where(np.isfinite(out), out, np.nan)


def derive_stocks(Fmat, names, S_obs, n_train):
    """Implied stock path from the flow balances, anchored at the boundary.

    WP-1f's `BALANCE` map, re-used by import so the two packages accumulate the
    same identities.  Inside the estimation window the AR reproduces the
    observed flows by construction, so the implied path is carried from the
    observed stock at row 0 there and re-anchored at `n_train` for the
    forecast -- which is the generous reading: the AR is handed the true level
    at the start of the forecast and asked only for the changes.
    """
    from run_wp1f import BALANCE

    I = _flow_index(names)
    S_obs = np.asarray(S_obs, float)
    T = S_obs.shape[0]
    out = np.full((T, len(STOCK_NAMES)), np.nan)
    for k, nm in enumerate(STOCK_NAMES):
        inn, outf = BALANCE[nm]
        net = (Fmat[:, [I[x] for x in inn]].sum(1)
               - Fmat[:, [I[x] for x in outf]].sum(1))
        # An AR(p) has no fitted value for its own first p rows, so the
        # accumulation starts at the first row where every contributing flow
        # exists.  Earlier rows are left NaN rather than back-filled with the
        # data, which would credit the estimator with the answer.
        ok = np.isfinite(net)
        if not ok.any():
            continue
        p0 = int(np.argmax(ok))
        path = np.full(T, np.nan)
        path[p0] = S_obs[p0, k]
        path[p0 + 1:] = S_obs[p0, k] + np.cumsum(net[p0:])
        if p0 < n_train < T - 1:              # re-anchor at the boundary
            # flow row `n_train` closes year `n_train + 1`, so the forecast
            # stocks start one row later than the flow rows they come from --
            # WP-1f's own `implied` alignment.
            path[n_train] = S_obs[n_train, k]
            path[n_train + 1:] = S_obs[n_train, k] + np.cumsum(net[n_train:])
        out[:, k] = path
    return out


def derive_alpha(Fmat, names, years, S_obs):
    """`alpha_hat = F_hat / exposure`, `(T, 4)`, row 0 NaN.

    `zinc_interp_lab._exposure` is the core's own trapezoid denominator, so
    the AR modeller's alpha is built the way `alpha_obs` is and only the
    numerator differs.  WP-5's construction, unchanged.
    """
    import zinc_colloc_v5 as v5
    import zinc_interp_lab as IL

    I = _flow_index(names)
    num_names = [IL.ALPHA_FLOW[c] for c in ALPHA_NAMES]
    num = np.asarray(Fmat[:, [I[n] for n in num_names]], float)
    exposure = IL._exposure(np.asarray(years, float), np.asarray(S_obs, float))
    out = np.full((len(years), len(ALPHA_NAMES)), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[1:, :] = np.where(exposure > 1e-12, num / np.maximum(exposure, 1e-12),
                              np.nan)
    _ = v5                                    # imported for the parent-index map
    return out


def _arx_ensemble(data, spec_name=ARX_SPEC):
    """One independent ARX per observed flow: WP-1f's selector, WP-5's path.

    Neither primitive is re-implemented.  The lag order and driver subset come
    from `run_wp1f.fit_arx_one` (AIC, the package's primary specification) and
    the predicted path from `zinc_interp_lab.ar_fitted_path`.

    The composition matters and is the one place this differs from WP-1f's own
    `fit_ensemble`.  That function keeps the *observed* series inside the
    estimation window, because WP-1f only ever scores the forecast; used here
    it would give the AR a zero in-sample error by construction and the
    "understanding value" half of the learning curve would be meaningless.
    `ar_fitted_path` instead returns the **one-step-ahead fitted value** at
    every in-sample row (observed lags, the most accurate in-sample path an AR
    can produce -- WP-5's deliberately generous choice) and the dynamic
    multi-step forecast out of sample, which is what WP-1f scores.  Both halves
    of the curve are then the model's own output.

    Returns `(pred_levels, per_flow_fits, n_train)`.
    """
    from run_wp1f import SPECS, fit_arx_one
    from zinc_interp_lab import ar_fitted_path

    spec = SPECS[spec_name]
    Fmat = np.asarray(data["flows_obs"], float)
    fnames = list(data["flow_names"])
    yrs_f = np.asarray(data["years_flow"], float)
    boundary = float(np.asarray(data["years"], float)[data["mask_val"]][-1])
    is_test = yrs_f > boundary
    n_train = int(np.argmax(is_test)) if is_test.any() else len(yrs_f)
    n_total = len(yrs_f)

    Xex = np.asarray(data["exog_values"], float)[1:, :]   # closing endpoints
    mu, sd = Xex[:n_train].mean(0), Xex[:n_train].std(0)
    Z = (Xex - mu) / np.where(sd > 0, sd, 1.0)

    pred = np.full_like(Fmat, np.nan)
    fits = []
    for j, nm in enumerate(fnames):
        raw = Fmat[:, j]
        y = np.log(np.maximum(raw, 1e-12)) if spec["log"] else raw.copy()
        f = fit_arx_one(y, Z, n_train, max_lag=spec["max_lag"],
                        max_drivers=spec["max_drivers"],
                        all_drivers=spec["all_drivers"])
        f = dict(f, y=y, flow=nm, log=bool(spec["log"]))
        path = ar_fitted_path(f, Z, n_train, n_total)
        if spec["log"]:
            corr = np.exp(0.5 * f["sigma2"]) if spec["smear"] else 1.0
            pred[:, j] = np.exp(path) * corr
        else:
            pred[:, j] = path
        fits.append(f)
    return pred, fits, n_train


def fit_arx(tag, spec=ARX_SPEC, out_dir=FIT_DIR, seed=0, skip_existing=False):
    """The per-flow AR/ARX ensemble on one arm, dumped in WP-11a's format.

    Returns `(path, status)`.  The three derived families (stocks, alpha, tau)
    are built by `derive_*` above; the flow table is the ensemble's own.
    """
    import zinc_colloc_v5 as v5

    lab.integrity_check()
    row = arm_of(tag)
    path = fit_path(tag, "arx", seed, out_dir)
    base = dict(tag=tag, model="arx", seed=int(seed), spec=spec,
                n_obs=row["n_obs"], delta=row["delta"], span=row["span"],
                family=row["family"])
    if skip_existing and os.path.exists(path):
        print(f"[{tag} arx] exists, skipped", flush=True)
        return path, dict(base, status_="skipped")

    dp = F.dataset_path(tag)
    ds = S.load_dataset(dp)
    truth = S.load_truth(dp)
    annual = F._annual_reference(row.get("twin", F.TWIN_ARM))
    sp = F.split_for(ds["years"])
    si = sp["split_indices"]
    years = np.asarray(ds["years"], float)
    T = years.size

    mask_val = np.zeros(T, bool)
    mask_val[int(si["train_end"]) + 1:int(si["val_end"]) + 1] = True
    mask_test = np.zeros(T, bool)
    mask_test[int(si["test_start"]):] = True
    mask_train = np.zeros(T, bool)
    mask_train[:int(si["train_end"]) + 1] = True

    data = dict(years=years, years_flow=years[1:],
                stocks_obs=np.asarray(ds["stocks_obs"], float),
                flows_obs=np.asarray(ds["flows_obs"], float),
                flow_names=[str(x) for x in ds["flow_obs_names"]],
                exog_values=np.asarray(ds["exog_values"], float),
                exog_cols=[str(x) for x in ds["exog_cols"]],
                mask_val=mask_val, mask_test=mask_test)

    t0 = time.time()
    status = dict(base)
    try:
        pred_F, per_flow, n_train = _arx_ensemble(data, spec)
        status.update(n_train_rows=int(n_train),
                      max_lag_used=int(max(f["p"] for f in per_flow)),
                      n_degenerate=int(sum(1 for f in per_flow
                                           if not np.isfinite(f["aic"]))))
    except Exception as exc:
        status.update(status_="failed", error=f"{type(exc).__name__}: {exc}",
                      wall_s=time.time() - t0)
        print(f"[{tag} arx] failed: {exc}", flush=True)
        return None, status

    fnames = data["flow_names"]
    S_hat = derive_stocks(pred_F, fnames, data["stocks_obs"], n_train)
    a_hat = derive_alpha(pred_F, fnames, years, data["stocks_obs"])
    t_hat = np.full((T, len(TAU_SUP_NAMES)), np.nan)
    t_hat[1:, :] = derive_tau(pred_F, fnames)

    # lift the 18 observable columns into the model's 19-column internal order
    Fpred = np.full((T - 1, v5.N_FLOWS), np.nan)
    Fpred[:, np.asarray(ds["flow_obs_to_pred_idx"], int)] = pred_F

    meta = dict(row, model="arx", spec=spec,
                **{k: v for k, v in sp.items() if k != "split_indices"})
    arrays = dict(
        years=years,
        alpha_names=np.array(ALPHA_NAMES, dtype=object),
        tau_names=np.array(TAU_SUP_NAMES, dtype=object),
        stock_names=np.array(STOCK_NAMES, dtype=object),
        flow_names=np.array(fnames, dtype=object),
        flow_names_pred=np.array(list(v5.FLOW_NAMES), dtype=object),
        alpha_obs_arm=np.asarray(ds["alpha_obs"], float),
        tau_obs_arm=np.asarray(ds["tau_sup_obs"], float),
        S_obs_arm=np.asarray(ds["stocks_obs"], float),
        F_obs_arm=np.asarray(ds["flows_obs"], float),
        alpha_true=np.asarray(truth["alphas_point"], float),
        tau_true=np.asarray(truth["tau_sup_point"], float),
        S_true=np.asarray(truth["stocks_clean"], float),
        F_true=np.asarray(truth["F_int_clean"], float),
        years_annual=np.asarray(annual["years"], float),
        alpha_obs_annual=np.asarray(annual["alpha_obs"], float),
        alpha_true_annual=np.asarray(annual["alphas_point"], float),
        tau_true_annual=np.asarray(annual["tau_sup_point"], float),
        S_true_annual=np.asarray(annual["stocks_clean"], float),
        F_true_annual=np.asarray(annual["F_int_clean"], float),
        alpha_pred_B_at_obs=a_hat,
        tau_pred_B_at_obs=t_hat,
        S_pred_B=S_hat,
        F_pred_B=Fpred,
        mask_train=mask_train, mask_val=mask_val, mask_test=mask_test,
        meta_json=np.asarray(json.dumps(dict(meta, path=os.path.basename(path))),
                             dtype=object),
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **arrays)
    status.update(status_="ok", wall_s=time.time() - t0, rss_mb=_rss_mb())
    print(f"[{tag} arx] {time.time()-t0:.1f}s -> {os.path.basename(path)}",
          flush=True)
    return path, status


# ---------------------------------------------------------------------------
# self-tests
# ---------------------------------------------------------------------------
def tau_identity_check(tag="wp11a_freq_1y", verbose=True):
    """How far the flow-ratio tau is from the point tau it will be scored on.

    Run on the **observed** flows, so no estimator is involved: whatever gap
    appears is the aggregation bias of the identity itself, and it bounds how
    much of the `arx` tau column is construction rather than estimation.
    """
    ds = S.load_dataset(F.dataset_path(tag))
    der = derive_tau(np.asarray(ds["flows_obs"], float),
                     [str(x) for x in ds["flow_obs_names"]])
    obs = np.asarray(ds["tau_sup_obs"], float)[1:, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = 100.0 * np.abs(der - obs) / np.maximum(np.abs(obs), 1e-12)
    rows = []
    for k, nm in enumerate(TAU_SUP_NAMES):
        rows.append(dict(tag=tag, tau=nm,
                         max_abs_diff=float(np.nanmax(np.abs(der[:, k] - obs[:, k]))),
                         max_rel_pct=float(np.nanmax(rel[:, k])),
                         median_rel_pct=float(np.nanmedian(rel[:, k]))))
    if verbose:
        print(f"  tau identity on {tag} (observed flows, no estimator):")
        for r in rows:
            print(f"    {r['tau']:<14s} median {r['median_rel_pct']:7.3f}%  "
                  f"max {r['max_rel_pct']:7.3f}%")
    return rows


def alpha_identity_check(tag="wp11a_freq_1y", verbose=True):
    """`derive_alpha` on the observed flows must reproduce `alpha_obs` exactly.

    Same formula, same denominator, so anything but machine precision means
    the AR's alpha is not the target the UDE was supervised on.
    """
    ds = S.load_dataset(F.dataset_path(tag))
    der = derive_alpha(np.asarray(ds["flows_obs"], float),
                       [str(x) for x in ds["flow_obs_names"]],
                       ds["years"], ds["stocks_obs"])
    obs = np.asarray(ds["alpha_obs"], float)
    d = np.abs(der - obs)
    worst = float(np.nanmax(d))
    rel = float(np.nanmax(d / np.maximum(np.abs(obs), 1e-12)))
    if verbose:
        print(f"  alpha identity on {tag}: max abs {worst:.3e}, "
              f"max rel {rel:.3e}")
    return dict(tag=tag, max_abs=worst, max_rel=rel, ok=bool(rel < 1e-10))


def noop_check(verbose=True):
    """Three identities that must hold before any WP-11d number is trusted.

    1. **The rebound loader is a no-op when disarmed.**  `zinc_baseline`'s
       `load_zinc_data`, after `install()`, must return the real record
       array-for-array against the reference it held before.
    2. **The split substitution reproduces the published one.**  At the annual
       full span, `split_for`'s explicit indices must equal what
       `trainval_frac = 0.7, val_frac = 0.2` produces inside the baseline.
    3. **`derive_alpha` reproduces `alpha_obs`.**  The AR's alpha must be the
       same construction as the target, differing only in the numerator.
    """
    import zinc_colloc_v5 as v5
    import zinc_baseline as zb

    ref_loader = zb.load_zinc_data
    ref = ref_loader(lab.load_anchor_config()["xlsx_path"],
                     extra_exog_cols=_load_gam_config().get("extra_exog_cols"))
    install(v5)
    disarm()
    got = zb.load_zinc_data(lab.load_anchor_config()["xlsx_path"],
                            extra_exog_cols=_load_gam_config().get("extra_exog_cols"))
    bad = []
    for k, v in ref.items():
        if isinstance(v, np.ndarray):
            w = np.asarray(got[k])
            if v.shape != w.shape or not np.array_equal(np.nan_to_num(v, nan=-1e30),
                                                        np.nan_to_num(w, nan=-1e30)):
                bad.append(k)
        elif list(v) != list(got[k]) if isinstance(v, list) else v != got[k]:
            bad.append(k)
    loader_ok = not bad

    # 2 — the split substitution, against the baseline's own fraction rule
    ds = S.load_dataset(F.dataset_path("wp11a_freq_1y"))
    T = len(ds["years"])
    n_trainval = min(max(int(np.floor(0.7 * T)), 3), T - 1)
    n_val = max(3, int(np.ceil(0.2 * n_trainval)))
    cut = max(n_trainval - n_val, 2)
    frac_idx = dict(train_end=cut - 1, val_end=n_trainval - 1)
    ours = F.split_for(ds["years"])["split_indices"]
    split_ok = (int(ours["train_end"]) == frac_idx["train_end"]
                and int(ours["val_end"]) == frac_idx["val_end"])

    a = alpha_identity_check(verbose=False)

    if verbose:
        print("=" * 74)
        print("zinc_scale_lab --noop")
        print("=" * 74)
        print(f"  1 loader no-op when disarmed        : "
              f"{'PASS' if loader_ok else 'FAIL ' + str(bad)}")
        print(f"  2 split_for == fraction split (d=1) : "
              f"{'PASS' if split_ok else 'FAIL'}  "
              f"ours={dict(ours)} fractions={frac_idx}")
        print(f"  3 derive_alpha == alpha_obs         : "
              f"{'PASS' if a['ok'] else 'FAIL'}  max rel {a['max_rel']:.2e}")
        print("=" * 74)
    return dict(loader_noop=loader_ok, loader_bad=bad, split_ok=split_ok,
                alpha_identity=a, ok=bool(loader_ok and split_ok and a["ok"]))


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------
def check(verbose=True, cfg=None):
    """Resolved drivers, `input_dim`, the arm grid, and the UDE fits on disk."""
    import zinc_colloc_v5 as v5
    install(v5)
    disarm()
    info = lab.check(verbose=verbose)
    rows = arms()
    have = {r["tag"]: len(ude_fit_paths(r["tag"])) for r in rows}
    info = dict(info, patches=list(PATCHES), models=list(MODELS),
                arms=[dict(tag=r["tag"], family=r["family"], kind=r["kind"],
                           delta=r["delta"], span=r["span"], n_obs=r["n_obs"],
                           ude_seeds=have[r["tag"]]) for r in rows])
    if verbose:
        print("=" * 78)
        print("zinc_scale_lab --check")
        print("=" * 78)
        print(f"  patches applied : {PATCHES}")
        print(f"  model classes   : {list(MODELS)}  (+ constTC_B variant)")
        print(f"  ARX spec        : {ARX_SPEC}  (run_wp1f)")
        print(f"  arms            : {len(rows)}")
        print(f"  {'tag':<28s} {'family':<18s} {'delta':>8s} {'span':>6s} "
              f"{'N':>5s} {'UDE':>5s}")
        for r in rows:
            print(f"  {r['tag']:<28s} {r['family']:<18s} {r['delta']:8.4f} "
                  f"{r['span']:6.0f} {r['n_obs']:5d} {have[r['tag']]:5d}")
        missing = [t for t, n in have.items() if n == 0]
        print(f"\n  arms with no UDE fit on disk : "
              f"{missing if missing else 'none'}")
        print("=" * 78)
    return info


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="WP-11d lab module")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--noop", action="store_true")
    ap.add_argument("--tau-identity", action="store_true")
    ap.add_argument("--fit", default=None,
                    choices=list(FITTED_MODELS) + ["arx"])
    ap.add_argument("--tags", default=None,
                    help="comma-separated arm tags (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    if args.check or not any([args.noop, args.tau_identity, args.fit]):
        check()
    if args.noop:
        r = noop_check()
        if not r["ok"]:
            return 1
    if args.tau_identity:
        tau_identity_check()
    if args.fit:
        tt = args.tags.split(",") if args.tags else tags()
        for t in tt:
            if args.fit == "arx":
                fit_arx(t.strip(), seed=args.seed)
            else:
                fit_baseline(t.strip(), args.fit, seed=args.seed,
                             verbose=args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
