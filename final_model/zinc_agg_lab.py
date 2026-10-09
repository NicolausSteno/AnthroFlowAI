#!/usr/bin/env python3
"""
zinc_agg_lab.py — lab module for WP-11h (temporal aggregation bias)
===================================================================

WP-11h asks the one question about this model class that is *not* about data
volume: **does fitting a discrete-time model to annual flow totals put a
systematic error in the transfer coefficients that more data cannot remove?**
The spec's own words:

    A discrete-time model fitted to annual flow totals implicitly treats a
    period integral as if it were a state variable, and cannot separate an
    instantaneous rate from an accumulated total.  The continuous formulation
    expresses the mixed observation operator directly (stocks point-in-time,
    flows as integrals) and therefore should not inherit that bias.

    Hypothesis: the discrete model's coefficient bias is a function of the
    aggregation interval, not of N, so it does *not* vanish as data
    accumulate.

`run_wp11h.py` drives it.

What is new here, and what is re-used
-------------------------------------
Almost nothing is fitted.  `COMPUTE_STATUS.md` puts WP-11h in the "shares
WP-3/11a fits" row and that is what happens: every UDE number comes from
WP-11a's dumps and every base-twin AR number from WP-11d's, unchanged.  The
new fits are **thirteen AR/ARX ensembles at ~4 s each** — four on the `season`
twin, which WP-11d never touched because it restricted itself to `base`; eight
on WP-5's independent noise draws, so the deterministic discrete estimator has
an uncertainty band on its *bias* rather than a single number; and one on the
noise-free `base_d1y_clean` record, which is the control that separates
aggregation bias from noise-induced bias.

Bias, and why it is not relRMSE
-------------------------------
Every other package in WP-11 reports relRMSE, which is a magnitude and cannot
tell a systematic offset from dispersion.  Aggregation bias is by definition
the systematic part, so the estimand here is the **signed** error in log units,

    b_k = median over scored years of  log( alpha_hat_k(Y) / alpha_true_k(Y) )

reported as `100 * (exp(b_k) - 1)` percent, with the row-to-row IQR of the
same log ratio carried alongside as the dispersion.  The log scale is the
scale the model is supervised on (`w_alpha * |log a_pred - log a_obs|`,
v5:1440/1601) and it makes the decomposition below additive.

The estimand ladder — what a discrete model *can* identify
----------------------------------------------------------
The comparison `alpha_hat` against `alpha_true(Y)` mixes two different things:
how well the estimator did, and whether the quantity it can identify from
Delta-aggregated data is `alpha_true(Y)` at all.  `zinc_synth_lab
.alpha_target_bias` already separates them exactly, per window, and this
module lifts its three rungs onto the annual scoring grid:

    alpha_true(Y)          the instantaneous coefficient at the node -- the
                           object Chapter 2's A(t) is made of, and what both
                           model classes claim to report
    a_unweighted           (1/Delta) int alpha dt over the window closing at Y
                           -- the best a Delta-aggregated record can identify
    a_weighted             int F dt / int S dt -- exposure weighting, the
                           within-window Cov_w(a,S)/mean_w(S) term
    a_obs                  int F dt / trapz(S) -- the target actually built,
                           adding the trapezoid quadrature term

so that, in log units and exactly,

    log(a_obs / alpha_true) = window_mean + covariance + quadrature

The **aggregation floor** is the first rung: `log(a_unweighted/alpha_true)` is
what any estimator of a Delta-window quantity inherits before it has made a
single error of its own, it is a function of Delta and of the truth alone, and
**no amount of data at that Delta removes it**.  That is the spec's hypothesis
stated as a measurable quantity rather than as a claim about estimators, and
`run_wp11h` tests whether each fitted class actually sits on it.

The design, and why N and Delta have to be crossed
--------------------------------------------------
A curve of bias against N cannot distinguish "bias falls with N" from "bias
falls with Delta" unless the two are varied separately.  WP-11a's arms already
cross them, and this module reads that crossing off rather than building new
arms:

    resolution   Delta varies, span fixed at 39 yr
                 base   Delta = 10, 5, 2 (coarse), 1, 1/2, 1/4, 1/12
                 season Delta = 1, 1/2, 1/4, 1/12
    length       Delta fixed, N varies -- three separate levels
                 Delta = 1     N = 11, 21, 40   (span 10 / 20 / 39 yr)
                 Delta = 1/4   N = 41, 157      (span 10 / 39 yr)
                 Delta = 1/12  N = 157, 469     (span 13 / 39 yr)
    replicate    base, Delta = 1, N = 40, eight independent noise draws
    control      base, Delta = 1, N = 40, no observation noise

The `length` family is the test.  Seven records at three fixed aggregation
intervals, N spanning 11 to 469: if bias is a property of Delta the three
levels separate and nothing moves within a level.

Two properties of the arms that constrain what may be claimed
-------------------------------------------------------------
1.  **The `coarse` arms are WP-6b's operator, not a re-integration.**
    `zinc_coarse_lab._flat_window` spreads a Delta-year window total evenly
    across the Delta annual rows it covers, preserving the total.  The
    information content is a Delta-year integral -- which is why they extend
    the Delta axis above one year here -- but the row grid stays annual and
    the operator is not identical to re-sampling the twin at Delta.  Marked
    `operator="coarse"` in every table.
2.  **Every arm is a different noise draw.**  There is no construction that
    makes a monthly record and an annual record the same draw (WP-11a part
    2b).  The eight `replicate` fits exist to size that confound at the
    headline arm; it is not removable elsewhere and no single-arm difference
    below the replicate spread is reported as a finding.

Scoring region
--------------
`common` (to 1987) for anything that compares arms of different span, because
the 10-yr arms sit entirely before every turning point in
`zinc_synth_lab.TRUTH` and scoring each arm on its own window would confound
sample size with which years were seen.  `fitted` (to the trainval boundary)
for anything that does not.  Both are carried in every table; WP-11a's regions,
WP-11a's scorer for the mapping onto the annual grid.

Pattern (CLAUDE.md rule 1)
--------------------------
`zinc_colloc_v5.py` is imported and never edited; `install()` delegates to
`zinc_synth_lab.install` and adds nothing.  One rebinding is made and it is on
a *lab* module, not the core: `zinc_scale_lab.fit_arx` resolves its arm row
through `zinc_scale_lab.arm_of`, which only knows WP-11d's eleven base-twin
tags, so `_arm_registry()` is installed in its place for the duration of a fit
so the season, replicate and clean records can be fitted by exactly the same
code path.  Nothing else about that function changes -- same ARX spec, same
`derive_*`, same dump format -- which is the point: the AR numbers in this
package and in WP-11d are the same estimator.

CLI
---
    python zinc_agg_lab.py --check
    python zinc_agg_lab.py --noop
    python zinc_agg_lab.py --ladder
    python zinc_agg_lab.py --fit-arx --tags wp11a_season_1y
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import resource
import sys
import time

import numpy as np

import zinc_alpha_lab as lab
import zinc_synth_lab as S
import zinc_freq_lab as F
import zinc_scale_lab as SC
import zinc_interp_lab as IL

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "analysis", "wp11h")
FIT_DIR = os.path.join(OUT_DIR, "fits")
LADDER_DIR = os.path.join(OUT_DIR, "ladder")
SYNTH_DIR = F.SYNTH_DIR
UDE_FIT_DIR = F.FIT_DIR
WP11D_FIT_DIR = SC.FIT_DIR

ALPHA_NAMES = F.ALPHA_NAMES
STOCK_NAMES = F.STOCK_NAMES

# WP-11a's annual scoring grid and its cross-arm region, re-used unchanged so
# that a number here and a number in WP-11a/11d mean the same thing.
ANNUAL = np.arange(1980.0, 2020.0)
COMMON_SCORE_END = 1987.0

MODELS = ("ude", "arx")

PATCHES: list[str] = []


def _rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024.0 ** 2) if sys.platform == "darwin" else r / 1024.0


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------
def install(v5mod=None):
    """Delegate to `zinc_synth_lab.install`; WP-11h adds no core patch."""
    global PATCHES
    PATCHES = list(S.install(v5mod))
    return PATCHES


@contextlib.contextmanager
def _scale_lab_knows(rows):
    """Lend `zinc_scale_lab.arm_of` this package's arm registry.

    WP-11d's `fit_arx` is the estimator this package wants -- same spec, same
    derivations, same dump format -- and the only thing standing between it
    and a `season` or replicate record is that its `arm_of` enumerates
    WP-11d's own eleven tags.  Rebinding it for the duration of the call is
    the smallest intervention that keeps the two packages' AR ensembles the
    same object; nothing else in `zinc_scale_lab` is touched.
    """
    by_tag = {r["tag"]: r for r in rows}
    prev = SC.arm_of

    def arm_of(tag):
        if tag in by_tag:
            r = by_tag[tag]
            return dict(tag=r["tag"], kind=r["kind"], twin=r["twin"],
                        delta=r["delta"], span=r["span"], n_obs=r["n_obs"],
                        family=r["family"])
        return prev(tag)

    SC.arm_of = arm_of
    try:
        yield
    finally:
        SC.arm_of = prev


# ---------------------------------------------------------------------------
# the arm registry
# ---------------------------------------------------------------------------
# Extra records that are not in `zinc_freq_lab.design()`: WP-5's independent
# noise draws of the annual base twin, and WP-3's noise-free annual record.
REPLICATE_TAGS = tuple(IL.REPLICATE_TAG.format(r) for r in IL.REPLICATE_SEEDS)
CLEAN_TAG = "base_d1y_clean"

# `kind` -> the family it contributes to here.  `freq` at Delta = 1 is both
# the finest `length` point and the 1/yr `resolution` point, as in WP-11d.
_FAMILY = {"freq": "resolution", "season": "resolution", "coarse": "resolution",
           "span": "length", "matchedN": "length"}


def _delta_agg(row):
    """The aggregation interval the record's flow observations carry, in years.

    For every kind but `coarse` this is the sampling width itself.  For
    `coarse` it is WP-6b's retained-reading spacing: `_flat_window` preserves
    the Delta-year window total and spreads it flat over the annual rows, so
    the independent information is a Delta-year integral even though the rows
    are annual.
    """
    return float(row["delta"])


def arms():
    """Every record WP-11h scores, in a fixed order.

    Drawn from `zinc_freq_lab.design()` -- the `freqsm`/`freqfw` controls are
    dropped, since they are the same records refitted under a different
    smoothing weight and a different curriculum and belong to WP-11a's
    question, not this one -- plus the replicate and clean records.
    """
    rows = []
    for r in F.design():
        fam = _FAMILY.get(r["kind"])
        if fam is None:
            continue
        twin = r.get("twin", F.TWIN_ARM)
        rows.append(dict(
            tag=r["tag"], twin=twin, kind=r["kind"], family=fam,
            delta=float(r["delta"]), delta_agg=_delta_agg(r),
            span=float(r["span"]), n_obs=int(r["n_obs"]),
            operator="coarse" if r["kind"] == "coarse" else "resample",
            noise=True))
    # the annual full-span base arm anchors all three families
    for r in rows:
        if r["tag"] == F.arm_tag("freq", 1.0, F.FULL_SPAN):
            r["family"] = "resolution+length"
    for tag in REPLICATE_TAGS:
        rows.append(dict(tag=tag, twin="base", kind="replicate",
                         family="replicate", delta=1.0, delta_agg=1.0,
                         span=F.FULL_SPAN, n_obs=40, operator="resample",
                         noise=True))
    rows.append(dict(tag=CLEAN_TAG, twin="base", kind="clean", family="control",
                     delta=1.0, delta_agg=1.0, span=F.FULL_SPAN, n_obs=40,
                     operator="resample", noise=False))
    rows.sort(key=lambda r: (r["family"], r["twin"], -r["delta_agg"],
                             r["n_obs"], r["tag"]))
    return rows


def arm_of(tag):
    for r in arms():
        if r["tag"] == tag:
            return r
    raise KeyError(f"{tag!r} is not a WP-11h arm; known: "
                   f"{[r['tag'] for r in arms()]}")


def tags(family=None, twin=None):
    return [r["tag"] for r in arms()
            if (family is None or family in r["family"].split("+"))
            and (twin is None or r["twin"] == twin)]


# The `length` family, grouped by the aggregation interval that is held fixed.
# This is the design that separates the spec's two candidate explanations, and
# it is written out rather than inferred so the grouping is auditable.
LENGTH_LEVELS = {
    1.0: ("wp11a_span_1y_s10y", "wp11a_span_1y_s20y", "wp11a_freq_1y"),
    0.25: ("wp11a_matchedN_0p25y_s10y", "wp11a_freq_0p25y"),
    1.0 / 12.0: ("wp11a_matchedN_0p083y_s13y", "wp11a_freq_0p083y"),
}

# The matched-N pairs: same N, different route to it (WP-11a's control).
MATCHED_PAIRS = (("wp11a_matchedN_0p25y_s10y", "wp11a_freq_1y"),
                 ("wp11a_matchedN_0p083y_s13y", "wp11a_freq_0p25y"))


def dataset_path(tag):
    """The record behind an arm; replicate/clean tags are plain files."""
    p = os.path.join(SYNTH_DIR, f"{tag}.npz")
    if os.path.exists(p) and tag in set(REPLICATE_TAGS) | {CLEAN_TAG}:
        return p
    return F.dataset_path(tag)


def ude_fit_paths(tag, fit_dir=UDE_FIT_DIR):
    return [p for p in sorted(glob.glob(os.path.join(fit_dir, f"{tag}_seed*.npz")))
            if not p.endswith("_spec.npz")]


def arx_fit_path(tag):
    """Where an arm's AR ensemble lives.

    WP-11d's dumps are read where they exist and never rewritten; the thirteen
    records it did not fit land in this package's own directory.
    """
    p = SC.fit_path(tag, "arx", 0, WP11D_FIT_DIR)
    if os.path.exists(p):
        return p
    return SC.fit_path(tag, "arx", 0, FIT_DIR)


def missing_arx():
    return [r["tag"] for r in arms() if not os.path.exists(arx_fit_path(r["tag"]))]


# ---------------------------------------------------------------------------
# the thirteen new AR fits
# ---------------------------------------------------------------------------
def fit_arx(tag, out_dir=FIT_DIR, skip_existing=True):
    """WP-11d's AR/ARX ensemble on one record, unchanged, via its own code."""
    rows = arms()
    with _scale_lab_knows(rows):
        return SC.fit_arx(tag, out_dir=out_dir, seed=0,
                          skip_existing=skip_existing)


# ---------------------------------------------------------------------------
# the estimand ladder
# ---------------------------------------------------------------------------
def ladder_path(twin, delta):
    return os.path.join(LADDER_DIR, f"ladder_{twin}_{S._dtag(float(delta))}.npz")


def build_ladder(twin, deltas, *, cfg=None, verbose=True, skip_existing=True):
    """`alpha_target_bias` at each Delta, cached, one dense solve per twin.

    # Ch2 App. time_varying_product_shares — the appendix states that it does
    # not develop a full non-autonomous dynamic stock model.  The ladder is
    # the observational counterpart of that gap: a time-varying `A(t)` is only
    # identifiable up to whatever the reporting window averages away, and the
    # floor computed here is exactly what an annual reporting convention
    # averages away from `alpha_1..alpha_22`.

    Computed from the twin's own dense trajectory rather than read out of the
    arm files, for two reasons: the `coarse` arms store the *annual* truth
    (`build_arm` copies it from the annual arm), which is the wrong window for
    a Delta-year aggregate, and the replicate records store only some of the
    rungs.  One source for every rung keeps the decomposition additive.
    """
    os.makedirs(LADDER_DIR, exist_ok=True)
    want = [d for d in deltas
            if not (skip_existing and os.path.exists(ladder_path(twin, d)))]
    if not want:
        return [ladder_path(twin, d) for d in deltas]
    ctx = S.driver_context(cfg)
    t0 = time.time()
    dense = S.dense_solve(ctx, twin)
    if verbose:
        print(f"[ladder/{twin}] dense solve {time.time()-t0:.1f}s, "
              f"{dense['t'].size} nodes", flush=True)
    for d in want:
        t1 = time.time()
        b = S.alpha_target_bias(ctx, dense, float(d), arm_name=twin)
        np.savez_compressed(
            ladder_path(twin, d),
            years=np.asarray(b["years"], float),
            a_unweighted=np.asarray(b["a_unweighted"], float),
            a_weighted=np.asarray(b["a_weighted"], float),
            a_obs=np.asarray(b["a_obs"], float),
            cov_term=np.asarray(b["cov_term"], float),
            corr_FS=np.asarray(b["corr_FS"], float),
            delta=np.asarray(float(d)),
            alpha_names=np.array(ALPHA_NAMES, dtype=object))
        if verbose:
            print(f"[ladder/{twin}] delta={float(d):.4f}  "
                  f"{b['a_obs'].shape[0]} windows  {time.time()-t1:.1f}s",
                  flush=True)
    import jax
    jax.clear_caches()
    return [ladder_path(twin, d) for d in deltas]


def build_ladder_variant(name, deltas, season, *, cfg=None, verbose=True,
                         skip_existing=True):
    """A ladder for a hand-specified subset of the `season` components.

    `zinc_synth_lab.dense_solve` and `alpha_target_bias` both take a `season`
    override, so the twin can be re-integrated carrying only the two annual
    *harmonics* (1 and 2 cycles/yr, annihilated by an annual window) or only
    the two *non-harmonic* components (1.35 and 2.70/yr, attenuated by
    `|sinc|` and then aliased).  Splitting the season floor that way is the
    estimator-side counterpart of WP-11f's transfer-function decomposition:
    the harmonic part is a constant log offset that annual data cannot see at
    all, the non-harmonic part is dispersion that it can.
    """
    os.makedirs(LADDER_DIR, exist_ok=True)
    want = [d for d in deltas
            if not (skip_existing and os.path.exists(ladder_path(name, d)))]
    if not want:
        return [ladder_path(name, d) for d in deltas]
    ctx = S.driver_context(cfg)
    dense = S.dense_solve(ctx, "season", season=season)
    for d in want:
        b = S.alpha_target_bias(ctx, dense, float(d), arm_name="season",
                                season=season)
        np.savez_compressed(
            ladder_path(name, d),
            years=np.asarray(b["years"], float),
            a_unweighted=np.asarray(b["a_unweighted"], float),
            a_weighted=np.asarray(b["a_weighted"], float),
            a_obs=np.asarray(b["a_obs"], float),
            cov_term=np.asarray(b["cov_term"], float),
            corr_FS=np.asarray(b["corr_FS"], float),
            delta=np.asarray(float(d)),
            season_json=np.asarray(json.dumps(
                [[str(c), float(f), float(a), float(p)]
                 for (c, f, a, p) in season]), dtype=object),
            alpha_names=np.array(ALPHA_NAMES, dtype=object))
        if verbose:
            print(f"[ladder/{name}] delta={float(d):.4f}  "
                  f"{b['a_obs'].shape[0]} windows", flush=True)
    # the point truth under the same override, for the numerator of the floor
    p = os.path.join(LADDER_DIR, f"truth_{name}.npz")
    if not os.path.exists(p):
        tc = S.truth_coefficients(ctx, ANNUAL, "season", season=season)
        np.savez_compressed(p, alphas=np.asarray(tc["alphas"], float))
    import jax
    jax.clear_caches()
    return [ladder_path(name, d) for d in deltas]


def truth_point_variant(name):
    p = os.path.join(LADDER_DIR, f"truth_{name}.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return np.asarray(np.load(p)["alphas"], float)


# The two halves of `zinc_synth_lab.TRUTH["season"]`, split on whether the
# component sits at an exact harmonic of a one-year window.
SEASON_HARMONIC = tuple(c for c in S.TRUTH["season"]
                        if abs(float(c[1]) - round(float(c[1]))) < 1e-9)
SEASON_NONHARMONIC = tuple(c for c in S.TRUTH["season"]
                           if abs(float(c[1]) - round(float(c[1]))) >= 1e-9)


def load_ladder(twin, delta):
    p = ladder_path(twin, delta)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} — run `python run_wp11h.py --ladder` first")
    return np.load(p, allow_pickle=True)


def truth_point_annual(twin, *, cfg=None):
    """`alpha_true` at the annual nodes, per twin, from the dense truth.

    The arm dumps already carry `alpha_true_annual`, and this reproduces it —
    `noop_check` asserts that it does — but the ladder needs the same object
    built from the same dense solve as its own rungs, or the decomposition
    would be additive only up to two different quadratures.
    """
    ctx = S.driver_context(cfg)
    tc = S.truth_coefficients(ctx, ANNUAL, twin)
    return np.asarray(tc["alphas"], float)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def _region_masks(years_arm, val_end_year):
    """WP-11a's two regions on the annual grid, plus the arm's own span cap."""
    hi = float(np.asarray(years_arm, float)[-1])
    inside = ANNUAL <= hi + 1e-9
    return {"fitted": (ANNUAL <= float(val_end_year) + 1e-9) & inside,
            "common": (ANNUAL <= COMMON_SCORE_END + 1e-9) & inside}


def log_bias(pred, true, mask):
    """Signed bias and dispersion of `log(pred/true)` over the masked rows.

    Returns percent, `100 * (exp(median log ratio) - 1)`, so a bias of +10%
    means the estimator reports a coefficient a tenth larger than the truth.
    The IQR is reported in the same units around the median.
    """
    p, t = np.asarray(pred, float), np.asarray(true, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(p / t)
    v = lr[np.asarray(mask, bool) & np.isfinite(lr)]
    if v.size == 0:
        return dict(bias_pct=np.nan, bias_log=np.nan, disp_pct=np.nan,
                    absbias_pct=np.nan, n_rows=0)
    m = float(np.median(v))
    q1, q3 = float(np.quantile(v, 0.25)), float(np.quantile(v, 0.75))
    return dict(bias_pct=100.0 * (np.exp(m) - 1.0), bias_log=m,
                disp_pct=100.0 * (np.exp(q3) - np.exp(q1)),
                absbias_pct=abs(100.0 * (np.exp(m) - 1.0)), n_rows=int(v.size))


def read_dump(path):
    """Normalise the two dump formats this package reads into one dict.

    WP-11a/11d write `alpha_pred_B_at_obs` and carry their own `meta_json`
    (`zinc_freq_lab.dump_fit`, `zinc_scale_lab.fit_arx`).  WP-3's and WP-5's
    fits of the annual base twin — the noise-free control and the eight
    replicate draws — predate that format and are `zinc_A_lab.dump_A` files,
    which carry the network's coefficients under `alphas` and no metadata
    block.

    **`alphas` and `alphas_at_obs` are the same array on these fits**, and
    that is a property of the configuration rather than an approximation:
    `predictions()` builds the two by evaluating the network at `S_pred` and
    at `S_obs` (v5:2523-2524), `anchor_v4` sets `use_stock_input: false`, so
    the four stock features are identically zero in both (COMPUTE_STATUS flag
    2) and nothing else differs — there is no `at_obs` override for alpha.
    `_assert_stockless` checks the flag rather than trusting this note.
    """
    d = np.load(path, allow_pickle=True)
    base = os.path.basename(path)
    seed = int(base.split("seed")[-1].split(".")[0])
    if "meta_json" in d.files:
        meta = json.loads(str(d["meta_json"]))
        tag = str(meta.get("tag", base.split("_seed")[0]))
        return dict(tag=tag, seed=seed, years=np.asarray(d["years"], float),
                    alpha_pred=np.asarray(d["alpha_pred_B_at_obs"], float),
                    alpha_true_annual=np.asarray(d["alpha_true_annual"], float),
                    val_end_year=float(meta["val_end_year"]),
                    dump_format="wp11a")
    _assert_stockless()
    tag = base.split("_seed")[0]
    yrs = np.asarray(d["years"], float)
    row = arm_of(tag)
    return dict(tag=tag, seed=seed, years=yrs,
                alpha_pred=np.asarray(d["alphas"], float),
                alpha_true_annual=truth_point_annual(row["twin"]),
                val_end_year=float(F.split_for(yrs)["val_end_year"]),
                dump_format="A_lab")


_STOCKLESS = None


def _assert_stockless(cfg=None):
    """`use_stock_input` must be false for `read_dump`'s A-lab branch to hold."""
    global _STOCKLESS
    if _STOCKLESS is None:
        c = cfg or lab.load_anchor_config()
        _STOCKLESS = (not bool(c.get("use_stock_input", False)),
                      not bool(c.get("use_time_input", False)))
        if not _STOCKLESS[0]:
            raise RuntimeError(
                "anchor_v4 has use_stock_input=True, so `alphas` and "
                "`alphas_at_obs` are different arrays and the A_lab dumps "
                "cannot stand in for WP-11a's `alpha_pred_B_at_obs`.")
    return _STOCKLESS


def score_fit(path, model, *, twin=None):
    """Every WP-11h row for one fitted object.

    Three references per channel, all on the annual grid:
      `vs_point`   against `alpha_true(Y)`   — the reported estimand
      `vs_window`  against `(1/D) int a dt`  — what a D-aggregated record can
                                               identify at all
      `floor`      the gap between those two — the aggregation floor, an
                                               estimator-free property of D
    """
    dd = read_dump(path)
    tag = dd["tag"]
    row = arm_of(tag)
    tw = twin or row["twin"]
    yrs = dd["years"]
    reg = _region_masks(yrs, dd["val_end_year"])

    a_pred = _to_annual(yrs, dd["alpha_pred"])
    a_point = dd["alpha_true_annual"]
    lad = load_ladder(tw, row["delta_agg"])
    a_win = _to_annual(lad["years"], lad["a_unweighted"])

    seed = dd["seed"]
    common = dict(tag=tag, model=model, twin=tw, kind=row["kind"],
                  family=row["family"], operator=row["operator"],
                  noise=bool(row["noise"]), delta=row["delta"],
                  delta_agg=row["delta_agg"], span=row["span"],
                  n_obs=row["n_obs"], seed=seed,
                  dump_format=dd["dump_format"])
    out = []
    for split, m in reg.items():
        for k, ch in enumerate(ALPHA_NAMES):
            b_pt = log_bias(a_pred[:, k], a_point[:, k], m)
            b_wn = log_bias(a_pred[:, k], a_win[:, k], m)
            fl = log_bias(a_win[:, k], a_point[:, k], m)
            out.append(dict(
                common, channel=ch, split=split,
                bias_vs_point_pct=b_pt["bias_pct"],
                bias_vs_point_log=b_pt["bias_log"],
                disp_vs_point_pct=b_pt["disp_pct"],
                bias_vs_window_pct=b_wn["bias_pct"],
                bias_vs_window_log=b_wn["bias_log"],
                floor_pct=fl["bias_pct"], floor_log=fl["bias_log"],
                relrmse_vs_point_pct=_rel_rmse(a_pred[:, k], a_point[:, k], m),
                n_rows=b_pt["n_rows"]))
    return out


def _to_annual(x, values):
    return _interp_cols(np.asarray(x, float).ravel(), np.asarray(values, float))


def _interp_cols(x, V):
    """Linear interpolation onto `ANNUAL`, NaN outside the source span.

    Identical in behaviour to `run_wp11a.to_annual_point`; written here so the
    ladder and the fits go through one function and the `noop` check can
    assert the two agree.
    """
    V = np.atleast_2d(V.T).T
    out = np.full((ANNUAL.size, V.shape[1]), np.nan)
    inside = (ANNUAL >= x[0] - 1e-9) & (ANNUAL <= x[-1] + 1e-9)
    for j in range(V.shape[1]):
        col = V[:, j]
        ok = np.isfinite(col)
        if ok.sum() < 2:
            continue
        out[inside, j] = np.interp(ANNUAL[inside], x[ok], col[ok])
    return out


def _rel_rmse(pred, obs, mask):
    p, o = np.asarray(pred, float), np.asarray(obs, float)
    m = np.isfinite(p) & np.isfinite(o) & np.asarray(mask, bool)
    if not m.any():
        return np.nan
    den = float(np.mean(np.abs(o[m])))
    if den < 1e-12:
        return np.nan
    return 100.0 * float(np.sqrt(np.mean((p[m] - o[m]) ** 2))) / den


def score_arm(tag):
    """Both classes on one record: 8 UDE seeds (where fitted) and the AR."""
    rows = []
    for p in ude_fit_paths(tag):
        rows.extend(score_fit(p, "ude"))
    ap = arx_fit_path(tag)
    if os.path.exists(ap):
        rows.extend(score_fit(ap, "arx"))
    return rows


# ---------------------------------------------------------------------------
# self-tests
# ---------------------------------------------------------------------------
def ladder_identity_check(twin="base", delta=1.0, verbose=True):
    """The three rungs must add to the total, exactly, in log units.

        log(a_obs/a_true) = log(a_unw/a_true) + log(a_w/a_unw) + log(a_obs/a_w)

    A tautology as written, so what it actually tests is that the arrays are
    the ones the identity is about: `cov_term` is `Cov_w(a,S)/mean_w(S)` and
    must reproduce `a_weighted - a_unweighted` to the Simpson error, which is
    the check WP-3 ran per window and this re-runs on the cached ladder.
    """
    lad = load_ladder(twin, delta)
    a_u = np.asarray(lad["a_unweighted"], float)
    a_w = np.asarray(lad["a_weighted"], float)
    cov = np.asarray(lad["cov_term"], float)
    gap = a_w - a_u
    err = np.abs(gap - cov) / np.maximum(np.abs(a_u), 1e-12)
    worst = float(np.nanmax(err))
    if verbose:
        print(f"  cov identity ({twin}, delta={delta:g}): "
              f"max rel {worst:.3e}  ({'PASS' if worst < 1e-6 else 'FAIL'})")
    return dict(twin=twin, delta=float(delta), max_rel=worst,
                ok=bool(worst < 1e-6))


def truth_agreement_check(tag="wp11a_freq_1y", verbose=True):
    """The ladder's truth and the dumps' `alpha_true_annual` must be one object.

    They are built by different calls -- the dumps by `zinc_freq_lab.dump_fit`
    from the arm's own annual reference, the ladder here from a fresh dense
    solve -- and if they disagree, the "floor" this package reports is the gap
    between two truths rather than a property of the observation operator.
    """
    ps = ude_fit_paths(tag)
    if not ps:
        return dict(tag=tag, ok=False, reason="no UDE fit on disk")
    d = np.load(ps[0], allow_pickle=True)
    ref = np.asarray(d["alpha_true_annual"], float)
    got = truth_point_annual(arm_of(tag)["twin"])
    m = np.isfinite(ref) & np.isfinite(got) & (np.abs(ref) > 1e-12)
    worst = float(np.nanmax(np.abs(got[m] - ref[m]) / np.abs(ref[m])))
    if verbose:
        print(f"  truth agreement ({tag}): max rel {worst:.3e}  "
              f"({'PASS' if worst < 1e-9 else 'FAIL'})")
    return dict(tag=tag, max_rel=worst, ok=bool(worst < 1e-9))


def interp_agreement_check(tag="wp11a_freq_0p25y", verbose=True):
    """`_interp_cols` must be `run_wp11a.to_annual_point`, not a lookalike."""
    import run_wp11a as RA
    ps = ude_fit_paths(tag)
    if not ps:
        return dict(tag=tag, ok=False, reason="no UDE fit on disk")
    d = np.load(ps[0], allow_pickle=True)
    yrs = np.asarray(d["years"], float)
    a = RA.to_annual_point(yrs, d["alpha_pred_B_at_obs"], ANNUAL)
    b = _to_annual(yrs, d["alpha_pred_B_at_obs"])
    same = np.array_equal(np.nan_to_num(a, nan=-1e30),
                          np.nan_to_num(b, nan=-1e30))
    if verbose:
        print(f"  interp == run_wp11a.to_annual_point ({tag}): "
              f"{'PASS' if same else 'FAIL'}")
    return dict(tag=tag, ok=bool(same))


def noop_check(verbose=True):
    """Four identities that must hold before any WP-11h number is trusted."""
    import zinc_colloc_v5 as v5
    install(v5)
    S.disarm()
    ide = SC.alpha_identity_check(verbose=False)
    lid = ladder_identity_check(verbose=False)
    tru = truth_agreement_check(verbose=False)
    itp = interp_agreement_check(verbose=False)
    # `_scale_lab_knows` must put `zinc_scale_lab.arm_of` back
    before = SC.arm_of
    with _scale_lab_knows(arms()):
        pass
    restored = SC.arm_of is before
    ok = bool(ide["ok"] and lid["ok"] and tru.get("ok") and itp.get("ok")
              and restored)
    if verbose:
        print("=" * 74)
        print("zinc_agg_lab --noop")
        print("=" * 74)
        print(f"  1 derive_alpha == alpha_obs        : "
              f"{'PASS' if ide['ok'] else 'FAIL'}  max rel {ide['max_rel']:.2e}")
        print(f"  2 ladder covariance identity       : "
              f"{'PASS' if lid['ok'] else 'FAIL'}  max rel {lid['max_rel']:.2e}")
        print(f"  3 ladder truth == dumps' truth     : "
              f"{'PASS' if tru.get('ok') else 'FAIL'}  "
              f"max rel {tru.get('max_rel', float('nan')):.2e}")
        print(f"  4 interp == WP-11a's own           : "
              f"{'PASS' if itp.get('ok') else 'FAIL'}")
        print(f"  5 arm_of rebinding restored        : "
              f"{'PASS' if restored else 'FAIL'}")
        print("=" * 74)
    return dict(alpha_identity=ide, ladder=lid, truth=tru, interp=itp,
                restored=restored, ok=ok)


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------
def check(verbose=True, cfg=None):
    """Resolved drivers, `input_dim`, the (Delta, N) design, and what is on disk."""
    import zinc_colloc_v5 as v5
    install(v5)
    S.disarm()
    info = lab.check(verbose=verbose)
    rows = arms()
    have_u = {r["tag"]: len(ude_fit_paths(r["tag"])) for r in rows}
    have_a = {r["tag"]: os.path.exists(arx_fit_path(r["tag"])) for r in rows}
    info = dict(info, patches=list(PATCHES), models=list(MODELS),
                arms=[dict(r, ude_seeds=have_u[r["tag"]],
                           arx=bool(have_a[r["tag"]])) for r in rows])
    if verbose:
        print("=" * 84)
        print("zinc_agg_lab --check")
        print("=" * 84)
        print(f"  core patches      : {PATCHES}")
        print(f"  lab rebinding     : zinc_scale_lab.arm_of, for the duration "
              f"of a fit only")
        print(f"  model classes     : {list(MODELS)}   ARX spec {SC.ARX_SPEC}")
        print(f"  scoring grid      : {ANNUAL[0]:.0f}-{ANNUAL[-1]:.0f} annual; "
              f"regions fitted / common(<= {COMMON_SCORE_END:.0f})")
        print(f"  arms              : {len(rows)}")
        print(f"  {'tag':<30s} {'twin':<7s} {'family':<18s} {'D_agg':>7s} "
              f"{'N':>5s} {'op':<9s} {'UDE':>4s} {'ARX':>4s}")
        for r in rows:
            print(f"  {r['tag']:<30s} {r['twin']:<7s} {r['family']:<18s} "
                  f"{r['delta_agg']:7.4f} {r['n_obs']:5d} {r['operator']:<9s} "
                  f"{have_u[r['tag']]:4d} {'y' if have_a[r['tag']] else '-':>4s}")
        print(f"\n  length family, by fixed aggregation interval:")
        for d, tt in sorted(LENGTH_LEVELS.items(), reverse=True):
            ns = [arm_of(t)["n_obs"] for t in tt]
            print(f"    D = {d:7.4f} yr : N = {ns}")
        miss = missing_arx()
        print(f"\n  AR fits still to run : {miss if miss else 'none'}")
        lad = sorted(os.path.basename(p) for p in
                     glob.glob(os.path.join(LADDER_DIR, "ladder_*.npz")))
        print(f"  ladder cached        : {len(lad)} files")
        print("=" * 84)
    return info


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="WP-11h lab module")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--noop", action="store_true")
    ap.add_argument("--ladder", action="store_true")
    ap.add_argument("--fit-arx", action="store_true")
    ap.add_argument("--tags", default=None)
    args = ap.parse_args(argv)

    if args.check or not any([args.noop, args.ladder, args.fit_arx]):
        check()
    if args.ladder:
        for tw, ds in ladder_design().items():
            build_ladder(tw, ds)
    if args.fit_arx:
        tt = args.tags.split(",") if args.tags else missing_arx()
        for t in tt:
            fit_arx(t.strip())
    if args.noop:
        r = noop_check()
        if not r["ok"]:
            return 1
    return 0


def ladder_design():
    """Which (twin, Delta) rungs the package needs, from the arms themselves.

    Deduplicated by `zinc_synth_lab._dtag`, never by a rounded float:
    `alpha_target_bias` needs `delta * n_sub` to be an exact integer, so
    1/12 must survive as 1/12 and not as 0.083333.
    """
    out = {}
    for r in arms():
        out.setdefault(r["twin"], {})[S._dtag(float(r["delta_agg"]))] = \
            float(r["delta_agg"])
    return {k: sorted(v.values(), reverse=True) for k, v in out.items()}


if __name__ == "__main__":
    sys.exit(main())
