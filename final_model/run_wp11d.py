#!/usr/bin/env python3
"""
run_wp11d.py — WP-11d: scaling behaviour and the crossover point
================================================================

The spec's question: *"On the synthetic twin, vary N (through both arms of
11a) and plot learning curves for four model classes on the same data: the
UDE, the GAM, per-flow AR/ARX, and a constant-TC MFA."*  And then the number
it exists to produce: *"Report the crossover N\\* — the sample size at which
the UDE overtakes each baseline."*

`zinc_scale_lab.py` supplies the three estimators that were not yet fitted
objects and puts all four into one file format.  This driver scores them,
builds the curves, fits the saturation model, and reports N\\*.

Why this cost 30 baseline fits and not 160
------------------------------------------
`COMPUTE_STATUS.md` put WP-11d in the CX3 table at "~160 for the UDE arm
alone, ~15 h".  That estimate assumed the UDE arm had to be fitted.  It does
not: WP-11a already fitted **exactly** the design this package needs -- eleven
`base`-twin arms spanning N = 5 to 469, eight seeds each -- and the spec's
"the same data" is a constraint that makes re-fitting them the wrong thing to
do anyway, since this model is not reproducible across thread counts
(COMPUTE_STATUS flag under WP-6b).  What was missing was the other three
classes on those same records, and those are cheap: the GAM is ~2 min a fit,
the constant-TC MFA ~12 s, the AR ensemble ~2 s.  The row should move out of
the CX3 table.

The two arms, and what a "learning curve" assumes
-------------------------------------------------
N is varied two ways, exactly as WP-11a varied it:

    resolution   span fixed at 39 yr, delta from 10 yr to 1 month
                 N = 5, 9, 21, 40, 79, 157, 469
    length       delta fixed at 1 yr, span 10 / 20 / 39 yr
                 N = 11, 21, 40

A learning curve drawn against N alone assumes those two routes to the same N
are interchangeable.  **They are not, and WP-11a already knew it** -- its
`matchedN` arms exist for this -- so the assumption is *tested* here rather
than made: `part5_matched` puts (delta = 1/4, 10 yr) against (delta = 1, 39
yr) at N = 41 vs 40, and (delta = 1/12, 13 yr) against (delta = 1/4, 39 yr) at
N = 157 both ways.  If the two disagree, N\\* is a statement about one arm and
must be reported as such.  The short-span arms also sit entirely before every
turning point in the twin's truth (1997-2008), which is why WP-11a's `common`
scoring region exists and why it is carried through here as a third split.

Three splits, and which one answers which half
----------------------------------------------
WP-11a's scorer is used unchanged, so the three regions mean what they mean
there: `fitted` (to the trainval boundary), `test` (past it), `common` (to
1987, the latest year every arm was fitted on).  The spec's two halves map on
as: **forecast value** = `test` on stocks and flows; **understanding value** =
`fitted` on alpha, which is where a coefficient estimate is a coefficient
estimate rather than an extrapolation.

Family aggregates, stated rather than assumed
---------------------------------------------
The scorer returns one relRMSE per channel.  Both the channel **mean** and the
channel **median** are carried through every table, because they are not close
for this comparison and the difference is itself informative: the AR ensemble's
Refined stock error runs to four digits (WP-1f's mass-imbalance result,
reproduced here on the twin), so a mean is dominated by one channel and a
median hides how badly that channel fails.  N\\* is reported under both.

Parts
-----
  1  fit         gam / constTC / constTC_B / arx on every arm  (~50 min)
  2  score       every dump, all four classes, one scorer      (seconds)
  3  curves      `wp11d_scaling.csv`                           (seconds)
  4  saturation  err(N) = a + b N^-c, asymptote with CI
  5  matched     the N-is-not-enough control
  6  crossover   `wp11d_crossover.csv` -- the N* table
  7  figures

    python run_wp11d.py --check
    python run_wp11d.py --fit
    python run_wp11d.py --score --curves --saturation --matched --crossover --figures
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import zinc_alpha_lab as lab
import zinc_freq_lab as F
import zinc_scale_lab as SC
import run_wp11a as RA

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "analysis")
FIT_DIR = SC.FIT_DIR

COL = {"ude": "#0072B2", "gam": "#009E73", "arx": "#D55E00",
       "constTC": "#CC79A7", "constTC_B": "#E69F00"}
MARK = {"ude": "o", "gam": "s", "arx": "^", "constTC": "D", "constTC_B": "v"}
LABEL = {"ude": "UDE", "gam": "GAM", "arx": "per-flow AR/ARX",
         "constTC": "constant-TC MFA", "constTC_B": "constant-TC MFA (+Stage B)"}
GREY = "#555555"

ALPHA_NAMES = SC.ALPHA_NAMES
STOCK_NAMES = SC.STOCK_NAMES

# The two headline readouts, in the spec's own division of the question.
READOUTS = (
    ("alpha", "fitted", "understanding value — alpha recovery vs truth"),
    ("stock", "test", "forecast value — stocks, held-out window"),
    ("flow", "test", "forecast value — flows, held-out window"),
    ("tau", "fitted", "understanding value — tau recovery vs truth"),
)

# Everything except the UDE is deterministic given the record, so a single
# "seed" is all they have; the UDE carries WP-11a's eight.
BASELINES = ("gam", "arx", "constTC", "constTC_B")


# ===========================================================================
# part 1 — fit
# ===========================================================================
STATUS_JSONL = os.path.join(SC.OUT_DIR, "wp11d_fit_status.jsonl")


def _append_status(row, path=STATUS_JSONL):
    """One JSON line per fit, appended and fsynced.

    The sweep is run as several concurrent processes over disjoint arm lists,
    so a single CSV rewritten at the end of each process would race and the
    slowest writer would win.  Append-and-fsync is the project's own pattern
    for exactly this (CLAUDE.md rule 7) and it makes the record survive a kill
    partway through as well.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps({k: (None if isinstance(v, float) and np.isnan(v)
                                 else v) for k, v in row.items()},
                            default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def collect_status(out_dir=OUT_DIR, path=STATUS_JSONL):
    """Fold the JSONL mirror into `wp11d_fit_status.csv`, last write wins."""
    if not os.path.exists(path):
        return pd.DataFrame()
    rows = [json.loads(l) for l in open(path) if l.strip()]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["model", "tag", "seed"], keep="last")
        df = df.sort_values(["model", "n_obs"])
    csv = os.path.join(out_dir, "wp11d_fit_status.csv")
    df.to_csv(csv, index=False)
    print(f"[status] {len(df)} fits -> {csv}", flush=True)
    return df


def part1_fit(models=BASELINES, tags=None, seeds=(0,), out_dir=OUT_DIR,
              skip_existing=True):
    """Fit the baseline classes on every arm.  Records every outcome."""
    tt = list(tags) if tags else SC.tags()
    n = 0
    t0 = time.time()
    for model in models:
        for tag in tt:
            for sd in seeds:
                if model == "arx":
                    _p, st = SC.fit_arx(tag, seed=int(sd),
                                        skip_existing=skip_existing)
                else:
                    _p, st = SC.fit_baseline(tag, model, seed=int(sd),
                                             skip_existing=skip_existing)
                _append_status(dict(st, model=model, tag=tag, seed=int(sd)))
                n += 1
    print(f"[fit] {n} fits in {(time.time()-t0)/60:.1f} min", flush=True)
    return collect_status(out_dir)


def part1b_determinism(tag="wp11a_freq_1y", model="gam", seeds=(0, 1, 2),
                       out_dir=OUT_DIR):
    """Is the GAM actually deterministic, as the project's convention assumes?

    `CLAUDE.md` says to plot the GAM as a horizontal reference line and never
    as a paired bar, which is a claim about its variance.  Here it is measured
    rather than inherited: the same arm is fitted under three seeds and the
    predictions compared.  If they differ, every baseline curve in this package
    needs a band and the convention needs revisiting.
    """
    paths = []
    for sd in seeds:
        p, _st = SC.fit_baseline(tag, model, seed=int(sd), skip_existing=True)
        paths.append(p)
    ref = np.load(paths[0], allow_pickle=True)
    rows = []
    for p, sd in zip(paths[1:], seeds[1:]):
        d = np.load(p, allow_pickle=True)
        for key in ("alpha_pred_B_at_obs", "S_pred_B", "F_pred_B"):
            a, b = np.asarray(ref[key], float), np.asarray(d[key], float)
            den = np.maximum(np.abs(a), 1e-12)
            rows.append(dict(tag=tag, model=model, seed=int(sd), array=key,
                             max_abs_diff=float(np.nanmax(np.abs(a - b))),
                             max_rel_diff=float(np.nanmax(np.abs(a - b) / den))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "wp11d_determinism.csv"), index=False)
    if not df.empty:
        print(f"[determinism] {model} max rel diff across seeds "
              f"{df.max_rel_diff.max():.3e}", flush=True)
    return df


# ===========================================================================
# part 2 — score
# ===========================================================================
def _model_of(path):
    """`ude` for a WP-11a dump, otherwise the token in the WP-11d filename."""
    base = os.path.basename(path)
    if "__" not in base:
        return "ude"
    return base.split("__", 1)[1].rsplit("_seed", 1)[0]


def part2_score(out_dir=OUT_DIR):
    """Every fit of every class through WP-11a's scorer, unchanged.

    Using `run_wp11a.score_one` rather than a second scorer is deliberate: the
    annual comparison grid, the three reference columns and the region
    definitions are the ones WP-11a's operator self-test already verified, and
    a learning curve built on a different grid would not be comparable to the
    frequency curve it sits next to.
    """
    rows = []
    for r in SC.arms():
        tag = r["tag"]
        paths = SC.ude_fit_paths(tag) + [
            SC.fit_path(tag, m, 0) for m in BASELINES]
        for p in paths:
            if not os.path.exists(p):
                continue
            model = _model_of(p)
            for row in RA.score_one(p):
                rows.append(dict(row, model=model, curve_family=r["family"]))
    per_seed = pd.DataFrame(rows)
    if per_seed.empty:
        print("[score] no fits found", flush=True)
        return per_seed
    per_seed.to_csv(os.path.join(out_dir, "wp11d_per_seed.csv"), index=False)
    print(f"[score] {len(per_seed)} rows, "
          f"{per_seed.model.nunique()} classes -> wp11d_per_seed.csv", flush=True)
    return per_seed


# ===========================================================================
# part 3 — the learning curves
# ===========================================================================
def _agg_over_channels(df, how):
    f = np.nanmean if how == "mean" else np.nanmedian
    return f(np.asarray(df["pred_vs_true"], float))


def part3_curves(per_seed, out_dir=OUT_DIR):
    """One row per (curve, model, family, split, agg, N): median +- IQR of seeds.

    The channel aggregate is taken **within** a seed and the seed distribution
    formed from those, so the IQR is across fits and not across channels.
    """
    rows = []
    keys = ["curve_family", "model", "family", "split", "tag", "n_obs",
            "delta", "span", "seed"]
    for how in ("mean", "median"):
        g = per_seed.groupby(keys, as_index=False, dropna=False)
        part = g.apply(lambda d: _agg_over_channels(d, how),
                       include_groups=False)
        # `groupby.apply` returning a scalar names the value column `None`
        # on pandas 2.x; position is stable, the label is not.
        part.columns = list(part.columns[:-1]) + ["err"]
        part["agg"] = how
        rows.append(part)
    per_fit = pd.concat(rows, ignore_index=True)
    per_fit.to_csv(os.path.join(out_dir, "wp11d_per_fit.csv"), index=False)

    ck = ["curve_family", "model", "family", "split", "agg", "tag", "n_obs",
          "delta", "span"]
    curve = per_fit.groupby(ck, as_index=False).agg(
        n_seeds=("seed", "nunique"),
        err=("err", "median"),
        err_q1=("err", lambda v: v.quantile(0.25)),
        err_q3=("err", lambda v: v.quantile(0.75)))
    curve = curve.sort_values(ck[:5] + ["n_obs"]).reset_index(drop=True)
    curve.to_csv(os.path.join(out_dir, "wp11d_scaling.csv"), index=False)
    print(f"[curves] {len(curve)} curve points -> wp11d_scaling.csv", flush=True)
    return per_fit, curve


# ===========================================================================
# part 4 — saturation
# ===========================================================================
def _sat_model(N, a, b, c):
    """`err(N) = a + b N^-c`: an asymptote `a` the estimator cannot beat."""
    return a + b * np.power(N, -c)


def fit_saturation(N, y, n_boot=400, rng=None):
    """Least squares on `a + b N^-c`, with a bootstrap CI on the asymptote.

    Three points cannot determine three parameters, so a curve with fewer than
    four N values returns NaN rather than a number: an asymptote read off three
    points is not a measurement and the `length` arm has exactly three.
    """
    from scipy.optimize import curve_fit

    N = np.asarray(N, float)
    y = np.asarray(y, float)
    ok = np.isfinite(N) & np.isfinite(y)
    N, y = N[ok], y[ok]
    out = dict(n_points=int(N.size), a=np.nan, b=np.nan, c=np.nan,
               a_lo=np.nan, a_hi=np.nan, rmse=np.nan, converged=False)
    if N.size < 4:
        return out
    p0 = [max(float(np.min(y)) * 0.5, 1e-6), float(np.max(y)), 0.5]
    bounds = ([0.0, 0.0, 0.0], [np.inf, np.inf, 5.0])
    try:
        p, _cov = curve_fit(_sat_model, N, y, p0=p0, bounds=bounds, maxfev=20000)
    except Exception:
        return out
    out.update(a=float(p[0]), b=float(p[1]), c=float(p[2]), converged=True,
               rmse=float(np.sqrt(np.mean((y - _sat_model(N, *p)) ** 2))))

    # residual bootstrap: the curve is a fit to K medians, not to K samples,
    # so the honest uncertainty is on the curve's own residuals.
    rng = np.random.default_rng(0 if rng is None else rng)
    res = y - _sat_model(N, *p)
    boot = []
    for _ in range(int(n_boot)):
        yb = _sat_model(N, *p) + rng.choice(res, size=res.size, replace=True)
        try:
            pb, _ = curve_fit(_sat_model, N, yb, p0=p, bounds=bounds, maxfev=20000)
            boot.append(pb[0])
        except Exception:
            continue
    if len(boot) >= 20:
        out["a_lo"] = float(np.percentile(boot, 2.5))
        out["a_hi"] = float(np.percentile(boot, 97.5))
    return out


def part4_saturation(curve, out_dir=OUT_DIR):
    """`a + b N^-c` per (curve, class, family, split, agg).

    The annual full-span arm belongs to both curves (`resolution+length`), so
    the subsets are built the way the crossover builds them rather than by
    grouping on `curve_family`, which would strand it in a group of one.
    """
    rows = []
    for cf in ("resolution", "length"):
        sub = curve[curve.curve_family.isin([cf, "resolution+length"])]
        for (model, fam, split, agg), d in sub.groupby(
                ["model", "family", "split", "agg"]):
            d = d.sort_values("n_obs")
            r = fit_saturation(d["n_obs"], d["err"])
            rows.append(dict(curve_family=cf, model=model, family=fam,
                             split=split, agg=agg,
                             n_min=int(d["n_obs"].min()),
                             n_max=int(d["n_obs"].max()),
                             err_at_nmin=float(d["err"].iloc[0]),
                             err_at_nmax=float(d["err"].iloc[-1]), **r))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "wp11d_saturation.csv"), index=False)
    print(f"[saturation] {len(df)} fits -> wp11d_saturation.csv", flush=True)
    return df


# ===========================================================================
# part 5 — the matched-N control
# ===========================================================================
def part5_matched(per_fit, out_dir=OUT_DIR):
    """Does N alone predict the error, or does it matter how N was reached?

    Two pairs at (near) equal N, one reached by resolution and one by record
    length.  A learning curve against N assumes the members of a pair agree;
    the assumption is reported, not made.
    """
    pairs = (("wp11a_matchedN_0p25y_s10y", "wp11a_freq_1y", 41, 40),
             ("wp11a_matchedN_0p083y_s13y", "wp11a_freq_0p25y", 157, 157))
    rows = []
    for fine, coarse, n_f, n_c in pairs:
        for (model, fam, split, agg), d in per_fit.groupby(
                ["model", "family", "split", "agg"]):
            a = d[d.tag == fine]["err"]
            b = d[d.tag == coarse]["err"]
            if a.empty or b.empty:
                continue
            ea, eb = float(np.nanmedian(a)), float(np.nanmedian(b))
            rows.append(dict(
                pair=f"{fine} vs {coarse}", short_span_tag=fine,
                long_span_tag=coarse, n_short=n_f, n_long=n_c,
                model=model, family=fam, split=split, agg=agg,
                err_short_span=ea, err_long_span=eb,
                ratio=(ea / eb) if eb > 1e-12 else np.nan,
                n_seeds_short=int(a.size), n_seeds_long=int(b.size)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "wp11d_matched_n.csv"), index=False)
    print(f"[matched] {len(df)} comparisons -> wp11d_matched_n.csv", flush=True)
    return df


# ===========================================================================
# part 6 — the crossover
# ===========================================================================
def _cross_log(N, y_ude, y_base):
    """Smallest N at which the UDE curve drops below the baseline's.

    Solved on `(log N, log err)` by linear interpolation between the bracketing
    observed points -- no functional form, so a crossing is a property of the
    measured curve.  Returns `(N_star, status)` where status is one of
    `crossed`, `always_below` (the UDE wins at the smallest N measured),
    `never` (it does not win by the largest N measured).
    """
    N = np.asarray(N, float)
    d = np.asarray(y_ude, float) - np.asarray(y_base, float)
    ok = np.isfinite(N) & np.isfinite(d)
    N, d = N[ok], d[ok]
    order = np.argsort(N)
    N, d = N[order], d[order]
    if N.size < 2:
        return np.nan, "insufficient"
    if d[0] < 0:
        return float(N[0]), "always_below"
    for i in range(1, N.size):
        if d[i] < 0:
            x0, x1 = np.log(N[i - 1]), np.log(N[i])
            f = d[i - 1] / (d[i - 1] - d[i])
            return float(np.exp(x0 + f * (x1 - x0))), "crossed"
    return np.nan, "never"


def _stays_below(N, y_ude, y_base, n_star):
    """Does the UDE stay ahead once it is ahead, or does the lead not hold?

    A first crossing is what the spec asks for, but a curve that crosses and
    re-crosses is not a capacity story and the table should say so.
    """
    N = np.asarray(N, float)
    d = np.asarray(y_ude, float) - np.asarray(y_base, float)
    ok = np.isfinite(N) & np.isfinite(d)
    N, d = N[ok], d[ok]
    if not np.isfinite(n_star) or N.size == 0:
        return False
    after = d[N >= n_star - 1e-9]
    return bool(after.size > 0 and np.all(after < 0))


def part6_crossover(per_fit, curve, out_dir=OUT_DIR):
    """N* per (baseline, family, split, agg), with a seed-resampled CI.

    The point estimate uses the UDE's median curve.  The interval comes from
    re-computing the crossing on each of the eight seeds' own curves, which is
    the only source of sampling variation the design has -- the baselines are
    deterministic given the record (part 1b measures that), so all of the
    spread in N* is the UDE's initialisation.
    """
    rows = []
    for cf in ("resolution", "length"):
        sub = curve[curve.curve_family.isin([cf, "resolution+length"])]
        pf = per_fit[per_fit.curve_family.isin([cf, "resolution+length"])]
        for (fam, split, agg), d in sub.groupby(["family", "split", "agg"]):
            ude = d[d.model == "ude"].sort_values("n_obs")
            if ude.empty:
                continue
            for base in BASELINES:
                b = d[d.model == base].sort_values("n_obs")
                if b.empty:
                    continue
                common = sorted(set(ude.n_obs) & set(b.n_obs))
                if len(common) < 2:
                    continue
                yu = [float(ude[ude.n_obs == n].err.iloc[0]) for n in common]
                yb = [float(b[b.n_obs == n].err.iloc[0]) for n in common]
                nstar, status = _cross_log(common, yu, yb)

                per_seed_n = []
                for sd, ds in pf[(pf.family == fam) & (pf["split"] == split)
                                 & (pf["agg"] == agg) & (pf.model == "ude")
                                 ].groupby("seed"):
                    yu_s = []
                    for n in common:
                        v = ds[ds.n_obs == n]["err"]
                        yu_s.append(float(v.iloc[0]) if not v.empty else np.nan)
                    ns, st = _cross_log(common, yu_s, yb)
                    if st in ("crossed", "always_below"):
                        per_seed_n.append(ns)
                        _ = sd
                psn = np.asarray(per_seed_n, float)
                rows.append(dict(
                    curve_family=cf, baseline=base, family=fam, split=split,
                    agg=agg, n_star=nstar, status=status,
                    stays_below=_stays_below(common, yu, yb, nstar),
                    n_star_seed_min=float(psn.min()) if psn.size else np.nan,
                    n_star_seed_q1=float(np.percentile(psn, 25)) if psn.size else np.nan,
                    n_star_seed_q3=float(np.percentile(psn, 75)) if psn.size else np.nan,
                    n_star_seed_max=float(psn.max()) if psn.size else np.nan,
                    n_seeds_crossing=len(per_seed_n),
                    n_seeds_total=int(pf[(pf.family == fam) & (pf["split"] == split)
                                         & (pf["agg"] == agg)
                                         & (pf.model == "ude")].seed.nunique()),
                    n_min=int(min(common)), n_max=int(max(common)),
                    ude_at_nmax=float(ude[ude.n_obs == max(common)].err.iloc[0]),
                    base_at_nmax=float(b[b.n_obs == max(common)].err.iloc[0])))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "wp11d_crossover.csv"), index=False)
    print(f"[crossover] {len(df)} entries -> wp11d_crossover.csv", flush=True)
    return df


# ===========================================================================
# part 7 — figures
# ===========================================================================
def figures(curve, cross, out_dir=OUT_DIR, agg="median"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [(f, s, t) for f, s, t in READOUTS]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2), sharex=True)
    for ax, (fam, split, title) in zip(axes.ravel(), panels):
        # tau is the one family the channel median cannot show: five of the
        # eight TAU_SUP_NAMES slots are pinned to the data, which on the twin
        # reproduces the truth exactly, so the median sits on a zero and a log
        # axis collapses.  That panel is drawn on the channel mean and says so.
        a = "mean" if fam == "tau" else agg
        if a != agg:
            title = f"{title}  [channel mean]"
        d = curve[(curve.family == fam) & (curve["split"] == split)
                  & (curve["agg"] == a)
                  & (curve.curve_family.isin(["resolution", "resolution+length"]))]
        for model in ("ude",) + BASELINES:
            m = d[d.model == model].sort_values("n_obs")
            if m.empty:
                continue
            ax.plot(m.n_obs, m.err, marker=MARK[model], ms=5, lw=1.6,
                    color=COL[model], label=LABEL[model],
                    ls="-" if model == "ude" else "--")
            if model == "ude" and m.n_seeds.max() > 1:
                ax.fill_between(m.n_obs, m.err_q1, m.err_q3, color=COL[model],
                                alpha=0.18, lw=0)
        cs = cross[(cross.family == fam) & (cross["split"] == split)
                   & (cross["agg"] == a) & (cross.curve_family == "resolution")
                   & (cross.status == "crossed")]
        for _, r in cs.iterrows():
            ax.axvline(r.n_star, color=COL[r.baseline], lw=0.9, ls=":", alpha=0.8)
        ax.axvline(40, color=GREY, lw=0.9, alpha=0.6)
        ax.text(40, ax.get_ylim()[1], " N=40 (real record)", fontsize=7,
                color=GREY, va="top", rotation=90)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=9.5)
        ax.set_ylabel("relRMSE vs truth  [%]", fontsize=8.5)
        ax.grid(alpha=0.25, which="both", lw=0.4)
        ax.tick_params(labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("N  (effective observations, resolution arm)", fontsize=8.5)
    axes[0, 0].legend(fontsize=7.5, frameon=False, loc="best")
    fig.suptitle("WP-11d — learning curves on the synthetic twin "
                 f"(channel {agg}; span fixed at 39 yr, N varied by resolution)",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"wp11d_scaling.{ext}"), dpi=200)
    plt.close(fig)
    print(f"[figures] -> wp11d_scaling.png/pdf", flush=True)


def figure_arms(curve, out_dir=OUT_DIR, agg="median"):
    """The two arms side by side: is N reached by resolution the same as by span?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    for ax, (fam, split, title) in zip(axes, READOUTS[:3]):
        for cf, ls, mk in (("resolution", "-", "o"), ("length", "--", "s")):
            d = curve[(curve.family == fam) & (curve["split"] == split)
                      & (curve["agg"] == agg) & (curve.model == "ude")
                      & (curve.curve_family.isin([cf, "resolution+length"]))]
            d = d.sort_values("n_obs")
            if d.empty:
                continue
            ax.plot(d.n_obs, d.err, ls=ls, marker=mk, ms=5, lw=1.6,
                    color=COL["ude"] if cf == "resolution" else GREY,
                    label=f"UDE, {cf}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("N", fontsize=8.5)
        ax.grid(alpha=0.25, which="both", lw=0.4)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("relRMSE vs truth [%]", fontsize=8.5)
    axes[0].legend(fontsize=7.5, frameon=False)
    fig.suptitle("WP-11d — the same N reached two ways", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"wp11d_arms.{ext}"), dpi=200)
    plt.close(fig)
    print("[figures] -> wp11d_arms.png/pdf", flush=True)


# ===========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="WP-11d driver")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--determinism", action="store_true")
    ap.add_argument("--models", default=",".join(BASELINES))
    ap.add_argument("--tags", default=None)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--curves", action="store_true")
    ap.add_argument("--saturation", action="store_true")
    ap.add_argument("--matched", action="store_true")
    ap.add_argument("--crossover", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args(argv)

    if args.check or not any(vars(args).values()):
        SC.check()
        return 0
    lab.integrity_check()

    if args.fit:
        part1_fit(models=tuple(m.strip() for m in args.models.split(",")),
                  tags=[t.strip() for t in args.tags.split(",")] if args.tags
                  else None)
    if args.determinism:
        part1b_determinism()

    per_seed = curve = per_fit = cross = None
    if args.score or args.all:
        per_seed = part2_score()
    if any([args.curves, args.saturation, args.matched, args.crossover,
            args.figures, args.all]):
        if per_seed is None:
            per_seed = pd.read_csv(os.path.join(OUT_DIR, "wp11d_per_seed.csv"))
        per_fit, curve = part3_curves(per_seed)
    if args.saturation or args.all:
        part4_saturation(curve)
    if args.matched or args.all:
        part5_matched(per_fit)
    if args.crossover or args.all or args.figures:
        cross = part6_crossover(per_fit, curve)
    if args.figures or args.all:
        figures(curve, cross)
        figure_arms(curve)
    return 0


if __name__ == "__main__":
    sys.exit(main())
