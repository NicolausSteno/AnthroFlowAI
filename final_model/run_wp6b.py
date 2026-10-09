#!/usr/bin/env python3
"""
run_wp6b.py — WP-6b, temporal coarsening of the observed record
================================================================

*How far does coefficient identifiability degrade when the zinc record is
reported every 2nd / 5th / 10th year instead of annually?*

Driver for `zinc_coarse_lab.py`, which owns the observation operator and the
single import-time patch (CLAUDE.md rule 1).  Five parts, in increasing cost:

  1  **reach**       which of the 30 endogenous series can move the anchor_v4
                     objective at all — no fits, and it decides the grid
  2  **targets**     how far coarsening moves `alpha_obs` itself, per series
                     and per Delta — no fits, the WP-11b observation-side
                     statistic carried down to individual series
  3  **curve**       the refit curve: the whole record coarsened together,
                     Delta in {1, 2, 5, 10} x 8 seeds = 32 fits.  This is the
                     object WP-11b's gate is waiting for
  4  **gate**        hand the curve to `run_wp11b.load_refit_curve` and report
                     what it does to the verdict
  5  **grid**        the per-series design (`reach` x 3 levels x 8 seeds)
                     written out as a manifest + PBS array for CX3
  6  **matrix**      score that grid once it has landed — a no-op until then

Parts 1, 2, 4 and 5 need no fits and run in minutes.  Part 3 is the expensive
one and is the only part that has to be run before the others mean anything.

What is scored, and against what
--------------------------------
Every arm is fitted against its own degraded record and **scored against the
annual record**.  Keeping those two apart is the whole design: WP-11b already
showed that coarsening moves `alpha_obs` itself by 3-45% before any estimator
is involved, so a curve scored against the coarse target would confound the
estimator's degradation with the target's.  Part 2 reports the target's
movement separately, in the same units, so the two can be read side by side.

The headline metric is `alpha_relRMSE_pct` on the **trainval** span — the
region the fit actually saw — because the question is identifiability, not
extrapolation.  The test-window version is written alongside
(`wp6b_coarsening_test.csv`) and the free-run stock errors go to
`wp6b_stocks.csv`; nothing is selected on any of them (CLAUDE.md rule 3).

CLI
---
    python run_wp6b.py --check
    python run_wp6b.py --part 1,2          # no fits
    python run_wp6b.py --fit --seeds 0-7   # part 3, 32 fits
    python run_wp6b.py --part 3,4,5
    python run_wp6b.py --part 6            # after pbs/wp6b_grid.pbs has run        # aggregate + gate + CX3 grid
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import zinc_alpha_lab as lab
import zinc_coarse_lab as cl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "analysis")
FIT_DIR = os.path.join(OUT_DIR, "wp6b")

ALPHA_NAMES = ("alpha_cc", "alpha_refc", "alpha_win", "alpha_dr")
DELTAS = cl.DELTAS
N_SEEDS = 8                     # the spec's ablation-grid seed count (WP-6 note)
GATE_SPLIT = "trainval"


def _rel_rmse_pct(pred, obs, eps=1e-12):
    """100 * RMSE / mean|obs|, over jointly finite entries.

    Identical to `zinc_colloc_v5._rel_rmse_pct` (:2589) and to WP-1a's, so the
    Delta = 1 arm is directly comparable to `wp1a_alpha_channels.csv`.
    """
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    m = np.isfinite(pred) & np.isfinite(obs)
    if not m.any():
        return float("nan")
    den = float(np.mean(np.abs(obs[m])))
    if den < eps:
        return float("nan")
    return 100.0 * float(np.sqrt(np.mean((pred[m] - obs[m]) ** 2))) / den


def _hl(x):
    """Hodges-Lehmann location: median of pairwise means (project convention)."""
    x = np.asarray([v for v in np.asarray(x, float).ravel() if np.isfinite(v)])
    if x.size == 0:
        return float("nan")
    if x.size == 1:
        return float(x[0])
    i, j = np.triu_indices(x.size, 0)
    return float(np.median((x[i] + x[j]) / 2.0))


def _iqr_row(x, prefix):
    x = np.asarray([v for v in np.asarray(x, float).ravel() if np.isfinite(v)])
    if x.size == 0:
        return {f"{prefix}_median": np.nan, f"{prefix}_q1": np.nan,
                f"{prefix}_q3": np.nan, f"{prefix}_n": 0}
    return {f"{prefix}_median": float(np.median(x)),
            f"{prefix}_q1": float(np.percentile(x, 25)),
            f"{prefix}_q3": float(np.percentile(x, 75)),
            f"{prefix}_n": int(x.size)}


# ===========================================================================
# Part 1 — which series reach the objective
# ===========================================================================
def part1_reach(out_dir=OUT_DIR, fit_dir=FIT_DIR):
    """Consume `zinc_coarse_lab --reach` and turn it into a statement.

    `anchor_v4` sets `stageB_w_F = 0.0`, so the flow-divergence term is off and
    a flow column reaches the objective only if it is one of the four alpha
    numerators or the pinned `cp` forcing.  This part checks that empirically
    rather than reading it off the config: a series whose coarsening moves
    neither the Stage A nor the Stage B loss, at fixed parameters, has no path
    into the objective and refitting it would return `anchor_v4` exactly.
    """
    src = os.path.join(fit_dir, "reach.csv")
    if not os.path.exists(src):
        raise SystemExit(f"missing {src} — run `python zinc_coarse_lab.py --reach` first")
    df = pd.read_csv(src)
    df["kind"] = df.series.str.split(":").str[0]
    df["reaches_stageA"] = df.dA > 0
    df["reaches_stageB"] = df.dB > 0
    df["reaches_loss"] = df.reaches_stageA | df.reaches_stageB
    df["route"] = np.where(
        df.reaches_stageA & df.reaches_stageB, "alpha/tau target + dynamics",
        np.where(df.reaches_stageB, "dynamics only",
                 np.where(df.reaches_stageA, "target only", "none")))
    df = df.sort_values(["reaches_loss", "dB"], ascending=[False, False])
    df.to_csv(os.path.join(out_dir, "wp6b_reach.csv"), index=False)
    return df


# ===========================================================================
# Part 2 — what coarsening does to the target, before any estimator
# ===========================================================================
def part2_targets(out_dir=OUT_DIR, cfg=None):
    """Per-series, per-Delta movement of `alpha_obs` and of the record itself.

    Reported on the **trainval** span (rows 0..val_end), because that is the
    only region the loss ever sees and a movement in the test window cannot
    degrade a fit.  `alpha_target_relRMSE_pct` is the coarse target against
    the annual target at the rows where the coarse target exists — the
    supervision bias the fit is handed.  It is *not* WP-11b's
    `degradation_pct`, which compared two constructions on the same window;
    this compares a Delta-year window average, placed at the window's closing
    year by the core's year-end convention (`zinc_colloc_v5.py:846`), against
    the annual value at that year.  The phase offset of ~Delta/2 years is part
    of what a coarse record costs and is deliberately not corrected out.
    """
    import zinc_colloc_v5 as v5
    cfg = dict(cfg or lab.load_anchor_config())
    ref = cl.raw_data(cfg)
    universe = cl.series_universe(cfg=cfg)

    T = len(ref["years"])
    tv_end = _trainval_end(cfg, T)
    rows = []
    for delta in DELTAS:
        for series in (["all"] + [[s] for s in universe]):
            tag = "all" if series == "all" else series[0]
            c = cl.coarsen(ref, delta, series, cfg=cfg)
            a_c = np.asarray(c["alpha_obs"], float)[:tv_end]
            a_r = np.asarray(ref["alpha_obs"], float)[:tv_end]
            row = dict(series=tag, delta=int(delta),
                       n_alpha_rows_trainval=int(np.isfinite(a_c[:, 0]).sum()))
            for k, nm in enumerate(ALPHA_NAMES):
                row[f"{nm}_target_relRMSE_pct"] = _rel_rmse_pct(a_c[:, k], a_r[:, k])
            # how far the record itself moved (finite entries only)
            for key, lbl in (("stocks_obs", "stock"), ("flows_obs", "flow"),
                             ("tau_sup_obs", "tau"), ("cp_obs", "cp")):
                g = np.asarray(c[key], float)
                b = np.asarray(ref[key], float)
                n = min(tv_end, g.shape[0])
                row[f"{lbl}_record_relRMSE_pct"] = _rel_rmse_pct(g[:n], b[:n])
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "wp6b_target_degradation.csv"), index=False)
    return df


def _trainval_end(cfg, T):
    """Row index one past the last year the loss can see (`:1880-1892`)."""
    import zinc_colloc_v5 as v5
    c = dict(v5.DEFAULT_CONFIG, **dict(cfg))
    n_trainval = min(max(int(np.floor(c["trainval_frac"] * T)), 3), T - 1)
    return n_trainval


# ===========================================================================
# Part 3 — the refit curve
# ===========================================================================
def fit_curve(seeds, deltas=DELTAS, fit_dir=FIT_DIR, cfg=None, verbose=False,
              skip_existing=True):
    """The `all`-series arm at each Delta.  This is the expensive part."""
    cfg = dict(cfg or lab.load_anchor_config())
    os.makedirs(fit_dir, exist_ok=True)
    for delta in deltas:
        for sd in seeds:
            cl.fit_arm(sd, delta, "all", out_dir=fit_dir, cfg=cfg,
                       verbose=verbose, skip_existing=skip_existing)


def part3_curve(seeds=range(N_SEEDS), out_dir=OUT_DIR, fit_dir=FIT_DIR, cfg=None):
    """Score every fitted arm against the **annual** record and aggregate."""
    cfg = dict(cfg or lab.load_anchor_config())
    per_seed, missing = [], []
    for delta in DELTAS:
        tag = cl.arm_tag(delta, "all")
        for sd in seeds:
            p = os.path.join(fit_dir, f"{tag}_seed{sd}.npz")
            if not os.path.exists(p):
                missing.append(p)
                continue
            d = np.load(p, allow_pickle=True)
            masks = dict(train=d["mask_train"], val=d["mask_val"],
                         test=d["mask_test"])
            masks["trainval"] = masks["train"] | masks["val"]
            a_pred = d["alpha_pred_B_at_obs"]
            a_ann = d["alpha_obs_annual"]
            a_arm = d["alpha_obs_arm"]
            for split, m in masks.items():
                for k, nm in enumerate(ALPHA_NAMES):
                    per_seed.append(dict(
                        seed=int(sd), delta=int(delta), channel=nm, split=split,
                        alpha_relRMSE_pct=_rel_rmse_pct(a_pred[m, k], a_ann[m, k]),
                        alpha_relRMSE_vs_arm_pct=_rel_rmse_pct(a_pred[m, k],
                                                               a_arm[m, k]),
                        n_obs_supervised=int(np.isfinite(a_arm[m, k]).sum()),
                    ))
            # stock free-run error, annual reference
            S_p, S_o = d["S_pred_B"], d["S_obs_annual"]
            for split, m in masks.items():
                for k, nm in enumerate(cl.STOCK_NAMES):
                    per_seed.append(dict(
                        seed=int(sd), delta=int(delta), channel=f"stock:{nm}",
                        split=split,
                        alpha_relRMSE_pct=_rel_rmse_pct(S_p[m, k], S_o[m, k]),
                        alpha_relRMSE_vs_arm_pct=np.nan, n_obs_supervised=-1))
    if missing:
        print(f"[part3] {len(missing)} arm files missing, e.g. {missing[0]}")
    if not per_seed:
        raise SystemExit("[part3] no fitted arms found — run --fit first")

    ps = pd.DataFrame(per_seed)
    ps.to_csv(os.path.join(out_dir, "wp6b_per_seed.csv"), index=False)

    # aggregate, paired by seed against the Delta = 1 arm of the same seed
    base = ps[ps.delta == 1].set_index(["seed", "channel", "split"])[
        "alpha_relRMSE_pct"]
    ps = ps.join(base.rename("base_relRMSE_pct"),
                 on=["seed", "channel", "split"])
    ps["delta_pp"] = ps.alpha_relRMSE_pct - ps.base_relRMSE_pct

    agg = []
    for (delta, ch, split), sub in ps.groupby(["delta", "channel", "split"]):
        row = dict(delta=int(delta), channel=ch, split=split,
                   n_seeds=int(sub.seed.nunique()),
                   n_obs_supervised=int(sub.n_obs_supervised.median()))
        row.update(_iqr_row(sub.alpha_relRMSE_pct, "relRMSE"))
        row.update(_iqr_row(sub.delta_pp, "paired_delta_pp"))
        row["paired_HL_pp"] = _hl(sub.delta_pp)
        row["n_worse"] = int((sub.delta_pp > 0).sum())
        agg.append(row)
    ag = pd.DataFrame(agg).sort_values(["split", "channel", "delta"])
    ag.to_csv(os.path.join(out_dir, "wp6b_curve_summary.csv"), index=False)

    # the two files WP-11b's loader reads: long-form (channel, delta, metric)
    for split, name in ((GATE_SPLIT, "wp6b_coarsening.csv"),
                        ("test", "wp6b_coarsening_test.csv")):
        sub = ps[(ps.split == split) & (ps.channel.isin(ALPHA_NAMES))]
        sub[["channel", "delta", "seed", "split", "alpha_relRMSE_pct"]].to_csv(
            os.path.join(out_dir, name), index=False)

    stocks = ag[ag.channel.str.startswith("stock:")]
    stocks.to_csv(os.path.join(out_dir, "wp6b_stocks.csv"), index=False)
    return ps, ag


def verify_delta1(seeds=(0, 1), fit_dir=FIT_DIR, out_dir=OUT_DIR):
    """The Delta = 1 arm must reproduce the unpatched model — check, don't assume.

    Two comparisons, and the distinction between them matters.

    `vs_control` fits the **unpatched** core (`zinc_alpha_lab.fit_seed`) in the
    *same process environment* as the coarsening arms and compares.  This is
    the comparison that tests the patch, and it must be exactly 0.0: the
    dispatcher at Delta = 1 returns the loader's own arrays bit-for-bit
    (`zinc_coarse_lab --noop`), so the fit downstream of it cannot differ.

    `vs_wp1a` compares against the published `analysis/wp1a/alpha_seed{n}.npz`,
    which was fitted on this machine under different thread settings.  It is
    **not** expected to be bit-exact and is reported for completeness only:
    4 000 Adam steps through a stiff ODE amplify the reduction-order
    differences that come with a different `OMP_NUM_THREADS`, and the observed
    gap is ~1e-4 relative.  `max_abs_dalpha_obs` — the *record*, which no
    optimiser touches — is the part that must be 0.0 in both comparisons, and
    it is.
    """
    ctl_dir = os.path.join(fit_dir, "control")
    rows = []
    for sd in seeds:
        a = os.path.join(fit_dir, f"{cl.arm_tag(1, 'all')}_seed{sd}.npz")
        if not os.path.exists(a):
            rows.append(dict(seed=sd, comparison="vs_control", present=False))
            continue
        da = np.load(a, allow_pickle=True)
        for tag, b in (("vs_control", os.path.join(ctl_dir, f"alpha_seed{sd}.npz")),
                       ("vs_wp1a", os.path.join(HERE, "analysis", "wp1a",
                                                f"alpha_seed{sd}.npz"))):
            if not os.path.exists(b):
                rows.append(dict(seed=sd, comparison=tag, present=False))
                continue
            db = np.load(b, allow_pickle=True)
            d_alpha = float(np.nanmax(np.abs(da["alpha_pred_B_at_obs"]
                                             - db["alpha_pred_B_at_obs"])))
            d_S = float(np.nanmax(np.abs(da["S_pred_B"] - db["S_pred_B"])))
            d_obs = float(np.nanmax(np.abs(da["alpha_obs_arm"] - db["alpha_obs"])))
            rows.append(dict(
                seed=sd, comparison=tag, present=True,
                max_abs_dalpha=d_alpha, max_abs_dS=d_S,
                max_abs_dalpha_obs=d_obs,
                rel_dalpha=d_alpha / max(float(np.nanmax(np.abs(
                    db["alpha_pred_B_at_obs"]))), 1e-12),
                rel_dS=d_S / max(float(np.nanmax(np.abs(db["S_pred_B"]))), 1e-12),
                bit_exact=bool(d_alpha == 0.0 and d_S == 0.0 and d_obs == 0.0),
                is_the_guard=(tag == "vs_control")))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "wp6b_reproduction_check.csv"), index=False)
    return df


# ===========================================================================
# Part 4 — hand the curve to WP-11b's gate
# ===========================================================================
def part4_gate(out_dir=OUT_DIR):
    """Load the curve through WP-11b's own loader and report the verdict.

    WP-11b pre-registered the criteria and self-tested them before either
    refit curve existed; its `load_refit_curve` states the schema it expects
    and its flag 5 warns that `GATE_METRIC` was a guess at WP-6b's column
    name.  This calls that loader unchanged — if it reads the file, the guess
    was right and the gate is one command away from closing; the gate itself
    stays OPEN until WP-11a supplies the other curve.
    """
    sys.path.insert(0, HERE)
    import run_wp11b as w11

    path = os.path.join(out_dir, "wp6b_coarsening.csv")
    curve = w11.load_refit_curve(path)
    rows = []
    if curve is None:
        rows.append(dict(check="wp6b_curve_present", ok=False, detail=path))
    else:
        rows.append(dict(check="wp6b_curve_loads_under_wp11b_loader", ok=True,
                         detail=f"{len(curve)} (channel, delta) cells, "
                                f"metric={w11.GATE_METRIC}"))
        for ch, sub in curve.groupby("channel"):
            s = sub.sort_values("delta")
            v = s.value.to_numpy()
            mono = bool(np.all(np.diff(v) >= -w11.GATE_MONOTONE_TOL
                               * np.abs(v[:-1])))
            rows.append(dict(check=f"G1_monotone_in_delta[{ch}]", ok=mono,
                             detail=",".join(f"{x:.3g}" for x in v)))
    other = os.path.join(out_dir, "wp11a_frequency_value.csv")
    rows.append(dict(check="wp11a_curve_present", ok=os.path.exists(other),
                     detail=other))
    rows.append(dict(
        check="wp11b_gate",
        ok=bool(curve is not None and os.path.exists(other)),
        detail="PASS/FAIL computable only when both curves exist; "
               "WP-6b half now supplied, WP-11a still outstanding"))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "wp6b_gate_handoff.csv"), index=False)

    # re-run WP-11b's own part 1 so `wp11b_status.csv` stops saying the
    # WP-6b curve is absent
    try:
        w11.part1_status(out_dir=out_dir)
    except Exception as e:
        print(f"[part4] could not refresh wp11b_status.csv: "
              f"{type(e).__name__}: {e}")
    return df


# ===========================================================================
# Part 5 — the per-series grid for CX3
# ===========================================================================
def part5_grid(reach, out_dir=OUT_DIR, n_seeds=N_SEEDS):
    """Write the per-series design as a manifest the PBS array indexes into.

    The spec's 6b is "same design" as 6a — one arm per series — which
    `COMPUTE_STATUS.md` sized at 3 x ~22 x 8 = 528 fits.  Part 1 cuts that to
    the series that can actually move the objective; the rest are listed with
    `included=False` and the reason, so the reduction is auditable rather than
    silent.
    """
    live = reach[reach.reaches_loss]
    rows = []
    for _, r in reach.iterrows():
        for delta in [d for d in DELTAS if d > 1]:
            rows.append(dict(series=r.series, delta=int(delta),
                             n_seeds=n_seeds, included=bool(r.reaches_loss),
                             route=r.route,
                             reason=("" if r.reaches_loss else
                                     "no path into the anchor_v4 objective "
                                     "(stageB_w_F = 0); refit would reproduce "
                                     "anchor_v4 exactly")))
    man = pd.DataFrame(rows)
    man["task_id"] = np.where(man.included, man.included.cumsum() - 1, -1)
    man.to_csv(os.path.join(out_dir, "wp6b_grid_manifest.csv"), index=False)
    n_arms = int(man.included.sum())
    _stamp_pbs_range(n_arms)
    print(f"[part5] per-series grid: {n_arms} arms x {n_seeds} seeds "
          f"= {n_arms * n_seeds} fits "
          f"({len(man) - n_arms} arms dropped as structurally inert)")
    return man


def _stamp_pbs_range(n_arms):
    """Keep `pbs/wp6b_grid.pbs`'s `-J` range in step with the manifest.

    The array range and the manifest have to agree exactly or elements index
    rows that are not there; regenerating one without the other is the
    obvious way to get a silently short grid, so part 5 writes both.
    """
    import re
    path = os.path.join(HERE, "pbs", "wp6b_grid.pbs")
    if not os.path.exists(path):
        return
    with open(path) as fh:
        txt = fh.read()
    new = re.sub(r"^#PBS -J 0-\S+$", f"#PBS -J 0-{max(n_arms - 1, 0)}",
                 txt, count=1, flags=re.M)
    if new != txt:
        with open(path, "w") as fh:
            fh.write(new)
        print(f"[part5] pbs/wp6b_grid.pbs -J range set to 0-{n_arms - 1}")


# ===========================================================================
# Part 6 — the per-series matrix, once the CX3 grid has landed
# ===========================================================================
def part6_matrix(out_dir=OUT_DIR, fit_dir=FIT_DIR, seeds=range(N_SEEDS)):
    """(series coarsened x variable degraded), the spec's 6a-shaped output.

    Reads whatever per-series arms exist under `fit_dir` — the CX3 array
    writes one `<series>_d<delta>_seed<n>.npz` per arm — and scores each
    against the annual record, paired seed-by-seed with the `all_d1` arm,
    which is the published model.  Absent arms are reported as absent; this
    part is a no-op until `pbs/wp6b_grid.pbs` has run.
    """
    man_path = os.path.join(out_dir, "wp6b_grid_manifest.csv")
    if not os.path.exists(man_path):
        print(f"[part6] no manifest at {man_path} — run --part 1,5 first")
        return None
    man = pd.read_csv(man_path)
    base = {}
    for sd in seeds:
        p = os.path.join(fit_dir, f"{cl.arm_tag(1, 'all')}_seed{sd}.npz")
        if os.path.exists(p):
            base[sd] = np.load(p, allow_pickle=True)
    if not base:
        print("[part6] the Delta = 1 baseline arm is missing — run --fit first")
        return None

    rows, n_found = [], 0
    for _, r in man[man.included].iterrows():
        tag = cl.arm_tag(int(r.delta), [r.series])
        for sd in seeds:
            p = os.path.join(fit_dir, f"{tag}_seed{sd}.npz")
            if not os.path.exists(p) or sd not in base:
                continue
            n_found += 1
            d, b = np.load(p, allow_pickle=True), base[sd]
            m = d["mask_train"] | d["mask_val"]
            mt = d["mask_test"]
            for k, nm in enumerate(ALPHA_NAMES):
                rows.append(dict(
                    series_coarsened=r.series, delta=int(r.delta), seed=int(sd),
                    degraded=nm,
                    trainval_relRMSE_pct=_rel_rmse_pct(
                        d["alpha_pred_B_at_obs"][m, k],
                        d["alpha_obs_annual"][m, k]),
                    base_trainval_relRMSE_pct=_rel_rmse_pct(
                        b["alpha_pred_B_at_obs"][m, k],
                        b["alpha_obs_annual"][m, k])))
            for k, nm in enumerate(cl.STOCK_NAMES):
                rows.append(dict(
                    series_coarsened=r.series, delta=int(r.delta), seed=int(sd),
                    degraded=f"stock:{nm}",
                    trainval_relRMSE_pct=_rel_rmse_pct(d["S_pred_B"][mt, k],
                                                       d["S_obs_annual"][mt, k]),
                    base_trainval_relRMSE_pct=_rel_rmse_pct(
                        b["S_pred_B"][mt, k], b["S_obs_annual"][mt, k])))
    if not rows:
        print(f"[part6] no per-series arms found under {fit_dir} "
              f"({int(man.included.sum())} expected) — the CX3 grid has not run")
        return None
    df = pd.DataFrame(rows)
    df["delta_pp"] = df.trainval_relRMSE_pct - df.base_trainval_relRMSE_pct
    df.to_csv(os.path.join(out_dir, "wp6b_matrix_per_seed.csv"), index=False)
    agg = []
    for (sr, dl, dg), sub in df.groupby(["series_coarsened", "delta", "degraded"]):
        row = dict(series_coarsened=sr, delta=int(dl), degraded=dg,
                   n_seeds=int(sub.seed.nunique()),
                   paired_HL_pp=_hl(sub.delta_pp),
                   n_worse=int((sub.delta_pp > 0).sum()))
        row.update(_iqr_row(sub.trainval_relRMSE_pct, "relRMSE"))
        agg.append(row)
    ag = pd.DataFrame(agg)
    ag.to_csv(os.path.join(out_dir, "wp6b_matrix.csv"), index=False)
    print(f"[part6] {n_found} per-series arm files scored -> wp6b_matrix.csv")
    return ag


# ===========================================================================
# figure
# ===========================================================================
def figure(ps, agg, targets, out_dir=OUT_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Okabe-Ito, colourblind-safe (project convention)
    cols = {"alpha_cc": "#0072B2", "alpha_refc": "#D55E00",
            "alpha_win": "#009E73", "alpha_dr": "#CC79A7"}
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.3))

    a0 = ax[0]
    sub = agg[(agg.split == GATE_SPLIT) & (agg.channel.isin(ALPHA_NAMES))]
    for ch, s in sub.groupby("channel"):
        s = s.sort_values("delta")
        a0.plot(s.delta, s.relRMSE_median, "o-", color=cols[ch], label=ch, lw=1.8)
        a0.fill_between(s.delta, s.relRMSE_q1, s.relRMSE_q3,
                        color=cols[ch], alpha=0.18, lw=0)
    a0.set_xscale("log"); a0.set_xticks(list(DELTAS))
    a0.set_xticklabels([str(d) for d in DELTAS])
    a0.set_xlabel("reporting interval Δ [yr]")
    a0.set_ylabel("α relRMSE vs annual record [%]")
    a0.set_title(f"(a) estimator degradation ({GATE_SPLIT})", loc="left")
    a0.legend(frameon=False, fontsize=8)

    a1 = ax[1]
    t = targets[targets.series == "all"].sort_values("delta")
    for ch in ALPHA_NAMES:
        a1.plot(t.delta, t[f"{ch}_target_relRMSE_pct"], "s--",
                color=cols[ch], label=ch, lw=1.5, ms=5)
    a1.set_xscale("log"); a1.set_xticks(list(DELTAS))
    a1.set_xticklabels([str(d) for d in DELTAS])
    a1.set_xlabel("reporting interval Δ [yr]")
    a1.set_ylabel("target relRMSE vs annual α [%]")
    a1.set_title("(b) target degradation, before any fit", loc="left")
    a1.legend(frameon=False, fontsize=8)

    a2 = ax[2]
    # same Okabe-Ito ordering as panels (a)/(b), so a colour means one series
    # across the whole figure
    scols = {"stock:Concentrate": "#0072B2", "stock:Refined": "#D55E00",
             "stock:In-Use": "#009E73", "stock:Scrap": "#CC79A7"}
    st = agg[(agg.split == "test") & (agg.channel.str.startswith("stock:"))]
    for ch, s in st.groupby("channel"):
        s = s.sort_values("delta")
        a2.plot(s.delta, s.relRMSE_median, "o-", lw=1.6,
                color=scols.get(ch), label=ch.split(":")[1])
    a2.set_xscale("log"); a2.set_xticks(list(DELTAS))
    a2.set_xticklabels([str(d) for d in DELTAS])
    a2.set_xlabel("reporting interval Δ [yr]")
    a2.set_ylabel("free-run stock relRMSE, test [%]")
    a2.set_title("(c) forecast cost", loc="left")
    a2.legend(frameon=False, fontsize=8)

    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"wp6b_coarsening.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--part", default="1,2,3,4,5",
                    help="comma-separated parts to run")
    ap.add_argument("--fit", action="store_true",
                    help="run the part-3 fits before aggregating")
    ap.add_argument("--seeds", default=f"0-{N_SEEDS - 1}")
    ap.add_argument("--deltas", default=",".join(str(d) for d in DELTAS))
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--fit-dir", default=FIT_DIR)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    info = cl.check(verbose=True)
    if args.check:
        return 0

    def _seeds(spec):
        out = []
        for tok in spec.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if "-" in tok:
                a, b = tok.split("-")
                out.extend(range(int(a), int(b) + 1))
            else:
                out.append(int(tok))
        return out

    parts = {p.strip() for p in args.part.split(",") if p.strip()}
    os.makedirs(args.out, exist_ok=True)
    cfg = lab.load_anchor_config()

    if args.fit:
        fit_curve(_seeds(args.seeds),
                  deltas=[int(d) for d in args.deltas.split(",")],
                  fit_dir=args.fit_dir, cfg=cfg, verbose=args.verbose)

    reach = targets = ps = agg = None
    if "1" in parts:
        reach = part1_reach(out_dir=args.out, fit_dir=args.fit_dir)
        n = int(reach.reaches_loss.sum())
        print(f"[part1] {n}/{len(reach)} series reach the anchor_v4 objective")
        print(reach[reach.reaches_loss][["series", "route", "dA", "dB"]]
              .to_string(index=False))
    if "2" in parts:
        targets = part2_targets(out_dir=args.out, cfg=cfg)
        print(f"[part2] target degradation written "
              f"({len(targets)} series x delta rows)")
    if "3" in parts:
        ps, agg = part3_curve(seeds=_seeds(args.seeds), out_dir=args.out,
                              fit_dir=args.fit_dir, cfg=cfg)
        rep = verify_delta1(fit_dir=args.fit_dir, out_dir=args.out)
        print(f"[part3] reproduction check:\n{rep.to_string(index=False)}")
        sub = agg[(agg.split == GATE_SPLIT) & (agg.channel.isin(ALPHA_NAMES))]
        print(f"[part3] the curve ({GATE_SPLIT}):\n"
              f"{sub[['channel','delta','n_seeds','n_obs_supervised','relRMSE_median','paired_HL_pp','n_worse']].to_string(index=False)}")
    if "4" in parts:
        g = part4_gate(out_dir=args.out)
        print(f"[part4] gate handoff:\n{g.to_string(index=False)}")
    if "5" in parts:
        if reach is None:
            reach = part1_reach(out_dir=args.out, fit_dir=args.fit_dir)
        part5_grid(reach, out_dir=args.out)
    if "6" in parts:
        part6_matrix(out_dir=args.out, fit_dir=args.fit_dir,
                     seeds=_seeds(args.seeds))
    if agg is not None and targets is not None:
        figure(ps, agg, targets, out_dir=args.out)
        print("[fig] analysis/wp6b_coarsening.{png,pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
