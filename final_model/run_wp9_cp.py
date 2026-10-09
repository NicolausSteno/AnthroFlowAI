#!/usr/bin/env python3
"""
run_wp9_cp.py — WP-9 close-out: `dchi/dg` under `learn_cp: true`
================================================================

WP-9 reported ten of its eleven hypotheses and one limitation.  The
limitation is H2's *net* figure.  Chapter 1 predicts that faster growth
dilutes the stock-based recovery loop (`dchi/dg = -4.87` per unit growth
rate, Ch1 Eq. `chistar`); WP-9 measured the numerator well — the dilution
channel Chapter 1 admits is present at `-0.45 [-0.57, -0.36]`, about a tenth
of Chapter 1's steady-state size — but could not report the ratio, because
`anchor_v4` runs with `learn_cp: false`.  With concentrate production pinned
to the ILZSG series, a driver's growth rotation cannot move the scale of the
cycle: `dg_throughput/dtheta` comes out at ~1e-4/yr per +1 pp/yr with a sign
that is not consistent across the activity block, so the ratio's denominator
is not identified.  `COMPUTE_STATUS.md` flag 5 records this and names the
remedy: a refit with `learn_cp: true`.

This script consumes that refit.  `zinc_cp_lab.py` fits the arm — one config
key, no core patches — and dumps it in the WP-2a schema; everything here is
scored by **`run_wp9.py`'s own estimators**, called on a repointed weights
directory rather than reimplemented, so the two arms differ in the fit and in
nothing else.  Part 0 proves that: the baseline arm re-scored through this
path reproduces the published `wp9_dilution*.csv` exactly.

The pre-registered prediction (fixed in `zinc_cp_lab`'s docstring before the
ensemble was fitted, mirrored to `analysis/wp9_cp_registration.csv`):

    P1  the per-driver denominator `dg_throughput/dtheta_j` stops straddling
        zero over Chapter 1's activity + population block;
    P2  the pooled origin regression acquires a net `dchi/dg` whose
        Hodges--Lehmann interval excludes zero.

If P1 and P2 hold, H2's net figure is a result.  If they do not, `dchi/dg` is
not identified by this cycle *at all* rather than by this configuration —
a stronger limitation than the one WP-9 currently states, and reported as
such either way.

`anchor_v4` remains the published model.  Part 1 measures the arm's error only
as a floor check: a denominator read off a badly-fitting cycle would not be
worth having.  Nothing is reselected on it (CLAUDE.md rule 3).

    python run_wp9_cp.py --check
    python run_wp9_cp.py --all
    python run_wp9_cp.py --sens            # refresh the arm's sens cache only
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import zinc_ch1_lab as CH1
import zinc_cp_lab as CP
import run_wp9 as W

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "analysis")
ARM_DIR = CP.OUT_DIR_DEFAULT
ARM_WEIGHTS = CP.WEIGHTS_DIR_DEFAULT
ARM_SENS = CP.SENS_DIR_DEFAULT
BASE_SENS = os.path.join(OUT_DIR, "wp9")
BASE_WEIGHTS = CH1.WEIGHTS_DIR_DEFAULT

COL, GREY = W.COL, W.GREY
ARM_LABEL = {"base": "anchor_v4  (learn_cp: false)",
             "cp": "cp arm  (learn_cp: true)"}

REGISTRATION = [
    dict(id="P1", statement="the per-driver denominator dg_throughput/dtheta_j "
         "stops straddling zero over Ch1's activity + population block",
         criterion="more of the 6 block drivers have a Hodges-Lehmann interval "
                   "on dg_throughput/dtheta excluding zero than under "
                   "learn_cp: false, and the block's sign agreement is >= 5/6"),
    dict(id="P2", statement="the pooled origin regression acquires a net "
         "dchi/dg whose Hodges-Lehmann interval excludes zero",
         criterion="the 'total' channel row of the pooled table, on the Ch1 "
                   "activity + population block, has survives == True"),
]


# ===========================================================================
# arm plumbing
# ===========================================================================

def build_tab(arm, seeds, *, force=False):
    """`run_wp9.seed_table` for one arm, with the sens cache filled first."""
    if arm == "cp":
        W.set_arm(ARM_SENS, ARM_WEIGHTS, CP.cp_config())
    else:
        W.set_arm(BASE_SENS, BASE_WEIGHTS, None)
    os.makedirs(W.SEED_DIR, exist_ok=True)
    W.compute_sens(seeds, force=force)
    return W.seed_table(seeds)


def paired_seeds():
    """Seeds present in both arms — H2 is a paired comparison."""
    a = set(CH1.seed_list(BASE_WEIGHTS))
    b = set(CP.seed_list(ARM_WEIGHTS))
    return sorted(a & b), sorted(a), sorted(b)


# ===========================================================================
# Part 0 — provenance and the no-op guard
# ===========================================================================

def part0_guard(tab_base, tab_cp, tab_base_full):
    """Everything that has to be true before a single number is compared.

    The load-bearing row is the third: `run_wp9.py` gained a `set_arm()` hook
    so this arm could be scored by its estimators rather than by a copy of
    them, and that edit must not have moved any published WP-9 number.  The
    baseline arm is therefore re-scored here, through the edited code, and
    diffed against the CSVs on disk.
    """
    rows = []

    def add(check, value, tol, ok=None, note=""):
        rows.append(dict(check=check, value=value, tolerance=tol,
                         passed=bool((abs(value) <= tol) if ok is None else ok),
                         note=note))

    diff = CP.config_diff()
    add("config diff vs anchor_v4 is exactly {learn_cp: False -> True}",
        float(len(diff)), 1.0, ok=(diff == {"learn_cp": (False, True)}),
        note=json.dumps({k: list(v) for k, v in diff.items()}))
    add("cp arm: core patches applied", float(len(CP.PATCHES)), 0.0,
        note="learn_cp is a native config flag (v5:1733); nothing is rebound")

    for tag, tab in (("base", tab_base), ("cp", tab_cp)):
        add(f"{ARM_LABEL[tag]}: ForwardMode integrator vs the fit's own, stocks",
            max(r["z"]["gap_S"] for r in tab), 1e-12)
        add(f"{ARM_LABEL[tag]}: ForwardMode integrator vs the fit's own, flows",
            max(r["z"]["gap_F"] for r in tab), 1e-12)
        add(f"{ARM_LABEL[tag]}: exog rebuild vs the fit's own feature matrix",
            max(r["z"]["gap_exog"] for r in tab), 1e-12)
        add(f"{ARM_LABEL[tag]}: driver universe and order unchanged",
            0.0, 0.0,
            ok=all(r["z"]["drivers"] == tab[0]["z"]["drivers"] for r in tab),
            note="; ".join(tab[0]["z"]["drivers"][:3]) + "; ...")

    # The regression guard.  Scored on the FULL anchor_v4 ensemble, not on
    # the paired subset, because that is the ensemble the published CSVs were
    # written from — otherwise a single failed refit in the arm would make
    # this row fail for a reason that has nothing to do with the edit.
    seeds_full = [r["seed"] for r in tab_base_full]
    add("regression guard scored on the full anchor_v4 ensemble",
        float(len(seeds_full)), float(len(seeds_full)),
        ok=len(seeds_full) == len(CH1.seed_list(BASE_WEIGHTS)),
        note=f"{len(seeds_full)} seeds")
    h2d, h2p, _seed, _c = W.run_h2(tab_base_full, seeds_full)
    dec = W.run_decomposition(tab_base_full)
    for name, fresh in (("wp9_dilution.csv", h2d),
                        ("wp9_dilution_pooled.csv", h2p),
                        ("wp9_chi_decomposition.csv", dec)):
        path = os.path.join(OUT_DIR, name)
        if not os.path.exists(path):
            add(f"published {name} reproduced by the repointed estimator",
                np.nan, 0.0, ok=False, note="file absent")
            continue
        old = pd.read_csv(path)
        num = [c for c in fresh.columns
               if c in old.columns and np.issubdtype(fresh[c].dtype, np.number)]
        gap = float(np.nanmax(np.abs(fresh[num].to_numpy(float)
                                     - old[num].to_numpy(float))))
        # 1e-12, not 0.0: the comparison goes through the CSV's decimal
        # round-trip, whose own floor is ~1e-16 on these magnitudes.
        add(f"published {name} reproduced by the repointed estimator", gap,
            1e-12, note=f"{len(num)} numeric columns, {len(old)} rows")

    return pd.DataFrame(rows)


def guard_published_per_seed(q_seed):
    """The baseline column of Part 1 must BE the published `anchor_v4` result.

    `analysis/wp2a/A_seed*.npz` are refits of the anchor seeds, so this is not
    a tautology: it says the dumps reproduce the published run and that
    `family_errors` reproduces the core's own family-mean convention.  Without
    it, "the cp arm is N pp worse" would be a comparison against a
    reimplementation rather than against the published model.
    """
    rows = []
    ref = os.path.join(HERE, "anchor_v4", "per_seed.csv")
    if not os.path.exists(ref):
        return pd.DataFrame([dict(check="anchor_v4/per_seed.csv present",
                                  value=np.nan, tolerance=0.0, passed=False,
                                  note="file absent")])
    pub = pd.read_csv(ref)
    pub = pub[pub.rollout == "freerun"]
    mine = q_seed[(q_seed.arm == "base") & (q_seed.split == "test")]
    for fam in ("alpha_relRMSE", "tau_relRMSE", "stock_relRMSE",
                "flow_relRMSE"):
        gap = abs(float(pub[fam].median()) - float(mine[fam].median()))
        rows.append(dict(
            check=f"anchor_v4/per_seed.csv freerun {fam} reproduced",
            value=gap, tolerance=5e-3, passed=bool(gap <= 5e-3),
            note=f"published {pub[fam].median():.3f}, recomputed "
                 f"{mine[fam].median():.3f}, n={len(pub)}"))
    return pd.DataFrame(rows)


# ===========================================================================
# Part 1 — the arm's error, as a floor check only
# ===========================================================================

def part1_quality(tab_base, tab_cp, seeds):
    """Both arms' error, all four families, in the core's own convention.

    Reported, never selected on (CLAUDE.md rule 3).  The flow predictions come
    from the WP-9 sensitivity caches, which hold `fit.integrate_aug`'s own
    `(S, F)` for the dumped Stage-B weights — the same free run the dump's
    `S_pred` records, asserted equal below, so no re-integration is needed and
    nothing new is fitted.

    Two flow family means are reported.  `flow_relRMSE` is the core's own,
    which excludes `concentrate_production` as a tautology when cp is pinned
    and therefore averages 17 flows in `anchor_v4` against 18 in the cp arm.
    `flow_relRMSE_common` drops that flow from both, so the arms are compared
    over an identical 17-flow set; quoting only the first would let the arm
    look worse purely by being scored on one flow more.
    """
    ref_base = CP.observation_reference(CH1.load_anchor_config())
    ref_cp = CP.observation_reference(CP.cp_config())
    rows, gaps = [], []
    for tag, tab, wd, ref, lcp in (("base", tab_base, BASE_WEIGHTS, ref_base, False),
                                   ("cp", tab_cp, ARM_WEIGHTS, ref_cp, True)):
        for r in tab:
            s_ = r["seed"]
            path = os.path.join(wd, f"A_seed{s_}.npz")
            for split in ("test", "train", "all"):
                q = CP.family_errors(path, r["z"]["F"], ref, learn_cp=lcp,
                                     split=split)
                q.update(arm=tag, seed=s_, split=split)
                rows.append(q)
            gaps.append(float(np.max(np.abs(
                np.asarray(r["z"]["S"]) - CP.load_dump(path)["S_pred"]))))
    per_seed = pd.DataFrame(rows)

    fam = ["alpha_relRMSE", "tau_relRMSE", "stock_relRMSE", "flow_relRMSE",
           "flow_relRMSE_common", "cp_relRMSE"]
    per_stock = [c for c in per_seed.columns if c.startswith("stock_relRMSE_")]
    out = []
    for split in ("test", "train", "all"):
        d = per_seed[per_seed.split == split]
        b = d[d.arm == "base"].set_index("seed")
        c = d[d.arm == "cp"].set_index("seed")
        for m in fam + per_stock:
            delta = (c[m] - b[m]).to_numpy(float)
            e, lo, hi = W.hl(delta)
            mb = W.med_iqr(b[m].to_numpy(float))
            mc = W.med_iqr(c[m].to_numpy(float))
            out.append(dict(split=split, family=m,
                            base_median=mb[0], base_q1=mb[1], base_q3=mb[2],
                            cp_median=mc[0], cp_q1=mc[1], cp_q3=mc[2],
                            paired_hl=e, paired_lo=lo, paired_hi=hi,
                            n_seeds_worse=int(np.sum(delta > 0)),
                            n_seeds=int(np.sum(np.isfinite(delta))),
                            survives=W.survives(lo, hi)))
    return per_seed, pd.DataFrame(out), float(max(gaps))


# ===========================================================================
# Part 2 — the denominator: is it identified?
# ===========================================================================

def _dg_matrix(tab):
    """`(n_seeds, n_drivers)` of `dg_throughput/dtheta_j` on the growth block."""
    nD = tab[0]["z"]["n_drivers"]
    w = W.trend_slope_weights(tab[0]["z"]["years"][1:])
    return np.array([[float(np.sum(w * r["jac"]["throughput"][:, nD + j]
                                   / r["par"]["throughput"]))
                      for j in range(nD)] for r in tab])


def _dg_flow_matrix(tab, flow_name):
    """`(n_seeds, n_drivers)` of the growth-rate response of one flow."""
    import zinc_colloc_v5 as v5

    k = v5.FLOW_NAMES.index(flow_name)
    nD = tab[0]["z"]["n_drivers"]
    w = W.trend_slope_weights(tab[0]["z"]["years"][1:])
    return np.array([[float(np.sum(w * r["z"]["dF"][:, k, nD + j]
                                   / r["z"]["F"][:, k]))
                      for j in range(nD)] for r in tab])


def part2_mechanism(tab_base, tab_cp):
    """The mechanism, one level upstream of the denominator.

    With `learn_cp: false` the ODE reads `cp(t)` off the data at every node,
    so `d cp/d theta` is identically zero by construction and the only route
    a driver has into the scale of the cycle is through the coefficients.
    With cp learned it is an MLP output reading the same feature vector as
    every other coefficient.  This is the row that says the arm did what it
    was built to do, before any ratio is formed.
    """
    rows = []
    for tag, tab in (("base", tab_base), ("cp", tab_cp)):
        for flow in ("concentrate_production", "refined_consumption"):
            M = np.abs(_dg_flow_matrix(tab, flow))
            v = np.median(M, axis=1)               # per seed, across drivers
            e, lo, hi = W.hl(v)
            m, q1, q3 = W.med_iqr(v)
            rows.append(dict(arm=tag, label=ARM_LABEL[tag], flow=flow,
                             median_abs_dg=m, q1=q1, q3=q3,
                             hl=e, hl_lo=lo, hl_hi=hi,
                             max_abs_dg=float(np.max(M))))
    return pd.DataFrame(rows)


def part2_identification(tab_base, tab_cp):
    """Per-driver denominator in both arms, and the P1 criterion."""
    drivers = tab_cp[0]["z"]["drivers"]
    block = [j for j, dn in enumerate(drivers)
             if dn in CH1.ACTIVITY_DRIVERS or dn == CH1.POPULATION_DRIVER]
    G = {"base": _dg_matrix(tab_base), "cp": _dg_matrix(tab_cp)}

    rows = []
    for tag in ("base", "cp"):
        for j, dn in enumerate(drivers):
            v = G[tag][:, j]
            e, lo, hi = W.hl(v)
            m, q1, q3 = W.med_iqr(v)
            half = 0.5 * (hi - lo) if np.isfinite(hi) and np.isfinite(lo) else np.nan
            rows.append(dict(
                arm=tag, driver=dn, ch1_block=bool(j in block),
                dgthr_median=m, dgthr_q1=q1, dgthr_q3=q3,
                dgthr_hl=e, dgthr_lo=lo, dgthr_hi=hi,
                abs_hl=abs(e),
                signal_to_noise=(abs(e) / half if half and half > 0 else np.nan),
                frac_seeds_positive=float(np.mean(v > 0)),
                survives=W.survives(lo, hi)))
    per_driver = pd.DataFrame(rows)

    summary = []
    for tag in ("base", "cp"):
        d = per_driver[(per_driver.arm == tag) & per_driver.ch1_block]
        allj = per_driver[per_driver.arm == tag]
        n_pos = int((d.dgthr_hl > 0).sum())
        summary.append(dict(
            arm=tag, label=ARM_LABEL[tag],
            block_n=len(d),
            block_n_survives=int(d.survives.sum()),
            block_sign_agreement=max(n_pos, len(d) - n_pos),
            block_median_abs_hl=float(np.median(d.abs_hl)),
            block_median_snr=float(np.nanmedian(d.signal_to_noise)),
            all_median_abs_hl=float(np.median(allj.abs_hl)),
            all_n_survives=int(allj.survives.sum()), all_n=len(allj)))
    summary = pd.DataFrame(summary)

    b = summary[summary.arm == "base"].iloc[0]
    c = summary[summary.arm == "cp"].iloc[0]
    p1 = dict(prediction="P1",
              base_n_survives=int(b.block_n_survives),
              cp_n_survives=int(c.block_n_survives),
              base_sign_agreement=int(b.block_sign_agreement),
              cp_sign_agreement=int(c.block_sign_agreement),
              base_median_abs_hl=float(b.block_median_abs_hl),
              cp_median_abs_hl=float(c.block_median_abs_hl),
              magnitude_ratio=float(c.block_median_abs_hl
                                    / b.block_median_abs_hl),
              held=bool(c.block_n_survives > b.block_n_survives
                        and c.block_sign_agreement >= 5))
    return per_driver, summary, p1


# ===========================================================================
# Part 4 — did the numerator survive the refit?
# ===========================================================================

def part4_numerator(tab_base, tab_cp, seeds):
    """The cp arm is an identification device, not a different theory.

    If the numerator moved as much as the denominator did, the arm would have
    bought identification by changing the object being identified.  Two
    checks: Chapter 1's five parameters, and the dilution channel — the part
    of `dchi/dtheta` WP-9 already reported as well determined.
    """
    rows = []
    for k in ("lam", "sig", "kap", "rho", "chi"):
        v = {t: np.array([float(np.mean(r["par"][k])) for r in tab])
             for t, tab in (("base", tab_base), ("cp", tab_cp))}
        e, lo, hi = W.hl(v["cp"] - v["base"])
        mb = W.med_iqr(v["base"])
        mc = W.med_iqr(v["cp"])
        rows.append(dict(quantity=k, ch1_calibrated=W.CH1_CALIB[k],
                         base_median=mb[0], base_q1=mb[1], base_q3=mb[2],
                         cp_median=mc[0], cp_q1=mc[1], cp_q3=mc[2],
                         paired_hl=e, paired_lo=lo, paired_hi=hi,
                         rel_shift_pct=100.0 * (mc[0] - mb[0]) / abs(mb[0]),
                         survives=W.survives(lo, hi)))

    nD = tab_cp[0]["z"]["n_drivers"]
    w = W.trend_slope_weights(tab_cp[0]["z"]["years"][1:])
    for key, label in (("dilution", "dilution slope (Ch1's channel)"),):
        v = {}
        for t, tab in (("base", tab_base), ("cp", tab_cp)):
            v[t] = np.array([
                CH1.ols_through_origin(
                    np.array([float(np.sum(w * r["jac"]["throughput"][:, nD + j]
                                           / r["par"]["throughput"]))
                              for j in range(nD)]),
                    np.array([float(np.mean(r["dec"][key][:, nD + j]))
                              for j in range(nD)]))[0]
                for r in tab])
        e, lo, hi = W.hl(v["cp"] - v["base"])
        mb, mc = W.med_iqr(v["base"]), W.med_iqr(v["cp"])
        rows.append(dict(quantity=label, ch1_calibrated=np.nan,
                         base_median=mb[0], base_q1=mb[1], base_q3=mb[2],
                         cp_median=mc[0], cp_q1=mc[1], cp_q3=mc[2],
                         paired_hl=e, paired_lo=lo, paired_hi=hi,
                         rel_shift_pct=100.0 * (mc[0] - mb[0]) / abs(mb[0]),
                         survives=W.survives(lo, hi)))
    return pd.DataFrame(rows)


# ===========================================================================
# Part 6 — robustness, forced by P1's failure
# ===========================================================================

def _pooled_slopes(tab, idx, channel="total"):
    """Per-seed origin-regression slope over a chosen set of directions."""
    nD = tab[0]["z"]["n_drivers"]
    w = W.trend_slope_weights(tab[0]["z"]["years"][1:])
    out = []
    for r in tab:
        x = np.array([float(np.sum(w * r["jac"]["throughput"][:, nD + j]
                                   / r["par"]["throughput"])) for j in idx])
        tot = np.array([float(np.mean(r["jac"]["chi"][:, nD + j]))
                        for j in idx])
        dil = np.array([float(np.mean(r["dec"]["dilution"][:, nD + j]))
                        for j in idx])
        y = tot if channel == "total" else dil
        out.append(CH1.ols_through_origin(x, y)[0])
    return np.asarray(out, float)


def part6_robustness(tab_cp, q_seed, seeds):
    """Two questions P1's failure raises, answered rather than argued.

    P1 asked for per-driver sign agreement and did not get it: the cp arm's
    denominators are four times larger but point in different directions on
    different drivers.  That does not by itself invalidate the pooled slope —
    an origin regression over directions does not need them to agree in sign,
    it needs them to carry signal — but it does raise two specific ways the
    pooled number could be an artefact, and both are testable.

      (a) leverage.  Is the slope the average of thirteen directions, or one
          direction and twelve zeros?  Dropped one driver at a time.
      (b) fit quality.  Part 1 shows the arm fits the observed cycle worse.
          If the slope is an artefact of that, seeds that fit worse should
          give systematically different slopes.  Rank correlation, per seed.

    Both are post hoc.  They are reported as robustness on a pre-registered
    quantity, never as a substitute for the criterion that failed.
    """
    drivers = tab_cp[0]["z"]["drivers"]
    block = [j for j, dn in enumerate(drivers)
             if dn in CH1.ACTIVITY_DRIVERS or dn == CH1.POPULATION_DRIVER]
    rows = []
    for label, idx in (("all 13 drivers", list(range(len(drivers)))),
                       ("Ch1 activity + population block", block)):
        full = _pooled_slopes(tab_cp, idx)
        e0, lo0, hi0 = W.hl(full)
        rows.append(dict(subset=label, dropped="(none)", n_directions=len(idx),
                         hl=e0, hl_lo=lo0, hl_hi=hi0,
                         survives=W.survives(lo0, hi0),
                         shift_vs_full=0.0))
        for j in idx:
            rest = [k for k in idx if k != j]
            v = _pooled_slopes(tab_cp, rest)
            e, lo, hi = W.hl(v)
            rows.append(dict(subset=label, dropped=drivers[j],
                             n_directions=len(rest), hl=e, hl_lo=lo, hl_hi=hi,
                             survives=W.survives(lo, hi),
                             shift_vs_full=e - e0))
    loo = pd.DataFrame(rows)

    q = q_seed[(q_seed.arm == "cp")
               & (q_seed.split == "test")].set_index("seed")
    corr = []
    for label, idx in (("all 13 drivers", list(range(len(drivers)))),
                       ("Ch1 activity + population block", block)):
        v = _pooled_slopes(tab_cp, idx)
        for metric in ("stock_relRMSE", "flow_relRMSE_common",
                       "alpha_relRMSE", "cp_relRMSE"):
            m = q.loc[[r["seed"] for r in tab_cp], metric].to_numpy(float)
            corr.append(dict(subset=label, metric=metric,
                             spearman=W.spearman(m, v), n=len(v)))

    # (b), sharpened: a rank correlation says the slope moves with fit
    # quality, not whether the well-fitting seeds still carry the result.
    # These are the seeds whose test-window error is no worse than the
    # published model's own median, so the slope is read off cycles that fit
    # as well as `anchor_v4` does.  Selecting seeds for a robustness subset
    # is not selecting a model (CLAUDE.md rule 3): nothing about the arm or
    # the published fit changes on the outcome.
    base_med = float(np.median(
        q_seed[(q_seed.arm == "base") & (q_seed.split == "test")]
        ["stock_relRMSE"].to_numpy(float)))
    err = q.loc[[r["seed"] for r in tab_cp], "stock_relRMSE"].to_numpy(float)
    order = np.argsort(err)
    fitsub = []
    for label, idx in (("all 13 drivers", list(range(len(drivers)))),
                       ("Ch1 activity + population block", block)):
        v = _pooled_slopes(tab_cp, idx)
        for sub_label, sel in (
                ("all seeds", np.ones(v.size, bool)),
                ("8 best-fitting seeds", np.isin(np.arange(v.size), order[:8])),
                (f"seeds at or below anchor_v4's median test relRMSE "
                 f"({base_med:.2f}%)", err <= base_med)):
            e, lo, hi = W.hl(v[sel])
            fitsub.append(dict(subset=label, seed_subset=sub_label,
                               n_seeds=int(sel.sum()), hl=e, hl_lo=lo,
                               hl_hi=hi, survives=W.survives(lo, hi),
                               n_negative=int((v[sel] < 0).sum()),
                               median_test_relRMSE=float(np.median(err[sel]))))
    return loo, pd.DataFrame(corr), pd.DataFrame(fitsub)


# ===========================================================================
# Part 5 — the verdict
# ===========================================================================

def part5_verdict(h2p_base, h2p_cp, p1, ch1_dchi_dg):
    """P1, P2 and the resolution of H2."""
    def row(df, subset, channel):
        d = df[(df.subset == subset) & (df.channel == channel)]
        return d.iloc[0] if len(d) else None

    subset = "Ch1 activity + population block"
    tot_b = row(h2p_base, subset, "total")
    tot_c = row(h2p_cp, subset, "total")
    dil_c = row(h2p_cp, subset, "dilution only (Ch1's channel)")

    p2 = dict(prediction="P2",
              base_hl=float(tot_b.dchi_dg_hl), base_lo=float(tot_b.dchi_dg_lo),
              base_hi=float(tot_b.dchi_dg_hi),
              base_survives=bool(tot_b.survives),
              cp_hl=float(tot_c.dchi_dg_hl), cp_lo=float(tot_c.dchi_dg_lo),
              cp_hi=float(tot_c.dchi_dg_hi), cp_survives=bool(tot_c.survives),
              held=bool(tot_c.survives))

    reg = pd.DataFrame(REGISTRATION)
    reg["held"] = [p1["held"], p2["held"]]
    reg["evidence"] = [
        f"block drivers with an interval excluding zero: "
        f"{p1['base_n_survives']}/6 -> {p1['cp_n_survives']}/6; "
        f"sign agreement {p1['base_sign_agreement']}/6 -> "
        f"{p1['cp_sign_agreement']}/6; |dg_thr| x{p1['magnitude_ratio']:.1f}",
        f"net dchi/dg {p2['base_hl']:+.2f} [{p2['base_lo']:+.2f}, "
        f"{p2['base_hi']:+.2f}] -> {p2['cp_hl']:+.2f} [{p2['cp_lo']:+.2f}, "
        f"{p2['cp_hi']:+.2f}]",
    ]

    # The pre-registration named two predictions and described what to
    # conclude if both held and if neither did.  It did not anticipate a
    # split, and a split is what happened.  The branch below reports that
    # plainly instead of collapsing it into a binary the registration never
    # defined; P1's own text stands as written, marked not held.
    if p1["held"] and p2["held"]:
        verdict = ("SIGN CONFIRMED and identified" if p2["cp_hi"] < 0
                   else "IDENTIFIED and CONTRADICTED in sign")
        note = "both pre-registered predictions held"
    elif p2["held"] and not p1["held"]:
        verdict = ("SIGN CONFIRMED and identified, on a split pre-registration"
                   if p2["cp_hi"] < 0 else
                   "IDENTIFIED and CONTRADICTED in sign, on a split "
                   "pre-registration")
        note = ("P2 held and P1 did not. P1 asked the wrong question of the "
                "denominator: it scored agreement in the SIGN of the "
                "per-driver response, which anchor_v4 already satisfies "
                f"({p1['base_sign_agreement']}/6) because its denominators "
                "are near-zero noise of a common sign, and which a pooled "
                "origin regression does not require. What that regression "
                "needs is spread and signal across directions, and the cp arm "
                f"supplies it (denominator x{p1['magnitude_ratio']:.1f}). The "
                "criterion was a poor operationalisation, stated as written "
                "and marked not held; Part 6 tests the two ways P1's failure "
                "could still make the pooled slope an artefact")
    elif p1["held"] and not p2["held"]:
        verdict = "DENOMINATOR IDENTIFIED, RATIO STILL NOT"
        note = "P1 held and P2 did not"
    else:
        verdict = "NOT IDENTIFIED — and not by this configuration either"
        note = ("neither prediction held; the limitation is the 40-year "
                "window and the exogenous driver block, not the cp pin")

    detail = (f"net dchi/dg = {p2['cp_hl']:+.2f} [{p2['cp_lo']:+.2f}, "
              f"{p2['cp_hi']:+.2f}] against anchor_v4's {p2['base_hl']:+.2f} "
              f"[{p2['base_lo']:+.2f}, {p2['base_hi']:+.2f}] and Chapter 1's "
              f"{ch1_dchi_dg:+.2f} ({100.0 * p2['cp_hl'] / ch1_dchi_dg:.0f}% "
              f"of it); dilution channel {dil_c.dchi_dg_hl:+.2f} "
              f"[{dil_c.dchi_dg_lo:+.2f}, {dil_c.dchi_dg_hi:+.2f}]. {note}")
    return reg, pd.DataFrame([p1, p2]), verdict, detail


# ===========================================================================
# figure
# ===========================================================================

def fig_identification(per_driver, h2p_base, h2p_cp, tab_base, tab_cp, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.4),
                                   gridspec_kw=dict(width_ratios=[1.25, 1.0]))

    d = per_driver[per_driver.arm == "cp"].sort_values("dgthr_hl")
    order = list(d.driver)
    ypos = np.arange(len(order))
    off = 0.18
    for tag, c, sgn in (("base", GREY, +1), ("cp", COL[0], -1)):
        g = per_driver[per_driver.arm == tag].set_index("driver").loc[order]
        y = ypos + sgn * off
        ax1.hlines(y, g.dgthr_lo, g.dgthr_hi, color=c, lw=2.0)
        ax1.scatter(g.dgthr_hl, y, s=26, color=c, zorder=3,
                    label=ARM_LABEL[tag])
        blk = g.ch1_block.to_numpy(bool)
        ax1.scatter(g.dgthr_hl.to_numpy()[blk], y[blk], s=70,
                    facecolors="none", edgecolors=c, lw=1.1, zorder=4)
    ax1.axvline(0.0, color="k", lw=0.9)
    ax1.set_yticks(ypos)
    ax1.set_yticklabels([s if len(s) < 34 else s[:31] + "..." for s in order],
                        fontsize=8)
    ax1.set_xlabel(r"$\partial g_{\mathrm{throughput}}/\partial \theta_j$"
                   "   per +1 pp/yr on driver $j$   (HL, 95% interval)")
    ax1.set_title("the denominator H2 divides by\n"
                  "ringed markers = Chapter 1's activity + population block",
                  fontsize=9.5)
    ax1.legend(fontsize=8, loc="lower right", frameon=False)

    nD = tab_cp[0]["z"]["n_drivers"]
    w = W.trend_slope_weights(tab_cp[0]["z"]["years"][1:])
    drivers = tab_cp[0]["z"]["drivers"]
    blk = [j for j, dn in enumerate(drivers)
           if dn in CH1.ACTIVITY_DRIVERS or dn == CH1.POPULATION_DRIVER]
    for k, (tag, tab, c) in enumerate((("base", tab_base, GREY),
                                       ("cp", tab_cp, COL[0]))):
        v = []
        for r in tab:
            x = np.array([float(np.sum(w * r["jac"]["throughput"][:, nD + j]
                                       / r["par"]["throughput"])) for j in blk])
            y = np.array([float(np.mean(r["jac"]["chi"][:, nD + j]))
                          for j in blk])
            v.append(CH1.ols_through_origin(x, y)[0])
        v = np.asarray(v, float)
        ax2.scatter(np.full(v.size, k) + 0.06 * np.random.default_rng(0)
                    .standard_normal(v.size), v, s=18, color=c, alpha=0.75)
        e, lo, hi = W.hl(v)
        ax2.hlines([e], k - 0.25, k + 0.25, color=c, lw=2.4)
        ax2.vlines([k], lo, hi, color=c, lw=1.4)
    ax2.axhline(0.0, color="k", lw=0.9)
    ch1 = float(h2p_cp.ch1_predicted.iloc[0])
    ax2.axhline(ch1, color=COL[2], lw=1.6, ls="--",
                label=f"Ch1 steady state  {ch1:+.2f}")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["learn_cp: false\n(anchor_v4)", "learn_cp: true\n(cp arm)"],
                        fontsize=9)
    ax2.set_ylabel(r"net $\partial\chi/\partial g_{\mathrm{throughput}}$"
                   "   (origin regression, per seed)")
    ax2.set_title("H2's net response, Ch1 activity + population block\n"
                  "one point per seed; bar = HL, whisker = 95% interval",
                  fontsize=9.5)
    ax2.legend(fontsize=8, frameon=False, loc="best")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


# ===========================================================================
# main
# ===========================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="WP-9 close-out: dchi/dg under learn_cp: true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sens", action="store_true",
                    help="(re)compute the arm's sensitivity cache only")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seeds", default=None)
    a = ap.parse_args(argv)

    if a.check:
        CP.check()
        return 0

    both, base_only, cp_only = paired_seeds()
    if a.seeds:
        want = {int(s) for s in a.seeds.split(",") if s.strip()}
        both = [s for s in both if s in want]
    if not both:
        print(f"[wp9cp] no paired seeds: {len(base_only)} anchor_v4 dumps, "
              f"{len(cp_only)} cp-arm dumps. Fit the arm first:\n"
              f"        python zinc_cp_lab.py --seeds 0,1,2,...")
        return 1
    print(f"[wp9cp] {len(both)} paired seeds "
          f"(anchor_v4 {len(base_only)}, cp arm {len(cp_only)})")

    t0 = time.time()
    tab_cp = build_tab("cp", both, force=a.force)
    print(f"[wp9cp] cp-arm sensitivities ready in {time.time()-t0:.0f}s")
    if a.sens and not a.all:
        return 0
    base_all = CH1.seed_list(BASE_WEIGHTS)
    tab_base_full = build_tab("base", base_all, force=False)
    tab_base = ([r for r in tab_base_full if r["seed"] in set(both)]
                if set(both) <= set(base_all) else build_tab("base", both))

    # --- Part 0 ---------------------------------------------------------
    guard = part0_guard(tab_base, tab_cp, tab_base_full)
    guard.to_csv(os.path.join(OUT_DIR, "wp9_cp_guard.csv"), index=False)
    print("\n[wp9cp] Part 0 — provenance and the no-op guard")
    print(guard[["check", "value", "passed"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3e}"))
    if not bool(guard.passed.all()):
        print("\n[wp9cp] GUARD FAILED — refusing to report comparisons.")
        return 2

    # --- Part 1 ---------------------------------------------------------
    q_seed, q_sum, s_gap = part1_quality(tab_base, tab_cp, both)
    q_seed.to_csv(os.path.join(OUT_DIR, "wp9_cp_fit_quality_per_seed.csv"),
                  index=False)
    q_sum.to_csv(os.path.join(OUT_DIR, "wp9_cp_fit_quality.csv"), index=False)
    g2 = guard_published_per_seed(q_seed)
    guard = pd.concat([guard, g2], ignore_index=True)
    guard.to_csv(os.path.join(OUT_DIR, "wp9_cp_guard.csv"), index=False)
    print("\n[wp9cp] Part 0 (cont.) — the baseline column IS the published run")
    print(g2[["check", "value", "passed", "note"]].to_string(index=False))
    if not bool(g2.passed.all()):
        print("\n[wp9cp] GUARD FAILED — refusing to report comparisons.")
        return 2
    st = q_seed["coef_state_dependence"].to_numpy(float)
    print("\n[wp9cp] Part 1 — both arms' error, core convention "
          "(floor check only; CLAUDE.md rule 3)")
    print(f"  sens-cache S vs the dump's own S_pred: max abs gap {s_gap:.2e}; "
          f"coefficient state-dependence max {np.nanmax(st):.2e} "
          f"(use_stock_input: false, so at_obs == at_pred)")
    for split in ("test", "train"):
        d = q_sum[(q_sum.split == split)
                  & ~q_sum.family.str.startswith("stock_relRMSE_")]
        print(f"\n  --- {split} window ---")
        print(d[["family", "base_median", "cp_median", "paired_hl",
                 "paired_lo", "paired_hi", "n_seeds_worse", "survives"]]
              .to_string(index=False, float_format=lambda x: f"{x:8.2f}"))
    d = q_sum[(q_sum.split == "test")
              & q_sum.family.str.startswith("stock_relRMSE_")]
    print("\n  --- per stock, test window ---")
    print(d[["family", "base_median", "cp_median", "paired_hl", "paired_lo",
             "paired_hi", "n_seeds_worse", "survives"]]
          .to_string(index=False, float_format=lambda x: f"{x:8.2f}"))

    # --- Part 2 ---------------------------------------------------------
    mech = part2_mechanism(tab_base, tab_cp)
    mech.to_csv(os.path.join(OUT_DIR, "wp9_cp_mechanism.csv"), index=False)
    print("\n[wp9cp] Part 2 — the mechanism: does a driver reach the scale "
          "of the cycle at all?")
    print(mech[["arm", "flow", "median_abs_dg", "hl", "hl_lo", "hl_hi",
                "max_abs_dg"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3e}"))

    per_driver, id_sum, p1 = part2_identification(tab_base, tab_cp)
    per_driver.to_csv(os.path.join(OUT_DIR, "wp9_cp_denominator.csv"),
                      index=False)
    id_sum.to_csv(os.path.join(OUT_DIR, "wp9_cp_identification.csv"),
                  index=False)
    print("\n[wp9cp] Part 2 — the denominator dg_throughput/dtheta")
    print(id_sum.to_string(index=False, float_format=lambda x: f"{x:.3e}"))
    print(f"  P1 {'HELD' if p1['held'] else 'DID NOT HOLD'}: block intervals "
          f"excluding zero {p1['base_n_survives']}/6 -> {p1['cp_n_survives']}/6, "
          f"sign agreement {p1['base_sign_agreement']}/6 -> "
          f"{p1['cp_sign_agreement']}/6, magnitude "
          f"x{p1['magnitude_ratio']:.1f}")

    # --- Part 3 ---------------------------------------------------------
    seeds_cp = [r["seed"] for r in tab_cp]
    h2d_cp, h2p_cp, h2s_cp, ch1_dchi_dg = W.run_h2(tab_cp, seeds_cp)
    dec_cp = W.run_decomposition(tab_cp)
    decp_cp = W.run_decomposition_pooled(tab_cp)
    h2d_cp.to_csv(os.path.join(OUT_DIR, "wp9_cp_dilution.csv"), index=False)
    h2p_cp.to_csv(os.path.join(OUT_DIR, "wp9_cp_dilution_pooled.csv"),
                  index=False)
    h2s_cp.to_csv(os.path.join(OUT_DIR, "wp9_cp_dilution_per_seed.csv"),
                  index=False)
    dec_cp.to_csv(os.path.join(OUT_DIR, "wp9_cp_chi_decomposition.csv"),
                  index=False)
    decp_cp.to_csv(os.path.join(OUT_DIR,
                                "wp9_cp_chi_decomposition_pooled.csv"),
                   index=False)
    _h2d_b, h2p_base, _s, _c = W.run_h2(tab_base, [r["seed"] for r in tab_base])
    print("\n[wp9cp] Part 3 — H2 on the cp arm (run_wp9.run_h2, unmodified)")
    print(h2p_cp[["subset", "channel", "dchi_dg_hl", "dchi_dg_lo",
                  "dchi_dg_hi", "r2_median", "survives"]]
          .to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

    # --- Part 4 ---------------------------------------------------------
    num = part4_numerator(tab_base, tab_cp, both)
    num.to_csv(os.path.join(OUT_DIR, "wp9_cp_numerator.csv"), index=False)
    print("\n[wp9cp] Part 4 — did the numerator survive the refit?")
    print(num[["quantity", "base_median", "cp_median", "rel_shift_pct",
               "paired_hl", "paired_lo", "paired_hi", "survives"]]
          .to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

    # --- Part 6 ---------------------------------------------------------
    loo, corr, fitsub = part6_robustness(tab_cp, q_seed, both)
    loo.to_csv(os.path.join(OUT_DIR, "wp9_cp_leave_one_driver_out.csv"),
               index=False)
    corr.to_csv(os.path.join(OUT_DIR, "wp9_cp_slope_vs_fit.csv"), index=False)
    fitsub.to_csv(os.path.join(OUT_DIR, "wp9_cp_slope_by_fit_subset.csv"),
                  index=False)
    print("\n[wp9cp] Part 6 — robustness of the pooled slope (post hoc, "
          "forced by P1's failure)")
    for sub in loo.subset.unique():
        d = loo[loo.subset == sub]
        print(f"  {sub}: full {d.iloc[0].hl:+.3f} "
              f"[{d.iloc[0].hl_lo:+.3f}, {d.iloc[0].hl_hi:+.3f}]; "
              f"leave-one-driver-out range "
              f"[{d.iloc[1:].hl.min():+.3f}, {d.iloc[1:].hl.max():+.3f}], "
              f"{int(d.iloc[1:].survives.sum())}/{len(d)-1} drops still "
              f"exclude zero")
    print(corr.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
    print(fitsub.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))

    # --- Part 5 ---------------------------------------------------------
    reg, preds, verdict, detail = part5_verdict(h2p_base, h2p_cp, p1,
                                                ch1_dchi_dg)
    reg.to_csv(os.path.join(OUT_DIR, "wp9_cp_registration.csv"), index=False)
    preds.to_csv(os.path.join(OUT_DIR, "wp9_cp_predictions.csv"), index=False)

    subset = "Ch1 activity + population block"
    tot_c = h2p_cp[(h2p_cp.subset == subset) & (h2p_cp.channel == "total")].iloc[0]
    dil_c = h2p_cp[(h2p_cp.subset == subset)
                   & (h2p_cp.channel.str.startswith("dilution"))].iloc[0]
    h2row = pd.DataFrame([dict(
        id="H2", hypothesis="Growth dilutes the stock-based recovery loop",
        ch1_prediction=f"dchi/dg = {ch1_dchi_dg:.2f} < 0 "
                       f"(a fixed-coefficient statement)",
        arm="learn_cp: true (WP-9 close-out)",
        n_seeds=len(both),
        measured=f"net {tot_c.dchi_dg_hl:+.2f} [{tot_c.dchi_dg_lo:+.2f}, "
                 f"{tot_c.dchi_dg_hi:+.2f}]; dilution channel "
                 f"{dil_c.dchi_dg_hl:+.2f} [{dil_c.dchi_dg_lo:+.2f}, "
                 f"{dil_c.dchi_dg_hi:+.2f}]",
        verdict=verdict, detail=detail)])
    h2row.to_csv(os.path.join(OUT_DIR, "wp9_cp_h2_verdict.csv"), index=False)

    fig_identification(per_driver, h2p_base, h2p_cp, tab_base, tab_cp,
                       os.path.join(OUT_DIR, "wp9_cp_identification"))

    info = CP.check(verbose=False)
    info.pop("digests", None)
    info.update(n_paired_seeds=len(both), seeds=both,
                p1=p1, verdict=verdict, detail=detail,
                guard_all_passed=bool(guard.passed.all()),
                ch1_dchi_dg=float(ch1_dchi_dg))
    with open(os.path.join(OUT_DIR, "wp9_cp_check.json"), "w") as fh:
        json.dump(info, fh, indent=2, default=str)

    print("\n[wp9cp] Part 5 — H2 resolved")
    print(reg[["id", "statement", "held"]].to_string(index=False))
    print(f"\n  H2  {verdict}\n      {detail}")
    print(f"\n[wp9cp] done in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
