#!/usr/bin/env python3
"""
run_wp11b.py — WP-11b: the synthetic-real agreement gate
========================================================

The spec's instruction is unambiguous: overlay WP-6b's real-data coarsening
curve (10 yr -> 5 -> 2 -> 1) on the corresponding region of WP-11a's synthetic
frequency-value curve, and decide from the overlap whether the twin has earned
the right to extrapolate *above* annual frequency.  "**This is a genuine gate,
not a formality — report the outcome either way.**"

Status, stated first because it decides how everything below should be read
--------------------------------------------------------------------------
**The gate is CLOSED and it FAILS.**  *Revised 2026-08-26.*  The two refit
curves it compares now both exist — WP-6b's real-data coarsening curve
(4 widths x 8 seeds) and WP-11a's synthetic frequency-value curve (4 frequency
+ 3 coarse full-span arms x 8 seeds) — and applying the criteria
pre-registered below gives **2 of 4 channels, against a threshold of 3**.
`alpha_refc` and `alpha_win` pass; `alpha_cc` fails both conditions and
`alpha_dr` fails (G2) at a ratio of 0.485 against a band floor of 0.5.  The
twin degrades too *slowly* under coarsening on three of the four channels,
which is the same direction and the same magnitude as the observation-side
surrogate found before either refit existed.

Per the spec's own instruction — "if it does not, the extrapolation must be
dropped and the paper limited to the coarsening result" — **the twin has not
earned the right to extrapolate above annual frequency.**  That costs the
paper less than it sounds: WP-11a's own curve above annual is nearly flat, so
there was no strong extrapolation waiting to be licensed.

The original version of this file did three things instead, all of which
stand and two of which are still the only evidence in their currency:

  1. **pre-register the gate** — the comparison, the statistic and the
     pass/fail threshold were fixed here, in code, before either curve
     existed, so that closing it was arithmetic and not a judgement call;
  2. **build and test the machinery** on placeholders, so the gate closed by
     re-running one command once the fits landed;
  3. **run the two agreement checks that need no fits at all**, in the two
     currencies that are available — the *observation* side and the
     *information* side — and report what they say.  Neither is the gate.
     Both are informative about the same question, and both are computed
     identically on the real record and on the twin, which is the property
     that makes them comparable.

One correction was needed to close it, and it does not decide the outcome
-------------------------------------------------------------------------
The pre-registered statistic below says the curves are compared "normalised to
its own value at Delta = 1 yr so that the comparison is of shape and not of
level".  `part4_gate` did not do that — it compared raw levels — and the
discrepancy went unnoticed because the two surrogate arms are already
relative by construction (the Fisher arm's metric *is* `sd_rel_to_annual`, and
the observation arm's is identically zero at Delta = 1, flag 1).  The
implementation has been brought into line with the text.  **The raw-level
comparison is reported alongside as `gate_curves_absolute_level`, and it also
fails, 2/4** — on a different pair of channels.  The verdict is therefore not
an artefact of the correction, and `gate_selftest` now exercises both modes
(8 constructed cases) so the difference between them is on the record: under
normalisation a pure factor-of-three level shift passes *by construction*,
which is the deliberate consequence of a shape test and the reason the
absolute row is printed too.

Part 2 — the observation-side overlay (exact, no fits)
------------------------------------------------------
Coarsening the record degrades `alpha_obs` before any estimator sees it,
because `alpha_obs = F_int / trapz(S)` is an integral over an interpolated
point stock and the quadrature error grows with the window.  On the real
record that degradation is computable exactly: annual flow integrals sum to a
coarse window without approximation, and the best available reference for the
window's exposure-weighted alpha is the same numerator over the *annual*
trapezoid sum.  The identical statistic is computed on the twin, where the
analytic truth is also available, so the twin says both what the statistic is
and what it is a proxy *for*.

Part 3 — the information-side overlay (Fisher, no refits)
---------------------------------------------------------
The estimator-side question WP-6b answers by refitting can be asked of the
curvature instead.  For a record observed at width `Delta`, build
`FIM(Delta) = C_Delta^T Sigma^-1 C_Delta + mu I` over the whole observation
set — four stocks point-in-time, nine flows as period integrals, every width
an exact re-aggregation of one monthly solve, using `zinc_voi_lab`'s operators
— and read off the marginal SD of each transfer coefficient.  Run on the
published real fit and on WP-3's twin fit with the same ridge, this is a
like-for-like coarsening curve that extends *above* annual as well as below,
which is exactly the region the gate is about.

    python run_wp11b.py --check
    python run_wp11b.py --observation
    python run_wp11b.py --fisher --seeds 0,1,2,3,4
    python run_wp11b.py --gate --figures
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import zinc_market_lab as M
import zinc_synth_lab as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "analysis")
SYNTH_DIR = os.path.join(OUT_DIR, "synth")
FIT_DIR = os.path.join(SYNTH_DIR, "fits")
WEIGHTS_DIR = os.path.join(OUT_DIR, "wp2a")

COL = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00", "#56B4E9"]
GREY = "#555555"

ALPHA_NAMES = S.ALPHA_NAMES
TRAINVAL_END = 2007.0

# The overlapping region the gate is evaluated on: widths a *real* annual
# record can be coarsened to.  Anything finer than 1 yr exists only on the
# twin and is the extrapolation the gate is deciding whether to allow.
OVERLAP_DELTAS = (1.0, 2.0, 5.0, 10.0)
FINE_DELTAS = (1.0 / 12.0, 0.25, 1.0)

# ---------------------------------------------------------------------------
# THE PRE-REGISTERED GATE.  Fixed here before either curve exists.
# ---------------------------------------------------------------------------
# Statistic: on each of the four alpha channels, the *degradation curve* is
# the metric of interest evaluated at Delta in OVERLAP_DELTAS, normalised to
# its own value at Delta = 1 yr so that the comparison is of shape and not of
# level (the twin is not a fit and its level is not expected to match — WP-3
# flag 4).
#
# Agreement is declared on two conditions, both of which must hold:
#
#   (G1) MONOTONE AGREEMENT.  The real and synthetic curves must both be
#        non-decreasing in Delta on every channel, to within the tolerance
#        below.  A twin that improves under coarsening where the real record
#        degrades is not reproducing the mechanism.
#
#   (G2) RATIO AGREEMENT.  At every Delta in the overlap, the ratio of the
#        synthetic degradation to the real degradation must lie inside
#        [1/GATE_RATIO, GATE_RATIO], on at least GATE_MIN_CHANNELS of the four
#        channels.
#
# A factor of two either way is a deliberately loose band: the claim the gate
# licenses is qualitative ("the twin reproduces the observed degradation"),
# and a tighter band would fail on seed noise rather than on mechanism.
GATE_RATIO = 2.0
GATE_MIN_CHANNELS = 3
GATE_MONOTONE_TOL = 0.10        # fractional slack allowed on (G1)


def rel_rmse_pct(pred, obs, mask=None):
    pred = np.asarray(pred, float); obs = np.asarray(obs, float)
    m = np.isfinite(pred) & np.isfinite(obs)
    if mask is not None:
        m &= np.asarray(mask, bool)
    if not np.any(m):
        return np.nan
    return 100.0 * np.sqrt(np.mean((pred[m] - obs[m]) ** 2)) / max(
        np.mean(np.abs(obs[m])), 1e-12)


# ===========================================================================
# Part 1 — what the gate needs, and whether it is there
# ===========================================================================
def part1_status(out_dir=OUT_DIR):
    """Look for the two refit curves the gate compares and say what is
    missing, with the fit counts, so the note does not have to guess."""
    want = {
        # Both counts are what the packages actually ran, not what this file
        # estimated before they existed.  WP-6b's all-series curve is 4 widths
        # x 8 seeds; its per-series matrix is a different object and is not
        # what the gate consumes.  WP-11a is 11 arms x 8 seeds, of which the
        # 7 full-span ones (4 frequency + 3 coarse) enter the gate.
        "wp6b_coarsening": dict(
            path=os.path.join(out_dir, "wp6b_coarsening.csv"),
            fits=4 * 8,
            note="4 reporting intervals x 8 seeds (all-series curve)"),
        "wp11a_frequency_value": dict(
            path=os.path.join(out_dir, "wp11a_frequency_value.csv"),
            fits=7 * 8,
            note="4 frequency + 3 coarse full-span arms x 8 seeds"),
    }
    rows = []
    for k, v in want.items():
        rows.append(dict(curve=k, path=v["path"], present=os.path.exists(v["path"]),
                         fits_required=v["fits"], design=v["note"]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "wp11b_status.csv"), index=False)
    return df


# ===========================================================================
# Part 2 — the observation-side overlay
# ===========================================================================
def _alpha_ref_coarse(F_int_coarse, trapz_annual_sum):
    """The best window-alpha the *annual* record can form, at coarse width.

    `alpha_obs` at width Delta divides the summed flow integral by
    `trapz(S)` over the whole window — two stock readings, Delta apart.  The
    annual record can do better: sum the annual trapezoids.  The gap between
    the two is precisely the quadrature error coarsening introduces, and it
    is the same construction on real data and on the twin.
    """
    return F_int_coarse / np.maximum(trapz_annual_sum, 1e-12)


def _trapz_annual_sums(years, S_parent, keep):
    """Sum of annual trapezoids of the parent stock inside each coarse window."""
    dt = np.diff(years)
    tr = 0.5 * (S_parent[:-1] + S_parent[1:]) * dt[:, None]
    return np.stack([tr[keep[i]:keep[i + 1]].sum(axis=0)
                     for i in range(keep.size - 1)])


def part2_observation(ctx, cfg, deltas=OVERLAP_DELTAS, out_dir=OUT_DIR):
    """The coarsening degradation of `alpha_obs`, real and twin, same statistic.

    On the twin the *true* window-mean alpha is available as well, so the
    proxy can be checked against what it is a proxy for.  If the proxy tracks
    the truth on the twin, a real-data proxy curve is informative about the
    real degradation; if it does not, this whole arm is uninformative and
    says so.
    """
    import zinc_colloc_v5 as v5

    parent = list(v5.ALPHA_PARENT_STOCK_IDX)
    rows = []

    # ---- real ----
    S.disarm()
    base1 = M.coarsen_real(1, cfg=cfg)
    yrs1 = base1["years"]
    Sp1 = base1["stocks_obs"][:, parent]
    for d in deltas:
        c = M.coarsen_real(int(d), cfg=cfg)
        tr = _trapz_annual_sums(yrs1, Sp1, c["keep"])
        a_ref = _alpha_ref_coarse(c["F_int"], tr)
        a_obs = c["alpha_obs"][1:]                     # row 0 is the copy
        tv = c["years"][1:] <= TRAINVAL_END
        for k, name in enumerate(ALPHA_NAMES):
            rows.append(dict(source="real", delta=float(d), channel=name,
                             n_windows=int(a_obs.shape[0]),
                             degradation_pct=rel_rmse_pct(a_obs[:, k],
                                                          a_ref[:, k], tv),
                             degradation_vs_truth_pct=np.nan))

    # ---- twin ----
    dense = S.dense_solve(ctx, "base")
    idx1 = S._window_indices(dense, 1.0)
    yrs_t = dense["t"][idx1]
    Sp_t = dense["S4"][idx1][:, parent]
    F1 = (dense["C"][idx1[1:]] - dense["C"][idx1[:-1]])[:, [1, 2, 3, 4]]
    for d in deltas:
        step = int(round(d))
        nW = (yrs_t.size - 1) // step
        keep = np.arange(nW + 1) * step
        F_c = np.stack([F1[keep[i]:keep[i + 1]].sum(axis=0) for i in range(nW)])
        tr = _trapz_annual_sums(yrs_t, Sp_t, keep)
        a_ref = _alpha_ref_coarse(F_c, tr)
        samp = S.sample(dense, float(d), ctx, noise_scale=0.0)
        a_obs = samp["alpha_obs"][1:]
        bias = S.alpha_target_bias(ctx, dense, float(d), arm_name="base")
        a_true = bias["a_unweighted"]
        tv = samp["years"][1:] <= TRAINVAL_END
        for k, name in enumerate(ALPHA_NAMES):
            rows.append(dict(source="twin", delta=float(d), channel=name,
                             n_windows=int(a_obs.shape[0]),
                             degradation_pct=rel_rmse_pct(a_obs[:, k],
                                                          a_ref[:, k], tv),
                             degradation_vs_truth_pct=rel_rmse_pct(
                                 a_obs[:, k], a_true[:, k], tv)))
        import jax
        jax.clear_caches()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "wp11b_observation.csv"), index=False)
    return df


# ===========================================================================
# Part 3 — the information-side overlay
# ===========================================================================
def _record_fim_curve(fit, params, deltas, sigma, *, lam_rel=1e-4, n_sub=12,
                      data=None):
    """Marginal SD of every reported coefficient for a record observed at
    each width in `deltas`.

    `C_Delta` stacks the four stocks (point-in-time) and the nine reported
    flows (period integrals), each row divided by its calibrated relative
    sigma, re-aggregated from one monthly solve by `zinc_voi_lab`'s own
    operators.  The ridge `mu` is fixed once, from the Delta = 1 spectrum, so
    that every width is read under the same prior and the curve measures data
    and not regularisation.
    """
    import jax
    import zinc_voi_lab as V
    import zinc_fisher_lab as F

    theta, _un = jax.flatten_util.ravel_pytree(params)
    theta = np.asarray(theta, float)
    # The curvature and the targets must live on the same window: the
    # observation Jacobian defaults to `fit.data_trainval`, so the coefficient
    # Jacobian is pinned to it too rather than to `fit.data_all`, which would
    # silently mix in the pre-registered test window.
    data = fit.data_trainval if data is None else data
    J, y, lay = V.observable_jacobian(fit, theta, data=data, n_sub=n_sub)
    logS, JS, logF, JF, _lE, _JE = V.split_observables(J, y, lay)

    g = F.make_coeff_fn(fit, data=data)
    G = np.asarray(jax.jit(jax.jacrev(g))(theta))          # (T, 17, n_par)
    T, n_coef, n_par = G.shape
    Q = G.reshape(-1, n_par)

    def C_at(step):
        rows = []
        for j, name in enumerate(lay["stock_names"]):
            _lv, Jv = V._agg_point(logS[:, j], JS[:, j], step)
            rows.append(Jv / max(sigma["stock"][name], 1e-12))
        for j, name in enumerate(lay["flow_names"]):
            _lv, Jv = V._agg_integral(logF[:, j], JF[:, j], step)
            s = sigma["flow"].get(name, 0.0)
            if s <= 0.0:                     # pinned: carries no information
                continue
            rows.append(Jv / s)
        C = np.concatenate(rows, axis=0)
        return C[np.any(C != 0.0, axis=1)]

    # Ridge, fixed once from the Delta = 1 spectrum so every width is read
    # under the same prior.  The top eigenvalue is enough and is taken by
    # power iteration rather than a full 2410-dimensional eigendecomposition.
    C1 = C_at(lay["n_sub"])
    A1 = C1.T @ C1
    v = np.ones(A1.shape[0]) / np.sqrt(A1.shape[0])
    for _ in range(200):
        w = A1 @ v
        n = np.linalg.norm(w)
        if n < 1e-300:
            break
        v = w / n
    mu = float(lam_rel * max(float(v @ (A1 @ v)), 1e-30))

    out = {}
    for d in deltas:
        step = int(round(d * lay["n_sub"]))
        if step < 1 or step > (lay["n_node"] - 1):
            continue
        from scipy.linalg import solve_triangular
        C = C_at(step)
        A = (A1.copy() if step == lay["n_sub"] else C.T @ C)
        A[np.diag_indices_from(A)] += mu
        L = np.linalg.cholesky(A)
        Z = solve_triangular(L, Q.T, lower=True)
        var = np.einsum("ij,ij->j", Z, Z).reshape(T, n_coef)
        out[float(d)] = np.sqrt(np.maximum(var, 0.0))
    return out, lay, F.COEF_NAMES


def part3_fisher(cfg, seeds, deltas=None, out_dir=OUT_DIR, n_sub=12):
    """The same curve on the published real fit and on WP-3's twin fit."""
    import zinc_cf_lab as cflab
    import zinc_voi_lab as V

    deltas = deltas or tuple(sorted(set(FINE_DELTAS) | set(OVERLAP_DELTAS)))
    sigma = V.load_sigma()
    rows = []

    for source, weights, arm_ds in (
            ("real", os.path.join(WEIGHTS_DIR, "A_seed{s}.npz"), None),
            ("twin", os.path.join(FIT_DIR, "base_d1y_clean_seed{s}.npz"),
             os.path.join(SYNTH_DIR, "base_d1y_clean.npz"))):
        if arm_ds is None:
            S.disarm()
        else:
            S.arm(S.load_dataset(arm_ds))
        fit = cflab.init_fit(cfg=cfg, seed=0)
        for s in seeds:
            p = weights.format(s=s)
            if not os.path.exists(p):
                continue
            t0 = time.time()
            params = cflab.load_params(p)
            sd, lay, coef_names = _record_fim_curve(
                fit, params, deltas, sigma, n_sub=n_sub)
            for d, arr in sd.items():
                for j, cn in enumerate(coef_names):
                    rows.append(dict(source=source, seed=int(s), delta=float(d),
                                     coefficient=cn,
                                     marginal_sd=float(np.median(arr[:, j]))))
            print(f"[wp11b] {source} seed {s}: {time.time() - t0:.0f}s",
                  flush=True)
            import jax
            jax.clear_caches()
        S.disarm()

    df = pd.DataFrame(rows)
    if df.empty:
        return df, df
    df.to_csv(os.path.join(out_dir, "wp11b_fisher_per_seed.csv"), index=False)
    base = df[np.abs(df.delta - 1.0) < 1e-9].set_index(
        ["source", "seed", "coefficient"]).marginal_sd
    df["sd_rel_to_annual"] = [
        r.marginal_sd / max(base.get((r.source, r.seed, r.coefficient), np.nan), 1e-30)
        for r in df.itertuples()]
    summ = df.groupby(["source", "delta", "coefficient"]).agg(
        n_seeds=("seed", "nunique"),
        marginal_sd=("marginal_sd", "median"),
        sd_rel_to_annual=("sd_rel_to_annual", "median"),
        sd_rel_q1=("sd_rel_to_annual", lambda v: np.percentile(v, 25)),
        sd_rel_q3=("sd_rel_to_annual", lambda v: np.percentile(v, 75))).reset_index()
    summ.to_csv(os.path.join(out_dir, "wp11b_fisher.csv"), index=False)
    return df, summ


# ===========================================================================
# Part 4 — evaluate the gate
# ===========================================================================
def _curve_from(df, source, value_col, key_col="channel"):
    out = {}
    for k, sub in df[df.source == source].groupby(key_col):
        sub = sub.sort_values("delta")
        out[k] = (sub.delta.to_numpy(), sub[value_col].to_numpy())
    return out


# The two refit curves the gate compares do not exist yet, so the loader
# states the schema it needs rather than guessing at read time.  Both are
# expected long-form with one row per (channel, delta): a `channel` column
# holding the four alpha names, a `delta` column in years (WP-11a may report
# `obs_per_year` instead, which is inverted here), and a metric column named
# by `GATE_METRIC` — the alpha-recovery error the spec's "understanding
# value" half asks for.  If WP-6b or WP-11a land with different column names,
# change this function and nothing else.
GATE_METRIC = "alpha_relRMSE_pct"


def load_refit_curve(path, metric=GATE_METRIC):
    """Long-form (channel, delta, metric) from a WP-6b / WP-11a output."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "delta" not in df.columns:
        if "obs_per_year" in df.columns:
            df["delta"] = 1.0 / df["obs_per_year"].astype(float)
        elif "window_yr" in df.columns:
            df["delta"] = df["window_yr"].astype(float)
        else:
            raise KeyError(f"{path}: no `delta`, `obs_per_year` or `window_yr` "
                           f"column; the gate cannot align the curves")
    if metric not in df.columns:
        raise KeyError(f"{path}: no `{metric}` column.  Available: "
                       f"{sorted(df.columns)}")
    keep = df[df.channel.isin(ALPHA_NAMES)]
    g = keep.groupby(["channel", "delta"])[metric].median().reset_index()
    return g.rename(columns={metric: "value"})


def _selftest_norm(curves):
    """Divide each channel's curve by its own Delta = 1 value."""
    out = {}
    for c, (d, v) in curves.items():
        at1 = np.asarray(v, float)[np.isclose(np.asarray(d, float), 1.0)]
        out[c] = (np.asarray(d, float),
                  np.asarray(v, float) / float(at1[0]) if at1.size else
                  np.asarray(v, float))
    return out


def gate_selftest():
    """Feed the criteria constructed curve pairs and check the verdicts.

    The gate has to be trustworthy before it is applied.  It is exercised in
    **both** modes, because the two are not the same test and the difference
    is load-bearing:

    *absolute* -- the criteria on raw levels.  A pair differing by a constant
    factor of three fails (G2), which is what an earlier version of this
    function asserted, and it is the sense in which the two surrogate arms'
    observation-side comparison was made.

    *normalised* -- the criteria after dividing each curve by its own
    Delta = 1 value, which is what the pre-registered statistic at the top of
    this file actually specifies ("normalised to its own value at Delta = 1 yr
    so that the comparison is of shape and not of level").  Under it a
    constant factor of three **passes by construction**, and that is asserted
    here rather than left to be discovered: it is the deliberate consequence
    of a shape test, and it is why `part4_gate` also reports the raw-level
    comparison as a separate, explicitly non-gate row.  What a shape test must
    still catch is a curve that *degrades at a different rate* -- the
    `steeper_shape` case -- and a curve that moves the wrong way.
    """
    d = np.array([1.0, 2.0, 5.0, 10.0])
    real = {c: (d, np.array([1.0, 2.0, 4.0, 8.0])) for c in ALPHA_NAMES}
    cases = {
        "match": {c: (d, np.array([1.0, 2.2, 4.4, 8.8])) for c in ALPHA_NAMES},
        "factor_three": {c: (d, np.array([3.0, 6.0, 12.0, 24.0]))
                         for c in ALPHA_NAMES},
        "steeper_shape": {c: (d, np.array([1.0, 3.0, 12.0, 40.0]))
                          for c in ALPHA_NAMES},
        "wrong_sign": {c: (d, np.array([8.0, 4.0, 2.0, 1.0]))
                       for c in ALPHA_NAMES},
    }
    want = {
        "absolute": {"match": True, "factor_three": False,
                     "steeper_shape": False, "wrong_sign": False},
        # a pure level shift is *expected* to pass a shape test
        "normalised": {"match": True, "factor_three": True,
                       "steeper_shape": False, "wrong_sign": False},
    }
    out = {}
    for mode in ("absolute", "normalised"):
        r_ = _selftest_norm(real) if mode == "normalised" else real
        for name, synth in cases.items():
            s_ = _selftest_norm(synth) if mode == "normalised" else synth
            rows = []
            _eval_into(rows, name, r_, s_)
            df = pd.DataFrame(rows)
            got = bool(df.channel_pass.sum() >= GATE_MIN_CHANNELS)
            exp = want[mode][name]
            out[f"{mode}/{name}"] = dict(
                expected=exp, got=got, ok=(got == exp),
                n_pass=int(df.channel_pass.sum()))
    return out


def _eval_into(rows, tag, real, synth):
    """Apply (G1) and (G2) to one pair of curve dicts, appending to `rows`."""
    for ch in sorted(set(real) & set(synth)):
        dr, vr = real[ch]
        ds, vs = synth[ch]
        common = np.array(sorted(set(np.round(dr, 6)) & set(np.round(ds, 6))))
        if common.size < 2:
            continue
        r = np.array([vr[np.argmin(np.abs(dr - c))] for c in common])
        s = np.array([vs[np.argmin(np.abs(ds - c))] for c in common])
        mono_r = bool(np.all(np.diff(r) >= -GATE_MONOTONE_TOL * np.abs(r[:-1])))
        mono_s = bool(np.all(np.diff(s) >= -GATE_MONOTONE_TOL * np.abs(s[:-1])))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = s / r
        fin = np.isfinite(ratio) & (r > 0)
        inband = bool(np.all((ratio[fin] >= 1.0 / GATE_RATIO)
                             & (ratio[fin] <= GATE_RATIO))) if fin.any() else False
        rows.append(dict(arm=tag, channel=ch,
                         n_common_deltas=int(common.size),
                         deltas=",".join(f"{c:g}" for c in common),
                         real=",".join(f"{x:.3g}" for x in r),
                         synth=",".join(f"{x:.3g}" for x in s),
                         ratio_median=float(np.nanmedian(ratio[fin])) if fin.any() else np.nan,
                         ratio_min=float(np.nanmin(ratio[fin])) if fin.any() else np.nan,
                         ratio_max=float(np.nanmax(ratio[fin])) if fin.any() else np.nan,
                         G1_monotone=bool(mono_r and mono_s),
                         G2_ratio_in_band=inband,
                         channel_pass=bool(mono_r and mono_s and inband)))


def part4_gate(obs_df, fisher_summ, status, out_dir=OUT_DIR):
    """Apply the pre-registered criteria to whatever curves exist.

    The refit gate is evaluated only if both WP-6b and WP-11a are present.
    The two no-fit arms are evaluated against the same criteria and reported
    as *surrogate* verdicts, explicitly labelled, because passing them is not
    the same thing as passing the gate.
    """
    rows = []

    def _eval(tag, real, synth):
        _eval_into(rows, tag, real, synth)

    if obs_df is not None and not obs_df.empty:
        _eval("surrogate_observation",
              _curve_from(obs_df, "real", "degradation_pct"),
              _curve_from(obs_df, "twin", "degradation_pct"))

    if fisher_summ is not None and not fisher_summ.empty:
        f = fisher_summ[fisher_summ.delta.isin(OVERLAP_DELTAS)]
        f = f[f.coefficient.isin(ALPHA_NAMES)].rename(
            columns={"coefficient": "channel"})
        _eval("surrogate_information",
              _curve_from(f, "real", "sd_rel_to_annual"),
              _curve_from(f, "twin", "sd_rel_to_annual"))

    # the gate itself, if the refit curves have landed
    r6 = load_refit_curve(os.path.join(out_dir, "wp6b_coarsening.csv"))
    r11 = load_refit_curve(os.path.join(out_dir, "wp11a_frequency_value.csv"))
    have_gate = (r6 is not None) and (r11 is not None)
    if have_gate:
        def _as_dict(df, normalise):
            out = {}
            for c, sub in df.groupby("channel"):
                sub = sub[sub.delta.isin(OVERLAP_DELTAS)].sort_values("delta")
                d, v = sub.delta.to_numpy(), sub.value.to_numpy().astype(float)
                if normalise:
                    at1 = v[np.isclose(d, 1.0)]
                    if at1.size and np.isfinite(at1[0]) and at1[0] > 0:
                        v = v / float(at1[0])
                out[c] = (d, v)
            return out
        # The pre-registration above is explicit that the statistic is the
        # degradation curve "normalised to its own value at Delta = 1 yr so
        # that the comparison is of shape and not of level".  The two
        # surrogate arms already satisfy that by construction -- the Fisher
        # arm's metric *is* `sd_rel_to_annual`, and the observation arm's is
        # identically zero at Delta = 1 so it cannot be normalised (flag 1) --
        # and this is the arm where the clause has to be applied by hand.
        _eval("wp11b_gate_curves", _as_dict(r6, True), _as_dict(r11, True))
        # Reported alongside, and NOT the gate: the same criteria applied to
        # WP-11a's *truth* currency.  The twin can score its fits against the
        # analytic alpha and the real record cannot, so this is not a
        # like-for-like comparison and cannot be the gate -- WP-6b has no such
        # column and never will.  It is computed because it is the obvious
        # thing a reader will wonder about, and because the answer is not the
        # same: see the verdict table.  Nothing is reselected on it.
        r11t = load_refit_curve(
            os.path.join(out_dir, "wp11a_frequency_value.csv"),
            metric="alpha_relRMSE_true_pct")
        if r11t is not None:
            _eval("gate_curves_truth_currency_NOT_the_gate",
                  _as_dict(r6, True), _as_dict(r11t, True))
        # Reported alongside, and NOT the gate: the same criteria on the raw
        # levels.  The twin is not a fit and WP-3 flag 4 says its level is not
        # expected to match, which is exactly why the pre-registration
        # normalises -- but a reader is entitled to see what the un-normalised
        # comparison says rather than take the choice on trust.
        _eval("gate_curves_absolute_level",
              _as_dict(r6, False), _as_dict(r11, False))
    df = pd.DataFrame(rows)
    verdicts = []
    for tag, sub in (df.groupby("arm") if not df.empty else []):
        if tag == "wp11b_gate_curves":
            continue                      # reported once, below, as the gate
        n_pass = int(sub.channel_pass.sum())
        verdicts.append(dict(arm=tag, n_channels=int(len(sub)),
                             n_pass=n_pass,
                             threshold=GATE_MIN_CHANNELS,
                             verdict="PASS" if n_pass >= GATE_MIN_CHANNELS else "FAIL",
                             is_the_gate=False))
    if have_gate:
        sub = df[df.arm == "wp11b_gate_curves"]
        n_pass = int(sub.channel_pass.sum())
        gate_verdict = "PASS" if n_pass >= GATE_MIN_CHANNELS else "FAIL"
        n_ch = int(len(sub))
    else:
        gate_verdict = "OPEN — WP-6b and WP-11a not run (592 fits, CX3)"
        n_pass = n_ch = 0
    verdicts.append(dict(arm="wp11b_gate", n_channels=n_ch, n_pass=n_pass,
                         threshold=GATE_MIN_CHANNELS,
                         verdict=gate_verdict, is_the_gate=True))
    vf = pd.DataFrame(verdicts)
    if not df.empty:
        df.to_csv(os.path.join(out_dir, "wp11b_overlap.csv"), index=False)
    vf.to_csv(os.path.join(out_dir, "wp11b_verdict.csv"), index=False)
    return df, vf


# ===========================================================================
# figures
# ===========================================================================
def figures(obs_df, fisher_summ, verdict, out_dir=OUT_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 4, figsize=(17.2, 4.2))

    a0 = ax[0]
    if obs_df is not None and not obs_df.empty:
        for i, ch in enumerate(ALPHA_NAMES):
            r = obs_df[(obs_df.source == "real") & (obs_df.channel == ch)].sort_values("delta")
            t = obs_df[(obs_df.source == "twin") & (obs_df.channel == ch)].sort_values("delta")
            a0.plot(r.delta, r.degradation_pct, "o-", color=COL[i], lw=1.3, ms=4,
                    label=ch)
            a0.plot(t.delta, t.degradation_pct, "s--", color=COL[i], lw=1.0,
                    ms=4, alpha=0.75)
    a0.set_xscale("log"); a0.set_yscale("log")
    a0.set_xlabel(r"observation width $\Delta$ [yr]")
    a0.set_ylabel(r"degradation of $\alpha_{\rm obs}$ [%]")
    a0.set_title("(a) observation side\nsolid = real, dashed = twin")
    a0.legend(fontsize=7, frameon=False)

    a1 = ax[1]
    if fisher_summ is not None and not fisher_summ.empty:
        for i, ch in enumerate(ALPHA_NAMES):
            for src, ls, mk in (("real", "-", "o"), ("twin", "--", "s")):
                s = fisher_summ[(fisher_summ.source == src)
                                & (fisher_summ.coefficient == ch)].sort_values("delta")
                if s.empty:
                    continue
                a1.plot(s.delta, s.sd_rel_to_annual, mk + ls, color=COL[i],
                        lw=1.2 if src == "real" else 1.0, ms=4,
                        alpha=1.0 if src == "real" else 0.75,
                        label=ch if src == "real" else None)
    a1.axvline(1.0, color=GREY, ls=":", lw=1.0)
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlabel(r"observation width $\Delta$ [yr]")
    a1.set_ylabel(r"marginal SD relative to $\Delta=1$")
    a1.set_title("(b) information side\nleft of the line is the extrapolation")
    a1.legend(fontsize=7, frameon=False)

    # (c) THE GATE: the two refit curves, on the pre-registered statistic
    a2 = ax[2]
    r6 = load_refit_curve(os.path.join(out_dir, "wp6b_coarsening.csv"))
    r11 = load_refit_curve(os.path.join(out_dir, "wp11a_frequency_value.csv"))
    have = (r6 is not None) and (r11 is not None)
    if have:
        for i, ch in enumerate(ALPHA_NAMES):
            for df_, ls, mk, src in ((r6, "-", "o", "real"),
                                     (r11, "--", "s", "twin")):
                sub = df_[(df_.channel == ch)
                          & df_.delta.isin(OVERLAP_DELTAS)].sort_values("delta")
                if sub.empty:
                    continue
                v = sub.value.to_numpy(float)
                at1 = v[np.isclose(sub.delta.to_numpy(float), 1.0)]
                if at1.size and at1[0] > 0:
                    v = v / float(at1[0])
                a2.plot(sub.delta, v, mk + ls, color=COL[i], lw=1.2, ms=4,
                        alpha=1.0 if src == "real" else 0.75,
                        label=ch if src == "real" else None)
        a2.set_xscale("log"); a2.set_yscale("log")
        a2.legend(fontsize=7, frameon=False)
    else:
        a2.text(0.5, 0.5, "refit curves not both present", ha="center",
                va="center", fontsize=8, color=GREY, transform=a2.transAxes)
    a2.set_xlabel(r"observation width $\Delta$ [yr]")
    a2.set_ylabel(r"$\alpha$ relRMSE relative to $\Delta=1$")
    a2.set_title("(c) the gate: refit curves\nsolid = real (WP-6b), "
                 "dashed = twin (WP-11a)")

    a3 = ax[3]
    a3.set_axis_off()
    lines = ["WP-11b gate", ""]
    if verdict is not None and not verdict.empty:
        for _, r in verdict.iterrows():
            mark = "**" if r.is_the_gate else "  "
            lines.append(f"{mark} {r.arm}: {r.verdict}")
            if not r.is_the_gate:
                lines.append(f"     {r.n_pass}/{r.n_channels} channels "
                             f"(threshold {r.threshold})")
    lines += ["", f"pre-registered band: ratio in "
                  f"[1/{GATE_RATIO:g}, {GATE_RATIO:g}]",
              f"on >= {GATE_MIN_CHANNELS} of 4 channels",
              "normalised to each curve's own", "value at Delta = 1 yr", ""]
    st = part1_status(out_dir)
    for _, r in st.iterrows():
        lines.append(f"{r.curve}: {'present' if r.present else 'MISSING'}")
    a3.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=8,
            family="monospace", transform=a3.transAxes)

    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"wp11b_overlap.{e}"), dpi=200)
    plt.close(fig)


# ===========================================================================
# CLI
# ===========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--observation", action="store_true")
    ap.add_argument("--fisher", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    a = ap.parse_args(argv)

    import zinc_colloc_v5 as v5
    M.install(v5)
    cfg = M.load_anchor_config()
    ctx = S.driver_context(cfg)
    n_ex = len(ctx["exog_cols"])
    orders = len(cfg.get("exog_feature_orders", (0, 1)))
    print(f"[wp11b] drivers: {list(S.DRIVER_ALIASES.values())}")
    print(f"[wp11b] exog columns = {n_ex}, feature orders = {orders}, "
          f"input_dim = {1 + 4 + n_ex * orders}  "
          f"(CLAUDE.md rule 2 says 23 — SCHEMA §9 flag 1, open)")

    if a.check:
        M.check(verbose=True)
        print("\n--- gate self-test (constructed curves) ---")
        for k, v in gate_selftest().items():
            print(f"  {k:26s} expected {v['expected']!s:5s} "
                  f"got {v['got']!s:5s} "
                  f"({v['n_pass']}/4 channels)  {'ok' if v['ok'] else 'FAILED'}")
        return 0

    status = part1_status()
    print(status.to_string(index=False))

    obs_df = fsumm = None
    if a.observation or a.all:
        obs_df = part2_observation(ctx, cfg)
        print(obs_df.to_string(index=False))
    if a.fisher or a.all:
        seeds = [int(x) for x in a.seeds.split(",") if x != ""]
        _, fsumm = part3_fisher(cfg, seeds)
        if fsumm is not None and not fsumm.empty:
            print(fsumm[fsumm.coefficient.isin(ALPHA_NAMES)].to_string(index=False))
    if a.gate or a.figures or a.all:
        if obs_df is None:
            p = os.path.join(OUT_DIR, "wp11b_observation.csv")
            obs_df = pd.read_csv(p) if os.path.exists(p) else None
        if fsumm is None:
            p = os.path.join(OUT_DIR, "wp11b_fisher.csv")
            fsumm = pd.read_csv(p) if os.path.exists(p) else None
    verdict = None
    if a.gate or a.all:
        ov, verdict = part4_gate(obs_df, fsumm, status)
        if not ov.empty:
            print(ov.to_string(index=False))
        print(verdict.to_string(index=False))
    if a.figures or a.all:
        if verdict is None:
            p = os.path.join(OUT_DIR, "wp11b_verdict.csv")
            verdict = pd.read_csv(p) if os.path.exists(p) else None
        figures(obs_df, fsumm, verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
