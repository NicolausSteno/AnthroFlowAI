#!/usr/bin/env python3
"""
run_wp4b.py — WP-4b: misspecification absorption
================================================

The spec's brief is two sentences: *"Put a dynamic in the truth the model
cannot represent (distributed delay, or a capacity gate on `alpha_refc`).  If
alpha absorbs it into a smooth, plausible-looking but wrong trajectory, that is
a named caution for the dynamic-MFA literature."*  This driver measures each
clause of that sentence separately, because the caution is only worth naming if
all of them hold at once: the fit has to look *good*, the coefficient has to
look *smooth and plausible*, and it has to be *wrong* — and wrong in a way that
costs something a modeller would care about.

The misspecification
--------------------
`zinc_synth_lab`'s `state` arm, generated and frozen in WP-3.  A capacity gate

    alpha_refc(t) <- alpha_refc(t) * exp( -gamma * (S_ref(t)/S_ref_bar - 1) ),
    gamma = 0.35                                  (`S.TRUTH["state"]`)

The rate out of the refined stock falls as refined inventory builds.  It is
smooth, bounded and physically ordinary — a plant running closer to capacity,
or a destocking rule.  `anchor_v4` runs with `use_stock_input: false`, so the
network never sees `S` at all: this dynamic is outside its representable set by
construction, and not by an accident of capacity.  The `base` arm is the same
truth with `gamma = 0`, is inside the representable set, and is the control
throughout — every state-arm number in this file is reported against its
seed-paired base-arm twin.

The five parts
--------------
Part 1  **the representability ceiling** — analytic, no fits.  The gate term
        `g(t) = -gamma*(S_ref/S_ref_bar - 1)` is a fixed time series once the
        twin is solved.  Regress it on the 26 exogenous features the network
        actually sees, fitting on the pre-registered trainval window and
        scoring on the held-out one.  This bounds *any* stock-blind estimator,
        the UDE included, and it is where the answer to WP-4b already is:
        in-sample the gate is almost perfectly absorbable, out of sample the
        absorption is not merely imperfect but divergent.

Part 2  **the fits** — `state_d1y_{clean,noisy}` x 8 seeds, the unmodified
        `anchor_v4` pipeline armed on the twin.  Seed-matched to WP-3's
        existing `base_d1y_{clean,noisy}` fits.

Part 3  **absorption, scored six ways** — (a) does any standard goodness-of-fit
        diagnostic notice; (b) where does the coefficient error land; (c) how
        much of `g` does alpha_refc actually absorb, against Part 1's ceiling;
        (d) does the error contaminate the other channels through mass balance;
        (e) is the absorbed trajectory smooth, i.e. "plausible-looking" made
        numerical; (f) what does a modeller who fits the truth's own functional
        form to the estimated coefficient conclude — the turning point, the
        trend amplitude, and the driver elasticities they would report.

Part 4  **the cost** — a counterfactual.  Transfer coefficients exist to be
        used off the observed path.  Two shocks: a permanent step in
        concentrate production (which moves `S_ref` and nothing the network
        sees, so the true coefficient responds and the estimated one *cannot*),
        and a permanent shift in GDP (which moves both, and is the shock the
        absorption is built to handle).  The contrast between the two is the
        finding, and the response *asymmetry* in the sign of the shock is the
        signature of the feedback that survives absorption intact.

Part 4b **the stability boundary.**  The gate is a positive feedback in the
        destocking direction, so the truth has an adverse-shock threshold past
        which the refining route stalls.  A coefficient that is a fixed
        function of the drivers has no feedback and therefore no threshold at
        all, at any shock size.  That is a qualitative failure rather than a
        magnitude error, and it is the one scenario work would care about most.

Part 5  figures and the tables the note quotes.

Lab module
----------
None is added.  CLAUDE.md rule 1 asks that behaviour changes go through
import-time patching in a lab module; WP-4b introduces no new behaviour change.
The single patch in play is `zinc_synth_lab`'s arm-aware `load_zinc_data`
dispatcher, already written and checked in WP-3, and everything else here
composes `zinc_interp_lab` (driver perturbation and the frozen-weight alpha
evaluator), `zinc_cf_lab` (weight loading, exog rebuild) and the core's own
`make_rhs` / `_make_Y0` / `ode_to_4obs`.  `zinc_colloc_v5.py` is untouched.

    python run_wp4b.py --check
    python run_wp4b.py --fit --tags state_d1y_clean,state_d1y_noisy
    python run_wp4b.py --gate --absorb --counterfactual --figures

`--counterfactual` also runs Part 4b.  The `state` arm datasets themselves come
from WP-3 (`zinc_synth_lab --generate`) and are not regenerated here.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import zinc_synth_lab as S
import zinc_interp_lab as I
import zinc_cf_lab as cflab
import zinc_circ_lab as C

import contextlib
import io as _io


@contextlib.contextmanager
def _quiet():
    """`cflab.init_fit` re-runs the core's zero-step summary; keep it out of
    the log (same reason `run_wp5._quiet` exists)."""
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "analysis")
SYNTH_DIR = os.path.join(OUT_DIR, "synth")
FIT_DIR = os.path.join(SYNTH_DIR, "fits")

COL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
GREY = "#555555"

ALPHA_NAMES = S.ALPHA_NAMES
TAU_SUP_NAMES = S.TAU_SUP_NAMES
K_REFC = ALPHA_NAMES.index("alpha_refc")        # the gated channel

TRAINVAL_END = 2007.0
SEEDS_DEFAULT = (0, 1, 2, 3, 4, 5, 6, 7)
ARM_PAIRS = (("state_d1y_clean", "base_d1y_clean"),
             ("state_d1y_noisy", "base_d1y_noisy"))


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------
def hl(v):
    """`(estimate, lo, hi)` Hodges-Lehmann; NaN interval below six points."""
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if v.size == 0:
        return (np.nan, np.nan, np.nan)
    d = C.hodges_lehmann(v)
    return (float(d["hl"]), float(d["lo"]), float(d["hi"]))


def rel_rmse_pct(pred, obs, mask=None):
    p, o = np.asarray(pred, float), np.asarray(obs, float)
    m = np.isfinite(p) & np.isfinite(o)
    if mask is not None:
        m &= np.asarray(mask, bool)
    if not m.any():
        return np.nan
    return 100.0 * np.sqrt(np.mean((p[m] - o[m]) ** 2)) / max(np.mean(np.abs(o[m])), 1e-12)


def _log(x):
    return np.log(np.maximum(np.asarray(x, float), 1e-12))


def gate_series(ctx, dense):
    """`g(t) = -gamma*(S_ref/S_ref_bar - 1)` on the dense grid.

    `S_ref_bar` is the mean of the *real* refined stock, which is what
    `make_truth_nn_eval` closes over; recomputing it the same way here is the
    check that this reconstruction is the gate the generator actually applied
    rather than a lookalike.  `verify_gate` proves the identity.
    """
    gamma = float(S.TRUTH["state"]["gamma"])
    S_ref_bar = float(np.mean(ctx["stocks_obs"][:, 1]))
    return -gamma * (np.asarray(dense["S4"][:, 1], float) / S_ref_bar - 1.0)


def verify_gate(ctx, dense_state):
    """`alpha_refc_state = alpha_refc_base * exp(g)` to round-off, on the state
    arm's own trajectory.  If this fails, nothing downstream means anything."""
    t = np.asarray(dense_state["t"], float)
    a_state = S.truth_coefficients(ctx, t, "state",
                                   S_path=dense_state["S4"])["alphas"][:, K_REFC]
    a_base = S.truth_coefficients(ctx, t, "base")["alphas"][:, K_REFC]
    g = gate_series(ctx, dense_state)
    return float(np.max(np.abs(_log(a_state) - _log(a_base) - g)))


# ===========================================================================
# Part 1 — the representability ceiling
# ===========================================================================
RIDGE_LAMBDAS = (0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def _ridge_fit(X, y, tr, lam):
    """Ridge on `[1, X]` fitted on rows `tr`; the intercept is not penalised."""
    A = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    At, yt = A[tr], np.asarray(y, float)[tr]
    P = lam * np.eye(A.shape[1]); P[0, 0] = 0.0
    w = np.linalg.solve(At.T @ At + P, At.T @ yt)
    return A @ w, w


def _r2(y, p, m):
    y, p, m = np.asarray(y, float), np.asarray(p, float), np.asarray(m, bool)
    den = np.sum((y[m] - y[m].mean()) ** 2)
    return float(1.0 - np.sum((y[m] - p[m]) ** 2) / max(den, 1e-30))


def part1_gate(out_dir=OUT_DIR, tag="state_d1y_clean"):
    """The gate, and how much of it lives in the span of what the model sees.

    Two statements, and they point in opposite directions, which is the whole
    of WP-4b in miniature:

      * the gate is *large* — a factor-of-two modulation of `alpha_refc`;
      * on the training window it is *almost exactly* a function of the
        drivers, so a stock-blind estimator can absorb it without strain;
      * off that window the same function diverges, because the co-movement it
        exploited between the refined stock and the activity drivers is a
        feature of the sample and not of the mechanism.

    The regression is deliberately *linear*.  A linear projection is a lower
    bound on what a 2 250-weight MLP can do, so a high in-sample linear `R^2`
    is the stronger statement: the absorption needs no capacity argument at
    all.
    """
    ctx = S.driver_context()
    cfg = S.load_anchor_config()
    ds = S.load_dataset(os.path.join(SYNTH_DIR, f"{tag}.npz"))
    dense = dict(np.load(os.path.join(SYNTH_DIR, "state_dense.npz"),
                         allow_pickle=True))

    ident = verify_gate(ctx, dense)
    g_dense = gate_series(ctx, dense)
    yrs = np.asarray(ds["years"], float).ravel()
    g = np.interp(yrs, np.asarray(dense["t"], float), g_dense)

    # the feature matrix the network is handed, rebuilt through the core
    t_full = np.asarray(ds["exog_times_full"], float)
    X_full = np.asarray(ds["exog_values_full"], float)
    cut, _ = cflab.split_cut(len(yrs), cfg.get("trainval_frac", 0.7),
                             cfg.get("val_frac", 0.2))
    F = np.asarray(cflab.rebuild_exog(cfg, yrs, t_full, X_full, cut), float)
    # `exog_values` on the dataset is the *raw* 13-column matrix the loader
    # returns; `preprocess_exog` runs inside `v5.run`.  So the cheap check
    # available here is that the order-0 feature block is log1p of those raw
    # columns at the annual nodes.  The full bit-for-bit comparison against
    # the fit's own feature matrix is `ude_context`'s, and Part 4 runs it.
    n_raw = X_full.shape[1]
    gap = float(np.max(np.abs(
        F[:, :n_raw] - np.log1p(np.maximum(np.asarray(ds["exog_values"], float), 0.0)))))

    tv = yrs <= TRAINVAL_END
    cols = [str(c) for c in ds["exog_cols"]]

    rows = []
    for lam in RIDGE_LAMBDAS:
        p, _ = _ridge_fit(F, g, tv, lam)
        rows.append(dict(
            lam=lam, n_features=F.shape[1], n_trainval=int(tv.sum()),
            r2_trainval=_r2(g, p, tv), r2_test=_r2(g, p, ~tv),
            rmse_trainval_nats=float(np.sqrt(np.mean((g[tv] - p[tv]) ** 2))),
            rmse_test_nats=float(np.sqrt(np.mean((g[~tv] - p[~tv]) ** 2))),
            sd_gate_trainval=float(g[tv].std()), sd_gate_test=float(g[~tv].std())))
    proj = pd.DataFrame(rows)

    # levels only (13 features) — what a modeller reading elasticities sees
    rows_lvl = []
    for lam in RIDGE_LAMBDAS:
        p, _ = _ridge_fit(F[:, :len(cols)], g, tv, lam)
        rows_lvl.append(dict(lam=lam, r2_trainval=_r2(g, p, tv),
                             r2_test=_r2(g, p, ~tv)))
    proj_lvl = pd.DataFrame(rows_lvl)

    corr = pd.DataFrame([
        dict(driver=cols[j],
             corr_level=float(np.corrcoef(F[:, j], g)[0, 1]),
             corr_level_trainval=float(np.corrcoef(F[tv, j], g[tv])[0, 1]),
             corr_level_test=float(np.corrcoef(F[~tv, j], g[~tv])[0, 1]))
        for j in range(len(cols))]).sort_values(
            "corr_level", key=np.abs, ascending=False).reset_index(drop=True)

    summ = pd.DataFrame([dict(
        gamma=float(S.TRUTH["state"]["gamma"]),
        S_ref_bar=float(np.mean(ctx["stocks_obs"][:, 1])),
        gate_min_nats=float(g_dense.min()), gate_max_nats=float(g_dense.max()),
        gate_sd_nats=float(g_dense.std()),
        alpha_factor_min=float(np.exp(g_dense.min())),
        alpha_factor_max=float(np.exp(g_dense.max())),
        S_ref_min=float(dense["S4"][:, 1].min()),
        S_ref_max=float(dense["S4"][:, 1].max()),
        corr_gate_time=float(np.corrcoef(yrs, g)[0, 1]),
        identity_max_abs_nats=ident, exog_rebuild_gap=gap,
        input_dim=int(1 + 4 + F.shape[1]))])

    summ.to_csv(os.path.join(out_dir, "wp4b_gate.csv"), index=False)
    proj.to_csv(os.path.join(out_dir, "wp4b_projection.csv"), index=False)
    proj_lvl.to_csv(os.path.join(out_dir, "wp4b_projection_levels.csv"), index=False)
    corr.to_csv(os.path.join(out_dir, "wp4b_gate_driver_corr.csv"), index=False)
    return dict(summary=summ, proj=proj, proj_levels=proj_lvl, corr=corr,
                years=yrs, gate=g, gate_dense=g_dense, features=F,
                trainval=tv, cols=cols)


# ===========================================================================
# Part 2 — the fits
# ===========================================================================
def fit_one(seed, tag, cfg, out_dir=FIT_DIR, verbose=False):
    """One unmodified `anchor_v4` fit against an armed synthetic dataset.

    Identical to `run_wp3.fit_one` — same config, same dump, same directory —
    so the state-arm fits and WP-3's base-arm fits are produced by one code
    path and the paired comparison in Part 3 is like-for-like.
    """
    import jax
    import zinc_colloc_v5 as v5
    import zinc_A_lab as Alab

    os.makedirs(out_dir, exist_ok=True)
    S.arm(S.load_dataset(os.path.join(SYNTH_DIR, f"{tag}.npz")))
    t0 = time.time()
    fit = v5.run(f"wp4b_{tag}_s{seed}", **dict(cfg, seed=int(seed), verbose=verbose))
    wall = time.time() - t0
    path = os.path.join(out_dir, f"{tag}_seed{seed}.npz")
    Alab.dump_A(fit, path, stage="B")
    S.disarm()
    del fit
    jax.clear_caches()                                   # CLAUDE.md rule 6
    return path, wall


def run_fits(tags, seeds, out_dir=FIT_DIR):
    cfg = S.load_anchor_config()
    for tag in tags:
        for seed in seeds:
            p = os.path.join(out_dir, f"{tag}_seed{seed}.npz")
            if os.path.exists(p):
                print(f"[fit] {tag} seed {seed} exists, skipping", flush=True)
                continue
            _, wall = fit_one(seed, tag, cfg)
            print(f"[fit] {tag} seed {seed}  {wall:6.1f}s  "
                  f"rss {S._rss_mb():.0f} MB", flush=True)


# ===========================================================================
# Part 3 — absorption, scored
# ===========================================================================
def _truth_for(tag):
    """Point-in-time truth arrays for a twin dataset, plus the no-gate
    counterfactual truth on the *same* trajectory.

    `alphas_nogate` is the base functional form at the same years — i.e. what
    `alpha_refc` would have been with `gamma = 0` — so
    `log(alphas) - log(alphas_nogate)` is exactly the gate on the state arm and
    exactly zero on the base arm.  That zero is what makes the base fits a
    control rather than merely a comparison.
    """
    ctx = S.driver_context()
    ds = S.load_dataset(os.path.join(SYNTH_DIR, f"{tag}.npz"))
    tr = S.load_truth(os.path.join(SYNTH_DIR, f"{tag}.npz"))
    yrs = np.asarray(ds["years"], float).ravel()
    arm = str(json.loads(str(ds["meta_json"]))["arm"]) if "meta_json" in ds \
        else tag.split("_")[0]
    a_nogate = S.truth_coefficients(ctx, yrs, "base")["alphas"]
    return dict(ctx=ctx, ds=ds, truth=tr, years=yrs, arm=arm,
                alphas=tr["alphas_point"], tau_sup=tr["tau_sup_point"],
                alphas_nogate=a_nogate)


def _roughness(y):
    """Second-difference roughness of a log-coefficient path, per year^2, and
    the total variation of its first difference.  Both are the quantities an
    eye is doing when it calls a trajectory 'smooth'."""
    y = np.asarray(y, float)
    d2 = np.diff(y, 2)
    return float(np.sqrt(np.mean(d2 ** 2))), float(np.sum(np.abs(np.diff(y))))


def part3_absorb(pairs=ARM_PAIRS, seeds=SEEDS_DEFAULT, out_dir=OUT_DIR,
                 fit_dir=FIT_DIR):
    """(a) fit quality, (b) coefficient error, (c) absorption, (d)
    contamination, (e) smoothness — per seed, per arm, seed-paired."""
    rows, traj = [], {}
    for state_tag, base_tag in pairs:
        for tag in (state_tag, base_tag):
            T = _truth_for(tag)
            yrs, tv = T["years"], T["years"] <= TRAINVAL_END
            g_true = _log(T["alphas"][:, K_REFC]) - _log(T["alphas_nogate"][:, K_REFC])
            for seed in seeds:
                p = os.path.join(fit_dir, f"{tag}_seed{seed}.npz")
                if not os.path.exists(p):
                    continue
                d = np.load(p, allow_pickle=True)
                a_pred = np.asarray(d["alphas"], float)
                t_pred = np.concatenate(
                    [d["taus"], d["frac_fu"][:, :2], d["frac_eu"][:, :2]], axis=1)
                S_pred = np.asarray(d["S_pred"], float)
                S_obs = np.asarray(d["S_obs"], float)
                traj[(tag, seed)] = dict(years=yrs, alphas=a_pred)

                # (c) absorption: the deviation the fit produces relative to
                # the no-gate truth, regressed on the gate it should equal.
                dev = _log(a_pred[:, K_REFC]) - _log(T["alphas_nogate"][:, K_REFC])
                rec = dict(tag=tag, arm=T["arm"], seed=int(seed))
                for split, m in (("trainval", tv), ("test", ~tv), ("all", np.ones_like(tv))):
                    gm, dm = g_true[m], dev[m]
                    sl = (float(np.sum((gm - gm.mean()) * (dm - dm.mean()))
                                / max(np.sum((gm - gm.mean()) ** 2), 1e-30))
                          if gm.std() > 1e-9 else np.nan)
                    rec[f"absorb_slope_{split}"] = sl
                    rec[f"absorb_r2_{split}"] = (
                        _r2(gm, dm, np.ones(len(gm), bool)) if gm.std() > 1e-9 else np.nan)
                    rec[f"dev_sd_{split}"] = float(dm.std())
                    rec[f"gate_sd_{split}"] = float(gm.std())
                    rec[f"resid_rms_{split}"] = float(np.sqrt(np.mean((dm - gm) ** 2)))

                # (a) fit quality — the diagnostics that actually get reported.
                # Split by window, because "would a modeller notice?" has two
                # different answers: an in-sample diagnostic is what dynamic
                # MFA normally reports, a held-out one is what it normally does
                # not, and the two need not agree.  The lag-1 autocorrelation
                # of the stock residual is the standard misspecification test
                # and is reported for the same reason.
                SN = ["Concentrate", "Refined", "In-Use", "Scrap"]
                for k, sn in enumerate(SN):
                    rec[f"stock_relRMSE_{sn}"] = rel_rmse_pct(S_pred[:, k], S_obs[:, k])
                    for split, m in (("trainval", tv), ("test", ~tv)):
                        rec[f"stock_relRMSE_{sn}_{split}"] = rel_rmse_pct(
                            S_pred[:, k], S_obs[:, k], m)
                    r = _log(S_pred[:, k]) - _log(S_obs[:, k])
                    r = r - r.mean()
                    rec[f"acf1_{sn}"] = float(
                        np.sum(r[1:] * r[:-1]) / max(np.sum(r ** 2), 1e-30))
                rec["stock_relRMSE_mean"] = float(np.mean(
                    [rec[f"stock_relRMSE_{s}"] for s in SN]))
                for split in ("trainval", "test"):
                    rec[f"stock_relRMSE_mean_{split}"] = float(np.mean(
                        [rec[f"stock_relRMSE_{s}_{split}"] for s in SN]))
                rec["acf1_mean"] = float(np.mean([rec[f"acf1_{s}"] for s in SN]))

                # (b)+(d) coefficient error, all channels, both splits
                for k, name in enumerate(ALPHA_NAMES):
                    for split, m in (("trainval", tv), ("test", ~tv)):
                        rec[f"alpha_{name}_{split}"] = rel_rmse_pct(
                            a_pred[:, k], T["alphas"][:, k], m)
                        rec[f"alpha_{name}_{split}_vs_nogate"] = rel_rmse_pct(
                            a_pred[:, k], T["alphas_nogate"][:, k], m)
                for j, name in enumerate(TAU_SUP_NAMES):
                    rec[f"tau_{name}_trainval"] = rel_rmse_pct(
                        t_pred[:, j], T["tau_sup"][:, j], tv)

                # (e) smoothness of the estimated coefficient
                for k, name in enumerate(ALPHA_NAMES):
                    r, tvv = _roughness(_log(a_pred[:, k]))
                    rec[f"rough_{name}"] = r
                    rec[f"tv_{name}"] = tvv
                rows.append(rec)

    per_seed = pd.DataFrame(rows)
    if per_seed.empty:
        return per_seed, per_seed, traj
    per_seed.to_csv(os.path.join(out_dir, "wp4b_absorption_per_seed.csv"), index=False)

    # the truth's own smoothness, for the "plausible-looking" comparison
    Ts = _truth_for(pairs[0][0])
    ref = []
    for k, name in enumerate(ALPHA_NAMES):
        r, tvv = _roughness(_log(Ts["alphas"][:, k]))
        rn, tvn = _roughness(_log(Ts["alphas_nogate"][:, k]))
        ref.append(dict(channel=name, truth_state_roughness=r, truth_state_tv=tvv,
                        truth_nogate_roughness=rn, truth_nogate_tv=tvn))
    pd.DataFrame(ref).to_csv(
        os.path.join(out_dir, "wp4b_truth_smoothness.csv"), index=False)

    # seed-paired state-minus-base contrasts, Hodges-Lehmann
    contrasts = []
    num = [c for c in per_seed.columns
           if c not in ("tag", "arm", "seed") and
           per_seed[c].dtype.kind == "f"]
    for state_tag, base_tag in pairs:
        a = per_seed[per_seed.tag == state_tag].set_index("seed")
        b = per_seed[per_seed.tag == base_tag].set_index("seed")
        common = sorted(set(a.index) & set(b.index))
        if not common:
            continue
        def _med(v):
            v = np.asarray(v, float); v = v[np.isfinite(v)]
            return float(np.median(v)) if v.size else np.nan

        for c in num:
            dv = (a.loc[common, c] - b.loc[common, c]).to_numpy(float)
            est, lo, hi = hl(dv)
            contrasts.append(dict(
                pair=f"{state_tag} - {base_tag}", quantity=c, n_seeds=len(common),
                state_median=_med(a.loc[common, c]),
                base_median=_med(b.loc[common, c]),
                diff_hl=est, diff_lo=lo, diff_hi=hi))
    contrast = pd.DataFrame(contrasts)
    contrast.to_csv(os.path.join(out_dir, "wp4b_contrasts.csv"), index=False)
    return per_seed, contrast, traj


# ---------------------------------------------------------------------------
# 3f — what a modeller reading the estimated coefficient would conclude
# ---------------------------------------------------------------------------
# The truth's own functional form for alpha_refc (`S.TRUTH["alpha"]`):
#
#   log alpha_refc(t) = log A + m*(tanh((t - tstar)/w) - centre(tstar, w))
#                       + beta_GDP * z_GDP(t) + beta_ZnPrice * z_ZnPrice(t)
#
# Fitting it *back* to an estimated coefficient path is what a dynamic-MFA
# modeller does when they report "the transfer coefficient turned in year X and
# responds to GDP with elasticity Y".  On the base arm the exercise recovers
# the registered constants; on the state arm it recovers something else, and
# the difference is the substantive claim the misspecification corrupts.
STRUCT_P0 = ("logA", "tstar", "w", "m", "beta_GDP", "beta_ZnPrice")
STRUCT_TRUTH = dict(logA=float(np.log(S.TRUTH["alpha"]["alpha_refc"]["A"])),
                    tstar=float(S.TRUTH["alpha"]["alpha_refc"]["tstar"]),
                    w=float(S.TRUTH["alpha"]["alpha_refc"]["w"]),
                    m=float(S.TRUTH["alpha"]["alpha_refc"]["m"]),
                    beta_GDP=float(S.TRUTH["alpha"]["alpha_refc"]["beta"]["GDP"]),
                    beta_ZnPrice=float(S.TRUTH["alpha"]["alpha_refc"]["beta"]["ZnPrice"]))


def _struct_fit(years, logalpha, ctx, mask=None, all_years=None):
    """Least-squares recovery of the six structural constants.

    `centre` is evaluated over the *full* 1980-2019 window exactly as
    `S._centre` does, so `logA` is comparable with `TRUTH`'s `A` and is not
    absorbing a window-dependent offset.
    """
    from scipy.optimize import least_squares

    years = np.asarray(years, float)
    all_years = years if all_years is None else np.asarray(all_years, float)
    m = np.ones(years.size, bool) if mask is None else np.asarray(mask, bool)
    y = np.asarray(logalpha, float)[m]
    t = years[m]
    zg = np.interp(t, ctx["years"], ctx["z"]["GDP"])
    zp = np.interp(t, ctx["years"], ctx["z"]["ZnPrice"])

    def resid(p):
        logA, tstar, w, mm, bg, bp = p
        c = float(np.mean(np.tanh((all_years - tstar) / w)))
        return (logA + mm * (np.tanh((t - tstar) / w) - c)
                + bg * zg + bp * zp) - y

    p0 = np.array([y.mean(), 2000.0, 8.0, 0.3, 0.0, 0.0])
    lo = np.array([-np.inf, 1975.0, 1.0, -5.0, -5.0, -5.0])
    hi = np.array([np.inf, 2025.0, 40.0, 5.0, 5.0, 5.0])
    r = least_squares(resid, p0, bounds=(lo, hi), max_nfev=20000)
    out = dict(zip(STRUCT_P0, [float(v) for v in r.x]))
    out["rms_resid_nats"] = float(np.sqrt(np.mean(r.fun ** 2)))
    out["A"] = float(np.exp(out["logA"]))
    return out


def part3f_structure(pairs=ARM_PAIRS, seeds=SEEDS_DEFAULT, out_dir=OUT_DIR,
                     fit_dir=FIT_DIR):
    ctx = S.driver_context()
    rows = []
    for state_tag, base_tag in pairs:
        for tag in (state_tag, base_tag):
            T = _truth_for(tag)
            yrs, tv = T["years"], T["years"] <= TRAINVAL_END
            # the truth itself, as a reference row per arm
            for lbl, arr in (("truth", T["alphas"][:, K_REFC]),
                             ("truth_nogate", T["alphas_nogate"][:, K_REFC])):
                r = _struct_fit(yrs, _log(arr), ctx, tv, yrs)
                rows.append(dict(tag=tag, source=lbl, seed=-1, **r))
            for seed in seeds:
                p = os.path.join(fit_dir, f"{tag}_seed{seed}.npz")
                if not os.path.exists(p):
                    continue
                a = np.load(p, allow_pickle=True)["alphas"][:, K_REFC]
                r = _struct_fit(yrs, _log(np.asarray(a, float)), ctx, tv, yrs)
                rows.append(dict(tag=tag, source="fit", seed=int(seed), **r))
    df = pd.DataFrame(rows)
    if df.empty:
        return df, df
    df.to_csv(os.path.join(out_dir, "wp4b_structure_per_seed.csv"), index=False)
    g = df[df.source == "fit"].groupby("tag")
    summ = g[list(STRUCT_P0) + ["A", "rms_resid_nats"]].median().reset_index()
    for c in STRUCT_P0:
        est, lo, hi = zip(*[hl(sub[c].to_numpy()) for _, sub in g])
        summ[f"{c}_hl_lo"], summ[f"{c}_hl_hi"] = lo, hi
        summ[f"{c}_truth"] = STRUCT_TRUTH[c]
    summ.to_csv(os.path.join(out_dir, "wp4b_structure.csv"), index=False)
    return df, summ


# ===========================================================================
# Part 4 — the cost: a counterfactual
# ===========================================================================
# A transfer coefficient exists in order to be used off the observed path.
# Parts 1-3 establish that the misspecification is absorbed in-sample; this
# part asks what the absorbed coefficient does when the system is moved.
#
# Two shocks, chosen so that the contrast between them is diagnostic:
#
#   cp    a permanent step in concentrate production from `SHOCK_YEAR`.  It
#         moves `S_ref` and therefore the true `alpha_refc` through the gate,
#         and it moves *nothing the network reads* — `learn_cp: false` pins cp
#         to data and `use_stock_input: false` hides the state — so the
#         estimated coefficient is provably inert.  The response error is
#         therefore the entire true response.
#
#   GDP   a permanent shift of the GDP driver.  It moves the true coefficient
#         both directly (beta_GDP) and indirectly (through S_ref and the gate),
#         and it moves the estimated one directly.  This is the shock the
#         absorption is built for, and the arm where it does best.
#
# Both are scored as *differences*: the counterfactual response
# `X(t; shock) - X(t; 0)`, which nets out the baseline fit error and leaves
# only the response error.
SHOCK_YEAR = 2000.0
CP_KAPPAS = (-0.20, -0.10, 0.10, 0.20)
# A -1 SD GDP shock pushes the *gated truth* out of its stable regime:
# lower GDP lowers alpha_refc directly (beta_GDP = +0.22), refined inventory
# builds, and the gate lowers alpha_refc further until the refining route
# stalls and refined metal simply accumulates.  That positive feedback is a
# real property of the truth and is measured on its own in `part4b_stability`
# rather than being allowed to contaminate a response table with a trajectory
# from a different regime.
# +-0.2 SD: inside the truth's stable region, whose boundary `part4b_stability`
# locates between -0.25 and -0.50 SD.  Larger adverse shocks do not merely
# amplify the response, they change its regime, and a response *ratio* is not
# a meaningful summary across a regime change.
GDP_DELTAS = (-0.2, 0.2)
GDP_STABILITY_SWEEP = (-1.0, -0.75, -0.5, -0.45, -0.40, -0.35, -0.30,
                       -0.25, -0.20, -0.10, 0.10, 0.25, 0.5, 1.0)


def _make_solver(ev, rtol=S.TIGHT_RTOL, atol=S.TIGHT_ATOL,
                 max_steps=S.TIGHT_MAXSTEPS, dt0=0.1):
    """`(params, rhs_data, S0, ts) -> (S4, alphas, C)` for any `nn_eval`.

    Composed from `make_rhs` / `_make_Y0` / `ode_to_4obs` exactly as
    `zinc_synth_lab.dense_solve` and `zinc_xai_lab.make_forward_integrator` do
    — the core is read-only, so a solver at a chosen tolerance is *composed*
    rather than obtained by editing `make_integrator`.  The truth closure and
    the fitted network satisfy the same `nn_eval` contract, so the two solves
    differ in the coefficient function and in nothing else.
    """
    import jax
    import jax.numpy as jnp
    import diffrax as dfx
    import zinc_colloc_v5 as v5

    rhs = v5.make_rhs(ev)

    def solve(params, rhs_data, S0, ts):
        ts = jnp.asarray(np.asarray(ts, float))

        def f(t, y, args):
            p, dd = args
            return rhs(y, t, p, dd)

        sol = dfx.diffeqsolve(
            dfx.ODETerm(f), dfx.Tsit5(), t0=ts[0], t1=ts[-1],
            # `make_integrator` (v5:1348) hard-codes dt0 = 0.1.  With an
            # adaptive controller at rtol = 1e-5 the first step choice
            # propagates into the whole step sequence, so matching it is what
            # makes the composition check come out at round-off rather than at
            # a few parts in a thousand.
            dt0=jnp.asarray(float(dt0)),
            y0=v5._make_Y0(jnp.asarray(S0, float)),
            args=(params, rhs_data), saveat=dfx.SaveAt(ts=ts),
            stepsize_controller=dfx.PIDController(rtol=rtol, atol=atol),
            max_steps=max_steps)
        Y = sol.ys
        S_ode = Y[:, :v5.N_ODE_STOCKS]
        S4 = v5.ode_to_4obs(S_ode)
        S_coh = S_ode[:, 2:2 + v5.N_COHORTS]

        def coef(t, s4, s_c):
            ex = v5.exog_fn(t, rhs_data["exog_times"], rhs_data["exog_values"])
            return ev(params, t, s4, ex, rhs_data)["alphas"]

        A = jax.vmap(coef)(ts, S4, S_coh)
        return np.asarray(S4), np.asarray(A), np.asarray(Y[:, v5.N_ODE_STOCKS:])

    return solve


def _shocked_data(data, kind, amount, cfg, uctx):
    """A copy of the RHS `data` dict carrying one shock.

    `cp` scales the pinned concentrate-production series from `SHOCK_YEAR` on;
    `GDP` shifts the driver's whole raw history by `amount` within-sample SDs
    in log1p space and re-runs `preprocess_exog` through `cflab.rebuild_exog`,
    which is the same permanent-shift convention WP-5 and WP-8b use.
    """
    import jax.numpy as jnp

    out = dict(data)
    if kind == "cp":
        cp = np.asarray(data["cp_obs"], float).copy()
        yrs = np.asarray(data["years"], float).ravel()
        cp[yrs > SHOCK_YEAR] *= (1.0 + float(amount))
        out["cp_obs"] = jnp.asarray(cp)
    elif kind == "GDP":
        j = uctx["cols"].index(S.DRIVER_ALIASES["GDP"])
        w = np.zeros(len(uctx["cols"])); w[j] = 1.0
        Xp = I.perturb_many(uctx["X_full"], w, float(amount), uctx["sigma"])
        out["exog_values"] = jnp.asarray(cflab.rebuild_exog(
            cfg, uctx["years"], uctx["t_full"], Xp, uctx["cut"]))
    elif kind != "none":
        raise ValueError(kind)
    return out


def _shocked_ctx(ctx, kind, amount, uctx):
    """The same shock in the *truth's* coordinates.

    The truth reads drivers from `ctx["z"]` (log1p, z-scored) and never from
    `exog_values`, so a GDP shift of `amount` SDs is `z_GDP += amount` — the
    same size of move as `perturb_many`'s, because `z` is scaled by the SD of
    the same log1p series `driver_sigma` uses.  `check()` verifies the two
    agree.
    """
    if kind != "GDP":
        return ctx
    out = dict(ctx)
    z = {k: v.copy() for k, v in ctx["z"].items()}
    z["GDP"] = z["GDP"] + float(amount)
    out["z"] = z
    return out


def part4_counterfactual(pairs=ARM_PAIRS, seeds=SEEDS_DEFAULT, out_dir=OUT_DIR,
                         fit_dir=FIT_DIR):
    """The counterfactual response error, truth versus fitted, per seed.

    For each arm the truth solve is seed-independent and is run once.  Both
    solves consume the *same* `rhs_data` — the armed twin's own pins and
    drivers — so the only difference between them is the coefficient closure.
    Two correctness gates are computed before any shock is applied: the truth
    solve must reproduce the generator's stored dense trajectory, and the
    fitted solve must reproduce the trajectory persisted in the fit's dump.
    """
    import jax
    import zinc_colloc_v5 as v5

    cfg = S.load_anchor_config()
    ctx0 = S.driver_context()
    rows, checks, curves, quality = [], [], [], []

    for state_tag, base_tag in pairs:
        for tag in (state_tag, base_tag):
            arm = tag.split("_")[0]
            with _quiet():
                fit, uctx = I.ude_context(tag, cfg)
            uctx["sigma"] = I.driver_sigma(ctx0)
            years = uctx["years"]
            data0 = dict(fit.data_all)
            S0 = np.asarray(data0["stocks_obs"], float)[0]

            truth_ev = S.make_truth_nn_eval(ctx0, arm)
            solve_truth = _make_solver(truth_ev)
            solve_fit = _make_solver(fit.nn_eval)
            # the same fitted solve at the *core's own* tolerance, used only to
            # separate a tolerance gap from a composition error in gate 2
            solve_fit_loose = _make_solver(fit.nn_eval, rtol=1e-5, atol=1e-7,
                                           max_steps=20000)
            noisy = tag.endswith("noisy")

            def rd(d):
                return v5._make_rhs_data(d)

            # --- gate 1: the truth solve reproduces the generator ------------
            St0, At0, _ = solve_truth(None, rd(data0), S0, years)
            dz = np.load(os.path.join(SYNTH_DIR, f"{arm}_dense.npz"),
                         allow_pickle=True)
            ref = np.array([np.interp(years, dz["t"], dz["S4"][:, k])
                            for k in range(4)]).T
            gap_truth = float(np.max(np.abs(St0 - ref) / np.maximum(np.abs(ref), 1.0)))

            # --- gate 2: the fitted solve reproduces the fit's own -----------
            # The reference is `fit.integrate_aug` at the fit's own tolerance,
            # which is the object `zinc_xai_lab` also checks against; the
            # composed solver must reproduce it to round-off, and that is the
            # composition check.  The gap to the persisted `S_pred` is reported
            # alongside but is *not* a gate: `predictions("B")` is not the same
            # call, and the tight tolerance used for the counterfactuals moves
            # the trajectory on its own (COMPUTE_STATUS flag 3 records that
            # rtol = 1e-5 is not negligible against this model's residuals).
            gap_fit = gap_fit_loose = gap_dump = np.nan
            post = years > SHOCK_YEAR

            # --- the truth's shocked trajectories, once per tag --------------
            # They do not depend on the seed, and re-solving them inside the
            # seed loop would double the work and recompile a fresh solver for
            # every GDP amount.
            truth_cf = {}
            for kind, amounts in (("cp", CP_KAPPAS), ("GDP", GDP_DELTAS)):
                for amt in amounts:
                    dS = _shocked_data(data0, kind, amt, cfg, uctx)
                    cS = _shocked_ctx(ctx0, kind, amt, uctx)
                    sv = (_make_solver(S.make_truth_nn_eval(cS, arm))
                          if kind == "GDP" else solve_truth)
                    St, At, _ = sv(None, rd(dS), S0, years)
                    truth_cf[(kind, amt)] = (dS, St[:, 1] - St0[:, 1],
                                             _log(At[:, K_REFC]) - _log(At0[:, K_REFC]))

            for seed in seeds:
                p = os.path.join(fit_dir, f"{tag}_seed{seed}.npz")
                if not os.path.exists(p):
                    continue
                params = cflab.load_params(p)
                dump = np.load(p, allow_pickle=True)
                Sf0, Af0, _ = solve_fit(params, rd(data0), S0, years)
                Sref_own, _, _ = fit.integrate_aug(params, data0, S0, years)
                Sref_own = np.asarray(Sref_own, float)
                den = np.maximum(np.abs(Sref_own), 1.0)
                g = float(np.max(np.abs(Sf0 - Sref_own) / den))
                gap_fit = g if not np.isfinite(gap_fit) else max(gap_fit, g)
                Sl, _, Cl = solve_fit_loose(params, rd(data0), S0, years)
                gl = float(np.max(np.abs(Sl - Sref_own) / den))
                gap_fit_loose = gl if not np.isfinite(gap_fit_loose) else max(
                    gap_fit_loose, gl)
                Sdump = np.asarray(dump["S_pred"], float)
                gd = float(np.max(np.abs(Sl - Sdump)
                                  / np.maximum(np.abs(Sdump), 1.0)))
                gap_dump = gd if not np.isfinite(gap_dump) else max(gap_dump, gd)
                # Flow fit quality, free from the baseline solve.  Part 3a can
                # only reach the stocks (`dump_A` does not persist flows), and
                # "would a modeller notice?" has to be answered on the flows
                # too, since Stage B's divergence term is a flow term.
                F_int = np.asarray(Cl[1:] - Cl[:-1], float)
                # the observed->predicted flow column map lives on the
                # dataset, not on `data_all`
                idx = np.asarray(uctx["ds"]["flow_obs_to_pred_idx"], int)
                F_obs = np.asarray(fit.data_all["flows_obs"], float)
                quality.append(dict(
                    tag=tag, seed=int(seed),
                    stock_relRMSE=rel_rmse_pct(Sl, np.asarray(
                        fit.data_all["stocks_obs"], float)),
                    flow_relRMSE=rel_rmse_pct(F_int[:, idx], F_obs)))

                for kind, amounts in (("cp", CP_KAPPAS), ("GDP", GDP_DELTAS)):
                    for amt in amounts:
                        dS, rt_S, rt_a = truth_cf[(kind, amt)]
                        Sf, Af, _ = solve_fit(params, rd(dS), S0, years)

                        # responses: the counterfactual difference
                        rf_S = Sf[:, 1] - Sf0[:, 1]
                        rf_a = _log(Af[:, K_REFC]) - _log(Af0[:, K_REFC])
                        rows.append(dict(
                            tag=tag, arm=arm, seed=int(seed), shock=kind,
                            amount=float(amt),
                            true_alpha_resp=float(np.mean(rt_a[post])),
                            fit_alpha_resp=float(np.mean(rf_a[post])),
                            alpha_resp_ratio=float(np.mean(rf_a[post])
                                                   / (np.mean(rt_a[post]) + 1e-30)),
                            alpha_resp_relRMSE=rel_rmse_pct(rf_a, rt_a, post),
                            true_Sref_resp=float(np.mean(rt_S[post])),
                            fit_Sref_resp=float(np.mean(rf_S[post])),
                            Sref_resp_ratio=float(np.mean(rf_S[post])
                                                  / (np.mean(rt_S[post]) + 1e-30)),
                            Sref_resp_relRMSE=rel_rmse_pct(rf_S, rt_S, post),
                            Sref_resp_terminal_true=float(rt_S[-1]),
                            Sref_resp_terminal_fit=float(rf_S[-1])))
                        if seed == seeds[0] and kind == "cp" and amt == CP_KAPPAS[-1]:
                            curves.append(pd.DataFrame(dict(
                                tag=tag, shock=kind, amount=amt, year=years,
                                true_alpha_resp=rt_a, fit_alpha_resp=rf_a,
                                true_Sref_resp=rt_S, fit_Sref_resp=rf_S)))
                del params

            checks.append(dict(
                tag=tag, pins_noisy=bool(noisy),
                # for a noisy tag the pinned cp/tau the RHS reads carry the
                # dataset's observation noise, so the truth solve is *not* the
                # generator's clean trajectory and this number is that noise,
                # not a solver gap.  It is a gate only on the clean tags.
                gap_truth_vs_dense=gap_truth,
                gap_fit_vs_own_tight=gap_fit,
                gap_fit_vs_own_same_tol=gap_fit_loose,
                gap_vs_persisted_S_pred=gap_dump,
                gap_exog_rebuild=float(uctx["gap_exog"])))
            del fit
            jax.clear_caches()                               # CLAUDE.md rule 6
            print(f"[wp4b/cf] {tag} done, rss {S._rss_mb():.0f} MB", flush=True)

    per_seed = pd.DataFrame(rows)
    chk = pd.DataFrame(checks)
    chk.to_csv(os.path.join(out_dir, "wp4b_cf_checks.csv"), index=False)
    if quality:
        pd.DataFrame(quality).to_csv(
            os.path.join(out_dir, "wp4b_fit_quality.csv"), index=False)
    if per_seed.empty:
        return per_seed, per_seed, chk, per_seed
    per_seed.to_csv(os.path.join(out_dir, "wp4b_counterfactual_per_seed.csv"),
                    index=False)
    if curves:
        pd.concat(curves).to_csv(
            os.path.join(out_dir, "wp4b_cf_curves.csv"), index=False)
    # --- response asymmetry -------------------------------------------------
    # The gate makes the true response asymmetric in the sign of the shock:
    # a cp increase raises S_ref, which closes the gate and damps the rise,
    # while a cp cut opens it and amplifies the fall.  A coefficient that is a
    # fixed function of the drivers cannot do this — its response is odd in the
    # shock by construction — so the asymmetry index is a signature of the
    # feedback that survives the estimator's in-sample absorption intact.
    asym = []
    kap = max(CP_KAPPAS)
    for (tag, seed), sub in per_seed[per_seed.shock.eq("cp")].groupby(
            ["tag", "seed"]):
        up = sub[np.isclose(sub.amount, kap)]
        dn = sub[np.isclose(sub.amount, -kap)]
        if up.empty or dn.empty:
            continue

        def idx(a, b):
            return float((a + b) / max(abs(a - b), 1e-30))
        asym.append(dict(
            tag=tag, seed=int(seed), kappa=kap,
            truth_asym=idx(float(up.true_Sref_resp.iloc[0]),
                           float(dn.true_Sref_resp.iloc[0])),
            fit_asym=idx(float(up.fit_Sref_resp.iloc[0]),
                         float(dn.fit_Sref_resp.iloc[0]))))
    asym = pd.DataFrame(asym)
    if not asym.empty:
        asym.to_csv(os.path.join(out_dir, "wp4b_cf_asymmetry.csv"), index=False)

    g = per_seed.groupby(["tag", "shock", "amount"])
    summ = g.agg(n_seeds=("seed", "nunique"),
                 true_alpha_resp=("true_alpha_resp", "median"),
                 fit_alpha_resp=("fit_alpha_resp", "median"),
                 alpha_resp_relRMSE=("alpha_resp_relRMSE", "median"),
                 true_Sref_resp=("true_Sref_resp", "median"),
                 fit_Sref_resp=("fit_Sref_resp", "median"),
                 Sref_resp_ratio=("Sref_resp_ratio", "median"),
                 Sref_resp_relRMSE=("Sref_resp_relRMSE", "median")).reset_index()
    lo, hi = zip(*[hl(sub.Sref_resp_relRMSE.to_numpy())[1:] for _, sub in g])
    summ["Sref_relRMSE_hl_lo"], summ["Sref_relRMSE_hl_hi"] = lo, hi
    summ.to_csv(os.path.join(out_dir, "wp4b_counterfactual.csv"), index=False)
    return per_seed, summ, chk, asym


def part4b_stability(seeds=SEEDS_DEFAULT, out_dir=OUT_DIR, fit_dir=FIT_DIR,
                     tag="state_d1y_clean"):
    """Where the gated truth loses stability, and the fact that the estimated
    model has no such boundary.

    The gate is a *positive* feedback in the destocking direction: a shock that
    lowers `alpha_refc` lets refined inventory build, which lowers `alpha_refc`
    further.  Below some GDP shock the truth runs away.  A coefficient that is
    a fixed function of the drivers cannot run away at all, because nothing
    feeds back — so the estimated model is unconditionally stable and reports
    no boundary.  That is a qualitative failure, not a magnitude error, and it
    is the one a modeller using the coefficient for scenario work would care
    about most.
    """
    import jax
    import zinc_colloc_v5 as v5

    cfg = S.load_anchor_config()
    ctx0 = S.driver_context()
    arm = tag.split("_")[0]
    with _quiet():
        fit, uctx = I.ude_context(tag, cfg)
    uctx["sigma"] = I.driver_sigma(ctx0)
    years, data0 = uctx["years"], dict(fit.data_all)
    S0 = np.asarray(data0["stocks_obs"], float)[0]
    solve_fit = _make_solver(fit.nn_eval)

    seed = next((s_ for s_ in seeds
                 if os.path.exists(os.path.join(fit_dir, f"{tag}_seed{s_}.npz"))),
                None)
    params = (cflab.load_params(os.path.join(fit_dir, f"{tag}_seed{seed}.npz"))
              if seed is not None else None)

    base_max = None
    rows = []
    for amt in (0.0,) + tuple(GDP_STABILITY_SWEEP):
        dS = _shocked_data(data0, "GDP", amt, cfg, uctx)
        cS = _shocked_ctx(ctx0, "GDP", amt, uctx)
        St, At, _ = _make_solver(S.make_truth_nn_eval(cS, arm))(
            None, v5._make_rhs_data(dS), S0, years)
        r = dict(tag=tag, delta_sd=float(amt),
                 truth_Sref_max=float(np.max(St[:, 1])),
                 truth_stalled=bool(np.min(At[:, K_REFC]) < 1e-3),
                 truth_Sref_terminal=float(St[-1, 1]),
                 truth_alpha_refc_min=float(np.min(At[:, K_REFC])))
        if params is not None:
            Sf, Af, _ = solve_fit(params, v5._make_rhs_data(dS), S0, years)
            r.update(fit_seed=int(seed), fit_Sref_max=float(np.max(Sf[:, 1])),
                     fit_Sref_terminal=float(Sf[-1, 1]),
                     fit_alpha_refc_min=float(np.min(Af[:, K_REFC])))
        if amt == 0.0:
            base_max = r["truth_Sref_max"]
        r["truth_Sref_max_x_baseline"] = r["truth_Sref_max"] / max(base_max, 1e-9)
        if params is not None:
            r["fit_Sref_max_x_baseline"] = (
                r["fit_Sref_max"] / max(rows[0]["fit_Sref_max"], 1e-9)
                if rows else 1.0)
        rows.append(r)
    del fit
    jax.clear_caches()
    df = pd.DataFrame(rows).sort_values("delta_sd").reset_index(drop=True)
    df.to_csv(os.path.join(out_dir, "wp4b_stability.csv"), index=False)
    return df


# ===========================================================================
# --check
# ===========================================================================
def check(verbose=True):
    """Resolve drivers and `input_dim`, prove the gate identity, and prove the
    truth's driver coordinates and the perturbation's agree — before any work.
    """
    info = S.check(verbose=False)
    ctx = S.driver_context()
    dense = dict(np.load(os.path.join(SYNTH_DIR, "state_dense.npz"),
                         allow_pickle=True))
    info["gate_identity_max_abs_nats"] = verify_gate(ctx, dense)

    # a +1 SD shift applied through `perturb_many` must move `z_GDP` by +1
    sig = I.driver_sigma(ctx)
    j = list(ctx["exog_cols"]).index(S.DRIVER_ALIASES["GDP"])
    w = np.zeros(len(ctx["exog_cols"])); w[j] = 1.0
    Xp = I.perturb_many(ctx["exog_values"], w, 1.0, sig)
    col = np.log1p(np.maximum(Xp[:, j], 0.0))
    c0 = np.log1p(np.maximum(ctx["exog_values"][:, j], 0.0))
    info["z_shift_per_unit_delta"] = float(np.mean(
        (col - c0) / max(c0.std(), 1e-9)))

    g = gate_series(ctx, dense)
    info["gate_sd_nats"] = float(g.std())
    info["gate_factor_range"] = [float(np.exp(g.min())), float(np.exp(g.max()))]

    present = {t: sum(os.path.exists(os.path.join(FIT_DIR, f"{t}_seed{s}.npz"))
                      for s in SEEDS_DEFAULT)
               for pair in ARM_PAIRS for t in pair}
    info["fits_present"] = present

    if verbose:
        print("=" * 74)
        print("run_wp4b --check")
        print("=" * 74)
        for label, (got, ok) in info["digests"].items():
            print(f"  {label:20s} md5 {got}  {'OK' if ok else 'MISMATCH'}")
        print(f"  patches applied      : {S.PATCHES}")
        print(f"\n  resolved drivers (canonical order, {info['n_universe']}):")
        for i, c in enumerate(info["exog_cols"]):
            mark = ""
            for short, name in S.DRIVER_ALIASES.items():
                if name == c:
                    mark = f"   <- z[{short}]"
            print(f"    [{i:2d}] {c}{mark}")
        print(f"\n  exog_feature_orders  : {info['orders']}")
        print(f"  input_dim            : 1 (t) + 4 (S) + "
              f"{info['n_exog_features']} = {info['input_dim']}")
        if info["input_dim"] != 23:
            print(f"  !! input_dim is {info['input_dim']}, not the 23 asserted by "
                  f"CLAUDE.md rule 2 (SCHEMA §9 flag 1, unresolved).")
        print(f"\n  misspecification     : capacity gate on "
              f"{S.TRUTH['state']['channel']}, gamma = {S.TRUTH['state']['gamma']}")
        print(f"  gate identity        : max |log a_state - log a_base - g| = "
              f"{info['gate_identity_max_abs_nats']:.3e}")
        print(f"  gate size            : sd {info['gate_sd_nats']:.4f} nats, "
              f"alpha_refc factor {info['gate_factor_range'][0]:.3f} .. "
              f"{info['gate_factor_range'][1]:.3f}")
        print(f"  use_stock_input      : "
              f"{S.load_anchor_config().get('use_stock_input')}  "
              f"(the gate is unrepresentable by construction)")
        print(f"  driver shift scaling : +1.0 delta -> {info['z_shift_per_unit_delta']:+.6f} "
              f"z units (must be +1)")
        print(f"\n  shocks               : cp {list(CP_KAPPAS)} from {SHOCK_YEAR:.0f}; "
              f"GDP {list(GDP_DELTAS)} SD")
        print("  fits on disk         : " + ", ".join(
            f"{k} {v}/{len(SEEDS_DEFAULT)}" for k, v in present.items()))
        print("=" * 74)
    return info


# ===========================================================================
# figures
# ===========================================================================
def figures(gate, per_seed, contrast, cf_summ, struct, traj, out_dir=OUT_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11.0, 7.6))

    # (a) the gate, and its in-sample linear shadow in the driver span
    a = ax[0, 0]
    yrs, g, F, tv = gate["years"], gate["gate"], gate["features"], gate["trainval"]
    p, _ = _ridge_fit(F, g, tv, 0.0)
    a.plot(yrs, g, color=COL[0], lw=2.0, label="capacity gate $g(t)$")
    a.plot(yrs, p, color=COL[1], lw=1.6, ls="--",
           label="best linear function of the\ndrivers (fitted on trainval)")
    a.axvspan(TRAINVAL_END, yrs[-1], color=GREY, alpha=0.10)
    a.text(TRAINVAL_END + 0.4, a.get_ylim()[1], " held out", va="top",
           fontsize=8, color=GREY)
    a.set_ylim(min(g.min(), p[tv].min()) - 0.25, max(g.max(), p[tv].max()) + 0.35)
    a.set_ylabel(r"$\log$ multiplier on $\alpha_{\rm refc}$")
    a.set_title("(a) the misspecification is a driver function in-sample",
                fontsize=10, loc="left")
    a.legend(fontsize=7.5, frameon=False, loc="lower left")

    # (b) alpha_refc: truth, no-gate truth, and the fits
    a = ax[0, 1]
    T = _truth_for(ARM_PAIRS[0][0])
    a.plot(T["years"], T["alphas"][:, K_REFC], color=COL[0], lw=2.4, zorder=5,
           label=r"truth (gated)")
    a.plot(T["years"], T["alphas_nogate"][:, K_REFC], color=GREY, lw=1.5,
           ls=":", zorder=4, label=r"truth without the gate")
    lab = True
    for (tag, seed), d in traj.items():
        if not tag.startswith("state") or "clean" not in tag:
            continue
        a.plot(d["years"], d["alphas"][:, K_REFC], color=COL[1], lw=0.9,
               alpha=0.65, label="UDE fits (8 seeds)" if lab else None)
        lab = False
    a.axvspan(TRAINVAL_END, T["years"][-1], color=GREY, alpha=0.10)
    a.set_ylabel(r"$\alpha_{\rm refc}$  (yr$^{-1}$)")
    a.set_title("(b) absorbed in-sample, and smooth", fontsize=10, loc="left")
    a.legend(fontsize=7.5, frameon=False)

    # (c) coefficient error by channel, state vs base
    a = ax[1, 0]
    if per_seed is not None and not per_seed.empty:
        ps = per_seed[per_seed.tag.str.endswith("clean")]
        w, xs = 0.36, np.arange(len(ALPHA_NAMES))
        for i, (pref, c, lbl) in enumerate(
                [("base", GREY, "base arm (representable)"),
                 ("state", COL[1], "state arm (gate)")]):
            sub = ps[ps.arm == pref]
            if sub.empty:
                continue
            med = [np.nanmedian(sub[f"alpha_{n}_trainval"]) for n in ALPHA_NAMES]
            q1 = [np.nanpercentile(sub[f"alpha_{n}_trainval"], 25) for n in ALPHA_NAMES]
            q3 = [np.nanpercentile(sub[f"alpha_{n}_trainval"], 75) for n in ALPHA_NAMES]
            a.bar(xs + (i - 0.5) * w, med, w, color=c, label=lbl,
                  yerr=[np.array(med) - q1, np.array(q3) - np.array(med)],
                  error_kw=dict(lw=0.9, capsize=2.5, ecolor="#333333"))
        a.set_xticks(xs)
        a.set_xticklabels([n.replace("alpha_", r"$\alpha_{\rm ") + "}$"
                           for n in ALPHA_NAMES], fontsize=8)
    a.set_ylabel("relRMSE vs truth, trainval (%)")
    a.set_title("(c) where the error lands", fontsize=10, loc="left")
    a.legend(fontsize=7.5, frameon=False)

    # (d) counterfactual response, truth vs fitted
    a = ax[1, 1]
    if cf_summ is not None and not cf_summ.empty:
        sub = cf_summ[cf_summ.tag.eq("state_d1y_clean")].copy()
        sub = sub[np.isfinite(sub.true_Sref_resp) & np.isfinite(sub.fit_Sref_resp)]
        for kind, c, mk in (("cp", COL[1], "o"), ("GDP", COL[2], "s")):
            s2 = sub[sub.shock == kind].sort_values("true_Sref_resp")
            if s2.empty:
                continue
            a.plot(s2.true_Sref_resp, s2.fit_Sref_resp, mk + "-", color=c,
                   ms=5, lw=1.2, label=f"{kind} shock")
        lim = np.nanmax(np.abs(np.r_[sub.true_Sref_resp, sub.fit_Sref_resp])) * 1.15
        a.plot([-lim, lim], [-lim, lim], color=GREY, lw=1.0, ls="--",
               label="perfect counterfactual")
        a.axhline(0, color="#999999", lw=0.7); a.axvline(0, color="#999999", lw=0.7)
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
    a.set_xlabel("true response of $S_{\\rm ref}$ (kt)")
    a.set_ylabel("estimated response (kt)")
    a.set_title("(d) the cost: counterfactual response", fontsize=10, loc="left")
    a.legend(fontsize=7.5, frameon=False, loc="upper left")

    for r in ax:
        for a_ in r:
            a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"wp4b_absorption.{ext}"),
                    dpi=220, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# main
# ===========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="WP-4b driver")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--fit", action="store_true", help="run the state-arm fits")
    ap.add_argument("--gate", action="store_true", help="Part 1")
    ap.add_argument("--absorb", action="store_true", help="Part 3")
    ap.add_argument("--counterfactual", action="store_true", help="Part 4")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--tags", default="state_d1y_clean,state_d1y_noisy")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS_DEFAULT))
    args = ap.parse_args(argv)

    import zinc_colloc_v5 as v5
    S.integrity_check()
    S.install(v5)
    if args.check:
        check()
        return 0

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if args.fit:
        run_fits(tags, seeds)

    gate = per_seed = contrast = cf_ps = cf_summ = struct = None
    traj = {}

    if args.gate or args.figures:
        gate = part1_gate()
        print("\n[Part 1] the capacity gate")
        print(gate["summary"].T.to_string(header=False))
        print("\n[Part 1] gate projected on the 26 driver features "
              "(fitted on trainval, scored on both):")
        print(gate["proj"][["lam", "r2_trainval", "r2_test",
                            "rmse_trainval_nats", "rmse_test_nats"]]
              .round(4).to_string(index=False))
        print("\n[Part 1] strongest driver correlations with the gate:")
        print(gate["corr"].head(6).round(3).to_string(index=False))

    if args.absorb or args.figures:
        per_seed, contrast, traj = part3_absorb(seeds=seeds)
        if per_seed.empty:
            print("\n[Part 3] no fits on disk yet (run --fit).")
        else:
            print("\n[Part 3a] does any standard diagnostic notice? "
                  "(stock relRMSE %, median over seeds)")
            print(per_seed.groupby("tag")[
                ["stock_relRMSE_mean", "stock_relRMSE_Concentrate",
                 "stock_relRMSE_Refined", "stock_relRMSE_In-Use",
                 "stock_relRMSE_Scrap"]].median().round(2).to_string())
            print("\n[Part 3a] in-sample versus held-out, and the residual "
                  "autocorrelation (the standard misspecification test):")
            print(per_seed.groupby("tag")[
                ["stock_relRMSE_mean_trainval", "stock_relRMSE_mean_test",
                 "acf1_mean", "acf1_Refined"]].median().round(3).to_string())
            print("\n[Part 3b/d] coefficient relRMSE vs truth, trainval "
                  "(median over seeds):")
            print(per_seed.groupby("tag")[
                [f"alpha_{n}_trainval" for n in ALPHA_NAMES]]
                .median().round(2).to_string())
            print("\n[Part 3c] absorption of the gate by alpha_refc:")
            print(per_seed.groupby("tag")[
                ["absorb_slope_trainval", "absorb_r2_trainval",
                 "absorb_slope_test", "absorb_r2_test",
                 "resid_rms_trainval", "resid_rms_test"]]
                .median().round(4).to_string())
            print("\n[Part 3e] smoothness of the estimated alpha_refc "
                  "(rms 2nd difference of log, per yr^2):")
            print(per_seed.groupby("tag")[
                [f"rough_{n}" for n in ALPHA_NAMES]].median().round(5).to_string())
            print("\n[Part 3] seed-paired state - base contrasts "
                  "(Hodges-Lehmann):")
            key = (["stock_relRMSE_mean"]
                   + [f"alpha_{n}_trainval" for n in ALPHA_NAMES]
                   + ["rough_alpha_refc"])
            print(contrast[contrast.quantity.isin(key)][
                ["pair", "quantity", "state_median", "base_median",
                 "diff_hl", "diff_lo", "diff_hi"]].round(3).to_string(index=False))

        struct, sstruct = part3f_structure(seeds=seeds)
        if not struct.empty:
            print("\n[Part 3f] what a modeller fitting the truth's own form to "
                  "the estimated alpha_refc would report:")
            cols = ["tag", "source", "A", "tstar", "w", "m",
                    "beta_GDP", "beta_ZnPrice", "rms_resid_nats"]
            show = pd.concat([
                struct[struct.source != "fit"][cols].drop_duplicates(
                    ["tag", "source"]),
                struct[struct.source == "fit"].groupby("tag")[cols[2:]]
                .median().reset_index().assign(source="fit (median)")[cols]])
            print(show.round(3).to_string(index=False))
            # The seed spread matters as much as the median here: on a
            # misspecified truth the fits disagree about the elasticity they
            # report, and a median alone would hide that.
            sp = struct[struct.source == "fit"].groupby("tag")[
                ["tstar", "w", "m", "beta_GDP"]].quantile([0.25, 0.75]).unstack()
            sp.columns = [f"{a}_q{int(b*100)}" for a, b in sp.columns]
            print("\n  seed spread of the reported constants (IQR):")
            print(sp[sorted(sp.columns)].round(3).to_string())
            print(f"\n  registered truth: A={np.exp(STRUCT_TRUTH['logA']):.3f} "
                  f"tstar={STRUCT_TRUTH['tstar']:.0f} w={STRUCT_TRUTH['w']:.0f} "
                  f"m={STRUCT_TRUTH['m']:+.2f} beta_GDP={STRUCT_TRUTH['beta_GDP']:+.2f} "
                  f"beta_ZnPrice={STRUCT_TRUTH['beta_ZnPrice']:+.2f}")

    if args.counterfactual or args.figures:
        cf_ps, cf_summ, chk, asym = part4_counterfactual(seeds=seeds)
        print("\n[Part 4] solver gates.  `gap_fit_vs_own_same_tol` is the "
              "composition check and must be ~0; the tight column is the "
              "rtol=1e-5 effect (COMPUTE_STATUS flag 3); "
              "`gap_truth_vs_dense` is a gate only where pins are not noisy.")
        print(chk.round(10).to_string(index=False))
        if not cf_summ.empty:
            print("\n[Part 4] counterfactual response of S_ref, truth vs "
                  "estimated (median over seeds):")
            print(cf_summ[["tag", "shock", "amount", "n_seeds",
                           "true_alpha_resp", "fit_alpha_resp",
                           "true_Sref_resp", "fit_Sref_resp",
                           "Sref_resp_ratio", "Sref_resp_relRMSE"]]
                  .round(4).to_string(index=False))
        fq = os.path.join(OUT_DIR, "wp4b_fit_quality.csv")
        if os.path.exists(fq):
            q = pd.read_csv(fq)
            print("\n[Part 4] fit quality against the twin's own observations, "
                  "pooled across stocks / flows (median over seeds).  This is "
                  "the other half of Part 3a: the flows are what Stage B's "
                  "divergence term sees, and `dump_A` does not persist them.")
            print(q.groupby("tag")[["stock_relRMSE", "flow_relRMSE"]]
                  .median().round(3).to_string())
        if asym is not None and not asym.empty:
            print("\n[Part 4] response asymmetry in the sign of the cp shock "
                  "(0 = perfectly odd; the estimator is odd by construction):")
            print(asym.groupby("tag")[["truth_asym", "fit_asym"]]
                  .median().round(5).to_string())
        stab = part4b_stability(seeds=seeds)
        print("\n[Part 4b] the stability boundary of the gated truth, and the "
              "estimated model's absence of one (S_ref peak, x baseline):")
        cols = [c for c in ["delta_sd", "truth_Sref_max_x_baseline",
                            "fit_Sref_max_x_baseline", "truth_alpha_refc_min",
                            "fit_alpha_refc_min"] if c in stab.columns]
        print(stab[cols].round(4).to_string(index=False))

    if args.figures:
        figures(gate, per_seed, contrast, cf_summ, struct, traj)
        print("\n[figures] analysis/wp4b_absorption.{png,pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
