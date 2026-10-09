#!/usr/bin/env python3
"""
run_wp8.py — WP-8: the explainability suite
===========================================

The spec calls WP-8 "the standalone contribution for the journal audience"
and sets the organising question: *what does a physics-constrained neural
model let you learn about a resource system that a regression or
autoregressive model does not?*  Every sub-package produces a number or a
figure.

  8a  input-gradient attribution      vanilla / gradient x input / integrated
                                      gradients, per driver and per order,
                                      across all 35 seeds, plus the d/dS block
  8b  temporal attribution            influence surfaces I_d(s, t) and the
                                      memory length, checked against the
                                      cohort lifetimes 10 / 20 / 44 yr
  8c  forward-mode ODE sensitivities   and the 8a-vs-8c contrast, which is what
                                      measures how much the physics modifies
                                      the network's raw response
  8d  permutation importance          circular block bootstrap, 200 reps
  8e  global sensitivity              Morris + Sobol on 17 structural
                                      coefficients, with the joint stated
  8f  regime-break recovery           Bai-Perron + CUSUM against a
                                      pre-registered event list
  8g  shape versus level              already complete (`analysis/wp8g_*`)
  8h  coefficient uncertainty product already complete (`analysis/wp8h_*`)
  8i  comparator capability table     assembled here from the packages that
                                      populate it

No refits anywhere: every instrument reads the `zinc_A_lab` weight dumps in
`analysis/wp2a/`.  Attribution is always reported across the seed ensemble —
median with IQR and a Hodges-Lehmann interval — never from a single fit, and
attribution that does not survive the ensemble is reported as noise rather
than dropped.

    python run_wp8.py --check
    python run_wp8.py --8a --8f --8i
    python run_wp8.py --all --seeds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import zinc_xai_lab as X
import zinc_circ_lab as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "analysis")
SEED_DIR = os.path.join(OUT_DIR, "wp8")

COL = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00", "#56B4E9"]
GREY = "#555555"

MU_COHORTS = (10.0, 20.0, 44.0)
ALPHA_NAMES = X.ALPHA_NAMES
COEF_NAMES = X.COEF_NAMES


def hl(v):
    """`(estimate, lo, hi)` from `zinc_circ_lab.hodges_lehmann`, which returns
    a dict and leaves the interval NaN below six observations."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (np.nan, np.nan, np.nan)
    d = C.hodges_lehmann(v)
    return (float(d["hl"]), float(d["lo"]), float(d["hi"]))


def iqr(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return (np.nan, np.nan) if v.size == 0 else (float(np.percentile(v, 25)),
                                                 float(np.percentile(v, 75)))


def seed_list(arg, weights_dir):
    if arg:
        return [int(s) for s in str(arg).split(",") if s.strip()]
    return sorted(int(f[len("A_seed"):-len(".npz")])
                  for f in os.listdir(weights_dir)
                  if f.startswith("A_seed") and f.endswith(".npz"))


# ===========================================================================
# 8a — input-gradient attribution
# ===========================================================================
def run_8a(fit, ctx, lay, seeds, out_dir=OUT_DIR, seed_dir=SEED_DIR):
    os.makedirs(seed_dir, exist_ok=True)
    cfg, t_full, X_full, cols, cut, weights_dir = ctx
    per_seed, tensors = [], {}
    t0 = time.time()
    for si, seed in enumerate(seeds):
        params = X.load_params(seed, weights_dir)
        att = X.attribution_seed(fit, params, lay)
        for key in ("grad", "grad_log", "gxi", "ig"):
            tensors.setdefault(key, []).append(att[key])
        tensors.setdefault("value", []).append(att["value"])
        tensors.setdefault("dS", []).append(att["dS"])
        tensors.setdefault("completeness", []).append(att["completeness"])
        if si == 0:
            years, x_all = att["years"], att["x"]
        per_seed.append(seed)
        import jax
        jax.clear_caches()
    A = {k: np.asarray(v) for k, v in tensors.items()}      # (S, T, C, F)
    print(f"[8a] {len(seeds)} seeds in {time.time()-t0:.1f}s", flush=True)

    np.savez_compressed(
        os.path.join(out_dir, "wp8a_attribution.npz"),
        seeds=np.asarray(seeds), years=years, x=x_all,
        coef_names=np.asarray(COEF_NAMES, dtype=object),
        driver_names=np.asarray(lay["drivers"], dtype=object),
        orders=np.asarray(lay["orders"]), driver_of=lay["driver_of"],
        order_of=lay["order_of"], learned=X.learned_mask(fit), **A)

    rows = []
    learned = X.learned_mask(fit)
    for method in ("grad", "grad_log", "gxi", "ig"):
        pd_ = X.per_driver(A[method], lay)                  # (S, T, C, n_u)
        po_ = X.per_order(A[method], lay)                   # (S, T, C, n_o, n_u)
        for ci, cname in enumerate(COEF_NAMES):
            if not learned[ci]:
                continue
            for j, dname in enumerate(lay["drivers"]):
                v = pd_[:, :, ci, j]                        # (S, T)
                sm = np.mean(v, axis=1)                     # per-seed time mean
                sa = np.mean(np.abs(v), axis=1)             # per-seed magnitude
                est, lo, hi = hl(sm)
                q1, q3 = iqr(sm)
                rows.append(dict(
                    method=method, coefficient=cname, driver=dname,
                    driver_index=j, order="sum",
                    median=float(np.median(sm)), q1=q1, q3=q3,
                    mean_abs=float(np.median(sa)),
                    hl=est, hl_lo=lo, hl_hi=hi,
                    sign_consistency=float(np.mean(np.sign(sm) == np.sign(np.median(sm)))),
                    survives=bool(np.isfinite(lo) and np.isfinite(hi)
                                  and lo * hi > 0)))
                for oi, o in enumerate(lay["orders"]):
                    vo = po_[:, :, ci, oi, j]
                    smo = np.mean(vo, axis=1)
                    sao = np.mean(np.abs(vo), axis=1)
                    est, lo, hi = hl(smo)
                    q1, q3 = iqr(smo)
                    rows.append(dict(
                        method=method, coefficient=cname, driver=dname,
                        driver_index=j, order=str(o),
                        median=float(np.median(smo)), q1=q1, q3=q3,
                        mean_abs=float(np.median(sao)),
                        hl=est, hl_lo=lo, hl_hi=hi,
                        sign_consistency=float(np.mean(np.sign(smo)
                                                       == np.sign(np.median(smo)))),
                        survives=bool(np.isfinite(lo) and np.isfinite(hi)
                                      and lo * hi > 0)))
    att_df = pd.DataFrame(rows)
    att_df.to_csv(os.path.join(out_dir, "wp8a_attribution.csv"), index=False)

    # method agreement: Spearman of the per-driver ranking, per coefficient
    agree = []
    from scipy.stats import spearmanr
    for cname in [c for c, b in zip(COEF_NAMES, learned) if b]:
        sub = att_df[(att_df.coefficient == cname) & (att_df.order == "sum")]
        piv = sub.pivot(index="driver", columns="method", values="mean_abs")
        for a, b in (("grad", "gxi"), ("grad", "ig"), ("gxi", "ig")):
            if a in piv and b in piv:
                r = spearmanr(piv[a], piv[b]).statistic
                agree.append(dict(coefficient=cname, method_a=a, method_b=b,
                                  spearman=float(r)))
    ag = pd.DataFrame(agree)
    ag.to_csv(os.path.join(out_dir, "wp8a_method_agreement.csv"), index=False)

    # ---- why the orders differ: the level block is nearly collinear ------
    # Order-1 attribution dominating order-0 is not an artefact of the
    # normalisation (both blocks are z-scored, so both gradients are per one
    # SD).  It has a mechanism: thirteen trending macro levels carry far fewer
    # independent directions than their thirteen first differences, so the
    # level gradients are individually unidentified and partly cancel while
    # the difference gradients are not.  This is the same collinearity that
    # WP-4c sees as the sloppy end of the spectrum, measured in the input
    # space instead of the parameter space.
    E = np.asarray(fit.data_all["exog_values"], float)
    n_u = lay["n_universe"]
    coll = []
    for oi, o in enumerate(lay["orders"]):
        blk = E[:, lay["order_of"] == o]
        Z = (blk - blk[:cut].mean(0)) / np.maximum(blk[:cut].std(0), 1e-12)
        Cm = np.corrcoef(Z.T)
        off = Cm[~np.eye(n_u, dtype=bool)]
        sv = np.linalg.svd(Z, compute_uv=False)
        cum = np.cumsum(sv ** 2) / max(np.sum(sv ** 2), 1e-30)
        coll.append(dict(order=int(o),
                         label="level" if o == 0 else f"difference (order {o})",
                         n_features=int(n_u),
                         mean_abs_corr=float(np.abs(off).mean()),
                         max_abs_corr=float(np.abs(off).max()),
                         condition_number=float(sv[0] / max(sv[-1], 1e-30)),
                         eff_rank_99pct=int(np.searchsorted(cum, 0.99) + 1)))
    coll = pd.DataFrame(coll)
    coll.to_csv(os.path.join(out_dir, "wp8a_feature_collinearity.csv"), index=False)

    # the d/dS block, reported prominently because it is exactly zero
    ds = pd.DataFrame([dict(
        statistic="max_abs_dcoef_dS", value=float(np.max(np.abs(A["dS"]))),
        note="use_stock_input=false -> the four stock features of _norm_inputs "
             "are identically zero (v5:505); the block is structurally dead, "
             "not measured small"),
        dict(statistic="max_ig_completeness_error",
             value=float(np.nanmax(A["completeness"])),
             note=f"|sum(IG) - (f(x)-f(baseline))| / |f(x)|, {X.IG_STEPS} steps"),
        dict(statistic="n_seeds", value=float(len(seeds)), note=""),
    ])
    ds.to_csv(os.path.join(out_dir, "wp8a_dalpha_dS.csv"), index=False)
    return att_df, ag, ds, A, years, coll


# ===========================================================================
# 8b — temporal attribution
# ===========================================================================
def run_8b(fit, ctx, lay, seeds, out_dir=OUT_DIR, drivers=None):
    """Influence surfaces, the memory length, and the cohort-signature test."""
    cfg, t_full, X_full, cols, cut, weights_dir = ctx
    drivers = drivers or list(cols)
    mach = X.machinery(fit)
    T = int(np.asarray(fit.data_all["years"]).size)
    surf = np.zeros((len(seeds), len(drivers), T, T, 4))
    coef = np.zeros((len(seeds), len(drivers), T, T, X.N_COEF))
    mem = np.zeros((len(seeds), len(drivers), T, 4))
    mem_abs = np.zeros((len(seeds), len(drivers), T, 4))
    peaklag = np.zeros((len(seeds), len(drivers), T, 4))
    S_fact = np.zeros((len(seeds), T, 4))
    floors = np.zeros((len(seeds), 4))
    t0 = time.time()
    for si, seed in enumerate(seeds):
        params = X.load_params(seed, weights_dir)
        for di, dn in enumerate(drivers):
            s = X.influence_surface(fit, params, ctx, dn, mach=mach)
            surf[si, di] = s["dS"]
            coef[si, di] = s["dC"]
            m1, m2, pl = X.memory_length(s)
            mem[si, di], mem_abs[si, di], peaklag[si, di] = m1, m2, pl
            S_fact[si] = s["S_factual"]
            floors[si] = s["floor_S"]
            years = s["years"]
        print(f"[8b] seed {seed} done  {time.time()-t0:6.1f}s cumulative",
              flush=True)
        import jax
        jax.clear_caches()
    np.savez_compressed(
        os.path.join(out_dir, "wp8b_influence.npz"),
        seeds=np.asarray(seeds), years=years,
        drivers=np.asarray(drivers, dtype=object), dS=surf, dC=coef,
        memory=mem, memory_abs=mem_abs, peak_lag=peaklag, floor=floors,
        S_factual=S_fact,
        coef_names=np.asarray(COEF_NAMES, dtype=object))

    stock_names = ["Concentrate", "Refined", "In-Use", "Scrap"]
    rows = []
    for di, dn in enumerate(drivers):
        for k, sname in enumerate(stock_names):
            v = np.nanmedian(mem[:, di, :, k], axis=1)      # per seed
            va = np.nanmedian(mem_abs[:, di, :, k], axis=1)
            vp = np.nanmedian(peaklag[:, di, :, k], axis=1)
            est, lo, hi = hl(v)
            q1, q3 = iqr(v)
            rows.append(dict(driver=dn, stock=sname,
                             memory_median_yr=float(np.median(v)),
                             q1=q1, q3=q3, hl=est, hl_lo=lo, hl_hi=hi,
                             memory_abs_median_yr=float(np.median(va)),
                             peak_lag_median_yr=float(np.median(vp)),
                             solver_floor_kt=float(np.median(floors[:, k]))))
    memdf = pd.DataFrame(rows)
    memdf.to_csv(os.path.join(out_dir, "wp8b_memory_length.csv"), index=False)

    # is the memory kernel structured by the cohort lifetimes?
    # The kernel is built from the **relative** response |dS|/S_factual, not
    # from kt.  In-Use grows 2.4x over the record, and lag `l` can only be
    # sampled from pairs with `t = s + l`, so long lags are drawn exclusively
    # from late years where the stock — and hence any absolute response — is
    # largest.  On absolute kt that growth alone makes the kernel rise with
    # lag, which would be read as memory structure and is not.
    rel = np.abs(surf) / np.maximum(S_fact[:, None, None, :, :], 1.0)
    kern_rows = []
    for k, sname in enumerate(stock_names):
        prof = np.zeros(T)
        cnt = np.zeros(T)
        for i in range(T):
            for t in range(i, T):
                prof[t - i] += rel[:, :, i, t, k].mean()
                cnt[t - i] += 1
        prof = prof / np.maximum(cnt, 1)
        prof_n = prof / max(prof.max(), 1e-30)
        peaks = [int(l) for l in range(1, T - 1)
                 if prof_n[l] > prof_n[l - 1] and prof_n[l] > prof_n[l + 1]
                 and prof_n[l] > 0.05]
        kern_rows.append(dict(stock=sname,
                              **{f"lag{l}": float(prof_n[l]) for l in range(0, T, 2)},
                              local_maxima=json.dumps(peaks),
                              nearest_cohort_hit=json.dumps(
                                  [float(min(MU_COHORTS, key=lambda m: abs(m - p)))
                                   for p in peaks])))
    kern = pd.DataFrame(kern_rows)
    kern.to_csv(os.path.join(out_dir, "wp8b_memory_kernel.csv"), index=False)

    kk = cohort_signature(surf, S_fact, seeds, weights_dir, T, out_dir=out_dir)
    return memdf, kern, surf, coef, drivers, years, kk, S_fact


def cohort_signature(surf, S_fact, seeds, weights_dir, T, out_dir=OUT_DIR):
    """Is the in-use memory kernel structured by the cohort parameterisation?

    A fixed-rate cohort model is a mixture of exponentials: a transient pulse
    of inflow at lag 0 leaves in-use at rate 1/mu_k from each cohort, so the
    stock response is `sum_k f_k exp(-lag / mu_k)` — **monotone decreasing,
    with no interior maximum, for any positive f**.  Chapter 2's Erlang
    replacement (n compartments at rate kappa) has mode (n-1)/kappa > 0 and
    would put a hump near the mean lifetime.  So interior structure in the
    measured kernel discriminates between the two, and this is where WP-8b
    meets WP-10.

    The kernel is built from the **relative** response |dS| / S_factual.  In-Use
    grows 2.4x over the record and lag `l` can only be sampled from pairs with
    `t = s + l`, so long lags are drawn exclusively from late years where any
    absolute response is largest; on kt that growth alone makes the kernel rise
    and it would be read as memory.

    Also reported: the kernel's excess over the mixture at the mean lifetime,
    against the frozen-time use count `upsilon` from WP-2d.  A second passage
    through use — material discharged as end-of-life, recovered as old scrap
    and manufactured back into products — arrives about one mean in-use
    lifetime after the first, and `upsilon - 1` is how much of it there is.
    """
    fbar = np.zeros(3)
    for seed in seeds:
        z = np.load(os.path.join(weights_dir, f"A_seed{seed}.npz"), allow_pickle=True)
        fbar += np.asarray(z["f_cohort"], float).mean(axis=0)
    fbar /= max(len(seeds), 1)
    mean_life = float(np.sum(fbar * np.asarray(MU_COHORTS)))

    lags = np.arange(T, dtype=float)
    theory = np.sum([fbar[c] * np.exp(-lags / MU_COHORTS[c]) for c in range(3)],
                    axis=0)
    theory /= max(theory.max(), 1e-30)

    rel = np.abs(surf) / np.maximum(S_fact[:, None, None, :, :], 1.0)
    prof = np.zeros(T); cnt = np.zeros(T)
    for i in range(T):
        for t in range(i, T):
            prof[t - i] += rel[:, :, i, t, 2].mean(); cnt[t - i] += 1
    prof = prof / np.maximum(cnt, 1)
    prof_n = prof / max(prof.max(), 1e-30)

    tail = slice(3, T)
    maxima = [int(l) for l in range(1, T - 1)
              if prof_n[l] > prof_n[l - 1] and prof_n[l] > prof_n[l + 1]
              and prof_n[l] > 0.05]
    # the secondary structure: the largest interior maximum beyond the
    # immediate (1-3 yr) response, and where it sits relative to mean_life
    late = [l for l in maxima if l >= 8]
    l_sec = int(max(late, key=lambda l: prof_n[l])) if late else -1
    l_ref = int(np.clip(round(mean_life), 0, T - 1))
    # WP-2d's `use_entry` count is Chapter 2's upsilon: the expected number of
    # passages through use before final disposal, for a unit starting in the
    # concentrate stock.  `upsilon - 1` is the share that comes back for a
    # second passage, which is the size the recirculation bump should have.
    ups = np.nan
    p_ups = os.path.join(out_dir, "wp2d_indicators.csv")
    if os.path.exists(p_ups):
        try:
            u = pd.read_csv(p_ups)
            sel = u[(u.indicator == "use_entry") & (u.form == "frozen")
                    & (u.start_stock == "S_conc")]
            if len(sel):
                ups = float(sel["median"].median())
        except Exception:
            pass

    kk = pd.DataFrame([dict(
        response_units="|dS| / S_factual (relative)",
        f_cohort_mean=json.dumps([float(x) for x in fbar]),
        mean_lifetime_yr=mean_life,
        corr_measured_vs_exponential_mixture=float(
            np.corrcoef(prof_n[tail], theory[tail])[0, 1]),
        interior_maxima_measured=json.dumps(maxima),
        interior_maxima_exponential_mixture=json.dumps(
            [int(l) for l in range(1, T - 1)
             if theory[l] > theory[l - 1] and theory[l] > theory[l + 1]]),
        secondary_maximum_lag_yr=l_sec,
        kernel_at_secondary=float(prof_n[l_sec]) if l_sec >= 0 else np.nan,
        mixture_at_secondary=float(theory[l_sec]) if l_sec >= 0 else np.nan,
        excess_at_secondary=float(prof_n[l_sec] - theory[l_sec]) if l_sec >= 0 else np.nan,
        excess_at_mean_lifetime=float(prof_n[l_ref] - theory[l_ref]),
        upsilon_wp2d_median=ups,
        upsilon_minus_one=(ups - 1.0) if np.isfinite(ups) else np.nan,
        min_before_secondary=(float(np.min(prof_n[3:max(l_sec, 4)]))
                              if l_sec > 4 else np.nan),
        lag_to_10pct_measured=float(np.max(np.where(prof_n > 0.10)[0])
                                    if (prof_n > 0.10).any() else np.nan),
        lag_to_10pct_exponential_mixture=float(np.max(np.where(theory > 0.10)[0])
                                               if (theory > 0.10).any() else np.nan),
        **{f"measured_lag{l}": float(prof_n[l]) for l in range(0, T, 2)},
        **{f"theory_lag{l}": float(theory[l]) for l in range(0, T, 2)})])
    kk.to_csv(os.path.join(out_dir, "wp8b_cohort_signature.csv"), index=False)
    return kk


# ===========================================================================
# 8c — forward-mode ODE sensitivities, and the 8a-vs-8c contrast
# ===========================================================================
def run_8c(fit, ctx, lay, seeds, out_dir=OUT_DIR, A8a=None):
    cfg, t_full, X_full, cols, cut, weights_dir = ctx
    dS, dF, gaps = [], [], []
    t0 = time.time()
    for seed in seeds:
        params = X.load_params(seed, weights_dir)
        fs = X.forward_sensitivity(fit, params, ctx)
        dS.append(fs["dS"]); dF.append(fs["dF"])
        gaps.append(dict(seed=int(seed), gap_S=fs["gap_S"], gap_F=fs["gap_F"],
                         gap_exog=fs["gap_exog"]))
        years, drivers = fs["years"], fs["drivers"]
        print(f"[8c] seed {seed}  {time.time()-t0:6.1f}s cumulative", flush=True)
        import jax
        jax.clear_caches()
    dS = np.asarray(dS)            # (S, T, 4, D)
    dF = np.asarray(dF)            # (S, T-1, NF, D)
    gp = pd.DataFrame(gaps)
    gp.to_csv(os.path.join(out_dir, "wp8c_validation.csv"), index=False)
    np.savez_compressed(os.path.join(out_dir, "wp8c_forward_sens.npz"),
                        seeds=np.asarray(seeds), years=years,
                        drivers=np.asarray(drivers, dtype=object),
                        dS=dS, dF=dF)

    stock_names = ["Concentrate", "Refined", "In-Use", "Scrap"]
    rows = []
    for di, dn in enumerate(drivers):
        for k, sname in enumerate(stock_names):
            # terminal response and its time profile, normalised by the stock
            rel = dS[:, :, k, di] / np.maximum(
                np.abs(np.asarray(fit.data_all["stocks_obs"])[None, :, k]), 1.0)
            per_seed_end = rel[:, -1]
            est, lo, hi = hl(per_seed_end)
            q1, q3 = iqr(per_seed_end)
            rows.append(dict(driver=dn, stock=sname,
                             terminal_rel_response=float(np.median(per_seed_end)),
                             q1=q1, q3=q3, hl=est, hl_lo=lo, hl_hi=hi,
                             peak_rel_response=float(np.median(
                                 rel[np.arange(rel.shape[0]),
                                     np.argmax(np.abs(rel), axis=1)])),
                             peak_year=float(np.median(
                                 years[np.argmax(np.abs(rel), axis=1)])),
                             survives=bool(np.isfinite(lo) and np.isfinite(hi)
                                           and lo * hi > 0)))
    sens = pd.DataFrame(rows)
    sens.to_csv(os.path.join(out_dir, "wp8c_sensitivity.csv"), index=False)

    # ---- the 8a-vs-8c contrast ------------------------------------------
    # 8a is d(coefficient)/d(driver) — the network's local map.
    # 8c is d(stock)/d(driver)      — the same driver move after mass balance
    #     and accumulation have propagated it.
    # Both are per one-SD driver move, so their per-driver *rankings* are
    # directly comparable and the rank gap is the quantity of interest.
    contrast = pd.DataFrame()
    if A8a is not None:
        from scipy.stats import spearmanr
        pd_a = X.per_driver(A8a["grad_log"], lay)           # (S, T, C, D)
        rows = []
        for k, sname in enumerate(stock_names):
            r8c = np.median(np.abs(dS[:, -1, k, :]), axis=0)
            for ci, cname in enumerate(ALPHA_NAMES):
                r8a = np.median(np.abs(pd_a[:, :, ci, :]).mean(axis=1), axis=0)
                rho = spearmanr(r8a, r8c).statistic
                top_a = [drivers[i] for i in np.argsort(-r8a)[:3]]
                top_c = [drivers[i] for i in np.argsort(-r8c)[:3]]
                rows.append(dict(stock=sname, coefficient=cname,
                                 spearman_8a_vs_8c=float(rho),
                                 top3_8a=json.dumps(top_a),
                                 top3_8c=json.dumps(top_c),
                                 jaccard_top3=len(set(top_a) & set(top_c)) / 3.0))
        contrast = pd.DataFrame(rows)
        contrast.to_csv(os.path.join(out_dir, "wp8c_vs_wp8a.csv"), index=False)
    return sens, contrast, gp, dS, drivers, years


# ===========================================================================
# 8d — permutation importance
# ===========================================================================
def run_8d(fit, ctx, lay, seeds, out_dir=OUT_DIR, n_perm=None):
    cfg, t_full, X_full, cols, cut, weights_dir = ctx
    n_perm = n_perm or X.N_PERM
    mach = X.machinery(fit)
    rows = []
    t0 = time.time()
    for seed in seeds:
        params = X.load_params(seed, weights_dir)
        res = X.permutation_seed(fit, params, ctx, n_perm=n_perm, mach=mach,
                                 rng_seed=int(seed))
        for r in res["rows"]:
            rows.append(dict(seed=int(seed), n_perm=res["n_perm"],
                             block_len=res["block_len"], **r))
        print(f"[8d] seed {seed}  {time.time()-t0:6.1f}s cumulative", flush=True)
        import jax
        jax.clear_caches()
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(os.path.join(out_dir, "wp8d_permutation_per_seed.csv"),
                    index=False)

    out = []
    for dn, sub in per_seed.groupby("driver"):
        for fam in ("stock", "flow", "alpha"):
            deg = (sub[f"perm_{fam}_median"] - sub[f"base_{fam}"]).to_numpy()
            est, lo, hi = hl(deg)
            q1, q3 = iqr(deg)
            # "outside its own null": does the factual metric sit below the
            # 5th percentile of the permuted distribution, per seed?
            outside = float(np.mean(sub[f"base_{fam}"].to_numpy()
                                    < sub[f"perm_{fam}_q05"].to_numpy()))
            out.append(dict(driver=dn, family=fam,
                            degradation_median=float(np.median(deg)),
                            q1=q1, q3=q3, hl=est, hl_lo=lo, hl_hi=hi,
                            frac_seeds_outside_null=outside,
                            survives=bool(np.isfinite(lo) and np.isfinite(hi)
                                          and lo > 0)))
    summ = pd.DataFrame(out)
    summ.to_csv(os.path.join(out_dir, "wp8d_permutation.csv"), index=False)
    return per_seed, summ


def contrast_8a_8d(out_dir=OUT_DIR):
    """8a against 8d: does a local gradient rank drivers the way a global,
    derivative-free perturbation does?

    The spec's reason for running both: gradients are local and assume
    differentiability, permutation is global and assumes nothing, so agreement
    is evidence and disagreement is a finding about local versus global
    structure.  Both are reduced to a per-driver magnitude and compared by
    Spearman rank correlation and top-3 overlap.
    """
    pa = os.path.join(out_dir, "wp8a_attribution.csv")
    pd_ = os.path.join(out_dir, "wp8d_permutation.csv")
    if not (os.path.exists(pa) and os.path.exists(pd_)):
        return pd.DataFrame()
    from scipy.stats import spearmanr
    att = pd.read_csv(pa)
    perm = pd.read_csv(pd_)
    rows = []
    for fam, coefs in (("alpha", ALPHA_NAMES), ("stock", ALPHA_NAMES),
                       ("flow", ALPHA_NAMES)):
        p = perm[perm.family == fam].set_index("driver").degradation_median
        for method in ("grad", "grad_log", "gxi", "ig"):
            a = (att[(att.method == method) & (att.order == "sum")
                     & att.coefficient.isin(coefs)]
                 .groupby("driver").mean_abs.mean())
            common = [d for d in a.index if d in p.index]
            if len(common) < 4:
                continue
            rho = spearmanr(a.loc[common], p.loc[common]).statistic
            ta = set(a.loc[common].nlargest(3).index)
            tp = set(p.loc[common].nlargest(3).index)
            rows.append(dict(permutation_family=fam, attribution_method=method,
                             n_drivers=len(common), spearman=float(rho),
                             jaccard_top3=len(ta & tp) / 3.0,
                             top3_8a=json.dumps(sorted(ta)),
                             top3_8d=json.dumps(sorted(tp))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "wp8d_vs_wp8a.csv"), index=False)
    return df


# ===========================================================================
# 8e — global sensitivity
# ===========================================================================
def run_8e(out_dir=OUT_DIR, weights_dir=X.WEIGHTS_DIR_DEFAULT,
           k_scan=(1.0, 2.0, 4.0)):
    rows_m, rows_s, spec = [], [], []
    for k in k_scan:
        b = X.struct_bounds(weights_dir, k=k)
        mo = X.morris(b)
        for i, nm in enumerate(b["free_names"]):
            for oi, on in enumerate(X.STRUCT_OUT_NAMES):
                rows_m.append(dict(k_sd=float(k), parameter=nm, output=on,
                                   mu_star=float(mo["mu_star"][i, oi]),
                                   mu=float(mo["mu"][i, oi]),
                                   sigma=float(mo["sigma"][i, oi])))
        so = X.sobol(b)
        for i, nm in enumerate(b["free_names"]):
            for oi, on in enumerate(X.STRUCT_OUT_NAMES):
                rows_s.append(dict(k_sd=float(k), parameter=nm, output=on,
                                   S1=float(so["S1"][i, oi]),
                                   ST=float(so["ST"][i, oi])))
        spec.append(dict(k_sd=float(k), n_seeds=int(b["n_seeds"]),
                         n_free=len(b["free_names"]),
                         free_params=json.dumps(b["free_names"]),
                         morris_traj=X.MORRIS_TRAJ, morris_levels=X.MORRIS_LEVELS,
                         sobol_base_n=X.SOBOL_BASE_N,
                         distribution=json.dumps(X.STRUCT_DIST)))
        print(f"[8e] k = {k} done", flush=True)
    mo_df, so_df = pd.DataFrame(rows_m), pd.DataFrame(rows_s)
    mo_df.to_csv(os.path.join(out_dir, "wp8e_morris.csv"), index=False)
    so_df.to_csv(os.path.join(out_dir, "wp8e_sobol.csv"), index=False)
    pd.DataFrame(spec).to_csv(os.path.join(out_dir, "wp8e_spec.csv"), index=False)

    # sensitivity of the ranking to the distributional choice
    from scipy.stats import spearmanr
    rows = []
    for on in X.STRUCT_OUT_NAMES:
        for df, lab, val in ((mo_df, "morris_mu_star", "mu_star"),
                             (so_df, "sobol_ST", "ST")):
            piv = df[df.output == on].pivot(index="parameter", columns="k_sd",
                                            values=val)
            ks = list(piv.columns)
            for a in range(len(ks)):
                for b2 in range(a + 1, len(ks)):
                    rows.append(dict(output=on, measure=lab, k_a=ks[a], k_b=ks[b2],
                                     spearman=float(spearmanr(piv[ks[a]],
                                                              piv[ks[b2]]).statistic)))
    st = pd.DataFrame(rows)
    st.to_csv(os.path.join(out_dir, "wp8e_spec_sensitivity.csv"), index=False)
    return mo_df, so_df, st


# ===========================================================================
# 8f — regime-break recovery
# ===========================================================================
def run_8f(seeds, out_dir=OUT_DIR, weights_dir=X.WEIGHTS_DIR_DEFAULT):
    per_seed = []
    for seed in seeds:
        d = np.load(os.path.join(weights_dir, f"A_seed{seed}.npz"),
                    allow_pickle=True)
        years = np.asarray(d["years"], float).ravel()
        series = {}
        for k, nm in enumerate(ALPHA_NAMES):
            series[nm] = d["alphas"][:, k]
        series["tau_olds"] = d["taus"][:, 2]
        series["frac_fu_new"] = d["frac_fu"][:, 0]
        series["frac_eu_new"] = d["frac_eu"][:, 0]
        for c in range(3):
            series[f"f_cohort_{int(d['mu_cohorts'][c])}yr"] = d["f_cohort"][:, c]
        for nm, y in series.items():
            r = X.breaks_for_series(years, y)
            for model, ks, bys in (("trend", r["k"], r["break_years"]),
                                   ("diff", r["k_diff"], r["break_years_diff"])):
                for by in (bys or [np.nan]):
                    per_seed.append(dict(seed=int(seed), series=nm, model=model,
                                         k=int(ks), break_year=float(by),
                                         cusum_year=r["cusum_year"],
                                         cusum_sup=r["cusum_sup"],
                                         cusum_signif=r["cusum_signif"]))
    ps = pd.DataFrame(per_seed)
    ps.to_csv(os.path.join(out_dir, "wp8f_breaks_per_seed.csv"), index=False)

    # Match table against the pre-registered events.  A break index is the
    # last observation of the outgoing regime, so a break reported at year b
    # places the transition in (b, b+1]; the acceptance window is therefore
    # [y0 - 1, y1] rather than centred on the event.
    rows = []
    n_seeds = ps.seed.nunique()
    ps_tr = ps[ps.model == "trend"]
    sub_years_span = float(np.ptp(np.asarray(years, float)))
    for name, y0, y1 in X.EVENTS:
        for nm, sub in ps_tr.groupby("series"):
            b = sub.break_year.dropna().to_numpy()
            inwin = (b >= y0 - 1) & (b <= y1)
            hit_seeds = sub.loc[sub.break_year.between(y0 - 1, y1), "seed"].nunique()
            # Chance rate: a break placed uniformly at random in the sample
            # would land in this window with probability width/span, and the
            # number of breaks per series is `k`.  Without it a six-year
            # window (the China boom) and a two-year one are not comparable.
            span = float(sub_years_span)
            width = float(y1 - (y0 - 1) + 1)
            k_med = float(np.median(sub.k.to_numpy())) if len(sub) else 0.0
            chance = 1.0 - (1.0 - min(width / span, 1.0)) ** max(k_med, 0.0)
            frac = float(hit_seeds / max(n_seeds, 1))
            rows.append(dict(event=name, documented=f"{y0}-{y1}", series=nm,
                             n_seeds=int(n_seeds), window_yr=width,
                             seeds_with_break_in_window=int(hit_seeds),
                             frac_seeds=frac, chance_rate=float(chance),
                             lift=float(frac / max(chance, 1e-9)),
                             median_break_year=(float(np.median(b[inwin]))
                                                if inwin.any() else np.nan),
                             hit=bool(frac >= 0.5 and frac > chance)))
    match = pd.DataFrame(rows)
    match.to_csv(os.path.join(out_dir, "wp8f_event_match.csv"), index=False)

    # break-date distribution, all series pooled, for the figure
    dist = ps.dropna(subset=["break_year"]).groupby(
        ["model", "series", "break_year"]).seed.nunique().reset_index(
        name="n_seeds").sort_values(["model", "series", "break_year"])
    dist.to_csv(os.path.join(out_dir, "wp8f_break_distribution.csv"), index=False)
    return ps, match, dist


# ===========================================================================
# 8i — comparator capability table
# ===========================================================================
def run_8i(out_dir=OUT_DIR):
    """One table across model classes, populated from measurements where they
    exist and flagged where they do not.

    The spec's instruction is to populate the AR column "from WP-1f and WP-5
    rather than from argument".  WP-1f exists and supplies the mass-balance
    numbers.  **WP-5 does not**: it needs WP-3's twin *and* the refits that
    score an AR-derived `alpha_hat = F_hat / S_obs` against `alpha_true`, which
    are CX3 jobs.  The two cells that would come from WP-5 are marked
    `pending WP-5` rather than filled in by assertion.
    """
    src = {}
    p = os.path.join(out_dir, "wp1f_mass_imbalance.csv")
    if os.path.exists(p):
        mi = pd.read_csv(p)
        last = mi.sort_values("year").groupby(["model", "stock"]).tail(1)
        worst = last.loc[last.observed_stock_pct.abs().idxmax()]
        src["ar_mass_worst_pct"] = float(worst.observed_stock_pct)
        src["ar_mass_worst_stock"] = str(worst["stock"])
        src["ar_mass_worst_model"] = str(worst["model"])
    p = os.path.join(out_dir, "wp8h_coefficients.csv")
    src["ude_coef_uncertainty"] = os.path.exists(p)
    p = os.path.join(out_dir, "wp2d_indicators.csv")
    src["ude_circularity"] = os.path.exists(p)
    p = os.path.join(out_dir, "wp7_summary.csv")
    src["ude_coef_counterfactual"] = os.path.exists(p)

    ar_mass = (f"no — cumulative imbalance up to "
               f"{abs(src.get('ar_mass_worst_pct', float('nan'))):.0f}% of the "
               f"reported {src.get('ar_mass_worst_stock','?')} stock (WP-1f)")
    rows = [
        dict(capability="produces time-varying transfer coefficients alpha(t)",
             UDE="yes — 4 alphas + 3 learned tau slots at every year node "
                 "(analysis/wp8h_coefficients.csv)",
             GAM="yes, but as a smooth of alpha_obs; no dynamical constraint",
             AR_ARX="only as a derived ratio F_hat/S_obs, which inherits the "
                    "denominator's noise (pending WP-5 for the variance)",
             constant_TC_MFA="no — a single time-invariant coefficient by "
                             "construction",
             evidence="WP-8h, WP-1f"),
        dict(capability="mass-consistent by construction",
             UDE="yes — the RHS is a balance; terminal imbalance +-1.3e-9 kt "
                 "(WP-1f)",
             GAM="yes when integrated through the same RHS (that is what the "
                 "baseline does)",
             AR_ARX=ar_mass,
             constant_TC_MFA="yes",
             evidence="WP-1f"),
        dict(capability="supports counterfactuals ON COEFFICIENTS",
             UDE="yes — WP-7 steps tau_olds at the coefficient and reports a "
                 "time-to-effect that emerges from the cohort structure",
             GAM="partially — coefficients are functions of drivers, so driver "
                 "counterfactuals work; a coefficient step has no dynamics to "
                 "propagate through unless integrated",
             AR_ARX="no — there is no coefficient to intervene on",
             constant_TC_MFA="yes but statically: the effect is instantaneous "
                             "by construction, so no time-to-effect exists",
             evidence="WP-7"),
        dict(capability="yields uncertainty on the coefficients",
             UDE="yes — seed ensemble + Laplace, pooled by Rubin's rules "
                 "(WP-4c, WP-8h); e.g. alpha_14 to +-15% on the profile",
             GAM="no — deterministic, one fit, no coefficient posterior",
             AR_ARX="parameter standard errors exist, but on flow coefficients, "
                    "not on transfer coefficients (pending WP-5)",
             constant_TC_MFA="only through error propagation on the input data",
             evidence="WP-4c, WP-8h"),
        dict(capability="supports eigen / circularity diagnostics",
             UDE="yes — A(t) assembled per year and seed; frozen-time and "
                 "non-autonomous indicators and their discrepancy (WP-2a-2e)",
             GAM="yes, via the same A(t) assembly, single realisation",
             AR_ARX="no — no A matrix exists",
             constant_TC_MFA="yes, but one time-invariant A, so no "
                             "non-autonomous indicator and no discrepancy",
             evidence="WP-2a, WP-2b, WP-2d, WP-2e"),
        dict(capability="extrapolates beyond the sample",
             UDE="yes, with a measured penalty — coefficient uncertainty grows "
                 "2-7x from train to test window (WP-4c)",
             GAM="yes, with spline extrapolation behaviour outside the knots",
             AR_ARX="yes for flows, but the implied stocks diverge (WP-1f)",
             constant_TC_MFA="yes trivially; the coefficients cannot change",
             evidence="WP-4c, WP-1f"),
        dict(capability="state-dependent coefficients",
             UDE="not in anchor_v4 — `use_stock_input: false` makes "
                 "d alpha / d S exactly 0 (WP-8a); the architecture supports "
                 "it and the published configuration does not use it",
             GAM="no", AR_ARX="no", constant_TC_MFA="no",
             evidence="WP-8a"),
        dict(capability="attributes a coefficient to a driver's RATE of change",
             UDE="yes — order-1 features are first differences and carry "
                 "separable attribution (WP-8a)",
             GAM="yes if the difference is offered as a basis term",
             AR_ARX="implicitly, through its own lag structure",
             constant_TC_MFA="no",
             evidence="WP-8a"),
    ]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "wp8i_capability_table.csv"), index=False)
    with open(os.path.join(out_dir, "wp8i_capability_table.md"), "w") as fh:
        fh.write("# WP-8i — comparator capability table\n\n")
        fh.write("Populated from the packages named in the `evidence` column, "
                 "not from argument.\n\n")
        fh.write("| capability | UDE | GAM | per-flow AR/ARX | constant-TC MFA "
                 "| evidence |\n|---|---|---|---|---|---|\n")
        for r in rows:
            fh.write(f"| {r['capability']} | {r['UDE']} | {r['GAM']} | "
                     f"{r['AR_ARX']} | {r['constant_TC_MFA']} | "
                     f"{r['evidence']} |\n")
        fh.write("\n**Not measured here.** The two AR/ARX cells marked "
                 "`pending WP-5` need the interpretability head-to-head, which "
                 "scores an AR-derived `alpha_hat = F_hat / S_obs` against a "
                 "known `alpha_true`.  WP-3's twin now exists, so WP-5 is "
                 "unblocked, but it needs refits and is a CX3 job.\n")
    return df


# ===========================================================================
# figures
# ===========================================================================
def fig_8a(att_df, ag, A, lay, years, out_dir=OUT_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    drivers = lay["drivers"]
    fig, ax = plt.subplots(1, 4, figsize=(17, 5.0), sharey=True)
    for ci, cname in enumerate(ALPHA_NAMES):
        sub = att_df[(att_df.method == "grad_log") & (att_df.coefficient == cname)
                     & (att_df.order == "sum")].set_index("driver").loc[drivers]
        y = np.arange(len(drivers))
        colr = [COL[0] if s else GREY for s in sub.survives]
        ax[ci].barh(y, sub["median"], color=colr,
                    xerr=np.abs(np.vstack([sub["median"] - sub.q1,
                                           sub.q3 - sub["median"]])),
                    error_kw=dict(lw=0.8, ecolor="k"))
        ax[ci].axvline(0, color="k", lw=0.8)
        ax[ci].set_yticks(y)
        if ci == 0:
            ax[ci].set_yticklabels([d[:34] for d in drivers], fontsize=8)
        ax[ci].set_title(cname)
        ax[ci].set_xlabel(r"$\partial \log\alpha/\partial x$  per 1 SD")
        ax[ci].grid(alpha=0.25, axis="x")
    fig.suptitle("WP-8a  input-gradient attribution, summed over feature orders — "
                 "median and IQR over the seed ensemble\n"
                 "grey = the Hodges–Lehmann interval covers zero, i.e. the "
                 "attribution does not survive the ensemble", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"wp8a_attribution.{ext}"), dpi=200)
    plt.close(fig)

    # per-order panel: which drivers act through the level, which through Δ
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.6), sharey=True)
    for ci, cname in enumerate(ALPHA_NAMES):
        for oi, o in enumerate(lay["orders"]):
            sub = att_df[(att_df.method == "grad_log")
                         & (att_df.coefficient == cname)
                         & (att_df.order == str(o))].set_index("driver").loc[drivers]
            y = np.arange(len(drivers)) + (oi - 0.5) * 0.36
            ax[ci].barh(y, sub["median"], 0.34, color=COL[oi],
                        label=f"order {o}" + (" (level)" if o == 0 else " (Δ)"))
        ax[ci].axvline(0, color="k", lw=0.8)
        ax[ci].set_yticks(np.arange(len(drivers)))
        if ci == 0:
            ax[ci].set_yticklabels([d[:34] for d in drivers], fontsize=8)
            ax[ci].legend(frameon=False, fontsize=8)
        ax[ci].set_title(cname); ax[ci].grid(alpha=0.25, axis="x")
        ax[ci].set_xlabel(r"$\partial \log\alpha/\partial x$")
    fig.suptitle("WP-8a  attribution split by feature order — a driver that acts "
                 "only at order 1 is one whose rate of change moves the coefficient",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"wp8a_by_order.{ext}"), dpi=200)
    plt.close(fig)


def fig_8b(surf, drivers, years, memdf, kern, S_fact, out_dir=OUT_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stock_names = ["Concentrate", "Refined", "In-Use", "Scrap"]
    med = np.median(surf, axis=0)                       # (D, S, T, 4)
    pick = [0, 2, 6]                                    # price, GDP, scrap PPI
    pick = [p for p in pick if p < len(drivers)]
    fig, ax = plt.subplots(len(pick), 4, figsize=(16, 3.4 * len(pick)),
                           squeeze=False)
    for r, di in enumerate(pick):
        for k in range(4):
            Z = med[di, :, :, k]
            v = np.max(np.abs(Z)) or 1.0
            im = ax[r][k].pcolormesh(years, years, Z, cmap="RdBu_r",
                                     vmin=-v, vmax=v, shading="nearest")
            ax[r][k].plot(years, years, color="k", lw=0.6, ls=":")
            ax[r][k].set_title(f"{drivers[di][:26]} → {stock_names[k]}", fontsize=9)
            ax[r][k].set_xlabel("response year $t$")
            if k == 0:
                ax[r][k].set_ylabel("shock year $s$")
            fig.colorbar(im, ax=ax[r][k], fraction=0.046, label="kt")
    fig.suptitle("WP-8b  influence surfaces $I_d(s,t)$ — median over the seed "
                 "ensemble.  Everything below the dotted diagonal is memory.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"wp8b_influence.{ext}"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    lags = np.arange(surf.shape[3])
    rel = np.abs(surf) / np.maximum(S_fact[:, None, None, :, :], 1.0)
    for k, sname in enumerate(stock_names):
        prof = np.zeros(surf.shape[3]); cnt = np.zeros(surf.shape[3])
        for i in range(surf.shape[2]):
            for t in range(i, surf.shape[3]):
                prof[t - i] += rel[:, :, i, t, k].mean(); cnt[t - i] += 1
        prof = prof / np.maximum(cnt, 1)
        ax[0].plot(lags, prof / max(prof.max(), 1e-30), lw=1.6, color=COL[k],
                   label=sname)
    fbar = np.array([0.1466, 0.3588, 0.4946])
    th = np.sum([fbar[c] * np.exp(-lags / MU_COHORTS[c]) for c in range(3)], axis=0)
    ax[0].plot(lags, th / th.max(), "k--", lw=1.3,
               label=r"$\sum_k f_k e^{-\ell/\mu_k}$ (fixed-rate cohorts)")
    for m in MU_COHORTS:
        ax[0].axvline(m, color=GREY, ls="--", lw=1.0)
        ax[0].text(m, 1.02, f"{m:.0f} yr", ha="center", fontsize=8, color=GREY)
    ax[0].set_xlabel("lag $t - s$ (yr)"); ax[0].set_ylabel("normalised |response|")
    ax[0].set_title("memory kernel against the cohort lifetimes")
    ax[0].legend(frameon=False, fontsize=8); ax[0].grid(alpha=0.25)

    m = memdf.pivot(index="driver", columns="stock", values="memory_median_yr")
    im = ax[1].imshow(m.to_numpy(), aspect="auto", cmap="viridis")
    ax[1].set_xticks(range(m.shape[1])); ax[1].set_xticklabels(m.columns, rotation=20)
    ax[1].set_yticks(range(m.shape[0]))
    ax[1].set_yticklabels([d[:30] for d in m.index], fontsize=7)
    fig.colorbar(im, ax=ax[1], label="memory length (yr)")
    ax[1].set_title("how far back a driver shock stays detectable")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"wp8b_memory.{ext}"), dpi=200)
    plt.close(fig)


def fig_8c(sens, contrast, dS, drivers, years, A, lay, out_dir=OUT_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stock_names = ["Concentrate", "Refined", "In-Use", "Scrap"]
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.0))
    med = np.median(dS, axis=0)
    q1 = np.percentile(dS, 25, axis=0); q3 = np.percentile(dS, 75, axis=0)
    top = np.argsort(-np.abs(med[-1]).max(axis=0))[:4] if med.ndim == 3 else []
    for k in range(4):
        order = np.argsort(-np.abs(med[-1, k, :]))[:4]
        for rank, di in enumerate(order):
            ax[k].plot(years, med[:, k, di], color=COL[rank], lw=1.5,
                       label=drivers[di][:24])
            ax[k].fill_between(years, q1[:, k, di], q3[:, k, di],
                               color=COL[rank], alpha=0.18, lw=0)
        ax[k].axhline(0, color="k", lw=0.8)
        ax[k].set_title(stock_names[k]); ax[k].set_xlabel("year")
        ax[k].set_ylabel("$dS/d\\theta$  (kt per 1 SD)")
        ax[k].legend(frameon=False, fontsize=7); ax[k].grid(alpha=0.25)
    fig.suptitle("WP-8c  forward-mode ODE sensitivities — exact, signed, "
                 "time-resolved; median and IQR over the seed ensemble", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"wp8c_forward_sens.{ext}"), dpi=200)
    plt.close(fig)

    if contrast is not None and not contrast.empty and A is not None:
        pd_a = X.per_driver(A["grad_log"], lay)
        fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
        r8a = np.median(np.abs(pd_a[:, :, 1, :]).mean(axis=1), axis=0)   # alpha_refc
        r8c = np.median(np.abs(dS[:, -1, 1, :]), axis=0)                 # Refined
        r8a_n = r8a / max(r8a.max(), 1e-30)
        r8c_n = r8c / max(r8c.max(), 1e-30)
        y = np.arange(len(drivers))
        ax[0].barh(y - 0.2, r8a_n, 0.38, color=COL[0],
                   label="8a  network's local map  $|\\partial\\log\\alpha_{refc}/\\partial x|$")
        ax[0].barh(y + 0.2, r8c_n, 0.38, color=COL[2],
                   label="8c  propagated response  $|\\partial S_{Refined}/\\partial\\theta|$")
        ax[0].set_yticks(y)
        ax[0].set_yticklabels([d[:30] for d in drivers], fontsize=7)
        ax[0].set_xlabel("normalised to the largest in each method")
        ax[0].legend(frameon=False, fontsize=8); ax[0].grid(alpha=0.25, axis="x")
        ax[0].set_title("what the physics does to the network's ranking")

        c = contrast.pivot(index="coefficient", columns="stock",
                           values="spearman_8a_vs_8c")
        im = ax[1].imshow(c.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1,
                          aspect="auto")
        ax[1].set_xticks(range(c.shape[1])); ax[1].set_xticklabels(c.columns,
                                                                  rotation=20)
        ax[1].set_yticks(range(c.shape[0])); ax[1].set_yticklabels(c.index)
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                ax[1].text(j, i, f"{c.to_numpy()[i, j]:.2f}", ha="center",
                           va="center", fontsize=8)
        fig.colorbar(im, ax=ax[1], label="Spearman, 8a vs 8c driver ranking")
        ax[1].set_title("rank agreement between the two attributions")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out_dir, f"wp8c_vs_wp8a.{ext}"), dpi=200)
        plt.close(fig)


def fig_8d(summ, per_seed, out_dir=OUT_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for fi, fam in enumerate(("stock", "flow", "alpha")):
        s = summ[summ.family == fam].sort_values("degradation_median")
        y = np.arange(len(s))
        colr = [COL[1] if v else GREY for v in s.survives]
        ax[fi].barh(y, s.degradation_median, color=colr,
                    xerr=np.abs(np.vstack([s.degradation_median - s.q1,
                                           s.q3 - s.degradation_median])),
                    error_kw=dict(lw=0.8, ecolor="k"))
        ax[fi].set_yticks(y)
        if fi == 0:
            ax[fi].set_yticklabels([d[:32] for d in s.driver], fontsize=8)
        ax[fi].axvline(0, color="k", lw=0.8)
        ax[fi].set_xlabel(f"Δ {fam} relRMSE (pp)")
        ax[fi].set_title(fam); ax[fi].grid(alpha=0.25, axis="x")
    fig.suptitle(f"WP-8d  block-permutation importance "
                 f"({int(per_seed.n_perm.iloc[0])} circular block reps, "
                 f"block {int(per_seed.block_len.iloc[0])} yr, no refit) — "
                 f"median and IQR over seeds; grey = HL interval covers zero",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"wp8d_permutation.{ext}"), dpi=200)
    plt.close(fig)


def fig_8e(mo_df, so_df, out_dir=OUT_DIR, k_ref=2.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.0))
    piv = so_df[so_df.k_sd == k_ref].pivot(index="parameter", columns="output",
                                           values="ST")
    im = ax[0].imshow(piv.to_numpy(), aspect="auto", cmap="magma", vmin=0)
    ax[0].set_xticks(range(piv.shape[1]))
    ax[0].set_xticklabels(piv.columns, rotation=35, ha="right", fontsize=8)
    ax[0].set_yticks(range(piv.shape[0]))
    ax[0].set_yticklabels(piv.index, fontsize=8)
    fig.colorbar(im, ax=ax[0], label="Sobol total index $S_T$")
    ax[0].set_title(f"WP-8e  total-order indices, $\\pm{k_ref:g}$ SD box")

    m = mo_df[(mo_df.k_sd == k_ref) & (mo_df.output == "tau_from_conc")]
    ax[1].scatter(m.mu_star, m.sigma, s=34, color=COL[0])
    for _, r in m.iterrows():
        if r.mu_star > m.mu_star.quantile(0.6):
            ax[1].annotate(r.parameter, (r.mu_star, r.sigma), fontsize=7,
                           xytext=(3, 3), textcoords="offset points")
    ax[1].set_xlabel(r"$\mu^*$ (mean |elementary effect|)")
    ax[1].set_ylabel(r"$\sigma$ (interaction / nonlinearity)")
    ax[1].set_title("Morris screening, $\\tau$ from the concentrate stock")
    ax[1].grid(alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"wp8e_sobol.{ext}"), dpi=200)
    plt.close(fig)


def fig_8f(ps, match, out_dir=OUT_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ps = ps[ps.model == "trend"]
    series = sorted(ps.series.unique())
    fig, ax = plt.subplots(figsize=(12, 0.42 * len(series) + 2.6))
    for i, nm in enumerate(series):
        b = ps[ps.series == nm].break_year.dropna().to_numpy()
        if b.size:
            ax.scatter(b + np.random.default_rng(0).normal(0, 0.08, b.size),
                       np.full(b.size, i), s=14, alpha=0.55, color=COL[0])
    for name, y0, y1 in X.EVENTS:
        ax.axvspan(y0 - 0.5, y1 + 0.5, color=COL[2], alpha=0.13, lw=0)
        ax.text((y0 + y1) / 2, len(series) - 0.3, name.split()[0], ha="center",
                fontsize=7, color=COL[2], rotation=0)
    ax.set_yticks(range(len(series))); ax.set_yticklabels(series, fontsize=8)
    ax.set_xlabel("break year")
    ax.set_title("WP-8f  break dates recovered from the learned coefficient "
                 "trajectories, one point per seed\n"
                 "shaded = pre-registered events, fixed before any break was "
                 "computed", fontsize=10)
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"wp8f_breaks.{ext}"), dpi=200)
    plt.close(fig)


# ===========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="WP-8 explainability suite")
    ap.add_argument("--check", action="store_true")
    for p in ("8a", "8b", "8c", "8d", "8e", "8f", "8i"):
        ap.add_argument(f"--{p}", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--perm-seeds", default="", help="subset for 8d (expensive)")
    ap.add_argument("--n-perm", type=int, default=0)
    ap.add_argument("--weights", default=X.WEIGHTS_DIR_DEFAULT)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    import zinc_colloc_v5 as v5
    X.integrity_check()
    X.install(v5)
    if args.check:
        X.check(weights_dir=args.weights)
        return 0

    want = {p: (getattr(args, p) or args.all)
            for p in ("8a", "8b", "8c", "8d", "8e", "8f", "8i")}
    if not any(want.values()):
        X.check(weights_dir=args.weights)
        return 0

    seeds = seed_list(args.seeds, args.weights)
    cfg = X.load_anchor_config()
    fit, ctx = X.build_context(cfg, args.weights)
    lay = X.layout_index(fit, cfg)
    os.makedirs(SEED_DIR, exist_ok=True)
    print(f"[wp8] {len(seeds)} seeds, {lay['n_universe']} drivers, "
          f"orders {lay['orders']}, input_dim {lay['input_dim']}", flush=True)

    A = att_df = ag = None
    if want["8a"]:
        att_df, ag, dsdf, A, years8a, coll = run_8a(fit, ctx, lay, seeds)
        print("\n[8a] d(coef)/dS and IG completeness:")
        print(dsdf.to_string(index=False))
        print("\n[8a] strongest surviving attributions (grad_log, order-summed):")
        s = att_df[(att_df.method == "grad_log") & (att_df.order == "sum")
                   & att_df.survives & att_df.coefficient.isin(ALPHA_NAMES)]
        print(s.reindex(s["median"].abs().sort_values(ascending=False).index)
              .head(12)[["coefficient", "driver", "median", "q1", "q3",
                         "sign_consistency"]].round(4).to_string(index=False))
        print("\n[8a] order-1 vs order-0 attribution, |median| ratio (top 8):")
        _o = att_df[(att_df.method == "grad_log") & att_df.order.isin(["0", "1"])
                    & att_df.coefficient.isin(ALPHA_NAMES)]
        _p = _o.pivot_table(index=["coefficient", "driver"], columns="order",
                            values="median")
        _p["ratio_1_over_0"] = _p["1"].abs() / _p["0"].abs().replace(0, np.nan)
        print(_p.sort_values("ratio_1_over_0", ascending=False).head(8)
              .round(3).to_string())
        print("\n[8a] input-block collinearity (the mechanism):")
        print(coll.round(3).to_string(index=False))
        print("\n[8a] method agreement (Spearman of |attribution| ranking):")
        print(ag.groupby(["method_a", "method_b"]).spearman
              .describe()[["mean", "min", "max"]].round(3).to_string())
        if not args.no_figures:
            fig_8a(att_df, ag, A, lay, years8a)
    elif os.path.exists(os.path.join(OUT_DIR, "wp8a_attribution.npz")):
        z = np.load(os.path.join(OUT_DIR, "wp8a_attribution.npz"), allow_pickle=True)
        A = {k: z[k] for k in ("grad", "grad_log", "gxi", "ig", "value", "dS")}

    if want["8b"]:
        (memdf, kern, surf, coefsurf, drivers8b, years8b,
         kk, S_fact8b) = run_8b(fit, ctx, lay, seeds)
        print("\n[8b] memory length (yr), median over seeds:")
        print(memdf.pivot(index="driver", columns="stock",
                          values="memory_median_yr").round(1).to_string())
        print("\n[8b] memory-kernel local maxima against the cohort lifetimes:")
        print(kern[["stock", "local_maxima", "nearest_cohort_hit"]].to_string(index=False))
        print("\n[8b] the cohort signature: measured In-Use kernel against the "
              "exponential-mixture prediction:")
        print(kk[["f_cohort_mean", "mean_lifetime_yr",
                  "corr_measured_vs_exponential_mixture",
                  "interior_maxima_measured",
                  "interior_maxima_exponential_mixture",
                  "lag_to_10pct_measured",
                  "lag_to_10pct_exponential_mixture"]].round(3).to_string(index=False))
        if not args.no_figures:
            fig_8b(surf, drivers8b, years8b, memdf, kern, S_fact8b)

    if want["8c"]:
        sens, contrast, gp, dS8c, drv8c, yrs8c = run_8c(fit, ctx, lay, seeds, A8a=A)
        print("\n[8c] ForwardMode vs the fit's own integrator (max rel gap):")
        print(gp.describe().loc[["max"]].round(12).to_string())
        print("\n[8c] largest terminal sensitivities:")
        print(sens.reindex(sens.terminal_rel_response.abs()
                           .sort_values(ascending=False).index)
              .head(10)[["driver", "stock", "terminal_rel_response",
                         "peak_year", "survives"]].round(4).to_string(index=False))
        if not contrast.empty:
            print("\n[8c] 8a-vs-8c ranking agreement:")
            print(contrast[["stock", "coefficient", "spearman_8a_vs_8c",
                            "jaccard_top3"]].round(3).to_string(index=False))
        if not args.no_figures:
            fig_8c(sens, contrast, dS8c, drv8c, yrs8c, A, lay)

    if want["8d"]:
        pseeds = seed_list(args.perm_seeds, args.weights) if args.perm_seeds else seeds
        ps8d, summ8d = run_8d(fit, ctx, lay, pseeds,
                              n_perm=(args.n_perm or None))
        print("\n[8d] permutation degradation (pp), median over seeds:")
        print(summ8d.pivot(index="driver", columns="family",
                           values="degradation_median").round(3).to_string())
        cx = contrast_8a_8d()
        if not cx.empty:
            print("\n[8d] against 8a — local gradient vs global permutation:")
            print(cx[["permutation_family", "attribution_method", "spearman",
                      "jaccard_top3"]].round(3).to_string(index=False))
        if not args.no_figures:
            fig_8d(summ8d, ps8d)

    if want["8e"]:
        mo, so, st = run_8e(weights_dir=args.weights)
        print("\n[8e] Sobol total indices at k = 2 SD (top by tau_from_conc):")
        s = so[(so.k_sd == 2.0) & (so.output == "tau_from_conc")]
        print(s.sort_values("ST", ascending=False).head(8)[
            ["parameter", "S1", "ST"]].round(4).to_string(index=False))
        print("\n[8e] sensitivity of the ranking to the distributional choice:")
        print(st.groupby("measure").spearman.describe()[
            ["mean", "min", "max"]].round(3).to_string())
        if not args.no_figures:
            fig_8e(mo, so)

    if want["8f"]:
        ps8f, match, dist = run_8f(seeds, weights_dir=args.weights)
        print("\n[8f] event match table (alpha channels):")
        m = match[match.series.isin(ALPHA_NAMES)]
        print(m[["event", "documented", "window_yr", "series",
                 "seeds_with_break_in_window", "n_seeds", "frac_seeds",
                 "chance_rate", "lift", "median_break_year",
                 "hit"]].round(3).to_string(index=False))
        print("\n[8f] hit rate by event, all series pooled:")
        print(match.groupby("event").agg(
            series_hit=("hit", "sum"), n_series=("hit", "size"),
            best_frac=("frac_seeds", "max")).to_string())
        if not args.no_figures:
            fig_8f(ps8f, match)

    if want["8i"]:
        cap = run_8i()
        print("\n[8i] capability table written "
              f"({len(cap)} rows) -> analysis/wp8i_capability_table.{{csv,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
