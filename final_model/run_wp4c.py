#!/usr/bin/env python3
"""
run_wp4c.py — WP-4c: Fisher information, sloppiness and practical
identifiability of the fitted zinc cycle
=================================================================

The spec's target statement: *per alpha channel, whether it is practically
identifiable at N = 40.*  If `alpha_win` and `alpha_dr` are not, that closes
the loop with WP-1a (they carry the error) and with the Chapter 2 circularity
argument (they are alpha_14 and alpha_13, the two old-scrap re-entry routes).

  Part 1  the sloppiness spectrum.  Gauss-Newton `J^T J` of the Stage B
          objective at the published optimum, from `jax.jacrev` on an
          explicit residual vector that reproduces `stageB_window_loss` to
          1e-13.  Eigenvalues on a log scale (Gutenkunst et al. 2007).
  Part 2  marginal uncertainty.  Laplace covariance `kappa^2 (J^T J + lam I)^-1`
          propagated to every coefficient at every year, reported as a CURVE
          over the regularisation scale rather than as one number — with a
          sloppy spectrum there is no scale-free marginal, and saying so is
          the honest version of the result.  What IS prior-free is reported
          separately: `frac_null`, the share of each coefficient's sensitivity
          orthogonal to everything the data constrain.
  Part 3  stiff combinations.  What the best- and worst-determined directions
          do to the coefficients — the "stiff combinations even where
          individual coefficients are loose" the spec asks for.
  Part 4  profile likelihood on the four alpha levels, which avoids the
          quadratic approximation Part 2 rests on.
  Part 5  Laplace against the across-seed spread.  Two different uncertainty
          sources measured on the same object; WP-8h combines them and WP-2f
          consumes the result as its `sigma`.

**What is NOT here.**  The spec asks for WP-4c "on both synthetic and real
fits".  Only the real fit is covered: the synthetic arm needs WP-3's twin
generator, which does not exist yet (no `zinc_synth_lab.py`, no
`analysis/synth/`).  That is a missing dependency, not a compute limit, and
it is flagged rather than approximated.

No refits.  Weights come from the `zinc_A_lab` dumps.

    python run_wp4c.py --check
    python run_wp4c.py                       # aggregate existing per-seed dumps
    python run_wp4c.py --profile --profile-seeds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import zinc_fisher_lab as F
import zinc_circ_lab as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "analysis")
SEED_DIR = os.path.join(OUT_DIR, "wp4c")

COL = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00", "#56B4E9"]
GREY = "#555555"

# Reference regularisation for the headline tables.  Relative to the largest
# data eigenvalue, so it is scale-free; the value is `anchor_v4`'s AdamW
# weight decay.  Every table that uses it says so, and the full scan is
# reported alongside because the LEVEL of the marginal depends on this choice
# even though the RANKING does not (verified in `ranking_stability`).
LAM_REF = F.LAMBDA_REF
CUT_REF = 1e-6

# Practical-identifiability bands on the relative marginal SD, stated up front
# so the verdict column is a rule and not a judgement made after seeing the
# numbers.
BAND_TIGHT = 0.10
BAND_LOOSE = 0.50
# A coefficient whose sensitivity is this far outside the data-informed
# subspace is not determined by the data at all, whatever the regulariser.
NULL_ALARM = 0.10

# The grid is concentrated where the chi^2_1 curve crosses 3.84.  A probe at
# +-0.2 nats gave 2*Delta log-posterior = 137, so the crossing sits near
# 0.03 nats and a grid spanning +-1 nat would put every point far outside the
# interval and resolve nothing.
PROFILE_OFFSETS = (-0.35, -0.2, -0.1, -0.06, -0.03, -0.01,
                   0.01, 0.03, 0.06, 0.1, 0.2, 0.35)
CHI2_95 = 3.841458820694124            # chi^2_1 at 95%


def hl(v):
    return C.hodges_lehmann(np.asarray(v, float))


# ===========================================================================
# load
# ===========================================================================
def load_seeds(seed_dir=SEED_DIR):
    paths = sorted(
        (p for p in os.listdir(seed_dir) if p.startswith("wp4c_seed")),
        key=lambda p: int(p[len("wp4c_seed"):-4]))
    if not paths:
        raise SystemExit(f"no wp4c_seed*.npz in {seed_dir} — run "
                         f"`python zinc_fisher_lab.py --all-seeds` first")
    ds = [np.load(os.path.join(seed_dir, p), allow_pickle=True) for p in paths]
    seeds = np.array([int(d["seed"]) for d in ds])
    return ds, seeds


def stack(ds, key):
    return np.stack([np.asarray(d[key]) for d in ds])


# ===========================================================================
# Part 1 — the sloppiness spectrum
# ===========================================================================
def part1(ds, seeds):
    eig = stack(ds, "eig")                                # (S, n_res)
    rank = stack(ds, "rank")
    rows = []
    for i, s in enumerate(seeds):
        # Only the directions above the rank tolerance: the tail of the
        # spectrum is exact numerical zero (2 250 - 300 null directions plus
        # the four the residual blocks share), and including it would report a
        # dynamic range that is a property of float64, not of the model.
        e = eig[i][:int(rank[i])]
        rows.append(dict(
            seed=int(s), n_res=int(ds[i]["n_res"]), n_par=int(ds[i]["n_par"]),
            n_dead=int(ds[i]["n_dead"]), rank=int(rank[i]),
            eig_max=float(e[0]), eig_min_retained=float(e[-1]),
            dynamic_range=float(e[0] / e[-1]),
            log10_decades=float(np.log10(e[0] / e[-1])),
            kappa2=float(ds[i]["kappa2"]), kappa=float(np.sqrt(ds[i]["kappa2"])),
            rss=float(ds[i]["rss"]), p_eff_ref=float(ds[i]["p_eff_ref"]),
            dof=float(ds[i]["dof"]),
            # how many directions carry the first 90 / 99% of the trace
            n_dir_90=int(np.searchsorted(np.cumsum(e) / e.sum(), 0.90) + 1),
            n_dir_99=int(np.searchsorted(np.cumsum(e) / e.sum(), 0.99) + 1),
        ))
    summ = pd.DataFrame(rows)

    long = []
    for i, s in enumerate(seeds):
        e = eig[i]
        long.append(pd.DataFrame(dict(
            seed=int(s), index=np.arange(e.size), eig=e,
            eig_rel=e / e[0], sv=np.sqrt(np.maximum(e, 0.0)))))
    return summ, pd.concat(long, ignore_index=True)


# ===========================================================================
# Part 2 — marginal uncertainty
# ===========================================================================
def part2(ds, seeds):
    years = np.asarray(ds[0]["years"], float).ravel()
    names = [str(x) for x in ds[0]["coef_names"]]
    lam_rel = np.asarray(ds[0]["lam_rel"], float)
    cut_rel = np.asarray(ds[0]["cut_rel"], float)
    coef = stack(ds, "coef")                              # (S, T, C)
    sd_lam = stack(ds, "sd_by_lambda")                    # (S, L, T, C)
    sd_cut = stack(ds, "sd_by_cut")                       # (S, K, T, C)
    fn_cut = stack(ds, "frac_null_by_cut")                # (S, K, T, C)
    est = stack(ds, "estimated")                          # (S, T, C)

    il = int(np.argmin(np.abs(lam_rel - LAM_REF)))
    ik = int(np.argmin(np.abs(cut_rel - CUT_REF)))
    sd_ref_by_seed = sd_lam[:, il]                        # (S, T, C)

    rows = []
    for i, s in enumerate(seeds):
        for t, y in enumerate(years):
            for j, n in enumerate(names):
                r = dict(seed=int(s), year=float(y), coef=n,
                         value=float(coef[i, t, j]),
                         estimated=bool(est[i, t, j]),
                         frac_null_ref=float(fn_cut[i, ik, t, j]))
                for a, lv in enumerate(lam_rel):
                    r[f"sd_lam{lv:g}"] = float(sd_lam[i, a, t, j])
                for a, cv in enumerate(cut_rel):
                    r[f"sd_cut{cv:g}"] = float(sd_cut[i, a, t, j])
                    r[f"fracnull_cut{cv:g}"] = float(fn_cut[i, a, t, j])
                r["sd_ref"] = float(sd_lam[i, il, t, j])
                r["rel_sd_ref"] = (float(sd_lam[i, il, t, j])
                                   / max(abs(float(coef[i, t, j])), 1e-300))
                rows.append(r)
    marg = pd.DataFrame(rows)

    # headline table, per coefficient: median +- IQR across seeds and years
    head = []
    for j, n in enumerate(names):
        sub = marg[(marg.coef == n)]
        estd = bool(est[:, :, j].any())
        rel = sub.rel_sd_ref.to_numpy()
        fn = sub.frac_null_ref.to_numpy()
        med = float(np.nanmedian(rel)) if estd else np.nan
        fnm = float(np.nanmedian(fn)) if estd else np.nan
        if not estd:
            verdict = "pinned to data — not estimated"
        elif not np.isfinite(med):
            verdict = "undefined"
        elif fnm > NULL_ALARM:
            verdict = "not determined by the data (prior-dependent)"
        elif med < BAND_TIGHT:
            verdict = "identifiable"
        elif med < BAND_LOOSE:
            verdict = "weakly identifiable"
        else:
            verdict = "practically unidentifiable"
        # HL across SEEDS of the per-seed median over years: the two axes are
        # not exchangeable and pooling them would treat 40 correlated years as
        # 40 independent observations.
        per_seed = np.nanmedian(
            sd_ref_by_seed[:, :, j] / np.maximum(np.abs(coef[:, :, j]), 1e-300),
            axis=1) if estd else np.full(len(seeds), np.nan)
        h = hl(per_seed) if estd else dict(hl=np.nan, lo=np.nan, hi=np.nan)
        head.append(dict(
            coef=n, ch2=C.CH2_LABEL.get(n, ""), estimated=estd,
            value_median=float(np.nanmedian(np.abs(coef[:, :, j]))),
            rel_sd_ref_median=med,
            rel_sd_ref_q1=float(np.nanpercentile(rel, 25)) if estd else np.nan,
            rel_sd_ref_q3=float(np.nanpercentile(rel, 75)) if estd else np.nan,
            rel_sd_hl=h["hl"], rel_sd_hl_lo=h["lo"], rel_sd_hl_hi=h["hi"],
            frac_null_median=fnm,
            sd_ref_median=float(np.nanmedian(sub.sd_ref.to_numpy())),
            verdict=verdict))
    return marg, pd.DataFrame(head), dict(lam_rel=lam_rel, cut_rel=cut_rel,
                                          il=il, ik=ik, names=names,
                                          years=years, coef=coef,
                                          sd_lam=sd_lam, est=est)


def ranking_stability(ctx):
    """Does the ORDER of the coefficients by relative marginal SD survive the
    choice of regulariser?  The level does not; if the order does, the result
    is reportable without committing to a prior."""
    names, coef, sd_lam, est = ctx["names"], ctx["coef"], ctx["sd_lam"], ctx["est"]
    keep = [j for j in range(len(names)) if est[:, :, j].any()]
    rows = []
    base = None
    for a, lv in enumerate(ctx["lam_rel"]):
        rel = np.nanmedian(sd_lam[:, a][:, :, keep]
                           / np.maximum(np.abs(coef[:, :, keep]), 1e-300),
                           axis=(0, 1))
        order = np.argsort(rel)
        if base is None:
            base = order
        # Spearman on the ranks, against the reference ordering
        r1 = np.empty(len(keep)); r1[order] = np.arange(len(keep))
        r0 = np.empty(len(keep)); r0[base] = np.arange(len(keep))
        rho = float(np.corrcoef(r0, r1)[0, 1])
        rows.append(dict(lam_rel=float(lv), spearman_vs_first=rho,
                         order=" > ".join(names[keep[k]] for k in order)))
    return pd.DataFrame(rows)


# ===========================================================================
# Part 3 — stiff / sloppy combinations
# ===========================================================================
def part3(ds, seeds):
    names = [str(x) for x in ds[0]["coef_names"]]
    years = np.asarray(ds[0]["years"], float).ravel()
    rows = []
    for i, s in enumerate(seeds):
        for tag in ("stiff", "sloppy"):
            R = np.asarray(ds[i][f"{tag}_response"])       # (m, T, C)
            sv = np.asarray(ds[i][f"sv_{tag}"], float)
            for m in range(R.shape[0]):
                w = np.abs(R[m]).sum(0)
                w = w / max(w.sum(), 1e-300)
                top = np.argsort(w)[::-1][:3]
                rows.append(dict(
                    seed=int(s), kind=tag, mode=m, sv=float(sv[m]),
                    eig=float(sv[m] ** 2),
                    response_norm=float(np.linalg.norm(R[m])),
                    top1=names[top[0]], top1_share=float(w[top[0]]),
                    top2=names[top[1]], top2_share=float(w[top[1]]),
                    top3=names[top[2]], top3_share=float(w[top[2]]),
                    **{f"share_{n}": float(w[j]) for j, n in enumerate(names)}))
    return pd.DataFrame(rows)


# ===========================================================================
# Part 4 — profile likelihood on the alpha levels
# ===========================================================================
def part4(fit, cfg, seed_list, ds, seeds, *, steps, lr, out_dir,
          budget_check=(0, 2, 400)):
    """Profile the LEVEL of each alpha channel, avoiding the quadratic
    approximation.

    The profile is taken under the trained objective from the fitted optimum
    with a FIXED optimisation budget, so `Phi(delta)` for `delta != 0` is an
    upper bound on the true constrained minimum while `Phi(0)` is already
    converged.  The interval that comes out is therefore NARROWER than the
    true profile interval — anti-conservative, and reported as such.
    """
    import jax
    import jax.numpy as jnp
    import zinc_cf_lab as cflab

    idx = {int(s): i for i, s in enumerate(seeds)}
    jobs = [(int(s), k, int(steps)) for s in seed_list for k in range(4)]
    if budget_check is not None:
        # One (seed, channel) repeated at a larger budget.  The profile is
        # anti-conservative by construction — Phi(0) is converged and
        # Phi(delta != 0) is not — so how much the interval widens with budget
        # is the only honest way to say how anti-conservative it is.
        jobs.append((int(budget_check[0]), int(budget_check[1]),
                     int(budget_check[2])))
    mask = np.asarray(fit._split_indices_in_all()["train"], bool) \
        | np.asarray(fit._split_indices_in_all()["val"], bool)
    rows = []
    cache = {}
    for s, k, nst in jobs:
        if s not in cache:
            cache[s] = jax.flatten_util.ravel_pytree(
                cflab.load_params(os.path.join(F.WEIGHTS_DIR_DEFAULT,
                                               f"A_seed{s}.npz")))[0]
        theta = cache[s]
        kap2 = float(ds[idx[s]]["kappa2"])
        t0 = time.time()
        recs, c_hat = F.profile_alpha(fit, cfg, jnp.asarray(theta), k,
                                      PROFILE_OFFSETS, steps=nst, lr=lr,
                                      mask=mask)
        base = [r for r in recs if r["offset"] == 0.0][0]
        phi0 = base["loss_data"] + base["loss_pen"]
        for r in recs:
            phi = r["loss_data"] + r["loss_pen"]
            rows.append(dict(seed=int(s), alpha=F.COEF_NAMES[k], c_hat=c_hat,
                             kappa2=kap2, steps=int(nst), **r,
                             phi=phi, dchi2=(phi - phi0) / kap2))
        print(f"  [seed {s}] {F.COEF_NAMES[k]:10s} steps={nst:4d} "
              f"{time.time()-t0:.0f}s", flush=True)
        jax.clear_caches()
    prof = pd.DataFrame(rows)

    return prof, profile_intervals(prof)


def profile_intervals(prof):
    """95% profile interval: the convex hull of `{delta : 2 dlogL <= 3.84}`.

    The OUTERMOST crossing on each side, not the innermost.  The curves are
    not monotone — the constrained optimum at a shifted level is sometimes
    *better* than at the fitted level, because the published weights are the
    endpoint of a stochastic window-curriculum objective and not a stationary
    point of the full-trainval objective profiled here — so an innermost-
    crossing rule truncates the interval at a numerical bump near the centre.
    A non-convex likelihood set is reported by its hull, which is the standard
    convention, and the whole curve is in `wp4c_profile.csv` for anyone who
    wants the set itself.
    """
    ivals = []
    for (s, k, nst), sub in prof.groupby(["seed", "channel", "steps"]):
        sub = sub.sort_values("offset")
        o = sub.offset.to_numpy()
        c = sub.dchi2.to_numpy()

        def cross(side):
            m = (o > 0) if side == "hi" else (o < 0)
            oo, cc = o[m], c[m]
            if side == "lo":
                oo, cc = oo[::-1], cc[::-1]          # outward from 0
            below = np.nonzero(cc < CHI2_95)[0]
            if below.size == 0:                       # already above at the
                x0, y0 = 0.0, 0.0                     # first grid point
                i = 0
            else:
                i = below[-1] + 1
                if i >= oo.size:
                    return np.nan                     # never crosses on the grid
                x0, y0 = oo[i - 1], cc[i - 1]
            x1, y1 = oo[i], cc[i]
            return x0 + (CHI2_95 - y0) * (x1 - x0) / max(y1 - y0, 1e-300)

        lo, hi = cross("lo"), cross("hi")
        half = [abs(v) for v in (lo, hi) if np.isfinite(v)]
        ivals.append(dict(
            seed=int(s), channel=int(k), steps=int(nst),
            alpha=F.COEF_NAMES[int(k)], c_hat=float(sub.c_hat.iloc[0]),
            delta_lo=lo, delta_hi=hi,
            width_nats=(hi - lo) if np.isfinite(hi) and np.isfinite(lo)
            else np.nan,
            half_width_nats=float(np.mean(half)) if half else np.nan,
            budget_gain=float(sub.loss_unoptimised_at_theta0.iloc[0]
                              - sub.phi[sub.offset == 0.0].iloc[0]),
            min_dchi2=float(c.min()),
            argmin_offset=float(o[int(np.argmin(c))]),
            max_violation=float(np.max(np.abs(sub.violation)))))
    return pd.DataFrame(ivals)





# ===========================================================================
# Part 5 — Laplace against the across-seed spread, and the sigma handoff
# ===========================================================================
def part5(ctx, ds, seeds, out_npz):
    """Both uncertainty sources on one axis, and the `sigma` / covariance
    arrays WP-2f and WP-8h consume.

    The across-seed spread (WP-1c flag 1) carries initialisation and
    optimisation path only — same data, same split — so it is a LOWER bound on
    the parameter variance and says nothing about sampling error.  The Laplace
    marginal is the sampling error of the estimator at fixed initialisation.
    They measure different things, which is why WP-8h combines them by Rubin's
    rules rather than choosing between them.

    Covariances are emitted alongside the marginals because the two objects
    WP-2f Part 5 and WP-8h need — the standard error of a characteristic root,
    and a band on an assembled `A(t)` entry — are delta-method pushforwards
    that are simply wrong if the coefficients are treated as independent.  The
    manufacturing shares are exactly anti-correlated by their simplex.
    """
    names, years = ctx["names"], ctx["years"]
    coef, sd_lam, il = ctx["coef"], ctx["sd_lam"], ctx["il"]
    sd_ref = sd_lam[:, il]                                # (S, T, C)
    seed_sd = np.std(coef, axis=0, ddof=1)                # (T, C)
    seed_sd_b = np.broadcast_to(seed_sd, coef.shape)

    rows = []
    for t, y in enumerate(years):
        for j, n in enumerate(names):
            lp = sd_ref[:, t, j]
            rows.append(dict(
                year=float(y), coef=n,
                value_median=float(np.median(coef[:, t, j])),
                laplace_sd_median=float(np.median(lp)),
                seed_sd=float(seed_sd[t, j]),
                ratio_laplace_over_seed=float(np.median(lp)
                                              / max(seed_sd[t, j], 1e-300)),
                combined_sd_median=float(np.median(
                    np.sqrt(lp ** 2 + seed_sd[t, j] ** 2)))))
    cmp = pd.DataFrame(rows)

    # --- covariances, padded to zinc_circ_lab.PARAM_NAMES -----------------
    C_lap = np.stack([np.asarray(d["coef_cov"])[il] for d in ds])  # (S,T,17,17)
    n_s, n_y, n_c = coef.shape
    P = len(C.PARAM_NAMES)

    def pad_v(a):
        out = np.zeros((n_s, n_y, P)); out[:, :, :n_c] = a; return out

    def pad_m(a):
        out = np.zeros(a.shape[:-2] + (P, P))
        out[..., :n_c, :n_c] = a
        return out

    # Rubin's rules across the seed ensemble (Rubin 1987): the total
    # covariance of the pooled estimate is the mean within-seed covariance
    # plus (1 + 1/S) times the between-seed covariance.
    gbar = coef.mean(0)                                          # (T, C)
    dev = coef - gbar[None]
    B = np.einsum("stc,std->tcd", dev, dev) / max(n_s - 1, 1)    # (T,C,C)
    Wbar = C_lap.mean(0)                                         # (T,C,C)
    Total = Wbar + (1.0 + 1.0 / n_s) * B

    sig = dict(sigma_laplace=pad_v(sd_ref),
               sigma_seed=pad_v(seed_sd_b),
               sigma_combined=pad_v(np.sqrt(sd_ref ** 2 + seed_sd_b ** 2)),
               cov_laplace=pad_m(C_lap),
               cov_between=pad_m(B), cov_within_mean=pad_m(Wbar),
               cov_rubin_total=pad_m(Total),
               coef_mean=coef.mean(0), coef_per_seed=coef)
    np.savez_compressed(
        out_npz, years=years, seeds=seeds,
        param_names=np.array(C.PARAM_NAMES, dtype=object),
        coef_names=np.array(names, dtype=object),
        lam_ref=np.asarray(LAM_REF),
        note=np.asarray(
            "sigma_laplace: Laplace marginal SD of each coefficient at "
            "lambda_rel=%g, per seed and year (run_wp4c).  sigma_seed: "
            "across-seed SD (WP-1c flag 1: a lower bound).  sigma_combined: "
            "per-seed quadrature sum.  cov_laplace: per-seed Laplace "
            "covariance.  cov_rubin_total = cov_within_mean + (1+1/S) "
            "cov_between, the WP-8h pooled product.  Columns 17-19 (mu) are "
            "zero: the cohort lifetimes are fixed by construction." % LAM_REF,
            dtype=object),
        **sig)
    return cmp


# ===========================================================================
# figures
# ===========================================================================
def figures(spec_long, spec_summ, marg, head, ctx, cmp, prof, ivals, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names, years = ctx["names"], ctx["years"]
    lam_rel = ctx["lam_rel"]

    # ---- fig 1: the spectrum -------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for s, sub in spec_long.groupby("seed"):
        e = sub.eig_rel.to_numpy()
        ax[0].semilogy(np.arange(e.size), np.maximum(e, 1e-20),
                       color=COL[0], alpha=0.25, lw=0.8)
    med = spec_long.groupby("index").eig_rel.median()
    ax[0].semilogy(med.index, np.maximum(med.to_numpy(), 1e-20), color="k", lw=1.8,
                   label="median over 35 seeds")
    ax[0].set_xlabel("eigenvalue index")
    ax[0].set_ylabel(r"$\lambda_i/\lambda_1$   ($J^\mathsf{T}J$)")
    ax[0].set_title("Gauss–Newton spectrum of the Stage B objective")
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].grid(alpha=0.25)

    ax[1].hist(spec_summ.log10_decades, bins=12, color=COL[1], alpha=0.85)
    ax[1].set_xlabel(r"$\log_{10}(\lambda_1/\lambda_{\mathrm{rank}})$")
    ax[1].set_ylabel("seeds")
    ax[1].set_title("dynamic range of the spectrum")
    ax[1].grid(alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"wp4c_spectrum.{ext}"), dpi=200)
    plt.close(fig)

    # ---- fig 2: marginals ----------------------------------------------
    est = [n for n in names if bool(head.set_index("coef").loc[n, "estimated"])]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8))

    ax = axes[0, 0]
    for j, n in enumerate(est):
        sub = marg[marg.coef == n]
        v = [np.nanmedian(sub[f"sd_lam{lv:g}"].to_numpy()
                          / np.maximum(np.abs(sub.value.to_numpy()), 1e-300))
             for lv in lam_rel]
        ax.loglog(lam_rel, v, marker="o", ms=3, lw=1.2,
                  color=COL[j % len(COL)],
                  ls=("-" if j < len(COL) else "--"), label=n)
    ax.axvline(LAM_REF, color=GREY, ls=":")
    ax.set_xlabel(r"ridge $\lambda/\lambda_1$")
    ax.set_ylabel("relative marginal SD (median)")
    ax.set_title("the level depends on the prior; the ranking does not")
    ax.legend(frameon=False, fontsize=6, ncol=2)
    ax.grid(alpha=0.25, which="both")

    ax = axes[0, 1]
    h = head[head.estimated].sort_values("rel_sd_ref_median")
    y = np.arange(len(h))
    ax.barh(y, h.rel_sd_ref_median, color=COL[0], height=0.6)
    for i, r in enumerate(h.itertuples()):
        ax.plot([r.rel_sd_ref_q1, r.rel_sd_ref_q3], [i, i], color="k", lw=1.0)
    ax.axvline(BAND_TIGHT, color=COL[1], ls="--", lw=1)
    ax.axvline(BAND_LOOSE, color=COL[2], ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(h.coef, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel(f"relative marginal SD at $\\lambda/\\lambda_1={LAM_REF:g}$")
    ax.set_title("practical identifiability (median, IQR)")
    ax.grid(alpha=0.25, axis="x", which="both")

    ax = axes[1, 0]
    h2 = head[head.estimated].sort_values("frac_null_median")
    y = np.arange(len(h2))
    ax.barh(y, h2.frac_null_median, color=COL[3], height=0.6)
    ax.axvline(NULL_ALARM, color=COL[2], ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(h2.coef, fontsize=7)
    ax.set_xlabel(f"share of $\\nabla g$ outside the data-informed subspace "
                  f"(cut {CUT_REF:g})")
    ax.set_title("prior-free: is the coefficient determined at all?")
    ax.grid(alpha=0.25, axis="x")

    ax = axes[1, 1]
    for j, n in enumerate(F.COEF_NAMES[:4]):
        sub = cmp[cmp.coef == n]
        ax.semilogy(sub.year, sub.laplace_sd_median / np.abs(sub.value_median),
                    color=COL[j], lw=1.5, label=f"{n} (Laplace)")
        ax.semilogy(sub.year, sub.seed_sd / np.abs(sub.value_median),
                    color=COL[j], lw=1.0, ls="--", alpha=0.7)
    ax.set_xlabel("year")
    ax.set_ylabel("relative SD")
    ax.set_title("Laplace (solid) vs across-seed spread (dashed)")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"wp4c_marginals.{ext}"), dpi=200)
    plt.close(fig)

    # ---- fig 3: profile likelihood -------------------------------------
    if prof is not None and len(prof):
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), sharey=True)
        for k in range(4):
            ax = axes[k]
            sub = prof[prof.channel == k]
            for s, ss in sub.groupby("seed"):
                ss = ss.sort_values("offset")
                ax.plot(ss.offset, ss.dchi2, color=COL[0], alpha=0.5, lw=1.0)
            ax.axhline(CHI2_95, color=COL[2], ls="--", lw=1)
            ax.set_xlabel(r"$\delta$ = shift in mean $\log\alpha$ (nats)")
            ax.set_title(F.COEF_NAMES[k], fontsize=9)
            ax.set_ylim(0, max(6.0, float(np.nanpercentile(sub.dchi2, 98))))
            ax.grid(alpha=0.25)
        axes[0].set_ylabel(r"$2\Delta$ log-posterior  ($\chi^2_1$)")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out_dir, f"wp4c_profile.{ext}"), dpi=200)
        plt.close(fig)


# ===========================================================================
# main
# ===========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--seed-dir", default=SEED_DIR)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--profile-seeds", default="0,1,2,3,4")
    ap.add_argument("--profile-steps", type=int, default=150)
    ap.add_argument("--profile-lr", type=float, default=1e-3)
    args = ap.parse_args(argv)

    if args.check:
        F.check(verbose=True)
        print("run_wp4c --check")
        print("=" * 74)
        print(f"  reference ridge      : lambda/lambda_1 = {LAM_REF:g} "
              f"(= anchor_v4 weight_decay)")
        print(f"  reference cut        : {CUT_REF:g} of the largest eigenvalue")
        print(f"  identifiability bands: rel SD < {BAND_TIGHT} identifiable, "
              f"< {BAND_LOOSE} weak, else practically unidentifiable")
        print(f"  null alarm           : frac_null > {NULL_ALARM} "
              f"=> not determined by the data")
        print(f"  profile offsets      : {PROFILE_OFFSETS} nats, "
              f"chi2_1(95%) = {CHI2_95:.4f}")
        print( "  SYNTHETIC ARM        : NOT RUN — needs WP-3 "
               "(zinc_synth_lab.py / analysis/synth) which does not exist.")
        print("=" * 74)
        return 0

    os.makedirs(args.out, exist_ok=True)
    ds, seeds = load_seeds(args.seed_dir)
    print(f"loaded {len(seeds)} seeds from {args.seed_dir}")

    spec_summ, spec_long = part1(ds, seeds)
    spec_summ.to_csv(os.path.join(args.out, "wp4c_spectrum_summary.csv"),
                     index=False)
    spec_long.to_csv(os.path.join(args.out, "wp4c_spectrum.csv"), index=False)
    print(f"  rank            : {spec_summ['rank'].median():.0f} of "
          f"{spec_summ.n_res.iloc[0]} residuals, "
          f"{spec_summ.n_par.iloc[0]} live parameters "
          f"({spec_summ.n_dead.iloc[0]} dead by construction)")
    print(f"  dynamic range   : 10^{spec_summ.log10_decades.median():.1f} "
          f"[{spec_summ.log10_decades.min():.1f}, "
          f"{spec_summ.log10_decades.max():.1f}]")
    print(f"  p_eff at ref    : {spec_summ.p_eff_ref.median():.1f}")
    print(f"  kappa           : {spec_summ.kappa.median():.4f}")

    marg, head, ctx = part2(ds, seeds)
    marg.to_csv(os.path.join(args.out, "wp4c_marginals.csv"), index=False)
    head.to_csv(os.path.join(args.out, "wp4c_identifiability.csv"), index=False)
    rank_stab = ranking_stability(ctx)
    rank_stab.to_csv(os.path.join(args.out, "wp4c_ranking_stability.csv"),
                     index=False)
    print("\n  practical identifiability at the reference ridge:")
    for r in head.itertuples():
        if not r.estimated:
            continue
        print(f"    {r.coef:16s} rel SD {r.rel_sd_ref_median*100:7.2f}%  "
              f"frac_null {r.frac_null_median:5.3f}   {r.verdict}")

    st = part3(ds, seeds)
    st.to_csv(os.path.join(args.out, "wp4c_stiff_directions.csv"), index=False)

    cmp = part5(ctx, ds, seeds,
                os.path.join(args.out, "wp4c_sigma.npz"))
    cmp.to_csv(os.path.join(args.out, "wp4c_vs_seed_spread.csv"), index=False)

    prof = ivals = None
    if args.profile:
        import zinc_cf_lab as cflab
        cfg = F.load_anchor_config()
        fit = cflab.init_fit(cfg=cfg, seed=0)
        sl = [int(x) for x in args.profile_seeds.split(",") if x.strip()]
        print(f"\n  profile likelihood on {len(sl)} seeds x 4 channels "
              f"x {len(PROFILE_OFFSETS)} offsets, {args.profile_steps} steps")
        prof, ivals = part4(fit, cfg, sl, ds, seeds,
                            steps=args.profile_steps, lr=args.profile_lr,
                            out_dir=args.out)
        prof.to_csv(os.path.join(args.out, "wp4c_profile.csv"), index=False)
        ivals.to_csv(os.path.join(args.out, "wp4c_profile_intervals.csv"),
                     index=False)
    elif os.path.exists(os.path.join(args.out, "wp4c_profile.csv")):
        prof = pd.read_csv(os.path.join(args.out, "wp4c_profile.csv"))
        ivals = profile_intervals(prof)
        ivals.to_csv(os.path.join(args.out, "wp4c_profile_intervals.csv"),
                     index=False)

    # validation table
    vrows = []
    for d in ds:
        if "validation_json" in d.files:
            v = json.loads(str(d["validation_json"]))
            for k, val in v.items():
                if isinstance(val, (int, float)):
                    vrows.append(dict(seed=int(d["seed"]), check=k, value=val))
    for d in ds:
        c = json.loads(str(d["residual_check"]))
        vrows.append(dict(seed=int(d["seed"]),
                          check="||r||^2 vs stageB_window_loss (rel)",
                          value=c["rel_err"]))
    pd.DataFrame(vrows).to_csv(os.path.join(args.out, "wp4c_validation.csv"),
                               index=False)

    figures(spec_long, spec_summ, marg, head, ctx, cmp, prof, ivals, args.out)
    print(f"\nwrote wp4c_* to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
