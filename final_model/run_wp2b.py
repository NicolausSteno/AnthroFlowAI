#!/usr/bin/env python3
"""
run_wp2b.py — WP-2b: eigenstructure trajectories of the estimated A(t)
======================================================================

Reads the WP-2a deliverable `analysis/wp2a_A_of_t.npz` — `A` of shape
(n_seeds, n_years, 6, 6) in the augmented state

    S_ode = [S_conc, S_ref, S_iu_short, S_iu_med, S_iu_long, S_scrap]

— and computes, per year node and per seed, the frozen-time eigensystem that
Chapter 2 derives but has no estimated matrix to evaluate on.  No refit: WP-2a
established that `anchor_v4` sets `use_stock_input=False`, so
`dS/dt = A(t)S + b(t)` holds identically and `A(t) = df/dS` exactly.  The
eigen-analysis below is therefore exact at the coefficient level rather than a
linearisation.

Reported                                     # Ch2 Eq. jordandecom, Eq. homogsol
--------                                     #     e^{At} = Q e^{Jt} Q^{-1}
  * the six eigenvalues at every (seed, year), sorted and, separately,
    matched to the compartment each mode lives in;
  * timescales `1/|Re lambda|` and, for any complex pair, the oscillation
    period `2*pi/|Im lambda|` — the object comparable to Chapter 2's copper
    upstream pair `-0.06 +/- 0.334i` (period 18.8 yr);
  * the dominant mode, and how far the *system's* slowest timescale departs
    from the slowest compartment lifetime (MU_COHORTS[2] = 44 yr) because the
    recycling loop returns material;
  * mode identification by participation factor
    `P_ki = q_ki v_ik / (q_k . v_k)`, which is exactly Chapter 2's diagonal
    eigenvalue sensitivity `dlambda_k/da_ii`;         # Ch2 Eq. asens
  * near-degeneracy diagnostics — eigenvalue condition number `1/|q_k.v_k|`
    and nearest-neighbour gaps — which are the guard WP-2c needs before
    applying Eq. eigenvalue_sensitivity;              # Ch2 Eq. eigenvalue_sensitivity
  * an adiabaticity index `|dlambda_k/dt| / lambda_k^2` quantifying how far
    the frozen-time reading can be trusted, since stability of the
    non-autonomous system is governed by Phi(t,t0), not by the frozen
    spectrum.                                         # Ch2 Eq. product_transition_matrices
                                                      # Ch2 Eq. tracking_error

Because the spectrum turns out to be real everywhere, the "any complex pair"
question is answered with a magnitude rather than a bare null: every
structurally non-zero coefficient is scanned over twelve orders of magnitude
(with the same column's diagonal absorbing the change, so the column loss rate
is unchanged) to see whether any single-coefficient direction reaches an
oscillatory regime, and a random draw over the same sparsity pattern
establishes that the topology itself does admit complex spectra.

    python run_wp2b.py --check
    python run_wp2b.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.linalg as sla
from scipy import stats
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "analysis")
A_NPZ = os.path.join(OUT_DIR, "wp2a_A_of_t.npz")

STATE = ["S_conc", "S_ref", "S_iu_short", "S_iu_med", "S_iu_long", "S_scrap"]

# Chapter 2, Table tab:params: the copper upstream block M produced the
# dominant oscillatory pair -0.06 +/- 0.334i, period 2*pi/0.334 = 18.8 yr.
CH2_SOA_PAIR = (-0.06, 0.334)

# Okabe-Ito, colourblind-safe (same assignment as run_wp2a.py).
COL = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00", "#56B4E9"]
GREY = "#555555"

IM_TOL = 1e-9        # |Im lambda| above this counts as a genuine complex pair


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_A(path=A_NPZ):
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found — run run_wp2a.py first")
    d = np.load(path, allow_pickle=True)
    A = np.asarray(d["A"], float)
    if A.ndim != 4 or A.shape[2:] != (6, 6):
        raise AssertionError(f"unexpected A shape {A.shape}")
    if not np.isfinite(A).all():
        raise AssertionError("A contains non-finite entries")
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    return d, A, md5


def sanity(A):
    """Re-assert the WP-2a invariants the eigen-analysis leans on, so a stale
    or hand-edited npz cannot pass silently."""
    off = ~np.eye(6, dtype=bool)
    return dict(
        min_offdiagonal=float(A[:, :, off].min()),          # Metzler
        max_diagonal=float(np.diagonal(A, axis1=2, axis2=3).max()),
        max_col_sum=float(A.sum(axis=2).max()),             # <= 0, Ch2 Eq. uniform_column_condition
        row0_offdiag_maxabs=float(np.abs(A[:, :, 0, 1:]).max()),
    )


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def hodges_lehmann(v, conf=0.95):
    """HL location estimate of a one-sample distribution + distribution-free CI.

    Point estimate is the median of the Walsh averages (x_i + x_j)/2, i <= j;
    the CI endpoints are order statistics of those Walsh averages indexed by
    the Wilcoxon signed-rank critical value (normal approximation with
    continuity correction, accurate at n = 35).  Project convention is HL
    intervals rather than significance tests.  Same implementation as
    `run_wp1b.py` / `run_wp1d.py`.
    """
    v = np.asarray([x for x in np.asarray(v, float) if np.isfinite(x)])
    n = v.size
    if n == 0:
        return dict(hl=np.nan, lo=np.nan, hi=np.nan, n=0)
    w = (v[:, None] + v[None, :]) / 2.0
    walsh = np.sort(w[np.triu_indices(n)])
    est = float(np.median(walsh))
    if n < 6:
        return dict(hl=est, lo=np.nan, hi=np.nan, n=n)
    N = walsh.size
    z = stats.norm.ppf(1.0 - (1.0 - conf) / 2.0)
    sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    k = max(int(np.floor(N / 2.0 - z * sd)), 0)
    return dict(hl=est, lo=float(walsh[k]),
                hi=float(walsh[min(N - 1 - k, N - 1)]), n=n)


# ---------------------------------------------------------------------------
# eigensystem
# ---------------------------------------------------------------------------
def eigensystem(A):
    """Right/left eigenvectors, participation factors and mode labels.

    Left eigenvectors are normalised so that `q_k . v_k = 1`, i.e. `Q^H` is
    the inverse of the modal matrix `V`.  Then                # Ch2 Eq. asens
    `P_ki = q_ki v_ik = [V^-1 I_ii V]_kk = dlambda_k / da_ii`, the
    participation of compartment `i` in mode `k`; `sum_i P_ki = 1` and
    `sum_k P_ki = 1` both hold and are asserted.

    The mode-to-compartment assignment is the one maximising total |P| over
    bijections (Hungarian); the greedy argmax is recorded alongside so that
    disagreement — which would mean the modes are not cleanly localised — is
    visible rather than hidden.
    """
    S, T = A.shape[:2]
    lam = np.zeros((S, T, 6), complex)
    V = np.zeros((S, T, 6, 6), complex)
    Q = np.zeros((S, T, 6, 6), complex)
    P = np.zeros((S, T, 6, 6), complex)       # [seed, year, mode k, state i]
    cond = np.zeros((S, T, 6))                # 1/|q_k.v_k| at unit norms
    lab = np.zeros((S, T, 6), int)            # Hungarian mode -> state
    lab_greedy = np.zeros((S, T, 6), int)

    for s in range(S):
        for t in range(T):
            w, ql, vr = sla.eig(A[s, t], left=True, right=True)
            vr = vr / np.linalg.norm(vr, axis=0)
            ql = ql / np.linalg.norm(ql, axis=0)
            denom = np.einsum("ik,ik->k", ql.conj(), vr)
            cond[s, t] = 1.0 / np.abs(denom)
            ql = ql / denom.conj()                       # now q_k . v_k = 1
            Pkt = (ql.conj() * vr).T                     # (mode, state)
            lam[s, t], V[s, t], Q[s, t], P[s, t] = w, vr, ql, Pkt
            absP = np.abs(Pkt)
            lab_greedy[s, t] = absP.argmax(axis=1)
            lab[s, t] = linear_sum_assignment(-absP)[1]

    assert np.abs(P.sum(axis=3) - 1.0).max() < 1e-8, "participation rows"
    assert np.abs(P.sum(axis=2) - 1.0).max() < 1e-8, "participation columns"
    return lam, V, Q, P, cond, lab, lab_greedy


def by_state(x, lab):
    """Re-index a per-mode quantity into compartment order, so mode `k` of the
    output is the mode localised on `STATE[k]` — comparable across seeds and
    years without relying on the sort order."""
    out = np.zeros_like(x)
    np.put_along_axis(out, lab, x, axis=-1)
    return out


def derived(lam):
    """Timescale, oscillation period and damping ratio of every mode."""
    re, im = lam.real, lam.imag
    with np.errstate(divide="ignore", invalid="ignore"):
        timescale = np.where(re != 0, 1.0 / np.abs(re), np.inf)
        period = np.where(np.abs(im) > IM_TOL, 2.0 * np.pi / np.abs(im), np.inf)
        damping = np.where(np.abs(lam) > 0, -re / np.abs(lam), np.nan)
    return timescale, period, damping


def gaps(lam):
    """Absolute and relative distance from each eigenvalue to its nearest
    neighbour in the spectrum — the quantity that has to be large for
    Ch2 Eq. eigenvalue_sensitivity to be applicable in WP-2c."""
    dist = np.abs(lam[..., :, None] - lam[..., None, :])
    dist[..., np.arange(6), np.arange(6)] = np.inf
    g = dist.min(axis=-1)
    return g, g / np.maximum(np.abs(lam), 1e-300)


def adiabaticity(lam_state, years):
    """`eps_k = |dlambda_k/dt| / lambda_k^2`, dimensionless.

    `eps << 1` is the condition under which the frozen-time spectrum is a
    faithful description of the non-autonomous dynamics: the mode's own decay
    (rate `|lambda|`) is fast relative to the rate at which the mode itself
    moves (`|dlambda/dt| / |lambda|`).  Reported twice — from the raw
    year-on-year gradient, and from a cubic trend, since annual jitter in the
    fitted coefficients inflates the raw derivative without corresponding to
    genuine drift of the mode.
    """
    re = lam_state.real
    eps_raw = np.abs(np.gradient(re, years, axis=1)) / re**2
    x = years - years.mean()
    trend = np.empty_like(re)
    for s in range(re.shape[0]):
        for k in range(re.shape[2]):
            trend[s, :, k] = np.polyval(np.polyfit(x, re[s, :, k], 3), x)
    eps_trend = np.abs(np.gradient(trend, years, axis=1)) / trend**2
    return eps_raw, eps_trend


# ---------------------------------------------------------------------------
# how far is the estimate from an oscillatory regime?
# ---------------------------------------------------------------------------
def oscillation_scan(A, n_grid=61, span=6.0):
    """Single-coefficient distance to a complex pair.

    Each structurally non-zero off-diagonal `a_ij` is scaled by `g` over
    `10^-span .. 10^+span`, with `a_jj` absorbing the change so that the
    column sum — and therefore the loss rate out of compartment `j`
    (Ch2 Eq. uniform_column_condition) — is unchanged.  The perturbed matrix
    stays Metzler for every `g > 0`, and its diagonal stays non-positive
    because `|a_jj| >= sum_i!=j a_ij`.  Returns, per entry, the fraction of
    (seed, year) nodes at which *any* `g` produces `|Im lambda| > IM_TOL`.
    """
    S, T = A.shape[:2]
    struct = np.any(np.abs(A) > 0, axis=(0, 1))
    entries = [(i, j) for i in range(6) for j in range(6)
               if i != j and struct[i, j]]
    gs = np.logspace(-span, span, n_grid)
    up, dn = gs > 1.0, gs < 1.0
    rows = []
    for i, j in entries:
        B = np.broadcast_to(A[:, :, None], (S, T, n_grid, 6, 6)).copy()
        delta = A[:, :, i, j][:, :, None] * (gs - 1.0)
        B[:, :, :, i, j] += delta
        B[:, :, :, j, j] -= delta
        assert np.diagonal(B, axis1=3, axis2=4).max() <= 1e-12
        assert B[:, :, :, i, j].min() >= 0.0
        cx = np.abs(np.linalg.eigvals(B).imag).max(axis=-1) > IM_TOL
        # smallest amplification, and mildest reduction, that reaches complex
        cu, cd = cx[:, :, up], cx[:, :, dn][:, :, ::-1]
        crit_up = np.where(cu.any(-1), gs[up][cu.argmax(-1)], np.nan)
        crit_dn = np.where(cd.any(-1), gs[dn][::-1][cd.argmax(-1)], np.nan)
        # Every off-diagonal is a share of its parent's total outflow rate, so
        # `a_ij <= -a_jj` bounds how far the entry can physically be pushed;
        # for A[5,k] = tau_olds/mu_k this is exactly `tau_olds <= 1`.
        ceiling = -A[:, :, j, j] / np.where(A[:, :, i, j] > 0,
                                            A[:, :, i, j], np.nan)
        rows.append(dict(row=int(i), col=int(j),
                         to_stock=STATE[i], from_stock=STATE[j],
                         median_value=float(np.median(A[:, :, i, j])),
                         reaches_complex=bool(cx.any()),
                         frac_nodes_complex=float(cx.any(axis=2).mean()),
                         crit_up_min=float(np.nanmin(crit_up))
                         if np.isfinite(crit_up).any() else np.inf,
                         crit_up_median=float(np.nanmedian(crit_up))
                         if np.isfinite(crit_up).any() else np.inf,
                         crit_down_max=float(np.nanmax(crit_dn))
                         if np.isfinite(crit_dn).any() else np.nan,
                         max_feasible_multiplier=float(np.nanmedian(ceiling)),
                         g_min=float(gs.min()), g_max=float(gs.max())))
    return pd.DataFrame(rows)


def topology_admits_complex(A, n=200000, rng_seed=0):
    """Control for the null: draw random Metzler matrices with *the same
    sparsity pattern* and column sums <= 0, and report how often the spectrum
    is complex.  If the topology never oscillated, the real spectrum of the
    estimate would be a structural fact rather than an estimate about the
    zinc cycle."""
    struct = np.any(np.abs(A) > 0, axis=(0, 1))
    entries = [(i, j) for i in range(6) for j in range(6)
               if i != j and struct[i, j]]
    rng = np.random.default_rng(rng_seed)
    B = np.zeros((n, 6, 6))
    for i, j in entries:
        B[:, i, j] = 10.0 ** rng.uniform(-3, 2, n)
    for j in range(6):
        B[:, j, j] = -B[:, :, j].sum(1) * 10.0 ** rng.uniform(0, 0.5, n)
    w = np.linalg.eigvals(B)
    cx = np.abs(w.imag).max(axis=-1) > IM_TOL
    per = np.where(cx, 2 * np.pi / np.maximum(np.abs(w.imag).max(axis=-1), 1e-300),
                   np.nan)
    return dict(n_draws=int(n), frac_complex=float(cx.mean()),
                max_abs_im=float(np.abs(w.imag).max()),
                median_period_yr=float(np.nanmedian(per)))


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def summary_table(lam_state, ts_state, per_state, years, seeds):
    rows = []
    for k in range(6):
        for ti, y in enumerate(years):
            re = lam_state[:, ti, k].real
            im = np.abs(lam_state[:, ti, k].imag)
            ts = ts_state[:, ti, k]
            q1, med, q3 = np.percentile(re, [25, 50, 75])
            t1, tm, t3 = np.percentile(ts, [25, 50, 75])
            rows.append(dict(
                mode=STATE[k], year=float(y),
                re_median=float(med), re_q1=float(q1), re_q3=float(q3),
                timescale_median=float(tm), timescale_q1=float(t1),
                timescale_q3=float(t3),
                max_abs_im=float(im.max()),
                n_complex=int((im > IM_TOL).sum()),
                period_median=float(np.median(per_state[:, ti, k])),
                n_seeds=len(seeds)))
    return pd.DataFrame(rows)


def per_seed_table(lam_state, ts_state, per_state, damp_state, cond_state,
                   gap_state, relgap_state, part_own, shift_state, years, seeds):
    S, T = lam_state.shape[:2]
    si, ti, ki = np.meshgrid(np.arange(S), np.arange(T), np.arange(6),
                             indexing="ij")
    return pd.DataFrame(dict(
        seed=seeds[si].ravel(),
        year=years[ti].ravel(),
        mode=np.array(STATE)[ki.ravel()],
        re_lambda=lam_state.real.ravel(),
        im_lambda=lam_state.imag.ravel(),
        timescale_yr=ts_state.ravel(),
        period_yr=per_state.ravel(),
        damping_ratio=damp_state.ravel(),
        cond_number=cond_state.ravel(),
        nn_gap=gap_state.ravel(),
        nn_relgap=relgap_state.ravel(),
        self_participation=part_own.ravel(),
        loop_shift=shift_state.ravel(),
    ))


def dominant_table(lam_state, lab, years, seeds):
    """Spectral abscissa and which compartment carries it."""
    re = lam_state.real
    k = re.argmax(axis=2)
    S, T = re.shape[:2]
    si, ti = np.meshgrid(np.arange(S), np.arange(T), indexing="ij")
    dom = re[si, ti, k]
    return pd.DataFrame(dict(
        seed=seeds[si].ravel(), year=years[ti].ravel(),
        dominant_mode=np.array(STATE)[k.ravel()],
        spectral_abscissa=dom.ravel(),
        dominant_timescale_yr=(1.0 / np.abs(dom)).ravel()))


def degeneracy_table(lam, cond, lab, years, seeds):
    """Per (seed, year): the WP-2c guard.  `min_relgap` small or `max_cond`
    large means Ch2 Eq. eigenvalue_sensitivity is not safe at that node."""
    g, rg = gaps(lam)
    order = np.argsort(-lam.real, axis=2)
    seq = np.take_along_axis(lab, order, axis=2)
    canonical = np.array([4, 3, 2, 5, 0, 1])          # slow -> fast, modal order
    swapped = ~(seq == canonical).all(axis=2)
    S, T = lam.shape[:2]
    si, ti = np.meshgrid(np.arange(S), np.arange(T), indexing="ij")
    return pd.DataFrame(dict(
        seed=seeds[si].ravel(), year=years[ti].ravel(),
        min_gap=g.min(axis=2).ravel(),
        min_relgap=rg.min(axis=2).ravel(),
        max_cond=cond.max(axis=2).ravel(),
        rank_order_swapped=swapped.ravel(),
        n_complex=(np.abs(lam.imag) > IM_TOL).sum(axis=2).ravel()))


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _band(ax, years, X, colour, label, lw=1.8, ls="-"):
    lo, med, hi = np.percentile(X, [25, 50, 75], axis=0)
    ax.fill_between(years, lo, hi, color=colour, alpha=0.22, lw=0)
    ax.plot(years, med, color=colour, lw=lw, ls=ls, label=label)
    return med


def fig_spectrum(lam_state, ts_state, shift_state, part_state, mu, years,
                 mask_test, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    t0 = float(years[np.asarray(mask_test, bool)][0])
    re = lam_state.real
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4))

    # (a) the six modes as decay rates
    ax = axes[0, 0]
    for k in range(6):
        _band(ax, years, -re[:, :, k], COL[k], STATE[k])
    for m in mu:
        ax.axhline(1.0 / m, color=GREY, lw=0.8, ls="--")
        ax.annotate(f"1/{m:.0f} yr", (years[1], 1.0 / m), fontsize=6,
                    color=GREY, va="bottom")
    ax.set_yscale("log")
    ax.set_ylabel(r"$-\mathrm{Re}\,\lambda_k(t)$   (yr$^{-1}$)")
    ax.set_title("(a) frozen-time spectrum — all six modes are real at every "
                 "seed and year", loc="left", fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    sec = ax.secondary_yaxis("right", functions=(lambda x: 1.0 / x,
                                                 lambda x: 1.0 / x))
    sec.set_ylabel("timescale  (yr)", fontsize=9)

    # (b) mode localisation.  P_ki = dlambda_k/da_ii (Ch2 Eq. asens), so this
    #     panel is simultaneously the modal decomposition's identifiability
    #     and the diagonal block of the WP-2c sensitivity.
    ax = axes[0, 1]
    M = np.median(part_state.real, axis=(0, 1))
    im = ax.imshow(np.abs(M), cmap="Greys",
                   norm=LogNorm(vmin=1e-4, vmax=1.0))
    for a in range(6):
        for b in range(6):
            if abs(M[a, b]) >= 5e-4:
                ax.text(b, a, f"{M[a, b]:.3f}", ha="center", va="center",
                        fontsize=6.5,
                        color="w" if abs(M[a, b]) > 0.1 else "k")
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels(STATE, rotation=60, ha="right", fontsize=7)
    ax.set_yticklabels(STATE, fontsize=7)
    ax.set_xlabel("compartment $i$", fontsize=8)
    ax.set_ylabel("mode $k$", fontsize=8)
    ax.set_title(r"(b) participation $P_{ki}=\partial\lambda_k/\partial a_{ii}$",
                 loc="left", fontsize=10)
    fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046, label=r"$|P_{ki}|$")

    # (c) the dominant mode against the slowest compartment
    ax = axes[1, 0]
    dom = ts_state[:, :, 4]
    _band(ax, years, dom, COL[4], "dominant mode  $1/|\\lambda_{\\max}|$")
    ax.axhline(mu[2], color="k", lw=1.2, ls="--",
               label=rf"slowest compartment $\mu_3$ = {mu[2]:.0f} yr")
    med = np.median(dom, axis=0)
    ax.annotate(f"+{100*(med[-1]/mu[2]-1):.1f}% in {years[-1]:.0f}",
                (years[-1], med[-1]), xytext=(-4, 6),
                textcoords="offset points", ha="right", fontsize=8)
    ax.set_ylabel("yr")
    ax.set_title("(c) the recycling loop makes the system slower than any of "
                 "its compartments", loc="left", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="upper left")

    # (d) how much of each mode is the loop rather than the compartment
    ax = axes[1, 1]
    for k in range(6):
        _band(ax, years, shift_state[:, :, k], COL[k], STATE[k])
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_ylabel(r"$\lambda_k - a_{kk}$   (yr$^{-1}$)")
    ax.set_title(r"(d) loop contribution; $S_{conc}$ is exactly $-\alpha_{cc}$",
                 loc="left", fontsize=10)

    for ax in (axes[0, 0], axes[1, 0], axes[1, 1]):
        ax.axvline(t0, color=GREY, lw=0.9, ls=":")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(lw=0.4, alpha=0.3)
        ax.set_xlabel("year")
    fig.suptitle(f"WP-2b  eigenstructure of the estimated A(t) — anchor_v4, "
                 f"{lam_state.shape[0]} seeds (median, IQR); "
                 f"dotted line = start of held-out window",
                 fontsize=11, x=0.01, ha="left")
    fig.legend(handles, labels, loc="upper center", ncol=6, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=300)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


def fig_plane(lam, cond, deg, eps_trend, years, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    # (a) the complex plane: everything on the real axis
    ax = axes[0]
    S, T = lam.shape[:2]
    yy = np.broadcast_to(years[None, :, None], lam.shape).ravel()
    sc = ax.scatter(lam.real.ravel(), lam.imag.ravel(), c=yy, s=9,
                    cmap="viridis", alpha=0.6, lw=0)
    re, im = CH2_SOA_PAIR
    ax.scatter([re, re], [im, -im], marker="*", s=210, color="#D55E00",
               zorder=5)
    ax.annotate("Ch. 2 copper SOA pair\n"
                r"$-0.06\pm0.334i$, period 18.8 yr",
                (re, im), xytext=(16, 14), textcoords="offset points",
                fontsize=7.5, color="#D55E00")
    ax.annotate(f"all {S*T*6} estimated eigenvalues\nlie on the real axis",
                xy=(-0.3, 0.0), xycoords="data",
                xytext=(0.97, 0.10), textcoords="axes fraction",
                fontsize=7.5, ha="right",
                arrowprops=dict(arrowstyle="->", lw=0.8, color=GREY))
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axvline(0.0, color="k", lw=0.8)
    ax.set_xscale("symlog", linthresh=1e-2)
    ax.set_xlabel(r"$\mathrm{Re}\,\lambda$   (yr$^{-1}$, symlog)")
    ax.set_ylabel(r"$\mathrm{Im}\,\lambda$   (yr$^{-1}$)")
    ax.set_ylim(-0.45, 0.45)
    ax.set_title(f"(a) spectrum in the complex plane — no complex pair",
                 loc="left", fontsize=10)
    fig.colorbar(sc, ax=ax, pad=0.02, label="year")

    # (b) the WP-2c guard
    ax = axes[1]
    yrs = np.sort(deg.year.unique())
    for col, colour, lab in (("min_relgap", COL[0], "min relative gap"),
                             ("max_cond", COL[2],
                              r"max cond. $1/|q_k\cdot v_k|$")):
        vals = np.stack([deg.loc[deg.year == y, col].values for y in yrs])
        _band(ax, yrs, vals.T, colour, lab)
    swap = deg.groupby("year").rank_order_swapped.sum()
    ax.scatter(swap.index[swap > 0], [1.0] * int((swap > 0).sum()),
               marker="v", color="k", s=26, zorder=5,
               label="years where the two fastest\nmodes swap rank")
    ax.set_yscale("log")
    ax.set_xlabel("year")
    ax.set_title("(b) near-degeneracy — the WP-2c guard", loc="left",
                 fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="lower left")

    # (c) is the frozen-time reading legitimate?
    ax = axes[2]
    for k in range(6):
        _band(ax, years, eps_trend[:, :, k], COL[k], STATE[k])
    ax.axhline(1.0, color="k", lw=1.0, ls="--")
    ax.annotate(r"$\varepsilon=1$: mode moves as fast as it decays",
                (years[1], 1.0), xytext=(0, 5), textcoords="offset points",
                fontsize=7)
    ax.set_yscale("log")
    ax.set_xlabel("year")
    ax.set_ylabel(r"$\varepsilon_k=|\dot\lambda_k|/\lambda_k^2$")
    ax.set_title("(c) adiabaticity of the frozen-time reading (cubic trend)",
                 loc="left", fontsize=10)
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="lower left")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(lw=0.4, alpha=0.3)
    fig.suptitle("WP-2b  the spectrum is real, well separated except upstream, "
                 "and slow enough to read at frozen time — except for the "
                 "in-use modes", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=300)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="WP-2b: eigenstructure of A(t)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--npz", default=A_NPZ)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--random-draws", type=int, default=200000)
    args = ap.parse_args(argv)

    import zinc_A_lab as lab
    lab.check(verbose=True)
    if args.check:
        return 0

    d, A, md5 = load_A(args.npz)
    years = np.asarray(d["years"], float)
    seeds = np.asarray(d["seeds"], int)
    mu = np.asarray(d["mu_cohorts"], float)
    S, T = A.shape[:2]
    print(f"\nloaded {os.path.basename(args.npz)}  md5 {md5}")
    print(f"  A {A.shape}  seeds {S}  years {years[0]:.0f}-{years[-1]:.0f}")
    inv = sanity(A)
    print("  invariants: " + "  ".join(f"{k}={v:.3e}" for k, v in inv.items()))

    lam, V, Q, P, cond, lab_h, lab_g = eigensystem(A)
    agree = float((lab_h == lab_g).all(axis=2).mean())
    print(f"  mode labelling: Hungarian == greedy argmax at "
          f"{100*agree:.1f}% of {S*T} nodes")
    own = np.take_along_axis(np.abs(P), lab_h[..., None], axis=3)[..., 0]
    print(f"  mode localisation: min self-participation {own.min():.3f}, "
          f"median {np.median(own):.3f} — modes map 1:1 onto compartments")

    # --- the required deliverables ------------------------------------------
    lam_sorted = np.take_along_axis(lam, np.argsort(-lam.real, axis=2), axis=2)
    lam_state = by_state(lam, lab_h)
    cond_state = by_state(cond, lab_h)
    part_state = np.zeros_like(P)                  # rows re-indexed to STATE
    np.put_along_axis(part_state, lab_h[..., None], P, axis=2)
    part_own = np.diagonal(part_state, axis1=2, axis2=3).real
    ts_state, per_state, damp_state = derived(lam_state)
    g, rg = gaps(lam)
    gap_state, relgap_state = by_state(g, lab_h), by_state(rg, lab_h)
    diag = np.diagonal(A, axis1=2, axis2=3)
    shift_state = lam_state.real - diag                  # loop contribution
    eps_raw, eps_trend = adiabaticity(lam_state, years)

    n_complex = int((np.abs(lam.imag) > IM_TOL).sum())
    print("\n--- complex pairs ---")
    print(f"  eigenvalues with |Im| > {IM_TOL:g}: {n_complex} of {S*T*6}"
          f"   (max |Im| = {np.abs(lam.imag).max():.3e} /yr)")

    # --- tables --------------------------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    summ = summary_table(lam_state, ts_state, per_state, years, seeds)
    ps = per_seed_table(lam_state, ts_state, per_state, damp_state, cond_state,
                        gap_state, relgap_state, part_own, shift_state,
                        years, seeds)
    dom = dominant_table(lam_state, lab_h, years, seeds)
    deg = degeneracy_table(lam, cond, lab_h, years, seeds)
    adia = pd.DataFrame([dict(
        mode=STATE[k],
        eps_raw_median=float(np.median(eps_raw[:, :, k])),
        eps_raw_p95=float(np.percentile(eps_raw[:, :, k], 95)),
        eps_trend_median=float(np.median(eps_trend[:, :, k])),
        eps_trend_p95=float(np.percentile(eps_trend[:, :, k], 95)),
        frac_nodes_eps_raw_gt1=float((eps_raw[:, :, k] > 1).mean()),
    ) for k in range(6)])

    print("\n--- single-coefficient distance to an oscillatory regime ---")
    osc = oscillation_scan(A)
    print(f"  {len(osc)} structural entries scanned over "
          f"g in [{osc.g_min.iloc[0]:.0e}, {osc.g_max.iloc[0]:.0e}] at every "
          f"one of {S*T} nodes")
    print(f"  entries reaching a complex pair anywhere: "
          f"{int(osc.reaches_complex.sum())} of {len(osc)}")
    for _, r in osc[osc.reaches_complex].iterrows():
        print(f"    A[{r.row},{r.col}] {r.to_stock:<8s}<- {r.from_stock:<11s}"
              f" needs x{r.crit_up_min:.0f} at best, x{r.crit_up_median:.0f} "
              f"typically; mass balance caps it at "
              f"x{r.max_feasible_multiplier:.1f}")
    topo = topology_admits_complex(A, n=args.random_draws)
    print(f"  control — random Metzler matrices with the same sparsity: "
          f"{100*topo['frac_complex']:.2f}% complex "
          f"({topo['n_draws']} draws, max |Im| {topo['max_abs_im']:.1f}/yr)")

    summ.to_csv(os.path.join(args.out, "wp2b_eigen_summary.csv"), index=False)
    ps.to_csv(os.path.join(args.out, "wp2b_modes_per_seed.csv"), index=False)
    dom.to_csv(os.path.join(args.out, "wp2b_dominant.csv"), index=False)
    deg.to_csv(os.path.join(args.out, "wp2b_degeneracy.csv"), index=False)
    adia.to_csv(os.path.join(args.out, "wp2b_adiabaticity.csv"), index=False)
    osc.to_csv(os.path.join(args.out, "wp2b_oscillation_scan.csv"), index=False)

    npz_path = os.path.join(args.out, "wp2b_eigen.npz")
    np.savez_compressed(
        npz_path, years=years, seeds=seeds, mu_cohorts=mu,
        state_names=np.array(STATE, object),
        lam_sorted=lam_sorted, lam_state=lam_state,
        eigvec_right=V, eigvec_left=Q, participation=part_state,
        mode_label=lab_h, cond_number=cond_state,
        timescale=ts_state, period=per_state, damping=damp_state,
        nn_gap=gap_state, nn_relgap=relgap_state, loop_shift=shift_state,
        eps_raw=eps_raw, eps_trend=eps_trend,
        source_md5=np.array(md5), ch2_soa_pair=np.array(CH2_SOA_PAIR),
        topology_control=np.array(json.dumps(topo)))

    fig_spectrum(lam_state, ts_state, shift_state, part_state, mu, years,
                 d["mask_test"], os.path.join(args.out, "wp2b_spectrum.png"))
    fig_plane(lam, cond, deg, eps_trend, years,
              os.path.join(args.out, "wp2b_complex_plane.png"))

    # --- what the numbers say ------------------------------------------------
    pd.set_option("display.width", 200, "display.max_columns", 40)
    print("\n--- sorted spectrum (seed median, yr^-1) ---")
    for ti in (0, T // 2, T - 1):
        v = np.median(lam_sorted[:, ti].real, axis=0)
        print(f"  {years[ti]:.0f}  " + "  ".join(f"{x:9.5f}" for x in v))

    print("\n--- modal timescales 1/|Re lambda|, seed median [IQR] (yr) ---")
    for k in range(6):
        a = np.percentile(ts_state[:, 0, k], [50, 25, 75])
        b = np.percentile(ts_state[:, -1, k], [50, 25, 75])
        print(f"  {STATE[k]:12s} {years[0]:.0f} {a[0]:8.3f} "
              f"[{a[1]:.3f},{a[2]:.3f}]   {years[-1]:.0f} {b[0]:8.3f} "
              f"[{b[1]:.3f},{b[2]:.3f}]")

    dm = dom.groupby("dominant_mode").size()
    print(f"\n--- dominant mode ---")
    print(f"  carried by: " + ", ".join(f"{k} {v}/{S*T}" for k, v in dm.items()))
    d0 = np.percentile(ts_state[:, 0, 4], [50, 25, 75])
    d1 = np.percentile(ts_state[:, -1, 4], [50, 25, 75])
    ch = np.percentile(ts_state[:, -1, 4] - ts_state[:, 0, 4], [50, 25, 75])
    print(f"  dominant timescale {years[0]:.0f}: {d0[0]:.2f} yr "
          f"[{d0[1]:.2f},{d0[2]:.2f}]   {years[-1]:.0f}: {d1[0]:.2f} yr "
          f"[{d1[1]:.2f},{d1[2]:.2f}]")
    print(f"  change over the window: {ch[0]:+.2f} yr "
          f"[{ch[1]:+.2f},{ch[2]:+.2f}]")
    print(f"  vs slowest compartment mu = {mu[2]:.0f} yr: "
          f"{100*(d0[0]/mu[2]-1):+.1f}% in {years[0]:.0f}, "
          f"{100*(d1[0]/mu[2]-1):+.1f}% in {years[-1]:.0f}")
    # Seed-paired contrasts, HL rather than a significance test (spec §2).
    hl_rows = []
    for name, v in (
            (f"dominant timescale {years[-1]:.0f} - {years[0]:.0f} (yr)",
             ts_state[:, -1, 4] - ts_state[:, 0, 4]),
            (f"dominant timescale {years[0]:.0f} - mu_3 (yr)",
             ts_state[:, 0, 4] - mu[2]),
            (f"dominant timescale {years[-1]:.0f} - mu_3 (yr)",
             ts_state[:, -1, 4] - mu[2])):
        h = hodges_lehmann(v)
        hl_rows.append(dict(contrast=name, **h))
        print(f"  HL {name}: {h['hl']:+.3f} "
              f"[{h['lo']:+.3f}, {h['hi']:+.3f}]  (n={h['n']})")
    pd.DataFrame(hl_rows).to_csv(
        os.path.join(args.out, "wp2b_dominant_hl.csv"), index=False)
    ab0 = np.percentile(dom.loc[dom.year == years[0], "spectral_abscissa"],
                        [50, 25, 75])
    print(f"  spectral abscissa {years[0]:.0f}: {ab0[0]:.5f} /yr "
          f"[{ab0[1]:.5f},{ab0[2]:.5f}]  (all Re lambda < 0 at "
          f"{100*(lam.real.max(axis=2) < 0).mean():.0f}% of nodes)")

    # Cross-check against WP-2a's 1-norm certificate: the frozen spectral
    # abscissa should be at least as negative as -eta.  # Ch2 Eq. uniform_column_condition
    mc = os.path.join(args.out, "wp2a_mass_conservation.csv")
    if os.path.exists(mc):
        eta = pd.read_csv(mc).set_index("seed")["eta_uniform"]
        worst = dom.groupby("seed").spectral_abscissa.max()
        slack = (-worst / eta.reindex(worst.index)).dropna()
        print(f"  WP-2a 1-norm certificate eta = "
              f"{np.median(eta):.4f} /yr; worst-year spectral abscissa "
              f"{np.median(worst):.4f} /yr")
        print(f"  the frozen spectrum decays {np.median(slack):.2f}x "
              f"[{np.percentile(slack, 25):.2f},{np.percentile(slack, 75):.2f}]"
              f" faster than the column-sum bound guarantees; the bound holds "
              f"for {int((slack >= 1).sum())}/{len(slack)} seeds")

    print("\n--- loop contribution lambda_k - a_kk, seed median (yr^-1) ---")
    for k in range(6):
        print(f"  {STATE[k]:12s} {np.median(shift_state[:, 0, k]):+.5f}   "
              f"{np.median(shift_state[:, -1, k]):+.5f}")
    print(f"  trace preserved: max |sum_k (lambda_k - a_kk)| = "
          f"{np.abs(shift_state.sum(axis=2)).max():.2e}")

    print("\n--- near-degeneracy (WP-2c guard) ---")
    print(f"  min relative gap over all nodes: {deg.min_relgap.min():.4f} "
          f"(median {deg.min_relgap.median():.3f})")
    print(f"  max condition number: {deg.max_cond.max():.1f} "
          f"(median {deg.max_cond.median():.3f})")
    sw = deg[deg.rank_order_swapped]
    if len(sw):
        print(f"  rank order of the two fastest modes swaps at {len(sw)} of "
              f"{S*T} nodes: seeds {sorted(sw.seed.unique().tolist())}, "
              f"years {sorted(sw.year.astype(int).unique().tolist())}")
    bad = deg[deg.max_cond > 10]
    where = sorted(set(ps.loc[ps.cond_number > 10, "mode"])) or ["none"]
    print(f"  nodes with condition number > 10: {len(bad)} "
          f"({100*len(bad)/len(deg):.1f}%) — in the "
          f"{'/'.join(where)} modes")

    print("\n--- adiabaticity of the frozen-time reading ---")
    print(adia.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nwrote {npz_path}, wp2b_eigen_summary.csv, "
          f"wp2b_modes_per_seed.csv, wp2b_dominant.csv, wp2b_dominant_hl.csv, "
          f"wp2b_degeneracy.csv, wp2b_adiabaticity.csv, "
          f"wp2b_oscillation_scan.csv, wp2b_spectrum.{{png,pdf}}, "
          f"wp2b_complex_plane.{{png,pdf}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
