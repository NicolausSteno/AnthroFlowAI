#!/usr/bin/env python3
"""
run_wp2a.py — WP-2a: assemble the time-varying transfer matrix A(t)
===================================================================

Stacks the per-seed dumps written by `zinc_A_lab.py` into the deliverable
`analysis/wp2a_A_of_t.npz` — `A` of shape (n_seeds, n_years, 6, 6) — and
reports the mass-conservation verification the spec asks for.

The object                                   # Ch2 Eq. system
                                             # Ch2 Eq. a1, Eq. vecb
    dS/dt = A(t) S + b(t),
    S = [S_conc, S_ref, S_iu_short, S_iu_med, S_iu_long, S_scrap],
    b = [cp(t), 0, 0, 0, 0, 0],

is Chapter 2's `A` restricted to the five stocks this dataset observes, with
the in-use stock resolved into its three Rostek lifetime cohorts.  Chapter 2
poses the non-autonomous formulation and has no estimated A(t) to evaluate it
on; this is that object.

Checks reported per seed:

  * **mass conservation** — `1^T A(t) + l(t)^T = 0`, where `l` is assembled
    independently from the six loss flows.  Absolute and relative residual;
  * **the RHS identity** — `A(t) S + b(t)` against `zinc_colloc_v5.make_rhs`
    at the ODE's own state and at the observed state with a randomised
    cohort split;
  * **the Metzler property** — off-diagonals >= 0, which Chapter 2 assumes in
    deriving Eq. uniform_column_condition;
  * **the uniform column condition** itself,          # Ch2 Eq. uniform_column_condition
    `max_j sum_i a_ij(t) <= -eta < 0`, giving an empirical uniform
    exponential-stability margin `eta` for the fitted non-autonomous system.

    python run_wp2a.py --check
    python run_wp2a.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DUMP_DIR = os.path.join(HERE, "analysis", "wp2a")
ANCHOR_DIR = os.path.join(HERE, "anchor_v4")
OUT_DIR = os.path.join(HERE, "analysis")

STATE = ["S_conc", "S_ref", "S_iu_short", "S_iu_med", "S_iu_long", "S_scrap"]

# Okabe–Ito, colourblind-safe.
COL = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00", "#56B4E9"]
GREY = "#555555"


def entry_label(i, j):
    """Symbolic content of A[i, j], with the Chapter 2 coefficient it maps to."""
    lab = {
        (0, 0): "-alpha_cc                       [Ch2 -(a1+a15)]",
        (1, 0): "(1-tau_ref)*alpha_cc            [Ch2 a1]",
        (1, 1): "-alpha_refc                     [Ch2 -(a8+a10+a17)]",
        (1, 5): "tau_waelz*alpha_win             [Ch2 a14]",
        (5, 1): "newscrap*alpha_refc             [Ch2 a10]",
        (5, 5): "newscrap*alpha_dr-alpha_win-alpha_dr  [Ch2 -(a13+a14+a19)]",
    }
    if (i, j) in lab:
        return lab[(i, j)]
    if 2 <= i <= 4 and j == 1:
        return f"f_cohort[{i-2}]*inflow*alpha_refc  [Ch2 a8]"
    if 2 <= i <= 4 and j == 5:
        return f"f_cohort[{i-2}]*inflow*alpha_dr    [Ch2 a13 pass-through]"
    if 2 <= i <= 4 and i == j:
        return f"-1/mu[{i-2}]                       [Ch2 -(a9+a18)]"
    if i == 5 and 2 <= j <= 4:
        return f"tau_olds/mu[{j-2}]                 [Ch2 a9]"
    return ""


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_dumps(dump_dir=DUMP_DIR):
    paths = sorted(glob.glob(os.path.join(dump_dir, "A_seed*.npz")),
                   key=lambda p: int(os.path.basename(p)[6:-4]))
    if not paths:
        raise SystemExit(f"no dumps in {dump_dir} — run zinc_A_lab.py first")
    return [(int(os.path.basename(p)[6:-4]), np.load(p, allow_pickle=True))
            for p in paths]


def verify_reproduction(dumps):
    """Each refit must reproduce the stored anchor_v4 free-run exactly; a
    silent divergence would mean A(t) belongs to a different model than the
    published one.  Same check as WP-1a."""
    rows = []
    for seed, d in dumps:
        stored = os.path.join(ANCHOR_DIR, f"pred_seed{seed}.npz")
        diff = np.nan
        if os.path.exists(stored):
            st = np.load(stored, allow_pickle=True)
            diff = float(np.nanmax(np.abs(d["S_pred"] - st["S_pred_B"])))
        rows.append(dict(seed=seed, S_pred_maxabsdiff=diff))
    return pd.DataFrame(rows)


def verification_table(dumps):
    """One row per seed of everything `zinc_A_lab.verify` recorded."""
    rows = []
    for seed, d in dumps:
        v = json.loads(str(d["verify_json"]))
        v = {k: (v[k] if not isinstance(v[k], list) else
                 ";".join(f"{x:.0f}" for x in v[k])) for k in v}
        rows.append(dict(seed=seed, **v))
    df = pd.DataFrame(rows)
    front = ["seed", "max_abs_col_sum_residual", "max_rel_col_sum_residual",
             "max_abs_rhs_residual_pred", "max_rel_rhs_residual_pred",
             "max_abs_rhs_residual_obs", "max_rel_rhs_residual_obs",
             "min_offdiagonal", "max_diagonal", "eta_uniform"]
    return df[[c for c in front if c in df] +
              [c for c in df.columns if c not in front]]


# ---------------------------------------------------------------------------
# stacking
# ---------------------------------------------------------------------------
def stack(dumps):
    seeds = np.array([s for s, _ in dumps], int)
    d0 = dumps[0][1]
    years = np.asarray(d0["years"], float).ravel()
    for s, d in dumps[1:]:
        if not np.array_equal(np.asarray(d["years"], float).ravel(), years):
            raise AssertionError(f"seed {s} has a different year axis")

    per_seed_keys = ["A", "b", "A_ode", "b_ode", "loss_rate", "col_sum",
                     "col_sum_residual", "rhs_residual_pred", "rhs_residual_obs",
                     "alphas", "taus", "frac_fu", "frac_eu", "f_cohort", "cp",
                     "S_ode_pred", "S_pred"]
    out = {k: np.stack([d[k] for _, d in dumps]) for k in per_seed_keys}
    out["seeds"] = seeds
    out["years"] = years
    for k in ("S_obs", "mu_cohorts", "ode_state_names", "stock_names",
              "alpha_names", "tau_binary_names", "mask_train", "mask_val",
              "mask_test"):
        out[k] = d0[k]
    return out


def entries_table(A, years, seeds):
    """Long-form seed median/IQR of every structurally non-zero entry."""
    rows = []
    nz = np.argwhere(np.any(np.abs(A) > 0, axis=(0, 1)))
    for i, j in nz:
        lab = entry_label(int(i), int(j))
        for ti, y in enumerate(years):
            v = A[:, ti, i, j]
            q1, q3 = np.percentile(v, [25, 75])
            rows.append(dict(row=int(i), col=int(j),
                             to_stock=STATE[int(i)], from_stock=STATE[int(j)],
                             entry=lab, year=float(y),
                             median=float(np.median(v)), q1=float(q1),
                             q3=float(q3), n_seeds=len(seeds)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _band(ax, years, X, colour, label, lw=1.8, ls="-"):
    lo, med, hi = np.percentile(X, [25, 50, 75], axis=0)
    ax.fill_between(years, lo, hi, color=colour, alpha=0.22, lw=0)
    ax.plot(years, med, color=colour, lw=lw, ls=ls, label=label)
    return med


def fig_A(st, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A, years = st["A"], st["years"]
    t0 = float(years[np.asarray(st["mask_test"], bool)][0])
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharex=True)

    # (a) diagonal: per-stock total outflow rate = 1 / residence time
    ax = axes[0, 0]
    for k in range(6):
        _band(ax, years, -A[:, :, k, k], COL[k],
              f"{STATE[k]}  ({1.0/abs(np.median(A[:, :, k, k])):.2g} yr)")
    ax.set_yscale("log")
    ax.set_ylabel(r"$-a_{ii}(t)$   (yr$^{-1}$)")
    ax.set_title("(a) diagonal — total outflow rate per unit stock",
                 loc="left", fontsize=10)
    ax.legend(fontsize=7, frameon=False, ncol=2)

    # (b) the two Chapter 2 circularity routes off S_scrap, plus the two
    #     forward routes, all as seed medians with IQR
    ax = axes[0, 1]
    _band(ax, years, A[:, :, 1, 5], COL[2],
          r"$s_5\!\to\! s_2$  $\tau_{waelz}\alpha_{win}$   [Ch2 $\alpha_{14}$]")
    _band(ax, years, A[:, :, 2:5, 5].sum(axis=2), COL[3],
          r"$s_5\!\to\!$ use  $\alpha_{dr}$ route   [Ch2 $\alpha_{13}$]")
    _band(ax, years, A[:, :, 5, 1], COL[1],
          r"$s_2\!\to\! s_5$  new scrap", ls="--")
    _band(ax, years, A[:, :, 5, 2:5].sum(axis=2), COL[4],
          r"use $\to s_5$  $\tau_{olds}/\mu$", ls="--")
    ax.set_yscale("log")
    ax.set_ylabel(r"$a_{ij}(t)$   (yr$^{-1}$)")
    ax.set_title("(b) old-scrap re-entry routes — the Chapter 2 circularity "
                 "coefficients", loc="left", fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    # (c) column sums = -loss rate, and the Ch2 uniform column condition
    ax = axes[1, 0]
    cs = st["col_sum"]
    for k in range(6):
        _band(ax, years, cs[:, :, k], COL[k], STATE[k])
    worst = cs.max(axis=2)                       # max_j over columns, per seed/yr
    eta = -np.median(worst, axis=0)
    ax.plot(years, -eta, color="k", lw=1.4, ls=":",
            label=r"$\max_j\sum_i a_{ij}$ (median seed)")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_ylabel(r"$\sum_i a_{ij}(t)$   (yr$^{-1}$)")
    ax.set_title("(c) column sums = −loss rate  "
                 "(Ch2 Eq. uniform_column_condition)", loc="left", fontsize=10)
    ax.legend(fontsize=7, frameon=False, ncol=2)

    # (d) mass-conservation residual, per seed
    ax = axes[1, 1]
    res = np.abs(st["col_sum_residual"]).max(axis=2)          # (S, T)
    for s in range(res.shape[0]):
        ax.plot(years, np.maximum(res[s], 1e-20), color=GREY, lw=0.6, alpha=0.5)
    ax.set_yscale("log")
    ax.set_ylim(1e-20, 1e-10)
    ax.set_ylabel(r"$\max_j |\,\sum_i a_{ij} + \ell_j\,|$")
    ax.set_title("(d) mass-conservation residual, all seeds "
                 "(double precision floor)", loc="left", fontsize=10)

    for ax in axes.ravel():
        ax.axvline(t0, color=GREY, lw=0.9, ls=":")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(lw=0.4, alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("year")
    n = st["A"].shape[0]
    fig.suptitle(f"WP-2a  empirical time-varying transfer matrix A(t) — "
                 f"anchor_v4, {n} seeds (median, IQR); "
                 f"dotted line = start of held-out window",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=300)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


def fig_structure(st, path):
    """Median A at the first, split-boundary and last year, plus the change."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    A, years = st["A"], st["years"]
    mt = np.asarray(st["mask_test"], bool)
    idx = [0, int(np.where(mt)[0][0]), len(years) - 1]
    med = np.median(A, axis=0)
    vmax = float(np.abs(med).max())
    # Structural zeros are masked (grey) so the topology reads separately from
    # entries that are merely small.
    struct = np.any(np.abs(A) > 0, axis=(0, 1))
    med = np.where(struct, med, np.nan)

    fig, axes = plt.subplots(1, 4, figsize=(15.0, 4.6),
                             gridspec_kw=dict(wspace=0.30))
    norm = SymLogNorm(linthresh=1e-3, vmin=-vmax, vmax=vmax, base=10)
    for ax, ti in zip(axes[:3], idx):
        im = ax.imshow(med[ti], cmap="RdBu_r", norm=norm)
        ax.set_title(f"median A({years[ti]:.0f})", loc="left", fontsize=10)
    fig.colorbar(im, ax=axes[:3].tolist(), orientation="horizontal",
                 fraction=0.055, pad=0.28, aspect=55,
                 label=r"$a_{ij}$  (yr$^{-1}$), symlog")

    ax = axes[3]
    base = med[idx[0]]
    rel = np.where(np.abs(base) > 0, med[idx[2]] / np.where(base != 0, base, 1.0),
                   np.nan)
    im2 = ax.imshow(np.log10(np.abs(rel)), cmap="PuOr_r", vmin=-1, vmax=1)
    ax.set_title(f"$\\log_{{10}}|A({years[idx[2]]:.0f})/A({years[idx[0]]:.0f})|$",
                 loc="left", fontsize=10)
    fig.colorbar(im2, ax=ax, orientation="horizontal", fraction=0.055,
                 pad=0.28, aspect=12)

    for n, ax in enumerate(axes):
        ax.set_facecolor("#dedede")
        ax.set_xticks(range(6)); ax.set_yticks(range(6))
        ax.set_xticklabels(STATE, rotation=60, ha="right", fontsize=7)
        ax.set_yticklabels(STATE if n in (0, 3) else [""] * 6, fontsize=7)
        ax.set_xlabel("from", fontsize=8)
    axes[0].set_ylabel("to", fontsize=8)
    fig.suptitle("WP-2a  structure of the estimated A(t): sparsity is fixed by "
                 "the cycle topology, magnitudes are learned and time-varying",
                 fontsize=11, x=0.01, ha="left")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="WP-2a: assemble A(t)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dumps", default=DUMP_DIR)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args(argv)

    import zinc_A_lab as lab
    lab.check(verbose=True)
    if args.check:
        return 0

    dumps = load_dumps(args.dumps)
    print(f"\nloaded {len(dumps)} seed dumps from {args.dumps}")

    repro = verify_reproduction(dumps)
    print(f"reproduction vs stored anchor_v4: max |ΔS_pred| = "
          f"{np.nanmax(repro.S_pred_maxabsdiff):.3e} over {len(repro)} seeds "
          f"({int((repro.S_pred_maxabsdiff > 0).sum())} non-identical)")

    ver = verification_table(dumps)
    st = stack(dumps)
    A, years = st["A"], st["years"]

    os.makedirs(args.out, exist_ok=True)
    npz_path = os.path.join(args.out, "wp2a_A_of_t.npz")
    np.savez_compressed(npz_path, **st)
    ver.to_csv(os.path.join(args.out, "wp2a_mass_conservation.csv"), index=False)
    repro.to_csv(os.path.join(args.out, "wp2a_reproduction_check.csv"), index=False)
    ent = entries_table(A, years, st["seeds"])
    ent.to_csv(os.path.join(args.out, "wp2a_A_entries.csv"), index=False)

    fig_A(st, os.path.join(args.out, "wp2a_A_of_t.png"))
    fig_structure(st, os.path.join(args.out, "wp2a_A_structure.png"))

    pd.set_option("display.width", 200, "display.max_columns", 40)
    print(f"\nA(t) stacked: {A.shape}  (n_seeds, n_years, 6, 6), "
          f"years {years[0]:.0f}–{years[-1]:.0f}")
    print("\n--- verification, worst seed of each column ---")
    for c in ("max_abs_col_sum_residual", "max_rel_col_sum_residual",
              "max_abs_rhs_residual_pred", "max_rel_rhs_residual_pred",
              "max_abs_rhs_residual_obs", "max_rel_rhs_residual_obs",
              "max_abs_col_sum_residual_naive_at_obs",
              "max_abs_coef_state_dependence", "max_abs_A_minus_A_ode"):
        if c in ver:
            print(f"  {c:42s} {ver[c].max():.3e}")
    print(f"  {'min off-diagonal (Metzler >= 0)':42s} "
          f"{ver['min_offdiagonal'].min():.3e}")
    print(f"  {'max diagonal (must be < 0)':42s} {ver['max_diagonal'].max():.3e}")
    print(f"  {'A vs A_ode differing years':42s} "
          f"{sorted(set(ver['A_minus_A_ode_years'].astype(str)))}")

    q1, med, q3 = np.percentile(ver["eta_uniform"].values, [25, 50, 75])
    print(f"\n--- Ch2 Eq. uniform_column_condition ---")
    print(f"  eta = -max_t max_j sum_i a_ij(t) : median {med:.4f} /yr "
          f"(IQR {q1:.4f}–{q3:.4f}) over {len(ver)} seeds")
    print(f"  implied 1-norm decay time 1/eta  : {1.0/med:.1f} yr")
    print(f"  satisfied (eta > 0) for {int((ver['eta_uniform'] > 0).sum())}"
          f"/{len(ver)} seeds → uniform exponential stability in the 1-norm")
    # Which column attains the bound: the system's slowest mass-exit channel.
    cs = st["col_sum"]                                     # (S, T, 6)
    flat = np.argmax(cs.reshape(cs.shape[0], -1), axis=1)
    bind_t, bind_j = np.unravel_index(flat, cs.shape[1:])
    names, counts = np.unique([STATE[j] for j in bind_j], return_counts=True)
    print(f"  binding column (slowest loss channel): "
          + ", ".join(f"{n} {c}/{len(ver)} seeds"
                      for n, c in zip(names, counts)))
    print(f"  binding year: median {np.median(years[bind_t]):.0f} "
          f"(range {years[bind_t].min():.0f}–{years[bind_t].max():.0f})")

    print("\n--- loss rate out of the cycle per stock (yr^-1), seed median ---")
    ell = -cs
    for k in range(6):
        print(f"  {STATE[k]:12s} min over years {np.median(ell[:, :, k].min(axis=1)):8.4f}"
              f"   median {np.median(ell[:, :, k]):8.4f}"
              f"   max {np.median(ell[:, :, k].max(axis=1)):8.4f}")

    print("\n--- residence times 1/|a_ii|, seed median (yr) ---")
    for k in range(6):
        v = -A[:, :, k, k]
        print(f"  {STATE[k]:12s} {np.median(1.0/v):8.3f}   "
              f"(first year {1.0/np.median(v[:, 0]):7.3f}, "
              f"last {1.0/np.median(v[:, -1]):7.3f})")

    print(f"\nwrote {npz_path}, wp2a_mass_conservation.csv, "
          f"wp2a_A_entries.csv, wp2a_reproduction_check.csv, "
          f"wp2a_A_of_t.{{png,pdf}}, wp2a_A_structure.{{png,pdf}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
