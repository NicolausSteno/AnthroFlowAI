#!/usr/bin/env python3
"""Build the eight figures of Chapter 4 from the persisted analysis artefacts.

One script, one style, no titles.  Every panel is drawn from a file already on
disk under ``final_model/anchor_v4/``, ``final_model/anchor_gam/`` or
``final_model/analysis/``; nothing is synthesised and nothing is re-fitted here.

Usage
-----
    python3 make_ch4_figures.py --check            # verify inputs, draw nothing
    python3 make_ch4_figures.py                    # build all eight
    python3 make_ch4_figures.py --only 3 5         # build a subset
    python3 make_ch4_figures.py --out <directory>  # write somewhere else

Defaults: repository root is inferred from this file's location, inputs are read
from ``<root>/final_model``, output PDFs are written to the thesis project,
``<root>/final_model/PhD_Thesis/figures/ch4``, where chapters/ch4_ude_zinc.tex
includes them.  Labels use the chapter's notation (Section ``sec:ude_notation``).

Conventions, following the drafting brief:
  * no titles and no suptitles anywhere; panel labels only, ``a)``, ``b)``, …
  * vector PDF, full-width 6.3 in, base font 9 pt, serif with Computer Modern
    mathematics, so text renders at body size when included at ``\\linewidth``
  * Okabe-Ito palette, with linestyle or marker varied so colour is never the
    only channel carrying information
  * axis labels carry units, in British English
  * frameless legends inside the axes, no heavy gridlines.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

HERE = Path(__file__).resolve()
# .../final_model/reproducibility/figures/make_ch4_figures.py
FINAL_MODEL = HERE.parents[2]
ANCHOR = FINAL_MODEL / "anchor_v4"
# The published comparator: zinc_baseline run with every smooth basis empty,
# so it is a cross-validated ridge regression on the link scale. The directory
# it was written to is named anchor_gam, hence the path constant's name.
GAM = FINAL_MODEL / "anchor_gam"
ANALYSIS = FINAL_MODEL / "analysis"
DEFAULT_OUT = FINAL_MODEL / "PhD_Thesis" / "figures" / "ch4"

N_SEEDS = 35

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #

# Okabe-Ito.  Yellow (#F0E442) is omitted from the line palette: it is
# illegible on white at 0.9 pt.
OI = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#666666",
}
CYCLE = [OI["blue"], OI["vermillion"], OI["green"], OI["orange"], OI["purple"], OI["sky"]]

FULL = 6.3
HALF = 3.1
# A full-page landscape figure, for `sidewaysfigure`: the book is A4 with 30 mm
# margins, so a rotated float has the text height, 9.33 in, across it and the
# text width, 5.91 in, down it.  Authoring at 9.3 in wide renders it at close to
# 1:1, which keeps the 9 pt base font at body size.
LAND = 9.3
# Height is left short of the 5.91 in text width so that the caption fits inside
# the rotated float alongside the graphic.
LAND_H = 5.0

RC = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "legend.handlelength": 1.9,
    "legend.borderaxespad": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "lines.linewidth": 1.1,
    "grid.linewidth": 0.4,
    "grid.color": "#D9D9D9",
    "grid.alpha": 0.8,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "pdf.fonttype": 42,
    "pdf.compression": 6,
}

# Pre-registered split, read off analysis/wp2a_A_of_t.npz's own masks.
TRAIN_END = 2001.0
VAL_END = 2007.0
YEAR_MIN, YEAR_MAX = 1980.0, 2019.0

STOCKS = ["Concentrate", "Refined", "In-Use", "Scrap"]
# Display names; STOCKS are the data keys and must stay as they are.
STOCK_DISPLAY = {"Concentrate": "Concentrate", "Refined": "Refined",
                 "In-Use": "In-use", "Scrap": "Scrap"}
STATES = ["S_conc", "S_ref", "S_iu_short", "S_iu_med", "S_iu_long", "S_scrap"]
STATE_LABEL = {
    "S_conc": "Concentrate",
    "S_ref": "Refined",
    "S_iu_short": "In-use, 10 yr",
    "S_iu_med": "In-use, 20 yr",
    "S_iu_long": "In-use, 44 yr",
    "S_scrap": "Scrap",
}
# Chapter 4 is also read as a standalone paper, so the transfer coefficients
# carry mnemonic subscripts rather than the positional indices of Chapters 2-3
# (alpha_1, alpha_2/alpha_8, alpha_13, alpha_14 respectively).  The subscript
# names the compartment the coefficient draws from.  Junction shares are written
# k (the MFA transfer-coefficient symbol), so tau is reserved for tau_FD.
ALPHA_SHORT = {
    "alpha_cc": r"$\alpha_{\mathrm{conc}}$",
    "alpha_refc": r"$\alpha_{\mathrm{ref}}$",
    "alpha_win": r"$\alpha_{\mathrm{waelz}}$",
    "alpha_dr": r"$\alpha_{\mathrm{reuse}}$",
}
ALPHA_NAME = {
    "alpha_cc": "concentrate consumption",
    "alpha_refc": "refined consumption",
    "alpha_win": "Waelz input",
    "alpha_dr": "direct reuse",
}
ALPHA_LABEL = {k: f"{ALPHA_SHORT[k]}, {ALPHA_NAME[k]}" for k in ALPHA_SHORT}
ALPHA_ORDER = ["alpha_cc", "alpha_refc", "alpha_win", "alpha_dr"]
ALPHA_COLOUR = {
    "alpha_cc": OI["blue"],
    "alpha_refc": OI["green"],
    "alpha_win": OI["vermillion"],
    "alpha_dr": OI["purple"],
}
ALPHA_DASH = {
    "alpha_cc": (0, ()),
    "alpha_refc": (0, (4, 1.5)),
    "alpha_win": (0, (1, 1.2)),
    "alpha_dr": (0, (5, 1.4, 1, 1.4)),
}


def prettify(name: str) -> str:
    """Flow and driver identifiers into prose labels."""
    special = {
        "waelz_input": "Waelz input",
        "waelz_recycling": "Waelz recycling",
        "waelz_losses": "Waelz losses",
        "inuse_inflow": "in-use inflow",
        "inuse_stock": "in-use stock",
        "eol_collection": "end-of-life collection",
        "eol_composition": "end-of-life composition",
        "end_of_life": "end-of-life flow",
        "concentrate_production": "concentrate production",
        "first_use_new_scrap": "first-use new scrap",
        "first_use_losses": "first-use losses",
        "end_use_new_scrap": "end-use new scrap",
        "end_use_losses": "end-use losses",
        "old_scrap_recovery": "end-of-life collection",
        "direct_reuse_recycling": "direct-reuse recycling",
    }
    if name in special:
        return special[name]
    return name.replace("_", " ")


def panel_label(ax, text: str, dx: float = 0.0, dy: float = 1.0) -> None:
    ax.text(
        dx, dy, text, transform=ax.transAxes, fontsize=10, fontweight="bold",
        va="bottom", ha="left",
    )


def shade_windows(ax, val=True, test=True, label=False) -> None:
    """Light shading on the validation and held-out windows."""
    if val:
        ax.axvspan(TRAIN_END, VAL_END, color="#000000", alpha=0.045, lw=0, zorder=0)
    if test:
        ax.axvspan(VAL_END, YEAR_MAX, color="#000000", alpha=0.10, lw=0, zorder=0)
    if label:
        ax.text(
            (VAL_END + YEAR_MAX) / 2, 0.965, "held out", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=7, color=OI["grey"],
        )


def light_grid(ax, axis: str = "y") -> None:
    ax.grid(True, axis=axis, lw=0.4, color="#D9D9D9", alpha=0.9)
    ax.set_axisbelow(True)


def save(fig, out_dir: Path, name: str) -> Path:
    path = out_dir / name
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return path


def median_iqr(a: np.ndarray, axis: int = 0):
    return (
        np.nanmedian(a, axis=axis),
        np.nanpercentile(a, 25, axis=axis),
        np.nanpercentile(a, 75, axis=axis),
    )


# --------------------------------------------------------------------------- #
# Input registry — what --check verifies
# --------------------------------------------------------------------------- #
# Each entry is (path, required npz keys or None, the panels that need it).

REQUIREMENTS = {
    "1": [
        (ANCHOR / "pred_seed0.npz", ("years_all", "stocks_obs", "S_pred_B"), "a-d"),
        (GAM / "pred_seed0.npz", ("years_all", "S_pred_B"), "a-d"),
    ],
    "1flowsa": [
        (ANCHOR / "pred_seed0.npz",
         ("years_flow", "flows_obs", "F_pred_B", "flow_names"), "a-h"),
        (GAM / "pred_seed0.npz", ("years_flow", "F_pred_B", "flow_names"), "a-h"),
    ],
    "1flowsb": [
        (ANCHOR / "pred_seed0.npz",
         ("years_flow", "flows_obs", "F_pred_B", "flow_names"), "a-h"),
        (GAM / "pred_seed0.npz", ("years_flow", "F_pred_B", "flow_names"), "a-h"),
    ],
    "1coefficients": [
        (ANCHOR / "pred_seed0.npz",
         ("years_flow", "flows_obs", "F_pred_B", "flow_names", "stocks_obs",
          "S_pred_B"), "a-i"),
        (GAM / "pred_seed0.npz", ("years_flow", "F_pred_B", "S_pred_B"), "a-i"),
    ],
    "2": [
        (ANCHOR / "per_stock.csv", None, "a"),
        (ANCHOR / "per_stock_vs_persistence.csv", None, "a"),
        (GAM / "per_stock.csv", None, "a"),
        (ANALYSIS / "wp1f_flow_error.csv", None, "b"),
        (ANALYSIS / "wp1f_ar_orders.csv", None, "b"),
        (GAM / "per_flow.csv", None, "b"),
        (ANCHOR / "per_flow.csv", None, "b"),
        (ANALYSIS / "wp1f_mass_imbalance.csv", None, "c"),
        (ANCHOR / "pred_seed0.npz", ("years_all", "S_pred_B"), "c"),
    ],
    "3": [
        (ANALYSIS / "wp8h_coefficients.csv", None, "a"),
        (ANALYSIS / "wp1c_variance_decomposition.csv", None, "b"),
        (ANALYSIS / "wp1a_alpha_channels.csv", None, "b"),
        (ANALYSIS / "wp1b_factor_vs_product.csv", None, "c"),
        (ANALYSIS / "wp4c_spectrum.csv", None, "d"),
        (ANALYSIS / "wp4c_spectrum_summary.csv", None, "d"),
    ],
    "4": [
        (ANALYSIS / "wp2b_eigen.npz", ("years", "timescale", "state_names"), "a"),
        (ANALYSIS / "wp2g_saturation.csv", None, "b, d"),
        (ANALYSIS / "wp2g_amplification.csv", None, "c"),
    ],
    "5": [
        (ANALYSIS / "wp8h_indicator_uncertainty_nonauto.csv", None, "a-c"),
    ],
    "6": [
        (ANALYSIS / "wp2c_sensitivity_summary.csv", None, "a"),
        (ANALYSIS / "wp2d_elasticities.csv", None, "a"),
        (ANALYSIS / "wp2e_vs_wp2c_rank.csv", None, "a"),
        (ANALYSIS / "wp2e_indicator_sensitivity.npz",
         ("param_names", "state_names", "term_direct_use_entry",
          "term_propagated_use_entry"), "b"),
        (ANALYSIS / "wp8d_permutation.csv", None, "c"),
    ],
    "7": [
        (ANALYSIS / "wp7_effects_by_year.csv", None, "a"),
        (ANALYSIS / "wp7_time_to_effect.csv", None, "a"),
        (ANALYSIS / "wp8b_memory_length.csv", None, "b"),
    ],
    "8": [
        (ANALYSIS / "wp6b_coarsening.csv", None, "a"),
        (ANALYSIS / "wp11g_observables.csv", None, "b"),
        (ANALYSIS / "wp11c_voi_headline.csv", None, "b"),
    ],
}


def check_inputs(figures) -> int:
    """Verify every input exists; every npz must carry the keys a panel needs."""
    missing = 0
    for fig_id in figures:
        print(f"Figure {fig_id}")
        for path, keys, panels in REQUIREMENTS[fig_id]:
            rel = path.relative_to(FINAL_MODEL)
            if not path.exists():
                print(f"  MISSING  {rel}   (panel {panels})")
                missing += 1
                continue
            note = ""
            if keys is not None:
                with np.load(path, allow_pickle=True) as d:
                    absent = [k for k in keys if k not in d.files]
                if absent:
                    print(f"  MISSING KEYS {rel}: {', '.join(absent)}   (panel {panels})")
                    missing += 1
                    continue
                note = f"  [{len(keys)} keys]"
            print(f"  ok       {rel}{note}   (panel {panels})")
    # The 35 per-seed prediction files are checked as a set rather than listed.
    seeds = sorted(ANCHOR.glob("pred_seed*.npz"))
    print(f"\nper-seed predictions: {len(seeds)} of {N_SEEDS} present in "
          f"{ANCHOR.relative_to(FINAL_MODEL)}")
    # Only Figures 1 (all four plates) and 2 read the per-seed predictions.
    needs_seeds = {"1", "1flowsa", "1flowsb", "1coefficients", "2"}
    if len(seeds) != N_SEEDS and needs_seeds & {str(f) for f in figures}:
        missing += 1
    print(f"\n{'FAILED' if missing else 'PASSED'}: {missing} problem(s)")
    return missing


# --------------------------------------------------------------------------- #
# Shared loaders
# --------------------------------------------------------------------------- #

def _seed_files():
    files = sorted(ANCHOR.glob("pred_seed*.npz"),
                   key=lambda p: int(p.stem.replace("pred_seed", "")))
    if not files:
        raise FileNotFoundError(f"no pred_seed*.npz under {ANCHOR}")
    return files


def load_ensemble_stocks():
    """Observed stocks, the 35-seed free-run predictions and the ridge regression's."""
    files = _seed_files()
    with np.load(files[0], allow_pickle=True) as d0:
        years = d0["years_all"]
        obs = d0["stocks_obs"]
    pred = np.stack([np.load(f, allow_pickle=True)["S_pred_B"] for f in files])
    with np.load(GAM / "pred_seed0.npz", allow_pickle=True) as dg:
        gam = dg["S_pred_B"]
    return years, obs, pred, gam


def load_ensemble_flows():
    """Observed flows, the 35-seed free-run predictions and the ridge regression's.

    Flow rows are period integrals: row k covers the interval between
    ``years_all[k]`` and ``years_all[k + 1]``, labelled by its closing year in
    ``years_flow``.
    """
    files = _seed_files()
    with np.load(files[0], allow_pickle=True) as d0:
        years = d0["years_flow"]
        obs = d0["flows_obs"]
        names = [str(x) for x in d0["flow_names"]]
    pred = np.stack([np.load(f, allow_pickle=True)["F_pred_B"] for f in files])
    with np.load(GAM / "pred_seed0.npz", allow_pickle=True) as dg:
        gam = dg["F_pred_B"]
    return years, obs, pred, gam, names


# The coefficients as they are assembled in ``zinc_colloc_v5``: every flow is a
# coefficient times a parent stock, so each coefficient is recoverable from the
# flow vector; the four transfer coefficients additionally need the parent
# stock too.  Applying the same operator to the observed record, to each seed and
# to the ridge regression gives the like-for-like comparison its own coefficient paths
# would give, since they are not persisted anywhere.  Verified against the
# stored instantaneous coefficients in ``analysis/wp2a_A_of_t.npz``: the median
# relative gap over 1980-2019 is under 1 per cent on thirteen of the fourteen
# and 2.5 per cent on tau_diss, whose pinned step is smeared by the annual
# integration.
# Every coefficient the flow record can recover.  `pin_*` in
# anchor_v4/config_used.json fixes tau_ref, tau_waelz, tau_diss, frac_fu_loss and
# frac_eu_loss to a schedule, so on those the three series coincide by
# construction and there is nothing to compare; they are recovered by
# `implied_coefficients` for the validation in the notes but are not plotted.
COEF_RECOVERABLE = [
    "alpha_cc", "alpha_refc", "alpha_win", "alpha_dr",
    "tau_ref", "tau_waelz", "tau_olds", "tau_diss",
    "frac_fu_new", "frac_fu_loss", "frac_fu_out",
    "frac_eu_new", "frac_eu_loss", "frac_eu_into",
]
COEF_PINNED = {"tau_ref", "tau_waelz", "tau_diss", "frac_fu_loss", "frac_eu_loss"}
COEF_ORDER = [c for c in COEF_RECOVERABLE if c not in COEF_PINNED]
COEF_LABEL = {
    **{k: ALPHA_LABEL[k] for k in ALPHA_ORDER},
    "tau_ref": r"$k_{\mathrm{ref}}$, refinery loss share",
    "tau_waelz": r"$k_{\mathrm{waelz}}$, Waelz yield",
    "tau_olds": r"$k_{\mathrm{olds}}$, end-of-life collection share",
    "tau_diss": r"$k_{\mathrm{diss}}$, dissipative share",
    "frac_fu_new": r"$\sigma_{\mathrm{fu}}$, first-use new-scrap share",
    "frac_fu_loss": r"$\lambda_{\mathrm{fu}}$, first-use loss share",
    "frac_fu_out": r"$\nu_{\mathrm{fu}}$, first-use pass-through share",
    "frac_eu_new": r"$\sigma_{\mathrm{eu}}$, end-use new-scrap share",
    "frac_eu_loss": r"$\lambda_{\mathrm{eu}}$, end-use loss share",
    "frac_eu_into": r"$\nu_{\mathrm{eu}}$, end-use pass-through share",
}
COEF_UNIT = {k: (r"yr$^{-1}$" if k.startswith("alpha") else "share")
             for k in COEF_RECOVERABLE}
# Two short lines, because the label is rotated into a panel about 1.6 in high:
# the descriptive name, then the symbol with its unit.
COEF_AXIS = {
    **{k: f"{ALPHA_NAME[k]}\n" + ALPHA_SHORT[k] + r" (yr$^{-1}$)"
       for k in ALPHA_ORDER},
    "tau_olds": "end-of-life collection\n" + r"$k_{\mathrm{olds}}$ (share)",
    "frac_fu_new": "first-use new scrap\n" + r"$\sigma_{\mathrm{fu}}$ (share)",
    "frac_fu_out": "first-use pass-through\n" + r"$\nu_{\mathrm{fu}}$ (share)",
    "frac_eu_new": "end-use new scrap\n" + r"$\sigma_{\mathrm{eu}}$ (share)",
    "frac_eu_into": "end-use pass-through\n" + r"$\nu_{\mathrm{eu}}$ (share)",
}


def implied_coefficients(flows, stocks, names):
    """Recover the fourteen flow-recoverable coefficients from one trajectory.

    ``flows`` is (n_interval, 18) of period integrals and ``stocks`` is
    (n_interval + 1, 4) of point-in-time stocks; the parent stock is taken on
    the trapezoid exposure over the interval, which is the operator the
    estimation target uses.
    """
    f = {n: flows[:, names.index(n)] for n in names}
    sbar = 0.5 * (stocks[:-1, :] + stocks[1:, :])
    parent = f["refined_consumption"] + f["direct_reuse_recycling"]
    eu_den = (f["total_products_into_use"] + f["end_use_new_scrap"]
              + f["end_use_losses"])
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {
            "alpha_cc": f["concentrate_consumption"] / sbar[:, 0],
            "alpha_refc": f["refined_consumption"] / sbar[:, 1],
            "alpha_win": f["waelz_input"] / sbar[:, 3],
            "alpha_dr": f["direct_reuse_recycling"] / sbar[:, 3],
            "tau_ref": f["refinery_losses"] / f["concentrate_consumption"],
            "tau_waelz": f["waelz_recycling"] / f["waelz_input"],
            "tau_olds": f["old_scrap_recovery"] / f["end_of_life"],
            "tau_diss": f["dissipative_use"] / f["total_products_into_use"],
            "frac_fu_new": f["first_use_new_scrap"] / parent,
            "frac_fu_loss": f["first_use_losses"] / parent,
            "frac_fu_out": 1.0 - (f["first_use_new_scrap"]
                                  + f["first_use_losses"]) / parent,
            "frac_eu_new": f["end_use_new_scrap"] / eu_den,
            "frac_eu_loss": f["end_use_losses"] / eu_den,
            "frac_eu_into": f["total_products_into_use"] / eu_den,
        }
    return out


# --------------------------------------------------------------------------- #
# Figure 1 — reproduction of the four stocks
# --------------------------------------------------------------------------- #

def figure1(out_dir: Path) -> Path:
    years, obs, pred, gam = load_ensemble_stocks()
    med, q1, q3 = median_iqr(pred, axis=0)

    fig = plt.figure(figsize=(FULL, 4.5), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig)
    labels = ["a)", "b)", "c)", "d)"]

    axes = []
    for k, stock in enumerate(STOCKS):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        axes.append(ax)
        # The in-use stock is two orders of magnitude above the others, so it is
        # drawn in megatonnes; every panel carries its own unit.
        scale, unit = (1e3, "Mt") if stock == "In-Use" else (1.0, "kt")
        shade_windows(ax, label=(k == 0))
        ax.fill_between(years, q1[:, k] / scale, q3[:, k] / scale, color=OI["blue"],
                        alpha=0.22, lw=0)
        ax.plot(years, med[:, k] / scale, color=OI["blue"], lw=1.2)
        ax.plot(years, gam[:, k] / scale, color=OI["vermillion"], lw=1.0,
                ls=(0, (4, 1.5)))
        ax.plot(years, obs[:, k] / scale, color=OI["black"], lw=0.0, marker="o",
                ms=2.4, mfc="none", mew=0.6)
        ax.set_xlim(YEAR_MIN - 0.8, YEAR_MAX + 0.8)
        ax.margins(y=0.10)
        ax.set_ylabel(f"{STOCK_DISPLAY[stock]} stock ({unit})")
        if k >= 2:
            ax.set_xlabel("Year")
        light_grid(ax)
        panel_label(ax, labels[k])

    handles = [
        Line2D([], [], color=OI["black"], lw=0, marker="o", ms=3.2, mfc="none",
               mew=0.7, label="Observed"),
        Line2D([], [], color=OI["blue"], lw=1.2, label="UDE, 35-seed median"),
        Patch(facecolor=OI["blue"], alpha=0.22, label="UDE, interquartile range"),
        Line2D([], [], color=OI["vermillion"], lw=1.0, ls=(0, (4, 1.5)),
               label="Ridge regression"),
    ]
    # The in-use panel is the only one with room for a legend inside the axes.
    axes[2].legend(handles=handles, loc="upper left", ncol=1)
    return save(fig, out_dir, "ch4_fig1_stocks.pdf")


# --------------------------------------------------------------------------- #
# Figure 1, flows — every flow in the cycle
# --------------------------------------------------------------------------- #

# Sixteen of the eighteen flows, grouped by stage of the cycle and split across
# two landscape plates of eight.  `concentrate_production` is pinned to the
# ILZSG series and `refinery_losses` is `tau_ref` times concentrate consumption
# with `tau_ref` pinned, so neither carries anything the other panels do not.
FLOW_ORDER_A = [
    "concentrate_consumption", "primary_refining", "refined_consumption",
    "first_use_new_scrap", "first_use_losses", "total_products_into_use",
    "end_use_new_scrap", "end_use_losses",
]
FLOW_ORDER_B = [
    "dissipative_use", "inuse_inflow", "end_of_life", "old_scrap_recovery",
    "waelz_input", "waelz_recycling", "waelz_losses", "direct_reuse_recycling",
]

SERIES_LEGEND = [
    Line2D([], [], color=OI["black"], lw=0, marker="o", ms=3.6, mfc="none",
           mew=0.7, label="Observed"),
    Line2D([], [], color=OI["blue"], lw=1.3, label="UDE, 35-seed median"),
    Patch(facecolor=OI["blue"], alpha=0.22, label="UDE, interquartile range"),
    Line2D([], [], color=OI["vermillion"], lw=1.1, ls=(0, (4, 1.5)),
           label="Ridge regression"),
]


def _flow_plate(out_dir: Path, order, filename: str) -> Path:
    """One landscape plate of eight flow panels, 2 rows by 4 columns."""
    years, obs, pred, gam, names = load_ensemble_flows()
    med, q1, q3 = median_iqr(pred, axis=0)

    ncol, nrow = 4, 2
    fig = plt.figure(figsize=(LAND, LAND_H), constrained_layout=True)
    gs = gridspec.GridSpec(nrow, ncol, figure=fig)

    for k, flow in enumerate(order):
        j = names.index(flow)
        ax = fig.add_subplot(gs[k // ncol, k % ncol])
        shade_windows(ax)
        ax.fill_between(years, q1[:, j], q3[:, j], color=OI["blue"], alpha=0.22, lw=0)
        ax.plot(years, med[:, j], color=OI["blue"], lw=1.2)
        ax.plot(years, gam[:, j], color=OI["vermillion"], lw=1.0, ls=(0, (4, 1.5)))
        ax.plot(years, obs[:, j], color=OI["black"], lw=0.0, marker="o", ms=2.4,
                mfc="none", mew=0.6)
        ax.set_xlim(YEAR_MIN - 0.8, YEAR_MAX + 0.8)
        ax.set_xticks([1980, 1990, 2000, 2010, 2019])
        ax.margins(y=0.12)
        ax.tick_params(labelsize=8)
        ax.set_ylabel(f"{prettify(flow)}\n(kt yr$^{{-1}}$)", fontsize=9)
        if k >= len(order) - ncol:
            ax.set_xlabel("Year")
        light_grid(ax)
        panel_label(ax, f"{chr(ord('a') + k)})")

    fig.legend(handles=SERIES_LEGEND, loc="outside lower center", ncol=4)
    return save(fig, out_dir, filename)


def figure1_flows_a(out_dir: Path) -> Path:
    return _flow_plate(out_dir, FLOW_ORDER_A, "ch4_fig1_flows_a.pdf")


def figure1_flows_b(out_dir: Path) -> Path:
    return _flow_plate(out_dir, FLOW_ORDER_B, "ch4_fig1_flows_b.pdf")


# --------------------------------------------------------------------------- #
# Figure 1, coefficients — every flow-recoverable coefficient
# --------------------------------------------------------------------------- #

def figure1_coefficients(out_dir: Path) -> Path:
    years, obs_f, pred_f, gam_f, names = load_ensemble_flows()
    _, obs_s, pred_s, gam_s = load_ensemble_stocks()
    # Each coefficient is an interval average, so it is placed at the midpoint
    # of the interval rather than at either endpoint.
    mid_years = years - 0.5

    obs_c = implied_coefficients(obs_f, obs_s, names)
    gam_c = implied_coefficients(gam_f, gam_s, names)
    seed_c = [implied_coefficients(pred_f[i], pred_s[i], names)
              for i in range(pred_f.shape[0])]

    ncol, nrow = 3, 3
    fig = plt.figure(figsize=(LAND, LAND_H + 0.4), constrained_layout=True)
    gs = gridspec.GridSpec(nrow, ncol, figure=fig)
    letters = [f"{chr(ord('a') + i)})" for i in range(len(COEF_ORDER))]

    for k, name in enumerate(COEF_ORDER):
        ax = fig.add_subplot(gs[k // ncol, k % ncol])
        stack = np.stack([c[name] for c in seed_c])
        med, q1, q3 = median_iqr(stack, axis=0)
        shade_windows(ax)
        ax.fill_between(mid_years, q1, q3, color=OI["blue"], alpha=0.22, lw=0)
        ax.plot(mid_years, med, color=OI["blue"], lw=1.2)
        ax.plot(mid_years, gam_c[name], color=OI["vermillion"], lw=1.0,
                ls=(0, (4, 1.5)))
        ax.plot(mid_years, obs_c[name], color=OI["black"], lw=0.0, marker="o",
                ms=2.4, mfc="none", mew=0.6)
        ax.set_xlim(YEAR_MIN - 0.8, YEAR_MAX + 0.8)
        ax.set_xticks([1980, 1990, 2000, 2010, 2019])
        ax.margins(y=0.12)
        ax.tick_params(labelsize=8)
        if name in ("alpha_win", "alpha_dr"):
            # The reported alpha_reuse and alpha_waelz leave the plotted range
            # entirely in 2018-19, when the reconstructed Scrap stock in their
            # shared denominator falls to 500 kt; a logarithmic axis keeps both
            # the level and that excursion legible.  The other two need none.
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())
        ax.set_ylabel(COEF_AXIS[name], fontsize=8.5)
        if k >= len(COEF_ORDER) - ncol:
            ax.set_xlabel("Year")
        light_grid(ax)
        panel_label(ax, letters[k])

    handles = list(SERIES_LEGEND)
    handles[0] = Line2D([], [], color=OI["black"], lw=0, marker="o", ms=3.6,
                        mfc="none", mew=0.7, label="Reported record")
    fig.legend(handles=handles, loc="outside lower center", ncol=4)
    return save(fig, out_dir, "ch4_fig1_coefficients.pdf")


# --------------------------------------------------------------------------- #
# Figure 2 — accuracy and mass consistency
# --------------------------------------------------------------------------- #

def figure2(out_dir: Path) -> Path:
    per_stock = pd.read_csv(ANCHOR / "per_stock.csv")
    per_stock = per_stock[per_stock.rollout == "freerun"]
    gam_stock = pd.read_csv(GAM / "per_stock.csv")
    gam_stock = gam_stock[gam_stock.rollout == "freerun"]
    persist = pd.read_csv(ANCHOR / "per_stock_vs_persistence.csv")

    ude_flow = pd.read_csv(ANALYSIS / "wp1f_flow_error.csv")
    ar_flow = pd.read_csv(ANALYSIS / "wp1f_ar_orders.csv")
    gam_flow = pd.read_csv(GAM / "per_flow.csv")
    imbalance = pd.read_csv(ANALYSIS / "wp1f_mass_imbalance.csv")

    fig = plt.figure(figsize=(FULL, 6.6), constrained_layout=True)
    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[0.95, 1.30, 1.05])

    # a) per-stock held-out error, seed distribution, ridge and persistence markers
    ax = fig.add_subplot(gs[0])
    data = [per_stock.loc[per_stock.component == s, "relRMSE"].to_numpy() for s in STOCKS]
    bp = ax.boxplot(data, positions=np.arange(len(STOCKS)), widths=0.42,
                    patch_artist=True, medianprops=dict(color=OI["black"], lw=1.0),
                    whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6),
                    flierprops=dict(marker="o", ms=2.0, mfc="none",
                                    mec=OI["grey"], mew=0.5))
    for box in bp["boxes"]:
        box.set(facecolor=OI["blue"], alpha=0.28, lw=0.6, edgecolor=OI["blue"])
    for j, s in enumerate(STOCKS):
        g = gam_stock.loc[gam_stock.component == s, "relRMSE"]
        p = persist.loc[persist.stock == s, "persistence"]
        if len(g):
            ax.plot(j, float(g.iloc[0]), marker="D", ms=4.5, color=OI["vermillion"],
                    mfc="none", mew=1.1, zorder=5)
        if len(p):
            ax.plot(j, float(p.iloc[0]), marker="^", ms=5.0, color=OI["green"],
                    mfc="none", mew=1.1, zorder=5)
    ax.set_xticks(np.arange(len(STOCKS)))
    ax.set_xticklabels([STOCK_DISPLAY[s] for s in STOCKS])
    ax.set_ylabel("Held-out error (per cent)")
    ax.set_xlim(-0.6, len(STOCKS) - 0.4)
    light_grid(ax)
    handles = [
        Patch(facecolor=OI["blue"], alpha=0.28, edgecolor=OI["blue"],
              label="UDE, 35 seeds"),
        Line2D([], [], lw=0, marker="D", ms=4.5, color=OI["vermillion"], mfc="none",
               mew=1.1, label="Ridge regression"),
        Line2D([], [], lw=0, marker="^", ms=5.0, color=OI["green"], mfc="none",
               mew=1.1, label="Persistence"),
    ]
    ax.legend(handles=handles, loc="upper left", ncol=3)
    panel_label(ax, "a)")

    # b) per-flow held-out error, ranked, UDE against the per-flow ARX ensemble
    ax = fig.add_subplot(gs[1])
    # Concentrate production is pinned to the ILZSG series in both models, so its
    # error is a numerical residual rather than a forecast; it is left out here.
    ude_flow = ude_flow[ude_flow.flow != "concentrate_production"]
    ude = ude_flow.groupby("flow").test_relRMSE.agg(
        median="median",
        q1=lambda x: np.percentile(x, 25),
        q3=lambda x: np.percentile(x, 75),
    )
    best_spec = ar_flow.groupby("spec").test_relRMSE.mean().idxmin()
    ar = ar_flow[ar_flow.spec == best_spec].set_index("flow").test_relRMSE
    # The ridge regression is deterministic and is scored on the same rollout as
    # the UDE column of wp1f; its component names carry the pinned-flow tag.
    gf = gam_flow[gam_flow.rollout == "testrun"].copy()
    gf["flow"] = gf.component.str.replace(" [pinned]", "", regex=False)
    gam_err = gf[~gf.component.str.startswith("<")].set_index("flow").relRMSE
    order = ude.sort_values("median").index.tolist()

    # Family statistics, on the same window and on one consistent basis: the
    # per-seed mean over the seventeen scored flows, which is what per_flow.csv
    # stores as <flow-mean> and which excludes the pinned concentrate production
    # series. The same seventeen are then split into the three Waelz flows and
    # the fourteen others, because the family advantage is that one channel and
    # a reader who sees only the aggregate cannot tell.
    anchor_flow = pd.read_csv(ANCHOR / "per_flow.csv")
    anchor_flow = anchor_flow[anchor_flow.rollout == "testrun"]
    ude_family = anchor_flow.loc[anchor_flow.component == "<flow-mean>", "relRMSE"]
    gam_family = float(gf.loc[gf.component == "<flow-mean>", "relRMSE"].iloc[0])
    ar_family = float(ar_flow[ar_flow.spec == best_spec].test_relRMSE.mean())

    waelz = [f for f in order if "waelz" in f]
    rest = [f for f in order if "waelz" not in f]
    ude_wide = ude_flow.pivot(index="seed", columns="flow", values="test_relRMSE")
    ude_split = {"rest": ude_wide[rest].mean(axis=1).to_numpy(),
                 "waelz": ude_wide[waelz].mean(axis=1).to_numpy()}
    gam_split = {"rest": float(gam_err.reindex(rest).mean()),
                 "waelz": float(gam_err.reindex(waelz).mean())}
    ar_split = {"rest": float(ar.reindex(rest).mean()),
                "waelz": float(ar.reindex(waelz).mean())}
    x = np.arange(len(order))
    lo = (ude.loc[order, "median"] - ude.loc[order, "q1"]).to_numpy()
    hi = (ude.loc[order, "q3"] - ude.loc[order, "median"]).to_numpy()
    ax.errorbar(x - 0.14, ude.loc[order, "median"], yerr=np.vstack([lo, hi]),
                fmt="o", ms=3.4, color=OI["blue"], lw=0, elinewidth=0.8,
                capsize=1.8, label="UDE, median and interquartile range")
    ax.plot(x + 0.16, gam_err.reindex(order).to_numpy(), lw=0, marker="D", ms=3.2,
            color=OI["vermillion"], mfc="none", mew=1.0, label="Ridge regression")
    ax.plot(x + 0.34, ar.reindex(order).to_numpy(), lw=0, marker="s", ms=3.2,
            color=OI["purple"], mfc="none", mew=1.0,
            label=f"Per-flow autoregressive ensemble ({best_spec})")
    # The per-flow ranking and the flow-family mean point in opposite
    # directions, so the family statistic is drawn in its own slot at the right
    # rather than left for the reader to aggregate by eye: the UDE's family
    # advantage is the Waelz channel and nothing else.
    fam_x = len(order) + 0.9
    fam_ude = ude_family.to_numpy()
    ax.errorbar(fam_x - 0.14, np.median(fam_ude),
                yerr=[[np.median(fam_ude) - np.percentile(fam_ude, 25)],
                      [np.percentile(fam_ude, 75) - np.median(fam_ude)]],
                fmt="o", ms=4.4, color=OI["blue"], lw=0, elinewidth=0.9, capsize=2.0)
    ax.plot(fam_x + 0.16, gam_family, lw=0, marker="D", ms=4.2,
            color=OI["vermillion"], mfc="none", mew=1.2)
    ax.plot(fam_x + 0.34, ar_family, lw=0, marker="s", ms=4.2,
            color=OI["purple"], mfc="none", mew=1.2)
    for k, key in enumerate(("rest", "waelz"), start=1):
        sx = fam_x + 1.6 * k
        v = ude_split[key]
        ax.errorbar(sx - 0.14, np.median(v),
                    yerr=[[np.median(v) - np.percentile(v, 25)],
                          [np.percentile(v, 75) - np.median(v)]],
                    fmt="o", ms=4.4, color=OI["blue"], lw=0, elinewidth=0.9,
                    capsize=2.0)
        ax.plot(sx + 0.16, gam_split[key], lw=0, marker="D", ms=4.2,
                color=OI["vermillion"], mfc="none", mew=1.2)
        ax.plot(sx + 0.34, ar_split[key], lw=0, marker="s", ms=4.2,
                color=OI["purple"], mfc="none", mew=1.2)
    ax.axvline(len(order) - 0.35, color=OI["grey"], lw=0.6, ls=":")

    fam_ticks = [fam_x, fam_x + 1.6, fam_x + 3.2]
    fam_labels = ["all seventeen flows", "excluding the Waelz route",
                  "the Waelz route only"]
    ax.set_yscale("log")
    ax.set_xticks(list(x) + fam_ticks)
    ax.set_xticklabels([prettify(f) for f in order] + fam_labels,
                       rotation=42, ha="right", fontsize=7)
    for lab in ax.get_xticklabels()[-3:]:
        lab.set_fontweight("bold")
    ax.set_ylabel("Held-out error (per cent)")
    ax.set_xlim(-0.7, fam_ticks[-1] + 0.9)
    ax.set_ylim(1.6, 400)
    light_grid(ax)
    ax.legend(loc="upper left", ncol=1)
    panel_label(ax, "b)")

    # c) implied Refined stock under the autoregressive ensemble against the UDE
    ax = fig.add_subplot(gs[2])
    ref = imbalance[imbalance.stock == "Refined"]
    ar_colours = [OI["orange"], OI["purple"], OI["sky"], OI["green"]]
    ar_dashes = [(0, (4, 1.5)), (0, (1, 1.2)), (0, (5, 1.4, 1, 1.4)), (0, (3, 1, 1, 1))]
    for j, model in enumerate(sorted(ref.model.unique())):
        sub = ref[ref.model == model].sort_values("year")
        ax.plot(sub.year, sub.implied_stock_kt, color=ar_colours[j % 4],
                ls=ar_dashes[j % 4], lw=1.0, label=model)
    years, obs, pred, _ = load_ensemble_stocks()
    med = np.median(pred, axis=0)
    sel = years >= ref.year.min()
    ax.plot(years[sel], med[sel, 1], color=OI["blue"], lw=1.3, label="UDE, 35-seed median")
    ax.plot(years[sel], obs[sel, 1], color=OI["black"], lw=0, marker="o", ms=2.6,
            mfc="none", mew=0.6, label="Observed")
    ax.axhline(0.0, color=OI["grey"], lw=0.6, ls=":")
    ax.set_yscale("symlog", linthresh=1e3, linscale=0.45)
    ax.set_yticks([-3e4, -1e4, -1e3, 0, 1e3, 1e4])
    ax.set_yticklabels([r"$-3\times10^{4}$", r"$-10^{4}$", r"$-10^{3}$", "0",
                        r"$10^{3}$", r"$10^{4}$"])
    ax.set_yticks([], minor=True)
    ax.set_ylim(-5e4, 6e5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Implied refined stock (kt)")
    ax.set_xlim(ref.year.min() - 0.3, YEAR_MAX + 0.3)
    light_grid(ax)
    ax.legend(loc="upper left", ncol=2)
    panel_label(ax, "c)")

    return save(fig, out_dir, "ch4_fig2_accuracy.pdf")


# --------------------------------------------------------------------------- #
# Figure 3 — estimated coefficients and their identifiability
# --------------------------------------------------------------------------- #

def figure3(out_dir: Path) -> Path:
    coef = pd.read_csv(ANALYSIS / "wp8h_coefficients.csv")
    vdec = pd.read_csv(ANALYSIS / "wp1c_variance_decomposition.csv")
    chan = pd.read_csv(ANALYSIS / "wp1a_alpha_channels.csv")
    fvp = pd.read_csv(ANALYSIS / "wp1b_factor_vs_product.csv")
    spec = pd.read_csv(ANALYSIS / "wp4c_spectrum.csv")
    spec_sum = pd.read_csv(ANALYSIS / "wp4c_spectrum_summary.csv")

    fig = plt.figure(figsize=(FULL, 5.2), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # a) the four transfer coefficients with pooled 95 per cent bands
    ax = fig.add_subplot(gs[0, 0])
    shade_windows(ax, label=True)
    for name in ALPHA_ORDER:
        sub = coef[coef.coefficient == name].sort_values("year")
        ax.fill_between(sub.year, sub.lo95, sub.hi95, color=ALPHA_COLOUR[name],
                        alpha=0.16, lw=0)
        ax.plot(sub.year, sub.value, color=ALPHA_COLOUR[name], ls=ALPHA_DASH[name],
                lw=1.1, label=ALPHA_LABEL[name])
    # Labelled at the right-hand end rather than in a legend: the four
    # trajectories cover the whole panel and leave no room for a legend box.
    for name in ALPHA_ORDER:
        sub = coef[coef.coefficient == name].sort_values("year")
        ax.text(YEAR_MAX + 0.9, float(sub.value.iloc[-1]), ALPHA_SHORT[name],
                color=ALPHA_COLOUR[name], fontsize=8, va="center", ha="left")
    ax.set_yscale("log")
    ax.set_xlim(YEAR_MIN - 0.5, YEAR_MAX + 13.0)
    ax.set_xticks([1980, 1990, 2000, 2010])
    ax.set_xlabel("Year")
    ax.set_ylabel(r"Transfer coefficient (yr$^{-1}$)")
    light_grid(ax)
    panel_label(ax, "a)")

    # b) held-out coefficient error, split into bias and variance
    ax = fig.add_subplot(gs[0, 1])
    v = vdec[(vdec.split == "test") & (vdec.space == "level") & (vdec.stage == "B")]
    v = v.set_index("channel").reindex(ALPHA_ORDER)
    err = chan[chan.split == "test"].set_index("channel").reindex(ALPHA_ORDER)
    x = np.arange(len(ALPHA_ORDER))
    bias = v.bias_share.to_numpy() * 100.0
    var = (1.0 - v.bias_share.to_numpy()) * 100.0
    ax.bar(x, bias, width=0.55, color=OI["vermillion"], alpha=0.85, lw=0,
           label="Bias")
    ax.bar(x, var, bottom=bias, width=0.55, color=OI["sky"], alpha=0.85, lw=0,
           label="Across-seed variance")
    for j, name in enumerate(ALPHA_ORDER):
        ax.text(j, 103, f"{err.loc[name, 'relRMSE_median']:.0f}%", ha="center",
                va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([ALPHA_SHORT[n] for n in ALPHA_ORDER])
    ax.set_ylim(0, 138)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Share of squared error (per cent)")
    light_grid(ax)
    ax.legend(loc="upper center", ncol=2)
    panel_label(ax, "b)")

    # c) the factor against the product it enters
    ax = fig.add_subplot(gs[1, 0])
    for name in ALPHA_ORDER:
        sub = fvp[fvp.channel == name]
        ax.plot(sub.alpha_relRMSE, sub.flow_relRMSE, lw=0, marker="o", ms=2.6,
                color=ALPHA_COLOUR[name], alpha=0.42, mew=0)
        ax.plot(sub.alpha_relRMSE.median(), sub.flow_relRMSE.median(), lw=0,
                marker="o", ms=6.5, color=ALPHA_COLOUR[name], mec=OI["black"],
                mew=0.7, label=ALPHA_SHORT[name], zorder=5)
    lim = [1.2, 220.0]
    ax.plot(lim, lim, color=OI["grey"], lw=0.7, ls=":")
    ax.text(25, 32, "1:1", fontsize=7, color=OI["grey"], rotation=40)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(1.2, 220.0)
    ax.set_xlabel("Coefficient error, the factor (per cent)")
    ax.set_ylabel("Flow error, the product (per cent)")
    light_grid(ax, axis="both")
    ax.legend(loc="upper left", ncol=2)
    panel_label(ax, "c)")

    # d) the curvature spectrum, its cumulative trace share and p_eff
    ax = fig.add_subplot(gs[1, 1])
    piv = spec.pivot_table(index="index", columns="seed", values="eig")
    med = piv.median(axis=1).to_numpy()
    lo = piv.quantile(0.25, axis=1).to_numpy()
    hi = piv.quantile(0.75, axis=1).to_numpy()
    idx = piv.index.to_numpy() + 1
    ax.fill_between(idx, lo, hi, color=OI["blue"], alpha=0.2, lw=0)
    ax.plot(idx, med, color=OI["blue"], lw=1.1, label="Eigenvalues of the curvature")
    ax.set_yscale("log")
    ax.set_xscale("log")
    # Below the smallest retained eigenvalue the spectrum is numerical null
    # space rather than curvature, so the axis stops there.
    eig_min = float(spec_sum.eig_min_retained.median())
    ax.set_ylim(eig_min * 0.2, float(spec_sum.eig_max.median()) * 4)
    ax.set_xlabel("Index")
    ax.set_ylabel(r"Eigenvalue of $J^{\mathsf{T}}J$")
    light_grid(ax, axis="both")

    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)
    cum = np.cumsum(med) / np.sum(med) * 100.0
    ax2.plot(idx, cum, color=OI["orange"], lw=1.1, ls=(0, (4, 1.5)),
             label="Cumulative share of the trace")
    ax2.set_ylabel("Cumulative trace share (per cent)")
    ax2.set_ylim(0, 105)

    n90 = float(spec_sum.n_dir_90.median())
    p_eff = float(spec_sum.p_eff_ref.median())
    ax.axvline(n90, color=OI["green"], lw=0.9, ls=(0, (1, 1.2)))
    ax.axvline(p_eff, color=OI["vermillion"], lw=0.9, ls=(0, (5, 1.4, 1, 1.4)))
    # All four elements go in one legend rather than in notes beside them: the
    # spectrum and the cumulative share cross the panel diagonally and left no
    # clear lane for in-place labels.
    handles = [
        Line2D([], [], color=OI["blue"], lw=1.1, label="Curvature spectrum"),
        Line2D([], [], color=OI["orange"], lw=1.1, ls=(0, (4, 1.5)),
               label="Cumulative trace share"),
        Line2D([], [], color=OI["green"], lw=0.9, ls=(0, (1, 1.2)),
               label=f"{n90:.0f} directions carry 90 per cent"),
        Line2D([], [], color=OI["vermillion"], lw=0.9, ls=(0, (5, 1.4, 1, 1.4)),
               label=r"$p_{\mathrm{eff}} = $" + f"{p_eff:.0f}"),
    ]
    ax.legend(handles=handles, loc="lower left", ncol=1, labelspacing=0.35,
              borderaxespad=0.2)
    panel_label(ax, "d)")

    return save(fig, out_dir, "ch4_fig3_coefficients.pdf")


# --------------------------------------------------------------------------- #
# Figure 4 — the estimated cycle as a dynamical system
# --------------------------------------------------------------------------- #

def figure4(out_dir: Path) -> Path:
    with np.load(ANALYSIS / "wp2b_eigen.npz", allow_pickle=True) as d:
        years = d["years"]
        timescale = d["timescale"]          # (seed, year, state)
        state_names = [str(s) for s in d["state_names"]]
    sat = pd.read_csv(ANALYSIS / "wp2g_saturation.csv")
    amp = pd.read_csv(ANALYSIS / "wp2g_amplification.csv")

    fig = plt.figure(figsize=(FULL, 7.0), constrained_layout=True)
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[0.92, 1.16, 0.92])

    # a) modal timescales by compartment
    ax = fig.add_subplot(gs[0, :])
    shade_windows(ax)
    for j, name in enumerate(state_names):
        med, q1, q3 = median_iqr(timescale[:, :, j], axis=0)
        ax.fill_between(years, q1, q3, color=CYCLE[j % len(CYCLE)], alpha=0.18, lw=0)
        ax.plot(years, med, color=CYCLE[j % len(CYCLE)], lw=1.0,
                ls=[(0, ()), (0, (4, 1.5)), (0, (1, 1.2)), (0, (5, 1.4, 1, 1.4)),
                    (0, (3, 1, 1, 1)), (0, (6, 1.6))][j % 6],
                label=STATE_LABEL.get(name, name))
    ax.set_yscale("log")
    ax.set_xlim(YEAR_MIN, YEAR_MAX)
    # A decade of headroom below the fastest compartment, so the six-entry
    # legend runs along the foot of the panel without crossing a curve.
    ax.set_ylim(0.0045, 90)
    ax.set_xlabel("Year")
    ax.set_ylabel("Modal timescale (yr)")
    light_grid(ax)
    ax.legend(loc="lower center", ncol=6, columnspacing=1.0, handlelength=1.5,
              borderaxespad=0.25)
    panel_label(ax, "a)")

    # b) the in-use stock against its own contemporaneous equilibrium
    ax = fig.add_subplot(gs[1, 0])
    iu = sat[sat.stock == "In-Use"].sort_values("year")
    shade_windows(ax)
    ax.fill_between(iu.year, iu.Sstar_q1 / 1e3, iu.Sstar_q3 / 1e3,
                    color=OI["vermillion"], alpha=0.18, lw=0)
    ax.plot(iu.year, iu.Sstar_median / 1e3, color=OI["vermillion"], lw=1.1,
            ls=(0, (4, 1.5)), label=r"Frozen equilibrium $S^{*}_{\mathrm{iu}}(t)$")
    ax.plot(iu.year, iu.S_obs / 1e3, color=OI["black"], lw=1.1, label="Observed stock")
    ax.set_xlim(YEAR_MIN, YEAR_MAX)
    ax.set_ylim(0, 620)
    ax.set_xlabel("Year")
    ax.set_ylabel("In-use zinc (Mt)")
    light_grid(ax)

    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)
    # WP-2g reports the saturation ratio as a Hodges-Lehmann estimate with an
    # interval, so the same statistic is plotted here rather than the median.
    ax2.fill_between(iu.year, iu.sat_ratio_obs_lo, iu.sat_ratio_obs_hi,
                     color=OI["blue"], alpha=0.14, lw=0)
    ax2.plot(iu.year, iu.sat_ratio_obs_hl, color=OI["blue"], lw=1.0,
             ls=(0, (1, 1.2)), label="Saturation ratio")
    ax2.set_ylabel("Saturation ratio")
    ax2.set_ylim(0, 1.0)
    handles = [
        Line2D([], [], color=OI["black"], lw=1.1, label="Observed stock"),
        Line2D([], [], color=OI["vermillion"], lw=1.1, ls=(0, (4, 1.5)),
               label=r"Frozen equilibrium"),
        Line2D([], [], color=OI["blue"], lw=1.0, ls=(0, (1, 1.2)),
               label="Saturation ratio"),
    ]
    ax.legend(handles=handles, loc="upper left")
    panel_label(ax, "b)")

    # c) transient amplification
    ax = fig.add_subplot(gs[1, 1])
    shade_windows(ax)
    ax.fill_between(amp.year, amp.sup_norm2_q1, amp.sup_norm2_q3,
                    color=OI["green"], alpha=0.2, lw=0)
    ax.plot(amp.year, amp.sup_norm2_median, color=OI["green"], lw=1.1,
            label="Non-autonomous")
    ax.plot(amp.year, amp.frozen_sup_norm2_median, color=OI["purple"], lw=1.0,
            ls=(0, (4, 1.5)), label="Frozen")
    ax.axhline(1.0, color=OI["grey"], lw=0.7, ls=":")
    ax.text(YEAR_MIN + 0.6, 1.003, "no amplification", fontsize=7, color=OI["grey"],
            va="bottom")
    ax.set_xlim(YEAR_MIN, YEAR_MAX)
    ax.set_xlabel("Year")
    ax.set_ylabel(r"$\sup_{u}\,\|\Phi(t+u,t)\|_{2}$")
    ax.set_ylim(0.99, 1.47)
    light_grid(ax)
    ax.legend(loc="upper left", ncol=2, columnspacing=1.0, handlelength=1.5)
    panel_label(ax, "c)")

    # d) the three fast compartments against their own frozen equilibrium
    ax = fig.add_subplot(gs[2, :])
    shade_windows(ax)
    fast = [("Concentrate", OI["blue"], (0, ())),
            ("Refined", OI["vermillion"], (0, (4, 1.5))),
            ("Scrap", OI["green"], (0, (1, 1.2)))]
    for name, colour, dash in fast:
        sub = sat[sat.stock == name].sort_values("year")
        ax.fill_between(sub.year, sub.sat_ratio_obs_lo, sub.sat_ratio_obs_hi,
                        color=colour, alpha=0.16, lw=0)
        ax.plot(sub.year, sub.sat_ratio_obs_hl, color=colour, ls=dash, lw=1.0,
                label=name)
    ax.axhline(1.0, color=OI["grey"], lw=0.7, ls=":")
    ax.set_xlim(YEAR_MIN, YEAR_MAX)
    ax.set_ylim(0, 1.72)
    ax.set_xlabel("Year")
    ax.set_ylabel(r"Saturation ratio $S/S^{*}$")
    light_grid(ax)
    ax.legend(loc="upper left", ncol=3, columnspacing=1.0, handlelength=1.5,
              borderaxespad=0.25)
    panel_label(ax, "d)")

    return save(fig, out_dir, "ch4_fig4_system.pdf")


# --------------------------------------------------------------------------- #
# Figure 5 — the circularity indicators
# --------------------------------------------------------------------------- #

# The two route-level counts are the expected passages through the direct-reuse
# and Waelz routes, so they carry the same mnemonic subscripts as the transfer
# coefficients that govern those routes rather than the Chapter 2-3 indices
# (Upsilon_13 and Upsilon_14 respectively).
IND_LABEL = {
    "tau": r"$\tau_{\mathrm{FD}}$",
    "use_entry": r"$\upsilon$",
    "oldscrap_manufacture": r"$\Upsilon_{\mathrm{reuse}}$",
    "oldscrap_metallurgy": r"$\Upsilon_{\mathrm{waelz}}$",
}
IND_NAME = {
    "tau": "technological lifetime",
    "use_entry": "expected entries into use",
    "oldscrap_manufacture": "direct-reuse count",
    "oldscrap_metallurgy": "Waelz count",
}
IND_COLOUR = {
    "tau": OI["blue"],
    "use_entry": OI["green"],
    "oldscrap_manufacture": OI["purple"],
    "oldscrap_metallurgy": OI["vermillion"],
}
IND_ORDER = ["tau", "use_entry", "oldscrap_manufacture", "oldscrap_metallurgy"]
# One marker per indicator, so the four are separated by shape as well as
# colour; linestyle is reserved throughout the figure for the frozen/
# non-autonomous contrast.
IND_MARKER = {
    "tau": "o",
    "use_entry": "s",
    "oldscrap_manufacture": "^",
    "oldscrap_metallurgy": "D",
}


def figure5(out_dir: Path) -> Path:
    ind = pd.read_csv(ANALYSIS / "wp8h_indicator_uncertainty_nonauto.csv")
    ind = ind[(ind.state == "S_conc") & (ind["rule"] == "hold")]

    def series(name, form):
        return ind[(ind.indicator == name) & (ind.form == form)].sort_values("year")

    fig = plt.figure(figsize=(FULL, 5.1), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.15, 0.9])

    # One convention across all three panels: colour and marker name the
    # indicator, linestyle names the form.  The non-autonomous series carry the
    # marker; the frozen ones are the dashed twin in the same colour.
    def draw(ax, name, form):
        s = series(name, form)
        colour = IND_COLOUR[name]
        solid = form == "non-autonomous"
        ax.fill_between(s.year, s.lo95, s.hi95, color=colour,
                        alpha=0.20 if solid else 0.14, lw=0)
        ax.plot(s.year, s.value, color=colour, lw=1.1,
                ls=(0, ()) if solid else (0, (4, 1.5)),
                marker=IND_MARKER[name] if solid else "None",
                ms=2.8, markevery=5, mfc="none", mew=0.8)

    # a) the two aggregate indicators, frozen against non-autonomous
    ax = fig.add_subplot(gs[0, 0])
    shade_windows(ax)
    for form in ("frozen", "non-autonomous"):
        draw(ax, "tau", form)
    ax.set_xlim(YEAR_MIN, YEAR_MAX)
    ax.set_xlabel("Year")
    ax.set_ylim(14, 57)
    ax.set_ylabel(r"$\tau_{\mathrm{FD}}$ (yr)", color=IND_COLOUR["tau"])
    ax.tick_params(axis="y", colors=IND_COLOUR["tau"])
    light_grid(ax)

    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)
    for form in ("frozen", "non-autonomous"):
        draw(ax2, "use_entry", form)
    ax2.set_ylim(0.93, 1.44)
    ax2.set_ylabel(r"$\upsilon$ (entries into use)", color=IND_COLOUR["use_entry"])
    ax2.tick_params(axis="y", colors=IND_COLOUR["use_entry"])
    # The two axes are colour-coded to their indicator, so which is read left
    # and which right needs no legend entry of its own.
    panel_label(ax, "a)")

    # b) the two route-level counts
    ax = fig.add_subplot(gs[0, 1])
    shade_windows(ax)
    for name in ("oldscrap_manufacture", "oldscrap_metallurgy"):
        for form in ("frozen", "non-autonomous"):
            draw(ax, name, form)
    ax.axhline(0.0, color=OI["grey"], lw=0.6, ls=":")
    ax.set_xlim(YEAR_MIN, YEAR_MAX)
    ax.set_xlabel("Year")
    ax.set_ylabel("Expected passages per unit")
    light_grid(ax)
    panel_label(ax, "b)")

    # c) the frozen-time discrepancy against the estimation uncertainty
    ax = fig.add_subplot(gs[1, :])
    shade_windows(ax)
    for name in IND_ORDER:
        s = series(name, "discrepancy (nonauto - frozen)")
        # At the terminal year the two forms coincide exactly under the hold
        # rule, so both the discrepancy and its standard error are zero; the
        # ratio is defined as zero there rather than left undefined.
        val = np.abs(s.value.to_numpy())
        sd = s.sd_total.to_numpy()
        ratio = np.divide(val, sd, out=np.zeros_like(val), where=sd > 0)
        # A derived quantity, neither frozen nor non-autonomous, so it is drawn
        # solid; the marker is what separates the four.
        ax.plot(s.year, ratio, color=IND_COLOUR[name], lw=1.0,
                marker=IND_MARKER[name], ms=2.8, markevery=5, mfc="none",
                mew=0.8)
    ax.axhline(1.0, color=OI["grey"], lw=0.7, ls=":")
    # Labelled in words: sigma is the new-scrap share in Chapter 4's notation.
    ax.text(YEAR_MIN + 0.6, 1.03, "1 s.e.", fontsize=7, color=OI["grey"],
            va="bottom")
    ax.set_xlim(YEAR_MIN, YEAR_MAX)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Year")
    ax.set_ylabel("Frozen-time discrepancy\n(standard errors)")
    light_grid(ax)
    panel_label(ax, "c)")

    # One legend for the whole figure: four indicators, then the two forms.
    handles = [
        Line2D([], [], color=IND_COLOUR[n], lw=1.1, marker=IND_MARKER[n],
               ms=3.2, mfc="none", mew=0.8,
               label=f"{IND_LABEL[n]}, {IND_NAME[n]}")
        for n in IND_ORDER
    ] + [
        Line2D([], [], color=OI["grey"], lw=1.1, label="Non-autonomous"),
        Line2D([], [], color=OI["grey"], lw=1.1, ls=(0, (4, 1.5)), label="Frozen"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3,
               columnspacing=1.4)

    return save(fig, out_dir, "ch4_fig5_indicators.pdf")


# --------------------------------------------------------------------------- #
# Figure 6 — the three sensitivities
# --------------------------------------------------------------------------- #

PARAM_TEX = {
    **ALPHA_SHORT,
    "tau_ref": r"$k_{\mathrm{ref}}$", "tau_waelz": r"$k_{\mathrm{waelz}}$",
    "tau_olds": r"$k_{\mathrm{olds}}$", "tau_diss": r"$k_{\mathrm{diss}}$",
    "frac_fu_new": r"$\sigma_{\mathrm{fu}}$", "frac_fu_loss": r"$\lambda_{\mathrm{fu}}$",
    "frac_fu_out": r"$\nu_{\mathrm{fu}}$", "frac_eu_new": r"$\sigma_{\mathrm{eu}}$",
    "frac_eu_loss": r"$\lambda_{\mathrm{eu}}$", "frac_eu_into": r"$\nu_{\mathrm{eu}}$",
    "f_cohort_10yr": r"$f_{10}$", "f_cohort_20yr": r"$f_{20}$",
    "f_cohort_44yr": r"$f_{44}$", "mu_10yr": r"$\mu_{10}$",
    "mu_20yr": r"$\mu_{20}$", "mu_44yr": r"$\mu_{44}$",
}


def figure6(out_dir: Path) -> Path:
    modal = pd.read_csv(ANALYSIS / "wp2c_sensitivity_summary.csv")
    elas = pd.read_csv(ANALYSIS / "wp2d_elasticities.csv")
    rank = pd.read_csv(ANALYSIS / "wp2e_vs_wp2c_rank.csv")
    with np.load(ANALYSIS / "wp2e_indicator_sensitivity.npz", allow_pickle=True) as d:
        term_param_names = [str(x) for x in d["param_names"]]
        term_state_names = [str(x) for x in d["state_names"]]
        term_direct = d["term_direct_use_entry"]          # (seed, year, param, state)
        term_prop = d["term_propagated_use_entry"]
    perm = pd.read_csv(ANALYSIS / "wp8d_permutation.csv")

    fig = plt.figure(figsize=(FULL, 5.7), constrained_layout=True)
    # Two subfigures rather than one 2x2 grid.  Panel c spans both columns and
    # spends nearly two inches of its left-hand side on the driver names; in a
    # shared gridspec that margin is reserved across the whole of column 0, so
    # it emptied the left of the top row and squeezed a and b into what was
    # left.  A subfigure per row confines c's margin to its own row and gives
    # the top row the full text width.
    rows = fig.subfigures(2, 1, height_ratios=[1.22, 1.1])
    # Panel b spends about half an inch on the parameter names down its own
    # left-hand side, so a takes the larger share of the row to even the two
    # plotting areas up.
    gs = rows[0].add_gridspec(1, 2, width_ratios=[1.16, 1.0])

    # a) the modal ranking against the indicator ranking
    ax = rows[0].add_subplot(gs[0, 0])
    m = modal[(modal.year == YEAR_MAX) & (modal["mode"] == "S_iu_long")]
    m = m.set_index("param").elas_timescale_median
    e = elas[(elas.indicator == "tau_FD") & (elas.start_stock == "S_conc")
             & (elas.year == YEAR_MAX)].set_index("param").elasticity_median
    common = [p for p in m.index if p in e.index]
    xs = m.reindex(common).to_numpy()
    ys = e.reindex(common).to_numpy()
    ax.axvline(0.0, color=OI["grey"], lw=0.6, ls=":")
    ax.axhline(0.0, color=OI["grey"], lw=0.6, ls=":")
    ax.plot(xs, ys, lw=0, marker="o", ms=3.6, color=OI["blue"], mfc="none", mew=0.9)
    # Room around the cloud before anything is annotated, so no label is written
    # into a spine.
    ax.margins(x=0.17, y=0.15)
    ax.autoscale_view()
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    for p, x, y in zip(common, xs, ys):
        if abs(y) > 0.25 or abs(x) > 0.25 or p == "tau_ref":
            # Points in the right-hand third are labelled to their left, and
            # points near the ceiling from below, so the text stays inside.
            right = x > x_lo + 0.66 * (x_hi - x_lo)
            # tau_ref is the one negative point and sits under the zero line,
            # so it is labelled downwards rather than back across that line.
            high = y > y_lo + 0.88 * (y_hi - y_lo) or p == "tau_ref"
            # The two pass-through shares lie 0.08 apart near the ceiling; the
            # up/down rule sent their labels across each other, so both are
            # written level with their own marker, to its right.
            beside = p in ("frac_fu_out", "frac_eu_into")
            ax.annotate(
                PARAM_TEX.get(p, p), (x, y), textcoords="offset points",
                xytext=(5.0, 0.0) if beside else
                       (-4.0 if right else 4.0, -7.5 if high else 2.5),
                ha="left" if beside else ("right" if right else "left"),
                va="center" if beside else ("top" if high else "bottom"), fontsize=7,
                color=OI["vermillion"] if p == "tau_ref" else OI["black"])
    ax.plot(m.get("tau_ref", 0.0), e.get("tau_ref", 0.0), lw=0, marker="o",
            ms=4.6, color=OI["vermillion"], zorder=5)
    rho = float(rank.spearman_rho.median())
    # Top right: the cloud runs up the y axis and the one point out to the
    # right sits below the middle, so that corner is the one reliably empty.
    ax.text(0.97, 0.97, r"Spearman $\rho = $" + f"{rho:.3f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=7)
    ax.set_xlabel("Elasticity of the dominant timescale")
    ax.set_ylabel(r"Elasticity of $\tau_{\mathrm{FD}}$")
    light_grid(ax, axis="both")
    panel_label(ax, "a)")

    # b) the direct and propagated terms of the indicator sensitivity
    ax = rows[0].add_subplot(gs[0, 1])
    # Medians over the 35 x 40 (seed, year) nodes, from a unit entering at
    # Concentrate, which is the statistic WP-2e reports.
    k = term_state_names.index("S_conc")
    direct = np.median(term_direct[:, :, :, k], axis=(0, 1))
    propagated = np.median(term_prop[:, :, :, k], axis=(0, 1))
    net = np.median(term_direct[:, :, :, k] + term_prop[:, :, :, k], axis=(0, 1))
    # Every parameter that moves the indicator at all, ranked by the gross
    # movement |direct| + |propagated| so that the cancelling channels are
    # visible.  The three cohort shares are numerically identical here, so the
    # 10 yr and 20 yr rows are dropped as duplicates of the 44 yr one.
    gross = np.abs(direct) + np.abs(propagated)
    ref = term_param_names.index("f_cohort_44yr")
    drop = set()
    for dup in ("f_cohort_10yr", "f_cohort_20yr"):
        i = term_param_names.index(dup)
        if abs(direct[i] - direct[ref]) < 1e-9 and abs(propagated[i] - propagated[ref]) < 1e-9:
            drop.add(i)
    keep = [i for i in np.argsort(gross) if gross[i] > 0.01 and i not in drop]
    names = [term_param_names[i] for i in keep]
    y = np.arange(len(keep))
    ax.barh(y + 0.19, direct[keep], height=0.36, color=OI["blue"], alpha=0.85,
            lw=0, label="Direct")
    ax.barh(y - 0.19, propagated[keep], height=0.36, color=OI["orange"],
            alpha=0.85, lw=0, label="Propagated")
    ax.plot(net[keep], y, lw=0, marker="D", ms=3.6, color=OI["black"],
            mfc="none", mew=0.9, label="Net")
    ax.axvline(0.0, color=OI["grey"], lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([PARAM_TEX.get(p, p) for p in names], fontsize=7.5)
    ax.set_xlabel(r"$\partial\upsilon\,/\,\partial\log p$")
    ax.set_xlim(-2.4, 2.1)
    # Headroom above the top row so the legend does not sit on the bars.
    ax.set_ylim(-0.7, len(keep) + 3.4)
    light_grid(ax, axis="x")
    ax.legend(loc="upper left", ncol=1, handlelength=1.3, labelspacing=0.25)
    panel_label(ax, "b)")

    # c) block-permutation driver importance
    ax = rows[1].add_subplot()
    p = perm[perm.family == "alpha"].sort_values("degradation_median")
    y = np.arange(len(p))
    lo = (p.degradation_median - p.q1).to_numpy()
    hi = (p.q3 - p.degradation_median).to_numpy()
    # Points rather than bars: the axis is logarithmic; a bar drawn from an
    # arbitrary left-hand limit would misrepresent the ratios.
    ax.hlines(y, p.q1, p.q3, color=OI["green"], lw=1.0, alpha=0.9)
    ax.errorbar(p.degradation_median, y, xerr=np.vstack([lo, hi]), fmt="o", ms=4.0,
                color=OI["green"], ecolor=OI["green"], elinewidth=0.9, capsize=2.0)
    ax.set_yticks(y)
    ax.set_yticklabels([d if len(d) < 52 else d[:49] + "…" for d in p.driver],
                       fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Coefficient degradation on permutation (per cent)")
    ax.set_ylim(-0.7, len(p) - 0.3)
    light_grid(ax, axis="x")
    panel_label(ax, "c)")

    return save(fig, out_dir, "ch4_fig6_sensitivity.pdf")


# --------------------------------------------------------------------------- #
# Figure 7 — system response
# --------------------------------------------------------------------------- #

CF_VARS = [
    ("old_scrap_recovery", "End-of-life collection", OI["blue"], (0, ())),
    ("secondary_supply", "Secondary supply", OI["vermillion"], (0, (4, 1.5))),
    ("Scrap", "Scrap stock", OI["green"], (0, (1, 1.2))),
    ("In-Use", "In-use stock", OI["purple"], (0, (5, 1.4, 1, 1.4))),
]


def figure7(out_dir: Path) -> Path:
    eff = pd.read_csv(ANALYSIS / "wp7_effects_by_year.csv")
    tte = pd.read_csv(ANALYSIS / "wp7_time_to_effect.csv")
    mem = pd.read_csv(ANALYSIS / "wp8b_memory_length.csv")

    arm = "tau_olds_step10_2005"
    sub = eff[eff.arm == arm]

    fig = plt.figure(figsize=(FULL, 3.0), constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.18, 1.0])

    # a) a ten-percentage-point rise in end-of-life collection from 2005
    ax = fig.add_subplot(gs[0])
    intervention = float(tte.intervention_year.iloc[0])
    ax.axvline(intervention, color=OI["grey"], lw=0.7, ls=":")
    ax.text(intervention + 0.3, 0.97, "intervention", transform=ax.get_xaxis_transform(),
            fontsize=7, color=OI["grey"], va="top")
    for var, label, colour, dash in CF_VARS:
        s = sub[sub.variable == var].sort_values("year")
        if s.empty:
            continue
        s = s[s.year >= intervention - 2]
        ax.fill_between(s.year, s.diff_pct_q1, s.diff_pct_q3, color=colour,
                        alpha=0.16, lw=0)
        ax.plot(s.year, s.diff_pct_median, color=colour, ls=dash, lw=1.1, label=label)
    ax.axhline(0.0, color=OI["grey"], lw=0.6)
    ax.set_xlabel("Year")
    ax.set_ylabel("Deviation from the factual path\n(per cent)")
    ax.set_xlim(intervention - 2, YEAR_MAX)
    light_grid(ax)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.62), ncol=1)
    panel_label(ax, "a)")

    # b) driver-shock memory length by compartment
    ax = fig.add_subplot(gs[1])
    data = [mem.loc[mem.stock == s, "memory_median_yr"].to_numpy() for s in STOCKS]
    bp = ax.boxplot(data, positions=np.arange(len(STOCKS)), widths=0.45,
                    patch_artist=True, medianprops=dict(color=OI["black"], lw=1.0),
                    whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6),
                    flierprops=dict(marker="o", ms=2.0, mfc="none",
                                    mec=OI["grey"], mew=0.5))
    for box in bp["boxes"]:
        box.set(facecolor=OI["orange"], alpha=0.35, lw=0.6, edgecolor=OI["orange"])
    for j, s in enumerate(STOCKS):
        vals = mem.loc[mem.stock == s, "memory_median_yr"].to_numpy()
        ax.plot(np.full(vals.shape, j) + np.linspace(-0.13, 0.13, vals.size),
                vals, lw=0, marker="o", ms=2.2, color=OI["grey"], alpha=0.8, mew=0)
    ax.set_xticks(np.arange(len(STOCKS)))
    ax.set_xticklabels([STOCK_DISPLAY[s] for s in STOCKS], fontsize=7.0)
    ax.set_ylabel("Driver-shock memory length (yr)")
    ax.set_xlim(-0.6, len(STOCKS) - 0.4)
    light_grid(ax)
    panel_label(ax, "b)")

    return save(fig, out_dir, "ch4_fig7_response.pdf")


# --------------------------------------------------------------------------- #
# Figure 8 — the value of observation
# --------------------------------------------------------------------------- #

# Candidate series named as in tab:ch4_observation.
SERIES_LABEL = {
    "eol_composition": "end-of-life product composition",
    "waelz_input": "Waelz kiln input",
    "end_of_life": "aggregate end-of-life arisings",
    "scrap_collection": "scrap collection tonnage",
    "direct_reuse_recycling": "direct-reuse recycling",
    "refined_stock": "refined stock",
    "primary_refining": "primary refining",
    "concentrate_consumption": "concentrate consumption",
    "refined_consumption": "refined consumption",
    "scrap_stock": "scrap stock",
    "inuse_stock": "in-use stock",
}


def figure8(out_dir: Path) -> Path:
    coarse = pd.read_csv(ANALYSIS / "wp6b_coarsening.csv")
    obs = pd.read_csv(ANALYSIS / "wp11g_observables.csv")

    fig = plt.figure(figsize=(FULL, 3.0), constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.35])

    # a) coefficient error against the reporting interval
    ax = fig.add_subplot(gs[0])
    c = coarse[coarse.split == "trainval"]
    for name in ALPHA_ORDER:
        s = c[c.channel == name].groupby("delta").alpha_relRMSE_pct.agg(
            median="median",
            q1=lambda x: np.percentile(x, 25),
            q3=lambda x: np.percentile(x, 75),
        ).sort_index()
        ax.fill_between(s.index, s.q1, s.q3, color=ALPHA_COLOUR[name], alpha=0.14, lw=0)
        ax.plot(s.index, s["median"], color=ALPHA_COLOUR[name], ls=ALPHA_DASH[name],
                lw=1.1, marker="o", ms=3.0, label=ALPHA_SHORT[name])
    ax.set_xscale("log")
    ax.set_xticks(sorted(c.delta.unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Reporting interval (yr)")
    ax.set_ylabel("Coefficient error (per cent)")
    light_grid(ax)
    ax.legend(loc="upper left", ncol=2)
    panel_label(ax, "a)")

    # b) marginal reduction in coefficient uncertainty, by candidate series
    ax = fig.add_subplot(gs[1])
    o = obs.sort_values("reduction_annual_pct")
    y = np.arange(len(o))
    coef_colour = {
        "f_cohort_10yr": OI["blue"], "f_cohort_20yr": OI["sky"],
        "f_cohort_44yr": OI["green"], "alpha_win": OI["vermillion"],
        "tau_olds": OI["orange"], "alpha_dr": OI["purple"],
        "alpha_refc": OI["grey"], "alpha_cc": OI["black"],
    }
    colours = [coef_colour.get(c, OI["grey"]) for c in o.coefficient]
    ax.barh(y, o.reduction_annual_pct, height=0.66, color=colours, alpha=0.85, lw=0)
    ax.set_yticks(y)
    # Several series appear more than once because they constrain more than one
    # coefficient, so each row names the coefficient it is scored against.
    ax.set_yticklabels(
        [f"{SERIES_LABEL.get(ser, prettify(ser))} " + r"$\rightarrow$ " + f"{PARAM_TEX.get(c, c)}"
         for ser, c in zip(o.model_series, o.coefficient)], fontsize=7)
    ax.set_xlabel("Uncertainty reduction (per cent)")
    ax.set_ylim(-0.7, len(o) - 0.3)
    light_grid(ax, axis="x")
    panel_label(ax, "b)")

    return save(fig, out_dir, "ch4_fig8_observation.pdf")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

BUILDERS = {
    "1": figure1,
    "1flowsa": figure1_flows_a,
    "1flowsb": figure1_flows_b,
    "1coefficients": figure1_coefficients,
    "2": figure2, "3": figure3, "4": figure4,
    "5": figure5, "6": figure6, "7": figure7, "8": figure8,
}
# Declaration order, which is also the order they are built in.
FIGURE_IDS = list(BUILDERS)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="verify every input exists, then exit without plotting")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"output directory (default: {DEFAULT_OUT})")
    ap.add_argument("--only", nargs="+", choices=FIGURE_IDS, metavar="ID",
                    help="build a subset: " + ", ".join(FIGURE_IDS))
    args = ap.parse_args(argv)

    figures = args.only or FIGURE_IDS

    if args.check:
        return 1 if check_inputs(figures) else 0

    problems = check_inputs(figures)
    if problems:
        print("\nRefusing to plot with missing inputs; nothing is synthesised.",
              file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    print()
    with plt.rc_context(RC):
        for fig_id in figures:
            path = BUILDERS[fig_id](args.out)
            size = path.stat().st_size / 1024
            print(f"wrote {path}  ({size:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
