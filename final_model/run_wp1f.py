#!/usr/bin/env python3
"""
run_wp1f.py — WP-1f: mass imbalance of a per-flow AR/ARX ensemble
=================================================================

Spec: "Fit independent AR/ARX per observed flow (same drivers, lag order by
AIC) on the training window.  Integrate the flow paths to implied stocks.
Report residual mass imbalance per year in kt and cumulatively over the test
window.  A single number showing a per-flow AR ensemble is not a system model."

This is the answer to the anticipated objection that a per-flow autoregressive
model achieves lower forecast error than the UDE (spec §0).  The reply is not
that the AR forecasts badly — it is that the quantity it produces is not a
material flow system.

The mass-balance structure, read off the model's own RHS
--------------------------------------------------------
`make_rhs` (`zinc_colloc_v5.py:1240`) integrates exactly these four balances,
so they are identities for the UDE by construction, not fitted targets:

    dS_conc  =  concentrate_production − concentrate_consumption
    dS_ref   =  primary_refining + waelz_recycling − refined_consumption
    dS_inuse =  inuse_inflow − end_of_life
    dS_scrap =  old_scrap_recovery + first_use_new_scrap + end_use_new_scrap
                − waelz_input − direct_reuse_recycling

`compute_flows_from_nn` (`:1132`) additionally forces four identities among
the flows themselves, with no stock involved:

    primary_refining + refinery_losses            =  concentrate_consumption
    waelz_recycling  + waelz_losses               =  waelz_input
    dissipative_use  + inuse_inflow               =  total_products_into_use
    first_use_new_scrap + first_use_losses
      + end_use_new_scrap + end_use_losses
      + total_products_into_use                   =  refined_consumption
                                                     + direct_reuse_recycling

Every flow named above is in the observed inventory (only `end_of_life_losses`
is unobserved, and it enters none of them).  **The Rostek dataset satisfies all
eight exactly** — verified here, max residual 5.7e-13 kt on 39 intervals — so
any imbalance a model shows is the model's, not the data's.  That is what makes
the comparison clean: the AR ensemble is not being penalised for inconsistent
data.

Design choices, and why
-----------------------
*   **Fitted on 1980–2007** (the trainval window, everything before the
    pre-registered test split) and forecast dynamically over 2008–2019, which
    matches the UDE's `testrun` rollout launched at the trainval boundary.
*   **Log space.**  All 18 observed flows are strictly positive; the UDE's own
    loss is log-space; and exponentiating keeps forecasts positive, which a
    mass-balance argument needs.  The lognormal mean correction `exp(σ̂²/2)` is
    applied, since mass balance is a statement about levels, not medians.
*   **Actual driver values are used over the test window** — the ARX gets
    perfect foresight of its exogenous inputs.  This is deliberately generous:
    it removes any suspicion that the imbalance is a driver-forecasting
    artefact.
*   **Driver subsets capped at 3** in the AIC search.  With 27 usable training
    rows an ARX carrying all 13 drivers plus lags is not estimable in any
    meaningful sense; that specification is still run as a robustness arm
    (`ARX-all`) precisely to show the imbalance does not come from
    underfitting.

Because the point is structural rather than a property of one specification,
four specifications are fitted and the imbalance reported for all of them.

    python run_wp1f.py --check
    python run_wp1f.py
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DUMP_DIR = os.path.join(HERE, "analysis", "wp1d")
ALPHA_DUMP_DIR = os.path.join(HERE, "analysis", "wp1a")
ANCHOR_DIR = os.path.join(HERE, "anchor_v4")
OUT_DIR = os.path.join(HERE, "analysis")

STOCK_NAMES = ["Concentrate", "Refined", "In-Use", "Scrap"]

# dS_k = Σ(inflows) − Σ(outflows), by flow NAME (never by index — the observed
# column order is not FLOW_NAMES order).
BALANCE = {
    "Concentrate": (["concentrate_production"], ["concentrate_consumption"]),
    "Refined":     (["primary_refining", "waelz_recycling"], ["refined_consumption"]),
    "In-Use":      (["inuse_inflow"], ["end_of_life"]),
    "Scrap":       (["old_scrap_recovery", "first_use_new_scrap", "end_use_new_scrap"],
                    ["waelz_input", "direct_reuse_recycling"]),
}

# Identities among flows alone, no stock involved.
IDENTITIES = {
    "refining split":      (["primary_refining", "refinery_losses"],
                            ["concentrate_consumption"]),
    "waelz split":         (["waelz_recycling", "waelz_losses"], ["waelz_input"]),
    "use-phase split":     (["dissipative_use", "inuse_inflow"],
                            ["total_products_into_use"]),
    "manufacturing chain": (["first_use_new_scrap", "first_use_losses",
                             "end_use_new_scrap", "end_use_losses",
                             "total_products_into_use"],
                            ["refined_consumption", "direct_reuse_recycling"]),
}

SPECS = {
    # name          : (max_lag, max_drivers, all_drivers, log_space, smearing)
    "ARX-aic":        dict(max_lag=3, max_drivers=3, all_drivers=False,
                           log=True, smear=True),
    "AR-only":        dict(max_lag=3, max_drivers=0, all_drivers=False,
                           log=True, smear=True),
    "ARX-all":        dict(max_lag=1, max_drivers=13, all_drivers=True,
                           log=True, smear=True),
    "ARX-aic-levels": dict(max_lag=3, max_drivers=3, all_drivers=False,
                           log=False, smear=False),
}
PRIMARY_SPEC = "ARX-aic"

STOCK_COLOURS = {"Concentrate": "#0072B2", "Refined": "#009E73",
                 "In-Use": "#D55E00", "Scrap": "#CC79A7"}
GREY = "#555555"


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def rel_rmse_pct(pred, obs, eps=1e-12):
    """Mirrors `zinc_colloc_v5._rel_rmse_pct` — denominator `mean(|obs|)`."""
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    m = np.isfinite(pred) & np.isfinite(obs)
    if not m.any():
        return float("nan")
    den = float(np.mean(np.abs(obs[m])))
    if den < eps:
        return float("nan")
    return 100.0 * float(np.sqrt(np.mean((pred[m] - obs[m]) ** 2))) / den


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_series():
    """Observations plus the resolved drivers.

    Flows, stocks and masks come from a WP-1a/1d dump (they are data, identical
    across seeds).  Drivers come from `load_zinc_data` so the driver set is the
    one the model actually resolved, not one re-derived from the sheet.
    """
    import zinc_alpha_lab as lab
    import zinc_colloc_v5 as v5

    path = None
    for cand in (os.path.join(DUMP_DIR, "stage_seed0.npz"),
                 os.path.join(ALPHA_DUMP_DIR, "alpha_seed0.npz")):
        if os.path.exists(cand):
            path = cand
            break
    if path is None:
        raise SystemExit("no per-seed dump found — run zinc_alpha_lab.py first")
    d = np.load(path, allow_pickle=True)

    cfg = lab.load_anchor_config()
    dn = v5.load_zinc_data(cfg["xlsx_path"], extra_exog_cols=cfg.get("extra_exog_cols"))

    years = np.asarray(d["years"], float).ravel()
    out = dict(
        years=years,
        stocks_obs=np.asarray(d["stocks_obs"], float),
        mask_train=np.asarray(d["mask_train"], bool),
        mask_val=np.asarray(d["mask_val"], bool),
        mask_test=np.asarray(d["mask_test"], bool),
        exog_cols=[str(c) for c in dn["exog_cols"]],
        exog_values=np.asarray(dn["exog_values"], float),
    )
    if "flows_obs" in d.files:
        out["flows_obs"] = np.asarray(d["flows_obs"], float)
        out["flow_names"] = [str(x) for x in d["flow_names"]]
    else:                                    # WP-1a dumps carry no flows
        st = np.load(os.path.join(ANCHOR_DIR, "pred_seed0.npz"), allow_pickle=True)
        out["flows_obs"] = np.asarray(st["flows_obs"], float)
        out["flow_names"] = [str(x) for x in st["flow_names"]]
    # Flow row i closes the interval (years[i], years[i+1]].
    out["years_flow"] = years[1:]
    return out


def check_data_consistency(S, F, years_flow, fnames):
    """The dataset's own mass balance — the baseline every model is read against."""
    I = {n: i for i, n in enumerate(fnames)}
    rows = []
    for k, nm in enumerate(STOCK_NAMES):
        inn, out = BALANCE[nm]
        net = F[:, [I[x] for x in inn]].sum(1) - F[:, [I[x] for x in out]].sum(1)
        resid = np.diff(S[:, k]) - net
        rows.append(dict(kind="stock balance", target=nm,
                         max_abs_kt=float(np.max(np.abs(resid))),
                         mean_abs_kt=float(np.mean(np.abs(resid))),
                         cumulative_kt=float(np.sum(resid))))
    for nm, (a, b) in IDENTITIES.items():
        resid = F[:, [I[x] for x in a]].sum(1) - F[:, [I[x] for x in b]].sum(1)
        rows.append(dict(kind="flow identity", target=nm,
                         max_abs_kt=float(np.max(np.abs(resid))),
                         mean_abs_kt=float(np.mean(np.abs(resid))),
                         cumulative_kt=float(np.sum(resid))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ARX
# ---------------------------------------------------------------------------
def _ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, float(resid @ resid)


def _aic(rss, n, k):
    """Gaussian AIC.  n is the number of usable rows after lag truncation."""
    if n <= 0 or rss <= 0:
        return np.inf
    return n * np.log(rss / n) + 2.0 * k


def fit_arx_one(y, Xex, n_train, max_lag=3, max_drivers=3, all_drivers=False):
    """Fit one flow's ARX by AIC over lag order and driver subset.

    `y` is the full-length series (train + test); only the first `n_train`
    rows are used for estimation.  Returns the chosen order, the coefficients
    and the residual variance, which the caller needs for the smearing
    correction and for the dynamic forecast.
    """
    n_ex = Xex.shape[1]
    if all_drivers:
        subsets = [tuple(range(n_ex))]
    else:
        subsets = [()]
        for k in range(1, min(max_drivers, n_ex) + 1):
            subsets += list(itertools.combinations(range(n_ex), k))

    best = None
    for p in range(0, max_lag + 1):
        rows = np.arange(p, n_train)
        if rows.size < 8:
            continue
        lags = [y[rows - i] for i in range(1, p + 1)]
        for sub in subsets:
            cols = [np.ones(rows.size)] + lags + [Xex[rows, j] for j in sub]
            X = np.column_stack(cols)
            k = X.shape[1]
            if rows.size - k < 3:            # keep some residual d.o.f.
                continue
            try:
                beta, rss = _ols(X, y[rows])
            except np.linalg.LinAlgError:
                continue
            a = _aic(rss, rows.size, k)
            if best is None or a < best["aic"]:
                best = dict(aic=a, p=p, sub=sub, beta=beta,
                            sigma2=rss / max(rows.size - k, 1), n_eff=rows.size)
    if best is None:                          # degenerate fallback: mean only
        m = float(np.mean(y[:n_train]))
        best = dict(aic=np.inf, p=0, sub=(), beta=np.array([m]),
                    sigma2=float(np.var(y[:n_train])), n_eff=n_train)
    return best


def forecast_arx(fit, y, Xex, n_train, n_total):
    """Dynamic (multi-step) forecast: lagged values inside the forecast window
    are the model's own predictions, not the observations."""
    p, sub, beta = fit["p"], fit["sub"], fit["beta"]
    path = np.array(y, float, copy=True)
    for t in range(n_train, n_total):
        x = [1.0] + [path[t - i] for i in range(1, p + 1)] + \
            [Xex[t, j] for j in sub]
        path[t] = float(np.dot(beta, np.asarray(x)))
    return path


def fit_ensemble(data, spec_name):
    """Fit one independent model per observed flow and forecast the test window.

    Independence is the point: nothing couples the 18 fits, which is exactly
    what a per-flow AR baseline is.
    """
    spec = SPECS[spec_name]
    F, fnames = data["flows_obs"], data["flow_names"]
    yrs_f = data["years_flow"]
    years, mtest = data["years"], data["mask_test"]

    # Split on flow rows so the forecast window matches the UDE's `testrun`
    # exactly: the ODE is re-launched at the trainval boundary (2007) and the
    # first predicted flow row closes 2008.  The AR therefore gets the same
    # 1980–2007 estimation window Stage B trains on, and forecasts the same 12
    # intervals — no year of advantage either way.
    boundary = float(years[data["mask_val"]][-1])          # 2007
    is_test = yrs_f > boundary
    n_train = int(np.argmax(is_test)) if is_test.any() else len(yrs_f)
    n_total = len(yrs_f)

    # Drivers at the closing year of each interval, standardised on the
    # training rows only (no leakage of test-window scale into the fit).
    ex_full = data["exog_values"]
    Xex = ex_full[1:, :]                     # year of the closing endpoint
    mu, sd = Xex[:n_train].mean(0), Xex[:n_train].std(0)
    Xex = (Xex - mu) / np.where(sd > 0, sd, 1.0)

    pred = np.empty_like(F)
    rows = []
    for j, nm in enumerate(fnames):
        raw = F[:, j]
        y = np.log(raw) if spec["log"] else raw.copy()
        fit = fit_arx_one(y, Xex, n_train, max_lag=spec["max_lag"],
                          max_drivers=spec["max_drivers"],
                          all_drivers=spec["all_drivers"])
        path = forecast_arx(fit, y, Xex, n_train, n_total)
        if spec["log"]:
            # Mass balance is about levels, so correct the lognormal median
            # back to a mean.  Applied to the forecast rows only — inside the
            # training window `path` is the observed series by construction.
            corr = np.exp(0.5 * fit["sigma2"]) if spec["smear"] else 1.0
            p_lvl = np.exp(path)
            p_lvl[n_train:] *= corr
        else:
            p_lvl = path
        pred[:, j] = p_lvl
        rows.append(dict(spec=spec_name, flow=nm, lag_order=fit["p"],
                         n_drivers=len(fit["sub"]),
                         drivers=";".join(data["exog_cols"][i] for i in fit["sub"]),
                         sigma2=fit["sigma2"], n_eff=fit["n_eff"],
                         test_relRMSE=rel_rmse_pct(p_lvl[n_train:], F[n_train:, j])))
    return pred, pd.DataFrame(rows), n_train, is_test


# ---------------------------------------------------------------------------
# imbalance
# ---------------------------------------------------------------------------
def mass_imbalance(pred_F, data, n_train, label):
    """Per-year and cumulative residual mass imbalance over the test window.

    For each stock, the per-year imbalance is

        δ_k(i) = [Σ inflows − Σ outflows](i)  −  [S_obs_k(i+1) − S_obs_k(i)]

    i.e. how much mass the flow paths create or destroy in that year relative
    to the reported stock change.  Its running sum is the implied-stock drift,
    anchored at the observed stock on the test boundary, so
    `implied − observed` at the final year is the cumulative imbalance.
    """
    F, S, fnames = pred_F, data["stocks_obs"], data["flow_names"]
    I = {n: i for i, n in enumerate(fnames)}
    yrs_f = data["years_flow"]
    rows = []
    for k, nm in enumerate(STOCK_NAMES):
        inn, out = BALANCE[nm]
        net = F[:, [I[x] for x in inn]].sum(1) - F[:, [I[x] for x in out]].sum(1)
        dS_obs = np.diff(S[:, k])
        delta = net - dS_obs                              # (T-1,)
        implied = S[n_train, k] + np.cumsum(net[n_train:])
        observed = S[n_train + 1:, k]
        for r, i in enumerate(range(n_train, len(yrs_f))):
            rows.append(dict(
                model=label, stock=nm, year=float(yrs_f[i]),
                imbalance_kt=float(delta[i]),
                cumulative_imbalance_kt=float(np.sum(delta[n_train:i + 1])),
                implied_stock_kt=float(implied[r]),
                observed_stock_kt=float(observed[r]),
                observed_stock_pct=100.0 * float(np.sum(delta[n_train:i + 1]))
                / max(abs(float(observed[r])), 1e-12)))
    return pd.DataFrame(rows)


def identity_violation(pred_F, data, n_train, label):
    """Inconsistency among a model's own flows — no stock involved.

    The sharpest form of the result: these are quantities the data satisfies
    to machine precision and the UDE satisfies by construction, so a nonzero
    value is pure incoherence in the flow set itself.
    """
    fnames = data["flow_names"]
    I = {n: i for i, n in enumerate(fnames)}
    yrs_f = data["years_flow"]
    rows = []
    for nm, (a, b) in IDENTITIES.items():
        r = (pred_F[:, [I[x] for x in a]].sum(1)
             - pred_F[:, [I[x] for x in b]].sum(1))
        scale = np.mean(np.abs(data["flows_obs"][:, [I[x] for x in b]].sum(1)))
        for i in range(n_train, len(yrs_f)):
            rows.append(dict(model=label, identity=nm, year=float(yrs_f[i]),
                             violation_kt=float(r[i]),
                             violation_pct=100.0 * float(r[i]) / max(scale, 1e-12)))
    return pd.DataFrame(rows)


def ude_flow_error(data, n_train, anchor_dir=ANCHOR_DIR):
    """The UDE's per-flow test error on exactly the AR's forecast window.

    Uses the stored `testrun` rollout — the ODE re-launched from observed
    stocks at the trainval boundary — because that is the like-for-like
    counterpart of a dynamic AR forecast from the same boundary.  Comparing
    against the 40-year freerun instead would hand the AR an advantage it has
    not earned.
    """
    import glob
    paths = sorted(glob.glob(os.path.join(anchor_dir, "pred_seed*.npz")))
    if not paths:
        return None
    F_obs = data["flows_obs"][n_train:, :]
    rows = []
    for p in paths:
        seed = int(os.path.basename(p)[len("pred_seed"):-len(".npz")])
        st = np.load(p, allow_pickle=True)
        if "F_pred_testrun" not in st.files:
            continue
        Fp = np.asarray(st["F_pred_testrun"], float)
        if Fp.shape != F_obs.shape:                    # alignment guard
            raise SystemExit(
                f"testrun flow block {Fp.shape} does not match the AR forecast "
                f"window {F_obs.shape} — the two are not comparable")
        for j, nm in enumerate(data["flow_names"]):
            rows.append(dict(model="UDE (testrun)", seed=seed, flow=nm,
                             test_relRMSE=rel_rmse_pct(Fp[:, j], F_obs[:, j])))
    return pd.DataFrame(rows)


def ude_reference(data, n_train, dump_dir=DUMP_DIR):
    """The UDE's own imbalance, over the seed ensemble.

    Should be zero to integrator tolerance — the four balances are what
    `make_rhs` integrates.  Verified rather than asserted, because the whole
    contrast rests on it.
    """
    import glob
    paths = sorted(glob.glob(os.path.join(dump_dir, "stage_seed*.npz")))
    if not paths:
        return None, None
    imb, ident = [], []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        if "F_int_B" not in d.files:
            continue
        seed = int(os.path.basename(p).split("_seed")[1].split(".")[0])
        # The UDE's implied stocks are its OWN predicted stocks, so the
        # balance is checked against S_pred, not against S_obs.
        sub = dict(data, stocks_obs=np.asarray(d["S_pred_B"], float))
        a = mass_imbalance(np.asarray(d["F_int_B"], float), sub, n_train,
                           f"UDE seed{seed}")
        a["seed"] = seed
        imb.append(a)
        b = identity_violation(np.asarray(d["F_int_B"], float), data, n_train,
                               f"UDE seed{seed}")
        b["seed"] = seed
        ident.append(b)
    if not imb:
        return None, None
    return pd.concat(imb, ignore_index=True), pd.concat(ident, ignore_index=True)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_imbalance(imb, ude_imb, data, n_train, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.4),
                             gridspec_kw=dict(width_ratios=[1.1, 1.1, 0.9]))
    sub = imb[imb.model == PRIMARY_SPEC]

    # (a) cumulative imbalance in kt
    ax = axes[0]
    for nm in STOCK_NAMES:
        s = sub[sub.stock == nm].sort_values("year")
        ax.plot(s.year, s.cumulative_imbalance_kt, color=STOCK_COLOURS[nm],
                lw=1.9, marker="o", ms=3, label=nm)
    if ude_imb is not None:
        u = (ude_imb.groupby(["stock", "year"]).cumulative_imbalance_kt
             .median().reset_index())
        for nm in STOCK_NAMES:
            s = u[u.stock == nm].sort_values("year")
            ax.plot(s.year, s.cumulative_imbalance_kt, color=STOCK_COLOURS[nm],
                    lw=1.0, ls="--", alpha=0.8)
    ax.axhline(0, color="k", lw=1.0)
    ax.set_ylabel("cumulative mass imbalance (kt)")
    ax.set_xlabel("year")
    ax.set_title(f"(a) implied − observed stock, {PRIMARY_SPEC}",
                 loc="left", fontsize=10)
    ax.legend(fontsize=7.6, frameon=False, ncol=2)
    if ude_imb is not None:
        # The UDE traces lie on the zero line at this scale, which is the
        # point — say so rather than leaving an apparently absent series.
        mx = float(np.max(np.abs(ude_imb.cumulative_imbalance_kt)))
        ax.annotate(f"UDE, all 35 seeds: |imbalance| < {mx:.0e} kt\n"
                    f"(on the zero line at this scale)",
                    xy=(0.5, 0.5), xycoords="axes fraction", fontsize=7.4,
                    color=GREY, ha="center")

    # (b) implied vs observed stock paths
    ax = axes[1]
    for nm in STOCK_NAMES:
        s = sub[sub.stock == nm].sort_values("year")
        ax.plot(s.year, s.implied_stock_kt, color=STOCK_COLOURS[nm], lw=1.9)
        ax.plot(s.year, s.observed_stock_kt, color=STOCK_COLOURS[nm], lw=1.1,
                ls=":", marker="o", ms=2.6)
    ax.set_yscale("symlog", linthresh=100)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("stock (kt, symlog)")
    ax.set_xlabel("year")
    ax.set_title("(b) implied stock path vs reported", loc="left", fontsize=10)
    ax.legend(handles=[Line2D([], [], color=GREY, lw=1.9, label="AR-implied"),
                       Line2D([], [], color=GREY, lw=1.1, ls=":", marker="o",
                              ms=2.6, label="reported")],
              fontsize=7.6, frameon=False)

    # (c) terminal imbalance by specification
    ax = axes[2]
    specs = [s for s in SPECS if s in set(imb.model)]
    w = 0.8 / max(len(STOCK_NAMES), 1)
    for i, spec in enumerate(specs):
        for j, nm in enumerate(STOCK_NAMES):
            s = imb[(imb.model == spec) & (imb.stock == nm)]
            if not len(s):
                continue
            v = float(s.sort_values("year").cumulative_imbalance_kt.iloc[-1])
            ax.bar(i + (j - 1.5) * w, v, width=w * 0.9,
                   color=STOCK_COLOURS[nm], edgecolor="none")
    ax.axhline(0, color="k", lw=1.0)
    ax.set_xticks(range(len(specs)))
    ax.set_xticklabels(specs, rotation=18, fontsize=7.6)
    ax.set_ylabel("terminal imbalance, 2019 (kt)")
    ax.set_title("(c) robustness across specifications", loc="left", fontsize=10)
    ax.legend(handles=[Patch(color=STOCK_COLOURS[nm], label=nm)
                       for nm in STOCK_NAMES],
              fontsize=6.8, frameon=False, loc="lower left", ncol=2)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", lw=0.4, alpha=0.3)
    fig.suptitle("WP-1f  a per-flow AR/ARX ensemble does not conserve mass — "
                 "dashed = UDE ensemble median (zero by construction)",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=300)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


def fig_identities(ident, ude_ident, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    sub = ident[ident.model == PRIMARY_SPEC]
    names = list(IDENTITIES)
    fig, axes = plt.subplots(1, len(names), figsize=(14.0, 3.9), sharex=True)
    for ax, nm in zip(np.atleast_1d(axes), names):
        s = sub[sub.identity == nm].sort_values("year")
        ax.bar(s.year, s.violation_kt, width=0.7, color="#D55E00")
        if ude_ident is not None:
            u = ude_ident[ude_ident.identity == nm]
            mx = float(np.max(np.abs(u.violation_kt))) if len(u) else 0.0
            ax.set_title(f"{nm}\nUDE max |violation| = {mx:.1e} kt",
                         loc="left", fontsize=8.5)
        else:
            ax.set_title(nm, loc="left", fontsize=9)
        ax.axhline(0, color="k", lw=1.0)
        ax.set_xlabel("year")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", lw=0.4, alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("identity violation (kt)")
    fig.suptitle("WP-1f  the AR ensemble's own flows do not add up — "
                 "identities the data satisfies to machine precision",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=300)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="WP-1f AR baseline mass imbalance")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args(argv)

    import zinc_alpha_lab as lab
    lab.check(verbose=True)
    if args.check:
        return 0

    data = load_series()
    print(f"\nflows: {len(data['flow_names'])} observed, "
          f"{data['flows_obs'].shape[0]} intervals "
          f"{data['years_flow'][0]:.0f}–{data['years_flow'][-1]:.0f}")
    print(f"drivers: {len(data['exog_cols'])} resolved")

    consist = check_data_consistency(data["stocks_obs"], data["flows_obs"],
                                     data["years_flow"], data["flow_names"])
    print("\n--- the dataset's own mass balance (the baseline) ---")
    print(consist.round(6).to_string(index=False))

    imb_all, ident_all, orders_all, flowerr = [], [], [], []
    n_train = None
    for spec in SPECS:
        pred, orders, n_train, _ = fit_ensemble(data, spec)
        orders_all.append(orders)
        imb_all.append(mass_imbalance(pred, data, n_train, spec))
        ident_all.append(identity_violation(pred, data, n_train, spec))
        flowerr.append(dict(
            model=spec,
            flow_relRMSE_test=float(np.mean(orders.test_relRMSE)),
            flow_relRMSE_test_median=float(np.median(orders.test_relRMSE))))
    imb = pd.concat(imb_all, ignore_index=True)
    ident = pd.concat(ident_all, ignore_index=True)
    orders = pd.concat(orders_all, ignore_index=True)

    ude_imb, ude_ident = ude_reference(data, n_train)
    ude_flow = ude_flow_error(data, n_train)

    os.makedirs(args.out, exist_ok=True)
    consist.to_csv(os.path.join(args.out, "wp1f_data_consistency.csv"), index=False)
    imb.to_csv(os.path.join(args.out, "wp1f_mass_imbalance.csv"), index=False)
    ident.to_csv(os.path.join(args.out, "wp1f_identity_violation.csv"), index=False)
    orders.to_csv(os.path.join(args.out, "wp1f_ar_orders.csv"), index=False)

    fig_imbalance(imb, ude_imb, data, n_train,
                  os.path.join(args.out, "wp1f_mass_imbalance.png"))
    fig_identities(ident, ude_ident,
                   os.path.join(args.out, "wp1f_identity_violation.png"))

    pd.set_option("display.width", 200, "display.max_columns", 40)
    print(f"\n--- selected orders, {PRIMARY_SPEC} ---")
    o = orders[orders.spec == PRIMARY_SPEC]
    print(o[["flow", "lag_order", "n_drivers", "test_relRMSE"]]
          .round(2).to_string(index=False))

    print("\n--- per-year mass imbalance, kt, test window ---")
    p = (imb[imb.model == PRIMARY_SPEC]
         .pivot(index="year", columns="stock", values="imbalance_kt")[STOCK_NAMES])
    print(p.round(1).to_string())

    print("\n--- cumulative imbalance over the test window (kt, and % of the "
          "reported 2019 stock) ---")
    term = (imb.sort_values("year").groupby(["model", "stock"])
            .agg(cumulative_kt=("cumulative_imbalance_kt", "last"),
                 pct_of_stock=("observed_stock_pct", "last"),
                 observed_2019_kt=("observed_stock_kt", "last")).reset_index())
    print(term[term.model == PRIMARY_SPEC].round(1).to_string(index=False))

    print("\n--- robustness: terminal cumulative imbalance by specification (kt) ---")
    print(term.pivot(index="model", columns="stock", values="cumulative_kt")
              [STOCK_NAMES].round(1).to_string())
    print("\n--- forecast quality on the SAME window: mean / median per-flow "
          "test relRMSE% ---")
    fe = pd.DataFrame(flowerr)
    if ude_flow is not None:
        per_seed = (ude_flow.groupby("seed").test_relRMSE
                    .agg(["mean", "median"]).reset_index())
        fe = pd.concat([fe, pd.DataFrame([dict(
            model=f"UDE (testrun, {per_seed.seed.nunique()} seeds, median seed)",
            flow_relRMSE_test=float(per_seed["mean"].median()),
            flow_relRMSE_test_median=float(per_seed["median"].median()))])],
            ignore_index=True)
        ude_flow.to_csv(os.path.join(args.out, "wp1f_flow_error.csv"), index=False)
    print(fe.round(2).to_string(index=False))

    print("\n--- identity violations, kt (mean |violation| over the test window) ---")
    iv = (ident.groupby(["model", "identity"]).violation_kt
          .apply(lambda v: float(np.mean(np.abs(v)))).reset_index())
    print(iv.pivot(index="model", columns="identity", values="violation_kt")
            .round(1).to_string())

    if ude_imb is not None:
        u = (ude_imb.sort_values("year").groupby(["seed", "stock"])
             .cumulative_imbalance_kt.last().reset_index())
        print(f"\n--- UDE reference, {u.seed.nunique()} seeds ---")
        print("max |cumulative imbalance| over all seeds and stocks: "
              f"{float(np.max(np.abs(u.cumulative_imbalance_kt))):.3e} kt")
        print("max |identity violation| over all seeds: "
              f"{float(np.max(np.abs(ude_ident.violation_kt))):.3e} kt")

    print(f"\nwrote {args.out}/wp1f_mass_imbalance.{{csv,png,pdf}}, "
          f"wp1f_identity_violation.{{csv,png,pdf}}, wp1f_ar_orders.csv, "
          f"wp1f_data_consistency.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
