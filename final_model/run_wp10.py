#!/usr/bin/env python3
"""
run_wp10.py — WP-10, the Erlang use-phase extension
===================================================

Author-gated in `EXPERIMENTS_SPEC.md`; run under explicit authorisation.
All behaviour changes go through `zinc_erlang_lab.install()` — `zinc_colloc_v5.py`
is imported and never edited (CLAUDE.md rule 1), and its MD5 is pinned.

The claim under test
--------------------
Chapter 2 (`paper_revised.tex`, §4 and App. `subsec:cohort`) proves that,
holding the mean product lifetime fixed, the shape of the residence-time
density leaves the equilibrium, the asymptotic stability, the dominant
oscillation and the circularity indicators exactly unchanged, and that it
governs *only* the in-use relaxation mode and the routing of material into
waste management:

    "The distribution shape governs only the in-use relaxation mode and the
     fast, damped routing of material into waste management, which is why
     s_5 is the one stock visibly sensitive to it."

`s_5` is this model's **Scrap** — 51.0% freerun relRMSE, the one observable
the published model fits worse than persistence (31.4%).  So Chapter 2
predicts the shape can move exactly the stock that is broken, and nothing
else.  WP-10 measures whether it does, and by how much.

Parts
-----
1  structure          the arm table, Erlang densities, TV distances.  No fit.
2  shape counterfactual   re-integrate the 35 PUBLISHED weight vectors under
                      n = 1, 5, 33 with the mean lifetime pinned at Rostek.
                      This is Chapter 2's ceteris paribus exactly: the
                      network, every alpha, every tau and every mean lifetime
                      are held fixed and only the density shape changes, so
                      whatever Scrap does is the structural effect with the
                      re-estimation effect removed by construction.  No fits.
3  refit arms         expk / erl5 / erl33 / erl33k, paired by seed against
                      anchor_v4.  This is the deliverable the spec asks for,
                      and it confounds shape with re-estimation — which is
                      why Part 2 exists to separate them.
4  fitted lifetimes   for the learn-kappa arms: tau_z = mu_z exp(-log_kappa_z)
                      against Rostek, with the adamw shrinkage stated.

Every part writes a CSV under `analysis/`; Parts 1–3 also write a figure.

Usage
-----
    python run_wp10.py --check
    python run_wp10.py --part 1
    python run_wp10.py --part 2
    python run_wp10.py --part 3 --arms erl33,erl5,expk,erl33k --seeds 0-7
    python run_wp10.py --part 4
"""

import argparse
import json
import os
import resource
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import zinc_erlang_lab as lab                                     # noqa: E402

OUT = os.path.join(HERE, "analysis")
FIGDIR = OUT
RSS_LIMIT_MB = 48_000.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def hodges_lehmann(v, conf=0.95):
    """HL location estimate + distribution-free CI.  Same implementation as
    `run_wp1b.py` / `run_wp2b.py` / `run_wp11c.py` / `zinc_circ_lab.py`."""
    from scipy import stats

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


def iqr_row(v):
    v = np.asarray([x for x in np.asarray(v, float) if np.isfinite(x)])
    if v.size == 0:
        return dict(median=np.nan, q1=np.nan, q3=np.nan, n=0)
    return dict(median=float(np.median(v)), q1=float(np.percentile(v, 25)),
                q3=float(np.percentile(v, 75)), n=int(v.size))


def _rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024.0 ** 2) if sys.platform == "darwin" else r / 1024.0


def rss_watchdog(tag=""):
    """CLAUDE.md rule 6.  Long runs die to the OOM killer without this."""
    mb = _rss_mb()
    if mb > RSS_LIMIT_MB:
        raise MemoryError(f"RSS {mb:.0f} MB > {RSS_LIMIT_MB:.0f} MB limit {tag}")
    return mb


def parse_seeds(spec):
    out = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return sorted(set(out))


def _write(df, name):
    path = os.path.join(OUT, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  wrote {os.path.relpath(path, HERE)}  ({len(df)} rows)")
    return path


def _savefig(fig, stem):
    for ext in ("png", "pdf"):
        p = os.path.join(FIGDIR, f"{stem}.{ext}")
        fig.savefig(p, dpi=160, bbox_inches="tight")
    print(f"  wrote {stem}.{{png,pdf}}")


# ---------------------------------------------------------------------------
# Part 1 — structure (no fit)
# ---------------------------------------------------------------------------
def part1():
    """The arm table and the lifetime densities it implies."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lab.integrity_check()
    rows = []
    for a in lab.ARM_SPEC:
        rows.extend(lab.arm_structure(a))
    df = pd.DataFrame(rows)
    _write(df, "wp10_structure.csv")

    # Ch2 Table tab:fiterr is a TV distance against the Gloser reference; the
    # zinc analogue we can compute without that reference is the TV distance
    # of each Erlang against the exponential it replaces, at matched mean.
    mu = [10.0, 20.0, 44.0]
    names = ["short (mu=10 yr)", "medium (mu=20 yr)", "long (mu=44 yr)"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for z, (ax, m, nm) in enumerate(zip(axes, mu, names)):
        ell = np.linspace(0.0, 4.0 * m, 2000)
        for n_z, ls, c in ((1, "-", "0.25"), (5, "--", "#c0392b"),
                           (33, "-", "#1f77b4")):
            g = lab.lifetime_density(n_z, m, ell)
            ax.plot(ell, g, ls, color=c, lw=1.8,
                    label=f"n={n_z}  (CV={1/np.sqrt(n_z):.2f})")
        ax.axvline(m, color="k", lw=0.8, ls=":", alpha=0.6)
        ax.set_title(nm, fontsize=10)
        ax.set_xlabel("product age $\\ell$ (yr)")
        if z == 0:
            ax.set_ylabel("discard density $g(\\ell)$")
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle("WP-10 — Erlang serial chains at matched mean lifetime "
                 "(Ch2 Eq. erlang_density)", fontsize=11)
    _savefig(fig, "wp10_densities")
    plt.close(fig)

    print("\n  arm table")
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Part 2 — shape counterfactual on the published weights (no refit)
# ---------------------------------------------------------------------------
def part2(seeds, arms=("exp", "erl5", "erl33")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import zinc_colloc_v5 as v5

    t0 = time.time()
    rows, traj = lab.shape_counterfactual(seeds, arms=arms)
    df = pd.DataFrame(rows)
    _write(df, "wp10_shape_counterfactual_per_seed.csv")
    print(f"  {len(seeds)} seeds x {len(arms)} arms in {time.time()-t0:.0f}s "
          f"(RSS {rss_watchdog('part2'):.0f} MB)", flush=True)

    # summary: median +- IQR per (arm, window, component), and the PAIRED
    # delta against the n=1 arm on the same seed.
    key = ["window", "family", "component"]
    base = df[df.arm == "exp"].set_index(key + ["seed"])["relRMSE"]
    summ, pair = [], []
    for a in arms:
        sub = df[df.arm == a]
        for (win, fam, comp), g in sub.groupby(key):
            summ.append(dict(arm=a, window=win, family=fam, component=comp,
                             **iqr_row(g["relRMSE"].values)))
            if a == "exp":
                continue
            gg = g.set_index(key + ["seed"])["relRMSE"]
            d = (gg - base.reindex(gg.index)).dropna().values
            if not len(d):
                continue
            hl = hodges_lehmann(d)
            pair.append(dict(arm=a, window=win, family=fam, component=comp,
                             delta_median=float(np.median(d)), hl=hl["hl"],
                             lo=hl["lo"], hi=hl["hi"], n=hl["n"],
                             n_seeds_improved=int((d < 0).sum()),
                             ci_excludes_zero=bool(np.isfinite(hl["lo"])
                                                   and hl["lo"] * hl["hi"] > 0)))
    _write(pd.DataFrame(summ), "wp10_shape_counterfactual_summary.csv")
    dfp = pd.DataFrame(pair)
    _write(dfp, "wp10_shape_counterfactual_paired.csv")

    fit = lab.init_fit()
    years = np.asarray(fit.data_all["years"], float).ravel()
    S_obs = np.asarray(fit.data_all["stocks_obs"], float)
    np.savez_compressed(
        os.path.join(OUT, "wp10_shape_counterfactual.npz"),
        years=years, arms=np.array(list(arms), dtype=object),
        seeds=np.asarray(sorted(seeds)), stocks_obs=S_obs,
        **{f"S_{a}": np.stack([traj[a][s][0] for s in sorted(seeds)]) for a in arms},
        **{f"F_{a}": np.stack([traj[a][s][1] for s in sorted(seeds)]) for a in arms},
        **{f"C_{a}": np.stack([traj[a][s][2] for s in sorted(seeds)]) for a in arms})
    print("  wrote analysis/wp10_shape_counterfactual.npz")

    # ---- figure ----------------------------------------------------------
    colours = {"exp": "0.25", "erl5": "#c0392b", "erl33": "#1f77b4"}
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.1))
    for k, ki in enumerate((3, 2)):                      # Scrap, In-Use
        ax = axes[k]
        for a in arms:
            A = np.stack([traj[a][s][0][:, ki] for s in sorted(seeds)])
            ax.fill_between(years, np.percentile(A, 25, 0), np.percentile(A, 75, 0),
                            color=colours.get(a, "0.5"), alpha=0.18, lw=0)
            ax.plot(years, np.median(A, 0), color=colours.get(a, "0.5"), lw=1.8,
                    label=a)
        ax.plot(years, S_obs[:, ki], "k.", ms=4, label="observed")
        ax.axvspan(2007, years[-1], color="0.85", alpha=0.35, lw=0, zorder=0)
        ax.set_title(f"{v5.STOCK_NAMES[ki]} — free run, published weights",
                     fontsize=10)
        ax.set_xlabel("year")
        if k == 0:
            ax.set_ylabel("stock (kt)")
            ax.legend(fontsize=8, frameon=False)

    ax = axes[2]
    comps = list(v5.STOCK_NAMES)
    sub = dfp[(dfp.window == "test") & (dfp.family == "stock")]
    ypos = np.arange(len(comps))
    for j, a in enumerate([x for x in arms if x != "exp"]):
        s2 = sub[sub.arm == a].set_index("component")
        v = [s2.loc[c, "hl"] for c in comps]
        lo = [s2.loc[c, "hl"] - s2.loc[c, "lo"] for c in comps]
        hi = [s2.loc[c, "hi"] - s2.loc[c, "hl"] for c in comps]
        ax.errorbar(v, ypos + 0.16 * (j - 0.5), xerr=[lo, hi], fmt="o", ms=5,
                    capsize=3, color=colours.get(a, "0.5"), label=a)
    ax.axvline(0.0, color="k", lw=0.9)
    ax.set_yticks(ypos); ax.set_yticklabels(comps); ax.invert_yaxis()
    ax.set_xlabel("paired $\\Delta$ relRMSE vs n=1, test window (pp)\n[HL, 95% CI]")
    ax.set_title("shape effect at fixed mean lifetime", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.suptitle("WP-10 Part 2 — Chapter 2's ceteris paribus: same weights, same "
                 "mean lifetimes, only the density shape changes", fontsize=11)
    _savefig(fig, "wp10_shape_counterfactual")
    plt.close(fig)

    show = dfp[(dfp.window == "test") & (dfp.family.isin(("stock", "stock_mean",
                                                          "flow_mean")))]
    print("\n  paired delta vs n=1, TEST window (negative = Erlang better)")
    print(show.to_string(index=False), flush=True)
    return df, dfp


# ---------------------------------------------------------------------------
# Part 3 — refit arms
# ---------------------------------------------------------------------------
def _fit_one(arm_name, seed, cfg, v5):
    """One fit under one arm.  Returns (metric rows, per-component rows, extras)."""
    import jax

    lab.arm(arm_name)
    t0 = time.time()
    fit = v5.run(f"wp10_{arm_name}_s{seed}", **dict(cfg, seed=int(seed)))
    wall = time.time() - t0

    per_seed, per_comp = [], []
    for kind in ("freerun", "testrun"):
        rows = fit.diagnose(kind, verbose=False, return_rows=True,
                            include_pinned=False)
        d = pd.DataFrame(rows)
        if (d["stage"] == "B").any():
            d = d[d["stage"] == "B"]
        rec = dict(arm=arm_name, seed=int(seed), rollout=kind,
                   wall_s=round(wall, 1))
        for fam in ("stock", "flow", "alpha", "tau"):
            r = d[d["kind"] == f"{fam}_mean"]
            if len(r):
                rec[f"{fam}_relRMSE"] = float(r.iloc[0]["test_relRMSE%"])
                rec[f"{fam}_MAPE"] = float(r.iloc[0]["test_MAPE%"])
        per_seed.append(rec)
        for _, r in d[d["kind"].isin(("stock", "flow", "alpha", "tau"))].iterrows():
            per_comp.append(dict(
                arm=arm_name, seed=int(seed), rollout=kind, family=str(r["kind"]),
                component=str(r["component"]).replace(" [testrun]", "").strip(),
                relRMSE=float(r["test_relRMSE%"]), MAPE=float(r["test_MAPE%"])))

    p = fit._params_for("B")
    extras = dict(arm=arm_name, seed=int(seed), wall_s=round(wall, 1))
    if isinstance(p[-1], dict) and "log_kappa" in p[-1]:
        g = np.asarray(p[-1]["log_kappa"], float)
        mu = np.asarray(v5.MU_COHORTS_NP, float)
        n_z = np.asarray(lab.armed()["n_chain"], float)
        for z, nm in enumerate(("short", "medium", "long")):
            extras[f"log_kappa_{nm}"] = float(g[z])
            extras[f"tau_fitted_{nm}_yr"] = float(mu[z] * np.exp(-g[z]))
            extras[f"kappa_fitted_{nm}"] = float(n_z[z] * np.exp(g[z]) / mu[z])

    pr = fit.predictions("B")
    f_idx = np.asarray(fit.flow_obs_to_pred_idx, int)
    arrays = dict(
        years_all=np.asarray(fit.data_all["years"], float).ravel(),
        stocks_obs=np.asarray(fit.data_all["stocks_obs"], float),
        S_pred_B=np.asarray(pr["S_pred"], float),
        S_cohorts_B=np.asarray(pr["S_cohorts"], float),
        alphas_B=np.asarray(pr["alphas"], float),
        taus_B=np.asarray(pr["taus"], float),
        f_cohort_B=np.asarray(pr["f_cohort"], float),
        flows_obs=np.asarray(fit.data_all["flows_obs"], float),
        F_pred_B=np.asarray(pr["F_int"], float)[:, f_idx],
        flow_names=np.array([v5.FLOW_NAMES[i] for i in f_idx], dtype=object))
    try:
        pr_t = fit._predictions_testrun("B")
        arrays.update(
            years_testrun=np.asarray(fit.data_test["years"], float).ravel(),
            stocks_obs_testrun=np.asarray(fit.data_test["stocks_obs"], float),
            S_pred_testrun=np.asarray(pr_t["S_pred"], float),
            F_pred_testrun=np.asarray(pr_t["F_int"], float)[:, f_idx])
    except Exception as e:
        print(f"    note: no testrun rollout ({type(e).__name__}: {e})")
    for i, layer in enumerate(p):
        for key in ("W", "b"):
            arrays[f"param{i}_{key}"] = np.asarray(layer[key], float)
    if isinstance(p[-1], dict) and "log_kappa" in p[-1]:
        arrays["log_kappa"] = np.asarray(p[-1]["log_kappa"], float)

    d = os.path.join(OUT, "wp10", arm_name)
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, f"pred_seed{seed}.npz"), **arrays)

    jax.clear_caches()                       # CLAUDE.md rule 6
    return per_seed, per_comp, extras


def part3(arms, seeds, resume=True):
    import zinc_colloc_v5 as v5

    lab.integrity_check()
    lab.install(v5)
    cfg = lab.load_anchor_config()
    cfg["verbose"] = False        # verbose=True activates Stage B select (WP-1d flag 3)

    os.makedirs(os.path.join(OUT, "wp10"), exist_ok=True)
    jsonl = os.path.join(OUT, "wp10", "fits.jsonl")
    done = set()
    if resume and os.path.exists(jsonl):
        with open(jsonl) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    done.add((r["arm"], int(r["seed"])))
                except Exception:
                    pass
        print(f"  resume: {len(done)} (arm, seed) fits already on disk")

    per_seed, per_comp, extras = [], [], []
    # seed-major: every arm at seed 0, then every arm at seed 1, ...  A run cut
    # short by walltime then leaves a BALANCED design across arms rather than
    # a complete first arm and nothing else.
    for sd in seeds:
        for a in arms:
            if (a, sd) in done:
                continue
            print(f"[{a} seed {sd}] fitting …", flush=True)
            try:
                ps, pc, ex = _fit_one(a, sd, cfg, v5)
            except Exception as e:
                print(f"  [{a} seed {sd}] FAILED: {type(e).__name__}: {e}")
                continue
            per_seed += ps
            per_comp += pc
            extras.append(ex)
            # mirror per-fit as JSONL with fsync (CLAUDE.md rule 7)
            with open(jsonl, "a") as fh:
                for r in ps:
                    fh.write(json.dumps(r) + "\n")
                fh.write(json.dumps(dict(ex, kind="extras")) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            fr = [r for r in ps if r["rollout"] == "freerun"][0]
            print(f"  [{a} seed {sd}] {ex['wall_s']/60:.1f} min  "
                  f"freerun stock={fr.get('stock_relRMSE', float('nan')):.2f}%  "
                  f"alpha={fr.get('alpha_relRMSE', float('nan')):.2f}%  "
                  f"RSS {rss_watchdog(a):.0f} MB", flush=True)

    if per_seed:
        _append(pd.DataFrame(per_seed), "wp10_refit_per_seed.csv",
                ["arm", "seed", "rollout"])
    if per_comp:
        _append(pd.DataFrame(per_comp), "wp10_refit_per_component.csv",
                ["arm", "seed", "rollout", "family", "component"])
    if extras:
        _append(pd.DataFrame(extras), "wp10_refit_lifetimes.csv", ["arm", "seed"])
    return per_seed


def _append(df, name, keys):
    path = os.path.join(OUT, name)
    if os.path.exists(path):
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
        df = df.drop_duplicates(subset=keys, keep="last")
    df = df.sort_values(keys).reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"  wrote {os.path.relpath(path, HERE)}  ({len(df)} rows)")


# ---------------------------------------------------------------------------
# Part 4 — summary of the refit arms against anchor_v4, paired by seed
# ---------------------------------------------------------------------------
def part4():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import zinc_colloc_v5 as v5

    ps_path = os.path.join(OUT, "wp10_refit_per_seed.csv")
    pc_path = os.path.join(OUT, "wp10_refit_per_component.csv")
    if not os.path.exists(ps_path):
        print("  no refit results yet — run --part 3 first")
        return None
    ps = pd.read_csv(ps_path)
    pc = pd.read_csv(pc_path)

    # anchor_v4 baseline, restricted to the seeds the arms actually used
    anc_seed = pd.read_csv(os.path.join(HERE, "anchor_v4", "per_seed.csv"))
    anc_stock = pd.read_csv(os.path.join(HERE, "anchor_v4", "per_stock.csv"))
    anc_seed["arm"] = "anchor_v4"
    anc_stock["family"] = "stock"

    rows, paired = [], []
    for rollout in ("freerun", "testrun"):
        base = anc_seed[anc_seed.rollout == rollout].set_index("seed")
        for a in sorted(ps.arm.unique()):
            sub = ps[(ps.arm == a) & (ps.rollout == rollout)].set_index("seed")
            seeds = sorted(set(sub.index) & set(base.index))
            for fam in ("stock", "flow", "alpha", "tau"):
                col = f"{fam}_relRMSE"
                rows.append(dict(arm=a, rollout=rollout, family=fam,
                                 **iqr_row(sub[col].values)))
                d = (sub.loc[seeds, col] - base.loc[seeds, col]).values
                hl = hodges_lehmann(d)
                paired.append(dict(arm=a, rollout=rollout, family=fam,
                                   delta_median=float(np.median(d)) if len(d) else np.nan,
                                   hl=hl["hl"], lo=hl["lo"], hi=hl["hi"], n=hl["n"],
                                   n_improved=int((d < 0).sum())))
            # per-stock
            sc = pc[(pc.arm == a) & (pc.rollout == rollout) & (pc.family == "stock")]
            bs = anc_stock[anc_stock.rollout == rollout]
            for nm in v5.STOCK_NAMES:
                x = sc[sc.component == nm].set_index("seed")["relRMSE"]
                y = bs[bs.component == nm].set_index("seed")["relRMSE"]
                sd_common = sorted(set(x.index) & set(y.index))
                rows.append(dict(arm=a, rollout=rollout, family=f"stock:{nm}",
                                 **iqr_row(x.values)))
                if sd_common:
                    d = (x.loc[sd_common] - y.loc[sd_common]).values
                    hl = hodges_lehmann(d)
                    paired.append(dict(arm=a, rollout=rollout, family=f"stock:{nm}",
                                       delta_median=float(np.median(d)),
                                       hl=hl["hl"], lo=hl["lo"], hi=hl["hi"],
                                       n=hl["n"], n_improved=int((d < 0).sum())))
            # per-alpha channel
            ac = pc[(pc.arm == a) & (pc.rollout == rollout) & (pc.family == "alpha")]
            for nm in sorted(ac.component.unique()):
                rows.append(dict(arm=a, rollout=rollout, family=f"alpha:{nm}",
                                 **iqr_row(ac[ac.component == nm]["relRMSE"].values)))

    dfr = pd.DataFrame(rows)
    dfp = pd.DataFrame(paired)
    _write(dfr, "wp10_refit_summary.csv")
    _write(dfp, "wp10_refit_paired.csv")

    lt_path = os.path.join(OUT, "wp10_refit_lifetimes.csv")
    if os.path.exists(lt_path):
        lt = pd.read_csv(lt_path)
        cols = [c for c in lt.columns if c.startswith("tau_fitted_")]
        if cols:
            srows = []
            for a in sorted(lt.arm.unique()):
                sub = lt[lt.arm == a]
                for c in cols:
                    if sub[c].notna().any():
                        srows.append(dict(arm=a, quantity=c, **iqr_row(sub[c].values)))
            if srows:
                _write(pd.DataFrame(srows), "wp10_fitted_lifetimes_summary.csv")
                print("\n  fitted mean lifetimes (yr), median [IQR]")
                print(pd.DataFrame(srows).to_string(index=False))

    # ---- figure ----------------------------------------------------------
    fams = ["stock:Scrap", "stock:In-Use", "stock:Refined", "stock:Concentrate",
            "stock", "flow", "alpha", "tau"]
    sub = dfp[(dfp.rollout == "freerun") & (dfp.family.isin(fams))]
    if len(sub):
        arms = sorted(sub.arm.unique())
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        ypos = np.arange(len(fams))
        cmap = plt.get_cmap("tab10")
        for j, a in enumerate(arms):
            s = sub[sub.arm == a].set_index("family")
            v = [s.loc[f, "hl"] if f in s.index else np.nan for f in fams]
            lo = [s.loc[f, "hl"] - s.loc[f, "lo"] if f in s.index else np.nan for f in fams]
            hi = [s.loc[f, "hi"] - s.loc[f, "hl"] if f in s.index else np.nan for f in fams]
            ax.errorbar(v, ypos + 0.18 * (j - (len(arms) - 1) / 2),
                        xerr=[lo, hi], fmt="o", ms=5, capsize=3,
                        color=cmap(j), label=a)
        ax.axvline(0.0, color="k", lw=0.9)
        ax.set_yticks(ypos); ax.set_yticklabels(fams); ax.invert_yaxis()
        ax.set_xlabel("paired $\\Delta$ relRMSE vs anchor_v4 (pp)   [HL, 95% CI]")
        ax.set_title("WP-10 Part 3 — Erlang refit arms, freerun, paired by seed",
                     fontsize=11)
        ax.legend(fontsize=8, frameon=False)
        _savefig(fig, "wp10_refit_paired")
        plt.close(fig)

    print("\n  paired vs anchor_v4 (freerun; negative = arm is better)")
    print(dfp[dfp.rollout == "freerun"].to_string(index=False))
    return dfr, dfp


# ---------------------------------------------------------------------------
# Part 5 — the WP-8b discriminator
# ---------------------------------------------------------------------------
def part5():
    """WP-8b handed WP-10 a test; this evaluates it, and corrects it.

    `wp8_findings.md` §8b ends:

        "Chapter 2's Erlang chain -- mode at (n-1)/kappa > 0 -- would put a
         genuine hump near each mean lifetime *in the single-passage response
         itself*.  So the kernel is a discriminator between the two use-phase
         parameterisations, and running it on an Erlang refit is a cheap and
         decisive test."

    The mode at `(n-1)/kappa` is the mode of the Erlang *density* g(l)
    (Ch2 Eq. erlang_density) -- that is, of the DISCHARGE out of use.  The
    single-passage response of the in-use STOCK to an inflow pulse is not the
    density but the survival function `P(lifetime > l) = 1 - G(l)`, and a
    survival function is non-increasing by definition, for every n.  So:

      * on the In-Use stock the kernel does NOT discriminate: no Erlang order
        can produce an interior maximum there, so 8b's measured hump at lag
        26 yr is not a lifetime-shape artefact and its recirculation reading
        survives the extension -- strengthened, because the alternative
        explanation is now excluded rather than merely untested;
      * on the discharge -- end_of_life, and hence the Scrap inflow -- it
        discriminates sharply: the mode moves from 0 (exponential) to
        mu (1 - 1/n).

    Which is Chapter 2's own conclusion arrived at from the other side:
    "s_5 is the one stock visibly sensitive to" the shape.

    Both kernels are computed here in closed form on the published fitted
    cohort split, checked against a direct numerical integration of the chain,
    and compared with 8b's stored measured kernel.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lab.integrity_check()
    mu = np.array([10.0, 20.0, 44.0])

    # fitted cohort split, published weights, median over the 35 seeds
    sig = pd.read_csv(os.path.join(OUT, "wp8b_cohort_signature.csv"))
    f_coh = np.asarray(json.loads(sig.loc[0, "f_cohort_mean"]), float)
    f_coh = f_coh / f_coh.sum()
    mean_life = float((f_coh * mu).sum())

    lags = np.linspace(0.0, 39.0, 3901)
    rows, curves = [], {}
    for n_z in (1, 5, 33):
        # discharge kernel: mixture of Erlang densities (Ch2 Eq. erlang_density)
        h = sum(f_coh[z] * lab.lifetime_density(n_z, mu[z], lags) for z in range(3))
        # in-use stock kernel: mixture of Erlang survival functions.
        # P(lifetime > l) = P(N(kappa l) < n) for N Poisson -- non-increasing
        # in l for every n, which is the whole point of this part.
        from scipy import stats
        surv = np.zeros_like(lags)
        for z in range(3):
            kappa = n_z / mu[z]
            surv += f_coh[z] * stats.poisson.cdf(n_z - 1, kappa * lags)
        curves[n_z] = (h, surv)

        def _interior_max(y):
            i = np.arange(1, len(y) - 1)
            m = i[(y[i] > y[i - 1]) & (y[i] >= y[i + 1]) & (y[i] > 1e-12)]
            return [float(lags[k]) for k in m]

        rows.append(dict(
            n_chain=n_z,
            discharge_mode_yr=float(lags[int(np.argmax(h))]),
            discharge_mode_theory_yr=float(0.0 if n_z == 1 else np.nan),
            discharge_interior_maxima=str([round(x, 2) for x in _interior_max(h)]),
            stock_kernel_monotone=bool(np.all(np.diff(surv) <= 1e-12)),
            stock_interior_maxima=str([round(x, 2) for x in _interior_max(surv)]),
            stock_kernel_lag26=float(np.interp(26.0, lags, surv)),
            mean_lifetime_yr=mean_life,
            cohort_modes_yr=str([round(float(mu[z] * (1 - 1.0 / n_z)), 2)
                                 for z in range(3)])))
    df = pd.DataFrame(rows)
    _write(df, "wp10_wp8b_discriminator.csv")

    # ---- numerical check of the chain against the closed form -------------
    # Integrate du/dt for a unit impulse with an explicit RK4, and compare the
    # discharge u_n/stage_time against the Erlang density.  This validates the
    # lab's chain construction independently of diffrax.
    chk = []
    for n_z in (1, 5, 33):
        for z in range(3):
            st = mu[z] / n_z
            dt, T = 0.001, 4.0 * mu[z]
            u = np.zeros(n_z); u[0] = 1.0 / st * dt   # unit mass into stage 1
            u = np.zeros(n_z); u[0] = 1.0
            t, out_t, out_v = 0.0, [], []

            def f(u):
                o = u / st
                return np.concatenate([[-o[0]], o[:-1] - o[1:]])
            while t < T:
                k1 = f(u); k2 = f(u + 0.5 * dt * k1)
                k3 = f(u + 0.5 * dt * k2); k4 = f(u + dt * k3)
                u = u + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
                t += dt
                out_t.append(t); out_v.append(u[-1] / st)
            out_t, out_v = np.array(out_t), np.array(out_v)
            ref = lab.lifetime_density(n_z, mu[z], out_t)
            chk.append(dict(n_chain=n_z, cohort=["short", "medium", "long"][z],
                            mu_yr=mu[z],
                            max_abs_err=float(np.max(np.abs(out_v - ref))),
                            max_rel_err=float(np.max(np.abs(out_v - ref))
                                              / max(ref.max(), 1e-12)),
                            numeric_mode_yr=float(out_t[int(np.argmax(out_v))]),
                            theory_mode_yr=float(mu[z] * (1 - 1.0 / n_z))))
    dchk = pd.DataFrame(chk)
    _write(dchk, "wp10_chain_validation.csv")

    # ---- figure -----------------------------------------------------------
    meas = np.array([float(sig.loc[0, f"measured_lag{l}"]) for l in range(0, 40, 2)])
    theo = np.array([float(sig.loc[0, f"theory_lag{l}"]) for l in range(0, 40, 2)])
    mlags = np.arange(0, 40, 2)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    ax = axes[0]
    for n_z, c in ((1, "0.25"), (5, "#c0392b"), (33, "#1f77b4")):
        ax.plot(lags, curves[n_z][1], color=c, lw=1.8, label=f"single passage, n={n_z}")
    ax.plot(mlags, meas / meas[0], "k.-", ms=5, lw=1.2,
            label="WP-8b measured In-Use kernel")
    ax.axvline(26, color="k", ls=":", lw=0.9)
    ax.text(26.4, 0.9, "8b secondary\nmaximum, 26 yr", fontsize=7.5, va="top")
    ax.set_xlabel("lag (yr)"); ax.set_ylabel("normalised response")
    ax.set_title("In-Use stock: no Erlang order has an interior maximum",
                 fontsize=10)
    ax.legend(fontsize=7.5, frameon=False)

    ax = axes[1]
    for n_z, c in ((1, "0.25"), (5, "#c0392b"), (33, "#1f77b4")):
        ax.plot(lags, curves[n_z][0], color=c, lw=1.8, label=f"n={n_z}")
    for z in range(3):
        ax.axvline(mu[z], color="0.6", ls=":", lw=0.8)
    ax.set_xlabel("lag (yr)"); ax.set_ylabel("discharge density")
    ax.set_title("End-of-life discharge: the mode moves to $\\mu(1-1/n)$",
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.suptitle("WP-10 Part 5 — the WP-8b discriminator, evaluated", fontsize=11)
    _savefig(fig, "wp10_wp8b_discriminator")
    plt.close(fig)

    print("\n  kernels")
    print(df.to_string(index=False))
    print("\n  chain validation (closed form vs RK4)")
    print(dchk.to_string(index=False))
    return df, dchk


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="WP-10 — Erlang use-phase extension")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--verify-noop", action="store_true")
    ap.add_argument("--verify-gradient", action="store_true")
    ap.add_argument("--part", default=None,
                    help="1=structure 2=shape counterfactual 3=refits 4=summary; "
                         'comma-separated, or "all"')
    ap.add_argument("--arms", default="erl33,erl5,expk,erl33k")
    ap.add_argument("--seeds", default="0-34")
    ap.add_argument("--refit-seeds", default="0-7")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)

    if args.check:
        lab.check()
        return 0
    if args.verify_noop:
        print("verify_noop — re-integrating published weights at n=(1,1,1)")
        rows = lab.verify_noop(parse_seeds("0-4"))
        ok = all(r["exact"] for r in rows) and all(r["rehydration_exact"] for r in rows)
        _write(pd.DataFrame(rows), "wp10_noop_check.csv")
        print(f"\n  {'PASS — bitwise' if ok else 'FAIL'}")
        return 0 if ok else 1

    if args.verify_gradient:
        print("verify_gradient — is log_kappa trained, and is its gradient right?")
        rows = pd.DataFrame(lab.verify_gradient())
        _write(rows, "wp10_gradient_check.csv")
        print(rows.to_string(index=False))
        return 0

    parts = ["1", "2", "3", "4", "5"] if args.part in (None, "all") else \
        [p.strip() for p in str(args.part).split(",")]
    seeds = parse_seeds(args.seeds)
    refit_seeds = parse_seeds(args.refit_seeds)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    if "1" in parts:
        print("\n== Part 1 — structure ==")
        part1()
    if "2" in parts:
        print("\n== Part 2 — shape counterfactual (no refit) ==")
        part2(seeds)
    if "3" in parts:
        print(f"\n== Part 3 — refit arms {arms} on seeds {refit_seeds} ==")
        part3(arms, refit_seeds, resume=not args.no_resume)
    if "4" in parts:
        print("\n== Part 4 — summary vs anchor_v4 ==")
        part4()
    if "5" in parts:
        print("\n== Part 5 — the WP-8b discriminator ==")
        part5()
    return 0


if __name__ == "__main__":
    sys.exit(main())
