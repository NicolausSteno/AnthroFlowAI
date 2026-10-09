#!/usr/bin/env python3
"""
run_wp10b.py -- WP-10 follow-up: zinc's own Erlang order, and a mean lifetime
that is actually free to move
============================================================================

Two defects in WP-10's design (see analysis/notes/wp10b_findings.md):

1. n = 33 was transported from copper (Gloser "moderate", CV 0.175).  Zinc's
   own lifetimes, Rostek 2022 SI Table S4, grouped to the three cohorts as in
   zinc_pinn_v12b_1.py, give mixture CVs 0.41 / 0.40 / 0.37 -> n = (6, 6, 7).
2. The learn-kappa arms could move the mean lifetime by at most +-9.5 %
   (Adam step budget, wp10_findings §1.3).  Under growing inflow a narrower
   shape holds more stock at fixed mean, and matching the exponential stock
   needs the means 7-35 % shorter -- outside that budget.

Arms (zinc_erlang_lab.ARM_SPEC):
    erlz     n = (6,6,7), kappa fixed at Rostek        zinc shape alone
    erlzk    n = (6,6,7), kappa learned, wide budget   zinc shape + free mean
    expkw    n = (1,1,1), kappa learned, wide budget   control: free mean alone
    erl33kw  n = 33,      kappa learned, wide budget   re-test of erl33k uncapped
plus `erl5` seed 0 re-fitted as a reproduction check of the environment.

Each stream writes its own JSONL under analysis/wp10b/ (fsync per fit), so
concurrent streams never share a file.  `--summary` merges them with the
WP-10 arms and pairs everything against anchor_v4 by seed.

Usage
-----
    python run_wp10b.py --check
    python run_wp10b.py --arms erlz --seeds 0-7          # one stream
    python run_wp10b.py --summary
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import zinc_erlang_lab as lab                                     # noqa: E402
import run_wp10 as w10                                            # noqa: E402

OUT = os.path.join(HERE, "analysis")
OUTB = os.path.join(OUT, "wp10b")
NEW_ARMS = ("erlz", "erlzk", "expkw", "erl33kw")
OLD_ARMS = ("expk", "erl5", "erl33", "erl33k")


def _fit_stream(arms, seeds, tag):
    import zinc_colloc_v5 as v5

    lab.integrity_check()
    lab.install(v5)
    cfg = lab.load_anchor_config()
    cfg["verbose"] = False
    os.makedirs(OUTB, exist_ok=True)
    w10.OUT = OUTB                      # npz dumps -> analysis/wp10b/wp10/<arm>/
    jl = os.path.join(OUTB, f"fits_{tag}.jsonl")
    done = set()
    if os.path.exists(jl):
        for line in open(jl):
            try:
                r = json.loads(line)
                if r.get("kind") == "per_seed":
                    done.add((r["arm"], int(r["seed"])))
            except Exception:
                pass
    for sd in seeds:
        for a in arms:
            if (a, sd) in done:
                continue
            print(f"[{tag}] {a} seed {sd} fitting ...", flush=True)
            try:
                ps, pc, ex = w10._fit_one(a, sd, cfg, v5)
            except Exception as e:
                print(f"  [{a} seed {sd}] FAILED: {type(e).__name__}: {e}", flush=True)
                continue
            s = float(lab.armed().get("kappa_scale", 1.0))
            n_z = np.asarray(lab.armed()["n_chain"], float)
            mu = np.asarray(v5.MU_COHORTS_NP, float)
            for z, nm in enumerate(("short", "medium", "long")):
                k = f"log_kappa_{nm}"
                if k in ex:                         # correct for the scale
                    ge = s * ex[k]
                    ex[f"tau_fitted_{nm}_yr"] = float(mu[z] * np.exp(-ge))
                    ex[f"kappa_fitted_{nm}"] = float(n_z[z] * np.exp(ge) / mu[z])
            ex["kappa_scale"] = s
            ex["n_chain"] = list(map(int, n_z))
            with open(jl, "a") as fh:
                for r in ps:
                    fh.write(json.dumps(dict(r, kind="per_seed")) + "\n")
                for r in pc:
                    fh.write(json.dumps(dict(r, kind="per_comp")) + "\n")
                fh.write(json.dumps(dict(ex, kind="extras")) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            fr = [r for r in ps if r["rollout"] == "freerun"][0]
            print(f"  [{a} seed {sd}] {ex['wall_s']/60:.1f} min  "
                  f"stock={fr.get('stock_relRMSE', np.nan):.2f}%  "
                  f"RSS {w10.rss_watchdog(a):.0f} MB", flush=True)


def _load_new():
    ps, pc, ex = [], [], []
    for f in sorted(os.listdir(OUTB)):
        if not (f.startswith("fits_") and f.endswith(".jsonl")):
            continue
        for line in open(os.path.join(OUTB, f)):
            r = json.loads(line)
            k = r.pop("kind")
            r["stream"] = f[5:-6]
            {"per_seed": ps, "per_comp": pc, "extras": ex}[k].append(r)
    return pd.DataFrame(ps), pd.DataFrame(pc), pd.DataFrame(ex)


def summary():
    import zinc_colloc_v5 as v5

    ps_n, pc_n, ex_n = _load_new()
    rep = ps_n[ps_n.stream == "repro"]
    ps_n, pc_n = ps_n[ps_n.stream != "repro"], pc_n[pc_n.stream != "repro"]
    ex_n = ex_n[ex_n.stream != "repro"]

    # --- reproduction check: erl5 seed 0 here vs the stored WP-10 fit -------
    old_ps = pd.read_csv(os.path.join(OUT, "wp10_refit_per_seed.csv"))
    chk = []
    for _, r in rep.iterrows():
        o = old_ps[(old_ps.arm == r.arm) & (old_ps.seed == r.seed) &
                   (old_ps.rollout == r.rollout)]
        for fam in ("stock", "flow", "alpha", "tau"):
            c = f"{fam}_relRMSE"
            chk.append(dict(arm=r.arm, seed=r.seed, rollout=r.rollout, family=fam,
                            stored=float(o[c].iloc[0]) if len(o) else np.nan,
                            rerun=float(r[c])))
    chk = pd.DataFrame(chk)
    if len(chk):
        chk["abs_diff"] = (chk.rerun - chk.stored).abs()
        w10._write(chk, "wp10b/wp10b_repro_check.csv")

    # --- merge with WP-10 arms ---------------------------------------------
    old_pc = pd.read_csv(os.path.join(OUT, "wp10_refit_per_component.csv"))
    old_ex = pd.read_csv(os.path.join(OUT, "wp10_refit_lifetimes.csv"))
    ps = pd.concat([old_ps[old_ps.arm.isin(OLD_ARMS)], ps_n.drop(columns="stream")],
                   ignore_index=True)
    pc = pd.concat([old_pc[old_pc.arm.isin(OLD_ARMS)], pc_n.drop(columns="stream")],
                   ignore_index=True)
    anc_seed = pd.read_csv(os.path.join(HERE, "anchor_v4", "per_seed.csv"))
    anc_stock = pd.read_csv(os.path.join(HERE, "anchor_v4", "per_stock.csv"))

    rows, paired = [], []
    for rollout in ("freerun", "testrun"):
        base = anc_seed[anc_seed.rollout == rollout].set_index("seed")
        seeds_b = sorted(set(base.index))
        rows += [dict(arm="anchor_v4", rollout=rollout, family=f,
                      **w10.iqr_row(base.loc[[s for s in range(8) if s in seeds_b],
                                             f"{f}_relRMSE"].values))
                 for f in ("stock", "flow", "alpha", "tau")]
        for nm in v5.STOCK_NAMES:
            y = anc_stock[(anc_stock.rollout == rollout) & (anc_stock.component == nm)]
            y = y[y.seed.isin(range(8))]
            rows.append(dict(arm="anchor_v4", rollout=rollout, family=f"stock:{nm}",
                             **w10.iqr_row(y.relRMSE.values)))
        for a in list(OLD_ARMS) + list(NEW_ARMS):
            sub = ps[(ps.arm == a) & (ps.rollout == rollout)].set_index("seed")
            if not len(sub):
                continue
            seeds = sorted(set(sub.index) & set(base.index))
            for fam in ("stock", "flow", "alpha", "tau"):
                col = f"{fam}_relRMSE"
                rows.append(dict(arm=a, rollout=rollout, family=fam,
                                 **w10.iqr_row(sub[col].values)))
                d = (sub.loc[seeds, col] - base.loc[seeds, col]).values
                hl = w10.hodges_lehmann(d)
                paired.append(dict(arm=a, rollout=rollout, family=fam,
                                   hl=hl["hl"], lo=hl["lo"], hi=hl["hi"], n=hl["n"],
                                   n_improved=int((d < 0).sum())))
            sc = pc[(pc.arm == a) & (pc.rollout == rollout) & (pc.family == "stock")]
            bs = anc_stock[anc_stock.rollout == rollout]
            for nm in v5.STOCK_NAMES:
                x = sc[sc.component == nm].set_index("seed")["relRMSE"]
                y = bs[bs.component == nm].set_index("seed")["relRMSE"]
                cm = sorted(set(x.index) & set(y.index))
                rows.append(dict(arm=a, rollout=rollout, family=f"stock:{nm}",
                                 **w10.iqr_row(x.values)))
                d = (x.loc[cm] - y.loc[cm]).values
                hl = w10.hodges_lehmann(d)
                paired.append(dict(arm=a, rollout=rollout, family=f"stock:{nm}",
                                   hl=hl["hl"], lo=hl["lo"], hi=hl["hi"], n=hl["n"],
                                   n_improved=int((d < 0).sum())))
            for fam in ("alpha", "flow"):
                ac = pc[(pc.arm == a) & (pc.rollout == rollout) & (pc.family == fam)]
                for nm in sorted(ac.component.unique()):
                    rows.append(dict(arm=a, rollout=rollout, family=f"{fam}:{nm}",
                                     **w10.iqr_row(ac[ac.component == nm]["relRMSE"].values)))
    w10._write(pd.DataFrame(rows), "wp10b/wp10b_summary.csv")
    w10._write(pd.DataFrame(paired), "wp10b/wp10b_paired.csv")

    # --- fitted lifetimes ---------------------------------------------------
    ex = pd.concat([old_ex[old_ex.arm.isin(OLD_ARMS)], ex_n], ignore_index=True)
    lt = []
    for a in ex.arm.unique():
        sub = ex[ex.arm == a]
        for nm, mu in zip(("short", "medium", "long"), (10.0, 20.0, 44.0)):
            c = f"tau_fitted_{nm}_yr"
            if c in sub and sub[c].notna().any():
                r = w10.iqr_row(sub[c].values)
                lt.append(dict(arm=a, cohort=nm, mu_rostek=mu, **r,
                               change_pct=100 * (r["median"] / mu - 1)))
    w10._write(pd.DataFrame(lt), "wp10b/wp10b_fitted_lifetimes.csv")
    return pd.DataFrame(rows), pd.DataFrame(paired), pd.DataFrame(lt), chk


def main():
    ap = argparse.ArgumentParser(description="WP-10b")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--arms", default=",".join(NEW_ARMS))
    ap.add_argument("--seeds", default="0-7")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.check:
        print("zinc S4 orders:", lab.zinc_s4_n())
        for nm in NEW_ARMS:
            print(nm, lab.arm(nm))
        lab.arm("exp")
        lab.check()
        return 0
    if a.summary:
        summary()
        return 0
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    _fit_stream(arms, w10.parse_seeds(a.seeds), a.tag or "_".join(arms))
    return 0


if __name__ == "__main__":
    sys.exit(main())
