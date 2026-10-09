#!/usr/bin/env python3
"""
run_anchor.py  (v2)
===================

Fits the v4 hand-tuned PINN config, the Phase 1 HPO winner (trial 5 / src 9),
and the zinc_baseline GAM-ridge regression model under an IDENTICAL
protocol -- same split, same exog columns, same pins -- over multiple seeds.

Saves stocks AND flows so downstream plotting can show both, plus per-family,
per-stock and per-flow metric CSVs.

Configs
-------
  v4            hand-tuned notebook config (the ~23% anchor)
  trial5        Phase 1 HPO winner, zinc_optuna_phase1_B_w1.db trial 5
  baseline_gam  zinc_baseline, GAM estimator with empty smooth bases
                (pure ridge), Stage A + Stage B (700 steps)

Examples
--------
  python run_anchor.py --config v4            --seeds 0,1,2,3
  python run_anchor.py --config trial5        --seeds 0,1,2,3 --budget v4
  python run_anchor.py --config baseline_gam  --seeds 0,1,2,3
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

STOCK_NAMES = ["Concentrate", "Refined", "In-Use", "Scrap"]

# --- exog columns ----------------------------------------------------------
# Order matters: it sets the input feature order, hence the weight init.
# v4 notebook order (as in DEFAULT_CONFIG).
EXTRAS_V4 = [
    "Precious metal index",
    "Full metal real index",   # workbook has no "Metal real index excl iron"
    "World Stock Market Capitalisation (% of GDP)",
    "China Total Manufacturing Output",
    "Population",
]
# order as stored in the Phase 1 trial's user_attrs["extras"]
EXTRAS_TRIAL5 = [
    "China Total Manufacturing Output",
    "Precious metal index",
    "Population",
    "World Stock Market Capitalisation (% of GDP)",
    "Full metal real index",
]

# From the Phase 1 user_attr "forced_config" -- identical to the v4 notebook.
FORCED = dict(
    do_stage_B=True,
    learn_cp=False,
    use_stock_input=False,
    pin_tau_ref=True,
    pin_tau_waelz=True,
    pin_tau_diss=True,
    pin_tau_olds=False,
    pin_frac_fu_loss=True,
    pin_frac_eu_loss=True,
    pin_frac_fu_new=False,
    pin_frac_eu_new=False,
)

BUDGETS = {
    "v4": dict(stageA_steps=2000,
               stageB_curriculum=[(800, 8), (800, 16), (400, 22)],
               final_refit_curriculum=None,
               final_refit_steps=400),
    "cheap": dict(stageA_steps=1200,
                  stageB_curriculum=[(400, 8), (400, 16), (250, 22)],
                  final_refit_curriculum=[(150, 22)],
                  final_refit_steps=400),
}

V4 = dict(
    stageB_w_F=0.0,
    stageB_w_S=5.0,
    stock_loss_kind="std_scaled",
    stock_term_weights=(3.0, 3.0, 1.0, 1.5),
    final_refit_on_trainval=False,
    hidden_width=32, hidden_depth=2,
    exog_log1p=True, exog_detrend=False,
    exog_feature_orders=(0, 1), exog_pca_components=None,
    stock_log_scale=1.0,
    stageA_lr=3e-4, stageB_lr=1e-4, stageB_batch_size=4,
    stageA_w_smooth_alpha=0.05, stageA_w_smooth_tau=0.05,
    stageA_w_raw_reg=1e-4, stageA_w_cohort_prior=0.0,
    stageB_w_smooth_alpha=0.05, stageB_w_smooth_tau=0.05,
    stageB_w_raw_reg=1e-4, stageB_w_cohort_prior=0.0,
    weight_decay=1e-4,
)

TRIAL5 = dict(
    exog_detrend=True, exog_feature_orders=(0, 1, 2), exog_log1p=False,
    exog_pca_components=None, hidden_depth=3, hidden_width=48,
    stageA_lr=0.000638611677967022, stageA_w_cohort_prior=0.0,
    stageA_w_raw_reg=1.6953883696802934e-06,
    stageA_w_smooth_alpha=0.0011199923666280402,
    stageA_w_smooth_tau=0.0010591651432149314,
    stageB_batch_size=2, stageB_lr=9.998441686397945e-05,
    stageB_w_F=1.3019732600815468, stageB_w_S=5.75774107639095,
    stageB_w_cohort_prior=0.0, stageB_w_raw_reg=1.4402666443512115e-06,
    stageB_w_smooth_alpha=0.0010168310923319873,
    stageB_w_smooth_tau=0.10548142326689702,
    stock_log_scale=0.7, weight_decay=1.7973850108962665e-05,
    stock_loss_kind="log", final_refit_on_trainval=True,
    stock_term_weights=(1.0, 1.0, 1.0, 1.0),
)

# zinc_baseline, GAM estimator with the smooth bases emptied -> pure ridge.
# These are the settings that fit stably; the Elastic Net variant is not used.
# LEARNED must list exactly the targets left free by the pin flags below.
LEARNED = ["alpha_cc", "alpha_refc", "alpha_win", "alpha_dr",
           "tau_olds", "frac_fu_new", "frac_eu_new"]

BASELINE_COMMON = dict(
    use_time_input=False,
    use_stock_input=True,          # baseline uses stock as a feature (see note)
    stock_norm_mode="tanh_log", stock_log_scale=1.0, stock_ref_mode="mean",
    learn_cp=False,
    pin_tau_ref=True, pin_tau_waelz=True, pin_tau_diss=True, pin_tau_olds=False,
    pin_frac_fu_loss=True, pin_frac_eu_loss=True,
    pin_frac_fu_new=False, pin_frac_eu_new=False,
    exog_log1p=True, exog_detrend=False,
    exog_feature_orders=(0, 1), exog_diff_pad="edge", exog_pca_components=None,
    trainval_frac=0.7, val_frac=0.2,
    regressor_kinds="gam",
    gam_smooth_features={n: [] for n in LEARNED},   # no smooths -> pure ridge
    n_cv_splits=5,
    do_stage_B=True,
    stageB_steps=700,
    stock_term_weights=(1.0, 1.0, 1.0, 1.0),
    verbose=False,
)

IS_BASELINE = {"baseline_gam": "gam"}


def rel_rmse_pct(pred, obs):
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    m = np.isfinite(pred) & np.isfinite(obs)
    if not m.any():
        return float("nan")
    return 100.0 * np.sqrt(np.mean((pred[m] - obs[m]) ** 2)) / np.mean(np.abs(obs[m]))


def persistence_benchmark(fit):
    yrs = np.asarray(fit.data_all["years"], float).ravel()
    S = np.asarray(fit.data_all["stocks_obs"], float)
    yt = np.asarray(fit.data_test["years"], float).ravel()
    t0 = int(np.argmin(np.abs(yrs - yt.min())))
    ic = max(t0 - 1, 0)
    sel = np.isin(yrs, yt)
    out = {nm: rel_rmse_pct(np.full(int(sel.sum()), S[ic, k]), S[sel, k])
           for k, nm in enumerate(STOCK_NAMES)}
    out["<stock-mean>"] = float(np.nanmean(list(out.values())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="../inputs/zinc_dataset.xlsx")
    ap.add_argument("--config", default="v4",
                    choices=["v4", "trial5", "baseline_gam"])
    ap.add_argument("--budget", choices=["v4", "cheap"], default="v4")
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--out", default=None)
    ap.add_argument("--module", default="zinc_colloc_v5")
    ap.add_argument("--baseline-module", default="zinc_baseline")
    ap.add_argument("--add-extras", default=None,
                    help='extra columns appended to the exog set, "|"-separated')
    ap.add_argument("--stock-loss-kind", choices=["log", "std_scaled"], default=None)
    ap.add_argument("--final-refit", type=int, choices=[0, 1], default=None)
    ap.add_argument("--stock-term-weights", default=None, help='e.g. "3,3,1,1.5"')
    ap.add_argument("--baseline-stage-b", type=int, choices=[0, 1], default=1)
    ap.add_argument("--baseline-stock-input", type=int, choices=[0, 1], default=1,
                    help="0 matches the PINN config (no stock feature); "
                         "1 is the baseline's own stable setting")
    ap.add_argument("--append", action="store_true",
                    help="merge these seeds into an existing --out directory "
                         "instead of overwriting its CSVs.  Refuses to run if "
                         "the stored config differs from the current one.")
    args = ap.parse_args()

    baseline_kind = IS_BASELINE.get(args.config)
    label = args.config if baseline_kind else f"{args.config}_bud{args.budget}"
    out = args.out or f"anchor_{label}"
    os.makedirs(out, exist_ok=True)

    extras = list(EXTRAS_TRIAL5 if args.config == "trial5" else EXTRAS_V4)
    if args.add_extras:
        extras += [c.strip() for c in args.add_extras.split("|") if c.strip()]

    # FLOW_NAMES always comes from the v5 module (zinc_baseline imports it).
    v5mod = __import__(args.module)

    if baseline_kind:
        mod = __import__(args.baseline_module)
        runner = mod.run_baseline
        cfg = dict(BASELINE_COMMON)
        cfg["regressor_kinds"] = baseline_kind
        cfg["do_stage_B"] = bool(args.baseline_stage_b)
        cfg["use_stock_input"] = bool(args.baseline_stock_input)
        if not args.baseline_stock_input:
            label += "_nostockin"
        cfg.update(extra_exog_cols=extras, xlsx_path=args.xlsx)
    else:
        runner = v5mod.run
        cfg = dict(FORCED)
        cfg.update(V4 if args.config == "v4" else TRIAL5)
        cfg.update(BUDGETS[args.budget])
        cfg.update(extra_exog_cols=extras, xlsx_path=args.xlsx, verbose=False)
        if args.stock_loss_kind:
            cfg["stock_loss_kind"] = args.stock_loss_kind
            label += f"_{args.stock_loss_kind}"
        if args.final_refit is not None:
            cfg["final_refit_on_trainval"] = bool(args.final_refit)
            label += f"_fr{args.final_refit}"
        if args.stock_term_weights:
            cfg["stock_term_weights"] = tuple(
                float(x) for x in args.stock_term_weights.split(","))
            label += "_stw" + args.stock_term_weights.replace(",", "-")

    print("=" * 74)
    print(f"config={args.config}  label={label}  n_exog_extras={len(extras)}")
    for k in ("regressor_kinds", "stock_loss_kind", "final_refit_on_trainval",
              "stock_term_weights", "stageB_w_F", "stageB_w_S", "hidden_width",
              "hidden_depth", "stageA_steps", "stageB_curriculum", "do_stage_B"):
        if k in cfg:
            print(f"  {k:26s} = {cfg[k]}")
    print("=" * 74)
    cfg_path = os.path.join(out, "config_used.json")
    cfg_json = {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in cfg.items()}
    if args.append and os.path.exists(cfg_path):
        # Appending seeds is only valid if they come from the same estimator.
        # Silently mixing two configs into one "seed spread" would be a far
        # worse failure than stopping here.
        with open(cfg_path) as fh:
            old_cfg = json.load(fh)
        new_cfg = json.loads(json.dumps(cfg_json, default=str))
        diff = sorted(set(old_cfg) | set(new_cfg))
        diff = [k for k in diff if old_cfg.get(k, "<missing>") != new_cfg.get(k, "<missing>")]
        if diff:
            print("ERROR: --append refused, stored config differs from this run:")
            for k in diff:
                print(f"  {k}: existing={old_cfg.get(k, '<missing>')!r}  "
                      f"new={new_cfg.get(k, '<missing>')!r}")
            print("Use a fresh --out directory, or drop --append to overwrite.")
            return 2
        print(f"[append] config matches {cfg_path} — merging into {out}")
    with open(cfg_path, "w") as fh:
        json.dump(cfg_json, fh, indent=2, sort_keys=True, default=str)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    per_seed, per_stock, per_flow, bench = [], [], [], None

    for sd in seeds:
        t0 = time.time()
        try:
            fit = runner(f"{label}_s{sd}", **dict(cfg, seed=sd))
        except Exception as e:
            print(f"[seed {sd}] FAILED: {type(e).__name__}: {e}")
            continue
        dt = time.time() - t0
        if bench is None:
            bench = persistence_benchmark(fit)

        for kind in ("freerun", "testrun"):
            rows = fit.diagnose(kind, verbose=False, return_rows=True,
                                include_pinned=False)
            df = pd.DataFrame(rows)
            if (df["stage"] == "B").any():
                df = df[df["stage"] == "B"]
            rec = dict(config=args.config, label=label, seed=sd,
                       rollout=kind, wall_s=round(dt, 1))
            for fam in ("stock", "flow", "alpha", "tau"):
                r = df[df["kind"] == f"{fam}_mean"]
                if len(r):
                    rec[f"{fam}_relRMSE"] = float(r.iloc[0]["test_relRMSE%"])
                    rec[f"{fam}_MAPE"] = float(r.iloc[0]["test_MAPE%"])
            per_seed.append(rec)

            # per-component: include the *_mean rows too (v1 dropped them,
            # which is why <stock-mean> printed as nan)
            for sink, kinds in ((per_stock, ("stock", "stock_mean")),
                                (per_flow, ("flow", "flow_mean"))):
                for _, r in df[df["kind"].isin(kinds)].iterrows():
                    sink.append(dict(
                        config=args.config, label=label, seed=sd, rollout=kind,
                        component=str(r["component"]).replace(" [testrun]", "").strip(),
                        relRMSE=float(r["test_relRMSE%"]),
                        MAPE=float(r["test_MAPE%"])))

        # ---- save stocks AND flows for plotting ----
        pr = fit.predictions("B")
        f_idx = np.asarray(fit.flow_obs_to_pred_idx, int)
        fnames = [v5mod.FLOW_NAMES[i] for i in f_idx]
        years = np.asarray(fit.data_all["years"], float).ravel()
        arrays = dict(
            years_all=years,
            stocks_obs=np.asarray(fit.data_all["stocks_obs"], float),
            S_pred_B=np.asarray(pr["S_pred"], float),
            years_flow=years[1:],
            flows_obs=np.asarray(fit.data_all["flows_obs"], float),
            F_pred_B=np.asarray(pr["F_int"], float)[:, f_idx],
            flow_names=np.array(fnames, dtype=object),
            years_test=np.asarray(fit.data_test["years"], float).ravel())

        # Test-window rollout: the ODE re-launched at the trainval boundary
        # from observed stocks, so it isolates extrapolation from the drift
        # the free-run above accumulates over ~27 years.  Saved alongside so
        # plot_anchor can overlay it across models; best-effort because it
        # goes through a private accessor.
        try:
            pr_tr = fit._predictions_testrun("B")
            arrays.update(
                years_testrun=np.asarray(fit.data_test["years"], float).ravel(),
                stocks_obs_testrun=np.asarray(fit.data_test["stocks_obs"], float),
                S_pred_testrun=np.asarray(pr_tr["S_pred"], float),
                F_pred_testrun=np.asarray(pr_tr["F_int"], float)[:, f_idx])
        except Exception as e:
            print(f"[seed {sd}] note: no testrun rollout saved "
                  f"({type(e).__name__}: {e})")

        np.savez_compressed(os.path.join(out, f"pred_seed{sd}.npz"), **arrays)

        st = [x for x in per_seed if x["seed"] == sd and x["rollout"] == "testrun"]
        got = st[0].get("stock_relRMSE", float("nan")) if st else float("nan")
        print(f"[seed {sd}] {dt/60:.1f} min   testrun stock relRMSE = {got:.2f}%")

    if not per_seed:
        print("no successful seeds")
        return 1

    def _write(rows, fname, keys):
        """Write `rows`, merging with what's already on disk when --append.

        Dedup is on `keys` with the NEW rows winning, so re-running a seed
        that already exists overwrites it rather than double-counting."""
        df = pd.DataFrame(rows)
        path = os.path.join(out, fname)
        if args.append and os.path.exists(path):
            try:
                old = pd.read_csv(path)
                n_old = len(old)
                df = (pd.concat([old, df], ignore_index=True)
                        .drop_duplicates(subset=keys, keep="last")
                        .sort_values(keys)
                        .reset_index(drop=True))
                print(f"  merged {fname}: {n_old} existing + {len(rows)} new "
                      f"-> {len(df)} rows")
            except Exception as e:
                print(f"  WARNING: could not merge {fname} "
                      f"({type(e).__name__}: {e}) — writing new rows only")
        df.to_csv(path, index=False)
        return df

    ps = _write(per_seed, "per_seed.csv", ["label", "seed", "rollout"])
    pk = _write(per_stock, "per_stock.csv",
                ["label", "seed", "rollout", "component"])
    _write(per_flow, "per_flow.csv", ["label", "seed", "rollout", "component"])

    print("\n" + "=" * 74)
    print(f"{label}   n_seeds={ps['seed'].nunique()}")
    print("=" * 74)
    for kind in ("freerun", "testrun"):
        sub = ps[ps["rollout"] == kind]
        if not len(sub):
            continue
        print(f"\n--- {kind}: family-mean relRMSE% across seeds ---")
        print(f"{'family':8s} {'median':>9s} {'IQR':>8s} {'min':>8s} {'max':>8s}")
        for fam in ("stock", "flow", "alpha", "tau"):
            c = f"{fam}_relRMSE"
            if c not in sub:
                continue
            v = sub[c].dropna().values
            if not len(v):
                continue
            q1, q3 = np.percentile(v, [25, 75])
            print(f"{fam:8s} {np.median(v):9.2f} {q3-q1:8.2f} {v.min():8.2f} {v.max():8.2f}")

    print("\n--- per-stock relRMSE% (median across seeds) ---")
    print(f"{'stock':16s} {'freerun':>9s} {'testrun':>9s} {'persist':>9s}")
    for nm in STOCK_NAMES + ["<stock-mean>"]:
        fr = pk[(pk.component == nm) & (pk.rollout == "freerun")]["relRMSE"]
        tr = pk[(pk.component == nm) & (pk.rollout == "testrun")]["relRMSE"]
        print(f"{nm:16s} "
              f"{np.nanmedian(fr) if len(fr) else float('nan'):9.2f} "
              f"{np.nanmedian(tr) if len(tr) else float('nan'):9.2f} "
              f"{bench.get(nm, float('nan')):9.2f}")

    print(f"\nwrote {out}/per_seed.csv per_stock.csv per_flow.csv "
          f"config_used.json pred_seed*.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
