#!/usr/bin/env python3
"""
zinc_alpha_lab.py — lab module for the α-diagnostics work packages (WP-1a–1d)
=============================================================================

First lab module in the project, so it also establishes the pattern that
`CLAUDE.md` hard rule #1 requires (it names `zinc_exog_lab.py` as the
exemplar; that file does not exist — see `analysis/SCHEMA.md` §9 flag 4).

Pattern
-------
1.  `zinc_colloc_v5.py` is imported, never edited.  `integrity_check()`
    pins its MD5 so an accidental edit is caught rather than silently
    absorbed into a result.
2.  Behaviour changes, when needed, are applied by rebinding attributes on
    the imported module object at import time, inside `install()`, and
    recorded in `PATCHES` so every run can print what was patched.
3.  **WP-1a needs no patch.**  Everything it wants is reachable through the
    public `FitResult` API (`predictions("A"|"B")`, `data_all`), it is
    simply never written to disk by `run_anchor.py` (SCHEMA §1, §3).
    `install()` therefore registers nothing and exists so that later
    packages have somewhere to hang their patches.

What this module adds
---------------------
`dump_alpha(fit, path)` persists the per-channel α arrays that
`run_anchor.py` computes internally and discards:

    alpha_obs               (T, 4)  empirical target, row 0 NaN
    alpha_pred_{A,B}_at_obs (T, 4)  NN α evaluated at OBSERVED stocks
    alpha_pred_{A,B}_at_pred(T, 4)  NN α evaluated at the rollout's own stocks
    S_obs, S_pred_{A,B}     (T, 4)  for the WP-1b compensation test
    mask_{train,val,test}   (T,)    the exact masks `diagnose()` uses
    alpha_log_std           (4,)    the Stage A/B loss denominator
    alpha_scale             (4,)    per-channel geometric-mean prefactor

Stage A is dumped alongside Stage B at zero marginal cost, which also
removes the WP-1d blocker recorded in SCHEMA §1 ("Stage A predictions are
not persisted separately").

α does not depend on the rollout
--------------------------------
`FitResult.diagnose` evaluates α/τ/cp **at observed stocks** for every
`kind`, so the α columns of `per_seed.csv` are bit-identical between the
`freerun` and `testrun` rows (verified across all 35 seeds of `anchor_v4`).
The spec's "freerun and testrun" for WP-1a is therefore one number, not
two, under the current model.  This module reports the rollout-independent
α-at-observed-stocks metric and, separately, the α implied along each
rollout (`*_at_pred`), which *is* rollout-dependent and is what feeds the
integrated flows.

CLI
---
    python zinc_alpha_lab.py --check
    python zinc_alpha_lab.py --seeds 0,1,2 --out analysis/wp1a
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Pinned digests: the read-only core model and the data copy that produced
# anchor_v4 (SCHEMA §7 — four .xlsx copies with four different hashes exist).
CORE_MD5 = "8f57a3d702a0d7d9b0b406c663e9effa"      # zinc_colloc_v5.py
DATA_MD5 = "2349fe8e2a7872b3f4691210652da96b"      # final_model/zinc_dataset.xlsx

ANCHOR_DIR = os.path.join(HERE, "anchor_v4")
ANCHOR_CFG = os.path.join(ANCHOR_DIR, "config_used.json")

# Config keys that zinc_colloc_v5 wants as tuples but JSON stores as lists.
_TUPLE_KEYS = ("exog_feature_orders", "stock_term_weights")

PATCHES: list[str] = []          # populated by install(); empty for WP-1a


# ---------------------------------------------------------------------------
# integrity + config
# ---------------------------------------------------------------------------
def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def integrity_check(strict=True):
    """Verify the read-only core and the anchor_v4 data copy are unchanged."""
    out = {}
    for label, path, want in (("zinc_colloc_v5.py", os.path.join(HERE, "zinc_colloc_v5.py"), CORE_MD5),
                              ("zinc_dataset.xlsx", os.path.join(HERE, "zinc_dataset.xlsx"), DATA_MD5)):
        got = _md5(path)
        out[label] = (got, got == want)
        if strict and got != want:
            raise RuntimeError(
                f"{label} MD5 {got} != pinned {want}. The core model is "
                f"read-only (CLAUDE.md rule 1) and the dataset copy is the "
                f"one anchor_v4 was fitted on (SCHEMA §7). Refusing to run."
            )
    return out


def load_anchor_config():
    """The authoritative anchor_v4 config (SCHEMA §5), JSON types repaired.

    `xlsx_path` is stored as a bare relative name and would otherwise
    resolve against the caller's cwd; SCHEMA §7 recommends pinning it to
    the `final_model/` copy explicitly, which is what happens here.
    """
    with open(ANCHOR_CFG) as fh:
        cfg = json.load(fh)
    for k in _TUPLE_KEYS:
        if k in cfg and isinstance(cfg[k], list):
            cfg[k] = tuple(cfg[k])
    if isinstance(cfg.get("stageB_curriculum"), list):
        cfg["stageB_curriculum"] = [tuple(x) for x in cfg["stageB_curriculum"]]
    cfg["xlsx_path"] = os.path.join(HERE, os.path.basename(cfg["xlsx_path"]))
    return cfg


def install(v5mod=None):
    """Apply import-time patches to the core module.  None needed for WP-1a.

    Later packages add their patches here and append a one-line description
    to `PATCHES` so `--check` and every run log state what was rebound.
    """
    if v5mod is None:
        import zinc_colloc_v5 as v5mod           # noqa: F401
    return PATCHES


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------
def check(verbose=True):
    """Resolve drivers and input_dim without fitting anything.

    Repeats `train_model`'s own preprocessing path (`load_zinc_data` →
    `preprocess_exog`) so the printed layout is the one the model will
    actually see, not one inferred from the config.
    """
    import zinc_colloc_v5 as v5

    digests = integrity_check()
    install(v5)
    cfg = load_anchor_config()

    data_np = v5.load_zinc_data(cfg["xlsx_path"],
                                extra_exog_cols=cfg.get("extra_exog_cols"))
    exog_cols = list(data_np["exog_cols"])
    years = np.asarray(data_np["years"], float).ravel()

    # train core: same trainval/val fractions train_model uses by default
    tv_frac = cfg.get("trainval_frac", v5.DEFAULT_CONFIG.get("trainval_frac"))
    v_frac = cfg.get("val_frac", v5.DEFAULT_CONFIG.get("val_frac"))
    T = len(years)
    n_tv = int(round(tv_frac * T))
    cut = n_tv - int(round(v_frac * T))
    exog_proc = v5.preprocess_exog(
        years, data_np["exog_values"], years[:cut],
        do_log1p=cfg.get("exog_log1p", False),
        do_detrend=cfg.get("exog_detrend", False),
        feature_orders=cfg.get("exog_feature_orders", (0,)),
        years_source=data_np["exog_times_full"],
        exog_values_source=data_np["exog_values_full"],
    )
    orders = tuple(sorted(set(int(o) for o in cfg.get("exog_feature_orders", (0,)))))
    n_universe = len(exog_cols)
    input_dim = 1 + 4 + exog_proc.shape[1]

    info = dict(exog_cols=exog_cols, n_universe=n_universe, orders=orders,
                input_dim=input_dim, n_exog_features=int(exog_proc.shape[1]),
                years=(float(years[0]), float(years[-1])), T=T,
                alpha_names=list(v5.ALPHA_NAMES),
                tau_sup_names=list(v5.TAU_SUP_NAMES),
                digests=digests, patches=list(PATCHES))

    if verbose:
        print("=" * 74)
        print("zinc_alpha_lab --check")
        print("=" * 74)
        for label, (got, ok) in digests.items():
            print(f"  {label:20s} md5 {got}  {'OK' if ok else 'MISMATCH'}")
        print(f"  patches applied      : {PATCHES or 'none (WP-1a needs none)'}")
        print(f"  xlsx_path            : {cfg['xlsx_path']}")
        print(f"  years                : {years[0]:.0f}–{years[-1]:.0f}  (T={T})")
        print(f"  train core           : {years[0]:.0f}–{years[cut-1]:.0f} "
              f"(cut={cut}, n_trainval={n_tv})")
        print(f"\n  resolved drivers (canonical order, {n_universe}):")
        for i, c in enumerate(exog_cols):
            print(f"    [{i:2d}] {c}")
        print(f"\n  exog_feature_orders  : {orders}")
        print(f"  exog feature block   : {n_universe} drivers x {len(orders)} orders "
              f"= {exog_proc.shape[1]}")
        print(f"  input_dim            : 1 (t) + 4 (S) + {exog_proc.shape[1]} "
              f"= {input_dim}")
        if input_dim != 23:
            print(f"\n  !! input_dim is {input_dim}, not the 23 asserted by "
                  f"CLAUDE.md rule 2 / spec §1.")
            print( "     SCHEMA §9 flag 1 — unresolved, flagged for the author.")
        print(f"\n  alpha channels       : {v5.ALPHA_NAMES}")
        print("=" * 74)
    return info


# ---------------------------------------------------------------------------
# the dump
# ---------------------------------------------------------------------------
def dump_alpha(fit, path):
    """Persist the per-channel α arrays `run_anchor.py` computes and discards."""
    import zinc_colloc_v5 as v5

    masks = fit._split_indices_in_all()
    stats = fit.data_all["stats"]
    arrays = dict(
        years=np.asarray(fit.data_all["years"], float).ravel(),
        alpha_names=np.array(v5.ALPHA_NAMES, dtype=object),
        stock_names=np.array(["Concentrate", "Refined", "In-Use", "Scrap"], dtype=object),
        alpha_obs=np.asarray(fit.data_all["alpha_obs"], float),
        S_obs=np.asarray(fit.data_all["stocks_obs"], float),
        alpha_log_std=np.asarray(stats["alpha_log_std"], float),
        alpha_scale=np.asarray(stats["alpha_scale"], float),
        mask_train=np.asarray(masks["train"], bool),
        mask_val=np.asarray(masks["val"], bool),
        mask_test=np.asarray(masks["test"], bool),
    )
    for stage in fit._stages():                      # ["A", "B"] when B ran
        pr = fit.predictions(stage)
        arrays[f"alpha_pred_{stage}_at_obs"] = np.asarray(pr["alphas_at_obs"], float)
        arrays[f"alpha_pred_{stage}_at_pred"] = np.asarray(pr["alphas"], float)
        arrays[f"S_pred_{stage}"] = np.asarray(pr["S_pred"], float)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def dump_stage_compare(fit, path):
    """Persist everything WP-1d needs to compare Stage A against Stage B.

    `dump_alpha` covers the α family only.  WP-1d is specified "per family
    and channel", which brings in τ, stocks and flows, and those are not
    reconstructable from the WP-1a dumps: `F_int` and `taus_at_obs` were
    never written, and `run_anchor.py` persists Stage B alone (SCHEMA §1).

    Two things are written per seed:

    *   `<path>` (.npz) — the raw per-stage arrays, so any metric can be
        recomputed later without another refit;
    *   `<path minus .npz>_diagnose.csv` — `FitResult.diagnose(kind="both",
        include_pinned=True)`, i.e. the model's OWN per-component metrics
        for both stages, both rollouts and all four splits.  Using the core
        module's `_rel_rmse_pct` / `_logmae` rather than a reimplementation
        is what makes the WP-1d numbers commensurable with `per_seed.csv`
        and with the spec's quoted family means.

    Both stages' stocks and flows are ODE free-runs of their own weights
    (`zinc_colloc_v5.py:2477` — never teacher-forced), so the comparison is
    closed-loop on both sides.
    """
    import pandas as pd
    import zinc_colloc_v5 as v5

    masks = fit._split_indices_in_all()
    f_idx = np.asarray(fit.flow_obs_to_pred_idx, int)
    arrays = dict(
        years=np.asarray(fit.data_all["years"], float).ravel(),
        alpha_names=np.array(v5.ALPHA_NAMES, dtype=object),
        tau_sup_names=np.array(v5.TAU_SUP_NAMES, dtype=object),
        stock_names=np.array(["Concentrate", "Refined", "In-Use", "Scrap"], dtype=object),
        flow_names=np.array([v5.FLOW_NAMES[i] for i in f_idx], dtype=object),
        tau_pinned=np.asarray(fit._pinned_flags_per_tau_sup(), bool),
        cp_pinned=np.asarray(bool(fit.layout["pin_cp"])),
        alpha_obs=np.asarray(fit.data_all["alpha_obs"], float),
        tau_sup_obs=np.asarray(fit.data_all["tau_sup_obs"], float),
        cp_obs=np.asarray(fit.data_all["cp_obs"], float),
        stocks_obs=np.asarray(fit.data_all["stocks_obs"], float),
        flows_obs=np.asarray(fit.data_all["flows_obs"], float),
        mask_train=np.asarray(masks["train"], bool),
        mask_val=np.asarray(masks["val"], bool),
        mask_test=np.asarray(masks["test"], bool),
    )
    stats = fit.data_all["stats"]
    for key in ("alpha_log_std", "alpha_scale"):
        if key in stats:
            arrays[key] = np.asarray(stats[key], float)

    for stage in fit._stages():
        pr = fit.predictions(stage)
        arrays[f"S_pred_{stage}"] = np.asarray(pr["S_pred"], float)
        # F_int is emitted over the full FLOW_NAMES inventory; reindex to the
        # observed subset so its columns line up with `flows_obs`.
        arrays[f"F_int_{stage}"] = np.asarray(pr["F_int"], float)[:, f_idx]
        arrays[f"alpha_pred_{stage}_at_obs"] = np.asarray(pr["alphas_at_obs"], float)
        arrays[f"alpha_pred_{stage}_at_pred"] = np.asarray(pr["alphas"], float)
        arrays[f"cp_pred_{stage}_at_obs"] = np.asarray(pr["cp_at_obs"], float)
        # TAU_SUP_NAMES is 4 τ + frac_fu_{new,loss} + frac_eu_{new,loss}.  The
        # frac heads carry a third column (the retained fraction) that has no
        # supervised counterpart, so each is sliced exactly as `diagnose` does
        # (`frac_*_at_obs[:, j - 4]` / `[:, j - 6]`, zinc_colloc_v5.py:2971).
        arrays[f"tau_pred_{stage}_at_obs"] = np.concatenate(
            [np.asarray(pr["taus_at_obs"], float)[:, :4],
             np.asarray(pr["frac_fu_at_obs"], float)[:, :2],
             np.asarray(pr["frac_eu_at_obs"], float)[:, :2]], axis=1)

    # How far Stage B actually moved the weights — the scalar that says
    # whether an A ≈ B result means "the trajectory constraint changed
    # nothing" or "Stage B barely moved" (zinc_colloc_v5.py:3222).
    pd_info = fit.params_distance()
    arrays["params_rel_l2_B_vs_A"] = np.asarray(float(pd_info["rel_l2"]))
    arrays["params_l2_diff_B_vs_A"] = np.asarray(float(pd_info["total_l2_diff"]))

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **arrays)

    rows = fit.diagnose(kind="both", include_pinned=True, verbose=False,
                        return_rows=True)
    csv_path = (path[:-4] if path.endswith(".npz") else path) + "_diagnose.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return path, csv_path


# ---------------------------------------------------------------------------
# WP-1d control arms
# ---------------------------------------------------------------------------
# The published A→B delta bundles three changes, so on its own it does not
# isolate the trajectory constraint the spec wants:
#
#   1. the stock-divergence term L_S on the integrated trajectory;
#   2. six more years of training data — Stage A steps on `data_train`
#      (1980–2001, zinc_colloc_v5.py:2244), Stage B on `data_trainval`
#      (1980–2007, :2357);
#   3. 2,000 further optimiser steps over a window curriculum.
#
# Two control arms split them apart:
#
#   arm "wS0"      Stage B with `stageB_w_S = 0`, everything else identical —
#                  same θ_A warm start, same data, same steps, same curriculum,
#                  same window RNG (`default_rng(seed + 7)`).  B − wS0 isolates
#                  L_S exactly; wS0 − A is the data-and-steps effect.
#   arm "Atrainval" Stage A refitted on `data_trainval`, same step count, same
#                  initialisation.  Atrainval − A isolates the extra six years
#                  alone, and cross-checks wS0 − A.
#
# `Atrainval` needs Stage A to step on a different data object, which
# `train_model` does not expose.  Rather than patch the read-only core, the
# Stage A loop is rebuilt here from the core's own public pieces
# (`make_stageA_loss`, the optimiser recipe at :2208) and VERIFIED: run on
# `data_train` it must reproduce the core's `params_A` predictions bit-exactly
# (`--verify-armC`).  A reimplementation that reproduces the original is
# trustworthy on the one input that was changed; one that does not is a bug,
# and the check is what tells the two apart.

ARMS = {
    "AB":         dict(),                          # published anchor_v4 arm
    "wS0":        dict(stageB_w_S=0.0),             # control: no trajectory term
    "Atrainval":  dict(do_stage_B=False),           # control: Stage A on trainval
}


def _stage_a_loop(fit0, data, cfg, n_steps):
    """Re-run the core's Stage A optimiser on an arbitrary data slice.

    Mirrors `train_model`'s Stage A block (`zinc_colloc_v5.py:2208–2244`):
    cosine-decayed AdamW, `alpha=0.1` floor, weight decay from the config,
    no gradient clipping (`grad_clip = 0.0` in `anchor_v4`), full batch, the
    same loss closure from `make_stageA_loss`.
    """
    import jax
    import optax
    import zinc_colloc_v5 as v5

    stageA_loss, _terms = v5.make_stageA_loss(fit0.nn_eval_sup, fit0.layout)
    sched = optax.cosine_decay_schedule(init_value=float(cfg["stageA_lr"]),
                                        decay_steps=max(int(n_steps), 1),
                                        alpha=0.1)
    chain = []
    if cfg.get("grad_clip", 0.0):
        chain.append(optax.clip_by_global_norm(float(cfg["grad_clip"])))
    chain.append(optax.adamw(learning_rate=sched,
                             weight_decay=float(cfg["weight_decay"])))
    opt = optax.chain(*chain)
    params = fit0.params_A                       # the post-warm-start init
    state = opt.init(params)

    @jax.jit
    def step(params, state, data):
        loss, grads = jax.value_and_grad(stageA_loss)(
            params, data,
            w_alpha=cfg["stageA_w_alpha"], w_tau=cfg["stageA_w_tau"],
            w_cp=cfg["stageA_w_cp"],
            w_smooth_alpha=cfg["stageA_w_smooth_alpha"],
            w_smooth_tau=cfg["stageA_w_smooth_tau"],
            w_cohort_prior=cfg["stageA_w_cohort_prior"],
            w_raw_reg=cfg["stageA_w_raw_reg"],
        )
        updates, state = opt.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss

    for _ in range(int(n_steps)):
        params, state, _loss = step(params, state, data)
    return params


def _init_fit(seed, cfg):
    """A zero-step run: gives the exact post-warm-start initialisation and
    every data object and closure `train_model` builds, without
    reimplementing the initialisation (which warm-starts head biases from
    training-window statistics)."""
    import zinc_colloc_v5 as v5
    return v5.run(f"wp1d_init_s{seed}",
                  **dict(cfg, seed=seed, stageA_steps=0, do_stage_B=False))


def fit_arm_Atrainval(seed, cfg, which="trainval"):
    """Stage A refitted on `data_trainval` (or `data_train`, for the check).

    Returns a `FitResult` whose Stage A weights are the refit.  `params_A` is
    rebound on the zero-step fit and its prediction cache cleared, so
    `diagnose` / `predictions` recompute against the new weights using the
    core's own code paths.
    """
    fit0 = _init_fit(seed, cfg)
    data = fit0.data_trainval if which == "trainval" else fit0.data_train
    params = _stage_a_loop(fit0, data, cfg, cfg["stageA_steps"])
    fit0.params_A = params
    fit0.params_B = None
    fit0._cache = {}
    return fit0


def verify_arm_C(seed, cfg=None, tol=0.0):
    """The reimplemented Stage A loop must reproduce the core's own Stage A.

    Runs `_stage_a_loop` on `data_train` — the slice `train_model` uses — and
    compares its α and stock predictions against a normal `do_stage_B=False`
    fit of the same seed.  Any difference means the rebuilt loop is not the
    core's Stage A and the `Atrainval` arm cannot be read as a control.
    """
    import jax
    import zinc_colloc_v5 as v5

    integrity_check()
    install(v5)
    cfg = dict(v5.DEFAULT_CONFIG, **dict(cfg or load_anchor_config()))
    cfg["verbose"] = False

    ref = v5.run(f"wp1d_refA_s{seed}", **dict(cfg, seed=seed, do_stage_B=False))
    ref_pr = ref.predictions("A")
    got = fit_arm_Atrainval(seed, cfg, which="train")
    got_pr = got.predictions("A")

    out = {}
    for key in ("alphas_at_obs", "taus_at_obs", "S_pred"):
        out[key] = float(np.nanmax(np.abs(np.asarray(got_pr[key])
                                          - np.asarray(ref_pr[key]))))
    out["ok"] = all(v <= tol for v in out.values())
    del ref, got
    jax.clear_caches()
    return out


def fit_seed_arm(seed, arm, out_dir, cfg=None, verbose=False):
    """Fit one seed under one WP-1d arm and dump it.  Returns the .npz path."""
    import jax
    import zinc_colloc_v5 as v5

    if arm not in ARMS:
        raise ValueError(f"arm must be one of {sorted(ARMS)}, got {arm!r}")
    integrity_check()
    install(v5)
    cfg = dict(v5.DEFAULT_CONFIG, **dict(cfg or load_anchor_config()))
    cfg["verbose"] = bool(verbose)
    cfg.update(ARMS[arm])
    path = os.path.join(out_dir, f"{arm}_seed{seed}.npz")

    t0 = time.time()
    if arm == "Atrainval":
        fit = fit_arm_Atrainval(seed, cfg, which="trainval")
    else:
        fit = v5.run(f"wp1d_{arm}_s{seed}", **dict(cfg, seed=seed))
    dump_stage_compare(fit, path)
    dt = time.time() - t0

    del fit
    jax.clear_caches()                       # CLAUDE.md rule 6
    print(f"[{arm} seed {seed}] {dt/60:.1f} min -> {path}", flush=True)
    return path


def fit_seed(seed, out_dir, cfg=None, verbose=False, what="alpha"):
    """Refit one anchor_v4 seed and dump it.  Returns the .npz path.

    `what="alpha"` writes the WP-1a α dump; `what="stage"` writes the fuller
    WP-1d stage-comparison dump.  The fit itself is identical either way —
    only what is persisted differs.
    """
    import jax
    import zinc_colloc_v5 as v5

    integrity_check()
    install(v5)
    cfg = dict(cfg or load_anchor_config())
    cfg["verbose"] = bool(verbose)
    stem = {"alpha": "alpha", "stage": "stage"}[what]
    path = os.path.join(out_dir, f"{stem}_seed{seed}.npz")

    t0 = time.time()
    fit = v5.run(f"wp1_s{seed}", **dict(cfg, seed=seed))
    if what == "alpha":
        dump_alpha(fit, path)
    else:
        dump_stage_compare(fit, path)
    dt = time.time() - t0

    # CLAUDE.md rule 6 — long multi-seed runs must not accumulate JAX caches.
    del fit
    jax.clear_caches()
    print(f"[seed {seed}] {dt/60:.1f} min -> {path}", flush=True)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--check", action="store_true",
                    help="print resolved drivers and input_dim, then exit")
    ap.add_argument("--seeds", default="",
                    help="comma-separated seeds to refit and dump")
    ap.add_argument("--out", default=os.path.join(HERE, "analysis", "wp1a"))
    ap.add_argument("--what", choices=("alpha", "stage"), default="alpha",
                    help="alpha = WP-1a α dump; stage = WP-1d stage-comparison dump")
    ap.add_argument("--arm", choices=sorted(ARMS), default=None,
                    help="WP-1d arm to fit (overrides --what)")
    ap.add_argument("--verify-armC", action="store_true",
                    help="check the rebuilt Stage A loop reproduces the core's")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    info = check(verbose=True)
    if args.check:
        return 0

    if args.verify_armC:
        cfg = load_anchor_config()
        bad = 0
        for sd in [int(s) for s in (args.seeds or "0").split(",") if s.strip()]:
            r = verify_arm_C(sd, cfg=cfg)
            print(f"[verify arm C, seed {sd}] max|Δα|={r['alphas_at_obs']:.3e}  "
                  f"max|Δτ|={r['taus_at_obs']:.3e}  max|ΔS|={r['S_pred']:.3e}  "
                  f"{'OK' if r['ok'] else 'MISMATCH'}", flush=True)
            bad += (not r["ok"])
        return 1 if bad else 0

    if not args.seeds:
        return 0

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "check.json"), "w") as fh:
        json.dump({k: v for k, v in info.items() if k != "digests"}, fh,
                  indent=2, default=str)

    cfg = load_anchor_config()
    stem = args.arm if args.arm else args.what
    for sd in [int(s) for s in args.seeds.split(",") if s.strip()]:
        path = os.path.join(args.out, f"{stem}_seed{sd}.npz")
        if args.skip_existing and os.path.exists(path):
            print(f"[seed {sd}] exists, skipped", flush=True)
            continue
        try:
            if args.arm:
                fit_seed_arm(sd, args.arm, args.out, cfg=cfg, verbose=args.verbose)
            else:
                fit_seed(sd, args.out, cfg=cfg, verbose=args.verbose, what=args.what)
        except Exception as e:                     # keep the sweep going
            print(f"[seed {sd}] FAILED: {type(e).__name__}: {e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
