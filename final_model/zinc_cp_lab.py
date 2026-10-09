#!/usr/bin/env python3
"""
zinc_cp_lab.py — lab module for the WP-9 close-out: the `learn_cp=True` arm
==========================================================================

WP-9 left exactly one quantity unresolved.  Chapter 1's H2 predicts that
faster growth *dilutes* the stock-based recovery loop,

    dchi/dg = -4.87 per unit growth rate                (Ch1 Eq. `chistar`)

and WP-9 measured the numerator of that ratio well: the dilution channel
Chapter 1 admits is present, correctly signed, at -0.45 [-0.57, -0.36].  The
*net* figure could not be reported, because its denominator is the response of
aggregate throughput growth to a driver's growth rotation, and `anchor_v4`
runs with `learn_cp: false`.  With cp pinned, concentrate production is read
straight off the ILZSG series at every node, so no driver perturbation can
move the scale of the cycle: the measured `dg_throughput/dtheta` is ~1e-4/yr
per +1 pp/yr and its sign is not consistent across the activity block.  The
ratio's numerator is identified; its denominator is not.
`analysis/notes/COMPUTE_STATUS.md` flag 5 records this and says what closes
it — a refit with `learn_cp: true`.  This module is that refit.

What changes, and what deliberately does not
--------------------------------------------
**One config key.**  `cp_config()` is `anchor_v4/config_used.json` with
`learn_cp` flipped to `True` and nothing else touched — same data copy, same
drivers, same feature orders, same curriculum, same loss weights (the core's
`stageA_w_cp` / `stageB_w_cp` already default to 1.0 and the anchor config
never overrode them), same seeds.  `assert_single_key_change()` enforces that
mechanically rather than by inspection, and `--check` prints the diff.

**The core is untouched.**  `PATCHES` is empty.  `learn_cp` is a first-class
argument of `zinc_colloc_v5.train_model` (v5:1733), so nothing has to be
rebound to exercise this arm — CLAUDE.md rule 1 is satisfied by not needing
it.  The one structural consequence is in the core's own hands: with
`learn_cp=True`, `build_nn_layout` (v5:519) gives the head a `cp_slot`, so the
raw output width goes 10 -> 11 and the MLP acquires 33 more parameters
(2410 -> 2443).
`layout_diff()` reports that rather than asserting it.

**The published model does not move.**  This arm exists to identify one
derivative, not to be compared on held-out error and preferred (CLAUDE.md
rule 3).  `anchor_v4` remains the published fit; every WP-9 number already
reported stays as it is, and the cp arm is reported alongside as the
identification experiment it is.  `fit_quality()` computes the arm's error
because a denominator measured on a badly-fitting cycle would not be worth
having — as a sanity floor, never as a selection criterion.

Why the denominator should become identified
--------------------------------------------
With cp learned, concentrate production is an MLP output reading the same
exogenous feature vector as every other coefficient, so a growth rotation of
GDP propagates into the scale of the cycle through `b(t) = [cp(t), 0, ...]`
(WP-2a's assembly) instead of being absorbed by the data pin.  The prediction
this module is here to test, fixed in source before the refits were run:

    the ensemble spread of `dg_throughput/dtheta_j` over the Ch1 activity +
    population block stops straddling zero, and the pooled origin regression
    of `dchi/dtheta_j` on `dg_throughput/dtheta_j` acquires a `dchi/dg` whose
    Hodges--Lehmann interval excludes zero.

It is a prediction, not a guarantee.  If the denominator is still weak the
result is that `dchi/dg` is not identified by this cycle *at all* rather than
by this configuration, which is a stronger and more useful limitation than
the one WP-9 currently states.  Either way the arm is reported.

Outputs
-------
`analysis/wp9_cp/weights/A_seed{N}.npz` — the WP-2a dump schema exactly,
written by `zinc_A_lab.dump_A`, so every downstream WP-9 consumer
(`zinc_ch1_lab.build_context`, `load_params`, `sensitivity`) reads this arm by
pointing `weights_dir` at it and changing nothing else.

Pattern
-------
Follows `zinc_ch1_lab.py` / `zinc_A_lab.py` (CLAUDE.md rule 1).
`zinc_colloc_v5.py` is imported and never edited.  Integrity checking, config
loading and the driver / `input_dim` part of `--check` come from
`zinc_alpha_lab`; the fit-and-dump step is `zinc_A_lab.fit_seed`, unmodified,
called with this arm's config.

CLI
---
    python zinc_cp_lab.py --check
    python zinc_cp_lab.py --seeds 0,1,2 [--skip-existing]
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

import numpy as np

import zinc_alpha_lab as alab
from zinc_alpha_lab import integrity_check, load_anchor_config      # noqa: F401
import zinc_A_lab as Alab

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR_DEFAULT = os.path.join(HERE, "analysis", "wp9_cp")
WEIGHTS_DIR_DEFAULT = os.path.join(OUT_DIR_DEFAULT, "weights")
SENS_DIR_DEFAULT = os.path.join(OUT_DIR_DEFAULT, "sens")
BASE_WEIGHTS_DIR = os.path.join(HERE, "analysis", "wp2a")

PATCHES: list[str] = []          # none — `learn_cp` is a core config flag


def install(v5mod=None):
    """Apply import-time patches to the core module.  None needed here."""
    if v5mod is None:
        import zinc_colloc_v5 as v5mod          # noqa: F401
    return PATCHES


# ===========================================================================
# 1.  The arm's configuration — one key, enforced
# ===========================================================================

ARM_KEY = "learn_cp"
ARM_VALUE = True


def cp_config():
    """`anchor_v4`'s config with `learn_cp=True`.  Nothing else changes."""
    cfg = dict(load_anchor_config())
    cfg[ARM_KEY] = ARM_VALUE
    return cfg


def config_diff(cfg=None, base=None):
    """`{key: (base, arm)}` for every key that differs.  Should be one key."""
    cfg = dict(cfg or cp_config())
    base = dict(base or load_anchor_config())
    keys = sorted(set(cfg) | set(base))
    return {k: (base.get(k, "<absent>"), cfg.get(k, "<absent>"))
            for k in keys if cfg.get(k, "<absent>") != base.get(k, "<absent>")}


def assert_single_key_change(cfg=None):
    """The arm is one flag.  Refuse to run if it has quietly become more."""
    diff = config_diff(cfg)
    if diff != {ARM_KEY: (False, True)}:
        raise RuntimeError(
            f"the learn_cp arm must differ from anchor_v4 in exactly "
            f"{ARM_KEY!r}; got {diff!r}. Any further change makes the "
            f"identification comparison uninterpretable.")
    return diff


# ===========================================================================
# 2.  What the flag does to the network
# ===========================================================================

def layout(cfg):
    """The core's own NN layout for a config (v5:519), by keyword."""
    import zinc_colloc_v5 as v5

    keys = ("learn_cp", "pin_tau_ref", "pin_tau_waelz", "pin_tau_olds",
            "pin_tau_diss", "pin_frac_fu_loss", "pin_frac_eu_loss",
            "pin_frac_fu_new", "pin_frac_eu_new")
    return v5.build_nn_layout(**{k: bool(cfg.get(k, False)) for k in keys})


def mlp_param_count(cfg, input_dim, n_raw):
    """Parameters in `init_mlp`'s dense stack for this config (v5:445)."""
    w = int(cfg.get("hidden_width", 32))
    d = int(cfg.get("hidden_depth", 2))
    dims = [int(input_dim)] + [w] * d + [int(n_raw)]
    return sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))


def layout_diff(input_dim=None):
    """Head width, cp slot and parameter count under both flags."""
    base_cfg, arm_cfg = load_anchor_config(), cp_config()
    out = {}
    for tag, cfg in (("anchor_v4", base_cfg), ("learn_cp", arm_cfg)):
        lay = layout(cfg)
        row = dict(n_raw=int(lay["n_raw"]),
                   cp_slot=(None if lay["cp_slot"] is None
                            else int(lay["cp_slot"])),
                   pin_cp=bool(lay["pin_cp"]))
        if input_dim is not None:
            row["n_params"] = mlp_param_count(cfg, input_dim, lay["n_raw"])
        out[tag] = row
    return out


# ===========================================================================
# 3.  Fitting
# ===========================================================================

def seed_list(weights_dir=WEIGHTS_DIR_DEFAULT):
    if not os.path.isdir(weights_dir):
        return []
    return sorted(int(f[len("A_seed"):-len(".npz")])
                  for f in os.listdir(weights_dir)
                  if f.startswith("A_seed") and f.endswith(".npz"))


def fit_seed(seed, out_dir=WEIGHTS_DIR_DEFAULT, cfg=None, verbose=False):
    """One `learn_cp=True` refit, dumped in the WP-2a schema.

    Delegates to `zinc_A_lab.fit_seed`, which already does the integrity
    check, the `A(t)` assembly and its verification, `jax.clear_caches()`
    per fit (CLAUDE.md rule 6) and the dump.  Nothing about the fit itself
    is re-implemented here; only the config differs.
    """
    cfg = dict(cfg or cp_config())
    assert_single_key_change(cfg)
    os.makedirs(out_dir, exist_ok=True)
    return Alab.fit_seed(seed, out_dir, cfg=cfg, verbose=verbose)


# ===========================================================================
# 4.  Reading a dump back: is cp actually doing anything?
# ===========================================================================

def _rel_rmse_pct(pred, obs):
    """`100 * rmse / mean(|obs|)` — the core's own `_rel_rmse_pct` (v5:3283)
    and the project's stated denominator (CLAUDE.md conventions)."""
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    m = np.isfinite(pred) & np.isfinite(obs)
    if not m.any():
        return float("nan")
    den = float(np.mean(np.abs(obs[m])))
    if den < 1e-12:
        return float("nan")
    return 100.0 * float(np.sqrt(np.mean((pred[m] - obs[m]) ** 2))) / den


def _family_mean(values):
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    return float(v.mean()) if v.size else float("nan")


def load_dump(path):
    d = np.load(path, allow_pickle=True)
    return dict(years=np.asarray(d["years"], float),
                S_obs=np.asarray(d["S_obs"], float),
                S_pred=np.asarray(d["S_pred"], float),
                alphas=np.asarray(d["alphas"], float),
                taus=np.asarray(d["taus"], float),
                frac_fu=np.asarray(d["frac_fu"], float),
                frac_eu=np.asarray(d["frac_eu"], float),
                cp=np.asarray(d["cp"], float).ravel(),
                stock_names=[str(x) for x in d["stock_names"]],
                mask_train=np.asarray(d["mask_train"], bool),
                mask_val=np.asarray(d["mask_val"], bool),
                mask_test=np.asarray(d["mask_test"], bool),
                verify=json.loads(str(d["verify_json"])))


def observation_reference(cfg=None):
    """Everything a family-mean error needs, built once from a zero-step fit.

    Carries the observation arrays, the `flows_obs -> FLOW_NAMES` column map,
    and the pin flags that decide which components enter a family mean.  The
    pinned set differs between the arms by exactly one entry — with cp learned,
    `concentrate_production` stops being a tautology and joins the flow family
    — which is why the mask is read off each arm's own layout rather than
    fixed here.
    """
    import zinc_colloc_v5 as v5
    import zinc_cf_lab as cflab

    cfg = dict(cfg or load_anchor_config())
    fit = cflab.init_fit(cfg)
    d = fit.data_all
    return dict(
        alpha_obs=np.asarray(d["alpha_obs"], float),
        tau_sup_obs=np.asarray(d["tau_sup_obs"], float),
        flows_obs=np.asarray(d["flows_obs"], float),
        cp_obs=np.asarray(d["cp_obs"], float).ravel(),
        stocks_obs=np.asarray(d["stocks_obs"], float),
        flow_obs_to_pred_idx=[int(i) for i in fit.flow_obs_to_pred_idx],
        flow_obs_names=[v5.FLOW_NAMES[int(i)] for i in fit.flow_obs_to_pred_idx],
        stock_names=list(v5.STOCK_NAMES),
        alpha_names=list(v5.ALPHA_NAMES),
        n_alphas=int(v5.N_ALPHAS), n_stocks=int(v5.N_STOCKS),
        n_tau_sup=int(v5.N_TAU_SUP),
        tau_pinned=[bool(x) for x in fit._pinned_flags_per_tau_sup()],
    )


def family_errors(dump_path, F_pred, ref, *, learn_cp, split="test"):
    """`{family: relRMSE%}` on one seed, in the core's own convention.

    The core reports a family mean of PER-COMPONENT relRMSE (v5:3417), not a
    relRMSE pooled over components: pooling lets the largest stock set the
    family number.  `fit.summary()` / `per_seed.csv` use the per-component
    mean, so that is what is reproduced here and what these numbers may be
    compared against.

    `alphas`, `taus` and `cp` are evaluated at the observed stocks in the core
    (`*_at_obs`).  `anchor_v4` runs `use_stock_input: false`, so the MLP never
    sees the state and `at_obs == at_pred` exactly; the dump records that as
    `max_abs_coef_state_dependence` and the caller asserts it is 0.
    """
    d = load_dump(dump_path)
    m = {"test": d["mask_test"], "train": d["mask_train"],
         "val": d["mask_val"], "all": np.ones_like(d["mask_test"])}[split]
    mf = m[1:]                                  # F row i closes years[i+1]
    out = {}

    out["alpha_relRMSE"] = _family_mean([
        _rel_rmse_pct(d["alphas"][m, k], ref["alpha_obs"][m, k])
        for k in range(ref["n_alphas"])])

    tau_rows = []
    for j in range(ref["n_tau_sup"]):
        if ref["tau_pinned"][j]:
            continue
        pre = (d["taus"][:, j] if j < 4 else
               d["frac_fu"][:, j - 4] if j < 6 else d["frac_eu"][:, j - 6])
        tau_rows.append(_rel_rmse_pct(pre[m], ref["tau_sup_obs"][m, j]))
    out["tau_relRMSE"] = _family_mean(tau_rows)

    out["stock_relRMSE"] = _family_mean([
        _rel_rmse_pct(d["S_pred"][m, k], ref["stocks_obs"][m, k])
        for k in range(ref["n_stocks"])])

    # `concentrate_production` is a tautology when cp is pinned and is
    # excluded from the flow family then, exactly as the core does it.
    f_idx, f_names = ref["flow_obs_to_pred_idx"], ref["flow_obs_names"]
    flow_rows, flow_rows_common = [], []
    for k, nm in enumerate(f_names):
        e = _rel_rmse_pct(np.asarray(F_pred)[mf, f_idx[k]],
                          ref["flows_obs"][mf, k])
        if not (not learn_cp and nm == "concentrate_production"):
            flow_rows.append(e)
        if nm != "concentrate_production":
            flow_rows_common.append(e)      # like-for-like across the arms
        out[f"flow_relRMSE_{nm}"] = e
    out["flow_relRMSE"] = _family_mean(flow_rows)
    out["flow_relRMSE_common"] = _family_mean(flow_rows_common)

    out["cp_relRMSE"] = _rel_rmse_pct(d["cp"][m], ref["cp_obs"][m])
    for k, nm in enumerate(ref["stock_names"]):
        out[f"stock_relRMSE_{nm}"] = _rel_rmse_pct(d["S_pred"][m, k],
                                                   ref["stocks_obs"][m, k])
    out["coef_state_dependence"] = float(
        d["verify"].get("max_abs_coef_state_dependence", np.nan))
    return out


def _rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024.0 ** 2) if sys.platform == "darwin" else r / 1024.0


# ===========================================================================
# 5.  --check
# ===========================================================================

def check(verbose=True, weights_dir=WEIGHTS_DIR_DEFAULT):
    """`zinc_alpha_lab.check()` (drivers, `input_dim`, digests) plus the one
    config key this arm changes and what it does to the network."""
    import zinc_colloc_v5 as v5

    info = alab.check(verbose=verbose)
    install(v5)
    diff = assert_single_key_change()
    lay = layout_diff(input_dim=info.get("input_dim"))
    info.update(arm_key=ARM_KEY, config_diff={k: list(v) for k, v in diff.items()},
                layout=lay, patches=list(PATCHES),
                weights_dir=weights_dir,
                n_weight_dumps=len(seed_list(weights_dir)),
                base_weights_dir=BASE_WEIGHTS_DIR,
                n_base_weight_dumps=len(seed_list(BASE_WEIGHTS_DIR)))
    if verbose:
        print("zinc_cp_lab --check  (WP-9 close-out: the learn_cp arm)")
        print("=" * 76)
        print(f"  config diff vs anchor_v4 : "
              + ", ".join(f"{k}: {a!r} -> {b!r}" for k, (a, b) in diff.items()))
        for tag, row in lay.items():
            print(f"  {tag:<10} head n_raw={row['n_raw']}  "
                  f"cp_slot={row['cp_slot']}  pin_cp={row['pin_cp']}"
                  + (f"  n_params={row['n_params']}" if "n_params" in row else ""))
        b, a = lay["anchor_v4"], lay["learn_cp"]
        if "n_params" in a:
            print(f"  -> the cp slot costs {a['n_params'] - b['n_params']} "
                  f"parameters and makes b(t) = [cp(t), 0, 0, 0, 0, 0] a "
                  f"function of the drivers")
        print(f"  patches applied          : {PATCHES or 'none (the flag is native)'}")
        print(f"  arm weights              : {info['n_weight_dumps']} dumps in "
              f"{weights_dir}")
        print(f"  anchor_v4 weights        : {info['n_base_weight_dumps']} dumps in "
              f"{BASE_WEIGHTS_DIR}")
        print(f"  RSS {_rss_mb():.0f} MB")
        print("=" * 76)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="WP-9 close-out: learn_cp=True refits")
    ap.add_argument("--check", action="store_true",
                    help="print resolved drivers, input_dim and the arm diff")
    ap.add_argument("--seeds", default="", help="comma-separated seeds to fit")
    ap.add_argument("--out", default=WEIGHTS_DIR_DEFAULT)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    info = check(verbose=True, weights_dir=a.out)
    if a.check or not a.seeds:
        return 0

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "check.json"), "w") as fh:
        json.dump({k: v for k, v in info.items() if k != "digests"}, fh,
                  indent=2, default=str)

    cfg = cp_config()
    for sd in [int(s) for s in a.seeds.split(",") if s.strip()]:
        path = os.path.join(a.out, f"A_seed{sd}.npz")
        if a.skip_existing and os.path.exists(path):
            print(f"[seed {sd}] exists, skipped", flush=True)
            continue
        t0 = time.time()
        fit_seed(sd, a.out, cfg=cfg, verbose=a.verbose)
        print(f"[seed {sd}] total {time.time()-t0:.0f}s  RSS {_rss_mb():.0f} MB",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
