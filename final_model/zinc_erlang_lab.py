#!/usr/bin/env python3
"""
zinc_erlang_lab.py — lab module for WP-10: Erlang use-phase extension
====================================================================

WP-10 in `EXPERIMENTS_SPEC.md` is author-gated and time-boxed.  It replaces
the three fixed-lifetime in-use cohorts (mu = 10, 20, 44 yr, Rostek et al.
2022 SI Table S4), each of which implies an *exponential* residence-time
density, with Chapter 2's Erlang serial chains: `n_z` compartments in series
at internal rate `kappa_z`, so that

    mean lifetime  tau_z = n_z / kappa_z                (Ch2 Eq. erlangmoments)
    density        g(l)  = kappa^n l^(n-1) e^(-kappa l) / (n-1)!
                                                        (Ch2 Eq. erlang_density)
    CV                   = 1 / sqrt(n)

Chapter 2 (`paper_revised.tex`, App. `subsec:cohort`) derives the chain, and
Table `tab:fiterr` reports that matching the reference coefficient of
variation of Gloser-Chahoud et al. (2013), `n_z = (tau_z / sigma_z)^2 ~ 33`,
takes the total-variation distance from 66.8% (fixed exponential) to 4.43%.
`n_z = 5` is Chapter 2's own intermediate comparison point (42.8%).

Why this is the intervention worth making
-----------------------------------------
Chapter 2's Section 4 result is that, holding the mean lifetime fixed, the
lifetime *shape* leaves the equilibrium, the asymptotic stability, the
dominant oscillation and the circularity indicators exactly unchanged --
with one exception, stated explicitly:

    "The distribution shape governs only the in-use relaxation mode and the
     fast, damped routing of material into waste management, which is why
     s_5 is the one stock visibly sensitive to it."

`s_5` is waste management, which is this model's **Scrap**.  Scrap is the
observable at 51.0% freerun relRMSE -- the only stock the published model
fits worse than persistence (31.4%).  So Chapter 2 predicts that the one
stock the shape can move is the one stock that is broken.  WP-10 tests that
prediction directly.

Pattern
-------
Follows `zinc_alpha_lab.py` / `zinc_weight_lab.py` (CLAUDE.md rule 1):
`zinc_colloc_v5.py` is imported and never edited, its MD5 is pinned, and every
behaviour change goes through `install()`, which appends a one-line
description to `PATCHES` so `--check` and every run log state what was
rebound.

What `install()` rebinds
------------------------
Six names, all of them ODE-side.  Stage A never integrates, so Stage A is
bit-identical under every arm; the entire intervention lives in Stage B and
in the free-run.

    N_ODE_STOCKS   2 + M + 1        where M = sum(n_chain)
    IDX_SCRAP      2 + M
    ode_to_4obs    sums M compartments into S_inuse
    _make_Y0       expands the IC over M compartments
    make_rhs       the serial-chain law of motion
    make_integrator  re-aggregates M compartments to 3 cohort totals on the
                     way out, so every downstream consumer of
                     `S_cohorts_year` keeps its (T, 3) shape

`compute_flows_from_nn` is deliberately NOT patched.  Its use-phase discharge
is `eol_k = max(S_cohorts, 0) / (MU_COHORTS + 1e-12)`, so passing it the
*effective* vector

    S_k_eff = max(u_{z,n_z}, 0) * gain_z ,   gain_z = n_z * exp(g_z)

reproduces the Erlang discharge `kappa_z * u_{z,n_z}` exactly, because
`kappa_z = gain_z / (MU_COHORTS + 1e-12)` by construction.  This keeps the
read-only core's flow algebra -- the whole mass-balance vector, all 19 flows
-- untouched and unduplicated.

n = (1,1,1) is an exact no-op
-----------------------------
Every gain is written so that the published model is recovered *bitwise*,
not merely to solver tolerance, when `n_chain = (1,1,1)` and `kappa` is
fixed:

    gain      = n * exp(0) = 1.0                     exact
    stage_time = (MU_COHORTS + 1e-12) / 1.0          exact
    outflow   = u / stage_time = u / (MU + 1e-12)    the core's own expression
    S_k_eff   = max(u,0) * 1.0 = max(u,0)            the core's own argument

`verify_noop()` asserts this against `anchor_v4/pred_seed*.npz` and must be
run before any Erlang arm is believed.  See `--check`.

Learnable kappa
---------------
The spec asks for learnable `kappa_z` with `n_z` fixed.  `params` is a list
of `{"W","b"}` layer dicts and `mlp_apply` reads only those two keys, so the
chain rates ride as a third key on the last layer:

    params[-1]["log_kappa"]  -- shape (3,), initialised to zeros

`kappa_z = n_z * exp(log_kappa_z) / (MU_COHORTS[z] + 1e-12)`, i.e. the mean
lifetime is `tau_z = mu_z * exp(-log_kappa_z)` and `log_kappa = 0` is exactly
Rostek.  Two consequences must be reported rather than buried:

  1. `optax.adamw(weight_decay=1e-4)` decays *every* leaf, `log_kappa`
     included.  That is a weak shrinkage prior toward the Rostek lifetimes.
     It is not a bug and it is not neutral; `arm_summary()` reports the
     implied half-life of the shrinkage alongside the fitted lifetimes.
  2. Stage A carries no ODE, so `log_kappa` receives no gradient there, and
     adamw leaves a zero leaf with zero gradient at zero.  Every lifetime is
     therefore learned in Stage B alone, from 2000 steps of a 22-year window
     curriculum.

Arms
----
    exp      n=(1,1,1)     kappa fixed      == anchor_v4, no fit needed
    expk     n=(1,1,1)     kappa learned    isolates "learnable mean lifetime"
    erl5     n=(5,5,5)     kappa fixed      Ch2's intermediate shape point
    erl33    n=(33,33,33)  kappa fixed      Ch2's CV-matched shape
    erl33k   n=(33,33,33)  kappa learned    the full WP-10 proposal

`erl5` / `erl33` hold the mean lifetime at Rostek and vary only the shape,
which is exactly the ceteris paribus Chapter 2's invariance result is stated
under.  `expk` varies only the mean.  `erl33k` varies both, and is the only
arm in which the two are confounded -- which is why the other three exist.
"""

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Pinned digests: the read-only core model and the data copy that produced
# anchor_v4 (SCHEMA §7 — four .xlsx copies with four different hashes exist).
CORE_MD5 = "8f57a3d702a0d7d9b0b406c663e9effa"      # zinc_colloc_v5.py
DATA_MD5 = "2349fe8e2a7872b3f4691210652da96b"      # final_model/zinc_dataset.xlsx

ANCHOR_DIR = os.path.join(HERE, "anchor_v4")
ANCHOR_CFG = os.path.join(ANCHOR_DIR, "config_used.json")

_TUPLE_KEYS = ("exog_feature_orders", "stock_term_weights")

PATCHES: list[str] = []

# ---------------------------------------------------------------------------
# arm table
# ---------------------------------------------------------------------------
# n_chain is per-cohort and FIXED (spec: integer, not learnable).
ARM_SPEC = {
    "exp":    dict(n_chain=(1, 1, 1),    learn_kappa=False),
    "expk":   dict(n_chain=(1, 1, 1),    learn_kappa=True),
    "erl5":   dict(n_chain=(5, 5, 5),    learn_kappa=False),
    "erl33":  dict(n_chain=(33, 33, 33), learn_kappa=False),
    "erl33k": dict(n_chain=(33, 33, 33), learn_kappa=True),
    # --- WP-10b (2026-09-24) -------------------------------------------------
    # erlz*: n_z from zinc's OWN lifetimes, Rostek 2022 SI Table S4, grouped by
    #   the cohort mapping in zinc_pinn_v12b_1.py (equal weights, which is what
    #   reproduces the long-cohort mean 43.8 ~ 44).  Mixture CV per cohort
    #   0.412 / 0.401 / 0.373 -> n = 1/CV^2 = 5.9 / 6.2 / 7.2 -> (6, 6, 7).
    #   See ZINC_S4_N below.
    # *kw: learnable kappa with a WIDENED step budget.  The effective log-rate
    #   is kappa_scale * log_kappa, so under Adam (step in log_kappa <= lr) the
    #   total movement of the mean is capped at kappa_scale * sum(lr) ~ 1.0 in
    #   log, i.e. a factor e^{+-1}, instead of +-9.5% (wp10_findings §1.3).
    "erlz":    dict(n_chain=(6, 6, 7),    learn_kappa=False),
    "erlzk":   dict(n_chain=(6, 6, 7),    learn_kappa=True, kappa_scale=10.0),
    "expkw":   dict(n_chain=(1, 1, 1),    learn_kappa=True, kappa_scale=10.0),
    "erl33kw": dict(n_chain=(33, 33, 33), learn_kappa=True, kappa_scale=10.0),
}

# Rostek et al. (2022) SI Table S4, mean (SD) yr, grouped to the three cohorts
# as in zinc_pinn_v12b_1.py; coinage (34, 11) is not assigned there.
ZINC_S4_COHORTS = {
    "short":  [(10, 4), (8, 3), (12, 3), (9, 5), (15, 3)],
    "medium": [(19, 8), (17, 9), (16, 4), (23, 6)],
    "long":   [(50, 23), (50, 23), (40, 7), (40, 7), (39, 7)],
}


def zinc_s4_n():
    """Equal-weight Gaussian-mixture CV per cohort and n = round(1/CV^2)."""
    out = {}
    for k, v in ZINC_S4_COHORTS.items():
        m = np.array([a for a, _ in v], float)
        s = np.array([b for _, b in v], float)
        M = m.mean()
        cv = np.sqrt((s ** 2 + m ** 2).mean() - M ** 2) / M
        out[k] = dict(mean=M, cv=cv, n=int(round(1.0 / cv ** 2)))
    return out
# `exp` is the published structure; it is reproduced, not refitted.
NO_FIT = ("exp",)
FIT_ARMS = [a for a in ARM_SPEC if a not in NO_FIT]

# Chapter 2's CV-matching rule, kept here so the choice of 33 is auditable.
CH2_SIGMA_OVER_TAU = 0.175          # Gloser-Chahoud "moderate" SD scenario
CH2_N_FROM_CV = int(round(1.0 / CH2_SIGMA_OVER_TAU ** 2))     # -> 33


def n_from_cv(cv):
    """Ch2 Eq. erlangmoments inverted: n = (tau/sigma)^2 = 1/CV^2."""
    return int(round(1.0 / float(cv) ** 2))


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
    for label, path, want in (
            ("zinc_colloc_v5.py", os.path.join(HERE, "zinc_colloc_v5.py"), CORE_MD5),
            ("zinc_dataset.xlsx", os.path.join(HERE, "zinc_dataset.xlsx"), DATA_MD5)):
        got = _md5(path)
        out[label] = (got, got == want)
        if strict and got != want:
            raise RuntimeError(
                f"{label} MD5 {got} != pinned {want}. The core model is "
                f"read-only (CLAUDE.md rule 1) and the dataset copy is the "
                f"one anchor_v4 was fitted on (SCHEMA §7). Refusing to run.")
    return out


def load_anchor_config():
    """The authoritative anchor_v4 config (SCHEMA §5), JSON types repaired."""
    with open(ANCHOR_CFG) as fh:
        cfg = json.load(fh)
    for k in _TUPLE_KEYS:
        if k in cfg and isinstance(cfg[k], list):
            cfg[k] = tuple(cfg[k])
    if isinstance(cfg.get("stageB_curriculum"), list):
        cfg["stageB_curriculum"] = [tuple(x) for x in cfg["stageB_curriculum"]]
    cfg["xlsx_path"] = os.path.join(HERE, os.path.basename(cfg["xlsx_path"]))
    return cfg


# ---------------------------------------------------------------------------
# the patch
# ---------------------------------------------------------------------------
_ARMED = {"n_chain": (1, 1, 1), "learn_kappa": False, "arm": "exp",
          "kappa_scale": 1.0}
_ORIG = {}


def arm(name=None, *, n_chain=None, learn_kappa=None):
    """Select the use-phase structure the next `train_model` will integrate.

    Idempotent and re-callable: `install()` reads `_ARMED` at ODE-build time,
    so arming after installing is fine, which is what the runner does.
    """
    if name is not None:
        if name not in ARM_SPEC:
            raise ValueError(f"unknown arm {name!r}; have {sorted(ARM_SPEC)}")
        spec = ARM_SPEC[name]
        n_chain = spec["n_chain"] if n_chain is None else n_chain
        learn_kappa = spec["learn_kappa"] if learn_kappa is None else learn_kappa
    if n_chain is None:
        n_chain = _ARMED["n_chain"]
    if learn_kappa is None:
        learn_kappa = _ARMED["learn_kappa"]
    n_chain = tuple(int(x) for x in n_chain)
    if len(n_chain) != 3 or any(x < 1 for x in n_chain):
        raise ValueError(f"n_chain must be 3 positive integers, got {n_chain}")
    kappa_scale = float(ARM_SPEC[name].get("kappa_scale", 1.0)) \
        if name is not None else 1.0
    _ARMED.update(n_chain=n_chain, learn_kappa=bool(learn_kappa),
                  arm=(name or "custom"), kappa_scale=kappa_scale)
    return dict(_ARMED)


def armed():
    return dict(_ARMED)


def _layout():
    """Static compartment layout for the armed chain lengths."""
    n = _ARMED["n_chain"]
    M = int(sum(n))
    off = np.concatenate([[0], np.cumsum(n)[:-1]]).astype(int)
    last = (off + np.asarray(n) - 1).astype(int)
    return n, M, off, last


def install(v5mod=None):
    """Rebind the six ODE-side names.  Idempotent.

    With `arm("exp")` (the default) every rebound function is algebraically
    AND bitwise identical to the one it replaces -- see `verify_noop`.
    """
    if v5mod is None:
        import zinc_colloc_v5 as v5mod
    if _ORIG:
        return PATCHES

    import jax
    import jax.numpy as jnp

    _ORIG["N_ODE_STOCKS"] = v5mod.N_ODE_STOCKS
    _ORIG["IDX_SCRAP"] = v5mod.IDX_SCRAP
    _ORIG["ode_to_4obs"] = v5mod.ode_to_4obs
    _ORIG["_make_Y0"] = v5mod._make_Y0
    _ORIG["make_rhs"] = v5mod.make_rhs
    _ORIG["make_integrator"] = v5mod.make_integrator
    _ORIG["init_mlp"] = v5mod.init_mlp

    MU = v5mod.MU_COHORTS
    N_COH = v5mod.N_COHORTS
    N_FLOWS = v5mod.N_FLOWS
    IDX_CONC, IDX_REF = v5mod.IDX_CONC, v5mod.IDX_REF
    IDX_INUSE_INFLOW = v5mod.IDX_INUSE_INFLOW

    # -- chain rate helpers -------------------------------------------------
    def _gain(params):
        """gain_z = n_z * exp(log_kappa_z).  Exactly 1.0 for n=1, g=0."""
        n, _, _, _ = _layout()
        n_vec = jnp.asarray(n, dtype=jnp.float64 if jax.config.jax_enable_x64
                            else jnp.float32)
        g = params[-1].get("log_kappa") if isinstance(params[-1], dict) else None
        if g is None:
            return n_vec
        s = float(_ARMED.get("kappa_scale", 1.0))
        if s != 1.0:                 # WP-10b wide-budget arms only; the
            g = s * g                # original arms take the unchanged path
        return n_vec * jnp.exp(g)

    def _stage_time(params):
        """Per-compartment residence time.  For n=1, g=0 this is MU + 1e-12,
        the core's own divisor, so the outflow expression is bitwise equal."""
        return (MU + 1e-12) / _gain(params)

    def kappa_of(params):
        """Internal chain rate kappa_z (yr^-1)."""
        return 1.0 / _stage_time(params)

    def mean_lifetime_of(params):
        """tau_z = n_z / kappa_z = n_z * stage_time_z."""
        n, _, _, _ = _layout()
        return jnp.asarray(n, dtype=_stage_time(params).dtype) * _stage_time(params)

    # -- N_ODE_STOCKS / IDX_SCRAP are read as module globals at call time,
    #    so they have to be rebound whenever the arm changes; `_sync_consts`
    #    is called from `arm()` via the runner and from every builder below.
    def _sync_consts():
        _, M, _, _ = _layout()
        v5mod.N_ODE_STOCKS = 2 + M + 1
        v5mod.IDX_SCRAP = 2 + M

    # -- ode_to_4obs --------------------------------------------------------
    def ode_to_4obs(S_ode):
        _, M, _, _ = _layout()
        S_inuse = jnp.sum(S_ode[..., 2:2 + M], axis=-1)
        return jnp.stack(
            [S_ode[..., IDX_CONC], S_ode[..., IDX_REF], S_inuse,
             S_ode[..., 2 + M]], axis=-1)

    # -- _make_Y0 -----------------------------------------------------------
    def _make_Y0(S0_4):
        """IC: Rostek between-cohort split, uniform within each chain.

        Uniform is the stationary shape: under constant inflow I_z, every
        compartment of chain z holds I_z / kappa_z, so cohort z holds
        n_z I_z / kappa_z = I_z tau_z (Little's law, Ch2 §4).  For n=1 this
        is `IC_COHORT_FRACS * S0[2]` unchanged.
        """
        n, M, off, _ = _layout()
        frac = np.asarray(v5mod.IC_COHORT_FRACS, float)
        per = np.concatenate([np.full(int(nz), frac[z] / float(nz))
                              for z, nz in enumerate(n)])
        S_cohorts_0 = jnp.asarray(per) * S0_4[2]
        Y0_stocks = jnp.concatenate([S0_4[:2], S_cohorts_0, S0_4[3:4]])
        return jnp.concatenate([Y0_stocks, jnp.zeros(N_FLOWS)])

    # -- make_rhs -----------------------------------------------------------
    def make_rhs(nn_eval):
        """Ch2 Eqs. erlang_first_stage / erlang_intermediate_stage /
        erlang_last_stage, with the manufacturing inflow split across chains
        by the network's f_cohort simplex."""
        n, M, off, last = _layout()
        n_stocks = 2 + M + 1
        idx_scrap = 2 + M

        def rhs(Y, t, params, data):
            S_ode = Y[:n_stocks]
            U = S_ode[2:2 + M]
            S_inuse_total = jnp.sum(U)
            S4 = jnp.array([S_ode[IDX_CONC], S_ode[IDX_REF],
                            S_inuse_total, S_ode[idx_scrap]])

            exog_t = v5mod.exog_fn(t, data["exog_times"], data["exog_values"])

            Uc = jnp.maximum(U, 0.0)
            st = _stage_time(params)                     # (3,)
            gain = _gain(params)                         # (3,)

            # Effective cohort vector handed to the untouched core flow
            # algebra so that its `eol_k = max(S,0)/(MU+1e-12)` evaluates to
            # the Erlang discharge kappa_z * u_{z,n_z}.
            u_last = Uc[last]
            S_k_eff = u_last * gain

            flows, f_cohort = v5mod.compute_flows_from_nn(
                nn_eval, params, t, S4, exog_t, data, S_cohorts=S_k_eff)

            cp = flows[0]
            cc = flows[1]
            refc = flows[2]
            w_in = flows[3]
            dr = flows[4]
            primary_refining = flows[6]
            waelz_recycling = flows[8]
            inuse_inflow = flows[IDX_INUSE_INFLOW]
            old_scrap_recovery = flows[17]
            first_use_new_scrap = flows[13]
            end_use_new_scrap = flows[15]

            # Serial chain: compartment r drains at u_r / stage_time_z into
            # r+1; the last drains out of use altogether.
            parts = []
            for z in range(N_COH):
                u_z = Uc[off[z]:off[z] + n[z]]
                out_z = u_z / st[z]
                upstream = jnp.concatenate(
                    [jnp.reshape(f_cohort[z] * inuse_inflow, (1,)), out_z[:-1]])
                parts.append(upstream - out_z)
            dU = jnp.concatenate(parts)

            dS_conc = cp - cc
            dS_ref = primary_refining + waelz_recycling - refc
            dS_scrap = (old_scrap_recovery + first_use_new_scrap
                        + end_use_new_scrap - w_in - dr)

            dSdt = jnp.concatenate([
                jnp.array([dS_conc, dS_ref]), dU, jnp.array([dS_scrap])])
            return jnp.concatenate([dSdt, flows])
        return rhs

    # -- make_integrator ----------------------------------------------------
    def make_integrator(integrator_kind, nn_eval, adjoint=None):
        """Same contract as the core's: returns (S_year, F_int, S_cohorts_year)
        with `S_cohorts_year` re-aggregated to (T, 3) so that every downstream
        consumer -- `predictions`, `diagnose`, `zinc_A_lab.dump_A`, the fig-4
        cohort panel -- keeps working unchanged."""
        _sync_consts()
        n, M, off, _ = _layout()
        n_stocks = 2 + M + 1
        rhs = make_rhs(nn_eval)

        def _agg(S_ode):
            cols = [jnp.sum(S_ode[:, 2 + off[z]:2 + off[z] + n[z]], axis=1)
                    for z in range(N_COH)]
            return jnp.stack(cols, axis=-1)

        if integrator_kind == "diffrax":
            if not v5mod.HAS_DIFFRAX:
                raise ImportError("diffrax not installed.")
            import diffrax as dfx

            def integrate(params, data, S0, years):
                years = jnp.asarray(years)
                rhs_data = v5mod._make_rhs_data(data)

                def f(t, y, args):
                    p, d = args
                    return rhs(y, t, p, d)

                Y0 = _make_Y0(S0)
                sol = dfx.diffeqsolve(
                    dfx.ODETerm(f), dfx.Tsit5(),
                    t0=years[0], t1=years[-1],
                    dt0=jnp.asarray(0.1, dtype=years.dtype),
                    y0=Y0, args=(params, rhs_data),
                    saveat=dfx.SaveAt(ts=years),
                    stepsize_controller=dfx.PIDController(rtol=RTOL[0],
                                                          atol=ATOL[0]),
                    max_steps=MAX_STEPS[0],
                    **({} if adjoint is None else {"adjoint": adjoint}),
                )
                Y = sol.ys
                S_ode = Y[:, :n_stocks]
                S_year = ode_to_4obs(S_ode)
                C_year = Y[:, n_stocks:]
                return S_year, C_year[1:] - C_year[:-1], _agg(S_ode)
        else:
            if adjoint is not None:
                raise ValueError("adjoint only supported by the diffrax backend")

            def integrate(params, data, S0, years):
                rhs_data = v5mod._make_rhs_data(data)
                Y0 = _make_Y0(S0)
                Y = v5mod.odeint(rhs, Y0, years, params, rhs_data)
                S_ode = Y[:, :n_stocks]
                S_year = ode_to_4obs(S_ode)
                C_year = Y[:, n_stocks:]
                return S_year, C_year[1:] - C_year[:-1], _agg(S_ode)

        return integrate

    # -- init_mlp -----------------------------------------------------------
    def init_mlp(layer_sizes, key):
        """The core's He init, plus the chain-rate leaf on the last layer when
        the armed arm learns kappa.  `mlp_apply` reads only W and b, and
        `train_model`'s head surgery only rewrites b, so the extra key rides
        through untouched and optax sees it as one more (3,) leaf."""
        params = _ORIG["init_mlp"](layer_sizes, key)
        if _ARMED["learn_kappa"]:
            params[-1] = dict(params[-1])
            params[-1]["log_kappa"] = jnp.zeros((N_COH,))
        return params

    v5mod.ode_to_4obs = ode_to_4obs
    v5mod._make_Y0 = _make_Y0
    v5mod.make_rhs = make_rhs
    v5mod.make_integrator = make_integrator
    v5mod.init_mlp = init_mlp
    _sync_consts()

    # exported so callers (and the runner's dumps) can read the fitted rates
    globals()["kappa_of"] = kappa_of
    globals()["mean_lifetime_of"] = mean_lifetime_of
    globals()["_sync_consts"] = _sync_consts

    PATCHES.append(
        "zinc_colloc_v5.{N_ODE_STOCKS,IDX_SCRAP,ode_to_4obs,_make_Y0,"
        "make_rhs,make_integrator,init_mlp} -> Erlang serial-chain use phase "
        "(Ch2 App. subsec:cohort); no-op at n_chain=(1,1,1), kappa fixed")
    return PATCHES


# `max_steps` and the solver tolerances are one-element lists so a diagnostic
# can change them without rebuilding the patch.  The DEFAULTS ARE THE CORE'S
# OWN VALUES (`zinc_colloc_v5.py:1350`) and every fitted arm runs on them;
# only `verify_gradient` tightens them, and it restores them afterwards.
MAX_STEPS = [20000]
RTOL = [1e-5]
ATOL = [1e-7]


def uninstall(v5mod=None):
    """Restore the core's own bindings (used by `verify_noop`)."""
    if not _ORIG:
        return
    if v5mod is None:
        import zinc_colloc_v5 as v5mod
    for k, v in _ORIG.items():
        setattr(v5mod, k, v)
    _ORIG.clear()
    PATCHES.clear()


# ---------------------------------------------------------------------------
# derived structural quantities (no fit needed)
# ---------------------------------------------------------------------------
def lifetime_density(n, tau, ell):
    """Ch2 Eq. erlang_density, parameterised by the mean tau rather than kappa."""
    n = int(n)
    kappa = n / float(tau)
    ell = np.asarray(ell, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        logg = (n * math.log(kappa) + (n - 1) * np.log(np.where(ell > 0, ell, 1.0))
                - kappa * ell - math.lgamma(n))
    g = np.exp(logg)
    return np.where(ell > 0, g, np.where(n == 1, kappa, 0.0))


def tv_distance(n_a, tau_a, n_b, tau_b, ell_max=None, n_grid=200001):
    """Total-variation distance IAE/2 between two Erlang densities.
    Ch2 Table tab:fiterr reports exactly this statistic."""
    ell_max = ell_max or 12.0 * max(tau_a, tau_b)
    ell = np.linspace(0.0, ell_max, n_grid)
    ga = lifetime_density(n_a, tau_a, ell)
    gb = lifetime_density(n_b, tau_b, ell)
    return 0.5 * float(np.trapezoid(np.abs(ga - gb), ell))


def arm_structure(name):
    """Everything about an arm that is known before any fit."""
    spec = ARM_SPEC[name]
    n = spec["n_chain"]
    mu = [10.0, 20.0, 44.0]
    rows = []
    for z, (nz, mz) in enumerate(zip(n, mu)):
        rows.append(dict(
            arm=name, cohort=["short", "medium", "long"][z], n_chain=int(nz),
            mu_rostek_yr=mz, kappa_init=nz / mz, cv=1.0 / math.sqrt(nz),
            sd_yr=mz / math.sqrt(nz),
            tv_vs_exponential=tv_distance(nz, mz, 1, mz) if nz > 1 else 0.0,
            learn_kappa=bool(spec["learn_kappa"])))
    return rows


# ---------------------------------------------------------------------------
# rehydration + no-op verification
# ---------------------------------------------------------------------------
WEIGHTS_DIR = os.path.join(HERE, "analysis", "wp2a")


def load_params(seed, weights_dir=None):
    """Stage-B MLP weights from a `zinc_A_lab` dump (`param{i}_{W,b}`).

    Same accessor `zinc_cf_lab` / `zinc_xai_lab` use, so WP-10's
    no-refit arm reads exactly the weights WP-7 and WP-8 read.
    """
    import jax.numpy as jnp
    path = os.path.join(weights_dir or WEIGHTS_DIR, f"A_seed{int(seed)}.npz")
    d = np.load(path, allow_pickle=True)
    n = sum(1 for k in d.files if k.startswith("param") and k.endswith("_W"))
    if n == 0:
        raise ValueError(f"{path} carries no param*_W arrays")
    return [{"W": jnp.asarray(d[f"param{i}_W"]),
             "b": jnp.asarray(d[f"param{i}_b"])} for i in range(n)]


def init_fit(cfg=None, seed=0):
    """A zero-step run: the data objects, layout and closures `train_model`
    builds, without any training.  Follows `zinc_cf_lab.init_fit`."""
    import zinc_colloc_v5 as v5
    cfg = dict(cfg or load_anchor_config())
    cfg["verbose"] = False
    return v5.run("wp10_init", **dict(cfg, seed=int(seed),
                                      stageA_steps=0, do_stage_B=False))


def verify_noop(seeds=(0, 1, 2), verbose=True):
    """Require the patched ODE at n=(1,1,1) to reproduce anchor_v4 BITWISE.

    No refit is needed: `analysis/wp2a/A_seed*.npz` carries the published
    Stage-B weights, and re-integrating them through the *unpatched* core
    already reproduces `anchor_v4/pred_seed*.npz` exactly (verified here as
    `rehydration`).  The no-op claim is then a comparison of two integrators
    on identical weights, which is the sharpest form the check can take —
    a refit would confound the patch with optimiser nondeterminism.

    Returns one row per seed; every `exact` flag must be True before any
    Erlang arm is interpreted.
    """
    import zinc_colloc_v5 as v5
    import jax

    integrity_check()
    fit = init_fit()
    data, S0, years = fit.data_all, fit.data_all["stocks_obs"][0], fit.data_all["years"]
    f_idx = np.asarray(fit.flow_obs_to_pred_idx, int)

    # Reference integrator built BEFORE install(), off the core's own names.
    integ_ref = v5.make_integrator("diffrax", fit.nn_eval)
    install(v5)
    arm("exp")
    integ_pat = v5.make_integrator("diffrax", fit.nn_eval)

    rows = []
    for sd in seeds:
        p = load_params(sd)
        Sr, Fr, Cr = (np.asarray(x) for x in integ_ref(p, data, S0, years))
        Sp, Fp, Cp = (np.asarray(x) for x in integ_pat(p, data, S0, years))
        ref = np.load(os.path.join(ANCHOR_DIR, f"pred_seed{sd}.npz"),
                      allow_pickle=True)
        rec = dict(
            seed=int(sd),
            rehydration_max_abs_dS=float(np.abs(Sr - ref["S_pred_B"]).max()),
            rehydration_max_abs_dF=float(np.abs(Fr[:, f_idx] - ref["F_pred_B"]).max()),
            noop_max_abs_dS=float(np.abs(Sp - Sr).max()),
            noop_max_abs_dF=float(np.abs(Fp - Fr).max()),
            noop_max_abs_dCohort=float(np.abs(Cp - Cr).max()))
        rec["rehydration_exact"] = (rec["rehydration_max_abs_dS"] == 0.0
                                    and rec["rehydration_max_abs_dF"] == 0.0)
        rec["exact"] = (rec["noop_max_abs_dS"] == 0.0
                        and rec["noop_max_abs_dF"] == 0.0
                        and rec["noop_max_abs_dCohort"] == 0.0)
        rows.append(rec)
        if verbose:
            print(f"  seed {sd}: rehydration max|dS|={rec['rehydration_max_abs_dS']:.3e} "
                  f"| no-op max|dS|={rec['noop_max_abs_dS']:.3e} "
                  f"max|dF|={rec['noop_max_abs_dF']:.3e}  "
                  f"{'BITWISE EXACT' if rec['exact'] else 'NOT EXACT'}")
        jax.clear_caches()
    return rows


def verify_gradient(seed=0, arm_name="erl33k"):
    """Is `log_kappa` actually trained, and is its gradient right?

    It rides as a third key on the last layer dict, which is a deliberate
    piggyback on `mlp_apply` reading only W and b -- so it deserves a check
    rather than an assumption.  Three sources are compared on the same
    objective (log-stock MSE on the free run):

        default adjoint (RecursiveCheckpointAdjoint, what `train_model` uses)
        DirectAdjoint   (CLAUDE.md rule 4)
        central finite differences

    at the production solver tolerance and at rtol = 1e-10.  The finite
    difference is the reason the tolerance sweep is needed: COMPUTE_STATUS
    flag 3 records that re-integrating at rtol 1e-11 instead of 1e-5 moves
    the Stage B residual by up to 4.8e-3, so at the production tolerance a
    small-eps difference quotient is pure solver noise and says nothing.
    """
    import zinc_colloc_v5 as v5
    import jax
    import jax.numpy as jnp
    import diffrax as dfx

    integrity_check()
    install(v5)
    arm(arm_name)
    fit = init_fit()
    p = load_params(seed)
    p[-1] = dict(p[-1])
    p[-1]["log_kappa"] = jnp.zeros((3,))
    data = fit.data_all
    S_obs = jnp.asarray(data["stocks_obs"])

    def make_loss(adjoint):
        integ = v5.make_integrator("diffrax", fit.nn_eval, adjoint=adjoint)

        def loss(params):
            S, _, _ = integ(params, data, data["stocks_obs"][0], data["years"])
            return jnp.mean((jnp.log(jnp.maximum(S, 1.0))
                             - jnp.log(jnp.maximum(S_obs, 1.0))) ** 2)
        return loss

    def bump(i, e):
        q = [dict(l) for l in p]
        v = np.array(q[-1]["log_kappa"])
        v[i] += e
        q[-1]["log_kappa"] = jnp.asarray(v)
        return q

    rt0, at0 = RTOL[0], ATOL[0]
    rows = []
    try:
        for rtol, atol in ((rt0, at0), (1e-10, 1e-12)):
            RTOL[0], ATOL[0] = rtol, atol
            L, Ld = make_loss(None), make_loss(dfx.DirectAdjoint())
            g_def = np.asarray(jax.grad(L)(p)[-1]["log_kappa"], float)
            g_dir = np.asarray(jax.grad(Ld)(p)[-1]["log_kappa"], float)
            for eps in (1e-2, 1e-1):
                fd = np.array([(float(L(bump(i, eps))) - float(L(bump(i, -eps))))
                               / (2 * eps) for i in range(3)])
                for z, nm in enumerate(("short", "medium", "long")):
                    rows.append(dict(rtol=rtol, eps=eps, cohort=nm,
                                     grad_default=g_def[z], grad_direct=g_dir[z],
                                     grad_findiff=float(fd[z]),
                                     rel_gap_direct_vs_fd=abs(g_dir[z] - fd[z])
                                     / max(abs(fd[z]), 1e-30)))
            jax.clear_caches()
    finally:
        RTOL[0], ATOL[0] = rt0, at0
    return rows


def shape_counterfactual(seeds, arms=("exp", "erl5", "erl33"), cfg=None,
                         weights_dir=None):
    """Re-integrate the PUBLISHED weights under each chain length, no refit.

    This is Chapter 2's ceteris paribus exactly: the network, every alpha,
    every tau and the mean lifetimes are held at their published values and
    only the shape of the residence-time density changes.  Whatever Scrap
    does here is the structural effect alone, with the re-estimation effect
    removed by construction -- which the refit arms cannot separate.

    Metrics are reported on TWO windows, because they are not interchangeable:

      `all`   1980-2019, every year of the free run.
      `test`  2007-2019, the pre-registered held-out window, which is the
              window `fit.diagnose` uses and therefore the only one
              comparable with the spec's headline numbers (Scrap 51.02).

    Returns (rows, trajectories) with `trajectories[arm][seed]` =
    (S_pred (T,4), F_int (T-1,19), S_cohorts (T,3)).
    """
    import zinc_colloc_v5 as v5
    import jax

    integrity_check()
    install(v5)
    fit = init_fit(cfg)
    data, S0, years = fit.data_all, fit.data_all["stocks_obs"][0], fit.data_all["years"]
    yrs = np.asarray(years, float).ravel()
    S_obs = np.asarray(data["stocks_obs"], float)
    F_obs = np.asarray(data["flows_obs"], float)
    f_idx = np.asarray(fit.flow_obs_to_pred_idx, int)
    fnames = [v5.FLOW_NAMES[i] for i in f_idx]

    y_test = np.asarray(fit.data_test["years"], float).ravel()
    m_stock = {"all": np.ones(len(yrs), bool), "test": np.isin(yrs, y_test)}
    yrs_flow = yrs[1:]
    m_flow = {"all": np.ones(len(yrs_flow), bool),
              "test": np.isin(yrs_flow, y_test)}

    rows, traj = [], {a: {} for a in arms}
    for a in arms:
        arm(a)
        integ = v5.make_integrator("diffrax", fit.nn_eval)
        for sd in seeds:
            p = load_params(sd, weights_dir)
            S, F, C = (np.asarray(x) for x in integ(p, data, S0, years))
            traj[a][int(sd)] = (S, F, C)
            Fm = F[:, f_idx]
            for win in ("all", "test"):
                ms, mf = m_stock[win], m_flow[win]
                base = dict(arm=a, seed=int(sd), window=win,
                            n_chain=str(list(_ARMED["n_chain"])))
                per_stock = []
                for k, nm in enumerate(v5.STOCK_NAMES):
                    v = _rel_rmse_pct(S[ms, k], S_obs[ms, k])
                    per_stock.append(v)
                    rows.append(dict(base, family="stock", component=nm, relRMSE=v))
                rows.append(dict(base, family="stock_mean", component="<stock-mean>",
                                 relRMSE=float(np.nanmean(per_stock))))
                per_flow = []
                for j, nm in enumerate(fnames):
                    v = _rel_rmse_pct(Fm[mf, j], F_obs[mf, j])
                    per_flow.append(v)
                    rows.append(dict(base, family="flow", component=nm, relRMSE=v))
                rows.append(dict(base, family="flow_mean", component="<flow-mean>",
                                 relRMSE=float(np.nanmean(per_flow))))
        jax.clear_caches()
    return rows, traj


def _rel_rmse_pct(pred, obs):
    """relRMSE with the project's convention: denominator mean(|obs|)."""
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    m = np.isfinite(pred) & np.isfinite(obs)
    if not m.any():
        return float("nan")
    return 100.0 * np.sqrt(np.mean((pred[m] - obs[m]) ** 2)) / np.mean(np.abs(obs[m]))


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------
def check(verbose=True, arm_name=None):
    """Resolve drivers, input_dim and the armed chain layout without fitting."""
    import zinc_colloc_v5 as v5

    digests = integrity_check()
    install(v5)
    if arm_name:
        arm(arm_name)
    _sync_consts()
    cfg = load_anchor_config()

    data_np = v5.load_zinc_data(cfg["xlsx_path"],
                                extra_exog_cols=cfg.get("extra_exog_cols"))
    exog_cols = list(data_np["exog_cols"])
    years = np.asarray(data_np["years"], float).ravel()

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
    input_dim = 1 + 4 + exog_proc.shape[1]
    n, M, off, last = _layout()

    info = dict(exog_cols=exog_cols, n_universe=len(exog_cols), orders=orders,
                input_dim=input_dim, n_exog_features=int(exog_proc.shape[1]),
                years=(float(years[0]), float(years[-1])), T=T,
                alpha_names=list(v5.ALPHA_NAMES),
                tau_sup_names=list(v5.TAU_SUP_NAMES),
                digests=digests, patches=list(PATCHES),
                arm=_ARMED["arm"], n_chain=list(n), learn_kappa=_ARMED["learn_kappa"],
                M_compartments=M, N_ODE_STOCKS=int(v5.N_ODE_STOCKS),
                IDX_SCRAP=int(v5.IDX_SCRAP),
                mu_cohorts=[float(x) for x in v5.MU_COHORTS_NP],
                structure=arm_structure(_ARMED["arm"]) if _ARMED["arm"] in ARM_SPEC else None)

    if verbose:
        print("=" * 74)
        print("zinc_erlang_lab --check   (WP-10, Erlang use-phase extension)")
        print("=" * 74)
        for label, (got, ok) in digests.items():
            print(f"  {label:20s} md5 {got}  {'OK' if ok else 'MISMATCH'}")
        print(f"  patches applied      : {PATCHES[0] if PATCHES else 'none'}")
        print(f"  xlsx_path            : {cfg['xlsx_path']}")
        print(f"  years                : {years[0]:.0f}–{years[-1]:.0f}  (T={T})")
        print(f"  input_dim            : {input_dim}   "
              f"(1 t + 4 S + {exog_proc.shape[1]} exog = "
              f"{len(exog_cols)} drivers x {len(orders)} orders)")
        print(f"  exog_feature_orders  : {orders}")
        print(f"\n  resolved drivers (canonical order, {len(exog_cols)}):")
        for i, c in enumerate(exog_cols):
            print(f"    [{i:2d}] {c}")
        print(f"\n  armed arm            : {_ARMED['arm']}")
        print(f"  n_chain              : {list(n)}  (fixed, not learnable)")
        print(f"  learn_kappa          : {_ARMED['learn_kappa']}")
        print(f"  compartments M       : {M}   offsets {list(off)}  last {list(last)}")
        print(f"  N_ODE_STOCKS         : {v5.N_ODE_STOCKS}   IDX_SCRAP {v5.IDX_SCRAP}")
        print(f"  augmented ODE dim    : {v5.N_ODE_STOCKS + v5.N_FLOWS}")
        if info["structure"]:
            print(f"\n  {'cohort':8s} {'n':>3s} {'mu(yr)':>7s} {'kappa0':>8s} "
                  f"{'CV':>6s} {'SD(yr)':>7s} {'TV vs exp':>10s}")
            for r in info["structure"]:
                print(f"  {r['cohort']:8s} {r['n_chain']:3d} {r['mu_rostek_yr']:7.1f} "
                      f"{r['kappa_init']:8.4f} {r['cv']:6.3f} {r['sd_yr']:7.2f} "
                      f"{r['tv_vs_exponential']*100:9.2f}%")
        print("=" * 74)
    return info


def main():
    ap = argparse.ArgumentParser(description="WP-10 Erlang use-phase lab")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--arm", default=None, choices=sorted(ARM_SPEC))
    ap.add_argument("--verify-noop", action="store_true",
                    help="re-integrate the published weights at n=(1,1,1) and "
                         "require bitwise anchor_v4 reproduction")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()

    if args.check or not (args.verify_noop):
        check(arm_name=args.arm)
    if args.verify_noop:
        print("\nverify_noop: re-integrating published weights at n=(1,1,1) …")
        rows = verify_noop([int(s) for s in args.seeds.split(",") if s.strip()])
        ok = all(r["exact"] for r in rows)
        print(f"\n  no-op verification: {'PASS (bitwise)' if ok else 'FAIL'}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
