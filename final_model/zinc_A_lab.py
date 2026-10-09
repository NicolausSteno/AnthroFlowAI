#!/usr/bin/env python3
"""
zinc_A_lab.py — lab module for WP-2: the time-varying transfer matrix A(t)
==========================================================================

WP-2a assembles, for every seed of the `anchor_v4` ensemble and every year
node, the 6x6 transfer matrix of the augmented state

    S_ode = [S_conc, S_ref, S_iu_short, S_iu_med, S_iu_long, S_scrap]

such that                                    # Ch2 Eq. system  (dS/dt = A s + b)
                                             # Ch2 Eq. a1      (the 8-stock A)
    dS_ode/dt = A(t) . S_ode + b(t),         # Ch2 Eq. vecb    (b = [E(t),0,...])

with b(t) = [cp(t), 0, 0, 0, 0, 0] — concentrate production is the only
exogenous inflow, exactly Chapter 2's E(t).

Pattern
-------
Follows `zinc_alpha_lab.py` (CLAUDE.md rule 1): `zinc_colloc_v5.py` is
imported and never edited, its MD5 is pinned, and behaviour changes would go
through `install()`.  WP-2a needs no patch — the coefficient set is fully
reachable through `FitResult.predictions()`, it is simply never written to
disk (SCHEMA §1).  Integrity checking, config loading and `--check` are
imported from `zinc_alpha_lab` rather than duplicated.

Why a refit is needed
---------------------
`run_anchor.py` persists stocks and flows only; α, τ and the manufacturing
splits are discarded (SCHEMA §1, §3), and the WP-1 dumps added α, τ and cp
but not `f_cohort` — which is what distributes `inuse_inflow` across the
three cohort rows and is therefore required for the 6-state form.  Fitted
weights are not persisted anywhere either.  So each seed is refitted (bit-
reproducible; verified against the stored rollouts, as in WP-1a) and this
module additionally dumps the raw MLP weights so no later work package in
WP-2 has to pay for the refit again.

The algebra
-----------
From `zinc_colloc_v5.make_rhs` / `compute_flows_from_nn` (v5:1132–1285), with
`fu = frac_fu = [new, loss, out]`, `eu = frac_eu = [new, loss, into_use]`:

    parent_manu = alpha_refc*S_ref + alpha_dr*S_scrap
    into_use    = eu[2]*fu[2] * parent_manu
    inflow      = (1 - tau_diss) * into_use          -> in-use cohorts
    newscrap    = (fu[0] + eu[0]*fu[2]) * parent_manu -> S_scrap

giving, with mu = MU_COHORTS = [10, 20, 44] yr and `g = 1/(mu + 1e-12)`
(the code's own guarded reciprocal):

    A[0,0] = -alpha_cc
    A[1,0] = (1 - tau_ref)*alpha_cc      A[1,1] = -alpha_refc
    A[1,5] = tau_waelz*alpha_win
    A[2+k,1] = f_cohort[k]*inflow_share*alpha_refc
    A[2+k,5] = f_cohort[k]*inflow_share*alpha_dr
    A[2+k,2+k] = -g[k]
    A[5,1]   = newscrap_share*alpha_refc
    A[5,2+k] = tau_olds*g[k]
    A[5,5]   = newscrap_share*alpha_dr - (alpha_win + alpha_dr)

Mass conservation is a *column* statement: summing column j gives minus the
per-unit-stock rate at which mass leaves the system from stock j,

    1^T A(t) = -l(t)^T,
    l = [tau_ref*alpha_cc,
         alpha_refc*(fu[1] + fu[2]*(eu[1] + tau_diss*eu[2])),
         (1 - tau_olds)*g[k]   (k = 0,1,2),
         (1 - tau_waelz)*alpha_win + alpha_dr*(fu[1] + fu[2]*(eu[1] + tau_diss*eu[2]))]

so d(1^T S)/dt = cp - l^T S: the cycle gains mass only through concentrate
production and loses it only through the six named loss flows.  `l` is
assembled independently of `A` from the flow definitions, so agreement
between `1^T A` and `-l` is a real check on the assembly rather than a
tautology.  Two further checks are run per seed and year:

  * the **RHS identity** — `A(t) S + b(t)` against `make_rhs(nn_eval)` itself,
    evaluated at the ODE's own state and (with a randomised cohort split) at
    the observed state.  This is the definitive test: it compares against the
    integrated model rather than against a second copy of the same algebra;
  * the **Metzler property** — off-diagonal entries >= 0 (Ch2 assumes it in
    deriving Eq. uniform_column_condition), plus the resulting uniform
    exponential-stability margin `eta = -max_t max_j sum_i A_ij(t)`.

Which evaluation point
----------------------
`anchor_v4` sets `use_stock_input=False`, so the MLP never sees the state
(`_norm_inputs`, v5:503, multiplies the stock block by 0).  The coefficients
are therefore identical at observed and at predicted stocks — verified per
seed and reported as `max_abs_A_state_dependence`, not assumed.  Two variants
are still dumped because the *pinned* components differ:

    A      : pinned components replaced by their exact reported values, with
             the manufacturing simplices re-closed (`_reporting_coeffs`).
             The reporting object — this is the estimated A(t).
    A_ode  : from `nn_eval` raw, i.e. what the integrator actually used.
             Pinned components come from `_interval_pick`, which at t = years[0]
             clips to index 1; the two therefore differ at 1980 only.

Note that `predictions()["*_at_obs"]` cannot be used directly: its override
overwrites one entry of each manufacturing simplex and leaves the other two,
so the simplex no longer sums to 1 and the assembled A violates mass balance.
`_reporting_coeffs` re-closes it instead; the cost of not doing so is
recorded per seed as `max_abs_col_sum_residual_naive_at_obs`.

CLI
---
    python zinc_A_lab.py --check
    python zinc_A_lab.py --seeds 0,1,2 --out analysis/wp2a
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

import zinc_alpha_lab as alab
from zinc_alpha_lab import integrity_check, load_anchor_config   # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR_DEFAULT = os.path.join(HERE, "analysis", "wp2a")

# Augmented state order, fixed by `_make_Y0` / `make_rhs` (v5:1230, 1247).
ODE_STATE_NAMES = ["S_conc", "S_ref", "S_iu_short", "S_iu_med", "S_iu_long",
                   "S_scrap"]
N_ODE = 6

PATCHES: list[str] = []          # WP-2a needs none; kept for later packages


def install(v5mod=None):
    """Apply import-time patches to the core module.  None needed for WP-2a."""
    if v5mod is None:
        import zinc_colloc_v5 as v5mod          # noqa: F401
    return PATCHES


# ---------------------------------------------------------------------------
# the assembly
# ---------------------------------------------------------------------------
def assemble_A(alphas, taus, frac_fu, frac_eu, f_cohort, cp, mu=None):
    """Build A(t) (T,6,6) and b(t) (T,6) from a coefficient time series.

    Pure numpy, no JAX, no model state — every argument is a per-year array
    of NN outputs.  Column order of `alphas` is `ALPHA_NAMES`
    (alpha_cc, alpha_refc, alpha_win, alpha_dr); of `taus`, `TAU_BINARY_NAMES`
    (tau_ref, tau_waelz, tau_olds, tau_diss); `frac_fu` / `frac_eu` are the
    3-simplices [new, loss, out] / [new, loss, into_use]; `f_cohort` is the
    3-simplex over MU_COHORTS.

    # Ch2 Eq. system, Eq. a1, Eq. vecb
    """
    import zinc_colloc_v5 as v5

    mu = np.asarray(v5.MU_COHORTS_NP if mu is None else mu, float)
    alphas = np.asarray(alphas, float)
    taus = np.asarray(taus, float)
    frac_fu = np.asarray(frac_fu, float)
    frac_eu = np.asarray(frac_eu, float)
    f_cohort = np.asarray(f_cohort, float)
    cp = np.asarray(cp, float).ravel()
    T = alphas.shape[0]
    nk = mu.size

    a_cc, a_refc, a_win, a_dr = alphas.T
    t_ref, t_wae, t_old, t_dis = taus.T
    fu_new, fu_loss, fu_out = frac_fu.T
    eu_new, eu_loss, eu_into = frac_eu.T

    # `compute_flows_from_nn` guards the reciprocal the same way (v5:1187).
    g = 1.0 / (mu + 1e-12)

    into_use_share = eu_into * fu_out            # of parent_manu
    inflow_share = (1.0 - t_dis) * into_use_share
    newscrap_share = fu_new + eu_new * fu_out

    A = np.zeros((T, N_ODE, N_ODE))
    A[:, 0, 0] = -a_cc
    A[:, 1, 0] = (1.0 - t_ref) * a_cc
    A[:, 1, 1] = -a_refc
    A[:, 1, 5] = t_wae * a_win
    for k in range(nk):
        A[:, 2 + k, 1] = f_cohort[:, k] * inflow_share * a_refc
        A[:, 2 + k, 5] = f_cohort[:, k] * inflow_share * a_dr
        A[:, 2 + k, 2 + k] = -g[k]
        A[:, 5, 2 + k] = t_old * g[k]
    A[:, 5, 1] = newscrap_share * a_refc
    A[:, 5, 5] = newscrap_share * a_dr - (a_win + a_dr)

    b = np.zeros((T, N_ODE))
    b[:, 0] = cp                                  # Ch2 Eq. vecb: b = [E(t), 0...]
    return A, b


def loss_rates(alphas, taus, frac_fu, frac_eu, mu=None):
    """Per-unit-stock rate at which mass leaves the system, l(t) (T,6).

    Assembled directly from the six loss flows of `compute_flows_from_nn`
    (refinery, waelz, first-use, end-use, dissipative, end-of-life), NOT from
    `A`, so that `1^T A + l^T = 0` is an independent check on the assembly.
    """
    import zinc_colloc_v5 as v5

    mu = np.asarray(v5.MU_COHORTS_NP if mu is None else mu, float)
    alphas, taus = np.asarray(alphas, float), np.asarray(taus, float)
    frac_fu, frac_eu = np.asarray(frac_fu, float), np.asarray(frac_eu, float)
    a_cc, a_refc, a_win, a_dr = alphas.T
    t_ref, t_wae, t_old, t_dis = taus.T
    _fu_new, fu_loss, fu_out = frac_fu.T
    _eu_new, eu_loss, eu_into = frac_eu.T
    g = 1.0 / (mu + 1e-12)

    # share of `parent_manu` lost in manufacturing, use-phase entry and
    # dissipation: first_use_losses + end_use_losses + dissipative_use
    manu_loss_share = fu_loss + fu_out * (eu_loss + t_dis * eu_into)

    ell = np.zeros((alphas.shape[0], N_ODE))
    ell[:, 0] = t_ref * a_cc                              # refinery_losses
    ell[:, 1] = a_refc * manu_loss_share
    for k in range(mu.size):
        ell[:, 2 + k] = (1.0 - t_old) * g[k]              # eol_losses
    ell[:, 5] = (1.0 - t_wae) * a_win + a_dr * manu_loss_share
    return ell


def obs_to_ode_state(S4, cohort_shares):
    """Lift the 4 observable stocks to the 6-state, splitting In-Use by
    `cohort_shares` (T,3), rows summing to 1."""
    S4 = np.asarray(S4, float)
    w = np.asarray(cohort_shares, float)
    return np.concatenate([S4[:, :2], w * S4[:, 2:3], S4[:, 3:4]], axis=1)


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------
def rhs_dSdt(fit, params, S_ode, years):
    """dS/dt from the core model's own RHS at each (year, 6-state) pair.

    Uses `make_rhs(fit.nn_eval)` — the exact closure the integrator runs — so
    the comparison is against the fitted dynamics, not against a second
    transcription of the same equations.
    """
    import jax
    import jax.numpy as jnp
    import zinc_colloc_v5 as v5

    rhs = v5.make_rhs(fit.nn_eval)
    rhs_data = v5._make_rhs_data(fit.data_all)
    S_ode = jnp.asarray(np.asarray(S_ode, float))
    years = jnp.asarray(np.asarray(years, float).ravel())

    def one(t, S6):
        Y = jnp.concatenate([S6, jnp.zeros(v5.N_FLOWS)])
        return rhs(Y, t, params, rhs_data)[:v5.N_ODE_STOCKS]

    return np.asarray(jax.vmap(one)(years, S_ode))


def verify(A, b, ell, years, *, fit=None, params=None,
           S_ode_pred=None, S_ode_obs=None):
    """Column sums, Metzler property and (optionally) the RHS identity.

    The RHS identity is only meaningful for the coefficient set the
    integrator actually used, so it is skipped when `fit` is None.
    Returns a dict of scalars plus per-year residual arrays.
    """
    col_sum = A.sum(axis=1)                                   # (T,6)
    col_resid = col_sum + ell
    scale = np.maximum(np.abs(A).max(axis=(1, 2)), 1e-300)     # (T,)

    offdiag = A.copy()
    idx = np.arange(N_ODE)
    offdiag[:, idx, idx] = 0.0

    out = dict(
        col_sum=col_sum,
        col_sum_residual=col_resid,
        max_abs_col_sum_residual=float(np.max(np.abs(col_resid))),
        max_rel_col_sum_residual=float(np.max(np.abs(col_resid)
                                              / scale[:, None])),
        min_offdiagonal=float(offdiag.min()),
        max_diagonal=float(np.max(A[:, idx, idx])),
        # Ch2 Eq. uniform_column_condition: max_j sum_i a_ij(t) <= -eta < 0
        max_col_sum_over_years=float(np.max(col_sum)),
        eta_uniform=float(-np.max(col_sum)),
        argmax_col_sum_year=float(np.asarray(years).ravel()[
            np.unravel_index(np.argmax(col_sum), col_sum.shape)[0]]),
    )

    if fit is None:
        return out
    for tag, S6 in (("pred", S_ode_pred), ("obs", S_ode_obs)):
        lhs = np.einsum("tij,tj->ti", A, S6) + b
        ref = rhs_dSdt(fit, params, S6, years)
        den = max(float(np.max(np.abs(ref))), 1e-300)
        out[f"rhs_residual_{tag}"] = lhs - ref
        out[f"max_abs_rhs_residual_{tag}"] = float(np.max(np.abs(lhs - ref)))
        out[f"max_rel_rhs_residual_{tag}"] = float(np.max(np.abs(lhs - ref)) / den)
    return out


# ---------------------------------------------------------------------------
# the dump
# ---------------------------------------------------------------------------
def _coeffs(pr, where):
    """Pull the six coefficient arrays out of a `predictions()` dict."""
    sfx = "_at_obs" if where == "at_obs" else ""
    return dict(
        alphas=np.asarray(pr["alphas" + sfx], float),
        taus=np.asarray(pr["taus" + sfx], float),
        frac_fu=np.asarray(pr["frac_fu" + sfx], float),
        frac_eu=np.asarray(pr["frac_eu" + sfx], float),
        f_cohort=np.asarray(pr["f_cohort" + sfx], float),
        cp=np.asarray(pr["cp" + sfx], float),
    )


def _reclose_simplex(f, kind, new_data, loss_data):
    """Substitute a pinned simplex entry with its reported value and re-close
    the simplex the way `_simplex_value` does (v5:729).

    `_predict_paths_at`'s diagnostic override instead overwrites the pinned
    entry alone and leaves the other two untouched (v5:2548–2555).  That is
    correct for a per-component diagnostic but NOT for assembling A: the
    three entries then no longer sum to 1, and mass conservation fails by
    that residual.  Here the learned sigmoid split is recovered from the
    model's own output (`sigma = new / (new + out)`, exact because
    `new + out = avail`) and re-applied to the reported availability.
    """
    f = np.asarray(f, float)
    if kind == "loss_pinned":
        loss = np.clip(loss_data, 0.0, 1.0 - 1e-6)
        sigma = f[:, 0] / np.maximum(f[:, 0] + f[:, 2], 1e-300)
        avail = np.maximum(1.0 - loss, 1e-6)
        return np.stack([sigma * avail, loss, (1.0 - sigma) * avail], axis=1)
    if kind == "new_pinned":
        new = np.clip(new_data, 0.0, 1.0 - 1e-6)
        sigma = f[:, 1] / np.maximum(f[:, 1] + f[:, 2], 1e-300)
        avail = np.maximum(1.0 - new, 1e-6)
        return np.stack([new, sigma * avail, (1.0 - sigma) * avail], axis=1)
    if kind == "both_pinned":
        new = np.clip(new_data, 0.0, 1.0 - 1e-6)
        loss = np.clip(loss_data, 0.0, 1.0 - 1e-6)
        return np.stack([new, loss, np.maximum(1.0 - new - loss, 1e-6)], axis=1)
    return f.copy()                                  # "full": nothing pinned


def _reporting_coeffs(fit, pr):
    """The coefficient set A(t) is reported from: the raw MLP evaluation with
    every pinned component replaced by its exact reported value.

    Identical to what the integrator used at every node except `years[0]`,
    where `_interval_pick`'s `clip(1, N-1)` makes the ODE read year 1's
    pinned values (v5:638).
    """
    lay = fit.layout
    tau_obs = np.asarray(fit.data_all["tau_sup_obs"], float)
    c = _coeffs(pr, "at_pred")
    taus = c["taus"].copy()
    for j, pinned in enumerate(lay["pin_taus_tuple"]):
        if bool(pinned):
            taus[:, j] = tau_obs[:, j]
    c["taus"] = taus
    c["frac_fu"] = _reclose_simplex(c["frac_fu"], lay["fu_kind"],
                                    tau_obs[:, 4], tau_obs[:, 5])
    c["frac_eu"] = _reclose_simplex(c["frac_eu"], lay["eu_kind"],
                                    tau_obs[:, 6], tau_obs[:, 7])
    if bool(lay["pin_cp"]):
        c["cp"] = np.asarray(fit.data_all["cp_obs"], float).ravel()
    return c


def _learned_slot_masks(fit):
    """Per coefficient array, a boolean mask of the slots the MLP actually
    learns (the rest are pinned to interpolated data, `build_nn_layout`).

    Simplex semantics from `_simplex_value` (v5:729): with the loss entry
    pinned, `new` and `out` are still learned through the sigmoid split, so
    only index 1 is pinned; with `new` pinned, only index 0 is.
    """
    import zinc_colloc_v5 as v5

    lay = fit.layout
    m = {
        "alphas": np.ones(v5.N_ALPHAS, bool),          # always learned
        "f_cohort": np.ones(v5.N_COHORTS, bool),       # always learned
        "taus": ~np.asarray(lay["pin_taus_tuple"], bool),
    }
    for tag, kind in (("frac_fu", lay["fu_kind"]), ("frac_eu", lay["eu_kind"])):
        mask = np.ones(3, bool)
        if kind == "loss_pinned":
            mask[1] = False
        elif kind == "new_pinned":
            mask[0] = False
        elif kind == "both_pinned":
            mask[:] = False
        m[tag] = mask
    return m


def _flatten_params(params):
    """MLP weights as flat npz entries (`init_mlp` returns a list of
    {'W','b'} dicts, v5:445), so WP-2b onward can reload the fit instead of
    repeating it."""
    out = {}
    for i, layer in enumerate(params):
        for key in ("W", "b"):
            out[f"param{i}_{key}"] = np.asarray(layer[key], float)
    return out


def dump_A(fit, path, stage="B", rng_seed=0):
    """Assemble A(t) for one fit, verify it, and persist everything WP-2b–2e
    needs.  Returns the verification dict."""
    import zinc_colloc_v5 as v5

    pr = fit.predictions(stage)
    years = np.asarray(pr["years"], float).ravel()
    masks = fit._split_indices_in_all()

    c_ode = _coeffs(pr, "at_pred")        # exactly what the integrator evaluated
    c_rep = _reporting_coeffs(fit, pr)    # pinned -> exact reported data

    A, b = assemble_A(**c_rep)
    A_ode, b_ode = assemble_A(**c_ode)
    ell = loss_rates(c_rep["alphas"], c_rep["taus"],
                     c_rep["frac_fu"], c_rep["frac_eu"])
    ell_ode = loss_rates(c_ode["alphas"], c_ode["taus"],
                         c_ode["frac_fu"], c_ode["frac_eu"])

    # States.  The predicted 6-state is the integrator's own; the observed
    # one is lifted with a RANDOM cohort split, so the RHS identity has to
    # hold for a split the model never produced — a stronger test of the
    # three cohort rows than reusing the model's own shares.
    S_pred4 = np.asarray(pr["S_pred"], float)
    S_coh_pred = np.asarray(pr["S_cohorts"], float)
    S_ode_pred = np.concatenate(
        [S_pred4[:, :2], S_coh_pred, S_pred4[:, 3:4]], axis=1)
    S_obs4 = np.asarray(fit.data_all["stocks_obs"], float)
    rng = np.random.default_rng(rng_seed)
    w = rng.dirichlet(np.ones(v5.N_COHORTS), size=len(years))
    S_ode_obs = obs_to_ode_state(S_obs4, w)

    params = fit._params_for(stage)
    ver = verify(A_ode, b_ode, ell_ode, years, fit=fit, params=params,
                 S_ode_pred=S_ode_pred, S_ode_obs=S_ode_obs)
    ver_rep = verify(A, b, ell, years)
    ver["max_abs_col_sum_residual_reporting"] = ver_rep["max_abs_col_sum_residual"]
    # What assembling A straight from `predictions()["*_at_obs"]` would cost:
    # that override leaves the manufacturing simplices unclosed, so mass
    # conservation fails.  Quantified here rather than merely warned about.
    c_naive = _coeffs(pr, "at_obs")
    A_naive, b_naive = assemble_A(**c_naive)
    ver["max_abs_col_sum_residual_naive_at_obs"] = verify(
        A_naive, b_naive,
        loss_rates(c_naive["alphas"], c_naive["taus"],
                   c_naive["frac_fu"], c_naive["frac_eu"]),
        years)["max_abs_col_sum_residual"]
    ver["min_offdiagonal_reporting"] = ver_rep["min_offdiagonal"]
    ver["eta_uniform_reporting"] = ver_rep["eta_uniform"]
    ver["max_abs_A_minus_A_ode"] = float(np.max(np.abs(A - A_ode)))
    ver["max_abs_b_minus_b_ode"] = float(np.max(np.abs(b - b_ode)))
    # Nodes where the two variants genuinely disagree.  The tolerance keeps
    # out the ~1e-16 round-off of re-closing the simplex at nodes where the
    # pinned data and `_interval_pick` already agree.
    ver["A_minus_A_ode_years"] = [
        float(y) for y in years[np.any(np.abs(A - A_ode) > 1e-12, axis=(1, 2))]]

    # State-dependence of the coefficients: the same raw MLP evaluated at
    # observed vs at predicted stocks, over the LEARNED slots only (the
    # pinned ones carry the `at_obs` data override, which is not a state
    # effect).  With `use_stock_input=False` the MLP never sees the state
    # and this is exactly 0 — reported rather than assumed, because it is
    # the quantity WP-8a's "d(alpha)/dS block" will measure.
    learned = _learned_slot_masks(fit)
    ver["max_abs_coef_state_dependence"] = float(max(
        (np.max(np.abs(pr[k + "_at_obs"][..., m] - pr[k][..., m]))
         if np.any(m) else 0.0)
        for k, m in learned.items()))

    arrays = dict(
        years=years,
        A=A, b=b, A_ode=A_ode, b_ode=b_ode,
        loss_rate=ell, loss_rate_ode=ell_ode,
        col_sum=ver["col_sum"], col_sum_residual=ver["col_sum_residual"],
        rhs_residual_pred=ver["rhs_residual_pred"],
        rhs_residual_obs=ver["rhs_residual_obs"],
        S_ode_pred=S_ode_pred, S_obs=S_obs4, S_pred=S_pred4,
        cohort_split_obs_random=w,
        mu_cohorts=np.asarray(v5.MU_COHORTS_NP, float),
        ode_state_names=np.array(ODE_STATE_NAMES, dtype=object),
        stock_names=np.array(v5.STOCK_NAMES, dtype=object),
        alpha_names=np.array(v5.ALPHA_NAMES, dtype=object),
        tau_binary_names=np.array(v5.TAU_BINARY_NAMES, dtype=object),
        mask_train=np.asarray(masks["train"], bool),
        mask_val=np.asarray(masks["val"], bool),
        mask_test=np.asarray(masks["test"], bool),
        stage=np.asarray(stage, dtype=object),
    )
    for tag, c in (("", c_rep), ("_ode", c_ode)):
        for k, v in c.items():
            arrays[f"{k}{tag}"] = v
    arrays.update(_flatten_params(params))
    arrays["verify_json"] = np.asarray(json.dumps(
        {k: v for k, v in ver.items() if not isinstance(v, np.ndarray)}),
        dtype=object)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **arrays)
    return ver


def fit_seed(seed, out_dir, cfg=None, verbose=False):
    """Refit one anchor_v4 seed, assemble A(t), dump it.  Returns the path."""
    import jax
    import zinc_colloc_v5 as v5

    integrity_check()
    install(v5)
    cfg = dict(cfg or load_anchor_config())
    cfg["verbose"] = bool(verbose)
    path = os.path.join(out_dir, f"A_seed{seed}.npz")

    t0 = time.time()
    fit = v5.run(f"wp2a_s{seed}", **dict(cfg, seed=seed))
    ver = dump_A(fit, path)
    dt = time.time() - t0

    del fit
    jax.clear_caches()                       # CLAUDE.md rule 6
    print(f"[seed {seed}] {dt/60:.1f} min -> {path}  "
          f"colsum={ver['max_abs_col_sum_residual']:.2e}  "
          f"rhs={ver['max_abs_rhs_residual_pred']:.2e}  "
          f"eta={ver['eta_uniform']:.4f}", flush=True)
    return path


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------
def check(verbose=True):
    """`zinc_alpha_lab.check()` (drivers, input_dim, digests) plus the
    WP-2 state layout."""
    import zinc_colloc_v5 as v5

    info = alab.check(verbose=verbose)
    install(v5)
    cfg = load_anchor_config()
    info["ode_state_names"] = list(ODE_STATE_NAMES)
    info["mu_cohorts"] = [float(x) for x in v5.MU_COHORTS_NP]
    info["use_stock_input"] = bool(cfg.get("use_stock_input", True))
    info["pinned"] = {k: bool(v) for k, v in cfg.items() if k.startswith("pin_")}
    info["learn_cp"] = bool(cfg.get("learn_cp", True))
    info["patches"] = list(PATCHES)
    if verbose:
        print("zinc_A_lab --check  (WP-2a additions)")
        print("=" * 74)
        print(f"  augmented state      : {ODE_STATE_NAMES}")
        print(f"  MU_COHORTS (yr)      : {list(v5.MU_COHORTS_NP)}")
        print(f"  b(t)                 : [cp(t), 0, 0, 0, 0, 0]   "
              f"(cp {'pinned to data' if not info['learn_cp'] else 'learned'})")
        print(f"  pinned components    : "
              f"{sorted(k for k, v in info['pinned'].items() if v)}")
        print(f"  use_stock_input      : {info['use_stock_input']}"
              + ("   -> MLP is state-blind; A(t) does not depend on where it "
                 "is evaluated" if not info["use_stock_input"] else ""))
        print(f"  patches applied      : {PATCHES or 'none (WP-2a needs none)'}")
        print("=" * 74)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description="WP-2a: assemble A(t) per seed")
    ap.add_argument("--check", action="store_true",
                    help="print resolved drivers, input_dim and state layout")
    ap.add_argument("--seeds", default="", help="comma-separated seeds to refit")
    ap.add_argument("--out", default=OUT_DIR_DEFAULT)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    info = check(verbose=True)
    if args.check:
        return 0
    if not args.seeds:
        return 0

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "check.json"), "w") as fh:
        json.dump({k: v for k, v in info.items() if k != "digests"}, fh,
                  indent=2, default=str)

    cfg = load_anchor_config()
    for sd in [int(s) for s in args.seeds.split(",") if s.strip()]:
        path = os.path.join(args.out, f"A_seed{sd}.npz")
        if args.skip_existing and os.path.exists(path):
            print(f"[seed {sd}] exists, skipped", flush=True)
            continue
        try:
            fit_seed(sd, args.out, cfg=cfg, verbose=args.verbose)
        except Exception as e:                     # keep the sweep going
            print(f"[seed {sd}] FAILED: {type(e).__name__}: {e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
