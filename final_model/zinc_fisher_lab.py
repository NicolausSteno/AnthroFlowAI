#!/usr/bin/env python3
"""
zinc_fisher_lab.py — lab module for WP-4c: Fisher information and sloppiness
===========================================================================

WP-4c asks, of the fitted model rather than of a synthetic one: *is each α
channel practically identifiable at N = 40?*  The instrument is the
Gauss–Newton approximation to the curvature of the trained objective at the
optimum,

    J = dr/dtheta        (residual Jacobian)          FIM = J^T J / kappa^2

eigendecomposed on a log scale (Gutenkunst et al. 2007's "sloppiness"
spectrum), with per-coefficient marginal uncertainties from the Laplace
approximation and the stiff/sloppy combination structure read off the
eigenvectors.

Pattern
-------
Follows `zinc_alpha_lab.py` / `zinc_A_lab.py` / `zinc_cf_lab.py`
(CLAUDE.md rule 1).  `zinc_colloc_v5.py` is imported and never edited; its
MD5 and the MD5 of the `anchor_v4` data copy are pinned through
`zinc_alpha_lab`; `PATCHES` is empty because everything here is built by
*composing* the core's public factories (`make_stageB_loss`,
`make_integrator`, `nn_eval`, `_norm_inputs`) rather than by rebinding
anything inside it.  Weights are reloaded from the `zinc_A_lab` dumps at
`analysis/wp2a/A_seed*.npz` — no refits.

The residual vector
-------------------
`stageB_window_loss` (v5:1594) is a weighted sum of masked per-feature mean
squared errors.  Any such object is exactly a sum of squares, so it can be
written as ``L = ||r||^2`` for an explicit residual vector, and that vector
is what `jax.jacrev` differentiates.  Blocks, with the anchor_v4 weights
(w_alpha = w_tau = 1, w_S = 5, w_F = 0, w_cp inactive because `pin_cp`):

    alpha     (W, 4)   sqrt(w_alpha / (4 * cnt_k))       * dlog(alpha)/s_k
    tau       (W, 8)   sqrt(w_tau   / (3 * cnt_j))       * d(tau)/s_j      [3 learned slots]
    stock     (W, 4)   sqrt(w_S * W_k / (sum W * W))     * dS/S_target_std_k
    ---- penalty blocks, kept separate: these are the prior, not the data ----
    smooth_a  (W-1,4)  sqrt(w_sa / ((W-1) * 4))          * diff(log alpha)
    smooth_t  (W-1,8)  sqrt(w_st / ((W-1) * 8))          * diff(logit tau)
    raw       (W, 10)  sqrt(w_raw / (W * n_raw))         * raw

`residual_check` verifies ``||r_data||^2 + ||r_pen||^2`` against the core's
own `stageB_window_loss` to floating-point tolerance before any Jacobian is
computed.  A residual vector that does not reproduce the loss it claims to
linearise is a bug, and the check is what tells the two apart.

Noise scale
-----------
A weighted sum of squares is the negative log-likelihood of a Gaussian model
with sigma_i^2 = s_i^2 kappa^2 / omega_i, where omega_i is the weight the
loss attaches to residual i and s_i its scale denominator.  Then
NLL = L_data / (2 kappa^2), FIM = J^T J / kappa^2 and the Laplace covariance
is Sigma_theta = kappa^2 (J^T J + H_pen + lambda I)^-1 with
kappa^2 = L_data(theta_hat) / (n_res - p_eff).  kappa is reported, never
assumed to be 1.

Adjoint
-------
CLAUDE.md rule 4 says `DirectAdjoint` only.  The *published* fit does not use
it: `train_model` calls `make_integrator(integrator, nn_eval)` with no
`adjoint` argument (v5:2265), which leaves diffrax's default
`RecursiveCheckpointAdjoint` in place; only `zinc_baseline.py:1989` passes
`DirectAdjoint`.  Reported as a flag, not worked around.  Since WP-4c wants
the curvature of the objective *as it was optimised*, the fit's own
`integrate_aug` is used, and the Jacobian is cross-checked against a
`DirectAdjoint` integrator and against central differences.  Forward mode is
unavailable either way (`RecursiveCheckpointAdjoint` is a `custom_vjp`), so
`jacrev` is used — which is also the cheaper direction here: 304 residuals
against 2 410 parameters.

Structural zeros
----------------
`anchor_v4` sets `use_time_input = False` and `use_stock_input = False`, so
`_norm_inputs` (v5:477) emits an input vector whose first five entries — the
time feature and the four stock features — are identically zero.  The 160
first-layer weights multiplying them have exactly zero gradient and are
unidentifiable by construction, not by data poverty.  `structural_zero_mask`
returns them so they can be excluded from every rank and spectrum statement.

CLI
---
    python zinc_fisher_lab.py --check
    python zinc_fisher_lab.py --seeds 0,1,2 --out analysis/wp4c
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
import zinc_cf_lab as cflab

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR_DEFAULT = os.path.join(HERE, "analysis", "wp4c")
WEIGHTS_DIR_DEFAULT = os.path.join(HERE, "analysis", "wp2a")

# The 17 time-varying coefficients of `zinc_circ_lab.PARAM_NAMES[:17]`, in the
# same order, so WP-2f's `sigma` provider and WP-8h's export can consume this
# module's marginals without any reindexing.  The three `mu` are fixed by
# construction and carry no uncertainty, so they are not emitted here.
COEF_NAMES = [
    "alpha_cc", "alpha_refc", "alpha_win", "alpha_dr",
    "tau_ref", "tau_waelz", "tau_olds", "tau_diss",
    "frac_fu_new", "frac_fu_loss", "frac_fu_out",
    "frac_eu_new", "frac_eu_loss", "frac_eu_into",
    "f_cohort_10yr", "f_cohort_20yr", "f_cohort_44yr",
]
N_COEF = len(COEF_NAMES)
ALPHA_IDX = (0, 1, 2, 3)

# Ridge scan for the Laplace covariance.  lambda is expressed as a multiple of
# the largest data eigenvalue so the scan is scale-free; `LAMBDA_REF` is the
# reference used in the headline tables and is the AdamW weight decay of
# `anchor_v4` read as a Gaussian prior precision in the loss's own units.
LAMBDA_REL_SCAN = (1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0)
LAMBDA_REF = 1e-4                      # = cfg["weight_decay"]

# Singular values below this fraction of the largest count as numerically
# zero, i.e. outside the data-informed subspace.
RANK_TOL = 1e-10

PATCHES: list[str] = []          # WP-4c needs none — see module docstring


def install(v5mod=None):
    """Apply import-time patches to the core module.  None needed for WP-4c."""
    if v5mod is None:
        import zinc_colloc_v5 as v5mod          # noqa: F401
    return PATCHES


def _rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024.0 ** 2) if sys.platform == "darwin" else r / 1024.0


# ---------------------------------------------------------------------------
# residual vector
# ---------------------------------------------------------------------------
def stageB_weights(cfg):
    """The Stage B loss weights actually in force, defaults resolved."""
    import zinc_colloc_v5 as v5
    d = dict(v5.DEFAULT_CONFIG)
    d.update(cfg)
    return dict(w_alpha=float(d["stageB_w_alpha"]), w_tau=float(d["stageB_w_tau"]),
                w_cp=float(d["stageB_w_cp"]), w_S=float(d["stageB_w_S"]),
                w_F=float(d["stageB_w_F"]),
                w_smooth_alpha=float(d["stageB_w_smooth_alpha"]),
                w_smooth_tau=float(d["stageB_w_smooth_tau"]),
                w_cohort_prior=float(d["stageB_w_cohort_prior"]),
                w_raw_reg=float(d["stageB_w_raw_reg"]),
                stock_loss_kind=str(d["stock_loss_kind"]))


def make_residual_fns(fit, cfg, *, start_idx=0, window_len=None, data=None,
                      integrate=None):
    """Residual blocks for the Stage B objective, as flat JAX functions of a
    flat parameter vector.

    Returns ``(r_data, r_pen, meta)`` where `r_data(theta)` and `r_pen(theta)`
    are 1-D arrays with ``||r_data||^2 + ||r_pen||^2 == stageB_window_loss``.
    """
    import jax
    import jax.numpy as jnp
    import zinc_colloc_v5 as v5

    W = stageB_weights(cfg)
    d = fit.data_trainval if data is None else data
    if window_len is None:
        window_len = int(np.asarray(d["years"]).shape[0]) - int(start_idx)
    integ = fit.integrate_aug if integrate is None else integrate

    lay = fit.layout
    learned_tau = np.array(
        [not p for p in lay["pin_taus_tuple"]]
        + [not lay["pin_fu_new"], not lay["pin_fu_loss"]]
        + [not lay["pin_eu_new"], not lay["pin_eu_loss"]], dtype=bool)

    stats = d["stats"]
    sl = slice(int(start_idx), int(start_idx) + int(window_len))
    years_w = jnp.asarray(d["years"])[sl]
    S_obs_w = jnp.asarray(d["stocks_obs"])[sl]
    a_obs_w = jnp.asarray(d["alpha_obs"])[sl]
    t_obs_w = jnp.asarray(d["tau_sup_obs"])[sl]

    # Masks are static: they depend on the data, not on theta.  `alpha_p` is
    # strictly positive by construction (v5:773, an exp), so the model side of
    # the alpha mask can never turn off.
    mask_a = np.isfinite(np.asarray(a_obs_w))
    mask_t = np.isfinite(np.asarray(t_obs_w)) & learned_tau[None, :]
    cnt_a = np.maximum(mask_a.sum(0), 1)                        # (4,)
    cnt_t = np.maximum(mask_t.sum(0), 1)                        # (8,)
    stw = np.asarray(stats["stock_term_weights"], float)
    ia, ja = np.nonzero(mask_a)
    it, jt = np.nonzero(mask_t)

    # Per-element residual prefactors, sqrt of the coefficient each squared
    # term carries in the loss (see module docstring).
    ca = np.sqrt(W["w_alpha"] * 1.0 / (float(np.sum(np.ones(v5.N_ALPHAS)))
                                       * cnt_a[ja]))
    ct = np.sqrt(W["w_tau"] * 1.0 / (float(learned_tau.sum()) * cnt_t[jt]))
    cS = np.sqrt(W["w_S"] * stw / (stw.sum() * window_len))     # (4,)
    n_raw = int(lay["n_raw"])
    c_sa = np.sqrt(W["w_smooth_alpha"] / ((window_len - 1) * v5.N_ALPHAS))
    c_st = np.sqrt(W["w_smooth_tau"] / ((window_len - 1) * t_obs_w.shape[1]))
    c_rw = np.sqrt(W["w_raw_reg"] / (window_len * n_raw))

    flat0, unravel = jax.flatten_util.ravel_pytree(fit.params_A)

    def _paths(theta):
        p = unravel(theta)
        S_pred_w, _F_int, _S_coh = integ(p, d, S_obs_w[0], years_w)

        def one(t, S):
            ex = v5.exog_fn(t, d["exog_times"], d["exog_values"])
            raw = v5.mlp_apply(p, v5._norm_inputs(t, S, ex, stats))
            cp, al, ta, fc, _rn = fit.nn_eval_sup(p, t, S, ex, d)
            return al, ta, raw

        alpha_p, tau_p, raw_p = jax.vmap(one)(years_w, S_pred_w)
        return S_pred_w, alpha_p, tau_p, raw_p

    def r_data(theta):
        S_pred_w, alpha_p, tau_p, _raw = _paths(theta)
        d_a = (v5._safe_log(alpha_p) - v5._safe_log(a_obs_w)) / stats["alpha_log_std"]
        d_t = (tau_p - t_obs_w) / stats["tau_sup_std"]
        if W["stock_loss_kind"] == "log":
            d_S = (v5._safe_log(S_pred_w) - v5._safe_log(S_obs_w)) / stats["S_log_std"]
        else:
            d_S = (S_pred_w - S_obs_w) / stats["S_target_std"]
        return jnp.concatenate([
            jnp.asarray(ca) * d_a[ia, ja],
            jnp.asarray(ct) * d_t[it, jt],
            (jnp.asarray(cS)[None, :] * d_S).ravel(),
        ])

    def r_pen(theta):
        _S, alpha_p, tau_p, raw_p = _paths(theta)
        d1a = jnp.diff(v5._safe_log(alpha_p), axis=0)
        d1t = jnp.diff(v5._logit(tau_p), axis=0)
        return jnp.concatenate([(c_sa * d1a).ravel(),
                                (c_st * d1t).ravel(),
                                (c_rw * raw_p).ravel()])

    meta = dict(
        window=(int(start_idx), int(window_len)),
        n_alpha=int(mask_a.sum()), n_tau=int(mask_t.sum()),
        n_stock=int(window_len * 4),
        block_names=(["alpha"] * int(mask_a.sum())
                     + ["tau"] * int(mask_t.sum())
                     + ["stock"] * int(window_len * 4)),
        block_channel=(list(ja.astype(int))
                       + list(jt.astype(int) + v5.N_ALPHAS)
                       + list(np.tile(np.arange(4), window_len) + 12)),
        weights=W, n_params=int(flat0.size),
        learned_tau=learned_tau.tolist(),
        stock_loss_kind=W["stock_loss_kind"],
    )
    return r_data, r_pen, meta


def residual_check(fit, cfg, theta, *, start_idx=0, window_len=None, data=None):
    """``||r||^2`` against the core's own `stageB_window_loss`.  Exact answer 0."""
    import jax
    import jax.numpy as jnp
    import zinc_colloc_v5 as v5

    W = stageB_weights(cfg)
    d = fit.data_trainval if data is None else data
    if window_len is None:
        window_len = int(np.asarray(d["years"]).shape[0]) - int(start_idx)
    stageB_loss = v5.make_stageB_loss(fit.integrate_aug, fit.nn_eval_sup,
                                      fit.layout, fit.flow_obs_to_pred_idx,
                                      stock_loss_kind=W["stock_loss_kind"])
    kw = {k: v for k, v in W.items() if k != "stock_loss_kind"}
    flat0, unravel = jax.flatten_util.ravel_pytree(fit.params_A)
    core, _terms = stageB_loss(unravel(jnp.asarray(theta)), d,
                               int(start_idx), int(window_len), **kw)
    r_data, r_pen, _m = make_residual_fns(fit, cfg, start_idx=start_idx,
                                          window_len=window_len, data=d)
    mine = float(jnp.sum(r_data(theta) ** 2) + jnp.sum(r_pen(theta) ** 2))
    core = float(core)
    return dict(core=core, residual=mine, abs_err=abs(core - mine),
                rel_err=abs(core - mine) / max(abs(core), 1e-300))


# ---------------------------------------------------------------------------
# coefficient map  g(theta)
# ---------------------------------------------------------------------------
def make_coeff_fn(fit, data=None):
    """`g(theta) -> (T, 17)`, the reported coefficients in `COEF_NAMES` order.

    `nn_eval` already composes learned NN outputs with the pinned data
    look-ups and closes both manufacturing simplices internally
    (`_simplex_value`, v5:729), so this is the same object
    `zinc_A_lab._reporting_coeffs` assembles — verified in `check()` — except
    at `years[0]`, where `_interval_pick`'s `clip(1, N-1)` (v5:638) makes the
    pinned look-ups read year 1.  Pinned entries are data, not functions of
    theta, so their gradient is exactly zero and they appear in the output
    with zero marginal uncertainty.  That is a statement about *this*
    estimator, not about the underlying quantity being certain.
    """
    import jax
    import jax.numpy as jnp
    import zinc_colloc_v5 as v5

    d = fit.data_all if data is None else data
    years = jnp.asarray(d["years"]).ravel()
    S_obs = jnp.asarray(d["stocks_obs"])
    _flat0, unravel = jax.flatten_util.ravel_pytree(fit.params_A)

    def g(theta):
        p = unravel(theta)

        def one(t, S):
            ex = v5.exog_fn(t, d["exog_times"], d["exog_values"])
            o = fit.nn_eval(p, t, S, ex, d)
            return jnp.concatenate([o["alphas"], o["taus"], o["frac_fu"],
                                    o["frac_eu"], o["f_cohort"]])

        return jax.vmap(one)(years, S_obs)

    return g


def structural_zero_mask(fit):
    """First-layer weights that multiply an identically-zero input feature.

    With `use_time_input = False` and `use_stock_input = False` the first five
    entries of `_norm_inputs` are exactly 0 (v5:490, 505), so the
    corresponding columns of W0 cannot influence any output.  Returns a flat
    boolean mask over the parameter vector, True where the parameter is dead
    by construction.
    """
    import jax
    import jax.numpy as jnp

    stats = fit.data_all["stats"]
    dead_feat = np.zeros(1 + 4, bool)
    dead_feat[0] = not bool(np.asarray(stats["use_time_input"]))
    dead_feat[1:] = not bool(np.asarray(stats["use_stock_input"]))
    tree = jax.tree_util.tree_map(jnp.zeros_like, fit.params_A)
    tree = [dict(t) for t in tree]
    m0 = np.zeros_like(np.asarray(tree[0]["W"]), bool)
    m0[:, : len(dead_feat)] = dead_feat[None, :]
    tree[0] = dict(W=jnp.asarray(m0, float), b=tree[0]["b"])
    flat, _ = jax.flatten_util.ravel_pytree(tree)
    return np.asarray(flat) > 0.5


# ---------------------------------------------------------------------------
# Jacobians
# ---------------------------------------------------------------------------
def jacobians(fit, cfg, theta, *, start_idx=0, window_len=None, data=None,
              integrate=None):
    """`(J_data, J_pen, G, meta)` at `theta`, by reverse-mode AD.

    `G` is `dg/dtheta` of shape (T, 17, n_params) — the coefficient map, which
    needs no ODE and is therefore free relative to `J_data`.
    """
    import jax
    import jax.numpy as jnp

    r_data, r_pen, meta = make_residual_fns(fit, cfg, start_idx=start_idx,
                                            window_len=window_len, data=data,
                                            integrate=integrate)
    th = jnp.asarray(theta)
    J_data = np.asarray(jax.jit(jax.jacrev(r_data))(th))
    J_pen = np.asarray(jax.jit(jax.jacrev(r_pen))(th))
    g = make_coeff_fn(fit)
    G = np.asarray(jax.jit(jax.jacrev(g))(th))
    meta["r_data"] = np.asarray(r_data(th))
    meta["r_pen"] = np.asarray(r_pen(th))
    meta["g"] = np.asarray(g(th))
    return J_data, J_pen, G, meta


def make_tight_integrator(fit, rtol=1e-11, atol=1e-13, max_steps=200000):
    """A validation-only integrator at much tighter tolerance.

    `make_integrator` hard-codes `rtol = 1e-5, atol = 1e-7` (v5:1351) and the
    core is read-only, so the tight solve is *composed* from the core's public
    pieces (`make_rhs`, `_make_rhs_data`, `_make_Y0`, `ode_to_4obs`) exactly
    the way `zinc_A_lab.rhs_dSdt` composes them.  Nothing in the fitted model
    changes: this integrator is used only to give finite differences something
    smooth enough to differentiate.
    """
    import jax.numpy as jnp
    import diffrax as dfx
    import zinc_colloc_v5 as v5

    rhs = v5.make_rhs(fit.nn_eval)

    def integrate(params, data, S0, years):
        years = jnp.asarray(years)
        rhs_data = v5._make_rhs_data(data)

        def f(t, y, args):
            p, d = args
            return rhs(y, t, p, d)

        sol = dfx.diffeqsolve(
            dfx.ODETerm(f), dfx.Tsit5(), t0=years[0], t1=years[-1],
            dt0=jnp.asarray(0.01, dtype=years.dtype), y0=v5._make_Y0(S0),
            args=(params, rhs_data), saveat=dfx.SaveAt(ts=years),
            stepsize_controller=dfx.PIDController(rtol=rtol, atol=atol),
            max_steps=max_steps, adjoint=dfx.DirectAdjoint())
        Y = sol.ys
        S_ode = Y[:, :v5.N_ODE_STOCKS]
        C = Y[:, v5.N_ODE_STOCKS:]
        return (v5.ode_to_4obs(S_ode), C[1:] - C[:-1],
                S_ode[:, 2:2 + v5.N_COHORTS])

    return integrate


def jacobian_validation(fit, cfg, theta, *, n_probe=4, seed=0,
                        h_scan=(1e-2, 1e-3, 1e-4, 1e-5, 1e-6),
                        start_idx=0, window_len=None):
    """Cross-checks on `J_data`: a second adjoint, and central differences.

    Central differences are taken along random unit directions rather than
    over all 2 410 coordinates — the object being validated is a linear map,
    and a handful of random probes certifies it as well as the full matrix
    would, at a fraction of the cost.

    Two blocks are checked separately because they are numerically different
    problems.  The alpha/tau residuals are a pure MLP evaluation and finite
    differences reach machine precision on them.  The stock residuals come
    out of an *adaptive* solve at `rtol = 1e-5`, whose accepted step sequence
    is a discontinuous function of theta; the resulting O(1e-5) roughness
    divided by a step of 1e-6 swamps the derivative, so the check is run
    against `make_tight_integrator` and over a scan of `h`, and the
    loose-vs-tight residual gap is reported as the solver noise floor that
    sets the achievable accuracy.
    """
    import jax
    import jax.numpy as jnp
    import diffrax as dfx
    import zinc_colloc_v5 as v5

    th = jnp.asarray(theta)
    r_data, _rp, meta = make_residual_fns(fit, cfg, start_idx=start_idx,
                                          window_len=window_len)
    J = np.asarray(jax.jit(jax.jacrev(r_data))(th))

    direct = v5.make_integrator("diffrax", fit.nn_eval,
                                adjoint=dfx.DirectAdjoint())
    r_dir, _p2, _m2 = make_residual_fns(fit, cfg, start_idx=start_idx,
                                        window_len=window_len,
                                        integrate=direct)
    J_dir = np.asarray(jax.jit(jax.jacrev(r_dir))(th))

    tight = make_tight_integrator(fit)
    r_tight, _p3, _m3 = make_residual_fns(fit, cfg, start_idx=start_idx,
                                          window_len=window_len,
                                          integrate=tight)
    J_tight = np.asarray(jax.jit(jax.jacrev(r_tight))(th))

    n_ode = meta["n_stock"]
    ana = slice(0, len(meta["block_names"]) - n_ode)      # alpha + tau
    ode = slice(len(meta["block_names"]) - n_ode, None)   # stock

    rng = np.random.default_rng(seed)
    f_tight = jax.jit(r_tight)
    best = {"analytic": np.inf, "ode": np.inf}
    best_h = {"analytic": np.nan, "ode": np.nan}
    for h in h_scan:
        ea, eo = [], []
        for _ in range(n_probe):
            v = rng.normal(size=th.size)
            v /= np.linalg.norm(v)
            fd = (np.asarray(f_tight(th + h * v))
                  - np.asarray(f_tight(th - h * v))) / (2 * h)
            ad = J_tight @ v
            for key, sl, acc in (("analytic", ana, ea), ("ode", ode, eo)):
                den = max(np.max(np.abs(ad[sl])), 1e-300)
                acc.append(np.max(np.abs(fd[sl] - ad[sl])) / den)
        for key, acc in (("analytic", ea), ("ode", eo)):
            m = float(np.max(acc))
            if m < best[key]:
                best[key], best_h[key] = m, float(h)

    r_loose = np.asarray(r_data(th))
    r_tt = np.asarray(r_tight(th))
    return dict(
        max_abs_J_minus_J_directadjoint=float(np.max(np.abs(J - J_dir))),
        max_abs_J=float(np.max(np.abs(J))),
        rel_J_vs_directadjoint=float(np.max(np.abs(J - J_dir))
                                     / max(np.max(np.abs(J)), 1e-300)),
        rel_J_vs_tight_solver=float(np.max(np.abs(J - J_tight))
                                    / max(np.max(np.abs(J)), 1e-300)),
        solver_noise_floor_resid=float(np.max(np.abs(r_loose - r_tt))),
        max_rel_err_central_difference_analytic=best["analytic"],
        max_rel_err_central_difference_ode=best["ode"],
        best_h_analytic=best_h["analytic"], best_h_ode=best_h["ode"],
        n_probe=int(n_probe), h_scan=list(h_scan))


# ---------------------------------------------------------------------------
# spectrum, Laplace marginals, stiff/sloppy structure
# ---------------------------------------------------------------------------
def spectrum(J_data, J_pen=None, dead=None):
    """SVD of the residual Jacobian(s).

    `J^T J` has rank at most the number of residuals (304 here against 2 410
    parameters), so its spectrum is obtained from the SVD of `J` at a
    thousandth of the cost of forming and eigendecomposing the 2 410 x 2 410
    matrix — and with no loss: every remaining eigenvalue is exactly zero.
    """
    J = np.asarray(J_data, float)
    if dead is not None:
        J = J[:, ~dead]
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    out = dict(sv=s, eig=s ** 2, V=Vt.T, n_res=J.shape[0], n_par=J.shape[1])
    out["rank"] = int(np.sum(s > s[0] * RANK_TOL)) if s.size else 0
    if J_pen is not None:
        Jp = np.asarray(J_pen, float)
        if dead is not None:
            Jp = Jp[:, ~dead]
        sp = np.linalg.svd(Jp, compute_uv=False)
        out["sv_pen"] = sp
        Jb = np.vstack([J, Jp])
        out["sv_joint"] = np.linalg.svd(Jb, compute_uv=False)
    return out


def laplace_marginals(sp, G, kappa2, *, lambdas, cuts=None, dead=None):
    """Marginal standard deviations of every coefficient under the Laplace
    approximation, plus the identifiability decomposition.

    For a target `g` with gradient `q = dg/dtheta`,

        var_lambda(g) = kappa^2 * q^T (J^T J + lambda I)^-1 q
                      = kappa^2 / lambda * ( ||q||^2
                          - sum_i s_i^2/(s_i^2 + lambda) (v_i^T q)^2 )

    evaluated from the SVD without forming any 2 410 x 2 410 matrix.  Two
    further families are returned, because the ridge-dependent number on its
    own is not a scientific statement:

    `sd_by_cut`   the Cramer–Rao bound restricted to the directions whose
                  eigenvalue exceeds a stated fraction of the largest,
                  `kappa * sqrt( sum_{s_i^2 > cut * s_max^2} (v_i^T q)^2 / s_i^2 )`.
                  This is a *lower* bound on the marginal standard deviation
                  that uses only the retained directions, and it is reported
                  over a grid of cuts because a sloppy spectrum makes it
                  dominated by whichever direction is retained last — that
                  dependence is the finding, not an artefact to be hidden.
    `frac_null_by_cut`  the share of `||q||^2` orthogonal to the retained
                  subspace.  If it is appreciable the coefficient is not
                  determined by the data at that resolution and its posterior
                  width is set by the prior, which is the operational meaning
                  of "practically unidentifiable".

    Coefficients pinned to data have `q = 0` exactly: they are not estimated,
    so every quantity here is zero and `frac_null` is NaN rather than 1.
    """
    s = sp["sv"]
    V = sp["V"]
    cuts = (1e-10, 1e-8, 1e-6, 1e-4, 1e-2) if cuts is None else cuts
    q = np.asarray(G, float)
    shp = q.shape[:-1]
    q = q.reshape(-1, q.shape[-1])
    if dead is not None:
        q = q[:, ~dead]
    proj = q @ V                                          # (n_g, r)
    q2 = np.einsum("ij,ij->i", q, q)
    p2 = proj ** 2
    live = q2 > 0.0
    smax2 = float(s[0] ** 2) if s.size else 0.0

    out = {}
    for lam in lambdas:
        w = s ** 2 / (s ** 2 + lam)
        var = (q2 - p2 @ w) / lam
        out[float(lam)] = np.sqrt(np.maximum(var, 0.0) * kappa2).reshape(shp)

    sd_cut, null_cut = {}, {}
    for c in cuts:
        keep = s ** 2 > c * smax2
        sd_cut[float(c)] = np.sqrt(
            np.maximum((p2[:, keep] / s[keep] ** 2).sum(1), 0.0)
            * kappa2).reshape(shp)
        fn = np.full(q2.shape, np.nan)
        fn[live] = 1.0 - p2[live][:, keep].sum(1) / q2[live]
        null_cut[float(c)] = fn.reshape(shp)

    return dict(sd_by_lambda=out, sd_by_cut=sd_cut,
                frac_null_by_cut=null_cut,
                grad_norm=np.sqrt(q2).reshape(shp),
                estimated=live.reshape(shp))


def coeff_covariance(sp, G, kappa2, *, lambdas, dead=None):
    """Per-year covariance of the 17 coefficients, `(n_lam, T, 17, 17)`.

    Marginal standard deviations answer "how well is this coefficient known";
    the covariance answers "how well is any FUNCTION of them known", which is
    what WP-8h needs to put a band on an assembled `A(t)` entry and what
    Gandolfo's own stability test needs to put a standard error on a
    characteristic root.  Both are delta-method pushforwards through
    `zinc_circ_lab.dA_dparams`, and both are wrong if the coefficients are
    treated as independent — `frac_fu_new` and `frac_fu_out` are perfectly
    anti-correlated by the simplex, to take the clearest case.

        Cov(g_t) = kappa^2 Q_t (J^T J + lam I)^-1 Q_t^T
                 = kappa^2/lam ( Q_t Q_t^T - (Q_t V) diag(w) (Q_t V)^T )

    with `w_i = s_i^2/(s_i^2 + lam)`.
    """
    s, V = sp["sv"], sp["V"]
    G = np.asarray(G, float)
    if dead is not None:
        G = G[..., ~dead]
    T = G.shape[0]
    QQ = np.einsum("tap,tbp->tab", G, G)
    QV = np.einsum("tap,pi->tai", G, V)
    out = np.empty((len(lambdas), T, G.shape[1], G.shape[1]))
    for a, lam in enumerate(lambdas):
        w = s ** 2 / (s ** 2 + lam)
        out[a] = kappa2 / lam * (QQ - np.einsum("tai,i,tbi->tab", QV, w, QV))
    return out


def kappa_squared(r_data, n_res, p_eff):
    """Noise scale: `||r_data||^2 / (n_res - p_eff)`, floored at 1 dof."""
    rss = float(np.sum(np.asarray(r_data) ** 2))
    dof = max(float(n_res) - float(p_eff), 1.0)
    return rss / dof, rss, dof


def p_effective(sp, lam):
    """`tr[(J^T J + lambda I)^-1 J^T J]` — the effective number of parameters
    the data constrains at ridge `lam` (Hastie–Tibshirani–Friedman §3.4.4)."""
    s = np.asarray(sp["sv"], float)
    return float(np.sum(s ** 2 / (s ** 2 + lam)))


def stiff_directions(sp, G, n_modes=6, dead=None):
    """What the stiffest and sloppiest data-informed directions do to the
    coefficients.

    Returns `(n_modes, T, 17)` coefficient responses to a unit step along the
    leading eigenvectors and along the last retained ones, which is the
    "stiff combinations even where individual coefficients are loose" the
    spec asks for: a direction can be tightly determined while every
    coefficient it moves is individually loose.
    """
    V, s = sp["V"], sp["sv"]
    r = sp["rank"]
    G = np.asarray(G, float)
    if dead is not None:
        G = G[..., ~dead]
    idx_stiff = np.arange(min(n_modes, r))
    idx_sloppy = np.arange(max(r - n_modes, 0), r)
    resp = lambda idx: np.einsum("tcp,pk->ktc", G, V[:, idx])
    return dict(stiff=resp(idx_stiff), sloppy=resp(idx_sloppy),
                sv_stiff=s[idx_stiff], sv_sloppy=s[idx_sloppy],
                idx_stiff=idx_stiff, idx_sloppy=idx_sloppy)


def sample_coef_paths(fit, cfg, theta, *, n_draw=64, lam_rel=None, rng_seed=0,
                      start_idx=0, window_len=None, integrate=None):
    """Draws of the whole coefficient PATH from one seed's Laplace posterior.

    The marginals in `laplace_marginals` are per year and carry no information
    about how the error at 1990 relates to the error at 2019.  Anything that
    integrates over the coefficient path — Chapter 2's non-autonomous lifetime
    and use-count, Ch2 Eqs. `nonautonomous_lifetime` and
    `nonautonomous_use_count` — needs that cross-year structure, and assuming
    it away in either direction gives the wrong answer: perfect correlation
    over-counts, independence under-counts.

    There is no need to assume anything.  The coefficient path is a
    deterministic function of the weights, so a single draw
    `dtheta ~ N(0, kappa^2 (J^T J + lambda I)^-1)` induces a coherent
    perturbation of every year at once, `dg(t) = G(t) dtheta`, with exactly
    the cross-year covariance the posterior implies.  Sampling in theta and
    pushing forward is therefore both simpler and correct.

        Sigma^(1/2) = kappa [ V diag((s^2+lam)^(-1/2)) V^T
                              + lam^(-1/2) (I - V V^T) ]

    using the same SVD the marginals come from, so no 2 250 x 2 250 matrix is
    ever formed.  Returns `(n_draw, T, 17)` perturbations and a diagnostics
    dict; `check_marginals` in the dict is the maximum relative gap between
    the sample standard deviation of the draws and the analytic marginal,
    which is O(1/sqrt(n_draw)) when the sampler is right.
    """
    lam_rel = LAMBDA_REF if lam_rel is None else lam_rel
    J, Jp, G, meta = jacobians(fit, cfg, theta, start_idx=start_idx,
                               window_len=window_len, integrate=integrate)
    dead = structural_zero_mask(fit)
    sp = spectrum(J, Jp, dead=dead)
    lam = float(lam_rel) * float(sp["eig"][0])
    p_eff = p_effective(sp, lam)
    kap2, rss, dof = kappa_squared(meta["r_data"], sp["n_res"], p_eff)

    s_, V = sp["sv"], sp["V"]
    n_live = V.shape[0]
    rng = np.random.default_rng(rng_seed)
    xi = rng.standard_normal((int(n_draw), n_live))
    Vx = xi @ V                                              # (K, r)
    dth = ((Vx / np.sqrt(s_ ** 2 + lam)) @ V.T
           + (xi - Vx @ V.T) / np.sqrt(lam)) * np.sqrt(kap2)
    Gl = np.asarray(G, float)[..., ~dead]                    # (T, 17, n_live)
    dg = np.einsum("tcp,kp->ktc", Gl, dth)                   # (K, T, 17)

    marg = laplace_marginals(sp, G, kap2, lambdas=[lam], dead=dead)
    sd_ref = marg["sd_by_lambda"][float(lam)]
    sd_mc = dg.std(0, ddof=1)
    ok = sd_ref > 0
    rel = np.max(np.abs(sd_mc[ok] - sd_ref[ok]) / sd_ref[ok]) if ok.any() else 0.0
    info = dict(kappa2=kap2, lam=lam, lam_rel=float(lam_rel),
                p_eff=p_eff, n_draw=int(n_draw), rank=sp["rank"],
                check_marginals=float(rel), coef=meta["g"])
    del J, Jp, G
    return dg, info


# ---------------------------------------------------------------------------
# profile likelihood
# ---------------------------------------------------------------------------
def profile_alpha(fit, cfg, theta0, channel, offsets, *, steps=200, lr=1e-3,
                  penalty=1e4, start_idx=0, window_len=None, mask=None,
                  verbose=False):
    """Profile likelihood of the *level* of one alpha channel.

    The quadratic approximation the Laplace marginal rests on is exactly what
    a profile is meant to avoid, so the constraint is imposed on a scalar
    functional of the trajectory rather than on the 2 410 weights: for each
    offset `delta`, minimise the Stage B loss subject to

        mean_t [ log alpha_k(t) ] = c_hat_k + delta        (t in the fit window)

    implemented as an exterior quadratic penalty and re-optimised with Adam
    from the previous grid point (continuation), so each point starts inside
    the basin of the last.  Returns the attained data loss and the realised
    constraint violation at every offset; a point whose violation has not
    collapsed is reported, not silently kept.
    """
    import jax
    import jax.numpy as jnp
    import optax

    r_data, r_pen, meta = make_residual_fns(fit, cfg, start_idx=start_idx,
                                            window_len=window_len)
    g = make_coeff_fn(fit)
    _flat0, unravel = jax.flatten_util.ravel_pytree(fit.params_A)
    m = (np.ones(np.asarray(fit.data_all["years"]).size, bool)
         if mask is None else np.asarray(mask, bool))
    mj = jnp.asarray(m.astype(float))
    k = int(channel)

    def level(theta):
        lg = jnp.log(jnp.maximum(g(theta)[:, k], 1e-12))
        return jnp.sum(mj * lg) / jnp.sum(mj)

    c_hat = float(level(jnp.asarray(theta0)))

    def loss_at(theta, target):
        ld = jnp.sum(r_data(theta) ** 2)
        lp = jnp.sum(r_pen(theta) ** 2)
        v = level(theta) - target
        return ld + lp + penalty * v ** 2, (ld, lp, v)

    grad_fn = jax.jit(jax.value_and_grad(loss_at, has_aux=True))
    rows = []
    up = sorted([float(o) for o in offsets if o > 0])
    down = sorted([float(o) for o in offsets if o < 0], reverse=True)
    # The delta = 0 point is optimised under the SAME budget as every other
    # point.  It has to be: `theta0` is the published Stage B optimum of a
    # WINDOW-CURRICULUM objective (batch 4, W = 22, random starts), not a
    # stationary point of the full-trainval objective profiled here, so a few
    # unconstrained steps lower the loss whatever the constraint is.  Taking
    # the raw value as the reference makes every Phi(delta) - Phi(0) negative
    # and the profile meaningless — measured, not hypothesised: see
    # `loss_data_unoptimised` in the returned rows.
    for branch in ([0.0], up, down):
        th = jnp.asarray(theta0)                    # continuation from the fit
        for delta in branch:
            target = c_hat + delta
            opt = optax.adam(lr)
            st = opt.init(th)
            for _ in range(int(steps)):
                (_L, _aux), gr = grad_fn(th, target)
                upd, st = opt.update(gr, st, th)
                th = optax.apply_updates(th, upd)
            (_L, (ld, lp, v)), _gr = grad_fn(th, target)
            rows.append(dict(channel=k, offset=delta, target=float(target),
                             loss_data=float(ld), loss_pen=float(lp),
                             violation=float(v), level=float(level(th))))
            if verbose:
                print(f"    delta={delta:+.3f}  L={float(ld)+float(lp):.6f}  "
                      f"viol={float(v):+.2e}", flush=True)
    (_L, (ld0, lp0, v0)), _ = grad_fn(jnp.asarray(theta0), c_hat)
    raw = float(ld0) + float(lp0)
    for r in rows:
        r["loss_unoptimised_at_theta0"] = raw
    return sorted(rows, key=lambda r: r["offset"]), c_hat


# ---------------------------------------------------------------------------
# per-seed driver
# ---------------------------------------------------------------------------
def run_seed(seed, fit, cfg, out_dir, *, weights_dir=WEIGHTS_DIR_DEFAULT,
             lambdas=None, validate=False):
    """Everything WP-4c needs for one seed, dumped to `wp4c_seed{N}.npz`."""
    import jax
    import jax.numpy as jnp

    lambdas = LAMBDA_REL_SCAN if lambdas is None else lambdas
    params = cflab.load_params(os.path.join(weights_dir, f"A_seed{seed}.npz"))
    theta, _unravel = jax.flatten_util.ravel_pytree(params)
    theta = jnp.asarray(theta)

    chk = residual_check(fit, cfg, theta)
    J, Jp, G, meta = jacobians(fit, cfg, theta)
    dead = structural_zero_mask(fit)
    sp = spectrum(J, Jp, dead=dead)

    # kappa is solved for jointly with p_eff at the reference ridge: p_eff
    # depends on the spectrum only, so one pass suffices.
    lam_abs = {float(rel): float(rel) * float(sp["eig"][0]) for rel in lambdas}
    lam_ref = LAMBDA_REF * float(sp["eig"][0])
    p_eff = p_effective(sp, lam_ref)
    kap2, rss, dof = kappa_squared(meta["r_data"], sp["n_res"], p_eff)

    cuts = (1e-10, 1e-8, 1e-6, 1e-4, 1e-2)
    marg = laplace_marginals(sp, G, kap2, lambdas=list(lam_abs.values()),
                             cuts=cuts, dead=dead)
    stiff = stiff_directions(sp, G, dead=dead)
    cov = coeff_covariance(sp, G, kap2, lambdas=list(lam_abs.values()),
                           dead=dead)                     # (n_lam, T, 17, 17)

    arrays = dict(
        seed=np.asarray(int(seed)),
        years=np.asarray(fit.data_all["years"], float).ravel(),
        coef_names=np.array(COEF_NAMES, dtype=object),
        coef=meta["g"],
        sv=sp["sv"], eig=sp["eig"], sv_pen=sp["sv_pen"], sv_joint=sp["sv_joint"],
        rank=np.asarray(sp["rank"]), n_res=np.asarray(sp["n_res"]),
        n_par=np.asarray(sp["n_par"]), n_dead=np.asarray(int(dead.sum())),
        r_data=meta["r_data"], r_pen=meta["r_pen"],
        block_names=np.array(meta["block_names"], dtype=object),
        block_channel=np.asarray(meta["block_channel"], int),
        kappa2=np.asarray(kap2), rss=np.asarray(rss), dof=np.asarray(dof),
        p_eff_ref=np.asarray(p_eff),
        lam_rel=np.asarray(list(lam_abs.keys()), float),
        lam_abs=np.asarray(list(lam_abs.values()), float),
        sd_by_lambda=np.stack([marg["sd_by_lambda"][v]
                               for v in lam_abs.values()]),      # (n_lam,T,17)
        cut_rel=np.asarray(cuts, float),
        sd_by_cut=np.stack([marg["sd_by_cut"][c] for c in cuts]),
        frac_null_by_cut=np.stack([marg["frac_null_by_cut"][c] for c in cuts]),
        grad_norm=marg["grad_norm"], estimated=marg["estimated"],
        coef_cov=cov,
        stiff_response=stiff["stiff"], sloppy_response=stiff["sloppy"],
        sv_stiff=stiff["sv_stiff"], sv_sloppy=stiff["sv_sloppy"],
        residual_check=np.asarray(json.dumps(chk), dtype=object),
    )
    if validate:
        arrays["validation_json"] = np.asarray(
            json.dumps(jacobian_validation(fit, cfg, theta)), dtype=object)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"wp4c_seed{seed}.npz")
    np.savez_compressed(path, **arrays)
    del J, Jp, G
    jax.clear_caches()                                    # CLAUDE.md rule 6
    return path, chk, sp, kap2


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------
def check(verbose=True, weights_dir=WEIGHTS_DIR_DEFAULT):
    """`zinc_alpha_lab.check()` plus the WP-4c-specific layout facts."""
    import jax
    import jax.numpy as jnp
    import zinc_colloc_v5 as v5

    info = alab.check(verbose=verbose)
    cfg = load_anchor_config()
    fit = cflab.init_fit(cfg=cfg, seed=0)
    theta, _ = jax.flatten_util.ravel_pytree(
        cflab.load_params(os.path.join(weights_dir, "A_seed0.npz")))
    theta = jnp.asarray(theta)
    W = stageB_weights(cfg)
    chk = residual_check(fit, cfg, theta)
    _rd, _rp, meta = make_residual_fns(fit, cfg)
    dead = structural_zero_mask(fit)

    # The coefficient map against the WP-2a dump, which is the object every
    # Chapter 2 package already consumes.  `zinc_A_lab.dump_A` stores both the
    # reporting assembly (pinned entries replaced by the exact year-node data)
    # and the ODE assembly (`*_ode`, what the integrator evaluated); this must
    # equal the second exactly and the first at years[1:] — the year-0 gap is
    # `_interval_pick`'s clip (v5:638) and touches pinned entries only.
    gv = np.asarray(make_coeff_fn(fit)(theta))
    cmp_path = os.path.join(weights_dir, "A_seed0.npz")
    coef_chk = {}
    if os.path.exists(cmp_path):
        dd = np.load(cmp_path, allow_pickle=True)
        cat = lambda sfx: np.concatenate(
            [dd["alphas" + sfx], dd["taus" + sfx], dd["frac_fu" + sfx],
             dd["frac_eu" + sfx], dd["f_cohort" + sfx]], axis=1)
        coef_chk = dict(
            max_abs_vs_ode_assembly=float(np.max(np.abs(gv - cat("_ode")))),
            max_abs_vs_reporting_from_year1=float(
                np.max(np.abs(gv[1:] - cat("")[1:]))))
    info.update(n_params=int(theta.size), n_dead=int(dead.sum()),
                n_res=len(meta["block_names"]), window=meta["window"],
                weights=W, residual_check=chk, coeff_check=coef_chk)

    if verbose:
        print("zinc_fisher_lab --check  (WP-4c additions)")
        print("=" * 74)
        print(f"  weights dir          : {weights_dir}")
        print(f"  parameters           : {theta.size}  "
              f"({int(dead.sum())} dead by construction — "
              f"use_time_input={bool(np.asarray(fit.data_all['stats']['use_time_input']))}, "
              f"use_stock_input={bool(np.asarray(fit.data_all['stats']['use_stock_input']))})")
        print(f"  Stage B window       : start={meta['window'][0]} "
              f"len={meta['window'][1]}  on data_trainval")
        print(f"  residual blocks      : alpha {meta['n_alpha']} + "
              f"tau {meta['n_tau']} + stock {meta['n_stock']} "
              f"= {len(meta['block_names'])} data residuals")
        print(f"  loss weights         : " +
              ", ".join(f"{k}={v}" for k, v in W.items()))
        print(f"  ||r||^2 vs stageB_window_loss : {chk['residual']:.12f} vs "
              f"{chk['core']:.12f}   rel {chk['rel_err']:.2e}")
        print(f"  coefficients emitted : {N_COEF}  {COEF_NAMES}")
        if coef_chk:
            print(f"  g(theta) vs wp2a A_ode assembly        : "
                  f"{coef_chk['max_abs_vs_ode_assembly']:.2e}  (exact answer 0)")
            print(f"  g(theta) vs wp2a reporting, years[1:]  : "
                  f"{coef_chk['max_abs_vs_reporting_from_year1']:.2e}")
        print(f"  adjoint              : fit.integrate_aug = diffrax default "
              f"(RecursiveCheckpointAdjoint, v5:2265)")
        print( "                         CLAUDE.md rule 4 asks for "
               "DirectAdjoint; the PUBLISHED fit does not use it.")
        print( "                         WP-4c differentiates the objective as "
               "it was optimised and cross-checks")
        print( "                         against DirectAdjoint and central "
               "differences (jacobian_validation).")
        print(f"  ridge scan (relative): {LAMBDA_REL_SCAN}, reference "
              f"{LAMBDA_REF} (= weight_decay)")
        print("=" * 74)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--all-seeds", action="store_true")
    ap.add_argument("--out", default=OUT_DIR_DEFAULT)
    ap.add_argument("--weights", default=WEIGHTS_DIR_DEFAULT)
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args(argv)

    if args.check:
        check(verbose=True, weights_dir=args.weights)
        return 0

    cfg = load_anchor_config()
    fit = cflab.init_fit(cfg=cfg, seed=0)
    seeds = (list(range(35)) if args.all_seeds
             else [int(s) for s in args.seeds.split(",") if s.strip()])
    for i, s in enumerate(seeds):
        t0 = time.time()
        path, chk, sp, kap2 = run_seed(
            s, fit, cfg, args.out, weights_dir=args.weights,
            validate=(args.validate and i == 0))
        print(f"[seed {s}] {time.time()-t0:.1f}s  rank={sp['rank']}/"
              f"{sp['n_res']}  kappa^2={kap2:.4e}  "
              f"resid_rel={chk['rel_err']:.1e}  rss={_rss_mb():.0f}MB  "
              f"-> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
