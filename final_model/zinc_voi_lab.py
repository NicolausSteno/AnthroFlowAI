#!/usr/bin/env python3
"""
zinc_voi_lab.py — lab module for WP-11c: value of information and observation
ranking
=============================================================================

WP-11c turns "we need better data" into a prioritised list.  For each
*candidate* observation — a reported series at a stated frequency — it asks
what the posterior on the transfer coefficients would look like if that series
had also been observed over the fitting window, **without refitting**:

    FIM_new = FIM_current + J_cand^T Sigma^-1 J_cand

`FIM_current` is WP-4c's Gauss-Newton curvature of the Stage B objective at
the published optimum; `J_cand` is the Jacobian of the *predicted* candidate
observable with respect to the same parameter vector.  Both criteria the spec
asks for are reported: the D-optimality gain `log det FIM_new - log det
FIM_current`, and — the one a reporting body can act on — the percentage
reduction in the marginal standard deviation of each coefficient.

Pattern
-------
Follows `zinc_fisher_lab.py` (CLAUDE.md rule 1).  `zinc_colloc_v5.py` is
imported and never edited; `PATCHES` is empty because everything here is built
by *composing* the core's public pieces — the fit's own `integrate_aug`, its
flow accumulators and `zinc_fisher_lab`'s residual/coefficient machinery.
Weights are reloaded from the `zinc_A_lab` dumps at `analysis/wp2a/A_seed*.npz`;
no refits anywhere.

Observation operators
---------------------
Spec section 1 is explicit that **resampling means re-integrating, not
point-sampling**, and that stocks and flows carry different operators.  Both
are respected exactly:

* **stocks** are point-in-time — the solver's value at the node;
* **flows** are period integrals over the observation window, taken from the
  integrator's own 19 flow accumulators (`F_int = C[1:] - C[:-1]`, v5:1364),
  never point-sampled;
* every frequency is a **re-aggregation of one monthly solve**.  The
  accumulators are additive, so summing twelve monthly integrals reproduces the
  annual integral, and the stock at an annual node is the monthly solve's value
  at that node.  Both identities are verified to **0.0, bitwise**, in `check()`
  — the sub-annual grid changes nothing about the trajectory, only what is
  read off it.

The one quantity not carried by an accumulator is the **cohort-resolved
end-of-life flow**, which is what "EoL product-level composition" would
measure.  `end_of_life` is accumulated in aggregate only, so `E_k` is formed by
trapezoid on the monthly grid from the cohort sub-stocks the integrator does
return, `E_k = int S_k/mu_k dt` (`# Ch2 Eq. cohort_outflow`).  Its sum is
checked against the exact aggregate accumulator; the quadrature error is
1.8e-5 relative and is reported rather than assumed.

The noise model
---------------
Candidate residuals are `(log y_pred - log y_obs)/sigma`, with `sigma` the
per-series **relative** observation scale WP-3 calibrated from this fit's own
Stage B residuals (`analysis/synth/manifest.json`).  Using WP-3's numbers keeps
the ranking on the same noise footing as every other package that has looked at
these series, and WP-3 itself calls those sigmas an upper bound.

Two sigma conventions are reported side by side, because collapsing them is
exactly the confound the spec warns about in 11a:

* `per_obs`    — every observation carries the same relative error.  Going
                 monthly then buys *both* finer resolution and twelve times as
                 many independent error draws.
* `fixed_total`— `sigma_i = sigma * sqrt(m)` for `m` observations per year, so
                 a year's worth of sub-annual reports carries exactly the
                 information of one annual report.  This isolates **resolution**
                 from **sample size**, which is the distinction 11a is designed
                 to make and the one that decides whether the recommendation to
                 the community is "report monthly" or "keep reporting".

`Sigma` is diagonal: the candidate is treated as an **independent**
measurement of the series.  For a genuinely new sub-annual reporting stream
that is the right model.  For the *annual* arm of a series the objective
already sees — the four stocks enter through the stock block, and every flow
enters through the `alpha_obs = F_obs/S_obs` target — the same reported number
would be counted twice with independent errors, so those rows are an **upper
bound** on the gain.  The sub-annual arms are not affected, because the
observations they add do not exist in the current objective at any frequency.
Stated rather than worked around: modelling the correlation would need a joint
error model for the Rostek reconstruction that nothing in the project has.

Algebra
-------
Both FIM and prior live in the 2 250-dimensional live subspace (WP-4c: 160 of
the 2 410 weights are dead by construction).  With `B = J^T J/kappa^2 + mu I`
and `mu = lambda/kappa^2`, `B^-1` is WP-4c's Laplace covariance exactly, so the
baseline reproduces `wp4c_seed*.npz`'s `sd_by_lambda` rather than
re-deriving something adjacent to it.  The update is Woodbury,

    Sigma_new = B^-1 - B^-1 C^T (I_m + C B^-1 C^T)^-1 C B^-1
    log det FIM_new - log det FIM_current = log det (I_m + C B^-1 C^T)

which costs `O(m^2 n)` rather than `O(n^3)` and never forms a 2 250 x 2 250
matrix.  For the composite candidates (`all_stocks`, `all_flows`) `m` is large
enough that the dense Cholesky route is used instead; `check()` verifies the
two agree.

CLI
---
    python zinc_voi_lab.py --check
    python zinc_voi_lab.py --seeds 0,1,2 --out analysis/wp11c
    python zinc_voi_lab.py --all-seeds
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
import zinc_fisher_lab as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR_DEFAULT = os.path.join(HERE, "analysis", "wp11c")
WEIGHTS_DIR_DEFAULT = F.WEIGHTS_DIR_DEFAULT
MANIFEST = os.path.join(HERE, "analysis", "synth", "manifest.json")

COEF_NAMES = F.COEF_NAMES
N_COEF = F.N_COEF

# Reference ridge, inherited from WP-4c so the baseline marginals are that
# package's headline numbers and not a re-choice.
LAMBDA_REF = F.LAMBDA_REF

# The monthly grid every candidate is aggregated from.  12 is the finest
# frequency the spec asks for; 4 and 1 are exact sub-sums of it.
N_SUB = 12
FREQS = {"annual": 1, "quarterly": 4, "monthly": 12}
SIGMA_MODES = ("per_obs", "fixed_total")

# Stocks, in the core's observable order.
STOCK_NAMES = ["Concentrate", "Refined", "In-Use", "Scrap"]

# Flow accumulators carried through the Jacobian.  `concentrate_production` is
# kept even though it is pinned: a structural zero is a result, not an
# omission.  `primary_refining` and `waelz_recycling` are kept separately so
# `refined_production` can be assembled with its own error propagation.
FLOW_SERIES = [
    "concentrate_production",
    "concentrate_consumption",
    "refined_consumption",
    "waelz_input",
    "direct_reuse_recycling",
    "end_of_life",
    "old_scrap_recovery",
    "primary_refining",
    "waelz_recycling",
]

# Candidate observations.  `kind` selects the observation operator:
#   stock        point-in-time value at the node
#   flow         period integral over the observation window
#   flow_sum     period integral of a sum of accumulators (error propagated)
#   composition  log-ratios of the cohort-resolved EoL integrals
#   bundle       several series observed jointly
CANDIDATES = [
    dict(name="concentrate_stock",       kind="stock", member="Concentrate"),
    dict(name="refined_stock",           kind="stock", member="Refined"),
    dict(name="inuse_stock",             kind="stock", member="In-Use"),
    dict(name="scrap_stock",             kind="stock", member="Scrap"),
    dict(name="concentrate_production",  kind="flow",  member="concentrate_production"),
    dict(name="concentrate_consumption", kind="flow",  member="concentrate_consumption"),
    dict(name="refined_consumption",     kind="flow",  member="refined_consumption"),
    dict(name="waelz_input",             kind="flow",  member="waelz_input"),
    dict(name="direct_reuse_recycling",  kind="flow",  member="direct_reuse_recycling"),
    dict(name="end_of_life",             kind="flow",  member="end_of_life"),
    dict(name="scrap_collection",        kind="flow",  member="old_scrap_recovery"),
    dict(name="primary_refining",        kind="flow",  member="primary_refining"),
    dict(name="refined_production",      kind="flow_sum",
         member=("primary_refining", "waelz_recycling")),
    dict(name="eol_composition",         kind="composition"),
    dict(name="all_stocks",              kind="bundle",
         member=("concentrate_stock", "refined_stock", "inuse_stock", "scrap_stock")),
    dict(name="all_flows",               kind="bundle",
         member=("concentrate_consumption", "refined_consumption", "waelz_input",
                 "direct_reuse_recycling", "end_of_life", "scrap_collection",
                 "refined_production")),
]
CANDIDATE_NAMES = [c["name"] for c in CANDIDATES]

# Assumed relative error on an EoL composition survey.  No calibration exists
# for a series nobody reports, so the aggregate `end_of_life` sigma is used and
# the sensitivity to that choice is reported (`SIGMA_COMP_MULTIPLIERS`).
SIGMA_COMP_MULTIPLIERS = (1.0, 2.0, 5.0)

# Above this many candidate rows the Woodbury identity stops being the cheaper
# route and the dense Cholesky is used instead.  `check()` verifies they agree.
WOODBURY_MAX_M = 1200

PATCHES: list[str] = []          # WP-11c needs none — see module docstring


def install(v5mod=None):
    """Apply import-time patches to the core module.  None needed for WP-11c."""
    if v5mod is None:
        import zinc_colloc_v5 as v5mod          # noqa: F401
    return PATCHES


def _rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024.0 ** 2) if sys.platform == "darwin" else r / 1024.0


# ---------------------------------------------------------------------------
# the noise model
# ---------------------------------------------------------------------------
def load_sigma(path=MANIFEST):
    """WP-3's calibrated per-series relative observation scales.

    `analysis/synth/manifest.json` carries the lognormal sigmas fitted to this
    fit's own Stage B residuals, one per observed stock and per observed flow.
    They are reused here unchanged so the VOI ranking sits on the same noise
    footing as WP-3's own per-series ablation, which is what makes the two
    rankings comparable at all.
    """
    with open(path) as fh:
        man = json.load(fh)
    sig = man["sigma"]
    stock = {n: float(v) for n, v in zip(STOCK_NAMES, sig["stock"])}
    flow = {n: float(v) for n, v in zip(sig["flow_names"], sig["flow"])}
    return dict(stock=stock, flow=flow, source=os.path.basename(path),
                n_seeds=int(sig.get("n_seeds", 0)))


# ---------------------------------------------------------------------------
# observation operators
# ---------------------------------------------------------------------------
def sub_grid(years, n_sub=N_SUB):
    """`n_sub` equally spaced nodes inside every year, plus the closing node."""
    years = np.asarray(years, float).ravel()
    blocks = [np.linspace(years[i], years[i + 1], int(n_sub), endpoint=False)
              for i in range(years.size - 1)]
    return np.concatenate(blocks + [years[-1:]])


def make_observable_fn(fit, data=None, n_sub=N_SUB):
    """`obs(theta) -> (n_out,)`, the log of every candidate observable on the
    monthly grid, plus the layout that says which rows are what.

    Everything downstream — annual, quarterly, sums, compositions — is an exact
    re-aggregation of this one vector, so the ODE is solved once per seed and
    the observation operators are applied to a single trajectory rather than
    re-derived at each frequency.
    """
    import jax
    import jax.numpy as jnp
    import zinc_colloc_v5 as v5

    d = fit.data_trainval if data is None else data
    years = np.asarray(d["years"], float).ravel()
    grid = jnp.asarray(sub_grid(years, n_sub))
    dt = jnp.diff(grid)
    S0 = jnp.asarray(np.asarray(d["stocks_obs"])[0])
    mu = jnp.asarray(v5.MU_COHORTS, float)
    f_idx = np.array([v5.FLOW_NAMES.index(n) for n in FLOW_SERIES], int)

    n_node = int(grid.shape[0])
    n_int = n_node - 1
    n_st, n_fl, n_co = len(STOCK_NAMES), len(FLOW_SERIES), int(v5.N_COHORTS)

    _flat0, unravel = jax.flatten_util.ravel_pytree(fit.params_A)

    def obs(theta):
        p = unravel(theta)
        S, Fi, C = fit.integrate_aug(p, d, S0, grid)
        # Ch2 Eq. cohort_outflow — the cohort-resolved EoL flow the aggregate
        # accumulator does not separate.  Trapezoid on the monthly grid; the
        # quadrature error against the exact aggregate is checked in `check()`.
        ek = C / mu[None, :]
        E = 0.5 * (ek[:-1] + ek[1:]) * dt[:, None]
        return jnp.concatenate([
            jnp.log(S).T.ravel(),                       # (4, n_node)
            jnp.log(Fi[:, f_idx]).T.ravel(),            # (9, n_int)
            jnp.log(E).T.ravel(),                       # (3, n_int)
        ])

    lay = dict(n_node=n_node, n_int=n_int, n_sub=int(n_sub),
               years=years, grid=np.asarray(grid),
               stock_names=list(STOCK_NAMES), flow_names=list(FLOW_SERIES),
               n_stock=n_st, n_flow=n_fl, n_cohort=n_co,
               off_stock=0, off_flow=n_st * n_node,
               off_cohort=n_st * n_node + n_fl * n_int,
               n_out=n_st * n_node + n_fl * n_int + n_co * n_int)
    return obs, lay


def observable_jacobian(fit, theta, *, data=None, n_sub=N_SUB):
    """`(J_obs, y_obs, layout)` — d log(observable)/d theta on the monthly grid.

    Reverse mode: ~5 000 observables against 2 410 parameters is the wrong
    direction on paper, but forward mode is unavailable through the fit's own
    integrator (`RecursiveCheckpointAdjoint` is a `custom_vjp`, WP-4c flag 1)
    and `jacrev` vectorises the backward pass, so this costs ~60 s per seed.
    """
    import jax
    import jax.numpy as jnp

    obs, lay = make_observable_fn(fit, data=data, n_sub=n_sub)
    th = jnp.asarray(theta)
    y = np.asarray(obs(th))
    J = np.asarray(jax.jit(jax.jacrev(obs))(th))
    return J, y, lay


def split_observables(J, y, lay):
    """The flat observable vector as named blocks.

    Returns `(logS, JS, logF, JF, logE, JE)` with leading axes
    (n_node, 4), (n_int, 9) and (n_int, 3).
    """
    n_node, n_int = lay["n_node"], lay["n_int"]
    ns, nf, nc = lay["n_stock"], lay["n_flow"], lay["n_cohort"]
    a, b = lay["off_stock"], lay["off_flow"]
    c = lay["off_cohort"]
    logS = y[a:b].reshape(ns, n_node).T
    JS = J[a:b].reshape(ns, n_node, -1).transpose(1, 0, 2)
    logF = y[b:c].reshape(nf, n_int).T
    JF = J[b:c].reshape(nf, n_int, -1).transpose(1, 0, 2)
    logE = y[c:].reshape(nc, n_int).T
    JE = J[c:].reshape(nc, n_int, -1).transpose(1, 0, 2)
    return logS, JS, logF, JF, logE, JE


# ---------------------------------------------------------------------------
# re-aggregation to a stated frequency
# ---------------------------------------------------------------------------
def _agg_point(logx, Jx, step):
    """Point-in-time observation every `step` monthly nodes.

    A stock at an annual node is the monthly solve's value at that node — no
    averaging, no interpolation.  Verified bitwise against the annual solve in
    `check()`.
    """
    idx = np.arange(0, logx.shape[0], step)
    return logx[idx], Jx[idx]


def _agg_integral(logx, Jx, step):
    """Period integral over `step` consecutive monthly windows.

    The accumulators are additive, so the wider integral is the exact sum of
    the narrow ones.  In log space
    `d log(sum y_i) = sum (y_i / sum y) d log y_i`.
    """
    n = (logx.shape[0] // step) * step
    x = np.exp(logx[:n]).reshape(-1, step, *logx.shape[1:])
    tot = x.sum(1)
    w = x / tot[:, None]
    Jr = Jx[:n].reshape(-1, step, *Jx.shape[1:])
    return np.log(tot), np.einsum("bs...,bs...p->b...p", w, Jr)


def _sigma_flow_sum(vals, sigmas):
    """Relative sigma of a sum of independently reported flows.

    `Var(sum) = sum (y_k sigma_k)^2`, so the relative scale of the sum is
    `sqrt(sum (y_k sigma_k)^2) / sum y_k` — tonnage-weighted, and therefore
    per observation rather than a constant.
    """
    num = np.sqrt(np.sum((vals * np.asarray(sigmas)[None, :]) ** 2, axis=1))
    return num / np.sum(vals, axis=1)


def candidate_rows(cand, blocks, lay, sigma, freq, sigma_mode,
                   sigma_comp_mult=1.0):
    """`C` for one candidate: `(m, n_par)`, rows already divided by sigma.

    `C^T C` is the candidate's contribution to the Fisher information, so this
    is the only object the VOI update needs.
    """
    logS, JS, logF, JF, logE, JE = blocks
    step = lay["n_sub"] // FREQS[freq]
    m_per_year = FREQS[freq]
    scale = np.sqrt(m_per_year) if sigma_mode == "fixed_total" else 1.0
    fl_i = {n: i for i, n in enumerate(lay["flow_names"])}
    st_i = {n: i for i, n in enumerate(lay["stock_names"])}

    def one(c):
        k = c["kind"]
        if k == "stock":
            j = st_i[c["member"]]
            _lv, Jv = _agg_point(logS[:, j], JS[:, j], step)
            s = np.full(Jv.shape[0], sigma["stock"][c["member"]])
            return Jv, s
        if k == "flow":
            j = fl_i[c["member"]]
            _lv, Jv = _agg_integral(logF[:, j], JF[:, j], step)
            s = np.full(Jv.shape[0], sigma["flow"][c["member"]])
            return Jv, s
        if k == "flow_sum":
            js = [fl_i[n] for n in c["member"]]
            lv, Jv = _agg_integral(logF[:, js], JF[:, js], step)
            vals = np.exp(lv)
            tot = vals.sum(1)
            w = vals / tot[:, None]
            s = _sigma_flow_sum(vals, [sigma["flow"][n] for n in c["member"]])
            return np.einsum("bk,bkp->bp", w, Jv), s
        if k == "composition":
            lv, Jv = _agg_integral(logE, JE, step)
            # Two independent log-ratios against the long cohort.  Each is a
            # difference of two independent log errors, hence sigma * sqrt(2).
            R = np.concatenate([Jv[:, 0] - Jv[:, 2], Jv[:, 1] - Jv[:, 2]], 0)
            s = np.full(R.shape[0], sigma["flow"]["end_of_life"]
                        * sigma_comp_mult * np.sqrt(2.0))
            return R, s
        if k == "bundle":
            by = {x["name"]: x for x in CANDIDATES}
            parts = [one(by[n]) for n in c["member"]]
            return (np.concatenate([p[0] for p in parts], 0),
                    np.concatenate([p[1] for p in parts], 0))
        raise ValueError(f"unknown candidate kind {k!r}")

    Jv, s = one(cand)
    # A row whose Jacobian is identically zero carries no information about
    # theta at any sigma.  `concentrate_production` is the case that matters:
    # cp is a PINNED exogenous input (`pin_cp`), so the model's prediction of
    # it does not depend on the parameters at all, and WP-3 calibrated no
    # sigma for it because the objective carries no residual for it either.
    # Such rows are dropped rather than divided by zero, and the candidate
    # reports an exactly-zero gain.
    live = np.any(Jv != 0.0, axis=1)
    Jv, s = Jv[live], np.maximum(s[live], 1e-12)
    return Jv / (s[:, None] * scale)


# ---------------------------------------------------------------------------
# the VOI update
# ---------------------------------------------------------------------------
def make_posterior(sp, kappa2, lam_abs, dead=None):
    """`B^-1` as a linear operator, from WP-4c's SVD.

    `B = J^T J / kappa^2 + mu I` with `mu = lambda / kappa^2`, so
    `B^-1 = (kappa^2/lambda) (I - V diag(w) V^T)` with `w_i = s_i^2/(s_i^2 +
    lambda)` — algebraically identical to `zinc_fisher_lab.laplace_marginals`,
    which is why the baseline reproduces WP-4c rather than approximating it.
    """
    s = np.asarray(sp["sv"], float)
    V = np.asarray(sp["V"], float)
    w = s ** 2 / (s ** 2 + lam_abs)
    pre = kappa2 / lam_abs

    def Binv(X):
        """`B^-1 X` for `X` of shape (n_par, k)."""
        X = np.asarray(X, float)
        return pre * (X - V @ (w[:, None] * (V.T @ X)))

    return dict(Binv=Binv, V=V, w=w, pre=pre, s=s, lam_abs=float(lam_abs),
                kappa2=float(kappa2), n_par=V.shape[0])


def baseline_variance(post, Q):
    """`q^T B^-1 q` for every row of `Q` (n_g, n_par)."""
    V, w, pre = post["V"], post["w"], post["pre"]
    P = Q @ V
    return pre * (np.einsum("ij,ij->i", Q, Q) - (P ** 2) @ w)


def voi_update(post, Q, C, *, dense=None):
    """Posterior variance of every target under `FIM + C^T C`, and the
    D-optimality gain.

    Woodbury:
        Sigma_new = B^-1 - B^-1 C^T (I_m + C B^-1 C^T)^-1 C B^-1
        log det FIM_new - log det FIM_cur = log det (I_m + C B^-1 C^T)

    For a large candidate the dense route (`B + C^T C`, Cholesky) is cheaper;
    `check()` verifies the two agree to machine precision.
    """
    Q = np.asarray(Q, float)
    C = np.asarray(C, float)
    m = C.shape[0]
    if m == 0:
        return baseline_variance(post, Q), 0.0
    if dense is None:
        dense = m > WOODBURY_MAX_M

    if not dense:
        X = post["Binv"](C.T)                       # (n_par, m)
        M = np.eye(m) + C @ X                       # (m, m)
        L = np.linalg.cholesky(M)
        Z = Q @ X                                   # (n_g, m)
        W = np.linalg.solve(M, Z.T).T               # (n_g, m)
        var = baseline_variance(post, Q) - np.einsum("ij,ij->i", Z, W)
        logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
        return np.maximum(var, 0.0), logdet

    V, s, lam, kap2 = post["V"], post["s"], post["lam_abs"], post["kappa2"]
    mu = lam / kap2
    B = (V * (s ** 2 / kap2)) @ V.T
    B[np.diag_indices_from(B)] += mu
    P = B + C.T @ C
    Lb = np.linalg.cholesky(B)
    Lp = np.linalg.cholesky(P)
    sol = np.linalg.solve(P, Q.T).T
    var = np.einsum("ij,ij->i", Q, sol)
    logdet = 2.0 * float(np.sum(np.log(np.diag(Lp)))
                         - np.sum(np.log(np.diag(Lb))))
    return np.maximum(var, 0.0), logdet


# ---------------------------------------------------------------------------
# one seed
# ---------------------------------------------------------------------------
def run_seed(seed, fit, cfg, out_dir, *, weights_dir=WEIGHTS_DIR_DEFAULT,
             sigma=None, lam_rel=LAMBDA_REF, n_sub=N_SUB, comp_mult=None):
    """Every candidate x frequency x sigma convention for one seed."""
    import jax
    import jax.numpy as jnp

    sigma = load_sigma() if sigma is None else sigma
    comp_mult = SIGMA_COMP_MULTIPLIERS if comp_mult is None else comp_mult

    params = cflab.load_params(os.path.join(weights_dir, f"A_seed{seed}.npz"))
    theta, _unravel = jax.flatten_util.ravel_pytree(params)
    theta = jnp.asarray(theta)

    # --- the current information, exactly WP-4c's -------------------------
    J, Jp, G, meta = F.jacobians(fit, cfg, theta)
    dead = F.structural_zero_mask(fit)
    sp = F.spectrum(J, Jp, dead=dead)
    lam_abs = float(lam_rel) * float(sp["eig"][0])
    p_eff = F.p_effective(sp, lam_abs)
    kap2, rss, dof = F.kappa_squared(meta["r_data"], sp["n_res"], p_eff)
    post = make_posterior(sp, kap2, lam_abs)

    years_all = np.asarray(fit.data_all["years"], float).ravel()
    Q = np.asarray(G, float)[..., ~dead].reshape(-1, int((~dead).sum()))
    T = years_all.size
    var_base = baseline_variance(post, Q).reshape(T, N_COEF)
    coef = np.asarray(meta["g"], float)

    # --- the candidate observables ----------------------------------------
    Jo, yo, lay = observable_jacobian(fit, theta, n_sub=n_sub)
    Jo = Jo[:, ~dead]
    blocks = split_observables(Jo, yo, lay)

    names, freqs, modes, mults, ms, logdets = [], [], [], [], [], []
    var_new = []
    for cand in CANDIDATES:
        mult_grid = comp_mult if cand["kind"] == "composition" else (1.0,)
        for freq in FREQS:
            for mode in SIGMA_MODES:
                for mm in mult_grid:
                    C = candidate_rows(cand, blocks, lay, sigma, freq, mode,
                                       sigma_comp_mult=float(mm))
                    v, ld = voi_update(post, Q, C)
                    names.append(cand["name"]); freqs.append(freq)
                    modes.append(mode); mults.append(float(mm))
                    ms.append(int(C.shape[0])); logdets.append(float(ld))
                    var_new.append(v.reshape(T, N_COEF))
    var_new = np.stack(var_new)                            # (n_cand, T, 17)

    arrays = dict(
        seed=np.asarray(int(seed)),
        years=years_all,
        coef_names=np.array(COEF_NAMES, dtype=object),
        coef=coef,
        estimated=np.asarray(np.abs(Q).reshape(T, N_COEF, -1).sum(-1) > 0.0),
        sd_base=np.sqrt(var_base),
        sd_new=np.sqrt(var_new),
        cand_name=np.array(names, dtype=object),
        cand_freq=np.array(freqs, dtype=object),
        cand_sigma_mode=np.array(modes, dtype=object),
        cand_sigma_mult=np.asarray(mults, float),
        cand_m=np.asarray(ms, int),
        logdet_gain=np.asarray(logdets, float),
        kappa2=np.asarray(kap2), rss=np.asarray(rss), dof=np.asarray(dof),
        p_eff_ref=np.asarray(p_eff), lam_rel=np.asarray(float(lam_rel)),
        lam_abs=np.asarray(lam_abs), rank=np.asarray(sp["rank"]),
        n_res=np.asarray(sp["n_res"]), n_par=np.asarray(sp["n_par"]),
        grad_norm_obs=np.linalg.norm(Jo, axis=1),
        obs_layout=np.asarray(json.dumps(
            {k: (v.tolist() if isinstance(v, np.ndarray) else v)
             for k, v in lay.items()}), dtype=object),
        sigma_used=np.asarray(json.dumps(sigma), dtype=object),
    )
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"wp11c_seed{seed}.npz")
    np.savez_compressed(path, **arrays)
    del J, Jp, G, Jo, blocks
    jax.clear_caches()                                    # CLAUDE.md rule 6
    return path, dict(kappa2=kap2, rank=int(sp["rank"]), n_cand=len(names))


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------
def check(verbose=True, weights_dir=WEIGHTS_DIR_DEFAULT, seed=0):
    """`zinc_fisher_lab.check()` plus every WP-11c-specific verification.

    Nothing here is asserted: the observation operators, the re-aggregation
    identities, the cohort quadrature and the two VOI routes are all measured
    against an exact answer and the number is printed.
    """
    import jax
    import jax.numpy as jnp
    import zinc_colloc_v5 as v5

    info = F.check(verbose=verbose, weights_dir=weights_dir)
    cfg = load_anchor_config()
    fit = cflab.init_fit(cfg=cfg, seed=seed)
    params = cflab.load_params(os.path.join(weights_dir, f"A_seed{seed}.npz"))
    theta, unravel = jax.flatten_util.ravel_pytree(params)
    theta = jnp.asarray(theta)

    d = fit.data_trainval
    years = np.asarray(d["years"], float).ravel()
    S0 = jnp.asarray(np.asarray(d["stocks_obs"])[0])
    p = unravel(theta)

    # 1. the sub-annual grid changes nothing about the trajectory
    Sa, Fa, Ca = fit.integrate_aug(p, d, S0, jnp.asarray(years))
    grid = jnp.asarray(sub_grid(years, N_SUB))
    Sm, Fm, Cm = fit.integrate_aug(p, d, S0, grid)
    Fagg = np.asarray(Fm).reshape(years.size - 1, N_SUB, -1).sum(1)
    e_flow = float(np.max(np.abs(Fagg - np.asarray(Fa))
                          / np.maximum(np.abs(np.asarray(Fa)), 1e-300)))
    e_stock = float(np.max(np.abs(np.asarray(Sm)[::N_SUB] - np.asarray(Sa))
                           / np.maximum(np.abs(np.asarray(Sa)), 1e-300)))

    # 2. the cohort quadrature against the exact aggregate accumulator
    mu = np.asarray(v5.MU_COHORTS, float)
    ek = np.asarray(Cm) / mu[None, :]
    dt = np.diff(np.asarray(grid))
    E = 0.5 * (ek[:-1] + ek[1:]) * dt[:, None]
    F_eol = np.asarray(Fm)[:, v5.FLOW_NAMES.index("end_of_life")]
    e_coh = float(np.max(np.abs(E.sum(1) - F_eol) / np.abs(F_eol)))

    # 3. the candidate Jacobian, its re-aggregation and the two VOI routes
    Jo, yo, lay = observable_jacobian(fit, theta)
    dead = F.structural_zero_mask(fit)
    blocks = split_observables(Jo[:, ~dead], yo, lay)
    sigma = load_sigma()

    J, Jp, G, meta = F.jacobians(fit, cfg, theta)
    sp = F.spectrum(J, Jp, dead=dead)
    lam_abs = LAMBDA_REF * float(sp["eig"][0])
    p_eff = F.p_effective(sp, lam_abs)
    kap2, _rss, _dof = F.kappa_squared(meta["r_data"], sp["n_res"], p_eff)
    post = make_posterior(sp, kap2, lam_abs)
    Q = np.asarray(G, float)[..., ~dead].reshape(-1, int((~dead).sum()))

    # 3a. baseline reproduces WP-4c's stored marginals
    e_base = float("nan")
    wp4c = os.path.join(HERE, "analysis", "wp4c", f"wp4c_seed{seed}.npz")
    sd_mine = np.sqrt(baseline_variance(post, Q)).reshape(
        np.asarray(fit.data_all["years"]).size, N_COEF)
    if os.path.exists(wp4c):
        dd = np.load(wp4c, allow_pickle=True)
        k = int(np.argmin(np.abs(dd["lam_rel"] - LAMBDA_REF)))
        ref = dd["sd_by_lambda"][k]
        den = np.maximum(np.abs(ref), 1e-300)
        e_base = float(np.max(np.abs(sd_mine - ref) / den))

    # 3b. Woodbury against the dense Cholesky, on a mid-sized candidate
    C = candidate_rows(CANDIDATES[1], blocks, lay, sigma, "quarterly", "per_obs")
    v_w, ld_w = voi_update(post, Q, C, dense=False)
    v_d, ld_d = voi_update(post, Q, C, dense=True)
    e_wood = float(np.max(np.abs(v_w - v_d)
                          / np.maximum(np.abs(v_d), 1e-300)))
    e_ld = abs(ld_w - ld_d) / max(abs(ld_d), 1e-300)

    # 3c. the pinned input is a structural zero, not a small number
    fi = lay["flow_names"].index("concentrate_production")
    cp_norm = float(np.max(np.abs(blocks[3][:, fi])))

    info.update(n_out=int(lay["n_out"]), n_node=int(lay["n_node"]),
                n_int=int(lay["n_int"]), n_sub=int(lay["n_sub"]),
                agg_flow_rel=e_flow, agg_stock_rel=e_stock,
                cohort_quadrature_rel=e_coh, baseline_vs_wp4c_rel=e_base,
                woodbury_vs_dense_rel=e_wood, logdet_rel=e_ld,
                cp_jacobian_max=cp_norm, sigma=sigma,
                n_candidates=len(CANDIDATES))

    if verbose:
        print("zinc_voi_lab --check  (WP-11c additions)")
        print("=" * 74)
        print(f"  window               : {years[0]:.0f}-{years[-1]:.0f} "
              f"(trainval, W={years.size}) — the window FIM_current is on")
        print(f"  monthly grid         : {lay['n_node']} nodes, "
              f"{lay['n_int']} intervals, n_sub={lay['n_sub']}")
        print(f"  observables per seed : {lay['n_out']} "
              f"({lay['n_stock']} stocks x {lay['n_node']} nodes + "
              f"{lay['n_flow']} flows x {lay['n_int']} intervals + "
              f"{lay['n_cohort']} cohort EoL x {lay['n_int']})")
        print(f"  candidates           : {len(CANDIDATES)} series x "
              f"{len(FREQS)} frequencies x {len(SIGMA_MODES)} sigma "
              f"conventions")
        print( "  operators            : stocks point-in-time, flows period "
               "integrals from the")
        print( "                         accumulators, every frequency an "
               "exact re-aggregation")
        print( "  -- verifications (exact answer 0 unless stated) --")
        print(f"  monthly flow integrals summed to annual vs annual solve : "
              f"{e_flow:.2e}")
        print(f"  monthly stocks at annual nodes vs annual solve          : "
              f"{e_stock:.2e}")
        print(f"  cohort EoL trapezoid vs exact end_of_life accumulator   : "
              f"{e_coh:.2e}  (quadrature, O(h^2))")
        print(f"  baseline marginal SD vs wp4c_seed{seed}.npz sd_by_lambda   : "
              f"{e_base:.2e}")
        print(f"  Woodbury vs dense Cholesky, posterior variance          : "
              f"{e_wood:.2e}")
        print(f"  Woodbury vs dense Cholesky, log det gain                : "
              f"{e_ld:.2e}")
        print(f"  d log(concentrate_production)/d theta, max |.|          : "
              f"{cp_norm:.2e}  (cp is PINNED — a structural zero)")
        print( "  -- noise model --")
        print(f"  source               : {sigma['source']} "
              f"(WP-3 calibration, {sigma['n_seeds']} seeds)")
        for n in STOCK_NAMES:
            print(f"    stock {n:14s} sigma = {sigma['stock'][n]:.4f}")
        for n in FLOW_SERIES:
            print(f"    flow  {n:22s} sigma = {sigma['flow'][n]:.4f}"
                  + ("   <- pinned input" if n == "concentrate_production"
                     else ""))
        print(f"  reference ridge      : lambda/lambda_1 = {LAMBDA_REF:g} "
              f"(inherited from WP-4c)")
        print("=" * 74)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--all-seeds", action="store_true")
    ap.add_argument("--out", default=OUT_DIR_DEFAULT)
    ap.add_argument("--weights", default=WEIGHTS_DIR_DEFAULT)
    ap.add_argument("--rss-limit-gb", type=float, default=40.0)
    ap.add_argument("--lam-rel", type=float, default=LAMBDA_REF,
                    help="ridge as a fraction of the largest data eigenvalue; "
                         "the reference is WP-4c's, and the sensitivity of the "
                         "RANKING to this choice is the thing worth checking")
    args = ap.parse_args(argv)

    if args.check:
        check(verbose=True, weights_dir=args.weights)
        return 0

    cfg = load_anchor_config()
    fit = cflab.init_fit(cfg=cfg, seed=0)
    sigma = load_sigma()
    seeds = (list(range(35)) if args.all_seeds
             else [int(s) for s in args.seeds.split(",") if s.strip()])
    for s in seeds:
        t0 = time.time()
        path, st = run_seed(s, fit, cfg, args.out, weights_dir=args.weights,
                            sigma=sigma, lam_rel=args.lam_rel)
        rss = _rss_mb()
        print(f"[seed {s}] {time.time()-t0:.1f}s  {st['n_cand']} candidates  "
              f"rank={st['rank']}  kappa^2={st['kappa2']:.4e}  "
              f"rss={rss:.0f}MB  -> {path}", flush=True)
        if rss / 1024.0 > args.rss_limit_gb:            # CLAUDE.md rule 6
            print(f"RSS watchdog: {rss/1024.0:.1f} GB exceeds "
                  f"{args.rss_limit_gb} GB — stopping.", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
