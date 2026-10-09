"""
zinc_baseline.py
================

Regression baseline for the global anthropogenic zinc-cycle PINN.  This module
is a drop-in benchmark that mirrors the v5 collocation architecture **but
replaces the neural network with regularised per-target regressions** (Elastic
Net and/or penalised B-spline GAM).  It is designed to be imported alongside
``zinc_colloc_v5`` and to reuse as much of v5's machinery as possible:

  * the same ``load_zinc_data`` (Stocks / Flows / Exogenous / extras),
  * the same exogenous preprocessor (``preprocess_exog``),
  * the same ODE integrator (``make_integrator``), with the same
    cohort-augmented state and the same ``compute_flows_from_nn``,
  * the same pinning logic (``build_nn_layout``), so cp / τ_ref / τ_waelz /
    τ_olds / τ_diss / frac_fu / frac_eu can be pinned to data exactly as in v5,
  * the same ``FitResult`` plot / diagnose / summary surface — the baseline
    returns a ``BaselineFitResult`` that subclasses ``FitResult`` so every
    plot method works without changes.

The single function the baseline replaces is the parameter-function evaluator
``nn_eval(params, t, S, exog_t, data) -> dict``.  v5's integrator and loss code
treat this evaluator as a black box (any pytree of parameters, any smooth
function), so a regression-based evaluator slots in without touching the
solver, the flow code, or the diagnostics.


Motivation (small-N rationale)
------------------------------
With T ≈ 40 years of supervision per target (and per-year supervision tied
together by mass balance, so the *effective* independent sample size is
smaller still), an unconstrained NN is forced to rely heavily on architectural
priors and weight decay.  A *constrained* regression with explicit
regularisation — Elastic Net for variable selection, P-spline-style ridge for
smoothness — is the natural statistical baseline.  At this T, the structural
econometric literature (Richard 1978, Pindyck 1982) consistently uses a
handful of equations with strong shrinkage; the question this module helps
answer is whether the v5 NN earns its complexity over that baseline.

The two-stage design from v5 carries over unchanged:

* **Stage A** — *Collocation / observation fitting* (always run).
      Each *learned* (non-pinned) target is fit *offline*, by ordinary
      regression, to the empirical α / τ / cp observations from the
      **training split only**.  No ODE integration anywhere in Stage A.
      Hyper-parameters (Elastic Net's α, l1_ratio, GAM smoothing penalty) are
      chosen by k-fold CV on the training split.
* **Stage A diagnostic** — integrate the cohort-augmented ODE with the fitted
      regressors and report stock + flow trajectory error on the data grid.
      This is the **headline result** of the baseline: how globally
      consistent are the locally-fitted rate functions?
* **Stage B** — *Shooting fine-tune* (optional).
      Warm-start from the Stage A coefficients (converted to JAX arrays) and
      run a small number of through-ODE optimisation steps with an anchor
      penalty ``λ ‖θ − θ̂_A‖²`` to prevent drift back into Stage-A failure
      modes.  Because the regression parameter functions are linear in their
      coefficients (Elastic Net) or linear in a fixed basis (GAM), the
      gradients through the ODE solver are clean.


Estimator menu per target
-------------------------
For every *learned* (non-pinned) target the user can pick:

  ``"enet"``   — Elastic Net (sklearn ``ElasticNetCV``) on the full feature
                  vector ``[t_norm, S_norm (4), exog_norm (n_exog)]``.
                  Default.  Variable selection comes from the L1 part.
  ``"gam"``    — Penalised B-spline GAM.  One or two designated "smooth"
                  features get a truncated-power cubic basis with K interior
                  knots placed at quantiles of the training distribution.
                  Remaining features enter linearly and are Elastic-Net
                  regularised.  Smooth coefficients get an L2 (ridge)
                  penalty whose strength is GCV-tuned.
  ``"const"``  — Constant prediction at the training-mean target value.
                  Useful for near-stationary τ's, and as a sanity baseline.

Each target's regressor lives behind its **own link function** so the
prediction is on the right side of its physical bound:

  cp                    → softplus (positive)
  α_*                   → exp      (positive, log link)
  binary τ_*            → sigmoid  (in (0,1))
  frac_fu / frac_eu     → softmax over the un-pinned components on the
                          simplex (pinning consumes DOFs exactly as in v5).
  f_cohort              → softmax over 3 logits  *(see below)*

There is no observation of ``f_cohort``.  Stage A cannot identify it.  By
default the baseline pins f_cohort at ``IC_COHORT_FRACS`` (Rostek Fig. 3 c.
1980 share); Stage B, if run, identifies it through the cohort dynamics
+ stock divergence exactly as the NN version does.


Sharing with zinc_colloc_v5
---------------------------
This module imports symbolically (does not re-define):

  Constants    — N_STOCKS, STOCK_NAMES, N_ALPHAS, ALPHA_NAMES, etc.
  Data loading — load_zinc_data, preprocess_exog, _build_empirical_alphas.
  ODE         — make_integrator, make_rhs, compute_flows_from_nn,
                ode_to_4obs, _make_Y0.
  Diagnostics  — _logmae, _rel_rmse_pct, _mape_pct, _nanmean_safe, _fmt_pct,
                _predict_paths_at, FitResult.
  Layout      — build_nn_layout (which slots are learned vs pinned).

``BaselineFitResult`` subclasses ``FitResult`` and inherits every plot /
diagnose / summary method.  The only override is ``__init__`` (to accept the
regression-coefficient pytree instead of NN weights) plus a small
``estimator_summary()`` helper.

Typical usage::

    import zinc_baseline as zb
    fit_bl = zb.run_baseline("enet_default")
    fit_bl.summary()
    fit_bl.plot_all(save_dir="./plots/baseline")

    # Side-by-side with v5
    import zinc_colloc_v5 as zc
    fit_nn = zc.run("v5_default")
    fit_bl.diagnose(); fit_nn.diagnose()

Dependencies
------------
Required: numpy, pandas, scipy, scikit-learn, jax, optax, openpyxl,
          zinc_colloc_v5 (must be importable on the path).
Optional: diffrax (for the Stage B integrator if used), matplotlib (plots).
"""

from __future__ import annotations

import dataclasses as dc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import optax

# scikit-learn estimators for offline Stage A fitting.  We only use these in
# numpy land; the resulting coefficient arrays are converted to JAX arrays
# for online ODE evaluation.
from sklearn.linear_model import (
    ElasticNetCV, RidgeCV, LinearRegression, LogisticRegression,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

try:
    import diffrax as dfx
    HAS_DIFFRAX = True
except Exception:
    dfx = None
    HAS_DIFFRAX = False

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# Pull v5's shared infrastructure — keep imports explicit so the dependency
# surface is visible at the top of this file.
import zinc_colloc_v5 as zc  # noqa: F401  (we use the module-qualified names below)

from zinc_colloc_v5 import (
    # Constants
    N_STOCKS, STOCK_NAMES,
    N_ALPHAS, ALPHA_NAMES, ALPHA_PARENT_STOCK_IDX,
    N_TAUS_BINARY, TAU_BINARY_NAMES,
    N_MANU_SUP, MANU_SUP_NAMES,
    N_TAU_SUP, TAU_SUP_NAMES,
    N_COHORTS, COHORT_NAMES, MU_COHORTS_NP, MU_COHORTS, IC_COHORT_FRACS,
    N_ODE_STOCKS, IDX_CONC, IDX_REF, IDX_SCRAP,
    N_LOGIT_FU, N_LOGIT_EU, N_LOGIT_COHORT,
    FLOW_NAMES, N_FLOWS, IDX_INUSE_INFLOW,
    # Functions
    _safe_log, _logit,
    exog_fn, _norm_inputs,
    _interval_pick, _yearpoint_pick, _interp_cp, _interp_tau,
    build_nn_layout,
    _build_empirical_alphas,
    load_zinc_data, preprocess_exog,
    ode_to_4obs, compute_flows_from_nn, _make_Y0, make_rhs, make_integrator,
    _predict_paths_at,
    _logmae, _rel_rmse_pct, _mape_pct, _nanmean_safe, _fmt_pct,
    FitResult,
)


# ============================================================================
# 1) Target inventory — what gets fit, on what link, with which estimator
# ============================================================================
# We enumerate every *supervised* scalar quantity emitted by the parameter
# function.  Each entry carries:
#   * `name`   — display name (matches v5 conventions).
#   * `family` — one of {"cp", "alpha", "tau_binary", "frac_fu_new",
#                "frac_fu_loss", "frac_eu_new", "frac_eu_loss"}, which
#                determines the link function and the supervision source.
#   * `link`   — "softplus" | "exp" | "sigmoid" | "logit_pair" | "identity".
#                "logit_pair" is the soft-simplex parametrisation used for
#                the manufacturing fractions when both DOFs are learned
#                (we fit each component on its own logit, then softmax-
#                renormalise at eval time).
#   * `tau_idx`— column index in ``tau_sup_obs`` (only for tau-family
#                targets); -1 for cp / α.
#   * `alpha_idx` — column index in ``alpha_obs`` (only for α).
#
# This list is the *complete* target inventory regardless of pinning.  The
# layout dict (from ``build_nn_layout``) tells us which ones are actually
# learned in a given run; pinned ones are skipped at fit time.

TARGET_INVENTORY: List[Dict[str, Any]] = [
    # cp (always one target if learned)
    dict(name="cp",           family="cp",           link="softplus",
         tau_idx=-1, alpha_idx=-1),

    # α (always learned, one per parent stock)
    dict(name="alpha_cc",     family="alpha",        link="exp",
         tau_idx=-1, alpha_idx=0),
    dict(name="alpha_refc",   family="alpha",        link="exp",
         tau_idx=-1, alpha_idx=1),
    dict(name="alpha_win",    family="alpha",        link="exp",
         tau_idx=-1, alpha_idx=2),
    dict(name="alpha_dr",     family="alpha",        link="exp",
         tau_idx=-1, alpha_idx=3),

    # binary τ (4)
    dict(name="tau_ref",      family="tau_binary",   link="sigmoid",
         tau_idx=0,  alpha_idx=-1),
    dict(name="tau_waelz",    family="tau_binary",   link="sigmoid",
         tau_idx=1,  alpha_idx=-1),
    dict(name="tau_olds",     family="tau_binary",   link="sigmoid",
         tau_idx=2,  alpha_idx=-1),
    dict(name="tau_diss",     family="tau_binary",   link="sigmoid",
         tau_idx=3,  alpha_idx=-1),

    # Manufacturing simplex components — fit on the logit of each component
    # individually; softmax-renormalisation happens at eval time inside
    # ``baseline_eval`` so the simplex constraint is honoured exactly.
    # These are only fit when the corresponding DOF is *learned* in the layout.
    dict(name="frac_fu_new",  family="frac_fu_new",  link="logit",
         tau_idx=4,  alpha_idx=-1),
    dict(name="frac_fu_loss", family="frac_fu_loss", link="logit",
         tau_idx=5,  alpha_idx=-1),
    dict(name="frac_eu_new",  family="frac_eu_new",  link="logit",
         tau_idx=6,  alpha_idx=-1),
    dict(name="frac_eu_loss", family="frac_eu_loss", link="logit",
         tau_idx=7,  alpha_idx=-1),
]
TARGET_INVENTORY_BY_NAME = {t["name"]: t for t in TARGET_INVENTORY}


def _learned_target_names(layout: Dict[str, Any]) -> List[str]:
    """List of inventory names that are LEARNED in the given v5 layout.

    Mirrors the pinning semantics of ``build_nn_layout``:
      * cp is learned iff ``learn_cp`` (i.e. ``layout["pin_cp"]`` is False).
      * All four α's are always learned (no pin flag).
      * Binary τ_j is learned iff ``layout["tau_slots"][j] is not None``.
      * Each manufacturing simplex contributes 0 / 1 / 2 learned DOFs
        depending on (pin_loss, pin_new):
            "full":         both new and loss are learned (2 DOFs).
            "loss_pinned":  only new is learned (1 DOF).
            "new_pinned":   only loss is learned (1 DOF).
            "both_pinned":  neither is learned (0 DOFs).
    """
    learned: List[str] = []
    if not layout["pin_cp"]:
        learned.append("cp")
    learned.extend(ALPHA_NAMES)
    pin_taus = layout["pin_taus_tuple"]
    for j, nm in enumerate(TAU_BINARY_NAMES):
        if not pin_taus[j]:
            learned.append(nm)
    for kind_key, new_nm, loss_nm in (
        ("fu_kind", "frac_fu_new", "frac_fu_loss"),
        ("eu_kind", "frac_eu_new", "frac_eu_loss"),
    ):
        kind = layout[kind_key]
        if kind == "full":
            learned.extend([new_nm, loss_nm])
        elif kind == "loss_pinned":   # loss is pinned → new is learned
            learned.append(new_nm)
        elif kind == "new_pinned":    # new is pinned → loss is learned
            learned.append(loss_nm)
        # both_pinned: nothing learned for this simplex
    return learned


# ============================================================================
# 2) Feature vector and spline basis  (numpy + JAX, kept bit-identical)
# ============================================================================
# The regressors operate on the SAME feature vector as the v5 NN — at fit time
# (numpy) and at ODE-eval time (JAX).  Reusing v5's `_norm_inputs` for the
# JAX side guarantees consistency, but we also need a numpy version because
# Stage A fits offline before any JAX optimisation happens.

def _norm_inputs_np(t, S, exog_t, stats):
    """NumPy mirror of ``zc._norm_inputs``.  Keeps the offline feature vector
    bit-identical to the online (JAX) one.  Both stats dicts (the numpy one
    used here and ``stats_j`` used in JAX) are derived from the same
    ``stats`` dict in ``train_baseline_model`` so there is no drift.
    """
    t_feat = (t - float(stats["t_mean"])) / float(stats["t_std"]) * float(stats["use_time_input"])
    mode = float(stats["stock_norm_mode"])
    S_ref = np.asarray(stats["S_ref"], dtype=float)
    log_scale = float(stats["S_log_scale"])
    S = np.asarray(S, dtype=float)
    S_safe = np.maximum(S, 1.0)
    S_log = np.log(S_safe / np.maximum(S_ref, 1.0))
    S_z   = (S - stats["S_mean"]) / stats["S_std"]
    S_tanh = np.tanh(S_log / max(log_scale, 1e-6))

    if abs(mode - 0.0) < 0.5:
        S_feat_raw = S_z
    elif abs(mode - 1.0) < 0.5:
        S_feat_raw = S_log
    else:
        S_feat_raw = S_tanh
    S_feat = S_feat_raw * float(stats["use_stock_input"])

    exog_norm = (np.asarray(exog_t, dtype=float) - stats["exog_mean"]) / stats["exog_std"]
    return np.concatenate([np.array([t_feat]), S_feat, exog_norm])


def _build_feature_matrix_np(years, S_arr, exog_arr, stats_np):
    """Apply ``_norm_inputs_np`` row-wise.  Used at Stage A fit time.

    Parameters
    ----------
    years    : (M,)
    S_arr    : (M, 4)         observable stocks at the M evaluation rows
    exog_arr : (M, n_exog)    preprocessed exogenous matrix at the same rows
    stats_np : dict           the numpy version of `stats` (same as `stats_j`
                              but with plain numpy arrays).

    Returns
    -------
    X : (M, 1 + 4 + n_exog)   feature matrix, same column ordering as
                              what ``_norm_inputs`` produces in JAX land.
    """
    M = len(years)
    rows = []
    for i in range(M):
        rows.append(_norm_inputs_np(float(years[i]), S_arr[i], exog_arr[i], stats_np))
    return np.asarray(rows, dtype=float)


def feature_dim(n_exog: int) -> int:
    """Number of columns in the per-row feature vector."""
    return 1 + N_STOCKS + n_exog


# ----------------------------------------------------------------------------
# Truncated-power cubic basis for the GAM "smooth" features.
# ----------------------------------------------------------------------------
# We use a TRUNCATED POWER basis rather than a B-spline basis so we can write
# the JAX-side evaluation by hand without depending on scipy at ODE-solve
# time.  For one continuous predictor x with K interior knots
# (k_1 < ... < k_K) the cubic truncated-power basis is
#
#     B(x) = [ x, x^2, x^3, (x - k_1)_+^3, ..., (x - k_K)_+^3 ]    (K+3 cols)
#
# The intercept is supplied separately by the host design matrix so we keep
# it out of `B`.  This basis is numerically less well-conditioned than a
# B-spline basis for large K, but for the K ∈ [3, 6] knots typical of small-N
# zinc-cycle data it is well-behaved, and it has the major advantage of being
# trivially expressible in pure JAX (the smoothness penalty becomes a plain
# ridge on the truncated-power coefficients).
#
# Smoothness penalty
# ------------------
# A pure ridge penalty on truncated-power coefficients is the simplest
# analogue of the P-spline second-difference penalty (Eilers & Marx 1996).
# It is not *identical* — the P-spline penalty acts on adjacent B-spline
# coefficients, whereas a ridge here penalises absolute coefficient size.
# For K ≤ 6 and standardised x the two penalties produce very similar
# smoothing-parameter behaviour; the docstring on ``fit_gam_target`` flags
# this for reproducibility.

@dataclass
class SplineBasisSpec:
    """Knots + degree for a single smooth feature."""
    knots: np.ndarray            # shape (K,)  interior knots, sorted ascending
    degree: int = 3              # cubic by default
    # Indicates which input-feature column this basis acts on.  -1 means
    # "no smooth basis for this target".
    feature_idx: int = -1

    @property
    def n_basis(self) -> int:
        """K + degree, excluding intercept (intercept is handled separately)."""
        return int(self.degree + len(self.knots))


def make_quantile_knots(x_train: np.ndarray, n_interior_knots: int) -> np.ndarray:
    """Place K interior knots at evenly-spaced quantiles of the training data.

    Endpoints (the boundary knots) are intentionally *not* included; for a
    truncated-power basis they correspond to the global cubic term, which
    already lives in the basis.
    """
    x = np.asarray(x_train, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < n_interior_knots + 2:
        n_interior_knots = max(0, x.size - 2)
    if n_interior_knots <= 0:
        return np.array([], dtype=float)
    qs = np.linspace(0.0, 1.0, n_interior_knots + 2)[1:-1]
    knots = np.quantile(x, qs)
    # Ensure uniqueness in case of repeated values (rare with standardised features).
    knots = np.unique(knots)
    return knots.astype(float)


def tpower_basis_np(x: np.ndarray, spec: SplineBasisSpec) -> np.ndarray:
    """Evaluate the truncated power basis on a vector x.

    Returns
    -------
    B : (len(x), spec.n_basis)
    """
    x = np.asarray(x, dtype=float)
    cols = []
    for p in range(1, spec.degree + 1):
        cols.append(x ** p)
    for k in spec.knots:
        d = np.maximum(x - k, 0.0)
        cols.append(d ** spec.degree)
    return np.stack(cols, axis=1) if cols else np.zeros((x.shape[0], 0), dtype=float)


def tpower_basis_jax(x, knots, degree: int):
    """JAX version of ``tpower_basis_np`` operating on a scalar x.

    Returns a (degree + len(knots),) vector — flattened for one input row.
    """
    # Polynomial terms
    poly_terms = jnp.stack([x ** p for p in range(1, degree + 1)])
    # Truncated power terms (knots is a static numpy array baked into the closure)
    if len(knots) == 0:
        return poly_terms
    knots_j = jnp.asarray(knots, dtype=x.dtype)
    diffs = jnp.maximum(x - knots_j, 0.0)
    trunc_terms = diffs ** degree
    return jnp.concatenate([poly_terms, trunc_terms])


# ============================================================================
# 3) Per-target regressor — JAX-friendly parameter container
# ============================================================================
# Each learned target carries:
#   * an intercept (scalar),
#   * a linear coefficient vector matching the raw feature dimension,
#   * (optionally) one or two "smooth" coefficient vectors matching the
#     truncated-power bases for selected smooth features.
#
# The numpy-side fitting produces these as plain numpy arrays.  At the
# boundary between Stage A and Stage A diagnostic / Stage B, we promote
# everything to JAX arrays so the same data structure can be optimised
# with optax + diffrax if Stage B is enabled.
#
# `kind` and `smooth_specs` are STATIC (not in the JAX pytree); they get
# baked into the closure that ``make_baseline_eval`` returns.

@dataclass
class TargetCoefficients:
    """Fitted coefficients for one target.

    intercept   : ()                   the scalar intercept on the linear predictor
    coef_lin    : (n_features,)         linear-term coefficients
    coef_smooth : list of (n_basis_i,)  smooth-term coefficients (one array
                                        per smooth feature, in the order of
                                        `smooth_specs` in the matching TargetSpec).
    """
    intercept: np.ndarray
    coef_lin: np.ndarray
    coef_smooth: List[np.ndarray] = field(default_factory=list)

    def to_jax(self) -> Dict[str, Any]:
        d = dict(intercept=jnp.asarray(self.intercept, dtype=jnp.float64),
                 coef_lin=jnp.asarray(self.coef_lin,  dtype=jnp.float64))
        if self.coef_smooth:
            d["coef_smooth"] = [jnp.asarray(c, dtype=jnp.float64)
                                for c in self.coef_smooth]
        else:
            # JAX-compatible empty placeholder; we use a 0-length list of arrays
            d["coef_smooth"] = []
        return d


@dataclass
class TargetSpec:
    """Static (non-pytree) description of one target's estimator."""
    name: str
    family: str
    link: str
    kind: str                            # "enet" | "gam" | "const"
    smooth_specs: List[SplineBasisSpec] = field(default_factory=list)
    # For diagnostics / inspection only; not used inside the JAX closure.
    notes: str = ""


# ============================================================================
# 4) Link / inverse-link helpers  (numpy and JAX, in pairs)
# ============================================================================
def _apply_link_np(eta: np.ndarray, link: str) -> np.ndarray:
    if link == "softplus":
        return np.log1p(np.exp(np.clip(eta, -50.0, 50.0)))
    if link == "exp":
        return np.exp(np.clip(eta, -50.0, 50.0))
    if link == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -50.0, 50.0)))
    if link == "identity":
        return eta
    if link == "logit":
        # Linear predictor IS the logit; caller does the sigmoid/softmax.
        return eta
    raise ValueError(f"unknown link {link!r}")


def _apply_link_jax(eta, link: str):
    if link == "softplus":
        return jax.nn.softplus(eta)
    if link == "exp":
        return jnp.exp(jnp.clip(eta, -8.0, 8.0))
    if link == "sigmoid":
        return jax.nn.sigmoid(eta)
    if link in ("identity", "logit"):
        return eta
    raise ValueError(f"unknown link {link!r}")


def _inverse_link_obs_np(y: np.ndarray, link: str, eps: float = 1e-6) -> np.ndarray:
    """Compute the observed *linear predictor* η_obs = g(y_obs) for fitting.

    Linear regression then solves η_obs ≈ Xβ.  This is the textbook
    transformation-then-OLS approach to GLM-style problems and is the
    Achilles heel/strength of the regression baseline: it is fast, stable
    and well-understood, but it does not weight residuals by the variance
    function of a "true" GLM.  For our purposes the trade-off is fine —
    the dataset is small enough that the simpler closed-form solve is
    more reproducible than IRLS, and the targets we use this on are
    bounded and reasonably well-behaved.
    """
    y = np.asarray(y, dtype=float)
    if link == "exp":
        return np.log(np.maximum(y, eps))
    if link == "softplus":
        # softplus^{-1}(y) = log(exp(y) - 1) — well-defined for y > 0.
        z = np.maximum(y, eps)
        return np.log(np.expm1(z))
    if link == "sigmoid":
        y_clip = np.clip(y, eps, 1.0 - eps)
        return np.log(y_clip / (1.0 - y_clip))
    if link == "logit":
        # Treat input as a (0,1)-valued component on a simplex.
        y_clip = np.clip(y, eps, 1.0 - eps)
        return np.log(y_clip / (1.0 - y_clip))
    if link == "identity":
        return y
    raise ValueError(f"unknown link {link!r}")


# ============================================================================
# 5) Per-target fitting routines (offline, numpy land)
# ============================================================================
# The high-level contract for every fitter:
#
#   fit_target_*(X, y, *, spec, weights=None, cv_kwargs=None) ->
#       TargetCoefficients
#
# where X is the (M, n_features) raw feature matrix (NOT yet basis-expanded),
# y is the (M,) target on its NATURAL scale (not the linear predictor — the
# fitter takes care of transforming via `spec.link`), and spec is a
# ``TargetSpec`` describing kind / link / smooth features.
#
# The returned coefficients always satisfy:
#
#   eta(x) = intercept + coef_lin @ x + Σ_s coef_smooth_s @ basis_s(x[feat_s])
#   y_hat  = link(eta(x))
#
# even when (e.g.) `kind == "const"` — in which case `coef_lin` is the zero
# vector and `intercept` carries the constant on the linear-predictor scale.
# This uniform parameterisation lets the JAX-side eval be a single arithmetic
# expression regardless of the chosen estimator.

def _expand_design(X_lin: np.ndarray, spec: TargetSpec) -> Tuple[np.ndarray, np.ndarray, List[slice]]:
    """Build the full augmented design [1 | X_lin | B_smooth_1 | ... | B_smooth_S].

    Returns
    -------
    X_full           : (M, 1 + n_features + Σ n_basis_s)
    penalty_diagonal : (1 + n_features + Σ n_basis_s,)   ridge penalty
        weights per column.  Intercept and linear coefs get penalty 1
        (relative weight is set by the alpha hyper-param at fit time);
        smooth coefs get penalty 1 here too — the actual smoothness
        strength is chosen by CV/GCV in the caller.
    smooth_slices    : list of slice objects pointing to each smooth
                       block in ``X_full``.
    """
    M = X_lin.shape[0]
    n_lin = X_lin.shape[1]
    cols  = [np.ones((M, 1)), X_lin]
    smooth_slices: List[slice] = []
    col_offset = 1 + n_lin
    for s in spec.smooth_specs:
        if s.feature_idx < 0 or s.feature_idx >= n_lin:
            raise ValueError(
                f"smooth spec for target {spec.name!r} references invalid "
                f"feature_idx={s.feature_idx} (n_lin={n_lin})"
            )
        Bs = tpower_basis_np(X_lin[:, s.feature_idx], s)
        cols.append(Bs)
        smooth_slices.append(slice(col_offset, col_offset + Bs.shape[1]))
        col_offset += Bs.shape[1]
    X_full = np.concatenate(cols, axis=1) if len(cols) > 1 else cols[0]
    return X_full, smooth_slices


def _coefs_from_solution(beta: np.ndarray, n_lin: int,
                          smooth_slices: List[slice]) -> TargetCoefficients:
    """Slice a stacked coefficient vector back into intercept / linear / smooth parts."""
    intercept = np.array(beta[0])
    coef_lin  = np.array(beta[1:1 + n_lin])
    coef_smooth = [np.array(beta[s]) for s in smooth_slices]
    return TargetCoefficients(
        intercept=intercept, coef_lin=coef_lin, coef_smooth=coef_smooth,
    )


def fit_target_const(X_lin: np.ndarray, y: np.ndarray, *, spec: TargetSpec,
                      weights: Optional[np.ndarray] = None,
                      **_unused) -> TargetCoefficients:
    """Constant prediction at the (weighted) mean of `y` on the linear-predictor scale.

    For the sigmoid link this collapses to a clamped mean; for `exp` it
    collapses to the geometric mean.  The fitted ``intercept`` is the
    transformed mean; ``coef_lin`` is all zeros.
    """
    eta = _inverse_link_obs_np(y, spec.link)
    mask = np.isfinite(eta)
    if weights is None:
        w = np.ones(eta.shape[0])
    else:
        w = np.asarray(weights, dtype=float)
    w = w * mask
    total_w = float(w.sum())
    if total_w <= 0:
        eta_mean = 0.0
    else:
        eta_mean = float(np.sum(np.where(mask, eta * w, 0.0)) / max(total_w, 1e-12))
    n_lin = X_lin.shape[1]
    return TargetCoefficients(
        intercept=np.array(eta_mean),
        coef_lin=np.zeros(n_lin, dtype=float),
        coef_smooth=[],
    )


def fit_target_enet(X_lin: np.ndarray, y: np.ndarray, *, spec: TargetSpec,
                     weights: Optional[np.ndarray] = None,
                     alphas: Optional[np.ndarray] = None,
                     l1_ratios: Tuple[float, ...] = (0.1, 0.5, 0.9),
                     n_cv_splits: int = 5,
                     random_state: int = 0) -> TargetCoefficients:
    """Fit an Elastic Net on the linear-predictor scale.

    Stage A targets are *deterministic* transforms of stocks and flows under
    mass balance, so the residuals we fit are dominated by structural
    misspecification + sample-error rather than i.i.d. Gaussian noise.
    `ElasticNetCV` with K-fold CV nonetheless gives a defensible regularisation
    path: it picks (α, l1_ratio) that minimise out-of-sample MSE on the
    training split alone (we never see the held-out test in Stage A).

    Notes
    -----
    * `weights` are NOT used by `ElasticNetCV` directly — we instead drop
      rows where ``weights == 0`` (a NaN-mask on `y`, in practice).  This is
      a simplification: full sample-weighting would require a custom solver.
      For the zinc dataset the only "weight" we ever use is the missing-data
      mask on ``alpha_obs`` (the first row is NaN by definition), so the
      simplification is harmless.
    * The intercept is included in the model (sklearn's default).
    """
    eta = _inverse_link_obs_np(y, spec.link)
    mask = np.isfinite(eta)
    if weights is not None:
        mask = mask & (np.asarray(weights, dtype=float) > 0)
    Xm = X_lin[mask]
    em = eta[mask]
    if Xm.shape[0] < 4:
        # Not enough rows to fit; fall back to constant.
        return fit_target_const(X_lin, y, spec=spec, weights=weights)

    n_lin = X_lin.shape[1]
    n_cv = max(2, min(n_cv_splits, Xm.shape[0] - 1))
    # Contiguous-block CV (shuffle=False).  The rows are an annual time
    # series with strongly autocorrelated targets, so shuffled K-fold puts
    # year t-1 in train and year t in test, leaking almost the whole signal
    # and biasing the selected penalty toward under-regularisation.
    # shuffle=False keeps each fold a contiguous span of years and makes the
    # estimator exactly deterministic (`random_state` no longer affects the
    # fold assignment; it is retained only for sklearn's solver).
    cv = KFold(n_splits=n_cv, shuffle=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        enet = ElasticNetCV(
            l1_ratio=list(l1_ratios),
            alphas=alphas, cv=cv, fit_intercept=True,
            max_iter=20000, tol=1e-6, random_state=random_state,
        )
        enet.fit(Xm, em)

    intercept = float(enet.intercept_)
    coef_lin  = np.asarray(enet.coef_, dtype=float)
    return TargetCoefficients(
        intercept=np.array(intercept),
        coef_lin=coef_lin,
        coef_smooth=[],
    )


def fit_target_gam(X_lin: np.ndarray, y: np.ndarray, *, spec: TargetSpec,
                    weights: Optional[np.ndarray] = None,
                    n_cv_splits: int = 5,
                    lambda_lin_grid: Optional[np.ndarray] = None,
                    lambda_smooth_grid: Optional[np.ndarray] = None,
                    random_state: int = 0) -> TargetCoefficients:
    """Fit a P-spline-style additive model on the linear-predictor scale.

    Model
    -----
        η(x) = β_0 + Σ_j x_j β_j + Σ_s B_s(x[feat_s]) γ_s

    Penalty
    -------
        λ_lin    Σ_j β_j²        (ridge on linear-effect coefs)
        λ_smooth Σ_s Σ_p γ_{s,p}²  (ridge on truncated-power coefs)

    The intercept is unpenalised.  λ_lin and λ_smooth are chosen jointly by
    a small grid + K-fold CV on the training data, with *contiguous*
    (unshuffled) folds so that neighbouring years cannot leak across the
    split.  This is the natural closed-form analogue of an
    Elastic-Net-regularised B-spline GAM at very small N.

    Caveat — penalty interpretation
    -------------------------------
    The literature P-spline penalty (Eilers & Marx 1996) acts on adjacent
    B-spline coefficient *differences*, which enforces smoothness directly.
    A pure ridge penalty on truncated-power coefs penalises *magnitude*
    instead.  For the standardised, K ∈ [3, 6]-knot bases we use, the two
    behave very similarly under CV; we accept the trade-off in exchange for
    a JAX-friendly evaluator at ODE-solve time.
    """
    if lambda_lin_grid is None:
        lambda_lin_grid = np.logspace(-3, 2, 8)
    if lambda_smooth_grid is None:
        lambda_smooth_grid = np.logspace(-3, 2, 8)

    eta = _inverse_link_obs_np(y, spec.link)
    mask = np.isfinite(eta)
    if weights is not None:
        mask = mask & (np.asarray(weights, dtype=float) > 0)

    Xm = X_lin[mask]
    em = eta[mask]
    if Xm.shape[0] < 6:
        return fit_target_const(X_lin, y, spec=spec, weights=weights)

    # Build augmented design matrix once
    X_full, smooth_slices = _expand_design(Xm, spec)
    p = X_full.shape[1]
    n_lin = Xm.shape[1]

    # Penalty mask: 0 for intercept, λ_lin for linear, λ_smooth for smooth
    is_intercept = np.zeros(p, dtype=bool); is_intercept[0] = True
    is_lin       = np.zeros(p, dtype=bool); is_lin[1:1 + n_lin] = True
    is_smooth    = np.zeros(p, dtype=bool)
    for s in smooth_slices:
        is_smooth[s] = True

    n_cv = max(2, min(n_cv_splits, Xm.shape[0] - 1))
    # Contiguous-block CV — see the note in `fit_target_enet`.  Shuffled
    # folds leak neighbouring years across the split on this autocorrelated
    # annual series; shuffle=False also makes lambda selection (and hence
    # the whole baseline) independent of `random_state`.
    cv = KFold(n_splits=n_cv, shuffle=False)

    best_score = float("inf")
    best_lam = (float(lambda_lin_grid[0]), float(lambda_smooth_grid[0]))

    for lam_l in lambda_lin_grid:
        for lam_s in lambda_smooth_grid:
            pen = np.where(is_lin,    lam_l,
                   np.where(is_smooth, lam_s, 0.0))
            cv_mse = 0.0
            n_used = 0
            for tr, te in cv.split(X_full):
                A   = X_full[tr].T @ X_full[tr] + np.diag(pen)
                rhs = X_full[tr].T @ em[tr]
                try:
                    beta_cv = np.linalg.solve(A, rhs)
                except np.linalg.LinAlgError:
                    cv_mse = float("inf"); break
                pred = X_full[te] @ beta_cv
                cv_mse += float(np.mean((pred - em[te]) ** 2))
                n_used += 1
            if n_used == 0:
                continue
            cv_mse /= n_used
            if cv_mse < best_score:
                best_score = cv_mse
                best_lam = (float(lam_l), float(lam_s))

    lam_l, lam_s = best_lam
    pen = np.where(is_lin,    lam_l,
           np.where(is_smooth, lam_s, 0.0))
    A   = X_full.T @ X_full + np.diag(pen)
    rhs = X_full.T @ em
    beta = np.linalg.solve(A, rhs)
    return _coefs_from_solution(beta, n_lin, smooth_slices)


_FITTERS: Dict[str, Callable[..., TargetCoefficients]] = {
    "enet":  fit_target_enet,
    "gam":   fit_target_gam,
    "const": fit_target_const,
}


# ============================================================================
# 6) Stage A orchestrator — fit every learned target offline
# ============================================================================
@dataclass
class BaselineParamsStatic:
    """The non-pytree part of the baseline's parameter representation.

    Carries enough information for ``make_baseline_eval`` to reconstruct the
    per-target evaluator without knowing anything about how the coefficients
    were fit.
    """
    target_specs: Dict[str, TargetSpec]   # name → TargetSpec
    n_features: int                        # raw linear-feature dimension
    # Diagnostic info from Stage A
    cv_diagnostics: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def _resolve_estimator_kinds(layout: Dict[str, Any],
                              regressor_kinds: Union[str, Dict[str, str]],
                              ) -> Dict[str, str]:
    """Normalise the `regressor_kinds` config into a per-target dict."""
    learned = _learned_target_names(layout)
    if isinstance(regressor_kinds, str):
        return {nm: regressor_kinds for nm in learned}
    if not isinstance(regressor_kinds, dict):
        raise TypeError(
            f"regressor_kinds must be a str or dict, got {type(regressor_kinds).__name__}"
        )
    # Fill in defaults: anything not specified uses "enet"
    out: Dict[str, str] = {}
    for nm in learned:
        kind = regressor_kinds.get(nm, "enet")
        if kind not in _FITTERS:
            raise ValueError(
                f"regressor_kinds[{nm!r}]={kind!r}; must be one of {list(_FITTERS)}"
            )
        out[nm] = kind
    return out


def _resolve_smooth_specs(layout: Dict[str, Any],
                           regressor_kinds: Dict[str, str],
                           gam_smooth_features: Optional[Dict[str, List[Tuple[int, int]]]],
                           X_train: np.ndarray,
                           ) -> Dict[str, List[SplineBasisSpec]]:
    """Determine the smooth-feature configuration per GAM target.

    ``gam_smooth_features`` is a per-target dict ``{name: [(feat_idx, K), ...]}``.
    Any GAM target without an entry gets a default of one smooth on feature
    index 0 (the t feature) with K=4 interior knots — note this will be
    inactive unless ``use_time_input=True`` makes feature 0 non-zero.  When
    that default is uninformative, users should pass an explicit mapping.

    The returned knots are placed at quantiles of the TRAINING distribution
    of the chosen feature.
    """
    out: Dict[str, List[SplineBasisSpec]] = {}
    for nm, kind in regressor_kinds.items():
        if kind != "gam":
            out[nm] = []
            continue
        cfg = (gam_smooth_features or {}).get(nm)
        if cfg is None:
            # Fallback default: smooth on the FIRST stock feature (idx=1) which
            # is almost always non-trivial.  Users should override per target.
            cfg = [(1, 4)]
        specs: List[SplineBasisSpec] = []
        for feat_idx, n_knots in cfg:
            if not (0 <= feat_idx < X_train.shape[1]):
                raise ValueError(
                    f"gam_smooth_features[{nm!r}]: feat_idx {feat_idx} out of range "
                    f"[0, {X_train.shape[1]}); n_features={X_train.shape[1]}"
                )
            knots = make_quantile_knots(X_train[:, feat_idx], int(n_knots))
            specs.append(SplineBasisSpec(knots=knots, degree=3, feature_idx=int(feat_idx)))
        out[nm] = specs
    return out


def fit_baseline_stage_A(*, layout: Dict[str, Any],
                          data_train_np: Dict[str, np.ndarray],
                          stats_np: Dict[str, Any],
                          regressor_kinds: Union[str, Dict[str, str]] = "enet",
                          gam_smooth_features: Optional[Dict[str, List[Tuple[int, int]]]] = None,
                          enet_l1_ratios: Tuple[float, ...] = (0.1, 0.5, 0.9),
                          n_cv_splits: int = 5,
                          random_state: int = 0,
                          verbose: bool = True,
                          ) -> Tuple[Dict[str, TargetCoefficients], BaselineParamsStatic]:
    """Fit every learned target on the training split.

    Parameters
    ----------
    layout : v5 NN-layout dict.  Tells us which targets are learned vs pinned.
    data_train_np : dict with numpy arrays for the training split.  Must include
        ``years_alpha``  : (M_alpha,)         midpoint-year supervision grid for α
                                              (or the year grid if v5 used year-end α)
        ``S_alpha``      : (M_alpha, 4)       stocks at α supervision times
        ``exog_alpha``   : (M_alpha, n_exog)  exog at α supervision times
        ``alpha_obs``    : (M_alpha, 4)       empirical α targets
        ``years_year``   : (M_year,)          integer-year supervision grid for τ/cp
        ``S_year``       : (M_year, 4)        stocks at integer-year supervision times
        ``exog_year``    : (M_year, n_exog)   exog at integer-year supervision times
        ``tau_sup_obs``  : (M_year, 8)        τ supervision (8 columns, TAU_SUP_NAMES)
        ``cp_obs``       : (M_year,)          cp supervision
    stats_np : the v5 ``stats`` dict in numpy form (so ``_norm_inputs_np``
        builds the same feature vector the JAX-side ``_norm_inputs`` would).

    Returns
    -------
    coeffs_by_name : dict name → TargetCoefficients
    static_params  : BaselineParamsStatic (target specs + diagnostics)
    """
    learned = _learned_target_names(layout)
    kinds = _resolve_estimator_kinds(layout, regressor_kinds)

    # Build the two design matrices (α supervision uses midpoint stocks; τ/cp
    # supervision uses integer-year stocks).
    X_alpha = _build_feature_matrix_np(
        data_train_np["years_alpha"], data_train_np["S_alpha"],
        data_train_np["exog_alpha"], stats_np,
    )
    X_year = _build_feature_matrix_np(
        data_train_np["years_year"], data_train_np["S_year"],
        data_train_np["exog_year"], stats_np,
    )
    n_features = X_alpha.shape[1]

    # Smooth-feature config per GAM target — knots derived from the relevant
    # training design matrix (α-grid for α targets; year-grid for τ/cp).
    smooth_specs_alpha = _resolve_smooth_specs(layout, kinds, gam_smooth_features, X_alpha)
    smooth_specs_year  = _resolve_smooth_specs(layout, kinds, gam_smooth_features, X_year)

    coeffs: Dict[str, TargetCoefficients] = {}
    specs:  Dict[str, TargetSpec] = {}
    diagnostics: Dict[str, Dict[str, Any]] = {}

    for nm in learned:
        inv = TARGET_INVENTORY_BY_NAME[nm]
        family = inv["family"]
        kind = kinds[nm]

        # Pick the correct supervision and design matrix per family.
        if family == "alpha":
            X = X_alpha
            y = data_train_np["alpha_obs"][:, int(inv["alpha_idx"])]
            smooth = smooth_specs_alpha.get(nm, [])
        elif family == "cp":
            X = X_year
            y = data_train_np["cp_obs"]
            smooth = smooth_specs_year.get(nm, [])
        else:
            # tau_binary OR a frac_*_new / frac_*_loss component
            X = X_year
            y = data_train_np["tau_sup_obs"][:, int(inv["tau_idx"])]
            smooth = smooth_specs_year.get(nm, [])

        spec = TargetSpec(
            name=nm, family=family, link=inv["link"],
            kind=kind, smooth_specs=smooth,
        )

        fitter = _FITTERS[kind]
        # Drop NaN observations
        mask = np.isfinite(y)
        if not mask.any():
            if verbose:
                print(f"[Stage A] {nm}: no finite observations — using zero intercept.")
            coeffs[nm] = TargetCoefficients(
                intercept=np.array(0.0),
                coef_lin=np.zeros(X.shape[1], dtype=float),
                coef_smooth=[np.zeros(s.n_basis, dtype=float) for s in smooth],
            )
            specs[nm] = spec
            diagnostics[nm] = dict(n_obs=0, kind=kind)
            continue

        fit_kwargs: Dict[str, Any] = dict(spec=spec, weights=mask.astype(float),
                                          n_cv_splits=n_cv_splits,
                                          random_state=random_state)
        if kind == "enet":
            fit_kwargs["l1_ratios"] = enet_l1_ratios
        coeff = fitter(X, y, **fit_kwargs)
        coeffs[nm] = coeff
        specs[nm]  = spec

        # In-sample fit summary (linear predictor scale) — cheap diagnostic.
        if mask.sum() >= 2:
            X_full, sl = _expand_design(X[mask], spec)
            beta = np.concatenate([
                coeff.intercept.reshape(1),
                coeff.coef_lin,
                *coeff.coef_smooth,
            ])
            eta_hat = X_full @ beta
            eta_obs = _inverse_link_obs_np(y[mask], spec.link)
            sse = float(np.sum((eta_hat - eta_obs) ** 2))
            sst = float(np.sum((eta_obs - eta_obs.mean()) ** 2))
            r2  = 1.0 - sse / max(sst, 1e-12)
            diagnostics[nm] = dict(
                kind=kind, link=spec.link, n_obs=int(mask.sum()),
                n_smooth=len(smooth),
                in_sample_r2_eta=r2,
                nonzero_lin=int(np.sum(np.abs(coeff.coef_lin) > 1e-9)),
            )
            if verbose:
                msg = f"  [{kind:5s}] {nm:18s}  n={mask.sum():3d}  R²(η)={r2:+.3f}"
                if kind == "enet":
                    msg += f"  nnz={diagnostics[nm]['nonzero_lin']}/{X.shape[1]}"
                if smooth:
                    knot_summary = ",".join(str(s.n_basis) for s in smooth)
                    msg += f"  smooth_basis={knot_summary}"
                print(msg)
        else:
            diagnostics[nm] = dict(kind=kind, n_obs=int(mask.sum()))

    static = BaselineParamsStatic(
        target_specs=specs, n_features=n_features, cv_diagnostics=diagnostics,
    )
    return coeffs, static


# ============================================================================
# 7) JAX-side evaluator — the nn_eval drop-in
# ============================================================================
# The integrator in v5 calls
#     nn_eval(params, t, S, exog_t, data) -> {cp, alphas, taus,
#                                              frac_fu, frac_eu, f_cohort,
#                                              raw_norm}
# We build a closure with EXACTLY that signature, where the per-target
# coefficient pytree replaces the MLP weights.

def _eta_for_target_jax(coeffs_t: Dict[str, jnp.ndarray],
                        x_lin: jnp.ndarray,
                        spec: TargetSpec) -> jnp.ndarray:
    """Compute the linear predictor η for one target at one input row.

    Parameters
    ----------
    coeffs_t : dict with keys "intercept", "coef_lin", "coef_smooth" (list)
    x_lin    : (n_features,) feature row
    spec     : static TargetSpec (link / smooth_specs)
    """
    eta = coeffs_t["intercept"] + jnp.dot(coeffs_t["coef_lin"], x_lin)
    for k, s in enumerate(spec.smooth_specs):
        b = tpower_basis_jax(x_lin[s.feature_idx], s.knots, s.degree)
        eta = eta + jnp.dot(coeffs_t["coef_smooth"][k], b)
    return eta


def _f_cohort_default_logits():
    """Logits whose softmax equals ``IC_COHORT_FRACS``.

    These are the no-data baseline for f_cohort: in the regression baseline
    we cannot identify f_cohort from local information (it has no
    supervision signal), so we default to the Rostek-fig.-3 c. 1980 prior
    and let Stage B move it if the user enables that.
    """
    p = np.asarray(IC_COHORT_FRACS, dtype=float)
    return np.log(np.maximum(p, 1e-6))


def make_baseline_eval(layout: Dict[str, Any],
                       static: BaselineParamsStatic):
    """Build a JIT-friendly closure with the same signature as v5's ``nn_eval``.

    Returns
    -------
    baseline_eval(params, t, S, exog_t, data) -> dict
        params is a pytree (a dict of per-target coefficient dicts plus a
        ``f_cohort_logits`` array).
    """
    target_specs = static.target_specs
    learned_names = list(target_specs.keys())

    # Static pin info (mirrors what `make_nn_eval` does — but instead of a
    # raw NN output we read from the per-target coefficient pytree).
    pin_cp      = layout["pin_cp"]
    pin_taus    = layout["pin_taus_tuple"]
    pin_fu_loss = layout["pin_fu_loss"]
    pin_eu_loss = layout["pin_eu_loss"]
    pin_fu_new  = layout["pin_fu_new"]
    pin_eu_new  = layout["pin_eu_new"]

    # Index lookups for each scalar target — keep them as Python tuples of
    # (name, spec) so we can iterate inside the JAX closure without dict
    # lookup overhead.
    spec_alpha = tuple((nm, target_specs[nm]) for nm in ALPHA_NAMES)
    # tau / frac specs may be absent (pinned); store None in that slot.
    spec_tau_binary = tuple(
        (nm, target_specs[nm]) if nm in target_specs else (nm, None)
        for nm in TAU_BINARY_NAMES
    )
    spec_fu_new = target_specs.get("frac_fu_new",  None)
    spec_fu_loss= target_specs.get("frac_fu_loss", None)
    spec_eu_new = target_specs.get("frac_eu_new",  None)
    spec_eu_loss= target_specs.get("frac_eu_loss", None)
    spec_cp     = target_specs.get("cp", None)

    def _simplex_value(eta_new, eta_loss, kind,
                       loss_data, new_data):
        """Compose (new, loss, out) on the simplex from per-component logits +
        pinning configuration.  Mirrors v5's ``_simplex_value`` semantics:

          "full":        softmax([η_new, η_loss, 0])
          "loss_pinned": loss := loss_data; split (1 - loss) between
                          new and out via sigmoid(η_new)
          "new_pinned":  new := new_data; split (1 - new) between
                          loss and out via sigmoid(η_loss)
          "both_pinned": both pinned; out determined by closure.
        """
        if kind == "full":
            logits = jnp.stack([eta_new, eta_loss, jnp.zeros_like(eta_new)])
            return jax.nn.softmax(logits)
        if kind == "loss_pinned":
            loss  = jnp.clip(loss_data, 0.0, 1.0 - 1e-6)
            sigma = jax.nn.sigmoid(eta_new)
            avail = jnp.maximum(1.0 - loss, 1e-6)
            new   = sigma         * avail
            out   = (1.0 - sigma) * avail
            return jnp.stack([new, loss, out])
        if kind == "new_pinned":
            new   = jnp.clip(new_data, 0.0, 1.0 - 1e-6)
            sigma = jax.nn.sigmoid(eta_loss)
            avail = jnp.maximum(1.0 - new, 1e-6)
            loss  = sigma         * avail
            out   = (1.0 - sigma) * avail
            return jnp.stack([new, loss, out])
        # both_pinned
        new   = jnp.clip(new_data,  0.0, 1.0 - 1e-6)
        loss  = jnp.clip(loss_data, 0.0, 1.0 - 1e-6)
        out   = jnp.maximum(1.0 - new - loss, 1e-6)
        return jnp.stack([new, loss, out])

    fu_kind = layout["fu_kind"]
    eu_kind = layout["eu_kind"]

    def baseline_eval(params, t, S, exog_t, data):
        # `params` is a pytree of the form
        #     {"targets": {name: {"intercept", "coef_lin", "coef_smooth"}},
        #      "f_cohort_logits": (3,) jnp array}
        targets = params["targets"]
        x_lin   = _norm_inputs(t, S, exog_t, data["stats"])

        # cp
        if pin_cp or (spec_cp is None):
            cp = _interp_cp(t, data)
        else:
            eta = _eta_for_target_jax(targets["cp"], x_lin, spec_cp)
            cp  = _apply_link_jax(eta, spec_cp.link)

        # alphas
        a_list = []
        for nm, spec in spec_alpha:
            eta = _eta_for_target_jax(targets[nm], x_lin, spec)
            a_list.append(_apply_link_jax(eta, spec.link))
        alphas = jnp.stack(a_list)

        # binary τ
        tau_list = []
        for j, (nm, spec) in enumerate(spec_tau_binary):
            if pin_taus[j] or spec is None:
                tau_list.append(_interp_tau(t, data, j))
            else:
                eta = _eta_for_target_jax(targets[nm], x_lin, spec)
                tau_list.append(_apply_link_jax(eta, spec.link))
        taus = jnp.stack(tau_list)

        # frac_fu — two un-pinned logits (or one, or zero, per layout)
        fu_loss_data = _interp_tau(t, data, 5)
        fu_new_data  = _interp_tau(t, data, 4)
        if spec_fu_new is not None:
            eta_fu_new = _eta_for_target_jax(targets["frac_fu_new"], x_lin, spec_fu_new)
        else:
            eta_fu_new = jnp.array(0.0)
        if spec_fu_loss is not None:
            eta_fu_loss = _eta_for_target_jax(targets["frac_fu_loss"], x_lin, spec_fu_loss)
        else:
            eta_fu_loss = jnp.array(0.0)
        frac_fu = _simplex_value(eta_fu_new, eta_fu_loss, fu_kind,
                                  fu_loss_data, fu_new_data)

        # frac_eu
        eu_loss_data = _interp_tau(t, data, 7)
        eu_new_data  = _interp_tau(t, data, 6)
        if spec_eu_new is not None:
            eta_eu_new = _eta_for_target_jax(targets["frac_eu_new"], x_lin, spec_eu_new)
        else:
            eta_eu_new = jnp.array(0.0)
        if spec_eu_loss is not None:
            eta_eu_loss = _eta_for_target_jax(targets["frac_eu_loss"], x_lin, spec_eu_loss)
        else:
            eta_eu_loss = jnp.array(0.0)
        frac_eu = _simplex_value(eta_eu_new, eta_eu_loss, eu_kind,
                                  eu_loss_data, eu_new_data)

        # f_cohort — softmax of the standalone logits leaf.
        f_cohort = jax.nn.softmax(params["f_cohort_logits"])

        # `raw_norm` — kept for API parity with v5's nn_eval (used by some
        # diagnostics).  We compute it as the mean squared linear-predictor
        # magnitude across learned targets — close in spirit to the NN's
        # `mean(raw**2)`.
        eta_pool = []
        for nm, t_coef in targets.items():
            eta_pool.append(jnp.sum(t_coef["coef_lin"] ** 2))
            for cs in t_coef["coef_smooth"]:
                eta_pool.append(jnp.sum(cs ** 2))
        raw_norm = (sum(eta_pool) / max(len(eta_pool), 1)) if eta_pool else jnp.array(0.0)

        return dict(
            cp=cp, alphas=alphas, taus=taus,
            frac_fu=frac_fu, frac_eu=frac_eu, f_cohort=f_cohort,
            raw_norm=raw_norm,
        )

    return baseline_eval


def coeffs_to_jax_pytree(coeffs: Dict[str, TargetCoefficients],
                          f_cohort_logits: Optional[np.ndarray] = None,
                          ) -> Dict[str, Any]:
    """Convert the Stage-A numpy result to the JAX pytree consumed by
    ``baseline_eval``.  ``f_cohort_logits`` defaults to the IC-COHORT-FRACS
    prior since the baseline has no Stage-A signal for f_cohort.
    """
    if f_cohort_logits is None:
        f_cohort_logits = _f_cohort_default_logits()
    return dict(
        targets={nm: c.to_jax() for nm, c in coeffs.items()},
        f_cohort_logits=jnp.asarray(f_cohort_logits, dtype=jnp.float64),
    )


def pytree_to_coeffs(params: Dict[str, Any]
                      ) -> Tuple[Dict[str, TargetCoefficients], np.ndarray]:
    """Inverse of ``coeffs_to_jax_pytree`` — convert back to numpy for
    inspection / saving after Stage B has moved the coefficients.
    """
    coeffs: Dict[str, TargetCoefficients] = {}
    for nm, td in params["targets"].items():
        coef_smooth = [np.asarray(c) for c in td.get("coef_smooth", [])]
        coeffs[nm] = TargetCoefficients(
            intercept=np.asarray(td["intercept"]),
            coef_lin=np.asarray(td["coef_lin"]),
            coef_smooth=coef_smooth,
        )
    f_logits = np.asarray(params["f_cohort_logits"])
    return coeffs, f_logits


# ============================================================================
# 8) Optional Stage B — refine through the ODE (anchor-penalised)
# ============================================================================
# Stage B for the regression baseline is *much* simpler than for the NN.
# Because the parameter functions are linear in their coefficients (Elastic
# Net) or linear in a fixed truncated-power basis (GAM), gradients through
# the ODE solver are very well-behaved.  We:
#
#   (a) convert Stage-A coefficients to a JAX pytree,
#   (b) integrate the augmented ODE forward,
#   (c) compute the same Stage B loss the v5 NN uses on stocks / flows /
#       α / τ / cp,
#   (d) ADD an anchor penalty ``λ_anchor ‖θ − θ̂_A‖²`` so we do not drift back
#       to the local optima that the Stage A regression already rejected.
#
# Anchor strength controls the "trust region" — small λ allows large moves
# (closer to a full re-fit on the ODE), large λ keeps Stage B as a gentle
# correction on top of Stage A.

def _make_anchor_norm(anchor_pytree):
    """Returns a function ``f(params) -> scalar`` computing ‖p − anchor‖²."""
    def _norm(p):
        ssq = 0.0
        for nm, t in p["targets"].items():
            a = anchor_pytree["targets"][nm]
            ssq = ssq + jnp.sum((t["intercept"] - a["intercept"]) ** 2)
            ssq = ssq + jnp.sum((t["coef_lin"]  - a["coef_lin"])  ** 2)
            for c_t, c_a in zip(t["coef_smooth"], a["coef_smooth"]):
                ssq = ssq + jnp.sum((c_t - c_a) ** 2)
        ssq = ssq + jnp.sum((p["f_cohort_logits"]
                              - anchor_pytree["f_cohort_logits"]) ** 2)
        return ssq
    return _norm


def _build_stageB_loss(*, baseline_eval, integrate_aug, layout,
                        flow_obs_to_pred_idx, anchor_pytree,
                        w_S, w_F, w_alpha, w_tau, w_cp,
                        w_anchor):
    """Through-ODE loss for the regression baseline.

    Closely mirrors the v5 Stage B loss but stays self-contained — we only
    need stocks-log-MSE + flows-log-MSE on supervised columns + the same
    α / τ / cp residuals at observed years, plus the anchor penalty.
    """
    anchor_norm = _make_anchor_norm(anchor_pytree)

    @jax.jit
    def loss_fn(params, data, S0):
        years = data["years"]
        S_pred, F_int, _ = integrate_aug(params, data, S0, years)
        S_obs = data["stocks_obs"]

        # ------------------------------------------------------------------
        # NaN-safe masking pattern.  The data dict's *_obs arrays use NaN
        # to mark missing observations (e.g. alpha_obs[0] is all-NaN by
        # construction because alphas live on midpoints).  Naively writing
        #     L = sum(where(mask, residual**2, 0)) / count(mask)
        # gives a clean forward value (mask zeros the NaN entries) but
        # produces NaN GRADIENTS via the well-known JAX "where + NaN in the
        # unused branch" pitfall: even though the false-branch (0.0) is
        # selected at masked positions, JAX still differentiates the
        # true-branch expression there, and `mask * d(NaN_expr)/dθ`
        # evaluates to `0 * NaN = NaN`, which then poisons the gradient.
        #
        # The fix is to replace NaN entries with a safe finite dummy value
        # BEFORE building the residual.  The mask still zeroes their
        # contribution forward, and the gradient is now clean because the
        # true-branch expression never sees a NaN.
        # ------------------------------------------------------------------

        # log-MSE on stocks (per-stock floor std handled by `stats`).
        mask_S = jnp.isfinite(S_obs)
        S_obs_safe = jnp.where(mask_S, S_obs, 1.0)
        log_S_pred = jnp.log(jnp.maximum(S_pred,     1e-6))
        log_S_obs  = jnp.log(jnp.maximum(S_obs_safe, 1e-6))
        S_log_std  = data["stats"]["S_log_std"]
        S_resid2   = ((log_S_pred - log_S_obs) / S_log_std) ** 2
        L_S = jnp.sum(jnp.where(mask_S, S_resid2, 0.0)) \
              / jnp.maximum(jnp.sum(mask_S.astype(S_resid2.dtype)), 1.0)

        # log-MSE on yearly flow integrals
        F_obs = data["flows_obs"]
        F_pred = F_int[:, jnp.asarray(flow_obs_to_pred_idx)]
        mask_F = jnp.isfinite(F_obs)
        F_obs_safe = jnp.where(mask_F, F_obs, 1.0)
        log_Fp = jnp.log(jnp.maximum(F_pred,      1e-6))
        log_Fo = jnp.log(jnp.maximum(F_obs_safe,  1e-6))
        F_log_std = data["stats"]["flow_log_std"]
        F_resid2  = ((log_Fp - log_Fo) / F_log_std) ** 2
        L_F = jnp.sum(jnp.where(mask_F, F_resid2, 0.0)) \
              / jnp.maximum(jnp.sum(mask_F.astype(F_resid2.dtype)), 1.0)

        # α / τ / cp at observed stocks (parameter-ID quality)
        def _eval_at(t, S):
            ex = exog_fn(t, data["exog_times"], data["exog_values"])
            return baseline_eval(params, t, S, ex, data)
        outs = jax.vmap(_eval_at)(years, S_obs)

        # α loss (log-MSE)
        a_obs = data["alpha_obs"]
        a_pre = outs["alphas"]
        mask_a = jnp.isfinite(a_obs)
        a_obs_safe = jnp.where(mask_a, a_obs, 1.0)
        a_resid2 = (jnp.log(jnp.maximum(a_pre,      1e-12))
                    - jnp.log(jnp.maximum(a_obs_safe, 1e-12))) ** 2
        L_a = jnp.sum(jnp.where(mask_a, a_resid2, 0.0)) \
              / jnp.maximum(jnp.sum(mask_a.astype(a_resid2.dtype)), 1.0)

        # τ loss (MSE on the 8-vector, masking pinned columns via tau_sup_std)
        t_obs = data["tau_sup_obs"]
        tau_pred_8 = jnp.concatenate([
            outs["taus"],
            jnp.stack([outs["frac_fu"][:, 0], outs["frac_fu"][:, 1],
                       outs["frac_eu"][:, 0], outs["frac_eu"][:, 1]], axis=1),
        ], axis=1)
        mask_t = jnp.isfinite(t_obs)
        t_obs_safe = jnp.where(mask_t, t_obs, 0.0)
        t_resid2 = ((tau_pred_8 - t_obs_safe) / data["stats"]["tau_sup_std"]) ** 2
        L_t = jnp.sum(jnp.where(mask_t, t_resid2, 0.0)) \
              / jnp.maximum(jnp.sum(mask_t.astype(t_resid2.dtype)), 1.0)

        # cp loss
        cp_obs = data["cp_obs"]
        cp_pre = outs["cp"]
        mask_cp = jnp.isfinite(cp_obs)
        cp_obs_safe = jnp.where(mask_cp, cp_obs, 1.0)
        cp_resid2 = ((jnp.log(jnp.maximum(cp_pre,      1e-6))
                      - jnp.log(jnp.maximum(cp_obs_safe, 1e-6)))
                     / data["stats"]["cp_log_std"]) ** 2
        L_cp = jnp.sum(jnp.where(mask_cp, cp_resid2, 0.0)) \
               / jnp.maximum(jnp.sum(mask_cp.astype(cp_resid2.dtype)), 1.0)

        # Anchor penalty
        L_anchor = anchor_norm(params)

        total = (w_S * L_S + w_F * L_F + w_alpha * L_a
                 + w_tau * L_t + w_cp * L_cp + w_anchor * L_anchor)
        aux = dict(L_S=L_S, L_F=L_F, L_alpha=L_a, L_tau=L_t,
                   L_cp=L_cp, L_anchor=L_anchor)
        return total, aux
    return loss_fn


def _tree_has_nonfinite(tree) -> bool:
    """True iff any leaf contains NaN/inf.  Cheap numpy check."""
    for leaf in jax.tree_util.tree_leaves(tree):
        arr = np.asarray(leaf)
        if not np.all(np.isfinite(arr)):
            return True
    return False


def refine_stage_B(params_A, *, baseline_eval, integrate_aug, data_trainval,
                    layout, flow_obs_to_pred_idx,
                    n_steps: int = 200, lr: float = 1e-4,
                    w_S: float = 2.0, w_F: float = 0.5,
                    w_alpha: float = 1.0, w_tau: float = 1.0, w_cp: float = 1.0,
                    w_anchor: float = 1e-2,
                    grad_clip: float = 1.0,
                    verbose: bool = True) -> Tuple[Dict[str, Any], List[float]]:
    """Anchor-penalised through-ODE refinement.

    Stage B for the regression baseline is best-effort: when Stage A's local
    fit produces a globally inconsistent trajectory (large L_S at step 0),
    the ODE adjoint can produce NaN gradients near integrator stability
    limits.  We guard against this in three layers:

      1.  Gradient norm clipping at ``grad_clip`` (default 1.0).
      2.  NaN/Inf detection on the gradient pytree — if any leaf has
          non-finite values, the step is *skipped* and the optimiser
          state is preserved.
      3.  The whole Stage B can be wrapped in a try/except by the caller
          (``train_baseline_model`` does this).

    A skipped step does not advance ``step``-count progress logging, so
    you'll see the actual ratio of usable steps in the output.

    Returns the refined params pytree (= params_A if every step was
    skipped) and a list of loss values for the steps that succeeded.
    """
    loss_fn = _build_stageB_loss(
        baseline_eval=baseline_eval, integrate_aug=integrate_aug,
        layout=layout, flow_obs_to_pred_idx=flow_obs_to_pred_idx,
        anchor_pytree=jax.tree_util.tree_map(lambda x: x, params_A),
        w_S=w_S, w_F=w_F, w_alpha=w_alpha, w_tau=w_tau, w_cp=w_cp,
        w_anchor=w_anchor,
    )

    opt = optax.chain(
        optax.clip_by_global_norm(float(grad_clip)) if grad_clip > 0 else optax.identity(),
        optax.adam(lr),
    )
    opt_state = opt.init(params_A)
    params = params_A
    S0 = data_trainval["stocks_obs"][0]
    history: List[float] = []
    n_skipped = 0
    val0, aux0 = loss_fn(params_A, data_trainval, S0)   # outside the loop
    print({k: float(v) for k, v in aux0.items()})
    for step in range(n_steps):
        try:
            (val, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params, data_trainval, S0
            )
        except Exception as e:
            # Integrator (forward or adjoint) failed.  Skip this step.
            n_skipped += 1
            if verbose:
                print(f"  Stage B step {step:4d}  [skipped — integrator failure: "
                      f"{type(e).__name__}]")
            continue

        # Guard against NaN/Inf gradients (typical near integrator stability
        # boundaries).  We also guard against a NaN loss.
        if not np.isfinite(float(val)) or _tree_has_nonfinite(grads):
            n_skipped += 1
            if verbose:
                print(f"  Stage B step {step:4d}  [skipped — non-finite "
                      f"loss/gradient]")
            continue

        updates, opt_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # One last guard: refuse to apply an update that produced non-finite
        # parameters (rare, but possible with extreme grads + Adam scaling).
        if _tree_has_nonfinite(new_params):
            n_skipped += 1
            if verbose:
                print(f"  Stage B step {step:4d}  [skipped — update produced "
                      f"non-finite params]")
            continue
        params = new_params
        history.append(float(val))

        if verbose and (step % max(n_steps // 10, 1) == 0 or step == n_steps - 1):
            print(f"  Stage B step {step:4d}  loss={float(val):.4e}  "
                  f"(S={float(aux['L_S']):.2e} F={float(aux['L_F']):.2e} "
                  f"α={float(aux['L_alpha']):.2e} τ={float(aux['L_tau']):.2e} "
                  f"anchor={float(aux['L_anchor']):.2e})")
    if verbose and n_skipped > 0:
        print(f"  Stage B summary: {len(history)}/{n_steps} steps applied "
              f"({n_skipped} skipped due to numerical issues)")
    return params, history


# ============================================================================
# 9) BaselineFitResult — subclass of v5's FitResult
# ============================================================================
class BaselineFitResult(FitResult):
    """Drop-in subclass that swaps NN-specific phrasing for regression-specific
    phrasing in plot titles and adds an ``estimator_summary`` method.

    All v5 plot / diagnose / summary methods are inherited unchanged; the
    integrator and the parameter-function evaluator are the only pieces that
    differ behind the scenes.
    """

    def __init__(self, *, name, cfg, params, params_A, layout,
                 data_train, data_val, data_trainval, data_test, data_all,
                 integrate_aug, baseline_eval, baseline_eval_sup,
                 flow_obs_to_pred_idx, do_stage_B,
                 static_params: BaselineParamsStatic,
                 stageA_diagnostics: Optional[Dict[str, Dict[str, Any]]] = None,
                 stageB_history: Optional[List[float]] = None):
        # Call v5's FitResult constructor with the right argument names.
        # FitResult expects `nn_eval` / `nn_eval_sup`; in the baseline these
        # ARE the regression evaluators — semantically identical signatures.
        super().__init__(
            name=name, cfg=cfg,
            params=params, params_A=params_A, layout=layout,
            data_train=data_train, data_val=data_val, data_trainval=data_trainval,
            data_test=data_test, data_all=data_all,
            integrate_aug=integrate_aug,
            nn_eval=baseline_eval, nn_eval_sup=baseline_eval_sup,
            flow_obs_to_pred_idx=flow_obs_to_pred_idx,
            do_stage_B=do_stage_B,
        )
        self.static_params = static_params
        self.stageA_diagnostics = stageA_diagnostics or {}
        self.stageB_history = list(stageB_history) if stageB_history else []

    # ------------------------------------------------------------------
    # Estimator summary — what kind / how many basis terms per target
    # ------------------------------------------------------------------
    def estimator_summary(self, *, verbose: bool = True) -> Dict[str, Dict[str, Any]]:
        """Per-target summary of estimator kind, regularisation choices,
        in-sample R² on the linear-predictor scale, and (for Elastic Net)
        the number of non-zero linear coefficients.

        Returns
        -------
        dict name → dict
        """
        if verbose:
            print(f"\n[estimator_summary {self.name}] "
                  f"per-target Stage-A regression diagnostics:")
            hdr = (f"    {'target':18s}  {'kind':6s}  {'link':9s}  "
                   f"{'n_obs':>5s}  {'R²(η)':>7s}  {'nnz_lin':>8s}  {'smooth':>10s}")
            print(hdr)
        out: Dict[str, Dict[str, Any]] = {}
        for nm, info in self.stageA_diagnostics.items():
            r2 = info.get("in_sample_r2_eta")
            r2_str = _fmt_pct(100.0 * r2, fmt="{:+.2f}%") if r2 is not None else "—"
            spec = self.static_params.target_specs.get(nm)
            smooth_str = (
                ",".join(str(s.n_basis) for s in spec.smooth_specs)
                if (spec and spec.smooth_specs) else "—"
            )
            line = (f"    {nm:18s}  {info.get('kind','?'):6s}  "
                    f"{info.get('link','?'):9s}  "
                    f"{info.get('n_obs','?'):>5}  "
                    f"{r2_str:>7s}  "
                    f"{info.get('nonzero_lin','—'):>8}  "
                    f"{smooth_str:>10s}")
            if verbose:
                print(line)
            out[nm] = info
        return out

    # ------------------------------------------------------------------
    # Override `plot_alphas`/`plot_taus` legend labels so they read
    # "baseline" rather than "NN" — purely cosmetic.
    # ------------------------------------------------------------------
    def plot_alphas(self, save_dir=None, *, include_pinned=False):
        # Delegate the heavy lifting to the parent.  We just re-do the loop
        # with relabelled legend entries.  Easier than overriding selectively.
        if not HAS_MPL:
            print("matplotlib not available; skipping plot.")
            return []
        years = np.asarray(self.data_all["years"])
        alpha_o = np.asarray(self.data_all["alpha_obs"])
        preds = {m: self._predictions(m) for m in self._stages()}
        out_paths = []
        for k, nm in enumerate(ALPHA_NAMES):
            fig, ax = plt.subplots(figsize=(7, 3.6))
            self._shade_splits(ax)
            ax.plot(years, alpha_o[:, k], "ko-", label="empirical α", lw=1.0, ms=3.5)
            for m, p in preds.items():
                ax.plot(years, p["alphas_at_obs"][:, k],
                        "-" if m == "A" else "--",
                        label=f"baseline_{m} @ S_obs", lw=1.5)
            ax.set_yscale("log")
            ax.set_xlabel("Year"); ax.set_ylabel("rate (1/yr)")
            ax.set_title(f"{self.name}  {nm}")
            ax.grid(True, alpha=0.4); ax.legend(loc="best", fontsize=8)
            plt.tight_layout()
            out_paths.append(self._save_or_show(fig, save_dir,
                                                f"{self.name}_alpha_{nm}.png"))
        return [p for p in out_paths if p is not None]

    def plot_taus(self, save_dir=None, *, include_pinned=False):
        if not HAS_MPL:
            print("matplotlib not available; skipping plot.")
            return []
        years = np.asarray(self.data_all["years"])
        tau_o = np.asarray(self.data_all["tau_sup_obs"])
        preds = {m: self._predictions(m) for m in self._stages()}
        pin_flags = self._pinned_flags_per_tau_sup()

        def _tau_full(p):
            return np.concatenate([
                p["taus_at_obs"],
                np.stack([p["frac_fu_at_obs"][:, 0], p["frac_fu_at_obs"][:, 1],
                          p["frac_eu_at_obs"][:, 0], p["frac_eu_at_obs"][:, 1]],
                         axis=1),
            ], axis=1)

        tau_full = {m: _tau_full(p) for m, p in preds.items()}
        out_paths = []
        for j, nm in enumerate(TAU_SUP_NAMES):
            if pin_flags[j] and not include_pinned:
                continue
            fig, ax = plt.subplots(figsize=(7, 3.6))
            self._shade_splits(ax)
            ax.plot(years, tau_o[:, j], "ko-", label="observed", lw=1.0, ms=3.5)
            tag = " [pinned]" if pin_flags[j] else ""
            for m, T in tau_full.items():
                ax.plot(years, T[:, j], "-" if m == "A" else "--",
                        label=f"baseline_{m} @ S_obs{tag}", lw=1.5)
            ax.set_xlabel("Year"); ax.set_ylabel("coefficient")
            ax.set_title(f"{self.name}  {nm}{tag}")
            ax.grid(True, alpha=0.4); ax.legend(loc="best", fontsize=8)
            plt.tight_layout()
            out_paths.append(self._save_or_show(fig, save_dir,
                                                f"{self.name}_tau_{nm}.png"))
        return [p for p in out_paths if p is not None]


# ============================================================================
# 10) Top-level orchestrator — train_baseline_model
# ============================================================================
# Mirrors `zinc_colloc_v5.train_model` for the regression baseline.  The
# function signature is intentionally close to v5's: every option that has a
# v5 analogue uses the same default, and any extra baseline-specific option
# is prefixed `baseline_` for clarity.

def train_baseline_model(
    xlsx_path: str,
    *,
    seed: int = 0,

    # ---------- Input feature controls (same defaults as v5) ----------
    use_time_input: bool = False,
    use_stock_input: bool = True,
    stock_norm_mode: str = "tanh_log",
    stock_log_scale: float = 1.0,
    stock_ref_mode: str = "mean",

    # ---------- Pin flags (same names as v5) ----------
    learn_cp: bool = False,
    pin_tau_ref: bool = False,
    pin_tau_waelz: bool = False,
    pin_tau_olds: bool = False,
    pin_tau_diss: bool = False,
    pin_frac_fu_loss: bool = False,
    pin_frac_eu_loss: bool = False,
    pin_frac_fu_new: bool = False,
    pin_frac_eu_new: bool = False,

    # ---------- Exogenous features (same defaults as v5) ----------
    extra_exog_cols=None,
    exog_log1p: bool = True,
    exog_detrend: bool = False,
    exog_detrend_kind: str = "linear",
    exog_feature_orders: Tuple[int, ...] = (0, 1),
    exog_diff_pad: str = "edge",
    exog_pca_components: Optional[int] = None,

    # ---------- Split (same as v5) ----------
    trainval_frac: float = 0.7,
    val_frac: float = 0.2,
    split_indices: Optional[Dict[str, int]] = None,

    # ---------- Stage A — baseline-specific ----------
    regressor_kinds: Union[str, Dict[str, str]] = "enet",
    gam_smooth_features: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    enet_l1_ratios: Tuple[float, ...] = (0.1, 0.5, 0.9),
    n_cv_splits: int = 5,

    # ---------- Stage B (optional through-ODE refinement) ----------
    do_stage_B: bool = False,
    stageB_steps: int = 200,
    stageB_lr: float = 1e-4,
    stageB_w_S: float = 2.0,
    stageB_w_F: float = 0.5,
    stageB_w_alpha: float = 1.0,
    stageB_w_tau: float = 1.0,
    stageB_w_cp: float = 1.0,
    stageB_w_anchor: float = 1e-2,
    stageB_grad_clip: float = 1.0,

    # ---------- ODE backend (same as v5) ----------
    integrator: Optional[str] = None,
    stock_term_weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0),

    verbose: bool = True,
):
    """Train the regression baseline end-to-end.

    Returns a tuple matching ``FitResult.__init__``'s required arguments:
        (params, params_A, layout, dt, dv, dtv, dte, da,
         integrate_aug, baseline_eval, baseline_eval_sup,
         flow_obs_to_pred_idx, static_params, stageA_diagnostics,
         stageB_history, do_stage_B_flag)
    """
    if integrator is None:
        integrator = "diffrax" if HAS_DIFFRAX else "odeint"

    # ---- 0) NN-equivalent layout (which slots are learned vs pinned) ----
    layout = build_nn_layout(
        learn_cp=learn_cp,
        pin_tau_ref=pin_tau_ref, pin_tau_waelz=pin_tau_waelz,
        pin_tau_olds=pin_tau_olds, pin_tau_diss=pin_tau_diss,
        pin_frac_fu_loss=pin_frac_fu_loss, pin_frac_eu_loss=pin_frac_eu_loss,
        pin_frac_fu_new=pin_frac_fu_new, pin_frac_eu_new=pin_frac_eu_new,
    )

    # ---- 1) Load raw data ----
    data_np = load_zinc_data(xlsx_path, extra_exog_cols=extra_exog_cols)
    years_all = data_np["years"]
    S_all     = data_np["stocks_obs"]
    cp_all    = data_np["cp_obs"]
    alpha_all = data_np["alpha_obs"]
    tau_all   = data_np["tau_sup_obs"]
    exog_full_t = data_np["exog_times_full"]
    exog_full_v = data_np["exog_values_full"]
    T = len(years_all)

    # ---- 2) Split (mirrors v5 logic exactly) ----
    if split_indices is None:
        if not (0.0 < trainval_frac < 1.0):
            raise ValueError("trainval_frac must be in (0,1).")
        n_trainval = min(max(int(np.floor(trainval_frac * T)), 3), T - 1)
        n_tr = n_trainval
        n_val = max(3, int(np.ceil(val_frac * n_tr)))
        cut   = max(n_tr - n_val, 2)
        test_ic_idx = max(n_trainval - 1, 0)
    else:
        train_end_idx  = int(split_indices["train_end"])
        val_end_idx    = int(split_indices["val_end"])
        test_start_idx = int(split_indices.get("test_start", val_end_idx + 1))
        if not (0 <= train_end_idx < val_end_idx < T):
            raise ValueError(
                f"split_indices must satisfy 0 <= train_end({train_end_idx}) "
                f"< val_end({val_end_idx}) < T({T})."
            )
        if test_start_idx <= val_end_idx or test_start_idx > T - 1:
            raise ValueError(
                f"split_indices['test_start']={test_start_idx} invalid."
            )
        n_trainval = val_end_idx + 1
        cut        = train_end_idx + 1
        test_ic_idx = max(test_start_idx - 1, 0)

    years_train = years_all[:cut]
    S_train     = S_all[:cut]
    cp_train    = cp_all[:cut]
    alpha_train = alpha_all[:cut]
    tau_train   = tau_all[:cut]
    years_val   = years_all[cut - 1: n_trainval]
    years_test  = years_all[test_ic_idx:]

    if verbose:
        print(f"[baseline split] T={T}  "
              f"train={cut} ({int(years_train[0])}–{int(years_train[-1])})  "
              f"val={len(years_val)} ({int(years_val[0])}–{int(years_val[-1])})  "
              f"test={len(years_test)} ({int(years_test[0])}–{int(years_test[-1])})")

    # ---- 3) Exogenous preprocessing (fit on training core) ----
    exog_proc = preprocess_exog(
        years_all, data_np["exog_values"], years_train,
        do_log1p=exog_log1p, do_detrend=exog_detrend,
        detrend_kind=exog_detrend_kind,
        feature_orders=exog_feature_orders, diff_pad=exog_diff_pad,
        years_source=exog_full_t, exog_values_source=exog_full_v,
    )
    if exog_pca_components is not None:
        k = int(exog_pca_components)
        if 0 < k < int(exog_proc.shape[1]):
            Xtr = np.asarray(exog_proc[:cut], dtype=float)
            mu  = Xtr.mean(axis=0, keepdims=True)
            _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
            W = Vt[:k].T
            exog_proc = (np.asarray(exog_proc, dtype=float) - mu) @ W
            if verbose:
                print(f"[baseline exog PCA] compressed to {k} components")

    # ---- 4) Statistics on training core (mirrors v5 stats) ----
    t_mean = float(years_train.mean())
    t_std  = max(float(years_train.std()), 1.0)
    S_mean = S_train.mean(axis=0)
    S_std  = np.maximum(S_train.std(axis=0),  1.0)
    exog_train = exog_proc[:cut]
    exog_mean  = exog_train.mean(axis=0)
    exog_std   = np.maximum(exog_train.std(axis=0), 1.0)
    S_target_std = np.maximum(S_train.std(axis=0), 1.0)

    alpha_train_finite = np.where(np.isfinite(alpha_train), alpha_train, np.nan)
    log_alpha_train = np.log(np.maximum(alpha_train_finite, 1e-12))
    alpha_log_mean = np.nan_to_num(np.nanmean(log_alpha_train, axis=0), nan=0.0)
    alpha_log_std  = np.maximum(np.nan_to_num(np.nanstd(log_alpha_train, axis=0),
                                              nan=1.0, posinf=1.0, neginf=1.0), 0.05)
    alpha_scale = np.maximum(np.exp(alpha_log_mean).astype(float), 1e-6)

    tau_train_finite = np.where(np.isfinite(tau_train), tau_train, np.nan)
    tau_sup_std = np.maximum(np.nan_to_num(np.nanstd(tau_train_finite, axis=0),
                                           nan=0.10, posinf=0.10, neginf=0.10), 0.10)

    cp_train_finite = cp_train[np.isfinite(cp_train)]
    cp_scale = float(cp_train_finite.mean()) if cp_train_finite.size > 0 else 1.0
    if not np.isfinite(cp_scale) or cp_scale <= 0.0:
        cp_scale = 1.0
    cp_log_std = (float(np.std(np.log(np.maximum(cp_train_finite, 1e-12))))
                  if cp_train_finite.size > 1 else 1.0)
    cp_log_std = max(cp_log_std, 0.05)

    if stock_ref_mode == "median":
        S_ref = np.maximum(np.median(S_train, axis=0), 1.0)
    elif stock_ref_mode == "geomean":
        S_ref = np.maximum(np.exp(np.mean(np.log(np.maximum(S_train, 1.0)),
                                          axis=0)), 1.0)
    else:
        S_ref = np.maximum(S_train.mean(axis=0), 1.0)

    stock_norm_mode_s = str(stock_norm_mode).lower().strip()
    _norm_mode_map = {"zscore": 0.0, "log": 1.0, "tanh_log": 2.0}
    if stock_norm_mode_s not in _norm_mode_map:
        raise ValueError(f"stock_norm_mode must be one of {list(_norm_mode_map)}.")

    stock_term_weights_arr = np.asarray(stock_term_weights, dtype=float)
    if stock_term_weights_arr.shape != (N_STOCKS,):
        raise ValueError(f"stock_term_weights must have shape ({N_STOCKS},).")

    flows_all = data_np["flows_obs"]
    flows_train = flows_all[: max(cut - 1, 1)]
    log_F = np.log(np.maximum(flows_train, 1e-6))
    log_F = np.where(np.isfinite(log_F), log_F, np.nan)
    flow_log_std = np.maximum(np.nan_to_num(np.nanstd(log_F, axis=0),
                                            nan=0.5, posinf=0.5, neginf=0.5), 0.05)
    log_S_train = np.log(np.maximum(S_train, 1e-6))
    log_S_train = np.where(np.isfinite(log_S_train), log_S_train, np.nan)
    S_log_std = np.maximum(np.nan_to_num(np.nanstd(log_S_train, axis=0),
                                         nan=0.5, posinf=0.5, neginf=0.5), 0.05)

    # NumPy + JAX twin dicts.  stats_np for offline Stage A; stats_j for ODE.
    stats_np = dict(
        t_mean=t_mean, t_std=t_std,
        S_mean=S_mean, S_std=S_std,
        exog_mean=exog_mean, exog_std=exog_std,
        S_target_std=S_target_std,
        alpha_scale=alpha_scale.astype(float),
        alpha_log_std=alpha_log_std.astype(float),
        tau_sup_std=tau_sup_std.astype(float),
        cp_scale=np.array(cp_scale, dtype=float),
        cp_log_std=np.array(cp_log_std, dtype=float),
        flow_log_std=flow_log_std.astype(float),
        S_log_std=S_log_std.astype(float),
        use_time_input=np.array(1.0 if use_time_input else 0.0, dtype=float),
        use_stock_input=np.array(1.0 if use_stock_input else 0.0, dtype=float),
        stock_norm_mode=np.array(_norm_mode_map[stock_norm_mode_s], dtype=float),
        S_ref=S_ref.astype(float),
        S_log_scale=np.array(float(stock_log_scale), dtype=float),
        stock_term_weights=stock_term_weights_arr.astype(float),
    )
    stats_j = {k: jnp.array(v) for k, v in stats_np.items()}

    # ---- 5) Pack JAX-side data dicts for each split (identical to v5) ----
    flow_obs_to_pred_idx_np = np.asarray(data_np["flow_obs_to_pred_idx"], dtype=np.int32)

    def _pack(y_idx0, y_idx1):
        sl      = slice(y_idx0, y_idx1)
        flow_sl = slice(y_idx0, max(y_idx1 - 1, y_idx0))
        return dict(
            years=jnp.array(years_all[sl]),
            stocks_obs=jnp.array(S_all[sl]),
            cp_obs=jnp.array(cp_all[sl]),
            alpha_obs=jnp.array(alpha_all[sl]),
            tau_sup_obs=jnp.array(tau_all[sl]),
            flows_obs=jnp.array(flows_all[flow_sl]),
            exog_times=jnp.array(years_all[sl]),
            exog_values=jnp.array(exog_proc[sl]),
            stats=stats_j,
        )
    data_train     = _pack(0,            cut)
    data_val       = _pack(cut - 1,      n_trainval)
    data_trainval  = _pack(0,            n_trainval)
    data_test      = _pack(test_ic_idx,  T)
    data_all       = _pack(0,            T)

    # ---- 6) Build Stage A supervision tables (training split only) ----
    # α supervision (α_obs is year-end, so it aligns with years_all);
    # row 0 is NaN by construction.  τ / cp supervision is integer-year.
    train_idx = slice(0, cut)
    data_train_np = dict(
        # α supervision rows are at integer years 1..cut-1 (row 0 NaN).
        years_alpha=years_all[train_idx],
        S_alpha=S_all[train_idx],
        exog_alpha=exog_proc[train_idx],
        alpha_obs=alpha_all[train_idx],
        # τ / cp on the same integer-year grid
        years_year=years_all[train_idx],
        S_year=S_all[train_idx],
        exog_year=exog_proc[train_idx],
        tau_sup_obs=tau_all[train_idx],
        cp_obs=cp_all[train_idx],
    )

    if verbose:
        print(f"\n[Stage A] fitting regression baseline "
              f"(targets={len(_learned_target_names(layout))}, "
              f"n_features={feature_dim(exog_proc.shape[1])})")

    coeffs_A, static_params = fit_baseline_stage_A(
        layout=layout, data_train_np=data_train_np, stats_np=stats_np,
        regressor_kinds=regressor_kinds,
        gam_smooth_features=gam_smooth_features,
        enet_l1_ratios=enet_l1_ratios,
        n_cv_splits=n_cv_splits, random_state=seed, verbose=verbose,
    )

    # ---- 7) Build the baseline_eval closure + ODE integrator ----
    baseline_eval = make_baseline_eval(layout, static_params)

    # `baseline_eval_sup` mirrors v5's `make_nn_eval_supervised` shape so
    # that FitResult's diagnose / plot methods (which expect the
    # supervised tuple) work without modification.
    def baseline_eval_sup(params, t, S, exog_t, data):
        out = baseline_eval(params, t, S, exog_t, data)
        tau_sup = jnp.concatenate([
            out["taus"],
            jnp.array([out["frac_fu"][0], out["frac_fu"][1],
                       out["frac_eu"][0], out["frac_eu"][1]]),
        ])
        return out["cp"], out["alphas"], tau_sup, out["f_cohort"], out["raw_norm"]

    # DirectAdjoint autodiffs through the discrete solver steps rather than
    # solving the adjoint ODE, which is what keeps Stage B stable when the
    # learned alphas make the Jacobian diagonal large.  `dfx` is None when
    # diffrax is absent, in which case make_integrator raises a clean
    # ImportError for the 'diffrax' backend — don't AttributeError first.
    _adjoint = dfx.DirectAdjoint() if (HAS_DIFFRAX and integrator == "diffrax") else None
    integrate_aug = make_integrator(
        integrator, baseline_eval,
        adjoint=_adjoint,
    )

    # Convert Stage A coefficients to JAX pytree
    params_A = coeffs_to_jax_pytree(coeffs_A)
    params   = params_A

    # ---- 8) Optional Stage B refinement ----
    stageB_history: List[float] = []
    stageB_ran_flag = False
    if do_stage_B:
        if verbose:
            print(f"\n[Stage B] anchor-penalised through-ODE refinement "
                  f"({stageB_steps} steps, lr={stageB_lr:.0e}, "
                  f"w_anchor={stageB_w_anchor:.0e})")
        try:
            params, stageB_history = refine_stage_B(
                params_A,
                baseline_eval=baseline_eval, integrate_aug=integrate_aug,
                data_trainval=data_trainval,
                layout=layout,
                flow_obs_to_pred_idx=flow_obs_to_pred_idx_np,
                n_steps=stageB_steps, lr=stageB_lr,
                w_S=stageB_w_S, w_F=stageB_w_F,
                w_alpha=stageB_w_alpha, w_tau=stageB_w_tau, w_cp=stageB_w_cp,
                w_anchor=stageB_w_anchor,
                grad_clip=stageB_grad_clip,
                verbose=verbose,
            )
            # If refine_stage_B managed to take at least one usable step it
            # genuinely modified params; otherwise treat Stage B as not run.
            stageB_ran_flag = (len(stageB_history) > 0)
            if not stageB_ran_flag:
                if verbose:
                    print("[Stage B] all steps were skipped — falling back to "
                          "Stage A only.")
                    print("  Hints: Stage A's free-run trajectory is too "
                          "ill-conditioned for through-ODE refinement.  "
                          "Try one or more of:")
                    print("   • use 'gam' (not 'enet') for α targets to "
                          "tame extrapolation,")
                    print("   • use 'const' for high-variance taus that "
                          "dominate the L_S gradient (e.g. tau_diss),")
                    print("   • pin nearly-stationary taus to data "
                          "(pin_tau_ref=True, pin_tau_waelz=True, …),")
                    print("   • increase stageB_w_anchor (e.g. 1e-1 or 1.0) "
                          "to keep the refinement closer to Stage A.")
                params = params_A
        except Exception as e:
            # Catch-all: any numerical failure during Stage B falls back to
            # Stage A.  This keeps the run usable even when Stage A's local
            # fit is too poor for through-ODE refinement.
            if verbose:
                print(f"[Stage B] aborted due to {type(e).__name__}: {e}  "
                      "— falling back to Stage A only.")
            params = params_A
            stageB_history = []
            stageB_ran_flag = False

    return (
        params, params_A, layout,
        data_train, data_val, data_trainval, data_test, data_all,
        integrate_aug, baseline_eval, baseline_eval_sup,
        flow_obs_to_pred_idx_np,
        static_params, static_params.cv_diagnostics, stageB_history,
        bool(stageB_ran_flag),
    )


# ============================================================================
# 11) run_baseline — high-level entry point analogous to zc.run
# ============================================================================
BASELINE_DEFAULT_CONFIG = dict(
    xlsx_path="zinc_dataset.xlsx",
    seed=0,

    # Same input controls as v5
    use_time_input=False,
    use_stock_input=True,
    stock_norm_mode="tanh_log",
    stock_log_scale=1.0,
    stock_ref_mode="mean",

    # Same pin flags
    learn_cp=False,
    pin_tau_ref=False, pin_tau_waelz=False,
    pin_tau_olds=False, pin_tau_diss=False,
    pin_frac_fu_loss=False, pin_frac_eu_loss=False,
    pin_frac_fu_new=False,  pin_frac_eu_new=False,

    extra_exog_cols=[
        "Precious metal index",
        "Metal real index excl iron",
        "World Stock Market Capitalisation (% of GDP)",
        "China Total Manufacturing Output",
        "Population",
    ],
    exog_log1p=True, exog_detrend=False,
    exog_feature_orders=(0, 1),
    exog_diff_pad="edge",
    exog_pca_components=None,

    trainval_frac=0.7, val_frac=0.2,

    # Baseline-specific
    regressor_kinds="enet",
    gam_smooth_features=None,
    enet_l1_ratios=(0.1, 0.5, 0.9),
    n_cv_splits=5,

    # Stage B off by default — Stage A is the headline result of the baseline.
    do_stage_B=False,
    stageB_steps=200, stageB_lr=1e-4,
    stageB_w_S=2.0, stageB_w_F=0.5,
    stageB_w_alpha=1.0, stageB_w_tau=1.0, stageB_w_cp=1.0,
    stageB_w_anchor=1e-2,
    stageB_grad_clip=1.0,

    integrator=None,
    stock_term_weights=(1.0, 1.0, 1.0, 1.0),
    verbose=True,
)


def run_baseline(name: str = "baseline_default", **overrides) -> BaselineFitResult:
    """High-level entry point — analogue of ``zinc_colloc_v5.run``.

    Example
    -------
    >>> import zinc_baseline as zb
    >>> fit_bl = zb.run_baseline("enet_default")
    >>> fit_bl.summary()
    >>> fit_bl.estimator_summary()
    >>> fit_bl.plot_all(save_dir="./plots/baseline")

    A GAM-flavoured run mixing estimators::

    >>> fit_bl = zb.run_baseline(
    ...     "gam_mix",
    ...     regressor_kinds={"alpha_dr": "gam", "alpha_win": "gam",
    ...                      "tau_waelz": "const"},  # rest default to enet
    ...     gam_smooth_features={
    ...         "alpha_dr":  [(0, 4)],   # smooth on t (if use_time_input=True)
    ...         "alpha_win": [(5, 4)],   # smooth on first exog (price)
    ...     },
    ...     use_time_input=True,
    ... )

    Side-by-side comparison with the v5 PINN::

    >>> import zinc_colloc_v5 as zc
    >>> fit_nn = zc.run("v5_default")
    >>> fit_bl.diagnose("testrun")
    >>> fit_nn.diagnose("testrun")
    """
    cfg = dict(BASELINE_DEFAULT_CONFIG)
    cfg.update(overrides)
    print(f"\n=== zinc_baseline run: {name} ===")
    (params, params_A, layout,
     dt, dv, dtv, dte, da,
     integrate_aug, baseline_eval, baseline_eval_sup,
     flow_obs_to_pred_idx,
     static_params, stageA_diag, stageB_hist,
     do_stage_B_flag) = train_baseline_model(**cfg)

    fit = BaselineFitResult(
        name=name, cfg=cfg,
        params=params, params_A=params_A, layout=layout,
        data_train=dt, data_val=dv, data_trainval=dtv,
        data_test=dte, data_all=da,
        integrate_aug=integrate_aug,
        baseline_eval=baseline_eval,
        baseline_eval_sup=baseline_eval_sup,
        flow_obs_to_pred_idx=flow_obs_to_pred_idx,
        do_stage_B=do_stage_B_flag,
        static_params=static_params,
        stageA_diagnostics=stageA_diag,
        stageB_history=stageB_hist,
    )
    fit.summary()
    return fit


__all__ = [
    # High-level API
    "run_baseline", "BaselineFitResult",
    "train_baseline_model",
    "BASELINE_DEFAULT_CONFIG",
    # Building blocks (for advanced use / unit tests)
    "TARGET_INVENTORY", "TARGET_INVENTORY_BY_NAME",
    "_learned_target_names",
    "feature_dim", "_norm_inputs_np", "_build_feature_matrix_np",
    "SplineBasisSpec", "make_quantile_knots",
    "tpower_basis_np", "tpower_basis_jax",
    "TargetCoefficients", "TargetSpec", "BaselineParamsStatic",
    "fit_target_const", "fit_target_enet", "fit_target_gam",
    "fit_baseline_stage_A",
    "make_baseline_eval",
    "coeffs_to_jax_pytree", "pytree_to_coeffs",
    "refine_stage_B",
]
