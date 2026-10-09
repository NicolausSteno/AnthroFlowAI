"""
zinc_colloc_v5.py
=================

Two-stage Physics-Informed Neural Network for the global anthropogenic zinc
cycle.  Stage A is observation-fitting (collocation); Stage B is shooting
fine-tune through the cohort-augmented ODE.  This file is v5 of the
collocation rewrite — see "Changes vs v4" below.

Changes vs v4
-------------
v5 adds **rolling-origin cross-validation** with optional **fold ensembling**
on top of v4.  The single-split machinery (``trainval_frac``, ``val_frac``)
still works exactly as before — when ``split_indices`` is left as ``None``
``train_model`` falls back to the v4 fraction-based split so legacy callers
need no changes.  The new pieces are:

* **``train_model(split_indices=...)``** — when given, overrides the
  fraction-based split with explicit indices
  ``{"train_end": int, "val_end": int, "test_start": int}`` (all 0-based,
  inclusive end-points).  ``test_start`` decouples the held-out test window
  from the trainval boundary so test stays fixed across folds.

* **``rolling_origin_splits(T, ...)``** — generates a list of expanding-window
  fold specs.  Defaults follow walk-forward CV with a held-out test (Tashman
  2000; Bergmeir et al. 2018):
    - test region [test_start, T-1] is fixed (last ``test_frac`` of the series)
    - trainval region [0, test_start-1] is folded by an expanding train window
    - each fold has a ``val_horizon``-year validation block immediately after
      the train block
    - origins are spaced by ``step`` years (auto-computed if ``None``)
  Expanding window only in v5 — sliding window would require discarding
  early observations that anchor τ_in-use identification, so it's not the
  right default for this dataset.

* **``train_model_rolling_origin(...)``** — runs ``train_model`` once per
  fold, refitting *everything* (NN, Stage A, Stage B, exog/α/τ/S stats) on
  the fold's train slice only.  Returns a ``RollingOriginResult`` with
  per-fold ``FitResult`` objects, an aggregate CV score, and an
  ``ensemble_predict`` method that averages free-run / test-run trajectories
  across the K fold models.  Fold ensembling (Lakshminarayanan et al. 2017)
  is the standard variance-reduction technique on top of CV — for short
  series with high init variance it typically beats any single-fold or
  full-trainval-refit model on the held-out test.

* **``run_cv(name, ...)``** — convenience entry point analogous to
  ``run`` but for the rolling-origin pipeline.

Other v5 quality-of-life additions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``RollingOriginResult.cv_summary()`` prints mean ± std of family-mean MAPE
  / relRMSE across folds, plus a per-fold breakdown — turning fold variance
  into a first-class diagnostic.
* ``RollingOriginResult.ensemble_diagnose(kind="testrun")`` evaluates the
  K-model ensemble on the held-out test set and prints the v4-style table.
* When using the default expanding-window CV, training data only ever grows
  across folds — no observation is ever discarded — which is the right
  default for short series with cohort dynamics that need long τ histories
  to identify.

Why rolling origin (not purged k-fold or k-fold blocked)?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Purged / embargo CV (Lopez de Prado, 2018) targets sample-correlation in
features that arises from overlapping return horizons in finance — that
isn't the dominant concern here.  Stage A supervises midpoint α and
integer-year τ at non-overlapping times, and Stage B teacher-forces from
observed IC at each window start.  Walk-forward (rolling origin) is the
appropriate default per Tashman 2000 and Bergmeir et al. 2018 (Pattern
Recognition).  For genuine test-performance gains, deep ensembling across
folds (Lakshminarayanan et al. 2017) is the lever that compounds with CV.

Modelling framework (aligned with Pauliuk-style dynamic MFA + Rostek 2022)
-------------------------------------------------------------------------
Observable stocks  (4)            S_conc, S_ref, S_inuse_total, S_scrap   [kt]
Cohort sub-stocks  (3)            S_inuse split into short/medium/long-lived
                                  pools with FIXED Rostek lifetimes
                                  MU = (10, 20, 44) yr.
Stock-proportional rates (4)      α_cc, α_refc, α_win, α_dr           [1/yr]
Binary transfer coefficients (4)  τ_ref, τ_waelz, τ_olds, τ_diss      [0,1]
Manufacturing simplexes (2 × 3)   (frac_fu_new, frac_fu_loss, frac_fu_out)
                                  (frac_eu_new, frac_eu_loss, frac_into_use)
Cohort split simplex (3)          f_cohort  — split of inuse_inflow across
                                  the three lifetime cohorts.
Concentrate production            cp(t)  — typically pinned to ILZSG data.

The end-of-life (EoL) flow is NOT a single rate over a stock — it is the
sum of cohort decays, F_eol = Σ_i S_cohort_i / MU_i, where the cohort
sub-stocks evolve via  dS_cohort_i/dt = f_cohort_i · inuse_inflow - S_cohort_i / MU_i.
There is therefore NO α_eol — the only learnable EoL parameter is f_cohort
(see "Cohort identification" below).

Stage A — *Collocation / observation fitting* (always run)
    The NN is trained, at the data grid only, against parameter observations
    derived from the dataset:

        α_k_emp(t_mid)    = ∫_{t_n}^{t_{n+1}} F_k(s) ds  /  S_parent_exposure
                            assigned to the MIDPOINT  t_mid = t_n + 0.5
                            (yearly-integral semantics, not left-endpoint)
        τ_j_obs(t)        = published 0–1 transfer coefficients (Rostek SI)
        cp_obs(t)         = ILZSG concentrate production       [if learn_cp]

    NN-α is supervised at midpoint years (interpolated stocks, exog).  τ
    is supervised at integer years (its observed values are dimensionless
    annual ratios).  No ODE integration anywhere in Stage A.

Stage B — *Shooting fine-tune* (optional)
    Integrate the augmented ODE  dY/dt = f(Y, NN(S, u; θ))  from S_obs(t_0)
    using diffrax (or jax.experimental.ode), producing endogenous stocks
    S_pred(t).  The NN is then evaluated at S_pred(t) and supervised against
    the same α/τ/cp targets, with an additional stock-divergence term
        L_S = Σ_k MSE(S_pred_k(t), S_obs_k(t)) / σ²_{S_k}.

    f_cohort is identified primarily HERE — through the cohort dynamics +
    stock divergence on S_inuse and S_scrap.  In Stage A f_cohort is only
    constrained by an optional KL-prior toward IC_COHORT_FRACS.

Pinning quantities to data
--------------------------
Each of cp, τ_ref, τ_waelz, τ_olds, τ_diss, frac_fu, frac_eu can be PINNED
to its observed value (linear interp of the dataset over time).  A pinned
quantity:
  - is removed from the NN's raw output head (the NN literally becomes
    smaller, no wasted slots);
  - is pulled directly from data inside both the loss and the ODE RHS;
  - is excluded from the loss (its residual is identically 0) and from the
    diagnostics (reported as `[pinned]`).

Default config pins cp.  Recommendation: also pin τ_waelz, τ_ref, τ_diss
when their observed values are nearly time-invariant — this materially
improves test-period generalisation by removing degrees of freedom that
the NN would otherwise spend extrapolating.

Cohort identification
---------------------
There is no observation of f_cohort.  Three identification levers, in
order of strength:
  1.  Stage B stock divergence: mis-allocating cohorts changes the
      build-up of S_inuse and S_scrap → identifiable from data.
  2.  KL prior toward IC_COHORT_FRACS = (0.09, 0.27, 0.64) [Rostek Fig. 3].
      Set `stageA_w_cohort_prior` and/or `stageB_w_cohort_prior` > 0.
  3.  None in Stage A by default — f_cohort is essentially free.

Changes vs v3
-------------
* **Full flow inventory loaded from the dataset.**  v3's `_flow_obs_specs`
  loaded only 5 mandatory + 5 optional columns, AND three of the optional
  column names were stale ("Old scrap recovery", "End-of-life flow",
  "Waelz recycling") so they silently failed to match the actual sheet
  headers.  That meant the manufacturing waste flows, new-scrap flows,
  EOL/old-scrap, products-into-use, and the dissipative split were all
  invisible to Stage-B flow loss, the diagnose() flow table, AND
  plot_flows.  v4 loads every flow that appears in `FLOW_NAMES` (the
  only one not in the dataset is `end_of_life_losses`, which is purely
  derived).  Missing columns are tolerated with a one-line note.
* **Pin flags for the new-scrap fractions.**  v3 had `pin_frac_*_loss`
  which pinned ONLY the manufacturing-waste fraction of each simplex.
  v4 adds `pin_frac_fu_new` / `pin_frac_eu_new` and supports the full
  Cartesian product:
    (pin_loss=F, pin_new=F) → "full":         3 logits + softmax
    (pin_loss=T, pin_new=F) → "loss_pinned":  1 sigmoid (new vs out)
    (pin_loss=F, pin_new=T) → "new_pinned":   1 sigmoid (loss vs out)   [NEW]
    (pin_loss=T, pin_new=T) → "both_pinned":  0 NN slots                [NEW]
  Warm-start, layout, learned-tau mask, and at-obs override paths all
  handle the four cases independently for fu and eu.
* **`fit.diagnose(kind)` accepts "freerun" | "testrun" | "both"**.  The
  v3 diagnose only reported the freerun rollout (ODE launched at year
  1980 from S(1980), integrated all the way through 2019), so the
  `test_*` columns reflected errors accumulated across train + val +
  test.  v4's `kind="testrun"` re-launches the integrator at the LAST
  VALIDATION YEAR (≈2006) using observed S there as IC and integrates
  forward through the test window only — numerically comparable to v1's
  `stock_relRMSE_freerun` metric.  The IC row itself (last validation
  year) is excluded from the testrun metric since predicted == observed
  there by construction.
* **`fit.summary(kind="both")`** — default — prints a four-column
  table (MAPE / relRMSE × freerun / testrun) for stocks and flows.
  α / τ / cp are rollout-independent (evaluated at S_obs) so a single
  pair of values is shown.  `kind="freerun"` and `kind="testrun"` give
  the single-rollout views.
* **summary reports both MAPE and %RMSE.**  v3 reported only family-mean
  MAPE.  Both metrics are now visible side by side.  Pinned variables
  are excluded from the family means by default (set
  `include_pinned=True` to include them) — this includes a new pinning
  category: when `learn_cp=False`, the `concentrate_production` flow
  is tagged `[pinned]` (it IS cp itself, computed straight from the
  data lookup) and excluded from the flows family mean / hidden in
  plots.  Without this, the freerun "flows" mean was being deflated by
  a tautology.
* **`fit.diagnose(...)` no longer dumps its row dicts at the REPL.**
  The printed table is the intended output; the function now returns
  `None` by default.  Programmatic users who want the rows can pass
  `return_rows=True`.
* **Configurable final-refit curriculum.**  v3's final refit was a fixed
  single-window replay loop with `final_refit_steps` SGD steps over the
  full trainval window.  v4 accepts
  `final_refit_curriculum=[(steps, window), …]` à la `stageB_curriculum`,
  with LR decay 0.6**ci across stages and a configurable
  `final_refit_batch_size` (defaults to `stageB_batch_size`).  When
  `final_refit_curriculum=None` the legacy single-window behaviour is
  preserved exactly.
* **Two-panel `plot_cohorts`.**  Top panel = actual cohort STOCK
  fractions (S_k / Σ S_k) extracted from the ODE state — these start
  at IC_COHORT_FRACS = (0.09, 0.27, 0.64) by construction (the prior
  markers sit on the lines at 1980).  Bottom panel = NN-predicted
  INFLOW SPLIT f_cohort, the 3-simplex applied to inuse_inflow.  v3
  showed only the inflow split with an IC marker that didn't actually
  apply (it's a stock-distribution prior, not a flow split), creating
  the impression that the curves "should" pass through it.
* **Cohort sub-stocks exposed in predictions.**  The integrator now
  returns S_cohorts_year(T, N_COHORTS) alongside S_year and F_int.
  Available as `predictions["S_cohorts"]` for downstream analysis.
* **Stage A α residual difference vs v1 explained, not fixed.**  When
  `learn_cp=False`, v3/v4 omit the cp slot from the NN head, so the
  head W has shape (17, 32) vs v1's (18, 32).  Different random init
  → different local minimum → different (and not directly comparable)
  Stage-A α MSE.  Not a regression; documented here for future readers
  who notice the gap.

Changes vs v2 (historical)
--------------------------
* **Piecewise-constant interpolation for pinned data-driven quantities**
  (cp, τ).  v2 linearly interpolated annual-integral data (e.g. ILZSG cp)
  between integer year-end samples.  Integrating that signal over (Y-1, Y]
  yielded ≈ ½(cp_obs[Y-1] + cp_obs[Y]) ≈ cp_obs[Y - ½], introducing a
  systematic bias ≈ ½ × growth-rate (≈ 1–2 %) into F_int.  v3 reverts to
  the same convention used in `zinc_pinn_v10_2`: cp(t) = cp_obs[Y] for
  t ∈ (Y-1, Y], i.e. the only interpolation that makes the per-year
  integral  ∫_{Y-1}^{Y} cp(t) dt = cp_obs[Y]  exactly (every continuous
  scheme leaks ≈ d²cp/dt² of bias per year, which on noisy data is ≫ the
  growth-rate bias).  τ uses the same convention.  After this fix the cp
  residual under "flows" diagnostics drops from ~1.5 % to the ODE
  truncation noise floor (~10⁻⁴ %).
* **Partial pinning of the manufacturing simplexes**.  v2's `pin_frac_fu` /
  `pin_frac_eu` pinned the entire 3-simplex to data (loss + new + out all
  fixed).  In practice we usually only know `frac_*_loss` well (it's
  manufacturing waste — short-run roughly invariant), while `frac_*_new` is
  highly variable.  v3 introduces `pin_frac_fu_loss` / `pin_frac_eu_loss`
  which pin ONLY the loss fraction; the NN emits 1 sigmoid that splits the
  remaining (1 - loss) probability mass between new-scrap and out-flow.
  (The old all-or-nothing pin flags are removed.)
* **Stage A and Stage B parameters are now stored separately**.  v2 returned
  only the post-Stage-B params, so the diagnose / plot routines couldn't
  distinguish "Stage A predictions" (NN_A @ S_obs) from a degraded view of
  Stage B.  v3 stores `params_A` (the Stage-A fitted weights) AND `params_B`
  (the Stage-B fine-tuned weights) on the FitResult object.  All Stage-A
  evaluations use NN_A; all Stage-B evaluations use NN_B.
* **Stocks and flows are evaluated by ODE integration in BOTH stages**.  v2
  evaluated stocks via ODE integration only after Stage B.  v3 builds the
  integrator always (cheap), and Stage A's stocks/flows trajectories come
  from integrating the ODE with NN_A — the same closed-loop free-run as
  Stage B but with Stage-A weights.  This makes "Stage A vs Stage B" a
  fair comparison of dynamics quality, not Teacher-forcing-vs-free-run.
* **`run()` prints only training steps + a compact family-average MAPE
  summary** at the end (alphas / taus / stocks / flows) on the **TEST
  SET only**, since that is the held-out metric that matters for
  generalisation.  Pinned variables are excluded from those averages.
  The full per-component table on every split (train / val / test / all)
  is now produced ONLY by an explicit `fit.diagnose()` call.
* **`fit.diagnose()` reports logMAE, %RMSE, MAPE per component** and
  per-family averages, for both stages, with `include_pinned=False` by
  default (pinned components are reported only when explicitly requested).
* **Plot helpers skip pinned variables by default** (use
  `include_pinned=True` to override).  Stock and flow plots show NN_A's
  ODE free-run alongside NN_B's ODE free-run.
* **`plot_cohorts` (single-panel)** — all three cohort fractions
  (short / medium / long) on one axis, distinct colour per cohort and
  distinct linestyle per stage, with the IC prior at 1980 marked as
  hollow circles for context.
* **Diagnostic identity for pinned quantities is exact** — when cp / τ /
  frac_*_loss are pinned, the `*_at_obs` diagnostic is overridden with
  the data lookup itself (rather than re-evaluating the NN, which would
  use the `_interval_pick` boundary clip and leak a cosmetic ~0.04 % at
  year 0).  Pinned components now show MAPE = 0.000 in `diagnose()`.
* **`fit.params_distance()` and `summary()` print  ‖θ_B − θ_A‖₂ / ‖θ_A‖₂**
  — a one-line answer to "did Stage B do anything?".  Useful diagnostic
  when Stage B looks too similar to Stage A in plots.
* **Stock loss switched to log-MSE** — Stage B's stock-divergence term
  was  ‖S_pred − S_obs‖² / S_target_std²  in v2.  v3 defaults to
  ‖log S_pred − log S_obs‖² / S_log_std²,  which (i) aligns the training
  objective with the MAPE we report, (ii) prevents large stocks (in-use)
  from dominating purely because of scale, and (iii) prevents low-
  variability stocks from being under-weighted because of small std.
  Selectable via `stock_loss_kind ∈ {"log", "std_scaled"}`; the legacy
  v2 behaviour is reachable as `"std_scaled"`.
* The negative-stocks ("neg=…") diagnostic from earlier versions is gone.

Changes vs v1 (kept for the historical record)
----------------------------------------------
* `learn_cp=False` genuinely removes cp from the NN head.
* Per-quantity pin flags: `pin_tau_ref`, `pin_tau_waelz`, `pin_tau_olds`,
  `pin_tau_diss`.
* Stage A loss only adds terms for *unpinned* quantities.
* Stage B uses the same pin-aware NN composition; the ODE pulls pinned
  values from interpolated data.
* `stock_term_weights` removed from defaults (uniform).
* `grad_clip` defaults to 0.0 (disabled) — log-space losses + bounded
  parametrisation make it unnecessary.

Dependencies
------------
    Required: numpy, pandas, openpyxl, jax, jaxlib, optax
    Optional: diffrax, matplotlib
"""

from __future__ import annotations

import copy
import json
import math
import time

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
import optax

jax.config.update("jax_enable_x64", True)

try:
    import diffrax as dfx
    HAS_DIFFRAX = True
except Exception:  # pragma: no cover
    dfx = None
    HAS_DIFFRAX = False

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:  # pragma: no cover
    plt = None
    HAS_MPL = False


# ============================================================================
# 1) Constants, names, and topology
# ============================================================================
# Observable stocks (4):  Concentrate, Refined, In-Use total, Scrap
N_STOCKS  = 4
STOCK_NAMES = ["Concentrate", "Refined", "In-Use", "Scrap"]

# Stock-proportional parent rates (4):
#   alpha_cc   : concentrate consumption  rate  (per year, ×S_conc)
#   alpha_refc : refined consumption rate       (per year, ×S_ref)
#   alpha_win  : waelz input rate               (per year, ×S_scrap)
#   alpha_dr   : direct re-use recycling rate   (per year, ×S_scrap)
N_ALPHAS = 4
ALPHA_NAMES = ["alpha_cc", "alpha_refc", "alpha_win", "alpha_dr"]
ALPHA_PARENT_STOCK_IDX = (0, 1, 3, 3)   # which observable stock drives each rate

# Transfer coefficients in (0, 1) (4 binary + 4 simplex residuals):
#   tau_ref        : refinery loss share of concentrate consumption
#   tau_waelz      : waelz process recovery share of waelz input
#   tau_olds       : old-scrap recovery share of EoL flow
#   tau_diss       : dissipative-use share of products into use
#
#   frac_fu_new    : new-scrap share at first-use manufacturing
#   frac_fu_loss   : loss share at first-use manufacturing
#   (frac_fu_out implied = 1 - frac_fu_new - frac_fu_loss)
#
#   frac_eu_new    : new-scrap share at end-use manufacturing
#   frac_eu_loss   : loss share at end-use manufacturing
#   (frac_into_use implied = 1 - frac_eu_new - frac_eu_loss)
N_TAUS_BINARY = 4
TAU_BINARY_NAMES = ["tau_ref", "tau_waelz", "tau_olds", "tau_diss"]

# We keep the manufacturing splits as 3-way simplexes inside the NN, but for
# *supervision* we only need their two free degrees of freedom (new + loss),
# since the third (out / into_use) follows by closure.
N_MANU_SUP = 4   # frac_fu_new, frac_fu_loss, frac_eu_new, frac_eu_loss
MANU_SUP_NAMES = ["frac_fu_new", "frac_fu_loss", "frac_eu_new", "frac_eu_loss"]

# All "tau-like" supervised quantities (0–1 valued, observed in the dataset):
N_TAU_SUP = N_TAUS_BINARY + N_MANU_SUP   # = 8
TAU_SUP_NAMES = TAU_BINARY_NAMES + MANU_SUP_NAMES

# In-use cohort lifetimes (fixed, from Rostek et al. 2022 SI Table S4)
N_COHORTS = 3
COHORT_NAMES = ["f_short_lived", "f_medium_lived", "f_long_lived"]
MU_COHORTS_NP = np.array([10.0, 20.0, 44.0])
MU_COHORTS    = jnp.array([10.0, 20.0, 44.0])
# Initial cohort split of S_inuse(t0) — Rostek Fig. 3 c. 1980 share
IC_COHORT_FRACS = np.array([0.09, 0.27, 0.64])

# Augmented ODE state: [S_conc, S_ref, S_iu_short, S_iu_med, S_iu_long, S_scrap]
N_ODE_STOCKS = 2 + N_COHORTS + 1   # = 6
IDX_CONC, IDX_REF = 0, 1
IDX_SCRAP         = 2 + N_COHORTS   # = 5

# NN raw output layout (DYNAMIC, depends on which quantities are pinned).
# In v1 N_RAW_OUT was a fixed 18; in v2 the head is built with `build_nn_layout`
# at train_model time and the head width depends on pin flags.  See `make_nn_eval`.
N_LOGIT_FU      = 3
N_LOGIT_EU      = 3
N_LOGIT_COHORT  = N_COHORTS
# (no fixed N_RAW_OUT here — the head width is per-layout)


# ============================================================================
# 2) Small numeric utilities
# ============================================================================
def _safe_log(x, eps=1e-12):
    return jnp.log(jnp.maximum(x, eps))


def _masked_mean(x, mask, eps=1e-12):
    """Mean of x over locations where mask is True."""
    m = mask.astype(x.dtype)
    return jnp.sum(m * x) / (jnp.sum(m) + eps)


def _per_feature_mse(resid, scale, weights, eps=1e-12):
    """Per-feature squared error averaged over leading axes, weighted."""
    z = (jnp.asarray(resid) / jnp.asarray(scale)) ** 2
    if z.ndim == 1:
        z_mean = z
    else:
        z_mean = jnp.mean(z, axis=tuple(range(z.ndim - 1)))
    w = jnp.asarray(weights, dtype=z.dtype)
    return jnp.sum(z_mean * w) / (jnp.sum(w) + eps)


def _per_feature_masked_mse(resid, scale, weights, mask, eps=1e-12):
    """Like _per_feature_mse but with a per-element finite mask (for NaN obs)."""
    resid = jnp.where(mask, resid / jnp.asarray(scale), 0.0)
    z     = resid ** 2
    cnt   = jnp.sum(mask.astype(z.dtype), axis=tuple(range(z.ndim - 1)))   # (F,)
    s     = jnp.sum(z,                    axis=tuple(range(z.ndim - 1)))    # (F,)
    z_mean = s / jnp.maximum(cnt, eps)
    w = jnp.asarray(weights, dtype=z.dtype)
    return jnp.sum(z_mean * w) / (jnp.sum(w) + eps)


def _logit(p, eps=1e-6):
    """Inverse sigmoid; safe-clipped."""
    p = jnp.clip(p, eps, 1.0 - eps)
    return jnp.log(p / (1.0 - p))


# ============================================================================
# 3) MLP
# ============================================================================
def init_mlp(layer_sizes, key):
    """He-initialised MLP, biases zero."""
    params = []
    keys   = jax.random.split(key, len(layer_sizes) - 1)
    for k, (m, n) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
        w_key, _ = jax.random.split(k)
        W = jax.random.normal(w_key, (n, m)) * jnp.sqrt(2.0 / m)
        b = jnp.zeros((n,))
        params.append({"W": W, "b": b})
    return params


def mlp_apply(params, x):
    """Forward MLP with tanh hidden activations; linear head."""
    h = x
    for layer in params[:-1]:
        h = jnp.tanh(layer["W"] @ h + layer["b"])
    last = params[-1]
    return last["W"] @ h + last["b"]


# ============================================================================
# 4) Exogenous driver interpolation
# ============================================================================
def exog_fn(t, exog_times, exog_values):
    """Linear interpolation of exogenous values at (scalar) time t."""
    return jax.vmap(lambda col: jnp.interp(t, exog_times, col))(exog_values.T)


# ============================================================================
# 5) NN input normalisation
# ============================================================================
def _norm_inputs(t, S, exog_t, stats):
    """Build the NN feature vector  [t_feat, S_feat (4), exog_feat (n_exog)].

    Stock normalisation modes (selected by stats["stock_norm_mode"]):
      0  zscore     :  (S - S_mean) / S_std
      1  log        :  log(max(S, 1) / S_ref)
      2  tanh_log   :  tanh(log(max(S, 1) / S_ref) / log_scale)        [default]

    `tanh_log` bounds the NN's stock features in (-1, 1) regardless of
    magnitude, which is safer than zscore when the test-period stocks lie
    several training-σ away from the training mean.
    """
    t_feat = (t - stats["t_mean"]) / stats["t_std"] * stats["use_time_input"]

    mode      = stats["stock_norm_mode"]
    S_ref     = stats["S_ref"]
    log_scale = stats["S_log_scale"]
    S_safe = jnp.maximum(S, 1.0)
    S_log  = jnp.log(S_safe / jnp.maximum(S_ref, 1.0))
    S_z    = (S - stats["S_mean"]) / stats["S_std"]
    S_tanh = jnp.tanh(S_log / jnp.maximum(log_scale, 1e-6))

    is_z   = jnp.where(jnp.abs(mode - 0.0) < 0.5, 1.0, 0.0)
    is_l   = jnp.where(jnp.abs(mode - 1.0) < 0.5, 1.0, 0.0)
    is_tl  = jnp.where(jnp.abs(mode - 2.0) < 0.5, 1.0, 0.0)
    S_feat_raw = is_z * S_z + is_l * S_log + is_tl * S_tanh
    S_feat = S_feat_raw * stats["use_stock_input"]

    exog_norm = (exog_t - stats["exog_mean"]) / stats["exog_std"]
    return jnp.concatenate([jnp.array([t_feat]), S_feat, exog_norm])


# ============================================================================
# 6) NN forward — dynamic-width head, pin-aware composition
# ============================================================================
# In v2, each of {cp, τ_ref, τ_waelz, τ_olds, τ_diss, frac_fu, frac_eu} can be
# PINNED to its observed value (linear interp of data over time).  When pinned,
# the corresponding slot is REMOVED from the NN's raw output head — the network
# literally becomes smaller, no wasted slots and no leakage into diagnostics.
# `f_cohort` and the four α's are always learned (no observation for f_cohort,
# and α's drive the dynamics).

def build_nn_layout(*, learn_cp,
                    pin_tau_ref, pin_tau_waelz, pin_tau_olds, pin_tau_diss,
                    pin_frac_fu_loss, pin_frac_eu_loss,
                    pin_frac_fu_new=False, pin_frac_eu_new=False):
    """Compute the NN raw-output layout given pin flags.

    Manufacturing simplex pinning semantics (extended in v4)
    --------------------------------------------------------
    The first-use simplex (frac_fu_new, frac_fu_loss, frac_fu_out) and the
    end-use simplex (frac_eu_new, frac_eu_loss, frac_into_use) each have
    2 degrees of freedom (3 components on the simplex).  v4 supports the
    full Cartesian product of (pin_loss, pin_new) flags:

      (pin_loss=F, pin_new=F) → "full":         3 logits + softmax  (default)
      (pin_loss=T, pin_new=F) → "loss_pinned":  1 sigmoid (new vs out)
      (pin_loss=F, pin_new=T) → "new_pinned":   1 sigmoid (loss vs out)   [NEW]
      (pin_loss=T, pin_new=T) → "both_pinned":  0 NN slots (out determined) [NEW]

    Same logic applies independently to the end-use simplex.  When ``both_pinned``
    is selected the third component is fully determined by closure
    ``frac_*_out = max(1 - frac_*_new - frac_*_loss, 1e-6)`` — the NN gets no
    say in either of the manufacturing-split DOFs for that simplex.

    Returns a dict with:
      n_raw            : total raw output dimension
      cp_slot          : int or None (None ⇔ cp is pinned)
      alpha_start      : int (4 contiguous α slots — always learned)
      tau_slots        : tuple length 4 of (int or None) — per-binary-τ slot
      fu_kind          : "full" | "loss_pinned" | "new_pinned" | "both_pinned"
      fu_logits_start  : int    if fu_kind=="full"        (3 slots)
      fu_split_slot    : int    if fu_kind in {loss_pinned, new_pinned} (1 slot)
                                — sigmoid output gives the ratio between the
                                two un-pinned components on the remaining mass
      eu_kind, eu_logits_start, eu_split_slot   : same for end-use
      coh_start        : int (3 contiguous cohort logits — always learned)
      pin_taus_tuple   : tuple of bool (for closure capture)
      pin_cp           : bool
      pin_fu_loss / pin_eu_loss : bool
      pin_fu_new / pin_eu_new   : bool      (NEW in v4)
    """
    pin_cp_flag      = (not learn_cp)
    pin_tau_list     = [pin_tau_ref, pin_tau_waelz, pin_tau_olds, pin_tau_diss]
    pin_fu_loss_flag = bool(pin_frac_fu_loss)
    pin_eu_loss_flag = bool(pin_frac_eu_loss)
    pin_fu_new_flag  = bool(pin_frac_fu_new)
    pin_eu_new_flag  = bool(pin_frac_eu_new)

    def _simplex_layout(cur, pin_loss, pin_new, n_logit_full):
        """Decide simplex parametrisation and consume slots.

        Returns (cur, kind, logits_start, split_slot).  `logits_start` is
        non-None only for "full"; `split_slot` is non-None only for the
        single-sigmoid forms; "both_pinned" consumes zero slots.
        """
        if not pin_loss and not pin_new:
            kind         = "full"
            logits_start = cur; cur += n_logit_full
            split_slot   = None
        elif pin_loss and not pin_new:
            kind         = "loss_pinned"
            logits_start = None
            split_slot   = cur; cur += 1
        elif not pin_loss and pin_new:
            kind         = "new_pinned"
            logits_start = None
            split_slot   = cur; cur += 1
        else:
            kind         = "both_pinned"
            logits_start = None
            split_slot   = None
        return cur, kind, logits_start, split_slot

    cur = 0
    if not pin_cp_flag:
        cp_slot = cur; cur += 1
    else:
        cp_slot = None

    alpha_start = cur; cur += N_ALPHAS

    tau_slots = []
    for pin_j in pin_tau_list:
        if pin_j:
            tau_slots.append(None)
        else:
            tau_slots.append(cur); cur += 1
    tau_slots = tuple(tau_slots)

    cur, fu_kind, fu_logits_start, fu_split_slot = _simplex_layout(
        cur, pin_fu_loss_flag, pin_fu_new_flag, N_LOGIT_FU
    )
    cur, eu_kind, eu_logits_start, eu_split_slot = _simplex_layout(
        cur, pin_eu_loss_flag, pin_eu_new_flag, N_LOGIT_EU
    )

    coh_start = cur; cur += N_LOGIT_COHORT
    n_raw     = cur

    return dict(
        n_raw=n_raw,
        cp_slot=cp_slot,
        alpha_start=alpha_start,
        tau_slots=tau_slots,
        fu_kind=fu_kind,
        fu_logits_start=fu_logits_start,
        fu_split_slot=fu_split_slot,
        eu_kind=eu_kind,
        eu_logits_start=eu_logits_start,
        eu_split_slot=eu_split_slot,
        coh_start=coh_start,
        pin_taus_tuple=tuple(bool(p) for p in pin_tau_list),
        pin_cp=pin_cp_flag,
        pin_fu_loss=pin_fu_loss_flag,
        pin_eu_loss=pin_eu_loss_flag,
        pin_fu_new=pin_fu_new_flag,
        pin_eu_new=pin_eu_new_flag,
    )


def _interval_pick(t, years, series):
    """Piecewise-constant pick: returns ``series[i]`` for  t ∈ (years[i-1], years[i]],
    with edge clipping ``[1, N-1]`` so out-of-range queries fall on the
    nearest end interval.  This is the same convention used in
    ``zinc_pinn_v10_2`` (function of the same name) and is the *only* way
    to make  ∫_{Y-1}^{Y} cp(t) dt = cp_obs[Y] hold *exactly* — every other
    interpolation scheme leaks ≈ ½·d²cp/dt²·Δt² of bias per year, which on
    a noisy time-series easily exceeds 0.5 % even though the *mean* growth
    rate is only ~1.7 %/yr.

    `searchsorted(side="left")` returns the leftmost insertion index, so
    for t exactly equal to years[i] we get  idx = i  → series[i] (year-end
    inclusive on the right, matching the integration convention).
    """
    idx = jnp.clip(jnp.searchsorted(years, t, side="left"), 1, years.shape[0] - 1)
    return series[idx]


def _yearpoint_pick(t, years, series):
    """Piecewise-constant pick clipped to ``[0, N-1]`` (no left-shift).
    Used for τ-style coefficients where the value "during year Y" is
    naturally indexed by Y itself.  At t < years[0] we return series[0]
    (rather than series[1] as in `_interval_pick`), which slightly matters
    only if the integrator ever queries τ before the start of the data
    window — it doesn't, in normal operation.
    """
    idx = jnp.clip(jnp.searchsorted(years, t, side="left"), 0, years.shape[0] - 1)
    return series[idx]


def _interp_cp(t, data):
    """Pinned cp(t):  piecewise-constant equal to cp_obs[Y] on (Y-1, Y].

    cp_obs[Y] is the ILZSG annual integral  ∫_{Y-1}^{Y} cp(t) dt.
    With cp(t) = cp_obs[Y] on (Y-1, Y] this integral is exactly cp_obs[Y]
    — the only interpolation scheme that achieves this.  v3 originally
    tried a midpoint-linear scheme on the assumption that cp_obs[Y]
    represented an average *rate* at  Y - ½; that gives exact integrals
    only when cp has zero second difference, which the real noisy series
    decidedly does not (~0.5 % residual bias).  Switching to piecewise-
    constant (matching `zinc_pinn_v10_2`) brings the integration error
    down to the ODE solver's truncation noise (≈ 10⁻⁴ %).
    """
    return _interval_pick(t, data["years"], data["cp_obs"])


def _interp_tau(t, data, j):
    """Pinned τ_j(t):  piecewise-constant equal to τ_obs[Y] on (Y-1, Y].

    τ values are "effective during year Y" transfer coefficients used to
    multiply parent-flow integrals over (Y-1, Y].  Setting
    τ(t) = τ_obs[Y] on (Y-1, Y] makes  ∫_{Y-1}^{Y} τ(t) F(t) dt
    = τ_obs[Y] · ∫_{Y-1}^{Y} F(t) dt  — exact factorisation, identically
    matching how the supervised observation is constructed.  Same
    convention as `zinc_pinn_v10_2`.
    """
    return _interval_pick(t, data["years"], data["tau_sup_obs"][:, j])


def make_nn_eval(layout):
    """Build a JIT-friendly closure that evaluates the full set of NN
    parameters at a single (t, S, exog_t), composing learned NN outputs with
    pinned interpolated-from-data values.

    Returns a function `nn_eval(params, t, S, exog_t, data)` that yields a
    dict with keys:
        cp        : ()
        alphas    : (N_ALPHAS,)
        taus      : (N_TAUS_BINARY,)        (binary taus only)
        frac_fu   : (3,)                     (simplex)
        frac_eu   : (3,)                     (simplex)
        f_cohort  : (N_COHORTS,)             (simplex)
        raw_norm  : ()                       mean(raw²) over learned slots
    """
    cp_slot         = layout["cp_slot"]
    alpha_start     = layout["alpha_start"]
    tau_slots       = layout["tau_slots"]
    fu_kind         = layout["fu_kind"]
    fu_logits_start = layout["fu_logits_start"]
    fu_split_slot   = layout["fu_split_slot"]
    eu_kind         = layout["eu_kind"]
    eu_logits_start = layout["eu_logits_start"]
    eu_split_slot   = layout["eu_split_slot"]
    coh_start       = layout["coh_start"]
    pin_cp          = layout["pin_cp"]
    pin_taus        = layout["pin_taus_tuple"]
    pin_fu_loss     = layout["pin_fu_loss"]
    pin_eu_loss     = layout["pin_eu_loss"]
    pin_fu_new      = layout["pin_fu_new"]
    pin_eu_new      = layout["pin_eu_new"]

    def _simplex_value(raw, kind, split_slot, logits_start, n_logits,
                       loss_data, new_data):
        """Compose (new, loss, out) given pinning kind.

        loss_data, new_data : scalars from data interpolation, used only
                              when the corresponding component is pinned.
        Returns a (3,) array on the simplex.
        """
        if kind == "full":
            logits = jax.lax.dynamic_slice_in_dim(raw, logits_start, n_logits, axis=0)
            return jax.nn.softmax(logits)
        if kind == "loss_pinned":
            loss  = jnp.clip(loss_data, 0.0, 1.0 - 1e-6)
            sigma = jax.nn.sigmoid(raw[split_slot])
            avail = jnp.maximum(1.0 - loss, 1e-6)
            new   = sigma         * avail
            out   = (1.0 - sigma) * avail
            return jnp.stack([new, loss, out])
        if kind == "new_pinned":
            new   = jnp.clip(new_data, 0.0, 1.0 - 1e-6)
            sigma = jax.nn.sigmoid(raw[split_slot])
            avail = jnp.maximum(1.0 - new, 1e-6)
            loss  = sigma         * avail
            out   = (1.0 - sigma) * avail
            return jnp.stack([new, loss, out])
        # both_pinned: out is fully determined by closure
        new   = jnp.clip(new_data,  0.0, 1.0 - 1e-6)
        loss  = jnp.clip(loss_data, 0.0, 1.0 - 1e-6)
        out   = jnp.maximum(1.0 - new - loss, 1e-6)
        return jnp.stack([new, loss, out])

    def nn_eval(params, t, S, exog_t, data):
        stats = data["stats"]
        x   = _norm_inputs(t, S, exog_t, stats)
        raw = mlp_apply(params, x)

        # cp
        if pin_cp:
            cp = _interp_cp(t, data)
        else:
            cp = stats["cp_scale"] * jnp.exp(jnp.clip(raw[cp_slot], -8.0, 8.0))

        # alphas (always learned)
        alpha_raw = jax.lax.dynamic_slice_in_dim(raw, alpha_start, N_ALPHAS, axis=0)
        alphas    = stats["alpha_scale"] * jnp.exp(jnp.clip(alpha_raw, -8.0, 8.0))

        # binary taus (per-slot pinning)
        tau_list = []
        for j, slot in enumerate(tau_slots):
            if slot is None:
                tau_list.append(_interp_tau(t, data, j))
            else:
                tau_list.append(jax.nn.sigmoid(raw[slot]))
        taus = jnp.stack(tau_list)

        # frac_fu (3-simplex). tau_sup_obs columns:  4=frac_fu_new, 5=frac_fu_loss
        fu_loss_data = _interp_tau(t, data, 5)
        fu_new_data  = _interp_tau(t, data, 4)
        frac_fu = _simplex_value(raw, fu_kind, fu_split_slot, fu_logits_start,
                                 N_LOGIT_FU, fu_loss_data, fu_new_data)

        # frac_eu (3-simplex). tau_sup_obs columns:  6=frac_eu_new, 7=frac_eu_loss
        eu_loss_data = _interp_tau(t, data, 7)
        eu_new_data  = _interp_tau(t, data, 6)
        frac_eu = _simplex_value(raw, eu_kind, eu_split_slot, eu_logits_start,
                                 N_LOGIT_EU, eu_loss_data, eu_new_data)

        # cohort (always learned, no observation)
        logits_coh = jax.lax.dynamic_slice_in_dim(raw, coh_start, N_LOGIT_COHORT, axis=0)
        f_cohort   = jax.nn.softmax(logits_coh)

        return dict(
            cp=cp, alphas=alphas, taus=taus,
            frac_fu=frac_fu, frac_eu=frac_eu, f_cohort=f_cohort,
            raw_norm=jnp.mean(raw ** 2),
        )

    return nn_eval


def make_nn_eval_supervised(layout):
    """Same as make_nn_eval, but returns the bundled supervised tuple:
        (cp, alphas, tau_sup_8, frac_fu, frac_eu, f_cohort, raw_norm)
    where tau_sup_8 is the 8-vector
        [tau_ref, tau_waelz, tau_olds, tau_diss,
         frac_fu_new, frac_fu_loss, frac_eu_new, frac_eu_loss].
    """
    nn_eval = make_nn_eval(layout)
    def nn_eval_sup(params, t, S, exog_t, data):
        out = nn_eval(params, t, S, exog_t, data)
        tau_sup = jnp.concatenate([
            out["taus"],
            jnp.array([out["frac_fu"][0], out["frac_fu"][1],
                       out["frac_eu"][0], out["frac_eu"][1]]),
        ])
        return out["cp"], out["alphas"], tau_sup, out["f_cohort"], out["raw_norm"]
    return nn_eval_sup



# ============================================================================
# 7) Data loading — observation tables and empirical (alpha, tau, cp)
# ============================================================================
def _build_empirical_alphas(years_arr, S_obs, F_obs_for_alpha):
    """Empirical α_k(t) from data, defined at YEAR-END t_n.

    Yearly-integral semantics
    -------------------------
    Following Rostek et al. (2022) and the standard ILZSG/economic-data
    convention, the dataset reports F_k[Y] as the annual integral over
    (Y-1, Y], aligned to the closing year Y.  The empirical rate

        α_k[Y] = ∫_{Y-1}^{Y} F_k(s) ds   /
                 ∫_{Y-1}^{Y} S_parent(s) ds
               ≈ F_k_obs[Y]            /
                 ½ (S_parent[Y-1] + S_parent[Y])             (trapezoid)

    is therefore stored at year Y (year-end convention) and supervises NN-α
    at (year=Y, S=S_obs[Y]) — same time alignment as the published data.

    NB. Mathematically a midpoint placement (Y-½) would be a slightly more
    accurate identification of α(t) for very smooth α paths, but the
    dataset reports at year-end and so does Rostek; we follow that
    convention for direct comparability.

    Returns
    -------
    alpha_obs : (T, N_ALPHAS)   row 0 is NaN (no prior interval to integrate);
                                row i for i≥1 is α at year years_arr[i].
    """
    T = len(years_arr)
    alpha_obs = np.full((T, N_ALPHAS), np.nan, dtype=float)
    if T < 2:
        return alpha_obs

    dt        = years_arr[1:] - years_arr[:-1]
    S_parents = np.array([S_obs[:, k] for k in ALPHA_PARENT_STOCK_IDX]).T   # (T, 4)
    exposure  = 0.5 * (S_parents[:-1] + S_parents[1:]) * dt[:, None]         # (T-1, 4)

    F_int     = np.clip(F_obs_for_alpha[1:, :], 0.0, np.inf)                # (T-1, 4)
    valid     = np.isfinite(F_int) & np.isfinite(exposure) & (exposure > 1e-12)
    out       = np.full_like(F_int, np.nan)
    out[valid] = F_int[valid] / np.maximum(exposure[valid], 1e-12)
    alpha_obs[1:, :] = out
    return alpha_obs


def load_zinc_data(xlsx_path, extra_exog_cols=None, extra_sheet="extra"):
    """Load Stocks/Flows/Exogenous (+ optional extra) sheets and assemble
    the observation tables that Stage A and Stage B need.

    Returns a dict containing (NumPy arrays unless noted):
        years           : (T,)              year grid
        stocks_obs      : (T, 4)            S_conc, S_ref, S_inuse_total, S_scrap
        cp_obs          : (T,)              concentrate production [kt/yr]
        alpha_years     : (T-1,)            MIDPOINT years for α supervision
        alpha_obs       : (T-1, 4)          empirical α at midpoint years
        S_at_alpha      : (T-1, 4)          stocks interpolated to midpoint years
                                            (used as NN input for α loss)
        tau_sup_obs     : (T, 8)            observed tau-like coefficients,
                                            layout matches TAU_SUP_NAMES
        flows_obs       : (T-1, N_FLOW_OBS) yearly-integrated observed flows
                                            (annual integrals over (t_n, t_{n+1}])
        flow_obs_names  : list[str]         column labels for flows_obs
        flow_obs_to_pred_idx : list[int]    map each flows_obs column to its
                                            index in the FULL flow vector emitted
                                            by compute_flows_from_nn (used by
                                            diagnostics & Stage-B flow loss)
        exog_times_full : (Tfull,)
        exog_values_full: (Tfull, n_exog)
        exog_times      : (T,)
        exog_values     : (T, n_exog)
        exog_cols       : list[str]
    """
    stocks_df = pd.read_excel(xlsx_path, sheet_name="Stocks").dropna(subset=["Year"])
    flows_df  = pd.read_excel(xlsx_path, sheet_name="Flows").dropna(subset=["Year"])
    exog_df   = pd.read_excel(xlsx_path, sheet_name="Exogenous").dropna(subset=["Year"])

    if extra_exog_cols is not None:
        try:
            extra_df = pd.read_excel(xlsx_path, sheet_name=extra_sheet)
        except Exception as e:
            print(f"[load_zinc_data] could not read extra sheet '{extra_sheet}': {e}")
            extra_df = None
        if extra_df is not None:
            if extra_df.shape[0] != exog_df.shape[0]:
                print(f"[load_zinc_data] WARNING: extra sheet has "
                      f"{extra_df.shape[0]} rows, Exogenous has "
                      f"{exog_df.shape[0]}.  Aligning by head.")
                n_common = min(extra_df.shape[0], exog_df.shape[0])
                extra_df = extra_df.iloc[:n_common].reset_index(drop=True)
                exog_df  = exog_df.iloc[:n_common].reset_index(drop=True)
            cols = (list(extra_df.columns) if str(extra_exog_cols).lower() == "all"
                    else list(extra_exog_cols))
            missing = [c for c in cols if c not in extra_df.columns]
            if missing:
                raise ValueError(f"extra_exog_cols not found in '{extra_sheet}': {missing}")
            extra_sel = extra_df[cols].reset_index(drop=True)
            exog_df   = pd.concat([exog_df.reset_index(drop=True), extra_sel], axis=1)
            print(f"[load_zinc_data] appended {len(cols)} extra exog columns: {cols}")

    # Observable horizon = intersection of {Stocks, Flows, Exogenous}
    common_years = sorted(set(stocks_df["Year"]) & set(flows_df["Year"]) & set(exog_df["Year"]))
    years_arr    = np.array(common_years, dtype=float)

    stocks_df = stocks_df.set_index("Year").loc[years_arr].sort_index()
    flows_df  = flows_df.set_index("Year").loc[years_arr].sort_index()

    # Full exogenous (kept for differenced features that look back beyond t0)
    exog_full = exog_df.set_index("Year").sort_index()
    exog_full = exog_full.apply(pd.to_numeric, errors="coerce")
    exog_full = exog_full.dropna(axis=1, how="all")
    exog_full = exog_full.interpolate(axis=0, limit_direction="both").ffill().bfill()
    exog_cols = list(exog_full.columns)
    exog_times_full  = exog_full.index.to_numpy(dtype=float)
    exog_values_full = exog_full.to_numpy(dtype=float)

    # Re-index onto the observable horizon
    exog_model = exog_full.reindex(years_arr)
    exog_model = exog_model.interpolate(axis=0, limit_direction="both").ffill().bfill()
    exog_values = exog_model.to_numpy(dtype=float)

    # Observable stocks (4)
    S_obs = np.stack([
        stocks_df["Concentrate Stock"].to_numpy(dtype=float),
        stocks_df["Refined Stock"].to_numpy(dtype=float),
        stocks_df["In-Use Stock"].to_numpy(dtype=float),
        stocks_df["Scrap Stock"].to_numpy(dtype=float),
    ], axis=1)

    # Concentrate production (cp) — observed every year
    cp_obs = flows_df["Concentrate production (ILZSG)"].to_numpy(dtype=float)

    # Build the parent-flow matrix needed for empirical α
    F_for_alpha = np.stack([
        flows_df["Concentrate consumption"].to_numpy(dtype=float),
        flows_df["Refined zinc consumption (ILZSG)"].to_numpy(dtype=float),
        flows_df["Waelz process input"].to_numpy(dtype=float),
        flows_df["Direct reuse recycling"].to_numpy(dtype=float),
    ], axis=1)
    alpha_obs = _build_empirical_alphas(years_arr, S_obs, F_for_alpha)

    # Observed tau-like coefficients (in [0,1])
    tau_sup_obs = np.full((len(years_arr), N_TAU_SUP), np.nan, dtype=float)
    tau_sup_obs[:, 0] = flows_df["Refinery loss transfer coefficient"].to_numpy(float)
    tau_sup_obs[:, 1] = flows_df["Waelz transfer coefficient"].to_numpy(float)
    tau_sup_obs[:, 2] = flows_df["Old scrap transfer coefficient"].to_numpy(float)
    tau_sup_obs[:, 3] = flows_df["Dissipative use transfer coefficient"].to_numpy(float)
    tau_sup_obs[:, 4] = flows_df["New scrap (first use) transfer coefficient"].to_numpy(float)
    tau_sup_obs[:, 5] = flows_df["First use loss transfer coefficient"].to_numpy(float)
    tau_sup_obs[:, 6] = flows_df["New scrap (end use) transfer coefficient"].to_numpy(float)
    tau_sup_obs[:, 7] = flows_df["End use loss transfer coefficient"].to_numpy(float)

    # Observable annual-integrated flows for diagnostics & Stage-B flow loss.
    # We treat each Flows-sheet column reported in kt/yr as an annual integral
    # over (t_n, t_{n+1}], aligned with the closing year n+1 — same convention
    # as F_for_alpha above.  We map each to its semantic index in the
    # compute_flows_from_nn output vector (FLOW_NAMES).
    #
    # IMPORTANT (v4 vs v3):
    #   v3 only loaded 5 mandatory columns plus 5 optional ones, AND the
    #   optional column names were stale ("Old scrap recovery", "End-of-life
    #   flow", "Waelz recycling") so they silently failed to match the actual
    #   dataset headers.  As a result Stage-B flow loss, the diagnose() flow
    #   table, and plot_flows were all missing the manufacturing waste flows,
    #   the new-scrap flows, the EOL/old-scrap flow, the products-into-use
    #   flow, and the dissipative split.  v4 loads every flow that exists in
    #   FLOW_NAMES (the only one not in the dataset is `end_of_life_losses`
    #   which is purely derived).  All loads are guarded with `if column in
    #   flows_df.columns` so missing columns degrade gracefully.
    _flow_obs_specs_all = [
        # Mandatory primary flows
        ("Concentrate production (ILZSG)",          "concentrate_production"),
        ("Concentrate consumption",                  "concentrate_consumption"),
        ("Refined zinc consumption (ILZSG)",         "refined_consumption"),
        ("Waelz process input",                      "waelz_input"),
        ("Direct reuse recycling",                   "direct_reuse_recycling"),
        # Recycling / EOL chain
        ("End of life",                              "end_of_life"),
        ("Old scrap (i.e. separated EoL scrap)",     "old_scrap_recovery"),
        ("Waelz process recycling",                  "waelz_recycling"),
        ("Waelz losses",                             "waelz_losses"),
        # Refinery
        ("Primary Refining",                         "primary_refining"),
        ("Refinery losses",                          "refinery_losses"),
        # Manufacturing chain (NEW in v4)
        ("Total products into use",                  "total_products_into_use"),
        ("Non-dissipative products into use",        "inuse_inflow"),
        ("Dissipative use",                          "dissipative_use"),
        ("New scrap (first use)",                    "first_use_new_scrap"),
        ("First use manufacturing losses",           "first_use_losses"),
        ("New scrap (end use)",                      "end_use_new_scrap"),
        ("End use manufacturing losses",             "end_use_losses"),
    ]
    _flow_obs_specs = [(c, s) for (c, s) in _flow_obs_specs_all
                       if c in flows_df.columns]
    _missing_cols = [c for (c, _) in _flow_obs_specs_all if c not in flows_df.columns]
    if _missing_cols:
        print(f"[load_zinc_data] {len(_missing_cols)} flow column(s) absent from "
              f"sheet, skipped: {_missing_cols}")

    flow_obs_cols   = [s[0] for s in _flow_obs_specs]
    flow_obs_names  = [s[1] for s in _flow_obs_specs]
    flow_obs_to_pred_idx = [FLOW_NAMES.index(s[1]) for s in _flow_obs_specs]
    F_obs_full = np.stack(
        [flows_df[c].to_numpy(dtype=float) for c in flow_obs_cols], axis=1
    )                                       # (T, N_FLOW_OBS)
    flows_obs   = np.clip(F_obs_full[1:, :], 0.0, np.inf)  # (T-1, N_FLOW_OBS)

    return dict(
        years=years_arr,
        stocks_obs=S_obs,
        cp_obs=cp_obs,
        alpha_obs=alpha_obs,
        tau_sup_obs=tau_sup_obs,
        flows_obs=flows_obs,
        flow_obs_names=flow_obs_names,
        flow_obs_to_pred_idx=np.asarray(flow_obs_to_pred_idx, dtype=np.int32),
        exog_times=years_arr.copy(),
        exog_values=exog_values,
        exog_times_full=exog_times_full,
        exog_values_full=exog_values_full,
        exog_cols=exog_cols,
    )


# ============================================================================
# 8) Exogenous preprocessing (logging, detrending, finite differences)
# ============================================================================
def preprocess_exog(years_all, exog_values, years_train, *,
                    do_log1p=False, do_detrend=False, detrend_kind="linear",
                    feature_orders=(0,), diff_pad="edge",
                    years_source=None, exog_values_source=None):
    """Build the exogenous feature matrix: optional log1p + linear detrend on
    training years + concatenated [order-0, order-1 (Δ), order-2 (ΔΔ)] diffs.

    `years_source` lets pre-t0 history be used when computing diff-based
    features (so first-year diffs are not synthetic zeros)."""
    years_all   = np.asarray(years_all,   dtype=float)
    years_train = np.asarray(years_train, dtype=float)
    if years_source is None:
        years_source = years_all
        exog_values_source = exog_values
    years_source = np.asarray(years_source, dtype=float)
    X_src        = np.array(exog_values_source, dtype=float, copy=True)

    orders = tuple(sorted(set(int(o) for o in (feature_orders if
                              isinstance(feature_orders, (list, tuple, np.ndarray))
                              else [feature_orders]))))
    for o in orders:
        if o not in (0, 1, 2):
            raise ValueError(f"feature_orders elements must be in (0,1,2). Got {orders}")

    if do_log1p:
        X_src = np.log1p(np.maximum(X_src, 0.0))

    if do_detrend:
        idx_tr = np.searchsorted(years_source, years_train)
        idx_tr = np.clip(idx_tr, 0, max(len(years_source) - 1, 0))
        t_tr   = years_source[idx_tr]
        A      = np.column_stack([np.ones_like(t_tr), t_tr])
        beta   = np.zeros((2, X_src.shape[1]), dtype=float)
        for j in range(X_src.shape[1]):
            y    = X_src[idx_tr, j]
            mask = np.isfinite(y) & np.isfinite(t_tr)
            if mask.sum() < 2:
                continue
            beta[:, j], *_ = np.linalg.lstsq(A[mask], y[mask], rcond=None)
        trend = beta[0] + years_source[:, None] * beta[1]
        X_src -= trend

    target_idx = np.searchsorted(years_source, years_all)
    target_idx = np.clip(target_idx, 0, max(len(years_source) - 1, 0))

    n0    = X_src.shape[1]
    feats = []
    for o in orders:
        if o == 0:
            Xo_src = X_src
        else:
            D = np.diff(X_src, n=o, axis=0)
            if diff_pad == "edge" and D.shape[0] > 0:
                pad = np.repeat(D[:1], o, axis=0)
            else:
                pad = np.zeros((o, n0), dtype=float)
            Xo_src = np.vstack([pad, D])
        feats.append(Xo_src[target_idx])
    return np.concatenate(feats, axis=1) if len(feats) > 1 else feats[0]


# ============================================================================
# 9) ODE flows + RHS  (used only by Stage B)
# ============================================================================
def ode_to_4obs(S_ode):
    """Reduce the 6-element augmented stock vector to 4 observable stocks
    (sum the 3 cohort sub-stocks into S_inuse_total)."""
    S_inuse = jnp.sum(S_ode[..., 2:2 + N_COHORTS], axis=-1)
    return jnp.stack(
        [S_ode[..., IDX_CONC], S_ode[..., IDX_REF], S_inuse, S_ode[..., IDX_SCRAP]],
        axis=-1,
    )


def compute_flows_from_nn(nn_eval, params, t, S4, exog_t, data, S_cohorts=None):
    """Compute the full mass-conserving flow vector at (t, S4, exog_t).

    `nn_eval` is the layout-aware closure built by `make_nn_eval` — it
    transparently swaps in pinned (data-interpolated) values for cp / any
    τ / either manufacturing simplex according to the pin flags baked
    into the closure.

    `S_cohorts` of length N_COHORTS is the cohort sub-stock vector inside
    the ODE state (Stage B).  Stage A does not call this function.
    """
    out      = nn_eval(params, t, S4, exog_t, data)
    cp       = out["cp"]
    alphas   = out["alphas"]                 # (4,)  alpha_cc, alpha_refc, alpha_win, alpha_dr
    taus     = out["taus"]                   # (4,)  tau_ref, tau_waelz, tau_olds, tau_diss
    frac_fu  = out["frac_fu"]                # (3,)
    frac_eu  = out["frac_eu"]                # (3,)
    f_cohort = out["f_cohort"]               # (3,)

    # Effective stocks: clip negatives so flows can't pull from impossible reservoirs
    S4_eff = jnp.maximum(S4, 0.0)
    S_conc, S_ref, S_inuse, S_scrap = S4_eff
    alpha_cc, alpha_refc, alpha_win, alpha_dr = alphas
    tau_ref, tau_waelz, tau_olds, tau_diss    = taus

    cc   = alpha_cc   * S_conc
    refc = alpha_refc * S_ref
    w_in = alpha_win  * S_scrap
    dr   = alpha_dr   * S_scrap

    primary_refining = (1.0 - tau_ref)   * cc
    refinery_losses  = tau_ref           * cc
    waelz_recycling  = tau_waelz         * w_in
    waelz_losses     = (1.0 - tau_waelz) * w_in

    parent_manu          = refc + dr
    first_use_new_scrap  = frac_fu[0] * parent_manu
    first_use_losses     = frac_fu[1] * parent_manu
    fu_out               = frac_fu[2] * parent_manu

    end_use_new_scrap    = frac_eu[0] * fu_out
    end_use_losses       = frac_eu[1] * fu_out
    total_into_use       = frac_eu[2] * fu_out

    dissipative_use      = tau_diss         * total_into_use
    inuse_inflow         = (1.0 - tau_diss) * total_into_use

    # In-use cohort discharge (Rostek lifetime cohorts)
    if S_cohorts is None:
        # Steady-state cohort split (used only when called outside the ODE)
        phi  = f_cohort * MU_COHORTS
        phi  = phi / (jnp.sum(phi) + 1e-12)
        S_k  = phi * jnp.maximum(S_inuse, 0.0)
    else:
        S_k  = jnp.maximum(S_cohorts, 0.0)
    eol_k              = S_k / (MU_COHORTS + 1e-12)
    eol                = jnp.sum(eol_k)
    old_scrap_recovery = tau_olds         * eol
    eol_losses         = (1.0 - tau_olds) * eol

    # Pack flows (matches v10_4 ordering for plotting compatibility)
    flows = jnp.array([
        cp, cc, refc, w_in, dr, eol,
        primary_refining, refinery_losses,
        waelz_recycling,  waelz_losses,
        total_into_use, dissipative_use, inuse_inflow,
        first_use_new_scrap, first_use_losses,
        end_use_new_scrap,   end_use_losses,
        old_scrap_recovery,  eol_losses,
    ])
    return flows, f_cohort


FLOW_NAMES = [
    "concentrate_production",   # 0
    "concentrate_consumption",  # 1
    "refined_consumption",      # 2
    "waelz_input",              # 3
    "direct_reuse_recycling",   # 4
    "end_of_life",              # 5
    "primary_refining",         # 6
    "refinery_losses",          # 7
    "waelz_recycling",          # 8
    "waelz_losses",             # 9
    "total_products_into_use",  # 10
    "dissipative_use",          # 11
    "inuse_inflow",             # 12
    "first_use_new_scrap",      # 13
    "first_use_losses",         # 14
    "end_use_new_scrap",        # 15
    "end_use_losses",           # 16
    "old_scrap_recovery",       # 17
    "end_of_life_losses",       # 18
]
N_FLOWS = len(FLOW_NAMES)
IDX_INUSE_INFLOW = FLOW_NAMES.index("inuse_inflow")


def _make_Y0(S0_4):
    """Augmented IC: 6-stock vector + 19 zeroed flow accumulators."""
    S_cohorts_0 = jnp.array(IC_COHORT_FRACS) * S0_4[2]
    Y0_stocks = jnp.concatenate([
        S0_4[:2],         # S_conc, S_ref
        S_cohorts_0,      # S_iu_short, S_iu_med, S_iu_long
        S0_4[3:4],        # S_scrap
    ])
    return jnp.concatenate([Y0_stocks, jnp.zeros(N_FLOWS)])


def make_rhs(nn_eval):
    """Return an ODE RHS closure that uses `nn_eval` to compose NN outputs
    with pinned (data-interpolated) values.  cp pinning, τ pinning and
    manufacturing-simplex pinning are all handled inside `nn_eval` — the
    RHS itself is unconditional.
    """
    def rhs(Y, t, params, data):
        S_ode         = Y[:N_ODE_STOCKS]
        S_cohorts     = S_ode[2:2 + N_COHORTS]
        S_inuse_total = jnp.sum(S_cohorts)
        S4            = jnp.array([S_ode[IDX_CONC], S_ode[IDX_REF],
                                   S_inuse_total, S_ode[IDX_SCRAP]])

        exog_t  = exog_fn(t, data["exog_times"], data["exog_values"])

        flows, f_cohort = compute_flows_from_nn(
            nn_eval, params, t, S4, exog_t, data, S_cohorts=S_cohorts,
        )
        cp                 = flows[0]
        cc                 = flows[1]
        refc               = flows[2]
        w_in               = flows[3]
        dr                 = flows[4]
        primary_refining   = flows[6]
        waelz_recycling    = flows[8]
        inuse_inflow       = flows[IDX_INUSE_INFLOW]
        old_scrap_recovery = flows[17]
        first_use_new_scrap= flows[13]
        end_use_new_scrap  = flows[15]

        eol_k     = jnp.maximum(S_cohorts, 0.0) / (MU_COHORTS + 1e-12)
        dS_cohort = f_cohort * inuse_inflow - eol_k

        dS_conc  = cp - cc
        dS_ref   = primary_refining + waelz_recycling - refc
        dS_scrap = (old_scrap_recovery + first_use_new_scrap + end_use_new_scrap
                    - w_in - dr)

        dSdt = jnp.concatenate([
            jnp.array([dS_conc, dS_ref]),
            dS_cohort,
            jnp.array([dS_scrap]),
        ])
        return jnp.concatenate([dSdt, flows])
    return rhs


def _make_rhs_data(data):
    """Return a small dict containing only the data keys that the ODE RHS
    actually uses.  This keeps the diffrax adjoint cotangent state small
    (it doesn't have to track flows_obs / alpha_obs / etc. as cotangents)."""
    return {
        "years":       data["years"],
        "cp_obs":      data["cp_obs"],
        "tau_sup_obs": data["tau_sup_obs"],
        "exog_times":  data["exog_times"],
        "exog_values": data["exog_values"],
        "stats":       data["stats"],
    }


def make_integrator(integrator_kind, nn_eval, adjoint=None):
    """Build (and JIT) the augmented-state integrator.

    Returns a function with signature ``integrate(params, data, S0, years)``
    that returns ``(S_year, F_int, S_cohorts_year)``:

      * S_year         : (T, 4)              4-observable stocks (cohorts summed)
      * F_int          : (T-1, N_FLOWS)      yearly flow integrals
      * S_cohorts_year : (T, N_COHORTS)      per-cohort sub-stocks (from ODE state)

    The cohort sub-stocks are exposed for diagnostics — at ``t=years[0]``
    they equal ``IC_COHORT_FRACS * S0[2]`` by construction, so the
    cohort *stock-fraction* trajectory ``S_cohorts_year[t] /
    sum(S_cohorts_year[t], axis=-1)`` always starts at ``IC_COHORT_FRACS``.
    The Stage-B loss and other callers that only need the first two
    outputs can simply unpack and discard the third.

    Parameters
    ----------
    adjoint : diffrax adjoint instance or None
        Reverse-mode differentiation strategy handed to ``dfx.diffeqsolve``.
        ``None`` (the default) leaves diffrax's own default in place
        (``RecursiveCheckpointAdjoint``) — i.e. behaviour is unchanged for
        every existing v5 caller.  Downstream modules that hit adjoint
        instability with large rate constants (e.g. ``zinc_baseline``) can
        pass ``dfx.DirectAdjoint()``, which autodiffs through the discrete
        solver steps instead of solving the adjoint ODE.  Note that
        ``DirectAdjoint`` trades memory for stability: with ``max_steps``
        this large it can be substantially heavier than the default.
        Only meaningful for ``integrator_kind == "diffrax"``.
    """
    rhs = make_rhs(nn_eval)

    if integrator_kind == "diffrax":
        if not HAS_DIFFRAX:
            raise ImportError("diffrax not installed.  pip install diffrax")
        def integrate(params, data, S0, years):
            years = jnp.asarray(years)
            rhs_data = _make_rhs_data(data)
            def f(t, y, args):
                p, d = args
                return rhs(y, t, p, d)
            Y0  = _make_Y0(S0)
            sol = dfx.diffeqsolve(
                dfx.ODETerm(f), dfx.Tsit5(),
                t0=years[0], t1=years[-1],
                dt0=jnp.asarray(0.1, dtype=years.dtype),
                y0=Y0, args=(params, rhs_data),
                saveat=dfx.SaveAt(ts=years),
                stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-7),
                max_steps=20000,
                # Only pass `adjoint` when explicitly requested, so the
                # default code path is bit-identical to before this kwarg
                # existed (diffrax default = RecursiveCheckpointAdjoint).
                **({} if adjoint is None else {"adjoint": adjoint}),
            )
            Y      = sol.ys
            S_ode  = Y[:, :N_ODE_STOCKS]
            S_year = ode_to_4obs(S_ode)
            # Cohort sub-stocks live at indices 2..2+N_COHORTS in the ODE state
            S_cohorts_year = S_ode[:, 2:2 + N_COHORTS]
            C_year = Y[:, N_ODE_STOCKS:]
            F_int  = C_year[1:] - C_year[:-1]
            return S_year, F_int, S_cohorts_year
    else:
        if adjoint is not None:
            raise ValueError(
                "make_integrator: `adjoint` is only supported by the "
                "'diffrax' backend; got integrator_kind="
                f"{integrator_kind!r}.  Drop the argument or switch "
                "integrator to 'diffrax'."
            )
        def integrate(params, data, S0, years):
            rhs_data = _make_rhs_data(data)
            Y0 = _make_Y0(S0)
            Y  = odeint(rhs, Y0, years, params, rhs_data)
            S_ode  = Y[:, :N_ODE_STOCKS]
            S_year = ode_to_4obs(S_ode)
            S_cohorts_year = S_ode[:, 2:2 + N_COHORTS]
            C_year = Y[:, N_ODE_STOCKS:]
            F_int  = C_year[1:] - C_year[:-1]
            return S_year, F_int, S_cohorts_year

    return integrate


# ============================================================================
# 10) Stage A loss — collocation / observation fitting (no ODE)
# ============================================================================
def _smoothness_pen(path, mask=None, eps=1e-12):
    """First-order squared finite-difference penalty on `path` (T, F).

    Optional `mask` of shape (F,) zero-weights features that have no
    supervision (e.g. cohort fractions when not directly observed)."""
    if path.shape[0] < 2:
        return jnp.array(0.0)
    d1 = path[1:] - path[:-1]
    if mask is None:
        return jnp.mean(d1 ** 2)
    m = jnp.asarray(mask, dtype=d1.dtype)
    return jnp.sum((d1 ** 2) * m[None, :]) / (jnp.sum(m) * d1.shape[0] + eps)


def stageA_predict_paths(nn_eval_sup, params, data):
    """NN-predicted (cp, α, τ_sup_8, f_cohort, raw_norm) at integer years
    evaluated at observed stocks (teacher forcing)."""
    def _pred(t, S):
        ex = exog_fn(t, data["exog_times"], data["exog_values"])
        return nn_eval_sup(params, t, S, ex, data)
    cp_p, alpha_p, tau_p, fc_p, raw_p = jax.vmap(_pred)(
        data["years"], data["stocks_obs"]
    )
    return cp_p, alpha_p, tau_p, fc_p, jnp.mean(raw_p)


def make_stageA_loss(nn_eval_sup, layout):
    """Build a Stage A loss closure that knows which components are pinned.

    L_A = w_α  · Σ_k MSE(log α̂_k(t_mid),  log α_k_emp(t_mid))         [midpoint years]
        + w_τ  · Σ_j MSE(τ̂_j(t),         τ_j_obs(t))                  [year grid; learned only]
        + w_cp · MSE(log cp̂(t),          log cp_obs(t))               [year grid; learned only]
        + w_smooth_α · ‖Δ log α̂‖²    +   w_smooth_τ · ‖Δ logit τ̂‖²
        + w_cohort_prior · KL( prior  ‖  f̂_cohort )                   [optional, weak]
        + w_raw_reg · mean(raw²)
    """
    pin_cp      = layout["pin_cp"]
    pin_taus    = layout["pin_taus_tuple"]      # (4,) bool — binary τ pinning
    pin_fu_loss = layout["pin_fu_loss"]
    pin_eu_loss = layout["pin_eu_loss"]
    pin_fu_new  = layout["pin_fu_new"]          # NEW in v4
    pin_eu_new  = layout["pin_eu_new"]          # NEW in v4

    # Per-tau-sup-slot mask: True means LEARNED (NN-driven), False = pinned to data.
    # tau_sup layout (length 8):
    #   0..3 : binary τ_ref / τ_waelz / τ_olds / τ_diss
    #   4    : frac_fu_new                      — pinned iff pin_fu_new
    #   5    : frac_fu_loss                     — pinned iff pin_fu_loss
    #   6    : frac_eu_new                      — pinned iff pin_eu_new
    #   7    : frac_eu_loss                     — pinned iff pin_eu_loss
    learned_tau_mask = np.array(
        [not p for p in pin_taus]
        + [not pin_fu_new, not pin_fu_loss]
        + [not pin_eu_new, not pin_eu_loss],
        dtype=bool,
    )

    def stageA_loss(params, data, *,
                    w_alpha=1.0, w_tau=1.0, w_cp=1.0,
                    w_smooth_alpha=0.05, w_smooth_tau=0.05,
                    w_cohort_prior=0.0, w_raw_reg=1e-4):
        stats        = data["stats"]
        alpha_obs    = data["alpha_obs"]            # (T, 4) — row 0 NaN
        cp_obs       = data["cp_obs"]               # (T,)
        tau_sup_obs  = data["tau_sup_obs"]          # (T, 8)

        cp_p, alpha_p, tau_p, fc_p, raw_norm = stageA_predict_paths(
            nn_eval_sup, params, data
        )

        # ---- α loss (log-space, masked on NaN observations) ----
        log_alpha_p = _safe_log(alpha_p)
        log_alpha_o = _safe_log(alpha_obs)
        mask_alpha  = jnp.isfinite(alpha_obs) & jnp.isfinite(alpha_p)
        loss_alpha = _per_feature_masked_mse(
            log_alpha_p - log_alpha_o,
            stats["alpha_log_std"],
            weights=jnp.ones((N_ALPHAS,), dtype=alpha_p.dtype),
            mask=mask_alpha,
        )

        # ---- τ loss (raw probability space, masked on NaN AND on pinned components)
        mask_tau   = jnp.isfinite(tau_sup_obs) & jnp.asarray(learned_tau_mask)[None, :]
        learned_w  = jnp.asarray(learned_tau_mask, dtype=tau_p.dtype)
        loss_tau   = _per_feature_masked_mse(
            tau_p - tau_sup_obs,
            stats["tau_sup_std"],
            weights=learned_w,
            mask=mask_tau,
        )

        # ---- cp loss (log-space) — skipped if pinned ----
        if not pin_cp:
            mask_cp = jnp.isfinite(cp_obs)
            diff    = jnp.where(mask_cp, _safe_log(cp_p) - _safe_log(cp_obs), 0.0)
            denom   = jnp.maximum(stats["cp_log_std"], 1e-3)
            loss_cp = _masked_mean((diff / denom) ** 2, mask_cp)
        else:
            loss_cp = jnp.array(0.0)

        # ---- smoothness penalties (mild) ----
        loss_smooth_alpha = _smoothness_pen(log_alpha_p)
        loss_smooth_tau   = _smoothness_pen(_logit(tau_p))

        # ---- cohort prior: keep f_cohort near IC_COHORT_FRACS weakly ----
        if w_cohort_prior > 0.0:
            prior   = jnp.array(IC_COHORT_FRACS, dtype=fc_p.dtype)
            loss_coh = jnp.mean(jnp.sum(
                prior[None, :] * (jnp.log(prior[None, :] + 1e-12) - jnp.log(fc_p + 1e-12)),
                axis=-1,
            ))
        else:
            loss_coh = jnp.array(0.0)

        return (
            w_alpha          * loss_alpha
            + w_tau          * loss_tau
            + w_cp           * loss_cp
            + w_smooth_alpha * loss_smooth_alpha
            + w_smooth_tau   * loss_smooth_tau
            + w_cohort_prior * loss_coh
            + w_raw_reg      * raw_norm
        )

    def stageA_loss_terms(params, data):
        """Diagnostic decomposition of the Stage A loss into (α, τ, cp)."""
        stats        = data["stats"]
        alpha_obs    = data["alpha_obs"]
        cp_obs       = data["cp_obs"]
        tau_sup_obs  = data["tau_sup_obs"]

        cp_p, alpha_p, tau_p, _fc_p, _raw = stageA_predict_paths(
            nn_eval_sup, params, data
        )

        log_alpha_p = _safe_log(alpha_p)
        log_alpha_o = _safe_log(alpha_obs)
        mask_alpha  = jnp.isfinite(alpha_obs) & jnp.isfinite(alpha_p)
        loss_alpha  = _per_feature_masked_mse(
            log_alpha_p - log_alpha_o, stats["alpha_log_std"],
            weights=jnp.ones((N_ALPHAS,), dtype=alpha_p.dtype), mask=mask_alpha,
        )

        mask_tau = jnp.isfinite(tau_sup_obs) & jnp.asarray(learned_tau_mask)[None, :]
        learned_w = jnp.asarray(learned_tau_mask, dtype=tau_p.dtype)
        loss_tau = _per_feature_masked_mse(
            tau_p - tau_sup_obs, stats["tau_sup_std"],
            weights=learned_w, mask=mask_tau,
        )

        if not pin_cp:
            mask_cp = jnp.isfinite(cp_obs)
            diff    = jnp.where(mask_cp, _safe_log(cp_p) - _safe_log(cp_obs), 0.0)
            denom   = jnp.maximum(stats["cp_log_std"], 1e-3)
            loss_cp = _masked_mean((diff / denom) ** 2, mask_cp)
        else:
            loss_cp = jnp.array(0.0)

        return loss_alpha, loss_tau, loss_cp

    return stageA_loss, stageA_loss_terms


# ============================================================================
# 11) Stage B loss — shooting / endogenous-stock fine-tune
# ============================================================================
def make_stageB_loss(integrate_aug, nn_eval_sup, layout, flow_obs_to_pred_idx,
                     *, stock_loss_kind="log"):
    """Build a window-aware Stage B loss closure.

    Stage B
    -------
    1. Pick a starting index `start_idx` and window length `W`.
    2. Integrate the ODE from S_obs[start_idx] for W years using the NN as
       the dynamics model: S_pred (W, 4),  F_int (W-1, N_FLOWS).
    3. Evaluate the NN at the *predicted* stocks (this is the key difference
       from Stage A) and supervise it against the same α / τ / cp targets.
    4. Add a stock-divergence correction term  ‖S_pred - S_obs‖² / S_target_std²,
       which keeps the rollout near data.
    5. Optional flow-divergence term (w_F > 0): compare the ODE-integrated
       annual flow integrals  F_int[i, k] = ∫_{t_i}^{t_{i+1}} F_k ds  to the
       observed annual flows in `data["flows_obs"]`.  This is the physically
       correct objective: dataset flows are yearly integrals, not instantaneous.

    The α targets are at midpoint years (interval-defined); τ and cp are at
    integer years; flow integrals are interval-defined.
    """
    pin_cp           = layout["pin_cp"]
    pin_taus         = layout["pin_taus_tuple"]
    pin_fu_loss      = layout["pin_fu_loss"]
    pin_eu_loss      = layout["pin_eu_loss"]
    pin_fu_new       = layout["pin_fu_new"]      # NEW in v4
    pin_eu_new       = layout["pin_eu_new"]      # NEW in v4
    learned_tau_mask = np.array(
        [not p for p in pin_taus]
        + [not pin_fu_new, not pin_fu_loss]
        + [not pin_eu_new, not pin_eu_loss],
        dtype=bool,
    )

    def stageB_window_loss(
        params, data, start_idx, window_len, *,
        w_alpha=1.0, w_tau=1.0, w_cp=1.0, w_S=2.0, w_F=0.0,
        w_smooth_alpha=0.05, w_smooth_tau=0.05,
        w_cohort_prior=0.0, w_raw_reg=1e-4,
    ):
        years_full = data["years"]
        S_obs_full = data["stocks_obs"]
        tau_obs_full   = data["tau_sup_obs"]
        cp_obs_full    = data["cp_obs"]
        stats          = data["stats"]

        start_idx = jnp.asarray(start_idx, dtype=jnp.int32)
        years_w = jax.lax.dynamic_slice_in_dim(years_full, start_idx, window_len, axis=0)
        S_obs_w = jax.lax.dynamic_slice_in_dim(S_obs_full, start_idx, window_len, axis=0)
        t_obs_w = jax.lax.dynamic_slice_in_dim(tau_obs_full, start_idx, window_len, axis=0)
        c_obs_w = jax.lax.dynamic_slice_in_dim(cp_obs_full, start_idx, window_len, axis=0)

        # Integrate from observed IC at start_idx, returning predicted stocks
        # and the per-interval flow integrals F_int (W-1, N_FLOWS).
        # (cohort sub-stocks are also returned but ignored here.)
        S_pred_w, F_int_w, _S_coh_w = integrate_aug(params, data, S_obs_w[0], years_w)

        # NN evaluations on the predicted (endogenous) trajectory at year nodes
        def _pred(t, S):
            ex = exog_fn(t, data["exog_times"], data["exog_values"])
            return nn_eval_sup(params, t, S, ex, data)
        cp_p, alpha_p, tau_p, _fc_p, raw_p = jax.vmap(_pred)(years_w, S_pred_w)

        # ---- α loss at YEAR-END (Rostek convention; α_obs[i] aligned to year[i]) ----
        a_obs_full = data["alpha_obs"]                                       # (T, 4)
        a_obs_w    = jax.lax.dynamic_slice_in_dim(a_obs_full, start_idx, window_len, axis=0)
        log_alpha_p = _safe_log(alpha_p)
        log_alpha_o = _safe_log(a_obs_w)
        mask_a      = jnp.isfinite(a_obs_w) & jnp.isfinite(alpha_p)
        loss_alpha  = _per_feature_masked_mse(
            log_alpha_p - log_alpha_o, stats["alpha_log_std"],
            weights=jnp.ones((N_ALPHAS,), dtype=alpha_p.dtype), mask=mask_a,
        )

        # ---- τ loss (mask both NaNs and pinned components) ----
        mask_t   = jnp.isfinite(t_obs_w) & jnp.asarray(learned_tau_mask)[None, :]
        learned_w = jnp.asarray(learned_tau_mask, dtype=tau_p.dtype)
        loss_tau = _per_feature_masked_mse(
            tau_p - t_obs_w, stats["tau_sup_std"],
            weights=learned_w, mask=mask_t,
        )

        # ---- cp loss (skipped if pinned — ensures no leakage) ----
        if not pin_cp:
            mask_c = jnp.isfinite(c_obs_w)
            diff   = jnp.where(mask_c, _safe_log(cp_p) - _safe_log(c_obs_w), 0.0)
            denom  = jnp.maximum(stats["cp_log_std"], 1e-3)
            loss_cp = _masked_mean((diff / denom) ** 2, mask_c)
        else:
            loss_cp = jnp.array(0.0)

        # ---- Stock-divergence correction term — keeps rollout near data ----
        # Two flavours, selected at make-time via `stock_loss_kind`:
        #   "log" (default): per-feature MSE on  log S_pred − log S_obs,
        #                    scaled by per-stock log-std (training period).
        #                    This aligns the training objective with the
        #                    MAPE we report and prevents large stocks
        #                    (in-use) from dominating purely because of
        #                    scale, or near-stationary stocks from being
        #                    under-penalised because of small std.
        #   "std_scaled" (legacy v2): MSE on  S_pred − S_obs,  scaled by
        #                    per-stock training std.  Kept for back-compat.
        if stock_loss_kind == "log":
            log_S_pred = _safe_log(S_pred_w)
            log_S_obs  = _safe_log(S_obs_w)
            loss_S = _per_feature_mse(
                log_S_pred - log_S_obs,
                stats["S_log_std"],
                weights=stats["stock_term_weights"],
            )
        else:  # "std_scaled"
            loss_S = _per_feature_mse(
                S_pred_w - S_obs_w,
                stats["S_target_std"],
                weights=stats["stock_term_weights"],
            )

        # ---- Optional flow-divergence (yearly integrals from data) ----
        if w_F > 0.0:
            F_obs_full = data["flows_obs"]                                      # (T-1, K)
            f_idx      = jnp.asarray(flow_obs_to_pred_idx, dtype=jnp.int32)     # (K,) static
            F_obs_w    = jax.lax.dynamic_slice_in_dim(F_obs_full, start_idx, window_len - 1, axis=0)
            F_pred_w   = F_int_w[:, f_idx]                                      # (W-1, K)
            log_F_pred = _safe_log(F_pred_w)
            log_F_obs  = _safe_log(F_obs_w)
            mask_F     = jnp.isfinite(F_obs_w) & (F_obs_w > 0)
            denom_F    = jnp.maximum(stats["flow_log_std"], 0.05)
            diff_F     = jnp.where(mask_F, (log_F_pred - log_F_obs) / denom_F, 0.0)
            loss_F     = _masked_mean(diff_F ** 2, mask_F)
        else:
            loss_F = jnp.array(0.0)

        # ---- Mild smoothness penalties on the predicted parameter paths ----
        loss_smooth_alpha = _smoothness_pen(log_alpha_p)
        loss_smooth_tau   = _smoothness_pen(_logit(tau_p))

        # raw NN output regulariser (computed at predicted stocks)
        raw_norm = jnp.mean(raw_p)

        loss = (
            w_alpha          * loss_alpha
            + w_tau          * loss_tau
            + w_cp           * loss_cp
            + w_S            * loss_S
            + w_F            * loss_F
            + w_smooth_alpha * loss_smooth_alpha
            + w_smooth_tau   * loss_smooth_tau
            + w_raw_reg      * raw_norm
        )
        terms = dict(loss_alpha=loss_alpha, loss_tau=loss_tau, loss_cp=loss_cp,
                     loss_S=loss_S, loss_F=loss_F, raw_norm=raw_norm)
        return loss, terms

    return stageB_window_loss


# ============================================================================
# 12) Training loop — Stage A (always) + Stage B (optional)
# ============================================================================
def train_model(
    xlsx_path,
    *,
    seed=0,

    # ---------- inputs / NN architecture ----------
    use_time_input=False,
    use_stock_input=True,
    stock_norm_mode="tanh_log",     # "zscore" | "log" | "tanh_log"
    stock_log_scale=1.0,
    stock_ref_mode="mean",          # "mean" | "median" | "geomean"
    hidden_width=32,
    hidden_depth=2,

    # ---------- supervision targets / pin flags ----------
    # When pinned, the corresponding NN output slot is REMOVED from the head
    # (the network gets smaller) and the value is interpolated from data.
    learn_cp=False,                 # default: cp is well-observed → pin to data
    pin_tau_ref=False,              # binary tau pin flags
    pin_tau_waelz=False,
    pin_tau_olds=False,
    pin_tau_diss=False,
    pin_frac_fu_loss=False,         # pin frac_fu_loss only
    pin_frac_eu_loss=False,         # pin frac_eu_loss only
    pin_frac_fu_new=False,          # NEW in v4: pin frac_fu_new only
    pin_frac_eu_new=False,          # NEW in v4: pin frac_eu_new only
                                    # When both `pin_frac_*_new` and `pin_frac_*_loss`
                                    # are True for the same simplex the third
                                    # component (out / into-use) is fully
                                    # determined and the NN gets no slot for it.

    # ---------- exogenous features ----------
    extra_exog_cols=None,
    exog_log1p=True,
    exog_detrend=False,
    exog_detrend_kind="linear",
    exog_feature_orders=(0, 1),
    exog_diff_pad="edge",
    exog_pca_components=None,

    # ---------- data split ----------
    trainval_frac=0.7,
    val_frac=0.2,
    split_indices=None,             # NEW in v5: dict {"train_end", "val_end", "test_start"}
                                    # (0-based, inclusive end-points).  When given, overrides
                                    # trainval_frac/val_frac.  Used by rolling_origin_splits to
                                    # decouple held-out test from per-fold trainval boundary.

    # ---------- Stage A (collocation) ----------
    stageA_steps=3000,
    stageA_lr=3e-4,
    stageA_w_alpha=1.0,
    stageA_w_tau=1.0,
    stageA_w_cp=1.0,
    stageA_w_smooth_alpha=0.05,
    stageA_w_smooth_tau=0.05,
    stageA_w_cohort_prior=0.0,
    stageA_w_raw_reg=1e-4,

    # ---------- Stage B (optional shooting fine-tune) ----------
    do_stage_B=False,
    stageB_steps=2000,
    stageB_lr=1e-4,
    stageB_window=None,             # None = full pre-test horizon; int = fixed window
    stageB_curriculum=None,         # None or list[(steps, window)] — overrides stageB_steps/stageB_window
    stageB_batch_size=4,
    stageB_w_alpha=1.0,
    stageB_w_tau=1.0,
    stageB_w_cp=1.0,
    stageB_w_S=2.0,
    stageB_w_F=0.5,                 # NEW: log-space flow-divergence on yearly integrals
    stageB_w_smooth_alpha=0.05,
    stageB_w_smooth_tau=0.05,
    stageB_w_cohort_prior=0.0,
    stageB_w_raw_reg=1e-4,
    stock_loss_kind="log",          # "log" (default) | "std_scaled" (legacy v2)

    # ---------- final refit on train+val (after early stopping) ----------
    final_refit_on_trainval=True,
    final_refit_steps=400,                    # used only when curriculum=None
    final_refit_lr=3e-5,                      # base LR (decays 0.6**ci with curriculum)
    final_refit_curriculum=None,              # None ⇔ legacy single-window full-trainval refit;
                                              # otherwise list[(steps, window_len)] like
                                              # `stageB_curriculum`.  Each curriculum stage runs
                                              # `steps` SGD steps over windows of length
                                              # `window_len`, with LR = final_refit_lr * 0.6^ci.
    final_refit_batch_size=None,              # None ⇔ inherit `stageB_batch_size`

    # ---------- optimiser ----------
    weight_decay=1e-4,
    grad_clip=0.0,                  # 0 ⇔ disabled (default in v2; log-space losses + bounded
                                    # parametrisation make grad-norm clipping unnecessary)

    # ---------- early stopping (Stage B only) ----------
    early_stop=True,
    eval_every=200,
    patience=10,

    # ---------- ODE backend ----------
    integrator=None,                # "odeint" | "diffrax" — None = best available

    # ---------- weights for the stock-divergence term in Stage B ----------
    stock_term_weights=(1.0, 1.0, 1.0, 1.0),  # uniform default in v2

    # ---------- misc ----------
    verbose=True,
):
    """End-to-end fit on the zinc dataset.

    Returns
    -------
    params_final          : params after Stage B (or after Stage A if Stage B
                            was disabled).  Used as the "current best" weights.
    params_A              : a snapshot of params after Stage A finished but
                            BEFORE Stage B modified anything.  Used by the
                            FitResult to evaluate "Stage A predictions" with
                            the un-fine-tuned weights.
    layout                : NN head layout dict.
    data_train, data_val,
    data_trainval,
    data_test, data_all   : JAX-side data dicts for each split.
    integrate_aug         : the augmented-state ODE integrator (always built
                            even when Stage B is disabled — needed for
                            diagnostics and plot helpers).
    nn_eval, nn_eval_sup  : NN evaluation closures (layout-baked).
    flow_obs_to_pred_idx  : numpy array mapping observed-flow columns to
                            their indices in the full predicted-flow vector.
    """
    if integrator is None:
        integrator = "diffrax" if HAS_DIFFRAX else "odeint"

    # ---- 0) NN layout (which slots exist, given pin flags) ----
    layout = build_nn_layout(
        learn_cp=learn_cp,
        pin_tau_ref=pin_tau_ref, pin_tau_waelz=pin_tau_waelz,
        pin_tau_olds=pin_tau_olds, pin_tau_diss=pin_tau_diss,
        pin_frac_fu_loss=pin_frac_fu_loss,
        pin_frac_eu_loss=pin_frac_eu_loss,
        pin_frac_fu_new=pin_frac_fu_new,
        pin_frac_eu_new=pin_frac_eu_new,
    )
    nn_eval     = make_nn_eval(layout)
    nn_eval_sup = make_nn_eval_supervised(layout)

    # ---- 1) Load raw data ----
    data_np = load_zinc_data(xlsx_path, extra_exog_cols=extra_exog_cols)

    years_all   = data_np["years"]
    S_all       = data_np["stocks_obs"]
    cp_all      = data_np["cp_obs"]
    alpha_all   = data_np["alpha_obs"]
    tau_all     = data_np["tau_sup_obs"]
    exog_full_t = data_np["exog_times_full"]
    exog_full_v = data_np["exog_values_full"]
    T = len(years_all)

    # ---- 2) Train / val / test split on the *time axis* ----
    # Two paths:
    #  (a) split_indices=None ⇒ legacy v4 fraction-based split.  test starts
    #      immediately after trainval (no gap).
    #  (b) split_indices given ⇒ explicit absolute indices.  test_start is
    #      DECOUPLED from val_end so the held-out test window can stay fixed
    #      across folds in rolling-origin CV (val moves; test doesn't).
    if split_indices is None:
        if not (0.0 < trainval_frac < 1.0):
            raise ValueError("trainval_frac must be in (0,1).")
        n_trainval = min(max(int(np.floor(trainval_frac * T)), 3), T - 1)
        n_tr = n_trainval
        n_val = max(3, int(np.ceil(val_frac * n_tr)))
        cut   = max(n_tr - n_val, 2)
        # In legacy mode, test starts at the trainval boundary (no gap).
        test_ic_idx = max(n_trainval - 1, 0)
    else:
        if not isinstance(split_indices, dict) or "train_end" not in split_indices \
                or "val_end" not in split_indices:
            raise ValueError(
                "split_indices must be a dict with keys 'train_end' and 'val_end' "
                "(and optionally 'test_start')."
            )
        train_end_idx  = int(split_indices["train_end"])
        val_end_idx    = int(split_indices["val_end"])
        # default: test starts immediately after val (no gap, v4-compatible)
        test_start_idx = int(split_indices.get("test_start", val_end_idx + 1))

        if not (0 <= train_end_idx < val_end_idx < T):
            raise ValueError(
                f"split_indices must satisfy 0 <= train_end({train_end_idx}) "
                f"< val_end({val_end_idx}) < T({T})."
            )
        if test_start_idx < 1 or test_start_idx > T - 1:
            raise ValueError(
                f"split_indices['test_start']={test_start_idx} must be in [1, T-1]."
            )
        if test_start_idx <= val_end_idx:
            raise ValueError(
                f"split_indices['test_start']={test_start_idx} must be strictly > "
                f"val_end={val_end_idx} (else val and test overlap)."
            )

        n_trainval = val_end_idx + 1
        cut        = train_end_idx + 1
        n_val      = n_trainval - cut
        test_ic_idx = max(test_start_idx - 1, 0)

    years_trainval = years_all[:n_trainval]
    S_trainval     = S_all[:n_trainval]

    years_train = years_trainval[:cut]
    S_train     = S_trainval[:cut]
    cp_train    = cp_all[:cut]
    alpha_train = alpha_all[:cut]
    tau_train   = tau_all[:cut]

    years_val   = years_trainval[cut-1:]      # overlaps last train year for IC
    years_test  = years_all[test_ic_idx:]     # overlaps last pre-test year for IC

    if verbose:
        print(f"[split] T={T}  train={cut} ({int(years_train[0])}–{int(years_train[-1])})  "
              f"val={len(years_val)} ({int(years_val[0])}–{int(years_val[-1])})  "
              f"test={len(years_test)} ({int(years_test[0])}–{int(years_test[-1])})")

    # ---- 3) Exogenous preprocessing (fit on the training core) ----
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
            Xtr  = np.asarray(exog_proc[:cut], dtype=float)
            mu   = Xtr.mean(axis=0, keepdims=True)
            _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
            W   = Vt[:k].T
            exog_proc = (np.asarray(exog_proc, dtype=float) - mu) @ W
            if verbose:
                print(f"[exog PCA] compressed to {k} components")

    # ---- 4) Statistics computed on the train core only ----
    t_mean = years_train.mean()
    t_std  = max(float(years_train.std()), 1.0)

    S_mean = S_train.mean(axis=0)
    S_std  = np.maximum(S_train.std(axis=0),  1.0)

    exog_train = exog_proc[:cut]
    exog_mean  = exog_train.mean(axis=0)
    exog_std   = np.maximum(exog_train.std(axis=0), 1.0)

    S_target_std = np.maximum(S_train.std(axis=0), 1.0)

    # alpha scale (geometric mean per channel of empirical α on the train core)
    alpha_train_finite = np.where(np.isfinite(alpha_train), alpha_train, np.nan)
    log_alpha_train = np.log(np.maximum(alpha_train_finite, 1e-12))
    alpha_log_mean  = np.nanmean(log_alpha_train, axis=0)
    alpha_log_std   = np.maximum(np.nan_to_num(np.nanstd(log_alpha_train, axis=0),
                                               nan=1.0, posinf=1.0, neginf=1.0), 0.05)
    alpha_log_mean  = np.nan_to_num(alpha_log_mean, nan=0.0, posinf=0.0, neginf=0.0)
    alpha_scale     = np.exp(alpha_log_mean).astype(float)
    alpha_scale     = np.maximum(alpha_scale, 1e-6)

    # tau supervision std (per-feature) — train core.
    #   Some taus (e.g. tau_waelz) are nearly time-invariant, so their temporal
    #   std is tiny (~0.005).  Using that as the per-feature normaliser would
    #   make the τ loss dominate the others by orders of magnitude at init
    #   (random NN outputs ~0.5, observed τ_waelz ~0.92, residual ~0.42).
    #   We floor the per-feature scale at 0.10 so the τ loss is balanced
    #   with the α and cp terms at init.  This still preserves *relative*
    #   weighting between τ features once they are within ~0.1 of the data.
    tau_train_finite = np.where(np.isfinite(tau_train), tau_train, np.nan)
    tau_sup_std = np.maximum(np.nan_to_num(np.nanstd(tau_train_finite, axis=0),
                                           nan=0.10, posinf=0.10, neginf=0.10), 0.10)

    # cp scale (mean) and cp log-std (for the loss denominator)
    cp_train_finite = cp_train[np.isfinite(cp_train)]
    cp_scale = float(cp_train_finite.mean()) if cp_train_finite.size > 0 else 1.0
    if not np.isfinite(cp_scale) or cp_scale <= 0.0:
        cp_scale = 1.0
    cp_log_std = (float(np.std(np.log(np.maximum(cp_train_finite, 1e-12))))
                  if cp_train_finite.size > 1 else 1.0)
    cp_log_std = max(cp_log_std, 0.05)

    # stock reference for tanh_log normalisation
    if stock_ref_mode == "median":
        S_ref = np.maximum(np.median(S_train, axis=0), 1.0)
    elif stock_ref_mode == "geomean":
        S_ref = np.maximum(np.exp(np.mean(np.log(np.maximum(S_train, 1.0)),
                                          axis=0)), 1.0)
    else:
        S_ref = np.maximum(S_train.mean(axis=0), 1.0)

    stock_norm_mode = str(stock_norm_mode).lower().strip()
    _norm_mode_map = {"zscore": 0.0, "log": 1.0, "tanh_log": 2.0}
    if stock_norm_mode not in _norm_mode_map:
        raise ValueError(f"stock_norm_mode must be one of {list(_norm_mode_map)}.")

    stock_term_weights = np.asarray(stock_term_weights, dtype=float)
    if stock_term_weights.shape != (N_STOCKS,):
        raise ValueError(f"stock_term_weights must have shape ({N_STOCKS},).")

    stock_loss_kind = str(stock_loss_kind).lower().strip()
    if stock_loss_kind not in ("log", "std_scaled"):
        raise ValueError(
            f"stock_loss_kind must be 'log' or 'std_scaled', got {stock_loss_kind!r}."
        )

    # Per-flow log-std on training-period flow integrals (for Stage B w_F term)
    flows_all = data_np["flows_obs"]                      # (T-1, K)
    flows_train = flows_all[: max(cut - 1, 1)]            # rows up to (year cut-1, year cut]
    log_F = np.log(np.maximum(flows_train, 1e-6))
    log_F = np.where(np.isfinite(log_F), log_F, np.nan)
    flow_log_std = np.maximum(np.nan_to_num(np.nanstd(log_F, axis=0),
                                            nan=0.5, posinf=0.5, neginf=0.5), 0.05)

    # Per-stock log-std on training-period stocks (for the new log-stock loss).
    # Floored at 0.05 nat (~5% relative) so a stock that happens to be
    # near-constant on the training span doesn't tyrannise the loss.
    log_S_train = np.log(np.maximum(S_train, 1e-6))
    log_S_train = np.where(np.isfinite(log_S_train), log_S_train, np.nan)
    S_log_std = np.maximum(np.nan_to_num(np.nanstd(log_S_train, axis=0),
                                         nan=0.5, posinf=0.5, neginf=0.5), 0.05)

    stats = dict(
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
        stock_norm_mode=np.array(_norm_mode_map[stock_norm_mode], dtype=float),
        S_ref=S_ref.astype(float),
        S_log_scale=np.array(float(stock_log_scale), dtype=float),
        stock_term_weights=stock_term_weights.astype(float),
    )
    stats_j = {k: jnp.array(v) for k, v in stats.items()}

    # ---- 5) Pack JAX-side data dicts for each split ----
    flow_obs_to_pred_idx_np = np.asarray(data_np["flow_obs_to_pred_idx"], dtype=np.int32)

    def _pack(y_idx0, y_idx1, full=True):
        """Slice all observation arrays consistently.

        Stocks/cp/τ/α live on the year grid (T-axis).  flows_obs has shape
        (T-1, K); row i corresponds to the year-interval (years[i], years[i+1]],
        so for a year slice [y_idx0:y_idx1] (length L = y_idx1 - y_idx0) the
        consistent flow slice is [y_idx0 : y_idx1 - 1] of length L-1, where
        slice-relative flow row k aligns with year-pair (k, k+1) of the slice.

        flow_obs_to_pred_idx is a plain numpy int array (NOT inserted into the
        JAX data dict) — placing an integer array inside `data` would force
        odeint's reverse pass to compute float0 cotangents and crash.
        """
        sl      = slice(y_idx0, y_idx1)
        flow_sl = slice(y_idx0, max(y_idx1 - 1, y_idx0))
        return dict(
            years=jnp.array(years_all[sl]),
            stocks_obs=jnp.array(S_all[sl]),
            cp_obs=jnp.array(cp_all[sl]),
            alpha_obs=jnp.array(alpha_all[sl]),
            tau_sup_obs=jnp.array(tau_all[sl]),
            flows_obs=jnp.array(flows_all[flow_sl]),
            exog_times=jnp.array(years_all if full else years_all[sl]),
            exog_values=jnp.array(exog_proc if full else exog_proc[sl]),
            stats=stats_j,
        )
    data_train     = _pack(0,            cut,         full=False)
    data_val       = _pack(cut-1,        n_trainval,  full=False)
    data_trainval  = _pack(0,            n_trainval,  full=False)
    data_test      = _pack(test_ic_idx,  T,           full=False)
    data_all       = _pack(0,            T,           full=False)

    # ---- 6) Initialise the MLP (dynamic head width based on layout) ----
    n_exog    = exog_proc.shape[1]
    input_dim = 1 + N_STOCKS + n_exog
    layer_sizes = [input_dim] + [int(hidden_width)] * int(hidden_depth) + [layout["n_raw"]]
    params = init_mlp(layer_sizes, jax.random.PRNGKey(seed))

    # Warm-start the last-layer biases so the bounded NN outputs at init are
    # already near the observed means.  This costs nothing and removes a huge
    # chunk of training that would otherwise just be the head's bias terms
    # walking to the observed level.  Only LEARNED slots get warm-started; the
    # PINNED quantities don't have NN slots at all (the head is layout-sized).
    head = params[-1]
    new_b = np.array(head["b"])

    # cp slot: leave bias 0 → cp = cp_scale (= train mean) at init  (if learned)
    # alpha slots: leave bias 0 → α = α_scale_k (= train geom-mean) at init  (always learned)

    # binary τ slots (per-slot pinning)
    tau_means_obs = np.nanmean(tau_train, axis=0)[:N_TAUS_BINARY]
    tau_means_obs = np.clip(np.where(np.isfinite(tau_means_obs),
                                     tau_means_obs, 0.5), 1e-3, 1.0 - 1e-3)
    for j, slot in enumerate(layout["tau_slots"]):
        if slot is not None:
            new_b[slot] = np.log(tau_means_obs[j] / (1.0 - tau_means_obs[j]))

    # first-use simplex
    #   "full"        : warm-start the 3 logits at observed (new, loss, out) shares.
    #   "loss_pinned" : warm-start the single sigmoid logit at observed
    #                   p_new = new / (new + out)  (frac_fu_loss is pinned).
    #   "new_pinned"  : warm-start the single sigmoid logit at observed
    #                   p_loss = loss / (loss + out)  (frac_fu_new is pinned).
    #   "both_pinned" : NO NN slot to warm-start (both DOFs are pinned).
    fu_new_obs  = float(np.nanmean(tau_train[:, 4]))
    fu_loss_obs = float(np.nanmean(tau_train[:, 5]))
    fu_out_obs  = max(1.0 - fu_new_obs - fu_loss_obs, 1e-3)
    if layout["fu_kind"] == "full":
        fu_vec  = np.array([fu_new_obs, fu_loss_obs, fu_out_obs])
        fu_vec  = np.clip(fu_vec, 1e-3, 1.0 - 1e-3)
        fu_vec /= fu_vec.sum()
        new_b[layout["fu_logits_start"]:layout["fu_logits_start"] + N_LOGIT_FU] = np.log(fu_vec)
        fu_warm_descr = fu_vec.round(3).tolist()
    elif layout["fu_kind"] == "loss_pinned":
        p_new = fu_new_obs / max(fu_new_obs + fu_out_obs, 1e-6)
        p_new = float(np.clip(p_new, 1e-3, 1.0 - 1e-3))
        new_b[layout["fu_split_slot"]] = np.log(p_new / (1.0 - p_new))
        fu_warm_descr = {"p_new_given_remaining": round(p_new, 3)}
    elif layout["fu_kind"] == "new_pinned":
        p_loss = fu_loss_obs / max(fu_loss_obs + fu_out_obs, 1e-6)
        p_loss = float(np.clip(p_loss, 1e-3, 1.0 - 1e-3))
        new_b[layout["fu_split_slot"]] = np.log(p_loss / (1.0 - p_loss))
        fu_warm_descr = {"p_loss_given_remaining": round(p_loss, 3)}
    else:  # both_pinned
        fu_warm_descr = "(both DOFs pinned — no NN slot)"

    # end-use simplex
    eu_new_obs  = float(np.nanmean(tau_train[:, 6]))
    eu_loss_obs = float(np.nanmean(tau_train[:, 7]))
    eu_use_obs  = max(1.0 - eu_new_obs - eu_loss_obs, 1e-3)
    if layout["eu_kind"] == "full":
        eu_vec  = np.array([eu_new_obs, eu_loss_obs, eu_use_obs])
        eu_vec  = np.clip(eu_vec, 1e-3, 1.0 - 1e-3)
        eu_vec /= eu_vec.sum()
        new_b[layout["eu_logits_start"]:layout["eu_logits_start"] + N_LOGIT_EU] = np.log(eu_vec)
        eu_warm_descr = eu_vec.round(3).tolist()
    elif layout["eu_kind"] == "loss_pinned":
        p_new = eu_new_obs / max(eu_new_obs + eu_use_obs, 1e-6)
        p_new = float(np.clip(p_new, 1e-3, 1.0 - 1e-3))
        new_b[layout["eu_split_slot"]] = np.log(p_new / (1.0 - p_new))
        eu_warm_descr = {"p_new_given_remaining": round(p_new, 3)}
    elif layout["eu_kind"] == "new_pinned":
        p_loss = eu_loss_obs / max(eu_loss_obs + eu_use_obs, 1e-6)
        p_loss = float(np.clip(p_loss, 1e-3, 1.0 - 1e-3))
        new_b[layout["eu_split_slot"]] = np.log(p_loss / (1.0 - p_loss))
        eu_warm_descr = {"p_loss_given_remaining": round(p_loss, 3)}
    else:  # both_pinned
        eu_warm_descr = "(both DOFs pinned — no NN slot)"

    # cohort logits — always learned; align to IC_COHORT_FRACS (Rostek)
    coh_vec = np.clip(IC_COHORT_FRACS.astype(float), 1e-3, 1.0)
    coh_vec /= coh_vec.sum()
    new_b[layout["coh_start"]:layout["coh_start"] + N_LOGIT_COHORT] = np.log(coh_vec)

    head["b"] = jnp.array(new_b)
    params[-1] = head

    if verbose:
        pinned_summary = [
            ("cp",           not learn_cp),
            ("τ_ref",        pin_tau_ref),
            ("τ_waelz",      pin_tau_waelz),
            ("τ_olds",       pin_tau_olds),
            ("τ_diss",       pin_tau_diss),
            ("frac_fu_new",  pin_frac_fu_new),
            ("frac_fu_loss", pin_frac_fu_loss),
            ("frac_eu_new",  pin_frac_eu_new),
            ("frac_eu_loss", pin_frac_eu_loss),
        ]
        pinned_str = ",".join(n for n, p in pinned_summary if p) or "(none)"
        print(f"[model] input_dim={input_dim}  output_dim={layout['n_raw']}  "
              f"hidden={hidden_width}x{hidden_depth}")
        print(f"[model] alpha_scale={alpha_scale.round(3).tolist()}  "
              f"cp_scale={cp_scale:.0f}  pinned_to_data: {pinned_str}")
        print(f"[model] stock_loss_kind={stock_loss_kind!r}  "
              f"S_log_std={np.asarray(S_log_std).round(3).tolist()}  "
              f"(used in Stage B; w_S={stageB_w_S})")
        warm_msgs = [f"τ̄≈{tau_means_obs.round(3).tolist()}",
                     f"f_fu={fu_warm_descr}",
                     f"f_eu={eu_warm_descr}",
                     f"f_coh={coh_vec.round(3).tolist()}"]
        print(f"[model] head bias warm-start  " + "  ".join(str(m) for m in warm_msgs))

    # ---- 7) Stage A — collocation / observation fitting ----
    def _make_optimiser(total_steps, lr_init, p):
        sched = optax.cosine_decay_schedule(init_value=lr_init,
                                            decay_steps=max(int(total_steps), 1),
                                            alpha=0.1)
        steps = []
        if grad_clip and grad_clip > 0.0:
            steps.append(optax.clip_by_global_norm(grad_clip))
        steps.append(optax.adamw(learning_rate=sched, weight_decay=weight_decay))
        opt = optax.chain(*steps)
        return opt, opt.init(p)

    stageA_loss, stageA_loss_terms = make_stageA_loss(nn_eval_sup, layout)

    optA, optA_state = _make_optimiser(stageA_steps, stageA_lr, params)

    @jax.jit
    def stageA_step(params, opt_state, data):
        loss_val, grads = jax.value_and_grad(stageA_loss)(
            params, data,
            w_alpha=stageA_w_alpha, w_tau=stageA_w_tau, w_cp=stageA_w_cp,
            w_smooth_alpha=stageA_w_smooth_alpha,
            w_smooth_tau=stageA_w_smooth_tau,
            w_cohort_prior=stageA_w_cohort_prior,
            w_raw_reg=stageA_w_raw_reg,
        )
        updates, opt_state = optA.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss_val

    @jax.jit
    def stageA_terms(params, data):
        return stageA_loss_terms(params, data)

    if verbose:
        print(f"\n[Stage A] collocation fit on observations  ({stageA_steps} steps)…")
    t_a0 = time.time()
    for k in range(int(stageA_steps)):
        params, optA_state, loss_val = stageA_step(params, optA_state, data_train)
        if verbose and (k % max(int(eval_every), 1) == 0 or k == stageA_steps - 1):
            loss_val.block_until_ready()
            l_a, l_t, l_c = stageA_terms(params, data_train)
            l_a_v, l_t_v, l_c_v = stageA_terms(params, data_val)
            print(f"  Stage A step {k:5d}: "
                  f"train α={float(l_a):.3e} τ={float(l_t):.3e} cp={float(l_c):.3e} "
                  f"| val α={float(l_a_v):.3e} τ={float(l_t_v):.3e} cp={float(l_c_v):.3e} "
                  f"| L={float(loss_val):.3e}")
    if verbose:
        print(f"[Stage A] done in {time.time()-t_a0:.1f}s")

    # Snapshot the Stage-A weights BEFORE Stage B may modify them.
    # JAX pytrees are immutable so a reference copy is sufficient — the next
    # Stage-B optimiser produces NEW pytrees, leaving `params_A` untouched.
    params_A = jax.tree_util.tree_map(lambda x: x, params)

    # Build the integrator unconditionally — diagnostics, plots, and the
    # Stage-A "ODE free-run with NN_A" view all need it.  When `do_stage_B`
    # is False we never gradient-descend through the integrator, so
    # constructing it is essentially free (just JIT compilation).
    integrate_aug = make_integrator(integrator, nn_eval)

    # If Stage B is disabled we are finished.  We still return the integrator
    # so the FitResult can use it for the Stage-A free-run diagnostics.
    if not do_stage_B:
        return (params, params_A, layout, data_train, data_val, data_trainval,
                data_test, data_all, integrate_aug, nn_eval, nn_eval_sup,
                flow_obs_to_pred_idx_np)

    # ---- 8) Stage B — shooting fine-tune ----
    stageB_window_loss = make_stageB_loss(integrate_aug, nn_eval_sup, layout,
                                          flow_obs_to_pred_idx_np,
                                          stock_loss_kind=stock_loss_kind)

    # Resolve curriculum
    if stageB_curriculum is None:
        if stageB_window is None:
            stageB_curriculum = [(int(stageB_steps), int(len(years_train)))]
        else:
            stageB_curriculum = [(int(stageB_steps), int(stageB_window))]

    rng = np.random.default_rng(seed + 7)
    best_params = params
    best_val    = float("inf")
    best_step   = -1

    # Validation metric: shoot once from the start of val for the full val span.
    # `val_window_len` is fixed for the whole of Stage B, so we close over it
    # in the jit (it must be static for `dynamic_slice_in_dim`'s slice_size).
    val_window_len = int(data_val["years"].shape[0])

    @jax.jit
    def stageB_full_eval_loss(params, data):
        loss, _ = stageB_window_loss(
            params, data, jnp.array(0, dtype=jnp.int32), val_window_len,
            w_alpha=stageB_w_alpha, w_tau=stageB_w_tau, w_cp=stageB_w_cp,
            w_S=stageB_w_S, w_F=stageB_w_F,
            w_smooth_alpha=0.0, w_smooth_tau=0.0,
            w_cohort_prior=0.0, w_raw_reg=0.0,
        )
        return loss

    for ci, (n_steps_ci, win_ci) in enumerate(stageB_curriculum):
        n_steps_ci = int(n_steps_ci)
        win_ci     = min(int(win_ci), int(data_trainval["years"].shape[0]))
        if win_ci < 2:
            print(f"[Stage B / stage {ci}] window {win_ci} too short — skipping.")
            continue

        max_start = int(data_trainval["years"].shape[0]) - win_ci
        if max_start < 0:
            max_start = 0

        # Fresh optimizer per curriculum stage with decaying base LR
        stage_lr = stageB_lr * (0.6 ** ci)
        optB, optB_state = _make_optimiser(n_steps_ci, stage_lr, params)

        @jax.jit
        def stageB_step(params, opt_state, data, starts):
            def per_start_loss(p, s):
                l, _ = stageB_window_loss(
                    p, data, s, win_ci,
                    w_alpha=stageB_w_alpha, w_tau=stageB_w_tau, w_cp=stageB_w_cp,
                    w_S=stageB_w_S, w_F=stageB_w_F,
                    w_smooth_alpha=stageB_w_smooth_alpha,
                    w_smooth_tau=stageB_w_smooth_tau,
                    w_cohort_prior=stageB_w_cohort_prior,
                    w_raw_reg=stageB_w_raw_reg,
                )
                return l

            def total_loss(p):
                return jnp.mean(jax.vmap(lambda s: per_start_loss(p, s))(starts))

            loss_val, grads = jax.value_and_grad(total_loss)(params)
            updates, opt_state = optB.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss_val

        if verbose:
            print(f"\n[Stage B / stage {ci}] window={win_ci}  steps={n_steps_ci}  "
                  f"lr={stage_lr:.2e}  max_start={max_start}")

        bad = 0
        is_final = (ci == len(stageB_curriculum) - 1)
        es_active = bool(early_stop) and is_final and (data_val["years"].shape[0] >= 2)
        # (val_window_len is closed over by stageB_full_eval_loss above)

        t_b0 = time.time()
        for k in range(n_steps_ci):
            starts = rng.integers(0, max_start + 1, size=int(stageB_batch_size))
            starts_j = jnp.asarray(starts, dtype=jnp.int32)
            params, optB_state, loss_val = stageB_step(
                params, optB_state, data_trainval, starts_j,
            )

            if verbose and (k % max(int(eval_every), 1) == 0 or k == n_steps_ci - 1):
                loss_val.block_until_ready()
                v_loss = float(stageB_full_eval_loss(params, data_val))
                print(f"  Stage B step {k:5d}: train={float(loss_val):.3e}  "
                      f"val_freerun={v_loss:.3e}")
                if es_active:
                    if v_loss < best_val * (1.0 - 1e-4):
                        best_val, best_params, best_step, bad = v_loss, params, k, 0
                    else:
                        bad += 1
                        if bad >= patience:
                            print(f"  Early stopping at step {k}  best_val={best_val:.3e}")
                            params = best_params
                            break

        if verbose:
            print(f"[Stage B / stage {ci}] done in {time.time()-t_b0:.1f}s")

    if early_stop and best_step >= 0 and do_stage_B:
        if verbose:
            print(f"[Stage B] restoring best params from step {best_step} "
                  f"(val={best_val:.3e})")
        params = best_params

    # ---- 9) Optional final refit on train+val combined ----
    if final_refit_on_trainval and do_stage_B:
        # Resolve the refit curriculum.  Two modes:
        #   (a) curriculum=None   → legacy "single window = full trainval"
        #                            replay loop, `final_refit_steps` steps.
        #   (b) curriculum given  → Stage-B-style (steps, window) curriculum
        #                            with batched random starts and LR decay
        #                            0.6**ci across stages.
        T_trainval = int(data_trainval["years"].shape[0])
        if final_refit_curriculum is None:
            refit_curr = [(int(final_refit_steps), int(T_trainval))]
            curr_origin = "legacy single-window"
        else:
            refit_curr = [(int(s), int(w)) for (s, w) in final_refit_curriculum]
            curr_origin = "user-supplied curriculum"
        refit_batch = int(final_refit_batch_size) if final_refit_batch_size is not None \
            else int(stageB_batch_size)

        if verbose:
            print(f"\n[Final refit] continuing on train+val (no early stopping)  "
                  f"{curr_origin}  base_lr={final_refit_lr:.2e}  "
                  f"batch={refit_batch}")
            print(f"               curriculum (steps, window) = {refit_curr}")

        for ci, (n_steps_ci, win_ci) in enumerate(refit_curr):
            n_steps_ci = int(n_steps_ci)
            win_ci     = min(int(win_ci), T_trainval)
            if win_ci < 2:
                if verbose:
                    print(f"[Final refit / stage {ci}] window {win_ci} too short — skipping.")
                continue

            max_start = max(T_trainval - win_ci, 0)
            stage_lr  = float(final_refit_lr) * (0.6 ** ci)
            optF, optF_state = _make_optimiser(n_steps_ci, stage_lr, params)

            # Build a fresh JIT'd step closure per curriculum stage so the
            # window length and batch dimension are statically known to JAX.
            @jax.jit
            def stageF_step(params, opt_state, data, starts, _win=win_ci):
                def per_start_loss(p, s):
                    l, _ = stageB_window_loss(
                        p, data, s, _win,
                        w_alpha=stageB_w_alpha, w_tau=stageB_w_tau, w_cp=stageB_w_cp,
                        w_S=stageB_w_S, w_F=stageB_w_F,
                        w_smooth_alpha=stageB_w_smooth_alpha,
                        w_smooth_tau=stageB_w_smooth_tau,
                        w_cohort_prior=stageB_w_cohort_prior,
                        w_raw_reg=stageB_w_raw_reg,
                    )
                    return l
                def total_loss(p):
                    return jnp.mean(jax.vmap(lambda s: per_start_loss(p, s))(starts))
                loss_val, grads = jax.value_and_grad(total_loss)(params)
                updates, opt_state = optF.update(grads, opt_state, params)
                return optax.apply_updates(params, updates), opt_state, loss_val

            if verbose:
                print(f"\n[Final refit / stage {ci}] window={win_ci}  "
                      f"steps={n_steps_ci}  lr={stage_lr:.2e}  "
                      f"max_start={max_start}")

            t_f0 = time.time()
            for k in range(n_steps_ci):
                if max_start == 0:
                    starts = np.zeros((refit_batch,), dtype=np.int32)
                else:
                    starts = rng.integers(0, max_start + 1,
                                          size=int(refit_batch))
                starts_j = jnp.asarray(starts, dtype=jnp.int32)
                params, optF_state, loss_val = stageF_step(
                    params, optF_state, data_trainval, starts_j,
                )
                if verbose and (k % max(int(eval_every), 1) == 0
                                or k == n_steps_ci - 1):
                    loss_val.block_until_ready()
                    print(f"  Final refit / stage {ci} step {k:5d}: "
                          f"trainval={float(loss_val):.3e}")
            if verbose:
                print(f"[Final refit / stage {ci}] done in "
                      f"{time.time() - t_f0:.1f}s")

    return (params, params_A, layout, data_train, data_val, data_trainval,
            data_test, data_all, integrate_aug, nn_eval, nn_eval_sup,
            flow_obs_to_pred_idx_np)


# ============================================================================
# 13) Evaluation API — FitResult class with diagnose / plot methods
# ============================================================================
# v3 stores BOTH params_A (Stage-A weights, frozen before Stage B) and
# params_B = params (post-Stage-B weights, or = params_A if Stage B disabled).
# Stage-A diagnostics use NN_A; Stage-B diagnostics use NN_B.  In particular,
# stocks and flows are evaluated by ODE-integrating the corresponding NN —
# never by teacher-forcing — so the comparison "Stage A vs Stage B" measures
# closed-loop dynamics quality on a level playing field.

def _predict_paths_at(nn_eval, params, data, *, integrate_aug, layout=None):
    """ODE-integrate the augmented state from S_obs[0] using the NN closure
    `nn_eval` parameterised by `params`, then evaluate the NN at every
    observed-year time-stamp at the *predicted* (free-run) stocks.

    Parameters
    ----------
    layout : dict or None
        If provided, the at-observed-stocks evaluations of *pinned*
        quantities (cp, individual binary τ's, frac_*_loss) are overridden
        with the exact data lookups so that the diagnostic identity
        cp_at_obs ≡ cp_obs holds tautologically.  Without this override
        the year-0 boundary in `_interval_pick` (clip(1, N-1)) leaks a
        ~0.04 % off-by-one into the diagnostic, which is purely cosmetic
        (the ODE integral over (years[0], years[1]] is unaffected) but
        easy to mistake for a model error.  When `layout=None` the legacy
        behaviour (NN as-evaluated, including the boundary leak) is kept
        for back-compat.

    Returns
    -------
    dict with arrays:
        S_pred   : (T, 4)        ODE-integrated stocks (4 observable)
        F_int    : (T-1, N_FLOWS) ODE-accumulated yearly flow integrals
        cp       : (T,)
        alphas   : (T, 4)
        taus     : (T, 4)        binary taus
        frac_fu  : (T, 3)
        frac_eu  : (T, 3)
        f_cohort : (T, 3)
        years    : (T,)
        # Also evaluated AT OBSERVED STOCKS, for parameter-identification
        # diagnostics (these don't depend on dynamics quality):
        cp_at_obs, alphas_at_obs, taus_at_obs, frac_fu_at_obs,
        frac_eu_at_obs, f_cohort_at_obs.
    """
    years = data["years"]
    S_obs = data["stocks_obs"]
    S_pred, F_int, S_cohorts_pred = integrate_aug(params, data, S_obs[0], years)

    def _pred(t, S):
        ex = exog_fn(t, data["exog_times"], data["exog_values"])
        out = nn_eval(params, t, S, ex, data)
        return (out["cp"], out["alphas"], out["taus"],
                out["frac_fu"], out["frac_eu"], out["f_cohort"])

    cp_p, a_p, t_p, ffu, feu, fc = jax.vmap(_pred)(years, S_pred)
    cp_o, a_o, t_o, ffu_o, feu_o, fc_o = jax.vmap(_pred)(years, S_obs)

    # ----- at_obs override for pinned quantities --------------------------
    # When a quantity is pinned, the "NN evaluation" at observed time IS the
    # data lookup by construction.  We replace the result of `_interp_*` —
    # which uses the v10_2 piecewise-constant convention with clip(1, N-1)
    # giving an off-by-one at year 0 — with the data array itself, so the
    # diagnostic shows true identity (MAPE = 0.000) for pinned components.
    if layout is not None:
        cp_o = jnp.where(layout["pin_cp"],
                         jnp.asarray(data["cp_obs"]), cp_o)
        # Binary τ's: layout["pin_taus_tuple"] is a 4-tuple of booleans
        pin_taus = layout["pin_taus_tuple"]
        for j in range(4):
            if bool(pin_taus[j]):
                t_o = t_o.at[:, j].set(data["tau_sup_obs"][:, j])
        # Manufacturing simplex slots:
        #   tau_sup col 4 = frac_fu_new,  col 5 = frac_fu_loss
        #   tau_sup col 6 = frac_eu_new,  col 7 = frac_eu_loss
        # In the (frac_fu) 3-vec, index 0 = new, index 1 = loss, index 2 = out.
        if bool(layout["pin_fu_new"]):
            ffu_o = ffu_o.at[:, 0].set(data["tau_sup_obs"][:, 4])
        if bool(layout["pin_fu_loss"]):
            ffu_o = ffu_o.at[:, 1].set(data["tau_sup_obs"][:, 5])
        if bool(layout["pin_eu_new"]):
            feu_o = feu_o.at[:, 0].set(data["tau_sup_obs"][:, 6])
        if bool(layout["pin_eu_loss"]):
            feu_o = feu_o.at[:, 1].set(data["tau_sup_obs"][:, 7])

    return dict(
        S_pred=np.asarray(S_pred),
        F_int=np.asarray(F_int),
        S_cohorts=np.asarray(S_cohorts_pred),    # (T, N_COHORTS) — ODE sub-stocks
        cp=np.asarray(cp_p),
        alphas=np.asarray(a_p),
        taus=np.asarray(t_p),
        frac_fu=np.asarray(ffu),
        frac_eu=np.asarray(feu),
        f_cohort=np.asarray(fc),
        cp_at_obs=np.asarray(cp_o),
        alphas_at_obs=np.asarray(a_o),
        taus_at_obs=np.asarray(t_o),
        frac_fu_at_obs=np.asarray(ffu_o),
        frac_eu_at_obs=np.asarray(feu_o),
        f_cohort_at_obs=np.asarray(fc_o),
        years=np.asarray(years),
    )


# ----------------------------------------------------------------------------
# Per-component error metrics
# ----------------------------------------------------------------------------
def _logmae(pred, obs, eps=1e-12):
    pred = np.asarray(pred); obs = np.asarray(obs)
    m    = np.isfinite(pred) & np.isfinite(obs) & (obs > 0) & (pred > 0)
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(np.log(np.maximum(pred[m], eps))
                                - np.log(np.maximum(obs[m],  eps)))))


def _rel_rmse_pct(pred, obs, eps=1e-12):
    """Relative RMSE expressed as a percentage:
        100% × sqrt(mean((pred-obs)^2)) / mean(|obs|).
    """
    pred = np.asarray(pred); obs = np.asarray(obs)
    m    = np.isfinite(pred) & np.isfinite(obs)
    if not m.any():
        return float("nan")
    rmse = float(np.sqrt(np.mean((pred[m] - obs[m]) ** 2)))
    den  = float(np.mean(np.abs(obs[m])))
    if den < eps:
        return float("nan")
    return 100.0 * rmse / den


def _mape_pct(pred, obs, eps=1e-9):
    pred = np.asarray(pred); obs = np.asarray(obs)
    m    = np.isfinite(pred) & np.isfinite(obs) & (np.abs(obs) > eps)
    if not m.any():
        return float("nan")
    return 100.0 * float(np.mean(np.abs((pred[m] - obs[m]) / obs[m])))


def _nanmean_safe(values):
    """Mean of finite values; NaN if no finite values."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def _fmt_pct(v, fmt="{:.3f}%"):
    """Format a percentage value, returning '—' for NaN/inf."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(v):
        return "—"
    return fmt.format(v)


class FitResult:
    """Trained model + datasets, with diagnose / plot methods.

    Attributes
    ----------
    params         : final model parameters after the LAST training stage
                     (= params_A if Stage B was disabled, else = params_B).
    params_A       : Stage-A weights, snapshotted before Stage B started.
    params_B       : Stage-B weights — same object as `params` when Stage B
                     ran, else None (i.e. Stage A only).
    layout         : NN head layout dict (which slots exist, pin flags).
    cfg            : the full config used for the run.
    data_train, data_val, data_trainval, data_test, data_all
                   : JAX-side data dicts.
    integrate_aug  : ODE integrator closure (always built).
    nn_eval        : NN evaluation closure (params, t, S, exog, data) → dict.
    nn_eval_sup    : same but returns the supervised tuple.
    name           : run name (for plot titles).

    Public methods
    --------------
    diagnose(include_pinned=False, save_csv=None, verbose=True)
        Per-component logMAE / %RMSE / MAPE table, plus per-family averages,
        for both stages.  Pinned components are hidden by default.
    summary(include_pinned=False, prefer="auto")
        Compact family-average MAPE printout (alphas / taus / stocks / flows)
        on the **TEST SET only** plus a one-line ‖θ_B − θ_A‖₂ / ‖θ_A‖₂
        summary that answers "did Stage B do anything?".  `prefer="auto"`
        picks Stage B if it ran.  For per-component metrics on every split,
        call ``fit.diagnose()``.
    params_distance(verbose=False)
        Per-leaf and total L2 distance between Stage-A and Stage-B
        parameter snapshots.  Returns a dict; set verbose=True to print
        the top-5 layers that moved most.
    predictions(stage="A")
        Return the dict of ODE-integrated trajectories + NN evaluations
        (S_pred, F_int, cp, alphas, taus, frac_*, f_cohort, *_at_obs)
        for the requested stage.  Stage A predictions use the *snapshotted*
        Stage-A weights, so they remain valid even after Stage B ran.
        For *pinned* quantities, `*_at_obs` is overridden to the data
        lookup so the diagnostic is exactly tautological (MAPE = 0).
    save_predictions(path, stage="A")
        Save the prediction arrays for one stage to a .npz archive.
    plot_alphas / plot_taus / plot_stocks / plot_flows / plot_cohorts / plot_all
        All accept `include_pinned=False` (default) and `save_dir=None`.
        Stock and flow plots overlay BOTH stages' free-run trajectories.
        `plot_cohorts` shows all three (unobserved) cohort fractions on a
        single panel (PINN-style), with the IC prior at 1980 marked.
    """

    def __init__(self, *, name, cfg, params, params_A, layout,
                 data_train, data_val, data_trainval, data_test, data_all,
                 integrate_aug, nn_eval, nn_eval_sup, flow_obs_to_pred_idx,
                 do_stage_B):
        self.name           = name
        self.cfg            = cfg
        self.params         = params
        self.params_A       = params_A
        self.params_B       = params if do_stage_B else None
        self.layout         = layout
        self.data_train     = data_train
        self.data_val       = data_val
        self.data_trainval  = data_trainval
        self.data_test      = data_test
        self.data_all       = data_all
        self.integrate_aug  = integrate_aug
        self.nn_eval        = nn_eval
        self.nn_eval_sup    = nn_eval_sup
        self.flow_obs_to_pred_idx = np.asarray(flow_obs_to_pred_idx, dtype=int)
        self._stage_B_ran   = bool(do_stage_B)

        # Per-stage prediction cache, populated lazily by diagnose / plotting.
        self._cache = {}                           # mode -> _predict_paths_at output

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _params_for(self, mode):
        if mode == "A":
            return self.params_A
        elif mode == "B":
            return self.params_B if self.params_B is not None else self.params_A
        else:
            raise ValueError(f"mode must be 'A' or 'B', got {mode!r}")

    def _stages(self):
        """List of stages to evaluate: ['A','B'] if Stage B ran, else ['A']."""
        return ["A", "B"] if self._stage_B_ran else ["A"]

    def _predictions(self, mode):
        """Cached _predict_paths_at output for stage `mode`.

        Free-run: integrates over `data_all` starting from S_obs[0] (1980).
        Errors compound across the entire 1980-to-2019 horizon, so the test
        split metrics from this run are an upper bound on what the model
        could achieve if launched fresh from observed test-set IC.
        """
        if mode not in self._cache:
            self._cache[mode] = _predict_paths_at(
                self.nn_eval, self._params_for(mode), self.data_all,
                integrate_aug=self.integrate_aug,
                layout=self.layout,
            )
        return self._cache[mode]

    def _predictions_testrun(self, mode):
        """Cached test-run prediction: integrate over `data_test` only,
        starting from the observed stocks at the LAST VALIDATION YEAR
        (year `n_trainval - 1`, ≈ 2006), and integrate forward through
        the test window.  This is the v1-style "next-N-year projection
        from the trainval/test boundary" rollout — much shorter than the
        freerun, so accumulated error is bounded by the test horizon
        length.  Numerically comparable to v1's ``stock_relRMSE_freerun``.

        ``data_test`` is constructed in ``train_model`` as
        ``_pack(n_trainval-1, T)``, so its first row IS the last
        validation year and ``data_test["stocks_obs"][0]`` is exactly the
        IC the integrator wants.

        Returns the same dict shape as ``_predictions`` but the arrays are
        sized over the test slice (length T_test = T - n_trainval + 1)
        and metrics derived from them are computed AFTER excluding the
        IC row (which corresponds to the validation boundary, not a test
        point).
        """
        cache_key = f"testrun_{mode}"
        if cache_key not in self._cache:
            self._cache[cache_key] = _predict_paths_at(
                self.nn_eval, self._params_for(mode), self.data_test,
                integrate_aug=self.integrate_aug,
                layout=self.layout,
            )
        return self._cache[cache_key]

    # ------------------------------------------------------------------
    # public accessors — Stage A and Stage B trajectories / parameters
    # ------------------------------------------------------------------
    def predictions(self, stage="A"):
        """Return the full ODE-integrated prediction dict for the given stage.

        Parameters
        ----------
        stage : {"A", "B"}
            Which stage's NN weights to evaluate.  "B" falls back to "A"
            if Stage B was not run.

        Returns
        -------
        dict with keys:
            S_pred           : (T, 4)        ODE-integrated stocks
            F_int            : (T-1, N_FLOWS) ODE-accumulated yearly flow integrals
            cp, alphas, taus, frac_fu, frac_eu, f_cohort  (NN @ S_pred)
            *_at_obs                          (NN @ S_obs — parameter-ID quality)
            years            : (T,)
        """
        if stage not in ("A", "B"):
            raise ValueError(f"stage must be 'A' or 'B', got {stage!r}")
        if stage == "B" and not self._stage_B_ran:
            stage = "A"
        return self._predictions(stage)

    def save_predictions(self, path, *, stage="A"):
        """Save the Stage-A (or Stage-B) prediction arrays to a .npz archive.

        Parameters
        ----------
        path : str
            Output filename (``.npz`` suffix will be added if missing).
        stage : {"A", "B"}
            Which stage's predictions to save.  Default "A" so Stage-A
            outputs are easy to preserve before any Stage-B fine-tuning
            modifies the snapshot.

        Returns
        -------
        path : str   The actual filename written.
        """
        import os
        if not path.endswith(".npz"):
            path = path + ".npz"
        preds = self.predictions(stage)
        # Persist as plain numpy arrays
        np.savez(
            path,
            stage=np.asarray(stage),
            years=np.asarray(self.data_all["years"]),
            **{k: np.asarray(v) for k, v in preds.items() if k != "years"},
        )
        return path

    def _split_year_ranges(self):
        return (np.asarray(self.data_train["years"]),
                np.asarray(self.data_val["years"]),
                np.asarray(self.data_test["years"]))

    def _split_indices_in_all(self):
        """Boolean masks over `data_all['years']` for each split."""
        years_all = np.asarray(self.data_all["years"])
        ty, vy, tey = self._split_year_ranges()
        train_lo, train_hi = float(ty[0]),  float(ty[-1])
        val_lo,   val_hi   = float(vy[0]),  float(vy[-1])
        test_lo,  test_hi  = float(tey[0]), float(tey[-1])
        m_train = (years_all >= train_lo) & (years_all <= train_hi)
        m_val   = (years_all >= val_lo)   & (years_all <= val_hi)
        m_test  = (years_all >= test_lo)  & (years_all <= test_hi)
        return dict(train=m_train, val=m_val, test=m_test,
                    all=np.ones_like(m_train, dtype=bool))

    def _pinned_flags_per_tau_sup(self):
        """Length-8 list of bools (True ⇔ pinned) over TAU_SUP_NAMES."""
        return list(self.layout["pin_taus_tuple"]) + [
            self.layout["pin_fu_new"],   self.layout["pin_fu_loss"],
            self.layout["pin_eu_new"],   self.layout["pin_eu_loss"],
        ]

    # ------------------------------------------------------------------
    # diagnose — full per-component table
    # ------------------------------------------------------------------
    def diagnose(self, kind="freerun", *, include_pinned=False,
                 save_csv=None, verbose=True, return_rows=False):
        """Per-component logMAE / %RMSE / MAPE on each split (train/val/test/all).

        For each stage in {A, B (if run)} and each variable family
        (alphas, taus, cp, stocks, flows), reports per-component metrics
        plus a family-mean row.  Pinned components are hidden by default
        (set `include_pinned=True` to include them).  Family-mean rows are
        always computed over LEARNED components only — pinned components
        do not skew the means even when shown via `include_pinned=True`.

        Test-set evaluation kind  (NEW in v4)
        -------------------------------------
        Stocks and flows are dynamics-dependent: their ``test_*`` columns
        depend on WHERE you started integrating.  The two sensible choices:

        ``kind='freerun'``  (default — what v3 did)
            ODE is launched at year 1980 from observed S(1980) and
            integrated all the way through 2019.  Errors that accumulated
            during 1980–2006 (i.e. across train and val) are STILL PRESENT
            in the test slice — this answers "if you only knew the 1980
            initial state, how does the model do in 2007–2019?".

        ``kind='testrun'``
            ODE is RE-LAUNCHED at the LAST VALIDATION YEAR (year ≈2006)
            from observed S(2006), then integrated over the test years
            only.  This answers "given a known IC at the trainval/test
            boundary, how does the model project the next ~13 years?".
            The IC row itself (last validation year) is excluded from the
            metric — only test years contribute.  Numerically comparable
            to v1's ``stock_relRMSE_freerun`` metric.

        ``kind='both'``
            Reports both views.  The diagnose table prints the freerun
            stocks/flows family first, then re-prints the same family with
            ``[testrun]`` appended to component names and metric values
            recomputed under the testrun rollout.

        For α / τ / cp the kind argument has no effect: they are evaluated
        AT OBSERVED STOCKS (parameter-identification quality), which does
        not depend on the rollout starting point.

        Parameters
        ----------
        kind : {"freerun", "testrun", "both"}
            Selects the dynamics-evaluation rollout for stocks and flows.
        include_pinned : bool
            If True, also report metrics for pinned components.
        save_csv : str or None
            If given, write the full table (incl. pinned) as CSV to that path.
        verbose : bool
            If True, print the table.
        return_rows : bool
            If True, return the per-component row dicts; if False (default),
            return None.  The default avoids dumping the long list-of-dicts
            to stdout when called interactively at the REPL — the printed
            table is the intended output.  Programmatic users should pass
            ``return_rows=True``.

        Returns
        -------
        rows : list of dict OR None
            Per-component rows when ``return_rows=True``, else ``None``.
        """
        if kind not in ("freerun", "testrun", "both"):
            raise ValueError(
                f"diagnose kind must be 'freerun' | 'testrun' | 'both', got {kind!r}"
            )

        masks      = self._split_indices_in_all()
        flow_masks = {nm: masks[nm][1:] for nm in masks}    # F_int row i closes year[i+1]

        # Test-only masks for the testrun rollout (which is sized over data_test).
        # `data_test` is constructed in `train_model` as
        # ``_pack(n_trainval-1, T)``: its first row corresponds to the LAST
        # VALIDATION YEAR (year ≈ 2006).  We hand that observed stock vector
        # to the integrator as the IC, so the testrun ODE is launched from
        # the last validation year's S_obs and integrated forward through
        # the test window (years 2007–2019).
        #
        # The IC row itself is NOT a test point — it's the boundary year of
        # the validation split — so we EXCLUDE it from the test metric.
        # Including it would just deflate the metric (predicted == observed
        # at the IC by construction, so MAPE = 0 there for stocks).
        n_test_rows = int(np.asarray(self.data_test["years"]).shape[0])
        m_test_in_testrun = np.zeros(n_test_rows, dtype=bool)
        m_test_in_testrun[1:] = True                          # exclude IC row
        # Flow array F_int has length n_test_rows-1; F_int[i] closes year[i+1].
        # F_int[0] closes the first test year — keep all flow rows.
        m_test_in_testrun_flow = np.ones(n_test_rows - 1, dtype=bool)

        cp_pinned   = self.layout["pin_cp"]
        tau_pinned  = self._pinned_flags_per_tau_sup()

        rows = []
        for mode in self._stages():
            preds = self._predictions(mode)
            preds_tr = self._predictions_testrun(mode) if kind in ("testrun", "both") else None

            # ----- α (4 components) -----
            alpha_rows = []
            for k, nm_a in enumerate(ALPHA_NAMES):
                obs = np.asarray(self.data_all["alpha_obs"])[:, k]
                pre = preds["alphas_at_obs"][:, k]   # NN @ S_obs (parameter ID quality)
                row = {"component": nm_a, "kind": "alpha", "stage": mode,
                       "pinned": False}
                for nm in ("train", "val", "test", "all"):
                    sel = masks[nm]
                    row[f"{nm}_logMAE"]   = _logmae(pre[sel],   obs[sel])
                    row[f"{nm}_relRMSE%"] = _rel_rmse_pct(pre[sel], obs[sel])
                    row[f"{nm}_MAPE%"]    = _mape_pct(pre[sel],  obs[sel])
                alpha_rows.append(row)
            rows.extend(alpha_rows)
            rows.append(self._family_mean_row(alpha_rows, "alpha", mode))

            # ----- τ (8 supervised components) -----
            tau_rows = []
            for j, nm_t in enumerate(TAU_SUP_NAMES):
                obs = np.asarray(self.data_all["tau_sup_obs"])[:, j]
                if j < 4:
                    pre = preds["taus_at_obs"][:, j]
                elif j < 6:
                    pre = preds["frac_fu_at_obs"][:, j - 4]
                else:
                    pre = preds["frac_eu_at_obs"][:, j - 6]
                pinned = tau_pinned[j]
                row = {"component": nm_t + (" [pinned]" if pinned else ""),
                       "kind": "tau", "stage": mode, "pinned": pinned}
                for nm in ("train", "val", "test", "all"):
                    sel = masks[nm]
                    row[f"{nm}_logMAE"]   = _logmae(pre[sel],   obs[sel])
                    row[f"{nm}_relRMSE%"] = _rel_rmse_pct(pre[sel], obs[sel])
                    row[f"{nm}_MAPE%"]    = _mape_pct(pre[sel],  obs[sel])
                tau_rows.append(row)
            rows.extend(tau_rows)
            rows.append(self._family_mean_row(
                [r for r in tau_rows if not r["pinned"]], "tau", mode,
                empty_label="(no learned τ)",
            ))

            # ----- cp (singleton family) -----
            obs = np.asarray(self.data_all["cp_obs"])
            pre = preds["cp_at_obs"]
            row = {"component": "cp" + (" [pinned]" if cp_pinned else ""),
                   "kind": "cp", "stage": mode, "pinned": cp_pinned}
            for nm in ("train", "val", "test", "all"):
                sel = masks[nm]
                row[f"{nm}_logMAE"]   = _logmae(pre[sel],   obs[sel])
                row[f"{nm}_relRMSE%"] = _rel_rmse_pct(pre[sel], obs[sel])
                row[f"{nm}_MAPE%"]    = _mape_pct(pre[sel],  obs[sel])
            rows.append(row)

            # ----- stocks (always ODE-integrated, never pinned) -----
            stock_rows = self._stock_rows(preds, mode, masks, suffix="")
            if kind in ("freerun", "both"):
                rows.extend(stock_rows)
                rows.append(self._family_mean_row(stock_rows, "stock", mode))
            if kind in ("testrun", "both"):
                stock_rows_tr = self._stock_rows_testrun(
                    preds_tr, mode, m_test_in_testrun
                )
                rows.extend(stock_rows_tr)
                rows.append(self._family_mean_row(stock_rows_tr, "stock", mode,
                                                  suffix=" [testrun]"))

            # ----- flows (ODE F_int vs observed yearly integrals) -----
            flow_rows = self._flow_rows(preds, mode, flow_masks, suffix="")
            if kind in ("freerun", "both"):
                rows.extend(flow_rows)
                rows.append(self._family_mean_row(flow_rows, "flow", mode))
            if kind in ("testrun", "both"):
                flow_rows_tr = self._flow_rows_testrun(
                    preds_tr, mode, m_test_in_testrun_flow
                )
                rows.extend(flow_rows_tr)
                rows.append(self._family_mean_row(flow_rows_tr, "flow", mode,
                                                  suffix=" [testrun]"))

        if save_csv is not None:
            self._save_diagnose_csv(rows, save_csv, verbose=verbose)

        if verbose:
            tag = {"freerun": "freerun", "testrun": "testrun",
                   "both": "freerun + testrun"}[kind]
            print(f"\n[diagnose {self.name}]   stage(s): {','.join(self._stages())}   "
                  f"rollout: {tag}")
            self._print_table(rows, include_pinned=include_pinned)

        # By default we return None so an interactive call like
        # ``fit.diagnose("testrun")`` doesn't dump the underlying list of
        # dicts to stdout (the printed table is the intended output).
        # Programmatic users who want the rows pass ``return_rows=True``.
        return rows if return_rows else None

    # ------------------------------------------------------------------
    # helpers for stock / flow rows  (factored out to avoid duplication
    # between the freerun and testrun branches in diagnose)
    # ------------------------------------------------------------------
    def _stock_rows(self, preds, mode, masks, *, suffix=""):
        """Build per-stock diagnostic rows over data_all using ``preds``."""
        out = []
        for k, nm_s in enumerate(STOCK_NAMES):
            obs = np.asarray(self.data_all["stocks_obs"])[:, k]
            pre = preds["S_pred"][:, k]
            row = {"component": nm_s + suffix, "kind": "stock", "stage": mode,
                   "pinned": False}
            for nm in ("train", "val", "test", "all"):
                sel = masks[nm]
                row[f"{nm}_logMAE"]   = _logmae(pre[sel],   obs[sel])
                row[f"{nm}_relRMSE%"] = _rel_rmse_pct(pre[sel], obs[sel])
                row[f"{nm}_MAPE%"]    = _mape_pct(pre[sel],  obs[sel])
            out.append(row)
        return out

    def _stock_rows_testrun(self, preds_tr, mode, m_test_local):
        """Per-stock diagnostic rows from a TESTRUN integration over data_test.

        Train / val columns are NaN here (the testrun rollout doesn't cover
        those years), only the test column is populated.  The "all" column
        is set equal to the test column for table-readability.
        """
        out = []
        s_obs_test = np.asarray(self.data_test["stocks_obs"])
        for k, nm_s in enumerate(STOCK_NAMES):
            obs = s_obs_test[:, k]
            pre = preds_tr["S_pred"][:, k]
            row = {"component": nm_s + " [testrun]", "kind": "stock",
                   "stage": mode, "pinned": False}
            for nm in ("train", "val"):
                row[f"{nm}_logMAE"]   = float("nan")
                row[f"{nm}_relRMSE%"] = float("nan")
                row[f"{nm}_MAPE%"]    = float("nan")
            for nm in ("test", "all"):
                row[f"{nm}_logMAE"]   = _logmae(pre[m_test_local],
                                                obs[m_test_local])
                row[f"{nm}_relRMSE%"] = _rel_rmse_pct(pre[m_test_local],
                                                     obs[m_test_local])
                row[f"{nm}_MAPE%"]    = _mape_pct(pre[m_test_local],
                                                  obs[m_test_local])
            out.append(row)
        return out

    def _flow_rows(self, preds, mode, flow_masks, *, suffix=""):
        f_idx     = self.flow_obs_to_pred_idx
        flow_nms  = [FLOW_NAMES[i] for i in f_idx]
        F_int_full = preds["F_int"]
        F_obs_full = np.asarray(self.data_all["flows_obs"])
        cp_pinned  = bool(self.layout["pin_cp"])
        out = []
        for k, nm_f in enumerate(flow_nms):
            obs = F_obs_full[:, k]
            pre = F_int_full[:, int(f_idx[k])]
            # concentrate_production IS cp (computed by `compute_flows_from_nn`
            # straight from `out["cp"]`).  When cp is pinned the flow is
            # tautologically equal to the data lookup — mark as pinned so
            # diagnose() / plot_flows hide it by default.
            is_pinned = cp_pinned and (nm_f == "concentrate_production")
            comp_lbl  = nm_f + (" [pinned]" if is_pinned else "") + suffix
            row = {"component": comp_lbl, "kind": "flow", "stage": mode,
                   "pinned": is_pinned}
            for nm in ("train", "val", "test", "all"):
                sel = flow_masks[nm]
                if sel.sum() == 0:
                    row[f"{nm}_logMAE"]   = float("nan")
                    row[f"{nm}_relRMSE%"] = float("nan")
                    row[f"{nm}_MAPE%"]    = float("nan")
                else:
                    row[f"{nm}_logMAE"]   = _logmae(pre[sel],   obs[sel])
                    row[f"{nm}_relRMSE%"] = _rel_rmse_pct(pre[sel], obs[sel])
                    row[f"{nm}_MAPE%"]    = _mape_pct(pre[sel],  obs[sel])
            out.append(row)
        return out

    def _flow_rows_testrun(self, preds_tr, mode, m_test_flow_local):
        f_idx     = self.flow_obs_to_pred_idx
        flow_nms  = [FLOW_NAMES[i] for i in f_idx]
        F_int_test = preds_tr["F_int"]
        F_obs_test = np.asarray(self.data_test["flows_obs"])
        cp_pinned  = bool(self.layout["pin_cp"])
        out = []
        for k, nm_f in enumerate(flow_nms):
            obs = F_obs_test[:, k]
            pre = F_int_test[:, int(f_idx[k])]
            is_pinned = cp_pinned and (nm_f == "concentrate_production")
            comp_lbl  = nm_f + (" [pinned]" if is_pinned else "") + " [testrun]"
            row = {"component": comp_lbl, "kind": "flow",
                   "stage": mode, "pinned": is_pinned}
            for nm in ("train", "val"):
                row[f"{nm}_logMAE"]   = float("nan")
                row[f"{nm}_relRMSE%"] = float("nan")
                row[f"{nm}_MAPE%"]    = float("nan")
            for nm in ("test", "all"):
                if m_test_flow_local.sum() == 0:
                    row[f"{nm}_logMAE"]   = float("nan")
                    row[f"{nm}_relRMSE%"] = float("nan")
                    row[f"{nm}_MAPE%"]    = float("nan")
                else:
                    row[f"{nm}_logMAE"]   = _logmae(pre[m_test_flow_local],
                                                    obs[m_test_flow_local])
                    row[f"{nm}_relRMSE%"] = _rel_rmse_pct(pre[m_test_flow_local],
                                                         obs[m_test_flow_local])
                    row[f"{nm}_MAPE%"]    = _mape_pct(pre[m_test_flow_local],
                                                      obs[m_test_flow_local])
            out.append(row)
        return out

    @staticmethod
    def _family_mean_row(family_rows, kind, mode, *, empty_label=None, suffix=""):
        """Build a family-mean row from a list of per-component rows.

        The averages are over LEARNED components only — pinned-and-included
        rows are filtered out before averaging so pinned components never
        contaminate the family means.
        """
        learned = [r for r in family_rows if not r.get("pinned", False)]
        out = {"component": (empty_label if (not learned and empty_label)
                             else f"<{kind}-mean{suffix}>"),
               "kind": f"{kind}_mean", "stage": mode, "pinned": False}
        for nm in ("train", "val", "test", "all"):
            for stat in ("logMAE", "relRMSE%", "MAPE%"):
                key = f"{nm}_{stat}"
                vals = [r[key] for r in learned] if learned else []
                out[key] = _nanmean_safe(vals)
        return out

    @staticmethod
    def _print_table(rows, *, include_pinned=False):
        """Pretty-print the diagnose rows, grouped by (stage, kind)."""
        from collections import OrderedDict
        groups = OrderedDict()
        for r in rows:
            if r["pinned"] and not include_pinned:
                continue
            kind_key = r["kind"].replace("_mean", "")     # group means with their family
            groups.setdefault((r["stage"], kind_key), []).append(r)

        for (stg, kind), grp in groups.items():
            metric_keys = [k for k in grp[0].keys()
                           if k not in ("component", "kind", "stage", "pinned")]
            print(f"\n  Stage {stg} — {kind}")
            print(f"    {'component':28s} " +
                  "  ".join(f"{k:>14s}" for k in metric_keys))
            for r in grp:
                line = f"    {r['component']:28s} "
                for mk in metric_keys:
                    v = r.get(mk, float("nan"))
                    if not np.isfinite(v):
                        line += f"  {'      —':>14s}"
                    else:
                        line += f"  {v:13.4f}"
                print(line)

    @staticmethod
    def _save_diagnose_csv(rows, path, *, verbose=True):
        import csv
        if not rows:
            return
        all_keys = []
        for r in rows:
            for k in r.keys():
                if k not in all_keys:
                    all_keys.append(k)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=all_keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        if verbose:
            print(f"[diagnose] wrote per-component table → {path}")

    # ------------------------------------------------------------------
    # params_distance — quantifies how much Stage B moved relative to A
    # ------------------------------------------------------------------
    def params_distance(self, *, verbose=False):
        """Layer-wise L2 distance between Stage-A and Stage-B parameter snapshots.

        A scalar surrogate for "did Stage B do anything?".  If
        ``rel_l2 ≈ 0``, Stage B converged back to Stage A (or was
        early-stopped at step 0); if ``rel_l2`` is non-trivial (say > 1e-3
        on this problem) Stage B genuinely altered the fit.

        Returns
        -------
        dict with keys
          stage_B_ran   : bool
          total_l2_diff : ‖Δθ‖₂  over all leaves
          total_l2_normA: ‖θ_A‖₂ over all leaves
          rel_l2        : total_l2_diff / total_l2_normA
          n_leaves      : number of pytree leaves
          per_leaf      : list of {leaf_index, shape, l2_diff, l2_normA, rel}

        If Stage B did not run, returns a stub with ``rel_l2 = 0.0``.
        """
        info = {"stage_B_ran": bool(self._stage_B_ran),
                "total_l2_diff": 0.0, "total_l2_normA": 0.0,
                "rel_l2": 0.0, "n_leaves": 0, "per_leaf": []}
        if not self._stage_B_ran or self.params_B is None:
            if verbose:
                print(f"[params_distance {self.name}]  "
                      "Stage B did not run — params_A is the only snapshot.")
            return info

        leaves_A = jax.tree_util.tree_leaves(self.params_A)
        leaves_B = jax.tree_util.tree_leaves(self.params_B)
        if len(leaves_A) != len(leaves_B):
            raise RuntimeError(
                f"params_A and params_B have different pytree leaf counts "
                f"({len(leaves_A)} vs {len(leaves_B)}); aborting."
            )

        per_leaf = []
        sq_diff = 0.0
        sq_normA = 0.0
        for k, (la, lb) in enumerate(zip(leaves_A, leaves_B)):
            la_np = np.asarray(la); lb_np = np.asarray(lb)
            d  = float(np.sqrt(np.sum((la_np - lb_np) ** 2)))
            na = float(np.sqrt(np.sum(la_np ** 2)))
            sq_diff  += d * d
            sq_normA += na * na
            per_leaf.append({
                "leaf_index": k,
                "shape": tuple(la_np.shape),
                "l2_diff": d,
                "l2_normA": na,
                "rel": d / na if na > 0 else float("nan"),
            })

        total_diff  = float(np.sqrt(sq_diff))
        total_normA = float(np.sqrt(sq_normA))
        info.update(
            total_l2_diff = total_diff,
            total_l2_normA = total_normA,
            rel_l2 = total_diff / total_normA if total_normA > 0 else float("nan"),
            n_leaves = len(per_leaf),
            per_leaf = per_leaf,
        )

        if verbose:
            print(f"\n[params_distance {self.name}]  "
                  f"‖θ_B − θ_A‖₂ / ‖θ_A‖₂ = {info['rel_l2']:.3e}  "
                  f"(‖Δθ‖={total_diff:.3e}, ‖θ_A‖={total_normA:.3e}, "
                  f"{info['n_leaves']} leaves)")
            # Print top movers
            sorted_leaves = sorted(per_leaf, key=lambda r: -r["rel"])
            print(f"  largest relative moves (top 5 of {len(per_leaf)}):")
            for r in sorted_leaves[:5]:
                print(f"    leaf #{r['leaf_index']:3d}  shape={str(r['shape']):14s}  "
                      f"rel={r['rel']:.3e}  ‖Δ‖={r['l2_diff']:.3e}")

        return info

    # ------------------------------------------------------------------
    # summary — the compact family-average printout used by run()
    # ------------------------------------------------------------------
    def summary(self, kind="both", *, include_pinned=False, prefer="auto",
                verbose=True):
        """Compact per-family summary on the **TEST SET** only.

        Reports family-mean MAPE and %RMSE for alphas, taus, stocks, flows
        on the held-out test split.  Pinned components are excluded from
        family means by default so they cannot artificially deflate the
        averages.  See ``fit.diagnose()`` for per-component metrics on
        every split.

        Parameters
        ----------
        kind : {"freerun", "testrun", "both"}    (NEW in v4)
            For stocks and flows, controls which rollout the test metrics
            come from:
              - "freerun"  : ODE launched at year 1980 from S(1980)
                             (errors compound over 1980–2019)
              - "testrun"  : ODE re-launched at start of test from S(2006)
                             (~13-year horizon, comparable to v1's metric)
              - "both"     : prints both side-by-side  (default)
            α / τ / cp are not affected — they're evaluated at S_obs.
        include_pinned : bool
            If True, pinned components are included in family-means.
            Default False (recommended — avoids skewing).
        prefer : {"A","B","auto"}
            Which stage's weights to use.  "auto" picks B if it ran.

        Returns
        -------
        dict {family: {"MAPE%": v, "relRMSE%": v}, ..., "stage": mode,
              "split": "test", "kind": kind}
        """
        if kind not in ("freerun", "testrun", "both"):
            raise ValueError(
                f"summary kind must be 'freerun' | 'testrun' | 'both', got {kind!r}"
            )
        if prefer == "auto":
            mode = "B" if self._stage_B_ran else "A"
        elif prefer == "A":
            mode = "A"
        elif prefer == "B":
            mode = "B" if self._stage_B_ran else "A"
        else:
            raise ValueError(f"prefer must be 'auto'|'A'|'B', got {prefer!r}")

        masks   = self._split_indices_in_all()
        m_test  = masks["test"]
        m_test_flow = m_test[1:]    # F_int row i closes year[i+1]

        tau_pinned = self._pinned_flags_per_tau_sup()
        cp_pinned  = self.layout["pin_cp"]

        # ---- α / τ / cp (rollout-independent, evaluated at S_obs) ----
        preds = self._predictions(mode)

        a_obs = np.asarray(self.data_all["alpha_obs"])
        a_pre = preds["alphas_at_obs"]
        # Family-mean over per-channel MAPE / relRMSE.  α is never pinned.
        alpha_mape  = _nanmean_safe(
            [_mape_pct(a_pre[m_test, k], a_obs[m_test, k])
             for k in range(N_ALPHAS)])
        alpha_rrmse = _nanmean_safe(
            [_rel_rmse_pct(a_pre[m_test, k], a_obs[m_test, k])
             for k in range(N_ALPHAS)])

        t_obs = np.asarray(self.data_all["tau_sup_obs"])
        tau_mape_list, tau_rrmse_list = [], []
        for j in range(N_TAU_SUP):
            if tau_pinned[j] and not include_pinned:
                continue
            if j < 4:
                pre = preds["taus_at_obs"][:, j]
            elif j < 6:
                pre = preds["frac_fu_at_obs"][:, j - 4]
            else:
                pre = preds["frac_eu_at_obs"][:, j - 6]
            tau_mape_list.append( _mape_pct(   pre[m_test], t_obs[m_test, j]))
            tau_rrmse_list.append(_rel_rmse_pct(pre[m_test], t_obs[m_test, j]))
        tau_mape  = _nanmean_safe(tau_mape_list)
        tau_rrmse = _nanmean_safe(tau_rrmse_list)

        cp_obs_arr = np.asarray(self.data_all["cp_obs"])
        cp_pre_arr = preds["cp_at_obs"]
        if cp_pinned and not include_pinned:
            cp_mape  = float("nan")
            cp_rrmse = float("nan")
        else:
            cp_mape  = _mape_pct(   cp_pre_arr[m_test], cp_obs_arr[m_test])
            cp_rrmse = _rel_rmse_pct(cp_pre_arr[m_test], cp_obs_arr[m_test])

        # ---- stocks / flows: depend on rollout, may need both ----
        out = {
            "alphas":  {"MAPE%": alpha_mape,  "relRMSE%": alpha_rrmse},
            "taus":    {"MAPE%": tau_mape,    "relRMSE%": tau_rrmse},
            "cp":      {"MAPE%": cp_mape,     "relRMSE%": cp_rrmse,
                        "pinned": bool(cp_pinned)},
            "stage": mode, "split": "test", "kind": kind,
        }

        f_idx = self.flow_obs_to_pred_idx
        F_obs = np.asarray(self.data_all["flows_obs"])
        # Build a "flow learned" mask: every flow column that isn't a pinned
        # tautology.  Currently the only pinnable flow is concentrate_production
        # (= cp itself), pinned iff `layout["pin_cp"]`.
        flow_names_native = [FLOW_NAMES[i] for i in f_idx]
        flow_learned_mask = np.array([
            not (cp_pinned and nm == "concentrate_production")
            for nm in flow_names_native
        ], dtype=bool)
        if include_pinned:
            flow_learned_mask[:] = True

        def _stocks_flows_from(s_pre, s_obs, m_s, F_int_arr, F_obs_arr, m_f):
            s_mape  = _nanmean_safe(
                [_mape_pct(s_pre[m_s, k], s_obs[m_s, k]) for k in range(N_STOCKS)])
            s_rrmse = _nanmean_safe(
                [_rel_rmse_pct(s_pre[m_s, k], s_obs[m_s, k]) for k in range(N_STOCKS)])
            f_mape  = _nanmean_safe(
                [_mape_pct(F_int_arr[m_f, int(f_idx[k])], F_obs_arr[m_f, k])
                 for k in range(F_obs_arr.shape[1]) if flow_learned_mask[k]])
            f_rrmse = _nanmean_safe(
                [_rel_rmse_pct(F_int_arr[m_f, int(f_idx[k])], F_obs_arr[m_f, k])
                 for k in range(F_obs_arr.shape[1]) if flow_learned_mask[k]])
            return s_mape, s_rrmse, f_mape, f_rrmse

        if kind in ("freerun", "both"):
            s_obs_all = np.asarray(self.data_all["stocks_obs"])
            s_pre_fr  = preds["S_pred"]
            s_mape, s_rrmse, f_mape, f_rrmse = _stocks_flows_from(
                s_pre_fr, s_obs_all, m_test,
                preds["F_int"], F_obs, m_test_flow,
            )
            out["stocks_freerun"] = {"MAPE%": s_mape, "relRMSE%": s_rrmse}
            out["flows_freerun"]  = {"MAPE%": f_mape, "relRMSE%": f_rrmse}

        if kind in ("testrun", "both"):
            preds_tr = self._predictions_testrun(mode)
            s_obs_test = np.asarray(self.data_test["stocks_obs"])
            F_obs_test = np.asarray(self.data_test["flows_obs"])
            n_test_rows = s_obs_test.shape[0]
            # IC row (index 0) corresponds to the LAST VALIDATION YEAR — it
            # is the integration starting point, not a test point.  Exclude
            # it from the metric so we don't artificially deflate stock errors
            # (predicted == observed there by construction).
            m_test_local = np.zeros(n_test_rows, dtype=bool)
            m_test_local[1:] = True
            m_test_flow_local = np.ones(n_test_rows - 1, dtype=bool)
            s_mape, s_rrmse, f_mape, f_rrmse = _stocks_flows_from(
                preds_tr["S_pred"], s_obs_test, m_test_local,
                preds_tr["F_int"],  F_obs_test, m_test_flow_local,
            )
            out["stocks_testrun"] = {"MAPE%": s_mape, "relRMSE%": s_rrmse}
            out["flows_testrun"]  = {"MAPE%": f_mape, "relRMSE%": f_rrmse}

        # For backward-compat-ish naming, expose plain "stocks"/"flows" keys
        # equal to the freerun version when kind="both", or to the only one
        # available otherwise.
        if "stocks_freerun" in out:
            out["stocks"] = out["stocks_freerun"]; out["flows"] = out["flows_freerun"]
        else:
            out["stocks"] = out["stocks_testrun"]; out["flows"] = out["flows_testrun"]

        if verbose:
            tag = f"Stage {mode}"
            if mode == "B" and not self._stage_B_ran:
                tag += " (fallback: A only)"
            ty, vy, tey = self._split_year_ranges()
            test_lo, test_hi = int(tey[0]), int(tey[-1])
            kind_tag = {"freerun": "freerun rollout from 1980",
                        "testrun": f"testrun rollout from {test_lo - 1}",
                        "both":    f"freerun (1980→) | testrun ({test_lo - 1}→)"}[kind]
            pin_tag = "incl." if include_pinned else "excl."
            print(f"\n[summary {self.name}]  {tag}  "
                  f"family means on TEST SET ({test_lo}–{test_hi}, "
                  f"n={int(m_test.sum())} obs; pinned={pin_tag}; "
                  f"{kind_tag}):")

            # Layout per row depends on kind: stocks/flows have 1 or 2 rollout columns
            if kind == "both":
                hdr = (f"    {'family':24s}  {'MAPE%':>10s}  {'relRMSE%':>10s}  "
                       f"{'MAPE%':>10s}  {'relRMSE%':>10s}")
                print(f"    {'':24s}  {'-- freerun --':>22s}  {'-- testrun --':>22s}")
                print(hdr)
                # rollout-independent families
                for fam_lbl, fam_key in (("alphas", "alphas"), ("taus", "taus")):
                    fam = out[fam_key]
                    line = (f"    {fam_lbl:24s}  "
                            f"{_fmt_pct(fam['MAPE%']):>10s}  "
                            f"{_fmt_pct(fam['relRMSE%']):>10s}  "
                            f"{_fmt_pct(fam['MAPE%']):>10s}  "
                            f"{_fmt_pct(fam['relRMSE%']):>10s}")
                    print(line)
                # cp (always shown unless hidden via include_pinned)
                if not (cp_pinned and not include_pinned):
                    cp_lbl = "cp" + (" [pinned]" if cp_pinned else "")
                    line = (f"    {cp_lbl:24s}  "
                            f"{_fmt_pct(cp_mape):>10s}  "
                            f"{_fmt_pct(cp_rrmse):>10s}  "
                            f"{_fmt_pct(cp_mape):>10s}  "
                            f"{_fmt_pct(cp_rrmse):>10s}")
                    print(line)
                # stocks / flows have separate freerun and testrun values
                for fam_lbl in ("stocks", "flows"):
                    fr = out[f"{fam_lbl}_freerun"]; tr = out[f"{fam_lbl}_testrun"]
                    line = (f"    {fam_lbl:24s}  "
                            f"{_fmt_pct(fr['MAPE%']):>10s}  "
                            f"{_fmt_pct(fr['relRMSE%']):>10s}  "
                            f"{_fmt_pct(tr['MAPE%']):>10s}  "
                            f"{_fmt_pct(tr['relRMSE%']):>10s}")
                    print(line)
            else:
                hdr = (f"    {'family':24s}  {'MAPE%':>10s}  {'relRMSE%':>10s}")
                print(hdr)
                for fam_lbl, fam_key in (("alphas", "alphas"), ("taus", "taus")):
                    fam = out[fam_key]
                    print(f"    {fam_lbl:24s}  "
                          f"{_fmt_pct(fam['MAPE%']):>10s}  "
                          f"{_fmt_pct(fam['relRMSE%']):>10s}")
                if not (cp_pinned and not include_pinned):
                    cp_lbl = "cp" + (" [pinned]" if cp_pinned else "")
                    print(f"    {cp_lbl:24s}  "
                          f"{_fmt_pct(cp_mape):>10s}  "
                          f"{_fmt_pct(cp_rrmse):>10s}")
                for fam_lbl in ("stocks", "flows"):
                    fam_key = f"{fam_lbl}_{kind}"
                    fam = out[fam_key]
                    print(f"    {fam_lbl:24s}  "
                          f"{_fmt_pct(fam['MAPE%']):>10s}  "
                          f"{_fmt_pct(fam['relRMSE%']):>10s}")

            # Stage A vs Stage B parameter movement
            pd = self.params_distance(verbose=False)
            if pd["stage_B_ran"]:
                rel = pd["rel_l2"]
                if not np.isfinite(rel):
                    note = "‖θ_A‖ = 0 (degenerate)"
                elif rel < 1e-6:
                    note = "essentially identical to Stage A — Stage B did not move"
                elif rel < 1e-3:
                    note = "minor drift — Stage B made small adjustments"
                else:
                    note = "non-trivial fine-tune — Stage B reshaped the fit"
                print(f"    ‖θ_B − θ_A‖₂ / ‖θ_A‖₂ = {rel:.3e}   ({note})")
        return out

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def _shade_splits(self, ax):
        if not HAS_MPL:
            return
        ty, vy, tey = self._split_year_ranges()
        ax.axvspan(float(ty[0]),  float(ty[-1]),  color="#cfe6ff", alpha=0.35,
                   lw=0, label="train")
        ax.axvspan(float(ty[-1]), float(vy[-1]),  color="#cdf2cd", alpha=0.40,
                   lw=0, label="val")
        ax.axvspan(float(vy[-1]), float(tey[-1]), color="#ffd9d9", alpha=0.40,
                   lw=0, label="test")

    def _save_or_show(self, fig, save_dir, fname):
        if save_dir is None:
            plt.show()
            return None
        import os
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, fname)
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return out

    def plot_alphas(self, save_dir=None, *, include_pinned=False):
        # Alphas are never pinned, so include_pinned has no effect — kept for API symmetry.
        del include_pinned
        if not HAS_MPL:
            print("matplotlib not available; skipping plot.")
            return []
        years   = np.asarray(self.data_all["years"])
        alpha_o = np.asarray(self.data_all["alpha_obs"])
        preds   = {m: self._predictions(m) for m in self._stages()}
        out_paths = []
        for k, nm in enumerate(ALPHA_NAMES):
            fig, ax = plt.subplots(figsize=(7, 3.6))
            self._shade_splits(ax)
            ax.plot(years, alpha_o[:, k], "ko-", label="empirical α",
                    lw=1.0, ms=3.5)
            for m, p in preds.items():
                ax.plot(years, p["alphas_at_obs"][:, k],
                        "-" if m == "A" else "--",
                        label=f"NN_{m} @ S_obs", lw=1.5)
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
        years     = np.asarray(self.data_all["years"])
        tau_o     = np.asarray(self.data_all["tau_sup_obs"])
        preds     = {m: self._predictions(m) for m in self._stages()}
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
            ax.plot(years, tau_o[:, j], "ko-", label="observed",
                    lw=1.0, ms=3.5)
            tag = " [pinned]" if pin_flags[j] else ""
            for m, T in tau_full.items():
                ax.plot(years, T[:, j], "-" if m == "A" else "--",
                        label=f"NN_{m} @ S_obs{tag}", lw=1.5)
            ax.set_xlabel("Year"); ax.set_ylabel("coefficient")
            ax.set_title(f"{self.name}  {nm}{tag}")
            ax.grid(True, alpha=0.4); ax.legend(loc="best", fontsize=8)
            plt.tight_layout()
            out_paths.append(self._save_or_show(fig, save_dir,
                                                f"{self.name}_tau_{nm}.png"))
        return [p for p in out_paths if p is not None]

    def plot_stocks(self, save_dir=None, *, include_pinned=False):
        del include_pinned   # stocks are never pinned
        if not HAS_MPL:
            print("matplotlib not available; skipping plot.")
            return []
        years = np.asarray(self.data_all["years"])
        S_obs = np.asarray(self.data_all["stocks_obs"])
        preds = {m: self._predictions(m) for m in self._stages()}
        out_paths = []
        for k, nm in enumerate(STOCK_NAMES):
            fig, ax = plt.subplots(figsize=(7, 3.6))
            self._shade_splits(ax)
            ax.plot(years, S_obs[:, k], "ko-", label="obs", lw=1.0, ms=3.5)
            for m, p in preds.items():
                ax.plot(years, p["S_pred"][:, k],
                        "-" if m == "A" else "--",
                        label=f"ODE free-run (NN_{m})", lw=1.5)
            ax.set_xlabel("Year"); ax.set_ylabel("stock (kt)")
            ax.set_title(f"{self.name}  {nm} stock")
            ax.grid(True, alpha=0.4); ax.legend(loc="best", fontsize=8)
            plt.tight_layout()
            out_paths.append(self._save_or_show(fig, save_dir,
                                                f"{self.name}_stock_{nm}.png"))
        return [p for p in out_paths if p is not None]

    def plot_flows(self, save_dir=None, *, include_pinned=False):
        """Per-flow ODE-integrated trajectories vs observed yearly integrals.

        Pinned flows (currently: ``concentrate_production`` whenever cp is
        pinned, since the flow IS cp) are skipped by default — they are
        tautologies of the data lookup.  Pass ``include_pinned=True`` to
        force-render them.
        """
        if not HAS_MPL:
            print("matplotlib not available; skipping plot.")
            return []
        years     = np.asarray(self.data_all["years"])
        F_obs     = np.asarray(self.data_all["flows_obs"])           # (T-1, K)
        f_idx     = self.flow_obs_to_pred_idx
        flow_nms  = [FLOW_NAMES[i] for i in f_idx]
        cp_pinned = bool(self.layout["pin_cp"])
        year_close = years[1:]
        preds     = {m: self._predictions(m) for m in self._stages()}

        out_paths = []
        for k, nm in enumerate(flow_nms):
            is_pinned = cp_pinned and (nm == "concentrate_production")
            if is_pinned and not include_pinned:
                continue
            tag = " [pinned]" if is_pinned else ""
            fig, ax = plt.subplots(figsize=(7, 3.6))
            self._shade_splits(ax)
            ax.plot(year_close, F_obs[:, k], "ko-",
                    label="obs (yearly integral)", lw=1.0, ms=3.5)
            for m, p in preds.items():
                ax.plot(year_close, p["F_int"][:, int(f_idx[k])],
                        "-" if m == "A" else "--",
                        label=f"ODE-integrated (NN_{m}){tag}", lw=1.5)
            ax.set_xlabel("Year (closing)"); ax.set_ylabel("flow (kt/yr)")
            ax.set_title(f"{self.name}  {nm}{tag}")
            ax.grid(True, alpha=0.4); ax.legend(loc="best", fontsize=8)
            plt.tight_layout()
            out_paths.append(self._save_or_show(fig, save_dir,
                                                f"{self.name}_flow_{nm}.png"))
        return [p for p in out_paths if p is not None]

    def plot_cohorts(self, save_dir=None, *, include_pinned=False):
        """Two-panel plot of in-use cohort dynamics.

        Top panel — actual cohort STOCK FRACTIONS (S_k / Σ S_k) extracted
        from the ODE state.  By the IC `_make_Y0`, S_cohorts(t₀) =
        IC_COHORT_FRACS * S_inuse(t₀), so these lines start at exactly
        IC_COHORT_FRACS = (0.09, 0.27, 0.64) in 1980, with the prior
        markers drawn on top.  This is the quantity Rostek Fig. 3 shows.

        Bottom panel — NN-predicted cohort INFLOW SPLIT f_cohort, the
        3-simplex over (short, medium, long) lifetimes that splits each
        year's inuse_inflow across the three cohort sub-stocks.  This is
        a *flow* split, not a stock distribution; the warm-start sets the
        bias to log(IC_COHORT_FRACS) so init shows up near the prior, but
        once Stage A runs the NN can move freely (only direct levers are
        Stage-B stock divergence and the optional KL prior).  The IC
        prior reference at 1980 is a stock-distribution prior — not
        directly applicable to f_cohort, but kept as a visual anchor.

        Solid line = NN_A (Stage A), dashed = NN_B (Stage B if it ran).

        ``include_pinned`` is accepted for API symmetry but has no effect —
        cohort fractions are never pinnable in v4.
        """
        del include_pinned   # cohorts are never pinned in v4
        if not HAS_MPL:
            print("matplotlib not available; skipping plot.")
            return []
        years = np.asarray(self.data_all["years"])
        preds = {m: self._predictions(m) for m in self._stages()}

        fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.0), sharex=True)
        ax_stk, ax_inf = axes
        for ax in axes:
            self._shade_splits(ax)

        colours = ["tab:blue", "tab:orange", "tab:green"]

        # Top: actual cohort STOCK FRACTIONS from ODE state
        for k, nm in enumerate(COHORT_NAMES):
            for m, p in preds.items():
                S_coh   = p["S_cohorts"]                          # (T, 3)
                tot     = np.sum(S_coh, axis=1, keepdims=True)
                tot     = np.where(tot > 1e-12, tot, 1e-12)
                stk_fr  = S_coh / tot                              # (T, 3)
                ls    = "-" if m == "A" else "--"
                ax_stk.plot(years, stk_fr[:, k], ls, color=colours[k],
                            lw=1.6, label=f"{nm} stock-frac (NN_{m})")
            # IC prior at 1980 — by construction the lines pass through this
            ax_stk.scatter([float(years[0])], [float(IC_COHORT_FRACS[k])],
                           s=46, marker="o", facecolors="none",
                           edgecolors=colours[k], linewidth=1.4, zorder=5)

        ax_stk.set_ylim(0.0, 1.0)
        ax_stk.set_ylabel("share of S_inuse(t)")
        ax_stk.set_title(
            f"{self.name}  in-use cohort STOCK fractions  (S_k / Σ S_k)  "
            f"○ = IC prior 1980 (matches by construction)", fontsize=10
        )
        ax_stk.grid(True, alpha=0.4); ax_stk.legend(loc="best", fontsize=8, ncol=2)

        # Bottom: NN-predicted INFLOW SPLIT f_cohort
        for k, nm in enumerate(COHORT_NAMES):
            for m, p in preds.items():
                ls    = "-" if m == "A" else "--"
                ax_inf.plot(years, p["f_cohort_at_obs"][:, k],
                            ls, color=colours[k], lw=1.6,
                            label=f"{nm} inflow-split (NN_{m})")
            # Reference marker — note this is a STOCK prior, only an anchor here
            ax_inf.scatter([float(years[0])], [float(IC_COHORT_FRACS[k])],
                           s=46, marker="^", facecolors="none",
                           edgecolors=colours[k], linewidth=1.4, zorder=5)

        ax_inf.set_ylim(0.0, 1.0)
        ax_inf.set_xlabel("Year")
        ax_inf.set_ylabel("share of inuse_inflow(t)")
        ax_inf.set_title(
            f"NN-predicted INFLOW SPLIT f_cohort  "
            f"(μ = {[float(x) for x in MU_COHORTS_NP]} yr)  "
            f"△ = stock-prior reference (not directly applicable)", fontsize=10
        )
        ax_inf.grid(True, alpha=0.4); ax_inf.legend(loc="best", fontsize=8, ncol=2)

        plt.tight_layout()
        out_path = self._save_or_show(fig, save_dir,
                                      f"{self.name}_cohorts.png")
        return [out_path] if out_path is not None else []

    def plot_all(self, save_dir=None, *, include_pinned=False):
        out = []
        out += self.plot_alphas(save_dir=save_dir,  include_pinned=include_pinned)
        out += self.plot_taus(save_dir=save_dir,    include_pinned=include_pinned)
        out += self.plot_stocks(save_dir=save_dir,  include_pinned=include_pinned)
        out += self.plot_flows(save_dir=save_dir,   include_pinned=include_pinned)
        out += self.plot_cohorts(save_dir=save_dir, include_pinned=include_pinned)
        return out


# ============================================================================
# 14) Default config + entry point
# ============================================================================
DEFAULT_CONFIG = dict(
    xlsx_path="zinc_dataset.xlsx",
    seed=0,

    # NN
    use_time_input=False,
    use_stock_input=True,
    stock_norm_mode="tanh_log",
    stock_log_scale=1.0,
    stock_ref_mode="mean",
    hidden_width=32,
    hidden_depth=2,

    # Pin flags (which quantities are pinned to observed-data interpolation
    # rather than emitted by the NN).  cp is pinned by default since it's
    # well-observed (ILZSG, every year).  Others are learned by default;
    # pin individual τ's when their observed values are nearly stationary.
    learn_cp=False,
    pin_tau_ref=False,
    pin_tau_waelz=False,
    pin_tau_olds=False,
    pin_tau_diss=False,
    pin_frac_fu_loss=False,
    pin_frac_eu_loss=False,
    pin_frac_fu_new=False,                # NEW in v4
    pin_frac_eu_new=False,                # NEW in v4

    # exogenous
    extra_exog_cols=[
        "Precious metal index",
        "Metal real index excl iron",
        "World Stock Market Capitalisation (% of GDP)",
        "China Total Manufacturing Output",
        "Population",
    ],
    exog_log1p=True,
    exog_detrend=False,
    exog_feature_orders=(0, 1),
    exog_diff_pad="edge",
    exog_pca_components=None,

    # split
    trainval_frac=0.7,
    val_frac=0.2,

    # Stage A
    stageA_steps=3000,
    stageA_lr=3e-4,
    stageA_w_alpha=1.0,
    stageA_w_tau=1.0,
    stageA_w_cp=1.0,
    stageA_w_smooth_alpha=0.05,
    stageA_w_smooth_tau=0.05,
    stageA_w_cohort_prior=0.0,
    stageA_w_raw_reg=1e-4,

    # Stage B
    do_stage_B=False,
    stageB_steps=2000,
    stageB_lr=1e-4,
    stageB_window=None,
    stageB_curriculum=None,
    stageB_batch_size=4,
    stageB_w_alpha=1.0,
    stageB_w_tau=1.0,
    stageB_w_cp=1.0,
    stageB_w_S=2.0,
    stageB_w_F=0.5,                # log-space yearly-flow divergence
    stageB_w_smooth_alpha=0.05,
    stageB_w_smooth_tau=0.05,
    stageB_w_cohort_prior=0.0,
    stageB_w_raw_reg=1e-4,
    stock_loss_kind="log",         # v3 default — log-MSE on stocks (was std-scaled in v2)

    # Final refit on train+val (only used when do_stage_B=True).  Common
    # practice: after early-stopping selects best params on val, do a short
    # additional Stage-B pass on the combined train+val to use all the data.
    final_refit_on_trainval=True,
    final_refit_steps=400,
    final_refit_lr=3e-5,
    final_refit_curriculum=None,        # NEW in v4: list[(steps, window)] like
                                        # `stageB_curriculum`.  None ⇒ legacy
                                        # single-window-full-trainval refit.
    final_refit_batch_size=None,        # NEW in v4: None ⇒ inherits stageB_batch_size

    # optimiser
    weight_decay=1e-4,
    grad_clip=0.0,                 # 0 disables (default in v2)

    # early stopping (Stage B only)
    early_stop=True,
    eval_every=200,
    patience=10,

    integrator=None,

    stock_term_weights=(1.0, 1.0, 1.0, 1.0),

    verbose=True,
)


def run(name="default", **overrides):
    """Convenience entry point: train_model + return a FitResult.

    After training, prints a compact summary of family-mean MAPE and
    %RMSE on the test set, side-by-side for both the freerun rollout
    (ODE launched at year 1980) and the testrun rollout (ODE re-launched
    at the last validation year).  Pinned components are excluded from
    the averages.  For full per-component tables, call
    ``fit.diagnose(kind="freerun"|"testrun"|"both")``.

    Examples
    --------
    >>> fit = run("AB", do_stage_B=True,
    ...           stageB_curriculum=[(800, 8), (800, 16), (400, 22)],
    ...           final_refit_curriculum=[(300, 22)])
    >>> fit.summary()                          # both rollouts, MAPE + relRMSE
    >>> fit.diagnose("testrun")                # test column from t0 = 2006
    >>> fit.diagnose("both")                   # both rollouts shown
    >>> fit.diagnose("freerun", return_rows=True)  # rows back as list[dict]
    >>> fit.plot_all(save_dir='./plots/AB')    # save all panels (skips pinned)
    """
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(overrides)
    print(f"\n=== zinc_colloc_v5 run: {name} ===")
    (params, params_A, layout, dt, dv, dtv, dte, da,
     integrate_aug, nn_eval, nn_eval_sup,
     flow_obs_to_pred_idx) = train_model(**cfg)
    fit = FitResult(
        name=name, cfg=cfg,
        params=params, params_A=params_A, layout=layout,
        data_train=dt, data_val=dv, data_trainval=dtv,
        data_test=dte, data_all=da,
        integrate_aug=integrate_aug,
        nn_eval=nn_eval, nn_eval_sup=nn_eval_sup,
        flow_obs_to_pred_idx=flow_obs_to_pred_idx,
        do_stage_B=bool(cfg.get("do_stage_B", False)),
    )
    fit.summary()
    return fit


# ============================================================================
# 14) [v5] Rolling-origin cross-validation + fold ensembling
# ============================================================================
# Walk-forward CV with held-out test (Tashman 2000; Bergmeir et al. 2018).
# Default strategy is expanding window: train slice grows fold by fold while
# test stays fixed.  Each fold refits EVERYTHING — NN params, Stage A/B,
# exog/α/τ/S statistics — on the fold's train slice only, so no future
# information leaks via normalisation.
#
# Fold ensembling (Lakshminarayanan et al. 2017) averages the K fold-models'
# predicted trajectories on the held-out test window.  Trajectory averaging
# (rather than parameter averaging) is robust to NN permutation symmetries
# and behaves well under the non-linear cohort dynamics.

def rolling_origin_splits(
    T,
    *,
    test_frac=0.3,
    n_folds=5,
    val_horizon=10,
    min_train=15,
    step=None,
    test_start=None,
):
    """Generate expanding-window rolling-origin fold specifications.

    Parameters
    ----------
    T : int
        Total length of the time series (number of observed years).
    test_frac : float in (0, 1)
        Held-out test fraction.  Test window = the last
        ``ceil(test_frac * T)`` years.  Ignored if ``test_start`` is given.
    test_start : int or None
        If given, use this as the fixed test_start across all folds (overrides
        ``test_frac``).  0-based index.
    n_folds : int
        Number of CV folds.  Origins are spaced uniformly between the earliest
        feasible (=``min_train - 1``) and the latest feasible
        (=``test_start - 1 - val_horizon``).
    val_horizon : int
        Length (in years) of the validation block following each fold's
        train block.  Should roughly match the test horizon for comparability.
    min_train : int
        Smallest train fold size (in years).  Constrains the earliest origin.
    step : int or None
        Spacing between origins.  ``None`` ⇒ uniform spacing across feasible
        range; otherwise an explicit integer step.

    Returns
    -------
    list of dicts, one per fold, each with keys:
      ``fold``        : int, 0-indexed
      ``train_end``   : int, last train year index (inclusive)
      ``val_end``     : int, last val year index (inclusive); val starts at train_end+1
      ``test_start``  : int, first test year index (FIXED across all folds)

    Notes
    -----
    Every fold is fit on train [0, train_end], validated on
    [train_end+1, val_end], and (after CV) evaluated on the SAME held-out
    test [test_start, T-1].  By construction val and test never overlap.
    """
    T = int(T)
    if test_start is None:
        n_test = max(int(np.ceil(test_frac * T)), 3)
        test_start = T - n_test
    else:
        test_start = int(test_start)
    if not (1 <= test_start < T):
        raise ValueError(f"test_start={test_start} must be in [1, T-1] (T={T}).")

    if test_start < min_train + val_horizon:
        raise ValueError(
            f"Not enough trainval data: test_start={test_start}, but need ≥ "
            f"min_train ({min_train}) + val_horizon ({val_horizon}) = "
            f"{min_train + val_horizon}.  Reduce test_frac/val_horizon/min_train, "
            f"or pass a later test_start."
        )

    earliest_te = int(min_train - 1)
    latest_te   = int(test_start - 1 - val_horizon)
    n_folds = max(int(n_folds), 1)

    if n_folds == 1:
        train_ends = [latest_te]
    else:
        if step is None:
            step_f = (latest_te - earliest_te) / (n_folds - 1)
        else:
            step_f = float(step)
        train_ends_raw = [earliest_te + i * step_f for i in range(n_folds)]
        train_ends = [int(round(te)) for te in train_ends_raw]
        train_ends = [min(max(te, earliest_te), latest_te) for te in train_ends]
        # de-duplicate while preserving order
        seen = set()
        train_ends = [te for te in train_ends if not (te in seen or seen.add(te))]

    splits = [
        dict(
            fold=fold,
            train_end=int(te),
            val_end=int(te + val_horizon),
            test_start=int(test_start),
        )
        for fold, te in enumerate(train_ends)
    ]
    return splits


# ----------------------------------------------------------------------------
# Orchestrator: run train_model once per fold
# ----------------------------------------------------------------------------
def train_model_rolling_origin(
    xlsx_path,
    *,
    cv_n_folds=5,
    cv_test_frac=0.3,
    cv_val_horizon=10,
    cv_min_train=15,
    cv_step=None,
    cv_test_start=None,
    cv_seed_per_fold=True,
    verbose=True,
    **train_kwargs,
):
    """Run rolling-origin CV: train one ``train_model`` call per fold.

    Each fold refits NN params, Stage A, Stage B (if enabled), AND the
    train-only statistics (S_mean, S_std, exog stats, α-scale, τ stds,
    cp_scale, S_log_std, …) on the fold's train slice — so no future
    information leaks via normalisation, which is the most common subtle
    bug in time-series CV implementations.

    Parameters
    ----------
    xlsx_path : str
        Path to the zinc dataset.
    cv_n_folds : int
        Number of CV folds.
    cv_test_frac : float
        Held-out test fraction (last cv_test_frac of the series).
    cv_test_start : int or None
        Override for explicit test_start (ignores cv_test_frac).
    cv_val_horizon : int
        Validation block length per fold (years).
    cv_min_train : int
        Smallest train fold size (years).
    cv_step : int or None
        Origin spacing; None ⇒ uniform.
    cv_seed_per_fold : bool
        If True (default), seed = base_seed + fold so fold models have
        independent inits — required for genuine ensemble diversity.  Set
        False to use the same seed across folds (debug / reproducibility).
    train_kwargs : dict
        Forwarded to ``train_model``.  ``trainval_frac``, ``val_frac``, and
        ``split_indices`` are ignored / overridden.

    Returns
    -------
    RollingOriginResult
    """
    import copy as _copy

    # Load just enough to know T (cheap — no JAX yet)
    data_np = load_zinc_data(
        xlsx_path,
        extra_exog_cols=train_kwargs.get("extra_exog_cols"),
    )
    T = len(data_np["years"])
    years_all = np.asarray(data_np["years"]).astype(int)

    splits = rolling_origin_splits(
        T,
        test_frac=cv_test_frac,
        test_start=cv_test_start,
        n_folds=cv_n_folds,
        val_horizon=cv_val_horizon,
        min_train=cv_min_train,
        step=cv_step,
    )
    K = len(splits)
    test_start = splits[0]["test_start"]

    if verbose:
        print(f"\n=== Rolling-origin CV: {K} folds, T={T}, "
              f"held-out test = years[{test_start}:{T}] "
              f"({int(years_all[test_start])}–{int(years_all[-1])}) ===")
        for s in splits:
            yr_tr  = (int(years_all[0]),               int(years_all[s["train_end"]]))
            yr_val = (int(years_all[s["train_end"]+1]), int(years_all[s["val_end"]]))
            yr_te  = (int(years_all[s["test_start"]]), int(years_all[-1]))
            print(f"  fold {s['fold']}: train {yr_tr[0]}–{yr_tr[1]} "
                  f"({s['train_end']+1}y)   val {yr_val[0]}–{yr_val[1]} "
                  f"({s['val_end'] - s['train_end']}y)   "
                  f"test {yr_te[0]}–{yr_te[1]} ({T - s['test_start']}y, fixed)")

    # Strip caller-supplied conflicting kwargs
    base_kwargs = _copy.deepcopy(train_kwargs)
    for k in ("trainval_frac", "val_frac", "split_indices"):
        base_kwargs.pop(k, None)
    base_seed = int(base_kwargs.pop("seed", 0))

    fits = []
    t0 = time.time()
    for s in splits:
        if verbose:
            print(f"\n--- [CV] fold {s['fold']}/{K-1} "
                  f"(train_end={s['train_end']}, val_end={s['val_end']}) ---")
        fold_kwargs = _copy.deepcopy(base_kwargs)
        fold_kwargs["seed"] = base_seed + s["fold"] if cv_seed_per_fold else base_seed
        fold_kwargs["split_indices"] = {
            "train_end":  s["train_end"],
            "val_end":    s["val_end"],
            "test_start": s["test_start"],
        }
        fold_kwargs.setdefault("verbose", verbose)

        (params, params_A, layout, dt, dv, dtv, dte, da,
         integrate_aug, nn_eval, nn_eval_sup,
         flow_obs_to_pred_idx) = train_model(xlsx_path, **fold_kwargs)

        fit = FitResult(
            name=f"fold_{s['fold']}",
            cfg=fold_kwargs,
            params=params, params_A=params_A, layout=layout,
            data_train=dt, data_val=dv, data_trainval=dtv,
            data_test=dte, data_all=da,
            integrate_aug=integrate_aug,
            nn_eval=nn_eval, nn_eval_sup=nn_eval_sup,
            flow_obs_to_pred_idx=flow_obs_to_pred_idx,
            do_stage_B=bool(fold_kwargs.get("do_stage_B", False)),
        )
        fits.append((s, fit))

    if verbose:
        print(f"\n[CV] all {K} folds finished in {time.time() - t0:.1f}s")

    return RollingOriginResult(splits=splits, fits=fits, T=T,
                               years_all=years_all)


# ----------------------------------------------------------------------------
# Aggregator class
# ----------------------------------------------------------------------------
class RollingOriginResult:
    """Aggregated output of ``train_model_rolling_origin``.

    Attributes
    ----------
    splits     : list of dicts (fold specs)
    fits       : list of (split_dict, FitResult) tuples, K entries
    T          : int, length of the series
    years_all  : 1-D np.array of integer years
    K          : int, number of folds

    Public methods
    --------------
    fold_summaries(kind="testrun", prefer="auto") -> list of dicts
        Per-fold ``FitResult.summary`` outputs (no printing).

    cv_summary(kind="testrun", prefer="auto", verbose=True) -> dict
        Aggregate (mean ± std) of per-fold family-mean MAPE / relRMSE on
        the held-out test, plus a per-fold table.  ``kind="testrun"``
        rolls each fold's ODE from the FIXED test IC, so cross-fold
        comparison is honest (same evaluation window, same IC).

    ensemble_predictions(prefer="auto") -> dict
        Per-fold + ensemble-averaged trajectories on the held-out test.
        Trajectory averaging: each fold integrates from S_obs(test_start-1)
        with its own NN, then we average S_pred and F_int pointwise across
        the K folds.

    ensemble_diagnose(prefer="auto", verbose=True) -> dict
        Family-mean MAPE / relRMSE of the ENSEMBLE prediction on the
        held-out test.  This is the headline test metric for v5.
    """

    def __init__(self, *, splits, fits, T, years_all):
        self.splits     = splits
        self.fits       = list(fits)
        self.T          = int(T)
        self.years_all  = np.asarray(years_all).astype(int)
        if not self.fits:
            raise ValueError("RollingOriginResult requires at least one fold.")
        self.test_start = int(splits[0]["test_start"])

    @property
    def K(self):
        return len(self.fits)

    # ------------------------------------------------------------------
    # Per-fold helpers
    # ------------------------------------------------------------------
    def _fit_for(self, fold):
        return self.fits[fold][1]

    def fold_summaries(self, kind="testrun", *, prefer="auto",
                       include_pinned=False):
        out = []
        for split, fit in self.fits:
            d = fit.summary(kind=kind, prefer=prefer,
                            include_pinned=include_pinned, verbose=False)
            d = dict(d)
            d["fold"]      = split["fold"]
            d["train_end"] = split["train_end"]
            d["val_end"]   = split["val_end"]
            out.append(d)
        return out

    # ------------------------------------------------------------------
    # CV summary (per-fold + aggregate)
    # ------------------------------------------------------------------
    def cv_summary(self, kind="testrun", *, prefer="auto",
                   include_pinned=False, verbose=True):
        rows = self.fold_summaries(kind=kind, prefer=prefer,
                                   include_pinned=include_pinned)

        # families to aggregate
        if kind == "freerun":
            stocks_key, flows_key = "stocks_freerun", "flows_freerun"
        elif kind == "testrun":
            stocks_key, flows_key = "stocks_testrun", "flows_testrun"
        else:
            stocks_key, flows_key = "stocks_freerun", "flows_freerun"

        family_keys = [
            ("alphas",  "alphas"),
            ("taus",    "taus"),
            ("cp",      "cp"),
            ("stocks",  stocks_key),
            ("flows",   flows_key),
        ]
        metric_keys = ("MAPE%", "relRMSE%")

        agg = {}
        for label, src in family_keys:
            for mk in metric_keys:
                vals = np.asarray(
                    [float(r.get(src, {}).get(mk, np.nan)) for r in rows],
                    dtype=float,
                )
                vals_finite = vals[np.isfinite(vals)]
                agg[f"{label}_{mk}_mean"]   = (float(vals_finite.mean())
                                                if vals_finite.size else float("nan"))
                agg[f"{label}_{mk}_std"]    = (float(vals_finite.std(ddof=1))
                                                if vals_finite.size > 1 else 0.0)
                agg[f"{label}_{mk}_perfold"] = vals.tolist()

        agg["K"]          = self.K
        agg["test_start"] = self.test_start
        agg["kind"]       = kind

        if verbose:
            ya = self.years_all
            yr_test = (int(ya[self.test_start]), int(ya[-1]))
            print(f"\n=== CV summary  (K={self.K} folds, "
                  f"test {yr_test[0]}–{yr_test[1]} fixed, kind={kind!r}) ===")
            # Per-fold table
            hdr = (f"{'fold':>4s}  {'train_end':>9s}  {'val_end':>7s}  "
                   f"{'α MAPE':>8s}  {'τ MAPE':>8s}  "
                   f"{'S MAPE':>8s}  {'F MAPE':>8s}  "
                   f"{'S relR':>8s}  {'F relR':>8s}")
            print(hdr)
            print("-" * len(hdr))
            for r in rows:
                a_m = float(r.get("alphas",   {}).get("MAPE%",     np.nan))
                t_m = float(r.get("taus",     {}).get("MAPE%",     np.nan))
                s_m = float(r.get(stocks_key, {}).get("MAPE%",     np.nan))
                f_m = float(r.get(flows_key,  {}).get("MAPE%",     np.nan))
                s_r = float(r.get(stocks_key, {}).get("relRMSE%",  np.nan))
                f_r = float(r.get(flows_key,  {}).get("relRMSE%",  np.nan))
                print(f"{r['fold']:>4d}  {r['train_end']:>9d}  {r['val_end']:>7d}  "
                      f"{a_m:>8.4f}  {t_m:>8.4f}  "
                      f"{s_m:>8.4f}  {f_m:>8.4f}  "
                      f"{s_r:>8.4f}  {f_r:>8.4f}")
            # Aggregate row
            print("-" * len(hdr))
            print(f"{'mean':>4s}  {'':>9s}  {'':>7s}  "
                  f"{agg['alphas_MAPE%_mean']:>8.4f}  "
                  f"{agg['taus_MAPE%_mean']:>8.4f}  "
                  f"{agg['stocks_MAPE%_mean']:>8.4f}  "
                  f"{agg['flows_MAPE%_mean']:>8.4f}  "
                  f"{agg['stocks_relRMSE%_mean']:>8.4f}  "
                  f"{agg['flows_relRMSE%_mean']:>8.4f}")
            print(f"{'std':>4s}  {'':>9s}  {'':>7s}  "
                  f"{agg['alphas_MAPE%_std']:>8.4f}  "
                  f"{agg['taus_MAPE%_std']:>8.4f}  "
                  f"{agg['stocks_MAPE%_std']:>8.4f}  "
                  f"{agg['flows_MAPE%_std']:>8.4f}  "
                  f"{agg['stocks_relRMSE%_std']:>8.4f}  "
                  f"{agg['flows_relRMSE%_std']:>8.4f}")

        return agg

    # ------------------------------------------------------------------
    # Ensemble prediction
    # ------------------------------------------------------------------
    def ensemble_predictions(self, *, prefer="auto"):
        """Average per-fold testrun trajectories on the held-out test.

        Returns
        -------
        dict with keys:
          "S_pred"     : (n_test_rows, N_STOCKS), ensemble-mean stocks
          "F_int"      : (n_test_rows-1, N_FLOWS), ensemble-mean integrated flows
          "S_pred_perfold" : (K, n_test_rows, N_STOCKS)
          "F_int_perfold"  : (K, n_test_rows-1, N_FLOWS)
          "alphas_at_obs"  : (n_test_rows, N_ALPHAS), ensemble-mean
          "taus_at_obs"    : (n_test_rows, N_TAUS_BINARY), ensemble-mean
          "frac_fu_at_obs" : (n_test_rows, 2), ensemble-mean
          "frac_eu_at_obs" : (n_test_rows, 2), ensemble-mean
          "cp_at_obs"      : (n_test_rows,), ensemble-mean
          "years"          : (n_test_rows,) — same across folds (test fixed)
        """
        per_S, per_F = [], []
        per_alpha, per_tau, per_fu, per_eu, per_cp = [], [], [], [], []
        years_ref = None
        for split, fit in self.fits:
            mode = ("B" if (prefer == "auto" and fit._stage_B_ran)
                    else (prefer if prefer in ("A", "B") else "A"))
            if mode == "B" and not fit._stage_B_ran:
                mode = "A"
            preds_tr = fit._predictions_testrun(mode)
            per_S.append(np.asarray(preds_tr["S_pred"]))
            per_F.append(np.asarray(preds_tr["F_int"]))
            per_alpha.append(np.asarray(preds_tr["alphas_at_obs"]))
            per_tau.append(np.asarray(preds_tr["taus_at_obs"]))
            per_fu.append(np.asarray(preds_tr["frac_fu_at_obs"]))
            per_eu.append(np.asarray(preds_tr["frac_eu_at_obs"]))
            per_cp.append(np.asarray(preds_tr["cp_at_obs"]))
            yrs = np.asarray(fit.data_test["years"])
            years_ref = yrs if years_ref is None else years_ref
        per_S = np.stack(per_S, axis=0)        # (K, n_rows, N_STOCKS)
        per_F = np.stack(per_F, axis=0)        # (K, n_rows-1, N_FLOWS)

        return dict(
            S_pred         = per_S.mean(axis=0),
            F_int          = per_F.mean(axis=0),
            S_pred_perfold = per_S,
            F_int_perfold  = per_F,
            alphas_at_obs  = np.stack(per_alpha, axis=0).mean(axis=0),
            taus_at_obs    = np.stack(per_tau,   axis=0).mean(axis=0),
            frac_fu_at_obs = np.stack(per_fu,    axis=0).mean(axis=0),
            frac_eu_at_obs = np.stack(per_eu,    axis=0).mean(axis=0),
            cp_at_obs      = np.stack(per_cp,    axis=0).mean(axis=0),
            years          = np.asarray(years_ref),
        )

    # ------------------------------------------------------------------
    # Ensemble diagnostic on held-out test
    # ------------------------------------------------------------------
    def ensemble_diagnose(self, *, prefer="auto", include_pinned=False,
                          verbose=True):
        """Family-mean MAPE / relRMSE of the K-fold ensemble on held-out test.

        Uses the testrun rollout (each fold integrates from FIXED test IC
        through the test window).  Trajectory averaging across folds.
        """
        ens = self.ensemble_predictions(prefer=prefer)

        # Build masks: exclude IC row (predicted == observed by construction)
        fit0 = self._fit_for(0)
        s_obs_test = np.asarray(fit0.data_test["stocks_obs"])
        F_obs_test = np.asarray(fit0.data_test["flows_obs"])
        alpha_obs_test = np.asarray(fit0.data_test["alpha_obs"])
        tau_obs_test   = np.asarray(fit0.data_test["tau_sup_obs"])
        cp_obs_test    = np.asarray(fit0.data_test["cp_obs"])
        n_rows = s_obs_test.shape[0]
        m_test = np.zeros(n_rows, dtype=bool); m_test[1:] = True
        m_flow = np.ones(max(n_rows - 1, 0), dtype=bool)

        # Pinning flags (uniform across folds — same cfg)
        cp_pinned   = bool(fit0.layout["pin_cp"])
        tau_pinned  = fit0._pinned_flags_per_tau_sup()
        f_idx       = fit0.flow_obs_to_pred_idx
        flow_names_native = [FLOW_NAMES[i] for i in f_idx]
        flow_learned_mask = np.array([
            not (cp_pinned and nm == "concentrate_production")
            for nm in flow_names_native
        ], dtype=bool)
        if include_pinned:
            flow_learned_mask[:] = True

        # alpha (always learned)
        a_pre = ens["alphas_at_obs"]
        alpha_mape  = _nanmean_safe(
            [_mape_pct(    a_pre[m_test, k], alpha_obs_test[m_test, k])
             for k in range(N_ALPHAS)])
        alpha_rrmse = _nanmean_safe(
            [_rel_rmse_pct(a_pre[m_test, k], alpha_obs_test[m_test, k])
             for k in range(N_ALPHAS)])

        # taus
        tau_mape_list, tau_rrmse_list = [], []
        for j in range(N_TAU_SUP):
            if tau_pinned[j] and not include_pinned:
                continue
            if j < 4:
                pre = ens["taus_at_obs"][:, j]
            elif j < 6:
                pre = ens["frac_fu_at_obs"][:, j - 4]
            else:
                pre = ens["frac_eu_at_obs"][:, j - 6]
            tau_mape_list.append( _mape_pct(    pre[m_test], tau_obs_test[m_test, j]))
            tau_rrmse_list.append(_rel_rmse_pct(pre[m_test], tau_obs_test[m_test, j]))
        tau_mape  = _nanmean_safe(tau_mape_list)
        tau_rrmse = _nanmean_safe(tau_rrmse_list)

        # cp
        if cp_pinned and not include_pinned:
            cp_mape, cp_rrmse = float("nan"), float("nan")
        else:
            cp_pre = ens["cp_at_obs"]
            cp_mape  = _mape_pct(    cp_pre[m_test], cp_obs_test[m_test])
            cp_rrmse = _rel_rmse_pct(cp_pre[m_test], cp_obs_test[m_test])

        # stocks (ensemble)
        S_pre = ens["S_pred"]
        s_mape  = _nanmean_safe(
            [_mape_pct(    S_pre[m_test, k], s_obs_test[m_test, k])
             for k in range(N_STOCKS)])
        s_rrmse = _nanmean_safe(
            [_rel_rmse_pct(S_pre[m_test, k], s_obs_test[m_test, k])
             for k in range(N_STOCKS)])

        # flows (ensemble)
        F_pre = ens["F_int"]
        f_mape_list, f_rrmse_list = [], []
        for k in range(F_obs_test.shape[1]):
            if not flow_learned_mask[k]:
                continue
            f_mape_list.append( _mape_pct(    F_pre[m_flow, int(f_idx[k])], F_obs_test[m_flow, k]))
            f_rrmse_list.append(_rel_rmse_pct(F_pre[m_flow, int(f_idx[k])], F_obs_test[m_flow, k]))
        f_mape  = _nanmean_safe(f_mape_list)
        f_rrmse = _nanmean_safe(f_rrmse_list)

        out = {
            "alphas":         {"MAPE%": alpha_mape,  "relRMSE%": alpha_rrmse},
            "taus":           {"MAPE%": tau_mape,    "relRMSE%": tau_rrmse},
            "cp":             {"MAPE%": cp_mape,     "relRMSE%": cp_rrmse,
                               "pinned": bool(cp_pinned)},
            "stocks":         {"MAPE%": s_mape,      "relRMSE%": s_rrmse},
            "flows":          {"MAPE%": f_mape,      "relRMSE%": f_rrmse},
            "K":              self.K,
            "kind":           "testrun_ensemble",
        }

        if verbose:
            ya = self.years_all
            yr_lo, yr_hi = int(ya[self.test_start]), int(ya[-1])
            print(f"\n=== Ensemble diagnose  ({self.K}-fold avg, testrun, "
                  f"test {yr_lo}–{yr_hi}) ===")
            pin_tag = "incl." if include_pinned else "excl."
            print(f"  family means (pinned={pin_tag}):")
            for fam in ("alphas", "taus", "cp", "stocks", "flows"):
                m = out[fam]["MAPE%"]
                r = out[fam]["relRMSE%"]
                pin = " [pinned]" if out[fam].get("pinned") else ""
                print(f"    {fam:<8s}  MAPE = {m:7.4f}    relRMSE = {r:7.4f}{pin}")

        return out


# ----------------------------------------------------------------------------
# Convenience entry point
# ----------------------------------------------------------------------------
ROLLING_ORIGIN_DEFAULTS = dict(
    cv_n_folds=5,
    cv_test_frac=0.30,
    cv_val_horizon=10,
    cv_min_train=15,
    cv_step=None,
    cv_test_start=None,
    cv_seed_per_fold=True,
)


def run_cv(name="cv", **overrides):
    """Convenience entry point: rolling-origin CV with v4 defaults + fold ensemble.

    Splits ``overrides`` into CV-specific kwargs (those starting with ``cv_``)
    and passes the rest through to ``train_model`` per fold.

    Examples
    --------
    >>> res = run_cv("AB_cv", do_stage_B=True, cv_n_folds=5,
    ...              stageB_curriculum=[(800, 8), (800, 16), (400, 22)])
    >>> res.cv_summary()           # per-fold + aggregate
    >>> res.ensemble_diagnose()    # K-fold avg test metrics
    """
    train_cfg = dict(DEFAULT_CONFIG)
    cv_cfg    = dict(ROLLING_ORIGIN_DEFAULTS)

    for k, v in overrides.items():
        if k.startswith("cv_"):
            cv_cfg[k] = v
        else:
            train_cfg[k] = v

    print(f"\n=== zinc_colloc_v5 run_cv: {name} ===")
    res = train_model_rolling_origin(**cv_cfg, **train_cfg)
    res.cv_summary()
    res.ensemble_diagnose()
    return res


if __name__ == "__main__":
    # ---- (a) Legacy single-split path (v4-compatible) ----
    fit_A  = run("stageA_only", do_stage_B=False)

    # Optional: Stage A + Stage B (shooting fine-tune)
    # fit_AB = run("stageA_then_B", do_stage_B=True,
    #              stageB_curriculum=[(800, 8), (800, 16), (400, 22)])

    # ---- (b) NEW in v5: Rolling-origin CV with fold ensembling ----
    # Default: 5 folds, expanding window, val_horizon=10y, test=last ~30%.
    # Each fold refits NN + Stage A; fold models then ensemble on held-out test.
    # cv_res = run_cv("stageA_cv", do_stage_B=False)

    # With Stage B (slower — ~5x cost, but per-fold variance bars + ensembling
    # typically beat any single-split refit on the held-out test):
    # cv_res = run_cv("AB_cv", do_stage_B=True,
    #                 stageB_curriculum=[(800, 8), (800, 16), (400, 22)],
    #                 cv_n_folds=5, cv_val_horizon=10)