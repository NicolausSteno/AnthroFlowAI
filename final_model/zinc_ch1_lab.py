#!/usr/bin/env python3
"""
zinc_ch1_lab.py — lab module for WP-9: Chapter 1 hypothesis tests
=================================================================

WP-9 asks whether the coefficients the UDE learned agree with what Chapter 1
(`ch1_macro_materials.tex`) predicts.  Chapter 1 derives the comparative
dynamics of a capital--material growth model and of metal intensity of use;
Chapter 4 supplies the first empirically estimated time-varying coefficients
for the same cycle.  The two have never been confronted.  This module does the
confrontation, and reports contradictions with the same prominence as
confirmations (spec, WP-9).

Nothing is refitted.  Every empirical number is read off the `anchor_v4`
weight dumps in `analysis/wp2a/`, either directly or through one new
forward-mode solve per seed.

What Chapter 1 actually predicts
--------------------------------
Chapter 1's material block, in per-capita form (Ch1 Eqs. `oc_udot`,
`oc_ddot`, `oc_s_reduced`), gives the stationary circularity index

    chi* = (sigma + lambda * B_U) * [ rho + gamma*q*(1-rho)/(n+gamma) ],     (Ch1 Eq. chistar / ext_chi_factorised)
    B_U  = (1-sigma)*kappa/(n+lambda),
    B_D  = q*(1-rho)*(lambda*B_U + sigma)/(n+gamma),

and the whole recovery loop then enters the reduced-form technology term only
through the scalar multiplier `(1-chi)^-1` (Ch1 Prop. `ext_level_not_growth`),
so that for any recovery or mass-balance parameter `z`

    dln k*/dz = nu * chi_z / [(1-alpha-vartheta*nu)(1-chi)],                 (Ch1 Eq. ext_master_elasticities)
    dln s*/dz = (1-alpha) * chi_z / [(1-alpha-vartheta*nu)(1-chi)],
    dln u*/dz = dln B_U/dz + dln s*/dz,
    dln d*/dz = dln B_D/dz + dln s*/dz.

Chapter 1 is per-capita and Chapter 4 is aggregate mass.  The translation is
exact and is used throughout: differentiating `U/L` introduces the `n` that
appears in `n+lambda` and `n+gamma`, so in aggregate terms `n` is replaced by
the growth rate of aggregate throughput.  Chapter 1's own balanced-growth
version does the same thing with `n+lambda -> n+lambda+vartheta*g`
(Ch1 Eq. `ext_dgk_dz`).  One consequence carries the whole of H2: `chi`
*falls* when the system grows faster, because the stock-based recovery loop is
diluted.  At the zinc calibration `dchi/dg = -4.87` per unit growth rate.

The five Chapter 1 parameters, measured on the UDE
--------------------------------------------------
Chapter 1 calibrated `(lambda, sigma, kappa, rho)` for zinc directly from
Rostek et al. (2022) as flow ratios (Ch1 Sec. `calib`, Table
`tab:calibration`).  The identical ratios are reconstructed here from the
UDE's own period-integral flows and point-in-time stocks, so the comparison is
like for like:

    lambda = end_of_life / S_inuse                (mid-interval stock)
    sigma  = (new scrap + manufacturing losses) / throughput
    kappa  = inuse_inflow / total_products_into_use   ( = 1 - tau_diss )
    rho    = (waelz_recycling + direct_reuse) / (end_of_life + new scrap)
    chi    = (waelz_recycling + direct_reuse) / throughput
    throughput = refined_consumption + direct_reuse_recycling

`throughput` is Chapter 1's `s`: "total usable metal throughput ... refined
production together with direct reuse required by the material balance"
(Ch1 Sec. `calib`), which is exactly the UDE's `parent_manu = refc + dr`
(v5:1168).  `gamma` and `q` have no UDE analogue — the UDE carries no landfill
stock — and `gamma = 0` in Chapter 1's zinc baseline anyway, so every
`gamma`-dependent prediction is reported as not testable rather than forced.

Chapter 1's `d` is *not* the UDE's Scrap stock.  Ch1 Sec. `calib` builds
`d_Zn` as "the cumulative unrecovered end-of-life flow reconstructed from
Rostek", explicitly "outside current scrap markets"; the UDE's Scrap is the
in-market pool that `alpha_win` and `alpha_dr` draw on.  The Chapter 1
analogue is therefore reconstructed the way Chapter 1 built it, as the running
integral of the unrecovered streams.

Two perturbations, and why both are needed
------------------------------------------
WP-8c perturbs a driver's *level* (a one-SD additive shift of the whole raw
history).  Chapter 1's `n` and `g` are *growth rates*, and a level shift is not
a growth shift, so H2 cannot be tested on the 8c artefact.  `sensitivity()`
therefore runs one `jax.jacfwd` over a 26-vector: the 13 level shifts, in 8c's
exact convention so the result can be checked against
`analysis/wp8c_forward_sens.npz`, and 13 growth-rate rotations

    X_j(t) -> X_j(t) * exp(g_j * (t - t_mid)),

centred on the sample midpoint so that a growth perturbation is not also a
level perturbation.  `g_j` is in yr^-1 and results are quoted per +1 pp/yr.

Everything downstream — `lambda`, `sigma`, `kappa`, `rho`, `chi`, throughput,
the in-use stock, the reconstructed discard stock — is a smooth function of
the returned `(S, F)`, so its driver derivative follows by the exact quotient
rule rather than by a second finite difference.  `check()` verifies that chain
rule against a central difference of the solver itself.

Pattern
-------
Follows `zinc_cf_lab.py` / `zinc_xai_lab.py` (CLAUDE.md rule 1).
`zinc_colloc_v5.py` is imported and never edited.  `PATCHES` is empty:
everything here is composed from the core's public factories and from
`zinc_xai_lab.make_forward_integrator`, whose `ForwardMode` integrator WP-8c
already verified reproduces the fitted trajectory exactly.  Integrity
checking, config loading and the driver/`input_dim` part of `--check` come
from `zinc_alpha_lab`.

The hypotheses are pre-registered in `HYPOTHESES` below, fixed in source
before any empirical number was computed — the same discipline `zinc_xai_lab`
applies to its `EVENTS` list.

CLI
---
    python zinc_ch1_lab.py --check
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys

import numpy as np

import zinc_alpha_lab as alab
from zinc_alpha_lab import integrity_check, load_anchor_config      # noqa: F401
import zinc_cf_lab as cflab
import zinc_xai_lab as xlab

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR_DEFAULT = os.path.join(HERE, "analysis", "wp9")
WEIGHTS_DIR_DEFAULT = os.path.join(HERE, "analysis", "wp2a")
ANCHOR_DIR = os.path.join(HERE, "anchor_v4")

PATCHES: list[str] = []          # WP-9 needs none — see module docstring


def install(v5mod=None):
    """Apply import-time patches to the core module.  None needed for WP-9."""
    if v5mod is None:
        import zinc_colloc_v5 as v5mod          # noqa: F401
    return PATCHES


# ===========================================================================
# 1.  Chapter 1, as a computable object
# ===========================================================================

# Ch1 Table `tab:calibration`, zinc column.  `gamma = 0` is Chapter 1's zinc
# baseline; `GAMMA_SENS` is its stated sensitivity case, in which `rho` is
# "adjusted mechanically to avoid double counting the same recovered material".
CH1_ZINC = dict(lam=0.0375, sig=0.179, rho=0.363, gam=0.0,
                kap=0.983, q=0.300, n=0.01)
CH1_ZINC_GAMMA_SENS = dict(CH1_ZINC, gam=0.003, rho=0.307)

# Ch1 Table `tab:steady_state_stability`, zinc rows — the targets `ch1_state`
# must reproduce.
CH1_ZINC_STEADY = dict(s_star=2.89, u_star=49.06, d_star=45.03, chi_star=0.296)
CH1_ZINC_STEADY_GAMMA_SENS = dict(s_star=2.69, u_star=45.76, d_star=35.15,
                                  chi_star=0.290)

# Reference in-use and discard stocks Ch1 calibrated from the same MFA sources
# (Ch1 Table `tab:calibration`, lower block), kg/cap.
CH1_ZINC_REFERENCE = dict(u_ref=32.1, d_ref=24.36)

# Chapter 1 does not calibrate the production block of the capital--material
# extension.  It states one worked numerical example (Ch1 Sec.
# `capital_material_extension`, the paragraph beginning "The magnitudes
# involved are worth discussing"): alpha = 0.3, nu = 0.01, g_Y = 0.02, with nu
# argued to be "of the order of a few tenths of one per cent of world GDP" for
# an individual metal.  vartheta is never pinned beyond vartheta <= 1.  Those
# are used as the central case and every result that depends on them is
# reported over the sensitivity grid below rather than at a point.
CH1_PRODUCTION = dict(alpha=0.30, nu=0.01, vartheta=1.0)
CH1_PRODUCTION_GRID = dict(alpha=(0.25, 0.30, 0.40),
                           nu=(0.003, 0.01, 0.03, 0.10),
                           vartheta=(0.50, 0.75, 1.00))


def ch1_BU(p, g=0.0, vartheta=1.0):
    """`B_U` — in-use stock per unit throughput.  Ch1 Eq. `ext_au`.

    `g` is the growth rate of aggregate throughput; Chapter 1's balanced-growth
    form replaces `n+lambda` by `n+lambda+vartheta*g` (Ch1 Eq. `ext_dgk_dz`).
    """
    return (1.0 - p["sig"]) * p["kap"] / (p["n"] + p["lam"] + vartheta * g)


def ch1_BD(p, g=0.0, vartheta=1.0):
    """`B_D` — recoverable discard stock per unit throughput.  Ch1 Eq. `ext_ad`."""
    num = p["q"] * (1.0 - p["rho"]) * (p["lam"] * ch1_BU(p, g, vartheta) + p["sig"])
    return num / (p["n"] + p["gam"] + vartheta * g)


def ch1_chi(p, g=0.0, vartheta=1.0):
    """Circularity index.  Ch1 Eq. `ext_chi_factorised`.

    The secondary share of usable throughput: the recoverable share of
    throughput times the recovery propensity.
    """
    den = p["n"] + p["gam"] + vartheta * g
    prop = p["rho"] + (p["gam"] * p["q"] * (1.0 - p["rho"]) / den if den > 0 else 0.0)
    return (p["sig"] + p["lam"] * ch1_BU(p, g, vartheta)) * prop


def ch1_chi_derivs(p, g=0.0, vartheta=1.0):
    """Analytic `chi_z` for the parameters Chapter 1 gives in closed form,
    plus `dchi/dn` and `dchi/dg` by central difference on `ch1_chi`.

    Ch1 Eqs. `ext_chi_rho_q`, `ext_chi_kappa_gamma`, `ext_sigma_ambiguity`.
    The closed forms are returned *and* cross-checked numerically; `check()`
    reports the worst gap.
    """
    n, lam, sig, kap, rho, gam, q = (p["n"], p["lam"], p["sig"], p["kap"],
                                     p["rho"], p["gam"], p["q"])
    nl = n + lam + vartheta * g
    ng = n + gam + vartheta * g
    BU = ch1_BU(p, g, vartheta)
    BD = ch1_BD(p, g, vartheta)
    closed = {
        # Ch1 Eq. ext_chi_rho_q
        "rho": (sig + lam * BU) * (ng - gam * q) / ng,
        "q": (gam * BD / q) if q > 0 else 0.0,
        # Ch1 Eq. ext_chi_kappa_gamma
        "kap": (lam * (1.0 - sig) / nl) * (rho + gam * q * (1.0 - rho) / ng),
        "gam": (n + vartheta * g) / ng * BD if ng > 0 else 0.0,
        # Ch1 Eq. ext_sigma_ambiguity
        "sig": ((n + lam * (1.0 - kap) + vartheta * g) / nl
                * (rho + gam * q * (1.0 - rho) / ng)),
    }
    numeric = {k: _central(lambda pp: ch1_chi(pp, g, vartheta), p, k)
               for k in ("rho", "q", "kap", "gam", "sig", "lam", "n")}
    numeric["g"] = ((ch1_chi(p, g + 1e-6, vartheta)
                     - ch1_chi(p, g - 1e-6, vartheta)) / 2e-6)
    gap = max(abs(closed[k] - numeric[k]) for k in closed)
    return dict(closed=closed, numeric=numeric, max_closed_vs_numeric=float(gap))


def _central(f, p, key, h=1e-6):
    a, b = dict(p), dict(p)
    a[key] -= h
    b[key] += h
    return (f(b) - f(a)) / (2.0 * h)


def ch1_elasticities(p, prod=None, g=0.0):
    """`dln k*/dz`, `dln s*/dz`, `dln u*/dz`, `dln d*/dz`.  Ch1 Eq.
    `ext_master_elasticities`, with the `B_U`/`B_D` composition terms added for
    the stocks as Ch1 Sec. `Comparative dynamics` specifies."""
    prod = prod or CH1_PRODUCTION
    al, nu, th = prod["alpha"], prod["nu"], prod["vartheta"]
    chi = ch1_chi(p, g, th)
    kern = 1.0 / ((1.0 - al - th * nu) * (1.0 - chi))
    d = ch1_chi_derivs(p, g, th)["numeric"]
    out = {}
    for z in ("rho", "gam", "kap", "q", "sig", "lam", "n"):
        chz = d[z]
        dlnk = nu * chz * kern
        dlns = (1.0 - al) * chz * kern
        dlnBU = _central(lambda pp: np.log(ch1_BU(pp, g, th)), p, z)
        dlnBD = _central(lambda pp: np.log(ch1_BD(pp, g, th)), p, z)
        out[z] = dict(chi_z=chz, dlnk=dlnk, dlns=dlns,
                      dlnu=dlnBU + dlns, dlnd=dlnBD + dlns)
    return out


def ch1_multiplier(p, prod=None, g=0.0):
    """`dln s*/dchi` split into the mass-balance multiplier and the
    general-equilibrium amplification it is multiplied by.

    Ch1 Eq. `ext_dlns_dchi_simplified`:
        dln s*/dchi = (1-alpha) / [(1-alpha-vartheta*nu)(1-chi)]
                    = [1/(1-chi)] * [(1-alpha)/(1-alpha-vartheta*nu)].
    The first factor is pure mass balance and is inside the UDE; the second is
    the macro feedback the UDE does not carry.
    """
    prod = prod or CH1_PRODUCTION
    al, nu, th = prod["alpha"], prod["nu"], prod["vartheta"]
    chi = ch1_chi(p, g, th)
    mass = 1.0 / (1.0 - chi)
    amp = (1.0 - al) / (1.0 - al - th * nu)
    return dict(chi=chi, mass_balance=mass, ge_amplification=amp,
                total=mass * amp)


def ch1_state(p, prod=None, s_star=None):
    """Chapter 1's own steady state, for the reproduction check against Ch1
    Table `tab:steady_state_stability`."""
    prod = prod or CH1_PRODUCTION
    chi = ch1_chi(p)
    BU, BD = ch1_BU(p), ch1_BD(p)
    s = s_star if s_star is not None else CH1_ZINC_STEADY["s_star"]
    return dict(chi=chi, BU=BU, BD=BD, u=BU * s, d=BD * s)


# ---------------------------------------------------------------------------
# The two appendix extensions (`app_ch1_trans_dyn.tex`)
# ---------------------------------------------------------------------------
# Both are valuation/technology refinements layered on the same tonnage
# balance, so what the UDE can and cannot see differs sharply between them and
# is worked out explicitly rather than assumed.

def ch1_X_multiplier(chi, pi):
    """Realised circularity multiplier.  Ch1 App. Eq. `pi_Xcal_def`:

        X(chi, pi) = [1 - (1-pi) chi] / (1 - chi) = 1 + pi*chi/(1-chi),

    the factor by which the recovery loop leverages a unit of primary supply
    into *usable services*.  It runs from 1 at `pi = 0`, where circularity buys
    no service dividend at all, to the full `(1-chi)^-1` at `pi = 1`.
    """
    chi = np.asarray(chi, float)
    pi = np.asarray(pi, float)
    return 1.0 + pi * chi / (1.0 - chi)


def ch1_slope_with_pi(chi, pi, prod=None):
    """`dln s/dchi` under the substitutability extension.

    The tonnage balance is untouched by `pi` (Ch1 App. Sec. `pi_extension`:
    "deliberately a valuation change, not a physical one"), so `pi` reaches the
    throughput response only through the capital block, whose technology term
    now carries `X(chi, pi)` instead of `(1-chi)^-1`:

        dln s*/dchi = 1/(1-chi)
                    + vartheta*nu*pi / [(1-alpha-vartheta*nu)(1-chi)(1-(1-pi)chi)].

    At `pi = 1` this collapses to Ch1 Eq. `ext_dlns_dchi_simplified`, which is
    what `ch1_multiplier` returns.  The whole `pi` dependence therefore sits
    inside the general-equilibrium term that H5 measures at ~1.4% — which is
    the point of H9.
    """
    prod = prod or CH1_PRODUCTION
    al, nu, th = prod["alpha"], prod["nu"], prod["vartheta"]
    chi, pi = float(chi), float(pi)
    return (1.0 / (1.0 - chi)
            + th * nu * pi / ((1.0 - al - th * nu) * (1.0 - chi)
                              * (1.0 - (1.0 - pi) * chi)))


def ch1_s_rho(par):
    """`dln s/drho` in the short run.  Ch1 App. Eq. `rho_s_rho`:

        s_rho = (lambda*u + sigma*s)/(1 - rho*sigma),

    the material-balance derivative at fixed stocks, where `lambda*u + sigma*s`
    is "the collectable flow of end-of-life metal and fabrication scrap".
    Returned in elasticity form, `s_rho/s`, so it is comparable with the
    measured pooled slope.  This is a closed form with **no free parameters**
    once the UDE's own `(lambda, sigma, rho, u, s)` are substituted, which
    makes it the sharpest test the recovery-capacity extension offers.
    """
    return ((par["lam"] * par["inuse"] / par["throughput"] + par["sig"])
            / (1.0 - par["rho"] * par["sig"]))


def collectable_flow(par):
    """`lambda*u + sigma*s` — Ch1 App. Eq. `rho_s_rho`'s collectable flow, and
    the observable proxy for the scale the collection system must process."""
    return par["lam"] * par["inuse"] + par["sig"] * par["throughput"]


def ch1_hill(m, rmax, mbar, h):
    """Saturating recovery technology.  Ch1 App. Eq. `rho_hill`:
    `rho(m) = rmax * m^h / (mbar^h + m^h)`.  `h <= 1` is Michaelis-Menten /
    Holling type II, concave throughout; `h > 1` is a Hill sigmoid with
    `rho'(0) = 0`, which is the case carrying the Skiba threshold, the
    circularity trap and the possible Hopf bifurcation."""
    m = np.asarray(m, float)
    return rmax * m ** h / (mbar ** h + m ** h)


def ch1_inflection(mbar, h):
    """`m_infl = mbar ((h-1)/(h+1))^(1/h)`.  Ch1 App. Eq. `rho_inflection`.

    Zero for `h <= 1` (concave everywhere).  Above the inflection the
    technology is concave and an interior stationary point "behaves like the
    maintained model with the corresponding fixed recovery rate"; below it,
    concavity of the maximised Hamiltonian fails and the trap geometry lives.
    Locating the observed cycle relative to this scale — not the value of `h`
    alone — is what decides which of the appendix's two propositions applies.
    """
    return mbar * ((h - 1.0) / (h + 1.0)) ** (1.0 / h) if h > 1.0 else 0.0


# Ch1 App. Sec. `rho_extension`, the copper numerical illustration: an interior
# circular configuration at rho = 0.61 against a ceiling rho_max = 0.9, "about
# two thirds of the ceiling and strikingly close to the recovery rates the
# chapter calibrates for copper".  Quoted so the zinc analogue can be placed
# against it.
CH1_COPPER_RHO_EXAMPLE = dict(rho_max=0.9, h=2.0, vartheta_R=0.6, mbar=0.08,
                              rho_star=0.61, m_star=0.116, m_infl=0.046,
                              ratio_to_ceiling=0.61 / 0.9)


# ===========================================================================
# 2.  Pre-registered hypotheses
# ===========================================================================
# Fixed in source before any WP-9 number was computed.  `predicted` is what
# Chapter 1 says; `alternative` is the stated rival where Chapter 1 itself
# names one.  Cells Chapter 1 leaves ambiguous carry `predicted = None` and
# act as controls: a procedure that "confirms" a sign where Chapter 1 makes no
# prediction is measuring its own priors.

HYPOTHESES = [
    dict(
        id="H1",
        name="Transfer coefficients are technology, not economics",
        ch1_ref="Sec. `optimal_intensity_growth`, scope paragraph: "
                "\"the recovery and mass-balance parameters (rho, gamma, "
                "sigma, kappa, q) are treated as technology rather than as "
                "instruments\"; constant along the path in Prop. "
                "`oc_intensity_growth`",
        predicted="d coef / d driver = 0 for every macro driver",
        statistic="share of (driver x learned coefficient) cells whose "
                  "Hodges-Lehmann interval over 35 seeds excludes zero",
        ch1_value=0.05,
        alternative="Ch4: the coefficients are cycle- and condition-dependent, "
                    "so the share is far above the nominal 5%",
    ),
    dict(
        id="H2",
        name="Growth dilutes the stock-based recovery loop",
        ch1_ref="Eq. `ext_chi_factorised` through n+lambda, n+gamma; "
                "balanced-growth form Eq. `ext_dgk_dz` with chi'(vartheta g) < 0",
        predicted="d chi / d g < 0",
        statistic="d chi / d (driver growth rate), per +1 pp/yr, for the "
                  "activity drivers and Population",
        ch1_value=None,          # filled from the calibration at run time
        alternative="none stated by Ch1; a positive response would contradict "
                    "the dilution mechanism outright",
    ),
    dict(
        id="H3",
        name="The recovery loop collapses into one scalar multiplier",
        ch1_ref="Prop. `ext_level_not_growth`; Eq. `ext_master_elasticities`",
        predicted="dln s / d driver = [dln s*/dchi] x d chi / d driver, "
                  "across drivers, through the origin, R^2 = 1",
        statistic="OLS through the origin of dln(throughput)/ddriver on "
                  "dchi/ddriver over the 13 drivers, per seed: slope and R^2",
        ch1_value=None,          # the multiplier, filled at run time
        alternative="a multi-dimensional composition response, R^2 well below 1",
    ),
    dict(
        id="H4",
        name="In-use and discard stocks respond with opposite sign",
        ch1_ref="Sec. `Comparative dynamics`: dln u*/drho = dln s*/drho > 0 "
                "since dln B_U/drho = 0, while dln B_D/drho = -1/(1-rho)",
        predicted="sign(dln u / d driver) = -sign(dln d / d driver) for "
                  "drivers that move circularity",
        statistic="Spearman correlation across drivers between "
                  "dln(in-use)/ddriver and dln(discard)/ddriver",
        ch1_value=None,
        alternative="none; Ch1 is determinate at the zinc calibration",
    ),
    dict(
        id="H5",
        name="General-equilibrium amplification is negligible for one metal",
        ch1_ref="Eq. `ext_dlns_dchi_simplified` with nu ~ 0.01 "
                "(Sec. `capital_material_extension`, magnitudes paragraph)",
        predicted="(1-alpha)/(1-alpha-vartheta*nu) is within a few per cent "
                  "of 1, so the throughput response to circularity is almost "
                  "entirely the mass-balance multiplier 1/(1-chi)",
        statistic="the amplification factor over the (alpha, nu, vartheta) grid",
        ch1_value=None,
        alternative="a large nu, which Ch1 argues is not the case for an "
                    "individual metal",
    ),
    dict(
        id="H6",
        name="Energy rations throughput proportionally, so it has no "
             "composition effect",
        ch1_ref="Eq. `Phi_energy`, Phi = min{1, Ebar/Omega}, applied to "
                "throughput as a whole; the footnote there names the "
                "cost-minimising alternative that would reallocate towards "
                "the low-energy route",
        predicted="d chi / d (energy index) = 0",
        statistic="the Energy index cell of d chi / d driver, level and growth",
        ch1_value=0.0,
        alternative="Ch1's own footnoted alternative — a planner facing energy "
                    "costs allocates to the lowest-energy option first, giving "
                    "d chi / d energy > 0",
    ),
    dict(
        id="H7",
        name="Metal intensity of use is constant on the sustainable path",
        ch1_ref="Prop. `oc_intensity_growth`, Eq. `oc_IU_def`: u, s, d and y "
                "all grow at g_I, so I_U = u/y is constant",
        predicted="elasticity of the in-use stock to GDP = 1",
        statistic="OLS slope of ln(in-use stock) on ln(GDP), 1980-2019, per seed",
        ch1_value=1.0,
        alternative="Ch1's own trichotomy: < 1 is relative dematerialisation "
                    "(vartheta < 1, Ch1 Sec. `Levels, growth and material "
                    "reproducibility`), 0 is absolute dematerialisation, > 1 is "
                    "an intensifying cycle",
    ),
    dict(
        id="H8",
        name="The cycle is on a balanced growth path",
        ch1_ref="Sec. `Balanced growth`: g_s = g_u = g_d = vartheta*g_k, so "
                "the stock-flow ratios B_U = u/s and B_D = d/s are constant; "
                "App. Sec. `j6_numerical` uses exactly these ratios as its "
                "own empirical cross-check for copper",
        predicted="a single elasticity to GDP shared by throughput, the "
                  "in-use stock and the discard stock; trendless u/s and d/s",
        statistic="the three log-log elasticities to GDP and their pairwise "
                  "differences; the OLS trend in ln(u/s) and ln(d/s)",
        ch1_value=None,
        alternative="a transitional path, on which the ratios drift",
    ),
    dict(
        id="H9",
        name="Substitutability between virgin and recycled material",
        ch1_ref="App. Sec. `pi_extension`, Eqs. `pi_x_def`, `pi_Xcal_def`, "
                "Prop. `pi_neutrality`",
        predicted="pi is a valuation parameter only — the tonnage balance and "
                  "the stock equations are untouched — so it must be "
                  "unidentifiable from a mass-flow model; its only route into "
                  "an observable is the general-equilibrium term H5 measures",
        statistic="the span of the H3 slope prediction over pi in [0,1], "
                  "against the measured slope's own interval; and the realised "
                  "circularity multiplier X(chi, pi) evaluated at the measured "
                  "chi",
        ch1_value=None,
        alternative="none; a physical model that *could* identify pi would "
                    "contradict the extension's own construction",
    ),
    dict(
        id="H10",
        name="Recovery capacity: which branch of the Hill technology",
        ch1_ref="App. Sec. `rho_extension`, Eqs. `rho_hill`, `rho_inflection`; "
                "Prop. 1 (h <= 1, no new phenomena) versus Prop. 2 (h > 1, "
                "Skiba threshold, circularity trap, possible Hopf)",
        predicted="the two propositions are mutually exclusive and the data "
                  "must select one: concave saturation, or an "
                  "increasing-returns branch below the inflection scale",
        statistic="the sign of the quadratic term of rho on the log "
                  "collectable flow; the fitted Hill exponent h; and the "
                  "position of every observed year relative to m_infl",
        ch1_value=None,
        alternative="both are Ch1's own; the test is which one applies to "
                    "global zinc over 1981-2019",
    ),
    dict(
        id="H11",
        name="The short-run material-balance derivative of throughput",
        ch1_ref="App. Eq. `rho_s_rho`: s_rho = (lambda*u + sigma*s)/(1-rho*sigma)",
        predicted="dln s/drho equals the closed form evaluated on the UDE's "
                  "own measured (lambda, sigma, rho, u, s) — no free parameters",
        statistic="origin regression of dln s/dtheta on drho/dtheta across all "
                  "26 perturbation directions, per seed",
        ch1_value=None,
        alternative="the long-run elasticity from Ch1 Eq. "
                    "`ext_master_elasticities`; a 40-year transitional sample "
                    "should sit between the two and nearer the short-run one",
    ),
    dict(
        id="C1",
        name="CONTROL — fabrication loss is sign-ambiguous in Ch1",
        ch1_ref="Eq. `ext_sigma_ambiguity`: sigma enters both the service "
                "factor and the scrap loop, \"the net level effect depends on "
                "whether the recovery loop is strong enough\"",
        predicted=None,
        statistic="reported alongside H1; sigma maps to frac_fu_loss and "
                  "frac_eu_loss, both pinned in anchor_v4",
        ch1_value=None,
        alternative=None,
    ),
    dict(
        id="C2",
        name="CONTROL — urban mining is worthless without dilution",
        ch1_ref="Eq. `ext_chi_kappa_gamma`: chi_gamma = n/(n+gamma) * B_D "
                "vanishes as n -> 0",
        predicted=None,
        statistic="not testable: gamma = 0 in Ch1's zinc baseline and the UDE "
                  "carries no landfill stock",
        ch1_value=None,
        alternative=None,
    ),
]

# Chapter 1's structural parameters, and where each lives in the UDE.
# `pinned` slots are interpolations of data under `anchor_v4`, so an
# invariance test on them measures the pinning, not the model.
CH1_TO_UDE = {
    "lam": dict(ch1="lambda — end-of-life depreciation rate",
                ude_flows=("end_of_life",), ude_stock="In-Use",
                coefficients=("f_cohort_10yr", "f_cohort_20yr", "f_cohort_44yr"),
                learned=True),
    "sig": dict(ch1="sigma — fabrication loss share",
                ude_flows=("first_use_new_scrap", "first_use_losses",
                           "end_use_new_scrap", "end_use_losses"),
                ude_stock=None,
                coefficients=("frac_fu_new", "frac_fu_loss",
                              "frac_eu_new", "frac_eu_loss"),
                learned=None),          # mixed: new learned, loss pinned
    "kap": dict(ch1="kappa — durable-use share",
                ude_flows=("inuse_inflow", "total_products_into_use"),
                ude_stock=None, coefficients=("tau_diss",), learned=False),
    "rho": dict(ch1="rho — conventional recycling coefficient",
                ude_flows=("waelz_recycling", "direct_reuse_recycling",
                           "end_of_life", "first_use_new_scrap",
                           "end_use_new_scrap"),
                ude_stock=None,
                coefficients=("alpha_win", "alpha_dr", "tau_olds",
                              "tau_waelz"),
                learned=True),
    "gam": dict(ch1="gamma — recovery from the discard stock",
                ude_flows=(), ude_stock=None, coefficients=(), learned=None),
    "q": dict(ch1="q — recoverable share of the unrecycled residual",
              ude_flows=(), ude_stock=None, coefficients=(), learned=None),
}

# The macro drivers Ch1's `g` (growth of aggregate throughput) maps onto.  The
# activity block is what Ch1 means by the scale of the economy; Population is
# Ch1's `n` directly.  Fixed here, not chosen after looking at the results.
ACTIVITY_DRIVERS = (
    "GDP",
    "US Industrial Production Total Index",
    "EU28 Industrial Production Total Index",
    "China All Industry Value Added (Constant 2015 USD)",
    "China Total Manufacturing Output",
)
POPULATION_DRIVER = "Population"
ENERGY_DRIVER = "Energy index"
GDP_DRIVER = "GDP"


# ===========================================================================
# 3.  Chapter 1's parameters, measured on the UDE
# ===========================================================================
# Flow names in the core's own order (v5.FLOW_NAMES).  Everything below
# resolves by name, never by position.

def _fidx(name):
    import zinc_colloc_v5 as v5
    return v5.FLOW_NAMES.index(name)


# Ch1 Table `tab:calibration` reference stocks, kg/cap: d/u = 24.36/32.1.
# The UDE's window opens in 1980 with an already-accumulated discard stock
# that no flow in the sample can reconstruct, so the running integral is
# anchored on Chapter 1's own reference ratio applied to the 1980 in-use
# stock.  Every H4 statistic that is a rank correlation is invariant to this
# anchor (a positive constant added to `d` cannot change the sign of
# `dln d/dtheta`); the one statistic that is not is flagged where it is
# reported.
DISCARD_ANCHOR_RATIO = (CH1_ZINC_REFERENCE["d_ref"]
                        / CH1_ZINC_REFERENCE["u_ref"])


def ch1_params_from_traj(S, F, discard_anchor_ratio=DISCARD_ANCHOR_RATIO):
    """Chapter 1's `(lambda, sigma, kappa, rho, chi)` plus throughput, the
    in-use stock and the reconstructed discard stock, from one trajectory.

    `S` is `(T, 4)` point-in-time stocks on integer years; `F` is `(T-1, 19)`
    period integrals over `(y_i, y_{i+1}]` (spec Sec. 1, observation
    operators).  Stocks are taken at the interval midpoint so that a ratio of
    a flow integral to a stock is the quantity Chapter 1 calibrated
    (Ch1 Sec. `calib`: "the ratio between the end-of-life flow and in-use
    stock").

    Returns arrays of length `T-1`, aligned to `years[1:]`.
    """
    f = lambda n: F[:, _fidx(n)]
    inuse_mid = 0.5 * (S[:-1, 2] + S[1:, 2])

    eol = f("end_of_life")
    new_scrap = f("first_use_new_scrap") + f("end_use_new_scrap")
    manu_loss = f("first_use_losses") + f("end_use_losses")
    recycled = f("waelz_recycling") + f("direct_reuse_recycling")
    throughput = f("refined_consumption") + f("direct_reuse_recycling")

    # Ch1 built d_Zn as the cumulative unrecovered end-of-life flow; the same
    # construction here, as a running integral of everything that leaves the
    # cycle without re-entering it, started from the pre-1980 stock the sample
    # cannot reconstruct (see `DISCARD_ANCHOR_RATIO`).
    unrecovered = (f("end_of_life_losses") + manu_loss + f("dissipative_use")
                   + f("refinery_losses") + f("waelz_losses"))

    return dict(
        lam=eol / inuse_mid,
        sig=(new_scrap + manu_loss) / throughput,
        kap=f("inuse_inflow") / f("total_products_into_use"),
        rho=recycled / (eol + new_scrap),
        chi=recycled / throughput,
        throughput=throughput,
        inuse=inuse_mid,
        discard=float(discard_anchor_ratio) * S[0, 2] + np.cumsum(unrecovered),
        recycled=recycled,
        eol=eol,
        new_scrap=new_scrap,
        unrecovered=unrecovered,
    )


def ch1_params_jacobian(S, F, dS, dF):
    """Exact derivatives of `ch1_params_from_traj`'s outputs.

    `dS` is `(T, 4, P)` and `dF` is `(T-1, 19, P)` for `P` perturbation
    directions; the quotient rule is applied analytically, so no second finite
    difference enters.  Returns `{name: (T-1, P)}` for the five Chapter 1
    parameters plus `throughput`, `inuse` and `discard`.
    """
    f = lambda n: F[:, _fidx(n)]
    df = lambda n: dF[:, _fidx(n), :]
    inuse_mid = 0.5 * (S[:-1, 2] + S[1:, 2])
    d_inuse = 0.5 * (dS[:-1, 2, :] + dS[1:, 2, :])

    eol, d_eol = f("end_of_life"), df("end_of_life")
    new_scrap = f("first_use_new_scrap") + f("end_use_new_scrap")
    d_new = df("first_use_new_scrap") + df("end_use_new_scrap")
    manu_loss = f("first_use_losses") + f("end_use_losses")
    d_manu = df("first_use_losses") + df("end_use_losses")
    recycled = f("waelz_recycling") + f("direct_reuse_recycling")
    d_rec = df("waelz_recycling") + df("direct_reuse_recycling")
    thr = f("refined_consumption") + f("direct_reuse_recycling")
    d_thr = df("refined_consumption") + df("direct_reuse_recycling")
    unrec = (f("end_of_life_losses") + manu_loss + f("dissipative_use")
             + f("refinery_losses") + f("waelz_losses"))
    d_unrec = (df("end_of_life_losses") + d_manu + df("dissipative_use")
               + df("refinery_losses") + df("waelz_losses"))

    def quot(num, den, dnum, dden):
        return (dnum * den[:, None] - num[:, None] * dden) / (den ** 2)[:, None]

    return dict(
        lam=quot(eol, inuse_mid, d_eol, d_inuse),
        sig=quot(new_scrap + manu_loss, thr, d_new + d_manu, d_thr),
        kap=quot(f("inuse_inflow"), f("total_products_into_use"),
                 df("inuse_inflow"), df("total_products_into_use")),
        rho=quot(recycled, eol + new_scrap, d_rec, d_eol + d_new),
        chi=quot(recycled, thr, d_rec, d_thr),
        throughput=d_thr,
        inuse=d_inuse,
        discard=np.cumsum(d_unrec, axis=0),
    )


def chi_decomposition(par, jac):
    """Split `dchi/dtheta` into the channel Chapter 1 allows and the one it
    rules out.

    Chapter 1 writes `chi = rho * (sigma + lambda * u/s)`
    (Ch1 Eq. `ext_chi_factorised` with `gamma = 0`), so

        dchi = (sigma + lambda*u/s) drho          <- coefficient adaptation
             + rho dsigma                          <- coefficient adaptation
             + rho (u/s) dlambda                   <- coefficient adaptation
             + rho*lambda d(u/s).                  <- dilution: Ch1's channel

    Chapter 1 holds `(rho, sigma, lambda)` fixed as technology, so only the
    last term is admissible under its maintained closure.  The share of the
    total carried by the other three is the size of the Chapter 4 correction.
    """
    rho, sig, lam = par["rho"], par["sig"], par["lam"]
    ratio = par["inuse"] / par["throughput"]
    d_ratio = ((jac["inuse"] * par["throughput"][:, None]
                - par["inuse"][:, None] * jac["throughput"])
               / (par["throughput"] ** 2)[:, None])
    terms = dict(
        drho=(sig + lam * ratio)[:, None] * jac["rho"],
        dsigma=rho[:, None] * jac["sig"],
        dlambda=(rho * ratio)[:, None] * jac["lam"],
        dilution=(rho * lam)[:, None] * d_ratio,
    )
    terms["ch1_formula_total"] = sum(terms.values())
    terms["chi_formula"] = rho * (sig + lam * ratio)
    return terms


# ===========================================================================
# 4.  Forward-mode sensitivities: level shifts and growth-rate rotations
# ===========================================================================

def sensitivity(fit, params, ctx, *, drivers=None, growth_unit=0.01):
    """`(S, F)` and their Jacobian with respect to 13 level shifts and 13
    growth-rate rotations, in one `jax.jacfwd`.

    Level shift `theta_j`: driver `j`'s whole raw history moves up by
    `theta_j` within-sample SDs, additively — WP-8c's convention exactly, so
    the level block is directly comparable with
    `analysis/wp8c_forward_sens.npz`.

    Growth rotation `g_j`: driver `j`'s raw history is multiplied by
    `exp(g_j * (t - t_mid) / growth_unit)` with `t_mid` the sample midpoint,
    so `g_j = 1` is `+1 pp/yr` on that driver's growth rate and leaves the
    mid-sample level unchanged.  Chapter 1's `n` and `g` are growth rates and
    a level shift is not one; this is the perturbation H2 needs.

    Returns `S (T,4)`, `F (T-1,19)`, `dS (T,4,26)`, `dF (T-1,19,26)`, the
    perturbation labels, and the same-trajectory gaps WP-8c reports.
    """
    import jax
    import jax.numpy as jnp

    cfg, t_full, X_full, cols, cut, _wd = ctx
    if drivers is None:
        drivers = list(cols)
    jidx = np.array([list(cols).index(dn) for dn in drivers], int)
    sd = np.array([np.std(X_full[:, j]) for j in jidx], float)
    years = np.asarray(fit.data_all["years"], float).ravel()
    t_mid = 0.5 * (float(t_full[0]) + float(t_full[-1]))

    fwd = xlab.make_forward_integrator(fit)
    data0 = dict(fit.data_all)

    S_ref, F_ref, _ = fit.integrate_aug(params, data0, data0["stocks_obs"][0],
                                        data0["years"])
    S_fwd, F_fwd, _ = fwd(params, data0, data0["stocks_obs"][0], data0["years"])
    gap_S = float(np.max(np.abs(np.asarray(S_fwd) - np.asarray(S_ref))
                         / np.maximum(np.abs(np.asarray(S_ref)), 1.0)))
    gap_F = float(np.max(np.abs(np.asarray(F_fwd) - np.asarray(F_ref))
                         / np.maximum(np.abs(np.asarray(F_ref)), 1.0)))

    X_j = jnp.asarray(X_full)
    years_j = jnp.asarray(years)
    orders = tuple(sorted(set(int(o) for o in cfg.get("exog_feature_orders", (0,)))))
    do_log1p = bool(cfg.get("exog_log1p", True))
    diff_pad = cfg.get("exog_diff_pad", "edge")
    tgt = np.clip(np.searchsorted(t_full, years), 0, max(t_full.size - 1, 0))
    tgt_j = jnp.asarray(tgt)

    def _preprocess(X):
        """`preprocess_exog` in JAX for the `anchor_v4` settings; identical to
        `zinc_xai_lab.forward_sensitivity`'s reproduction, and verified against
        the fit's own feature matrix below."""
        Xs = jnp.log1p(jnp.maximum(X, 0.0)) if do_log1p else X
        feats = []
        for o in orders:
            if o == 0:
                Xo = Xs
            else:
                D = jnp.diff(Xs, n=o, axis=0)
                pad = (jnp.repeat(D[:1], o, axis=0) if diff_pad == "edge"
                       else jnp.zeros((o, Xs.shape[1]), Xs.dtype))
                Xo = jnp.concatenate([pad, D], axis=0)
            feats.append(Xo[tgt_j])
        return jnp.concatenate(feats, axis=1) if len(feats) > 1 else feats[0]

    gap_exog = float(np.max(np.abs(np.asarray(_preprocess(X_j))
                                   - np.asarray(fit.data_all["exog_values"]))))

    jidx_j, sd_j = jnp.asarray(jidx), jnp.asarray(sd)
    ramp = jnp.asarray((np.asarray(t_full, float) - t_mid) * float(growth_unit))
    nD = len(jidx)

    def traj(theta):
        lvl, grw = theta[:nD], theta[nD:]
        Xc = X_j[:, jidx_j]
        Xc = Xc * jnp.exp(ramp[:, None] * grw[None, :])
        Xc = Xc + (lvl * sd_j)[None, :]
        X = X_j.at[:, jidx_j].set(Xc)
        data = dict(data0)
        data["exog_values"] = _preprocess(X)
        S, F, _ = fwd(params, data, data0["stocks_obs"][0], years_j)
        return jnp.concatenate([S.reshape(-1), F.reshape(-1)])

    theta0 = jnp.zeros(2 * nD)
    J = np.asarray(jax.jacfwd(traj)(theta0))
    nS = years.size * 4
    dS = J[:nS].reshape(years.size, 4, 2 * nD)
    dF = J[nS:].reshape(years.size - 1, -1, 2 * nD)
    labels = ([f"level::{d}" for d in drivers] + [f"growth::{d}" for d in drivers])
    return dict(years=years, drivers=list(drivers), labels=labels,
                n_drivers=nD, dS=dS, dF=dF, sd=sd, growth_unit=float(growth_unit),
                t_mid=t_mid, gap_S=gap_S, gap_F=gap_F, gap_exog=gap_exog,
                S=np.asarray(S_ref), F=np.asarray(F_ref))


# ===========================================================================
# 5.  Shared helpers
# ===========================================================================

def build_context(cfg=None, weights_dir=WEIGHTS_DIR_DEFAULT):
    """`(fit, ctx)` — delegates to `zinc_cf_lab.build_context`, which verifies
    the `preprocess_exog` rebuild against the fit's own feature matrix."""
    return cflab.build_context(cfg, weights_dir)


def load_params(seed, weights_dir=WEIGHTS_DIR_DEFAULT):
    return cflab.load_params(os.path.join(weights_dir, f"A_seed{seed}.npz"))


def seed_list(weights_dir=WEIGHTS_DIR_DEFAULT):
    return sorted(int(f[len("A_seed"):-len(".npz")])
                  for f in os.listdir(weights_dir)
                  if f.startswith("A_seed") and f.endswith(".npz"))


def driver_series(ctx, name):
    """One driver's raw history, resampled to the model's year nodes."""
    _cfg, t_full, X_full, cols, _cut, _wd = ctx
    j = list(cols).index(name)
    return np.asarray(t_full, float), np.asarray(X_full[:, j], float)


def ols_through_origin(x, y):
    """Slope and `R^2` of `y = b x`, the form Ch1 Eq. `ext_master_elasticities`
    predicts (no intercept: a driver that does not move `chi` must not move
    `s`)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3 or np.all(x == 0):
        return np.nan, np.nan
    b = float(np.sum(x * y) / np.sum(x * x))
    ss_res = float(np.sum((y - b * x) ** 2))
    ss_tot = float(np.sum(y ** 2))
    return b, (1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan)


def _rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024.0 ** 2) if sys.platform == "darwin" else r / 1024.0


# ===========================================================================
# 6.  --check
# ===========================================================================

def check(verbose=True, weights_dir=WEIGHTS_DIR_DEFAULT):
    """Every precondition WP-9 needs, computed rather than asserted:

      * the driver universe and `input_dim`, from `zinc_alpha_lab`;
      * Chapter 1's own calibration reproduced from its equations, against
        the numbers printed in its Tables `tab:calibration` and
        `tab:steady_state_stability`;
      * the closed-form `chi_z` against a central difference;
      * the `ch1_params_jacobian` quotient rule against a central difference
        of the solver, on seed 0;
      * the `ForwardMode` trajectory against the fit's own.
    """
    import zinc_colloc_v5 as v5

    info = alab.check(verbose=False)
    install(v5)
    cfg = load_anchor_config()
    fit, ctx = build_context(cfg, weights_dir)
    lay = xlab.layout_index(fit, cfg)
    seeds = seed_list(weights_dir)

    # --- Chapter 1 reproduces itself -------------------------------------
    chi0 = ch1_chi(CH1_ZINC)
    chi1 = ch1_chi(CH1_ZINC_GAMMA_SENS)
    st0 = ch1_state(CH1_ZINC)
    st1 = ch1_state(CH1_ZINC_GAMMA_SENS, s_star=CH1_ZINC_STEADY_GAMMA_SENS["s_star"])
    ch1_repro = {
        "chi_star_baseline": (chi0, CH1_ZINC_STEADY["chi_star"]),
        "chi_star_gamma_sens": (chi1, CH1_ZINC_STEADY_GAMMA_SENS["chi_star"]),
        "u_star_baseline": (st0["u"], CH1_ZINC_STEADY["u_star"]),
        "d_star_baseline": (st0["d"], CH1_ZINC_STEADY["d_star"]),
        "u_star_gamma_sens": (st1["u"], CH1_ZINC_STEADY_GAMMA_SENS["u_star"]),
        "d_star_gamma_sens": (st1["d"], CH1_ZINC_STEADY_GAMMA_SENS["d_star"]),
    }
    repro_gap = max(abs(a - b) for a, b in ch1_repro.values())
    dchk = ch1_chi_derivs(CH1_ZINC)

    # --- the Jacobian chain rule, against the solver itself ---------------
    params = load_params(seeds[0], weights_dir)
    sens = sensitivity(fit, params, ctx)
    par = ch1_params_from_traj(sens["S"], sens["F"])
    jac = ch1_params_jacobian(sens["S"], sens["F"], sens["dS"], sens["dF"])

    # Central difference on one level and one growth direction.  The step is
    # swept rather than fixed: below h ~ 1e-2 the difference is dominated by
    # the core's own rtol = 1e-5 solve (COMPUTE_STATUS flag 3), above it by
    # genuine curvature, so the agreement is read off the minimum.
    fd_probe = []
    for pi, steps in ((0, (0.01, 0.05, 0.20)),
                      (sens["n_drivers"] + 2, (0.05, 0.20, 0.50))):
        for h in steps:
            vals = {}
            for s in (+1, -1):
                th = np.zeros(2 * sens["n_drivers"])
                th[pi] = s * h
                S_p, F_p = _perturbed_traj(fit, params, ctx, th,
                                           sens["growth_unit"])
                vals[s] = ch1_params_from_traj(S_p, F_p)
            worst = 0.0
            for key in ("chi", "rho", "lam"):
                fd = (vals[+1][key] - vals[-1][key]) / (2 * h)
                an = jac[key][:, pi]
                scale = max(float(np.max(np.abs(an))), 1e-12)
                worst = max(worst, float(np.max(np.abs(fd - an)) / scale))
            fd_probe.append((sens["labels"][pi], h, worst))
    fd_gap = min(g for _lab, _h, g in fd_probe)

    info.update(
        drivers=lay["drivers"], n_universe=lay["n_universe"],
        orders=lay["orders"], input_dim=lay["input_dim"],
        n_weight_dumps=len(seeds),
        ch1_chi_star=float(chi0), ch1_repro_max_gap=float(repro_gap),
        ch1_chi_deriv_closed_vs_numeric=dchk["max_closed_vs_numeric"],
        ch1_dchi_dg=float(dchk["numeric"]["g"]),
        fwd_gap_S=sens["gap_S"], fwd_gap_F=sens["gap_F"],
        fwd_gap_exog=sens["gap_exog"],
        jacobian_vs_finite_difference=float(fd_gap),
        hypotheses=[h["id"] for h in HYPOTHESES],
        patches=list(PATCHES),
    )

    if verbose:
        print("=" * 76)
        print("zinc_ch1_lab --check   (WP-9 Chapter 1 hypothesis tests)")
        print("=" * 76)
        for label, (got, ok) in info["digests"].items():
            print(f"  {label:20s} md5 {got}  {'OK' if ok else 'MISMATCH'}")
        print(f"  patches applied      : {PATCHES or 'none (WP-9 needs none)'}")
        print(f"  weights              : {len(seeds)} dumps in {weights_dir}")
        print(f"\n  resolved drivers (canonical order, {lay['n_universe']}):")
        for i, c in enumerate(lay["drivers"]):
            tag = ""
            if c in ACTIVITY_DRIVERS:
                tag = "   <- Ch1 activity / g"
            elif c == POPULATION_DRIVER:
                tag = "   <- Ch1 n"
            elif c == ENERGY_DRIVER:
                tag = "   <- Ch1 H6"
            print(f"    [{i:2d}] {c}{tag}")
        print(f"\n  exog_feature_orders  : {lay['orders']}")
        print(f"  input_dim            : {lay['input_dim']}")
        if lay["input_dim"] != 23:
            print(f"  !! input_dim is {lay['input_dim']}, not the 23 asserted by "
                  f"CLAUDE.md rule 2 (SCHEMA Sec. 9 flag 1, still unresolved).")
            print( "     WP-9 never indexes the input vector positionally: every "
                   "driver is resolved by name")
            print( "     and every sensitivity is taken w.r.t. the raw driver, so "
                   "the discrepancy does not")
            print( "     affect these results.")
        print("\n  Chapter 1 reproduces its own tables:")
        for k, (got, want) in ch1_repro.items():
            print(f"    {k:22s} computed {got:9.4f}   Ch1 table {want:9.4f}   "
                  f"gap {abs(got-want):.2e}")
        print(f"    max gap {repro_gap:.2e}")
        print(f"  closed-form chi_z vs central difference : "
              f"{dchk['max_closed_vs_numeric']:.2e}")
        print(f"  Ch1 dchi/dg at the zinc calibration     : "
              f"{dchk['numeric']['g']:+.4f} per unit growth rate")
        print(f"                                          "
              f"({dchk['numeric']['g']*0.01:+.5f} per +1 pp/yr, "
              f"{100*dchk['numeric']['g']*0.01/chi0:+.2f}% of chi*)")
        print("\n  forward-mode integrator vs the fit's own (seed "
              f"{seeds[0]}):")
        print(f"    max rel gap stocks {sens['gap_S']:.3e}   "
              f"flows {sens['gap_F']:.3e}   exog rebuild {sens['gap_exog']:.3e}")
        print("  Ch1-parameter chain rule vs central difference of the solver")
        print("  (worst of chi / rho / lambda; step swept — small steps are "
              "solver-noise limited):")
        for lab, h, g in fd_probe:
            print(f"    {lab:52s} h={h:<5} rel gap {g:.2e}")
        print(f"    best {fd_gap:.2e}")
        print("\n  UDE-measured Chapter 1 parameters, seed "
              f"{seeds[0]} (mean over 1981-2019) vs Ch1 Table tab:calibration:")
        for k, want in (("lam", CH1_ZINC["lam"]), ("sig", CH1_ZINC["sig"]),
                        ("kap", CH1_ZINC["kap"]), ("rho", CH1_ZINC["rho"]),
                        ("chi", CH1_ZINC_STEADY["chi_star"])):
            print(f"    {k:4s} UDE {np.mean(par[k]):8.4f}   Ch1 {want:8.4f}")
        print(f"\n  pre-registered hypotheses: "
              f"{', '.join(h['id'] for h in HYPOTHESES)}")
        print(f"  RSS {_rss_mb():.0f} MB")
        print("=" * 76)
    return info


def _perturbed_traj(fit, params, ctx, theta, growth_unit):
    """One perturbed trajectory, outside AD — used only by `check()`'s finite
    difference, so the chain rule is validated against the solver rather than
    against itself."""
    import numpy as _np
    import zinc_colloc_v5 as v5

    cfg, t_full, X_full, cols, cut, _wd = ctx
    nD = len(cols)
    lvl, grw = _np.asarray(theta[:nD]), _np.asarray(theta[nD:])
    sd = _np.array([_np.std(X_full[:, j]) for j in range(nD)], float)
    t_mid = 0.5 * (float(t_full[0]) + float(t_full[-1]))
    ramp = (_np.asarray(t_full, float) - t_mid) * float(growth_unit)
    X = _np.asarray(X_full, float).copy()
    X = X * _np.exp(ramp[:, None] * grw[None, :]) + (lvl * sd)[None, :]

    years = _np.asarray(fit.data_all["years"], float).ravel()
    data = dict(fit.data_all)
    data["exog_values"] = cflab.rebuild_exog(cfg, years, t_full, X, cut)
    S, F, _ = fit.integrate_aug(params, data, data["stocks_obs"][0],
                                data["years"])
    return _np.asarray(S), _np.asarray(F)


def main(argv=None):
    ap = argparse.ArgumentParser(description="WP-9 lab module")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--weights", default=WEIGHTS_DIR_DEFAULT)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    if a.check:
        info = check(weights_dir=a.weights)
        if a.json:
            with open(a.json, "w") as fh:
                json.dump({k: v for k, v in info.items()
                           if not isinstance(v, dict) or k != "digests"},
                          fh, indent=2, default=str)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
