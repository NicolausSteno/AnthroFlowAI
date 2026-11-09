#!/usr/bin/env python3
"""
D2 -- closing the loop versus settling down.  Thesis Discussion, sec:thdisc_stability.
The collection--stability trade-off of the zinc chapter read against the macro
chapter's viability threshold Gamma < Gamma_M, with the copper chapter's matrix
alongside.  Analytic; no refits.

Inputs
  * the 35 anchor_v4 UDE trajectories cached by WP-9 (analysis/wp9/ch1_seed*.npz),
    mapped to the macro chapter's (lambda, sigma, kappa, rho, chi) by
    zinc_ch1_lab.ch1_params_from_traj -- the WP-9 mapping, unchanged;
  * the WP-2a frozen matrices A(t) (analysis/wp2a_A_of_t.npz, 35 seeds x 40 years) and
    the WP-2c exact derivatives dA/dp (analysis/wp2c_eigen_sensitivity.npz);
  * the macro chapter's material block (Macro-Materials/mm_core.py: Gamma, Gamma_M,
    jacobian, chi_star; ZN_GAMMA, ZN_BASE, CU_GAMMA, CU_BASE);
  * the copper chapter's basic 8x8 matrix at the 1994 coefficients (builder copied from
    D12, d12_copper_ratios.py, which validates it against tab:eigen).

What is computed
  A. Macro chapter at its own calibrations: headroom 1 - Gamma/Gamma_M (validation:
     9.8 / 14.1 / 21.0 / 28.2 per cent), the equivalence Gamma < Gamma_M <=> psi + chi* < 1,
     the derivative of the headroom and of the dominant eigenvalue with respect to rho,
     and rho_crit.
  B. Gamma(t), Gamma_M(t) and the headroom on the MEASURED zinc coefficients, seed by seed,
     1981-2019.  psi is not measurable on the UDE (primary supply is exogenous there), so
     it is held at the macro calibration 0.613 and the distance is also reported in psi
     units, psi_crit(t) = 1 - chi*(t).  Variants: gamma = 0 at measured rho (primary,
     commensurable: the UDE has no discard-recovery flow); the calibration (gamma = 0.003,
     q = 0.300, rho offset by -0.018 as in D3); observed population trend n.  Swap
     decomposition of the 1981 -> 2019 headroom change by coefficient.
  C. The common mechanism.  The return fraction r_U: the probability that metal leaving
     the in-use stock at end of life comes back into use, computed identically in the
     three models as an absorption probability (in-use stocks made absorbing):
       zinc UDE   r_U = tau_olds * P(scrap -> use),  from A(t);
       copper     r_U = P(waste management -> in use), from the 8x8 matrix;
       macro      r_U(psi) = rho (1-sigma) kappa / (1 - psi - rho sigma)  (gamma = 0);
     and the slow in-use mode  s = -[loss + lambda_EoL (1 - r_U)]  (exact for the macro
     block at gamma = 0, where J11 is the eigenvalue; a check for the other two).
     Derivatives of s and r_U with respect to collection: UDE tau_olds (exact, via
     dA/dtau_olds); copper collection share c = (a13+a14)/(a11+a13+a14+d5) at fixed total
     exit and fixed a13:a14 split; macro rho.
  D. Frozen-equilibrium translation of tau_olds into the macro rho: d rho*/d tau_olds,
     so the macro headroom derivative can be stated per unit of UDE collection.

Run from AnthroFlowAI/final_model (needs analysis/wp9/ch1_seed*.npz from run_wp9.py and the
unmasked analysis/wp2a_A_of_t.npz and wp2c_eigen_sensitivity.npz, i.e. the observed record):
    PYTHONDONTWRITEBYTECODE=1 python discussion/d2_stability.py [--out discussion/outputs] [--mm <Macro-Materials>]
"""
import argparse, os, sys

# sibling repositories (clone Macro-Materials and Simulating-Resources next to AnthroFlowAI,
# or point the environment variables MACRO_MATERIALS_DIR / SIMULATING_RESOURCES_DIR at them)
_FM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPOS = os.path.dirname(os.path.dirname(_FM))
MM_DIR = os.environ.get('MACRO_MATERIALS_DIR', os.path.join(_REPOS, 'Macro-Materials'))
SR_DIR = os.environ.get('SIMULATING_RESOURCES_DIR', os.path.join(_REPOS, 'Simulating-Resources'))
OUT_DEFAULT = os.path.join(_FM, 'discussion', 'outputs')
from dataclasses import replace
import numpy as np, pandas as pd

sys.path.insert(0, os.getcwd())
import zinc_ch1_lab as CH1                                    # noqa: E402
from run_wp9 import hl, med_iqr, trend_slope_weights          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=OUT_DEFAULT)
ap.add_argument('--mm', default=MM_DIR)
args = ap.parse_args()
sys.path.insert(0, args.mm)
import mm_core as MM                                          # noqa: E402

OUT = args.out
os.makedirs(OUT, exist_ok=True)
val = []                                                      # (check, expected, got, ok)


def check(name, expected, got, tol):
    ok = bool(abs(float(expected) - float(got)) <= tol)
    val.append(dict(check=name, expected=expected, got=got, tol=tol, ok=ok))
    return ok


def summ(v):
    v = np.asarray(v, float); m, q1, q3 = med_iqr(v); e, lo, hi = hl(v)
    return dict(median=m, q1=q1, q3=q3, hl=e, hl_lo=lo, hl_hi=hi)


# =====================================================================================
# macro block, generic in its parameters (dict form so measured values can be plugged in)
# =====================================================================================
def mp(base, **kw):
    return replace(base, **kw)


def headroom(p):
    return 1.0 - MM.Gamma(p) / MM.Gamma_M(p)


def mu_dom(p):
    return float(np.max(np.linalg.eigvals(MM.jacobian(p)).real))


def rU_macro(p):                                             # gamma = 0 return fraction
    return p.rho * (1 - p.sigma) * p.kappa / (1 - p.psi - p.rho * p.sigma)


def d_central(f, p, key, h=1e-6):
    return (f(mp(p, **{key: getattr(p, key) + h})) - f(mp(p, **{key: getattr(p, key) - h}))) / (2 * h)


def rho_crit(p):
    from scipy.optimize import brentq
    f = lambda r: p.psi + MM.chi_star(mp(p, rho=r)) - 1.0
    lo, hi = 1e-6, min(0.999, (1 - p.psi) / p.sigma - 1e-9)
    return brentq(f, lo, hi) if f(lo) < 0 < f(hi) else np.nan


# ---------------------------------------------------------------- A. macro calibrations
rowsA = []
expect = {'Copper, gamma=0.002': 0.098, 'Copper, gamma=0': 0.141,
          'Zinc, gamma=0.003': 0.210, 'Zinc, gamma=0': 0.282}
for p in MM.CASES:
    G, GM = MM.Gamma(p), MM.Gamma_M(p)
    h = 1 - G / GM
    chs = MM.chi_star(p)
    check(f'headroom {p.name} (Ch2 fig caption)', expect[p.name], round(h, 3), 0.0005)
    check(f'Gamma<Gamma_M <=> psi+chi*<1, {p.name}', 1.0, float((G < GM) == (p.psi + chs < 1)), 0)
    # equivalence on a rho grid
    grid_ok = all(((MM.Gamma(q) < MM.Gamma_M(q)) == (q.psi + MM.chi_star(q) < 1))
                  for q in (mp(p, rho=r) for r in np.linspace(0.05, 0.95, 181))
                  if MM.Gamma_M(q) < np.inf and q.feasible())
    check(f'equivalence on rho grid, {p.name}', 1.0, float(grid_ok), 0)
    # next-generation splitting J = N - D, D = diag(n+lam, n+gam)
    J = MM.jacobian(p); D = np.diag([p.n + p.lam, p.n + p.gamma]); Nn = J + D
    Rng = float(np.max(np.abs(np.linalg.eigvals(Nn @ np.linalg.inv(D)))))
    if p.gamma == 0:
        check(f'next-generation radius = Gamma/Gamma_M at gamma=0, {p.name}', G / GM, Rng, 1e-12)
        check(f'J11 = -(n + lam (1 - r_U)) at gamma=0, {p.name}',
              -(p.n + p.lam * (1 - rU_macro(p))), J[0, 0], 1e-12)
    rowsA.append(dict(case=p.name, psi=p.psi, rho=p.rho, gamma=p.gamma, n=p.n,
                      Gamma=G, Gamma_M=GM, Gamma_over_GammaM=G / GM, headroom=h,
                      chi_star=chs, psi_plus_chi=p.psi + chs, psi_crit=1 - chs,
                      next_gen_radius=Rng, r_U_gamma0=rU_macro(p) if p.gamma == 0 else np.nan,
                      mu_dom=mu_dom(p), T_dom=-1 / mu_dom(p),
                      dheadroom_drho=d_central(headroom, p, 'rho'),
                      dmu_dom_drho=d_central(mu_dom, p, 'rho'),
                      elas_Tdom_rho=-d_central(mu_dom, p, 'rho') * p.rho / mu_dom(p),
                      rho_crit=rho_crit(p),
                      # same block with the investment channel switched off
                      headroom_psi0=headroom(mp(p, psi=0.0)),
                      mu_dom_psi0=mu_dom(mp(p, psi=0.0)),
                      dmu_dom_drho_psi0=d_central(mu_dom, mp(p, psi=0.0), 'rho'),
                      headroom_n0=headroom(mp(p, n=0.0)) if MM.Gamma_M(mp(p, n=0.0)) < np.inf else np.nan))
A_tab = pd.DataFrame(rowsA)
A_tab.to_csv(os.path.join(OUT, 'd2_macro_calibration.csv'), index=False)

# =====================================================================================
# B. Gamma(t), Gamma_M(t) on the measured zinc coefficients
# =====================================================================================
cfg = CH1.load_anchor_config()
_fit, ctx = CH1.build_context(cfg)
seeds = CH1.seed_list()
z0 = np.load(os.path.join('analysis', 'wp9', f'ch1_seed{seeds[0]}.npz'), allow_pickle=True)
years = z0['years'][1:]                                       # 1981-2019, WP-9 alignment
tp, L = CH1.driver_series(ctx, CH1.POPULATION_DRIVER)
Npop = np.interp(years, tp, L)
n_obs = float(np.sum(trend_slope_weights(years) * np.log(Npop)))
RHO_OFFSET = MM.ZN_BASE.rho - MM.ZN_GAMMA.rho                 # 0.018, D3 convention

variants = {
    'g0_n001': dict(gamma=0.0, n=0.01, offset=0.0),           # primary
    'calib_n001': dict(gamma=MM.ZN_GAMMA.gamma, n=0.01, offset=RHO_OFFSET),
    'g0_nobs': dict(gamma=0.0, n=n_obs, offset=0.0),
}

recB, recS = [], []
par_by_seed = {}
for s in seeds:
    z = np.load(os.path.join('analysis', 'wp9', f'ch1_seed{s}.npz'), allow_pickle=True)
    par = CH1.ch1_params_from_traj(z['S'], z['F'])
    par_by_seed[s] = par
    for vname, v in variants.items():
        for i, yr in enumerate(years):
            p = mp(MM.ZN_BASE, lam=float(par['lam'][i]), sigma=float(par['sig'][i]),
                   kappa=float(par['kap'][i]), rho=float(par['rho'][i]) - v['offset'],
                   gamma=v['gamma'], n=v['n'])
            G, GM = MM.Gamma(p), MM.Gamma_M(p); chs = MM.chi_star(p)
            recB.append(dict(variant=vname, seed=s, year=yr, lam=p.lam, sig=p.sigma, kap=p.kappa,
                             rho=p.rho, Gamma=G, Gamma_M=GM, headroom=1 - G / GM,
                             chi_star=chs, psi_crit=1 - chs, psi_gap=1 - chs - p.psi,
                             r_U=rU_macro(p) if p.gamma == 0 else np.nan,
                             mu_dom=mu_dom(p), dheadroom_drho=d_central(headroom, p, 'rho'),
                             dmu_dom_drho=d_central(mu_dom, p, 'rho'),
                             rho_crit=rho_crit(p)))
        # time-mean coefficients (the D3/D14 convention) and the 1981 -> 2019 swap
        pm = {k: float(np.mean(par[k])) for k in ('lam', 'sig', 'kap', 'rho')}
        mk = lambda d: mp(MM.ZN_BASE, lam=d['lam'], sigma=d['sig'], kappa=d['kap'],
                          rho=d['rho'] - v['offset'], gamma=v['gamma'], n=v['n'])
        p_mean = mk(pm)
        first = {k: float(par[k][0]) for k in pm}; last = {k: float(par[k][-1]) for k in pm}
        row = dict(variant=vname, seed=s, headroom_mean=headroom(p_mean),
                   psi_gap_mean=1 - MM.chi_star(p_mean) - p_mean.psi,
                   h1981=headroom(mk(first)), h2019=headroom(mk(last)))
        row['dh'] = row['h2019'] - row['h1981']
        for k in pm:                                          # one-at-a-time swap, averaged both ways
            fw = headroom(mk({**first, k: last[k]})) - row['h1981']
            bw = row['h2019'] - headroom(mk({**last, k: first[k]}))
            row[f'dh_{k}'] = 0.5 * (fw + bw)
        recS.append(row)

B = pd.DataFrame(recB)
S_ = pd.DataFrame(recS)
B.to_csv(os.path.join(OUT, 'd2_gamma_by_seed_year.csv'), index=False)
S_.to_csv(os.path.join(OUT, 'd2_headroom_per_seed.csv'), index=False)

byyear = []
for (vn, yr), g in B.groupby(['variant', 'year']):
    r = dict(variant=vn, year=yr)
    for k in ('rho', 'Gamma', 'Gamma_M', 'headroom', 'psi_crit', 'psi_gap', 'r_U', 'mu_dom',
              'dheadroom_drho', 'rho_crit'):
        m, q1, q3 = med_iqr(g[k]); r[k] = m; r[k + '_q1'] = q1; r[k + '_q3'] = q3
    byyear.append(r)
BY = pd.DataFrame(byyear)
BY.to_csv(os.path.join(OUT, 'd2_gamma_by_year.csv'), index=False)

# headline summary over seeds
headB = []
for vn, g in S_.groupby('variant'):
    for k in ('headroom_mean', 'psi_gap_mean', 'h1981', 'h2019', 'dh', 'dh_lam', 'dh_sig', 'dh_kap', 'dh_rho'):
        headB.append(dict(variant=vn, quantity=k, **summ(g[k])))
for vn, g in B.groupby('variant'):
    tr = g.groupby('seed').apply(lambda d: float(np.sum(trend_slope_weights(d.year.values) * d.headroom.values)),
                                 include_groups=False)
    headB.append(dict(variant=vn, quantity='headroom trend (/yr)', **summ(tr)))
    for k in ('dheadroom_drho', 'dmu_dom_drho', 'rho_crit', 'Gamma_M', 'Gamma'):
        headB.append(dict(variant=vn, quantity=f'{k} (all nodes)', **summ(g[k])))
HB = pd.DataFrame(headB)

# =====================================================================================
# C. the common mechanism: return fraction r_U and the slow in-use mode
# =====================================================================================
ZA = np.load(os.path.join('analysis', 'wp2a_A_of_t.npz'), allow_pickle=True)
ZE = np.load(os.path.join('analysis', 'wp2c_eigen_sensitivity.npz'), allow_pickle=True)
assert list(ZA['seeds']) == list(ZE['seeds']) == list(seeds)
A_all, b_all, yA = ZA['A'], ZA['b'], ZA['years']
pn = list(ZE['param_names']); j_old = pn.index('tau_olds')
dA_old = ZE['dA_dparams'][:, :, j_old]                        # (35, 40, 6, 6)
taus = ZA['taus']; t_old = taus[:, :, list(ZA['tau_binary_names']).index('tau_olds')]
mu = ZA['mu_cohorts']
USE, TR = [2, 3, 4], [0, 1, 5]


def absorb_to_use(A, start):
    """P(reach any in-use stock | start in transient state `start`), in-use made absorbing."""
    Q = A[np.ix_(TR, TR)]; R = A[np.ix_(USE, TR)]
    x = np.linalg.solve(-Q, np.eye(3)[TR.index(start)])       # expected time in each TR state
    return float(R.sum(axis=0) @ x)


def dom(A):
    w, V = np.linalg.eig(A); k = int(np.argmax(w.real)); return w[k].real, k


def ds_dp(A, dA):
    w, V = np.linalg.eig(A); k = int(np.argmax(w.real))
    wl, U = np.linalg.eig(A.T); kl = int(np.argmin(np.abs(wl - w[k])))
    v, u = V[:, k].real, U[:, kl].real
    return float(u @ dA @ v / (u @ v))


def nextgen(A):
    Dg = np.diag(-np.diag(A)); Nn = A + Dg
    # scrap self-loop: direct reuse that returns as new scrap is a transfer, not a stay
    return float(np.max(np.abs(np.linalg.eigvals(Nn @ np.linalg.inv(Dg)))))


recC = []
for si, s in enumerate(seeds):
    for ti, yr in enumerate(yA):
        A = A_all[si, ti]; dA = dA_old[si, ti]
        p_su = absorb_to_use(A, 5)
        rU = t_old[si, ti] * p_su
        # sensitivity of r_U to tau_olds: P(scrap->use) does not contain tau_olds
        h = 1e-6
        rU_h = (t_old[si, ti] + h) * absorb_to_use(A + h * dA, 5)
        sdom, k = dom(A)
        dsd = ds_dp(A, dA)
        # mass-weighted in-use exit rate at the dominant eigenvector
        w, V = np.linalg.eig(A); v = np.abs(V[:, int(np.argmax(w.real))].real)
        wu = v[USE] / v[USE].sum(); lam_eff = float(wu @ (1 / mu))
        recC.append(dict(seed=s, year=yr, tau_olds=t_old[si, ti], p_scrap_to_use=p_su, r_U=rU,
                         drU_dtau=(rU_h - rU) / h, s_dom=sdom, T_dom=-1 / sdom,
                         ds_dtau_olds=dsd, elas_Tdom_tau=-dsd * t_old[si, ti] / sdom,
                         dom_state=int(np.argmax(v)),
                         s_approx=-lam_eff * (1 - rU), lam_eff=lam_eff,
                         next_gen_radius=nextgen(A)))
C = pd.DataFrame(recC)

# wp2c cross-check: exact ds/dtau_olds from the stored modal sensitivities
dm = ZE['dominant_mode']
d_wp2c = np.take_along_axis(ZE['dlambda_state'][..., j_old], dm[..., None], axis=2)[..., 0]
check('ds_dom/dtau_olds recomputed vs WP-2c (max abs diff, all nodes)', 0.0,
      float(np.max(np.abs(C.ds_dtau_olds.values.reshape(35, 40) - d_wp2c))), 1e-9)
check('dominant mode is the 44-yr in-use cohort at every node', 1.0,
      float((C.dom_state == 4).all()), 0)
check('r_U from absorption vs finite-difference consistency (dr_U/dtau = P(scrap->use))', 0.0,
      float(np.max(np.abs(C.drU_dtau - C.p_scrap_to_use))), 1e-5)

# frozen-equilibrium macro rho and its derivative in tau_olds (block D)
recD = []
alph = ZA['alphas']; an = list(ZA['alpha_names']); tn = list(ZA['tau_binary_names'])
fu, eu = ZA['frac_fu'], ZA['frac_eu']
for si, s in enumerate(seeds):
    for ti, yr in enumerate(yA):
        a_cc, a_refc, a_win, a_dr = alph[si, ti]
        t_wae = taus[si, ti, tn.index('tau_waelz')]
        nss = fu[si, ti, 0] + eu[si, ti, 0] * fu[si, ti, 2]   # new-scrap share of manufacturing input

        def rho_star(A):
            Sx = np.linalg.solve(-A, b_all[si, ti])
            eol = float(np.sum(Sx[USE] / mu))
            manu = a_refc * Sx[1] + a_dr * Sx[5]
            rec = (t_wae * a_win + a_dr) * Sx[5]
            thr = a_refc * Sx[1] + a_dr * Sx[5]
            return rec / (eol + nss * manu), rec / thr

        A = A_all[si, ti]; h = 1e-6
        r0, c0 = rho_star(A); r1, c1 = rho_star(A + h * dA_old[si, ti])
        recD.append(dict(seed=s, year=yr, rho_star=r0, chi_frozen=c0, drho_dtau=(r1 - r0) / h,
                         dchi_frozen_dtau=(c1 - c0) / h))
Dd = pd.DataFrame(recD)
C = C.merge(Dd, on=['seed', 'year'])
C.to_csv(os.path.join(OUT, 'd2_ude_mechanism_by_seed_year.csv'), index=False)

# map: macro headroom derivative per unit tau_olds, at measured coefficients (g0_n001)
Bg = B[B.variant == 'g0_n001'][['seed', 'year', 'dheadroom_drho', 'dmu_dom_drho']]
CM = C.merge(Bg, on=['seed', 'year'], how='inner')
CM['dheadroom_dtau'] = CM.dheadroom_drho * CM.drho_dtau
CM['dmu_macro_dtau'] = CM.dmu_dom_drho * CM.drho_dtau

headC = []
for yr in (1980.0, 1981.0, 2019.0):
    g = C[C.year == yr]
    for k in ('tau_olds', 'p_scrap_to_use', 'r_U', 's_dom', 'T_dom', 's_approx', 'ds_dtau_olds',
              'elas_Tdom_tau', 'next_gen_radius', 'rho_star', 'drho_dtau'):
        headC.append(dict(year=yr, quantity=k, **summ(g[k])))
for k in ('ds_dtau_olds', 'elas_Tdom_tau', 'r_U', 'next_gen_radius'):
    headC.append(dict(year='all nodes', quantity=k, **summ(C[k])))
for yr in (1981.0, 2019.0):
    g = CM[CM.year == yr]
    for k in ('dheadroom_dtau', 'dmu_macro_dtau'):
        headC.append(dict(year=yr, quantity=k, **summ(g[k])))
# change in r_U and s_dom over the window, per seed
d_rU = C[C.year == 2019].set_index('seed').r_U - C[C.year == 1980].set_index('seed').r_U
d_T = C[C.year == 2019].set_index('seed').T_dom - C[C.year == 1980].set_index('seed').T_dom
headC.append(dict(year='2019-1980', quantity='delta r_U', **summ(d_rU)))
headC.append(dict(year='2019-1980', quantity='delta T_dom (yr)', **summ(d_T)))
rel_err = (C.s_approx - C.s_dom) / C.s_dom
headC.append(dict(year='all nodes', quantity='rel error of s = -lam_eff (1 - r_U)', **summ(rel_err)))
HC = pd.DataFrame(headC)

# ---------------------------------------------------------------- copper (D12 builder)
AL = {1: 9.60, 2: 11.20, 3: 0.60, 4: 1.30, 5: 0.01, 6: 0.33, 7: 0.0, 8: 11.80, 9: 0.02,
      10: 0.0, 11: 1.80, 12: 0.0, 13: 0.70, 14: 1.40}
DIS = {k: 0.01 for k in range(1, 9)}


def build_A(al=AL, dis=DIS):
    a, d, A = al, dis, np.zeros((8, 8))
    A[0, 0] = -(a[1] + a[4] + d[1]); A[0, 5] = a[5]
    A[1, 0] = a[1]; A[1, 1] = -(a[2] + a[6] + d[2]); A[1, 2] = a[3]; A[1, 4] = a[14]; A[1, 6] = a[7]
    A[2, 1] = a[2]; A[2, 2] = -(a[3] + a[8] + a[10] + d[3]); A[2, 4] = a[13]
    A[3, 2] = a[8]; A[3, 3] = -(a[9] + d[4])
    A[4, 2] = a[10]; A[4, 3] = a[9]; A[4, 4] = -(a[11] + a[13] + a[14] + d[5]); A[4, 7] = a[12]
    A[5, 0] = a[4]; A[5, 5] = -(a[5] + d[6])
    A[6, 1] = a[6]; A[6, 6] = -(a[7] + d[7])
    A[7, 4] = a[11]; A[7, 7] = -(a[12] + d[8])
    return A


def cu_with_c(c, al=AL, dis=DIS):
    tot = al[11] + al[13] + al[14] + dis[5]
    split = al[13] / (al[13] + al[14])
    rec = c * tot
    a = dict(al); a[13] = rec * split; a[14] = rec * (1 - split)
    a[11] = tot - rec - dis[5]
    return build_A(a, dis)


def cu_rU(A):
    tr = [0, 1, 2, 4, 5, 6, 7]
    Q = A[np.ix_(tr, tr)]; R = A[3, tr]
    x = np.linalg.solve(-Q, np.eye(7)[tr.index(4)])
    return float(R @ x)


Acu = build_A()
ev = np.sort(np.linalg.eigvals(Acu).real)
check('copper basic eigenvalues reproduce tab:eigen (in-use -0.0194)', -0.0194,
      round(float(ev[-4]), 4), 6e-5)
c0 = (AL[13] + AL[14]) / (AL[11] + AL[13] + AL[14] + DIS[5])
check('copper rebuild at c0 equals the published matrix', 0.0, float(np.max(np.abs(cu_with_c(c0) - Acu))), 1e-12)


def cu_inuse_mode(A):
    w, V = np.linalg.eig(A)
    part = np.abs(V[3, :]) / np.abs(V).sum(axis=0)
    k = int(np.argmax(part)); return w[k].real


s_cu = cu_inuse_mode(Acu); rU_cu = cu_rU(Acu)
h = 1e-6
ds_dc = (cu_inuse_mode(cu_with_c(c0 + h)) - cu_inuse_mode(cu_with_c(c0 - h))) / (2 * h)
drU_dc = (cu_rU(cu_with_c(c0 + h)) - cu_rU(cu_with_c(c0 - h))) / (2 * h)
# the tab:sens partials for alpha13, alpha14 on the in-use mode (absolute units)
def ds_dalpha(k):
    a1 = dict(AL); a2 = dict(AL); a1[k] += h; a2[k] -= h
    return (cu_inuse_mode(build_A(a1)) - cu_inuse_mode(build_A(a2))) / (2 * h)
# and on the whole spectrum, to locate tab:sens's -1.01 entries
def spec_d(k):
    a1 = dict(AL); a2 = dict(AL); a1[k] += h; a2[k] -= h
    return (np.sort(np.linalg.eigvals(build_A(a1)).real) - np.sort(np.linalg.eigvals(build_A(a2)).real)) / (2 * h)
sd13 = spec_d(13)
check('tab:sens alpha13 row: largest |dlambda/dalpha13| = 1.01', 1.01, round(float(np.max(np.abs(sd13))), 2), 0.005)
lamEoL, loss4 = AL[9], DIS[4]
cu = dict(model='copper basic, 1994', collection=c0, r_U=rU_cu, s_inuse=s_cu, T_inuse=-1 / s_cu,
          s_formula=-(loss4 + lamEoL * (1 - rU_cu)), ds_dcollection=ds_dc,
          elas_T_collection=-ds_dc * c0 / s_cu, drU_dcollection=drU_dc,
          ds_dalpha13=ds_dalpha(13), ds_dalpha14=ds_dalpha(14),
          ds_dalpha9=ds_dalpha(9), next_gen_radius=nextgen(Acu))
check('copper: s_inuse = -(d4 + a9 (1 - r_U)) (rel)', 0.0, abs(cu['s_formula'] / s_cu - 1), 0.02)
CU = pd.DataFrame([cu])
CU.to_csv(os.path.join(OUT, 'd2_copper_collection.csv'), index=False)

# ---------------------------------------------------------------- the side-by-side table
zi19 = C[C.year == 2019]; zi80 = C[C.year == 1980]
mz = A_tab.set_index('case')
b19 = B[(B.variant == 'g0_n001') & (B.year == 2019)]
b81 = B[(B.variant == 'g0_n001') & (B.year == 1981)]
side = pd.DataFrame([
    dict(model='macro, zinc calibration (gamma=0.003)', unit='per capita, n=0.01', collection_variable='rho',
         collection=mz.loc['Zinc, gamma=0.003', 'rho'], r_U=np.nan,
         s_slow=mz.loc['Zinc, gamma=0.003', 'mu_dom'], T_slow=mz.loc['Zinc, gamma=0.003', 'T_dom'],
         ds_dcollection=mz.loc['Zinc, gamma=0.003', 'dmu_dom_drho'],
         elas_T_collection=mz.loc['Zinc, gamma=0.003', 'elas_Tdom_rho'],
         threshold_distance=mz.loc['Zinc, gamma=0.003', 'headroom'], threshold_measure='1 - Gamma/Gamma_M'),
    dict(model='macro, zinc no-urban-mining (gamma=0)', unit='per capita, n=0.01', collection_variable='rho',
         collection=mz.loc['Zinc, gamma=0', 'rho'], r_U=mz.loc['Zinc, gamma=0', 'r_U_gamma0'],
         s_slow=mz.loc['Zinc, gamma=0', 'mu_dom'], T_slow=mz.loc['Zinc, gamma=0', 'T_dom'],
         ds_dcollection=mz.loc['Zinc, gamma=0', 'dmu_dom_drho'],
         elas_T_collection=mz.loc['Zinc, gamma=0', 'elas_Tdom_rho'],
         threshold_distance=mz.loc['Zinc, gamma=0', 'headroom'], threshold_measure='1 - Gamma/Gamma_M'),
    dict(model='macro block on measured zinc coefficients, 2019 (gamma=0, psi=0.613)', unit='per capita, n=0.01',
         collection_variable='rho', collection=b19.rho.median(), r_U=b19.r_U.median(),
         s_slow=b19.mu_dom.median(), T_slow=-1 / b19.mu_dom.median(),
         ds_dcollection=b19.dmu_dom_drho.median(),
         elas_T_collection=float(np.median(-b19.dmu_dom_drho * b19.rho / b19.mu_dom)),
         threshold_distance=b19.headroom.median(), threshold_measure='1 - Gamma/Gamma_M'),
    dict(model='copper basic, 1994', unit='aggregate', collection_variable='c = (a13+a14)/exit(s5)',
         collection=c0, r_U=rU_cu, s_slow=s_cu, T_slow=-1 / s_cu, ds_dcollection=ds_dc,
         elas_T_collection=cu['elas_T_collection'], threshold_distance=1 - cu['next_gen_radius'],
         threshold_measure='1 - next-generation radius'),
    dict(model='zinc UDE, 2019 (35-seed median)', unit='aggregate', collection_variable='tau_olds',
         collection=zi19.tau_olds.median(), r_U=zi19.r_U.median(), s_slow=zi19.s_dom.median(),
         T_slow=zi19.T_dom.median(), ds_dcollection=zi19.ds_dtau_olds.median(),
         elas_T_collection=zi19.elas_Tdom_tau.median(),
         threshold_distance=1 - zi19.next_gen_radius.median(), threshold_measure='1 - next-generation radius'),
    dict(model='zinc UDE, 1980 (35-seed median)', unit='aggregate', collection_variable='tau_olds',
         collection=zi80.tau_olds.median(), r_U=zi80.r_U.median(), s_slow=zi80.s_dom.median(),
         T_slow=zi80.T_dom.median(), ds_dcollection=zi80.ds_dtau_olds.median(),
         elas_T_collection=zi80.elas_Tdom_tau.median(),
         threshold_distance=1 - zi80.next_gen_radius.median(), threshold_measure='1 - next-generation radius'),
])
side.to_csv(os.path.join(OUT, 'd2_side_by_side.csv'), index=False)

summary = pd.concat([HB.assign(block='B'), HC.assign(block='C')], ignore_index=True)
summary.to_csv(os.path.join(OUT, 'd2_summary.csv'), index=False)
V = pd.DataFrame(val); V.to_csv(os.path.join(OUT, 'd2_validation.csv'), index=False)
pd.DataFrame([dict(n_obs_trend=n_obs, rho_offset=RHO_OFFSET, psi_zn=MM.ZN_BASE.psi)]).to_csv(
    os.path.join(OUT, 'd2_constants.csv'), index=False)

pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30)
print(V.to_string()); print()
print(A_tab.T.to_string()); print()
print(side.T.to_string()); print()
print(CU.T.to_string()); print()
print(HB.to_string()); print()
print(HC.to_string()); print('n_obs', n_obs, 'offset', RHO_OFFSET)


# =====================================================================================
# E. in-use mode of the macro block; the compositional rho; collection at its ceiling
# =====================================================================================
def mu_inuse(p):
    """Eigenvalue of J_M whose eigenvector loads most on u (J11 exactly when gamma = 0)."""
    w, V = np.linalg.eig(MM.jacobian(p))
    part = np.abs(V[0, :]) / np.abs(V).sum(axis=0)
    return float(w[int(np.argmax(part))].real)


rowsE = []
for p in MM.CASES:
    m = mu_inuse(p)
    rowsE.append(dict(case=p.name, mu_inuse=m, T_inuse=-1 / m,
                      dmu_inuse_drho=d_central(mu_inuse, p, 'rho'),
                      elas_Tinuse_rho=-d_central(mu_inuse, p, 'rho') * p.rho / m,
                      rU_psi0=p.rho * (1 - p.sigma) * p.kappa / (1 - p.rho * p.sigma),
                      rU_psi=rU_macro(p), rU_threshold=1 + p.n / p.lam,
                      mu_inuse_psi0=mu_inuse(mp(p, psi=0.0)),
                      dmu_inuse_drho_psi0=d_central(mu_inuse, mp(p, psi=0.0), 'rho')))
E_tab = pd.DataFrame(rowsE)
E_tab.to_csv(os.path.join(OUT, 'd2_macro_inuse_mode.csv'), index=False)

tw_i = tn.index('tau_waelz')
recE = []
for si, s in enumerate(seeds):
    par = par_by_seed[s]
    z = np.load(os.path.join('analysis', 'wp9', f'ch1_seed{s}.npz'), allow_pickle=True)
    Fz = z['F']; f = lambda nm: Fz[:, CH1._fidx(nm)]
    NS = f('first_use_new_scrap') + f('end_use_new_scrap')
    ML = f('first_use_losses') + f('end_use_losses')
    EOL = f('end_of_life')
    for i, yr in enumerate(years):
        ti = i + 1                                            # A(t) at the interval's end year
        a_cc, a_refc, a_win, a_dr = alph[si, ti]
        p_rec = (taus[si, ti, tw_i] * a_win + a_dr) / (a_win + a_dr)
        r_old = t_old[si, ti] * p_rec
        r_new = p_rec                                         # all new scrap enters the scrap stock
        r_new_sig = p_rec * NS[i] / (NS[i] + ML[i])            # per unit of sigma S (new scrap + losses)
        rho_agg = (r_old * EOL[i] + r_new * NS[i]) / (EOL[i] + NS[i])
        lam, sig, kap, rho = (float(par[k][i]) for k in ('lam', 'sig', 'kap', 'rho'))
        psi, n = MM.ZN_BASE.psi, 0.01
        Gs = kap / (1 - psi - r_new_sig * sig)
        R_split = (1 - sig) * r_old * lam * Gs / (n + lam)
        p1 = mp(MM.ZN_BASE, lam=lam, sigma=sig, kappa=kap, rho=rho, gamma=0.0, n=n)
        # collection at its ceiling: tau_olds -> 1 in the UDE's own A(t)
        A1 = A_all[si, ti] + (1.0 - t_old[si, ti]) * dA_old[si, ti]
        recE.append(dict(seed=s, year=yr, rho_wp9=rho, rho_old=r_old, rho_new=r_new,
                         rho_new_per_sigmaS=r_new_sig, rho_agg_check=rho_agg,
                         eol_weight=EOL[i] / (EOL[i] + NS[i]),
                         rU_macro_psi0=rho * (1 - sig) * kap / (1 - rho * sig),
                         rU_macro_psi=rU_macro(p1),
                         rU_ude=t_old[si, ti] * absorb_to_use(A_all[si, ti], 5),
                         rU_split_psi=(1 - sig) * r_old * kap / (1 - psi - r_new_sig * sig),
                         headroom_single_rho=headroom(p1), headroom_split_rho=1 - R_split,
                         s_dom_tau1=dom(A1)[0], rU_tau1=absorb_to_use(A1, 5)))
E = pd.DataFrame(recE)
E.to_csv(os.path.join(OUT, 'd2_compositional_rho_by_seed_year.csv'), index=False)

headE = []
for yr in (1981.0, 2019.0):
    g = E[E.year == yr]
    for k in ('rho_wp9', 'rho_agg_check', 'rho_old', 'rho_new', 'eol_weight', 'rU_ude', 'rU_macro_psi0',
              'rU_macro_psi', 'rU_split_psi', 'headroom_single_rho', 'headroom_split_rho', 's_dom_tau1', 'rU_tau1'):
        headE.append(dict(year=yr, quantity=k, **summ(g[k])))
headE.append(dict(year='all nodes', quantity='max s_dom at tau_olds = 1', median=float(E.s_dom_tau1.max()),
                  q1=np.nan, q3=np.nan, hl=np.nan, hl_lo=np.nan, hl_hi=np.nan))
headE.append(dict(year='all nodes', quantity='rel diff rho_agg_check vs rho_wp9',
                  **summ((E.rho_agg_check - E.rho_wp9) / E.rho_wp9)))
# decomposition of the 1981 -> 2019 rise of the aggregate rho (per seed, symmetric swap)
dec = []
for s, g in E.groupby('seed'):
    a, b = g[g.year == 1981.0].iloc[0], g[g.year == 2019.0].iloc[0]
    agg = lambda ro, rn, w: ro * w + rn * (1 - w)
    tot = agg(b.rho_old, b.rho_new, b.eol_weight) - agg(a.rho_old, a.rho_new, a.eol_weight)
    parts = {}
    for k, (x0, x1) in dict(rho_old=(a.rho_old, b.rho_old), rho_new=(a.rho_new, b.rho_new),
                            eol_weight=(a.eol_weight, b.eol_weight)).items():
        args0 = dict(ro=a.rho_old, rn=a.rho_new, w=a.eol_weight)
        args1 = dict(ro=b.rho_old, rn=b.rho_new, w=b.eol_weight)
        key = dict(rho_old='ro', rho_new='rn', eol_weight='w')[k]
        fw = agg(**{**args0, key: args1[key]}) - agg(**args0)
        bw = agg(**args1) - agg(**{**args1, key: args0[key]})
        parts[k] = 0.5 * (fw + bw)
    dec.append(dict(seed=s, total=tot, **parts))
DEC = pd.DataFrame(dec)
for k in ('total', 'rho_old', 'rho_new', 'eol_weight'):
    headE.append(dict(year='1981->2019', quantity=f'd rho_agg: {k}', **summ(DEC[k])))
HE = pd.DataFrame(headE)
HE.to_csv(os.path.join(OUT, 'd2_compositional_summary.csv'), index=False)
print(); print(E_tab.T.to_string()); print(); print(HE.to_string())
