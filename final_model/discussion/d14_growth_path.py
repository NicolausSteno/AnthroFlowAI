#!/usr/bin/env python3
"""
D14 -- the observed zinc growth path against the intensity-sustainable growth path of
the macro chapter (Proposition prop:oc_intensity_growth).  Thesis Discussion,
sec:thdisc_growth, paragraph block 4 step (ii).  Analytic; no refits.

Inputs
  * the 35 anchor_v4 UDE trajectories cached by WP-9 (analysis/wp9/ch1_seed*.npz),
    mapped to the macro chapter's (lambda, sigma, kappa, rho, chi, s, u, d) by
    zinc_ch1_lab.ch1_params_from_traj -- the WP-9 mapping, unchanged;
  * world GDP and population, the UDE's own exogenous drivers (zinc_ch1_lab.driver_series);
  * the macro chapter's zinc calibration (Macro-Materials/mm_core.py: ZN_GAMMA, ZN_BASE)
    and its ore-grade term eta*delta = 1 x 0.020 /yr (ENERGY_ZN);
  * the macro chapter's illustrative vartheta = 0.6 (mm_growth.TECH, mm_optimal.OCParams).

What is computed (per seed; median, IQR and Hodges-Lehmann interval over seeds)
  A. Observed growth: trend rates of world GDP g_Y, population n, per-capita GDP g_I = g_Y - n.
  B. Distance from the path: trend drift of the four intensities I_j = X_j / Y
     (j = S throughput, U in-use, D discard, P primary refining).  All are zero on the
     intensity-sustainable path (eq. oc_intensity_bgp_def).  I_U = U/Y = u/y, so the
     drift is invariant to the per-capita normalisation.
  C. Material feasibility (eq. oc_prop_material_feasibility): chi_I(g) at the measured
     coefficients (time means, gamma = 0: the UDE has no discard-recovery flow) and at the
     macro calibration, at g = observed g_I and at g = observed per-capita throughput
     growth; the margin 1 - chi_I; and the critical rate g_crit < 0 at which chi_I = 1.
     Also B_U(g) against the observed u/s, and the growth rate at which B_U(g) = u/s.
  D. vartheta (eq. oc_prop_primary_condition).  The per-capita elasticities of s, u, d, p to
     per-capita GDP are the positive model's vartheta (g_x = vartheta g_k, g_k = g_I at
     constant K/Y).  WP-9's H7/H8 elasticities are aggregate (totals on world GDP); the
     conversion is vartheta = (e g_Y - n) / g_I.  For each vartheta the primary condition
     gives the required g_P = eta delta + (1 - vartheta) g_I; with constant K/Y the realised
     g_P - required g_P equals the drift of I_P, whatever vartheta is.  The inverse map
     vartheta(g_P) = 1 - (g_P - eta delta)/g_I is tabulated for a grid of g_P.

Run from AnthroFlowAI/final_model (needs analysis/wp9/ch1_seed*.npz from run_wp9.py and the
observed record for the GDP and population drivers):
    PYTHONDONTWRITEBYTECODE=1 python discussion/d14_growth_path.py [--out discussion/outputs] [--mm <Macro-Materials>]
Writes d14_growth_rates.csv, d14_intensity_drift.csv, d14_feasibility.csv,
d14_vartheta.csv, d14_vartheta_inverse.csv, d14_validation.csv, d14_per_seed.csv.
"""
import argparse, os, sys

# sibling repositories (clone Macro-Materials and Simulating-Resources next to AnthroFlowAI,
# or point the environment variables MACRO_MATERIALS_DIR / SIMULATING_RESOURCES_DIR at them)
_FM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPOS = os.path.dirname(os.path.dirname(_FM))
MM_DIR = os.environ.get('MACRO_MATERIALS_DIR', os.path.join(_REPOS, 'Macro-Materials'))
SR_DIR = os.environ.get('SIMULATING_RESOURCES_DIR', os.path.join(_REPOS, 'Simulating-Resources'))
OUT_DEFAULT = os.path.join(_FM, 'discussion', 'outputs')
import numpy as np, pandas as pd

sys.path.insert(0, os.getcwd())
import zinc_ch1_lab as CH1                     # noqa: E402
from run_wp9 import hl, med_iqr, trend_slope_weights   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=OUT_DEFAULT)
ap.add_argument('--mm', default=MM_DIR)
args = ap.parse_args()
sys.path.insert(0, args.mm)
from mm_core import ZN_GAMMA, ZN_BASE, ENERGY_ZN          # noqa: E402
from mm_growth import chi_g, BU as mm_BU, TECH            # noqa: E402

ETA_DELTA = ENERGY_ZN.eta * ENERGY_ZN.delta
VARTHETA_CH2 = TECH.theta_v

# ---------------------------------------------------------------- data
cfg = CH1.load_anchor_config()
_fit, ctx = CH1.build_context(cfg)
seeds = CH1.seed_list()
z0 = np.load(os.path.join('analysis', 'wp9', f'ch1_seed{seeds[0]}.npz'), allow_pickle=True)
years = z0['years'][1:]                                   # WP-9 H8 alignment, 1981-2019
tg, G = CH1.driver_series(ctx, CH1.GDP_DRIVER)
tp, L = CH1.driver_series(ctx, CH1.POPULATION_DRIVER)
Y = np.interp(years, tg, G); N = np.interp(years, tp, L); y = Y / N
w = trend_slope_weights(years)
slope = lambda v: float(np.sum(w * np.log(v)))           # OLS trend of ln v, per yr

def elast(xv, yv):
    A = np.c_[np.log(yv), np.ones_like(yv)]
    return float(np.linalg.lstsq(A, np.log(xv), rcond=None)[0][0])

gY, n_obs, gI = slope(Y), slope(N), slope(y)
assert abs(gI - (gY - n_obs)) < 1e-12

def chiI(lam, sig, kap, rho, n, g, gam=0.0, q=0.0):
    BU = (1 - sig) * kap / (n + lam + g)
    BD = q * (1 - rho) * (lam * BU + sig) / (n + gam + g) if gam > 0 else 0.0
    return rho * sig + rho * lam * BU + gam * BD, BU

def g_crit(lam, sig, kap, rho, n):                         # chi_I = 1, gamma = 0
    return rho * lam * (1 - sig) * kap / (1 - rho * sig) - n - lam

# ---------------------------------------------------------------- per seed
rows = []
for s in seeds:
    z = np.load(os.path.join('analysis', 'wp9', f'ch1_seed{s}.npz'), allow_pickle=True)
    par = CH1.ch1_params_from_traj(z['S'], z['F'])
    prim = z['F'][:, CH1._fidx('primary_refining')]
    X = dict(S=par['throughput'], U=par['inuse'], D=par['discard'], P=prim)
    lam, sig, kap, rho = (float(np.mean(par[k])) for k in ('lam', 'sig', 'kap', 'rho'))
    chi_meas = float(np.mean(par['chi']))
    us_obs = float(np.mean(par['inuse'] / par['throughput']))
    r = dict(seed=s, lam=lam, sig=sig, kap=kap, rho=rho, chi_measured=chi_meas, us_obs=us_obs)
    for j, v in X.items():
        r[f'e_agg_{j}'] = elast(v, Y)                       # WP-9 convention (totals on GDP)
        r[f'vt_{j}'] = elast(v / N, y)                      # per capita on per capita: vartheta
        r[f'vt_trend_{j}'] = slope(v / N) / gI              # trend-ratio form
        r[f'drift_I{j}'] = slope(v / Y)                     # d ln I_j / dt
        r[f'g_pc_{j}'] = slope(v / N)
    r['us_trend'] = slope(par['inuse'] / par['throughput'])
    for lab, n_, g_ in [('gI_nobs', n_obs, gI), ('gI_n001', 0.01, gI),
                        ('gs_nobs', n_obs, r['g_pc_S']), ('g0_n001', 0.01, 0.0), ('g0_nobs', n_obs, 0.0)]:
        c, b = chiI(lam, sig, kap, rho, n_, g_)
        r[f'chiI_{lab}'], r[f'BU_{lab}'] = c, b
    r['gcrit_nobs'] = g_crit(lam, sig, kap, rho, n_obs)
    r['gcrit_n001'] = g_crit(lam, sig, kap, rho, 0.01)
    r['g_tot_implied_by_us'] = (1 - sig) * kap / us_obs - lam      # n + g at which B_U = u/s
    r['g_pc_implied_nobs'] = r['g_tot_implied_by_us'] - n_obs
    rows.append(r)
P = pd.DataFrame(rows)

def summ(v):
    v = np.asarray(v, float); m, q1, q3 = med_iqr(v); e, lo, hi = hl(v)
    return dict(median=m, q1=q1, q3=q3, hl=e, hl_lo=lo, hl_hi=hi)

# ---------------------------------------------------------------- A. growth rates
gr = pd.DataFrame([
    dict(quantity='world GDP, trend growth g_Y (/yr)', value=gY),
    dict(quantity='population, trend growth n (/yr)', value=n_obs),
    dict(quantity='GDP per capita, trend growth g_I (/yr)', value=gI),
    dict(quantity='macro chapter n (calibration, /yr)', value=ZN_GAMMA.n),
    dict(quantity='macro chapter eta*delta (/yr)', value=ETA_DELTA),
    dict(quantity='sample', value=f'{int(years[0])}-{int(years[-1])}'),
])

# ---------------------------------------------------------------- B. intensity drift
names = dict(S='throughput s', U='in-use stock u', D='discard stock d (1980 anchor)', P='primary refining p')
dr = []
for j, nm in names.items():
    d = summ(P[f'drift_I{j}']); d.update(quantity=f'd ln I_{j}/dt, {nm}', path_value=0.0,
                                         cum_change_pct_1981_2019=100 * (np.exp(d['median'] * (years[-1] - years[0])) - 1))
    dr.append(d)
dr = pd.DataFrame(dr)[['quantity', 'median', 'q1', 'q3', 'hl', 'hl_lo', 'hl_hi', 'path_value', 'cum_change_pct_1981_2019']]

# ---------------------------------------------------------------- C. feasibility
fe = []
def add(lbl, v, ref=None):
    d = summ(v) if np.ndim(v) else dict(median=float(v)); d.update(quantity=lbl, reference=ref); fe.append(d)
add('chi measured (UDE, time mean)', P.chi_measured)
add('chi_I(g_I obs), measured coefs, n obs', P.chiI_gI_nobs)
add('chi_I(g_I obs), measured coefs, n = 0.01', P.chiI_gI_n001)
add('chi_I(g = obs per-capita throughput growth), measured coefs, n obs', P.chiI_gs_nobs)
add('chi_I(0) = chi*, measured coefs, n = 0.01 (WP-9 steady state)', P.chiI_g0_n001)
add('margin 1 - chi_I(g_I obs), measured coefs, n obs', 1 - P.chiI_gI_nobs)
add('g_crit per capita where chi_I = 1, measured coefs, n obs (/yr)', P.gcrit_nobs)
add('g_crit per capita where chi_I = 1, measured coefs, n = 0.01 (/yr)', P.gcrit_n001)
for M, lab in [(ZN_GAMMA, 'calibration gamma=0.003, rho=0.345'), (ZN_BASE, 'no urban mining gamma=0, rho0=0.363')]:
    for n_, nl in [(n_obs, 'n obs'), (M.n, 'n = 0.01')]:
        Mn = M.__class__(**{**M.__dict__, 'n': n_})
        add(f'chi_I(g_I obs), macro {lab}, {nl}', chi_g(Mn, gI))
        add(f'B_U(g_I obs), macro {lab}, {nl} (yr)', mm_BU(Mn, gI))
    add(f'chi_I(0) = chi*, macro {lab}, n = 0.01', chi_g(M, 0.0))
add('u/s observed (UDE, yr)', P.us_obs)
add('trend in ln(u/s), per yr', P.us_trend)
add('B_U(g_I obs), measured coefs, n obs (yr)', P.BU_gI_nobs)
add('B_U(g = obs per-capita throughput growth), measured coefs, n obs (yr)', P.BU_gs_nobs)
add('B_U(0), measured coefs, n = 0.01 (yr; WP-9 reference)', P.BU_g0_n001)
add('saturation u/s / B_U(g_I obs), n obs', P.us_obs / P.BU_gI_nobs)
add('saturation u/s / B_U(g_s obs), n obs', P.us_obs / P.BU_gs_nobs)
add('saturation u/s / B_U(0), n = 0.01 (WP-9 0.805)', P.us_obs / P.BU_g0_n001)
add('n + g at which B_U(g) = observed u/s (/yr)', P.g_tot_implied_by_us)
add('per-capita g at which B_U(g) = observed u/s, n obs (/yr)', P.g_pc_implied_nobs)
add('observed per-capita throughput growth (/yr)', P.g_pc_S)
fe = pd.DataFrame(fe)[['quantity', 'median', 'q1', 'q3', 'hl', 'hl_lo', 'hl_hi', 'reference']]

# ---------------------------------------------------------------- D. vartheta
vt = []
for j, nm in names.items():
    e_agg, v_pc, v_tr = summ(P[f'e_agg_{j}']), summ(P[f'vt_{j}']), summ(P[f'vt_trend_{j}'])
    conv = (P[f'e_agg_{j}'] * gY - n_obs) / gI
    gP_req = ETA_DELTA + (1 - P[f'vt_{j}']) * gI
    vt.append(dict(source=nm, aggregate_elasticity_WP9=e_agg['median'],
                   vartheta_per_capita=v_pc['median'], vartheta_pc_q1=v_pc['q1'], vartheta_pc_q3=v_pc['q3'],
                   vartheta_pc_hl_lo=v_pc['hl_lo'], vartheta_pc_hl_hi=v_pc['hl_hi'],
                   vartheta_trend_ratio=v_tr['median'], vartheta_from_conversion=float(np.median(conv)),
                   gP_required=float(np.median(gP_req)), gP_net_required=float(np.median(gP_req - ETA_DELTA)),
                   realised_minus_required_gP=float(np.median(P['drift_IP']))))
for v, nm in [(VARTHETA_CH2, 'macro chapter illustrative'), (1.0, 'vartheta = 1 (footnote limit)')]:
    vt.append(dict(source=nm, vartheta_per_capita=v, gP_required=ETA_DELTA + (1 - v) * gI,
                   gP_net_required=(1 - v) * gI, realised_minus_required_gP=float(np.median(P['drift_IP']))))
vt = pd.DataFrame(vt)

inv = pd.DataFrame([dict(g_P=gp, net_gP=gp - ETA_DELTA, vartheta_implied=1 - (gp - ETA_DELTA) / gI,
                         admissible=bool(0 < 1 - (gp - ETA_DELTA) / gI < 1))
                    for gp in [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]])
inv.loc[len(inv)] = dict(g_P=np.nan, net_gP=np.nan, vartheta_implied=np.nan, admissible=None)
gp_band = (ETA_DELTA, ETA_DELTA + gI)                     # vartheta in (0,1)

# ---------------------------------------------------------------- validation
pub = pd.read_csv(os.path.join('analysis', 'wp9_balanced_growth.csv')).set_index('quantity')
h7 = pd.read_csv(os.path.join('analysis', 'wp9_intensity_of_use.csv'))
val = pd.DataFrame([
    ('H8 elasticity of throughput to GDP (median)', pub.loc['elasticity of throughput to GDP', 'median'], P.e_agg_S.median()),
    ('H8 elasticity of inuse to GDP (median)', pub.loc['elasticity of inuse to GDP', 'median'], P.e_agg_U.median()),
    ('H8 elasticity of discard to GDP (median)', pub.loc['elasticity of discard to GDP', 'median'], P.e_agg_D.median()),
    ('H8 u/s mean (median)', pub.loc['u_over_s (years of throughput)', 'median'], P.us_obs.median()),
    ('H8 trend ln(u/s) (median)', pub.loc['trend in ln(u_over_s), per yr', 'median'], P.us_trend.median()),
    ('H7 elasticity of in-use (WP-9 regresses on 1980-2019 stocks)', float(h7['median'].iloc[0]), np.nan),
    ('chi_I(0), n=0.01, macro calibration (Ch2 tab: 0.319)', 0.319, chi_g(ZN_GAMMA, 0.0)),
    ('chi_I(0), n=0.01, no urban mining (Ch2 tab: 0.296)', 0.2963, chi_g(ZN_BASE, 0.0)),
    ('WP-9 lambda/sigma/kappa/rho medians', '0.0362/0.1798/0.9832/0.3667',
     '/'.join(f'{P[k].median():.4f}' for k in ('lam', 'sig', 'kap', 'rho'))),
], columns=['check', 'published', 'reproduced'])

os.makedirs(args.out, exist_ok=True)
for nm, df in [('d14_growth_rates', gr), ('d14_intensity_drift', dr), ('d14_feasibility', fe),
               ('d14_vartheta', vt), ('d14_vartheta_inverse', inv.dropna(how='all')), ('d14_validation', val),
               ('d14_per_seed', P)]:
    df.to_csv(os.path.join(args.out, nm + '.csv'), index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(gr.to_string(index=False)); print(dr.round(5).to_string(index=False))
print(fe.round(4).to_string(index=False)); print(vt.round(4).T.to_string())
print(inv.dropna(how='all').round(4).to_string(index=False)); print('g_P band for vartheta in (0,1):', gp_band)
print(val.to_string(index=False))
