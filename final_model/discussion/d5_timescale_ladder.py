#!/usr/bin/env python3
"""
D5 -- one timescale ladder for the four core models.  Thesis Discussion, sec:thdisc_filling.
Analytic; no fitting.  Print chapter numbers: macro = Ch 2, simulation = Ch 3, copper = Ch 4, zinc UDE = Ch 5.

Relaxation times are -1/Re(lambda) of each model's linear (or frozen-time) state matrix; each mode is
labelled by the compartment with the largest participation factor P_ki = |v_ik w_ki| / sum_i.
Delays, cycle periods and approach horizons are kept in a separate class and never mixed with them.

Sources
  Ch 2  Macro-Materials/mm_core.py jacobian() at CU_GAMMA, ZN_GAMMA (calibration) and CU_BASE, ZN_BASE
        (gamma = 0, no-urban-mining comparison); per-capita system, n = 0.01.  Check: tab:steady_state_stability.
  Ch 3  Simulating-Resources/model_v2.R material block A = G U - diag(lambda) at the fitted point
        params_full__slade_cycle_drop_core.json (alpha3, alpha8, alpha9 fitted; other alphas and lambda from
        reference_parms(); lam4 held at 3e-4).  Recycling enters through b (market block), so alpha13 = alpha14 = 0.
        Delays tau_E = 8 (swept 2-32), tau_R = 2 (tab:params); band-pass price period 20.6 yr model, 23.2 yr data.
  Ch 4  copper chapter basic 8x8 matrix and SOA 9x9 matrix, 1994 coefficients (builders copied from
        d12_copper_ratios.py).  Checks: tab:eigen (dissipative column), tab:eigensoa.
  Ch 5  AnthroFlowAI/final_model/analysis/wp2b_eigen_summary.csv (frozen-time spectrum of A(t), 35 seeds x 40 yr).
  Horizons: D12 (copper matrix, constant input from the 1994 stocks) and WP-2g (zinc, A and b frozen at 2019).

Usage (from final_model/):  python discussion/d5_timescale_ladder.py [--macro DIR] [--simres DIR] [--zinc CSV] [--out DIR]
Writes d5_timescales.csv, d5_delays_periods.csv, d5_validation.csv.
"""
import argparse, json, os, sys

# sibling repositories (clone Macro-Materials and Simulating-Resources next to AnthroFlowAI,
# or point the environment variables MACRO_MATERIALS_DIR / SIMULATING_RESOURCES_DIR at them)
_FM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPOS = os.path.dirname(os.path.dirname(_FM))
MM_DIR = os.environ.get('MACRO_MATERIALS_DIR', os.path.join(_REPOS, 'Macro-Materials'))
SR_DIR = os.environ.get('SIMULATING_RESOURCES_DIR', os.path.join(_REPOS, 'Simulating-Resources'))
OUT_DEFAULT = os.path.join(_FM, 'discussion', 'outputs')
from dataclasses import replace
import numpy as np, pandas as pd

AL = {1: 9.60, 2: 11.20, 3: 0.60, 4: 1.30, 5: 0.01, 6: 0.33, 7: 0.0, 8: 11.80, 9: 0.02,
      10: 0.0, 11: 1.80, 12: 0.0, 13: 0.70, 14: 1.40}
DIS = {k: 0.01 for k in range(1, 9)}
SOA = dict(beta=0.20, gam=0.60, rho=1.0, b=0.60, c=0.15, a=2.5)
S94 = np.array([1, 1, 1, 220, 1, 50, 20, 70.])                 # tab:inverse, s1..s8, Tg
F94 = {2: 11.2, 3: 0.6, 8: 11.8, 9: 4.0, 11: 1.8, 13: 0.7, 14: 1.4}
E94 = 9.6 + 1.3 + 0.01 + 0.1 - 0.3                             # f1 + f4 + d1 + ds1/dt - f5
CU_MACRO0 = dict(lam=0.0262, sig=0.170, kap=0.994, rho=0.608, gam=0.0, q=0.630)   # gamma = 0 case
CU_MACRO_S, CU_MACRO_U = 244.1 / 11, 350.0                     # Gloser 2000-10 window mean S, 2010 U (Mt)
N_MACRO = 0.01


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


def build_soa(al=AL, dis=DIS, p=SOA):
    a, d = al, dis
    be, ga, rh, b, c, aa = p['beta'], p['gam'], p['rho'], p['b'], p['c'], p['a']
    k = 1 - b - c; M = np.zeros((9, 9))
    M[0, 0] = -ga * (1 - be * rh / k); M[0, 1] = -be * ga * (1 - d[1] * rh / k); M[1, 0] = 1
    M[2, 0] = b / k; M[2, 1] = b * d[1] / k; M[2, 2] = -(a[2] + a[6] + d[2]); M[2, 3] = a[3]
    M[2, 5] = a[14]; M[2, 7] = a[7]
    M[3, 2] = a[2]; M[3, 3] = -(a[3] + a[8] + a[10] + d[3]); M[3, 5] = a[13]
    M[4, 3] = a[8]; M[4, 4] = -(a[9] + d[4])
    M[5, 3] = a[10]; M[5, 4] = a[9]; M[5, 5] = -(a[11] + a[13] + a[14] + d[5]); M[5, 8] = a[12]
    M[6, 0] = c / k; M[6, 1] = c * d[1] / k; M[6, 6] = -(a[5] + d[6])
    M[7, 2] = a[6]; M[7, 7] = -(a[7] + d[7])
    M[8, 5] = a[11]; M[8, 8] = -(a[12] + d[8])
    bv = np.array([aa * be * ga * rh / k, 0, aa * (1 + b / k), 0, 0, 0, aa * c / k, 0, 0])
    return M, bv




NAMES8 = ['mining (s1)', 'refining (s2)', 'fabrication (s3)', 'in use (s4)', 'waste management (s5)',
          'tailings (s6)', 'slag (s7)', 'landfill (s8)']
GROUP = {'mining (s1)': 'upstream', 'refining (s2)': 'upstream', 'fabrication (s3)': 'upstream',
         'in use (s4)': 'in use', 'waste management (s5)': 'scrap / waste management',
         'tailings (s6)': 'sinks', 'slag (s7)': 'sinks', 'landfill (s8)': 'sinks',
         'SOA production (y)': 'investment (SOA)', 'SOA lag (dy)': 'investment (SOA)',
         'in use (u)': 'in use', 'discard (d)': 'sinks',
         'concentrate': 'upstream', 'refined': 'upstream', 'in use, short': 'in use',
         'in use, medium': 'in use', 'in use, long': 'in use', 'scrap': 'scrap / waste management'}


def participation(A):
    w, V = np.linalg.eig(A)
    W = np.linalg.inv(V)
    P = np.abs(V * W.T)
    return w, P / P.sum(axis=0)


def spectrum_rows(A, names, chapter, model, case):
    w, P = participation(A)
    out = []
    for k in np.argsort(-w.real):
        o = np.argsort(-P[:, k])
        out.append(dict(chapter=chapter, model=model, case=case, re=w[k].real, im=w[k].imag,
                        timescale_yr=-1.0 / w[k].real,
                        period_yr=(2 * np.pi / abs(w[k].imag)) if abs(w[k].imag) > 1e-12 else np.nan,
                        compartment=names[o[0]], participation=P[o[0], k],
                        second=names[o[1]], second_participation=P[o[1], k],
                        group=GROUP[names[o[0]]], ts_lo=np.nan, ts_hi=np.nan, ts_1980=np.nan, ts_2019=np.nan))
    return out


def ch3_matrix(simres):
    pf = json.load(open(f'{simres}/params_full__slade_cycle_drop_core.json'))['parameters']
    alpha = np.array([9.60, 11.20, 0.60, 1.30, 0.006, 0.30, 0, 11.80, 0.02, 0, 1.80, 0, 0, 0])
    alpha[2], alpha[7], alpha[8] = pf['alpha3'], pf['alpha8'], pf['alpha9']
    lam = np.array([0, 0, 0, pf['lam4'], 0, 1e-4, 1e-4, 1e-4])
    edges = [(1, 2, 1), (2, 3, 2), (3, 2, 3), (1, 6, 4), (6, 1, 5), (2, 7, 6), (7, 2, 7), (3, 4, 8),
             (4, 5, 9), (4, 5, 10), (5, 8, 11), (8, 5, 12), (5, 3, 13), (5, 2, 14)]
    G = np.zeros((8, 14))
    for a, b, j in edges:
        G[a - 1, j - 1] -= 1; G[b - 1, j - 1] += 1
    U = np.zeros((14, 8))
    for j, s in enumerate([1, 2, 3, 1, 6, 2, 7, 3, 4, 4, 5, 8, 5, 5]):
        U[j, s - 1] = alpha[j]
    return G @ U - np.diag(lam), pf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--macro', default=MM_DIR)
    ap.add_argument('--simres', default=SR_DIR)
    ap.add_argument('--zinc', default=os.path.join(_FM, 'analysis', 'wp2b_eigen_summary.csv'))
    ap.add_argument('--out', default=OUT_DEFAULT)
    a = ap.parse_args()
    rows, val = [], []

    # ---- Ch 2 macro (per capita, n = 0.01)
    sys.path.insert(0, a.macro)
    import mm_core as mc
    table = {'CU_GAMMA': (-0.0154, -0.00277), 'ZN_GAMMA': (-0.0222, -0.00584),
             'CU_BASE': (-0.0100, -0.00510), 'ZN_BASE': (-0.0134, -0.0100)}
    for nm, case in [('CU_GAMMA', 'copper, calibration'), ('ZN_GAMMA', 'zinc, calibration'),
                     ('CU_BASE', 'copper, gamma = 0'), ('ZN_BASE', 'zinc, gamma = 0')]:
        p = getattr(mc, nm)
        J = mc.jacobian(p)
        rows += spectrum_rows(J, ['in use (u)', 'discard (d)'], 'Ch2', 'macro (per capita, n = 0.01)', case)
        ev = np.sort(np.linalg.eigvals(J).real)
        val.append((f'Ch2 {nm} eigenvalues vs tab:steady_state_stability', ' '.join(map(str, sorted(table[nm]))),
                    ' '.join(f'{e:.3g}' for e in ev)))
        ev0 = np.sort(np.linalg.eigvals(mc.jacobian(replace(p, n=0.0))).real)
        val.append((f'Ch2 {nm} eigenvalues at n = 0 (aggregate frame, stationary population)', 'diagnostic',
                    ' '.join(f'{e:+.5f}' for e in ev0)))

    # ---- Ch 3 simulation, fitted material block
    A3, pf = ch3_matrix(a.simres)
    rows += spectrum_rows(A3, NAMES8, 'Ch3', 'simulation (fitted material block)', 'fitted 1960-2020')

    # ---- Ch 4 copper, basic and SOA
    A4 = build_A()
    rows += spectrum_rows(A4, NAMES8, 'Ch4', 'copper basic', '1994 coefficients')
    M, _ = build_soa()
    rows += spectrum_rows(M, ['SOA production (y)', 'SOA lag (dy)'] + NAMES8[1:], 'Ch4', 'copper SOA',
                          '1994 coefficients')
    val.append(('Ch4 basic spectrum vs tab:eigen (dissipative)', '-14.6 -10.9 -9.33 -3.93 -0.0194 -0.0188 -0.0100 -0.0100',
                ' '.join(f'{e:.3g}' for e in np.sort(np.linalg.eigvals(A4).real))))
    evs = np.linalg.eigvals(M)
    val.append(('Ch4 SOA spectrum vs tab:eigensoa', '-14.6 -9.33 -3.93 -0.0600+-0.334i -0.0200 -0.0194 -0.0100 -0.0100',
                ' '.join(f'{e.real:.3g}{e.imag:+.3g}i' if abs(e.imag) > 1e-9 else f'{e.real:.3g}'
                         for e in sorted(evs, key=lambda e: e.real))))

    # ---- Ch 5 zinc UDE, frozen-time spectrum by year (35 seeds)
    z = pd.read_csv(a.zinc)
    zl = {'S_conc': 'concentrate', 'S_ref': 'refined', 'S_iu_short': 'in use, short', 'S_iu_med': 'in use, medium',
          'S_iu_long': 'in use, long', 'S_scrap': 'scrap'}
    for m, g in z.groupby('mode'):
        g = g.sort_values('year')
        rows.append(dict(chapter='Ch5', model='zinc UDE (frozen A(t), 35 seeds)', case='1980-2019',
                         re=-1 / g.timescale_median.median(), im=0.0, timescale_yr=g.timescale_median.median(),
                         period_yr=np.nan, compartment=zl[m], participation=np.nan, second='', second_participation=np.nan,
                         group=GROUP[zl[m]], ts_lo=g.timescale_q1.min(), ts_hi=g.timescale_q3.max(),
                         ts_1980=g.timescale_median.iloc[0], ts_2019=g.timescale_median.iloc[-1]))
    val.append(('Ch5 max |Im lambda| over 35 seeds x 40 yr', '0 (wp2b)', f"{z.max_abs_im.max():.3g}"))

    ts = pd.DataFrame(rows)
    ts.to_csv(os.path.join(a.out, 'd5_timescales.csv'), index=False)

    dp = pd.DataFrame([
        ('Ch3', 'delay', 'mining lead time tau_E, fitted', 8.0, 2.0, 32.0, 'tab:params; sweep 2-32 yr (Winton Group 2017 range)'),
        ('Ch3', 'delay', 'recycling lead time tau_R', 2.0, np.nan, np.nan, 'tab:params, literature'),
        ('Ch3', 'adjustment', 'price damping to cost anchor 1/phi', 1 / pf['phi'], np.nan, np.nan, 'coefficient, not an eigenvalue'),
        ('Ch3', 'adjustment', 'excess-demand gain on price 1/c', 1 / pf['c'], np.nan, np.nan, 'coefficient, not an eigenvalue'),
        ('Ch3', 'period', 'band-pass price period, fitted model', 20.6, np.nan, np.nan, 'Ch3 l.719'),
        ('Ch3', 'period', 'band-pass price period, data', 23.2, np.nan, np.nan, 'Ch3 l.719'),
        ('Ch4', 'period', 'SOA oscillation period 2 pi / 0.334', 2 * np.pi / 0.334, np.nan, np.nan, 'tab:eigensoa'),
        ('Ch4', 'period', 'zinc refined stocks, about 3 cycles 1990-2019 (Rostek 2022)', 29 / 3, np.nan, np.nan, 'observed, approximate'),
        ('Ch5', 'period', 'zinc UDE: no complex eigenvalue', np.nan, np.nan, np.nan, 'wp2b, 8 400 eigenvalues'),
        ('Ch4', 'horizon', 'copper u/s to 90 / 95 % of fixed point, constant input from 1994', 70.0, 70.0, 105.0, 'D12'),
        ('Ch5', 'horizon', 'zinc in-use stock to 90 / 95 % of S*(2019), A, b frozen at 2019', 72.5, 72.5, 107.8, 'WP-2g'),
    ], columns=['chapter', 'class', 'quantity', 'value_yr', 'lo_yr', 'hi_yr', 'source'])
    dp.to_csv(os.path.join(a.out, 'd5_delays_periods.csv'), index=False)
    pd.DataFrame(val, columns=['check', 'published', 'recomputed']).to_csv(os.path.join(a.out, 'd5_validation.csv'), index=False)
    return ts, dp, pd.DataFrame(val, columns=['check', 'published', 'recomputed'])


if __name__ == '__main__':
    ts, dp, val = main()
    pd.set_option('display.width', 250)
    print(val.to_string(index=False))
    print(ts[['chapter', 'model', 'case', 'compartment', 'participation', 'timescale_yr', 'period_yr', 'ts_lo', 'ts_hi']]
          .round(4).to_string(index=False))
