#!/usr/bin/env python3
"""
D12 -- the copper chapter's calibrated matrix mapped onto the macro chapter's
(lambda, sigma, kappa, rho, chi) and u/s.   Thesis Discussion, sec:thdisc_growth
and sec:thdisc_filling.   Analytic; no fitting.

Inputs (all transcribed from the thesis sources, nothing estimated here):
  * copper chapter Table tab:ch3_params  -- alpha_1..14, alpha_15..22 = 0.01, SOA block
  * simulation chapter Table tab:inverse -- 1994 stocks (Tg) and flows (Tg/yr), Graedel (2004)
  * macro chapter Eqs. chistar / ext_au, Table tab:calibration (copper column, gamma = 0 case)
  * WP-9 zinc reference: analysis/wp9_chi_level_gap.csv (median over 35 seeds)

Flow-ratio definitions are WP-9's (zinc_ch1_lab.py docstring), applied compartment by compartment:
  S      = f2 + f13                        throughput into fabrication: refined metal + direct reuse
  sigma  = (f3 + f10 + d3) / outflow(s3)    new scrap + manufacturing loss share
  lambda = f9 / s4                          end-of-life rate                       [convention B]
  kappa  = alpha9 / (alpha9 + alpha18)      in-use dissipation -> dissipative use  [convention B]
  rho    = (f3 + f13 + f14) / (lambda U + sigma S)
  chi    = (f3 + f13 + f14) / S             gross, as WP-9; chi_net discounts slag + d2 losses
  Convention A instead puts in-use dissipation into lambda (= alpha9 + alpha18, kappa = 1), which
  keeps the copper chapter's 33.3-yr in-use residence time. A and B coincide at n = 0.
Tailings reprocessing (f5) is treated as primary input; landfill recovery alpha12 = 0, so gamma = 0.

Run from any directory:  python d12_copper_ratios.py [--wp9 path/to/wp9_chi_level_gap.csv]
Writes d12_parameter_map.csv, d12_saturation.csv, d12_dissipation_sensitivity.csv,
d12_chi_swap.csv, d12_validation.csv into the working directory.
"""
import argparse, os

# sibling repositories (clone Macro-Materials and Simulating-Resources next to AnthroFlowAI,
# or point the environment variables MACRO_MATERIALS_DIR / SIMULATING_RESOURCES_DIR at them)
_FM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPOS = os.path.dirname(os.path.dirname(_FM))
MM_DIR = os.environ.get('MACRO_MATERIALS_DIR', os.path.join(_REPOS, 'Macro-Materials'))
SR_DIR = os.environ.get('SIMULATING_RESOURCES_DIR', os.path.join(_REPOS, 'Simulating-Resources'))
OUT_DEFAULT = os.path.join(_FM, 'discussion', 'outputs')
import numpy as np, pandas as pd
from scipy.linalg import expm
from scipy.optimize import brentq

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


def flows(s, al=AL, dis=DIS):
    s1, s2, s3, s4, s5, s6, s7, s8 = s
    f = {1: al[1]*s1, 2: al[2]*s2, 3: al[3]*s3, 4: al[4]*s1, 5: al[5]*s6, 6: al[6]*s2, 7: al[7]*s7,
         8: al[8]*s3, 9: al[9]*s4, 10: al[10]*s3, 11: al[11]*s5, 12: al[12]*s8, 13: al[13]*s5,
         14: al[14]*s5}
    return f, {k: dis[k] * s[k - 1] for k in range(1, 9)}


def macro_map(s, conv, al=AL, dis=DIS):
    f, d = flows(s, al, dis); U = s[3]; S = f[2] + f[13]
    sig = (f[3] + f[10] + d[3]) / (f[3] + f[8] + f[10] + d[3])
    rec = f[3] + f[13] + f[14]
    pass2 = al[2] / (al[2] + al[6] + dis[2])
    rec_net = pass2 * (f[3] + f[14]) + f[13]
    if conv == 'B':
        lam, kap, eol = f[9] / U, al[9] / (al[9] + dis[4]), f[9]
    else:
        lam, kap, eol = (f[9] + d[4]) / U, 1.0, f[9] + d[4]
    den = eol + sig * S
    return dict(lam=lam, sig=sig, kap=kap, rho=rec / den, rho_net=rec_net / den, chi=rec / S,
                chi_net=rec_net / S, u_over_s=U / S, d_over_s=s[7] / S, q=f[11] / (den - rec),
                gam=al[12], S=S)


def BU(p, n):
    return (1 - p['sig']) * p['kap'] / (n + p['lam'])


def chistar(p, n):
    g, q = p.get('gam', 0.0), p.get('q', 0.0)
    prop = p['rho'] + (g * q * (1 - p['rho']) / (n + g) if (n + g) > 0 else 0.0)
    return (p['sig'] + p['lam'] * BU(p, n)) * prop


def us_of(s, al=AL, dis=DIS):
    f, _ = flows(s, al, dis); return s[3] / (f[2] + f[13])


def first(path, T, thr):
    i = np.argmax(path >= thr); return T[i] if path[i] >= thr else np.nan


def case(dis, T=np.arange(0, 401, 1.0)):
    A = build_A(AL, dis); b = np.zeros(8); b[0] = 1.0
    ss = np.linalg.solve(-A, b)
    fx = {c: macro_map(ss, c, AL, dis) for c in 'AB'}
    y94 = {c: macro_map(S94, c, AL, dis) for c in 'AB'}
    us94, usf = y94['B']['u_over_s'], fx['B']['u_over_s']
    g_imp = brentq(lambda g: us_of(np.linalg.solve(g * np.eye(8) - A, b), AL, dis) - us94, 1e-6, 0.5)
    ainf = np.linalg.solve(-A, b * E94)
    path = np.array([us_of(ainf + expm(A * t) @ (S94 - ainf), AL, dis) for t in T]) / usf
    return dict(A=A, fixed=fx, y1994=y94, us94=us94, us_fixed=usf, g_implied=g_imp,
                share_2100=path[2100 - 1994], yr90=first(path, T, .9), yr95=first(path, T, .95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wp9', default=os.path.join(_FM, 'analysis', 'wp9_chi_level_gap.csv'))
    ap.add_argument('--out', default=OUT_DEFAULT)
    args = ap.parse_args()

    # ---- validation against the copper chapter's published numbers
    A = build_A(); ev = np.sort(np.linalg.eigvals(A).real)
    N = np.linalg.inv(-A[:7, :7]); tau = N.sum(axis=0)
    M, bv = build_soa(); evs = np.linalg.eigvals(M); dom = evs[np.argmax(np.abs(evs.imag))]
    val = pd.DataFrame([
        ('basic eigenvalues (tab:eigen, dissipative)', '-14.6 -10.9 -9.33 -3.93 -0.0194 -0.0188 -0.0100 -0.0100',
         ' '.join(f'{e:.4g}' for e in ev)),
        ('tau_FD by start stock s1..s7 (tab:expected_times_by_stock)', '57.2 54.3 52.9 52.8 29.2 78.6 100.0',
         ' '.join(f'{t:.1f}' for t in tau)),
        ('upsilon_1 (tab:dynamic_circularity_indicators)', '1.39', f'{AL[8]*N[2,0]:.3f}'),
        ('SOA dominant pair (tab:eigensoa)', '-0.0600 +/- 0.334i', f'{dom.real:.4f} +/- {abs(dom.imag):.4f}i'),
    ], columns=['check', 'published', 'reproduced'])
    x = np.linalg.solve(-M, bv)
    soaB = macro_map(x[1:], 'B')

    # ---- main case (copper chapter as published) and the dissipation sensitivity
    base = case(DIS)
    a18_kappa = AL[9] * (1 - CU_MACRO0['kap']) / CU_MACRO0['kap']
    dis_k = dict(DIS); dis_k[4] = a18_kappa
    alt = case(dis_k)
    fxB, fxA, yB, yA = base['fixed']['B'], base['fixed']['A'], base['y1994']['B'], base['y1994']['A']
    assert abs(soaB['chi'] - fxB['chi']) < 1e-10 and abs(soaB['u_over_s'] - fxB['u_over_s']) < 1e-8
    assert abs(chistar(fxB, 0) - fxB['chi']) < 1e-10 and abs(BU(fxB, 0) - fxB['u_over_s']) < 1e-8

    z = pd.read_csv(args.wp9)
    zn = dict(us=z.ratio_measured.median(), BU=z.ratio_ch1_steady.median(), sat=z.saturation.median(),
              chi=z.chi_measured.median(), chistar=z.chi_at_ch1_steady_ratio.median())

    # ---- table 1: parameter map
    snap = dict(lam=F94[9]/220, sig=F94[3]/(F94[2]+F94[13]), kap=np.nan,
                rho=(F94[3]+F94[13]+F94[14])/(F94[9]+F94[3]),
                chi=(F94[3]+F94[13]+F94[14])/(F94[2]+F94[13]), u_over_s=220/(F94[2]+F94[13]))
    cols = {
        'macro_Cu_calibration_gamma0 (Gloser 2000-10)':
            dict(CU_MACRO0, chi=chistar(CU_MACRO0, N_MACRO), u_over_s=BU(CU_MACRO0, N_MACRO)),
        'Cu_matrix_fixed_point_convB': fxB, 'Cu_matrix_fixed_point_convA': fxA,
        'Cu_matrix_at_1994_stocks_convB': yB, 'Cu_matrix_at_1994_stocks_convA': yA,
        'Graedel_1994_snapshot_data_flows': snap,
    }
    keys = ['lam', 'sig', 'kap', 'rho', 'chi', 'chi_net', 'u_over_s', 'q', 'gam']
    pmap = pd.DataFrame({c: [v.get(k, np.nan) for k in keys] for c, v in cols.items()}, index=keys)
    pmap.index.name = 'parameter'
    pmap.loc['chi', 'macro_Cu_calibration_gamma0 (Gloser 2000-10)'] = chistar(CU_MACRO0, N_MACRO)
    pmap.loc['chi_star_n0', :] = [np.nan, chistar(fxB, 0), chistar(fxA, 0), chistar(yB, 0), chistar(yA, 0), np.nan]
    pmap.loc['chi_star_n001', :] = [chistar(CU_MACRO0, N_MACRO), chistar(fxB, N_MACRO), chistar(fxA, N_MACRO),
                                    chistar(yB, N_MACRO), chistar(yA, N_MACRO), np.nan]
    pmap.loc['BU_n001', :] = [BU(CU_MACRO0, N_MACRO), BU(fxB, N_MACRO), BU(fxA, N_MACRO),
                              BU(yB, N_MACRO), BU(yA, N_MACRO), np.nan]

    # ---- table 2: saturation measures
    sat = pd.DataFrame([
        ('Zn, UDE 1980-2019 (WP-9)', 'u/s vs macro B_U at measured coefficients, n = 0.01', zn['us'], zn['BU']),
        ('Cu, copper matrix, 1994 stocks', 'u/s vs the matrix own fixed point (n = 0); convention-free',
         base['us94'], base['us_fixed']),
        ('Cu, copper matrix, 1994 stocks', 'u/s vs macro B_U at matrix coefficients, n = 0.01, convention B',
         base['us94'], BU(fxB, N_MACRO)),
        ('Cu, copper matrix, 1994 stocks', 'u/s vs macro B_U at matrix coefficients, n = 0.01, convention A',
         base['us94'], BU(fxA, N_MACRO)),
        ('Cu, matrix with kappa = 0.994 (alpha18 matched to Gloser)', 'u/s vs own fixed point (n = 0)',
         alt['us94'], alt['us_fixed']),
        ('Cu, matrix with kappa = 0.994 (alpha18 matched to Gloser)', 'u/s vs macro B_U, n = 0.01, convention B',
         alt['us94'], BU(alt['fixed']['B'], N_MACRO)),
        ('Cu, macro chapter data (Gloser 2000-10)', '2010 U / window-mean S vs macro Cu B_U, n = 0.01 (approximate)',
         CU_MACRO_U / CU_MACRO_S, BU(CU_MACRO0, N_MACRO)),
    ], columns=['cycle', 'measure', 'u_over_s_observed', 'reference_u_over_s'])
    sat['saturation'] = sat.u_over_s_observed / sat.reference_u_over_s
    sat.loc[0, 'saturation'] = zn['sat']          # WP-9 headline: median of per-seed ratios (0.805)

    # ---- table 3: dissipation sensitivity and timing
    sens = []
    for lab, c, a18 in [('copper chapter, alpha18 = 0.01', base, 0.01),
                        ('alpha18 matched to macro kappa_Cu = 0.994', alt, a18_kappa)]:
        fb, yb = c['fixed']['B'], c['y1994']['B']
        sens.append(dict(case=lab, alpha18=a18, kappa=fb['kap'], rho_fixed=fb['rho'], rho_1994=yb['rho'],
                         chi_fixed=fb['chi'], chi_1994=yb['chi'], chistar_1994coefs_n0=chistar(yb, 0),
                         us_fixed=c['us_fixed'], us_1994=c['us94'], saturation_own=c['us94']/c['us_fixed'],
                         saturation_n001_convB=c['us94']/BU(fb, N_MACRO), g_implied_bgp=c['g_implied'],
                         us_share_2100_const_input=c['share_2100'], years_to_90pct=c['yr90'],
                         years_to_95pct=c['yr95'],
                         slowest_eigs=' '.join(f'{e:.4f}' for e in np.sort(np.linalg.eigvals(c['A']).real)[-4:])))
    sens = pd.DataFrame(sens)

    # ---- table 4: one-at-a-time swap, macro Cu -> copper matrix (conv B), chi*(n = 0.01)
    b0 = chistar(CU_MACRO0, N_MACRO); sw = []
    for k in ['lam', 'sig', 'kap', 'rho']:
        q = dict(CU_MACRO0); q[k] = fxB[k]
        sw.append(dict(parameter=k, macro_Cu=CU_MACRO0[k], copper_matrix=fxB[k],
                       chistar_after_swap=chistar(q, N_MACRO), delta=chistar(q, N_MACRO) - b0))
    allq = dict(CU_MACRO0); allq.update({k: fxB[k] for k in ['lam', 'sig', 'kap', 'rho']})
    sw.append(dict(parameter='all four', macro_Cu=np.nan, copper_matrix=np.nan,
                   chistar_after_swap=chistar(allq, N_MACRO), delta=chistar(allq, N_MACRO) - b0))
    swap = pd.DataFrame(sw)

    # chi gap decomposition at 1994 (convention B)
    chi_u = yB['rho'] * (yB['sig'] + yB['lam'] * fxB['u_over_s'])
    gap = fxB['chi'] - yB['chi']
    decomp = dict(chi_1994=yB['chi'], chi_fixed=fxB['chi'], gap=gap, from_u_over_s=chi_u - yB['chi'],
                  from_rho_composition=fxB['chi'] - chi_u)

    for name, df in [('d12_validation', val), ('d12_parameter_map', pmap), ('d12_saturation', sat),
                     ('d12_dissipation_sensitivity', sens), ('d12_chi_swap', swap)]:
        df.to_csv(os.path.join(args.out, name + '.csv'), index=(name == 'd12_parameter_map'))
    pd.DataFrame([decomp]).to_csv(os.path.join(args.out, 'd12_chi_gap_decomposition.csv'), index=False)
    pd.set_option('display.width', 200)
    print(val.to_string(index=False)); print(pmap.round(4).to_string()); print(sat.round(4).to_string(index=False))
    print(sens.round(4).T.to_string()); print(swap.round(4).to_string(index=False))
    print({k: round(v, 4) for k, v in decomp.items()}); print('Zn', {k: round(v, 4) for k, v in zn.items()})
    print('new-scrap recycling share %.4f, old-scrap collection share %.4f' % (
        AL[3] / (AL[3] + AL[10] + DIS[3]), (AL[13] + AL[14]) / (AL[11] + AL[13] + AL[14] + DIS[5])))


if __name__ == '__main__':
    main()
