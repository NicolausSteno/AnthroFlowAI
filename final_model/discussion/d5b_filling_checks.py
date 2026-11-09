#!/usr/bin/env python3
"""
D5b -- two supplementary checks for Discussion Section sec:thdisc_filling (analytic, no fitting).

(1) Saturation as the steady-state lag of a growing input.  For each seed, freeze the zinc UDE
    matrix A(t) at 1980 or 2019; let concentrate input grow as b*exp(g t).  The balanced-growth
    solution is S_g = (gI - A)^{-1} b, so the in-use saturation of a cycle growing at g is
    sat(g) = sum_iu S_g / sum_iu S*, with S* = -A^{-1} b.  Invert sat(g) = observed saturation
    (Chapter 5: 0.525 in 1980, 0.554 in 2019) for the implied g.  Compare with the OLS trend of
    concentrate input b(t) over 1981-2019.
(2) MaTrace-commensurable lifetime (Charpentier Poncelet et al. 2022: tau = sum_t msr(t), the
    expected cumulative time in use of one extracted kg, all cycles, stationary structure).
    Thesis counterpart: in-use entry of the fundamental-matrix column of the entry stock.
    Copper (Ch 4 basic matrix, 1994 coefficients; landfill absorbing): N[s4, s1] with
    N = (-A_FD)^{-1}; check = upsilon_1 / (alpha9 + alpha18).  Zinc (Ch 5): sum over the three
    in-use cohorts of N[:, concentrate]; by Little's law this equals S*_iu / b.

Inputs: final_model/analysis/wp2a_A_of_t.npz (the unmasked file: the concentrate input b is
an observed series, so the published masked copy must first be restored with
reproducibility/restore_observed.py); d12_copper_ratios.py (build_A, AL, DIS).
Run from final_model/:  python discussion/d5b_filling_checks.py > discussion/outputs/d5b_output.txt
"""
import argparse, os, sys, importlib.util, numpy as np
from scipy.optimize import brentq
_HERE = os.path.dirname(os.path.abspath(__file__)); _FM = os.path.dirname(_HERE)
ap = argparse.ArgumentParser()
ap.add_argument('--d12', default=os.path.join(_HERE, 'd12_copper_ratios.py'))
ap.add_argument('--npz', default=os.path.join(_FM, 'analysis', 'wp2a_A_of_t.npz'))
_a = ap.parse_args(); D12, NPZ = _a.d12, _a.npz
spec = importlib.util.spec_from_file_location('d12', D12); d12 = importlib.util.module_from_spec(spec); spec.loader.exec_module(d12)

A = d12.build_A(); idx = list(range(7)); N = np.linalg.inv(-A[np.ix_(idx, idx)])
print(f"copper tau_FD(s1) {N[:,0].sum():.2f}  upsilon1 {d12.AL[8]*N[2,0]:.3f}  time-in-use(s1) {N[3,0]:.2f}")
dis = dict(d12.DIS); dis[4] = 0.02*0.006/0.994
N2 = np.linalg.inv(-d12.build_A(dis=dis)[np.ix_(idx, idx)])
print(f"copper, Gloser-consistent in-use dissipation: time-in-use {N2[3,0]:.1f}")

z = np.load(NPZ, allow_pickle=True); Az, b, yrs = z['A'], z['b'], z['years']; ns = Az.shape[0]
T = np.array([[np.linalg.inv(-Az[s,t])[2:5,0].sum() for t in range(40)] for s in range(ns)])
for y in (1980, 2019):
    i = list(yrs).index(y); print(f"zinc time-in-use per kg concentrate {y}: median {np.median(T[:,i]):.2f} IQR {np.percentile(T[:,i],25):.2f}-{np.percentile(T[:,i],75):.2f}")
little = np.median([np.linalg.solve(-Az[s,-1], b[s,-1])[2:5].sum()/b[s,-1,0] for s in range(ns)])
print(f"Little's law check 2019 S*_iu/b = {little:.2f}")

def sat(Am, bv, g): return np.linalg.solve(g*np.eye(6)-Am, bv)[2:5].sum()/np.linalg.solve(-Am, bv)[2:5].sum()
for g in (0.015, 0.02, 0.025, 0.03):
    print(f"g={g:.3f}: sat 1980-A {np.median([sat(Az[s,0],b[s,0],g) for s in range(ns)]):.3f}  2019-A {np.median([sat(Az[s,-1],b[s,-1],g) for s in range(ns)]):.3f}")
for t, obs in ((0, 0.525), (39, 0.554)):
    gs = [brentq(lambda g: sat(Az[s,t],b[s,t],g)-obs, 1e-5, 0.2) for s in range(ns)]
    print(f"{int(yrs[t])}: implied g {np.median(gs)*100:.2f} %/yr IQR {np.percentile(gs,25)*100:.2f}-{np.percentile(gs,75)*100:.2f}")
w = yrs >= 1981
print(f"concentrate input trend 1981-2019: {np.median([np.polyfit(yrs[w], np.log(b[s,w,0]), 1)[0] for s in range(ns)])*100:.2f} %/yr")
