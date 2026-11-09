#!/usr/bin/env python3
"""
Growth-adjusted circularity chi(g) of Chapter 2 (Macro-Materials mm_growth.chi_g, Eq. ext_chi_g)
at the copper and zinc calibrations, n = 0.01.  Thesis Discussion, Section 6.1.6: copper
circularity falls from 0.484 in the stationary state to 0.335 at a material growth rate of
2 per cent a year and to 0.271 at 4 per cent.

Run from final_model/:  python discussion/chi_growth_check.py > discussion/outputs/chi_growth_check.txt
"""
import os, sys
_FM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MM_DIR = os.environ.get('MACRO_MATERIALS_DIR', os.path.join(os.path.dirname(os.path.dirname(_FM)), 'Macro-Materials'))
sys.path.insert(0, MM_DIR)
import mm_core as mc, mm_growth as mg          # noqa: E402

print('case       n     chi(0)  chi(0.02)  chi(0.04)')
for nm in ('CU_GAMMA', 'ZN_GAMMA', 'CU_BASE', 'ZN_BASE'):
    p = getattr(mc, nm)
    print(f'{nm:9s}  {p.n:.2f}  ' + '  '.join(f'{mg.chi_g(p, g):.3f}' for g in (0.0, 0.02, 0.04)))
