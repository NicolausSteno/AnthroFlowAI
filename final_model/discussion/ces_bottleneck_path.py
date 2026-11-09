"""Second CES check: growth rate of capital along the path (t = 100 ... 1400) for the cases of
ces_bottleneck.py, including finite-time divergence under Cobb-Douglas with alpha + theta*nu >= 1.
Run from final_model/:  python discussion/ces_bottleneck_path.py > discussion/outputs/ces_bottleneck_path.txt
"""
import sys,os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.integrate import solve_ivp
import ces_bottleneck as cb
th=cb.th
def run(alpha,eps,gnet,gM,T=1500,cap=600):
    ev=lambda t,X,*a: cap-X[0]; ev.terminal=True
    sol=solve_ivp(cb.rhs,(0,T),cb.X0,args=(alpha,eps,gnet,gM),events=ev,rtol=1e-9,atol=1e-11,dense_output=True,method="LSODA")
    Tend=sol.t[-1]
    g=lambda t: (sol.sol(t+1)[0]-sol.sol(t-1)[0])/2
    rows=[(t,g(t)) for t in [100,300,600,1000,1400] if t<Tend-2]
    return sol.status==1,Tend,rows
print("alpha eps gnet gM | pred | finite-time blow-up? | g_k at t=100,300,600,1000,1400")
for c in [(1.0,1.0,0.01,0.0),(1.0,0.5,0.01,0.0),(1.0,0.5,0.0,0.0),(1.0,0.5,0.01,0.005),(1.05,0.5,0.01,0.0),(1.0,0.8,0.01,0.0),(0.3,1.0,0.01,0.0),(0.3,0.5,0.01,0.0)]:
    bl,Te,rows=run(*c)
    print(c,f"| {(c[3]+c[2])/(1-th):.4f} |",f"yes, t={Te:.0f}" if bl else "no"," | "+", ".join(f"{g:.4f}" for t,g in rows))
