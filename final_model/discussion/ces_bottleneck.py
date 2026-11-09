"""Numerical check of the CES bottleneck growth rate in the Ch2 capital-material block
(thesis Discussion, Section 6.1.6, Eqs. thdisc_ces and thdisc_bottleneck).
Run from final_model/:  python discussion/ces_bottleneck.py > discussion/outputs/ces_bottleneck.txt
y = [(1-nu) z^r + nu (A_M x)^r]^(1/r), z = (A_Y k_Y^alpha)^(1/(1-nu)), r=(eps-1)/eps
p = B (b_P k)^theta * exp((gP-eta*delta) t); s=(p+rho*lam*u+gam*d)/(1-rho*sig)
Prediction (eps<1, z outgrowing x): g -> (gM + gP - eta*delta)/(1-theta), independent of alpha, iota, chi."""
import numpy as np
from scipy.integrate import solve_ivp
cu=dict(lam=0.0262,sig=0.170,rho=0.588,gam=0.002,kap=0.994,q=0.630); n=0.01
dK, iota, nu, th, bP, AY, B = 0.05, 0.20, 0.01, 0.6, 0.05, 1.0, 1.0
def rhs(t,X,alpha,eps,gnet,gM):
    lk,lu,ld=X; k,u,d=np.exp(X); c=cu
    p=B*(bP*k)**th*np.exp(gnet*t)
    s=(p+c['rho']*c['lam']*u+c['gam']*d)/(1-c['rho']*c['sig']); x=(1-c['sig'])*s*np.exp(gM*t)
    kY=(1-bP)*k
    if eps==1.0: y=AY*kY**alpha*x**nu
    else:
        r=(eps-1)/eps; lz=(np.log(AY)+alpha*np.log(kY))/(1-nu); lx=np.log(x)
        m=max(r*lz,r*lx); y=np.exp((m+np.log((1-nu)*np.exp(r*lz-m)+nu*np.exp(r*lx-m)))/r)
    s_=s
    return [iota*y/k-(n+dK), ((1-c['sig'])*c['kap']*s_-(n+c['lam'])*u)/u,
            (c['q']*(1-c['rho'])*(c['lam']*u+c['sig']*s_)-(n+c['gam'])*d)/d]
def blow(t,X,*a): return 60-X[0]
blow.terminal=True
X0=np.log([5.0,50.7,19.3])
if __name__=="__main__":
    print("case | alpha | eps | gP-eta*delta | gM | outcome | late g_k | late g_s | predicted")
    for alpha,eps,gnet,gM in [(1.0,1.0,0.01,0.0),(1.0,0.5,0.01,0.0),(1.0,0.5,0.0,0.0),(1.0,0.5,0.01,0.005),(1.05,0.5,0.01,0.0),(0.3,0.5,0.01,0.0),(1.0,0.8,0.01,0.0)]:
        sol=solve_ivp(rhs,(0,2500),X0,args=(alpha,eps,gnet,gM),events=blow,rtol=1e-9,atol=1e-11,dense_output=True,method="LSODA")
        pred=(gM+gnet)/(1-th)
        if sol.status==1: print(f"| {alpha} | {eps} | {gnet} | {gM} | ln k>60 at t={sol.t_events[0][0]:.0f} (explosive) | - | - | {pred:.4f}"); continue
        T=sol.t[-1]; a,b=T-200,T
        gk=(sol.sol(b)[0]-sol.sol(a)[0])/200
        # g_s from s
        def ls(t):
            lk,lu,ld=sol.sol(t); k,u,d=np.exp([lk,lu,ld]); c=cu
            return np.log((B*(bP*k)**th*np.exp(gnet*t)+c['rho']*c['lam']*u+c['gam']*d)/(1-c['rho']*c['sig']))
        gs=(ls(b)-ls(a))/200
        print(f"| {alpha} | {eps} | {gnet} | {gM} | bounded growth | {gk:.4f} | {gs:.4f} | {pred:.4f}")
