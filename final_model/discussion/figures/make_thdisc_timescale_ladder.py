"""fig:thdisc_ladder (thesis Discussion, Section 6.2): relaxation times, delays and horizons of the four
core models.  Reads discussion/outputs/d5_timescales.csv and d5_delays_periods.csv (d5_timescale_ladder.py).
Run from final_model/:  python discussion/figures/make_thdisc_timescale_ladder.py
"""
import os
_FM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AN = os.path.join(_FM, 'analysis'); OUTS = os.path.join(_FM, 'discussion', 'outputs')
FIGDIR = os.path.join(_FM, 'PhD_Thesis', 'figures', 'discussion'); os.makedirs(FIGDIR, exist_ok=True)
# skill:figure-style kernel.py (auto-injected on skill load)
META_GREY = "#888888"


def apply_figure_style(*, frame="open", font=None, sizes=(8, 7, 6), grid=False):
    import matplotlib as mpl
    if frame not in ("open", "boxed", "none"):
        raise ValueError(f"frame must be 'open'|'boxed'|'none', got {frame!r}")

    try:
        import os, sys, glob, matplotlib.font_manager as fm
        fdir = os.path.join(os.environ.get("CONDA_PREFIX") or sys.prefix, "fonts")
        if os.path.isdir(fdir):
            known = {f.fname for f in fm.fontManager.ttflist}
            for f in glob.glob(os.path.join(fdir, "*.ttf")):
                if f not in known:
                    fm.fontManager.addfont(f)
    except Exception:
        pass
    base, secondary, tick = sizes
    boxed = (frame == "boxed")
    rc = {
        "font.family": "sans-serif",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "legend.fontsize": secondary,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": boxed, "axes.spines.right": boxed,
        "axes.spines.left": frame != "none", "axes.spines.bottom": frame != "none",
        "axes.grid": bool(grid),
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.labelweight": "normal",
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    if font:
        rc["font.sans-serif"] = [font, "DejaVu Sans"]
    mpl.rcParams.update(rc)


def panel_letter(ax, letter, dx=-0.18, dy=1.02, case="lower", fontsize=None):
    import matplotlib.pyplot as plt
    if fontsize is None:
        fontsize = plt.rcParams.get("font.size", 8) + 1
    s = letter.lower() if case == "lower" else letter.upper()
    ax.text(dx, dy, s, transform=ax.transAxes,
            fontweight="bold", fontsize=fontsize, va="bottom", ha="left")


import os, sys, json, numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
apply_figure_style(sizes=(8,7,6))
ts = pd.read_csv(os.path.join(OUTS, 'd5_timescales.csv')); dp = pd.read_csv(os.path.join(OUTS, 'd5_delays_periods.csv'))
COL = {'upstream':'#0072B2','scrap / waste management':'#E69F00','in use':'#009E73','sinks':'#7F7F7F','investment (SOA)':'#CC79A7'}
rowsA = [('Growth theory, copper\n(per capita)', 'Ch2', 'copper'),
         ('Growth theory, zinc\n(per capita)', 'Ch2', 'zinc'),
         ('Structural simulation,\ncopper (fitted)', 'Ch3', None),
         ('Structural analysis,\ncopper (basic)', 'Ch4', 'copper basic'),
         ('Structural analysis,\ncopper (SOA)', 'Ch4', 'copper SOA'),
         ('Estimated cycle, zinc\n(1980–2019)', 'Ch5', None)]
XMAX = 2.5e4
fig, (axa, axb) = plt.subplots(2,1, figsize=(7.0,6.2), sharex=True, gridspec_kw=dict(height_ratios=[6,4], hspace=0.12))
for i,(lab,ch,key) in enumerate(rowsA):
    y = len(rowsA)-1-i
    if ch=='Ch2':
        sub = ts[(ts.chapter=='Ch2') & ts.case.str.startswith(key)]
    elif ch=='Ch4':
        sub = ts[(ts.chapter=='Ch4') & (ts.model==key)]
    else:
        sub = ts[ts.chapter==ch]
    if ch=='Ch5':
        for _,r in sub.iterrows():
            axa.plot([r.ts_lo,r.ts_hi],[y,y],color=COL[r.group],lw=5,alpha=0.35,solid_capstyle='butt',zorder=2)
            axa.plot(r.ts_1980,y,marker='|',ms=9,mew=1.4,color=COL[r.group],zorder=3)
            axa.plot(r.ts_2019,y,marker='o',ms=5,color=COL[r.group],mec='k',mew=0.4,zorder=4)
        continue
    for _,r in sub.drop_duplicates(['timescale_yr','compartment']).iterrows():
        t = r.timescale_yr; jit = 0
        if ch=='Ch2':
            filled = 'calibration' in r.case; jit = 0.13 if filled else -0.13
            axa.plot(t, y+jit, marker='o', ms=5.5, color=COL[r.group] if filled else 'white',
                     mec=COL[r.group], mew=1.3, zorder=3)
            continue
        if t>=1e4:
            axa.plot(XMAX*0.72, y, marker='>', ms=6, color=COL[r.group], mec='k', mew=0.4, zorder=3)
            continue
        axa.plot(t, y, marker='o', ms=5.5, color=COL[r.group], mec='k', mew=0.4, zorder=3, alpha=0.9)
axa.set_yticks(range(len(rowsA))); axa.set_yticklabels([r[0] for r in rowsA][::-1])
axa.set_ylim(-0.6, len(rowsA)-0.4)
axa.annotate('slag, landfill ≈10⁴ yr\n(leaching 10⁻⁴/yr only)', xy=(XMAX*0.72, 3), xytext=(1.6e3, 3.33), fontsize=6, va='center', ha='center')
axa.annotate('damped SOA pair', xy=(16.67, 1), xytext=(3.0, 1.33), fontsize=6, ha='center', va='center',
             arrowprops=dict(arrowstyle='-', lw=0.5, color='0.3'))
handles = [Line2D([],[],marker='o',ls='',color=c,mec='k',mew=0.4,ms=5,label=g) for g,c in COL.items()]
handles += [Line2D([],[],marker='o',ls='',color='0.4',ms=5,label='calibration'),
            Line2D([],[],marker='o',ls='',color='white',mec='0.4',mew=1.2,ms=5,label='γ = 0 comparison'),
            Line2D([],[],marker='o',ls='',color='0.6',mec='k',mew=0.4,ms=5,label='zinc: 2019 (| = 1980; band = seed IQR, all years)')]
leg = fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.16, 0.035), ncol=4, frameon=False, fontsize=6, handletextpad=0.3, columnspacing=1.0)
# --- panel b
rowsB = ['Mining lead time τ_E,\nsimulation (copper)', 'Recycling lead time τ_R,\nsimulation (copper)',
         'Cycle periods', 'Approach to 90 / 95 %\nof the frozen target']
yb = {k: len(rowsB)-1-i for i,k in enumerate(rowsB)}
y=yb[rowsB[0]]; axb.plot([2,32],[y,y],color='#CC79A7',lw=3,alpha=0.45,solid_capstyle='butt'); axb.plot(8,y,'D',color='#CC79A7',mec='k',mew=0.4,ms=5)
axb.text(33,y,'  swept 2–32 yr; fitted 8 yr',fontsize=6,va='center')
y=yb[rowsB[1]]; axb.plot(2,y,'D',color='#E69F00',mec='k',mew=0.4,ms=5)
y=yb['Cycle periods']
per = [(20.6,'s','#CC79A7',0.30,'copper price, simulation: 20.6 yr'),(23.2,'s','white',0.10,'copper price, data: 23.2 yr'),
       (18.8,'^','#CC79A7',-0.10,'copper SOA: 18.8 yr'),(29/3,'v','white',-0.30,'zinc refined stocks, data: ≈10 yr')]
for v,m,c,dy,l in per:
    axb.plot(v,y+dy,marker=m,ls='',ms=5,color=c,mec='#CC79A7' if c=='white' else 'k',mew=1.1 if c=='white' else 0.4,zorder=3)
    axb.text(34 if v>15 else v*0.77, y+dy, l, fontsize=6, va='center', ha='left' if v>15 else 'right')
axb.text(0.0032, y+0.2, 'estimated zinc cycle: no complex mode', fontsize=6, va='center', color='0.35')
y=yb[rowsB[3]]
for (a,b_,c,dy,l) in [(70,105,'#009E73',0.13,'copper u/s from 1994'),(72.5,107.8,'#009E73',-0.13,'zinc in use from 2019')]:
    axb.plot([a,b_],[y+dy,y+dy],color=c,lw=1.2); axb.plot(a,y+dy,'o',color=c,mec='k',mew=0.4,ms=4.5); axb.plot(b_,y+dy,'o',color='white',mec=c,mew=1.2,ms=4.5)
    axb.text(b_*1.25,y+dy,l,fontsize=6,va='center')
axb.set_yticks(list(yb.values())); axb.set_yticklabels(list(yb.keys()))
axb.set_ylim(-0.6,len(rowsB)-0.4)
axb.set_xscale('log'); axb.set_xlim(2.5e-3, XMAX)
axb.set_xlabel('Time (years, log scale)')
axb.set_xticks([1e-2,1e-1,1,10,100,1e3,1e4]); axb.set_xticklabels(['0.01','0.1','1','10','100','1,000','10,000'])
sec = axa.secondary_xaxis('top'); sec.set_xticks([1/365.25,7/365.25,1/12,1,10,100,1000]); sec.set_xticklabels(['day','week','month','year','decade','century','millennium'])
sec.tick_params(labelsize=6)
for ax in (axa,axb): ax.grid(axis='x', color='0.9', lw=0.5, zorder=0); ax.set_axisbelow(True)
panel_letter(axa,'a'); panel_letter(axb,'b')
fig.savefig(os.path.join(FIGDIR, 'thdisc_timescale_ladder.pdf'), bbox_inches='tight')
