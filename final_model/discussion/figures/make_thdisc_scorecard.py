"""fig:thdisc_scorecard (thesis Discussion, Section 6.1): the eleven Chapter 2 hypotheses scored on the
estimated zinc cycle.  Reads the published WP-9 tables (analysis/) and the D3 re-score at the current
Chapter 2 calibration (discussion/outputs/d3_calibration/).  Run from final_model/:
    python discussion/figures/make_thdisc_scorecard.py
"""
import os
_FM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AN = os.path.join(_FM, 'analysis'); OUTS = os.path.join(_FM, 'discussion', 'outputs')
FIGDIR = os.path.join(_FM, 'PhD_Thesis', 'figures', 'discussion'); os.makedirs(FIGDIR, exist_ok=True)
import pandas as pd, numpy as np, re, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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


apply_figure_style(sizes=(8,7,6))
ht=pd.read_csv(os.path.join(AN, 'wp9_hypothesis_table.csv')).set_index('id')
cpv=pd.read_csv(os.path.join(AN, 'wp9_cp_h2_verdict.csv')).iloc[0]
mA=pd.read_csv(os.path.join(AN, 'wp9_multiplier.csv')).set_index('perturbation').loc['level']
mB=pd.read_csv(os.path.join(OUTS, 'd3_calibration', 'wp9_multiplier.csv')).set_index('perturbation').loc['level']
iu=pd.read_csv(os.path.join(AN, 'wp9_intensity_of_use.csv')).iloc[0]
sr=pd.read_csv(os.path.join(OUTS, 'd3_calibration', 'wp9_s_rho.csv')).iloc[0]
net=[float(x) for x in re.search(r'net (-?[\d.]+) \[(-?[\d.]+), (-?[\d.]+)\]',cpv['measured']).groups()]
CLS={'Confirmed':'#0072B2','Sign right, size wrong':'#56B4E9','Refined':'#E69F00','Contradicted':'#D55E00','Extension':'#7F7F7F'}
def g(i,pat): return re.search(pat, ht.loc[i,'measured']).group(1)
rows_a=[
 ('H3','Recovery loop collapses into one multiplier','Confirmed', f"slope {mB.slope_hl:.2f}, $R^2$ {mB.r2_median:.3f}"),
 ('H4','In-use and discard respond with opposite sign','Confirmed', 'Spearman '+g('H4',r'Spearman (-?[\d.]+)').replace('-','−')),
 ('H5','General-equilibrium amplification negligible','Confirmed', 'amplification '+g('H5',r'^([\d.]+)')),
 ('H11','Short-run closed form for $\\partial\\ln s/\\partial\\rho$','Confirmed', f"{sr.hl:.2f} vs {sr.ch1_short_run:.2f}"),
 ('H2','Faster growth dilutes recovery','Sign right, size wrong', f"net {net[0]:.2f}".replace('-','−')),
 ('H7','Intensity of use constant','Refined', f"elasticity {iu.hl:.2f}"),
 ('H1','Coefficients invariant to the economy','Contradicted', g('H1',r'^([\d.]+) of Ch1')+' of cells respond'),
 ('H6','Energy rations throughput proportionally','Contradicted', '$\\partial\\chi$/energy +'+g('H6',r'^\+([\d.]+)')[:4]+' per SD'),
 ('H8','Balanced growth path','Contradicted', 'elasticities '+'/'.join(re.search(r'throughput ([\d.]+), in-use ([\d.]+), discard ([\d.]+)',ht.loc['H8','measured']).groups())),
 ('H9','Substitutability $\\pi$ unidentifiable from tonnage','Extension', 'confirmed'),
 ('H10','Recovery on saturated Hill branch','Extension', 'all years; min $m/m_{\\mathrm{infl}}$ '+g('H10',r'min m/m_infl = ([\d.]+)')),
]

h9=re.search(r'moves the H3 slope by ([\d.]+) .*interval ([\d.]+) wide',ht.loc['H9','measured']).groups()
rows_a[9]=('H9','Substitutability $\\pi$ unidentifiable from tonnage','Extension', f"$\\pi$ range {float(h9[0]):.3f} < 95% interval {float(h9[1]):.3f}")


plt.close('all')
fig=plt.figure(figsize=(6.5,2.55))
ax=fig.add_axes([0.01,0.02,0.98,0.96]); ax.set_xlim(0,1); ax.set_ylim(len(rows_a)-0.5,-0.9); ax.axis('off')
xs=(0.01,0.075,0.56,0.79)
for h,x in zip(('','Hypothesis of the growth theory','Outcome','Measured'),xs):
    ax.text(x,-0.85,h,fontsize=7,fontweight='bold',va='center')
prev=None
for k,(i,txt,cls,meas) in enumerate(rows_a):
    if prev is not None and cls!=prev: ax.axhline(k-0.5,xmin=0.01,color='0.85',lw=0.6)
    prev=cls
    ax.text(xs[0],k,i,fontsize=7,va='center',fontweight='bold')
    ax.text(xs[1],k,txt,fontsize=7,va='center')
    ax.add_patch(mpl.patches.FancyBboxPatch((xs[2],k-0.32),0.215,0.64,boxstyle='round,pad=0,rounding_size=0.01',fc=CLS[cls],ec='none'))
    ax.text(xs[2]+0.1075,k,cls,fontsize=6,va='center',ha='center',color='white' if cls not in('Sign right, size wrong','Refined') else 'black')
    ax.text(xs[3],k,meas,fontsize=6,va='center')
fig.savefig(os.path.join(FIGDIR, 'thdisc_scorecard.pdf'))
