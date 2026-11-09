"""fig:thdisc_price_cycles (thesis Discussion, Section 6.4): the secondary share of supply and the copper
price cycle in the fitted Chapter 3 model.  Reads discussion/outputs/d16_*.csv (d16_cycles.R, d16_summarise.R).
Run from final_model/:  python discussion/figures/make_thdisc_price_cycles.py
"""
import os
_FM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AN = os.path.join(_FM, 'analysis'); OUTS = os.path.join(_FM, 'discussion', 'outputs')
FIGDIR = os.path.join(_FM, 'PhD_Thesis', 'figures', 'discussion'); os.makedirs(FIGDIR, exist_ok=True)
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def panel_crops(fig, dpi=None, pad_px=6, bbox_inches=None, pad_inches=None):
    import matplotlib as mpl
    import matplotlib.text
    if dpi is None:
        dpi = mpl.rcParams.get("savefig.dpi", fig.dpi)
        if dpi == "figure":
            dpi = fig.dpi
    dpi = float(dpi)
    if bbox_inches is None:
        bbox_inches = mpl.rcParams.get("savefig.bbox")
    fig.canvas.draw()
    r = fig.canvas.get_renderer()

    if bbox_inches == "tight":
        if pad_inches is None:
            pad_inches = mpl.rcParams.get("savefig.pad_inches", 0.1)
        tb = fig.get_tightbbox(r).padded(pad_inches)
        ox_in, oy_in = tb.x0, tb.y0
        W_in, H_in = tb.width, tb.height
    elif isinstance(bbox_inches, mpl.transforms.BboxBase):
        ox_in, oy_in = bbox_inches.x0, bbox_inches.y0
        W_in, H_in = bbox_inches.width, bbox_inches.height
    else:
        ox_in, oy_in = 0.0, 0.0
        W_in, H_in = fig.get_size_inches()
    W_px, H_px = int(round(W_in * dpi)), int(round(H_in * dpi))
    lettered = {}
    for ax in fig.axes:
        for t in ax.findobj(matplotlib.text.Text):
            s = (t.get_text() or "").strip()
            if len(s) == 1 and s.isalpha() and t.get_fontweight() in ("bold", 700):
                lettered[ax] = s
                break

    if not lettered:
        lettered = {ax: str(i) for i, ax in enumerate(fig.axes)}
    out = {}
    for ax, letter in lettered.items():
        bbs = [ax.get_tightbbox(r)]
        ss = ax.get_subplotspec()
        for sib in fig.axes:
            if sib is ax or sib in lettered:
                continue
            ssib = sib.get_subplotspec()
            same_row = ss is None or ssib is None or ss.rowspan == ssib.rowspan
            same_col = ss is None or ssib is None or ss.colspan == ssib.colspan
            if ((ax.get_shared_y_axes().joined(ax, sib) and same_row)
                    or (ax.get_shared_x_axes().joined(ax, sib) and same_col)):
                bbs.append(sib.get_tightbbox(r))
        bb = mpl.transforms.Bbox.union(bbs)

        bx0 = (bb.x0 / fig.dpi - ox_in) * dpi
        bx1 = (bb.x1 / fig.dpi - ox_in) * dpi
        by0 = H_px - (bb.y1 / fig.dpi - oy_in) * dpi
        by1 = H_px - (bb.y0 / fig.dpi - oy_in) * dpi
        out[letter] = (
            max(int(bx0) - pad_px, 0),
            max(int(by0) - pad_px, 0),
            min(int(bx1) + pad_px, W_px),
            min(int(by1) + pad_px, H_px),
        )
    return out


# Load data
s = pd.read_csv(os.path.join(OUTS, 'd16_sweep.csv'))
dd = pd.read_csv(os.path.join(OUTS, 'd16_direction_by_delay.csv'))
band = pd.read_csv(os.path.join(OUTS, 'd16_band_historical.csv'))

M = s[s.set == 'main']

apply_figure_style(sizes=(8, 7, 6))
mpl.rcParams['mathtext.fontset'] = 'dejavusans'
VER = '#D55E00'; BLU = '#0072B2'; PUR = '#CC79A7'; GRY = '#4D4D4D'

fig = plt.figure(figsize=(6.9, 5.7))
gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.42, left=0.09, right=0.97, top=0.97, bottom=0.09)
axa = fig.add_subplot(gs[0, 0]); axb = fig.add_subplot(gs[0, 1]); axc = fig.add_subplot(gs[1, 0]); axd = fig.add_subplot(gs[1, 1])

# a heatmap
tE = [4, 8, 12, 16]; tR = [1, 2, 4, 8]
Z = np.full((4, 4), np.nan)
for _, r in dd.iterrows(): Z[tE.index(r.tau_E), tR.index(r.tau_R)] = r.slope_lnamp
lim = 7.5; norm = mpl.colors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
im = axa.imshow(Z, origin='lower', cmap='RdBu', norm=norm, aspect='auto')
for i in range(4):
    for j in range(4):
        v = Z[i, j]
        if np.isnan(v): axa.text(j, i, 'n.d.', ha='center', va='center', fontsize=6, color='0.45')
        else: axa.text(j, i, f'{v:+.1f}', ha='center', va='center', fontsize=6, color='white' if abs(v) > 4 else 'black')
axa.add_patch(mpl.patches.Rectangle((0.5, 0.5), 1, 1, fill=False, lw=1.4, ec='black'))
axa.set_xticks(range(4), [str(x) for x in tR]); axa.set_yticks(range(4), [str(x) for x in tE])
axa.set_xlabel(r'recycling lead time $\tau_R$ (yr)'); axa.set_ylabel(r'mining lead time $\tau_E$ (yr)')
cb = fig.colorbar(im, ax=axa, fraction=0.05, pad=0.03); cb.set_label('slope of ln amplitude on share\n(negative = damping)', fontsize=7); cb.ax.tick_params(labelsize=6)

# b,d series
fit = M[(M.tau_E == 8) & (M.tau_R == 2) & M.solved]
ser = [('fitted model', fit, VER, '-', 'o', 1.8),
       (r'scrap limiter on ($n=0.5$)', s[(s.set == 'n=0.5')], GRY, '--', 's', 1.1),
       (r'price-elastic demand ($\varepsilon_P=0.2$)', s[(s.set == 'eps_P=0.2') & s.solved], BLU, '-.', '^', 1.1),
       ('equal response gains', s[(s.set == 'k_R') & (s.k_R.round(3) == 0.353) & s.solved], PUR, ':', 'D', 1.1)]
for ax, col, yl, scale in [(axb, 'amp_long', 'cycle amplitude,\nsd of band-passed ln price', 1), (axd, 'sd_dlnE', 'extraction volatility,\nsd of annual change in ln E (%)', 100)]:
    ax.axvspan(0.275, 0.42, color='0.92', zorder=0)
    ax.axvline(0.1876, color='0.55', lw=0.8, zorder=1)
    for lab, d, c, ls, mk, lw in ser:
        ax.plot(d.sigma0, d[col] * scale, color=c, ls=ls, marker=mk, ms=3.2, lw=lw, label=lab, zorder=3 if c == VER else 2)
    ax.set_xlim(0.03, 0.42); ax.set_xticks([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
    ax.set_xlabel('secondary share of supply in 1961\n(total supply fixed)'); ax.set_ylabel(yl)
    ax.tick_params(axis='x', labelsize=6)
axb.text(0.19, 0.86, 'observed 1961', fontsize=6, color='0.4', ha='left', transform=axb.get_xaxis_transform())
axb.text(0.282, 0.86, 'fitted model:\nscrap exhausted', fontsize=6, color='0.4', ha='left', va='center', transform=axb.get_xaxis_transform())
axb.legend(loc='center right', bbox_to_anchor=(1.0, 0.52), fontsize=6, frameon=False, handlelength=2.6)
axb.set_ylim(0, 0.95)

# c
shades = {0.05: ('#F2B48C', 'share 0.05', 1.2), 0.1876: (VER, 'share 0.19 (observed)', 1.9), 0.25: ('#7A2E00', 'share 0.25', 1.2)}
for sg, (c, lab, lw) in shades.items():
    d = band[np.isclose(band.sigma0, sg)]
    axc.plot(d.year, d.band, color=c, lw=lw, label=lab)
axc.axhline(0, color='0.7', lw=0.6)
axc.set_xlabel('year'); axc.set_ylabel('band-passed ln price (15–56 yr)')
axc.legend(loc='upper right', fontsize=6, frameon=False)

for ax, l in zip([axa, axb, axc, axd], 'abcd'): panel_letter(ax, l)

fig.savefig(os.path.join(FIGDIR, 'thdisc_price_cycles.pdf'))

r_ = fig.canvas.get_renderer()
texts = [(t, t.get_window_extent(r_)) for t in fig.findobj(mpl.text.Text) if t.get_text().strip() and t.get_visible()]
tl = set()
for ax in fig.axes: tl |= set(ax.get_xticklabels() + ax.get_yticklabels())

for t in axb.texts[:]:
    if t.get_text().startswith('observed'): t.set_ha('right'); t.set_x(0.183)
cb.ax.yaxis.labelpad = 8
for ax in (axb, axd): ax.xaxis.labelpad = 6
axc.xaxis.labelpad = 4; axa.xaxis.labelpad = 4

pa = axa.get_position(); axa.set_position([pa.x0, pa.y0, pa.width * 0.90, pa.height])
pc = cb.ax.get_position(); cb.ax.set_position([pa.x0 + pa.width * 0.90 + 0.012, pc.y0, pc.width, pc.height])
cb.set_label('slope of ln amplitude\non share (< 0 damps)', fontsize=7); cb.ax.yaxis.labelpad = 4
for ax in (axb, axd):
    p = ax.get_position(); ax.set_position([p.x0 + 0.025, p.y0, p.width - 0.025, p.height])

axb.get_legend().remove()
axb.legend(loc='upper right', bbox_to_anchor=(1.0, 0.79), fontsize=6, frameon=False, handlelength=2.4)
fig.savefig(os.path.join(FIGDIR, 'thdisc_price_cycles.pdf'))
