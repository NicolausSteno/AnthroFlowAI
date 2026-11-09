#!/usr/bin/env python3
"""
D3 -- the eleven Chapter 2 hypotheses (WP-9) scored against the CURRENT Chapter 2 zinc
calibration.  Thesis Discussion, Section 6.1 (sec:thdisc_growth) and fig:thdisc_scorecard.

WP-9 (run_wp9.py, published with Chapter 5) was registered on 24 Aug 2026 against the
Chapter 2 zinc case without urban mining (gamma = 0, rho_0 = 0.363).  The Chapter 2
calibration used in the thesis (gamma = 0.003, rho = 0.345; s* 3.83, u* 65.06, d* 47.24,
chi* 0.319) came later.  This script re-runs WP-9 unchanged except for four constants in
zinc_ch1_lab.py, which make the calibration the main case and gamma = 0 the comparison
case.  The UDE side (cached per-seed Jacobians in analysis/wp9/) is untouched and nothing
is refitted.  All eleven verdicts are identical under the two cases; only the reference
values move (see outputs/d3_calibration_deltas.csv).

The published package (zinc_ch1_lab.py, run_wp9.py, analysis/wp9_*.csv) is NOT modified:
the re-run happens in a scratch copy of final_model/ and its outputs are written to
discussion/outputs/d3_calibration/.

Requires the per-seed caches analysis/wp9/ch1_seed*.npz, which run_wp9.py builds from the
anchor_v4 fits and the observed record (zinc_dataset.xlsx, not redistributable; see the
repository README).

Run from final_model/:
    python discussion/d3_wp9_calibration.py
"""
import glob, json, os, re, shutil, subprocess, sys, tempfile

import numpy as np
import pandas as pd

FM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(FM, "discussion", "outputs")
OUT_B = os.path.join(OUT, "d3_calibration")

PATCH = [
    ("CH1_ZINC = dict(lam=0.0375, sig=0.179, rho=0.363, gam=0.0,",
     "CH1_ZINC = dict(lam=0.0375, sig=0.179, rho=0.345, gam=0.003,"),
    ("CH1_ZINC_GAMMA_SENS = dict(CH1_ZINC, gam=0.003, rho=0.307)",
     "CH1_ZINC_GAMMA_SENS = dict(CH1_ZINC, gam=0.0, rho=0.363)   # D3: no-urban-mining comparison"),
    ("CH1_ZINC_STEADY = dict(s_star=2.89, u_star=49.06, d_star=45.03, chi_star=0.296)",
     "CH1_ZINC_STEADY = dict(s_star=3.83, u_star=65.06, d_star=47.24, chi_star=0.319)"),
    ("CH1_ZINC_STEADY_GAMMA_SENS = dict(s_star=2.69, u_star=45.76, d_star=35.15,\n"
     "                                  chi_star=0.290)",
     "CH1_ZINC_STEADY_GAMMA_SENS = dict(s_star=2.89, u_star=49.06, d_star=45.03,\n"
     "                                  chi_star=0.296)"),
]

LEVELGAP = r'''
import numpy as np, pandas as pd, run_wp9 as W, zinc_ch1_lab as CH1
tab = W.seed_table(CH1.seed_list())
off = 0.363 - 0.345           # Chapter 2's offset between observed rho_0 and calibrated rho
rows = []
for r in tab:
    p = r["par"]; m = {k: float(np.mean(p[k])) for k in ("lam", "sig", "kap", "rho", "chi")}
    base = dict(CH1.CH1_ZINC); base.update(lam=m["lam"], sig=m["sig"], kap=m["kap"])
    v0 = dict(base, gam=0.0, rho=m["rho"]); v1 = dict(base, gam=0.003, rho=m["rho"] - off)
    vdc = dict(base, gam=0.003, rho=m["rho"])
    rows.append(dict(seed=r["seed"], chi_measured=m["chi"], rho_measured=m["rho"],
        chi_star_at_measured_gamma0=CH1.ch1_chi(v0),
        chi_star_at_measured_gamma003_offset=CH1.ch1_chi(v1),
        chi_star_at_measured_gamma003_doublecount=CH1.ch1_chi(vdc)))
pd.DataFrame(rows).to_csv("analysis/d3_chi_level_gap_variants.csv", index=False)
'''


def scratch_copy(tmp):
    for f in glob.glob(os.path.join(FM, "*.py")):
        shutil.copy(f, tmp)
    for d in ("anchor_v4", "anchor_gam"):
        if os.path.isdir(os.path.join(FM, d)):
            os.symlink(os.path.join(FM, d), os.path.join(tmp, d))
    for f in ("zinc_dataset.xlsx",):
        if os.path.exists(os.path.join(FM, f)):
            os.symlink(os.path.join(FM, f), os.path.join(tmp, f))
    # analysis/: link every published input read-only, but COPY the WP-9 cache and never link a
    # wp9* name, so that the re-run cannot write through a link into the published files
    an, tan = os.path.join(FM, "analysis"), os.path.join(tmp, "analysis")
    os.makedirs(tan)
    for name in os.listdir(an):
        if not name.startswith("wp9"):
            os.symlink(os.path.join(an, name), os.path.join(tan, name))
    shutil.copytree(os.path.join(an, "wp9"), os.path.join(tan, "wp9"))
    p = os.path.join(tmp, "zinc_ch1_lab.py"); s = open(p).read()
    for old, new in PATCH:
        assert s.count(old) == 1, old
        s = s.replace(old, new)
    open(p, "w").write(s)


def run(tmp, args):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg", PYTHONPATH=tmp)
    r = subprocess.run([sys.executable] + args, cwd=tmp, env=env, capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stdout[-3000:] + r.stderr[-3000:])


def deltas():
    A, B = os.path.join(FM, "analysis"), OUT_B
    pA = pd.read_csv(f"{A}/wp9_ch1_parameters.csv").set_index("parameter")
    pB = pd.read_csv(f"{B}/wp9_ch1_parameters.csv").set_index("parameter")
    dA = pd.read_csv(f"{A}/wp9_dilution_pooled.csv"); dB = pd.read_csv(f"{B}/wp9_dilution_pooled.csv")
    cpv = pd.read_csv(f"{A}/wp9_cp_h2_verdict.csv").iloc[0]
    sA = pd.read_csv(f"{A}/wp9_s_rho.csv").iloc[0]; sB = pd.read_csv(f"{B}/wp9_s_rho.csv").iloc[0]
    mA = pd.read_csv(f"{A}/wp9_multiplier.csv").set_index("perturbation").loc["level"]
    mB = pd.read_csv(f"{B}/wp9_multiplier.csv").set_index("perturbation").loc["level"]
    jA = json.load(open(f"{A}/wp9_check.json")); jB = json.load(open(f"{B}/wp9_check.json"))
    net = [float(x) for x in re.search(r"net (-?[\d.]+) \[(-?[\d.]+), (-?[\d.]+)\]", cpv["measured"]).groups()]
    lg = pd.read_csv(f"{B}/d3_chi_level_gap_variants.csv").median()
    rows = []
    def add(q, a, b, meas, note):
        rows.append(dict(quantity=q, published_gamma0=a, current_calibration=b, ude_measured=meas, note=note))
    add("chi* (Ch2 steady state)", jA["ch1_chi_star"], jB["ch1_chi_star"], pB.loc["chi", "ude_median"],
        "measured chi gap: %.1f%% -> %.1f%%" % (pA.loc["chi", "rel_gap_pct"], pB.loc["chi", "rel_gap_pct"]))
    add("rho reference", pA.loc["rho", "ch1_calibrated"], pB.loc["rho", "ch1_calibrated"], pB.loc["rho", "ude_median"],
        "gap %+.1f%% -> %+.1f%%; UDE rho is total recovery, commensurable with rho_0"
        % (pA.loc["rho", "rel_gap_pct"], pB.loc["rho", "rel_gap_pct"]))
    add("H2 predicted dchi/dg", jA["ch1_dchi_dg"], jB["ch1_dchi_dg"], net[0],
        "calibration splits -4.628 conventional + -3.455 urban-mining term (no UDE analogue)")
    add("H2 net (learn_cp) / prediction", net[0] / jA["ch1_dchi_dg"], net[0] / jB["ch1_dchi_dg"], np.nan,
        "vs conventional term only: %.3f" % (net[0] / -4.628))
    add("H2 dilution channel / prediction", dA.loc[1, "ratio_to_ch1"], dB.loc[1, "ratio_to_ch1"],
        dB.loc[1, "dchi_dg_hl"], "anchor_v4 arm")
    add("H3 predicted slope", mA.ch1_slope_full, mB.ch1_slope_full, mB.slope_hl,
        "measured outside both; verdict unchanged")
    add("H11 long-run dln s/drho", sA.ch1_long_run, sB.ch1_long_run, sB.hl,
        "short-run 0.738 unchanged; verdict tested on short run")
    add("chi* at measured coefficients", lg.chi_star_at_measured_gamma0,
        lg.chi_star_at_measured_gamma003_offset, lg.chi_measured,
        "calibration column applies the rho offset 0.018; without it 0.333 (double count)")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "d3_calibration_deltas.csv"), index=False)


def main():
    os.makedirs(OUT_B, exist_ok=True)
    scratch = os.environ.get("D3_SCRATCH")     # default: the system temporary directory
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        scratch_copy(tmp)
        run(tmp, ["run_wp9.py", "--all"])
        with open(os.path.join(tmp, "levelgap.py"), "w") as f:
            f.write(LEVELGAP)
        run(tmp, ["levelgap.py"])
        for f in glob.glob(os.path.join(tmp, "analysis", "wp9_*")) + \
                 [os.path.join(tmp, "analysis", "d3_chi_level_gap_variants.csv")]:
            if os.path.isfile(f):
                shutil.copy(f, OUT_B)
    deltas()
    # verdicts must be identical under the two cases
    a = pd.read_csv(os.path.join(FM, "analysis", "wp9_hypothesis_table.csv"))
    b = pd.read_csv(os.path.join(OUT_B, "wp9_hypothesis_table.csv"))
    same = (a.set_index("id")["outcome"] == b.set_index("id")["outcome"]).all()
    print("verdicts identical under both cases:", bool(same))
    print(pd.read_csv(os.path.join(OUT, "d3_calibration_deltas.csv")).to_string(index=False))


if __name__ == "__main__":
    main()
