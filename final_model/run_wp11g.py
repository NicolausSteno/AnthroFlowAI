#!/usr/bin/env python3
"""
run_wp11g.py — WP-11g: translation of the WP-11c ranking into observables
========================================================================

WP-11g is a writing task, and the spec is explicit about what makes it a
*grounded* one: the mapping "must be grounded in the 11c output rather than
written from general plausibility".  This driver therefore does not invent the
priority order.  It reads `analysis/wp11c_voi_headline.csv` and
`analysis/wp11c_frequency_decomposition.csv`, joins each coefficient's own
best-ranked model series to the physical observables that would measure it and
to named reporting streams that already carry them, and emits the table in
priority order with 11c's numbers attached to every row.

Three things from WP-11c decide the shape of the recommendation, and the table
carries all three rather than smoothing them:

  * **Every coefficient has a different best series, and each one is its own
    parent flow or parent stock.**  There is no general-purpose measurement to
    recommend, so the output is a per-coefficient list, not a wish list.
  * **The single highest-value measurement is annual, not sub-annual** —
    end-of-life product composition, 61-65% marginal-SD reduction at annual
    frequency on all three in-use cohort shares.
  * **About 5% of the apparent value of monthly reporting is resolution**;
    the rest is having twelve independent observations instead of one.  So a
    stream that reports annually but *completely* can be worth more than one
    that reports monthly but partially, and the table separates the two.

Sourcing caveat
---------------
The reporting streams named in `STREAMS` below are entered from domain
knowledge and were **not** verified against their publishers in this
environment (no data access).  Every row carries `verify_before_citing`, and
the coverage/frequency fields should be confirmed against the current
publications before the paragraph goes into the paper.

    python run_wp11g.py --check
    python run_wp11g.py --table
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "analysis")

HEADLINE = os.path.join(OUT_DIR, "wp11c_voi_headline.csv")
FREQ_DECOMP = os.path.join(OUT_DIR, "wp11c_frequency_decomposition.csv")
CANDIDATES = os.path.join(OUT_DIR, "wp11c_candidates.csv")

# Chapter 2 labels, so the thesis cross-reference is mechanical.
CH2_LABEL = {
    "alpha_cc": "alpha_1", "alpha_refc": "alpha_2 / alpha_8",
    "alpha_dr": "alpha_13", "alpha_win": "alpha_14",
    "tau_olds": "alpha_9 share", "f_cohort_10yr": "—",
    "f_cohort_20yr": "—", "f_cohort_44yr": "1/kappa_3 share",
}

# ---------------------------------------------------------------------------
# The translation.  One entry per *model series* that WP-11c ranks first or
# second for some coefficient; nothing here is written for a series the
# ranking does not reach.
# ---------------------------------------------------------------------------
STREAMS = {
    "waelz_input": dict(
        physical_observable="Zinc-bearing electric-arc-furnace dust delivered "
                            "to Waelz kilns, as received tonnage and assay",
        instrumented_as="Weighbridge tickets and X-ray-fluorescence assay at "
                        "the kiln gate; kiln feed telemetry",
        existing_streams=[
            "worldsteel monthly crude steel production, EAF route share "
            "(dust arisings scale with EAF throughput)",
            "Recycler production reports for steel-dust treatment volumes "
            "(e.g. Befesa, quarterly)",
            "EU Waste Shipment Regulation Annex VII / Basel notifications for "
            "transboundary EAF-dust movements",
            "E-PRTR facility-level hazardous-waste transfers, annual",
        ],
        best_available_frequency="monthly (steel proxy) / quarterly (operator "
                                 "reports) / annual (regulatory)",
        gap="No public series reports Waelz kiln *input* tonnage directly; "
            "the steel proxy needs a dust-arising factor and a zinc assay, "
            "both of which drift with scrap composition."),
    "direct_reuse_recycling": dict(
        physical_observable="Zinc-containing scrap re-melted without passing "
                            "through a refinery — brass and die-cast returns, "
                            "galvanisers' drosses and ashes",
        instrumented_as="Weighbridge and assay records at brass mills, "
                        "die-casters and galvanising lines; internal returns "
                        "logged as new-scrap credits",
        existing_streams=[
            "Customs trade in zinc waste and scrap, HS 7902, monthly "
            "(UN Comtrade, Eurostat Comext, US Census, China Customs)",
            "Trade-association scrap-flow surveys (BIR, EuRIC, ISRI)",
            "Producer Price Index for non-ferrous scrap (already a model "
            "driver, so it constrains the price response and not the tonnage)",
        ],
        best_available_frequency="monthly (customs) / annual (surveys)",
        gap="Customs data are cross-border only and miss domestic re-use, "
            "which is most of the flow; the survey data that would cover it "
            "are annual and partial."),
    "refined_stock": dict(
        physical_observable="Refined zinc metal held as inventory — exchange "
                            "warehouses, producer, consumer and merchant stocks",
        instrumented_as="Warehouse warrant systems; already fully "
                        "instrumented and reported daily for the exchange "
                        "share",
        existing_streams=[
            "LME zinc warehouse stocks, daily, plus monthly off-warrant "
            "stock reports",
            "SHFE zinc inventory, weekly",
            "ILZSG monthly reported stocks (producer / consumer / merchant)",
        ],
        best_available_frequency="daily to monthly",
        gap="Exchange stocks are a small and volatile subset of total refined "
            "stock; the reconciliation between reported inventories and the "
            "MFA's `Refined Stock` is the missing step, not the raw data."),
    "primary_refining": dict(
        physical_observable="Refined zinc produced from concentrate, and the "
                            "concentrate drawn to produce it",
        instrumented_as="Smelter throughput telemetry; concentrate receipts "
                        "and assays at the smelter gate",
        existing_streams=[
            "ILZSG monthly refined zinc metal production and mine production",
            "Company quarterly production reports (Korea Zinc, Nyrstar, "
            "Hindustan Zinc, Glencore)",
            "Treatment-charge benchmarks and spot TCs (already a driver)",
        ],
        best_available_frequency="monthly",
        gap="None of substance: this is the best-served quantity in the "
            "system, which is why WP-11c ranks its marginal value lowest in "
            "absolute terms even though it is `alpha_cc`'s best series."),
    "concentrate_consumption": dict(
        physical_observable="Concentrate drawn into smelting",
        instrumented_as="Smelter gate receipts and assay",
        existing_streams=["ILZSG monthly concentrate/mine production and "
                          "smelter intake"],
        best_available_frequency="monthly",
        gap="Not separable from `primary_refining` by this model "
            "(`tau_ref` is pinned, so the two carry the same parameter "
            "Jacobian — WP-11c flag 4).  Recommend one, not both."),
    "scrap_collection": dict(
        physical_observable="Old scrap separated from end-of-life products "
                            "and delivered to processors",
        instrumented_as="Weighbridge records at scrap processors and "
                        "shredders, with zinc assay on the non-ferrous "
                        "fraction",
        existing_streams=[
            "EU WEEE Directive Art. 16 collection tonnages, annual per "
            "member state and per category",
            "EU End-of-Life Vehicles Directive reporting and national vehicle "
            "deregistration registers, annual",
            "Eurostat waste statistics (WStatR) and construction & demolition "
            "waste reporting, biennial",
            "Shredder-residue and metal-recovery reporting at facility level",
        ],
        best_available_frequency="annual (regulatory) / continuous but "
                                 "unpublished (weighbridge)",
        gap="The weighbridge data exist and are continuous; they are simply "
            "not aggregated or published.  This is the cheapest genuine "
            "improvement in the table."),
    "eol_composition": dict(
        physical_observable="How end-of-life zinc splits across short-, "
                            "medium- and long-lived product cohorts",
        instrumented_as="Deregistration and demolition records tagged by "
                        "product age; WEEE category tonnages; galvanised "
                        "steel end-use surveys",
        existing_streams=[
            "National vehicle registers, deregistrations by vehicle age "
            "(DVLA, KBA and equivalents), annual",
            "EU WEEE Directive collection by category, annual",
            "Building demolition permits and building-stock age profiles",
            "IZA / galvanisers' association end-use statistics",
        ],
        best_available_frequency="annual",
        gap="No compiled global series exists.  Constructing one from the "
            "national registers is a compilation problem, not a measurement "
            "problem, and WP-11c makes it the highest-value single item in "
            "the whole table."),
    "end_of_life": dict(
        physical_observable="Aggregate end-of-life zinc arisings",
        instrumented_as="As `eol_composition`, without the cohort split",
        existing_streams=["The same registers, aggregated"],
        best_available_frequency="annual",
        gap="Ranks second to `eol_composition` on every cohort share: the "
            "value is in the split, not in the total."),
    "refined_consumption": dict(
        physical_observable="Refined zinc drawn into first-use manufacturing",
        instrumented_as="Galvanising line and die-caster metal receipts",
        existing_streams=["ILZSG monthly refined zinc usage"],
        best_available_frequency="monthly",
        gap="Ranks first on D-optimality and nobody's best on the marginal "
            "SD of a transfer coefficient — WP-11c's warning about the "
            "criterion, made concrete."),
    "scrap_stock": dict(
        physical_observable="Zinc held in collected but unprocessed scrap",
        instrumented_as="Yard inventory at processors",
        existing_streams=["No public series"],
        best_available_frequency="none",
        gap="The 2019 value in the reconstruction is a round-number anchor "
            "(500 kt), which WP-3 and WP-4c both flag; an actual measurement "
            "would replace an assumption rather than tighten an estimate."),
    "inuse_stock": dict(
        physical_observable="Zinc in products in service",
        instrumented_as="Bottom-up product-stock surveys",
        existing_streams=["National material-stock accounts, irregular"],
        best_available_frequency="irregular",
        gap="Ranks a distant second for `tau_olds` (1.7% against 18.3%)."),
}


def build_table(out_dir=OUT_DIR):
    """Join the 11c ranking to the translation, in 11c's own priority order."""
    for p in (HEADLINE, FREQ_DECOMP, CANDIDATES):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} is missing — WP-11g is grounded in WP-11c's output and "
                f"cannot be written without it")
    head = pd.read_csv(HEADLINE)
    fd = pd.read_csv(FREQ_DECOMP)
    cand = pd.read_csv(CANDIDATES).set_index("candidate")

    res = (fd[fd.freq == "monthly"]
           .groupby(["candidate", "coef"]).resolution_share.median())

    rows = []
    for coef, sub in head.groupby("coef", sort=False):
        ann = sub[sub.freq == "annual"].iloc[0]
        mon = sub[sub.freq == "monthly"].iloc[0]
        for rank, series in ((1, ann.best_series), (2, ann.runner_up)):
            if not isinstance(series, str) or series not in STREAMS:
                continue
            tr = STREAMS[series]
            c = cand.loc[series] if series in cand.index else None
            rows.append(dict(
                coefficient=coef,
                ch2_label=CH2_LABEL.get(coef, ""),
                rank=rank,
                model_series=series,
                operator=(str(c.operator) if c is not None else ""),
                sigma_rel=(float(c.sigma_rel) if c is not None
                           and np.isfinite(c.sigma_rel) else np.nan),
                rel_sd_now_pct=float(ann.rel_sd_base_pct),
                reduction_annual_pct=(float(ann.reduction_pct) if rank == 1
                                      else float(ann.runner_up_pct)),
                reduction_monthly_pct=(float(mon.reduction_pct) if rank == 1
                                       else float(mon.runner_up_pct)),
                resolution_share_of_monthly=float(
                    res.get((series, coef), np.nan)),
                physical_observable=tr["physical_observable"],
                instrumented_as=tr["instrumented_as"],
                existing_streams=" | ".join(tr["existing_streams"]),
                best_available_frequency=tr["best_available_frequency"],
                gap=tr["gap"],
                verify_before_citing=True))
    df = pd.DataFrame(rows)
    # priority: the reduction the *annual* arm buys, because 11c says
    # resolution is a small part of the monthly figure
    df["annual_value_rank"] = df.reduction_annual_pct.rank(ascending=False,
                                                           method="min").astype(int)
    df = df.sort_values(["annual_value_rank", "coefficient", "rank"])
    df.to_csv(os.path.join(out_dir, "wp11g_observables.csv"), index=False)
    return df


def write_note(df, out_dir=OUT_DIR):
    """The prose paragraph, generated from the table so the numbers cannot
    drift away from `wp11g_observables.csv`."""
    top = df[df["rank"] == 1].sort_values("reduction_annual_pct", ascending=False)
    lines = []
    for _, r in top.iterrows():
        first = r.existing_streams.split(" | ")[0]
        lines.append(
            f"- **{r.coefficient}** ({r.ch2_label}) — best measured by "
            f"`{r.model_series}`, physically *{r.physical_observable[0].lower()}"
            f"{r.physical_observable[1:]}*.  Annual reporting cuts its marginal "
            f"SD by {r.reduction_annual_pct:.0f}%, monthly by "
            f"{r.reduction_monthly_pct:.0f}%, of which "
            f"{100 * r.resolution_share_of_monthly:.0f}% is resolution rather "
            f"than extra observations.  Closest existing stream: {first}.  "
            f"Gap: {r.gap}")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args(argv)

    if a.check:
        # Project convention: every script prints the resolved driver names
        # and `input_dim` before doing work, even one that only joins CSVs.
        import zinc_colloc_v5 as v5
        import zinc_synth_lab as S
        S.install(v5)
        cfg = S.load_anchor_config()
        ctx = S.driver_context(cfg)
        print("  drivers resolved by name (never by position):")
        for short, name in S.DRIVER_ALIASES.items():
            print(f"    {short:9s} -> {name!r} "
                  f"[col {list(ctx['exog_cols']).index(name)}]")
        n_ex = len(ctx["exog_cols"])
        orders = len(cfg.get("exog_feature_orders", (0, 1)))
        print(f"  exog columns = {n_ex}, feature orders = {orders}, "
              f"input_dim = {1 + 4 + n_ex * orders}  "
              f"(CLAUDE.md rule 2 says 23 — SCHEMA §9 flag 1, open)")
        for q in (HEADLINE, FREQ_DECOMP, CANDIDATES):
            print(f"  {'ok      ' if os.path.exists(q) else 'MISSING '} {q}")
        print(f"  translations written for {len(STREAMS)} model series: "
              f"{sorted(STREAMS)}")
        return 0

    df = build_table()
    cols = ["annual_value_rank", "coefficient", "rank", "model_series",
            "reduction_annual_pct", "reduction_monthly_pct",
            "resolution_share_of_monthly", "best_available_frequency"]
    print(df[cols].to_string(index=False))
    print()
    for line in write_note(df):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
