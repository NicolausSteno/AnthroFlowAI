#!/usr/bin/env bash
# Rebuild the computations and figures of the thesis Discussion (Chapter 6).
# Run from final_model/:   bash discussion/run_discussion.sh
# Needs Macro-Materials and Simulating-Resources cloned next to AnthroFlowAI (or the variables
# MACRO_MATERIALS_DIR / SIMULATING_RESOURCES_DIR), and R with the Simulating-Resources renv library.
# The zinc experiments (D2, D3, D5b, D14) also need the observed record: set WITH_ZINC_RECORD=1
# after placing zinc_dataset.xlsx in final_model/, restoring the masked files
# (reproducibility/restore_observed.py --xlsx zinc_dataset.xlsx) and building the WP-9 caches
# (python run_wp9.py --all).  Without it their published outputs are used for the figures.
set -euo pipefail
PY=${PYTHON:-python}; RS=${RSCRIPT:-Rscript}; O=discussion/outputs
mkdir -p "$O"
# copper and macro models only
$PY discussion/d12_copper_ratios.py    > "$O/d12_stdout.txt"
$PY discussion/d5_timescale_ladder.py  > "$O/d5_stdout.txt"
$PY discussion/chi_growth_check.py     > "$O/chi_growth_check.txt"
$PY discussion/ces_bottleneck.py       > "$O/ces_bottleneck.txt"
$PY discussion/ces_bottleneck_path.py  > "$O/ces_bottleneck_path.txt"
$RS discussion/d16_cycles.R            > "$O/d16_stdout.txt"
$RS discussion/d16_summarise.R
# zinc experiments (observed record required)
if [ "${WITH_ZINC_RECORD:-0}" = 1 ]; then
  $PY discussion/d3_wp9_calibration.py > "$O/d3_stdout.txt"
  $PY discussion/d2_stability.py       > "$O/d2_stdout.txt"
  $PY discussion/d14_growth_path.py    > "$O/d14_stdout.txt"
  $PY discussion/d5b_filling_checks.py > "$O/d5b_output.txt"
fi
# the three figures, written to PhD_Thesis/figures/discussion/
$PY discussion/figures/make_thdisc_scorecard.py
$PY discussion/figures/make_thdisc_timescale_ladder.py
$PY discussion/figures/make_thdisc_price_cycles.py
