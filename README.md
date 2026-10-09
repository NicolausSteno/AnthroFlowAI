# AnthroFlowAI — Chapter 5 – universal differential equation model of the global zinc cycle

Code and results for the PhD thesis chapter on forecasting the global
anthropogenic zinc cycle with universal differential equations (Chapter 5; Appendix K),
together with the experiments taken up in the thesis discussion (Chapter 6).

The model is a mass-conserving, continuous-time system of ODEs whose transfer
coefficients are produced by a neural network from exogenous drivers
(JAX / diffrax / equinox).

## Data

The observed zinc record (stocks, flows and drivers, 1980-2019;
`zinc_dataset.xlsx`) is derived from International Lead and Zinc Study Group
(ILZSG) data and **is not included**: we do not have permission to redistribute it.

Some result files store values of that record: the observed stocks and flows
saved next to each prediction, coefficients pinned to observed transfer shares,
the observed concentrate input, driver paths, trajectories that start from
observed initial stocks, and a few columns computed directly from the observed
series. Those files are published in **masked** form under
`final_model/masked_results/` (same relative paths), with every such value
removed. `masked_results/restore_map.npz` records where each removed value came
from in the loader output (array and position, never the value), and
`masked_results/manifest.json` holds SHA-256 digests of the original arrays and
files. All other result files are published unchanged.

```
cd final_model
# with the dataset (place it at final_model/zinc_dataset.xlsx):
python reproducibility/restore_observed.py --xlsx zinc_dataset.xlsx
# without it (observed values stay missing):
python reproducibility/restore_observed.py --without-data
```

With the dataset, 68 of the 70 masked files are rebuilt bit for bit and checked
against the stored digests; the imbalance columns of `wp1f_mass_imbalance.csv`
are recomputed to within 1e-13 relative error, and `rhs_residual_obs` in
`wp2a_A_of_t.npz` (a diagnostic not used in the thesis) is regenerated only by
re-running `run_wp2a.py`. `reproducibility/mask_observed.py` is the script that
produced the masked copies.

## Layout

| Path | Contents |
|---|---|
| `final_model/zinc_colloc_v5.py` | core UDE model and estimator (Stage A collocation, Stage B trajectory fit) |
| `final_model/zinc_baseline.py` | ridge-regression comparator (same mass-conserving ODEs) |
| `final_model/zinc_*_lab.py` | experiment modules; each patches the core model at import time |
| `final_model/run_*.py` | one driver per analysis (table below) |
| `final_model/anchor_v4/`, `anchor_gam/` | configs and accuracy tables of the published runs |
| `final_model/analysis/` | result tables of each analysis |
| `final_model/masked_results/` | masked copies of the result files that contain observed values |
| `final_model/reproducibility/` | figure builder, masking and restoration scripts |
| `final_model/PhD_Thesis/figures/ch4/` | the ten figures as printed in the chapter |

| Driver | Analysis |
|---|---|
| `run_anchor.py` | 35-seed UDE ensemble and ridge-regression comparator |
| `run_wp1a.py` | per-channel coefficient accuracy |
| `run_wp1b.py` | coefficient/stock compensation test |
| `run_wp1c.py` | bias-variance decomposition of the coefficients |
| `run_wp1f.py` | autoregressive (AR/ARX) comparator and its mass imbalance |
| `run_wp2a.py` | time-varying transfer matrix A(t) |
| `run_wp2b.py` | eigenstructure of A(t) |
| `run_wp2c.py` | eigenvalue sensitivity |
| `run_wp2d.py` | frozen-time vs non-autonomous circularity indicators |
| `run_wp2e.py` | indicator sensitivities (fundamental matrix) |
| `run_wp2f.py` | structural stability / distance to bifurcation |
| `run_wp2g.py` | frozen equilibrium, saturation, transient amplification |
| `run_wp3.py` | the synthetic cycle |
| `run_wp4c.py` | Fisher information and practical identifiability |
| `run_wp6b.py` | temporal coarsening (thinning) of the record |
| `run_wp7.py` | price scenarios and counterfactuals |
| `run_wp8.py` | memory length (8b), driver permutation (8d), Sobol (8e), breaks (8f); gradient attribution (8a), forward sensitivities (8c) and capability table (8i) for Ch. 6 |
| `run_wp8h.py` | coefficient uncertainty product |
| `run_wp10.py` | Erlang use-phase arms, incl. learned mean lifetimes (arm expk) |
| `run_wp11a.py` | reporting frequency on the synthetic cycle |
| `run_wp11b.py` | synthetic-real agreement gate |
| `run_wp11c.py` | value of information / observation ranking |
| `run_wp11d.py` | learning curves and equal-sample comparison |
| `run_wp11f.py` | annual-harmonic (aliasing) bias |
| `run_wp11g.py` | translation of the ranking into observables |
| `run_wp10b.py` | Erlang follow-up: zinc's own order, free mean (Ch. 6: invariance test) |
| `run_wp9.py` | Chapter 2 hypothesis tests, circularity index (Ch. 6) |
| `run_wp9_cp.py` | WP-9 close-out with learned concentrate input (Ch. 6) |
| `run_wp4b.py` | misspecification absorption / state-blindness (Ch. 6) |
| `run_wp4a.py` | attenuation and aliasing of period-integral observations (Ch. 6) |
| `run_wp11e.py` | sub-annual dynamics with market relevance (Ch. 6) |
| `run_wp11h.py` | temporal aggregation bias, continuous vs discrete time (Ch. 6) |

## Environment

Python 3.12.7; `pip install -r requirements.txt` (jax 0.4.38, diffrax 0.7.1,
equinox 0.13.4, optax 0.2.5, numpy 2.4.1, pandas, scipy, matplotlib, openpyxl).

## Reproducing the figures

```
cd final_model
python reproducibility/restore_observed.py --xlsx zinc_dataset.xlsx   # or --without-data
python reproducibility/figures/make_ch4_figures.py --check
python reproducibility/figures/make_ch4_figures.py                    # all figures
```

After a restore with the dataset, every figure is identical to the printed one.
Without the dataset every figure still builds; Figures 1, 2 and 4 then lack the
observed points, and Figures 5-8 are unaffected.
