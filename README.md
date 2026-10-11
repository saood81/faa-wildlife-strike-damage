# Predicting Aircraft Damage from Wildlife Strikes

EAM04DS Machine Learning project, Emirates Aviation University (Fall 2026).
Saood Al Ali, 10493808.

Binary classification of `INDICATED_DAMAGE` in the FAA National Wildlife Strike
Database (2016 to mid-2026 extract), with an explicit leakage audit, label-noise
and informative-missingness handling, and a prevalence-preserving temporal
evaluation (train 2016-2023, validate 2024, test 2025-2026).

## Data
`data/raw/faa_2016_2026.csv.gz` is the full FAA extract downloaded from
https://wildlife.faa.gov (U.S. Government work, public domain). Reporter names
are already redacted by the FAA.

## Reproduce
```bash
python -m pip install -r requirements.txt
python src/01_prepare.py      # cleaning, leakage audit, split          (~15 s)
python src/02_eda.py          # figures 1-3                             (~10 s)
python src/03_models.py       # tune + fit the six models               (~20 min on 2 cores)
python src/04_evaluate.py     # test metrics, CIs, figures 4-6          (~5 min)
```
Every random step uses the seed in `src/config.py` (`RANDOM_STATE = 42`).

## Repository layout
| Path | Contents |
|---|---|
| `src/config.py` | Paths, split years, leakage list, feature sets A and B |
| `src/01_prepare.py` | Cleaning, label handling, feature engineering, leakage screen, temporal split |
| `src/pipeline.py` | Shared preprocessing and the six model definitions with their tuning grids |
| `src/02_eda.py` | Exploratory figures (training years only, except the drift figure) |
| `src/03_models.py` | Grid search with 5-fold CV, refit, threshold chosen on 2024 |
| `src/04_evaluate.py` | Test-set metrics with bootstrap 95% CIs, PR curves, calibration, permutation importance |
| `outputs/predictions/` | Predicted damage probabilities for 2024-26 |
| `outputs/tables/` | Audit and results tables (CSV) |
| `outputs/figures/` | Figures used in the report |
| `docs/decisions.md` | Every design decision and its justification |
