# Design decisions log

Each entry: what was decided, why, and the evidence. Numbers come from
`outputs/tables/01_*.csv` unless stated.

## Population and labels
| # | Decision | Reason / evidence |
|---|---|---|
| D1 | Keep only strikes with known aircraft (`AC_MASS` present): 185,290 -> 126,180 rows | Unknown-aircraft records are mostly carcasses found with no aircraft identified; damage rate ~0.1%. They are not the population the model serves. |
| D2 | Target = `INDICATED_DAMAGE`; `M?` (damaged, extent unknown) counts as damaged | FAA definition of `M?` states the aircraft *was* damaged. |
| D3 | Remove 9 records where `INDICATED_DAMAGE` contradicts `DAMAGE_LEVEL` | Label errors; too few to matter, safer to remove than guess. |
| D4 | Main models use verified labels only (`DAMAGE_LEVEL` recorded): 108,935 rows, 7,431 damaged (6.8%) | A blank `DAMAGE_LEVEL` means no assessment was recorded, so its 0 is unverified. 17,236 such rows are kept, flagged, and re-added only in experiment E2. |

## Leakage
| # | Decision | Reason |
|---|---|---|
| L1 | Drop all `DAM_*`, `COST_*`, `AOS`, `EFFECT*`, injuries/fatalities, free text, report-process fields | Recorded after the event; they describe the outcome. |
| L2 | Drop `REMAINS_COLLECTED`, `REMAINS_SENT`, `LOCATION` | Post-event carcass handling; `LOCATION` often reads "FOUND AT <airport>". |
| L3 | Automatic screen: flag any single feature with AUC > 0.90 | None flagged. Highest: SPECIES 0.83 (in-sample category rate, optimistic by construction), SIZE 0.75. |

## Features
| # | Decision | Reason |
|---|---|---|
| F1 | Set A = pre-strike features; Set B = A + 14 `STR_*` strike-location flags + `ANY_INGESTION` | B is only valid for post-flight inspection triage, because strike location is known only after inspection. |
| F2 | `ANY_INGESTION` = max of `ING_ENG1-4` and `INGESTED_OTHER`; set B only | `INGESTED_OTHER` changed meaning on 29 Mar 2021 (4.8% of records before vs 0.1% after). The combined flag is consistent over time. |
| F3 | Month and hour encoded as sine/cosine pairs | They are circular: December is next to January. |
| F4 | `SIZE`, `NUM_STRUCK`, `NUM_SEEN` as ordered numbers 1..n | Preserves order; one column each. |
| F5 | `SPECIES`, `AIRPORT_ID`, `AIRCRAFT`, `OPID` (hundreds of categories) -> target encoding, fitted inside cross-validation | One-hot would create ~3,500 columns and break KNN/NB; cross-fitted encoding prevents the label leaking through the encoding. |
| F6 | Drop `STATE`, `RUNWAY`, `AMA/AMO/EMA/EMO`, `ENG_*_POS`, text duplicates | Redundant with kept columns or too fine-grained. |
| F7 | `INCIDENT_YEAR` is not a feature | With a temporal split, the test years are never seen in training. |
| F8 | `WARNED = "Unknown"` and `PHASE = "Unknown"` treated as missing | They are missing values written as text. |

## Split
| # | Decision | Reason |
|---|---|---|
| S1 | Train 2016-2023 (71,509; 7.4% damaged), validate 2024 (13,503; 6.0%), test 2025-26 (23,923; 5.5%) | Mimics deployment: train on the past, predict the future. Real prevalence is kept (no resampling of the test set). |
| S2 | Prevalence falls over time (8.4% in 2017 -> 5.3% in 2026) | A drift that a random split would hide; calibration on the test years must be checked. |

## Informative missingness (training set, verified labels)
Damage is 1.4-1.8x more common when SPEED, NUM_SEEN, HEIGHT, PHASE, SIZE, WARNED
or SKY are filled in. Hypothesis (not proven): damaging strikes trigger fuller
reports. This motivates experiment E1 (missingness indicators vs plain imputation).

## Modelling (03_models.py)
| # | Decision | Reason |
|---|---|---|
| M1 | Every model is one Pipeline (imputation, scaling, encoding, model) | Preprocessing is re-fitted inside each CV fold, so no information leaks from held-out folds. |
| M2 | Tuning: grid search, stratified 5-fold CV on 2016-23, scored by PR-AUC | PR-AUC focuses on the rare damaged class; accuracy would reward predicting "no damage" for everything (94.5% accuracy, zero use). |
| M3 | No resampling (no SMOTE/undersampling); `class_weight` is a tuned option instead | Resampling distorts probabilities; keeping the real class mix keeps them interpretable. The grid lets each model choose. |
| M4 | Decision threshold = the cut-off maximising F1 on 2024 | Chosen on data the model never trained on and that is not the test set. |
| M5 | SVM (RBF) trained on a stratified 15,000-row subsample; probabilities via Platt scaling (3-fold) | SVC cost grows with rows squared; 71,509 rows is impractical on a laptop. Stated as a limitation. |
| M6 | Feature set A only for the main comparison; set B is experiment E4 with the same tuned settings | Holding settings fixed isolates the effect of the extra features. |

## Evaluation (04_evaluate.py)
| # | Decision | Reason |
|---|---|---|
| V1 | Test set 2025-26 used once, at real prevalence (5.5%) | Honest estimate of future performance. |
| V2 | 1,000 bootstrap resamples, same resamples for every model | Gives 95% CIs and paired CIs for differences between models. |
| V3 | Permutation importance on the test set, scored by PR-AUC, 5 repeats | Model-agnostic; measures reliance on each original input column. |

## Findings to discuss
- RF best on test: PR-AUC 0.413 (95% CI 0.381-0.442), about 7.5x the no-skill level (0.055). Its lead over every other model is outside chance (paired CIs exclude 0).
- CV PR-AUC (2016-23) is higher than 2024 and 2025-26 for every model. Part of this is mechanical: PR-AUC's floor equals the damage rate, which fell from 7.4% to 5.5%. Part is drift.
- Models over-predict on 2025-26 (RF mean probability 0.064 vs actual 0.055), as expected from the falling damage rate.
- SPECIES dominates the RF (PR-AUC drop 0.20). Checked: the signal is physical, not a reporting artefact. White-tailed deer 85% damaging, turkey vulture 57%, Canada goose 53% (train years), while a simple "species unknown" flag alone scores PR-AUC 0.065, barely above no-skill.
- GNB is poorly calibrated (Brier 0.086, threshold 0.99): its independence assumption is violated by correlated inputs (mass, engine type, aircraft), which pushes probabilities to 0 or 1.
