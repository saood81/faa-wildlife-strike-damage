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
