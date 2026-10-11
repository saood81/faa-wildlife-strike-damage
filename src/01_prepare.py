"""
Step 1 - Data preparation.

Input : data/raw/faa_2016_2026.csv.gz   (FAA National Wildlife Strike Database extract)
Output: data/processed/analytic.csv.gz  (model-ready table, one row per strike)
        outputs/tables/01_*.csv          (data-flow, prevalence, missingness, leakage screen)

What this script does, in order:
  1. Load and tidy the raw extract (blank strings -> missing).
  2. Keep the decision population: strikes where the aircraft is known (AC_MASS present).
  3. Label handling: flag verified vs unverified labels; drop contradictory labels.
  4. Feature engineering (cyclic month/hour, ordinal codes, any-ingestion flag).
  5. Leakage removal, then an automatic single-feature leakage screen.
  6. Temporal split column (train 2016-23 / valid 2024 / test 2025-26).
  7. Save the analytic table and the audit tables.

Run from the repository root:  python src/01_prepare.py
"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config as C

C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
C.TABLE_DIR.mkdir(parents=True, exist_ok=True)

flow = []  # (step, rows, positives) -> how many records survive each step


def log_step(name, frame, label_col=C.TARGET):
    flow.append({"step": name, "rows": len(frame), "positives": int(frame[label_col].sum())})
    print(f"{name:60s} rows={len(frame):7,d}  positives={int(frame[label_col].sum()):6,d}")


# ------------------------------------------------------------ 1. load and tidy
df = pd.read_csv(C.RAW_FILE, low_memory=False)
for col in df.columns:
    if df[col].dtype == object or str(df[col].dtype).startswith("str"):
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)
log_step("1. Raw extract 2016-2026", df)

# ------------------------------------------------------------ 2. decision population
# Strikes with unknown aircraft are mostly carcasses found on runways with no
# known aircraft; their damage rate is ~0.1%, so they are not the population a
# damage model is meant for.
df = df[df["AC_MASS"].notna()].copy()
log_step("2. Aircraft known (AC_MASS present)", df)

# ------------------------------------------------------------ 3. labels
# DAMAGE_LEVEL blank = the FAA never recorded a damage assessment, so a 0 in
# INDICATED_DAMAGE is an *unverified* negative. These rows are kept but flagged;
# the main models use verified rows only (experiment E2 re-adds them).
df["LABEL_VERIFIED"] = df["DAMAGE_LEVEL"].notna().astype(int)

# Contradictions between the two damage fields (e.g. level 'N' but flag 1) are
# label errors. There are only a handful, so they are removed.
level_says_damage = df["DAMAGE_LEVEL"].isin(["M", "M?", "S", "D"])
conflict_verified = (df["LABEL_VERIFIED"] == 1) & (level_says_damage.astype(int) != df[C.TARGET])
conflict_unverified = (df["LABEL_VERIFIED"] == 0) & (df[C.TARGET] == 1)
n_conflict = int(conflict_verified.sum() + conflict_unverified.sum())
df = df[~(conflict_verified | conflict_unverified)].copy()
log_step(f"3. Contradictory labels removed (n={n_conflict})", df)
# 'M?' (damaged, extent unknown) counts as damaged - consistent with INDICATED_DAMAGE.

# ------------------------------------------------------------ 4. feature engineering
# Month and hour are circular (December is next to January), so each becomes
# a sine/cosine pair instead of a single number.
month = df["INCIDENT_MONTH"].astype(float)
df["MONTH_SIN"] = np.sin(2 * np.pi * month / 12)
df["MONTH_COS"] = np.cos(2 * np.pi * month / 12)

hhmm = df["TIME"].str.extract(r"^(\d{1,2}):(\d{2})$").astype(float)
hour = hhmm[0] + hhmm[1] / 60
hour[(hour < 0) | (hour >= 24)] = np.nan
df["HOUR_SIN"] = np.sin(2 * np.pi * hour / 24)
df["HOUR_COS"] = np.cos(2 * np.pi * hour / 24)

for col in ["LATITUDE", "LONGITUDE", "HEIGHT", "SPEED", "DISTANCE", "NUM_ENGS"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Ordered categories -> ordered numbers (keeps the order information).
df["SIZE_ORD"] = df["SIZE"].map({"Small": 1, "Medium": 2, "Large": 3})
count_map = {"1": 1, "2-10": 2, "11-100": 3, "More than 100": 4}
df["NUM_STRUCK_ORD"] = df["NUM_STRUCK"].map(count_map)
df["NUM_SEEN_ORD"] = df["NUM_SEEN"].map(count_map)

# 'Unknown' written as a value is really a missing value.
df["WARNED"] = df["WARNED"].replace("Unknown", pd.NA)
df["PHASE_OF_FLIGHT"] = df["PHASE_OF_FLIGHT"].replace("Unknown", pd.NA)

# Ingestion (feature set B only). INGESTED_OTHER changed meaning on 29 Mar 2021:
# before that date it held ALL ingestions, after it only non-engine ones.
# Combining all five flags into one 'any ingestion' flag is consistent across
# the whole period.
ing_cols = ["ING_ENG1", "ING_ENG2", "ING_ENG3", "ING_ENG4", "INGESTED_OTHER"]
df["ANY_INGESTION"] = df[ing_cols].max(axis=1)

# ------------------------------------------------------------ 5. leakage removal
df = df.drop(columns=[c for c in C.LEAKAGE_COLS + C.DROP_COLS if c in df.columns])

num, ordn, cat, high = C.feature_lists("B")
features = num + ordn + cat + high
leaked = set(features) & set(C.LEAKAGE_COLS)
assert not leaked, f"Leakage column in feature list: {leaked}"

# Automatic screen: any single feature that predicts damage almost perfectly on
# its own is suspicious (a post-event field slipped in). Computed on verified
# training rows only. AUC 0.5 = useless, 1.0 = perfect.
train_v = df[df["INCIDENT_YEAR"].isin(C.TRAIN_YEARS) & (df["LABEL_VERIFIED"] == 1)]
screen = []
for col in features:
    x = train_v[col]
    if col in cat + high:
        x = x.astype("string").fillna("MISSING")
        x = x.map(train_v.groupby(x)[C.TARGET].mean())   # in-sample category rate (screen only)
    else:
        x = x.astype(float).fillna(x.astype(float).median())
    auc = roc_auc_score(train_v[C.TARGET], x)
    screen.append({"feature": col, "single_feature_auc": round(max(auc, 1 - auc), 3),
                   "set": "B only" if col in C.EXTRA_B else "A"})
screen = pd.DataFrame(screen).sort_values("single_feature_auc", ascending=False)
screen["flag"] = np.where(screen["single_feature_auc"] > 0.90, "CHECK", "")
screen.to_csv(C.TABLE_DIR / "01_leakage_screen.csv", index=False)
if (screen["flag"] == "CHECK").any():
    print("WARNING - suspicious features:\n", screen[screen["flag"] == "CHECK"])

# ------------------------------------------------------------ 6. temporal split
year = df["INCIDENT_YEAR"]
df["SPLIT"] = np.select(
    [year.isin(C.TRAIN_YEARS), year.isin(C.VALID_YEARS), year.isin(C.TEST_YEARS)],
    ["train", "valid", "test"], default="other")
assert (df["SPLIT"] != "other").all()

verified = df[df["LABEL_VERIFIED"] == 1]
log_step("4. Verified-label analytic set (main experiments)", verified)
log_step("   + unverified negatives (used only in experiment E2)", df)

# ------------------------------------------------------------ 7. save
keep = ["INCIDENT_YEAR", "SPLIT", "LABEL_VERIFIED", C.TARGET] + features
df[keep].to_csv(C.ANALYTIC_FILE, index=False, compression="gzip")

pd.DataFrame(flow).to_csv(C.TABLE_DIR / "01_data_flow.csv", index=False)

prev = (df.groupby(["SPLIT", "LABEL_VERIFIED"])[C.TARGET]
          .agg(rows="size", positives="sum", prevalence="mean")
          .reset_index())
prev["prevalence"] = prev["prevalence"].round(4)
prev.to_csv(C.TABLE_DIR / "01_prevalence_by_split.csv", index=False)

by_year = (verified.groupby("INCIDENT_YEAR")[C.TARGET]
                   .agg(rows="size", positives="sum", prevalence="mean").round(4))
by_year.to_csv(C.TABLE_DIR / "01_prevalence_by_year.csv")

# Informative-missingness table: is damage more/less common when a field is blank?
miss_rows = []
for col in features:
    if col in C.EXTRA_B:
        continue
    m = train_v[col].isna()
    if m.sum() == 0:
        continue
    miss_rows.append({"feature": col,
                      "pct_missing": round(100 * m.mean(), 1),
                      "damage_rate_if_missing": round(train_v.loc[m, C.TARGET].mean(), 4),
                      "damage_rate_if_present": round(train_v.loc[~m, C.TARGET].mean(), 4)})
miss = pd.DataFrame(miss_rows).sort_values("pct_missing", ascending=False)
miss["ratio_present_to_missing"] = (miss["damage_rate_if_present"]
                                    / miss["damage_rate_if_missing"]).round(2)
miss.to_csv(C.TABLE_DIR / "01_missingness_vs_damage.csv", index=False)

print(f"\nSaved {C.ANALYTIC_FILE.relative_to(C.ROOT)}  shape={df[keep].shape}")
print("Prevalence by split (verified only):")
print(prev[prev["LABEL_VERIFIED"] == 1].to_string(index=False))
sys.exit(0)
