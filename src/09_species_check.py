"""
Step 9 - Is the SPECIES signal physical, or a reporting artefact?

SPECIES dominates the random forest (permutation drop in PR-AUC 0.197,
step 4). A species field could matter for a non-physical reason: damaging
strikes are inspected more closely, so their species is more often
identified. Two single-input checks separate the two explanations:

  species_rate  : score = the training-years damage rate of the species,
                  smoothed towards the overall training rate (m-estimate,
                  m = 20) so that rare species are not given extreme rates.
                  If species carries physical risk, this alone should rank
                  damaging strikes well.
  identified    : score = 1 if the species was identified, 0 if it was
                  recorded as "Unknown ..." . If the signal were only about
                  report completeness, this flag would score almost as well.

Both scores use training years (2016-23, verified labels) only and are
evaluated on the test years (2025-26) by PR-AUC (no-skill = damage rate).
Also listed: training damage rates of the three species quoted in the report.

Input : data/processed/analytic.csv.gz (built by 01_prepare.py)
Output: outputs/tables/09_species_check.csv

Run from the repository root:  python src/09_species_check.py   (~10 s)
"""
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import config as C
import pipeline as P

M_SMOOTH = 20
QUOTED = ["White-tailed deer", "Turkey vulture", "Canada goose"]

df = P.load_data()
df = df[df["LABEL_VERIFIED"] == 1]
train = df[df["SPLIT"] == "train"]
test = df[df["SPLIT"] == "test"]
y_tr = train[C.TARGET].astype(int)
y_te = test[C.TARGET].astype(int)

# m-estimate of each species' damage rate, learned on training years only
overall = y_tr.mean()
stats = y_tr.groupby(train["SPECIES"]).agg(["mean", "size"])
smoothed = (stats["mean"] * stats["size"] + overall * M_SMOOTH) / (stats["size"] + M_SMOOTH)
score_rate = test["SPECIES"].map(smoothed).fillna(overall)

# identified (1) vs "Unknown ..." (0)
score_ident = (~test["SPECIES"].astype(str).str.startswith("Unknown")).astype(int)

rows = []
for name, s in [("species_rate", score_rate), ("identified_flag", score_ident)]:
    rows.append({"check": name,
                 "test_pr_auc": round(average_precision_score(y_te, s), 3),
                 "test_roc_auc": round(roc_auc_score(y_te, s), 3),
                 "no_skill_pr_auc": round(y_te.mean(), 3)})
for sp in QUOTED:
    if sp in stats.index:
        rows.append({"check": f"train_damage_rate: {sp}",
                     "test_pr_auc": None, "test_roc_auc": None,
                     "no_skill_pr_auc": None,
                     "train_rate": round(stats.loc[sp, "mean"], 3),
                     "train_n": int(stats.loc[sp, "size"])})

out = pd.DataFrame(rows)
C.TABLE_DIR.mkdir(parents=True, exist_ok=True)
out.to_csv(C.TABLE_DIR / "09_species_check.csv", index=False)
print(out.to_string(index=False))
