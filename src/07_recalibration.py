"""
Training damage was 7.4% versus 5.5% in testing, leading to over-predicted
probabilities.
Fit all corrections on 2024 only, then apply unchanged to 2025-26; do not select methods
on test data.
prior_shift: pre-specified primary; adjust log-odds for the changed damage rate (Saerens
et al., 2002).
platt: fit logistic regression of the label on model log-odds.
isotonic: fit a monotone, piecewise-constant probability mapping.
Input : outputs/predictions/predictions_A.csv.gz, predictions_HGB.csv.gz
Output: outputs/tables/07_recalibration.csv, 07_recalibration_by_year.csv
Run from the repository root: python src/07_recalibration.py (~1 min)
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import config as C
import pipeline as P

N_BOOT = 1000
EPS = 1e-6
PRED = C.ROOT / "outputs" / "predictions"

pred = pd.read_csv(PRED / "predictions_A.csv.gz")
hgb = PRED / "predictions_HGB.csv.gz"
if hgb.exists():
    pred["HGB"] = pd.read_csv(hgb)["HGB"].to_numpy()
MODELS = [m for m in ["LR", "GNB", "KNN", "SVM", "DT", "RF", "HGB"] if m in pred]

valid = pred[pred["split"] == "valid"].reset_index(drop=True)
test = pred[pred["split"] == "test"].reset_index(drop=True)
y_va, y = valid["y"].to_numpy(), test["y"].to_numpy()

df = P.load_data()
pi_train = df[(df["SPLIT"] == "train") & (df["LABEL_VERIFIED"] == 1)][C.TARGET].mean()
pi_valid = y_va.mean()
test_year = df[(df["SPLIT"] == "test") & (df["LABEL_VERIFIED"] == 1)]["INCIDENT_YEAR"].to_numpy()
assert len(test_year) == len(y)


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expit(z):
    return 1 / (1 + np.exp(-z))


def ece(y_true, prob, bins=10):
    edges = np.unique(np.quantile(prob, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, prob, side="right") - 1, 0, len(edges) - 2)
    return sum(abs(prob[idx == b].mean() - y_true[idx == b].mean()) * (idx == b).mean()
               for b in range(len(edges) - 1) if (idx == b).any())


def recalibrators(p_va):
    shift = logit(np.array([pi_valid]))[0] - logit(np.array([pi_train]))[0]
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p_va).reshape(-1, 1), y_va)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_va, y_va)
    return {"none": lambda p: p,
            "prior_shift": lambda p: expit(logit(p) + shift),
            "platt": lambda p: platt.predict_proba(logit(p).reshape(-1, 1))[:, 1],
            "isotonic": lambda p: iso.predict(p)}


rng = np.random.default_rng(C.RANDOM_STATE)
boot_idx = [rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)]

rows, by_year = [], []
for m in MODELS:
    fns = recalibrators(valid[m].to_numpy())
    p_raw = test[m].to_numpy()
    raw_boot = np.array([brier_score_loss(y[b], p_raw[b]) for b in boot_idx])
    for name, f in fns.items():
        p = f(p_raw)
        r = {"model": m, "method": name,
             "brier": round(brier_score_loss(y, p), 4),
             "mean_pred": round(p.mean(), 4), "observed": round(y.mean(), 4),
             "ece": round(ece(y, p), 4),
             "pr_auc": round(average_precision_score(y, p), 4),
             "roc_auc": round(roc_auc_score(y, p), 4)}
        if name != "none":
            d = np.array([brier_score_loss(y[b], p[b]) for b in boot_idx]) - raw_boot
            r["brier_change"] = round(brier_score_loss(y, p) - brier_score_loss(y, p_raw), 5)
            r["brier_change_ci"] = "[{:.5f}, {:.5f}]".format(*np.percentile(d, [2.5, 97.5]))
        rows.append(r)
        if m == "RF":
            for yr in (2025, 2026):
                k = test_year == yr
                by_year.append({"model": m, "method": name, "year": yr,
                                "mean_pred": round(p[k].mean(), 4), "observed": round(y[k].mean(), 4)})

out = pd.DataFrame(rows)
out.to_csv(C.TABLE_DIR / "07_recalibration.csv", index=False)
pd.DataFrame(by_year).to_csv(C.TABLE_DIR / "07_recalibration_by_year.csv", index=False)
print(f"Training damage rate {pi_train:.4f}; 2024 rate {pi_valid:.4f}; test rate {y.mean():.4f}")
print(f"No-skill Brier on test (always predict the test rate): {y.mean() * (1 - y.mean()):.4f}")
print(out.to_string(index=False))
print(pd.DataFrame(by_year).to_string(index=False))
