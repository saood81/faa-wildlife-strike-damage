"""
Step 3 - Tune, fit and score the six models (feature set A).

For every model:
  1. Hyper-parameter tuning: grid search with stratified 5-fold
     cross-validation on the TRAINING years (2016-23), scored by PR-AUC
     (average precision).
  2. The best settings are refitted on all training years.
  3. Decision threshold: chosen on the VALIDATION year (2024) as the cut-off
     that maximises F1. The test years are not used for any choice.
  4. Damage probabilities are saved for validation and test (2025-26) rows;
     04_evaluate.py turns them into metrics, curves and confidence intervals.

The SVM (RBF kernel) is tuned and trained on a stratified 15,000-row
subsample of the training years, because SVC training time grows with the
square of the row count. Its scores are turned into probabilities with Platt
(sigmoid) scaling, fitted by 3-fold cross-validation on the same subsample.

Input : data/processed/analytic.csv.gz
Output: data/processed/models/<model>_A.joblib       (fitted pipelines; not stored in git)
        outputs/tables/03_tuning_results.csv          (best settings + CV PR-AUC per model)
        outputs/predictions/predictions_A.csv.gz      (probabilities for 2024-26 rows)

Run from the repository root:  python src/03_models.py
"""
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

import config as C
import pipeline as P

warnings.filterwarnings("ignore", category=UserWarning)   # rare one-hot categories in some folds

MODEL_DIR = C.PROCESSED_DIR / "models"
PRED_DIR = C.ROOT / "outputs" / "predictions"
for d in (MODEL_DIR, PRED_DIR, C.TABLE_DIR):
    d.mkdir(parents=True, exist_ok=True)


def best_f1_threshold(y_true, prob):
    """Cut-off on the probability that gives the highest F1 score."""
    prec, rec, thr = precision_recall_curve(y_true, prob)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    i = int(np.nanargmax(f1[:-1]))           # last point has no threshold
    return float(thr[i])


df = P.load_data()
X_tr, y_tr = P.get_xy(df, "train")
X_va, y_va = P.get_xy(df, "valid")
X_te, y_te = P.get_xy(df, "test")
print(f"train {X_tr.shape}  valid {X_va.shape}  test {X_te.shape}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)
rows, preds = [], {"split": ["valid"] * len(y_va) + ["test"] * len(y_te),
                   "y": np.r_[y_va.values, y_te.values]}

for name, (estimator, grid) in P.model_specs().items():
    t0 = time.time()
    X_fit, y_fit = X_tr, y_tr
    if name == "SVM":
        X_fit, _, y_fit, _ = train_test_split(X_tr, y_tr, train_size=P.SVM_SUBSAMPLE,
                                              stratify=y_tr, random_state=C.RANDOM_STATE)
    # Models that already use both CPU cores internally run their CV folds one at a time.
    n_jobs = 1 if name in ("RF", "KNN") else -1
    search = GridSearchCV(P.make_pipeline(estimator), grid, scoring="average_precision",
                          cv=cv, n_jobs=n_jobs, refit=True)
    search.fit(X_fit, y_fit)
    model = search.best_estimator_
    if name == "SVM":   # SVC gives distances, not probabilities -> Platt scaling
        model = CalibratedClassifierCV(search.best_estimator_, method="sigmoid", cv=3)
        model.fit(X_fit, y_fit)

    p_va, p_te = P.scores(model, X_va), P.scores(model, X_te)
    thr = best_f1_threshold(y_va, p_va)
    joblib.dump({"model": model, "threshold": thr}, MODEL_DIR / f"{name}_A.joblib")
    preds[name] = np.r_[p_va, p_te]

    i = search.best_index_
    rows.append({
        "model": name,
        "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
        "cv_pr_auc_mean": round(search.cv_results_["mean_test_score"][i], 4),
        "cv_pr_auc_sd": round(search.cv_results_["std_test_score"][i], 4),
        "valid_pr_auc": round(average_precision_score(y_va, p_va), 4),
        "threshold_from_valid": round(thr, 4),
        "train_rows": len(y_fit),
        "grid_size": len(search.cv_results_["params"]),
        "minutes": round((time.time() - t0) / 60, 1),
    })
    print(f"{name:4s} CV PR-AUC {rows[-1]['cv_pr_auc_mean']:.3f} +/- {rows[-1]['cv_pr_auc_sd']:.3f}"
          f" | valid PR-AUC {rows[-1]['valid_pr_auc']:.3f} | threshold {thr:.3f}"
          f" | {rows[-1]['minutes']} min | {rows[-1]['best_params']}")

pd.DataFrame(rows).to_csv(C.TABLE_DIR / "03_tuning_results.csv", index=False)
pd.DataFrame(preds).to_csv(PRED_DIR / "predictions_A.csv.gz", index=False, compression="gzip")
print("Done. Next: python src/04_evaluate.py")
